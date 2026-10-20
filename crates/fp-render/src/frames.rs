//! The hardware decoder's pictures, copied out on the GPU (see
//! [`fp_media::gpu`]). Each CAPTURE buffer is imported once as a DMA-BUF;
//! each picture is copied into a GPU buffer of its own, so the decoder has
//! its buffer back as soon as the copy is done and the picture never passes
//! through the CPU. Copies run on the one queue in submission order, so
//! whatever reads a picture later sees it complete without waiting here.

use crate::gpu::Gpu;
use crate::mem::Buffer;
use crate::{Error, Result, VkContext};
use ash::vk;
use fp_core::sync::lock;
use fp_media::{DecoderBuffer, GpuBuffer, GpuBytes, GpuCopier};
use gpu_allocator::MemoryLocation;
use std::any::Any;
use std::os::fd::{AsRawFd, IntoRawFd, OwnedFd};
use std::sync::{Arc, Mutex};

/// Picture buffers kept for reuse once their frames are gone.
const SPARE: usize = 4;

/// Copies the hardware decoder's pictures out on the GPU.
#[derive(Clone)]
pub struct GpuFrames(Arc<Inner>);

struct Inner {
    gpu: Arc<Gpu>,
    fd: ash::khr::external_memory_fd::Device,
    /// The copies' command buffer and fence: one copy at a time.
    work: Mutex<Work>,
    spare: Mutex<Vec<Buffer>>,
}

struct Work {
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
}

impl std::fmt::Debug for GpuFrames {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GpuFrames")
    }
}

impl GpuFrames {
    /// The copier, when this GPU can import the decoder's buffers.
    pub fn new(gpu: &Arc<Gpu>) -> Result<Option<GpuFrames>> {
        if !gpu.dma_buf {
            return Ok(None);
        }
        let d = &gpu.device;
        // SAFETY: a pool, command buffer and fence owned by Inner, which
        // destroys them.
        let work = unsafe {
            let pool = d
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(gpu.queue_family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .ctx("picture copy pool")?;
            let cmd = d
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .command_buffer_count(1),
                )
                .ctx("picture copy commands")?[0];
            let fence = d
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .ctx("picture copy fence")?;
            Work { pool, cmd, fence }
        };
        Ok(Some(GpuFrames(Arc::new(Inner {
            gpu: gpu.clone(),
            fd: ash::khr::external_memory_fd::Device::new(&gpu.instance, &gpu.device),
            work: Mutex::new(work),
            spare: Mutex::new(Vec::new()),
        }))))
    }
}

impl GpuCopier for GpuFrames {
    fn import(
        &self,
        fd: OwnedFd,
        len: usize,
    ) -> std::result::Result<Box<dyn DecoderBuffer>, String> {
        Imported::new(&self.0, fd, len)
            .map(|b| Box::new(b) as Box<dyn DecoderBuffer>)
            .map_err(|e| e.to_string())
    }
}

impl Inner {
    /// Records `cmd` (from the pool `_pool` holds) with `record` and submits
    /// it, signalling `fence`. Both must be idle: their last submission done.
    fn record_submit(
        &self,
        _pool: &Work,
        cmd: vk::CommandBuffer,
        fence: vk::Fence,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<()> {
        let d = &self.gpu.device;
        // SAFETY: the pool is held, and the command buffer and fence are idle
        // (callers wait for the fence first).
        unsafe {
            d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .ctx("reset picture copy")?;
            d.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .ctx("begin picture copy")?;
            record(d, cmd);
            d.end_command_buffer(cmd).ctx("end picture copy")?;
            d.reset_fences(&[fence]).ctx("reset picture copy fence")?;
        }
        let cmds = [cmd];
        self.gpu
            .submit(&[vk::SubmitInfo::default().command_buffers(&cmds)], fence)
    }

    /// Runs commands recorded with `record`, waiting until done.
    fn run(&self, record: impl FnOnce(&ash::Device, vk::CommandBuffer)) -> Result<()> {
        let w = lock(&self.work);
        self.record_submit(&w, w.cmd, w.fence, record)?;
        // SAFETY: the fence of the submission just made.
        unsafe { self.gpu.device.wait_for_fences(&[w.fence], true, u64::MAX) }
            .ctx("wait for picture readback")
    }

    /// A buffer for a picture of `size` bytes: a spare one, or a new one.
    fn picture_buffer(&self, size: u64) -> Result<Buffer> {
        let mut spare = lock(&self.spare);
        if let Some(i) = spare.iter().position(|b| b.size == size) {
            return Ok(spare.swap_remove(i));
        }
        drop(spare);
        Buffer::new(
            &self.gpu,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            "decoded picture",
        )
    }

    /// Starts copying `ranges` (offset, length) of the decoder buffer `src`
    /// one after another into a new picture, with `cmd`, signalling `fence`.
    fn copy(
        self: &Arc<Self>,
        src: vk::Buffer,
        cmd: vk::CommandBuffer,
        fence: vk::Fence,
        ranges: &[(usize, usize)],
    ) -> Result<Picture> {
        let len: usize = ranges.iter().map(|r| r.1).sum();
        let buffer = self.picture_buffer(len as u64)?;
        let mut at = 0;
        let regions: Vec<_> = ranges
            .iter()
            .map(|&(offset, n)| {
                let r = vk::BufferCopy {
                    src_offset: offset as u64,
                    dst_offset: at,
                    size: n as u64,
                };
                at += n as u64;
                r
            })
            .collect();
        let family = self.gpu.queue_family;
        let dst = buffer.buffer;
        let pool = lock(&self.work);
        let ran = self.record_submit(&pool, cmd, fence, |d, cmd| {
            // Taken from the decoder, which wrote it outside Vulkan: the GPU
            // must not read stale cached lines of the picture before.
            let acquire = vk::BufferMemoryBarrier2::default()
                .src_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
                .dst_queue_family_index(family)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .buffer(src)
                .size(vk::WHOLE_SIZE);
            // Given back to the decoder, and the copy made visible to what
            // reads the picture later on this queue.
            let release = vk::BufferMemoryBarrier2::default()
                .src_queue_family_index(family)
                .dst_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .buffer(src)
                .size(vk::WHOLE_SIZE);
            let written = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ);
            // SAFETY: recording; both buffers are valid and the regions lie
            // inside them.
            unsafe {
                d.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&[acquire]),
                );
                d.cmd_copy_buffer(cmd, src, dst, &regions);
                d.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default()
                        .memory_barriers(&[written])
                        .buffer_memory_barriers(&[release]),
                );
            }
        });
        let picture = Picture {
            frames: self.clone(),
            buffer: Some(buffer),
            len,
        };
        ran.map(|()| picture)
    }

    /// Reads a picture's bytes back through a host-visible buffer.
    fn read(&self, src: vk::Buffer, len: usize) -> Result<Vec<u8>> {
        let mut host = Buffer::new(
            &self.gpu,
            len as u64,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
            "picture readback",
        )?;
        let dst = host.buffer;
        let ran = self.run(|d, cmd| {
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: len as u64,
            };
            let to_host = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ);
            // SAFETY: recording; both buffers hold `len` bytes.
            unsafe {
                d.cmd_copy_buffer(cmd, src, dst, &[region]);
                d.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().memory_barriers(&[to_host]),
                );
            }
        });
        let bytes = ran.map(|()| host.bytes()[..len].to_vec());
        host.destroy(&self.gpu);
        bytes
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        let w = self.work.get_mut().unwrap_or_else(|e| e.into_inner());
        // SAFETY: every run waited for its fence, so nothing is in flight.
        unsafe {
            self.gpu.device.destroy_fence(w.fence, None);
            self.gpu.device.destroy_command_pool(w.pool, None);
        }
        for mut b in self
            .spare
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .drain(..)
        {
            b.destroy(&self.gpu);
        }
    }
}

/// A decoder buffer imported for copies, with the command buffer and fence
/// of the copy out of it (one at a time).
struct Imported {
    frames: Arc<Inner>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
}

impl Imported {
    fn new(frames: &Arc<Inner>, fd: OwnedFd, len: usize) -> Result<Imported> {
        let d = &frames.gpu.device;
        let handle = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
        let mut props = vk::MemoryFdPropertiesKHR::default();
        // SAFETY: a live DMA-BUF descriptor.
        unsafe {
            frames
                .fd
                .get_memory_fd_properties(handle, fd.as_raw_fd(), &mut props)
        }
        .ctx("DMA-BUF memory types")?;
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle);
        // SAFETY: valid create info; destroyed on every path below.
        let buffer = unsafe {
            d.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(len as u64)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .push_next(&mut external),
                None,
            )
        }
        .ctx("decoder buffer")?;
        // SAFETY: valid buffer.
        let req = unsafe { d.get_buffer_memory_requirements(buffer) };
        let types = req.memory_type_bits & props.memory_type_bits;
        if types == 0 {
            // SAFETY: unused.
            unsafe { d.destroy_buffer(buffer, None) };
            return Err(Error::Unsupported(
                "no memory type fits the decoder's buffers".into(),
            ));
        }
        let mut import = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(handle)
            .fd(fd.as_raw_fd());
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        // SAFETY: imports the DMA-BUF; on success Vulkan owns the descriptor.
        let memory = unsafe {
            d.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(types.trailing_zeros())
                    .push_next(&mut import)
                    .push_next(&mut dedicated),
                None,
            )
        };
        let memory = match memory {
            Ok(m) => {
                let _ = fd.into_raw_fd();
                m
            }
            Err(e) => {
                // SAFETY: unused.
                unsafe { d.destroy_buffer(buffer, None) };
                return Err(Error::Vk {
                    context: "import DMA-BUF",
                    result: e,
                });
            }
        };
        // SAFETY: memory imported for this buffer.
        if let Err(e) = unsafe { d.bind_buffer_memory(buffer, memory, 0) } {
            // SAFETY: unused.
            unsafe {
                d.destroy_buffer(buffer, None);
                d.free_memory(memory, None);
            }
            return Err(Error::Vk {
                context: "bind decoder buffer",
                result: e,
            });
        }
        let mut imported = Imported {
            frames: frames.clone(),
            buffer,
            memory,
            cmd: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
        };
        let pool = lock(&frames.work);
        // SAFETY: from the held pool; freed by Drop, which also destroys
        // the fence (signalled: no copy in flight yet).
        unsafe {
            imported.cmd = d
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool.pool)
                        .command_buffer_count(1),
                )
                .ctx("picture copy commands")?[0];
            imported.fence = d
                .create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
                .ctx("picture copy fence")?;
        }
        Ok(imported)
    }
}

impl DecoderBuffer for Imported {
    fn copy(&self, ranges: &[(usize, usize)]) -> std::result::Result<GpuBytes, String> {
        // The command buffer and fence are free once the last copy is done.
        self.copied(true);
        self.frames
            .copy(self.buffer, self.cmd, self.fence, ranges)
            .map(|p| Arc::new(p) as GpuBytes)
            .map_err(|e| e.to_string())
    }

    fn copied(&self, wait: bool) -> bool {
        let d = &self.frames.gpu.device;
        // SAFETY: our fence.
        unsafe {
            if wait {
                let _ = d.wait_for_fences(&[self.fence], true, u64::MAX);
            }
            d.get_fence_status(self.fence).unwrap_or(true)
        }
    }
}

impl Drop for Imported {
    fn drop(&mut self) {
        let d = &self.frames.gpu.device;
        // SAFETY: after its last copy the GPU is done with all of it; the
        // command buffer goes back to the held pool. Null handles (a failed
        // `new`) are ignored.
        unsafe {
            if self.fence != vk::Fence::null() {
                let _ = d.wait_for_fences(&[self.fence], true, u64::MAX);
                d.destroy_fence(self.fence, None);
            }
            if self.cmd != vk::CommandBuffer::null() {
                let pool = lock(&self.frames.work);
                d.free_command_buffers(pool.pool, &[self.cmd]);
            }
            d.destroy_buffer(self.buffer, None);
            d.free_memory(self.memory, None);
        }
    }
}

/// A picture's bytes in GPU memory, as the decoder laid them out.
pub(crate) struct Picture {
    frames: Arc<Inner>,
    buffer: Option<Buffer>,
    len: usize,
}

impl Picture {
    pub(crate) fn buffer(&self) -> vk::Buffer {
        self.buffer
            .as_ref()
            .map_or(vk::Buffer::null(), |b| b.buffer)
    }
}

/// The GPU buffer of `bytes`, when this renderer's copier made it.
pub(crate) fn picture_buffer(bytes: &GpuBytes) -> Option<vk::Buffer> {
    bytes
        .as_any()
        .downcast_ref::<Picture>()
        .map(Picture::buffer)
}

impl GpuBuffer for Picture {
    fn len(&self) -> usize {
        self.len
    }

    fn read(&self) -> std::result::Result<Vec<u8>, String> {
        self.frames
            .read(self.buffer(), self.len)
            .map_err(|e| e.to_string())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Drop for Picture {
    fn drop(&mut self) {
        let Some(mut b) = self.buffer.take() else {
            return;
        };
        let mut spare = lock(&self.frames.spare);
        if spare.len() < SPARE {
            spare.push(b);
        } else {
            drop(spare);
            b.destroy(&self.frames.gpu);
        }
    }
}
