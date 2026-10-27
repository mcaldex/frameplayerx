//! Memory the decoders write pictures into ([`fp_media::FrameMemory`]):
//! host-visible buffers the GPU copies the planes from as they are, so the
//! render thread no longer copies each picture into a staging buffer first.

use crate::gpu::Gpu;
use ash::vk;
use fp_core::sync::lock;
use fp_media::{BLOCK_ALIGN, Block, FrameMemory};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

/// Coherent host-visible memory for decoded pictures; clones share it.
#[derive(Clone)]
pub(crate) struct DecodeMemory(Arc<Inner>);

struct Inner {
    gpu: Arc<Gpu>,
    /// Memory types for pictures the CPU reads back (cached), and for those
    /// it only writes (uncached where there is such: the GPU then reads
    /// them without looking in the CPU's caches).
    cached: u32,
    written: u32,
    /// The blocks alive, by address: their length and buffer.
    blocks: Mutex<BTreeMap<usize, (usize, vk::Buffer)>>,
}

impl DecodeMemory {
    /// `None` without coherent host-visible memory.
    pub(crate) fn new(gpu: &Arc<Gpu>) -> Option<DecodeMemory> {
        // SAFETY: valid physical device.
        let (memory, kind) = unsafe {
            (
                gpu.instance.get_physical_device_memory_properties(gpu.pdev),
                gpu.instance
                    .get_physical_device_properties(gpu.pdev)
                    .device_type,
            )
        };
        let (mut cached, mut uncached) = (0, 0);
        for (i, t) in memory.memory_types_as_slice().iter().enumerate() {
            let f = t.property_flags;
            // A discrete GPU's own host-visible memory is a small window:
            // pictures go in the computer's memory.
            let window = kind == vk::PhysicalDeviceType::DISCRETE_GPU
                && f.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL);
            if window
                || !f.contains(
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
            {
                continue;
            }
            if f.contains(vk::MemoryPropertyFlags::HOST_CACHED) {
                cached |= 1 << i;
            } else {
                uncached |= 1 << i;
            }
        }
        let written = if uncached != 0 { uncached } else { cached };
        if written == 0 {
            log::info!("decoded pictures: no coherent host-visible memory, copied for the GPU");
            return None;
        }
        log::info!(
            "decoded pictures: written where the GPU reads them; software decoders' {}",
            if cached != 0 {
                "too"
            } else {
                "copied (no cached memory)"
            }
        );
        Some(DecodeMemory(Arc::new(Inner {
            gpu: gpu.clone(),
            cached,
            written,
            blocks: Mutex::new(BTreeMap::new()),
        })))
    }

    /// The buffer holding the `len` bytes at `ptr`, and where they start in
    /// it.
    pub(crate) fn find(&self, ptr: *const u8, len: usize) -> Option<(vk::Buffer, u64)> {
        let at = ptr as usize;
        let blocks = lock(&self.0.blocks);
        let (&start, &(size, buffer)) = blocks.range(..=at).next_back()?;
        (at + len <= start + size).then_some((buffer, (at - start) as u64))
    }
}

impl FrameMemory for DecodeMemory {
    fn alloc(&self, len: usize, reads: bool) -> Option<Block> {
        let inner = &self.0;
        let types = if reads { inner.cached } else { inner.written };
        let gpu = &inner.gpu;
        if types == 0 {
            return None;
        }
        // SAFETY: valid create info.
        let buffer = unsafe {
            gpu.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(len as u64)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                None,
            )
        }
        .ok()?;
        let mut owner = Owned {
            memory: inner.clone(),
            buffer,
            alloc: None,
            at: None,
        };
        // SAFETY: valid buffer.
        let mut requirements = unsafe { gpu.device.get_buffer_memory_requirements(buffer) };
        requirements.memory_type_bits &= types;
        let alloc = gpu
            .alloc()
            .allocate(&AllocationCreateDesc {
                name: "decoded picture",
                requirements,
                location: MemoryLocation::Unknown,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedBuffer(buffer),
            })
            .ok()?;
        let alloc = owner.alloc.insert(alloc);
        // SAFETY: memory from the allocator matches the requirements.
        unsafe {
            gpu.device
                .bind_buffer_memory(buffer, alloc.memory(), alloc.offset())
        }
        .ok()?;
        let ptr = alloc.mapped_ptr()?.cast::<u8>();
        if !(ptr.as_ptr() as usize).is_multiple_of(BLOCK_ALIGN) {
            return None;
        }
        let at = ptr.as_ptr() as usize;
        lock(&inner.blocks).insert(at, (len, buffer));
        owner.at = Some(at);
        // SAFETY: the mapping holds `len` bytes until the owner frees it.
        Some(unsafe { Block::new(ptr, len, Box::new(owner)) })
    }
}

/// A block's buffer and memory, given back when the block is freed.
struct Owned {
    memory: Arc<Inner>,
    buffer: vk::Buffer,
    alloc: Option<Allocation>,
    /// Where it is mapped, once in the blocks.
    at: Option<usize>,
}

impl Drop for Owned {
    fn drop(&mut self) {
        let gpu = &self.memory.gpu;
        if let Some(at) = self.at {
            lock(&self.memory.blocks).remove(&at);
        }
        if let Some(a) = self.alloc.take() {
            let _ = gpu.alloc().free(a);
        }
        // SAFETY: a block is freed once no picture in it is read: the
        // renderer keeps the picture it copies until the copy is done.
        unsafe { gpu.device.destroy_buffer(self.buffer, None) };
    }
}
