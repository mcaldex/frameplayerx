//! Uploads decoded video frames into sampled plane images: the GPU copies
//! them from the decode memory the decoders wrote them into, else from a
//! staging buffer the CPU copies them into first.

use crate::decode_memory::DecodeMemory;
use crate::gpu::Gpu;
use crate::mem::{self, Buffer, Image};
use crate::params::FrameDesc;
use crate::q10c::{self, Converter, Picture};
use crate::{Error, Result};
use ash::vk;
use fp_media::frame::Plane;
use fp_media::{PixelLayout, VideoFrame};
use gpu_allocator::MemoryLocation;
use std::sync::{Arc, LazyLock, mpsc};

/// The plane images of one frame slot.
pub(crate) struct VideoPlanes {
    pub images: Vec<Image>,
    pub desc: FrameDesc,
    /// Id of the frame currently uploaded.
    pub key: Option<u64>,
    /// Upload buffer for frames the CPU copies.
    staging: Option<Buffer>,
    /// The frame last copied from decode memory as it is, kept until the
    /// set is next uploaded: by then the frames reading the set, that copy
    /// among them, have finished.
    source: Option<Arc<VideoFrame>>,
    /// Q10C frames are copied here and converted into `images` on the GPU.
    q10c: Option<Picture>,
}

/// A plane's bytes in a buffer, for the GPU to copy into its image.
struct Region {
    buffer: vk::Buffer,
    offset: u64,
    /// Texels from one row's start to the next.
    row_len: u32,
    width: u32,
    height: u32,
}

/// Where each plane lies in decode memory, when all of them do in a way
/// the GPU can copy from.
fn in_place(memory: &DecodeMemory, planes: &[Plane]) -> Option<Vec<Region>> {
    planes
        .iter()
        .map(|p| {
            let texel = (p.components * p.bytes_per_sample) as usize;
            let row = p.width as usize * texel;
            if p.height == 0 || p.stride < row || !p.stride.is_multiple_of(texel) {
                return None;
            }
            let len = p.stride * (p.height as usize - 1) + row;
            if p.data.len() < len {
                return None;
            }
            let (buffer, offset) = memory.find(p.data.as_ptr(), len)?;
            offset.is_multiple_of(texel as u64).then_some(Region {
                buffer,
                offset,
                row_len: (p.stride / texel) as u32,
                width: p.width,
                height: p.height,
            })
        })
        .collect()
}

fn plane_formats(layout: PixelLayout) -> Option<Vec<vk::Format>> {
    Some(match layout {
        PixelLayout::I420 { bits: 8 }
        | PixelLayout::I422 { bits: 8 }
        | PixelLayout::I444 { bits: 8 } => vec![vk::Format::R8_UNORM; 3],
        PixelLayout::I420 { .. } | PixelLayout::I422 { .. } | PixelLayout::I444 { .. } => {
            vec![vk::Format::R16_UNORM; 3]
        }
        PixelLayout::Nv12 => vec![vk::Format::R8_UNORM, vk::Format::R8G8_UNORM],
        PixelLayout::P010 => vec![vk::Format::R16_UNORM, vk::Format::R16G16_UNORM],
        PixelLayout::DrmPrime | PixelLayout::Q10c => return None,
    })
}

fn frame_key(f: &VideoFrame) -> u64 {
    f.id
}

/// Copies frames into upload memory on helper threads started once, the
/// calling thread taking a share: an 8K frame is 50 MB, several
/// milliseconds for one core, and memory bandwidth goes further.
struct Copier {
    helpers: Vec<mpsc::Sender<Part>>,
}

static COPIER: LazyLock<Copier> = LazyLock::new(|| Copier::new(3));

/// A helper's share of a copy. [`Copier::copy`] waits until every part is
/// dropped, so the memory it points at outlives it.
struct Part {
    dst: *mut u8,
    src: *const u8,
    len: usize,
    _done: mpsc::Sender<()>,
}

// SAFETY: the pointers are only used while the caller of `Copier::copy`
// waits for the part to be dropped.
unsafe impl Send for Part {}

impl Part {
    fn run(self) {
        // SAFETY: `dst` and `src` are disjoint, `len` bytes each, and live
        // until the part is dropped.
        unsafe { std::ptr::copy_nonoverlapping(self.src, self.dst, self.len) };
    }
}

impl Copier {
    fn new(helpers: usize) -> Copier {
        let helpers = (0..helpers)
            .filter_map(|i| {
                let (tx, rx) = mpsc::channel::<Part>();
                std::thread::Builder::new()
                    .name(format!("frame copy {i}"))
                    .spawn(move || rx.into_iter().for_each(Part::run))
                    .ok()
                    .map(|_| tx)
            })
            .collect();
        Copier { helpers }
    }

    /// `dst.copy_from_slice(src)`.
    fn copy(&self, dst: &mut [u8], src: &[u8]) {
        const MIN_CHUNK: usize = 1 << 20;
        if dst.len() < MIN_CHUNK * 2 || self.helpers.is_empty() {
            dst.copy_from_slice(src);
            return;
        }
        assert_eq!(
            dst.len(),
            src.len(),
            "copy between slices of different lengths"
        );
        let chunk = dst
            .len()
            .div_ceil(self.helpers.len() + 1)
            .next_multiple_of(64);
        let (done, finished) = mpsc::channel();
        let mut parts = dst.chunks_mut(chunk).zip(src.chunks(chunk));
        let own = parts.next();
        for (helper, (d, s)) in self.helpers.iter().zip(parts.by_ref()) {
            let part = Part {
                dst: d.as_mut_ptr(),
                src: s.as_ptr(),
                len: d.len(),
                _done: done.clone(),
            };
            if let Err(mpsc::SendError(part)) = helper.send(part) {
                part.run();
            }
        }
        drop(done);
        for (d, s) in parts.chain(own) {
            d.copy_from_slice(s);
        }
        // Ends once every part has been dropped, its copy done.
        while finished.recv().is_ok() {}
    }
}

impl VideoPlanes {
    /// Uploads `frame` unless it is already resident. Recreates images when
    /// the size or layout changes. Returns true when images were recreated
    /// (descriptor sets must be rewritten).
    pub fn upload(
        slot: &mut Option<VideoPlanes>,
        gpu: &Gpu,
        cmd: vk::CommandBuffer,
        frame: &Arc<VideoFrame>,
        garbage: &mut Vec<Image>,
        converter: Option<&Converter>,
        memory: Option<&DecodeMemory>,
    ) -> Result<bool> {
        if frame.layout == PixelLayout::Q10c {
            return Self::upload_q10c(slot, gpu, cmd, frame, garbage, converter);
        }
        let Some(formats) = plane_formats(frame.layout) else {
            return Err(Error::Unsupported(
                "unsupported frame layout for upload".into(),
            ));
        };
        if frame.width == 0 || frame.height == 0 {
            return Ok(false);
        }
        if slot
            .as_ref()
            .is_some_and(|s| s.key == Some(frame_key(frame)))
        {
            return Ok(false);
        }
        let planes = frame.planes();
        if planes.is_empty() {
            return Ok(false);
        }
        let mut recreated = false;
        let needs_new = slot.as_ref().is_none_or(|s| {
            s.desc.width != frame.width
                || s.desc.height != frame.height
                || s.desc.layout != frame.layout
        });
        if needs_new {
            if let Some(old) = slot.take() {
                old.release(gpu, garbage);
            }
            let mut images = Vec::new();
            for (p, &f) in planes.iter().zip(&formats) {
                images.push(Image::new(
                    gpu,
                    p.width,
                    p.height,
                    &[f],
                    vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                    1,
                    "video plane",
                )?);
            }
            *slot = Some(VideoPlanes {
                images,
                desc: FrameDesc {
                    width: frame.width,
                    height: frame.height,
                    layout: frame.layout,
                    color: frame.color,
                },
                key: None,
                staging: None,
                source: None,
                q10c: None,
            });
            recreated = true;
        }
        let Some(s) = slot.as_mut() else {
            return Ok(recreated);
        };
        s.desc.color = frame.color;
        let regions = match memory.and_then(|m| in_place(m, &planes)) {
            Some(regions) => {
                s.source = Some(frame.clone());
                regions
            }
            None => {
                s.source = None;
                s.stage(gpu, &planes)?
            }
        };
        for (img, r) in s.images.iter().zip(regions) {
            mem::barrier(
                gpu,
                cmd,
                img.image,
                mem::range(0, 1),
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                mem::TOP,
                mem::TRANSFER_WRITE,
            );
            let region = vk::BufferImageCopy::default()
                .buffer_offset(r.offset)
                .buffer_row_length(r.row_len)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: r.width,
                    height: r.height,
                    depth: 1,
                });
            // SAFETY: recording; buffer and image are valid and sized.
            unsafe {
                gpu.device.cmd_copy_buffer_to_image(
                    cmd,
                    r.buffer,
                    img.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
            };
            mem::barrier(
                gpu,
                cmd,
                img.image,
                mem::range(0, 1),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                mem::TRANSFER_WRITE,
                mem::SHADER_READ,
            );
        }
        s.key = Some(frame_key(frame));
        Ok(recreated)
    }

    /// Copies the planes into the staging buffer, made or grown to fit.
    fn stage(&mut self, gpu: &Gpu, planes: &[Plane]) -> Result<Vec<Region>> {
        let mut total_bytes: usize = 0;
        for p in planes {
            let row_len = (p.width * p.components * p.bytes_per_sample) as usize;
            let len = if p.height > 0 {
                p.stride * (p.height as usize - 1) + row_len
            } else {
                0
            };
            total_bytes = (total_bytes + len + 63) & !63;
        }
        let size = self.staging.as_ref().map_or(0, |b| b.size);
        if total_bytes as u64 > size {
            let bigger = Buffer::new(
                gpu,
                (total_bytes as u64 + 65536).max(size * 2),
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
                "video staging",
            )?;
            if let Some(mut old) = self.staging.replace(bigger) {
                old.destroy(gpu);
            }
        }
        let Some(staging) = self.staging.as_mut() else {
            return Ok(Vec::new());
        };
        let buffer = staging.buffer;
        let dst = staging.bytes();
        let mut offset = 0usize;
        let mut regions = Vec::new();
        for p in planes {
            let row_len = (p.width * p.components * p.bytes_per_sample) as usize;
            let texel = (p.components * p.bytes_per_sample) as usize;
            let len = if p.height > 0 {
                p.stride * (p.height as usize - 1) + row_len
            } else {
                0
            };
            if p.data.len() < len || offset + len > dst.len() {
                continue;
            }
            let region = |offset: usize, row_len| Region {
                buffer,
                offset: offset as u64,
                row_len,
                width: p.width,
                height: p.height,
            };
            if p.stride >= row_len && p.stride % texel == 0 {
                COPIER.copy(&mut dst[offset..offset + len], &p.data[..len]);
                regions.push(region(offset, (p.stride / texel) as u32));
                offset = (offset + len + 63) & !63;
            } else {
                // Stride is misaligned: pack rows tightly without padding
                let tight_len = row_len * p.height as usize;
                if offset + tight_len <= dst.len() {
                    for y in 0..p.height as usize {
                        let src_start = y * p.stride;
                        let src_end = src_start + row_len;
                        if src_end <= p.data.len() {
                            let dst_start = offset + y * row_len;
                            dst[dst_start..dst_start + row_len]
                                .copy_from_slice(&p.data[src_start..src_end]);
                        }
                    }
                    regions.push(region(offset, p.width));
                    offset = (offset + tight_len + 63) & !63;
                }
            }
        }
        Ok(regions)
    }

    /// Converts a Q10C frame into P010 planes on the GPU.
    fn upload_q10c(
        slot: &mut Option<VideoPlanes>,
        gpu: &Gpu,
        cmd: vk::CommandBuffer,
        frame: &VideoFrame,
        garbage: &mut Vec<Image>,
        converter: Option<&Converter>,
    ) -> Result<bool> {
        let Some(conv) = converter else {
            return Err(Error::Unsupported(
                "this GPU can't show 10-bit hardware-decoded (Q10C) frames".into(),
            ));
        };
        let Some(pic) = frame.q10c() else {
            return Err(Error::Unsupported("Q10C frame without its buffer".into()));
        };
        if slot
            .as_ref()
            .is_some_and(|s| s.key == Some(frame_key(frame)))
        {
            return Ok(false);
        }
        let geometry = q10c::Geometry {
            width: frame.width,
            height: frame.height,
            stride: pic.stride,
            rows: pic.rows,
            uv_offset: pic.uv_offset,
            len: pic.data.len(),
        };
        let needs_new = slot.as_ref().is_none_or(|s| {
            s.desc.layout != frame.layout || s.q10c.as_ref().is_none_or(|p| p.geometry != geometry)
        });
        let mut recreated = false;
        if needs_new {
            if let Some(old) = slot.take() {
                old.release(gpu, garbage);
            }
            let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE;
            let (cw, ch) = (frame.width.div_ceil(2), frame.height.div_ceil(2));
            let mut images = vec![Image::new(
                gpu,
                frame.width,
                frame.height,
                &[vk::Format::R16_UNORM],
                usage,
                1,
                "video plane",
            )?];
            let made = Image::new(
                gpu,
                cw,
                ch,
                &[vk::Format::R16G16_UNORM],
                usage,
                1,
                "video plane",
            )
            .and_then(|uv| {
                images.push(uv);
                conv.picture(gpu, geometry, &images)
            });
            let picture = match made {
                Ok(p) => p,
                Err(e) => {
                    garbage.append(&mut images);
                    return Err(e);
                }
            };
            *slot = Some(VideoPlanes {
                images,
                desc: FrameDesc {
                    width: frame.width,
                    height: frame.height,
                    layout: frame.layout,
                    color: frame.color,
                },
                key: None,
                staging: None,
                source: None,
                q10c: Some(picture),
            });
            recreated = true;
        }
        let Some(s) = slot.as_mut() else {
            return Ok(recreated);
        };
        let Some(p) = s.q10c.as_mut() else {
            return Ok(recreated);
        };
        s.desc.color = frame.color;
        COPIER.copy(p.bytes(), pic.data);
        conv.record(gpu, cmd, p, &s.images);
        s.key = Some(frame_key(frame));
        Ok(recreated)
    }

    /// Gives up the set when it is replaced: its images to `garbage` (the
    /// frame in flight may still name them), the rest at once (the set is
    /// only replaced once the last frame reading it has finished).
    fn release(mut self, gpu: &Gpu, garbage: &mut Vec<Image>) {
        garbage.append(&mut self.images);
        if let Some(mut b) = self.staging.take() {
            b.destroy(gpu);
        }
        if let Some(mut p) = self.q10c.take() {
            p.destroy(gpu);
        }
    }

    pub fn destroy(&mut self, gpu: &Gpu) {
        for i in &mut self.images {
            i.destroy(gpu);
        }
        if let Some(b) = self.staging.as_mut() {
            b.destroy(gpu);
        }
        if let Some(p) = self.q10c.as_mut() {
            p.destroy(gpu);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_like_copy_from_slice() {
        for copier in [
            Copier::new(3),
            Copier {
                helpers: Vec::new(),
            },
        ] {
            for len in [0, 100, (2 << 20) - 1, (3 << 20) + 17, 8 << 20] {
                let src: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
                let mut dst = vec![0u8; len];
                copier.copy(&mut dst, &src);
                assert!(dst == src, "{len} bytes");
            }
        }
    }

    #[test]
    fn a_gone_helper_leaves_its_share_to_the_caller() {
        let (tx, rx) = mpsc::channel();
        drop(rx);
        let copier = Copier { helpers: vec![tx] };
        let src: Vec<u8> = (0..4 << 20).map(|i| (i % 251) as u8).collect();
        let mut dst = vec![0u8; src.len()];
        copier.copy(&mut dst, &src);
        assert!(dst == src);
    }
}
