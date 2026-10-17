//! Uploads decoded video frames into sampled plane images.

use crate::Result;
use crate::gpu::Gpu;
use crate::mem::{self, Buffer, Image};
use crate::params::FrameDesc;
use ash::vk;
use fp_media::{PixelLayout, VideoFrame};
use gpu_allocator::MemoryLocation;

/// The plane images of one frame slot.
pub(crate) struct VideoPlanes {
    pub images: Vec<Image>,
    pub desc: FrameDesc,
    /// Id of the frame currently uploaded.
    pub key: Option<u64>,
    staging: Buffer,
}

fn plane_formats(layout: PixelLayout) -> Option<Vec<vk::Format>> {
    Some(match layout {
        PixelLayout::I420 { bits: 8 } => vec![vk::Format::R8_UNORM; 3],
        PixelLayout::I420 { .. } => vec![vk::Format::R16_UNORM; 3],
        PixelLayout::Nv12 => vec![vk::Format::R8_UNORM, vk::Format::R8G8_UNORM],
        PixelLayout::P010 => vec![vk::Format::R16_UNORM, vk::Format::R16G16_UNORM],
        PixelLayout::DrmPrime => return None,
    })
}

fn frame_key(f: &VideoFrame) -> u64 {
    f.id
}

/// `dst.copy_from_slice(src)` across a few threads: an 8K frame is 50 MB,
/// several milliseconds for one core, and memory bandwidth goes further.
fn par_copy(dst: &mut [u8], src: &[u8]) {
    const THREADS: usize = 4;
    const MIN_CHUNK: usize = 1 << 20;
    if dst.len() < MIN_CHUNK * 2 {
        dst.copy_from_slice(src);
        return;
    }
    let chunk = dst.len().div_ceil(THREADS).next_multiple_of(64);
    std::thread::scope(|scope| {
        for (d, s) in dst.chunks_mut(chunk).zip(src.chunks(chunk)) {
            scope.spawn(move || d.copy_from_slice(s));
        }
    });
}

impl VideoPlanes {
    /// Uploads `frame` unless it is already resident. Recreates images when
    /// the size or layout changes. Returns true when images were recreated
    /// (descriptor sets must be rewritten).
    pub fn upload(
        slot: &mut Option<VideoPlanes>,
        gpu: &Gpu,
        cmd: vk::CommandBuffer,
        frame: &VideoFrame,
        garbage: &mut Vec<Image>,
    ) -> Result<bool> {
        let Some(formats) = plane_formats(frame.layout) else {
            return Err(crate::Error::Unsupported(
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
            if let Some(mut old) = slot.take() {
                garbage.append(&mut old.images);
                old.staging.destroy(gpu);
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
            let bytes: u64 = planes
                .iter()
                .map(|p| (p.stride * p.height as usize) as u64 + 4096)
                .sum::<u64>()
                .max(65536);
            let staging = Buffer::new(
                gpu,
                bytes,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
                "video staging",
            )?;
            *slot = Some(VideoPlanes {
                images,
                desc: FrameDesc {
                    width: frame.width,
                    height: frame.height,
                    layout: frame.layout,
                    color: frame.color,
                },
                key: None,
                staging,
            });
            recreated = true;
        }
        let Some(s) = slot.as_mut() else {
            return Ok(recreated);
        };
        s.desc.color = frame.color;

        // Ensure staging buffer is large enough for all planes (in case strides vary across frames)
        let mut total_bytes: usize = 0;
        for p in &planes {
            let row_len = (p.width * p.components * p.bytes_per_sample) as usize;
            let len = if p.height > 0 {
                p.stride * (p.height as usize - 1) + row_len
            } else {
                0
            };
            total_bytes = (total_bytes + len + 63) & !63;
        }
        if total_bytes as u64 > s.staging.size {
            let bigger = Buffer::new(
                gpu,
                (total_bytes as u64 + 65536).max(s.staging.size * 2),
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
                "video staging",
            )?;
            let mut old = std::mem::replace(&mut s.staging, bigger);
            old.destroy(gpu);
        }

        // Copy every plane into staging
        let mut offset = 0usize;
        let mut regions = Vec::new();
        {
            let dst = s.staging.bytes();
            for p in &planes {
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
                if p.stride >= row_len && p.stride % texel == 0 {
                    par_copy(&mut dst[offset..offset + len], &p.data[..len]);
                    regions.push((offset as u64, (p.stride / texel) as u32, p.width, p.height));
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
                        regions.push((offset as u64, p.width, p.width, p.height));
                        offset = (offset + tight_len + 63) & !63;
                    }
                }
            }
        }
        s.key = Some(frame_key(frame));
        for (img, (off, row_len, w, h)) in s.images.iter().zip(regions) {
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
                .buffer_offset(off)
                .buffer_row_length(row_len)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: w,
                    height: h,
                    depth: 1,
                });
            // SAFETY: recording; buffer and image are valid and sized.
            unsafe {
                gpu.device.cmd_copy_buffer_to_image(
                    cmd,
                    s.staging.buffer,
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

    pub fn destroy(&mut self, gpu: &Gpu) {
        for i in &mut self.images {
            i.destroy(gpu);
        }
        self.staging.destroy(gpu);
    }
}
