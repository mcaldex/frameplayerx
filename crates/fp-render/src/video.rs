//! Uploads decoded video frames into sampled plane images.

use crate::gpu::Gpu;
use crate::mem::{self, Buffer, Image};
use crate::params::FrameDesc;
use crate::q10c::{self, Converter, Picture};
use crate::{Error, Result};
use ash::vk;
use fp_media::{GpuBytes, PixelLayout, VideoFrame};
use gpu_allocator::MemoryLocation;

/// The plane images of one frame slot.
pub(crate) struct VideoPlanes {
    pub images: Vec<Image>,
    pub desc: FrameDesc,
    /// Id of the frame currently uploaded.
    pub key: Option<u64>,
    /// Upload buffer for frames the CPU can read.
    staging: Option<Buffer>,
    /// Q10C frames are copied here and converted into `images` on the GPU.
    q10c: Option<Picture>,
    /// The GPU picture last copied into this set, kept until its next
    /// upload: the frame reading the set may still be copying from it.
    source: Option<GpuBytes>,
}

fn plane_formats(layout: PixelLayout) -> Option<Vec<vk::Format>> {
    Some(match layout {
        PixelLayout::I420 { bits: 8 } => vec![vk::Format::R8_UNORM; 3],
        PixelLayout::I420 { .. } => vec![vk::Format::R16_UNORM; 3],
        PixelLayout::Nv12 => vec![vk::Format::R8_UNORM, vk::Format::R8G8_UNORM],
        PixelLayout::P010 => vec![vk::Format::R16_UNORM, vk::Format::R16G16_UNORM],
        PixelLayout::Q10c => return None,
    })
}

/// Each plane's size: luma, then the 4:2:0 chroma plane(s).
fn plane_sizes(layout: PixelLayout, width: u32, height: u32) -> Vec<(u32, u32)> {
    let chroma = (width.div_ceil(2), height.div_ceil(2));
    match layout {
        PixelLayout::I420 { .. } => vec![(width, height), chroma, chroma],
        _ => vec![(width, height), chroma],
    }
}

/// Where a plane is in a buffer: offset, row length in texels, width and
/// height.
type Region = (u64, u32, u32, u32);

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
        converter: Option<&Converter>,
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
            for ((w, h), &f) in plane_sizes(frame.layout, frame.width, frame.height)
                .into_iter()
                .zip(&formats)
            {
                images.push(Image::new(
                    gpu,
                    w,
                    h,
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
                q10c: None,
                source: None,
            });
            recreated = true;
        }
        let Some(s) = slot.as_mut() else {
            return Ok(recreated);
        };
        s.desc.color = frame.color;
        let (src, regions) = match frame.gpu() {
            // The decoder's NV12: planes stacked at the decoder's stride.
            Some(g) => {
                let src = crate::frames::picture_buffer(&g.bytes).ok_or_else(|| {
                    Error::Unsupported("a GPU picture from another renderer".into())
                })?;
                let (cw, ch) = (frame.width.div_ceil(2), frame.height.div_ceil(2));
                s.source = Some(g.bytes.clone());
                let regions = vec![
                    (0, g.stride, frame.width, frame.height),
                    (g.uv_offset as u64, g.stride / 2, cw, ch),
                ];
                (src, regions)
            }
            None => {
                s.source = None;
                let regions = s.stage(gpu, frame)?;
                let Some(staging) = &s.staging else {
                    return Ok(recreated);
                };
                (staging.buffer, regions)
            }
        };
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
                    src,
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

    /// Copies a frame the CPU can read into the staging buffer, returning
    /// each plane's [`Region`] there.
    fn stage(&mut self, gpu: &Gpu, frame: &VideoFrame) -> Result<Vec<Region>> {
        let planes = frame.planes();
        // Enough for every plane (strides vary across frames).
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
        if self
            .staging
            .as_ref()
            .is_none_or(|b| total_bytes as u64 > b.size)
        {
            let grown = self.staging.as_ref().map_or(0, |b| b.size * 2);
            let bigger = Buffer::new(
                gpu,
                (total_bytes as u64 + 65536).max(grown),
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
        let mut offset = 0usize;
        let mut regions = Vec::new();
        let dst = staging.bytes();
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
        if slot
            .as_ref()
            .is_some_and(|s| s.key == Some(frame_key(frame)))
        {
            return Ok(false);
        }
        let (stride, rows, uv_offset, len) = match (frame.gpu(), frame.q10c()) {
            (Some(g), _) => (g.stride, g.rows, g.uv_offset, g.bytes.len()),
            (None, Some(pic)) => (pic.stride, pic.rows, pic.uv_offset, pic.data.len()),
            (None, None) => {
                return Err(Error::Unsupported("Q10C frame without its buffer".into()));
            }
        };
        let geometry = q10c::Geometry {
            width: frame.width,
            height: frame.height,
            stride,
            rows,
            uv_offset,
            len,
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
                q10c: Some(picture),
                source: None,
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
        match (frame.gpu(), frame.q10c()) {
            (Some(g), _) => {
                let src = crate::frames::picture_buffer(&g.bytes).ok_or_else(|| {
                    Error::Unsupported("a GPU picture from another renderer".into())
                })?;
                p.copy_from(gpu, cmd, src)?;
                s.source = Some(g.bytes.clone());
            }
            (None, Some(pic)) => {
                par_copy(p.bytes(), pic.data);
                s.source = None;
            }
            (None, None) => {}
        }
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
