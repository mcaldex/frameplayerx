//! The depth measurement pass for automatic stereo focus: one
//! measurement's samples ([`fp_core::depth::grid`]) drawn from the frame
//! the eye passes show (shaders/measure.wgsl), with the scene's own
//! uniforms and the frame's own descriptors, recorded in the frame's
//! command buffer right after the eye passes while a frame in the
//! decoder's buffer is still acquired. A small float target is copied to
//! host memory in the same submission and read once that slot's fence has
//! signalled, so nothing waits for it. One measurement is on its way at a
//! time.

use crate::Result;
use crate::gpu::Gpu;
use crate::mem::{self, Buffer, Image};
use crate::pipeline::{self, PipelineDesc, ShadingRate};
use crate::tp10_eyes::Tp10Eyes;
use ash::vk;
use fp_core::depth::grid::COARSE;
use fp_core::depth::{GridSpec, Grids};
use gpu_allocator::MemoryLocation;

/// The target, pixels: both grids side by side and the coarse picture
/// beside them. A grid is 30 to 110 samples wide (a 32-sample window and
/// disparities to 0.24 m at 120 mm) and 30 to 60 tall, so 512 x 128 holds
/// any measurement planned, at 256 KB.
const WIDTH: u32 = 512;
const HEIGHT: u32 = 128;

/// `struct Measure` in measure.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Push {
    origin: [f32; 4],
    grid: [f32; 4],
}

/// How the frame's descriptors are bound for the pass.
pub(crate) enum Bind<'a> {
    /// The slot's scene descriptor set (uploaded planes, NV12 in place).
    Set(vk::DescriptorSet),
    /// The slot's TP10 descriptor buffer (Q10C in place).
    Tp10(&'a Tp10Eyes),
}

pub(crate) struct MeasurePass {
    target: Image,
    readback: Buffer,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    /// The pipeline and layout for Q10C read in place, where that path
    /// exists and its pipeline could be made.
    tp10: Option<(vk::PipelineLayout, vk::Pipeline)>,
    /// Asked for (tag, grids), not drawn yet.
    asked: Option<(u64, GridSpec)>,
    /// Drawn in this slot's submission.
    flying: Option<(u64, GridSpec, usize)>,
    done: Option<(u64, Grids)>,
}

/// The target's extent for `spec`; `None` if it doesn't fit.
fn extent(spec: &GridSpec) -> Option<(u32, u32)> {
    let w = 2 * spec.cols as u32 + COARSE as u32;
    let h = (spec.rows as u32).max(COARSE as u32);
    (spec.cols > 0 && spec.rows > 0 && w <= WIDTH && h <= HEIGHT).then_some((w, h))
}

impl MeasurePass {
    /// The pass for frames bound through `scene_layout`'s sets, and through
    /// `tp10`'s descriptor buffers where 10-bit frames are read in place.
    pub fn new(
        gpu: &Gpu,
        scene_layout: vk::DescriptorSetLayout,
        tp10: Option<&Tp10Eyes>,
    ) -> Result<MeasurePass> {
        let mut target = Image::new(
            gpu,
            WIDTH,
            HEIGHT,
            &[vk::Format::R32_SFLOAT],
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            1,
            "depth measurement",
        )?;
        let readback = match Buffer::new(
            gpu,
            (WIDTH * HEIGHT * 4) as u64,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
            "depth measurement readback",
        ) {
            Ok(b) => b,
            Err(e) => {
                target.destroy(gpu);
                return Err(e);
            }
        };
        let mut m = MeasurePass {
            target,
            readback,
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            tp10: None,
            asked: None,
            flying: None,
            done: None,
        };
        if let Err(e) = m.create(gpu, scene_layout, tp10) {
            m.destroy(gpu);
            return Err(e);
        }
        Ok(m)
    }

    fn create(
        &mut self,
        gpu: &Gpu,
        scene_layout: vk::DescriptorSetLayout,
        tp10: Option<&Tp10Eyes>,
    ) -> Result<()> {
        let push = size_of::<Push>() as u32;
        let desc = |spv, layout, descriptor_buffer| PipelineDesc {
            spv,
            layout,
            color_format: vk::Format::R32_SFLOAT,
            vertex_stride: 0,
            attributes: &[],
            blend: false,
            descriptor_buffer,
            shading_rate: ShadingRate::Unused,
        };
        self.layout = pipeline::layout(gpu, scene_layout, push)?;
        self.pipeline = pipeline::create(gpu, &desc(crate::MEASURE_SPV, self.layout, false))?;
        if let Some(t) = tp10 {
            let layout = pipeline::layout(gpu, t.set_layout(), push)?;
            match pipeline::create(gpu, &desc(crate::MEASURE_TP10_SPV, layout, true)) {
                Ok(p) => self.tp10 = Some((layout, p)),
                Err(e) => {
                    log::warn!("depth isn't measured in 10-bit video read in place: {e}");
                    // SAFETY: just made, not in use.
                    unsafe { gpu.device.destroy_pipeline_layout(layout, None) };
                }
            }
        }
        Ok(())
    }

    /// Asks for `spec`'s grids, tagged `tag`; refused while one is asked
    /// for, on its way or not yet taken, and for grids that don't fit.
    pub fn ask(&mut self, tag: u64, spec: &GridSpec) -> bool {
        if self.busy() || self.done.is_some() || extent(spec).is_none() {
            return false;
        }
        self.asked = Some((tag, *spec));
        true
    }

    /// Whether one is asked for or on its way.
    pub fn busy(&self) -> bool {
        self.asked.is_some() || self.flying.is_some()
    }

    /// Forgets the one asked for, on its way or done: it never comes back.
    pub fn cancel(&mut self) {
        self.asked = None;
        self.flying = None;
        self.done = None;
    }

    /// The finished measurement, once.
    pub fn take(&mut self) -> Option<(u64, Grids)> {
        self.done.take()
    }

    /// Whether the measurement asked for can be drawn with `bind`'s kind of
    /// frame (a 10-bit one read in place needs the TP10 pipeline).
    pub fn can_draw(&self, tp10: bool) -> bool {
        self.asked.is_some() && (!tp10 || self.tp10.is_some())
    }

    /// Records the measurement asked for into slot `slot`'s command buffer
    /// `cmd`, after the eye passes, reading the frame through `bind`.
    pub fn record(&mut self, gpu: &Gpu, cmd: vk::CommandBuffer, slot: usize, bind: Bind) {
        let Some((tag, spec)) = self.asked else {
            return;
        };
        let Some((w, h)) = extent(&spec) else {
            self.asked = None;
            return;
        };
        let (layout, pipe) = match bind {
            Bind::Set(_) => (self.layout, self.pipeline),
            Bind::Tp10(_) => match self.tp10 {
                Some(t) => t,
                None => return,
            },
        };
        let push = Push {
            origin: [
                spec.origin[0].alpha,
                spec.origin[0].beta,
                spec.origin[1].alpha,
                spec.origin[1].beta,
            ],
            grid: [spec.pitch, spec.cols as f32, spec.rows as f32, 0.0],
        };
        let d = &gpu.device;
        let image = self.target.image;
        mem::barrier(
            gpu,
            cmd,
            image,
            mem::range(0, 1),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            mem::TOP,
            mem::COLOR_WRITE,
        );
        let area = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: w,
                height: h,
            },
        };
        let attachment = [vk::RenderingAttachmentInfo::default()
            .image_view(self.target.level0)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let region = vk::BufferImageCopy::default()
            .buffer_row_length(w)
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
        // SAFETY: recording outside any rendering, after the eye passes
        // that read the same frame and uniforms; the target and readback
        // aren't in use (the last measurement's slot has finished, as it
        // was taken before this one was asked for).
        unsafe {
            d.cmd_begin_rendering(
                cmd,
                &vk::RenderingInfo::default()
                    .render_area(area)
                    .layer_count(1)
                    .color_attachments(&attachment),
            );
            d.cmd_set_viewport(
                cmd,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: w as f32,
                    height: h as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            d.cmd_set_scissor(cmd, 0, &[area]);
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipe);
            match bind {
                Bind::Set(set) => d.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout,
                    0,
                    &[set],
                    &[],
                ),
                Bind::Tp10(t) => t.bind_descriptors(cmd, slot, layout),
            }
            d.cmd_push_constants(
                cmd,
                layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push),
            );
            d.cmd_draw(cmd, 3, 1, 0, 0);
            d.cmd_end_rendering(cmd);
            mem::barrier(
                gpu,
                cmd,
                image,
                mem::range(0, 1),
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                mem::COLOR_WRITE,
                mem::TRANSFER_READ,
            );
            d.cmd_copy_image_to_buffer(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.readback.buffer,
                &[region],
            );
            let host = [vk::BufferMemoryBarrier2::default()
                .buffer(self.readback.buffer)
                .size(vk::WHOLE_SIZE)
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ)];
            d.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().buffer_memory_barriers(&host),
            );
        }
        self.asked = None;
        self.flying = Some((tag, spec, slot));
    }

    /// Reads the measurement slot `slot`'s last submission drew, if it drew
    /// one; call once its fence has signalled.
    pub fn collect(&mut self, slot: usize) {
        let Some((tag, spec, s)) = self.flying else {
            return;
        };
        if s != slot {
            return;
        }
        self.flying = None;
        let Some((w, _)) = extent(&spec) else {
            return;
        };
        let bytes = self.readback.bytes();
        let at = |x: usize, y: usize| {
            let i = (y * w as usize + x) * 4;
            f32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]])
        };
        let grid = |camera: usize| {
            let mut v = Vec::with_capacity(spec.len());
            for row in 0..spec.rows {
                for col in 0..spec.cols {
                    v.push(at(camera * spec.cols + col, row));
                }
            }
            v
        };
        let samples = [grid(0), grid(1)];
        let mut coarse = Vec::with_capacity(COARSE * COARSE);
        for row in 0..COARSE {
            for col in 0..COARSE {
                coarse.push(at(2 * spec.cols + col, row));
            }
        }
        self.done = Some((
            tag,
            Grids {
                spec,
                samples,
                coarse,
            },
        ));
    }

    pub fn destroy(&mut self, gpu: &Gpu) {
        self.target.destroy(gpu);
        self.readback.destroy(gpu);
        // SAFETY: not in use (the device is idle); nulls are ignored.
        unsafe {
            let d = &gpu.device;
            if let Some((layout, pipe)) = self.tp10.take() {
                d.destroy_pipeline(pipe, None);
                d.destroy_pipeline_layout(layout, None);
            }
            d.destroy_pipeline(std::mem::take(&mut self.pipeline), None);
            d.destroy_pipeline_layout(std::mem::take(&mut self.layout), None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fp_core::depth::{Angles, Matcher, Request, Window};

    #[test]
    fn every_measurement_planned_fits_the_target() {
        let m = Matcher::new();
        for baseline in [0.03, 0.065, 0.12] {
            for window in [Window::HEAD, Window::EYES] {
                for alpha in [0.45, 1.0, std::f32::consts::FRAC_PI_2, 2.0, 2.6] {
                    let plan = m
                        .plan(&Request {
                            aim: Angles { alpha, beta: 0.3 },
                            baseline,
                            nearest: 0.24,
                            window,
                        })
                        .unwrap();
                    assert!(
                        extent(&plan.grid).is_some(),
                        "{baseline} {window:?} {alpha}: {} x {}",
                        plan.grid.cols,
                        plan.grid.rows
                    );
                }
            }
        }
    }
}
