//! Gaze-driven comfort focus for stereo 180° video.
//!
//! The eye passes normally sample each eye's half of the frame by ray
//! direction alone, so the video's disparity stays horizontal in the room
//! whatever the head does. Roll or swing the head and the eyes' baseline no
//! longer matches it: near detail gets vertical disparity and won't fuse.
//! Comfort focus gives each camera a position, [`LENS_SPACING`] apart, and
//! models its picture as a sphere of radius [`Focus::distance`] centred on
//! it. Each eye's ray starts at that eye's real offset from its camera and
//! is intersected with the sphere, then the hit is looked up from the
//! camera, which is geometrically right for any point at that radius. A
//! level, forward-facing head with eyes at the lens spacing has zero
//! offsets, so the picture is unchanged.
//!
//! Close objects also ask the eyes to converge strongly, while the
//! headset's lenses stay focused at a fixed distance; the mismatch
//! (vergence-accommodation conflict) is what makes close-ups tiring. Comfort
//! focus eases it the way stereo depth grading and auto-convergence do: when
//! what the viewer looks at is close, it turns both eyes' images apart
//! symmetrically by a small angle about the head's up, so it asks for less
//! convergence ([`Focus::relief`]). Nothing nearer than ~0.85 m leaves that
//! untouched, and the cyclopean picture never moves or scales. Less
//! convergence makes what is looked at seem farther, and so bigger (half
//! again at 0.4 m), and any change of the easing shows as it growing or
//! shrinking.
//!
//! Measurements aim at the tracked gaze, smoothed, whenever the headset
//! tracks the eyes ([`AimFilter`]), and at the head's forward otherwise.
//! Only what lies there is measured and eased: everything else keeps its
//! disparity, so a far background is pushed past infinity only while the
//! viewer looks at something close. The easing follows the distance in
//! focus, which moves once the latest two readings agree on another, and
//! only for a change of a fifth or more straight ahead ([`RELIEF_STEP`]),
//! more to the side where readings are less precise ([`side_scale`]), so the
//! eyes
//! roaming over the nearer and farther parts of what they look at leave it
//! still. It glides in gently, over about a second, so what is looked at
//! doesn't visibly grow; out more promptly, as what is then looked at is
//! farther and pushed past infinity while the easing lasts, once something
//! farther has stayed in focus for a quarter of a second (a glance there and
//! back leaves it be) or nothing has been readable for a second. The
//! re-projection follows the distance in focus only once holding its own
//! would misalign either eye's picture of what is looked at by more than
//! [`REDRAW_TOLERANCE_RAD`] ([`redraw_lever`]): it holds still with the
//! head level and facing the video's front, where it changes nothing, and
//! follows closely lying down, where it keeps things in one piece.
//!
//! After the eye passes, whenever there is something new to measure (a new
//! video frame, or the eyes moved on by [`REMEASURE_RAD`]) and at least
//! every [`MEASURE_EVERY`], `fs_focus` in `scene.wgsl` draws
//! strips of both eyes' luma round the aim into a small float target, with
//! the eye passes' own bindings, so every frame they can read is measured
//! (a 10-bit one in the decoder's buffer too). The strips lie on a lattice
//! of angles about the cameras' baseline ([`aim_angles`]): each row is the
//! line both cameras see a point on, so a disparity is a shift along it by
//! its own angle, with no vertical part, wherever the aim is. [`FocusPass`]
//! then matches them (`shaders/focus.wgsl`), one 64-thread workgroup per
//! patch of a 3 x 3 cluster, scoring 64 candidate disparities per patch.
//! [`ComfortFocus`] reads the curves back once the slot's fence has
//! signalled and triangulates the middle patch's disparity ([`nearness`]),
//! or takes the ring's median when the middle has nothing to match (biased
//! towards "far", so mistakes mean less correction rather than a wrong
//! one).

use crate::gpu::Gpu;
use crate::mem::{self, Buffer, Image};
use crate::params::{MAX_COMFORT_CORR_RAD, SceneParams};
use crate::{Error, Result, VkContext};
use ash::vk;
use glam::{Mat3, Quat, Vec2, Vec3};
use gpu_allocator::MemoryLocation;
use std::f32::consts::FRAC_PI_2;
use std::time::{Duration, Instant};

/// Number of disparity candidates evaluated per patch (`focus.wgsl`).
pub(crate) const CANDIDATES: usize = 64;
/// Candidates below zero disparity (`focus.wgsl`): footage shot with the
/// cameras slightly converged puts distant things there, and without these
/// they would have no candidate to match.
const NEGATIVE: usize = 4;
/// Spacing of the sample lattice, radians: 0.2° (`scene.wgsl`). Each
/// sample averages taps over its cell, which low-passes the picture to a
/// resolution both eyes share, and a 21-sample window still covers only
/// about 4°, a hand's width at arm's length. Candidates are half a step
/// apart, per eye.
const STEP: f32 = 0.0035;
/// Patch grid columns and rows round the aim (`scene.wgsl`).
const GRID_W: usize = 3;
const GRID_H: usize = 3;
/// Patches measured; the middle one is at the aim.
pub(crate) const PATCHES: usize = GRID_W * GRID_H;
const AIM_PATCH: usize = PATCHES / 2;
/// Lattice angle between neighbouring patches' columns, radians
/// (`FOCUS_GRID_X` in `scene.wgsl`).
const GRID_ALONG: f32 = 0.06;
/// Farthest the aim may be from straight ahead along the baseline, radians:
/// 65°. Past it the cameras look nearly along their own baseline: a
/// disparity is less than sin 25° = 0.42 of what it is straight ahead and the
/// lattice's rows close in, so a reading there would be mostly noise.
const MAX_ALONG: f32 = 65.0 * std::f32::consts::PI / 180.0;
/// Each patch's strips in the samples target (`scene.wgsl`, `focus.wgsl`):
/// columns for each eye, and rows.
const SAMPLE_COLS: u32 = 53;
const SAMPLE_ROWS: u32 = 19;
/// The samples target's format: luma, and -1 off the picture.
pub(crate) const SAMPLES_FORMAT: vk::Format = vk::Format::R32_SFLOAT;
/// How often the view is measured at the least: 20 times a second, so a
/// still picture looked at keeps being read. Something new to read is
/// measured at once: a new video frame, or the aim moved by
/// [`REMEASURE_RAD`]. The same frame read again at about the same aim says
/// nothing new, and would let one wrong reading count as two.
pub(crate) const MEASURE_EVERY: Duration = Duration::from_millis(50);
/// How far (radians, 1°: about what the eye tracker is off by) the aim must
/// move for what lies there to be measured before [`MEASURE_EVERY`].
pub(crate) const REMEASURE_RAD: f32 = 0.0175;
/// Ring patches that must agree, by their median, when the middle one has
/// nothing to match.
const MIN_RING_READINGS: usize = 3;

/// Disparity (radians per eye) at and below which no correction is applied
/// (~0.85 m with a 65 mm baseline).
const KAPPA_FAR: f32 = 0.038;
/// Softness of the onset of correction past [`KAPPA_FAR`]: correction grows
/// quadratically at first, then about linearly.
const KAPPA_KNEE: f32 = 0.020;
/// Minimum weighted luma standard deviation in $[0, 1]$ to trust a patch.
const MIN_PATCH_STD: f32 = 0.010;
/// Minimum similarity at the peak for a measurement.
const MIN_PEAK_SCORE: f32 = 0.6;
/// Minimum margin of the peak over the curve's mean, rejecting flat or
/// repetitive profiles.
const MIN_PROMINENCE: f32 = 0.05;
/// A near peak must beat the best far candidate by this much; otherwise the
/// patch is read as far.
const NEAR_MARGIN: f32 = 0.04;
/// Another peak of a patch's curve scoring within this of its best makes
/// the patch ambiguous: a pattern repeating along the baseline (a railing,
/// tiles, stripes) matches at more than one disparity, and the one that
/// happens to score higher may be a depth nothing is at. Such a patch gives
/// no reading, as one with nothing to match does.
const AMBIGUITY: f32 = 0.1;
/// Candidates either side of the best that belong to its own peak.
const PEAK_HALF_WIDTH: usize = 2;
/// How long (seconds) the correction stays with no reading at all. The eyes
/// may rest on a plain or dark spot of what is in focus for a moment, but
/// past this they may well be on something else, perhaps far, which the
/// correction would push past infinity.
const UNREAD_HOLD_SECS: f32 = 1.0;
/// How much the distance in focus must change, as a share, for the easing
/// to follow it: a fifth straight ahead, [`side_scale`] times as much to the
/// side (24% at 35°, 31% at 50°, 40% at 60°). Something 0.4 m away has
/// parts a few centimetres nearer and farther, which would otherwise be
/// eased by different amounts and make it grow and shrink as the eyes move
/// over it.
const RELIEF_STEP: f32 = 0.2;
/// Natural frequency (rad/s) of the critically damped spring easing the
/// correction in: 90% of a change in about a second, gently enough that
/// what is looked at doesn't visibly swell.
const EASE_IN_OMEGA: f32 = 3.9;
/// Fastest the correction may grow (radians per eye per second, ~3.4°/s).
const EASE_IN_MAX_SPEED: f32 = 0.06;
/// The same easing out: 90% of a change in ~0.45 s, about as long as the
/// eyes take to converge on something new. What they then look at is
/// farther, and pushed past infinity until it is done.
const EASE_OUT_OMEGA: f32 = 9.0;
/// Fastest the correction may shrink (radians per eye per second,
/// ~6.9°/s): the largest goes in 0.4 s.
const EASE_OUT_MAX_SPEED: f32 = 0.12;
/// How long (seconds) something farther must stay in focus before the
/// correction eases out for it: a glance there and back, as most take
/// less, leaves it be.
const EASE_OUT_HOLD_SECS: f32 = 0.25;
/// A blink this long (seconds) or shorter keeps the aim where the eyes were.
const BLINK_SECS: f32 = 0.4;
/// A gaze further than this (radians, 45°) from where the head points is
/// the tracker's error: the eyes look through the lenses within about that.
const PLAUSIBLE_GAZE_RAD: f32 = std::f32::consts::FRAC_PI_4;
/// The gaze smoothing's cutoff at rest, Hz: the One Euro filter (Casiez,
/// Roussel and Vogel, CHI 2012) takes out most of the tracker's jitter
/// while the eyes rest on something, and lets a jump of the eyes through
/// at once, by raising its cutoff with the gaze's speed. Lower takes out
/// more jitter but drifts behind the eyes following something slow.
const GAZE_MIN_CUTOFF_HZ: f32 = 1.0;
/// How much the cutoff rises with the gaze's speed, Hz per radian a
/// second: a 20° jump is followed to within a degree in three frames at
/// 90 Hz.
const GAZE_BETA: f32 = 3.0;
/// The cutoff of the speed the cutoff rises by, Hz: a jump's speed shows
/// in a frame or two, a resting gaze's jitter mostly doesn't.
const GAZE_SPEED_CUTOFF_HZ: f32 = 5.0;
/// A gap between gaze samples longer than this (seconds) starts the
/// smoothing afresh rather than smoothing across it.
const GAZE_GAP_SECS: f32 = 0.1;
/// The smoothed gaze further than this (radians, 2°) from the latest
/// sample is catching up with a jump of the eyes ([`AimFilter::settled`]);
/// a resting gaze's jitter keeps within it.
const SETTLED_RAD: f32 = 0.035;

/// The matching pass reading the samples the fragment pass drew.
const COMPUTE_READ: (vk::PipelineStageFlags2, vk::AccessFlags2) = (
    vk::PipelineStageFlags2::COMPUTE_SHADER,
    vk::AccessFlags2::SHADER_SAMPLED_READ,
);

/// The aim's angles about the cameras' baseline, `[along, tilt]`, in the
/// video's own frame with its Position corrections applied, where the
/// cameras sit on the x axis: `fs_focus` lays its lattice out in them
/// (`lattice` in `scene.wgsl`). A point and the two cameras span a plane
/// through that axis; `tilt` is the plane's turn about it and `along` the
/// angle within it from straight ahead towards +x. `None` too near the
/// baseline's axis to measure ([`MAX_ALONG`]).
pub(crate) fn aim_angles(scene: &SceneParams, aim: Vec3) -> Option<[f32; 2]> {
    let room = scene.anchor_inv.transform_vector3(aim);
    let d = scene.correction.transform_vector3(room).try_normalize()?;
    let along = d.x.clamp(-1.0, 1.0).asin();
    (along.abs() <= MAX_ALONG).then(|| [along, d.y.atan2(-d.z)])
}

struct SlotFocus {
    samples: Image,
    scores: Buffer,
    set: vk::DescriptorSet,
    /// The lattice's `along` angle where the measurement recorded in this
    /// slot aimed ([`aim_angles`]).
    along: f32,
    pending: bool,
}

pub(crate) struct FocusPass {
    pool: vk::DescriptorPool,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    slots: Vec<SlotFocus>,
}

impl FocusPass {
    pub fn new(gpu: &Gpu, slots: usize) -> Result<FocusPass> {
        let mut p = FocusPass {
            pool: vk::DescriptorPool::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            slots: Vec::new(),
        };
        if let Err(e) = p.create(gpu, slots) {
            p.destroy(gpu);
            return Err(e);
        }
        Ok(p)
    }

    fn create(&mut self, gpu: &Gpu, slots: usize) -> Result<()> {
        let d = &gpu.device;
        let types = [
            vk::DescriptorType::STORAGE_BUFFER,
            vk::DescriptorType::SAMPLED_IMAGE,
        ];
        let bindings = [0, 1].map(|b| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(types[b as usize])
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
        let n = slots as u32;
        let sizes = types.map(|ty| vk::DescriptorPoolSize {
            ty,
            descriptor_count: n,
        });
        // SAFETY: valid device; all created objects are cleaned up in `destroy`.
        unsafe {
            self.set_layout = d
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .ctx("focus descriptor set layout")?;
            let sets = [self.set_layout];
            self.layout = d
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&sets),
                    None,
                )
                .ctx("focus pipeline layout")?;
            let module = gpu.shader(crate::FOCUS_SPV)?;
            let info = vk::ComputePipelineCreateInfo::default()
                .stage(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(module)
                        .name(c"main"),
                )
                .layout(self.layout);
            let pipes = d.create_compute_pipelines(vk::PipelineCache::null(), &[info], None);
            d.destroy_shader_module(module, None);
            self.pipeline = pipes.map_err(|(_, result)| Error::Vk {
                context: "focus pipeline",
                result,
            })?[0];
            self.pool = d
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(n)
                        .pool_sizes(&sizes),
                    None,
                )
                .ctx("focus descriptor pool")?;
        }
        for _ in 0..slots {
            let slot = self.slot(gpu)?;
            self.slots.push(slot);
        }
        Ok(())
    }

    /// A slot's samples target, scores buffer and descriptors, which never
    /// change.
    fn slot(&self, gpu: &Gpu) -> Result<SlotFocus> {
        let mut samples = Image::new(
            gpu,
            2 * SAMPLE_COLS,
            SAMPLE_ROWS * PATCHES as u32,
            &[SAMPLES_FORMAT],
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            1,
            "focus samples",
        )?;
        let mut scores = match Buffer::new(
            gpu,
            (PATCHES * CANDIDATES * 8) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuToCpu,
            "focus scores",
        ) {
            Ok(b) => b,
            Err(e) => {
                samples.destroy(gpu);
                return Err(e);
            }
        };
        let d = &gpu.device;
        let layouts = [self.set_layout];
        // SAFETY: the pool has room for one set per slot.
        let set = unsafe {
            d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.pool)
                    .set_layouts(&layouts),
            )
        };
        let set = match set.ctx("focus descriptor set") {
            Ok(sets) => sets[0],
            Err(e) => {
                samples.destroy(gpu);
                scores.destroy(gpu);
                return Err(e);
            }
        };
        let buffers = [vk::DescriptorBufferInfo::default()
            .buffer(scores.buffer)
            .range(vk::WHOLE_SIZE)];
        let images = [vk::DescriptorImageInfo::default()
            .image_view(samples.level0)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffers),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&images),
        ];
        // SAFETY: the set is new and the buffer and view are valid.
        unsafe { d.update_descriptor_sets(&writes, &[]) };
        Ok(SlotFocus {
            samples,
            scores,
            set,
            along: 0.0,
            pending: false,
        })
    }

    /// The target `fs_focus` draws `slot`'s samples into, ready to draw.
    pub fn begin_samples(
        &mut self,
        gpu: &Gpu,
        cmd: vk::CommandBuffer,
        slot: usize,
    ) -> Option<(vk::ImageView, vk::Extent2D)> {
        let s = self.slots.get(slot)?;
        mem::barrier(
            gpu,
            cmd,
            s.samples.image,
            mem::range(0, 1),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            mem::TOP,
            mem::COLOR_WRITE,
        );
        Some((s.samples.level0, s.samples.extent))
    }

    /// Matches the samples drawn for `slot`, which aimed at lattice angle
    /// `along`, for reading back once the slot's fence has signalled.
    pub fn record(&mut self, gpu: &Gpu, cmd: vk::CommandBuffer, slot: usize, along: f32) {
        let Some(s) = self.slots.get_mut(slot) else {
            return;
        };
        mem::barrier(
            gpu,
            cmd,
            s.samples.image,
            mem::range(0, 1),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            mem::COLOR_WRITE,
            COMPUTE_READ,
        );
        let d = &gpu.device;
        // SAFETY: recording into the slot's command buffer; the pipeline,
        // set and buffer stay valid for its life.
        unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[s.set],
                &[],
            );
            d.cmd_dispatch(cmd, PATCHES as u32, 1, 1);
            let done = [vk::BufferMemoryBarrier2::default()
                .buffer(s.scores.buffer)
                .size(vk::WHOLE_SIZE)
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ)];
            d.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().buffer_memory_barriers(&done),
            );
        }
        s.along = along;
        s.pending = true;
    }

    /// Reads the per-patch scores (`PATCHES * CANDIDATES` `(score, std)`
    /// pairs) and the lattice angle `along` where they aimed, from `slot`
    /// once its fence has signalled.
    pub fn take(&mut self, slot: usize) -> Option<(Vec<[f32; 2]>, f32)> {
        let s = self.slots.get_mut(slot)?;
        if !std::mem::take(&mut s.pending) {
            return None;
        }
        let along = s.along;
        let bytes = s.scores.bytes();
        if bytes.len() < PATCHES * CANDIDATES * 8 {
            return None;
        }
        let out = bytes.as_chunks::<8>().0[..PATCHES * CANDIDATES]
            .iter()
            .map(|&[a0, a1, a2, a3, b0, b1, b2, b3]| {
                [
                    f32::from_le_bytes([a0, a1, a2, a3]),
                    f32::from_le_bytes([b0, b1, b2, b3]),
                ]
            })
            .collect();
        Some((out, along))
    }

    pub fn discard(&mut self) {
        for s in &mut self.slots {
            s.pending = false;
        }
    }

    pub fn destroy(&mut self, gpu: &Gpu) {
        for mut s in self.slots.drain(..) {
            s.samples.destroy(gpu);
            s.scores.destroy(gpu);
        }
        // SAFETY: renderer waits for all fences before destroying passes.
        unsafe {
            let d = &gpu.device;
            d.destroy_descriptor_pool(std::mem::take(&mut self.pool), None);
            d.destroy_pipeline(std::mem::take(&mut self.pipeline), None);
            d.destroy_pipeline_layout(std::mem::take(&mut self.layout), None);
            d.destroy_descriptor_set_layout(std::mem::take(&mut self.set_layout), None);
        }
    }
}

/// Maps (fractional) candidate index `idx` in `0..=63` to per-eye disparity:
/// half a lattice step apart, the first [`NEGATIVE`] below zero, the last
/// 0.103 rad (~0.32 m with a 65 mm baseline).
pub(crate) fn kappa_at(idx: f32) -> f32 {
    (idx.clamp(0.0, (CANDIDATES - 1) as f32) - NEGATIVE as f32) * 0.5 * STEP
}

/// Per-eye vergence relief (radians) for an object at per-eye disparity
/// `kappa`: exactly zero at or beyond [`KAPPA_FAR`], then growing smoothly
/// (zero slope at the threshold) and capped at [`MAX_COMFORT_CORR_RAD`].
pub(crate) fn correction_from_kappa(kappa: f32) -> f32 {
    let e = kappa - KAPPA_FAR;
    if e <= 0.0 || !e.is_finite() {
        return 0.0;
    }
    (e * e / (e + KAPPA_KNEE)).min(MAX_COMFORT_CORR_RAD)
}

/// Finds the sub-step per-eye disparity of one patch from its 64-candidate
/// `(score, std)` curve, or `None` when it lacks texture or a clear peak.
/// `far` is the disparity at and below which what is there needs no
/// relief: [`KAPPA_FAR`] straight ahead, less where the baseline is seen
/// at a slant ([`far_kappa`]).
pub(crate) fn estimate_kappa(scores: &[[f32; 2]; CANDIDATES], far: f32) -> Option<f32> {
    let mut valid_count = 0usize;
    let mut sum = 0.0f32;
    let mut best: Option<usize> = None;
    let mut best_far: Option<usize> = None;

    for (i, &[score, std_dev]) in scores.iter().enumerate() {
        if !score.is_finite() || !std_dev.is_finite() || std_dev < MIN_PATCH_STD {
            continue;
        }
        valid_count += 1;
        sum += score;
        if best.is_none_or(|b| score > scores[b][0]) {
            best = Some(i);
        }
        if kappa_at(i as f32) <= far && best_far.is_none_or(|b| score > scores[b][0]) {
            best_far = Some(i);
        }
    }

    if valid_count < CANDIDATES / 2 {
        return None;
    }
    let mut best_i = best?;
    let mean = sum / valid_count as f32;
    let peak = scores[best_i][0];
    if peak < MIN_PEAK_SCORE || peak - mean < MIN_PROMINENCE {
        return None;
    }
    // Another peak nearly as good: more than one depth fits.
    let valid = |i: usize| {
        let [score, std_dev] = scores[i];
        (score.is_finite() && std_dev.is_finite() && std_dev >= MIN_PATCH_STD).then_some(score)
    };
    let rival = (0..CANDIDATES)
        .filter(|&i| i.abs_diff(best_i) > PEAK_HALF_WIDTH)
        .filter_map(|i| {
            let s = valid(i)?;
            let below = i.checked_sub(1).and_then(valid);
            let above = (i + 1 < CANDIDATES).then(|| valid(i + 1)).flatten();
            (below.is_none_or(|b| s >= b) && above.is_none_or(|a| s >= a)).then_some(s)
        })
        .fold(f32::NEG_INFINITY, f32::max);
    if rival >= peak - AMBIGUITY {
        return None;
    }
    // Asymmetric decision: a near reading must clearly beat every far one.
    if kappa_at(best_i as f32) > far
        && let Some(f) = best_far
        && peak - scores[f][0] < NEAR_MARGIN
    {
        best_i = f;
    }

    // 3-point parabolic refinement around the peak.
    let mut refined = best_i as f32;
    if best_i > 0 && best_i + 1 < CANDIDATES {
        let y0 = scores[best_i - 1][0];
        let y1 = scores[best_i][0];
        let y2 = scores[best_i + 1][0];
        let denom = y0 - 2.0 * y1 + y2;
        if denom < -1e-5 {
            refined += (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5);
        }
    }
    Some(kappa_at(refined))
}

/// Lens spacing assumed for every video, metres: the average adult eye
/// spacing, which stereo rigs are built to match.
pub(crate) const LENS_SPACING: f32 = 0.063;
/// The distance in focus before the first reading: the farthest, where the
/// re-projection moves nothing, so the picture starts as filmed.
const START_DISTANCE: f32 = FARTHEST;
/// The nearest and farthest the distance in focus goes, metres.
const NEAREST: f32 = 0.3;
const FARTHEST: f32 = 50.0;
/// Time constant (seconds) of the re-projection's glide to a new distance,
/// in nearness: 90% of a change in ~0.7 s.
const DISTANCE_GLIDE_SECS: f32 = 0.3;
/// How far (radians, 0.15°) keeping the re-projection's distance may
/// misalign either eye's picture of what is looked at before it follows the
/// distance in focus: the eye's [`redraw_lever`] times the change in
/// nearness. With the head turned 45° that is a change of 11% at 0.4 m;
/// lying down, 2%; with the head level and facing the video's front, none
/// ever, as the re-projection changes nothing there.
const REDRAW_TOLERANCE_RAD: f32 = 0.0026;
/// Readings the first distance in focus is decided on: the newest and the
/// two before it.
const RECENT: usize = 3;
/// Two readings agree within this, in nearness (1/metres), straight ahead:
/// three times what the matcher resolves with a 63 mm baseline, under a
/// sixth of a degree of the eyes' convergence (1 m against 1.04 m, 3 m
/// against 3.4 m). To the side, [`side_scale`] times as much.
const AGREE: f32 = 0.04;
/// How far a reading agreeing with the distance in focus moves it towards
/// itself: an average over about the last two, steady against the
/// matcher's noise while following something moving slowly.
const REFINE: f32 = 0.5;

/// How many times less precise a reading at lattice angle `along`
/// ([`aim_angles`]) is than one straight ahead. To the side the cameras'
/// baseline is seen at a slant and spans `cos(along)` of its angle there,
/// so the same disparity step is `1 / cos(along)` as much nearness: 1.2 at
/// 35°, 1.6 at 50°, 2 at 60°.
pub(crate) fn side_scale(along: f32) -> f32 {
    1.0 / along.cos().max(MAX_ALONG.cos())
}

/// How far, across the aim (play space), the farther of the eyes is from
/// where its camera was, metres, with the eyes `offsets` (room frame) from
/// the point between them: a change of the re-projection's distance by `dn`
/// in nearness moves that eye's picture of what is looked at by about
/// `dn` times this, radians. 0 with the head level, facing the video's
/// front, and the eyes the lens spacing apart.
pub(crate) fn redraw_lever(scene: &SceneParams, aim: Vec3, offsets: [Vec3; 2]) -> f32 {
    let room = scene.anchor_inv.transform_vector3(aim);
    let Some(d) = scene.correction.transform_vector3(room).try_normalize() else {
        return 0.0;
    };
    [-1.0f32, 1.0]
        .into_iter()
        .zip(offsets)
        .map(|(side, o)| {
            let camera = Vec3::X * (side * 0.5 * LENS_SPACING);
            let off = scene.correction.transform_vector3(o) - camera;
            (off - off.dot(d) * d).length()
        })
        .fold(0.0, f32::max)
}

/// How near a point is, 1/metres from between the cameras, that the
/// cameras ([`LENS_SPACING`] apart) see `kappa` either side of lattice
/// angle `along` ([`aim_angles`]): the left one at `along + kappa`, the
/// right one at `along - kappa`, on the same row. In the triangle of the
/// cameras and the point, the angle at the point is `2 kappa`, opposite the
/// baseline, so the law of sines gives the sides: exact wherever the point
/// is, where `0.5 * LENS_SPACING / tan(kappa)` holds only straight ahead
/// (off to the side the baseline is seen at a slant, and reads too far by
/// 1/cos(along): 22% at 35°). 0 for a disparity at or below zero.
pub(crate) fn nearness(along: f32, kappa: f32) -> f32 {
    if !(kappa > 0.0 && along.is_finite()) {
        return 0.0;
    }
    // Each camera's angle from the baseline's +x end.
    let (left, right) = (FRAC_PI_2 - along - kappa, FRAC_PI_2 - along + kappa);
    let sin_right = right.sin();
    if sin_right < 1e-6 {
        return 0.0;
    }
    // 1 / the point's range from the left camera.
    let inv_left = (2.0 * kappa).sin() / (LENS_SPACING * sin_right);
    // Its range from between the cameras, along the left camera's ray.
    let b = LENS_SPACING * inv_left;
    let m = 1.0 - b * left.cos() + 0.25 * b * b;
    inv_left / m.max(1e-12).sqrt()
}

/// The disparity per eye, radians, below which something seen at lattice
/// angle `along` is farther than [`KAPPA_FAR`] means straight ahead: the
/// baseline spans `cos(along)` of its angle there, to first order.
pub(crate) fn far_kappa(along: f32) -> f32 {
    KAPPA_FAR * along.cos()
}

/// Per-eye vergence relief (radians) for something `nearness` (1/metres)
/// away: [`correction_from_kappa`] of its disparity straight ahead, which is
/// how far the eyes converge on it once the re-projection draws it at that
/// distance from where they are.
pub(crate) fn relief_at(nearness: f32) -> f32 {
    correction_from_kappa((0.5 * LENS_SPACING * nearness).atan())
}

/// How near (1/metres) what lies at the aim is: the middle patch's reading,
/// else the median of the ring round it, or `None` when neither can be
/// measured. `along` is the aim's lattice angle ([`aim_angles`]); each patch
/// is triangulated at its own.
pub(crate) fn aim_nearness(scores: &[[f32; 2]], along: f32) -> Option<f32> {
    let curves = scores.as_chunks::<CANDIDATES>().0;
    let read = |i: usize| {
        let at = along + ((i % GRID_W) as f32 - 1.0) * GRID_ALONG;
        let kappa = estimate_kappa(curves.get(i)?, far_kappa(at))?;
        Some(nearness(at, kappa))
    };
    if let Some(n) = read(AIM_PATCH) {
        return Some(n);
    }
    let mut ring: Vec<f32> = (0..curves.len())
        .filter(|&i| i != AIM_PATCH)
        .filter_map(read)
        .collect();
    if ring.len() < MIN_RING_READINGS {
        return None;
    }
    ring.sort_by(f32::total_cmp);
    // The lower middle of an even count: the farther one.
    Some(ring[(ring.len() - 1) / 2])
}

/// What comfort focus does to a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Focus {
    /// How far each eye's picture turns apart, radians.
    pub relief: f32,
    /// The radius of the camera spheres, metres: points at this distance
    /// line up for any head pose.
    pub distance: f32,
}

/// Correction tracker across frames.
#[derive(Default)]
pub(crate) struct ComfortFocus {
    /// Newest reading not yet folded in by [`ComfortFocus::step`]: how near
    /// what lay at its aim was, 1/metres (`None`: nothing to match there),
    /// and the lattice angle `along` there.
    pending: Option<(Option<f32>, f32)>,
    /// The latest [`RECENT`] readings, wherever the eyes were, newest last:
    /// how near (`None`: nothing to match there).
    recent: Vec<Option<f32>>,
    /// What the distance in focus glides to, as nearness, once two
    /// readings have agreed on it.
    accepted: Option<f32>,
    /// When the last reading with a disparity came.
    last_read: Option<Instant>,
    /// The distance in focus (nearness) `target` was set for, while it is.
    eased_for: Option<f32>,
    /// The correction (radians per eye) being eased towards.
    target: f32,
    /// A smaller correction decided on, the distance in focus (nearness) it
    /// is for, and since when: eased to once [`EASE_OUT_HOLD_SECS`] have
    /// passed with nothing nearer looked at again.
    lower: Option<(f32, f32, Instant)>,
    /// Current correction (radians per eye) and its rate of change.
    corr: f32,
    vel: f32,
    /// The re-projection's distance as nearness (1/metres), gliding to
    /// `near_target`, which follows the distance in focus within
    /// [`REDRAW_TOLERANCE_RAD`]; 0 until the first step.
    near: f32,
    near_target: f32,
    last_step_at: Option<Instant>,
}

impl ComfortFocus {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Takes the read-back patch scores of a completed frame and the lattice
    /// angle `along` where they aimed ([`aim_angles`]).
    pub fn ingest(&mut self, scores: &[[f32; 2]], along: f32) {
        self.pending = Some((aim_nearness(scores, along), along));
    }

    /// Advances to `now`, with the eyes' [`redraw_lever`] `lever`. Where the
    /// eyes look doesn't enter: the readings come from there, and the eyes
    /// roaming over what is in focus (or a blink) say nothing about how
    /// near it is.
    pub fn step(&mut self, now: Instant, lever: f32) -> Focus {
        let dt = self
            .last_step_at
            .map(|prev| now.saturating_duration_since(prev).as_secs_f32().min(0.1))
            .unwrap_or(1.0 / 90.0);
        self.last_step_at = Some(now);
        if self.near == 0.0 {
            self.near = 1.0 / START_DISTANCE;
            self.near_target = self.near;
        }

        match self.pending.take() {
            Some((Some(n), along)) => {
                self.last_read = Some(now);
                self.read(n.clamp(1.0 / FARTHEST, 1.0 / NEAREST), along, now);
            }
            // Nothing to match where it was read: older readings count for
            // less.
            Some((None, _)) if !self.recent.is_empty() => self.remember(None),
            _ => {}
        }
        // Something farther has stayed in focus: ease out for it.
        if let Some((want, a, since)) = self.lower
            && now.saturating_duration_since(since).as_secs_f32() >= EASE_OUT_HOLD_SECS
        {
            self.target = want;
            self.eased_for = Some(a);
            self.lower = None;
        }
        // Nothing readable for a while: let go of the easing. The distance in
        // focus stays, as the nearest guess there is.
        if self
            .last_read
            .is_none_or(|t| now.saturating_duration_since(t).as_secs_f32() > UNREAD_HOLD_SECS)
        {
            self.target = 0.0;
            self.eased_for = None;
            self.lower = None;
        }

        // Critically damped spring towards the target, speed-limited: gently
        // in, more promptly out.
        let (omega, max_speed) = if self.target > self.corr {
            (EASE_IN_OMEGA, EASE_IN_MAX_SPEED)
        } else {
            (EASE_OUT_OMEGA, EASE_OUT_MAX_SPEED)
        };
        let x = self.corr - self.target;
        let decay = (-omega * dt).exp();
        let tmp = (self.vel + omega * x) * dt;
        let next = self.target + (x + tmp) * decay;
        let max_step = max_speed * dt;
        let step = (next - self.corr).clamp(-max_step, max_step);
        self.vel = ((self.vel - omega * tmp) * decay).clamp(-max_speed, max_speed);
        let corr = self.corr + step;
        // A target moved mid-glide, towards where it was heading, never makes
        // it swing past and back.
        if (corr - self.target) * x < 0.0 {
            self.corr = self.target;
            self.vel = 0.0;
        } else {
            self.corr = corr;
        }
        self.corr = self.corr.clamp(0.0, MAX_COMFORT_CORR_RAD);
        if self.target == 0.0 && self.corr < 1e-5 {
            self.corr = 0.0;
            self.vel = 0.0;
        }
        // The re-projection follows the distance in focus once keeping its
        // own would misalign what is looked at.
        if let Some(a) = self.accepted
            && (a - self.near_target).abs() * lever > REDRAW_TOLERANCE_RAD
        {
            self.near_target = a;
        }
        self.near += (self.near_target - self.near) * (1.0 - (-dt / DISTANCE_GLIDE_SECS).exp());
        Focus {
            relief: self.corr,
            distance: 1.0 / self.near,
        }
    }

    fn remember(&mut self, reading: Option<f32>) {
        self.recent.push(reading);
        if self.recent.len() > RECENT {
            self.recent.remove(0);
        }
    }

    /// Folds in a reading of nearness `n` where the eyes are, at lattice
    /// angle `along`, where readings agree within [`AGREE`] times
    /// [`side_scale`]. The first two
    /// to agree set the distance in focus; one agreeing with it refines it.
    /// Another distance takes over as soon as the latest two readings agree
    /// on it, both apart from the one in focus: a lone wrong reading never
    /// moves it, and with the eyes on the edge of something near, readings
    /// going back and forth between it and what is behind keep it where it
    /// is rather than halfway between the two.
    fn read(&mut self, n: f32, along: f32, now: Instant) {
        let scale = side_scale(along);
        let agree = AGREE * scale;
        let before = self.recent.last().copied().flatten();
        self.remember(Some(n));
        self.accepted = match self.accepted {
            None => {
                let agreeing: Vec<f32> = self
                    .recent
                    .iter()
                    .flatten()
                    .copied()
                    .filter(|p| (p - n).abs() <= agree)
                    .collect();
                (agreeing.len() >= 2).then(|| agreeing.iter().sum::<f32>() / agreeing.len() as f32)
            }
            Some(a) if (n - a).abs() <= agree => Some(a + (n - a) * REFINE),
            Some(a) => Some(match before {
                Some(p) if (p - n).abs() <= agree && (p - a).abs() > agree => 0.5 * (p + n),
                _ => a,
            }),
        };
        let Some(a) = self.accepted else {
            return;
        };
        // Eased for what is in focus once this reading sees it, and only for
        // a change of distance of [`RELIEF_STEP`] or more, wider to the side
        // (or out to none): more at once, less once it has lasted
        // ([`EASE_OUT_HOLD_SECS`]).
        if (n - a).abs() > agree {
            return;
        }
        let step = 1.0 + RELIEF_STEP * scale;
        let moved = self.eased_for.is_none_or(|e| a > e * step || a * step < e);
        let want = relief_at(a);
        if !moved && !(want == 0.0 && self.target > 0.0) {
            // Back on what it is eased for: a glance away is over.
            self.lower = None;
        } else if want >= self.target {
            self.target = want;
            self.eased_for = Some(a);
            self.lower = None;
        } else {
            let since = self.lower.map_or(now, |(.., t)| t);
            self.lower = Some((want, a, since));
        }
    }
}

/// The head's turn in the play space from the axes the eye views give it:
/// `back` (+Z, opposite where it looks) and `up`, neither needing to be
/// unit or exactly square to the other.
pub(crate) fn head_turn(back: Vec3, up: Vec3) -> Quat {
    let z = back.normalize_or(Vec3::Z);
    let x = up.cross(z).normalize_or(Vec3::X);
    Quat::from_mat3(&Mat3::from_cols(x, z.cross(x), z)).normalize()
}

/// A direction in head axes (-Z ahead) as yaw and pitch, radians: yaw from
/// ahead towards +X (right), pitch up.
fn yaw_pitch(d: Vec3) -> Vec2 {
    Vec2::new(d.x.atan2(-d.z), d.y.clamp(-1.0, 1.0).asin())
}

/// The direction (head axes) of [`yaw_pitch`] `yp`.
fn from_yaw_pitch(yp: Vec2) -> Vec3 {
    let (sy, cy) = yp.x.sin_cos();
    let (sp, cp) = yp.y.sin_cos();
    Vec3::new(sy * cp, sp, -cy * cp)
}

/// A first-order low-pass's weight for the newest value at cutoff `hz`
/// over `dt` seconds.
fn low_pass(hz: f32, dt: f32) -> f32 {
    let tau = 1.0 / (std::f32::consts::TAU * hz);
    1.0 / (1.0 + tau / dt)
}

/// Picks the direction comfort focus measures in: the tracked gaze while it
/// looks believable, smoothed, held where it was for the length of a blink,
/// then the head's forward. The tracker's ray jitters by about a degree
/// while the eyes rest, which would move the measurement on and off the
/// edge of what is looked at; the smoothing is in head axes, where the
/// tracker's jitter is, so a gaze held through a blink turns with the head.
#[derive(Default)]
pub(crate) struct AimFilter {
    /// The smoothed gaze (head axes, as [`yaw_pitch`]), and its speed.
    smoothed: Option<Vec2>,
    speed: Vec2,
    /// When the last gaze sample was taken in, and where it was.
    last_sample: Option<Instant>,
    raw: Option<Vec2>,
}

impl AimFilter {
    /// The aim at `now` (play space) from the tracker's `gaze` (play space),
    /// when it has one, for a head turned `head`.
    pub fn aim(&mut self, now: Instant, gaze: Option<Vec3>, head: Quat) -> Vec3 {
        if let Some(g) = gaze.and_then(Vec3::try_normalize)
            && g.angle_between(head * Vec3::NEG_Z) <= PLAUSIBLE_GAZE_RAD
        {
            self.take(now, yaw_pitch(head.inverse() * g));
        }
        match (self.smoothed, self.last_sample) {
            (Some(yp), Some(t)) if now.saturating_duration_since(t).as_secs_f32() <= BLINK_SECS => {
                head * from_yaw_pitch(yp)
            }
            _ => head * Vec3::NEG_Z,
        }
    }

    /// Smooths in one sample, `yp` (head axes), by the One Euro filter.
    /// Whether the aim is where the eyes are, rather than catching up with
    /// a jump: in a jump it sweeps across whatever lies between, which is
    /// not worth measuring. Always, aiming with the head.
    pub fn settled(&self) -> bool {
        match (self.smoothed, self.raw) {
            (Some(s), Some(r)) => s.distance(r) <= SETTLED_RAD,
            _ => true,
        }
    }

    fn take(&mut self, now: Instant, yp: Vec2) {
        let dt = self
            .last_sample
            .map(|t| now.saturating_duration_since(t).as_secs_f32());
        self.last_sample = Some(now);
        self.raw = Some(yp);
        let Some((x, dt)) = self.smoothed.zip(dt).filter(|&(_, dt)| dt <= GAZE_GAP_SECS) else {
            self.smoothed = Some(yp);
            self.speed = Vec2::ZERO;
            return;
        };
        let dt = dt.max(1e-4);
        self.speed += ((yp - x) / dt - self.speed) * low_pass(GAZE_SPEED_CUTOFF_HZ, dt);
        let cutoff = GAZE_MIN_CUTOFF_HZ + GAZE_BETA * self.speed.length();
        self.smoothed = Some(x + (yp - x) * low_pass(cutoff, dt));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Typical stereo camera baseline in metres.
    const CAMERA_BASELINE: f32 = 0.065;

    fn kappa_for(distance: f32) -> f32 {
        (CAMERA_BASELINE / (2.0 * distance)).atan()
    }

    fn curve(distance: f32, std_dev: f32) -> [[f32; 2]; CANDIDATES] {
        let true_kappa = kappa_for(distance);
        let mut scores = [[0.0f32; 2]; CANDIDATES];
        for (i, slot) in scores.iter_mut().enumerate() {
            let diff = (kappa_at(i as f32) - true_kappa) / 0.012;
            *slot = [0.95 * (-0.5 * diff * diff).exp(), std_dev];
        }
        scores
    }

    /// Nothing to match: no texture.
    fn blank() -> [[f32; 2]; CANDIDATES] {
        curve(1.0, 0.001)
    }

    /// The cluster's curves: `aim` for the middle patch, `ring` round it.
    fn cluster(aim: [[f32; 2]; CANDIDATES], ring: [[f32; 2]; CANDIDATES]) -> Vec<[f32; 2]> {
        (0..PATCHES)
            .flat_map(|i| if i == AIM_PATCH { aim } else { ring })
            .collect()
    }

    fn at(distance: f32) -> Vec<[f32; 2]> {
        cluster(curve(distance, 0.08), curve(distance, 0.08))
    }

    /// The distance read for something `distance` metres straight ahead,
    /// filmed with a 65 mm baseline.
    fn read_as(distance: f32) -> f32 {
        1.0 / nearness(0.0, kappa_for(distance))
    }

    /// The correction for what lies at the aim of `scores`.
    fn aim_correction(scores: &[[f32; 2]]) -> Option<f32> {
        aim_nearness(scores, 0.0).map(relief_at)
    }

    const FRAME: Duration = Duration::from_millis(11);
    /// A [`redraw_lever`] so long the re-projection follows every change.
    const LEVER: f32 = 1.0;
    const AHEAD: Vec3 = Vec3::NEG_Z;

    /// `degrees` to the right of straight ahead.
    fn right_of_ahead(degrees: f32) -> Vec3 {
        Quat::from_rotation_y(-degrees.to_radians()) * AHEAD
    }

    /// Feeds `scores`, one reading a frame, for `frames` frames, returning
    /// every frame's focus.
    fn run_focus(
        cf: &mut ComfortFocus,
        t: &mut Instant,
        scores: &[[f32; 2]],
        frames: usize,
    ) -> Vec<Focus> {
        (0..frames)
            .map(|_| {
                *t += FRAME;
                cf.ingest(scores, 0.0);
                cf.step(*t, LEVER)
            })
            .collect()
    }

    /// The same, the easing only.
    fn run(cf: &mut ComfortFocus, t: &mut Instant, scores: &[[f32; 2]], frames: usize) -> Vec<f32> {
        run_focus(cf, t, scores, frames)
            .iter()
            .map(|f| f.relief)
            .collect()
    }

    #[test]
    fn parabolic_refinement_recovers_sub_step_disparity() {
        for &d in &[0.33f32, 0.45, 0.7, 1.6, 3.5] {
            let k = estimate_kappa(&curve(d, 0.08), KAPPA_FAR).expect("valid lock");
            let err = (k - kappa_for(d)).abs();
            assert!(err < 0.0005, "{d} m: kappa {k} vs {}", kappa_for(d));
        }
    }

    #[test]
    fn rejects_featureless_or_ambiguous_patches() {
        assert!(estimate_kappa(&blank(), KAPPA_FAR).is_none());
        assert!(estimate_kappa(&[[0.85, 0.05]; CANDIDATES], KAPPA_FAR).is_none());
    }

    #[test]
    fn near_peak_that_barely_beats_far_reads_as_far() {
        // Just nearer than the threshold: the far side of its own peak
        // scores within the margin.
        let k = estimate_kappa(&curve(0.83, 0.08), KAPPA_FAR).expect("valid");
        assert!(kappa_for(0.83) > KAPPA_FAR);
        assert_eq!(correction_from_kappa(k), 0.0);
    }

    #[test]
    fn a_pattern_matching_at_two_depths_gives_no_reading() {
        // A second peak nearly as good, near or far: a repeating pattern.
        let mut c = curve(0.4, 0.08);
        c[3][0] = 0.93;
        assert_eq!(estimate_kappa(&c, KAPPA_FAR), None);
        let mut c = curve(1.5, 0.08);
        let near = curve(0.45, 0.08);
        for (i, s) in c.iter_mut().enumerate() {
            s[0] = s[0].max(0.9 / 0.95 * near[i][0]);
        }
        assert_eq!(estimate_kappa(&c, KAPPA_FAR), None);
        // A clearly lower one is the pattern's texture: read as usual.
        let mut c = curve(1.5, 0.08);
        for (i, s) in c.iter_mut().enumerate() {
            s[0] = s[0].max(0.7 / 0.95 * near[i][0]);
        }
        let k = estimate_kappa(&c, KAPPA_FAR).expect("valid");
        assert!((k - kappa_for(1.5)).abs() < 0.0005, "{k}");
    }

    #[test]
    fn correction_is_zero_far_and_grows_only_for_near_objects() {
        for &d in &[0.86f32, 1.0, 2.0, 5.0, 50.0] {
            assert_eq!(correction_from_kappa(kappa_for(d)), 0.0, "{d} m");
        }
        let mut prev = 0.0;
        for &d in &[0.8f32, 0.6, 0.45, 0.35, 0.3] {
            let c = correction_from_kappa(kappa_for(d));
            assert!(c > prev, "{d} m: {c} <= {prev}");
            assert!(c < kappa_for(d) - KAPPA_FAR + 1e-6);
            prev = c;
        }
        assert!(correction_from_kappa(1.0) <= MAX_COMFORT_CORR_RAD);
    }

    #[test]
    fn triangulation_is_exact_wherever_the_point_is() {
        // The lattice angles of a direction (`aim_angles`).
        let angles = |p: Vec3| {
            let d = p.normalize();
            [d.x.asin(), d.y.atan2(-d.z)]
        };
        let half = Vec3::X * (0.5 * LENS_SPACING);
        for d in [0.3f32, 0.5, 1.0, 3.0] {
            for (along, tilt) in [
                (0.0f32, 0.0f32),
                (35.0, 0.0),
                (20.0, -20.0),
                (35.0, -35.0),
                (-50.0, 30.0),
                (60.0, -10.0),
            ] {
                let (a, t) = (along.to_radians(), tilt.to_radians());
                let p = Vec3::new(a.sin(), a.cos() * t.sin(), -a.cos() * t.cos()) * d;
                // The left camera is at -x, the right one at +x.
                let [l, l_tilt] = angles(p + half);
                let [r, r_tilt] = angles(p - half);
                assert!((l_tilt - r_tilt).abs() < 1e-5, "both on one row");
                let n = nearness((l + r) / 2.0, (l - r) / 2.0);
                assert!(
                    (n * d - 1.0).abs() < 1e-3,
                    "{d} m at {along}°, {tilt}° read as {} m",
                    1.0 / n
                );
            }
        }
        // The lens spacing assumed (63 mm) against the scene's (65 mm).
        for d in [0.5f32, 1.0, 3.0, 10.0] {
            let got = read_as(d);
            assert!(
                (got / d - 0.063 / 0.065).abs() < 0.01,
                "{d} m read as {got} m"
            );
        }
        assert_eq!(nearness(0.0, 0.0), 0.0);
        assert_eq!(nearness(0.0, -0.01), 0.0);
    }

    #[test]
    fn aims_too_near_the_baseline_axis_are_not_measured() {
        let video = crate::VideoParams {
            format: fp_core::format::VideoFormat::new(
                fp_core::format::Projection::EQUIRECT_180,
                fp_core::format::StereoLayout::SideBySide,
            ),
            ..Default::default()
        };
        let p = crate::params::build(&video, None, [glam::Mat4::IDENTITY; 2]);
        let down = Quat::from_rotation_x(-30f32.to_radians());
        let [along, tilt] = aim_angles(&p, down * right_of_ahead(35.0)).unwrap();
        assert!((along - 35f32.to_radians()).abs() < 1e-4, "{along}");
        assert!((tilt + 30f32.to_radians()).abs() < 1e-4, "{tilt}");
        assert!(aim_angles(&p, right_of_ahead(64.0)).is_some());
        assert!(aim_angles(&p, right_of_ahead(66.0)).is_none());
        assert!(aim_angles(&p, right_of_ahead(-66.0)).is_none());
    }

    #[test]
    fn what_lies_at_the_aim_sets_the_correction_whatever_surrounds_it() {
        let near = correction_from_kappa(kappa_for(0.4));
        // A close thing looked at against a far wall: eased in full.
        let c = aim_correction(&cluster(curve(0.4, 0.08), curve(3.0, 0.08))).unwrap();
        assert!((c - near).abs() < 1e-3, "{c} vs {near}");
        // The far wall looked at past close things: left alone.
        let c = aim_correction(&cluster(curve(3.0, 0.08), curve(0.4, 0.08))).unwrap();
        assert_eq!(c, 0.0);
    }

    #[test]
    fn with_nothing_to_match_at_the_aim_the_ring_decides() {
        let near = correction_from_kappa(kappa_for(0.4));
        let c = aim_correction(&cluster(blank(), curve(0.4, 0.08))).unwrap();
        assert!((c - near).abs() < 1e-3, "{c} vs {near}");
        // Too few of the ring readable: no reading.
        let mut scores = cluster(blank(), blank());
        scores[..2 * CANDIDATES].copy_from_slice(&[curve(0.4, 0.08), curve(0.4, 0.08)].concat());
        assert_eq!(aim_correction(&scores), None);
    }

    #[test]
    fn far_aims_are_left_untouched() {
        let mut cf = ComfortFocus::default();
        let mut t = Instant::now();
        for d in [1.0f32, 3.0, 1000.0] {
            let out = run(&mut cf, &mut t, &at(d), 60);
            assert!(out.iter().all(|&c| c == 0.0), "{d} m moved: {out:?}");
        }
    }

    #[test]
    fn eases_in_gently_and_lets_go_when_the_eyes_stay_on_something_farther() {
        let mut cf = ComfortFocus::default();
        let mut t = Instant::now();
        let want = correction_from_kappa(kappa_for(0.4));
        let out = run(&mut cf, &mut t, &at(0.4), 180);
        let max_step = EASE_IN_MAX_SPEED * FRAME.as_secs_f32() + 1e-6;
        let mut prev = 0.0;
        for &c in &out {
            assert!(c >= prev - 1e-6, "overshoot/oscillation: {out:?}");
            assert!(c - prev <= max_step, "too fast: {}", c - prev);
            prev = c;
        }
        // Under half way in 0.4 s, mostly there by 1.05 s, settled by 2 s.
        assert!(out[36] < 0.5 * want, "abrupt: {}", out[36]);
        assert!(out[95] > 0.9 * want, "slow: {}", out[95]);
        assert!((out[179] - want).abs() < 0.001, "{} vs {want}", out[179]);

        // The eyes move to the wall behind and stay: held a quarter of a
        // second, then within 0.1° of none by 0.9 s.
        let back = run(&mut cf, &mut t, &at(3.0), 82);
        assert!(
            back[..22].iter().all(|&c| (c - want).abs() < 0.001),
            "{back:?}"
        );
        assert!(*back.last().unwrap() < 0.002, "{back:?}");
    }

    #[test]
    fn a_glance_at_something_farther_and_back_leaves_it_be() {
        let mut cf = ComfortFocus::default();
        let mut t = Instant::now();
        let held = *run(&mut cf, &mut t, &at(0.4), 300).last().unwrap();
        let mut out = Vec::new();
        for _ in 0..4 {
            // The wall for a sixth of a second, then back.
            out.extend(run(&mut cf, &mut t, &at(3.0), 15));
            out.extend(run(&mut cf, &mut t, &at(0.4), 40));
        }
        assert!(out.iter().all(|&c| (c - held).abs() < 1e-4), "{out:?}");
    }

    #[test]
    fn a_change_of_mind_mid_glide_never_swings_past() {
        let mut cf = ComfortFocus::default();
        let mut t = Instant::now();
        run(&mut cf, &mut t, &at(0.4), 300);
        let (low, high) = (
            correction_from_kappa(kappa_for(0.4)),
            correction_from_kappa(kappa_for(0.3)),
        );
        // Something nearer for a fifth of a second, mid-glide, then the first
        // again for good.
        let mut out = run(&mut cf, &mut t, &at(0.3), 18);
        let back = run(&mut cf, &mut t, &at(0.4), 200);
        out.extend(&back);
        assert!(out.iter().all(|&c| c <= high + 1e-5), "{out:?}");
        assert!(back.iter().all(|&c| c >= low - 1e-5), "{back:?}");
        assert!((back.last().unwrap() - low).abs() < 1e-4, "{back:?}");
    }

    #[test]
    fn the_distance_in_focus_glides_to_what_is_looked_at_and_stays_without_a_reading() {
        let mut cf = ComfortFocus::default();
        let mut t = Instant::now();
        let first = run_focus(&mut cf, &mut t, &cluster(blank(), blank()), 1);
        assert_eq!(first[0].distance, START_DISTANCE);
        let want = read_as(0.4);
        let out = run_focus(&mut cf, &mut t, &at(0.4), 150);
        // In nearness, 90% of the way within 0.75 s, there by 1.65 s.
        let n = |d: f32| 1.0 / d;
        let from = n(START_DISTANCE);
        assert!(
            n(out[68].distance) - from > 0.9 * (n(want) - from),
            "{:?}",
            out[68]
        );
        assert!((out[149].distance - want).abs() < 0.01, "{:?}", out[149]);
        // Looking away at something unreadable keeps it, even once the
        // easing has let go.
        let away = run_focus(&mut cf, &mut t, &cluster(blank(), blank()), 180);
        assert!((away.last().unwrap().distance - want).abs() < 0.01);
        assert!(away.last().unwrap().relief < 1e-3, "{:?}", away.last());
        // A far wall moves it out.
        let wall = run_focus(&mut cf, &mut t, &at(3.0), 150);
        let far = read_as(3.0);
        assert!(
            (wall.last().unwrap().distance - far).abs() < 0.05 * far,
            "{:?}",
            wall.last()
        );
    }

    #[test]
    fn nothing_readable_holds_it_for_a_second_then_lets_go() {
        let mut cf = ComfortFocus::default();
        let mut t = Instant::now();
        let held = *run(&mut cf, &mut t, &at(0.4), 300).last().unwrap();
        assert!(held > 0.0);
        // A plain spot, nothing to match for most of a second.
        let blank = cluster(blank(), blank());
        let out = run(&mut cf, &mut t, &blank, 85);
        assert!(out.iter().all(|&c| (c - held).abs() < 1e-4), "{out:?}");
        // Longer: within 0.1° of none 0.6 s on.
        let out = run(&mut cf, &mut t, &blank, 60);
        assert!(*out.last().unwrap() < 0.002, "{out:?}");
    }

    #[test]
    fn the_eyes_roaming_over_what_is_in_focus_keep_it_still() {
        let mut cf = ComfortFocus::default();
        let mut t = Instant::now();
        let held = *run(&mut cf, &mut t, &at(0.4), 300).last().unwrap();
        // Over one thing: readings of it between plain spots with nothing to
        // match, for up to a third of a second at a time, for 5 s.
        let blank = cluster(blank(), blank());
        let mut out = Vec::new();
        for gap in [5, 30, 12, 25, 8, 30, 3, 20].repeat(3) {
            out.extend(run(&mut cf, &mut t, &blank, gap));
            out.extend(run(&mut cf, &mut t, &at(0.4), 2));
        }
        assert!(out.iter().all(|&c| (c - held).abs() < 1e-4), "{out:?}");
    }

    #[test]
    fn the_easing_follows_only_a_change_of_a_fifth_or_more() {
        let mut cf = ComfortFocus::default();
        let mut t = Instant::now();
        let held = *run(&mut cf, &mut t, &at(0.4), 300).last().unwrap();
        // The eyes over nearer and farther parts of one thing: the distance in
        // focus follows them, the easing doesn't.
        let mut out = Vec::new();
        for d in [0.37f32, 0.43, 0.38, 0.42, 0.4].repeat(2) {
            out.extend(run_focus(&mut cf, &mut t, &at(d), 60));
            let want = read_as(d);
            let got = out.last().unwrap().distance;
            assert!((got - want).abs() < 0.01, "{got} m vs {want} m");
        }
        assert!(
            out.iter().all(|f| (f.relief - held).abs() < 1e-4),
            "{out:?}"
        );
        // Something a quarter nearer: eased more.
        let nearer = *run(&mut cf, &mut t, &at(0.3), 180).last().unwrap();
        let want = correction_from_kappa(kappa_for(0.3));
        assert!((nearer - want).abs() < 0.001, "{nearer} vs {want}");
    }

    #[test]
    fn another_distance_takes_over_on_the_second_reading_of_it() {
        let mut cf = ComfortFocus::default();
        let mut t = Instant::now();
        run(&mut cf, &mut t, &at(0.4), 300);
        let near = cf.accepted.unwrap();
        run(&mut cf, &mut t, &at(3.0), 1);
        assert_eq!(cf.accepted, Some(near));
        run(&mut cf, &mut t, &at(3.0), 1);
        let far = 1.0 / read_as(3.0);
        assert!(
            (cf.accepted.unwrap() - far).abs() < 0.01,
            "{:?}",
            cf.accepted
        );
    }

    /// Feeds readings of nearness `n` at lattice angle `along` (radians),
    /// one a frame, for `frames` frames.
    fn feed(cf: &mut ComfortFocus, t: &mut Instant, n: f32, along: f32, frames: usize) {
        for _ in 0..frames {
            *t += FRAME;
            cf.pending = Some((Some(n), along));
            cf.step(*t, LEVER);
        }
    }

    #[test]
    fn to_the_side_where_readings_are_less_precise_it_takes_more_to_move_it() {
        let side = 50f32.to_radians();
        assert!((side_scale(side) - 1.556).abs() < 0.01);
        assert!((side_scale(0.0) - 1.0).abs() < 1e-6);
        assert!(side_scale(80f32.to_radians()) < 2.4);
        let near = 2.5;
        for along in [0.0, side] {
            let mut cf = ComfortFocus::default();
            let mut t = Instant::now();
            feed(&mut cf, &mut t, near, along, 300);
            let eased = cf.target;
            assert!(eased > 0.0);
            // Two readings 1.4 times [`AGREE`] nearer: another distance
            // straight ahead, the same one less precisely read to the side.
            let n = near + 1.4 * AGREE;
            feed(&mut cf, &mut t, n, along, 2);
            let a = cf.accepted.unwrap();
            if along == 0.0 {
                assert!((a - n).abs() < 1e-5, "{a} vs {n}");
            } else {
                assert!(a > near && n - a > 0.01, "{a} between {near} and {n}");
            }
            // A quarter nearer: eased for straight ahead, not to the side.
            feed(&mut cf, &mut t, 1.25 * near, along, 300);
            assert_eq!(cf.target > eased, along == 0.0, "{along}: {}", cf.target);
        }
    }

    #[test]
    fn the_re_projection_follows_only_changes_that_would_misalign_what_is_looked_at() {
        // The head turned 45° towards something 0.4 m away, and lying down.
        let turned = 0.5 * LENS_SPACING * (1.0 - std::f32::consts::FRAC_1_SQRT_2);
        let lying = 0.5 * LENS_SPACING * std::f32::consts::SQRT_2;
        for (lever, follows_part) in [(turned, false), (lying, true), (0.0, false)] {
            let mut cf = ComfortFocus::default();
            let mut t = Instant::now();
            let mut go = |n: f32, frames: usize| {
                for _ in 0..frames {
                    t += FRAME;
                    cf.pending = Some((Some(n), 0.0));
                    cf.step(t, lever);
                }
                cf.step(t, lever).distance
            };
            let first = go(2.5, 300);
            // A part of it 6% nearer.
            let part = go(2.5 * 1.06, 300);
            assert_eq!(
                (1.0 / part - 1.0 / first).abs() > 0.05,
                follows_part,
                "lever {lever}: {first} then {part}"
            );
            if lever > 0.0 {
                assert!((1.0 / first - 2.5).abs() < 0.01, "{first}");
                // Something a quarter nearer: followed either way.
                let nearer = go(2.5 * 1.25, 300);
                assert!((1.0 / nearer - 2.5 * 1.25).abs() < 0.05, "{nearer}");
            }
        }
    }

    #[test]
    fn small_changes_and_single_spikes_keep_it_still() {
        let mut cf = ComfortFocus::default();
        let mut t = Instant::now();
        let held = *run(&mut cf, &mut t, &at(0.4), 300).last().unwrap();
        for d in [0.41f32, 0.395, 0.405] {
            let out = run(&mut cf, &mut t, &at(d), 20);
            assert!(out.iter().all(|&c| (c - held).abs() < 1e-4), "{out:?}");
        }
        // A single far reading never moves it.
        let mut out = run(&mut cf, &mut t, &at(5.0), 1);
        out.extend(run(&mut cf, &mut t, &at(0.4), 20));
        assert!(out.iter().all(|&c| (c - held).abs() < 1e-4), "{out:?}");
    }

    #[test]
    fn on_the_edge_of_something_near_it_stays_where_it_is() {
        let mut cf = ComfortFocus::default();
        let mut t = Instant::now();
        let near = read_as(0.4);
        let held = *run_focus(&mut cf, &mut t, &at(0.4), 300).last().unwrap();
        assert!((held.distance - near).abs() < 0.01, "{held:?}");
        // The eyes on where it meets the wall behind: readings of both, by
        // turns.
        let mut out = Vec::new();
        for i in 0..60 {
            let d = if i % 2 == 0 { 3.0 } else { 0.4 };
            out.extend(run_focus(&mut cf, &mut t, &at(d), 1));
        }
        assert!(
            out.iter()
                .all(|f| (f.distance - near).abs() < 0.01 && (f.relief - held.relief).abs() < 1e-4),
            "{out:?}"
        );
        // Only the wall, read twice running: it goes there.
        let wall = run_focus(&mut cf, &mut t, &at(3.0), 150);
        let far = read_as(3.0);
        let last = wall.last().unwrap();
        assert!((last.distance - far).abs() < 0.05 * far, "{last:?}");
        assert_eq!(last.relief, 0.0);
    }

    #[test]
    fn the_aim_follows_the_eyes_through_a_blink_and_the_head_otherwise() {
        let mut f = AimFilter::default();
        let t = Instant::now();
        let ms = |n: u64| t + Duration::from_millis(n);
        let level = Quat::IDENTITY;
        let looked = right_of_ahead(15.0);
        assert!(f.aim(ms(0), Some(looked), level).abs_diff_eq(looked, 1e-6));
        // A blink keeps it where the eyes were, turning with the head.
        assert!(f.aim(ms(300), None, level).abs_diff_eq(looked, 1e-6));
        let turned = Quat::from_rotation_y(0.3);
        assert!(
            f.aim(ms(350), None, turned)
                .abs_diff_eq(turned * looked, 1e-6)
        );
        // A longer loss hands it to the head.
        assert!(f.aim(ms(500), None, level).abs_diff_eq(AHEAD, 1e-6));
        // A gaze 60° off the head is the tracker's error.
        let mut f = AimFilter::default();
        assert!(
            f.aim(ms(0), Some(right_of_ahead(60.0)), level)
                .abs_diff_eq(AHEAD, 1e-6)
        );
        // Without a tracker it is the head.
        assert!(f.aim(ms(0), None, turned).abs_diff_eq(turned * AHEAD, 1e-6));
    }

    #[test]
    fn the_head_turn_comes_from_the_eye_views_axes() {
        let turn = Quat::from_euler(glam::EulerRot::YXZ, 0.7, -0.3, 1.2);
        let got = head_turn(
            turn * Vec3::Z * 2.0,
            turn * Vec3::Y * 2.0 + turn * Vec3::Z * 0.01,
        );
        assert!(got.angle_between(turn) < 0.01, "{got:?} vs {turn:?}");
    }

    /// Repeatable normal noise (xorshift, Box-Muller).
    struct Noise(u64);

    impl Noise {
        fn uniform(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 40) as f32 + 0.5) / (1u64 << 24) as f32
        }

        fn normal(&mut self) -> f32 {
            let (a, b) = (self.uniform(), self.uniform());
            (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
        }
    }

    /// Runs gaze samples at `hz` for `seconds` with the head level, from
    /// `t`, `trace` giving each one's yaw and pitch in degrees; the aims'
    /// yaw and pitch in degrees.
    fn follow(
        f: &mut AimFilter,
        t: &mut Instant,
        hz: f32,
        seconds: f32,
        mut trace: impl FnMut() -> (f32, f32),
    ) -> Vec<Vec2> {
        let n = (seconds * hz).round() as usize;
        (0..n)
            .map(|_| {
                *t += Duration::from_secs_f32(1.0 / hz);
                let (y, p) = trace();
                let g = from_yaw_pitch(Vec2::new(y, p) * std::f32::consts::PI / 180.0);
                yaw_pitch(f.aim(*t, Some(g), Quat::IDENTITY)) * 180.0 / std::f32::consts::PI
            })
            .collect()
    }

    #[test]
    fn a_resting_gaze_comes_out_with_under_half_its_jitter() {
        for hz in [72.0f32, 90.0, 120.0] {
            let mut f = AimFilter::default();
            let mut t = Instant::now();
            follow(&mut f, &mut t, hz, 1.0, || (5.0, -3.0));
            let mut noise = Noise(7);
            let mut raw = Vec::new();
            let out = follow(&mut f, &mut t, hz, 4.0, || {
                let s = (5.0 + 0.5 * noise.normal(), -3.0 + 0.5 * noise.normal());
                raw.push(Vec2::new(s.0, s.1));
                s
            });
            let spread = |v: &[Vec2]| {
                let mean = v.iter().copied().sum::<Vec2>() / v.len() as f32;
                (v.iter().map(|s| (*s - mean).length_squared()).sum::<f32>() / v.len() as f32)
                    .sqrt()
            };
            let ratio = spread(&out) / spread(&raw);
            assert!(ratio < 0.5, "{hz} Hz: jitter x{ratio}");
        }
    }

    #[test]
    fn a_jump_of_the_eyes_is_followed_within_a_degree_in_three_frames() {
        for hz in [72.0f32, 90.0, 120.0] {
            let mut f = AimFilter::default();
            let mut t = Instant::now();
            follow(&mut f, &mut t, hz, 1.0, || (-10.0, 0.0));
            let out = follow(&mut f, &mut t, hz, 0.2, || (10.0, 0.0));
            // Within a degree by 33 ms: three frames at 72 and 90 Hz, four
            // at 120.
            let frames = (0.0333 * hz).ceil() as usize;
            let off = (out[frames - 1].x - 10.0).abs();
            assert!(off < 1.0, "{hz} Hz: {off}° off after {frames} frames");
            assert!(f.settled());
            // Unsettled while it catches up with the eyes.
            follow(&mut f, &mut t, hz, 1.0 / hz, || (-10.0, 0.0));
            assert!(!f.settled());
            follow(&mut f, &mut t, hz, frames as f32 / hz, || (-10.0, 0.0));
            assert!(f.settled());
        }
    }

    #[test]
    fn a_resting_gaze_stays_settled() {
        let mut f = AimFilter::default();
        let mut t = Instant::now();
        follow(&mut f, &mut t, 90.0, 1.0, || (5.0, -3.0));
        let mut noise = Noise(11);
        let mut unsettled = 0;
        for _ in 0..360 {
            follow(&mut f, &mut t, 90.0, 1.0 / 90.0, || {
                (5.0 + 0.5 * noise.normal(), -3.0 + 0.5 * noise.normal())
            });
            unsettled += usize::from(!f.settled());
        }
        assert!(unsettled < 4, "unsettled {unsettled} of 360 frames");
    }
}
