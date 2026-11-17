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
//! untouched, and the cyclopean picture never moves or scales.
//!
//! Measurements aim at the tracked gaze while foveated rendering uses it
//! ([`AimFilter`] bridges blinks), and at the head's forward otherwise.
//! Only what lies there is measured and eased: everything else keeps its
//! disparity, so a far background is pushed past infinity only while the
//! viewer looks at something close, and the correction glides back once they
//! look away, at about the pace the eyes re-converge.
//!
//! [`MEASURE_EVERY`], after the eye passes, `fs_focus` in `scene.wgsl` draws
//! strips of both eyes' luma round the aim into a small float target, with
//! the eye passes' own bindings, so every frame they can read is measured
//! (a 10-bit one in the decoder's buffer too). [`FocusPass`] then matches
//! them (`shaders/focus.wgsl`), one 64-thread workgroup per patch of a 3 x 3
//! cluster, scoring 64 candidate disparities per patch. [`ComfortFocus`]
//! reads the curves back once the slot's fence has signalled and takes the
//! middle patch's disparity, or the ring's median when the middle has
//! nothing to match (biased towards "far", so mistakes mean less correction
//! rather than a wrong one).

use crate::gpu::Gpu;
use crate::mem::{self, Buffer, Image};
use crate::params::{MAX_COMFORT_CORR_RAD, SceneParams};
use crate::{Error, Result, VkContext};
use ash::vk;
use glam::Vec3;
use gpu_allocator::MemoryLocation;
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
/// Each patch's strips in the samples target (`scene.wgsl`, `focus.wgsl`):
/// columns for each eye, and rows.
const SAMPLE_COLS: u32 = 53;
const SAMPLE_ROWS: u32 = 19;
/// The samples target's format: luma, and -1 off the picture.
pub(crate) const SAMPLES_FORMAT: vk::Format = vk::Format::R32_SFLOAT;
/// How often the view is measured: 20 times a second, so a reading's median
/// with the two before it spans a tenth of a second.
pub(crate) const MEASURE_EVERY: Duration = Duration::from_millis(50);
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
/// How long (seconds) to keep the last correction without a reading, while
/// the aim stays where it was read (a dark or plain spot of the same thing).
const MEASURE_HOLD_SECS: f32 = 0.5;
/// How far (radians, ~4°) the aim may move from its last reading before
/// that reading no longer says anything about what is looked at: no reading
/// there relaxes the correction, and a new one starts the history afresh.
const AIM_MOVED_RAD: f32 = 0.07;
/// Changes of the target correction smaller than this (radians per eye,
/// ~0.2°) are ignored, so measurement noise never shows.
const DEADBAND_RAD: f32 = 0.0035;
/// Natural frequency (rad/s) of the critically damped spring easing the
/// correction: 90% of a change in ~0.45 s, about as long as the eyes take
/// to converge on something new.
const SPRING_OMEGA: f32 = 9.0;
/// Fastest the correction may change (radians per eye per second, ~6.9°/s):
/// the largest correction goes in 0.4 s.
const MAX_SPEED_RAD_PER_SEC: f32 = 0.12;
/// A blink this long (seconds) or shorter keeps the aim where the eyes were.
const BLINK_SECS: f32 = 0.4;
/// A gaze further than this (radians, 45°) from where the head points is
/// the tracker's error: the eyes look through the lenses within about that.
const PLAUSIBLE_GAZE_RAD: f32 = std::f32::consts::FRAC_PI_4;

/// The matching pass reading the samples the fragment pass drew.
const COMPUTE_READ: (vk::PipelineStageFlags2, vk::AccessFlags2) = (
    vk::PipelineStageFlags2::COMPUTE_SHADER,
    vk::AccessFlags2::SHADER_SAMPLED_READ,
);

/// The aim's longitude and latitude in the video's own frame, its Position
/// corrections applied, where its cameras sit side by side: `fs_focus` lays
/// its lattice out there, so its rows run along the video's disparity.
pub(crate) fn aim_angles(scene: &SceneParams, aim: Vec3) -> Option<[f32; 2]> {
    let room = scene.anchor_inv.transform_vector3(aim);
    let d = scene.correction.transform_vector3(room).try_normalize()?;
    Some([d.x.atan2(-d.z), d.y.clamp(-1.0, 1.0).asin()])
}

struct SlotFocus {
    samples: Image,
    scores: Buffer,
    set: vk::DescriptorSet,
    /// Where the measurement recorded in this slot aimed, in the play space,
    /// and the lattice's latitude there.
    aim: Vec3,
    lat: f32,
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
            aim: Vec3::NEG_Z,
            lat: 0.0,
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

    /// Matches the samples drawn for `slot`, which aimed at `aim` (play
    /// space), for reading back once the slot's fence has signalled.
    pub fn record(&mut self, gpu: &Gpu, cmd: vk::CommandBuffer, slot: usize, aim: Vec3, lat: f32) {
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
        s.aim = aim;
        s.lat = lat;
        s.pending = true;
    }

    /// Reads the per-patch scores (`PATCHES * CANDIDATES` `(score, std)`
    /// pairs) and where they aimed from `slot` once its fence has signalled.
    pub fn take(&mut self, slot: usize) -> Option<(Vec<[f32; 2]>, Vec3, f32)> {
        let s = self.slots.get_mut(slot)?;
        if !std::mem::take(&mut s.pending) {
            return None;
        }
        let (aim, lat) = (s.aim, s.lat);
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
        Some((out, aim, lat))
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
pub(crate) fn estimate_kappa(scores: &[[f32; 2]; CANDIDATES]) -> Option<f32> {
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
        if kappa_at(i as f32) <= KAPPA_FAR && best_far.is_none_or(|b| score > scores[b][0]) {
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
    // Asymmetric decision: a near reading must clearly beat every far one.
    if kappa_at(best_i as f32) > KAPPA_FAR
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
/// Time constant (seconds) of the distance in focus's glide, in nearness:
/// 90% of a change in ~0.45 s, with the easing's spring, so the two arrive
/// together.
const DISTANCE_GLIDE_SECS: f32 = 0.2;

/// The per-eye disparity of what lies at the aim: the middle patch's, else
/// the median of the ring round it, or `None` when neither can be measured.
pub(crate) fn aim_kappa(scores: &[[f32; 2]]) -> Option<f32> {
    let curves = scores.as_chunks::<CANDIDATES>().0;
    if let Some(k) = curves.get(AIM_PATCH).and_then(estimate_kappa) {
        return Some(k);
    }
    let mut ring: Vec<f32> = curves
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != AIM_PATCH)
        .filter_map(|(_, curve)| estimate_kappa(curve))
        .collect();
    if ring.len() < MIN_RING_READINGS {
        return None;
    }
    ring.sort_by(f32::total_cmp);
    // The lower middle of an even count: the farther one.
    Some(ring[(ring.len() - 1) / 2])
}

/// How far from the cameras something at per-eye disparity `kappa` is,
/// metres, within [`NEAREST`] and [`FARTHEST`].
pub(crate) fn distance_from_kappa(kappa: f32) -> f32 {
    if kappa <= 0.0 || !kappa.is_finite() {
        return FARTHEST;
    }
    (0.5 * LENS_SPACING / kappa.tan()).clamp(NEAREST, FARTHEST)
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
    /// Newest reading not yet folded in by [`ComfortFocus::step`]: the
    /// disparity of what lay at its aim (`None`: nothing to match there),
    /// and that aim.
    pending: Option<(Option<f32>, Vec3)>,
    recent: [f32; 3],
    recent_len: usize,
    /// Where the last reading with a disparity aimed, and when it came.
    last_read: Option<(Vec3, Instant)>,
    /// Committed target correction (radians per eye) after the deadband.
    target: f32,
    /// Current correction (radians per eye) and its rate of change.
    corr: f32,
    vel: f32,
    /// The distance in focus as nearness (1/metres), gliding to
    /// `near_target`; 0 until the first step.
    near: f32,
    near_target: f32,
    last_step_at: Option<Instant>,
}

impl ComfortFocus {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Takes the read-back patch scores of a completed frame, where they
    /// aimed (play space) and the lattice's latitude there: its steps of
    /// longitude span `cos(lat)` of their angle.
    pub fn ingest(&mut self, scores: &[[f32; 2]], aim: Vec3, lat: f32) {
        self.pending = Some((aim_kappa(scores).map(|k| k * lat.cos()), aim));
    }

    /// Advances to `now` with the eyes on `aim` (play space).
    pub fn step(&mut self, now: Instant, aim: Vec3) -> Focus {
        let dt = self
            .last_step_at
            .map(|prev| now.saturating_duration_since(prev).as_secs_f32().min(0.1))
            .unwrap_or(1.0 / 90.0);
        self.last_step_at = Some(now);
        if self.near == 0.0 {
            self.near = 1.0 / START_DISTANCE;
            self.near_target = self.near;
        }
        // Whether a reading at `at` from `t` still speaks for the eyes at `to`.
        let current = |(at, t): (Vec3, Instant), to: Vec3| {
            at.angle_between(to) <= AIM_MOVED_RAD
                && now.saturating_duration_since(t).as_secs_f32() <= MEASURE_HOLD_SECS
        };

        if let Some((Some(k), at)) = self.pending.take() {
            // A reading of something else says nothing about this one.
            if self.last_read.is_none_or(|last| !current(last, at)) {
                self.recent_len = 0;
            }
            self.last_read = Some((at, now));
            self.recent.rotate_left(1);
            self.recent[2] = k;
            self.recent_len = (self.recent_len + 1).min(3);
            // A lone first reading waits for a second; two take the farther
            // (conservative); three take the median, rejecting spikes.
            let filtered = match self.recent_len {
                1 => None,
                2 => Some(self.recent[1].min(self.recent[2])),
                _ => Some(median3(self.recent[0], self.recent[1], self.recent[2])),
            };
            if let Some(k) = filtered {
                let want = correction_from_kappa(k);
                if want == 0.0 || (want - self.target).abs() > DEADBAND_RAD {
                    self.target = want;
                }
                self.near_target = 1.0 / distance_from_kappa(k);
            }
        }
        // Nothing read lately where the eyes are: let go of the easing. The
        // distance in focus stays, as the nearest guess there is.
        if self.last_read.is_none_or(|last| !current(last, aim)) {
            self.target = 0.0;
        }

        // Critically damped spring towards the target, speed-limited.
        let x = self.corr - self.target;
        let decay = (-SPRING_OMEGA * dt).exp();
        let tmp = (self.vel + SPRING_OMEGA * x) * dt;
        let next = self.target + (x + tmp) * decay;
        let max_step = MAX_SPEED_RAD_PER_SEC * dt;
        let step = (next - self.corr).clamp(-max_step, max_step);
        self.vel = ((self.vel - SPRING_OMEGA * tmp) * decay)
            .clamp(-MAX_SPEED_RAD_PER_SEC, MAX_SPEED_RAD_PER_SEC);
        self.corr = (self.corr + step).clamp(0.0, MAX_COMFORT_CORR_RAD);
        if self.target == 0.0 && self.corr < 1e-5 {
            self.corr = 0.0;
            self.vel = 0.0;
        }
        self.near += (self.near_target - self.near) * (1.0 - (-dt / DISTANCE_GLIDE_SECS).exp());
        Focus {
            relief: self.corr,
            distance: 1.0 / self.near,
        }
    }
}

fn median3(a: f32, b: f32, c: f32) -> f32 {
    a.max(b).min(c.max(a.min(b)))
}

/// Picks the direction comfort focus measures in: the tracked gaze while it
/// looks believable, the last gaze for the length of a blink, then the
/// head's forward.
#[derive(Default)]
pub(crate) struct AimFilter {
    last_gaze: Option<(Vec3, Instant)>,
}

impl AimFilter {
    /// The aim at `now` (play space) from the tracker's `gaze`, when it has
    /// one, and the head's forward `head`.
    pub fn aim(&mut self, now: Instant, gaze: Option<Vec3>, head: Vec3) -> Vec3 {
        let head = head.normalize_or(Vec3::NEG_Z);
        if let Some(g) = gaze.and_then(Vec3::try_normalize)
            && g.angle_between(head) <= PLAUSIBLE_GAZE_RAD
        {
            self.last_gaze = Some((g, now));
            return g;
        }
        match self.last_gaze {
            Some((g, t)) if now.saturating_duration_since(t).as_secs_f32() <= BLINK_SECS => g,
            _ => head,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;
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

    /// The correction for what lies at the aim of `scores`.
    fn aim_correction(scores: &[[f32; 2]]) -> Option<f32> {
        aim_kappa(scores).map(correction_from_kappa)
    }

    const FRAME: Duration = Duration::from_millis(11);
    const AHEAD: Vec3 = Vec3::NEG_Z;

    /// `degrees` to the right of straight ahead.
    fn right_of_ahead(degrees: f32) -> Vec3 {
        Quat::from_rotation_y(-degrees.to_radians()) * AHEAD
    }

    /// Feeds `scores` measured at `aim`, with the eyes there, for `frames`
    /// frames, returning every frame's focus.
    fn run_focus(
        cf: &mut ComfortFocus,
        t: &mut Instant,
        scores: &[[f32; 2]],
        aim: Vec3,
        frames: usize,
    ) -> Vec<Focus> {
        (0..frames)
            .map(|_| {
                *t += FRAME;
                cf.ingest(scores, aim, 0.0);
                cf.step(*t, aim)
            })
            .collect()
    }

    /// The same, the easing only.
    fn run(
        cf: &mut ComfortFocus,
        t: &mut Instant,
        scores: &[[f32; 2]],
        aim: Vec3,
        frames: usize,
    ) -> Vec<f32> {
        run_focus(cf, t, scores, aim, frames)
            .iter()
            .map(|f| f.relief)
            .collect()
    }

    #[test]
    fn parabolic_refinement_recovers_sub_step_disparity() {
        for &d in &[0.33f32, 0.45, 0.7, 1.6, 3.5] {
            let k = estimate_kappa(&curve(d, 0.08)).expect("valid lock");
            let err = (k - kappa_for(d)).abs();
            assert!(err < 0.0005, "{d} m: kappa {k} vs {}", kappa_for(d));
        }
    }

    #[test]
    fn rejects_featureless_or_ambiguous_patches() {
        assert!(estimate_kappa(&blank()).is_none());
        assert!(estimate_kappa(&[[0.85, 0.05]; CANDIDATES]).is_none());
    }

    #[test]
    fn near_peak_that_barely_beats_far_reads_as_far() {
        let mut c = curve(0.4, 0.08);
        c[3][0] = 0.93;
        let k = estimate_kappa(&c).expect("valid");
        assert_eq!(correction_from_kappa(k), 0.0);
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
    fn distance_comes_from_disparity_within_its_limits() {
        // The lens spacing assumed (63 mm) against the scene's (65 mm).
        for d in [0.5f32, 1.0, 3.0, 10.0] {
            let got = distance_from_kappa(kappa_for(d));
            assert!(
                (got / d - 0.063 / 0.065).abs() < 0.01,
                "{d} m read as {got} m"
            );
        }
        assert_eq!(distance_from_kappa(0.0), FARTHEST);
        assert_eq!(distance_from_kappa(-0.01), FARTHEST);
        assert_eq!(distance_from_kappa(1.0), NEAREST);
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
            let out = run(&mut cf, &mut t, &at(d), AHEAD, 60);
            assert!(out.iter().all(|&c| c == 0.0), "{d} m moved: {out:?}");
        }
    }

    #[test]
    fn eases_towards_what_is_looked_at_as_fast_as_the_eyes_turn_and_lets_go_when_they_move_on() {
        let mut cf = ComfortFocus::default();
        let mut t = Instant::now();
        let want = correction_from_kappa(kappa_for(0.4));
        let out = run(&mut cf, &mut t, &at(0.4), AHEAD, 120);
        let max_step = MAX_SPEED_RAD_PER_SEC * FRAME.as_secs_f32() + 1e-6;
        let mut prev = 0.0;
        for &c in &out {
            assert!(c >= prev - 1e-6, "overshoot/oscillation: {out:?}");
            assert!(c - prev <= max_step, "too fast: {}", c - prev);
            prev = c;
        }
        // Mostly there within 0.6 s, settled by 1.3 s.
        assert!(out[55] > 0.9 * want, "slow: {}", out[55]);
        assert!((out[119] - want).abs() < 0.001, "{} vs {want}", out[119]);

        // The eyes move 10° to the wall behind: within 0.1° of none in 0.6 s.
        let back = run(&mut cf, &mut t, &at(3.0), right_of_ahead(10.0), 55);
        assert!(*back.last().unwrap() < 0.002, "{back:?}");
    }

    #[test]
    fn the_distance_in_focus_glides_to_what_is_looked_at_and_stays_without_a_reading() {
        let mut cf = ComfortFocus::default();
        let mut t = Instant::now();
        let first = run_focus(&mut cf, &mut t, &cluster(blank(), blank()), AHEAD, 1);
        assert_eq!(first[0].distance, START_DISTANCE);
        let want = distance_from_kappa(kappa_for(0.4));
        let out = run_focus(&mut cf, &mut t, &at(0.4), AHEAD, 100);
        // In nearness, 90% of the way within 0.65 s, there by 1.1 s.
        let n = |d: f32| 1.0 / d;
        let from = n(START_DISTANCE);
        assert!(
            n(out[58].distance) - from > 0.9 * (n(want) - from),
            "{:?}",
            out[58]
        );
        assert!((out[99].distance - want).abs() < 0.01, "{:?}", out[99]);
        // Looking away at something unreadable keeps it.
        let away = run_focus(
            &mut cf,
            &mut t,
            &cluster(blank(), blank()),
            right_of_ahead(20.0),
            60,
        );
        assert!((away.last().unwrap().distance - want).abs() < 0.01);
        // A far wall moves it out.
        let wall = run_focus(&mut cf, &mut t, &at(3.0), right_of_ahead(20.0), 150);
        let far = distance_from_kappa(kappa_for(3.0));
        assert!(
            (wall.last().unwrap().distance - far).abs() < 0.05 * far,
            "{:?}",
            wall.last()
        );
    }

    #[test]
    fn looking_away_at_something_unreadable_lets_go() {
        let mut cf = ComfortFocus::default();
        let mut t = Instant::now();
        let held = *run(&mut cf, &mut t, &at(0.4), AHEAD, 150).last().unwrap();
        assert!(held > 0.0);
        let out = run(
            &mut cf,
            &mut t,
            &cluster(blank(), blank()),
            right_of_ahead(10.0),
            55,
        );
        assert!(*out.last().unwrap() < 0.002, "{out:?}");
    }

    #[test]
    fn a_plain_spot_of_what_is_looked_at_holds_briefly() {
        let mut cf = ComfortFocus::default();
        let mut t = Instant::now();
        let held = *run(&mut cf, &mut t, &at(0.4), AHEAD, 150).last().unwrap();
        // A little to the side, still on it, nothing to match for 0.3 s.
        let blank = cluster(blank(), blank());
        let out = run(&mut cf, &mut t, &blank, right_of_ahead(2.0), 27);
        assert!(out.iter().all(|&c| (c - held).abs() < 1e-4), "{out:?}");
        // Longer: it eases off.
        let out = run(&mut cf, &mut t, &blank, right_of_ahead(2.0), 40);
        assert!(*out.last().unwrap() < held);
    }

    #[test]
    fn small_changes_and_single_spikes_keep_it_still() {
        let mut cf = ComfortFocus::default();
        let mut t = Instant::now();
        let held = *run(&mut cf, &mut t, &at(0.4), AHEAD, 150).last().unwrap();
        for d in [0.41f32, 0.395, 0.405] {
            let out = run(&mut cf, &mut t, &at(d), AHEAD, 20);
            assert!(out.iter().all(|&c| (c - held).abs() < 1e-4), "{out:?}");
        }
        // A single-frame far reading is rejected by the median.
        let mut out = run(&mut cf, &mut t, &at(5.0), AHEAD, 1);
        out.extend(run(&mut cf, &mut t, &at(0.4), AHEAD, 20));
        assert!(out.iter().all(|&c| (c - held).abs() < 1e-4), "{out:?}");
    }

    #[test]
    fn the_aim_follows_the_eyes_through_a_blink_and_the_head_otherwise() {
        let mut f = AimFilter::default();
        let t = Instant::now();
        let ms = |n: u64| t + Duration::from_millis(n);
        let looked = right_of_ahead(15.0);
        assert!(f.aim(ms(0), Some(looked), AHEAD).abs_diff_eq(looked, 1e-6));
        // A blink keeps it where the eyes were.
        assert!(f.aim(ms(300), None, AHEAD).abs_diff_eq(looked, 1e-6));
        // A longer loss hands it to the head.
        assert!(f.aim(ms(500), None, AHEAD).abs_diff_eq(AHEAD, 1e-6));
        // A gaze 60° off the head is the tracker's error.
        let mut f = AimFilter::default();
        assert!(
            f.aim(ms(0), Some(right_of_ahead(60.0)), AHEAD)
                .abs_diff_eq(AHEAD, 1e-6)
        );
        // Without a tracker it is the head.
        assert!(f.aim(ms(0), None, AHEAD * 2.0).abs_diff_eq(AHEAD, 1e-6));
    }
}
