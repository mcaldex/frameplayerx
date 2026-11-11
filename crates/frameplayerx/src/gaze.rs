//! A gaze steady enough to aim with. The eye tracker gives one ray a
//! frame: it jitters while the eyes rest, vanishes in every blink, and can
//! come stale or wildly off. Once a frame [`GazeFilter::update`] turns it
//! into an aim in head axes (so one being held or gliding turns with the
//! head): plausible, fresh samples smoothed by the One Euro filter (Casiez,
//! Roussel and Vogel, CHI 2012), held through a blink, the head's forward
//! after a longer loss, and blended back to when the eyes return. Pure:
//! the times are inputs.

use glam::{Quat, Vec2, Vec3};

/// The One Euro filter's cutoff at rest, Hz: 1. Tuned as the paper says,
/// with `BETA` at 0 first, on synthetic fixations of 0.5° of noise at 72,
/// 90 and 120 Hz: lower takes out more jitter but drifts behind slow
/// pursuit; higher leaves the jitter in.
const MIN_CUTOFF: f32 = 1.0;
/// How much the cutoff rises with the gaze's speed, Hz per radian a
/// second: then raised until a 20° saccade is caught within a degree in
/// three frames at 90 Hz. Higher catches saccades sooner but lets the
/// noise's own speed through at rest.
const BETA: f32 = 3.0;
/// The cutoff of the speed the cutoff is raised by, Hz: 5, so a saccade's
/// speed shows in a frame or two while a fixation's noise mostly doesn't.
const D_CUTOFF: f32 = 5.0;

/// A sample further than this from the head's forward is implausible,
/// radians: 45°. Behind a headset's lenses the eyes turn up to about 35°
/// and look through the lens out to about 45°; past that a sample is the
/// tracker's error.
const PLAUSIBLE: f32 = 45.0 * std::f32::consts::PI / 180.0;
/// A sample older than this, seconds, is stale: 0.25. On the Frame the
/// first sample came 79 ms before the predicted display time, so a limit
/// near 0.1 s would throw good ones away; the log gives the ages seen, to
/// tune it by.
const STALE: f32 = 0.25;
/// A gap between samples longer than this, seconds, starts the filter
/// afresh rather than smoothing across it: 0.1, nine frames at 90 Hz.
const GAP: f64 = 0.1;
/// A loss up to this long is a blink, seconds: 0.4, the longest a blink
/// takes. Through it the aim holds where it was.
const BLINK: f64 = 0.4;
/// No sample for this long, or none since the start, is a tracker not
/// delivering (not allowed for apps, switched off, the eyes not found),
/// seconds: 3, longer than looking away or closing the eyes for a moment.
const IDLE: f64 = 3.0;
/// After a longer loss the aim glides to the head's forward with this
/// time constant, seconds: 0.2, so it is within a degree of it 0.6 s on
/// from 15° away: about as fast as a look led by the head.
const TO_HEAD: f32 = 0.2;
/// When the eyes come back after a longer loss, how long they must have
/// been back before the aim follows them, seconds: 0.1, so a flicker of
/// samples while the tracker finds the eyes doesn't pull it.
const REACQUIRE: f64 = 0.1;
/// And how long it blends from the head's forward to the gaze, seconds:
/// 0.3, so it doesn't jump.
const BLEND: f64 = 0.3;
/// While the tracker hands the same sample back (one slower than the
/// display does, for a frame or two), the gaze stays current this long
/// after its last new sample, seconds: 0.05. A frame with no sample at
/// all ends it at once.
const CURRENT: f64 = 0.05;

/// The gaze dot's ring's angular radius, radians: 1.25°, about the
/// tracker's error left after a gaze check (and a third of what it was
/// before one), so the viewer sees how close the gaze can be.
pub const DOT_RING: f32 = 1.25 * std::f32::consts::PI / 180.0;

/// How far away the gaze dot is when nothing sets it, metres: 2, as far
/// as the panels and where the eyes rest easily.
pub const DOT_DISTANCE: f32 = 2.0;

/// Where the gaze dot goes for a head at `head` (position, orientation,
/// world) looking along `gaze` (head axes): centred on the gaze ray
/// `distance` metres out, facing back along it, its own up the head's,
/// and how wide a square it is, metres, for its picture's half-width to
/// span `half_angle` radians whatever the distance.
pub fn dot_place(
    head: (Vec3, Quat),
    gaze: Vec3,
    distance: f32,
    half_angle: f32,
) -> (glam::Mat4, f32) {
    let (at, turn) = head;
    let along = turn * Quat::from_rotation_arc(Vec3::NEG_Z, gaze.normalize_or(Vec3::NEG_Z));
    let centre = at + along * Vec3::NEG_Z * distance;
    (
        glam::Mat4::from_rotation_translation(along, centre),
        2.0 * distance * half_angle.tan(),
    )
}

/// One gaze reading from the headset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reading {
    /// Where the eyes look, in the world (unit).
    pub direction: Vec3,
    /// When the eyes were sampled, seconds on the runtime's clock, if it
    /// says.
    pub time: Option<f64>,
    /// How long before the frame is shown they were sampled, seconds.
    pub age: Option<f32>,
}

/// What aims, in words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GazeStatus {
    /// The eyes.
    Eyes,
    /// Holding where the eyes were, through a blink.
    Blink,
    /// The gaze was lost a while: the head.
    Lost,
    /// The tracker isn't delivering: the head.
    NotDelivering,
    /// The headset has no eye tracking: the head.
    NoEyeTracking,
}

/// The aim this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aim {
    /// Where to aim, head axes (-Z the head's forward).
    pub direction: Vec3,
    /// A measurement may start along it: not while held through a blink.
    pub usable: bool,
    /// It is the eyes' (not the head's forward, nor gliding between).
    pub from_eyes: bool,
    pub status: GazeStatus,
    /// The latest smoothed gaze, head axes, and whether it is current.
    pub gaze: Option<Vec3>,
    pub current: bool,
}

/// A correction of the tracker's error for this viewer, from a gaze check:
/// `gaze' = gain * gaze + offset` on (yaw, pitch) in radians.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Correction {
    pub offset: [f32; 2],
    /// Rows: yaw, pitch.
    pub gain: [[f32; 2]; 2],
}

impl Default for Correction {
    fn default() -> Correction {
        Correction {
            offset: [0.0; 2],
            gain: [[1.0, 0.0], [0.0, 1.0]],
        }
    }
}

impl Correction {
    pub fn apply(&self, yp: Vec2) -> Vec2 {
        let g = self.gain;
        Vec2::new(
            g[0][0] * yp.x + g[0][1] * yp.y + self.offset[0],
            g[1][0] * yp.x + g[1][1] * yp.y + self.offset[1],
        )
    }
}

/// A direction (head axes) as yaw and pitch, radians: yaw from -Z towards
/// +X (right), pitch up.
pub fn yaw_pitch(d: Vec3) -> Vec2 {
    let d = d.normalize_or(Vec3::NEG_Z);
    Vec2::new(d.x.atan2(-d.z), d.y.clamp(-1.0, 1.0).asin())
}

/// The direction of `yaw_pitch`.
pub fn direction(yp: Vec2) -> Vec3 {
    let (sy, cy) = yp.x.sin_cos();
    let (sp, cp) = yp.y.sin_cos();
    Vec3::new(sy * cp, sp, -cy * cp)
}

/// A first-order low-pass's weight for the new value at cutoff `hz` over
/// `dt` seconds.
fn alpha(hz: f32, dt: f32) -> f32 {
    let tau = 1.0 / (2.0 * std::f32::consts::PI * hz);
    1.0 / (1.0 + tau / dt)
}

/// The One Euro filter on a 2D value.
#[derive(Clone, Copy, Debug, Default)]
struct OneEuro {
    x: Option<Vec2>,
    dx: Vec2,
}

impl OneEuro {
    fn filter(&mut self, v: Vec2, dt: f32) -> Vec2 {
        let Some(x) = self.x else {
            self.x = Some(v);
            return v;
        };
        let dt = dt.max(1e-4);
        let a = alpha(D_CUTOFF, dt);
        self.dx += ((v - x) / dt - self.dx) * a;
        let cutoff = MIN_CUTOFF + BETA * self.dx.length();
        let out = x + (v - x) * alpha(cutoff, dt);
        self.x = Some(out);
        out
    }
}

/// Where the aim is in its life.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Phase {
    /// Following the eyes.
    Eyes,
    /// Without them for a while: the head; `back` since when they have
    /// been back, if they are.
    Head { back: Option<f64> },
    /// Blending from the head's forward to the eyes since `since`, from
    /// `from`.
    Blend { since: f64, from: Vec2 },
}

/// The gaze filter of one session.
#[derive(Clone, Debug)]
pub struct GazeFilter {
    euro: OneEuro,
    /// The smoothed gaze and when (app time) its last sample came.
    smoothed: Option<Vec2>,
    last_at: Option<f64>,
    /// The last sample's own time, to skip one handed back twice.
    last_time: Option<f64>,
    aim: Vec2,
    phase: Phase,
    /// The app time of the last update.
    now: Option<f64>,
}

impl Default for GazeFilter {
    fn default() -> GazeFilter {
        GazeFilter {
            euro: OneEuro::default(),
            smoothed: None,
            last_at: None,
            last_time: None,
            aim: Vec2::ZERO,
            phase: Phase::Head { back: None },
            now: None,
        }
    }
}

impl GazeFilter {
    /// The aim at `now` (seconds, the app's clock) for a head turned `head`
    /// (world), from `reading` if the tracker gave one, `tracking` saying
    /// whether the headset can track the eyes, with `correction` applied to
    /// each sample before smoothing.
    pub fn update(
        &mut self,
        now: f64,
        head: Quat,
        tracking: bool,
        reading: Option<Reading>,
        correction: &Correction,
    ) -> Aim {
        let dt = self.now.map_or(0.0, |t| (now - t).max(0.0)) as f32;
        self.now = Some(now);
        if !tracking {
            *self = GazeFilter {
                now: Some(now),
                ..GazeFilter::default()
            };
            return Aim {
                direction: Vec3::NEG_Z,
                usable: true,
                from_eyes: false,
                status: GazeStatus::NoEyeTracking,
                gaze: None,
                current: false,
            };
        }
        if let Some(r) = reading {
            self.take(now, head, r, correction);
        }
        let lost = self.last_at.map_or(f64::INFINITY, |t| now - t);
        let fresh = reading.is_some() && lost <= CURRENT;
        let gaze = self.smoothed;
        let to_head = |aim: Vec2| aim * (1.0 - alpha_time(dt, TO_HEAD));
        let (usable, from_eyes, status) = match (self.phase, gaze) {
            (Phase::Eyes, Some(g)) if fresh => {
                self.aim = g;
                (true, true, GazeStatus::Eyes)
            }
            (Phase::Eyes, _) if lost <= BLINK => (false, false, GazeStatus::Blink),
            (Phase::Head { back }, Some(_)) if fresh => {
                let since = back.unwrap_or(now);
                if now - since >= REACQUIRE {
                    self.phase = Phase::Blend {
                        since: now,
                        from: self.aim,
                    };
                } else {
                    self.phase = Phase::Head { back: Some(since) };
                }
                self.aim = to_head(self.aim);
                (true, false, GazeStatus::Lost)
            }
            (Phase::Blend { since, from }, Some(g)) if fresh => {
                let t = ((now - since) / BLEND).clamp(0.0, 1.0) as f32;
                self.aim = from + (g - from) * t;
                if t >= 1.0 {
                    self.phase = Phase::Eyes;
                }
                let status = if t >= 1.0 {
                    GazeStatus::Eyes
                } else {
                    GazeStatus::Lost
                };
                (true, t >= 1.0, status)
            }
            // Without the eyes past a blink: the head.
            _ => {
                self.phase = Phase::Head { back: None };
                self.aim = to_head(self.aim);
                (true, false, idle_or_lost(lost))
            }
        };
        Aim {
            direction: direction(self.aim),
            usable,
            from_eyes,
            status,
            gaze: gaze.map(direction),
            current: fresh,
        }
    }

    /// Takes in one reading, if it is new, fresh and plausible.
    fn take(&mut self, now: f64, head: Quat, r: Reading, correction: &Correction) {
        if r.age.is_some_and(|a| a > STALE) {
            return;
        }
        if let (Some(t), Some(last)) = (r.time, self.last_time)
            && t <= last
        {
            return;
        }
        let yp = yaw_pitch(head.inverse() * r.direction);
        if direction(yp).angle_between(Vec3::NEG_Z) > PLAUSIBLE {
            return;
        }
        let yp = correction.apply(yp);
        // The samples' own times step the filter, where the runtime gives
        // them; a gap starts it afresh.
        let dt = match (r.time, self.last_time) {
            (Some(t), Some(last)) => t - last,
            _ => self.last_at.map_or(f64::INFINITY, |l| now - l),
        };
        if dt > GAP {
            self.euro = OneEuro::default();
        }
        self.smoothed = Some(self.euro.filter(yp, dt.min(GAP) as f32));
        self.last_at = Some(now);
        self.last_time = r.time.or(self.last_time);
    }
}

/// The share of the way a first-order glide of time constant `tau` makes
/// over `dt` seconds.
fn alpha_time(dt: f32, tau: f32) -> f32 {
    1.0 - (-dt / tau).exp()
}

fn idle_or_lost(lost: f64) -> GazeStatus {
    if lost >= IDLE {
        GazeStatus::NotDelivering
    } else {
        GazeStatus::Lost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn deg(yp: Vec2) -> Vec2 {
        yp * 180.0 / std::f32::consts::PI
    }

    /// A reading of `yaw` and `pitch` degrees with the head still, sampled
    /// at `t`, 30 ms old.
    fn reading(yaw: f32, pitch: f32, t: f64) -> Reading {
        Reading {
            direction: direction(Vec2::new(yaw, pitch).map(f32::to_radians)),
            time: Some(t),
            age: Some(0.03),
        }
    }

    /// Runs `trace(t)` (degrees, or `None` for no sample) at `hz` for
    /// `seconds`, from `t0`; the aims.
    fn run(
        f: &mut GazeFilter,
        hz: f64,
        t0: f64,
        seconds: f64,
        mut trace: impl FnMut(f64) -> Option<(f32, f32)>,
    ) -> Vec<(f64, Aim)> {
        let n = (seconds * hz).round() as usize;
        (1..=n)
            .map(|i| {
                let t = t0 + i as f64 / hz;
                let r = trace(t).map(|(y, p)| reading(y, p, t));
                (
                    t,
                    f.update(t, Quat::IDENTITY, true, r, &Correction::default()),
                )
            })
            .collect()
    }

    /// A filter following the eyes at (`yaw`, `pitch`) by `t0 + 1` s.
    fn settled(hz: f64, yaw: f32, pitch: f32) -> GazeFilter {
        let mut f = GazeFilter::default();
        run(&mut f, hz, 0.0, 1.0, |_| Some((yaw, pitch)));
        f
    }

    #[test]
    fn a_fixation_comes_out_with_under_half_its_jitter() {
        for hz in [72.0, 90.0, 120.0] {
            let mut f = settled(hz, 5.0, -3.0);
            let mut noise = Noise(7);
            let mut raw = Vec::new();
            let aims = run(&mut f, hz, 1.0, 4.0, |_| {
                let s = (5.0 + 0.5 * noise.normal(), -3.0 + 0.5 * noise.normal());
                raw.push(s);
                Some(s)
            });
            let spread = |v: &[(f32, f32)]| {
                let n = v.len() as f32;
                let my = v.iter().map(|s| s.0).sum::<f32>() / n;
                (v.iter().map(|s| (s.0 - my).powi(2)).sum::<f32>() / n).sqrt()
            };
            let out: Vec<(f32, f32)> = aims
                .iter()
                .map(|(_, a)| {
                    let d = deg(yaw_pitch(a.direction));
                    (d.x, d.y)
                })
                .collect();
            let ratio = spread(&out) / spread(&raw);
            eprintln!("{hz} Hz: fixation jitter x{ratio:.2}");
            assert!(ratio < 0.5, "{hz} Hz: {ratio}");
            assert!(
                aims.iter()
                    .all(|(_, a)| a.status == GazeStatus::Eyes && a.usable)
            );
        }
    }

    #[test]
    fn a_saccade_is_caught_within_a_degree_in_three_frames() {
        for hz in [72.0, 90.0, 120.0] {
            let mut f = settled(hz, -10.0, 0.0);
            let aims = run(&mut f, hz, 1.0, 0.2, |_| Some((10.0, 0.0)));
            let off: Vec<f32> = aims
                .iter()
                .map(|(_, a)| (deg(yaw_pitch(a.direction)).x - 10.0).abs())
                .collect();
            eprintln!(
                "{hz} Hz: 20° saccade, off by {:?} after 1-4 frames",
                &off[..4]
            );
            // Within a degree by 33 ms: three frames at 90 Hz (four at 120,
            // three at 72).
            let frames = (0.0333 * hz).ceil() as usize;
            assert!(
                off[frames - 1] < 1.0,
                "{hz} Hz: {} after {frames} frames",
                off[frames - 1]
            );
        }
    }

    #[test]
    fn a_blink_holds_unusable_and_carries_on_without_a_blend() {
        for hz in [72.0, 90.0, 120.0] {
            let mut f = settled(hz, 8.0, 4.0);
            let before = run(&mut f, hz, 1.0, 0.05, |_| Some((8.0, 4.0)))
                .last()
                .unwrap()
                .1;
            let blink = run(&mut f, hz, 1.05, 0.2, |_| None);
            for (_, a) in &blink {
                assert_eq!(a.status, GazeStatus::Blink);
                assert!(!a.usable && !a.current);
                assert!(a.direction.angle_between(before.direction) < 1e-4, "held");
            }
            // The eyes back somewhere else: followed at once.
            let after = run(&mut f, hz, 1.25, 0.05, |_| Some((-6.0, 0.0)));
            let (_, a) = after[0];
            assert_eq!(a.status, GazeStatus::Eyes);
            assert!(a.usable && a.from_eyes);
            assert!(
                (deg(yaw_pitch(a.direction)).x + 6.0).abs() < 0.01,
                "no blend"
            );
        }
    }

    #[test]
    fn a_long_loss_glides_to_the_head_and_the_eyes_take_over_again() {
        for hz in [72.0, 90.0, 120.0] {
            let mut f = settled(hz, 15.0, 0.0);
            let lost = run(&mut f, hz, 1.0, 1.0, |_| None);
            let (_, end) = lost.last().unwrap();
            assert_eq!(end.status, GazeStatus::Lost);
            assert!(end.usable && !end.from_eyes);
            // 0.6 s past the blink at 0.3 s: under a degree from forward.
            assert!(end.direction.angle_between(Vec3::NEG_Z).to_degrees() < 1.0);
            // Back: after a moment, blended over, following the eyes.
            let back = run(&mut f, hz, 2.0, 0.6, |_| Some((-12.0, 3.0)));
            let first_eyes = back.iter().position(|(_, a)| a.from_eyes).expect("back");
            let t = back[first_eyes].0 - 2.0;
            eprintln!("{hz} Hz: the eyes take over {t:.2} s after they return");
            assert!((0.35..0.5).contains(&t), "{t}");
            // No jump: each frame moves it less than the blend's share.
            for w in back.windows(2) {
                assert!(
                    w[0].1
                        .direction
                        .angle_between(w[1].1.direction)
                        .to_degrees()
                        < 2.0
                );
            }
            // Never delivering again: the tracker isn't.
            let idle = run(&mut f, hz, 2.6, 3.5, |_| None);
            assert_eq!(idle.last().unwrap().1.status, GazeStatus::NotDelivering);
        }
    }

    #[test]
    fn stale_far_off_and_repeated_samples_change_nothing() {
        let mut f = settled(90.0, 4.0, 2.0);
        let t = 1.0 + 1.0 / 90.0;
        let before = f.clone();
        let mut r = reading(-20.0, 10.0, t);
        r.age = Some(0.4);
        f.update(t, Quat::IDENTITY, true, Some(r), &Correction::default());
        let far = reading(70.0, 0.0, t);
        f.update(t, Quat::IDENTITY, true, Some(far), &Correction::default());
        let repeated = reading(-20.0, 10.0, 1.0);
        let a = f.update(
            t,
            Quat::IDENTITY,
            true,
            Some(repeated),
            &Correction::default(),
        );
        assert_eq!(f.smoothed, before.smoothed);
        assert!((deg(yaw_pitch(a.direction)) - Vec2::new(4.0, 2.0)).length() < 0.01);
    }

    #[test]
    fn the_dot_sits_on_the_gaze_facing_back_at_one_angular_size() {
        let head = (Vec3::new(0.2, 1.6, -0.1), Quat::from_rotation_y(0.4));
        let gaze = direction(Vec2::new(0.2, -0.1));
        let mut sizes = Vec::new();
        for distance in [0.5f32, 2.0, 8.0] {
            let (pose, size) = dot_place(head, gaze, distance, DOT_RING);
            let centre = pose.transform_point3(Vec3::ZERO);
            let ray = head.1 * gaze;
            // On the gaze ray, that far out.
            assert!(((centre - head.0) - ray * distance).length() < 1e-4);
            // Facing back along it: its +Z towards the head.
            let facing = pose.transform_vector3(Vec3::Z);
            assert!(facing.angle_between(-ray) < 1e-4);
            // The same angle across.
            sizes.push((size / 2.0 / distance).atan());
        }
        assert!(
            sizes.iter().all(|a| (a - DOT_RING).abs() < 1e-5),
            "{sizes:?}"
        );
    }

    #[test]
    fn without_eye_tracking_the_head_aims() {
        let mut f = GazeFilter::default();
        let a = f.update(0.0, Quat::IDENTITY, false, None, &Correction::default());
        assert_eq!(
            (a.status, a.usable, a.from_eyes),
            (GazeStatus::NoEyeTracking, true, false)
        );
        assert_eq!(a.direction, Vec3::NEG_Z);
        // Tracking but nothing ever came: the tracker isn't delivering.
        let a = f.update(0.1, Quat::IDENTITY, true, None, &Correction::default());
        assert_eq!(a.status, GazeStatus::NotDelivering);
    }

    #[test]
    fn the_aim_turns_with_the_head() {
        let head = Quat::from_rotation_y(0.5);
        let mut f = GazeFilter::default();
        let mut a = f.update(0.0, head, true, None, &Correction::default());
        for i in 1..90 {
            let t = i as f64 / 90.0;
            let r = Reading {
                direction: head * direction(Vec2::new(0.1, 0.05)),
                time: Some(t),
                age: None,
            };
            a = f.update(t, head, true, Some(r), &Correction::default());
        }
        assert!((yaw_pitch(a.direction) - Vec2::new(0.1, 0.05)).length() < 1e-3);
    }
}
