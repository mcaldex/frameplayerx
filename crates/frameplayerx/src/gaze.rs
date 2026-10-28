//! Where you look, for whatever follows it: autofocus, the performance
//! overlay's eye line, the rings of Show where you look ([`Ring`]), and
//! anything else that wants the eyes (foveated rendering, say).
//!
//! It takes the eye tracker's gaze ([`fp_xr::gaze`]) when the headset has
//! one and Settings › General › Eye tracking allows it, else the head's
//! forward. It gives two signals:
//! - [`GazeTracker::now`]: the eyes this frame, as tracked, for what must
//!   keep up with them.
//! - [`GazeTracker::fixation`]: what you are looking at. It holds through
//!   saccades (the eyes' jumps, when they take little in) and blinks, and
//!   smooths a fixation's tremor, so what follows it doesn't jitter.

use fp_render::QuadDraw;
use fp_xr::Gaze;
use glam::{Quat, Vec2, Vec3};

/// Faster than this, degrees a second, the gaze is jumping (a saccade)
/// or sweeping with the head. Fixations, and following something that
/// moves, stay well under it.
const SACCADE: f32 = 100.0;
/// A fixation's tremor is smoothed over this, seconds.
const STEADY: f32 = 0.05;
/// Seconds the last fixation is kept while the eyes aren't tracked (a
/// blink is a tenth to a third of a second) before the head takes over.
const HOLD: f32 = 0.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Eyes,
    /// Straight ahead: no eye tracker, not allowed, or the eyes lost.
    Head,
}

/// Where you are looking.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Looking {
    /// In the world, a unit vector.
    pub direction: Vec3,
    pub source: Source,
}

/// What eye tracking there is, for Settings to say.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Status {
    /// The headset doesn't offer it.
    #[default]
    None,
    /// Offered, but not following your eyes: turned off here, eye data not
    /// allowed in the headset's settings, or the eyes lost.
    Offered,
    Tracking,
}

#[derive(Debug, Default)]
pub struct GazeTracker {
    /// This frame's gaze, when there is one and it may be used.
    now: Option<Gaze>,
    /// The fixation, in the world.
    fixation: Option<Vec3>,
    /// Last frame's gaze, for how fast it moves.
    last: Option<Vec3>,
    /// The eyes are mid-saccade: the fixation waits for where they land.
    jumping: bool,
    /// This frame a new fixation started.
    landed: bool,
    /// Seconds since the eyes were last tracked.
    lost: f32,
    /// The head's forward, in the world.
    head: Option<Vec3>,
}

impl GazeTracker {
    /// Each frame: the tracker's gaze (if the headset has one) and the head,
    /// both in the world; `allowed` is the Eye tracking setting.
    pub fn update(
        &mut self,
        eyes: Option<Gaze>,
        head: Option<(Vec3, Quat)>,
        allowed: bool,
        dt: f32,
    ) {
        self.head = head.map(|(_, q)| q * Vec3::NEG_Z);
        self.now = eyes.filter(|_| allowed);
        self.landed = false;
        let Some(gaze) = self.now else {
            self.lost += dt;
            self.last = None;
            if self.lost > HOLD || !allowed {
                self.fixation = None;
            }
            return;
        };
        self.lost = 0.0;
        // In the world: turning the head while looking at one thing (the
        // eyes turn back as fast) keeps the gaze still there.
        let speed = match self.last {
            Some(a) if dt > 0.0 => a.angle_between(gaze.direction).to_degrees() / dt,
            _ => 0.0,
        };
        self.last = Some(gaze.direction);
        if speed > SACCADE {
            self.jumping = true;
            return;
        }
        self.fixation = Some(match self.fixation {
            Some(f) if !self.jumping => f
                .lerp(gaze.direction, 1.0 - (-dt / STEADY).exp())
                .normalize(),
            // Landed: a new fixation starts where the eyes are.
            _ => {
                self.landed = true;
                gaze.direction
            }
        });
        self.jumping = false;
    }

    /// Where the eyes point this frame, as tracked; `None` without eye
    /// tracking.
    pub fn now(&self) -> Option<Gaze> {
        self.now
    }

    /// The eyes landed on something new this frame: what was looked at
    /// before is something else.
    pub fn landed(&self) -> bool {
        self.landed
    }

    /// The eyes are mid-saccade or closed (a blink), so take in little of
    /// what is shown: a change made now is hard to see.
    pub fn unseen(&self) -> bool {
        self.fixation.is_some() && (self.jumping || self.now.is_none())
    }

    /// What you are looking at: the eyes' fixation, else straight ahead;
    /// `None` until the head is tracked.
    pub fn fixation(&self) -> Option<Looking> {
        match self.fixation {
            Some(direction) => Some(Looking {
                direction,
                source: Source::Eyes,
            }),
            None => self.head.map(|direction| Looking {
                direction,
                source: Source::Head,
            }),
        }
    }

    /// `offered`: the headset has an eye tracker.
    pub fn status(&self, offered: bool) -> Status {
        match (offered, self.fixation.is_some()) {
            (false, _) => Status::None,
            (true, false) => Status::Offered,
            (true, true) => Status::Tracking,
        }
    }
}

/// How far `direction` (in the world) is from where the head faces:
/// degrees right and up.
pub fn from_head(direction: Vec3, head: Quat) -> [f32; 2] {
    let d = head.inverse() * direction;
    [
        d.x.atan2(-d.z).to_degrees(),
        d.y.clamp(-1.0, 1.0).asin().to_degrees(),
    ]
}

/// Degrees across the ring where the eyes are this frame: about how far
/// off a tracker can be.
const NOW_RING: f32 = 2.0;
/// The rings' pictures, pixels square.
pub const RING_PX: u32 = 128;
/// sRGB.
const RED: [u8; 3] = [235, 38, 38];

/// A red ring where you look, for Settings › General › Show where you look:
/// to see the tracker keep up and what autofocus reads. It faces the eyes
/// and is the same angle across at any distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ring {
    /// Where the eyes are this frame, as tracked; else what they fixate,
    /// larger, around the patch autofocus reads.
    pub now: bool,
    pub centre: Vec3,
    /// Turns the picture's +Z to the eyes.
    pub facing: Quat,
    /// Across, metres.
    pub size: f32,
}

impl Ring {
    /// The ring `distance` along `direction` (a unit vector) from `origin`.
    pub fn new(now: bool, origin: Vec3, direction: Vec3, distance: f32) -> Ring {
        let degrees = if now {
            NOW_RING
        } else {
            2.0 * fp_render::focus::PATCH
        };
        Ring {
            now,
            centre: origin + direction * distance,
            facing: Quat::from_rotation_arc(Vec3::Z, -direction),
            size: 2.0 * distance * (degrees / 2.0).to_radians().tan(),
        }
    }

    /// Where the ring starts inside, of its outer radius, and how opaque
    /// it is.
    fn look(now: bool) -> (f32, f32) {
        if now { (0.55, 1.0) } else { (0.9, 0.7) }
    }

    /// Drawn into the eyes, as the desktop preview does: a loop of short
    /// lines.
    pub fn quads(&self, eye: Vec3) -> Vec<QuadDraw> {
        const SIDES: usize = 32;
        let (inner, alpha) = Ring::look(self.now);
        let outer = self.size / 2.0;
        let mid = outer * (1.0 + inner) / 2.0;
        let point = |i: usize| {
            let a = i as f32 * std::f32::consts::TAU / SIDES as f32;
            self.centre + self.facing * Vec3::new(a.cos(), a.sin(), 0.0) * mid
        };
        let [r, g, b] = RED.map(|c| egui::ecolor::linear_from_gamma(f32::from(c) / 255.0) * alpha);
        (0..SIDES)
            .map(|i| {
                QuadDraw::line(
                    point(i),
                    point(i + 1),
                    outer * (1.0 - inner),
                    eye,
                    [r, g, b, alpha],
                )
            })
            .collect()
    }

    /// The ring's picture for the compositor: [`RING_PX`] square, soft
    /// edged, premultiplied.
    pub fn picture(now: bool) -> Vec<u8> {
        let (inner, alpha) = Ring::look(now);
        let half = RING_PX as f32 / 2.0;
        let outer = half - 1.0;
        // How much of a pixel `d` pixels inside an edge covers.
        let cover = |d: f32| (d + 0.5).clamp(0.0, 1.0);
        let mut px = Vec::with_capacity((RING_PX * RING_PX * 4) as usize);
        for y in 0..RING_PX {
            for x in 0..RING_PX {
                let r = Vec2::new(x as f32 + 0.5 - half, y as f32 + 0.5 - half).length();
                let a = cover(outer - r).min(cover(r - inner * outer)) * alpha;
                px.extend(RED.map(|c| (f32::from(c) * a).round() as u8));
                px.push((255.0 * a).round() as u8);
            }
        }
        px
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 90.0;

    fn dir(yaw: f32, pitch: f32) -> Vec3 {
        Quat::from_rotation_y(-yaw.to_radians())
            * Quat::from_rotation_x(pitch.to_radians())
            * Vec3::NEG_Z
    }

    fn at(direction: Vec3) -> Option<Gaze> {
        Some(Gaze {
            origin: Vec3::Y * 1.6,
            direction,
        })
    }

    fn head() -> Option<(Vec3, Quat)> {
        Some((Vec3::Y * 1.6, Quat::IDENTITY))
    }

    fn looking(t: &GazeTracker) -> Looking {
        t.fixation().expect("a direction")
    }

    #[test]
    fn without_eyes_it_is_where_the_head_faces() {
        let mut t = GazeTracker::default();
        assert_eq!(t.fixation(), None, "nothing tracked yet");
        let turned = Quat::from_rotation_y(0.5);
        t.update(None, Some((Vec3::ZERO, turned)), true, DT);
        assert_eq!(looking(&t).source, Source::Head);
        assert!(looking(&t).direction.distance(turned * Vec3::NEG_Z) < 1e-6);
        assert_eq!(t.status(false), Status::None);
        assert_eq!(t.status(true), Status::Offered);
    }

    #[test]
    fn it_follows_the_eyes_when_allowed() {
        let mut t = GazeTracker::default();
        t.update(at(dir(20.0, -5.0)), head(), true, DT);
        assert_eq!(looking(&t).source, Source::Eyes);
        assert!(looking(&t).direction.distance(dir(20.0, -5.0)) < 1e-6);
        assert!(t.now().is_some());
        assert_eq!(t.status(true), Status::Tracking);
        // Turned off, the head again, and nothing raw either.
        t.update(at(dir(20.0, -5.0)), head(), false, DT);
        assert_eq!(looking(&t).source, Source::Head);
        assert_eq!(t.now(), None);
    }

    #[test]
    fn a_saccade_is_held_until_the_eyes_land() {
        let mut t = GazeTracker::default();
        for _ in 0..10 {
            t.update(at(dir(0.0, 0.0)), head(), true, DT);
        }
        // 30° in three frames is 900°/s: held where it was, while `now`
        // keeps up.
        for step in [10.0, 20.0, 30.0] {
            t.update(at(dir(step, 0.0)), head(), true, DT);
            assert!(looking(&t).direction.distance(dir(0.0, 0.0)) < 1e-3);
            assert!(t.now().expect("raw").direction.distance(dir(step, 0.0)) < 1e-6);
            assert!(t.unseen() && !t.landed(), "mid-jump");
        }
        // Landed, it is there at once.
        t.update(at(dir(30.0, 0.0)), head(), true, DT);
        assert!(looking(&t).direction.distance(dir(30.0, 0.0)) < 1e-6);
        assert!(t.landed() && !t.unseen());
        t.update(at(dir(30.0, 0.0)), head(), true, DT);
        assert!(!t.landed(), "only the frame it lands");
    }

    #[test]
    fn tremor_is_smoothed_and_a_moving_thing_followed() {
        let mut t = GazeTracker::default();
        t.update(at(dir(0.0, 0.0)), head(), true, DT);
        // Half a degree of tremor each frame (45°/s, not a saccade) moves
        // the fixation much less.
        for i in 0..20 {
            let wobble = if i % 2 == 0 { 0.5 } else { -0.5 };
            t.update(at(dir(wobble, 0.0)), head(), true, DT);
        }
        let [yaw, _] = from_head(looking(&t).direction, Quat::IDENTITY);
        assert!(yaw.abs() < 0.2, "{yaw}°");
        // Following something at 20°/s keeps close behind it.
        for i in 1..=90 {
            t.update(at(dir(i as f32 * 20.0 * DT, 0.0)), head(), true, DT);
        }
        let [yaw, _] = from_head(looking(&t).direction, Quat::IDENTITY);
        assert!((yaw - 20.0).abs() < 1.5, "{yaw}°");
    }

    #[test]
    fn turning_the_head_while_looking_at_one_thing_is_no_saccade() {
        let mut t = GazeTracker::default();
        for i in 0..30 {
            // The head turns at 200°/s; the eyes stay on the same point.
            let q = Quat::from_rotation_y((i as f32 * 200.0 * DT).to_radians());
            t.update(at(dir(10.0, 0.0)), Some((Vec3::ZERO, q)), true, DT);
            assert!(looking(&t).direction.distance(dir(10.0, 0.0)) < 1e-3);
        }
    }

    #[test]
    fn a_blink_is_held_then_the_head_takes_over() {
        let mut t = GazeTracker::default();
        t.update(at(dir(15.0, 0.0)), head(), true, DT);
        for _ in 0..(0.3 / DT) as usize {
            t.update(None, head(), true, DT);
        }
        assert_eq!(looking(&t).source, Source::Eyes);
        assert!(t.unseen(), "a blink");
        for _ in 0..(0.3 / DT) as usize {
            t.update(None, head(), true, DT);
        }
        assert_eq!(looking(&t).source, Source::Head);
        assert!(!t.unseen(), "the eyes lost, not a blink");
    }

    #[test]
    fn rings_face_the_eyes_the_same_angle_across_at_any_distance() {
        let d = dir(10.0, 5.0);
        let near = Ring::new(true, Vec3::Y, d, 1.0);
        let far = Ring::new(true, Vec3::Y, d, 4.0);
        assert!(near.centre.distance(Vec3::Y + d) < 1e-6);
        assert!((near.facing * Vec3::Z).distance(-d) < 1e-5);
        assert!((near.size - 2.0 * 1f32.to_radians().tan()).abs() < 1e-6);
        assert!((far.size / near.size - 4.0).abs() < 1e-4);
        let fixation = Ring::new(false, Vec3::Y, d, 1.0);
        assert!(fixation.size > near.size, "around autofocus's patch");
    }

    #[test]
    fn a_ring_is_clear_inside_and_out() {
        for now in [true, false] {
            let px = Ring::picture(now);
            assert_eq!(px.len(), (RING_PX * RING_PX * 4) as usize);
            let alpha = |x: u32, y: u32| px[((y * RING_PX + x) * 4 + 3) as usize];
            let h = RING_PX / 2;
            assert_eq!(alpha(h, h), 0, "what you look at shows through");
            assert_eq!(alpha(0, 0), 0, "a corner");
            assert!(alpha(h, 1) > 150, "the ring: {}", alpha(h, 1));
        }
    }

    #[test]
    fn angles_from_the_head() {
        let [yaw, pitch] = from_head(dir(30.0, -10.0), Quat::IDENTITY);
        assert!((yaw - 30.0).abs() < 1e-4 && (pitch + 10.0).abs() < 1e-4);
    }
}
