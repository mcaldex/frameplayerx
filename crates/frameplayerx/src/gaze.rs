//! Where you look, for whatever follows it: autofocus, the performance
//! overlay's eye line, and anything else that wants the eyes (foveated
//! rendering, say).
//!
//! It takes the eye tracker's gaze ([`fp_xr::gaze`]) when the headset has
//! one and Settings › General › Eye tracking allows it, else the head's
//! forward. It gives two signals:
//! - [`GazeTracker::now`]: the eyes this frame, as tracked, for what must
//!   keep up with them.
//! - [`GazeTracker::fixation`]: what you are looking at. It holds through
//!   saccades (the eyes' jumps, when they take little in) and blinks, and
//!   smooths a fixation's tremor, so what follows it doesn't jitter.

use fp_xr::Gaze;
use glam::{Quat, Vec3};

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
            _ => gaze.direction,
        });
        self.jumping = false;
    }

    /// Where the eyes point this frame, as tracked; `None` without eye
    /// tracking.
    pub fn now(&self) -> Option<Gaze> {
        self.now
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
        }
        // Landed, it is there at once.
        t.update(at(dir(30.0, 0.0)), head(), true, DT);
        assert!(looking(&t).direction.distance(dir(30.0, 0.0)) < 1e-6);
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
        for _ in 0..(0.3 / DT) as usize {
            t.update(None, head(), true, DT);
        }
        assert_eq!(looking(&t).source, Source::Head);
    }

    #[test]
    fn angles_from_the_head() {
        let [yaw, pitch] = from_head(dir(30.0, -10.0), Quat::IDENTITY);
        assert!((yaw - 30.0).abs() < 1e-4 && (pitch + 10.0).abs() < 1e-4);
    }
}
