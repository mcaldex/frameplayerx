//! Autofocus (Settings › Video position, experimental), as DeoVR has it:
//! in a 3D 180°/360° or fisheye video, what you look at that is nearer
//! than the distance chosen is shown at that distance, so a close-up
//! doesn't split in two. The eyes' pictures turn apart as Depth turns them;
//! the picture's own depth stays.
//!
//! The renderer reads how far apart the eyes see what is at the centre of
//! view ([`fp_render::focus`]); this turns those readings into Depth. The
//! whole picture moving in depth is felt, so it moves as little and as
//! gently as it can:
//! - It goes by what is ahead only once the latest readings agree, so a
//!   stray frame, a glance or an edge between near and far moves nothing.
//! - Something nearer is pushed back at once (it would split in two);
//!   for something farther it waits until you have looked there a while,
//!   and changes smaller than a fraction of a degree are left alone.
//! - It starts and stops gently, settling over the refocus time, and never
//!   faster than [`MAX_SPEED`].
//! - With the eyes tracked ([`Eyes`]), each thing they land on is read
//!   afresh, and most of the moving is done while they jump or blink,
//!   when it is hardest to see.

use fp_render::Focus;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Eyes this far apart, metres, are who the distances are worked out for.
const IPD: f32 = 0.063;
/// What is ahead is the median of this many sure readings, once they agree.
const READINGS: usize = 6;
/// Readings older than this, seconds, no longer count.
const FRESH: f32 = 0.25;
/// The readings agree when all but the farthest and the nearest are within
/// this, degrees: an edge between near and far, read one way then the
/// other, doesn't.
const AGREE: f32 = 0.5;
/// Seconds what is ahead is kept once the readings stop agreeing (a blank
/// wall, the sky, an edge) before autofocus lets go.
const HOLD: f32 = 1.0;
/// Something needing this much more Depth, degrees, is pushed back at once.
const NEARER: f32 = 0.3;
/// Something needing this much less, degrees, and asked for [`RELAX_AFTER`]
/// seconds, no sooner than [`SETTLED`] seconds after the last change, lets
/// the picture come forward again. A glance farther moves nothing.
const FARTHER: f32 = 0.75;
const RELAX_AFTER: f32 = 0.5;
const SETTLED: f32 = 1.0;
/// The fastest the picture moves in depth, degrees a second.
pub const MAX_SPEED: f32 = 12.0;
/// While the eyes jump or blink it settles this many times faster, and
/// moves as fast as [`UNSEEN_SPEED`].
const UNSEEN_FASTER: f32 = 3.0;
pub const UNSEEN_SPEED: f32 = 60.0;
/// Readings come this many frames late ([`fp_render::Renderer::focus`]):
/// once the eyes land, the first ones are still of where they were.
const LATE: u8 = 2;
/// A critically damped move is 95% of the way after this many of its time
/// constants.
const SETTLE: f32 = 4.74;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AutofocusSettings {
    pub enabled: bool,
    /// What you look at is shown no nearer than this, metres.
    pub nearest: f32,
    /// Seconds the picture takes to settle on something new.
    pub refocus_secs: f32,
}

impl Default for AutofocusSettings {
    fn default() -> Self {
        AutofocusSettings {
            enabled: false,
            nearest: 0.7,
            refocus_secs: 0.6,
        }
    }
}

/// What the eyes do this frame, as autofocus uses it (see
/// [`crate::gaze::GazeTracker`]); all false while they aren't tracked.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Eyes {
    /// They landed on something new: what was read before is of something
    /// else.
    pub landed: bool,
    /// They are mid-jump or closed: a change in depth goes unseen.
    pub unseen: bool,
}

/// Autofocus for the open video.
#[derive(Debug, Default)]
pub struct Autofocus {
    /// The latest sure readings, newest last: seconds since, and disparity
    /// in degrees.
    seen: VecDeque<(f32, f32)>,
    /// Frames whose readings are still of where the eyes were before they
    /// landed.
    late: u8,
    /// What is ahead, degrees, from the last readings that agreed.
    ahead: Option<f32>,
    /// Seconds since the readings last agreed.
    unsure: f32,
    /// The Depth aimed for, degrees, and seconds since it changed.
    target: f32,
    held: f32,
    /// Seconds less Depth has been asked for.
    farther: f32,
    /// The Depth added now, degrees, and how fast it moves, degrees a
    /// second.
    shown: f32,
    speed: f32,
}

impl Autofocus {
    /// Takes this frame's reading (`None`: nothing sure was read) and moves
    /// towards the Depth it asks for, given the video's own `depth`
    /// (degrees) and what the `eyes` do. Returns the Depth to add, degrees.
    pub fn update(
        &mut self,
        reading: Option<Focus>,
        depth: f32,
        s: &AutofocusSettings,
        eyes: Eyes,
        dt: f32,
    ) -> f32 {
        for r in &mut self.seen {
            r.0 += dt;
        }
        while self.seen.front().is_some_and(|r| r.0 > FRESH) {
            self.seen.pop_front();
        }
        if eyes.landed {
            self.seen.clear();
            self.late = LATE;
        }
        let late = self.late > 0;
        self.late = self.late.saturating_sub(1);
        if let Some(f) = reading.filter(|_| !late) {
            if self.seen.len() == READINGS {
                self.seen.pop_front();
            }
            self.seen.push_back((0.0, f.disparity));
        }
        self.unsure += dt;
        if let Some(d) = agreed(&self.seen) {
            self.ahead = Some(d);
            self.unsure = 0.0;
        } else if self.unsure > HOLD {
            self.ahead = None;
        }
        let wanted = self
            .ahead
            .map_or(0.0, |d| (d - depth - angle(s.nearest)).max(0.0));
        self.held += dt;
        if wanted > self.target + NEARER {
            self.retarget(wanted, depth, "nearer");
        } else if wanted < self.target - FARTHER {
            self.farther += dt;
            if self.farther >= RELAX_AFTER && self.held >= SETTLED {
                self.retarget(wanted, depth, "farther");
            }
        } else {
            self.farther = 0.0;
        }
        self.ease(s.refocus_secs, eyes.unseen, dt);
        self.shown
    }

    /// Aims for `target` Depth, and says why in the log: how a refocus
    /// came about is what tuning it needs.
    fn retarget(&mut self, target: f32, depth: f32, why: &str) {
        let what = match self.ahead.map(|d| distance(d - depth)) {
            Some(Some(m)) => format!("{why}, {m:.2} m ahead"),
            Some(None) => format!("{why}, far ahead"),
            None => "nothing sure ahead".into(),
        };
        log::info!(
            "autofocus: {what}: Depth +{:.2}° to +{target:.2}°",
            self.target
        );
        self.target = target;
        self.held = 0.0;
        self.farther = 0.0;
    }

    /// One frame of a critically damped move towards the target: no jolt
    /// as it starts, no overshoot, 95% of the way in `settle` seconds, and
    /// never faster than [`MAX_SPEED`]; faster while the change goes
    /// `unseen`.
    fn ease(&mut self, settle: f32, unseen: bool, dt: f32) {
        let (faster, cap) = if unseen {
            (UNSEEN_FASTER, UNSEEN_SPEED)
        } else {
            (1.0, MAX_SPEED)
        };
        let w = SETTLE / settle.max(0.05) * faster;
        let off = self.shown - self.target;
        let j = self.speed + w * off;
        let fade = (-w * dt).exp();
        let next = self.target + (off + j * dt) * fade;
        let step = next - self.shown;
        let capped = step.clamp(-cap * dt, cap * dt);
        self.speed = if capped == step {
            (self.speed - w * j * dt) * fade
        } else {
            capped / dt
        };
        self.shown += capped;
    }

    /// How far apart the eyes see what is ahead, degrees, from the last
    /// readings that agreed.
    pub fn disparity(&self) -> Option<f32> {
        self.ahead
    }

    /// The Depth added now, degrees.
    pub fn shown(&self) -> f32 {
        self.shown
    }
}

/// The median of `seen`, when there are [`READINGS`] of them and all but the
/// farthest and the nearest are within [`AGREE`].
fn agreed(seen: &VecDeque<(f32, f32)>) -> Option<f32> {
    if seen.len() < READINGS {
        return None;
    }
    let mut d: Vec<f32> = seen.iter().map(|r| r.1).collect();
    d.sort_by(f32::total_cmp);
    let n = d.len();
    (d[n - 2] - d[1] <= AGREE).then(|| (d[(n - 1) / 2] + d[n / 2]) / 2.0)
}

/// The angle, degrees, between the eyes' lines of sight to something `m`
/// metres away.
fn angle(m: f32) -> f32 {
    2.0 * (IPD / 2.0 / m.max(0.01)).atan().to_degrees()
}

/// How far away, metres, something looks when the eyes' lines of sight
/// meet at `deg` degrees; `None` when they don't meet (far away).
pub fn distance(deg: f32) -> Option<f32> {
    (deg > 0.05).then(|| IPD / 2.0 / (deg / 2.0).to_radians().tan())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 90.0;

    fn at(disparity: f32) -> Option<Focus> {
        Some(Focus {
            disparity,
            score: 0.9,
        })
    }

    /// Feeds `reading` for `secs` at 90 Hz; the Depth added at the end.
    fn run(af: &mut Autofocus, reading: Option<Focus>, depth: f32, secs: f32) -> f32 {
        let s = AutofocusSettings::default();
        let mut shown = 0.0;
        for _ in 0..(secs / DT).round() as usize {
            shown = af.update(reading, depth, &s, Eyes::default(), DT);
        }
        shown
    }

    /// The Depth something `m` metres away needs to be shown at the
    /// nearest distance.
    fn needs(m: f32) -> f32 {
        angle(m) - angle(AutofocusSettings::default().nearest)
    }

    #[test]
    fn something_near_is_shown_at_the_nearest_distance() {
        let mut af = Autofocus::default();
        // 0.3 m away.
        let near = angle(0.3);
        let added = run(&mut af, at(near), 0.0, 2.0);
        let shown_at = distance(near - added).expect("still near");
        assert!((shown_at - 0.7).abs() < 0.01, "{shown_at} m");
        // The video's own Depth counts towards it.
        let mut af = Autofocus::default();
        let with_depth = run(&mut af, at(near), 2.0, 2.0);
        assert!((added - 2.0 - with_depth).abs() < 0.01);
    }

    #[test]
    fn something_far_enough_is_left_alone() {
        let mut af = Autofocus::default();
        assert_eq!(run(&mut af, at(angle(1.5)), 0.0, 1.0), 0.0);
        assert_eq!(run(&mut af, at(-1.0), 0.0, 1.0), 0.0);
    }

    #[test]
    fn it_starts_gently_and_settles_in_the_refocus_time() {
        let s = AutofocusSettings::default();
        let mut af = Autofocus::default();
        let target = needs(0.5);
        // The readings must agree first: nothing moves on the first ones.
        assert_eq!(run(&mut af, at(angle(0.5)), 0.0, DT * 5.0), 0.0);
        // Then a slow start, with no jump.
        let first = af.update(at(angle(0.5)), 0.0, &s, Eyes::default(), DT);
        assert!(first > 0.0 && first < target * 0.02, "{first} of {target}");
        let settled = run(&mut af, at(angle(0.5)), 0.0, s.refocus_secs);
        assert!(settled > target * 0.95, "{settled} of {target}");
        // And no overshoot.
        let later = run(&mut af, at(angle(0.5)), 0.0, 1.0);
        assert!(later <= target + 1e-4 && later > target * 0.999);
    }

    #[test]
    fn it_never_moves_faster_than_the_cap() {
        let mut af = Autofocus::default();
        let s = AutofocusSettings {
            refocus_secs: 0.1,
            ..Default::default()
        };
        let mut last = 0.0;
        for _ in 0..180 {
            let shown = af.update(at(angle(0.2)), 0.0, &s, Eyes::default(), DT);
            let speed = (shown - last) / DT;
            assert!(speed <= MAX_SPEED + 1e-3, "{speed}°/s");
            last = shown;
        }
        assert!((last - needs(0.2)).abs() < 0.01);
    }

    #[test]
    fn a_stray_reading_does_not_move_it() {
        let mut af = Autofocus::default();
        let steady = run(&mut af, at(angle(0.4)), 0.0, 2.0);
        let s = AutofocusSettings::default();
        for stray in [at(angle(0.15)), at(-2.0), at(angle(0.15))] {
            af.update(stray, 0.0, &s, Eyes::default(), DT);
        }
        assert!((af.shown() - steady).abs() < 1e-3);
        assert!((run(&mut af, at(angle(0.4)), 0.0, 1.0) - steady).abs() < 1e-3);
    }

    #[test]
    fn an_edge_read_both_ways_moves_nothing() {
        let mut af = Autofocus::default();
        let steady = run(&mut af, at(angle(0.4)), 0.0, 2.0);
        // A face against the background, read as one then the other, for
        // less than the hold.
        let s = AutofocusSettings::default();
        for i in 0..60 {
            let d = if i % 2 == 0 { angle(0.3) } else { angle(3.0) };
            af.update(at(d), 0.0, &s, Eyes::default(), DT);
        }
        assert!((af.shown() - steady).abs() < 1e-3);
    }

    #[test]
    fn a_glance_farther_moves_nothing() {
        let mut af = Autofocus::default();
        let steady = run(&mut af, at(angle(0.35)), 0.0, 2.0);
        // Looking from a close-up to the background and back, a third of a
        // second each, for six seconds: it stays pushed back.
        for _ in 0..9 {
            run(&mut af, at(angle(3.0)), 0.0, 0.33);
            run(&mut af, at(angle(0.35)), 0.0, 0.33);
            assert!((af.shown() - steady).abs() < 1e-3);
        }
    }

    #[test]
    fn looking_farther_a_while_lets_it_come_forward() {
        let mut af = Autofocus::default();
        let near = run(&mut af, at(angle(0.3)), 0.0, 2.0);
        let half = run(&mut af, at(angle(0.5)), 0.0, 0.4);
        assert!((half - near).abs() < 1e-3, "it waits");
        let later = run(&mut af, at(angle(0.5)), 0.0, 1.5);
        assert!((later - needs(0.5)).abs() < 0.01, "{later}");
    }

    #[test]
    fn a_small_change_is_left_alone() {
        let mut af = Autofocus::default();
        let steady = run(&mut af, at(angle(0.4)), 0.0, 2.0);
        // A little nearer, then a little farther: under the thresholds.
        let nearer = steady + NEARER * 0.8;
        let farther = steady - FARTHER * 0.8;
        let base = angle(AutofocusSettings::default().nearest);
        for d in [nearer, farther] {
            assert!((run(&mut af, at(d + base), 0.0, 2.0) - steady).abs() < 1e-3);
        }
    }

    #[test]
    fn it_holds_what_it_saw_then_lets_go() {
        let mut af = Autofocus::default();
        let held = run(&mut af, at(angle(0.3)), 0.0, 2.0);
        assert!((run(&mut af, None, 0.0, 0.9) - held).abs() < 1e-3);
        assert!(run(&mut af, None, 0.0, 2.0) < held * 0.05);
    }

    #[test]
    fn readings_from_before_the_eyes_landed_are_left_out() {
        let s = AutofocusSettings::default();
        let mut af = Autofocus::default();
        run(&mut af, at(angle(0.4)), 0.0, 2.0);
        // The eyes land on something 0.3 m away; the next two readings are
        // still of what they left.
        let landed = Eyes {
            landed: true,
            ..Eyes::default()
        };
        af.update(at(angle(0.4)), 0.0, &s, landed, DT);
        af.update(at(angle(0.4)), 0.0, &s, Eyes::default(), DT);
        // Five readings of the new thing aren't enough yet...
        for _ in 0..5 {
            af.update(at(angle(0.3)), 0.0, &s, Eyes::default(), DT);
        }
        assert_eq!(af.disparity(), Some(angle(0.4)));
        // ...six are, with nothing from before among them.
        af.update(at(angle(0.3)), 0.0, &s, Eyes::default(), DT);
        assert_eq!(af.disparity(), Some(angle(0.3)));
    }

    #[test]
    fn while_the_eyes_jump_it_moves_faster() {
        let s = AutofocusSettings::default();
        let jumping = Eyes {
            unseen: true,
            ..Eyes::default()
        };
        let mut moved = [0.0; 2];
        for (i, eyes) in [Eyes::default(), jumping].into_iter().enumerate() {
            let mut af = Autofocus::default();
            // Sure of something near and on the move, then a saccade's
            // five frames.
            for _ in 0..6 {
                af.update(at(angle(0.3)), 0.0, &s, Eyes::default(), DT);
            }
            let before = af.shown();
            for _ in 0..5 {
                af.update(at(angle(0.3)), 0.0, &s, eyes, DT);
            }
            moved[i] = af.shown() - before;
        }
        assert!(moved[1] > moved[0] * 2.0, "{moved:?}");
        assert!(moved[1] / (5.0 * DT) <= UNSEEN_SPEED + 1e-3);
    }

    #[test]
    fn distances_and_angles_agree() {
        for m in [0.2, 0.7, 3.0] {
            assert!((distance(angle(m)).expect("near") - m).abs() < 1e-3);
        }
        assert_eq!(distance(0.0), None);
    }
}
