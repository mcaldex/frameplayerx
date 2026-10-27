//! Autofocus (Settings › Video position, experimental), as DeoVR has it:
//! in a 3D 180°/360° or fisheye video, what you look at that is nearer
//! than the distance chosen is shown at that distance, so a close-up
//! doesn't split in two. The eyes' pictures turn apart as Depth turns them,
//! easing over the refocus time; the picture's own depth stays.
//!
//! The renderer reads how far apart the eyes see what is at the centre of
//! view ([`fp_render::focus`]); this turns those readings into Depth.

use fp_render::Focus;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Eyes this far apart, metres, are who the distances are worked out for.
const IPD: f32 = 0.063;
/// The disparity aimed for is the median of this many readings, a tenth
/// of a second at 90 Hz, so one stray frame doesn't move the picture.
const READINGS: usize = 9;
/// Seconds without a sure reading (a blank wall, the sky) before the
/// picture eases back to the video's own depth.
const HOLD: f32 = 1.5;

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

/// Autofocus for the open video.
#[derive(Debug, Default)]
pub struct Autofocus {
    /// The latest sure readings' disparities, degrees, newest last.
    seen: VecDeque<f32>,
    /// Seconds since the last of them.
    since: f32,
    /// The Depth added, degrees.
    shown: f32,
}

impl Autofocus {
    /// Takes this frame's reading (`None`: nothing sure was read) and eases
    /// towards the Depth it asks for, given the video's own `depth`
    /// (degrees). Returns the Depth to add, degrees.
    pub fn update(
        &mut self,
        reading: Option<Focus>,
        depth: f32,
        s: &AutofocusSettings,
        dt: f32,
    ) -> f32 {
        self.since += dt;
        if let Some(f) = reading {
            if self.seen.len() == READINGS {
                self.seen.pop_front();
            }
            self.seen.push_back(f.disparity);
            self.since = 0.0;
        } else if self.since > HOLD {
            self.seen.clear();
        }
        let target = self
            .disparity()
            .map_or(0.0, |d| (d - depth - angle(s.nearest)).max(0.0));
        // 95% of the way there in the refocus time.
        let k = 1.0 - (-3.0 * dt / s.refocus_secs.max(0.05)).exp();
        self.shown += (target - self.shown) * k;
        self.shown
    }

    /// How far apart the eyes see what is ahead, degrees: the median of
    /// the latest readings.
    pub fn disparity(&self) -> Option<f32> {
        let mut d: Vec<f32> = self.seen.iter().copied().collect();
        d.sort_by(f32::total_cmp);
        d.get(d.len() / 2).copied()
    }

    /// The Depth added now, degrees.
    pub fn shown(&self) -> f32 {
        self.shown
    }
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
        for _ in 0..(secs * 90.0) as usize {
            shown = af.update(reading, depth, &s, 1.0 / 90.0);
        }
        shown
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
    fn it_eases_over_the_refocus_time() {
        let s = AutofocusSettings::default();
        let mut af = Autofocus::default();
        let target = angle(0.3) - angle(s.nearest);
        let first = af.update(at(angle(0.3)), 0.0, &s, 1.0 / 90.0);
        assert!(first > 0.0 && first < target * 0.1, "{first}");
        let settled = run(&mut af, at(angle(0.3)), 0.0, s.refocus_secs);
        assert!(
            (settled / target - 0.95).abs() < 0.02,
            "{settled} of {target}"
        );
    }

    #[test]
    fn a_stray_reading_does_not_move_it() {
        let mut af = Autofocus::default();
        let steady = run(&mut af, at(angle(0.4)), 0.0, 2.0);
        let s = AutofocusSettings::default();
        for stray in [at(angle(0.15)), at(-2.0), at(angle(0.15))] {
            af.update(stray, 0.0, &s, 1.0 / 90.0);
        }
        assert!((af.shown() - steady).abs() < 1e-3);
    }

    #[test]
    fn it_holds_what_it_saw_then_lets_go() {
        let mut af = Autofocus::default();
        let held = run(&mut af, at(angle(0.3)), 0.0, 2.0);
        assert!((run(&mut af, None, 0.0, 1.0) - held).abs() < 1e-3);
        assert!(run(&mut af, None, 0.0, 2.0) < held * 0.2);
    }

    #[test]
    fn distances_and_angles_agree() {
        for m in [0.2, 0.7, 3.0] {
            assert!((distance(angle(m)).expect("near") - m).abs() < 1e-3);
        }
        assert_eq!(distance(0.0), None);
    }
}
