//! The gaze check: an eye tracker is often a few degrees off for a given
//! person, which the gaze dot makes plain. Five dots, fixed in front of
//! the head so turning it doesn't matter, show one at a time; the raw gaze
//! (head axes, before smoothing or any earlier correction) is collected
//! over the last second of each, and a correction fitted to their medians:
//! an offset, or a fuller map (a gain or a little rotation as well) when
//! that clearly does better. Pure: the times are inputs.

use crate::gaze::{Correction, Reading, direction, yaw_pitch};
use glam::{Quat, Vec2, Vec3};

/// Where the dots are, degrees of (yaw, pitch) from the head's forward:
/// the middle, then 12° above, below, left and right of it. Far enough
/// apart to show a gain, near enough that the eyes look through the
/// middle of the lenses, where trackers are best.
pub const DOTS: [[f32; 2]; 5] = [
    [0.0, 0.0],
    [0.0, 12.0],
    [0.0, -12.0],
    [-12.0, 0.0],
    [12.0, 0.0],
];

/// How long each dot shows, seconds, and how much of its start goes before
/// collecting: 1.5 and 0.5, so the eyes have arrived and settled (a
/// saccade's latency is about 0.2 s, its end a few tenths more).
const DOT_TIME: f64 = 1.5;
const SETTLE: f64 = 0.5;

/// A dot needs a good reading on at least this share of its collecting
/// frames: half. Blinks take a tenth or so; more is the eyes being lost.
const MIN_SHARE: f32 = 0.5;

/// No reading at all for this long, seconds, stops the check: the tracker
/// stopped. 0.6, longer than any blink.
const TRACKER_GONE: f64 = 0.6;

/// A sample older than this, seconds, isn't used (as the gaze filter).
const STALE: f32 = 0.25;

/// How strongly the fuller map is pulled towards no change, radians
/// squared: 0.03, about what the check's own two dots 12° out along an
/// axis give it (0.088), a third of it. Five dots can't pin six numbers
/// down alone; this keeps a gain or rotation the dots don't clearly show
/// from being fitted to their noise.
const RIDGE: f32 = 0.03;

/// The fuller map is kept only when its error is under this share of the
/// offset's alone, and at least [`CLEARLY_DEG`] under it: noise alone
/// lowers it a little, a real gain a lot.
const CLEARLY: f32 = 0.6;
const CLEARLY_DEG: f32 = 0.3;

/// Left this far off on average after the fit, degrees, the eyes didn't
/// follow the dots, and nothing is kept: 2°, past the 1.3° the worst of
/// our headset checks left.
const NOT_FOLLOWED: f32 = 2.0;

/// A correction fitted, and how far off the gaze was on average before and
/// after it, degrees.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Fit {
    pub correction: Correction,
    pub before: f32,
    pub after: f32,
}

/// Why a check stopped without a correction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// The tracker stopped sending where the eyes look.
    Tracker,
    /// A dot had a good reading on too few of its frames.
    Lost,
    /// The fit still leaves a large error: the eyes didn't follow.
    NotFollowed,
}

/// What the check does this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Step {
    /// Show a dot at this direction, head axes.
    Dot(Vec3),
    Done(Result<Fit, Stop>),
}

/// One gaze check, from start to fit.
#[derive(Clone, Debug)]
pub struct GazeCheck {
    start: f64,
    /// Each dot's samples, (yaw, pitch) radians, head axes, raw.
    samples: [Vec<Vec2>; 5],
    /// Each dot's collecting frames, and those with a good reading.
    frames: [u32; 5],
    good: [u32; 5],
    last_time: Option<f64>,
    last_reading: f64,
}

/// A dot's place, radians.
fn target(i: usize) -> Vec2 {
    Vec2::from(DOTS[i]) * std::f32::consts::PI / 180.0
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(f32::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// The mean distance, degrees, between `targets` and `seen` corrected by
/// `c`.
fn mean_error(c: &Correction, seen: &[Vec2], targets: &[Vec2]) -> f32 {
    let sum: f32 = seen
        .iter()
        .zip(targets)
        .map(|(m, t)| (c.apply(*m) - *t).length())
        .sum();
    (sum / seen.len() as f32).to_degrees()
}

/// The correction from `seen` (each dot's median gaze) to `targets`:
/// the offset that fits best, or the fuller map, fitted by least squares
/// pulled towards no change, if it does clearly better.
pub fn fit(seen: &[Vec2], targets: &[Vec2]) -> Fit {
    let n = seen.len() as f32;
    let offset = targets.iter().zip(seen).map(|(t, m)| *t - *m).sum::<Vec2>() / n;
    let shifted = Correction {
        offset: offset.to_array(),
        ..Correction::default()
    };
    let before = mean_error(&Correction::default(), seen, targets);
    let by_offset = mean_error(&shifted, seen, targets);
    // Each output axis on its own: (g_yaw, g_pitch, o) minimising the
    // squared error plus RIDGE times the gains' distance from no change.
    let mut full = Correction::default();
    for axis in 0..2 {
        let mut a = [[0.0f64; 3]; 3];
        let mut b = [0.0f64; 3];
        for (m, t) in seen.iter().zip(targets) {
            let x = [m.x as f64, m.y as f64, 1.0];
            for i in 0..3 {
                for j in 0..3 {
                    a[i][j] += x[i] * x[j];
                }
                b[i] += x[i] * t[axis] as f64;
            }
        }
        for i in 0..2 {
            a[i][i] += RIDGE as f64;
            b[i] += RIDGE as f64 * if i == axis { 1.0 } else { 0.0 };
        }
        let Some(s) = solve3(a, b) else {
            return Fit {
                correction: shifted,
                before,
                after: by_offset,
            };
        };
        full.gain[axis] = [s[0] as f32, s[1] as f32];
        full.offset[axis] = s[2] as f32;
    }
    let by_map = mean_error(&full, seen, targets);
    if by_map < CLEARLY * by_offset && by_offset - by_map > CLEARLY_DEG {
        Fit {
            correction: full,
            before,
            after: by_map,
        }
    } else {
        Fit {
            correction: shifted,
            before,
            after: by_offset,
        }
    }
}

/// Solves a 3x3 system by Cramer's rule; `None` if it is singular.
fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(a);
    if d.abs() < 1e-18 {
        return None;
    }
    Some([0, 1, 2].map(|c| {
        let mut m = a;
        for r in 0..3 {
            m[r][c] = b[r];
        }
        det(m) / d
    }))
}

impl GazeCheck {
    /// A check starting at `now` (seconds, the app's clock).
    pub fn new(now: f64) -> GazeCheck {
        GazeCheck {
            start: now,
            samples: Default::default(),
            frames: [0; 5],
            good: [0; 5],
            last_time: None,
            last_reading: now,
        }
    }

    /// One frame at `now`, the head turned `head`, `tracking` whether the
    /// headset tracks the eyes and `reading` the raw gaze if one came.
    pub fn update(
        &mut self,
        now: f64,
        head: Quat,
        tracking: bool,
        reading: Option<Reading>,
    ) -> Step {
        if reading.is_some() {
            self.last_reading = now;
        } else if !tracking || now - self.last_reading > TRACKER_GONE {
            return Step::Done(Err(Stop::Tracker));
        }
        let t = now - self.start;
        let i = (t / DOT_TIME).max(0.0) as usize;
        if i >= DOTS.len() {
            return Step::Done(self.finish());
        }
        if t - i as f64 * DOT_TIME >= SETTLE {
            self.frames[i] += 1;
            if let Some(r) = reading
                && r.age.is_none_or(|a| a <= STALE)
            {
                self.good[i] += 1;
                // A sample handed back twice counts once.
                let new = match (r.time, self.last_time) {
                    (Some(t), Some(last)) => t > last,
                    _ => true,
                };
                if new {
                    self.samples[i].push(yaw_pitch(head.inverse() * r.direction));
                    self.last_time = r.time.or(self.last_time);
                }
            }
        }
        Step::Dot(direction(target(i)))
    }

    fn finish(&self) -> Result<Fit, Stop> {
        let mut seen = Vec::new();
        for i in 0..DOTS.len() {
            let s = &self.samples[i];
            if s.is_empty() || (self.good[i] as f32) < MIN_SHARE * self.frames[i] as f32 {
                return Err(Stop::Lost);
            }
            seen.push(Vec2::new(
                median(s.iter().map(|v| v.x).collect()),
                median(s.iter().map(|v| v.y).collect()),
            ));
        }
        let targets: Vec<Vec2> = (0..DOTS.len()).map(target).collect();
        let f = fit(&seen, &targets);
        if f.after > NOT_FOLLOWED {
            return Err(Stop::NotFollowed);
        }
        Ok(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rad(d: [f32; 2]) -> Vec2 {
        Vec2::from(d) * std::f32::consts::PI / 180.0
    }

    fn targets() -> Vec<Vec2> {
        (0..DOTS.len()).map(target).collect()
    }

    #[test]
    fn an_offset_is_found_exactly() {
        let seen: Vec<Vec2> = targets().iter().map(|t| *t - rad([2.0, -1.0])).collect();
        let f = fit(&seen, &targets());
        assert!(f.after < 1e-3, "{f:?}");
        assert!((f.before - 5f32.sqrt()).abs() < 1e-3);
        assert_eq!(
            f.correction.gain,
            Correction::default().gain,
            "an offset alone"
        );
    }

    #[test]
    fn a_tracker_reading_eighty_percent_of_each_turn_takes_the_fuller_map() {
        let seen: Vec<Vec2> = targets().iter().map(|t| *t * 0.8).collect();
        let f = fit(&seen, &targets());
        let offset_alone = mean_error(
            &Correction {
                offset: [0.0; 2],
                ..Correction::default()
            },
            &seen,
            &targets(),
        );
        eprintln!(
            "80% tracker: {offset_alone:.2}° by the offset alone, {:.2}° by the map",
            f.after
        );
        assert!(f.correction.gain[0][0] > 1.1, "{f:?}");
        assert!(f.after < 0.8 && offset_alone > 1.8, "{f:?}");
    }

    #[test]
    fn noise_alone_keeps_to_the_offset() {
        let jitter = [
            [0.3, -0.2],
            [-0.25, 0.1],
            [0.15, 0.3],
            [-0.1, -0.3],
            [0.2, 0.15],
        ];
        let seen: Vec<Vec2> = targets()
            .iter()
            .zip(jitter)
            .map(|(t, j)| *t + rad([1.5, 0.5]) + rad(j))
            .collect();
        let f = fit(&seen, &targets());
        assert_eq!(f.correction.gain, Correction::default().gain, "{f:?}");
    }

    /// A whole check at 90 Hz: the head turning, eyes reaching each dot
    /// `late` seconds after it shows, biased by `bias` degrees, with
    /// `noise` degrees of jitter and blinks of 0.2 s every `blink_every`
    /// seconds; `gone` stops the tracker from then on.
    fn simulate(
        bias: [f32; 2],
        late: f64,
        blink_every: f64,
        gone: Option<f64>,
    ) -> Result<Fit, Stop> {
        let mut c = GazeCheck::new(10.0);
        let mut seed = 99u64;
        let mut noise = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.4
        };
        let mut looking = Vec2::ZERO;
        for f in 1..2000 {
            let t = f as f64 / 90.0;
            let now = 10.0 + t;
            let head = Quat::from_rotation_y((t * 0.4).sin() as f32 * 0.5);
            let i = ((t / DOT_TIME) as usize).min(4);
            if t - i as f64 * DOT_TIME >= late {
                looking = target(i);
            }
            let blinking = blink_every > 0.0 && t % blink_every < 0.2;
            let reading = (!blinking && gone.is_none_or(|g| t < g)).then(|| Reading {
                direction: head * direction(looking + rad(bias) + rad([noise(), noise()])),
                time: Some(now),
                age: Some(0.03),
            });
            if let Step::Done(r) = c.update(now, head, true, reading) {
                return r;
            }
        }
        panic!("never finished");
    }

    #[test]
    fn whole_checks_fit_a_bias_or_say_why_not() {
        // Eyes 0.3 s late to each dot, blinking every 1.1 s, 2° and 1° off.
        let f = simulate([2.0, 1.0], 0.3, 1.1, None).expect("fitted");
        let got = Vec2::from(f.correction.offset) * 180.0 / std::f32::consts::PI;
        eprintln!(
            "bias 2°, 1°: offset {got:?}, {:.2}° -> {:.2}°",
            f.before, f.after
        );
        assert!((got - Vec2::new(-2.0, -1.0)).length() < 0.05, "{got:?}");
        // The tracker stops.
        assert_eq!(simulate([0.0; 2], 0.3, 0.0, Some(4.0)), Err(Stop::Tracker));
        // Blinking most of the time: the eyes are lost too often.
        assert_eq!(simulate([0.0; 2], 0.3, 0.25, None), Err(Stop::Lost));
        // The eyes stay put: nothing kept.
        assert_eq!(simulate([0.0; 2], 99.0, 0.0, None), Err(Stop::NotFollowed));
    }
}
