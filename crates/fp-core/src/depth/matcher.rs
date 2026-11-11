//! Block matching of one measurement's grids: a window of the left
//! camera's grid around the aim slid along the right one's rows, each
//! disparity scored by zero-mean normalised cross-correlation (Szeliski,
//! *Computer Vision: Algorithms and Applications*, ch. 12; Scharstein and
//! Szeliski 2002), coarse first, then fine around the best, then a
//! parabola through the neighbouring scores for the fraction of a sample.
//!
//! The windows move apart symmetrically about the aim, the left one by
//! half the disparity and the right by the rest, so what is measured is
//! what lies along the aim from between the cameras, not beside it. The
//! middle of the window counts for more than its rim (a Gaussian), so a
//! subject at the aim wins over a busier background around it.
//!
//! It takes milliseconds: never call it on the render thread.

use super::geometry::{self, Angles};
use super::grid::{GridSpec, Grids, OUTSIDE, PITCH, SEE_THROUGH};

/// The match window: `size` samples square (a multiple of 4, as it is
/// halved twice for the coarse search), weighted by a Gaussian of `sigma`
/// samples round its middle (infinite: all alike).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Window {
    pub size: usize,
    pub sigma: f32,
}

impl Window {
    /// Aimed by the head: 24 samples, 4.8°, about a face at 2 m. Its rim
    /// counts e^-2 of its middle (sigma a quarter of it): a subject over
    /// most of the middle outweighs a background round it.
    pub const HEAD: Window = Window {
        size: 24,
        sigma: 6.0,
    };
    /// Aimed by the eyes: wider by about the gaze's error left after a
    /// check, near a degree each way: 32 samples, 6.4°.
    pub const EYES: Window = Window {
        size: 32,
        sigma: 8.0,
    };
}

/// Coarse levels below the samples: 2, so the first search is at 0.8°
/// (a quarter of the samples each way) over the whole range, cheap, and
/// the 24-sample window is still 6 there.
const LEVELS: usize = 2;

/// Nearest the aim may be to the baseline's axis, radians: 25°. There the
/// cameras look along their own baseline, a disparity is sin 25° = 0.42 of
/// what it is ahead and the grid's rows squeeze; nearer, measurements are
/// noise.
const MIN_ALPHA: f32 = 25.0 * std::f32::consts::PI / 180.0;

/// Disparities below 0 searched, samples: 0.6°, for a rig whose lenses
/// toe out a little, so its far scenes still match (as beyond infinity).
const NEGATIVE: i32 = 3;

/// Rows the right camera's window may be off the left's while the rig's
/// vertical offset is unknown: ±2°, the misalignment real rigs have.
const FREE_ROWS: i32 = 10;
/// And once it is known: ±0.4°, room for the fit's error and a roll's
/// change across the picture between aims.
const KNOWN_ROWS: i32 = 2;

/// Least weighted luminance spread (standard deviation, 0..1) of the
/// aim's window to match on: 0.01, 2.5 of 255 levels. A plain wall's
/// video noise, averaged over a sample's taps, is under half that; a
/// texture anything can be matched on is several times it.
const MIN_TEXTURE: f32 = 0.01;

/// Least correlation of a match: 0.6. Matching windows of real video
/// score 0.8 to 0.99; the best of several hundred tries between unrelated
/// pictures stays near 0.3 for a window this size.
const MIN_SCORE: f32 = 0.6;

/// A second depth scoring within this of the best is as likely: a
/// repeating pattern, or a subject and background weighing alike.
const AMBIGUITY: f32 = 0.1;

/// Share of the window's weight that must be opaque video to match
/// on: past half see-through, the aim is on the room, not the video.
const OPAQUE_SHARE: f32 = 0.5;

/// How fast the rig model forgets: each new observation's weight 1, the
/// older ones' times 0.98, so about the last 50 count (a second or two
/// of measurements). A fixed rig's offset settles to a hundredth of a
/// degree over that; a video changing under it is forgotten as fast.
const RIG_FADE: f64 = 0.98;
/// Observations, faded, before the rig's offset counts as known.
const RIG_KNOWN: f64 = 8.0;
/// The fit's spread of residuals under which it is trusted, radians:
/// 0.1°, half a sample.
const RIG_SPREAD: f64 = 0.1 * std::f64::consts::PI / 180.0;
/// Least confidence of a measurement to teach the rig model.
const RIG_CONFIDENCE: f32 = 0.5;
/// How strongly the fit's roll term is held at 0: as one observation
/// saying so 13° from the baseline's axis would (where the roll's effect
/// is its whole size), a quarter of one 25° off straight ahead. Aims near
/// the middle say nothing about roll (its effect there is nil); this
/// keeps the fit from inventing one from them, and gives way to a few
/// aims off to the side.
const RIG_ROLL_PRIOR: f64 = 0.05;

/// What to measure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Request {
    /// Where, in the video's axes.
    pub aim: Angles,
    /// The camera's lens spacing, metres.
    pub baseline: f32,
    /// The nearest distance to search, metres: nearer matches aren't
    /// looked for, which is what keeps a repeating pattern from fitting
    /// at a depth nothing is at.
    pub nearest: f32,
    pub window: Window,
}

/// A measurement planned: the grids to sample and how to search them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    pub grid: GridSpec,
    pub request: Request,
    /// Disparities searched, samples, lowest and highest.
    pub disparities: (i32, i32),
    /// Rows the right window is searched off the left, around the rig's
    /// predicted offset, lowest and highest.
    pub rows: (i32, i32),
    /// The vertical offset the right grid was placed with, radians: the
    /// rig model's prediction at the aim.
    pub predicted: f32,
    /// Where each window starts (column, row) at disparity 0 and no
    /// vertical offset.
    left: (usize, usize),
    right: (usize, usize),
}

/// Why a measurement gave no depth, or the depth it gave.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Outcome {
    /// `nearness`, 1/metres from between the cameras (0 at infinity,
    /// below 0 beyond), with `confidence` 0..1.
    Depth { nearness: f32, confidence: f32 },
    /// Nothing to match: a plain wall, darkness.
    Flat,
    /// The two cameras don't show the same thing there: an occlusion, a
    /// highlight.
    Mismatch,
    /// More than one depth fits: a repeating pattern.
    Ambiguous,
    /// Too near the picture's edge or the baseline's axis.
    Edge,
    /// See-through at the aim.
    SeeThrough,
}

/// A measurement's outcome and what it was decided on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Measurement {
    pub outcome: Outcome,
    /// The best correlation, and the best of any other depth (-1: none).
    pub score: f32,
    pub runner_up: f32,
    /// The aim window's luminance spread.
    pub texture: f32,
    /// The disparity found, radians.
    pub disparity: f32,
    /// How far up the right camera's picture was from the left's there,
    /// radians.
    pub vertical: f32,
}

impl Measurement {
    fn failed(outcome: Outcome, texture: f32) -> Measurement {
        Measurement {
            outcome,
            score: 0.0,
            runner_up: -1.0,
            texture,
            disparity: 0.0,
            vertical: 0.0,
        }
    }
}

/// How the right camera's picture sits vertically against the left's,
/// learned from confident matches: a camera pitched about the baseline
/// shifts every direction's `beta` by a constant, one rolled about the
/// forward axis by `roll * cot(alpha) * cos(beta)` (the first-order change
/// of `beta` under a turn about z), so `offset = pitch + roll * cot(alpha)
/// * cos(beta)`, fitted by least squares with older observations fading.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct RigModel {
    /// The fit's normal equations, faded.
    n: [[f64; 2]; 2],
    r: [f64; 2],
    /// Observations, faded, and the sum of their squares.
    count: f64,
    yy: f64,
}

impl RigModel {
    fn features(a: Angles) -> [f64; 2] {
        let (alpha, beta) = (a.alpha as f64, a.beta as f64);
        [1.0, beta.cos() / alpha.tan()]
    }

    fn fit(&self) -> [f64; 2] {
        // A tiny weight on the pitch keeps a fresh model solvable.
        let n = [
            [self.n[0][0] + 1e-6, self.n[0][1]],
            [self.n[1][0], self.n[1][1] + RIG_ROLL_PRIOR],
        ];
        let det = n[0][0] * n[1][1] - n[0][1] * n[1][0];
        if det.abs() < 1e-12 {
            return [0.0; 2];
        }
        [
            (self.r[0] * n[1][1] - self.r[1] * n[0][1]) / det,
            (n[0][0] * self.r[1] - n[1][0] * self.r[0]) / det,
        ]
    }

    /// The predicted vertical offset at `a`, radians (0 untaught).
    fn predict(&self, a: Angles) -> f32 {
        let [p, roll] = self.fit();
        let x = Self::features(a);
        (p * x[0] + roll * x[1]) as f32
    }

    fn add(&mut self, a: Angles, offset: f32) {
        let x = Self::features(a);
        let y = offset as f64;
        for (i, xi) in x.iter().enumerate() {
            for (j, xj) in x.iter().enumerate() {
                self.n[i][j] = self.n[i][j] * RIG_FADE + xi * xj;
            }
            self.r[i] = self.r[i] * RIG_FADE + xi * y;
        }
        self.count = self.count * RIG_FADE + 1.0;
        self.yy = self.yy * RIG_FADE + y * y;
    }

    /// The observations' spread about the fit, radians (root mean square).
    fn spread(&self) -> f64 {
        let t = self.fit();
        let fitted = t[0] * self.r[0] + t[1] * self.r[1];
        let model = (0..2)
            .map(|i| (0..2).map(|j| t[i] * self.n[i][j] * t[j]).sum::<f64>())
            .sum::<f64>();
        // Sum of (y - t.x)^2 = yy - 2 t.r + t'Nt.
        ((self.yy - 2.0 * fitted + model).max(0.0) / self.count.max(1e-9)).sqrt()
    }

    fn known(&self) -> bool {
        self.count >= RIG_KNOWN && self.spread() < RIG_SPREAD
    }
}

/// Matches measurements, learning the rig's vertical offset as it goes.
/// Start a new one, or [`Matcher::forget_rig`], when the video, its
/// layout, the eye swap or the lens spacing change.
#[derive(Clone, Debug, Default)]
pub struct Matcher {
    rig: RigModel,
}

/// One level of the search: both grids, halved `level` times.
struct Level {
    cols: usize,
    rows: usize,
    samples: [Vec<f32>; 2],
}

impl Level {
    /// Two by two samples into one: outside if any is, see-through if
    /// most are, else the mean of the luminance ones.
    fn halved(&self) -> Level {
        let (cols, rows) = (self.cols / 2, self.rows / 2);
        let samples = [0, 1].map(|g| {
            let s = &self.samples[g];
            let mut out = Vec::with_capacity(cols * rows);
            for r in 0..rows {
                for c in 0..cols {
                    let four = [
                        s[2 * r * self.cols + 2 * c],
                        s[2 * r * self.cols + 2 * c + 1],
                        s[(2 * r + 1) * self.cols + 2 * c],
                        s[(2 * r + 1) * self.cols + 2 * c + 1],
                    ];
                    let luma: Vec<f32> = four.iter().copied().filter(|v| *v >= 0.0).collect();
                    out.push(if four.contains(&OUTSIDE) {
                        OUTSIDE
                    } else if luma.len() < 2 {
                        SEE_THROUGH
                    } else {
                        luma.iter().sum::<f32>() / luma.len() as f32
                    });
                }
            }
            out
        });
        Level {
            cols,
            rows,
            samples,
        }
    }
}

/// Weighted sums of a window pair, for its correlation.
#[derive(Default)]
struct Sums {
    w: f32,
    a: f32,
    b: f32,
    aa: f32,
    bb: f32,
    ab: f32,
}

impl Sums {
    fn correlation(&self) -> f32 {
        let va = self.aa - self.a * self.a / self.w;
        let vb = self.bb - self.b * self.b / self.w;
        let cov = self.ab - self.a * self.b / self.w;
        if va <= 1e-12 || vb <= 1e-12 {
            return 0.0;
        }
        cov / (va * vb).sqrt()
    }
}

/// A Gaussian window of `size` samples square and `sigma`.
fn weights(size: usize, sigma: f32) -> Vec<f32> {
    let c = (size as f32 - 1.0) / 2.0;
    let mut w = Vec::with_capacity(size * size);
    for r in 0..size {
        for col in 0..size {
            let d2 = (r as f32 - c).powi(2) + (col as f32 - c).powi(2);
            w.push(if sigma.is_finite() {
                (-d2 / (2.0 * sigma * sigma)).exp()
            } else {
                1.0
            });
        }
    }
    w
}

/// The search over one measurement.
struct Search<'a> {
    plan: &'a Plan,
    levels: Vec<Level>,
    weights: Vec<Vec<f32>>,
    total: Vec<f32>,
}

impl Search<'_> {
    /// Where the left window starts at disparity `d` (level 0): half of it
    /// to the left of the aim, rounded down.
    fn left(&self, d: i32) -> (i64, i64) {
        (
            self.plan.left.0 as i64 - d.div_euclid(2) as i64,
            self.plan.left.1 as i64,
        )
    }

    /// And the right window, `v` rows off: the rest of the disparity.
    fn right(&self, d: i32, v: i32) -> (i64, i64) {
        (
            self.plan.right.0 as i64 + (d - d.div_euclid(2)) as i64,
            self.plan.right.1 as i64 + v as i64,
        )
    }

    /// The correlation of the left window at `l` and the right at `r`, at
    /// level `k` (positions in its samples); `None` if either window leaves
    /// its grid or touches the outside of its picture, or too little of
    /// them is opaque video.
    fn score_at(&self, k: usize, l: (i64, i64), r: (i64, i64)) -> Option<f32> {
        let level = &self.levels[k];
        let size = self.plan.request.window.size >> k;
        let inside = |(c, row): (i64, i64)| {
            c >= 0
                && row >= 0
                && c as usize + size <= level.cols
                && row as usize + size <= level.rows
        };
        if !inside(l) || !inside(r) {
            return None;
        }
        let (a, b) = (&level.samples[0], &level.samples[1]);
        let w = &self.weights[k];
        let mut s = Sums::default();
        for row in 0..size {
            let ia = (l.1 as usize + row) * level.cols + l.0 as usize;
            let ib = (r.1 as usize + row) * level.cols + r.0 as usize;
            for col in 0..size {
                let (x, y) = (a[ia + col], b[ib + col]);
                if x == OUTSIDE || y == OUTSIDE {
                    return None;
                }
                if x < 0.0 || y < 0.0 {
                    continue;
                }
                let wt = w[row * size + col];
                s.w += wt;
                s.a += wt * x;
                s.b += wt * y;
                s.aa += wt * x * x;
                s.bb += wt * y * y;
                s.ab += wt * x * y;
            }
        }
        (s.w >= OPAQUE_SHARE * self.total[k]).then(|| s.correlation())
    }

    /// The correlation at disparity `d` and `v` rows (level-0 units),
    /// looked at on level `k`.
    fn score(&self, k: usize, d: i32, v: i32) -> Option<f32> {
        let (l, r) = (self.left(d), self.right(d, v));
        let down = |(c, row): (i64, i64)| (c >> k, row >> k);
        self.score_at(k, down(l), down(r))
    }

    /// The best (score, d, v) over `ds` x `vs` on level `k`.
    fn best(
        &self,
        k: usize,
        ds: impl Iterator<Item = i32> + Clone,
        vs: impl Iterator<Item = i32> + Clone,
    ) -> Option<(f32, i32, i32)> {
        let mut best: Option<(f32, i32, i32)> = None;
        for d in ds {
            for v in vs.clone() {
                if let Some(s) = self.score(k, d, v)
                    && best.is_none_or(|b| s > b.0)
                {
                    best = Some((s, d, v));
                }
            }
        }
        best
    }
}

/// Where the peak of three scores at -1, 0 and 1 is, -0.5..0.5: the
/// vertex of the parabola through them.
fn vertex(minus: Option<f32>, mid: f32, plus: Option<f32>) -> f32 {
    let (Some(m), Some(p)) = (minus, plus) else {
        return 0.0;
    };
    let curve = m - 2.0 * mid + p;
    if curve >= 0.0 {
        return 0.0;
    }
    (0.5 * (m - p) / curve).clamp(-0.5, 0.5)
}

/// 0 at `lo`, 1 at `hi`, straight between.
fn ramp(x: f32, lo: f32, hi: f32) -> f32 {
    ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
}

impl Matcher {
    pub fn new() -> Matcher {
        Matcher::default()
    }

    /// Forgets the rig's vertical offset: for another video, layout, eye
    /// swap or lens spacing.
    pub fn forget_rig(&mut self) {
        self.rig = RigModel::default();
    }

    /// The rig's vertical offset at `aim`, radians, once it is known.
    pub fn rig_offset(&self, aim: Angles) -> Option<f32> {
        self.rig.known().then(|| self.rig.predict(aim))
    }

    /// What to sample for `req`; `Err` with the outcome when it can't be
    /// measured at all (too near the baseline's axis).
    pub fn plan(&self, req: &Request) -> Result<Plan, Outcome> {
        let aim = req.aim;
        if !(MIN_ALPHA..=std::f32::consts::PI - MIN_ALPHA).contains(&aim.alpha)
            || !aim.alpha.is_finite()
            || !aim.beta.is_finite()
        {
            return Err(Outcome::Edge);
        }
        let size = req.window.size;
        let max = geometry::disparity(1.0 / req.nearest.max(0.05), aim.alpha, req.baseline);
        let disparities = (-NEGATIVE, (max / PITCH).ceil() as i32 + 1);
        let known = self.rig.known();
        let free = if known { KNOWN_ROWS } else { FREE_ROWS };
        let rows = (-free, free);
        let predicted = if known { self.rig.predict(aim) } else { 0.0 };
        // At disparity d the left window sits floor(d/2) samples below the
        // aim in alpha and the right one the rest above it, a sample
        // further either way for the parabola, and up or down by the rows
        // searched.
        let (lo, hi) = disparities;
        let left_col = hi.div_euclid(2) as usize;
        let right_col = 1 + (-lo).div_euclid(2).max(0) as usize;
        let cols = (left_col + (-lo).div_euclid(2) as usize + 1)
            .max(right_col + (hi - hi.div_euclid(2)) as usize + 1)
            + size;
        let row = (1 + free) as usize;
        let grid_rows = row + free as usize + 1 + size;
        let mid = (size as f32 - 1.0) / 2.0;
        let origin = |col: usize, offset: f32| Angles {
            alpha: aim.alpha - (col as f32 + mid) * PITCH,
            beta: aim.beta + offset - (row as f32 + mid) * PITCH,
        };
        Ok(Plan {
            grid: GridSpec {
                pitch: PITCH,
                cols,
                rows: grid_rows,
                origin: [origin(left_col, 0.0), origin(right_col, predicted)],
            },
            request: *req,
            disparities,
            rows,
            predicted,
            left: (left_col, row),
            right: (right_col, row),
        })
    }

    /// Matches `grids`, sampled as `plan` said.
    pub fn measure(&mut self, plan: &Plan, grids: &Grids) -> Measurement {
        let size = plan.request.window.size;
        let spec = &plan.grid;
        if grids.spec != *spec || grids.samples.iter().any(|s| s.len() != spec.len()) {
            return Measurement::failed(Outcome::Edge, 0.0);
        }
        let mut levels = vec![Level {
            cols: spec.cols,
            rows: spec.rows,
            samples: grids.samples.clone(),
        }];
        for k in 0..LEVELS {
            let next = levels[k].halved();
            levels.push(next);
        }
        let weights: Vec<Vec<f32>> = (0..=LEVELS)
            .map(|k| weights(size >> k, plan.request.window.sigma / (1 << k) as f32))
            .collect();
        let total = weights.iter().map(|w| w.iter().sum()).collect();
        let search = Search {
            plan,
            levels,
            weights,
            total,
        };

        // The aim's own window, as the left camera saw it.
        let (l0, w0) = (search.left(0), &search.weights[0]);
        let left = &grids.samples[0];
        let (mut sw, mut opaque, mut sum, mut sq, mut outside) = (0.0, 0.0, 0.0, 0.0, false);
        for row in 0..size {
            for col in 0..size {
                let v = left[(l0.1 as usize + row) * spec.cols + l0.0 as usize + col];
                let wt = w0[row * size + col];
                sw += wt;
                outside |= v == OUTSIDE;
                if v >= 0.0 {
                    opaque += wt;
                    sum += wt * v;
                    sq += wt * v * v;
                }
            }
        }
        if outside {
            return Measurement::failed(Outcome::Edge, 0.0);
        }
        if opaque < OPAQUE_SHARE * sw {
            return Measurement::failed(Outcome::SeeThrough, 0.0);
        }
        let mean = sum / opaque;
        let texture = (sq / opaque - mean * mean).max(0.0).sqrt();
        if texture < MIN_TEXTURE {
            return Measurement::failed(Outcome::Flat, texture);
        }

        // Coarse: everything, on the top level.
        let (dl, dh) = plan.disparities;
        let (vl, vh) = plan.rows;
        let top = LEVELS;
        let step = 1i32 << top;
        let span = |lo: i32, hi: i32, step: i32| {
            (lo.div_euclid(step)..=hi.div_euclid(step) + 1).map(move |i| (i * step).clamp(lo, hi))
        };
        let Some(coarse) = search.best(top, span(dl, dh, step), span(vl, vh, step)) else {
            return Measurement::failed(Outcome::Edge, texture);
        };
        // Then down the levels, around it.
        let (mut d, mut v) = (coarse.1, coarse.2);
        for k in (0..top).rev() {
            let s = 1i32 << k;
            let around =
                |c: i32, lo: i32, hi: i32| (-2..=2).map(move |j| (c + j * s).clamp(lo, hi));
            if let Some(b) = search.best(k, around(d, dl, dh), around(v, vl, vh)) {
                (d, v) = (b.1, b.2);
            }
        }
        // Every disparity on the samples, at that row: the coarse levels
        // can miss a narrow peak, and the curve shows any rival.
        let (mut curve, mut score);
        let mut tries = 0;
        loop {
            curve = (dl..=dh).map(|d| search.score(0, d, v)).collect::<Vec<_>>();
            let top = curve
                .iter()
                .enumerate()
                .filter_map(|(i, s)| s.map(|s| (s, dl + i as i32)))
                .max_by(|a, b| a.0.total_cmp(&b.0));
            let Some(top) = top else {
                return Measurement::failed(Outcome::Edge, texture);
            };
            (score, d) = top;
            // A better row at that disparity: search the curve there too.
            tries += 1;
            match search.best(0, std::iter::once(d), (v - 1).max(vl)..=(v + 1).min(vh)) {
                Some(b) if b.2 != v && b.0 > score && tries < 3 => v = b.2,
                _ => break,
            }
        }
        let at = |i: i32| curve.get((i - dl) as usize).copied().flatten();
        // The best of the curve's other peaks.
        let mut runner_up = -1.0f32;
        for i in dl..=dh {
            if (i - d).abs() < 3 {
                continue;
            }
            if let Some(s) = at(i)
                && at(i - 1).is_none_or(|p| s >= p)
                && at(i + 1).is_none_or(|n| s >= n)
            {
                runner_up = runner_up.max(s);
            }
        }

        // The fraction of a sample, with the left window held where it is
        // so the three scores compare like with like.
        let l = search.left(d);
        let r = search.right(d, v);
        let shift = |dc: i64, dr: i64| search.score_at(0, l, (r.0 + dc, r.1 + dr));
        let x = vertex(shift(-1, 0), score, shift(1, 0));
        let y = vertex(shift(0, -1), score, shift(0, 1));
        let mid = (size as f32 - 1.0) / 2.0;
        let alpha_left = spec.angles(0, l.0 as f32 + mid, 0.0).alpha;
        let alpha_right = spec.angles(1, r.0 as f32 + x + mid, 0.0).alpha;
        let vertical = plan.predicted + (v as f32 + y) * spec.pitch;
        let disparity = alpha_right - alpha_left;
        let nearness = geometry::nearness(alpha_left, alpha_right, plan.request.baseline);

        let outcome = if score < MIN_SCORE {
            Outcome::Mismatch
        } else if runner_up >= score - AMBIGUITY {
            Outcome::Ambiguous
        } else {
            let confidence = ramp(score, MIN_SCORE, 0.95)
                * ramp(score - runner_up, AMBIGUITY, AMBIGUITY + 0.3)
                * ramp(texture, MIN_TEXTURE, 4.0 * MIN_TEXTURE);
            Outcome::Depth {
                nearness,
                confidence,
            }
        };
        // Teach the rig model; a match at the edge of a narrowed search
        // says the model is wrong (another video under it).
        if let Outcome::Depth { confidence, .. } = outcome {
            let narrowed = plan.rows.1 == KNOWN_ROWS;
            if narrowed && (v == vl || v == vh) {
                self.forget_rig();
            } else if confidence >= RIG_CONFIDENCE && v > vl && v < vh {
                // Where the right camera saw it: a roll's offset changes
                // with alpha, by 0.04° over a metre's disparity.
                let seen = Angles {
                    alpha: alpha_right,
                    beta: plan.request.aim.beta,
                };
                self.rig.add(seen, vertical);
            }
        }
        Measurement {
            outcome,
            score,
            runner_up,
            texture,
            disparity,
            vertical,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::depth::grid;
    use crate::depth::picture::PictureGeometry;
    use crate::depth::scenes::{Picture, Rig, Scene, Shape, Surface, Texture, turn};
    use crate::format::{Projection, StereoLayout, VideoFormat};
    use crate::view::ViewSettings;
    use std::f32::consts::FRAC_PI_2;

    const B: f32 = 0.065;

    /// A frame about a thousand pixels a picture, side by side or top and
    /// bottom.
    fn layout(projection: Projection, stereo: StereoLayout) -> (PictureGeometry, usize, usize) {
        let (w, h) = if stereo.is_side_by_side() {
            (2000, 1000)
        } else {
            (1000, 2000)
        };
        let g = PictureGeometry::new(
            &VideoFormat::new(projection, stereo),
            &ViewSettings::default(),
            w as u32,
            h as u32,
            1.0,
            0,
        );
        (g, w, h)
    }

    fn sbs() -> (PictureGeometry, usize, usize) {
        layout(Projection::EQUIRECT_180, StereoLayout::SideBySide)
    }

    /// Texture whose detail is about the same size seen from the cameras
    /// at `distance`: finest 0.34°, near two samples, as an 8K picture's
    /// detail is once a sample's taps have averaged it.
    fn texture(distance: f32, seed: u32) -> Texture {
        Texture::Noise {
            feature: 0.006 * distance,
            contrast: 1.0,
            seed,
        }
    }

    fn aim(yaw_deg: f32, pitch_deg: f32) -> Angles {
        Angles {
            alpha: FRAC_PI_2 - yaw_deg.to_radians(),
            beta: pitch_deg.to_radians(),
        }
    }

    fn request(aim: Angles, nearest: f32) -> Request {
        Request {
            aim,
            baseline: B,
            nearest,
            window: Window::HEAD,
        }
    }

    fn measure(m: &mut Matcher, picture: &Picture, req: Request) -> Measurement {
        let plan = m.plan(&req).expect("plan");
        let grids = grid::sample(picture, &picture.geometry, &plan.grid);
        m.measure(&plan, &grids)
    }

    fn nearness(m: &Measurement) -> f32 {
        match m.outcome {
            Outcome::Depth { nearness, .. } => nearness,
            other => panic!("no depth: {other:?} ({m:?})"),
        }
    }

    /// One resolution step at `nearness` along `aim`, 1/metres.
    fn step(nearness: f32, aim: Angles) -> f32 {
        let d = geometry::disparity(nearness, aim.alpha, B);
        let a = FRAC_PI_2 - d / 2.0;
        (geometry::nearness(a, a + d + crate::depth::RESOLUTION, B) - nearness).abs()
    }

    #[test]
    fn textured_planes_are_measured_at_their_distance_everywhere() {
        let aims = [
            (0.0, 0.0),
            (25.0, 0.0),
            (-25.0, 0.0),
            (0.0, 22.0),
            (0.0, -22.0),
        ];
        let (mut worst, mut lowest): (f32, f32) = (0.0, 1.0);
        for projection in [Projection::EQUIRECT_180, Projection::fisheye(180.0)] {
            for stereo in [StereoLayout::SideBySide, StereoLayout::TopBottom] {
                let (g, w, h) = layout(projection, stereo);
                for distance in [0.3, 0.6, 1.0, 2.0, 5.0] {
                    let scene = Scene::new(vec![Surface::wall(distance, texture(distance, 7))]);
                    let picture = Picture::new([&scene, &scene], Rig::new(B), g, w, h);
                    for (yaw, pitch) in aims {
                        let a = aim(yaw, pitch);
                        let mut m = Matcher::new();
                        let got = measure(&mut m, &picture, request(a, 0.25));
                        // Along the aim from between the cameras.
                        let want = -a.direction()[2] / distance;
                        let off = (nearness(&got) - want).abs() / step(want, a);
                        worst = worst.max(off);
                        let label =
                            format!("{projection:?} {stereo:?} {distance} m at {yaw}° {pitch}°");
                        assert!(
                            off <= 1.0,
                            "{label}: {} m, {off:.2} steps off ({got:?})",
                            1.0 / nearness(&got)
                        );
                        let Outcome::Depth { confidence, .. } = got.outcome else {
                            unreachable!()
                        };
                        lowest = lowest.min(confidence);
                        assert!(
                            confidence > 0.3,
                            "{label}: confidence {confidence} ({got:?})"
                        );
                    }
                }
            }
        }
        eprintln!("planes: within {worst:.2} resolution steps, confidence at least {lowest:.2}");
    }

    #[test]
    fn each_reason_for_no_depth_is_told_apart() {
        let (g, w, h) = sbs();
        let one = |scenes: [&Scene; 2], a: Angles| {
            let picture = Picture::new(scenes, Rig::new(B), g, w, h);
            measure(&mut Matcher::new(), &picture, request(a, 0.25)).outcome
        };
        let ahead = aim(0.0, 0.0);
        // A plain wall, and a dark one with a trace of texture.
        let plain = Scene::new(vec![Surface::wall(1.5, Texture::Plain(0.4))]);
        assert_eq!(one([&plain, &plain], ahead), Outcome::Flat);
        let dark = Scene::new(vec![Surface::wall(
            1.5,
            Texture::Noise {
                feature: 0.02,
                contrast: 0.02,
                seed: 3,
            },
        )]);
        assert_eq!(one([&dark, &dark], ahead), Outcome::Flat);
        // Each camera seeing something else.
        // (Lattices of another size, so they don't line up.)
        let a = Scene::new(vec![Surface::wall(1.0, texture(1.0, 1))]);
        for (scale, seed) in [(1.37, 1000), (0.8, 2000), (1.0, 3000), (2.0, 4000)] {
            let b = Scene::new(vec![Surface::wall(1.0, texture(scale, seed))]);
            let picture = Picture::new([&a, &b], Rig::new(B), g, w, h);
            let got = measure(&mut Matcher::new(), &picture, request(ahead, 0.25));
            eprintln!("unrelated {scale} {seed}: {got:?}");
        }
        let b = Scene::new(vec![Surface::wall(1.0, texture(1.37, 1000))]);
        assert_eq!(one([&a, &b], ahead), Outcome::Mismatch);
        // Stripes that fit at several depths.
        let stripes = Scene::new(vec![Surface::wall(
            2.0,
            Texture::Repeating {
                period: 0.12,
                feature: 0.03,
                contrast: 1.0,
                seed: 5,
            },
        )]);
        assert_eq!(one([&stripes, &stripes], ahead), Outcome::Ambiguous);
        // Near the top edge of the picture, and near the baseline's axis.
        let wall = Scene::new(vec![Surface::wall(2.0, texture(2.0, 4))]);
        assert_eq!(one([&wall, &wall], aim(0.0, 88.0)), Outcome::Edge);
        let m = Matcher::new();
        assert_eq!(
            m.plan(&request(aim(70.0, 0.0), 0.25)).err(),
            Some(Outcome::Edge)
        );
        // A see-through hole at the aim; a small see-through spot doesn't
        // matter.
        let hole = |radius| {
            let mut s = Scene::new(vec![Surface::wall(1.5, texture(1.5, 6))]);
            s.surfaces.insert(
                0,
                Surface {
                    shape: Shape::Disc {
                        centre: [0.0, 0.0, -1.4],
                        normal: [0.0, 0.0, 1.0],
                        radius,
                    },
                    texture: Texture::Plain(0.1),
                    see_through: true,
                },
            );
            s
        };
        let big = hole(0.2);
        assert_eq!(one([&big, &big], ahead), Outcome::SeeThrough);
        let spot = hole(0.012);
        let got = one([&spot, &spot], ahead);
        assert!(matches!(got, Outcome::Depth { .. }), "{got:?}");
    }

    #[test]
    fn the_subject_at_the_aim_wins_over_a_busier_background() {
        let (g, w, h) = sbs();
        // A disc 1 m away over about 60% of the window, in front of a
        // wall at 3 m with more contrast and finer detail.
        let window_deg = Window::HEAD.size as f32 * 0.2;
        let radius = (0.6 * window_deg * window_deg / std::f32::consts::PI)
            .sqrt()
            .to_radians();
        let scene = Scene::new(vec![
            Surface::new(
                Shape::Disc {
                    centre: [0.0, 0.0, -1.0],
                    normal: [0.0, 0.0, 1.0],
                    radius: radius.tan(),
                },
                Texture::Noise {
                    feature: 0.006,
                    contrast: 0.8,
                    seed: 11,
                },
            ),
            Surface::wall(
                3.0,
                Texture::Noise {
                    feature: 0.012,
                    contrast: 1.0,
                    seed: 12,
                },
            ),
        ]);
        let picture = Picture::new([&scene, &scene], Rig::new(B), g, w, h);
        let ahead = aim(0.0, 0.0);
        let got = measure(&mut Matcher::new(), &picture, request(ahead, 0.25));
        eprintln!("subject weighted to the middle: {got:?}");
        assert!((nearness(&got) - 1.0).abs() < step(1.0, ahead), "{got:?}");
        // Every sample alike: the background weighs as much.
        let flat = Request {
            window: Window {
                size: Window::HEAD.size,
                sigma: f32::INFINITY,
            },
            ..request(ahead, 0.25)
        };
        let got = measure(&mut Matcher::new(), &picture, flat);
        eprintln!("subject with every sample alike: {:?}", got.outcome);
        assert!(
            !matches!(got.outcome, Outcome::Depth { nearness, .. } if (nearness - 1.0).abs() < 0.1),
            "{got:?}"
        );
    }

    /// A world direction as a camera turned `right_turn` sees it.
    fn turned_back(d: [f32; 3], right_turn: [f32; 3]) -> [f32; 3] {
        let [p, y, r] = right_turn;
        turn(
            turn(turn(d, [0.0, 0.0, -r]), [0.0, -y, 0.0]),
            [-p, 0.0, 0.0],
        )
    }

    /// The vertical offset at `a` of a right camera turned `right_turn`,
    /// for a far point: the turn's change of `beta`.
    fn true_offset(a: Angles, right_turn: [f32; 3]) -> f32 {
        Angles::of(turned_back(a.direction(), right_turn)).beta - a.beta
    }

    /// The nearness the turned rig `rig` gives a wall `distance` away along
    /// `a`: a rolled right camera also moves things sideways, by its roll
    /// times sin(beta), which reads as disparity.
    fn rig_nearness(a: Angles, distance: f32, rig: Rig) -> f32 {
        let d = a.direction();
        let p = d.map(|c| c * distance / -d[2]);
        let half = rig.baseline / 2.0;
        let left = Angles::of([p[0] + half, p[1], p[2]]).alpha;
        let right = Angles::of(turned_back([p[0] - half, p[1], p[2]], rig.right_turn)).alpha;
        geometry::nearness(left, right, rig.baseline)
    }

    #[test]
    fn a_rigs_vertical_offset_is_learned_and_narrows_the_search() {
        let (g, w, h) = sbs();
        let scene = Scene::new(vec![Surface::wall(1.2, texture(1.2, 21))]);
        for right_turn in [[1.0, 0.0, 0.0], [0.0, 0.0, 1.5]] {
            let rig = Rig {
                baseline: B,
                right_turn,
            };
            let picture = Picture::new([&scene, &scene], rig, g, w, h);
            let mut m = Matcher::new();
            let aims = [
                aim(0.0, 0.0),
                aim(25.0, 10.0),
                aim(-25.0, -10.0),
                aim(20.0, -15.0),
            ];
            for i in 0..16 {
                let a = aims[i % aims.len()];
                let got = measure(&mut m, &picture, request(a, 0.25));
                let want = rig_nearness(a, 1.2, rig);
                assert!(
                    (nearness(&got) - want).abs() < step(want, a),
                    "{right_turn:?}: {got:?}"
                );
            }
            let probe = aim(10.0, 5.0);
            let plan = m.plan(&request(probe, 0.25)).unwrap();
            assert_eq!(
                plan.rows,
                (-KNOWN_ROWS, KNOWN_ROWS),
                "{right_turn:?}: narrowed"
            );
            let learned = m.rig_offset(probe).expect("known");
            let want = true_offset(probe, right_turn);
            eprintln!(
                "rig {right_turn:?}: learned {:.4}°, true {:.4}°",
                learned.to_degrees(),
                want.to_degrees()
            );
            assert!((learned - want).abs().to_degrees() < 0.02, "{right_turn:?}");
            // Narrowed, it still finds the depth; forgotten, it searches wide.
            let got = measure(&mut m, &picture, request(probe, 0.25));
            assert!(matches!(got.outcome, Outcome::Depth { .. }));
            m.forget_rig();
            assert_eq!(
                m.plan(&request(probe, 0.25)).unwrap().rows,
                (-FREE_ROWS, FREE_ROWS)
            );
        }
    }

    #[test]
    fn searching_no_nearer_than_followed_makes_a_repeating_wall_measurable() {
        let (g, w, h) = sbs();
        let ahead = aim(0.0, 0.0);
        // A pattern on a wall 2 m away that repeats where it would also fit
        // a wall 0.25 m away.
        let shift = geometry::disparity(4.0, FRAC_PI_2, B) - geometry::disparity(0.5, FRAC_PI_2, B);
        let scene = Scene::new(vec![Surface::wall(
            2.0,
            Texture::Repeating {
                period: 2.0 * shift.tan(),
                feature: 0.025,
                contrast: 1.0,
                seed: 31,
            },
        )]);
        let picture = Picture::new([&scene, &scene], Rig::new(B), g, w, h);
        let every = measure(&mut Matcher::new(), &picture, request(ahead, 0.22));
        assert_eq!(every.outcome, Outcome::Ambiguous, "{every:?}");
        // Following no nearer than 0.4 m, searching to 0.32 m.
        let near = measure(&mut Matcher::new(), &picture, request(ahead, 0.32));
        assert!((nearness(&near) - 0.5).abs() < step(0.5, ahead), "{near:?}");
    }

    #[test]
    fn a_slowly_receding_wall_reads_steadily() {
        let (g, w, h) = sbs();
        let ahead = aim(0.0, 0.0);
        let mut last = f32::INFINITY;
        let mut m = Matcher::new();
        for i in 0..24 {
            let distance = 1.0 + 0.004 * i as f32;
            // The same texture, seen from a little farther each time.
            let scene = Scene::new(vec![Surface::wall(distance, texture(1.0, 41))]);
            let picture = Picture::new([&scene, &scene], Rig::new(B), g, w, h);
            let n = nearness(&measure(&mut m, &picture, request(ahead, 0.25)));
            assert!(
                (n - 1.0 / distance).abs() < step(1.0, ahead),
                "{distance} m: {}",
                1.0 / n
            );
            // Never back towards where it was, past rounding.
            assert!(n <= last + 0.002, "{distance} m: {n} after {last}");
            last = n;
        }
    }
}
