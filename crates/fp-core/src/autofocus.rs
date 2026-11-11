//! Automatic stereo focus: turns depth measurements ([`crate::depth`])
//! into the distance to draw at. A pure state machine: the caller ticks
//! it with the time and the playback position, asks it what to measure
//! next, and hands back each result, so every behaviour is a script.
//!
//! Its arithmetic is in nearness, 1/metres: for eyes a fixed distance
//! apart the convergence angle is that distance times it (to within 1% from
//! 0.3 m on), so equal steps in it are equal steps in what the eyes feel,
//! and depth resolution is even in it.

use crate::depth::Outcome;
use crate::view::FOCUS_DISTANCES;

/// Where around the aim a measurement is taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Centre,
    /// [`SIDE`] to the left of the aim, in the video's axes.
    Left,
    Right,
}

/// How far either side of the aim the side measurements are, radians: 2°.
/// The head's window is 4.8° across with its weight within about 1.2° of
/// the middle, so a side window overlaps the aim's but weighs what lies
/// past its edge: at a subject's edge one side sees the subject and the
/// other what is behind it.
pub const SIDE: f32 = 2.0 * std::f32::consts::PI / 180.0;

/// Two measurements agree within this, 1/metres: 0.04, three resolution
/// steps at 65 mm, under a sixth of a degree of convergence for 63 mm eyes
/// (1 m against 1.04 m, 3 m against 3.4 m). It is what changing the focus
/// for would barely show: an eye 3 cm off its camera is drawn 0.07° off by
/// it.
pub const AGREE: f32 = 0.04;

/// Least confidence of a measurement to count: 0.2. Textured surfaces
/// measure 0.75 or more; a subject before a busier background 0.3; a
/// chance match between unrelated pictures, when it gets past the
/// matcher, 0.3 at most, and two of them agreeing is what it would take.
pub const MIN_CONFIDENCE: f32 = 0.2;

/// How long the shown distance takes to glide most of the way to a new
/// one, seconds: the time constant of a first-order glide in nearness, at
/// speed 1. The eyes' own vergence takes a few hundred milliseconds to a
/// second to re-converge; at 0.25 s, 90% of a change is made in 0.58 s and
/// 99% in 1.15 s. Shorter, the picture's depth jumps ahead of the eyes;
/// longer, it lags after they have moved.
pub const GLIDE: f32 = 0.25;

/// The longest time step a glide takes, seconds: a stall or a dropped
/// frame moves it no more than a tenth of a second would.
pub const MAX_STEP: f32 = 0.1;

/// How much a centre measurement agreeing with the focus moves it towards
/// itself: an average over about the last three, steady against the
/// matcher's noise while following a subject moving slowly.
const REFINE: f32 = 0.3;
/// The same aimed by the eyes, which pursue a moving thing: over about the
/// last one and a half, following it twice as closely.
const REFINE_EYES: f32 = 0.6;

/// How old a side's measurement may be and still bear out (or speak
/// against) a change, seconds: a few cycles of measurements.
const FRESH: f64 = 0.5;

/// A jump forward of the playback position more than this many times the
/// time passed, plus [`SEEK_SLACK`] seconds, is a seek: fast-forward up to
/// 4x isn't, nor is a frame arriving late.
const SEEK_SPEED: f64 = 4.0;
const SEEK_SLACK: f64 = 1.0;
/// A step back of more than this, seconds, is a seek (or a loop).
const SEEK_BACK: f64 = 0.25;

/// Two coarse pictures correlating less than this are a cut: consecutive
/// measurements a few frames apart correlate 0.9 or more through pans and
/// fades.
const CUT_CORRELATION: f32 = 0.5;
/// Pictures with less spread than this, 0..1, are told apart by their
/// mean instead (a fade to black), which changes by over [`CUT_MEAN`] at
/// a cut.
const FLAT_PICTURE: f32 = 0.03;
const CUT_MEAN: f32 = 0.15;

/// A request's tag: results of requests asked before a start-over are
/// told by it and ignored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tag {
    generation: u64,
    pub side: Side,
}

/// The settings it follows by (per video, see `ViewSettings`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Follow {
    /// Glide speed, 1 neutral.
    pub speed: f32,
    /// The nearest and farthest distances it follows, metres; outside
    /// them it holds. Either at the end of [`FOCUS_DISTANCES`] is no
    /// limit: it follows there, as far as that end.
    pub nearest: f32,
    pub farthest: f32,
    /// The eyes aim it (not the head): small changes are followed faster.
    pub eyes: bool,
}

impl Default for Follow {
    fn default() -> Follow {
        Follow {
            speed: 1.0,
            nearest: *FOCUS_DISTANCES.start(),
            farthest: *FOCUS_DISTANCES.end(),
            eyes: false,
        }
    }
}

/// Why it is holding where it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hold {
    /// Nothing to match there (a plain wall, darkness).
    Flat,
    /// A repeating pattern: more than one depth fits.
    Repeating,
    /// The two eyes don't show the same thing.
    Mismatch,
    /// Too near the edge of the picture.
    Edge,
    /// See-through there.
    SeeThrough,
    /// What is there is nearer than it follows, or farther.
    TooNear,
    TooFar,
}

/// What it is doing, for the status row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Activity {
    /// Nothing to go on yet: waiting for two measurements to agree.
    Measuring,
    /// Following what is at the aim.
    Following,
    /// Holding where it is.
    Holding(Hold),
}

/// A good measurement: metres and how sure, 0..1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reading {
    pub distance: f32,
    pub confidence: f32,
}

/// One measurement as it came back.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Got {
    side: Side,
    time: f64,
    /// Nearness, when it gave one confidently enough.
    nearness: Option<f32>,
}

/// The focus controller of one open video.
#[derive(Clone, Debug)]
pub struct Controller {
    generation: u64,
    /// Nearness shown, and what it glides to (`None`: nothing to go on).
    shown: f32,
    accepted: Option<f32>,
    /// Recent measurements, newest last.
    recent: Vec<Got>,
    /// The order measurements are asked for in.
    cycle: usize,
    activity: Activity,
    last: Option<Reading>,
    /// The playback position and time at the last tick.
    clock: Option<(f64, f64)>,
    /// The coarse picture of the last measurement, for noticing cuts.
    picture: Option<Vec<f32>>,
}

/// What the measurements ask for, in order once locked: the aim, a side,
/// the aim, the other side.
const CYCLE: [Side; 4] = [Side::Centre, Side::Left, Side::Centre, Side::Right];

fn nearness_of(distance: f32) -> f32 {
    1.0 / distance.clamp(*FOCUS_DISTANCES.start(), *FOCUS_DISTANCES.end())
}

impl Controller {
    /// Shows `distance` metres until it has something to go on.
    pub fn new(distance: f32) -> Controller {
        Controller {
            generation: 0,
            shown: nearness_of(distance),
            accepted: None,
            recent: Vec::new(),
            cycle: 0,
            activity: Activity::Measuring,
            last: None,
            clock: None,
            picture: None,
        }
    }

    /// The distance to draw at now, metres.
    pub fn distance(&self) -> f32 {
        1.0 / self.shown
    }

    /// Whether it has something to go on and follows it.
    pub fn locked(&self) -> bool {
        self.accepted.is_some()
    }

    pub fn activity(&self) -> Activity {
        self.activity
    }

    /// The last good measurement.
    pub fn last_reading(&self) -> Option<Reading> {
        self.last
    }

    /// Starts over from nothing to go on, showing `distance` metres: a
    /// seek, a cut, another video, another lens spacing, or turning on.
    /// Results of what was asked for before are ignored.
    pub fn start_over(&mut self, distance: f32) {
        self.generation += 1;
        self.shown = nearness_of(distance);
        self.accepted = None;
        self.recent.clear();
        self.cycle = 0;
        self.activity = Activity::Measuring;
        self.picture = None;
    }

    /// Advances the glide to `now` (seconds, any clock) and watches the
    /// playback `position` (seconds) for seeks: going back, or forward
    /// much faster than time. A pause or a stall is neither.
    pub fn tick(&mut self, now: f64, position: f64, follow: &Follow) {
        if let Some((t0, p0)) = self.clock {
            let dt = (now - t0).max(0.0);
            let moved = position - p0;
            if moved < -SEEK_BACK || moved > SEEK_SPEED * dt + SEEK_SLACK {
                self.start_over(self.distance());
            }
            if let Some(target) = self.accepted {
                let step = (dt as f32).min(MAX_STEP) * follow.speed.max(0.05);
                self.shown += (target - self.shown) * (1.0 - (-step / GLIDE).exp());
            }
        }
        self.clock = Some((now, position));
    }

    /// What to measure next, tagged; ask when the last has come back (or
    /// was given up).
    pub fn ask(&mut self) -> Tag {
        // Nothing to go on: the aim, for the first two to agree.
        let side = if self.locked() {
            let s = CYCLE[self.cycle % CYCLE.len()];
            self.cycle += 1;
            s
        } else {
            Side::Centre
        };
        Tag {
            generation: self.generation,
            side,
        }
    }

    /// Takes the result of measurement `tag`, finished at `now`, with the
    /// coarse picture it sampled (for noticing a cut); ignored if asked for
    /// before a start-over.
    pub fn take(&mut self, tag: Tag, outcome: Outcome, picture: &[f32], now: f64, follow: &Follow) {
        if tag.generation != self.generation {
            return;
        }
        if let Some(before) = &self.picture
            && cut(before, picture)
        {
            self.start_over(self.distance());
        }
        self.picture = Some(picture.to_vec());
        // Outside what it follows it holds, but each end of the distances
        // offered is no limit: past it, it follows as far as that end.
        let (near_limit, far_limit) = (nearness_of(follow.nearest), nearness_of(follow.farthest));
        let open_near = follow.nearest <= *FOCUS_DISTANCES.start();
        let open_far = follow.farthest >= *FOCUS_DISTANCES.end();
        let nearness = match outcome {
            Outcome::Depth {
                nearness,
                confidence,
            } if confidence >= MIN_CONFIDENCE => {
                if tag.side == Side::Centre {
                    self.last = Some(Reading {
                        distance: 1.0 / nearness.max(nearness_of(*FOCUS_DISTANCES.end())),
                        confidence,
                    });
                }
                Some(if nearness > near_limit && open_near {
                    near_limit
                } else if nearness < far_limit && open_far {
                    far_limit
                } else {
                    nearness
                })
            }
            _ => None,
        };
        self.recent.push(Got {
            side: tag.side,
            time: now,
            nearness,
        });
        // A few cycles' worth.
        if self.recent.len() > 12 {
            self.recent.remove(0);
        }
        if tag.side != Side::Centre {
            return;
        }
        let Some(n) = nearness else {
            if let Some(hold) = hold_for(outcome) {
                self.activity = Activity::Holding(hold);
            }
            return;
        };
        if n > near_limit + 1e-6 {
            self.activity = Activity::Holding(Hold::TooNear);
            return;
        }
        if n < far_limit - 1e-6 {
            self.activity = Activity::Holding(Hold::TooFar);
            return;
        }
        let centres: Vec<f32> = self
            .recent
            .iter()
            .rev()
            .filter(|g| g.side == Side::Centre)
            .take(3)
            .filter_map(|g| g.nearness)
            .collect();
        match self.accepted {
            // The first two to agree are taken at once.
            None => {
                if centres.get(1).is_some_and(|p| (p - n).abs() <= AGREE) {
                    self.accepted = Some((n + centres[1]) / 2.0);
                    self.activity = Activity::Following;
                } else {
                    self.activity = Activity::Measuring;
                }
            }
            // Agreeing with the focus: follow it closely.
            Some(a) if (n - a).abs() <= AGREE => {
                let refine = if follow.eyes { REFINE_EYES } else { REFINE };
                self.accepted = Some(a + (n - a) * refine);
                self.activity = Activity::Following;
            }
            // Something else at the aim: changed only if the aim agrees
            // twice, a side bears it out and no fresh side says otherwise.
            Some(_) => {
                let twice = centres.iter().filter(|p| (*p - n).abs() <= AGREE).count() >= 2;
                let sides = [Side::Left, Side::Right].map(|s| {
                    self.recent
                        .iter()
                        .rev()
                        .find(|g| g.side == s && now - g.time <= FRESH)
                        .and_then(|g| g.nearness)
                        .map(|p| (p - n).abs() <= AGREE)
                });
                let borne_out = sides.contains(&Some(true)) && !sides.contains(&Some(false));
                if twice && borne_out {
                    let agreeing: Vec<f32> = centres
                        .iter()
                        .copied()
                        .filter(|p| (p - n).abs() <= AGREE)
                        .collect();
                    self.accepted = Some(agreeing.iter().sum::<f32>() / agreeing.len() as f32);
                }
                self.activity = Activity::Following;
            }
        }
    }
}

/// Why a measurement that gave no depth means holding.
fn hold_for(outcome: Outcome) -> Option<Hold> {
    Some(match outcome {
        Outcome::Flat => Hold::Flat,
        Outcome::Ambiguous => Hold::Repeating,
        Outcome::Mismatch => Hold::Mismatch,
        Outcome::Edge => Hold::Edge,
        Outcome::SeeThrough => Hold::SeeThrough,
        Outcome::Depth { .. } => return None,
    })
}

/// Whether two coarse pictures of the same view are a cut apart.
fn cut(a: &[f32], b: &[f32]) -> bool {
    if a.len() != b.len() || a.is_empty() {
        return false;
    }
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let (mut va, mut vb, mut cov) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        va += (x - ma) * (x - ma);
        vb += (y - mb) * (y - mb);
        cov += (x - ma) * (y - mb);
    }
    let (sa, sb) = ((va / n).sqrt(), (vb / n).sqrt());
    if sa < FLAT_PICTURE || sb < FLAT_PICTURE {
        return (ma - mb).abs() > CUT_MEAN;
    }
    cov / (va * vb).sqrt() < CUT_CORRELATION
}

#[cfg(test)]
mod tests {
    use super::*;

    const HZ: f64 = 90.0;

    fn depth(distance: f32) -> Outcome {
        Outcome::Depth {
            nearness: 1.0 / distance,
            confidence: 0.9,
        }
    }

    /// A coarse picture; another `seed` is another shot.
    fn shot(seed: u32) -> Vec<f32> {
        (0..64u32)
            .map(|i| {
                let mut h = (i + seed * 1000).wrapping_mul(2_654_435_761);
                h ^= h >> 15;
                h = h.wrapping_mul(0x2c1b_3c6d);
                (h >> 8) as f32 / (1u32 << 24) as f32
            })
            .collect()
    }

    /// The player at 90 Hz: a measurement asked for every third frame
    /// comes back two frames later from `scene`.
    struct Sim {
        c: Controller,
        follow: Follow,
        now: f64,
        position: f64,
        playing: bool,
        frame: u64,
        pending: Option<(Tag, u64)>,
        picture: Vec<f32>,
    }

    impl Sim {
        fn new(distance: f32) -> Sim {
            Sim {
                c: Controller::new(distance),
                follow: Follow::default(),
                now: 0.0,
                position: 0.0,
                playing: true,
                frame: 0,
                pending: None,
                picture: shot(1),
            }
        }

        fn run(&mut self, seconds: f64, scene: impl Fn(Side) -> Outcome) {
            for _ in 0..(seconds * HZ).round() as u64 {
                self.now += 1.0 / HZ;
                if self.playing {
                    self.position += 1.0 / HZ;
                }
                self.frame += 1;
                self.c.tick(self.now, self.position, &self.follow);
                if let Some((tag, at)) = self.pending
                    && self.frame >= at
                {
                    self.c
                        .take(tag, scene(tag.side), &self.picture, self.now, &self.follow);
                    self.pending = None;
                }
                if self.pending.is_none() && self.frame.is_multiple_of(3) {
                    self.pending = Some((self.c.ask(), self.frame + 2));
                }
            }
        }

        fn nearness(&self) -> f32 {
            1.0 / self.c.distance()
        }
    }

    fn everywhere(distance: f32) -> impl Fn(Side) -> Outcome {
        move |_| depth(distance)
    }

    #[test]
    fn agreeing_measurements_lock_and_glide_as_the_eyes_converge() {
        let mut s = Sim::new(2.0);
        // The first two agree: locked at once (a measurement every third
        // frame, back two frames later: the second by the eighth frame).
        s.run(8.0 / HZ, everywhere(1.0));
        assert!(s.c.locked() && s.c.activity() == Activity::Following);
        let start = s.nearness();
        // One time constant on, about 63% of the way (frames are discrete).
        s.run(GLIDE as f64, everywhere(1.0));
        let done = (s.nearness() - start) / (1.0 - start);
        assert!((done - 0.632).abs() < 0.03, "{done}");
        // Within half a resolution step of 1 m by 1.5 s from the start.
        s.run(1.5 - 8.0 / HZ - GLIDE as f64, everywhere(1.0));
        assert!((s.nearness() - 1.0).abs() < 0.0065, "{}", s.c.distance());
        assert_eq!(s.c.last_reading().map(|r| r.distance), Some(1.0));

        // Twice the speed: as far in half the time.
        let mut fast = Sim::new(2.0);
        fast.follow.speed = 2.0;
        fast.run(8.0 / HZ, everywhere(1.0));
        let start = fast.nearness();
        fast.run(GLIDE as f64 / 2.0, everywhere(1.0));
        let done = (fast.nearness() - start) / (1.0 - start);
        assert!((done - 0.632).abs() < 0.05, "{done}");

        // The same glide at 72 and 120 frames a second.
        let glide = |hz: f64| {
            let mut c = Controller::new(2.0);
            let f = Follow::default();
            c.tick(0.0, 0.0, &f);
            for t in [c.ask(), c.ask()] {
                c.take(t, depth(1.0), &shot(1), 0.0, &f);
            }
            for i in 1..=(0.25 * hz).round() as u32 {
                let now = i as f64 / hz;
                c.tick(now, now, &f);
            }
            c.distance()
        };
        assert!((glide(72.0) - glide(120.0)).abs() < 1e-4);

        // A stall of a second moves it no more than a tenth would.
        let mut c = Controller::new(2.0);
        let f = Follow::default();
        c.tick(0.0, 0.0, &f);
        for t in [c.ask(), c.ask()] {
            c.take(t, depth(1.0), &shot(1), 0.0, &f);
        }
        c.tick(1.0, 0.0, &f);
        let want = 0.5 + 0.5 * (1.0 - (-MAX_STEP / GLIDE).exp());
        assert!((1.0 / c.distance() - want).abs() < 1e-5);
    }

    #[test]
    fn one_wrong_measurement_never_moves_it() {
        let mut s = Sim::new(1.0);
        s.run(1.0, everywhere(1.0));
        let before = s.nearness();
        for wrong in [Side::Centre, Side::Left, Side::Right] {
            // The first answer from that side reads 3 m, then all 1 m.
            let left = std::cell::Cell::new(true);
            let scene = |side: Side| {
                if side == wrong && left.replace(false) {
                    depth(3.0)
                } else {
                    depth(1.0)
                }
            };
            let mut s2 = s_clone(&s);
            s2.run(1.0, scene);
            assert!(!left.get(), "{wrong:?} was measured");
            assert!(
                (s2.nearness() - before).abs() < 0.002,
                "{wrong:?}: {}",
                s2.c.distance()
            );
        }
    }

    fn s_clone(s: &Sim) -> Sim {
        Sim {
            c: s.c.clone(),
            follow: s.follow,
            now: s.now,
            position: s.position,
            playing: s.playing,
            frame: s.frame,
            pending: s.pending,
            picture: s.picture.clone(),
        }
    }

    #[test]
    fn a_subject_entering_from_one_side_switches_it_only_when_borne_out() {
        let mut s = Sim::new(3.0);
        s.run(1.5, everywhere(3.0));
        let far = s.nearness();
        // The subject's edge at the aim: the left and the aim see it,
        // the right still sees the wall behind.
        let edge = |side: Side| match side {
            Side::Right => depth(3.0),
            _ => depth(1.0),
        };
        s.run(1.0, edge);
        assert!((s.nearness() - far).abs() < 0.002, "{}", s.c.distance());
        // The aim flickering between the two at the edge doesn't either.
        let mut flip = false;
        let flicker = |side: Side| match side {
            Side::Left => depth(1.0),
            Side::Right => depth(3.0),
            Side::Centre => {
                flip = !flip;
                if flip { depth(1.0) } else { depth(3.0) }
            }
        };
        let flicker = std::cell::RefCell::new(flicker);
        s.run(1.0, |side| (flicker.borrow_mut())(side));
        assert!((s.nearness() - far).abs() < 0.002, "{}", s.c.distance());
        // Across the aim and both sides: it follows.
        s.run(1.5, everywhere(1.0));
        assert!((s.nearness() - 1.0).abs() < 0.01, "{}", s.c.distance());
    }

    #[test]
    fn outside_what_it_follows_it_holds_and_says_which_side() {
        let mut s = Sim::new(1.5);
        s.follow.nearest = 0.8;
        s.run(1.0, everywhere(0.5));
        assert_eq!(s.c.activity(), Activity::Holding(Hold::TooNear));
        assert!((s.c.distance() - 1.5).abs() < 1e-4, "held");
        s.follow.nearest = 0.4;
        s.run(1.5, everywhere(0.5));
        assert!((s.c.distance() - 0.5).abs() < 0.01, "{}", s.c.distance());

        let mut s = Sim::new(1.5);
        s.follow.farthest = 5.0;
        s.run(1.0, everywhere(10.0));
        assert_eq!(s.c.activity(), Activity::Holding(Hold::TooFar));
        s.follow.farthest = 20.0;
        s.run(2.0, everywhere(10.0));
        assert!((s.c.distance() - 10.0).abs() < 0.5, "{}", s.c.distance());
        // At the farthest offered, farther still is followed to there.
        s.run(2.0, |_| Outcome::Depth {
            nearness: 0.0,
            confidence: 0.9,
        });
        assert!((s.c.distance() - 20.0).abs() < 1.0, "{}", s.c.distance());
        // A measurement that gives no depth holds, saying why.
        s.run(0.2, |_| Outcome::Ambiguous);
        assert_eq!(s.c.activity(), Activity::Holding(Hold::Repeating));
    }

    #[test]
    fn seeks_cuts_and_restarts_start_over_and_pauses_dont() {
        let locked = || {
            let mut s = Sim::new(2.0);
            s.run(1.5, everywhere(1.0));
            assert!(s.c.locked());
            s
        };
        // Playing on, pausing and stalling keep it.
        let mut s = locked();
        s.playing = false;
        s.run(2.0, everywhere(1.0));
        assert!(s.c.locked(), "paused or stalled");
        // Going back, and jumping ahead, start over where it shows.
        for jump in [-10.0, -0.5, 60.0, 3.0] {
            let mut s = locked();
            let shown = s.c.distance();
            s.position += jump;
            s.run(1.0 / HZ, everywhere(1.0));
            assert!(!s.c.locked(), "jumped {jump} s");
            assert!((s.c.distance() - shown).abs() < 1e-3, "nothing jumps");
        }
        // A cut: the coarse picture changes.
        let mut s = locked();
        s.picture = shot(2);
        s.run(3.0 / HZ, everywhere(1.0));
        assert!(!s.c.locked(), "cut");
        // Then the first two to agree are taken at once.
        s.run(0.1, everywhere(3.0));
        assert!(s.c.locked());
        // An answer asked for before starting over is ignored.
        let mut c = Controller::new(2.0);
        let f = Follow::default();
        let old = c.ask();
        c.start_over(2.0);
        let new = c.ask();
        c.take(old, depth(1.0), &shot(1), 0.0, &f);
        c.take(new, depth(1.0), &shot(1), 0.0, &f);
        assert!(!c.locked(), "only one answer counted");
        assert_eq!(c.last_reading().map(|r| r.distance), Some(1.0));
    }

    #[test]
    fn a_fade_is_not_a_cut() {
        let a = shot(5);
        let darker: Vec<f32> = a.iter().map(|v| v * 0.7).collect();
        assert!(!cut(&a, &darker));
        assert!(cut(&a, &shot(6)));
        // Nearly black pictures go by their brightness.
        let black = vec![0.02; 64];
        assert!(!cut(&black, &vec![0.05; 64]));
        assert!(cut(&black, &vec![0.4; 64]));
    }
}
