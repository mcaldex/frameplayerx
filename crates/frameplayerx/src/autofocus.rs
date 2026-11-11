//! Automatic stereo focus for the open video. Each frame it ticks the
//! focus controller ([`fp_core::autofocus`]), takes in a finished match,
//! hands grids the renderer has finished to a worker thread to match
//! ([`fp_core::depth::Matcher`]), and, when both are free, asks the
//! renderer for the next measurement along the aim. The render thread only
//! ever tries the worker's channel; nothing waits.

use fp_core::autofocus::{Activity, Controller, Follow, Hold, MIN_CONFIDENCE, SIDE, Side, Tag};
use fp_core::depth::{Angles, Grids, Matcher, Measurement, Outcome, Plan, Request, Window};
use fp_core::format::VideoFormat;
use fp_render::params::look_in_video;
use fp_render::{Renderer, VideoParams};
use glam::Vec3;
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TryRecvError, sync_channel};
use std::time::{Duration, Instant};

/// How much nearer than the nearest distance followed it searches, as a
/// factor of nearness: a quarter nearer (0.4 m for 0.5 m), so a subject
/// just inside the limit is still measured and held at, rather than
/// mistaken for something farther.
const SEARCH_NEARER: f32 = 1.25;

/// How often the status may change, seconds: at most four times a second,
/// so it reads rather than flickers, and costs the adjust panel few paints.
const STATUS_EVERY: f64 = 0.25;

/// What it is doing, for the status row and the preview's `status`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Status {
    /// Nothing to go on yet.
    Measuring,
    /// Following, at `distance` metres, the last measurement `confidence`
    /// (0..1) sure.
    Following { distance: f32, confidence: f32 },
    /// Holding at `distance` metres, for `why`.
    Holding { distance: f32, why: Hold },
    /// It can't measure here (the pass or the worker failed): the video's
    /// own distance, at `distance` metres.
    Unavailable { distance: f32 },
}

impl Status {
    /// The same to the eye: distances to the centimetre, confidence to a
    /// tenth.
    fn shows_as(&self, other: &Status) -> bool {
        let cm = |d: f32| (d * 100.0).round() as i32;
        match (*self, *other) {
            (
                Status::Following {
                    distance: a,
                    confidence: ca,
                },
                Status::Following {
                    distance: b,
                    confidence: cb,
                },
            ) => cm(a) == cm(b) && (ca * 10.0).round() == (cb * 10.0).round(),
            (
                Status::Holding { distance: a, why },
                Status::Holding {
                    distance: b,
                    why: w,
                },
            ) => cm(a) == cm(b) && why == w,
            (Status::Unavailable { distance: a }, Status::Unavailable { distance: b }) => {
                cm(a) == cm(b)
            }
            (a, b) => a == b,
        }
    }
}

/// What makes a rig's learned vertical offset another rig's: the format,
/// the eye swap and the lens spacing.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Rig {
    format: VideoFormat,
    swapped: bool,
    spacing: f32,
}

struct Job {
    tag: Tag,
    rig: Rig,
    plan: Plan,
    grids: Grids,
    matcher: Matcher,
}

struct Done {
    tag: Tag,
    rig: Rig,
    measurement: Measurement,
    coarse: Vec<f32>,
    matcher: Matcher,
    took: Duration,
}

/// The thread that matches, one measurement at a time; it ends when its
/// owner is dropped.
struct Worker {
    jobs: SyncSender<Job>,
    done: Receiver<Done>,
}

impl Worker {
    fn start() -> std::io::Result<Worker> {
        let (jobs, inbox) = sync_channel::<Job>(1);
        let (outbox, done) = sync_channel(1);
        std::thread::Builder::new()
            .name("autofocus".into())
            .spawn(move || {
                for mut job in inbox {
                    let start = Instant::now();
                    let measurement = job.matcher.measure(&job.plan, &job.grids);
                    let done = Done {
                        tag: job.tag,
                        rig: job.rig,
                        measurement,
                        coarse: std::mem::take(&mut job.grids.coarse),
                        matcher: job.matcher,
                        took: start.elapsed(),
                    };
                    if outbox.send(done).is_err() {
                        break;
                    }
                }
            })?;
        Ok(Worker { jobs, done })
    }
}

/// What the 10-second log line says.
#[derive(Default)]
struct Tally {
    measured: u32,
    with_depth: u32,
    match_ms: Vec<f32>,
}

/// Automatic focus's state for one open video.
pub struct AutoFocus {
    controller: Controller,
    /// Here between matches, on the worker during one.
    matcher: Option<Matcher>,
    worker: Option<Worker>,
    /// The worker couldn't start, or stopped: no more measuring.
    broken: bool,
    /// Asked of the renderer: its tag there, the controller's, the plan.
    asked: Option<(u64, Tag, Plan)>,
    next_id: u64,
    rig: Option<Rig>,
    /// The coarse picture of the last measurement back.
    picture: Vec<f32>,
    status: Status,
    status_at: f64,
    status_changed: bool,
    tally: Tally,
    /// Waits for a match handed over, as if it took no time: for tests,
    /// which run frames faster than real time. A match takes a few
    /// milliseconds, under a frame at 90 Hz.
    lockstep: bool,
}

/// The aim for side `side` of `look`, in the video's axes.
fn aim(video: &VideoParams, look: Vec3, side: Side) -> Angles {
    let a = Angles::of(look_in_video(video, look).to_array());
    let alpha = match side {
        Side::Centre => a.alpha,
        Side::Left => a.alpha + SIDE,
        Side::Right => a.alpha - SIDE,
    };
    Angles { alpha, ..a }
}

impl AutoFocus {
    /// Starts at `distance` metres, the video's manual distance, so nothing
    /// jumps.
    pub fn new(distance: f32) -> AutoFocus {
        AutoFocus {
            controller: Controller::new(distance),
            matcher: Some(Matcher::new()),
            worker: None,
            broken: false,
            asked: None,
            next_id: 0,
            rig: None,
            picture: Vec::new(),
            status: Status::Measuring,
            status_at: f64::MIN,
            status_changed: true,
            tally: Tally::default(),
            lockstep: false,
        }
    }

    /// The distance to draw at now, metres.
    pub fn distance(&self) -> f32 {
        self.controller.distance()
    }

    /// What it shows it is doing (changing at most every
    /// [`STATUS_EVERY`]).
    pub fn status(&self) -> Status {
        self.status
    }

    /// Whether the status changed since this was last asked.
    pub fn take_status_change(&mut self) -> bool {
        std::mem::take(&mut self.status_changed)
    }

    /// Gives up the measurement asked of the renderer, if any: it is
    /// stopping (another mode, another kind of video).
    pub fn stop(&mut self, renderer: &mut Renderer) {
        if self.asked.take().is_some() {
            renderer.cancel_depth();
        }
    }

    /// One frame at `now` (seconds) with the playback at `position`
    /// (seconds), drawing `video`, the viewer looking along `look` (world)
    /// and `shown` whether a frame of the video is shown. Returns the
    /// distance to draw this frame at, metres.
    #[allow(clippy::too_many_arguments)]
    pub fn frame(
        &mut self,
        renderer: &mut Renderer,
        now: f64,
        position: f64,
        video: &VideoParams,
        look: Vec3,
        shown: bool,
        follow: &Follow,
    ) -> f32 {
        let s = &video.settings;
        let rig = Rig {
            format: video.format,
            swapped: s.swap_eyes,
            spacing: s.lens_spacing,
        };
        if self.rig.is_some_and(|r| r != rig) {
            // Another rig: start over where it shows, and forget what was
            // learned of the old one.
            self.controller.start_over(self.controller.distance());
            if let Some(m) = self.matcher.as_mut() {
                m.forget_rig();
            }
            self.stop(renderer);
        }
        self.rig = Some(rig);
        self.controller.tick(now, position, follow);
        self.take_match(now, follow);
        if let Some((id, grids)) = renderer.take_depth() {
            self.hand_over(id, grids);
        }
        if shown && !self.broken {
            self.ask(renderer, video, look, now, follow);
        }
        self.update_status(now, renderer.measures_depth());
        self.controller.distance()
    }

    /// Takes in a finished match, if there is one.
    fn take_match(&mut self, now: f64, follow: &Follow) {
        let Some(w) = &self.worker else {
            return;
        };
        let got = if self.lockstep && self.matcher.is_none() {
            w.done
                .recv_timeout(Duration::from_secs(10))
                .map_err(|e| match e {
                    RecvTimeoutError::Timeout => TryRecvError::Empty,
                    RecvTimeoutError::Disconnected => TryRecvError::Disconnected,
                })
        } else {
            w.done.try_recv()
        };
        match got {
            Ok(d) => {
                let mut matcher = d.matcher;
                if Some(d.rig) != self.rig {
                    matcher.forget_rig();
                }
                self.matcher = Some(matcher);
                self.tally.measured += 1;
                self.tally.match_ms.push(d.took.as_secs_f32() * 1000.0);
                if matches!(d.measurement.outcome, Outcome::Depth { confidence, .. } if confidence >= MIN_CONFIDENCE)
                {
                    self.tally.with_depth += 1;
                }
                self.controller
                    .take(d.tag, d.measurement.outcome, &d.coarse, now, follow);
                self.picture = d.coarse;
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                log::warn!("automatic focus stopped measuring: its worker thread ended");
                self.broken = true;
                self.worker = None;
            }
        }
    }

    /// Hands the grids of measurement `id` to the worker.
    fn hand_over(&mut self, id: u64, grids: Grids) {
        let Some((asked, tag, plan)) = self.asked.take() else {
            return;
        };
        let (Some(rig), Some(matcher)) = (self.rig, self.matcher.take()) else {
            return;
        };
        if asked != id {
            self.matcher = Some(matcher);
            return;
        }
        if self.worker.is_none() {
            match Worker::start() {
                Ok(w) => self.worker = Some(w),
                Err(e) => {
                    log::warn!("automatic focus can't measure: no worker thread: {e}");
                    self.broken = true;
                    self.matcher = Some(matcher);
                    return;
                }
            }
        }
        let job = Job {
            tag,
            rig,
            plan,
            grids,
            matcher,
        };
        if let Some(w) = &self.worker
            && let Err(e) = w.jobs.try_send(job)
        {
            // It is free whenever a matcher is here, so only a worker that
            // has gone refuses.
            let job = match e {
                std::sync::mpsc::TrySendError::Full(j)
                | std::sync::mpsc::TrySendError::Disconnected(j) => j,
            };
            log::warn!("automatic focus stopped measuring: its worker thread ended");
            self.matcher = Some(job.matcher);
            self.broken = true;
            self.worker = None;
        }
    }

    /// Asks the renderer for the next measurement, when it and the worker
    /// are free.
    fn ask(
        &mut self,
        renderer: &mut Renderer,
        video: &VideoParams,
        look: Vec3,
        now: f64,
        follow: &Follow,
    ) {
        let Some(matcher) = &self.matcher else {
            return;
        };
        if self.asked.is_some() || renderer.depth_pending() || !renderer.measures_depth() {
            return;
        }
        let tag = self.controller.ask();
        let request = Request {
            aim: aim(video, look, tag.side),
            baseline: video.settings.lens_spacing / 1000.0,
            nearest: follow.nearest / SEARCH_NEARER,
            window: Window::HEAD,
        };
        match matcher.plan(&request) {
            Ok(plan) => {
                self.next_id += 1;
                if renderer.ask_depth(self.next_id, &plan.grid) {
                    self.asked = Some((self.next_id, tag, plan));
                }
            }
            // Nothing to sample there: the answer is known now.
            Err(outcome) => {
                let picture = std::mem::take(&mut self.picture);
                self.controller.take(tag, outcome, &picture, now, follow);
                self.picture = picture;
            }
        }
    }

    fn update_status(&mut self, now: f64, measures: bool) {
        let distance = self.controller.distance();
        let current = if self.broken || !measures {
            Status::Unavailable { distance }
        } else {
            match self.controller.activity() {
                Activity::Measuring => Status::Measuring,
                Activity::Following => Status::Following {
                    distance,
                    confidence: self.controller.last_reading().map_or(0.0, |r| r.confidence),
                },
                Activity::Holding(why) => Status::Holding { distance, why },
            }
        };
        if !current.shows_as(&self.status) && now - self.status_at >= STATUS_EVERY {
            self.status = current;
            self.status_at = now;
            self.status_changed = true;
        }
    }

    /// The figures since the last call, for the 10-second log line, if it
    /// measured: made, with a depth, typical and longest match, the
    /// distance now. Nothing in it says what the video is.
    pub fn take_log_line(&mut self) -> Option<String> {
        let t = std::mem::take(&mut self.tally);
        if t.measured == 0 {
            return None;
        }
        let mut ms = t.match_ms;
        ms.sort_by(f32::total_cmp);
        Some(format!(
            "automatic focus: {} measured, {} with a depth; match ms median {:.2}, max {:.2}; at {:.2} m",
            t.measured,
            t.with_depth,
            ms[ms.len() / 2],
            ms[ms.len() - 1],
            self.controller.distance()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fp_core::depth::picture::PictureGeometry;
    use fp_core::depth::scenes::{Picture, Rig, Scene, Surface, Texture};
    use fp_core::view::{Focus, ViewSettings};
    use fp_render::capture::OffscreenEyes;
    use fp_render::math::{Fov, projection, view};
    use fp_render::{EyeView, Gpu};
    use glam::Quat;
    use std::sync::Arc;

    /// A 180° side-by-side frame of a textured wall `distance` metres
    /// ahead, as a camera 65 mm wide films it, a thousand pixels a picture.
    fn wall(distance: f32) -> Arc<fp_media::VideoFrame> {
        let (w, h) = (2048, 1024);
        let g = PictureGeometry::new(
            &VideoFormat::FALLBACK,
            &ViewSettings::default(),
            w,
            h,
            1.0,
            0,
        );
        let scene = Scene::new(vec![Surface::wall(
            distance,
            Texture::Noise {
                feature: 0.006 * distance,
                contrast: 1.0,
                seed: 17,
            },
        )]);
        let (luma, _) =
            Picture::new([&scene, &scene], Rig::new(0.065), g, w as usize, h as usize).render();
        let rgba: Vec<u8> = luma
            .iter()
            .flat_map(|l| {
                let v = (l * 255.0).round() as u8;
                [v, v, v, 255]
            })
            .collect();
        Arc::new(fp_media::VideoFrame::from_rgba(w, h, &rgba).unwrap())
    }

    struct Player {
        renderer: Renderer,
        eyes: OffscreenEyes,
        focus: AutoFocus,
        video: VideoParams,
        now: f64,
        position: f64,
    }

    impl Player {
        fn eyes() -> [EyeView; 2] {
            [-0.032f32, 0.032].map(|x| EyeView {
                view: view(Vec3::new(x, 1.6, 0.0), Quat::IDENTITY),
                proj: projection(Fov::symmetric(90.0), 0.05, 100.0),
            })
        }

        /// `seconds` of frames at 90 Hz showing `frame`, playing or paused;
        /// the last frame's left eye.
        fn run(
            &mut self,
            seconds: f64,
            frame: &Arc<fp_media::VideoFrame>,
            playing: bool,
        ) -> Vec<u8> {
            let follow = Follow::default();
            for _ in 0..(seconds * 90.0).round() as u32 {
                self.now += 1.0 / 90.0;
                if playing {
                    self.position += 1.0 / 90.0;
                }
                let r = &mut self.renderer;
                r.begin_frame().unwrap();
                r.set_video(Some(frame)).unwrap();
                let d = self.focus.frame(
                    r,
                    self.now,
                    self.position,
                    &self.video,
                    Vec3::NEG_Z,
                    true,
                    &follow,
                );
                let shown = VideoParams {
                    auto_focus: Some(d),
                    ..self.video
                };
                r.draw(&self.eyes.targets(), &Self::eyes(), &shown, &[])
                    .unwrap();
                r.end_frame().unwrap();
            }
            self.eyes.read(&self.renderer, 0).unwrap()
        }
    }

    #[test]
    fn it_settles_on_a_wall_follows_it_back_and_hands_over_to_manual() {
        let gpu = match Gpu::headless() {
            Ok(g) => Arc::new(g),
            Err(e) => {
                eprintln!("skipping: no Vulkan device ({e})");
                return;
            }
        };
        let renderer = Renderer::new(gpu, ash::vk::Format::R8G8B8A8_SRGB).unwrap();
        let eyes = OffscreenEyes::new(&renderer, 48, 48).unwrap();
        let settings = ViewSettings {
            focus: Focus::Auto,
            focus_distance: 2.0,
            lens_spacing: 65.0,
            ..ViewSettings::default()
        };
        let video = VideoParams {
            format: VideoFormat::FALLBACK,
            settings,
            ..Default::default()
        };
        let mut focus = AutoFocus::new(settings.focus_distance);
        focus.lockstep = true;
        let mut p = Player {
            renderer,
            eyes,
            focus,
            video,
            now: 0.0,
            position: 0.0,
        };
        // Half a resolution step at 65 mm, 1/metres.
        let half = fp_core::depth::RESOLUTION / 0.065 / 2.0;
        let near = wall(1.0);
        p.run(1.5, &near, true);
        let n = 1.0 / p.focus.distance();
        assert!(
            (n - 1.0).abs() < half,
            "at {} m after 1.5 s",
            p.focus.distance()
        );
        assert!(
            matches!(p.focus.status(), Status::Following { confidence, .. } if confidence > 0.5),
            "{:?}",
            p.focus.status()
        );
        // Paused, it holds.
        let held = p.focus.distance();
        p.run(1.0, &near, false);
        assert!((p.focus.distance() - held).abs() < 0.002);
        // The wall moved back to 2 m.
        let far = wall(2.0);
        p.run(2.0, &far, true);
        let n = 1.0 / p.focus.distance();
        assert!((n - 0.5).abs() < 2.0 * half, "at {} m", p.focus.distance());
        // Manual from there draws the same picture.
        let auto = p.run(1.0 / 90.0, &far, true);
        let at = p.focus.distance();
        let manual = VideoParams {
            settings: ViewSettings {
                focus: Focus::Manual,
                focus_distance: at,
                ..settings
            },
            ..video
        };
        let r = &mut p.renderer;
        r.begin_frame().unwrap();
        r.set_video(Some(&far)).unwrap();
        r.draw(&p.eyes.targets(), &Player::eyes(), &manual, &[])
            .unwrap();
        r.end_frame().unwrap();
        assert!(p.eyes.read(r, 0).unwrap() == auto);
        // None of it touched the video's adjustments.
        assert_eq!(p.video.settings, settings);
        let line = p.focus.take_log_line().expect("measured");
        eprintln!("{line}");
        p.eyes.destroy(&p.renderer);
    }
}
