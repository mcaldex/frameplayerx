//! Stereo focus on a real (software) Vulkan device: each eye is shown a
//! point at the focus distance where a real one would be, for any head
//! pose and eye spacing, and nothing changes while the eyes are where the
//! cameras were.

use fp_core::format::{Projection, StereoLayout, VideoFormat};
use fp_core::view::{Focus, ViewSettings};
use fp_render::capture::OffscreenEyes;
use fp_render::math::{Fov, projection, view};
use fp_render::params::video_axes;
use fp_render::{EyeView, Gpu, Renderer, VideoParams};
use glam::{DVec3, EulerRot, Quat, Vec3};
use std::sync::Arc;

/// Eye images' size, pixels.
const SIZE: u32 = 256;
/// Eye images' field of view, degrees: 0.23° a pixel in the middle.
const FOV: f32 = 60.0;

fn setup() -> Option<(Renderer, OffscreenEyes)> {
    let gpu = match Gpu::headless() {
        Ok(g) => Arc::new(g),
        Err(e) => {
            eprintln!("skipping: no Vulkan device ({e})");
            return None;
        }
    };
    let r = Renderer::new(gpu, ash::vk::Format::R8G8B8A8_SRGB).expect("renderer");
    let eyes = OffscreenEyes::new(&r, SIZE, SIZE).expect("eyes");
    Some((r, eyes))
}

fn draw(
    r: &mut Renderer,
    e: &mut OffscreenEyes,
    f: &Arc<fp_media::VideoFrame>,
    p: &VideoParams,
    eyes: &[EyeView; 2],
) -> [Vec<u8>; 2] {
    r.begin_frame().unwrap();
    r.set_video(Some(f)).unwrap();
    r.draw(&e.targets(), eyes, p, &[]).unwrap();
    r.end_frame().unwrap();
    [e.read(r, 0).unwrap(), e.read(r, 1).unwrap()]
}

/// Eyes `ipd` metres apart at `head`, turned by `turn`.
fn eyes(head: Vec3, turn: Quat, ipd: f32) -> [EyeView; 2] {
    [-0.5, 0.5].map(|x| EyeView {
        view: view(head + turn * Vec3::X * (x * ipd), turn),
        proj: projection(Fov::symmetric(FOV), 0.05, 100.0),
    })
}

/// The direction a point of an eye's picture shows, in the video's axes,
/// from its coordinates in that picture (0..1, origin top-left).
fn direction(projection: Projection, u: f64, v: f64) -> Option<DVec3> {
    match projection {
        Projection::Equirect { h_fov, v_fov } => {
            let lon = (u - 0.5) * (h_fov as f64).to_radians();
            let lat = (0.5 - v) * (v_fov as f64).to_radians();
            Some(DVec3::new(
                lon.sin() * lat.cos(),
                lat.sin(),
                -lon.cos() * lat.cos(),
            ))
        }
        Projection::Fisheye { fov } => {
            let (x, y) = (u - 0.5, 0.5 - v);
            let r = x.hypot(y);
            if r > 0.5 {
                return None;
            }
            let theta = r * (fov as f64).to_radians();
            let (dx, dy) = if r > 0.0 { (x / r, y / r) } else { (0.0, 0.0) };
            Some(DVec3::new(dx * theta.sin(), dy * theta.sin(), -theta.cos()))
        }
        _ => unreachable!(),
    }
}

/// A side-by-side frame, `eye` pixels square per eye, black but for a
/// soft white dot in each camera's picture where it saw `seen[camera]`,
/// `sigma` degrees across. `right_first` packs the right camera's picture
/// on the left.
fn dots(
    projection: Projection,
    eye: u32,
    seen: [DVec3; 2],
    sigma: f64,
    right_first: bool,
) -> Arc<fp_media::VideoFrame> {
    let (w, h) = (eye * 2, eye);
    let sigma = sigma.to_radians();
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let half = (x >= eye) as usize;
            let camera = half ^ right_first as usize;
            let (u, v) = (
                ((x % eye) as f64 + 0.5) / eye as f64,
                (y as f64 + 0.5) / h as f64,
            );
            let level = direction(projection, u, v).map_or(0.0, |d| {
                let angle = 2.0 * ((d - seen[camera]).length() / 2.0).asin();
                (-angle * angle / (2.0 * sigma * sigma)).exp()
            });
            let c = (level * 255.0).round() as u8;
            let i = ((y * w + x) * 4) as usize;
            rgba[i..i + 4].copy_from_slice(&[c, c, c, 255]);
        }
    }
    Arc::new(fp_media::VideoFrame::from_rgba(w, h, &rgba).expect("frame"))
}

/// Where the brightest spot of an eye image peaks, pixels from the top-left
/// pixel's centre: a quadratic fitted to the log of the bright pixels.
/// Seen through any smooth warp, a soft dot still peaks where its centre
/// lands, unlike its centroid.
fn peak(img: &[u8]) -> (f64, f64) {
    let at = |x: i64, y: i64| -> f64 {
        if x < 0 || y < 0 || x >= SIZE as i64 || y >= SIZE as i64 {
            return 0.0;
        }
        img[((y * SIZE as i64 + x) * 4 + 1) as usize] as f64
    };
    let (mut bx, mut by, mut best) = (0, 0, -1.0);
    for y in 0..SIZE as i64 {
        for x in 0..SIZE as i64 {
            if at(x, y) > best {
                (bx, by, best) = (x, y, at(x, y));
            }
        }
    }
    assert!(best > 100.0, "no dot in the image (brightest {best})");
    // Weighted least squares of ln I = a + bx + cy + dx² + exy + fy²; a
    // pixel's weight is I², as rounding moves ln I by about 1/I.
    let mut m = [[0.0f64; 7]; 6];
    for dy in -8..=8i64 {
        for dx in -8..=8i64 {
            let i = at(bx + dx, by + dy);
            if i < 0.3 * best {
                continue;
            }
            let (x, y) = (dx as f64, dy as f64);
            let row = [1.0, x, y, x * x, x * y, y * y];
            let wt = i * i;
            for (j, rj) in row.iter().enumerate() {
                for (k, rk) in row.iter().enumerate() {
                    m[j][k] += wt * rj * rk;
                }
                m[j][6] += wt * rj * i.ln();
            }
        }
    }
    // Gauss-Jordan elimination with partial pivoting.
    for c in 0..6 {
        let p = (c..6)
            .max_by(|&a, &b| m[a][c].abs().total_cmp(&m[b][c].abs()))
            .unwrap();
        m.swap(c, p);
        let pivot = m[c];
        for (r, row) in m.iter_mut().enumerate() {
            if r != c {
                let f = row[c] / pivot[c];
                for (x, p) in row.iter_mut().zip(pivot).skip(c) {
                    *x -= f * p;
                }
            }
        }
    }
    let s: Vec<f64> = (0..6).map(|i| m[i][6] / m[i][i]).collect();
    let (b, c, d, e, f) = (s[1], s[2], s[3], s[4], s[5]);
    // Where the gradient is zero: [2d e; e 2f] (x, y) = -(b, c).
    let det = 4.0 * d * f - e * e;
    let x = (-b * 2.0 * f + c * e) / det;
    let y = (-c * 2.0 * d + b * e) / det;
    (bx as f64 + x, by as f64 + y)
}

/// Where world point `p` shows in eye image `eye`, pixels from the
/// top-left pixel's centre.
fn projected(eye: &EyeView, p: Vec3) -> (f64, f64) {
    let c = eye.proj * eye.view * p.extend(1.0);
    let (x, y) = (c.x / c.w, c.y / c.w);
    (
        ((x + 1.0) * 0.5 * SIZE as f32 - 0.5) as f64,
        ((y + 1.0) * 0.5 * SIZE as f32 - 0.5) as f64,
    )
}

/// One arrangement: a camera `spacing` metres wide saw a point `distance`
/// from each lens, 12° above straight ahead; the viewer's eyes are `ipd`
/// apart, their head turned `yaw` and tilted `pitch` away from looking at
/// it, and rolled `roll` degrees; the video was recentred at `anchor`
/// degrees and turned `picture` degrees (yaw, pitch, roll).
struct Pose {
    spacing: f32,
    distance: f32,
    ipd: f32,
    yaw: f32,
    pitch: f32,
    roll: f32,
    anchor: f32,
    picture: [f32; 3],
}

impl Pose {
    fn video(&self, format: VideoFormat, focus: Focus, swap: bool) -> VideoParams {
        let [yaw, pitch, roll] = self.picture;
        VideoParams {
            format,
            settings: ViewSettings {
                focus,
                focus_distance: self.distance,
                lens_spacing: self.spacing * 1000.0,
                swap_eyes: swap,
                yaw,
                pitch,
                roll,
                ..ViewSettings::default()
            },
            anchor: Quat::from_rotation_y(self.anchor.to_radians()),
            ..Default::default()
        }
    }

    /// The point in the video's axes, origin between the eyes.
    fn point(&self) -> Vec3 {
        let up = (12f32.to_radians()).sin() * self.distance;
        let half = self.spacing / 2.0;
        Vec3::new(
            0.0,
            up,
            -(self.distance.powi(2) - half * half - up * up).sqrt(),
        )
    }

    /// What each camera saw it along.
    fn seen(&self) -> [DVec3; 2] {
        [-0.5, 0.5].map(|x| {
            (self.point() - Vec3::X * x * self.spacing)
                .normalize()
                .as_dvec3()
        })
    }

    /// The head's position, its eyes, and the point in the world.
    fn eyes(&self, video: &VideoParams) -> ([EyeView; 2], Vec3) {
        let head = Vec3::new(0.3, 1.6, -0.2);
        let point = head + video_axes(video).inverse() * self.point();
        let at = Quat::from_rotation_arc(Vec3::NEG_Z, (point - head).normalize());
        let turn = at
            * Quat::from_euler(
                EulerRot::YXZ,
                self.yaw.to_radians(),
                self.pitch.to_radians(),
                self.roll.to_radians(),
            );
        (eyes(head, turn, self.ipd), point)
    }
}

/// The two eyes' dots, measured and where the real point projects.
fn measure(
    r: &mut Renderer,
    e: &mut OffscreenEyes,
    frame: &Arc<fp_media::VideoFrame>,
    video: &VideoParams,
    pose: &Pose,
) -> [((f64, f64), (f64, f64)); 2] {
    let (views, point) = pose.eyes(video);
    let imgs = draw(r, e, frame, video, &views);
    [0, 1].map(|i| (peak(&imgs[i]), projected(&views[i], point)))
}

const ROLLED: Pose = Pose {
    spacing: 0.06,
    distance: 1.2,
    ipd: 0.07,
    yaw: 0.0,
    pitch: 0.0,
    roll: 20.0,
    anchor: 0.0,
    picture: [0.0; 3],
};

#[test]
fn stereo_focus_shows_a_point_at_its_distance_where_a_real_one_is() {
    let Some((mut r, mut e)) = setup() else {
        return;
    };
    let poses = [
        (
            "eyes wider than the lenses",
            Pose {
                roll: 0.0,
                ..ROLLED
            },
        ),
        ("rolled", ROLLED),
        (
            "turned, tilted and rolled",
            Pose {
                yaw: 18.0,
                pitch: -12.0,
                ..ROLLED
            },
        ),
        (
            "recentred aside, picture turned",
            Pose {
                yaw: -15.0,
                pitch: 10.0,
                roll: -25.0,
                ipd: 0.058,
                anchor: 70.0,
                picture: [20.0, -8.0, 4.0],
                ..ROLLED
            },
        ),
    ];
    let mut worst = 0.0f64;
    for projection in [Projection::EQUIRECT_180, Projection::fisheye(180.0)] {
        for (name, pose) in &poses {
            let sbs = VideoFormat::new(projection, StereoLayout::SideBySide);
            let frame = dots(projection, 1024, pose.seen(), 0.45, false);
            let video = pose.video(sbs, Focus::Manual, false);
            let got = measure(&mut r, &mut e, &frame, &video, pose);
            for (i, (seen, want)) in got.iter().enumerate() {
                let off = (seen.0 - want.0).hypot(seen.1 - want.1);
                worst = worst.max(off);
                assert!(
                    off < 0.03,
                    "{projection:?} {name}, eye {i}: dot at {seen:?}, a real point at {want:?}"
                );
            }
            // A real point is as high in both eyes.
            let vertical = got[0].0.1 - got[1].0.1;
            assert!(
                vertical.abs() < 0.03,
                "{name}: {vertical} px vertical disparity"
            );
        }
    }
    eprintln!("stereo focus: dots within {worst:.4} px of the real point");

    // Off, the rolled head sees the picture's disparity tilted: vertical
    // disparity the eyes can't fuse.
    let sbs = VideoFormat::new(Projection::EQUIRECT_180, StereoLayout::SideBySide);
    let frame = dots(Projection::EQUIRECT_180, 1024, ROLLED.seen(), 0.45, false);
    let got = measure(
        &mut r,
        &mut e,
        &frame,
        &ROLLED.video(sbs, Focus::Off, false),
        &ROLLED,
    );
    let vertical = got[0].0.1 - got[1].0.1;
    eprintln!("stereo focus off, rolled: {vertical:.2} px vertical disparity");
    assert!(vertical.abs() > 2.0, "{vertical} px");
    e.destroy(&r);
}

#[test]
fn each_eye_keeps_its_own_camera_in_a_right_eye_first_file() {
    let Some((mut r, mut e)) = setup() else {
        return;
    };
    let pose = Pose {
        yaw: 10.0,
        ..ROLLED
    };
    let frame = dots(Projection::EQUIRECT_180, 1024, pose.seen(), 0.45, true);
    let sbs = VideoFormat::new(Projection::EQUIRECT_180, StereoLayout::SideBySide);
    let labelled = VideoFormat {
        eyes_swapped: true,
        ..sbs
    };
    // Flagged by its format, or by the viewer's Swap eyes.
    for (name, video) in [
        ("format", pose.video(labelled, Focus::Manual, false)),
        ("swap eyes", pose.video(sbs, Focus::Manual, true)),
    ] {
        for (i, (seen, want)) in measure(&mut r, &mut e, &frame, &video, &pose)
            .iter()
            .enumerate()
        {
            let off = (seen.0 - want.0).hypot(seen.1 - want.1);
            assert!(off < 0.03, "{name}, eye {i}: {seen:?} vs {want:?}");
        }
    }
    e.destroy(&r);
}

#[test]
fn stereo_focus_changes_nothing_while_the_eyes_are_where_the_cameras_were() {
    let Some((mut r, mut e)) = setup() else {
        return;
    };
    // Busy pictures in both eyes.
    let w = 512u32;
    let rgba: Vec<u8> = (0..w * w / 2)
        .flat_map(|i| {
            let (x, y) = (i % w, i / w);
            let c = ((x * 7 + y * 13) % 255) as u8;
            [c, (x % 200) as u8, (y * 2 % 255) as u8, 255]
        })
        .collect();
    let f = Arc::new(fp_media::VideoFrame::from_rgba(w, w / 2, &rgba).unwrap());
    let head = Vec3::new(0.0, 1.6, 0.0);
    for format in [
        VideoFormat::new(Projection::EQUIRECT_180, StereoLayout::SideBySide),
        VideoFormat::new(Projection::fisheye(200.0), StereoLayout::TopBottom),
    ] {
        for (anchor, [yaw, pitch, roll]) in [(0.0f32, [0.0; 3]), (70.0, [20.0, -8.0, 4.0])] {
            let mut video = VideoParams {
                format,
                settings: ViewSettings {
                    lens_spacing: 64.0,
                    focus_distance: 0.6,
                    yaw,
                    pitch,
                    roll,
                    zoom: 1.2,
                    ..ViewSettings::default()
                },
                anchor: Quat::from_rotation_y(anchor.to_radians()),
                ..Default::default()
            };
            // Level, facing the picture's front, eyes as far apart as the
            // lenses.
            let views = eyes(head, video_axes(&video).inverse(), 0.064);
            let off = draw(&mut r, &mut e, &f, &video, &views);
            video.settings.focus = Focus::Manual;
            let on = draw(&mut r, &mut e, &f, &video, &views);
            assert!(off == on, "{format:?} anchor {anchor}");
            // Sanity: a tilted head does see a change.
            let tilted = eyes(
                head,
                video_axes(&video).inverse() * Quat::from_rotation_z(0.3),
                0.064,
            );
            let on = draw(&mut r, &mut e, &f, &video, &tilted);
            video.settings.focus = Focus::Off;
            let off = draw(&mut r, &mut e, &f, &video, &tilted);
            assert!(off != on, "{format:?}: focus did nothing when tilted");
        }
    }
    e.destroy(&r);
}
