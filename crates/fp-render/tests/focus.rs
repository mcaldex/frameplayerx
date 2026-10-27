//! Autofocus's reading, on a real (software) Vulkan device: 3D 180° videos
//! of a textured wall the eyes see a known disparity apart, read straight
//! ahead or where the eyes look.

use fp_core::format::{Projection, StereoLayout, VideoFormat};
use fp_core::view::ViewSettings;
use fp_render::capture::OffscreenEyes;
use fp_render::math::{Fov, projection, view};
use fp_render::{EyeView, Focus, Gpu, Renderer, VideoParams};
use glam::{Quat, Vec3};
use std::sync::Arc;

const SIZE: u32 = 128;
/// Each eye's picture, pixels: 180° across and down.
const EYE: u32 = 1024;

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

/// Smooth value noise, one random level per unit cell.
fn noise(x: f32, y: f32) -> f32 {
    let level = |a: i32, b: i32| {
        let mut n = (a as u32).wrapping_mul(0x9E37_79B1) ^ (b as u32).wrapping_mul(0x85EB_CA77);
        n ^= n >> 15;
        n = n.wrapping_mul(0x2C1B_3C6D);
        n ^= n >> 12;
        (n & 0xffff) as f32 / 65535.0
    };
    let (i, j) = (x.floor() as i32, y.floor() as i32);
    let ease = |t: f32| t * t * (3.0 - 2.0 * t);
    let (sx, sy) = (ease(x - x.floor()), ease(y - y.floor()));
    let top = level(i, j) + (level(i + 1, j) - level(i, j)) * sx;
    let bottom = level(i, j + 1) + (level(i + 1, j + 1) - level(i, j + 1)) * sx;
    top + (bottom - top) * sy
}

/// A wall seen at `lon`, `lat` degrees.
fn wall(lon: f32, lat: f32) -> u8 {
    (30.0 + 195.0 * noise(lon / 1.2, lat / 1.2)) as u8
}

/// Side-by-side 180°: the left eye sees the wall `disparity` degrees to
/// the right of where the right eye does, as a camera pair sees something
/// near.
fn stereo_wall(disparity: f32) -> Arc<fp_media::VideoFrame> {
    stereo_scene(|_| disparity)
}

/// Like [`stereo_wall`], `disparity(longitude)` degrees apart, longitude in
/// degrees right of the front.
fn stereo_scene(disparity: impl Fn(f32) -> f32) -> Arc<fp_media::VideoFrame> {
    let (w, h) = (EYE * 2, EYE / 2);
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        let lat = (0.5 - (y as f32 + 0.5) / h as f32) * 180.0;
        for x in 0..w {
            let lon = ((x % EYE) as f32 + 0.5) / EYE as f32 * 180.0 - 90.0;
            let shift = if x < EYE { -0.5 } else { 0.5 } * disparity(lon);
            let g = wall(lon + shift, lat);
            rgba.extend_from_slice(&[g, g, g, 255]);
        }
    }
    Arc::new(fp_media::VideoFrame::from_rgba(w, h, &rgba).expect("frame"))
}

fn eyes(head: Quat) -> [EyeView; 2] {
    let ev = |x: f32| EyeView {
        view: view(head * Vec3::new(x, 0.0, 0.0) + Vec3::Y * 1.6, head),
        proj: projection(Fov::symmetric(90.0), 0.05, 100.0),
    };
    [ev(-0.032), ev(0.032)]
}

fn video(stereo: StereoLayout, settings: ViewSettings) -> VideoParams {
    VideoParams {
        format: VideoFormat::new(Projection::EQUIRECT_180, stereo),
        settings,
        focus: Some(Vec3::NEG_Z),
        ..Default::default()
    }
}

/// Draws a few frames and returns the reading that came back, with both
/// eyes' pictures of the last.
fn look(
    r: &mut Renderer,
    e: &mut OffscreenEyes,
    f: &Arc<fp_media::VideoFrame>,
    p: &VideoParams,
    head: Quat,
) -> (Option<Focus>, [Vec<u8>; 2]) {
    let mut got = None;
    for _ in 0..3 {
        r.begin_frame().unwrap();
        got = r.focus().or(got);
        r.set_video(Some(f)).unwrap();
        r.draw(&e.targets(), &eyes(head), p, &[]).unwrap();
        r.end_frame().unwrap();
    }
    (got, [e.read(r, 0).unwrap(), e.read(r, 1).unwrap()])
}

/// How far apart the two eyes' pictures are in the middle of the view,
/// 0..255 per pixel.
fn mismatch([l, r]: &[Vec<u8>; 2]) -> f32 {
    let mut sum = 0.0;
    let mut n = 0.0;
    for y in SIZE / 2 - 8..SIZE / 2 + 8 {
        for x in SIZE / 2 - 8..SIZE / 2 + 8 {
            let i = ((y * SIZE + x) * 4) as usize;
            sum += (l[i] as f32 - r[i] as f32).abs();
            n += 1.0;
        }
    }
    sum / n
}

#[test]
fn reads_how_far_apart_the_eyes_see_what_is_ahead() {
    let Some((mut r, mut e)) = setup() else {
        return;
    };
    for (disparity, head) in [
        (6.0, Quat::IDENTITY),
        (0.0, Quat::IDENTITY),
        (14.5, Quat::IDENTITY),
        // Looking up and to the side reads what is there.
        (9.0, Quat::from_rotation_y(0.4) * Quat::from_rotation_x(0.3)),
    ] {
        let f = stereo_wall(disparity);
        let p = VideoParams {
            focus: Some(head * Vec3::NEG_Z),
            ..video(StereoLayout::SideBySide, ViewSettings::default())
        };
        let (got, _) = look(&mut r, &mut e, &f, &p, head);
        let got = got.unwrap_or_else(|| panic!("{disparity}°: no reading"));
        assert!(
            (got.disparity - disparity).abs() < 0.3,
            "{disparity}°: read {got:?}"
        );
    }
    e.destroy(&r);
}

#[test]
fn reads_where_the_eyes_look_not_where_the_head_points() {
    let Some((mut r, mut e)) = setup() else {
        return;
    };
    // Something near on the left, further on the right; the head faces the
    // front, where they meet.
    let f = stereo_scene(|lon| if lon < 0.0 { 12.0 } else { 4.0 });
    for (yaw, disparity) in [(25.0f32, 12.0), (-25.0, 4.0)] {
        let gaze = Quat::from_rotation_y(yaw.to_radians()) * Vec3::NEG_Z;
        let p = VideoParams {
            focus: Some(gaze),
            ..video(StereoLayout::SideBySide, ViewSettings::default())
        };
        let (got, _) = look(&mut r, &mut e, &f, &p, Quat::IDENTITY);
        let got = got.unwrap_or_else(|| panic!("looking {yaw}°: no reading"));
        assert!(
            (got.disparity - disparity).abs() < 0.3,
            "looking {yaw}°: read {got:?}"
        );
    }
    e.destroy(&r);
}

#[test]
fn that_much_depth_brings_the_eyes_together() {
    let Some((mut r, mut e)) = setup() else {
        return;
    };
    let f = stereo_wall(8.0);
    let p = video(StereoLayout::SideBySide, ViewSettings::default());
    let (got, apart) = look(&mut r, &mut e, &f, &p, Quat::IDENTITY);
    let got = got.expect("a reading");
    let fused = VideoParams {
        settings: ViewSettings {
            ipd_offset: got.disparity,
            ..Default::default()
        },
        ..p
    };
    let (again, together) = look(&mut r, &mut e, &f, &fused, Quat::IDENTITY);
    // The reading is the video's own, whatever Depth shows it with.
    assert!((again.expect("a reading").disparity - got.disparity).abs() < 0.05);
    assert!(
        mismatch(&together) < mismatch(&apart) / 3.0,
        "{} vs {}",
        mismatch(&together),
        mismatch(&apart)
    );
    e.destroy(&r);
}

#[test]
fn reads_nothing_it_cannot_trust() {
    let Some((mut r, mut e)) = setup() else {
        return;
    };
    // A 2D video has one picture; a blank 3D one has nothing to match.
    let f = stereo_wall(6.0);
    let p = video(StereoLayout::Mono, ViewSettings::default());
    assert_eq!(look(&mut r, &mut e, &f, &p, Quat::IDENTITY).0, None);
    let blank = Arc::new(
        fp_media::VideoFrame::from_rgba(256, 128, &[128u8; 256 * 128 * 4]).expect("frame"),
    );
    let p = video(StereoLayout::SideBySide, ViewSettings::default());
    assert_eq!(look(&mut r, &mut e, &blank, &p, Quat::IDENTITY).0, None);
    // Not asked, nothing is read.
    let p = VideoParams { focus: None, ..p };
    assert_eq!(look(&mut r, &mut e, &f, &p, Quat::IDENTITY).0, None);
    e.destroy(&r);
}
