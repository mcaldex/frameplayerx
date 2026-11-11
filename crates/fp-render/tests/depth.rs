//! The depth measurement pass on a real (software) Vulkan device: the grids
//! it draws agree with fp-core's CPU reference sampler, it waits for a 3D
//! frame and comes back without anything waiting, and where a look lands in
//! the video is where the eye passes show it.

use fp_core::depth::grid::{self, OUTSIDE, SEE_THROUGH};
use fp_core::depth::picture::{Frame, PictureGeometry};
use fp_core::depth::scenes::{Picture, Rig, Scene, Shape, Surface, Texture};
use fp_core::depth::{Angles, GridSpec, Grids, Matcher, Request, Window};
use fp_core::format::{Projection, StereoLayout, VideoFormat};
use fp_core::view::ViewSettings;
use fp_render::capture::OffscreenEyes;
use fp_render::math::{Fov, projection, view};
use fp_render::params::look_in_video;
use fp_render::{EyeView, Gpu, Renderer, VideoParams};
use glam::{EulerRot, Quat, Vec3};
use std::sync::Arc;

mod common;
use common::{dots, peak};

const SIZE: u32 = 256;

fn setup(size: u32) -> Option<(Renderer, OffscreenEyes)> {
    let gpu = match Gpu::headless() {
        Ok(g) => Arc::new(g),
        Err(e) => {
            eprintln!("skipping: no Vulkan device ({e})");
            return None;
        }
    };
    let r = Renderer::new(gpu, ash::vk::Format::R8G8B8A8_SRGB).expect("renderer");
    let eyes = OffscreenEyes::new(&r, size, size).expect("eyes");
    Some((r, eyes))
}

fn eyes_ahead() -> [EyeView; 2] {
    let ev = |x: f32| EyeView {
        view: view(Vec3::new(x, 1.6, 0.0), Quat::IDENTITY),
        proj: projection(Fov::symmetric(90.0), 0.05, 100.0),
    };
    [ev(-0.032), ev(0.032)]
}

/// One frame drawn: `ask` asks for a measurement first.
fn frame(
    r: &mut Renderer,
    e: &OffscreenEyes,
    f: &Arc<fp_media::VideoFrame>,
    video: &VideoParams,
    ask: Option<(u64, &GridSpec)>,
) -> bool {
    r.begin_frame().unwrap();
    r.set_video(Some(f)).unwrap();
    let asked = ask.is_some_and(|(tag, spec)| r.ask_depth(tag, spec));
    r.draw(&e.targets(), &eyes_ahead(), video, &[]).unwrap();
    r.end_frame().unwrap();
    asked
}

/// A grey frame of `luma` (0..1, row by row), as 8-bit RGBA, and the
/// luminance the GPU reads from it: the same 8-bit levels.
fn grey(luma: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let levels: Vec<u8> = luma.iter().map(|l| (l * 255.0).round() as u8).collect();
    let rgba = levels.iter().flat_map(|&l| [l, l, l, 255]).collect();
    (rgba, levels.iter().map(|&l| l as f32 / 255.0).collect())
}

/// Each sample's kind: 0 luminance, 1 outside, 2 see-through.
fn kind(v: f32) -> u8 {
    if v == OUTSIDE {
        1
    } else if v == SEE_THROUGH {
        2
    } else {
        0
    }
}

/// Whether any sample within two of (`col`, `row`) of `g` differs in kind
/// from it: a sample by a boundary. A tap a hair over the picture's edge
/// changes a sample's kind; and the video's chroma is stored at half
/// size, so on the GPU a chroma key's edge spreads by up to two of its
/// pixels (about two samples) where the reference's opacity is sharp.
fn boundary(g: &[f32], spec: &GridSpec, col: usize, row: usize) -> bool {
    let k = kind(g[row * spec.cols + col]);
    (row.saturating_sub(2)..=(row + 2).min(spec.rows - 1)).any(|r| {
        (col.saturating_sub(2)..=(col + 2).min(spec.cols - 1))
            .any(|c| kind(g[r * spec.cols + c]) != k)
    })
}

/// How the GPU's grids differ from the reference: samples of another kind
/// off a boundary, the largest luminance difference off one, and how many
/// samples were outside and see-through.
fn compare(gpu: &Grids, cpu: &Grids) -> (usize, f32, [usize; 2]) {
    let spec = &cpu.spec;
    let (mut wrong, mut worst, mut kinds) = (0, 0.0f32, [0, 0]);
    for camera in 0..2 {
        let (g, c) = (&gpu.samples[camera], &cpu.samples[camera]);
        for row in 0..spec.rows {
            for col in 0..spec.cols {
                let i = row * spec.cols + col;
                match kind(c[i]) {
                    1 => kinds[0] += 1,
                    2 => kinds[1] += 1,
                    _ => {}
                }
                if boundary(c, spec, col, row) {
                    continue;
                }
                if kind(g[i]) != kind(c[i]) {
                    wrong += 1;
                } else if kind(c[i]) == 0 && !boundary(g, spec, col, row) {
                    // (Beside the GPU's wider key edge, chroma half green
                    // takes red and blue below 0, and their clamp lifts a
                    // dark sample's luminance.)
                    worst = worst.max((g[i] - c[i]).abs());
                }
            }
        }
    }
    for (g, c) in gpu.coarse.iter().zip(&cpu.coarse) {
        worst = worst.max((g - c).abs());
    }
    (wrong, worst, kinds)
}

#[test]
fn grids_drawn_on_the_gpu_agree_with_the_cpu_reference() {
    let Some((mut r, e)) = setup(64) else {
        return;
    };
    // Something to see at two depths, its finest detail some 4 pixels
    // across, as filmed video's is: on detail a pixel across, a tap moved
    // a hundredth of a pixel (the GPU's arc tangents may be thousands of
    // ULP off) changes what it reads by a twentieth.
    let scene = Scene::new(vec![
        Surface::new(
            Shape::Disc {
                centre: [0.3, 0.1, -0.8],
                normal: [0.0, 0.0, 1.0],
                radius: 0.15,
            },
            Texture::Noise {
                feature: 0.012,
                contrast: 1.0,
                seed: 3,
            },
        ),
        // A wall of 2 m round, so looking up sees the sky rather than the
        // wall edge-on, its texture squeezed below a pixel.
        Surface::new(
            Shape::Disc {
                centre: [0.0, 0.0, -1.5],
                normal: [0.0, 0.0, 1.0],
                radius: 2.0,
            },
            Texture::Noise {
                feature: 0.02,
                contrast: 1.0,
                seed: 4,
            },
        ),
    ]);
    let aims = [
        (
            "ahead",
            Angles {
                alpha: std::f32::consts::FRAC_PI_2,
                beta: 0.0,
            },
        ),
        (
            "off-centre",
            Angles {
                alpha: 1.05,
                beta: 0.35,
            },
        ),
        (
            "side",
            Angles {
                alpha: 0.48,
                beta: -0.1,
            },
        ),
    ];
    // Towards the top: an equirectangular picture's top edge is a pole,
    // where longitude is a ratio of two vanishing numbers and the GPU's
    // and the CPU's rounding pick different texels, so its top is tried
    // short of it; a fisheye's circle edge is well conditioned, so its
    // is tried past it.
    let top = |p: &Projection| {
        let beta = if matches!(p, Projection::Fisheye { .. }) {
            1.5
        } else {
            1.3
        };
        (
            "top",
            Angles {
                alpha: std::f32::consts::FRAC_PI_2,
                beta,
            },
        )
    };
    let centres = ViewSettings {
        lens_centre: [[0.02, -0.01], [-0.015, 0.02]],
        lens_k1: 0.03,
        ..ViewSettings::default()
    };
    let sbs = |p| VideoFormat::new(p, StereoLayout::SideBySide);
    let cases = [
        (
            "180 side by side",
            sbs(Projection::EQUIRECT_180),
            ViewSettings::default(),
            1024,
            512,
            0,
        ),
        (
            "180 swapped, odd size",
            VideoFormat {
                eyes_swapped: true,
                ..sbs(Projection::EQUIRECT_180)
            },
            ViewSettings::default(),
            1001,
            500,
            0,
        ),
        (
            "180 top/bottom",
            VideoFormat::new(Projection::EQUIRECT_180, StereoLayout::TopBottom),
            ViewSettings::default(),
            600,
            1200,
            0,
        ),
        (
            "fisheye, lens centres moved",
            sbs(Projection::fisheye(180.0)),
            centres,
            1280,
            640,
            0,
        ),
        (
            "180 stored turned",
            sbs(Projection::EQUIRECT_180),
            ViewSettings {
                swap_eyes: true,
                ..ViewSettings::default()
            },
            512,
            1024,
            90,
        ),
    ];
    let mut tag = 0;
    let mut worst_all = 0.0f32;
    let mut kinds_all = [0, 0];
    for (name, format, settings, w, h, rotation) in cases {
        let g = PictureGeometry::new(&format, &settings, w as u32, h as u32, 1.0, rotation);
        let picture = Picture::new([&scene, &scene], Rig::new(0.065), g, w, h);
        let (luma, _) = picture.render();
        let (mut rgba, mut levels) = grey(&luma);
        // A green screen seen by the left camera only, round straight ahead.
        let mut alpha = vec![1.0f32; w * h];
        let (a, b) = (g.picture_uv(0, [0.47, 0.45]), g.picture_uv(0, [0.53, 0.55]));
        let (lo, hi) = (
            [a[0].min(b[0]), a[1].min(b[1])],
            [a[0].max(b[0]), a[1].max(b[1])],
        );
        for y in 0..h {
            for x in 0..w {
                let s = [(x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32];
                if (lo[0]..hi[0]).contains(&s[0]) && (lo[1]..hi[1]).contains(&s[1]) {
                    let i = y * w + x;
                    rgba[i * 4..i * 4 + 3].copy_from_slice(&[0, 255, 0]);
                    levels[i] = 0.7152;
                    alpha[i] = 0.0;
                }
            }
        }
        let f = Arc::new(fp_media::VideoFrame::from_rgba(w as u32, h as u32, &rgba).unwrap());
        let video = VideoParams {
            format,
            settings: ViewSettings {
                chroma_key: true,
                key_color: [0.0, 1.0, 0.0],
                ..settings
            },
            rotation,
            ..Default::default()
        };
        let reference = Frame {
            width: w,
            height: h,
            luma: &levels,
            alpha: Some(&alpha),
        };
        for (aim_name, aim) in aims.into_iter().chain([top(&format.projection)]) {
            let plan = Matcher::new()
                .plan(&Request {
                    aim,
                    baseline: 0.065,
                    nearest: 0.3,
                    window: Window::HEAD,
                })
                .unwrap();
            tag += 1;
            assert!(frame(&mut r, &e, &f, &video, Some((tag, &plan.grid))));
            frame(&mut r, &e, &f, &video, None);
            frame(&mut r, &e, &f, &video, None);
            let (got_tag, gpu) = r.take_depth().expect("back within two frames");
            assert_eq!(got_tag, tag);
            let cpu = grid::sample(&reference, &g, &plan.grid);
            let (wrong, worst, kinds) = compare(&gpu, &cpu);
            let label = format!("{name}, {aim_name}");
            assert_eq!(wrong, 0, "{label}: samples of another kind off a boundary");
            // The GPU filters 8-bit levels with 8-bit weights, under 0.002
            // a tap; and its sin, cos and arc cosine are about 1e-4 off
            // (Vulkan allows 2^-11), which moves a tap's longitude by that
            // over cos(beta) and a fisheye's radius by about as much: 0.03
            // of a pixel ahead, 0.1 at 75° up, a few thousandths of
            // luminance over this picture's detail.
            let bound = 0.006 / aim.beta.cos().max(0.2);
            assert!(
                worst <= bound,
                "{label}: luminance off by {worst}, more than {bound}"
            );
            worst_all = worst_all.max(worst * aim.beta.cos().max(0.2));
            kinds_all[0] += kinds[0];
            kinds_all[1] += kinds[1];
        }
    }
    eprintln!(
        "GPU grids: luminance within {worst_all:.4} / cos(beta) of the reference; \
         {} samples outside, {} see-through",
        kinds_all[0], kinds_all[1]
    );
    assert!(
        kinds_all[0] > 0 && kinds_all[1] > 0,
        "both kinds were tried"
    );
}

#[test]
fn a_measurement_waits_for_a_3d_frame_and_comes_back_without_waiting() {
    let Some((mut r, e)) = setup(32) else {
        return;
    };
    let rgba: Vec<u8> = (0..256 * 128)
        .flat_map(|i| [(i % 251) as u8, 90, 40, 255])
        .collect();
    let f = Arc::new(fp_media::VideoFrame::from_rgba(256, 128, &rgba).unwrap());
    let flat = VideoParams {
        format: VideoFormat::new(Projection::Flat, StereoLayout::SideBySide),
        ..Default::default()
    };
    let vr = VideoParams {
        format: VideoFormat::FALLBACK,
        ..Default::default()
    };
    let spec = Matcher::new()
        .plan(&Request {
            aim: Angles {
                alpha: std::f32::consts::FRAC_PI_2,
                beta: 0.0,
            },
            baseline: 0.065,
            nearest: 0.4,
            window: Window::HEAD,
        })
        .unwrap()
        .grid;
    assert!(r.measures_depth());
    // Asked for, it waits through frames of a flat video.
    assert!(frame(&mut r, &e, &f, &flat, Some((1, &spec))));
    assert!(!r.ask_depth(2, &spec), "one at a time");
    for _ in 0..3 {
        frame(&mut r, &e, &f, &flat, None);
        assert!(r.take_depth().is_none() && r.depth_pending());
    }
    // Drawn from the first 3D frame, back two frames later.
    frame(&mut r, &e, &f, &vr, None);
    assert!(!r.ask_depth(2, &spec), "on its way");
    frame(&mut r, &e, &f, &vr, None);
    assert!(r.take_depth().is_none());
    frame(&mut r, &e, &f, &vr, None);
    let (tag, grids) = r.take_depth().expect("back");
    assert_eq!(tag, 1);
    assert_eq!(grids.samples[0].len(), spec.len());
    assert!(!r.depth_pending() && r.take_depth().is_none());
    // A cancelled one never comes back.
    assert!(frame(&mut r, &e, &f, &vr, Some((3, &spec))));
    r.cancel_depth();
    for _ in 0..3 {
        frame(&mut r, &e, &f, &vr, None);
        assert!(r.take_depth().is_none() && !r.depth_pending());
    }
}

#[test]
fn a_look_lands_in_the_video_where_the_function_says() {
    let Some((mut r, mut e)) = setup(SIZE) else {
        return;
    };
    let head = Quat::from_euler(EulerRot::YXZ, 0.35, -0.2, 0.1);
    let look = head * Vec3::NEG_Z;
    let mut worst = 0.0f64;
    for (name, settings, anchor) in [
        ("plain", ViewSettings::default(), 0.0f32),
        (
            "turned, recentred, raised, forward, zoomed",
            ViewSettings {
                yaw: 15.0,
                pitch: -10.0,
                roll: 6.0,
                height: 0.3,
                forward: 0.4,
                zoom: 1.3,
                ..ViewSettings::default()
            },
            40.0,
        ),
    ] {
        let video = VideoParams {
            format: VideoFormat::FALLBACK,
            settings,
            anchor: Quat::from_rotation_y(anchor.to_radians()),
            ..Default::default()
        };
        let at = look_in_video(&video, look).as_dvec3();
        let f = dots(Projection::EQUIRECT_180, 1024, [at, at], 0.45, false);
        let views = [-0.032f32, 0.032].map(|x| EyeView {
            view: view(Vec3::new(0.0, 1.6, 0.0) + head * Vec3::X * x, head),
            proj: projection(Fov::symmetric(60.0), 0.05, 100.0),
        });
        r.begin_frame().unwrap();
        r.set_video(Some(&f)).unwrap();
        r.draw(&e.targets(), &views, &video, &[]).unwrap();
        r.end_frame().unwrap();
        let middle = SIZE as f64 / 2.0 - 0.5;
        for eye in 0..2 {
            let (x, y) = peak(&e.read(&r, eye).unwrap(), SIZE);
            let off = (x - middle).hypot(y - middle);
            worst = worst.max(off);
            assert!(off < 0.03, "{name}, eye {eye}: at ({x:.3}, {y:.3})");
        }
    }
    eprintln!("look in video: within {worst:.4} px of the middle");
    e.destroy(&r);
}
