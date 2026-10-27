//! Pictures decoded into the renderer's decode memory reach the eyes as they
//! do from ordinary memory, the GPU copying them as they are, on a real
//! (software) Vulkan device.

use fp_core::format::{Projection, StereoLayout, VideoFormat};
use fp_core::source::FileSource;
use fp_render::capture::OffscreenEyes;
use fp_render::math::{Fov, projection, view};
use fp_render::{EyeView, Gpu, Renderer, VideoParams};
use glam::{Quat, Vec3};
use std::sync::Arc;

const SIZE: u32 = 128;

fn first_frame() -> Arc<fp_media::VideoFrame> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fp-media/tests/data/h264_aac_180_LR.mp4");
    let src = Arc::new(FileSource::open(&path).unwrap());
    Arc::new(fp_media::thumb::decode_first_frame(src, "x.mp4").unwrap())
}

fn show(r: &mut Renderer, e: &mut OffscreenEyes, f: &Arc<fp_media::VideoFrame>) -> [Vec<u8>; 2] {
    let ev = |x: f32| EyeView {
        view: view(Vec3::new(x, 1.6, 0.0), Quat::IDENTITY),
        proj: projection(Fov::symmetric(90.0), 0.05, 100.0),
    };
    let p = VideoParams {
        format: VideoFormat::new(Projection::EQUIRECT_180, StereoLayout::SideBySide),
        ..Default::default()
    };
    r.begin_frame().unwrap();
    r.set_video(Some(f)).unwrap();
    r.draw(&e.targets(), &[ev(-0.032), ev(0.032)], &p, &[])
        .unwrap();
    r.end_frame().unwrap();
    [e.read(r, 0).unwrap(), e.read(r, 1).unwrap()]
}

/// One test, as the frame memory is the whole process's.
#[test]
fn pictures_in_decode_memory_show_as_from_ordinary_memory() {
    let gpu = match Gpu::headless() {
        Ok(g) => Arc::new(g),
        Err(e) => {
            eprintln!("skipping: no Vulkan device ({e})");
            return;
        }
    };
    let mut r = Renderer::new(gpu, ash::vk::Format::R8G8B8A8_SRGB).expect("renderer");
    let mut e = OffscreenEyes::new(&r, SIZE, SIZE).expect("eyes");
    let ordinary = first_frame();
    let want = show(&mut r, &mut e, &ordinary);
    assert_eq!(Arc::strong_count(&ordinary), 1, "copied, not kept");

    fp_media::set_frame_memory(Some(r.frame_memory().expect("decode memory")));
    let in_memory = first_frame();
    fp_media::set_frame_memory(None);
    assert_eq!(show(&mut r, &mut e, &in_memory), want);
    assert_eq!(
        Arc::strong_count(&in_memory),
        2,
        "kept while the GPU may still copy it"
    );
    // The set it was copied into is uploaded again two pictures on.
    show(&mut r, &mut e, &first_frame());
    show(&mut r, &mut e, &first_frame());
    assert_eq!(Arc::strong_count(&in_memory), 1, "let go");
}
