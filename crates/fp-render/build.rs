//! Compiles the WGSL shaders to SPIR-V with naga (pure Rust, no external
//! tools), and embeds the cinema rooms.

use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Each SPIR-V module and the WGSL files joined into it: WGSL has no
/// includes, so shared code is put in front.
const MODULES: [(&str, &[&str]); 8] = [
    ("scene", &["hdr", "scene", "planes"]),
    ("scene_tp10", &["hdr", "scene", "planes_tp10"]),
    ("quad", &["quad"]),
    ("cinema", &["cinema"]),
    ("egui", &["egui"]),
    ("q10c", &["q10c"]),
    ("peak", &["hdr", "peak"]),
    ("focus", &["focus"]),
];

fn main() {
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    prepare_cinema(&out);
    for (name, files) in MODULES {
        let mut src = String::new();
        for file in files {
            let path = format!("shaders/{file}.wgsl");
            println!("cargo:rerun-if-changed={path}");
            src += &std::fs::read_to_string(&path).expect("shader source");
            src.push('\n');
        }
        let module = match naga::front::wgsl::parse_str(&src) {
            Ok(m) => m,
            Err(e) => panic!("{name} ({files:?}):\n{}", e.emit_to_string(&src)),
        };
        let info = match naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        {
            Ok(i) => i,
            Err(e) => panic!("{name}: validation failed:\n{}", e.emit_to_string(&src)),
        };
        let options = naga::back::spv::Options {
            lang_version: (1, 3),
            flags: naga::back::spv::WriterFlags::empty(),
            ..Default::default()
        };
        let words = naga::back::spv::write_vec(&module, &info, &options, None)
            .unwrap_or_else(|e| panic!("{name}: SPIR-V generation failed: {e}"));
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(out.join(format!("{name}.spv")), bytes).expect("write spv");
    }
}

/// A room's `params.json`: its screen and seats in metres, the origin
/// being one seat's eyes, looking along -Z.
#[derive(Deserialize)]
struct RoomParams {
    screen_width: f32,
    screen_height: f32,
    screen_center: [f32; 3],
    /// The most the screen may curve, 1 over its radius in metres.
    max_curvature: f32,
    /// Eye positions on or near the screen's centre line, one per row.
    seats: Vec<[f32; 3]>,
}

fn prepare_cinema(out: &Path) {
    let rooms = ["classic", "luxe"].map(cinema_asset);
    std::fs::write(
        out.join("cinema_assets.rs"),
        format!(
            "const ASSETS: [Option<CinemaAsset>; 2] = [{}];\n",
            rooms.join(", ")
        ),
    )
    .expect("write cinema asset table");
}

/// The room in `rooms/<name>`, if it is there (`tools/cinema/make_cinema.py`
/// writes it): its mesh file, and its textures in the order the mesh file
/// counts them: `cinema-hall.jpg`, then `props.jpg` and `detail.png` where
/// the room has them.
fn cinema_asset(name: &str) -> String {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("rooms")
        .join(name);
    println!("cargo:rerun-if-changed={}", dir.display());
    let metadata = dir.join("params.json");
    if !metadata.exists() {
        return "None".to_string();
    }
    let json = std::fs::read_to_string(&metadata)
        .unwrap_or_else(|e| panic!("reading {}: {e}", metadata.display()));
    let room: RoomParams =
        serde_json::from_str(&json).unwrap_or_else(|e| panic!("{name} params.json: {e}"));
    let finite = |v: &[f32; 3]| v.iter().all(|c| c.is_finite());
    assert!(
        room.screen_width > 0.0
            && room.screen_height > 0.0
            && finite(&room.screen_center)
            && room.max_curvature.is_finite()
            && room.max_curvature >= 0.0
            && !room.seats.is_empty()
            && room
                .seats
                .iter()
                .all(|seat| finite(seat) && seat[2] > room.screen_center[2]),
        "{name} params.json: needs a screen and at least one seat in front of it"
    );
    let embed = |file: &str| {
        let path = dir.join(file);
        assert!(path.exists(), "{} is missing", path.display());
        format!("include_bytes!({:?})", path.display().to_string())
    };
    let textures: Vec<String> = ["cinema-hall.jpg", "props.jpg", "detail.png"]
        .into_iter()
        .filter(|file| *file == "cinema-hall.jpg" || dir.join(file).exists())
        .map(embed)
        .collect();
    let seats: Vec<String> = room.seats.iter().map(|s| format!("{s:?}")).collect();
    format!(
        r#"Some(CinemaAsset {{
            geometry: {},
            textures: &[{}],
            screen_width: {:?},
            screen_height: {:?},
            screen_center: {:?},
            max_curvature: {:?},
            seats: &[{}],
        }})"#,
        embed("cinema.bin"),
        textures.join(", "),
        room.screen_width,
        room.screen_height,
        room.screen_center,
        room.max_curvature,
        seats.join(", "),
    )
}
