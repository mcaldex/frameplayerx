//! Compiles the WGSL shaders to SPIR-V with naga (pure Rust, no external tools).

use std::path::PathBuf;

fn main() {
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    // The scene shader's cinema lives in a file of its own.
    for (name, files) in [
        ("scene", &["scene", "cinema"][..]),
        ("quad", &["quad"]),
        ("egui", &["egui"]),
    ] {
        let path = format!("shaders/{name}.wgsl");
        let mut src = String::new();
        for file in files {
            let file = format!("shaders/{file}.wgsl");
            println!("cargo:rerun-if-changed={file}");
            src += &std::fs::read_to_string(&file).expect("shader source");
        }
        let module = match naga::front::wgsl::parse_str(&src) {
            Ok(m) => m,
            Err(e) => panic!("{path}:\n{}", e.emit_to_string(&src)),
        };
        let info = match naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        {
            Ok(i) => i,
            Err(e) => panic!("{path}: validation failed:\n{}", e.emit_to_string(&src)),
        };
        let options = naga::back::spv::Options {
            lang_version: (1, 3),
            flags: naga::back::spv::WriterFlags::empty(),
            ..Default::default()
        };
        let words = naga::back::spv::write_vec(&module, &info, &options, None)
            .unwrap_or_else(|e| panic!("{path}: SPIR-V generation failed: {e}"));
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(out.join(format!("{name}.spv")), bytes).expect("write spv");
    }
}
