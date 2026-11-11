// One depth measurement's samples (fp_core::depth::grid), read from the
// frame the eye passes show through their own code (picture.wgsl, then
// planes.wgsl or planes_tp10.wgsl; build.rs joins them), with the scene's
// own uniform block. Without the viewer's picture adjustments, stereo
// focus or per-eye tweaks: the video as filmed.
//
// One fragment a sample, into a single-channel float target: each
// camera's grid side by side from the left (left camera first), then the
// coarse picture of the left camera's view. Luminance is 0..1; a sample
// outside the picture is -1 and a see-through one -2.

struct Measure {
    // Each camera's first sample's (alpha, beta), radians: the left
    // camera's in xy, the right's in zw
    origin: vec4<f32>,
    // x: radians between samples, y: columns, z: rows of each grid
    grid: vec4<f32>,
};

var<immediate> m: Measure;

// fp_core::depth::grid's TAPS, COARSE, COARSE_TAPS, OUTSIDE, SEE_THROUGH
// and OPAQUE.
const TAPS: u32 = 3u;
const COARSE: u32 = 32u;
const COARSE_TAPS: u32 = 2u;
const OUTSIDE: f32 = -1.0;
const SEE_THROUGH: f32 = -2.0;
const OPAQUE: f32 = 0.5;

// fp_core::depth::Angles::direction.
fn direction(a: vec2<f32>) -> vec3<f32> {
    return vec3<f32>(cos(a.x), sin(a.x) * sin(a.y), -sin(a.x) * cos(a.y));
}

// Perceptual luminance of the video at `uv` of camera `camera`'s picture,
// as filmed (the gamma-encoded luma of its linear light, HDR tone
// mapped), and how opaque it is there.
fn seen(uv: vec2<f32>, camera: u32) -> vec2<f32> {
    let rgb = sample_yuv(stored_uv(to_eye_rect(uv, camera)));
    var alpha = 1.0;
    if (params.key2.y > 0.5) {
        alpha = chroma_key(rgb).a;
    }
    if (params.key2.z > 0.0) {
        alpha *= packed_mask(uv, camera);
    }
    let g = pow(max(linear_light(rgb), vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
    return vec2<f32>(dot(g, vec3<f32>(0.2126, 0.7152, 0.0722)), alpha);
}

// Where tap `t` of `n` x `n` is in its cell, in cells from its middle.
fn tap(t: u32, n: u32) -> vec2<f32> {
    return (vec2<f32>(f32(t % n), f32(t / n)) + 0.5) / f32(n) - 0.5;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let px = vec2<u32>(in.pos.xy);
    let cols = u32(m.grid.y);
    let rows = u32(m.grid.z);
    if (px.x < 2u * cols) {
        let camera = px.x / cols;
        if (px.y >= rows) {
            return vec4<f32>(0.0);
        }
        let cell = vec2<f32>(f32(px.x % cols), f32(px.y));
        let origin = select(m.origin.xy, m.origin.zw, camera == 1u);
        var luma = 0.0;
        var alpha = 0.0;
        for (var t = 0u; t < TAPS * TAPS; t++) {
            let uv = sphere_uv(direction(origin + (cell + tap(t, TAPS)) * m.grid.x), camera);
            // A cell over the picture's edge can't be matched.
            if (uv.x < 0.0) {
                return vec4<f32>(OUTSIDE, 0.0, 0.0, 0.0);
            }
            let s = seen(uv, camera);
            luma += s.x;
            alpha += s.y;
        }
        let n = f32(TAPS * TAPS);
        if (alpha / n < OPAQUE) {
            return vec4<f32>(SEE_THROUGH, 0.0, 0.0, 0.0);
        }
        return vec4<f32>(luma / n, 0.0, 0.0, 0.0);
    }
    let c = px - vec2<u32>(2u * cols, 0u);
    if (c.x >= COARSE || c.y >= COARSE) {
        return vec4<f32>(0.0);
    }
    var sum = 0.0;
    for (var t = 0u; t < COARSE_TAPS * COARSE_TAPS; t++) {
        sum += seen((vec2<f32>(c) + 0.5 + tap(t, COARSE_TAPS)) / f32(COARSE), 0u).x;
    }
    return vec4<f32>(sum / f32(COARSE_TAPS * COARSE_TAPS), 0.0, 0.0, 0.0);
}
