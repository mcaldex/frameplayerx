// How the eye passes read a video's picture, shared with the depth
// measurement pass so it reads a frame exactly as they do (build.rs puts
// this file in front of scene.wgsl and measure.wgsl, each followed by
// planes.wgsl or planes_tp10.wgsl): the uniform block and its bindings,
// the full-screen triangle, the projections, the eye's half of the frame,
// the frame's turn, the chroma key and packed mask, and the transfer to
// linear light.

struct Params {
    inv_view_proj: array<mat4x4<f32>, 2>,
    // Rotation applied to rays in the room frame before projection
    // (inverse of the video's yaw/pitch/roll correction).
    correction: mat4x4<f32>,
    // World -> the room frame the video was centred in (inverse of the
    // recentering anchor's rotation).
    anchor_inv: mat4x4<f32>,
    // World -> screen-local transform for flat screens.
    screen_inv: mat4x4<f32>,
    // x: projection kind (0 flat, 1 equirect, 2 fisheye, 3 eac)
    // y: stereo (0 mono, 1 sbs, 2 tb), z: swap eyes, w: plane layout (0 I420, 1 NV12/P010)
    mode: vec4<u32>,
    // x: h_fov, y: v_fov (radians), z: zoom, w: ipd offset (radians)
    proj: vec4<f32>,
    // x: lens k1, y: lens k2, z: vertical align, w: rotation align (radians)
    lens: vec4<f32>,
    // Fisheye: xy the image circle's diameter in eye-image units
    circle: vec4<f32>,
    // Fisheye: each image circle's centre offset in eye-image units, the
    // left or top image's in xy, the other's in zw
    centres: vec4<f32>,
    // x: screen width, y: screen height, z: curvature arc (radians), w: has video
    screen: vec4<f32>,
    // YUV -> RGB: rgb = m * (yuv * scale - offset)
    yuv_r: vec4<f32>,
    yuv_g: vec4<f32>,
    yuv_b: vec4<f32>,
    // xyz: offsets, w: sample scale (10-bit-in-16 = 65535/1023)
    yuv_offset: vec4<f32>,
    // x brightness, y contrast, z saturation, w gamma
    picture: vec4<f32>,
    // x sharpen, y transfer (0 sdr, 1 pq, 2 hlg), z wide gamut, w background alpha
    extra: vec4<f32>,
    // x width, y height, z 1/width, w 1/height (luma texels)
    tex_size: vec4<f32>,
    // Premultiplied; alpha 0 lets passthrough show through.
    bg_color: vec4<f32>,
    // Chroma key: x, y the key colour's Cb, Cr; z similarity; w smoothness
    key: vec4<f32>,
    // x spill, y on, z alpha-packed mask scale (0 off)
    key2: vec4<f32>,
    // xyz: the viewer's offset (height, forward) in the room frame, metres,
    // for spherical video; w: radius of the sphere it is drawn on when offset
    viewer: vec4<f32>,
    // x: clockwise quarter turns that show the frame upright
    display: vec4<u32>,
    // HDR, as PQ codes: x the scene's peak, y the screen's white
    hdr: vec4<f32>,
    // A Q10C picture's overlapping bands (planes_tp10.wgsl), luma then
    // chroma: x first-row step, y rows, z the plane's rows, w count
    tp10: vec4<f32>,
    tp10_uv: vec4<f32>,
    // Stereo focus, per eye: xyz the eye's offset from its own camera in
    // the video's axes, metres; w the distance it is exact at, metres (0 off)
    focus: array<vec4<f32>, 2>,
};

// The video planes and `sample_yuv` / `luma_at` come from planes.wgsl, or
// planes_tp10.wgsl for the pipelines sampling Q10C in place (build.rs).
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    // One triangle covering the screen.
    let xy = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
    var o: VsOut;
    o.pos = vec4<f32>(xy, 0.0, 1.0);
    o.ndc = xy;
    return o;
}

const PI: f32 = 3.14159265358979;
const INVALID: vec2<f32> = vec2<f32>(-1.0, -1.0);

// Ray (forward = -Z, right = +X, up = +Y) to normalised video coordinates
// (0..1, origin top-left) for spherical projections.
fn equirect_uv(d: vec3<f32>, h_fov: f32, v_fov: f32) -> vec2<f32> {
    let lon = atan2(d.x, -d.z);
    let lat = asin(clamp(d.y, -1.0, 1.0));
    var u = 0.5 + lon / h_fov;
    let v = 0.5 - lat / v_fov;
    if (h_fov >= 2.0 * PI - 1e-3) {
        u = fract(u);
    } else if (u < 0.0 || u > 1.0) {
        return INVALID;
    }
    if (v < 0.0 || v > 1.0) { return INVALID; }
    return vec2<f32>(u, v);
}

fn fisheye_uv(d: vec3<f32>, fov: f32, k1: f32, k2: f32, size: vec2<f32>, centre: vec2<f32>) -> vec2<f32> {
    let theta = acos(clamp(-d.z, -1.0, 1.0));
    var r = theta / fov; // 0.5 at the edge of the image circle
    if (r > 0.5) { return INVALID; }
    let r2 = r * r * 4.0;
    r = r * (1.0 + k1 * r2 + k2 * r2 * r2);
    let len = length(d.xy);
    var dir = vec2<f32>(0.0, 0.0);
    if (len > 1e-6) { dir = d.xy / len; }
    let uv = centre + vec2<f32>(dir.x, -dir.y) * r * size;
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) { return INVALID; }
    return uv;
}

// YouTube equi-angular cubemap, 3x2 layout as FFmpeg's v360 "eac" output:
// top row left|front|right, bottom row down|back|up, bottom faces rotated.
fn eac_face_uv(a: f32, b: f32) -> vec2<f32> {
    // Equi-angular warp of cube-face coordinates in [-1, 1].
    return vec2<f32>(atan(a) * 4.0 / PI, atan(b) * 4.0 / PI) * 0.5 + 0.5;
}

fn eac_uv(d: vec3<f32>, h_fov: f32) -> vec2<f32> {
    let ax = abs(d.x); let ay = abs(d.y); let az = abs(d.z);
    var face: i32;
    var a: f32; var b: f32; // face-local: a right, b down
    if (az >= ax && az >= ay) {
        if (d.z < 0.0) { face = 1; a = d.x / az; b = -d.y / az; }      // front
        else { face = 4; a = -d.x / az; b = -d.y / az; }               // back
    } else if (ax >= ay) {
        if (d.x > 0.0) { face = 2; a = d.z / ax; b = -d.y / ax; }      // right
        else { face = 0; a = -d.z / ax; b = -d.y / ax; }               // left
    } else {
        if (d.y > 0.0) { face = 5; a = d.x / ay; b = d.z / ay; }       // up
        else { face = 3; a = d.x / ay; b = -d.z / ay; }                // down
    }
    var f = eac_face_uv(a, b);
    var col: f32; var row: f32;
    switch face {
        case 0: { col = 0.0; row = 0.0; }
        case 1: { col = 1.0; row = 0.0; }
        case 2: { col = 2.0; row = 0.0; }
        // Orientations verified against FFmpeg's v360 EAC output
        // (tests/projections.rs): down and up are anti-transposed, back is
        // rotated 90 degrees.
        case 3: { col = 0.0; row = 1.0; f = vec2<f32>(1.0 - f.y, 1.0 - f.x); }
        case 4: { col = 1.0; row = 1.0; f = vec2<f32>(1.0 - f.y, f.x); }
        default: { col = 2.0; row = 1.0; f = vec2<f32>(1.0 - f.y, 1.0 - f.x); }
    }
    let uv = vec2<f32>((col + f.x) / 3.0, (row + f.y) / 2.0);
    if (h_fov < 2.0 * PI - 1e-3) {
        // EAC 180: only the front half of the layout is stored (columns
        // mapped onto the full width).
        if (d.z > 0.0) { return INVALID; }
    }
    return uv;
}

// Where eye `eye` (0 left, 1 right) finds direction `d` of the picture's
// own axes in its picture, for the spherical projections (INVALID: not in
// it).
fn sphere_uv(d: vec3<f32>, eye: u32) -> vec2<f32> {
    var uv: vec2<f32>;
    let kind = params.mode.x;
    if (kind == 1u) {
        uv = equirect_uv(d, params.proj.x, params.proj.y);
    } else if (kind == 2u) {
        // The centres belong to the images, so they follow a swap.
        let slot = select(0u, eye ^ params.mode.z, params.mode.y != 0u);
        let centre = 0.5 + select(params.centres.xy, params.centres.zw, slot == 1u);
        uv = fisheye_uv(d, params.proj.x, params.lens.x, params.lens.y, params.circle.xy, centre);
    } else {
        uv = eac_uv(d, params.proj.x);
    }
    return uv;
}

fn to_eye_rect(uv: vec2<f32>, eye: u32) -> vec2<f32> {
    let slot = f32(eye ^ params.mode.z);
    switch params.mode.y {
        case 1u: { return vec2<f32>(uv.x * 0.5 + slot * 0.5, uv.y); }
        case 2u: { return vec2<f32>(uv.x, uv.y * 0.5 + slot * 0.5); }
        default: { return uv; }
    }
}

// Where a point of the upright picture is in the stored frame.
fn stored_uv(uv: vec2<f32>) -> vec2<f32> {
    switch params.display.x {
        case 1u: { return vec2<f32>(uv.y, 1.0 - uv.x); }
        case 2u: { return 1.0 - uv; }
        case 3u: { return vec2<f32>(1.0 - uv.y, uv.x); }
        default: { return uv; }
    }
}

// BT.2390's EETF on PQ code `e`, for a source peaking at PQ `src` and a
// display peaking at PQ `dst`: unchanged below the knee, a Hermite spline
// above it.
fn eetf(e: f32, src: f32, dst: f32) -> f32 {
    if (src <= dst) { return e; }
    let e1 = e / src;
    let max_lum = dst / src;
    let ks = max(1.5 * max_lum - 0.5, 0.0);
    if (e1 < ks) { return e; }
    let t = (min(e1, 1.0) - ks) / (1.0 - ks);
    let t2 = t * t;
    let t3 = t2 * t;
    let p = (2.0 * t3 - 3.0 * t2 + 1.0) * ks + (t3 - 2.0 * t2 + t) * (1.0 - ks)
        + (-2.0 * t3 + 3.0 * t2) * max_lum;
    return p * src;
}

// Nits, tone mapped through the brightest channel so hue is kept; 1.0 is
// the screen's white.
fn tone_map(c: vec3<f32>) -> vec3<f32> {
    let m = max3(c);
    if (m <= 0.0) { return vec3<f32>(0.0); }
    let mapped = pq_eotf1(eetf(pq_oetf1(m), params.hdr.x, params.hdr.y));
    return c * (mapped / (m * 203.0));
}

// The ACES reference gamut compression, sized for BT.2020 in BT.709 (see
// src/hdr.rs): each channel's distance below the brightest one is
// compressed past 0.8, bringing the most saturated BT.2020 colours to
// BT.709's edge and leaving most colours alone.
const GAMUT_THRESHOLD: f32 = 0.8;
const GAMUT_LIMIT: vec3<f32> = vec3<f32>(1.60, 1.09, 1.12);
const GAMUT_POWER: f32 = 1.2;

fn compress_distance(d: f32, limit: f32) -> f32 {
    if (d < GAMUT_THRESHOLD) { return d; }
    let s = (limit - GAMUT_THRESHOLD)
        / pow(pow((1.0 - GAMUT_THRESHOLD) / (limit - GAMUT_THRESHOLD), -GAMUT_POWER) - 1.0, 1.0 / GAMUT_POWER);
    let x = (d - GAMUT_THRESHOLD) / s;
    return GAMUT_THRESHOLD + s * x / pow(1.0 + pow(x, GAMUT_POWER), 1.0 / GAMUT_POWER);
}

fn gamut_compress(c: vec3<f32>) -> vec3<f32> {
    let ach = max3(c);
    if (ach <= 0.0) { return vec3<f32>(0.0); }
    let d = (ach - c) / ach;
    let cd = vec3<f32>(
        compress_distance(d.r, GAMUT_LIMIT.r),
        compress_distance(d.g, GAMUT_LIMIT.g),
        compress_distance(d.b, GAMUT_LIMIT.b));
    return ach - cd * ach;
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return pow(max(c, vec3<f32>(0.0)), vec3<f32>(2.2));
}

// The chroma key (as OBS does it) on a pixel's colour: its colour with the
// key colour's fringe desaturated out, and how opaque it stays.
fn chroma_key(colour: vec3<f32>) -> vec4<f32> {
    var rgb = colour;
    let c = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let cbcr = vec2<f32>(dot(c, vec3<f32>(-0.1146, -0.3854, 0.5)), dot(c, vec3<f32>(0.5, -0.4542, -0.0458)));
    let d = distance(cbcr, params.key.xy) - params.key.z;
    let alpha = pow(clamp(d / max(params.key.w, 1e-4), 0.0, 1.0), 1.5);
    let spill = pow(clamp(d / max(params.key2.x, 1e-4), 0.0, 1.0), 1.5);
    rgb = mix(vec3<f32>(dot(c, vec3<f32>(0.2126, 0.7152, 0.0722))), rgb, spill);
    return vec4<f32>(rgb, alpha);
}

// How opaque an alpha-packed passthrough video (`_alpha` names) is at
// `uv` of eye `eye`'s picture: each eye's mask, scaled down, sits in the
// red channel outside the fisheye circles, centred on the frame's top
// middle for the left half and on its corner for the right half, wrapping
// round the frame's edges.
fn packed_mask(uv: vec2<f32>, eye: u32) -> f32 {
    let slot = f32(eye ^ params.mode.z);
    let k = params.key2.z;
    let m = fract(vec2<f32>((1.0 - slot) * 0.5 + (uv.x - 0.5) * k * 0.5, (uv.y - 0.5) * k));
    return clamp((sample_yuv(stored_uv(m)).r - 0.08) / 0.84, 0.0, 1.0);
}

// A pixel's linear light in BT.709, 1.0 the screen's white, from its
// R'G'B' in the video's transfer and gamut: HDR to nits, into the gamut,
// then the tone curve, which also keeps every channel within the screen's
// white.
fn linear_light(rgb: vec3<f32>) -> vec3<f32> {
    var lin: vec3<f32>;
    let transfer = u32(params.extra.y + 0.5);
    let wide = params.extra.z > 0.5;
    if (transfer == 1u || transfer == 2u) {
        var nits = hdr_nits(rgb, transfer, wide);
        if (wide) { nits = gamut_compress(nits); }
        lin = tone_map(nits);
    } else {
        lin = srgb_to_linear(rgb);
        if (wide) {
            lin = gamut_compress(bt2020_to_709(lin));
            lin = lin / max(max3(lin), 1.0);
        }
    }
    return lin;
}
