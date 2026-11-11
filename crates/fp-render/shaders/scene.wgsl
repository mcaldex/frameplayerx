// Per-pixel VR video projection.
//
// Every fragment builds its view ray from the eye's inverse view-projection,
// maps the ray to a video coordinate for the active projection, picks the
// eye's half of a stereo frame, converts YUV to RGB, applies HDR tone mapping
// and picture adjustments, and writes linear colour to an sRGB target. What
// it shares with the measurement pass is in picture.wgsl (build.rs).

struct PushConstants {
    eye: u32,
};

var<immediate> pc: PushConstants;

fn rot_y(a: f32) -> mat3x3<f32> {
    let c = cos(a); let s = sin(a);
    return mat3x3<f32>(vec3<f32>(c, 0.0, -s), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(s, 0.0, c));
}
fn rot_x(a: f32) -> mat3x3<f32> {
    let c = cos(a); let s = sin(a);
    return mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, c, s), vec3<f32>(0.0, -s, c));
}
fn rot_z(a: f32) -> mat3x3<f32> {
    let c = cos(a); let s = sin(a);
    return mat3x3<f32>(vec3<f32>(c, s, 0.0), vec3<f32>(-s, c, 0.0), vec3<f32>(0.0, 0.0, 1.0));
}

// Angle-preserving zoom: scales the angle from the forward axis.
fn zoom_dir(d: vec3<f32>, zoom: f32) -> vec3<f32> {
    if (abs(zoom - 1.0) < 1e-4) { return d; }
    let theta = acos(clamp(-d.z, -1.0, 1.0));
    let r = length(d.xy);
    if (r < 1e-6) { return d; }
    let t2 = min(theta / zoom, PI);
    let phi = d.xy / r;
    return vec3<f32>(phi * sin(t2), -cos(t2));
}

// The direction from the sphere's centre to where a ray from `p` (inside a
// sphere of radius `r`) along `d` meets it: seen from off the centre, near
// parts of the picture grow and slide against far ones. Both eyes use the
// same `p`, so the video's own stereo depth is kept.
fn off_centre(d: vec3<f32>, p: vec3<f32>, r: f32) -> vec3<f32> {
    if (dot(p, p) < 1e-8) { return d; }
    let q = p * min(1.0, 0.9 * r / length(p));
    let b = dot(q, d);
    let t = -b + sqrt(b * b - dot(q, q) + r * r);
    return normalize(q + t * d);
}

fn flat_uv(origin: vec3<f32>, d: vec3<f32>) -> vec2<f32> {
    let o = (params.screen_inv * vec4<f32>(origin, 1.0)).xyz;
    let dir = normalize((params.screen_inv * vec4<f32>(d, 0.0)).xyz);
    let w = params.screen.x; let h = params.screen.y; let arc = params.screen.z;
    var p: vec3<f32>;
    var u: f32;
    if (arc < 1e-3) {
        if (dir.z >= -1e-6) { return INVALID; }
        let t = -o.z / dir.z;
        if (t <= 0.0) { return INVALID; }
        p = o + t * dir;
        u = p.x / w + 0.5;
    } else {
        // Concave cylinder of radius R centred at (0, 0, R), screen centre at origin.
        let r = w / arc;
        let oc = vec2<f32>(o.x, o.z - r);
        let dd = vec2<f32>(dir.x, dir.z);
        let qa = dot(dd, dd);
        let qb = 2.0 * dot(oc, dd);
        let qc = dot(oc, oc) - r * r;
        let disc = qb * qb - 4.0 * qa * qc;
        if (disc < 0.0) { return INVALID; }
        let t = (-qb + sqrt(disc)) / (2.0 * qa);
        if (t <= 0.0) { return INVALID; }
        p = o + t * dir;
        let phi = atan2(p.x, r - p.z);
        u = 0.5 + phi * r / w;
    }
    let v = 0.5 - p.y / h;
    if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0) { return INVALID; }
    return vec2<f32>(u, v);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let eye = pc.eye;
    let near = params.inv_view_proj[eye] * vec4<f32>(in.ndc, 0.0, 1.0);
    let far = params.inv_view_proj[eye] * vec4<f32>(in.ndc, 1.0, 1.0);
    let origin = near.xyz / near.w;
    let world_dir = normalize(far.xyz / far.w - origin);

    if (params.screen.w < 0.5) {
        return params.bg_color;
    }

    var uv: vec2<f32>;
    let kind = params.mode.x;
    if (kind == 0u) {
        uv = flat_uv(origin, world_dir);
    } else {
        // In the room frame the video was centred in, so per-eye stereo
        // corrections stay the eyes' own; then the picture's rotation.
        let s = select(-0.5, 0.5, eye == 1u);
        var d = (params.anchor_inv * vec4<f32>(world_dir, 0.0)).xyz;
        d = rot_y(params.proj.w * s) * d;
        d = rot_x(params.lens.z * s) * d;
        d = rot_z(params.lens.w * s) * d;
        d = normalize((params.correction * vec4<f32>(d, 0.0)).xyz);
        d = off_centre(d, (params.correction * vec4<f32>(params.viewer.xyz, 0.0)).xyz,
                       params.viewer.w);
        // Stereo focus: each eye's picture as if painted on a sphere round
        // its own camera, seen from where the eye is. After the height and
        // forward, which move the whole scene, not by the focus distance.
        let focus = params.focus[eye];
        if (focus.w > 0.0) {
            d = off_centre(d, focus.xyz, focus.w);
        }
        d = zoom_dir(d, params.proj.z);
        uv = sphere_uv(d, eye);
    }
    if (uv.x < 0.0) {
        return params.bg_color;
    }
    let tuv = stored_uv(to_eye_rect(uv, eye));
    var rgb = sample_yuv(tuv);

    // Chroma key: pixels whose chroma is near the key colour's turn
    // see-through, with a soft edge.
    var alpha = 1.0;
    if (params.key2.y > 0.5) {
        let keyed = chroma_key(rgb);
        rgb = keyed.rgb;
        alpha = keyed.a;
    }

    // Alpha-packed passthrough video: each eye's own mask.
    if (params.key2.z > 0.0) {
        alpha *= packed_mask(uv, eye);
    }

    // Sharpen: unsharp mask on luma.
    if (params.extra.x > 0.0) {
        let dx = vec2<f32>(params.tex_size.z, 0.0);
        let dy = vec2<f32>(0.0, params.tex_size.w);
        let yc = luma_at(tuv);
        let blur = (luma_at(tuv + dx) + luma_at(tuv - dx) + luma_at(tuv + dy) + luma_at(tuv - dy)) * 0.25;
        rgb += vec3<f32>((yc - blur) * params.extra.x * 2.0);
    }

    let lin = linear_light(rgb);

    // Picture adjustments, applied in a perceptual (gamma) space.
    var g = pow(max(lin, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
    g = (g - 0.5) * params.picture.y + 0.5 + params.picture.x;
    let luma = dot(g, vec3<f32>(0.2126, 0.7152, 0.0722));
    g = mix(vec3<f32>(luma), g, params.picture.z);
    g = pow(max(g, vec3<f32>(0.0)), vec3<f32>(1.0 / max(params.picture.w, 0.05)));
    let col = srgb_to_linear(clamp(g, vec3<f32>(0.0), vec3<f32>(1.0)));
    // Over the (premultiplied) background: keyed-out pixels show it.
    return vec4<f32>(col * alpha + params.bg_color.rgb * (1.0 - alpha), alpha + params.bg_color.a * (1.0 - alpha));
}
