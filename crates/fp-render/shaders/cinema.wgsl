
// Cinema (params.env.x > 0.5): a hall around a flat screen, ray traced in
// the screen's space (its centre at the origin facing +Z, the viewer at
// (0, 0, distance), metres) and lit by the picture. Row k of seats counts
// from the viewer's (0) towards the screen: rows ahead step down so none
// hides the screen, rows behind step up. Compiled after scene.wgsl; the
// sizes match cinema.rs.

const ROW_PITCH: f32 = 1.1;
const BACK_RISE: f32 = 0.35;
const SEAT_PITCH: f32 = 0.6;
// The highest point of a seat above its row's floor.
const SEAT_TOP: f32 = 1.06;
// A row's strip of floor reaches this far behind its seats.
const ROW_BACK: f32 = 0.3;
// From the viewer's eyes back to their seat.
const EYE_TO_SEAT: f32 = 0.3;
const FRONT_WALL: f32 = -0.2;
// Seat backs lean back this far, radians.
const RECLINE: f32 = 0.21;
// How much brighter than physical the picture lights the hall, and how
// much of its light the hall bounces back.
const SCREEN_GAIN: f32 = 7.0;
const BOUNCE: f32 = 0.6;
const WARM: vec3<f32> = vec3<f32>(1.0, 0.62, 0.32);

const M_WALL: u32 = 0u;
const M_FABRIC: u32 = 1u;
const M_FLOOR: u32 = 2u;
const M_STEP: u32 = 3u;
const M_CEILING: u32 = 4u;
const M_FRONT: u32 = 5u;
const M_BACK: u32 = 6u;
const M_SHELL: u32 = 7u;
const M_STANDARD: u32 = 8u;
const M_STAGE: u32 = 9u;
const M_SPEAKER: u32 = 10u;

struct Hit {
    t: f32,
    n: vec3<f32>,
    mat: u32,
    // A step's top
    top: f32,
};

fn miss() -> Hit {
    return Hit(1e9, vec3<f32>(0.0), M_WALL, 0.0);
}

fn row_z(k: i32) -> f32 {
    return params.seats.y - f32(k) * ROW_PITCH;
}

// The row whose strip of floor holds z.
fn row_at(z: f32) -> i32 {
    return i32(floor((params.seats.y + ROW_BACK - z) / ROW_PITCH));
}

fn row_floor(k: i32) -> f32 {
    if (k > i32(params.env.z)) { return params.room.w; }
    if (k <= 0) { return params.seats.x - f32(k) * BACK_RISE; }
    // Under the line from the viewer's eyes to the bottom of the screen.
    let d = params.seats.y - EYE_TO_SEAT;
    let z = row_z(k) + 0.15;
    return min(params.seats.x, params.seats.w * (d - z) / d - SEAT_TOP - 0.08);
}

fn is_seated(k: i32) -> bool {
    return k >= i32(params.env.w) && k <= i32(params.env.z);
}

// Where a ray from outside enters a box: xyz the face's normal, w the
// distance (negative when it misses).
fn box_hit(o: vec3<f32>, inv: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> vec4<f32> {
    let a = (lo - o) * inv;
    let b = (hi - o) * inv;
    let near = min(a, b);
    let far = max(a, b);
    let t = max(max(near.x, near.y), near.z);
    if (t > min(min(far.x, far.y), far.z) || t <= 0.0) { return vec4<f32>(-1.0); }
    if (t == near.x) { return vec4<f32>(-sign(inv.x), 0.0, 0.0, t); }
    if (t == near.y) { return vec4<f32>(0.0, -sign(inv.y), 0.0, t); }
    return vec4<f32>(0.0, 0.0, -sign(inv.z), t);
}

// Where a ray from outside meets a rod along x through (y, z) = c, of
// radius r and reaching x = ±half.
fn rod_hit(o: vec3<f32>, d: vec3<f32>, c: vec2<f32>, r: f32, half: f32) -> vec4<f32> {
    let oc = o.yz - c;
    let a = max(dot(d.yz, d.yz), 1e-8);
    let b = dot(oc, d.yz);
    let disc = b * b - a * (dot(oc, oc) - r * r);
    if (disc < 0.0) { return vec4<f32>(-1.0); }
    let t = (-b - sqrt(disc)) / a;
    let p = o + d * t;
    if (t <= 0.0 || abs(p.x) > half) { return vec4<f32>(-1.0); }
    return vec4<f32>(0.0, (p.yz - c) / r, t);
}

fn nearer(hit: Hit, h: vec4<f32>, mat: u32) -> Hit {
    if (h.w > 0.0 && h.w < hit.t) { return Hit(h.w, h.xyz, mat, 0.0); }
    return hit;
}

// One seat, the ray in its space: x across from the seat's middle, y up
// from the floor, z back from the foot of its backrest.
fn seat_hit(o: vec3<f32>, d: vec3<f32>) -> Hit {
    let inv = 1.0 / d;
    var hit = miss();
    // The standards either side, with the armrests on top.
    for (var s = -1.0; s < 2.0; s += 2.0) {
        let x = s * 0.5 * SEAT_PITCH;
        hit = nearer(hit, box_hit(o, inv, vec3<f32>(x - 0.03, 0.0, -0.5), vec3<f32>(x + 0.03, 0.64, 0.08)), M_STANDARD);
    }
    // The seat, its front edge rolled.
    hit = nearer(hit, box_hit(o, inv, vec3<f32>(-0.26, 0.36, -0.47), vec3<f32>(0.26, 0.47, -0.04)), M_FABRIC);
    hit = nearer(hit, rod_hit(o, d, vec2<f32>(0.415, -0.47), 0.055, 0.26), M_FABRIC);
    // The backrest, leaning back about its foot: v up it, u through it.
    let up = vec2<f32>(cos(RECLINE), sin(RECLINE));
    let through = vec2<f32>(-up.y, up.x);
    let q = o.yz - vec2<f32>(0.42, 0.0);
    let lo = vec3<f32>(o.x, dot(q, up), dot(q, through));
    let ld = vec3<f32>(d.x, dot(d.yz, up), dot(d.yz, through));
    let safe = select(ld, vec3<f32>(1e-6), abs(ld) < vec3<f32>(1e-6));
    var back = box_hit(lo, 1.0 / safe, vec3<f32>(-0.27, 0.0, -0.05), vec3<f32>(0.27, 0.56, 0.06));
    var mat = select(M_SHELL, M_FABRIC, back.z < -0.5);
    let top = rod_hit(lo, safe, vec2<f32>(0.56, 0.005), 0.058, 0.27);
    if (top.w > 0.0 && (back.w <= 0.0 || top.w < back.w)) {
        back = top;
        mat = M_FABRIC;
    }
    if (back.w > 0.0 && back.w < hit.t) {
        let n = vec3<f32>(back.x, back.y * up.x + back.z * through.x, back.y * up.y + back.z * through.y);
        hit = Hit(back.w, n, mat, 0.0);
    }
    return hit;
}

// The nearest seat of row k between t0 and t1, seat by seat along the
// ray's way across the row.
fn row_hit(o: vec3<f32>, d: vec3<f32>, k: i32, t0: f32, t1: f32) -> Hit {
    let n = params.seats.z / SEAT_PITCH - 0.5;
    let reach = params.seats.z + 0.03;
    var ta = t0;
    var tb = t1;
    if (abs(d.x) > 1e-6) {
        let a = (-reach - o.x) / d.x;
        let b = (reach - o.x) / d.x;
        ta = max(ta, min(a, b));
        tb = min(tb, max(a, b));
    } else if (abs(o.x) > reach) {
        return miss();
    }
    var best = miss();
    if (ta >= tb) { return best; }
    best.t = tb;
    let base = vec3<f32>(0.0, row_floor(k), row_z(k));
    let step = select(-1.0, 1.0, d.x > 0.0);
    var i = clamp(round((o.x + d.x * ta) / SEAT_PITCH), -n, n);
    for (var c = 0; c < 10; c++) {
        let h = seat_hit(o - base - vec3<f32>(i * SEAT_PITCH, 0.0, 0.0), d);
        if (h.t < best.t) { best = h; }
        let leave = ((i + 0.5 * step) * SEAT_PITCH - o.x) / d.x;
        i += step;
        if (best.t <= leave || leave >= tb || abs(i) > n) { break; }
    }
    if (best.t >= tb) { return miss(); }
    return best;
}

// The nearest seat, floor or step closer than `far`, row by row from the
// viewer's: each row stands on its own strip of floor ROW_PITCH deep.
fn trace_rows(o: vec3<f32>, d: vec3<f32>, inv: vec3<f32>, far: f32) -> Hit {
    var hit = Hit(far, vec3<f32>(0.0), M_WALL, 0.0);
    let front = i32(params.env.z);
    let back = i32(params.env.w);
    let stage = front + 1;
    let step = select(-1, 1, d.z < 0.0);
    var k = clamp(row_at(o.z), back, stage);
    for (var i = 0; i < 48; i++) {
        let edge = row_z(k) + ROW_BACK;
        let ta = (select(edge - ROW_PITCH, -1e9, k == stage) - o.z) * inv.z;
        let tb = (select(edge, 1e9, k == back) - o.z) * inv.z;
        let t0 = max(min(ta, tb), 0.0);
        let t1 = min(max(ta, tb), hit.t);
        if (t0 >= t1) { break; }
        let y = row_floor(k);
        if (k < stage && min(o.y + d.y * t0, o.y + d.y * t1) < y + SEAT_TOP) {
            let s = row_hit(o, d, k, t0, t1);
            if (s.t < hit.t) { hit = s; }
        }
        let tf = (y - o.y) * inv.y;
        if (d.y < 0.0 && tf >= t0 && tf < hit.t) { hit = Hit(tf, vec3<f32>(0.0, 1.0, 0.0), M_FLOOR, 0.0); }
        if (hit.t <= t1) { break; }
        let next = k + step;
        if (next < back || next > stage) { break; }
        // Into a higher row: the step up.
        let yn = row_floor(next);
        if (o.y + d.y * t1 < yn) {
            hit = Hit(t1, vec3<f32>(0.0, 0.0, -sign(d.z)), M_STEP, yn);
            break;
        }
        k = next;
    }
    return hit;
}

fn speaker_y() -> f32 {
    return min(params.seats.x + 2.2, params.room.y - 0.45);
}

// Walls, ceiling, the stage under the screen and the speakers on the walls.
fn hall_hit(o: vec3<f32>, d: vec3<f32>, inv: vec3<f32>) -> Hit {
    let side = select(-1.0, 1.0, d.x > 0.0);
    var hit = Hit((side * params.room.x - o.x) * inv.x, vec3<f32>(-side, 0.0, 0.0), M_WALL, 0.0);
    let tz = (select(FRONT_WALL, params.room.z, d.z > 0.0) - o.z) * inv.z;
    if (tz < hit.t) {
        hit = Hit(tz, vec3<f32>(0.0, 0.0, -sign(d.z)), select(M_FRONT, M_BACK, d.z > 0.0), 0.0);
    }
    let ty = (params.room.y - o.y) * inv.y;
    if (d.y > 0.0 && ty < hit.t) { hit = Hit(ty, vec3<f32>(0.0, -1.0, 0.0), M_CEILING, 0.0); }
    let w = params.room.x;
    hit = nearer(hit, box_hit(o, inv, vec3<f32>(-w, params.room.w - 1.0, FRONT_WALL), vec3<f32>(w, params.stage.x, params.stage.y)), M_STAGE);
    // The speakers near where the ray reaches the wall they hang on.
    let tf = (side * (w - 0.22) - o.x) * inv.x;
    if (tf > 0.0 && tf < hit.t) {
        let last = floor((params.room.z - 3.0) / 3.0);
        let j = clamp(round((o.z + d.z * tf - 2.0) / 3.0), 0.0, last);
        let y = speaker_y();
        for (var n = 0.0; n < 2.0; n += 1.0) {
            let z = clamp(j + n * sign(d.z), 0.0, last) * 3.0 + 2.0;
            let lo = vec3<f32>(min(side * w, side * (w - 0.22)), y - 0.28, z - 0.2);
            let hi = vec3<f32>(max(side * w, side * (w - 0.22)), y + 0.28, z + 0.2);
            hit = nearer(hit, box_hit(o, inv, lo, hi), M_SPEAKER);
        }
    }
    return hit;
}

// Light from the picture falling on `p` facing `n`: each region as a small
// disc, so that it doesn't blow out what is close to it.
fn screen_light(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    var e = vec3<f32>(0.0);
    for (var i = 0; i < 12; i++) {
        let c = params.light_pos[i];
        let v = c.xyz - p;
        let r2 = dot(v, v);
        let l = v * inverseSqrt(r2);
        let facing = max(dot(n, l), 0.0) * max(-dot(params.light_dir[i].xyz, l), 0.0);
        e += params.light_rgb[i].rgb * (c.w * facing / (r2 + c.w / PI));
    }
    return e * SCREEN_GAIN;
}

// How much of the picture's light gets past the seat backs of the row in
// front of `p`: soft, as from the top and bottom halves of the screen.
fn seat_shadow(p: vec3<f32>) -> f32 {
    let k = row_at(p.z) + 1;
    if (!is_seated(k)) { return 1.0; }
    let z = row_z(k) + 0.1;
    if (z >= p.z) { return 1.0; }
    let top = row_floor(k) + SEAT_TOP;
    var lit = 0.0;
    for (var i = 0; i < 2; i++) {
        let ly = select(-0.25, 0.25, i == 1) * params.screen.y;
        let y = p.y + (ly - p.y) * (z - p.z) / (-p.z);
        lit += smoothstep(top - 0.1, top + 0.2, y);
    }
    return 0.5 * lit;
}

// Seams `width` wide every `period`, darker than between them; they fade
// out where a pixel covers more than a seam, rather than shimmer.
fn seams(x: f32, period: f32, width: f32, pixel: f32) -> f32 {
    let dist = abs(fract(x / period + 0.5) - 0.5) * period;
    let line = 1.0 - smoothstep(0.0, width, dist);
    return 1.0 - 0.6 * line * (1.0 - smoothstep(width, 4.0 * width, pixel));
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

// A cinema carpet: dim gold diamonds on burgundy.
fn carpet(xz: vec2<f32>, pixel: f32) -> vec3<f32> {
    let base = vec3<f32>(0.05, 0.01, 0.016);
    let f = abs(fract(xz / 0.5) - 0.5);
    let ring = abs(f.x + f.y - 0.3);
    let motif = (1.0 - smoothstep(0.015, 0.035, ring)) + (1.0 - smoothstep(0.03, 0.05, length(f)));
    let fade = 1.0 - smoothstep(0.02, 0.06, pixel);
    return mix(base, vec3<f32>(0.075, 0.05, 0.02), clamp(motif, 0.0, 1.0) * fade * 0.8);
}

// Velvet catches light at grazing angles.
fn sheen(n: vec3<f32>, d: vec3<f32>) -> f32 {
    return 1.0 + 1.5 * pow(1.0 - abs(dot(n, d)), 3.0);
}

fn cinema(origin: vec3<f32>, world_dir: vec3<f32>, frag: vec2<f32>) -> vec4<f32> {
    let o = (params.screen_inv * vec4<f32>(origin, 1.0)).xyz;
    let d0 = normalize((params.screen_inv * vec4<f32>(world_dir, 0.0)).xyz);
    let d = select(d0, vec3<f32>(1e-6), abs(d0) < vec3<f32>(1e-6));
    let inv = 1.0 / d;
    var hit = hall_hit(o, d, inv);
    let rows = trace_rows(o, d, inv, hit.t);
    if (rows.t < hit.t) { hit = rows; }
    let p = o + d * hit.t;
    var n = hit.n;
    // Roughly what one pixel covers there.
    let pixel = hit.t * 0.0015 / max(abs(dot(n, d)), 0.2);
    let house = params.env.y;
    let floor_y = params.seats.x;
    let w = params.room.x;
    var albedo = vec3<f32>(0.03);
    var glow = vec3<f32>(0.0);
    var shade = 1.0;
    var shadow = 1.0;
    var shine = 1.0;
    let k = row_at(p.z);
    let seat_x = p.x - round(p.x / SEAT_PITCH) * SEAT_PITCH;
    switch hit.mat {
        case M_FABRIC: {
            // Red velvet, in three panels up the backrest.
            albedo = vec3<f32>(0.17, 0.02, 0.028) * seams(seat_x + 0.09, 0.18, 0.006, pixel);
            shadow = seat_shadow(p);
            shine = sheen(n, d);
        }
        case M_SHELL: {
            albedo = vec3<f32>(0.018, 0.018, 0.02);
            shadow = seat_shadow(p);
        }
        case M_STANDARD: {
            albedo = vec3<f32>(0.012);
            shadow = seat_shadow(p);
            // Aisle lights low on the end of each row.
            if (abs(p.x) > params.seats.z - 0.05 && n.x * p.x > 0.5 && p.y - row_floor(k) < 0.12) {
                glow = WARM * 0.5;
            }
        }
        case M_FLOOR: {
            albedo = carpet(p.xz, pixel);
            shadow = seat_shadow(p);
            if (is_seated(k)) {
                // Dark under the seats.
                if (abs(p.x) < params.seats.z) {
                    let z = p.z - row_z(k);
                    let under = max(max(-0.5 - z, z - 0.08), 0.0);
                    shade *= 0.3 + 0.7 * smoothstep(0.0, 0.3, under);
                }
                // A light along the edge of each step in the aisles.
                let edge = row_z(k) + ROW_BACK - ROW_PITCH;
                if (abs(p.x) > params.seats.z + 0.2 && abs(p.x) < w - 0.2 && p.z - edge < 0.04) {
                    glow = WARM * 0.25;
                }
            }
        }
        case M_STEP: {
            albedo = carpet(p.xy, pixel) * 0.7;
            if (abs(abs(p.x) - params.seats.z - 0.35) < 0.07 && p.y > hit.top - 0.06) {
                glow = WARM * 0.35;
            }
        }
        case M_CEILING: {
            // Acoustic tiles, and downlights that come up with the house lights.
            albedo = vec3<f32>(0.022) * seams(p.x, 0.6, 0.01, pixel) * seams(p.z, 0.6, 0.01, pixel);
            let cell = abs(fract(p.xz / 2.4) - 0.5) * 2.4;
            if (max(cell.x, cell.y) < 0.11) {
                glow = mix(vec3<f32>(0.004), WARM * 1.4, house);
            }
        }
        case M_FRONT: {
            // Black masking round the screen, red curtains beside it.
            albedo = vec3<f32>(0.005);
            if (abs(p.x) > 0.5 * params.screen.x + 0.35) {
                let fold = sin(p.x * 2.0 * PI / 0.3) * (1.0 - smoothstep(0.02, 0.08, pixel));
                n = normalize(vec3<f32>(0.7 * fold, 0.0, 1.0));
                albedo = vec3<f32>(0.2, 0.02, 0.03) * (0.7 + 0.3 * fold);
                shine = sheen(n, d);
            }
        }
        case M_STAGE: {
            albedo = select(vec3<f32>(0.006), vec3<f32>(0.03, 0.02, 0.015), n.y > 0.5);
            if (n.z > 0.5 && p.y > params.stage.x - 0.03) { glow = WARM * 0.3; }
        }
        case M_SPEAKER: {
            albedo = vec3<f32>(0.01) * seams(p.y, 0.02, 0.004, pixel);
        }
        case M_BACK: {
            // Panels, and the projection booth's windows.
            albedo = vec3<f32>(0.022, 0.026, 0.04) * seams(p.x, 1.2, 0.01, pixel);
            let b = vec2<f32>(abs(abs(p.x) - 0.8), p.y - params.room.y + 0.6);
            if (b.x < 0.28 && abs(b.y) < 0.18) {
                albedo = vec3<f32>(0.004);
                if (b.x < 0.24 && abs(b.y) < 0.14) {
                    let lamp = vec3<f32>(0.07, 0.065, 0.06) * (1.0 - 1.5 * b.y);
                    glow = lamp + params.glow.rgb * 0.1;
                }
            }
        }
        default: {
            // Fabric panels over a wooden dado, lit up from wall washers.
            albedo = vec3<f32>(0.022, 0.026, 0.04) * seams(p.z, 1.2, 0.01, pixel);
            let ground = row_floor(clamp(k, i32(params.env.w), i32(params.env.z) + 1));
            if (p.y < ground + 0.9) { albedo = vec3<f32>(0.045, 0.026, 0.016) * seams(p.y, 0.15, 0.004, pixel); }
            let wz = clamp(round((p.z - 3.5) / 3.0), 0.0, floor((params.room.z - 4.5) / 3.0)) * 3.0 + 3.5;
            let s = vec2<f32>(p.z - wz, p.y - floor_y - 1.5);
            let wash = exp(-s.x * s.x * 12.0) * select(exp(-s.y * 1.6), exp(s.y * 18.0), s.y < 0.0);
            glow = WARM * mix(0.02, 0.1, house) * wash;
            if (abs(s.x) < 0.1 && s.y < 0.0 && s.y > -0.07) { glow += WARM * mix(0.35, 0.9, house); }
            // An exit sign near the screen.
            let e = vec2<f32>(p.z - 0.7, p.y - params.room.y + 0.3);
            if (abs(e.x) < 0.17 && abs(e.y) < 0.07) { glow += vec3<f32>(0.05, 0.6, 0.2); }
            glow += vec3<f32>(0.0, 0.03, 0.01) * exp(-dot(e, e) * 8.0);
            shade *= mix(0.5, 1.0, smoothstep(0.0, 0.3, p.y - ground));
        }
    }
    // Darker into the corners.
    if (hit.mat == M_WALL || hit.mat == M_FRONT || hit.mat == M_BACK) {
        shade *= mix(0.65, 1.0, smoothstep(0.0, 0.4, params.room.y - p.y));
    } else if (hit.mat != M_SPEAKER) {
        shade *= mix(0.6, 1.0, smoothstep(0.0, 0.5, w - abs(p.x)));
    }
    let e = screen_light(p, n) * shadow + params.glow.rgb * BOUNCE + vec3<f32>(mix(0.004, 0.08, house));
    var col = albedo / PI * e * shade * shine + glow;
    // Dithered by about one 8-bit sRGB step, as dark gradients band:
    // a step in linear light is about 2.2 * sqrt of it.
    let noise = hash(frag) - 0.5;
    col = max(col + noise * (2.2 / 255.0) * sqrt(col + 1e-5), vec3<f32>(0.0));
    return vec4<f32>(col, 1.0);
}
