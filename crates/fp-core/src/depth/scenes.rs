//! Synthetic 3D videos for tests: textured surfaces ray-traced into two
//! cameras' pictures, packed into a frame as a [`PictureGeometry`] says.
//! Only for tests: fp-core's own, and other crates' through the
//! `test-support` feature their dev-dependencies turn on.

use super::picture::{PictureGeometry, Pixels};
use crate::format::{Projection, StereoLayout};
use std::cell::RefCell;

/// What a surface looks like: luminance 0..1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Texture {
    /// Lattice value noise, finest `feature` across (metres on a surface,
    /// degrees on the sky), with three octaves coarser, `contrast` about
    /// its standard deviation's fifth.
    Noise {
        feature: f32,
        contrast: f32,
        seed: u32,
    },
    /// The same, repeating every `period` along the surface's first axis.
    Repeating {
        period: f32,
        feature: f32,
        contrast: f32,
        seed: u32,
    },
    /// One luminance.
    Plain(f32),
}

/// Where a surface is, in the left camera's axes (the video's, with the
/// origin between the cameras).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    /// The plane through `centre` facing `normal`.
    Plane { centre: [f32; 3], normal: [f32; 3] },
    /// The disc of `radius` metres in that plane round `centre`.
    Disc {
        centre: [f32; 3],
        normal: [f32; 3],
        radius: f32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Surface {
    pub shape: Shape,
    pub texture: Texture,
    /// The room shows through it (a chroma key or mask).
    pub see_through: bool,
}

impl Surface {
    pub fn new(shape: Shape, texture: Texture) -> Surface {
        Surface {
            shape,
            texture,
            see_through: false,
        }
    }

    /// A plane `distance` metres straight ahead, facing the cameras.
    pub fn wall(distance: f32, texture: Texture) -> Surface {
        Surface::new(
            Shape::Plane {
                centre: [0.0, 0.0, -distance],
                normal: [0.0, 0.0, 1.0],
            },
            texture,
        )
    }
}

/// Surfaces in front of a sky at infinity.
#[derive(Clone, Debug, PartialEq)]
pub struct Scene {
    pub surfaces: Vec<Surface>,
    pub sky: Texture,
}

impl Scene {
    pub fn new(surfaces: Vec<Surface>) -> Scene {
        Scene {
            surfaces,
            sky: Texture::Noise {
                feature: 1.5,
                contrast: 1.0,
                seed: 99,
            },
        }
    }
}

/// The two cameras.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rig {
    /// Metres between the lenses.
    pub baseline: f32,
    /// The right camera turned against the left about x (pitch, up), y
    /// (yaw, left) and z (roll, anticlockwise as it looks), degrees.
    pub right_turn: [f32; 3],
}

impl Rig {
    pub fn new(baseline: f32) -> Rig {
        Rig {
            baseline,
            right_turn: [0.0; 3],
        }
    }
}

/// A 32-bit hash of a lattice point.
fn hash(x: i32, y: i32, seed: u32) -> f32 {
    let mut h = (x as u32)
        .wrapping_mul(0x8da6_b343)
        .wrapping_add((y as u32).wrapping_mul(0xd816_3841))
        .wrapping_add(seed.wrapping_mul(0xcb1a_b31f));
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297a_2d39);
    h ^= h >> 15;
    (h >> 8) as f32 / (1u32 << 24) as f32
}

/// Smoothly interpolated value noise at `(x, y)` lattice units, 0..1;
/// `wrap` repeats it every that many lattice cells along x.
fn value_noise(x: f32, y: f32, seed: u32, wrap: Option<i32>) -> f32 {
    let (xf, yf) = (x.floor(), y.floor());
    let (fx, fy) = (x - xf, y - yf);
    let s = |t: f32| t * t * (3.0 - 2.0 * t);
    let (sx, sy) = (s(fx), s(fy));
    let (xi, yi) = (xf as i32, yf as i32);
    let at = |dx: i32, dy: i32| {
        let mut px = xi + dx;
        if let Some(w) = wrap {
            px = px.rem_euclid(w);
        }
        hash(px, yi + dy, seed)
    };
    let top = at(0, 0) + (at(1, 0) - at(0, 0)) * sx;
    let bottom = at(0, 1) + (at(1, 1) - at(0, 1)) * sx;
    top + (bottom - top) * sy
}

impl Texture {
    /// Luminance at `(x, y)` on the surface (or degrees on the sky).
    pub fn at(&self, x: f32, y: f32) -> f32 {
        let (feature, contrast, seed, period) = match *self {
            Texture::Plain(l) => return l,
            Texture::Noise {
                feature,
                contrast,
                seed,
            } => (feature, contrast, seed, None),
            Texture::Repeating {
                period,
                feature,
                contrast,
                seed,
            } => (feature, contrast, seed, Some(period)),
        };
        let mut sum = 0.0;
        for octave in 0..4 {
            let scale = feature * (1 << octave) as f32;
            let wrap = period.map(|p| ((p / scale).round() as i32).max(1));
            // A period that isn't a whole number of cells is stretched to one.
            let sx = match (period, wrap) {
                (Some(p), Some(w)) => p / w as f32,
                _ => scale,
            };
            sum += value_noise(x / sx, y / scale, seed.wrapping_add(octave), wrap);
        }
        (0.5 + contrast * (sum / 4.0 - 0.5) * 2.0).clamp(0.02, 0.98)
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(a: [f32; 3]) -> [f32; 3] {
    let l = dot(a, a).sqrt();
    [a[0] / l, a[1] / l, a[2] / l]
}

/// `d` turned about x, then y, then z, by `deg` degrees each.
pub fn turn(d: [f32; 3], deg: [f32; 3]) -> [f32; 3] {
    let [p, y, r] = deg.map(f32::to_radians);
    let (sp, cp) = p.sin_cos();
    let d = [d[0], cp * d[1] - sp * d[2], sp * d[1] + cp * d[2]];
    let (sy, cy) = y.sin_cos();
    let d = [cy * d[0] + sy * d[2], d[1], -sy * d[0] + cy * d[2]];
    let (sr, cr) = r.sin_cos();
    [cr * d[0] - sr * d[1], sr * d[0] + cr * d[1], d[2]]
}

impl Scene {
    /// Luminance and opacity seen from `origin` along `d` (unit).
    pub fn trace(&self, origin: [f32; 3], d: [f32; 3]) -> (f32, f32) {
        let mut best: Option<(f32, &Surface, [f32; 3])> = None;
        for s in &self.surfaces {
            let (centre, normal, radius) = match s.shape {
                Shape::Plane { centre, normal } => (centre, normalize(normal), None),
                Shape::Disc {
                    centre,
                    normal,
                    radius,
                } => (centre, normalize(normal), Some(radius)),
            };
            let den = dot(d, normal);
            if den.abs() < 1e-9 {
                continue;
            }
            let t = dot(sub(centre, origin), normal) / den;
            if t <= 0.0 || best.is_some_and(|b| b.0 <= t) {
                continue;
            }
            let hit = [
                origin[0] + t * d[0],
                origin[1] + t * d[1],
                origin[2] + t * d[2],
            ];
            if radius.is_some_and(|r| dot(sub(hit, centre), sub(hit, centre)) > r * r) {
                continue;
            }
            best = Some((t, s, hit));
        }
        match best {
            Some((_, s, hit)) => {
                let (centre, normal) = match s.shape {
                    Shape::Plane { centre, normal } | Shape::Disc { centre, normal, .. } => {
                        (centre, normalize(normal))
                    }
                };
                // Axes in the surface: its first across, level when it can be.
                let across = if normal[1].abs() > 0.99 {
                    [1.0, 0.0, 0.0]
                } else {
                    normalize(cross([0.0, 1.0, 0.0], normal))
                };
                let up = cross(normal, across);
                let p = sub(hit, centre);
                let luma = s.texture.at(dot(p, across), dot(p, up));
                (luma, if s.see_through { 0.0 } else { 1.0 })
            }
            None => {
                let lon = d[0].atan2(-d[2]).to_degrees();
                let lat = d[1].clamp(-1.0, 1.0).asin().to_degrees();
                (self.sky.at(lon, lat), 1.0)
            }
        }
    }
}

/// The direction a point of a camera's picture shows, in that camera's
/// axes; `None` outside it. The inverse of [`PictureGeometry::uv`].
fn direction(g: &PictureGeometry, slot: usize, [u, v]: [f32; 2]) -> Option<[f32; 3]> {
    match g.projection {
        Projection::Equirect { h_fov, v_fov } => {
            let lon = (u - 0.5) * h_fov.to_radians();
            let lat = (0.5 - v) * v_fov.to_radians();
            Some([lon.sin() * lat.cos(), lat.sin(), -lon.cos() * lat.cos()])
        }
        Projection::Fisheye { fov } => {
            let c = g.centres[slot];
            let x = (u - 0.5 - c[0]) / g.circle[0];
            let y = -(v - 0.5 - c[1]) / g.circle[1];
            let rd = (x * x + y * y).sqrt();
            // The lens's radius before distortion, by Newton's method.
            let [k1, k2] = g.lens;
            let mut r = rd;
            for _ in 0..8 {
                let r2 = 4.0 * r * r;
                let f = r * (1.0 + k1 * r2 + k2 * r2 * r2) - rd;
                let df = 1.0 + 3.0 * k1 * r2 + 5.0 * k2 * r2 * r2;
                r -= f / df;
            }
            if r > 0.5 {
                return None;
            }
            let theta = r * fov.to_radians();
            let (dx, dy) = if rd > 1e-9 {
                (x / rd, y / rd)
            } else {
                (0.0, 0.0)
            };
            Some([dx * theta.sin(), dy * theta.sin(), -theta.cos()])
        }
        _ => None,
    }
}

/// A frame of `scenes` (the left camera's and the right's: the same one
/// for a real scene) filmed by `rig`, `width` x `height` pixels laid out
/// as `geometry` says. Each pixel is traced the first time it is read.
pub struct Picture<'a> {
    pub scenes: [&'a Scene; 2],
    pub rig: Rig,
    pub geometry: PictureGeometry,
    pub width: usize,
    pub height: usize,
    cache: RefCell<Vec<Option<(f32, f32)>>>,
}

impl<'a> Picture<'a> {
    pub fn new(
        scenes: [&'a Scene; 2],
        rig: Rig,
        geometry: PictureGeometry,
        width: usize,
        height: usize,
    ) -> Picture<'a> {
        Picture {
            scenes,
            rig,
            geometry,
            width,
            height,
            cache: RefCell::new(vec![None; width * height]),
        }
    }

    /// Every pixel, traced: luminance and opacity, row by row.
    pub fn render(&self) -> (Vec<f32>, Vec<f32>) {
        let mut luma = Vec::with_capacity(self.width * self.height);
        let mut alpha = Vec::with_capacity(self.width * self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let (l, a) = self.pixel(x, y);
                luma.push(l);
                alpha.push(a);
            }
        }
        (luma, alpha)
    }

    fn trace(&self, x: usize, y: usize) -> (f32, f32) {
        let g = &self.geometry;
        let s = [
            (x as f32 + 0.5) / self.width as f32,
            (y as f32 + 0.5) / self.height as f32,
        ];
        // Upright, then which picture and where in it.
        let [u, v] = match g.turns {
            1 => [1.0 - s[1], s[0]],
            2 => [1.0 - s[0], 1.0 - s[1]],
            3 => [s[1], 1.0 - s[0]],
            _ => s,
        };
        let (slot, pu, pv) = if g.stereo.is_side_by_side() {
            let slot = (u >= 0.5) as usize;
            (slot, u * 2.0 - slot as f32, v)
        } else if g.stereo.is_top_bottom() {
            let slot = (v >= 0.5) as usize;
            (slot, u, v * 2.0 - slot as f32)
        } else {
            (0, u, v)
        };
        let camera = if g.stereo == StereoLayout::Mono {
            0
        } else {
            slot ^ g.swapped as usize
        };
        let Some(d) = direction(g, slot, [pu, pv]) else {
            return (0.0, 1.0);
        };
        let d = if camera == 1 {
            turn(d, self.rig.right_turn)
        } else {
            d
        };
        let origin = [(camera as f32 - 0.5) * self.rig.baseline, 0.0, 0.0];
        self.scenes[camera].trace(origin, d)
    }
}

impl Pixels for Picture<'_> {
    fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    fn pixel(&self, x: usize, y: usize) -> (f32, f32) {
        let i = y * self.width + x;
        if let Some(p) = self.cache.borrow()[i] {
            return p;
        }
        let p = self.trace(x, y);
        self.cache.borrow_mut()[i] = Some(p);
        p
    }
}
