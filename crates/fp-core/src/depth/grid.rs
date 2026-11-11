//! What one measurement samples: for each camera, a small grid of
//! perceptual luminance on the `(alpha, beta)` lattice of
//! [`super::geometry`] around the aim, so matching is a search along rows
//! for any direction and either projection; and a coarse picture of the
//! left camera's whole view, for noticing cuts. The GPU pass fills it from
//! the frame the eye passes show; for tests, `sample` fills it from a
//! frame in memory, as the yardstick.

use super::geometry::Angles;
#[cfg(any(test, feature = "test-support"))]
use super::picture::{PictureGeometry, Pixels};

/// The angle between neighbouring samples, radians: 0.2°. A 4K 180°
/// picture has 0.088° a pixel and an 8K one 0.044°, so finer would mostly
/// resample the same detail; coarser loses the texture of faces and
/// fabric at a metre or two. It sets the disparity step the matcher
/// refines a quarter of (see `super::RESOLUTION`) and, with the window
/// and the depths searched, how many samples a measurement takes.
pub const PITCH: f32 = 0.2 * std::f32::consts::PI / 180.0;

/// Taps a sample averages along each side: 3, 0.067° apart, so on an 8K
/// picture (0.044° a pixel, read bilinearly) a sample covers its whole
/// cell instead of picking one spot of it, which would alias differently
/// in the two pictures.
pub const TAPS: usize = 3;

/// The coarse picture of the left camera's view: `COARSE` x `COARSE`
/// samples of `COARSE_TAPS` x `COARSE_TAPS` taps. 1024 values tell a cut
/// from a pan or a fade; the 4096 taps they take are under a tenth of a
/// measurement's.
pub const COARSE: usize = 32;
pub const COARSE_TAPS: usize = 2;

/// A sample outside the camera's picture (beyond a fisheye's circle, past
/// a 180° picture's edge).
pub const OUTSIDE: f32 = -1.0;
/// A sample where the video is see-through (chroma-keyed or masked), so
/// the room would show.
pub const SEE_THROUGH: f32 = -2.0;
/// Opacity, averaged over a sample's taps, below which it is see-through.
pub const OPAQUE: f32 = 0.5;

/// Where each camera's grid lies. Column `c`, row `r` of camera `i`'s grid
/// is at `alpha = origin[i].alpha + c * pitch`, `beta = origin[i].beta +
/// r * pitch`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridSpec {
    /// Radians between samples.
    pub pitch: f32,
    pub cols: usize,
    pub rows: usize,
    pub origin: [Angles; 2],
}

impl GridSpec {
    /// The angles of a point of camera `camera`'s grid, `col` and `row`
    /// counted in samples (fractions between them).
    pub fn angles(&self, camera: usize, col: f32, row: f32) -> Angles {
        let o = self.origin[camera];
        Angles {
            alpha: o.alpha + col * self.pitch,
            beta: o.beta + row * self.pitch,
        }
    }

    /// How many samples both grids hold.
    pub fn len(&self) -> usize {
        self.cols * self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Where tap `i` of `n` sits in its sample's cell, in samples from its
/// middle.
pub fn tap_offset(i: usize, n: usize) -> f32 {
    (i as f32 + 0.5) / n as f32 - 0.5
}

/// One measurement's samples.
#[derive(Clone, Debug, PartialEq)]
pub struct Grids {
    pub spec: GridSpec,
    /// Each camera's grid, row by row from row 0: luminance 0..1, or
    /// [`OUTSIDE`] or [`SEE_THROUGH`].
    pub samples: [Vec<f32>; 2],
    /// The left camera's whole picture, [`COARSE`] rows of [`COARSE`]
    /// luminance values from its top-left.
    pub coarse: Vec<f32>,
}

/// One sample from its taps' luminance and opacity, `None` for a tap
/// outside the picture: outside if any tap is (a cell over the edge can't
/// be matched), see-through if they are mostly, else their mean luminance.
#[cfg(any(test, feature = "test-support"))]
pub fn combine(taps: impl IntoIterator<Item = Option<(f32, f32)>>) -> f32 {
    let (mut luma, mut alpha, mut n) = (0.0, 0.0, 0.0);
    for t in taps {
        let Some((l, a)) = t else {
            return OUTSIDE;
        };
        luma += l;
        alpha += a;
        n += 1.0;
    }
    if n == 0.0 {
        OUTSIDE
    } else if alpha / n < OPAQUE {
        SEE_THROUGH
    } else {
        luma / n
    }
}

/// The CPU reference sampler: `spec`'s grids and the coarse picture read
/// from `frame`, laid out as `geometry` says. Each tap is filtered
/// bilinearly from the frame's luminance, where the GPU filters the
/// gamma-encoded YUV and converts after; the two agree exactly for grey
/// and differ by the curvature of the conversion over a pixel otherwise.
#[cfg(any(test, feature = "test-support"))]
pub fn sample(frame: &impl Pixels, geometry: &PictureGeometry, spec: &GridSpec) -> Grids {
    let samples = [0, 1].map(|camera| {
        let mut out = Vec::with_capacity(spec.len());
        for row in 0..spec.rows {
            for col in 0..spec.cols {
                let taps = (0..TAPS * TAPS).map(|t| {
                    let a = spec.angles(
                        camera,
                        col as f32 + tap_offset(t % TAPS, TAPS),
                        row as f32 + tap_offset(t / TAPS, TAPS),
                    );
                    geometry
                        .uv(camera, a.direction())
                        .map(|uv| frame.sample(uv))
                });
                out.push(combine(taps));
            }
        }
        out
    });
    Grids {
        spec: *spec,
        samples,
        coarse: coarse(frame, geometry),
    }
}

/// The coarse picture of the left camera's view from `frame`.
#[cfg(any(test, feature = "test-support"))]
pub fn coarse(frame: &impl Pixels, geometry: &PictureGeometry) -> Vec<f32> {
    let mut out = Vec::with_capacity(COARSE * COARSE);
    for row in 0..COARSE {
        for col in 0..COARSE {
            let mut sum = 0.0;
            for t in 0..COARSE_TAPS * COARSE_TAPS {
                let u =
                    (col as f32 + 0.5 + tap_offset(t % COARSE_TAPS, COARSE_TAPS)) / COARSE as f32;
                let v =
                    (row as f32 + 0.5 + tap_offset(t / COARSE_TAPS, COARSE_TAPS)) / COARSE as f32;
                sum += frame.sample(geometry.picture_uv(0, [u, v])).0;
            }
            out.push(sum / (COARSE_TAPS * COARSE_TAPS) as f32);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sample_is_outside_if_any_tap_is_and_see_through_if_most_are() {
        assert_eq!(combine([Some((0.2, 1.0)), Some((0.4, 1.0))]), 0.3);
        assert_eq!(combine([Some((0.2, 1.0)), None]), OUTSIDE);
        assert_eq!(combine([Some((0.2, 0.0)), Some((0.4, 0.2))]), SEE_THROUGH);
        // A see-through spot inside a sample is still its luminance.
        assert_eq!(combine([Some((0.2, 0.0)), Some((0.4, 1.0))]), 0.3);
        let offsets: Vec<f32> = (0..3).map(|i| tap_offset(i, 3)).collect();
        assert!((offsets[0] + 1.0 / 3.0).abs() < 1e-6 && offsets[1] == 0.0);
    }
}
