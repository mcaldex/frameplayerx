//! How a frame holds a 3D video's two pictures, as the projection
//! shader's uniforms describe it; and, for tests, where a direction of a
//! camera's view is in the frame, as the eye passes find it (fp-render's
//! `picture.wgsl`: `equirect_uv`, `fisheye_uv`, `to_eye_rect`, `stored_uv`),
//! and a frame in memory read there as the GPU reads one: the CPU
//! yardstick for the measurement pass.

use crate::format::{Projection, StereoLayout, VideoFormat};
use crate::view::ViewSettings;

/// How a frame holds its two cameras' pictures, as the projection shader's
/// uniforms describe it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PictureGeometry {
    /// Equirectangular or fisheye; nothing else is two forward cameras'.
    pub projection: Projection,
    pub stereo: StereoLayout,
    /// The frame holds the right camera's picture first (the file's
    /// packing and the viewer's Swap eyes together).
    pub swapped: bool,
    /// Fisheye: radial lens distortion (k1, k2).
    pub lens: [f32; 2],
    /// Fisheye: the image circle's diameter in eye-image widths and heights.
    pub circle: [f32; 2],
    /// Fisheye: each picture's circle centre from its middle, in
    /// eye-image units, right and down; the first picture in the frame
    /// first.
    pub centres: [[f32; 2]; 2],
    /// Clockwise quarter turns that show the stored frame upright.
    pub turns: u32,
}

impl PictureGeometry {
    /// The geometry of a frame `width` x `height` pixels stored, of pixels
    /// `pixel_aspect` times as wide as tall, turned `rotation` degrees
    /// clockwise to show upright, for `format` adjusted by `settings`.
    /// The same as fp-render's `params::build` gives the shader.
    pub fn new(
        format: &VideoFormat,
        settings: &ViewSettings,
        width: u32,
        height: u32,
        pixel_aspect: f32,
        rotation: u32,
    ) -> PictureGeometry {
        let pixel_aspect = format.stereo.eye_pixel_aspect(pixel_aspect as f64) as f32;
        let turns = rotation / 90 % 4;
        let (pw, ph) = (
            width.max(1) as f32 * pixel_aspect.max(0.01),
            height.max(1) as f32,
        );
        let (pw, ph) = if turns % 2 == 1 { (ph, pw) } else { (pw, ph) };
        let (ew, eh) = match format.stereo {
            StereoLayout::SideBySide | StereoLayout::FrameSequential => (pw / 2.0, ph),
            StereoLayout::TopBottom => (pw, ph / 2.0),
            StereoLayout::HalfSideBySide | StereoLayout::HalfTopBottom | StereoLayout::Mono => {
                (pw, ph)
            }
        };
        let [c0, c1] = settings.lens_centre;
        PictureGeometry {
            projection: format.projection,
            stereo: format.stereo,
            swapped: format.eyes_swapped ^ settings.swap_eyes,
            lens: [settings.lens_k1, settings.lens_k2],
            circle: [(eh / ew).min(1.0), (ew / eh).min(1.0)],
            centres: [[c0[0], -c0[1]], [c1[0], -c1[1]]],
            turns,
        }
    }
}

#[cfg(any(test, feature = "test-support"))]
impl PictureGeometry {
    /// Where camera `camera` (0 left, 1 right) saw direction `d` (video
    /// axes, unit length), in the stored frame (0..1, origin top-left);
    /// `None` outside its picture.
    pub fn uv(&self, camera: usize, d: [f32; 3]) -> Option<[f32; 2]> {
        let stereo = self.stereo != StereoLayout::Mono;
        let slot = if stereo {
            (camera ^ self.swapped as usize) as f32
        } else {
            0.0
        };
        let uv = match self.projection {
            Projection::Equirect { h_fov, v_fov } => {
                equirect_uv(d, h_fov.to_radians(), v_fov.to_radians())?
            }
            Projection::Fisheye { fov } => {
                let c = self.centres[slot as usize];
                fisheye_uv(
                    d,
                    fov.to_radians(),
                    self.lens,
                    self.circle,
                    [0.5 + c[0], 0.5 + c[1]],
                )?
            }
            Projection::Flat | Projection::Eac { .. } => return None,
        };
        Some(self.picture_uv(camera, uv))
    }

    /// The stored frame's coordinates of a point of camera `camera`'s
    /// picture (0..1 across that picture), wherever it is packed.
    pub fn picture_uv(&self, camera: usize, uv: [f32; 2]) -> [f32; 2] {
        let slot = if self.stereo != StereoLayout::Mono {
            (camera ^ self.swapped as usize) as f32
        } else {
            0.0
        };
        let [u, v] = uv;
        let eye = if self.stereo.is_side_by_side() {
            [u * 0.5 + slot * 0.5, v]
        } else if self.stereo.is_top_bottom() {
            [u, v * 0.5 + slot * 0.5]
        } else {
            [u, v]
        };
        self.stored(eye)
    }

    /// The shader's `stored_uv`: a point of the upright picture in the
    /// stored frame.
    fn stored(&self, [u, v]: [f32; 2]) -> [f32; 2] {
        match self.turns {
            1 => [v, 1.0 - u],
            2 => [1.0 - u, 1.0 - v],
            3 => [1.0 - v, u],
            _ => [u, v],
        }
    }
}

/// The shader's `equirect_uv`.
#[cfg(any(test, feature = "test-support"))]
fn equirect_uv(d: [f32; 3], h_fov: f32, v_fov: f32) -> Option<[f32; 2]> {
    let lon = d[0].atan2(-d[2]);
    let lat = d[1].clamp(-1.0, 1.0).asin();
    let mut u = 0.5 + lon / h_fov;
    let v = 0.5 - lat / v_fov;
    if h_fov >= 2.0 * std::f32::consts::PI - 1e-3 {
        u = u - u.floor();
    } else if !(0.0..=1.0).contains(&u) {
        return None;
    }
    (0.0..=1.0).contains(&v).then_some([u, v])
}

/// The shader's `fisheye_uv`.
#[cfg(any(test, feature = "test-support"))]
fn fisheye_uv(
    d: [f32; 3],
    fov: f32,
    [k1, k2]: [f32; 2],
    size: [f32; 2],
    centre: [f32; 2],
) -> Option<[f32; 2]> {
    let theta = (-d[2]).clamp(-1.0, 1.0).acos();
    let mut r = theta / fov;
    if r > 0.5 {
        return None;
    }
    let r2 = r * r * 4.0;
    r *= 1.0 + k1 * r2 + k2 * r2 * r2;
    let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
    let dir = if len > 1e-6 {
        [d[0] / len, d[1] / len]
    } else {
        [0.0, 0.0]
    };
    let uv = [
        centre[0] + dir[0] * r * size[0],
        centre[1] - dir[1] * r * size[1],
    ];
    uv.iter().all(|c| (0.0..=1.0).contains(c)).then_some(uv)
}

/// A stored frame's pixels as the reference sampler reads them:
/// perceptual (gamma-encoded) luminance, 0..1, and opacity, 1 where the
/// video shows and 0 where it is see-through.
#[cfg(any(test, feature = "test-support"))]
pub trait Pixels {
    /// Width and height, pixels.
    fn size(&self) -> (usize, usize);
    /// Luminance and opacity of pixel (`x`, `y`), from the top-left.
    fn pixel(&self, x: usize, y: usize) -> (f32, f32);

    /// Luminance and opacity at `uv` (0..1, origin top-left), filtered as
    /// a GPU's linear sampler with clamp-to-edge filters a texture: between
    /// the four nearest pixel centres.
    fn sample(&self, [u, v]: [f32; 2]) -> (f32, f32) {
        let (w, h) = self.size();
        let (w, h) = (w.max(1), h.max(1));
        let x = u * w as f32 - 0.5;
        let y = v * h as f32 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let clamp = |c: f32, n: usize| (c.max(0.0) as usize).min(n - 1);
        let (xa, xb) = (clamp(x0, w), clamp(x0 + 1.0, w));
        let (ya, yb) = (clamp(y0, h), clamp(y0 + 1.0, h));
        let [p00, p10, p01, p11] =
            [(xa, ya), (xb, ya), (xa, yb), (xb, yb)].map(|(x, y)| self.pixel(x, y));
        let mix = |a: f32, b: f32, c: f32, d: f32| {
            let top = a + (b - a) * fx;
            let bottom = c + (d - c) * fx;
            top + (bottom - top) * fy
        };
        (
            mix(p00.0, p10.0, p01.0, p11.0),
            mix(p00.1, p10.1, p01.1, p11.1),
        )
    }
}

/// A stored frame in memory.
#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Copy, Debug)]
pub struct Frame<'a> {
    pub width: usize,
    pub height: usize,
    /// Row by row from the top.
    pub luma: &'a [f32],
    /// `None`: opaque everywhere.
    pub alpha: Option<&'a [f32]>,
}

#[cfg(any(test, feature = "test-support"))]
impl Pixels for Frame<'_> {
    fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    fn pixel(&self, x: usize, y: usize) -> (f32, f32) {
        let i = y * self.width + x;
        (self.luma[i], self.alpha.map_or(1.0, |a| a[i]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(projection: Projection, stereo: StereoLayout) -> PictureGeometry {
        PictureGeometry::new(
            &VideoFormat::new(projection, stereo),
            &ViewSettings::default(),
            2048,
            1024,
            1.0,
            0,
        )
    }

    #[test]
    fn each_camera_is_found_in_its_half_whatever_the_packing() {
        let ahead = [0.0, 0.0, -1.0];
        let sbs = geometry(Projection::EQUIRECT_180, StereoLayout::SideBySide);
        assert_eq!(sbs.uv(0, ahead), Some([0.25, 0.5]));
        assert_eq!(sbs.uv(1, ahead), Some([0.75, 0.5]));
        let swapped = PictureGeometry {
            swapped: true,
            ..sbs
        };
        assert_eq!(swapped.uv(0, ahead), Some([0.75, 0.5]));
        let tb = geometry(Projection::fisheye(180.0), StereoLayout::TopBottom);
        assert_eq!(tb.uv(1, ahead), Some([0.5, 0.75]));
        // Behind a 180° picture there is nothing.
        assert_eq!(sbs.uv(0, [0.0, 0.0, 1.0]), None);
        assert_eq!(tb.uv(0, [0.0, 0.0, 1.0]), None);
        // A quarter turn: the stored frame is the upright one turned back.
        let turned = PictureGeometry { turns: 1, ..sbs };
        assert_eq!(turned.uv(0, ahead), Some([0.5, 0.75]));
    }

    #[test]
    fn a_frame_is_filtered_between_pixel_centres() {
        // 2 x 1 pixels: 0 and 1.
        let f = Frame {
            width: 2,
            height: 1,
            luma: &[0.0, 1.0],
            alpha: None,
        };
        assert_eq!(f.sample([0.25, 0.5]), (0.0, 1.0));
        assert_eq!(f.sample([0.5, 0.5]).0, 0.5);
        assert_eq!(f.sample([0.0, 0.5]).0, 0.0, "clamped to the edge");
        assert_eq!(f.sample([1.0, 0.5]).0, 1.0);
    }
}
