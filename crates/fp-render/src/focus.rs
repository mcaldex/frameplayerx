//! Autofocus's reading: how far apart the two eyes' pictures are where you
//! look in a 3D 180°/360° or fisheye video, which says how near what you
//! look at is.
//!
//! When [`VideoParams::focus`](crate::VideoParams::focus) asks, `draw` runs
//! `focus_main` (shaders/scene.wgsl): for each of 96 disparities it turns
//! the eyes' pictures that far apart, as the Depth setting does, and
//! correlates them over a patch where you look. Once the GPU is done
//! with the frame, `read` finds the disparity they match best at, for
//! [`Renderer::focus`](crate::Renderer::focus).

use crate::renderer::EyeView;
use fp_core::format::{Projection, VideoFormat};
use glam::{Vec3, Vec4};

/// The disparities tried, degrees: from `FIRST`, `STEP` apart. A little
/// below 0 is a camera pair slightly off parallel; the last, 21.75°, is
/// something about 17 cm away.
const FIRST: f32 = -2.0;
const STEP: f32 = 0.25;
pub(crate) const STEPS: usize = 96;
/// Half the width of the patch matched, degrees.
const PATCH: f32 = 3.0;
/// Below this correlation the pictures aren't taken to match.
const MIN_MATCH: f32 = 0.5;
/// How much of the patch both pictures must cover.
const MIN_COVER: f32 = 0.5;
/// Another match within this of the best (a repeating pattern) leaves the
/// reading unsure.
const RIVAL: f32 = 0.08;

/// What autofocus sees where you look.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Focus {
    /// How far apart the eyes' pictures are there, in degrees as
    /// [`ViewSettings::ipd_offset`](fp_core::view::ViewSettings::ipd_offset)
    /// turns them: with that much Depth it shows at infinity. About 0 for
    /// something far away, more the nearer it is.
    pub disparity: f32,
    /// How well the pictures match at that disparity: a correlation, 0.5
    /// (the least trusted) to 1.
    pub score: f32,
}

/// Whether autofocus can read `format`: a 3D video on a sphere (180°/360°,
/// fisheye, EAC). A flat 3D film's depth is set for its screen.
pub fn applies(format: &VideoFormat) -> bool {
    format.projection != Projection::Flat && format.stereo.is_stereo()
}

/// The uniform block's autofocus vectors: `towards` (a direction in the
/// world) with the patch's size, the patch's right with the first
/// disparity, and its up with the step. The patch stands as upright as the
/// head between the `eyes` does.
pub(crate) fn aim(towards: Vec3, eyes: &[EyeView; 2]) -> [Vec4; 3] {
    let axis = |v: Vec3| {
        eyes.iter()
            .map(|e| e.view.inverse().transform_vector3(v))
            .sum::<Vec3>()
            .normalize_or_zero()
    };
    let forward = towards.try_normalize().unwrap_or(axis(Vec3::NEG_Z));
    let right = forward
        .cross(axis(Vec3::Y))
        .try_normalize()
        .unwrap_or(axis(Vec3::X));
    let up = right.cross(forward);
    [
        forward.extend(PATCH.to_radians().tan()),
        right.extend(FIRST.to_radians()),
        up.extend(STEP.to_radians()),
    ]
}

/// The best match in one frame's results (per disparity tried: correlation
/// and cover), or `None` when there is none to trust: a flat or repeating
/// patch, or one mostly off the picture.
pub(crate) fn read(results: &[[f32; 2]]) -> Option<Focus> {
    let score = |i: usize| results[i][0];
    let best = (0..results.len())
        .filter(|&i| score(i).is_finite())
        .max_by(|&a, &b| score(a).total_cmp(&score(b)))?;
    let [top, cover] = results[best];
    if top < MIN_MATCH || cover < MIN_COVER {
        return None;
    }
    // Another peak nearly as good: the pattern repeats, and either could be
    // it. The best one's own slopes are not peaks.
    let peak = |i: usize| {
        (i == 0 || score(i) >= score(i - 1)) && (i + 1 == results.len() || score(i) >= score(i + 1))
    };
    if (0..results.len()).any(|i| i.abs_diff(best) > 1 && peak(i) && score(i) > top - RIVAL) {
        return None;
    }
    // Between steps: the top of the parabola through the best and its
    // neighbours.
    let mut between = 0.0;
    if best > 0 && best + 1 < results.len() {
        let (a, c) = (score(best - 1), score(best + 1));
        let curve = a - 2.0 * top + c;
        if curve < -1e-6 {
            between = (0.5 * (a - c) / curve).clamp(-0.5, 0.5);
        }
    }
    Some(Focus {
        disparity: FIRST + (best as f32 + between) * STEP,
        score: top,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame's results with one peak of `top` at disparity `at` degrees,
    /// `width` steps wide, over a floor of 0.1.
    fn curve(at: f32, top: f32, width: f32) -> Vec<[f32; 2]> {
        (0..STEPS)
            .map(|i| {
                let d = (FIRST + i as f32 * STEP - at) / STEP / width;
                [0.1 + (top - 0.1) * (-d * d).exp(), 1.0]
            })
            .collect()
    }

    #[test]
    fn finds_the_peak_between_steps() {
        for at in [0.0, 3.1, 8.0, 15.37] {
            let f = read(&curve(at, 0.9, 2.0)).expect("a clear peak");
            assert!((f.disparity - at).abs() < 0.05, "{at}: {f:?}");
            assert!(f.score > 0.85);
        }
    }

    #[test]
    fn a_weak_match_or_one_off_the_picture_is_unsure() {
        assert_eq!(read(&curve(5.0, 0.4, 2.0)), None);
        let off: Vec<_> = curve(5.0, 0.9, 2.0)
            .into_iter()
            .map(|[s, _]| [s, 0.3])
            .collect();
        assert_eq!(read(&off), None);
        assert_eq!(read(&[[f32::NAN, 1.0]; STEPS]), None);
    }

    #[test]
    fn a_repeating_pattern_is_unsure() {
        let mut r = curve(4.0, 0.9, 1.5);
        for (i, near) in curve(14.0, 0.88, 1.5).into_iter().enumerate() {
            r[i][0] = r[i][0].max(near[0]);
        }
        assert_eq!(read(&r), None);
        // A much weaker second peak is only a lookalike.
        let mut r = curve(4.0, 0.9, 1.5);
        for (i, near) in curve(14.0, 0.6, 1.5).into_iter().enumerate() {
            r[i][0] = r[i][0].max(near[0]);
        }
        assert!((read(&r).expect("the best peak").disparity - 4.0).abs() < 0.05);
    }

    #[test]
    fn the_patch_faces_where_you_look_upright() {
        use crate::math::{Fov, projection, view};
        use glam::Quat;
        let head = Quat::from_rotation_y(0.5) * Quat::from_rotation_x(-0.2);
        let ev = |x: f32| EyeView {
            view: view(head * Vec3::new(x, 0.0, 0.0), head),
            proj: projection(Fov::symmetric(90.0), 0.05, 100.0),
        };
        let eyes = [ev(-0.032), ev(0.032)];
        let [f, r, u] = aim(head * Vec3::NEG_Z, &eyes);
        assert!(f.truncate().distance(head * Vec3::NEG_Z) < 1e-5);
        assert!(r.truncate().distance(head * Vec3::X) < 1e-5);
        assert!(u.truncate().distance(head * Vec3::Y) < 1e-5);
        // The eyes turned up and to the left of the head: the patch faces
        // that way, its right still level with the head's.
        let gaze = head * (Quat::from_rotation_y(0.3) * Quat::from_rotation_x(0.2)) * Vec3::NEG_Z;
        let [f, r, u] = aim(gaze * 3.0, &eyes);
        assert!(f.truncate().distance(gaze) < 1e-5);
        assert!(r.truncate().dot(head * Vec3::Y).abs() < 1e-5);
        assert!(u.truncate().dot(gaze).abs() < 1e-5);
        assert!((r.truncate().length() - 1.0).abs() < 1e-5);
    }
}
