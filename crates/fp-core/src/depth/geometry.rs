//! Two-view geometry on the sphere. In the video's axes (x right, y up, -z
//! the picture's front) the two cameras sit on the x axis, `baseline`
//! metres apart with the origin between them. Any point and the two
//! cameras span a plane through the x axis, so a direction is two angles:
//! `alpha` from +x (π/2 straight ahead) and `beta`, the turn of that plane
//! about the x axis (0 ahead, positive up). A point has the same `beta`
//! from both cameras; only `alpha` differs, by the disparity
//! `alpha_right - alpha_left`, which is the angle at the point in the
//! triangle it makes with the cameras: 0 at infinity, larger nearer.

/// A direction as its angles about the baseline, radians.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Angles {
    pub alpha: f32,
    pub beta: f32,
}

impl Angles {
    /// The angles of direction `d` (any length but zero).
    pub fn of(d: [f32; 3]) -> Angles {
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
            .sqrt()
            .max(f32::MIN_POSITIVE);
        Angles {
            alpha: (d[0] / len).clamp(-1.0, 1.0).acos(),
            beta: d[1].atan2(-d[2]),
        }
    }

    /// The unit direction: `(cos α, sin α sin β, -sin α cos β)`.
    pub fn direction(self) -> [f32; 3] {
        let (sa, ca) = self.alpha.sin_cos();
        let (sb, cb) = self.beta.sin_cos();
        [ca, sa * sb, -sa * cb]
    }
}

/// How near a point is, 1/metres from the middle of the cameras, that a
/// pair `baseline` metres apart sees at `alpha_left` and `alpha_right`
/// (radians, the same `beta`). 0 at infinity; negative for rays that part
/// (a disparity below 0, which a rig's own misalignment can give), which
/// count as farther than anything. Nearness rather than distance, as depth
/// resolution is even in it and it has no infinity.
pub fn nearness(alpha_left: f32, alpha_right: f32, baseline: f32) -> f32 {
    let disparity = alpha_right - alpha_left;
    // The law of sines in the triangle of the cameras and the point: the
    // angle at the point is the disparity, opposite the baseline.
    let sin_right = alpha_right.sin();
    if sin_right.abs() < 1e-6 || baseline <= 0.0 {
        return 0.0;
    }
    // 1 / range from the left camera.
    let inv_left = disparity.sin() / (baseline * sin_right);
    // The range from the middle, through the left camera's ray.
    let m = 1.0 - baseline * inv_left * alpha_left.cos()
        + 0.25 * baseline * baseline * inv_left * inv_left;
    inv_left / m.max(1e-12).sqrt()
}

/// The disparity, radians, of a point `1 / nearness` metres from the
/// middle of cameras `baseline` metres apart, in direction `alpha` from
/// there. The inverse of [`nearness`]; 0 at infinity (`nearness` 0).
pub fn disparity(nearness: f32, alpha: f32, baseline: f32) -> f32 {
    if nearness <= 0.0 {
        return 0.0;
    }
    // In the plane of the point: x along the baseline, y away from it.
    let (y, x) = (alpha.sin(), alpha.cos());
    let h = 0.5 * baseline * nearness;
    let left = y.atan2(x + h);
    let right = y.atan2(x - h);
    right - left
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn angles_and_directions_agree() {
        assert_eq!(
            Angles {
                alpha: FRAC_PI_2,
                beta: 0.0
            }
            .direction()
            .map(|c| (c * 1e6).round()),
            [0.0, 0.0, -1e6]
        );
        let up = Angles::of([0.0, 1.0, 0.0]);
        assert!((up.alpha - FRAC_PI_2).abs() < 1e-6 && (up.beta - FRAC_PI_2).abs() < 1e-6);
        for (a, b) in [(0.3, 0.0), (1.2, -0.7), (2.9, 1.4), (1.6, 2.5)] {
            let back = Angles::of(Angles { alpha: a, beta: b }.direction());
            assert!((back.alpha - a).abs() < 1e-5 && (back.beta - b).abs() < 1e-5);
        }
    }

    #[test]
    fn a_point_has_one_beta_and_its_disparity_gives_its_distance() {
        let baseline = 0.065;
        let cameras = [[-baseline / 2.0, 0.0, 0.0], [baseline / 2.0, 0.0, 0.0]];
        for point in [
            [0.0, 0.0, -1.0],
            [0.3, 0.2, -0.25],
            [-2.0, 1.0, -3.0],
            [1.5, -0.6, -0.4],
            [-0.5, 3.0, 2.0],
        ] {
            let seen =
                cameras.map(|c| Angles::of([point[0] - c[0], point[1] - c[1], point[2] - c[2]]));
            assert!((seen[0].beta - seen[1].beta).abs() < 1e-6, "{point:?}");
            let range = (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt();
            let n = nearness(seen[0].alpha, seen[1].alpha, baseline);
            assert!((n * range - 1.0).abs() < 1e-4, "{point:?}: {}", 1.0 / n);
            let middle = Angles::of(point);
            let d = disparity(n, middle.alpha, baseline);
            assert!(
                (d - (seen[1].alpha - seen[0].alpha)).abs() < 1e-6,
                "{point:?}"
            );
        }
        // Straight ahead at 65 mm: 12.4° at 0.3 m, 3.7° at 1 m, 0.19° at 20 m.
        for (m, deg) in [(0.3, 12.36), (1.0, 3.72), (20.0, 0.186)] {
            let d = disparity(1.0 / m, FRAC_PI_2, baseline).to_degrees();
            assert!((d - deg).abs() < 0.01, "{m} m: {d}°");
        }
        assert_eq!(
            nearness(1.0, 1.0, baseline),
            0.0,
            "parallel: infinitely far"
        );
        assert!(nearness(1.0, 0.99, baseline) < 0.0, "parting: beyond");
    }
}
