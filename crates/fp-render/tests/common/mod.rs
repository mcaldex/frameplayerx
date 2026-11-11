//! What the GPU tests share: soft dots painted where cameras saw points,
//! and finding where a dot lands in an eye image to a fraction of a pixel.

#![allow(dead_code)]

use fp_core::format::Projection;
use glam::DVec3;
use std::sync::Arc;

/// The direction a point of an eye's picture shows, in the video's axes,
/// from its coordinates in that picture (0..1, origin top-left).
pub fn direction(projection: Projection, u: f64, v: f64) -> Option<DVec3> {
    match projection {
        Projection::Equirect { h_fov, v_fov } => {
            let lon = (u - 0.5) * (h_fov as f64).to_radians();
            let lat = (0.5 - v) * (v_fov as f64).to_radians();
            Some(DVec3::new(
                lon.sin() * lat.cos(),
                lat.sin(),
                -lon.cos() * lat.cos(),
            ))
        }
        Projection::Fisheye { fov } => {
            let (x, y) = (u - 0.5, 0.5 - v);
            let r = x.hypot(y);
            if r > 0.5 {
                return None;
            }
            let theta = r * (fov as f64).to_radians();
            let (dx, dy) = if r > 0.0 { (x / r, y / r) } else { (0.0, 0.0) };
            Some(DVec3::new(dx * theta.sin(), dy * theta.sin(), -theta.cos()))
        }
        _ => unreachable!(),
    }
}

/// A side-by-side frame, `eye` pixels square per eye, black but for a
/// soft white dot in each camera's picture where it saw `seen[camera]`,
/// `sigma` degrees across. `right_first` packs the right camera's picture
/// on the left.
pub fn dots(
    projection: Projection,
    eye: u32,
    seen: [DVec3; 2],
    sigma: f64,
    right_first: bool,
) -> Arc<fp_media::VideoFrame> {
    let (w, h) = (eye * 2, eye);
    let sigma = sigma.to_radians();
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let half = (x >= eye) as usize;
            let camera = half ^ right_first as usize;
            let (u, v) = (
                ((x % eye) as f64 + 0.5) / eye as f64,
                (y as f64 + 0.5) / h as f64,
            );
            let level = direction(projection, u, v).map_or(0.0, |d| {
                let angle = 2.0 * ((d - seen[camera]).length() / 2.0).asin();
                (-angle * angle / (2.0 * sigma * sigma)).exp()
            });
            let c = (level * 255.0).round() as u8;
            let i = ((y * w + x) * 4) as usize;
            rgba[i..i + 4].copy_from_slice(&[c, c, c, 255]);
        }
    }
    Arc::new(fp_media::VideoFrame::from_rgba(w, h, &rgba).expect("frame"))
}

/// Where the brightest spot of a `size`-pixel-square eye image peaks,
/// pixels from the top-left pixel's centre: a quadratic fitted to the log
/// of the bright pixels. Seen through any smooth warp, a soft dot still
/// peaks where its centre lands, unlike its centroid.
pub fn peak(img: &[u8], size: u32) -> (f64, f64) {
    let size = size as i64;
    let at = |x: i64, y: i64| -> f64 {
        if x < 0 || y < 0 || x >= size || y >= size {
            return 0.0;
        }
        img[((y * size + x) * 4 + 1) as usize] as f64
    };
    let (mut bx, mut by, mut best) = (0, 0, -1.0);
    for y in 0..size {
        for x in 0..size {
            if at(x, y) > best {
                (bx, by, best) = (x, y, at(x, y));
            }
        }
    }
    assert!(best > 100.0, "no dot in the image (brightest {best})");
    // Weighted least squares of ln I = a + bx + cy + dx² + exy + fy²; a
    // pixel's weight is I², as rounding moves ln I by about 1/I.
    let mut m = [[0.0f64; 7]; 6];
    for dy in -8..=8i64 {
        for dx in -8..=8i64 {
            let i = at(bx + dx, by + dy);
            if i < 0.3 * best {
                continue;
            }
            let (x, y) = (dx as f64, dy as f64);
            let row = [1.0, x, y, x * x, x * y, y * y];
            let wt = i * i;
            for (j, rj) in row.iter().enumerate() {
                for (k, rk) in row.iter().enumerate() {
                    m[j][k] += wt * rj * rk;
                }
                m[j][6] += wt * rj * i.ln();
            }
        }
    }
    // Gauss-Jordan elimination with partial pivoting.
    for c in 0..6 {
        let p = (c..6)
            .max_by(|&a, &b| m[a][c].abs().total_cmp(&m[b][c].abs()))
            .unwrap();
        m.swap(c, p);
        let pivot = m[c];
        for (r, row) in m.iter_mut().enumerate() {
            if r != c {
                let f = row[c] / pivot[c];
                for (x, p) in row.iter_mut().zip(pivot).skip(c) {
                    *x -= f * p;
                }
            }
        }
    }
    let s: Vec<f64> = (0..6).map(|i| m[i][6] / m[i][i]).collect();
    let (b, c, d, e, f) = (s[1], s[2], s[3], s[4], s[5]);
    // Where the gradient is zero: [2d e; e 2f] (x, y) = -(b, c).
    let det = 4.0 * d * f - e * e;
    let x = (-b * 2.0 * f + c * e) / det;
    let y = (-c * 2.0 * d + b * e) / det;
    (bx as f64 + x, by as f64 + y)
}
