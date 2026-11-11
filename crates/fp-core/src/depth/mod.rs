//! Depth from a 3D 180° or fisheye video's two pictures, for automatic
//! stereo focus: how far away what the viewer looks at is.
//!
//! - [`geometry`]: two-view geometry on the sphere, as angles about the
//!   cameras' baseline.
//! - [`picture`]: how a frame holds the cameras' pictures, and (for
//!   tests) where a direction is in it, as the eye passes find it.
//! - [`grid`]: what one measurement samples, and (for tests) the CPU
//!   reference sampler the GPU pass is checked against.
//! - [`Matcher`]: block matching of a measurement's grids, and why it
//!   gave no depth when it didn't.
//! - `scenes` (for tests): ray-traced synthetic videos.
//!
//! What is only for tests is built for fp-core's own and, through the
//! `test-support` feature, for other crates', never into the app.

pub mod geometry;
pub mod grid;
mod matcher;
pub mod picture;
#[cfg(any(test, feature = "test-support"))]
pub mod scenes;

pub use geometry::Angles;
pub use grid::{GridSpec, Grids};
pub use matcher::{Matcher, Measurement, Outcome, Plan, Request, Window};
pub use picture::PictureGeometry;

/// The disparity a measurement resolves, radians: a quarter of a sample
/// (0.05°), what the parabola through the scores round the best finds on
/// textured surfaces. In nearness it is this over the baseline: 0.013
/// 1/m at 65 mm, 1.3 cm at 1 m and 33 cm at 5 m.
pub const RESOLUTION: f32 = grid::PITCH / 4.0;
