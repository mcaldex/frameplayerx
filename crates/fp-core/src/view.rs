//! Per-video picture and placement corrections (HereSphere-style).

use crate::format::VideoFormat;
use serde::{Deserialize, Serialize};

/// The colour around a flat screen to start with, a near-black grey.
pub const BACKGROUND: [f32; 3] = [0.086, 0.086, 0.11];

/// The focus distances offered, metres: an interface range we chose. At
/// 0.3 m a point is 12.4° of disparity apart in the two pictures of a
/// 65 mm camera, about as near as anything is filmed; at 20 m it is 0.19°,
/// about four pixels of an 8K picture, hard to tell from infinitely far.
pub const FOCUS_DISTANCES: std::ops::RangeInclusive<f32> = 0.3..=20.0;

/// The focus distance a video starts with, metres. Advice for filming
/// 180° video puts people 1 to 2 m from the camera, where they look
/// present and are comfortable to look at; 1.5 m sits between that and
/// the background behind them, which is what most of the picture is, so
/// neither is far off when the head turns before a distance is set.
/// Nearer, the background breaks up sooner as the head tilts; farther, the
/// people do.
pub const FOCUS_DISTANCE: f32 = 1.5;

/// The lens spacings offered, millimetres: an interface range we chose,
/// well around the 60 to 65 mm of most 180° 3D cameras.
pub const LENS_SPACINGS: std::ops::RangeInclusive<f32> = 30.0..=120.0;

/// The lens spacing a video starts with, millimetres: the middle of most
/// 180° 3D cameras' 60 to 65 mm, and the average adult's eye spacing, so
/// for most videos and viewers, looking straight ahead, turning stereo
/// focus on changes nothing.
pub const LENS_SPACING: f32 = 63.0;

/// Stereo focus: whether a 3D 180° or fisheye video is redrawn for where
/// the viewer's eyes are ([`VideoFormat::is_forward_stereo`]). Without it
/// each eye is shown its camera's picture as if infinitely far away, which
/// is right only while the eyes are where the cameras were: head level,
/// facing the picture's front, eyes as far apart as the lenses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Focus {
    #[default]
    Off,
    /// Exact for things [`ViewSettings::focus_distance`] from the camera.
    Manual,
    /// Exact for what the viewer looks at: its distance measured from the
    /// video's two pictures ([`crate::autofocus`]), followed between
    /// [`ViewSettings::focus_nearest`] and [`ViewSettings::focus_farthest`]
    /// at [`ViewSettings::focus_speed`].
    Auto,
}

/// The nearest distance automatic focus follows to start with, metres.
/// Advice for filming 180° video keeps people a metre or more from the
/// camera and close-ups past half of one; nearer is rare, and searching no
/// nearer than this (less a quarter) halves the room a repeating pattern
/// has to fit a second depth, against searching to 0.3 m.
pub const FOCUS_NEAREST: f32 = 0.5;

/// Automatic focus's glide speeds offered: a quarter to four times the
/// eyes' own pace (an interface range we chose).
pub const FOCUS_SPEEDS: std::ops::RangeInclusive<f32> = 0.25..=4.0;

/// Everything the viewer can adjust about how a video is shown. Defaults are
/// neutral. Persisted per video by the library and applied by the renderer.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ViewSettings {
    /// Rotates the whole image: yaw (left/right), pitch (tilt), roll, degrees.
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    /// Magnification; 1.0 is natural size.
    pub zoom: f32,
    /// Moves the viewer inside a 180°/360° video, metres: up, and towards
    /// the front of the picture.
    /// Unlike pitch this shifts near things against far ones, as DeoVR's
    /// height does.
    pub height: f32,
    pub forward: f32,
    /// Software IPD: extra horizontal convergence between eyes, degrees.
    /// Positive pushes the image further away.
    pub ipd_offset: f32,
    /// Per-eye stereo alignment errors from the camera, degrees.
    pub vertical_align: f32,
    pub rotation_align: f32,
    /// Radial lens distortion correction (k1, k2) applied in source space.
    pub lens_k1: f32,
    pub lens_k2: f32,
    /// Fisheye: where each image circle's centre sits, from the centre of
    /// its eye image, as fractions of that image's width (right) and height
    /// (up). The left or top image first; a mono video uses only that one.
    pub lens_centre: [[f32; 2]; 2],
    /// Swap left and right eyes.
    pub swap_eyes: bool,
    /// Picture: brightness and contrast around 0 and 1, saturation 1,
    /// gamma 1, sharpen 0..1.
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub gamma: f32,
    pub sharpen: f32,
    /// Flat screen: distance in metres, width in metres, curvature 0..1.
    pub screen_distance: f32,
    pub screen_width: f32,
    pub screen_curvature: f32,
    /// The colour around a flat screen and the menus while the room doesn't
    /// show, sRGB 0..1.
    pub background: [f32; 3],
    /// Subtitle depth in metres.
    pub subtitle_distance: f32,
    /// Chroma key: makes one colour of the video see-through (a green or
    /// blue screen), so passthrough or the background shows there.
    pub chroma_key: bool,
    /// The colour to remove, sRGB 0..1.
    pub key_color: [f32; 3],
    /// How far (in chroma) from the key colour a pixel still disappears.
    pub key_similarity: f32,
    /// Width of the soft edge beyond that.
    pub key_smoothness: f32,
    /// How much key-coloured fringe to take out of what stays.
    pub key_spill: f32,
    /// Whether this video keeps its own chroma key rather than following the
    /// global one (Settings > Passthrough). `None` for settings saved before
    /// the choice existed: those keep their own if they had the key on.
    #[serde(default)]
    pub key_own: Option<bool>,
    /// Stereo focus: each eye's picture is treated as painted on a sphere
    /// round its camera, `focus_distance` metres in radius, and drawn as
    /// that eye sees it from where it is. Things that far from the camera
    /// are where real ones would be, whatever the head does.
    pub focus: Focus,
    pub focus_distance: f32,
    /// Automatic focus: how fast it glides, 1 the eyes' own pace; and the
    /// nearest and farthest distances it follows, metres (outside them it
    /// holds; either end of [`FOCUS_DISTANCES`] is no limit).
    pub focus_speed: f32,
    pub focus_nearest: f32,
    pub focus_farthest: f32,
    /// How far apart the two lenses of the camera that filmed the video
    /// were, millimetres: a property of the video, not of the viewer.
    pub lens_spacing: f32,
}

impl Default for ViewSettings {
    fn default() -> Self {
        ViewSettings {
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            zoom: 1.0,
            height: 0.0,
            forward: 0.0,
            ipd_offset: 0.0,
            vertical_align: 0.0,
            rotation_align: 0.0,
            lens_k1: 0.0,
            lens_k2: 0.0,
            lens_centre: [[0.0; 2]; 2],
            swap_eyes: false,
            brightness: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            gamma: 1.0,
            sharpen: 0.0,
            screen_distance: 4.0,
            screen_width: 6.0,
            screen_curvature: 0.0,
            background: BACKGROUND,
            subtitle_distance: 2.5,
            chroma_key: false,
            key_color: [0.08, 0.04, 1.0],
            key_similarity: 0.4,
            key_smoothness: 0.08,
            key_spill: 0.1,
            key_own: Some(false),
            focus: Focus::Off,
            focus_distance: FOCUS_DISTANCE,
            focus_speed: 1.0,
            focus_nearest: FOCUS_NEAREST,
            focus_farthest: *FOCUS_DISTANCES.end(),
            lens_spacing: LENS_SPACING,
        }
    }
}

impl ViewSettings {
    /// Linear blend, used for keyframed settings.
    pub fn lerp(&self, other: &ViewSettings, t: f32) -> ViewSettings {
        let t = t.clamp(0.0, 1.0);
        let l = |a: f32, b: f32| a + (b - a) * t;
        ViewSettings {
            yaw: l(self.yaw, other.yaw),
            pitch: l(self.pitch, other.pitch),
            roll: l(self.roll, other.roll),
            zoom: l(self.zoom, other.zoom),
            height: l(self.height, other.height),
            forward: l(self.forward, other.forward),
            ipd_offset: l(self.ipd_offset, other.ipd_offset),
            vertical_align: l(self.vertical_align, other.vertical_align),
            rotation_align: l(self.rotation_align, other.rotation_align),
            lens_k1: l(self.lens_k1, other.lens_k1),
            lens_k2: l(self.lens_k2, other.lens_k2),
            lens_centre: [0, 1]
                .map(|i| [0, 1].map(|j| l(self.lens_centre[i][j], other.lens_centre[i][j]))),
            swap_eyes: if t < 0.5 {
                self.swap_eyes
            } else {
                other.swap_eyes
            },
            brightness: l(self.brightness, other.brightness),
            contrast: l(self.contrast, other.contrast),
            saturation: l(self.saturation, other.saturation),
            gamma: l(self.gamma, other.gamma),
            sharpen: l(self.sharpen, other.sharpen),
            screen_distance: l(self.screen_distance, other.screen_distance),
            screen_width: l(self.screen_width, other.screen_width),
            screen_curvature: l(self.screen_curvature, other.screen_curvature),
            background: [0, 1, 2].map(|i| l(self.background[i], other.background[i])),
            subtitle_distance: l(self.subtitle_distance, other.subtitle_distance),
            chroma_key: if t < 0.5 {
                self.chroma_key
            } else {
                other.chroma_key
            },
            key_color: [
                l(self.key_color[0], other.key_color[0]),
                l(self.key_color[1], other.key_color[1]),
                l(self.key_color[2], other.key_color[2]),
            ],
            key_similarity: l(self.key_similarity, other.key_similarity),
            key_smoothness: l(self.key_smoothness, other.key_smoothness),
            key_spill: l(self.key_spill, other.key_spill),
            key_own: if t < 0.5 { self.key_own } else { other.key_own },
            focus: if t < 0.5 { self.focus } else { other.focus },
            focus_distance: lerp_distance(self.focus_distance, other.focus_distance, t),
            // A multiplier: even steps in its logarithm.
            focus_speed: (self.focus_speed.max(0.01).ln()
                + (other.focus_speed.max(0.01).ln() - self.focus_speed.max(0.01).ln())
                    * t.clamp(0.0, 1.0))
            .exp(),
            focus_nearest: lerp_distance(self.focus_nearest, other.focus_nearest, t),
            focus_farthest: lerp_distance(self.focus_farthest, other.focus_farthest, t),
            // A spacing between two cameras' means nothing.
            lens_spacing: if t < 0.5 {
                self.lens_spacing
            } else {
                other.lens_spacing
            },
        }
    }

    /// Whether the chroma key is this video's own (see [`Self::key_own`]).
    pub fn own_key(&self) -> bool {
        self.key_own.unwrap_or(self.chroma_key)
    }

    /// The chroma key settings: on, colour, similarity, smoothness, spill.
    pub fn key(&self) -> (bool, [f32; 3], f32, f32, f32) {
        (
            self.chroma_key,
            self.key_color,
            self.key_similarity,
            self.key_smoothness,
            self.key_spill,
        )
    }

    /// Copies the chroma key settings of `from`.
    pub fn set_key(&mut self, from: &ViewSettings) {
        (
            self.chroma_key,
            self.key_color,
            self.key_similarity,
            self.key_smoothness,
            self.key_spill,
        ) = from.key();
    }

    /// These settings with `global`'s chroma key, unless the video has its own.
    pub fn with_global_key(mut self, global: &ViewSettings) -> ViewSettings {
        if !self.own_key() {
            self.set_key(global);
        }
        self
    }
}

/// `a` to `b` metres at `t` (0..1), in even steps of how far the eyes
/// turn in to look at it (1/distance), not of metres: from 0.5 m to 20 m,
/// halfway is 0.98 m. Distances under [`FOCUS_DISTANCES`]' nearest count as
/// it, so a zero can't divide.
pub fn lerp_distance(a: f32, b: f32, t: f32) -> f32 {
    let near = *FOCUS_DISTANCES.start();
    let (ia, ib) = (1.0 / a.max(near), 1.0 / b.max(near));
    1.0 / (ia + (ib - ia) * t.clamp(0.0, 1.0))
}

/// A named set of adjustments any video can take on: everything saved for a
/// video but its keyframes, which belong to its timeline.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Preset {
    pub name: String,
    /// The format picked for the video it was saved from; none when that
    /// video was on Automatic, so the format stays the detected one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<VideoFormat>,
    pub view: ViewSettings,
}

impl Preset {
    /// The adjustments it gives a video now showing `current`: all of its
    /// own, but a flat screen stays where it was moved to (its yaw and
    /// pitch), as that isn't saved for a video either.
    pub fn view_for(&self, current: &ViewSettings, flat: bool) -> ViewSettings {
        let mut view = self.view;
        if flat {
            view.yaw = current.yaw;
            view.pitch = current.pitch;
        }
        view
    }

    /// Whether a video showing `current`, with `picked` its picked format
    /// (none on Automatic), has everything this preset gives it.
    pub fn matches(&self, current: &ViewSettings, picked: Option<VideoFormat>, flat: bool) -> bool {
        self.view_for(current, flat) == *current && self.format.is_none_or(|f| picked == Some(f))
    }
}

/// Settings that change over the video's timeline.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Keyframes {
    /// (time in seconds, settings), sorted by time.
    pub frames: Vec<(f64, ViewSettings)>,
}

impl Keyframes {
    pub fn insert(&mut self, time: f64, settings: ViewSettings) {
        match self
            .frames
            .iter()
            .position(|(t, _)| (*t - time).abs() < 1e-3)
        {
            Some(i) => self.frames[i].1 = settings,
            None => {
                self.frames.push((time, settings));
                self.frames.sort_by(|a, b| a.0.total_cmp(&b.0));
            }
        }
    }

    /// Settings at `time`, interpolated; `base` when there are no keyframes.
    pub fn at(&self, time: f64, base: &ViewSettings) -> ViewSettings {
        match self.frames.len() {
            0 => *base,
            _ => {
                let i = self.frames.partition_point(|(t, _)| *t <= time);
                if i == 0 {
                    self.frames[0].1
                } else if i == self.frames.len() {
                    self.frames[i - 1].1
                } else {
                    let (t0, a) = &self.frames[i - 1];
                    let (t1, b) = &self.frames[i];
                    a.lerp(b, ((time - t0) / (t1 - t0)) as f32)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_preset_gives_everything_but_a_flat_screens_place() {
        let saved = ViewSettings {
            yaw: 30.0,
            zoom: 1.4,
            brightness: -0.2,
            lens_k1: 0.1,
            chroma_key: true,
            ..ViewSettings::default()
        };
        let sbs = VideoFormat::new(
            crate::Projection::EQUIRECT_180,
            crate::StereoLayout::SideBySide,
        );
        let preset = Preset {
            name: "Night".into(),
            format: Some(sbs),
            view: saved,
        };
        let moved = ViewSettings {
            yaw: -90.0,
            pitch: 10.0,
            ..ViewSettings::default()
        };
        // A 180° video takes all of it, turn included.
        let vr = preset.view_for(&moved, false);
        assert_eq!(vr, saved);
        assert!(preset.matches(&vr, Some(sbs), false));
        assert!(
            !preset.matches(&vr, None, false),
            "its format is part of it"
        );
        // A flat screen keeps where it was moved to.
        let flat = preset.view_for(&moved, true);
        assert_eq!((flat.yaw, flat.pitch, flat.zoom), (-90.0, 10.0, 1.4));
        assert!(flat.chroma_key && flat.lens_k1 == 0.1);
        // Saved on Automatic, it matches whatever the format is.
        let automatic = Preset {
            format: None,
            ..preset.clone()
        };
        assert!(automatic.matches(&vr, None, false));
        // A preset saved before a field existed reads back with its default.
        let old: Preset = serde_json::from_str(r#"{"name":"Old","view":{"zoom":2.0}}"#).unwrap();
        assert_eq!(
            (old.view.zoom, old.view.contrast, old.format),
            (2.0, 1.0, None)
        );
    }

    #[test]
    fn keyframes_interpolate_and_clamp() {
        let base = ViewSettings::default();
        let mut k = Keyframes::default();
        assert_eq!(k.at(5.0, &base), base);
        k.insert(10.0, ViewSettings { zoom: 2.0, ..base });
        k.insert(0.0, ViewSettings { zoom: 1.0, ..base });
        assert_eq!(k.frames[0].0, 0.0);
        assert!((k.at(5.0, &base).zoom - 1.5).abs() < 1e-6);
        assert_eq!(k.at(-1.0, &base).zoom, 1.0);
        assert_eq!(k.at(99.0, &base).zoom, 2.0);
        k.insert(10.0, ViewSettings { zoom: 3.0, ..base });
        assert_eq!(k.frames.len(), 2);
        assert_eq!(k.at(10.0, &base).zoom, 3.0);
    }

    #[test]
    fn stereo_focus_keyframes_glide_in_how_far_the_eyes_turn() {
        let near = ViewSettings {
            focus: Focus::Manual,
            focus_distance: 0.5,
            lens_spacing: 60.0,
            ..ViewSettings::default()
        };
        let far = ViewSettings {
            focus: Focus::Off,
            focus_distance: 20.0,
            lens_spacing: 65.0,
            ..ViewSettings::default()
        };
        let mid = near.lerp(&far, 0.5);
        // Halfway in 1/distance: 1 / ((2 + 0.05) / 2).
        assert!((mid.focus_distance - 1.0 / 1.025).abs() < 1e-5);
        assert!((near.lerp(&far, 0.25).focus_distance - 1.0 / 1.5125).abs() < 1e-5);
        assert_eq!(near.lerp(&far, 0.0).focus_distance, 0.5);
        assert!((near.lerp(&far, 1.0).focus_distance - 20.0).abs() < 1e-3);
        // The mode and the lens spacing switch halfway, never in between.
        for t in [0.0, 0.1, 0.49] {
            let s = near.lerp(&far, t);
            assert_eq!((s.focus, s.lens_spacing), (Focus::Manual, 60.0), "{t}");
        }
        for t in [0.5, 0.9, 1.0] {
            let s = near.lerp(&far, t);
            assert_eq!((s.focus, s.lens_spacing), (Focus::Off, 65.0), "{t}");
        }
        // A zero or broken distance can't divide by zero.
        let zero = ViewSettings {
            focus_distance: 0.0,
            ..near
        };
        let d = zero.lerp(&far, 0.0).focus_distance;
        assert_eq!(d, *FOCUS_DISTANCES.start());
        let nan = ViewSettings {
            focus_distance: f32::NAN,
            ..near
        };
        assert!(nan.lerp(&far, 0.5).focus_distance.is_finite());
    }

    #[test]
    fn automatic_focus_settings_glide_as_the_distance_does() {
        let a = ViewSettings {
            focus: Focus::Auto,
            focus_speed: 0.5,
            focus_nearest: 0.4,
            focus_farthest: 20.0,
            ..ViewSettings::default()
        };
        let b = ViewSettings {
            focus_speed: 2.0,
            focus_nearest: 1.0,
            focus_farthest: 4.0,
            ..a
        };
        let mid = a.lerp(&b, 0.5);
        assert!((mid.focus_speed - 1.0).abs() < 1e-5, "{}", mid.focus_speed);
        assert!((mid.focus_nearest - 1.0 / 1.75).abs() < 1e-5);
        assert!((mid.focus_farthest - 1.0 / 0.15).abs() < 1e-3);
        let old: ViewSettings = serde_json::from_str(r#"{"focus":"manual"}"#).unwrap();
        assert_eq!(
            (old.focus_speed, old.focus_nearest, old.focus_farthest),
            (1.0, FOCUS_NEAREST, 20.0)
        );
        let json = serde_json::to_string(&b).unwrap();
        assert!(json.contains(r#""focus":"auto""#), "{json}");
        assert_eq!(serde_json::from_str::<ViewSettings>(&json).unwrap(), b);
    }

    #[test]
    fn stereo_focus_loads_off_from_old_settings_and_round_trips() {
        let old: ViewSettings = serde_json::from_str(r#"{"zoom":1.2,"swap_eyes":true}"#).unwrap();
        assert_eq!(
            (old.focus, old.focus_distance, old.lens_spacing),
            (Focus::Off, FOCUS_DISTANCE, LENS_SPACING)
        );
        let on = ViewSettings {
            focus: Focus::Manual,
            focus_distance: 0.8,
            lens_spacing: 60.0,
            ..ViewSettings::default()
        };
        let json = serde_json::to_string(&on).unwrap();
        assert!(json.contains(r#""focus":"manual""#), "{json}");
        assert_eq!(serde_json::from_str::<ViewSettings>(&json).unwrap(), on);
    }

    #[test]
    fn partial_json_uses_defaults() {
        let v: ViewSettings = serde_json::from_str(r#"{"zoom":1.2}"#).unwrap();
        assert_eq!(v.zoom, 1.2);
        assert_eq!(v.contrast, 1.0);
        assert_eq!(v.background, BACKGROUND, "saved before it: today's colour");
    }

    #[test]
    fn videos_follow_the_global_key_unless_they_have_their_own() {
        let global = ViewSettings {
            chroma_key: true,
            key_color: [0.0, 0.0, 1.0],
            ..Default::default()
        };
        let video = ViewSettings::default();
        let shown = video.with_global_key(&global);
        assert!(shown.chroma_key);
        assert_eq!(shown.key_color, [0.0, 0.0, 1.0]);

        let own = ViewSettings {
            key_own: Some(true),
            ..video
        };
        assert!(!own.with_global_key(&global).chroma_key);

        // Saved before the choice existed: a keyed video keeps its key, an
        // unkeyed one follows the global setting.
        let old: ViewSettings =
            serde_json::from_str(r#"{"chroma_key":true,"key_color":[1,0,1]}"#).unwrap();
        assert_eq!(old.with_global_key(&global).key_color, [1.0, 0.0, 1.0]);
        let old: ViewSettings = serde_json::from_str(r#"{"zoom":1.2}"#).unwrap();
        assert!(old.with_global_key(&global).chroma_key);
    }
}
