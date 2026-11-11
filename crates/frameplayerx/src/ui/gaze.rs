//! Eye tracking: its group in Settings › General, and the gaze dot's
//! picture.

use super::{View, theme, widgets};
use crate::gaze::GazeStatus;
use egui::{Color32, RichText, Stroke};

/// The gaze dot's picture, pixels square, and the radius of its ring in
/// it ([`crate::gaze::DOT_RING`]).
pub const DOT_PX: u32 = 96;
pub const DOT_RING_PX: f32 = 40.0;

/// The gaze dot: a small white dot in a thin white ring, each with a dark
/// rim, so it reads on bright pictures and dark ones.
pub fn dot_picture(ctx: &egui::Context) {
    egui::CentralPanel::default()
        .frame(egui::Frame::NONE)
        .show(ctx, |ui| {
            let p = ui.painter();
            let c = ui.max_rect().center();
            let dark = Color32::from_black_alpha(170);
            p.circle_stroke(c, DOT_RING_PX, Stroke::new(5.0_f32, dark));
            p.circle_stroke(
                c,
                DOT_RING_PX,
                Stroke::new(2.0_f32, Color32::from_white_alpha(230)),
            );
            p.circle_filled(c, 8.0, dark);
            p.circle_filled(c, 5.5, Color32::WHITE);
        });
}

/// Eye tracking's group: whether the tracker sends where you look (with
/// a hint when it doesn't), and the gaze dot. Only where the headset
/// tracks the eyes.
pub fn group(ui: &mut egui::Ui, v: &mut View) {
    let Some(status) = v.gaze.filter(|s| *s != GazeStatus::NoEyeTracking) else {
        return;
    };
    ui.add_space(12.0);
    widgets::section_label(ui, tr!("settings-eye-tracking"));
    let s = &mut *v.settings;
    widgets::rows(ui, |r| {
        let delivering = status != GazeStatus::NotDelivering;
        r.info(
            tr!("settings-eye-tracker"),
            if delivering {
                tr!("settings-eye-tracker-on")
            } else {
                tr!("settings-eye-tracker-off")
            },
        );
        if !delivering {
            r.content(|ui| {
                ui.label(RichText::new(tr!("settings-eye-tracker-hint")).color(theme::TEXT_2));
            });
        }
        r.switch(
            tr!("settings-gaze-dot"),
            Some(tr!("settings-gaze-dot-desc")),
            &mut s.gaze_dot,
        );
    });
}
