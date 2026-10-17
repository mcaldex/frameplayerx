//! What the headset holds: the videos indexed from its folders, microSD
//! cards and USB drives (probed, so every filter and order works), and the
//! folders themselves.

use super::page;
use super::theme::{self, Weight};
use super::widgets::{self, Kind, Tip};
use super::{Action, Browse, LibraryGrid, View, icons};
use egui::{Align, Layout, RichText};
use fp_library::Sort;

pub struct LocalView {
    /// Showing the folders rather than the videos.
    pub folders: bool,
    pub grid: LibraryGrid,
    /// The folder being browsed in the folders tab.
    pub folder: Option<Browse>,
}

impl Default for LocalView {
    fn default() -> Self {
        LocalView {
            folders: false,
            grid: LibraryGrid::local(),
            folder: None,
        }
    }
}

/// The headset's videos or folders; true when Back was clicked.
pub fn page(ui: &mut egui::Ui, v: &mut View, l: &mut LocalView) -> bool {
    let title = super::sources::LOCAL_NAME;
    if let Some(b) = l.folder.as_mut() {
        let root = crate::services::builtin_sources()
            .into_iter()
            .find(|c| c.id() == b.source)
            .map(|c| c.name().to_string())
            .unwrap_or_default();
        return super::folders::page(ui, v, title, &root, b);
    }
    // The scan while it runs, else what the page holds.
    let subtitle = match v.state.scan_status.as_deref() {
        Some(s) if s.starts_with("Scanning") => s.to_string(),
        _ if l.folders => "Folders".to_string(),
        _ => "Videos on the headset, its microSD card and USB drives".to_string(),
    };
    let back = page::header(ui, title, Some(&subtitle), |ui| {
        if widgets::icon_button(ui, icons::GEAR_SIX, 44.0, false)
            .tip("Folders and drives searched for videos")
            .clicked()
        {
            v.state.screen = super::Screen::Settings;
            v.state.settings_tab = super::SettingsTab::Device;
        }
        if widgets::button_sized(
            ui,
            Some(icons::ARROWS_CLOCKWISE),
            "Scan now",
            Kind::Secondary,
            44.0,
        )
        .clicked()
        {
            v.actions.push(Action::Rescan { force: false });
        }
    });
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 8.0;
        if widgets::chip_icon(ui, Some(icons::FILM_STRIP), "Videos", !l.folders).clicked() {
            l.folders = false;
        }
        if widgets::chip_icon(ui, Some(icons::FOLDERS), "Folders", l.folders).clicked() {
            l.folders = true;
        }
    });
    ui.add_space(8.0);
    if l.folders {
        roots(ui, v, l);
    } else {
        super::library::grid_page(
            ui,
            v,
            &mut l.grid,
            &[
                (Sort::Added, "Newest"),
                (Sort::Title, "Title"),
                (Sort::Duration, "Length"),
                (Sort::Rating, "Rating"),
                (Sort::Random, "Shuffle"),
            ],
            |ui, v| {
                let folders: Vec<String> = v
                    .settings
                    .library_folders
                    .iter()
                    .map(|f| f.display().to_string())
                    .collect();
                let body = if folders.is_empty() {
                    "Copy videos to the Frame with FrameDrop or a USB drive, then scan.".to_string()
                } else {
                    format!(
                        "Videos in {} and on microSD cards and USB drives are added \
                         automatically. Copy some over with FrameDrop or a USB drive.",
                        folders.join(", ")
                    )
                };
                page::empty_state(ui, icons::FILM_STRIP, "No videos here yet", &body, |ui| {
                    if widgets::button(ui, Some(icons::ARROWS_CLOCKWISE), "Scan now", Kind::Primary)
                        .clicked()
                    {
                        v.actions.push(Action::Rescan { force: false });
                    }
                });
            },
        );
    }
    back
}

/// The places to browse: the home folder and the removable drives.
fn roots(ui: &mut egui::Ui, v: &mut View, l: &mut LocalView) {
    let mounts = crate::drives::removable_mounts();
    let drives = if mounts.is_empty() {
        "None inserted".to_string()
    } else {
        mounts
            .iter()
            .filter_map(|m| m.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let builtin = crate::services::builtin_sources();
    egui::ScrollArea::vertical()
        .auto_shrink(false)
        .show(ui, |ui| {
            widgets::card(ui, |ui| {
                for (i, c) in builtin.iter().enumerate() {
                    if i > 0 {
                        let x = ui.max_rect().x_range();
                        ui.painter().hline(
                            (x.min + 84.0)..=(x.max - 16.0),
                            ui.cursor().top(),
                            egui::Stroke::new(1.0_f32, theme::STROKE),
                        );
                    }
                    let removable = c.id() == crate::services::REMOVABLE_SOURCE;
                    let (icon, desc) = if removable {
                        (icons::USB, drives.clone())
                    } else {
                        (icons::HARD_DRIVE, c.address())
                    };
                    let mut open = false;
                    egui::Frame::new()
                        .inner_margin(egui::Margin::symmetric(16, 12))
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing.x = 16.0;
                                super::sources::icon_tile(ui, icon, 52.0, theme::ACCENT_HOVER);
                                ui.vertical(|ui| {
                                    ui.spacing_mut().item_spacing.y = 2.0;
                                    ui.label(
                                        RichText::new(c.name())
                                            .font(theme::font(Weight::SemiBold, 19.0))
                                            .color(theme::TEXT),
                                    );
                                    ui.label(
                                        RichText::new(&desc)
                                            .font(theme::font(Weight::Regular, 15.0))
                                            .color(theme::TEXT_2),
                                    );
                                });
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    open = widgets::button_sized(
                                        ui,
                                        Some(icons::FOLDER_OPEN),
                                        "Browse",
                                        Kind::Secondary,
                                        44.0,
                                    )
                                    .clicked();
                                });
                            });
                        });
                    if open {
                        l.folder = Some(Browse {
                            source: c.id().to_string(),
                            loading: true,
                            ..Default::default()
                        });
                        v.actions.push(Action::Browse {
                            source: c.id().to_string(),
                            location: None,
                        });
                    }
                }
            });
        });
}
