//! OpenSubtitles search, results and account forms.

use super::{View, icons, page, sources, theme, widgets};
use egui::RichText;
use fp_sources::opensubtitles::{
    ContentFilter, Error, Language, MatchKind, MediaKind, Quota, SdhFilter, SearchQuery, Subtitle,
};
use widgets::{Kind, ListArea};

#[derive(Clone, Debug)]
pub enum Action {
    Open,
    Close,
    Submit,
    TitleOnly,
    RestoreIdentification,
    Use(u64),
    SignIn,
    SignOut,
    Remove,
    Delay(Option<i32>),
}

#[derive(Default)]
pub struct Account {
    pub available: bool,
    pub signed_in: bool,
    pub rejected: bool,
    pub quota: Quota,
    pub busy: bool,
    pub error: Option<Error>,
    pub username: String,
    pub password: String,
}

pub struct Search {
    pub query: SearchQuery,
    pub submitted: Option<SearchQuery>,
    pub manual: bool,
    pub dirty: bool,
    pub languages: Vec<Language>,
    pub results: Vec<ResultFile>,
    pub selected: Option<u64>,
    pub busy: bool,
    pub downloading: Option<u64>,
    pub error: Option<Error>,
    refine_open: bool,
    pub(crate) account_open: bool,
    /// Whether the selected file is already on the headset, for that file
    /// and that many of the video's subtitles: found once, not each frame.
    kept: Option<(u64, usize, bool)>,
}

impl Search {
    pub fn new(title: String, language: String) -> Self {
        Self {
            query: SearchQuery {
                title,
                language: Some(language).filter(|s| !s.is_empty()),
                ..Default::default()
            },
            submitted: None,
            manual: false,
            dirty: false,
            languages: Vec::new(),
            results: Vec::new(),
            selected: None,
            busy: false,
            downloading: None,
            error: None,
            refine_open: false,
            account_open: false,
            kept: None,
        }
    }

    pub fn set_results(&mut self, subtitles: Vec<Subtitle>) {
        self.results.clear();
        for sub in subtitles {
            let language_name = display_language(&self.languages, &sub.language);
            // OpenSubtitles numbers a lone file as part 1 too.
            let parts = sub.files.len() > 1;
            for file in sub.files {
                if self.results.iter().any(|existing| existing.id == file.id) {
                    continue;
                }
                self.results.push(ResultFile {
                    id: file.id,
                    name: file.name,
                    part: file.part.filter(|_| parts),
                    release: sub.release.clone(),
                    fps: sub.fps,
                    language: sub.language.clone(),
                    language_name: language_name.clone(),
                    match_kind: sub.match_kind,
                    sdh: sub.sdh,
                    forced: sub.forced,
                    machine_translated: sub.machine_translated,
                    ai_translated: sub.ai_translated,
                });
            }
        }
    }
}

#[derive(Clone)]
pub struct ResultFile {
    pub id: u64,
    pub name: String,
    pub part: Option<u32>,
    pub release: String,
    pub fps: Option<f64>,
    pub language: String,
    pub language_name: String,
    pub match_kind: MatchKind,
    pub sdh: bool,
    pub forced: bool,
    pub machine_translated: bool,
    pub ai_translated: bool,
}

pub fn panel(ctx: &egui::Context, v: &mut View) {
    let Some(search) = v.state.subtitle_search.as_mut() else {
        return;
    };
    let account = &mut v.state.subtitle_account;
    let actions = &mut v.actions;
    let body = ctx.screen_rect() - widgets::SHADOW_ROOM;
    widgets::slab(ctx, body, 24.0);
    if ctx.input(|i| {
        i.pointer.any_click() && i.pointer.interact_pos().is_some_and(|p| !body.contains(p))
    }) {
        actions.push(super::Action::SubtitleSearch(Action::Close));
    }
    let attachments = v
        .playback
        .as_ref()
        .map_or(&[][..], |p| p.managed_subtitles.as_slice());
    egui::CentralPanel::default()
        .frame(egui::Frame::NONE)
        .show(ctx, |ui| {
            let mut ui = ui.new_child(egui::UiBuilder::new().max_rect(body.shrink(20.0)));
            let mut dispatch = |action| actions.push(super::Action::SubtitleSearch(action));
            let title = if search.account_open {
                "OpenSubtitles"
            } else {
                tr!("subtitle-find")
            };
            if page::header(&mut ui, title, None, |ui| {
                if !search.account_open && account.available {
                    let label = if account.signed_in {
                        tr!("subtitle-account")
                    } else {
                        tr!("subtitle-sign-in")
                    };
                    if widgets::button(ui, None, label, Kind::Ghost).clicked() {
                        search.account_open = true;
                    }
                }
            }) {
                dispatch(Action::Close);
            }
            if search.account_open {
                egui::ScrollArea::vertical()
                    .id_salt("opensubtitles-account")
                    .show_list(&mut ui, |ui| {
                        account_form(ui, account, &mut dispatch);
                    });
                return;
            }
            // Not while a search runs or a file downloads, whose answers
            // would be for what was asked before.
            let editable = account.available && !search.busy && search.downloading.is_none();
            ui.add_enabled_ui(editable, |ui| {
                ui.horizontal(|ui| {
                    let edited = page::search_field(
                        ui,
                        &mut search.query.title,
                        tr!("subtitle-title"),
                        150.0,
                    );
                    search.manual |= edited;
                    search.dirty |= edited;
                    language_menu(ui, search);
                    if widgets::button(ui, None, tr!("jf-search"), Kind::Primary).clicked() {
                        dispatch(Action::Submit);
                    }
                });
            });
            ui.horizontal_wrapped(|ui| {
                ui.add_enabled_ui(editable, |ui| {
                    if widgets::button(ui, None, tr!("subtitle-refine"), Kind::Ghost).clicked() {
                        search.refine_open = !search.refine_open;
                    }
                });
                if account.available {
                    ui.label(RichText::new(quota_text(&account.quota)).color(theme::TEXT_3));
                }
            });
            ui.add_enabled_ui(editable, |ui| {
                if search.refine_open {
                    refine(ui, search);
                    if search.manual
                        && widgets::button(
                            ui,
                            None,
                            tr!("subtitle-restore-identification"),
                            Kind::Ghost,
                        )
                        .clicked()
                    {
                        dispatch(Action::RestoreIdentification);
                    }
                }
            });
            if let Some(error) = &search.error {
                ui.label(RichText::new(error_text(error)).color(theme::WARN));
            }
            if search.busy {
                ui.label(tr!("common-loading"));
            } else if !search.dirty
                && search.submitted.is_some()
                && search.results.is_empty()
                && search.error.is_none()
            {
                ui.label(tr!("subtitle-no-results"));
                if search.submitted.as_ref().is_some_and(identified)
                    && widgets::button(&mut ui, None, tr!("subtitle-search-title"), Kind::Secondary)
                        .clicked()
                {
                    dispatch(Action::TitleOnly);
                }
            }
            if search.results.is_empty() || search.dirty {
                return;
            }
            let row_height = (ui.available_height() - 56.0).max(50.0);
            egui::ScrollArea::vertical()
                .id_salt("opensubtitles-results")
                .max_height(row_height)
                .auto_shrink([false, false])
                .show_list(&mut ui, |ui| {
                    for file in &search.results {
                        let on = search.selected == Some(file.id);
                        ui.push_id(file.id, |ui| {
                            let row = super::player::track_row(
                                ui,
                                &file_title(file),
                                &file_details(file),
                                on,
                            )
                            .on_hover_text(format!("{}\n{}", file.name, file.release));
                            if row.clicked() && search.downloading.is_none() {
                                search.selected = Some(file.id);
                            }
                        });
                    }
                });
            if let Some(id) = search.selected {
                let cached = match search.kept {
                    Some((kept, count, cached)) if kept == id && count == attachments.len() => {
                        cached
                    }
                    _ => {
                        let cached = attachments.iter().any(|s| {
                            is_download(s, id)
                                && fp_library::SubtitleStore::default()
                                    .open(&s.location())
                                    .is_ok()
                        });
                        search.kept = Some((id, attachments.len(), cached));
                        cached
                    }
                };
                let label = if search.downloading == Some(id) {
                    tr!("subtitle-downloading")
                } else if cached {
                    tr!("subtitle-use")
                } else {
                    tr!("subtitle-download-use")
                };
                ui.horizontal(|ui| {
                    ui.add_enabled_ui(search.downloading.is_none(), |ui| {
                        if widgets::button(ui, Some(icons::CHECK), label, Kind::Primary).clicked() {
                            dispatch(Action::Use(id));
                        }
                    });
                    if search.downloading.is_some()
                        && widgets::button(ui, None, tr!("common-cancel"), Kind::Ghost).clicked()
                    {
                        dispatch(Action::Close);
                    }
                });
            }
        });
}

pub fn is_download(s: &fp_library::UserSubtitle, id: u64) -> bool {
    s.provider == "opensubtitles"
        && u64::try_from(s.provider_file_id) == Ok(id)
        && s.download_parameters == fp_sources::opensubtitles::DOWNLOAD_PARAMETERS
}

fn identified(query: &SearchQuery) -> bool {
    query.file_name.is_some()
        || query.moviehash.is_some()
        || query.imdb_id.is_some()
        || query.tmdb_id.is_some()
        || query.parent_imdb_id.is_some()
        || query.parent_tmdb_id.is_some()
}

fn display_language(languages: &[Language], code: &str) -> String {
    if let Some(l) = languages.iter().find(|l| l.code == code) {
        return l.name.clone();
    }
    if code == "pt-br" {
        return tr!("subtitle-language-pt-br").into();
    }
    super::language_name(fp_sources::opensubtitles::app_language(code))
        .unwrap_or_else(|| code.into())
}

fn language_menu(ui: &mut egui::Ui, search: &mut Search) {
    let label = search.query.language.as_ref().map_or_else(
        || tr!("subtitle-any-language").to_string(),
        |code| display_language(&search.languages, code),
    );
    let previous = search.query.language.clone();
    egui::ComboBox::from_id_salt("opensubtitles-language")
        .selected_text(label)
        .width(130.0)
        .height(240.0)
        .show_ui(ui, |ui| {
            ui.selectable_value(
                &mut search.query.language,
                None,
                tr!("subtitle-any-language"),
            );
            if search.languages.is_empty() {
                for (code, _, _) in fp_core::language::LANGUAGES {
                    ui.selectable_value(
                        &mut search.query.language,
                        Some(fp_sources::opensubtitles::language_code(code).into()),
                        display_language(&[], code),
                    );
                }
                ui.selectable_value(
                    &mut search.query.language,
                    Some("pt-br".into()),
                    tr!("subtitle-language-pt-br"),
                );
            } else {
                for language in &search.languages {
                    ui.selectable_value(
                        &mut search.query.language,
                        Some(language.code.clone()),
                        &language.name,
                    );
                }
            }
        });
    search.dirty |= search.query.language != previous;
}

fn refine(ui: &mut egui::Ui, search: &mut Search) {
    let query = &mut search.query;
    let mut identification_edited = false;
    ui.horizontal_wrapped(|ui| {
        for (label, kind) in [
            (tr!("subtitle-movie"), Some(MediaKind::Movie)),
            (tr!("subtitle-episode"), Some(MediaKind::Episode)),
        ] {
            if ui.selectable_value(&mut query.kind, kind, label).changed() {
                identification_edited = true;
            }
        }
        identification_edited |= number(ui, tr!("subtitle-year"), &mut query.year, 9999);
        if query.kind == Some(MediaKind::Episode) {
            identification_edited |= number(ui, tr!("subtitle-season"), &mut query.season, 999);
            identification_edited |= number(ui, tr!("subtitle-episode"), &mut query.episode, 9999);
        }
    });
    search.manual |= identification_edited;
    search.dirty |= identification_edited;
    let filters = (query.sdh, query.content, query.include_translations);
    ui.horizontal_wrapped(|ui| {
        choice(
            ui,
            "SDH",
            &mut query.sdh,
            [
                (tr!("subtitle-any"), SdhFilter::Any),
                (tr!("subtitle-only"), SdhFilter::Only),
                (tr!("subtitle-exclude"), SdhFilter::Exclude),
            ],
        );
        choice(
            ui,
            tr!("settings-subtitles"),
            &mut query.content,
            [
                (tr!("subtitle-full"), ContentFilter::Full),
                (tr!("tracks-forced"), ContentFilter::Forced),
                (tr!("subtitle-any"), ContentFilter::Any),
            ],
        );
        ui.checkbox(
            &mut query.include_translations,
            tr!("subtitle-include-translations"),
        );
    });
    search.dirty |= filters != (query.sdh, query.content, query.include_translations);
}

fn choice<T: Copy + PartialEq>(
    ui: &mut egui::Ui,
    label: &str,
    state: &mut T,
    values: [(&str, T); 3],
) {
    let current = values
        .iter()
        .find(|(_, value)| value == state)
        .map_or("", |(label, _)| *label);
    ui.label(label);
    egui::ComboBox::from_id_salt(label)
        .selected_text(current)
        .show_ui(ui, |ui| {
            for (label, value) in values {
                ui.selectable_value(state, value, label);
            }
        });
}

fn number(ui: &mut egui::Ui, label: &str, field: &mut Option<u32>, limit: u32) -> bool {
    ui.label(label);
    let mut value = field.map_or(-1, |n| n as i32);
    let edited = ui
        .add(
            egui::DragValue::new(&mut value)
                .range(-1..=limit as i32)
                .custom_formatter(|n, _| {
                    if n < 0.0 {
                        String::from("-")
                    } else {
                        format!("{n:.0}")
                    }
                })
                .custom_parser(|s| if s == "-" { Some(-1.0) } else { s.parse().ok() }),
        )
        .changed();
    if edited {
        *field = u32::try_from(value).ok();
    }
    edited
}

fn file_title(file: &ResultFile) -> String {
    let name = if file.name.is_empty() {
        &file.release
    } else {
        &file.name
    };
    if let Some(part) = file.part.filter(|_| !name.is_empty()) {
        tr!("subtitle-file-part", part = part, name = name.as_str())
    } else {
        name.clone()
    }
}

fn file_details(file: &ResultFile) -> String {
    let status = match file.match_kind {
        MatchKind::File => tr!("subtitle-match-file"),
        MatchKind::Identity => tr!("subtitle-match-identity"),
        MatchKind::Title => tr!("subtitle-match-title"),
    };
    let mut parts = vec![file.language_name.clone(), status.into()];
    for (show, label) in [
        (file.sdh, "SDH"),
        (file.forced, tr!("tracks-forced")),
        (file.ai_translated, tr!("subtitle-ai-translated")),
        (
            file.machine_translated && !file.ai_translated,
            tr!("subtitle-machine-translated"),
        ),
    ] {
        if show {
            parts.push(label.into());
        }
    }
    if let Some(fps) = file.fps.filter(|fps| *fps > 0.0) {
        parts.push(format!("{fps:.3} fps"));
    }
    if !file.release.is_empty() && file.release != file.name {
        parts.push(file.release.clone());
    }
    parts.join(" · ")
}

pub fn settings(ui: &mut egui::Ui, v: &mut View) {
    widgets::section_label(ui, "OpenSubtitles");
    let mut dispatch = |action| v.actions.push(super::Action::SubtitleSearch(action));
    account_form(ui, &mut v.state.subtitle_account, &mut dispatch);
}

fn account_form(ui: &mut egui::Ui, account: &mut Account, dispatch: &mut impl FnMut(Action)) {
    if !account.available {
        ui.label(error_text(&Error::Unconfigured));
        return;
    }
    ui.label(quota_text(&account.quota));
    if account.signed_in {
        if !account.rejected {
            ui.label(tr!("subtitle-signed-in"));
        }
        if ui
            .add_enabled_ui(!account.busy, |ui| {
                widgets::button(ui, None, tr!("subtitle-sign-out"), Kind::Secondary)
            })
            .inner
            .clicked()
        {
            dispatch(Action::SignOut);
        }
    }
    if !account.signed_in || account.rejected {
        ui.label(tr!("subtitle-sign-in-note"));
        ui.add_enabled_ui(!account.busy, |ui| {
            sources::text_field(ui, &mut account.username, tr!("subtitle-username"), false);
            sources::text_field(ui, &mut account.password, tr!("source-password"), true);
            let ready = !account.username.trim().is_empty() && !account.password.is_empty();
            if ui
                .add_enabled_ui(ready, |ui| {
                    widgets::button(ui, None, tr!("subtitle-sign-in"), Kind::Primary)
                })
                .inner
                .clicked()
            {
                dispatch(Action::SignIn);
            }
        });
    }
    if account.busy {
        ui.label(tr!("common-loading"));
    }
    if let Some(error) = &account.error {
        ui.label(RichText::new(error_text(error)).color(theme::WARN));
    }
}

fn quota_text(quota: &Quota) -> String {
    let now = fp_sources::timeutil::system_time_to_unix(std::time::SystemTime::now());
    let reset = quota
        .reset_time_utc
        .as_deref()
        .and_then(fp_sources::timeutil::parse_iso8601);
    // Past its reset, the count is stale until the next request.
    if reset.is_some_and(|reset| reset <= now) {
        return tr!("subtitle-quota-unknown").into();
    }
    let minutes = reset.map(|reset| (reset - now + 59) / 60);
    match (quota.remaining, minutes) {
        (Some(count), Some(minutes)) => {
            let (hours, minutes): (i64, i64) = (minutes / 60, minutes % 60);
            tr!(
                "subtitle-quota-reset",
                count = count,
                hours = hours,
                minutes = minutes
            )
        }
        (Some(count), None) => tr!("subtitle-quota", count = count),
        _ => tr!("subtitle-quota-unknown").into(),
    }
}

pub fn error_text(error: &Error) -> String {
    match error {
        Error::Unconfigured => tr!("subtitle-error-unconfigured").into(),
        Error::InvalidSearch => tr!("subtitle-error-query").into(),
        Error::UnavailableLanguage => tr!("subtitle-error-language").into(),
        Error::CredentialsRejected => tr!("subtitle-error-credentials").into(),
        Error::AccountRequired => tr!("subtitle-error-account").into(),
        Error::AllowanceExhausted => tr!("subtitle-error-allowance").into(),
        Error::RateLimited { retry_after } => tr!(
            "subtitle-error-rate",
            seconds = retry_after.as_secs().max(1)
        ),
        Error::Unavailable | Error::InvalidResponse => tr!("subtitle-error-service").into(),
        Error::InvalidSubtitle => tr!("subtitle-error-invalid").into(),
        Error::Storage => tr!("subtitle-error-storage").into(),
        Error::Cancelled => tr!("common-cancel").into(),
    }
}

pub(super) fn timing(ui: &mut egui::Ui, delay: i32, actions: &mut Vec<super::Action>) {
    ui.label(
        RichText::new(tr!(
            "subtitle-timing",
            delay = format!("{:+.1}", f64::from(delay) / 1000.0)
        ))
        .color(theme::TEXT),
    );
    ui.horizontal_wrapped(|ui| {
        for (label, step, enabled) in [
            (tr!("subtitle-earlier"), Some(-100), delay > -10_000),
            (tr!("subtitle-later"), Some(100), delay < 10_000),
            (tr!("adjust-reset"), None, delay != 0),
        ] {
            if ui
                .add_enabled_ui(enabled, |ui| widgets::button(ui, None, label, Kind::Ghost))
                .inner
                .clicked()
            {
                actions.push(super::Action::SubtitleSearch(Action::Delay(step)));
            }
        }
    });
    if widgets::button(ui, None, tr!("subtitle-remove"), Kind::Ghost).clicked() {
        actions.push(super::Action::SubtitleSearch(Action::Remove));
    }
}
