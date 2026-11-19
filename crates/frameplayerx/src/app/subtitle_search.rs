//! OpenSubtitles jobs and their connection to the current video.

use super::{App, Job as AppJob, OpenRequest, playback};
use crate::subtitles::{self, Pick};
use crate::ui::subtitle_search::{Action, ResultFile, Search, is_download};
use fp_core::source::{Entry, EntryKind, FileSource, SubtitleFlags};
use fp_library::{
    MediaAttachmentTarget, MediaId, NewUserSubtitle, OriginalSubtitleSignature, SubtitleStore,
    UserSubtitle,
};
use fp_media::Cue;
use fp_sources::media_identity::{MediaIdentity, movie_hash};
use fp_sources::opensubtitles::{self as api, Client, Error, Language, SearchQuery, Subtitle};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub(super) struct Service {
    client: Arc<Client>,
    store: SubtitleStore,
    generation: u64,
    pub(super) selection: u64,
    pending_search: Option<Arc<AtomicBool>>,
    identification: Option<(u64, Identification)>,
    downloads: HashMap<(MediaId, u64), DownloadRequest>,
}

impl Service {
    pub(super) fn new() -> Self {
        Self {
            client: Arc::new(Client::configured(
                concat!("FramePlayerX v", env!("CARGO_PKG_VERSION")),
                fp_core::dirs::config_dir().join("opensubtitles-account.json"),
            )),
            store: SubtitleStore::default(),
            generation: 0,
            selection: 0,
            pending_search: None,
            identification: None,
            downloads: HashMap::new(),
        }
    }
}

#[derive(Clone, Default)]
pub(super) struct Identification {
    identity: MediaIdentity,
    signature: OriginalSubtitleSignature,
}

#[derive(Clone, Copy)]
struct DownloadRequest {
    open: u64,
    search: u64,
    selection: u64,
}

pub(super) enum Job {
    Results {
        open: u64,
        generation: u64,
        query: SearchQuery,
        identification: Identification,
        languages: Vec<Language>,
        result: Result<Vec<Subtitle>, Error>,
    },
    Downloaded {
        media: MediaId,
        file: u64,
        result: Result<(UserSubtitle, Vec<Cue>), Error>,
    },
    Account(Result<api::Quota, Error>),
}

impl App {
    pub(super) fn refresh_subtitle_account(&mut self) {
        let client = &self.subtitle_search.client;
        self.ui.subtitle_account.available = client.available();
        self.ui.subtitle_account.signed_in = client.signed_in();
        self.ui.subtitle_account.rejected = client.account_rejected();
        self.ui.subtitle_account.quota = client.quota();
    }

    pub(super) fn maintain_subtitle_downloads(&self) {
        let store = self.subtitle_search.store.clone();
        let library = self.services.library.clone();
        self.jobs.spawn("subtitle-cleanup", move || {
            if let Err(e) = store.cleanup(&library) {
                log::warn!("cleaning up subtitle downloads: {e}");
            }
            AppJob::Done
        });
    }

    pub(super) fn subtitle_search_action(&mut self, action: Action) {
        match action {
            Action::Open => self.open_subtitle_search(),
            Action::Close => self.back_from_subtitle_search(),
            Action::Submit => self.search_subtitles(),
            Action::TitleOnly => {
                if let Some(search) = &mut self.ui.subtitle_search {
                    search.manual = true;
                }
                self.search_subtitles();
            }
            Action::RestoreIdentification => {
                if let Some(search) = &mut self.ui.subtitle_search {
                    search.manual = false;
                    search.submitted = None;
                }
                self.search_subtitles();
            }
            Action::Use(file) => self.download_subtitle(file),
            Action::SignIn => self.subtitle_account_job(false),
            Action::SignOut => self.subtitle_account_job(true),
            Action::Remove => self.remove_subtitle_download(),
            Action::Delay(delta) => self.adjust_subtitle_delay(delta),
        }
        self.repaint_all();
    }

    fn open_subtitle_search(&mut self) {
        if self.ui.subtitle_search.is_some() {
            return;
        }
        let Some(p) = &self.playback else { return };
        if self.locked_source_of(&p.request).is_some() {
            return;
        }
        let title = subtitle_entry(&p.request, &p.title).name;
        let language = api::language_code(&self.settings.subtitle_language).to_string();
        self.refresh_subtitle_account();
        self.ui.subtitle_search = Some(Search::new(title, language));
        self.search_subtitles();
    }

    pub(super) fn back_from_subtitle_search(&mut self) {
        if let Some(search) = &mut self.ui.subtitle_search
            && search.account_open
        {
            search.account_open = false;
            self.ui.subtitle_account.password.clear();
            self.panels[super::TRACKS].push(super::escape());
            self.repaint_all();
        } else {
            self.close_subtitle_search();
        }
    }

    pub(super) fn close_subtitle_search(&mut self) {
        if let Some(token) = self.subtitle_search.pending_search.take() {
            token.store(true, Ordering::Relaxed);
        }
        if self.ui.subtitle_search.take().is_some() {
            self.subtitle_search.generation += 1;
            self.ui.subtitle_account.password.clear();
            self.panels[super::TRACKS].push(super::escape());
            self.repaint_all();
        }
    }

    fn search_subtitles(&mut self) {
        let Some(p) = &self.playback else { return };
        if self.locked_source_of(&p.request).is_some() {
            self.close_subtitle_search();
            return;
        }
        let Some(search) = &mut self.ui.subtitle_search else {
            return;
        };
        if search.busy {
            return;
        }
        if !self.subtitle_search.client.available() {
            search.error = Some(Error::Unconfigured);
            return;
        }
        let mut query = search.query.clone();
        let use_identification = !search.manual;
        let first = search.submitted.is_none() && use_identification;
        let discover_languages = search.languages.is_empty();
        let signature = p.subtitle_signature.clone();
        search.error = None;
        search.busy = true;
        search.dirty = false;
        search.results.clear();
        search.selected = None;
        if let Some(token) = self.subtitle_search.pending_search.take() {
            token.store(true, Ordering::Relaxed);
        }
        let cancel = Arc::new(AtomicBool::new(false));
        self.subtitle_search.pending_search = Some(cancel.clone());
        self.subtitle_search.generation += 1;
        search.downloading = None;
        let generation = self.subtitle_search.generation;
        let open = self.open_seq;
        let cached = self
            .subtitle_search
            .identification
            .as_ref()
            .filter(|(seq, _)| *seq == open)
            .map(|(_, info)| info.clone());
        let entry = subtitle_entry(&p.request, &p.title);
        let source = p
            .request
            .source_id
            .as_deref()
            .and_then(|id| self.services.opener.source(id));
        let direct = p.transcoding.is_none();
        let client = self.subtitle_search.client.clone();
        self.jobs.spawn("subtitle-search", move || {
            let cancelled = || cancel.load(Ordering::Relaxed);
            let identification =
                cached.unwrap_or_else(|| identify(entry, source, direct, signature, &cancel));
            clear_identification(&mut query);
            if use_identification {
                apply_identification(&mut query, &identification, first);
            }
            let result = client.search_cancellable(&query, cancelled);
            let languages = if discover_languages && !cancelled() {
                client.languages_cancellable(cancelled).unwrap_or_default()
            } else {
                Vec::new()
            };
            if !cancelled()
                && client.signed_in()
                && !client.account_rejected()
                && client.quota().remaining.is_none()
            {
                let _ = client.user_info_cancellable(cancelled);
            }
            AppJob::SubtitleSearch(Box::new(Job::Results {
                open,
                generation,
                query,
                identification,
                languages,
                result,
            }))
        });
    }

    fn download_subtitle(&mut self, file_id: u64) {
        let Some(p) = &self.playback else { return };
        if self.locked_source_of(&p.request).is_some() {
            self.close_subtitle_search();
            return;
        }
        let Some(search) = &mut self.ui.subtitle_search else {
            return;
        };
        if search.dirty || search.downloading.is_some() || search.selected != Some(file_id) {
            return;
        }
        let Some(file) = search.results.iter().find(|f| f.id == file_id).cloned() else {
            return;
        };
        if let Some(index) = p
            .managed_subtitles
            .iter()
            .find(|s| {
                is_download(s, file_id) && self.subtitle_search.store.open(&s.location()).is_ok()
            })
            .and_then(|s| {
                p.subtitle_files
                    .iter()
                    .position(|f| f.location == s.location())
            })
        {
            self.subtitle_search.selection += 1;
            search.downloading = Some(file_id);
            search.error = None;
            self.load_subtitle_file(index, Some(self.subtitle_search.generation));
            return;
        }
        let library = self.services.library.clone();
        // Only this explicit action can index a previously unindexed video.
        let Some(media) = p
            .record_id
            .or_else(|| super::actions::media_of(&library, &p.request))
        else {
            search.error = Some(Error::Storage);
            return;
        };
        let Ok(target) = library.subtitle_target(media) else {
            search.error = Some(Error::Storage);
            return;
        };
        if let Some(p) = &mut self.playback {
            p.record_id = Some(media);
        }
        self.subtitle_search.selection += 1;
        let request = DownloadRequest {
            open: self.open_seq,
            search: self.subtitle_search.generation,
            selection: self.subtitle_search.selection,
        };
        search.downloading = Some(file_id);
        search.error = None;
        if self
            .subtitle_search
            .downloads
            .insert((media, file_id), request)
            .is_some()
        {
            return;
        }
        let signature = self
            .subtitle_search
            .identification
            .as_ref()
            .filter(|(seq, _)| *seq == self.open_seq)
            .map(|(_, info)| info.signature.clone());
        let client = self.subtitle_search.client.clone();
        let store = self.subtitle_search.store.clone();
        self.jobs.spawn("subtitle-download", move || {
            let result = persist_download(&client, &store, &library, &target, file, signature);
            AppJob::SubtitleSearch(Box::new(Job::Downloaded {
                media,
                file: file_id,
                result,
            }))
        });
    }

    fn subtitle_account_job(&mut self, sign_out: bool) {
        let account = &mut self.ui.subtitle_account;
        if account.busy {
            return;
        }
        let username = account.username.trim().to_string();
        let password = std::mem::take(&mut account.password);
        if !sign_out && (username.is_empty() || password.is_empty()) {
            return;
        }
        account.busy = true;
        account.error = None;
        let client = self.subtitle_search.client.clone();
        self.jobs.spawn("subtitle-account", move || {
            let result = if sign_out {
                client.sign_out().map(|()| client.quota())
            } else {
                client.sign_in(&username, &password)
            };
            AppJob::SubtitleSearch(Box::new(Job::Account(result)))
        });
    }

    pub(super) fn subtitle_search_finished(&mut self, job: Job) {
        self.refresh_subtitle_account();
        match job {
            Job::Results {
                open,
                generation,
                query,
                identification,
                languages,
                result,
            } => {
                if open != self.open_seq
                    || generation != self.subtitle_search.generation
                    || !self.subtitle_panel_current()
                {
                    return;
                }
                self.subtitle_search.pending_search = None;
                let search = self.ui.subtitle_search.as_mut().expect("open search panel");
                self.subtitle_search.identification = Some((open, identification));
                search.busy = false;
                if !languages.is_empty() {
                    search.languages = languages;
                }
                if search.submitted.is_none() && !search.manual {
                    search.query = query.clone();
                }
                search.submitted = Some(query);
                match result {
                    Ok(results) => search.set_results(results),
                    Err(e) => search.error = Some(e),
                }
            }
            Job::Downloaded {
                media,
                file,
                result,
            } => {
                let Some(request) = self.subtitle_search.downloads.remove(&(media, file)) else {
                    return;
                };
                let panel_current = self.subtitle_panel_current()
                    && self.subtitle_search.generation == request.search
                    && self.open_seq == request.open;
                if panel_current
                    && let Some(search) = &mut self.ui.subtitle_search
                    && search.downloading == Some(file)
                {
                    search.downloading = None;
                    search.error = result.as_ref().err().cloned();
                }
                let current_media = self.playback.as_ref().is_some_and(|p| {
                    p.record_id == Some(media) && self.locked_source_of(&p.request).is_none()
                });
                if let Ok((attachment, cues)) = result
                    && current_media
                    && self
                        .services
                        .library
                        .user_subtitles(media)
                        .is_ok_and(|rows| rows.iter().any(|row| row.id == attachment.id))
                    && let Some(p) = &mut self.playback
                {
                    let location = attachment.location();
                    let file = attachment.as_file();
                    let index = if let Some(index) =
                        p.subtitle_files.iter().position(|f| f.location == location)
                    {
                        p.subtitle_files[index] = file;
                        index
                    } else {
                        p.subtitle_files.push(file);
                        p.subtitle_files.len() - 1
                    };
                    if let Some(old) = p
                        .managed_subtitles
                        .iter_mut()
                        .find(|a| a.id == attachment.id)
                    {
                        *old = attachment;
                    } else {
                        p.managed_subtitles.push(attachment);
                    }
                    if panel_current
                        && self.subtitle_search.selection == request.selection
                        && self.ui.subtitle_search.as_ref().is_some_and(|s| !s.dirty)
                    {
                        self.use_subtitle_cues(index, cues);
                        self.close_subtitle_search();
                    }
                }
            }
            Job::Account(result) => {
                self.ui.subtitle_account.busy = false;
                self.ui.subtitle_account.error = result.err();
            }
        }
        self.repaint_all();
    }

    fn subtitle_panel_current(&self) -> bool {
        self.ui.subtitle_search.is_some()
            && self.ui.tracks_open
            && self.ui_visible
            && !self.show_browser
            && self
                .playback
                .as_ref()
                .is_some_and(|p| self.locked_source_of(&p.request).is_none())
    }

    pub(super) fn select_subtitles(&mut self, pick: Pick) {
        self.subtitle_search.selection += 1;
        if let Some(search) = &mut self.ui.subtitle_search {
            search.downloading = None;
        }
        if let Pick::File(i) = pick {
            self.load_subtitle_file(i, None);
            return;
        }
        if pick == Pick::Off && self.playback.as_ref().is_some_and(server_subtitle_active) {
            self.play_source_track(None, Some(None));
            return;
        }
        if let Some(p) = &mut self.playback {
            p.active_subtitle_file = None;
            p.player.clear_external_subtitles();
            if p.player.subtitle_stream() != pick.stream() {
                p.player.select_subtitle(pick.stream());
            }
            save_pick(p, &self.services.library, pick);
        }
    }

    fn load_subtitle_file(&self, i: usize, search: Option<u64>) {
        let Some(file) = self
            .playback
            .as_ref()
            .and_then(|p| p.subtitle_files.get(i))
            .cloned()
        else {
            return;
        };
        let opener = self.services.opener.clone();
        let seq = self.open_seq;
        let selection = self.subtitle_search.selection;
        self.jobs.spawn("subtitles", move || AppJob::Subtitles {
            seq,
            selection,
            search,
            result: playback::read_subtitle_file(&opener, &file).and_then(|cues| {
                if cues.is_empty() {
                    Err(tr!("subtitle-error-invalid").into())
                } else {
                    Ok(cues)
                }
            }),
            location: file.location,
        });
    }

    pub(super) fn subtitle_file_finished(
        &mut self,
        seq: u64,
        selection: u64,
        search: Option<u64>,
        location: String,
        result: Result<Vec<Cue>, String>,
    ) {
        let Some(p) = &self.playback else { return };
        if seq != self.open_seq
            || selection != self.subtitle_search.selection
            || search.is_some_and(|generation| {
                generation != self.subtitle_search.generation
                    || !self.subtitle_panel_current()
                    || self.ui.subtitle_search.as_ref().is_some_and(|s| s.dirty)
            })
            || self.locked_source_of(&p.request).is_some()
        {
            // A download the search has moved on from still ends.
            if search.is_some()
                && let Some(s) = &mut self.ui.subtitle_search
            {
                s.downloading = None;
            }
            return;
        }
        let Some(index) = p.subtitle_files.iter().position(|s| s.location == location) else {
            return;
        };
        let from_search = search.is_some();
        match result {
            Ok(cues) => {
                self.use_subtitle_cues(index, cues);
                if from_search {
                    self.close_subtitle_search();
                }
            }
            Err(e) if from_search => {
                if let Some(search) = &mut self.ui.subtitle_search {
                    search.downloading = None;
                    search.error = Some(Error::InvalidSubtitle);
                }
                log::debug!("reading downloaded subtitles: {e}");
            }
            Err(e) => self.ui.toast(tr!("toast-subtitles", title = e)),
        }
        self.repaint_all();
    }

    fn use_subtitle_cues(&mut self, index: usize, cues: Vec<Cue>) {
        let Some(p) = &mut self.playback else { return };
        let restart = server_subtitle_active(p);
        p.active_subtitle_file = Some(index);
        p.player.set_external_subtitles(cues);
        if let Some(attachment) = p.active_managed_subtitle() {
            p.player.set_external_subtitle_delay(attachment.delay_ms);
        }
        save_pick(p, &self.services.library, Pick::File(index));
        if restart && p.request.source_subtitle.as_deref() == Some("0") {
            let request = OpenRequest {
                start_at: Some(p.player.position()),
                paused: p.player.is_paused(),
                ..p.request.clone()
            };
            self.open_request(request);
            self.reopen_tracks = true;
        }
    }

    fn adjust_subtitle_delay(&mut self, delta: Option<i32>) {
        let Some(p) = &mut self.playback else { return };
        let (Some(media), Some(attachment)) = (p.record_id, p.active_managed_subtitle()) else {
            return;
        };
        let id = attachment.id.clone();
        let delay_ms = delta.map_or(0, |delta| {
            (attachment.delay_ms + delta).clamp(-10_000, 10_000)
        });
        match self
            .services
            .library
            .set_subtitle_delay(media, &id, delay_ms)
        {
            Ok(true) => {
                if let Some(attachment) = p.managed_subtitles.iter_mut().find(|a| a.id == id) {
                    attachment.delay_ms = delay_ms;
                }
                p.player.set_external_subtitle_delay(delay_ms);
            }
            Ok(false) | Err(_) => self.ui.toast(tr!("subtitle-error-storage")),
        }
    }

    fn remove_subtitle_download(&mut self) {
        let Some(p) = &mut self.playback else { return };
        let (Some(media), Some(index), Some(attachment)) = (
            p.record_id,
            p.active_subtitle_file,
            p.active_managed_subtitle(),
        ) else {
            return;
        };
        let id = attachment.id.clone();
        match self
            .subtitle_search
            .store
            .remove(&self.services.library, media, &id)
        {
            Ok(Some(_)) => {
                self.subtitle_search.selection += 1;
                p.active_subtitle_file = None;
                p.last_subtitles = None;
                p.player.clear_external_subtitles();
                if p.player.subtitle_stream().is_some() {
                    p.player.select_subtitle(None);
                }
                p.subtitle_files.remove(index);
                p.managed_subtitles.retain(|a| a.id != id);
                save_pick(p, &self.services.library, Pick::Off);
            }
            Ok(None) | Err(_) => self.ui.toast(tr!("subtitle-error-storage")),
        }
    }
}

fn server_subtitle_active(p: &playback::Playback) -> bool {
    p.transcoding.is_some()
        && !p.source_tracks.is_empty()
        && p.request
            .source_subtitle
            .as_deref()
            .is_some_and(|id| id != "0")
}

fn save_pick(p: &mut playback::Playback, library: &fp_library::Library, pick: Pick) {
    p.picked.subtitles = subtitles::saved(pick, &p.subtitle_files);
    if !p.source_tracks.is_empty() {
        let track = p.source_tracks.iter().find(|t| {
            t.subtitle
                && match pick {
                    Pick::Stream(i) => t.index == Some(i),
                    Pick::File(i) => p
                        .subtitle_files
                        .get(i)
                        .is_some_and(|s| Some(&s.location) == t.location.as_ref()),
                    Pick::Off => false,
                }
        });
        p.request.source_subtitle = track.map(|t| t.id.clone()).or_else(|| Some("0".into()));
    }
    p.save_tracks(library);
}

fn subtitle_entry(request: &OpenRequest, title: &str) -> Entry {
    request.entry.clone().unwrap_or_else(|| {
        let name = request.title.clone().unwrap_or_else(|| {
            if fp_sources::is_source_location(&request.location) {
                String::new()
            } else {
                let basename = fp_sources::urlutil::last_segment(&request.location);
                if fp_core::source::is_video_name(&basename) {
                    basename
                } else {
                    title.to_string()
                }
            }
        });
        Entry::new(name, &request.location, EntryKind::Video)
    })
}

fn identify(
    entry: Entry,
    source: Option<Arc<dyn fp_sources::Source>>,
    direct: bool,
    mut signature: OriginalSubtitleSignature,
    cancel: &AtomicBool,
) -> Identification {
    let mut identity = MediaIdentity::from_entry(&entry);
    if let Some(source) = &source
        && !cancel.load(Ordering::Relaxed)
        && let Ok(Some(resolved)) = source.media_identity(&entry)
    {
        identity = resolved.with_fallback(&entry);
    }
    if direct && !cancel.load(Ordering::Relaxed) {
        let deadline = Instant::now() + Duration::from_secs(2);
        let reader = match source {
            Some(source) => source.open_original(&entry, deadline).ok().flatten(),
            None => playback::local_subtitle_original(&entry.location)
                .map(|file| Arc::new(file) as Arc<dyn fp_core::ByteSource>),
        };
        if let Some(reader) = reader
            && let Ok(Some(hash)) =
                movie_hash(&*reader, deadline, || cancel.load(Ordering::Relaxed))
        {
            // Opening the video again compares the size its listing gives,
            // which can be rounded (a web folder's "1.2G"), so that one stays.
            // A remote HTTP Last-Modified can use a different clock than the
            // media catalog, so the mtime stays as at playback open too.
            signature.size.get_or_insert(hash.size);
            signature.moviehash = Some(hash.hash);
        }
    }
    Identification {
        identity,
        signature,
    }
}

fn clear_identification(query: &mut SearchQuery) {
    query.imdb_id = None;
    query.tmdb_id = None;
    query.parent_imdb_id = None;
    query.parent_tmdb_id = None;
    query.moviehash = None;
    query.file_name = None;
}

fn apply_identification(query: &mut SearchQuery, known: &Identification, first: bool) {
    let id = &known.identity;
    if first {
        if let Some(title) = id.series_title.as_ref().or(id.title.as_ref()) {
            query.title.clone_from(title);
        }
        query.kind = id.kind;
        query.year = id.year;
        query.season = id.season;
        query.episode = id.episode;
    }
    query.imdb_id = id.imdb_id;
    query.tmdb_id = id.tmdb_id;
    query.parent_imdb_id = id.series_imdb_id;
    query.parent_tmdb_id = id.series_tmdb_id;
    query.moviehash.clone_from(&known.signature.moviehash);
    query.file_name.clone_from(&id.release_basename);
}

fn persist_download(
    client: &Client,
    store: &SubtitleStore,
    library: &fp_library::Library,
    target: &MediaAttachmentTarget,
    file: ResultFile,
    original_signature: Option<OriginalSubtitleSignature>,
) -> Result<(UserSubtitle, Vec<Cue>), Error> {
    let file_id = i64::try_from(file.id).map_err(|_| Error::InvalidResponse)?;
    let download = client.download(file.id)?;
    let staged = store.stage(&download.bytes).map_err(|_| Error::Storage)?;
    let input = FileSource::open(staged.path()).map_err(|_| Error::Storage)?;
    let cues = fp_media::subtitle::load_file(Arc::new(input), "download.srt")
        .map_err(|_| Error::InvalidSubtitle)?;
    if !cues
        .iter()
        .any(|c| c.end > c.start && !c.text.trim().is_empty())
    {
        return Err(Error::InvalidSubtitle);
    }
    let hash = ring::digest::digest(&ring::digest::SHA256, &download.bytes);
    let mut checksum = String::with_capacity(64);
    for byte in hash.as_ref() {
        use std::fmt::Write;
        write!(&mut checksum, "{byte:02x}").expect("write to String");
    }
    let attachment = store
        .commit(
            staged,
            library,
            target,
            NewUserSubtitle {
                provider: "opensubtitles".into(),
                provider_file_id: file_id,
                download_parameters: api::DOWNLOAD_PARAMETERS.into(),
                display_name: file.name,
                language: file.language,
                language_name: Some(file.language_name),
                release: Some(file.release).filter(|s| !s.is_empty()),
                flags: SubtitleFlags {
                    sdh: file.sdh,
                    forced: file.forced,
                    machine_translated: file.machine_translated,
                    ai_translated: file.ai_translated,
                },
                checksum: checksum.clone(),
                original_signature,
            },
        )
        .map_err(|e| {
            log::warn!("saving subtitle download: {e}");
            Error::Storage
        })?;
    let cues = if attachment.checksum == checksum {
        cues
    } else {
        let input = store
            .open(&attachment.location())
            .map_err(|_| Error::Storage)?;
        fp_media::subtitle::load_file(Arc::new(input), "download.srt")
            .map_err(|_| Error::InvalidSubtitle)?
    };
    Ok((attachment, cues))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_download_keeps_the_size_the_listing_gave() {
        let path = std::env::temp_dir().join(format!("fpx-identify-{}.mp4", std::process::id()));
        // Big enough for the fingerprint, which reads 64 KiB at each end.
        std::fs::write(&path, vec![7u8; 256 * 1024]).unwrap();
        let entry = Entry::new(
            "clip",
            format!("file://{}", path.display()),
            EntryKind::Video,
        );
        let listed = OriginalSubtitleSignature {
            size: Some(1_200_000_000),
            ..Default::default()
        };
        let found = identify(entry, None, true, listed, &AtomicBool::new(false));
        let _ = std::fs::remove_file(&path);
        assert_eq!(found.signature.size, Some(1_200_000_000));
        assert!(found.signature.moviehash.is_some());
    }
}
