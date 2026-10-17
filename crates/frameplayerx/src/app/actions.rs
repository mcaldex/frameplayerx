//! Carrying out what the interface asks for.

use super::*;

impl App {
    pub(crate) fn apply(&mut self, a: Action) {
        match a {
            Action::Open(req) => {
                self.queue.clear();
                self.open_request(req);
            }
            Action::OpenList(list, pos) => self.open_list(list, pos),
            Action::TogglePause => self.toggle_pause(),
            Action::Seek(t) => {
                if let Some(p) = self.player() {
                    p.seek(t, true);
                }
            }
            Action::SeekRelative(d) => {
                if let Some(p) = self.player() {
                    p.seek_relative(d);
                }
            }
            Action::SetSpeed(s) => {
                if let Some(p) = self.player() {
                    p.set_speed(s.clamp(0.25, 3.0));
                }
            }
            Action::SetVolume(v) => {
                if let Some(p) = self.player() {
                    p.set_volume(v);
                }
                self.settings.volume = v;
            }
            Action::ClosePlayback => self.close_playback(),
            Action::SetFormat(f) => {
                if let Some(p) = &mut self.playback {
                    p.set_format(&self.services.library, f);
                }
                self.ui.invalidate();
            }
            Action::SaveKey => {
                if let Some(p) = &mut self.playback {
                    p.key_changed();
                }
            }
            Action::SaveView => {
                if let Some(p) = &mut self.playback {
                    p.save_settings(&self.services.library);
                    self.ui.toast("Adjustments saved");
                }
            }
            Action::ResetView => self.edit_view(|p, s| {
                p.settings = s.new_video_view();
                p.keyframes = Default::default();
            }),
            Action::AddKeyframe => self.edit_view(|p, _| {
                p.keyframes.insert(p.player.position(), p.settings);
            }),
            Action::ClearKeyframes => self.edit_view(|p, _| p.keyframes = Default::default()),
            Action::AddBookmark => self.add_bookmark(),
            Action::SelectAudio(i) => {
                if let Some(p) = self.player() {
                    p.select_audio(i);
                }
            }
            Action::SelectSubtitleStream(i) => {
                if let Some(p) = self.player() {
                    p.select_subtitle(i);
                }
            }
            Action::SelectSubtitleFile(i) => self.select_subtitle_file(i),
            Action::SetFavorite(id, f) => {
                let r = self.services.library.set_favorite(id, f);
                self.library_changed("the favorite", r);
            }
            Action::SetRating(id, r) => {
                let r = self.services.library.set_rating(id, r);
                self.library_changed("the rating", r);
            }
            Action::SetUserFormat(id, f) => {
                let r = self.services.library.set_user_format(id, f);
                self.library_changed("the format", r);
            }
            Action::RemoveFromHistory(id) => {
                let r = self.services.library.remove_from_history(id);
                self.library_changed("the change", r);
            }
            Action::Rescan { force } => self.rescan(force),
            Action::RetryFailed => {
                let n = self.services.library.retry_failed_probes().unwrap_or(0);
                self.services.wake_worker();
                self.ui.toast(format!("Retrying {n} videos"));
            }
            Action::ClearHistory => self.clear_history(),
            Action::AddFolder(p) => {
                if !self.settings.library_folders.contains(&p) {
                    self.settings.library_folders.push(p);
                    self.rescan(false);
                }
            }
            Action::RemoveFolder(i) => {
                if i < self.settings.library_folders.len() {
                    self.settings.library_folders.remove(i);
                }
            }
            Action::AddSource(cfg) => self.add_source(cfg),
            Action::UpdateSource(cfg) => self.update_source(cfg),
            Action::RemoveSource(id) => self.remove_source(&id),
            Action::UnlockSource { source, pin } => self.unlock_source(source, &pin),
            Action::LockSource(id) => self.lock_source(&id),
            Action::SetSourcePin { source, pin } => self.set_source_pin(&source, pin),
            Action::OpenSource { source, back_to } => self.open_source(source, back_to),
            Action::Browse { source, location } => self.browse(source, location),
            Action::Jellyfin { source, seq, load } => {
                self.jellyfin_job(source, seq, load, Duration::ZERO);
            }
            Action::JellyfinMark {
                source,
                id,
                mark,
                on,
            } => self.mark_jellyfin(&source, id, mark, on),
            Action::FeedLoad { source } => self.load_feed(source),
            Action::FeedFill {
                source,
                urls,
                front,
            } => self.queue_feed_scenes(&source, urls, front),
            Action::FeedScene { source, url } => self.load_feed_scene(source, url),
            Action::DiscoverDlna => self.discover_dlna(),
            Action::QuickConnectInitiate {
                address,
                insecure_tls,
            } => self.quick_connect_initiate(address, insecure_tls),
            Action::QuickConnectPoll {
                address,
                secret,
                insecure_tls,
            } => self.quick_connect_poll(address, secret, insecure_tls),
            Action::AddDevice(cfg) => {
                if !self.settings.haptic_devices.contains(&cfg) {
                    self.settings.haptic_devices.push(cfg.clone());
                }
                self.connect_device(cfg);
            }
            Action::RemoveDevice(i) => {
                if i < self.settings.haptic_devices.len() {
                    let cfg = self.settings.haptic_devices.remove(i);
                    self.services.remove_device(&cfg);
                }
            }
            Action::ReconnectDevices => {
                self.services.remove_devices();
                for cfg in self.settings.haptic_devices.clone() {
                    self.connect_device(cfg);
                }
            }
            Action::RegenerateToken => {
                if let Some(h) = &self.services.remote
                    && let Err(e) = h.regenerate_token()
                {
                    self.ui.toast(e.to_string());
                }
            }
            Action::CheckUpdates => self.check_updates(),
            Action::InstallUpdate => self.install_update(),
            Action::Recenter => self.recenter(),
            Action::TogglePassthrough => self.settings.passthrough = !self.settings.passthrough,
            Action::ShowBrowser(show) => {
                self.show_browser = show;
                if show {
                    self.ui.invalidate();
                    self.refresh_source();
                }
            }
            Action::Quit => self.quit = true,
            Action::HideControls => {
                self.ui_visible = false;
                self.show_browser = false;
            }
        }
    }

    /// The open video's player.
    pub(super) fn player(&self) -> Option<&fp_media::Player> {
        self.playback.as_ref().map(|p| &p.player)
    }

    fn open_list(&mut self, list: Vec<OpenRequest>, pos: usize) {
        if let Some(req) = list.get(pos).cloned() {
            self.queue = list;
            self.queue_pos = pos;
            self.open_request(req);
        }
    }

    fn toggle_pause(&self) {
        if let Some(p) = self.player() {
            if p.state() == PlayerState::Ended {
                p.seek(0.0, false);
                p.play();
            } else {
                p.toggle_pause();
            }
        }
    }

    fn close_playback(&mut self) {
        if let Some(p) = self.playback.take() {
            p.close(&self.services.library);
        }
        self.services.haptics.clear();
        self.show_browser = true;
        self.ui_visible = true;
        self.ui.invalidate();
        self.refresh_source();
    }

    /// Changes the open video's view adjustments (saved later with the
    /// rest of them).
    fn edit_view(&mut self, edit: impl FnOnce(&mut Playback, &Settings)) {
        if let Some(p) = &mut self.playback {
            edit(p, &self.settings);
            p.settings_dirty = true;
        }
    }

    fn add_bookmark(&mut self) {
        let Some(p) = &mut self.playback else { return };
        match p.add_bookmark(&self.services.library) {
            Ok(()) => self.ui.toast("Bookmark added"),
            Err(e) => self.not_saved("the bookmark", &e),
        }
    }

    fn select_subtitle_file(&mut self, i: Option<usize>) {
        let Some(p) = &mut self.playback else { return };
        p.player.clear_external_subtitles();
        p.active_subtitle_file = None;
        let Some((name, loc)) = i.and_then(|i| p.subtitle_files.get(i).cloned()) else {
            return;
        };
        let loaded = self
            .services
            .opener
            .open(&loc)
            .and_then(|s| p.player.load_subtitles(s, &name).map_err(|e| e.to_string()));
        match loaded {
            Ok(_) => p.active_subtitle_file = i,
            Err(e) => self.ui.toast(format!("Subtitles: {e}")),
        }
    }

    fn clear_history(&mut self) {
        match self.services.library.clear_history() {
            Ok(()) => self.ui.toast("Watch history cleared"),
            Err(e) => self.ui.toast(format!("Could not clear the history: {e}")),
        }
        self.ui.invalidate();
    }

    /// After a change to the library: re-query what shows it, and say so
    /// if it could not be saved.
    fn library_changed(&mut self, what: &str, r: fp_library::Result<()>) {
        if let Err(e) = r {
            self.not_saved(what, &e);
        }
        self.ui.invalidate();
    }

    /// Tells the user a change to the library could not be saved.
    fn not_saved(&mut self, what: &str, e: &fp_library::Error) {
        log::warn!("saving {what}: {e}");
        self.ui.toast(format!("Couldn't save {what}: {e}"));
    }
}
