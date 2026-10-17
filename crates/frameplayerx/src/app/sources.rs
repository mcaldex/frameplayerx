//! Browsing sources: folders, scene feeds, Jellyfin/Emby, and PIN locks.

use super::*;

impl App {
    /// Whether source `id` has a PIN and is not unlocked.
    pub(super) fn source_locked(&self, id: &str) -> bool {
        crate::pin::locked(&self.settings.source_pins, &self.ui.unlocked_sources, id)
    }

    pub(crate) fn source_config(&self, id: &str) -> Option<&fp_sources::SourceConfig> {
        self.services.source_config(id)
    }

    pub(crate) fn source_name(&self, id: &str) -> String {
        self.services
            .source_config(id)
            .map(|c| c.name().to_string())
            .unwrap_or_else(|| id.to_string())
    }

    /// Asks for the PIN of a locked source.
    pub(super) fn ask_pin(&mut self, source: &str) {
        self.ui.source = None;
        self.ui.screen = Screen::Sources;
        self.show_browser = true;
        self.ui_visible = true;
        self.ui.pin_form = Some(ui::PinForm {
            source: source.to_string(),
            name: self.source_name(source),
            ..Default::default()
        });
    }

    /// The locked source whose folder or address `location` is under.
    pub(super) fn locked_by(&self, location: &str) -> Option<String> {
        self.services
            .source_configs
            .iter()
            .filter(|c| self.source_locked(c.id()))
            .find(|c| crate::pin::source_contains(c, location))
            .map(|c| c.id().to_string())
    }

    /// The locked source keeping `source` shut: itself, or a locked source
    /// its folder or address is inside (the share a locked folder is in
    /// stays open, a folder inside a locked share does not).
    pub(super) fn lock_on(&self, source: &str) -> Option<String> {
        if self.source_locked(source) {
            return Some(source.to_string());
        }
        let root = self
            .services
            .source_config(source)
            .map(crate::pin::source_root)?;
        self.locked_by(&root)
    }

    /// The locked source a video belongs to, if any: the one it was opened
    /// from, the one its library entry came from, or one whose folder or
    /// address it is under, whichever source it was opened from.
    pub(super) fn locked_source_of(&self, req: &OpenRequest) -> Option<String> {
        if self.settings.source_pins.is_empty() {
            return None;
        }
        let from_library = self
            .services
            .library
            .get_by_location(&req.location)
            .ok()
            .flatten()
            .map(|r| r.source_id);
        [req.source_id.clone(), from_library]
            .into_iter()
            .flatten()
            .find(|id| self.source_locked(id))
            .or_else(|| self.locked_by(&req.location))
    }

    /// What locked sources hide: their videos, and everything under their
    /// folders or addresses whichever source lists or added it.
    pub(super) fn locked_hidden(&self) -> fp_library::Hidden {
        let sources = crate::pin::locked_ids(&self.settings.source_pins, &self.ui.unlocked_sources);
        let locations = self
            .services
            .source_configs
            .iter()
            .filter(|c| sources.iter().any(|s| s == c.id()))
            .map(crate::pin::source_root)
            .filter(|r| !r.is_empty())
            .collect();
        fp_library::Hidden { sources, locations }
    }

    /// Leaves locked sources' videos out of the library's listings (and the
    /// remotes'); nothing changes while no source has a PIN.
    pub(super) fn sync_locks(&mut self) {
        if self.settings.source_pins.is_empty() && self.services.library.hidden().is_empty() {
            return;
        }
        let hidden = self.locked_hidden();
        if hidden != self.services.library.hidden() {
            self.services.library.set_hidden(hidden);
            self.ui.invalidate();
            // A folder being browsed may hold what is hidden now.
            if let Some(b) = self.ui.folder_mut() {
                let (source, location) = (b.source.clone(), b.location.clone());
                self.browse(source, location);
            }
        }
    }

    pub fn browse(&mut self, source: String, location: Option<String>) {
        if let Some(lock) = self.lock_on(&source) {
            self.ask_pin(&lock);
            return;
        }
        let Some(src) = self.services.opener.source(&source) else {
            self.ui.toast("That source is not available");
            return;
        };
        // Into the listing on screen, else a folder page of its own.
        if !self.ui.folder_mut().is_some_and(|b| b.source == source) {
            self.ui.source = Some(SourceView {
                source: source.clone(),
                name: src.name().to_string(),
                back_to: Screen::Sources,
                page: SourcePage::Folders(Browse {
                    source: source.clone(),
                    ..Default::default()
                }),
            });
        }
        let Some(b) = self.ui.folder_mut() else {
            return;
        };
        b.location = location.clone();
        b.loading = true;
        b.error = None;
        let lib = self.services.library.clone();
        self.jobs.spawn("list", move || {
            let result = src.list(location.as_deref()).map(|mut e| {
                // Thumbnails the library already made for these videos.
                for entry in e.iter_mut().filter(|e| {
                    e.thumbnail_url.is_none() && e.kind == fp_core::source::EntryKind::Video
                }) {
                    if let Ok(Some(r)) = lib.get_by_location(&entry.location) {
                        entry.thumbnail_url = r.thumbnail.map(|p| p.display().to_string());
                    }
                }
                e
            });
            let result = result.map_err(|e| {
                let removable = source == crate::services::REMOVABLE_SOURCE;
                if removable && crate::services::removable_mounts().is_empty() {
                    "No microSD card or USB drive is inserted.".to_string()
                } else {
                    e.to_string()
                }
            });
            Job::Listed {
                source,
                location,
                result,
            }
        });
    }

    /// Opens a source in the browser its kind gets: the headset's videos
    /// and folders, a Jellyfin/Emby client, a feed's scenes, or folders.
    pub(super) fn open_source(&mut self, source: String, back_to: Screen) {
        self.ui.details = None;
        self.ui.pin_form = None;
        self.ui.source_form = None;
        self.ui.screen = Screen::Sources;
        if source == crate::services::LOCAL_HUB {
            self.ui.source = Some(SourceView {
                source,
                name: "On this device".into(),
                back_to,
                page: SourcePage::Local(Default::default()),
            });
            return;
        }
        if let Some(lock) = self.lock_on(&source) {
            self.ask_pin(&lock);
            return;
        }
        let Some(kind) = self
            .services
            .source_config(&source)
            .map(fp_sources::SourceConfig::kind)
        else {
            self.ui.toast("That source is not available");
            return;
        };
        let name = self.source_name(&source);
        self.feed_queue.clear();
        let view = |page| SourceView {
            source: source.clone(),
            name: name.clone(),
            back_to,
            page,
        };
        match kind {
            fp_sources::SourceKind::MediaBrowser => {
                let seq = self.ui.next_seq();
                let j = ui::jellyfin::JellyfinView::new(seq);
                self.ui.source = Some(view(SourcePage::Jellyfin(j)));
                self.apply(Action::Jellyfin {
                    source,
                    seq,
                    load: ui::jellyfin::Load::Home,
                });
            }
            fp_sources::SourceKind::DeoVr | fp_sources::SourceKind::HereSphere => {
                let f = ui::feeds::FeedView::new(kind);
                self.ui.source = Some(view(SourcePage::Feed(f)));
                self.apply(Action::FeedLoad { source });
            }
            _ => {
                self.ui.source = Some(view(SourcePage::Folders(Browse {
                    source: source.clone(),
                    ..Default::default()
                })));
                self.browse(source, None);
            }
        }
    }

    /// Fetches waiting feed scenes' details, a few at a time.
    pub(super) fn pump_feed(&mut self) {
        const WORKERS: usize = 6;
        let current = match self.ui.source.as_ref() {
            Some(SourceView {
                source,
                page: SourcePage::Feed(_),
                ..
            }) => source.clone(),
            _ => {
                self.feed_queue.clear();
                return;
            }
        };
        while self.feed_busy < WORKERS {
            let Some((source, url)) = self.feed_queue.pop_front() else {
                break;
            };
            if source != current {
                continue;
            }
            let Some(src) = self.services.opener.source(&source) else {
                continue;
            };
            self.feed_busy += 1;
            self.jobs.spawn("feed-scene", move || {
                let result = scene_details(src.as_ref(), &url, false);
                Job::FeedScene {
                    source,
                    url,
                    full: false,
                    result: Box::new(result),
                }
            });
        }
    }

    /// Brings a Jellyfin page up to date after watching (marks, resume
    /// points and "next up" changed on the server).
    pub(super) fn refresh_source(&mut self) {
        let Some(SourceView {
            source,
            page: SourcePage::Jellyfin(j),
            ..
        }) = self.ui.source.as_mut()
        else {
            return;
        };
        let source = source.clone();
        if let Some((seq, load)) = j.reload() {
            // After the server has heard that playback stopped.
            self.jellyfin_job(source, seq, load, Duration::from_millis(800));
        }
    }

    /// Asks a Jellyfin/Emby server for a page, after `delay`.
    pub(super) fn jellyfin_job(
        &mut self,
        source: String,
        seq: u64,
        load: ui::jellyfin::Load,
        delay: Duration,
    ) {
        let Some(src) = self.services.opener.source(&source) else {
            return;
        };
        self.jobs.spawn("jellyfin", move || {
            std::thread::sleep(delay);
            let result = match src.as_media_browser() {
                Some(mb) => ui::jellyfin::run(mb, &load).map_err(|e| e.to_string()),
                None => Err("not a Jellyfin or Emby server".to_string()),
            };
            Job::Jellyfin {
                source,
                seq,
                result: Box::new(result),
            }
        });
    }

    // ---- editing sources ----

    pub(super) fn add_source(&mut self, cfg: fp_sources::SourceConfig) {
        let mut list = self.services.source_configs.clone();
        list.push(cfg.clone());
        let errors = self.services.set_sources(list);
        match errors.into_iter().find(|e| e.id == cfg.id()) {
            Some(e) => self.ui.toast(e.to_string()),
            None => {
                self.ui.toast(format!("Added {}", cfg.name()));
                self.open_source(cfg.id().to_string(), Screen::Sources);
            }
        }
    }

    pub(super) fn update_source(&mut self, cfg: fp_sources::SourceConfig) {
        let id = cfg.id().to_string();
        // The form only opens for unlocked sources, but a PIN could have
        // been set since.
        if self.source_locked(&id) {
            self.ask_pin(&id);
            return;
        }
        let list = self
            .services
            .source_configs
            .iter()
            .map(|c| if c.id() == id { cfg.clone() } else { c.clone() })
            .collect();
        let errors = self.services.set_sources(list);
        match errors.into_iter().find(|e| e.id == id) {
            Some(e) => self.ui.toast(e.to_string()),
            None => self.ui.toast(format!("Saved {}", cfg.name())),
        }
        // Its browser and saved listing were made with the old details.
        self.close_source_view(&id);
        let _ = std::fs::remove_file(fp_sources::FeedLibrary::cache_path(&id));
        self.ui.home = None;
    }

    pub(super) fn remove_source(&mut self, id: &str) {
        if self.source_locked(id) {
            self.ask_pin(id);
            return;
        }
        self.settings.source_pins.remove(id);
        self.ui.unlocked_sources.remove(id);
        self.close_source_view(id);
        let _ = std::fs::remove_file(fp_sources::FeedLibrary::cache_path(id));
        // Its videos go with it, except what was watched or marked.
        match self.services.library.forget_unwatched(id) {
            Ok(n) if n > 0 => log::info!("forgot {n} unwatched videos of {id}"),
            Ok(_) => {}
            Err(e) => log::warn!("forgetting the videos of {id}: {e}"),
        }
        self.ui.invalidate();
        let list = self
            .services
            .source_configs
            .iter()
            .filter(|c| c.id() != id)
            .cloned()
            .collect();
        self.services.set_sources(list);
    }

    /// Leaves source `id`'s browser if it is open.
    fn close_source_view(&mut self, id: &str) {
        if self.ui.source.as_ref().is_some_and(|s| s.source == id) {
            self.ui.source = None;
        }
    }

    // ---- PINs ----

    pub(super) fn unlock_source(&mut self, source: String, pin: &str) {
        let right = self
            .settings
            .source_pins
            .get(&source)
            .is_some_and(|h| crate::pin::verify(h, pin));
        if right {
            self.ui.unlocked_sources.insert(source.clone());
            self.ui.pin_form = None;
            self.sync_locks();
            self.open_source(source, Screen::Sources);
        } else if let Some(f) = self.ui.pin_form.as_mut() {
            f.pin.clear();
            f.wrong = true;
        }
    }

    pub(super) fn lock_source(&mut self, id: &str) {
        self.ui.unlocked_sources.remove(id);
        self.close_source_view(id);
        self.ui.toast(format!("Locked {}", self.source_name(id)));
    }

    /// Gives a source a PIN, or takes it away (`None`).
    pub(super) fn set_source_pin(&mut self, source: &str, pin: Option<String>) {
        // Changing it needs the source open, as the PIN form only offers it
        // then.
        if self.source_locked(source) {
            self.ask_pin(source);
            return;
        }
        self.ui.pin_form = None;
        let name = self.source_name(source);
        match pin {
            Some(pin) if crate::pin::valid(&pin) => {
                self.settings
                    .source_pins
                    .insert(source.to_string(), crate::pin::hash(&pin));
                self.ui.unlocked_sources.remove(source);
                self.close_source_view(source);
                // Locked now: its details are not for showing.
                let editing =
                    |f: &ui::SourceForm| f.editing.as_ref().is_some_and(|c| c.id() == source);
                if self.ui.source_form.as_ref().is_some_and(editing) {
                    self.ui.source_form = None;
                }
                self.ui.toast(format!("{name} is locked with a PIN"));
            }
            Some(_) => {}
            None => {
                self.settings.source_pins.remove(source);
                self.ui.unlocked_sources.remove(source);
                self.ui.toast(format!("Removed the PIN of {name}"));
            }
        }
    }

    // ---- Jellyfin, feeds, DLNA ----

    /// Marks a Jellyfin/Emby item as a favorite, or as watched, on the
    /// server.
    pub(super) fn mark_jellyfin(
        &mut self,
        source: &str,
        id: String,
        mark: ui::jellyfin::Mark,
        on: bool,
    ) {
        let Some(src) = self.services.opener.source(source) else {
            return;
        };
        self.jobs.spawn("jellyfin-mark", move || {
            let Some(mb) = src.as_media_browser() else {
                return Job::Toast("Not a Jellyfin or Emby server".into());
            };
            let r = match mark {
                ui::jellyfin::Mark::Favorite => mb.set_favorite(&id, on),
                ui::jellyfin::Mark::Played => mb.set_played(&id, on),
            };
            match r {
                Ok(()) => Job::Done,
                Err(e) => Job::Toast(format!("The server did not take it: {e}")),
            }
        });
    }

    /// Lists a feed's scenes: the last listing at once, then the server's,
    /// then (HereSphere) its scan.
    pub(super) fn load_feed(&mut self, source: String) {
        let Some(src) = self.services.opener.source(&source) else {
            return;
        };
        let tx = self.jobs.sender();
        self.jobs.spawn("feed", move || {
            let path = fp_sources::FeedLibrary::cache_path(&source);
            let listed =
                |result: Result<fp_sources::FeedLibrary, String>, cached| Job::FeedLibrary {
                    source: source.clone(),
                    result,
                    cached,
                };
            // A DeoVR feed's root can take seconds.
            if let Some(lib) = fp_sources::FeedLibrary::load(&path) {
                let _ = tx.send(listed(Ok(lib), true));
            }
            let Some(feed) = src.as_scene_feed() else {
                return listed(Err("not a scene feed".into()), false);
            };
            let mut lib = match feed.library() {
                Ok(lib) => lib,
                Err(e) => return listed(Err(e.to_string()), false),
            };
            let _ = tx.send(listed(Ok(lib.clone()), false));
            let scenes = feed.scan();
            if let Some(s) = &scenes {
                lib.merge_scan(s.clone());
            }
            if let Err(e) = lib.save(&path) {
                log::warn!("saving the listing of {source}: {e}");
            }
            Job::FeedScan {
                source: source.clone(),
                scenes,
            }
        });
    }

    /// Queues feed scenes for their details: `front` (cards on screen)
    /// before those already waiting.
    pub(super) fn queue_feed_scenes(&mut self, source: &str, urls: Vec<String>, front: bool) {
        let items = urls.into_iter().map(|u| (source.to_string(), u));
        if front {
            for item in items.rev() {
                self.feed_queue.push_front(item);
            }
        } else {
            self.feed_queue.extend(items);
        }
        self.pump_feed();
    }

    /// Fetches everything about one scene, for its page.
    pub(super) fn load_feed_scene(&mut self, source: String, url: String) {
        let Some(src) = self.services.opener.source(&source) else {
            return;
        };
        self.jobs.spawn("feed-scene", move || {
            let result = scene_details(src.as_ref(), &url, true);
            Job::FeedScene {
                source,
                url,
                full: true,
                result: Box::new(result),
            }
        });
    }

    pub(super) fn discover_dlna(&mut self) {
        self.ui.dlna_searching = true;
        self.jobs.spawn("dlna", || {
            Job::Dlna(fp_sources::discover(Duration::from_secs(3)).map_err(|e| e.to_string()))
        });
    }

    // ---- Quick Connect ----

    pub(super) fn quick_connect_initiate(&mut self, address: String, insecure_tls: bool) {
        self.jobs.spawn("quick_connect_initiate", move || {
            let addr = fp_sources::urlutil::with_http_scheme(&address);
            Job::QuickConnectInitiated(
                fp_sources::quick_connect_initiate_url(&addr, insecure_tls)
                    .map_err(|e| e.to_string()),
            )
        });
    }

    pub(super) fn quick_connect_poll(
        &mut self,
        address: String,
        secret: String,
        insecure_tls: bool,
    ) {
        self.jobs.spawn("quick_connect_poll", move || {
            let addr = fp_sources::urlutil::with_http_scheme(&address);
            Job::QuickConnectPolled(
                fp_sources::quick_connect_poll_url(&addr, &secret, insecure_tls)
                    .map_err(|e| e.to_string()),
            )
        });
    }
}

/// A feed scene's details (`full`: with its media sources, for its page).
/// Blocking: call from a job thread.
fn scene_details(
    src: &dyn fp_sources::Source,
    url: &str,
    full: bool,
) -> Result<fp_sources::SceneInfo, String> {
    let feed = src.as_scene_feed().ok_or("not a scene feed")?;
    feed.scene(url, full).map_err(|e| e.to_string())
}
