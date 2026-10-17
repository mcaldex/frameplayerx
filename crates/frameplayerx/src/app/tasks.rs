//! Background work and its results: opening videos, scans, updates, drives.

use super::*;

pub(super) enum Job {
    Opened(u64, Box<Result<Opened, String>>),
    Listed {
        source: String,
        location: Option<String>,
        result: Result<Vec<fp_core::source::Entry>, String>,
    },
    Status(String),
    ScanDone(String),
    Jellyfin {
        source: String,
        seq: u64,
        result: Box<Result<ui::jellyfin::Loaded, String>>,
    },
    /// A scene feed's listing; `cached`: the one saved last time, shown
    /// while the server answers.
    FeedLibrary {
        source: String,
        result: Result<fp_sources::FeedLibrary, String>,
        cached: bool,
    },
    /// A HereSphere scan, after its listing (`None`: the server has none).
    FeedScan {
        source: String,
        scenes: Option<Vec<fp_sources::FeedScene>>,
    },
    /// A scene's details: for its page (`full`, with media) or a card.
    FeedScene {
        source: String,
        url: String,
        full: bool,
        result: Box<Result<fp_sources::SceneInfo, String>>,
    },
    Toast(String),
    /// Nothing to show (a server took a change).
    Done,
    Dlna(Result<Vec<fp_sources::DlnaDevice>, String>),
    Device(
        crate::settings::HapticDeviceConfig,
        Result<Box<dyn fp_haptics::Device>, String>,
    ),
    /// Download and install progress.
    Update(UpdateStatus),
    /// The result of update check number `.0`.
    UpdateChecked(u64, UpdateStatus, Option<Box<fp_updater::Update>>),
    Mounted(String, Result<std::path::PathBuf, String>),
    QuickConnectInitiated(Result<fp_sources::QuickConnectState, String>),
    QuickConnectPolled(Result<Option<(String, String)>, String>),
}

impl App {
    pub(super) fn rescan(&mut self, force: bool) {
        let mut folders = self.settings.library_folders.clone();
        if self.settings.index_removable {
            // Each drive on its own, so an absent card never marks videos
            // on another drive missing.
            let extra: Vec<_> = self
                .mounts
                .iter()
                .filter(|m| !folders.iter().any(|f| m.starts_with(f)))
                .cloned()
                .collect();
            folders.extend(extra);
        }
        let lib = self.services.library.clone();
        let tx = self.jobs.sender();
        self.ui.scan_status = Some("Scanning…".into());
        self.jobs.spawn("scan", move || {
            let opts = fp_library::ScanOptions {
                force,
                ..Default::default()
            };
            let (mut added, mut missing, mut errors) = (0, 0, Vec::new());
            for f in &folders {
                let mut last = Instant::now();
                let r = lib.scan_folder(f, crate::services::LIBRARY_SOURCE, &opts, |p| {
                    if last.elapsed() > Duration::from_millis(300) {
                        last = Instant::now();
                        let _ = tx.send(Job::Status(format!(
                            "Scanning {}: {} videos",
                            f.display(),
                            p.videos_found
                        )));
                    }
                });
                match r {
                    Ok(rep) => {
                        added += rep.added;
                        missing += rep.missing;
                    }
                    Err(e) => errors.push(format!("{}: {e}", f.display())),
                }
            }
            let mut msg = format!("Scan finished: {added} new, {missing} missing");
            if !errors.is_empty() {
                msg += &format!(" · {}", errors.join("; "));
            }
            Job::ScanDone(msg)
        });
    }

    /// Connects every configured device, while haptics are turned on.
    pub(super) fn connect_devices(&mut self) {
        for cfg in self.settings.haptic_devices.clone() {
            self.connect_device(cfg);
        }
    }

    pub(super) fn connect_device(&mut self, cfg: crate::settings::HapticDeviceConfig) {
        if !self.settings.haptics_enabled {
            return;
        }
        self.jobs.spawn("haptics", move || {
            let r = Services::connect_device(&cfg);
            Job::Device(cfg, r)
        });
    }

    pub(super) fn check_updates(&mut self) {
        self.ui.update = UpdateStatus::Checking;
        self.update = None;
        self.update_check += 1;
        let seq = self.update_check;
        let channel = self.settings.update_channel;
        self.jobs.spawn("update-check", move || {
            match fp_updater::check_channel(env!("CARGO_PKG_VERSION"), channel) {
                Ok(Some(u)) => Job::UpdateChecked(
                    seq,
                    UpdateStatus::Available {
                        version: u.version.to_string(),
                        notes: u.notes.clone(),
                    },
                    Some(Box::new(u)),
                ),
                Ok(None) => Job::UpdateChecked(seq, UpdateStatus::UpToDate, None),
                Err(e) => Job::UpdateChecked(seq, UpdateStatus::Failed(e.to_string()), None),
            }
        });
    }

    pub(super) fn install_update(&mut self) {
        let Some(u) = self.update.clone() else { return };
        let tx = self.jobs.sender();
        self.ui.update = UpdateStatus::Downloading {
            done: 0,
            total: u.artifact.size,
        };
        self.jobs.spawn("update-install", move || {
            let dir = fp_core::dirs::data_dir().join("updates");
            let cancel = AtomicBool::new(false);
            let mut last = Instant::now();
            let r = std::fs::create_dir_all(&dir)
                .map_err(|e| e.to_string())
                .and_then(|_| {
                    fp_updater::download(
                        &u,
                        &dir,
                        |done, total| {
                            if last.elapsed() > Duration::from_millis(250) {
                                last = Instant::now();
                                let _ =
                                    tx.send(Job::Update(UpdateStatus::Downloading { done, total }));
                            }
                        },
                        &cancel,
                    )
                    .map_err(|e| e.to_string())
                })
                .and_then(|zip| {
                    let install = fp_updater::current_install_dir().map_err(|e| e.to_string())?;
                    fp_updater::install::install_expecting(&zip, &install, &u.version)
                        .map_err(|e| e.to_string())
                });
            match r {
                Ok(v) => Job::Update(UpdateStatus::Installed {
                    version: v.to_string(),
                }),
                Err(e) => Job::Update(UpdateStatus::Failed(e)),
            }
        });
    }

    /// Takes in finished background work and the metadata worker's progress.
    pub(super) fn poll_jobs(&mut self) {
        for job in self.jobs.poll() {
            self.on_job(job);
        }
        self.poll_worker();
    }

    fn on_job(&mut self, job: Job) {
        match job {
            Job::Opened(seq, r) => self.opened(seq, *r),
            Job::Listed {
                source,
                location,
                result,
            } => self.listed(&source, location.as_deref(), result),
            Job::Status(s) => self.ui.scan_status = Some(s),
            Job::ScanDone(s) => {
                log::info!("{s}");
                self.ui.scan_status = Some(s);
                self.ui.invalidate();
                self.services.wake_worker();
            }
            Job::Jellyfin {
                source,
                seq,
                result,
            } => {
                if let Some(j) = self.ui.jellyfin_page(&source) {
                    j.loaded(seq, *result);
                }
            }
            Job::FeedLibrary {
                source,
                result,
                cached,
            } => {
                if let Some(f) = self.ui.feed_page(&source) {
                    f.loaded(result, cached);
                }
            }
            Job::FeedScan { source, scenes } => {
                if let Some(f) = self.ui.feed_page(&source) {
                    f.scanned(scenes);
                }
            }
            Job::FeedScene {
                source,
                url,
                full,
                result,
            } => {
                if !full {
                    self.feed_busy = self.feed_busy.saturating_sub(1);
                }
                if let Some(f) = self.ui.feed_page(&source) {
                    f.scene_loaded(&url, *result, full);
                }
                self.pump_feed();
            }
            Job::Toast(t) => self.ui.toast(t),
            Job::Done => {}
            Job::Dlna(r) => {
                self.ui.dlna_searching = false;
                match r {
                    Ok(d) => self.ui.dlna_found = d,
                    Err(e) => self.ui.toast(format!("Network search failed: {e}")),
                }
            }
            Job::Device(cfg, r) => {
                if let Err(e) = &r {
                    log::warn!("haptics {}: {e}", cfg.label());
                    self.ui.toast(format!("{}: {e}", cfg.label()));
                }
                self.services.device_connected(cfg, r);
            }
            Job::Mounted(device, r) => match r {
                // check_mounts announces the new drive.
                Ok(p) => log::info!("mounted /dev/{device} at {}", p.display()),
                Err(e) => {
                    log::warn!("mounting /dev/{device}: {e}");
                    self.ui.toast(format!("Couldn't open the drive: {e}"));
                }
            },
            Job::Update(status) => self.ui.update = status,
            Job::UpdateChecked(seq, status, update) => {
                if seq == self.update_check {
                    self.update = update.map(|u| *u);
                    self.ui.update = status;
                }
            }
            Job::QuickConnectInitiated(r) => self.quick_connect_code(r),
            Job::QuickConnectPolled(r) => self.quick_connect_answer(r),
        }
    }

    /// A video finished opening; one asked for before the latest is closed.
    fn opened(&mut self, seq: u64, r: Result<Opened, String>) {
        if seq != self.open_seq {
            if let Ok(o) = r {
                o.playback.close(&self.services.library);
            }
            return;
        }
        self.ui.opening = None;
        match r {
            Ok(o) => {
                if let Some(old) = self.playback.take() {
                    old.close(&self.services.library);
                }
                if let Some(h) = self.services.haptics() {
                    h.load(o.scripts);
                }
                self.playback = Some(o.playback);
                self.ui.open_error = None;
                self.ui.adjust_open = false;
                self.show_browser = false;
                self.ui_visible = true;
                self.last_activity = Instant::now();
                self.sub_key.clear();
                self.ui.invalidate();
            }
            Err(e) => {
                log::warn!("open failed: {e}");
                self.ui.open_error = Some(e);
            }
        }
    }

    /// A folder listing, for the folder on screen if it is still that one.
    fn listed(
        &mut self,
        source: &str,
        location: Option<&str>,
        result: Result<Vec<fp_core::source::Entry>, String>,
    ) {
        let Some(b) = self
            .ui
            .folder_mut()
            .filter(|b| b.source == source && b.location.as_deref() == location)
        else {
            return;
        };
        b.loading = false;
        match result {
            Ok(mut e) => {
                // Folders of locked sources inside this one.
                let hidden = self.services.library.hidden();
                e.retain(|entry| !hidden.hides(source, &entry.location));
                fp_sources::sort_entries_by(&mut e, self.settings.browse_sort);
                b.entries = e;
                b.error = None;
            }
            Err(e) => b.error = Some(e),
        }
    }

    // A Quick Connect answer for a form that has moved on (cancelled,
    // closed) is dropped.

    fn quick_connect_code(&mut self, r: Result<fp_sources::QuickConnectState, String>) {
        let Some(f) = self.ui.source_form.as_mut() else {
            return;
        };
        if f.quick_connect != QuickConnect::Requesting {
            return;
        }
        f.quick_connect = match r {
            Ok(qc) => QuickConnect::Waiting {
                code: qc.code,
                secret: qc.secret,
                polled: Instant::now(),
            },
            Err(e) => QuickConnect::Failed(e),
        };
        self.panels[MAIN].request_repaint();
    }

    fn quick_connect_answer(&mut self, r: Result<Option<(String, String)>, String>) {
        let (token, user_id) = match r {
            Ok(Some(t)) => t,
            Ok(None) => return,
            // Asked again shortly; a code that expired stays up until
            // cancelled.
            Err(e) => {
                log::warn!("Quick Connect poll error: {e}");
                return;
            }
        };
        let Some(f) = self.ui.source_form.as_mut() else {
            return;
        };
        if !matches!(f.quick_connect, QuickConnect::Waiting { .. }) {
            return;
        }
        f.quick_connect = QuickConnect::Authorized {
            token,
            user_id: Some(user_id),
        };
        self.ui.toast("Quick Connect authorized!");
        self.panels[MAIN].request_repaint();
    }

    /// The metadata worker's progress, shown while it makes thumbnails.
    fn poll_worker(&mut self) {
        let Some(rx) = &self.services.worker_events else {
            return;
        };
        for ev in rx.try_iter() {
            match ev {
                fp_library::WorkerEvent::Processed { done, total, .. }
                | fp_library::WorkerEvent::Failed { done, total, .. } => {
                    self.ui.worker_status = Some(format!("Preparing thumbnails {done}/{total}"));
                    if done % 12 == 0 || done == total {
                        self.ui.invalidate();
                        self.thumbs.retry_failed();
                    }
                }
                fp_library::WorkerEvent::Idle { .. } => {
                    if self.ui.worker_status.take().is_some() {
                        self.ui.invalidate();
                        self.thumbs.retry_failed();
                    }
                }
                fp_library::WorkerEvent::Error(e) => log::warn!("metadata worker: {e}"),
                _ => {}
            }
        }
    }

    /// Mounts cards and drives SteamOS leaves unmounted (exFAT/FAT/NTFS
    /// microSD cards, any USB drive), once per insertion.
    pub(super) fn mount_drives(&mut self) {
        let drives = crate::drives::unmounted_drives();
        // Forget drives that were removed (or mounted), so they are tried
        // again when inserted again.
        self.mount_tried
            .retain(|d| drives.iter().any(|u| &u.device == d));
        for d in drives {
            if self.mount_tried.contains(&d.device) {
                continue;
            }
            log::info!("mounting /dev/{} ({})", d.device, d.fs_type);
            self.mount_tried.push(d.device.clone());
            self.jobs.spawn("mount", move || {
                let r = crate::drives::mount_drive(&d);
                Job::Mounted(d.device, r)
            });
        }
    }

    /// Notices microSD cards and USB drives being inserted or removed.
    pub(super) fn check_mounts(&mut self) {
        if self.mounts_checked.elapsed() < Duration::from_secs(3) {
            return;
        }
        self.mounts_checked = Instant::now();
        self.mount_drives();
        let now = crate::drives::removable_mounts();
        if now == self.mounts {
            return;
        }
        let added: Vec<_> = now
            .iter()
            .filter(|m| !self.mounts.contains(m))
            .cloned()
            .collect();
        self.mounts = now;
        if let Some(m) = added.first() {
            let name = m
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            log::info!("drive mounted: {}", m.display());
            self.ui.toast(format!("Found drive \"{name}\""));
            if self.settings.index_removable {
                self.rescan(false);
            }
        } else {
            log::info!("a drive was removed");
        }
        self.ui.invalidate();
        if let Some(b) = self
            .ui
            .folder_mut()
            .filter(|b| b.source == crate::services::REMOVABLE_SOURCE)
        {
            let loc = b.location.clone();
            self.browse(crate::services::REMOVABLE_SOURCE.into(), loc);
        }
    }

    pub(super) fn open_request(&mut self, req: OpenRequest) {
        if let Some(source) = self.locked_source_of(&req) {
            self.ui
                .toast(format!("{} is locked", self.source_name(&source)));
            self.ask_pin(&source);
            return;
        }
        // Close the current video first: it holds a hardware decoder session,
        // and with Steam holding others the new video could not get one, fell
        // back to software and stuttered ("Hardware decoder busy"). It
        // closes on a thread of its own; the open job waits for it.
        if let Some(p) = self.playback.take() {
            p.close(&self.services.library);
            if let Some(h) = self.services.haptics() {
                h.clear();
            }
            self.ui.invalidate();
        }
        self.open_seq += 1;
        self.open_started = Instant::now();
        let lib = self.services.library.clone();
        let seq = self.open_seq;
        let opener = self.services.opener.clone();
        let settings = self.settings.clone();
        let reports = self.services.watch.clone();
        self.ui.opening = Some(
            req.entry
                .as_ref()
                .map(|e| e.name.clone())
                .unwrap_or_else(|| req.location.rsplit('/').next().unwrap_or("").to_string()),
        );
        self.ui.open_error = None;
        self.services.set_worker_busy(true);
        self.jobs.spawn("open", move || {
            let opened = playback::open(req, &opener, &lib, &settings, &reports);
            Job::Opened(seq, Box::new(opened))
        });
    }

    /// Gives up on a video that has been opening for [`OPEN_TIMEOUT`]: says
    /// so, and closes it if it opens after all (its answer is stale then).
    pub(super) fn give_up_slow_open(&mut self) {
        if self.ui.opening.is_none() || self.open_started.elapsed() < OPEN_TIMEOUT {
            return;
        }
        let title = self.ui.opening.take().unwrap_or_default();
        log::warn!("gave up opening {title} after {} s", OPEN_TIMEOUT.as_secs());
        self.open_seq += 1;
        self.ui.open_error = Some(format!(
            "{title}: the server sent no video in {} seconds",
            OPEN_TIMEOUT.as_secs()
        ));
        self.ui.invalidate();
    }
}
