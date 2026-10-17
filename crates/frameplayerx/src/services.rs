//! Long-lived subsystems: media sources, the library and its metadata
//! worker, haptic devices and the remote-control servers.

use crate::drives::REMOVABLE_ROOT;
use crate::settings::{HapticDeviceConfig, Settings};
use fp_core::ByteSource;
use fp_core::source::FileSource;
use fp_core::sync::{read, write};
use fp_haptics::HapticsEngine;
use fp_library::Library;
use fp_remote::{RemoteEvent, RemoteHub, RemoteItem, RemoteLibrary};
use fp_sources::{Source, SourceConfig};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// A configured source that could not be built.
#[derive(Debug)]
pub struct SourceError {
    pub id: String,
    pub name: String,
    pub error: fp_sources::Error,
}

impl std::fmt::Display for SourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.name, self.error)
    }
}

/// Resolves locations to readable byte sources: local paths directly,
/// everything else through the configured source that owns it.
pub struct Opener {
    sources: RwLock<HashMap<String, Arc<dyn Source>>>,
    http: fp_sources::http::HttpClient,
}

impl Opener {
    pub fn new() -> Opener {
        Opener {
            sources: RwLock::new(HashMap::new()),
            http: fp_sources::http::HttpClient::new(Default::default(), None),
        }
    }

    pub fn set_sources(&self, configs: &[SourceConfig]) -> Vec<SourceError> {
        let mut map = HashMap::new();
        let mut errors = Vec::new();
        for c in builtin_sources().iter().chain(configs) {
            match fp_sources::build(c) {
                Ok(s) => {
                    map.insert(c.id().to_string(), Arc::<dyn Source>::from(s));
                }
                Err(error) => errors.push(SourceError {
                    id: c.id().to_string(),
                    name: c.name().to_string(),
                    error,
                }),
            }
        }
        *write(&self.sources) = map;
        errors
    }

    pub fn source(&self, id: &str) -> Option<Arc<dyn Source>> {
        read(&self.sources).get(id).cloned()
    }

    /// Opens `location`, preferring the source `source_id` when given.
    pub fn open_in(
        &self,
        source_id: Option<&str>,
        location: &str,
    ) -> Result<Arc<dyn ByteSource>, String> {
        if location.starts_with('/') {
            let path = std::path::Path::new(location);
            if !path.exists() && location.starts_with(&format!("{REMOVABLE_ROOT}/")) {
                return Err(
                    "This video is on a microSD card or USB drive that is not inserted.".into(),
                );
            }
            return FileSource::open(path)
                .map(|f| Arc::new(f) as Arc<dyn ByteSource>)
                .map_err(|e| format!("{location}: {e}"));
        }
        if let Some(path) = location.strip_prefix("file://") {
            return self.open_in(None, path);
        }
        if let Some(src) = source_id.and_then(|id| self.source(id)) {
            return src.open(location).map_err(|e| e.to_string());
        }
        // Try sources whose scheme fits, then plain HTTP.
        let sources: Vec<Arc<dyn Source>> = read(&self.sources).values().cloned().collect();
        let mut last = String::from("no source can open this location");
        for s in sources {
            match s.open(location) {
                Ok(b) => return Ok(b),
                Err(e) => last = e.to_string(),
            }
        }
        if fp_sources::urlutil::is_http_url(location) {
            return self
                .http
                .open(location)
                .map(|f| Arc::new(f) as Arc<dyn ByteSource>)
                .map_err(|e| e.to_string());
        }
        Err(last)
    }

    pub fn open(&self, location: &str) -> Result<Arc<dyn ByteSource>, String> {
        self.open_in(None, location)
    }

    /// Small downloads (thumbnails, scripts) from any http(s) URL.
    pub fn fetch(&self, url: &str, limit: u64) -> Result<Vec<u8>, String> {
        if url.starts_with('/') {
            return std::fs::read(url).map_err(|e| e.to_string());
        }
        self.http.get_bytes(url, limit).map_err(|e| e.to_string())
    }
}

impl Default for Opener {
    fn default() -> Self {
        Opener::new()
    }
}

/// Sources that always exist: the home folder and removable drives.
pub fn builtin_sources() -> Vec<SourceConfig> {
    let mut out = Vec::new();
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        out.push(SourceConfig::Local(fp_sources::LocalConfig {
            id: HOME_SOURCE.into(),
            name: "This device".into(),
            root: home,
        }));
    }
    // Always offered, so a card inserted later shows up without a restart.
    out.push(SourceConfig::Local(fp_sources::LocalConfig {
        id: REMOVABLE_SOURCE.into(),
        name: "microSD and USB drives".into(),
        root: PathBuf::from(REMOVABLE_ROOT),
    }));
    out
}

/// The source id the library's folder scans record videos under.
pub const LIBRARY_SOURCE: &str = "local";
pub const HOME_SOURCE: &str = "device-home";
pub const REMOVABLE_SOURCE: &str = "device-removable";
/// The headset as one place to browse: its indexed videos, and the home
/// folder and removable drives (not a [`Source`] of its own).
pub const LOCAL_HUB: &str = "on-device";
/// Video folders worth offering for the library that are not in it yet
/// (removable drives are indexed automatically).
pub fn suggested_folders(existing: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        for d in ["Videos", "Downloads", "Movies"] {
            out.push(home.join(d));
        }
    }
    out.retain(|p| p.is_dir() && !existing.iter().any(|e| p.starts_with(e)));
    out
}

pub struct Services {
    pub library: Library,
    pub opener: Arc<Opener>,
    pub source_configs: Vec<SourceConfig>,
    pub worker: Option<fp_library::MetadataWorker>,
    pub worker_events: Option<std::sync::mpsc::Receiver<fp_library::WorkerEvent>>,
    /// Holds the metadata worker off (see [`crate::prober::FfmpegProber`]).
    worker_busy: Arc<std::sync::atomic::AtomicBool>,
    pub haptics: HapticsEngine,
    /// The configured devices that were tried, in the order they answered.
    haptic_devices: Vec<HapticDeviceSlot>,
    pub remote: Option<RemoteHub>,
    pub remote_error: Option<String>,
}

/// A configured haptic device and how connecting it went.
pub struct HapticDeviceSlot {
    pub config: HapticDeviceConfig,
    /// Its id in the haptics engine, once connected.
    pub id: Option<fp_haptics::DeviceId>,
    /// Why it could not connect.
    pub error: Option<String>,
}

/// Library access for the web remote.
struct WebLibrary(Library);

impl RemoteLibrary for WebLibrary {
    fn search(&self, query: &str, limit: usize, offset: usize) -> Vec<RemoteItem> {
        let q = fp_library::Query {
            limit: Some(limit),
            offset,
            ..fp_library::Query::text(query)
        };
        self.0
            .search(&q)
            .unwrap_or_default()
            .into_iter()
            .map(|r| RemoteItem {
                id: r.id.0,
                title: r.title.clone(),
                location: r.location.clone(),
                duration: r.duration,
                format_label: r.effective_format().format.label(),
                has_thumbnail: r.thumbnail.is_some(),
            })
            .collect()
    }

    fn thumbnail_path(&self, id: i64) -> Option<PathBuf> {
        let r = self.0.get(fp_library::MediaId(id)).ok().flatten()?;
        // Not for videos of a source locked with a PIN.
        if self.0.hidden().hides(&r.source_id, &r.location) {
            return None;
        }
        r.thumbnail
    }
}

impl Services {
    pub fn new(settings: &Settings, library: Library) -> Services {
        let opener = Arc::new(Opener::new());
        let source_configs = match fp_sources::load_configs(&Settings::sources_path()) {
            Ok(c) => c,
            Err(e) => {
                if Settings::sources_path().exists() {
                    log::warn!("sources: {e}");
                }
                Vec::new()
            }
        };
        for e in opener.set_sources(&source_configs) {
            log::warn!("source unavailable: {e}");
        }
        let haptics = HapticsEngine::new();
        haptics.set_settings(settings.haptics.clone());
        let mut s = Services {
            library,
            opener,
            source_configs,
            worker: None,
            worker_events: None,
            worker_busy: Default::default(),
            haptics,
            haptic_devices: Vec::new(),
            remote: None,
            remote_error: None,
        };
        s.start_worker();
        s.apply_remote(&settings.remote);
        s
    }

    pub fn start_worker(&mut self) {
        let prober = Arc::new(crate::prober::FfmpegProber {
            opener: self.opener.clone(),
            busy: self.worker_busy.clone(),
        });
        match fp_library::MetadataWorker::spawn(
            self.library.clone(),
            prober,
            fp_library::WorkerOptions::default(),
        ) {
            Ok((w, rx)) => {
                self.worker = Some(w);
                self.worker_events = Some(rx);
            }
            Err(e) => log::error!("metadata worker: {e}"),
        }
    }

    pub fn wake_worker(&self) {
        if let Some(w) = &self.worker {
            w.wake();
        }
    }

    /// Holds the metadata worker off while a video plays (or while
    /// quitting), cutting short the thumbnail it is making; it picks up
    /// where it left off once not busy.
    pub fn set_worker_busy(&self, busy: bool) {
        use std::sync::atomic::Ordering;
        if self.worker_busy.swap(busy, Ordering::AcqRel) && !busy {
            self.wake_worker();
        }
    }

    /// The configuration of source `id`, if the user added it.
    pub fn source_config(&self, id: &str) -> Option<&SourceConfig> {
        self.source_configs.iter().find(|c| c.id() == id)
    }

    /// Persists the source list (mode 0600) and rebuilds the sources.
    pub fn set_sources(&mut self, configs: Vec<SourceConfig>) -> Vec<SourceError> {
        if let Err(e) = fp_sources::save_configs(&Settings::sources_path(), &configs) {
            log::error!("saving sources: {e}");
        }
        let errors = self.opener.set_sources(&configs);
        self.source_configs = configs;
        errors
    }

    /// Connects one haptic device (blocking: call from a job thread).
    pub fn connect_device(cfg: &HapticDeviceConfig) -> Result<Box<dyn fp_haptics::Device>, String> {
        use fp_haptics::{buttplug, handy, tcode};
        fn boxed<D: fp_haptics::Device + 'static>(
            r: Result<D, impl std::fmt::Display>,
        ) -> Result<Box<dyn fp_haptics::Device>, String> {
            r.map(|d| Box::new(d) as Box<dyn fp_haptics::Device>)
                .map_err(|e| e.to_string())
        }
        match cfg {
            HapticDeviceConfig::Tcode { endpoint } => {
                let endpoint = tcode::TcodeEndpoint::parse(endpoint).map_err(|e| e.to_string())?;
                boxed(tcode::TcodeDevice::connect(tcode::TcodeConfig {
                    endpoint,
                    ..Default::default()
                }))
            }
            HapticDeviceConfig::Buttplug { url } => boxed(buttplug::ButtplugDevice::connect(
                buttplug::ButtplugConfig {
                    url: url.clone(),
                    client_name: "FramePlayerX".into(),
                    ..Default::default()
                },
            )),
            HapticDeviceConfig::Handy { key } => {
                boxed(handy::HandyDevice::connect(handy::HandyConfig {
                    connection_key: key.clone(),
                    ..Default::default()
                }))
            }
        }
    }

    /// Records how connecting `config` went, replacing an earlier attempt
    /// (and its device in the engine).
    pub fn device_connected(
        &mut self,
        config: HapticDeviceConfig,
        result: Result<Box<dyn fp_haptics::Device>, String>,
    ) {
        self.remove_device(&config);
        let (id, error) = match result {
            Ok(d) => (Some(self.haptics.add_device(d)), None),
            Err(e) => (None, Some(e)),
        };
        self.haptic_devices
            .push(HapticDeviceSlot { config, id, error });
    }

    /// Disconnects `config`'s device and forgets how connecting it went.
    pub fn remove_device(&mut self, config: &HapticDeviceConfig) {
        self.haptic_devices.retain(|s| {
            let same = s.config == *config;
            if same && let Some(id) = s.id {
                self.haptics.remove_device(id);
            }
            !same
        });
    }

    /// Disconnects every device.
    pub fn remove_devices(&mut self) {
        for s in self.haptic_devices.drain(..) {
            if let Some(id) = s.id {
                self.haptics.remove_device(id);
            }
        }
    }

    /// How connecting `config` went; `None` while it is being tried.
    pub fn haptic_device(&self, config: &HapticDeviceConfig) -> Option<&HapticDeviceSlot> {
        self.haptic_devices.iter().find(|s| s.config == *config)
    }

    /// Starts or stops the remote-control servers to match `config`.
    pub fn apply_remote(&mut self, config: &fp_remote::config::RemoteConfig) {
        if let Some(r) = self.remote.take() {
            r.stop();
        }
        self.remote_error = None;
        if !config.deovr_enabled && !config.web_enabled {
            return;
        }
        let mut cfg = config.clone();
        if cfg.token_path.is_none() {
            cfg.token_path = Some(fp_remote::config::RemoteConfig::default_token_path());
        }
        match RemoteHub::new(cfg.clone()) {
            Ok(hub) => {
                if cfg.deovr_enabled
                    && let Err(e) = hub.start_deovr_server()
                {
                    self.remote_error = Some(format!("DeoVR remote API: {e}"));
                }
                if cfg.web_enabled
                    && let Err(e) = hub.start_web(Arc::new(WebLibrary(self.library.clone())))
                {
                    self.remote_error = Some(format!("web remote: {e}"));
                }
                self.remote = Some(hub);
            }
            Err(e) => self.remote_error = Some(e.to_string()),
        }
    }

    pub fn remote_events(&self) -> Vec<RemoteEvent> {
        self.remote
            .as_ref()
            .map(|r| r.events().try_iter().collect())
            .unwrap_or_default()
    }
}
