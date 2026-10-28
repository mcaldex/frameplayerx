//! [`RemoteHub`]: the app-facing entry point that owns both servers.

use crate::deovr::DeovrServer;
use crate::details::{DetailsCell, RemoteDetails};
use crate::pairing::{Entered, Pairing};
use crate::status::StatusCell;
use crate::web::WebServer;
use crate::{RemoteConfig, RemoteError, token};
use crossbeam_channel::{Receiver, Sender, TrySendError};
use fp_core::sync::{lock, read, write};
use fp_core::{PlaybackStatus, PlayerCommand};
use std::collections::BTreeMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex, RwLock};

/// Something a remote client asked the player to do.
#[derive(Clone, Debug, PartialEq)]
pub enum RemoteEvent {
    /// A playback command (from either server).
    Command(PlayerCommand),
    /// Text typed on the web remote, for whatever text field is focused in
    /// the headset (library search, URL entry).
    Text(String),
}

/// Capacity of the event queue; when the app stops draining it, further
/// events are dropped (and logged) instead of growing without bound.
const EVENT_QUEUE: usize = 1024;

/// The web remote's wording, in the app's language: its language tag and
/// its strings by id.
#[derive(Clone, Debug, Default)]
pub(crate) struct PageText {
    pub(crate) lang: String,
    pub(crate) strings: BTreeMap<String, String>,
}

/// State shared by the hub and every server thread.
pub(crate) struct Shared {
    pub(crate) status: StatusCell,
    pub(crate) details: DetailsCell,
    pub(crate) token: RwLock<String>,
    pub(crate) page: RwLock<PageText>,
    pairing: Mutex<Pairing>,
    events: Sender<RemoteEvent>,
}

impl Shared {
    pub(crate) fn emit(&self, ev: RemoteEvent) {
        match self.events.try_send(ev) {
            Ok(()) | Err(TrySendError::Disconnected(_)) => {}
            Err(TrySendError::Full(ev)) => log::warn!("remote event queue full, dropping {ev:?}"),
        }
    }

    pub(crate) fn token(&self) -> String {
        read(&self.token).clone()
    }

    /// A code typed on a phone's pairing page.
    pub(crate) fn enter_code(&self, typed: &str) -> Entered {
        lock(&self.pairing).enter(typed)
    }
}

/// Owns the DeoVR API server and the web remote, the shared playback
/// status they report, and the queue of commands they receive.
///
/// Nothing listens until [`start_deovr_server`](Self::start_deovr_server)
/// or [`start_web`](Self::start_web) is called. Dropping the hub stops both.
pub struct RemoteHub {
    config: RemoteConfig,
    shared: Arc<Shared>,
    events: Receiver<RemoteEvent>,
    deovr: Mutex<Option<DeovrServer>>,
    web: Mutex<Option<WebServer>>,
}

impl RemoteHub {
    /// Creates the hub and loads (or creates and persists) the pairing token
    /// at `config.token_path`. Does not open any socket.
    pub fn new(config: RemoteConfig) -> Result<RemoteHub, RemoteError> {
        let token = match &config.token_path {
            Some(path) => token::load_or_create(path)?,
            None => token::generate_token()?,
        };
        let (tx, rx) = crossbeam_channel::bounded(EVENT_QUEUE);
        Ok(RemoteHub {
            config,
            shared: Arc::new(Shared {
                status: StatusCell::default(),
                details: DetailsCell::default(),
                token: RwLock::new(token),
                page: RwLock::new(PageText::default()),
                pairing: Mutex::new(Pairing::default()),
                events: tx,
            }),
            events: rx,
            deovr: Mutex::new(None),
            web: Mutex::new(None),
        })
    }

    /// The configuration the hub was created with.
    pub fn config(&self) -> &RemoteConfig {
        &self.config
    }

    /// The web remote's text in the app's language (`lang`, a language
    /// tag): each `{{id:English}}` of its pages, and the strings its script
    /// shows. A page open in another language reloads itself.
    pub fn set_page_text(&self, lang: &str, strings: impl IntoIterator<Item = (String, String)>) {
        *write(&self.shared.page) = PageText {
            lang: lang.to_string(),
            strings: strings.into_iter().collect(),
        };
    }

    /// Starts the DeoVR-compatible TCP API on `bind_address:deovr_port` and
    /// returns the bound address (useful with port 0).
    pub fn start_deovr_server(&self) -> Result<SocketAddr, RemoteError> {
        let mut slot = lock(&self.deovr);
        if slot.is_some() {
            return Err(RemoteError::AlreadyRunning("DeoVR"));
        }
        let bind = SocketAddr::new(self.config.bind_address, self.config.deovr_port);
        let server = DeovrServer::start(
            bind,
            self.shared.clone(),
            self.config.max_deovr_clients.max(1),
        )?;
        let addr = server.addr();
        *slot = Some(server);
        Ok(addr)
    }

    /// Starts the web remote on `bind_address:web_port` and returns the
    /// bound address.
    pub fn start_web(&self) -> Result<SocketAddr, RemoteError> {
        let mut slot = lock(&self.web);
        if slot.is_some() {
            return Err(RemoteError::AlreadyRunning("web remote"));
        }
        let bind = SocketAddr::new(self.config.bind_address, self.config.web_port);
        let server = WebServer::start(
            bind,
            self.shared.clone(),
            self.config.max_event_streams.max(1),
        )?;
        let addr = server.addr();
        *slot = Some(server);
        Ok(addr)
    }

    /// Stops the DeoVR API server and disconnects its clients.
    pub fn stop_deovr_server(&self) {
        if let Some(mut s) = lock(&self.deovr).take() {
            s.stop();
        }
    }

    /// Stops the web remote. Open live-status streams end within ~100 ms.
    pub fn stop_web(&self) {
        if let Some(mut s) = lock(&self.web).take() {
            s.stop();
        }
    }

    /// Stops both servers.
    pub fn stop(&self) {
        self.stop_deovr_server();
        self.stop_web();
    }

    /// Records the current playback status. Cheap enough to call every
    /// frame: it compares, copies into existing buffers and never blocks on
    /// the network. The servers pick changes up on their own threads.
    pub fn publish(&self, status: &PlaybackStatus) {
        self.shared.status.publish(status);
    }

    /// Records what the web remote shows of the open video beyond its
    /// status ([`RemoteDetails::default`] when none is open). Compares
    /// first, so calling it often costs only that.
    pub fn publish_details(&self, details: &RemoteDetails) {
        self.shared.details.publish(details);
    }

    /// The command/text queue. Each receiver clone competes for events, so
    /// drain it from one place (the main loop) with `try_recv`.
    pub fn events(&self) -> Receiver<RemoteEvent> {
        self.events.clone()
    }

    /// Address the DeoVR API is bound to, while running.
    pub fn deovr_addr(&self) -> Option<SocketAddr> {
        lock(&self.deovr).as_ref().map(DeovrServer::addr)
    }

    /// Address the web remote is bound to, while running.
    pub fn web_addr(&self) -> Option<SocketAddr> {
        lock(&self.web).as_ref().map(WebServer::addr)
    }

    /// Connected DeoVR API clients (script players and similar tools).
    pub fn deovr_client_count(&self) -> usize {
        lock(&self.deovr)
            .as_ref()
            .map_or(0, DeovrServer::client_count)
    }

    /// Web remote pages currently open (live-status streams).
    pub fn web_client_count(&self) -> usize {
        lock(&self.web).as_ref().map_or(0, WebServer::client_count)
    }

    /// The current pairing token. Treat as a secret.
    pub fn token(&self) -> String {
        self.shared.token()
    }

    /// Replaces the pairing token (persisting it when a path is configured),
    /// which unpairs every phone. Returns the new token.
    pub fn regenerate_token(&self) -> Result<String, RemoteError> {
        let new = token::generate_token()?;
        if let Some(path) = &self.config.token_path {
            token::store(path, &new)?;
        }
        *write(&self.shared.token) = new.clone();
        Ok(new)
    }

    /// The address to open on a phone, `<lan_ip>:<port>`, using the bound
    /// port while the web remote runs, else the configured one.
    pub fn pairing_address(&self, lan_ip: impl Into<IpAddr>) -> String {
        let port = self.web_addr().map_or(self.config.web_port, |a| a.port());
        match lan_ip.into() {
            IpAddr::V4(v4) => format!("{v4}:{port}"),
            IpAddr::V6(v6) => format!("[{v6}]:{port}"),
        }
    }

    /// The code the headset shows for a phone to pair with: kept while it
    /// is shown (call this each time it is drawn), and replaced once a
    /// phone pairs with it or ten minutes after it was last shown. `None`
    /// after five wrong codes, until [`new_pairing_code`](Self::new_pairing_code).
    pub fn pairing_code(&self) -> Result<Option<String>, RemoteError> {
        lock(&self.shared.pairing).show()
    }

    /// Starts pairing again after wrong codes stopped it, with a new code.
    pub fn new_pairing_code(&self) {
        lock(&self.shared.pairing).restart();
    }
}

impl Drop for RemoteHub {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[test]
    fn pairing_address_and_token_rotation() {
        let hub = RemoteHub::new(RemoteConfig::default()).unwrap();
        let t = hub.token();
        assert_eq!(
            hub.pairing_address(Ipv4Addr::new(192, 168, 1, 9)),
            "192.168.1.9:8790"
        );
        let v6: IpAddr = "fe80::1".parse().unwrap();
        assert_eq!(hub.pairing_address(v6), "[fe80::1]:8790");
        let t2 = hub.regenerate_token().unwrap();
        assert_ne!(t, t2);
        assert_eq!(hub.token(), t2);
        assert_eq!(hub.deovr_client_count(), 0);
        assert_eq!(hub.web_client_count(), 0);
        assert!(hub.deovr_addr().is_none());
    }
}
