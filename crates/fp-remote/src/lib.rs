//! Remote control of FramePlayerX from other devices on the LAN.
//!
//! Two servers, both off until the app starts them through [`RemoteHub`]:
//!
//! - a DeoVR-compatible TCP API ([`deovr`], default port 23554) so script
//!   players and haptics tools that already speak DeoVR's protocol (ohdoki,
//!   MultiFunPlayer, ScriptPlayer) work unchanged;
//! - a web remote ([`web`], default port 8790): an embedded single-page app
//!   for phones with the player's controls and text entry.
//!   A phone pairs by entering the short code the headset shows
//!   ([`RemoteHub::pairing_code`]), which gives it a per-install token.
//!
//! The render thread calls [`RemoteHub::publish`] every frame and drains
//! [`RemoteHub::events`] for [`RemoteEvent`]s. Only loopback and
//! private/link-local peers are served ([`is_allowed_peer`]).
//!
//! All servers use plain threads and blocking sockets.

pub mod config;
pub mod deovr;
mod details;
mod error;
mod hub;
pub mod net;
mod pairing;
mod status;
pub mod token;
pub mod web;

pub use config::{DEFAULT_DEOVR_PORT, DEFAULT_WEB_PORT, RemoteConfig};
pub use details::{RemoteAudio, RemoteDetails, RemoteMarker, RemoteSubtitles};
pub use error::RemoteError;
pub use hub::{RemoteEvent, RemoteHub};
pub use net::{is_allowed_peer, lan_addresses};
pub use status::significant_change;
