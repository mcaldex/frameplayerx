//! Library managers that publish VR scenes as JSON: the DeoVR feed
//! (`/deovr` in XBVR, Stash plugins, VR video sites) and the HereSphere
//! JSON API (`/heresphere` in XBVR and Stash).
//!
//! Both map to the same shape: the root lists scene groups as folders, a
//! group lists its scenes as videos, and [`Source::details`] fetches one
//! scene's full description (declared projection and stereo layout,
//! scripts, subtitles, markers). [`Source::open`] resolves a scene to its
//! best media URL (the tallest encoding not above
//! [`FeedConfig::max_height`]) and reads it over HTTP.
//!
//! Parsing is tolerant: numbers may arrive as strings, unknown fields are
//! ignored and missing ones fall back to sensible defaults.
//!
//! Locations: group folders are `<feed url>#group=<encoded name>`, scenes
//! are their scene/API URL, scripts and subtitles their direct URL.

use crate::config::{Credentials, FeedConfig, SourceKind};
use crate::error::{Error, Result};
use crate::http::{HttpClient, HttpOptions, HttpReply};
use crate::httpdir::clean_url_and_credentials;
use crate::json::{array, boolean, get, num, text, uint};
use crate::sidecar::Sidecars;
use crate::urlutil::{
    encode_form_value, encode_segment, is_http_url, last_segment, percent_decode, redact, resolve,
};
use crate::{Source, dir_entry};
use fp_core::format::{Projection, StereoLayout, VideoFormat, detect_from_name};
use fp_core::source::{
    ByteSource, Entry, EntryKind, is_script_name, is_subtitle_name, is_video_name,
};
use fp_core::sync::lock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Largest JSON document accepted from a feed.
const JSON_LIMIT: u64 = 64 << 20;

/// Parallel requests when a HereSphere group is listed.
const DETAIL_WORKERS: usize = 6;

/// One encoding of a scene.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MediaSource {
    /// Absolute media URL.
    pub url: String,
    /// Frame height in pixels, when known.
    pub height: Option<u32>,
    /// Frame width in pixels, when known.
    pub width: Option<u32>,
    /// Encoding name (`h265`, `h264`, ...).
    pub label: String,
    /// File size, when the feed says.
    pub size: Option<u64>,
}

impl MediaSource {
    /// A copy the server transcodes as it streams (stash-vr's "transcoding"
    /// encoding) rather than the file itself: it can't seek, and a server
    /// transcoding a large file may not send a frame for minutes.
    pub fn is_transcode(&self) -> bool {
        self.label.to_ascii_lowercase().contains("transcod")
    }
}

/// Everything a feed says about one scene.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SceneInfo {
    /// Scene title.
    pub title: String,
    /// Length in seconds.
    pub duration: Option<f64>,
    /// Poster image URL.
    pub thumbnail_url: Option<String>,
    /// Projection and stereo layout the feed declares.
    pub format: Option<VideoFormat>,
    /// Available encodings.
    pub sources: Vec<MediaSource>,
    /// Haptic script URLs.
    pub scripts: Vec<String>,
    /// Subtitle URLs.
    pub subtitles: Vec<String>,
    /// Named timestamps in seconds.
    pub markers: Vec<(f64, String)>,
    /// Synopsis.
    pub description: Option<String>,
    /// Release date, Unix seconds.
    pub released: Option<i64>,
    /// When the server added it, Unix seconds.
    pub added: Option<i64>,
    /// Studio or site.
    pub studio: Option<String>,
    /// Performers.
    pub actors: Vec<String>,
    /// Categories and tags.
    pub tags: Vec<String>,
    /// Rating out of 5.
    pub rating: Option<f32>,
    /// The user's favourite on the server.
    pub favorite: bool,
}

impl SceneInfo {
    /// The tallest source not above `max_height`; when every source is
    /// taller, the smallest one. Sources of unknown height rank lowest. The
    /// files themselves come before the server's transcodes of them, which
    /// are only picked when no file fits (or there is none).
    pub fn best_source(&self, max_height: Option<u32>) -> Option<&MediaSource> {
        let h = |s: &MediaSource| s.height.unwrap_or(0);
        let fits = |s: &&MediaSource| max_height.is_none_or(|m| h(s) <= m);
        // `max_by_key` keeps the last of equal keys; reverse to prefer the
        // first listed encoding among equals.
        let tallest = |transcodes: bool| {
            self.sources
                .iter()
                .rev()
                .filter(|s| s.is_transcode() == transcodes)
                .filter(fits)
                .max_by_key(|s| h(s))
        };
        tallest(false)
            .or_else(|| tallest(true))
            .or_else(|| self.sources.iter().min_by_key(|s| (s.is_transcode(), h(s))))
    }

    /// Copies the scene's details into a listing entry (keeping its
    /// location).
    pub fn apply_to(&self, entry: &mut Entry) {
        if !self.title.is_empty() {
            entry.name = self.title.clone();
        }
        entry.kind = EntryKind::Video;
        entry.duration = self.duration.or(entry.duration);
        if self.thumbnail_url.is_some() {
            entry.thumbnail_url = self.thumbnail_url.clone();
        }
        entry.format = self
            .format
            .or_else(|| detect_from_name(&entry.name))
            .or(entry.format);
        if entry.size.is_none() {
            entry.size = self.best_source(None).and_then(|s| s.size);
        }
        entry.scripts = self.scripts.clone();
        entry.subtitles = self.subtitles.clone();
        entry.markers = self.markers.clone();
    }
}

fn url_field(v: &Value, key: &str, base: &str) -> Option<String> {
    text(v, key).and_then(|u| resolve(base, &u).ok())
}

/// A date given as Unix seconds (or milliseconds) or as an ISO 8601 string.
fn date_field(v: &Value, key: &str) -> Option<i64> {
    match get(v, key)? {
        Value::Number(n) => n
            .as_f64()
            .filter(|t| *t > 0.0)
            .map(|t| if t > 1e11 { t / 1000.0 } else { t } as i64),
        Value::String(s) => crate::timeutil::parse_iso8601(s),
        _ => None,
    }
}

/// A rating out of 5.
fn rating_field(v: &Value, key: &str) -> Option<f32> {
    num(v, key)
        .filter(|r| *r > 0.0 && *r <= 5.0)
        .map(|r| r as f32)
}

/// Names from `[{"name": ...}]` (or plain strings).
fn names(v: &Value, key: &str) -> Vec<String> {
    array(v, key)
        .iter()
        .filter_map(|x| text(x, "name").or_else(|| x.as_str().map(|s| s.trim().to_string())))
        .filter(|s| !s.is_empty())
        .collect()
}

fn media_source(v: &Value, label: &str, base: &str) -> Option<MediaSource> {
    let url = url_field(v, "url", base)?;
    let height = uint(v, "height").or_else(|| uint(v, "resolution"));
    Some(MediaSource {
        url,
        height,
        width: uint(v, "width"),
        label: label.to_string(),
        size: num(v, "size").filter(|s| *s > 0.0).map(|s| s as u64),
    })
}

// ---- DeoVR ----

/// Projection for a DeoVR `screenType`.
fn deovr_projection(screen_type: &str) -> Option<Projection> {
    Some(match screen_type.trim().to_ascii_lowercase().as_str() {
        "dome" | "180" => Projection::EQUIRECT_180,
        "sphere" | "360" => Projection::EQUIRECT_360,
        "flat" => Projection::Flat,
        "fisheye" => Projection::fisheye(180.0),
        "mkx200" => Projection::fisheye(200.0),
        "mkx220" | "vrca220" => Projection::fisheye(220.0),
        "rf52" => Projection::fisheye(190.0),
        _ => return None,
    })
}

fn stereo_from(mode: &str) -> Option<StereoLayout> {
    Some(match mode.trim().to_ascii_lowercase().as_str() {
        "sbs" | "lr" | "sidebyside" | "fsbs" => StereoLayout::SideBySide,
        "hsbs" | "half_sbs" | "half-sbs" | "halfsbs" => StereoLayout::HalfSideBySide,
        "tb" | "ou" | "topbottom" | "fou" | "ftb" => StereoLayout::TopBottom,
        "hou" | "htb" | "half_tb" | "half-tb" | "half_ou" | "half-ou" | "halfou" | "halftb" => {
            StereoLayout::HalfTopBottom
        }
        "off" | "mono" | "none" | "" => StereoLayout::Mono,
        _ => return None,
    })
}

/// Declared format of a DeoVR scene, `None` when it says nothing.
fn deovr_format(v: &Value) -> Option<VideoFormat> {
    let is3d = boolean(v, "is3d");
    let screen = text(v, "screenType").and_then(|s| deovr_projection(&s));
    let stereo = text(v, "stereoMode").and_then(|s| stereo_from(&s));
    if is3d.is_none() && screen.is_none() && stereo.is_none() {
        return None;
    }
    let (projection, stereo) = match is3d {
        Some(false) if screen.is_none() => (Projection::Flat, stereo.unwrap_or_default()),
        _ => (
            screen.unwrap_or(Projection::EQUIRECT_180),
            stereo.unwrap_or(if is3d == Some(false) {
                StereoLayout::Mono
            } else {
                StereoLayout::SideBySide
            }),
        ),
    };
    Some(VideoFormat::new(projection, stereo))
}

/// Parses a DeoVR scene document fetched from `base`.
pub fn parse_deovr_scene(v: &Value, base: &str) -> SceneInfo {
    let mut sources = Vec::new();
    for enc in array(v, "encodings") {
        let label = text(enc, "name").unwrap_or_default();
        for s in array(enc, "videoSources") {
            if let Some(m) = media_source(s, &label, base) {
                sources.push(m);
            }
        }
    }
    // Some feeds put a single URL on the scene itself.
    if sources.is_empty()
        && let Some(url) =
            url_field(v, "videoUrl", base).or_else(|| url_field(v, "video_url", base))
        && is_video_name(&last_segment(&url))
    {
        sources.push(MediaSource {
            url,
            ..MediaSource::default()
        });
    }
    let mut markers: Vec<(f64, String)> = array(v, "timeStamps")
        .iter()
        .filter_map(|t| Some((num(t, "ts")?, text(t, "name").unwrap_or_default())))
        .collect();
    markers.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Categories are `{"tag": {"name"}}` in XBVR, `{"name"}` elsewhere.
    let tags = array(v, "categories")
        .iter()
        .filter_map(|c| {
            get(c, "tag")
                .and_then(|t| text(t, "name"))
                .or_else(|| text(c, "name"))
                .or_else(|| c.as_str().map(|s| s.trim().to_string()))
        })
        .filter(|s| !s.is_empty())
        .collect();
    SceneInfo {
        title: text(v, "title").unwrap_or_default(),
        duration: num(v, "videoLength").filter(|d| *d > 0.0),
        thumbnail_url: url_field(v, "thumbnailUrl", base),
        format: deovr_format(v),
        sources,
        scripts: array(v, "fleshlight")
            .iter()
            .filter_map(|s| url_field(s, "url", base))
            .collect(),
        subtitles: array(v, "subtitles")
            .iter()
            .filter_map(|s| url_field(s, "url", base))
            .collect(),
        markers,
        description: text(v, "description"),
        released: date_field(v, "date").or_else(|| date_field(v, "releaseDate")),
        added: None,
        studio: get(v, "paysite")
            .and_then(|p| text(p, "name"))
            .or_else(|| text(v, "studio")),
        actors: names(v, "actors"),
        tags,
        rating: rating_field(v, "rating_avg").or_else(|| rating_field(v, "rating")),
        favorite: boolean(v, "isFavorite").unwrap_or(false),
    }
}

/// A group of scenes in a feed's root.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneGroup {
    /// Group name ("Recent", "Favourites", ...).
    pub name: String,
    /// Scenes, as video entries located at their scene URL.
    pub scenes: Vec<Entry>,
}

/// Parses a DeoVR feed root (`{"scenes":[{"name","list":[...]}]}`).
pub fn parse_deovr_root(v: &Value, base: &str) -> Vec<SceneGroup> {
    array(v, "scenes")
        .iter()
        .enumerate()
        .map(|(i, g)| SceneGroup {
            name: text(g, "name").unwrap_or_else(|| format!("Group {}", i + 1)),
            scenes: array(g, "list")
                .iter()
                .filter_map(|s| {
                    let url = url_field(s, "video_url", base)
                        .or_else(|| url_field(s, "videoUrl", base))?;
                    let title = text(s, "title").unwrap_or_else(|| last_segment(&url));
                    let mut e = Entry::new(title, url, EntryKind::Video);
                    e.duration = num(s, "videoLength").filter(|d| *d > 0.0);
                    e.thumbnail_url = url_field(s, "thumbnailUrl", base);
                    e.format = detect_from_name(&e.name);
                    Some(e)
                })
                .collect(),
        })
        .collect()
}

// ---- HereSphere ----

/// Declared format of a HereSphere video.
fn heresphere_format(v: &Value) -> Option<VideoFormat> {
    let projection = text(v, "projection")?;
    let fov = num(v, "fov")
        .filter(|f| *f > 0.0 && *f <= 360.0)
        .map(|f| f as f32);
    let lens = text(v, "lens").unwrap_or_default().to_ascii_uppercase();
    let projection = match projection.to_ascii_lowercase().as_str() {
        "equirectangular" => Projection::Equirect {
            h_fov: fov.unwrap_or(180.0),
            v_fov: 180.0,
        },
        "equirectangular360" => Projection::EQUIRECT_360,
        "fisheye" => match lens.as_str() {
            "MKX220" | "VRCA220" => Projection::fisheye(220.0),
            "MKX200" => Projection::fisheye(200.0),
            _ => Projection::fisheye(fov.unwrap_or(180.0)),
        },
        "perspective" => Projection::Flat,
        // fp-core has no plain cubemap; EAC is the closest renderer.
        "cubemap" | "equiangularcubemap" => Projection::Eac { h_fov: 360.0 },
        _ => return None,
    };
    let stereo = text(v, "stereo")
        .and_then(|s| stereo_from(&s))
        .unwrap_or_default();
    Some(VideoFormat::new(projection, stereo))
}

/// Parses a HereSphere video document fetched from `base`.
pub fn parse_heresphere_video(v: &Value, base: &str) -> SceneInfo {
    let mut sources = Vec::new();
    for m in array(v, "media") {
        let label = text(m, "name").unwrap_or_default();
        for s in array(m, "sources") {
            if let Some(src) = media_source(s, &label, base) {
                sources.push(src);
            }
        }
    }
    let mut markers = Vec::new();
    let mut meta = HereSphereTags::default();
    for t in array(v, "tags") {
        let Some(name) = text(t, "name") else {
            continue;
        };
        // Timed tags are chapters; the rest describe the video.
        let start = num(t, "start").filter(|s| *s > 0.0);
        let end = num(t, "end").filter(|e| *e > 0.0);
        match (start, end) {
            (None, None) => meta.add(&name),
            _ => markers.push((start.unwrap_or(0.0) / 1000.0, untagged(&name).to_string())),
        }
    }
    markers.sort_by(|a, b| a.0.total_cmp(&b.0));
    SceneInfo {
        title: text(v, "title").unwrap_or_default(),
        duration: num(v, "duration")
            .filter(|d| *d > 0.0)
            .map(|ms| ms / 1000.0),
        thumbnail_url: url_field(v, "thumbnailImage", base),
        format: heresphere_format(v),
        sources,
        scripts: array(v, "scripts")
            .iter()
            .filter_map(|s| url_field(s, "url", base))
            .collect(),
        subtitles: array(v, "subtitles")
            .iter()
            .filter_map(|s| url_field(s, "url", base))
            .collect(),
        markers,
        description: text(v, "description"),
        released: date_field(v, "dateReleased"),
        added: date_field(v, "dateAdded"),
        studio: meta.studio,
        actors: meta.actors,
        tags: meta.tags,
        rating: rating_field(v, "rating"),
        favorite: boolean(v, "isFavorite").unwrap_or(false),
    }
}

/// What HereSphere's prefixed tags (`Studio:`, `Talent:`, `Category:`...)
/// say about a video.
#[derive(Default)]
struct HereSphereTags {
    studio: Option<String>,
    actors: Vec<String>,
    tags: Vec<String>,
}

impl HereSphereTags {
    fn add(&mut self, tag: &str) {
        let (prefix, value) = match tag.split_once(':') {
            Some((p, v)) if !p.contains(' ') || p.eq_ignore_ascii_case("tag group") => {
                (p.trim().to_ascii_lowercase(), v.trim())
            }
            _ => (String::new(), tag.trim()),
        };
        if value.is_empty() {
            return;
        }
        match prefix.as_str() {
            "studio" | "site" => {
                self.studio.get_or_insert_with(|| value.to_string());
            }
            "talent" | "performer" | "actor" | "cast" => self.actors.push(value.to_string()),
            // Computed facts (resolution, frame rate...) and groupings.
            "feature" | "tag group" | "director" | "movie" | "group" => {}
            _ => self.tags.push(value.to_string()),
        }
    }
}

/// A tag's name without its `Prefix:`.
fn untagged(name: &str) -> &str {
    match name.split_once(':') {
        Some((p, v)) if !p.contains(' ') && !v.trim().is_empty() => v.trim(),
        _ => name,
    }
}

/// A scan's duration in seconds. A video's own document gives milliseconds,
/// but stash-vr's scan gives seconds (4893.57 for 4893570): a day or more
/// can only be milliseconds, anything less is read as seconds (milliseconds
/// would be a clip of under a minute and a half).
fn scan_seconds(d: f64) -> f64 {
    if d >= 86_400.0 { d / 1000.0 } else { d }
}

/// Parses a HereSphere scan (`{"scanData": [...]}`): the details of every
/// video at once, without thumbnails or media.
pub fn parse_heresphere_scan(v: &Value, base: &str) -> Vec<FeedScene> {
    array(v, "scanData")
        .iter()
        .filter_map(|s| {
            let url = url_field(s, "link", base)?;
            let mut meta = HereSphereTags::default();
            for t in array(s, "tags") {
                if let Some(n) = text(t, "name") {
                    meta.add(&n);
                }
            }
            let title = text(s, "title").unwrap_or_else(|| last_segment(&url));
            Some(FeedScene {
                format: detect_from_name(&title),
                title,
                duration: num(s, "duration").filter(|d| *d > 0.0).map(scan_seconds),
                released: date_field(s, "dateReleased"),
                added: date_field(s, "dateAdded"),
                rating: rating_field(s, "rating"),
                favorite: boolean(s, "isFavorite").unwrap_or(false),
                studio: meta.studio,
                actors: meta.actors,
                tags: meta.tags,
                url,
                ..FeedScene::default()
            })
        })
        .collect()
}

/// A scene as a feed's grid shows it: what the listing says, completed as
/// its details arrive ([`FeedScene::apply`]).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FeedScene {
    /// Scene URL: the location to open and fetch details from.
    pub url: String,
    /// Title.
    pub title: String,
    /// Length in seconds.
    pub duration: Option<f64>,
    /// Poster image URL.
    pub thumbnail_url: Option<String>,
    /// Release date, Unix seconds.
    pub released: Option<i64>,
    /// When the server added it, Unix seconds.
    pub added: Option<i64>,
    /// Rating out of 5.
    pub rating: Option<f32>,
    /// The user's favourite on the server.
    pub favorite: bool,
    /// Studio or site.
    pub studio: Option<String>,
    /// Performers.
    pub actors: Vec<String>,
    /// Categories and tags.
    pub tags: Vec<String>,
    /// Has haptic scripts.
    pub scripted: bool,
    /// Declared projection and stereo layout.
    pub format: Option<VideoFormat>,
    /// The scene's own document was read (so nothing more will come).
    pub detailed: bool,
    /// Only its address is known (a HereSphere listing, before its scan or
    /// details): the title is the address's last part.
    pub bare: bool,
}

impl FeedScene {
    fn from_entry(e: &Entry) -> FeedScene {
        FeedScene {
            url: e.location.clone(),
            title: e.name.clone(),
            duration: e.duration,
            thumbnail_url: e.thumbnail_url.clone(),
            format: e.format,
            scripted: !e.scripts.is_empty(),
            ..FeedScene::default()
        }
    }

    /// Completes the card with a scene's details.
    pub fn apply(&mut self, info: &SceneInfo) {
        if !info.title.is_empty() {
            self.title = info.title.clone();
        }
        self.duration = info.duration.or(self.duration);
        if info.thumbnail_url.is_some() {
            self.thumbnail_url = info.thumbnail_url.clone();
        }
        self.format = info
            .format
            .or(self.format)
            .or_else(|| detect_from_name(&self.title));
        self.released = info.released.or(self.released);
        self.added = info.added.or(self.added);
        self.rating = info.rating.or(self.rating);
        self.favorite |= info.favorite;
        if info.studio.is_some() {
            self.studio = info.studio.clone();
        }
        if !info.actors.is_empty() {
            self.actors = info.actors.clone();
        }
        if !info.tags.is_empty() {
            self.tags = info.tags.clone();
        }
        self.scripted |= !info.scripts.is_empty();
        self.detailed = true;
        self.bare = false;
    }

    /// Takes what a scan learned (titles, dates, tags) without its gaps.
    pub fn merge(&mut self, scan: FeedScene) {
        self.bare = false;
        if !scan.title.is_empty() {
            self.title = scan.title;
        }
        self.duration = scan.duration.or(self.duration);
        self.released = scan.released.or(self.released);
        self.added = scan.added.or(self.added);
        self.rating = scan.rating.or(self.rating);
        self.favorite |= scan.favorite;
        self.studio = scan.studio.or(self.studio.take());
        if !scan.actors.is_empty() {
            self.actors = scan.actors;
        }
        if !scan.tags.is_empty() {
            self.tags = scan.tags;
        }
        self.format = self.format.or(scan.format);
    }

    /// A browse entry to open it with.
    pub fn to_entry(&self) -> Entry {
        let mut e = Entry::new(self.title.clone(), self.url.clone(), EntryKind::Video);
        e.duration = self.duration;
        e.thumbnail_url = self.thumbnail_url.clone();
        e.format = self.format;
        e.modified = self.released.or(self.added);
        e
    }

    /// Whether every word (lower case) is in the title, studio, a
    /// performer or a tag.
    pub fn matches(&self, words: &[String]) -> bool {
        let hay = [
            self.title.as_str(),
            self.studio.as_deref().unwrap_or(""),
            &self.actors.join(" "),
            &self.tags.join(" "),
        ]
        .join(" ")
        .to_lowercase();
        words.iter().all(|w| hay.contains(w.as_str()))
    }
}

/// Everything a feed lists: its scene groups and their scenes.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FeedLibrary {
    /// Groups in feed order, each with the URLs of its scenes.
    pub groups: Vec<(String, Vec<String>)>,
    /// Every scene once, in the order first listed.
    pub scenes: Vec<FeedScene>,
}

impl FeedLibrary {
    /// Takes a HereSphere scan into the scenes it describes.
    pub fn merge_scan(&mut self, scan: Vec<FeedScene>) {
        let mut scanned: std::collections::HashMap<String, FeedScene> =
            scan.into_iter().map(|s| (s.url.clone(), s)).collect();
        for s in &mut self.scenes {
            if let Some(found) = scanned.remove(&s.url) {
                s.merge(found);
            }
        }
    }

    /// Where source `source_id`'s last listing is kept, to show at once
    /// the next time it is opened.
    pub fn cache_path(source_id: &str) -> std::path::PathBuf {
        let name: String = source_id
            .chars()
            .map(|c| match c {
                'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => c,
                _ => '_',
            })
            .collect();
        fp_core::dirs::cache_dir()
            .join("feeds")
            .join(format!("{name}.json"))
    }

    /// A listing saved by [`FeedLibrary::save`]; `None` when there is none
    /// (or it can't be read).
    pub fn load(path: &Path) -> Option<FeedLibrary> {
        let data = std::fs::read(path).ok()?;
        serde_json::from_slice(&data)
            .map_err(|e| log::debug!("feed listing {}: {e}", path.display()))
            .ok()
    }

    /// Saves the listing readable only by the user: its poster and scene
    /// URLs can carry the server's API key.
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_vec(self)
            .map_err(|e| Error::Config(format!("cannot serialize the feed listing: {e}")))?;
        crate::config::write_private(path, &json)?;
        Ok(())
    }

    fn from_groups(groups: Vec<SceneGroup>) -> FeedLibrary {
        let mut seen = std::collections::HashSet::new();
        let mut scenes = Vec::new();
        let groups = groups
            .into_iter()
            .map(|g| {
                let urls = g
                    .scenes
                    .iter()
                    .map(|e| {
                        if seen.insert(e.location.clone()) {
                            scenes.push(FeedScene::from_entry(e));
                        }
                        e.location.clone()
                    })
                    .collect();
                (g.name, urls)
            })
            .collect();
        FeedLibrary { groups, scenes }
    }
}

/// Parses a HereSphere library (`{"library":[{"name","list":[urls]}]}`).
/// Scenes carry only their URL (named after its last segment) until their
/// details are fetched.
pub fn parse_heresphere_library(v: &Value, base: &str) -> Result<Vec<SceneGroup>> {
    if num(v, "access").is_some_and(|a| a < 0.0) {
        return Err(Error::Auth(redact(base)));
    }
    Ok(array(v, "library")
        .iter()
        .enumerate()
        .map(|(i, g)| SceneGroup {
            name: text(g, "name").unwrap_or_else(|| format!("Group {}", i + 1)),
            scenes: array(g, "list")
                .iter()
                .filter_map(|u| u.as_str())
                .filter_map(|u| resolve(base, u).ok())
                .map(|url| {
                    let name = last_segment(&url);
                    Entry::new(name, url, EntryKind::Video)
                })
                .collect(),
        })
        .collect())
}

// ---- shared plumbing ----

/// Location of a group folder.
fn group_location(feed_url: &str, name: &str) -> String {
    format!("{feed_url}#group={}", encode_segment(name))
}

/// Splits a group location into feed URL and group name.
fn split_group(location: &str) -> Option<(&str, String)> {
    let (feed, name) = location.split_once("#group=")?;
    Some((feed, percent_decode(name)))
}

/// The scan endpoint beside a HereSphere library URL (query kept).
fn scan_url(library: &str) -> String {
    let (path, query) = match library.split_once('?') {
        Some((p, q)) => (p, Some(q)),
        None => (library, None),
    };
    let mut url = format!("{}/scan", path.trim_end_matches('/'));
    if let Some(q) = query {
        url.push('?');
        url.push_str(q);
    }
    url
}

/// True when a URL obviously points at a file rather than a scene document.
fn is_direct_file(url: &str) -> bool {
    let name = last_segment(url);
    is_video_name(&name) || is_script_name(&name) || is_subtitle_name(&name)
}

/// True when a response looks like JSON rather than media.
fn is_json_reply(r: &HttpReply) -> bool {
    match r.header("content-type") {
        None => true,
        Some(ct) => {
            let ct = ct.to_ascii_lowercase();
            ct.contains("json") || ct.starts_with("text/") || ct.contains("javascript")
        }
    }
}

struct FeedCore {
    id: String,
    name: String,
    url: String,
    creds: Option<Credentials>,
    client: HttpClient,
    max_height: Option<u32>,
}

impl std::fmt::Debug for FeedCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeedCore")
            .field("id", &self.id)
            .field("url", &redact(&self.url))
            .field("credentials", &self.creds)
            .field("max_height", &self.max_height)
            .finish_non_exhaustive()
    }
}

impl FeedCore {
    fn new(config: &FeedConfig) -> Result<FeedCore> {
        if !is_http_url(&config.url) {
            return Err(Error::invalid(
                &config.url,
                "expected an http:// or https:// feed URL",
            ));
        }
        let (url, creds) = clean_url_and_credentials(&config.url, config.credentials.as_ref())?;
        let opts = HttpOptions {
            insecure_tls: config.insecure_tls,
            ..HttpOptions::default()
        };
        Ok(FeedCore {
            id: config.id.clone(),
            name: config.name.clone(),
            client: HttpClient::new(opts, creds.as_ref()),
            url,
            creds,
            max_height: config.max_height,
        })
    }

    /// Sends a request and parses the JSON answer. `Ok(None)` when the
    /// server answered with something that is not JSON (a media stream),
    /// without downloading it.
    fn json(
        &self,
        method: &str,
        url: &str,
        content_type: Option<&str>,
        body: Option<&[u8]>,
    ) -> Result<Option<Value>> {
        let mut headers = vec![("Accept", "application/json")];
        if let Some(ct) = content_type {
            headers.push(("Content-Type", ct));
        }
        let (reply, read) =
            self.client
                .request_if(method, url, &headers, body, JSON_LIMIT, is_json_reply)?;
        let reply = reply.error_for_status(url)?;
        if !read {
            return Ok(None);
        }
        serde_json::from_slice(&reply.body)
            .map(Some)
            .map_err(|e| Error::parse(format!("JSON from {}", redact(url)), e))
    }

    fn open_url(&self, url: &str) -> Result<Arc<dyn ByteSource>> {
        Ok(Arc::new(self.client.open(url)?))
    }

    fn group_entries(&self, groups: &[SceneGroup]) -> Vec<Entry> {
        groups
            .iter()
            .map(|g| {
                let mut e = dir_entry(g.name.clone(), group_location(&self.url, &g.name), None);
                e.thumbnail_url = g.scenes.iter().find_map(|s| s.thumbnail_url.clone());
                e
            })
            .collect()
    }

    fn pick(&self, scene: &SceneInfo, location: &str) -> Result<String> {
        scene
            .best_source(self.max_height)
            .map(|s| s.url.clone())
            .ok_or_else(|| Error::NotFound(format!("no playable source in {}", redact(location))))
    }
}

fn find_group(groups: Vec<SceneGroup>, name: &str, location: &str) -> Result<Vec<Entry>> {
    groups
        .into_iter()
        .find(|g| g.name == name)
        .map(|g| g.scenes)
        .ok_or_else(|| Error::NotFound(redact(location)))
}

/// A DeoVR JSON feed (XBVR `/deovr`, Stash and others).
///
/// With credentials, documents are fetched with a `POST` of the form
/// fields `login` and `password` (the DeoVR app's login); otherwise with
/// `GET`.
#[derive(Debug)]
pub struct DeoVrSource {
    core: FeedCore,
}

impl DeoVrSource {
    /// Creates the source.
    pub fn new(config: &FeedConfig) -> Result<DeoVrSource> {
        Ok(DeoVrSource {
            core: FeedCore::new(config)?,
        })
    }

    fn fetch(&self, url: &str) -> Result<Option<Value>> {
        match &self.core.creds {
            Some(c) => {
                let form = format!(
                    "login={}&password={}",
                    encode_form_value(&c.username),
                    encode_form_value(&c.password)
                );
                self.core.json(
                    "POST",
                    url,
                    Some("application/x-www-form-urlencoded"),
                    Some(form.as_bytes()),
                )
            }
            None => self.core.json("GET", url, None, None),
        }
    }

    fn fetch_required(&self, url: &str) -> Result<Value> {
        self.fetch(url)?
            .ok_or_else(|| Error::parse("DeoVR feed", format!("{} is not JSON", redact(url))))
    }

    fn groups(&self, feed_url: &str) -> Result<Vec<SceneGroup>> {
        let v = self.fetch_required(feed_url)?;
        Ok(parse_deovr_root(&v, feed_url))
    }

    /// Fetches and parses one scene.
    pub fn scene(&self, scene_url: &str) -> Result<SceneInfo> {
        let v = self.fetch_required(scene_url)?;
        Ok(parse_deovr_scene(&v, scene_url))
    }

    /// Every group and scene of the feed, from its root document (which
    /// carries titles, lengths and posters).
    pub fn library(&self) -> Result<FeedLibrary> {
        Ok(FeedLibrary::from_groups(self.groups(&self.core.url)?))
    }
}

impl Source for DeoVrSource {
    fn id(&self) -> &str {
        &self.core.id
    }

    fn name(&self) -> &str {
        &self.core.name
    }

    fn kind(&self) -> SourceKind {
        SourceKind::DeoVr
    }

    fn describe(&self) -> String {
        format!("DeoVR feed {}", redact(&self.core.url))
    }

    fn list(&self, location: Option<&str>) -> Result<Vec<Entry>> {
        match location {
            None => {
                let groups = self.groups(&self.core.url)?;
                Ok(self.core.group_entries(&groups))
            }
            Some(loc) => match split_group(loc) {
                Some((feed, name)) => find_group(self.groups(feed)?, &name, loc),
                None => {
                    let groups = self.groups(loc)?;
                    Ok(self.core.group_entries(&groups))
                }
            },
        }
    }

    fn open(&self, location: &str) -> Result<Arc<dyn ByteSource>> {
        if is_direct_file(location) {
            return self.core.open_url(location);
        }
        match self.fetch(location)? {
            Some(v) => {
                let scene = parse_deovr_scene(&v, location);
                self.core.open_url(&self.core.pick(&scene, location)?)
            }
            None => self.core.open_url(location),
        }
    }

    fn details(&self, entry: &Entry) -> Result<Entry> {
        if entry.kind != EntryKind::Video || is_direct_file(&entry.location) {
            return Ok(entry.clone());
        }
        let mut e = entry.clone();
        self.scene(&entry.location)?.apply_to(&mut e);
        Ok(e)
    }

    fn sidecars(&self, video: &Entry) -> Result<Sidecars> {
        let full = if video.scripts.is_empty() && video.subtitles.is_empty() {
            self.details(video)?
        } else {
            video.clone()
        };
        Ok(Sidecars::from_entry(&full))
    }
}

/// A HereSphere JSON API (XBVR and Stash `/heresphere`).
///
/// Every request is a `POST` with a JSON body carrying `username` and
/// `password` when credentials are configured.
#[derive(Debug)]
pub struct HereSphereSource {
    core: FeedCore,
}

impl HereSphereSource {
    /// Creates the source.
    pub fn new(config: &FeedConfig) -> Result<HereSphereSource> {
        Ok(HereSphereSource {
            core: FeedCore::new(config)?,
        })
    }

    fn post(&self, url: &str, needs_media: Option<bool>) -> Result<Option<Value>> {
        let mut body = serde_json::Map::new();
        if let Some(c) = &self.core.creds {
            body.insert("username".into(), Value::String(c.username.clone()));
            body.insert("password".into(), Value::String(c.password.clone()));
        }
        if let Some(n) = needs_media {
            body.insert("needsMediaSource".into(), Value::Bool(n));
        }
        let bytes = serde_json::to_vec(&Value::Object(body))?;
        self.core
            .json("POST", url, Some("application/json"), Some(&bytes))
    }

    fn post_required(&self, url: &str, needs_media: Option<bool>) -> Result<Value> {
        self.post(url, needs_media)?
            .ok_or_else(|| Error::parse("HereSphere API", format!("{} is not JSON", redact(url))))
    }

    fn groups(&self, feed_url: &str) -> Result<Vec<SceneGroup>> {
        let v = self.post_required(feed_url, None)?;
        parse_heresphere_library(&v, feed_url)
    }

    /// Fetches and parses one video. `with_media` asks for media sources
    /// (slower on some servers).
    pub fn video(&self, url: &str, with_media: bool) -> Result<SceneInfo> {
        let v = self.post_required(url, Some(with_media))?;
        Ok(parse_heresphere_video(&v, url))
    }

    /// Every group and scene of the API, quickly: the library lists only
    /// scene URLs, so its scenes are [`FeedScene::bare`] until [`Self::scan`]
    /// or [`Self::video`] fills them in.
    pub fn library(&self) -> Result<FeedLibrary> {
        let mut lib = FeedLibrary::from_groups(self.groups(&self.core.url)?);
        for s in &mut lib.scenes {
            s.bare = true;
        }
        Ok(lib)
    }

    /// Titles, dates and tags of every scene at once, from servers that
    /// answer a scan (`<url>/scan`, stash-vr); `None` from those that don't
    /// (XBVR). Slower than the library: seconds, and megabytes, for a large
    /// one. See [`FeedLibrary::merge_scan`].
    pub fn scan(&self) -> Option<Vec<FeedScene>> {
        match self.post(&scan_url(&self.core.url), None) {
            Ok(Some(v)) => Some(parse_heresphere_scan(&v, &self.core.url)),
            Ok(None) => None,
            Err(e) => {
                log::debug!("no HereSphere scan at {}: {e}", redact(&self.core.url));
                None
            }
        }
    }

    /// Fills in titles and details for a group's scenes, several requests
    /// at a time. Scenes whose details fail keep their URL-derived name.
    fn fill(&self, scenes: Vec<Entry>) -> Vec<Entry> {
        let next = AtomicUsize::new(0);
        let slots: Vec<std::sync::Mutex<Entry>> =
            scenes.into_iter().map(std::sync::Mutex::new).collect();
        std::thread::scope(|scope| {
            for _ in 0..DETAIL_WORKERS.min(slots.len()) {
                scope.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(slot) = slots.get(i) else { break };
                        let mut e = lock(slot);
                        match self.video(&e.location, false) {
                            Ok(info) => info.apply_to(&mut e),
                            Err(err) => log::debug!(
                                "HereSphere details for {} failed: {err}",
                                redact(&e.location)
                            ),
                        }
                    }
                });
            }
        });
        slots
            .into_iter()
            .map(|m| {
                m.into_inner()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
            })
            .collect()
    }
}

impl Source for HereSphereSource {
    fn id(&self) -> &str {
        &self.core.id
    }

    fn name(&self) -> &str {
        &self.core.name
    }

    fn kind(&self) -> SourceKind {
        SourceKind::HereSphere
    }

    fn describe(&self) -> String {
        format!("HereSphere API {}", redact(&self.core.url))
    }

    fn list(&self, location: Option<&str>) -> Result<Vec<Entry>> {
        match location {
            None => {
                let groups = self.groups(&self.core.url)?;
                Ok(self.core.group_entries(&groups))
            }
            Some(loc) => match split_group(loc) {
                Some((feed, name)) => Ok(self.fill(find_group(self.groups(feed)?, &name, loc)?)),
                None => {
                    let groups = self.groups(loc)?;
                    Ok(self.core.group_entries(&groups))
                }
            },
        }
    }

    fn open(&self, location: &str) -> Result<Arc<dyn ByteSource>> {
        if is_direct_file(location) {
            return self.core.open_url(location);
        }
        match self.post(location, Some(true))? {
            Some(v) => {
                let scene = parse_heresphere_video(&v, location);
                self.core.open_url(&self.core.pick(&scene, location)?)
            }
            None => self.core.open_url(location),
        }
    }

    fn details(&self, entry: &Entry) -> Result<Entry> {
        if entry.kind != EntryKind::Video || is_direct_file(&entry.location) {
            return Ok(entry.clone());
        }
        let mut e = entry.clone();
        self.video(&entry.location, true)?.apply_to(&mut e);
        Ok(e)
    }

    fn sidecars(&self, video: &Entry) -> Result<Sidecars> {
        let full = if video.scripts.is_empty() && video.subtitles.is_empty() {
            self.details(video)?
        } else {
            video.clone()
        };
        Ok(Sidecars::from_entry(&full))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn deovr_root() {
        let v = json!({
            "scenes": [
                {"name": "Recent", "list": [
                    {"title": "Beach_180_LR", "videoLength": 1800, "thumbnailUrl": "/img/1.jpg",
                     "video_url": "http://xbvr:9999/deovr/1"},
                    {"title": "No URL"},
                    {"title": "Rel", "videoLength": "60", "video_url": "/deovr/2"}
                ]},
                {"list": []}
            ]
        });
        let g = parse_deovr_root(&v, "http://xbvr:9999/deovr");
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].name, "Recent");
        assert_eq!(g[1].name, "Group 2");
        assert_eq!(g[0].scenes.len(), 2);
        let s = &g[0].scenes[0];
        assert_eq!(s.location, "http://xbvr:9999/deovr/1");
        assert_eq!(s.duration, Some(1800.0));
        assert_eq!(
            s.thumbnail_url.as_deref(),
            Some("http://xbvr:9999/img/1.jpg")
        );
        assert_eq!(s.format.unwrap().stereo, StereoLayout::SideBySide);
        assert_eq!(g[0].scenes[1].duration, Some(60.0));
        assert_eq!(g[0].scenes[1].location, "http://xbvr:9999/deovr/2");
        assert!(parse_deovr_root(&json!({"x": 1}), "http://h/").is_empty());
    }

    #[test]
    fn deovr_scene_and_best_source() {
        let v = json!({
            "title": "Beach",
            "videoLength": 1800,
            "is3d": true,
            "screenType": "mkx200",
            "stereoMode": "sbs",
            "thumbnailUrl": "http://xbvr/img/1.jpg",
            "encodings": [
                {"name": "h265", "videoSources": [
                    {"resolution": 4096, "height": 4096, "width": 8192, "url": "http://xbvr/f/8k.mp4"},
                    {"resolution": 2880, "url": "http://xbvr/f/6k.mp4"}
                ]},
                {"name": "h264", "videoSources": [
                    {"resolution": "1920", "height": 1920, "width": 3840, "url": "/f/4k.mp4"},
                    {"height": 1920}
                ]}
            ],
            "timeStamps": [{"ts": 300, "name": "Two"}, {"ts": 10, "name": "One"}],
            "fleshlight": [{"title": "Beach.funscript", "url": "http://xbvr/s/1.funscript"}]
        });
        let s = parse_deovr_scene(&v, "http://xbvr/deovr/1");
        assert_eq!(s.title, "Beach");
        assert_eq!(s.duration, Some(1800.0));
        assert_eq!(
            s.format,
            Some(VideoFormat::new(
                Projection::fisheye(200.0),
                StereoLayout::SideBySide
            ))
        );
        assert_eq!(s.sources.len(), 3);
        assert_eq!(s.best_source(None).unwrap().url, "http://xbvr/f/8k.mp4");
        assert_eq!(
            s.best_source(Some(2880)).unwrap().url,
            "http://xbvr/f/6k.mp4"
        );
        assert_eq!(
            s.best_source(Some(2000)).unwrap().url,
            "http://xbvr/f/4k.mp4"
        );
        assert_eq!(
            s.best_source(Some(100)).unwrap().url,
            "http://xbvr/f/4k.mp4"
        );
        assert_eq!(
            s.markers,
            [(10.0, "One".to_string()), (300.0, "Two".to_string())]
        );
        assert_eq!(s.scripts, ["http://xbvr/s/1.funscript"]);

        let mut e = Entry::new("x", "http://xbvr/deovr/1", EntryKind::Video);
        s.apply_to(&mut e);
        assert_eq!(e.name, "Beach");
        assert_eq!(e.location, "http://xbvr/deovr/1");
        assert_eq!(e.scripts.len(), 1);
        assert_eq!(e.markers.len(), 2);
        assert_eq!(e.format, s.format);
    }

    #[test]
    fn deovr_screen_types() {
        let f = |v: Value| deovr_format(&v);
        let p = |v: Value| f(v).map(|x| x.projection);
        assert_eq!(
            p(json!({"screenType": "dome"})),
            Some(Projection::EQUIRECT_180)
        );
        assert_eq!(
            p(json!({"screenType": "sphere"})),
            Some(Projection::EQUIRECT_360)
        );
        assert_eq!(p(json!({"screenType": "flat"})), Some(Projection::Flat));
        assert_eq!(
            p(json!({"screenType": "fisheye"})),
            Some(Projection::fisheye(180.0))
        );
        assert_eq!(
            p(json!({"screenType": "mkx220"})),
            Some(Projection::fisheye(220.0))
        );
        assert_eq!(
            p(json!({"screenType": "VRCA220"})),
            Some(Projection::fisheye(220.0))
        );
        assert_eq!(
            p(json!({"screenType": "rf52"})),
            Some(Projection::fisheye(190.0))
        );
        assert_eq!(
            f(json!({"is3d": true, "stereoMode": "tb"})),
            Some(VideoFormat::new(
                Projection::EQUIRECT_180,
                StereoLayout::TopBottom
            ))
        );
        assert_eq!(
            f(json!({"is3d": false})),
            Some(VideoFormat::new(Projection::Flat, StereoLayout::Mono))
        );
        assert_eq!(
            f(json!({"is3d": "true", "screenType": "sphere", "stereoMode": "off"})),
            Some(VideoFormat::new(
                Projection::EQUIRECT_360,
                StereoLayout::Mono
            ))
        );
        assert_eq!(f(json!({"title": "x"})), None);
    }

    #[test]
    fn heresphere_library_and_video() {
        let lib = json!({"access": 1, "library": [
            {"name": "All", "list": ["http://xbvr:9999/heresphere/1", "/heresphere/2"]}
        ]});
        let g = parse_heresphere_library(&lib, "http://xbvr:9999/heresphere").unwrap();
        assert_eq!(g[0].scenes.len(), 2);
        assert_eq!(g[0].scenes[1].location, "http://xbvr:9999/heresphere/2");
        assert_eq!(g[0].scenes[1].name, "2");
        assert!(matches!(
            parse_heresphere_library(&json!({"access": -1}), "http://h/"),
            Err(Error::Auth(_))
        ));

        let v = json!({
            "title": "Beach", "duration": 1_800_000.0, "thumbnailImage": "http://x/t.jpg",
            "projection": "fisheye", "stereo": "sbs", "lens": "MKX220", "fov": 180.0,
            "media": [{"name": "h265", "sources": [
                {"resolution": 2880, "height": 2880, "width": 5760, "size": 123, "url": "http://x/6k.mp4"},
                {"resolution": 1440, "height": 1440, "width": 2880, "url": "http://x/3k.mp4"}
            ]}],
            "scripts": [{"name": "Beach.funscript", "url": "http://x/Beach.funscript"}],
            "subtitles": [{"name": "English", "language": "en", "url": "http://x/Beach.en.srt"}],
            "tags": [{"name": "Talk", "start": 1500.0, "end": 9000.0}, {"name": "Genre:VR"}]
        });
        let s = parse_heresphere_video(&v, "http://x/heresphere/1");
        assert_eq!(s.duration, Some(1800.0));
        assert_eq!(
            s.format,
            Some(VideoFormat::new(
                Projection::fisheye(220.0),
                StereoLayout::SideBySide
            ))
        );
        assert_eq!(s.best_source(Some(2000)).unwrap().url, "http://x/3k.mp4");
        assert_eq!(s.best_source(None).unwrap().size, Some(123));
        assert_eq!(s.subtitles, ["http://x/Beach.en.srt"]);
        assert_eq!(s.markers, [(1.5, "Talk".to_string())]);
    }

    #[test]
    fn listings_take_a_scan_and_save() {
        let scene = |n: &str| Entry::new(n, format!("http://x/heresphere/{n}"), EntryKind::Video);
        let mut lib = FeedLibrary::from_groups(vec![SceneGroup {
            name: "Recent".into(),
            scenes: vec![scene("12"), scene("13")],
        }]);
        for s in &mut lib.scenes {
            s.bare = true;
        }
        lib.merge_scan(vec![FeedScene {
            url: "http://x/heresphere/12".into(),
            title: "Beach".into(),
            tags: vec!["Sun".into()],
            ..FeedScene::default()
        }]);
        assert!(!lib.scenes[0].bare && lib.scenes[0].title == "Beach");
        assert!(lib.scenes[1].bare && lib.scenes[1].title == "13");

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("feeds").join("x.json");
        lib.save(&path).unwrap();
        assert_eq!(FeedLibrary::load(&path).as_ref(), Some(&lib));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "poster URLs can carry an API key");
        }
        assert_eq!(FeedLibrary::load(&dir.path().join("none.json")), None);
        std::fs::write(&path, b"not json").unwrap();
        assert_eq!(FeedLibrary::load(&path), None);
    }

    #[test]
    fn the_file_comes_before_the_servers_transcode() {
        // As stash-vr answers: DeoVR lists the live transcodes first, and
        // they claim more than the file (2160 against 2048).
        let transcodes = json!({"name": "transcoding", "videoSources": [
            {"resolution": 2160, "url": "https://stash/scene/1/stream.mp4?resolution=FOUR_K"},
            {"resolution": 1080, "url": "https://stash/scene/1/stream.mp4?resolution=FULL_HD"}
        ]});
        let direct = json!({"name": "direct", "videoSources": [
            {"resolution": 2048, "url": "https://stash/scene/1/stream"}
        ]});
        let deovr = json!({"title": "A", "encodings": [transcodes, direct]});
        let s = parse_deovr_scene(&deovr, "https://vr/deovr/1");
        assert!(s.sources[0].is_transcode() && !s.sources[2].is_transcode());
        assert_eq!(
            s.best_source(None).unwrap().url,
            "https://stash/scene/1/stream"
        );
        // A limit the file is over: the transcode that fits.
        assert_eq!(
            s.best_source(Some(1080)).unwrap().url,
            "https://stash/scene/1/stream.mp4?resolution=FULL_HD"
        );
        // Nothing fits: the smallest file, before any transcode.
        assert_eq!(
            s.best_source(Some(100)).unwrap().url,
            "https://stash/scene/1/stream"
        );

        // HereSphere lists the file first; the same choice.
        let heresphere = json!({"title": "A", "media": [
            {"name": "direct", "sources": [{"resolution": 2048, "url": "https://stash/scene/1/stream"}]},
            {"name": "transcoding", "sources": [
                {"resolution": 2160, "url": "https://stash/scene/1/stream.mp4?resolution=FOUR_K"}
            ]}
        ]});
        let s = parse_heresphere_video(&heresphere, "https://vr/heresphere/1");
        assert_eq!(
            s.best_source(None).unwrap().url,
            "https://stash/scene/1/stream"
        );

        // Only transcodes: one of them.
        let only = json!({"title": "A", "encodings": [{"name": "Transcoding", "videoSources": [
            {"resolution": 720, "url": "https://stash/scene/1/stream.mp4?resolution=STANDARD_HD"}
        ]}]});
        let s = parse_deovr_scene(&only, "https://vr/deovr/1");
        assert!(s.best_source(None).unwrap().is_transcode());
    }

    #[test]
    fn heresphere_projections() {
        let p = |v: Value| heresphere_format(&v).map(|f| (f.projection, f.stereo));
        assert_eq!(
            p(json!({"projection": "equirectangular", "stereo": "tb"})),
            Some((Projection::EQUIRECT_180, StereoLayout::TopBottom))
        );
        assert_eq!(
            p(json!({"projection": "equirectangular360", "stereo": "mono"})),
            Some((Projection::EQUIRECT_360, StereoLayout::Mono))
        );
        assert_eq!(
            p(json!({"projection": "fisheye", "lens": "Linear", "fov": 190})),
            Some((Projection::fisheye(190.0), StereoLayout::Mono))
        );
        assert_eq!(
            p(json!({"projection": "fisheye", "lens": "MKX200"})),
            Some((Projection::fisheye(200.0), StereoLayout::Mono))
        );
        assert_eq!(
            p(json!({"projection": "perspective"})).map(|x| x.0),
            Some(Projection::Flat)
        );
        assert_eq!(
            p(json!({"projection": "equiangularCubemap"})).map(|x| x.0),
            Some(Projection::Eac { h_fov: 360.0 })
        );
        assert_eq!(p(json!({"projection": "weird"})), None);
        assert_eq!(p(json!({})), None);
    }

    #[test]
    fn describe_and_debug_hide_secrets() {
        let config = FeedConfig {
            id: "f".into(),
            name: "Stash".into(),
            url: "http://u:pwx9@stash:9999/deovr?apikey=SECRETKEY".into(),
            credentials: Some(Credentials::new("bob", "hunter2")),
            insecure_tls: false,
            max_height: None,
        };
        let d = DeoVrSource::new(&config).unwrap();
        let h = HereSphereSource::new(&config).unwrap();
        for t in [
            d.describe(),
            format!("{d:?}"),
            h.describe(),
            format!("{h:?}"),
        ] {
            assert!(
                !t.contains("SECRETKEY") && !t.contains("hunter2") && !t.contains("pwx9"),
                "{t}"
            );
        }
        assert!(
            DeoVrSource::new(&FeedConfig {
                url: "ftp://x/".into(),
                ..config
            })
            .is_err()
        );
    }

    #[test]
    fn deovr_scene_metadata() {
        let v = json!({
            "title": "Beach", "description": "Sun and sand.", "date": 1_700_000_000,
            "paysite": {"id": 3, "name": "VRStudio"},
            "actors": [{"id": 1, "name": "Jane"}, {"id": 2, "name": "Ann"}],
            "categories": [{"tag": {"id": 1, "name": "Outdoor"}}, {"name": "POV"}],
            "rating_avg": 4.5, "isFavorite": true
        });
        let s = parse_deovr_scene(&v, "http://xbvr/deovr/1");
        assert_eq!(s.description.as_deref(), Some("Sun and sand."));
        assert_eq!(s.released, Some(1_700_000_000));
        assert_eq!(s.studio.as_deref(), Some("VRStudio"));
        assert_eq!(s.actors, ["Jane", "Ann"]);
        assert_eq!(s.tags, ["Outdoor", "POV"]);
        assert_eq!(s.rating, Some(4.5));
        assert!(s.favorite);
        // Milliseconds and ISO dates both work.
        assert_eq!(
            parse_deovr_scene(&json!({"date": 1_700_000_000_000_i64}), "http://h/").released,
            Some(1_700_000_000)
        );
        assert!(
            parse_deovr_scene(&json!({"date": "2024-01-12"}), "http://h/")
                .released
                .is_some()
        );
    }

    #[test]
    fn heresphere_tags_describe_or_mark() {
        let v = json!({
            "title": "Beach", "dateReleased": "2023-05-01", "dateAdded": "2024-02-03",
            "rating": 4, "isFavorite": true,
            "tags": [
                {"name": "Studio:VRStudio"}, {"name": "Talent:Jane"}, {"name": "Talent:Ann"},
                {"name": "Category:Outdoor"}, {"name": "Feature:8K"}, {"name": "Tag Group:Set"},
                {"name": "POV"}, {"name": "Intro", "start": 2000.0, "end": 9000.0},
                {"name": "Marker:Dance", "start": 60000.0}
            ]
        });
        let s = parse_heresphere_video(&v, "http://x/heresphere/1");
        assert_eq!(s.studio.as_deref(), Some("VRStudio"));
        assert_eq!(s.actors, ["Jane", "Ann"]);
        assert_eq!(s.tags, ["Outdoor", "POV"]);
        assert_eq!(
            s.markers,
            [(2.0, "Intro".to_string()), (60.0, "Dance".to_string())]
        );
        assert!(s.released.is_some() && s.added > s.released);
        assert_eq!(s.rating, Some(4.0));
        assert!(s.favorite);
    }

    #[test]
    fn libraries_scans_and_cards() {
        let base = "http://x/heresphere";
        let g = parse_heresphere_library(
            &json!({"library": [
                {"name": "Recent", "list": ["/heresphere/1", "/heresphere/2"]},
                {"name": "Favs", "list": ["/heresphere/2", "/heresphere/3"]}
            ]}),
            base,
        )
        .unwrap();
        let mut lib = FeedLibrary::from_groups(g);
        assert_eq!(lib.groups.len(), 2);
        assert_eq!(
            lib.groups[1].1,
            ["http://x/heresphere/2", "http://x/heresphere/3"]
        );
        assert_eq!(lib.scenes.len(), 3, "scenes are listed once");
        assert_eq!(lib.scenes[0].title, "1");

        let scan = parse_heresphere_scan(
            &json!({"scanData": [
                {"link": "/heresphere/2", "title": "Lake_180_LR", "duration": 90000,
                 "dateReleased": "2022-01-01", "isFavorite": true,
                 "tags": [{"name": "Talent:Jane"}, {"name": "Category:Nature"}]},
                {"title": "no link"},
                // stash-vr's scan gives seconds where the video's own
                // document gives milliseconds.
                {"link": "/heresphere/3", "duration": 4893.57}
            ]}),
            base,
        );
        assert_eq!(scan.len(), 2);
        assert_eq!(scan[1].duration, Some(4893.57));
        lib.scenes[1].merge(scan.into_iter().next().unwrap());
        let s = &lib.scenes[1];
        assert_eq!((s.title.as_str(), s.duration), ("Lake_180_LR", Some(90.0)));
        assert!(s.favorite && s.format.is_some() && !s.detailed);
        let words = |t: &str| vec![t.to_string()];
        assert!(s.matches(&words("jane")) && s.matches(&words("nature")));
        assert!(s.matches(&["lake".into(), "jane".into()]));
        assert!(!s.matches(&words("forest")));

        let mut card = lib.scenes[0].clone();
        card.apply(&SceneInfo {
            title: "Forest".into(),
            thumbnail_url: Some("http://x/t.jpg".into()),
            scripts: vec!["http://x/a.funscript".into()],
            ..SceneInfo::default()
        });
        assert!(card.detailed && card.scripted);
        assert_eq!(
            card.to_entry().thumbnail_url.as_deref(),
            Some("http://x/t.jpg")
        );
        assert_eq!(card.to_entry().location, "http://x/heresphere/1");

        assert_eq!(scan_url("http://h/heresphere/"), "http://h/heresphere/scan");
        assert_eq!(
            scan_url("http://h/hs?apikey=1"),
            "http://h/hs/scan?apikey=1"
        );
    }

    #[test]
    fn group_locations() {
        let l = group_location("http://h/deovr?apikey=1", "Recent & New");
        assert_eq!(l, "http://h/deovr?apikey=1#group=Recent%20%26%20New");
        assert_eq!(
            split_group(&l),
            Some(("http://h/deovr?apikey=1", "Recent & New".to_string()))
        );
        assert!(is_direct_file("http://h/a/b.mp4?x=1"));
        assert!(!is_direct_file("http://h/deovr/12"));
    }
}
