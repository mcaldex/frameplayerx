use serde::Deserialize;
use std::collections::BTreeMap;

use super::{Error, Result};
pub use crate::media_identity::MediaKind;

/// Which subtitles for deaf and hard-of-hearing viewers to include.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SdhFilter {
    #[default]
    Any,
    Only,
    Exclude,
}

/// Full subtitles or forced subtitles for foreign parts only.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ContentFilter {
    #[default]
    Full,
    Forced,
    Any,
}

/// Search criteria and optional matching hints.
#[derive(Clone, Debug, Default)]
pub struct SearchQuery {
    pub title: String,
    pub language: Option<String>,
    pub kind: Option<MediaKind>,
    pub year: Option<u32>,
    pub season: Option<u32>,
    pub episode: Option<u32>,
    pub imdb_id: Option<u64>,
    pub tmdb_id: Option<u64>,
    pub parent_imdb_id: Option<u64>,
    pub parent_tmdb_id: Option<u64>,
    pub moviehash: Option<String>,
    /// The video's file name, searched for in place of the title when no ID
    /// is known. OpenSubtitles reads the title, year and episode from it,
    /// as Jellyfin's OpenSubtitles plugin relies on.
    pub file_name: Option<String>,
    pub sdh: SdhFilter,
    pub content: ContentFilter,
    pub include_translations: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Identity {
    Imdb(u64),
    Tmdb(u64),
    EpisodeImdb(u64, u32, u32),
    EpisodeTmdb(u64, u32, u32),
}

impl SearchQuery {
    fn media_kind(&self) -> Option<MediaKind> {
        self.kind.or_else(|| {
            (self.season.is_some() || self.episode.is_some()).then_some(MediaKind::Episode)
        })
    }

    fn identity(&self) -> Option<Identity> {
        let valid = |id: &u64| *id > 0;
        if let Some(id) = self.imdb_id.filter(valid) {
            return Some(Identity::Imdb(id));
        }
        if let Some(id) = self.tmdb_id.filter(valid) {
            return Some(Identity::Tmdb(id));
        }
        if self.media_kind() == Some(MediaKind::Episode) {
            let (season, episode) = (self.season?, self.episode?);
            if let Some(id) = self.parent_imdb_id.filter(valid) {
                return Some(Identity::EpisodeImdb(id, season, episode));
            }
            if let Some(id) = self.parent_tmdb_id.filter(valid) {
                return Some(Identity::EpisodeTmdb(id, season, episode));
            }
        }
        None
    }

    pub(super) fn canonical(&self, page: u32) -> Result<String> {
        let mut params = BTreeMap::<_, String>::new();
        let kind = self.media_kind();
        let moviehash = self
            .moviehash
            .as_deref()
            .filter(|s| s.len() == 16 && s.bytes().all(|b| b.is_ascii_hexdigit()));
        let identity = self.identity();
        let title = self.title.split_whitespace().collect::<Vec<_>>().join(" ");
        match identity {
            Some(Identity::Imdb(id)) => {
                params.insert("imdb_id", id.to_string());
            }
            Some(Identity::Tmdb(id)) => {
                params.insert("tmdb_id", id.to_string());
            }
            Some(Identity::EpisodeImdb(id, season, episode)) => {
                params.insert("episode_number", episode.to_string());
                params.insert("parent_imdb_id", id.to_string());
                params.insert("season_number", season.to_string());
            }
            Some(Identity::EpisodeTmdb(id, season, episode)) => {
                params.insert("episode_number", episode.to_string());
                params.insert("parent_tmdb_id", id.to_string());
                params.insert("season_number", season.to_string());
            }
            None => {
                let file_name = self
                    .file_name
                    .as_deref()
                    .map(str::trim)
                    .filter(|name| valid_title(name));
                if let Some(name) = file_name {
                    params.insert("query", name.to_lowercase());
                } else if valid_title(&title) {
                    params.insert("query", title.to_lowercase());
                    if let Some(year) = self.year {
                        params.insert("year", year.to_string());
                    }
                } else if moviehash.is_none() {
                    return Err(Error::InvalidSearch);
                }
                if kind == Some(MediaKind::Episode) {
                    if let Some(season) = self.season {
                        params.insert("season_number", season.to_string());
                    }
                    if let Some(episode) = self.episode {
                        params.insert("episode_number", episode.to_string());
                    }
                }
            }
        }
        if let Some(hash) = moviehash {
            params.insert("moviehash", hash.to_ascii_lowercase());
        }
        if let Some(kind) = kind {
            params.insert(
                "type",
                match kind {
                    MediaKind::Movie => "movie",
                    MediaKind::Episode => "episode",
                }
                .into(),
            );
        }
        if let Some(language) = self
            .language
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            if !language
                .bytes()
                .all(|b| b.is_ascii_alphabetic() || b == b'-')
            {
                return Err(Error::UnavailableLanguage);
            }
            params.insert("languages", language.to_ascii_lowercase());
        }
        if let Some(sdh) = match self.sdh {
            SdhFilter::Any => None,
            SdhFilter::Only => Some("only"),
            SdhFilter::Exclude => Some("exclude"),
        } {
            params.insert("hearing_impaired", sdh.into());
        }
        if let Some(content) = match self.content {
            ContentFilter::Full => Some("exclude"),
            ContentFilter::Forced => Some("only"),
            ContentFilter::Any => None,
        } {
            params.insert("foreign_parts_only", content.into());
        }
        if self.include_translations {
            params.insert("machine_translated", "include".into());
        } else {
            params.insert("ai_translated", "exclude".into());
        }
        if page > 1 {
            params.insert("page", page.to_string());
        }
        Ok(url::form_urlencoded::Serializer::new(String::new())
            .extend_pairs(params)
            .finish())
    }
}

fn valid_title(title: &str) -> bool {
    let drive_path = title.as_bytes().get(..3).is_some_and(|prefix| {
        prefix[0].is_ascii_alphabetic() && prefix[1] == b':' && matches!(prefix[2], b'/' | b'\\')
    });
    !title.is_empty() && !title.contains("://") && !title.starts_with(['/', '\\']) && !drive_path
}

/// How this subtitle was matched to the video.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MatchKind {
    Title,
    Identity,
    File,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Language {
    #[serde(rename = "language_code")]
    pub code: String,
    #[serde(rename = "language_name")]
    pub name: String,
}

/// One requestable part of a release, identified by its provider file ID.
#[derive(Clone, Debug, Deserialize)]
pub struct SubtitleFile {
    #[serde(rename = "file_id")]
    pub id: u64,
    #[serde(rename = "file_name", default)]
    pub name: String,
    #[serde(rename = "cd_number")]
    pub part: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct Subtitle {
    pub language: String,
    pub release: String,
    pub fps: Option<f64>,
    pub sdh: bool,
    pub forced: bool,
    pub machine_translated: bool,
    pub ai_translated: bool,
    pub match_kind: MatchKind,
    pub files: Vec<SubtitleFile>,
}

#[derive(Clone, Debug, Deserialize)]
pub(super) struct SearchReply {
    page: u32,
    total_pages: u32,
    data: Vec<SubtitleRecord>,
}

#[derive(Clone, Debug, Deserialize)]
struct SubtitleRecord {
    attributes: Attributes,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct Attributes {
    language: String,
    release: Option<String>,
    fps: Option<f64>,
    hearing_impaired: bool,
    foreign_parts_only: bool,
    ai_translated: bool,
    machine_translated: bool,
    moviehash_match: bool,
    from_trusted: bool,
    ratings: f64,
    download_count: u64,
    feature_details: FeatureDetails,
    files: Vec<SubtitleFile>,
}

/// The movie or episode OpenSubtitles filed a subtitle under.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct FeatureDetails {
    feature_type: String,
    #[serde(deserialize_with = "lenient_number")]
    season_number: Option<u64>,
    #[serde(deserialize_with = "lenient_number")]
    episode_number: Option<u64>,
    #[serde(deserialize_with = "lenient_number")]
    imdb_id: Option<u64>,
}

/// A number the service may also send as a string or null.
fn lenient_number<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<u64>, D::Error> {
    Ok(match serde_json::Value::deserialize(deserializer)? {
        serde_json::Value::Number(n) => n.as_u64(),
        serde_json::Value::String(s) => s.trim().parse().ok(),
        _ => None,
    })
}

impl SearchReply {
    pub(super) fn last_page(&self) -> u32 {
        self.total_pages.max(self.page)
    }

    pub(super) fn append(&mut self, other: SearchReply) {
        self.data.extend(other.data);
    }

    /// Filters and orders the results as Jellyfin's OpenSubtitles plugin
    /// does: only the same kind of video, episode and IMDb ID, then file
    /// matches first, then the most downloaded, best rated and trusted.
    pub(super) fn rank(mut self, query: &SearchQuery) -> Vec<Subtitle> {
        let identity = query.identity();
        let kind = query.media_kind();
        let imdb_id = query.imdb_id.filter(|id| *id > 0);
        let same_media = |f: &FeatureDetails| match kind {
            Some(MediaKind::Episode) => {
                f.feature_type.eq_ignore_ascii_case("episode")
                    && query
                        .season
                        .is_none_or(|s| f.season_number == Some(u64::from(s)))
                    && query
                        .episode
                        .is_none_or(|e| f.episode_number == Some(u64::from(e)))
            }
            Some(MediaKind::Movie) => {
                f.feature_type.eq_ignore_ascii_case("movie")
                    && imdb_id.is_none_or(|id| f.imdb_id == Some(id))
            }
            None => true,
        };
        let file_match = |a: &Attributes| a.moviehash_match && query.moviehash.is_some();
        let mut seen = std::collections::HashSet::new();
        self.data.retain_mut(|record| {
            let a = &mut record.attributes;
            a.files.retain(|f| f.id > 0);
            !a.files.is_empty()
                && seen.insert(a.files[0].id)
                && same_media(&a.feature_details)
                && query
                    .language
                    .as_deref()
                    .filter(|s| !s.trim().is_empty())
                    .is_none_or(|s| a.language.eq_ignore_ascii_case(s.trim()))
                && (query.include_translations || !(a.ai_translated || a.machine_translated))
                && !matches!(
                    (query.sdh, a.hearing_impaired),
                    (SdhFilter::Only, false) | (SdhFilter::Exclude, true)
                )
                && !matches!(
                    (query.content, a.foreign_parts_only),
                    (ContentFilter::Full, true) | (ContentFilter::Forced, false)
                )
        });
        // File IDs give a stable order when everything else ties.
        self.data.sort_by_cached_key(|record| {
            use std::cmp::Reverse;
            let a = &record.attributes;
            Reverse((
                file_match(a),
                a.download_count,
                (a.ratings * 100.0) as i64,
                a.from_trusted,
                Reverse(a.files[0].id),
            ))
        });
        self.data
            .into_iter()
            .map(|record| {
                let a = record.attributes;
                let match_kind = if file_match(&a) {
                    MatchKind::File
                } else if identity.is_some() {
                    MatchKind::Identity
                } else {
                    MatchKind::Title
                };
                Subtitle {
                    language: a.language,
                    release: a.release.unwrap_or_default(),
                    fps: a.fps.filter(|v| *v > 0.0),
                    sdh: a.hearing_impaired,
                    forced: a.foreign_parts_only,
                    machine_translated: a.machine_translated,
                    ai_translated: a.ai_translated,
                    match_kind,
                    files: a.files,
                }
            })
            .collect()
    }
}
