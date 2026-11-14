//! Metadata and original-file fingerprints for matching a playing video.

use crate::urlutil::last_segment;
use fp_core::source::{ByteSource, Entry, is_video_name};
use std::io;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaKind {
    Movie,
    Episode,
}

/// Matching information that never carries a full file path or server URL.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MediaIdentity {
    pub title: Option<String>,
    pub year: Option<u32>,
    pub kind: Option<MediaKind>,
    pub series_title: Option<String>,
    pub season: Option<u32>,
    pub episode: Option<u32>,
    /// The movie or episode's IMDb and TMDb identifiers.
    pub imdb_id: Option<u64>,
    pub tmdb_id: Option<u64>,
    /// The series' identifiers, only usable with episode coordinates.
    pub series_imdb_id: Option<u64>,
    pub series_tmdb_id: Option<u64>,
    pub release_basename: Option<String>,
}

impl MediaIdentity {
    pub fn from_entry(video: &Entry) -> Self {
        Self::default().with_fallback(video)
    }

    /// Fills gaps in source metadata from a human-readable name or filename.
    pub fn with_fallback(mut self, video: &Entry) -> Self {
        if self.kind != Some(MediaKind::Movie)
            && let Some((series, title, season, episode)) = catalog_episode(&video.name)
        {
            self.kind = Some(MediaKind::Episode);
            self.title = self.title.or_else(|| display_title(title));
            self.series_title = self.series_title.or_else(|| display_title(series));
            self.season = self.season.or(season);
            self.episode = self.episode.or(Some(episode));
        }
        if self.release_basename.is_none() {
            self.release_basename = release_basename(&video.name)
                .or_else(|| release_basename(&last_segment(&video.location)));
        }
        let hints = self.release_basename.as_deref().map(filename_hints);
        if let Some(hints) = hints {
            self.year = self.year.or(hints.year);
            if self.kind != Some(MediaKind::Movie) && hints.episode.is_some() {
                self.kind = Some(MediaKind::Episode);
                self.series_title = self.series_title.or_else(|| hints.title.clone());
                self.season = self.season.or(hints.season);
                self.episode = self.episode.or(hints.episode);
            }
            if self.title.is_none() {
                let from_filename = self.release_basename.as_deref().is_some_and(|name| {
                    let stem = name.rsplit_once('.').map_or(name, |(stem, _)| stem);
                    stem.replace(['.', '_'], " ").trim() == video.name.trim()
                });
                self.title = if from_filename {
                    hints.title
                } else {
                    display_title(&video.name).or(hints.title)
                };
            }
        }
        self.title = self.title.or_else(|| display_title(&video.name));
        self
    }
}

/// Catalog entries display an episode as `Series S1:E2 · Episode title`.
fn catalog_episode(name: &str) -> Option<(&str, &str, Option<u32>, u32)> {
    let (series, title) = name.split_once(" · ")?;
    let (series, label) = series.rsplit_once(' ')?;
    let (season, episode) = match label.split_once(':') {
        Some((season, episode)) => (Some(season.strip_prefix('S')?.parse().ok()?), episode),
        None => (None, label),
    };
    Some((
        series,
        title,
        season,
        episode.strip_prefix('E')?.parse().ok()?,
    ))
}

/// Extracts a video filename from a path supplied as metadata, including
/// Windows paths supplied by a media server to a Linux client.
pub(crate) fn release_basename(path: &str) -> Option<String> {
    if crate::is_source_location(path) || path.starts_with("mb:") {
        return None;
    }
    let basename = if path.contains("://") {
        last_segment(path)
    } else {
        path.rsplit(['/', '\\']).next()?.trim().to_owned()
    };
    (is_video_name(&basename) && !basename.contains(['/', '\\', '?', '\0'])).then_some(basename)
}

pub(crate) fn display_title(value: &str) -> Option<String> {
    let title = value.trim();
    (!title.is_empty()
        && !title.contains("://")
        && !title.starts_with("mb:")
        && !title.starts_with('/')
        && !title.contains(['\\', '\0'])
        && title.as_bytes().get(1..3) != Some(b":/")
        && release_basename(title).is_none())
    .then(|| title.to_owned())
}

pub(crate) fn external_id(value: &str) -> Option<u64> {
    let value = value.trim();
    value
        .strip_prefix("tt")
        .unwrap_or(value)
        .parse::<u64>()
        .ok()
        .filter(|&id| id != 0)
}

fn filename_hints(name: &str) -> MediaIdentity {
    let stem = name.rsplit_once('.').map_or(name, |(stem, _)| stem);
    let words: Vec<_> = stem
        .split(|ch: char| ch.is_whitespace() || matches!(ch, '.' | '_' | '[' | ']' | '(' | ')'))
        .filter(|word| !word.is_empty())
        .collect();
    let episode =
        words.iter().enumerate().skip(1).find_map(|(i, word)| {
            episode_number(word).map(|(season, episode)| (i, season, episode))
        });
    let release_start = words
        .iter()
        .enumerate()
        .skip(1)
        .find(|(_, word)| is_release_tag(word))
        .map_or(words.len(), |(i, _)| i);
    let year_limit = episode.map_or(release_start, |(i, _, _)| release_start.min(i));
    let year = words
        .get(1..year_limit)
        .and_then(|prefix| prefix.last())
        .and_then(|word| word.parse::<u32>().ok())
        .filter(|year| (1888..=2100).contains(year));
    let title_end = year_limit - usize::from(year.is_some());
    let title = (title_end > 0).then(|| words[..title_end].join(" "));
    MediaIdentity {
        title,
        year,
        season: episode.map(|(_, season, _)| season),
        episode: episode.map(|(_, _, episode)| episode),
        ..Default::default()
    }
}

fn episode_number(word: &str) -> Option<(u32, u32)> {
    let lower = word.to_ascii_lowercase();
    let (season, episode) = lower.strip_prefix('s')?.split_once('e')?;
    if season.len() > 3 || episode.len() > 3 {
        return None;
    }
    Some((season.parse().ok()?, episode.parse().ok()?))
}

fn is_release_tag(word: &str) -> bool {
    let lower = word.to_ascii_lowercase();
    let tag = lower.split('-').next().unwrap_or_default();
    matches!(
        tag,
        "2160p"
            | "1080p"
            | "1080i"
            | "720p"
            | "576p"
            | "480p"
            | "bluray"
            | "bdrip"
            | "brrip"
            | "dvdrip"
            | "hdtv"
            | "webrip"
            | "webdl"
            | "x264"
            | "x265"
            | "hevc"
    ) || lower.starts_with("web-dl")
}

pub(crate) const REGION: usize = 65_536;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MovieHash {
    pub hash: String,
    pub size: u64,
    /// Modification time of the original file, in Unix seconds when available.
    pub modified: Option<i64>,
}

/// Fingerprints two regions of a separately opened original file. Unsupported
/// or small files return `None` without any reads. The deadline and cancel
/// check bound the job between reads. The reader must bound its own I/O waits.
pub fn movie_hash(
    source: &dyn ByteSource,
    deadline: Instant,
    cancelled: impl Fn() -> bool,
) -> io::Result<Option<MovieHash>> {
    if source.streamed() || !source.efficient_random_access() {
        return Ok(None);
    }
    let Some(size) = source.size().filter(|&size| size >= 2 * REGION as u64) else {
        return Ok(None);
    };
    let mut hash = size;
    let mut bytes = vec![0u8; REGION];
    let check = || -> io::Result<()> {
        if cancelled() {
            Err(io::ErrorKind::Interrupted.into())
        } else if Instant::now() >= deadline {
            Err(io::ErrorKind::TimedOut.into())
        } else {
            Ok(())
        }
    };
    for offset in [0, size - REGION as u64] {
        let mut filled = 0;
        while filled < bytes.len() {
            check()?;
            let read = source.read_at(offset + filled as u64, &mut bytes[filled..])?;
            if read == 0 || read > bytes.len() - filled {
                return Err(io::ErrorKind::UnexpectedEof.into());
            }
            filled += read;
        }
        for &word in bytes.as_chunks::<8>().0 {
            hash = hash.wrapping_add(u64::from_le_bytes(word));
        }
    }
    check()?;
    Ok(Some(MovieHash {
        hash: format!("{hash:016x}"),
        size,
        modified: source.modified(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fp_core::source::EntryKind;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn fallback_uses_release_hints_and_ignores_opaque_resources() {
        let movie = Entry::new(
            "2001.A.Space.Odyssey.1968.1080p.BluRay.mkv",
            "source://source/http-library/http/2001.mkv",
            EntryKind::Video,
        );
        let identity = MediaIdentity::from_entry(&movie);
        assert_eq!(identity.title.as_deref(), Some("2001 A Space Odyssey"));
        assert_eq!(identity.year, Some(1968));
        assert_eq!(identity.release_basename, Some(movie.name));

        let direct = Entry::new(
            "Film 1999 1080p",
            "/my/media/Film.1999.1080p.mkv",
            EntryKind::Video,
        );
        let identity = MediaIdentity::from_entry(&direct);
        assert_eq!(identity.title.as_deref(), Some("Film"));
        assert_eq!(identity.year, Some(1999));

        let episode = Entry::new(
            "The.Show.S00E02.mkv",
            "C:\\Media\\The.Show.S00E02.mkv",
            EntryKind::Video,
        );
        let identity = MediaIdentity::from_entry(&episode);
        assert_eq!(identity.series_title.as_deref(), Some("The Show"));
        assert_eq!(identity.kind, Some(MediaKind::Episode));
        assert_eq!((identity.season, identity.episode), (Some(0), Some(2)));

        let opaque = Entry::new("", "plex://source/x/server/item/628", EntryKind::Video);
        assert_eq!(MediaIdentity::from_entry(&opaque), MediaIdentity::default());
        let catalog = Entry::new(
            "Show S0:E2 · Episode title",
            opaque.location,
            EntryKind::Video,
        );
        let identity = MediaIdentity::from_entry(&catalog);
        assert_eq!(identity.series_title.as_deref(), Some("Show"));
        assert_eq!(identity.title.as_deref(), Some("Episode title"));
        assert_eq!((identity.season, identity.episode), (Some(0), Some(2)));
    }

    struct Reader {
        len: u64,
        bytes_per_read: usize,
        efficient: bool,
        streamed: bool,
        reads: AtomicUsize,
    }

    impl ByteSource for Reader {
        fn size(&self) -> Option<u64> {
            Some(self.len)
        }
        fn read_at(&self, offset: u64, out: &mut [u8]) -> io::Result<usize> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            let len = out.len().min(self.bytes_per_read);
            for (i, value) in out[..len].iter_mut().enumerate() {
                *value = (offset + i as u64) as u8;
            }
            Ok(len)
        }
        fn describe(&self) -> String {
            "test file".to_owned()
        }
        fn efficient_random_access(&self) -> bool {
            self.efficient
        }
        fn streamed(&self) -> bool {
            self.streamed
        }
    }

    const LEN: u64 = 3 * REGION as u64 + 1;

    fn reader() -> Reader {
        Reader {
            len: LEN,
            bytes_per_read: 2137,
            efficient: true,
            streamed: false,
            reads: AtomicUsize::new(0),
        }
    }

    fn deadline() -> Instant {
        Instant::now() + Duration::from_secs(2)
    }

    #[test]
    fn hashes_two_complete_regions_despite_short_reads() {
        let file = reader();
        assert_eq!(
            movie_hash(&file, deadline(), || false).unwrap(),
            Some(MovieHash {
                hash: "c0803fffbf822001".into(),
                size: LEN,
                modified: None,
            })
        );
        assert_eq!(file.reads.load(Ordering::Relaxed), 62);
    }

    #[test]
    fn skips_or_stops_reads_when_they_cannot_produce_a_hash() {
        for file in [
            Reader {
                efficient: false,
                ..reader()
            },
            Reader {
                streamed: true,
                ..reader()
            },
            Reader {
                len: 2 * REGION as u64 - 1,
                ..reader()
            },
        ] {
            assert_eq!(movie_hash(&file, deadline(), || false).unwrap(), None);
            assert_eq!(file.reads.load(Ordering::Relaxed), 0);
        }
        let file = reader();
        let checks = Cell::new(0);
        let error = movie_hash(&file, deadline(), || {
            checks.set(checks.get() + 1);
            checks.get() == 2
        })
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Interrupted);
        assert_eq!(file.reads.load(Ordering::Relaxed), 1);
        let error = movie_hash(&file, Instant::now(), || false).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        let no_bytes = Reader {
            bytes_per_read: 0,
            ..reader()
        };
        assert_eq!(
            movie_hash(&no_bytes, deadline(), || false)
                .unwrap_err()
                .kind(),
            io::ErrorKind::UnexpectedEof
        );
    }
}
