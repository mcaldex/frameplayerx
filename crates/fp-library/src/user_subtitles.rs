//! Subtitles attached by the user, independently of source sidecars and caches.

use std::collections::HashSet;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use fp_core::source::{FileSource, SubtitleFile, SubtitleFlags};
use fp_core::sync::lock;
use rusqlite::types::Type;
use rusqlite::{Connection, OptionalExtension, Row, params};
use serde::{Deserialize, Serialize};

use crate::record::now;
use crate::{Error, Library, MediaId, Result};

pub const SUBTITLE_SCHEME: &str = "subtitle://";
pub const MAX_SUBTITLE_BYTES: usize = 8 << 20;
const LOCAL_PREFIX: &str = "subtitle://local/";
const STAGING_PREFIX: &str = ".download-";
const STAGING_SUFFIX: &str = ".part";
const STAGING_RANDOM_LEN: usize = 12;
const ORPHAN_GRACE: Duration = Duration::from_secs(24 * 60 * 60);
const MAX_DELAY_MS: i32 = 10_000;

// Cleanup must see the outstanding temporary files and cannot run between
// publication of a final filename and its database commit.
static STAGING: LazyLock<Mutex<HashSet<PathBuf>>> = LazyLock::new(Mutex::default);

const COLUMNS: &str = "id, media_id, provider, provider_file_id, download_parameters, storage_key,
    display_name, language, language_name, release, flags, checksum, original_signature,
    delay_ms, added_at";

/// The observed original file identity when the subtitle was downloaded.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OriginalSubtitleSignature {
    pub version: Option<String>,
    pub size: Option<u64>,
    pub mtime: Option<i64>,
    pub moviehash: Option<String>,
}

impl OriginalSubtitleSignature {
    /// Whether any observation establishes that the original has changed.
    pub fn differs_from(&self, current: &Self) -> bool {
        fn differs<T: PartialEq>(a: &Option<T>, b: &Option<T>) -> bool {
            a.as_ref().zip(b.as_ref()).is_some_and(|(a, b)| a != b)
        }
        differs(&self.version, &current.version)
            || differs(&self.size, &current.size)
            || differs(&self.mtime, &current.mtime)
            || differs(&self.moviehash, &current.moviehash)
    }
}

/// An existing media row captured before starting a download. SQLite may
/// reuse a row ID after deletion; the token ties a download to its own row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaAttachmentTarget {
    pub media_id: MediaId,
    token: Vec<u8>,
}

/// Metadata for a newly downloaded and validated subtitle file.
#[derive(Debug)]
pub struct NewUserSubtitle {
    pub provider: String,
    pub provider_file_id: i64,
    pub download_parameters: String,
    pub display_name: String,
    pub language: String,
    pub language_name: Option<String>,
    pub release: Option<String>,
    pub flags: SubtitleFlags,
    pub checksum: String,
    pub original_signature: Option<OriginalSubtitleSignature>,
}

/// A subtitle download owned by the video in the library.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UserSubtitle {
    pub id: String,
    pub media_id: MediaId,
    pub provider: String,
    pub provider_file_id: i64,
    pub download_parameters: String,
    pub storage_key: String,
    pub display_name: String,
    pub language: String,
    pub language_name: Option<String>,
    pub release: Option<String>,
    pub flags: SubtitleFlags,
    pub checksum: String,
    pub original_signature: Option<OriginalSubtitleSignature>,
    pub delay_ms: i32,
    pub added_at: i64,
}

impl UserSubtitle {
    pub fn location(&self) -> String {
        format!("{LOCAL_PREFIX}{}", self.storage_key)
    }

    pub fn as_file(&self) -> SubtitleFile {
        SubtitleFile {
            name: self.display_name.clone(),
            location: self.location(),
            language: Some(self.language.clone()),
            language_name: self.language_name.clone(),
            flags: self.flags,
        }
    }
}

impl Library {
    /// Capture the existing row to which a background job may attach data.
    pub fn subtitle_target(&self, id: MediaId) -> Result<MediaAttachmentTarget> {
        self.in_tx(|conn| {
            conn.execute(
                "UPDATE media SET subtitle_target_token = randomblob(16)
                 WHERE id = ?1 AND subtitle_target_token IS NULL",
                [id.0],
            )?;
            let token = conn
                .query_row(
                    "SELECT subtitle_target_token FROM media WHERE id = ?1",
                    [id.0],
                    |row| row.get(0),
                )
                .optional()?
                .ok_or(Error::MediaNotFound(id))?;
            Ok(MediaAttachmentTarget {
                media_id: id,
                token,
            })
        })
    }

    pub fn user_subtitles(&self, id: MediaId) -> Result<Vec<UserSubtitle>> {
        self.with(|conn| {
            let mut stmt = conn.prepare_cached(&format!(
                "SELECT {COLUMNS} FROM user_subtitles WHERE media_id = ?1 ORDER BY added_at, rowid"
            ))?;
            let rows = stmt.query_map([id.0], from_row)?;
            Ok(rows.collect::<rusqlite::Result<_>>()?)
        })
    }

    /// Set the signed offset for an attachment, bounded to ten seconds.
    /// Returns whether the attachment still exists.
    pub fn set_subtitle_delay(&self, media_id: MediaId, id: &str, delay_ms: i32) -> Result<bool> {
        self.with(|conn| {
            Ok(conn.execute(
                "UPDATE user_subtitles SET delay_ms = ?3 WHERE media_id = ?1 AND id = ?2",
                params![media_id.0, id, delay_ms.clamp(-MAX_DELAY_MS, MAX_DELAY_MS)],
            )? != 0)
        })
    }

    fn find_user_subtitle(
        &self,
        target: &MediaAttachmentTarget,
        subtitle: &NewUserSubtitle,
    ) -> Result<Option<UserSubtitle>> {
        self.with(|conn| {
            validate_target(conn, target)?;
            find_user_subtitle(conn, target.media_id, subtitle)
        })
    }

    fn save_user_subtitle(
        &self,
        target: &MediaAttachmentTarget,
        id: &str,
        key: &str,
        sub: &NewUserSubtitle,
        replacing_missing: bool,
    ) -> Result<UserSubtitle> {
        let flags = serde_json::to_string(&sub.flags)?;
        let signature = sub
            .original_signature
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        self.in_tx(|conn| {
            let media_id = target.media_id;
            validate_target(conn, target)?;
            if replacing_missing {
                conn.execute(
                    "UPDATE user_subtitles SET display_name = ?3, language = ?4, language_name = ?5,
                         release = ?6, flags = ?7, checksum = ?8, original_signature = ?9
                     WHERE media_id = ?1 AND id = ?2",
                    params![media_id.0, id, sub.display_name, sub.language, sub.language_name,
                        sub.release, flags, sub.checksum, signature],
                )?;
            } else {
                conn.execute(
                    "INSERT INTO user_subtitles (
                         id, media_id, provider, provider_file_id, download_parameters, storage_key,
                         display_name, language, language_name, release, flags, checksum,
                         original_signature, added_at
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
                     ON CONFLICT (media_id, provider, provider_file_id, download_parameters) DO NOTHING",
                    params![id, media_id.0, sub.provider, sub.provider_file_id, sub.download_parameters,
                        key, sub.display_name, sub.language, sub.language_name, sub.release, flags,
                        sub.checksum, signature, now()],
                )?;
            }
            find_user_subtitle(conn, media_id, sub)?.ok_or(Error::MediaNotFound(media_id))
        })
    }

    fn remove_user_subtitle(&self, media_id: MediaId, id: &str) -> Result<Option<UserSubtitle>> {
        self.in_tx(|conn| {
            let subtitle = conn
                .query_row(
                    &format!(
                        "SELECT {COLUMNS} FROM user_subtitles WHERE media_id = ?1 AND id = ?2"
                    ),
                    params![media_id.0, id],
                    from_row,
                )
                .optional()?;
            let Some(subtitle) = subtitle else {
                return Ok(None);
            };
            conn.execute("DELETE FROM user_subtitles WHERE id = ?1", [id])?;
            conn.execute(
                "UPDATE media SET tracks = NULLIF(json_remove(tracks, '$.subtitles'), '{}')
                 WHERE id = ?1 AND json_extract(tracks, '$.subtitles.file') = ?2",
                params![media_id.0, subtitle.location()],
            )?;
            Ok(Some(subtitle))
        })
    }

    fn subtitle_storage_keys(&self) -> Result<HashSet<String>> {
        self.with(|conn| {
            let mut stmt = conn.prepare("SELECT storage_key FROM user_subtitles")?;
            let rows = stmt.query_map([], |row| row.get(0))?;
            Ok(rows.collect::<rusqlite::Result<_>>()?)
        })
    }
}

fn validate_target(conn: &Connection, target: &MediaAttachmentTarget) -> Result<()> {
    let valid: bool = conn.query_row(
        "SELECT EXISTS (SELECT 1 FROM media WHERE id = ?1 AND subtitle_target_token = ?2)",
        params![target.media_id.0, target.token],
        |row| row.get(0),
    )?;
    if valid {
        Ok(())
    } else {
        Err(Error::MediaNotFound(target.media_id))
    }
}

fn find_user_subtitle(
    conn: &Connection,
    id: MediaId,
    sub: &NewUserSubtitle,
) -> Result<Option<UserSubtitle>> {
    Ok(conn
        .query_row(
            &format!(
                "SELECT {COLUMNS} FROM user_subtitles WHERE media_id = ?1 AND provider = ?2
            AND provider_file_id = ?3 AND download_parameters = ?4"
            ),
            params![
                id.0,
                sub.provider,
                sub.provider_file_id,
                sub.download_parameters
            ],
            from_row,
        )
        .optional()?)
}

fn from_row(row: &Row<'_>) -> rusqlite::Result<UserSubtitle> {
    fn parse<T: serde::de::DeserializeOwned>(value: String, index: usize) -> rusqlite::Result<T> {
        serde_json::from_str(&value).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(index, Type::Text, Box::new(error))
        })
    }
    Ok(UserSubtitle {
        id: row.get(0)?,
        media_id: MediaId(row.get(1)?),
        provider: row.get(2)?,
        provider_file_id: row.get(3)?,
        download_parameters: row.get(4)?,
        storage_key: row.get(5)?,
        display_name: row.get(6)?,
        language: row.get(7)?,
        language_name: row.get(8)?,
        release: row.get(9)?,
        flags: parse(row.get(10)?, 10)?,
        checksum: row.get(11)?,
        original_signature: row
            .get::<_, Option<String>>(12)?
            .map(|s| parse(s, 12))
            .transpose()?,
        delay_ms: row.get(13)?,
        added_at: row.get(14)?,
    })
}

/// Holds a temporary file while the subtitle loader checks it.
pub struct StagedSubtitle {
    temp: tempfile::NamedTempFile,
    _active: ActivePath,
}

impl StagedSubtitle {
    pub fn path(&self) -> &Path {
        self.temp.path()
    }
}

struct ActivePath(PathBuf);

impl Drop for ActivePath {
    fn drop(&mut self) {
        lock(&STAGING).remove(&self.0);
    }
}

/// Manages durable subtitle bytes in the application's data directory.
#[derive(Clone, Debug)]
pub struct SubtitleStore {
    root: PathBuf,
}

impl Default for SubtitleStore {
    fn default() -> Self {
        Self::new(fp_core::dirs::data_dir().join("subtitles"))
    }
}

impl SubtitleStore {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// Maps a managed location to a filename without I/O or source lookup.
    pub fn resolve(&self, location: &str) -> Option<PathBuf> {
        let key = location.strip_prefix(LOCAL_PREFIX)?;
        valid_storage_key(key).then(|| self.root.join(key))
    }

    /// A regular file within the data directory; reject redirected names.
    pub fn open(&self, location: &str) -> io::Result<FileSource> {
        let path = self
            .resolve(location)
            .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
        if !fs::symlink_metadata(&path)?.is_file() {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        FileSource::open(&path)
    }

    /// Stages bytes in the destination filesystem. Dropping the stage rolls
    /// it back, for example if the decoder cannot load usable cues.
    pub fn stage(&self, bytes: &[u8]) -> Result<StagedSubtitle> {
        if bytes.is_empty() || bytes.len() > MAX_SUBTITLE_BYTES {
            return Err(Error::InvalidArgument(
                "invalid subtitle download size".into(),
            ));
        }
        fs::create_dir_all(&self.root).map_err(|error| Error::io(&self.root, error))?;
        let root = self
            .root
            .canonicalize()
            .map_err(|error| Error::io(&self.root, error))?;
        let (mut temp, active) = {
            let mut paths = lock(&STAGING);
            let temp = tempfile::Builder::new()
                .prefix(STAGING_PREFIX)
                .suffix(STAGING_SUFFIX)
                .rand_bytes(STAGING_RANDOM_LEN)
                .tempfile_in(&root)
                .map_err(|error| Error::io(&root, error))?;
            let path = temp.path().to_owned();
            paths.insert(path.clone());
            (temp, ActivePath(path))
        };
        let result = temp
            .write_all(bytes)
            .and_then(|()| temp.as_file().sync_all());
        if let Err(error) = result {
            return Err(Error::io(temp.path(), error));
        }
        Ok(StagedSubtitle {
            temp,
            _active: active,
        })
    }

    /// Publish a validated file without overwriting any existing download,
    /// then register it. A duplicate returns the attached record or restores
    /// missing bytes at its original key, preserving the selection and timing.
    pub fn commit(
        &self,
        staged: StagedSubtitle,
        library: &Library,
        target: &MediaAttachmentTarget,
        subtitle: NewUserSubtitle,
    ) -> Result<UserSubtitle> {
        let root = self
            .root
            .canonicalize()
            .map_err(|error| Error::io(&self.root, error))?;
        if staged.path().parent() != Some(&root) {
            return Err(Error::InvalidArgument(
                "subtitle belongs to a different store".into(),
            ));
        }
        let StagedSubtitle { temp, _active } = staged;
        let _maintaining = lock(&STAGING);
        let existing = library.find_user_subtitle(target, &subtitle)?;
        let repairing = existing.is_some();
        let (id, key) = if let Some(existing) = existing {
            if !valid_storage_key(&existing.storage_key) {
                return Err(Error::InvalidArgument(
                    "invalid subtitle storage key".into(),
                ));
            }
            let path = root.join(&existing.storage_key);
            match fs::symlink_metadata(&path) {
                Ok(meta) if meta.is_file() => return Ok(existing),
                Ok(_) => return Err(Error::io(path, io::ErrorKind::InvalidInput.into())),
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    (existing.id, existing.storage_key)
                }
                Err(error) => return Err(Error::io(path, error)),
            }
        } else {
            let id = random_uuid()?;
            let key = format!("{id}.srt");
            (id, key)
        };
        let path = root.join(&key);
        let file = temp
            .persist_noclobber(&path)
            .map_err(|error| Error::io(&path, error.error))?;
        drop(file);
        let saved = library.save_user_subtitle(target, &id, &key, &subtitle, repairing);
        if !matches!(&saved, Ok(row) if row.id == id) {
            let _ = fs::remove_file(&path);
        }
        saved
    }

    /// Delete the attachment and any saved choice of it. Leftover bytes
    /// after a filesystem error will be retried by orphan cleanup.
    pub fn remove(
        &self,
        library: &Library,
        media_id: MediaId,
        id: &str,
    ) -> Result<Option<UserSubtitle>> {
        let _maintaining = lock(&STAGING);
        let removed = library.remove_user_subtitle(media_id, id)?;
        if let Some(path) = removed
            .as_ref()
            .and_then(|sub| self.resolve(&sub.location()))
        {
            let _ = fs::remove_file(path);
        }
        Ok(removed)
    }

    pub fn cleanup(&self, library: &Library) -> Result<usize> {
        self.cleanup_older_than(library, ORPHAN_GRACE)
    }

    fn cleanup_older_than(&self, library: &Library, grace: Duration) -> Result<usize> {
        let active = lock(&STAGING);
        let root = match self.root.canonicalize() {
            Ok(root) => root,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(0),
            Err(error) => return Err(Error::io(&self.root, error)),
        };
        let paths = library.subtitle_storage_keys()?;
        let mut removed = 0;
        for item in fs::read_dir(&root).map_err(|error| Error::io(&root, error))? {
            let item = item.map_err(|error| Error::io(&root, error))?;
            let name = item.file_name();
            let Some(name) = name.to_str() else { continue };
            let path = item.path();
            if (!valid_storage_key(name) && !is_staging_key(name))
                || paths.contains(name)
                || active.contains(&path)
            {
                continue;
            }
            let old_file = item.metadata().is_ok_and(|meta| {
                meta.is_file()
                    && meta
                        .modified()
                        .is_ok_and(|time| time.elapsed().is_ok_and(|age| age >= grace))
            });
            if old_file && fs::remove_file(path).is_ok() {
                removed += 1;
            }
        }
        Ok(removed)
    }
}

fn is_staging_key(key: &str) -> bool {
    key.strip_prefix(STAGING_PREFIX)
        .and_then(|random| random.strip_suffix(STAGING_SUFFIX))
        .is_some_and(|random| {
            random.len() == STAGING_RANDOM_LEN && random.bytes().all(|b| b.is_ascii_alphanumeric())
        })
}

fn valid_storage_key(key: &str) -> bool {
    key.strip_suffix(".srt").is_some_and(|id| {
        id.len() == 36
            && id.bytes().enumerate().all(|(i, b)| match i {
                8 | 13 | 18 | 23 => b == b'-',
                14 => b == b'4',
                19 => matches!(b, b'8' | b'9' | b'a' | b'b'),
                _ => b.is_ascii_digit() || matches!(b, b'a'..=b'f'),
            })
    })
}

fn random_uuid() -> Result<String> {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes)
        .map_err(|error| Error::io("subtitle id", io::Error::other(error.to_string())))?;
    // UUID v4 version and RFC 9562 variant bits.
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let hex = format!("{:032x}", u128::from_be_bytes(bytes));
    Ok(format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MediaUpsert, PickedSubtitles, PickedTracks};
    use fp_core::ByteSource;

    const SRT: &[u8] = b"1\n00:00:01,500 --> 00:00:03,500\nGood evening.\n";
    const VIDEO: &str = "http://old-server/videos/film.mkv";

    fn setup() -> (tempfile::TempDir, SubtitleStore, Library, MediaId) {
        let temp = tempfile::tempdir().unwrap();
        let store = SubtitleStore::new(temp.path().join("subtitles"));
        let library = Library::open(temp.path().join("library.sqlite3")).unwrap();
        let id = library
            .upsert(&MediaUpsert::new(VIDEO, "server"))
            .unwrap()
            .id;
        (temp, store, library, id)
    }

    fn subtitle() -> NewUserSubtitle {
        NewUserSubtitle {
            provider: "opensubtitles".into(),
            provider_file_id: 42,
            download_parameters: "srt:utf-8".into(),
            display_name: "Film Release pt-BR.srt".into(),
            language: "pt-br".into(),
            language_name: Some("Brazilian Portuguese".into()),
            release: Some("Film Release".into()),
            flags: SubtitleFlags {
                sdh: true,
                ..Default::default()
            },
            checksum: "checksum".into(),
            original_signature: Some(OriginalSubtitleSignature {
                size: Some(1_048_576),
                moviehash: Some("0123456789abcdef".into()),
                ..Default::default()
            }),
        }
    }

    fn attach(store: &SubtitleStore, library: &Library, video: MediaId) -> UserSubtitle {
        store
            .commit(
                store.stage(SRT).unwrap(),
                library,
                &library.subtitle_target(video).unwrap(),
                subtitle(),
            )
            .unwrap()
    }

    #[test]
    fn attachments_survive_refresh_source_migration_history_clear_and_data_move() {
        let (temp, store, library, media) = setup();
        let original = attach(&store, &library, media);
        let tracks = PickedTracks {
            audio: Some(1),
            subtitles: Some(PickedSubtitles::File(original.location())),
        };
        library.set_tracks(media, &tracks).unwrap();
        library
            .set_subtitle_delay(media, &original.id, 700)
            .unwrap();

        let mut refreshed = MediaUpsert::new(VIDEO, "server");
        refreshed.subtitles = vec!["http://old-server/videos/film.srt".into()];
        assert_eq!(library.upsert(&refreshed).unwrap().id, media);
        let migrated = "source://server/item/film";
        library
            .rewrite_source_locations("server", |location| {
                (location == VIDEO).then(|| migrated.into())
            })
            .unwrap();
        library.clear_history().unwrap();
        assert_eq!(library.forget_unwatched("server").unwrap(), 0);
        let video = library.get(media).unwrap().unwrap();
        assert_eq!(video.location, migrated);
        assert_eq!(video.tracks, tracks);
        assert_eq!(video.subtitles, refreshed.subtitles);
        drop(library);

        let relocated = tempfile::tempdir().unwrap();
        let relocated_db = relocated.path().join("library.sqlite3");
        let relocated_subtitles = relocated.path().join("subtitles");
        fs::copy(temp.path().join("library.sqlite3"), &relocated_db).unwrap();
        fs::create_dir(&relocated_subtitles).unwrap();
        fs::copy(
            store.resolve(&original.location()).unwrap(),
            relocated_subtitles.join(&original.storage_key),
        )
        .unwrap();
        let reopened = Library::open(relocated_db).unwrap();
        let new_store = SubtitleStore::new(relocated_subtitles);
        let attachment = reopened.user_subtitles(media).unwrap().remove(0);
        assert_eq!(attachment.delay_ms, 700);
        assert_eq!(attachment.location(), original.location());
        assert_eq!(attachment.as_file().language.as_deref(), Some("pt-br"));
        assert!(attachment.as_file().flags.sdh);
        assert_eq!(attachment.original_signature, original.original_signature);
        let file = new_store.open(&attachment.location()).unwrap();
        let mut bytes = [0; 64];
        let size = file.read_at(0, &mut bytes).unwrap();
        assert_eq!(&bytes[..size], SRT);
    }

    #[test]
    fn duplicate_keeps_bytes_and_another_version_gets_its_own_attachment() {
        let (_temp, store, library, media) = setup();
        let attachment = attach(&store, &library, media);
        let duplicate = store
            .commit(
                store.stage(b"different bytes").unwrap(),
                &library,
                &library.subtitle_target(media).unwrap(),
                subtitle(),
            )
            .unwrap();
        assert_eq!(attachment, duplicate);
        assert_eq!(
            fs::read(store.resolve(&attachment.location()).unwrap()).unwrap(),
            SRT
        );

        let version = library
            .upsert(&MediaUpsert::new("source://server/version2", "server"))
            .unwrap()
            .id;
        let another = attach(&store, &library, version);
        assert_ne!(attachment.id, another.id);
        assert_eq!(library.user_subtitles(media).unwrap(), [attachment]);
        assert_eq!(library.user_subtitles(version).unwrap(), [another]);
        assert_eq!(fs::read_dir(&store.root).unwrap().count(), 2);
    }

    #[test]
    fn restored_missing_file_has_new_content_metadata_and_keeps_the_users_choice() {
        let (_temp, store, library, media) = setup();
        let original = attach(&store, &library, media);
        let path = store.resolve(&original.location()).unwrap();
        let tracks = PickedTracks {
            audio: None,
            subtitles: Some(PickedSubtitles::File(original.location())),
        };
        library.set_tracks(media, &tracks).unwrap();
        library
            .set_subtitle_delay(media, &original.id, -300)
            .unwrap();
        fs::remove_file(&path).unwrap();
        let replacement_bytes = b"1\n00:00:01,500 --> 00:00:03,500\nGood morning.\n";
        let mut replacement = subtitle();
        replacement.checksum = "new-checksum".into();
        replacement.original_signature = None;
        let repaired = store
            .commit(
                store.stage(replacement_bytes).unwrap(),
                &library,
                &library.subtitle_target(media).unwrap(),
                replacement,
            )
            .unwrap();
        assert_eq!(repaired.id, original.id);
        assert_eq!(repaired.location(), original.location());
        assert_eq!(repaired.added_at, original.added_at);
        assert_eq!(repaired.delay_ms, -300);
        assert_eq!(repaired.checksum, "new-checksum");
        assert_eq!(repaired.original_signature, None);
        assert_eq!(library.get(media).unwrap().unwrap().tracks, tracks);
        assert_eq!(library.user_subtitles(media).unwrap(), [repaired]);
        assert_eq!(fs::read(path).unwrap(), replacement_bytes);
    }

    #[test]
    fn deleting_a_media_row_rejects_a_late_job_even_after_row_id_is_reused() {
        let (_temp, store, library, media) = setup();
        let old = attach(&store, &library, media);
        let target = library.subtitle_target(media).unwrap();
        let staged = store.stage(SRT).unwrap();
        library.remove_media(media).unwrap();
        let new = library
            .upsert(&MediaUpsert::new(VIDEO, "server"))
            .unwrap()
            .id;
        assert_eq!(media, new);
        assert_ne!(target, library.subtitle_target(new).unwrap());
        assert!(
            matches!(store.commit(staged, &library, &target, subtitle()), Err(Error::MediaNotFound(id)) if id == media)
        );
        assert!(library.user_subtitles(new).unwrap().is_empty());

        let old_path = store.resolve(&old.location()).unwrap();
        assert!(old_path.exists());
        assert_eq!(store.cleanup(&library).unwrap(), 0);
        assert_eq!(
            store.cleanup_older_than(&library, Duration::ZERO).unwrap(),
            1
        );
        assert!(!old_path.exists());
    }

    #[test]
    fn removing_selected_subtitle_retains_audio_and_removes_bytes() {
        let (_temp, store, library, media) = setup();
        let attachment = attach(&store, &library, media);
        let path = store.resolve(&attachment.location()).unwrap();
        library
            .set_tracks(
                media,
                &PickedTracks {
                    audio: Some(2),
                    subtitles: Some(PickedSubtitles::File(attachment.location())),
                },
            )
            .unwrap();
        assert_eq!(
            store.remove(&library, media, &attachment.id).unwrap(),
            Some(attachment.clone())
        );
        assert_eq!(
            library.get(media).unwrap().unwrap().tracks,
            PickedTracks {
                audio: Some(2),
                subtitles: None
            }
        );
        assert!(!path.exists());
        assert!(
            !library
                .set_subtitle_delay(media, &attachment.id, -200)
                .unwrap()
        );
    }

    #[test]
    fn only_the_managed_namespace_is_touched_and_active_staging_is_preserved() {
        let (temp, store, library, media) = setup();
        let staged = store.stage(SRT).unwrap();
        let stage_path = staged.path().to_owned();
        let unmanaged = store.root.join("other.srt");
        fs::write(&unmanaged, SRT).unwrap();
        let attachment = attach(&store, &library, media);
        assert_eq!(
            store.cleanup_older_than(&library, Duration::ZERO).unwrap(),
            0
        );
        assert!(stage_path.exists());
        assert!(unmanaged.exists());
        assert!(store.resolve(&attachment.location()).unwrap().exists());
        drop(staged);
        assert!(!stage_path.exists());
        assert!(store.stage(&vec![0; MAX_SUBTITLE_BYTES + 1]).is_err());

        let key = &attachment.storage_key;
        for location in [
            format!("subtitle://local/../{key}"),
            format!("subtitle://local/%2e%2e/{key}"),
            format!("subtitle://local/..\\{key}"),
            format!("subtitle://local/{key}?query"),
            format!("subtitle://elsewhere/{key}"),
            format!("file://{key}"),
        ] {
            assert!(store.resolve(&location).is_none(), "{location}");
        }
        #[cfg(unix)]
        {
            let outside = temp.path().join("outside.srt");
            fs::write(&outside, SRT).unwrap();
            let managed = store.resolve(&attachment.location()).unwrap();
            fs::remove_file(&managed).unwrap();
            std::os::unix::fs::symlink(outside, managed).unwrap();
            assert!(store.open(&attachment.location()).is_err());
        }
        drop(temp);
    }
}
