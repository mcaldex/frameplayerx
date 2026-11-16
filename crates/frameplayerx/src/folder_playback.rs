//! Collect videos for Play all when it includes subfolders.

use crate::settings::DEFAULT_PLAY_ALL_LIMIT;
use fp_core::source::{Entry, EntryKind};
use fp_sources::{EntrySort, Source, SourceKind};
use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Incomplete {
    /// A subfolder couldn't be listed.
    Unavailable,
    /// More videos were found than the limit takes.
    LimitReached(usize),
}

pub struct Scan {
    pub videos: Vec<Entry>,
    pub incomplete: Option<Incomplete>,
}

pub struct ScanOptions<'a> {
    pub sort: EntrySort,
    pub filter: &'a str,
    /// The most videos taken; `None` takes them all.
    pub max_videos: Option<usize>,
}

#[derive(PartialEq, Eq, Hash)]
enum Location {
    Local(PathBuf),
    Source(String),
}

impl Location {
    fn of(source: &dyn Source, location: &str) -> Self {
        // Local listings follow symlinks. Compare their filesystem targets
        // to keep a link to an ancestor from starting the walk again.
        if matches!(source.kind(), SourceKind::Local | SourceKind::Library)
            && let Some(path) = fp_core::volume::resolve(location)
            && let Ok(path) = std::fs::canonicalize(path.as_ref())
        {
            return Self::Local(path);
        }
        Self::Source(
            source
                .canonical_location(location)
                .unwrap_or_else(|| location.to_owned()),
        )
    }
}

fn dav_path(key: &str) -> &str {
    key.split_once('?')
        .map_or(key, |(path, _)| path)
        .trim_end_matches('/')
}

fn dav_folder_key(source: &dyn Source, folder: Option<&str>) -> fp_sources::Result<String> {
    let Some(folder) = folder else {
        return Ok(String::new());
    };
    source
        .canonical_location(folder)
        .as_deref()
        .and_then(fp_sources::location::decode)
        .map(|location| location.key)
        .ok_or_else(|| fp_sources::Error::InvalidLocation {
            location: fp_sources::urlutil::redact(folder),
            reason: "folder is outside the WebDAV source".into(),
        })
}

fn is_dav_child(source: &dyn Source, folder: &str, location: &str) -> bool {
    let Some(location) = fp_sources::location::decode(location) else {
        return false;
    };
    if location.source_id != source.id() || location.kind != "webdav" {
        return false;
    }
    let path = dav_path(&location.key);
    let (parent, name) = path.rsplit_once('/').unwrap_or(("", path));
    if name.is_empty() {
        return false;
    }
    let folder = dav_path(folder);
    if parent == folder {
        return true;
    }
    // A server may spell the same parent with different URL escapes.
    let parent = fp_sources::urlutil::percent_decode(parent);
    !parent.contains('\u{fffd}') && parent == fp_sources::urlutil::percent_decode(folder)
}

/// The videos of the folder `root` and every folder below it, reusing the
/// listing already on screen: the first [`ScanOptions::max_videos`] found,
/// folder by folder from the top, then sorted. A scan with no limit calls
/// `large_scan` once it has found more than [`DEFAULT_PLAY_ALL_LIMIT`].
pub fn collect(
    source: &dyn Source,
    root: Option<&str>,
    root_entries: Vec<Entry>,
    options: ScanOptions<'_>,
    cancelled: impl Fn() -> bool,
    large_scan: impl Fn(),
) -> fp_sources::Result<Scan> {
    let words: Vec<_> = options
        .filter
        .split_whitespace()
        .map(str::to_lowercase)
        .collect();
    let mut pending = VecDeque::from([root.map(str::to_owned)]);
    let mut folders = HashSet::new();
    let mut seen_videos = HashSet::new();
    let mut scan = Scan {
        videos: Vec::new(),
        incomplete: None,
    };
    let mut root_entries = Some(root_entries);
    let webdav = source.kind() == SourceKind::WebDav;
    'walk: while let Some(folder) = pending.pop_front() {
        if cancelled() {
            return Err(std::io::Error::from(std::io::ErrorKind::Interrupted).into());
        }
        if let Some(location) = &folder
            && !folders.insert(Location::of(source, location))
        {
            continue;
        }
        let dav_folder = webdav
            .then(|| dav_folder_key(source, folder.as_deref()))
            .transpose()?;
        let entries = if let Some(entries) = root_entries.take() {
            entries
        } else {
            match source.list(folder.as_deref()) {
                Ok(entries) => entries,
                Err(_) => {
                    scan.incomplete = Some(Incomplete::Unavailable);
                    continue;
                }
            }
        };
        for entry in entries {
            // Ordinary browsing shows the server's response. An automatic
            // walk only follows URLs within this folder's Depth: 1 scope.
            if dav_folder
                .as_ref()
                .is_some_and(|folder| !is_dav_child(source, folder, &entry.location))
            {
                continue;
            }
            match entry.kind {
                EntryKind::Directory => pending.push_back(Some(entry.location)),
                EntryKind::Video => {
                    let name = entry.name.to_lowercase();
                    if !words.iter().all(|w| name.contains(w.as_str()))
                        || !seen_videos.insert(Location::of(source, &entry.location))
                    {
                        continue;
                    }
                    if let Some(limit) = options.max_videos
                        && scan.videos.len() == limit
                    {
                        scan.incomplete = Some(Incomplete::LimitReached(limit));
                        break 'walk;
                    }
                    if options.max_videos.is_none() && scan.videos.len() == DEFAULT_PLAY_ALL_LIMIT {
                        large_scan();
                    }
                    scan.videos.push(entry);
                }
                _ => {}
            }
        }
    }
    if cancelled() {
        return Err(std::io::Error::from(std::io::ErrorKind::Interrupted).into());
    }
    fp_sources::sort_entries_by(&mut scan.videos, options.sort);
    Ok(scan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fp_sources::{HttpConfig, WebDavSource};
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::thread;

    fn dav<P: AsRef<str> + Send + 'static>(
        responses: Vec<(P, &'static str, String)>,
    ) -> (WebDavSource, thread::JoinHandle<()>) {
        let socket = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/videos/", socket.local_addr().unwrap());
        let source = WebDavSource::new(&HttpConfig {
            id: "dav".into(),
            name: "Test server".into(),
            url,
            credentials: None,
            insecure_tls: false,
            parallel_downloads: false,
        })
        .unwrap();
        let server = thread::spawn(move || {
            for (path, status, body) in responses {
                let (mut stream, _) = socket.accept().unwrap();
                let mut reader = BufReader::new(&mut stream);
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                assert_eq!(
                    line.trim_end(),
                    format!("PROPFIND {} HTTP/1.1", path.as_ref())
                );
                let mut length = 0;
                let mut depth_one = false;
                loop {
                    line.clear();
                    if reader.read_line(&mut line).unwrap() == 0 || line == "\r\n" {
                        break;
                    }
                    let (name, value) = line.trim_end().split_once(':').unwrap();
                    if name.eq_ignore_ascii_case("content-length") {
                        length = value.trim().parse().unwrap();
                    }
                    depth_one |= name.eq_ignore_ascii_case("depth") && value.trim() == "1";
                }
                assert!(depth_one);
                reader.read_exact(&mut vec![0; length]).unwrap();
                write!(
                    stream,
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .unwrap();
            }
        });
        (source, server)
    }

    fn dav_list(entries: &[(&str, bool, u64)]) -> String {
        let mut xml = String::from("<D:multistatus xmlns:D='DAV:'>");
        for (href, directory, size) in entries {
            let collection = if *directory { "<D:collection/>" } else { "" };
            xml.push_str(&format!(
                "<D:response><D:href>{href}</D:href><D:propstat><D:prop>\
                 <D:resourcetype>{collection}</D:resourcetype>\
                 <D:getcontentlength>{size}</D:getcontentlength></D:prop>\
                 <D:status>HTTP/1.1 200 OK</D:status></D:propstat></D:response>"
            ));
        }
        xml.push_str("</D:multistatus>");
        xml
    }

    fn options(max_videos: Option<usize>) -> ScanOptions<'static> {
        ScanOptions {
            sort: EntrySort::Name,
            filter: "",
            max_videos,
        }
    }

    #[test]
    fn nested_webdav_folders_stay_in_scope_and_keep_locations_and_metadata() {
        let (source, server) = dav(vec![
            (
                "/videos/selected/",
                "207 Multi-Status",
                dav_list(&[
                    ("/videos/%73elected/Root%20Clip.mp4", false, 100),
                    ("/videos/selected/Folder%20A/", true, 0),
                    ("/videos/selected/Notes.txt", false, 10),
                    ("/videos/sibling/", true, 0),
                    ("/videos/sibling/Clip%20Root.mp4", false, 100),
                    ("/outside/Clip%20Root.mp4", false, 100),
                    ("http://127.0.0.1:1/elsewhere/", true, 0),
                ]),
            ),
            (
                "/videos/selected/Folder%20A/",
                "207 Multi-Status",
                dav_list(&[
                    ("/videos/selected/Folder%20A/FolderB/", true, 0),
                    ("/videos/selected/Folder%20A/Root%20ignored.mp4", false, 400),
                ]),
            ),
            (
                "/videos/selected/Folder%20A/FolderB/",
                "207 Multi-Status",
                dav_list(&[
                    (
                        "/videos/selected/Folder%20A/FolderB/Clip%20Root_180_LR.mkv",
                        false,
                        300,
                    ),
                    (
                        "/videos/selected/Folder%20A/FolderB/Clip%20Root_180_LR.mkv",
                        false,
                        300,
                    ),
                ]),
            ),
        ]);
        let root = fp_sources::location::encode("dav", "webdav", "selected/");
        let entries = source.list(Some(&root)).unwrap();
        let scan = collect(
            &source,
            Some(&root),
            entries,
            ScanOptions {
                sort: EntrySort::Largest,
                filter: "cLiP root",
                ..options(Some(2))
            },
            || false,
            || panic!("limited scan warned"),
        )
        .unwrap();
        server.join().unwrap();
        assert_eq!(scan.incomplete, None);
        let videos = scan.videos;
        assert_eq!(videos.len(), 2);
        assert_eq!(videos[0].name, "Clip Root_180_LR.mkv");
        assert_eq!(videos[0].size, Some(300));
        assert!(videos[0].format.is_some());
        assert_eq!(
            fp_sources::location::decode(&videos[0].location)
                .unwrap()
                .key,
            "selected/Folder%20A/FolderB/Clip%20Root_180_LR.mkv"
        );
        assert_eq!(videos[1].name, "Root Clip.mp4");
    }

    #[test]
    fn inaccessible_children_are_reported_and_other_children_are_still_listed() {
        let (source, server) = dav(vec![
            (
                "/videos/",
                "207 Multi-Status",
                dav_list(&[
                    ("/videos/clip.mp4", false, 1),
                    ("/videos/gone/", true, 0),
                    ("/videos/good/", true, 0),
                ]),
            ),
            ("/videos/gone/", "403 Forbidden", String::new()),
            (
                "/videos/good/",
                "207 Multi-Status",
                dav_list(&[("/videos/good/child.mp4", false, 1)]),
            ),
        ]);
        let entries = source.list(None).unwrap();
        let scan = collect(&source, None, entries, options(Some(3)), || false, || {}).unwrap();
        server.join().unwrap();
        assert_eq!(scan.incomplete, Some(Incomplete::Unavailable));
        let names: Vec<_> = scan.videos.into_iter().map(|entry| entry.name).collect();
        assert_eq!(names, ["child.mp4", "clip.mp4"]);
    }

    #[test]
    fn the_video_limit_stops_the_walk_before_further_folders() {
        let (source, server) = dav(vec![
            (
                "/videos/",
                "207 Multi-Status",
                dav_list(&[
                    ("/videos/root.mp4", false, 1),
                    ("/videos/a/", true, 0),
                    ("/videos/untouched/", true, 0),
                ]),
            ),
            (
                "/videos/a/",
                "207 Multi-Status",
                dav_list(&[
                    ("/videos/a/a1.mp4", false, 1),
                    ("/videos/a/a2.mp4", false, 1),
                ]),
            ),
        ]);
        let entries = source.list(None).unwrap();
        let scan = collect(
            &source,
            None,
            entries,
            options(Some(2)),
            || false,
            || panic!("limited scan warned"),
        )
        .unwrap();
        server.join().unwrap();
        assert_eq!(scan.incomplete, Some(Incomplete::LimitReached(2)));
        let names: Vec<_> = scan.videos.into_iter().map(|entry| entry.name).collect();
        assert_eq!(names, ["a1.mp4", "root.mp4"]);
    }

    #[test]
    fn an_unlimited_scan_warns_once_past_the_default_limit_and_takes_every_video() {
        let per_folder = DEFAULT_PLAY_ALL_LIMIT / 3 + 1;
        let folders: Vec<_> = (0..3).map(|n| format!("/videos/folder-{n}/")).collect();
        let root_children: Vec<_> = folders
            .iter()
            .map(|folder| (folder.as_str(), true, 0))
            .collect();
        let mut responses = vec![(
            "/videos/".to_string(),
            "207 Multi-Status",
            dav_list(&root_children),
        )];
        responses.extend(folders.into_iter().map(|folder| {
            let videos: Vec<_> = (0..per_folder)
                .map(|n| format!("{folder}clip-{n}.mp4"))
                .collect();
            let list: Vec<_> = videos.iter().map(|v| (v.as_str(), false, 10)).collect();
            let body = dav_list(&list);
            (folder, "207 Multi-Status", body)
        }));
        let (source, server) = dav(responses);
        let entries = source.list(None).unwrap();
        let warnings = std::cell::Cell::new(0);
        let scan = collect(
            &source,
            None,
            entries,
            options(None),
            || false,
            || warnings.set(warnings.get() + 1),
        )
        .unwrap();
        assert_eq!(scan.incomplete, None);
        assert_eq!(warnings.get(), 1);
        assert_eq!(scan.videos.len(), 3 * per_folder);
        server.join().unwrap();
    }

    #[test]
    fn an_out_of_source_webdav_folder_is_rejected_before_listing_it() {
        let (source, server) = dav(Vec::<(&str, _, _)>::new());
        let result = collect(
            &source,
            Some("http://127.0.0.1:1/elsewhere/"),
            Vec::new(),
            options(Some(3)),
            || false,
            || {},
        );
        assert!(matches!(
            result,
            Err(fp_sources::Error::InvalidLocation { .. })
        ));
        server.join().unwrap();
    }

    #[test]
    fn cancellation_stops_before_listing_the_next_folder() {
        let (source, server) = dav(vec![(
            "/videos/",
            "207 Multi-Status",
            dav_list(&[("/videos/child/", true, 0)]),
        )]);
        let entries = source.list(None).unwrap();
        let checks = std::cell::Cell::new(0);
        let result = collect(
            &source,
            None,
            entries,
            options(None),
            || {
                let checked = checks.get();
                checks.set(checked + 1);
                checked > 0
            },
            || panic!("cancelled scan warned"),
        );
        assert!(matches!(result, Err(fp_sources::Error::Io(error))
            if error.kind() == std::io::ErrorKind::Interrupted));
        server.join().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_local_symlink_to_an_ancestor_does_not_expand_the_tree() {
        let root = std::env::temp_dir().join(format!("fp-folder-walk-{}", std::process::id()));
        let subfolder = root.join("subfolder");
        std::fs::create_dir_all(&subfolder).unwrap();
        std::fs::write(root.join("root.mp4"), []).unwrap();
        std::fs::write(subfolder.join("nested.mp4"), []).unwrap();
        std::os::unix::fs::symlink("..", subfolder.join("back")).unwrap();
        let source = fp_sources::local::LocalSource::new(&fp_sources::LocalConfig {
            id: "local".into(),
            name: "Test folder".into(),
            root: root.clone(),
        });
        let entries = source.list(root.to_str()).unwrap();
        let scan = collect(
            &source,
            root.to_str(),
            entries,
            options(Some(10)),
            || false,
            || {},
        );
        std::fs::remove_dir_all(root).unwrap();
        let scan = scan.unwrap();
        assert_eq!(scan.incomplete, None);
        let names: Vec<_> = scan.videos.into_iter().map(|e| e.name).collect();
        assert_eq!(names, ["nested.mp4", "root.mp4"]);
    }
}
