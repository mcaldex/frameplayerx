//! Blu-ray disc images (`.iso`): the main title, played clip after clip
//! straight out of the image. 3D titles play from their clips' SSIF files,
//! which interleave both views. Menus and the other titles aren't offered.

pub(crate) mod mpls;
pub(crate) mod udf;

use crate::info::Chapter;
use crate::{Error, Result};
use fp_core::ByteSource;
use std::sync::Arc;

/// One clip of a title: its file in the image and the stretch of it played,
/// in 90 kHz ticks of the clip's clock.
pub(crate) struct Clip {
    pub src: Arc<dyn ByteSource>,
    pub name: String,
    pub in_time: i64,
    pub out_time: i64,
    /// Where on the title it starts, seconds.
    pub start: f64,
}

pub(crate) struct Title {
    pub clips: Vec<Clip>,
    pub duration: f64,
    pub chapters: Vec<Chapter>,
    /// The audio and subtitle streams' languages by PID.
    pub languages: Vec<(u16, String)>,
}

/// Whether `name` is a disc image this module plays.
pub(crate) fn is_disc_image(name: &str) -> bool {
    let name = name.split(['?', '#']).next().unwrap_or(name);
    name.rsplit('.')
        .next()
        .is_some_and(|e| e.eq_ignore_ascii_case("iso"))
}

/// The main title of the Blu-ray image in `src`: its longest playlist,
/// preferring one that plays no clip twice (discs hide the real one among
/// decoys that do).
pub(crate) fn main_title(src: Arc<dyn ByteSource>) -> Result<Title> {
    let fs = udf::Udf::open(src)?;
    let playlists = fs
        .list("BDMV/PLAYLIST")
        .map_err(|_| Error::Disc("no BDMV/PLAYLIST: not a Blu-ray (a DVD?)".into()))?;
    let mut best: Option<(mpls::Playlist, (u64, bool))> = None;
    for (name, icb) in playlists {
        if !name.to_ascii_lowercase().ends_with(".mpls") {
            continue;
        }
        let Some(pl) = fs
            .node(icb)
            .and_then(|n| fs.read_file(&n))
            .ok()
            .and_then(|b| mpls::parse(&b))
        else {
            continue;
        };
        let mut clips: Vec<&str> = pl.items.iter().map(|i| i.clip.as_str()).collect();
        clips.sort_unstable();
        let unique = clips.windows(2).all(|w| w[0] != w[1]);
        let rank = (pl.duration().round() as u64, unique);
        if pl.duration() > 0.0 && best.as_ref().is_none_or(|(_, r)| rank > *r) {
            best = Some((pl, rank));
        }
    }
    let (pl, _) = best.ok_or_else(|| Error::Disc("no title to play".into()))?;
    let streams = fs.list("BDMV/STREAM")?;
    let ssifs = fs.list("BDMV/STREAM/SSIF").unwrap_or_default();
    let find = |list: &[(String, udf::Icb)], name: &str| {
        list.iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, icb)| *icb)
    };
    let ssif = |clip: &str| find(&ssifs, &format!("{clip}.ssif"));
    let stereo = pl.items.iter().all(|i| ssif(&i.clip).is_some());
    let mut clips = Vec::new();
    let mut start = 0.0;
    for item in &pl.items {
        let file = if stereo {
            format!("{}.ssif", item.clip)
        } else {
            format!("{}.m2ts", item.clip)
        };
        let icb = find(if stereo { &ssifs } else { &streams }, &file)
            .ok_or_else(|| Error::Disc(format!("no {file}")))?;
        let node = fs.node(icb)?;
        let length = f64::from(item.out_time.saturating_sub(item.in_time)) / 45_000.0;
        clips.push(Clip {
            src: fs.file(&node, file.clone()),
            name: file,
            in_time: i64::from(item.in_time) * 2,
            out_time: i64::from(item.out_time) * 2,
            start,
        });
        start += length;
    }
    let chapters: Vec<f64> = pl
        .marks
        .iter()
        .map(|&(item, t)| {
            clips[item].start + f64::from(t.saturating_sub(pl.items[item].in_time)) / 45_000.0
        })
        .collect();
    let chapters = chapters
        .iter()
        .enumerate()
        .map(|(n, &at)| Chapter {
            start: at,
            end: chapters.get(n + 1).copied().unwrap_or(start),
            title: format!("Chapter {}", n + 1),
        })
        .collect();
    Ok(Title {
        clips,
        duration: start,
        chapters,
        languages: pl.languages,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::Packet;
    use crate::input::Input;
    use fp_core::source::MemorySource;
    use fp_ffmpeg_sys as ff;
    use std::sync::atomic::AtomicBool;

    const S: usize = 2048;
    /// The partition's first sector, past the anchor at 256.
    const PART: usize = 300;

    /// Writes a descriptor tag with a valid checksum.
    fn tag(b: &mut [u8], id: u16) {
        b[..2].copy_from_slice(&id.to_le_bytes());
        b[2..4].copy_from_slice(&3u16.to_le_bytes());
        b[4] = 0;
        b[4] = b[..16].iter().fold(0u8, |s, v| s.wrapping_add(*v));
    }

    fn put32(b: &mut [u8], at: usize, v: u32) {
        b[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }

    /// A UDF image with one physical partition, written block by block.
    struct Image {
        data: Vec<u8>,
        next: u32,
    }

    impl Image {
        fn new() -> Image {
            let mut img = Image {
                data: vec![0; (PART + 1) * S],
                next: 1,
            };
            let avdp = &mut img.data[256 * S..257 * S];
            put32(avdp, 16, 3 * S as u32);
            put32(avdp, 20, 32);
            tag(avdp, 2);
            let pd = &mut img.data[32 * S..33 * S];
            put32(pd, 188, PART as u32);
            tag(pd, 5);
            let lvd = &mut img.data[33 * S..34 * S];
            put32(lvd, 212, S as u32);
            put32(lvd, 248, S as u32);
            put32(lvd, 264, 6);
            put32(lvd, 268, 1);
            lvd[440..446].copy_from_slice(&[1, 6, 1, 0, 0, 0]);
            tag(lvd, 6);
            tag(&mut img.data[34 * S..35 * S], 8);
            img
        }

        fn block(&mut self, n: u32) -> &mut [u8] {
            let at = (PART + n as usize) * S;
            if self.data.len() < at + S {
                self.data.resize(at + S, 0);
            }
            &mut self.data[at..at + S]
        }

        /// `data` in new blocks; their first.
        fn put(&mut self, data: &[u8]) -> u32 {
            let first = self.next;
            self.next += data.len().div_ceil(S).max(1) as u32;
            for (i, c) in data.chunks(S).enumerate() {
                self.block(first + i as u32)[..c.len()].copy_from_slice(c);
            }
            first
        }

        /// A file entry: directory or file, `size` bytes, its allocation
        /// descriptors of `kind`.
        fn entry(&mut self, dir: bool, size: usize, kind: u16, ads: &[u8]) -> u32 {
            let at = self.put(&[0]);
            let b = self.block(at);
            b[27] = if dir { 4 } else { 5 };
            b[34..36].copy_from_slice(&kind.to_le_bytes());
            b[56..64].copy_from_slice(&(size as u64).to_le_bytes());
            put32(b, 172, ads.len() as u32);
            b[176..176 + ads.len()].copy_from_slice(ads);
            tag(b, 261);
            at
        }

        /// A file whose data is in one run.
        fn file(&mut self, data: &[u8]) -> u32 {
            let at = self.put(data);
            self.entry(false, data.len(), 0, &short(data.len(), at))
        }

        /// A directory of (name, entry, is a directory).
        fn dir(&mut self, entries: &[(&str, u32, bool)], embedded: bool) -> u32 {
            let mut fids = fid("", 0, 0x0a);
            for (name, at, dir) in entries {
                fids.extend(fid(name, *at, if *dir { 2 } else { 0 }));
            }
            if embedded {
                return self.entry(true, fids.len(), 3, &fids);
            }
            let at = self.put(&fids);
            self.entry(true, fids.len(), 0, &short(fids.len(), at))
        }

        fn finish(mut self, root: u32) -> Arc<MemorySource> {
            let fsd = self.block(0);
            put32(fsd, 400, S as u32);
            put32(fsd, 404, root);
            tag(fsd, 256);
            Arc::new(MemorySource(self.data))
        }
    }

    fn short(len: usize, block: u32) -> Vec<u8> {
        let mut v = (len as u32).to_le_bytes().to_vec();
        v.extend_from_slice(&block.to_le_bytes());
        v
    }

    fn long(len: usize, block: u32) -> Vec<u8> {
        let mut v = short(len, block);
        v.extend_from_slice(&[0; 8]);
        v
    }

    /// A file identifier descriptor.
    fn fid(name: &str, at: u32, flags: u8) -> Vec<u8> {
        let mut f = vec![0u8; 38];
        f[18] = flags;
        if !name.is_empty() {
            f[19] = name.len() as u8 + 1;
            f.push(8);
            f.extend_from_slice(name.as_bytes());
        }
        put32(&mut f, 20, S as u32);
        put32(&mut f, 24, at);
        f.resize(f.len().div_ceil(4) * 4, 0);
        tag(&mut f, 257);
        f
    }

    /// Clips A and B: the test video on clocks starting at 10 s and 100 s.
    fn clips() -> (Vec<u8>, Vec<u8>) {
        let a = crate::ts_fixture::h264_aac_ts(10.0).0.clone();
        let b = crate::ts_fixture::h264_aac_ts(100.0).0.clone();
        (a, b)
    }

    /// A disc whose main title plays A then B (2 s each), with chapters at
    /// its start, 1 s in and B's start, beside a shorter playlist of A.
    fn disc() -> Arc<MemorySource> {
        let (a, b) = clips();
        let mut img = Image::new();
        let main = mpls::tests::playlist(
            &[("00001", 450_000, 540_000), ("00002", 4_500_000, 4_590_000)],
            &[(0, 450_000), (0, 495_000), (1, 4_500_000)],
        );
        let at = img.put(&main);
        let main = img.entry(false, main.len(), 1, &long(main.len(), at));
        let decoy = mpls::tests::playlist(&[("00001", 450_000, 500_000)], &[]);
        let decoy = img.file(&decoy);
        let playlists = img.dir(
            &[("00000.mpls", main, false), ("00001.mpls", decoy, false)],
            false,
        );
        let a = img.file(&a);
        // B split across two runs, the second listed in a continuation.
        let half = (b.len() / 2).div_ceil(S) * S;
        let second = img.put(&b[half..]);
        let first = img.put(&b[..half]);
        let aed = img.put(&[0]);
        {
            let rest = short(b.len() - half, second);
            let blk = img.block(aed);
            put32(blk, 20, rest.len() as u32);
            blk[24..24 + rest.len()].copy_from_slice(&rest);
            tag(blk, 258);
        }
        let mut ads = short(half, first);
        ads.extend(short(S, aed));
        let n = ads.len();
        ads[n - 5] |= 0xc0;
        let b = img.entry(false, b.len(), 0, &ads);
        let stream = img.dir(&[("00001.M2TS", a, false), ("00002.m2ts", b, false)], true);
        let bdmv = img.dir(
            &[("PLAYLIST", playlists, true), ("STREAM", stream, true)],
            false,
        );
        let root = img.dir(&[("BDMV", bdmv, true)], false);
        img.finish(root)
    }

    #[test]
    fn reads_files_of_the_image() {
        let (a, b) = clips();
        let fs = udf::Udf::open(disc()).unwrap();
        let names: Vec<_> = fs
            .list("bdmv/stream")
            .unwrap()
            .into_iter()
            .map(|e| e.0)
            .collect();
        assert_eq!(names, ["00001.M2TS", "00002.m2ts"]);
        let file = |path: &str| fs.read_file(&fs.lookup(path).unwrap()).unwrap();
        assert_eq!(file("BDMV/STREAM/00001.m2ts"), a);
        assert_eq!(file("BDMV/STREAM/00002.m2ts"), b, "split and continued");
        assert!(file("BDMV/PLAYLIST/00000.mpls").starts_with(b"MPLS"));
        assert!(fs.lookup("BDMV/STREAM/00003.m2ts").is_err());
        let not_udf = Arc::new(MemorySource(vec![0; 300 * S]));
        assert!(udf::Udf::open(not_udf).is_err());
    }

    #[test]
    fn picks_the_longest_playlist() {
        let title = main_title(disc()).unwrap();
        let names: Vec<_> = title.clips.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["00001.m2ts", "00002.m2ts"]);
        assert_eq!(title.duration, 4.0);
        assert_eq!(
            (title.clips[1].in_time, title.clips[1].start),
            (9_000_000, 2.0)
        );
        let chapters: Vec<_> = title.chapters.iter().map(|c| (c.start, c.end)).collect();
        assert_eq!(chapters, [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0)]);
        assert_eq!(title.languages[0], (0x1100, "eng".to_string()));
    }

    /// Times (title seconds) of the video packets read from here on.
    fn video_times(input: &mut Input) -> Vec<f64> {
        let video = input.best_stream(ff::AVMEDIA_TYPE_VIDEO).unwrap();
        // SAFETY: the stream is valid while the input is open.
        let tb = crate::q2d(unsafe { (*input.streams()[video]).time_base });
        let mut pkt = Packet::new();
        let mut out = Vec::new();
        while input.read(pkt.as_ptr()).unwrap() {
            // SAFETY: a packet just read.
            let p = unsafe { &*pkt.as_ptr() };
            if pkt.stream_index() == video {
                out.push(p.pts as f64 * tb - input.origin());
            }
            pkt.unref();
        }
        out
    }

    #[test]
    fn plays_the_title_clip_after_clip() {
        let stop = Arc::new(AtomicBool::new(false));
        let mut input = Input::open(disc(), "Movie.iso", stop).unwrap();
        let info = crate::info::read_info(&input);
        assert_eq!((info.container.as_str(), info.duration), ("bluray", 4.0));
        assert_eq!(info.chapters.len(), 3);
        let audio = info.audio_streams().next().unwrap();
        assert_eq!(
            audio.language, None,
            "the fixture's PIDs aren't the playlist's"
        );
        let times = video_times(&mut input);
        let (a, b): (Vec<f64>, Vec<f64>) = times.iter().partition(|t| **t < 2.0);
        assert_eq!(a.len(), b.len(), "both clips whole");
        assert!(a.iter().all(|t| *t >= 0.0) && a[0] < 0.1);
        assert!(
            (b.iter().copied().fold(f64::MAX, f64::min) - 2.0).abs() < 0.1,
            "B starts at 2 s"
        );
        assert!(b.iter().all(|t| *t < 4.0));
        // Into B, back into A.
        input.seek(3.0).unwrap();
        let after = video_times(&mut input);
        assert!(
            after[0] > 2.0 && after[0] <= 3.0 + 0.1,
            "landed at {}",
            after[0]
        );
        input.seek(0.5).unwrap();
        let after = video_times(&mut input);
        assert!(after[0] < 1.0, "landed at {}", after[0]);
        assert_eq!(
            after.iter().filter(|t| **t >= 2.0).count(),
            b.len(),
            "B again"
        );
    }

    #[test]
    fn names_disc_images() {
        assert!(is_disc_image("smb://nas/Movies/Avatar 3D.ISO"));
        assert!(is_disc_image("http://srv/x.iso?token=1"));
        assert!(!is_disc_image("/v/a.m2ts"));
    }
}
