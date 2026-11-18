//! Subtitle cues: decoded from embedded streams or external files.

use crate::decode::{Decoder, HwDecode, Packet};
use crate::input::Input;
use crate::{Result, check};
use fp_core::ByteSource;
use fp_ffmpeg_sys as ff;
use std::ffi::CStr;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// A bitmap subtitle (PGS, DVD, DVB) in RGBA.
#[derive(Clone, Debug, PartialEq)]
pub struct SubBitmap {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// One timed subtitle.
#[derive(Clone, Debug, PartialEq)]
pub struct Cue {
    pub start: f64,
    pub end: f64,
    /// Plain text with line breaks; empty for bitmap cues.
    pub text: String,
    pub bitmaps: Vec<SubBitmap>,
}

/// Strips an ASS `Dialogue` payload to plain text: drops the leading fields
/// and `{...}` override blocks, turns `\N`/`\n` into newlines, `\h` into spaces.
pub fn ass_to_text(ass: &str) -> String {
    // FFmpeg's ASS packets: ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,Effect,Text
    let text = ass.splitn(9, ',').nth(8).unwrap_or(ass);
    let mut out = String::with_capacity(text.len());
    let mut depth = 0;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' => depth += 1,
            '}' if depth > 0 => depth -= 1,
            _ if depth > 0 => {}
            '\\' => match chars.peek() {
                Some('N') | Some('n') => {
                    chars.next();
                    out.push('\n');
                }
                Some('h') => {
                    chars.next();
                    out.push(' ');
                }
                _ => out.push('\\'),
            },
            _ => out.push(c),
        }
    }
    out.trim().to_string()
}

/// Converts a decoded `AVSubtitle` to cues. `pkt_pts`/`pkt_duration` are in
/// seconds and used when the subtitle carries no end time. One with nothing
/// in it, which picture subtitles send to clear the screen, is an empty cue
/// (see [`add`]).
pub(crate) fn convert(sub: &ff::AVSubtitle, pkt_pts: f64, pkt_duration: f64) -> Option<Cue> {
    let base = if sub.pts != ff::AV_NOPTS_VALUE {
        sub.pts as f64 / ff::AV_TIME_BASE as f64
    } else {
        pkt_pts
    };
    let start = base + sub.start_display_time as f64 / 1000.0;
    if sub.num_rects == 0 {
        return Some(Cue {
            start,
            end: start,
            text: String::new(),
            bitmaps: Vec::new(),
        });
    }
    let end = if sub.end_display_time > 0 && sub.end_display_time != u32::MAX {
        base + sub.end_display_time as f64 / 1000.0
    } else if pkt_duration > 0.0 {
        pkt_pts + pkt_duration
    } else {
        start + 4.0
    };
    let mut text = Vec::new();
    let mut bitmaps = Vec::new();
    if sub.rects.is_null() {
        return None;
    }
    // SAFETY: rects holds num_rects valid pointers owned by `sub`.
    for &r in unsafe { std::slice::from_raw_parts(sub.rects, sub.num_rects as usize) } {
        // SAFETY: as above.
        let r = unsafe { &*r };
        match r.type_ {
            ff::SUBTITLE_ASS if !r.ass.is_null() => {
                // SAFETY: non-null, NUL-terminated and owned by `sub`.
                let ass = unsafe { CStr::from_ptr(r.ass) };
                text.push(ass_to_text(&ass.to_string_lossy()));
            }
            ff::SUBTITLE_TEXT if !r.text.is_null() => {
                // SAFETY: non-null, NUL-terminated and owned by `sub`.
                let t = unsafe { CStr::from_ptr(r.text) };
                text.push(t.to_string_lossy().trim().to_string());
            }
            ff::SUBTITLE_BITMAP
                if r.w > 0 && r.h > 0 && !r.data[0].is_null() && !r.data[1].is_null() =>
            {
                let (w, h) = (r.w as usize, r.h as usize);
                // SAFETY: data[0] is w*h palette indices with linesize[0];
                // data[1] is a 256-entry RGBA (native-endian u32) palette.
                let palette = unsafe { std::slice::from_raw_parts(r.data[1] as *const u32, 256) };
                let mut rgba = Vec::with_capacity(w * h * 4);
                for y in 0..h {
                    // SAFETY: row y of data[0], see above.
                    let row = unsafe {
                        std::slice::from_raw_parts(r.data[0].add(y * r.linesize[0] as usize), w)
                    };
                    for &i in row {
                        let c = palette[i as usize];
                        rgba.extend_from_slice(&[
                            (c >> 16) as u8,
                            (c >> 8) as u8,
                            c as u8,
                            (c >> 24) as u8,
                        ]);
                    }
                }
                bitmaps.push(SubBitmap {
                    x: r.x,
                    y: r.y,
                    width: w as u32,
                    height: h as u32,
                    rgba,
                });
            }
            _ => {}
        }
    }
    let text = text
        .into_iter()
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    if text.is_empty() && bitmaps.is_empty() {
        return None;
    }
    Some(Cue {
        start,
        end,
        text,
        bitmaps,
    })
}

pub(crate) fn decode_packet(dec: &mut Decoder, pkt: &Packet) -> Option<Cue> {
    // SAFETY: decoder open for a subtitle stream; sub freed after use.
    unsafe {
        let mut sub: ff::AVSubtitle = std::mem::zeroed();
        let mut got = 0;
        let r = ff::avcodec_decode_subtitle2(dec.as_ptr(), &mut sub, &mut got, pkt.as_ptr());
        if r < 0 || got == 0 {
            return None;
        }
        let tb = crate::q2d(dec.time_base);
        let p = &*pkt.as_ptr();
        let pts = if p.pts != ff::AV_NOPTS_VALUE {
            p.pts as f64 * tb
        } else {
            0.0
        };
        let cue = convert(&sub, pts, p.duration as f64 * tb).map(|mut c| {
            c.start -= dec.origin;
            c.end -= dec.origin;
            c
        });
        ff::avsubtitle_free(&mut sub);
        cue
    }
}

/// Loads every cue from a subtitle file (SRT, ASS, VTT) or from the first
/// subtitle stream of any container.
pub fn load_file(src: Arc<dyn ByteSource>, name: &str) -> Result<Vec<Cue>> {
    let mut input = Input::open(src, name, Arc::new(AtomicBool::new(false)))?;
    let idx = input
        .best_stream(ff::AVMEDIA_TYPE_SUBTITLE)
        .ok_or(crate::Error::NoStream("subtitle"))?;
    let mut dec = Decoder::open(&input, idx, HwDecode::Off)?;
    let mut decoded = Vec::new();
    let mut pkt = Packet::new();
    while input.read(pkt.as_ptr())? {
        if pkt.stream_index() == idx
            && let Some(c) = decode_packet(&mut dec, &pkt)
        {
            decoded.push(c);
        }
        pkt.unref();
    }
    check(0, "")?;
    // In order first, so each goes on the end: a file out of order would
    // have each one moving all those after it.
    decoded.sort_by(|a, b| a.start.total_cmp(&b.start));
    let mut cues = Vec::with_capacity(decoded.len());
    for c in decoded {
        add(&mut cues, c);
    }
    Ok(cues)
}

/// Adds a decoded cue to `cues`, kept in order of start. Picture subtitles
/// (PGS, DVD) give no end time: a picture lasts until the empty cue that
/// clears the screen, or until the next picture replaces it.
pub(crate) fn add(cues: &mut Vec<Cue>, cue: Cue) {
    let clear = cue.text.is_empty() && cue.bitmaps.is_empty();
    if clear || !cue.bitmaps.is_empty() {
        let before = cues.partition_point(|c| c.start < cue.start);
        for c in &mut cues[..before] {
            let replaced = !cue.bitmaps.is_empty() && !c.bitmaps.is_empty();
            if (clear || replaced) && cue.start < c.end {
                c.end = cue.start;
            }
        }
    }
    if clear {
        return;
    }
    // A seek reads cues again.
    let near = cues.partition_point(|c| c.start < cue.start - 1e-3)
        ..cues.partition_point(|c| c.start <= cue.start + 1e-3);
    if cues[near]
        .iter()
        .any(|c| (c.start - cue.start).abs() < 1e-3 && c.text == cue.text)
    {
        return;
    }
    let at = cues.partition_point(|c| c.start <= cue.start);
    cues.insert(at, cue);
}

/// Cues shown at once at most, however many a file has overlap.
const MAX_ACTIVE: usize = 64;

/// Cues visible at time `t`.
pub fn active(cues: &[Cue], t: f64) -> Vec<&Cue> {
    let started = cues.partition_point(|c| c.start <= t);
    cues[..started]
        .iter()
        .filter(|c| t < c.end)
        .take(MAX_ACTIVE)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fp_core::source::FileSource;

    #[test]
    fn strips_ass() {
        assert_eq!(
            ass_to_text("0,0,Default,,0,0,0,,Hello {\\i1}world{\\i0}"),
            "Hello world"
        );
        assert_eq!(
            ass_to_text("1,0,Default,,0,0,0,,Line one\\NLine, two\\hok"),
            "Line one\nLine, two ok"
        );
        assert_eq!(ass_to_text("plain"), "plain");
    }

    #[test]
    fn loads_srt_file() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/subs.srt");
        let cues = load_file(Arc::new(FileSource::open(&path).unwrap()), "subs.srt").unwrap();
        assert_eq!(cues.len(), 2);
        assert!(
            (cues[0].start - 0.2).abs() < 1e-6 && (cues[0].end - 1.0).abs() < 1e-6,
            "{:?}",
            cues[0]
        );
        assert_eq!(cues[0].text, "Hello world");
        assert_eq!(cues[1].text, "Second line\nwith break");
        assert_eq!(active(&cues, 0.5).len(), 1);
        assert_eq!(active(&cues, 1.1).len(), 0);
    }

    #[test]
    fn a_picture_shows_until_cleared_or_replaced() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/pgs.mkv");
        let cues = load_file(Arc::new(FileSource::open(&path).unwrap()), "pgs.mkv").unwrap();
        let times: Vec<_> = cues.iter().map(|c| (c.start, c.end)).collect();
        assert_eq!(times, [(0.5, 1.5), (2.0, 3.0), (3.0, 3.5)]);
        assert!(
            cues.iter()
                .all(|c| c.bitmaps.len() == 1 && c.text.is_empty())
        );
        assert_eq!(active(&cues, 1.0).len(), 1);
        assert_eq!(active(&cues, 1.7).len(), 0);
        assert_eq!(active(&cues, 2.5).len(), 1);
    }

    fn cue(start: f64, end: f64, text: &str) -> Cue {
        Cue {
            start,
            end,
            text: text.into(),
            bitmaps: Vec::new(),
        }
    }

    #[test]
    fn text_cues_may_overlap() {
        let mut cues = Vec::new();
        add(&mut cues, cue(1.0, 4.0, "Over there"));
        add(&mut cues, cue(2.0, 3.0, "Sign"));
        add(&mut cues, cue(1.0, 4.0, "Over there"));
        assert_eq!(cues, [cue(1.0, 4.0, "Over there"), cue(2.0, 3.0, "Sign")]);
    }

    #[test]
    fn a_file_of_many_cues_loads_without_rescanning_them() {
        let mut cues = Vec::new();
        for i in 0..100_000 {
            add(&mut cues, cue(f64::from(i), f64::from(i) + 0.5, "Line"));
        }
        assert_eq!(cues.len(), 100_000);
        // A seek reading one again adds nothing.
        add(&mut cues, cue(500.0, 500.5, "Line"));
        assert_eq!(cues.len(), 100_000);
        assert_eq!(active(&cues, 500.2).len(), 1);
    }

    #[test]
    fn only_so_many_cues_show_at_once() {
        let mut cues = Vec::new();
        for i in 0..200 {
            add(
                &mut cues,
                cue(f64::from(i) * 0.01, 10.0, &format!("Line {i}")),
            );
        }
        assert_eq!(active(&cues, 5.0).len(), MAX_ACTIVE);
    }

    #[test]
    fn loads_embedded_mkv_subtitles() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/hevc10_tb.mkv");
        let cues = load_file(Arc::new(FileSource::open(&path).unwrap()), "hevc10_tb.mkv").unwrap();
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[1].text, "Second line\nwith break");
        assert!((cues[1].end - 1.9).abs() < 1e-3, "{}", cues[1].end);
    }
}
