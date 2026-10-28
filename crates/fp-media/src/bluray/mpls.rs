//! Blu-ray playlists (`BDMV/PLAYLIST/*.mpls`): the clips a title plays in
//! turn, its chapter marks, and its streams' languages. Times are in the
//! playlist's 45 kHz ticks.

/// One clip of a title, from `in_time` to `out_time` of the clip's clock.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Item {
    /// The clip's name: `00800` for `STREAM/00800.m2ts`.
    pub clip: String,
    pub in_time: u32,
    pub out_time: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Playlist {
    pub items: Vec<Item>,
    /// Chapters: the item each starts in and its time on that item's clip.
    pub marks: Vec<(usize, u32)>,
    /// The audio and subtitle streams of the first item: PID and language.
    pub languages: Vec<(u16, String)>,
}

impl Playlist {
    /// Seconds.
    pub(crate) fn duration(&self) -> f64 {
        self.items
            .iter()
            .map(|i| f64::from(i.out_time.saturating_sub(i.in_time)) / 45_000.0)
            .sum()
    }
}

/// Big-endian reads that give up (None) past the end.
struct Bytes<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Bytes<'a> {
    fn u8(&mut self) -> Option<u8> {
        let v = *self.b.get(self.at)?;
        self.at += 1;
        Some(v)
    }
    fn u16(&mut self) -> Option<u16> {
        Some(u16::from(self.u8()?) << 8 | u16::from(self.u8()?))
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from(self.u16()?) << 16 | u32::from(self.u16()?))
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.b.get(self.at..self.at + n)?;
        self.at += n;
        Some(s)
    }
}

pub(crate) fn parse(b: &[u8]) -> Option<Playlist> {
    if b.get(..4)? != b"MPLS" {
        return None;
    }
    let mut r = Bytes { b, at: 8 };
    let (list_at, marks_at) = (r.u32()? as usize, r.u32()? as usize);
    let mut r = Bytes { b, at: list_at + 6 };
    let items = r.u16()?;
    r.u16()?;
    let mut pl = Playlist::default();
    for i in 0..items {
        let len = usize::from(r.u16()?);
        let end = r.at + len;
        let clip = String::from_utf8_lossy(r.take(5)?).into_owned();
        r.take(4)?;
        let multi_angle = r.u16()? & 0x10 != 0;
        r.u8()?;
        let (in_time, out_time) = (r.u32()?, r.u32()?);
        // UO mask, random access flag, still mode and time.
        r.take(12)?;
        if multi_angle {
            let angles = usize::from(r.u8()?);
            r.u8()?;
            r.take(angles.saturating_sub(1) * 10)?;
        }
        if i == 0 {
            pl.languages = languages(&mut r).unwrap_or_default();
        }
        pl.items.push(Item {
            clip,
            in_time,
            out_time,
        });
        r.at = end;
    }
    let mut r = Bytes {
        b,
        at: marks_at + 4,
    };
    for _ in 0..r.u16().unwrap_or(0) {
        let Some(m) = r.take(14) else { break };
        let item = usize::from(u16::from_be_bytes([m[2], m[3]]));
        // Entry marks are chapters; link points aren't.
        if m[1] == 1 && item < pl.items.len() {
            pl.marks
                .push((item, u32::from_be_bytes([m[4], m[5], m[6], m[7]])));
        }
    }
    Some(pl)
}

/// The audio and subtitle (presentation graphics) streams of a play item's
/// stream number table.
fn languages(r: &mut Bytes) -> Option<Vec<(u16, String)>> {
    let len = usize::from(r.u16()?);
    let end = r.at + len;
    r.u16()?;
    let (video, audio, pg) = (r.u8()?, r.u8()?, r.u8()?);
    r.take(3)?;
    let pip_pg = r.u8()?;
    r.take(5)?;
    let mut out = Vec::new();
    for n in 0..u16::from(video) + u16::from(audio) + u16::from(pg) + u16::from(pip_pg) {
        // The stream entry: where the stream is.
        let elen = usize::from(r.u8()?);
        let eend = r.at + elen;
        let pid = match r.u8()? {
            1 => r.u16()?,
            2 => {
                r.take(2)?;
                r.u16()?
            }
            _ => {
                r.u8()?;
                r.u16()?
            }
        };
        r.at = eend;
        // Its attributes: the coding type, then the language.
        let alen = usize::from(r.u8()?);
        let aend = r.at + alen;
        let coding = r.u8()?;
        let lang = match coding {
            // Audio (a format byte first), and text subtitles (a
            // character set).
            0x03 | 0x04 | 0x80..=0x86 | 0xa1 | 0xa2 | 0x92 => {
                r.u8()?;
                Some(r.take(3)?)
            }
            0x90 | 0x91 => Some(r.take(3)?),
            _ => None,
        };
        r.at = aend;
        if n >= u16::from(video)
            && let Some(l) = lang.map(|l| String::from_utf8_lossy(l).trim().to_lowercase())
            && l.len() == 3
        {
            out.push((pid, l));
        }
    }
    (r.at <= end).then_some(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A playlist of `items` (clip, in, out), English and French audio and
    /// German subtitles on the first, and chapters at (item, time).
    pub(crate) fn playlist(items: &[(&str, u32, u32)], marks: &[(u16, u32)]) -> Vec<u8> {
        let mut list = Vec::new();
        for (n, (clip, a, b)) in items.iter().enumerate() {
            let mut item = Vec::new();
            item.extend_from_slice(clip.as_bytes());
            item.extend_from_slice(b"M2TS");
            item.extend_from_slice(&[0, 0x01, 0]);
            item.extend_from_slice(&a.to_be_bytes());
            item.extend_from_slice(&b.to_be_bytes());
            item.extend_from_slice(&[0; 12]);
            let mut stn = vec![0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
            let mut entry = |pid: u16, attrs: &[u8]| {
                stn.extend_from_slice(&[3, 1]);
                stn.extend_from_slice(&pid.to_be_bytes());
                stn.push(attrs.len() as u8);
                stn.extend_from_slice(attrs);
            };
            entry(0x1011, &[0x1b, 0x61]);
            entry(0x1100, &[0x83, 0x31, b'e', b'n', b'g']);
            entry(0x1101, &[0x81, 0x31, b'f', b'r', b'a']);
            entry(0x1200, &[0x90, b'd', b'e', b'u']);
            if n > 0 {
                stn.truncate(14);
                stn[3..7].fill(0);
            }
            item.extend_from_slice(&(stn.len() as u16).to_be_bytes());
            item.extend_from_slice(&stn);
            list.extend_from_slice(&(item.len() as u16).to_be_bytes());
            list.extend_from_slice(&item);
        }
        let mut b = b"MPLS0200".to_vec();
        let list_at = 40u32;
        let marks_at = list_at + 10 + list.len() as u32;
        b.extend_from_slice(&list_at.to_be_bytes());
        b.extend_from_slice(&marks_at.to_be_bytes());
        b.resize(list_at as usize, 0);
        b.extend_from_slice(&((list.len() + 6) as u32).to_be_bytes());
        b.extend_from_slice(&[0, 0]);
        b.extend_from_slice(&(items.len() as u16).to_be_bytes());
        b.extend_from_slice(&[0, 0]);
        b.extend_from_slice(&list);
        b.extend_from_slice(&((marks.len() * 14 + 2) as u32).to_be_bytes());
        b.extend_from_slice(&(marks.len() as u16 + 1).to_be_bytes());
        for (item, t) in marks {
            b.extend_from_slice(&[0, 1]);
            b.extend_from_slice(&item.to_be_bytes());
            b.extend_from_slice(&t.to_be_bytes());
            b.extend_from_slice(&[0xff, 0xff, 0, 0, 0, 0]);
        }
        // A link point, not a chapter.
        b.extend_from_slice(&[0, 2, 0, 0, 0, 0, 0, 9, 0xff, 0xff, 0, 0, 0, 0]);
        b
    }

    #[test]
    fn reads_items_chapters_and_languages() {
        let b = playlist(
            &[
                ("00800", 900_000, 10_350_000),
                ("00801", 450_000, 1_350_000),
            ],
            &[(0, 900_000), (0, 4_500_000), (1, 450_000)],
        );
        let pl = parse(&b).unwrap();
        assert_eq!(
            pl.items,
            vec![
                Item {
                    clip: "00800".into(),
                    in_time: 900_000,
                    out_time: 10_350_000
                },
                Item {
                    clip: "00801".into(),
                    in_time: 450_000,
                    out_time: 1_350_000
                }
            ]
        );
        assert_eq!(pl.duration(), 230.0);
        assert_eq!(pl.marks, vec![(0, 900_000), (0, 4_500_000), (1, 450_000)]);
        assert_eq!(
            pl.languages,
            vec![
                (0x1100, "eng".into()),
                (0x1101, "fra".into()),
                (0x1200, "deu".into())
            ]
        );
        assert!(parse(b"MPLX0200").is_none());
        assert!(parse(&b[..60]).is_none(), "cut short");
    }
}
