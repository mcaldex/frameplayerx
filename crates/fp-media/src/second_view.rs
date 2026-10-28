//! The second view of 3D Blu-ray video (MVC) in a transport stream, read
//! straight from the file. FFmpeg's H.264 parser doesn't know MVC's units,
//! finds no picture in that stream and holds it back whole, so FFmpeg is
//! told to discard it; [`Input`](crate::input::Input) adds each access unit
//! read here to the base view's packet of the same decoding time.

use fp_core::ByteSource;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Blu-ray's PIDs for MVC's base view and second view.
pub(crate) const BASE_PID: i32 = 0x1011;
pub(crate) const SECOND_PID: i32 = 0x1012;

const CHUNK: u64 = 1 << 20;
/// How far apart (90 kHz ticks) the unit wanted and the one at hand may be
/// before the reader looks for its place again rather than reading on.
const FAR: i64 = 5 * 90_000;

/// One access unit of the second view: its decoding time and NAL units
/// (Annex B).
struct Unit {
    dts: i64,
    data: Vec<u8>,
}

pub(crate) struct SecondView {
    src: Arc<dyn ByteSource>,
    interrupt: Arc<AtomicBool>,
    pid: u16,
    size: u64,
    /// Bytes per transport packet: 192 in Blu-ray's files (a 4-byte
    /// arrival time first), 188 in plain MPEG-TS.
    stride: u64,
    /// Start of the next packet.
    pos: u64,
    buf: Vec<u8>,
    buf_at: u64,
    gathering: Option<Unit>,
    next: Option<Unit>,
    /// Find the place again before reading on: the input seeked.
    lost: bool,
}

impl SecondView {
    /// A reader for `pid` of the transport stream in `src`, when it holds
    /// MVC's second view (its first unit has MVC's parameter set or slices).
    pub(crate) fn open(
        src: Arc<dyn ByteSource>,
        interrupt: Arc<AtomicBool>,
        pid: u16,
    ) -> Option<SecondView> {
        let size = src.size()?;
        let mut head = [0u8; 400];
        let n = src.read_at(0, &mut head).ok()?;
        let stride = match head.get(..n)? {
            h if h.len() > 196 && h[4] == 0x47 && h[196] == 0x47 => 192,
            h if h.len() > 188 && h[0] == 0x47 && h[188] == 0x47 => 188,
            _ => return None,
        };
        let mut view = SecondView {
            src,
            interrupt,
            pid,
            size,
            stride,
            pos: 0,
            buf: Vec::new(),
            buf_at: 0,
            gathering: None,
            next: None,
            lost: false,
        };
        let first = view.peek()?;
        let mvc = crate::nal::units(&first.data, 0)
            .any(|n| n.first().is_some_and(|h| matches!(h & 0x1f, 15 | 20)));
        mvc.then_some(view)
    }

    /// The next read finds its place again first (call after a seek).
    pub(crate) fn seek(&mut self) {
        self.lost = true;
    }

    /// The second view of the access unit decoded at `dts`, whose base
    /// view's packet starts at byte `pos` of the same file; None when it
    /// has none. Units before it are passed over.
    pub(crate) fn take(&mut self, dts: i64, pos: u64) -> Option<Vec<u8>> {
        let mut refound = std::mem::take(&mut self.lost);
        if refound {
            self.find(dts, pos);
        }
        loop {
            let at = self.peek()?.dts;
            if at == dts {
                return self.next.take().map(|u| u.data);
            }
            if !refound && (at < dts - FAR || at > dts + FAR) {
                refound = true;
                self.find(dts, pos);
                continue;
            }
            if at > dts {
                return None;
            }
            self.next = None;
        }
    }

    /// Moves to a unit before `dts`. Blu-ray's SSIF files interleave the
    /// views in extents of a few seconds, the second view's first, so it is
    /// at most a few extents before the base view's packet at `pos`; files
    /// interleaving them unit by unit have it right beside it.
    fn find(&mut self, dts: i64, pos: u64) {
        let mut back = 256 << 10;
        loop {
            let start = pos.saturating_sub(back);
            self.reset(start - start % self.stride);
            match self.peek() {
                Some(u) if u.dts > dts && start > 0 => back *= 4,
                _ => return,
            }
        }
    }

    fn reset(&mut self, pos: u64) {
        self.pos = pos;
        self.gathering = None;
        self.next = None;
    }

    fn peek(&mut self) -> Option<&Unit> {
        if self.next.is_none() {
            self.next = self.read_unit();
        }
        self.next.as_ref()
    }

    /// The next whole unit of the PID, in file order.
    fn read_unit(&mut self) -> Option<Unit> {
        while let Some((pusi, payload)) = self.packet() {
            let payload = &self.buf[payload];
            if pusi {
                if let Some(done) = std::mem::replace(&mut self.gathering, pes_start(payload)) {
                    return Some(done);
                }
            } else if let Some(u) = &mut self.gathering {
                u.data.extend_from_slice(payload);
            }
        }
        self.gathering.take()
    }

    /// Where in the buffer the payload of the next packet of the PID is,
    /// and whether a PES starts in it; None at the end of the file.
    fn packet(&mut self) -> Option<(bool, std::ops::Range<usize>)> {
        loop {
            let at = self.raw_packet()?;
            let pkt = &self.buf[at..at + 188];
            let pid = u16::from(pkt[1] & 0x1f) << 8 | u16::from(pkt[2]);
            let afc = (pkt[3] >> 4) & 3;
            if pkt[0] != 0x47 || pid != self.pid || afc & 1 == 0 {
                continue;
            }
            let start = if afc & 2 != 0 {
                5 + usize::from(pkt[4])
            } else {
                4
            };
            return Some((pkt[1] & 0x40 != 0, (at + start).min(at + 188)..at + 188));
        }
    }

    /// Where in the buffer the 188 bytes of the next transport packet are.
    fn raw_packet(&mut self) -> Option<usize> {
        if self.pos + self.stride > self.size {
            return None;
        }
        let end = self.pos + self.stride;
        if self.pos < self.buf_at || end > self.buf_at + self.buf.len() as u64 {
            if self.interrupt.load(Ordering::Relaxed) {
                return None;
            }
            let len = (CHUNK - CHUNK % self.stride).min(self.size - self.pos);
            self.buf.resize(len as usize, 0);
            self.buf_at = self.pos;
            let mut got = 0;
            while got < self.buf.len() {
                match self
                    .src
                    .read_at(self.pos + got as u64, &mut self.buf[got..])
                {
                    Ok(0) | Err(_) => break,
                    Ok(n) => got += n,
                }
            }
            self.buf.truncate(got);
            if end > self.buf_at + got as u64 {
                return None;
            }
        }
        let at = (self.pos - self.buf_at) as usize;
        self.pos = end;
        Some(at + self.stride as usize - 188)
    }
}

/// The unit a PES packet starts, with the payload after its header.
fn pes_start(p: &[u8]) -> Option<Unit> {
    if p.len() < 14 || p[..3] != [0, 0, 1] {
        return None;
    }
    let flags = p[7];
    let header = 9 + usize::from(p[8]);
    let ts = |b: &[u8]| {
        i64::from((b[0] >> 1) & 7) << 30
            | i64::from(b[1]) << 22
            | i64::from(b[2] >> 1) << 15
            | i64::from(b[3]) << 7
            | i64::from(b[4] >> 1)
    };
    let pts = (flags & 0x80 != 0).then(|| ts(&p[9..14]))?;
    let dts = if flags & 0x40 != 0 && p.len() >= 19 {
        ts(&p[14..19])
    } else {
        pts
    };
    Some(Unit {
        dts,
        data: p.get(header..).unwrap_or(&[]).to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use fp_core::source::MemorySource;

    /// Blu-ray transport packets (192 bytes): `pid`'s PES packet of
    /// `payload` at `dts`, cut across packets.
    fn pes_packets(pid: u16, dts: i64, payload: &[u8], out: &mut Vec<u8>) {
        let ts = |prefix: u8, t: i64| {
            let t = t as u64;
            [
                prefix << 4 | (((t >> 30) & 7) as u8) << 1 | 1,
                (t >> 22) as u8,
                ((t >> 15) as u8) << 1 | 1,
                (t >> 7) as u8,
                (t as u8) << 1 | 1,
            ]
        };
        let mut pes = vec![0, 0, 1, 0xe1, 0, 0, 0x80, 0xc0, 10];
        pes.extend_from_slice(&ts(3, dts));
        pes.extend_from_slice(&ts(1, dts));
        pes.extend_from_slice(payload);
        for (n, chunk) in pes.chunks(184).enumerate() {
            let mut p = vec![0u8; 4];
            p.extend_from_slice(&[0x47, ((n == 0) as u8 * 0x40) | (pid >> 8) as u8, pid as u8]);
            if chunk.len() < 184 {
                // Stuffing in an adaptation field fills the packet.
                let fill = 184 - chunk.len();
                p.push(0x30);
                p.push((fill - 1) as u8);
                if fill > 1 {
                    p.push(0);
                    p.extend(std::iter::repeat_n(0xff, fill - 2));
                }
            } else {
                p.push(0x10);
            }
            p.extend_from_slice(chunk);
            assert_eq!(p.len(), 192);
            out.extend_from_slice(&p);
        }
    }

    /// The second view's units at `dts` 0, 3003, 6006... (NAL types 24, 15
    /// and 20, then 24 and 20), and a base view packet after each, the
    /// second view's units interleaved `run` at a time as in SSIF files.
    /// Returns the stream and each base packet's position.
    fn stream(units: i64, run: i64) -> (Arc<MemorySource>, Vec<u64>) {
        let mut out = Vec::new();
        let mut base_at = Vec::new();
        let unit = |i: i64| {
            let mut u = vec![0, 0, 0, 1, 24, 0, 0, 0, 1];
            if i % 10 == 0 {
                u.extend_from_slice(&[15, 0x80, 0, 0, 0, 1]);
            }
            u.extend_from_slice(&[20, 0x80, 0x40, (i & 0xff) as u8]);
            u.extend(std::iter::repeat_n(0x55, 300 + i as usize % 7));
            u
        };
        for start in (0..units).step_by(run as usize) {
            for i in start..(start + run).min(units) {
                pes_packets(SECOND_PID as u16, i * 3003, &unit(i), &mut out);
            }
            for i in start..(start + run).min(units) {
                base_at.push(out.len() as u64);
                pes_packets(BASE_PID as u16, i * 3003, &[0, 0, 0, 1, 1, 0x88], &mut out);
            }
        }
        (Arc::new(MemorySource(out)), base_at)
    }

    fn open(src: Arc<MemorySource>) -> SecondView {
        SecondView::open(src, Arc::new(AtomicBool::new(false)), SECOND_PID as u16)
            .expect("a second view")
    }

    /// The unit's number, from its slice.
    fn number(unit: &[u8]) -> u8 {
        let at = unit
            .windows(2)
            .position(|w| w == [0x80, 0x40])
            .expect("a slice");
        unit[at + 2]
    }

    #[test]
    fn pairs_units_by_decoding_time() {
        let (src, base_at) = stream(40, 8);
        let mut view = open(src);
        for i in 0..40 {
            let u = view.take(i * 3003, base_at[i as usize]).expect("unit");
            assert_eq!(number(&u), i as u8);
            assert_eq!(u[..5], [0, 0, 0, 1, 24]);
        }
        assert!(view.take(40 * 3003, 0).is_none(), "past the end");
    }

    #[test]
    fn a_base_picture_without_a_second_view_gets_none_and_reading_goes_on() {
        let (src, base_at) = stream(20, 4);
        let mut view = open(src);
        assert!(view.take(1500, base_at[0]).is_none());
        assert_eq!(number(&view.take(3003, base_at[1]).expect("unit")), 1);
        // Units passed over aren't given again.
        assert!(view.take(0, base_at[0]).is_none());
        assert_eq!(number(&view.take(2 * 3003, base_at[2]).expect("unit")), 2);
    }

    #[test]
    fn finds_its_place_after_seeks_either_way() {
        // Long runs: each unit's second view is well before its base view.
        let (src, base_at) = stream(400, 100);
        let mut view = open(src);
        assert_eq!(number(&view.take(0, base_at[0]).expect("unit")), 0);
        for i in [350i64, 120, 5, 399, 200] {
            view.seek();
            let u = view.take(i * 3003, base_at[i as usize]).expect("unit");
            assert_eq!(number(&u), (i & 0xff) as u8, "unit {i}");
            let u = view.take((i + 1) * 3003, 0);
            assert_eq!(
                u.map(|u| number(&u)),
                (i < 399).then(|| ((i + 1) & 0xff) as u8)
            );
        }
    }

    #[test]
    fn only_mvc_counts() {
        let mut out = Vec::new();
        pes_packets(SECOND_PID as u16, 0, &[0, 0, 0, 1, 0x65, 0x88], &mut out);
        let src = Arc::new(MemorySource(out));
        assert!(SecondView::open(src, Arc::new(AtomicBool::new(false)), 0x1012).is_none());
        let src = Arc::new(MemorySource(b"not a transport stream at all".repeat(20)));
        assert!(SecondView::open(src, Arc::new(AtomicBool::new(false)), 0x1012).is_none());
    }
}
