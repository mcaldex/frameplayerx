//! Files of a UDF disc image, Blu-ray's file system: UDF 2.50, which keeps
//! file entries and directories in a metadata partition, or the older
//! versions without one. Read-only, straight from the image.

use crate::{Error, Result};
use fp_core::ByteSource;
use std::sync::Arc;

const SECTOR: u64 = 2048;

fn bad(what: impl Into<String>) -> Error {
    Error::Disc(what.into())
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    b.get(at..at + 2)
        .map_or(0, |s| u16::from_le_bytes([s[0], s[1]]))
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    b.get(at..at + 4)
        .map_or(0, |s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from(u32_at(b, at)) | u64::from(u32_at(b, at + 4)) << 32
}

/// The descriptor tag's identifier, when its checksum holds.
fn tag(b: &[u8]) -> Option<u16> {
    let sum = b
        .get(..16)?
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != 4)
        .fold(0u8, |s, (_, v)| s.wrapping_add(*v));
    (sum == b[4]).then(|| u16_at(b, 0))
}

/// A run of bytes of the image; unrecorded runs read as zeros.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Extent {
    at: u64,
    len: u64,
    recorded: bool,
}

#[derive(Clone, Copy, Debug)]
enum Map {
    /// Blocks counted from the partition's first sector.
    Physical { start: u64 },
    /// Blocks of the metadata file.
    Metadata,
}

/// Where a file entry is: a partition reference and a block in it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Icb {
    part: u16,
    block: u32,
}

/// What the blocks of short allocation descriptors count in.
#[derive(Clone, Copy, Debug)]
enum Base {
    /// A partition reference: the partition of the file entry.
    Ref(u16),
    /// The metadata file's own partition, from this sector.
    Start(u64),
}

/// A file or directory.
#[derive(Clone, Debug)]
pub(crate) struct Node {
    pub dir: bool,
    pub size: u64,
    extents: Vec<Extent>,
    /// The data itself, when the entry embeds it.
    inline: Option<Vec<u8>>,
}

pub(crate) struct Udf {
    src: Arc<dyn ByteSource>,
    maps: Vec<Map>,
    /// The metadata partition's blocks, in order.
    metadata: Vec<Extent>,
    root: Node,
}

impl Udf {
    pub(crate) fn open(src: Arc<dyn ByteSource>) -> Result<Udf> {
        let size = src.size().ok_or_else(|| bad("its size is unknown"))?;
        let mut udf = Udf {
            src,
            maps: Vec::new(),
            metadata: Vec::new(),
            root: Node {
                dir: true,
                size: 0,
                extents: Vec::new(),
                inline: None,
            },
        };
        let last = size / SECTOR;
        let anchor = [256, last.saturating_sub(1), last.saturating_sub(257)]
            .into_iter()
            .find_map(|s| {
                let b = udf.read(s * SECTOR, 32).ok()?;
                (tag(&b) == Some(2)).then_some(b)
            })
            .ok_or_else(|| bad("no UDF file system (not a Blu-ray image?)"))?;
        let (vds_len, vds_at) = (u32_at(&anchor, 16), u32_at(&anchor, 20));
        let mut partitions = Vec::new();
        let mut volume = None;
        for s in 0..u64::from(vds_len).div_ceil(SECTOR).min(64) {
            let b = udf.read((u64::from(vds_at) + s) * SECTOR, SECTOR as usize)?;
            match tag(&b) {
                Some(5) => partitions.push((u16_at(&b, 22), u64::from(u32_at(&b, 188)))),
                Some(6) => volume = Some(b),
                Some(8) => break,
                _ => {}
            }
        }
        let lvd = volume.ok_or_else(|| bad("no logical volume"))?;
        if u32_at(&lvd, 212) != SECTOR as u32 {
            return Err(bad(format!("blocks of {} bytes", u32_at(&lvd, 212))));
        }
        let start_of = |number: u16| {
            partitions
                .iter()
                .find(|p| p.0 == number)
                .map(|p| p.1)
                .ok_or_else(|| bad(format!("no partition {number}")))
        };
        let mut metadata_file = None;
        let mut at = 440;
        for _ in 0..u32_at(&lvd, 268) {
            let (kind, len) = (
                lvd.get(at).copied(),
                usize::from(lvd.get(at + 1).copied().unwrap_or(0)),
            );
            match kind {
                Some(1) => udf.maps.push(Map::Physical {
                    start: start_of(u16_at(&lvd, at + 4))?,
                }),
                Some(2) => {
                    let id = lvd.get(at + 5..at + 28).unwrap_or(&[]);
                    let start = start_of(u16_at(&lvd, at + 38))?;
                    if id.starts_with(b"*UDF Metadata Partition") {
                        let file = u32_at(&lvd, at + 40);
                        let mirror = u32_at(&lvd, at + 44);
                        metadata_file = Some((start, file, mirror));
                        udf.maps.push(Map::Metadata);
                    } else if id.starts_with(b"*UDF Sparable Partition") {
                        udf.maps.push(Map::Physical { start });
                    } else {
                        let id = String::from_utf8_lossy(id)
                            .trim_end_matches('\0')
                            .to_string();
                        return Err(bad(format!("partition kind {id}")));
                    }
                }
                _ => return Err(bad("unknown partition map")),
            }
            at += len.max(6);
        }
        if let Some((start, file, mirror)) = metadata_file {
            // The mirror stands in for a damaged main file.
            let entry =
                |block: u32| udf.entry_at((start + u64::from(block)) * SECTOR, Base::Start(start));
            udf.metadata = entry(file).or_else(|_| entry(mirror))?.extents;
        }
        let fsd_at = Icb {
            part: u16_at(&lvd, 256),
            block: u32_at(&lvd, 252),
        };
        let fsd = udf.read(udf.offset(fsd_at)?, 512)?;
        if tag(&fsd) != Some(256) {
            return Err(bad("no file set"));
        }
        udf.root = udf.entry(Icb {
            block: u32_at(&fsd, 404),
            part: u16_at(&fsd, 408),
        })?;
        Ok(udf)
    }

    /// The file or directory at `path` (`BDMV/PLAYLIST/00000.mpls`), names
    /// compared without case.
    pub(crate) fn lookup(&self, path: &str) -> Result<Node> {
        let mut node = self.root.clone();
        for part in path.split('/').filter(|p| !p.is_empty()) {
            let icb = self
                .entries(&node)?
                .into_iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(part))
                .map(|(_, icb)| icb)
                .ok_or_else(|| bad(format!("no {path}")))?;
            node = self.entry(icb)?;
        }
        Ok(node)
    }

    /// The entries of the directory at `path`, each opened with
    /// [`Udf::node`]: a directory read once however many it holds.
    pub(crate) fn list(&self, path: &str) -> Result<Vec<(String, Icb)>> {
        self.entries(&self.lookup(path)?)
    }

    pub(crate) fn node(&self, icb: Icb) -> Result<Node> {
        self.entry(icb)
    }

    /// A whole small file.
    pub(crate) fn read_file(&self, node: &Node) -> Result<Vec<u8>> {
        if node.size > 16 << 20 {
            return Err(bad("file too large to read whole"));
        }
        let file = self.file(node, String::new());
        let mut out = vec![0; node.size as usize];
        let mut got = 0;
        while got < out.len() {
            match file.read_at(got as u64, &mut out[got..])? {
                0 => break,
                n => got += n,
            }
        }
        out.truncate(got);
        Ok(out)
    }

    /// The file as a byte source of its own.
    pub(crate) fn file(&self, node: &Node, name: String) -> Arc<dyn ByteSource> {
        let mut runs = Vec::new();
        let mut pos = 0;
        for e in &node.extents {
            let len = e.len.min(node.size.saturating_sub(pos));
            if len == 0 {
                break;
            }
            runs.push((pos, Extent { len, ..*e }));
            pos += len;
        }
        Arc::new(UdfFile {
            src: self.src.clone(),
            runs,
            inline: node.inline.clone(),
            size: node.size,
            name,
        })
    }

    fn read(&self, at: u64, len: usize) -> Result<Vec<u8>> {
        let mut out = vec![0; len];
        let mut got = 0;
        while got < len {
            match self.src.read_at(at + got as u64, &mut out[got..])? {
                0 => return Err(bad("the image ends early")),
                n => got += n,
            }
        }
        Ok(out)
    }

    /// Where block `icb.block` of partition `icb.part` starts in the image.
    fn offset(&self, icb: Icb) -> Result<u64> {
        let runs = self.ranges(icb.part, icb.block, 1)?;
        runs.first()
            .map(|e| e.at)
            .ok_or_else(|| bad("block outside its partition"))
    }

    /// The image's bytes of `len` bytes from a block of a partition: one run
    /// in a physical partition, several where the metadata file is split.
    fn ranges(&self, part: u16, block: u32, len: u64) -> Result<Vec<Extent>> {
        match self.maps.get(usize::from(part)) {
            Some(Map::Physical { start }) => Ok(vec![Extent {
                at: (start + u64::from(block)) * SECTOR,
                len,
                recorded: true,
            }]),
            Some(Map::Metadata) => {
                let (mut skip, mut left) = (u64::from(block) * SECTOR, len);
                let mut out = Vec::new();
                for e in &self.metadata {
                    if left == 0 {
                        break;
                    }
                    if skip >= e.len {
                        skip -= e.len;
                        continue;
                    }
                    let take = (e.len - skip).min(left);
                    out.push(Extent {
                        at: e.at + skip,
                        len: take,
                        recorded: e.recorded,
                    });
                    (skip, left) = (0, left - take);
                }
                if left > 0 {
                    return Err(bad("block outside the metadata partition"));
                }
                Ok(out)
            }
            None => Err(bad(format!("no partition reference {part}"))),
        }
    }

    fn entry(&self, icb: Icb) -> Result<Node> {
        self.entry_at(self.offset(icb)?, Base::Ref(icb.part))
    }

    /// The (extended) file entry at byte `at`.
    fn entry_at(&self, at: u64, base: Base) -> Result<Node> {
        let b = self.read(at, SECTOR as usize)?;
        let (info_len, ea_at) = match tag(&b) {
            Some(261) => (u64_at(&b, 56), 168),
            Some(266) => (u64_at(&b, 56), 208),
            t => return Err(bad(format!("expected a file entry, found tag {t:?}"))),
        };
        let file_type = b[27];
        let ad_kind = u16_at(&b, 34) & 7;
        let (l_ea, l_ad) = (u32_at(&b, ea_at) as usize, u32_at(&b, ea_at + 4) as usize);
        let ads = b
            .get(ea_at + 8 + l_ea..ea_at + 8 + l_ea + l_ad)
            .ok_or_else(|| bad("file entry overruns its block"))?;
        let mut node = Node {
            dir: file_type == 4,
            size: info_len,
            extents: Vec::new(),
            inline: None,
        };
        if ad_kind == 3 {
            node.inline = Some(ads.to_vec());
            return Ok(node);
        }
        self.allocation(ads, ad_kind, base, &mut node.extents, 0)?;
        Ok(node)
    }

    /// Adds the extents of allocation descriptors `ads` of kind `kind`,
    /// following continuations.
    fn allocation(
        &self,
        ads: &[u8],
        kind: u16,
        base: Base,
        out: &mut Vec<Extent>,
        depth: u32,
    ) -> Result<()> {
        if depth > 64 {
            return Err(bad("allocation descriptors loop"));
        }
        let step = match kind {
            0 => 8,
            1 => 16,
            2 => 20,
            _ => return Err(bad(format!("allocation descriptor kind {kind}"))),
        };
        for ad in ads.chunks_exact(step) {
            let raw = u32_at(ad, 0);
            let (len, how) = (u64::from(raw & 0x3fff_ffff), raw >> 30);
            if len == 0 {
                break;
            }
            let (block, ad_base) = match kind {
                0 => (u32_at(ad, 4), base),
                1 => (u32_at(ad, 4), Base::Ref(u16_at(ad, 8))),
                _ => (u32_at(ad, 12), Base::Ref(u16_at(ad, 16))),
            };
            let runs = match ad_base {
                Base::Ref(p) => self.ranges(p, block, len)?,
                Base::Start(start) => vec![Extent {
                    at: (start + u64::from(block)) * SECTOR,
                    len,
                    recorded: true,
                }],
            };
            if how == 3 {
                // The next allocation extent descriptor.
                let at = runs.first().map(|e| e.at).unwrap_or(0);
                let b = self.read(at, SECTOR as usize)?;
                if tag(&b) != Some(258) {
                    return Err(bad("broken allocation extent"));
                }
                let l_ad = u32_at(&b, 20) as usize;
                let more = b
                    .get(24..24 + l_ad)
                    .ok_or_else(|| bad("broken allocation extent"))?;
                return self.allocation(more, kind, base, out, depth + 1);
            }
            out.extend(runs.into_iter().map(|e| Extent {
                recorded: how == 0,
                ..e
            }));
        }
        Ok(())
    }

    /// A directory's entries: names and where their file entries are.
    fn entries(&self, dir: &Node) -> Result<Vec<(String, Icb)>> {
        if !dir.dir {
            return Err(bad("not a directory"));
        }
        let data = self.read_file(dir)?;
        let mut out = Vec::new();
        let mut at = 0;
        while at + 38 <= data.len() {
            let fid = &data[at..];
            if tag(fid) != Some(257) {
                break;
            }
            let flags = fid[18];
            let l_fi = usize::from(fid[19]);
            let l_iu = usize::from(u16_at(fid, 36));
            let name = fid.get(38 + l_iu..38 + l_iu + l_fi).unwrap_or(&[]);
            // Not deleted, not the parent.
            if flags & 0x0c == 0 && !name.is_empty() {
                out.push((
                    osta_name(name),
                    Icb {
                        block: u32_at(fid, 24),
                        part: u16_at(fid, 28),
                    },
                ));
            }
            at += (38 + l_iu + l_fi + 3) & !3;
        }
        Ok(out)
    }
}

/// An OSTA compressed Unicode name: 8 bits a character, or 16 (big-endian).
fn osta_name(b: &[u8]) -> String {
    match b.split_first() {
        Some((16, rest)) => String::from_utf16_lossy(
            &rest
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_be_bytes(*c))
                .collect::<Vec<_>>(),
        ),
        Some((_, rest)) => rest.iter().map(|&c| char::from(c)).collect(),
        None => String::new(),
    }
}

/// A file of the image, read through its extents.
struct UdfFile {
    src: Arc<dyn ByteSource>,
    /// Each extent with the file offset it starts at.
    runs: Vec<(u64, Extent)>,
    inline: Option<Vec<u8>>,
    size: u64,
    name: String,
}

impl ByteSource for UdfFile {
    fn size(&self) -> Option<u64> {
        Some(self.size)
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<usize> {
        if offset >= self.size || buf.is_empty() {
            return Ok(0);
        }
        if let Some(data) = &self.inline {
            let start = (offset as usize).min(data.len());
            let n = buf.len().min(data.len() - start);
            buf[..n].copy_from_slice(&data[start..start + n]);
            return Ok(n);
        }
        let i = self.runs.partition_point(|(at, _)| *at <= offset);
        let Some((at, e)) = i.checked_sub(1).map(|i| self.runs[i]) else {
            return Ok(0);
        };
        let within = offset - at;
        let n = (e.len - within).min(buf.len() as u64) as usize;
        if n == 0 {
            return Ok(0);
        }
        if !e.recorded {
            buf[..n].fill(0);
            return Ok(n);
        }
        self.src.read_at(e.at + within, &mut buf[..n])
    }

    fn describe(&self) -> String {
        format!("{} in {}", self.name, self.src.describe())
    }
}
