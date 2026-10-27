//! Software decoders write their pictures into the frame memory offered
//! ([`fp_media::set_frame_memory`]), and decode them the same as into
//! FFmpeg's own buffers.

use fp_core::source::FileSource;
use fp_media::bench::{BenchOptions, decode_bench};
use fp_media::thumb::decode_first_frame;
use fp_media::{BLOCK_ALIGN, Block, FrameMemory, HwDecode, set_frame_memory};
use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Ordinary memory that remembers what it gave out.
#[derive(Default)]
struct Watched {
    /// Blocks alive: (start, length).
    live: Arc<Mutex<Vec<(usize, usize)>>>,
    made: AtomicUsize,
}

struct Owned {
    ptr: NonNull<u8>,
    layout: Layout,
    live: Arc<Mutex<Vec<(usize, usize)>>>,
}

// SAFETY: the allocation is only freed, once, on drop.
unsafe impl Send for Owned {}
// SAFETY: as for Send.
unsafe impl Sync for Owned {}

impl Drop for Owned {
    fn drop(&mut self) {
        let start = self.ptr.as_ptr() as usize;
        self.live.lock().unwrap().retain(|&(s, _)| s != start);
        // SAFETY: allocated with this layout in `alloc`.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

impl FrameMemory for Watched {
    fn alloc(&self, len: usize, reads: bool) -> Option<Block> {
        assert!(reads, "software decoders read their pictures back");
        let layout = Layout::from_size_align(len, BLOCK_ALIGN).ok()?;
        // SAFETY: a non-zero size.
        let ptr = NonNull::new(unsafe { std::alloc::alloc(layout) })?;
        self.made.fetch_add(1, Ordering::Relaxed);
        self.live.lock().unwrap().push((ptr.as_ptr() as usize, len));
        let owner = Owned {
            ptr,
            layout,
            live: self.live.clone(),
        };
        // SAFETY: `len` bytes, aligned, freed by their owner.
        Some(unsafe { Block::new(ptr, len, Box::new(owner)) })
    }
}

fn fixture(name: &str) -> Arc<FileSource> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    Arc::new(FileSource::open(&path).unwrap())
}

fn checksums(name: &str) -> Vec<(usize, f64, u64)> {
    let opts = BenchOptions {
        frames: 30,
        hw: HwDecode::Off,
        seek: Some(0.5),
        frames_after_seek: 10,
        checksum: vec![0, 1, 14, 29],
    };
    decode_bench(fixture(name), name, &opts).unwrap().checksums
}

/// One test, as the frame memory is the whole process's.
#[test]
fn software_pictures_decode_alike_in_frame_memory() {
    let mut expected = Vec::new();
    let names = [
        "h264_aac_180_LR.mp4",
        "vp9.webm",
        "hevc10_tb.mkv",
        "av1_opus.webm",
    ];
    for name in names {
        expected.push(checksums(name));
    }

    let memory = Arc::new(Watched::default());
    set_frame_memory(Some(memory.clone()));
    for (name, want) in names.iter().zip(&expected) {
        let made = memory.made.load(Ordering::Relaxed);
        assert_eq!(&checksums(name), want, "{name}");
        let used = memory.made.load(Ordering::Relaxed) > made;
        // FFmpeg's dav1d wrapper allocates AV1 pictures itself.
        assert_eq!(used, !name.starts_with("av1"), "{name}");
    }
    assert!(memory.live.lock().unwrap().is_empty(), "every block freed");

    let frame = decode_first_frame(fixture("h264_aac_180_LR.mp4"), "x.mp4").unwrap();
    let live = memory.live.lock().unwrap().clone();
    for p in frame.planes() {
        let (start, end) = (
            p.data.as_ptr() as usize,
            p.data.as_ptr() as usize + p.data.len(),
        );
        assert!(
            live.iter().any(|&(s, len)| s <= start && end <= s + len),
            "a plane outside the blocks"
        );
        assert_eq!(start % BLOCK_ALIGN, 0);
    }
    drop(frame);
    set_frame_memory(None);
    assert!(
        memory.live.lock().unwrap().is_empty(),
        "the picture's block freed with it"
    );
}
