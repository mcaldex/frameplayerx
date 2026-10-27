//! Memory for decoded pictures. The renderer can offer memory the GPU reads
//! ([`set_frame_memory`]), and decoders then write pictures where the GPU
//! copies them from, instead of the render thread copying each one there
//! first (an 8K picture is 50 MB).
//!
//! The V4L2 decoder's 8-bit pictures and those of FFmpeg's own software
//! decoders (`get_buffer2`) go there. dav1d's don't: FFmpeg's wrapper
//! allocates AV1 pictures itself.

use crate::ff;
use fp_core::sync::{lock, read, write};
use std::ffi::{c_int, c_void};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, RwLock};

/// Memory decoded pictures can be written into.
pub trait FrameMemory: Send + Sync {
    /// A block of `len` bytes, or `None`: the decoder uses ordinary memory.
    /// `reads`: the decoder reads the block back too (software decoders
    /// predict from earlier pictures), which uncached memory would make
    /// many times slower.
    fn alloc(&self, len: usize, reads: bool) -> Option<Block>;
}

/// `len` bytes at `ptr`, until `owner` is dropped.
pub struct Block {
    ptr: NonNull<u8>,
    len: usize,
    _owner: Box<dyn Send + Sync>,
}

// SAFETY: a block is only reached through the FFmpeg buffer lent over it,
// whose reference counting gives it one writer at a time.
unsafe impl Send for Block {}
// SAFETY: as for Send.
unsafe impl Sync for Block {}

/// Pictures' planes start on this, as FFmpeg's SIMD code wants.
pub const BLOCK_ALIGN: usize = 64;

impl Block {
    /// # Safety
    /// `ptr` must point at `len` writable bytes, aligned to [`BLOCK_ALIGN`],
    /// that stay valid until `owner` is dropped and that nothing else uses
    /// meanwhile.
    pub unsafe fn new(ptr: NonNull<u8>, len: usize, owner: Box<dyn Send + Sync>) -> Block {
        Block {
            ptr,
            len,
            _owner: owner,
        }
    }

    /// A block of ordinary memory.
    fn heap(len: usize) -> Option<Block> {
        let layout = std::alloc::Layout::from_size_align(len.max(1), BLOCK_ALIGN).ok()?;
        // SAFETY: the layout has a non-zero size.
        let ptr = NonNull::new(unsafe { std::alloc::alloc(layout) })?;
        // SAFETY: freshly allocated for `len` bytes, freed by its owner.
        Some(unsafe { Block::new(ptr, len, Box::new(Heap { ptr, layout })) })
    }
}

struct Heap {
    ptr: NonNull<u8>,
    layout: std::alloc::Layout,
}

// SAFETY: the allocation is only freed, once, by its owner.
unsafe impl Send for Heap {}
// SAFETY: as for Send.
unsafe impl Sync for Heap {}

impl Drop for Heap {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `Block::heap`.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

static MEMORY: RwLock<Option<Arc<dyn FrameMemory>>> = RwLock::new(None);

/// The memory decoders opened from now on write pictures into; `None`, ordinary memory.
pub fn set_frame_memory(memory: Option<Arc<dyn FrameMemory>>) {
    *write(&MEMORY) = memory;
}

pub(crate) fn frame_memory() -> Option<Arc<dyn FrameMemory>> {
    read(&MEMORY).clone()
}

/// Blocks of one size, reused, lent out as FFmpeg buffers: what an
/// `AVBufferPool` does, in a [`FrameMemory`]'s blocks when it has them.
#[derive(Clone)]
pub(crate) struct BlockPool(Arc<Shared>);

struct Shared {
    memory: Option<Arc<dyn FrameMemory>>,
    len: usize,
    reads: bool,
    free: Mutex<Vec<Block>>,
}

/// A block out on loan, with the pool it goes back to.
struct Lent {
    block: Block,
    pool: Arc<Shared>,
}

impl BlockPool {
    pub fn new(memory: Option<Arc<dyn FrameMemory>>, len: usize, reads: bool) -> BlockPool {
        BlockPool(Arc::new(Shared {
            memory,
            len,
            reads,
            free: Mutex::new(Vec::new()),
        }))
    }

    pub fn len(&self) -> usize {
        self.0.len
    }

    /// A buffer of [`Self::len`] bytes, zeroed when new; null when out of
    /// memory.
    pub fn get(&self) -> *mut ff::AVBufferRef {
        let s = &self.0;
        let block = lock(&s.free).pop().or_else(|| {
            let b = s
                .memory
                .as_ref()
                .and_then(|m| m.alloc(s.len, s.reads))
                .filter(|b| b.len >= s.len && (b.ptr.as_ptr() as usize).is_multiple_of(BLOCK_ALIGN))
                .or_else(|| Block::heap(s.len))?;
            // SAFETY: the block holds `len` bytes no one else uses yet.
            unsafe { std::ptr::write_bytes(b.ptr.as_ptr(), 0, s.len) };
            Some(b)
        });
        let Some(block) = block else {
            return std::ptr::null_mut();
        };
        let data = block.ptr.as_ptr();
        let lent = Box::into_raw(Box::new(Lent {
            block,
            pool: s.clone(),
        }));
        // SAFETY: the block holds `len` bytes until `give_back` takes the
        // loan back, which FFmpeg calls once, when the last reference goes.
        let buf = unsafe { ff::av_buffer_create(data, s.len, Some(give_back), lent.cast(), 0) };
        if buf.is_null() {
            // SAFETY: FFmpeg didn't take the loan.
            drop(unsafe { Box::from_raw(lent) });
        }
        buf
    }
}

unsafe extern "C" fn give_back(opaque: *mut c_void, _data: *mut u8) {
    // SAFETY: `opaque` is the loan `BlockPool::get` made for this buffer,
    // given back once.
    let lent = unsafe { Box::from_raw(opaque.cast::<Lent>()) };
    let Lent { block, pool } = *lent;
    lock(&pool.free).push(block);
}

/// Pictures' layouts the renderer shows as they are; others are converted
/// anyway, so they keep FFmpeg's own buffers.
const SHOWN: [ff::AVPixelFormat; 11] = [
    ff::AV_PIX_FMT_YUV420P,
    ff::AV_PIX_FMT_YUVJ420P,
    ff::AV_PIX_FMT_YUV420P10LE,
    ff::AV_PIX_FMT_YUV422P,
    ff::AV_PIX_FMT_YUVJ422P,
    ff::AV_PIX_FMT_YUV422P10LE,
    ff::AV_PIX_FMT_YUV444P,
    ff::AV_PIX_FMT_YUVJ444P,
    ff::AV_PIX_FMT_YUV444P10LE,
    ff::AV_PIX_FMT_NV12,
    ff::AV_PIX_FMT_P010LE,
];

/// FFmpeg's padding past each plane (`16 + STRIDE_ALIGN - 1`), for the
/// widest SIMD it builds.
const PLANE_PADDING: usize = 16 + BLOCK_ALIGN - 1;

/// A software decoder's pictures, all planes in one block of a frame
/// memory: its `AVCodecContext::get_buffer2`, with this as `opaque`.
pub(crate) struct Pictures {
    memory: Arc<dyn FrameMemory>,
    /// The last picture shape's layout and pool.
    shape: Mutex<Option<(Shape, Layout, BlockPool)>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Shape {
    format: ff::AVPixelFormat,
    width: c_int,
    height: c_int,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Layout {
    linesize: [c_int; 4],
    /// Each plane's start in the block; `None` past the last plane.
    offset: [Option<usize>; 4],
    len: usize,
}

impl Pictures {
    pub fn new(memory: Arc<dyn FrameMemory>) -> Pictures {
        Pictures {
            memory,
            shape: Mutex::new(None),
        }
    }

    /// Lends `frame` a block laid out for it, as `avcodec_default_get_buffer2`
    /// would lay out its buffers; `false` leaves the frame to FFmpeg.
    ///
    /// # Safety
    /// `ctx` and `frame` are the arguments FFmpeg calls `get_buffer2` with.
    unsafe fn lend(&self, ctx: *mut ff::AVCodecContext, frame: *mut ff::AVFrame) -> bool {
        // SAFETY: as the caller promises.
        let f = unsafe { &mut *frame };
        let shape = Shape {
            format: f.format,
            width: f.width,
            height: f.height,
        };
        if !SHOWN.contains(&shape.format) || shape.width <= 0 || shape.height <= 0 {
            return false;
        }
        let (layout, pool) = {
            let mut current = lock(&self.shape);
            if current.as_ref().is_none_or(|(s, ..)| *s != shape) {
                // SAFETY: as the caller promises.
                let Some(layout) = (unsafe { layout(ctx, shape) }) else {
                    return false;
                };
                let pool = BlockPool::new(Some(self.memory.clone()), layout.len, true);
                *current = Some((shape, layout, pool));
            }
            match current.as_ref() {
                Some((_, layout, pool)) => (*layout, pool.clone()),
                None => return false,
            }
        };
        let buf = pool.get();
        if buf.is_null() {
            return false;
        }
        // SAFETY: the buffer holds `layout.len` bytes, each plane at its
        // offset inside it.
        unsafe {
            let base = (*buf).data;
            f.buf[0] = buf;
            for i in 0..ff::AV_NUM_DATA_POINTERS as usize {
                (f.data[i], f.linesize[i]) = match layout.offset.get(i).copied().flatten() {
                    Some(o) => (base.add(o), layout.linesize[i]),
                    None => (std::ptr::null_mut(), 0),
                };
            }
            f.extended_data = f.data.as_mut_ptr();
        }
        true
    }
}

/// `get_buffer2` for decoders whose `opaque` is a [`Pictures`].
///
/// # Safety
/// FFmpeg calls it, with such a context.
pub(crate) unsafe extern "C" fn get_buffer2(
    ctx: *mut ff::AVCodecContext,
    frame: *mut ff::AVFrame,
    flags: c_int,
) -> c_int {
    // SAFETY: FFmpeg's arguments; `opaque` is the decoder's Pictures, which
    // outlives its context.
    unsafe {
        let pictures = (*ctx).opaque.cast::<Pictures>().cast_const();
        let hardware = !(*ctx).hw_frames_ctx.is_null();
        if hardware || pictures.is_null() || !(*pictures).lend(ctx, frame) {
            return ff::avcodec_default_get_buffer2(ctx, frame, flags);
        }
    }
    0
}

/// Where `shape`'s planes go in one block: FFmpeg's strides
/// (`update_frame_pool`), each plane padded and starting on [`BLOCK_ALIGN`].
///
/// # Safety
/// `ctx` is a decoder's context.
unsafe fn layout(ctx: *mut ff::AVCodecContext, shape: Shape) -> Option<Layout> {
    let (mut w, mut h) = (shape.width, shape.height);
    let mut align = [0 as c_int; ff::AV_NUM_DATA_POINTERS as usize];
    let mut linesize = [0 as c_int; 4];
    // SAFETY: valid context; the arrays are as long as FFmpeg writes.
    unsafe {
        ff::avcodec_align_dimensions2(ctx, &mut w, &mut h, align.as_mut_ptr());
        loop {
            if ff::av_image_fill_linesizes(linesize.as_mut_ptr(), shape.format, w) < 0 {
                return None;
            }
            if (0..4).all(|i| align[i] <= 0 || linesize[i] % align[i] == 0) {
                break;
            }
            // As FFmpeg does: widen by w's lowest set bit and try again.
            w += w & w.wrapping_neg();
        }
    }
    let strides = linesize.map(|l| l as isize);
    let mut size = [0usize; 4];
    // SAFETY: as above.
    if unsafe {
        ff::av_image_fill_plane_sizes(size.as_mut_ptr(), shape.format, h, strides.as_ptr())
    } < 0
    {
        return None;
    }
    let mut offset = [None; 4];
    let mut len = 0;
    for (o, &s) in offset.iter_mut().zip(&size).take_while(|(_, s)| **s > 0) {
        *o = Some(len);
        len = (len + s + PLANE_PADDING).next_multiple_of(BLOCK_ALIGN);
    }
    Some(Layout {
        linesize,
        offset,
        len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Ordinary memory, counted.
    #[derive(Default)]
    struct Counted {
        live: Arc<AtomicUsize>,
        made: AtomicUsize,
    }

    struct Count {
        live: Arc<AtomicUsize>,
        _block: Block,
    }

    impl FrameMemory for Counted {
        fn alloc(&self, len: usize, _reads: bool) -> Option<Block> {
            let inner = Block::heap(len)?;
            self.made.fetch_add(1, Ordering::Relaxed);
            self.live.fetch_add(1, Ordering::Relaxed);
            let ptr = inner.ptr;
            let count = Count {
                live: self.live.clone(),
                _block: inner,
            };
            // SAFETY: the inner block's bytes, kept with it.
            Some(unsafe { Block::new(ptr, len, Box::new(count)) })
        }
    }

    impl Drop for Count {
        fn drop(&mut self) {
            self.live.fetch_sub(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn blocks_are_reused_and_freed_with_their_last_buffer() {
        let memory = Arc::new(Counted::default());
        let pool = BlockPool::new(Some(memory.clone()), 1000, false);
        let a = pool.get();
        let b = pool.get();
        assert!(!a.is_null() && !b.is_null());
        // SAFETY: two live buffers of the pool's size.
        unsafe {
            assert_eq!((*a).size, 1000);
            assert!(
                std::slice::from_raw_parts((*a).data, 1000)
                    .iter()
                    .all(|&x| x == 0)
            );
            assert_eq!((*a).data as usize % BLOCK_ALIGN, 0);
            let a_data = (*a).data;
            ff::av_buffer_unref(&mut { a });
            let c = pool.get();
            assert_eq!((*c).data, a_data, "the free block comes back");
            assert_eq!(memory.made.load(Ordering::Relaxed), 2);
            drop(pool);
            assert_eq!(memory.live.load(Ordering::Relaxed), 2, "still lent");
            ff::av_buffer_unref(&mut { b });
            ff::av_buffer_unref(&mut { c });
        }
        assert_eq!(memory.live.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn without_frame_memory_blocks_are_ordinary_memory() {
        let pool = BlockPool::new(None, 64, true);
        let mut buf = pool.get();
        assert!(!buf.is_null());
        // SAFETY: a live buffer.
        unsafe { ff::av_buffer_unref(&mut buf) };
    }
}
