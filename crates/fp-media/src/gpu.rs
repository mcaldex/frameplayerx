//! Decoded pictures the GPU copies out of the hardware decoder's buffers, so
//! that 8K frames never pass through the CPU. fp-media has no GPU of its
//! own: the renderer implements [`GpuCopier`] and the player is handed it
//! ([`crate::PlayerConfig::gpu`]).

use std::any::Any;
use std::os::fd::OwnedFd;
use std::sync::Arc;

/// Takes decoded pictures out of decoder buffers on the GPU.
pub trait GpuCopier: Send + Sync + std::fmt::Debug {
    /// Imports a decoder buffer of `len` bytes, exported as a DMA-BUF.
    fn import(&self, fd: OwnedFd, len: usize) -> Result<Box<dyn DecoderBuffer>, String>;
}

/// A decoder buffer the GPU can read.
pub trait DecoderBuffer: Send {
    /// Starts copying `ranges` (offset, length) one after another into a new
    /// GPU buffer. GPU work submitted later sees the copy complete; the
    /// decoder must not have its buffer back until [`DecoderBuffer::copied`].
    fn copy(&self, ranges: &[(usize, usize)]) -> Result<GpuBytes, String>;
    /// Whether the last copy out of this buffer has finished, waiting for it
    /// when `wait`.
    fn copied(&self, wait: bool) -> bool;
}

/// A picture's bytes in GPU memory.
pub trait GpuBuffer: Send + Sync + Any {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The bytes, read back (for tests and the decode benchmark).
    fn read(&self) -> Result<Vec<u8>, String>;
    fn as_any(&self) -> &dyn Any;
}

pub type GpuBytes = Arc<dyn GpuBuffer>;
