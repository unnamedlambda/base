// Artifacts address the arena with 64-bit little-endian integers, and pass
// pointers and lengths as `i64`.
#[cfg(not(all(target_pointer_width = "64", target_endian = "little")))]
compile_error!("base runs only on 64-bit little-endian targets");

use base_types::Artifact;
use std::{
    collections::HashMap,
    ptr::NonNull,
    sync::atomic::AtomicU8,
};

pub mod c_api;
mod clif_decode;
mod builtin;
mod link;
mod jit;


#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The bytes are not an encoded artifact.
    Encoding(String),
    /// The program is malformed.
    Clif(String),
    /// The arena is larger than this host can allocate.
    ArenaTooLarge { size: u64 },
    /// A data segment starts before the previous one ends.
    SegmentOverlap { offset: u64, previous_end: usize },
    /// A data segment ends past the arena.
    SegmentPastEnd { offset: u64, len: usize, arena: usize },
    /// No function has this name.
    NoSuchFunction(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Encoding(m) => write!(f, "{m}"),
            Error::Clif(m) => write!(f, "clif: {m}"),
            Error::ArenaTooLarge { size } => write!(f, "the arena is {size} bytes, more than this host can hold"),
            Error::SegmentOverlap { offset, previous_end } => write!(
                f,
                "a segment at {offset} is out of order or overlaps the one before it, which ends at {previous_end}"
            ),
            Error::SegmentPastEnd { offset, len, arena } => {
                write!(f, "a segment of {len} bytes at {offset} ends past the {arena} bytes of the arena")
            }
            Error::NoSuchFunction(name) => write!(f, "the artifact has no function named {name:?}"),
        }
    }
}

impl std::error::Error for Error {}

/// The arena: zeroed, page-aligned pages of its own. A pointer, not a slice,
/// since running programs write it from any thread.
struct Arena {
    ptr: NonNull<u8>,
    len: usize,
    /// Unmapped on drop; none for an empty arena.
    _map: Option<region::Allocation>,
}

impl Arena {
    /// `len` zeroed bytes, or `None` if the system will not map them.
    fn zeroed(len: usize) -> Option<Self> {
        if len == 0 {
            // Never read or written.
            return Some(Arena { ptr: NonNull::<u128>::dangling().cast(), len, _map: None });
        }
        let mut map = region::alloc(len, region::Protection::READ_WRITE).ok()?;
        let ptr = NonNull::new(map.as_mut_ptr::<u8>())?;
        Some(Arena { ptr, len, _map: Some(map) })
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: `len` initialized bytes, and `&mut self` is the only access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

pub use jit::BaseFunction;


pub struct Driver {
    arena: Arena,
    /// The named functions.
    exports: HashMap<String, BaseFunction>,
    /// Taken only by `drop`, which frees its code.
    module: Option<cranelift_jit::JITModule>,
}

unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

impl Drop for Driver {
    fn drop(&mut self) {
        // SAFETY: no call, and no thread a program started, outlives the driver.
        if let Some(module) = self.module.take() {
            unsafe { module.free_memory() };
        }
    }
}

impl Driver {
    /// Decodes an encoded artifact, allocates its arena and compiles its
    /// functions.
    pub fn load(artifact: &[u8]) -> Result<Self, Error> {
        let artifact = Artifact::from_bytes(artifact).map_err(Error::Encoding)?;
        let too_large = Error::ArenaTooLarge { size: artifact.arena_size };
        let size = usize::try_from(artifact.arena_size).map_err(|_| too_large.clone())?;
        let mut free_from = 0usize;
        for s in &artifact.data {
            let at = usize::try_from(s.offset).ok().filter(|&at| at >= free_from);
            let Some(at) = at else {
                return Err(Error::SegmentOverlap { offset: s.offset, previous_end: free_from });
            };
            let end = at.checked_add(s.bytes.len());
            if end.is_none_or(|end| end > size) {
                return Err(Error::SegmentPastEnd { offset: s.offset, len: s.bytes.len(), arena: size });
            }
            free_from = at + s.bytes.len();
        }
        let mut arena = Arena::zeroed(size).ok_or(too_large)?;
        let bytes = arena.bytes_mut();
        for s in &artifact.data {
            let at = s.offset as usize;
            bytes[at..at + s.bytes.len()].copy_from_slice(&s.bytes);
        }
        // The code is linked to the arena's address, which never moves.
        let program = jit::compile(&artifact.primitives, &artifact.functions, arena.ptr.as_ptr()).map_err(Error::Clif)?;
        Ok(Driver { arena, exports: program.exports, module: Some(program.module) })
    }

    /// The arena, at an address fixed for the driver's life. Calls may run
    /// while it is held, so every byte is reached atomically.
    pub fn arena(&self) -> &[AtomicU8] {
        // SAFETY: `len` bytes, live as long as the driver, and `AtomicU8` has
        // `u8`'s size and alignment.
        unsafe { std::slice::from_raw_parts(self.arena.ptr.as_ptr().cast(), self.arena.len) }
    }

    /// The function named `name`, as a Rust function: `f(&input, &mut output)`
    /// runs it on the caller's buffers in place and answers its status (`0`
    /// when it returns nothing). It is `Copy`, any number of threads may call
    /// it at once, and it borrows the driver, which outlives every call.
    ///
    /// # Safety
    ///
    /// The one trust this asks for, as `libloading`'s `Library::get` does: the
    /// function is safe to call on any `input` and `output`. It keeps the
    /// contracts of what it calls and reaches only its arena, `input` and
    /// `output`. For an artifact Lean built, that is its no-misuse theorem.
    /// The floating-point environment safe Rust runs in is the one it expects.
    pub unsafe fn function(
        &self,
        name: &str,
    ) -> Result<impl Fn(&[u8], &mut [u8]) -> i64 + Copy + Send + Sync + '_, Error> {
        let f = self.exports.get(name).copied().ok_or_else(|| Error::NoSuchFunction(name.to_owned()))?;
        Ok(move |input: &[u8], output: &mut [u8]| {
            // SAFETY: loading verified the signature, the code lives as long as
            // the borrowed driver, and the rest the caller of `function` vouched.
            unsafe { f(input.as_ptr(), input.len(), output.as_mut_ptr(), output.len()) }
        })
    }
}
