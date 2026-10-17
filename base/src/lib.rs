// Artifacts address the arena with 64-bit little-endian integers, and pass
// pointers and lengths as `i64`.
#[cfg(not(all(target_pointer_width = "64", target_endian = "little")))]
compile_error!("base runs only on 64-bit little-endian targets");

pub use base_types::Artifact;
use std::{
    collections::HashMap,
    ptr::NonNull,
    sync::atomic::{AtomicU64, AtomicU8, Ordering},
};

pub mod c_api;
mod clif_decode;
mod builtin;
mod link;
mod jit;


#[derive(Debug)]
pub enum Error {
    /// The program is malformed.
    Clif(String),
    /// The arena cannot be allocated, or its segments are malformed.
    Arena(String),
    /// No function has the name.
    NoSuchFunction(String),
    /// A read reaches past the arena.
    Range(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Clif(m) => write!(f, "clif: {m}"),
            Error::Arena(m) => write!(f, "arena: {m}"),
            Error::NoSuchFunction(m) => write!(f, "no such function: {m}"),
            Error::Range(m) => write!(f, "out of range: {m}"),
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

/// A function of a [`Driver`], as the C function it is, borrowing the driver.
#[derive(Clone, Copy)]
pub struct Function<'d> {
    f: jit::BaseFunction,
    _driver: std::marker::PhantomData<&'d Driver>,
}

impl Function<'_> {
    /// Calls the function on `input` and `output` in place, answering its status
    /// (`0` when it returns nothing). Any number of threads may call at once.
    ///
    /// # Safety
    ///
    /// As any foreign call: the program keeps the contracts of what it calls
    /// and reaches only its arena, `input` and `output` (for one Lean built,
    /// what its no-misuse theorem states). The thread is in the default
    /// floating-point environment, and `input` and `output` meet the
    /// function's own contract.
    pub unsafe fn call(&self, input: &[u8], output: &mut [u8]) -> i64 {
        // SAFETY: loading verified the signature, the code lives as long as the
        // borrowed driver, and the rest is the caller's.
        unsafe { (self.f)(input.as_ptr(), input.len(), output.as_mut_ptr(), output.len()) }
    }

    /// The C function itself.
    pub fn as_ptr(&self) -> BaseFunction {
        self.f
    }
}

pub use jit::BaseFunction;

/// Waiting on a word, waking, and the clock: the calls programs make, for hosts.
pub use builtin::sync::{now, wait, wake};

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
    /// Allocates the artifact's arena and compiles its functions.
    pub fn load(artifact: Artifact) -> Result<Self, Error> {
        let size = usize::try_from(artifact.arena_size).map_err(|_| {
            Error::Arena(format!("the arena is {} bytes, more than this host can hold", artifact.arena_size))
        })?;
        let mut free_from = 0usize;
        for s in &artifact.data {
            let at = usize::try_from(s.offset).ok().filter(|&at| at >= free_from);
            let Some(at) = at else {
                return Err(Error::Arena(format!(
                    "a segment at {} is out of order or overlaps the one before it, which ends at {free_from}",
                    s.offset
                )));
            };
            let end = at.checked_add(s.bytes.len());
            if end.is_none_or(|end| end > size) {
                return Err(Error::Arena(format!(
                    "a segment of {} bytes at {} ends past the {size} bytes of the arena",
                    s.bytes.len(),
                    s.offset
                )));
            }
            free_from = at + s.bytes.len();
        }
        let mut arena = Arena::zeroed(size).ok_or_else(|| {
            Error::Arena(format!("the arena is {size} bytes, more than this host can hold"))
        })?;
        let bytes = arena.bytes_mut();
        for s in &artifact.data {
            let at = s.offset as usize;
            bytes[at..at + s.bytes.len()].copy_from_slice(&s.bytes);
        }
        // The code is linked to the arena's address, which never moves.
        let program = jit::compile(&artifact.primitives, &artifact.functions, arena.ptr.as_ptr()).map_err(Error::Clif)?;
        Ok(Driver { arena, exports: program.exports, module: Some(program.module) })
    }

    /// The arena, borrowed exclusively, so no call runs while it is held.
    pub fn arena(&mut self) -> &[u8] {
        self.arena.bytes_mut()
    }

    pub fn arena_size(&self) -> usize {
        self.arena.len
    }

    /// The arena's address, fixed for the driver's life. While calls run, bytes
    /// a program may touch are reached only atomically.
    pub fn arena_ptr(&self) -> *mut u8 {
        self.arena.ptr.as_ptr()
    }

    /// Copies the arena from `offset` into `out` while calls may run: each
    /// aligned word with one atomic load, so none is read torn.
    pub fn read(&self, offset: usize, out: &mut [u8]) -> Result<(), Error> {
        let len = self.arena.len;
        if offset.checked_add(out.len()).is_none_or(|end| end > len) {
            return Err(Error::Range(format!(
                "{offset}..{offset}+{} is outside the {len} bytes of the arena",
                out.len()
            )));
        }
        let base = self.arena.ptr.as_ptr();
        let mut i = 0;
        // SAFETY: in bounds, aligned where loaded as a word, and atomic.
        while i < out.len() {
            let at = unsafe { base.add(offset + i) };
            if at as usize % 8 == 0 && out.len() - i >= 8 {
                let word = unsafe { AtomicU64::from_ptr(at.cast()) }.load(Ordering::Relaxed);
                out[i..i + 8].copy_from_slice(&word.to_ne_bytes());
                i += 8;
            } else {
                out[i] = unsafe { AtomicU8::from_ptr(at) }.load(Ordering::Relaxed);
                i += 1;
            }
        }
        Ok(())
    }

    /// The function named `name`.
    pub fn function(&self, name: &str) -> Result<Function<'_>, Error> {
        let f = *self.exports.get(name).ok_or_else(|| {
            Error::NoSuchFunction(format!("the artifact has no function named {name:?}"))
        })?;
        Ok(Function { f, _driver: std::marker::PhantomData })
    }

    /// [`Driver::function`], then [`Function::call`].
    ///
    /// # Safety
    ///
    /// As [`Function::call`].
    pub unsafe fn execute(&self, name: &str, input: &[u8], output: &mut [u8]) -> Result<i64, Error> {
        Ok(unsafe { self.function(name)?.call(input, output) })
    }
}

/// Loads `artifact` and calls `name` once.
///
/// # Safety
///
/// As [`Function::call`].
pub unsafe fn run(artifact: Artifact, name: &str) -> Result<i64, Error> {
    let driver = Driver::load(artifact)?;
    unsafe { driver.execute(name, &[], &mut []) }
}

/// The decoded functions, and the name each `FuncId` stands for.
fn decoded(artifact: &Artifact) -> Result<(Vec<cranelift_codegen::ir::Function>, Vec<String>), String> {
    let d = clif_decode::decode(&artifact.primitives, &artifact.functions, jit::isa().default_call_conv())?;
    let names = artifact
        .primitives
        .iter()
        .map(|p| format!("%{}!{}", p.lib, p.symbol))
        .chain((0..artifact.functions.len()).map(|n| format!("u0:{n}")))
        .collect();
    Ok((d.functions, names))
}

/// The program as CLIF text.
pub fn clif_text(artifact: &Artifact) -> Result<String, String> {
    let (functions, names) = decoded(artifact)?;
    let mut out = String::new();
    for func in functions {
        for line in func.display().to_string().lines() {
            // A callee prints as its `FuncId`, `u0:K sig`.
            let named = line.find("u0:").and_then(|at| {
                let digits = line[at + 3..].find(|c: char| !c.is_ascii_digit()).map(|n| at + 3 + n)?;
                let k: usize = line[at + 3..digits].parse().ok()?;
                let name = names.get(k).filter(|_| line[digits..].starts_with(" sig"))?;
                Some(format!("{}{name}{}", &line[..at], &line[digits..]))
            });
            out.push_str(named.as_deref().unwrap_or(line));
            out.push('\n');
        }
        out.push('\n');
    }
    Ok(out)
}

/// The machine code each function compiles to on this host.
pub fn disassemble(artifact: &Artifact) -> Result<String, String> {
    let isa = jit::isa();
    let mut out = String::new();
    for (i, func) in decoded(artifact)?.0.into_iter().enumerate() {
        let mut ctx = cranelift_codegen::Context::for_function(func);
        ctx.set_disasm(true);
        let code = ctx
            .compile(&*isa, &mut Default::default())
            .map_err(|e| format!("compiling u0:{i}: {:?}", e.inner))?;
        out.push_str(&format!("=== u0:{i} ===\n"));
        out.push_str(code.vcode.as_deref().unwrap_or(""));
        out.push('\n');
    }
    Ok(out)
}
