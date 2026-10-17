// The platform contract. Artifacts address the arena with 64-bit integers and
// write multi-byte values little-endian, and the FFI passes pointers and
// lengths as `i64`. Elsewhere those bytes would be misread rather than
// rejected, so the build refuses instead.
#[cfg(not(all(target_pointer_width = "64", target_endian = "little")))]
compile_error!("base runs only on 64-bit little-endian targets");

pub use base_types::Artifact;
use std::{
    collections::HashMap,
    ptr::NonNull,
    sync::{
        atomic::{AtomicU64, AtomicU8, Ordering},
        Once,
    },
};
use tracing::{info, info_span};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

pub mod capi;
mod clif_decode;
mod ffi;
mod libs;
mod jit;


#[derive(Debug)]
pub enum Error {
    /// The program an artifact carries could not be built. Nothing parses, so
    /// this is a malformed program — a value used before it is defined, a
    /// branch to a block that is not declared — rather than bad syntax.
    Clif(String),
    /// The arena could not be allocated. Nothing has run: this is the size the
    /// artifact asked for against what the host can give.
    Arena(String),
    /// The artifact has no function by that name.
    NoSuchFunction(String),
    /// A read reached past the end of the arena.
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

/// The arena a program runs in: its own pages, mapped anonymous and zeroed,
/// as a loader maps a library's `.bss`.
///
/// Held as a pointer rather than a slice: running programs write it from any
/// number of threads while the `Driver` is shared, and a `&mut [u8]` asserts
/// that nothing else reaches its bytes.
///
/// Page-aligned, so an address a program keeps naturally aligned relative to
/// its base is naturally aligned in fact, which an atomic needs, and huge
/// pages can back it: Linux aligns a large anonymous mapping to them itself.
/// Pages a program never touches never cost anything.
struct Arena {
    ptr: NonNull<u8>,
    len: usize,
    /// The mapping, unmapped when the arena goes; none for an empty arena.
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

/// A function of a [`Driver`], as the C function it is: calling it runs the
/// program's code directly, with nothing of the engine's in between.
///
/// It borrows the driver, so it cannot outlive it. A host outside Rust gets the
/// same function from `base_driver_function`, and keeping the driver alive for as
/// long as it calls it is then its own part.
#[derive(Clone, Copy)]
pub struct Function<'d> {
    f: jit::FunctionPtr,
    _driver: std::marker::PhantomData<&'d Driver>,
}

impl Function<'_> {
    /// Call the function with `input`, answering in `output`, and answer
    /// the status it returned (`0` from one that returns nothing). The buffers
    /// are the program's to read and write directly: nothing is copied.
    ///
    /// Any number of threads may call functions of one driver at once. Each
    /// call runs on its caller's thread against the one arena, and how calls
    /// share it is the program's to decide: with atomics, as `shared_sum`
    /// does, or by keeping to disjoint bytes.
    ///
    /// # Safety
    ///
    /// A load answers whether the artifact is a program, not whether running
    /// it is defined: that is the program's to keep, as with any foreign
    /// function. It keeps every contract of the calls it makes and reaches
    /// only its arena, its input and its output, which for an artifact Lean
    /// built is what its no-misuse theorem states, under the capabilities the
    /// theorem names. The thread calling it is in the default floating-point
    /// environment, and `input` and `output` are as the function's own
    /// contract asks: for one Lean built, not bytes of the arena.
    pub unsafe fn call(&self, input: &[u8], output: &mut [u8]) -> i64 {
        // SAFETY: the function has `FunctionPtr`'s signature, which loading
        // declared and verified, and is mapped for as long as the driver this
        // borrows; the buffers are valid for their lengths; the rest is the
        // caller's, above.
        unsafe { (self.f)(input.as_ptr(), input.len(), output.as_mut_ptr(), output.len()) }
    }

    /// The function itself, for a host that calls it through a C pointer.
    pub fn as_ptr(&self) -> FunctionPtr {
        self.f
    }
}

pub use jit::FunctionPtr;

/// Waiting on a word of the arena, waking, and the clock, for a host:
/// the system's own calls, which programs make too, so a host's thread and a
/// program's meet at a word ([`Driver::arena_ptr`] plus an offset).
pub use ffi::sync::{now, wait, wake};

pub struct Driver {
    arena: Arena,
    /// Each function a host may call, by its name.
    exports: HashMap<String, FunctionPtr>,
    _module: Option<cranelift_jit::JITModule>,
}

unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

impl Drop for Driver {
    /// Release the program's code. A JIT module otherwise keeps its code
    /// mapped for the life of the process, and the region it was placed in is
    /// reserved up front, so every dropped `Driver` would leak that reservation.
    fn drop(&mut self) {
        // Nothing runs the code once the driver goes: a host may call a function
        // only while the driver lives, and a program finishes the
        // threads it started before then: both are what a caller of
        // `Function::call` keeps.
        if let Some(module) = self._module.take() {
            unsafe { module.free_memory() };
        }
    }
}

impl Driver {
    /// Load an artifact: compile its functions and allocate the arena its
    /// program runs in. There is no driver without an artifact, so this is the
    /// only way to obtain one.
    pub fn load(artifact: Artifact) -> Result<Self, Error> {
        // The arena is the size the artifact states, which covers its image:
        // the generator decides it, and a segment past it is refused.
        let size = usize::try_from(artifact.arena_size).map_err(|_| {
            Error::Arena(format!("the arena is {} bytes, more than this host can hold", artifact.arena_size))
        })?;
        // In order and apart, so the image has one spelling.
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
        // Zeros everywhere a segment does not reach, which is what an artifact
        // leaves out rather than shipping.
        let bytes = arena.bytes_mut();
        for s in &artifact.data {
            let at = s.offset as usize;
            bytes[at..at + s.bytes.len()].copy_from_slice(&s.bytes);
        }
        Self::from_parts(&artifact.primitives, &artifact.functions, arena)
    }

    fn from_parts(
        primitives: &[base_types::clif::Primitive],
        functions: &[base_types::clif::Function],
        arena: Arena,
    ) -> Result<Self, Error> {
        let _span = info_span!("base_driver_load", memory_size = arena.len).entered();
        info!("loading artifact");

        let driver = if functions.is_empty() {
            Driver { arena, exports: HashMap::new(), _module: None }
        } else {
            // The program's code is linked to the arena's address, so it is
            // fixed before compiling, and it never moves.
            let program = jit::compile(primitives, functions, arena.ptr.as_ptr()).map_err(Error::Clif)?;
            Driver {
                arena,
                exports: program.exports,
                _module: Some(program.module),
            }
        };
        info!("driver ready");
        Ok(driver)
    }

    /// The arena, borrowed exclusively: nothing can execute while the slice is
    /// held, which is what makes a plain slice of it sound. [`Driver::read`] is
    /// the way in while programs run.
    pub fn arena(&mut self) -> &[u8] {
        self.arena.bytes_mut()
    }

    /// How many bytes the arena holds.
    pub fn arena_size(&self) -> usize {
        self.arena.len
    }

    /// Where the arena is: the same address for as long as the driver lives,
    /// and the one its programs see as their base.
    ///
    /// This is how a host shares the arena with running programs rather than
    /// copying out of it. While any call runs, the host reaches bytes a
    /// program may also touch only with atomic operations, as the program
    /// does, and is free with bytes no running program touches. That is the
    /// rule two threads of one C program follow, and it lets a host take the
    /// ordering it needs: an acquire load of a flag a program set with a
    /// release store, then the data the flag guards.
    pub fn arena_ptr(&self) -> *mut u8 {
        self.arena.ptr.as_ptr()
    }

    /// Copy `out.len()` bytes of the arena from `offset` into `out`.
    ///
    /// Sound while any number of calls run. Every naturally aligned 8-byte
    /// word the range covers is read with one relaxed atomic load, so a word a
    /// program updates atomically, like `shared_sum`'s total, is never copied
    /// torn: it is a value the word held at some point during the read. The
    /// bytes outside such words are read one at a time, and different words
    /// may be copied at different moments.
    ///
    /// A range reaching past the end is an error rather than a short read, so
    /// a truncated read cannot be mistaken for a result.
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
        // SAFETY, for both loads: in bounds, checked above; the word's address
        // is 8-aligned where it is loaded as one; and every access a host makes
        // while programs run is atomic.
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

    /// The function named `name`, as the C function it is. A host calling one
    /// function often finds it once and calls it.
    pub fn function(&self, name: &str) -> Result<Function<'_>, Error> {
        let f = *self.exports.get(name).ok_or_else(|| {
            Error::NoSuchFunction(format!("the artifact has no function named {name:?}"))
        })?;
        Ok(Function { f, _driver: std::marker::PhantomData })
    }

    /// Call the function named `name` with `input`, answering in `output`:
    /// [`Driver::function`], then [`Function::call`].
    ///
    /// # Safety
    ///
    /// As [`Function::call`].
    pub unsafe fn execute(&self, name: &str, input: &[u8], output: &mut [u8]) -> Result<i64, Error> {
        Ok(unsafe { self.function(name)?.call(input, output) })
    }
}

/// Compile an artifact and call its function named `name`, once.
///
/// # Safety
///
/// As [`Function::call`].
pub unsafe fn run(artifact: Artifact, name: &str) -> Result<i64, Error> {
    let driver = Driver::load(artifact)?;
    unsafe { driver.execute(name, &[], &mut []) }
}

pub fn init_tracing() {
    static INIT: Once = Once::new();

    INIT.call_once(|| {
        tracing_subscriber::registry()
            .with(
                fmt::layer()
                    .with_writer(std::io::stderr)
                    .with_target(true)
                    .with_thread_ids(true)
                    .with_filter(
                        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("off")),
                    ),
            )
            .init();
    });
}

/// The program as CLIF text — what was built, rather than what the caller
/// believes was built. For eyeballing a test or a generated artifact; nothing
/// in the pipeline reads it.
pub fn clif_text(artifact: &Artifact) -> Result<String, String> {
    use base_types::clif::Callee;
    let mut out = String::new();
    let isa = cranelift_native::builder().map_err(|e| e.to_string())?;
    let cc = cranelift_codegen::isa::CallConv::triple_default(isa.triple());
    for (i, f) in artifact.functions.iter().enumerate() {
        // Cranelift prints a callee as the `FuncId` it was declared with, so
        // the stub resolver records what each id stood for and the names are
        // put back afterward.
        let mut names: Vec<String> = Vec::new();
        let mut resolve = |c: Callee| {
            let (name, sig) = match c {
                Callee::Function(n) => {
                    let callee = artifact
                        .functions
                        .get(n as usize)
                        .ok_or_else(|| format!("a call to u0:{n}, which the program does not define"))?;
                    (format!("u0:{n}"), clif_decode::signature_of(callee, n as usize, cc)?)
                }
                Callee::Primitive(n) => {
                    let p = artifact
                        .primitives
                        .get(n as usize)
                        .ok_or_else(|| format!("a call to primitive {n}, which the program does not declare"))?;
                    (format!("%{}!{}", p.lib, p.symbol), clif_decode::signature(&p.signature, cc)?)
                }
            };
            names.push(name);
            Ok((clif_decode::Resolved { id: names.len() as u32 - 1, colocated: false }, sig))
        };
        let mut module = clif_decode::Module { primitives: &artifact.primitives, arena: 0, resolve: &mut resolve };
        let mut text = format!("{}", clif_decode::decode_function(f, i, cc, &mut module)?);
        for (i, name) in names.iter().enumerate().rev() {
            text = text.replace(&format!("= u0:{i} sig"), &format!("= {name} sig"));
        }
        out.push_str(&text);
        out.push('\n');
    }
    Ok(out)
}
