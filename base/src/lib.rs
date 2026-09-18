// The platform contract. Artifacts address the arena with 64-bit integers and
// write multi-byte values little-endian, and the FFI passes pointers and
// lengths as `i64`. Elsewhere those bytes would be misread rather than
// rejected, so the build refuses instead.
#[cfg(not(all(target_pointer_width = "64", target_endian = "little")))]
compile_error!("base runs only on 64-bit little-endian targets");

pub use base_types::Artifact;
use std::{
    collections::HashMap,
    pin::Pin,
    sync::{Arc, Once},
};
use tracing::{debug, info, info_span};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

pub mod capi;
mod clif_decode;
mod ffi;
mod imports;
mod jit;

use crate::jit::THREAD_COMPILED_FNS;

#[derive(Debug)]
pub enum Error {
    /// The program an artifact carries could not be built. Nothing parses, so
    /// this is a malformed program — a value used before it is defined, a
    /// branch to a block that is not declared — rather than bad syntax.
    Clif(String),
    Execution(String),
}

pub struct Base {
    memory: Pin<Box<[u8]>>,
    mem_ptr: *mut u8,
    clif_fns: Option<Arc<Vec<jit::Compiled>>>,
    /// Each exported function's index, by its name.
    exports: HashMap<String, u32>,
    _module: Option<cranelift_jit::JITModule>,
}

unsafe impl Send for Base {}
unsafe impl Sync for Base {}

impl Drop for Base {
    /// Release the program's code. A JIT module otherwise keeps its code
    /// mapped for the life of the process, and the region it was placed in is
    /// reserved up front, so every dropped `Base` would leak that reservation.
    fn drop(&mut self) {
        // The functions stay reachable through this thread's installed table,
        // but nothing calls them without a `Base` to execute: a program's
        // workers are joined by `cl_thread_cleanup` before the program that
        // spawned them returns.
        self.clif_fns = None;
        if let Some(module) = self._module.take() {
            unsafe { module.free_memory() };
        }
    }
}

impl Base {
    pub fn new(artifact: Artifact) -> Result<Self, Error> {
        // The arena holds what the program asked for and the image it ships
        // with, and nothing else: the caller's buffers are arguments, so there
        // is no header the engine has to make room for.
        let too_big = |what: String| Error::Execution(format!("{what}, more than this host can hold"));
        let mut size = usize::try_from(artifact.required_memory)
            .map_err(|_| too_big(format!("required_memory is {}", artifact.required_memory)))?;
        for s in &artifact.data {
            let end = usize::try_from(s.offset)
                .ok()
                .and_then(|o| o.checked_add(s.bytes.len()))
                .ok_or_else(|| too_big(format!("a segment at {} ends past the address space", s.offset)))?;
            size = size.max(end);
        }
        let mut memory =
            zeroed(size).ok_or_else(|| too_big(format!("the program's memory is {size} bytes")))?;
        // Zeros everywhere a segment does not reach, which is what an artifact
        // leaves out rather than shipping.
        for s in &artifact.data {
            let at = s.offset as usize;
            memory[at..at + s.bytes.len()].copy_from_slice(&s.bytes);
        }
        Self::from_parts(artifact.functions, memory)
    }

    fn from_parts(
        functions: Vec<base_types::clif::Function>,
        memory: Box<[u8]>,
    ) -> Result<Self, Error> {
        let _span = info_span!("base_new", memory_size = memory.len()).entered();
        info!("creating Base instance");

        let mut memory = Pin::new(memory);
        let mem_ptr = memory.as_mut().as_mut_ptr();

        let (module, clif_fns) = if functions.is_empty() {
            (None, None)
        } else {
            let (module, fns) = jit::compile(&functions).map_err(Error::Clif)?;
            (Some(module), Some(fns))
        };
        let exports = exports_of(&functions, clif_fns.as_deref().map_or(&[], |f| f))?;

        info!("Base instance created");
        Ok(Base {
            memory,
            mem_ptr,
            clif_fns,
            exports,
            _module: module,
        })
    }

    /// The shared memory a program reads and writes, borrowed for as long as
    /// this `Base` is not executing.
    pub fn memory(&self) -> &[u8] {
        &self.memory
    }

    /// Call the entry point exported as `name` with `data` as its input, and
    /// answer the status it returned. The entry answers in `out`, which it
    /// sees directly: nothing is copied either way. The status is the value
    /// the program returned, passed through without being read; a program
    /// that returns nothing has status `0`.
    ///
    /// A name is the only way a host reaches a function. A function's position
    /// is the generator's to choose and free to change, so it never leaves the
    /// artifact.
    pub fn execute(&mut self, name: &str, data: &[u8], out: &mut [u8]) -> Result<i64, Error> {
        let _span = info_span!("execute", name).entered();
        info!("starting execution");

        let fn_idx = *self.exports.get(name).ok_or_else(|| {
            Error::Execution(format!("the artifact exports no entry point named {name:?}"))
        })? as usize;
        // Only a program with functions can export a name.
        let fns = self.clif_fns.clone().expect("an exported name implies compiled functions");
        // The FFI entry points a program calls — `cl_thread_init` and what
        // it spawns — reach the compiled functions through a thread-local,
        // with no `Base` in hand. Installing them here rather than at
        // construction is what lets a host execute from any thread, and
        // makes two instances on one thread each find their own.
        THREAD_COMPILED_FNS.with(|cell| *cell.borrow_mut() = Some(fns.clone()));
        debug!(fn_idx, "clif_call");
        // The caller's buffers are arguments, not a place in the arena the
        // program is told to look at. The arity is the one the function's
        // own entry block declared, and `exports_of` admitted only these two.
        let f = fns[fn_idx];
        let status = unsafe {
            match (f.arity, f.answers) {
                (5, true) => {
                    let entry: unsafe extern "C" fn(
                        *mut u8,
                        *const u8,
                        usize,
                        *mut u8,
                        usize,
                    ) -> i64 = std::mem::transmute(f.addr);
                    entry(self.mem_ptr, data.as_ptr(), data.len(), out.as_mut_ptr(), out.len())
                }
                (5, false) => {
                    let entry: unsafe extern "C" fn(*mut u8, *const u8, usize, *mut u8, usize) =
                        std::mem::transmute(f.addr);
                    entry(self.mem_ptr, data.as_ptr(), data.len(), out.as_mut_ptr(), out.len());
                    0
                }
                (1, true) => {
                    let entry: unsafe extern "C" fn(*mut u8) -> i64 =
                        std::mem::transmute(f.addr);
                    entry(self.mem_ptr)
                }
                (1, false) => {
                    let entry: unsafe extern "C" fn(*mut u8) = std::mem::transmute(f.addr);
                    entry(self.mem_ptr);
                    0
                }
                (n, _) => unreachable!("{name:?} is exported but takes {n} parameters"),
            }
        };
        info!(status, "execution complete");
        Ok(status)
    }
}

/// `size` zeroed bytes, or `None` if the allocator cannot provide them.
///
/// Zeroed by the allocator rather than written: memories run to hundreds of
/// megabytes, and pages a program never touches then never cost anything.
fn zeroed(size: usize) -> Option<Box<[u8]>> {
    if size == 0 {
        return Some(Box::default());
    }
    let layout = std::alloc::Layout::array::<u8>(size).ok()?;
    // SAFETY: `layout` is non-zero-sized, and a non-null result is `size`
    // initialized bytes allocated with the layout `Box<[u8]>` frees with.
    unsafe {
        let ptr = std::alloc::alloc_zeroed(layout);
        (!ptr.is_null()).then(|| Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, size)))
    }
}

/// The exported names of `functions`, compiled as `compiled`.
///
/// A name is refused twice over: a second function under it would make which
/// one a host reaches depend on the order a map was filled, and a named
/// function that is not shaped like an entry point would be called with
/// arguments it does not take.
fn exports_of(
    functions: &[base_types::clif::Function],
    compiled: &[jit::Compiled],
) -> Result<HashMap<String, u32>, Error> {
    let mut exports = HashMap::new();
    for (i, (f, c)) in functions.iter().zip(compiled).enumerate() {
        let Some(name) = &f.entry_name else { continue };
        if c.arity != 5 && c.arity != 1 {
            return Err(Error::Clif(format!(
                "u0:{i} is exported as {name:?} but takes {} parameters; an entry point \
                 takes the memory base, optionally followed by the input and output buffers",
                c.arity
            )));
        }
        if let Some(prev) = exports.insert(name.clone(), i as u32) {
            return Err(Error::Clif(format!(
                "u0:{prev} and u0:{i} are both exported as {name:?}"
            )));
        }
    }
    Ok(exports)
}

/// Compile an artifact and call the entry point it exports as `name`, once.
pub fn run(artifact: Artifact, name: &str) -> Result<i64, Error> {
    let mut base = Base::new(artifact)?;
    base.execute(name, &[], &mut [])
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
pub fn clif_text(functions: &[base_types::clif::Function]) -> Result<String, String> {
    use base_types::clif::Callee;
    let mut out = String::new();
    let isa = cranelift_native::builder().map_err(|e| e.to_string())?;
    let cc = cranelift_codegen::isa::CallConv::triple_default(isa.triple());
    for (i, f) in functions.iter().enumerate() {
        // Cranelift prints a callee as the `FuncId` it was declared with, so
        // the stub resolver records what each id stood for and the names are
        // put back afterward.
        let mut names: Vec<String> = Vec::new();
        let mut declare = |c: &Callee| {
            let sig = match c {
                Callee::Import(n) => {
                    let import = imports::lookup(n)
                        .ok_or_else(|| format!("u0:{i} imports {n}, which base does not provide"))?;
                    clif_decode::signature(&import.params, import.result, cc)
                }
                Callee::Local(n) => {
                    let callee = functions
                        .get(*n as usize)
                        .ok_or_else(|| format!("call to u0:{n}, which the program does not define"))?;
                    clif_decode::signature_of(callee, *n as usize, cc)?
                }
            };
            names.push(match c {
                Callee::Import(n) => format!("%{n}"),
                Callee::Local(i) => format!("u0:{i}"),
            });
            Ok((clif_decode::Resolved { id: names.len() as u32 - 1, colocated: false }, sig))
        };
        let text = format!("{}", clif_decode::decode_function(f, i, cc, &mut declare)?);
        let mut text = text;
        for (i, name) in names.iter().enumerate().rev() {
            text = text.replace(&format!("= u0:{i} sig"), &format!("= {name} sig"));
        }
        out.push_str(&text);
        out.push('\n');
    }
    Ok(out)
}
