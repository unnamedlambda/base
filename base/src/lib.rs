// The platform contract. Artifacts address the arena with 64-bit integers and
// write multi-byte values little-endian, and the FFI passes pointers and
// lengths as `i64`. Elsewhere those bytes would be misread rather than
// rejected, so the build refuses instead.
#[cfg(not(all(target_pointer_width = "64", target_endian = "little")))]
compile_error!("base runs only on 64-bit little-endian targets");

pub use base_types::Artifact;
use std::{
    pin::Pin,
    sync::{Arc, Once},
};
use tracing::{debug, info, info_span};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

pub mod capi;
mod clif_decode;
mod ffi;
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
        let past_last = artifact
            .data
            .iter()
            .map(|s| s.offset + s.bytes.len())
            .max()
            .unwrap_or(0);
        let mut memory = vec![0u8; artifact.memory_size.max(past_last)];
        // Zeros everywhere a segment does not reach, which is what an artifact
        // leaves out rather than shipping.
        for s in &artifact.data {
            memory[s.offset..s.offset + s.bytes.len()].copy_from_slice(&s.bytes);
        }
        Self::from_parts(artifact.functions, memory.into_boxed_slice())
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

        info!("Base instance created");
        Ok(Base {
            memory,
            mem_ptr,
            clif_fns,
            _module: module,
        })
    }

    /// The shared memory a program reads and writes, borrowed for as long as
    /// this `Base` is not executing.
    pub fn memory_bytes(&self) -> &[u8] {
        &self.memory
    }

    /// Call the entry point at `fn_idx`, with nothing to answer through.
    pub fn execute(&mut self, fn_idx: u32, data: &[u8]) -> Result<(), Error> {
        self.execute_into(fn_idx, data, &mut [])
    }

    /// Call the entry point at `fn_idx`, which answers in `out`.
    ///
    /// Which index does what is the artifact generator's knowledge: base
    /// checks only that the index exists and that the function it names is
    /// shaped like an entry point.
    pub fn execute_into(
        &mut self,
        fn_idx: u32,
        data: &[u8],
        out: &mut [u8],
    ) -> Result<(), Error> {
        let _span = info_span!("execute", fn_idx).entered();
        info!("starting execution");

        if let Some(ref fns) = self.clif_fns {
            let fn_idx = fn_idx as usize;
            if fn_idx >= fns.len() {
                return Err(Error::Execution(format!(
                    "fn_idx {fn_idx} out of range (have {} fns)",
                    fns.len()
                )));
            }
            // The FFI entry points a program calls — `cl_thread_init` and what
            // it spawns — reach the compiled functions through a thread-local,
            // with no `Base` in hand. Installing them here rather than at
            // construction is what lets a host execute from any thread, and
            // makes two instances on one thread each find their own.
            THREAD_COMPILED_FNS.with(|cell| *cell.borrow_mut() = Some(fns.clone()));
            debug!(fn_idx, "clif_call");
            // The caller's buffers are arguments, not a place in the arena the
            // program is told to look at. The arity is the one the function's
            // own entry block declared: an entry point takes the arena base
            // and both buffers, a program with no use for them may take the
            // base alone, and anything else is not shaped like an entry point
            // and would read registers of whatever happened to be in them.
            let f = fns[fn_idx];
            unsafe {
                match f.arity {
                    5 => {
                        let entry: unsafe extern "C" fn(*mut u8, *const u8, usize, *mut u8, usize) =
                            std::mem::transmute(f.addr);
                        entry(self.mem_ptr, data.as_ptr(), data.len(), out.as_mut_ptr(), out.len());
                    }
                    1 => {
                        let entry: unsafe extern "C" fn(*mut u8) = std::mem::transmute(f.addr);
                        entry(self.mem_ptr);
                    }
                    n => {
                        return Err(Error::Execution(format!(
                            "fn_idx {fn_idx} takes {n} parameters; an entry point takes the memory base, optionally followed by the input and output buffers"
                        )));
                    }
                }
            }
        }

        info!("execution complete");
        Ok(())
    }
}

/// Compile an artifact and call one of its entry points, once.
pub fn run(artifact: Artifact, fn_idx: u32) -> Result<(), Error> {
    let mut base = Base::new(artifact)?;
    base.execute(fn_idx, &[])
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
    for f in functions {
        // Cranelift prints a callee as the `FuncId` it was declared with, so
        // the stub resolver records what each id stood for and the names are
        // put back afterward.
        let mut names: Vec<String> = Vec::new();
        let mut declare = |c: &Callee, _: &cranelift_codegen::ir::Signature| {
            names.push(match c {
                Callee::Import(n) => format!("%{n}"),
                Callee::Local(i) => format!("u0:{i}"),
            });
            Ok(clif_decode::Resolved { id: names.len() as u32 - 1, colocated: false })
        };
        let text = format!("{}", clif_decode::decode_function(f, cc, &mut declare)?);
        let mut text = text;
        for (i, name) in names.iter().enumerate().rev() {
            text = text.replace(&format!("= u0:{i} sig"), &format!("= {name} sig"));
        }
        out.push_str(&text);
        out.push('\n');
    }
    Ok(out)
}
