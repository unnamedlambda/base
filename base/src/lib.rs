// The platform contract. Artifacts address the arena with 64-bit integers and
// write multi-byte values little-endian, and the FFI passes pointers and
// lengths as `i64`. Elsewhere those bytes would be misread rather than
// rejected, so the build refuses instead.
#[cfg(not(all(target_pointer_width = "64", target_endian = "little")))]
compile_error!("base runs only on 64-bit little-endian targets");

pub use arrow_array::RecordBatch;
use arrow_array::{ArrayRef, Float64Array, Int64Array, StringArray};
use arrow_schema::{DataType, Field, Schema};
pub use base_types::{Algorithm, Artifact, OutputBatchSchema, OutputColumn, OutputType, Setup};
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

impl Base {
    pub fn new(setup: Setup) -> Result<Self, Error> {
        // The arena holds what the program asked for and the image it ships
        // with, and nothing else: the caller's buffers are arguments, so there
        // is no header the engine has to make room for.
        let needed = setup.memory_size.max(setup.initial_memory.len());
        let mut memory = setup.initial_memory;
        memory.resize(needed, 0);
        Self::from_parts(setup.clif, memory.into_boxed_slice())
    }

    fn from_parts(
        clif: base_types::clif::Program,
        memory: Box<[u8]>,
    ) -> Result<Self, Error> {
        let _span = info_span!("base_new", memory_size = memory.len()).entered();
        info!("creating Base instance");

        let mut memory = Pin::new(memory);
        let mem_ptr = memory.as_mut().as_mut_ptr();

        let (module, clif_fns) = if clif.is_empty() {
            (None, None)
        } else {
            let (module, fns) = jit::compile_program(&clif).map_err(Error::Clif)?;
            (Some(module), Some(fns))
        };

        // Set thread-local compiled fns so FFI functions (cl_thread_init etc.) work on interpreter thread
        if let Some(ref fns) = clif_fns {
            THREAD_COMPILED_FNS.with(|cell| {
                *cell.borrow_mut() = Some(fns.clone());
            });
        }

        info!("Base instance created");
        Ok(Base {
            memory,
            mem_ptr,
            clif_fns,
            _module: module,
        })
    }

    /// The shared memory a program reads and writes.
    ///
    /// An output schema names offsets into this, so a caller that does not want
    /// the Arrow view — `capi`'s hosts — reads the same bytes directly.
    pub fn memory_bytes(&self) -> &[u8] {
        &self.memory
    }

    /// Install this instance's compiled functions on the calling thread.
    ///
    /// `from_parts` does this for the thread that built the instance. The
    /// functions live in a thread-local as well as in the instance because the
    /// FFI entry points a program calls reach them with no `Base` in hand, so a
    /// caller executing from a thread that did not build it must do this first
    /// or those calls find nothing.
    pub fn bind_current_thread(&self) {
        if let Some(ref fns) = self.clif_fns {
            THREAD_COMPILED_FNS.with(|cell| {
                *cell.borrow_mut() = Some(fns.clone());
            });
        }
    }

    pub fn execute(
        &mut self,
        algorithm: &Algorithm,
        data: &[u8],
    ) -> Result<Vec<RecordBatch>, Error> {
        self.execute_into(algorithm, data, &mut [])
    }

    pub fn execute_into(
        &mut self,
        algorithm: &Algorithm,
        data: &[u8],
        out: &mut [u8],
    ) -> Result<Vec<RecordBatch>, Error> {
        let _span = info_span!("execute", fn_idx = algorithm.fn_idx).entered();
        info!("starting execution");

        if let Some(ref fns) = self.clif_fns {
            let fn_idx = algorithm.fn_idx as usize;
            if fn_idx >= fns.len() {
                return Err(Error::Execution(format!(
                    "fn_idx {fn_idx} out of range (have {} fns)",
                    fns.len()
                )));
            }
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

        let batches = build_record_batches(&self.memory, &algorithm.output);
        info!("execution complete");
        Ok(batches)
    }
}

pub fn run(setup: Setup, algorithm: Algorithm) -> Result<Vec<RecordBatch>, Error> {
    let mut base = Base::new(setup)?;
    base.execute(&algorithm, &[])
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

fn build_record_batches(memory: &[u8], schemas: &[OutputBatchSchema]) -> Vec<RecordBatch> {
    let mut batches = Vec::with_capacity(schemas.len());
    for schema in schemas {
        let row_count = if schema.row_count_offset + 8 <= memory.len() {
            let bytes: [u8; 8] = memory[schema.row_count_offset..schema.row_count_offset + 8]
                .try_into()
                .unwrap();
            u64::from_le_bytes(bytes) as usize
        } else {
            0
        };
        if row_count == 0 {
            continue;
        }

        let mut fields = Vec::with_capacity(schema.columns.len());
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(schema.columns.len());

        for col in &schema.columns {
            match col.dtype {
                OutputType::I64 => {
                    fields.push(Field::new(&col.name, DataType::Int64, false));
                    let mut values = Vec::with_capacity(row_count);
                    for i in 0..row_count {
                        let off = col.data_offset + i * 8;
                        if off + 8 <= memory.len() {
                            let bytes: [u8; 8] = memory[off..off + 8].try_into().unwrap();
                            values.push(i64::from_le_bytes(bytes));
                        } else {
                            values.push(0);
                        }
                    }
                    arrays.push(Arc::new(Int64Array::from(values)) as ArrayRef);
                }
                OutputType::F64 => {
                    fields.push(Field::new(&col.name, DataType::Float64, false));
                    let mut values = Vec::with_capacity(row_count);
                    for i in 0..row_count {
                        let off = col.data_offset + i * 8;
                        if off + 8 <= memory.len() {
                            let bytes: [u8; 8] = memory[off..off + 8].try_into().unwrap();
                            values.push(f64::from_le_bytes(bytes));
                        } else {
                            values.push(0.0);
                        }
                    }
                    arrays.push(Arc::new(Float64Array::from(values)) as ArrayRef);
                }
                OutputType::Utf8 => {
                    fields.push(Field::new(&col.name, DataType::Utf8, false));
                    let mut strings = Vec::with_capacity(row_count);
                    let total_byte_len = if col.len_offset + 8 <= memory.len() {
                        let bytes: [u8; 8] = memory[col.len_offset..col.len_offset + 8]
                            .try_into()
                            .unwrap();
                        u64::from_le_bytes(bytes) as usize
                    } else {
                        0
                    };
                    if row_count == 1 {
                        let end = (col.data_offset + total_byte_len).min(memory.len());
                        let slice = &memory[col.data_offset..end];
                        let s = std::str::from_utf8(slice).unwrap_or("");
                        strings.push(s.to_string());
                    } else {
                        let mut pos = col.data_offset;
                        for _ in 0..row_count {
                            let start = pos;
                            while pos < memory.len() && memory[pos] != 0 {
                                pos += 1;
                            }
                            let s = std::str::from_utf8(&memory[start..pos]).unwrap_or("");
                            strings.push(s.to_string());
                            pos += 1;
                        }
                    }
                    arrays.push(Arc::new(StringArray::from(strings)) as ArrayRef);
                }
            }
        }

        if let Ok(batch) = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays) {
            batches.push(batch);
        }
    }
    batches
}

/// The program as CLIF text — what was built, rather than what the caller
/// believes was built. For eyeballing a test or a generated artifact; nothing
/// in the pipeline reads it.
pub fn clif_text(prog: &base_types::clif::Program) -> Result<String, String> {
    use base_types::clif::Callee;
    let mut out = String::new();
    let isa = cranelift_native::builder().map_err(|e| e.to_string())?;
    let cc = cranelift_codegen::isa::CallConv::triple_default(isa.triple());
    for f in &prog.functions {
        // Cranelift prints a callee as the `FuncId` it was declared with, so
        // the stub resolver records what each id stood for and the names are
        // put back afterward.
        let mut names: Vec<String> = Vec::new();
        let mut declare = |c: &Callee, _: &cranelift_codegen::ir::Signature| {
            names.push(match c {
                Callee::Import(n) => format!("%{n}"),
                Callee::Local(i) => format!("u0:{i}"),
            });
            Ok(names.len() as u32 - 1)
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
