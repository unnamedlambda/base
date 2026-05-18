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

mod ffi;
mod jit;

use crate::jit::{compile_cranelift_ir, THREAD_COMPILED_FNS};
use base_types::RuntimeHeader;

#[derive(Debug)]
pub enum Error {
    ClifParse(String),
    Execution(String),
}

pub struct Base {
    memory: Pin<Box<[u8]>>,
    mem_ptr: *mut u8,
    clif_fns: Option<Arc<Vec<unsafe extern "C" fn(*mut u8)>>>,
    _module: Option<cranelift_jit::JITModule>,
    runtime_header: RuntimeHeader,
}

unsafe impl Send for Base {}
unsafe impl Sync for Base {}

impl Base {
    pub fn new(setup: Setup) -> Result<Self, Error> {
        let header_end = setup
            .runtime_header
            .out_len_offset
            .saturating_add(std::mem::size_of::<usize>());
        let needed = setup
            .memory_size
            .max(setup.initial_memory.len())
            .max(header_end);
        let mut memory = setup.initial_memory;
        memory.resize(needed, 0);
        Self::from_parts(
            setup.cranelift_ir,
            setup.runtime_header,
            memory.into_boxed_slice(),
        )
    }

    fn from_parts(
        cranelift_ir: String,
        runtime_header: RuntimeHeader,
        memory: Box<[u8]>,
    ) -> Result<Self, Error> {
        let _span = info_span!("base_new", memory_size = memory.len()).entered();
        info!("creating Base instance");

        let mut memory = Pin::new(memory);
        let mem_ptr = memory.as_mut().as_mut_ptr();

        let (module, clif_fns) = if !cranelift_ir.is_empty() {
            let (module, fns) = compile_cranelift_ir(&cranelift_ir).map_err(Error::ClifParse)?;
            (Some(module), Some(fns))
        } else {
            (None, None)
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
            runtime_header,
        })
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

        // Write data/out pointer + length into reserved region so CLIF code can access
        // the caller's buffer directly via pointer (zero-copy).
        unsafe {
            std::ptr::write_unaligned(
                self.memory[self.runtime_header.data_ptr_offset..].as_mut_ptr() as *mut *const u8,
                data.as_ptr(),
            );
            std::ptr::write_unaligned(
                self.memory[self.runtime_header.data_len_offset..].as_mut_ptr() as *mut usize,
                data.len(),
            );
            std::ptr::write_unaligned(
                self.memory[self.runtime_header.out_ptr_offset..].as_mut_ptr() as *mut *mut u8,
                out.as_mut_ptr(),
            );
            std::ptr::write_unaligned(
                self.memory[self.runtime_header.out_len_offset..].as_mut_ptr() as *mut usize,
                out.len(),
            );
        }

        if let Some(ref fns) = self.clif_fns {
            let fn_idx = algorithm.fn_idx as usize;
            if fn_idx >= fns.len() {
                return Err(Error::Execution(format!(
                    "fn_idx {fn_idx} out of range (have {} fns)",
                    fns.len()
                )));
            }
            debug!(fn_idx, "clif_call");
            unsafe { fns[fn_idx](self.mem_ptr) };
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
