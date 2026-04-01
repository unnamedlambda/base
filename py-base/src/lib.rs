use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use base_types::{Algorithm, BaseConfig};
use arrow_array::{Array, RecordBatch, StructArray};
use arrow_array::ffi::{to_ffi, FFI_ArrowArray};
use arrow_schema::ffi::FFI_ArrowSchema;

#[pyclass(name = "BaseConfig")]
struct PyBaseConfig {
    inner: BaseConfig,
}

#[pymethods]
impl PyBaseConfig {
    #[new]
    fn new(json: &str) -> PyResult<Self> {
        let inner: BaseConfig = serde_json::from_str(json)
            .map_err(|e| PyValueError::new_err(format!("Invalid BaseConfig JSON: {}", e)))?;
        Ok(Self { inner })
    }
}

#[pyclass(name = "Algorithm")]
struct PyAlgorithm {
    inner: Algorithm,
}

#[pymethods]
impl PyAlgorithm {
    #[new]
    fn new(json: &str) -> PyResult<Self> {
        let inner: Algorithm = serde_json::from_str(json)
            .map_err(|e| PyValueError::new_err(format!("Invalid Algorithm JSON: {}", e)))?;
        Ok(Self { inner })
    }
}

/// Convert a Vec<RecordBatch> to a Python list of PyArrow RecordBatches via the C Data Interface.
/// Zero-copy: PyArrow takes ownership of the Arrow buffers through the C FFI pointers.
fn batches_to_pyarrow(py: Python<'_>, batches: Vec<RecordBatch>) -> PyResult<PyObject> {
    if batches.is_empty() {
        return Ok(pyo3::types::PyList::empty_bound(py).into());
    }

    let rb_class = py.import_bound("pyarrow")?.getattr("RecordBatch")?;
    let mut py_batches = Vec::with_capacity(batches.len());

    for batch in batches {
        let struct_array = StructArray::from(batch);
        let data = struct_array.into_data();
        let (mut ffi_array, mut ffi_schema) = to_ffi(&data)
            .map_err(|e| PyValueError::new_err(format!("Arrow FFI export failed: {}", e)))?;

        let array_ptr = &mut ffi_array as *mut FFI_ArrowArray as usize;
        let schema_ptr = &mut ffi_schema as *mut FFI_ArrowSchema as usize;

        let py_batch = rb_class.call_method1("_import_from_c", (array_ptr, schema_ptr))?;
        py_batches.push(py_batch);
    }

    Ok(pyo3::types::PyList::new_bound(py, &py_batches).into())
}

/// Wrapper that asserts a closure is Ungil (safe to run without the GIL).
/// Caller must ensure captured references remain valid during execution
/// and that no Python objects are accessed inside the closure.
struct UnsafeUngil<F>(F);
unsafe impl<F> Send for UnsafeUngil<F> {}
unsafe impl<F> Sync for UnsafeUngil<F> {}
impl<F: FnOnce() -> T, T> UnsafeUngil<F> {
    fn call(self) -> T {
        (self.0)()
    }
}

fn allow_threads_unsafe<F, T>(py: Python<'_>, f: F) -> T
where
    F: FnOnce() -> T,
    T: Send,
{
    let wrapped = UnsafeUngil(f);
    py.allow_threads(move || wrapped.call())
}

/// Base execution engine. JIT compiles once, executes many times.
/// Releases the GIL during execution so other Python threads can run.
#[pyclass(name = "Base")]
struct PyBase {
    inner: base::Base,
}

#[pymethods]
impl PyBase {
    #[new]
    fn new(config: &PyBaseConfig) -> PyResult<Self> {
        let inner = base::Base::new(config.inner.clone())
            .map_err(|e| PyValueError::new_err(format!("Base::new failed: {:?}", e)))?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (algorithm, data=None))]
    fn execute(
        &mut self,
        py: Python<'_>,
        algorithm: &PyAlgorithm,
        data: Option<&[u8]>,
    ) -> PyResult<PyObject> {
        let data = data.unwrap_or(&[]);
        let batches = allow_threads_unsafe(py, || {
            self.inner.execute(&algorithm.inner, data)
        }).map_err(|e| PyValueError::new_err(format!("execute failed: {:?}", e)))?;
        batches_to_pyarrow(py, batches)
    }

    fn execute_into(
        &mut self,
        py: Python<'_>,
        algorithm: &PyAlgorithm,
        data: &[u8],
        out: &Bound<'_, pyo3::types::PyByteArray>,
    ) -> PyResult<PyObject> {
        let out_slice = unsafe {
            std::slice::from_raw_parts_mut(out.data() as *mut u8, out.len())
        };
        let batches = allow_threads_unsafe(py, || {
            self.inner.execute_into(&algorithm.inner, data, out_slice)
        }).map_err(|e| PyValueError::new_err(format!("execute_into failed: {:?}", e)))?;
        batches_to_pyarrow(py, batches)
    }
}

/// One-shot execution: JIT compile and execute in a single call.
#[pyfunction]
fn run(py: Python<'_>, config: &PyBaseConfig, algorithm: &PyAlgorithm) -> PyResult<PyObject> {
    let config = config.inner.clone();
    let algorithm = algorithm.inner.clone();
    let batches = allow_threads_unsafe(py, || {
        base::run(config, algorithm)
    }).map_err(|e| PyValueError::new_err(format!("run failed: {:?}", e)))?;
    batches_to_pyarrow(py, batches)
}

#[pymodule]
fn py_base(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBaseConfig>()?;
    m.add_class::<PyAlgorithm>()?;
    m.add_class::<PyBase>()?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(())
}
