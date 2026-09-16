//! Python over base's C ABI.
//!
//! Nothing here reaches into the runtime's Rust types: every call goes through
//! `base::capi`, the same six functions a C or Lean host calls. That is what
//! keeps the three hosts on one core rather than three surfaces that drift —
//! the way `execute` answering Arrow on one side and bytes on another once did.
//!
//! What this adds is Python's conventions: an exception instead of a failure
//! value and the message the runtime left, the buffer protocol for the caller's
//! input and output, and the GIL released while a program runs.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use base::capi;

/// The message the failed call left on this thread, as a Python exception.
fn last_error(what: &str) -> PyErr {
    let len = unsafe { capi::base_last_error(std::ptr::null_mut(), 0) };
    let mut buf = vec![0u8; len];
    unsafe { capi::base_last_error(buf.as_mut_ptr(), buf.len()) };
    match String::from_utf8(buf) {
        Ok(message) if !message.is_empty() => PyValueError::new_err(format!("{what}: {message}")),
        _ => PyValueError::new_err(format!("{what} failed")),
    }
}

/// What a generator emits, held as the JSON it was written as: the runtime is
/// what parses it, so there is no second reader here to disagree with it.
#[pyclass(name = "Artifact")]
#[derive(Clone)]
struct PyArtifact {
    json: Vec<u8>,
}

#[pymethods]
impl PyArtifact {
    #[new]
    fn new(json: &str) -> PyResult<Self> {
        Ok(Self { json: json.as_bytes().to_vec() })
    }
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

/// A compiled artifact and the memory it runs in. Compiling is the expensive
/// step, so build one and call it as often as you like.
#[pyclass(name = "Base", unsendable)]
struct PyBase {
    handle: *mut base::Base,
}

impl Drop for PyBase {
    fn drop(&mut self) {
        unsafe { capi::base_free(self.handle) };
    }
}

#[pymethods]
impl PyBase {
    #[new]
    fn new(artifact: &PyArtifact) -> PyResult<Self> {
        let handle = unsafe { capi::base_new(artifact.json.as_ptr(), artifact.json.len()) };
        if handle.is_null() {
            return Err(last_error("Base"));
        }
        Ok(Self { handle })
    }

    /// Call the entry point at `fn_idx`, returning the status it answered.
    ///
    /// Which index is which stage is the artifact generator's knowledge, so a
    /// caller names them itself. The status is `0` unless the entry's body
    /// ends in a `return` carrying a value.
    #[pyo3(signature = (fn_idx, data=None))]
    fn execute(&mut self, py: Python<'_>, fn_idx: u32, data: Option<&[u8]>) -> PyResult<i64> {
        self.execute_into(py, fn_idx, data.unwrap_or(&[]), None)
    }

    /// Call the entry point at `fn_idx`, which answers in `out`.
    ///
    /// Both buffers are the caller's own memory, handed to the program as
    /// pointers: nothing is copied in or out. Returns the status, as
    /// `execute` does.
    #[pyo3(signature = (fn_idx, data, out=None))]
    fn execute_into(
        &mut self,
        py: Python<'_>,
        fn_idx: u32,
        data: &[u8],
        out: Option<&Bound<'_, pyo3::types::PyByteArray>>,
    ) -> PyResult<i64> {
        let (out_ptr, out_len) = match out {
            Some(out) => (out.data(), out.len()),
            None => (std::ptr::null_mut(), 0),
        };
        let handle = self.handle;
        let mut status = 0i64;
        let rc = allow_threads_unsafe(py, || unsafe {
            capi::base_execute(
                handle, fn_idx, data.as_ptr(), data.len(), out_ptr, out_len, &mut status,
            )
        });
        if rc != 0 {
            return Err(last_error("execute"));
        }
        Ok(status)
    }

    /// `length` bytes of the program's memory from `offset`, copied.
    ///
    /// This is how a host reads what a program left behind, at an address its
    /// generator says it wrote. A range past the end is an error rather than a
    /// short answer, so a truncated read cannot be mistaken for a result.
    fn read_memory(&self, offset: usize, length: usize) -> PyResult<Vec<u8>> {
        let mut have = 0usize;
        let memory = unsafe { capi::base_memory(self.handle, &mut have) };
        if memory.is_null() || offset > have || length > have - offset {
            return Err(PyValueError::new_err(format!(
                "{offset}..{} is outside the {have} bytes of memory",
                offset + length
            )));
        }
        Ok(unsafe { std::slice::from_raw_parts(memory.add(offset), length) }.to_vec())
    }

    /// How many bytes of memory this program runs in.
    fn memory_size(&self) -> usize {
        let mut len = 0usize;
        unsafe { capi::base_memory(self.handle, &mut len) };
        len
    }
}

/// Read an artifact from the JSON a generator wrote.
#[pyfunction]
fn load_artifact(path: &str) -> PyResult<PyArtifact> {
    let json = std::fs::read(path)
        .map_err(|e| PyValueError::new_err(format!("Cannot read {}: {}", path, e)))?;
    Ok(PyArtifact { json })
}

/// Compile an artifact and call one of its entry points, once.
#[pyfunction]
#[pyo3(signature = (artifact, fn_idx, data=None))]
fn run(
    py: Python<'_>,
    artifact: &PyArtifact,
    fn_idx: u32,
    data: Option<&[u8]>,
) -> PyResult<i64> {
    let mut base = PyBase::new(artifact)?;
    base.execute(py, fn_idx, data)
}

/// Whether this extension was compiled without optimisations.
///
/// `maturin develop` builds a debug extension unless told otherwise, and
/// nothing about the result says so: it imports and runs, about fifteen times
/// slower at everything. Measured on this machine, loading a 19 MB artifact
/// took 812 ms debug against 68 ms release. So say it once, at import.
fn warn_if_unoptimised(py: Python<'_>) -> PyResult<()> {
    if !cfg!(debug_assertions) {
        return Ok(());
    }
    let warnings = PyModule::import_bound(py, "warnings")?;
    warnings.call_method1(
        "warn",
        ("py_base was built without optimisations and is roughly 15x slower \
          than it should be; rebuild with `maturin develop --release`",),
    )?;
    Ok(())
}

#[pymodule]
fn py_base(m: &Bound<'_, PyModule>) -> PyResult<()> {
    warn_if_unoptimised(m.py())?;
    m.add_class::<PyArtifact>()?;
    m.add_class::<PyBase>()?;
    m.add_function(wrap_pyfunction!(load_artifact, m)?)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(())
}
