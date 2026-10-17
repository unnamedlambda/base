//! base's C API in Python's terms, through `base::capi` only: exceptions for
//! failures, buffers for input, output and the arena, and the GIL released
//! while a program runs.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyMemoryView;

use base::capi;

/// An encoded artifact.
#[pyclass(name = "Artifact")]
#[derive(Clone)]
struct PyArtifact {
    bytes: Vec<u8>,
}

#[pymethods]
impl PyArtifact {
    #[new]
    fn new(bytes: &[u8]) -> PyResult<Self> {
        Ok(Self { bytes: bytes.to_vec() })
    }
}

/// A closure run without the GIL; it must touch no Python object.
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

/// A driver from `base_driver_load`.
struct Handle(*mut base::Driver);

// SAFETY: the C API admits these calls from any thread; the free runs in
// `Drop`, when nothing else holds the driver.
unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}

/// A compiled artifact and its arena. Load once; call its functions from any thread.
#[pyclass(name = "Driver", frozen)]
struct PyDriver {
    handle: Handle,
}

impl Drop for PyDriver {
    fn drop(&mut self) {
        unsafe { capi::base_driver_free(self.handle.0) };
    }
}

/// A function of a `Driver`, which it keeps alive.
#[pyclass(name = "Function", frozen)]
struct PyFunction {
    _driver: Py<PyDriver>,
    f: base::FunctionPtr,
}

#[pymethods]
impl PyFunction {
    /// Runs the function on `input` (bytes) and `output` (any writable
    /// contiguous buffer), in place, and returns its status. The GIL is
    /// released meanwhile; `output` cannot be resized during the call.
    #[pyo3(signature = (input=None, output=None))]
    fn __call__(
        &self,
        py: Python<'_>,
        input: Option<&[u8]>,
        output: Option<PyBuffer<u8>>,
    ) -> PyResult<i64> {
        let input = input.unwrap_or(&[]);
        let (output_ptr, output_len) = match &output {
            Some(output) => {
                if output.readonly() || !output.is_c_contiguous() {
                    return Err(PyValueError::new_err("output must be a writable contiguous buffer"));
                }
                (output.buf_ptr() as *mut u8, output.len_bytes())
            }
            None => (std::ptr::null_mut(), 0),
        };
        let f = self.f;
        let input_ptr = input.as_ptr() as usize;
        let input_len = input.len();
        let output_ptr = output_ptr as usize;
        let status = allow_threads_unsafe(py, || unsafe {
            f(input_ptr as *const u8, input_len, output_ptr as *mut u8, output_len)
        });
        // Released only now that the program is done with it.
        drop(output);
        Ok(status)
    }
}

#[pymethods]
impl PyDriver {
    #[new]
    fn new(artifact: &PyArtifact) -> PyResult<Self> {
        let mut err = [0u8; 1024];
        let handle = unsafe {
            capi::base_driver_load(artifact.bytes.as_ptr(), artifact.bytes.len(), err.as_mut_ptr(), err.len())
        };
        if handle.is_null() {
            let end = err.iter().position(|&b| b == 0).unwrap_or(err.len());
            return Err(PyValueError::new_err(format!("Driver: {}", String::from_utf8_lossy(&err[..end]))));
        }
        Ok(Self { handle: Handle(handle) })
    }

    /// The function named `name`.
    fn function(slf: &Bound<'_, Self>, name: &str) -> PyResult<PyFunction> {
        let handle = slf.get().handle.0;
        let f = unsafe { capi::base_driver_function(handle, name.as_ptr(), name.len()) }
            .ok_or_else(|| PyValueError::new_err(format!("the artifact has no function named {name:?}")))?;
        Ok(PyFunction { _driver: slf.clone().unbind(), f })
    }

    /// `length` bytes of the arena from `offset`, copied; safe while calls run.
    fn read<'py>(
        &self,
        py: Python<'py>,
        offset: usize,
        length: usize,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let mut have = 0usize;
        let arena = unsafe { capi::base_driver_arena(self.handle.0, &mut have) };
        if offset.checked_add(length).is_none_or(|end| end > have) {
            return Err(PyValueError::new_err(format!(
                "{offset}..{offset}+{length} is outside the {have} bytes of the arena"
            )));
        }
        // Each aligned word with one atomic load, so none is read torn.
        let mut bytes = vec![0u8; length];
        let mut i = 0;
        while i < length {
            // SAFETY: in bounds, checked above; the arena lives as long as the
            // driver, which `self` holds.
            let at = unsafe { arena.add(offset + i) };
            if at as usize % 8 == 0 && length - i >= 8 {
                let word = unsafe { AtomicU64::from_ptr(at.cast()) }.load(Ordering::Relaxed);
                bytes[i..i + 8].copy_from_slice(&word.to_ne_bytes());
                i += 8;
            } else {
                bytes[i] = unsafe { AtomicU8::from_ptr(at) }.load(Ordering::Relaxed);
                i += 1;
            }
        }
        // `bytes`, not the list of ints a `Vec<u8>` would convert to.
        Ok(pyo3::types::PyBytes::new_bound(py, &bytes))
    }

    /// Sleeps while the word at `offset` holds `expected`, until woken,
    /// `timeout_ns` passes (never if negative), or spuriously.
    #[pyo3(signature = (offset, expected, timeout_ns=-1))]
    fn wait(&self, py: Python<'_>, offset: usize, expected: u32, timeout_ns: i64) -> PyResult<()> {
        let word = self.word(offset)? as usize;
        // SAFETY: a word of the arena, which `self` keeps alive, reached
        // atomically by every thread.
        allow_threads_unsafe(py, || base::wait(unsafe { AtomicU32::from_ptr(word as *mut u32) }, expected, timeout_ns));
        Ok(())
    }

    /// Wakes one waiter on the word at `offset` when `count` is 1, all otherwise.
    #[pyo3(signature = (offset, count=-1))]
    fn wake(&self, offset: usize, count: i32) -> PyResult<()> {
        let word = self.word(offset)?;
        // SAFETY: as in `wait`.
        base::wake(unsafe { AtomicU32::from_ptr(word) }, count);
        Ok(())
    }

    /// The arena itself as a writable `memoryview`, keeping the driver alive.
    /// While calls run, read bytes a program may touch with `read`.
    #[getter]
    fn arena<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyMemoryView>> {
        PyMemoryView::from_bound(slf.as_any())
    }

    /// The buffer protocol over the arena, which `arena` is a view of.
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        let mut len = 0usize;
        let ptr = capi::base_driver_arena(slf.get().handle.0, &mut len);
        // An empty arena has no address; a view of nothing still needs one.
        let ptr = if ptr.is_null() { std::ptr::NonNull::<u8>::dangling().as_ptr() } else { ptr };
        // The view holds a reference to the driver, so the arena outlives it.
        if pyo3::ffi::PyBuffer_FillInfo(view, slf.as_ptr(), ptr.cast(), len as isize, 0, flags) == -1 {
            return Err(PyErr::fetch(slf.py()));
        }
        Ok(())
    }
}

impl PyDriver {
    /// The word at `offset`: in the arena and 4-byte aligned, or an error.
    fn word(&self, offset: usize) -> PyResult<*mut u32> {
        let mut have = 0usize;
        let arena = unsafe { capi::base_driver_arena(self.handle.0, &mut have) };
        if offset % 4 != 0 || offset.checked_add(4).is_none_or(|end| end > have) {
            return Err(PyValueError::new_err(format!(
                "{offset} is not a 4-byte-aligned word of the {have} bytes of the arena"
            )));
        }
        Ok(unsafe { arena.add(offset) }.cast())
    }
}

/// Nanoseconds on the monotonic clock programs read.
#[pyfunction]
fn now() -> i64 {
    base::now()
}

/// The artifact in a `.cbor` file.
#[pyfunction]
fn load_artifact(path: &str) -> PyResult<PyArtifact> {
    let bytes = std::fs::read(path)
        .map_err(|e| PyValueError::new_err(format!("Cannot read {}: {}", path, e)))?;
    Ok(PyArtifact { bytes })
}

/// Warns at import when built without optimisations, which runs ~15x slower.
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
    m.add_class::<PyDriver>()?;
    m.add_class::<PyFunction>()?;
    m.add_function(wrap_pyfunction!(load_artifact, m)?)?;
    m.add_function(wrap_pyfunction!(now, m)?)?;
    Ok(())
}
