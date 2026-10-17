//! Python over base's C ABI.
//!
//! Nothing here reaches into the runtime's Rust types: every call goes through
//! `base::capi`, the same functions a C or Lean host calls. That is what
//! keeps the three hosts on one core rather than three surfaces that drift —
//! the way `execute` answering Arrow on one side and bytes on another once did.
//!
//! It is the C API in Python's terms: a `Driver` loads an artifact, `function`
//! answers one of its functions as a `Function`, which is the C function
//! itself, and calling it calls the program. What this adds is Python's
//! conventions: an exception instead of a failure value, the buffer protocol
//! for the caller's input and output and for the arena, a `Function` holding
//! its driver alive, and the GIL released
//! while a program runs, so threads sharing one `Driver` run their calls at
//! once.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyMemoryView;

use base::capi;

/// What a generator emits, held as the bytes it was written as: the runtime is
/// what decodes them, so there is no second reader here to disagree with it.
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

/// A driver from `base_driver_load`, which the C API lets any number of
/// threads call and read through at once.
struct Handle(*mut base::Driver);

// SAFETY: the calls made through the handle are the ones the C API admits
// concurrently; the one that needs it alone, the free, runs in `Drop`, when
// Python holds no other reference and so no call is running.
unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}

/// A compiled artifact and the arena it runs in. Compiling is the expensive
/// step, so build one, find its functions with `function`, and call them as
/// often as you like, from as many threads as you like.
#[pyclass(name = "Driver", frozen)]
struct PyDriver {
    handle: Handle,
}

impl Drop for PyDriver {
    fn drop(&mut self) {
        unsafe { capi::base_driver_free(self.handle.0) };
    }
}

/// A function of a `Driver`, as the C function it is. It holds its driver,
/// so the driver lives as long as any `Function` of it does.
#[pyclass(name = "Function", frozen)]
struct PyFunction {
    _driver: Py<PyDriver>,
    f: base::FunctionPtr,
}

#[pymethods]
impl PyFunction {
    /// Run the function with `input`, answering in `output`, and return the
    /// status it answered: `0` unless the function's body ends in a `return`
    /// carrying a value.
    ///
    /// Both buffers are the caller's own memory, handed to the program as
    /// pointers: nothing is copied in or out. The GIL is released while the
    /// program runs, so calls from several threads run at once.
    ///
    /// `output` is any writable contiguous buffer (a `bytearray`, a NumPy
    /// array, a `memoryview` of one). It is held as an exported buffer for the
    /// whole call, so another thread cannot resize or free it under the
    /// program: Python refuses with `BufferError` instead.
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

    /// The function named `name`, as the C function it is. Find it once and
    /// call it: the lookup is the only part that is not the program.
    fn function(slf: &Bound<'_, Self>, name: &str) -> PyResult<PyFunction> {
        let handle = slf.get().handle.0;
        let f = unsafe { capi::base_driver_function(handle, name.as_ptr(), name.len()) }
            .ok_or_else(|| PyValueError::new_err(format!("the artifact has no function named {name:?}")))?;
        Ok(PyFunction { _driver: slf.clone().unbind(), f })
    }

    /// `length` bytes of the arena from `offset`, copied.
    ///
    /// This is how a host reads what a program left behind, at an address its
    /// generator says it wrote, including while other threads' calls run. A
    /// range past the end is an error rather than a short answer, so a
    /// truncated read cannot be mistaken for a result.
    fn read<'py>(
        &self,
        py: Python<'py>,
        offset: usize,
        length: usize,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        // Refused before allocating, so a length nothing could hold is an
        // exception rather than an allocation failure.
        let mut have = 0usize;
        let arena = unsafe { capi::base_driver_arena(self.handle.0, &mut have) };
        if offset.checked_add(length).is_none_or(|end| end > have) {
            return Err(PyValueError::new_err(format!(
                "{offset}..{offset}+{length} is outside the {have} bytes of the arena"
            )));
        }
        // Other threads' calls may be writing the arena, so it is read as they
        // write it: each aligned 8-byte word with one atomic load, so a word a
        // program updates atomically is never seen torn, and the bytes either
        // side one at a time.
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

    /// Sleep while the 32-bit word at `offset` of the arena holds
    /// `expected`, until a wake on it, `timeout_ns` nanoseconds (none if
    /// negative), or for no reason; the GIL is released meanwhile. It is the
    /// system's own wait, so programs and Python threads meet at the word.
    /// Loop on the condition.
    #[pyo3(signature = (offset, expected, timeout_ns=-1))]
    fn wait(&self, py: Python<'_>, offset: usize, expected: u32, timeout_ns: i64) -> PyResult<()> {
        let word = self.word(offset)? as usize;
        // SAFETY: a word of the arena, which `self` keeps alive, reached
        // atomically by every thread.
        allow_threads_unsafe(py, || base::wait(unsafe { AtomicU32::from_ptr(word as *mut u32) }, expected, timeout_ns));
        Ok(())
    }

    /// Wake one thread waiting on the word at `offset` when `count` is 1, every
    /// one for any other `count` but 0: a program's and Python's alike.
    #[pyo3(signature = (offset, count=-1))]
    fn wake(&self, offset: usize, count: i32) -> PyResult<()> {
        let word = self.word(offset)?;
        // SAFETY: as in `wait`.
        base::wake(unsafe { AtomicU32::from_ptr(word) }, count);
        Ok(())
    }

    /// The arena itself, as a `memoryview`: no copy, and writable, so NumPy
    /// and anything else that takes a buffer reads and writes it in place. It
    /// holds this driver, so the view stays valid however long it is kept, and
    /// `len(driver.arena)` is its size.
    ///
    /// While calls run, a program may be writing it, and a plain view reads
    /// without atomics: reach bytes a running program may touch with `read`,
    /// and the rest, or all of it between calls, through the view.
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

/// Nanoseconds on the system's monotonic clock, the one programs read.
#[pyfunction]
fn now() -> i64 {
    base::now()
}

/// Read an artifact from the `.cbor` file a generator wrote.
#[pyfunction]
fn load_artifact(path: &str) -> PyResult<PyArtifact> {
    let bytes = std::fs::read(path)
        .map_err(|e| PyValueError::new_err(format!("Cannot read {}: {}", path, e)))?;
    Ok(PyArtifact { bytes })
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
    m.add_class::<PyDriver>()?;
    m.add_class::<PyFunction>()?;
    m.add_function(wrap_pyfunction!(load_artifact, m)?)?;
    m.add_function(wrap_pyfunction!(now, m)?)?;
    Ok(())
}
