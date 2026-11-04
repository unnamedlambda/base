//! base's C API in Python's terms, through `base::c_api` only: exceptions for
//! failures, buffers for input, output and the arena, and the GIL released
//! while a program runs.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyMemoryView;

use base::c_api;

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
        unsafe { c_api::base_driver_free(self.handle.0) };
    }
}

/// A function of a `Driver`, which it keeps alive.
#[pyclass(name = "Function", frozen)]
struct PyFunction {
    _driver: Py<PyDriver>,
    f: base::BaseFunction,
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
    /// Loads an encoded artifact: `Driver(artifact)`, as `ctypes.CDLL(path)`
    /// loads a library.
    #[new]
    fn load(artifact: &[u8]) -> PyResult<Self> {
        let mut err = [0u8; 1024];
        let handle = unsafe { c_api::base_driver_load(artifact.as_ptr(), artifact.len(), err.as_mut_ptr(), err.len()) };
        if handle.is_null() {
            let end = err.iter().position(|&b| b == 0).unwrap_or(err.len());
            return Err(PyValueError::new_err(format!("load: {}", String::from_utf8_lossy(&err[..end]))));
        }
        Ok(Self { handle: Handle(handle) })
    }

    /// The function named `name`.
    fn function(slf: &Bound<'_, Self>, name: &str) -> PyResult<PyFunction> {
        let handle = slf.get().handle.0;
        let f = unsafe { c_api::base_driver_function(handle, name.as_ptr(), name.len()) }
            .ok_or_else(|| PyValueError::new_err(format!("the artifact has no function named {name:?}")))?;
        Ok(PyFunction { _driver: slf.clone().unbind(), f })
    }

    /// The arena itself as a writable `memoryview`, keeping the driver alive.
    /// While calls run, bytes a program may touch are the program's.
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
        let ptr = c_api::base_driver_arena(slf.get().handle.0, &mut len);
        // An empty arena has no address; a view of nothing still needs one.
        let ptr = if ptr.is_null() { std::ptr::NonNull::<u8>::dangling().as_ptr() } else { ptr };
        // The view holds a reference to the driver, so the arena outlives it.
        if pyo3::ffi::PyBuffer_FillInfo(view, slf.as_ptr(), ptr.cast(), len as isize, 0, flags) == -1 {
            return Err(PyErr::fetch(slf.py()));
        }
        Ok(())
    }
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
    m.add_class::<PyDriver>()?;
    m.add_class::<PyFunction>()?;
    Ok(())
}
