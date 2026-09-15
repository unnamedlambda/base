use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use base_types::{Algorithm, Artifact, Setup};

#[pyclass(name = "Setup")]
struct PySetup {
    inner: Setup,
}

#[pymethods]
impl PySetup {
    #[new]
    fn new(json: &str) -> PyResult<Self> {
        let inner: Setup = serde_json::from_str(json)
            .map_err(|e| PyValueError::new_err(format!("Invalid Setup JSON: {}", e)))?;
        Ok(Self { inner })
    }
}

#[pyclass(name = "Algorithm")]
#[derive(Clone)]
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

#[pyclass(name = "Artifact")]
struct PyArtifact {
    inner: Artifact,
}

#[pymethods]
impl PyArtifact {
    #[getter]
    fn setup(&self) -> PySetup {
        PySetup {
            inner: self.inner.setup.clone(),
        }
    }

    #[getter]
    fn main(&self) -> PyAlgorithm {
        PyAlgorithm {
            inner: self.inner.main.clone(),
        }
    }

    #[getter]
    fn extras(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new_bound(py);
        for (name, alg) in &self.inner.extras {
            let py_alg = Py::new(
                py,
                PyAlgorithm {
                    inner: alg.clone(),
                },
            )?;
            dict.set_item(name, py_alg)?;
        }
        Ok(dict.into())
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

/// Base execution engine. JIT compiles once, executes many times.
/// Releases the GIL during execution so other Python threads can run.
#[pyclass(name = "Base")]
struct PyBase {
    inner: base::Base,
}

#[pymethods]
impl PyBase {
    #[new]
    fn new(setup: &PySetup) -> PyResult<Self> {
        let inner = base::Base::new(setup.inner.clone())
            .map_err(|e| PyValueError::new_err(format!("Base::new failed: {:?}", e)))?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (algorithm, data=None))]
    fn execute(
        &mut self,
        py: Python<'_>,
        algorithm: &PyAlgorithm,
        data: Option<&[u8]>,
    ) -> PyResult<()> {
        let data = data.unwrap_or(&[]);
        allow_threads_unsafe(py, || self.inner.execute(&algorithm.inner, data))
            .map_err(|e| PyValueError::new_err(format!("execute failed: {:?}", e)))
    }

    fn execute_into(
        &mut self,
        py: Python<'_>,
        algorithm: &PyAlgorithm,
        data: &[u8],
        out: &Bound<'_, pyo3::types::PyByteArray>,
    ) -> PyResult<()> {
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out.data() as *mut u8, out.len()) };
        allow_threads_unsafe(py, || {
            self.inner.execute_into(&algorithm.inner, data, out_slice)
        })
        .map_err(|e| PyValueError::new_err(format!("execute_into failed: {:?}", e)))
    }
}

/// Read and deserialize an Artifact from a JSON file.
/// Returns an Artifact object exposing `.setup`, `.main`, and `.extras`.
#[pyfunction]
fn load_artifact(path: &str) -> PyResult<PyArtifact> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| PyValueError::new_err(format!("Cannot read {}: {}", path, e)))?;
    let inner: Artifact = serde_json::from_str(&text)
        .map_err(|e| PyValueError::new_err(format!("Invalid artifact JSON in {}: {}", path, e)))?;
    Ok(PyArtifact { inner })
}

/// One-shot execution: JIT compile and execute in a single call.
#[pyfunction]
fn run(py: Python<'_>, setup: &PySetup, algorithm: &PyAlgorithm) -> PyResult<()> {
    let setup = setup.inner.clone();
    let algorithm = algorithm.inner.clone();
    allow_threads_unsafe(py, || base::run(setup, algorithm))
        .map_err(|e| PyValueError::new_err(format!("run failed: {:?}", e)))
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
    m.add_class::<PySetup>()?;
    m.add_class::<PyAlgorithm>()?;
    m.add_class::<PyArtifact>()?;
    m.add_class::<PyBase>()?;
    m.add_function(wrap_pyfunction!(load_artifact, m)?)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(())
}
