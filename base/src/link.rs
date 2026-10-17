//! Binding each primitive to an address: a library built into the engine, or
//! a system library opened once for the process's life. A missing library or
//! symbol binds to a stub, so the program loads and can ask what is present.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use base_types::clif::Primitive;

/// A missing function: `-1`, whatever it was called with (the caller pops).
extern "C" fn absent() -> i64 {
    -1
}

extern "C" fn present_yes() -> i32 {
    1
}

extern "C" fn present_no() -> i32 {
    0
}

/// Opened libraries by name; `None` when none of its files opened.
fn opened() -> &'static Mutex<HashMap<String, Option<&'static libloading::Library>>> {
    static LIBS: OnceLock<Mutex<HashMap<String, Option<&'static libloading::Library>>>> =
        OnceLock::new();
    LIBS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The files `lib` is on this system, newest first.
fn files(lib: &str) -> &'static [&'static str] {
    match lib {
        #[cfg(target_os = "linux")]
        "cuda" => &["libcuda.so.1"],
        #[cfg(target_os = "linux")]
        "cublas" => &["libcublas.so.12"],
        #[cfg(windows)]
        "cuda" => &["nvcuda.dll"],
        #[cfg(windows)]
        "cublas" => &["cublas64_12.dll"],
        _ => &[],
    }
}

/// The library `f` names, opened on first use.
fn library(f: &Primitive) -> Option<&'static libloading::Library> {
    let mut libs = opened().lock().unwrap_or_else(|e| e.into_inner());
    *libs.entry(f.lib.clone()).or_insert_with(|| {
        files(&f.lib).iter().find_map(|file| {
            // SAFETY: opening a library runs its initialisers. The files are
            // the system libraries above, whose initialisers ask nothing.
            let lib = unsafe { libloading::Library::new(file) }.ok()?;
            Some(&*Box::leak(Box::new(lib)))
        })
    })
}

/// A library built into the engine.
fn builtin(lib: &str) -> Option<fn(&str) -> Option<usize>> {
    match lib {
        "base" => Some(crate::builtin::linked),
        "c" => Some(crate::builtin::libc::linked),
        "wgpu" => Some(crate::builtin::wgpu::linked),
        "window" => Some(crate::builtin::window::linked),
        "cpu" => Some(crate::builtin::cpu::linked),
        "serial" => Some(crate::builtin::serial::linked),
        "usb" => Some(crate::builtin::usb::linked),
        _ => None,
    }
}

/// The address a call to `f` jumps to.
pub(crate) fn bind(f: &Primitive) -> usize {
    if let Some(find) = builtin(&f.lib) {
        return if f.symbol.is_empty() { present_yes as usize } else { find(&f.symbol).unwrap_or(absent as usize) };
    }
    let lib = library(f);
    if f.symbol.is_empty() {
        return if lib.is_some() { present_yes as usize } else { present_no as usize };
    }
    let Some(lib) = lib else { return absent as usize };
    let mut name = f.symbol.clone().into_bytes();
    name.push(0);
    // SAFETY: the address is only ever called, at the signature the program
    // declares for the symbol, which is the library's own C declaration.
    match unsafe { lib.get::<unsafe extern "C" fn()>(&name) } {
        Ok(sym) => *sym as usize,
        Err(_) => absent as usize,
    }
}

/// The JIT-module name of a primitive, unique per library and symbol.
pub(crate) fn link_name(f: &Primitive) -> String {
    format!("extern:{}:{}", f.lib, f.symbol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base_types::clif::{ClifTy, Convention, Signature};

    fn missing(symbol: &str) -> Primitive {
        Primitive {
            lib: "base-test-missing".into(),
            symbol: symbol.into(),
            signature: Signature {
                convention: Convention::Host,
                params: vec![ClifTy::I64.into()],
                returns: vec![ClifTy::I64.into()],
            },
        }
    }

    #[test]
    fn a_missing_library_is_absent() {
        let probe: extern "C" fn() -> i32 = unsafe { std::mem::transmute(bind(&missing(""))) };
        assert_eq!(probe(), 0);
        let f: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(bind(&missing("anything"))) };
        assert_eq!(f(42), -1);
    }
}
