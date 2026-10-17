//! The C libraries a program calls directly.
//!
//! A program declares each primitive by its library, its symbol and its
//! signature ([`Primitive`]), and this binds the symbol to an address. The engine's own
//! functions (`base`), the C library, wgpu and the window, CPU, serial and USB
//! libraries are built into it, so their symbols bind to the linked-in
//! functions. The others are the system libraries this engine knows how to
//! find, each under the file names its vendor gives it on each operating
//! system ([`files`]), listing only releases the corpora have run against,
//! newest first: which file a library is, is how the system packages it, not
//! what a program means, so a new release is a change here and to no
//! artifact. Such a library is opened the first time any program names it and
//! stays open for the life of the process, so an address handed out is never
//! unmapped under code that calls it. A library this engine does not know, or
//! one not installed, binds to a stub, so the program still loads and can ask
//! whether the library is present.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use base_types::clif::Primitive;

/// What every function of a missing library, or a missing symbol, answers.
/// Returning an integer in the return register is well defined whatever the
/// callee's parameters, because every supported convention has the caller
/// pop its own arguments.
extern "C" fn absent() -> i64 {
    -1
}

extern "C" fn present_yes() -> i32 {
    1
}

extern "C" fn present_no() -> i32 {
    0
}

/// Opened libraries by name, `None` for one none of whose files opened.
fn opened() -> &'static Mutex<HashMap<String, Option<&'static libloading::Library>>> {
    static LIBS: OnceLock<Mutex<HashMap<String, Option<&'static libloading::Library>>>> =
        OnceLock::new();
    LIBS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The files the system library `lib` is on this operating system, newest
/// release first; none for a library this engine does not open.
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

/// The library `f` names, opened on first use from its files, in order.
fn library(f: &Primitive) -> Option<&'static libloading::Library> {
    let mut libs = opened().lock().unwrap_or_else(|e| e.into_inner());
    *libs.entry(f.lib.clone()).or_insert_with(|| {
        files(&f.lib).iter().find_map(|file| {
            // SAFETY: opening a library runs its initialisers. The files are
            // the system libraries listed above, whose initialisers have no
            // preconditions on the caller; an artifact cannot name another.
            let lib = unsafe { libloading::Library::new(file) }.ok()?;
            Some(&*Box::leak(Box::new(lib)))
        })
    })
}

/// A library built into the engine: how to find its symbols.
fn builtin(lib: &str) -> Option<fn(&str) -> Option<usize>> {
    match lib {
        "base" => Some(crate::ffi::linked),
        "c" => Some(crate::ffi::libc::linked),
        "wgpu" => Some(crate::ffi::wgpu::linked),
        "window" => Some(crate::ffi::window::linked),
        "cpu" => Some(crate::ffi::cpu::linked),
        "serial" => Some(crate::ffi::serial::linked),
        "usb" => Some(crate::ffi::usb::linked),
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

/// The name a bound function is declared under in the JIT module: unique per
/// library and symbol, and not a name anything else can declare.
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
