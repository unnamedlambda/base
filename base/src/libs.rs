//! The C libraries a program calls directly.
//!
//! A program names each function with its library, the library's files per
//! operating system, the symbol and the signature (`ExternFn`). This binds the
//! symbol to an address. The C library, wgpu and the window, CPU, serial and
//! USB libraries are built into the engine, so their symbols bind to the
//! linked-in functions. Any other library is opened the first time any
//! program names it and stays open for the life of the process, so an address
//! handed out is never unmapped under code that calls it. What is not there
//! binds to a stub, so the program still loads and can ask whether the
//! library is present.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use base_types::clif::ExternFn;

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

/// The library `f` names, opened on first use from the files this operating
/// system names it with, in order.
fn library(f: &ExternFn) -> Option<&'static libloading::Library> {
    let mut libs = opened().lock().unwrap_or_else(|e| e.into_inner());
    *libs.entry(f.lib.clone()).or_insert_with(|| {
        let os = std::env::consts::OS;
        let files = f.files.iter().find(|(o, _)| o == os).map(|(_, fs)| fs.as_slice());
        files.unwrap_or(&[]).iter().find_map(|file| {
            // SAFETY: opening a library runs its initialisers. The libraries
            // a program may name are the portable C APIs `Lib` lists, whose
            // initialisers have no preconditions on the caller.
            let lib = unsafe { libloading::Library::new(file) }.ok()?;
            Some(&*Box::leak(Box::new(lib)))
        })
    })
}

/// A library built into the engine: how to find its symbols.
fn builtin(lib: &str) -> Option<fn(&str) -> Option<usize>> {
    match lib {
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
pub(crate) fn bind(f: &ExternFn) -> usize {
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
pub(crate) fn link_name(f: &ExternFn) -> String {
    format!("extern:{}:{}", f.lib, f.symbol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base_types::clif::ClifTy;

    fn missing(symbol: &str) -> ExternFn {
        ExternFn {
            lib: "base-test-missing".into(),
            files: vec![(std::env::consts::OS.into(), vec!["libbase-test-missing.so.0".into()])],
            symbol: symbol.into(),
            params: vec![ClifTy::I64],
            result: Some(ClifTy::I64),
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
