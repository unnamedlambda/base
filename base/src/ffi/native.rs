//! Machine code a program carries as data, loaded at run time and called
//! directly (`Callee::Native`), for instructions CLIF cannot spell.
//!
//! A body keeps one convention per architecture, whatever the OS:
//!
//! * x86-64: SysV (rdi, rsi, rdx, rcx in; rax out; rbx, rbp, r12–r15, rsp
//!   preserved), on Windows too.
//! * AArch64: AAPCS64 (x0–x3 in; x0 out; x19–x30, sp preserved; x18 untouched).
//!
//! It is position-independent, calls nothing and makes no system call. None of
//! this is checked here.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

/// A mapping `cl_native_load` made.
struct Mapping(#[allow(dead_code)] region::Allocation);

// SAFETY: the allocation is owned memory addressed by a raw pointer; nothing
// here dereferences it after loading or reads it through a shared reference,
// and dropping it (unmapping) from any thread is what the OS calls allow.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

/// Live mappings by address; a free of any other address is refused.
fn loaded() -> &'static Mutex<BTreeMap<usize, Mapping>> {
    static LOADED: OnceLock<Mutex<BTreeMap<usize, Mapping>>> = OnceLock::new();
    LOADED.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Copy `len` bytes at `src` into fresh W^X executable memory; answers the
/// address, or 0 on failure.
pub(crate) unsafe extern "C" fn cl_native_load(src: *const u8, len: i64) -> i64 {
    if src.is_null() || len <= 0 {
        return 0;
    }
    let len = len as usize;
    let mut alloc = match region::alloc(len, region::Protection::READ_WRITE) {
        Ok(a) => a,
        Err(_) => return 0,
    };
    let dst = alloc.as_mut_ptr::<u8>();
    std::ptr::copy_nonoverlapping(src, dst, len);
    if wasmtime_internal_jit_icache_coherence::clear_cache(dst as *const _, len).is_err() {
        return 0;
    }
    if region::protect(dst, alloc.len(), region::Protection::READ_EXECUTE).is_err() {
        return 0;
    }
    if wasmtime_internal_jit_icache_coherence::pipeline_flush_mt().is_err() {
        return 0;
    }
    let addr = dst as usize;
    loaded().lock().unwrap().insert(addr, Mapping(alloc));
    addr as i64
}

/// Unmap what `cl_native_load` answered `addr` for. Answers 0, or -1 if `addr`
/// is not one it answered (or was already freed).
pub(crate) unsafe extern "C" fn cl_native_free(addr: i64) -> i32 {
    match loaded().lock().unwrap().remove(&(addr as usize)) {
        Some(_) => 0,
        None => -1,
    }
}

/// Which architecture's code this host runs: 1 for x86-64, 2 for AArch64,
/// 0 for any other.
pub(crate) unsafe extern "C" fn cl_native_arch() -> i32 {
    if cfg!(target_arch = "x86_64") {
        1
    } else if cfg!(target_arch = "aarch64") {
        2
    } else {
        0
    }
}

/// 1 if the CPU has the feature `name` (Rust's spelling), 0 if not, -1 for a
/// name unknown here, so a typo is not mistaken for "absent".
pub(crate) unsafe extern "C" fn cl_cpu_has(name: *const u8) -> i32 {
    let name = super::read_cstr_ptr(name);
    match cpu_has(&name) {
        Some(true) => 1,
        Some(false) => 0,
        None => -1,
    }
}

fn cpu_has(name: &str) -> Option<bool> {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::is_x86_feature_detected as has;
        Some(match name {
            "sse2" => has!("sse2"),
            "sse3" => has!("sse3"),
            "ssse3" => has!("ssse3"),
            "sse4.1" => has!("sse4.1"),
            "sse4.2" => has!("sse4.2"),
            "popcnt" => has!("popcnt"),
            "lzcnt" => has!("lzcnt"),
            "bmi1" => has!("bmi1"),
            "bmi2" => has!("bmi2"),
            "avx" => has!("avx"),
            "avx2" => has!("avx2"),
            "fma" => has!("fma"),
            "f16c" => has!("f16c"),
            "aes" => has!("aes"),
            "pclmulqdq" => has!("pclmulqdq"),
            "sha" => has!("sha"),
            "avx512f" => has!("avx512f"),
            "avx512bw" => has!("avx512bw"),
            "avx512vl" => has!("avx512vl"),
            _ => return None,
        })
    }
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::is_aarch64_feature_detected as has;
        Some(match name {
            "neon" => has!("neon"),
            "aes" => has!("aes"),
            "sha2" => has!("sha2"),
            "sha3" => has!("sha3"),
            "crc" => has!("crc"),
            "dotprod" => has!("dotprod"),
            "sve" => has!("sve"),
            "sve2" => has!("sve2"),
            _ => return None,
        })
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = name;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `mov rax, rdi; add rax, rsi; ret` -- a + b under SysV.
    #[cfg(target_arch = "x86_64")]
    const ADD: &[u8] = &[0x48, 0x89, 0xf8, 0x48, 0x01, 0xf0, 0xc3];

    /// `add x0, x0, x1; ret`
    #[cfg(target_arch = "aarch64")]
    const ADD: &[u8] = &[0x00, 0x00, 0x01, 0x8b, 0xc0, 0x03, 0x5f, 0xd6];

    /// Called here as a program's native call calls it: under the one
    /// convention of the architecture.
    #[cfg(target_arch = "x86_64")]
    type Body = unsafe extern "sysv64" fn(i64, i64, i64, i64) -> i64;
    #[cfg(target_arch = "aarch64")]
    type Body = unsafe extern "C" fn(i64, i64, i64, i64) -> i64;

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn loaded_code_runs_and_answers() {
        unsafe {
            let f = cl_native_load(ADD.as_ptr(), ADD.len() as i64);
            assert_ne!(f, 0);
            let body: Body = std::mem::transmute(f as usize);
            assert_eq!(body(40, 2, 0, 0), 42);
            assert_eq!(cl_native_free(f), 0);
            // not freeable twice
            assert_eq!(cl_native_free(f), -1);
        }
    }

    /// Only what was loaded can be freed.
    #[test]
    fn an_address_not_loaded_is_not_freed() {
        let not_code = [0u8; 16];
        assert_eq!(unsafe { cl_native_free(not_code.as_ptr() as i64) }, -1);
    }

    #[test]
    fn nothing_to_load_answers_zero() {
        unsafe {
            assert_eq!(cl_native_load(std::ptr::null(), 8), 0);
            assert_eq!(cl_native_load([0xc3u8].as_ptr(), 0), 0);
        }
    }

    #[test]
    fn a_feature_name_this_runtime_does_not_know_is_not_answered_no() {
        assert_eq!(unsafe { cl_cpu_has(b"no-such-feature\0".as_ptr()) }, -1);
        #[cfg(target_arch = "x86_64")]
        assert_eq!(unsafe { cl_cpu_has(b"sse2\0".as_ptr()) }, 1);
    }
}
