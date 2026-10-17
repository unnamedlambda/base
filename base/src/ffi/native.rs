//! Machine code a program carries, made executable and called at the
//! architecture's convention: System V on x86-64 (Windows too), AAPCS64 on
//! AArch64. It must be position-independent and make no calls.

/// What a free gives back to the system: the mapping `cl_native_load` made.
type Mapping = region::Allocation;

/// Bytes ahead of the code, holding the box its mapping is kept in, so a free
/// finds it from the code's own address. Sixteen keeps the code 16-aligned.
const HEADER: usize = 16;

/// Copy `len` bytes at `src` into fresh W^X executable memory; answers the
/// address, or 0 on failure.
pub(crate) unsafe extern "C" fn cl_native_load(src: *const u8, len: i64) -> i64 {
    if src.is_null() || len <= 0 {
        return 0;
    }
    let Some(size) = usize::try_from(len).ok().and_then(|n| n.checked_add(HEADER)) else { return 0 };
    let mut alloc = match region::alloc(size, region::Protection::READ_WRITE) {
        Ok(a) => a,
        Err(_) => return 0,
    };
    let base = alloc.as_mut_ptr::<u8>();
    let code = base.add(HEADER);
    std::ptr::copy_nonoverlapping(src, code, size - HEADER);
    if wasmtime_internal_jit_icache_coherence::clear_cache(code as *const _, size - HEADER).is_err() {
        return 0;
    }
    let len = alloc.len();
    let kept = Box::into_raw(Box::new(alloc));
    base.cast::<*mut Mapping>().write(kept);
    if region::protect(base, len, region::Protection::READ_EXECUTE).is_err()
        || wasmtime_internal_jit_icache_coherence::pipeline_flush_mt().is_err()
    {
        drop(Box::from_raw(kept));
        return 0;
    }
    code as i64
}

/// Unmap what `cl_native_load` answered for `addr`.
///
/// # Safety
///
/// `addr` is an address `cl_native_load` answered, other than 0, not freed
/// already, and no thread is running the code at it.
pub(crate) unsafe extern "C" fn cl_native_free(addr: i64) {
    let header = (addr as *const u8).sub(HEADER);
    drop(Box::from_raw(header.cast::<*mut Mapping>().read()));
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

/// 1 if the CPU has the feature named in the `cap`-byte field at `name`
/// (Rust's spelling, [`super::read_field`]), 0 if not, -1 for a name unknown
/// here, so a typo is not mistaken for "absent".
pub(crate) unsafe extern "C" fn cl_cpu_has(name: *const u8, cap: i64) -> i32 {
    let Some(name) = super::read_field(name, cap) else { return -1 };
    match cpu_has(name) {
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
            assert_eq!(f % 16, 0, "code starts 16-aligned");
            cl_native_free(f);
        }
    }

    /// Mappings are independent: freeing one leaves another running.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn freeing_one_load_keeps_another() {
        unsafe {
            let (f, g) = (cl_native_load(ADD.as_ptr(), ADD.len() as i64), cl_native_load(ADD.as_ptr(), ADD.len() as i64));
            assert!(f != 0 && g != 0 && f != g);
            cl_native_free(f);
            let body: Body = std::mem::transmute(g as usize);
            assert_eq!(body(1, 2, 0, 0), 3);
            cl_native_free(g);
        }
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
        assert_eq!(unsafe { cl_cpu_has(b"no-such-feature\0".as_ptr(), 16) }, -1);
        #[cfg(target_arch = "x86_64")]
        assert_eq!(unsafe { cl_cpu_has(b"sse2".as_ptr(), 4) }, 1);
    }
}
