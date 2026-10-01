//! The performance controls a program may ask the system for.
//!
//! Each is one stateless call answering `0` when the system grants it and `-1`
//! when it declines — for want of privilege, a limit, or an OS that has no such
//! thing — and none changes what memory holds. A program that needs one to
//! hold says so in what it claims; everything else only runs faster with it.

/// Keep `len` bytes from `ptr` in physical memory.
pub(crate) unsafe extern "C" fn cl_mem_lock(ptr: *const u8, len: i64) -> i32 {
    if ptr.is_null() || len <= 0 {
        return -1;
    }
    match region::lock(ptr, len as usize) {
        Ok(guard) => {
            // Unlocking is the program's to ask for, not the guard's.
            std::mem::forget(guard);
            0
        }
        Err(_) => -1,
    }
}

/// Let the system page `len` bytes from `ptr` out again.
pub(crate) unsafe extern "C" fn cl_mem_unlock(ptr: *const u8, len: i64) -> i32 {
    if ptr.is_null() || len <= 0 {
        return -1;
    }
    match region::unlock(ptr, len as usize) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Ask for huge pages under the whole pages inside `len` bytes from `ptr`.
pub(crate) unsafe extern "C" fn cl_mem_advise_huge(ptr: *const u8, len: i64) -> i32 {
    if ptr.is_null() || len <= 0 {
        return -1;
    }
    #[cfg(target_os = "linux")]
    {
        let page = region::page::size();
        let start = (ptr as usize).div_ceil(page) * page;
        let end = (ptr as usize + len as usize) / page * page;
        if start >= end {
            return -1;
        }
        if libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_HUGEPAGE) == 0 {
            0
        } else {
            -1
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        -1
    }
}

/// Raise (`1`) or lower (`-1`) the calling thread's priority by a step from
/// where it stands, or reset it (`0`).
pub(crate) unsafe extern "C" fn cl_thread_priority(level: i32) -> i32 {
    #[cfg(target_os = "linux")]
    {
        // Relative to where the thread stands: lowering is always the
        // thread's own to do, raising takes privilege.
        let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
        *libc::__errno_location() = 0;
        let now = libc::getpriority(libc::PRIO_PROCESS, tid);
        if now == -1 && *libc::__errno_location() != 0 {
            return -1;
        }
        let nice = match level {
            1 => now - 5,
            0 => 0,
            -1 => (now + 5).min(19),
            _ => return -1,
        };
        if libc::setpriority(libc::PRIO_PROCESS, tid, nice) == 0 {
            0
        } else {
            -1
        }
    }
    #[cfg(all(unix, not(target_os = "linux")))]
    {
        let _ = level;
        -1
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Threading::{
            GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL,
            THREAD_PRIORITY_BELOW_NORMAL, THREAD_PRIORITY_NORMAL,
        };
        let p = match level {
            1 => THREAD_PRIORITY_ABOVE_NORMAL,
            0 => THREAD_PRIORITY_NORMAL,
            -1 => THREAD_PRIORITY_BELOW_NORMAL,
            _ => return -1,
        };
        if SetThreadPriority(GetCurrentThread(), p) != 0 {
            0
        } else {
            -1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_then_unlock_a_page() {
        let buf = vec![0u8; 8192];
        unsafe {
            let locked = cl_mem_lock(buf.as_ptr(), 4096);
            if locked == 0 {
                assert_eq!(cl_mem_unlock(buf.as_ptr(), 4096), 0);
            }
            assert_eq!(cl_mem_lock(std::ptr::null(), 4096), -1);
            assert_eq!(cl_mem_lock(buf.as_ptr(), 0), -1);
        }
    }

    #[test]
    fn lowering_priority_is_granted_and_nonsense_is_not() {
        unsafe {
            assert_eq!(cl_thread_priority(7), -1);
            #[cfg(target_os = "linux")]
            assert_eq!(cl_thread_priority(-1), 0);
        }
    }

    #[test]
    fn huge_pages_over_less_than_a_page_are_declined() {
        let buf = vec![0u8; 64];
        unsafe {
            assert_eq!(cl_mem_advise_huge(buf.as_ptr(), 64), -1);
        }
    }
}
