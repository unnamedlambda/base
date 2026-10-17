//! Performance controls: memory locking, huge pages, thread priority. Each
//! answers `0` when granted and `-1` when not, and changes no memory.

/// Keep `len` bytes from `ptr` in physical memory.
pub(crate) unsafe extern "C" fn cl_mem_lock(ptr: *const u8, len: i64) -> i32 {
    let Ok(len) = usize::try_from(len) else { return -1 };
    match region::lock(ptr, len) {
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
    let Ok(len) = usize::try_from(len) else { return -1 };
    match region::unlock(ptr, len) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Ask for huge pages under the whole pages inside `len` bytes from `ptr`.
pub(crate) unsafe extern "C" fn cl_mem_advise_huge(ptr: *const u8, len: i64) -> i32 {
    let Some(end) = usize::try_from(len).ok().and_then(|n| (ptr as usize).checked_add(n)) else { return -1 };
    #[cfg(target_os = "linux")]
    {
        let page = region::page::size();
        let start = (ptr as usize).div_ceil(page) * page;
        let end = end / page * page;
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
        let _ = end;
        -1
    }
}

/// The calling thread's priority, `0` lowest to `99` highest; raising it may
/// need privilege. `0`, or `-1`.
pub(crate) unsafe extern "C" fn cl_thread_priority(level: i32) -> i32 {
    use thread_priority::{set_current_thread_priority, ThreadPriority, ThreadPriorityValue};
    let Some(v) = u8::try_from(level).ok().and_then(|v| ThreadPriorityValue::try_from(v).ok()) else {
        return -1;
    };
    if set_current_thread_priority(ThreadPriority::Crossplatform(v)).is_ok() {
        0
    } else {
        -1
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
    fn lowering_priority_is_granted_and_off_the_scale_is_not() {
        unsafe {
            assert_eq!(cl_thread_priority(100), -1);
            assert_eq!(cl_thread_priority(-1), -1);
            #[cfg(target_os = "linux")]
            assert_eq!(cl_thread_priority(0), 0);
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
