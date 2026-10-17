//! The performance controls a program may ask the system for.
//!
//! Each is one stateless call answering `0` when the system grants it and `-1`
//! when it declines — for want of privilege, a limit, or an OS that has no such
//! thing — and none changes what memory holds. A program that needs one to
//! hold says so in what it claims; everything else only runs faster with it.

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

/// The calling thread's priority set to `level`, on the scale the
/// `thread-priority` crate makes the same on every system: `0` the lowest,
/// `99` the highest, carried to niceness on Linux and to the priority classes
/// on Windows. Lowering is the thread's own to do; raising past where it
/// started takes privilege. `-1` for a level off the scale or one not granted.
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
