//! Threads running a program's own functions: started on a function's
//! address with one pointer, and finished once. The program finishes its
//! threads before its driver is freed.

use std::thread::JoinHandle;

type Worker = unsafe extern "C" fn(*mut u8);

/// Run the worker at `func` on `arg`, on a new thread: the thread as a handle,
/// or `-1` when no thread could be made.
///
/// # Safety
///
/// `func` is the address of a function of the calling program taking one
/// pointer and answering nothing, which stays mapped until the thread is
/// finished.
pub(crate) unsafe extern "C" fn cl_thread_start(func: *const u8, arg: *mut u8) -> i64 {
    let worker = std::mem::transmute::<*const u8, Worker>(func);
    let arg = arg as usize;
    match std::thread::Builder::new().spawn(move || worker(arg as *mut u8)) {
        Ok(join) => Box::into_raw(Box::new(join)) as i64,
        Err(_) => -1,
    }
}

/// Wait for the thread `handle` names and release it: `0`, or `-1` when it
/// panicked.
///
/// # Safety
///
/// `handle` is one `cl_thread_start` answered, other than `-1`, not finished
/// already.
pub(crate) unsafe extern "C" fn cl_thread_finish(handle: i64) -> i64 {
    let join = Box::from_raw(handle as *mut JoinHandle<()>);
    match join.join() {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn write_42(p: *mut u8) {
        *(p as *mut u64) = 42;
    }

    unsafe extern "C" fn slow_write_77(p: *mut u8) {
        std::thread::sleep(std::time::Duration::from_millis(20));
        *(p as *mut u64) = 77;
    }

    #[test]
    fn start_then_finish_runs_the_worker() {
        let mut val: u64 = 0;
        unsafe {
            let h = cl_thread_start(write_42 as *const u8, &mut val as *mut u64 as *mut u8);
            assert!(h != -1 && h != 0);
            assert_eq!(cl_thread_finish(h), 0);
        }
        assert_eq!(val, 42);
    }

    #[test]
    fn finish_waits_for_the_worker() {
        let mut val: u64 = 0;
        unsafe {
            let h = cl_thread_start(slow_write_77 as *const u8, &mut val as *mut u64 as *mut u8);
            assert_eq!(cl_thread_finish(h), 0);
        }
        assert_eq!(val, 77);
    }

}
