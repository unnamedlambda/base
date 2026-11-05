//! Threads running a program's own functions, through `std::thread`: as
//! `pthread_create` and `pthread_join`, one word in and one word out. A program
//! joins every thread it starts before its driver is freed.

use std::thread::JoinHandle;

/// A function of the program a thread runs: one word in, one word out.
type Worker = unsafe extern "C" fn(i64) -> i64;

/// A thread running the function at `func` on `arg`, with a stack of `stack`
/// bytes (the system's default for `0` or less): its handle, or `0` when the
/// system makes no thread.
///
/// # Safety
///
/// `func` is the address of a function of the calling program taking one word
/// and answering one, at the host's calling convention.
pub(crate) unsafe extern "C" fn base_thread_spawn(func: *const u8, arg: i64, stack: i64) -> i64 {
    let worker = std::mem::transmute::<*const u8, Worker>(func);
    let mut builder = std::thread::Builder::new();
    if let Ok(n @ 1..) = usize::try_from(stack) {
        builder = builder.stack_size(n);
    }
    match builder.spawn(move || worker(arg)) {
        Ok(join) => Box::into_raw(Box::new(join)) as i64,
        Err(_) => 0,
    }
}

/// Waits for the thread `handle` names to end, releases it, and answers the
/// word its function answered.
///
/// # Safety
///
/// `handle` is one [`base_thread_spawn`] answered, other than `0`, not joined
/// already.
pub(crate) unsafe extern "C" fn base_thread_join(handle: i64) -> i64 {
    let join = Box::from_raw(handle as *mut JoinHandle<i64>);
    // A program's function cannot unwind: a trap ends the process.
    join.join().unwrap_or_else(|_| std::process::abort())
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn double(x: i64) -> i64 {
        x * 2
    }

    unsafe extern "C" fn write_then_answer(p: i64) -> i64 {
        std::thread::sleep(std::time::Duration::from_millis(20));
        *(p as *mut u64) = 77;
        5
    }

    #[test]
    fn a_thread_answers_its_function_at_the_join() {
        unsafe {
            let h = base_thread_spawn(double as *const u8, 21, 0);
            assert_ne!(h, 0);
            assert_eq!(base_thread_join(h), 42);
        }
    }

    #[test]
    fn the_join_waits_for_the_thread() {
        let mut val: u64 = 0;
        unsafe {
            let h = base_thread_spawn(write_then_answer as *const u8, &mut val as *mut u64 as i64, 1 << 20);
            assert_eq!(base_thread_join(h), 5);
        }
        assert_eq!(val, 77);
    }
}
