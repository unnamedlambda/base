//! Starting and finishing an OS thread that runs one of the program's own
//! functions.
//!
//! Two stateless calls. `cl_thread_start` runs function `fn_index` of the
//! program on a new thread, handing it one pointer, and answers the thread as
//! an owned handle; `cl_thread_finish` waits for that thread and releases the
//! handle, once. Which handles are live, and which of them a program may still
//! finish, is the program's to keep: `Lib.Thread` keeps it in CLIF. What stays
//! here is the one part only the engine can do, finding the function's code in
//! the table the JIT built.

use std::thread::JoinHandle;

use crate::jit::{Compiled, THREAD_COMPILED_FNS};

type Worker = unsafe extern "C" fn(*mut u8);

/// The code of function `fn_index`, when it is shaped like a worker: one
/// pointer in, nothing out. Anything else would read registers nobody set.
unsafe fn worker(fns: &[Compiled], fn_index: i64) -> Option<Worker> {
    let f = fns.get(usize::try_from(fn_index).ok()?)?;
    (f.arity == 1 && !f.answers).then(|| std::mem::transmute::<*const u8, Worker>(f.addr))
}

/// Run function `fn_index` on `arg` on a new thread: the thread as a handle,
/// or `-1` when there is no such worker or no thread could be made.
pub(crate) unsafe extern "C" fn cl_thread_start(fn_index: i64, arg: *mut u8) -> i64 {
    let Some(fns) = THREAD_COMPILED_FNS.with(|cell| cell.borrow().clone()) else {
        return -1;
    };
    let Some(func) = worker(&fns, fn_index) else {
        return -1;
    };
    let arg = arg as usize;
    let spawned = std::thread::Builder::new().spawn(move || {
        THREAD_COMPILED_FNS.with(|cell| *cell.borrow_mut() = Some(fns));
        func(arg as *mut u8);
    });
    match spawned {
        Ok(join) => Box::into_raw(Box::new(join)) as i64,
        Err(_) => -1,
    }
}

/// Wait for the thread `handle` names and release it: `0`, or `-1` when it
/// panicked or `handle` is not one. A handle may be finished once.
pub(crate) unsafe extern "C" fn cl_thread_finish(handle: i64) -> i64 {
    if handle == 0 || handle == -1 {
        return -1;
    }
    let join = Box::from_raw(handle as *mut JoinHandle<()>);
    match join.join() {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// The table the JIT installs, for a caller outside an `execute`.
#[cfg(test)]
fn install(fns: Vec<Compiled>) {
    THREAD_COMPILED_FNS.with(|cell| *cell.borrow_mut() = Some(std::sync::Arc::new(fns)));
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

    fn table(fns: &[Worker]) -> Vec<Compiled> {
        fns.iter().map(|f| Compiled { addr: *f as *const u8, arity: 1, answers: false }).collect()
    }

    #[test]
    fn start_then_finish_runs_the_worker() {
        install(table(&[write_42]));
        let mut val: u64 = 0;
        unsafe {
            let h = cl_thread_start(0, &mut val as *mut u64 as *mut u8);
            assert!(h != -1 && h != 0);
            assert_eq!(cl_thread_finish(h), 0);
        }
        assert_eq!(val, 42);
    }

    #[test]
    fn finish_waits_for_the_worker() {
        install(table(&[slow_write_77]));
        let mut val: u64 = 0;
        unsafe {
            let h = cl_thread_start(0, &mut val as *mut u64 as *mut u8);
            assert_eq!(cl_thread_finish(h), 0);
        }
        assert_eq!(val, 77);
    }

    /// A function not shaped like a worker, or no function at all, is refused
    /// and nothing runs.
    #[test]
    fn a_function_not_shaped_like_a_worker_is_refused() {
        install(vec![
            Compiled { addr: write_42 as *const u8, arity: 5, answers: false },
            Compiled { addr: write_42 as *const u8, arity: 1, answers: true },
        ]);
        let mut val: u64 = 0;
        unsafe {
            for idx in [0, 1, 2, -1] {
                assert_eq!(cl_thread_start(idx, &mut val as *mut u64 as *mut u8), -1);
            }
            assert_eq!(cl_thread_finish(-1), -1);
            assert_eq!(cl_thread_finish(0), -1);
        }
        assert_eq!(val, 0);
    }

    #[test]
    fn no_table_no_thread() {
        THREAD_COMPILED_FNS.with(|cell| *cell.borrow_mut() = None);
        unsafe {
            assert_eq!(cl_thread_start(0, std::ptr::null_mut()), -1);
        }
    }
}
