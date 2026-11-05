//! Waiting on a word, through the system's own calls (futex, `WaitOnAddress`,
//! `os_sync_wait_on_address`), so a host's threads and a program's meet at the
//! same words, and a wait may end spuriously; the monotonic and wall clocks;
//! and the system's randomness.

use std::sync::atomic::AtomicU32;
use std::time::Duration;

use wait_on_address::AtomicWait;

/// Sleeps while `word` holds `expected`, until woken, `timeout_ns` passes (never
/// if negative), or spuriously.
pub(crate) fn wait(word: &AtomicU32, expected: u32, timeout_ns: i64) {
    match u64::try_from(timeout_ns) {
        Ok(ns) => word.wait_timeout(expected, Duration::from_nanos(ns)),
        Err(_) => word.wait(expected),
    }
}

/// Wakes one waiter when `count` is 1, none for 0, and all otherwise.
pub(crate) fn wake(word: &AtomicU32, count: i32) {
    match count {
        0 => {}
        1 => word.notify_one(),
        _ => word.notify_all(),
    }
}

/// [`wait`] on the word at `addr`.
///
/// # Safety
///
/// `addr` is 4-byte aligned and points to 4 readable bytes, which every access
/// to while waiters may exist is atomic.
pub(crate) unsafe extern "C" fn base_mem_wait(addr: *const u8, expected: i32, timeout_ns: i64) {
    wait(AtomicU32::from_ptr(addr as *mut u32), expected as u32, timeout_ns);
}

/// [`wake`] on the word at `addr`.
///
/// # Safety
///
/// As [`base_mem_wait`].
pub(crate) unsafe extern "C" fn base_mem_wake(addr: *const u8, count: i32) {
    wake(AtomicU32::from_ptr(addr as *mut u32), count);
}

/// Nanoseconds on the system's monotonic clock.
pub(crate) extern "C" fn base_clock_now() -> i64 {
    #[cfg(unix)]
    {
        #[cfg(target_vendor = "apple")]
        let clock = libc::CLOCK_UPTIME_RAW;
        #[cfg(not(target_vendor = "apple"))]
        let clock = libc::CLOCK_MONOTONIC;
        let mut t = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        // SAFETY: `t` is a valid timespec to write; the clock always exists.
        unsafe { libc::clock_gettime(clock, &mut t) };
        t.tv_sec as i64 * 1_000_000_000 + t.tv_nsec as i64
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
        let (mut ticks, mut freq) = (0i64, 0i64);
        // SAFETY: both write one i64; neither fails on Windows XP or later.
        unsafe {
            QueryPerformanceCounter(&mut ticks);
            QueryPerformanceFrequency(&mut freq);
        }
        (ticks as i128 * 1_000_000_000 / freq as i128) as i64
    }
}

/// Nanoseconds since 1970-01-01 00:00:00 UTC on the system's wall clock,
/// negative before it. The system may set it back or forward.
pub(crate) extern "C" fn base_clock_wall() -> i64 {
    let ns = |d: std::time::Duration| i64::try_from(d.as_nanos()).unwrap_or(i64::MAX);
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => ns(d),
        Err(e) => -ns(e.duration()),
    }
}

/// `len` bytes from the system's source of randomness written at `dst`: `len`,
/// or `-1` for a negative length or a source that fails.
///
/// # Safety
///
/// `dst` has room for `len` bytes.
pub(crate) unsafe extern "C" fn base_random(dst: *mut u8, len: i64) -> i64 {
    let Ok(n) = usize::try_from(len) else { return -1 };
    if n == 0 {
        return 0;
    }
    match getrandom::fill(std::slice::from_raw_parts_mut(dst, n)) {
        Ok(()) => len,
        Err(_) => -1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;
    use std::time::Instant;

    #[test]
    fn a_word_that_changed_is_not_slept_on() {
        let w = AtomicU32::new(5);
        let t0 = Instant::now();
        unsafe { base_mem_wait(w.as_ptr() as *const u8, 4, 5_000_000_000) };
        assert!(t0.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn a_wait_ends_by_its_timeout() {
        let w = AtomicU32::new(5);
        let t0 = Instant::now();
        unsafe { base_mem_wait(w.as_ptr() as *const u8, 5, 20_000_000) };
        assert!(t0.elapsed() < Duration::from_secs(5));
    }

    /// A waiter sleeps until the word changes and it is woken. Many rounds, so
    /// a wake lost between check and sleep shows.
    #[test]
    fn a_change_and_a_wake_end_the_wait() {
        let word = Arc::new(AtomicU32::new(0));
        let w2 = Arc::clone(&word);
        let waiter = std::thread::spawn(move || {
            for round in 0..2000u32 {
                while w2.load(Ordering::SeqCst) == round {
                    unsafe { base_mem_wait(w2.as_ptr() as *const u8, round as i32, -1) };
                }
            }
        });
        for round in 1..=2000u32 {
            word.store(round, Ordering::SeqCst);
            unsafe { base_mem_wake(word.as_ptr() as *const u8, -1) };
        }
        waiter.join().unwrap();
    }

    /// A host's thread waits with the system's own call and a program's wake
    /// reaches it: the two meet at the word.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_wake_reaches_a_host_waiting_with_the_system_call() {
        let word = Arc::new(AtomicU32::new(0));
        let w2 = Arc::clone(&word);
        let host = std::thread::spawn(move || {
            while w2.load(Ordering::SeqCst) == 0 {
                unsafe {
                    libc::syscall(
                        libc::SYS_futex,
                        w2.as_ptr(),
                        libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
                        0u32,
                        std::ptr::null::<libc::timespec>(),
                    )
                };
            }
        });
        std::thread::sleep(Duration::from_millis(20));
        word.store(1, Ordering::SeqCst);
        unsafe { base_mem_wake(word.as_ptr() as *const u8, 1) };
        host.join().unwrap();
    }

    /// The clock is the system's: it reads what the system's own call reads.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_clock_is_the_systems() {
        let mut t = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut t) };
        let system = t.tv_sec as i64 * 1_000_000_000 + t.tv_nsec as i64;
        let ours = base_clock_now();
        assert!(ours >= system && ours - system < 1_000_000_000, "{system} {ours}");
    }

    #[test]
    fn the_clock_does_not_go_backwards() {
        let mut last = base_clock_now();
        for _ in 0..1000 {
            let now = base_clock_now();
            assert!(now >= last);
            last = now;
        }
    }
}
