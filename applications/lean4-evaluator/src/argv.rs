use std::sync::OnceLock;

static ARGS: OnceLock<Vec<String>> = OnceLock::new();

pub fn init() {
    ARGS.get_or_init(|| std::env::args().collect());
}

/// Get the number of arguments.
/// FFI signature: fn(*mut u8) -> i64
/// The arg pointer is unused.
#[no_mangle]
pub extern "C" fn get_argc(_arg: *mut u8) -> i64 {
    ARGS.get().map(|a| a.len() as i64).unwrap_or(0)
}

/// Get an argument by index.
/// FFI signature: fn(*mut u8) -> i64
///
/// Memory layout at arg pointer:
/// - [0..4]: index (u32 little-endian)
/// - [4..8]: max_len (u32 little-endian)
/// - [8..]: buffer to write the argument string
///
/// Returns: length written, or -1 if index out of bounds
#[no_mangle]
pub extern "C" fn get_argv(arg: *mut u8) -> i64 {
    let args = match ARGS.get() {
        Some(a) => a,
        None => return -1,
    };

    unsafe {
        let index = *(arg as *const u32) as usize;
        let max_len = *((arg as *const u32).add(1)) as usize;
        let buffer = arg.add(8);

        if index >= args.len() {
            return -1;
        }

        let s = &args[index];
        let len = s.len().min(max_len);
        std::ptr::copy_nonoverlapping(s.as_ptr(), buffer, len);
        len as i64
    }
}

pub fn get_argc_ptr() -> usize {
    get_argc as usize
}

pub fn get_argv_ptr() -> usize {
    get_argv as usize
}
