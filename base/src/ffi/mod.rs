pub(crate) mod cuda;
pub(crate) mod file;
pub(crate) mod ht;
pub(crate) mod lmdb;
pub(crate) mod net;
pub(crate) mod stdio;
pub(crate) mod thread;
pub(crate) mod wgpu;
pub(crate) mod window;

pub(super) unsafe fn read_ctx_ref<T>(ctx_ptr: *const T) -> Option<&'static T> {
    ctx_ptr.as_ref()
}

pub(super) unsafe fn read_ctx_mut<T>(ctx_ptr: *mut T) -> Option<&'static mut T> {
    ctx_ptr.as_mut()
}

pub(super) unsafe fn write_ctx_slot<T>(slot_ptr: *mut *mut T, raw: *mut T) -> bool {
    if slot_ptr.is_null() {
        return false;
    }
    std::ptr::write_unaligned(slot_ptr, raw);
    true
}

pub(super) unsafe fn clear_ctx_slot<T>(slot_ptr: *mut *mut T) -> *mut T {
    if slot_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let raw = std::ptr::read_unaligned(slot_ptr as *const *mut T);
    if !raw.is_null() {
        std::ptr::write_unaligned(slot_ptr, std::ptr::null_mut());
    }
    raw
}

pub(super) unsafe fn read_cstr(ptr: *mut u8, off: usize) -> String {
    let start = ptr.add(off);
    read_cstr_ptr(start)
}

pub(super) unsafe fn read_cstr_ptr(start: *const u8) -> String {
    let mut len = 0;
    while *start.add(len) != 0 {
        len += 1;
    }
    String::from_utf8_lossy(std::slice::from_raw_parts(start, len)).into_owned()
}

// Stateless libm wrappers — exposed as FFI for CLIF code that needs trig/pow.

pub(crate) unsafe extern "C" fn cl_sinf(x: f32) -> f32 {
    x.sin()
}

pub(crate) unsafe extern "C" fn cl_cosf(x: f32) -> f32 {
    x.cos()
}

pub(crate) unsafe extern "C" fn cl_powf(base: f32, exp: f32) -> f32 {
    base.powf(exp)
}
