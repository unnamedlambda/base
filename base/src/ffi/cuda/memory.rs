//! Device allocation, host/device transfer, and pinned host staging.
use super::shared::*;
use crate::ffi::{read_ctx_mut, read_ctx_ref};

pub(crate) unsafe extern "C" fn cl_cuda_create_buffer(ctx_ptr: *mut CraneliftCudaContext, size: i64) -> i32 {
    if size <= 0 {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        match ctx.device.alloc_zeros::<u8>(size as usize) {
            Ok(buf) => {
                let idx = state.buffers.len() as i32;
                state.buffers.push(Some(buf));
                idx
            }
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_upload(
    ctx_ptr: *mut CraneliftCudaContext,
    buf_id: i32,
    src_ptr: *const u8,
    size: i64,
) -> i32 {
    if buf_id < 0 || size <= 0 || src_ptr.is_null() {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let bid = buf_id as usize;
        if bid >= state.buffers.len() {
            return -1;
        }
        let data = std::slice::from_raw_parts(src_ptr, size as usize);
        let Some(buf) = state.buffers[bid].as_mut() else {
            return -1;
        };
        match ctx.device.htod_sync_copy_into(data, buf) {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_upload_ptr(
    ctx_ptr: *mut CraneliftCudaContext,
    buf_id: i32,
    src_ptr: *const u8,
    size: i64,
) -> i32 {
    if buf_id < 0 || size <= 0 || src_ptr.is_null() {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let bid = buf_id as usize;
        if bid >= state.buffers.len() {
            return -1;
        }
        let data = std::slice::from_raw_parts(src_ptr, size as usize);
        let Some(buf) = state.buffers[bid].as_mut() else {
            return -1;
        };
        match ctx.device.htod_sync_copy_into(data, buf) {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_upload_ptr_offset(
    ctx_ptr: *mut CraneliftCudaContext,
    buf_id: i32,
    buf_offset: i64,
    src_ptr: *const u8,
    size: i64,
) -> i32 {
    if buf_id < 0 || buf_offset < 0 || size <= 0 || src_ptr.is_null() {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let data = std::slice::from_raw_parts(src_ptr, size as usize);
        let Some(dst) = (unsafe { cuda_buffer_range_ptr(&state, buf_id, buf_offset, size) }) else {
            return -1;
        };
        match unsafe { cudarc::driver::result::memcpy_htod_sync(dst, data) } {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}


pub(crate) unsafe extern "C" fn cl_cuda_upload_ptr_async(
    ctx_ptr: *mut CraneliftCudaContext,
    buf_id: i32,
    src_ptr: *const u8,
    size: i64,
    stream_id: i32,
) -> i32 {
    if buf_id < 0 || size <= 0 || src_ptr.is_null() {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_ref::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        if !bind_cuda_ctx_if_needed(ctx) {
            return -1;
        }
        let Ok(state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };
        let Some(dst) = (unsafe { cuda_buffer_range_ptr(&state, buf_id, 0, size) }) else {
            return -1;
        };
        let data = std::slice::from_raw_parts(src_ptr, size as usize);
        match unsafe { cudarc::driver::result::memcpy_htod_async(dst, data, stream) } {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_upload_ptr_offset_async(
    ctx_ptr: *mut CraneliftCudaContext,
    buf_id: i32,
    buf_offset: i64,
    src_ptr: *const u8,
    size: i64,
    stream_id: i32,
) -> i32 {
    if buf_id < 0 || buf_offset < 0 || size <= 0 || src_ptr.is_null() {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_ref::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        if !bind_cuda_ctx_if_needed(ctx) {
            return -1;
        }
        let Ok(state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };
        let Some(dst) = (unsafe { cuda_buffer_range_ptr(&state, buf_id, buf_offset, size) }) else {
            return -1;
        };
        let data = std::slice::from_raw_parts(src_ptr, size as usize);
        match unsafe { cudarc::driver::result::memcpy_htod_async(dst, data, stream) } {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_download_ptr_async(
    ctx_ptr: *mut CraneliftCudaContext,
    buf_id: i32,
    dst_ptr: *mut u8,
    size: i64,
    stream_id: i32,
) -> i32 {
    if buf_id < 0 || size <= 0 || dst_ptr.is_null() {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_ref::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        if !bind_cuda_ctx_if_needed(ctx) {
            return -1;
        }
        let Ok(state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };
        let Some(src) = (unsafe { cuda_buffer_range_ptr(&state, buf_id, 0, size) }) else {
            return -1;
        };
        let dst = std::slice::from_raw_parts_mut(dst_ptr, size as usize);
        match unsafe { cudarc::driver::result::memcpy_dtoh_async(dst, src, stream) } {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_download_ptr(
    ctx_ptr: *mut CraneliftCudaContext,
    buf_id: i32,
    dst_ptr: *mut u8,
    size: i64,
) -> i32 {
    if buf_id < 0 || size <= 0 || dst_ptr.is_null() {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let bid = buf_id as usize;
        if bid >= state.buffers.len() {
            return -1;
        }
        let dst = std::slice::from_raw_parts_mut(dst_ptr, size as usize);
        let Some(buf) = state.buffers[bid].as_ref() else {
            return -1;
        };
        match ctx.device.dtoh_sync_copy_into(buf, dst) {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

/// Download `size` bytes from a GPU buffer at `buf_offset` into a host pointer.
pub(crate) unsafe extern "C" fn cl_cuda_download_ptr_offset(
    ctx_ptr: *mut CraneliftCudaContext,
    buf_id: i32,
    buf_offset: i64,
    dst_ptr: *mut u8,
    size: i64,
) -> i32 {
    if buf_id < 0 || buf_offset < 0 || size <= 0 || dst_ptr.is_null() {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Some(dev) = (unsafe { cuda_buffer_range_ptr(&state, buf_id, buf_offset, size) }) else {
            return -1;
        };
        let dst = std::slice::from_raw_parts_mut(dst_ptr, size as usize);
        match unsafe { cudarc::driver::result::memcpy_dtoh_sync(dst, dev) } {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_download(
    ctx_ptr: *mut CraneliftCudaContext,
    buf_id: i32,
    dst_ptr: *mut u8,
    size: i64,
) -> i32 {
    if buf_id < 0 || size <= 0 || dst_ptr.is_null() {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let bid = buf_id as usize;
        if bid >= state.buffers.len() {
            return -1;
        }
        let dst = std::slice::from_raw_parts_mut(dst_ptr, size as usize);
        let Some(buf) = state.buffers[bid].as_ref() else {
            return -1;
        };
        match ctx.device.dtoh_sync_copy_into(buf, dst) {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_free_buffer(ctx_ptr: *mut CraneliftCudaContext, buf_id: i32) -> i32 {
    if buf_id < 0 {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let bid = buf_id as usize;
        if bid >= state.buffers.len() {
            return -1;
        }
        match state.buffers[bid].take() {
            Some(_) => 0,
            None => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_pinned_alloc(ctx_ptr: *mut CraneliftCudaContext, size: i64) -> i32 {
    if size <= 0 {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        if !bind_cuda_ctx_if_needed(ctx) {
            return -1;
        }
        let mut ptr = std::ptr::null_mut();
        let result =
            unsafe { cudarc::driver::sys::lib().cuMemAllocHost_v2(&mut ptr, size as usize) };
        if result.result().is_err() || ptr.is_null() {
            return -1;
        }
        let Ok(mut state) = lock_cuda_state(ctx) else {
            let _ = unsafe { cudarc::driver::sys::lib().cuMemFreeHost(ptr) }.result();
            return -1;
        };
        let pid = state.pinned_buffers.len() as i32;
        state.pinned_buffers.push(Some(CudaPinnedHostBuffer {
            ptr,
            _size: size as usize,
        }));
        pid
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_pinned_ptr(ctx_ptr: *mut CraneliftCudaContext, pinned_id: i32) -> i64 {
    if pinned_id < 0 {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_ref::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let pid = pinned_id as usize;
        let Some(buf) = state.pinned_buffers.get(pid).and_then(|b| b.as_ref()) else {
            return -1;
        };
        buf.ptr as i64
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_pinned_free(
    ctx_ptr: *mut CraneliftCudaContext,
    pinned_id: i32,
) -> i32 {
    if pinned_id < 0 {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let pid = pinned_id as usize;
        if pid >= state.pinned_buffers.len() {
            return -1;
        }
        match state.pinned_buffers[pid].take() {
            Some(_) => 0,
            None => -1,
        }
    }))
    .unwrap_or(-1)
}
