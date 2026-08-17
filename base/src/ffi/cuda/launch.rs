//! Kernel launch, in its four forms.
use super::shared::*;
use crate::ffi::{read_ctx_mut};

/// Launch a PTX kernel.
///
/// - `kernel_off`: offset of null-terminated PTX source in shared memory
/// - `n_bufs`: number of buffer arguments
/// - `bind_off`: offset of packed i32 buffer IDs (4 bytes each)
/// - `grid_x/y/z`: grid dimensions
/// - `block_x/y/z`: block dimensions
///
/// The PTX must define `.entry main(...)` with `n_bufs` pointer parameters.
pub(crate) unsafe extern "C" fn cl_cuda_launch(
    ctx_ptr: *mut CraneliftCudaContext,
    kernel_ptr: *const u8,
    n_bufs: i32,
    bind_ptr: *const u8,
    grid_x: i32,
    grid_y: i32,
    grid_z: i32,
    block_x: i32,
    block_y: i32,
    block_z: i32,
) -> i32 {
    if n_bufs < 0
        || grid_x <= 0
        || grid_y <= 0
        || grid_z <= 0
        || block_x <= 0
        || block_y <= 0
        || block_z <= 0
    {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Ok(function) = load_raw_cuda_main_kernel(ctx, &mut state, kernel_ptr) else {
            eprintln!("cl_cuda_launch: load kernel failed");
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, -1) else {
            return -1;
        };
        unsafe {
            launch_raw_cuda_kernel(
                &state, function, n_bufs, bind_ptr, grid_x, grid_y, grid_z, block_x, block_y,
                block_z, stream,
            )
        }
    }))
    .unwrap_or(-1)
}


/// Launch a PTX kernel by name.
///
/// Same as `cl_cuda_launch` but reads the entry-point name from shared memory
/// at `name_off` instead of hardcoding `"main"`.
pub(crate) unsafe extern "C" fn cl_cuda_launch_named(
    ctx_ptr: *mut CraneliftCudaContext,
    kernel_ptr: *const u8,
    name_ptr: *const u8,
    n_bufs: i32,
    bind_ptr: *const u8,
    grid_x: i32,
    grid_y: i32,
    grid_z: i32,
    block_x: i32,
    block_y: i32,
    block_z: i32,
) -> i32 {
    if n_bufs < 0
        || grid_x <= 0
        || grid_y <= 0
        || grid_z <= 0
        || block_x <= 0
        || block_y <= 0
        || block_z <= 0
    {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Ok(function) = load_raw_cuda_named_kernel(ctx, &mut state, kernel_ptr, name_ptr) else {
            eprintln!("cl_cuda_launch_named: load kernel failed");
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, -1) else {
            return -1;
        };
        unsafe {
            launch_raw_cuda_kernel(
                &state, function, n_bufs, bind_ptr, grid_x, grid_y, grid_z, block_x, block_y,
                block_z, stream,
            )
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_launch_on_stream(
    ctx_ptr: *mut CraneliftCudaContext,
    kernel_ptr: *const u8,
    n_bufs: i32,
    bind_ptr: *const u8,
    grid_x: i32,
    grid_y: i32,
    grid_z: i32,
    block_x: i32,
    block_y: i32,
    block_z: i32,
    stream_id: i32,
) -> i32 {
    if n_bufs < 0
        || grid_x <= 0
        || grid_y <= 0
        || grid_z <= 0
        || block_x <= 0
        || block_y <= 0
        || block_z <= 0
    {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Ok(function) = load_raw_cuda_main_kernel(ctx, &mut state, kernel_ptr) else {
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };
        unsafe {
            launch_raw_cuda_kernel(
                &state, function, n_bufs, bind_ptr, grid_x, grid_y, grid_z, block_x, block_y,
                block_z, stream,
            )
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_launch_named_on_stream(
    ctx_ptr: *mut CraneliftCudaContext,
    kernel_ptr: *const u8,
    name_ptr: *const u8,
    n_bufs: i32,
    bind_ptr: *const u8,
    grid_x: i32,
    grid_y: i32,
    grid_z: i32,
    block_x: i32,
    block_y: i32,
    block_z: i32,
    stream_id: i32,
) -> i32 {
    if n_bufs < 0
        || grid_x <= 0
        || grid_y <= 0
        || grid_z <= 0
        || block_x <= 0
        || block_y <= 0
        || block_z <= 0
    {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Ok(function) = load_raw_cuda_named_kernel(ctx, &mut state, kernel_ptr, name_ptr) else {
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };
        unsafe {
            launch_raw_cuda_kernel(
                &state, function, n_bufs, bind_ptr, grid_x, grid_y, grid_z, block_x, block_y,
                block_z, stream,
            )
        }
    }))
    .unwrap_or(-1)
}
