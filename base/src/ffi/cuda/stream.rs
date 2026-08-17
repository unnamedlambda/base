//! Stream-ordered execution: streams, the events that order them, and
//! graph capture and replay.
use super::shared::*;
use crate::ffi::{read_ctx_mut, read_ctx_ref};

pub(crate) unsafe extern "C" fn cl_cuda_stream_create(ctx_ptr: *mut CraneliftCudaContext) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        if !bind_cuda_ctx_if_needed(ctx) {
            return -1;
        }
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        match cudarc::driver::result::stream::create(
            cudarc::driver::result::stream::StreamKind::NonBlocking,
        ) {
            Ok(raw) => {
                let fence_event = match cudarc::driver::result::event::create(
                    cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
                ) {
                    Ok(event) => event,
                    Err(_) => {
                        let _ = unsafe { cudarc::driver::result::stream::destroy(raw) };
                        return -1;
                    }
                };
                let default_stream = *ctx.device.cu_stream();
                let wait_ok = unsafe {
                    cudarc::driver::result::event::record(fence_event, default_stream).and_then(
                        |_| {
                            cudarc::driver::result::stream::wait_event(
                                raw,
                                fence_event,
                                cudarc::driver::sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
                            )
                        },
                    )
                }
                .is_ok();
                let _ = unsafe { cudarc::driver::result::event::destroy(fence_event) };
                if !wait_ok {
                    let _ = unsafe { cudarc::driver::result::stream::destroy(raw) };
                    return -1;
                }
                let sid = state.streams.len() as i32;
                state.streams.push(Some(CudaOwnedStream { raw }));
                sid
            }
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_stream_sync(
    ctx_ptr: *mut CraneliftCudaContext,
    stream_id: i32,
) -> i32 {
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
        match unsafe { cudarc::driver::result::stream::synchronize(stream) } {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_stream_destroy(
    ctx_ptr: *mut CraneliftCudaContext,
    stream_id: i32,
) -> i32 {
    if stream_id < 0 {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        if !bind_cuda_ctx_if_needed(ctx) {
            return -1;
        }
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let sid = stream_id as usize;
        if sid >= state.streams.len() {
            return -1;
        }
        match state.streams[sid].take() {
            Some(stream) => {
                state.stream_blas.remove(&stream_id);
                let fence_event = match cudarc::driver::result::event::create(
                    cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
                ) {
                    Ok(event) => event,
                    Err(_) => return -1,
                };
                let default_stream = *ctx.device.cu_stream();
                let wait_ok = unsafe {
                    cudarc::driver::result::event::record(fence_event, stream.raw).and_then(|_| {
                        cudarc::driver::result::stream::wait_event(
                            default_stream,
                            fence_event,
                            cudarc::driver::sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
                        )
                    })
                }
                .is_ok();
                let _ = unsafe { cudarc::driver::result::event::destroy(fence_event) };
                if wait_ok {
                    drop(stream);
                    0
                } else {
                    drop(stream);
                    -1
                }
            }
            None => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_event_create(ctx_ptr: *mut CraneliftCudaContext) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        if !bind_cuda_ctx_if_needed(ctx) {
            return -1;
        }
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        match cudarc::driver::result::event::create(
            cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
        ) {
            Ok(raw) => {
                let eid = state.events.len() as i32;
                state.events.push(Some(CudaOwnedEvent { raw }));
                eid
            }
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_event_record(
    ctx_ptr: *mut CraneliftCudaContext,
    event_id: i32,
    stream_id: i32,
) -> i32 {
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
        let Some(event) = resolve_cuda_event(&state, event_id) else {
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };
        match unsafe { cudarc::driver::result::event::record(event, stream) } {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_stream_wait_event(
    ctx_ptr: *mut CraneliftCudaContext,
    stream_id: i32,
    event_id: i32,
) -> i32 {
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
        let Some(event) = resolve_cuda_event(&state, event_id) else {
            return -1;
        };
        match unsafe {
            cudarc::driver::result::stream::wait_event(
                stream,
                event,
                cudarc::driver::sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
            )
        } {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_event_elapsed_ms_bits(
    ctx_ptr: *mut CraneliftCudaContext,
    start_event_id: i32,
    end_event_id: i32,
) -> i32 {
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
        let Some(start) = resolve_cuda_event(&state, start_event_id) else {
            return -1;
        };
        let Some(end) = resolve_cuda_event(&state, end_event_id) else {
            return -1;
        };
        match unsafe { cudarc::driver::result::event::elapsed(start, end) } {
            Ok(ms) => i32::from_ne_bytes(ms.to_bits().to_ne_bytes()),
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_event_destroy(
    ctx_ptr: *mut CraneliftCudaContext,
    event_id: i32,
) -> i32 {
    if event_id < 0 {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let eid = event_id as usize;
        if eid >= state.events.len() {
            return -1;
        }
        match state.events[eid].take() {
            Some(_) => 0,
            None => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_graph_begin_capture(
    ctx_ptr: *mut CraneliftCudaContext,
    stream_id: i32,
) -> i32 {
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
        match unsafe {
            cudarc::driver::sys::lib().cuStreamBeginCapture_v2(
                stream,
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
            )
        }
        .result()
        {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_graph_end_capture(
    ctx_ptr: *mut CraneliftCudaContext,
    stream_id: i32,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        if !bind_cuda_ctx_if_needed(ctx) {
            return -1;
        }
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };
        let mut graph: cudarc::driver::sys::CUgraph = std::ptr::null_mut();
        if unsafe { cudarc::driver::sys::lib().cuStreamEndCapture(stream, &mut graph) }
            .result()
            .is_err()
            || graph.is_null()
        {
            return -1;
        }
        let mut exec: cudarc::driver::sys::CUgraphExec = std::ptr::null_mut();
        let instantiate_ok =
            unsafe { cudarc::driver::sys::lib().cuGraphInstantiateWithFlags(&mut exec, graph, 0) }
                .result()
                .is_ok()
                && !exec.is_null();
        let _ = unsafe { cudarc::driver::sys::lib().cuGraphDestroy(graph) }.result();
        if !instantiate_ok {
            return -1;
        }
        let gid = state.graphs.len() as i32;
        state.graphs.push(Some(CudaOwnedGraphExec { raw: exec }));
        gid
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_graph_upload(
    ctx_ptr: *mut CraneliftCudaContext,
    graph_id: i32,
    stream_id: i32,
) -> i32 {
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
        let Some(graph) = resolve_cuda_graph_exec(&state, graph_id) else {
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };
        match unsafe { cudarc::driver::sys::lib().cuGraphUpload(graph, stream) }.result() {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_graph_launch(
    ctx_ptr: *mut CraneliftCudaContext,
    graph_id: i32,
    stream_id: i32,
) -> i32 {
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
        let Some(graph) = resolve_cuda_graph_exec(&state, graph_id) else {
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };
        match unsafe { cudarc::driver::sys::lib().cuGraphLaunch(graph, stream) }.result() {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_graph_destroy(
    ctx_ptr: *mut CraneliftCudaContext,
    graph_id: i32,
) -> i32 {
    if graph_id < 0 {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let gid = graph_id as usize;
        if gid >= state.graphs.len() {
            return -1;
        }
        match state.graphs[gid].take() {
            Some(_) => 0,
            None => -1,
        }
    }))
    .unwrap_or(-1)
}
