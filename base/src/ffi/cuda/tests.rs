//! The CUDA FFI's own tests, kept beside the modules they exercise.

use super::*;

// ── PTX kernels ──────────────────────────────────────────────────────────

// result[i] = a[i] + b[i] (3 buffer args)
const PTX_VEC_ADD: &str = "\
.version 7.0
.target sm_50
.address_size 64

.visible .entry main(
.param .u64 a_ptr,
.param .u64 b_ptr,
.param .u64 result_ptr
)
{
.reg .u32 %r0;
.reg .u64 %ra, %rb, %rc, %off;
.reg .f32 %fa, %fb, %fr;

mov.u32 %r0, %tid.x;
cvt.u64.u32 %off, %r0;
shl.b64 %off, %off, 2;

ld.param.u64 %ra, [a_ptr];
ld.param.u64 %rb, [b_ptr];
ld.param.u64 %rc, [result_ptr];

add.u64 %ra, %ra, %off;
add.u64 %rb, %rb, %off;
add.u64 %rc, %rc, %off;

ld.global.f32 %fa, [%ra];
ld.global.f32 %fb, [%rb];
add.f32 %fr, %fa, %fb;
st.global.f32 [%rc], %fr;

ret;
}\0";

// data[i] *= 2.0
const PTX_MUL2: &str = "\
.version 7.0
.target sm_50
.address_size 64

.visible .entry main(
.param .u64 data_ptr
)
{
.reg .u32 %r0;
.reg .u64 %rd, %off;
.reg .f32 %fv, %fc;

mov.u32 %r0, %tid.x;
cvt.u64.u32 %off, %r0;
shl.b64 %off, %off, 2;

ld.param.u64 %rd, [data_ptr];
add.u64 %rd, %rd, %off;

ld.global.f32 %fv, [%rd];
mov.f32 %fc, 0f40000000;
mul.f32 %fv, %fv, %fc;
st.global.f32 [%rd], %fv;

ret;
}\0";

// data[i] *= 3.0
const PTX_MUL3: &str = "\
.version 7.0
.target sm_50
.address_size 64

.visible .entry main(
.param .u64 data_ptr
)
{
.reg .u32 %r0;
.reg .u64 %rd, %off;
.reg .f32 %fv, %fc;

mov.u32 %r0, %tid.x;
cvt.u64.u32 %off, %r0;
shl.b64 %off, %off, 2;

ld.param.u64 %rd, [data_ptr];
add.u64 %rd, %rd, %off;

ld.global.f32 %fv, [%rd];
mov.f32 %fc, 0f40400000;
mul.f32 %fv, %fv, %fc;
st.global.f32 [%rd], %fv;

ret;
}\0";

// entry point named "add_one"; data[i] += 1.0
const PTX_ADD_ONE_NAMED: &str = "\
.version 7.0
.target sm_50
.address_size 64

.visible .entry add_one(
.param .u64 data_ptr
)
{
.reg .u32 %r0;
.reg .u64 %rd, %off;
.reg .f32 %fv, %fc;

mov.u32 %r0, %tid.x;
cvt.u64.u32 %off, %r0;
shl.b64 %off, %off, 2;

ld.param.u64 %rd, [data_ptr];
add.u64 %rd, %rd, %off;

ld.global.f32 %fv, [%rd];
mov.f32 %fc, 0f3F800000;
add.f32 %fv, %fv, %fc;
st.global.f32 [%rd], %fv;

ret;
}\0";

const NAME_ADD_ONE: &str = "add_one\0";

// ── Helpers ──────────────────────────────────────────────────────────────

unsafe fn init_ctx() -> *mut CraneliftCudaContext {
    let mut slot: *mut CraneliftCudaContext = std::ptr::null_mut();
    cl_cuda_init(&mut slot);
    assert!(!slot.is_null(), "cuda init failed (no GPU?)");
    slot
}

unsafe fn cleanup_ctx(ctx: *mut CraneliftCudaContext) {
    let mut slot = ctx;
    cl_cuda_cleanup(&mut slot);
    assert!(slot.is_null());
}

fn pack_bind_ids(ids: &[i32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(ids.len() * 4);
    for &id in ids {
        v.extend_from_slice(&id.to_le_bytes());
    }
    v
}

fn f32s_to_bytes(xs: &[f32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(xs.len() * 4);
    for &x in xs {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v
}

fn bytes_to_f32s(bs: &[u8]) -> Vec<f32> {
    bs.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

unsafe fn make_buf_with(ctx: *mut CraneliftCudaContext, data: &[f32]) -> i32 {
    let bytes = f32s_to_bytes(data);
    let buf = cl_cuda_create_buffer(ctx, bytes.len() as i64);
    assert!(buf >= 0);
    let rc = cl_cuda_upload(ctx, buf, bytes.as_ptr(), bytes.len() as i64);
    assert_eq!(rc, 0);
    buf
}

unsafe fn download_f32(ctx: *mut CraneliftCudaContext, buf: i32, n: usize) -> Vec<f32> {
    let mut out = vec![0u8; n * 4];
    let rc = cl_cuda_download(ctx, buf, out.as_mut_ptr(), out.len() as i64);
    assert_eq!(rc, 0);
    bytes_to_f32s(&out)
}

fn approx_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
}

// ── Lifecycle / buffers ──────────────────────────────────────────────────

#[test]
fn init_then_cleanup_lifecycle() {
    unsafe {
        let ctx = init_ctx();
        cleanup_ctx(ctx);
    }
}

#[test]
fn create_buffers_return_sequential_ids() {
    unsafe {
        let ctx = init_ctx();
        let a = cl_cuda_create_buffer(ctx, 64);
        let b = cl_cuda_create_buffer(ctx, 64);
        let c = cl_cuda_create_buffer(ctx, 64);
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(c, 2);
        cleanup_ctx(ctx);
    }
}

#[test]
fn create_buffer_invalid_size_returns_neg1() {
    unsafe {
        let ctx = init_ctx();
        assert_eq!(cl_cuda_create_buffer(ctx, 0), -1);
        assert_eq!(cl_cuda_create_buffer(ctx, -8), -1);
        cleanup_ctx(ctx);
    }
}

#[test]
fn upload_download_roundtrip() {
    unsafe {
        let ctx = init_ctx();
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let buf = make_buf_with(ctx, &data);
        let got = download_f32(ctx, buf, data.len());
        assert!(approx_eq(&got, &data));
        cleanup_ctx(ctx);
    }
}

#[test]
fn upload_ptr_download_ptr_roundtrip() {
    unsafe {
        let ctx = init_ctx();
        let data = [10.0f32, 20.0, 30.0, 40.0];
        let bytes = f32s_to_bytes(&data);
        let buf = cl_cuda_create_buffer(ctx, bytes.len() as i64);
        assert!(buf >= 0);
        assert_eq!(
            cl_cuda_upload_ptr(ctx, buf, bytes.as_ptr(), bytes.len() as i64),
            0
        );
        let mut out = vec![0u8; bytes.len()];
        assert_eq!(
            cl_cuda_download_ptr(ctx, buf, out.as_mut_ptr(), out.len() as i64),
            0
        );
        assert!(approx_eq(&bytes_to_f32s(&out), &data));
        cleanup_ctx(ctx);
    }
}

#[test]
fn upload_offset_download_offset() {
    unsafe {
        let ctx = init_ctx();
        let buf = cl_cuda_create_buffer(ctx, 32);
        let zeros = vec![0u8; 32];
        assert_eq!(cl_cuda_upload(ctx, buf, zeros.as_ptr(), 32), 0);

        let part = f32s_to_bytes(&[10.0, 20.0, 30.0, 40.0]);
        assert_eq!(
            cl_cuda_upload_ptr_offset(ctx, buf, 16, part.as_ptr(), part.len() as i64),
            0
        );
        // Out-of-bounds offset write fails.
        assert_eq!(
            cl_cuda_upload_ptr_offset(ctx, buf, 20, part.as_ptr(), part.len() as i64),
            -1
        );

        let mut out = vec![0u8; 16];
        assert_eq!(
            cl_cuda_download_ptr_offset(ctx, buf, 16, out.as_mut_ptr(), out.len() as i64),
            0
        );
        assert!(approx_eq(&bytes_to_f32s(&out), &[10.0, 20.0, 30.0, 40.0]));
        cleanup_ctx(ctx);
    }
}

/// Every transfer entry point refuses a range that leaves the buffer.
///
/// The offset *upload* was checked and the offset *download* was not, so the
/// asymmetry in the code was matched by an asymmetry here. The async pair
/// reach `memcpy_htod_async`/`memcpy_dtoh_async` directly, which check
/// nothing at all: past the end they wrote and read another allocation.
#[test]
fn transfers_out_of_range_return_neg1() {
    unsafe {
        let ctx = init_ctx();
        let buf = cl_cuda_create_buffer(ctx, 32);
        assert!(buf >= 0);
        let src = vec![7u8; 64];
        let mut dst = vec![0u8; 64];

        // Offset past the end, both directions.
        assert_eq!(cl_cuda_upload_ptr_offset(ctx, buf, 24, src.as_ptr(), 16), -1);
        assert_eq!(
            cl_cuda_download_ptr_offset(ctx, buf, 24, dst.as_mut_ptr(), 16),
            -1
        );
        // Offset inside but length running past it.
        assert_eq!(
            cl_cuda_download_ptr_offset(ctx, buf, 0, dst.as_mut_ptr(), 64),
            -1
        );

        // The async pair, with no offset at all — size alone leaves the buffer.
        let s = cl_cuda_stream_create(ctx);
        assert!(s >= 0);
        assert_eq!(cl_cuda_upload_ptr_async(ctx, buf, src.as_ptr(), 64, s), -1);
        assert_eq!(
            cl_cuda_download_ptr_async(ctx, buf, dst.as_mut_ptr(), 64, s),
            -1
        );
        assert_eq!(
            cl_cuda_upload_ptr_offset_async(ctx, buf, 24, src.as_ptr(), 16, s),
            -1
        );

        // …and the whole buffer still works, so this is a bound and not a ban.
        assert_eq!(cl_cuda_upload_ptr_async(ctx, buf, src.as_ptr(), 32, s), 0);
        assert_eq!(
            cl_cuda_download_ptr_async(ctx, buf, dst.as_mut_ptr(), 32, s),
            0
        );
        cleanup_ctx(ctx);
    }
}

#[test]
fn free_buffer_and_double_free_returns_neg1() {
    unsafe {
        let ctx = init_ctx();
        let buf = cl_cuda_create_buffer(ctx, 32);
        assert!(buf >= 0);
        assert_eq!(cl_cuda_free_buffer(ctx, buf), 0);
        assert_eq!(cl_cuda_free_buffer(ctx, buf), -1);
        assert_eq!(cl_cuda_free_buffer(ctx, 999), -1);
        assert_eq!(cl_cuda_free_buffer(ctx, -1), -1);
        cleanup_ctx(ctx);
    }
}

// ── Launch ───────────────────────────────────────────────────────────────

#[test]
fn launch_vec_add() {
    unsafe {
        let ctx = init_ctx();
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [10.0f32, 20.0, 30.0, 40.0];
        let a_buf = make_buf_with(ctx, &a);
        let b_buf = make_buf_with(ctx, &b);
        let c_buf = cl_cuda_create_buffer(ctx, 16);
        assert!(c_buf >= 0);

        let binds = pack_bind_ids(&[a_buf, b_buf, c_buf]);
        let rc = cl_cuda_launch(
            ctx,
            PTX_VEC_ADD.as_ptr(),
            3,
            binds.as_ptr(),
            1, 1, 1,
            4, 1, 1,
        );
        assert_eq!(rc, 0);
        assert_eq!(cl_cuda_sync(ctx as *const _), 0);
        assert!(approx_eq(&download_f32(ctx, c_buf, 4), &[11.0, 22.0, 33.0, 44.0]));
        cleanup_ctx(ctx);
    }
}

#[test]
fn launch_mul3_twice_reuses_kernel_cache() {
    // Same PTX pointer twice -> second launch hits main_kernel_cache.
    unsafe {
        let ctx = init_ctx();
        let buf = make_buf_with(ctx, &[1.0, 2.0, 3.0, 4.0]);
        let binds = pack_bind_ids(&[buf]);
        for _ in 0..2 {
            assert_eq!(
                cl_cuda_launch(ctx, PTX_MUL3.as_ptr(), 1, binds.as_ptr(), 1, 1, 1, 4, 1, 1),
                0
            );
        }
        assert_eq!(cl_cuda_sync(ctx as *const _), 0);
        // *3 *3 -> *9
        assert!(approx_eq(&download_f32(ctx, buf, 4), &[9.0, 18.0, 27.0, 36.0]));
        cleanup_ctx(ctx);
    }
}

#[test]
fn launch_named_add_one() {
    unsafe {
        let ctx = init_ctx();
        let buf = make_buf_with(ctx, &[1.0, 2.0, 3.0, 4.0]);
        let binds = pack_bind_ids(&[buf]);
        let rc = cl_cuda_launch_named(
            ctx,
            PTX_ADD_ONE_NAMED.as_ptr(),
            NAME_ADD_ONE.as_ptr(),
            1,
            binds.as_ptr(),
            1, 1, 1,
            4, 1, 1,
        );
        assert_eq!(rc, 0);
        assert_eq!(cl_cuda_sync(ctx as *const _), 0);
        assert!(approx_eq(&download_f32(ctx, buf, 4), &[2.0, 3.0, 4.0, 5.0]));
        cleanup_ctx(ctx);
    }
}

#[test]
fn launch_invalid_dimensions_returns_neg1() {
    unsafe {
        let ctx = init_ctx();
        let buf = make_buf_with(ctx, &[1.0]);
        let binds = pack_bind_ids(&[buf]);
        assert_eq!(
            cl_cuda_launch(ctx, PTX_MUL2.as_ptr(), 1, binds.as_ptr(), 0, 1, 1, 1, 1, 1),
            -1
        );
        assert_eq!(
            cl_cuda_launch(ctx, PTX_MUL2.as_ptr(), 1, binds.as_ptr(), 1, 1, 1, 1, 0, 1),
            -1
        );
        assert_eq!(
            cl_cuda_launch(ctx, PTX_MUL2.as_ptr(), -1, binds.as_ptr(), 1, 1, 1, 1, 1, 1),
            -1
        );
        cleanup_ctx(ctx);
    }
}

#[test]
fn launch_with_bad_buf_id_returns_neg1() {
    unsafe {
        let ctx = init_ctx();
        let binds = pack_bind_ids(&[999]);
        let rc = cl_cuda_launch(
            ctx, PTX_MUL2.as_ptr(), 1, binds.as_ptr(), 1, 1, 1, 1, 1, 1,
        );
        assert_eq!(rc, -1);
        cleanup_ctx(ctx);
    }
}

// ── Sync ─────────────────────────────────────────────────────────────────

#[test]
fn sync_after_init_succeeds() {
    unsafe {
        let ctx = init_ctx();
        assert_eq!(cl_cuda_sync(ctx as *const _), 0);
        cleanup_ctx(ctx);
    }
}

// ── Streams ──────────────────────────────────────────────────────────────

#[test]
fn stream_create_sync_destroy() {
    unsafe {
        let ctx = init_ctx();
        let s = cl_cuda_stream_create(ctx);
        assert!(s >= 0);
        assert_eq!(cl_cuda_stream_sync(ctx, s), 0);
        // Default stream (id < 0) syncs too.
        assert_eq!(cl_cuda_stream_sync(ctx, -1), 0);
        assert_eq!(cl_cuda_stream_destroy(ctx, s), 0);
        assert_eq!(cl_cuda_stream_destroy(ctx, s), -1);
        assert_eq!(cl_cuda_stream_destroy(ctx, 999), -1);
        assert_eq!(cl_cuda_stream_destroy(ctx, -1), -1);
        cleanup_ctx(ctx);
    }
}

#[test]
fn launch_on_stream_executes_and_syncs() {
    unsafe {
        let ctx = init_ctx();
        let s = cl_cuda_stream_create(ctx);
        assert!(s >= 0);
        let buf = make_buf_with(ctx, &[1.0, 2.0, 3.0, 4.0]);
        let binds = pack_bind_ids(&[buf]);
        assert_eq!(
            cl_cuda_launch_on_stream(
                ctx, PTX_MUL2.as_ptr(), 1, binds.as_ptr(), 1, 1, 1, 4, 1, 1, s,
            ),
            0
        );
        assert_eq!(cl_cuda_stream_sync(ctx, s), 0);
        assert!(approx_eq(&download_f32(ctx, buf, 4), &[2.0, 4.0, 6.0, 8.0]));
        assert_eq!(cl_cuda_stream_destroy(ctx, s), 0);
        cleanup_ctx(ctx);
    }
}

#[test]
fn launch_named_on_stream() {
    unsafe {
        let ctx = init_ctx();
        let s = cl_cuda_stream_create(ctx);
        let buf = make_buf_with(ctx, &[1.0, 2.0, 3.0, 4.0]);
        let binds = pack_bind_ids(&[buf]);
        assert_eq!(
            cl_cuda_launch_named_on_stream(
                ctx,
                PTX_ADD_ONE_NAMED.as_ptr(),
                NAME_ADD_ONE.as_ptr(),
                1,
                binds.as_ptr(),
                1, 1, 1, 4, 1, 1, s,
            ),
            0
        );
        assert_eq!(cl_cuda_stream_sync(ctx, s), 0);
        assert!(approx_eq(&download_f32(ctx, buf, 4), &[2.0, 3.0, 4.0, 5.0]));
        assert_eq!(cl_cuda_stream_destroy(ctx, s), 0);
        cleanup_ctx(ctx);
    }
}

// ── Events ───────────────────────────────────────────────────────────────

#[test]
fn event_create_record_elapsed_destroy() {
    unsafe {
        let ctx = init_ctx();
        let e0 = cl_cuda_event_create(ctx);
        let e1 = cl_cuda_event_create(ctx);
        assert!(e0 >= 0 && e1 >= 0 && e0 != e1);

        assert_eq!(cl_cuda_event_record(ctx, e0, -1), 0);
        let buf = make_buf_with(ctx, &[1.0, 2.0, 3.0, 4.0]);
        let binds = pack_bind_ids(&[buf]);
        assert_eq!(
            cl_cuda_launch(ctx, PTX_MUL2.as_ptr(), 1, binds.as_ptr(), 1, 1, 1, 4, 1, 1),
            0
        );
        assert_eq!(cl_cuda_event_record(ctx, e1, -1), 0);
        assert_eq!(cl_cuda_sync(ctx as *const _), 0);

        let ms_bits = cl_cuda_event_elapsed_ms_bits(ctx, e0, e1);
        assert_ne!(ms_bits, -1);
        let ms = f32::from_bits(ms_bits as u32);
        assert!(ms >= 0.0 && ms < 5000.0, "elapsed ms = {ms}");

        assert_eq!(cl_cuda_event_destroy(ctx, e0), 0);
        assert_eq!(cl_cuda_event_destroy(ctx, e1), 0);
        assert_eq!(cl_cuda_event_destroy(ctx, e0), -1);
        assert_eq!(cl_cuda_event_destroy(ctx, 999), -1);
        assert_eq!(cl_cuda_event_destroy(ctx, -1), -1);
        cleanup_ctx(ctx);
    }
}

#[test]
fn elapsed_on_unrecorded_event_returns_neg1() {
    unsafe {
        let ctx = init_ctx();
        let e0 = cl_cuda_event_create(ctx);
        let e1 = cl_cuda_event_create(ctx);
        let rc = cl_cuda_event_elapsed_ms_bits(ctx, e0, e1);
        assert_eq!(rc, -1);
        cleanup_ctx(ctx);
    }
}

#[test]
fn stream_wait_event_synchronizes_streams() {
    unsafe {
        let ctx = init_ctx();
        let s0 = cl_cuda_stream_create(ctx);
        let s1 = cl_cuda_stream_create(ctx);
        let e = cl_cuda_event_create(ctx);
        assert!(s0 >= 0 && s1 >= 0 && e >= 0);

        let buf = make_buf_with(ctx, &[1.0, 2.0, 3.0, 4.0]);
        let binds = pack_bind_ids(&[buf]);
        assert_eq!(
            cl_cuda_launch_on_stream(
                ctx, PTX_MUL2.as_ptr(), 1, binds.as_ptr(), 1, 1, 1, 4, 1, 1, s0,
            ),
            0
        );
        assert_eq!(cl_cuda_event_record(ctx, e, s0), 0);
        assert_eq!(cl_cuda_stream_wait_event(ctx, s1, e), 0);
        assert_eq!(
            cl_cuda_launch_on_stream(
                ctx, PTX_MUL3.as_ptr(), 1, binds.as_ptr(), 1, 1, 1, 4, 1, 1, s1,
            ),
            0
        );
        assert_eq!(cl_cuda_stream_sync(ctx, s1), 0);
        assert!(approx_eq(
            &download_f32(ctx, buf, 4),
            &[6.0, 12.0, 18.0, 24.0]
        ));

        assert_eq!(cl_cuda_event_destroy(ctx, e), 0);
        assert_eq!(cl_cuda_stream_destroy(ctx, s0), 0);
        assert_eq!(cl_cuda_stream_destroy(ctx, s1), 0);
        cleanup_ctx(ctx);
    }
}

// ── Graphs ───────────────────────────────────────────────────────────────

#[test]
fn graph_capture_and_launch() {
    unsafe {
        let ctx = init_ctx();
        let s = cl_cuda_stream_create(ctx);
        let buf = make_buf_with(ctx, &[1.0, 2.0, 3.0, 4.0]);
        let binds = pack_bind_ids(&[buf]);

        assert_eq!(cl_cuda_graph_begin_capture(ctx, s), 0);
        assert_eq!(
            cl_cuda_launch_on_stream(
                ctx, PTX_MUL2.as_ptr(), 1, binds.as_ptr(), 1, 1, 1, 4, 1, 1, s,
            ),
            0
        );
        let g = cl_cuda_graph_end_capture(ctx, s);
        assert!(g >= 0);

        assert_eq!(cl_cuda_graph_upload(ctx, g, s), 0);
        assert_eq!(cl_cuda_graph_launch(ctx, g, s), 0);
        assert_eq!(cl_cuda_stream_sync(ctx, s), 0);
        assert!(approx_eq(&download_f32(ctx, buf, 4), &[2.0, 4.0, 6.0, 8.0]));

        assert_eq!(cl_cuda_graph_launch(ctx, g, s), 0);
        assert_eq!(cl_cuda_stream_sync(ctx, s), 0);
        assert!(approx_eq(&download_f32(ctx, buf, 4), &[4.0, 8.0, 12.0, 16.0]));

        assert_eq!(cl_cuda_graph_destroy(ctx, g), 0);
        assert_eq!(cl_cuda_graph_destroy(ctx, g), -1);
        assert_eq!(cl_cuda_graph_destroy(ctx, 999), -1);
        assert_eq!(cl_cuda_graph_destroy(ctx, -1), -1);
        assert_eq!(cl_cuda_stream_destroy(ctx, s), 0);
        cleanup_ctx(ctx);
    }
}

#[test]
fn graph_launch_invalid_handle_returns_neg1() {
    unsafe {
        let ctx = init_ctx();
        let s = cl_cuda_stream_create(ctx);
        assert_eq!(cl_cuda_graph_launch(ctx, 999, s), -1);
        assert_eq!(cl_cuda_graph_upload(ctx, 999, s), -1);
        assert_eq!(cl_cuda_graph_launch(ctx, -1, s), -1);
        assert_eq!(cl_cuda_stream_destroy(ctx, s), 0);
        cleanup_ctx(ctx);
    }
}

// ── Pinned memory ────────────────────────────────────────────────────────

#[test]
fn pinned_alloc_ptr_free() {
    unsafe {
        let ctx = init_ctx();
        let p = cl_cuda_pinned_alloc(ctx, 64);
        assert!(p >= 0);
        let host_ptr = cl_cuda_pinned_ptr(ctx, p);
        assert!(host_ptr > 0);

        let host = host_ptr as *mut f32;
        for i in 0..16 {
            *host.add(i) = (i + 1) as f32;
        }
        for i in 0..16 {
            assert_eq!(*host.add(i), (i + 1) as f32);
        }

        assert_eq!(cl_cuda_pinned_free(ctx, p), 0);
        assert_eq!(cl_cuda_pinned_free(ctx, p), -1);
        assert_eq!(cl_cuda_pinned_free(ctx, 999), -1);
        assert_eq!(cl_cuda_pinned_alloc(ctx, 0), -1);
        assert_eq!(cl_cuda_pinned_alloc(ctx, -8), -1);
        assert_eq!(cl_cuda_pinned_ptr(ctx, 999), -1);
        assert_eq!(cl_cuda_pinned_ptr(ctx, -1), -1);
        cleanup_ctx(ctx);
    }
}

// ── Async copies ─────────────────────────────────────────────────────────

#[test]
fn upload_async_download_async_roundtrip() {
    unsafe {
        let ctx = init_ctx();
        let s = cl_cuda_stream_create(ctx);
        let data = [11.0f32, 22.0, 33.0, 44.0];
        let bytes = f32s_to_bytes(&data);
        let buf = cl_cuda_create_buffer(ctx, bytes.len() as i64);
        assert_eq!(
            cl_cuda_upload_ptr_async(ctx, buf, bytes.as_ptr(), bytes.len() as i64, s),
            0
        );
        let mut out = vec![0u8; bytes.len()];
        assert_eq!(
            cl_cuda_download_ptr_async(ctx, buf, out.as_mut_ptr(), out.len() as i64, s),
            0
        );
        assert_eq!(cl_cuda_stream_sync(ctx, s), 0);
        assert!(approx_eq(&bytes_to_f32s(&out), &data));
        assert_eq!(cl_cuda_stream_destroy(ctx, s), 0);
        cleanup_ctx(ctx);
    }
}

#[test]
fn upload_offset_async_and_bounds_errors() {
    unsafe {
        let ctx = init_ctx();
        let s = cl_cuda_stream_create(ctx);
        let buf = cl_cuda_create_buffer(ctx, 32);
        let zeros = vec![0u8; 32];
        assert_eq!(cl_cuda_upload(ctx, buf, zeros.as_ptr(), 32), 0);

        let part = f32s_to_bytes(&[7.0, 8.0]);
        assert_eq!(
            cl_cuda_upload_ptr_offset_async(
                ctx, buf, 16, part.as_ptr(), part.len() as i64, s,
            ),
            0
        );
        // 28+8 > 32 → bounds error
        assert_eq!(
            cl_cuda_upload_ptr_offset_async(
                ctx, buf, 28, part.as_ptr(), part.len() as i64, s,
            ),
            -1
        );
        // negative offset
        assert_eq!(
            cl_cuda_upload_ptr_offset_async(
                ctx, buf, -1, part.as_ptr(), part.len() as i64, s,
            ),
            -1
        );

        assert_eq!(cl_cuda_stream_sync(ctx, s), 0);
        let got = download_f32(ctx, buf, 8);
        assert!(approx_eq(&got, &[0.0, 0.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0]));
        assert_eq!(cl_cuda_stream_destroy(ctx, s), 0);
        cleanup_ctx(ctx);
    }
}

// ── cuBLAS ───────────────────────────────────────────────────────────────

#[test]
fn cublas_sgemv_column_vector() {
    // A is 4×1 column [1,2,3,4]; x = [2.0]; y = A*x = [2,4,6,8].
    unsafe {
        let ctx = init_ctx();
        let a_buf = make_buf_with(ctx, &[1.0, 2.0, 3.0, 4.0]);
        let x_buf = make_buf_with(ctx, &[2.0]);
        let y_buf = make_buf_with(ctx, &[0.0, 0.0, 0.0, 0.0]);
        let alpha = 1.0f32.to_bits() as i32;
        let beta = 0.0f32.to_bits() as i32;
        let rc = cl_cublas_sgemv(ctx, 0, 4, 1, alpha, a_buf, x_buf, beta, y_buf);
        assert_eq!(rc, 0);
        assert_eq!(cl_cuda_sync(ctx as *const _), 0);
        assert!(approx_eq(&download_f32(ctx, y_buf, 4), &[2.0, 4.0, 6.0, 8.0]));
        cleanup_ctx(ctx);
    }
}

#[test]
fn cublas_sgemm_1x1() {
    unsafe {
        let ctx = init_ctx();
        let a = make_buf_with(ctx, &[2.0]);
        let b = make_buf_with(ctx, &[3.0]);
        let c = make_buf_with(ctx, &[0.0]);
        let alpha = 1.0f32.to_bits() as i32;
        let beta = 0.0f32.to_bits() as i32;
        let rc = cl_cublas_sgemm(ctx, 0, 0, 1, 1, 1, alpha, a, b, beta, c);
        assert_eq!(rc, 0);
        assert_eq!(cl_cuda_sync(ctx as *const _), 0);
        assert!(approx_eq(&download_f32(ctx, c, 1), &[6.0]));
        cleanup_ctx(ctx);
    }
}

#[test]
fn cublas_sgemm_strided_batched_elementwise() {
    // batch=2 of 1×1 GEMMs: C[i] = A[i]*B[i].
    unsafe {
        let ctx = init_ctx();
        let a = make_buf_with(ctx, &[2.0, 3.0]);
        let b = make_buf_with(ctx, &[4.0, 5.0]);
        let c = make_buf_with(ctx, &[0.0, 0.0]);
        let alpha = 1.0f32.to_bits() as i32;
        let beta = 0.0f32.to_bits() as i32;
        let rc = cl_cublas_sgemm_strided_batched(
            ctx, 0, 0, 1, 1, 1, alpha, a, 1, b, 1, beta, c, 1, 2, 0, 0, 0, 0, 0, 0,
        );
        assert_eq!(rc, 0);
        assert_eq!(cl_cuda_sync(ctx as *const _), 0);
        assert!(approx_eq(&download_f32(ctx, c, 2), &[8.0, 15.0]));
        cleanup_ctx(ctx);
    }
}

/// **Differential: a batch of one against a plain GEMM.**
///
/// Every contraction this repository's ViT issues goes through
/// `cl_cublas_sgemm_strided_batched`, and the question this settles is
/// whether that is the same arithmetic as `cl_cublas_sgemm` — i.e. whether
/// a stronger law could be claimed by moving the call.  The fifteen shapes
/// are the fifteen the model actually launches, at `batch_count = 1`,
/// compared bit for bit.
#[test]
fn cublas_gemm_batch_one_vs_plain() {
    const SHAPES: [(i32, i32, i32, i32, i32); 15] = [
        (1, 0, 192, 200, 192), (1, 0, 224, 200, 64),  (0, 0, 64, 200, 224),
        (1, 0, 768, 200, 192), (1, 0, 192, 200, 768), (1, 0, 128, 200, 192),
        (0, 1, 192, 128, 200), (0, 0, 192, 200, 128), (0, 1, 768, 192, 200),
        (0, 0, 768, 200, 192), (0, 1, 192, 768, 200), (0, 0, 192, 200, 768),
        (0, 1, 192, 192, 200), (0, 0, 192, 200, 192), (0, 1, 64, 224, 200),
    ];
    // Deterministic, and spread over enough of the exponent range that a
    // different summation order shows up in the low bits.
    fn val(i: usize) -> f32 {
        let x = ((i as u32).wrapping_mul(2654435761) >> 8) as f32 / 8388608.0;
        x - 1.0
    }
    unsafe {
        let ctx = init_ctx();
        let alpha = 1.0f32.to_bits() as i32;
        let beta = 0.0f32.to_bits() as i32;
        let mut differing = Vec::new();
        for (ta, tb, m, n, k) in SHAPES {
            let a: Vec<f32> = (0..(m * k) as usize).map(val).collect();
            let b: Vec<f32> = (0..(k * n) as usize).map(|i| val(i + 7)).collect();
            let zero = vec![0.0f32; (m * n) as usize];
            let a_buf = make_buf_with(ctx, &a);
            let b_buf = make_buf_with(ctx, &b);
            let c1 = make_buf_with(ctx, &zero);
            let c2 = make_buf_with(ctx, &zero);
            assert_eq!(
                cl_cublas_sgemm(ctx, ta, tb, m, n, k, alpha, a_buf, b_buf, beta, c1),
                0
            );
            assert_eq!(
                cl_cublas_sgemm_strided_batched(
                    ctx, ta, tb, m, n, k, alpha, a_buf, 0, b_buf, 0, beta, c2, 0, 1,
                    0, 0, 0, 0, 0, 0,
                ),
                0
            );
            assert_eq!(cl_cuda_sync(ctx as *const _), 0);
            let g1 = download_f32(ctx, c1, (m * n) as usize);
            let g2 = download_f32(ctx, c2, (m * n) as usize);
            let bad = g1
                .iter()
                .zip(g2.iter())
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            if bad > 0 {
                let worst = g1
                    .iter()
                    .zip(g2.iter())
                    .map(|(x, y)| (x - y).abs() / x.abs().max(1e-30))
                    .fold(0.0f32, f32::max);
                differing.push(((ta, tb, m, n, k), bad, g1.len(), worst));
            }
        }
        for (shape, bad, total, worst) in &differing {
            eprintln!(
                "shape {:?}: {}/{} elements differ, worst relative {:e}",
                shape, bad, total, worst
            );
        }
        assert!(
            differing.is_empty(),
            "{} of {} shapes are not bit-identical",
            differing.len(),
            SHAPES.len()
        );
        cleanup_ctx(ctx);
    }
}

#[test]
fn cublas_sgemv_on_stream() {
    unsafe {
        let ctx = init_ctx();
        let s = cl_cuda_stream_create(ctx);
        let a = make_buf_with(ctx, &[1.0, 2.0, 3.0, 4.0]);
        let x = make_buf_with(ctx, &[2.0]);
        let y = make_buf_with(ctx, &[0.0, 0.0, 0.0, 0.0]);
        let alpha = 1.0f32.to_bits() as i32;
        let beta = 0.0f32.to_bits() as i32;
        let rc = cl_cublas_sgemv_on_stream(ctx, 0, 4, 1, alpha, a, x, beta, y, s);
        assert_eq!(rc, 0);
        assert_eq!(cl_cuda_stream_sync(ctx, s), 0);
        assert!(approx_eq(&download_f32(ctx, y, 4), &[2.0, 4.0, 6.0, 8.0]));
        assert_eq!(cl_cuda_stream_destroy(ctx, s), 0);
        cleanup_ctx(ctx);
    }
}

#[test]
fn cublas_sgemm_batched_pointer_arrays() {
    // batch=2 of 1x1 GEMMs whose operands live in *separate* allocations:
    // what the pointer-array form buys over the strided one.
    unsafe {
        let ctx = init_ctx();
        let s = cl_cuda_stream_create(ctx);
        let a0 = make_buf_with(ctx, &[2.0]);
        let a1 = make_buf_with(ctx, &[3.0]);
        let b0 = make_buf_with(ctx, &[4.0]);
        let b1 = make_buf_with(ctx, &[5.0]);
        let c0 = make_buf_with(ctx, &[0.0]);
        let c1 = make_buf_with(ctx, &[0.0]);
        let aa = cl_cuda_create_buffer(ctx, 16);
        let ba = cl_cuda_create_buffer(ctx, 16);
        let ca = cl_cuda_create_buffer(ctx, 16);
        for (arr, bufs) in [(aa, [a0, a1]), (ba, [b0, b1]), (ca, [c0, c1])] {
            for (slot, buf) in bufs.iter().enumerate() {
                assert_eq!(cl_cublas_ptr_array(ctx, arr, slot as i32, *buf, 0), 0);
            }
        }
        let alpha = 1.0f32.to_bits() as i32;
        let beta = 0.0f32.to_bits() as i32;
        let rc = cl_cublas_sgemm_batched_on_stream(
            ctx, 0, 0, 1, 1, 1, alpha, aa, ba, beta, ca, 2, s,
        );
        assert_eq!(rc, 0);
        assert_eq!(cl_cuda_stream_sync(ctx, s), 0);
        assert!(approx_eq(&download_f32(ctx, c0, 1), &[8.0]));
        assert!(approx_eq(&download_f32(ctx, c1, 1), &[15.0]));

        // An array too small for the slot is refused rather than written past.
        let tiny = cl_cuda_create_buffer(ctx, 8);
        assert_eq!(cl_cublas_ptr_array(ctx, tiny, 1, a0, 0), -1);
        assert_eq!(cl_cublas_ptr_array(ctx, tiny, 0, 999, 0), -1);

        assert_eq!(cl_cuda_stream_destroy(ctx, s), 0);
        cleanup_ctx(ctx);
    }
}

#[test]
fn cublas_sgemm_strided_batched_on_stream() {
    unsafe {
        let ctx = init_ctx();
        let s = cl_cuda_stream_create(ctx);
        let a = make_buf_with(ctx, &[2.0, 3.0]);
        let b = make_buf_with(ctx, &[4.0, 5.0]);
        let c = make_buf_with(ctx, &[0.0, 0.0]);
        let alpha = 1.0f32.to_bits() as i32;
        let beta = 0.0f32.to_bits() as i32;
        let rc = cl_cublas_sgemm_strided_batched_on_stream(
            ctx, 0, 0, 1, 1, 1, alpha, a, 1, b, 1, beta, c, 1, 2, s, 0, 0, 0, 0, 0, 0,
        );
        assert_eq!(rc, 0);
        assert_eq!(cl_cuda_stream_sync(ctx, s), 0);
        assert!(approx_eq(&download_f32(ctx, c, 2), &[8.0, 15.0]));
        assert_eq!(cl_cuda_stream_destroy(ctx, s), 0);
        cleanup_ctx(ctx);
    }
}

#[test]
fn cublas_invalid_buf_returns_neg1() {
    unsafe {
        let ctx = init_ctx();
        let alpha = 1.0f32.to_bits() as i32;
        let beta = 0.0f32.to_bits() as i32;
        assert_eq!(
            cl_cublas_sgemv(ctx, 0, 1, 1, alpha, 999, 999, beta, 999),
            -1
        );
        assert_eq!(
            cl_cublas_sgemm(ctx, 0, 0, 1, 1, 1, alpha, 999, 999, beta, 999),
            -1
        );
        assert_eq!(
            cl_cublas_sgemm_strided_batched(
                ctx, 0, 0, 1, 1, 1, alpha, 999, 1, 999, 1, beta, 999, 1, 1, 0, 0, 0, 0, 0, 0
            ),
            -1
        );
        // Invalid dims rejected upfront.
        assert_eq!(
            cl_cublas_sgemm_strided_batched(
                ctx, 0, 0, 0, 1, 1, alpha, 0, 1, 0, 1, beta, 0, 1, 1, 0, 0, 0, 0, 0, 0
            ),
            -1
        );
        assert_eq!(
            cl_cublas_sgemm_strided_batched(
                ctx, 0, 0, 1, 1, 1, alpha, 0, 1, 0, 1, beta, 0, 1, 0, 0, 0, 0, 0, 0, 0
            ),
            -1
        );
        cleanup_ctx(ctx);
    }
}

// ── Null ctx / general error paths ───────────────────────────────────────

#[test]
fn null_ctx_returns_neg1() {
    let null_ctx = std::ptr::null_mut::<CraneliftCudaContext>();
    let binds = pack_bind_ids(&[0]);
    unsafe {
        assert_eq!(cl_cuda_create_buffer(null_ctx, 64), -1);
        assert_eq!(cl_cuda_upload(null_ctx, 0, [0u8; 4].as_ptr(), 4), -1);
        assert_eq!(cl_cuda_upload_ptr(null_ctx, 0, [0u8; 4].as_ptr(), 4), -1);
        assert_eq!(
            cl_cuda_upload_ptr_offset(null_ctx, 0, 0, [0u8; 4].as_ptr(), 4),
            -1
        );
        let mut buf = [0u8; 4];
        assert_eq!(cl_cuda_download(null_ctx, 0, buf.as_mut_ptr(), 4), -1);
        assert_eq!(cl_cuda_download_ptr(null_ctx, 0, buf.as_mut_ptr(), 4), -1);
        assert_eq!(
            cl_cuda_download_ptr_offset(null_ctx, 0, 0, buf.as_mut_ptr(), 4),
            -1
        );
        assert_eq!(cl_cuda_free_buffer(null_ctx, 0), -1);
        assert_eq!(
            cl_cuda_launch(
                null_ctx, PTX_MUL2.as_ptr(), 1, binds.as_ptr(), 1, 1, 1, 1, 1, 1
            ),
            -1
        );
        assert_eq!(cl_cuda_sync(null_ctx as *const _), -1);
        assert_eq!(cl_cuda_stream_create(null_ctx), -1);
        assert_eq!(cl_cuda_stream_sync(null_ctx, 0), -1);
        assert_eq!(cl_cuda_stream_destroy(null_ctx, 0), -1);
        assert_eq!(cl_cuda_event_create(null_ctx), -1);
        assert_eq!(cl_cuda_event_record(null_ctx, 0, 0), -1);
        assert_eq!(cl_cuda_event_destroy(null_ctx, 0), -1);
        assert_eq!(cl_cuda_pinned_alloc(null_ctx, 16), -1);
        assert_eq!(cl_cuda_pinned_ptr(null_ctx, 0), -1);
        assert_eq!(cl_cuda_pinned_free(null_ctx, 0), -1);
        let alpha = 1.0f32.to_bits() as i32;
        let beta = 0.0f32.to_bits() as i32;
        assert_eq!(
            cl_cublas_sgemv(null_ctx, 0, 1, 1, alpha, 0, 0, beta, 0),
            -1
        );
    }
}

#[test]
fn cleanup_on_null_slot_is_noop() {
    let mut null_slot: *mut CraneliftCudaContext = std::ptr::null_mut();
    unsafe { cl_cuda_cleanup(&mut null_slot) };
    assert!(null_slot.is_null());
}
