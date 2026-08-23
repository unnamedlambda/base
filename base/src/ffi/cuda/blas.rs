//! cuBLAS: the vendor calls this development does not emit kernels for.
use super::shared::*;
use crate::ffi::{read_ctx_mut};

fn ensure_default_cuda_blas<'a>(
    ctx: &CraneliftCudaContext,
    state: &'a mut CraneliftCudaState,
) -> Result<&'a cudarc::cublas::CudaBlas, i32> {
    if state.default_blas.is_none() {
        match cudarc::cublas::CudaBlas::new(ctx.device.clone()) {
            Ok(b) => state.default_blas = Some(b),
            Err(e) => {
                eprintln!("CUDA BLAS init failed: {:?}", e);
                return Err(-1);
            }
        }
    }
    Ok(state.default_blas.as_ref().unwrap())
}

fn set_cuda_blas_stream(
    blas: &cudarc::cublas::CudaBlas,
    stream: cudarc::driver::sys::CUstream,
) -> Result<(), i32> {
    match unsafe { cudarc::cublas::result::set_stream(*blas.handle(), stream as *mut _) } {
        Ok(_) => Ok(()),
        Err(e) => {
            eprintln!("cuBLAS set_stream failed: {:?}", e);
            Err(-1)
        }
    }
}

fn ensure_stream_cuda_blas<'a>(
    ctx: &CraneliftCudaContext,
    state: &'a mut CraneliftCudaState,
    stream_id: i32,
    stream: cudarc::driver::sys::CUstream,
) -> Result<&'a cudarc::cublas::CudaBlas, i32> {
    if let std::collections::hash_map::Entry::Vacant(entry) = state.stream_blas.entry(stream_id) {
        let blas = match cudarc::cublas::CudaBlas::new(ctx.device.clone()) {
            Ok(blas) => blas,
            Err(e) => {
                eprintln!("CUDA BLAS init failed: {:?}", e);
                return Err(-1);
            }
        };
        if let Err(rc) = set_cuda_blas_stream(&blas, stream) {
            return Err(rc);
        }
        entry.insert(blas);
    }
    Ok(state.stream_blas.get(&stream_id).unwrap())
}

/// cuBLAS SGEMM: C = alpha * op(A) * op(B) + beta * C
///
/// - `transa`: 0 = NoTrans, 1 = Trans (for A)
/// - `transb`: 0 = NoTrans, 1 = Trans (for B)
/// - `m`, `n`, `k`: dimensions of op(A) = m×k, op(B) = k×n, C = m×n
/// - `alpha_bits`, `beta_bits`: f32 scalars reinterpreted as i32 bits
/// - `a_buf`, `b_buf`, `c_buf`: buffer IDs (from cl_cuda_create_buffer)
///
/// Leading dimensions follow standard cuBLAS column-major rules:
///   lda = k if transa else m,  ldb = n if transb else k,  ldc = m
pub(crate) unsafe extern "C" fn cl_cublas_sgemm(
    ctx_ptr: *mut CraneliftCudaContext,
    transa: i32,
    transb: i32,
    m: i32,
    n: i32,
    k: i32,
    alpha_bits: i32,
    a_buf: i32,
    b_buf: i32,
    beta_bits: i32,
    c_buf: i32,
) -> i32 {
    use cudarc::cublas::sys::cublasOperation_t;

    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };

        let alpha = f32::from_bits(alpha_bits as u32);
        let beta = f32::from_bits(beta_bits as u32);
        let op_a = if transa != 0 {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let op_b = if transb != 0 {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let lda = if transa != 0 { k } else { m };
        let ldb = if transb != 0 { n } else { k };
        let ldc = m;

        let a_dev = match unsafe { cuda_buffer_device_ptr(&state, a_buf) } {
            Some(p) => p,
            None => return -1,
        };
        let b_dev = match unsafe { cuda_buffer_device_ptr(&state, b_buf) } {
            Some(p) => p,
            None => return -1,
        };
        let c_dev = match unsafe { cuda_buffer_device_ptr(&state, c_buf) } {
            Some(p) => p,
            None => return -1,
        };
        let blas = match ensure_default_cuda_blas(ctx, &mut state) {
            Ok(blas) => blas,
            Err(rc) => return rc,
        };

        if let Err(e) = unsafe {
            cudarc::cublas::result::sgemm(
                *blas.handle(),
                op_a,
                op_b,
                m,
                n,
                k,
                &alpha,
                a_dev as *const f32,
                lda,
                b_dev as *const f32,
                ldb,
                &beta,
                c_dev as *mut f32,
                ldc,
            )
        } {
            eprintln!("cl_cublas_sgemm: sgemm failed: {:?}", e);
            return -1;
        }

        0
    }))
    .unwrap_or(-1)
}

/// cuBLAS SGEMV: y = alpha * op(A) * x + beta * y
///
/// - `trans`: 0 = NoTrans, 1 = Trans
/// - `m`, `n`: dimensions of A in column-major form
/// - `alpha_bits`, `beta_bits`: f32 scalars reinterpreted as i32 bits
/// - `a_buf`, `x_buf`, `y_buf`: buffer IDs
///
/// For row-major weights shaped [rows, cols], call with `trans=1`, `m=cols`, `n=rows`.
pub(crate) unsafe extern "C" fn cl_cublas_sgemv(
    ctx_ptr: *mut CraneliftCudaContext,
    trans: i32,
    m: i32,
    n: i32,
    alpha_bits: i32,
    a_buf: i32,
    x_buf: i32,
    beta_bits: i32,
    y_buf: i32,
) -> i32 {
    use cudarc::cublas::sys::cublasOperation_t;

    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };

        let alpha = f32::from_bits(alpha_bits as u32);
        let beta = f32::from_bits(beta_bits as u32);
        let op = if trans != 0 {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let lda = m;

        let a_dev = match unsafe { cuda_buffer_device_ptr(&state, a_buf) } {
            Some(p) => p,
            None => return -1,
        };
        let x_dev = match unsafe { cuda_buffer_device_ptr(&state, x_buf) } {
            Some(p) => p,
            None => return -1,
        };
        let y_dev = match unsafe { cuda_buffer_device_ptr(&state, y_buf) } {
            Some(p) => p,
            None => return -1,
        };
        let blas = match ensure_default_cuda_blas(ctx, &mut state) {
            Ok(blas) => blas,
            Err(rc) => return rc,
        };

        if let Err(e) = unsafe {
            cudarc::cublas::result::sgemv(
                *blas.handle(),
                op,
                m,
                n,
                &alpha,
                a_dev as *const f32,
                lda,
                x_dev as *const f32,
                1,
                &beta,
                y_dev as *mut f32,
                1,
            )
        } {
            eprintln!("cl_cublas_sgemv: sgemv failed: {:?}", e);
            return -1;
        }

        0
    }))
    .unwrap_or(-1)
}

/// cuBLAS SGEMM strided batched: C_i = alpha * op(A_i) * op(B_i) + beta * C_i
///
/// - `transa`: 0 = NoTrans, 1 = Trans
/// - `transb`: 0 = NoTrans, 1 = Trans
/// - `m`, `n`, `k`: dimensions of op(A) = m×k, op(B) = k×n, C = m×n
/// - `stride_a`, `stride_b`, `stride_c`: batch strides in number of f32 elements
/// - `batch_count`: number of matrices/vectors in the batch
/// - `alpha_bits`, `beta_bits`: f32 scalars reinterpreted as i32 bits
///
/// For row-major matrices stored as contiguous `[rows, cols]`, use the same
/// transpose conventions as `cl_cublas_sgemv` / `cl_cublas_sgemm`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe extern "C" fn cl_cublas_sgemm_strided_batched(
    ctx_ptr: *mut CraneliftCudaContext,
    transa: i32,
    transb: i32,
    m: i32,
    n: i32,
    k: i32,
    alpha_bits: i32,
    a_buf: i32,
    stride_a: i64,
    b_buf: i32,
    stride_b: i64,
    beta_bits: i32,
    c_buf: i32,
    stride_c: i64,
    batch_count: i32,
    off_a: i64,
    off_b: i64,
    off_c: i64,
    ld_a: i32,
    ld_b: i32,
    ld_c: i32,
) -> i32 {
    use cudarc::cublas::sys::cublasOperation_t;

    if m <= 0
        || n <= 0
        || k <= 0
        || stride_a < 0
        || stride_b < 0
        || stride_c < 0
        || batch_count <= 0
        || off_a < 0
        || off_b < 0
        || off_c < 0
        || ld_a < 0
        || ld_b < 0
        || ld_c < 0
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

        let alpha = f32::from_bits(alpha_bits as u32);
        let beta = f32::from_bits(beta_bits as u32);
        let op_a = if transa != 0 {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let op_b = if transb != 0 {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        // A leading dimension of zero means "the one this shape implies".  A
        // non-zero one says the operand is a *column slice* of a wider matrix:
        // the rows it contracts are unchanged, they are just further apart, so
        // like the offsets this costs no law.
        let lda = if ld_a != 0 { ld_a } else if transa != 0 { k } else { m };
        let ldb = if ld_b != 0 { ld_b } else if transb != 0 { n } else { k };
        let ldc = if ld_c != 0 { ld_c } else { m };

        // An operand may name a slice of its buffer: the offset moves the
        // pointer and leaves the matrix it contracts alone, which is why this
        // costs no law.  Counted in f32 elements, as `stride_a` already is.
        let a_dev = match unsafe { cuda_buffer_device_ptr(&state, a_buf) } {
            Some(p) => p + (off_a as u64) * 4,
            None => return -1,
        };
        let b_dev = match unsafe { cuda_buffer_device_ptr(&state, b_buf) } {
            Some(p) => p + (off_b as u64) * 4,
            None => return -1,
        };
        let c_dev = match unsafe { cuda_buffer_device_ptr(&state, c_buf) } {
            Some(p) => p + (off_c as u64) * 4,
            None => return -1,
        };
        let blas = match ensure_default_cuda_blas(ctx, &mut state) {
            Ok(blas) => blas,
            Err(rc) => return rc,
        };

        if let Err(e) = unsafe {
            cudarc::cublas::result::sgemm_strided_batched(
                *blas.handle(),
                op_a,
                op_b,
                m,
                n,
                k,
                &alpha,
                a_dev as *const f32,
                lda,
                stride_a,
                b_dev as *const f32,
                ldb,
                stride_b,
                &beta,
                c_dev as *mut f32,
                ldc,
                stride_c,
                batch_count,
            )
        } {
            // Name the shape as well as the status: a cuBLAS status alone does
            // not say whether the fault is the handle or the arguments.
            eprintln!(
                "cl_cublas_sgemm_strided_batched: sgemm failed: {:?} \
                 (transa={} transb={} m={} n={} k={} lda={} ldb={} ldc={} \
                 batch={} bufs a={} b={} c={})",
                e, transa, transb, m, n, k, lda, ldb, ldc, batch_count,
                a_buf, b_buf, c_buf
            );
            return -1;
        }

        0
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cublas_sgemv_on_stream(
    ctx_ptr: *mut CraneliftCudaContext,
    trans: i32,
    m: i32,
    n: i32,
    alpha_bits: i32,
    a_buf: i32,
    x_buf: i32,
    beta_bits: i32,
    y_buf: i32,
    stream_id: i32,
) -> i32 {
    use cudarc::cublas::sys::cublasOperation_t;

    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };

        let alpha = f32::from_bits(alpha_bits as u32);
        let beta = f32::from_bits(beta_bits as u32);
        let op = if trans != 0 {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let lda = m;

        let a_dev = match unsafe { cuda_buffer_device_ptr(&state, a_buf) } {
            Some(p) => p,
            None => return -1,
        };
        let x_dev = match unsafe { cuda_buffer_device_ptr(&state, x_buf) } {
            Some(p) => p,
            None => return -1,
        };
        let y_dev = match unsafe { cuda_buffer_device_ptr(&state, y_buf) } {
            Some(p) => p,
            None => return -1,
        };
        let blas = match ensure_stream_cuda_blas(ctx, &mut state, stream_id, stream) {
            Ok(blas) => blas,
            Err(rc) => return rc,
        };

        if let Err(e) = unsafe {
            cudarc::cublas::result::sgemv(
                *blas.handle(),
                op,
                m,
                n,
                &alpha,
                a_dev as *const f32,
                lda,
                x_dev as *const f32,
                1,
                &beta,
                y_dev as *mut f32,
                1,
            )
        } {
            eprintln!("cl_cublas_sgemv_on_stream: sgemv failed: {:?}", e);
            return -1;
        }

        0
    }))
    .unwrap_or(-1)
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe extern "C" fn cl_cublas_sgemm_strided_batched_on_stream(
    ctx_ptr: *mut CraneliftCudaContext,
    transa: i32,
    transb: i32,
    m: i32,
    n: i32,
    k: i32,
    alpha_bits: i32,
    a_buf: i32,
    stride_a: i64,
    b_buf: i32,
    stride_b: i64,
    beta_bits: i32,
    c_buf: i32,
    stride_c: i64,
    batch_count: i32,
    stream_id: i32,
    off_a: i64,
    off_b: i64,
    off_c: i64,
    ld_a: i32,
    ld_b: i32,
    ld_c: i32,
) -> i32 {
    use cudarc::cublas::sys::cublasOperation_t;

    if m <= 0
        || n <= 0
        || k <= 0
        || stride_a < 0
        || stride_b < 0
        || stride_c < 0
        || batch_count <= 0
        || off_a < 0
        || off_b < 0
        || off_c < 0
        || ld_a < 0
        || ld_b < 0
        || ld_c < 0
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
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };

        let alpha = f32::from_bits(alpha_bits as u32);
        let beta = f32::from_bits(beta_bits as u32);
        let op_a = if transa != 0 {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let op_b = if transb != 0 {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        // A leading dimension of zero means "the one this shape implies".  A
        // non-zero one says the operand is a *column slice* of a wider matrix:
        // the rows it contracts are unchanged, they are just further apart, so
        // like the offsets this costs no law.
        let lda = if ld_a != 0 { ld_a } else if transa != 0 { k } else { m };
        let ldb = if ld_b != 0 { ld_b } else if transb != 0 { n } else { k };
        let ldc = if ld_c != 0 { ld_c } else { m };

        // An operand may name a slice of its buffer: the offset moves the
        // pointer and leaves the matrix it contracts alone, which is why this
        // costs no law.  Counted in f32 elements, as `stride_a` already is.
        let a_dev = match unsafe { cuda_buffer_device_ptr(&state, a_buf) } {
            Some(p) => p + (off_a as u64) * 4,
            None => return -1,
        };
        let b_dev = match unsafe { cuda_buffer_device_ptr(&state, b_buf) } {
            Some(p) => p + (off_b as u64) * 4,
            None => return -1,
        };
        let c_dev = match unsafe { cuda_buffer_device_ptr(&state, c_buf) } {
            Some(p) => p + (off_c as u64) * 4,
            None => return -1,
        };
        let blas = match ensure_stream_cuda_blas(ctx, &mut state, stream_id, stream) {
            Ok(blas) => blas,
            Err(rc) => return rc,
        };

        if let Err(e) = unsafe {
            cudarc::cublas::result::sgemm_strided_batched(
                *blas.handle(),
                op_a,
                op_b,
                m,
                n,
                k,
                &alpha,
                a_dev as *const f32,
                lda,
                stride_a,
                b_dev as *const f32,
                ldb,
                stride_b,
                &beta,
                c_dev as *mut f32,
                ldc,
                stride_c,
                batch_count,
            )
        } {
            eprintln!(
                "cl_cublas_sgemm_strided_batched_on_stream: sgemm failed: {:?}",
                e
            );
            return -1;
        }

        0
    }))
    .unwrap_or(-1)
}

/// Write one buffer's device pointer into an array of pointers held in another
/// buffer.
///
/// `cublasSgemmBatched` takes three device arrays of pointers rather than three
/// base pointers and a stride, which is what lets a batch be assembled out of
/// buffers that were allocated separately — no adjacency, no merged allocation,
/// and no strided view of anybody's output.  The pointers are fixed once the
/// buffers exist, so the arrays are filled at load time and reused by every
/// launch.
///
/// - `arr_buf`: the buffer holding the array; must be at least `8*(slot+1)` bytes
/// - `slot`: which entry to write
/// - `src_buf`: the buffer whose device pointer to store
/// - `off`: element offset into `src_buf`, in f32 elements, as elsewhere
pub(crate) unsafe extern "C" fn cl_cublas_ptr_array(
    ctx_ptr: *mut CraneliftCudaContext,
    arr_buf: i32,
    slot: i32,
    src_buf: i32,
    off: i64,
) -> i32 {
    if arr_buf < 0 || slot < 0 || src_buf < 0 || off < 0 {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Some(src) = (unsafe { cuda_buffer_device_ptr(&state, src_buf) }) else {
            return -1;
        };
        let val = (src + (off as u64) * 4).to_ne_bytes();

        let aid = arr_buf as usize;
        if aid >= state.buffers.len() {
            return -1;
        }
        let Some(arr) = state.buffers[aid].as_ref() else {
            return -1;
        };
        let Some(dst) = (unsafe { cuda_buffer_device_ptr(&state, arr_buf) }) else {
            return -1;
        };
        if cudarc::driver::DeviceSlice::len(arr) < 8 * (slot as usize + 1) {
            return -1;
        }
        // A byte-level copy into the middle of an allocation: `htod_sync_copy_into`
        // takes the whole buffer, and only one entry moves here.
        let rc = unsafe {
            cudarc::driver::sys::lib().cuMemcpyHtoD_v2(
                dst + (slot as u64) * 8,
                val.as_ptr() as *const std::ffi::c_void,
                8,
            )
        };
        if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return -1;
        }
        0
    }))
    .unwrap_or(-1)
}

/// cuBLAS SGEMM batched over arrays of pointers: `C_i = alpha*op(A_i)*op(B_i)`.
///
/// The dimensions are shared by every member; only the pointers differ, and they
/// come from `cl_cublas_ptr_array`.  This is the shape a per-head projection
/// has: one contraction repeated over heads whose operands are separate
/// buffers.  Unlike `cl_cublas_sgemm_strided_batched` the members need no
/// uniform stride, so nothing has to be reallocated to use it.
///
/// Each `C_i` is a full `m x n` matrix in its own buffer, so a member's output
/// stays contiguous — which is why the callers downstream read it with an
/// ordinary buffer binding rather than a strided view.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe extern "C" fn cl_cublas_sgemm_batched_on_stream(
    ctx_ptr: *mut CraneliftCudaContext,
    transa: i32,
    transb: i32,
    m: i32,
    n: i32,
    k: i32,
    alpha_bits: i32,
    a_arr: i32,
    b_arr: i32,
    beta_bits: i32,
    c_arr: i32,
    batch_count: i32,
    stream_id: i32,
) -> i32 {
    use cudarc::cublas::sys::cublasOperation_t;

    if m <= 0 || n <= 0 || k <= 0 || batch_count <= 0 || a_arr < 0 || b_arr < 0 || c_arr < 0 {
        return -1;
    }

    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };

        let alpha = f32::from_bits(alpha_bits as u32);
        let beta = f32::from_bits(beta_bits as u32);
        let op_a = if transa != 0 {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let op_b = if transb != 0 {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let lda = if transa != 0 { k } else { m };
        let ldb = if transb != 0 { n } else { k };
        let ldc = m;

        let a_dev = match unsafe { cuda_buffer_device_ptr(&state, a_arr) } {
            Some(p) => p,
            None => return -1,
        };
        let b_dev = match unsafe { cuda_buffer_device_ptr(&state, b_arr) } {
            Some(p) => p,
            None => return -1,
        };
        let c_dev = match unsafe { cuda_buffer_device_ptr(&state, c_arr) } {
            Some(p) => p,
            None => return -1,
        };
        let blas = match ensure_stream_cuda_blas(ctx, &mut state, stream_id, stream) {
            Ok(blas) => blas,
            Err(rc) => return rc,
        };

        let st = unsafe {
            cudarc::cublas::sys::lib().cublasSgemmBatched(
                *blas.handle(),
                op_a,
                op_b,
                m,
                n,
                k,
                &alpha,
                a_dev as *const *const f32,
                lda,
                b_dev as *const *const f32,
                ldb,
                &beta,
                c_dev as *const *mut f32,
                ldc,
                batch_count,
            )
        };
        if st != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            eprintln!(
                "cl_cublas_sgemm_batched_on_stream: failed: {:?} \
                 (transa={} transb={} m={} n={} k={} lda={} ldb={} ldc={} batch={})",
                st, transa, transb, m, n, k, lda, ldb, ldc, batch_count
            );
            return -1;
        }
        0
    }))
    .unwrap_or(-1)
}

/// Batched `cublasGemmEx` with bf16 inputs and an f32 result.
///
/// The strided-batched sgemm's shape with the element types of `gemm_ex_bf16`.
/// Attention wants exactly this: a key cache held in bf16, one batch per KV
/// head, contracted against a query that has been narrowed to match -- cuBLAS
/// rejects a mixed (bf16, f32) operand pair, so both inputs are bf16 and only
/// the accumulator is f32.
///
/// As in `gemm_ex_bf16`, the offsets and strides count *elements*, and an
/// element is two bytes on the inputs and four on the output.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn cl_cublas_gemm_strided_batched_ex_bf16(
    ctx_ptr: *mut CraneliftCudaContext,
    transa: i32,
    transb: i32,
    m: i32,
    n: i32,
    k: i32,
    alpha_bits: i32,
    a_buf: i32,
    stride_a: i64,
    b_buf: i32,
    stride_b: i64,
    beta_bits: i32,
    c_buf: i32,
    stride_c: i64,
    batch_count: i32,
    off_a: i64,
    off_b: i64,
    off_c: i64,
    ld_a: i32,
    ld_b: i32,
    ld_c: i32,
) -> i32 {
    use cudarc::cublas::sys::{
        cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t, cublasStatus_t, cudaDataType,
    };

    if m <= 0
        || n <= 0
        || k <= 0
        || batch_count <= 0
        || off_a < 0
        || off_b < 0
        || off_c < 0
        || stride_a < 0
        || stride_b < 0
        || stride_c < 0
        || ld_a < 0
        || ld_b < 0
        || ld_c < 0
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

        let alpha = f32::from_bits(alpha_bits as u32);
        let beta = f32::from_bits(beta_bits as u32);
        let op_a = if transa != 0 {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let op_b = if transb != 0 {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let lda = if ld_a != 0 { ld_a } else if transa != 0 { k } else { m };
        let ldb = if ld_b != 0 { ld_b } else if transb != 0 { n } else { k };
        let ldc = if ld_c != 0 { ld_c } else { m };

        let a_dev = match unsafe { cuda_buffer_device_ptr(&state, a_buf) } {
            Some(p) => p + (off_a as u64) * 2,
            None => return -1,
        };
        let b_dev = match unsafe { cuda_buffer_device_ptr(&state, b_buf) } {
            Some(p) => p + (off_b as u64) * 2,
            None => return -1,
        };
        let c_dev = match unsafe { cuda_buffer_device_ptr(&state, c_buf) } {
            Some(p) => p + (off_c as u64) * 4,
            None => return -1,
        };
        let blas = match ensure_default_cuda_blas(ctx, &mut state) {
            Ok(blas) => blas,
            Err(rc) => return rc,
        };

        let st = unsafe {
            cudarc::cublas::sys::lib().cublasGemmStridedBatchedEx(
                *blas.handle(),
                op_a,
                op_b,
                m,
                n,
                k,
                &alpha as *const f32 as *const std::ffi::c_void,
                a_dev as *const std::ffi::c_void,
                cudaDataType::CUDA_R_16BF,
                lda,
                stride_a,
                b_dev as *const std::ffi::c_void,
                cudaDataType::CUDA_R_16BF,
                ldb,
                stride_b,
                &beta as *const f32 as *const std::ffi::c_void,
                c_dev as *mut std::ffi::c_void,
                cudaDataType::CUDA_R_32F,
                ldc,
                stride_c,
                batch_count,
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        };
        if st != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            eprintln!(
                "cl_cublas_gemm_strided_batched_ex_bf16: failed: {:?} \
                 (transa={} transb={} m={} n={} k={} lda={} ldb={} ldc={} \
                 sa={} sb={} sc={} batch={} bufs a={} b={} c={})",
                st, transa, transb, m, n, k, lda, ldb, ldc, stride_a, stride_b,
                stride_c, batch_count, a_buf, b_buf, c_buf
            );
            return -1;
        }
        0
    }))
    .unwrap_or(-1)
}

/// cuBLAS GEMM over bf16 operands: C = alpha * op(A) * op(B) + beta * C, where
/// A and B are bf16, C is f32, and the accumulation is f32
/// (`CUBLAS_COMPUTE_32F`).
///
/// Halving what a matvec reads from device memory is the point: decode is bound
/// by that read, and bf16 weights halve it.  The accumulator and the result stay
/// f32 — a bf16 accumulator would lose the sum — so `alpha`, `beta` and every
/// element of C are f32 as before.
///
/// **Both operands are bf16, and that is cuBLAS's rule, not a choice here.**
/// `cublasGemmEx` rejects a mixed (bf16, f32) pair with
/// `CUBLAS_STATUS_NOT_SUPPORTED` — measured on this card, not read off a table.
/// So an f32 activation must be narrowed to bf16 before it reaches this call,
/// which is what a bf16 pipeline does anyway; the rounding it costs is the same
/// rounding the weights already carry.
///
/// Argument shape follows `cl_cublas_sgemm_strided_batched`: offsets name a
/// slice of a buffer and leading dimensions may say the operand is a column
/// slice of something wider, neither of which changes the rows contracted.
/// `off_a` and `off_b` count **bf16 elements** (2 bytes), `off_c` counts f32
/// elements (4 bytes) — each offset is in the units of its own operand.
///
/// - `transa`, `transb`: 0 = NoTrans, 1 = Trans
/// - `m`, `n`, `k`: op(A) is m×k, op(B) is k×n, C is m×n.  `n == 1` is the
///   matvec decode uses; cuBLAS has no `gemvEx`, so it goes through here.
/// - `alpha_bits`, `beta_bits`: f32 scalars reinterpreted as i32 bits
/// - `ld_a`, `ld_b`, `ld_c`: 0 means "the one this shape implies"
pub(crate) unsafe extern "C" fn cl_cublas_gemm_ex_bf16(
    ctx_ptr: *mut CraneliftCudaContext,
    transa: i32,
    transb: i32,
    m: i32,
    n: i32,
    k: i32,
    alpha_bits: i32,
    a_buf: i32,
    b_buf: i32,
    beta_bits: i32,
    c_buf: i32,
    off_a: i64,
    off_b: i64,
    off_c: i64,
    ld_a: i32,
    ld_b: i32,
    ld_c: i32,
) -> i32 {
    use cudarc::cublas::sys::{
        cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t, cublasStatus_t, cudaDataType,
    };

    if m <= 0 || n <= 0 || k <= 0 || off_a < 0 || off_b < 0 || off_c < 0
        || ld_a < 0 || ld_b < 0 || ld_c < 0
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

        let alpha = f32::from_bits(alpha_bits as u32);
        let beta = f32::from_bits(beta_bits as u32);
        let op_a = if transa != 0 {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let op_b = if transb != 0 {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let lda = if ld_a != 0 { ld_a } else if transa != 0 { k } else { m };
        let ldb = if ld_b != 0 { ld_b } else if transb != 0 { n } else { k };
        let ldc = if ld_c != 0 { ld_c } else { m };

        // Element sizes differ per operand, so the offsets scale differently:
        // two bytes for each bf16 input, four for the f32 result.
        let a_dev = match unsafe { cuda_buffer_device_ptr(&state, a_buf) } {
            Some(p) => p + (off_a as u64) * 2,
            None => return -1,
        };
        let b_dev = match unsafe { cuda_buffer_device_ptr(&state, b_buf) } {
            Some(p) => p + (off_b as u64) * 2,
            None => return -1,
        };
        let c_dev = match unsafe { cuda_buffer_device_ptr(&state, c_buf) } {
            Some(p) => p + (off_c as u64) * 4,
            None => return -1,
        };
        let blas = match ensure_default_cuda_blas(ctx, &mut state) {
            Ok(blas) => blas,
            Err(rc) => return rc,
        };

        let st = unsafe {
            cudarc::cublas::sys::lib().cublasGemmEx(
                *blas.handle(),
                op_a,
                op_b,
                m,
                n,
                k,
                &alpha as *const f32 as *const std::ffi::c_void,
                a_dev as *const std::ffi::c_void,
                cudaDataType::CUDA_R_16BF,
                lda,
                b_dev as *const std::ffi::c_void,
                cudaDataType::CUDA_R_16BF,
                ldb,
                &beta as *const f32 as *const std::ffi::c_void,
                c_dev as *mut std::ffi::c_void,
                cudaDataType::CUDA_R_32F,
                ldc,
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        };
        if st != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            // Name the shape as well as the status: mixed-dtype GEMM rejects
            // some (dtype, compute, algo) combinations outright, and the status
            // alone does not say whether the fault is that or the arguments.
            eprintln!(
                "cl_cublas_gemm_ex_bf16: cublasGemmEx failed: {:?} \
                 (transa={} transb={} m={} n={} k={} lda={} ldb={} ldc={} \
                 bufs a={} b={} c={})",
                st, transa, transb, m, n, k, lda, ldb, ldc, a_buf, b_buf, c_buf
            );
            return -1;
        }
        0
    }))
    .unwrap_or(-1)
}
