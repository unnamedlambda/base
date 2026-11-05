//! The GPU side: NPP, with cudarc for the device memory, pinned memory,
//! streams and copies, as a Rust program would put them together.

use crate::{distance, frame, frame_bytes, planes, show_distance, DST, FRAMES, ROUNDS, SRC};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, PinnedHostSlice};
use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;

#[repr(C)]
#[derive(Clone, Copy)]
struct NppiSize {
    width: i32,
    height: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NppiRect {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NppStreamContext {
    stream: *mut c_void,
    device: i32,
    multiprocessors: i32,
    max_threads_per_multiprocessor: i32,
    max_threads_per_block: i32,
    shared_mem_per_block: usize,
    major: i32,
    minor: i32,
    stream_flags: u32,
    reserved: i32,
}

#[link(name = "nppc")]
extern "C" {
    fn nppGetStreamContext(ctx: *mut NppStreamContext) -> i32;
}

#[link(name = "nppig")]
extern "C" {
    fn nppiResize_8u_C1R_Ctx(
        src: *const u8,
        src_step: i32,
        src_size: NppiSize,
        src_roi: NppiRect,
        dst: *mut u8,
        dst_step: i32,
        dst_size: NppiSize,
        dst_roi: NppiRect,
        interpolation: i32,
        ctx: NppStreamContext,
    ) -> i32;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NppiImageDescriptor {
    data: *mut c_void,
    step: i32,
    size: NppiSize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NppiResizeBatchRoi {
    src: NppiRect,
    dst: NppiRect,
}

#[link(name = "nppig")]
extern "C" {
    fn nppiResizeBatch_8u_C1R_Advanced_Ctx(
        max_width: i32,
        max_height: i32,
        src: *mut NppiImageDescriptor,
        dst: *mut NppiImageDescriptor,
        roi: *mut NppiResizeBatchRoi,
        n: u32,
        interpolation: i32,
        ctx: NppStreamContext,
    ) -> i32;
}

/// The bytes of plain values.
fn bytes_of<T: Copy>(v: &[T]) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }.to_vec()
}

const NPPI_INTER_SUPER: i32 = 8;

/// One frame's three planes resized on `stream`, NPP's way.
fn resize(npp: NppStreamContext, stream: &Arc<CudaStream>, src: &CudaSlice<u8>, dst: &mut CudaSlice<u8>) {
    let ctx = NppStreamContext { stream: stream.cu_stream() as *mut c_void, ..npp };
    let (s, _rs) = src.device_ptr(stream);
    let (d, _rd) = dst.device_ptr_mut(stream);
    for ((sw, sh, so), (dw, dh, d_o)) in planes(SRC).into_iter().zip(planes(DST)) {
        let (ss, ds) = (NppiSize { width: sw as i32, height: sh as i32 }, NppiSize { width: dw as i32, height: dh as i32 });
        let rc = unsafe {
            nppiResize_8u_C1R_Ctx(
                (s + so as u64) as *const u8,
                sw as i32,
                ss,
                NppiRect { x: 0, y: 0, width: ss.width, height: ss.height },
                (d + d_o as u64) as *mut u8,
                dw as i32,
                ds,
                NppiRect { x: 0, y: 0, width: ds.width, height: ds.height },
                NPPI_INTER_SUPER,
                ctx,
            )
        };
        assert_eq!(rc, 0, "nppiResize_8u_C1R_Ctx answered {rc}");
    }
}

/// The fastest of `ROUNDS` runs of `f`, which processes `n` frames, as frames
/// per second.
fn rate(n: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut best = f64::MAX;
    for _ in 0..ROUNDS {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    n as f64 / best
}

/// The four-byte words of a status output.
fn words(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()
}

pub fn run(srcs: &[u8], fir_box: &[u8], drv: &'static base::Driver) {
    let cx = CudaContext::new(0).expect("a CUDA device");
    let streams = [cx.new_stream().expect("stream"), cx.new_stream().expect("stream")];
    let mut npp = unsafe { std::mem::zeroed::<NppStreamContext>() };
    assert_eq!(unsafe { nppGetStreamContext(&mut npp) }, 0);
    let (sb, db) = (frame_bytes(SRC), frame_bytes(DST));

    // Frames already on the device.
    let s0 = &streams[0];
    let on_device: Vec<CudaSlice<u8>> = (0..FRAMES).map(|k| s0.clone_htod(&frame(srcs, k)[..sb]).expect("upload")).collect();
    // every frame its own output, so a batch never has two writers
    let mut outs: Vec<CudaSlice<u8>> = (0..FRAMES).map(|_| s0.alloc_zeros::<u8>(db).expect("alloc")).collect();
    resize(npp, s0, &on_device[0], &mut outs[0]);
    let first = s0.clone_dtoh(&outs[0]).expect("download");
    println!();
    println!("same pixels as fast_image_resize? (per plane: largest, mean difference)");
    show_distance("NPP SUPER vs fir Box", distance(&first, fir_box));

    let resident = rate(FRAMES, || {
        for (f, o) in on_device.iter().zip(outs.iter_mut()) {
            resize(npp, s0, f, o);
        }
        s0.synchronize().expect("sync");
    });

    // NPP's batched resize: every plane of every frame in one call, its lists
    // in device memory as the call takes them
    let out_ptrs: Vec<u64> = outs.iter_mut().map(|o| o.device_ptr_mut(s0).0).collect();
    let src_ptrs: Vec<u64> = on_device.iter().map(|f| f.device_ptr(s0).0).collect();
    let (mut srcd, mut dstd, mut rois) = (Vec::new(), Vec::new(), Vec::new());
    for k in 0..FRAMES {
        for ((sw, sh, so), (dw, dh, d_o)) in planes(SRC).into_iter().zip(planes(DST)) {
            let (ss, ds) = (NppiSize { width: sw as i32, height: sh as i32 }, NppiSize { width: dw as i32, height: dh as i32 });
            srcd.push(NppiImageDescriptor { data: (src_ptrs[k] + so as u64) as *mut c_void, step: sw as i32, size: ss });
            dstd.push(NppiImageDescriptor { data: (out_ptrs[k] + d_o as u64) as *mut c_void, step: dw as i32, size: ds });
            rois.push(NppiResizeBatchRoi {
                src: NppiRect { x: 0, y: 0, width: ss.width, height: ss.height },
                dst: NppiRect { x: 0, y: 0, width: ds.width, height: ds.height },
            });
        }
    }
    let (srcl, dstl, roil) = (
        s0.clone_htod(&bytes_of(&srcd)).expect("upload"),
        s0.clone_htod(&bytes_of(&dstd)).expect("upload"),
        s0.clone_htod(&bytes_of(&rois)).expect("upload"),
    );
    let lists = (srcl.device_ptr(s0).0, dstl.device_ptr(s0).0, roil.device_ptr(s0).0);
    s0.synchronize().expect("sync");
    let batch = |npp: NppStreamContext| {
        let ctx = NppStreamContext { stream: s0.cu_stream() as *mut c_void, ..npp };
        let rc = unsafe {
            nppiResizeBatch_8u_C1R_Advanced_Ctx(DST.0 as i32, DST.1 as i32, lists.0 as *mut _, lists.1 as *mut _,
                lists.2 as *mut _, srcd.len() as u32, NPPI_INTER_SUPER, ctx)
        };
        assert_eq!(rc, 0, "nppiResizeBatch answered {rc}");
    };
    batch(npp);
    s0.synchronize().expect("sync");
    let batched_first = s0.clone_dtoh(&outs[0]).expect("download");
    println!("  NPP's batched output for frame 0 differs from its per-plane output in {} bytes",
        batched_first.iter().zip(&first).filter(|(a, b)| a != b).count());
    let resident_batch = rate(FRAMES, || {
        batch(npp);
        s0.synchronize().expect("sync");
    });

    // The pipeline: frames in pinned host memory, uploaded, resized and
    // downloaded on two streams in turn, so one stream's copies overlap the
    // other's work.
    let pinned_src: Vec<PinnedHostSlice<u8>> = (0..FRAMES)
        .map(|k| {
            let mut p = unsafe { cx.alloc_pinned::<u8>(sb) }.expect("pinned");
            p.as_mut_slice().expect("pinned").copy_from_slice(&frame(srcs, k)[..sb]);
            p
        })
        .collect();
    let mut pinned_out: Vec<PinnedHostSlice<u8>> =
        (0..2).map(|_| unsafe { cx.alloc_pinned_with_flags::<u8>(db, 0) }.expect("pinned")).collect();
    let mut dev_in: Vec<CudaSlice<u8>> = streams.iter().map(|s| s.alloc_zeros::<u8>(sb).expect("alloc")).collect();
    let mut dev_out: Vec<CudaSlice<u8>> = streams.iter().map(|s| s.alloc_zeros::<u8>(db).expect("alloc")).collect();
    let n = 4 * FRAMES;
    let pipeline = rate(n, || {
        for i in 0..n {
            let j = i % 2;
            let s = &streams[j];
            s.memcpy_htod(&pinned_src[i % FRAMES], &mut dev_in[j]).expect("upload");
            resize(npp, s, &dev_in[j], &mut dev_out[j]);
            s.memcpy_dtoh(&dev_out[j], &mut pinned_out[j]).expect("download");
        }
        for s in &streams {
            s.synchronize().expect("sync");
        }
    });

    // The copies alone: what the link carries.
    let upload = rate(n, || {
        for i in 0..n {
            let j = i % 2;
            streams[j].memcpy_htod(&pinned_src[i % FRAMES], &mut dev_in[j]).expect("upload");
        }
        for s in &streams {
            s.synchronize().expect("sync");
        }
    });

    // Base: the same two jobs through the artifact, which calls the driver
    // itself: its own module, streams and device frames, on the same context.
    for s in &streams {
        s.synchronize().expect("sync");
    }
    let call = |name: &str, input: &[u8], out: &mut [u8]| drv.function(name).expect(name)(input, out);
    let mut st = [0u8; 48];
    call("gpu_init", &[], &mut st);
    assert_eq!(words(&st), vec![0; 12], "gpu_init's steps all answered 0");
    let pairs = |ks: &[usize]| -> Vec<u8> {
        ks.iter().flat_map(|&k| [src_ptrs[k].to_le_bytes(), out_ptrs[k].to_le_bytes()].concat()).collect()
    };
    let base_frame = |k: usize| -> Vec<u8> {
        let mut st = [0u8; 8];
        call("gpu_box", &pairs(&[k]), &mut st);
        assert_eq!(words(&st), vec![0, 0], "gpu_box launched and waited");
        s0.clone_dtoh(&outs[k]).expect("download")
    };
    let base_first = base_frame(0);
    show_distance("Base box2 vs fir Box", distance(&base_first, fir_box));
    show_distance("Base box2 vs NPP SUPER", distance(&base_first, &first));
    let all = pairs(&(0..FRAMES).collect::<Vec<_>>());
    let base_resident = rate(FRAMES, || {
        let mut st = [0u8; 8];
        call("gpu_box", &all, &mut st);
        assert_eq!(words(&st), vec![0, 0]);
    });
    let mut pipe_in: Vec<u8> = Vec::new();
    for o in pinned_out.iter_mut() {
        pipe_in.extend((o.as_mut_ptr().expect("pinned") as u64).to_le_bytes());
    }
    for i in 0..n {
        pipe_in.extend((pinned_src[i % FRAMES].as_ptr().expect("pinned") as u64).to_le_bytes());
    }
    let base_pipeline = rate(n, || {
        let mut st = [0u8; 12];
        call("gpu_pipe", &pipe_in, &mut st);
        assert_eq!(words(&st), vec![0, 0, 0], "gpu_pipe's calls all answered 0");
    });
    // the last frame came back through stream 1's host output
    let last = (n - 1) % FRAMES;
    let want = base_frame(last);
    let got = pinned_out[1].as_slice().expect("pinned").to_vec();
    println!("  Base's pipeline output for frame {last} differs from its on-device result in {} bytes",
        got.iter().zip(&want).filter(|(a, b)| a != b).count());

    println!();
    println!("GPU, a 2x2 average (frames/s)          NPP     Base");
    println!("  {:28} {resident:9.1} {base_resident:8.1}", "frames on the device");
    println!("  {:28} {resident_batch:9.1}", "  NPP, one batched call");
    println!("  {:28} {pipeline:9.1} {base_pipeline:8.1}", "pinned host -> GPU -> host");
    println!("  {:28} {upload:9.1}   ({:.1} GB/s)", "uploads alone", upload * sb as f64 / 1e9);
}
