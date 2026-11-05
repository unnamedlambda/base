//! Downscaling a stream of 4K video frames to 1080p, as Rust programs do it.
//!
//! A frame is YUV 4:2:0, the planar layout video decoders hand out: a
//! full-size luma plane and two half-size chroma planes. Eight distinct frames
//! are cycled, so no frame is still in cache when it comes round again.
//!
//! Three filters, each a table: the antialiased bilinear filter, the 2x2
//! average, and Lanczos3. CPU columns: the `image` crate (what most Rust code
//! calls), `fast_image_resize` and `pic-scale` (the Rust SIMD crates), FFmpeg's
//! swscale, Google's libyuv (whose 2:1 path is a hand-written 2x2 average), and
//! Base running the artifact `Bench.Image` writes, named on the command line:
//! each filter through CLIF and through AVX2 machine code. GPU columns: NVIDIA's
//! NPP and Base, the 2x2 average on frames already on the device and as a
//! pipeline from pinned host memory and back.
//!
//! The method, the same for every column:
//!
//! * Same work: each output is compared, plane by plane, with a reference at
//!   the same filter (`fast_image_resize`; for Lanczos3 the `image` crate,
//!   which keeps the intermediate rows unclipped as Base does), so a row is
//!   only read beside rows computing the same pixels.
//! * Same setup: contexts, plans, resizers and Base's machine code are made
//!   once per column, outside the timing; a timed call does only the frame.
//! * Same machine state: columns take turns, each timed call after an untimed
//!   one of its own; one core is `taskset` by `run.sh`; on several threads
//!   every thread has its own scaler and output.
//! * Each library at its fastest: `audit` shows FFmpeg and libyuv slower with
//!   their SIMD switched off, fast_image_resize reporting AVX2, and swscale on
//!   one thread; NPP's batched call is timed beside its per-plane calls.
//! * Ceilings: two memory floors, a plain one and one writing its output with
//!   non-temporal stores, bound what any code could reach.

mod gpu;

use fast_image_resize as fir;
use std::hint::black_box;
use std::time::Instant;

pub const SRC: (usize, usize) = (3840, 2160);
pub const DST: (usize, usize) = (1920, 1080);
pub const FRAMES: usize = 8;
pub const ROUNDS: usize = 9;

/// The three planes of a frame `(w, h)`: each one's width, height and offset.
pub fn planes((w, h): (usize, usize)) -> [(usize, usize, usize); 3] {
    let (cw, ch) = (w / 2, h / 2);
    [(w, h, 0), (cw, ch, w * h), (cw, ch, w * h + cw * ch)]
}

pub const fn frame_bytes(size: (usize, usize)) -> usize {
    size.0 * size.1 * 3 / 2
}

/// Room after a frame's last row, as FFmpeg's frame allocator leaves: its
/// vector loops read and write whole vectors at the end of a row.
pub const PAD: usize = 64;

/// What Base's `resize` asks of its output: the frame, then its scratch rows
/// (`OUT_NEED` in `Bench.Image`).
const BASE_OUT: usize = frame_bytes(DST) + 16 * 3840 + 1920 + 128;

/// Room in every output buffer: the frame and its padding, or what Base asks.
pub const OUT_ROOM: usize = if BASE_OUT > frame_bytes(DST) + PAD { BASE_OUT } else { frame_bytes(DST) + PAD };

/// Bytes on a 64-byte boundary, zeroed: every frame buffer, source and output,
/// so every library sees the alignment the strictest (zimg) asks for.
pub struct Buf {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for Buf {}
unsafe impl Sync for Buf {}

impl Buf {
    pub fn zeroed(len: usize) -> Buf {
        let layout = std::alloc::Layout::from_size_align(len, 64).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "allocated");
        Buf { ptr, len }
    }
}

impl std::ops::Deref for Buf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl std::ops::DerefMut for Buf {
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr, std::alloc::Layout::from_size_align(self.len, 64).unwrap()) }
    }
}

/// `FRAMES` frames of noise, back to back. What the pixels are does not
/// change the work.
fn sources() -> Buf {
    let mut x = 0x9E3779B97F4A7C15u64;
    let mut b = Buf::zeroed(FRAMES * frame_bytes(SRC) + PAD);
    for v in b.iter_mut() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *v = x as u8;
    }
    b
}

/// Frame `k` and the bytes after it, at least `PAD` of them.
pub fn frame(all: &[u8], k: usize) -> &[u8] {
    &all[k * frame_bytes(SRC)..]
}

/// A downscaler: one source frame into one destination frame.
type Scale<'a> = Box<dyn FnMut(&[u8], &mut [u8]) + 'a>;

fn image_scale(f: image::imageops::FilterType) -> Scale<'static> {
    Box::new(move |src, dst| {
        for ((sw, sh, so), (dw, dh, d_o)) in planes(SRC).into_iter().zip(planes(DST)) {
            let plane = image::ImageBuffer::<image::Luma<u8>, &[u8]>::from_raw(sw as u32, sh as u32, &src[so..so + sw * sh])
                .expect("sized");
            let out = image::imageops::resize(&plane, dw as u32, dh as u32, f);
            dst[d_o..d_o + dw * dh].copy_from_slice(out.as_raw());
        }
    })
}

fn fir_scale(f: fir::FilterType) -> Scale<'static> {
    let mut r = fir::Resizer::new();
    let opts = fir::ResizeOptions::new().resize_alg(fir::ResizeAlg::Convolution(f));
    Box::new(move |src, dst| {
        for ((sw, sh, so), (dw, dh, d_o)) in planes(SRC).into_iter().zip(planes(DST)) {
            let s = fir::images::ImageRef::new(sw as u32, sh as u32, &src[so..so + sw * sh], fir::PixelType::U8)
                .expect("sized");
            let mut d =
                fir::images::Image::from_slice_u8(dw as u32, dh as u32, &mut dst[d_o..d_o + dw * dh], fir::PixelType::U8)
                    .expect("sized");
            r.resize(&s, &mut d, &opts).expect("fir resizes");
        }
    })
}

/// FFmpeg's swscale, one context for the stream, as FFmpeg's scale filter
/// keeps one.
fn sws_scale(flags: i32) -> Scale<'static> {
    use ffmpeg_sys_next as ff;
    let fmt = ff::AVPixelFormat::AV_PIX_FMT_YUV420P;
    let ctx = unsafe {
        ff::sws_getContext(
            SRC.0 as i32,
            SRC.1 as i32,
            fmt,
            DST.0 as i32,
            DST.1 as i32,
            fmt,
            flags,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null(),
        )
    };
    assert!(!ctx.is_null(), "swscale takes these sizes");
    struct Ctx(*mut ff::SwsContext);
    impl Drop for Ctx {
        fn drop(&mut self) {
            unsafe { ff::sws_freeContext(self.0) }
        }
    }
    let ctx = Ctx(ctx);
    Box::new(move |src, dst| {
        // The whole context, so the closure owns it and frees it last.
        let ctx = &ctx;
        let (sp, dp) = (planes(SRC), planes(DST));
        let src_ptrs = [sp[0].2, sp[1].2, sp[2].2, 0].map(|o| unsafe { src.as_ptr().add(o) });
        let dst_ptrs = [dp[0].2, dp[1].2, dp[2].2, 0].map(|o| unsafe { dst.as_mut_ptr().add(o) });
        let src_strides = [sp[0].0 as i32, sp[1].0 as i32, sp[2].0 as i32, 0];
        let dst_strides = [dp[0].0 as i32, dp[1].0 as i32, dp[2].0 as i32, 0];
        let rows = unsafe {
            ff::sws_scale(ctx.0, src_ptrs.as_ptr(), src_strides.as_ptr(), 0, SRC.1 as i32, dst_ptrs.as_ptr(), dst_strides.as_ptr())
        };
        assert_eq!(rows, DST.1 as i32, "swscale wrote every row");
    })
}

/// Google's libyuv: the whole frame in one `I420Scale`, which at exactly half
/// size takes its 2:1 path (`ScaleRowDown2Box`, AVX2 here).
fn libyuv_scale(filter: yuv_sys::FilterMode) -> Scale<'static> {
    Box::new(move |src, dst| {
        let (sp, dp) = (planes(SRC), planes(DST));
        let rc = unsafe {
            yuv_sys::rs_I420Scale(
                src.as_ptr().add(sp[0].2),
                sp[0].0 as i32,
                src.as_ptr().add(sp[1].2),
                sp[1].0 as i32,
                src.as_ptr().add(sp[2].2),
                sp[2].0 as i32,
                SRC.0 as i32,
                SRC.1 as i32,
                dst.as_mut_ptr().add(dp[0].2),
                dp[0].0 as i32,
                dst.as_mut_ptr().add(dp[1].2),
                dp[1].0 as i32,
                dst.as_mut_ptr().add(dp[2].2),
                dp[2].0 as i32,
                DST.0 as i32,
                DST.1 as i32,
                filter,
            )
        };
        assert_eq!(rc, 0, "I420Scale answered {rc}");
    })
}

/// pic-scale: a plan per plane, made once, single-threaded (its default), each
/// with its own scratch.
fn picscale_scale(f: pic_scale::ResamplingFunction) -> Scale<'static> {
    use pic_scale::{ImageSize, ImageStore, ImageStoreMut, Scaler};
    let scaler = Scaler::new(f);
    let plans: Vec<_> = planes(SRC)
        .into_iter()
        .zip(planes(DST))
        .map(|((sw, sh, _), (dw, dh, _))| {
            scaler.plan_planar_resampling(ImageSize::new(sw, sh), ImageSize::new(dw, dh)).expect("pic-scale plans")
        })
        .collect();
    let mut scratch: Vec<Vec<u8>> = plans.iter().map(|p| vec![0u8; p.scratch_size()]).collect();
    Box::new(move |src, dst| {
        for (k, ((sw, sh, so), (dw, dh, d_o))) in planes(SRC).into_iter().zip(planes(DST)).enumerate() {
            let s = ImageStore::<u8, 1>::from_slice(&src[so..so + sw * sh], sw, sh).expect("sized");
            let mut d = ImageStoreMut::<u8, 1>::from_slice(&mut dst[d_o..d_o + dw * dh], dw, dh).expect("sized");
            plans[k].resample_with_scratch(&s, &mut d, &mut scratch[k]).expect("pic-scale resizes");
        }
    })
}

extern "C" {
    fn zimg_lanczos_new(sw: u32, sh: u32, dw: u32, dh: u32, simd: i32) -> *mut std::ffi::c_void;
    fn zimg_lanczos_run(h: *mut std::ffi::c_void, src: *const u8, ss: i64, dst: *mut u8, ds: i64) -> i32;
    fn vips_bench_init() -> i32;
    fn vips_bench_simd(on: i32);
    fn vips_lanczos_plane(src: *const u8, w: i32, h: i32, dst: *mut u8) -> i32;
}

/// zimg (the resizer behind FFmpeg's zscale and VapourSynth): a Lanczos3
/// graph per plane size, built once, each plane treated as grey so no chroma
/// siting moves it.
fn zimg_scale(simd: bool) -> Scale<'static> {
    let graphs: Vec<_> = planes(SRC)
        .into_iter()
        .zip(planes(DST))
        .map(|((sw, sh, _), (dw, dh, _))| {
            let g = unsafe { zimg_lanczos_new(sw as u32, sh as u32, dw as u32, dh as u32, simd as i32) };
            assert!(!g.is_null(), "zimg builds the graph");
            g
        })
        .collect();
    Box::new(move |src, dst| {
        for (k, ((sw, _, so), (dw, _, d_o))) in planes(SRC).into_iter().zip(planes(DST)).enumerate() {
            let rc = unsafe { zimg_lanczos_run(graphs[k], src.as_ptr().add(so), sw as i64, dst.as_mut_ptr().add(d_o), dw as i64) };
            assert_eq!(rc, 0, "zimg answered {rc}");
        }
    })
}

static VIPS: std::sync::Once = std::sync::Once::new();

/// libvips (the library behind sharp): each plane through `vips_reduce` with
/// its Lanczos3 kernel, one worker per pipeline, no operation cache.
fn vips_scale() -> Scale<'static> {
    VIPS.call_once(|| assert_eq!(unsafe { vips_bench_init() }, 0, "libvips starts"));
    Box::new(|src, dst| {
        for ((sw, sh, so), (_, _, d_o)) in planes(SRC).into_iter().zip(planes(DST)) {
            let rc = unsafe { vips_lanczos_plane(src.as_ptr().add(so), sw as i32, sh as i32, dst.as_mut_ptr().add(d_o)) };
            assert_eq!(rc, 0, "libvips answered {rc}");
        }
    })
}

/// The driver for the artifact at `path`, its machine code mapped. Leaked: it
/// lives as long as the process.
fn base_driver(path: &str) -> &'static base::Driver {
    let bytes = std::fs::read(path).expect("the artifact is readable");
    let drv: &'static base::Driver = Box::leak(Box::new(unsafe { base::Driver::load(&bytes).expect("the artifact loads") }));
    let mut slots = [0u8; 40];
    drv.function("asm_load").expect("asm_load")(&[], &mut slots);
    let word = |k: usize| u64::from_le_bytes(slots[8 * k..8 * k + 8].try_into().unwrap());
    assert!(word(0) == 4 && (1..=4).all(|k| word(k) != 0), "every AVX2 loop loaded");
    drv
}

/// Base's entry `name`: one frame in, the frame and its scratch out.
fn base_scale(drv: &'static base::Driver, name: &str) -> Scale<'static> {
    let f = drv.function(name).expect("the artifact has the entry");
    Box::new(move |src, dst| {
        f(&src[..frame_bytes(SRC)], &mut dst[..BASE_OUT]);
    })
}

/// How many output bytes differ between `a` and `b`.
fn differing(a: &[u8], b: &[u8]) -> usize {
    let n = frame_bytes(DST);
    a[..n].iter().zip(&b[..n]).filter(|(x, y)| x != y).count()
}

/// Where swscale's outputs differ from `fir`'s by more than 1: how many, and
/// how many of them lie on the outermost ring of a plane.
fn far_pixels(a: &[u8], b: &[u8]) -> (usize, usize) {
    let (mut far, mut edge) = (0, 0);
    for (w, h, o) in planes(DST) {
        for y in 0..h {
            for x in 0..w {
                if a[o + y * w + x].abs_diff(b[o + y * w + x]) > 1 {
                    far += 1;
                    if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
                        edge += 1;
                    }
                }
            }
        }
    }
    (far, edge)
}

/// The memory floor: reading a source frame 32 bytes at a time and writing an
/// output frame, the traffic every downscaler here moves, with no arithmetic.
fn floor_scale() -> Scale<'static> {
    Box::new(|src, dst| {
        let n = frame_bytes(SRC);
        let mut acc = [0u8; 32];
        for c in src[..n].chunks_exact(32) {
            for (a, b) in acc.iter_mut().zip(c) {
                *a ^= *b;
            }
        }
        let m = frame_bytes(DST);
        for c in dst[..m].chunks_exact_mut(32) {
            c.copy_from_slice(&acc);
        }
    })
}

/// The memory floor with the output written by non-temporal stores, which
/// skip reading the output's lines before writing them: the traffic a
/// downscaler could get away with at best.
fn floor_nt_scale() -> Scale<'static> {
    Box::new(|src, dst| unsafe {
        use std::arch::x86_64::*;
        let n = frame_bytes(SRC);
        let mut acc = _mm256_setzero_si256();
        for k in (0..n).step_by(32) {
            acc = _mm256_xor_si256(acc, _mm256_loadu_si256(src.as_ptr().add(k) as _));
        }
        let m = frame_bytes(DST);
        // the output starts where the caller's buffer does; stream from the
        // first 32-byte boundary and write the head plainly
        let head = dst.as_ptr().align_offset(32).min(m);
        dst[..head].fill(0);
        let mut k = head;
        while k + 32 <= m {
            _mm256_stream_si256(dst.as_mut_ptr().add(k) as _, acc);
            k += 32;
        }
        dst[k..m].fill(0);
        _mm_sfence();
    })
}

/// How far `a` is from `b`, per plane: the largest and the mean difference.
pub fn distance(a: &[u8], b: &[u8]) -> [(u8, f64); 3] {
    planes(DST).map(|(w, h, o)| {
        let (x, y) = (&a[o..o + w * h], &b[o..o + w * h]);
        let max = x.iter().zip(y).map(|(p, q)| p.abs_diff(*q)).max().unwrap_or(0);
        let sum: u64 = x.iter().zip(y).map(|(p, q)| p.abs_diff(*q) as u64).sum();
        (max, sum as f64 / x.len() as f64)
    })
}

pub fn show_distance(name: &str, d: [(u8, f64); 3]) {
    println!(
        "  {name:36} Y {:3} {:6.3}   U {:3} {:6.3}   V {:3} {:6.3}",
        d[0].0, d[0].1, d[1].0, d[1].1, d[2].0, d[2].1
    );
}

/// Frames per second of each column at its best round, the columns taking
/// turns; a round scales every source frame once, after one untimed frame.
fn fps(cols: &mut [Scale], srcs: &[u8]) -> Vec<f64> {
    let mut dst = Buf::zeroed(OUT_ROOM);
    let mut best = vec![f64::MAX; cols.len()];
    for _ in 0..ROUNDS {
        for (j, c) in cols.iter_mut().enumerate() {
            c(frame(srcs, FRAMES - 1), &mut dst);
            let t = Instant::now();
            for k in 0..FRAMES {
                c(frame(srcs, k), &mut dst);
                black_box(&mut dst);
            }
            best[j] = best[j].min(t.elapsed().as_secs_f64() / FRAMES as f64);
        }
    }
    best.into_iter().map(|s| 1.0 / s).collect()
}

/// Frames per second of `threads` threads together, each scaling its own
/// frames with its own scaler from `make` into its own output, at the best of
/// `ROUNDS` rounds; a thread starts each round on a different source frame.
fn fps_threads(make: &(dyn Fn() -> Scale<'static> + Sync), srcs: &[u8], threads: usize) -> f64 {
    let per = 2 * FRAMES;
    let mut best = f64::MAX;
    for _ in 0..ROUNDS {
        let start = std::sync::Barrier::new(threads + 1);
        let wall = std::thread::scope(|sc| {
            for t in 0..threads {
                let start = &start;
                sc.spawn(move || {
                    let mut scale = make();
                    let mut dst = Buf::zeroed(OUT_ROOM);
                    scale(frame(srcs, t % FRAMES), &mut dst);
                    start.wait();
                    for i in 0..per {
                        scale(frame(srcs, (t + i) % FRAMES), &mut dst);
                        black_box(&mut dst);
                    }
                });
            }
            start.wait();
            Instant::now()
        });
        best = best.min(wall.elapsed().as_secs_f64());
    }
    (threads * per) as f64 / best
}

/// Whether each library ran its SIMD: what it reports, and its speed with
/// SIMD switched off where it lets a caller do that.
fn audit(srcs: &[u8]) {
    use ffmpeg_sys_next as ff;
    println!();
    println!("fast_image_resize uses {:?}", fir::Resizer::new().cpu_extensions());
    let flags = unsafe { ff::av_get_cpu_flags() };
    println!("FFmpeg sees AVX2: {}", flags & ff::AV_CPU_FLAG_AVX2 as i32 != 0);
    let sws_flags = ff::SwsFlags::SWS_BILINEAR as i32;
    let fmt = ff::AVPixelFormat::AV_PIX_FMT_YUV420P;
    let ctx = unsafe {
        ff::sws_getContext(SRC.0 as i32, SRC.1 as i32, fmt, DST.0 as i32, DST.1 as i32, fmt, sws_flags,
            std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null())
    };
    let mut threads = 0i64;
    let rc = unsafe { ff::av_opt_get_int(ctx as *mut _, c"threads".as_ptr(), 0, &mut threads) };
    unsafe { ff::sws_freeContext(ctx) };
    println!("swscale's threads option: {threads} (lookup answered {rc})");
    let with = fps(&mut [sws_scale(sws_flags)], srcs)[0];
    unsafe { ff::av_force_cpu_flags(0) };
    let without = fps(&mut [sws_scale(sws_flags)], srcs)[0];
    unsafe { ff::av_force_cpu_flags(-1) };
    println!("swscale BILINEAR: {with:.1} frames/s as built, {without:.1} with FFmpeg's SIMD off");
    let with = fps(&mut [libyuv_scale(yuv_sys::FilterMode_kFilterBox)], srcs)[0];
    unsafe { yuv_sys::rs_MaskCpuFlags(1) };
    let without = fps(&mut [libyuv_scale(yuv_sys::FilterMode_kFilterBox)], srcs)[0];
    unsafe { yuv_sys::rs_MaskCpuFlags(-1) };
    println!("libyuv Box: {with:.1} frames/s as built, {without:.1} with its SIMD off");
    let with = fps(&mut [zimg_scale(true)], srcs)[0];
    let without = fps(&mut [zimg_scale(false)], srcs)[0];
    println!("zimg Lanczos: {with:.1} frames/s as built, {without:.1} with its SIMD off");
    let with = fps(&mut [vips_scale()], srcs)[0];
    unsafe { vips_bench_simd(0) };
    let without = fps(&mut [vips_scale()], srcs)[0];
    unsafe { vips_bench_simd(1) };
    println!("libvips Lanczos3: {with:.1} frames/s as built, {without:.1} with its SIMD off");
}

/// A column: its name and how to make a scaler for it.
type Maker = (&'static str, Box<dyn Fn() -> Scale<'static> + Sync>);

/// Frames per second of each column in `makers`, on one core.
fn one_core(title: &str, makers: &[Maker], reference: &[u8], fir_name: &str, srcs: &[u8], one: &dyn Fn(Scale) -> Vec<u8>) {
    println!();
    println!("{title}: same pixels as {fir_name}? (per plane: largest, mean difference)");
    for (name, make) in makers {
        if !name.starts_with("memory floor") && *name != fir_name {
            show_distance(name, distance(&one(make()), reference));
        }
    }
    let mut cols: Vec<Scale> = makers.iter().map(|(_, m)| m()).collect();
    let rates = fps(&mut cols, srcs);
    println!();
    println!("{title}, one core (frames/s)");
    for ((n, _), r) in makers.iter().zip(rates) {
        println!("  {n:32} {r:8.1}");
    }
}

/// Frames per second of each column in `makers` on 1, 2, 4 and 6 threads.
fn all_cores(title: &str, makers: &[Maker], srcs: &[u8]) {
    let counts = [1, 2, 4, 6];
    println!();
    println!("{title}, frames on several threads at once (frames/s)");
    print!("  {:32}", "threads");
    for n in counts {
        print!(" {n:8}");
    }
    println!();
    for (name, make) in makers {
        print!("  {name:32}");
        for n in counts {
            print!(" {:8.1}", fps_threads(make.as_ref(), srcs, n));
        }
        println!();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (art, mode) = match &args[1..] {
        [art, mode] => (art.as_str(), mode.as_str()),
        _ => panic!("usage: image-bench <image_bench.cbor> one|cores|gpu"),
    };
    let srcs = sources();
    let drv = base_driver(art);
    use fir::FilterType as F;
    use image::imageops::FilterType as I;
    use pic_scale::ResamplingFunction as P;
    let sws_bilinear = ffmpeg_sys_next::SwsFlags::SWS_BILINEAR as i32;
    let sws_area = ffmpeg_sys_next::SwsFlags::SWS_AREA as i32;
    let one = |mut s: Scale| {
        let mut d = Buf::zeroed(OUT_ROOM);
        s(frame(&srcs, 0), &mut d);
        d.to_vec()
    };
    let bilinear: Vec<Maker> = vec![
        ("memory floor (read + write)", Box::new(floor_scale)),
        ("memory floor, streaming stores", Box::new(floor_nt_scale)),
        ("image, Triangle", Box::new(|| image_scale(I::Triangle))),
        ("fast_image_resize, Bilinear", Box::new(|| fir_scale(F::Bilinear))),
        ("pic-scale, Bilinear", Box::new(|| picscale_scale(P::Bilinear))),
        ("FFmpeg swscale, BILINEAR", Box::new(move || sws_scale(sws_bilinear))),
        ("Base, CLIF, a byte a trip", Box::new(move || base_scale(drv, "resize_clif"))),
        ("Base, CLIF, 8 bytes a trip", Box::new(move || base_scale(drv, "resize_swar"))),
        ("Base, AVX2 machine code", Box::new(move || base_scale(drv, "resize"))),
    ];
    let boxes: Vec<Maker> = vec![
        ("memory floor (read + write)", Box::new(floor_scale)),
        ("memory floor, streaming stores", Box::new(floor_nt_scale)),
        ("fast_image_resize, Box", Box::new(|| fir_scale(F::Box))),
        ("FFmpeg swscale, AREA", Box::new(move || sws_scale(sws_area))),
        ("libyuv, I420Scale Box", Box::new(|| libyuv_scale(yuv_sys::FilterMode_kFilterBox))),
        ("Base, CLIF, 8 bytes a trip", Box::new(move || base_scale(drv, "box_clif"))),
        ("Base, AVX2 machine code", Box::new(move || base_scale(drv, "box"))),
    ];
    let sws_lanczos = ffmpeg_sys_next::SwsFlags::SWS_LANCZOS as i32;
    let lanczos: Vec<Maker> = vec![
        ("memory floor (read + write)", Box::new(floor_scale)),
        ("memory floor, streaming stores", Box::new(floor_nt_scale)),
        ("image, Lanczos3", Box::new(|| image_scale(I::Lanczos3))),
        ("fast_image_resize, Lanczos3", Box::new(|| fir_scale(F::Lanczos3))),
        ("pic-scale, Lanczos3", Box::new(|| picscale_scale(P::Lanczos3))),
        ("FFmpeg swscale, LANCZOS", Box::new(move || sws_scale(sws_lanczos))),
        ("zimg, Lanczos 3 taps", Box::new(|| zimg_scale(true))),
        ("libvips, reduce Lanczos3", Box::new(vips_scale)),
        ("Base, CLIF, an output a trip", Box::new(move || base_scale(drv, "lanczos_clif"))),
        ("Base, AVX2 machine code", Box::new(move || base_scale(drv, "lanczos"))),
    ];
    println!("{}x{} -> {}x{} YUV 4:2:0, {FRAMES} distinct frames", SRC.0, SRC.1, DST.0, DST.1);
    match mode {
        "one" => {
            let fir_bilinear = one(fir_scale(F::Bilinear));
            one_core("Bilinear", &bilinear, &fir_bilinear, "fast_image_resize, Bilinear", &srcs, &one);
            let asm = one(base_scale(drv, "resize"));
            let (clif, swar) = (one(base_scale(drv, "resize_clif")), one(base_scale(drv, "resize_swar")));
            let img = one(image_scale(I::Triangle));
            println!(
                "  Base's AVX2 path differs from its CLIF paths in {} and {} bytes, and from image in {} of {}",
                differing(&asm, &clif),
                differing(&asm, &swar),
                differing(&asm, &img),
                frame_bytes(DST)
            );
            let (far, edge) = far_pixels(&one(sws_scale(sws_bilinear)), &fir_bilinear);
            println!("  swscale is more than 1 from fir at {far} outputs, {edge} of them on a plane's outermost ring");
            let fir_box = one(fir_scale(F::Box));
            one_core("2x2 average", &boxes, &fir_box, "fast_image_resize, Box", &srcs, &one);
            let (asm, clif) = (one(base_scale(drv, "box")), one(base_scale(drv, "box_clif")));
            let yuv = one(libyuv_scale(yuv_sys::FilterMode_kFilterBox));
            println!(
                "  Base's AVX2 box differs from its CLIF box in {} bytes, and from libyuv in {} of {}",
                differing(&asm, &clif),
                differing(&asm, &yuv),
                frame_bytes(DST)
            );
            let img_lz = one(image_scale(I::Lanczos3));
            one_core("Lanczos3", &lanczos, &img_lz, "image, Lanczos3", &srcs, &one);
            let (asm, clif) = (one(base_scale(drv, "lanczos")), one(base_scale(drv, "lanczos_clif")));
            let fir_lz = one(fir_scale(F::Lanczos3));
            let (far, edge) = far_pixels(&asm, &fir_lz);
            println!(
                "  Base's AVX2 Lanczos3 differs from its CLIF in {} bytes; it is more than 1 from fast_image_resize at {far} outputs ({edge} on the outermost ring), which clips its intermediate rows to bytes",
                differing(&asm, &clif)
            );
            for (name, out) in [("zimg", one(zimg_scale(true))), ("libvips", one(vips_scale()))] {
                let (far, edge) = far_pixels(&out, &img_lz);
                println!("  {name} is more than 1 from image at {far} outputs, {edge} of them on a plane's outermost ring");
            }
        }
        "cores" => {
            all_cores("Bilinear", &bilinear, &srcs);
            all_cores("2x2 average", &boxes, &srcs);
            all_cores("Lanczos3", &lanczos, &srcs);
        }
        "dump" => {
            // the frames and each table's reference, for pillow_bench.py
            let dir = std::env::var("DUMP").expect("DUMP names a directory");
            std::fs::write(format!("{dir}/src.raw"), &srcs[..FRAMES * frame_bytes(SRC)]).expect("written");
            for (name, out) in [
                ("bilinear", one(fir_scale(F::Bilinear))),
                ("box", one(fir_scale(F::Box))),
                ("lanczos", one(image_scale(I::Lanczos3))),
            ] {
                std::fs::write(format!("{dir}/ref_{name}.raw"), &out[..frame_bytes(DST)]).expect("written");
            }
        }
        "audit" => audit(&srcs),
        "gpu" => gpu::run(&srcs, &one(fir_scale(F::Box)), drv),
        _ => panic!("the mode is one, cores or gpu"),
    }
}
