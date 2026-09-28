//! The Rust side: each workload's kernels, and the inputs they share with
//! Base. Every kernel takes the inputs and answers a `u64`, which Base's
//! entry for the same workload must match. No fast-math: neither compiler
//! may reorder a float operation, so both compute the same bits.

use std::alloc::{alloc_zeroed, dealloc, Layout};

/// Elements on a 64-byte boundary, so no access straddles a cache line
/// because of where the allocator put the buffer.
pub struct Lines<T: Copy> {
    ptr: *mut T,
    len: usize,
}

impl<T: Copy> Lines<T> {
    fn layout(len: usize) -> Layout {
        Layout::array::<T>(len.max(1)).unwrap().align_to(64).unwrap()
    }

    pub fn from_slice(v: &[T]) -> Self {
        let ptr = unsafe { alloc_zeroed(Self::layout(v.len())) } as *mut T;
        assert!(!ptr.is_null());
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), ptr, v.len()) };
        Lines { ptr, len: v.len() }
    }
}

impl Lines<u8> {
    pub fn zeroed(len: usize) -> Self {
        let ptr = unsafe { alloc_zeroed(Self::layout(len)) };
        assert!(!ptr.is_null());
        Lines { ptr, len }
    }
}

impl<T: Copy> Drop for Lines<T> {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr as *mut u8, Self::layout(self.len)) }
    }
}

impl<T: Copy> std::ops::Deref for Lines<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T: Copy> std::ops::DerefMut for Lines<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

/// Everything the kernels read, generated deterministically.
pub struct In {
    /// About 1 MB of words from a small vocabulary, spaces and newlines.
    pub text: Lines<u8>,
    /// 2^16 floats in [-1, 1).
    pub f32a: Lines<f32>,
    /// 2^18 slots, `perm[k]` the next slot of one cycle through all of them.
    pub perm: Lines<u32>,
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

pub fn make_input() -> In {
    let mut r = Rng(0x9E3779B97F4A7C15);
    let words: [&[u8]; 12] = [
        b"that", b"the", b"of", b"and", b"benchmark", b"a", b"compiler", b"to",
        b"in", b"is", b"register", b"loop",
    ];
    let mut text = Vec::with_capacity(1 << 20);
    while text.len() < (1 << 20) - 16 {
        text.extend_from_slice(words[r.below(12) as usize]);
        text.push(if r.below(10) == 0 { b'\n' } else { b' ' });
    }
    let f32a: Vec<f32> = (0..(1 << 16)).map(|_| r.unit() as f32).collect();
    // one cycle through every slot, so the chase never short-circuits
    let n = 1usize << 18;
    let mut order: Vec<u32> = (0..n as u32).collect();
    for k in (1..n).rev() {
        order.swap(k, r.below(k as u64 + 1) as usize);
    }
    let mut perm = vec![0u32; n];
    for k in 0..n {
        perm[order[k] as usize] = order[(k + 1) % n];
    }
    In {
        text: Lines::from_slice(&text),
        f32a: Lines::from_slice(&f32a),
        perm: Lines::from_slice(&perm),
    }
}

// ---------------------------------------------------------------------------
// Where the CLIF says what LLVM's loop says
// ---------------------------------------------------------------------------

/// A count of each byte of the text, then the counts weighted by byte value.
#[no_mangle]
#[inline(never)]
pub fn k_histogram_low(i: &In) -> u64 {
    let mut h = [0u64; 256];
    let hp = h.as_mut_ptr();
    let p = i.text.as_ptr();
    let n = i.text.len();
    let mut k = 0;
    while k < n {
        unsafe { *hp.add(*p.add(k) as usize) += 1 };
        k += 1;
    }
    let mut s = 0u64;
    let mut j = 0;
    while j < 256 {
        s += unsafe { *hp.add(j) } * (j as u64 + 1);
        j += 1;
    }
    s
}

/// The side of mandel's grid, and the most iterations a point gets.
pub const MANDEL: usize = 128;
pub const MANDEL_ITER: u64 = 256;

/// The Mandelbrot set on a 128 x 128 grid over [-2, 1] x [-1.5, 1.5]: the
/// total of every point's iteration count. Each step waits on the last, so
/// the loop runs at the latency of its multiplies and adds.
#[no_mangle]
#[inline(never)]
pub fn k_mandel_low(_: &In) -> u64 {
    let d = 3.0 / MANDEL as f64;
    let mut total = 0u64;
    let mut py = 0;
    while py < MANDEL {
        let ci = py as f64 * d - 1.5;
        let mut px = 0;
        while px < MANDEL {
            let cr = px as f64 * d - 2.0;
            let (mut zr, mut zi) = (0.0f64, 0.0f64);
            let mut k = 0;
            while k < MANDEL_ITER {
                let (zr2, zi2) = (zr * zr, zi * zi);
                if zr2 + zi2 > 4.0 {
                    break;
                }
                zi = (zr + zr) * zi + ci;
                zr = zr2 - zi2 + cr;
                k += 1;
            }
            total += k;
            px += 1;
        }
        py += 1;
    }
    total
}

/// Once round the permutation's cycle, from slot 0, summing the slots: each
/// load's address is the last load's value, and the table is twice L2.
#[no_mangle]
#[inline(never)]
pub fn k_chase_low(i: &In) -> u64 {
    let p = i.perm.as_ptr();
    let mut at = 0u32;
    let mut s = 0u64;
    let mut k = 0;
    while k < i.perm.len() {
        at = unsafe { *p.add(at as usize) };
        s += at as u64;
        k += 1;
    }
    s
}

// ---------------------------------------------------------------------------
// Where it cannot
// ---------------------------------------------------------------------------

/// poly's coefficients, lowest degree first: powers of two, so the generator
/// writes each as an exact `f32`.
pub const POLY: [f32; 5] = [1.0, -0.5, 0.25, -0.125, 0.0625];

#[inline(always)]
fn poly(x: f32) -> f32 {
    let mut y = POLY[4];
    let mut d = 4;
    while d > 0 {
        d -= 1;
        y = y * x + POLY[d];
    }
    y
}

/// A degree-4 polynomial at every `f32`, the answer the XOR of the results'
/// bits: each element independent, so LLVM runs it eight lanes at a time.
#[no_mangle]
#[inline(never)]
pub fn k_poly_idiom(i: &In) -> u64 {
    i.f32a.iter().fold(0u32, |h, &x| h ^ poly(x).to_bits()) as u64
}

/// The stream workload's buffers, `STREAM_N` u64s (32 MB) each: twice the
/// last-level cache, so a store that first reads its line in costs a read of
/// memory for every line it writes. The runner points them at the input and
/// output it hands Base's entries.
pub static STREAM_SRC: std::sync::atomic::AtomicPtr<u64> = std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());
pub static STREAM_DST: std::sync::atomic::AtomicPtr<u64> = std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());
pub const STREAM_N: usize = 4 << 20;

fn stream_bufs() -> (*const u64, *mut u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (STREAM_SRC.load(Relaxed), STREAM_DST.load(Relaxed))
}

/// Copies `n` u64s with non-temporal stores, 128 bytes a trip. Its arguments
/// are Base's, so LLVM's code for it is the stream body in `CpuBenchAsm`.
/// (`copy_from_slice` also streams at this size but swings 4x with the
/// buffers' relative placement, so it is not the reference.)
///
/// # Safety
///
/// `src` and `dst` are 64-byte aligned, valid for `n` u64s and disjoint;
/// `n` is a nonzero multiple of 16; the CPU has AVX.
#[inline(never)]
pub unsafe extern "C" fn stream_nt(src: *const u64, n: usize, dst: *mut u64) -> u64 {
    use std::arch::x86_64::{__m256i, _mm256_load_si256, _mm256_stream_si256, _mm_sfence};
    let (s, d) = (src as *const __m256i, dst as *mut __m256i);
    let mut k = 0;
    while k < n / 4 {
        for j in 0..4 {
            _mm256_stream_si256(d.add(k + j), _mm256_load_si256(s.add(k + j)));
        }
        k += 4;
    }
    _mm_sfence();
    *dst.add(n / 2) ^ *dst.add(n - 1)
}

/// Taking its address keeps LLVM from specializing it to its one caller.
#[used]
static STREAM_NT: unsafe extern "C" fn(*const u64, usize, *mut u64) -> u64 = stream_nt;

#[no_mangle]
#[inline(never)]
pub fn k_stream_best(_: &In) -> u64 {
    let (a, b) = stream_bufs();
    // SAFETY: the runner points both at disjoint `Lines` of STREAM_N u64s,
    // and refuses to run unless the AVX bodies loaded.
    unsafe { stream_nt(a, STREAM_N, b) }
}
