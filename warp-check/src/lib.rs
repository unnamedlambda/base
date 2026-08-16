//! The host side of a proven-kernel demo: the committed fold order, the
//! byte conversions, and the timing.
//!
//! Every demo in `applications/` that checks a warp kernel needs the same CPU
//! reference — sequential within a lane, then a five-round butterfly across
//! lanes. Written once here, because a demo that writes its own butterfly and
//! gets it wrong reports "bit-exact" against the wrong thing, and that failure
//! is invisible: the kernel and the reference agree on being wrong together
//! only if the mistake is shared, so an independent copy per demo is not
//! redundancy, it is three chances to disagree with Lean.
//!
//! The one authority is `lean/lib/AlgorithmLib/ML/`: `bflyFold` is
//! [`butterfly`], `dotLane` is [`Walk::Vec4`], `dotStridedLane` is
//! [`Walk::Strided`], and `Sched.idx` is [`Walk::idx`].

use base::Base;
use base_types::Algorithm;

// ---------------------------------------------------------------------------
// The committed fold order
// ---------------------------------------------------------------------------

/// `bflyFold`: five xor rounds over 32 lanes, read out at lane 0.
///
/// Not a sum — a specific tree. Two orders that both "add 32 numbers" differ in
/// the last bits at `f32`, which is why the Lean theorems are stated against
/// this shape rather than against a mathematical sum.
pub fn butterfly(mut lane: [f32; 32]) -> f32 {
    for m in [16usize, 8, 4, 2, 1] {
        let prev = lane;
        for l in 0..32 {
            lane[l] = prev[l] + prev[l ^ m];
        }
    }
    lane[0]
}

/// The walk a schedule emits — the Rust image of Lean's `Sched.idx`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Walk {
    /// Four contiguous elements per lane per trip, warp striding by 128.
    Vec4,
    /// One element per lane per trip, warp striding by 32.
    Strided,
    /// One element per lane per trip, lane `l` walking the run `l·(n/32)…`.
    Blocked,
}

impl Walk {
    /// Trips needed to cover `n` elements with one warp.
    pub fn trips(self, n: usize) -> usize {
        match self {
            Walk::Vec4 => n / 128,
            Walk::Strided | Walk::Blocked => n / 32,
        }
    }

    /// Elements consumed per lane per trip.
    pub fn width(self) -> usize {
        match self {
            Walk::Vec4 => 4,
            Walk::Strided | Walk::Blocked => 1,
        }
    }

    /// The first index lane `l` reads on trip `i`; `Vec4` reads this and the
    /// three after it.
    pub fn idx(self, n: usize, base: usize, i: usize, l: usize) -> usize {
        match self {
            Walk::Vec4 => i * 128 + l * 4 + base,
            Walk::Strided => i * 32 + l + base,
            Walk::Blocked => l * (n / 32) + i + base,
        }
    }
}

/// **The general committed fold**: `trips` trips of `width` products per lane,
/// accumulated in load order, then the butterfly.
///
/// The two index functions take `(trip, lane)`, which is what lets a caller
/// express a transposed walk — a column of a row-major matrix strides by the
/// row length and no `Walk` describes it.
pub fn dot_by(
    trips: usize,
    width: usize,
    a: &[f32],
    b: &[f32],
    ia: impl Fn(usize, usize) -> usize,
    ib: impl Fn(usize, usize) -> usize,
) -> f32 {
    let mut lane = [0f32; 32];
    for (l, acc) in lane.iter_mut().enumerate() {
        for t in 0..trips {
            let (oa, ob) = (ia(t, l), ib(t, l));
            for q in 0..width {
                *acc += a[oa + q] * b[ob + q];
            }
        }
    }
    butterfly(lane)
}

/// `Σ a[base_a + k]·b[base_b + k]` for `k < n`, folded the way `w` folds it.
///
/// With `a` and `b` the same slice and one base, this is the sum of squares
/// RMSNorm needs — the same schema, which is why it costs no extra reference.
pub fn dot(w: Walk, a: &[f32], b: &[f32], base_a: usize, base_b: usize, n: usize) -> f32 {
    dot_by(
        w.trips(n),
        w.width(),
        a,
        b,
        |t, l| w.idx(n, base_a, t, l),
        |t, l| w.idx(n, base_b, t, l),
    )
}

// ---------------------------------------------------------------------------
// Bytes
// ---------------------------------------------------------------------------

/// Little-endian `f32`s out of a device download.
pub fn floats(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// …and back, for a host upload.
pub fn le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// How far a result is from its reference.
#[derive(Clone, Copy, Debug)]
pub struct Diff {
    /// Values matching bit-for-bit.
    pub exact: usize,
    /// Values compared.
    pub total: usize,
    /// Largest relative error, with a floor so a near-zero reference does not
    /// turn a rounding into a headline.
    pub worst: f32,
    /// Where `worst` occurred.
    pub worst_at: usize,
    /// Largest *absolute* error, and the reference value there.
    ///
    /// Reported alongside the relative one because relative error is not
    /// meaningful near a zero of the function under test: `silu'` has a root
    /// near `z = -1.28`, and an element that lands on it shows a large relative
    /// error for an absolute error of a few ulps. A relative-only report turns
    /// that into a headline; a relative-only *assertion* turns it into a false
    /// failure, which is worse.
    pub worst_abs: f32,
    /// The reference value at `worst_at` — what the relative error is against.
    pub ref_at_worst: f32,
}

impl Diff {
    pub fn is_exact(&self) -> bool {
        self.exact == self.total
    }
}

impl std::fmt::Display for Diff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "bit-exact {}/{}   worst rel {:.2e}",
            self.exact, self.total, self.worst
        )
    }
}

/// Compare against a reference evaluated per index.
pub fn compare(got: &[f32], want: impl Fn(usize) -> f32) -> Diff {
    let mut d = Diff {
        exact: 0, total: got.len(), worst: 0.0, worst_at: 0,
        worst_abs: 0.0, ref_at_worst: 0.0,
    };
    for (i, &g) in got.iter().enumerate() {
        let w = want(i);
        if g.to_bits() == w.to_bits() {
            d.exact += 1;
        }
        let abs = (g - w).abs();
        if abs > d.worst_abs {
            d.worst_abs = abs;
        }
        let e = abs / w.abs().max(1e-6);
        if e > d.worst {
            d.worst = e;
            d.worst_at = i;
            d.ref_at_worst = w;
        }
    }
    d
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

/// Peak memory bandwidth of the card these demos report against — an RTX 3060
/// (12 GB GDDR6, 192-bit at 15 Gbps: 24 B × 15 GT/s = 360 GB/s). Every
/// "% of roofline" below is against this number, so it is stated once and
/// named.
pub const ROOFLINE_GBS: f64 = 360.0;

/// Seconds per launch, after one warm-up.
pub fn time(base: &mut Base, alg: &Algorithm, reps: usize) -> f64 {
    base.execute_into(alg, b"", &mut []).expect("warm-up launch");
    let t = std::time::Instant::now();
    for _ in 0..reps {
        base.execute_into(alg, b"", &mut []).expect("timed launch");
    }
    t.elapsed().as_secs_f64() / reps as f64
}

/// Achieved bandwidth for a kernel that must move `bytes` bytes.
pub fn gbs(bytes: usize, secs: f64) -> f64 {
    bytes as f64 / secs / 1e9
}

/// …as a fraction of [`ROOFLINE_GBS`].
pub fn roofline(bytes: usize, secs: f64) -> f64 {
    gbs(bytes, secs) / ROOFLINE_GBS
}

// ---------------------------------------------------------------------------
// Reproducible inputs
// ---------------------------------------------------------------------------

/// A deterministic uniform stream on `[-1, 1)`, so a failing demo is a failing
/// demo again on the next run.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Lcg(seed)
    }

    pub fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (u32::MAX >> 1) as f32 * 2.0 - 1.0
    }

    pub fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three walks visit the same index set — the coverage half of what
    /// `Sched.fold_eq_flatSum` proves, checked here so the Rust image cannot
    /// drift from the Lean one without a test failing.
    #[test]
    fn walks_cover_the_same_indices() {
        let n = 256;
        for w in [Walk::Vec4, Walk::Strided, Walk::Blocked] {
            let mut seen: Vec<usize> = Vec::new();
            for t in 0..w.trips(n) {
                for l in 0..32 {
                    for q in 0..w.width() {
                        seen.push(w.idx(n, 0, t, l) + q);
                    }
                }
            }
            seen.sort_unstable();
            assert_eq!(seen, (0..n).collect::<Vec<_>>(), "{w:?} must cover 0..n exactly once");
        }
    }

    /// A butterfly over ones is 32 — a shape check, not a numerics claim.
    #[test]
    fn butterfly_sums() {
        assert_eq!(butterfly([1.0; 32]), 32.0);
    }
}
