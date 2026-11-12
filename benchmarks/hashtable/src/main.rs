//! u64 -> u64 hash tables: base's SwissTable (`lean/algorithms/Bench/HashTable.lean`)
//! against hashbrown (Rust's, with its default foldhash), measured as the
//! field measures them: for each size, the time an operation of
//!
//! * `build`: n distinct keys inserted into an empty table, no reserve;
//! * `hit`: every key looked up, in a random order;
//! * `miss`: n keys that are not in the table looked up;
//! * `zipf`: n lookups of keys drawn Zipfian (0.99) over the table's.
//!
//! Every lookup pass's sum of values and count found is checked equal across
//! the engines (and to what it must be) before its time counts. Each
//! (engine, operation) is timed SAMPLES times after one untimed run; every
//! sample is printed, as one JSON line per measure:
//!
//!     hashtable-bench <hashtable.cbor>      # lines to stdout
use base::Driver;
use std::hint::black_box;
use std::time::Instant;

const SIZES: [usize; 4] = [1 << 10, 1 << 16, 1 << 20, 1 << 24];
const SAMPLES: usize = 7;

/// SplitMix64: the keys, the order and the Zipfian draws, from fixed seeds.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        ((self.next() as u128 * n as u128) >> 64) as usize
    }
}

/// n distinct keys, and n more distinct from them (the misses).
fn keys(n: usize, rng: &mut Rng) -> (Vec<u64>, Vec<u64>) {
    let mut seen = hashbrown::HashSet::with_capacity(2 * n);
    let mut all = Vec::with_capacity(2 * n);
    while all.len() < 2 * n {
        let k = rng.next();
        if seen.insert(k) {
            all.push(k);
        }
    }
    let miss = all.split_off(n);
    (all, miss)
}

/// n draws Zipfian 0.99 over `keys` (rank r with weight r^-0.99), by
/// inverting the cumulative weights.
fn zipf(keys: &[u64], n: usize, rng: &mut Rng) -> Vec<u64> {
    let mut cdf = Vec::with_capacity(keys.len());
    let mut s = 0.0;
    for r in 1..=keys.len() {
        s += (r as f64).powf(-0.99);
        cdf.push(s);
    }
    (0..n)
        .map(|_| {
            let u = (rng.next() >> 11) as f64 / (1u64 << 53) as f64 * s;
            keys[cdf.partition_point(|&c| c < u).min(keys.len() - 1)]
        })
        .collect()
}

fn shuffled(keys: &[u64], rng: &mut Rng) -> Vec<u64> {
    let mut v = keys.to_vec();
    for i in (1..v.len()).rev() {
        v.swap(i, rng.below(i + 1));
    }
    v
}

fn samples(mut run: impl FnMut()) -> Vec<f64> {
    run();
    (0..SAMPLES)
        .map(|_| {
            let t = Instant::now();
            run();
            t.elapsed().as_secs_f64()
        })
        .collect()
}

fn emit(engine: &str, op: &str, n: usize, s: &[f64]) {
    let mut v = s.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ns = v[v.len() / 2] / n as f64 * 1e9;
    let list: Vec<String> = s.iter().map(|x| format!("{x:.9}")).collect();
    println!(
        "{{\"engine\": \"{engine}\", \"op\": \"{op}\", \"n\": {n}, \"ns_per_op\": {ns:.4}, \"samples\": [{}]}}",
        list.join(", ")
    );
}

fn main() {
    let art = std::fs::read(std::env::args().nth(1).expect("usage: hashtable-bench <hashtable.cbor>")).unwrap();
    // SAFETY: the artifact is base's own, written by its generator in run.sh.
    let drv = unsafe { Driver::load(&art) }.expect("the artifact loads");
    let (clear, insert, get) = (drv.function("clear").unwrap(), drv.function("insert").unwrap(), drv.function("get").unwrap());
    let mut out = [0u8; 16];
    for n in SIZES {
        let mut rng = Rng(n as u64);
        let (ks, miss) = keys(n, &mut rng);
        let vals: Vec<u64> = (0..n as u64).collect();
        let hit = shuffled(&ks, &mut rng);
        let zk = zipf(&ks, n, &mut rng);
        let batch = |xs: &[u64]| -> Vec<u8> {
            let mut b = (xs.len() as u64).to_le_bytes().to_vec();
            for x in xs {
                b.extend_from_slice(&x.to_le_bytes());
            }
            b
        };
        let mut ins = batch(&ks);
        for v in &vals {
            ins.extend_from_slice(&v.to_le_bytes());
        }
        let (bh, bm, bz) = (batch(&hit), batch(&miss), batch(&zk));

        // hashbrown
        let build_hb = || {
            let mut m: hashbrown::HashMap<u64, u64> = hashbrown::HashMap::new();
            for (k, v) in ks.iter().zip(&vals) {
                m.insert(*k, *v);
            }
            m
        };
        let s = samples(|| { black_box(build_hb()); });
        emit("hashbrown", "build", n, &s);
        let m = build_hb();
        let look = |xs: &[u64]| -> (u64, u64) {
            let (mut sum, mut hits) = (0u64, 0u64);
            for k in xs {
                if let Some(v) = m.get(black_box(k)) {
                    sum = sum.wrapping_add(*v);
                    hits += 1;
                }
            }
            (sum, hits)
        };
        let want = [look(&hit), look(&miss), look(&zk)];
        assert_eq!(want[0].1, n as u64);
        assert_eq!(want[1].1, 0);
        for (op, xs) in [("hit", &hit), ("miss", &miss), ("zipf", &zk)] {
            let s = samples(|| { black_box(look(xs)); });
            emit("hashbrown", op, n, &s);
        }

        // base
        let s = samples(|| { clear(&[], &mut out); insert(&ins, &mut out); });
        emit("base", "build", n, &s);
        clear(&[], &mut out);
        insert(&ins, &mut out);
        for ((op, b), w) in [("hit", &bh), ("miss", &bm), ("zipf", &bz)].into_iter().zip(want) {
            get(b, &mut out);
            let got = (u64::from_le_bytes(out[..8].try_into().unwrap()), u64::from_le_bytes(out[8..].try_into().unwrap()));
            assert_eq!(got, w, "base {op} at n = {n}");
            let s = samples(|| { get(b, &mut out); });
            emit("base", op, n, &s);
        }
    }
}
