// Throughput + byte-exact verification for the proven GPU LZ4 compressor: one warp
// per block, 4096-entry shared-memory hash table, cooperative match extend + emit.
// The kernels measured here are the artifacts `Lz4CompAlgorithm.warpArtifactDSL`
// emits, which embed `serializeKernel (WP.kernel w)` — the same definition
// `Algorithm.launch_correct` is proven about.
//
// The program prints its own methodology: what is inside each timed region on both
// sides, why the launch count can be trusted, and the compression ratios alongside
// the throughputs.
//
// Corpus: `baseline/setup.sh --corpus-only` (override with LZ4_CORPUS).
// Baseline: `baseline/bench_nvcomp_compress.py`, which prints its own methodology.
use arrow_array::{Int64Array, RecordBatch};
use base::{Artifact, Base};
use std::time::Instant;

const OUT_DIR: &str = env!("OUT_DIR");
/// Set up by `baseline/setup.sh`; override with `LZ4_CORPUS`.
const CORPUS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/corpus/silesia_all.bin");
/// The prefix the shipped kernels bake in (`Lz4CompAlgorithm.corpusBytes`).
const CORPUS_BYTES: usize = 209_715_200;

fn col(b: &RecordBatch, i: usize) -> i64 {
    b.column(i).as_any().downcast_ref::<Int64Array>().unwrap().value(0)
}

fn run(name: &str, block: usize, original: &[u8]) {
    let data = original.to_vec();
    let bin = std::fs::read(format!("{OUT_DIR}/Lz4CompAlgorithm/{name}.bin"))
        .unwrap_or_else(|e| panic!("read {name}.bin: {e}"));
    let art = Artifact::from_bytes(&bin);
    let mut base = Base::new(art.setup).expect("compile");

    // Layout mirrors Algorithm.WP in Lz4CompAlgorithm.lean; asserted vs columns.
    let num_blk = CORPUS_BYTES / block;
    let len_off = block + block / 16 + 256;
    let out_stride = len_off + 8;
    let mut out = vec![0u8; num_blk * out_stride];

    let batches = base.execute_into(&art.main, &data, &mut out).expect("run");
    let b = &batches[0];
    let (launches, bytes_per_launch) = (col(b, 1) as u64, col(b, 2) as u64);
    assert_eq!(col(b, 3) as usize, block, "in_stride mismatch");
    assert_eq!(col(b, 4) as usize, out_stride, "out_stride mismatch");
    assert_eq!(col(b, 5) as usize, len_off, "len_off mismatch");
    assert_eq!(col(b, 6) as usize, num_blk, "num_blk mismatch");

    // Byte-exact verification via a standard LZ4 decoder.
    let verify = |out: &[u8], tag: &str| -> usize {
        let mut total_comp = 0usize;
        let mut bad = 0usize;
        for i in 0..num_blk {
            let bo = i * out_stride;
            let clen =
                u32::from_le_bytes(out[bo + len_off..bo + len_off + 4].try_into().unwrap()) as usize;
            if clen == 0 || clen > len_off {
                if bad < 5 { eprintln!("{tag} block {i}: bad compLen {clen}"); }
                bad += 1;
                continue;
            }
            total_comp += clen;
            match lz4_flex::block::decompress(&out[bo..bo + clen], block) {
                Ok(d) if d.len() == block && d[..] == original[i * block..(i + 1) * block] => {}
                Ok(d) => {
                    if bad < 5 { eprintln!("{tag} block {i}: decoded {} bytes, mismatch", d.len()); }
                    bad += 1;
                }
                Err(e) => {
                    if bad < 5 { eprintln!("{tag} block {i}: decode error {e:?} (clen {clen})"); }
                    bad += 1;
                }
            }
        }
        if bad > 0 {
            panic!("{name} {tag}: {bad}/{num_blk} blocks failed byte-exact verification");
        }
        total_comp
    };
    let total_comp = verify(&out, "cold");
    let ratio = CORPUS_BYTES as f64 / total_comp as f64;

    // Output checksum over the actual compressed bytes of every block — lets us
    // compare two kernels' output byte-for-byte (proof object vs original).
    let mut ck: u64 = 0xcbf29ce484222325;
    for i in 0..num_blk {
        let bo = i * out_stride;
        let clen = u32::from_le_bytes(out[bo + len_off..bo + len_off + 4].try_into().unwrap()) as usize;
        for &b in &out[bo..bo + clen] { ck ^= b as u64; ck = ck.wrapping_mul(0x100000001b3); }
    }

    for _ in 0..2 { base.execute_into(&art.main, &data, &mut out).unwrap(); }
    let runs = 5u64;
    let mut times = Vec::new();
    for _ in 0..runs {
        let t = Instant::now();
        base.execute_into(&art.main, &data, &mut out).unwrap();
        times.push(t.elapsed().as_secs_f64());
    }
    let dt: f64 = times.iter().sum();
    let total = runs * launches * bytes_per_launch;

    // Steady-state re-verification: zero the host buffer, execute once more, and
    // re-check every block byte-exact — proves the timed executes still do the work.
    out.iter_mut().for_each(|b| *b = 0);
    base.execute_into(&art.main, &data, &mut out).unwrap();
    let recheck = verify(&out, "steady-state");
    assert_eq!(recheck, total_comp, "steady-state compressed size drifted");

    let (tmin, tmax) = (
        times.iter().cloned().fold(f64::MAX, f64::min) * 1e3,
        times.iter().cloned().fold(0.0f64, f64::max) * 1e3,
    );
    let nv = if block == 32768 { (2.27, 2.42, 1.930) } else { (2.11, 2.31, 1.986) };
    println!(
        "{name}: {block} B blocks x {num_blk}, {launches} launches/exec x {:.1} MB\n\
         \x20 verified byte-exact by lz4_flex, cold AND steady-state; output checksum {ck:016x}\n\
         \x20 ratio {ratio:.3}:1   (nvCOMP at this chunk size: {:.3}:1)\n\
         \x20 per-exec {tmin:.0}-{tmax:.0} ms over {runs} runs -> {:.2} GB/s of input\n\
         \x20 vs nvCOMP {:.2} GB/s raw / {:.2} GB/s marginal  =>  {:.2}x / {:.2}x",
        bytes_per_launch as f64 / 1e6,
        nv.2,
        total as f64 / dt / 1e9,
        nv.0,
        nv.1,
        total as f64 / dt / 1e9 / nv.0,
        total as f64 / dt / 1e9 / nv.1,
    );
}

fn main() {
    let path = std::env::var("LZ4_CORPUS").unwrap_or_else(|_| CORPUS.to_string());
    let corpus = std::fs::read(&path).unwrap_or_else(|e| {
        panic!("read {path}: {e} — run applications/lz4-ptx/baseline/setup.sh --corpus-only")
    });
    assert!(corpus.len() >= CORPUS_BYTES, "corpus too small: {} bytes", corpus.len());
    let original = &corpus[..CORPUS_BYTES];

    println!(
"warp-cooperative GPU LZ4 compression, {:.1} MB Silesia prefix (= Lz4CompAlgorithm.corpusBytes)

WHAT IS TIMED (ours): one Base::execute_into = device buffer allocation + H2D upload
  of the whole corpus + every kernel launch + D2H download of the whole output.
  JIT compilation is NOT timed: Base::new runs before the timed loop. 2 warmup execs,
  then {} timed. Transfers are ~4% of the number below (see LAUNCH MODEL).
WHAT IS TIMED (nvCOMP, {}): device-side encode only, input pre-staged on the GPU
  — host transfers excluded, i.e. generous to nvCOMP. Its encode() also allocates its
  output buffer per call; `marginal` below removes that fixed cost (a further 6-9%),
  `raw` does not. Both are reported. Same corpus prefix, same chunk sizes, best of 30.
CORRECTNESS: every emitted block is decoded by lz4_flex — a third-party decoder, not
  our spec — and compared byte-for-byte with the input, on the cold run and again
  after the timed runs from a zeroed buffer. The compression ratio is printed next to
  nvCOMP's, so \"faster\" cannot be hiding \"compresses less\".
LAUNCH MODEL: throughput divides by launches x bytes_per_launch, so the launches must
  really happen. Rebuilding with Lz4CompAlgorithm.rLaunches = 5 and solving
  T = T0 + n*L against the 20-launch time gave L = 48.5 ms/pass and T0 = 48.0 ms at
  32 KiB (T0 = PCIe traffic + allocation), predicting the 20-launch time to under 1 ms.
  So: kernel-only rate is ~4.3 GB/s at 32 KiB, ~3.7 GB/s at 64 KiB; the numbers below
  are the conservative transfers-included ones.
",
        CORPUS_BYTES as f64 / 1e6,
        5,
        "5.3.0.16",
    );

    run("lz4_comp_warpdsl", 32768, original);
    println!();
    run("lz4_comp_warpdsl64", 65536, original);
}
