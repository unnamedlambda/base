//! Port validation for the `HProg` pilots — **not** part of the trusted base.
//!
//! Each pilot is a generator rewritten as a first-order term. These tests run
//! the original generator's artifact and the term's artifact through the JIT
//! and check they produce the same bytes. The Lean interpreter is not involved
//! anywhere here: this says the rewrite preserved behavior, nothing more.
//!
//! It is deliberately the *wrong* shape for grounding a semantics, and worth
//! being explicit about why: the number of cases grows with every program
//! ported, so it could never cover anything. What grounds the CLIF model is
//! `hprog_corpus.rs`, whose cases are indexed by CLIF's vocabulary —
//! instructions and control-flow constructs — and therefore stop growing.
//!
//! Gated on `BASE_HPROG_DIR` pointing at a directory holding both artifacts of
//! each pair, which `lake env lean --run HProgPilots.lean <dir>` writes.

use base_types::Artifact;

/// Entry points of the pilot artifacts, as their generators number them.
const MAIN: u32 = 1;
const PREP: u32 = 2;
const INFER: u32 = 3;
use std::path::{Path, PathBuf};

fn artifact(dir: &Path, name: &str) -> Artifact {
    let p = dir.join(format!("{name}.json"));
    let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
}

fn dir() -> Option<PathBuf> {
    std::env::var("BASE_HPROG_DIR").ok().map(PathBuf::from)
}

// ---------------------------------------------------------------------------
// Pilot 1: the histogram — loops, an FFI read and write, byte and word traffic
// ---------------------------------------------------------------------------

fn run_histogram(a: &Artifact, input: &Path, output: &Path) -> Vec<u8> {
    let mut b = base::Base::new(a.clone()).expect("compile");
    let _ = std::fs::remove_file(output);
    let data = format!("{}\0{}\0", input.to_str().unwrap(), output.to_str().unwrap());
    b.execute(MAIN, data.as_bytes()).expect("execute");
    std::fs::read(output).expect("histogram written")
}

#[test]
fn histogram_matches_generator() {
    let Some(dir) = dir() else { return };

    let tmp = tempfile::TempDir::new().unwrap();
    let input = tmp.path().join("input.bin");
    // Deliberately not a multiple of four, so the unrolled scan and the scalar
    // tail both run.
    let values: Vec<u32> = (0..100_003u32).map(|i| (i * 7 + 13) % 256).collect();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&input, &bytes).unwrap();

    let mut counts = vec![0u64; 256];
    for v in &values {
        counts[*v as usize] += 1;
    }
    let expected: Vec<u8> = counts.iter().flat_map(|c| c.to_le_bytes()).collect();

    let gen = run_histogram(
        &artifact(&dir, "hist1_algorithm"),
        &input,
        &tmp.path().join("gen.bin"),
    );
    let hp = run_histogram(
        &artifact(&dir, "hist1_hprog"),
        &input,
        &tmp.path().join("hprog.bin"),
    );

    assert_eq!(gen, expected, "generator artifact computes the histogram");
    assert_eq!(hp, expected, "term artifact computes the histogram");
    assert_eq!(gen, hp, "byte-identical output");
}

// ---------------------------------------------------------------------------
// Pilot 2: the clamped sum — f32x4 and f64 carried through loop parameters
// ---------------------------------------------------------------------------

fn run_clamp_sum(a: &Artifact, input: &[f32]) -> f64 {
    let mut b = base::Base::new(a.clone()).expect("compile");
    let data: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut out = [0u8; 8];
    b.execute_into(MAIN, &data, &mut out).expect("execute");
    f64::from_le_bytes(out)
}

#[test]
fn clamp_sum_matches_generator() {
    let Some(dir) = dir() else { return };

    // A length that is neither a multiple of 16 nor of 4, so all three loops
    // run: the 16-wide body, the 4-wide body, and the scalar tail.
    let input: Vec<f32> = (0..1_002u32)
        .map(|i| (i as f32) * 0.001 - 0.5 + if i % 7 == 0 { 3.0 } else { 0.0 })
        .collect();

    let gen = run_clamp_sum(&artifact(&dir, "clamp_sum_algorithm"), &input);
    let hp = run_clamp_sum(&artifact(&dir, "clamp_sum_hprog"), &input);

    // The reference sums in the same order the kernels do: four vector lanes of
    // partial sums, then the horizontal reduce, then the scalar tail.
    assert_eq!(
        gen.to_bits(),
        hp.to_bits(),
        "term and generator agree bit-for-bit: {gen} vs {hp}"
    );
    // A plain left-to-right sum reassociates against the four-lane one, so the
    // agreement here is approximate by construction.
    let naive: f64 = input.iter().map(|v| v.clamp(-0.5, 0.5) as f64).sum();
    assert!(
        (gen - naive).abs() < 1e-4 * naive.abs().max(1.0),
        "and both agree with a plain sum: {gen} vs {naive}"
    );
}

// ---------------------------------------------------------------------------
// Nesting: a loop inside a loop, and a branch whose arms export a value
// ---------------------------------------------------------------------------

#[test]
fn nested_loops_and_branch_compute() {
    let Some(dir) = dir() else { return };
    let a = artifact(&dir, "nested_hprog");
    let mut b = base::Base::new(a.clone()).expect("compile");

    let tmp = tempfile::TempDir::new().unwrap();
    let out = tmp.path().join("nested.bin");
    // The copy loop reads until NUL, so the terminator has to be in the buffer.
    let data = format!("{}\0", out.to_str().unwrap());
    b.execute(MAIN, data.as_bytes()).expect("execute");

    let bytes = std::fs::read(&out).expect("output written");
    let sum = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    assert_eq!(sum, 66, "sum over i<3, j<4 of 4i+j");
    let branched = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    assert_eq!(branched, 1066, "the taken arm's export reaches the join");
}

// ---------------------------------------------------------------------------
// Pilot 3: RMSNorm's host side — four functions, i32/i64 callees, a branch
// ---------------------------------------------------------------------------

/// `load`, then `prep`, then `infer` — the stage order the app drives.
fn run_rmsnorm(a: &Artifact, weights: &[f32], x: &[f32]) -> Option<Vec<f32>> {
    let n = x.len();
    let mut data = Vec::new();
    data.extend_from_slice(&(n as u64).to_le_bytes());
    data.extend(weights.iter().flat_map(|v| v.to_le_bytes()));
    let mut b = base::Base::new(a.clone()).ok()?;
    b.execute(MAIN, &data).ok()?;

    let xs: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    b.execute(PREP, &xs).ok()?;
    let mut out = vec![0u8; n * 4];
    b.execute_into(INFER, &xs, &mut out).ok()?;
    Some(
        out.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    )
}

#[test]
fn rmsnorm_matches_generator() {
    let Some(dir) = dir() else { return };

    let n = 512;
    let weights: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let x: Vec<f32> = (0..n).map(|i| ((i % 17) as f32) - 8.0).collect();

    // Both artifacts drive the same device; without one, neither runs and there
    // is nothing to compare.
    let Some(gen) = run_rmsnorm(&artifact(&dir, "cuda_rmsnorm"), &weights, &x) else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let hp = run_rmsnorm(&artifact(&dir, "cuda_rmsnorm_hprog"), &weights, &x)
        .expect("term artifact runs wherever the generator's does");

    assert_eq!(gen.len(), n, "generator produced a full vector");
    assert_eq!(
        gen.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        hp.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "term and generator drive the kernel to the same result"
    );

    // The host code under test is the launch sequence, but a wrong launch would
    // still produce *some* numbers, so check them against the definition.
    let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (ms + 1e-5).sqrt();
    for (i, got) in hp.iter().enumerate() {
        let want = x[i] * scale * weights[i];
        assert!(
            (got - want).abs() < 1e-3,
            "lane {i}: got {got}, want {want}"
        );
    }
}
