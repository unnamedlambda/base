//! Runs the differential corpus: what `HProgSem` says each operation computes,
//! against what the machine does with the instruction `clif_decode` emitted.
//!
//! Two independent things can fail here, and the test tells them apart by which
//! case names disagree:
//!
//! * `base/src/clif_decode.rs` maps a term node to the wrong Cranelift builder
//!   — the migration's own risk, and the reason `udiv`/`sdiv` and `ctz`/`popcnt`
//!   appear here on operands that separate them.
//! * `HProgSem.evalOp` states Cranelift's semantics wrongly — the assumption
//!   that file names and cannot prove.
//!
//! The corpus is generated from `HProgCorpus.lean` into the build directory and
//! cached against that file's contents, so what the machine is compared against
//! is always what the model in the tree says. A corpus committed beside the test
//! would agree with the model only until someone changed the model, and this is
//! the comparison the migration rests on.
//!
//! Set `BASE_HPROG_DIR` to a directory holding both artifacts to run one
//! generated elsewhere.

use base_types::Artifact;

use std::path::PathBuf;

struct Corpus {
    stride: usize,
    names: Vec<String>,
    /// 0 = compare bytes; 4 or 8 = require a NaN of that width, payload free.
    modes: Vec<usize>,
    expected: Vec<u8>,
}

impl Corpus {
    fn read(path: &std::path::Path) -> Corpus {
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect("read corpus"))
                .expect("corpus shape");
        Corpus {
            stride: v["stride"].as_u64().expect("stride") as usize,
            names: v["names"]
                .as_array()
                .expect("names")
                .iter()
                .map(|n| n.as_str().expect("name").to_string())
                .collect(),
            modes: v["modes"]
                .as_array()
                .expect("modes")
                .iter()
                .map(|m| m.as_u64().expect("mode") as usize)
                .collect(),
            expected: v["expected"]
                .as_array()
                .expect("expected")
                .iter()
                .map(|b| b.as_u64().expect("byte") as u8)
                .collect(),
        }
    }
}

/// Where the build wrote the corpus.
///
/// `lean-artifacts` builds and runs every generator, so this test reads what
/// that produced rather than driving `lake` a second time.
fn generated_corpus() -> PathBuf {
    PathBuf::from(lean_artifacts::DIR).join("HProgCorpus")
}

#[test]
fn interpreter_and_machine_agree() {
    let dir = match std::env::var("BASE_HPROG_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => generated_corpus(),
    };

    let corpus = Corpus::read(&dir.join("expected/hprog_corpus_expected.json"));
    let a: Artifact = serde_json::from_str(
        &std::fs::read_to_string(dir.join("hprog_corpus.json")).expect("read artifact"),
    )
    .expect("artifact shape");

    assert_eq!(
        corpus.expected.len(),
        corpus.names.len() * corpus.stride,
        "one slot per case"
    );

    let mut b = base::Base::new(a).expect("compile");
    let mut out = vec![0u8; corpus.expected.len()];
    b.execute("main", &[], &mut out).expect("execute");

    /// A NaN of `w` bytes, read little-endian from the front of `b`.
    fn is_nan(b: &[u8], w: usize) -> bool {
        match w {
            4 => f32::from_le_bytes(b[0..4].try_into().unwrap()).is_nan(),
            _ => f64::from_le_bytes(b[0..8].try_into().unwrap()).is_nan(),
        }
    }

    let mut bad = Vec::new();
    let mut nan_class = 0;
    for (i, name) in corpus.names.iter().enumerate() {
        let lo = i * corpus.stride;
        let hi = lo + corpus.stride;
        let (got, want) = (&out[lo..hi], &corpus.expected[lo..hi]);
        let ok = match corpus.modes[i] {
            0 => got == want,
            w => {
                nan_class += 1;
                is_nan(got, w)
            }
        };
        if !ok {
            bad.push(format!(
                "  {name}: lean {:02x?} != jit {:02x?}{}",
                want,
                got,
                if corpus.modes[i] != 0 { "  (expected any NaN)" } else { "" }
            ));
        }
    }

    assert!(
        bad.is_empty(),
        "{} of {} cases disagree:\n{}",
        bad.len(),
        corpus.names.len(),
        bad.join("\n")
    );
    eprintln!(
        "{} cases agree ({} of them compared as NaN-of-width, payload free)",
        corpus.names.len(),
        nan_class
    );
}
