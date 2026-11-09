//! The differential corpus: what `Host.Sem` says each operation computes,
//! against the machine.

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

#[test]
fn interpreter_and_machine_agree() {
    let dir = PathBuf::from(lean_artifacts::DIR);

    let corpus = Corpus::read(&dir.join("hprog_corpus/expected.json"));

    assert_eq!(
        corpus.expected.len(),
        corpus.names.len() * corpus.stride,
        "one slot per case"
    );

    let b = unsafe { base::Driver::load(&std::fs::read(dir.join("hprog_corpus.cbor")).expect("read artifact")).expect("compile") };
    let mut out = vec![0u8; corpus.expected.len()];
    b.function("main").expect("execute")(&[], &mut out);

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
