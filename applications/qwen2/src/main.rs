//! Thin wrapper around the Qwen2 Lean algorithm.
//!
//! Packs `weights_path\0tokenizer_path\0` into a single buffer and runs one
//! `Base::execute_into` — Lean does everything (parse args, load weights, load
//! tokenizer, stdin/stdout chat loop). End-to-end behavior is covered by
//! `tests/golden_cli.rs`.

use base::{init_tracing, Base, Artifact};

const QWEN2_BINARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/Qwen2Algorithm/qwen2.bin"));

fn main() {
    init_tracing();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: qwen2 <weights.bin> <tokenizer.bin>");
        std::process::exit(1);
    }
    let weights_path = &args[1];
    let tokenizer_path = &args[2];

    let mut data = Vec::new();
    for s in [weights_path.as_str(), tokenizer_path.as_str()] {
        data.extend_from_slice(s.as_bytes());
        data.push(0);
    }

    let artifact = Artifact::from_bytes(QWEN2_BINARY);
    let mut base = Base::new(artifact.config).expect("Base::new");

    eprintln!("Starting qwen2 (weights={weights_path}, tokenizer={tokenizer_path})");
    base.execute_into(&artifact.main, &data, &mut [])
        .expect("qwen2 run");
}
