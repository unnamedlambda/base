//! Qwen2 chat: passes the weights and tokenizer paths to the Lean program,
//! which does everything else.

use base::{Driver, Artifact};

const QWEN2_BINARY: &[u8] = lean_artifacts::QWEN2;


pub fn main(args: Vec<String>) {

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

    let artifact = Artifact::from_bytes(QWEN2_BINARY).expect("the build checked this artifact");
    let base = Driver::load(artifact).expect("Driver::load");

    eprintln!("Starting qwen2 (weights={weights_path}, tokenizer={tokenizer_path})");
    unsafe { base.execute("main", &data, &mut []) }
        .expect("qwen2 run");
}
