//! On-disk Qwen2 CLI: weights are streamed layer-by-layer from disk into a
//! single-layer GPU working set, and the KV cache is also backed by a file on
//! disk.  Behaviorally identical to the in-memory `qwen2` binary; only the
//! storage strategy differs.  Useful on systems where the model is larger than
//! VRAM (e.g. point at a 32B weights file on NVMe).

use base::{init_tracing, Base, Artifact};

const QWEN2_ON_DISK_BINARY: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/Qwen2OnDiskAlgorithm/qwen2_on_disk.bin"
));

/// Path to the on-disk KV cache file.  Must match `KV_CACHE_PATH_OFF` in the
/// Lean algorithm.  We delete it at startup so each session begins with a
/// fresh, empty cache (otherwise stale K/V from a previous run would leak in).
const KV_CACHE_FILE: &str = "/tmp/qwen2_kvcache.bin";

fn main() {
    init_tracing();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: qwen2_on_disk <weights.bin> <tokenizer.bin>");
        eprintln!();
        eprintln!("The weights file can live anywhere on disk (e.g. /mnt/nvme/qwen-32b.bin).");
        eprintln!("Weights are streamed layer-by-layer into a small GPU working set; the model");
        eprintln!("never has to fit in VRAM.  The KV cache is also disk-backed.");
        eprintln!();
        eprintln!("KV cache file: {KV_CACHE_FILE} (deleted on startup).");
        std::process::exit(1);
    }
    let weights_path = &args[1];
    let tokenizer_path = &args[2];

    let _ = std::fs::remove_file(KV_CACHE_FILE);

    let mut data = Vec::new();
    for s in [weights_path.as_str(), tokenizer_path.as_str()] {
        data.extend_from_slice(s.as_bytes());
        data.push(0);
    }

    let artifact = Artifact::from_bytes(QWEN2_ON_DISK_BINARY);
    let mut base = Base::new(artifact.config).expect("Base::new");

    eprintln!("Starting qwen2_on_disk (weights={weights_path}, tokenizer={tokenizer_path})");
    base.execute_into(&artifact.main, &data, &mut [])
        .expect("qwen2_on_disk run");
}
