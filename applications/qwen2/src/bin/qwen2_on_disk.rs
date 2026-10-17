//! Qwen2 with weights streamed from disk layer by layer and the KV cache in a
//! file, for models larger than VRAM. Otherwise the same as `qwen2`.

use base::{Driver, Artifact};

const QWEN2_ON_DISK_BINARY: &[u8] = lean_artifacts::QWEN2_ON_DISK;


/// Path to the on-disk KV cache file.  Must match `KV_CACHE_PATH_OFF` in the
/// Lean algorithm.  We delete it at startup so each session begins with a
/// fresh, empty cache (otherwise stale K/V from a previous run would leak in).
const KV_CACHE_FILE: &str = "/tmp/qwen2_kvcache.bin";

pub fn main(args: Vec<String>) {

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

    let artifact = Artifact::from_bytes(QWEN2_ON_DISK_BINARY).expect("the build checked this artifact");
    let base = Driver::load(artifact).expect("Driver::load");

    eprintln!("Starting qwen2_on_disk (weights={weights_path}, tokenizer={tokenizer_path})");
    unsafe { base.execute("main", &data, &mut []) }
        .expect("qwen2_on_disk run");
}
