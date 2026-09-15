use base::Artifact;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::harness::{self, format_count, BenchResult};

const WC_ARTIFACT: &[u8] =
    build_support::artifact!("RustBenchmarks/wc_algorithm");

/// Entry points of this artifact, as its generator numbers them.
const MAIN: u32 = 1;

/// `INPUT_DATA` (0x14000) plus room for the largest input, which is under 4 MB.
/// The generator reserves 512 MiB; a fresh instance a round makes that
/// reservation, not the counting, the thing being measured.
const WC_ARENA_BYTES: usize = 0x14000 + 16 * 1024 * 1024;

const VOCABULARY: &[&str] = &[
    "the", "of", "and", "to", "in", "a", "is", "that", "for", "it", "was", "on", "are", "as",
    "with", "his", "they", "at", "be", "this", "from", "or", "had", "by", "not", "but", "some",
    "what", "we", "can", "out", "all", "your", "when", "up", "use", "how", "said", "an", "each",
];

fn generate_text(path: &str, num_words: usize) -> HashMap<Vec<u8>, u64> {
    let dir = Path::new(path).parent().unwrap();
    fs::create_dir_all(dir).ok();

    let mut f = std::io::BufWriter::new(fs::File::create(path).unwrap());
    let mut expected = HashMap::new();

    for i in 0..num_words {
        let idx = ((i * 7 + 13) * 31) % VOCABULARY.len();
        let word = VOCABULARY[idx];
        *expected.entry(word.as_bytes().to_vec()).or_insert(0u64) += 1;
        if i > 0 {
            write!(f, " ").unwrap();
        }
        write!(f, "{}", word).unwrap();
    }
    writeln!(f).unwrap();
    expected
}

/// Build payload: "input_path\0output_path\0"
fn build_payload(text_path: &str, output_path: &str) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(text_path.as_bytes());
    payload.push(0);
    payload.extend_from_slice(output_path.as_bytes());
    payload.push(0);
    payload
}

fn rust_wordcount(path: &str, output_path: &str, buf: &mut Vec<u8>) -> HashMap<Vec<u8>, u64> {
    // Bytes rather than `read_to_string`, into a buffer that outlives the call:
    // the generated program neither validates UTF-8 nor reallocates its arena.
    harness::read_into(path, buf).unwrap();
    let mut counts: HashMap<Vec<u8>, u64> = HashMap::new();
    for word in buf.split(|b| b.is_ascii_whitespace()) {
        if word.is_empty() {
            continue;
        }
        // `cl_ht_increment` allocates only when the key is new; `entry(to_string())`
        // would allocate once per occurrence, which is a different amount of work.
        if let Some(c) = counts.get_mut(word) {
            *c += 1;
        } else {
            counts.insert(word.to_vec(), 1);
        }
    }
    // The artifact renders the whole table and writes it durably. Do the same.
    let mut out = Vec::with_capacity(counts.len() * 16);
    for (w, c) in counts.iter() {
        out.extend_from_slice(w);
        out.push(b'\t');
        out.extend_from_slice(c.to_string().as_bytes());
        out.push(b'\n');
    }
    let mut f = fs::File::create(output_path).unwrap();
    f.write_all(&out).unwrap();
    f.sync_all().unwrap();
    counts
}

fn parse_output(content: &str) -> HashMap<Vec<u8>, u64> {
    let mut result = HashMap::new();
    for line in content.lines() {
        if let Some((word, count_str)) = line.split_once('\t') {
            if let Ok(count) = count_str.parse::<u64>() {
                result.insert(word.as_bytes().to_vec(), count);
            }
        }
    }
    result
}

pub fn run(iterations: usize) -> Vec<BenchResult> {
    let sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000];
    let mut results = Vec::new();

    for &n in &sizes {
        let text_path = format!("/tmp/bench-data/words_{}.txt", n);
        let output_path = format!("/tmp/bench-data/wc_result_{}.txt", n);

        let expected = generate_text(&text_path, n);
        let payload = build_payload(&text_path, &output_path);

        // Pure Rust
        let rust_out = format!("/tmp/bench-data/rust_result_wc_{}.txt", n);
        let mut rust_buf: Vec<u8> = Vec::new();
        let _ = rust_wordcount(&text_path, &rust_out, &mut rust_buf);

        let rust_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            let got = rust_wordcount(&text_path, &rust_out, &mut rust_buf);
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            if got != expected {
                eprintln!("WARNING: Rust counts mismatch (n={})", n);
            }
            ms
        });

        // Base (Cranelift JIT) — fresh instance per execution because HT state
        // accumulates across execute() calls (ht_increment on handle 0 persists).
        //
        // The warmup removes the output file and writes it back, so every timed
        // iteration overwrites a path that exists — the filesystem work the
        // baseline does. Removing it per iteration would charge Base for an
        // inode allocation a round that the baseline never pays.
        let _ = fs::remove_file(&output_path);
        {
            let mut artifact = Artifact::from_bytes(WC_ARTIFACT);
            artifact.memory_size = WC_ARENA_BYTES;
            let mut base_instance = base::Base::new(artifact).expect("Base::new failed");
            let _ = base_instance.execute(MAIN, &payload);
        }

        let base_ms = harness::median_of(iterations, || {
            let mut artifact = Artifact::from_bytes(WC_ARTIFACT);
            artifact.memory_size = WC_ARENA_BYTES;
            let mut base_instance = base::Base::new(artifact).expect("Base::new failed");
            let start = std::time::Instant::now();
            let _ = base_instance.execute(MAIN, &payload);
            start.elapsed().as_secs_f64() * 1000.0
        });

        // Run one more time with fresh instance for verification
        let _ = fs::remove_file(&output_path);
        let mut artifact = Artifact::from_bytes(WC_ARTIFACT);
        artifact.memory_size = WC_ARENA_BYTES;
        let mut base_instance = base::Base::new(artifact).expect("Base::new failed");
        let _ = base_instance.execute(MAIN, &payload);

        let verified = if let Ok(content) = fs::read_to_string(&output_path) {
            let got = parse_output(content.trim());
            if got == expected {
                Some(true)
            } else {
                eprintln!(
                    "WARNING: Base counts mismatch (n={}): got {} unique, expected {} unique",
                    n,
                    got.len(),
                    expected.len()
                );
                for (word, exp_count) in &expected {
                    if got.get(word) != Some(exp_count) {
                        eprintln!(
                            "  word {:?}: expected {}, got {:?}",
                            word,
                            exp_count,
                            got.get(word)
                        );
                    }
                }
                Some(false)
            }
        } else {
            eprintln!("WARNING: Could not read base output file {:?}", output_path);
            Some(false)
        };

        results.push(BenchResult {
            name: format!("WC ({})", format_count(n)),
            col_a_ms: Some(rust_ms),
            col_b_ms: None,
            base_ms,
            verified,
        });
    }

    results
}
