use base::{BaseConfig, Algorithm};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::harness::{self, BenchResult};

const WC_ALGORITHM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/wc_algorithm.bin"));

fn load_algorithm() -> (BaseConfig, Algorithm) {
    bincode::deserialize(WC_ALGORITHM).expect("Failed to deserialize wc algorithm")
}

const VOCABULARY: &[&str] = &[
    "the", "of", "and", "to", "in", "a", "is", "that",
    "for", "it", "was", "on", "are", "as", "with", "his",
    "they", "at", "be", "this", "from", "or", "had", "by",
    "not", "but", "some", "what", "we", "can", "out", "all",
    "your", "when", "up", "use", "how", "said", "an", "each",
];

fn generate_text(path: &str, num_words: usize) -> HashMap<String, u64> {
    let dir = Path::new(path).parent().unwrap();
    fs::create_dir_all(dir).ok();

    let mut f = std::io::BufWriter::new(fs::File::create(path).unwrap());
    let mut expected = HashMap::new();

    for i in 0..num_words {
        let idx = ((i * 7 + 13) * 31) % VOCABULARY.len();
        let word = VOCABULARY[idx];
        *expected.entry(word.to_string()).or_insert(0u64) += 1;
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

fn rust_wordcount(path: &str) -> HashMap<String, u64> {
    let data = fs::read_to_string(path).unwrap();
    let mut counts = HashMap::new();
    for word in data.split_whitespace() {
        *counts.entry(word.to_string()).or_insert(0u64) += 1;
    }
    counts
}

fn parse_output(content: &str) -> HashMap<String, u64> {
    let mut result = HashMap::new();
    for line in content.lines() {
        if let Some((word, count_str)) = line.split_once('\t') {
            if let Ok(count) = count_str.parse::<u64>() {
                result.insert(word.to_string(), count);
            }
        }
    }
    result
}

fn format_count(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        format!("{}", n)
    }
}

pub fn run(iterations: usize) -> Vec<BenchResult> {
    let sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000];
    let mut results = Vec::new();

    for &n in &sizes {
        let text_path = format!("/tmp/bench-data/words_{}.txt", n);
        let output_path = format!("/tmp/bench-data/wc_result_{}.txt", n);

        let expected = generate_text(&text_path, n);
        let payload = build_payload(&text_path, &output_path);

        // Python
        let python_ms = harness::median_of(iterations, || {
            match harness::run_python("wordcount.py", &[&text_path]) {
                Some((ms, stdout)) => {
                    let got = parse_output(&stdout);
                    if got != expected {
                        eprintln!("WARNING: Python counts mismatch (n={})", n);
                    }
                    ms
                }
                None => f64::NAN,
            }
        });

        // Pure Rust
        let rust_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            let got = rust_wordcount(&text_path);
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            if got != expected {
                eprintln!("WARNING: Rust counts mismatch (n={})", n);
            }
            ms
        });

        // Base (Cranelift JIT) — fresh instance per execution because HT state
        // accumulates across execute() calls (ht_increment on handle 0 persists).
        let base_ms = harness::median_of(iterations, || {
            let _ = fs::remove_file(&output_path);
            let (config, alg) = load_algorithm();
            let mut base_instance = base::Base::new(config).expect("Base::new failed");
            let start = std::time::Instant::now();
            let _ = base_instance.execute(&alg, &payload);
            start.elapsed().as_secs_f64() * 1000.0
        });

        // Run one more time with fresh instance for verification
        let _ = fs::remove_file(&output_path);
        let (config, alg) = load_algorithm();
        let mut base_instance = base::Base::new(config).expect("Base::new failed");
        let _ = base_instance.execute(&alg, &payload);

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
            python_ms: if python_ms.is_nan() {
                None
            } else {
                Some(python_ms)
            },
            rust_ms: Some(rust_ms),
            base_ms,
            verified,
            actions: None,
        });
    }

    results
}
