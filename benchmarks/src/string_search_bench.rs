use base::{BaseConfig, Algorithm};
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::harness::{self, BenchResult};

const STRSEARCH_ALGORITHM: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/strsearch_algorithm.bin"));

fn load_algorithm() -> (BaseConfig, Algorithm) {
    bincode::deserialize(STRSEARCH_ALGORITHM).expect("Failed to deserialize strsearch algorithm")
}

const VOCABULARY: &[&str] = &[
    "the", "of", "and", "to", "in", "a", "is", "that",
    "for", "it", "was", "on", "are", "as", "with", "his",
    "they", "at", "be", "this", "from", "or", "had", "by",
    "not", "but", "some", "what", "we", "can", "out", "all",
    "your", "when", "up", "use", "how", "said", "an", "each",
];

const PATTERN: &str = "that";

fn generate_text(path: &str, num_words: usize) -> usize {
    let dir = Path::new(path).parent().unwrap();
    fs::create_dir_all(dir).ok();

    let mut f = std::io::BufWriter::new(fs::File::create(path).unwrap());
    let mut text = String::new();

    for i in 0..num_words {
        let idx = ((i * 7 + 13) * 31) % VOCABULARY.len();
        let word = VOCABULARY[idx];
        if i > 0 {
            text.push(' ');
        }
        text.push_str(word);
    }
    writeln!(f, "{}", text).unwrap();

    // Count overlapping occurrences of PATTERN
    let mut count = 0;
    let mut start = 0;
    while let Some(pos) = text[start..].find(PATTERN) {
        count += 1;
        start += pos + 1;
    }
    count
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

fn rust_string_search(path: &str) -> usize {
    let data = fs::read_to_string(path).unwrap();
    let pattern = PATTERN.as_bytes();
    let data = data.as_bytes();
    let mut count = 0;
    let mut pos = 0;

    while pos + pattern.len() <= data.len() {
        if &data[pos..pos + pattern.len()] == pattern {
            count += 1;
        }
        pos += 1;
    }
    count
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
    let sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000];
    let mut results = Vec::new();

    // JIT compile once
    let (config, alg) = load_algorithm();
    let mut base_instance = base::Base::new(config).expect("Base::new failed");

    for &n in &sizes {
        let text_path = format!("/tmp/bench-data/strsearch_{}.txt", n);
        let output_path = format!("/tmp/bench-data/strsearch_result_{}.txt", n);

        let expected = generate_text(&text_path, n);
        let payload = build_payload(&text_path, &output_path);

        // Python
        let python_ms = harness::median_of(iterations, || {
            match harness::run_python("string_search.py", &[&text_path, PATTERN]) {
                Some((ms, stdout)) => {
                    if let Ok(count) = stdout.parse::<usize>() {
                        if count != expected {
                            eprintln!(
                                "WARNING: Python search count {} != expected {} (n={})",
                                count, expected, n
                            );
                        }
                    }
                    ms
                }
                None => f64::NAN,
            }
        });

        // Pure Rust
        let rust_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            let count = rust_string_search(&text_path);
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            if count != expected {
                eprintln!(
                    "WARNING: Rust search count {} != expected {} (n={})",
                    count, expected, n
                );
            }
            ms
        });

        // Base (Cranelift JIT) — execute with payload, verify output file
        // Warmup
        let _ = fs::remove_file(&output_path);
        let _ = base_instance.execute(&alg, &payload);

        let base_ms = harness::median_of(iterations, || {
            let _ = fs::remove_file(&output_path);
            let start = std::time::Instant::now();
            let _ = base_instance.execute(&alg, &payload);
            start.elapsed().as_secs_f64() * 1000.0
        });

        // Verify result by reading the output file
        let verified = if let Ok(content) = fs::read_to_string(&output_path) {
            if let Ok(count) = content.trim().parse::<usize>() {
                if count != expected {
                    eprintln!(
                        "WARNING: Base strsearch count {} != expected {} (n={})",
                        count, expected, n
                    );
                }
                Some(count == expected)
            } else {
                eprintln!("WARNING: Could not parse base strsearch output: {:?}", content.trim());
                Some(false)
            }
        } else {
            eprintln!("WARNING: Could not read base output file {:?}", output_path);
            Some(false)
        };

        results.push(BenchResult {
            name: format!("StrSearch ({})", format_count(n)),
            python_ms: if python_ms.is_nan() {
                None
            } else {
                Some(python_ms)
            },
            rust_ms: Some(rust_ms),
            base_ms,
            verified,
        });
    }

    results
}
