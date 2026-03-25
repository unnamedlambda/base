use base::{BaseConfig, Algorithm};
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::harness::{self, BenchResult};

const REGEX_ALGORITHM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/regex_algorithm.bin"));

fn load_algorithm() -> (BaseConfig, Algorithm) {
    bincode::deserialize(REGEX_ALGORITHM).expect("Failed to deserialize regex algorithm")
}

const VOCABULARY: &[&str] = &[
    "the", "running", "of", "singing", "and", "to", "jumping", "in",
    "a", "is", "that", "finding", "for", "it", "was", "making",
    "on", "are", "as", "with", "building", "they", "at", "be",
    "this", "from", "or", "testing", "had", "by", "not", "coding",
    "but", "some", "what", "we", "writing", "can", "out", "reading",
];

fn generate_text(path: &str, num_words: usize) -> usize {
    let dir = Path::new(path).parent().unwrap();
    fs::create_dir_all(dir).ok();

    let mut f = std::io::BufWriter::new(fs::File::create(path).unwrap());
    let mut expected = 0;

    for i in 0..num_words {
        let idx = ((i * 7 + 13) * 31) % VOCABULARY.len();
        let word = VOCABULARY[idx];
        if word.ends_with("ing") {
            expected += 1;
        }
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

fn rust_regex_count(path: &str) -> usize {
    let data = fs::read_to_string(path).unwrap();
    let mut count = 0;
    for word in data.split_whitespace() {
        if word.len() > 3
            && word.ends_with("ing")
            && word.bytes().all(|b| b.is_ascii_lowercase())
        {
            count += 1;
        }
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
        let text_path = format!("/tmp/bench-data/regex_{}.txt", n);
        let output_path = format!("/tmp/bench-data/regex_result_{}.txt", n);

        let expected = generate_text(&text_path, n);
        let payload = build_payload(&text_path, &output_path);

        // Python
        let python_ms = harness::median_of(iterations, || {
            match harness::run_python("regex_bench.py", &[&text_path]) {
                Some((ms, stdout)) => {
                    if let Ok(count) = stdout.parse::<usize>() {
                        if count != expected {
                            eprintln!(
                                "WARNING: Python regex count {} != expected {} (n={})",
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
            let count = rust_regex_count(&text_path);
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            if count != expected {
                eprintln!(
                    "WARNING: Rust regex count {} != expected {} (n={})",
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
                        "WARNING: Base regex count {} != expected {} (n={})",
                        count, expected, n
                    );
                }
                Some(count == expected)
            } else {
                eprintln!("WARNING: Could not parse base regex output: {:?}", content.trim());
                Some(false)
            }
        } else {
            eprintln!("WARNING: Could not read base output file {:?}", output_path);
            Some(false)
        };

        results.push(BenchResult {
            name: format!("Regex ({})", format_count(n)),
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
