use base::{BaseConfig, Algorithm};
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::harness::{self, BenchResult, format_count};

const JSON_ALGORITHM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/json_algorithm.bin"));

fn load_algorithm() -> (BaseConfig, Algorithm) {
    bincode::deserialize(JSON_ALGORITHM).expect("Failed to deserialize json algorithm")
}

fn generate_json(path: &str, n: usize) -> i64 {
    let dir = Path::new(path).parent().unwrap();
    fs::create_dir_all(dir).ok();

    let mut f = std::io::BufWriter::new(fs::File::create(path).unwrap());
    let mut total: i64 = 0;

    write!(f, "[\n").unwrap();
    for i in 0..n {
        let value = ((i as i64).wrapping_mul(137).wrapping_add(42)) % 10000;
        total += value;
        if i > 0 {
            write!(f, ",\n").unwrap();
        }
        write!(
            f,
            "  {{\"id\": {}, \"name\": \"item_{}\", \"value\": {}}}",
            i, i, value
        )
        .unwrap();
    }
    write!(f, "\n]\n").unwrap();
    total
}

/// Build payload: "input_path\0output_path\0"
fn build_payload(json_path: &str, output_path: &str) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(json_path.as_bytes());
    payload.push(0);
    payload.extend_from_slice(output_path.as_bytes());
    payload.push(0);
    payload
}

/// Streaming JSON parser: extracts "value" fields without building a tree.
/// Scans for the pattern `"value": <digits>` and sums the numbers.
fn rust_json_sum(path: &str) -> i64 {
    let data = fs::read(path).unwrap();
    let needle = b"\"value\": ";
    let mut total: i64 = 0;
    let mut pos = 0;

    while pos + needle.len() < data.len() {
        if &data[pos..pos + needle.len()] == needle {
            pos += needle.len();
            let mut acc: i64 = 0;
            while pos < data.len() && data[pos].is_ascii_digit() {
                acc = acc * 10 + (data[pos] - b'0') as i64;
                pos += 1;
            }
            total += acc;
        } else {
            pos += 1;
        }
    }
    total
}

pub fn run(iterations: usize) -> Vec<BenchResult> {
    let sizes = [1_000, 10_000, 50_000, 100_000, 500_000];
    let mut results = Vec::new();

    // JIT compile once
    let (config, alg) = load_algorithm();
    let mut base_instance = base::Base::new(config).expect("Base::new failed");

    for &n in &sizes {
        let json_path = format!("/tmp/bench-data/data_{}.json", n);
        let output_path = format!("/tmp/bench-data/json_result_{}.txt", n);

        let expected = generate_json(&json_path, n);
        let payload = build_payload(&json_path, &output_path);

        // Python
        let python_ms = harness::median_of(iterations, || {
            match harness::run_python("json_bench.py", &[&json_path]) {
                Some((ms, stdout)) => {
                    if let Ok(sum) = stdout.parse::<i64>() {
                        if sum != expected {
                            eprintln!(
                                "WARNING: Python JSON sum {} != expected {} (n={})",
                                sum, expected, n
                            );
                        }
                    }
                    ms
                }
                None => f64::NAN,
            }
        });

        // Pure Rust (streaming parser, no serde)
        let rust_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            let sum = rust_json_sum(&json_path);
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            if sum != expected {
                eprintln!(
                    "WARNING: Rust JSON sum {} != expected {} (n={})",
                    sum, expected, n
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
            if let Ok(sum) = content.trim().parse::<i64>() {
                if sum != expected {
                    eprintln!("WARNING: Base JSON sum {} != expected {} (n={})", sum, expected, n);
                }
                Some(sum == expected)
            } else {
                eprintln!("WARNING: Could not parse base JSON output: {:?}", content.trim());
                Some(false)
            }
        } else {
            eprintln!("WARNING: Could not read base output file {:?}", output_path);
            Some(false)
        };

        results.push(BenchResult {
            name: format!("JSON ({})", format_count(n)),
            col_a_ms: if python_ms.is_nan() {
                None
            } else {
                Some(python_ms)
            },
            col_b_ms: Some(rust_ms),
            base_ms,
            verified,
        });
    }

    results
}
