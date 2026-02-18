use base::Algorithm;
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::harness::{self, BenchResult};

const JSON_ALGORITHM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/json_algorithm.bin"));

// Memory layout offsets (must match JsonBenchAlgorithm.lean)
const INPUT_FILENAME: usize = 0x0020;
const OUTPUT_FILENAME: usize = 0x0120;
const FILE_SIZE: usize = 0x0250;
const INPUT_DATA: usize = 0x4000;

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

fn prepare_algorithm(json_path: &str, output_path: &str, file_size: usize) -> Algorithm {
    let mut alg: Algorithm =
        bincode::deserialize(JSON_ALGORITHM).expect("Failed to deserialize json algorithm");

    let needed = INPUT_DATA + file_size + 256;
    if alg.payloads.len() < needed {
        alg.payloads.resize(needed, 0);
    }

    // Patch input filename
    let inp = json_path.as_bytes();
    alg.payloads[INPUT_FILENAME..INPUT_FILENAME + inp.len()].copy_from_slice(inp);
    alg.payloads[INPUT_FILENAME + inp.len()] = 0;

    // Patch output filename
    let out = output_path.as_bytes();
    alg.payloads[OUTPUT_FILENAME..OUTPUT_FILENAME + out.len()].copy_from_slice(out);
    alg.payloads[OUTPUT_FILENAME + out.len()] = 0;

    // Patch FILE_SIZE
    let end = file_size as i32;
    alg.payloads[FILE_SIZE..FILE_SIZE + 4].copy_from_slice(&end.to_le_bytes());

    alg
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
    let sizes = [1_000, 10_000, 50_000, 100_000, 500_000];
    let mut results = Vec::new();

    for &n in &sizes {
        let json_path = format!("/tmp/bench-data/data_{}.json", n);
        let output_path = format!("/tmp/bench-data/json_result_{}.txt", n);

        let expected = generate_json(&json_path, n);
        let file_size = fs::metadata(&json_path).unwrap().len() as usize;

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

        // Base
        let mut verified = None;
        let base_ms = harness::median_of(iterations, || {
            let _ = fs::remove_file(&output_path);

            let alg = prepare_algorithm(&json_path, &output_path, file_size);
            let ms = harness::run_base(alg);

            if let Ok(content) = fs::read_to_string(&output_path) {
                if let Ok(sum) = content.trim().parse::<i64>() {
                    if sum == expected {
                        verified = Some(true);
                    } else {
                        eprintln!(
                            "WARNING: Base JSON sum {} != expected {} (n={})",
                            sum, expected, n
                        );
                        verified = Some(false);
                    }
                } else {
                    eprintln!("WARNING: Could not parse base JSON output: {:?}", content.trim());
                    verified = Some(false);
                }
            } else {
                eprintln!("WARNING: Could not read base output file {:?}", output_path);
                verified = Some(false);
            }

            ms
        });

        results.push(BenchResult {
            name: format!("JSON ({})", format_count(n)),
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
