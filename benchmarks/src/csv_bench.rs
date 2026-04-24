use base::{Algorithm, BaseConfig};
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::harness::{self, format_count, BenchResult};

const CSV_ALGORITHM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/csv_algorithm.bin"));

fn load_algorithm() -> (BaseConfig, Algorithm) {
    bincode::deserialize(CSV_ALGORITHM).expect("Failed to deserialize csv algorithm")
}

/// Generate a deterministic CSV file with a salary column.
/// Salaries are in range 1000–9999 to keep the i32 total within range.
/// Returns the expected salary sum.
fn generate_csv(path: &str, num_rows: usize) -> i64 {
    let dir = Path::new(path).parent().unwrap();
    fs::create_dir_all(dir).ok();

    let mut f = std::io::BufWriter::new(fs::File::create(path).unwrap());
    writeln!(f, "id,first_name,last_name,email,department,salary").unwrap();

    let mut total: i64 = 0;
    for i in 0..num_rows {
        let salary = 1000 + ((i as i64 * 137) % 9000);
        total += salary;
        writeln!(
            f,
            "{},First{},Last{},e{}@co.com,Dept{},{}",
            i,
            i,
            i,
            i,
            i % 10,
            salary
        )
        .unwrap();
    }
    total
}

/// Build payload: "input_path\0output_path\0"
fn build_payload(csv_path: &str, output_path: &str) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(csv_path.as_bytes());
    payload.push(0);
    payload.extend_from_slice(output_path.as_bytes());
    payload.push(0);
    payload
}

/// Pure Rust CSV salary sum — same algorithm as the CLIF IR, for comparison.
fn rust_csv_sum(path: &str) -> i64 {
    let data = fs::read(path).unwrap();
    let mut pos = 0;
    // Skip header line
    while pos < data.len() && data[pos] != b'\n' {
        pos += 1;
    }
    pos += 1;
    let mut total: i64 = 0;
    while pos < data.len() {
        // Skip 5 commas to reach salary column
        let mut commas = 0;
        while commas < 5 {
            if data[pos] == b',' {
                commas += 1;
            }
            pos += 1;
        }
        // Parse salary digits
        let mut acc: i64 = 0;
        while pos < data.len() && data[pos] != b'\n' {
            acc = acc * 10 + (data[pos] - b'0') as i64;
            pos += 1;
        }
        total += acc;
        if pos < data.len() {
            pos += 1; // skip newline
        }
    }
    total
}

pub fn run(iterations: usize) -> Vec<BenchResult> {
    let sizes = [
        10_000, 100_000, 500_000, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000,
    ];
    let mut results = Vec::new();

    // JIT compile once
    let (config, alg) = load_algorithm();
    let mut base_instance = base::Base::new(config).expect("Base::new failed");

    for &n in &sizes {
        let csv_path = format!("/tmp/bench-data/employees_{}.csv", n);
        let output_path = format!("/tmp/bench-data/base_result_{}.txt", n);

        let expected = generate_csv(&csv_path, n);
        let payload = build_payload(&csv_path, &output_path);

        // Python
        let python_ms = harness::median_of(iterations, || {
            match harness::run_python("csv_sum.py", &[&csv_path]) {
                Some((ms, stdout)) => {
                    if let Ok(sum) = stdout.parse::<i64>() {
                        if sum != expected {
                            eprintln!("WARNING: Python sum {} != expected {}", sum, expected);
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
            let sum = rust_csv_sum(&csv_path);
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            if sum != expected {
                eprintln!(
                    "WARNING: Rust sum {} != expected {} (n={})",
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
                    eprintln!(
                        "WARNING: Base (CL) sum {} != expected {} (n={})",
                        sum, expected, n
                    );
                }
                Some(sum == expected)
            } else {
                eprintln!(
                    "WARNING: Base (CL) output not a valid integer: {:?}",
                    content.trim()
                );
                Some(false)
            }
        } else {
            eprintln!("WARNING: Could not read base output file {:?}", output_path);
            Some(false)
        };

        results.push(BenchResult {
            name: format!("CSV ({})", format_count(n)),
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
