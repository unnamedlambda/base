use std::fs;
use std::path::Path;

use crate::harness::{self, BenchResult};

fn generate_data(path: &str, n: usize) -> (i32, i32) {
    let dir = Path::new(path).parent().unwrap();
    fs::create_dir_all(dir).ok();

    let mut values: Vec<i32> = Vec::with_capacity(n);
    for i in 0..n {
        // Deterministic pseudo-random i32 values
        let v = ((i as i64).wrapping_mul(1_103_515_245).wrapping_add(12_345) % 2_000_000) as i32
            - 1_000_000;
        values.push(v);
    }

    // Write as raw little-endian i32
    let mut bytes = Vec::with_capacity(n * 4);
    for &v in &values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, &bytes).unwrap();

    values.sort();
    (values[0], values[n - 1])
}

fn rust_sort(path: &str) -> (i32, i32) {
    let data = fs::read(path).unwrap();
    let n = data.len() / 4;
    let mut values: Vec<i32> = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * 4;
        let v = i32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        values.push(v);
    }
    values.sort_unstable();
    (values[0], values[n - 1])
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
    let sizes = [10_000, 100_000, 500_000, 1_000_000, 5_000_000];
    let mut results = Vec::new();

    for &n in &sizes {
        let data_path = format!("/tmp/bench-data/sort_{}.bin", n);

        let (expected_first, expected_last) = generate_data(&data_path, n);

        // Python
        let python_ms = harness::median_of(iterations, || {
            match harness::run_python("sort_bench.py", &[&data_path]) {
                Some((ms, stdout)) => {
                    if let Some((first_s, last_s)) = stdout.split_once(',') {
                        let first: i32 = first_s.parse().unwrap_or(0);
                        let last: i32 = last_s.parse().unwrap_or(0);
                        if first != expected_first || last != expected_last {
                            eprintln!(
                                "WARNING: Python sort mismatch (n={}): got ({},{}), expected ({},{})",
                                n, first, last, expected_first, expected_last
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
            let (first, last) = rust_sort(&data_path);
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            if first != expected_first || last != expected_last {
                eprintln!(
                    "WARNING: Rust sort mismatch (n={}): got ({},{}), expected ({},{})",
                    n, first, last, expected_first, expected_last
                );
            }
            ms
        });

        results.push(BenchResult {
            name: format!("Sort ({})", format_count(n)),
            python_ms: if python_ms.is_nan() {
                None
            } else {
                Some(python_ms)
            },
            rust_ms: Some(rust_ms),
            base_ms: f64::NAN,
            verified: None,
            actions: None,
        });
    }

    results
}
