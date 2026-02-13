use base::Algorithm;
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::harness::{self, BenchResult};

const CSV_ALGORITHM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/csv_algorithm.bin"));

// Memory layout offsets (must match CsvBenchAlgorithm.lean)
const INPUT_FILENAME: usize = 0x0020;
const OUTPUT_FILENAME: usize = 0x0120;
const END_POS: usize = 0x02F0;
const CSV_DATA: usize = 0x2000;

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
            i, i, i, i,
            i % 10,
            salary
        )
        .unwrap();
    }
    total
}

fn prepare_algorithm(csv_path: &str, output_path: &str, file_size: usize) -> Algorithm {
    let mut alg: Algorithm =
        bincode::deserialize(CSV_ALGORITHM).expect("Failed to deserialize algorithm");

    // Ensure payload can hold the CSV data that FileRead will write at CSV_DATA
    let needed = CSV_DATA + file_size + 256;
    if alg.payloads.len() < needed {
        alg.payloads.resize(needed, 0);
    }

    // Patch input filename (null-terminated)
    let inp = csv_path.as_bytes();
    alg.payloads[INPUT_FILENAME..INPUT_FILENAME + inp.len()].copy_from_slice(inp);
    alg.payloads[INPUT_FILENAME + inp.len()] = 0;

    // Patch output filename (null-terminated)
    let out = output_path.as_bytes();
    alg.payloads[OUTPUT_FILENAME..OUTPUT_FILENAME + out.len()].copy_from_slice(out);
    alg.payloads[OUTPUT_FILENAME + out.len()] = 0;

    // Patch END_POS (i32 little-endian = CSV file byte count)
    let end = file_size as i32;
    alg.payloads[END_POS..END_POS + 4].copy_from_slice(&end.to_le_bytes());

    alg
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
    let sizes = [10_000, 100_000, 500_000, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000];
    let mut results = Vec::new();

    for &n in &sizes {
        let csv_path = format!("/tmp/bench-data/employees_{}.csv", n);
        let output_path = format!("/tmp/bench-data/base_result_{}.txt", n);

        let expected = generate_csv(&csv_path, n);
        let file_size = fs::metadata(&csv_path).unwrap().len() as usize;

        // Python
        let python_ms = harness::median_of(iterations, || {
            match harness::run_python("csv_sum.py", &[&csv_path]) {
                Some((ms, stdout)) => {
                    if let Ok(sum) = stdout.parse::<i64>() {
                        if sum != expected {
                            eprintln!(
                                "WARNING: Python sum {} != expected {}",
                                sum, expected
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
            let sum = rust_csv_sum(&csv_path);
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            if sum != expected {
                eprintln!("WARNING: Rust sum {} != expected {} (n={})", sum, expected, n);
            }
            ms
        });

        // Base
        let mut verified = None;
        let base_ms = harness::median_of(iterations, || {
            let _ = fs::remove_file(&output_path);

            let alg = prepare_algorithm(&csv_path, &output_path, file_size);
            let ms = harness::run_base(alg);

            // Verify result by reading the output file
            if let Ok(content) = fs::read_to_string(&output_path) {
                if let Ok(sum) = content.trim().parse::<i64>() {
                    verified = Some(sum == expected);
                    if sum != expected {
                        eprintln!("WARNING: Base (CL) sum {} != expected {} (n={})", sum, expected, n);
                    }
                } else {
                    eprintln!(
                        "WARNING: Base (CL) output not a valid integer: {:?}",
                        content.trim()
                    );
                    verified = Some(false);
                }
            } else {
                eprintln!("WARNING: Could not read base output file {:?}", output_path);
                verified = Some(false);
            }

            ms
        });

        results.push(BenchResult {
            name: format!("CSV ({})", format_count(n)),
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
