use base::{BaseConfig, Algorithm};
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::harness::{self, BenchResult};

const REGEX_ALGORITHM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/regex_algorithm.bin"));

// Memory layout offsets (must match RegexBenchAlgorithm.lean)
const INPUT_FILENAME: usize = 0x0020;
const OUTPUT_FILENAME: usize = 0x0120;
const FILE_SIZE: usize = 0x0250;
const INPUT_DATA: usize = 0x4000;

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

fn prepare_algorithm(text_path: &str, output_path: &str, file_size: usize) -> (BaseConfig, Algorithm) {
    let (mut config, mut alg): (BaseConfig, Algorithm) =
        bincode::deserialize(REGEX_ALGORITHM).expect("Failed to deserialize regex algorithm");

    let needed = INPUT_DATA + file_size + 256;
    if alg.payloads.len() < needed {
        alg.payloads.resize(needed, 0);
    }
    config.memory_size = config.memory_size.max(alg.payloads.len());

    // Patch input filename
    let inp = text_path.as_bytes();
    alg.payloads[INPUT_FILENAME..INPUT_FILENAME + inp.len()].copy_from_slice(inp);
    alg.payloads[INPUT_FILENAME + inp.len()] = 0;

    // Patch output filename
    let out = output_path.as_bytes();
    alg.payloads[OUTPUT_FILENAME..OUTPUT_FILENAME + out.len()].copy_from_slice(out);
    alg.payloads[OUTPUT_FILENAME + out.len()] = 0;

    // Patch FILE_SIZE
    let end = file_size as i32;
    alg.payloads[FILE_SIZE..FILE_SIZE + 4].copy_from_slice(&end.to_le_bytes());

    (config, alg)
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

    for &n in &sizes {
        let text_path = format!("/tmp/bench-data/regex_{}.txt", n);
        let output_path = format!("/tmp/bench-data/regex_result_{}.txt", n);

        let expected = generate_text(&text_path, n);
        let file_size = fs::metadata(&text_path).unwrap().len() as usize;

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

        // Base
        let mut verified = None;
        let base_ms = harness::median_of(iterations, || {
            let _ = fs::remove_file(&output_path);

            let (cfg, alg) = prepare_algorithm(&text_path, &output_path, file_size);
            let ms = harness::run_base(cfg, alg);

            if let Ok(content) = fs::read_to_string(&output_path) {
                if let Ok(count) = content.trim().parse::<usize>() {
                    if count == expected {
                        verified = Some(true);
                    } else {
                        eprintln!(
                            "WARNING: Base regex count {} != expected {} (n={})",
                            count, expected, n
                        );
                        verified = Some(false);
                    }
                } else {
                    eprintln!("WARNING: Could not parse base regex output: {:?}", content.trim());
                    verified = Some(false);
                }
            } else {
                eprintln!("WARNING: Could not read base output file {:?}", output_path);
                verified = Some(false);
            }

            ms
        });

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
            actions: None,
        });
    }

    results
}
