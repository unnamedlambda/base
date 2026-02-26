use base::{BaseConfig, Algorithm};
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::harness::{self, BenchResult};

const STRSEARCH_ALGORITHM: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/strsearch_algorithm.bin"));

// Memory layout offsets (must match StringSearchAlgorithm.lean)
const INPUT_FILENAME: usize = 0x0020;
const OUTPUT_FILENAME: usize = 0x0120;
const FILE_SIZE: usize = 0x0250;
const INPUT_DATA: usize = 0x4000;

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

fn rust_string_search(path: &str) -> usize {
    let data = fs::read_to_string(path).unwrap();
    let pattern = PATTERN.as_bytes();
    let data = data.as_bytes();
    let mut count = 0;
    let mut pos = 0;

    while pos + pattern.len() <= data.len() {
        if &data[pos..pos + pattern.len()] == pattern {
            count += 1;
            pos += 1;
        } else {
            pos += 1;
        }
    }
    count
}

fn prepare_algorithm(text_path: &str, output_path: &str, file_size: usize) -> (BaseConfig, Algorithm) {
    let (mut config, mut alg): (BaseConfig, Algorithm) =
        bincode::deserialize(STRSEARCH_ALGORITHM).expect("Failed to deserialize strsearch algorithm");

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
        let text_path = format!("/tmp/bench-data/strsearch_{}.txt", n);
        let output_path = format!("/tmp/bench-data/strsearch_result_{}.txt", n);

        let expected = generate_text(&text_path, n);
        let file_size = fs::metadata(&text_path).unwrap().len() as usize;

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
                            "WARNING: Base strsearch count {} != expected {} (n={})",
                            count, expected, n
                        );
                        verified = Some(false);
                    }
                } else {
                    eprintln!("WARNING: Could not parse base strsearch output: {:?}", content.trim());
                    verified = Some(false);
                }
            } else {
                eprintln!("WARNING: Could not read base output file {:?}", output_path);
                verified = Some(false);
            }

            ms
        });

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
            actions: None,
        });
    }

    results
}
