use base::Algorithm;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::harness::{self, BenchResult};

const WC_ALGORITHM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/wc_algorithm.bin"));

// Memory layout offsets (must match WordCountAlgorithm.lean)
const INPUT_FILENAME: usize = 0x0050;
const OUTPUT_FILENAME: usize = 0x0150;
const FILE_SIZE: usize = 0x0250;
const INPUT_DATA: usize = 0x14000;

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

fn prepare_algorithm(text_path: &str, output_path: &str, file_size: usize) -> Algorithm {
    let mut alg: Algorithm =
        bincode::deserialize(WC_ALGORITHM).expect("Failed to deserialize algorithm");

    let needed = INPUT_DATA + file_size + 256;
    if alg.payloads.len() < needed {
        alg.payloads.resize(needed, 0);
    }

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

    // Reset scratch areas (CURRENT_KEY, NEW_VALUE, RESULT_SLOT)
    alg.payloads[0x0038..0x0048].fill(0);
    alg.payloads[0x3400..0x3408].fill(0);

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
    let sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000];
    let mut results = Vec::new();

    for &n in &sizes {
        let text_path = format!("/tmp/bench-data/words_{}.txt", n);
        let output_path = format!("/tmp/bench-data/wc_result_{}.txt", n);

        let expected = generate_text(&text_path, n);
        let file_size = fs::metadata(&text_path).unwrap().len() as usize;

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

        // Base
        let mut verified = None;
        let base_ms = harness::median_of(iterations, || {
            let _ = fs::remove_file(&output_path);

            let alg = prepare_algorithm(&text_path, &output_path, file_size);
            let ms = harness::run_base(alg);

            if let Ok(content) = fs::read_to_string(&output_path) {
                let got = parse_output(content.trim());
                if got == expected {
                    verified = Some(true);
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
                    verified = Some(false);
                }
            } else {
                eprintln!("WARNING: Could not read base output file {:?}", output_path);
                verified = Some(false);
            }

            ms
        });

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
