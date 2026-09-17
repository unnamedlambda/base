use base::Artifact;
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::harness::{self, format_count, BenchResult};

const REGEX_ARTIFACT: &[u8] = build_support::artifact!("RustBenchmarks/regex_algorithm");


const VOCABULARY: &[&str] = &[
    "the", "running", "of", "singing", "and", "to", "jumping", "in", "a", "is", "that", "finding",
    "for", "it", "was", "making", "on", "are", "as", "with", "building", "they", "at", "be",
    "this", "from", "or", "testing", "had", "by", "not", "coding", "but", "some", "what", "we",
    "writing", "can", "out", "reading",
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

fn rust_regex_count(path: &str, output_path: &str, buf: &mut Vec<u8>) -> usize {
    // Bytes, not `read_to_string`: the generated program never validates UTF-8,
    // so charging one side for a full validation pass measures the check rather
    // than the search.
    harness::read_into(path, buf).unwrap();
    let mut count = 0;
    for word in buf.split(|b| b.is_ascii_whitespace()) {
        if word.len() > 3
            && word.ends_with(b"ing")
            && word.iter().all(|b| b.is_ascii_lowercase())
        {
            count += 1;
        }
    }
    harness::write_result_sync(output_path, count as i64).unwrap();
    count
}

pub fn run(iterations: usize) -> Vec<BenchResult> {
    let sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000];
    let mut results = Vec::new();

    // JIT compile once
    let artifact = Artifact::from_bytes(REGEX_ARTIFACT);
    let mut base_instance = base::Base::new(artifact).expect("Base::new failed");

    for &n in &sizes {
        let text_path = format!("/tmp/bench-data/regex_{}.txt", n);
        let output_path = format!("/tmp/bench-data/regex_result_{}.txt", n);

        let expected = generate_text(&text_path, n);
        let payload = build_payload(&text_path, &output_path);

        // Pure Rust
        let rust_out = format!("/tmp/bench-data/rust_result_regex_{}.txt", n);
        let mut rust_buf: Vec<u8> = Vec::new();
        let _ = rust_regex_count(&text_path, &rust_out, &mut rust_buf);

        let rust_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            let count = rust_regex_count(&text_path, &rust_out, &mut rust_buf);
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            if count != expected {
                eprintln!(
                    "WARNING: Rust regex count {} != expected {} (n={})",
                    count, expected, n
                );
            }
            ms
        });

        // Base (Cranelift JIT) — execute with payload, verify output file.
        //
        // The output file is removed once, here, so this warmup creates it and
        // every timed iteration overwrites a path that already exists. That is
        // the filesystem work the baseline's `File::create` does. Removing it
        // per iteration instead charges Base for an inode allocation and a
        // directory insert a round that the baseline never pays.
        let _ = fs::remove_file(&output_path);
        let _ = base_instance.execute("main", &payload, &mut []);

        let base_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            let _ = base_instance.execute("main", &payload, &mut []);
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
                eprintln!(
                    "WARNING: Could not parse base regex output: {:?}",
                    content.trim()
                );
                Some(false)
            }
        } else {
            eprintln!("WARNING: Could not read base output file {:?}", output_path);
            Some(false)
        };

        results.push(BenchResult {
            name: format!("Regex ({})", format_count(n)),
            col_a_ms: Some(rust_ms),
            col_b_ms: None,
            base_ms,
            verified,
        });
    }

    results
}
