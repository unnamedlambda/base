use std::path::Path;
use std::process::Command;
use std::time::Instant;

pub struct BenchResult {
    pub name: String,
    pub col_a_ms: Option<f64>,
    pub col_b_ms: Option<f64>,
    pub base_ms: f64,
    pub verified: Option<bool>,
}


/// Run a Python script and return wall-clock time in ms, plus stdout.
/// Returns None if python3 is not available or the script printed SKIP to stderr.
pub fn run_python(script: &str, args: &[&str]) -> Option<(f64, String)> {
    let script_path = python_dir().join(script);
    if !script_path.exists() {
        return None;
    }

    let python_cmd = "python3";

    let start = Instant::now();
    let output = Command::new(&python_cmd)
        .arg(&script_path)
        .args(args)
        .output();

    let output = match output {
        Ok(o) => o,
        Err(_) => return None,
    };

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let stderr = String::from_utf8_lossy(&output.stderr);

    if stderr.contains("SKIP") || !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Some((elapsed_ms, stdout))
}

/// Run a benchmark function `iterations` times and return the median.
pub fn median_of(iterations: usize, mut f: impl FnMut() -> f64) -> f64 {
    let mut times: Vec<f64> = (0..iterations).map(|_| f()).collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    times[times.len() / 2]
}

fn fmt_ms(ms: Option<f64>) -> String {
    match ms {
        Some(v) if !v.is_nan() => format!("{:.1}ms", v),
        _ => "N/A".to_string(),
    }
}

fn fmt_check(v: Option<bool>) -> &'static str {
    match v {
        Some(true) => "\u{2713}",
        Some(false) => "\u{2717}",
        None => "\u{2014}",
    }
}

/// Print a 3-column comparison table with custom column labels.
pub fn print_results(results: &[BenchResult], col_a: &str, col_b: &str) {
    let name_w = 20;
    let col_w = 12;

    println!();
    println!(
        "{:<name_w$} {:>col_w$} {:>col_w$} {:>col_w$} {:>6}",
        "Benchmark", col_a, col_b, "Base", "Check",
        name_w = name_w, col_w = col_w
    );
    println!("{}", "-".repeat(name_w + col_w * 3 + 6 + 4));

    for r in results {
        println!(
            "{:<name_w$} {:>col_w$} {:>col_w$} {:>col_w$} {:>6}",
            r.name, fmt_ms(r.col_a_ms), fmt_ms(r.col_b_ms),
            fmt_ms(Some(r.base_ms)), fmt_check(r.verified),
            name_w = name_w, col_w = col_w
        );
    }
    println!();
}

pub fn print_table(results: &[BenchResult]) {
    print_results(results, "Python", "Rust");
}

pub fn print_burn_table(results: &[BenchResult]) {
    print_results(results, "Rust", "Burn");
}

/// Print a 2-column table (col_b + Base only, no col_a).
pub fn print_results_2col(results: &[BenchResult], col_b: &str) {
    let name_w = 20;
    let col_w = 12;

    println!();
    println!(
        "{:<name_w$} {:>col_w$} {:>col_w$} {:>6}",
        "Benchmark", col_b, "Base", "Check",
        name_w = name_w, col_w = col_w
    );
    println!("{}", "-".repeat(name_w + col_w * 2 + 6 + 3));

    for r in results {
        println!(
            "{:<name_w$} {:>col_w$} {:>col_w$} {:>6}",
            r.name, fmt_ms(r.col_b_ms),
            fmt_ms(Some(r.base_ms)), fmt_check(r.verified),
            name_w = name_w, col_w = col_w
        );
    }
    println!();
}

fn python_dir() -> std::path::PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest).join("python")
}

// ---------------------------------------------------------------------------
// Shared utilities used across benchmark files
// ---------------------------------------------------------------------------

pub fn gen_floats(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (state >> 33) as i32;
        out.push(bits as f32 / i32::MAX as f32);
    }
    out
}

pub fn format_count(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        format!("{}", n)
    }
}

pub fn build_f32_payload(slices: &[&[f32]]) -> Vec<u8> {
    let total: usize = slices.iter().map(|s| s.len()).sum();
    let mut payload = Vec::with_capacity(total * 4);
    for slice in slices {
        for &v in *slice {
            payload.extend_from_slice(&v.to_le_bytes());
        }
    }
    payload
}

pub fn f32_from_bytes(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

pub fn f32_sum(data: &[u8]) -> f64 {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
        .sum()
}
