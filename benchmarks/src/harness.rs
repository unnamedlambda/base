use base::Algorithm;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

pub struct BenchResult {
    pub name: String,
    pub python_ms: Option<f64>,
    pub rust_ms: Option<f64>,
    pub base_ms: f64,
    pub verified: Option<bool>,
    pub actions: Option<usize>,
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

/// Execute a Base algorithm and return wall-clock time in ms.
pub fn run_base(algorithm: Algorithm) -> f64 {
    let start = Instant::now();
    match base::execute(algorithm) {
        Ok(_) => {}
        Err(e) => eprintln!("Base execution failed: {:?}", e),
    }
    start.elapsed().as_secs_f64() * 1000.0
}

/// Run a benchmark function `iterations` times and return the median.
pub fn median_of(iterations: usize, mut f: impl FnMut() -> f64) -> f64 {
    let mut times: Vec<f64> = (0..iterations).map(|_| f()).collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    times[times.len() / 2]
}

/// Print a comparison table to stdout.
pub fn print_table(results: &[BenchResult]) {
    let name_w = 20;
    let col_w = 12;

    println!();
    println!(
        "{:<name_w$} {:>col_w$} {:>col_w$} {:>col_w$} {:>6}",
        "Benchmark", "Python", "Rust", "Base", "Check",
        name_w = name_w, col_w = col_w
    );
    println!("{}", "-".repeat(name_w + col_w * 3 + 6 + 4));

    for r in results {
        let python_str = match r.python_ms {
            Some(ms) => format!("{:.1}ms", ms),
            None => "N/A".to_string(),
        };
        let rust_str = match r.rust_ms {
            Some(ms) => format!("{:.1}ms", ms),
            None => "N/A".to_string(),
        };
        let base_str = format!("{:.1}ms", r.base_ms);

        let check_str = match r.verified {
            Some(true) => "✓",
            Some(false) => "✗",
            None => "—",
        };

        println!(
            "{:<name_w$} {:>col_w$} {:>col_w$} {:>col_w$} {:>6}",
            r.name, python_str, rust_str, base_str, check_str,
            name_w = name_w, col_w = col_w
        );
    }
    println!();
}

/// Print Rust vs Burn vs Base comparison table.
pub fn print_burn_table(results: &[BenchResult]) {
    let name_w = 20;
    let col_w = 12;

    println!();
    println!(
        "{:<name_w$} {:>col_w$} {:>col_w$} {:>col_w$} {:>6}",
        "Benchmark", "Rust", "Burn", "Base", "Check",
        name_w = name_w, col_w = col_w
    );
    println!("{}", "-".repeat(name_w + col_w * 3 + 6 + 4));

    for r in results {
        let rust_str = match r.rust_ms {
            Some(ms) => format!("{:.1}ms", ms),
            None => "N/A".to_string(),
        };
        let burn_str = match r.python_ms {
            Some(ms) => format!("{:.1}ms", ms),
            None => "N/A".to_string(),
        };
        let base_str = if r.base_ms.is_nan() {
            "N/A".to_string()
        } else {
            format!("{:.1}ms", r.base_ms)
        };

        let check_str = match r.verified {
            Some(true) => "\u{2713}",
            Some(false) => "\u{2717}",
            None => "\u{2014}",
        };

        println!(
            "{:<name_w$} {:>col_w$} {:>col_w$} {:>col_w$} {:>6}",
            r.name, rust_str, burn_str, base_str, check_str,
            name_w = name_w, col_w = col_w
        );
    }
    println!();
}

fn python_dir() -> std::path::PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest).join("python")
}
