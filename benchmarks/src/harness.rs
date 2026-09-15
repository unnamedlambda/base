pub struct BenchResult {
    pub name: String,
    pub col_a_ms: Option<f64>,
    pub col_b_ms: Option<f64>,
    pub base_ms: f64,
    pub verified: Option<bool>,
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
        "Benchmark",
        col_a,
        col_b,
        "Base",
        "Check",
        name_w = name_w,
        col_w = col_w
    );
    println!("{}", "-".repeat(name_w + col_w * 3 + 6 + 4));

    for r in results {
        println!(
            "{:<name_w$} {:>col_w$} {:>col_w$} {:>col_w$} {:>6}",
            r.name,
            fmt_ms(r.col_a_ms),
            fmt_ms(r.col_b_ms),
            fmt_ms(Some(r.base_ms)),
            fmt_check(r.verified),
            name_w = name_w,
            col_w = col_w
        );
    }
    println!();
}

pub fn print_burn_table(results: &[BenchResult]) {
    print_results(results, "Rust", "Burn");
}

/// Print a 2-column table (col_a + Base only, no col_b).
pub fn print_results_2col(results: &[BenchResult], col_a: &str) {
    let name_w = 20;
    let col_w = 12;

    println!();
    println!(
        "{:<name_w$} {:>col_w$} {:>col_w$} {:>6}",
        "Benchmark",
        col_a,
        "Base",
        "Check",
        name_w = name_w,
        col_w = col_w
    );
    println!("{}", "-".repeat(name_w + col_w * 2 + 6 + 3));

    for r in results {
        println!(
            "{:<name_w$} {:>col_w$} {:>col_w$} {:>6}",
            r.name,
            fmt_ms(r.col_a_ms),
            fmt_ms(Some(r.base_ms)),
            fmt_check(r.verified),
            name_w = name_w,
            col_w = col_w
        );
    }
    println!();
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

/// Read a whole file into a caller-owned buffer.
///
/// The Base side reads into an arena it allocated once, so a comparison that
/// let the Rust side allocate a fresh file-sized `Vec` every iteration would be
/// charging one side for an allocation and first-touch the other never pays.
/// At the largest CSV input that is 282 MB a round.
pub fn read_into(path: &str, buf: &mut Vec<u8>) -> std::io::Result<()> {
    use std::io::Read;
    buf.clear();
    std::fs::File::open(path)?.read_to_end(buf)?;
    Ok(())
}

/// Write a decimal result the way `cl_file_write` does, fsync included.
///
/// The generated programs finish by rendering their answer and writing it
/// durably. `cl_file_write` calls `sync_all`, which costs about 0.7 ms on ext4,
/// so a Rust comparison that only returned the value would be doing strictly
/// less work.
pub fn write_result_sync(path: &str, value: i64) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    write!(f, "{}\n", value)?;
    f.sync_all()?;
    Ok(())
}
