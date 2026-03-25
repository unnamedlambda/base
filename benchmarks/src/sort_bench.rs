use base::{BaseConfig, Algorithm};
use crate::harness::{self, BenchResult, format_count};

const SORT_ALGORITHM: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/sort_algorithm.bin"));

fn load_algorithm() -> (BaseConfig, Algorithm) {
    bincode::deserialize(SORT_ALGORITHM).expect("Failed to deserialize sort algorithm")
}

fn generate_data(n: usize) -> Vec<i32> {
    let mut values = Vec::with_capacity(n);
    for i in 0..n {
        let v = ((i as i64).wrapping_mul(1_103_515_245).wrapping_add(12_345) % 2_000_000) as i32
            - 1_000_000;
        values.push(v);
    }
    values
}

fn build_payload(values: &[i32]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(values.len() * 4);
    for &v in values {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    payload
}

fn i32_from_bytes(data: &[u8]) -> Vec<i32> {
    data.chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn rust_sort(values: &[i32]) -> Vec<i32> {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted
}

fn verify_sorted(actual: &[i32], expected: &[i32], label: &str, impl_name: &str) -> bool {
    if actual.len() != expected.len() {
        eprintln!("  VERIFY FAIL [{}] {}: length mismatch: {} vs {}", label, impl_name, actual.len(), expected.len());
        return false;
    }
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if a != e {
            eprintln!("  VERIFY FAIL [{}] {}: element {} got {}, expected {}", label, impl_name, i, a, e);
            return false;
        }
    }
    true
}

pub fn run(iterations: usize) -> Vec<BenchResult> {
    let sizes = [10_000, 100_000, 500_000, 1_000_000, 5_000_000];
    let mut results = Vec::new();

    let (config, alg) = load_algorithm();
    let mut base_instance = base::Base::new(config).expect("Base::new failed");

    for &n in &sizes {
        let label = format!("Sort ({})", format_count(n));
        let values = generate_data(n);
        let expected = rust_sort(&values);
        let payload = build_payload(&values);
        let out_size = n * 4 * 2;  // 2x: first half = sorted result, second half = radix temp
        let mut out_buf = vec![0u8; out_size];

        // Rust
        let rust_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            std::hint::black_box(rust_sort(&values));
            start.elapsed().as_secs_f64() * 1000.0
        });

        // Warmup Base
        let _ = base_instance.execute_into(&alg, &payload, &mut out_buf);

        // Base
        let base_ms = harness::median_of(iterations, || {
            let start = std::time::Instant::now();
            let _ = base_instance.execute_into(&alg, &payload, &mut out_buf);
            start.elapsed().as_secs_f64() * 1000.0
        });

        // Verify
        let rust_ok = verify_sorted(&rust_sort(&values), &expected, &label, "Rust");
        let base_result = i32_from_bytes(&out_buf[..n * 4]);
        let base_ok = verify_sorted(&base_result, &expected, &label, "Base");

        results.push(BenchResult {
            name: label,
            col_a_ms: None,
            col_b_ms: Some(rust_ms),
            base_ms,
            verified: Some(rust_ok && base_ok),
        });
    }

    results
}
