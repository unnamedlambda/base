use std::f32::consts::PI;
use std::fs;
use std::process::Command;

fn get_fft_binary() -> String {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    format!("{}/../../target/{}/fft", manifest_dir, profile)
}

/// Naive DFT reference implementation: O(N^2)
/// X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*n*k/N)
fn naive_dft(input: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let n = input.len();
    let mut output = vec![(0.0f32, 0.0f32); n];
    for k in 0..n {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (idx, &(xr, xi)) in input.iter().enumerate() {
            let angle = -2.0 * std::f64::consts::PI * (idx as f64) * (k as f64) / (n as f64);
            let wr = angle.cos();
            let wi = angle.sin();
            re += (xr as f64) * wr - (xi as f64) * wi;
            im += (xr as f64) * wi + (xi as f64) * wr;
        }
        output[k] = (re as f32, im as f32);
    }
    output
}

/// Write complex array as binary f32 pairs (little-endian)
fn write_complex_file(path: &std::path::Path, data: &[(f32, f32)]) {
    let mut bytes = Vec::with_capacity(data.len() * 8);
    for &(re, im) in data {
        bytes.extend_from_slice(&re.to_le_bytes());
        bytes.extend_from_slice(&im.to_le_bytes());
    }
    fs::write(path, bytes).unwrap();
}

/// Read complex array from binary f32 pairs (little-endian)
fn read_complex_file(path: &std::path::Path) -> Vec<(f32, f32)> {
    let bytes = fs::read(path).unwrap();
    assert!(bytes.len() % 8 == 0, "File size not multiple of 8");
    let n = bytes.len() / 8;
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let re = f32::from_le_bytes(bytes[i * 8..i * 8 + 4].try_into().unwrap());
        let im = f32::from_le_bytes(bytes[i * 8 + 4..i * 8 + 8].try_into().unwrap());
        result.push((re, im));
    }
    result
}

/// Run our FFT binary, return the output complex array
fn run_base_fft(input: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let binary = get_fft_binary();
    let tmpdir = tempfile::tempdir().expect("Failed to create temp dir");
    let input_path = tmpdir.path().join("input.bin");
    write_complex_file(&input_path, input);

    let output = Command::new(&binary)
        .arg(input_path.to_str().unwrap())
        .current_dir(tmpdir.path())
        .output()
        .unwrap_or_else(|e| panic!("Failed to run {}: {}", binary, e));

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("FFT binary failed: {}", stderr);
    }

    let output_path = tmpdir.path().join("fft_output.bin");
    read_complex_file(&output_path)
}

/// Compare two complex arrays with relative tolerance
fn assert_complex_eq(got: &[(f32, f32)], expected: &[(f32, f32)], tol: f32, name: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{}: length mismatch: got {}, expected {}",
        name,
        got.len(),
        expected.len()
    );

    // Compute max magnitude for relative error
    let max_mag = expected.iter()
        .map(|&(r, i)| (r * r + i * i).sqrt())
        .fold(0.0f32, f32::max)
        .max(1.0);  // avoid division by zero

    for (idx, (&(gr, gi), &(er, ei))) in got.iter().zip(expected.iter()).enumerate() {
        let dr = (gr - er).abs();
        let di = (gi - ei).abs();
        let err = (dr * dr + di * di).sqrt();
        let rel_err = err / max_mag;
        assert!(
            rel_err < tol,
            "{}: index {}: got ({}, {}), expected ({}, {}), rel_err = {} (tol = {})",
            name, idx, gr, gi, er, ei, rel_err, tol
        );
    }
}

fn test_fft(input: &[(f32, f32)], tol: f32, name: &str) {
    let expected = naive_dft(input);
    let got = run_base_fft(input);
    assert_complex_eq(&got, &expected, tol, name);
}

// --- Basic tests ---

#[test]
fn test_n1() {
    // FFT of single element is itself
    test_fft(&[(3.0, -1.0)], 1e-5, "n1");
}

#[test]
fn test_n2() {
    // FFT of [a, b] = [a+b, a-b]
    test_fft(&[(1.0, 0.0), (0.0, 0.0)], 1e-5, "n2_impulse");
}

#[test]
fn test_n2_dc() {
    test_fft(&[(1.0, 0.0), (1.0, 0.0)], 1e-5, "n2_dc");
}

#[test]
fn test_n4_impulse() {
    // Impulse: [1, 0, 0, 0] → all ones in frequency
    test_fft(
        &[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        1e-5,
        "n4_impulse",
    );
}

#[test]
fn test_n4_dc() {
    // DC: [1, 1, 1, 1] → [4, 0, 0, 0]
    test_fft(
        &[(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
        1e-5,
        "n4_dc",
    );
}

#[test]
fn test_n4_alternating() {
    // [1, -1, 1, -1] → [0, 0, 4, 0]
    test_fft(
        &[(1.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (-1.0, 0.0)],
        1e-5,
        "n4_alternating",
    );
}

#[test]
fn test_n8_cosine() {
    // cos(2*pi*k/8) for k=0..7 — single frequency bin
    let input: Vec<(f32, f32)> = (0..8)
        .map(|k| ((2.0 * PI * k as f32 / 8.0).cos(), 0.0))
        .collect();
    test_fft(&input, 1e-4, "n8_cosine");
}

#[test]
fn test_n8_complex() {
    // Complex input
    let input: Vec<(f32, f32)> = (0..8)
        .map(|k| {
            let t = k as f32 / 8.0;
            ((2.0 * PI * t).cos(), (2.0 * PI * 2.0 * t).sin())
        })
        .collect();
    test_fft(&input, 1e-4, "n8_complex");
}

#[test]
fn test_n16_ramp() {
    let input: Vec<(f32, f32)> = (0..16).map(|i| (i as f32, 0.0)).collect();
    test_fft(&input, 1e-3, "n16_ramp");
}

// --- Larger sizes ---

#[test]
fn test_n64() {
    let input: Vec<(f32, f32)> = (0..64)
        .map(|k| {
            let t = k as f32 / 64.0;
            ((2.0 * PI * 3.0 * t).cos() + 0.5 * (2.0 * PI * 7.0 * t).sin(), 0.0)
        })
        .collect();
    test_fft(&input, 1e-3, "n64_multi_freq");
}

#[test]
fn test_n256() {
    let input: Vec<(f32, f32)> = (0..256)
        .map(|k| {
            let t = k as f32 / 256.0;
            ((2.0 * PI * 10.0 * t).cos(), (2.0 * PI * 5.0 * t).sin())
        })
        .collect();
    test_fft(&input, 1e-3, "n256_complex_signal");
}

#[test]
fn test_n1024() {
    let input: Vec<(f32, f32)> = (0..1024)
        .map(|k| {
            let t = k as f32 / 1024.0;
            (
                (2.0 * PI * 50.0 * t).cos() + 0.3 * (2.0 * PI * 120.0 * t).cos(),
                0.0,
            )
        })
        .collect();
    test_fft(&input, 1e-2, "n1024_two_freq");
}

// --- Additional sizes (stage count coverage) ---

#[test]
fn test_n32() {
    // 5 stages (odd)
    let input: Vec<(f32, f32)> = (0..32)
        .map(|k| {
            let t = k as f32 / 32.0;
            ((2.0 * PI * 5.0 * t).cos(), (2.0 * PI * 3.0 * t).sin())
        })
        .collect();
    test_fft(&input, 1e-3, "n32_complex");
}

#[test]
fn test_n128() {
    // Exactly 64 butterflies = 1 workgroup at each stage
    let input: Vec<(f32, f32)> = (0..128)
        .map(|k| {
            let t = k as f32 / 128.0;
            ((2.0 * PI * 11.0 * t).cos(), 0.0)
        })
        .collect();
    test_fft(&input, 1e-3, "n128_wg_boundary");
}

#[test]
fn test_n512() {
    // 9 stages (odd)
    let input: Vec<(f32, f32)> = (0..512)
        .map(|k| {
            let t = k as f32 / 512.0;
            ((2.0 * PI * 20.0 * t).sin() + 0.7 * (2.0 * PI * 100.0 * t).cos(), 0.0)
        })
        .collect();
    test_fft(&input, 1e-2, "n512_two_freq");
}

#[test]
fn test_n4096() {
    // 12 stages, larger scale
    let input: Vec<(f32, f32)> = (0..4096)
        .map(|k| {
            let t = k as f32 / 4096.0;
            (
                (2.0 * PI * 100.0 * t).cos() + 0.5 * (2.0 * PI * 500.0 * t).cos(),
                0.3 * (2.0 * PI * 200.0 * t).sin(),
            )
        })
        .collect();
    test_fft(&input, 1e-2, "n4096_multi");
}

// --- Edge cases ---

#[test]
fn test_all_zeros() {
    let input = vec![(0.0f32, 0.0f32); 16];
    test_fft(&input, 1e-5, "all_zeros");
}

#[test]
fn test_all_ones_imag() {
    let input: Vec<(f32, f32)> = vec![(0.0, 1.0); 32];
    test_fft(&input, 1e-3, "all_ones_imag");
}

#[test]
fn test_negative_values() {
    let input: Vec<(f32, f32)> = (0..16)
        .map(|k| (-(k as f32) - 1.0, (k as f32) * 0.5 - 4.0))
        .collect();
    test_fft(&input, 1e-3, "negative_values");
}

#[test]
fn test_large_magnitudes() {
    let input: Vec<(f32, f32)> = (0..8)
        .map(|k| (1e6 * (k as f32 + 1.0), -5e5 * (k as f32)))
        .collect();
    test_fft(&input, 1e-2, "large_magnitudes");
}

// --- Hardcoded known-answer vectors (verified against numpy) ---

#[test]
fn test_known_n2_complex() {
    // numpy.fft.fft([1+2j, 3+4j]) = [4+6j, -2-2j]
    let input = vec![(1.0, 2.0), (3.0, 4.0)];
    let got = run_base_fft(&input);
    let expected = vec![(4.0, 6.0), (-2.0, -2.0)];
    assert_complex_eq(&got, &expected, 1e-5, "known_n2_complex");
}

#[test]
fn test_known_n8_ramp() {
    // numpy.fft.fft([1,2,3,4,5,6,7,8])
    // = [36, -4+9.657j, -4+4j, -4+1.657j, -4, -4-1.657j, -4-4j, -4-9.657j]
    let input: Vec<(f32, f32)> = (1..=8).map(|k| (k as f32, 0.0)).collect();
    let got = run_base_fft(&input);
    // numpy.fft.fft([1..8]) imaginary parts: 0, 9.657, 4, 1.657, 0, -1.657, -4, -9.657
    // 9.65685... = 4 + 4*sqrt(2), 1.65685... = -4 + 4*sqrt(2)
    let sqrt2 = std::f32::consts::SQRT_2;
    let expected = vec![
        (36.0, 0.0),
        (-4.0, 4.0 + 4.0 * sqrt2),       //  9.65685...
        (-4.0, 4.0),
        (-4.0, -4.0 + 4.0 * sqrt2),      //  1.65685...
        (-4.0, 0.0),
        (-4.0, 4.0 - 4.0 * sqrt2),       // -1.65685...
        (-4.0, -4.0),
        (-4.0, -(4.0 + 4.0 * sqrt2)),    // -9.65685...
    ];
    assert_complex_eq(&got, &expected, 1e-4, "known_n8_ramp");
}

// --- DFT mathematical properties ---

#[test]
fn test_conjugate_symmetry() {
    // For real-valued input, X[k] = conj(X[N-k])
    let n = 64;
    let input: Vec<(f32, f32)> = (0..n)
        .map(|k| ((k as f32 * 0.3).sin() + (k as f32 * 0.7).cos(), 0.0))
        .collect();

    let output = run_base_fft(&input);
    let max_mag = output.iter()
        .map(|&(r, i)| (r * r + i * i).sqrt())
        .fold(0.0f32, f32::max)
        .max(1.0);

    for k in 1..n / 2 {
        let (xr, xi) = output[k];
        let (yr, yi) = output[n - k];
        // X[k] should equal conj(X[N-k]), i.e., xr == yr and xi == -yi
        let err = ((xr - yr).powi(2) + (xi + yi).powi(2)).sqrt();
        let rel_err = err / max_mag;
        assert!(
            rel_err < 1e-4,
            "Conjugate symmetry: X[{}]=({},{}), conj(X[{}])=({},{}), rel_err={}",
            k, xr, xi, n - k, yr, -yi, rel_err
        );
    }
}

#[test]
fn test_shift_theorem() {
    // Circular shift by m: x'[n] = x[(n-m) mod N]
    // FFT(x')[k] = FFT(x)[k] * exp(-2*pi*i*k*m/N)
    let n = 32usize;
    let m = 7usize;

    let input: Vec<(f32, f32)> = (0..n)
        .map(|k| ((k as f32 * 0.4).sin(), (k as f32 * 0.2).cos()))
        .collect();

    // Circular shift
    let shifted: Vec<(f32, f32)> = (0..n)
        .map(|i| input[(i + n - m) % n])
        .collect();

    let fft_orig = run_base_fft(&input);
    let fft_shifted = run_base_fft(&shifted);

    // Verify: fft_shifted[k] ≈ fft_orig[k] * exp(-2πi*k*m/N)
    let max_mag = fft_orig.iter()
        .map(|&(r, i)| (r * r + i * i).sqrt())
        .fold(0.0f32, f32::max)
        .max(1.0);

    for k in 0..n {
        let angle = -2.0 * PI * (k as f32) * (m as f32) / (n as f32);
        let (cr, ci) = (angle.cos(), angle.sin());
        let (or, oi) = fft_orig[k];
        // complex multiply: (or+oi*j) * (cr+ci*j)
        let er = or * cr - oi * ci;
        let ei = or * ci + oi * cr;
        let (sr, si) = fft_shifted[k];
        let err = ((sr - er).powi(2) + (si - ei).powi(2)).sqrt();
        let rel_err = err / max_mag;
        assert!(
            rel_err < 1e-3,
            "Shift theorem: k={}, shifted=({},{}), expected=({},{}), rel_err={}",
            k, sr, si, er, ei, rel_err
        );
    }
}

#[test]
fn test_parseval() {
    // Parseval's theorem: sum|x[n]|^2 = (1/N) * sum|X[k]|^2
    let input: Vec<(f32, f32)> = (0..128)
        .map(|k| ((k as f32 * 0.1).sin(), (k as f32 * 0.07).cos()))
        .collect();
    let n = input.len();

    let time_energy: f64 = input.iter()
        .map(|&(r, i)| (r as f64) * (r as f64) + (i as f64) * (i as f64))
        .sum();

    let output = run_base_fft(&input);
    let freq_energy: f64 = output.iter()
        .map(|&(r, i)| (r as f64) * (r as f64) + (i as f64) * (i as f64))
        .sum::<f64>()
        / (n as f64);

    let rel_err = ((time_energy - freq_energy) / time_energy).abs();
    assert!(
        rel_err < 1e-3,
        "Parseval's theorem violated: time_energy={}, freq_energy={}, rel_err={}",
        time_energy,
        freq_energy,
        rel_err,
    );
}

#[test]
fn test_linearity() {
    // FFT(a*x + b*y) should equal a*FFT(x) + b*FFT(y)
    let n = 64;
    let a = 2.5f32;
    let b = -1.3f32;

    let x: Vec<(f32, f32)> = (0..n).map(|k| ((k as f32 * 0.2).sin(), 0.0)).collect();
    let y: Vec<(f32, f32)> = (0..n).map(|k| (0.0, (k as f32 * 0.3).cos())).collect();

    let combined: Vec<(f32, f32)> = x.iter().zip(y.iter())
        .map(|(&(xr, xi), &(yr, yi))| (a * xr + b * yr, a * xi + b * yi))
        .collect();

    let fft_x = run_base_fft(&x);
    let fft_y = run_base_fft(&y);
    let fft_combined = run_base_fft(&combined);

    let expected: Vec<(f32, f32)> = fft_x.iter().zip(fft_y.iter())
        .map(|(&(xr, xi), &(yr, yi))| (a * xr + b * yr, a * xi + b * yi))
        .collect();

    assert_complex_eq(&fft_combined, &expected, 1e-3, "linearity");
}

// --- External reference comparison (numpy) ---

fn run_numpy_fft(input: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let input_json: Vec<String> = input.iter()
        .map(|&(r, i)| {
            if i == 0.0 { format!("{}", r) }
            else { format!("complex({},{})", r, i) }
        })
        .collect();
    let script = format!(
        "import numpy as np; x = np.fft.fft([{}]); \
         print(' '.join(f'{{c.real}},{{c.imag}}' for c in x))",
        input_json.join(",")
    );
    let output = Command::new("python3")
        .args(["-c", &script])
        .output()
        .expect("python3 not found");
    assert!(output.status.success(), "numpy FFT failed");
    let stdout = String::from_utf8(output.stdout).unwrap();
    stdout.trim().split(' ')
        .map(|pair| {
            let parts: Vec<&str> = pair.split(',').collect();
            (parts[0].parse::<f32>().unwrap(), parts[1].parse::<f32>().unwrap())
        })
        .collect()
}

#[test]
fn test_vs_numpy_n16() {
    let input: Vec<(f32, f32)> = (0..16)
        .map(|k| {
            let t = k as f32 / 16.0;
            ((2.0 * PI * 3.0 * t).cos(), (2.0 * PI * 5.0 * t).sin())
        })
        .collect();
    let got = run_base_fft(&input);
    let expected = run_numpy_fft(&input);
    assert_complex_eq(&got, &expected, 1e-3, "vs_numpy_n16");
}

#[test]
fn test_vs_numpy_n256() {
    let input: Vec<(f32, f32)> = (0..256)
        .map(|k| {
            let t = k as f32 / 256.0;
            (
                (2.0 * PI * 7.0 * t).sin() + 2.0 * (2.0 * PI * 31.0 * t).cos(),
                -(2.0 * PI * 13.0 * t).sin(),
            )
        })
        .collect();
    let got = run_base_fft(&input);
    let expected = run_numpy_fft(&input);
    assert_complex_eq(&got, &expected, 1e-2, "vs_numpy_n256");
}

#[test]
fn test_vs_numpy_n4096() {
    let input: Vec<(f32, f32)> = (0..4096)
        .map(|k| {
            let t = k as f32 / 4096.0;
            ((2.0 * PI * 100.0 * t).cos(), 0.0)
        })
        .collect();
    let got = run_base_fft(&input);
    let expected = run_numpy_fft(&input);
    assert_complex_eq(&got, &expected, 1e-2, "vs_numpy_n4096");
}
