//! Proven warp GEMV and sum-of-squares against cuBLAS, across the three
//! schedules (vec4, strided, blocked) and four shapes. Each schedule is exact
//! against its own fold order, so columns may differ in the last bits.

use base::Driver;
use warp_check::{floats, gbs, roofline, time, Walk};

const ART_QWEN: &[u8] =
    lean_artifacts::GEMV_WARP;
const ART_2048: &[u8] =
    lean_artifacts::GEMV_WARP_2048;
const ART_8192: &[u8] =
    lean_artifacts::GEMV_WARP_8192;
const ART_WIDE: &[u8] =
    lean_artifacts::GEMV_WARP_WIDE;


/// One kernel family at one shape: time every schedule, check each against its
/// own committed fold, and report how far apart the answers are.
struct Row {
    n: usize,
    label: &'static str,
    times: [f64; 3],
    t_blas: Option<f64>,
}

fn bench(
    art_bytes: &[u8],
    n: usize,
    m: usize,
    label: &str,
    reps: usize,
    table: &mut Vec<Row>,
    fam_label: &'static str,
) {
    let a: Vec<f32> = (0..m * n).map(|i| ((i % 251) as f32 - 125.0) * 0.001).collect();
    let x: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
    let mut bytes: Vec<u8> = a.iter().flat_map(|f| f.to_le_bytes()).collect();
    bytes.extend(x.iter().flat_map(|f| f.to_le_bytes()));

    let base = unsafe { Driver::load(art_bytes).expect("Driver::load") };
    let fetch = base.function("fetch").expect("fetch");
    let dot = fam_label == "dot";
    let runs = if dot {
        [
            ("vec4", "run"),
            ("strided", "run_strided"),
            ("blocked", "run_blocked"),
        ]
    } else {
        [
            ("vec4", "sq"),
            ("strided", "sq_strided"),
            ("blocked", "sq_blocked"),
        ]
    };

    base.function("main").expect("main")(&bytes, &mut []);

    let mut times = Vec::new();
    let mut outs: Vec<Vec<f32>> = Vec::new();
    for (_, alg) in &runs {
        times.push(time(&base, *alg, reps));
        let mut buf = vec![0u8; m * 4];
        fetch(b"", &mut buf);
        outs.push(floats(&buf));
    }
    // cuBLAS is a baseline for the dot only; there is no sgemv for x·x.
    let t_blas = if dot {
        let t = time(&base, "blas", reps);
        let mut buf = vec![0u8; m * 4];
        fetch(b"", &mut buf);
        Some((t, floats(&buf)))
    } else {
        None
    };

    // The sum of squares streams A only; the GEMV also reads x.
    let moved = if dot { (m * n + n + m) * 4 } else { (m * n + m) * 4 };

    println!("\n=== {label}: {m} x {n} ===");
    for (i, (name, _)) in runs.iter().enumerate() {
        let vs = match &t_blas {
            Some((t, _)) => format!("   {:.2}x vs cuBLAS", t / times[i]),
            None => String::new(),
        };
        println!(
            "Sched.{name:<10}: {:.4} ms   {:6.1} GB/s   {:>3.0}% roofline{vs}",
            times[i] * 1e3,
            gbs(moved, times[i]),
            100.0 * roofline(moved, times[i]),
        );
    }
    if let Some((t, _)) = &t_blas {
        println!(
            "{:<16}: {:.4} ms   {:6.1} GB/s   {:>3.0}% roofline",
            "cuBLAS",
            t * 1e3,
            gbs(moved, *t),
            100.0 * roofline(moved, *t)
        );
    }
    println!(
        "{:<16}: {:.2}x between fastest and slowest schedule",
        "spread",
        times.iter().cloned().fold(0f64, f64::max) / times.iter().cloned().fold(f64::MAX, f64::min)
    );

    // Each schedule against its own committed fold, then against each other.
    // The menu, in the order the three runs above are listed, so `Walk` and
    // `Sched` cannot drift apart without a bit-exact line failing.
    const WALKS: [Walk; 3] = [Walk::Vec4, Walk::Strided, Walk::Blocked];
    for (i, (name, _)) in runs.iter().enumerate() {
        let exact = (0..m)
            .filter(|&r| {
                // The sum of squares is the same schema with both operands on
                // one buffer, at one row base — no second reference.
                let want = if dot {
                    warp_check::dot(WALKS[i], &a, &x, r * n, 0, n)
                } else {
                    warp_check::dot(WALKS[i], &a, &a, r * n, r * n, n)
                };
                outs[i][r].to_bits() == want.to_bits()
            })
            .count();
        println!("{:<16}: {exact}/{m} bit-exact vs its own spec", format!("{name} v spec"));
        assert_eq!(exact, m, "{name} must match the committed fold it is proven against");
    }
    let mut worst = 0f32;
    for i in 0..runs.len() {
        for j in (i + 1)..runs.len() {
            let agree = (0..m).filter(|&r| outs[i][r].to_bits() == outs[j][r].to_bits()).count();
            for r in 0..m {
                let d = (outs[i][r] - outs[j][r]).abs() / outs[j][r].abs().max(1e-6);
                if d > worst {
                    worst = d;
                }
            }
            println!(
                "{:<16}: {agree}/{m} bit-identical",
                format!("{} v {}", runs[i].0, runs[j].0)
            );
        }
    }
    match &t_blas {
        Some((_, bl)) => {
            let vs = (0..m)
                .map(|r| (outs[0][r] - bl[r]).abs() / bl[r].abs().max(1e-3))
                .fold(0f32, f32::max);
            println!("worst rel       : {worst:.2e} across schedules, {vs:.2e} vs cuBLAS");
        }
        None => println!("worst rel       : {worst:.2e} across schedules"),
    }
    assert!(outs[0].iter().any(|v| v.abs() > 1e-3), "reference must be non-degenerate");
    table.push(Row {
        n,
        label: if dot { "dot" } else { "sumsq" },
        times: [times[0], times[1], times[2]],
        t_blas: t_blas.map(|(t, _)| t),
    });
}

pub fn main(_args: Vec<String>) {
    let shapes: [(&[u8], usize, usize, &str, usize); 4] = [
        (ART_QWEN, 896, 4864, "Qwen2 FFN gate/up", 200),
        (ART_2048, 2048, 4096, "n=2048", 100),
        (ART_8192, 8192, 2048, "n=8192", 60),
        (ART_WIDE, 16384, 2048, "wide row", 50),
    ];
    let mut table: Vec<Row> = Vec::new();
    for (art, n, m, label, reps) in shapes {
        bench(art, n, m, &format!("dot   {label}"), reps, &mut table, "dot");
        bench(art, n, m, &format!("sumsq {label}"), reps, &mut table, "sumsq");
    }

    println!("\n=== why tuning helps: ms by schedule ===");
    println!(
        "{:<8} {:>6}  {:>9} {:>9} {:>9}  {:>7}  {:>9}",
        "kernel", "n", "vec4", "strided", "blocked", "spread", "cuBLAS"
    );
    for r in &table {
        let best = r.times.iter().cloned().fold(f64::MAX, f64::min);
        let worst = r.times.iter().cloned().fold(0f64, f64::max);
        let blas = match r.t_blas {
            Some(t) => format!("{:9.4}", t * 1e3),
            None => format!("{:>9}", "-"),
        };
        println!(
            "{:<8} {:>6}  {:9.4} {:9.4} {:9.4}  {:6.2}x  {blas}",
            r.label,
            r.n,
            r.times[0] * 1e3,
            r.times[1] * 1e3,
            r.times[2] * 1e3,
            worst / best
        );
    }
    println!(
        "\nEvery row above is the same value computed three ways. They are equal only\n\
         under Law.laneRegroup, which Float32 does not satisfy; each kernel is exact\n\
         against its own committed fold, which is what the bit-exact lines show."
    );
}
