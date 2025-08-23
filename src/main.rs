use base::{execute, Action, Algorithm, Kind};

fn main() {
    // Example 1: Default execution
    println!("Running default algorithm...");
    let default_alg = Algorithm::default();
    match execute(default_alg) {
        Ok(()) => println!("✓ Default execution complete"),
        Err(e) => eprintln!("✗ Error: {:?}", e),
    }

    // Example 2: Simple SIMD computation
    println!("\nRunning SIMD computation...");
    let mut simd_alg = Algorithm::default();
    simd_alg.actions = vec![
        Action {
            kind: Kind::SimdLoad,
            dst: 0,
            src: 0,
            offset: 0,
            size: 16,
        },
        Action {
            kind: Kind::SimdLoad,
            dst: 1,
            src: 0,
            offset: 16,
            size: 16,
        },
        Action {
            kind: Kind::SimdAdd,
            dst: 2,
            src: 0,
            offset: 1,
            size: 0,
        },
        Action {
            kind: Kind::SimdStore,
            dst: 0,
            src: 2,
            offset: 32,
            size: 16,
        },
    ];

    // Initialize some test data
    for i in 0..32 {
        simd_alg.payloads[i] = (i as f32).to_le_bytes()[0];
    }

    match execute(simd_alg) {
        Ok(()) => println!("✓ SIMD execution complete"),
        Err(e) => eprintln!("✗ Error: {:?}", e),
    }

    // Example 3: Custom configuration
    println!("\nRunning with custom configuration...");
    let mut custom_alg = Algorithm::default();
    custom_alg.worker_threads = Some(2);
    custom_alg.timeout_ms = Some(5000);
    custom_alg.units.simd_units = 2;
    custom_alg.state.unit_scratch_offsets = vec![0, 8192];

    match execute(custom_alg) {
        Ok(()) => println!("✓ Custom execution complete"),
        Err(e) => eprintln!("✗ Error: {:?}", e),
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_pipeline() {
        let mut alg = Algorithm::default();
        alg.actions = vec![
            Action {
                kind: Kind::SimdLoad,
                dst: 0,
                src: 0,
                offset: 0,
                size: 16,
            },
            Action {
                kind: Kind::SimdStore,
                dst: 0,
                src: 0,
                offset: 0,
                size: 16,
            },
        ];

        assert!(execute(alg).is_ok());
    }

    #[test]
    fn test_concurrent_units() {
        let mut alg = Algorithm::default();
        // Create actions for multiple SIMD units
        for i in 0..4 {
            alg.actions.push(Action {
                kind: Kind::SimdLoad,
                dst: 0,
                src: 0,
                offset: i * 16,
                size: 16,
            });
        }

        assert!(execute(alg).is_ok());
    }
}
