//! `byte_scrub` copies a fixed window with NULs replaced by spaces; the
//! patterns catch a wrong loop bound or a blend taking the wrong side.

use base::{Artifact, Driver};

const ARTIFACT: &[u8] = lean_artifacts::BYTE_SCRUB;


/// The run `ByteScrub.code` ships: 256 vectors of sixteen bytes.
const BYTES: usize = 4096;
/// The byte `ByteScrub.scrub` takes out, and what it puts in its place.
const OLD: u8 = 0;
const NEW: u8 = b' ';

fn scrub(base: &mut Driver, entry: &str, data: &[u8]) -> Vec<u8> {
    let mut out = vec![0xAAu8; BYTES];
    unsafe { base.execute(entry, data, &mut out) }
        .expect("execute failed");
    out
}

#[test]
fn scrubs_every_vector() {
    let artifact = Artifact::from_bytes(ARTIFACT).expect("the build checked this artifact");
    let mut base = Driver::load(artifact).expect("Driver::load");

    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("none", vec![b'x'; BYTES]),
        ("all", vec![OLD; BYTES]),
        (
            "alternating",
            (0..BYTES)
                .map(|i| if i % 2 == 0 { OLD } else { b'x' })
                .collect(),
        ),
        (
            "first only",
            (0..BYTES)
                .map(|i| if i == 0 { OLD } else { b'x' })
                .collect(),
        ),
        (
            "last only",
            (0..BYTES)
                .map(|i| if i == BYTES - 1 { OLD } else { b'x' })
                .collect(),
        ),
        (
            "final vector",
            (0..BYTES)
                .map(|i| if i >= BYTES - 16 { OLD } else { b'x' })
                .collect(),
        ),
        // Every byte value, so a blend that truncates or sign-extends a lane
        // fails here and nowhere else.
        ("all byte values", (0..BYTES).map(|i| i as u8).collect()),
    ];

    for (label, data) in &cases {
        let got = scrub(&mut base, "main", data);
        let want: Vec<u8> = data
            .iter()
            .map(|&b| if b == OLD { NEW } else { b })
            .collect();
        assert_eq!(got.len(), want.len(), "{label}: length");
        for i in 0..BYTES {
            assert_eq!(got[i], want[i], "{label}: byte {i}");
        }
        assert!(!got.contains(&OLD), "{label}: a NUL survived");
    }
}
