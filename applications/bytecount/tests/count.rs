//! The `byte_count` artifact counts several bytes in a fixed-size run.
//!
//! The generator ties the width of the output buffer to the length of the
//! needle table, so a table and a buffer that disagree is a build error rather
//! than a write past the end. `ByteCountProof.trip_counts` covers what one
//! vector's trip computes, under the CLIF semantics. This checks the rest ---
//! that the loops repeating it accumulate and store what they should.
//!
//! The patterns are chosen so a loop with the wrong bound disagrees: `last
//! only` and `final vector` catch one that stops early, `all` catches one that
//! runs long or double-counts. `mixed` additionally catches a scan that writes
//! its counters in the wrong order, which the single-needle version could not.
//!
//! One limit worth naming: the artifact scans exactly `WINDOW` bytes, so a host
//! passing fewer would have it read past the input. The extent the interface
//! checks is the output buffer's, not the input's --- the input's length is not
//! a generation-time fact.

use base::{Artifact, Driver};

const ARTIFACT: &[u8] = lean_artifacts::BYTE_COUNT;


/// The run `ByteCount.code` ships: `ByteCount.VECTORS` vectors of sixteen bytes.
const WINDOW: usize = 4096;
/// Must match `ByteCount.needles`, in order.
const NEEDLES: [u8; 3] = [b',', b'\n', b' '];

fn counts(base: &mut Driver, entry: &str, data: &[u8]) -> [u64; NEEDLES.len()] {
    let mut out = [0u8; 8 * NEEDLES.len()];
    base.execute(entry, data, &mut out)
        .expect("execute failed");
    let mut got = [0u64; NEEDLES.len()];
    for (i, slot) in got.iter_mut().enumerate() {
        *slot = u64::from_le_bytes(out[i * 8..(i + 1) * 8].try_into().unwrap());
    }
    got
}

#[test]
fn counts_every_needle() {
    let artifact = Artifact::from_bytes(ARTIFACT).expect("the build checked this artifact");
    let mut base = Driver::load(artifact).expect("Driver::load");

    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("none", vec![b'x'; WINDOW]),
        ("all commas", vec![NEEDLES[0]; WINDOW]),
        ("all newlines", vec![NEEDLES[1]; WINDOW]),
        (
            "every 7th comma",
            (0..WINDOW)
                .map(|i| if i % 7 == 0 { NEEDLES[0] } else { b'x' })
                .collect(),
        ),
        (
            "first only",
            (0..WINDOW)
                .map(|i| if i == 0 { NEEDLES[0] } else { b'x' })
                .collect(),
        ),
        (
            "last only",
            (0..WINDOW)
                .map(|i| if i == WINDOW - 1 { NEEDLES[0] } else { b'x' })
                .collect(),
        ),
        (
            "final vector",
            (0..WINDOW)
                .map(|i| if i >= WINDOW - 16 { NEEDLES[0] } else { b'x' })
                .collect(),
        ),
        // Distinct counts per needle, so a scan that writes its counters in the
        // wrong order fails here and nowhere else.
        (
            "mixed",
            (0..WINDOW)
                .map(|i| match i % 8 {
                    0 => NEEDLES[0],
                    1 | 2 => NEEDLES[1],
                    3 | 4 | 5 => NEEDLES[2],
                    _ => b'x',
                })
                .collect(),
        ),
    ];

    for (label, data) in &cases {
        let got = counts(&mut base, "main", data);
        for (i, needle) in NEEDLES.iter().enumerate() {
            let expected = data.iter().filter(|&b| b == needle).count() as u64;
            assert_eq!(got[i], expected, "{label}: needle {:?}", *needle as char);
        }
    }
}
