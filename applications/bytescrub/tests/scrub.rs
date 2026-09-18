//! The `byte_scrub` artifact copies a fixed-size run, replacing every NUL with
//! a space.
//!
//! `ByteScrub.icmp_masks_nuls` and `ByteScrub.bitselect_scrubs` cover what the
//! compare and the blend compute for every sixteen bytes, under the CLIF
//! semantics, on the two instructions `ByteScrub.blendInsts_ship` names as the
//! ones block 2 carries. This checks the rest --- that the loop repeating them
//! walks the whole run, and loads and stores where it should.
//!
//! The patterns are chosen so a loop with the wrong bound disagrees: `last
//! only` and `final vector` catch one that stops early, `all` catches one that
//! runs long, and `alternating` catches a blend that takes the wrong side.

use base::{Artifact, Base};

const ARTIFACT: &[u8] = build_support::artifact!("ByteScrubAlgorithm/byte_scrub");


/// The run `ByteScrub.code` ships: 256 vectors of sixteen bytes.
const BYTES: usize = 4096;
/// The byte `ByteScrub.scrub` takes out, and what it puts in its place.
const OLD: u8 = 0;
const NEW: u8 = b' ';

fn scrub(base: &mut Base, entry: &str, data: &[u8]) -> Vec<u8> {
    let mut out = vec![0xAAu8; BYTES];
    base.execute(entry, data, &mut out)
        .expect("execute failed");
    out
}

#[test]
fn scrubs_every_vector() {
    let artifact = Artifact::from_bytes(ARTIFACT).expect("the build checked this artifact");
    let mut base = Base::new(artifact).expect("Base::new");

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
