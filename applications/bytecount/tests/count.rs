//! `byte_count` counts needles over a fixed window; the patterns catch a loop
//! that stops early, runs long, or stores counters out of order.

use base::Driver;

const ARTIFACT: &[u8] = lean_artifacts::BYTE_COUNT;


/// The run `ByteCount.code` ships: `ByteCount.VECTORS` vectors of sixteen bytes.
const WINDOW: usize = 4096;
/// Must match `ByteCount.needles`, in order.
const NEEDLES: [u8; 3] = [b',', b'\n', b' '];

fn counts(base: &Driver, entry: &str, data: &[u8]) -> [u64; NEEDLES.len()] {
    let mut out = [0u8; 8 * NEEDLES.len()];
    unsafe { base.function(entry) }.expect("no such entry")(data, &mut out);
    let mut got = [0u64; NEEDLES.len()];
    for (i, slot) in got.iter_mut().enumerate() {
        *slot = u64::from_le_bytes(out[i * 8..(i + 1) * 8].try_into().unwrap());
    }
    got
}

#[test]
fn counts_every_needle() {
    let base = Driver::load(ARTIFACT).expect("Driver::load");

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
        let got = counts(&base, "main", data);
        for (i, needle) in NEEDLES.iter().enumerate() {
            let expected = data.iter().filter(|&b| b == needle).count() as u64;
            assert_eq!(got[i], expected, "{label}: needle {:?}", *needle as char);
        }
    }
}
