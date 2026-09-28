//! A host reading the `self_describing` artifact with no copy of its formats.
//!
//! The artifact describes none of its outputs; its entries answer in formats
//! the bytes carry. So these tests hold what a host would: a generic CBOR
//! decoder, and for the raw output one format id. Each asserts the whole
//! decoded value against a literal, which fails the same way whether the bytes
//! did not decode, decoded to another shape, or held another number.

use base::{Artifact, Driver};
use ciborium::Value;

const ARTIFACT: &[u8] = lean_artifacts::SELF_DESCRIBING;

/// The input every test counts: two `a`s, a `b`, a `c`.
const INPUT: &[u8] = b"abca";

fn driver() -> Driver {
    Driver::load(Artifact::from_bytes(ARTIFACT).expect("the build checked this artifact"))
        .expect("compile")
}

/// Call `entry` the way a host that does not know the output's size does: ask
/// with no buffer, then call again with as many bytes as it answered.
fn call(base: &mut Driver, entry: &str, data: &[u8]) -> Vec<u8> {
    let need = base.execute(entry, data, &mut []).expect("execute");
    let mut out = vec![0u8; usize::try_from(need).expect("a size")];
    assert_eq!(base.execute(entry, data, &mut out).expect("execute"), need);
    out
}

fn decode(bytes: &[u8]) -> Value {
    let mut rest = bytes;
    let v = ciborium::from_reader(&mut rest).expect("CBOR");
    assert!(rest.is_empty(), "{} bytes after the value", rest.len());
    v
}

fn map(entries: &[(&str, Value)]) -> Value {
    Value::Map(entries.iter().map(|(k, v)| (Value::Text(k.to_string()), v.clone())).collect())
}

fn text(s: &str) -> Value {
    Value::Text(s.to_string())
}

/// The histogram of `INPUT`, as the 256 little-endian `u64`s both outputs hold.
fn expected_histogram() -> Vec<u8> {
    let mut counts = [0u64; 256];
    counts[b'a' as usize] = 2;
    counts[b'b' as usize] = 1;
    counts[b'c' as usize] = 1;
    counts.iter().flat_map(|c| c.to_le_bytes()).collect()
}

#[test]
fn schema_describes_both_outputs() {
    let got = decode(&call(&mut driver(), "schema", &[]));
    let expected = map(&[
        (
            "stats",
            map(&[
                ("encoding", text("cbor")),
                ("format", text("base.u8stats/1")),
                (
                    "fields",
                    map(&[
                        ("format", text("text")),
                        ("count", text("uint")),
                        ("sum", text("uint")),
                        ("histogram", text("uint64le[256], tag 71")),
                    ]),
                ),
            ]),
        ),
        (
            "bulk",
            map(&[
                ("encoding", text("raw")),
                ("format_id", Value::Bytes(b"u8hist01".to_vec())),
                ("layout", Value::Array(vec![text("format_id"), text("uint64le[256]")])),
            ]),
        ),
    ]);
    assert_eq!(got, expected);
}

/// A generic decoder reads the run's values, and the map says which format it
/// is — nothing about the layout was needed to get here.
#[test]
fn stats_decode_without_a_layout() {
    let got = decode(&call(&mut driver(), "stats", INPUT));
    let expected = map(&[
        ("format", text("base.u8stats/1")),
        ("count", Value::Integer(4.into())),
        ("sum", Value::Integer((97 + 98 + 99 + 97).into())),
        ("histogram", Value::Tag(71, Box::new(Value::Bytes(expected_histogram())))),
    ]);
    assert_eq!(got, expected);
}

/// The values a run fills are written in eight bytes whatever they are, so a
/// larger input is still one decodable map with the same keys.
#[test]
fn stats_hold_values_of_any_size() {
    let input = vec![0xffu8; 70_000];
    let got = decode(&call(&mut driver(), "stats", &input));
    let Value::Map(entries) = got else { panic!("a map") };
    assert_eq!(entries[1], (text("count"), Value::Integer(70_000.into())));
    assert_eq!(entries[2], (text("sum"), Value::Integer((70_000 * 255).into())));
}

/// A host that reads raw output holds one id and one layout, and checks the id
/// once before trusting the layout.
fn read_bulk(bytes: &[u8], id: &[u8; 8]) -> Result<Vec<u64>, String> {
    let (got, body) = bytes.split_at(8.min(bytes.len()));
    if got != id {
        return Err(format!(
            "expected format {:?}, got {:?}",
            String::from_utf8_lossy(id),
            String::from_utf8_lossy(got)
        ));
    }
    if body.len() != 256 * 8 {
        return Err(format!("expected 2048 bytes after the id, got {}", body.len()));
    }
    Ok(body.chunks(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect())
}

#[test]
fn bulk_is_read_after_checking_its_id() {
    let bytes = call(&mut driver(), "bulk", INPUT);
    assert_eq!(bytes.len(), 8 + 2048);
    let expected: Vec<u64> =
        expected_histogram().chunks(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(read_bulk(&bytes, b"u8hist01"), Ok(expected));
}

/// A reader written against another version of the layout refuses the bytes
/// instead of reading them as its own.
#[test]
fn a_stale_bulk_reader_refuses() {
    let bytes = call(&mut driver(), "bulk", INPUT);
    assert_eq!(
        read_bulk(&bytes, b"u8hist00"),
        Err("expected format \"u8hist00\", got \"u8hist01\"".to_string())
    );
}

/// Every entry answers the size it needs and writes nothing into a buffer
/// that cannot hold it.
#[test]
fn a_buffer_too_small_is_left_alone() {
    let mut base = driver();
    for entry in ["schema", "stats", "bulk"] {
        let need = base.execute(entry, INPUT, &mut []).expect("execute");
        let mut out = vec![0xaau8; need as usize - 1];
        assert_eq!(base.execute(entry, INPUT, &mut out).expect("execute"), need, "{entry}");
        assert!(out.iter().all(|&b| b == 0xaa), "{entry} wrote into a buffer too small for it");
    }
}
