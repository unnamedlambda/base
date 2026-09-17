//! Print an artifact as JSON, for reading one rather than running it.
//!
//! Usage: `cargo run --release -p base-types --example artifact_json -- <artifact.cbor>`
//!
//! JSON is a view here and nothing reads it back: integers past 2^53 print
//! exactly, and a reader that parses them into doubles will round them.
use base_types::Artifact;

fn main() {
    let path = std::env::args().nth(1).expect("usage: artifact_json <artifact.cbor>");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let t = std::time::Instant::now();
    let artifact = Artifact::from_bytes(&bytes).unwrap_or_else(|e| panic!("{path}: {e}"));
    eprintln!("{path}: {} bytes, decoded in {:.1} ms", bytes.len(), t.elapsed().as_secs_f64() * 1e3);
    println!("{}", serde_json::to_string_pretty(&artifact).expect("an artifact prints as JSON"));
}
