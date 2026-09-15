//! Print an artifact's decoded CLIF and, with `BASE_DISASM=1`, its machine code.
//!
//! Usage: `BASE_DUMP_CLIF=1 cargo run --release -p base --example dump_clif -- <artifact.json>`
fn main() {
    let path = std::env::args().nth(1).expect("usage: dump_clif <artifact.json>");
    let text = std::fs::read_to_string(&path).expect("reading artifact");
    let artifact: base::Artifact = serde_json::from_str(&text).expect("parsing artifact");
    base::Base::new(artifact).expect("Base::new failed");
}
