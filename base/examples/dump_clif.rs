//! Print an artifact's decoded CLIF and, with `BASE_DISASM=1`, its machine code.
//!
//! Usage: `BASE_DUMP_CLIF=1 cargo run --release -p base --example dump_clif -- <artifact.cbor>`
fn main() {
    let path = std::env::args().nth(1).expect("usage: dump_clif <artifact.cbor>");
    let bytes = std::fs::read(&path).expect("reading artifact");
    let artifact = base::Artifact::from_bytes(&bytes).expect("parsing artifact");
    base::Base::new(artifact).expect("Base::new failed");
}
