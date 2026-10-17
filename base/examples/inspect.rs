//! An artifact's functions as CLIF, or with `--disasm`, as machine code:
//! `cargo run -p base --example inspect -- program.cbor [--disasm]`

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.iter().find(|a| !a.starts_with("--")) else {
        eprintln!("usage: inspect <artifact.cbor> [--disasm]");
        std::process::exit(2);
    };
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    let artifact = base::Artifact::from_bytes(&bytes).unwrap_or_else(|e| panic!("{path}: {e}"));
    let text = if args.iter().any(|a| a == "--disasm") {
        base::disassemble(&artifact)
    } else {
        base::clif_text(&artifact)
    };
    match text {
        Ok(t) => print!("{t}"),
        Err(e) => {
            eprintln!("{path}: {e}");
            std::process::exit(1);
        }
    }
}
