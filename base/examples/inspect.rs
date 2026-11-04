//! An artifact's functions as CLIF, or with `--disasm`, as machine code:
//! `cargo run -p base --example inspect -- program.cbor [--disasm]`
//!
//! It decodes as the driver does, with the driver's own decoder and target, so
//! what it prints is what a load compiles.

#[path = "../src/clif_decode.rs"]
#[allow(dead_code)]
mod clif_decode;

use base_types::Artifact;

/// The decoded functions, and the name each `FuncId` stands for.
fn decoded(artifact: &Artifact) -> Result<(Vec<cranelift_codegen::ir::Function>, Vec<String>), String> {
    let d = clif_decode::decode(&artifact.primitives, &artifact.functions, clif_decode::isa().default_call_conv())?;
    let names = artifact
        .primitives
        .iter()
        .map(|p| format!("%{}!{}", p.lib, p.symbol))
        .chain((0..artifact.functions.len()).map(|n| format!("u0:{n}")))
        .collect();
    Ok((d.functions, names))
}

/// The program as CLIF text, each callee by its name.
fn clif_text(artifact: &Artifact) -> Result<String, String> {
    let (functions, names) = decoded(artifact)?;
    let mut out = String::new();
    for func in functions {
        for line in func.display().to_string().lines() {
            // A callee prints as its `FuncId`, `u0:K sig`.
            let named = line.find("u0:").and_then(|at| {
                let digits = line[at + 3..].find(|c: char| !c.is_ascii_digit()).map(|n| at + 3 + n)?;
                let k: usize = line[at + 3..digits].parse().ok()?;
                let name = names.get(k).filter(|_| line[digits..].starts_with(" sig"))?;
                Some(format!("{}{name}{}", &line[..at], &line[digits..]))
            });
            out.push_str(named.as_deref().unwrap_or(line));
            out.push('\n');
        }
        out.push('\n');
    }
    Ok(out)
}

/// The machine code each function compiles to on this host.
fn disassemble(artifact: &Artifact) -> Result<String, String> {
    let isa = clif_decode::isa();
    let mut out = String::new();
    for (i, func) in decoded(artifact)?.0.into_iter().enumerate() {
        let mut ctx = cranelift_codegen::Context::for_function(func);
        ctx.set_disasm(true);
        let code = ctx
            .compile(&*isa, &mut Default::default())
            .map_err(|e| format!("compiling u0:{i}: {:?}", e.inner))?;
        out.push_str(&format!("=== u0:{i} ===\n"));
        out.push_str(code.vcode.as_deref().unwrap_or(""));
        out.push('\n');
    }
    Ok(out)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.iter().find(|a| !a.starts_with("--")) else {
        eprintln!("usage: inspect <artifact.cbor> [--disasm]");
        std::process::exit(2);
    };
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    let artifact = Artifact::from_bytes(&bytes).unwrap_or_else(|e| panic!("{path}: {e}"));
    let text = if args.iter().any(|a| a == "--disasm") { disassemble(&artifact) } else { clif_text(&artifact) };
    match text {
        Ok(t) => print!("{t}"),
        Err(e) => {
            eprintln!("{path}: {e}");
            std::process::exit(1);
        }
    }
}
