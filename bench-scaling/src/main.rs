//! Measures whether elaboration and proof cost stay bounded as a Lean-generated
//! codebase grows: bigger programs, more DSL constructs, deeper abstraction
//! stacks, and different ways of encoding a DSL.
//!
//!   cargo run --release -p bench-scaling
//!   cargo run --release -p bench-scaling -- --only derive,frontends

mod dslgen;
mod frontend;
mod gen;
mod measure;
mod print;
mod suites;

use clap::{Parser, ValueEnum};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Suite {
    /// compiles every generated variant at minimum size, so a broken generator
    /// fails in seconds rather than after a full sweep
    Smoke,
    Derive,
    Shapes,
    Depth,
    Frontends,
    Binds,
    Clif,
}

impl Suite {
    const ALL: [Suite; 7] = [
        Suite::Smoke,
        Suite::Derive,
        Suite::Shapes,
        Suite::Depth,
        Suite::Frontends,
        Suite::Binds,
        Suite::Clif,
    ];
    fn name(self) -> &'static str {
        match self {
            Suite::Smoke => "smoke",
            Suite::Derive => "derive",
            Suite::Shapes => "shapes",
            Suite::Depth => "depth",
            Suite::Frontends => "frontends",
            Suite::Binds => "binds",
            Suite::Clif => "clif",
        }
    }
}

#[derive(Parser)]
#[command(about, long_about = None)]
struct Args {
    /// run a subset of suites
    #[arg(long, value_delimiter = ',')]
    only: Vec<Suite>,
    /// leave generated inputs in .work for inspection
    #[arg(long)]
    keep_work: bool,
}

/// The toolchain the repo pins, not whatever is on PATH.
fn toolchain(repo: &std::path::Path) -> (PathBuf, String) {
    let name = ["lib", "algorithms"]
        .iter()
        .find_map(|d| {
            std::fs::read_to_string(repo.join("lean").join(d).join("lean-toolchain")).ok()
        })
        .map(|t| t.trim().to_string());
    if let (Some(nm), Some(home)) = (&name, std::env::var_os("HOME")) {
        let slug = nm.replace('/', "--").replace(':', "---");
        let p = PathBuf::from(home).join(".elan/toolchains").join(slug);
        if p.is_dir() {
            return (p.join("bin/lean"), nm.clone());
        }
    }
    if let Ok(o) = std::process::Command::new("elan").arg("which").arg("lean").output() {
        if o.status.success() {
            let p = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if !p.is_empty() {
                return (PathBuf::from(p), name.unwrap_or_else(|| "unknown".into()));
            }
        }
    }
    (PathBuf::from("lean"), name.unwrap_or_else(|| "unknown".into()))
}

fn main() -> std::process::ExitCode {
    let args = Args::parse();
    let names: Vec<Suite> = if args.only.is_empty() {
        Suite::ALL.to_vec()
    } else {
        args.only.clone()
    };

    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo = here.parent().unwrap().to_path_buf();
    // per-process, so two concurrent runs cannot delete each other's inputs
    let work = here.join(".work").join(format!("run{}", std::process::id()));
    let (lean, tc) = toolchain(&repo);

    let _ = std::fs::remove_dir_all(&work);
    if let Err(e) = std::fs::create_dir_all(&work) {
        eprintln!("cannot create {}: {e}", work.display());
        return std::process::ExitCode::from(2);
    }

    let ctx = suites::Ctx { repo, work: work.clone(), lean: lean.clone() };
    println!("{tc}  ({})", lean.display());

    if let Err(e) = suites::build_fixtures(&ctx) {
        eprintln!("\n{e}");
        return std::process::ExitCode::from(2);
    }

    let t0 = std::time::Instant::now();
    let mut raw: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for &suite in &names {
        let nm = suite.name();
        let s = std::time::Instant::now();
        let rows = match suite {
            Suite::Smoke => suites::smoke(&ctx),
            Suite::Derive => suites::derive(&ctx),
            Suite::Shapes => suites::shapes(&ctx),
            Suite::Depth => suites::depth(&ctx),
            Suite::Frontends => suites::frontends(&ctx),
            Suite::Binds => suites::binds(&ctx),
            Suite::Clif => suites::clif(&ctx),
        };
        // a suite whose measurements all failed yields plausible zeros rather
        // than an error, which reads as a pass
        let measured = rows
            .iter()
            .filter(|r| r.get("ok").and_then(|v| v.as_bool()).unwrap_or(false))
            .count();
        let skipped = rows.iter().any(|r| r.get("skipped").is_some());
        if measured == 0 && !skipped {
            eprintln!("\n{nm}: no measurement succeeded ({} attempted)", rows.len());
            return std::process::ExitCode::from(2);
        }
        let peak = rows
            .iter()
            .filter_map(|r| r.get("rss_mb").and_then(|v| v.as_f64()))
            .fold(0.0f64, f64::max);
        println!(
            "  {nm} {:.0}s ({measured} measured, peak {})",
            s.elapsed().as_secs_f64(),
            if peak >= 1024.0 {
                format!("{:.1}GB", peak / 1024.0)
            } else {
                format!("{peak:.0}MB")
            }
        );
        raw.insert(nm.to_string(), rows);
    }

    print::all(&raw);
    println!("\ntotal {:.0}s", t0.elapsed().as_secs_f64());

    if !args.keep_work {
        let _ = std::fs::remove_dir_all(&work);
    }
    std::process::ExitCode::SUCCESS
}
