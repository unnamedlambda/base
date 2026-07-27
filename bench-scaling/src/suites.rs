use crate::gen::{self, BindStyle, DeriveVariant};
use crate::measure::run;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub struct Ctx {
    pub repo: PathBuf,
    pub work: PathBuf,
    pub lean: PathBuf,
}

impl Ctx {
    /// Elaborate `src`, optionally emitting an olean.
    fn lean(&self, src: &Path, olean: Option<&Path>, timeout: u64) -> crate::measure::Meas {
        let mut args = vec![
            self.lean.display().to_string(),
            "-s".into(),
            "262144".into(),
            src.display().to_string(),
        ];
        if let Some(o) = olean {
            args.push("-o".into());
            args.push(o.display().to_string());
        }
        let env = [("LEAN_PATH".to_string(), self.work.display().to_string())];
        crate::measure::measure(&args, Some(&self.work), &env, timeout)
    }
}

impl Ctx {
}

fn err_of(m: &crate::measure::Meas) -> Value {
    if m.ok {
        return Value::Null;
    }
    let head: String = m.stdout.chars().take(400).collect();
    Value::from(if m.timed_out { format!("TIMEOUT: {head}") } else { head })
}

/// Compile the fixtures so generated files can import them.
pub fn build_fixtures(ctx: &Ctx) -> Result<(), String> {
    use crate::dslgen::{self, Index, WfStyle};

    let toy = ctx.work.join("ToyG.lean");
    dslgen::dsl(3, Index::None, WfStyle::Recursive, &toy)
        .map_err(|e| format!("writing ToyG.lean: {e}"))?;
    let eff = ctx.work.join("Eff.lean");
    gen::eff(&eff).map_err(|e| format!("writing Eff.lean: {e}"))?;

    for src in [&toy, &eff] {
        let m = ctx.lean(src, Some(&src.with_extension("olean")), 300);
        if !m.ok {
            return Err(format!(
                "{} failed to build -- the scale model is broken:\n{}",
                src.display(),
                m.stdout
            ));
        }
    }
    Ok(())
}

/// Proof-term size and check cost vs emitted AST size.
pub fn derive(ctx: &Ctx) -> Vec<Value> {
    // smoke and shapes also write ToyG.lean, so regenerate the one this suite
    // measures rather than inheriting whichever ran last
    let toy = ctx.work.join("ToyG.lean");
    if crate::dslgen::dsl(3, crate::dslgen::Index::None,
                          crate::dslgen::WfStyle::Recursive, &toy).is_ok() {
        let _ = ctx.lean(&toy, Some(&toy.with_extension("olean")), 300);
    }
    let depths: &[u32] = &[8, 10, 12];
    let variants = [
        DeriveVariant::Base,
        DeriveVariant::Rfl,
        DeriveVariant::Native,
        DeriveVariant::Tactic,
    ];
    let mut rows = Vec::new();
    for &d in depths {
        let mut base: (Option<f64>, Option<u64>) = (None, None);
        for &v in variants.iter() {
            if v == DeriveVariant::Tactic && d > 10 {
                continue;
            }
            let src = ctx.work.join(format!("derive_{}_{}.lean", d, v.name()));
            if gen::derive(d, v, &src).is_err() {
                continue;
            }
            let olean = src.with_extension("olean");
            let m = ctx.lean(&src, Some(&olean), 600);
            let size = if m.ok {
                std::fs::metadata(&olean).ok().map(|x| x.len())
            } else {
                None
            };
            if v == DeriveVariant::Base {
                base = (m.secs, size);
            }
            rows.push(json!({
                "leaves": 1u64 << d,
                "variant": v.name(),
                "ok": m.ok,
                "secs": m.secs,
                "rss_mb": m.rss_mb,
                "olean_bytes": size,
                // cost attributable to the proof, net of elaborating the program.
                "proof_secs": m.secs.zip(base.0).map(|(a, b)| a - b),
                "proof_bytes": size.zip(base.1).map(|(a, b)| a as i64 - b as i64),
            }));
        }
    }
    rows
}

pub fn binds(ctx: &Ctx) -> Vec<Value> {
    let ns = vec![100, 200, 400, 800];
    let mut rows = Vec::new();
    for &n in &ns {
        for style in [BindStyle::Do, BindStyle::Bind, BindStyle::Indexed] {
            if style != BindStyle::Do && n > 400 {
                continue;
            }
            let src = ctx.work.join(format!("binds_{}_{}.lean", n, style.name()));
            if gen::binds(n, style, &src).is_err() {
                continue;
            }
            let m = ctx.lean(&src, None, 600);
            rows.push(json!({
                "binds": n, "style": style.name(),
                "ok": m.ok, "secs": m.secs, "rss_mb": m.rss_mb,
            }));
        }
    }
    rows
}

//  clif: the real IRBuilder, then Cranelift

pub fn clif(ctx: &Ctx) -> Vec<Value> {
    let algo = ctx.repo.join("lean").join("algorithms");
    if !algo.is_dir() {
        return vec![json!({"skipped": "lean/algorithms not found"})];
    }
    let src = ctx.work.join("ClifGen.lean");
    if gen::clif_gen(&src).is_err() {
        return vec![json!({"skipped": "could not write generator"})];
    }
    let srcs = src.display().to_string();
    // The top two sizes are what lift generation clear of the ~1.4s import baseline; below them `gen_secs` is the difference of two nearly-equal numbers and the ratio is noise.
    let ns = vec![10_000, 100_000, 400_000, 800_000];
    let clifdir = ctx.work.join("clif");
    let _ = std::fs::create_dir_all(&clifdir);
    let mut rows = Vec::new();

    // baseline: process start + AlgorithmLib import, emitting nothing.
    let base = run(
        &["lake", "env", "lean", "--run", &srcs, "dead", "0"],
        Some(&algo),
        &[],
        900,
    );
    if !base.ok {
        return vec![json!({
            "skipped": "generator baseline failed -- is AlgorithmLib built? (`lake build` in lean/lib)",
            "output": base.stdout.chars().take(600).collect::<String>(),
        })];
    }
    rows.push(json!({"kind": "import_baseline", "secs": base.secs}));

    for mode in ["dead", "live"] {
        for &n in &ns {
            let path = clifdir.join(format!("{}{}.clif", mode, n));
            let m = run(
                &[
                    "lake", "env", "lean", "--run", &srcs, mode,
                    &n.to_string(), &path.display().to_string(),
                ],
                Some(&algo),
                &[],
                1800,
            );
            let bytes = m
                .stdout
                .lines()
                .last()
                .and_then(|l| l.split('\t').nth(1))
                .and_then(|s| s.trim().parse::<u64>().ok());
            let gen_secs = m.secs.zip(base.secs).map(|(a, b)| a - b);
            rows.push(json!({
                "kind": "generate", "mode": mode, "insts": n, "ok": m.ok,
                "secs": m.secs, "rss_mb": m.rss_mb, "clif_bytes": bytes,
                "gen_secs": gen_secs,
                "us_per_inst": gen_secs.map(|s| s * 1e6 / n as f64),
            }));
        }
    }

    // Cranelift parse+JIT, if the companion bin has been built.
    let bin = ctx.repo.join("target").join("release").join("clifbench");
    if bin.exists() {
        let mut cmd = vec![bin.display().to_string()];
        if let Ok(rd) = std::fs::read_dir(&clifdir) {
            let mut fs_: Vec<String> =
                rd.flatten().map(|e| e.path().display().to_string()).collect();
            fs_.sort();
            cmd.extend(fs_);
        }
        let m = crate::measure::measure(&cmd, None, &[], 1800);
        for line in m.stdout.lines() {
            let p: Vec<&str> = line.split('\t').collect();
            if p.len() == 3 && !p[2].starts_with("ERR") {
                rows.push(json!({
                    "kind": "jit", "ok": true, "file": p[0],
                    "bytes": p[1].parse::<u64>().unwrap_or(0),
                    "secs": p[2].parse::<f64>().unwrap_or(0.0),
                }));
            }
        }
    } else {
        rows.push(json!({
            "kind": "jit",
            "skipped": "cargo build --release -p clifbench for parse+JIT numbers"
        }));
    }
    rows
}

//  shapes: sweep the DSL's DESIGN, not just program size

pub fn shapes(ctx: &Ctx) -> Vec<Value> {
    use crate::dslgen::{self, Index, WfStyle};

    let kind_counts = vec![3, 10, 30];
    let wf_styles: &[WfStyle] = &[WfStyle::Recursive, WfStyle::Inductive];
    let depths: &[u32] = &[8, 10];
    let mut rows = Vec::new();

    for &kinds in &kind_counts {
        for &wf in wf_styles {
            // the DSL definition itself: emit, denote, WF, checker, emit_correct.
            let dsl_src = ctx.work.join("ToyG.lean");
            if dslgen::dsl(kinds, Index::None, wf, &dsl_src).is_err() {
                continue;
            }
            let m = ctx.lean(&dsl_src, Some(&dsl_src.with_extension("olean")), 900);
            rows.push(json!({
                "kind": "dsl", "kinds": kinds, "index": Index::None.name(), "wf": wf.name(),
                "ok": m.ok, "secs": m.secs, "rss_mb": m.rss_mb,
                "err": err_of(&m),
            }));
            if !m.ok {
                continue;
            }

            // programs over it -- baseline first, so proof cost can be netted out.
            for &d in depths {
                let base_src = ctx.work.join(format!("shp_{kinds}_{}_{d}_base.lean", wf.name()));
                let _ = std::fs::write(
                    &base_src,
                    format!(
                        "import ToyG\nopen ToyG\nset_option maxRecDepth 8000000\n\n\
                         def p : Stmt := mkD {d} 0\n"
                    ),
                );
                let bm = ctx.lean(&base_src, Some(&base_src.with_extension("olean")), 900);
                let bsz = std::fs::metadata(base_src.with_extension("olean")).ok().map(|x| x.len());

                let src = ctx.work.join(format!("shp_{kinds}_{}_{d}.lean", wf.name()));
                if dslgen::dsl_program(d, kinds, Index::None, &src).is_err() {
                    continue;
                }
                let pm = ctx.lean(&src, Some(&src.with_extension("olean")), 900);
                let psz = std::fs::metadata(src.with_extension("olean")).ok().map(|x| x.len());
                rows.push(json!({
                    "kind": "program", "kinds": kinds, "index": Index::None.name(), "wf": wf.name(),
                    "leaves": 1u64 << d, "ok": pm.ok,
                    "secs": pm.secs, "rss_mb": pm.rss_mb,
                    "proof_secs": pm.secs.zip(bm.secs).map(|(a, b)| a - b),
                    "proof_bytes": psz.zip(bsz).map(|(a, b)| a as i64 - b as i64),
                    "err": err_of(&pm),
                }));
            }
        }

        // Indexed variant, at one kind count only: measured 1.87 / 1.46 / 1.45s at 256 leaves across 3 / 10 / 30 kinds, so the index cost is independent of how many constructors the DSL has.
        if kinds != kind_counts[0] {
            continue;
        }
        for (idx, file) in [(Index::ExtentMinimal, "ToyM.lean"), (Index::Extent, "ToyX.lean")] {
            let src = ctx.work.join(file);
            if dslgen::dsl(kinds, idx, WfStyle::Recursive, &src).is_err() {
                continue;
            }
            let m = ctx.lean(&src, Some(&src.with_extension("olean")), 900);
            rows.push(json!({
                "kind": "dsl", "kinds": kinds, "index": idx.name(),
                "ok": m.ok, "secs": m.secs, "rss_mb": m.rss_mb, "err": err_of(&m),
            }));
            if !m.ok {
                continue;
            }
            for &d in depths {
                let p = ctx.work.join(format!("shp_{}_{d}.lean", idx.name()));
                if dslgen::dsl_program(d, kinds, idx, &p).is_err() {
                    continue;
                }
                let pm = ctx.lean(&p, None, 900);
                rows.push(json!({
                    "kind": "program", "kinds": kinds, "index": idx.name(),
                    "leaves": 1u64 << d, "ok": pm.ok,
                    "secs": pm.secs, "rss_mb": pm.rss_mb, "err": err_of(&pm),
                }));
            }
        }
    }
    rows
}

//  frontends: how users might BUILD a DSL, not what it contains

pub fn frontends(ctx: &Ctx) -> Vec<Value> {
    use crate::frontend::{program, Frontend};

    let sizes = vec![25, 50, 100, 400, 800];

    let mut rows = Vec::new();
    for fe in Frontend::ALL {
        for &n in &sizes {
            if n > fe.max_n() {
                continue;
            }
            let src = ctx.work.join(format!("fe_{}_{n}.lean", fe.name()));
            if program(fe, n, &src).is_err() {
                continue;
            }
            let m = ctx.lean(&src, None, 600);
            rows.push(json!({
                "frontend": fe.name(), "nodes": n, "ok": m.ok,
                "secs": m.secs, "rss_mb": m.rss_mb,
                "err": err_of(&m),
            }));
            // a variant that has fallen over stays fallen; don't burn the budget.
            if !m.ok {
                break;
            }
        }
    }
    rows
}

pub fn smoke(ctx: &Ctx) -> Vec<Value> {
    use crate::dslgen::{self, Index, WfStyle};
    use crate::frontend::{program, Frontend};

    let mut rows = Vec::new();
    let fail = |rows: &mut Vec<Value>, what: String, m: crate::measure::Meas| {
        rows.push(json!({
            "variant": what, "ok": m.ok, "secs": m.secs,
            "rss_mb": m.rss_mb, "err": err_of(&m),
        }));
    };

    for kinds in [3usize, 10] {
        for wf in [WfStyle::Recursive, WfStyle::Inductive] {
            let p = ctx.work.join("ToyG.lean");
            if dslgen::dsl(kinds, Index::None, wf, &p).is_ok() {
                let m = ctx.lean(&p, Some(&p.with_extension("olean")), 300);
                let ok = m.ok;
                fail(&mut rows, format!("dsl/plain/{kinds}/{}", wf.name()), m);
                if ok {
                    let q = ctx.work.join("smoke_prog.lean");
                    if dslgen::dsl_program(2, kinds, Index::None, &q).is_ok() {
                        let m = ctx.lean(&q, None, 300);
                        fail(&mut rows, format!("prog/plain/{kinds}/{}", wf.name()), m);
                    }
                }
            }
        }
        let p = ctx.work.join("ToyX.lean");
        if dslgen::dsl(kinds, Index::Extent, WfStyle::Recursive, &p).is_ok() {
            let m = ctx.lean(&p, Some(&p.with_extension("olean")), 300);
            let ok = m.ok;
            fail(&mut rows, format!("dsl/extent/{kinds}"), m);
            if ok {
                let q = ctx.work.join("smoke_progx.lean");
                if dslgen::dsl_program(2, kinds, Index::Extent, &q).is_ok() {
                    let m = ctx.lean(&q, None, 300);
                    fail(&mut rows, format!("prog/extent/{kinds}"), m);
                }
            }
        }
    }
    for fe in Frontend::ALL {
        let p = ctx.work.join(format!("smoke_fe_{}.lean", fe.name()));
        if program(fe, 4, &p).is_ok() {
            let m = ctx.lean(&p, None, 300);
            fail(&mut rows, format!("frontend/{}", fe.name()), m);
        }
    }
    rows
}

/// Cost of calling through a stack of abstractions, with the same layers declared in every file.
pub fn depth(ctx: &Ctx) -> Vec<Value> {
    const LAYERS: usize = 64;
    let mut rows = Vec::new();
    for call in [0usize, 1, 16, LAYERS] {
        let src = ctx.work.join(format!("depth_{call}.lean"));
        if gen::depth(LAYERS, call, 32, &src).is_err() {
            continue;
        }
        let m = ctx.lean(&src, None, 300);
        rows.push(json!({
            "layers": LAYERS, "calls_layer": call, "ok": m.ok,
            "secs": m.secs, "rss_mb": m.rss_mb, "err": err_of(&m),
        }));
    }
    rows
}
