//! Print each measurement: what was varied, the numbers, what they mean.

use serde_json::Value;
use std::collections::BTreeMap;

type Raw = BTreeMap<String, Vec<Value>>;

fn f(r: &Value, k: &str) -> Option<f64> {
    r.get(k).and_then(|v| v.as_f64())
}
fn s<'a>(r: &'a Value, k: &str) -> Option<&'a str> {
    r.get(k).and_then(|v| v.as_str())
}
fn ok(r: &Value) -> bool {
    r.get("ok").and_then(|v| v.as_bool()).unwrap_or(false)
}
fn rows<'a>(raw: &'a Raw, suite: &str) -> &'a [Value] {
    raw.get(suite).map(|v| v.as_slice()).unwrap_or(&[])
}

fn n(v: f64) -> String {
    if v >= 1e6 {
        format!("{:.1}M", v / 1e6)
    } else if v >= 1e3 {
        format!("{:.0}k", v / 1e3)
    } else {
        format!("{v:.0}")
    }
}

fn bytes(v: f64) -> String {
    if v >= 1e6 {
        format!("{:.1}MB", v / 1e6)
    } else if v >= 1e3 {
        format!("{:.1}KB", v / 1e3)
    } else {
        format!("{v:.0}B")
    }
}

/// (size, value) pairs plus sizes that failed, sorted by size.
fn series(rs: &[Value], size: &str, val: &str, pick: impl Fn(&Value) -> bool)
    -> (Vec<(f64, f64)>, Vec<f64>)
{
    let (mut pts, mut bad) = (Vec::new(), Vec::new());
    for r in rs.iter().filter(|r| pick(r)) {
        if r.get("skipped").is_some() {
            continue;
        }
        let Some(x) = f(r, size) else { continue };
        match (ok(r), f(r, val)) {
            (true, Some(y)) => pts.push((x, y)),
            _ => bad.push(x),
        }
    }
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    bad.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (pts, bad)
}

fn growth(pts: &[(f64, f64)]) -> String {
    match (pts.first(), pts.last()) {
        (Some((x0, y0)), Some((x1, y1))) if *x0 > 0.0 && *y0 > 0.0 && x1 > x0 => {
            format!("{:.0}x -> {:.1}x", x1 / x0, y1 / y0)
        }
        _ => String::new(),
    }
}

fn line(label: &str, pts: &[(f64, f64)], bad: &[f64], fmt: fn(f64) -> String, note: &str) {
    let mut cells: Vec<String> =
        pts.iter().map(|(x, y)| format!("{}:{}", n(*x), fmt(*y))).collect();
    cells.extend(bad.iter().map(|x| format!("{}:FAILS", n(*x))));
    println!("  {:<13} {:<46} {:<14} {}", label, cells.join("  "), growth(pts), note);
}

pub fn all(raw: &Raw) {
    // proof term size against program size.
    let d = rows(raw, "derive");
    if !d.is_empty() {
        println!("\nproof term size vs program size (leaves)");
        for (label, v, note) in [
            ("reflective", "rfl", "applies a proven lemma"),
            ("native", "native", "same, checked by the compiler"),
            ("tactic", "tactic", "walks the AST -- the Fiat shape; stops at 1k"),
        ] {
            let (pts, bad) = series(d, "leaves", "proof_bytes", |r| s(r, "variant") == Some(v));
            line(label, &pts, &bad, bytes, note);
        }
    }

    // proof term size against how many node kinds the DSL has.
    let sh = rows(raw, "shapes");
    let (kinds, _) = series(sh, "kinds", "proof_bytes", |r| {
        ok(r) && s(r, "kind") == Some("program") && s(r, "index") == Some("plain")
    });
    if !kinds.is_empty() {
        println!("\nproof term size vs DSL size (node kinds, heterogeneous constructors)");
        let mut by: BTreeMap<i64, f64> = BTreeMap::new();
        for (k, b) in &kinds {
            by.insert(*k as i64, *b);
        }
        let cells: Vec<String> =
            by.iter().map(|(k, b)| format!("{k} kinds:{}", bytes(*b))).collect();
        println!("  {:<13} {}", "", cells.join("  "));
    }

    // cost of calling through a stack of abstractions.
    let dp = rows(raw, "depth");
    if !dp.is_empty() {
        let layers = dp.first().and_then(|r| f(r, "layers")).unwrap_or(0.0);
        let (pts, _) = series(dp, "calls_layer", "secs", |_| true);
        println!("\nelaboration time vs abstraction depth ({layers:.0} layers declared in each)");
        let cells: Vec<String> =
            pts.iter().map(|(d, t)| format!("{}:{:.2}s", n(*d), t)).collect();
        println!("  {:<13} {}", "calls layer", cells.join("  "));
    }

    // elaboration time against program size, per DSL encoding.
    let fe = rows(raw, "frontends");
    if !fe.is_empty() {
        println!("\nelaboration time vs program size (nodes), by DSL encoding");
        let mut names: Vec<&str> = fe.iter().filter_map(|r| s(r, "frontend")).collect();
        names.sort_unstable();
        names.dedup();
        let mut out: Vec<(f64, String, Vec<(f64, f64)>, Vec<f64>)> = Vec::new();
        for name in names {
            let (pts, bad) = series(fe, "nodes", "secs", |r| s(r, "frontend") == Some(name));
            let rate = match (pts.first(), pts.last()) {
                (Some((x0, y0)), Some((x1, y1))) if *x0 > 0.0 && *y0 > 0.0 => {
                    (y1 / y0) / (x1 / x0)
                }
                _ => 0.0,
            };
            out.push((rate, name.to_string(), pts, bad));
        }
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for (rate, name, pts, bad) in &out {
            let note = match name.as_str() {
                _ if *rate > 2.0 => "superlinear",
                // `import Lean` costs ~0.9s and swamps the program entirely,
                // so this row shows a floor rather than a trend
                "elab" => "import Lean dominates",
                _ => "",
            };
            line(name, pts, bad, |t| format!("{t:.2}s"), note);
        }
    }

    // monadic chains.
    let b = rows(raw, "binds");
    if !b.is_empty() {
        println!("\nelaboration time vs chain length (binds), by monad syntax");
        for style in ["do", "bind", "indexed"] {
            let (pts, bad) = series(b, "binds", "secs", |r| s(r, "style") == Some(style));
            line(style, &pts, &bad, |t| format!("{t:.2}s"), "");
        }
    }

    // generation and JIT.
    let c = rows(raw, "clif");
    if !c.is_empty() {
        println!("\nCLIF generation and JIT vs instruction count");
        // generation is timed against a ~1.4s import baseline, so the smallest
        // sizes land under measurement resolution and can come out negative
        let (mut pts, bad) = series(c, "insts", "gen_secs", |r| {
            s(r, "kind") == Some("generate") && s(r, "mode") == Some("dead")
        });
        pts.retain(|(_, t)| *t > 0.02);
        line("generate", &pts, &bad, |t| format!("{t:.2}s"), "");
        // dead and live CLIF have different per-byte costs; interleaving them
        // by size produces a growth figure that means nothing
        for (label, tag) in [("JIT dead", "dead"), ("JIT live", "live")] {
            let (pts, bad) = series(c, "bytes", "secs", |r| {
                s(r, "kind") == Some("jit")
                    && s(r, "file").is_some_and(|f| f.starts_with(tag))
            });
            line(label, &pts, &bad, |t| format!("{t:.2}s"), "bytes of CLIF");
        }
    }

    // indexed families.
    if !sh.is_empty() {
        println!("\nelaboration time vs program size (leaves), indexed vs not");
        for (label, idx) in [
            ("unindexed", "plain"),
            ("indexed", "extent_min"),
            ("indexed+DSL", "extent"),
        ] {
            let (pts, bad) = series(sh, "leaves", "secs", |r| {
                s(r, "kind") == Some("program")
                    && s(r, "index") == Some(idx)
                    && f(r, "kinds") == Some(3.0)
                    && s(r, "wf").map_or(true, |w| w == "recfn")
            });
            line(label, &pts, &bad, |t| format!("{t:.2}s"), "");
        }
    }
}
