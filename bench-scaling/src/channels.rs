//! Where per-declaration proof cost lands, and what makes it superlinear.
//!
//! Two axes the other suites do not reach:
//!
//! * `regimes` — three pairs, each proving the same thing twice.  One of each
//!   pair is linear and the other is not, and the difference is a property of
//!   how the obligation was written, not of what it guarantees.
//! * `compose` — K separately-proven components chained in a user module.  The
//!   row that matters is composite cost against component *internal* size: if
//!   specs are opaque it is flat, and if they leak it is not.

use std::fs;
use std::io::Write;
use std::path::Path;

fn write(path: &Path, contents: &str) -> std::io::Result<()> {
    let mut f = fs::File::create(path)?;
    f.write_all(contents.as_bytes())
}

/// Shared by both suites: a state, a step that advances it, and a `bump` that
/// does not.  `step_n`/`bump_n` are the opaque statements — they say what a
/// step does to `n` without unfolding its argument, which is the whole point.
pub const BASE: &str = r#"structure St where
  v : Nat
  n : Nat

def step (k : Nat) (s : St) : St := { v := s.v + k, n := s.n + 1 }
theorem step_n (k : Nat) (s : St) : (step k s).n = s.n + 1 := rfl

def bump (s : St) : St := { s with v := s.v + 1 }
theorem bump_n (s : St) : (bump s).n = s.n := rfl

def s0 : St := { v := 0, n := 0 }
theorem h0 : s0.n = 0 := rfl
"#;

pub fn base(out: &Path) -> std::io::Result<()> {
    write(out, BASE)
}

// ── regimes ─────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
pub enum Regime {
    /// one decidable check over the whole structure, linear checker.
    Reflect,
    /// the same check with a checker that indexes instead of traversing.  The
    /// obligation is identical; only the algorithm behind it changed.
    ReflectQuad,
    /// one lemma application per step, with intermediates NAMED so each goal
    /// mentions the previous state abstractly and stays O(1).
    Named,
    /// one lemma application per step, with the prefix INLINED so statement k
    /// carries k steps of accumulated structure.
    Inlined,
    /// a property RELATING every element to every other -- n regions are
    /// pairwise disjoint -- checked the way it is stated.
    PairsAll,
    /// the same property, checked adjacent-only.  Sound because the generator
    /// emits regions in address order, which the check also verifies: the
    /// witness costs nothing because whoever built the structure knew it.
    PairsSorted,
}

impl Regime {
    pub fn name(self) -> &'static str {
        match self {
            Regime::Reflect => "reflect",
            Regime::ReflectQuad => "reflect-quad",
            Regime::Named => "named",
            Regime::Inlined => "inlined",
            Regime::PairsAll => "disjoint-pairs",
            Regime::PairsSorted => "disjoint-sorted",
        }
    }
}

pub fn regimes(n: usize, r: Regime, out: &Path) -> std::io::Result<()> {
    let mut s = String::from("import ChBase\nset_option maxRecDepth 1000000\n\n");
    match r {
        // the same literal list in both, so it does not skew the pair.
        Regime::Reflect | Regime::ReflectQuad => {
            s.push_str("def prog : List Nat := [");
            for i in 0..n {
                if i > 0 {
                    s.push_str(", ");
                }
                s.push_str(&i.to_string());
            }
            s.push_str("]\n\n");
            match r {
                Regime::Reflect => s.push_str(concat!(
                    "-- traverses the list once\n",
                    "def chk : List Nat -> Bool\n",
                    "  | [] => true\n",
                    "  | x :: xs => decide (x < 100000) && chk xs\n\n",
                    "theorem t : chk prog = true := rfl\n",
                )),
                Regime::ReflectQuad => s.push_str(concat!(
                    "-- `getD i` walks i cells, so the same check is O(n^2)\n",
                    "def chkQ (p : List Nat) : Bool :=\n",
                    "  (List.range p.length).all (fun i => decide (p.getD i 0 < 100000))\n\n",
                    "theorem t : chkQ prog = true := rfl\n",
                )),
                _ => {}
            }
        }
        Regime::Named => {
            for k in 1..=n {
                s.push_str(&format!(
                    "def s{k} : St := step {k} s{}\n\
                     theorem h{k} : s{k}.n = {k} := by unfold s{k}; rw [step_n, h{}]\n",
                    k - 1,
                    k - 1
                ));
            }
        }
        // n regions, laid out end to end, as a generator would emit them.
        Regime::PairsAll | Regime::PairsSorted => {
            s.push_str("def regs : List (Nat × Nat) := [");
            for i in 0..n {
                if i > 0 {
                    s.push_str(", ");
                }
                s.push_str(&format!("({}, {})", i * 16, i * 16 + 16));
            }
            s.push_str("]\n\n");
            if r == Regime::PairsAll {
                s.push_str(concat!(
                    "-- every region against every other\n",
                    "def disjAll (l : List (Nat × Nat)) : Bool :=\n",
                    "  l.all (fun a => l.all (fun b =>\n",
                    "    decide (a.1 = b.1) || decide (a.2 <= b.1) || decide (b.2 <= a.1)))\n\n",
                    "theorem t : disjAll regs = true := rfl\n",
                ));
            } else {
                s.push_str(concat!(
                    "-- neighbours only; the `<=` chain also establishes the order\n",
                    "-- that makes adjacent-disjointness imply pairwise-disjointness\n",
                    "def disjAdj : List (Nat × Nat) -> Bool\n",
                    "  | [] => true\n",
                    "  | [_] => true\n",
                    "  | a :: b :: r => decide (a.2 <= b.1) && disjAdj (b :: r)\n\n",
                    "theorem t : disjAdj regs = true := rfl\n",
                ));
            }
        }
        Regime::Inlined => {
            let mut term = String::from("s0");
            for k in 1..=n {
                term = format!("(step {k} {term})");
                s.push_str(&format!(
                    "theorem t{k} : {term}.n = {k} := by rw [step_n, {}]\n",
                    if k == 1 { "h0".to_string() } else { format!("t{}", k - 1) }
                ));
            }
        }
    }
    write(out, &s)
}

// ── compose ─────────────────────────────────────────────────────────────────

/// A component: a body of `w` internal operations, and a spec.
///
/// `leaky = false` states the spec in the shared vocabulary (`.n` after the
/// call), so the body is invisible to callers.  `leaky = true` states it as an
/// equation on the body, so every caller inherits `w` nodes of it.
pub fn component(k: usize, w: usize, leaky: bool, out: &Path) -> std::io::Result<()> {
    let mut body = String::from("s");
    for _ in 0..w {
        body = format!("(bump {body})");
    }
    let nm = if leaky { format!("g{k}") } else { format!("f{k}") };
    let mut s = String::from("import ChBase\nset_option maxRecDepth 1000000\n\n");
    s.push_str(&format!("def {nm} (s : St) : St := step 1 {body}\n\n"));
    if leaky {
        s.push_str(&format!(
            "theorem {nm}_spec (s : St) : {nm} s = step 1 {body} := rfl\n"
        ));
    } else {
        s.push_str(&format!(
            "theorem {nm}_spec (s : St) : ({nm} s).n = s.n + 1 := by\n  \
             unfold {nm}; rw [step_n]; simp only [bump_n]\n"
        ));
    }
    write(out, &s)
}

/// Chain `k` components and prove the composite, using only their statements.
///
/// `prove = false` emits the same imports and the same `pipe` and stops, so
/// subtracting it leaves the cost of the composite proof alone -- importing K
/// oleans is not free and grows with K.
pub fn composite(k: usize, w: usize, leaky: bool, prove: bool, out: &Path) -> std::io::Result<()> {
    let (p, nm) = if leaky { ("CmpL", "g") } else { ("Cmp", "f") };
    let mut s = String::new();
    for i in 0..k {
        s.push_str(&format!("import {p}{w}_{i}\n"));
    }
    s.push_str("set_option maxRecDepth 1000000\n\n");

    let mut call = String::from("s");
    for i in 0..k {
        call = format!("{nm}{i} {call}");
        if i + 1 < k {
            call = format!("({call})");
        }
    }
    s.push_str(&format!("def pipe (s : St) : St := {call}\n\n"));
    if prove {
        let specs: Vec<String> = (0..k).map(|i| format!("{nm}{i}_spec")).collect();
        s.push_str(&format!(
            "theorem pipe_spec (s : St) : (pipe s).n = s.n + {k} := by\n  \
             unfold pipe\n  \
             simp only [{}{}]\n  \
             all_goals omega\n",
            specs.join(", "),
            if leaky { ", step_n, bump_n" } else { "" }
        ));
    }
    write(out, &s)
}
