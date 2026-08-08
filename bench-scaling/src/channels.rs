//! Where per-declaration proof cost lands, and what makes it superlinear.
//!
//! `regimes` is an axis the other suites do not reach: an obligation over an
//! n-step structure, discharged two ways twice over.  One of each pair is
//! linear and the other is not, and the difference is a property of how the
//! obligation was written, not of what it guarantees.

use std::fs;
use std::io::Write;
use std::path::Path;

fn write(path: &Path, contents: &str) -> std::io::Result<()> {
    let mut f = fs::File::create(path)?;
    f.write_all(contents.as_bytes())
}

/// A state and a step that advances it.  `step_n` is the opaque statement — it
/// says what a step does to `n` without unfolding its argument, which is the
/// whole point of the `named` variant below.
pub const BASE: &str = r#"structure St where
  v : Nat
  n : Nat

def step (k : Nat) (s : St) : St := { v := s.v + k, n := s.n + 1 }
theorem step_n (k : Nat) (s : St) : (step k s).n = s.n + 1 := rfl

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
}

impl Regime {
    pub fn name(self) -> &'static str {
        match self {
            Regime::Reflect => "reflect",
            Regime::ReflectQuad => "reflect-quad",
            Regime::Named => "named",
            Regime::Inlined => "inlined",
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
