
use std::fs;
use std::io::Write;
use std::path::Path;

fn subst(tpl: &str, pairs: &[(&str, String)]) -> String {
    let mut s = tpl.to_string();
    for (k, v) in pairs {
        s = s.replace(&format!("@{}@", k), v);
    }
    s
}

fn write(path: &Path, contents: &str) -> std::io::Result<()> {
    let mut f = fs::File::create(path)?;
    f.write_all(contents.as_bytes())
}

//  derive: proof-term size and check cost vs emitted AST size   `mkD` builds a BALANCED tree of DISTINCT leaves

const DERIVE_HDR: &str = r#"import ToyG
open ToyG
set_option maxRecDepth 8000000
set_option maxHeartbeats 2000000

-- `mkD` comes from the generated DSL: a balanced AST of 2^@DEPTH@ distinct
-- leaves, from a tiny source term.
def p : Stmt := mkD @DEPTH@ 0

"#;

#[derive(Clone, Copy, PartialEq)]
pub enum DeriveVariant {
    /// elaborate the program, prove nothing -- the baseline to subtract.
    Base,
    /// reflection, kernel-checked: minimal TCB, kernel reduces `wfCheck p`.
    Rfl,
    /// reflection, compiler-checked: flat cost, adds Lean's compiler to the TCB.
    Native,
    /// A derive that WALKS the AST with tactics instead of applying a pre-proven lemma -- the shape that forced Fiat Cryptography to rewrite its pipeline.
    Tactic,
}

impl DeriveVariant {
    pub fn name(self) -> &'static str {
        match self {
            DeriveVariant::Base => "base",
            DeriveVariant::Rfl => "rfl",
            DeriveVariant::Native => "native",
            DeriveVariant::Tactic => "tactic",
        }
    }
    fn body(self) -> &'static str {
        match self {
            DeriveVariant::Base => "",
            DeriveVariant::Rfl => concat!(
                "theorem ok : WF p := wfCheck_sound p rfl\n",
                "theorem d : ∀ st, run (emit p) st = denote p st := derived p ok\n"
            ),
            DeriveVariant::Native => concat!(
                "theorem ok : WF p := wfCheck_sound p (by native_decide)\n",
                "theorem d : ∀ st, run (emit p) st = denote p st := derived p ok\n"
            ),
            DeriveVariant::Tactic => concat!(
                "theorem ok : WF p := by\n",
                "  simp only [p, mkD, WF]\n",
                "  omega\n",
                "theorem d : ∀ st, run (emit p) st = denote p st := derived p ok\n"
            ),
        }
    }
}

pub fn derive(depth: u32, v: DeriveVariant, out: &Path) -> std::io::Result<()> {
    let hdr = subst(DERIVE_HDR, &[("DEPTH", depth.to_string())]);
    write(out, &(hdr + v.body()))
}

//  binds: monadic chain elaboration   The axis that actually bit this project

const BINDS_HDR: &str = "import Eff\nopen Eff\nset_option maxRecDepth 1000000\n\n";
const EFFS: [&str; 3] = [".gpu", ".file", ".net"];

#[derive(Clone, Copy, PartialEq)]
pub enum BindStyle {
    Do,
    /// explicit `PF.bind .
    Bind,
    /// effect set carried as a TYPE INDEX, unified at every bind.
    Indexed,
}

impl BindStyle {
    pub fn name(self) -> &'static str {
        match self {
            BindStyle::Do => "do",
            BindStyle::Bind => "bind",
            BindStyle::Indexed => "indexed",
        }
    }
}

pub fn binds(n: usize, style: BindStyle, out: &Path) -> std::io::Result<()> {
    let mut s = String::from(BINDS_HDR);
    match style {
        BindStyle::Do => {
            s.push_str("def chain : PF Nat := do\n");
            for i in 0..n {
                s.push_str(&format!("  let _x{} <- leafF {} {}\n", i, EFFS[i % 3], i));
            }
            s.push_str(&format!("  pure {}\n", n));
        }
        BindStyle::Bind => {
            s.push_str("def chain : PF Nat :=\n");
            for i in 0..n {
                s.push_str(&format!(
                    "  PF.bind (leafF {} {}) fun x{} =>\n",
                    EFFS[i % 3], i, i
                ));
            }
            s.push_str(&format!("  PF.pure {}\n", n));
        }
        BindStyle::Indexed => {
            s.push_str("def chain :=\n");
            for i in 0..n {
                s.push_str(&format!(
                    "  pbind (leafI {} {}) fun x{} =>\n",
                    EFFS[i % 3], i, i
                ));
            }
            s.push_str(&format!("  pureI {}\n", n));
        }
    }
    write(out, &s)
}

//  clif: generation throughput through the real IRBuilder

const CLIF_GEN: &str = r#"import AlgorithmLib
open AlgorithmLib.IR

/-- dead values: trivial register allocation downstream -/
def deadProgram (n : Nat) : String := buildProgram do
  let _ ← entryBlock
  for _ in [0:n] do
    let _ ← iconst64 42
  ret

/-- k simultaneously-live values combined in a chain: real interference -/
def liveProgram (n k : Nat) : String := buildProgram do
  let _ ← entryBlock
  let mut acc : Array Val := #[]
  for i in [0:k] do
    acc := acc.push (← iconst64 (Int.ofNat i))
  let mut cur ← iconst64 1
  for i in [0:n] do
    cur ← iadd cur (acc.getD (i % k) cur)
  ret

def main (args : List String) : IO Unit := do
  let mode := args[0]!
  let n := (args[1]!).toNat!
  let s := if mode == "dead" then deadProgram n else liveProgram n 64
  -- annotate: without it, `IO.FS.writeFile` wanting a FilePath makes Lean try
  -- to elaborate the index as `GetElem? (List String) Nat System.FilePath`
  let out : Option String := args[2]?
  match out with
  | some path => IO.FS.writeFile path s
  | none => pure ()
  IO.println s!"{n}\t{s.utf8ByteSize}"
"#;

pub fn clif_gen(out: &Path) -> std::io::Result<()> {
    write(out, CLIF_GEN)
}

const EFF: &str = r#"/-
  Does putting the effect set in the *type* cost anything at scale?

    PI ε α  -- effect set as a type index, unified at every bind
    PF α    -- plain monad; effects computed post-hoc from the built AST
-/
namespace Eff

inductive E | gpu | file | net
  deriving DecidableEq, Repr

abbrev ESet := List E

-- ── variant IDX: effect set in the type ──────────────────────────────────────
structure PI (ε : ESet) (α : Type) where
  run : Nat → α × Nat

def pbind (x : PI ε₁ α) (f : α → PI ε₂ β) : PI (ε₁ ++ ε₂) β :=
  ⟨fun s => let r := x.run s; (f r.1).run r.2⟩

def leafI (e : E) (v : Nat) : PI [e] Nat := ⟨fun s => (v, s + v)⟩
def pureI (v : Nat) : PI ([] : ESet) Nat := ⟨fun s => (v, s)⟩

-- ── variant FLD: plain monad, effects are data ───────────────────────────────
structure PF (α : Type) where
  run : Nat → α × Nat

def PF.bind (x : PF α) (f : α → PF β) : PF β :=
  ⟨fun s => let r := x.run s; (f r.1).run r.2⟩
def PF.pure (v : α) : PF α := ⟨fun s => (v, s)⟩

instance : Monad PF where
  bind := PF.bind
  pure := PF.pure

def leafF (e : E) (v : Nat) : PF Nat := ⟨fun s => (v, s + v)⟩

end Eff
"#;

pub fn eff(out: &Path) -> std::io::Result<()> {
    write(out, EFF)
}

pub fn depth(layers: usize, call: usize, uses: usize, out: &Path) -> std::io::Result<()> {
    let mut s = String::from(
        "set_option maxRecDepth 8000000\nset_option maxHeartbeats 4000000\n\
         inductive Instr where | set (d : String) (v : Nat)\n\
         def op0 (v : Nat) : List Instr := [.set \"r\" v]\n\
         theorem op0_len (v : Nat) : (op0 v).length = 1 := rfl\n",
    );
    for k in 1..=layers {
        s.push_str(&format!(
            "def op{k} (v : Nat) : List Instr := op{} (v+1)\n\
             theorem op{k}_len (v : Nat) : (op{k} v).length = 1 := op{}_len _\n",
            k - 1, k - 1
        ));
    }
    fn bal(lo: usize, hi: usize, d: usize) -> String {
        if hi - lo == 1 {
            format!("(op{d} {lo})")
        } else {
            let m = lo + (hi - lo) / 2;
            format!("({} ++ {})", bal(lo, m, d), bal(m, hi, d))
        }
    }
    s.push_str(&format!("\ndef p : List Instr := {}\n", bal(0, uses, call)));
    s.push_str(&format!(
        "theorem p_len : p.length = {uses} := by \
         simp only [p, List.length_append, op{call}_len]\n"
    ));
    write(out, &s)
}
