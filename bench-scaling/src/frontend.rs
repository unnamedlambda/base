
//! Ten ways to encode the same DSL, to see which survive a large program.

use std::fs;
use std::io::Write;
use std::path::Path;

#[derive(Clone, Copy, PartialEq)]
pub enum Frontend {
    /// plain constructor application -- the baseline for the others
    Ctor,
    /// `macro_rules` over a custom syntax category
    Macro,
    /// nodes as separate types dispatched by a class, so the program's TYPE
    /// grows with the program
    Typeclass,
    /// recursion on a measure rather than a subterm
    WellFounded,
    /// mutually recursive inductives and functions -- an IR with expressions
    /// and statements is this shape
    Mutual,
    /// `Coe` firing at every node
    Coe,
    /// a custom `elab` in `TermElabM` rather than syntactic expansion
    Elab,
    /// `partial def`, no termination obligation
    Partial,
    /// a `class ... extends` diamond over nested types
    Extends,
    /// all the recommended patterns at once: macro syntax over one inductive,
    /// a record of ops rather than a class, and a do-notation builder
    Idiomatic,
}

impl Frontend {
    pub fn name(self) -> &'static str {
        match self {
            Frontend::Ctor => "ctor",
            Frontend::Macro => "macro",
            Frontend::Typeclass => "typeclass",
            Frontend::WellFounded => "wellfounded",
            Frontend::Mutual => "mutual",
            Frontend::Coe => "coe",
            Frontend::Elab => "elab",
            Frontend::Partial => "partial",
            Frontend::Extends => "extends",
            Frontend::Idiomatic => "idiomatic",
        }
    }

    pub const ALL: [Frontend; 10] = [
        Frontend::Ctor, Frontend::Macro, Frontend::Typeclass, Frontend::WellFounded,
        Frontend::Mutual, Frontend::Coe, Frontend::Elab, Frontend::Partial,
        Frontend::Extends, Frontend::Idiomatic,
    ];

    pub fn max_n(self) -> usize {
        match self {
            // past 200 it is unreachable at any synthInstance setting, and 200 alone costs 43s, so the sweep stops below it.
            Frontend::Typeclass => 100,
            Frontend::Extends => 100,
            Frontend::Macro | Frontend::Elab | Frontend::Idiomatic => 4000,
            _ => 16000,
        }
    }
}

const BASE: &str = r#"set_option maxRecDepth 8000000
set_option maxHeartbeats 4000000

inductive Instr where
  | set (d : String) (v : Nat)
  deriving Repr

inductive Stmt where
  | leaf (a b : Nat)
  | seq (s t : Stmt)

def emit : Stmt → List Instr
  | .leaf a b => [.set "r" (a + b)]
  | .seq s t => emit s ++ emit t

"#;

fn write(path: &Path, s: &str) -> std::io::Result<()> {
    fs::File::create(path)?.write_all(s.as_bytes())
}

/// Balanced nesting keeps recursion depth logarithmic, so a failure is a scaling result rather than a stack limit.
fn balanced(lo: usize, hi: usize, leaf: &dyn Fn(usize) -> String, node: &dyn Fn(&str, &str) -> String) -> String {
    if hi - lo == 1 {
        leaf(lo)
    } else {
        let mid = lo + (hi - lo) / 2;
        let l = balanced(lo, mid, leaf, node);
        let r = balanced(mid, hi, leaf, node);
        node(&l, &r)
    }
}

pub fn program(fe: Frontend, n: usize, out: &Path) -> std::io::Result<()> {
    let src = match fe {
        Frontend::Ctor => {
            let body = balanced(
                0,
                n,
                &|i| format!("(.leaf {} {})", i % 100, (i * 7) % 100),
                &|l, r| format!("(.seq {l} {r})"),
            );
            format!("{BASE}def p : Stmt :=\n  {body}\n\ndef n : Nat := (emit p).length\n")
        }

        Frontend::Macro => {
            let body = balanced(
                0,
                n,
                &|i| format!("(leaf {} {})", i % 100, (i * 7) % 100),
                &|l, r| format!("(seq {l} {r})"),
            );
            format!(
                "{BASE}\
                 declare_syntax_cat bstmt\n\
                 syntax \"leaf\" num num : bstmt\n\
                 syntax \"seq\" bstmt bstmt : bstmt\n\
                 syntax \"(\" bstmt \")\" : bstmt\n\
                 syntax \"[s|\" bstmt \"]\" : term\n\n\
                 macro_rules\n\
                 \x20 | `([s| ($x:bstmt)]) => `([s| $x])\n\
                 \x20 | `([s| leaf $a:num $b:num]) => `(Stmt.leaf $a $b)\n\
                 \x20 | `([s| seq $x:bstmt $y:bstmt]) => `(Stmt.seq [s| $x] [s| $y])\n\n\
                 def p : Stmt := [s| {body}]\n\n\
                 def n : Nat := (emit p).length\n"
            )
        }

        Frontend::Typeclass => {
            let ty = balanced(
                0,
                n,
                &|_| "Leaf".to_string(),
                &|l, r| format!("(Pair {l} {r})"),
            );
            let val = balanced(
                0,
                n,
                &|i| format!("(Leaf.mk {} {})", i % 100, (i * 7) % 100),
                &|l, r| format!("(Pair.mk {l} {r})"),
            );
            format!(
                "set_option maxRecDepth 8000000\n\
                 set_option maxHeartbeats 4000000\n\
                 set_option synthInstance.maxSize 20000\n\
                 set_option synthInstance.maxHeartbeats 2000000\n\n\
                 inductive Instr where\n  | set (d : String) (v : Nat)\n\n\
                 class Node (a : Type) where\n  emit : a → List Instr\n\n\
                 structure Leaf where\n  a : Nat\n  b : Nat\n\n\
                 instance : Node Leaf := ⟨fun l => [.set \"r\" (l.a + l.b)]⟩\n\n\
                 structure Pair (X Y : Type) where\n  s : X\n  t : Y\n\n\
                 instance [Node X] [Node Y] : Node (Pair X Y) :=\n\
                 \x20 ⟨fun p => Node.emit p.s ++ Node.emit p.t⟩\n\n\
                 abbrev P : Type := {ty}\n\n\
                 def p : P := {val}\n\n\
                 def n : Nat := (Node.emit p).length\n"
            )
        }

        Frontend::WellFounded => {
            let body = balanced(
                0,
                n,
                &|i| format!("(.leaf {} {})", i % 100, (i * 7) % 100),
                &|l, r| format!("(.seq {l} {r})"),
            );
            format!(
                "{BASE}\
                 def size : Stmt → Nat\n\
                 \x20 | .leaf _ _ => 1\n\
                 \x20 | .seq s t => size s + size t + 1\n\n\
                 theorem size_lt_seq_left (s t : Stmt) : size s < size (.seq s t) := by\n\
                 \x20 simp only [size]; omega\n\n\
                 theorem size_lt_seq_right (s t : Stmt) : size t < size (.seq s t) := by\n\
                 \x20 simp only [size]; omega\n\n\
                 def emitWF (s : Stmt) : List Instr :=\n\
                 \x20 match s with\n\
                 \x20 | .leaf a b => [.set \"r\" (a + b)]\n\
                 \x20 | .seq x y =>\n\
                 \x20     have := size_lt_seq_left x y\n\
                 \x20     have := size_lt_seq_right x y\n\
                 \x20     emitWF x ++ emitWF y\n\
                 termination_by size s\n\n\
                 def p : Stmt :=\n  {body}\n\n\
                 def n : Nat := (emitWF p).length\n"
            )
        }

        Frontend::Mutual => {
            let body = balanced(
                0, n,
                &|i| format!("(.set (.lit {}))", i % 100),
                &|l, r| format!("(.seq {l} {r})"),
            );
            format!(
                "set_option maxRecDepth 8000000\n\
                 set_option maxHeartbeats 4000000\n\n\
                 inductive Instr where\n  | set (d : String) (v : Nat)\n\n\
                 mutual\n\
                 inductive Exp where\n\
                 \x20 | lit (n : Nat)\n\
                 \x20 | ofStmt (s : Stm)\n\
                 inductive Stm where\n\
                 \x20 | set (e : Exp)\n\
                 \x20 | seq (a b : Stm)\n\
                 end\n\n\
                 mutual\n\
                 def emitE : Exp -> List Instr\n\
                 \x20 | .lit v => [.set \"r\" v]\n\
                 \x20 | .ofStmt s => emitS s\n\
                 def emitS : Stm -> List Instr\n\
                 \x20 | .set e => emitE e\n\
                 \x20 | .seq a b => emitS a ++ emitS b\n\
                 end\n\n\
                 def p : Stm :=\n  {body}\n\n\
                 def n : Nat := (emitS p).length\n"
            )
        }

        Frontend::Coe => {
            let body = balanced(
                0, n,
                &|i| format!("(.leaf (Idx.mk {}) (Idx.mk {}))", i % 100, (i * 7) % 100),
                &|l, r| format!("(.seq {l} {r})"),
            );
            format!(
                "set_option maxRecDepth 8000000\n\
                 set_option maxHeartbeats 4000000\n\n\
                 inductive Instr where\n  | set (d : String) (v : Nat)\n\n\
                 structure Reg where\n  id : Nat\n\
                 structure Idx where\n  v : Nat\n\n\
                 -- from a struct, not Nat: numerals go through OfNat, not Coe,\n\
                 -- so a Nat source would never exercise coercion at all\n\
                 instance : Coe Idx Reg := ⟨fun i => ⟨i.v⟩⟩\n\n\
                 inductive Stmt where\n\
                 \x20 | leaf (a b : Reg)\n\
                 \x20 | seq (s t : Stmt)\n\n\
                 def emit : Stmt -> List Instr\n\
                 \x20 | .leaf a b => [.set \"r\" (a.id + b.id)]\n\
                 \x20 | .seq s t => emit s ++ emit t\n\n\
                 def p : Stmt :=\n  {body}\n\n\
                 def n : Nat := (emit p).length\n"
            )
        }

        Frontend::Elab => {
            let body = balanced(
                0, n,
                &|i| format!("(leaf {} {})", i % 100, (i * 7) % 100),
                &|l, r| format!("(seq {l} {r})"),
            );
            format!(
                "import Lean\n\
                 set_option maxRecDepth 8000000\n\
                 set_option maxHeartbeats 4000000\n\
                 open Lean Meta Elab Term\n\n\
                 inductive Instr where\n  | set (d : String) (v : Nat)\n\n\
                 inductive Stmt where\n\
                 \x20 | leaf (a b : Nat)\n\
                 \x20 | seq (s t : Stmt)\n\n\
                 def emit : Stmt -> List Instr\n\
                 \x20 | .leaf a b => [.set \"r\" (a + b)]\n\
                 \x20 | .seq s t => emit s ++ emit t\n\n\
                 declare_syntax_cat estmt\n\
                 syntax \"leaf\" num num : estmt\n\
                 syntax \"seq\" estmt estmt : estmt\n\
                 syntax \"(\" estmt \")\" : estmt\n\
                 syntax \"[e|\" estmt \"]\" : term\n\n\
                 partial def toStmt : Syntax -> TermElabM Expr\n\
                 \x20 | `(estmt| leaf $a:num $b:num) => do\n\
                 \x20     mkAppM ``Stmt.leaf #[mkNatLit a.getNat, mkNatLit b.getNat]\n\
                 \x20 | `(estmt| seq $x:estmt $y:estmt) => do\n\
                 \x20     mkAppM ``Stmt.seq #[<- toStmt x, <- toStmt y]\n\
                 \x20 | `(estmt| ($x:estmt)) => toStmt x\n\
                 \x20 | _ => throwUnsupportedSyntax\n\n\
                 elab_rules : term\n\
                 \x20 | `([e| $x:estmt]) => toStmt x\n\n\
                 def p : Stmt := [e| {body}]\n\n\
                 def n : Nat := (emit p).length\n"
            )
        }

        Frontend::Partial => {
            let body = balanced(
                0, n,
                &|i| format!("(.leaf {} {})", i % 100, (i * 7) % 100),
                &|l, r| format!("(.seq {l} {r})"),
            );
            format!(
                "{BASE}\
                 partial def emitP : Stmt -> List Instr\n\
                 \x20 | .leaf a b => [.set \"r\" (a + b)]\n\
                 \x20 | .seq s t => emitP s ++ emitP t\n\n\
                 def p : Stmt :=\n  {body}\n\n\
                 def n : Nat := (emitP p).length\n"
            )
        }

        Frontend::Extends => {
            let ty = balanced(0, n, &|_| "Leaf".to_string(), &|l, r| format!("(Pair {l} {r})"));
            let val = balanced(
                0, n,
                &|i| format!("(Leaf.mk {} {})", i % 100, (i * 7) % 100),
                &|l, r| format!("(Pair.mk {l} {r})"),
            );
            format!(
                "set_option maxRecDepth 8000000\n\
                 set_option maxHeartbeats 4000000\n\
                 set_option synthInstance.maxSize 20000\n\
                 set_option synthInstance.maxHeartbeats 2000000\n\n\
                 inductive Instr where\n  | set (d : String) (v : Nat)\n\n\
                 class Sized (a : Type) where\n  size : a -> Nat\n\
                 class Emits (a : Type) extends Sized a where\n  emit : a -> List Instr\n\
                 class Named (a : Type) extends Sized a where\n  name : a -> String\n\
                 class Node (a : Type) extends Emits a, Named a\n\n\
                 structure Leaf where\n  a : Nat\n  b : Nat\n\n\
                 instance : Sized Leaf := ⟨fun _ => 1⟩\n\
                 instance : Emits Leaf := ⟨fun l => [.set \"r\" (l.a + l.b)]⟩\n\
                 instance : Named Leaf := ⟨fun _ => \"leaf\"⟩\n\
                 instance : Node Leaf where\n\n\
                 structure Pair (X Y : Type) where\n  s : X\n  t : Y\n\n\
                 instance [Node X] [Node Y] : Sized (Pair X Y) :=\n\
                 \x20 ⟨fun p => Sized.size p.s + Sized.size p.t⟩\n\
                 instance [Node X] [Node Y] : Emits (Pair X Y) :=\n\
                 \x20 ⟨fun p => Emits.emit p.s ++ Emits.emit p.t⟩\n\
                 instance [Node X] [Node Y] : Named (Pair X Y) := ⟨fun _ => \"pair\"⟩\n\
                 instance [Node X] [Node Y] : Node (Pair X Y) where\n\n\
                 abbrev P : Type := {ty}\n\n\
                 def p : P := {val}\n\n\
                 def n : Nat := (Emits.emit p).length\n"
            )
        }

        Frontend::Idiomatic => {
            let body = balanced(
                0, n,
                &|i| format!("(op {} {})", i % 4, i % 100),
                &|l, r| format!("(both {l} {r})"),
            );
            format!(
                "set_option maxRecDepth 8000000\n\
                 set_option maxHeartbeats 4000000\n\n\
                 inductive Instr where\n  | set (d : String) (v : Nat)\n\n\
                 /-- extensibility as a VALUE, not an instance -/\n\
                 structure Op where\n\
                 \x20 name : String\n\
                 \x20 emit : Nat -> List Instr\n\n\
                 def table : Array Op := #[\n\
                 \x20 ⟨\"add\", fun v => [.set \"r\" v]⟩,\n\
                 \x20 ⟨\"mul\", fun v => [.set \"m\" v]⟩,\n\
                 \x20 ⟨\"shl\", fun v => [.set \"s\" v]⟩,\n\
                 \x20 ⟨\"nop\", fun _ => []⟩ ]\n\n\
                 /-- one inductive for the AST -/\n\
                 inductive Stmt where\n\
                 \x20 | op (k v : Nat)\n\
                 \x20 | both (s t : Stmt)\n\n\
                 def emit : Stmt -> List Instr\n\
                 \x20 | .op k v => ((table.getD (k % 4) ⟨\"nop\", fun _ => []⟩).emit v)\n\
                 \x20 | .both s t => emit s ++ emit t\n\n\
                 /-- macro surface syntax -/\n\
                 declare_syntax_cat istmt\n\
                 syntax \"op\" num num : istmt\n\
                 syntax \"both\" istmt istmt : istmt\n\
                 syntax \"(\" istmt \")\" : istmt\n\
                 syntax \"[i|\" istmt \"]\" : term\n\n\
                 macro_rules\n\
                 \x20 | `([i| ($x:istmt)]) => `([i| $x])\n\
                 \x20 | `([i| op $k:num $v:num]) => `(Stmt.op $k $v)\n\
                 \x20 | `([i| both $x:istmt $y:istmt]) => `(Stmt.both [i| $x] [i| $y])\n\n\
                 def p : Stmt := [i| {body}]\n\n\
                 /-- a do-notation builder over the result -/\n\
                 def summarise : StateM Nat Unit := do\n\
                 \x20 for i in (emit p) do\n\
                 \x20   match i with\n\
                 \x20   | .set _ v => modify (· + v)\n\n\
                 def n : Nat := (summarise.run 0).2\n"
            )
        }
    };
    write(out, &src)
}
