//! Generates the DSL itself, not just programs in it: node-kind count, index
//! style, and side-condition style all vary.

use std::fs;
use std::io::Write;
use std::path::Path;

#[derive(Clone, Copy, PartialEq)]
pub enum Index {
    /// plain inductive type.
    None,
    /// `Stmt : Nat -> Type`, index computed at each constructor.
    Extent,
    /// the same indexed family with nothing else in it, to separate the cost of
    /// indexing from whatever else the fuller DSL brings
    ExtentMinimal,
}

impl Index {
    pub fn name(self) -> &'static str {
        match self {
            Index::None => "plain",
            Index::Extent => "extent",
            Index::ExtentMinimal => "extent_min",
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum WfStyle {
    /// `def WF : Stmt -> Prop` by structural recursion.
    Recursive,
    /// `inductive WF : Stmt -> Prop`.
    Inductive,
}

impl WfStyle {
    pub fn name(self) -> &'static str {
        match self {
            WfStyle::Recursive => "recfn",
            WfStyle::Inductive => "indfam",
        }
    }
}

fn write(path: &Path, s: &str) -> std::io::Result<()> {
    fs::File::create(path)?.write_all(s.as_bytes())
}

fn leaves_for(kinds: usize) -> usize {
    kinds.saturating_sub(2).max(1)
}

/// Leaf constructors are deliberately NOT uniform: real DSL constructs differ in
/// arity, argument type and side condition, so `emit_correct` is a genuinely
/// different case each time rather than the same induction repeated.
struct LeafShape {
    /// constructor binders, such as `(a b : Nat)`
    binders: &'static str,
    /// the emitted value, in terms of the binders.
    value: &'static str,
    /// the well-formedness side condition.
    wf: &'static str,
}

const SHAPES: [LeafShape; 4] = [
    LeafShape { binders: "(a b : Nat)", value: "a + b", wf: "a + b < 256" },
    LeafShape { binders: "(a : Nat) (bs : List Nat)", value: "a + bs.length",
                wf: "a + bs.length < 256" },
    LeafShape { binders: "(a b c : Nat)", value: "a + b + c", wf: "a + b + c < 256" },
    LeafShape { binders: "(s : String) (a : Nat)", value: "s.length + a",
                wf: "s.length + a < 256" },
];

impl LeafShape {
    /// the binder names alone, for use as a match pattern.
    fn bind_pat(&self) -> String {
        self.binders
            .split(')')
            .filter_map(|g| g.split('(').nth(1))
            .filter_map(|g| g.split(':').next())
            .map(|names| names.trim().to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// the same binders made implicit, for an inductive-family constructor.
    fn implicit_binders(&self) -> String {
        self.binders.replace('(', "{").replace(')', "}")
    }

    /// concrete arguments for this shape, all satisfying its side condition.
    fn args(&self, i: &str) -> String {
        match self.binders {
            "(a b : Nat)" => format!("({i} % 100) (({i}*7) % 100)"),
            "(a : Nat) (bs : List Nat)" => format!("({i} % 100) [1, 2]"),
            "(a b c : Nat)" => format!("({i} % 100) (({i}*7) % 100) 1"),
            _ => format!("\"x\" ({i} % 100)"),
        }
    }
}

fn shape(i: usize) -> &'static LeafShape {
    &SHAPES[i % SHAPES.len()]
}

fn plain(kinds: usize, wf: WfStyle) -> String {
    let n = leaves_for(kinds);
    let mut s = String::new();

    s.push_str("set_option maxRecDepth 8000000\nset_option maxHeartbeats 2000000\n");
    s.push_str("namespace ToyG\n\n");

    // source DSL
    s.push_str("inductive Stmt where\n");
    for i in 0..n {
        s.push_str(&format!("  | leaf{i} (d : String) {}\n", shape(i).binders));
    }
    s.push_str("  | seq (s t : Stmt)\n  | loop (n : Nat) (body : Stmt)\n\n");

    // machine
    s.push_str(
        "structure St where\n  regs : String → Nat\n\n\
         inductive Instr where\n  | set (d : String) (v : Nat)\n\n\
         def runI (i : Instr) (st : St) : St :=\n\
         \x20 match i with\n\
         \x20 | .set d v => ⟨fun x => if x = d then v % 256 else st.regs x⟩\n\n\
         def run : List Instr → St → St\n\
         \x20 | [], st => st\n\
         \x20 | i :: is, st => run is (runI i st)\n\n\
         theorem run_append : ∀ (l1 l2 : List Instr) (st : St),\n\
         \x20   run (l1 ++ l2) st = run l2 (run l1 st) := by\n\
         \x20 intro l1\n\
         \x20 induction l1 with\n\
         \x20 | nil => intro l2 st; rfl\n\
         \x20 | cons i is ih => intro l2 st; simp only [List.cons_append, run, ih]\n\n\
         def iterN : Nat → (St → St) → St → St\n\
         \x20 | 0, _, st => st\n\
         \x20 | k + 1, f, st => iterN k f (f st)\n\n",
    );

    // emitter: each leaf kind emits a distinct constant, so the kinds are genuinely different to the elaborator rather than alpha-variants
    s.push_str("def emit : Stmt → List Instr\n");
    for i in 0..n {
        let sh = shape(i);
        s.push_str(&format!("  | .leaf{i} d {} => [.set d ({})]\n", sh.bind_pat(), sh.value));
    }
    s.push_str(
        "  | .seq s t => emit s ++ emit t\n\
         \x20 | .loop n body => (List.replicate n (emit body)).flatten\n\n",
    );

    s.push_str("def denote : Stmt → St → St\n");
    for i in 0..n {
        let sh = shape(i);
        s.push_str(&format!(
            "  | .leaf{i} d {} => fun st => ⟨fun x => if x = d then {} else st.regs x⟩\n",
            sh.bind_pat(), sh.value
        ));
    }
    s.push_str(
        "  | .seq s t => fun st => denote t (denote s st)\n\
         \x20 | .loop n body => fun st => iterN n (denote body) st\n\n",
    );

    // side conditions
    match wf {
        WfStyle::Recursive => {
            s.push_str("def WF : Stmt → Prop\n");
            for i in 0..n {
                s.push_str(&format!("  | .leaf{i} _ {} => {}\n",
                    shape(i).bind_pat(), shape(i).wf));
            }
            s.push_str(
                "  | .seq s t => WF s ∧ WF t\n  | .loop n body => 0 < n ∧ WF body\n\n",
            );
        }
        WfStyle::Inductive => {
            s.push_str("inductive WF : Stmt → Prop where\n");
            for i in 0..n {
                let sh = shape(i);
                s.push_str(&format!(
                    "  | leaf{i} {{d}} {} : {} → WF (.leaf{i} d {})\n",
                    sh.implicit_binders(), sh.wf, sh.bind_pat()
                ));
            }
            s.push_str(
                "  | seq {s t} : WF s → WF t → WF (.seq s t)\n\
                 \x20 | loop {n body} : 0 < n → WF body → WF (.loop n body)\n\n",
            );
        }
    }

    s.push_str("def wfCheck : Stmt → Bool\n");
    for i in 0..n {
        s.push_str(&format!("  | .leaf{i} _ {} => decide ({})\n",
            shape(i).bind_pat(), shape(i).wf));
    }
    s.push_str(
        "  | .seq s t => wfCheck s && wfCheck t\n\
         \x20 | .loop n body => decide (0 < n) && wfCheck body\n\n",
    );

    // reflective soundness
    s.push_str("theorem wfCheck_sound : ∀ s : Stmt, wfCheck s = true → WF s := by\n");
    s.push_str("  intro s\n  induction s with\n");
    for i in 0..n {
        match wf {
            WfStyle::Recursive => s.push_str(&format!(
                "  | leaf{i} d {} => intro h; simpa [wfCheck, WF] using h\n", shape(i).bind_pat()
            )),
            WfStyle::Inductive => s.push_str(&format!(
                "  | leaf{i} d {} => intro h; exact .leaf{i} (by simpa [wfCheck] using h)\n",
                shape(i).bind_pat()
            )),
        }
    }
    match wf {
        WfStyle::Recursive => s.push_str(
            "  | seq s t ihs iht =>\n\
             \x20     intro h; simp only [wfCheck, Bool.and_eq_true] at h\n\
             \x20     exact ⟨ihs h.1, iht h.2⟩\n\
             \x20 | loop n b ih =>\n\
             \x20     intro h\n\
             \x20     simp only [wfCheck, Bool.and_eq_true, decide_eq_true_eq] at h\n\
             \x20     exact ⟨h.1, ih h.2⟩\n\n",
        ),
        WfStyle::Inductive => s.push_str(
            "  | seq s t ihs iht =>\n\
             \x20     intro h; simp only [wfCheck, Bool.and_eq_true] at h\n\
             \x20     exact .seq (ihs h.1) (iht h.2)\n\
             \x20 | loop n b ih =>\n\
             \x20     intro h\n\
             \x20     simp only [wfCheck, Bool.and_eq_true, decide_eq_true_eq] at h\n\
             \x20     exact .loop h.1 (ih h.2)\n\n",
        ),
    }

    // compile correctness: one induction, every constructor
    s.push_str(
        "theorem run_flatten_replicate (l : List Instr) (f : St → St)\n\
         \x20   (h : ∀ st, run l st = f st) :\n\
         \x20   ∀ (n : Nat) (st : St), run ((List.replicate n l).flatten) st = iterN n f st := by\n\
         \x20 intro n\n\
         \x20 induction n with\n\
         \x20 | zero => intro st; rfl\n\
         \x20 | succ k ih =>\n\
         \x20     intro st\n\
         \x20     simp only [List.replicate_succ, List.flatten_cons, run_append, h, iterN, ih]\n\n",
    );

    s.push_str("theorem emit_correct : ∀ (s : Stmt), WF s → ∀ st, run (emit s) st = denote s st := by\n");
    s.push_str("  intro s\n  induction s with\n");
    let unwrap_leaf = match wf {
        WfStyle::Recursive => "simp only [WF] at h",
        WfStyle::Inductive => "cases h with | _ h => rename_i h; skip",
    };
    for i in 0..n {
        match wf {
            WfStyle::Recursive => s.push_str(&format!(
                "  | leaf{i} d {} =>\n\
                 \x20     intro h st; {unwrap_leaf}\n\
                 \x20     simp only [emit, run, runI, denote, Nat.mod_eq_of_lt h]\n",
                shape(i).bind_pat()
            )),
            WfStyle::Inductive => s.push_str(&format!(
                "  | leaf{i} d {} =>\n\
                 \x20     intro h st; cases h with | leaf{i} hb =>\n\
                 \x20     simp only [emit, run, runI, denote, Nat.mod_eq_of_lt hb]\n",
                shape(i).bind_pat()
            )),
        }
    }
    match wf {
        WfStyle::Recursive => s.push_str(
            "  | seq s t ihs iht =>\n\
             \x20     intro h st; simp only [WF] at h\n\
             \x20     simp only [emit, run_append, ihs h.1, iht h.2, denote]\n\
             \x20 | loop n b ih =>\n\
             \x20     intro h st; simp only [WF] at h\n\
             \x20     simp only [emit, denote]\n\
             \x20     exact run_flatten_replicate (emit b) (denote b) (ih h.2) n st\n\n",
        ),
        WfStyle::Inductive => s.push_str(
            "  | seq s t ihs iht =>\n\
             \x20     intro h st; cases h with | seq h1 h2 =>\n\
             \x20     simp only [emit, run_append, ihs h1, iht h2, denote]\n\
             \x20 | loop n b ih =>\n\
             \x20     intro h st; cases h with | loop h1 h2 =>\n\
             \x20     simp only [emit, denote]\n\
             \x20     exact run_flatten_replicate (emit b) (denote b) (ih h2) n st\n\n",
        ),
    }

    s.push_str(
        "theorem derived (p : Stmt) (hp : WF p) : ∀ st, run (emit p) st = denote p st :=\n\
         \x20 emit_correct p hp\n\n",
    );

    s.push_str("def mkD : Nat -> Nat -> Stmt\n");
    s.push_str(&format!(
        "  | 0, i => match i % {n} with\n"
    ));
    for i in 0..n {
        if i + 1 == n {
            s.push_str(&format!("    | _ => .leaf{i} \"r\" {}\n", shape(i).args("i")));
        } else {
            s.push_str(&format!("    | {i} => .leaf{i} \"r\" {}\n", shape(i).args("i")));
        }
    }
    s.push_str("  | n+1, i => .seq (mkD n (2*i)) (mkD n (2*i+1))\n\n");
    s.push_str("end ToyG\n");
    s
}

fn extent(kinds: usize) -> String {
    let n = leaves_for(kinds);
    let mut s = String::new();
    s.push_str("set_option maxRecDepth 8000000\nset_option maxHeartbeats 2000000\n");
    s.push_str("namespace ToyX\n\n");
    s.push_str("inductive Stmt : Nat → Type where\n");
    for i in 0..n {
        s.push_str(&format!(
            "  | leaf{i} (d : String) {} : Stmt 1\n", shape(i).binders
        ));
    }
    s.push_str(
        "  | seq {m n} (s : Stmt m) (t : Stmt n) : Stmt (m + n)\n\
         \x20 | loop (k : Nat) {n} (body : Stmt n) : Stmt (k * n)\n\n",
    );
    s.push_str("inductive Instr where | set (d : String) (v : Nat)\n\n");
    s.push_str("def emit : {n : Nat} → Stmt n → List Instr\n");
    for i in 0..n {
        let sh = shape(i);
        s.push_str(&format!("  | _, .leaf{i} d {} => [.set d ({})]\n",
            sh.bind_pat(), sh.value));
    }
    s.push_str(
        "  | _, .seq s t => emit s ++ emit t\n\
         \x20 | _, .loop k body => (List.replicate k (emit body)).flatten\n\n",
    );
    s.push_str(
        "theorem len_flatten_replicate (l : List Instr) :\n\
         \x20   ∀ k, ((List.replicate k l).flatten).length = k * l.length := by\n\
         \x20 intro k\n\
         \x20 induction k with\n\
         \x20 | zero => simp\n\
         \x20 | succ j ih =>\n\
         \x20     simp only [List.replicate_succ, List.flatten_cons, List.length_append, ih,\n\
         \x20       Nat.succ_mul]\n\
         \x20     omega\n\n",
    );

    // the payoff of indexing: emitted length is fixed by the type.
    s.push_str(
        "theorem emit_length : ∀ {n : Nat} (s : Stmt n), (emit s).length = n := by\n\
         \x20 intro n s\n\
         \x20 induction s with\n",
    );
    for i in 0..n {
        s.push_str(&format!("  | leaf{i} => rfl\n"));
    }
    s.push_str(
        "  | seq s t ihs iht => simp only [emit, List.length_append, ihs, iht]\n\
         \x20 | loop k body ih => simp only [emit, len_flatten_replicate, ih]\n\n",
    );
    s.push_str("end ToyX\n");
    s
}

/// A balanced, EXPLICIT nesting of constructor applications.
fn nested_term(lo: usize, hi: usize, leaves: usize) -> String {
    if hi - lo == 1 {
        format!("(.leaf{} \"r\" {})", lo % leaves, shape(lo % leaves).args(&lo.to_string()))
    } else {
        let mid = lo + (hi - lo) / 2;
        format!(
            "(.seq {} {})",
            nested_term(lo, mid, leaves),
            nested_term(mid, hi, leaves)
        )
    }
}

fn minimal_term(lo: usize, hi: usize) -> String {
    if hi - lo == 1 {
        format!("(.leaf {} {})", lo % 100, (lo * 7) % 100)
    } else {
        let mid = lo + (hi - lo) / 2;
        format!("(.seq {} {})", minimal_term(lo, mid), minimal_term(mid, hi))
    }
}

pub fn dsl(kinds: usize, index: Index, wf: WfStyle, out: &Path) -> std::io::Result<()> {
    let src = match index {
        Index::None => plain(kinds, wf),
        Index::Extent => extent(kinds),
        Index::ExtentMinimal => EXTENT_MINIMAL.to_string(),
    };
    write(out, &src)
}

const EXTENT_MINIMAL: &str = r#"set_option maxRecDepth 8000000
set_option maxHeartbeats 2000000
namespace ToyM

inductive Instr where
  | set (d : String) (v : Nat)

inductive Stmt : Nat -> Type where
  | leaf (a b : Nat) : Stmt 1
  | seq {m n} (s : Stmt m) (t : Stmt n) : Stmt (m + n)

def emit : {n : Nat} -> Stmt n -> List Instr
  | _, .leaf a b => [.set "r" (a + b)]
  | _, .seq s t => emit s ++ emit t

theorem emit_length : forall {n : Nat} (s : Stmt n), (emit s).length = n := by
  intro n s
  induction s with
  | leaf => rfl
  | seq s t ihs iht => simp only [emit, List.length_append, ihs, iht]

end ToyM
"#;

pub fn dsl_program(depth: u32, kinds: usize, index: Index, out: &Path) -> std::io::Result<()> {
    let body = match index {
        Index::None => format!(
            "import ToyG\nopen ToyG\nset_option maxRecDepth 8000000\n\
             set_option maxHeartbeats 2000000\n\n\
             def p : Stmt := mkD {depth} 0\n\
             theorem ok : WF p := wfCheck_sound p rfl\n\
             theorem d : ∀ st, run (emit p) st = denote p st := derived p ok\n"
        ),
        Index::Extent => {
            let n = 1usize << depth;
            format!(
                "import ToyX\nopen ToyX\nset_option maxRecDepth 8000000\n\
                 set_option maxHeartbeats 2000000\n\n\
                 def p : Stmt {n} :=\n  {}\n\n\
                 -- the indexed payoff, for free from the type\n\
                 example : (emit p).length = {n} := emit_length p\n",
                nested_term(0, n, leaves_for(kinds))
            )
        }
        Index::ExtentMinimal => {
            let n = 1usize << depth;
            let body = minimal_term(0, n);
            format!(
                "import ToyM\nopen ToyM\nset_option maxRecDepth 8000000\n\
                 set_option maxHeartbeats 2000000\n\n\
                 def p : Stmt {n} :=\n  {body}\n\n\
                 example : (emit p).length = {n} := emit_length p\n"
            )
        }
    };
    write(out, &body)
}
