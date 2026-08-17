import AlgorithmLib.Clif
import AlgorithmLib.HProgBlocks

/-!
  # Checking the launch model against the machine

  `Clif.stepPure` is an abstract interpretation: it reports what it can prove
  about a value and `unknown` about everything else.  `HProg.Blocks.evalInst`
  is the concrete one, over the *same* `Inst`, and it computes through
  `Sem.evalOp` — the semantics `HProgCorpus` tests against a real machine.

  Two semantics, and only one of them was checked.  The launch structure every
  host proof reads out of a compiled function comes from the abstract one, so
  a wrong constant arm there is unsound in a way nothing downstream detects.

  What is checked here is exactly the promise the model makes:

  > if `stepPure` reports `const k` for a value, the machine's word for that
  > value has signed value `k`.

  Nothing is claimed about `unknown`, `offset`, `slot` or `derived` — those
  describe *provenance*, not a number, and the theorems consuming them carry
  their own side conditions.

  This is a test, not a proof.  It runs every tracked instruction at four
  widths over a sample sitting on the boundaries where the two disagreed: the
  width limits from both sides, shift amounts at and past each width, and
  negative operands.  A proof would be better and is not yet written; this is
  the footing `Sem` itself stands on.
-/

namespace AlgorithmLib.Clif.Check

open AlgorithmLib.IR
open AlgorithmLib.Clif
open AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Blocks

/-- **The promise, as a decision.**  A reported constant must be the machine's
    word read as a signed integer of its own width — the convention that makes
    `sextend64` sound to pass a constant through and `uextend64` not.

    A model value that is not `const` claims no number, so it passes.  So does
    an instruction the machine gets stuck on: it never runs. -/
def claimHolds (sym : SymVal) (conc : Option V) : Bool :=
  match sym, conc with
  | .const k, some (.sc t x) => signed t x == k
  | .const _, some _         => false
  | _,        _              => true

/-- One nullary instruction: the literal itself. -/
def constCase (t : ClifTy) (x : Int) : Bool :=
  let i : Inst := .iconst ⟨0⟩ t x
  claimHolds ((stepPure Env.empty i) ⟨0⟩) ((evalInst default #[] i).map (·.2))

/-- One unary instruction over a literal. -/
def unCase (t : ClifTy) (x : Int) (mk : Val → Val → Inst) : Bool :=
  let i := mk ⟨1⟩ ⟨0⟩
  let e := stepPure Env.empty (.iconst ⟨0⟩ t x)
  let vs : Vals := setV #[] ⟨0⟩ (ofInt t x)
  claimHolds ((stepPure e i) ⟨1⟩) ((evalInst default vs i).map (·.2))

/-- One binary instruction over two literals. -/
def binCase (t : ClifTy) (x y : Int) (mk : Val → Val → Val → Inst) : Bool :=
  let i := mk ⟨2⟩ ⟨0⟩ ⟨1⟩
  let e := stepPure (stepPure Env.empty (.iconst ⟨0⟩ t x)) (.iconst ⟨1⟩ t y)
  let vs : Vals := setV (setV #[] ⟨0⟩ (ofInt t x)) ⟨1⟩ (ofInt t y)
  claimHolds ((stepPure e i) ⟨2⟩) ((evalInst default vs i).map (·.2))

/-- Every integer width the tracked fragment is emitted at. -/
def types : List ClifTy := [.i8, .i16, .i32, .i64]

/-- **Boundaries, not a spread.**  Each width's limits from both sides, the
    shift amounts at and past every width, and negatives — the three shapes
    the two interpreters were found to disagree on. -/
def sample : List Int :=
  [ 0, 1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65,
    127, 128, 255, 256, 32767, 32768, 65535, 65536,
    2147483647, 2147483648, 2147483649, 4294967296,
    1099511627776, 4611686018427387904, 9223372036854775807,
    -1, -2, -3, -8, -127, -128, -129, -32768, -32769,
    -2147483648, -2147483649, -4294967296, -9223372036854775808 ]

def unOps : List (Val → Val → Inst) :=
  [ (fun d a => .ineg d a)
  , (fun d a => .ireduce32 d a)
  , (fun d a => .uextend64 d a)
  , (fun d a => .sextend64 d a) ]

def binOps : List (Val → Val → Val → Inst) :=
  [ (fun d a b => .iadd d a b)
  , (fun d a b => .isub d a b)
  , (fun d a b => .imul d a b)
  , (fun d a b => .ishl d a b)
  , (fun d a b => .ushr d a b) ]

def constOk : Bool := types.all fun t => sample.all fun x => constCase t x

def unOk : Bool :=
  types.all fun t => sample.all fun x => unOps.all fun f => unCase t x f

def binOk : Bool :=
  types.all fun t => sample.all fun x => sample.all fun y =>
    binOps.all fun f => binCase t x y f

/-- **The launch model never reports a constant the machine does not compute**,
    over every tracked instruction at every integer width on the sample above.

    `native_decide`, so this is checked by running compiled code — the footing
    `HProgCorpus` stands on, which is what makes the concrete side worth
    comparing against. -/
theorem stepPure_constants_agree :
    (constOk && unOk && binOk) = true := by native_decide

/-- Whether an arm reports a constant at all, so the check above can be shown
    to have something to check. -/
def reportsConst (t : ClifTy) (x y : Int) (mk : Val → Val → Val → Inst) : Bool :=
  let e := stepPure (stepPure Env.empty (.iconst ⟨0⟩ t x)) (.iconst ⟨1⟩ t y)
  match (stepPure e (mk ⟨2⟩ ⟨0⟩ ⟨1⟩)) ⟨2⟩ with
  | .const _ => true
  | _        => false

def reportsConst1 (t : ClifTy) (x : Int) (mk : Val → Val → Inst) : Bool :=
  let e := stepPure Env.empty (.iconst ⟨0⟩ t x)
  match (stepPure e (mk ⟨1⟩ ⟨0⟩)) ⟨1⟩ with
  | .const _ => true
  | _        => false

/-- **…and the check is not vacuous.**  Every arm reports a constant on at
    least one case, so a model that answered `unknown` everywhere would pass
    `stepPure_constants_agree` and fail this. -/
theorem check_is_live :
    (binOps.all (fun f => reportsConst .i64 6 7 f)
      && unOps.all (fun f => reportsConst1 .i32 6 f)
      && constCase .i32 6) = true := by native_decide

end AlgorithmLib.Clif.Check
