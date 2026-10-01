module
public import AlgorithmLib.Host.Clif
meta import AlgorithmLib.Host.Clif
public import AlgorithmLib.Host.Blocks
meta import AlgorithmLib.Host.Blocks
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

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

-- ---------------------------------------------------------------------------
-- The other claim: a `derived` value denotes the machine's number
-- ---------------------------------------------------------------------------

/-- The roots a `DExp` names are SSA values; their valuation is the machine's
    own word for them, read signed — the same convention `claimHolds` uses. -/
def rhoOf (vs : Vals) : Nat → Int := fun k =>
  match getV vs ⟨k⟩ with
  | some (.sc t w) => signed t w
  | _              => 0

/-- One instruction with a *runtime* left operand — the model cannot see `v0`,
    so it names it and builds an expression instead of folding. -/
def derivCase (ta tb : ClifTy) (x y : Int) (mk : Val → Val → Val → Inst) : Bool :=
  let i := mk ⟨2⟩ ⟨0⟩ ⟨1⟩
  let e := stepPure Env.empty (.iconst ⟨1⟩ tb y)
  let vs : Vals := setV (setV #[] ⟨0⟩ (ofInt ta x)) ⟨1⟩ (ofInt tb y)
  let conc : Option V := (evalInst default vs i).map (·.2)
  match (stepPure e i) ⟨2⟩, conc with
  | .derived d, some (.sc t' w) =>
      !DExp.Exact (rhoOf vs) d || signed t' w == DExp.eval (rhoOf vs) d
  | _, _ => true

/-- The three instructions that build an expression rather than fold. -/
def derivOps : List (Val → Val → Val → Inst) :=
  [ (fun d a b => .isub d a b)
  , (fun d a b => .ishl d a b)
  , (fun d a b => .ushr d a b) ]

def wideTypes : List ClifTy := [.i32, .i64]

/-- Cases the theorem above excludes: a shift whose *shifted* operand is
    narrower than `i32`. -/
def narrowDerivFails : Nat :=
  ([ClifTy.i8, .i16].flatMap fun ta => wideTypes.flatMap fun tb =>
    sample.flatMap fun x => sample.flatMap fun y =>
      derivOps.filter fun f => !(derivCase ta tb x y f)).length

/-- **The width restriction is real**, so `stepPure_derived_agree` is not quietly
    stronger than it reads.  `ishl` of `i8` `1` by `7` is `128` in the expression
    algebra and `-128` on the machine: the result wraps at the operand's width,
    which `DExp.Exact` measures against the signed `i32` range instead.  A
    consumer reading a bound out of a `derived` value owes that its roots are
    `i32` or wider. -/
theorem narrowShiftDisagrees : (800 < narrowDerivFails) = true := by native_decide

/-- The three instructions that retag a value rather than compute one. -/
def retagOps : List (Val → Val → Inst) :=
  [ (fun d a => .ireduce32 d a)
  , (fun d a => .uextend64 d a)
  , (fun d a => .sextend64 d a) ]

/-- **An expression with a retag on top of it.**

    `derivCase` builds one instruction, so it can only ever check an expression
    the model has just named.  Every claim `stepPure` makes about a *composed*
    value went unchecked, and that is where the model's `uextend64` arm was
    wrong: it passed an `offset` or a `derived` through unchanged, and zero
    extension does not preserve a negative value.  `v0 - 1` at `i32` with `v0`
    zero is `-1` in the expression algebra and `4294967295` after the extend. -/
def derivRetagCase (ta tb : ClifTy) (x y : Int)
    (f : Val → Val → Val → Inst) (g : Val → Val → Inst) : Bool :=
  let is := [f ⟨2⟩ ⟨0⟩ ⟨1⟩, g ⟨3⟩ ⟨2⟩]
  let e := evalPure (stepPure Env.empty (.iconst ⟨1⟩ tb y)) is
  let vs0 : Vals := setV (setV #[] ⟨0⟩ (ofInt ta x)) ⟨1⟩ (ofInt tb y)
  let vs := is.foldl (fun s i =>
    match evalInst default s i with | some (d, v) => setV s d v | none => s) vs0
  match e ⟨3⟩, getV vs ⟨3⟩ with
  | .derived d, some (.sc t' w) =>
      !DExp.Exact (rhoOf vs) d || signed t' w == DExp.eval (rhoOf vs) d
  | .offset p k, some (.sc t' w) =>
      let d := DExp.add (.root p.id) (.lit k)
      !DExp.Exact (rhoOf vs) d || signed t' w == DExp.eval (rhoOf vs) d
  | _, _ => true

/-- The same, counting cases the condition admits. -/
def derivRetagLive (ta tb : ClifTy) (x y : Int)
    (f : Val → Val → Val → Inst) (g : Val → Val → Inst) : Bool :=
  let is := [f ⟨2⟩ ⟨0⟩ ⟨1⟩, g ⟨3⟩ ⟨2⟩]
  let e := evalPure (stepPure Env.empty (.iconst ⟨1⟩ tb y)) is
  let vs0 : Vals := setV (setV #[] ⟨0⟩ (ofInt ta x)) ⟨1⟩ (ofInt tb y)
  let vs := is.foldl (fun s i =>
    match evalInst default s i with | some (d, v) => setV s d v | none => s) vs0
  match e ⟨3⟩ with
  | .derived d => DExp.Exact (rhoOf vs) d
  | .offset p k => DExp.Exact (rhoOf vs) (.add (.root p.id) (.lit k))
  | _          => false

def derivRetagOk : Bool :=
  wideTypes.all fun ta => wideTypes.all fun tb => sample.all fun x => sample.all fun y =>
    derivOps.all fun f => retagOps.all fun g => derivRetagCase ta tb x y f g

def derivRetagLiveCount : Nat :=
  (wideTypes.flatMap fun ta => wideTypes.flatMap fun tb => sample.flatMap fun x =>
    sample.flatMap fun y => derivOps.flatMap fun f =>
      retagOps.filter fun g => derivRetagLive ta tb x y f g).length

/-- **A retag over an expression denotes what the machine computes too.**

    The composition the single-instruction check could not reach.  It is what
    holds `stepPure`'s two passthrough arms to the same standard as the arms
    that build an expression: `ireduce32` truncates and `sextend64` preserves
    the signed value, both exact inside `foldableRange`; `uextend64` does not,
    and now says so. -/
theorem stepPure_derived_retag_agree : derivRetagOk = true := by native_decide

/-- …and the composed check admits cases rather than refusing all of them. -/
theorem derived_retag_is_live : (1000 < derivRetagLiveCount) = true := by native_decide

-- ---------------------------------------------------------------------------
-- The third claim: an `offset` is its base plus its displacement
-- ---------------------------------------------------------------------------

/-- Region bases as the runtime actually supplies them.  Checking this claim
    only at small bases would miss the case it exists for: a launch argument's
    `ptr` is an address like `0x1000000000`, far outside `foldableRange`, and
    the claim about it is still true because `i64` is what it wraps at. -/
def bases : List Int :=
  sample ++ [68719476736, 137438953472, 206158430208, 68719476740]

/-- One instruction with a runtime base — the model names the base and records
    a displacement rather than folding. -/
def offCase (ta tb : ClifTy) (base k : Int) (mk : Val → Val → Val → Inst) : Bool :=
  let i := mk ⟨2⟩ ⟨0⟩ ⟨1⟩
  let e := stepPure Env.empty (.iconst ⟨1⟩ tb k)
  let vs : Vals := setV (setV #[] ⟨0⟩ (ofInt ta base)) ⟨1⟩ (ofInt tb k)
  let conc : Option V := (evalInst default vs i).map (·.2)
  match (stepPure e i) ⟨2⟩, conc with
  | .offset p d, some (.sc t' w) =>
      let b := rhoOf vs p.id
      !inTy t' (b + d) || signed t' w == b + d
  | _, _ => true

def offLive (ta tb : ClifTy) (base k : Int) (mk : Val → Val → Val → Inst) : Bool :=
  let i := mk ⟨2⟩ ⟨0⟩ ⟨1⟩
  let e := stepPure Env.empty (.iconst ⟨1⟩ tb k)
  let vs : Vals := setV (setV #[] ⟨0⟩ (ofInt ta base)) ⟨1⟩ (ofInt tb k)
  let conc : Option V := (evalInst default vs i).map (·.2)
  match (stepPure e i) ⟨2⟩, conc with
  | .offset p d, some (.sc t' _) => inTy t' (rhoOf vs p.id + d)
  | _, _                         => false

def offOps : List (Val → Val → Val → Inst) :=
  [ (fun d a b => .iadd d a b), (fun d a b => .isub d a b) ]

def offOk : Bool :=
  types.all fun ta => wideTypes.all fun tb => bases.all fun b => sample.all fun k =>
    offOps.all fun f => offCase ta tb b k f

def offLiveCount : Nat :=
  (types.flatMap fun ta => wideTypes.flatMap fun tb => bases.flatMap fun b =>
    sample.flatMap fun k => offOps.filter fun f => offLive ta tb b k f).length

/-- **The `offset` claim at widths the proof does not reach.**

    `denotes_offset` proves this claim, and only at `i32` and `i64`: `Congr` is
    stated modulo `modOf`, which is defined at those two.  `offOk` ranges over
    all four widths, so what it still covers is `i8` and `i16` — where the same
    claim is true and unproved.  Kept for that reason, not for redundancy. -/
theorem stepPure_offset_agrees : offOk = true := by native_decide

theorem offset_check_is_live : (2000 < offLiveCount) = true := by native_decide

-- ---------------------------------------------------------------------------
-- The fourth claim: a `slot` came from the address it names
-- ---------------------------------------------------------------------------

/-- A region with distinguishable bytes, so a load from the wrong offset gives
    a different answer rather than the same zero. -/
def arenaBytes : ByteArray :=
  ⟨((List.range 512).map (fun i => UInt8.ofNat ((i * 37 + 11) % 256))).toArray⟩

def mem0 : Mem := { arena := arenaBytes, data := ByteArray.empty, out := ByteArray.empty }

def runInsts (m : Mem) (vs : Vals) : List Inst → Option Vals
  | []      => some vs
  | i :: is => match evalInst m vs i with
               | some (d, x) => runInsts m (setV vs d x) is
               | none        => none

/-- The fragment a generator writes as `slotWq.load ptr`: a runtime base in
    `v0`, the displacement, the address, the load. -/
def slotInsts (k : Nat) (op : LoadOp) : List Inst :=
  [ .iconst ⟨1⟩ .i64 (Int.ofNat k)
  , .iadd ⟨2⟩ ⟨0⟩ ⟨1⟩
  , .load ⟨3⟩ op ⟨2⟩ ]

/-- Every load `ProgFFI` emits, not only the one a bind table is read with.

    `uload8_64` is `scalarLoad`'s lowering, so a narrow load reaching a `slot`
    is not hypothetical. -/
def loadKinds : List LoadOp :=
  [ { ty := .i64 }, { ty := .i32 }
  , { kind := .uload8,  ty := .i64 }, { kind := .uload8,  ty := .i32 }
  , { kind := .uload32, ty := .i64 }, { kind := .sload8,  ty := .i64 } ]

/-- **What `slot p d` asserts**: the value came from the address `p + d`.

    Checked by re-reading that address with the *same* load, which is the claim
    at every kind.  Reading the whole eight-byte word instead is a different
    claim, true only of a plain 64-bit load — see `slotWordCase`. -/
def slotCase (k : Nat) (op : LoadOp) : Bool :=
  let vs0 : Vals := setV #[] ⟨0⟩ (.sc .i64 (addrOf .arena 0))
  match runInsts mem0 vs0 (slotInsts k op),
        (slotInsts k op).foldl stepPure Env.empty ⟨3⟩ with
  | some vs, .slot p d =>
      match getV vs ⟨3⟩, getV vs p with
      | some loaded, some (.sc _ pw) =>
          AlgorithmLib.HProg.Blocks.viaOp mem0
            [.sc ClifTy.i64 (pw + UInt64.ofNat d.toNat)] (.load op 0) == some loaded
      | _, _ => false
  | some _, _ => false
  | none,   _ => true

/-- **…and for a plain 64-bit load, the value is the word itself.**

    The stronger reading, and the one `valueClaimHolds` decides.  It is stated
    here at the kind that has it rather than left to stand for all of them: a
    `uload8` returns one byte and a `sload8` sign-extends it, so neither is the
    eight-byte word nor its truncation. -/
def slotWordCase (k : Nat) : Bool :=
  let op : LoadOp := { ty := .i64 }
  let vs0 : Vals := setV #[] ⟨0⟩ (.sc .i64 (addrOf .arena 0))
  match runInsts mem0 vs0 (slotInsts k op),
        (slotInsts k op).foldl stepPure Env.empty ⟨3⟩ with
  | some vs, .slot p d =>
      match getV vs ⟨3⟩, getV vs p with
      | some (.sc t loaded), some (.sc _ pw) =>
          (Mem.load mem0 (pw + UInt64.ofNat d.toNat) 8).map (· &&& widthMask t)
            == some loaded
      | _, _ => false
  | some _, _ => false
  | none,   _ => true

/-- Cases the word reading gets wrong, so `slotWordCase` is known to be stated
    at the kind that has it rather than at the kind that was sampled. -/
def slotWordFailsNarrow : Nat :=
  ((List.range 32).flatMap fun k =>
    ([{ kind := .uload8, ty := .i64 }, { kind := .uload32, ty := .i64 },
      { kind := .sload8, ty := .i64 }] : List LoadOp).filter fun op =>
        !(slotCaseWord k op)).length
where
  slotCaseWord (k : Nat) (op : LoadOp) : Bool :=
    let vs0 : Vals := setV #[] ⟨0⟩ (.sc .i64 (addrOf .arena 0))
    match runInsts mem0 vs0 (slotInsts k op),
          (slotInsts k op).foldl stepPure Env.empty ⟨3⟩ with
    | some vs, .slot p d =>
        match getV vs ⟨3⟩, getV vs p with
        | some (.sc t loaded), some (.sc _ pw) =>
            (Mem.load mem0 (pw + UInt64.ofNat d.toNat) 8).map (· &&& widthMask t)
              == some loaded
        | _, _ => false
    | some _, _ => false
    | none,   _ => true

/-- Whether the model reported a `slot` at all, so the check above is known to
    be testing the arm it names. -/
def slotLive (k : Nat) (op : LoadOp) : Bool :=
  match (slotInsts k op).foldl stepPure Env.empty ⟨3⟩ with
  | .slot _ _ => true
  | _         => false

def slotOk : Bool :=
  (List.range 200).all fun k => loadKinds.all (slotCase k)

def slotWordOk : Bool := (List.range 200).all slotWordCase

def slotLiveCount : Nat :=
  ((List.range 200).flatMap fun k => loadKinds.filter (slotLive k)).length

/-- **A handle the model calls `slot p d` came from the address `p + d`.**

    This is what makes a buffer handle identifiable — without it Qwen2's `Wq`,
    `Wk` and `Wv` launches are the same record — so it is worth checking that
    the address the model names is the address the load read.  At every load
    kind: the model's `.load` arm does not look at one. -/
theorem stepPure_slot_agrees : slotOk = true := by native_decide

/-- …and at a plain 64-bit load the value is the whole word. -/
theorem stepPure_slot_is_the_word : slotWordOk = true := by native_decide

/-- **The word reading is not the general claim.**  A narrow or sign-extending
    load reaches the same `.slot` binding and returns neither the eight-byte
    word nor its truncation, so `stepPure_slot_is_the_word` is stated at the
    kind that has it rather than assumed of all of them. -/
theorem slotWordFailsAtNarrowLoads : (90 < slotWordFailsNarrow) = true := by
  native_decide

theorem slot_check_is_live : (slotLiveCount == 1200) = true := by native_decide


-- ---------------------------------------------------------------------------
-- All four claims at once, over composed instructions
-- ---------------------------------------------------------------------------

/-- **Every claim the model makes about one value, decided against the
    machine.**

    The four checks above each build their own one-instruction program and look
    at one arm, which is why each of them could only ever see a value the model
    had *just* named.  `stepPure` propagates: `ireduce32` passes an expression
    through, `iadd` absorbs a displacement, `load` turns an offset into a slot.
    Every claim about a value two or more instructions old was unchecked, and
    that is where `uextend64` was found carrying an expression through a
    zero-extension.

    A value the model does not name claims nothing and passes; so does one the
    machine never computed. -/
def valueClaimHolds (m : Mem) (vs : Vals) (e : Env) (v : Val) : Bool :=
  match e v, getV vs v with
  | .const k,    some (.sc t w) => signed t w == k
  | .const _,    some _         => false
  | .offset p d, some (.sc t w) =>
      let b := rhoOf vs p.id
      !inTy t (b + d) || signed t w == b + d
  | .offset _ _, some _         => false
  | .derived dx, some (.sc t w) =>
      !DExp.Exact (rhoOf vs) dx || signed t w == DExp.eval (rhoOf vs) dx
  | .derived _,  some _         => false
  | .slot p d,   some (.sc t loaded) =>
      match getV vs p with
      | some (.sc _ pw) =>
          (Mem.load m (pw + UInt64.ofNat d.toNat) 8).map (· &&& widthMask t)
            == some loaded
      | _               => false
  | _,           _              => true

/-- Whether the model named the value at all, so a census can report how much
    of its grid is carrying the result rather than passing by silence. -/
def valueNamed (e : Env) (v : Val) : Bool :=
  match e v with
  | .unknown => false
  | _        => true

/-- The seeds every composed program starts from: a runtime value the model
    cannot name, a literal it can, and a region base. -/
def seedVals (ta tb : ClifTy) (x y : Int) : Vals :=
  setV (setV (setV #[] ⟨0⟩ (ofInt ta x)) ⟨1⟩ (ofInt tb y)) ⟨2⟩
    (.sc .i64 (addrOf .arena 0))

/-- Boundaries only.  The composed grid is a product of two of these, so it
    takes the values each width wraps at rather than the full spread. -/
def compSample : List Int :=
  [ 0, 1, 31, 32, 127, 128, 2147483647, 2147483648,
    -1, -128, -2147483648, 4294967296 ]

def compBin : List (Val → Val → Val → Inst) :=
  [ (fun d a b => .iadd d a b), (fun d a b => .isub d a b)
  , (fun d a b => .imul d a b), (fun d a b => .ishl d a b)
  , (fun d a b => .ushr d a b) ]

def compUn : List (Val → Val → Inst) :=
  [ (fun d a => .ireduce32 d a), (fun d a => .uextend64 d a)
  , (fun d a => .sextend64 d a), (fun d a => .ineg d a)
  , (fun d a => .load d { ty := .i64 } a) ]

/-- What a first instruction is given: the runtime value with the literal, and
    the region base with the literal — the two shapes that produce a `derived`
    and an `offset` respectively. -/
def firstPairs : List (Val × Val) := [(⟨0⟩, ⟨1⟩), (⟨2⟩, ⟨1⟩)]

/-- …and what a second is given, as a function of its destination alone: the
    first result against the literal, against the runtime value, or on its own
    through each unary arm — `load` included, so an `offset` composed by the
    first instruction is checked as the `slot` it becomes. -/
def compSecond : List (Val → Inst) :=
  (compBin.flatMap fun g => [(fun d => g d ⟨3⟩ ⟨1⟩), (fun d => g d ⟨3⟩ ⟨0⟩)])
    ++ compUn.map (fun g => fun d => g d ⟨3⟩)

/-- Two instructions, and every claim the model makes about either result. -/
def compCase (ta tb : ClifTy) (x y : Int)
    (f : Val → Val → Val → Inst) (p : Val × Val) (g : Val → Inst) : Bool :=
  let is := [f ⟨3⟩ p.1 p.2, g ⟨4⟩]
  let e := is.foldl stepPure (stepPure Env.empty (.iconst ⟨1⟩ tb y))
  match runInsts mem0 (seedVals ta tb x y) is with
  | some vs => valueClaimHolds mem0 vs e ⟨3⟩ && valueClaimHolds mem0 vs e ⟨4⟩
  | none    => true

/-- Whether the composed result is one the model named. -/
def compLive (ta tb : ClifTy) (x y : Int)
    (f : Val → Val → Val → Inst) (p : Val × Val) (g : Val → Inst) : Bool :=
  let is := [f ⟨3⟩ p.1 p.2, g ⟨4⟩]
  let e := is.foldl stepPure (stepPure Env.empty (.iconst ⟨1⟩ tb y))
  match runInsts mem0 (seedVals ta tb x y) is with
  | some _ => valueNamed e ⟨4⟩
  | none   => false

def compOk : Bool :=
  wideTypes.all fun ta => wideTypes.all fun tb =>
    compSample.all fun x => compSample.all fun y =>
      compBin.all fun f => firstPairs.all fun p =>
        compSecond.all fun g => compCase ta tb x y f p g

def compLiveCount : Nat :=
  (wideTypes.flatMap fun ta => wideTypes.flatMap fun tb =>
    compSample.flatMap fun x => compSample.flatMap fun y =>
      compBin.flatMap fun f => firstPairs.flatMap fun p =>
        compSecond.filter fun g => compLive ta tb x y f p g).length

/-- **Every claim survives composition.**

    The census the four single-instruction checks could not run: each of `iadd`,
    `isub`, `imul`, `ishl`, `ushr` over a runtime value and over a region base,
    followed by each of those again or by a retag or a load — and at both
    results, whichever of the four claims the model happens to make.

    This is the check that would have caught the `uextend64` arm, and it is the
    one to extend when `stepPure` gains an arm, because a new arm's hazard is
    almost never the instruction alone. -/
theorem stepPure_composed_claims_agree : compOk = true := by native_decide

/-- …and the grid is mostly cases the model does name.  A `compOk` that passed
    by answering `unknown` everywhere would say nothing. -/
theorem composed_check_is_live : (5000 < compLiveCount) = true := by native_decide

/-- **A `slot`, and then something else done to it.**

    `compCase` only ever loads *last*, so it never asks what happens to a slot
    that is carried further — and `stepPure` does carry it, because a retag
    passes its operand's binding through.  This builds the slot first: a region
    base, a displacement, the load, and then each second-stage arm on top. -/
def compSlotCase (k : Nat) (g : Val → Inst) : Bool :=
  let is : List Inst :=
    [ .iadd ⟨3⟩ ⟨2⟩ ⟨1⟩, .load ⟨4⟩ { ty := .i64 } ⟨3⟩, g ⟨5⟩ ]
  let e := is.foldl stepPure (stepPure Env.empty (.iconst ⟨1⟩ .i64 (Int.ofNat k)))
  match runInsts mem0 (seedVals .i64 .i64 0 (Int.ofNat k)) is with
  | some vs => valueClaimHolds mem0 vs e ⟨4⟩ && valueClaimHolds mem0 vs e ⟨5⟩
  | none    => true

def compSlotLive (k : Nat) (g : Val → Inst) : Bool :=
  let is : List Inst :=
    [ .iadd ⟨3⟩ ⟨2⟩ ⟨1⟩, .load ⟨4⟩ { ty := .i64 } ⟨3⟩, g ⟨5⟩ ]
  let e := is.foldl stepPure (stepPure Env.empty (.iconst ⟨1⟩ .i64 (Int.ofNat k)))
  match runInsts mem0 (seedVals .i64 .i64 0 (Int.ofNat k)) is with
  | some _ => valueNamed e ⟨5⟩
  | none   => false

/-- Displacements inside the arena, at the width a load is emitted for. -/
def slotOffsets : List Nat := (List.range 48).map (8 * ·)

/-- The second-stage arms, rebased on the loaded value rather than on `v3`.

    `ireduce32` is here: a truncated handle is the case the whole-word reading
    of a `slot` got wrong. -/
def compSlotSecond : List (Val → Inst) :=
  (compBin.flatMap fun g => [(fun d => g d ⟨4⟩ ⟨1⟩), (fun d => g d ⟨4⟩ ⟨0⟩)])
    ++ [ (fun d => .ireduce32 d ⟨4⟩), (fun d => .uextend64 d ⟨4⟩)
       , (fun d => .sextend64 d ⟨4⟩), (fun d => .ineg d ⟨4⟩)
       , (fun d => .load d { ty := .i64 } ⟨4⟩) ]

def compSlotOk : Bool :=
  slotOffsets.all fun k => compSlotSecond.all fun g => compSlotCase k g

def compSlotLiveCount : Nat :=
  (slotOffsets.flatMap fun k => compSlotSecond.filter fun g => compSlotLive k g).length



/-- **A claim about a loaded value survives being carried further.** -/
theorem stepPure_composed_slot_agrees : compSlotOk = true := by native_decide

theorem composed_slot_is_live : (100 < compSlotLiveCount) = true := by native_decide



/-! ### From checked to proved: the constant arm, against the machine

    Everything above *tests* the model against `evalInst` over a corpus.  This
    proves one arm of it outright, and the reasoning it needs is ordinary: the
    width mask is `Nat.and_two_pow_sub_one_eq_mod` from core, and once `signed`
    is stated as arithmetic on `toNat`, `omega` closes the rest.  No Mathlib is
    involved, and neither is `native_decide`. -/

theorem mask32 (x : UInt64) : (x &&& 4294967295).toNat = x.toNat % 2 ^ 32 := by
  rw [UInt64.toNat_and, show (4294967295 : UInt64).toNat = 2 ^ 32 - 1 from rfl,
      Nat.and_two_pow_sub_one_eq_mod]

theorem signed_i32_eq (x : UInt64) :
    signed .i32 x
      = if 2147483648 ≤ x.toNat % 2 ^ 32
        then ((x.toNat % 2 ^ 32 : Nat) : Int) - 4294967296
        else ((x.toNat % 2 ^ 32 : Nat) : Int) := by
  simp only [signed, AlgorithmLib.IR.ClifTy.width, widthMask,
             show (1 <<< UInt64.ofNat (32 - 1) : UInt64) = 2147483648 from by decide,
             ge_iff_le, UInt64.le_iff_toNat_le, mask32,
             show ((2147483648 : UInt64)).toNat = 2147483648 from rfl,
             show ¬ (64 ≤ 32) from by decide, if_false,
             show ((1 <<< 32 : Nat) : Int) = 4294967296 from rfl]

theorem signed_i32_of_toNat (k : Int) (n : Nat)
    (hn : (n : Int) = k.emod 18446744073709551616)
    (h0 : -2147483648 ≤ k) (h1 : k < 2147483648) :
    signed .i32 (UInt64.ofNat n) = k := by
  rw [signed_i32_eq,
      show (UInt64.ofNat n).toNat = n % 2 ^ 64 from by simp,
      Nat.mod_mod_of_dvd _ (by decide : (2:Nat) ^ 32 ∣ 2 ^ 64)]
  have d1 : 0 ≤ k.emod 18446744073709551616 := Int.emod_nonneg k (by decide)
  have d2 : k.emod 18446744073709551616 < 18446744073709551616 :=
    Int.emod_lt_of_pos k (by decide)
  have d3 : 18446744073709551616 * (k / 18446744073709551616)
              + k.emod 18446744073709551616 = k := Int.ediv_add_emod k _
  split <;> omega

theorem signed_i32_of_inFold (k : Int) (h0 : -2147483648 ≤ k) (h1 : k < 2147483648) :
    signed .i32 (UInt64.ofNat (k.emod (1 <<< 64)).toNat) = k := by
  refine signed_i32_of_toNat k _ ?_ h0 h1
  have hlo : (0 : Int) ≤ k.emod 18446744073709551616 :=
    Int.emod_nonneg k (by decide)
  simpa only [Int.ofNat_eq_natCast] using Int.toNat_of_nonneg hlo

theorem mask64 (x : UInt64) : (x &&& 18446744073709551615).toNat = x.toNat := by
  rw [UInt64.toNat_and, show (18446744073709551615 : UInt64).toNat = 2 ^ 64 - 1 from rfl,
      Nat.and_two_pow_sub_one_eq_mod]
  exact Nat.mod_eq_of_lt x.toNat_lt_size

theorem signed_i64_eq (x : UInt64) :
    signed .i64 x
      = if 9223372036854775808 ≤ x.toNat
        then (x.toNat : Int) - 18446744073709551616
        else (x.toNat : Int) := by
  simp only [signed, AlgorithmLib.IR.ClifTy.width, widthMask,
             ge_iff_le, UInt64.le_iff_toNat_le, mask64,
             show ((9223372036854775808 : UInt64)).toNat = 9223372036854775808 from rfl,
             show (64 ≤ 64) from by decide, if_true,
             show ((1 <<< 64 : Nat) : Int) = 18446744073709551616 from rfl]

theorem signed_i64_of_toNat (k : Int) (n : Nat)
    (hn : (n : Int) = k.emod 18446744073709551616)
    (h0 : -2147483648 ≤ k) (h1 : k < 2147483648) :
    signed .i64 (UInt64.ofNat n) = k := by
  rw [signed_i64_eq, show (UInt64.ofNat n).toNat = n % 2 ^ 64 from by simp]
  have d1 : 0 ≤ k.emod 18446744073709551616 := Int.emod_nonneg k (by decide)
  have d2 : k.emod 18446744073709551616 < 18446744073709551616 :=
    Int.emod_lt_of_pos k (by decide)
  have d3 : 18446744073709551616 * (k / 18446744073709551616)
              + k.emod 18446744073709551616 = k := Int.ediv_add_emod k _
  split <;> omega

theorem signed_i32_mask (x : UInt64) :
    signed .i32 (x &&& 4294967295) = signed .i32 x := by
  rw [signed_i32_eq, signed_i32_eq, mask32,
      Nat.mod_mod_of_dvd _ (Nat.dvd_refl (2 ^ 32))]

theorem signed_i64_mask (x : UInt64) :
    signed .i64 (x &&& 18446744073709551615) = signed .i64 x := by
  rw [signed_i64_eq, signed_i64_eq, mask64]

/-- **The constant arm of `stepPure` is sound against the machine.**

    `constLit` reports `const v` only when `litOk` holds — the literal is
    inside the foldable range and its type is one of the two the model tracks.
    This says that when it does, the word `Sem.ofInt` builds for the same
    literal reads back as exactly `v`, which is the claim `claimHolds` checks
    over a corpus and this proves outright. -/
theorem constLit_sound (t : ClifTy) (v : Int) (h : litOk t v = true) :
    ∃ x, ofInt t v = .sc t x ∧ signed t x = v := by
  have hf : inFold v = true := by
    simp only [litOk, Bool.and_eq_true] at h; exact h.1
  have hb : -2147483648 ≤ v ∧ v < 2147483648 := by
    simp only [inFold, foldableRange, decide_eq_true_eq, Bool.and_eq_true] at hf
    omega
  have hn : ((v.emod (1 <<< 64)).toNat : Int) = v.emod 18446744073709551616 := by
    have hlo : (0 : Int) ≤ v.emod 18446744073709551616 := Int.emod_nonneg v (by decide)
    simpa only [Int.ofNat_eq_natCast] using Int.toNat_of_nonneg hlo
  cases t <;> simp only [litOk, Bool.and_eq_true, and_false, Bool.false_eq_true] at h
  case i32 =>
    exact ⟨_, rfl, by
      rw [show (widthMask .i32) = 4294967295 from rfl, signed_i32_mask]
      exact signed_i32_of_toNat v _ hn hb.1 hb.2⟩
  case i64 =>
    exact ⟨_, rfl, by
      rw [show (widthMask .i64) = 18446744073709551615 from rfl, signed_i64_mask]
      exact signed_i64_of_toNat v _ hn hb.1 hb.2⟩

/-- A tracked type is one of the two the model admits. -/
def TrackedTy (t : ClifTy) : Prop := t = .i32 ∨ t = .i64

/-- A tracked word determines its signed value modulo `2 ^ 32`. -/
theorem congr32_of_signed {t : ClifTy} (ht : TrackedTy t) {b : UInt64} {y : Int}
    (h : signed t b = y) : (b.toNat : Int) % 4294967296 = y % 4294967296 := by
  have hlt : b.toNat < 2 ^ 64 := b.toNat_lt_size
  rcases ht with rfl | rfl
  · rw [signed_i32_eq] at h; split at h <;> omega
  · rw [signed_i64_eq] at h; split at h <;> omega


/-! ### The word a tracked type carries, as a congruence

    Every arm below is the same argument: the machine computes on `UInt64` and
    wraps, the model computes on `Int` and does not, and they agree exactly
    while the result stays inside `foldableRange`.  Stating "the word determines
    the signed value" as a congruence modulo the type's own width turns each arm
    into `Int.add_emod`/`Int.mul_emod` and one `omega`. -/

/-- The modulus at which a tracked type's word determines its signed value. -/
def modOf : ClifTy → Int
  | .i32 => 4294967296
  | _    => 18446744073709551616

theorem congr_of_signed {t : ClifTy} (ht : TrackedTy t) {b : UInt64} {y : Int}
    (h : signed t b = y) : (b.toNat : Int) % modOf t = y % modOf t := by
  have hlt : b.toNat < 2 ^ 64 := b.toNat_lt_size
  rcases ht with rfl | rfl
  · rw [signed_i32_eq] at h; split at h <;> simp only [modOf] <;> omega
  · rw [signed_i64_eq] at h; split at h <;> simp only [modOf] <;> omega

/-- …and the converse, for a value the model reports: inside `foldableRange`
    the congruence pins it. -/
theorem signed_of_congr {t : ClifTy} (ht : TrackedTy t) {w : UInt64} {k : Int}
    (hk : inFold k = true) (hc : (w.toNat : Int) % modOf t = k % modOf t) :
    signed t w = k := by
  have hb : -2147483648 ≤ k ∧ k < 2147483648 := by
    simp only [inFold, foldableRange, decide_eq_true_eq, Bool.and_eq_true] at hk
    omega
  have hlt : w.toNat < 2 ^ 64 := w.toNat_lt_size
  rcases ht with rfl | rfl
  · rw [signed_i32_eq]; simp only [modOf] at hc; split <;> omega
  · rw [signed_i64_eq]; simp only [modOf] at hc; split <;> omega

/-- The type's modulus divides the word's, so a `UInt64` result may be reduced
    at either. -/
theorem modOf_dvd {t : ClifTy} (ht : TrackedTy t) :
    modOf t ∣ (18446744073709551616 : Int) := by
  rcases ht with rfl | rfl
  · exact ⟨4294967296, by decide⟩
  · exact ⟨1, by decide⟩

theorem toNat_mod_pow {t : ClifTy} (ht : TrackedTy t) (w : UInt64) :
    ((w.toNat % 2 ^ 64 : Nat) : Int) % modOf t = (w.toNat : Int) % modOf t := by
  rw [Nat.mod_eq_of_lt w.toNat_lt_size]


theorem signed_mask_of {t : ClifTy} (ht : TrackedTy t) (x : UInt64) :
    signed t (x &&& widthMask t) = signed t x := by
  rcases ht with rfl | rfl
  · exact signed_i32_mask x
  · exact signed_i64_mask x

/-! ### Congruence on its own

    `signed_of_congr` needs `inFold` to turn a congruence into an equality, and
    every `signed_*` lemma below spends that bound.  The congruences themselves
    do not: wrapping is exactly what they survive.  Naming them separately is
    what lets an invariant carry a claim about a value whose *number* is out of
    range — which is the case the additive arms have to reach through. -/

/-- The machine's word represents `k` modulo the type's own width. -/
def Congr (t : ClifTy) (w : UInt64) (k : Int) : Prop :=
  (w.toNat : Int) % modOf t = k % modOf t

theorem congr_add' {t : ClifTy} (ht : TrackedTy t) {a b : UInt64} {x y : Int}
    (ha : Congr t a x) (hb : Congr t b y) : Congr t (a + b) (x + y) := by
  simp only [Congr] at ha hb ⊢
  rw [UInt64.toNat_add]
  show (((a.toNat + b.toNat : Nat) : Int) % ((2 ^ 64 : Nat) : Int)) % modOf t = _
  rw [show (((2 ^ 64 : Nat) : Int)) = 18446744073709551616 from rfl,
      Int.emod_emod_of_dvd _ (modOf_dvd ht), Int.natCast_add,
      Int.add_emod (a.toNat : Int), ha, hb, ← Int.add_emod]

theorem congr_sub' {t : ClifTy} (ht : TrackedTy t) {a b : UInt64} {x y : Int}
    (ha : Congr t a x) (hb : Congr t b y) : Congr t (a - b) (x - y) := by
  simp only [Congr] at ha hb ⊢
  have hble : b.toNat ≤ 2 ^ 64 := Nat.le_of_lt b.toNat_lt_size
  obtain ⟨q, hq⟩ := modOf_dvd ht
  rw [UInt64.toNat_sub]
  show ((((2 ^ 64 - b.toNat) + a.toNat : Nat) : Int) % ((2 ^ 64 : Nat) : Int)) % modOf t = _
  rw [show (((2 ^ 64 : Nat) : Int)) = 18446744073709551616 from rfl,
      Int.emod_emod_of_dvd _ (modOf_dvd ht), Int.natCast_add, Int.ofNat_sub hble,
      show (((2 ^ 64 : Nat) : Int)) = 18446744073709551616 from rfl,
      show (18446744073709551616 : Int) - (b.toNat : Int) + (a.toNat : Int)
         = ((a.toNat : Int) - (b.toNat : Int)) + 18446744073709551616 from by omega,
      hq, Int.add_mul_emod_self_left, Int.sub_emod, ha, hb, ← Int.sub_emod]

/-- Masking to the type's own width changes nothing modulo that width. -/
theorem congr_mask {t : ClifTy} (ht : TrackedTy t) {a : UInt64} {k : Int}
    (ha : Congr t a k) : Congr t (a &&& widthMask t) k := by
  simp only [Congr] at ha ⊢
  rcases ht with rfl | rfl
  · rw [show (widthMask .i32) = 4294967295 from rfl, mask32]
    simp only [modOf] at ha ⊢
    show ((a.toNat % 2 ^ 32 : Nat) : Int) % 4294967296 = _
    rw [show ((2 ^ 32 : Nat)) = 4294967296 from rfl] at *
    omega
  · rw [show (widthMask .i64) = 18446744073709551615 from rfl, mask64]; exact ha

/-- Truncation weakens the modulus, which a congruence survives. -/
theorem congr_reduce32 {t : ClifTy} (ht : TrackedTy t) {a : UInt64} {k : Int}
    (ha : Congr t a k) : Congr .i32 (a &&& widthMask .i32) k := by
  have h32 : Congr .i32 a k := by
    simp only [Congr, modOf] at ha ⊢
    rcases ht with rfl | rfl
    · exact ha
    · simp only [modOf] at ha; omega
  exact congr_mask (Or.inl rfl) h32

/-- The two directions between a congruence and the signed value it pins down.
    One is free; the other spends `inFold`. -/
theorem congr_of_signed' {t : ClifTy} (ht : TrackedTy t) {w : UInt64} {k : Int}
    (h : signed t w = k) : Congr t w k := congr_of_signed ht h

theorem signed_of_congr' {t : ClifTy} (ht : TrackedTy t) {w : UInt64} {k : Int}
    (hk : inFold k = true) (hc : Congr t w k) : signed t w = k :=
  signed_of_congr ht hk hc

/-- …and the same at the type's own width rather than `foldableRange`.  This is
    the bound an `offset` claim carries: the model bounds the *displacement*,
    but whether the sum wraps depends on a runtime base, so what a consumer
    supplies is `inTy` at the width the sum is computed at. -/
theorem signed_of_congr_inTy {t : ClifTy} (ht : TrackedTy t) {w : UInt64} {k : Int}
    (hk : inTy t k = true) (hc : Congr t w k) : signed t w = k := by
  have hlt : w.toNat < 2 ^ 64 := w.toNat_lt_size
  simp only [Congr] at hc
  rcases ht with rfl | rfl
  · simp only [inTy, decide_eq_true_eq, Bool.and_eq_true] at hk
    rw [signed_i32_eq]; simp only [modOf] at hc; split <;> omega
  · simp only [inTy, decide_eq_true_eq, Bool.and_eq_true] at hk
    rw [signed_i64_eq]; simp only [modOf] at hc; split <;> omega

theorem signed_add {t : ClifTy} (ht : TrackedTy t) {a b : UInt64} {x y : Int}
    (ha : signed t a = x) (hb : signed t b = y) (h : inFold (x + y) = true) :
    signed t (a + b) = x + y := by
  refine signed_of_congr ht h ?_
  have hA := congr_of_signed ht ha
  have hB := congr_of_signed ht hb
  rw [UInt64.toNat_add]
  show (((a.toNat + b.toNat : Nat) : Int) % ((2 ^ 64 : Nat) : Int)) % modOf t = _
  rw [show (((2 ^ 64 : Nat) : Int)) = 18446744073709551616 from rfl,
      Int.emod_emod_of_dvd _ (modOf_dvd ht), Int.natCast_add,
      Int.add_emod (a.toNat : Int), hA, hB, ← Int.add_emod]

theorem signed_mul {t : ClifTy} (ht : TrackedTy t) {a b : UInt64} {x y : Int}
    (ha : signed t a = x) (hb : signed t b = y) (h : inFold (x * y) = true) :
    signed t (a * b) = x * y := by
  refine signed_of_congr ht h ?_
  have hA := congr_of_signed ht ha
  have hB := congr_of_signed ht hb
  rw [UInt64.toNat_mul]
  show (((a.toNat * b.toNat : Nat) : Int) % ((2 ^ 64 : Nat) : Int)) % modOf t = _
  rw [show (((2 ^ 64 : Nat) : Int)) = 18446744073709551616 from rfl,
      Int.emod_emod_of_dvd _ (modOf_dvd ht), Int.natCast_mul,
      Int.mul_emod (a.toNat : Int), hA, hB, ← Int.mul_emod]

theorem signed_sub {t : ClifTy} (ht : TrackedTy t) {a b : UInt64} {x y : Int}
    (ha : signed t a = x) (hb : signed t b = y) (h : inFold (x - y) = true) :
    signed t (a - b) = x - y := by
  refine signed_of_congr ht h ?_
  have hA := congr_of_signed ht ha
  have hB := congr_of_signed ht hb
  have hble : b.toNat ≤ 2 ^ 64 := Nat.le_of_lt b.toNat_lt_size
  obtain ⟨q, hq⟩ := modOf_dvd ht
  rw [UInt64.toNat_sub]
  show ((((2 ^ 64 - b.toNat) + a.toNat : Nat) : Int) % ((2 ^ 64 : Nat) : Int)) % modOf t = _
  rw [show (((2 ^ 64 : Nat) : Int)) = 18446744073709551616 from rfl,
      Int.emod_emod_of_dvd _ (modOf_dvd ht), Int.natCast_add, Int.ofNat_sub hble,
      show (((2 ^ 64 : Nat) : Int)) = 18446744073709551616 from rfl,
      show (18446744073709551616 : Int) - (b.toNat : Int) + (a.toNat : Int)
         = ((a.toNat : Int) - (b.toNat : Int)) + 18446744073709551616 from by omega,
      hq, Int.add_mul_emod_self_left, Int.sub_emod, hA, hB, ← Int.sub_emod]

theorem signed_zero {t : ClifTy} (ht : TrackedTy t) : signed t 0 = 0 := by
  rcases ht with rfl | rfl
  · rw [signed_i32_eq]; decide
  · rw [signed_i64_eq]; decide

theorem signed_neg {t : ClifTy} (ht : TrackedTy t) {a : UInt64} {x : Int}
    (ha : signed t a = x) (h : inFold (-x) = true) : signed t (0 - a) = -x := by
  have hz : inFold (0 - x) = true := by rw [Int.zero_sub]; exact h
  have := signed_sub ht (signed_zero ht) ha hz
  rw [Int.zero_sub] at this; exact this

/-- A shift by `s` is a multiplication by `2 ^ s`, modulo the word. -/
theorem signed_shl {t : ClifTy} (ht : TrackedTy t) {a : UInt64} {x : Int} {s : Nat}
    (hs : s < 64) (ha : signed t a = x) (h : inFold (x * 2 ^ s) = true) :
    signed t (a <<< UInt64.ofNat s) = x * 2 ^ s := by
  refine signed_of_congr ht h ?_
  have hA := congr_of_signed ht ha
  have hsm : (UInt64.ofNat s).toNat % 64 = s := by
    rw [show (UInt64.ofNat s).toNat = s % 2 ^ 64 from by simp,
        Nat.mod_eq_of_lt (Nat.lt_trans hs (by decide)), Nat.mod_eq_of_lt hs]
  rw [UInt64.toNat_shiftLeft, hsm, Nat.shiftLeft_eq]
  show (((a.toNat * 2 ^ s : Nat) : Int) % ((2 ^ 64 : Nat) : Int)) % modOf t = _
  rw [show (((2 ^ 64 : Nat) : Int)) = 18446744073709551616 from rfl,
      Int.emod_emod_of_dvd _ (modOf_dvd ht), Int.natCast_mul,
      show (((2 ^ s : Nat) : Int)) = (2 : Int) ^ s from by simp,
      Int.mul_emod (a.toNat : Int), hA, ← Int.mul_emod]

/-- A non-negative tracked value's masked word *is* that value. -/
theorem masked_toNat {t : ClifTy} (ht : TrackedTy t) {a : UInt64} {x : Int}
    (ha : signed t a = x) (h0 : 0 ≤ x) (hf : inFold x = true) :
    (a &&& widthMask t).toNat = x.toNat := by
  have hb : x < 2147483648 := by
    simp only [inFold, foldableRange, decide_eq_true_eq, Bool.and_eq_true] at hf; omega
  have hx : ((x.toNat : Nat) : Int) = x := Int.toNat_of_nonneg h0
  have hlt : a.toNat < 2 ^ 64 := a.toNat_lt_size
  rcases ht with rfl | rfl
  · have hm : (a &&& widthMask .i32).toNat = a.toNat % 2 ^ 32 := by
      rw [show (widthMask .i32) = 4294967295 from rfl]; exact mask32 a
    rw [signed_i32_eq] at ha
    rw [hm]; split at ha <;> omega
  · have hm : (a &&& widthMask .i64).toNat = a.toNat := by
      rw [show (widthMask .i64) = 18446744073709551615 from rfl]; exact mask64 a
    rw [signed_i64_eq] at ha
    rw [hm]; split at ha <;> omega

/-- A logical shift right is division, on a non-negative operand. -/
theorem signed_ushr {t : ClifTy} (ht : TrackedTy t) {a : UInt64} {x : Int} {s : Nat}
    (hs : s < 64) (ha : signed t a = x) (h0 : 0 ≤ x) (hf : inFold x = true) :
    signed t ((a &&& widthMask t) >>> UInt64.ofNat s) = x / 2 ^ s := by
  have hb : x < 2147483648 := by
    simp only [inFold, foldableRange, decide_eq_true_eq, Bool.and_eq_true] at hf; omega
  have hpos : (0 : Int) < 2 ^ s := by
    have hn : (0 : Nat) < 2 ^ s := Nat.two_pow_pos s
    rw [show ((2 : Int) ^ s) = ((2 ^ s : Nat) : Int) from by simp]; omega
  have hdiv0 : 0 ≤ x / 2 ^ s := Int.ediv_nonneg h0 (Int.le_of_lt hpos)
  have hdivlt : x / 2 ^ s ≤ x := Int.ediv_le_self _ h0
  have hres : inFold (x / 2 ^ s) = true := by
    simp only [inFold, foldableRange, decide_eq_true_eq, Bool.and_eq_true]; omega
  refine signed_of_congr ht hres ?_
  have hsm : (UInt64.ofNat s).toNat % 64 = s := by
    rw [show (UInt64.ofNat s).toNat = s % 2 ^ 64 from by simp,
        Nat.mod_eq_of_lt (Nat.lt_trans hs (by decide)), Nat.mod_eq_of_lt hs]
  rw [UInt64.toNat_shiftRight, hsm, masked_toNat ht ha h0 hf,
      Nat.shiftRight_eq_div_pow]
  show ((x.toNat : Int) / ((2 ^ s : Nat) : Int)) % modOf t = _
  rw [Int.toNat_of_nonneg h0, show (((2 ^ s : Nat) : Int)) = (2 : Int) ^ s from by simp]

/-- The word `ofInt` builds for a foldable literal reads back as that literal. -/
theorem signed_ofInt {t : ClifTy} (ht : TrackedTy t) {k : Int} (h : inFold k = true) :
    ∃ w, ofInt t k = .sc t w ∧ signed t w = k := by
  refine constLit_sound t k ?_
  rcases ht with rfl | rfl <;> simp only [litOk, h, Bool.true_and]

/-- Narrowing to `i32` keeps a foldable value: the range is exact there. -/
theorem signed_reduce32 {t : ClifTy} (ht : TrackedTy t) {a : UInt64} {k : Int}
    (ha : signed t a = k) (hf : inFold k = true) :
    signed .i32 (a &&& widthMask .i32) = k := by
  rw [show (widthMask .i32) = 4294967295 from rfl, signed_i32_mask]
  refine signed_of_congr (Or.inl rfl) hf ?_
  have hA := congr_of_signed ht ha
  rcases ht with rfl | rfl
  · exact hA
  · simp only [modOf] at hA ⊢
    rw [show (4294967296 : Int) = 4294967296 from rfl]
    omega

/-- Zero-extension keeps a *non-negative* value; a negative one it does not,
    which is why `stepPure` refuses that case rather than passing it through. -/
theorem signed_uextend64 {t : ClifTy} (ht : TrackedTy t) {a : UInt64} {k : Int}
    (ha : signed t a = k) (h0 : 0 ≤ k) (hf : inFold k = true) :
    signed .i64 (a &&& widthMask t) = k := by
  have hm := masked_toNat ht ha h0 hf
  have hb : k < 2147483648 := by
    simp only [inFold, foldableRange, decide_eq_true_eq, Bool.and_eq_true] at hf; omega
  refine signed_of_congr (Or.inr rfl) hf ?_
  rw [hm, Int.toNat_of_nonneg h0]

/-! ### The invariant, and that one step preserves it -/

/-- **What a `const` claim owes the machine.**

    `claimHolds` checks the middle clause alone — its `const` arm reads
    `signed t x == k` with `t` unconstrained — so a narrow-typed constant would
    have passed the corpus check.  It cannot arise, because `litOk` refuses
    narrow literals and the retagging arms yield `i32` or `i64`, but that is a
    fact about `stepPure` and belongs in the invariant rather than in a comment.
    `inFold k` is here for the same reason: every fold guards on it, and every
    arm below needs it of its operands. -/
def Agree (vs : Vals) (e : Env) : Prop :=
  ∀ v k, e v = .const k →
    ∃ t w, getV vs v = some (.sc t w) ∧ TrackedTy t ∧ signed t w = k ∧ inFold k = true

/-- Binding a value leaves every value already bound alone. Stated of a slot
    that is already there, which is the only case the invariant reaches: a slot
    past the end reads as `none`, and no `const` claim can be owed of it. -/
theorem getV_setV_ne {vs : Vals} {d w : Val} {x u : V}
    (hne : w.id ≠ d.id) (h : getV vs w = some u) : getV (setV vs d x) w = some u := by
  simp only [getV] at h
  have hlt : w.id < vs.size := (Array.getElem?_eq_some_iff.mp h).1
  simp only [getV, setV, Array.set!, Array.getElem?_setIfInBounds,
             if_neg (Ne.symm hne)]
  rcases Nat.lt_or_ge d.id vs.size with hd | hd
  · simp only [hd, if_pos]; exact h
  · simp only [Nat.not_lt.mpr hd, if_false, Array.getElem?_append_left hlt]; exact h

/-- Every instruction that computes a value writes the destination its own
    syntax names. -/
theorem evalInst_dest {m : Mem} {vs : Vals} {i : Inst} {d : Val} {x : V}
    (h : evalInst m vs i = some (d, x)) : Inst.destOf? i = some d := by
  cases i <;>
    simp_all [evalInst, Inst.destOf?, evalInst.bin, evalInst.un,
              Option.bind_eq_some_iff, Prod.mk.injEq, Option.some.injEq] <;>
    grind

/-- **One step preserves the invariant, outside the tracked arithmetic.**

    Two halves, and both are about omission rather than arithmetic: an
    instruction the model does not compute binds its destination `unknown`,
    which claims nothing, and it leaves every slot it does not write alone.
    `evalInst_dest` is what ties those to the machine — every instruction that
    produces a value is one `destOf?` names. -/
theorem const_sound_untracked {m : Mem} {vs : Vals} {e : Env} {i : Inst} {d : Val} {x : V}
    (ht : Inst.TrackedB i = false) (hag : Agree vs e)
    (hev : evalInst m vs i = some (d, x)) :
    Agree (setV vs d x) (stepPure e i) := by
  intro v k hv
  have hd := evalInst_dest hev
  by_cases hvd : v.id = d.id
  · have hveq : v = d := by cases v; cases d; simp_all
    rw [hveq, stepPure_untracked i d e hd ht] at hv
    exact absurd hv (by simp)
  · have hframe : stepPure e i v = e v :=
      stepPure_frame i e v (fun d' hd' => by rw [hd] at hd'; cases hd'; exact hvd)
    rw [hframe] at hv
    obtain ⟨t0, w, hw, htt, hs, hf⟩ := hag v k hv
    exact ⟨t0, w, getV_setV_ne hvd hw, htt, hs, hf⟩

/-- **…and on the literal arm.**  `constLit` reports a constant only when
    `litOk` holds, which is exactly the invariant's two side conditions: the
    type is tracked and the value is foldable. -/
theorem const_sound_iconst {m : Mem} {vs : Vals} {e : Env} {d dd : Val}
    {t0 : ClifTy} {kk : Int} {x : V}
    (hag : Agree vs e) (hev : evalInst m vs (.iconst d t0 kk) = some (dd, x)) :
    Agree (setV vs dd x) (stepPure e (.iconst d t0 kk)) := by
  have hdd : dd = d ∧ x = ofInt t0 kk := by
    simp [evalInst, AlgorithmLib.HProg.Blocks.viaOp, evalOp] at hev
    exact ⟨hev.1.symm, hev.2.symm⟩
  obtain ⟨h1, h2⟩ := hdd
  subst h2
  subst h1
  intro v k hv
  by_cases hvd : v.id = dd.id
  · have hveq : v = dd := by cases v; cases dd; simp_all
    rw [hveq, stepPure, Env.set_eq _ _ _ _ rfl] at hv
    by_cases hok : litOk t0 kk = true
    · rw [constLit_eq hok] at hv
      have hk : kk = k := by injection hv
      subst hk
      obtain ⟨w, hofI, hsg⟩ := constLit_sound t0 kk hok
      have htt : TrackedTy t0 := by
        cases t0 <;> simp only [litOk, Bool.and_eq_true, and_false, Bool.false_eq_true] at hok
        · exact Or.inl rfl
        · exact Or.inr rfl
      have hf : inFold kk = true := by
        simp only [litOk, Bool.and_eq_true] at hok; exact hok.1
      refine ⟨t0, w, ?_, htt, hsg, hf⟩
      rw [hveq, hofI]
      simp only [getV, setV, Array.set!, Array.getElem?_setIfInBounds, if_pos rfl]
      rcases Nat.lt_or_ge dd.id vs.size with hd | hd
      · simp only [hd, if_pos]
      · simp only [Nat.not_lt.mpr hd, if_false, Array.size_append, Array.size_replicate]
        simp only [if_pos (show dd.id < vs.size + (dd.id + 1 - vs.size) by omega), if_true]
    · rw [constLit_ne (by simpa using hok)] at hv; exact absurd hv (by simp)
  · rw [stepPure_frame _ e v (fun d' hd' => by
        simp only [Inst.destOf?, Option.some.injEq] at hd'; exact hd' ▸ hvd)] at hv
    obtain ⟨t1, w, hw, htt, hs, hf⟩ := hag v k hv
    exact ⟨t1, w, getV_setV_ne hvd hw, htt, hs, hf⟩

/-- Two tracked types that compare equal are equal. `ClifTy` derives `BEq`
    but not `DecidableEq`, so the guard `Op.check` and `Sem.bin` share has to be
    turned into an equation by hand. -/
theorem ty_eq_of_beq {ta tb : ClifTy} (h : (ta == tb) = true) : ta = tb := by
  cases ta <;> cases tb <;> first | rfl | exact absurd h (by decide)

/-- **What an `iadd` that computed a value tells us**: both operands were
    scalars of one integer type, and the result is their sum at that width.
    The type agreement is not an assumption here — `Sem.bin` refuses without
    it, so the machine having answered is what supplies it. -/
theorem evalInst_iadd_inv {m : Mem} {vs : Vals} {d a b dd : Val} {x : V}
    (h : evalInst m vs (.iadd d a b) = some (dd, x)) :
    ∃ t wa wb, getV vs a = some (.sc t wa) ∧ getV vs b = some (.sc t wb)
      ∧ t.isInt = true ∧ dd = d ∧ x = .sc t ((wa + wb) &&& widthMask t) := by
  rcases hga : getV vs a with _ | u <;> rcases hgb : getV vs b with _ | v <;>
    simp [evalInst, evalInst.bin, AlgorithmLib.HProg.Blocks.viaOp, evalOp,
          AlgorithmLib.HProg.Sem.bin, AlgorithmLib.HProg.Sem.get, hga, hgb] at h
  cases u with
  | vec => simp at h
  | sc ta wa =>
    cases v with
    | vec => simp at h
    | sc tb wb =>
      simp only [] at h
      by_cases hc : (ta == tb) = true ∧ ta.isInt = true
      · rw [if_pos hc] at h
        simp only [Option.bind_some, Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨ht, hb⟩ := hc
        have ht' : ta = tb := ty_eq_of_beq ht
        subst ht'
        exact ⟨ta, wa, wb, rfl, rfl, hb, h.1.symm, by first | (rw [← h.2]; rfl) | rw [← h.2]⟩
      · rw [if_neg hc] at h; simp at h

/-- The same, for `isub`. -/
theorem evalInst_isub_inv {m : Mem} {vs : Vals} {d a b dd : Val} {x : V}
    (h : evalInst m vs (.isub d a b) = some (dd, x)) :
    ∃ t wa wb, getV vs a = some (.sc t wa) ∧ getV vs b = some (.sc t wb)
      ∧ t.isInt = true ∧ dd = d ∧ x = .sc t ((wa - wb) &&& widthMask t) := by
  rcases hga : getV vs a with _ | u <;> rcases hgb : getV vs b with _ | v <;>
    simp [evalInst, evalInst.bin, AlgorithmLib.HProg.Blocks.viaOp, evalOp,
          AlgorithmLib.HProg.Sem.bin, AlgorithmLib.HProg.Sem.get, hga, hgb] at h
  cases u with
  | vec => simp at h
  | sc ta wa =>
    cases v with
    | vec => simp at h
    | sc tb wb =>
      simp only [] at h
      by_cases hc : (ta == tb) = true ∧ ta.isInt = true
      · rw [if_pos hc] at h
        simp only [Option.bind_some, Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨ht, hb⟩ := hc
        have ht' : ta = tb := ty_eq_of_beq ht
        subst ht'
        exact ⟨ta, wa, wb, rfl, rfl, hb, h.1.symm, by first | (rw [← h.2]; rfl) | rw [← h.2]⟩
      · rw [if_neg hc] at h; simp at h

/-- The same, for `imul`. -/
theorem evalInst_imul_inv {m : Mem} {vs : Vals} {d a b dd : Val} {x : V}
    (h : evalInst m vs (.imul d a b) = some (dd, x)) :
    ∃ t wa wb, getV vs a = some (.sc t wa) ∧ getV vs b = some (.sc t wb)
      ∧ t.isInt = true ∧ dd = d ∧ x = .sc t ((wa * wb) &&& widthMask t) := by
  rcases hga : getV vs a with _ | u <;> rcases hgb : getV vs b with _ | v <;>
    simp [evalInst, evalInst.bin, AlgorithmLib.HProg.Blocks.viaOp, evalOp,
          AlgorithmLib.HProg.Sem.bin, AlgorithmLib.HProg.Sem.get, hga, hgb] at h
  cases u with
  | vec => simp at h
  | sc ta wa =>
    cases v with
    | vec => simp at h
    | sc tb wb =>
      simp only [] at h
      by_cases hc : (ta == tb) = true ∧ ta.isInt = true
      · rw [if_pos hc] at h
        simp only [Option.bind_some, Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨ht, hb⟩ := hc
        have ht' : ta = tb := ty_eq_of_beq ht
        subst ht'
        exact ⟨ta, wa, wb, rfl, rfl, hb, h.1.symm, by first | (rw [← h.2]; rfl) | rw [← h.2]⟩
      · rw [if_neg hc] at h; simp at h

/-- **What a `ishl` that computed a value tells us.**  The shift is the one
    binary integer operation whose operands may differ in width: the amount is
    taken at any integer type and reduced modulo the shifted operand's. -/
theorem evalInst_ishl_inv {m : Mem} {vs : Vals} {d a b dd : Val} {x : V}
    (h : evalInst m vs (.ishl d a b) = some (dd, x)) :
    ∃ ta tb wa wb, getV vs a = some (.sc ta wa) ∧ getV vs b = some (.sc tb wb)
      ∧ ta.isInt = true ∧ tb.isInt = true ∧ dd = d
      ∧ x = .sc ta (wa <<< (wb % UInt64.ofNat ta.width) &&& widthMask ta) := by
  rcases hga : getV vs a with _ | u <;> rcases hgb : getV vs b with _ | v <;>
    simp [evalInst, evalInst.bin, AlgorithmLib.HProg.Blocks.viaOp, evalOp,
          AlgorithmLib.HProg.Sem.shiftBin, AlgorithmLib.HProg.Sem.get, hga, hgb] at h
  cases u with
  | vec => simp at h
  | sc ta wa =>
    cases v with
    | vec => simp at h
    | sc tb wb =>
      simp only [] at h
      by_cases hc : ta.isInt = true ∧ tb.isInt = true
      · rw [if_pos hc] at h
        simp only [Option.bind_some, Option.some.injEq, Prod.mk.injEq] at h
        exact ⟨ta, tb, wa, wb, rfl, rfl, hc.1, hc.2, h.1.symm, by first | (rw [← h.2]; rfl) | rw [← h.2]⟩
      · rw [if_neg hc] at h; simp at h

/-- **What a `ushr` that computed a value tells us.**  The shift is the one
    binary integer operation whose operands may differ in width: the amount is
    taken at any integer type and reduced modulo the shifted operand's. -/
theorem evalInst_ushr_inv {m : Mem} {vs : Vals} {d a b dd : Val} {x : V}
    (h : evalInst m vs (.ushr d a b) = some (dd, x)) :
    ∃ ta tb wa wb, getV vs a = some (.sc ta wa) ∧ getV vs b = some (.sc tb wb)
      ∧ ta.isInt = true ∧ tb.isInt = true ∧ dd = d
      ∧ x = .sc ta ((wa &&& widthMask ta) >>> (wb % UInt64.ofNat ta.width) &&& widthMask ta) := by
  rcases hga : getV vs a with _ | u <;> rcases hgb : getV vs b with _ | v <;>
    simp [evalInst, evalInst.bin, AlgorithmLib.HProg.Blocks.viaOp, evalOp,
          AlgorithmLib.HProg.Sem.shiftBin, AlgorithmLib.HProg.Sem.get, hga, hgb] at h
  cases u with
  | vec => simp at h
  | sc ta wa =>
    cases v with
    | vec => simp at h
    | sc tb wb =>
      simp only [] at h
      by_cases hc : ta.isInt = true ∧ tb.isInt = true
      · rw [if_pos hc] at h
        simp only [Option.bind_some, Option.some.injEq, Prod.mk.injEq] at h
        exact ⟨ta, tb, wa, wb, rfl, rfl, hc.1, hc.2, h.1.symm, by first | (rw [← h.2]; rfl) | rw [← h.2]⟩
      · rw [if_neg hc] at h; simp at h

/-- The same, for `ineg`. -/
theorem evalInst_ineg_inv {m : Mem} {vs : Vals} {d a dd : Val} {x : V}
    (h : evalInst m vs (.ineg d a) = some (dd, x)) :
    ∃ t w, getV vs a = some (.sc t w) ∧ t.isInt = true ∧ dd = d ∧ x = .sc t ((0 - w) &&& widthMask t) := by
  rcases hga : getV vs a with _ | u <;>
    simp [evalInst, evalInst.un, AlgorithmLib.HProg.Blocks.viaOp, evalOp,
          AlgorithmLib.HProg.Sem.un, AlgorithmLib.HProg.Sem.get, hga] at h
  cases u with
  | vec => simp at h
  | sc t w =>
    simp only [] at h
    by_cases hc : t.isInt = true
    · rw [if_pos hc] at h
      simp only [Option.bind_some, Option.some.injEq, Prod.mk.injEq] at h
      exact ⟨t, w, rfl, hc, h.1.symm, by first | (rw [← h.2]; rfl) | rw [← h.2]⟩
    · rw [if_neg hc] at h; simp at h

/-- The same, for `ireduce32`. -/
theorem evalInst_ireduce32_inv {m : Mem} {vs : Vals} {d a dd : Val} {x : V}
    (h : evalInst m vs (.ireduce32 d a) = some (dd, x)) :
    ∃ t w, getV vs a = some (.sc t w) ∧ t.isInt = true ∧ t.width > 32 ∧ dd = d ∧ x = .sc ClifTy.i32 (w &&& widthMask ClifTy.i32) := by
  rcases hga : getV vs a with _ | u <;>
    simp [evalInst, evalInst.un, AlgorithmLib.HProg.Blocks.viaOp, evalOp,
          AlgorithmLib.HProg.Sem.un, AlgorithmLib.HProg.Sem.get, hga] at h
  cases u with
  | vec => simp at h
  | sc t w =>
    simp only [] at h
    by_cases hc : t.isInt = true ∧ t.width > 32
    · rw [if_pos hc] at h
      simp only [Option.bind_some, Option.some.injEq, Prod.mk.injEq] at h
      exact ⟨t, w, rfl, hc.1, hc.2, h.1.symm, by first | (rw [← h.2]; rfl) | rw [← h.2]⟩
    · rw [if_neg hc] at h; simp at h

/-- The same, for `uextend64`. -/
theorem evalInst_uextend64_inv {m : Mem} {vs : Vals} {d a dd : Val} {x : V}
    (h : evalInst m vs (.uextend64 d a) = some (dd, x)) :
    ∃ t w, getV vs a = some (.sc t w) ∧ t.isInt = true ∧ t.width < 64 ∧ dd = d ∧ x = .sc ClifTy.i64 (w &&& widthMask t) := by
  rcases hga : getV vs a with _ | u <;>
    simp [evalInst, evalInst.un, AlgorithmLib.HProg.Blocks.viaOp, evalOp,
          AlgorithmLib.HProg.Sem.un, AlgorithmLib.HProg.Sem.get, hga] at h
  cases u with
  | vec => simp at h
  | sc t w =>
    simp only [] at h
    by_cases hc : t.isInt = true ∧ t.width < 64
    · rw [if_pos hc] at h
      simp only [Option.bind_some, Option.some.injEq, Prod.mk.injEq] at h
      exact ⟨t, w, rfl, hc.1, hc.2, h.1.symm, by first | (rw [← h.2]; rfl) | rw [← h.2]⟩
    · rw [if_neg hc] at h; simp at h

/-- The same, for `sextend64`. -/
theorem evalInst_sextend64_inv {m : Mem} {vs : Vals} {d a dd : Val} {x : V}
    (h : evalInst m vs (.sextend64 d a) = some (dd, x)) :
    ∃ t w, getV vs a = some (.sc t w) ∧ t.isInt = true ∧ t.width < 64 ∧ dd = d ∧ x = ofInt ClifTy.i64 (signed t w) := by
  rcases hga : getV vs a with _ | u <;>
    simp [evalInst, evalInst.un, AlgorithmLib.HProg.Blocks.viaOp, evalOp,
          AlgorithmLib.HProg.Sem.un, AlgorithmLib.HProg.Sem.get, hga] at h
  cases u with
  | vec => simp at h
  | sc t w =>
    simp only [] at h
    by_cases hc : t.isInt = true ∧ t.width < 64
    · rw [if_pos hc] at h
      simp only [Option.bind_some, Option.some.injEq, Prod.mk.injEq] at h
      exact ⟨t, w, rfl, hc.1, hc.2, h.1.symm, by first | (rw [← h.2]; rfl) | rw [← h.2]⟩
    · rw [if_neg hc] at h; simp at h

/-- **What a `load` that produced a value tells us.**  Every kind returns a
    scalar of the type the access names; only a vector type gives something
    else. -/
theorem evalInst_load_inv {m : Mem} {vs : Vals} {d a dd : Val} {op : LoadOp} {x : V}
    (h : evalInst m vs (.load d op a) = some (dd, x)) (hl : op.ty.lanes = none) :
    dd = d ∧ ∃ w, x = .sc op.ty w := by
  rcases hga : getV vs a with _ | u <;>
    simp [evalInst, evalInst.un, AlgorithmLib.HProg.Blocks.viaOp, evalOp,
          AlgorithmLib.HProg.Sem.get, hga] at h
  cases u with
  | vec => simp at h
  | sc t w =>
    simp only [] at h
    cases hk : op.kind <;> rw [hk] at h <;> simp only [] at h
    case plain =>
      rw [hl] at h
      simp only [] at h
      rcases hb : Mem.load m w (tyBytes op.ty) with _ | b <;> rw [hb] at h <;> simp at h
      exact ⟨h.1.symm, _, h.2.symm⟩
    case uload8 =>
      rcases hb : Mem.load m w 1 with _ | b <;> rw [hb] at h <;> simp at h
      exact ⟨h.1.symm, _, by rw [← h.2]; rfl⟩
    case uload32 =>
      rcases hb : Mem.load m w 4 with _ | b <;> rw [hb] at h <;> simp at h
      exact ⟨h.1.symm, _, by rw [← h.2]; rfl⟩
    case sload8 =>
      rcases hb : Mem.load m w 1 with _ | b <;> rw [hb] at h <;> simp at h
      exact ⟨h.1.symm, _, by rw [← h.2]; rfl⟩

/-- `addSym` reports a constant only by folding two of them. -/
theorem addSym_const {e : Env} {a b : Val} {k : Int} (h : addSym e a b = .const k) :
    ∃ xa xb, e a = .const xa ∧ e b = .const xb ∧ constIf (xa + xb) = .const k := by
  unfold addSym at h
  cases hea : e a <;> cases heb : e b <;> rw [hea, heb] at h <;>
    first
      | exact ⟨_, _, rfl, rfl, h⟩
      | (exfalso; revert h; simp only [offsetIf, constIf]; split <;> simp)
      | (exfalso; revert h; simp)

/-- A `constIf` that reported a constant reported the value it was given, and
    that value is foldable. -/
theorem constIf_const {v k : Int} (h : constIf v = .const k) :
    v = k ∧ inFold v = true := by
  unfold constIf at h
  split at h
  · exact ⟨by injection h, by assumption⟩
  · exact absurd h (by simp)

/-- **The `iadd` arm.**  The two operands' types are not assumed to agree — the
    machine refuses a mixed-width `iadd`, so `hev` is what supplies it. -/
theorem const_sound_iadd {m : Mem} {vs : Vals} {e : Env} {d a b dd : Val} {x : V}
    (hag : Agree vs e) (hev : evalInst m vs (.iadd d a b) = some (dd, x)) :
    Agree (setV vs dd x) (stepPure e (.iadd d a b)) := by
  obtain ⟨t, wa, wb, hga, hgb, _, hdd, hx⟩ := evalInst_iadd_inv hev
  subst hdd
  intro v k hv
  by_cases hvd : v.id = dd.id
  · have hveq : v = dd := by cases v; cases dd; simp_all
    rw [hveq, stepPure, Env.set_eq _ _ _ _ rfl] at hv
    obtain ⟨xa, xb, hea, heb, hfold⟩ := addSym_const hv
    obtain ⟨hsum, hin⟩ := constIf_const hfold
    obtain ⟨ta, wa', hva, hta, hsa, _⟩ := hag a xa hea
    obtain ⟨tb, wb', hvb, _, hsb, _⟩ := hag b xb heb
    rw [hga] at hva; rw [hgb] at hvb
    have hta' : TrackedTy t := by
      injection hva with h1; injection h1 with h2 _; rw [h2]; exact hta
    have hsa' : signed t wa = xa := by
      injection hva with h1; injection h1 with h2 h3; rw [h2, h3]; exact hsa
    have hsb' : signed t wb = xb := by
      injection hvb with h1; injection h1 with h2 h3; rw [h2, h3]; exact hsb
    refine ⟨t, (wa + wb) &&& widthMask t, ?_, hta', ?_, hsum ▸ hin⟩
    · rw [hveq, hx]; exact AlgorithmLib.HProg.getV_setV_self vs dd _
    · rw [signed_mask_of hta', signed_add hta' hsa' hsb' hin, hsum]
  · rw [stepPure_frame _ e v (fun d' hd' => by
        simp only [Inst.destOf?, Option.some.injEq] at hd'; exact hd' ▸ hvd)] at hv
    obtain ⟨t1, w, hw, htt, hs, hf⟩ := hag v k hv
    exact ⟨t1, w, getV_setV_ne hvd hw, htt, hs, hf⟩

/-- `mulSym` reports a constant only by folding two of them. -/
theorem mulSym_const {e : Env} {a b : Val} {k : Int} (h : mulSym e a b = .const k) :
    ∃ xa xb, e a = .const xa ∧ e b = .const xb ∧ constIf (xa * xb) = .const k := by
  unfold mulSym at h
  cases hea : e a <;> cases heb : e b <;> rw [hea, heb] at h <;>
    first | exact ⟨_, _, rfl, rfl, h⟩ | (exfalso; revert h; simp)

/-- The `isub` arm folds two constants; its other two arms name an expression
    or a displacement, never a number. -/
theorem stepPure_isub_const {e : Env} {d a b : Val} {k : Int}
    (h : stepPure e (.isub d a b) d = .const k) :
    ∃ xa xb, e a = .const xa ∧ e b = .const xb ∧ constIf (xa - xb) = .const k := by
  rw [stepPure, Env.set_eq _ _ _ _ rfl] at h
  cases hea : e a <;> cases heb : e b <;> rw [hea, heb] at h <;>
    first
      | exact ⟨_, _, rfl, rfl, h⟩
      | (exfalso; revert h; simp only [offsetIf, constIf]; split <;> simp)
      | (exfalso; revert h; simp)

theorem const_sound_isub {m : Mem} {vs : Vals} {e : Env} {d a b dd : Val} {x : V}
    (hag : Agree vs e) (hev : evalInst m vs (.isub d a b) = some (dd, x)) :
    Agree (setV vs dd x) (stepPure e (.isub d a b)) := by
  obtain ⟨t, wa, wb, hga, hgb, _, hdd, hx⟩ := evalInst_isub_inv hev
  subst hdd
  intro v k hv
  by_cases hvd : v.id = dd.id
  · have hveq : v = dd := by cases v; cases dd; simp_all
    rw [hveq] at hv
    obtain ⟨xa, xb, hea, heb, hfold⟩ := stepPure_isub_const hv
    obtain ⟨hsum, hin⟩ := constIf_const hfold
    obtain ⟨ta, wa', hva, hta, hsa, _⟩ := hag a xa hea
    obtain ⟨tb, wb', hvb, _, hsb, _⟩ := hag b xb heb
    rw [hga] at hva; rw [hgb] at hvb
    have hta' : TrackedTy t := by
      injection hva with h1; injection h1 with h2 _; rw [h2]; exact hta
    have hsa' : signed t wa = xa := by
      injection hva with h1; injection h1 with h2 h3; rw [h2, h3]; exact hsa
    have hsb' : signed t wb = xb := by
      injection hvb with h1; injection h1 with h2 h3; rw [h2, h3]; exact hsb
    refine ⟨t, (wa - wb) &&& widthMask t, ?_, hta', ?_, hsum ▸ hin⟩
    · rw [hveq, hx]; exact AlgorithmLib.HProg.getV_setV_self vs dd _
    · rw [signed_mask_of hta', signed_sub hta' hsa' hsb' hin, hsum]
  · rw [stepPure_frame _ e v (fun d' hd' => by
        simp only [Inst.destOf?, Option.some.injEq] at hd'; exact hd' ▸ hvd)] at hv
    obtain ⟨t1, w, hw, htt, hs, hf⟩ := hag v k hv
    exact ⟨t1, w, getV_setV_ne hvd hw, htt, hs, hf⟩

theorem const_sound_imul {m : Mem} {vs : Vals} {e : Env} {d a b dd : Val} {x : V}
    (hag : Agree vs e) (hev : evalInst m vs (.imul d a b) = some (dd, x)) :
    Agree (setV vs dd x) (stepPure e (.imul d a b)) := by
  obtain ⟨t, wa, wb, hga, hgb, _, hdd, hx⟩ := evalInst_imul_inv hev
  subst hdd
  intro v k hv
  by_cases hvd : v.id = dd.id
  · have hveq : v = dd := by cases v; cases dd; simp_all
    rw [hveq, stepPure, Env.set_eq _ _ _ _ rfl] at hv
    obtain ⟨xa, xb, hea, heb, hfold⟩ := mulSym_const hv
    obtain ⟨hsum, hin⟩ := constIf_const hfold
    obtain ⟨ta, wa', hva, hta, hsa, _⟩ := hag a xa hea
    obtain ⟨tb, wb', hvb, _, hsb, _⟩ := hag b xb heb
    rw [hga] at hva; rw [hgb] at hvb
    have hta' : TrackedTy t := by
      injection hva with h1; injection h1 with h2 _; rw [h2]; exact hta
    have hsa' : signed t wa = xa := by
      injection hva with h1; injection h1 with h2 h3; rw [h2, h3]; exact hsa
    have hsb' : signed t wb = xb := by
      injection hvb with h1; injection h1 with h2 h3; rw [h2, h3]; exact hsb
    refine ⟨t, (wa * wb) &&& widthMask t, ?_, hta', ?_, hsum ▸ hin⟩
    · rw [hveq, hx]; exact AlgorithmLib.HProg.getV_setV_self vs dd _
    · rw [signed_mask_of hta', signed_mul hta' hsa' hsb' hin, hsum]
  · rw [stepPure_frame _ e v (fun d' hd' => by
        simp only [Inst.destOf?, Option.some.injEq] at hd'; exact hd' ▸ hvd)] at hv
    obtain ⟨t1, w, hw, htt, hs, hf⟩ := hag v k hv
    exact ⟨t1, w, getV_setV_ne hvd hw, htt, hs, hf⟩

/-- The `ineg` arm folds a constant; nothing else reports one. -/
theorem stepPure_ineg_const {e : Env} {d a : Val} {k : Int}
    (h : stepPure e (.ineg d a) d = .const k) :
    ∃ xa, e a = .const xa ∧ constIf (-xa) = .const k := by
  rw [stepPure, Env.set_eq _ _ _ _ rfl] at h
  cases hea : e a <;> rw [hea] at h <;>
    first | exact ⟨_, rfl, h⟩ | (exfalso; revert h; simp)

theorem const_sound_ineg {m : Mem} {vs : Vals} {e : Env} {d a dd : Val} {x : V}
    (hag : Agree vs e) (hev : evalInst m vs (.ineg d a) = some (dd, x)) :
    Agree (setV vs dd x) (stepPure e (.ineg d a)) := by
  obtain ⟨t, w, hga, _, hdd, hx⟩ := evalInst_ineg_inv hev
  subst hdd
  intro v k hv
  by_cases hvd : v.id = dd.id
  · have hveq : v = dd := by cases v; cases dd; simp_all
    rw [hveq] at hv
    obtain ⟨xa, hea, hfold⟩ := stepPure_ineg_const hv
    obtain ⟨hsum, hin⟩ := constIf_const hfold
    obtain ⟨ta, w', hva, hta, hsa, _⟩ := hag a xa hea
    rw [hga] at hva
    have hta' : TrackedTy t := by
      injection hva with h1; injection h1 with h2 _; rw [h2]; exact hta
    have hsa' : signed t w = xa := by
      injection hva with h1; injection h1 with h2 h3; rw [h2, h3]; exact hsa
    refine ⟨t, (0 - w) &&& widthMask t, ?_, hta', ?_, hsum ▸ hin⟩
    · rw [hveq, hx]; exact AlgorithmLib.HProg.getV_setV_self vs dd _
    · rw [signed_mask_of hta', signed_neg hta' hsa' hin, hsum]
  · rw [stepPure_frame _ e v (fun d' hd' => by
        simp only [Inst.destOf?, Option.some.injEq] at hd'; exact hd' ▸ hvd)] at hv
    obtain ⟨t1, w1, hw1, htt1, hs1, hf1⟩ := hag v k hv
    exact ⟨t1, w1, getV_setV_ne hvd hw1, htt1, hs1, hf1⟩

/-- The retagging arms pass a constant through unchanged, except that
    `uextend64` refuses a negative one. -/
theorem stepPure_sextend64_const {e : Env} {d a : Val} {k : Int}
    (h : stepPure e (.sextend64 d a) d = .const k) : e a = .const k := by
  rw [stepPure, Env.set_eq _ _ _ _ rfl] at h
  cases hea : e a <;> rw [hea] at h <;> first | exact h | exact absurd h (by simp)

theorem stepPure_ireduce32_const {e : Env} {d a : Val} {k : Int}
    (h : stepPure e (.ireduce32 d a) d = .const k) : e a = .const k := by
  rw [stepPure, Env.set_eq _ _ _ _ rfl] at h
  cases hea : e a <;> rw [hea] at h <;> first | exact h | exact absurd h (by simp)

theorem stepPure_uextend64_const {e : Env} {d a : Val} {k : Int}
    (h : stepPure e (.uextend64 d a) d = .const k) : e a = .const k ∧ 0 ≤ k := by
  rw [stepPure, Env.set_eq _ _ _ _ rfl] at h
  cases hea : e a <;> rw [hea] at h <;> simp_all
  split at h
  · rename_i h0
    injection h with hk
    subst hk
    exact ⟨rfl, h0⟩
  · exact absurd h (by simp)

theorem const_sound_sextend64 {m : Mem} {vs : Vals} {e : Env} {d a dd : Val} {x : V}
    (hag : Agree vs e) (hev : evalInst m vs (.sextend64 d a) = some (dd, x)) :
    Agree (setV vs dd x) (stepPure e (.sextend64 d a)) := by
  obtain ⟨t, w, hga, _, _, hdd, hx⟩ := evalInst_sextend64_inv hev
  subst hdd
  intro v k hv
  by_cases hvd : v.id = dd.id
  · have hveq : v = dd := by cases v; cases dd; simp_all
    rw [hveq] at hv
    have hea := stepPure_sextend64_const hv
    obtain ⟨ta, w', hva, hta, hsa, hf⟩ := hag a k hea
    rw [hga] at hva
    have hsa' : signed t w = k := by
      injection hva with h1; injection h1 with h2 h3; rw [h2, h3]; exact hsa
    obtain ⟨w2, hof, hsg⟩ := signed_ofInt (t := ClifTy.i64) (Or.inr rfl) hf
    refine ⟨ClifTy.i64, w2, ?_, Or.inr rfl, hsg, hf⟩
    rw [hveq, hx, hsa', hof]; exact AlgorithmLib.HProg.getV_setV_self vs dd _
  · rw [stepPure_frame _ e v (fun d' hd' => by
        simp only [Inst.destOf?, Option.some.injEq] at hd'; exact hd' ▸ hvd)] at hv
    obtain ⟨t1, w1, hw1, htt1, hs1, hf1⟩ := hag v k hv
    exact ⟨t1, w1, getV_setV_ne hvd hw1, htt1, hs1, hf1⟩

theorem const_sound_ireduce32 {m : Mem} {vs : Vals} {e : Env} {d a dd : Val} {x : V}
    (hag : Agree vs e) (hev : evalInst m vs (.ireduce32 d a) = some (dd, x)) :
    Agree (setV vs dd x) (stepPure e (.ireduce32 d a)) := by
  obtain ⟨t, w, hga, _, _, hdd, hx⟩ := evalInst_ireduce32_inv hev
  subst hdd
  intro v k hv
  by_cases hvd : v.id = dd.id
  · have hveq : v = dd := by cases v; cases dd; simp_all
    rw [hveq] at hv
    have hea := stepPure_ireduce32_const hv
    obtain ⟨ta, w', hva, hta, hsa, hf⟩ := hag a k hea
    rw [hga] at hva
    have hta' : TrackedTy t := by
      injection hva with h1; injection h1 with h2 _; rw [h2]; exact hta
    have hsa' : signed t w = k := by
      injection hva with h1; injection h1 with h2 h3; rw [h2, h3]; exact hsa
    refine ⟨ClifTy.i32, w &&& widthMask ClifTy.i32, ?_, Or.inl rfl,
            signed_reduce32 hta' hsa' hf, hf⟩
    rw [hveq, hx]; exact AlgorithmLib.HProg.getV_setV_self vs dd _
  · rw [stepPure_frame _ e v (fun d' hd' => by
        simp only [Inst.destOf?, Option.some.injEq] at hd'; exact hd' ▸ hvd)] at hv
    obtain ⟨t1, w1, hw1, htt1, hs1, hf1⟩ := hag v k hv
    exact ⟨t1, w1, getV_setV_ne hvd hw1, htt1, hs1, hf1⟩

theorem const_sound_uextend64 {m : Mem} {vs : Vals} {e : Env} {d a dd : Val} {x : V}
    (hag : Agree vs e) (hev : evalInst m vs (.uextend64 d a) = some (dd, x)) :
    Agree (setV vs dd x) (stepPure e (.uextend64 d a)) := by
  obtain ⟨t, w, hga, _, _, hdd, hx⟩ := evalInst_uextend64_inv hev
  subst hdd
  intro v k hv
  by_cases hvd : v.id = dd.id
  · have hveq : v = dd := by cases v; cases dd; simp_all
    rw [hveq] at hv
    obtain ⟨hea, hk0⟩ := stepPure_uextend64_const hv
    obtain ⟨ta, w', hva, hta, hsa, hf⟩ := hag a k hea
    rw [hga] at hva
    have hta' : TrackedTy t := by
      injection hva with h1; injection h1 with h2 _; rw [h2]; exact hta
    have hsa' : signed t w = k := by
      injection hva with h1; injection h1 with h2 h3; rw [h2, h3]; exact hsa
    refine ⟨ClifTy.i64, w &&& widthMask t, ?_, Or.inr rfl,
            signed_uextend64 hta' hsa' hk0 hf, hf⟩
    rw [hveq, hx]; exact AlgorithmLib.HProg.getV_setV_self vs dd _
  · rw [stepPure_frame _ e v (fun d' hd' => by
        simp only [Inst.destOf?, Option.some.injEq] at hd'; exact hd' ▸ hvd)] at hv
    obtain ⟨t1, w1, hw1, htt1, hs1, hf1⟩ := hag v k hv
    exact ⟨t1, w1, getV_setV_ne hvd hw1, htt1, hs1, hf1⟩

/-- **The shift amount the machine uses is the one the model folded with.**

    The machine reduces it modulo the *shifted* operand's width, which the model
    cannot see; the two agree because that width always divides the amount's own
    modulus, and `shiftOk` keeps the amount below both. -/
theorem shift_amount {tb : ClifTy} (htb : TrackedTy tb) {wb : UInt64} {y : Int}
    (hs : signed tb wb = y) (h0 : 0 ≤ y) (hlt : y < 32)
    {t : ClifTy} (ht : TrackedTy t) :
    wb % UInt64.ofNat t.width = UInt64.ofNat y.toNat := by
  have hc := congr_of_signed htb hs
  have hy : ((y.toNat : Nat) : Int) = y := Int.toNat_of_nonneg h0
  have hb : wb.toNat < 2 ^ 64 := wb.toNat_lt_size
  refine UInt64.toNat_inj.mp ?_
  rw [UInt64.toNat_mod]
  rcases ht with rfl | rfl <;> rcases htb with rfl | rfl <;>
    simp only [modOf, AlgorithmLib.IR.ClifTy.width] at hc ⊢ <;>
    simp only [show (UInt64.ofNat 32).toNat = 32 from rfl,
               show (UInt64.ofNat 64).toNat = 64 from rfl,
               show (UInt64.ofNat y.toNat).toNat = y.toNat % 2 ^ 64 from by simp] <;>
    omega

theorem shlLit_ne_const {d : DExp} {y k : Int} : shlLit d y ≠ .const k := by
  unfold shlLit; split <;> simp

theorem shrLit_ne_const {d : DExp} {y k : Int} : shrLit d y ≠ .const k := by
  unfold shrLit; split <;> simp

theorem stepPure_ishl_const {e : Env} {d a b : Val} {k : Int}
    (h : stepPure e (.ishl d a b) d = .const k) :
    ∃ xa xb, e a = .const xa ∧ e b = .const xb ∧ shiftOk xb = true
      ∧ constIf (xa * 2 ^ xb.toNat) = .const k := by
  rw [stepPure, Env.set_eq _ _ _ _ rfl] at h
  cases hea : e a <;> cases heb : e b <;> rw [hea, heb] at h <;> simp only [] at h <;>
    first
      | (exact absurd h shlLit_ne_const)
      | (exact absurd h (by simp))
      | (split at h
         · exact ⟨_, _, rfl, rfl, by assumption, h⟩
         · exact absurd h (by simp))

theorem stepPure_ushr_const {e : Env} {d a b : Val} {k : Int}
    (h : stepPure e (.ushr d a b) d = .const k) :
    ∃ xa xb, e a = .const xa ∧ e b = .const xb ∧ shiftOk xb = true ∧ 0 ≤ xa
      ∧ constIf (xa / 2 ^ xb.toNat) = .const k := by
  rw [stepPure, Env.set_eq _ _ _ _ rfl] at h
  cases hea : e a <;> cases heb : e b <;> rw [hea, heb] at h <;> simp only [] at h <;>
    first
      | (exact absurd h shrLit_ne_const)
      | (split at h
         · rename_i hcond
           simp only [Bool.and_eq_true, decide_eq_true_eq] at hcond
           exact ⟨_, _, rfl, rfl, hcond.1, hcond.2, h⟩
         · exact absurd h (by simp))
      | (exact absurd h (by simp))

theorem const_sound_ishl {m : Mem} {vs : Vals} {e : Env} {d a b dd : Val} {x : V}
    (hag : Agree vs e) (hev : evalInst m vs (.ishl d a b) = some (dd, x)) :
    Agree (setV vs dd x) (stepPure e (.ishl d a b)) := by
  obtain ⟨ta, tb, wa, wb, hga, hgb, _, _, hdd, hx⟩ := evalInst_ishl_inv hev
  subst hdd
  intro v k hv
  by_cases hvd : v.id = dd.id
  · have hveq : v = dd := by cases v; cases dd; simp_all
    rw [hveq] at hv
    obtain ⟨xa, xb, hea, heb, hok, hfold⟩ := stepPure_ishl_const hv
    obtain ⟨hsum, hin⟩ := constIf_const hfold
    obtain ⟨ta', wa', hva, hta, hsa, _⟩ := hag a xa hea
    obtain ⟨tb', wb', hvb, htb, hsb, _⟩ := hag b xb heb
    rw [hga] at hva; rw [hgb] at hvb
    have hta' : TrackedTy ta := by
      injection hva with h1; injection h1 with h2 _; rw [h2]; exact hta
    have hsa' : signed ta wa = xa := by
      injection hva with h1; injection h1 with h2 h3; rw [h2, h3]; exact hsa
    have htb' : TrackedTy tb := by
      injection hvb with h1; injection h1 with h2 _; rw [h2]; exact htb
    have hsb' : signed tb wb = xb := by
      injection hvb with h1; injection h1 with h2 h3; rw [h2, h3]; exact hsb
    obtain ⟨hlo, hhi⟩ : 0 ≤ xb ∧ xb < 32 := by
      simp only [shiftOk, Bool.and_eq_true, decide_eq_true_eq] at hok; omega
    have hamt := shift_amount htb' hsb' hlo hhi hta'
    have hs64 : xb.toNat < 64 := by omega
    refine ⟨ta, (wa <<< UInt64.ofNat xb.toNat) &&& widthMask ta, ?_, hta', ?_, hsum ▸ hin⟩
    · rw [hveq, hx, hamt]; exact AlgorithmLib.HProg.getV_setV_self vs dd _
    · rw [signed_mask_of hta', signed_shl hta' hs64 hsa' hin, hsum]
  · rw [stepPure_frame _ e v (fun d' hd' => by
        simp only [Inst.destOf?, Option.some.injEq] at hd'; exact hd' ▸ hvd)] at hv
    obtain ⟨t1, w1, hw1, htt1, hs1, hf1⟩ := hag v k hv
    exact ⟨t1, w1, getV_setV_ne hvd hw1, htt1, hs1, hf1⟩

theorem const_sound_ushr {m : Mem} {vs : Vals} {e : Env} {d a b dd : Val} {x : V}
    (hag : Agree vs e) (hev : evalInst m vs (.ushr d a b) = some (dd, x)) :
    Agree (setV vs dd x) (stepPure e (.ushr d a b)) := by
  obtain ⟨ta, tb, wa, wb, hga, hgb, _, _, hdd, hx⟩ := evalInst_ushr_inv hev
  subst hdd
  intro v k hv
  by_cases hvd : v.id = dd.id
  · have hveq : v = dd := by cases v; cases dd; simp_all
    rw [hveq] at hv
    obtain ⟨xa, xb, hea, heb, hok, hnn, hfold⟩ := stepPure_ushr_const hv
    obtain ⟨hsum, hin⟩ := constIf_const hfold
    obtain ⟨ta', wa', hva, hta, hsa, hfa⟩ := hag a xa hea
    obtain ⟨tb', wb', hvb, htb, hsb, _⟩ := hag b xb heb
    rw [hga] at hva; rw [hgb] at hvb
    have hta' : TrackedTy ta := by
      injection hva with h1; injection h1 with h2 _; rw [h2]; exact hta
    have hsa' : signed ta wa = xa := by
      injection hva with h1; injection h1 with h2 h3; rw [h2, h3]; exact hsa
    have htb' : TrackedTy tb := by
      injection hvb with h1; injection h1 with h2 _; rw [h2]; exact htb
    have hsb' : signed tb wb = xb := by
      injection hvb with h1; injection h1 with h2 h3; rw [h2, h3]; exact hsb
    obtain ⟨hlo, hhi⟩ : 0 ≤ xb ∧ xb < 32 := by
      simp only [shiftOk, Bool.and_eq_true, decide_eq_true_eq] at hok; omega
    have hamt := shift_amount htb' hsb' hlo hhi hta'
    have hs64 : xb.toNat < 64 := by omega
    refine ⟨ta, ((wa &&& widthMask ta) >>> UInt64.ofNat xb.toNat) &&& widthMask ta,
            ?_, hta', ?_, hsum ▸ hin⟩
    · rw [hveq, hx, hamt]; exact AlgorithmLib.HProg.getV_setV_self vs dd _
    · rw [signed_mask_of hta', signed_ushr hta' hs64 hsa' hnn hfa, hsum]
  · rw [stepPure_frame _ e v (fun d' hd' => by
        simp only [Inst.destOf?, Option.some.injEq] at hd'; exact hd' ▸ hvd)] at hv
    obtain ⟨t1, w1, hw1, htt1, hs1, hf1⟩ := hag v k hv
    exact ⟨t1, w1, getV_setV_ne hvd hw1, htt1, hs1, hf1⟩

/-- `load` is tracked, but what it reports is a *slot* — provenance, not a
    number — so it owes the invariant nothing at its destination. -/
theorem const_sound_load {m : Mem} {vs : Vals} {e : Env} {d dd a : Val}
    {op : LoadOp} {x : V}
    (hag : Agree vs e) (hev : evalInst m vs (.load d op a) = some (dd, x)) :
    Agree (setV vs dd x) (stepPure e (.load d op a)) := by
  have hd : d = dd := by
    have := evalInst_dest hev; simp only [Inst.destOf?, Option.some.injEq] at this; exact this
  subst hd
  intro v k hv
  by_cases hvd : v.id = d.id
  · have hveq : v = d := by cases v; cases d; simp_all
    rw [hveq, stepPure, Env.set_eq _ _ _ _ rfl] at hv
    cases hea : e a <;> rw [hea] at hv <;> exact absurd hv (by simp)
  · rw [stepPure_frame _ e v (fun d' hd' => by
        simp only [Inst.destOf?, Option.some.injEq] at hd'; exact hd' ▸ hvd)] at hv
    obtain ⟨t1, w1, hw1, htt1, hs1, hf1⟩ := hag v k hv
    exact ⟨t1, w1, getV_setV_ne hvd hw1, htt1, hs1, hf1⟩

/-- **The launch model never reports a constant the machine does not compute.**

    One step of the abstract interpretation preserves the promise: if every
    constant the model already claims is the machine's word read signed, then it
    still is after the step. Proved rather than sampled, which is what retired
    the corpus check this file used to carry for the same claim.

    Nothing is claimed about `offset`, `slot` or `derived`: those describe
    provenance rather than a number, and each carries its own side condition —
    `inTy` for a displacement, `DExp.Exact` and `i32`-or-wider roots for an
    expression. -/
theorem const_sound {m : Mem} {vs : Vals} {e : Env} {i : Inst} {d : Val} {x : V}
    (hag : Agree vs e) (hev : evalInst m vs i = some (d, x)) :
    Agree (setV vs d x) (stepPure e i) := by
  cases i with
  | iconst _ _ _ => exact const_sound_iconst hag hev
  | iadd _ _ _ => exact const_sound_iadd hag hev
  | isub _ _ _ => exact const_sound_isub hag hev
  | imul _ _ _ => exact const_sound_imul hag hev
  | ineg _ _ => exact const_sound_ineg hag hev
  | ishl _ _ _ => exact const_sound_ishl hag hev
  | ushr _ _ _ => exact const_sound_ushr hag hev
  | ireduce32 _ _ => exact const_sound_ireduce32 hag hev
  | uextend64 _ _ => exact const_sound_uextend64 hag hev
  | sextend64 _ _ => exact const_sound_sextend64 hag hev
  | load _ _ _ => exact const_sound_load hag hev
  | _ => exact const_sound_untracked rfl hag hev

/-- The invariant holds of a run that has bound nothing. -/
theorem agree_empty (vs : Vals) : Agree vs Env.empty := by
  intro v k hv
  simp only [Env.empty, Env.get] at hv
  exact absurd hv (by simp)

/-! ## The expression claim

    `stepPure` reports `.derived d` for the four operations the host's
    loop-bound and meta-publishing idiom uses, and `DExp.eval` says what `d`
    means — in `Int`, which neither wraps nor truncates.  Three things stand
    between that and the machine's word, and all three are conditions on the
    *run* rather than on the expression, which is why they appear here as
    hypotheses rather than as guards inside `stepPure`:

    * **the roots are valued by the machine.**  A value the model cannot resolve
      becomes `.root v.id`, and nothing in the model says what that root is
      worth.  `rhoOf` is the valuation the machine itself supplies.
    * **every value is tracked.**  `stepPure` never sees a type, so it names an
      expression for an `i8` subtraction just as readily as for an `i64` one,
      where the claim is false — that is what `narrowShiftDisagrees` records.
    * **`DExp.Exact`.**  Already defined, and now actually used. -/

/-- The roots an expression names, all below `n`. -/
def DExp.rootsLt (n : Nat) : DExp → Bool
  | .root v  => decide (v < n)
  | .lit _   => true
  | .add a b => DExp.rootsLt n a && DExp.rootsLt n b
  | .sub a b => DExp.rootsLt n a && DExp.rootsLt n b
  | .shr a _ => DExp.rootsLt n a
  | .shl a _ => DExp.rootsLt n a

theorem rootsLt_mono {n m : Nat} (h : n ≤ m) : ∀ {d : DExp},
    DExp.rootsLt n d = true → DExp.rootsLt m d = true
  | .root _, hd => by simp only [DExp.rootsLt, decide_eq_true_eq] at hd ⊢; omega
  | .lit _, _ => rfl
  | .add a b, hd => by
      simp only [DExp.rootsLt, Bool.and_eq_true] at hd ⊢
      exact ⟨rootsLt_mono h hd.1, rootsLt_mono h hd.2⟩
  | .sub a b, hd => by
      simp only [DExp.rootsLt, Bool.and_eq_true] at hd ⊢
      exact ⟨rootsLt_mono h hd.1, rootsLt_mono h hd.2⟩
  | .shr a _, hd => rootsLt_mono h (d := a) hd
  | .shl a _, hd => rootsLt_mono h (d := a) hd

/-- Evaluation looks only at the roots the expression names. -/
theorem eval_congr {rho rho' : Nat → Int} {n : Nat}
    (h : ∀ k, k < n → rho k = rho' k) : ∀ {d : DExp},
    DExp.rootsLt n d = true → DExp.eval rho d = DExp.eval rho' d
  | .root v, hd => by
      simp only [DExp.rootsLt, decide_eq_true_eq] at hd; exact h v hd
  | .lit _, _ => rfl
  | .add a b, hd => by
      simp only [DExp.rootsLt, Bool.and_eq_true] at hd
      simp only [DExp.eval, eval_congr h hd.1, eval_congr h hd.2]
  | .sub a b, hd => by
      simp only [DExp.rootsLt, Bool.and_eq_true] at hd
      simp only [DExp.eval, eval_congr h hd.1, eval_congr h hd.2]
  | .shr a _, hd => by simp only [DExp.eval, eval_congr h (d := a) hd]
  | .shl a _, hd => by simp only [DExp.eval, eval_congr h (d := a) hd]

/-- …and so does exactness. -/
theorem exact_congr {rho rho' : Nat → Int} {n : Nat}
    (h : ∀ k, k < n → rho k = rho' k) : ∀ {d : DExp},
    DExp.rootsLt n d = true → DExp.Exact rho d = DExp.Exact rho' d
  | .root v, hd => by
      simp only [DExp.rootsLt, decide_eq_true_eq] at hd
      simp only [DExp.Exact, h v hd]
  | .lit _, _ => rfl
  | .add a b, hd => by
      simp only [DExp.rootsLt, Bool.and_eq_true] at hd
      simp only [DExp.Exact, exact_congr h hd.1, exact_congr h hd.2,
                 eval_congr h hd.1, eval_congr h hd.2]
  | .sub a b, hd => by
      simp only [DExp.rootsLt, Bool.and_eq_true] at hd
      simp only [DExp.Exact, exact_congr h hd.1, exact_congr h hd.2,
                 eval_congr h hd.1, eval_congr h hd.2]
  | .shr a _, hd => by
      simp only [DExp.Exact, exact_congr h (d := a) hd, eval_congr h (d := a) hd]
  | .shl a _, hd => by
      simp only [DExp.Exact, exact_congr h (d := a) hd, eval_congr h (d := a) hd]

/-- A destination past the end of the value map — what an SSA program's fresh
    numbering gives, and what makes a binding an extension rather than an
    overwrite. -/
def Fresh (vs : Vals) (d : Val) : Prop := vs.size ≤ d.id

theorem rhoOf_setV {vs : Vals} {d : Val} {x : V} (hf : Fresh vs d) :
    ∀ k, k < vs.size → rhoOf (setV vs d x) k = rhoOf vs k := by
  intro k hk
  simp only [Fresh] at hf
  have hne : (⟨k⟩ : Val).id ≠ d.id := by simp only []; omega
  simp only [rhoOf]
  rcases hg : getV vs ⟨k⟩ with _ | u
  · simp only [getV, Array.getElem?_eq_none_iff] at hg; omega
  · rw [getV_setV_ne hne hg]

theorem size_le_setV (vs : Vals) (d : Val) (x : V) : vs.size ≤ (setV vs d x).size := by
  simp only [setV, Array.set!, Array.size_setIfInBounds]
  split <;> simp <;> omega

/-- An exact expression's value is foldable — every constructor's last
    conjunct says so.  This is what lets a retag reuse the constant arms'
    arithmetic without a separate bound. -/
theorem inFold_of_exact {rho : Nat → Int} : ∀ {d : DExp},
    DExp.Exact rho d = true → inFold (DExp.eval rho d) = true
  | .root _, h => h
  | .lit _, h => h
  | .add _ _, h => by simp only [DExp.Exact, Bool.and_eq_true] at h; exact h.2
  | .sub _ _, h => by simp only [DExp.Exact, Bool.and_eq_true] at h; exact h.2
  | .shl _ _, h => by simp only [DExp.Exact, Bool.and_eq_true] at h; exact h.2
  | .shr _ _, h => by simp only [DExp.Exact, Bool.and_eq_true] at h; exact h.2

/-- The shapes a `const` and an `offset` denote: a literal, or a root plus a
    literal.

    These are exactly the expressions the additive arms both produce and
    consume — `addSym` and the `isub` displacement arm read `offset`, `const`
    and `unknown` operands and never a `derived` one — so no `shr` can reach
    them, which is what makes an unconditional congruence available here and
    nowhere else. -/
def DExp.isBase : DExp → Bool
  | .lit _                  => true
  | .add (.root _) (.lit _) => true
  | _                       => false

/-- **What the model promises about one value it can name.**

    Two clauses, because the model makes two strengths of claim and conflating
    them is what left the additive arms unreachable:

    * a **congruence**, unconditionally, for the base shapes.  Wrapping
      preserves it, so it holds even where the number is out of range and the
      equality below says nothing — and that is precisely the case `iadd` of an
      `offset` has to reach through, since the result's `DExp.Exact` does not
      bound the operand's own sum.
    * an **equality**, whenever `DExp.Exact` holds.  This is the claim a
      consumer reads a launch bound out of, and the only one available for an
      expression containing a `shr`, division being no congruence.

    The roots are bounded so that extending the value map cannot change what
    the expression denotes. -/
def DenotesAt (vs : Vals) (v : Val) (d : DExp) : Prop :=
  ∃ t w, getV vs v = some (.sc t w) ∧ TrackedTy t
    ∧ DExp.rootsLt vs.size d = true
    ∧ (DExp.isBase d = true → Congr t w (DExp.eval (rhoOf vs) d))
    ∧ (DExp.Exact (rhoOf vs) d = true → signed t w = DExp.eval (rhoOf vs) d)

/-- **What the model claims of every value it can name.**

    One invariant for all three claims at once.  `SymVal.toD?` is the model's
    own statement of when it can name a value without inventing a root, and it
    answers for `const`, `offset` and `derived` alike — so this is not three
    invariants stapled together but the single property those three cases were
    always instances of.  It is also, verbatim, the value half of
    `Qwen2NonVacuity.MetaFaithful`. -/
def Denotes (vs : Vals) (e : Env) : Prop :=
  ∀ v d, (e v).toD? = some d → DenotesAt vs v d

/-- The claim holds of a run that has bound nothing. -/
theorem denotes_empty (vs : Vals) : Denotes vs Env.empty := by
  intro v d hv
  simp only [Env.empty, Env.get] at hv
  exact absurd hv (by simp [SymVal.toD?])

/-- **The one thing the model cannot check for itself, stated where it is
    spent.**

    `stepPure` never sees a type.  When it turns an operand it could *not* name
    into a root — `addSym`'s `.unknown, .const y => .offset a y`, and the `dOf`
    in the subtract and shift arms — it is making a claim about a width it never
    looked at, and at `i8` or `i16` that claim is false.  This is the exclusion
    `narrowShiftDisagrees` measures, carried as a hypothesis rather than assumed
    away.

    Deliberately **not** a condition on the whole value map.  A value the model
    has named already carries its own tracked type inside `Denotes`, and a
    program is free to hold floats and bytes in values the model says nothing
    about — which shipped programs do: `Qwen2Common` builds `.f32` values and
    `CliAlgorithm` uses `.i8`.  A condition over every value would be false of
    both, and a theorem resting on it would say nothing about either. -/
def UnnamedTracked (vs : Vals) (e : Env) (a : Val) : Prop :=
  ∀ t w, getV vs a = some (.sc t w) → (e a).toD? = none → TrackedTy t

/-- An operand's type is tracked either because the model named it — `Denotes`
    carries that — or because `UnnamedTracked` says so.  Every arm draws its
    `TrackedTy` from here, so the hypothesis is spent only where it must be. -/
theorem operand_tracked {vs : Vals} {e : Env} {a : Val} {t : ClifTy} {w : UInt64}
    (hden : Denotes vs e) (hut : UnnamedTracked vs e a)
    (hga : getV vs a = some (.sc t w)) : TrackedTy t := by
  rcases hd : (e a).toD? with _ | dd
  · exact hut t w hga hd
  · obtain ⟨t', w', hw', htt', _⟩ := hden a dd hd
    rw [hga] at hw'
    injection hw' with h1
    injection h1 with h2 _
    subst h2
    exact htt'

/-- An exact expression is in range, so its equality follows from its
    congruence.  This is what keeps the two clauses from drifting apart. -/
theorem denotesAt_of_congr {vs : Vals} {v : Val} {d : DExp} {t : ClifTy} {w : UInt64}
    (hw : getV vs v = some (.sc t w)) (htt : TrackedTy t)
    (hr : DExp.rootsLt vs.size d = true)
    (hc : Congr t w (DExp.eval (rhoOf vs) d)) : DenotesAt vs v d :=
  ⟨t, w, hw, htt, hr, fun _ => hc,
   fun hex => signed_of_congr htt (inFold_of_exact hex) hc⟩

/-- Binding a fresh destination leaves what the model already claimed intact:
    the word is still there, the roots are still in range, and the valuation
    did not move on any root the expression names. -/
theorem denotes_lift {vs : Vals} {d v : Val} {x : V} {dd : DExp}
    (hf : Fresh vs d) (hne : v.id ≠ d.id) (h : DenotesAt vs v dd) :
    DenotesAt (setV vs d x) v dd := by
  obtain ⟨t, w, hw, htt, hr, hcg, hc⟩ := h
  refine ⟨t, w, getV_setV_ne hne hw, htt, rootsLt_mono (size_le_setV vs d x) hr, ?_, ?_⟩
  · intro hb
    rw [eval_congr (rhoOf_setV hf) hr]
    exact hcg hb
  · intro hex
    rw [eval_congr (rhoOf_setV hf) hr]
    exact hc (by rw [← exact_congr (rhoOf_setV hf) hr]; exact hex)

/-- **What an operand denotes.**

    The one step that turns the invariant into arithmetic.  Either the model can
    name the operand, and the invariant says what its word is worth, or it
    cannot, and `dOf` makes it a root — whose valuation is *defined* to be that
    word, so the congruence is free and the shape is a base.  That second case
    is why the claim needs no hypothesis about values the model does not
    track. -/
theorem operand_denotes {vs : Vals} {e : Env} {a : Val} {t : ClifTy} {w : UInt64}
    (htt : TrackedTy t) (hden : Denotes vs e) (hga : getV vs a = some (.sc t w)) :
    DenotesAt vs a (dOf e a) := by
  rcases hd : (e a).toD? with _ | dd
  · have hroot : dOf e a = .root a.id := by
      cases hea : e a <;> rw [hea] at hd <;> simp_all [dOf, SymVal.toD?]
    have hsz : a.id < vs.size := by
      simp only [getV, Array.getElem?_eq_some_iff] at hga; exact hga.1
    have hval : DExp.eval (rhoOf vs) (dOf e a) = signed t w := by
      rw [hroot]
      simp only [DExp.eval, rhoOf]
      rw [show (⟨a.id⟩ : Val) = a from by cases a; rfl, hga]
    refine ⟨t, w, hga, htt,
      by rw [hroot]; simp only [DExp.rootsLt, decide_eq_true_eq]; exact hsz,
      fun _ => by rw [hval]; exact congr_of_signed htt rfl,
      fun _ => hval.symm⟩
  · obtain ⟨t', w', hw', htt', hr', hcg', hc'⟩ := hden a dd hd
    rw [dOf_of_toD? e a dd hd]
    rw [hga] at hw'
    injection hw' with h1
    injection h1 with ht hww
    subst ht; subst hww
    exact ⟨t, w, hga, htt', hr', hcg', hc'⟩

/-- A value the step leaves alone still denotes what it did. -/
theorem denotes_frame {m : Mem} {vs : Vals} {e : Env} {i : Inst} {d v : Val} {x : V}
    {dd : DExp} (hden : Denotes vs e) (hf : Fresh vs d)
    (hev : evalInst m vs i = some (d, x)) (hvd : v.id ≠ d.id)
    (hv : (stepPure e i v).toD? = some dd) :
    DenotesAt (setV vs d x) v dd := by
  rw [stepPure_frame _ e v (fun d' hd' => by
        rw [evalInst_dest hev] at hd'; injection hd' with hq; exact hq ▸ hvd)] at hv
  exact denotes_lift hf hvd (hden v dd hv)

/-- The invariant at a freshly bound destination, from a congruence alone.
    Both clauses follow: the equality spends `inFold`, which `DExp.Exact`
    supplies whenever it is the clause being asked for. -/
theorem denotesAt_dest {vs : Vals} {d : Val} {x : V} {t : ClifTy} {w : UInt64}
    {dd : DExp} (hf : Fresh vs d) (hx : x = .sc t w) (htt : TrackedTy t)
    (hr : DExp.rootsLt vs.size dd = true)
    (hc : Congr t w (DExp.eval (rhoOf vs) dd)) : DenotesAt (setV vs d x) d dd := by
  have hgd : getV (setV vs d x) d = some (.sc t w) := by
    rw [← hx]; exact AlgorithmLib.HProg.getV_setV_self vs d x
  refine ⟨t, w, hgd, htt, rootsLt_mono (size_le_setV vs d x) hr, ?_, ?_⟩
  · intro _; rw [eval_congr (rhoOf_setV hf) hr]; exact hc
  · intro hex
    rw [eval_congr (rhoOf_setV hf) hr]
    refine signed_of_congr htt ?_ hc
    have := inFold_of_exact hex
    rwa [eval_congr (rhoOf_setV hf) hr] at this

/-- What `Agree` says at a value the machine has computed. -/
theorem agree_at {vs : Vals} {e : Env} {a : Val} {t : ClifTy} {w : UInt64} {k : Int}
    (hag : Agree vs e) (hga : getV vs a = some (.sc t w)) (hea : e a = .const k) :
    signed t w = k ∧ inFold k = true := by
  obtain ⟨t', w', hw', _, hs, hfk⟩ := hag a k hea
  rw [hga] at hw'
  injection hw' with h1
  injection h1 with h2 h3
  subst h2; subst h3
  exact ⟨hs, hfk⟩

/-- An operand the model named as an offset: its congruence, at the base shape
    the invariant carries unconditionally. -/
theorem operand_offset {vs : Vals} {e : Env} {a p : Val} {t : ClifTy} {w : UInt64}
    {k : Int} (htt : TrackedTy t) (hden : Denotes vs e)
    (hga : getV vs a = some (.sc t w)) (hea : e a = .offset p k) :
    Congr t w (rhoOf vs p.id + k) ∧ p.id < vs.size := by
  have hd : (e a).toD? = some (.add (.root p.id) (.lit k)) := by rw [hea]; rfl
  obtain ⟨t', w', hw', _, hr', hcg', _⟩ := hden a _ hd
  rw [hga] at hw'
  injection hw' with h1
  injection h1 with h2 h3
  subst h2; subst h3
  refine ⟨hcg' rfl, ?_⟩
  simp only [DExp.rootsLt, Bool.and_eq_true, decide_eq_true_eq] at hr'
  exact hr'.1

/-- An operand the model could not name at all is a root, and a root's
    valuation is the machine's own word for it. -/
theorem operand_root {vs : Vals} {a : Val} {t : ClifTy} {w : UInt64}
    (hga : getV vs a = some (.sc t w)) :
    rhoOf vs a.id = signed t w ∧ a.id < vs.size := by
  have hsz : a.id < vs.size := by
    simp only [getV, Array.getElem?_eq_some_iff] at hga; exact hga.1
  refine ⟨?_, hsz⟩
  simp only [rhoOf]
  rw [show (⟨a.id⟩ : Val) = a from by cases a; rfl, hga]

/-- A value the model reports as a constant denotes the literal, and
    `const_sound` has already proved the machine agrees. -/
theorem denotes_of_const {vs : Vals} {e : Env} {v : Val} {k : Int}
    (hag : Agree vs e) (hv : e v = .const k) : DenotesAt vs v (.lit k) := by
  obtain ⟨t, w, hw, htt, hs, _⟩ := hag v k hv
  exact ⟨t, w, hw, htt, rfl, fun _ => congr_of_signed htt hs, fun _ => hs⟩

/-- **The additive arm, which the congruence clause is for.**

    `iadd` names an `offset` in four ways and folds a constant in the fifth.
    Every one is the same argument — both operands' congruences, added, masked
    — and none needs either operand's number to be in range, which is exactly
    what the equality clause could not have supplied. -/
theorem denotes_iadd {m : Mem} {vs : Vals} {e : Env} {d a b d0 : Val} {x : V}
    (hag : Agree vs e) (hden : Denotes vs e) (hua : UnnamedTracked vs e a)
    (hf : Fresh vs d0) (hev : evalInst m vs (.iadd d a b) = some (d0, x)) :
    Denotes (setV vs d0 x) (stepPure e (.iadd d a b)) := by
  obtain ⟨t, wa, wb, hga, hgb, _, hdd, hx⟩ := evalInst_iadd_inv hev
  rw [hdd] at hf ⊢
  have htt : TrackedTy t := operand_tracked hden hua hga
  intro v dd hv
  by_cases hvd : v.id = d.id
  · have hveq : v = d := by cases v; cases d; simp_all
    subst hveq
    -- the one shape every naming case reduces to
    have mk : ∀ (P : Val) (K x0 y0 : Int), P.id < vs.size →
        Congr t wa x0 → Congr t wb y0 → x0 + y0 = rhoOf vs P.id + K →
        DenotesAt (setV vs v x) v (.add (.root P.id) (.lit K)) := by
      intro P K x0 y0 hsz ca cb hsum
      refine denotesAt_dest hf hx htt (by simp [DExp.rootsLt, hsz]) ?_
      have := congr_mask htt (congr_add' htt ca cb)
      rwa [hsum] at this
    have mkLit : ∀ (K x0 y0 : Int), Congr t wa x0 → Congr t wb y0 → x0 + y0 = K →
        DenotesAt (setV vs v x) v (.lit K) := by
      intro K x0 y0 ca cb hsum
      refine denotesAt_dest hf hx htt rfl ?_
      have := congr_mask htt (congr_add' htt ca cb)
      rwa [hsum] at this
    rw [stepPure, Env.set_eq _ _ _ _ rfl] at hv
    unfold addSym at hv
    split at hv
    · -- const + const
      rename_i x0 y0 hea heb
      obtain ⟨hsa, _⟩ := agree_at hag hga hea
      obtain ⟨hsb, _⟩ := agree_at hag hgb heb
      unfold constIf at hv; split at hv
      · injection hv with hq; subst hq
        exact mkLit _ x0 y0 (congr_of_signed htt hsa) (congr_of_signed htt hsb) rfl
      · exact absurd hv (by simp [SymVal.toD?])
    · -- offset + const
      rename_i p k y0 hea heb
      obtain ⟨ca, hsz⟩ := operand_offset htt hden hga hea
      obtain ⟨hsb, _⟩ := agree_at hag hgb heb
      unfold offsetIf at hv; split at hv
      · injection hv with hq; subst hq
        exact mk p _ _ y0 hsz ca (congr_of_signed htt hsb) (by omega)
      · exact absurd hv (by simp [SymVal.toD?])
    · -- const + offset
      rename_i x0 p k hea heb
      obtain ⟨hsa, _⟩ := agree_at hag hga hea
      obtain ⟨cb, hsz⟩ := operand_offset htt hden hgb heb
      unfold offsetIf at hv; split at hv
      · injection hv with hq; subst hq
        exact mk p _ x0 _ hsz (congr_of_signed htt hsa) cb (by omega)
      · exact absurd hv (by simp [SymVal.toD?])
    · -- unresolved + const: the destination is an offset of the operand itself
      rename_i y0 hea heb
      obtain ⟨hra, hsz⟩ := operand_root hga
      obtain ⟨hsb, _⟩ := agree_at hag hgb heb
      injection hv with hq; subst hq
      exact mk a y0 _ y0 hsz (congr_of_signed htt rfl) (congr_of_signed htt hsb)
        (by rw [hra])
    · -- const + unresolved
      rename_i x0 hea heb
      obtain ⟨hsa, _⟩ := agree_at hag hga hea
      obtain ⟨hrb, hsz⟩ := operand_root hgb
      injection hv with hq; subst hq
      exact mk b x0 x0 _ hsz (congr_of_signed htt hsa) (congr_of_signed htt rfl)
        (by rw [hrb]; omega)
    · exact absurd hv (by simp [SymVal.toD?])
  · rw [← hdd]
    exact denotes_frame hden (by rw [hdd]; exact hf) hev (by rw [hdd]; exact hvd) hv

/-- **`isub`, which names both shapes.**

    Its displacement arm is a base and goes by congruence like `iadd`; its
    fallback names an expression, which is not a base, so only the conditional
    equality is owed there — and `DExp.Exact` of a `sub` hands over exactly the
    two operand exactnesses and the bound that `signed_sub` needs. -/
theorem denotes_isub {m : Mem} {vs : Vals} {e : Env} {d a b d0 : Val} {x : V}
    (hag : Agree vs e) (hden : Denotes vs e) (hua : UnnamedTracked vs e a)
    (hf : Fresh vs d0) (hev : evalInst m vs (.isub d a b) = some (d0, x)) :
    Denotes (setV vs d0 x) (stepPure e (.isub d a b)) := by
  obtain ⟨t, wa, wb, hga, hgb, _, hdd, hx⟩ := evalInst_isub_inv hev
  rw [hdd] at hf ⊢
  have htt : TrackedTy t := operand_tracked hden hua hga
  have oa := operand_denotes htt hden hga
  have ob := operand_denotes htt hden hgb
  intro v dd hv
  by_cases hvd : v.id = d.id
  · have hveq : v = d := by cases v; cases d; simp_all
    subst hveq
    have mk : ∀ (P : Val) (K x0 y0 : Int), P.id < vs.size →
        Congr t wa x0 → Congr t wb y0 → x0 - y0 = rhoOf vs P.id + K →
        DenotesAt (setV vs v x) v (.add (.root P.id) (.lit K)) := by
      intro P K x0 y0 hsz ca cb hsum
      refine denotesAt_dest hf hx htt (by simp [DExp.rootsLt, hsz]) ?_
      have := congr_mask htt (congr_sub' htt ca cb)
      rwa [hsum] at this
    rw [stepPure, Env.set_eq _ _ _ _ rfl] at hv
    split at hv
    · -- const − const
      rename_i x0 y0 hea heb
      obtain ⟨hsa, _⟩ := agree_at hag hga hea
      obtain ⟨hsb, _⟩ := agree_at hag hgb heb
      unfold constIf at hv; split at hv
      · injection hv with hq; subst hq
        refine denotesAt_dest hf hx htt rfl ?_
        have := congr_mask htt
          (congr_sub' htt (congr_of_signed htt hsa) (congr_of_signed htt hsb))
        exact this
      · exact absurd hv (by simp [SymVal.toD?])
    · -- offset − const
      rename_i p k y0 hea heb
      obtain ⟨ca, hsz⟩ := operand_offset htt hden hga hea
      obtain ⟨hsb, _⟩ := agree_at hag hgb heb
      unfold offsetIf at hv; split at hv
      · injection hv with hq; subst hq
        exact mk p _ _ y0 hsz ca (congr_of_signed htt hsb) (by omega)
      · exact absurd hv (by simp [SymVal.toD?])
    · -- anything else names the expression
      injection hv with hq
      subst hq
      obtain ⟨ta, wa', hwa, _, hra, _, hca⟩ := oa
      obtain ⟨tb, wb', hwb, _, hrb, _, hcb⟩ := ob
      rw [hga] at hwa; injection hwa with q1; injection q1 with q2 q3
      subst q2; subst q3
      rw [hgb] at hwb; injection hwb with q1; injection q1 with q2 q3
      subst q2; subst q3
      refine ⟨t, (wa - wb) &&& widthMask t,
        by rw [← hx]; exact AlgorithmLib.HProg.getV_setV_self vs v x, htt,
        by simp only [DExp.rootsLt, Bool.and_eq_true]
           exact ⟨rootsLt_mono (size_le_setV vs v x) hra,
                  rootsLt_mono (size_le_setV vs v x) hrb⟩,
        fun hb => absurd hb (by simp [DExp.isBase]), ?_⟩
      intro hex
      have hrsub : DExp.rootsLt vs.size (.sub (dOf e a) (dOf e b)) = true := by
        simp only [DExp.rootsLt, Bool.and_eq_true]; exact ⟨hra, hrb⟩
      rw [eval_congr (rhoOf_setV hf) hrsub]
      have hex' : DExp.Exact (rhoOf vs) (.sub (dOf e a) (dOf e b)) = true := by
        rw [← exact_congr (rhoOf_setV hf) hrsub]; exact hex
      simp only [DExp.Exact, Bool.and_eq_true] at hex'
      obtain ⟨⟨ea, eb⟩, ebound⟩ := hex'
      rw [signed_mask_of htt, DExp.eval]
      exact signed_sub htt (hca ea) (hcb eb) ebound
  · rw [← hdd]
    exact denotes_frame hden (by rw [hdd]; exact hf) hev (by rw [hdd]; exact hvd) hv

/-- The expression arm, shared by `ishl` and `ushr`: the model named
    `f (dOf e a)`, which is no base, so only the conditional equality is owed
    and `DExp.Exact` supplies the operand's. -/
theorem denotesAt_expr {vs : Vals} {d : Val} {x : V} {t : ClifTy}
    {w : UInt64} {dx : DExp} (hf : Fresh vs d) (hx : x = .sc t w) (htt : TrackedTy t)
    (hshape : DExp.isBase dx = false)
    (hroots : DExp.rootsLt vs.size dx = true)
    (heq : DExp.Exact (rhoOf vs) dx = true → signed t w = DExp.eval (rhoOf vs) dx) :
    DenotesAt (setV vs d x) d dx := by
  refine ⟨t, w, by rw [← hx]; exact AlgorithmLib.HProg.getV_setV_self vs d x, htt,
    rootsLt_mono (size_le_setV vs d x) hroots,
    fun hb => absurd (hb.symm.trans hshape) (by simp), ?_⟩
  intro hex
  rw [eval_congr (rhoOf_setV hf) hroots]
  exact heq (by rw [← exact_congr (rhoOf_setV hf) hroots]; exact hex)

theorem denotes_ishl {m : Mem} {vs : Vals} {e : Env} {d a b d0 : Val} {x : V}
    (hag : Agree vs e) (hden : Denotes vs e) (hua : UnnamedTracked vs e a)
    (hub : UnnamedTracked vs e b)
    (hf : Fresh vs d0) (hev : evalInst m vs (.ishl d a b) = some (d0, x)) :
    Denotes (setV vs d0 x) (stepPure e (.ishl d a b)) := by
  obtain ⟨ta, tb, wa, wb, hga, hgb, _, _, hdd, hx⟩ := evalInst_ishl_inv hev
  rw [hdd] at hf hev ⊢
  have hta : TrackedTy ta := operand_tracked hden hua hga
  have htb : TrackedTy tb := operand_tracked hden hub hgb
  have oa := operand_denotes hta hden hga
  intro v dd hv
  by_cases hvd : v.id = d.id
  · have hveq : v = d := by cases v; cases d; simp_all
    subst hveq
    rw [stepPure, Env.set_eq _ _ _ _ rfl] at hv
    split at hv
    · -- both constant: a fold, so `const_sound` answers
      rename_i x0 y0 hea heb
      split at hv
      · unfold constIf at hv; split at hv
        · injection hv with hq; subst hq
          refine denotes_of_const (const_sound hag hev) ?_
          rw [stepPure, Env.set_eq _ _ _ _ rfl, hea, heb]
          simp_all [constIf]
        · exact absurd hv (by simp [SymVal.toD?])
      · exact absurd hv (by simp [SymVal.toD?])
    · -- runtime operand, constant amount: the model names a shift
      rename_i y0 heb hnc
      unfold shlLit at hv; split at hv
      · rename_i hok
        injection hv with hq; subst hq
        obtain ⟨tb', wb', hwb, _, hsb, _⟩ := hag b y0 heb
        rw [hgb] at hwb; injection hwb with q1; injection q1 with q2 q3
        subst q2; subst q3
        obtain ⟨hlo, hhi⟩ : 0 ≤ y0 ∧ y0 < 32 := by
          simp only [shiftOk, Bool.and_eq_true, decide_eq_true_eq] at hok; omega
        obtain ⟨t', w', hwa, _, hra, _, hca⟩ := oa
        rw [hga] at hwa; injection hwa with q1; injection q1 with q2 q3
        subst q2; subst q3
        rw [hx, shift_amount htb hsb hlo hhi hta]
        refine denotesAt_expr hf rfl hta rfl hra ?_
        intro hex
        simp only [DExp.Exact, Bool.and_eq_true] at hex
        rw [signed_mask_of hta, DExp.eval]
        exact signed_shl hta (by omega) (hca hex.1) hex.2
      · exact absurd hv (by simp [SymVal.toD?])
    · exact absurd hv (by simp [SymVal.toD?])
  · exact denotes_frame hden hf hev hvd hv

theorem denotes_ushr {m : Mem} {vs : Vals} {e : Env} {d a b d0 : Val} {x : V}
    (hag : Agree vs e) (hden : Denotes vs e) (hua : UnnamedTracked vs e a)
    (hub : UnnamedTracked vs e b)
    (hf : Fresh vs d0) (hev : evalInst m vs (.ushr d a b) = some (d0, x)) :
    Denotes (setV vs d0 x) (stepPure e (.ushr d a b)) := by
  obtain ⟨ta, tb, wa, wb, hga, hgb, _, _, hdd, hx⟩ := evalInst_ushr_inv hev
  rw [hdd] at hf hev ⊢
  have hta : TrackedTy ta := operand_tracked hden hua hga
  have htb : TrackedTy tb := operand_tracked hden hub hgb
  have oa := operand_denotes hta hden hga
  intro v dd hv
  by_cases hvd : v.id = d.id
  · have hveq : v = d := by cases v; cases d; simp_all
    subst hveq
    rw [stepPure, Env.set_eq _ _ _ _ rfl] at hv
    split at hv
    · rename_i x0 y0 hea heb
      split at hv
      · unfold constIf at hv; split at hv
        · injection hv with hq; subst hq
          refine denotes_of_const (const_sound hag hev) ?_
          rw [stepPure, Env.set_eq _ _ _ _ rfl, hea, heb]
          simp_all [constIf]
        · exact absurd hv (by simp [SymVal.toD?])
      · exact absurd hv (by simp [SymVal.toD?])
    · rename_i y0 heb hnc
      unfold shrLit at hv; split at hv
      · rename_i hok
        injection hv with hq; subst hq
        obtain ⟨tb', wb', hwb, _, hsb, _⟩ := hag b y0 heb
        rw [hgb] at hwb; injection hwb with q1; injection q1 with q2 q3
        subst q2; subst q3
        obtain ⟨hlo, hhi⟩ : 0 ≤ y0 ∧ y0 < 32 := by
          simp only [shiftOk, Bool.and_eq_true, decide_eq_true_eq] at hok; omega
        obtain ⟨t', w', hwa, _, hra, _, hca⟩ := oa
        rw [hga] at hwa; injection hwa with q1; injection q1 with q2 q3
        subst q2; subst q3
        rw [hx, shift_amount htb hsb hlo hhi hta]
        refine denotesAt_expr hf rfl hta rfl hra ?_
        intro hex
        simp only [DExp.Exact, Bool.and_eq_true, decide_eq_true_eq] at hex
        rw [signed_mask_of hta, DExp.eval]
        exact signed_ushr hta (by omega) (hca hex.1.1) hex.1.2
          (inFold_of_exact hex.1.1)
      · exact absurd hv (by simp [SymVal.toD?])
    · exact absurd hv (by simp [SymVal.toD?])
  · exact denotes_frame hden hf hev hvd hv

/-- A binding the invariant is answered for by `const_sound`, or one it is
    owed nothing about. -/
def ConstOrOpaque (sv : SymVal) : Prop :=
  (∃ k, sv = .const k) ∨ sv.toD? = none

/-- **The arms that name nothing but a constant.**

    Most instructions bind their destination a constant, a slot or nothing.
    `SymVal.toD?` is `none` on the last two, so the invariant is owed nothing
    there, and on the first `const_sound` has already proved the machine agrees
    — which is why these arms need no arithmetic of their own. -/
theorem denotes_const_dest {m : Mem} {vs : Vals} {e : Env} {i : Inst} {d : Val} {x : V}
    (hag : Agree vs e) (hden : Denotes vs e) (hf : Fresh vs d)
    (hev : evalInst m vs i = some (d, x))
    (hdst : ConstOrOpaque (stepPure e i d)) :
    Denotes (setV vs d x) (stepPure e i) := by
  intro v dd hv
  by_cases hvd : v.id = d.id
  · have hveq : v = d := by cases v; cases d; simp_all
    subst hveq
    rcases hdst with ⟨k, hc⟩ | hn
    · rw [hc] at hv
      simp only [SymVal.toD?, Option.some.injEq] at hv
      subst hv
      exact denotes_of_const (const_sound hag hev) hc
    · rw [hn] at hv; exact absurd hv (by simp)
  · exact denotes_frame hden hf hev hvd hv

/-- **`stepPure` never names a base shape as a `derived` value.**

    The expressions it builds have `sub`, `shl` or `shr` at the top, and the two
    passthrough arms carry whatever their operand had.  A `derived (lit k)`
    therefore cannot arise — which matters because sign-extending one would be
    the same widening hazard an `offset` had, and no guard would catch it. -/
def NoBaseDerived (e : Env) : Prop :=
  ∀ v dx, e v = .derived dx → DExp.isBase dx = false

theorem noBaseDerived_empty : NoBaseDerived Env.empty := by
  intro v dx hv
  simp only [Env.empty, Env.get] at hv
  exact absurd hv (by simp)

theorem noBaseDerived_dest {e : Env} {i : Inst} {d : Val} {dx : DExp}
    (h : NoBaseDerived e) (hd : Inst.destOf? i = some d)
    (hv : stepPure e i d = .derived dx) : DExp.isBase dx = false := by
  cases i <;> simp [Inst.destOf?] at hd <;> subst hd <;>
    rw [stepPure, Env.set_eq _ _ _ _ rfl] at hv
  case iconst =>
    simp only [constLit] at hv; split at hv <;> exact absurd hv (by simp)
  case iadd =>
    simp only [addSym] at hv; split at hv <;>
      (try simp only [constIf, offsetIf] at hv) <;>
      first
        | exact absurd hv (by simp)
        | (split at hv <;> exact absurd hv (by simp))
  case imul =>
    simp only [mulSym] at hv; split at hv <;>
      (try simp only [constIf] at hv) <;>
      first
        | exact absurd hv (by simp)
        | (split at hv <;> exact absurd hv (by simp))
  case ineg =>
    split at hv <;> (try simp only [constIf] at hv) <;>
      first
        | exact absurd hv (by simp)
        | (split at hv <;> exact absurd hv (by simp))
  case isub =>
    split at hv
    · simp only [constIf] at hv; split at hv <;> exact absurd hv (by simp)
    · simp only [offsetIf] at hv; split at hv <;> exact absurd hv (by simp)
    · injection hv with hq2; subst hq2; rfl
  case ishl =>
    split at hv
    · split at hv
      · simp only [constIf] at hv; split at hv <;> exact absurd hv (by simp)
      · exact absurd hv (by simp)
    · simp only [shlLit] at hv; split at hv
      · injection hv with hq2; subst hq2; rfl
      · exact absurd hv (by simp)
    · exact absurd hv (by simp)
  case ushr =>
    split at hv
    · split at hv
      · simp only [constIf] at hv; split at hv <;> exact absurd hv (by simp)
      · exact absurd hv (by simp)
    · simp only [shrLit] at hv; split at hv
      · injection hv with hq2; subst hq2; rfl
      · exact absurd hv (by simp)
    · exact absurd hv (by simp)
  case load => split at hv <;> exact absurd hv (by simp)
  case uextend64 =>
    split at hv
    · split at hv <;> exact absurd hv (by simp)
    · exact absurd hv (by simp)
  case ireduce32 =>
    -- the arm passes its operand's binding through unchanged
    rename_i a
    exact h a dx hv
  case sextend64 =>
    rename_i a
    split at hv
    · exact absurd hv (by simp)
    · rename_i xd hea
      injection hv with hq2; subst hq2; exact h a xd hea
    · exact absurd hv (by simp)
  all_goals exact absurd hv (by simp)


theorem noBaseDerived_step {e : Env} {i : Inst} (h : NoBaseDerived e) :
    NoBaseDerived (stepPure e i) := by
  intro v dx hv
  by_cases hvd : ∃ d, Inst.destOf? i = some d ∧ v.id = d.id
  · obtain ⟨d, hd, hq⟩ := hvd
    have hveq : v = d := by cases v; cases d; simp_all
    subst hveq
    exact noBaseDerived_dest h hd hv
  · have hfr : ∀ d, Inst.destOf? i = some d → v.id ≠ d.id :=
      fun d hd hq => hvd ⟨d, hd, hq⟩
    rw [stepPure_frame _ e v hfr] at hv
    exact h v dx hv

/-- `ofInt` is a scalar of the type it names, whatever the value. -/
theorem ofInt_sc (t : ClifTy) (k : Int) :
    ofInt t k = .sc t (UInt64.ofNat (k.emod (1 <<< 64)).toNat &&& widthMask t) := rfl

/-- **Truncation.**  `ireduce32` passes its operand's binding through, and the
    narrower type is where both clauses survive: a congruence only weakens with
    the modulus, and the equality holds because `DExp.Exact` puts the value
    inside `foldableRange`, which `i32` represents exactly. -/
theorem denotes_ireduce32 {m : Mem} {vs : Vals} {e : Env} {d a d0 : Val} {x : V}
    (hag : Agree vs e) (hden : Denotes vs e) (hua : UnnamedTracked vs e a)
    (hf : Fresh vs d0) (hev : evalInst m vs (.ireduce32 d a) = some (d0, x)) :
    Denotes (setV vs d0 x) (stepPure e (.ireduce32 d a)) := by
  obtain ⟨t, w, hga, _, _, hdd, hx⟩ := evalInst_ireduce32_inv hev
  rw [hdd] at hf hev ⊢
  have htt : TrackedTy t := operand_tracked hden hua hga
  intro v dd hv
  by_cases hvd : v.id = d.id
  · have hveq : v = d := by cases v; cases d; simp_all
    subst hveq
    rw [stepPure, Env.set_eq _ _ _ _ rfl] at hv
    have hdxa : dOf e a = dd := dOf_of_toD? e a dd hv
    obtain ⟨t', w', hwa, _, hra, hcga, hca⟩ := operand_denotes htt hden hga
    rw [hga] at hwa; injection hwa with q1; injection q1 with q2 q3
    subst q2; subst q3
    rw [hdxa] at hra hcga hca
    refine ⟨ClifTy.i32, w &&& widthMask ClifTy.i32,
      by rw [← hx]; exact AlgorithmLib.HProg.getV_setV_self vs v x, Or.inl rfl,
      rootsLt_mono (size_le_setV vs v x) hra, ?_, ?_⟩
    · intro hb
      rw [eval_congr (rhoOf_setV hf) hra]
      exact congr_reduce32 htt (hcga hb)
    · intro hex
      rw [eval_congr (rhoOf_setV hf) hra]
      have hex' : DExp.Exact (rhoOf vs) dd = true := by
        rw [← exact_congr (rhoOf_setV hf) hra]; exact hex
      exact signed_reduce32 htt (hca hex') (inFold_of_exact hex')
  · exact denotes_frame hden hf hev hvd hv

/-- **Sign extension**, which keeps only what its own guard bounds.

    A constant survives because `litOk` puts it inside `foldableRange`, where
    the narrow word and its extension are the same integer, and `const_sound`
    already decides it.  An expression survives for the same reason, by way of
    `DExp.Exact` — and `NoBaseDerived` is what says the expression cannot be a
    base, which is the case where the two would come apart. -/
theorem denotes_sextend64 {m : Mem} {vs : Vals} {e : Env} {d a d0 : Val} {x : V}
    (hag : Agree vs e) (hden : Denotes vs e) (hua : UnnamedTracked vs e a)
    (hnb : NoBaseDerived e)
    (hf : Fresh vs d0) (hev : evalInst m vs (.sextend64 d a) = some (d0, x)) :
    Denotes (setV vs d0 x) (stepPure e (.sextend64 d a)) := by
  obtain ⟨t, w, hga, _, _, hdd, hx⟩ := evalInst_sextend64_inv hev
  rw [hdd] at hf hev ⊢
  have htt : TrackedTy t := operand_tracked hden hua hga
  intro v dd hv
  by_cases hvd : v.id = d.id
  · have hveq : v = d := by cases v; cases d; simp_all
    subst hveq
    rw [stepPure, Env.set_eq _ _ _ _ rfl] at hv
    split at hv
    · -- a constant travels through
      rename_i k hea
      injection hv with hq; subst hq
      refine denotes_of_const (const_sound hag hev) ?_
      rw [stepPure, Env.set_eq _ _ _ _ rfl, hea]
    · -- so does an expression, which `NoBaseDerived` says is not a base
      rename_i dx hea
      injection hv with hq; subst hq
      have hnbx : DExp.isBase dx = false := hnb a dx hea
      have hdxa : dOf e a = dx := dOf_of_toD? e a dx (by rw [hea]; rfl)
      obtain ⟨t', w', hwa, _, hra, _, hca⟩ := operand_denotes htt hden hga
      rw [hga] at hwa; injection hwa with q1; injection q1 with q2 q3
      subst q2; subst q3
      rw [hdxa] at hra hca
      refine ⟨ClifTy.i64,
        UInt64.ofNat ((signed t w).emod (1 <<< 64)).toNat &&& widthMask ClifTy.i64,
        by rw [← ofInt_sc, ← hx]; exact AlgorithmLib.HProg.getV_setV_self vs v x,
        Or.inr rfl, rootsLt_mono (size_le_setV vs v x) hra,
        fun hb => absurd (hb.symm.trans hnbx) (by simp), ?_⟩
      intro hex
      rw [eval_congr (rhoOf_setV hf) hra]
      have hex' : DExp.Exact (rhoOf vs) dx = true := by
        rw [← exact_congr (rhoOf_setV hf) hra]; exact hex
      obtain ⟨w2, hof, hsg⟩ :=
        signed_ofInt (t := ClifTy.i64) (Or.inr rfl) (inFold_of_exact hex')
      rw [ofInt_sc] at hof
      injection hof with q1 q2
      rw [hca hex', q2]
      exact hsg
    · exact absurd hv (by simp [SymVal.toD?])
  · exact denotes_frame hden hf hev hvd hv

/-- **The operands whose width the model has to assume something about.**

    Exactly the arms that turn an operand into a root or a base: the additive
    pair, the two shifts, and the two retags.  Every other instruction reads
    nothing this condition covers, so it asks nothing of it. -/
def Inst.readsOf : Inst → List Val
  | .iadd _ a b | .isub _ a b | .imul _ a b | .ishl _ a b | .ushr _ a b => [a, b]
  | .ireduce32 _ a | .sextend64 _ a => [a]
  | _ => []

/-- **Every operand of the instruction is one the width hypothesis covers.**

    Stated over the whole instruction so the step below carries one hypothesis
    rather than one per arm, and over `readsOf` rather than over every value:
    a program is free to hold floats and bytes in values this says nothing
    about, and shipped ones do.  See `UnnamedTracked`. -/
def OperandsTracked (vs : Vals) (e : Env) (i : Inst) : Prop :=
  ∀ a ∈ Inst.readsOf i, UnnamedTracked vs e a

/-- **One step of the launch model is sound.**

    If every claim the model already makes about a named value is one the
    machine agrees with, then it still is after the step — over every `Inst`
    constructor, not a sampled grid.

    Three hypotheses, and each is there for a reason the model cannot remove:
    `Agree` and `NoBaseDerived` are invariants carried alongside, `Fresh` is
    what an SSA numbering gives, and `OperandsTracked` is the narrow-width
    exclusion `narrowShiftDisagrees` measures.

    `slot` is not covered here: `SymVal.toD?` is `none` on one, so this says
    nothing about it, and it has its own claim. -/
theorem denotes_step {m : Mem} {vs : Vals} {e : Env} {i : Inst} {d : Val} {x : V}
    (hag : Agree vs e) (hden : Denotes vs e) (hnb : NoBaseDerived e)
    (hot : OperandsTracked vs e i) (hf : Fresh vs d)
    (hev : evalInst m vs i = some (d, x)) :
    Denotes (setV vs d x) (stepPure e i) := by
  cases i with
  | iadd d' a b => exact denotes_iadd hag hden (hot a (by simp [Inst.readsOf])) hf hev
  | isub d' a b => exact denotes_isub hag hden (hot a (by simp [Inst.readsOf])) hf hev
  | ishl d' a b =>
      exact denotes_ishl hag hden (hot a (by simp [Inst.readsOf]))
        (hot b (by simp [Inst.readsOf])) hf hev
  | ushr d' a b =>
      exact denotes_ushr hag hden (hot a (by simp [Inst.readsOf]))
        (hot b (by simp [Inst.readsOf])) hf hev
  | ireduce32 d' a => exact denotes_ireduce32 hag hden (hot a (by simp [Inst.readsOf])) hf hev
  | sextend64 d' a => exact denotes_sextend64 hag hden (hot a (by simp [Inst.readsOf])) hnb hf hev
  | iconst d' t k =>
      refine denotes_const_dest hag hden hf hev ?_
      have hq : d' = d := by
        have hde := evalInst_dest hev
        simp only [Inst.destOf?, Option.some.injEq] at hde
        exact hde
      rw [hq, stepPure, Env.set_eq _ _ _ _ rfl]
      simp only [constLit]
      split
      · exact Or.inl ⟨_, rfl⟩
      · exact Or.inr rfl
  | imul d' a b =>
      refine denotes_const_dest hag hden hf hev ?_
      have hq : d' = d := by
        have hde := evalInst_dest hev
        simp only [Inst.destOf?, Option.some.injEq] at hde
        exact hde
      rw [hq, stepPure, Env.set_eq _ _ _ _ rfl]
      simp only [mulSym]
      split <;> (try simp only [constIf]) <;>
        first
          | exact Or.inr rfl
          | (split
             · exact Or.inl ⟨_, rfl⟩
             · exact Or.inr rfl)
  | ineg d' a =>
      refine denotes_const_dest hag hden hf hev ?_
      have hq : d' = d := by
        have hde := evalInst_dest hev
        simp only [Inst.destOf?, Option.some.injEq] at hde
        exact hde
      rw [hq, stepPure, Env.set_eq _ _ _ _ rfl]
      split <;> (try simp only [constIf]) <;>
        first
          | exact Or.inr rfl
          | (split
             · exact Or.inl ⟨_, rfl⟩
             · exact Or.inr rfl)
  | uextend64 d' a =>
      refine denotes_const_dest hag hden hf hev ?_
      have hq : d' = d := by
        have hde := evalInst_dest hev
        simp only [Inst.destOf?, Option.some.injEq] at hde
        exact hde
      rw [hq, stepPure, Env.set_eq _ _ _ _ rfl]
      split
      · split
        · exact Or.inl ⟨_, rfl⟩
        · exact Or.inr rfl
      · exact Or.inr rfl
  | load d' op a =>
      refine denotes_const_dest hag hden hf hev ?_
      have hq : d' = d := by
        have hde := evalInst_dest hev
        simp only [Inst.destOf?, Option.some.injEq] at hde
        exact hde
      rw [hq, stepPure, Env.set_eq _ _ _ _ rfl]
      split <;> exact Or.inr rfl
  | _ =>
      refine denotes_const_dest hag hden hf hev ?_
      exact Or.inr (by rw [stepPure_untracked _ _ _ (evalInst_dest hev) rfl]; rfl)

/-! ## The run

    One step is not what the host proofs read.  They fold the model over a whole
    block, so the invariant has to survive every instruction the machine
    actually executes — including the ones `denotes_step` says nothing about: a
    store, which writes memory and binds no value, and a call, whose result the
    model cannot see into. -/

/-- An instruction that binds nothing changes nothing. -/
theorem stepPure_nodest {e : Env} {i : Inst} {w : Val} (hd : Inst.destOf? i = none) :
    stepPure e i w = e w :=
  stepPure_frame i e w (fun _ hd' => by rw [hd] at hd'; exact absurd hd' (by simp))

theorem agree_nodest {vs : Vals} {e : Env} {i : Inst}
    (hag : Agree vs e) (hd : Inst.destOf? i = none) : Agree vs (stepPure e i) := by
  intro v k hv; rw [stepPure_nodest hd] at hv; exact hag v k hv

theorem denotes_nodest {vs : Vals} {e : Env} {i : Inst}
    (hden : Denotes vs e) (hd : Inst.destOf? i = none) : Denotes vs (stepPure e i) := by
  intro v dd hv; rw [stepPure_nodest hd] at hv; exact hden v dd hv

/-- **A destination the model refuses to name**, whatever the machine put there.

    This is the shape of a call: the import's result is a value `stepPure`
    cannot see into, and `stepPure_untracked` is what forces it to say so. -/
theorem agree_opaque_dest {vs : Vals} {e : Env} {i : Inst} {d : Val} {x : V}
    (hag : Agree vs e) (hd : Inst.destOf? i = some d)
    (hun : stepPure e i d = SymVal.unknown) : Agree (setV vs d x) (stepPure e i) := by
  intro v k hv
  by_cases hvd : v.id = d.id
  · have hveq : v = d := by cases v; cases d; simp_all
    rw [hveq, hun] at hv; exact absurd hv (by simp)
  · rw [stepPure_frame i e v (fun d' hd' => by rw [hd] at hd'; cases hd'; exact hvd)] at hv
    obtain ⟨t0, w, hw, htt, hs, hf⟩ := hag v k hv
    exact ⟨t0, w, getV_setV_ne hvd hw, htt, hs, hf⟩

theorem denotes_opaque_dest {vs : Vals} {e : Env} {i : Inst} {d : Val} {x : V}
    (hden : Denotes vs e) (hf : Fresh vs d) (hd : Inst.destOf? i = some d)
    (hun : stepPure e i d = SymVal.unknown) :
    Denotes (setV vs d x) (stepPure e i) := by
  intro v dd hv
  by_cases hvd : v.id = d.id
  · have hveq : v = d := by cases v; cases d; simp_all
    rw [hveq, hun] at hv; exact absurd hv (by simp [SymVal.toD?])
  · rw [stepPure_frame i e v (fun d' hd' => by rw [hd] at hd'; cases hd'; exact hvd)] at hv
    exact denotes_lift hf hvd (hden v dd hv)

/-! ### Freshness, as a property of the instruction list

    `Fresh` is what an SSA numbering gives, and it is a fact about the *text* of
    a block: destinations run upward from the value count the block entered
    with.  `DestsFrom` says exactly that, decidably, so a run instantiated at a
    shipped function discharges it by `decide` rather than by hypothesis. -/

/-- Binding a value grows the map to hold it and no further. -/
theorem size_setV (vs : Vals) (d : Val) (x : V) :
    (setV vs d x).size = max vs.size (d.id + 1) := by
  simp only [setV, Array.set!, Array.size_setIfInBounds]
  rcases Nat.lt_or_ge d.id vs.size with h | h
  · rw [if_pos h]; omega
  · rw [if_neg (Nat.not_lt.mpr h)]
    simp only [Array.size_append, Array.size_replicate]
    omega

def DestsFrom (n : Nat) : List Inst → Bool
  | []      => true
  | i :: is => match Inst.destOf? i with
               | some d => n ≤ d.id && DestsFrom (d.id + 1) is
               | none   => DestsFrom n is

/-! ### A partial static typing

    `stepPure` never sees a type, and `UnnamedTracked` is where that costs
    something.  Discharging it per program needs types the *text* determines,
    and only for the values it determines them for: a program is free to hold
    floats and bytes in values nothing here claims a type for. -/

/-- Where this names a type the machine holds a scalar of exactly that type.
    `none` claims nothing, which is what keeps it true of a program that also
    computes with `f32` and `i8`.

    Array-backed for the same reason `Clif.Env` is: a run instantiated at a
    shipped function looks a type up once per operand, and a closure chain
    makes that quadratic. -/
structure TyEnv where
  tys : Array (Option ClifTy)

def TyEnv.get (Θ : TyEnv) (v : Val) : Option ClifTy :=
  if h : v.id < Θ.tys.size then Θ.tys[v.id] else none

instance : CoeFun TyEnv (fun _ => Val → Option ClifTy) := ⟨TyEnv.get⟩

def TyEnv.empty : TyEnv := ⟨#[]⟩

def TyEnv.set (Θ : TyEnv) (v : Val) (t : Option ClifTy) : TyEnv :=
  if h : v.id < Θ.tys.size then ⟨Θ.tys.set v.id t h⟩
  else ⟨(Θ.tys ++ Array.replicate (v.id - Θ.tys.size) none).push t⟩

theorem TyEnv.set_apply (Θ : TyEnv) (v w : Val) (t : Option ClifTy) :
    (Θ.set v t) w = if w.id = v.id then t else Θ w := by
  by_cases h : v.id < Θ.tys.size
  · simp only [TyEnv.set, TyEnv.get, dif_pos h, Array.size_set]
    by_cases hw : w.id = v.id
    · rw [dif_pos (hw ▸ h), if_pos hw, Array.getElem_set, if_pos hw.symm]
    · by_cases hb : w.id < Θ.tys.size
      · rw [dif_pos hb, if_neg hw, dif_pos hb, Array.getElem_set,
            if_neg (fun hc => hw hc.symm)]
      · rw [dif_neg hb, if_neg hw, dif_neg hb]
  · have hs : Θ.tys.size ≤ v.id := Nat.le_of_not_lt h
    simp only [TyEnv.set, TyEnv.get, dif_neg h, Array.size_push, Array.size_append,
               Array.size_replicate]
    have hsz : Θ.tys.size + (v.id - Θ.tys.size) = v.id := Nat.add_sub_cancel' hs
    by_cases hw : w.id = v.id
    · rw [dif_pos (by omega), if_pos hw, Array.getElem_push, dif_neg (by simp; omega)]
    · by_cases hb : w.id < v.id + 1
      · rw [dif_pos (by omega), if_neg hw, Array.getElem_push, dif_pos (by simp; omega)]
        by_cases hb2 : w.id < Θ.tys.size
        · rw [Array.getElem_append_left hb2, dif_pos hb2]
        · rw [Array.getElem_append_right (by omega), dif_neg hb2,
              Array.getElem_replicate]
      · rw [dif_neg (by omega), if_neg hw, dif_neg (by omega)]

theorem TyEnv.set_eq (Θ : TyEnv) (v w : Val) (t : Option ClifTy) (h : w.id = v.id) :
    (Θ.set v t) w = t := by rw [TyEnv.set_apply, if_pos h]

theorem TyEnv.set_ne (Θ : TyEnv) (v w : Val) (t : Option ClifTy) (h : ¬ (w.id = v.id)) :
    (Θ.set v t) w = Θ w := by rw [TyEnv.set_apply, if_neg h]

/-- **What the static typing owes the machine.**  One direction only: where it
    names a type the value is a scalar of it.  A value it says nothing about is
    unconstrained, which is what a `none` is for. -/
def TypesAgree (Θ : TyEnv) (vs : Vals) : Prop :=
  ∀ v t, Θ v = some t → ∃ w, getV vs v = some (.sc t w)

theorem typesAgree_empty (vs : Vals) : TypesAgree TyEnv.empty vs := by
  intro v t hv
  simp only [TyEnv.empty, TyEnv.get] at hv
  exact absurd hv (by simp)

/-- The static typing, stepped alongside the model.  Only the arms whose result
    type the instruction itself determines say anything; every other
    destination is cleared, because a stale type is the same hazard as a stale
    binding. -/
def tyStep (Θ : TyEnv) : Inst → TyEnv
  | .iconst d t _   => Θ.set d (some t)
  | .iadd d a _ | .isub d a _ | .imul d a _ | .ishl d a _ | .ushr d a _
  | .ineg d a       => Θ.set d (Θ a)
  | .ireduce32 d _  => Θ.set d (some .i32)
  | .uextend64 d _ | .sextend64 d _ => Θ.set d (some .i64)
  -- a load's result type is the one the access names, at every kind; only a
  -- vector type gives a value this cannot describe
  | .load d op _    => Θ.set d (match op.ty.lanes with
                                | none   => some op.ty
                                | some _ => none)
  | i => match Inst.destOf? i with
         | some d => Θ.set d none
         | none   => Θ

/-- Clearing a destination is sound whatever the machine put there. -/
theorem typesAgree_clear {Θ : TyEnv} {vs : Vals} {d : Val} {x : V}
    (hta : TypesAgree Θ vs) : TypesAgree (Θ.set d none) (setV vs d x) := by
  intro v t hv
  by_cases hvd : v.id = d.id
  · rw [TyEnv.set_eq _ _ _ _ hvd] at hv; exact absurd hv (by simp)
  · rw [TyEnv.set_ne _ _ _ _ hvd] at hv
    obtain ⟨w, hw⟩ := hta v t hv
    exact ⟨w, getV_setV_ne hvd hw⟩

/-- Naming a destination's type is sound when the machine's word has it. -/
theorem typesAgree_dest {Θ : TyEnv} {vs : Vals} {d : Val} {x : V} {t : ClifTy}
    {w : UInt64} (hta : TypesAgree Θ vs) (hx : x = .sc t w) :
    TypesAgree (Θ.set d (some t)) (setV vs d x) := by
  intro v t' hv
  by_cases hvd : v.id = d.id
  · have hveq : v = d := by cases v; cases d; simp_all
    rw [TyEnv.set_eq _ _ _ _ hvd] at hv
    injection hv with ht
    exact ⟨w, by rw [hveq, ← ht, ← hx]; exact AlgorithmLib.HProg.getV_setV_self vs d x⟩
  · rw [TyEnv.set_ne _ _ _ _ hvd] at hv
    obtain ⟨w', hw'⟩ := hta v t' hv
    exact ⟨w', getV_setV_ne hvd hw'⟩

/-- An arm whose result carries an operand's type. -/
theorem typesAgree_copy {Θ : TyEnv} {vs : Vals} {a d : Val} {x : V} {t : ClifTy}
    {wa w : UInt64} (hta : TypesAgree Θ vs) (hga : getV vs a = some (.sc t wa))
    (hx : x = .sc t w) : TypesAgree (Θ.set d (Θ a)) (setV vs d x) := by
  rcases hqa : Θ a with _ | t'
  · exact typesAgree_clear hta
  · have hteq : t' = t := by
      obtain ⟨w', hw'⟩ := hta a t' hqa
      rw [hga] at hw'; injection hw' with h1; injection h1 with h2 _; exact h2.symm
    rw [hteq]; exact typesAgree_dest hta hx

/-- **The static typing is sound, over every instruction that computes.** -/
theorem tyStep_sound {m : Mem} {Θ : TyEnv} {vs : Vals} {i : Inst} {d : Val} {x : V}
    (hta : TypesAgree Θ vs) (hev : evalInst m vs i = some (d, x)) :
    TypesAgree (tyStep Θ i) (setV vs d x) := by
  cases i with
  | iconst d' t k =>
      have hq : d' = d ∧ x = ofInt t k := by
        simp [evalInst, AlgorithmLib.HProg.Blocks.viaOp, evalOp] at hev
        exact ⟨hev.1, hev.2.symm⟩
      rw [tyStep, hq.1]
      exact typesAgree_dest hta (by rw [hq.2]; exact ofInt_sc t k)
  | iadd d' a b =>
      obtain ⟨t, wa, wb, hga, _, _, hdd, hx⟩ := evalInst_iadd_inv hev
      rw [tyStep, hdd]; exact typesAgree_copy hta hga hx
  | isub d' a b =>
      obtain ⟨t, wa, wb, hga, _, _, hdd, hx⟩ := evalInst_isub_inv hev
      rw [tyStep, hdd]; exact typesAgree_copy hta hga hx
  | imul d' a b =>
      obtain ⟨t, wa, wb, hga, _, _, hdd, hx⟩ := evalInst_imul_inv hev
      rw [tyStep, hdd]; exact typesAgree_copy hta hga hx
  | ishl d' a b =>
      obtain ⟨ta, tb, wa, wb, hga, _, _, _, hdd, hx⟩ := evalInst_ishl_inv hev
      rw [tyStep, hdd]; exact typesAgree_copy hta hga hx
  | ushr d' a b =>
      obtain ⟨ta, tb, wa, wb, hga, _, _, _, hdd, hx⟩ := evalInst_ushr_inv hev
      rw [tyStep, hdd]; exact typesAgree_copy hta hga hx
  | ineg d' a =>
      obtain ⟨t, w, hga, _, hdd, hx⟩ := evalInst_ineg_inv hev
      rw [tyStep, hdd]; exact typesAgree_copy hta hga hx
  | ireduce32 d' a =>
      obtain ⟨t, w, _, _, _, hdd, hx⟩ := evalInst_ireduce32_inv hev
      rw [tyStep, hdd]; exact typesAgree_dest hta hx
  | uextend64 d' a =>
      obtain ⟨t, w, _, _, _, hdd, hx⟩ := evalInst_uextend64_inv hev
      rw [tyStep, hdd]; exact typesAgree_dest hta hx
  | sextend64 d' a =>
      obtain ⟨t, w, _, _, _, hdd, hx⟩ := evalInst_sextend64_inv hev
      rw [tyStep, hdd]
      exact typesAgree_dest hta (by rw [hx]; exact ofInt_sc _ _)
  | load d' op a =>
      rw [tyStep]
      rcases hl : op.ty.lanes with _ | ln
      · obtain ⟨hdd, w, hx⟩ := evalInst_load_inv hev hl
        rw [hdd]; exact typesAgree_dest hta hx
      · have hdd : d' = d := by
          have := evalInst_dest hev
          simp only [Inst.destOf?, Option.some.injEq] at this
          exact this
        rw [hdd]; exact typesAgree_clear hta
  | _ =>
      have hd := evalInst_dest hev
      simp only [Inst.destOf?, Option.some.injEq] at hd
      first
        | exact absurd hd (by simp)
        | (subst hd; exact typesAgree_clear hta)

/-- What the typing asks of an operand, decidably. -/
def TyTracked (Θ : TyEnv) (a : Val) : Bool :=
  match Θ a with
  | some .i32 | some .i64 => true
  | _                     => false

theorem trackedTy_of_tyTracked {Θ : TyEnv} {vs : Vals} {a : Val} {t : ClifTy} {w : UInt64}
    (hta : TypesAgree Θ vs) (h : TyTracked Θ a = true)
    (hga : getV vs a = some (.sc t w)) : TrackedTy t := by
  simp only [TyTracked] at h
  split at h
  · rename_i heq
    obtain ⟨w', hw'⟩ := hta a _ heq
    rw [hga] at hw'; injection hw' with h1; injection h1 with h2 _
    exact Or.inl h2
  · rename_i heq
    obtain ⟨w', hw'⟩ := hta a _ heq
    rw [hga] at hw'; injection hw' with h1; injection h1 with h2 _
    exact Or.inr h2
  · exact absurd h (by simp)

/-- The width exclusion, decided from the text.  Only the arms that turn an
    operand into a root or a base are asked, which is where `UnnamedTracked` is
    spent — see `narrowShiftDisagrees`. -/
def TyOperandsOk (Θ : TyEnv) (i : Inst) : Bool :=
  (Inst.readsOf i).all (TyTracked Θ)

theorem operandsTracked_of_ty {Θ : TyEnv} {vs : Vals} {e : Env} {i : Inst}
    (hta : TypesAgree Θ vs) (h : TyOperandsOk Θ i = true) :
    OperandsTracked vs e i := by
  intro a ha t w hga _
  exact trackedTy_of_tyTracked hta (by
    simp only [TyOperandsOk, List.all_eq_true] at h; exact h a (by simpa using ha)) hga

/-- The static typing, checked against a value map — one entry point a
    theorem instantiated at a shipped function can discharge by evaluation. -/
def tyCheckUpto (Θ : TyEnv) (vs : Vals) : Nat → Bool
  | 0     => true
  | k + 1 =>
      (match Θ ⟨k⟩ with
       | none   => true
       | some t => match getV vs ⟨k⟩ with
                   | some (.sc t' _) => t == t'
                   | _               => false)
      && tyCheckUpto Θ vs k

def TyEnv.checkAgainst (Θ : TyEnv) (vs : Vals) : Bool :=
  tyCheckUpto Θ vs Θ.tys.size

theorem tyCheckUpto_at {Θ : TyEnv} {vs : Vals} {t : ClifTy} :
    ∀ {k j : Nat}, tyCheckUpto Θ vs k = true → j < k → Θ ⟨j⟩ = some t →
      ∃ w, getV vs ⟨j⟩ = some (.sc t w) := by
  intro k
  induction k with
  | zero => intro j _ hj; omega
  | succ k ih =>
      intro j h hj hq
      simp only [tyCheckUpto, Bool.and_eq_true] at h
      rcases Nat.lt_or_ge j k with hlt | hge
      · exact ih h.2 hlt hq
      · have hjk : j = k := by omega
        subst hjk
        have h1 := h.1
        rw [hq] at h1
        simp only [] at h1
        split at h1
        · rename_i t' w' hg
          exact ⟨w', by rw [hg, ty_eq_of_beq h1]⟩
        · exact absurd h1 (by simp)

theorem typesAgree_of_check {Θ : TyEnv} {vs : Vals}
    (h : Θ.checkAgainst vs = true) : TypesAgree Θ vs := by
  intro v t hv
  have hlt : v.id < Θ.tys.size := by
    rcases Nat.lt_or_ge v.id Θ.tys.size with h1 | h1
    · exact h1
    · simp only [TyEnv.get, dif_neg (Nat.not_lt.mpr h1)] at hv
      exact absurd hv (by simp)
  exact tyCheckUpto_at h hlt hv

/-! ### The block

    A compiled block is a straight line ending in one terminator, and that is
    the shape stated here rather than assumed: `TermLast` is decidable, and it
    is what makes the model's fold and the machine's run consume the same
    instructions.  Everything else the machine does on the way — writing memory,
    calling an import — the invariants are indifferent to, because none of them
    mentions memory and the model refuses to name a call's result. -/

def Inst.isTerm : Inst → Bool
  | .ret _ | .jump _ _ | .brif _ _ _ _ _ => true
  | _ => false

/-- Only the last instruction is a terminator. -/
def TermLast : List Inst → Bool
  | []           => true
  | [_]          => true
  | i :: j :: is => !Inst.isTerm i && TermLast (j :: is)

/-- The static typing, folded over a block alongside the model. -/
def tyRun (Θ : TyEnv) : List Inst → TyEnv
  | []      => Θ
  | i :: is => tyRun (tyStep Θ i) is

/-- **The whole side condition, decided from the block's text.**

    Two things at once, because they are checked at the same place: every
    destination is numbered past the values already bound — what an SSA
    numbering gives, and what `Fresh` needs — and every operand whose width the
    model has to assume something about has a type that makes the assumption
    true. -/
def RunOk (Θ : TyEnv) (n : Nat) : List Inst → Bool
  | []      => true
  | i :: is => TyOperandsOk Θ i
      && (match Inst.destOf? i with
          | some d => (n ≤ d.id) && RunOk (tyStep Θ i) (d.id + 1) is
          | none   => RunOk (tyStep Θ i) n is)

/-- **What the model and the machine agree on at a point in the run.** -/
structure Sound (Θ : TyEnv) (vs : Vals) (e : Env) : Prop where
  agree   : Agree vs e
  denotes : Denotes vs e
  noBase  : NoBaseDerived e
  types   : TypesAgree Θ vs

theorem sound_empty : Sound TyEnv.empty #[] Env.empty :=
  ⟨agree_empty _, denotes_empty _, noBaseDerived_empty, typesAgree_empty _⟩

/-- One step, all four facts at once. -/
theorem sound_step {m : Mem} {Θ : TyEnv} {vs : Vals} {e : Env} {i : Inst} {d : Val} {x : V}
    (hs : Sound Θ vs e) (hty : TyOperandsOk Θ i = true) (hf : Fresh vs d)
    (hev : evalInst m vs i = some (d, x)) :
    Sound (tyStep Θ i) (setV vs d x) (stepPure e i) :=
  ⟨const_sound hs.agree hev,
   denotes_step hs.agree hs.denotes hs.noBase
     (operandsTracked_of_ty hs.types hty) hf hev,
   noBaseDerived_step hs.noBase,
   tyStep_sound hs.types hev⟩

/-- A destination the model refuses to name — a call's result. -/
theorem sound_opaque {Θ : TyEnv} {vs : Vals} {e : Env} {i : Inst} {d : Val} {x : V}
    (hs : Sound Θ vs e) (hf : Fresh vs d) (hd : Inst.destOf? i = some d)
    (hun : stepPure e i d = SymVal.unknown) (hcl : tyStep Θ i = Θ.set d none) :
    Sound (tyStep Θ i) (setV vs d x) (stepPure e i) :=
  ⟨agree_opaque_dest hs.agree hd hun,
   denotes_opaque_dest hs.denotes hf hd hun,
   noBaseDerived_step hs.noBase,
   by rw [hcl]; exact typesAgree_clear hs.types⟩

/-- An instruction that binds no value: a store, or a terminator. -/
theorem sound_nodest {Θ : TyEnv} {vs : Vals} {e : Env} {i : Inst}
    (hs : Sound Θ vs e) (hd : Inst.destOf? i = none) :
    Sound (tyStep Θ i) vs (stepPure e i) := by
  have hcl : tyStep Θ i = Θ := by
    cases i <;> simp_all [tyStep, Inst.destOf?]
  exact ⟨agree_nodest hs.agree hd, denotes_nodest hs.denotes hd,
         noBaseDerived_step hs.noBase, by rw [hcl]; exact hs.types⟩

/-- A store writes memory, and no value. -/
theorem doStore_vals {s s' : BSt} {v a : Val} {as : Option ClifTy}
    (h : doStore s v a as = .ok s') : s'.vals = s.vals := by
  simp only [doStore] at h
  split at h
  · split at h
    · split at h
      · injection h with h; rw [← h]
      · exact absurd h (by simp)
    · split at h
      · injection h with h; rw [← h]
      · exact absurd h (by simp)
  · exact absurd h (by simp)

/-- Binding a value past the end grows the map to exactly hold it. -/
theorem size_setV_fresh {vs : Vals} {d : Val} {x : V} (hf : Fresh vs d) :
    (setV vs d x).size = d.id + 1 := by
  simp only [Fresh] at hf
  rw [size_setV]; omega

theorem termLast_tail {i : Inst} {rest : List Inst} (h : TermLast (i :: rest) = true) :
    TermLast rest = true := by
  cases rest with
  | nil => rfl
  | cons j r => simp only [TermLast, Bool.and_eq_true] at h; exact h.2

theorem runOk_head {Θ : TyEnv} {n : Nat} {i : Inst} {rest : List Inst}
    (h : RunOk Θ n (i :: rest) = true) : TyOperandsOk Θ i = true := by
  simp only [RunOk, Bool.and_eq_true] at h; exact h.1

theorem runOk_tail_dest {Θ : TyEnv} {n : Nat} {i : Inst} {rest : List Inst} {d : Val}
    (h : RunOk Θ n (i :: rest) = true) (hd : Inst.destOf? i = some d) :
    n ≤ d.id ∧ RunOk (tyStep Θ i) (d.id + 1) rest = true := by
  simp only [RunOk, Bool.and_eq_true, hd, decide_eq_true_eq] at h
  exact ⟨h.2.1, h.2.2⟩

theorem runOk_tail_nodest {Θ : TyEnv} {n : Nat} {i : Inst} {rest : List Inst}
    (h : RunOk Θ n (i :: rest) = true) (hd : Inst.destOf? i = none) :
    RunOk (tyStep Θ i) n rest = true := by
  simp only [RunOk, Bool.and_eq_true, hd] at h; exact h.2

/-- **The launch model is sound over a whole block.**

    Everything above was one instruction; this is what the host proofs actually
    fold.  The machine runs `runInsts` — stores, calls and all — and the model
    runs `evalPure` over the same list, and at the terminator the two still
    agree about every value the model names.

    `RunOk` and `TermLast` are the only side conditions, and both are decided
    from the block's own text. -/
theorem sound_runInsts : ∀ (is : List Inst) (env : FnEnv) (s : BSt) (Θ : TyEnv)
    (e : Env) (r : BSt × Next) (w : World),
    Sound Θ s.vals e → TermLast is = true → RunOk Θ s.vals.size is = true →
    AlgorithmLib.HProg.Blocks.runInsts env s is = .ok r w →
    Sound (tyRun Θ is) r.1.vals (evalPure e is) := by
  intro is
  induction is with
  | nil => intro _ _ _ _ _ _ _ _ _ hr; exact absurd hr (by simp [AlgorithmLib.HProg.Blocks.runInsts])
  | cons i rest ih =>
    intro env s Θ e r w hs hterm hok hr
    -- the terminator arms: the machine stops, and `TermLast` says nothing follows
    have hlast : ∀ (_ : Inst.isTerm i = true), rest = [] := by
      intro hit
      cases rest with
      | nil => rfl
      | cons j rest' => simp [TermLast, hit] at hterm
    cases i with
    | ret =>
        rw [hlast rfl]
        simp only [AlgorithmLib.HProg.Blocks.runInsts, Outcome.ok.injEq] at hr
        rw [← hr.1]
        simp only [tyRun, evalPure]
        exact sound_nodest hs (by simp [Inst.destOf?])
    | jump t args =>
        rw [hlast rfl]
        simp only [AlgorithmLib.HProg.Blocks.runInsts] at hr
        split at hr
        · exact absurd hr (by simp)
        · simp only [Outcome.ok.injEq] at hr
          rw [← hr.1]
          simp only [tyRun, evalPure]
          exact sound_nodest hs (by simp [Inst.destOf?])
    | brif c tb ta eb ea =>
        rw [hlast rfl]
        simp only [AlgorithmLib.HProg.Blocks.runInsts] at hr
        split at hr
        · exact absurd hr (by simp)
        · split at hr
          · exact absurd hr (by simp)
          · simp only [Outcome.ok.injEq] at hr
            rw [← hr.1]
            simp only [tyRun, evalPure]
            exact sound_nodest hs (by simp [Inst.destOf?])
    | store v a =>
        have hnd : Inst.destOf? (Inst.store v a) = none := by simp [Inst.destOf?]
        simp only [AlgorithmLib.HProg.Blocks.runInsts] at hr
        simp only [tyRun, evalPure]
        split at hr
        · exact absurd hr (by simp)
        · rename_i s' hds
          have hv : s'.vals = s.vals := doStore_vals hds
          exact ih env s' _ _ r w (by rw [hv]; exact sound_nodest hs hnd)
            (termLast_tail hterm) (by rw [hv]; exact runOk_tail_nodest hok hnd) hr
    | storeTyped t v a =>
        have hnd : Inst.destOf? (Inst.storeTyped t v a) = none := by simp [Inst.destOf?]
        simp only [AlgorithmLib.HProg.Blocks.runInsts] at hr
        simp only [tyRun, evalPure]
        split at hr
        · exact absurd hr (by simp)
        · rename_i s' hds
          have hv : s'.vals = s.vals := doStore_vals hds
          exact ih env s' _ _ r w (by rw [hv]; exact sound_nodest hs hnd)
            (termLast_tail hterm) (by rw [hv]; exact runOk_tail_nodest hok hnd) hr
    | istore8 v a =>
        have hnd : Inst.destOf? (Inst.istore8 v a) = none := by simp [Inst.destOf?]
        simp only [AlgorithmLib.HProg.Blocks.runInsts] at hr
        simp only [tyRun, evalPure]
        split at hr
        · split at hr
          · exact ih env _ _ _ r w (by exact sound_nodest hs hnd)
              (termLast_tail hterm) (by exact runOk_tail_nodest hok hnd) hr
          · exact absurd hr (by simp)
        · exact absurd hr (by simp)
    | call dopt fn args =>
        simp only [AlgorithmLib.HProg.Blocks.runInsts] at hr
        simp only [tyRun, evalPure]
        split at hr
        · exact absurd hr (by simp)
        rename_i avs hargs
        split at hr
        · exact absurd hr (by simp)
        · exact absurd hr (by simp)
        rename_i fn hcallee
        split at hr
        · exact absurd hr (by simp)
        rename_i res w' hcall
        cases dopt with
        | none =>
            have hd : Inst.destOf? (Inst.call none fn args) = none := rfl
            exact ih env _ _ _ r w (by exact sound_nodest hs hd)
              (termLast_tail hterm) (by exact runOk_tail_nodest hok hd) hr
        | some dv =>
            cases res with
            | none => exact absurd hr (by simp)
            | some rv =>
                have hd : Inst.destOf? (Inst.call (some dv) fn args) = some dv := rfl
                obtain ⟨hle, hrest⟩ := runOk_tail_dest hok hd
                have hf : Fresh s.vals dv := hle
                have hun : stepPure e (Inst.call (some dv) fn args) dv = SymVal.unknown :=
                  stepPure_untracked _ _ _ hd (by simp [Inst.TrackedB])
                have hcl : tyStep Θ (Inst.call (some dv) fn args) = Θ.set dv none := rfl
                exact ih env _ _ _ r w (by exact sound_opaque hs hf hd hun hcl)
                  (termLast_tail hterm)
                  (by rw [show (setV s.vals dv rv).size = dv.id + 1 from size_setV_fresh hf]
                      exact hrest) hr
    | _ =>
        simp only [AlgorithmLib.HProg.Blocks.runInsts] at hr
        simp only [tyRun, evalPure]
        split at hr
        · exact absurd hr (by simp)
        · rename_i d xv hev
          have hd := evalInst_dest hev
          obtain ⟨hle, hrest⟩ := runOk_tail_dest hok hd
          have hf : Fresh s.vals d := hle
          exact ih env _ _ _ r w (by exact sound_step hs (runOk_head hok) hf hev)
            (termLast_tail hterm)
            (by rw [show (setV s.vals d xv).size = d.id + 1 from size_setV_fresh hf]
                exact hrest) hr

/-! ### The two claims, in the form the corpus used to sample

    `Denotes` states all three named claims at once, in terms of `DExp`.  What
    a consumer asks is narrower and phrased in the model's own vocabulary, so
    each is restated here — and each is now a theorem rather than a check over
    a grid. -/

/-- **A `derived` value denotes what the machine computes**, wherever
    `DExp.Exact` holds.  This is what `stepPure_derived_agree` sampled. -/
theorem denotes_derived {vs : Vals} {e : Env} {v : Val} {dx : DExp}
    {t : ClifTy} {w : UInt64} (hden : Denotes vs e) (hv : e v = .derived dx)
    (hg : getV vs v = some (.sc t w)) (hex : DExp.Exact (rhoOf vs) dx = true) :
    signed t w = DExp.eval (rhoOf vs) dx := by
  obtain ⟨t', w', hg', _, _, _, hc⟩ := hden v dx (by rw [hv]; rfl)
  rw [hg] at hg'
  injection hg' with h1; injection h1 with h2 h3
  subst h2; subst h3
  exact hc hex

/-- **An `offset` is its base plus its displacement**, wherever the sum is
    representable at the width it is computed at.

    The workhorse claim: every launch argument naming a PTX slot or a bind
    table is `ptr + k` for a runtime `ptr`.  `offsetIf` bounds the
    *displacement*, which is all the model can see; whether the sum wraps
    depends on a runtime base, so `inTy` is a condition the consumer carries.
    It is the congruence clause of `Denotes` that reaches this, not the
    equality clause — a pointer is far outside `foldableRange`, so `DExp.Exact`
    is false of it and says nothing. -/
theorem denotes_offset {vs : Vals} {e : Env} {v p : Val} {k : Int}
    {t : ClifTy} {w : UInt64} (hden : Denotes vs e) (hv : e v = .offset p k)
    (hg : getV vs v = some (.sc t w))
    (hin : inTy t (rhoOf vs p.id + k) = true) :
    signed t w = rhoOf vs p.id + k := by
  obtain ⟨t', w', hg', htt, _, hcg, _⟩ :=
    hden v (.add (.root p.id) (.lit k)) (by rw [hv]; rfl)
  rw [hg] at hg'
  injection hg' with h1; injection h1 with h2 h3
  subst h2; subst h3
  exact signed_of_congr_inTy htt hin (hcg rfl)

/-- The `.load` arm binds a slot only from an offset, so a consumer holding the
    model's answer holds the offset claim `slot_address` spends. -/
theorem stepPure_load_slot {e : Env} {d a p : Val} {k : Int} {op : LoadOp}
    (h : stepPure e (.load d op a) d = .slot p k) : e a = .offset p k := by
  rw [stepPure, Env.set_eq _ _ _ _ rfl] at h
  split at h
  · rename_i p' k' hea
    injection h with h1 h2
    rw [hea, h1, h2]
  · exact absurd h (by simp)

/-- **A `slot` names the address the load read.**

    The fourth claim's address half, and the half every consumer uses:
    `bufDescOf`'s near/far identity is about *which* address a handle came
    from, not about the word there.  It follows from the offset claim, so it
    needs no account of memory — which is what keeps it clear of the value
    half, where a byte-level reading of `Mem.load` and a condition on the
    block's stores both enter.

    `inTy` is the offset claim's own side condition; a pointer plus a
    displacement not representable at `i64` is not an address. -/
theorem slot_address {vs : Vals} {e : Env} {a p : Val} {k : Int} {wa pw : UInt64}
    (hden : Denotes vs e) (hea : e a = .offset p k)
    (hga : getV vs a = some (.sc .i64 wa)) (hgp : getV vs p = some (.sc .i64 pw))
    (hk : 0 ≤ k)
    (hin : inTy ClifTy.i64 (signed ClifTy.i64 pw + k) = true) :
    wa = pw + UInt64.ofNat k.toNat := by
  have htt : TrackedTy ClifTy.i64 := Or.inr rfl
  have hrho : rhoOf vs p.id = signed ClifTy.i64 pw := by
    simp only [rhoOf, hgp]
  have hsg : signed ClifTy.i64 wa = signed ClifTy.i64 pw + k := by
    have := denotes_offset hden hea hga (by rwa [hrho])
    rwa [hrho] at this
  have hca : Congr ClifTy.i64 wa (signed ClifTy.i64 pw + k) :=
    congr_of_signed' htt hsg
  have hcp : Congr ClifTy.i64 pw (signed ClifTy.i64 pw) := congr_of_signed' htt rfl
  have hck : Congr ClifTy.i64 (UInt64.ofNat k.toNat) k := by
    simp only [Congr, modOf]
    have h1 : (UInt64.ofNat k.toNat).toNat = k.toNat % 2 ^ 64 := rfl
    rw [h1]
    have h2 : ((k.toNat % 2 ^ 64 : Nat) : Int) = (k.toNat : Int) % ((2 ^ 64 : Nat) : Int) := rfl
    rw [h2, show (((2 ^ 64 : Nat) : Int)) = 18446744073709551616 from rfl,
        Int.toNat_of_nonneg hk]
    omega
  have hsum : Congr ClifTy.i64 (pw + UInt64.ofNat k.toNat) (signed ClifTy.i64 pw + k) :=
    congr_add' htt hcp hck
  have hwn : wa.toNat < 2 ^ 64 := wa.toNat_lt_size
  have hpn : (pw + UInt64.ofNat k.toNat).toNat < 2 ^ 64 :=
    (pw + UInt64.ofNat k.toNat).toNat_lt_size
  have hw : (wa.toNat : Int) < 18446744073709551616 := by omega
  have hp : ((pw + UInt64.ofNat k.toNat).toNat : Int) < 18446744073709551616 := by omega
  have e1 : (wa.toNat : Int) % 18446744073709551616 = (wa.toNat : Int) :=
    Int.emod_eq_of_lt (Int.natCast_nonneg _) hw
  have e2 : ((pw + UInt64.ofNat k.toNat).toNat : Int) % 18446744073709551616
      = ((pw + UInt64.ofNat k.toNat).toNat : Int) :=
    Int.emod_eq_of_lt (Int.natCast_nonneg _) hp
  simp only [Congr, modOf] at hca hsum
  have hq : (wa.toNat : Int) = ((pw + UInt64.ofNat k.toNat).toNat : Int) := by
    rw [← e1, ← e2, hca, hsum]
  have : wa.toNat = (pw + UInt64.ofNat k.toNat).toNat := by omega
  exact UInt64.toNat.inj this

/-! ### Instantiating a run at a compiled function

    A theorem whose side conditions no shipped function satisfies says nothing,
    so the way to apply this one is fixed here rather than rebuilt per caller:
    the entry block, the typing its parameters give, and both conditions decided
    from the function's own text. -/

/-- Block parameters, as the static typing knows them: the compiled form
    carries their types, and they are the only values a block enters with. -/
def seedTys (Θ : TyEnv) (ps : List (Val × ClifTy)) : TyEnv :=
  ps.foldl (fun Θ p => Θ.set p.1 (some p.2)) Θ

/-- The width condition alone, threaded through a block. -/
def TyRunOk (Θ : TyEnv) : List Inst → Bool
  | []      => true
  | i :: is => TyOperandsOk Θ i && TyRunOk (tyStep Θ i) is

/-- The width condition through every block in order, each entered with its own
    parameters bound.  This is the half that could be false of shipped code:
    the model assumes a width wherever it turns an operand it cannot name into
    a base, and at `i8` that assumption is wrong. -/
def TyBlocksOk (Θ : TyEnv) : List BlockData → Bool
  | []      => true
  | b :: bs =>
      let Θ' := seedTys Θ b.params
      TyRunOk Θ' b.insts && TyBlocksOk (tyRun Θ' b.insts) bs

def entryInsts (f : FuncData) : List Inst := (f.blocks.map (·.insts)).headD []

def entryParams (f : FuncData) : List (Val × ClifTy) :=
  (f.blocks.map (·.params)).headD []

def entryTys (f : FuncData) : TyEnv := seedTys TyEnv.empty (entryParams f)

/-- Both side conditions at the entry block, as one decidable check. -/
def EntryOk (f : FuncData) : Bool :=
  TermLast (entryInsts f)
    && RunOk (entryTys f) (entryParams f).length (entryInsts f)

/-- **The launch model is sound on a compiled function's entry block.**

    The entry block is where `launchesOf` starts from `Env.empty`, so it is the
    environment every later block inherits.  `EntryOk` is decided from the
    function's text; what is left is a statement about the caller — the machine
    entered with the parameters the compiled form declares. -/
theorem sound_entry {f : FuncData} {env : FnEnv} {s : BSt}
    {r : BSt × Next} {w : World}
    (hok : EntryOk f = true) (hsz : s.vals.size = (entryParams f).length)
    (hpar : TypesAgree (entryTys f) s.vals)
    (hr : AlgorithmLib.HProg.Blocks.runInsts env s (entryInsts f) = .ok r w) :
    Sound (tyRun (entryTys f) (entryInsts f)) r.1.vals
      (evalPure Env.empty (entryInsts f)) := by
  simp only [EntryOk, Bool.and_eq_true] at hok
  exact sound_runInsts (entryInsts f) env s (entryTys f) Env.empty r w
    ⟨agree_empty _, denotes_empty _, noBaseDerived_empty, hpar⟩
    hok.1 (by rw [hsz]; exact hok.2) hr

end AlgorithmLib.Clif.Check
