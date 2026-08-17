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

/-- Whether a case is one the condition admits, so the count below can show
    the check is not passing by refusing everything. -/
def derivLive (ta tb : ClifTy) (x y : Int) (mk : Val → Val → Val → Inst) : Bool :=
  let i := mk ⟨2⟩ ⟨0⟩ ⟨1⟩
  let e := stepPure Env.empty (.iconst ⟨1⟩ tb y)
  let vs : Vals := setV (setV #[] ⟨0⟩ (ofInt ta x)) ⟨1⟩ (ofInt tb y)
  match (stepPure e i) ⟨2⟩ with
  | .derived d => DExp.Exact (rhoOf vs) d
  | _          => false

/-- The three instructions that build an expression rather than fold. -/
def derivOps : List (Val → Val → Val → Inst) :=
  [ (fun d a b => .isub d a b)
  , (fun d a b => .ishl d a b)
  , (fun d a b => .ushr d a b) ]

def wideTypes : List ClifTy := [.i32, .i64]

def derivOk : Bool :=
  wideTypes.all fun ta => wideTypes.all fun tb => sample.all fun x => sample.all fun y =>
    derivOps.all fun f => derivCase ta tb x y f

def derivLiveCount : Nat :=
  (wideTypes.flatMap fun ta => wideTypes.flatMap fun tb => sample.flatMap fun x =>
    sample.flatMap fun y => derivOps.filter fun f => derivLive ta tb x y f).length

/-- **A `derived` value denotes what the machine computes, wherever
    `DExp.Exact` holds and the operands are `i32` or wider.**

    `DExp.Exact` is most of the content: `DExp.eval` is `Int` arithmetic, so it
    is the machine's answer only while nothing overflows and every `shr` shifts
    a non-negative value.  Both are conditions on runtime values, which is why
    they cannot be a guard inside `stepPure` and are instead what a theorem
    reading a launch bound out of a `derived` value has to carry.

    The width is the rest of it, and it is a restriction rather than a choice of
    sample.  `foldableRange` is exact at `i32` and wider; `litOk` refuses narrow
    literals, which is what lets the *constant* arms assume that width.  A
    `derived` value names a **runtime** operand instead, whose width the model
    never sees, so nothing carries the assumption across — see
    `narrowShiftDisagrees`. -/
theorem stepPure_derived_agree : derivOk = true := by native_decide

/-- **…on a condition that admits most of the sample rather than none.**  A
    `DExp.Exact` that was always false would satisfy the theorem above. -/
theorem derived_check_is_live : (2000 < derivLiveCount) = true := by native_decide

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

/-- **An `offset` is its base plus its displacement**, wherever the sum is
    representable at the width it is computed at.

    The workhorse claim: every launch argument naming a PTX slot or a bind
    table is `ptr + k` for a runtime `ptr`, so this is what the launch model
    rests on.  `offsetIf` bounds the *displacement*, which is all it can see;
    whether the sum wraps depends on the base, so — like `DExp.Exact` — it is
    a condition a consumer carries rather than a guard. -/
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
def slotInsts (k : Nat) : List Inst :=
  [ .iconst ⟨1⟩ .i64 (Int.ofNat k)
  , .iadd ⟨2⟩ ⟨0⟩ ⟨1⟩
  , .load ⟨3⟩ { ty := .i64 } ⟨2⟩ ]

/-- **What `slot p d` asserts**: the value was loaded from `p + d`.  Checked by
    reading that address again and comparing words — the model names an
    address, and the machine's own memory says what is there. -/
def slotCase (k : Nat) : Bool :=
  let vs0 : Vals := setV #[] ⟨0⟩ (.sc .i64 (addrOf .arena 0))
  match runInsts mem0 vs0 (slotInsts k), (slotInsts k).foldl stepPure Env.empty ⟨3⟩ with
  | some vs, .slot p d =>
      match getV vs ⟨3⟩, getV vs p with
      | some (.sc _ loaded), some (.sc _ pw) =>
          Mem.load mem0 (pw + UInt64.ofNat d.toNat) 8 == some loaded
      | _, _ => false
  | some _, _ => false
  | none,   _ => true

/-- Whether the model reported a `slot` at all, so the check above is known to
    be testing the arm it names. -/
def slotLive (k : Nat) : Bool :=
  match (slotInsts k).foldl stepPure Env.empty ⟨3⟩ with
  | .slot _ _ => true
  | _         => false

def slotOk : Bool := (List.range 200).all slotCase

def slotLiveCount : Nat := ((List.range 200).filter slotLive).length

/-- **A handle the model calls `slot p d` is the word at `p + d`.**

    This is what makes a buffer handle identifiable — without it Qwen2's `Wq`,
    `Wk` and `Wv` launches are the same record — so it is worth checking that
    the address the model names is the address the load read. -/
theorem stepPure_slot_agrees : slotOk = true := by native_decide

theorem slot_check_is_live : (slotLiveCount == 200) = true := by native_decide


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
  rw [stepPure, Env.set_eq _ _ _ _ rfl] at h; exact h

theorem stepPure_ireduce32_const {e : Env} {d a : Val} {k : Int}
    (h : stepPure e (.ireduce32 d a) d = .const k) : e a = .const k := by
  rw [stepPure, Env.set_eq _ _ _ _ rfl] at h; exact h

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

/-- **Every value the run has computed carries a tracked type.**

    The hypothesis `stepPure` cannot check for itself.  It is a property of the
    program's types, decidable from the emitted function, and it is exactly what
    rules out the narrow-width disagreements `narrowDerivFails` counts. -/
def TrackedVals (vs : Vals) : Prop :=
  ∀ (v : Val) (t : ClifTy) (w : UInt64), getV vs v = some (.sc t w) → TrackedTy t

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

/-- **What the model claims of a value it can name, and what it owes.**

    One invariant for all three claims at once.  `SymVal.toD?` is the model's
    own statement of when it can name a value without inventing a root, and it
    answers for `const`, `offset` and `derived` alike — so this is not three
    invariants stapled together but the single property those three cases were
    always instances of.  It is also, verbatim, the value half of
    `Qwen2NonVacuity.MetaFaithful`.

    Read the conclusion as an implication: the model does not assert that the
    machine's word equals `d.eval rho`, it asserts that it does *whenever the
    expression is exact*.  The roots are bounded so that extending the value map
    cannot change what the expression denotes.

    **Stated, with its pieces proved, and not yet preserved by a step.**  The
    frame, the operands, the constant arm and the retags all go through; the
    additive arms do not, and the reason is worth recording.  `iadd` of an
    `offset p k` by a constant `y` has to reach the operand's claim at
    `.add (root p) (lit k)`, whose `DExp.Exact` needs `inFold (rho p + k)` —
    which the *result*'s exactness does not supply, since `k` and `y` can be
    large and opposite.  The claim is nonetheless true, by the same
    congruence-modulo-width argument as `const_sound`; what it needs is an
    unconditional congruence clause beside the conditional equality.  `.shr` is
    what stops that being the whole invariant: division is not a congruence, so
    a `shrLit` value's claim is conditional however it is phrased. -/
def Denotes (vs : Vals) (e : Env) : Prop :=
  ∀ v d, (e v).toD? = some d →
    ∃ t w, getV vs v = some (.sc t w) ∧ TrackedTy t
      ∧ DExp.rootsLt vs.size d = true
      ∧ (DExp.Exact (rhoOf vs) d = true → signed t w = DExp.eval (rhoOf vs) d)

/-- The claim holds of a run that has bound nothing. -/
theorem denotes_empty (vs : Vals) : Denotes vs Env.empty := by
  intro v d hv
  simp only [Env.empty, Env.get] at hv
  exact absurd hv (by simp [SymVal.toD?])

/-- Binding a fresh destination leaves what the model already claimed intact:
    the word is still there, the roots are still in range, and the valuation
    did not move on any root the expression names. -/
theorem denotes_lift {vs : Vals} {d v : Val} {x : V} {t : ClifTy} {w : UInt64} {dd : DExp}
    (hf : Fresh vs d) (hne : v.id ≠ d.id)
    (hw : getV vs v = some (.sc t w)) (htt : TrackedTy t)
    (hr : DExp.rootsLt vs.size dd = true)
    (hc : DExp.Exact (rhoOf vs) dd = true → signed t w = DExp.eval (rhoOf vs) dd) :
    ∃ t' w', getV (setV vs d x) v = some (.sc t' w') ∧ TrackedTy t'
      ∧ DExp.rootsLt (setV vs d x).size dd = true
      ∧ (DExp.Exact (rhoOf (setV vs d x)) dd = true →
          signed t' w' = DExp.eval (rhoOf (setV vs d x)) dd) := by
  refine ⟨t, w, getV_setV_ne hne hw, htt, rootsLt_mono (size_le_setV vs d x) hr, ?_⟩
  intro hex
  rw [eval_congr (rhoOf_setV hf) hr]
  exact hc (by rw [← exact_congr (rhoOf_setV hf) hr]; exact hex)

/-- **What an operand denotes.**

    The one step that turns the invariant into arithmetic.  Either the model can
    name the operand, and the invariant says what its word is worth, or it
    cannot, and `dOf` makes it a root — whose valuation is *defined* to be that
    word.  The second case is why the claim needs no hypothesis about values the
    model does not track. -/
theorem operand_denotes {vs : Vals} {e : Env} {a : Val} {t : ClifTy} {w : UInt64}
    (hden : Denotes vs e) (hga : getV vs a = some (.sc t w)) :
    DExp.rootsLt vs.size (dOf e a) = true
      ∧ (DExp.Exact (rhoOf vs) (dOf e a) = true →
          signed t w = DExp.eval (rhoOf vs) (dOf e a)) := by
  rcases hd : (e a).toD? with _ | dd
  · have hroot : dOf e a = .root a.id := by
      cases hea : e a <;> rw [hea] at hd <;> simp_all [dOf, SymVal.toD?]
    have hsz : a.id < vs.size := by
      simp only [getV, Array.getElem?_eq_some_iff] at hga; exact hga.1
    refine ⟨by rw [hroot]; simp only [DExp.rootsLt, decide_eq_true_eq]; exact hsz, ?_⟩
    intro _
    rw [hroot]
    simp only [DExp.eval, rhoOf]
    rw [show (⟨a.id⟩ : Val) = a from by cases a; rfl, hga]
  · obtain ⟨t', w', hw', htt', hr', hc'⟩ := hden a dd hd
    rw [dOf_of_toD? e a dd hd]
    rw [hga] at hw'
    injection hw' with h1
    injection h1 with ht hww
    subst ht; subst hww
    exact ⟨hr', hc'⟩

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

/-- A value the step leaves alone still denotes what it did. -/
theorem denotes_frame {m : Mem} {vs : Vals} {e : Env} {i : Inst} {d v : Val} {x : V}
    {dd : DExp} (hden : Denotes vs e) (hf : Fresh vs d)
    (hev : evalInst m vs i = some (d, x)) (hvd : v.id ≠ d.id)
    (hv : (stepPure e i v).toD? = some dd) :
    ∃ t w, getV (setV vs d x) v = some (.sc t w) ∧ TrackedTy t
      ∧ DExp.rootsLt (setV vs d x).size dd = true
      ∧ (DExp.Exact (rhoOf (setV vs d x)) dd = true →
          signed t w = DExp.eval (rhoOf (setV vs d x)) dd) := by
  rw [stepPure_frame _ e v (fun d' hd' => by
        rw [evalInst_dest hev] at hd'; injection hd' with hq; exact hq ▸ hvd)] at hv
  obtain ⟨t, w, hw, htt, hr, hc⟩ := hden v dd hv
  exact denotes_lift hf hvd hw htt hr hc

/-- A value the model reports as a constant denotes the literal, and
    `const_sound` has already proved the machine agrees. -/
theorem denotes_of_const {vs : Vals} {e : Env} {v : Val} {k : Int}
    (hag : Agree vs e) (hv : e v = .const k) :
    ∃ t w, getV vs v = some (.sc t w) ∧ TrackedTy t
      ∧ DExp.rootsLt vs.size (.lit k) = true
      ∧ (DExp.Exact (rhoOf vs) (.lit k) = true →
          signed t w = DExp.eval (rhoOf vs) (.lit k)) := by
  obtain ⟨t, w, hw, htt, hs, _⟩ := hag v k hv
  exact ⟨t, w, hw, htt, rfl, fun _ => hs⟩

end AlgorithmLib.Clif.Check
