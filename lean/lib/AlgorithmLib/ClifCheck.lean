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
def derivCase (t : ClifTy) (x y : Int) (mk : Val → Val → Val → Inst) : Bool :=
  let i := mk ⟨2⟩ ⟨0⟩ ⟨1⟩
  let e := stepPure Env.empty (.iconst ⟨1⟩ t y)
  let vs : Vals := setV (setV #[] ⟨0⟩ (ofInt t x)) ⟨1⟩ (ofInt t y)
  let conc : Option V := (evalInst default vs i).map (·.2)
  match (stepPure e i) ⟨2⟩, conc with
  | .derived d, some (.sc t' w) =>
      !DExp.Exact (rhoOf vs) d || signed t' w == DExp.eval (rhoOf vs) d
  | _, _ => true

/-- Whether a case is one the condition admits, so the count below can show
    the check is not passing by refusing everything. -/
def derivLive (t : ClifTy) (x y : Int) (mk : Val → Val → Val → Inst) : Bool :=
  let i := mk ⟨2⟩ ⟨0⟩ ⟨1⟩
  let e := stepPure Env.empty (.iconst ⟨1⟩ t y)
  let vs : Vals := setV (setV #[] ⟨0⟩ (ofInt t x)) ⟨1⟩ (ofInt t y)
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
  wideTypes.all fun t => sample.all fun x => sample.all fun y =>
    derivOps.all fun f => derivCase t x y f

def derivLiveCount : Nat :=
  (wideTypes.flatMap fun t => sample.flatMap fun x => sample.flatMap fun y =>
    derivOps.filter fun f => derivLive t x y f).length

/-- **A `derived` value denotes what the machine computes, wherever
    `DExp.Exact` holds.**

    That condition is the whole content: `DExp.eval` is `Int` arithmetic, so
    it is the machine's answer only while nothing overflows and every `shr`
    shifts a non-negative value.  Both are conditions on runtime values, which
    is why they cannot be a guard inside `stepPure` and are instead what a
    theorem reading a launch bound out of a `derived` value has to carry. -/
theorem stepPure_derived_agree : derivOk = true := by native_decide

/-- **…on a condition that admits most of the sample rather than none.**  A
    `DExp.Exact` that was always false would satisfy the theorem above. -/
theorem derived_check_is_live : (2000 < derivLiveCount) = true := by native_decide

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
def offCase (t : ClifTy) (base k : Int) (mk : Val → Val → Val → Inst) : Bool :=
  let i := mk ⟨2⟩ ⟨0⟩ ⟨1⟩
  let e := stepPure Env.empty (.iconst ⟨1⟩ t k)
  let vs : Vals := setV (setV #[] ⟨0⟩ (ofInt t base)) ⟨1⟩ (ofInt t k)
  let conc : Option V := (evalInst default vs i).map (·.2)
  match (stepPure e i) ⟨2⟩, conc with
  | .offset p d, some (.sc t' w) =>
      let b := rhoOf vs p.id
      !inTy t (b + d) || signed t' w == b + d
  | _, _ => true

def offLive (t : ClifTy) (base k : Int) (mk : Val → Val → Val → Inst) : Bool :=
  let i := mk ⟨2⟩ ⟨0⟩ ⟨1⟩
  let e := stepPure Env.empty (.iconst ⟨1⟩ t k)
  let vs : Vals := setV (setV #[] ⟨0⟩ (ofInt t base)) ⟨1⟩ (ofInt t k)
  match (stepPure e i) ⟨2⟩ with
  | .offset p d => inTy t (rhoOf vs p.id + d)
  | _           => false

def offOps : List (Val → Val → Val → Inst) :=
  [ (fun d a b => .iadd d a b), (fun d a b => .isub d a b) ]

def offOk : Bool :=
  wideTypes.all fun t => bases.all fun b => sample.all fun k =>
    offOps.all fun f => offCase t b k f

def offLiveCount : Nat :=
  (wideTypes.flatMap fun t => bases.flatMap fun b => sample.flatMap fun k =>
    offOps.filter fun f => offLive t b k f).length

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

end AlgorithmLib.Clif.Check
