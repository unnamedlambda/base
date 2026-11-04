module
public import AlgorithmLib.Check.Calls
import all Init.Data.Repr
@[expose] public section

/-!
# `Check.Rel` — what the checker knows of slots relative to each other

A state's linear facts (`AS.syms`) and its slots' ranges make a `Ctx` the
prover reads (`AS.ctx`, sound by `AS.ctx_sound`). **`memRel`** asks it whether
bytes at an address lie inside one region, where the ranges alone cannot
say: the address no less than the region's base, and the bytes no further
than its size and its span.
-/

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare
  AlgorithmLib.HProg.DevSpec AlgorithmLib.Prog

namespace AlgorithmLib.Check

variable {vals : List UInt64}

open AlgorithmLib.HProg.Contracts (Fact TState holds_of_mem)

/-- The most room the facts give region `r`. -/
def roomOf (S : TState) (r : Region) : Nat :=
  S.foldl (fun m f => match f with | .room r' n => if r' = r then max m n else m | _ => m) 0

theorem roomOf_le {S : TState} {w : World} (hS : S.holds w) (r : Region) : roomOf S r ≤ w.mem.sizes r := by
  unfold roomOf
  suffices h : ∀ (l : List Fact) (m : Nat), m ≤ w.mem.sizes r → (∀ f ∈ l, f.holds w) →
      l.foldl (fun m f => match f with | .room r' n => if r' = r then max m n else m | _ => m) m ≤ w.mem.sizes r from
    h S 0 (Nat.zero_le _) hS
  intro l
  induction l with
  | nil => intro m hm _; exact hm
  | cons f l ih =>
      intro m hm hl
      rw [List.foldl_cons]
      refine ih _ ?_ (fun g hg => hl g (List.mem_cons_of_mem _ hg))
      have hf := hl f (List.mem_cons_self ..)
      cases f with
      | room r' n =>
          dsimp only
          split
          · rename_i e; subst e; exact Nat.max_le.mpr ⟨hm, hf⟩
          · exact hm
      | _ => exact hm

-- ---------------------------------------------------------------------------
-- The prover's context
-- ---------------------------------------------------------------------------

/-- The numbers a slot's bits may be, by what the checker knows of it. -/
def AV.rangeOf : AV → Int × Int
  | .known (.sc _ x) => (x.toNat, x.toNat)
  | .int n => (n.lo, n.hi)
  | .ptr g n => ((regionBase g).toNat + n.lo, (regionBase g).toNat + n.hi)
  | .cmp .. => (0, 1)
  | _ => (0, 2 ^ 64 - 1)

/-- A value's bits as a number, `0` for a vector. -/
def vVal : V → Int
  | .sc _ x => x.toNat
  | _ => 0

theorem slotVal_of {Γ : Env} {r : R} {v : V} (hv : Γ[r]? = some v) : slotVal Γ r = vVal v := by
  unfold slotVal vVal; rw [hv]; cases v <;> rfl

theorem AV.rangeOf_sound {Γ : Env} {w : World} {av : AV} {r : R} {v : V} (hv : Γ[r]? = some v)
    (hd : av.Desc Γ w v) : av.rangeOf.1 ≤ slotVal Γ r ∧ slotVal Γ r ≤ av.rangeOf.2 := by
  have h0 := slotVal_nonneg Γ r
  have h1 := slotVal_lt Γ r
  rw [slotVal_of hv] at h0 h1 ⊢
  cases av with
  | known u =>
      simp only [AV.Desc] at hd
      rw [hd] at h0 h1 ⊢
      cases u with
      | sc t x => simp [AV.rangeOf, vVal]
      | vec t xs => simp only [AV.rangeOf, vVal] at h0 h1 ⊢; omega
  | int n =>
      obtain ⟨x, rfl, hn⟩ := hd
      simp only [AV.rangeOf, vVal]
      exact ⟨by have := hn.1; omega, by have := hn.2.1; omega⟩
  | ptr g n =>
      obtain ⟨off, rfl, hn, hs⟩ := hd
      simp only [AV.rangeOf, vVal]
      rw [addr_toNat g (Nat.le_trans hn.2.1 hs)]
      exact ⟨by have := hn.1; omega, by have := hn.2.1; omega⟩
  | cmp c a b =>
      obtain ⟨t, x, y, -, -, -, rfl⟩ := hd
      simp only [AV.rangeOf, boolV, vVal]
      split <;> simp
  | sc t => exact ⟨h0, by simp only [AV.rangeOf]; omega⟩
  | hand k b => exact ⟨h0, by simp only [AV.rangeOf]; omega⟩
  | blk n b => exact ⟨h0, by simp only [AV.rangeOf]; omega⟩
  | into b lo hi => exact ⟨h0, by simp only [AV.rangeOf]; omega⟩
  | any => exact ⟨h0, by simp only [AV.rangeOf]; omega⟩

/-- A number a value's bits, as a number, are a multiple of. -/
def AV.absAl : AV → Nat
  | .known (.sc _ x) => x.toNat
  | .int n => n.al
  | .ptr g n => Nat.gcd n.al (regionBase g).toNat
  | _ => 1

theorem AV.absAl_dvd {Γ : Env} {w : World} {av : AV} {t : ClifTy} {x : UInt64}
    (hd : av.Desc Γ w (.sc t x)) : av.absAl ∣ x.toNat := by
  cases av with
  | known u => simp only [AV.Desc] at hd; subst hd; exact Nat.dvd_refl _
  | int n => obtain ⟨y, e, hn⟩ := hd; cases e; exact hn.2.2.1
  | ptr g n =>
      obtain ⟨off, e, hn, hs⟩ := hd
      cases e
      rw [addr_toNat g (Nat.le_trans hn.2.1 hs)]
      exact Nat.dvd_add (Nat.gcd_dvd_right _ _) (Nat.dvd_trans (Nat.gcd_dvd_left _ _) hn.2.2.1)
  | cmp => exact Nat.one_dvd _
  | sc => exact Nat.one_dvd _
  | hand => exact Nat.one_dvd _
  | blk => exact Nat.one_dvd _
  | into => exact Nat.one_dvd _
  | any => exact Nat.one_dvd _

/-- A slot's linear facts, with the bound its range's fit gives. -/
def Sym.withFit (σ : Sym) : AV → Sym
  | .int n =>
      match n.fit with
      | some (g, s, d) => if 0 < s then { σ with ubs := (s, ⟨[(.size g, 1)], -(d : Int)⟩) :: σ.ubs } else σ
      | none => σ
  | _ => σ

/-- What the prover reads of a state. -/
def AS.ctx (A : AS) : Ctx := ⟨fun r => (A.get r).rangeOf, fun r => (A.sym r).withFit (A.get r),
  fun g => roomOf A.facts g⟩

theorem AS.Holds.symHolds {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) (r : R) :
    (A.sym r).Holds Γ w (slotVal Γ r) := by
  simp only [AS.sym]
  split
  · rename_i hr; exact (h.1.2.1 r hr).1
  · exact Sym.holds_blank _ _ _

theorem AS.ctx_sound {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) : A.ctx.Sound Γ w := by
  refine ⟨fun r => ?_, fun r => ?_, fun g => by simp only [AS.ctx]; exact_mod_cast roomOf_le h.2.2 g⟩
  · simp only [AS.ctx]
    by_cases hr : r < A.len
    · obtain ⟨v, hv, hd⟩ := h.get hr
      exact AV.rangeOf_sound hv hd
    · rw [AS.get_out hr]
      have := slotVal_nonneg Γ r
      have := slotVal_lt Γ r
      simp only [AV.rangeOf]
      omega
  · simp only [AS.ctx]
    have hσ : (A.sym r).Holds Γ w (slotVal Γ r) := by
      simp only [AS.sym]
      split
      · rename_i hr; exact (h.1.2.1 r hr).1
      · exact Sym.holds_blank _ _ _
    unfold Sym.withFit
    split
    · rename_i n hn
      split
      · rename_i g s0 d hf
        split
        · refine ⟨hσ.1, fun p hp => ?_⟩
          simp only [List.mem_cons] at hp
          rcases hp with rfl | hp
          · by_cases hr : r < A.len
            · obtain ⟨v, hv, hd⟩ := h.get hr
              rw [hn] at hd
              obtain ⟨x, rfl, -, -, -, hfit⟩ := hd
              rw [hf] at hfit
              rw [slotVal_of hv]
              simp only [vVal, Lin.eval, Lin.evalT, Atom.val]
              have := hfit.2
              rw [Nat.mul_comm] at this
              push_cast
              omega
            · rw [AS.get_out hr] at hn; cases hn
          · exact hσ.2 p hp
        · exact hσ
      · exact hσ
    · exact hσ

-- ---------------------------------------------------------------------------
-- Memory by relations
-- ---------------------------------------------------------------------------

/-- How many rewriting steps the prover takes. -/
def proveFuel : Nat := 14

/-- Bytes `[a, a + len)` inside region `g`, by the prover: `len` not
    negative, `a` no less than the region's base and inside its span, and the
    bytes no further than its size, nor its span. -/
def memIn (A : AS) (a : R) (len : Lin) (g : Region) : Bool :=
  let b : Int := (regionBase g).toNat
  decide (g ≠ .pinned) && A.ctx.prove proveFuel (len.scale (-1)) &&
    A.ctx.prove proveFuel ((Lin.atom (.slot a)).add (len.add (((Lin.atom (.size g)).scale (-1)).add (Lin.const (-b))))) &&
    A.ctx.prove proveFuel (((Lin.atom (.slot a)).scale (-1)).add (Lin.const b)) &&
    A.ctx.prove proveFuel ((Lin.atom (.slot a)).add (len.add (Lin.const (-b - 2 ^ 36)))) &&
    A.ctx.prove proveFuel ((Lin.atom (.slot a)).add (Lin.const (1 - b - 2 ^ 36)))

/-- Bytes `[a, a + k)` inside some region, by the prover. -/
def memRel (A : AS) (a : R) (k : Nat) : Bool :=
  [Region.arena, .data, .out].any (memIn A a (Lin.const k))

/-- Bytes `[p, p + n)`, `n` a slot, inside some region, by the prover. -/
def memRel2 (A : AS) (p n : R) : Bool :=
  [Region.arena, .data, .out].any (memIn A p (Lin.atom (.slot n)))

theorem memIn_fits {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {a : R} {len : Lin} {g : Region}
    (hm : memIn A a len g = true) {t : ClifTy} {x : UInt64} (hx : Γ[a]? = some (.sc t x)) :
    Fits w.mem x (len.eval Γ w).toNat := by
  simp only [memIn, Bool.and_eq_true, decide_eq_true_eq] at hm
  obtain ⟨⟨⟨⟨⟨hg, h0⟩, h1⟩, h2⟩, h3⟩, h4⟩ := hm
  have hC := AS.ctx_sound h
  have e0 := Ctx.prove_sound hC h0
  have e1 := Ctx.prove_sound hC h1
  have e2 := Ctx.prove_sound hC h2
  have e3 := Ctx.prove_sound hC h3
  have e4 := Ctx.prove_sound hC h4
  have hv : slotVal Γ a = x.toNat := by unfold slotVal; rw [hx]
  simp only [Lin.eval_scale, Lin.eval_add, Lin.eval_atom, Lin.eval_const, Atom.val, hv] at e0 e1 e2 e3 e4
  have hb := Sem.regionBase_toNat g
  rw [Sem.regionSpan_toNat] at hb
  generalize len.eval Γ w = L at e0 e1 e3 ⊢
  obtain ⟨off, hoff⟩ : ∃ off, x.toNat = (regionBase g).toNat + off := ⟨x.toNat - (regionBase g).toNat, by omega⟩
  have hspan : off < span := by unfold span; omega
  have hxe : x = regionBase g + UInt64.ofNat off := by
    apply UInt64.toNat_inj.mp
    rw [addr_toNat g (Nat.le_of_lt hspan), hoff]
  subst hxe
  exact ⟨g, off, ptr_decode (Nat.le_refl _) hspan, hg, by omega, by omega⟩

theorem memRel_fits {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {a : R} {k : Nat}
    (hm : memRel A a k = true) {t : ClifTy} {x : UInt64} (hx : Γ[a]? = some (.sc t x)) : Fits w.mem x k := by
  obtain ⟨g, -, hg⟩ := List.any_eq_true.mp hm
  have := memIn_fits h hg hx
  simpa [Lin.eval_const] using this

theorem memRel2_fits {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {p n : R}
    (hm : memRel2 A p n = true) {t t' : ClifTy} {x k : UInt64} (hx : Γ[p]? = some (.sc t x))
    (hk : Γ[n]? = some (.sc t' k)) : Fits w.mem x k.toNat := by
  obtain ⟨g, -, hg⟩ := List.any_eq_true.mp hm
  have := memIn_fits h hg hx
  simpa [Lin.eval_atom, Atom.val, slotVal_of hk, vVal] using this

-- ---------------------------------------------------------------------------
-- What an operation's result is, relative to its operands
-- ---------------------------------------------------------------------------

def isI64 (A : AS) (r : R) : Bool := decide ((A.get r).scTy = some .i64)

def isIntSlot (A : AS) (r : R) : Bool :=
  match (A.get r).scTy with
  | some t => t.isInt
  | none => false

def lin1 (r : R) (c : Int) : Lin := ⟨[(.slot r, c)], 0⟩

/-- What an operation leaves, beyond its range: a 64-bit sum, difference,
    or multiple by a constant is a form of its operands modulo `2 ^ 64`; a
    shift right, quotient, mask, minimum or widening is no larger than what
    it is taken of. -/
def symOp (A : AS) : Op → Sym
  | .iadd a b =>
      if isI64 A a && isI64 A b then
        match (A.get b).bits with
        | some c =>
            -- a constant at or above `2^63` taken as the negative it is modulo `2^64`
            if 2 ^ 63 ≤ c.toNat then { dfn := some ⟨[(.slot a, 1)], (c.toNat : Int) - 2 ^ 64⟩ }
            else { dfn := some ⟨[(.slot a, 1), (.slot b, 1)], 0⟩ }
        | none => { dfn := some ⟨[(.slot a, 1), (.slot b, 1)], 0⟩ }
      else {}
  | .isub a b =>
      if isI64 A a && isI64 A b then { dfn := some ⟨[(.slot a, 1), (.slot b, -1)], 0⟩ } else {}
  | .imul a b =>
      match (A.get b).bits with
      | some c => if isI64 A a && isI64 A b then { dfn := some (lin1 a c.toNat) } else {}
      | none => {}
  | .ishl a b =>
      match (A.get b).bits with
      | some k => if isI64 A a && isIntSlot A b && k.toNat < 64 then { dfn := some (lin1 a (2 ^ k.toNat)) } else {}
      | none => {}
  | .ushr a b =>
      match (A.get b).bits with
      | some k => if isI64 A a && isIntSlot A b && k.toNat < 64 then { ubs := [(2 ^ k.toNat, lin1 a 1)] } else {}
      | none => {}
  | .udiv a b =>
      match (A.get b).bits with
      | some c => if isI64 A a && isI64 A b && 0 < c.toNat then { ubs := [(c.toNat, lin1 a 1)] } else {}
      | none => {}
  | .band a b => if isIntSlot A a && isIntSlot A b then { ubs := [(1, lin1 a 1), (1, lin1 b 1)] } else {}
  | .ibin .umin a b => if isIntSlot A a && isIntSlot A b then { ubs := [(1, lin1 a 1), (1, lin1 b 1)] } else {}
  | .uextend64 a => if isIntSlot A a then { ubs := [(1, lin1 a 1)] } else {}
  | .iun k a => if (k = .clz ∨ k = .bitrev) ∧ isI64 A a = true then { dfn := some (Lin.atom (.iun k a)) } else {}
  | _ => {}

theorem slot_i64 {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {r : R} (hr : isI64 A r = true) :
    ∃ x, Γ[r]? = some (.sc .i64 x) ∧ r < A.len := by
  have ht : (A.get r).scTy = some .i64 := by simpa [isI64] using hr
  obtain ⟨x, hx, -⟩ := h.getSc ht
  exact ⟨x, hx, AS.get_lt_scTy ht⟩

theorem slot_int {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {r : R} (hr : isIntSlot A r = true) :
    ∃ t x, Γ[r]? = some (.sc t x) ∧ t.isInt = true ∧ r < A.len := by
  unfold isIntSlot at hr
  split at hr
  · rename_i t ht
    obtain ⟨x, hx, -⟩ := h.getSc ht
    exact ⟨t, x, hx, hr, AS.get_lt_scTy ht⟩
  · cases hr

theorem AS.Holds.descAt {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {r : R} {u : V}
    (hr : Γ[r]? = some u) : (A.get r).Desc Γ w u := by
  have hl : r < A.len := by
    rw [← h.1.1]
    rcases Nat.lt_or_ge r Γ.size with h' | h'
    · exact h'
    · rw [Array.getElem?_eq_none h'] at hr; cases hr
  obtain ⟨u', hu', hd⟩ := h.get hl
  rw [hr] at hu'
  cases hu'
  exact hd

theorem slotVal_sc {Γ : Env} {r : R} {t : ClifTy} {x : UInt64} (h : Γ[r]? = some (.sc t x)) :
    slotVal Γ r = x.toNat := by
  rw [slotVal_of h]; rfl

/-- The new slot's facts, read in the environment it is pushed onto. -/
theorem fits_new {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {v : V} {σ : Sym}
    (hs : ∀ x ∈ σ.slots, x < A.len) (hg : ∀ g ∈ σ.regions, g ≠ .pinned)
    (hh : σ.Holds Γ w (vVal v)) : σ.Fits (Γ.push v) w A.len (A.len + 1) := by
  have hsz := h.1.1
  have hpush : ∀ x, x < A.len → slotVal (Γ.push v) x = slotVal Γ x :=
    fun x hx => slotVal_push (Γ := Γ) (v := v) (hsz ▸ hx)
  have hnew : slotVal (Γ.push v) A.len = vVal v := slotVal_of (by rw [Array.getElem?_push]; simp [hsz])
  have he : ∀ L : Lin, (∀ x ∈ L.slots, x ∈ σ.slots) → L.eval (Γ.push v) w = L.eval Γ w := fun L hL =>
    Lin.eval_congr (fun x hx => hpush x (hs x (hL x hx))) (fun _ _ => rfl)
  obtain ⟨hd, hu⟩ := hh
  refine ⟨⟨fun D e => ?_, fun p hp => ?_⟩, fun x hx => Nat.lt_succ_of_lt (hs x hx), hg⟩
  · rw [hnew, he D (fun x hx => by simp [Sym.slots, e, hx])]; exact hd D e
  · rw [hnew, he p.2 (fun x hx => by
      simp only [Sym.slots, List.mem_append, List.mem_flatten, List.mem_map]
      exact .inr ⟨_, ⟨p, hp, rfl⟩, hx⟩)]
    exact hu p hp

theorem lin1_slots (r : R) (c : Int) : ∀ x ∈ (lin1 r c).slots, x = r := by
  intro x hx; simp [lin1, Lin.slots] at hx; exact hx

theorem lin1_regions (r : R) (c : Int) : ∀ g ∈ (lin1 r c).regions, g ≠ .pinned := by
  intro g hg; simp [lin1, Lin.regions] at hg

theorem eval_lin1 (Γ : Env) (w : World) (r : R) (c : Int) : (lin1 r c).eval Γ w = c * slotVal Γ r := by
  simp [lin1, Lin.eval, Lin.evalT, Atom.val]

/-- A difference the prover shows does not wrap: no more than what is taken
    from. -/
def subNoWrap (A : AS) (a b : R) : Option AV :=
  if isI64 A a && isI64 A b && A.ctx.prove proveFuel ((lin1 b 1).sub (lin1 a 1)) then
    some (.int ⟨.i64, 0, ((A.get a).rangeOf.2).toNat, 1, none⟩)
  else none

/-- Two descriptions of one integer: the first, no larger than the second. -/
def AV.capBy (av : AV) : AV → AV
  | .int m =>
      match av with
      | .int n => if n.t = m.t then .int { n with hi := min n.hi m.hi } else av
      | .sc t => if t = m.t then .int m else av
      | _ => av
  | _ => av

/-- A heap block's start, or an address into one, and a number known exactly
    more: an address into the block. -/
def intoAV (A : AS) (a b : R) : Option AV :=
  if isI64 A a && isI64 A b then
    match (A.get b).bits with
    | some c =>
        match A.get a with
        | .blk _ _ => some (.into a c.toNat c.toNat)
        | .into r lo hi => some (.into r (lo + c.toNat) (hi + c.toNat))
        | _ => none
    | none => none
  else none

/-- What an operation leaves, sharpened by the linear facts, or an address
    into a heap block. -/
def opAV (A : AS) (o : Op) (av : AV) : AV :=
  match o with
  | .isub a b =>
      match subNoWrap A a b with
      | some m => av.capBy m
      | none => av
  | .iadd a b => (intoAV A a b).getD av
  | _ => av

theorem opAV_desc {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {o : Op} {av : AV} {v : V}
    (hv : evalOp w.mem Γ o = some v) (hd : av.Desc (Γ.push v) w v) : (opAV A o av).Desc (Γ.push v) w v := by
  cases o
  case isub a b =>
    simp only [opAV]
    cases e : subNoWrap A a b with
    | none => exact hd
    | some av' =>
        simp only
        suffices hm : av'.Desc (Γ.push v) w v by
          unfold AV.capBy
          split
          · rename_i m
            split
            · rename_i n
              split
              · rename_i ht
                obtain ⟨x, rfl, h1, h2, h3, h4⟩ := hd
                obtain ⟨y, ey, -, h2', -, -⟩ := hm
                simp only [V.sc.injEq] at ey
                obtain ⟨-, rfl⟩ := ey
                exact ⟨x, rfl, h1, Nat.le_min.mpr ⟨h2, h2'⟩, h3, h4⟩
              · exact hd
            · rename_i t
              split
              · rename_i ht; subst ht; exact hm
              · exact hd
            · exact hd
          · exact hd
        unfold subNoWrap at e
        split at e
        · rename_i hc
          cases e
          simp only [Bool.and_eq_true] at hc
          obtain ⟨⟨hai, hbi⟩, hp⟩ := hc
          obtain ⟨x, hx, ha⟩ := slot_i64 h hai
          obtain ⟨y, hy, hb⟩ := slot_i64 h hbi
          have e2 : v = .sc .i64 (x - y) := by
            simp [evalOp, bin, Sem.get, hx, hy, norm, and_wide64, ClifTy.isInt, clifTy_beq_self] at hv; exact hv.symm
          subst e2
          have hle := Ctx.prove_sound (AS.ctx_sound h) hp
          simp only [Lin.eval_sub, eval_lin1, slotVal_sc hx, slotVal_sc hy] at hle
          have hxy : y.toNat ≤ x.toNat := by omega
          have hr := (AV.rangeOf_sound hx (h.descAt hx)).2
          rw [slotVal_sc hx] at hr
          refine ⟨x - y, rfl, Nat.zero_le _, ?_, Nat.one_dvd _, trivial⟩
          rw [UInt64.toNat_sub_of_le _ _ (UInt64.le_iff_toNat_le.mpr hxy)]
          dsimp only
          omega
        · cases e
  case iadd a b =>
    simp only [opAV]
    cases e : intoAV A a b with
    | none => exact hd
    | some av' =>
        simp only [Option.getD_some]
        unfold intoAV at e
        split at e
        · rename_i hc
          simp only [Bool.and_eq_true] at hc
          obtain ⟨x, hx, ha⟩ := slot_i64 h hc.1
          obtain ⟨y, hy, hb⟩ := slot_i64 h hc.2
          have e2 : v = .sc .i64 (x + y) := by
            simp [evalOp, bin, Sem.get, hx, hy, norm, and_wide64, ClifTy.isInt, clifTy_beq_self] at hv; exact hv.symm
          subst e2
          have hxp : (Γ.push (V.sc .i64 (x + y)))[a]? = some (.sc .i64 x) := by
            have hlt : a < Γ.size := by rw [h.1.1]; exact ha
            rw [Array.getElem?_push]
            split
            · rename_i heq; exact absurd heq (Nat.ne_of_lt hlt)
            · exact hx
          split at e
          · rename_i c hcb
            obtain ⟨t, ey⟩ := AV.bits_desc (h.descAt hy) hcb
            simp only [V.sc.injEq] at ey
            obtain ⟨-, rfl⟩ := ey
            split at e
            · cases e
              exact ⟨x, y.toNat, hxp, by rw [UInt64.ofNat_toNat], Nat.le_refl _, Nat.le_refl _⟩
            · rename_i r lo hi hga
              cases e
              have hda := h.descAt hx
              rw [hga] at hda
              obtain ⟨base, off, hr, ex, hlo, hhi⟩ := hda
              simp only [V.sc.injEq, true_and] at ex
              subst ex
              refine ⟨base, off + y.toNat, ?_, ?_, by omega, by omega⟩
              · rw [Array.getElem?_push, if_neg]
                · exact hr
                · intro e; subst e; rw [Array.getElem?_eq_none (Nat.le_refl _)] at hr; cases hr
              · rw [UInt64.ofNat_add, UInt64.ofNat_toNat, UInt64.add_assoc]
            · cases e
          · cases e
        · cases e
  all_goals exact hd

/-- **What `symOp` says of an operation's result is true.** -/
theorem symOp_fits {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {o : Op} {v : V}
    (hv : evalOp w.mem Γ o = some v) : (symOp A o).Fits (Γ.push v) w A.len (A.len + 1) := by
  have blank : ({} : Sym).Fits (Γ.push v) w A.len (A.len + 1) := Sym.fits_blank _ _ _ _
  cases o
  case iadd a b =>
    simp only [symOp]
    split
    · rename_i hc
      simp only [Bool.and_eq_true] at hc
      obtain ⟨x, hx, ha⟩ := slot_i64 h hc.1
      obtain ⟨y, hy, hb⟩ := slot_i64 h hc.2
      have e : v = .sc .i64 (x + y) := by
        simp [evalOp, bin, Sem.get, hx, hy, norm, and_wide64, ClifTy.isInt, clifTy_beq_self] at hv; exact hv.symm
      subst e
      have both : ({ dfn := some ⟨[(.slot a, 1), (.slot b, 1)], 0⟩ } : Sym).Fits
          (Γ.push (.sc .i64 (x + y))) w A.len (A.len + 1) := by
        refine fits_new h (fun z hz => ?_) (fun g hg => by simp [Sym.regions, Lin.regions] at hg)
          ⟨fun D hD => ?_, fun p hp => by simp at hp⟩
        · rcases (by simpa [Sym.slots, Lin.slots] using hz : z = a ∨ z = b) with rfl | rfl <;> assumption
        · simp only [Option.some.injEq] at hD; subst hD
          simp only [Lin.eval, Lin.evalT, Atom.val, slotVal_sc hx, slotVal_sc hy, vVal, UInt64.toNat_add]
          omega
      split
      · rename_i c hcb
        split
        · rename_i hneg
          obtain ⟨t, e⟩ := AV.bits_desc (h.descAt hy) hcb
          cases e
          refine fits_new h (fun z hz => ?_) (fun g hg => by simp [Sym.regions, Lin.regions] at hg)
            ⟨fun D hD => ?_, fun p hp => by simp at hp⟩
          · rw [(by simpa [Sym.slots, Lin.slots] using hz : z = a)]; exact ha
          · simp only [Option.some.injEq] at hD; subst hD
            simp only [Lin.eval, Lin.evalT, Atom.val, slotVal_sc hx, vVal, UInt64.toNat_add]
            have := x.toNat_lt; have := y.toNat_lt
            omega
        · exact both
      · exact both
    · exact blank
  case isub a b =>
    simp only [symOp]
    split
    · rename_i hc
      simp only [Bool.and_eq_true] at hc
      obtain ⟨x, hx, ha⟩ := slot_i64 h hc.1
      obtain ⟨y, hy, hb⟩ := slot_i64 h hc.2
      have e : v = .sc .i64 (x - y) := by
        simp [evalOp, bin, Sem.get, hx, hy, norm, and_wide64, ClifTy.isInt, clifTy_beq_self] at hv; exact hv.symm
      subst e
      refine fits_new h (fun z hz => ?_) (fun g hg => by simp [Sym.regions, Lin.regions] at hg)
        ⟨fun D hD => ?_, fun p hp => by simp at hp⟩
      · rcases (by simpa [Sym.slots, Lin.slots] using hz : z = a ∨ z = b) with rfl | rfl <;> assumption
      · simp only [Option.some.injEq] at hD; subst hD
        simp only [Lin.eval, Lin.evalT, Atom.val, slotVal_sc hx, slotVal_sc hy, vVal, UInt64.toNat_sub]
        have := x.toNat_lt; have := y.toNat_lt
        omega
    · exact blank
  case imul a b =>
    simp only [symOp]
    split
    · rename_i c hcb
      split
      · rename_i hc
        simp only [Bool.and_eq_true] at hc
        obtain ⟨x, hx, ha⟩ := slot_i64 h hc.1
        obtain ⟨y, hy, hb⟩ := slot_i64 h hc.2
        obtain ⟨t, e⟩ := AV.bits_desc (h.descAt hy) hcb
        cases e
        have e : v = .sc .i64 (x * c) := by
          simp [evalOp, bin, Sem.get, hx, hy, norm, and_wide64, ClifTy.isInt, clifTy_beq_self] at hv; exact hv.symm
        subst e
        refine fits_new h (fun z hz => by rw [lin1_slots a _ z (by simpa [Sym.slots] using hz)]; exact ha)
          (fun g hg => lin1_regions a _ g (by simpa [Sym.regions] using hg)) ⟨fun D hD => ?_, fun p hp => by simp at hp⟩
        simp only [Option.some.injEq] at hD; subst hD
        rw [eval_lin1, slotVal_sc hx]
        simp only [vVal, UInt64.toNat_mul]
        rw [Int.mul_comm, ← Int.natCast_mul]
        generalize x.toNat * c.toNat = p
        omega
      · exact blank
    · exact blank
  case ishl a b =>
    simp only [symOp]
    split
    · rename_i k hkb
      split
      · rename_i hc
        simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
        obtain ⟨⟨hai, hbi⟩, hk⟩ := hc
        obtain ⟨x, hx, ha⟩ := slot_i64 h hai
        obtain ⟨tb, y, hy, htb, hb⟩ := slot_int h hbi
        obtain ⟨t, e⟩ := AV.bits_desc (h.descAt hy) hkb
        cases e
        have hmod : k % UInt64.ofNat (ClifTy.width .i64) = k := by
          apply UInt64.toNat_inj.mp
          rw [UInt64.toNat_mod]; simp only [ClifTy.width]; exact Nat.mod_eq_of_lt hk
        have e : v = .sc .i64 (x <<< k) := by
          simp [evalOp, shiftBin, Sem.get, hx, hy, norm, and_wide64, htb, show ClifTy.isInt .i64 = true from rfl] at hv
          rw [← hv, hmod]
        subst e
        refine fits_new h (fun z hz => by rw [lin1_slots a _ z (by simpa [Sym.slots] using hz)]; exact ha)
          (fun g hg => lin1_regions a _ g (by simpa [Sym.regions] using hg)) ⟨fun D hD => ?_, fun p hp => by simp at hp⟩
        simp only [Option.some.injEq] at hD; subst hD
        rw [eval_lin1, slotVal_sc hx]
        simp only [vVal, UInt64.toNat_shiftLeft, Nat.mod_eq_of_lt hk, Nat.shiftLeft_eq]
        have e2 : ((2 : Int) ^ k.toNat) = ((2 ^ k.toNat : Nat) : Int) := by simp
        rw [e2, Int.mul_comm, ← Int.natCast_mul]
        generalize x.toNat * 2 ^ k.toNat = p
        omega
      · exact blank
    · exact blank
  case ushr a b =>
    simp only [symOp]
    split
    · rename_i k hkb
      split
      · rename_i hc
        simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
        obtain ⟨⟨hai, hbi⟩, hk⟩ := hc
        obtain ⟨x, hx, ha⟩ := slot_i64 h hai
        obtain ⟨tb, y, hy, htb, hb⟩ := slot_int h hbi
        obtain ⟨t, e⟩ := AV.bits_desc (h.descAt hy) hkb
        cases e
        have hmod : k % UInt64.ofNat (ClifTy.width .i64) = k := by
          apply UInt64.toNat_inj.mp
          rw [UInt64.toNat_mod]; simp only [ClifTy.width]; exact Nat.mod_eq_of_lt hk
        have e : v = .sc .i64 (x >>> k) := by
          simp [evalOp, shiftBin, Sem.get, hx, hy, norm, and_wide64, htb, show ClifTy.isInt .i64 = true from rfl] at hv
          rw [← hv, hmod]
        subst e
        refine fits_new h (fun z hz => ?_) (fun g hg => ?_) ⟨fun D hD => by simp at hD, fun p hp => ?_⟩
        · rw [(by simpa [Sym.slots, lin1, Lin.slots] using hz : z = a)]; exact ha
        · simp [Sym.regions, lin1, Lin.regions] at hg
        · simp only [List.mem_singleton] at hp; subst hp
          simp only [eval_lin1, slotVal_sc hx, vVal, UInt64.toNat_shiftRight, Nat.mod_eq_of_lt hk,
            Nat.shiftRight_eq_div_pow]
          have := Nat.div_mul_le_self x.toNat (2 ^ k.toNat)
          rw [Nat.mul_comm] at this
          rw [Int.one_mul]
          exact_mod_cast this
      · exact blank
    · exact blank
  case udiv a b =>
    simp only [symOp]
    split
    · rename_i c hcb
      split
      · rename_i hc
        simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
        obtain ⟨⟨hai, hbi⟩, hc0⟩ := hc
        obtain ⟨x, hx, ha⟩ := slot_i64 h hai
        obtain ⟨y, hy, hb⟩ := slot_i64 h hbi
        obtain ⟨t, e⟩ := AV.bits_desc (h.descAt hy) hcb
        cases e
        have hc0' : (c == 0) = false := by
          simp only [beq_eq_false_iff_ne, ne_eq]; intro e; subst e; simp at hc0
        have e : v = .sc .i64 (x / c) := by
          simp [evalOp, bin, Sem.get, hx, hy, norm, and_wide64, ClifTy.isInt, clifTy_beq_self, hc0'] at hv
          exact hv.2.symm
        subst e
        refine fits_new h (fun z hz => ?_) (fun g hg => ?_) ⟨fun D hD => by simp at hD, fun p hp => ?_⟩
        · rw [(by simpa [Sym.slots, lin1, Lin.slots] using hz : z = a)]; exact ha
        · simp [Sym.regions, lin1, Lin.regions] at hg
        · simp only [List.mem_singleton] at hp; subst hp
          simp only [eval_lin1, slotVal_sc hx, vVal, UInt64.toNat_div]
          have := Nat.div_mul_le_self x.toNat c.toNat
          rw [Nat.mul_comm] at this
          rw [Int.one_mul]
          exact_mod_cast this
      · exact blank
    · exact blank
  case band a b =>
    simp only [symOp]
    split
    · rename_i hc
      simp only [Bool.and_eq_true] at hc
      obtain ⟨ta, x, hx, hta, ha⟩ := slot_int h hc.1
      obtain ⟨tb, y, hy, htb, hb⟩ := slot_int h hc.2
      have hv' : ∃ z, v = .sc ta z ∧ z.toNat ≤ x.toNat ∧ z.toNat ≤ y.toNat := by
        simp only [evalOp, Sem.get, hx, hy, Option.bind_eq_bind, Option.bind_some, zipIntBits] at hv
        split at hv
        · rename_i hc'
          cases hv
          refine ⟨_, rfl, ?_, ?_⟩ <;> simp only [UInt64.toNat_and]
          · exact Nat.le_trans Nat.and_le_left Nat.and_le_left
          · exact Nat.le_trans Nat.and_le_left Nat.and_le_right
        · cases hv
      obtain ⟨z, rfl, hzx, hzy⟩ := hv'
      refine fits_new h (fun r hr => ?_) (fun g hg => ?_) ⟨fun D hD => by simp at hD, fun p hp => ?_⟩
      · first
          | (rcases (by simpa [Sym.slots, lin1, Lin.slots] using hr : r = a ∨ r = b) with rfl | rfl <;> assumption)
          | (rw [(by simpa [Sym.slots, lin1, Lin.slots] using hr : r = a)]; exact ha)
      · simp [Sym.regions, lin1, Lin.regions] at hg
      · simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at hp
        rcases hp with rfl | rfl
        · simp only [eval_lin1, slotVal_sc hx, vVal]; push_cast; simp only [Int.one_mul]; omega
        · simp only [eval_lin1, slotVal_sc hy, vVal]; push_cast; simp only [Int.one_mul]; omega
    · exact blank
  case ibin k a b =>
    cases k
    case umin =>
      simp only [symOp]
      split
      · rename_i hc
        simp only [Bool.and_eq_true] at hc
        obtain ⟨ta, x, hx, hta, ha⟩ := slot_int h hc.1
        obtain ⟨tb, y, hy, htb, hb⟩ := slot_int h hc.2
        have hv' : ∃ z, v = .sc ta z ∧ z.toNat ≤ x.toNat ∧ z.toNat ≤ y.toNat := by
          simp only [evalOp, bin, Sem.get, hx, hy, Option.bind_eq_bind, Option.bind_some] at hv
          split at hv
          · rename_i hc'
            simp only [ibinOp, norm, Option.some.injEq] at hv
            subst hv
            refine ⟨_, rfl, ?_, ?_⟩
            all_goals
              rw [UInt64.toNat_and, UInt64.toNat_ofNat']
              refine Nat.le_trans Nat.and_le_left (Nat.le_trans (Nat.mod_le _ _) ?_)
            · exact Nat.le_trans (Nat.min_le_left _ _) (by rw [UInt64.toNat_and]; exact Nat.and_le_left)
            · exact Nat.le_trans (Nat.min_le_right _ _) (by rw [UInt64.toNat_and]; exact Nat.and_le_left)
          · cases hv
        obtain ⟨z, rfl, hzx, hzy⟩ := hv'
        refine fits_new h (fun r hr => ?_) (fun g hg => ?_) ⟨fun D hD => by simp at hD, fun p hp => ?_⟩
        · first
          | (rcases (by simpa [Sym.slots, lin1, Lin.slots] using hr : r = a ∨ r = b) with rfl | rfl <;> assumption)
          | (rw [(by simpa [Sym.slots, lin1, Lin.slots] using hr : r = a)]; exact ha)
        · simp [Sym.regions, lin1, Lin.regions] at hg
        · simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at hp
          rcases hp with rfl | rfl
          · simp only [eval_lin1, slotVal_sc hx, vVal]; push_cast; simp only [Int.one_mul]; omega
          · simp only [eval_lin1, slotVal_sc hy, vVal]; push_cast; simp only [Int.one_mul]; omega
      · exact blank
    all_goals exact blank
  case uextend64 a =>
    simp only [symOp]
    split
    · rename_i hc
      obtain ⟨ta, x, hx, hta, ha⟩ := slot_int h hc
      have hv' : ∃ z, v = .sc .i64 z ∧ z.toNat ≤ x.toNat := by
        simp only [evalOp, un, Sem.get, hx, Option.bind_eq_bind, Option.bind_some] at hv
        split at hv
        · cases hv; exact ⟨_, rfl, by rw [UInt64.toNat_and]; exact Nat.and_le_left⟩
        · cases hv
      obtain ⟨z, rfl, hzx⟩ := hv'
      refine fits_new h (fun r hr => ?_) (fun g hg => ?_) ⟨fun D hD => by simp at hD, fun p hp => ?_⟩
      · first
          | (rcases (by simpa [Sym.slots, lin1, Lin.slots] using hr : r = a ∨ r = b) with rfl | rfl <;> assumption)
          | (rw [(by simpa [Sym.slots, lin1, Lin.slots] using hr : r = a)]; exact ha)
      · simp [Sym.regions, lin1, Lin.regions] at hg
      · simp only [List.mem_singleton] at hp; subst hp
        simp only [eval_lin1, slotVal_sc hx, vVal]; push_cast; simp only [Int.one_mul]; omega
    · exact blank
  case iun k a =>
    simp only [symOp]
    split
    · rename_i hc
      obtain ⟨hk, hai⟩ := hc
      obtain ⟨x, hx, ha⟩ := slot_i64 h hai
      have hxv : slotVal Γ a = x.toNat := slotVal_sc hx
      have hv' : ∃ y, v = .sc .i64 y ∧ (vVal (.sc .i64 y) : Int) = iunVal k (slotVal Γ a) := by
        rcases hk with rfl | rfl
        · rw [AlgorithmLib.Prog.evalOp_clz64 hx] at hv
          cases hv
          exact ⟨_, rfl, by simp [vVal, iunVal, hxv, UInt64.ofNat_toNat]⟩
        · rw [AlgorithmLib.Prog.evalOp_bitrev64 hx] at hv
          cases hv
          exact ⟨_, rfl, by simp [vVal, iunVal, hxv, UInt64.ofNat_toNat]⟩
      obtain ⟨y, rfl, hy⟩ := hv'
      refine fits_new h (fun z hz => ?_) (fun g hg => ?_) ⟨fun D hD => ?_, fun p hp => by simp at hp⟩
      · have : z = a := by simpa [Sym.slots, Lin.atom, Lin.slots] using hz
        rw [this]; exact ha
      · simp [Sym.regions, Lin.atom, Lin.regions] at hg
      · simp only [Option.some.injEq] at hD; subst hD
        simp only [Lin.atom, Lin.eval, Lin.evalT, Atom.val, Int.one_mul, Int.add_zero]
        rw [hy]
    · exact blank
  all_goals exact blank

-- ---------------------------------------------------------------------------
-- What a comparison teaches
-- ---------------------------------------------------------------------------

/-- A slot replaced by a description known to hold of it too; one known
    exactly is kept. -/
def AS.setAV (A : AS) (r : R) (av : AV) : AS :=
  match A.get r with
  | .known _ => A
  | _ => if r < A.len then { A with slots := A.slots.set r av } else A

theorem AS.Holds.setAV {A : AS} {Γ : Env} {w : World} (hA : A.Holds vals Γ w) {r : R} {av : AV}
    (hv : ∀ v, Γ[r]? = some v → av.Desc Γ w v) : (A.setAV r av).Holds vals Γ w := by
  unfold AS.setAV
  split
  · exact hA
  · split
    · rename_i hr
      refine ⟨hA.1, fun i hi => ?_, hA.2.2⟩
      obtain ⟨u, hu, hd⟩ := hA.get hi
      refine ⟨u, hu, ?_⟩
      have e : AS.get { A with slots := A.slots.set r av } i = if i = r then av else A.get i := by
        unfold AS.get
        rw [if_pos hi, Slots.get_set]
        split
        · rfl
        · first | rfl | rw [if_pos hi]
      rw [e]
      split
      · rename_i h; subst h
        exact hv u hu
      · exact hd
    · exact hA

/-- A slot known to be at most `k / s`, where `s` times it is at most `k`. -/
def AS.tightenHi (A : AS) (r : R) (s : Nat) (k : Int) : AS :=
  if 0 ≤ k ∧ 0 < s then
    match A.get r with
    | .int n => A.setAV r (.int { n with hi := min n.hi (k.toNat / s) })
    | .sc t => if t.isInt then A.setAV r (.int ⟨t, 0, min (2 ^ 64 - 1) (k.toNat / s), 1, none⟩) else A
    | _ => A
  else A

theorem AS.Holds.tightenHi {A : AS} {Γ : Env} {w : World} (hA : A.Holds vals Γ w) {r : R} {s : Nat} {k : Int}
    (hv : (s : Int) * slotVal Γ r ≤ k) : (A.tightenHi r s k).Holds vals Γ w := by
  unfold AS.tightenHi
  split
  · rename_i hk
    split
    · rename_i n hn
      refine hA.setAV fun v hv' => ?_
      have hd := hA.descAt hv'
      rw [hn] at hd
      obtain ⟨x, rfl, h1, h2, h3, h4⟩ := hd
      refine ⟨x, rfl, h1, Nat.le_min.mpr ⟨h2, ?_⟩, h3, h4⟩
      rw [slotVal_sc hv'] at hv
      have : s * x.toNat ≤ k.toNat := by
        have := Int.toNat_of_nonneg hk.1
        omega
      exact (Nat.le_div_iff_mul_le hk.2).mpr (by rw [Nat.mul_comm]; exact this)
    · rename_i t ht
      split
      · rename_i hti
        refine hA.setAV fun v hv' => ?_
        have hd := hA.descAt hv'
        rw [ht] at hd
        have hl : t.lanes = none := by cases t <;> first | rfl | cases hti
        obtain ⟨x, rfl⟩ : ∃ x, v = .sc t x := by
          simp only [AV.Desc, TyV, hl] at hd; exact hd
        refine ⟨x, rfl, Nat.zero_le _, Nat.le_min.mpr ⟨by have := x.toNat_lt; omega, ?_⟩, Nat.one_dvd _, trivial⟩
        rw [slotVal_sc hv'] at hv
        have : s * x.toNat ≤ k.toNat := by
          have := Int.toNat_of_nonneg hk.1
          omega
        exact (Nat.le_div_iff_mul_le hk.2).mpr (by rw [Nat.mul_comm]; exact this)
      · exact hA
    · exact hA
  · exact hA


/-- At most this many bounds are kept for a slot, the latest first. -/
def ubsKept : Nat := 4

/-- A slot given one more bound. -/
def AS.addUb (A : AS) (a : R) (p : Nat × Lin) : AS :=
  if a < A.len then { A with syms := A.syms.set a { A.syms.get a with ubs := (p :: (A.syms.get a).ubs).take ubsKept } }
  else A

/-- Whether a bound may be kept: a positive multiple, reading slots of the
    state and sizes of regions other than the pinned one. -/
def ubOk (A : AS) (p : Nat × Lin) : Bool :=
  decide (0 < p.1) && p.2.slots.all (· < A.len) && p.2.regions.all (fun g => decide (g ≠ .pinned))

theorem AS.Holds.addUb {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {a : R} {p : Nat × Lin}
    (hv : (p.1 : Int) * slotVal Γ a ≤ p.2.eval Γ w) (hok : ubOk A p = true) : (A.addUb a p).Holds vals Γ w := by
  unfold AS.addUb
  split
  · rename_i ha
    simp only [ubOk, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at hok
    obtain ⟨⟨-, hs⟩, hg⟩ := hok
    refine ⟨⟨h.1.1, fun i hi => ?_, h.1.2.2⟩, h.2.1, h.2.2⟩
    show (Slots.set A.syms a _ |>.get i).Fits Γ w i A.len
    rw [Slots.get_set]
    split
    · rename_i e
      subst e
      obtain ⟨⟨hd, hu⟩, hsl, hrg⟩ := h.1.2.1 i hi
      have hmem : ∀ q ∈ (p :: (A.syms.get i).ubs).take ubsKept, q = p ∨ q ∈ (A.syms.get i).ubs := by
        intro q hq
        simpa using List.mem_of_mem_take hq
      refine ⟨⟨hd, fun q hq => ?_⟩, fun x hx => ?_, fun g hgr => ?_⟩
      · rcases hmem q hq with rfl | hq'
        · exact hv
        · exact hu q hq'
      · simp only [Sym.slots, List.mem_append, List.mem_flatten, List.mem_map] at hx
        rcases hx with hx | ⟨_, ⟨q, hq, rfl⟩, hx⟩
        · exact hsl x (by simp only [Sym.slots, List.mem_append]; exact .inl hx)
        · rcases hmem q hq with rfl | hq'
          · exact hs x hx
          · exact hsl x (by
              simp only [Sym.slots, List.mem_append, List.mem_flatten, List.mem_map]
              exact .inr ⟨_, ⟨q, hq', rfl⟩, hx⟩)
      · simp only [Sym.regions, List.mem_append, List.mem_flatten, List.mem_map] at hgr
        rcases hgr with hgr | ⟨_, ⟨q, hq, rfl⟩, hgr⟩
        · exact hrg g (by simp only [Sym.regions, List.mem_append]; exact .inl hgr)
        · rcases hmem q hq with rfl | hq'
          · exact hg g hgr
          · exact hrg g (by
              simp only [Sym.regions, List.mem_append, List.mem_flatten, List.mem_map]
              exact .inr ⟨_, ⟨q, hq', rfl⟩, hgr⟩)
    · exact h.1.2.1 i hi
  · exact h

/-- What `x + e ≤ y` teaches beyond ranges: `x` no more than `y - e`, and, with
    each exact definition put in, each slot it then reads with a positive
    coefficient no more than the rest of the form taken away; a slot bounded so
    by a number has its range cut to it. -/
def relLess (A : AS) (x y : R) (e : Nat) : AS :=
  if x < A.len ∧ y < A.len then
    let G : Lin := (A.ctx.expand proveFuel ⟨[(.slot x, 1), (.slot y, -1)], e⟩).norm
    let pos : List R := G.slots.filter fun r => decide (0 < Lin.coeff (.slot r) G.terms)
    let ubs : List (R × Nat × Lin) := (x, 1, ⟨[(.slot y, 1)], -(e : Int)⟩) ::
      pos.map fun r => (r, (Lin.coeff (.slot r) G.terms).toNat, (G.drop (.slot r)).scale (-1))
    let B := ubs.foldl (fun B (r, p) => if ubOk A p then B.addUb r p else B) A
    ubs.foldl (fun B (r, p) => match A.ctx.hiOf p.2 with
      | some k => B.tightenHi r p.1 k
      | none => B) B
  else A

theorem foldl_addUb {A : AS} {Γ : Env} {w : World} :
    ∀ (ubs : List (R × Nat × Lin)) (B : AS), B.Holds vals Γ w →
      (∀ q ∈ ubs, ubOk A q.2 = true → ((q.2.1 : Int) * slotVal Γ q.1 ≤ q.2.2.eval Γ w ∧ ubOk B q.2 = true)) →
      (∀ q ∈ ubs, ∀ C : AS, C.len = B.len → ubOk C q.2 = ubOk B q.2) →
      (ubs.foldl (fun B (r, p) => if ubOk A p then B.addUb r p else B) B).Holds vals Γ w
  | [], _, hB, _, _ => hB
  | (r, p) :: ubs, B, hB, hq, hlen => by
      rw [List.foldl_cons]
      have hB' : (if ubOk A p then B.addUb r p else B).Holds vals Γ w := by
        split
        · rename_i hok
          obtain ⟨hv, hokB⟩ := hq _ (List.mem_cons_self ..) hok
          exact hB.addUb hv hokB
        · exact hB
      have hl : (if ubOk A p then B.addUb r p else B).len = B.len := by
        split
        · unfold AS.addUb; split <;> rfl
        · rfl
      refine foldl_addUb ubs _ hB' (fun q hqm hok => ?_) (fun q hqm C hC => ?_)
      · obtain ⟨hv, hokB⟩ := hq q (List.mem_cons_of_mem _ hqm) hok
        exact ⟨hv, by rw [hlen q (List.mem_cons_of_mem _ hqm) _ hl]; exact hokB⟩
      · rw [hlen q (List.mem_cons_of_mem _ hqm) C (by rw [hC, hl]),
          hlen q (List.mem_cons_of_mem _ hqm) _ hl]

theorem ubOk_len {A C : AS} (h : C.len = A.len) (p : Nat × Lin) : ubOk C p = ubOk A p := by
  simp only [ubOk, h]

/-- The slot and coefficient of a form that is one slot's multiple. -/
def Lin.single? (L : Lin) : Option (R × Int) :=
  match L.terms with
  | [(.slot z, c)] => some (z, c)
  | _ => none

theorem Lin.single?_eq {L : Lin} {z : R} {c : Int} (h : L.single? = some (z, c)) :
    L.terms = [(.slot z, c)] := by
  unfold Lin.single? at h
  split at h
  · rename_i z' c' e; cases h; exact e
  · cases h

/-- An earlier shift of `a` right by `k`, where the facts show it: with
    `2^k · z ≤ a ≤ 2^k · z + 2^k - 1`, `z` is that shift. -/
def ushrTwin (A : AS) (a : R) (m : Nat) : Option R :=
  (A.sym a).ubs.findSome? fun p =>
    match p.2.single? with
    | some (z, c) =>
        if p.1 = 1 ∧ c = (m : Int) ∧ p.2.k = (m : Int) - 1 ∧ (m, lin1 a 1) ∈ (A.sym z).ubs then some z else none
    | none => none

theorem ushrTwin_spec {A : AS} {a z : R} {m : Nat} (h : ushrTwin A a m = some z) :
    ∃ L : Lin, (1, L) ∈ (A.sym a).ubs ∧ L.terms = [(.slot z, (m : Int))] ∧ L.k = (m : Int) - 1 ∧
      (m, lin1 a 1) ∈ (A.sym z).ubs := by
  unfold ushrTwin at h
  obtain ⟨⟨p1, L⟩, hp, hf⟩ := List.exists_of_findSome?_eq_some h
  dsimp only at hf
  split at hf
  · rename_i z' c hs
    split at hf
    · rename_i hc
      cases hf
      obtain ⟨rfl, rfl, hk, hm⟩ := hc
      exact ⟨L, hp, Lin.single?_eq hs, hk, hm⟩
    · cases hf
  · cases hf

/-- A form that is one operation of a slot, and the number added. -/
def iunOf (L : Lin) : Option (IUn × R × Int) :=
  match L.terms with
  | [(.iun k r, 1)] => some (k, r, L.k)
  | _ => none

theorem iunOf_eval {L : Lin} {k : IUn} {r : R} {c : Int} (h : iunOf L = some (k, r, c)) (Γ : Env) (w : World) :
    L.eval Γ w = iunVal k (slotVal Γ r) + c := by
  unfold iunOf at h
  split at h
  · rename_i k' r' e
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl, rfl⟩ := h
    simp [Lin.eval, e, Lin.evalT, Atom.val]
  · cases h

/-- What a shift of a bit-reversed index by one past the leading zeros of a
    bound above the index teaches: it is below the bound
    (`bitrev_index_lt`). -/
def bitrevFacts (A : AS) (a b : R) : List (R × Nat × Lin) :=
  match (A.sym a).dfn.bind iunOf, iunOf (A.ctx.foldKnown (A.ctx.deepen Ctx.defDepth (lin1 b 1))).norm with
  | some (.bitrev, j, 0), some (.clz, n, 1) =>
      if isI64 A a && isI64 A b && isI64 A j && isI64 A n &&
          A.ctx.prove proveFuel ⟨[(.slot j, 1), (.slot n, -1)], 1⟩ then
        [(A.len, 1, ⟨[(.slot n, 1)], -1⟩)]
      else []
  | _, _ => []

/-- What a shift right `x = a >> k` teaches beyond its own facts, `x` the slot
    it is pushed to: an earlier shift of `a` by `k` is the same number; where
    there is none, `a` is less than `2^k · (x + 1)`; and where `a` is at least
    one and `k` is, `x` is less than `a`. -/
def opFacts (A : AS) : Op → List (R × Nat × Lin)
  | .ushr a b =>
      match (A.get b).bits with
      | some k =>
          if isI64 A a && isIntSlot A b && k.toNat < 64 then
            let m := 2 ^ k.toNat
            let less : List (R × Nat × Lin) :=
              if 0 < k.toNat ∧ 1 ≤ (A.get a).rangeOf.1 then [(A.len, 1, ⟨[(.slot a, 1)], -1⟩)] else []
            match ushrTwin A a m with
            | some z => (A.len, 1, lin1 z 1) :: (z, 1, lin1 A.len 1) :: less
            | none => (a, 1, ⟨[(.slot A.len, (m : Int))], (m : Int) - 1⟩) :: less
          else []
      | none => bitrevFacts A a b
  | _ => []

theorem bitrevFacts_sound {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {a b : R} {v : V}
    (hv : evalOp w.mem Γ (.ushr a b) = some v) :
    ∀ q ∈ bitrevFacts A a b, (q.2.1 : Int) * slotVal (Γ.push v) q.1 ≤ q.2.2.eval (Γ.push v) w := by
  intro q hq
  unfold bitrevFacts at hq
  split at hq
  · rename_i j n hD hE
    split at hq
    · rename_i hc
      simp only [Bool.and_eq_true] at hc
      obtain ⟨⟨⟨⟨hai, hbi⟩, hji⟩, hni⟩, hp⟩ := hc
      simp only [List.mem_singleton] at hq
      subst hq
      obtain ⟨X, hx, ha⟩ := slot_i64 h hai
      obtain ⟨B, hb, -⟩ := slot_i64 h hbi
      obtain ⟨J, hj, -⟩ := slot_i64 h hji
      obtain ⟨N, hn, hnl⟩ := slot_i64 h hni
      have hC := AS.ctx_sound h
      -- `a` holds the bits of `j` reversed
      obtain ⟨D, hDd, hDi⟩ := Option.bind_eq_some_iff.mp hD
      have hXa := (h.symHolds a).1 D hDd
      rw [iunOf_eval hDi, slotVal_sc hx, slotVal_sc hj] at hXa
      have hX : X = AlgorithmLib.Prog.bitrev64 J := by
        apply UInt64.toNat_inj.mp
        simp only [iunVal, Int.toNat_natCast, UInt64.ofNat_toNat, Int.add_zero] at hXa
        have := X.toNat_lt; have := (AlgorithmLib.Prog.bitrev64 J).toNat_lt
        omega
      -- `b` is one more than the leading zeros of `n`
      have hBe := iunOf_eval hE Γ w
      rw [Lin.eval_norm, Ctx.foldKnown_eval hC] at hBe
      have hBm := Ctx.deepen_mod hC Ctx.defDepth (lin1 b 1)
      rw [hBe, eval_lin1, Int.one_mul, slotVal_sc hb, slotVal_sc hn] at hBm
      have hB : B = AlgorithmLib.Prog.clz64 N + 1 := by
        apply UInt64.toNat_inj.mp
        simp only [iunVal, Int.toNat_natCast, UInt64.ofNat_toNat] at hBm
        rw [UInt64.toNat_add]
        have := B.toNat_lt; have := (AlgorithmLib.Prog.clz64 N).toNat_lt
        simp only [UInt64.toNat_one]
        omega
      -- the shift
      have e : v = .sc .i64 (X >>> (B % UInt64.ofNat (ClifTy.width .i64))) := by
        simp [evalOp, shiftBin, Sem.get, hx, hb, norm, and_wide64, show ClifTy.isInt .i64 = true from rfl] at hv
        exact hv.symm
      subst e
      have hjn := Ctx.prove_sound hC hp
      simp only [Lin.eval, Lin.evalT, Atom.val, slotVal_sc hj, slotVal_sc hn] at hjn
      have hlt := AlgorithmLib.Prog.bitrev_index_lt J N 1 rfl
      have hw : (UInt64.ofNat (ClifTy.width .i64) : UInt64) = 64 := rfl
      rw [hX, hB, hw] at *
      have hsz := h.1.1
      generalize AlgorithmLib.Prog.bitrev64 J >>> ((AlgorithmLib.Prog.clz64 N + 1) % 64) = R at hlt ⊢
      have e1 : slotVal (Γ.push (.sc .i64 R)) A.len = R.toNat := by
        rw [slotVal_of (by rw [← hsz, Array.getElem?_push_size])]; rfl
      have e2 : slotVal (Γ.push (.sc .i64 R)) n = N.toNat := by
        rw [slotVal_push (hsz ▸ hnl), slotVal_sc hn]
      simp only [Lin.eval, Lin.evalT, Atom.val, e1, e2]
      omega
    · simp at hq
  · simp at hq

/-- **What `opFacts` teaches is true** of the operation's value pushed. -/
theorem opFacts_sound {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {o : Op} {v : V}
    (hv : evalOp w.mem Γ o = some v) :
    ∀ q ∈ opFacts A o, (q.2.1 : Int) * slotVal (Γ.push v) q.1 ≤ q.2.2.eval (Γ.push v) w := by
  intro q hq
  cases o
  case ushr a b =>
    simp only [opFacts] at hq
    split at hq
    · rename_i k hkb
      split at hq
      · rename_i hc
        simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
        obtain ⟨⟨hai, hbi⟩, hk⟩ := hc
        obtain ⟨x, hx, ha⟩ := slot_i64 h hai
        obtain ⟨tb, y, hy, htb, hb⟩ := slot_int h hbi
        obtain ⟨t, e⟩ := AV.bits_desc (h.descAt hy) hkb
        cases e
        have hmod : k % UInt64.ofNat (ClifTy.width .i64) = k := by
          apply UInt64.toNat_inj.mp
          rw [UInt64.toNat_mod]; simp only [ClifTy.width]; exact Nat.mod_eq_of_lt hk
        have e : v = .sc .i64 (x >>> k) := by
          simp [evalOp, shiftBin, Sem.get, hx, hy, norm, and_wide64, htb, show ClifTy.isInt .i64 = true from rfl] at hv
          rw [← hv, hmod]
        subst e
        have hsz := h.1.1
        have hnew : slotVal (Γ.push (.sc .i64 (x >>> k))) A.len = ((x.toNat / 2 ^ k.toNat : Nat) : Int) := by
          rw [slotVal_of (v := .sc .i64 (x >>> k)) (by rw [← hsz, Array.getElem?_push_size])]
          show (((x >>> k).toNat : Nat) : Int) = _
          rw [UInt64.toNat_shiftRight, Nat.mod_eq_of_lt hk, Nat.shiftRight_eq_div_pow]
        have hold : ∀ r, r < A.len → slotVal (Γ.push (.sc .i64 (x >>> k))) r = slotVal Γ r :=
          fun r hr => slotVal_push (hsz ▸ hr)
        have hxa : slotVal Γ a = x.toNat := slotVal_sc hx
        have hm0 : 0 < 2 ^ k.toNat := Nat.two_pow_pos _
        generalize hM : 2 ^ k.toNat = M at hm0 hnew hq
        have hless : ∀ q ∈ (if 0 < k.toNat ∧ 1 ≤ (A.get a).rangeOf.1 then
            ([(A.len, 1, ⟨[(.slot a, 1)], -1⟩)] : List (R × Nat × Lin)) else []),
            (q.2.1 : Int) * slotVal (Γ.push (.sc .i64 (x >>> k))) q.1 ≤ q.2.2.eval (Γ.push (.sc .i64 (x >>> k))) w := by
          intro q hq
          split at hq
          · rename_i hc
            simp only [List.mem_singleton] at hq
            subst hq
            have hlo := (AV.rangeOf_sound hx (h.descAt hx)).1
            rw [hxa] at hlo
            simp only [Lin.eval, Lin.evalT, Atom.val, hnew, hold a ha, hxa]
            have h2M : 2 ≤ M := by
              rw [← hM]
              have : 2 ^ 1 ≤ 2 ^ k.toNat := Nat.pow_le_pow_right (by decide) hc.1
              simpa using this
            have h1 : x.toNat / M ≤ x.toNat / 2 := Nat.div_le_div_left h2M (by decide)
            have h2 : x.toNat / 2 < x.toNat := Nat.div_lt_self (by omega) (by decide)
            omega
          · simp at hq
        split at hq
        · rename_i z hz
          obtain ⟨L, h1, hLt, hLk, h2⟩ := ushrTwin_spec hz
          have haL : a < A.len := ha
          have hzl : z < A.len := by
            have h1' : (1, L) ∈ (A.syms.get a).ubs := by simpa [AS.sym, haL] using h1
            exact (h.1.2.1 a haL).2.1 z (List.mem_append_right _ (List.mem_flatten.mpr
              ⟨_, List.mem_map.mpr ⟨_, h1', rfl⟩, by simp [Lin.slots, hLt]⟩))
          have hLe : L.eval Γ w = (M : Int) * slotVal Γ z + ((M : Int) - 1) := by
            simp only [Lin.eval, hLt, hLk, Lin.evalT, Atom.val]; omega
          have hza : (x.toNat : Int) ≤ (M : Int) * slotVal Γ z + ((M : Int) - 1) := by
            have := (h.symHolds a).2 _ h1
            rw [hLe, hxa] at this; simpa using this
          have haz : (M : Int) * slotVal Γ z ≤ x.toNat := by
            have := (h.symHolds z).2 _ h2
            rw [eval_lin1, hxa] at this; simpa using this
          have hzv : slotVal Γ z = ((x.toNat / M : Nat) : Int) := by
            obtain ⟨zn, ezn⟩ := Int.eq_ofNat_of_zero_le (slotVal_nonneg Γ z)
            rw [ezn] at hza haz ⊢
            have hlo : M * zn ≤ x.toNat := by exact_mod_cast haz
            have hhi : x.toNat ≤ M * zn + (M - 1) := by
              have : (x.toNat : Int) ≤ (M : Int) * (zn : Int) + ((M : Int) - 1) := hza
              omega
            have : x.toNat / M = zn :=
              Nat.div_eq_of_lt_le (by rw [Nat.mul_comm]; exact hlo) (by rw [Nat.add_mul, Nat.one_mul, Nat.mul_comm]; omega)
            rw [this]
          simp only [List.mem_cons] at hq
          rcases hq with rfl | rfl | hq
          · simp only [eval_lin1, hnew, hold z hzl, hzv]; omega
          · simp only [eval_lin1, hnew, hold z hzl, hzv]; omega
          · exact hless q hq
        · simp only [List.mem_cons] at hq
          rcases hq with rfl | hq
          · simp only [Lin.eval, Lin.evalT, Atom.val, hnew, hold a ha, hxa]
            have hN := Nat.lt_div_mul_add (a := x.toNat) hm0
            have key : (x.toNat : Int) < (M : Int) * ((x.toNat / M : Nat) : Int) + (M : Int) := by
              rw [Int.mul_comm]; exact_mod_cast hN
            omega
          · exact hless q hq
      · simp at hq
    · exact bitrevFacts_sound h hv q hq
  all_goals simp [opFacts] at hq

theorem foldl_tighten {A : AS} {Γ : Env} {w : World} (hA : A.Holds vals Γ w) :
    ∀ (ubs : List (R × Nat × Lin)) (B : AS), B.Holds vals Γ w →
      (∀ q ∈ ubs, (q.2.1 : Int) * slotVal Γ q.1 ≤ q.2.2.eval Γ w) →
      (ubs.foldl (fun B (r, p) => match A.ctx.hiOf p.2 with
        | some k => B.tightenHi r p.1 k
        | none => B) B).Holds vals Γ w
  | [], _, hB, _ => hB
  | (r, p) :: ubs, B, hB, hq => by
      rw [List.foldl_cons]
      have hB' : (match A.ctx.hiOf p.2 with
          | some k => B.tightenHi r p.1 k
          | none => B).Holds vals Γ w := by
        split
        · rename_i k hk
          have h := hq _ (List.mem_cons_self ..)
          exact hB.tightenHi (Int.le_trans h (Ctx.hiOf_sound (AS.ctx_sound hA) hk))
        · exact hB
      exact foldl_tighten hA ubs _ hB' (fun q hqm => hq q (List.mem_cons_of_mem _ hqm))

/-- **What `relLess` teaches is true** where `x + e ≤ y`. -/
theorem relLess_holds {A : AS} {Γ : Env} {w : World} (hA : A.Holds vals Γ w) {x y : R} {e : Nat}
    (hxy : slotVal Γ x + e ≤ slotVal Γ y) : (relLess A x y e).Holds vals Γ w := by
  unfold relLess
  split
  · rename_i hxy'
    have hC := AS.ctx_sound hA
    generalize hG : (A.ctx.expand proveFuel ⟨[(.slot x, 1), (.slot y, -1)], e⟩).norm = G
    have hGe : G.eval Γ w ≤ 0 := by
      rw [← hG, Lin.eval_norm, Ctx.expand_eval hC]
      simp only [Lin.eval, Lin.evalT, Atom.val]
      omega
    have hall : ∀ q ∈ ((x, 1, (⟨[(.slot y, 1)], -(e : Int)⟩ : Lin)) ::
        ((G.slots.filter fun r => decide (0 < Lin.coeff (.slot r) G.terms)).map fun r =>
          (r, (Lin.coeff (.slot r) G.terms).toNat, (G.drop (.slot r)).scale (-1)))),
        (q.2.1 : Int) * slotVal Γ q.1 ≤ q.2.2.eval Γ w := by
      intro q hqm
      simp only [List.mem_cons, List.mem_map, List.mem_filter, decide_eq_true_eq] at hqm
      rcases hqm with rfl | ⟨r, ⟨-, hc⟩, rfl⟩
      · simp only [Lin.eval, Lin.evalT, Atom.val]; omega
      · dsimp only
        have hsplit := Lin.eval_split Γ w (.slot r) G
        simp only [Atom.val] at hsplit
        rw [Lin.eval_scale]
        have hc' : ((Lin.coeff (.slot r) G.terms).toNat : Int) = Lin.coeff (.slot r) G.terms :=
          Int.toNat_of_nonneg (by omega)
        rw [hc']
        omega
    exact foldl_tighten hA _ _ (foldl_addUb _ _ hA (fun q hqm hok => ⟨hall q hqm, hok⟩)
      (fun q _ C hC => ubOk_len hC _)) hall
  · exact hA

/-- A slot known to be at least `⌈k / s⌉`, where `s` times it is at least `k`. -/
def AS.tightenLo (A : AS) (r : R) (s : Nat) (k : Int) : AS :=
  if 0 < k ∧ 0 < s then
    match A.get r with
    | .int n => A.setAV r (.int { n with lo := max n.lo ((k.toNat + s - 1) / s) })
    | .sc t => if t.isInt then A.setAV r (.int ⟨t, (k.toNat + s - 1) / s, 2 ^ 64 - 1, 1, none⟩) else A
    | _ => A
  else A

theorem ceil_le {k s x : Nat} (hs : 0 < s) (h : k ≤ s * x) : (k + s - 1) / s ≤ x :=
  Nat.lt_succ_iff.mp ((Nat.div_lt_iff_lt_mul hs).mpr (by rw [Nat.succ_mul, Nat.mul_comm x s]; omega))

theorem AS.Holds.tightenLo {A : AS} {Γ : Env} {w : World} (hA : A.Holds vals Γ w) {r : R} {s : Nat} {k : Int}
    (hv : k ≤ (s : Int) * slotVal Γ r) : (A.tightenLo r s k).Holds vals Γ w := by
  unfold AS.tightenLo
  split
  · rename_i hk
    have hk' : k.toNat ≤ s * (slotVal Γ r).toNat := by
      have := slotVal_nonneg Γ r
      have e1 := Int.toNat_of_nonneg (Int.le_of_lt hk.1)
      have e2 := Int.toNat_of_nonneg this
      have : (k.toNat : Int) ≤ (s : Int) * ((slotVal Γ r).toNat : Int) := by rw [e1, e2]; exact hv
      exact_mod_cast this
    split
    · rename_i n hn
      refine hA.setAV fun v hv' => ?_
      have hd := hA.descAt hv'
      rw [hn] at hd
      obtain ⟨x, rfl, h1, h2, h3, h4⟩ := hd
      rw [slotVal_sc hv'] at hk'
      simp only [Int.toNat_natCast] at hk'
      exact ⟨x, rfl, Nat.max_le.mpr ⟨h1, ceil_le hk.2 hk'⟩, h2, h3, h4⟩
    · rename_i t ht
      split
      · rename_i hti
        refine hA.setAV fun v hv' => ?_
        have hd := hA.descAt hv'
        rw [ht] at hd
        have hl : t.lanes = none := by cases t <;> first | rfl | cases hti
        obtain ⟨x, rfl⟩ : ∃ x, v = .sc t x := by
          simp only [AV.Desc, TyV, hl] at hd; exact hd
        rw [slotVal_sc hv'] at hk'
        simp only [Int.toNat_natCast] at hk'
        exact ⟨x, rfl, ceil_le hk.2 hk', by dsimp only; have := x.toNat_lt; omega, Nat.one_dvd _, trivial⟩
      · exact hA
    · exact hA
  · exact hA

/-- What a slot's exact definition teaches: it is no less than zero, so each
    slot it reads with a positive coefficient is no less than the rest taken
    away. -/
def relLo (A : AS) (x : R) : AS :=
  match A.ctx.exactDef x with
  | some D =>
      let D := D.norm
      (D.slots.filter fun r => decide (0 < Lin.coeff (.slot r) D.terms)).foldl (fun B r =>
        match A.ctx.hiOf (D.drop (.slot r)) with
        | some h => B.tightenLo r (Lin.coeff (.slot r) D.terms).toNat (-h)
        | none => B) A
  | none => A

theorem relLo_holds {A : AS} {Γ : Env} {w : World} (hA : A.Holds vals Γ w) (x : R) : (relLo A x).Holds vals Γ w := by
  unfold relLo
  split
  · rename_i D hD
    have hC := AS.ctx_sound hA
    have hx := Ctx.exactDef_sound hC hD
    have hD0 : 0 ≤ D.norm.eval Γ w := by rw [Lin.eval_norm, ← hx]; exact slotVal_nonneg Γ x
    generalize D.norm = E at hD0 ⊢
    suffices h : ∀ (rs : List R) (B : AS), B.Holds vals Γ w → (∀ r ∈ rs, 0 < Lin.coeff (.slot r) E.terms) →
        (rs.foldl (fun B r => match A.ctx.hiOf (E.drop (.slot r)) with
          | some h => B.tightenLo r (Lin.coeff (.slot r) E.terms).toNat (-h)
          | none => B) B).Holds vals Γ w from
      h _ A hA (fun r hr => by simpa using (List.mem_filter.mp hr).2)
    intro rs
    induction rs with
    | nil => intro B hB _; exact hB
    | cons r rs ih =>
        intro B hB hpos
        rw [List.foldl_cons]
        refine ih _ ?_ (fun q hq => hpos q (List.mem_cons_of_mem _ hq))
        split
        · rename_i h hh
          refine hB.tightenLo ?_
          have hs := Lin.eval_split Γ w (.slot r) E
          simp only [Atom.val] at hs
          have hb := Ctx.hiOf_sound hC hh
          have hc := hpos r (List.mem_cons_self ..)
          rw [Int.toNat_of_nonneg (by omega)]
          omega
        · exact hB
  · exact hA

-- ---------------------------------------------------------------------------
-- A call's bytes, by relations
-- ---------------------------------------------------------------------------

/-- Whether the bytes from the argument at `i`, as many as the argument at `j`
    says, lie inside a region, by the prover. -/
def relRb (A : AS) (args : List R) (i j : Nat) : Bool :=
  decide (i < args.length) && decide (j < args.length) && memRel2 A (args.getD i 0) (args.getD j 0)

theorem slot_bits {Γ : Env} : ∀ {args : List R} {vs : List V} {bits : List UInt64},
    args.mapM (fun r => Γ[r]?) = some vs → vs.mapM asBits = some bits →
    ∀ i, i < args.length → ∃ t, Γ[args.getD i 0]? = some (.sc t (bits.getD i 0))
  | [], _, _, _, _, i, hi => by simp at hi
  | r :: rs, vs, bits, hv, hb, i, hi => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at hv
      obtain ⟨v, hrv, vs', hvs', rfl⟩ := hv
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at hb
      obtain ⟨x, hx, xs, hxs, rfl⟩ := hb
      cases i with
      | zero =>
          cases v with
          | sc t y => simp only [asBits, Option.some.injEq] at hx; subst hx; exact ⟨t, by simpa using hrv⟩
          | vec _ _ => cases hx
      | succ i =>
          simp only [List.length_cons] at hi
          obtain ⟨t, ht⟩ := slot_bits hvs' hxs i (by omega)
          exact ⟨t, by simpa using ht⟩

theorem relRb_sound {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {args : List R} {vs : List V}
    {bits : List UInt64} (hv : args.mapM (fun r => Γ[r]?) = some vs) (hb : vs.mapM asBits = some bits) :
    RbSound w bits (relRb A args) := by
  intro i j hr
  simp only [relRb, Bool.and_eq_true, decide_eq_true_eq] at hr
  obtain ⟨⟨hi, hj⟩, hm⟩ := hr
  obtain ⟨t, ht⟩ := slot_bits hv hb i hi
  obtain ⟨t', ht'⟩ := slot_bits hv hb j hj
  exact memRel2_fits h hm ht ht'

end AlgorithmLib.Check
