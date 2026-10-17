module
public import AlgorithmLib.Proof.Typestate
import all Init.Data.Repr
@[expose] public section

/-!
# `Check.Absint` — a body's safety, checked by a function proven sound once

`chk` runs a body over *abstract* state: each slot is a value it knows, a
scalar of a type it knows, or anything, and the world is described by a
typestate (`TState`). It answers `true` only when every step it takes is safe
in every state its abstract state describes. **`chk_sound`**: then the body's
weakest precondition holds there. So a body's safety is `chk … = true`, which
the kernel evaluates: no proof term is built per body, and a new shape of body
costs nothing.

What a foreign call needs is decided by `preB`, one case per entry point, each
proven against the contract table's precondition (`preB_sound`). What a call
leaves is the contract table's own: `after` where every argument is known, and
otherwise the facts `keeps` keeps whatever the unknown arguments are.
-/

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare
  AlgorithmLib.HProg.DevSpec AlgorithmLib.Prog

namespace AlgorithmLib.Check

open AlgorithmLib.HProg.Contracts (Fact TState after after_sound keeps keeps_sound roomAt holds_of_mem)

/-- What the checker knows of a slot. -/
inductive AV where
  | known (v : V)
  /-- A scalar of this type, bits unknown. -/
  | sc (t : ClifTy)
  | any
  deriving Repr, Inhabited

/-- The values an abstract value describes. -/
def AV.Desc : AV → V → Prop
  | .known v', v => v = v'
  | .sc t, v => ∃ x, v = .sc t x
  | .any, _ => True

/-- The scalar type a slot is known to have. -/
def AV.scTy : AV → Option ClifTy
  | .known (.sc t _) => some t
  | .sc t => some t
  | _ => none

/-- The bits of a slot known to be a scalar of known bits. -/
def AV.bits : AV → Option UInt64
  | .known (.sc _ x) => some x
  | _ => none

def AV.isKnown : AV → Bool
  | .known _ => true
  | _ => false

theorem AV.asBits_of_scTy {av : AV} {v : V} {t : ClifTy} (hd : av.Desc v) (h : av.scTy = some t) :
    ∃ x, v = .sc t x := by
  cases av with
  | known u =>
      simp only [Desc] at hd; rw [hd]
      cases u with
      | sc t' x => simp only [scTy, Option.some.injEq] at h; subst h; exact ⟨x, rfl⟩
      | vec _ _ => cases h
  | sc t' => simp only [scTy, Option.some.injEq] at h; subst h; exact hd
  | any => cases h

theorem AV.asBits_of_bits {av : AV} {v : V} {x : UInt64} (hd : av.Desc v) (h : av.bits = some x) :
    asBits v = some x := by
  cases av with
  | known u =>
      simp only [Desc] at hd; rw [hd]
      cases u with
      | sc t' y => simp only [bits, Option.some.injEq] at h; subst h; rfl
      | vec _ _ => cases h
  | sc _ | any => cases h

/-- The checker's state: what it knows of each slot, and of the world. -/
structure AS where
  env : Array AV
  facts : TState

def AS.get (A : AS) (r : R) : AV := A.env.getD r .any

/-- The environments and worlds an abstract state describes. -/
def AS.Holds (A : AS) (Γ : Env) (w : World) : Prop :=
  Γ.size = A.env.size ∧ (∀ i, i < A.env.size → ∃ v, Γ[i]? = some v ∧ (A.get i).Desc v) ∧
    A.facts.holds w

theorem AS.get_eq {A : AS} {r : R} : A.get r = A.env[r]?.getD .any := by
  unfold AS.get; rw [Array.getD_eq_getD_getElem?]

theorem AS.get_out {A : AS} {r : R} (hr : ¬ r < A.env.size) : A.get r = .any := by
  rw [AS.get_eq, Array.getElem?_eq_none (Nat.le_of_not_lt hr)]
  rfl

theorem AS.get_lt_known {A : AS} {r : R} (hk : (A.get r).isKnown = true) : r < A.env.size := by
  rcases Nat.lt_or_ge r A.env.size with h | h
  · exact h
  · rw [AS.get_out (Nat.not_lt_of_ge h)] at hk; cases hk

theorem AS.get_lt_scTy {A : AS} {r : R} {t : ClifTy} (h : (A.get r).scTy = some t) : r < A.env.size := by
  rcases Nat.lt_or_ge r A.env.size with hl | hl
  · exact hl
  · rw [AS.get_out (Nat.not_lt_of_ge hl)] at h; cases h

theorem AS.Holds.get {A : AS} {Γ : Env} {w : World} (h : A.Holds Γ w) {r : R} (hr : r < A.env.size) :
    ∃ v, Γ[r]? = some v ∧ (A.get r).Desc v := h.2.1 r hr

/-- Pushing a slot the checker describes keeps the description. -/
theorem AS.Holds.push {A : AS} {Γ : Env} {w : World} (h : A.Holds Γ w) {av : AV} {v : V}
    (hv : av.Desc v) : AS.Holds ⟨A.env.push av, A.facts⟩ (Γ.push v) w := by
  obtain ⟨hs, he, hf⟩ := h
  refine ⟨by simp [hs], fun i hi => ?_, hf⟩
  simp only [Array.size_push] at hi
  by_cases hlt : i < A.env.size
  · obtain ⟨u, hu, hd⟩ := he i hlt
    refine ⟨u, ?_, ?_⟩
    · rw [Array.getElem?_push, if_neg (by omega)]; exact hu
    · rw [AS.get_eq] at hd ⊢
      simp only [Array.getElem?_push, if_neg (show i ≠ A.env.size by omega)]
      exact hd
  · have hi' : i = A.env.size := by omega
    subst hi'
    refine ⟨v, by rw [Array.getElem?_push, if_pos (by omega)], ?_⟩
    rw [AS.get_eq]
    simp only [Array.getElem?_push, if_pos, Option.getD_some]
    exact hv

/-- The same environment, a world the new facts describe. -/
theorem AS.Holds.facts {A : AS} {Γ : Env} {w w' : World} (h : A.Holds Γ w) {S : TState}
    (hS : S.holds w') : AS.Holds ⟨A.env, S⟩ Γ w' := ⟨h.1, h.2.1, hS⟩

-- ---------------------------------------------------------------------------
-- Operations
-- ---------------------------------------------------------------------------

/-- A known slot's value; any value for the rest. -/
def concV : AV → V
  | .known v => v
  | _ => .sc .i64 0

/-- A concrete environment agreeing with every one the state describes on the
    slots it knows. Built from a list: `Array.map` is well-founded recursion,
    which the kernel does not unfold. -/
def concEnv (A : AS) : Env := ⟨A.env.toList.map concV⟩

theorem concEnv_eq (A : AS) : concEnv A = A.env.map concV := by
  apply Array.toList_inj.mp; simp [concEnv]

theorem concEnv_get {A : AS} {Γ : Env} {w : World} (h : A.Holds Γ w) {r : R}
    (hk : (A.get r).isKnown = true) : (concEnv A)[r]? = Γ[r]? := by
  have hr := AS.get_lt_known hk
  obtain ⟨u, hu, hd⟩ := h.get hr
  rw [hu, concEnv_eq]
  rw [Array.getElem?_map, Array.getElem?_eq_getElem hr, Option.map_some]
  have e : A.get r = A.env[r] := by rw [AS.get_eq, Array.getElem?_eq_getElem hr]; rfl
  rw [e] at hd hk
  generalize A.env[r] = x at hd hk ⊢
  cases x with
  | known v => simp only [AV.Desc] at hd; rw [hd]; rfl
  | sc _ => cases hk
  | any => cases hk

/-- The value eight bytes at `a` hold, where the typestate names it. -/
def cellAt (S : TState) (a : UInt64) : Option UInt64 :=
  S.findSome? fun | .cell a' v => if a' == a then some v else none | _ => none

theorem cellAt_mem {S : TState} {a v : UInt64} (h : cellAt S a = some v) : Fact.cell a v ∈ S := by
  unfold cellAt at h
  obtain ⟨x, hx, hf⟩ := List.exists_of_findSome?_eq_some h
  cases x <;> simp only [reduceCtorEq] at hf
  rename_i a' v'
  split at hf
  · rename_i he; simp only [beq_iff_eq] at he; subst he; cases hf; exact hx
  · cases hf

/-- A value of type `t` read at `x`, where the typestate gives the room. -/
def absRoom (S : TState) (t : ClifTy) (x : UInt64) : Option AV :=
  if roomAt S x (tyBytes t) then some (.sc t) else none

/-- What a load leaves, from a known address: a cell the typestate names, or a
    value of the load's type where it gives the room. -/
def absLoad (S : TState) (op : LoadOp) (x : UInt64) : Option AV :=
  match op.kind, op.ty.lanes with
  | .plain, none =>
      match cellAt S x with
      | some v => if op.ty = .i64 then some (.known (.sc .i64 v)) else absRoom S op.ty x
      | none => absRoom S op.ty x
  | _, _ => none

/-- What an operation leaves in its slot: computed where its operands are
    known, typed where their types are, and for a load, what `absLoad` says.
    `none` where the checker cannot show the operation is defined. -/
def absOp (A : AS) (o : Op) : Option AV :=
  match o with
  | .load op a =>
      match A.get a with
      | .known (.sc _ x) => absLoad A.facts op x
      | _ => none
  | o =>
      if o.regs.all (fun r => (A.get r).isKnown) then (evalOp default (concEnv A) o).map .known
      else (o.regs.mapM (fun r => (A.get r).scTy) >>= tyOp o).map .sc

/-- An operation other than a load reads its operands and nothing else. -/
theorem evalOp_congr (m₁ m₂ : Mem) (Γ₁ Γ₂ : Env) (o : Op) (hl : ∀ op p, o ≠ .load op p)
    (h : ∀ r ∈ o.regs, Γ₁[r]? = Γ₂[r]?) : evalOp m₁ Γ₁ o = evalOp m₂ Γ₂ o := by
  cases o with
  | load op p => exact absurd rfl (hl op p)
  | iconst | fconst => rfl
  | iadd a b | isub a b | imul a b | udiv a b | ishl a b | ushr a b | band a b | bandNot a b
  | bor a b | bxor a b | icmp _ a b | fadd a b | fsub a b | fmul a b | fmax a b | fmin a b
  | fcmp _ a b | ibin _ a b | ishift _ a b | fbin _ a b =>
    have ha := h a (by simp [Op.regs]); have hb := h b (by simp [Op.regs])
    simp only [evalOp, bin, shiftBin, Sem.get, ha, hb]
  | ineg a | ctz a | popcnt a | ireduce32 a | uextend64 a | sextend64 a | fneg a | fpromote a
  | vhighBits a | fcvtFromSint _ a | fcvtToUint _ a | splat _ a | extractlane a _ | bitcast _ a
  | iun _ a | fun1 _ a | fconv _ _ a | iext _ _ a =>
    have ha := h a (by simp [Op.regs])
    simp only [evalOp, un, Sem.get, ha]
  | select c a b | bitselect c a b | fma c a b =>
    have hc := h c (by simp [Op.regs]); have ha := h a (by simp [Op.regs])
    have hb := h b (by simp [Op.regs])
    simp only [evalOp, Sem.get, hc, ha, hb]

theorem argsTy_of {A : AS} {Γ : Env} {w : World} (h : A.Holds Γ w) :
    ∀ {rs : List R} {ts : List ClifTy}, rs.mapM (fun r => (A.get r).scTy) = some ts → ArgsTy Γ rs ts
  | [], ts, e => by simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at e; subst e; trivial
  | r :: rs, ts, e => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at e
      obtain ⟨t, ht, ts', hts, rfl⟩ := e
      obtain ⟨v, hv, hd⟩ := h.get (AS.get_lt_scTy ht)
      obtain ⟨x, rfl⟩ := AV.asBits_of_scTy hd ht
      exact ⟨⟨x, hv⟩, argsTy_of h hts⟩

theorem absLoad_sound {A : AS} {Γ : Env} {w : World} (h : A.Holds Γ w) {op : LoadOp} {a : R}
    {t : ClifTy} {x : UInt64} (ha : Γ[a]? = some (.sc t x)) {av : AV} (hl : absLoad A.facts op x = some av) :
    ∃ v, evalOp w.mem Γ (.load op a) = some v ∧ av.Desc v := by
  have room : ∀ (hk : op.kind = .plain) (hl0 : op.ty.lanes = none),
      absRoom A.facts op.ty x = some av → ∃ v, evalOp w.mem Γ (.load op a) = some v ∧ av.Desc v := by
    intro hk hl0 hr
    unfold absRoom at hr
    split at hr
    · rename_i hroom
      cases hr
      obtain ⟨v, hv⟩ := Option.isSome_iff_exists.mp (Contracts.load_of_state h.2.2 hroom)
      exact ⟨_, evalOp_load hk hl0 ha hv, v, rfl⟩
    · cases hr
  unfold absLoad at hl
  split at hl
  · rename_i hk hl0
    split at hl
    · rename_i v hcell
      split at hl
      · rename_i hty
        cases hl
        have hload : w.mem.load x (tyBytes op.ty) = some v := by
          rw [hty]; exact h.2.2 _ (cellAt_mem hcell)
        refine ⟨_, evalOp_load hk hl0 ha hload, ?_⟩
        simp only [AV.Desc, hty]
      · exact room hk hl0 hl
    · exact room hk hl0 hl
  · cases hl

/-- **What `absOp` says of an operation is true of it**, in every state the
    abstract one describes. -/
theorem absOp_sound {A : AS} {Γ : Env} {w : World} (h : A.Holds Γ w) {o : Op} {av : AV}
    (ho : absOp A o = some av) : ∃ v, evalOp w.mem Γ o = some v ∧ av.Desc v := by
  unfold absOp at ho
  split at ho
  · rename_i op a
    split at ho
    · rename_i t x hk
      have hr : a < A.env.size := AS.get_lt_known (by rw [hk]; rfl)
      obtain ⟨u, hu, hd⟩ := h.get hr
      rw [hk] at hd; simp only [AV.Desc] at hd; subst hd
      exact absLoad_sound h hu ho
    · cases ho
  · rename_i hnl
    split at ho
    · rename_i hall
      obtain ⟨v, hv, rfl⟩ := Option.map_eq_some_iff.mp ho
      refine ⟨v, ?_, rfl⟩
      rw [evalOp_congr w.mem default Γ (concEnv A) o (fun op p e => hnl op p e)
        (fun r hr => (concEnv_get h (List.all_eq_true.mp hall r hr)).symm)]
      exact hv
    · obtain ⟨T, hT, rfl⟩ := Option.map_eq_some_iff.mp ho
      simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at hT
      obtain ⟨ts, hts, hT⟩ := hT
      obtain ⟨x, hx⟩ := tyOp_sound (m := w.mem) (argsTy_of h hts) hT
      exact ⟨_, hx, x, rfl⟩

-- ---------------------------------------------------------------------------
-- Calls
-- ---------------------------------------------------------------------------

/-- Each abstract value describes the value in its place. -/
def DescAll : List AV → List V → Prop
  | [], [] => True
  | a :: as, v :: vs => a.Desc v ∧ DescAll as vs
  | _, _ => False

/-- The bits of every argument, where all are known. -/
def allBits : List AV → Option (List UInt64)
  | [] => some []
  | a :: as => do let b ← a.bits; let bs ← allBits as; pure (b :: bs)

/-- The argument bits of values the abstract arguments describe: every one a
    scalar, and the known ones the bits the checker knows. -/
theorem bits_of_desc : ∀ {as : List AV} {vs : List V}, DescAll as vs →
    as.all (fun a => a.scTy.isSome) = true →
    ∃ bits, vs.mapM asBits = some bits ∧ ∀ bs, allBits as = some bs → bits = bs
  | [], [], _, _ => ⟨[], rfl, fun bs h => by simp only [allBits, Option.some.injEq] at h; exact h⟩
  | [], _ :: _, h, _ => h.elim
  | _ :: _, [], h, _ => h.elim
  | a :: as, v :: vs, ⟨hd, hds⟩, hs => by
      simp only [List.all_cons, Bool.and_eq_true] at hs
      obtain ⟨t, ht⟩ := Option.isSome_iff_exists.mp hs.1
      obtain ⟨x, rfl⟩ := AV.asBits_of_scTy hd ht
      obtain ⟨bits, hb, he⟩ := bits_of_desc hds hs.2
      refine ⟨x :: bits, by simp [List.mapM_cons, asBits, hb], fun bs h => ?_⟩
      simp only [allBits, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h
      obtain ⟨b, hb', bs', hbs', rfl⟩ := h
      have := AV.asBits_of_bits hd hb'
      simp only [asBits, Option.some.injEq] at this
      rw [this, he bs' hbs']

/-- Facts whose keeping does not read a call's arguments. -/
def bitFree : Fact → Bool
  | .part .. | .room .. | .devSeq | .oracles | .roomArg .. | .devBuf .. | .pinnedUsed .. => true
  | _ => false

theorem keeps_bitFree {f : Ffi} {x : Fact} (hx : bitFree x = true) (b b' : List UInt64) :
    keeps f b x = keeps f b' x := by
  cases x <;> first | rfl | cases hx

/-- What a call leaves of a typestate: the table's `after` where every argument
    is known, and otherwise what it keeps whatever the unknown ones are. -/
def postA (f : Ffi) (as : List AV) (S : TState) : TState :=
  match allBits as with
  | some bits => after f bits S
  | none => S.filter (fun x => bitFree x && keeps f [] x)

theorem postA_sound {f : Ffi} {as : List AV} {S : TState} {bits : List UInt64} {w : World}
    {r : Option V} {w' : World} (hb : ∀ bs, allBits as = some bs → bits = bs) (hS : S.holds w)
    (h : callBits f bits w = some (r, w')) : (postA f as S).holds w' := by
  unfold postA
  split
  · rename_i bs hbs; rw [← hb bs hbs]; exact after_sound hS h
  · intro x hx
    obtain ⟨hx, hk⟩ := List.mem_filter.mp hx
    simp only [Bool.and_eq_true] at hk
    exact keeps_sound h (by rw [keeps_bitFree hk.1 bits []]; exact hk.2) (hS x hx)

/-- What a CUDA copy needs: the live context, the host bytes it reads or
    writes, and the default stream ready for the buffer. -/
def preCopy (c d s : AV) (S : TState) : Bool :=
  match c.bits, d.bits, s.bits with
  | some ctx, some dst, some size =>
      ctx == cudaCtx && S.contains (.part .cuda true) && roomAt S dst size.toNat && S.contains .devSeq
  | _, _, _ => false

/-- **What a call needs, decided from the typestate.** One case per entry
    point; an entry point without one is refused. -/
def preB (f : Ffi) (as : List AV) (S : TState) : Bool :=
  if f = .cudaDownload ∨ f = .cudaUpload then
    match as with
    | [c, _, d, s] => preCopy c d s S
    | _ => false
  else false

theorem preCopy_spec {c d s : AV} {S : TState} (h : preCopy c d s S = true) :
    ∃ dst size, c.bits = some cudaCtx ∧ d.bits = some dst ∧ s.bits = some size ∧
      S.contains (.part .cuda true) = true ∧ roomAt S dst size.toNat = true ∧ S.contains .devSeq = true := by
  unfold preCopy at h
  split at h
  · rename_i ctx dst size hc hd hs
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨⟨rfl, h1⟩, h2⟩, h3⟩ := h
    exact ⟨dst, size, hc, hd, hs, h1, h2, h3⟩
  · cases h

theorem descAll_cons {a : AV} {as : List AV} {vs : List V} (h : DescAll (a :: as) vs) :
    ∃ v vs', vs = v :: vs' ∧ a.Desc v ∧ DescAll as vs' := by
  cases vs with
  | nil => exact h.elim
  | cons v vs' => exact ⟨v, vs', rfl, h⟩

theorem descAll_nil {vs : List V} (h : DescAll [] vs) : vs = [] := by
  cases vs with
  | nil => rfl
  | cons _ _ => exact h.elim

theorem descAll4 {c b d s : AV} {vs : List V} (h : DescAll [c, b, d, s] vs) :
    ∃ vc vb vd vs', vs = [vc, vb, vd, vs'] ∧ c.Desc vc ∧ d.Desc vd ∧ s.Desc vs' := by
  obtain ⟨vc, r1, rfl, hc, h⟩ := descAll_cons h
  obtain ⟨vb, r2, rfl, -, h⟩ := descAll_cons h
  obtain ⟨vd, r3, rfl, hd, h⟩ := descAll_cons h
  obtain ⟨vs', r4, rfl, hs, h⟩ := descAll_cons h
  rw [descAll_nil h]
  exact ⟨vc, vb, vd, vs', rfl, hc, hd, hs⟩

/-- The bits of four arguments whose first, third and fourth are known. -/
theorem bits4 {c b d s : AV} {vs : List V} {bits : List UInt64} {x y z : UInt64}
    (hd : DescAll [c, b, d, s] vs) (hb : vs.mapM asBits = some bits)
    (hc : c.bits = some x) (hdd : d.bits = some y) (hs : s.bits = some z) :
    ∃ xb, bits = [x, xb, y, z] := by
  obtain ⟨vc, vb, vd, vs', rfl, hc', hd', hs'⟩ := descAll4 hd
  have e1 := AV.asBits_of_bits hc' hc
  have e3 := AV.asBits_of_bits hd' hdd
  have e4 := AV.asBits_of_bits hs' hs
  simp only [List.mapM_cons, List.mapM_nil, e1, e3, e4, Option.bind_eq_bind, Option.bind_some,
    Option.pure_def] at hb
  revert hb; cases asBits vb with
  | none => intro hb; cases hb
  | some xb => intro hb; simp only [Option.bind_some, Option.some.injEq] at hb; exact ⟨xb, hb.symm⟩

theorem pre_cudaDownload {S : TState} {w : World} {xb dst size : UInt64} (hS : S.holds w)
    (hcu : S.contains (.part .cuda true) = true) (hroom : roomAt S dst size.toNat = true)
    (hseq : S.contains .devSeq = true) : Contracts.Pre .cudaDownload [cudaCtx, xb, dst, size] w := by
  simp only [Contracts.Pre]
  exact ⟨Contracts.ctxOk_of_state hS hcu, fun _ _ _ =>
    ⟨Contracts.writable_of_state hS hroom, Contracts.ready_of_state hS hseq _ _⟩⟩

theorem pre_cudaUpload {S : TState} {w : World} {xb dst size : UInt64} (hS : S.holds w)
    (hcu : S.contains (.part .cuda true) = true) (hroom : roomAt S dst size.toNat = true)
    (hseq : S.contains .devSeq = true) : Contracts.Pre .cudaUpload [cudaCtx, xb, dst, size] w := by
  simp only [Contracts.Pre]
  exact ⟨Contracts.ctxOk_of_state hS hcu, fun _ _ _ =>
    ⟨Contracts.readable_of_state hS hroom, Contracts.ready_of_state hS hseq _ _⟩⟩

/-- **`preB` is sound**: where it accepts, the contract table's precondition
    holds of the bits the arguments carry. -/
theorem preB_sound {f : Ffi} {as : List AV} {S : TState} {vs : List V} {bits : List UInt64} {w : World}
    (hd : DescAll as vs) (hb : vs.mapM asBits = some bits) (hp : preB f as S = true)
    (hS : S.holds w) : Contracts.Pre f bits w := by
  unfold preB at hp
  split at hp
  · rename_i hf
    split at hp
    · rename_i c b d s
      obtain ⟨dst, size, hc, hdd, hs, hcu, hroom, hseq⟩ := preCopy_spec hp
      obtain ⟨xb, rfl⟩ := bits4 hd hb hc hdd hs
      rcases hf with rfl | rfl
      · exact pre_cudaDownload hS hcu hroom hseq
      · exact pre_cudaUpload hS hcu hroom hseq
    · cases hp
  · cases hp

/-- The abstract arguments describe the values the argument slots hold. -/
theorem desc_slots {A : AS} {Γ : Env} {w : World} (h : A.Holds Γ w) :
    ∀ {rs : List R} {vs : List V}, (rs.map A.get).all (fun a => a.scTy.isSome) = true →
      rs.mapM (fun r => Γ[r]?) = some vs → DescAll (rs.map A.get) vs
  | [], vs, _, e => by simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at e; subst e; trivial
  | r :: rs, vs, hs, e => by
      simp only [List.map_cons, List.all_cons, Bool.and_eq_true] at hs
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at e
      obtain ⟨v, hv, vs', hvs, rfl⟩ := e
      obtain ⟨t, ht⟩ := Option.isSome_iff_exists.mp hs.1
      obtain ⟨u, hu, hd⟩ := h.get (AS.get_lt_scTy ht)
      rw [hv] at hu; cases hu
      exact ⟨hd, desc_slots h hs.2 hvs⟩

/-- **A call the checker accepts answers**, into a world `postA` describes,
    with an answer of the entry point's result type. -/
theorem call_ok (cfg : Cfg) {f : Ffi} {as : List AV} {S : TState} {vs : List V} {w : World}
    (hf : f ≠ .threadSpawn) (hz : S.contains (.part .frozen false) = true)
    (hd : DescAll as vs) (hs : as.all (fun a => a.scTy.isSome) = true)
    (hp : preB f as S = true) (hS : S.holds w) :
    ∃ r w', callOf cfg.locals (.ffi f) vs (obsCall w (.ffi f) vs) = some (r, w') ∧
      (postA f as S).holds w' ∧ ∀ v, r = some v → (AV.sc (f.result.getD .i64)).Desc v := by
  obtain ⟨bits, hb, he⟩ := bits_of_desc hd hs
  have hS1 : S.holds (obsCall w (.ffi f) vs) := TState.holds_obsCall hS
  rw [Contracts.callOf_ffi hf _ _ (holds_of_mem hS1 hz), hb, Option.bind_some]
  obtain ⟨⟨r, w'⟩, hr⟩ := Option.isSome_iff_exists.mp
    (Contracts.pre_safe f bits _ (preB_sound hd hb hp hS1))
  refine ⟨r, w', hr, postA_sound he hS1 hr, fun v hv => ?_⟩
  subst hv
  exact Contracts.callBits_scalar f hr

-- ---------------------------------------------------------------------------
-- The checker
-- ---------------------------------------------------------------------------

/-- **The checker.** `K` is asked of what the body hands its continuation and
    the state it ends in. Constructs it does not yet follow are refused. -/
def chk : AS → {α : Type} → Prog Slot Lvl α → (α → AS → Bool) → Bool
  | A, _, .ret a, K => K a A
  | A, _, .op o k, K =>
      match absOp A o.erase with
      | some av => chk ⟨A.env.push av, A.facts⟩ (k A.env.size) K
      | none => false
  | A, _, .call f args k, K =>
      let as := args.slots.map A.get
      if f != .threadSpawn && A.facts.contains (.part .frozen false) &&
          as.all (fun a => a.scTy.isSome) && preB f as A.facts then
        if f.result.isSome then
          chk ⟨A.env.push (.sc (f.result.getD .i64)), postA f as A.facts⟩ (k (resSlot f.result A.env.size)) K
        else chk ⟨A.env, postA f as A.facts⟩ (k (resSlot f.result 0)) K
      else false
  | A, _, .params tys k, K => chk A (k (carriesFrom 0 tys)) K
  | _, _, _, _ => false
termination_by structural _ _ p _ => p

/-- **The checker is sound.** Where it accepts a body from an abstract state,
    the body's weakest precondition holds in every state that one describes,
    for any post its continuation check guarantees — under any `Post`, so with
    `faultOk := false` too: an accepted body neither misuses nor faults. -/
theorem chk_sound (cfg : Cfg) (d : Nat) (J : Post) {α : Type} (p : Prog Slot Lvl α) :
    ∀ (A : AS) (K : α → AS → Bool) (Q : α → Env → World → Prop), chk A p K = true →
      (∀ a A' Γ w, K a A' = true → A'.Holds Γ w → Q a Γ w) →
      ∀ Γ w, A.Holds Γ w → wp cfg d J p Q Γ w := by
  induction p with
  | ret a => intro A K Q hc hK Γ w hA; exact hK a A Γ w hc hA
  | op o k ih =>
      intro A K Q hc hK Γ w hA
      rw [wp_op]
      unfold chk at hc
      split at hc
      · rename_i av hav
        obtain ⟨v, hv, hd⟩ := absOp_sound hA hav
        refine ⟨fun hn => (by rw [hv] at hn; cases hn), fun v' hv' => ?_⟩
        rw [hv] at hv'; cases hv'
        rw [hA.1]
        exact ih _ _ K Q hc hK _ _ (hA.push hd)
      · cases hc
  | call f args k ih =>
      intro A K Q hc hK Γ w hA
      rw [wp_call]
      intro vs hvs
      unfold chk at hc
      simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at hc
      split at hc
      · rename_i hcond
        obtain ⟨⟨⟨hf, hz⟩, hs⟩, hp⟩ := hcond
        obtain ⟨r, w', hcall, hpost, hans⟩ :=
          call_ok cfg hf hz (desc_slots hA hs hvs) hs hp hA.2.2
        refine ⟨r, w', hcall, ?_⟩
        split
        · rename_i hres
          rw [if_pos hres] at hc
          intro v hv
          rw [hA.1]
          exact ih _ _ K Q hc hK _ _ ((hA.facts hpost).push (hans v hv))
        · rename_i hres
          rw [if_neg hres] at hc
          exact ih _ _ K Q hc hK _ _ (hA.facts hpost)
      · cases hc
  | params tys k ih =>
      intro A K Q hc hK Γ w hA
      rw [wp_params]
      exact ih _ A K Q hc hK Γ w hA
  | store | storeUnaligned | istore8 | callLocal | loop | dloop | ite | br | cont =>
      intro A K Q hc; simp [chk] at hc

-- ---------------------------------------------------------------------------
-- Entry points
-- ---------------------------------------------------------------------------

/-- What the checker knows at an entry point: the three bases, and two lengths
    of unknown value. -/
def entryOf (S : TState) : AS :=
  ⟨#[.known (.sc .i64 (regionBase .arena)), .known (.sc .i64 (regionBase .data)), .sc .i64,
     .known (.sc .i64 (regionBase .out)), .sc .i64], S⟩

theorem entryOf_holds {S : TState} {w : World} (dataLen outLen : UInt64) (h : S.holds w) :
    (entryOf S).Holds (runArgs dataLen outLen).toArray w := by
  refine ⟨rfl, fun i hi => ?_, h⟩
  change i < 5 at hi
  match i, hi with
  | 0, _ => exact ⟨_, rfl, rfl⟩
  | 1, _ => exact ⟨_, rfl, rfl⟩
  | 2, _ => exact ⟨_, rfl, dataLen, rfl⟩
  | 3, _ => exact ⟨_, rfl, rfl⟩
  | 4, _ => exact ⟨_, rfl, outLen, rfl⟩
  | n + 5, h => exact absurd h (by omega)

/-- **An entry point the checker accepts neither misuses a call nor faults**,
    from any world its typestate describes, whatever lengths the caller passes. -/
theorem sound_of_chk {cfg : Cfg} {p : Body} (S : TState) (dataLen outLen : UInt64)
    (hc : chk (entryOf S) p (fun _ _ => true) = true)
    (hf : Fine (emitGo p ⟨ptrParams.length, 0, [], [], [], none⟩).2)
    {w : World} (hw : S.holds w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit p) ≠ .misuse m ∧
      Sem.run cfg (runArgs dataLen outLen) w (emit p) ≠ .fault m :=
  sound_of_wp_entry S.holds (runArgs dataLen outLen)
    (fun Γ w hΓ hw => by
      subst hΓ
      exact chk_sound cfg 0 _ p (entryOf S) _ _ hc (fun _ _ _ _ _ _ => trivial) _ _
        (entryOf_holds dataLen outLen hw))
    hf rfl hw m

end AlgorithmLib.Check
