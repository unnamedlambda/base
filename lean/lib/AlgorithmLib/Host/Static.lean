module
public import AlgorithmLib.Host.StaticCong
meta import AlgorithmLib.Host.StaticCong
public import AlgorithmLib.Host.Lifecycle
meta import AlgorithmLib.Host.Lifecycle
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Host.Static` — misuse ruled out before the program runs

`aCode` runs a term over *abstract* state: each slot is a value the checker
knows, a scalar it does not, or anything; memory is a canonical world that
agrees with every real run on everything except data, and the byte ranges it
knows. Control must follow known values: a loop's test is read at every trip,
and a branch on unknown data explores both arms. A foreign call must have known
arguments, be one the contract table says is blind to data (`Ffi.contract`'s
`blind`), with the bytes it needs known, and succeed on the canonical world.

**`progress`**: a term the checker accepts never ends in `misuse`, from any
state the abstract one describes --- so on any input. The argument is
`callOf_cong`: a call that succeeds on the canonical world succeeds on every
world that agrees with it up to data.
-/

namespace AlgorithmLib.HProg.Static

open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sem

-- ---------------------------------------------------------------------------
-- Abstract values
-- ---------------------------------------------------------------------------

/-- What the checker knows of a slot. -/
inductive AV where
  | known (v : V)
  /-- A scalar whose bits depend on data. -/
  | scalar
  | top
  deriving Inhabited

def AV.Rel : AV → V → Prop
  | .known v, x => x = v
  | .scalar, x => ∃ t b, x = .sc t b
  | .top, _ => True

/-- The value, where it is known. -/
def AV.val : AV → V
  | .known v => v
  | _ => default

def AV.get? : AV → Option V
  | .known v => some v
  | _ => none

abbrev AEnv := Array AV

/-- Each value described by the abstract one at its position. -/
def Rels : List AV → List V → Prop
  | [], [] => True
  | a :: as, v :: vs => a.Rel v ∧ Rels as vs
  | _, _ => False

theorem Rels.length_eq : ∀ {as : List AV} {vs : List V}, Rels as vs → as.length = vs.length
  | [], [], _ => rfl
  | _ :: _, _ :: _, h => by simp [Rels.length_eq h.2]
  | [], _ :: _, h => h.elim
  | _ :: _, [], h => h.elim

theorem Rels.get? : ∀ {as : List AV} {vs : List V}, Rels as vs → ∀ (i : Nat) (a : AV) (v : V),
    as[i]? = some a → vs[i]? = some v → a.Rel v
  | [], [], _, _, _, _, h, _ => by simp at h
  | a :: as, v :: vs, h, 0, b, x, hb, hx => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at hb hx; subst hb; subst hx; exact h.1
  | _ :: as, _ :: vs, h, i + 1, b, x, hb, hx => by
    simp only [List.getElem?_cons_succ] at hb hx; exact Rels.get? h.2 i b x hb hx
  | [], _ :: _, h, _, _, _, _, _ => h.elim
  | _ :: _, [], h, _, _, _, _, _ => h.elim

/-- Slots in scope in both, each described by its abstract value. -/
def EnvRel (E : AEnv) (Γ : Env) : Prop :=
  E.size = Γ.size ∧ ∀ (i : Nat) (a : AV) (v : V), E[i]? = some a → Γ[i]? = some v → a.Rel v

theorem EnvRel.get {E : AEnv} {Γ : Env} (h : EnvRel E Γ) (r : R) :
    (E[r]? = none ∧ Γ[r]? = none) ∨ ∃ a v, E[r]? = some a ∧ Γ[r]? = some v ∧ a.Rel v := by
  by_cases hr : r < Γ.size
  · right
    exact ⟨E[r]'(h.1 ▸ hr), Γ[r], by simp [h.1 ▸ hr], by simp [hr],
      h.2 r _ _ (by simp [h.1 ▸ hr]) (by simp [hr])⟩
  · left
    exact ⟨by simp [h.1, hr], by simp [hr]⟩

theorem EnvRel.known {E : AEnv} {Γ : Env} (h : EnvRel E Γ) {r : R} {v : V} (hk : E[r]? = some (.known v)) :
    Γ[r]? = some v := by
  rcases h.get r with ⟨h1, _⟩ | ⟨a, x, h1, h2, h3⟩
  · rw [hk] at h1; cases h1
  · rw [hk] at h1; injection h1 with h1; subst h1; rw [h2, h3]

theorem EnvRel.push {E : AEnv} {Γ : Env} (h : EnvRel E Γ) {a : AV} {v : V} (hv : a.Rel v) :
    EnvRel (E.push a) (Γ.push v) := by
  refine ⟨by simp [h.1], fun i b x hb hx => ?_⟩
  rw [Array.getElem?_push] at hb hx
  by_cases hi : i = Γ.size
  · rw [if_pos (h.1 ▸ hi)] at hb; rw [if_pos hi] at hx
    injection hb with hb; injection hx with hx; subst hb; subst hx; exact hv
  · rw [if_neg (h.1 ▸ hi)] at hb; rw [if_neg hi] at hx
    exact h.2 i b x hb hx

/-- `bindAt`, over abstract values. -/
def abindAt (E : AEnv) (n : Nat) (vs : List AV) : AEnv :=
  (E.take n ++ Array.replicate (n - E.size) (.known default)) ++ vs.toArray

theorem padBind_get? {α : Type} (xs : Array α) (n : Nat) (d : α) (ys : List α) (i : Nat) :
    ((xs.take n ++ Array.replicate (n - xs.size) d) ++ ys.toArray)[i]?
      = if i < n then (if i < xs.size then xs[i]? else some d) else ys[i - n]? := by
  simp only [Array.take_eq_extract, Array.getElem?_append, Array.size_append, Array.size_extract,
    Array.size_replicate, Array.getElem?_extract, Array.getElem?_replicate, List.getElem?_toArray]
  by_cases h1 : i < n
  · rw [if_pos h1]
    by_cases h2 : i < xs.size
    · rw [if_pos h2, if_pos (by omega), if_pos (by omega), if_pos (by omega)]; simp
    · rw [if_neg h2, if_pos (by omega), if_neg (by omega), if_pos (by omega)]
  · rw [if_neg h1, if_neg (by omega)]
    congr 1; omega

theorem EnvRel.bindAt {E : AEnv} {Γ : Env} (h : EnvRel E Γ) (n : Nat) {as : List AV} {vs : List V}
    (hv : Rels as vs) : EnvRel (abindAt E n as) (bindAt Γ n vs) := by
  have hl := hv.length_eq
  refine ⟨by simp [abindAt, Sem.bindAt, h.1, hl], fun i b x hb hx => ?_⟩
  simp only [abindAt, Sem.bindAt, padBind_get?] at hb hx
  rw [h.1] at hb
  split at hb
  · rename_i h1
    rw [if_pos h1] at hx
    split at hb
    · rename_i h2; rw [if_pos h2] at hx; exact h.2 i b x hb hx
    · rename_i h2; rw [if_neg h2] at hx
      simp only [Option.some.injEq] at hb hx; subst hb; subst hx; rfl
  · rename_i h1
    rw [if_neg h1] at hx
    exact hv.get? _ b x hb hx

theorem mapM_rel {E : AEnv} {Γ : Env} (h : EnvRel E Γ) :
    ∀ (rs : List R), ((rs.mapM fun r => E[r]?) = none ∧ (rs.mapM fun r => Γ[r]?) = none) ∨
      ∃ as vs, (rs.mapM fun r => E[r]?) = some as ∧ (rs.mapM fun r => Γ[r]?) = some vs ∧
        Rels as vs
  | [] => Or.inr ⟨[], [], rfl, rfl, trivial⟩
  | r :: rs => by
    simp only [List.mapM_cons, bind, Option.bind]
    rcases h.get r with ⟨h1, h2⟩ | ⟨a, v, h1, h2, h3⟩
    · left; rw [h1, h2]; exact ⟨rfl, rfl⟩
    · rw [h1, h2]
      rcases mapM_rel h rs with ⟨h4, h5⟩ | ⟨as, vs, h4, h5, h6⟩
      · left; rw [h4, h5]; exact ⟨rfl, rfl⟩
      · right; rw [h4, h5]; exact ⟨a :: as, v :: vs, rfl, rfl, h3, h6⟩

-- ---------------------------------------------------------------------------
-- Operations
-- ---------------------------------------------------------------------------

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

/-- The memory a load reads: `(address, bytes)` for each access. -/
def loadAccesses (op : LoadOp) (addr : UInt64) : List (UInt64 × Nat) :=
  match op.kind with
  | .uload8 | .sload8 => [(addr, 1)]
  | .uload16 | .sload16 => [(addr, 2)]
  | .uload32 | .sload32 => [(addr, 4)]
  | .plain =>
      match op.ty.lanes with
      | some (lane, n) => (List.range n).map fun i => (addr + UInt64.ofNat (i * tyBytes lane), tyBytes lane)
      | none => [(addr, tyBytes op.ty)]

theorem foldlM_congr_mem {α β : Type} (l : List α) (f g : β → α → Option β) (b : β)
    (h : ∀ x ∈ l, ∀ acc, f acc x = g acc x) : l.foldlM f b = l.foldlM g b := by
  induction l generalizing b with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.foldlM_cons]
    rw [h x (List.mem_cons_self ..)]
    cases g b x with
    | none => rfl
    | some b' => exact ih b' fun y hy => h y (List.mem_cons_of_mem x hy)

/-- A load reads the bytes of its accesses and nothing else. -/
theorem evalOp_load {K : Known} {m₁ m₂ : Mem} (hm : MemSame m₁ m₂) (hk : Agree K m₁ m₂) (Γ₁ Γ₂ : Env)
    (op : LoadOp) (p : R) (t : ClifTy) (addr : UInt64) (h₁ : Γ₁[p]? = some (.sc t addr))
    (h₂ : Γ₂[p]? = some (.sc t addr))
    (hc : (loadAccesses op addr).all (fun (x, n) => coversAt K x n) = true) :
    evalOp m₁ Γ₁ (.load op p) = evalOp m₂ Γ₂ (.load op p) := by
  simp only [evalOp, Sem.get, h₁, h₂, bind, Option.bind_some]
  unfold loadAccesses at hc
  cases hk' : op.kind with
  | uload8 =>
    rw [hk'] at hc; simp only [List.all_cons, List.all_nil, Bool.and_true] at hc
    simp only; rw [load_eq hm hk hc]
  | uload32 =>
    rw [hk'] at hc; simp only [List.all_cons, List.all_nil, Bool.and_true] at hc
    simp only; rw [load_eq hm hk hc]
  | uload16 =>
    rw [hk'] at hc; simp only [List.all_cons, List.all_nil, Bool.and_true] at hc
    simp only; rw [load_eq hm hk hc]
  | sload16 =>
    rw [hk'] at hc; simp only [List.all_cons, List.all_nil, Bool.and_true] at hc
    simp only; rw [load_eq hm hk hc]
  | sload32 =>
    rw [hk'] at hc; simp only [List.all_cons, List.all_nil, Bool.and_true] at hc
    simp only; rw [load_eq hm hk hc]
  | sload8 =>
    rw [hk'] at hc; simp only [List.all_cons, List.all_nil, Bool.and_true] at hc
    simp only; rw [load_eq hm hk hc]
  | plain =>
    rw [hk'] at hc
    simp only
    cases hl : op.ty.lanes with
    | none =>
      rw [hl] at hc
      simp only [List.all_cons, List.all_nil, Bool.and_true] at hc
      simp only; rw [load_eq hm hk hc]
    | some ln =>
      obtain ⟨lane, n⟩ := ln
      rw [hl] at hc
      simp only at hc ⊢
      have hf : (List.range n).foldlM (fun (acc : Array UInt64) i =>
            (m₁.load (addr + UInt64.ofNat (i * tyBytes lane)) (tyBytes lane)).bind fun b => pure (acc.push b)) #[]
          = (List.range n).foldlM (fun (acc : Array UInt64) i =>
            (m₂.load (addr + UInt64.ofNat (i * tyBytes lane)) (tyBytes lane)).bind fun b => pure (acc.push b)) #[] := by
        apply foldlM_congr_mem
        intro i hi acc
        have := List.all_eq_true.mp hc (_, _) (List.mem_map.mpr ⟨i, hi, rfl⟩)
        rw [load_eq hm hk this]
      rw [hf]

theorem bin_inv {Γ : Env} {a b : R} {f : ClifTy → UInt64 → UInt64 → Option V} {v : V}
    (h : bin Γ a b f = some v) : ∃ t x y, f t x y = some v := by
  unfold bin at h
  cases ha : Sem.get Γ a with
  | none => simp [ha, bind, Option.bind] at h
  | some va =>
    cases va with
    | vec _ _ => simp [ha, bind, Option.bind] at h
    | sc ta x =>
      cases hb : Sem.get Γ b with
      | none => simp [ha, hb, bind, Option.bind] at h
      | some vb =>
        cases vb with
        | vec _ _ => simp [ha, hb, bind, Option.bind] at h
        | sc tb y =>
          simp only [ha, hb, bind, Option.bind] at h
          split at h
          · exact ⟨_, _, _, h⟩
          · cases h

theorem shiftBin_inv {Γ : Env} {a b : R} {f : ClifTy → UInt64 → UInt64 → Option V} {v : V}
    (h : shiftBin Γ a b f = some v) : ∃ t x y, f t x y = some v := by
  unfold shiftBin at h
  cases ha : Sem.get Γ a with
  | none => simp [ha, bind, Option.bind] at h
  | some va =>
    cases va with
    | vec _ _ => simp [ha, bind, Option.bind] at h
    | sc ta x =>
      cases hb : Sem.get Γ b with
      | none => simp [ha, hb, bind, Option.bind] at h
      | some vb =>
        cases vb with
        | vec _ _ => simp [ha, hb, bind, Option.bind] at h
        | sc tb y =>
          simp only [ha, hb, bind, Option.bind] at h
          split at h
          · exact ⟨_, _, _, h⟩
          · cases h

theorem un_inv {Γ : Env} {a : R} {ok : ClifTy → Bool} {f : ClifTy → UInt64 → Option V} {v : V}
    (h : un Γ a ok f = some v) : ∃ t x, f t x = some v := by
  unfold un at h
  cases ha : Sem.get Γ a with
  | none => simp [ha, bind, Option.bind] at h
  | some va =>
    cases va with
    | vec _ _ => simp [ha, bind, Option.bind] at h
    | sc t x =>
      simp only [ha, bind, Option.bind] at h
      split at h
      · exact ⟨_, _, h⟩
      · cases h

/-- Integer operations whose result, when there is one, is a scalar. -/
def scalarOp : Op → Bool
  | .iconst .. | .iadd .. | .isub .. | .imul .. | .udiv .. | .ishl .. | .ushr .. | .ineg ..
  | .ireduce32 .. | .uextend64 .. | .sextend64 .. | .ctz .. | .popcnt .. => true
  | _ => false

theorem scalarOp_sc {m : Mem} {Γ : Env} {o : Op} (hs : scalarOp o = true) {v : V}
    (h : evalOp m Γ o = some v) : ∃ t b, v = .sc t b := by
  cases o <;> simp only [scalarOp, Bool.false_eq_true] at hs <;> simp only [evalOp] at h
  · injection h with h; exact ⟨_, _, h.symm⟩
  all_goals first
    | (obtain ⟨t, x, y, h⟩ := bin_inv h
       first
         | (injection h with h; exact ⟨_, _, h.symm⟩)
         | (split at h
            · cases h
            · injection h with h; exact ⟨_, _, h.symm⟩))
    | (obtain ⟨t, x, y, h⟩ := shiftBin_inv h; injection h with h; exact ⟨_, _, h.symm⟩)
    | (obtain ⟨t, x, h⟩ := un_inv h; injection h with h; exact ⟨_, _, h.symm⟩)

/-- A load whose result, when there is one, is a scalar. -/
def loadScalar (op : LoadOp) : Bool :=
  match op.kind with
  | .plain => op.ty.lanes.isNone
  | _ => true

theorem loadScalar_sc {m : Mem} {Γ : Env} {op : LoadOp} {p : R} (hs : loadScalar op = true) {v : V}
    (h : evalOp m Γ (.load op p) = some v) : ∃ t b, v = .sc t b := by
  simp only [evalOp] at h
  cases hp : Sem.get Γ p with
  | none => simp [hp, bind, Option.bind] at h
  | some vp =>
    cases vp with
    | vec _ _ => simp [hp, bind, Option.bind] at h
    | sc t addr =>
      simp only [hp, bind, Option.bind_some] at h
      unfold loadScalar at hs
      cases hk : op.kind <;> rw [hk] at h hs <;> simp only at h
      · cases hl : op.ty.lanes with
        | some ln => rw [hl] at hs; cases hs
        | none =>
          rw [hl] at h
          simp only at h
          cases hl2 : m.load addr (tyBytes op.ty) <;> rw [hl2] at h <;> simp at h
          exact ⟨_, _, h.symm⟩
      all_goals
        cases hl2 : m.load addr _ <;> rw [hl2] at h <;> simp at h
        exact ⟨_, _, h.symm⟩

-- ---------------------------------------------------------------------------
-- Abstract state
-- ---------------------------------------------------------------------------

/-- The environment with every unknown slot filled with a default. -/
def fill (E : AEnv) : Env := E.map AV.val

/-- What the checker holds: its slots, the canonical world, and the host bytes
    that world agrees with every real one on. -/
structure AS where
  env : AEnv
  w : World
  known : Known

/-- **A concrete state the abstract one describes.** -/
def Rel (a : AS) (Γ : Env) (w : World) : Prop := EnvRel a.env Γ ∧ Same a.known w a.w

theorem fill_get {E : AEnv} {r : R} {v : V} (h : E[r]? = some (.known v)) : (fill E)[r]? = some v := by
  simp [fill, Array.getElem?_map, h, AV.val]

/-- Whether the slot is known. -/
def AV.isKnown : AV → Bool
  | .known _ => true
  | _ => false

/-- An operation, abstractly: its value where every operand (and, for a load,
    every byte) is known; otherwise a scalar where it can only be one. -/
def aOp (a : AS) (o : Op) : AV :=
  match o with
  | .load op p =>
      match a.env[p]? with
      | some (.known (.sc _ addr)) =>
          if (loadAccesses op addr).all (fun (x, n) => coversAt a.known x n) then
            match evalOp a.w.mem (fill a.env) o with
            | some v => .known v
            | none => .top
          else if loadScalar op then .scalar else .top
      | _ => if loadScalar op then .scalar else .top
  | _ =>
      if o.regs.all (fun r => (a.env[r]?.map AV.isKnown).getD false) then
        match evalOp a.w.mem (fill a.env) o with
        | some v => .known v
        | none => .top
      else if scalarOp o then .scalar else .top

theorem aOp_nonload (a : AS) (o : Op) (hl : ∀ op p, o ≠ .load op p) :
    aOp a o = if o.regs.all (fun r => (a.env[r]?.map AV.isKnown).getD false) then
        (match evalOp a.w.mem (fill a.env) o with
         | some v => .known v
         | none => .top)
      else if scalarOp o then .scalar else .top := by
  cases o <;> first | rfl | exact absurd rfl (hl _ _)

theorem aOp_sound {a : AS} {Γ : Env} {w : World} (hr : Rel a Γ w) (o : Op) {v : V}
    (h : evalOp w.mem Γ o = some v) : (aOp a o).Rel v := by
  have fallback : ∀ (b : Bool), (b = true → ∃ t x, v = .sc t x) →
      (if b then AV.scalar else AV.top).Rel v := by
    intro b hb
    cases b
    · trivial
    · obtain ⟨t, x, e⟩ := hb rfl; exact ⟨t, x, e⟩
  by_cases hl : ∃ op p, o = .load op p
  · obtain ⟨op, p, rfl⟩ := hl
    simp only [aOp]
    split
    · rename_i t addr hp
      split
      · rename_i hc
        have hΓ := hr.1.known hp
        have e := evalOp_load hr.2.mem hr.2.agree Γ (fill a.env) op p t addr hΓ (fill_get hp) hc
        rw [e] at h
        rw [h]
        rfl
      · exact fallback _ fun hs => loadScalar_sc hs h
    · exact fallback _ fun hs => loadScalar_sc hs h
  · have hl' : ∀ op p, o ≠ .load op p := fun op p e => hl ⟨op, p, e⟩
    rw [aOp_nonload a o hl']
    split
    · rename_i hk
      have e := evalOp_congr w.mem a.w.mem Γ (fill a.env) o hl' (fun r hrm => by
        have := List.all_eq_true.mp hk r hrm
        cases hE : a.env[r]? with
        | none => rw [hE] at this; cases this
        | some x =>
          rw [hE] at this
          cases x with
          | known y => rw [hr.1.known hE, fill_get hE]
          | scalar => cases this
          | top => cases this)
      rw [e] at h
      rw [h]
      rfl
    · exact fallback _ fun hs => scalarOp_sc hs h

-- ---------------------------------------------------------------------------
-- Statements
-- ---------------------------------------------------------------------------

/-- `K` with the `n` bytes at `addr` added. -/
def addAt (addr : UInt64) (n : Nat) (K : Known) : Known :=
  match decodeAddr addr with
  | some (r, off) => (r, off, n) :: K
  | none => K

/-- `K` without the `n` bytes at `addr`. -/
def killAt (addr : UInt64) (n : Nat) (K : Known) : Known :=
  match decodeAddr addr with
  | some (r, off) => kill r off n K
  | none => K

/-- What a statement leaves: the state it goes on in, or the knowledge that the
    run stops here, and not by misuse. -/
inductive AOut where
  | go (a : AS)
  | halt

/-- A store of `n` bytes at a known address: of known bits, made on the
    canonical world too; of unknown ones, those bytes forgotten. -/
def aStore (a : AS) (addr : UInt64) (n : Nat) : Option UInt64 → AOut
  | some b =>
      match a.w.mem.store addr n b with
      | some m => .go { a with w := { a.w with mem := m }, known := addAt addr n a.known }
      | none => .halt
  | none => .go { a with known := killAt addr n a.known }

/-- A call, abstractly: every argument known, an entry the contract table says
    is blind to data with what it needs known, and success on the canonical
    world. -/
def aCall (a : AS) (c : Callee) (args : List R) (binds : Bool) : Option AOut :=
  match args.mapM (fun r => a.env[r]?) with
  | none => some .halt
  | some avs =>
    match avs.mapM AV.get?, c with
    | some vs, .ffi f =>
      match vs.mapM asBits with
      | none => none
      | some bits =>
        match f.contract.blind with
        | none => none
        | some b =>
          if b.need bits a.known a.w.mem then
            match callOf noLocals (.ffi f) vs (obsCall a.w (.ffi f) vs) with
            | none => none
            | some (res, w') =>
                if binds then
                  match res with
                  | some v => some (.go { env := a.env.push (.known v), w := w', known := b.post bits a.known })
                  | none => some .halt
                else some (.go { a with w := w', known := b.post bits a.known })
          else none
    | _, _ => none

def aStmt (a : AS) : Stmt → Option AOut
  | .op o => some (.go { a with env := a.env.push (aOp a o) })
  | .store ty v p =>
      match a.env[v]?, a.env[p]? with
      | some x, some (.known (.sc _ addr)) =>
          match x with
          | .known (.sc _ b) => some (aStore a addr (tyBytes ty) (some b))
          | .scalar => some (aStore a addr (tyBytes ty) none)
          | _ => some (.go { a with known := [] })
      | some _, some (.known (.vec _ _)) => some .halt
      | some _, some _ => some (.go { a with known := [] })
      | _, _ => some .halt
  | .storeUnaligned v p =>
      match a.env[v]?, a.env[p]? with
      | some (.known (.sc t b)), some (.known (.sc _ addr)) => some (aStore a addr (tyBytes t) (some b))
      | some .scalar, some (.known (.sc _ addr)) => some (aStore a addr 16 none)
      -- a vector, or a value not known to be a scalar: it stores lane by lane
      | some _, some (.known (.sc _ _)) => some (.go { a with known := [] })
      | some _, some (.known (.vec _ _)) => some .halt
      | some _, some _ => some (.go { a with known := [] })
      | _, _ => some .halt
  | .istore8 v p =>
      match a.env[v]?, a.env[p]? with
      | some (.known (.sc _ b)), some (.known (.sc _ addr)) => some (aStore a addr 1 (some (b &&& 0xff)))
      | some _, some (.known (.sc _ addr)) => some (aStore a addr 1 none)
      | some _, some (.known (.vec _ _)) => some .halt
      | some _, some _ => some (.go { a with known := [] })
      | _, _ => some .halt
  | .call c args => aCall a c args true
  | .callVoid c args => aCall a c args false

/-- What a concrete outcome must be, given the abstract one. -/
def StmtSim (o : AOut) : Outcome Env → Prop
  | .ok Γ' w' => ∃ a', o = .go a' ∧ Rel a' Γ' w'
  | .stuck _ => True
  | .misuse _ => False
  | .fault _ => True

theorem Rel.mem {a : AS} {Γ : Env} {w : World} (hr : Rel a Γ w) {m : Mem} (hm : MemSame m a.w.mem)
    (hk : Agree a.known m a.w.mem) : Rel a Γ { w with mem := m } :=
  ⟨hr.1, hm, hk, hr.2.dev, hr.2.kernel₁, hr.2.kernel₂, hr.2.cudaDevice⟩

theorem layout_store {m m' : Mem} {a : UInt64} {n : Nat} {v : UInt64} (h : m.store a n v = some m') :
    MemSame m m' := by
  obtain ⟨_, _, _, hf⟩ := store_frame h; exact hf.same

theorem layout_stores {α : Type} (g : α → UInt64) (n : α → Nat) (v : α → UInt64) :
    ∀ (l : List α) (m m' : Mem), l.foldlM (fun mm x => mm.store (g x) (n x) (v x)) m = some m' →
      MemSame m m'
  | [], m, m', h => by cases h; exact MemSame.refl m
  | x :: xs, m, m', h => by
    simp only [List.foldlM_cons] at h
    cases h1 : m.store (g x) (n x) (v x) with
    | none => rw [h1] at h; cases h
    | some m1 => rw [h1] at h; exact (layout_store h1).trans (layout_stores g n v xs m1 m' h)

/-- A store the checker could not follow: the layout still agrees, nothing
    known survives. -/
theorem Rel.forget {a : AS} {Γ : Env} {w : World} (hr : Rel a Γ w) {m : Mem} (hm : MemSame w.mem m) :
    Rel { a with known := [] } Γ { w with mem := m } :=
  ⟨hr.1, hm.symm.trans hr.2.mem, Agree.nil _ _, hr.2.dev, hr.2.kernel₁, hr.2.kernel₂, hr.2.cudaDevice⟩

theorem aStore_known {a : AS} {Γ : Env} {w : World} (hr : Rel a Γ w) {addr : UInt64} {n : Nat}
    {b : UInt64} {m : Mem} (h : w.mem.store addr n b = some m) :
    ∃ a', aStore a addr n (some b) = .go a' ∧ Rel a' Γ { w with mem := m } := by
  have hs := store_isSome hr.2.mem addr n b b
  rw [h] at hs
  cases h2 : a.w.mem.store addr n b with
  | none => rw [h2] at hs; cases hs
  | some m₂ =>
    obtain ⟨r, off, hd, hm, hk⟩ := store_both hr.2.mem hr.2.agree h h2
    refine ⟨{ a with w := { a.w with mem := m₂ }, known := addAt addr n a.known },
      by simp only [aStore, h2], hr.1, hm, ?_, hr.2.dev, hr.2.kernel₁, hr.2.kernel₂, hr.2.cudaDevice⟩
    show Agree (addAt addr n a.known) m m₂
    simp only [addAt, hd]; exact hk

theorem aStore_unknown {a : AS} {Γ : Env} {w : World} (hr : Rel a Γ w) {addr : UInt64} {n n' : Nat}
    (hn : n' ≤ n) {b : UInt64} {m : Mem} (h : w.mem.store addr n' b = some m) :
    ∃ a', aStore a addr n none = .go a' ∧ Rel a' Γ { w with mem := m } := by
  obtain ⟨r, off, hd, hf⟩ := store_frame h
  have hf' := hf.widen (Nat.le_refl off) (by omega : off + n' ≤ off + n)
  obtain ⟨hm, hk⟩ := hf'.agree hr.2.mem hr.2.agree
  refine ⟨{ a with known := killAt addr n a.known }, rfl, hr.1, hm, ?_, hr.2.dev, hr.2.kernel₁,
    hr.2.kernel₂, hr.2.cudaDevice⟩
  show Agree (killAt addr n a.known) m a.w.mem
  simp only [killAt, hd]; exact hk

theorem aStore_halt_sound {a : AS} {Γ : Env} {w : World} (hr : Rel a Γ w) {addr : UInt64} {n : Nat}
    {b : UInt64} (h : a.w.mem.store addr n b = none) : w.mem.store addr n b = none :=
  store_same_none hr.2.mem h

theorem tyBytes_le (t : ClifTy) : tyBytes t ≤ 16 := by cases t <;> decide

theorem Rel.congr {a : AS} {Γ : Env} {w w' : World} (h : Rel a Γ w) (hm : w'.mem = w.mem)
    (hd : w'.dev = w.dev) (hk : w'.kernel = w.kernel)
    (hc : w'.cudaDevice = w.cudaDevice := by first | rfl | assumption) : Rel a Γ w' :=
  ⟨h.1, hm ▸ h.2.mem, hm ▸ h.2.agree, hd ▸ h.2.dev, hk ▸ h.2.kernel₁, h.2.kernel₂, hc.trans h.2.cudaDevice⟩

/-- A vector store: the layout still agrees. -/
theorem vecStore_layout {m m' : Mem} {addr : UInt64} {lw : Nat} {ls : Array UInt64}
    (h : ls.zipIdx.foldlM (fun mm (x, i) => mm.store (addr + UInt64.ofNat (i * lw)) lw x) m = some m') :
    MemSame m m' := by
  rw [← Array.foldlM_toList] at h
  exact layout_stores (fun (p : UInt64 × Nat) => addr + UInt64.ofNat (p.2 * lw)) (fun _ => lw)
    (fun p => p.1) _ m m' h

theorem store_forget {a : AS} {Γ : Env} {w : World} (hr : Rel a Γ w) (cfg : Cfg) (ty : ClifTy)
    {v p : R} {x : V} {t : ClifTy} {addr : UInt64} (hv : Γ[v]? = some x) (hp : Γ[p]? = some (.sc t addr)) :
    StmtSim (.go { a with known := [] }) (runStmt cfg Γ w (.store ty v p)) := by
  rw [runStmt_store, hv, hp]
  cases x with
  | sc t' b =>
    simp only
    cases hs : w.mem.store addr (tyBytes ty) b with
    | none => trivial
    | some m => exact ⟨_, rfl, (hr.forget (layout_store hs)).congr rfl rfl rfl⟩
  | vec t' ls =>
    simp only
    split
    · rename_i m hm
      exact ⟨_, rfl, (hr.forget (vecStore_layout hm)).congr rfl rfl rfl⟩
    · trivial

theorem aStmt_store_sound {a : AS} {Γ : Env} {w : World} (hr : Rel a Γ w) (cfg : Cfg)
    (ty : ClifTy) (v p : R) {o : AOut} (h : aStmt a (.store ty v p) = some o) :
    StmtSim o (runStmt cfg Γ w (.store ty v p)) := by
  simp only [aStmt] at h
  rcases hr.1.get v with ⟨hv1, hv2⟩ | ⟨av, x, hv1, hv2, hvr⟩
  · rw [runStmt_store, hv2]; trivial
  rcases hr.1.get p with ⟨hp1, hp2⟩ | ⟨ap, y, hp1, hp2, hpr⟩
  · rw [runStmt_store, hv2, hp2]; trivial
  rw [hv1, hp1] at h
  cases y with
  | vec t ls => rw [runStmt_store, hv2, hp2]; trivial
  | sc t addr =>
    cases ap with
    | known y' =>
      simp only [AV.Rel] at hpr; subst hpr
      simp only at h
      cases x with
      | vec t' ls =>
        cases av with
        | known x' =>
          simp only [AV.Rel] at hvr; subst hvr
          simp only at h; injection h with h; subst h; exact store_forget hr cfg ty hv2 hp2
        | scalar => obtain ⟨_, _, e⟩ := hvr; cases e
        | top => simp only at h; injection h with h; subst h; exact store_forget hr cfg ty hv2 hp2
      | sc t' b =>
        rw [runStmt_store, hv2, hp2]
        simp only
        cases av with
        | known x' =>
          simp only [AV.Rel] at hvr; subst hvr
          simp only at h; injection h with h; subst h
          cases hs : w.mem.store addr (tyBytes ty) b with
          | none => trivial
          | some m =>
            obtain ⟨a', ha', hr'⟩ := aStore_known hr hs
            exact ⟨a', ha', hr'.congr rfl rfl rfl⟩
        | scalar =>
          simp only at h; injection h with h; subst h
          cases hs : w.mem.store addr (tyBytes ty) b with
          | none => trivial
          | some m =>
            obtain ⟨a', ha', hr'⟩ := aStore_unknown hr (Nat.le_refl _) hs
            exact ⟨a', ha', hr'.congr rfl rfl rfl⟩
        | top =>
          simp only at h; injection h with h; subst h
          cases hs : w.mem.store addr (tyBytes ty) b with
          | none => trivial
          | some m => exact ⟨_, rfl, (hr.forget (layout_store hs)).congr rfl rfl rfl⟩
    | scalar =>
      simp only at h; injection h with h; subst h; exact store_forget hr cfg ty hv2 hp2
    | top =>
      simp only at h; injection h with h; subst h; exact store_forget hr cfg ty hv2 hp2

theorem aStmt_storeUnaligned_sound {a : AS} {Γ : Env} {w : World} (hr : Rel a Γ w) (cfg : Cfg)
    (v p : R) {o : AOut} (h : aStmt a (.storeUnaligned v p) = some o) :
    StmtSim o (runStmt cfg Γ w (.storeUnaligned v p)) := by
  simp only [aStmt] at h
  rcases hr.1.get v with ⟨hv1, hv2⟩ | ⟨av, x, hv1, hv2, hvr⟩
  · rw [runStmt_storeUnaligned, hv2]; trivial
  rcases hr.1.get p with ⟨hp1, hp2⟩ | ⟨ap, y, hp1, hp2, hpr⟩
  · rw [runStmt_storeUnaligned, hv2, hp2]; cases x <;> trivial
  rw [hv1, hp1] at h
  rw [runStmt_storeUnaligned, hv2, hp2]
  -- a vector: the memory it leaves keeps its layout, and nothing is known of it
  have vecForget : ∀ {t' : ClifTy} {ls : Array UInt64} {addr : UInt64},
      StmtSim (.go { a with known := [] })
        (match ls.zipIdx.foldlM (fun mm (x, i) =>
            mm.store (addr + UInt64.ofNat (i * (((t'.lanes.map (·.1.width)).getD 8) / 8)))
              (((t'.lanes.map (·.1.width)).getD 8) / 8) x) w.mem with
          | some m => .ok Γ { obsStore w addr (tyBytes t') 0 with mem := m }
          | none => .fault s!"vector store to unmapped address {addr}") := by
    intro t' ls addr
    split
    · rename_i m hm
      exact ⟨_, rfl, (hr.forget (vecStore_layout hm)).congr rfl rfl rfl⟩
    · trivial
  cases y with
  | vec t ls => cases x <;> trivial
  | sc t addr =>
    cases x with
    | vec t' ls =>
      simp only
      cases ap with
      | known y' =>
        simp only [AV.Rel] at hpr; subst hpr
        cases av with
        | known x' =>
          simp only [AV.Rel] at hvr; subst hvr
          simp only at h; injection h with h; subst h; exact vecForget
        | scalar => obtain ⟨_, _, e⟩ := hvr; cases e
        | top => simp only at h; injection h with h; subst h; exact vecForget
      | scalar => cases av <;> (simp only at h; injection h with h; subst h; exact vecForget)
      | top => cases av <;> (simp only at h; injection h with h; subst h; exact vecForget)
    | sc t' b =>
      simp only
      cases hs : w.mem.store addr (tyBytes t') b with
      | none => trivial
      | some m =>
        cases ap with
        | known y' =>
          simp only [AV.Rel] at hpr; subst hpr
          cases av with
          | known x' =>
            simp only [AV.Rel] at hvr; subst hvr
            simp only at h; injection h with h; subst h
            obtain ⟨a', ha', hr'⟩ := aStore_known hr hs
            exact ⟨a', ha', hr'.congr rfl rfl rfl⟩
          | scalar =>
            simp only at h; injection h with h; subst h
            obtain ⟨a', ha', hr'⟩ := aStore_unknown hr (tyBytes_le t') hs
            exact ⟨a', ha', hr'.congr rfl rfl rfl⟩
          | top =>
            simp only at h; injection h with h; subst h
            exact ⟨_, rfl, (hr.forget (layout_store hs)).congr rfl rfl rfl⟩
        | scalar =>
          cases av <;>
          · simp only at h; injection h with h; subst h
            exact ⟨_, rfl, (hr.forget (layout_store hs)).congr rfl rfl rfl⟩
        | top =>
          cases av <;>
          · simp only at h; injection h with h; subst h
            exact ⟨_, rfl, (hr.forget (layout_store hs)).congr rfl rfl rfl⟩

theorem aStmt_istore8_sound {a : AS} {Γ : Env} {w : World} (hr : Rel a Γ w) (cfg : Cfg)
    (v p : R) {o : AOut} (h : aStmt a (.istore8 v p) = some o) :
    StmtSim o (runStmt cfg Γ w (.istore8 v p)) := by
  simp only [aStmt] at h
  rcases hr.1.get v with ⟨hv1, hv2⟩ | ⟨av, x, hv1, hv2, hvr⟩
  · rw [runStmt_istore8, hv2]; trivial
  rcases hr.1.get p with ⟨hp1, hp2⟩ | ⟨ap, y, hp1, hp2, hpr⟩
  · rw [runStmt_istore8, hv2, hp2]; cases x <;> trivial
  rw [hv1, hp1] at h
  rw [runStmt_istore8, hv2, hp2]
  cases y with
  | vec t ls => cases x <;> trivial
  | sc t addr =>
    cases x with
    | vec t' ls => trivial
    | sc t' b =>
      simp only
      cases hs : w.mem.store addr 1 (b &&& 0xff) with
      | none => trivial
      | some m =>
        cases ap with
        | known y' =>
          simp only [AV.Rel] at hpr; subst hpr
          cases av with
          | known x' =>
            simp only [AV.Rel] at hvr; subst hvr
            simp only at h; injection h with h; subst h
            obtain ⟨a', ha', hr'⟩ := aStore_known hr hs
            exact ⟨a', ha', hr'.congr rfl rfl rfl⟩
          | scalar =>
            simp only at h; injection h with h; subst h
            obtain ⟨a', ha', hr'⟩ := aStore_unknown hr (Nat.le_refl 1) hs
            exact ⟨a', ha', hr'.congr rfl rfl rfl⟩
          | top =>
            simp only at h; injection h with h; subst h
            obtain ⟨a', ha', hr'⟩ := aStore_unknown hr (Nat.le_refl 1) hs
            exact ⟨a', ha', hr'.congr rfl rfl rfl⟩
        | scalar =>
          cases av <;>
          · simp only at h; injection h with h; subst h
            exact ⟨_, rfl, (hr.forget (layout_store hs)).congr rfl rfl rfl⟩
        | top =>
          cases av <;>
          · simp only at h; injection h with h; subst h
            exact ⟨_, rfl, (hr.forget (layout_store hs)).congr rfl rfl rfl⟩

theorem rels_known : ∀ {avs : List AV} {vs' vs : List V}, Rels avs vs' → avs.mapM AV.get? = some vs →
    vs' = vs
  | [], [], vs, _, h => by simp at h; exact h.symm ▸ rfl
  | a :: as, v :: vs', vs, h, hk => by
    simp only [List.mapM_cons, bind, Option.bind] at hk
    cases ha : a.get? with
    | none => rw [ha] at hk; cases hk
    | some x =>
      rw [ha] at hk
      cases hr : as.mapM AV.get? with
      | none => rw [hr] at hk; cases hk
      | some xs =>
        rw [hr] at hk
        simp only [pure, Option.some.injEq] at hk
        subst hk
        cases a with
        | known y =>
          simp only [AV.get?, Option.some.injEq] at ha; subst ha
          rw [show v = y from h.1, rels_known h.2 hr]
        | scalar => cases ha
        | top => cases ha
  | [], _ :: _, _, h, _ => h.elim
  | _ :: _, [], _, h, _ => h.elim

/-- The part of a call both statements share: the arguments, the contract, and
    the world it leaves. -/
theorem aCall_core {a : AS} {Γ : Env} {w : World} (hr : Rel a Γ w) (cfg : Cfg) (c : Callee)
    (args : List R) (binds : Bool) {o : AOut} (h : aCall a c args binds = some o) :
    (args.mapM (fun r => Γ[r]?) = none) ∨
      ∃ vs f bits res w₁ w₂, args.mapM (fun r => Γ[r]?) = some vs ∧ c = .ffi f ∧
        vs.mapM asBits = some bits ∧
        callOf cfg.locals c vs (obsCall w c vs) = some (res, w₁) ∧
        Same (post f bits a.known) w₁ w₂ ∧
        (if binds then
          (match res with
           | some v => o = .go { env := a.env.push (.known v), w := w₂, known := post f bits a.known }
           | none => o = .halt)
         else o = .go { a with w := w₂, known := post f bits a.known }) := by
  unfold aCall at h
  rcases mapM_rel hr.1 args with ⟨h1, h2⟩ | ⟨avs, vs', h1, h2, h3⟩
  · exact Or.inl h2
  right
  rw [h1] at h
  simp only at h
  cases hk : avs.mapM AV.get? with
  | none => rw [hk] at h; cases h
  | some vs =>
    have e := rels_known h3 hk; subst e
    rw [hk] at h
    cases c with
    | «local» i => cases h
    | native => cases h
    | atomic _ => cases h
    | ext _ => cases h
    | ffi f =>
      simp only at h
      cases hb : vs'.mapM asBits with
      | none => rw [hb] at h; cases h
      | some bits =>
        rw [hb] at h
        simp only at h
        cases hbl : f.contract.blind with
        | none => rw [hbl] at h; cases h
        | some bl =>
        rw [hbl] at h
        obtain ⟨hsup, rfl⟩ := Contracts.blind_some hbl
        simp only at h
        split at h
        · rename_i hn
          have hc := callOf_cong (hr.2.obsCall (.ffi f) vs') hsup vs' hb hn cfg.locals noLocals
          cases hca : callOf noLocals (.ffi f) vs' (obsCall a.w (.ffi f) vs') with
          | none => rw [hca] at h; cases h
          | some p =>
            obtain ⟨res, w₂⟩ := p
            rw [hca] at h hc
            cases hcc : callOf cfg.locals (.ffi f) vs' (obsCall w (.ffi f) vs') with
            | none => rw [hcc] at hc; cases hc
            | some q =>
              obtain ⟨res', w₁⟩ := q
              rw [hcc] at hc
              obtain ⟨rfl, hsame⟩ := hc
              refine ⟨vs', f, bits, res', w₁, w₂, h2, rfl, hb, hcc, hsame, ?_⟩
              simp only at h
              cases binds
              · simp only [Bool.false_eq_true, if_false] at h ⊢
                injection h with h; exact h.symm
              · simp only [if_true] at h ⊢
                cases res' with
                | some v => simp only at h ⊢; injection h with h; exact h.symm
                | none => simp only at h ⊢; injection h with h; exact h.symm
        · cases h

theorem aStmt_call_sound {a : AS} {Γ : Env} {w : World} (hr : Rel a Γ w) (cfg : Cfg) (c : Callee)
    (args : List R) {o : AOut} (h : aStmt a (.call c args) = some o) :
    StmtSim o (runStmt cfg Γ w (.call c args)) := by
  rw [runStmt_call]
  rcases aCall_core hr cfg c args true h with h2 | ⟨vs, f, bits, res, w₁, w₂, h2, rfl, hb, hc, hs, ho⟩
  · rw [h2]; trivial
  rw [h2]
  simp only [hc]
  simp only [if_true] at ho
  cases res with
  | some v => subst ho; exact ⟨_, rfl, hr.1.push rfl, hs⟩
  | none => trivial

theorem aStmt_callVoid_sound {a : AS} {Γ : Env} {w : World} (hr : Rel a Γ w) (cfg : Cfg) (c : Callee)
    (args : List R) {o : AOut} (h : aStmt a (.callVoid c args) = some o) :
    StmtSim o (runStmt cfg Γ w (.callVoid c args)) := by
  rw [runStmt_callVoid]
  rcases aCall_core hr cfg c args false h with h2 | ⟨vs, f, bits, res, w₁, w₂, h2, rfl, hb, hc, hs, ho⟩
  · rw [h2]; trivial
  rw [h2]
  simp only [hc]
  simp only [Bool.false_eq_true, if_false] at ho
  subst ho
  exact ⟨_, rfl, hr.1, hs⟩

/-- **One statement, abstractly and concretely**: where the checker goes on,
    the run goes on in a state it describes, or stops for a reason other than
    misuse. -/
theorem aStmt_sound {a : AS} {Γ : Env} {w : World} (hr : Rel a Γ w) (cfg : Cfg) (s : Stmt) {o : AOut}
    (h : aStmt a s = some o) : StmtSim o (runStmt cfg Γ w s) := by
  cases s with
  | op o' =>
    simp only [aStmt] at h; injection h with h; subst h
    simp only [runStmt]
    cases he : evalOp w.mem Γ o' with
    | none => trivial
    | some v => exact ⟨_, rfl, hr.1.push (aOp_sound hr o' he), hr.2⟩
  | store ty v p => exact aStmt_store_sound hr cfg ty v p h
  | storeUnaligned v p => exact aStmt_storeUnaligned_sound hr cfg v p h
  | istore8 v p => exact aStmt_istore8_sound hr cfg v p h
  | call c args => exact aStmt_call_sound hr cfg c args h
  | callVoid c args => exact aStmt_callVoid_sound hr cfg c args h

def aStmts (a : AS) : List Stmt → Option AOut
  | [] => some (.go a)
  | s :: ss =>
      match aStmt a s with
      | none => none
      | some .halt => some .halt
      | some (.go a') => aStmts a' ss

theorem aStmts_sound (cfg : Cfg) : ∀ (ss : List Stmt) {a : AS} {Γ : Env} {w : World} {o : AOut},
    Rel a Γ w → aStmts a ss = some o → StmtSim o (runStmts cfg Γ w ss)
  | [], a, Γ, w, o, hr, h => by
    simp only [aStmts, Option.some.injEq] at h; subst h; exact ⟨a, rfl, hr⟩
  | s :: ss, a, Γ, w, o, hr, h => by
    simp only [aStmts] at h
    simp only [runStmts]
    cases hs : aStmt a s with
    | none => rw [hs] at h; cases h
    | some o1 =>
      have hsim := aStmt_sound hr cfg s hs
      rw [hs] at h
      cases hrun : runStmt cfg Γ w s with
      | stuck m => trivial
      | fault m => trivial
      | misuse m => rw [hrun] at hsim; exact hsim.elim
      | ok Γ1 w1 =>
        rw [hrun] at hsim
        obtain ⟨a1, rfl, hr1⟩ := hsim
        simp only at h
        exact aStmts_sound cfg ss hr1 h

-- ---------------------------------------------------------------------------
-- Regions
-- ---------------------------------------------------------------------------

/-- What a region can do, abstractly: every way it may finish, leave an
    enclosing loop or go round one again, with the state it does so in; or
    stop, and not by misuse. -/
inductive ARes where
  | ok (a : AS)
  | brk (depth : Nat) (a : AS) (vals : List AV)
  | cont (depth : Nat) (vals : List AV) (a : AS)
  | halt

/-- Follow every result, collecting what each leads to; refused if any is. -/
def bindAll (rs : List ARes) (g : ARes → Option (List ARes)) : Option (List ARes) :=
  rs.foldr (fun r acc => do let x ← g r; let y ← acc; pure (x ++ y)) (some [])

/-- The arm a branch takes on `f`, with its exports and the state it starts in. -/
def iteSel (a : AS) (thn els : List Piece) (thnR elsR : List R) (f : Bool) :
    List Piece × List R × AS :=
  if f then (thn, thnR, a) else (els, elsR, { a with env := abindAt a.env (slotsOf a.env.size thn) [] })

/-- A branch's join: an arm that finished binds its exports. -/
def iteDone (joinAt : Nat) (exports : List R) (rs : List ARes) : Option (List ARes) :=
  bindAll rs fun
    | .ok a' =>
        match exports.mapM (fun r => a'.env[r]?) with
        | none => some [.halt]
        | some vs => some [.ok { a' with env := abindAt a'.env joinAt vs }]
    | r => some [r]

/-- The test a bottom-tested loop reads: known, unknown (`some none`), or not a
    scalar in scope, where the run stops. -/
def aTest (E : AEnv) (r : R) : Option (Option Bool) :=
  match E[r]? with
  | some (.known (.sc _ f)) => some (some (f != 0))
  | some (.known (.vec _ _)) => none
  | some _ => some none
  | none => none

/-- Where a bottom-tested loop leaves: the carries its exit names. -/
def aLeave (l : DLoop) (afterBody : Nat) (a' : AS) (vs : List AV) : List ARes :=
  match l.exitIdx.mapM (fun i => vs[i]?) with
  | none => [.halt]
  | some outs => [.ok { a' with env := abindAt a'.env afterBody outs }]

mutual

def aPiece : Nat → AS → Piece → Option (List ARes)
  | 0, _, _ => none
  | _ + 1, a, .straight ss =>
      match aStmts a ss with
      | none => none
      | some .halt => some [.halt]
      | some (.go a') => some [.ok a']
  | _ + 1, a, .br depth args =>
      match args.mapM (fun r => a.env[r]?) with
      | none => some [.halt]
      | some vs => some [.brk depth a vs]
  | _ + 1, a, .cont depth args =>
      match args.mapM (fun r => a.env[r]?) with
      | none => some [.halt]
      | some vs => some [.cont depth vs a]
  | fuel + 1, a, .dloop l body =>
      match l.init.mapM (fun r => a.env[r]?) with
      | none => some [.halt]
      | some inits =>
          aDtrip fuel a l body a.env.size (slotsOf (a.env.size + l.pTys.length) body) inits
            l.guard.isSome
  | fuel + 1, a, .loop l pre body =>
      match l.init.mapM (fun r => a.env[r]?) with
      | none => some [.halt]
      | some inits =>
          aIter fuel a l pre body a.env.size
            (slotsOf (slotsOf (a.env.size + l.pTys.length) pre + l.pTys.length) body) inits
  | fuel + 1, a, .ite m thn els thnR elsR =>
      match a.env[m.flag]? with
      | some (.known (.sc _ f)) =>
          (aCode fuel (iteSel a thn els thnR elsR (f != 0)).2.2 (iteSel a thn els thnR elsR (f != 0)).1).bind
            (iteDone (slotsOf (slotsOf a.env.size thn) els) (iteSel a thn els thnR elsR (f != 0)).2.1)
      | some (.known (.vec _ _)) => some [.halt]
      | some _ => do
          let x ← (aCode fuel (iteSel a thn els thnR elsR true).2.2 (iteSel a thn els thnR elsR true).1).bind
            (iteDone (slotsOf (slotsOf a.env.size thn) els) (iteSel a thn els thnR elsR true).2.1)
          let y ← (aCode fuel (iteSel a thn els thnR elsR false).2.2 (iteSel a thn els thnR elsR false).1).bind
            (iteDone (slotsOf (slotsOf a.env.size thn) els) (iteSel a thn els thnR elsR false).2.1)
          pure (x ++ y)
      | none => some [.halt]

def aIter : Nat → AS → Loop → List Piece → List Piece → Nat → Nat → List AV → Option (List ARes)
  | 0, _, _, _, _, _, _, _ => none
  | fuel + 1, a, l, pre, body, n0, afterBody, carries =>
      (aCode fuel { a with env := abindAt a.env n0 carries } pre).bind fun rs => bindAll rs fun
        | .halt => some [.halt]
        | .brk 0 ab vs => some [.ok { ab with env := abindAt ab.env afterBody vs }]
        | .brk (d + 1) ab vs => some [.brk d ab vs]
        | .cont 0 vs a' => aIter fuel { a with w := a'.w, known := a'.known } l pre body n0 afterBody vs
        | .cont (d + 1) vs a' => some [.cont d vs a']
        | .ok a1 =>
            match a1.env[l.flag]? with
            | some (.known (.sc _ f)) =>
                if (f != 0) == l.exitOnTrue then
                  match l.exitR.mapM (fun r => a1.env[r]?) with
                  | none => some [.halt]
                  | some vs => some [.ok { a1 with env := abindAt a1.env afterBody vs }]
                else
                  (aCode fuel { a1 with env := abindAt a1.env (slotsOf (n0 + l.pTys.length) pre) carries }
                      body).bind fun rs2 => bindAll rs2 fun
                    | .halt => some [.halt]
                    | .brk 0 ab vs => some [.ok { ab with env := abindAt ab.env afterBody vs }]
                    | .brk (d + 1) ab vs => some [.brk d ab vs]
                    | .cont 0 vs a2 =>
                        aIter fuel { a with w := a2.w, known := a2.known } l pre body n0 afterBody vs
                    | .cont (d + 1) vs a2 => some [.cont d vs a2]
                    | .ok a2 =>
                        match l.cont.mapM (fun r => a2.env[r]?) with
                        | none => some [.halt]
                        | some next =>
                            aIter fuel { a with w := a2.w, known := a2.known } l pre body n0 afterBody next
            | some (.known (.vec _ _)) => some [.halt]
            | some _ => none
            | none => some [.halt]

def aDtrip : Nat → AS → DLoop → List Piece → Nat → Nat → List AV → Bool → Option (List ARes)
  | 0, _, _, _, _, _, _, _ => none
  | fuel + 1, a, l, body, n0, afterBody, carries, first =>
      if first then
        match l.guard with
        | none => aDtrip fuel a l body n0 afterBody carries false
        | some g =>
            match aTest a.env g with
            | none => some [.halt]
            | some none => none
            | some (some c) =>
                if c == l.contOnTrue then aDtrip fuel a l body n0 afterBody carries false
                else some (aLeave l afterBody a carries)
      else
        (aCode fuel { a with env := abindAt a.env n0 carries } body).bind fun rs => bindAll rs fun
          | .halt => some [.halt]
          | .brk 0 ab vs => some [.ok { ab with env := abindAt ab.env afterBody vs }]
          | .brk (d + 1) ab vs => some [.brk d ab vs]
          | .cont 0 vs a2 => aDtrip fuel { a with w := a2.w, known := a2.known } l body n0 afterBody vs false
          | .cont (d + 1) vs a' => some [.cont d vs a']
          | .ok a2 =>
              match l.cont.mapM (fun r => a2.env[r]?) with
              | none => some [.halt]
              | some next =>
                  match aTest a2.env l.flag with
                  | none => some [.halt]
                  | some none => none
                  | some (some c) =>
                      if c == l.contOnTrue then
                        aDtrip fuel { a with w := a2.w, known := a2.known } l body n0 afterBody next false
                      else some (aLeave l afterBody a2 next)

def aCode : Nat → AS → List Piece → Option (List ARes)
  | 0, _, _ => none
  | _ + 1, a, [] => some [.ok a]
  | fuel + 1, a, p :: ps =>
      (aPiece fuel a p).bind fun rs => bindAll rs fun
        | .ok a' => aCode fuel a' ps
        | r => some [r]

end

-- ---------------------------------------------------------------------------
-- Soundness
-- ---------------------------------------------------------------------------

/-- What a concrete result must be, given the abstract ones: one of them
    describes it, or the run stopped for a reason other than misuse. -/
def SimR (rs : List ARes) : CodeRes → Prop
  | .ok Γ w => ∃ a, ARes.ok a ∈ rs ∧ Rel a Γ w
  | .brk d Γb vs w => ∃ a avs, ARes.brk d a avs ∈ rs ∧ Rel a Γb w ∧ Rels avs vs
  | .cont d vs w => ∃ a avs, ARes.cont d avs a ∈ rs ∧ Same a.known w a.w ∧ Rels avs vs
  | .stuck _ => True
  | .misuse _ => False
  | .fault _ => True

theorem SimR.mono {rs rs' : List ARes} (h : ∀ x ∈ rs, x ∈ rs') : ∀ {r : CodeRes}, SimR rs r → SimR rs' r
  | .ok _ _, ⟨a, hm, hr⟩ => ⟨a, h _ hm, hr⟩
  | .brk _ _ _ _, ⟨a, avs, hm, hr, hv⟩ => ⟨a, avs, h _ hm, hr, hv⟩
  | .cont _ _ _, ⟨a, avs, hm, hr, hv⟩ => ⟨a, avs, h _ hm, hr, hv⟩
  | .stuck _, _ => trivial
  | .misuse _, hs => hs.elim
  | .fault _, _ => trivial

theorem bindAll_mem {g : ARes → Option (List ARes)} :
    ∀ {rs out : List ARes}, bindAll rs g = some out → ∀ r ∈ rs, ∃ o, g r = some o ∧ ∀ x ∈ o, x ∈ out
  | [], _, _, r, hr => by cases hr
  | r0 :: rs, out, h, r, hr => by
    simp only [bindAll, List.foldr_cons] at h
    change (do let x ← g r0; let y ← bindAll rs g; pure (x ++ y)) = some out at h
    cases h1 : g r0 with
    | none => rw [h1] at h; cases h
    | some x =>
      rw [h1] at h
      cases h2 : bindAll rs g with
      | none => rw [h2] at h; cases h
      | some y =>
        rw [h2] at h
        simp only [bind, Option.bind, pure, Option.some.injEq] at h
        subst h
        rcases List.mem_cons.mp hr with rfl | hr
        · exact ⟨x, h1, fun z hz => List.mem_append_left _ hz⟩
        · obtain ⟨o, ho, hsub⟩ := bindAll_mem h2 r hr
          exact ⟨o, ho, fun z hz => List.mem_append_right _ (hsub z hz)⟩

theorem Option.bind_eq_some' {α β : Type} {x : Option α} {f : α → Option β} {b : β}
    (h : x.bind f = some b) : ∃ a, x = some a ∧ f a = some b := by
  cases x with
  | none => cases h
  | some a => exact ⟨a, rfl, h⟩

/-- What `SoundAt` says at concrete fuel `k`, for the four functions at once,
    and every abstract fuel. -/
def SoundAt (k : Nat) : Prop :=
  (∀ (f : Nat) (a : AS) (Γ : Env) (w : World) (cfg : Cfg) (p : Piece) (rs : List ARes),
      aPiece f a p = some rs → Rel a Γ w → SimR rs (runPiece k cfg Γ w p)) ∧
  (∀ (f : Nat) (a : AS) (Γ : Env) (w : World) (cfg : Cfg) (c : List Piece) (rs : List ARes),
      aCode f a c = some rs → Rel a Γ w → SimR rs (runCode k cfg Γ w c)) ∧
  (∀ (f : Nat) (a : AS) (Γ : Env) (w : World) (cfg : Cfg) (l : Loop) (pre body : List Piece)
      (n0 afterBody : Nat) (acs : List AV) (cs : List V) (rs : List ARes),
      aIter f a l pre body n0 afterBody acs = some rs → Rel a Γ w → Rels acs cs →
      SimR rs (iter k cfg Γ w l pre body n0 afterBody cs)) ∧
  (∀ (f : Nat) (a : AS) (Γ : Env) (w : World) (cfg : Cfg) (l : DLoop) (body : List Piece)
      (n0 afterBody : Nat) (acs : List AV) (cs : List V) (first : Bool) (rs : List ARes),
      aDtrip f a l body n0 afterBody acs first = some rs → Rel a Γ w → Rels acs cs →
      SimR rs (dtrip k cfg Γ w l body n0 afterBody cs first))

theorem soundAt_zero : SoundAt 0 :=
  ⟨fun _ _ _ _ _ _ _ _ _ => by simp [runPiece, SimR],
   fun _ _ _ _ _ _ _ _ _ => by simp [runCode, SimR],
   fun _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ => by simp [iter, SimR],
   fun _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ => by simp [dtrip, SimR]⟩

/-- A result that leaves a region passes through it unchanged. -/
theorem pass_brk {rs out : List ARes} {g : ARes → Option (List ARes)}
    (hg : ∀ d b avs, g (.brk d b avs) = some [.brk d b avs])
    (h : bindAll rs g = some out) {d : Nat} {Γb : Env} {vs : List V} {w : World}
    (hs : SimR rs (.brk d Γb vs w)) : SimR out (.brk d Γb vs w) := by
  obtain ⟨b, avs, hm, hrb, hv⟩ := hs
  obtain ⟨o, ho, hsub⟩ := bindAll_mem h _ hm
  rw [hg] at ho; injection ho with ho; subst ho
  exact ⟨b, avs, hsub _ (List.mem_singleton_self _), hrb, hv⟩

theorem pass_cont {rs out : List ARes} {g : ARes → Option (List ARes)}
    (hg : ∀ d avs b, g (.cont d avs b) = some [.cont d avs b])
    (h : bindAll rs g = some out) {d : Nat} {vs : List V} {w : World}
    (hs : SimR rs (.cont d vs w)) : SimR out (.cont d vs w) := by
  obtain ⟨b, avs, hm, hrb, hv⟩ := hs
  obtain ⟨o, ho, hsub⟩ := bindAll_mem h _ hm
  rw [hg] at ho; injection ho with ho; subst ho
  exact ⟨b, avs, hsub _ (List.mem_singleton_self _), hrb, hv⟩

theorem code_step {k : Nat} (ih : SoundAt k) (f : Nat) (a : AS) (Γ : Env) (w : World) (cfg : Cfg)
    (c : List Piece) (rs : List ARes) (h : aCode f a c = some rs) (hr : Rel a Γ w) :
    SimR rs (runCode (k + 1) cfg Γ w c) := by
  cases f with
  | zero => simp [aCode] at h
  | succ f =>
    cases c with
    | nil =>
      simp only [aCode, Option.some.injEq] at h; subst h
      simp only [runCode]
      exact ⟨a, List.mem_singleton_self _, hr⟩
    | cons p ps =>
      simp only [aCode] at h
      obtain ⟨rs1, h1, h2⟩ := Option.bind_eq_some' h
      have hs := ih.1 f a Γ w cfg p rs1 h1 hr
      simp only [runCode]
      generalize runPiece k cfg Γ w p = r at hs ⊢
      cases r with
      | stuck => trivial
      | fault => trivial
      | misuse => exact hs.elim
      | brk d Γb vs w' => exact pass_brk (fun _ _ _ => rfl) h2 hs
      | cont d vs w' => exact pass_cont (fun _ _ _ => rfl) h2 hs
      | ok Γ' w' =>
        obtain ⟨b, hm, hrb⟩ := hs
        obtain ⟨o, ho, hsub⟩ := bindAll_mem h2 _ hm
        exact (ih.2.1 f b Γ' w' cfg ps o ho hrb).mono hsub

/-- A branch's join, as `runPiece` makes it. -/
def joinRes (joinAt : Nat) (exports : List R) : CodeRes → CodeRes
  | .stuck s => .stuck s
  | .misuse s => .misuse s
  | .fault s => .fault s
  | .brk d Γb vs w' => .brk d Γb vs w'
  | .cont d vs w' => .cont d vs w'
  | .ok Γ' w' =>
      match exports.mapM (Sem.get Γ') with
      | none => .stuck "branch export is not in scope"
      | some vs => .ok (bindAt Γ' joinAt vs) w'

/-- A branch's join, abstractly and concretely. -/
theorem join_sim {rs1 rs : List ARes} {r : CodeRes} (hs : SimR rs1 r) (joinAt : Nat) (exports : List R)
    (h : iteDone joinAt exports rs1 = some rs) : SimR rs (joinRes joinAt exports r) := by
  unfold iteDone at h
  cases r with
  | stuck => trivial
  | fault => trivial
  | misuse => exact hs.elim
  | brk d Γb vs w' => exact pass_brk (fun _ _ _ => rfl) h hs
  | cont d vs w' => exact pass_cont (fun _ _ _ => rfl) h hs
  | ok Γ' w' =>
    obtain ⟨b, hm, hrb⟩ := hs
    obtain ⟨o, ho, hsub⟩ := bindAll_mem h _ hm
    simp only at ho ⊢
    have e : exports.mapM (Sem.get Γ') = exports.mapM (fun r => Γ'[r]?) := rfl
    simp only [joinRes]
    rcases mapM_rel hrb.1 exports with ⟨h1, h2⟩ | ⟨avs, vs, h1, h2, h3⟩
    · rw [e, h2]; trivial
    · rw [e, h2]
      rw [h1] at ho; injection ho with ho; subst ho
      exact ⟨_, hsub _ (List.mem_singleton_self _), hrb.1.bindAt joinAt h3, hrb.2⟩

theorem piece_step {k : Nat} (ih : SoundAt k) (f : Nat) (a : AS) (Γ : Env) (w : World) (cfg : Cfg)
    (p : Piece) (rs : List ARes) (h : aPiece f a p = some rs) (hr : Rel a Γ w) :
    SimR rs (runPiece (k + 1) cfg Γ w p) := by
  cases f with
  | zero => simp [aPiece] at h
  | succ f =>
    have hsz : a.env.size = Γ.size := hr.1.1
    cases p with
    | straight ss =>
      simp only [aPiece] at h
      simp only [runPiece]
      cases ho : aStmts a ss with
      | none => rw [ho] at h; cases h
      | some o =>
        have hsim := aStmts_sound cfg ss hr ho
        rw [ho] at h
        cases hrun : runStmts cfg Γ w ss with
        | stuck => trivial
        | fault => trivial
        | misuse => rw [hrun] at hsim; exact hsim.elim
        | ok Γ' w' =>
          rw [hrun] at hsim
          obtain ⟨a', rfl, hr'⟩ := hsim
          simp only [Option.some.injEq] at h; subst h
          exact ⟨a', List.mem_singleton_self _, hr'⟩
    | br d args =>
      simp only [aPiece] at h
      simp only [runPiece]
      have e : args.mapM (Sem.get Γ) = args.mapM (fun r => Γ[r]?) := rfl
      rw [e]
      rcases mapM_rel hr.1 args with ⟨h1, h2⟩ | ⟨avs, vs, h1, h2, h3⟩
      · rw [h2]; trivial
      · rw [h2]; rw [h1] at h; injection h with h; subst h
        exact ⟨a, avs, List.mem_singleton_self _, hr, h3⟩
    | cont d args =>
      simp only [aPiece] at h
      simp only [runPiece]
      have e : args.mapM (Sem.get Γ) = args.mapM (fun r => Γ[r]?) := rfl
      rw [e]
      rcases mapM_rel hr.1 args with ⟨h1, h2⟩ | ⟨avs, vs, h1, h2, h3⟩
      · rw [h2]; trivial
      · rw [h2]; rw [h1] at h; injection h with h; subst h
        exact ⟨a, avs, List.mem_singleton_self _, hr.2, h3⟩
    | dloop l body =>
      simp only [aPiece] at h
      simp only [runPiece]
      have e : l.init.mapM (Sem.get Γ) = l.init.mapM (fun r => Γ[r]?) := rfl
      rw [e]
      rcases mapM_rel hr.1 l.init with ⟨h1, h2⟩ | ⟨avs, vs, h1, h2, h3⟩
      · rw [h2]; trivial
      · rw [h2]; rw [h1] at h
        simp only at h
        rw [hsz] at h
        exact ih.2.2.2 f a Γ w cfg l body _ _ avs vs _ rs h hr h3
    | loop l pre body =>
      simp only [aPiece] at h
      simp only [runPiece]
      have e : l.init.mapM (Sem.get Γ) = l.init.mapM (fun r => Γ[r]?) := rfl
      rw [e]
      rcases mapM_rel hr.1 l.init with ⟨h1, h2⟩ | ⟨avs, vs, h1, h2, h3⟩
      · rw [h2]; trivial
      · rw [h2]; rw [h1] at h
        simp only at h
        rw [hsz] at h
        exact ih.2.2.1 f a Γ w cfg l pre body _ _ avs vs rs h hr h3
    | ite m thn els thnR elsR =>
      simp only [aPiece] at h
      simp only [runPiece]
      -- one arm, taken on `b`
      have arm : ∀ (b : Bool) (rs' : List ARes),
          (aCode f (iteSel a thn els thnR elsR b).2.2 (iteSel a thn els thnR elsR b).1).bind
            (iteDone (slotsOf (slotsOf a.env.size thn) els) (iteSel a thn els thnR elsR b).2.1) = some rs' →
          SimR rs' (joinRes (slotsOf (slotsOf Γ.size thn) els) (if b then thnR else elsR)
            (runCode k cfg (if b then Γ else bindAt Γ (slotsOf Γ.size thn) []) w (if b then thn else els))) := by
        intro b rs' h'
        obtain ⟨rs1, h1, h2⟩ := Option.bind_eq_some' h'
        rw [hsz] at h2
        cases b
        · simp only [iteSel, Bool.false_eq_true, if_false] at h1 h2 ⊢
          have hr0 : Rel { a with env := abindAt a.env (slotsOf a.env.size thn) [] }
              (bindAt Γ (slotsOf Γ.size thn) []) w := by
            rw [hsz]; exact ⟨hr.1.bindAt _ trivial, hr.2⟩
          exact join_sim (ih.2.1 f _ _ w cfg els rs1 h1 hr0) _ _ h2
        · simp only [iteSel, if_true] at h1 h2 ⊢
          exact join_sim (ih.2.1 f a Γ w cfg thn rs1 h1 hr) _ _ h2
      rcases hr.1.get m.flag with ⟨h1, h2⟩ | ⟨av, v, h1, h2, h3⟩
      · simp only [Sem.get, h2]; trivial
      rw [h1] at h
      cases v with
      | vec t ls => simp only [Sem.get, h2]; trivial
      | sc t fv =>
        simp only [Sem.get, h2]
        -- the arm the run takes is one the checker followed
        have taken : ∀ (b : Bool), (fv != 0) = b → ∀ rs', (aCode f (iteSel a thn els thnR elsR b).2.2
              (iteSel a thn els thnR elsR b).1).bind
              (iteDone (slotsOf (slotsOf a.env.size thn) els) (iteSel a thn els thnR elsR b).2.1)
              = some rs' → (∀ x ∈ rs', x ∈ rs) →
            SimR rs (joinRes (slotsOf (slotsOf Γ.size thn) els) (if b then thnR else elsR)
              (runCode k cfg (if b then Γ else bindAt Γ (slotsOf Γ.size thn) []) w (if b then thn else els))) :=
          fun b _ rs' h' hsub => (arm b rs' h').mono hsub
        have hgoal : ∀ (b : Bool), (fv != 0) = b →
            SimR rs (joinRes (slotsOf (slotsOf Γ.size thn) els) (if b then thnR else elsR)
              (runCode k cfg (if b then Γ else bindAt Γ (slotsOf Γ.size thn) []) w (if b then thn else els))) := by
          intro b hb
          cases av with
          | known x =>
            simp only [AV.Rel] at h3; subst h3
            simp only at h
            rw [hb] at h
            exact taken b hb rs h (fun _ hx => hx)
          | scalar =>
            simp only [bind, Option.bind] at h
            obtain ⟨x, hx, h'⟩ := Option.bind_eq_some' h
            obtain ⟨y, hy, h''⟩ := Option.bind_eq_some' h'
            simp only [pure, Option.some.injEq] at h''; subst h''
            cases b
            · exact taken false hb y hy (fun _ hz => List.mem_append_right _ hz)
            · exact taken true hb x hx (fun _ hz => List.mem_append_left _ hz)
          | top =>
            simp only [bind, Option.bind] at h
            obtain ⟨x, hx, h'⟩ := Option.bind_eq_some' h
            obtain ⟨y, hy, h''⟩ := Option.bind_eq_some' h'
            simp only [pure, Option.some.injEq] at h''; subst h''
            cases b
            · exact taken false hb y hy (fun _ hz => List.mem_append_right _ hz)
            · exact taken true hb x hx (fun _ hz => List.mem_append_left _ hz)
        cases hb : (fv != 0)
        · have := hgoal false hb
          simp only [Bool.false_eq_true, if_false] at this ⊢
          exact this
        · have := hgoal true hb
          simp only [if_true] at this ⊢
          exact this

theorem iter_step {k : Nat} (ih : SoundAt k) (f : Nat) (a : AS) (Γ : Env) (w : World) (cfg : Cfg)
    (l : Loop) (pre body : List Piece) (n0 afterBody : Nat) (acs : List AV) (cs : List V) (rs : List ARes)
    (h : aIter f a l pre body n0 afterBody acs = some rs) (hr : Rel a Γ w) (hc : Rels acs cs) :
    SimR rs (iter (k + 1) cfg Γ w l pre body n0 afterBody cs) := by
  cases f with
  | zero => simp [aIter] at h
  | succ f =>
    simp only [aIter] at h
    obtain ⟨rs1, h1, h2⟩ := Option.bind_eq_some' h
    have hr0 : Rel { a with env := abindAt a.env n0 acs } (bindAt Γ n0 cs) w := ⟨hr.1.bindAt n0 hc, hr.2⟩
    have hs := ih.2.1 f _ _ w cfg pre rs1 h1 hr0
    simp only [iter]
    generalize runCode k cfg (bindAt Γ n0 cs) w pre = r at hs ⊢
    cases r with
    | stuck => trivial
    | fault => trivial
    | misuse => exact hs.elim
    | brk d Γb vs w' =>
      obtain ⟨b, avs, hm, hrb, hv⟩ := hs
      obtain ⟨o, ho, hsub⟩ := bindAll_mem h2 _ hm
      cases d with
      | zero =>
        simp only at ho; injection ho with ho; subst ho
        exact ⟨_, hsub _ (List.mem_singleton_self _), hrb.1.bindAt afterBody hv, hrb.2⟩
      | succ d =>
        simp only at ho; injection ho with ho; subst ho
        exact ⟨b, avs, hsub _ (List.mem_singleton_self _), hrb, hv⟩
    | cont d vs w' =>
      obtain ⟨b, avs, hm, hrb, hv⟩ := hs
      obtain ⟨o, ho, hsub⟩ := bindAll_mem h2 _ hm
      cases d with
      | zero =>
        simp only at ho
        exact (ih.2.2.1 f _ Γ w' cfg l pre body n0 afterBody avs vs o ho ⟨hr.1, hrb⟩ hv).mono hsub
      | succ d =>
        simp only at ho; injection ho with ho; subst ho
        exact ⟨b, avs, hsub _ (List.mem_singleton_self _), hrb, hv⟩
    | ok Γ1 w1 =>
      obtain ⟨b, hm, hrb⟩ := hs
      obtain ⟨o, ho, hsub⟩ := bindAll_mem h2 _ hm
      simp only at ho
      rcases hrb.1.get l.flag with ⟨g1, g2⟩ | ⟨av, v, g1, g2, g3⟩
      · simp only [Sem.get, g2]; trivial
      rw [g1] at ho
      cases v with
      | vec t ls => simp only [Sem.get, g2]; trivial
      | sc t fv =>
        simp only [Sem.get, g2]
        cases av with
        | scalar => cases ho
        | top => cases ho
        | known x =>
          simp only [AV.Rel] at g3; subst g3
          simp only at ho
          split
          · rename_i hx
            rw [if_pos hx] at ho
            have e : l.exitR.mapM (Sem.get Γ1) = l.exitR.mapM (fun r => Γ1[r]?) := rfl
            rw [e]
            rcases mapM_rel hrb.1 l.exitR with ⟨e1, e2⟩ | ⟨avs, vs, e1, e2, e3⟩
            · rw [e2]; trivial
            · rw [e2]; rw [e1] at ho; injection ho with ho; subst ho
              exact ⟨_, hsub _ (List.mem_singleton_self _), hrb.1.bindAt afterBody e3, hrb.2⟩
          · rename_i hx
            rw [if_neg hx] at ho
            obtain ⟨rs2, h3, h4⟩ := Option.bind_eq_some' ho
            have hr1 : Rel { b with env := abindAt b.env (slotsOf (n0 + l.pTys.length) pre) acs }
                (bindAt Γ1 (slotsOf (n0 + l.pTys.length) pre) cs) w1 := ⟨hrb.1.bindAt _ hc, hrb.2⟩
            have hs2 := ih.2.1 f _ _ w1 cfg body rs2 h3 hr1
            generalize runCode k cfg (bindAt Γ1 (slotsOf (n0 + l.pTys.length) pre) cs) w1 body = r2 at hs2 ⊢
            cases r2 with
            | stuck => trivial
            | fault => trivial
            | misuse => exact hs2.elim
            | brk d Γb vs w2 =>
              obtain ⟨b2, avs, hm2, hrb2, hv⟩ := hs2
              obtain ⟨o2, ho2, hsub2⟩ := bindAll_mem h4 _ hm2
              cases d with
              | zero =>
                simp only at ho2; injection ho2 with ho2; subst ho2
                exact ⟨_, hsub _ (hsub2 _ (List.mem_singleton_self _)), hrb2.1.bindAt afterBody hv, hrb2.2⟩
              | succ d =>
                simp only at ho2; injection ho2 with ho2; subst ho2
                exact ⟨b2, avs, hsub _ (hsub2 _ (List.mem_singleton_self _)), hrb2, hv⟩
            | cont d vs w2 =>
              obtain ⟨b2, avs, hm2, hrb2, hv⟩ := hs2
              obtain ⟨o2, ho2, hsub2⟩ := bindAll_mem h4 _ hm2
              cases d with
              | zero =>
                simp only at ho2
                exact (ih.2.2.1 f _ Γ w2 cfg l pre body n0 afterBody avs vs o2 ho2 ⟨hr.1, hrb2⟩ hv).mono
                  (fun x hx => hsub _ (hsub2 _ hx))
              | succ d =>
                simp only at ho2; injection ho2 with ho2; subst ho2
                exact ⟨b2, avs, hsub _ (hsub2 _ (List.mem_singleton_self _)), hrb2, hv⟩
            | ok Γ2 w2 =>
              obtain ⟨b2, hm2, hrb2⟩ := hs2
              obtain ⟨o2, ho2, hsub2⟩ := bindAll_mem h4 _ hm2
              simp only at ho2 ⊢
              have e : l.cont.mapM (Sem.get Γ2) = l.cont.mapM (fun r => Γ2[r]?) := rfl
              rw [e]
              rcases mapM_rel hrb2.1 l.cont with ⟨e1, e2⟩ | ⟨avs, vs, e1, e2, e3⟩
              · rw [e2]; trivial
              · rw [e2]; rw [e1] at ho2
                simp only at ho2
                exact (ih.2.2.1 f _ Γ w2 cfg l pre body n0 afterBody avs vs o2 ho2 ⟨hr.1, hrb2.2⟩ e3).mono
                  (fun x hx => hsub _ (hsub2 _ hx))

theorem rels_mapIdx {avs : List AV} {vs : List V} (h : Rels avs vs) :
    ∀ (idx : List Nat), (idx.mapM (fun i => avs[i]?) = none ∧ idx.mapM (fun i => vs[i]?) = none) ∨
      ∃ aos os, idx.mapM (fun i => avs[i]?) = some aos ∧ idx.mapM (fun i => vs[i]?) = some os ∧ Rels aos os
  | [] => Or.inr ⟨[], [], rfl, rfl, trivial⟩
  | i :: is => by
    simp only [List.mapM_cons, bind, Option.bind]
    have hl := h.length_eq
    by_cases hi : i < vs.length
    · have ha : avs[i]? = some avs[i] := by simp [hl ▸ hi]
      have hv : vs[i]? = some vs[i] := by simp [hi]
      rw [ha, hv]
      rcases rels_mapIdx h is with ⟨h1, h2⟩ | ⟨aos, os, h1, h2, h3⟩
      · left; rw [h1, h2]; exact ⟨rfl, rfl⟩
      · right; rw [h1, h2]
        exact ⟨_, _, rfl, rfl, h.get? i _ _ ha hv, h3⟩
    · have ha : avs[i]? = none := by simp [hl]; omega
      have hv : vs[i]? = none := by simp; omega
      left; rw [ha, hv]; exact ⟨rfl, rfl⟩

theorem leave_sim {a' : AS} {Γ' : Env} {w' : World} (hr : Rel a' Γ' w') {avs : List AV} {vs : List V}
    (hv : Rels avs vs) (l : DLoop) (afterBody : Nat) :
    SimR (aLeave l afterBody a' avs) (match l.exitIdx.mapM (fun i => vs[i]?) with
      | none => .stuck "loop exit value is not a carry"
      | some outs => .ok (bindAt Γ' afterBody outs) w') := by
  unfold aLeave
  rcases rels_mapIdx hv l.exitIdx with ⟨h1, h2⟩ | ⟨aos, os, h1, h2, h3⟩
  · rw [h2]; trivial
  · rw [h1, h2]
    exact ⟨_, List.mem_singleton_self _, hr.1.bindAt afterBody h3, hr.2⟩

theorem dtrip_step {k : Nat} (ih : SoundAt k) (f : Nat) (a : AS) (Γ : Env) (w : World) (cfg : Cfg)
    (l : DLoop) (body : List Piece) (n0 afterBody : Nat) (acs : List AV) (cs : List V) (first : Bool)
    (rs : List ARes) (h : aDtrip f a l body n0 afterBody acs first = some rs) (hr : Rel a Γ w)
    (hc : Rels acs cs) : SimR rs (dtrip (k + 1) cfg Γ w l body n0 afterBody cs first) := by
  cases f with
  | zero => simp [aDtrip] at h
  | succ f =>
    simp only [aDtrip] at h
    simp only [dtrip]
    cases first with
    | true =>
      simp only [if_true] at h ⊢
      cases hg : l.guard with
      | none =>
        rw [hg] at h; simp only at h ⊢
        exact ih.2.2.2 f a Γ w cfg l body n0 afterBody acs cs false rs h hr hc
      | some g =>
        rw [hg] at h; simp only at h ⊢
        unfold aTest at h
        rcases hr.1.get g with ⟨g1, g2⟩ | ⟨av, v, g1, g2, g3⟩
        · simp only [Sem.get, g2]; trivial
        rw [g1] at h
        cases v with
        | vec t ls => simp only [Sem.get, g2]; trivial
        | sc t fv =>
          simp only [Sem.get, g2]
          cases av with
          | scalar => cases h
          | top => cases h
          | known x =>
            simp only [AV.Rel] at g3; subst g3
            simp only at h
            split
            · rename_i hx
              rw [if_pos hx] at h
              exact ih.2.2.2 f a Γ w cfg l body n0 afterBody acs cs false rs h hr hc
            · rename_i hx
              rw [if_neg hx] at h
              injection h with h; subst h
              exact leave_sim hr hc l afterBody
    | false =>
      simp only [Bool.false_eq_true, if_false] at h ⊢
      obtain ⟨rs1, h1, h2⟩ := Option.bind_eq_some' h
      have hr0 : Rel { a with env := abindAt a.env n0 acs } (bindAt Γ n0 cs) w := ⟨hr.1.bindAt n0 hc, hr.2⟩
      have hs := ih.2.1 f _ _ w cfg body rs1 h1 hr0
      generalize runCode k cfg (bindAt Γ n0 cs) w body = r at hs ⊢
      cases r with
      | stuck => trivial
      | fault => trivial
      | misuse => exact hs.elim
      | brk d Γb vs w' =>
        obtain ⟨b, avs, hm, hrb, hv⟩ := hs
        obtain ⟨o, ho, hsub⟩ := bindAll_mem h2 _ hm
        cases d with
        | zero =>
          simp only at ho; injection ho with ho; subst ho
          exact ⟨_, hsub _ (List.mem_singleton_self _), hrb.1.bindAt afterBody hv, hrb.2⟩
        | succ d =>
          simp only at ho; injection ho with ho; subst ho
          exact ⟨b, avs, hsub _ (List.mem_singleton_self _), hrb, hv⟩
      | cont d vs w' =>
        obtain ⟨b, avs, hm, hrb, hv⟩ := hs
        obtain ⟨o, ho, hsub⟩ := bindAll_mem h2 _ hm
        cases d with
        | zero =>
          simp only at ho
          exact (ih.2.2.2 f _ Γ w' cfg l body n0 afterBody avs vs false o ho ⟨hr.1, hrb⟩ hv).mono hsub
        | succ d =>
          simp only at ho; injection ho with ho; subst ho
          exact ⟨b, avs, hsub _ (List.mem_singleton_self _), hrb, hv⟩
      | ok Γ2 w2 =>
        obtain ⟨b, hm, hrb⟩ := hs
        obtain ⟨o, ho, hsub⟩ := bindAll_mem h2 _ hm
        simp only at ho ⊢
        have e : l.cont.mapM (Sem.get Γ2) = l.cont.mapM (fun r => Γ2[r]?) := rfl
        rw [e]
        rcases mapM_rel hrb.1 l.cont with ⟨e1, e2⟩ | ⟨avs, vs, e1, e2, e3⟩
        · rw [e2]; trivial
        rw [e2]; rw [e1] at ho
        simp only at ho ⊢
        unfold aTest at ho
        rcases hrb.1.get l.flag with ⟨g1, g2⟩ | ⟨av, v, g1, g2, g3⟩
        · simp only [Sem.get, g2]; trivial
        rw [g1] at ho
        cases v with
        | vec t ls => simp only [Sem.get, g2]; trivial
        | sc t fv =>
          simp only [Sem.get, g2]
          cases av with
          | scalar => cases ho
          | top => cases ho
          | known x =>
            simp only [AV.Rel] at g3; subst g3
            simp only at ho
            split
            · rename_i hx
              rw [if_pos hx] at ho
              exact (ih.2.2.2 f _ Γ w2 cfg l body n0 afterBody avs vs false o ho ⟨hr.1, hrb.2⟩ e3).mono hsub
            · rename_i hx
              rw [if_neg hx] at ho
              injection ho with ho; subst ho
              exact (leave_sim hrb e3 l afterBody).mono hsub

/-- **The checker describes every run**, at every budget. -/
theorem soundAt : ∀ k, SoundAt k
  | 0 => soundAt_zero
  | k + 1 =>
    have ih := soundAt k
    ⟨piece_step ih, code_step ih, iter_step ih, dtrip_step ih⟩

-- ---------------------------------------------------------------------------
-- The check, and what it guarantees
-- ---------------------------------------------------------------------------

/-- **`effectsOk`**: the checker follows the term from `a`, within `fuel`,
    without refusing anything. -/
def effectsOk (fuel : Nat) (a : AS) (c : Code) : Bool := (aCode fuel a c).isSome

/-- **Progress**: a term `effectsOk` accepts never ends in `misuse` from a state
    the abstract one describes, at any budget. -/
theorem progress {fuel : Nat} {a : AS} {c : Code} (h : effectsOk fuel a c = true) {Γ : Env}
    {w : World} (hr : Rel a Γ w) (k : Nat) (cfg : Cfg) (m : String) :
    runCode k cfg Γ w c ≠ .misuse m := by
  obtain ⟨rs, hrs⟩ := Option.isSome_iff_exists.mp h
  intro he
  have := (soundAt k).2.1 fuel a Γ w cfg c rs hrs hr
  rw [he] at this
  exact this

/-- The state an entry point starts in, abstractly: its arguments known, the
    canonical world, and the host bytes every world it runs in shares with it. -/
def entry (args : List V) (w₀ : World) (K : Known) : AS :=
  { env := args.toArray.map .known, w := w₀, known := K }

theorem entry_rel (args : List V) {w₀ w : World} {K : Known} (hs : Same K w w₀) :
    Rel (entry args w₀ K) args.toArray w := by
  refine ⟨⟨by simp [entry], fun i b v hb hv => ?_⟩, hs⟩
  simp only [entry, Array.getElem?_map] at hb
  rw [hv] at hb
  injection hb with hb; subst hb; rfl

/-- **No run of an accepted body misuses a foreign call**: from any world that
    agrees with the canonical one up to data --- whatever the input bytes, the
    device buffers' contents and the kernels (so long as they keep lengths) ---
    `Sem.run` does not end in `misuse`. -/
theorem run_progress {fuel : Nat} {args : List V} {w₀ : World} {K : Known} {c : Code}
    (h : effectsOk fuel (entry args w₀ K) c = true) {w : World} (hs : Same K w w₀) (cfg : Cfg)
    (m : String) : Sem.run cfg args w c ≠ .misuse m := by
  have hp := progress h (entry_rel args hs) cfg.steps cfg m
  unfold Sem.run
  generalize runCode cfg.steps cfg args.toArray w c = r at hp
  cases r with
  | stuck => simp
  | fault => simp
  | misuse m' => intro he; simp only [Outcome.misuse.injEq] at he; subst he; exact hp rfl
  | brk => simp
  | cont => simp
  | ok => simp

end AlgorithmLib.HProg.Static
