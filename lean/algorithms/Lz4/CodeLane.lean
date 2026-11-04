module
public import Lz4.CodeKernel
public import AlgorithmLib.Proof.EndsRules
public import AlgorithmLib.Proof.EndsCont
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lz4.CodeLane` — one lane runs the warp's statement

What the code `Lz4Code.codeOf` writes for a warp statement does in one lane:
the lane's run of the statement (`laneRun`), its registers the warp's, its
memory the world's. A register's value is a slot of the code's environment
(`EnvOk`): a value register's an `i64`, a flag's a predicate, one the code
has not yet bound zero. A cooperative statement is the lanes' share of it,
given separately (`coop`).
-/

namespace Lz4Code

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Prog
  AlgorithmLib.LZ4WarpDSL AlgorithmLib.LibSpec

/-- A lane's state: the warp's registers, and the world. -/
structure LSt where
  regs : String → UInt64
  w : World

def LSt.set (s : LSt) (d : String) (v : UInt64) : LSt :=
  { s with regs := fun x => if x = d then v else s.regs x }

/-- A store of `n` bytes of `v` at `a`, as a store statement makes it. -/
def stw (w : World) (a : UInt64) (n : Nat) (v : UInt64) : Option World := do
  let m ← w.mem.store a n v
  some { obsStore w a n v with mem := m }

def wargL (s : LSt) : WArg → UInt64
  | .reg n => s.regs n
  | .imm v => UInt64.ofNat v

/-- **One lane's run of a warp statement**, `coop` giving a cooperative
    statement's: none where an access reaches nothing or a loop outruns the
    fuel. -/
def laneRun (coop : WStmt → LSt → Option LSt) (sm : UInt64) : Nat → WStmt → LSt → Option LSt
  | _, .skip, s => some s
  | f, .seq a b, s => do laneRun coop sm f b (← laneRun coop sm f a s)
  | _, .bin o d a b, s => some (s.set d (o.run (s.regs a) (wargL s b)))
  | _, .setp c d a b, s => some (s.set d (if c.run (s.regs a) (wargL s b) then 1 else 0))
  | _, .mov d a, s => some (s.set d (wargL s a))
  | _, .stgB addr val, s => do
      let w ← stw s.w (s.regs addr) 1 (s.regs val &&& 0xff)
      some { s with w }
  | f, .uif cond t e, s =>
      if s.regs cond == 1 then laneRun coop sm f t s else laneRun coop sm f e s
  | 0, .uwhile cond _, s => if s.regs cond == 1 then none else some s
  | f + 1, .uwhile cond body, s =>
      if s.regs cond == 1 then do laneRun coop sm f (.uwhile cond body) (← laneRun coop sm f body s)
      else some s
  | _, st@(.coopCopy ..), s | _, st@(.coopWindow ..), s | _, st@(.coopExtendStep ..), s => coop st s

/-- **The code's environment holds the lane's registers**: a value register
    its value, a flag its predicate, one not bound zero. -/
structure EnvOk (fl : List String) (e : RegEnv Slot) (regs : String → UInt64) (Γ : Env) : Prop where
  val : ∀ r v, e.vals.lookup r = some v → Γ[v]? = some (.sc .i64 (regs r))
  flag : ∀ r v, e.flags.lookup r = some v → Γ[v]? = some (.sc .i8 (regs r))
  bit : ∀ r, fl.contains r → regs r ≤ 1
  flag0 : ∀ r, fl.contains r → e.flags.lookup r = none → regs r = 0
  val0 : ∀ r, fl.contains r = false → e.vals.lookup r = none → regs r = 0
  fkind : ∀ r v, e.flags.lookup r = some v → fl.contains r = true
  vkind : ∀ r v, e.vals.lookup r = some v → fl.contains r = false

theorem EnvOk.congr {fl : List String} {e : RegEnv Slot} {regs regs' : String → UInt64} {Γ : Env}
    (h : EnvOk fl e regs Γ) (he : ∀ x, regs x = regs' x) : EnvOk fl e regs' Γ := by
  have : regs = regs' := funext he
  subst this; exact h

theorem EnvOk.extend {fl : List String} {e : RegEnv Slot} {regs : String → UInt64} {Γ Γ' : Env}
    (h : EnvOk fl e regs Γ) (hx : Extends Γ Γ') : EnvOk fl e regs Γ' :=
  ⟨fun r v hv => hx.get (h.val r v hv), fun r v hv => hx.get (h.flag r v hv), h.bit, h.flag0, h.val0,
    h.fkind, h.vkind⟩

theorem lookup_cons_ne {α : Type} {r d : String} {v : α} {l : List (String × α)} (h : r ≠ d) :
    ((d, v) :: l).lookup r = l.lookup r := by
  simp [List.lookup, show (r == d) = false by simpa using h]

theorem lookup_cons_eq {α : Type} {d : String} {v : α} {l : List (String × α)} :
    ((d, v) :: l).lookup d = some v := by simp [List.lookup]

/-- A value register bound to a slot holding its new value. -/
theorem EnvOk.setVal {fl : List String} {e : RegEnv Slot} {regs : String → UInt64} {Γ : Env}
    (h : EnvOk fl e regs Γ) (d : String) (hd : fl.contains d = false) (v : Slot .i64) (x : UInt64)
    (hv : Γ[v]? = some (.sc .i64 x)) :
    EnvOk fl (e.setVal d v) (fun r => if r = d then x else regs r) Γ where
  val r u hu := by
    simp only [RegEnv.setVal] at hu
    by_cases hrd : r = d
    · subst hrd; rw [lookup_cons_eq] at hu; cases hu; simpa using hv
    · rw [lookup_cons_ne hrd] at hu; rw [if_neg hrd]; exact h.val r u hu
  flag r u hu := by
    simp only [RegEnv.setVal] at hu
    have hrd : r ≠ d := fun e => by subst e; have := h.fkind r u hu; rw [hd] at this; cases this
    rw [if_neg hrd]; exact h.flag r u hu
  bit r hr := by
    have hrd : r ≠ d := fun e => by subst e; rw [hd] at hr; cases hr
    rw [if_neg hrd]; exact h.bit r hr
  flag0 r hr hn := by
    simp only [RegEnv.setVal] at hn
    have hrd : r ≠ d := fun e => by subst e; rw [hd] at hr; cases hr
    rw [if_neg hrd]; exact h.flag0 r hr hn
  val0 r hr hn := by
    simp only [RegEnv.setVal] at hn
    by_cases hrd : r = d
    · subst hrd; rw [lookup_cons_eq] at hn; cases hn
    · rw [lookup_cons_ne hrd] at hn; rw [if_neg hrd]; exact h.val0 r hr hn
  fkind r u hu := by simp only [RegEnv.setVal] at hu; exact h.fkind r u hu
  vkind r u hu := by
    simp only [RegEnv.setVal] at hu
    by_cases hrd : r = d
    · subst hrd; exact hd
    · rw [lookup_cons_ne hrd] at hu; exact h.vkind r u hu

/-- A flag bound to a slot holding its new predicate. -/
theorem EnvOk.setFlag {fl : List String} {e : RegEnv Slot} {regs : String → UInt64} {Γ : Env}
    (h : EnvOk fl e regs Γ) (d : String) (hd : fl.contains d = true) (v : Slot .i8) (x : UInt64)
    (hx : x ≤ 1) (hv : Γ[v]? = some (.sc .i8 x)) :
    EnvOk fl (e.setFlag d v) (fun r => if r = d then x else regs r) Γ where
  val r u hu := by
    simp only [RegEnv.setFlag] at hu
    have hrd : r ≠ d := fun e => by subst e; have := h.vkind r u hu; rw [hd] at this; cases this
    rw [if_neg hrd]; exact h.val r u hu
  flag r u hu := by
    simp only [RegEnv.setFlag] at hu
    by_cases hrd : r = d
    · subst hrd; rw [lookup_cons_eq] at hu; cases hu; simpa using hv
    · rw [lookup_cons_ne hrd] at hu; rw [if_neg hrd]; exact h.flag r u hu
  bit r hr := by
    by_cases hrd : r = d
    · subst hrd; simpa using hx
    · rw [if_neg hrd]; exact h.bit r hr
  flag0 r hr hn := by
    simp only [RegEnv.setFlag] at hn
    by_cases hrd : r = d
    · subst hrd; rw [lookup_cons_eq] at hn; cases hn
    · rw [lookup_cons_ne hrd] at hn; rw [if_neg hrd]; exact h.flag0 r hr hn
  val0 r hr hn := by
    simp only [RegEnv.setFlag] at hn
    have hrd : r ≠ d := fun e => by subst e; rw [hd] at hr; cases hr
    rw [if_neg hrd]; exact h.val0 r hr hn
  fkind r u hu := by
    simp only [RegEnv.setFlag] at hu
    by_cases hrd : r = d
    · subst hrd; exact hd
    · rw [lookup_cons_ne hrd] at hu; exact h.fkind r u hu
  vkind r u hu := by simp only [RegEnv.setFlag] at hu; exact h.vkind r u hu

theorem get_push_last (Γ : Env) (v : V) : (Γ.push v)[Γ.size]? = some v := by simp

theorem bit01 {x : UInt64} (h : x ≤ 1) : x = 0 ∨ x = 1 := by
  rcases Nat.lt_or_ge x.toNat 1 with h1 | h1
  · left; apply UInt64.toNat_inj.mp; simp; omega
  · right; apply UInt64.toNat_inj.mp; have := UInt64.le_iff_toNat_le.mp h; simp at this ⊢; omega

theorem shl_mod64 (x y : UInt64) : x <<< (y % 64) = x <<< y := by
  apply UInt64.toNat_inj.mp
  simp [UInt64.toNat_shiftLeft]

theorem shr_mod64 (x y : UInt64) : x >>> (y % 64) = x >>> y := by
  apply UInt64.toNat_inj.mp
  simp [UInt64.toNat_shiftRight]

theorem min_wide (x y : UInt64) :
    UInt64.ofNat (min (x.toNat &&& (widthMask ClifTy.i64).toNat) (y.toNat &&& (widthMask ClifTy.i64).toNat)) =
      if x ≤ y then x else y := by
  have hm : ∀ z : UInt64, z.toNat &&& (widthMask ClifTy.i64).toNat = z.toNat := fun z => by
    rw [← UInt64.toNat_and, and_wide64]
  rw [hm, hm]
  split
  · rename_i h; rw [Nat.min_eq_left (UInt64.le_iff_toNat_le.mp h), UInt64.ofNat_toNat]
  · rename_i h; rw [Nat.min_eq_right (by have := UInt64.lt_iff_toNat_lt.mp (UInt64.not_le.mp h); omega),
      UInt64.ofNat_toNat]

/-- A flag's 1 or 0, chosen between two constants pushed after it. -/
theorem eval_sel01 (m : Mem) (Γ0 : Env) (f : Slot .i8) (b : UInt64) (hb : b ≤ 1)
    (hf : Γ0[f]? = some (.sc .i8 b)) :
    evalOp m ((Γ0.push (.sc .i64 (c64 1))).push (.sc .i64 (c64 0)))
      (.select f Γ0.size (Γ0.push (.sc .i64 (c64 1))).size) = some (.sc .i64 b) := by
  have hf' := ((Extends.push Γ0 (.sc .i64 (c64 1))).push_of (.sc .i64 (c64 0))).get hf
  have h1 := (Extends.push (Γ0.push (.sc .i64 (c64 1))) (.sc .i64 (c64 0))).get (get_push_last Γ0 (.sc .i64 (c64 1)))
  have h0 := get_push_last (Γ0.push (.sc .i64 (c64 1))) (.sc .i64 (c64 0))
  simp only [evalOp, Sem.get, hf', h1, h0]
  rcases bit01 hb with rfl | rfl <;> rfl


section
variable {cfg : Cfg}

/-- **A flag's predicate**: the test `uif` and `uwhile` make of a register,
    whether it is one, in a slot of an environment keeping `Γ`. -/
theorem wpE_getFlag {fl : List String} {e : RegEnv Slot} {regs : String → UInt64} {Γ : Env} {w : World}
    {r : String} {Q : Slot .i8 → Env → World → Prop} (hE : EnvOk fl e regs Γ)
    (h : ∀ v Γ', Extends Γ Γ' → Γ'[v]? = some (boolV (regs r == 1)) → Q v Γ' w) :
    wpE cfg (e.getFlag fl r) Q Γ w := by
  unfold RegEnv.getFlag
  by_cases hfl : fl.contains r = true
  · rw [if_pos hfl]
    have hbool : boolV (regs r == 1) = .sc .i8 (regs r) := by
      rcases bit01 (hE.bit r hfl) with h0 | h0 <;> rw [h0] <;> rfl
    cases hl : e.flags.lookup r with
    | some v => exact wpE_pure (h v Γ (Extends.refl _) (by rw [hbool]; exact hE.flag r v hl))
    | none =>
        have h0 := hE.flag0 r hfl hl
        refine wpE_val (v := .sc .i8 0) (by simp [Op'.erase, evalOp, ofInt]; rfl) ?_
        exact h _ _ (Extends.push _ _) (by rw [hbool, h0]; simp)
  · rw [if_neg hfl]
    have hcmp : ∀ v Γ', Extends Γ Γ' → Γ'[v]? = some (.sc .i64 (regs r)) →
        wpE cfg (do icmp .eq v (← iconst64 1)) Q Γ' w := by
      intro v Γ' hx hv
      refine wpE_bind (wpE_iconst64 ?_)
      have hv' : (Γ'.push (.sc .i64 (c64 1)))[v]? = some (.sc .i64 (regs r)) := (Extends.push _ _).get hv
      refine wpE_val (v := boolV (regs r == 1)) ?_ ?_
      · simp [Op'.erase, evalOp, Sem.get, hv', zipIntCmp, cmpInt, ClifTy.isInt]; rfl
      · exact h _ _ ((hx.push_of _).push_of _) (get_push_last _ _)
    cases hl : e.vals.lookup r with
    | some v => exact wpE_bind (wpE_pure (hcmp v Γ (Extends.refl _) (hE.val r v hl)))
    | none =>
        refine wpE_bind (wpE_iconst64 ?_)
        exact hcmp _ _ (Extends.push _ _) (by rw [hE.val0 r (by simpa using hfl) hl]; simp; rfl)

/-- **A register's value**: its 64 bits, a flag 1 or 0, in a slot of an
    environment keeping `Γ`. -/
theorem wpE_get {fl : List String} {e : RegEnv Slot} {regs : String → UInt64} {Γ : Env} {w : World}
    {r : String} {Q : Slot .i64 → Env → World → Prop} (hE : EnvOk fl e regs Γ)
    (h : ∀ v Γ', Extends Γ Γ' → Γ'[v]? = some (.sc .i64 (regs r)) → Q v Γ' w) :
    wpE cfg (e.get fl r) Q Γ w := by
  unfold RegEnv.get
  by_cases hfl : fl.contains r = true
  · rw [if_pos hfl]
    refine wpE_bind (wpE_getFlag hE fun f Γ1 hx1 hf => ?_)
    refine wpE_bind (wpE_iconst64 ?_)
    refine wpE_bind (wpE_iconst64 ?_)
    have hbool : boolV (regs r == 1) = .sc .i8 (regs r) := by
      rcases bit01 (hE.bit r hfl) with h0 | h0 <;> rw [h0] <;> rfl
    refine wpE_val (v := .sc .i64 (regs r)) ?_ ?_
    · exact eval_sel01 _ Γ1 f _ (hE.bit r hfl) (by rw [hf, hbool])
    · exact h _ _ (((hx1.push_of _).push_of _).push_of _) (get_push_last _ _)
  · rw [if_neg hfl]
    cases hl : e.vals.lookup r with
    | some v => exact wpE_pure (h v Γ (Extends.refl _) (hE.val r v hl))
    | none =>
        refine wpE_val (v := .sc .i64 0) (by simp [Op'.erase, evalOp, ofInt]; rfl) ?_
        exact h _ _ (Extends.push _ _) (by rw [hE.val0 r (by simpa using hfl) hl]; simp)

/-- **An operation of the warp language**: its value on the two operands'. -/
theorem wpE_opOf {Γ : Env} {w : World} (o : WOp) {a b : Slot .i64} {x y : UInt64}
    {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : ∀ v Γ', Extends Γ Γ' → Γ'[v]? = some (.sc .i64 (o.run x y)) → Q v Γ' w) :
    wpE cfg (opOf o a b) Q Γ w := by
  cases o <;> unfold opOf <;>
    refine wpE_val (v := .sc .i64 _) ?_ (h _ _ (Extends.push _ _) (get_push_last _ _)) <;>
    simp [Op'.erase, evalOp, Sem.get, ha, hb, WOp.run, norm, ibinOp, and_wide64, ClifTy.isInt, bin,
      shiftBin, zipIntBits, ClifTy.width, shr_mod64, min_wide]

theorem c64_ofNat (v : Nat) : c64 (v : Int) = UInt64.ofNat v := by
  apply UInt64.toNat_inj.mp
  simp only [c64, UInt64.toNat_ofNat']
  rw [show (1 <<< 64 : Nat) = 2 ^ 64 from rfl,
    show (v : Int).emod ((2 ^ 64 : Nat) : Int) = (v : Int) % ((2 ^ 64 : Nat) : Int) from rfl]
  omega

theorem wpE_argOf {fl : List String} {e : RegEnv Slot} {st : LSt} {Γ : Env} {w : World} (a : WArg)
    {Q : Slot .i64 → Env → World → Prop} (hE : EnvOk fl e st.regs Γ)
    (h : ∀ v Γ', Extends Γ Γ' → Γ'[v]? = some (.sc .i64 (wargL st a)) → Q v Γ' w) :
    wpE cfg (argOf fl e a) Q Γ w := by
  cases a with
  | reg n => exact wpE_get hE h
  | imm v =>
      refine wpE_iconst64 ?_
      exact h _ _ (Extends.push _ _) (by simp [wargL, c64_ofNat])

theorem cmpOf_run (c : SCmp) (x y : UInt64) : cmpInt (cmpOf c) .i64 x y = c.run x y := by
  cases c <;> simp [cmpOf, cmpInt, SCmp.run] <;> omega

theorem boolV_bit {x : UInt64} (h : x ≤ 1) : boolV (x == 1) = .sc .i8 x := by
  rcases bit01 h with h0 | h0 <;> rw [h0] <;> rfl

/-- The values a join or a loop carries for registers `rv` then flags `rf`. -/
def carried (regs : String → UInt64) (rv rf : List String) : List V :=
  rv.map (fun r => .sc .i64 (regs r)) ++ rf.map (fun r => .sc .i8 (regs r))

theorem mapM_extends {Γ Γ' : Env} (hx : Extends Γ Γ') :
    ∀ {l : List Nat} {xs : List V}, l.mapM (fun r => Γ[r]?) = some xs → l.mapM (fun r => Γ'[r]?) = some xs
  | [], xs, h => h
  | r :: l, xs, h => by
      simp only [List.mapM_cons, bind, Option.bind] at h ⊢
      cases hr : Γ[r]? with
      | none => simp [hr] at h
      | some x =>
          rw [hr] at h
          cases hl : l.mapM (fun r => Γ[r]?) with
          | none => simp [hl] at h
          | some ys =>
              simp only [hl] at h
              have h' : x :: ys = xs := Option.some.inj h
              simp [hx.get hr, mapM_extends hx hl, h']

theorem appendV_slots {a b : List ClifTy} (x : Vals Slot a) (y : Vals Slot b) :
    (appendV x y).slots = x.slots ++ y.slots := by
  induction x with
  | nil => rfl
  | cons v vs ih => simp [appendV, Vals.slots, ih]

theorem splitV_carries {b : List ClifTy} : ∀ (a : List ClifTy) (n : Nat),
    splitV a (carriesFrom n (a ++ b)) = (carriesFrom n a, carriesFrom (n + a.length) b)
  | [], n => rfl
  | _ :: a, n => by
      simp only [List.cons_append, carriesFrom, splitV, splitV_carries a (n + 1), List.length_cons]
      rw [Nat.add_assoc, Nat.add_comm 1]

/-- **A register list's values**, in slots of an environment keeping `Γ`. -/
theorem wpE_packV {fl : List String} {e : RegEnv Slot} {regs : String → UInt64} {w : World} :
    ∀ (rs : List String) {Γ : Env} {Q : Vals Slot (rs.map fun _ => ClifTy.i64) → Env → World → Prop},
    EnvOk fl e regs Γ →
    (∀ pv Γ', Extends Γ Γ' → pv.slots.mapM (fun r => Γ'[r]?) = some (rs.map fun r => .sc .i64 (regs r)) →
      Q pv Γ' w) →
    wpE cfg (packV (L := Lvl) fl e rs) Q Γ w
  | [], Γ, Q, _, h => wpE_pure (h _ _ (Extends.refl _) rfl)
  | r :: rs, Γ, Q, hE, h => by
      simp only [packV]
      refine wpE_bind (wpE_get hE fun v Γ1 hx1 hv => ?_)
      refine wpE_bind (wpE_packV rs (hE.extend hx1) fun pv Γ2 hx2 hpv => ?_)
      refine wpE_pure (h _ _ (hx1.trans hx2) ?_)
      simp [Vals.slots, hx2.get hv, hpv]

/-- **A flag list's predicates**, in slots of an environment keeping `Γ`. -/
theorem wpE_packF {fl : List String} {e : RegEnv Slot} {regs : String → UInt64} {w : World} :
    ∀ (rs : List String) {Γ : Env} {Q : Vals Slot (rs.map fun _ => ClifTy.i8) → Env → World → Prop},
    EnvOk fl e regs Γ → (∀ r ∈ rs, fl.contains r = true) →
    (∀ pv Γ', Extends Γ Γ' → pv.slots.mapM (fun r => Γ'[r]?) = some (rs.map fun r => .sc .i8 (regs r)) →
      Q pv Γ' w) →
    wpE cfg (packF (L := Lvl) fl e rs) Q Γ w
  | [], Γ, Q, _, _, h => wpE_pure (h _ _ (Extends.refl _) rfl)
  | r :: rs, Γ, Q, hE, hf, h => by
      simp only [packF]
      refine wpE_bind (wpE_getFlag hE fun v Γ1 hx1 hv => ?_)
      refine wpE_bind (wpE_packF rs (hE.extend hx1) (fun r hr => hf r (by simp [hr])) fun pv Γ2 hx2 hpv => ?_)
      refine wpE_pure (h _ _ (hx1.trans hx2) ?_)
      rw [boolV_bit (hE.bit r (hf r (by simp)))] at hv
      simp [Vals.slots, hx2.get hv, hpv]

/-- **The carried registers' values**, flags `rf`. -/
theorem wpE_pack {fl : List String} {e : RegEnv Slot} {regs : String → UInt64} {w : World} {Γ : Env}
    {rv rf : List String} {Q : Vals Slot (carryTys rv rf) → Env → World → Prop}
    (hE : EnvOk fl e regs Γ) (hf : ∀ r ∈ rf, fl.contains r = true)
    (h : ∀ pv Γ', Extends Γ Γ' → pv.slots.mapM (fun r => Γ'[r]?) = some (carried regs rv rf) → Q pv Γ' w) :
    wpE cfg (pack (L := Lvl) fl e rv rf) Q Γ w := by
  simp only [pack]
  refine wpE_bind (wpE_packV rv hE fun a Γ1 hx1 ha => ?_)
  refine wpE_bind (wpE_packF rf (hE.extend hx1) hf fun b Γ2 hx2 hb => ?_)
  refine wpE_pure (h _ _ (hx1.trans hx2) ?_)
  have ha' : a.slots.mapM (fun r => Γ2[r]?) = some (rv.map fun r => .sc .i64 (regs r)) := by
    exact mapM_extends hx2 ha
  simp [appendV_slots, List.mapM_append, ha', hb, carried]

theorem bindV_ok {fl : List String} {Γ : Env} {regs' : String → UInt64} :
    ∀ (rs : List String) (k : Nat) (e : RegEnv Slot) (regs : String → UInt64), EnvOk fl e regs Γ →
    (∀ r ∈ rs, fl.contains r = false) →
    (∀ (i : Nat) (h : i < rs.length), Γ[k + i]? = some (.sc .i64 (regs' rs[i]))) →
    EnvOk fl (bindV e rs (carriesFrom k _)) (fun x => if x ∈ rs then regs' x else regs x) Γ
  | [], k, e, regs, hE, _, _ => by simpa [bindV] using hE
  | r :: rs, k, e, regs, hE, hv, hs => by
      simp only [List.map_cons, carriesFrom, bindV]
      have h1 := hE.setVal r (hv r (by simp)) k (regs' r) (by simpa using hs 0 (by simp))
      have h2 := bindV_ok rs (k + 1) _ _ h1 (fun r hr => hv r (by simp [hr]))
        (fun i hi => by simpa [Nat.add_assoc, Nat.add_comm 1 i] using hs (i + 1) (by simp; omega))
      exact h2.congr fun x => by by_cases hx : x = r <;> by_cases hm : x ∈ rs <;> simp [hx, hm]

theorem bindF_ok {fl : List String} {Γ : Env} {regs' : String → UInt64} :
    ∀ (rs : List String) (k : Nat) (e : RegEnv Slot) (regs : String → UInt64), EnvOk fl e regs Γ →
    (∀ r ∈ rs, fl.contains r = true) → (∀ r ∈ rs, regs' r ≤ 1) →
    (∀ (i : Nat) (h : i < rs.length), Γ[k + i]? = some (.sc .i8 (regs' rs[i]))) →
    EnvOk fl (bindF e rs (carriesFrom k _)) (fun x => if x ∈ rs then regs' x else regs x) Γ
  | [], k, e, regs, hE, _, _, _ => by simpa [bindF] using hE
  | r :: rs, k, e, regs, hE, hf, hb, hs => by
      simp only [List.map_cons, carriesFrom, bindF]
      have h1 := hE.setFlag r (hf r (by simp)) k (regs' r) (hb r (by simp)) (by simpa using hs 0 (by simp))
      have h2 := bindF_ok rs (k + 1) _ _ h1 (fun r hr => hf r (by simp [hr])) (fun r hr => hb r (by simp [hr]))
        (fun i hi => by simpa [Nat.add_assoc, Nat.add_comm 1 i] using hs (i + 1) (by simp; omega))
      exact h2.congr fun x => by by_cases hx : x = r <;> by_cases hm : x ∈ rs <;> simp [hx, hm]

/-- **The code's environment after a join or a loop** holds the registers
    the carries bring, and the ones they do not as before. -/
theorem unpack_ok {fl : List String} {e : RegEnv Slot} {regs regs' : String → UInt64} {Γ Γ' : Env} {n : Nat}
    {rv rf : List String} (hE : EnvOk fl e regs Γ) (hx : Extends Γ Γ') (hn : Γ.size ≤ n)
    (hv : ∀ r ∈ rv, fl.contains r = false) (hf : ∀ r ∈ rf, fl.contains r = true)
    (hb : ∀ r, fl.contains r = true → regs' r ≤ 1) (hfr : ∀ r, r ∉ rv → r ∉ rf → regs' r = regs r) :
    EnvOk fl (unpack e rv rf (carriesFrom n (carryTys rv rf))) regs' (bindAt Γ' n (carried regs' rv rf)) := by
  simp only [unpack, carryTys, splitV_carries]
  have hx' := hx.bindAt hn (carried regs' rv rf)
  have h1 := bindV_ok (regs' := regs') rv n e regs (hE.extend hx') hv (fun i hi => by
    rw [bindAt_at]; simp only [carried]; rw [List.getElem?_append_left (by simpa using hi)]; simp [hi])
  have h2 := bindF_ok (regs' := regs') rf (n + (rv.map fun _ => ClifTy.i64).length) _ _ h1 hf
    (fun r hr => hb r (hf r hr)) (fun i hi => by
      rw [Nat.add_assoc, bindAt_at]
      simp [carried, List.getElem?_append_right]; exact ⟨_, List.getElem?_eq_getElem hi, rfl⟩)
  refine h2.congr fun x => ?_
  by_cases h1 : x ∈ rf
  · simp [h1]
  · by_cases h2 : x ∈ rv
    · simp [h1, h2]
    · simp [h1, h2, hfr x h2 h1]

theorem carried_inj {a b : String → UInt64} {rv rf : List String} (h : carried a rv rf = carried b rv rf) :
    (∀ r ∈ rv, a r = b r) ∧ ∀ r ∈ rf, a r = b r := by
  unfold carried at h
  obtain ⟨h1, h2⟩ := List.append_inj h (by simp)
  exact ⟨fun r hr => by simpa using List.map_inj_left.mp h1 r hr,
    fun r hr => by simpa using List.map_inj_left.mp h2 r hr⟩

theorem carried_length (regs : String → UInt64) (rv rf : List String) :
    (carried regs rv rf).length = (carryTys rv rf).length := by simp [carried, carryTys]

/-- The carries a loop or a join binds, read back from where they are bound. -/
theorem carries_read {Γ : Env} {n : Nat} {xs : List V} : ∀ (tys : List ClifTy) (k : Nat),
    xs.length = k + tys.length →
    (carriesFrom (n + k) tys).slots.mapM (fun r => (bindAt Γ n xs)[r]?) = some (xs.drop k)
  | [], k, h => by simp at h; simp [carriesFrom, Vals.slots, List.drop_eq_nil_of_le (show xs.length ≤ k by omega)]
  | _ :: ts, k, h => by
      have hk : k < xs.length := by simp at h; omega
      have ih := carries_read (Γ := Γ) (n := n) (xs := xs) ts (k + 1) (by simp at h; omega)
      rw [show n + (k + 1) = n + k + 1 by omega] at ih
      simp only [carriesFrom, Vals.slots, List.mapM_cons, bindAt_at, ih, List.getElem?_eq_getElem hk]
      rw [List.drop_eq_getElem_cons hk]; rfl

open Classical in
/-- The least `m ≤ n` that `P` holds of, or `n`. -/
noncomputable def least (P : Nat → Prop) : Nat → Nat
  | 0 => 0
  | n + 1 => if P (least P n) then least P n else n + 1

theorem least_inv (P : Nat → Prop) : ∀ n, (∀ m, m < least P n → ¬P m) ∧ (P (least P n) ∨ least P n = n)
  | 0 => ⟨fun m h => absurd h (Nat.not_lt_zero _), .inr rfl⟩
  | n + 1 => by
      obtain ⟨ih1, ih2⟩ := least_inv P n
      by_cases hp : P (least P n)
      · simp only [least, if_pos hp]; exact ⟨ih1, .inl hp⟩
      · simp only [least, if_neg hp]
        have hn : least P n = n := ih2.resolve_left hp
        refine ⟨fun m hm hpm => ?_, .inr (by simp)⟩
        rcases Nat.lt_or_ge m n with h | h
        · exact ih1 m (by omega) hpm
        · have : m = n := by omega
          subst this; rw [hn] at hp; exact hp hpm

/-- **The least of what `P` holds of**, below a bound one of them is. -/
theorem least_spec {P : Nat → Prop} {k n : Nat} (hk : P k) (hn : k ≤ n) :
    P (least P n) ∧ ∀ m, P m → least P n ≤ m := by
  obtain ⟨h1, h2⟩ := least_inv P n
  have hmin : ∀ m, P m → least P n ≤ m := fun m hm => Nat.not_lt.mp fun h => h1 m h hm
  refine ⟨?_, hmin⟩
  rcases h2 with h | h
  · exact h
  · have := hmin k hk; rw [h] at this ⊢; rw [show n = k by omega]; exact hk

theorem laneRun_while_false (coop : WStmt → LSt → Option LSt) (smv : UInt64) {cond : String} {body : WStmt}
    {s : LSt} (h : (s.regs cond == 1) = false) : ∀ k, laneRun coop smv k (.uwhile cond body) s = some s
  | 0 => by simp [laneRun, h]
  | _ + 1 => by simp [laneRun, h]

theorem eval_ne_no {m : Mem} {Γ : Env} {c no : Slot .i8} {b : Bool} (hc : Γ[c]? = some (boolV b))
    (hno : Γ[no]? = some (.sc .i8 0)) : evalOp m Γ (.icmp .ne c no) = some (boolV b) := by
  cases b <;> simp [evalOp, Sem.get, hc, hno, zipIntCmp, ClifTy.isInt, boolV, cmpInt]

theorem mem_dedup_go (r : String) : ∀ (l acc : List String),
    r ∈ l.foldl (fun acc r => if acc.contains r then acc else acc ++ [r]) acc ↔ r ∈ acc ∨ r ∈ l
  | [], acc => by simp
  | x :: l, acc => by
      simp only [List.foldl_cons]
      rw [mem_dedup_go r l]
      split
      · rename_i hc
        have : x ∈ acc := by simpa using hc
        constructor
        · rintro (h | h)
          · exact .inl h
          · exact .inr (List.mem_cons_of_mem _ h)
        · rintro (h | h)
          · exact .inl h
          · rcases List.mem_cons.mp h with rfl | h
            · exact .inl this
            · exact .inr h
      · simp [or_assoc]

theorem mem_dedup {r : String} {l : List String} : r ∈ dedup l ↔ r ∈ l := by
  unfold dedup; rw [mem_dedup_go]; simp

/-- The code's environment, the lane's registers and the slots every
    statement reads: the shared base, the lane, and a false flag. -/
structure Slots (sm lane : Slot .i64) (no : Slot .i8) (smv lv : UInt64) (Γ : Env) : Prop where
  sm : Γ[sm]? = some (.sc .i64 smv)
  lane : Γ[lane]? = some (.sc .i64 lv)
  no : Γ[no]? = some (.sc .i8 0)

theorem Slots.extend {sm lane : Slot .i64} {no : Slot .i8} {smv lv : UInt64} {Γ Γ' : Env}
    (h : Slots sm lane no smv lv Γ) (hx : Extends Γ Γ') : Slots sm lane no smv lv Γ' :=
  ⟨hx.get h.sm, hx.get h.lane, hx.get h.no⟩

/-- Each register a statement sets other than by a comparison is a value
    register. -/
def Wf (fl : List String) (s : WStmt) : Prop := ∀ r, (r, false) ∈ assigns s → fl.contains r = false

theorem set_regs (st : LSt) (d : String) (x : UInt64) :
    (st.set d x).regs = (LSt.set ⟨st.regs, default⟩ d x).regs := rfl

/-- **A statement changes only the registers it sets**, given that the
    cooperative statements do so. -/
theorem laneRun_frame (coop : WStmt → LSt → Option LSt) (smv : UInt64)
    (hcoop : ∀ s st st', coop s st = some st' → ∀ r, r ∉ setsOf s → st'.regs r = st.regs r) :
    ∀ (s : WStmt) (f : Nat) (st st' : LSt), laneRun coop smv f s st = some st' →
      ∀ r, r ∉ setsOf s → st'.regs r = st.regs r := by
  intro s
  induction s with
  | skip => intro f st st' h r _; simp [laneRun] at h; subst h; rfl
  | seq a b iha ihb =>
      intro f st st' h r hr
      simp only [laneRun, bind, Option.bind] at h
      split at h
      · cases h
      rename_i st1 h1
      simp only [setsOf, List.mem_append, not_or] at hr
      rw [ihb f st1 st' h r hr.2, iha f st st1 h1 r hr.1]
  | bin o d a b | setp c d a b | mov d a =>
      intro f st st' h r hr
      simp [laneRun] at h; subst h
      simp only [setsOf, List.mem_singleton] at hr
      simp [LSt.set, hr]
  | stgB addr val =>
      intro f st st' h r _
      simp only [laneRun, bind, Option.bind] at h
      split at h
      · cases h
      simp at h; subst h; rfl
  | uif cond t e iht ihe =>
      intro f st st' h r hr
      simp only [setsOf, List.mem_append, not_or] at hr
      simp only [laneRun] at h
      split at h
      · exact iht f st st' h r hr.1
      · exact ihe f st st' h r hr.2
  | uwhile cond body ih =>
      intro f
      induction f with
      | zero =>
          intro st st' h r _
          simp only [laneRun] at h
          split at h
          · cases h
          · cases h; rfl
      | succ f ihf =>
          intro st st' h r hr
          simp only [laneRun] at h
          split at h
          · simp only [bind, Option.bind] at h
            split at h
            · cases h
            rename_i st1 h1
            simp only [setsOf, List.mem_cons, not_or] at hr
            rw [ihf st1 st' h r (by simp [setsOf, hr.1, hr.2]), ih f st st1 h1 r hr.2]
          · cases h; rfl
  | coopCopy dst src len =>
      intro f st st' h r hr
      exact hcoop _ st st' (by simpa [laneRun] using h) r hr
  | coopWindow found p0 cand0 sp inStride searchLim hashLog tbl =>
      intro f st st' h r hr
      exact hcoop _ st st' (by simpa [laneRun] using h) r hr
  | coopExtendStep adv p0 cand0 ml inStride endCap =>
      intro f st st' h r hr
      exact hcoop _ st st' (by simpa [laneRun] using h) r hr

/-- **What a statement's code does**: from a lane state its environment
    holds, where the lane's run of the statement ends, the code ends with an
    environment holding the lane's registers after. -/
def CodeOk (cfg : Cfg) (coop : WStmt → LSt → Option LSt) (fl : List String) (sm lane : Slot .i64)
    (no : Slot .i8) (smv lv : UInt64) (s : WStmt) : Prop :=
  ∀ (f : Nat) (e : RegEnv Slot) (Γ : Env) (st st' : LSt) (Q : RegEnv Slot → Env → World → Prop),
    Wf fl s → EnvOk fl e st.regs Γ → Slots sm lane no smv lv Γ → laneRun coop smv f s st = some st' →
    (∀ e' Γ', Extends Γ Γ' → EnvOk fl e' st'.regs Γ' → Q e' Γ' st'.w) →
    wpE cfg (codeOf fl sm lane no s e) Q Γ st.w

/-- **What a statement's code does at the end of a loop's body**: from a lane
    state its environment holds, where the lane's run of the statement then
    of the tail's statements ends, the code goes round the loop with the
    carries the lane ends with, its flags bits. -/
def TailOk (cfg : Cfg) (coop : WStmt → LSt → Option LSt) (fl : List String) (sm lane : Slot .i64)
    (no : Slot .i8) (smv lv : UInt64) (s : WStmt) : Prop :=
  ∀ (t : Tail Lvl) (f : Nat) (e : RegEnv Slot) (Γ : Env) (st st' : LSt) (C : List V → World → Prop),
    straight t.post = true → CodeOk cfg coop fl sm lane no smv lv t.post → Wf fl s → Wf fl t.post →
    (∀ r ∈ t.rf, fl.contains r = true) → EnvOk fl e st.regs Γ → Slots sm lane no smv lv Γ →
    laneRun coop smv f (.seq s t.post) st = some st' →
    ((∀ r, fl.contains r = true → st'.regs r ≤ 1) → C (carried st'.regs t.rv t.rf) st'.w) →
    wpC cfg t.lp (codeOfT fl sm lane no (some t) s e) C Γ st.w

/-- A straight statement's code is its code. -/
theorem straightCode_eq (fl : List String) (sm lane : Slot .i64) (no : Slot .i8) :
    ∀ (s : WStmt), straight s = true → ∀ (e : RegEnv Slot),
      straightCode (L := Lvl) fl sm lane no s e = codeOf fl sm lane no s e := by
  intro s
  induction s with
  | seq a b iha ihb =>
      intro h e
      simp only [straight, Bool.and_eq_true] at h
      simp only [straightCode, codeOf, codeOfT]
      rw [iha h.1 e]
      congr 1
      funext x
      exact ihb h.2 x
  | skip => intro _ _; rfl
  | bin => intro _ _; rfl
  | setp => intro _ _; rfl
  | mov => intro _ _; rfl
  | stgB => intro h; simp [straight] at h
  | uif => intro h; simp [straight] at h
  | uwhile => intro h; simp [straight] at h
  | coopCopy => intro h; simp [straight] at h
  | coopWindow => intro h; simp [straight] at h
  | coopExtendStep => intro h; simp [straight] at h

theorem laneRun_seq_assoc (coop : WStmt → LSt → Option LSt) (smv : UInt64) (f : Nat) (a b c : WStmt) (st : LSt) :
    laneRun coop smv f (.seq (.seq a b) c) st = laneRun coop smv f (.seq a (.seq b c)) st := by
  simp only [laneRun, bind, Option.bind_assoc]

theorem laneRun_seq_skip (coop : WStmt → LSt → Option LSt) (smv : UInt64) (f : Nat) (s : WStmt) (st : LSt) :
    laneRun coop smv f (.seq s .skip) st = laneRun coop smv f s st := by
  simp only [laneRun, bind]
  cases laneRun coop smv f s st <;> rfl

theorem CodeOk.skip (coop : WStmt → LSt → Option LSt) (fl : List String) (sm lane : Slot .i64) (no : Slot .i8)
    (smv lv : UInt64) : CodeOk cfg coop fl sm lane no smv lv .skip := by
  intro f e Γ st st' Q _ hE _ hr hQ
  simp only [laneRun, Option.some.injEq] at hr; subst hr
  exact wpE_pure (hQ e Γ (Extends.refl _) hE)

theorem CodeOk.seq {coop : WStmt → LSt → Option LSt} {fl : List String} {sm lane : Slot .i64} {no : Slot .i8}
    {smv lv : UInt64} {a b : WStmt} (ha : CodeOk cfg coop fl sm lane no smv lv a)
    (hb : CodeOk cfg coop fl sm lane no smv lv b) : CodeOk cfg coop fl sm lane no smv lv (.seq a b) := by
  intro f e Γ st st' Q hwf hE hS hr hQ
  simp only [laneRun, bind, Option.bind] at hr
  split at hr
  · cases hr
  rename_i st1 h1
  simp only [codeOf, codeOfT]
  exact wpE_bind (ha f e Γ st st1 _ (fun r hr => hwf r (by simp [assigns, hr])) hE hS h1
    fun e1 Γ1 hx1 hE1 => hb f e1 Γ1 st1 st' Q (fun r hr => hwf r (by simp [assigns, hr])) hE1
      (hS.extend hx1) hr fun e2 Γ2 hx2 hE2 => hQ e2 Γ2 (hx1.trans hx2) hE2)

/-- **Going round from a body's end**: the tail's statements, then the jump
    with the carries the lane ends with. -/
theorem goRound_ok {coop : WStmt → LSt → Option LSt} {fl : List String} {sm lane : Slot .i64} {no : Slot .i8}
    {smv lv : UInt64} {t : Tail Lvl} {f : Nat} {e : RegEnv Slot} {Γ : Env} {st st' : LSt} {C : List V → World → Prop}
    (hs : straight t.post = true) (hpost : CodeOk cfg coop fl sm lane no smv lv t.post) (hw : Wf fl t.post)
    (hrf : ∀ r ∈ t.rf, fl.contains r = true) (hE : EnvOk fl e st.regs Γ) (hS : Slots sm lane no smv lv Γ)
    (hr : laneRun coop smv f t.post st = some st')
    (hC : (∀ r, fl.contains r = true → st'.regs r ≤ 1) → C (carried st'.regs t.rv t.rf) st'.w) :
    wpC cfg t.lp (goRound fl sm lane no t e) C Γ st.w := by
  unfold goRound
  rw [straightCode_eq fl sm lane no t.post hs e]
  refine wpC_bind (hpost f e Γ st st' _ hw hE hS hr fun e' Γ' _ hE' => ?_)
  exact wpC_bind (wpE_pack hE' hrf fun pv Γ'' _ hpv => wpC_cont rfl hpv (hC hE'.bit))

/-- A statement whose code at a body's end is its code, then going round, goes round. -/
theorem tail_of_code {coop : WStmt → LSt → Option LSt} {fl : List String} {sm lane : Slot .i64} {no : Slot .i8}
    {smv lv : UInt64} {s : WStmt} (hA : CodeOk cfg coop fl sm lane no smv lv s)
    (heq : ∀ (t : Tail Lvl) e, codeOfT fl sm lane no (some t) s e =
      (codeOf fl sm lane no s e >>= fun e' => goRound fl sm lane no t e')) :
    TailOk cfg coop fl sm lane no smv lv s := by
  intro t f e Γ st st' C hs hpost hws hwp hrf hE hS hr hC
  rw [heq]
  simp only [laneRun, bind, Option.bind] at hr
  split at hr
  · cases hr
  rename_i s1 h1
  exact wpC_bind (hA f e Γ st s1 _ hws hE hS h1 fun e' Γ' hx hE' =>
    goRound_ok hs hpost hwp hrf hE' (hS.extend hx) hr hC)

/-- **A statement's code does what the lane does**, and at the end of a loop's
    body goes round with the carries the lane ends with. -/
theorem code_lane (coop : WStmt → LSt → Option LSt) (fl : List String) (sm lane : Slot .i64) (no : Slot .i8)
    (smv lv : UInt64)
    (hcoop : ∀ (s : WStmt) (e : RegEnv Slot) (Γ : Env) (st st' : LSt) (Q : RegEnv Slot → Env → World → Prop),
      (s matches .coopCopy .. | .coopWindow .. | .coopExtendStep ..) → Wf fl s →
      EnvOk fl e st.regs Γ → Slots sm lane no smv lv Γ → coop s st = some st' →
      (∀ e' Γ', Extends Γ Γ' → EnvOk fl e' st'.regs Γ' → Q e' Γ' st'.w) →
      wpE cfg (codeOf fl sm lane no s e) Q Γ st.w)
    (hcframe : ∀ s st st', coop s st = some st' → ∀ r, r ∉ setsOf s → st'.regs r = st.regs r) :
    ∀ (s : WStmt), CodeOk cfg coop fl sm lane no smv lv s ∧ TailOk cfg coop fl sm lane no smv lv s := by
  intro s
  induction s with
  | skip =>
      have hA : CodeOk cfg coop fl sm lane no smv lv .skip := by
        intro f e Γ st st' Q _ hE _ hr hQ
        simp only [laneRun, Option.some.injEq] at hr; subst hr
        exact wpE_pure (hQ e Γ (Extends.refl _) hE)
      exact ⟨hA, tail_of_code hA (fun _ _ => rfl)⟩
  | bin o d a b =>
      have hA : CodeOk cfg coop fl sm lane no smv lv (.bin o d a b) := by
        intro f e Γ st st' Q hwf hE hS hr hQ
        simp only [laneRun, Option.some.injEq] at hr; subst hr
        simp only [codeOf]
        refine wpE_bind (wpE_get hE fun va Γ1 hx1 hva => ?_)
        refine wpE_bind (wpE_argOf b (hE.extend hx1) fun vb Γ2 hx2 hvb => ?_)
        refine wpE_bind (wpE_opOf o (hx2.get hva) hvb fun vr Γ3 hx3 hvr => ?_)
        refine wpE_pure ?_
        exact hQ _ _ (hx1.trans (hx2.trans hx3))
          (((hE.extend (hx1.trans (hx2.trans hx3)))).setVal d (hwf d (by simp [assigns])) vr _ hvr)
      exact ⟨hA, tail_of_code hA (fun _ _ => rfl)⟩
  | setp c d a b =>
      have hA : CodeOk cfg coop fl sm lane no smv lv (.setp c d a b) := by
        intro f e Γ st st' Q hwf hE hS hr hQ
        simp only [laneRun, Option.some.injEq] at hr; subst hr
        simp only [codeOf]
        refine wpE_bind (wpE_get hE fun va Γ1 hx1 hva => ?_)
        refine wpE_bind (wpE_argOf b (hE.extend hx1) fun vb Γ2 hx2 hvb => ?_)
        have hva' := hx2.get hva
        refine wpE_op (v := boolV (c.run (st.regs a) (wargL st b))) (by
          simp [Op'.erase, evalOp, Sem.get, hva', hvb, zipIntCmp, ClifTy.isInt, cmpOf_run]) ?_
        have hp : (Γ2.push (boolV (c.run (st.regs a) (wargL st b))))[Γ2.size]? =
            some (boolV (c.run (st.regs a) (wargL st b))) := by simp
        have hx3 := (hx1.trans hx2).push_of (boolV (c.run (st.regs a) (wargL st b)))
        refine wpE_bind (wpE_ret ?_)
        by_cases hfl : fl.contains d = true
        · simp only [if_pos hfl]
          refine wpE_pure ?_
          refine hQ _ _ hx3 ((hE.extend hx3).setFlag d hfl _ (if c.run (st.regs a) (wargL st b) then 1 else 0)
            (by split <;> decide) (by rw [hp]; unfold boolV; split <;> rfl))
        · simp only [if_neg hfl]
          refine wpE_bind (wpE_iconst64 ?_)
          refine wpE_bind (wpE_iconst64 ?_)
          have hx4 := (hx3.push_of (.sc .i64 (c64 1))).push_of (.sc .i64 (c64 0))
          have hp' := ((Extends.push (Γ2.push (boolV (c.run (st.regs a) (wargL st b)))) (.sc .i64 (c64 1))).push_of
            (.sc .i64 (c64 0))).get hp
          let x : UInt64 := if c.run (st.regs a) (wargL st b) then 1 else 0
          refine wpE_op (v := .sc .i64 x) (eval_sel01 _ _ _ x (by simp only [x]; split <;> decide)
            (by rw [hp]; simp only [x]; unfold boolV; split <;> rfl)) ?_
          refine wpE_pure ?_
          exact hQ _ _ (hx4.push_of _) ((hE.extend (hx4.push_of _)).setVal d (by simpa using hfl) _ x (get_push_last _ _))
      exact ⟨hA, tail_of_code hA (fun _ _ => rfl)⟩
  | mov d a =>
      have hA : CodeOk cfg coop fl sm lane no smv lv (.mov d a) := by
        intro f e Γ st st' Q hwf hE hS hr hQ
        simp only [laneRun, Option.some.injEq] at hr; subst hr
        simp only [codeOf]
        refine wpE_bind (wpE_argOf a hE fun va Γ1 hx1 hva => ?_)
        exact wpE_pure (hQ _ _ hx1 ((hE.extend hx1).setVal d (hwf d (by simp [assigns])) va _ hva))
      exact ⟨hA, tail_of_code hA (fun _ _ => rfl)⟩
  | stgB addr val =>
      have hA : CodeOk cfg coop fl sm lane no smv lv (.stgB addr val) := by
        intro f e Γ st st' Q hwf hE hS hr hQ
        cases hm : st.w.mem.store (st.regs addr) 1 (st.regs val &&& 0xff) with
        | none => simp [laneRun, stw, hm] at hr
        | some m =>
        simp [laneRun, stw, hm] at hr; subst hr
        simp only [codeOf]
        refine wpE_bind (wpE_get hE fun vv Γ1 hx1 hvv => ?_)
        refine wpE_bind (wpE_get (hE.extend hx1) fun va Γ2 hx2 hva => ?_)
        refine wpE_istore8 (w' := { obsStore st.w (st.regs addr) 1 (st.regs val &&& 0xff) with mem := m })
          (by simp [runStmt, Sem.get, hx2.get hvv, hva, hm]) ?_
        exact wpE_pure (hQ _ _ (hx1.trans hx2) (hE.extend (hx1.trans hx2)))
      exact ⟨hA, tail_of_code hA (fun _ _ => rfl)⟩
  | coopCopy dst src len =>
      have hA : CodeOk cfg coop fl sm lane no smv lv (.coopCopy dst src len) := by
        intro f e Γ st st' Q hwf hE hS hr hQ
        exact hcoop _ e Γ st st' Q rfl hwf hE hS (by simpa [laneRun] using hr) hQ
      exact ⟨hA, tail_of_code hA (fun _ _ => rfl)⟩
  | coopWindow found p0 cand0 sp inStride searchLim hashLog tbl =>
      have hA : CodeOk cfg coop fl sm lane no smv lv (.coopWindow found p0 cand0 sp inStride searchLim hashLog tbl) := by
        intro f e Γ st st' Q hwf hE hS hr hQ
        exact hcoop _ e Γ st st' Q rfl hwf hE hS (by simpa [laneRun] using hr) hQ
      exact ⟨hA, tail_of_code hA (fun _ _ => rfl)⟩
  | coopExtendStep adv p0 cand0 ml inStride endCap =>
      have hA : CodeOk cfg coop fl sm lane no smv lv (.coopExtendStep adv p0 cand0 ml inStride endCap) := by
        intro f e Γ st st' Q hwf hE hS hr hQ
        exact hcoop _ e Γ st st' Q rfl hwf hE hS (by simpa [laneRun] using hr) hQ
      exact ⟨hA, tail_of_code hA (fun _ _ => rfl)⟩
  | seq a b iha ihb =>
      refine ⟨CodeOk.seq iha.1 ihb.1, ?_⟩
      intro t f e Γ st st' C hs hpost hws hwp hrf hE hS hr hC
      have hwa : Wf fl a := fun r hr => hws r (by simp [assigns, hr])
      have hwb : Wf fl b := fun r hr => hws r (by simp [assigns, hr])
      by_cases hb : straight b = true
      · simp only [codeOfT, hb, if_true]
        rw [laneRun_seq_assoc] at hr
        exact iha.2 { t with post := .seq b t.post } f e Γ st st' C (by simp [straight, hb, hs])
          (CodeOk.seq ihb.1 hpost) hwa (fun r hr => by
            simp only [assigns, List.mem_append] at hr
            exact hr.elim (hwb r) (hwp r)) hrf hE hS hr hC
      · simp only [codeOfT, hb, Bool.false_eq_true, if_false]
        rw [laneRun_seq_assoc] at hr
        simp only [laneRun, bind, Option.bind] at hr
        split at hr
        · cases hr
        rename_i s1 h1
        exact wpC_bind (iha.1 f e Γ st s1 _ hwa hE hS h1 fun e1 Γ1 hx1 hE1 =>
          ihb.2 t f e1 Γ1 s1 st' C hs hpost hwb hwp hrf hE1 (hS.extend hx1) (by simpa [laneRun] using hr) hC)
  | uif cond t el iht ihe =>
      have hA : CodeOk cfg coop fl sm lane no smv lv (.uif cond t el) := by
        intro fu e Γ st st' Q hwf hE hS hr hQ
        have hwt : Wf fl t := fun r hr => hwf r (by simp [assigns, hr])
        have hwe : Wf fl el := fun r hr => hwf r (by simp [assigns, hr])
        have hrv : ∀ r ∈ (dedup (setsOf t ++ setsOf el)).filter (!fl.contains ·), fl.contains r = false :=
          fun r hr => by simpa using (List.mem_filter.mp hr).2
        have hrf : ∀ r ∈ (dedup (setsOf t ++ setsOf el)).filter (fl.contains ·), fl.contains r = true :=
          fun r hr => by simpa using (List.mem_filter.mp hr).2
        have hfr : ∀ r, r ∉ (dedup (setsOf t ++ setsOf el)).filter (!fl.contains ·) →
            r ∉ (dedup (setsOf t ++ setsOf el)).filter (fl.contains ·) → st'.regs r = st.regs r := by
          intro r h1 h2
          have hn : r ∉ setsOf (.uif cond t el) := fun hm => by
            have hd : r ∈ dedup (setsOf t ++ setsOf el) := mem_dedup.mpr (by simpa [setsOf] using hm)
            cases hc : fl.contains r
            · exact h1 (List.mem_filter.mpr ⟨hd, by simpa using hc⟩)
            · exact h2 (List.mem_filter.mpr ⟨hd, by simpa using hc⟩)
          exact laneRun_frame coop smv hcframe _ fu st st' hr r hn
        simp only [codeOf]
        refine wpE_bind (wpE_getFlag hE fun c Γ1 hx1 hc => ?_)
        let Qj : Vals Slot (carryTys ((dedup (setsOf t ++ setsOf el)).filter (!fl.contains ·))
            ((dedup (setsOf t ++ setsOf el)).filter (fl.contains ·))) → Env → World → Prop :=
          fun jv Γ' w' => Extends Γ1 Γ' ∧
            jv.slots.mapM (fun r => Γ'[r]?) = some (carried st'.regs ((dedup (setsOf t ++ setsOf el)).filter (!fl.contains ·))
              ((dedup (setsOf t ++ setsOf el)).filter (fl.contains ·))) ∧
            w' = st'.w ∧ ∀ r, fl.contains r = true → st'.regs r ≤ 1
        have harm : ∀ (s : WStmt) (Γa : Env), Extends Γ1 Γa → Wf fl s →
            (∀ e Γ st st' Q, Wf fl s → EnvOk fl e st.regs Γ → Slots sm lane no smv lv Γ →
              laneRun coop smv fu s st = some st' →
              (∀ e' Γ', Extends Γ Γ' → EnvOk fl e' st'.regs Γ' → Q e' Γ' st'.w) →
              wpE cfg (codeOf fl sm lane no s e) Q Γ st.w) →
            laneRun coop smv fu s st = some st' →
            wpE cfg (codeOf fl sm lane no s e >>= fun e' => pack fl e'
              ((dedup (setsOf t ++ setsOf el)).filter (!fl.contains ·))
              ((dedup (setsOf t ++ setsOf el)).filter (fl.contains ·))) Qj Γa st.w := by
          intro s Γa hxa hws ih hrun
          refine wpE_bind (ih e Γa st st' _ hws (hE.extend (hx1.trans hxa)) (hS.extend (hx1.trans hxa)) hrun
            fun e' Γ' hx' hE' => ?_)
          exact wpE_pack hE' hrf fun pv Γ'' hx'' hpv => ⟨hxa.trans (hx'.trans hx''), hpv, rfl, hE'.bit⟩
        refine wpE_bind (wpE_ifteT (t := .i8) (x := if st.regs cond == 1 then 1 else 0) (y := 0) Qj rfl
          (by rw [hc]; rfl) (hS.extend hx1).no ?_ ?_ ?_ ?_)
        · intro hcmp
          have hcond : (st.regs cond == 1) = true := by
            cases hcd : st.regs cond == 1 <;> simp [hcd, cmpInt] at hcmp ⊢
          simp only [laneRun, hcond, if_true] at hr
          exact harm t _ (Extends.push _ _) hwt (iht.1 fu) hr
        · intro hcmp ne hne
          have hcond : (st.regs cond == 1) = false := by
            cases hcd : st.regs cond == 1 <;> simp [hcd, cmpInt] at hcmp ⊢
          simp only [laneRun, hcond, Bool.false_eq_true, if_false] at hr
          exact harm el _ ((Extends.push Γ1 _).bindAt_of hne []) hwe (ihe.1 fu) hr
        · exact fun jv Γ' w' hq => ⟨_, hq.2.1⟩
        · intro nJ Γ' jv vs w' hle hvs hq
          obtain ⟨hx', hm, rfl, hb⟩ := hq
          rw [hm] at hvs; cases hvs
          exact wpE_pure (hQ _ _ ((hx1.trans hx').bindAt_of hle _)
            (unpack_ok hE (hx1.trans hx') (Nat.le_trans (hx1.trans hx').1 hle) hrv hrf hb hfr))
      refine ⟨hA, ?_⟩
      intro tl fu e Γ st st' C hs hpost hws hwp hrf hE hS hr hC
      have hwt : Wf fl t := fun r hr => hws r (by simp [assigns, hr])
      have hwe : Wf fl el := fun r hr => hws r (by simp [assigns, hr])
      simp only [codeOfT]
      refine wpC_bind (wpE_getFlag hE fun c Γ1 hx1 hc => ?_)
      refine wpC_bind_ret (g := fun _ => e) (wpC_ifteT (t := .i8) (x := if st.regs cond == 1 then 1 else 0) (y := 0)
        rfl (by rw [hc]; rfl) (hS.extend hx1).no ?_ ?_)
      · intro hcmp
        have hcond : (st.regs cond == 1) = true := by
          cases hcd : st.regs cond == 1 <;> simp [hcd, cmpInt] at hcmp ⊢
        have hx : Extends Γ (Γ1.push (boolV true)) := hx1.trans (Extends.push _ _)
        exact wpC_bind_ret (iht.2 tl fu e _ st st' C hs hpost hwt hwp hrf (hE.extend hx) (hS.extend hx)
          (by simpa [laneRun, hcond] using hr) hC)
      · intro hcmp ne hne
        have hcond : (st.regs cond == 1) = false := by
          cases hcd : st.regs cond == 1 <;> simp [hcd, cmpInt] at hcmp ⊢
        have hx : Extends Γ (bindAt (Γ1.push (boolV false)) ne []) :=
          hx1.trans ((Extends.push Γ1 _).bindAt_of hne [])
        exact wpC_bind_ret (ihe.2 tl fu e _ st st' C hs hpost hwe hwp hrf (hE.extend hx) (hS.extend hx)
          (by simpa [laneRun, hcond] using hr) hC)
  | uwhile cond body ih =>
      have hA : CodeOk cfg coop fl sm lane no smv lv (.uwhile cond body) := by
        intro fu e Γ st st' Q hwf hE hS hr hQ
        have hwb : Wf fl body := fun r hr => hwf r (by simp [assigns, hr])
        have hrv0 : ∀ r ∈ (dedup (cond :: setsOf body)).filter (!fl.contains ·), fl.contains r = false :=
          fun r hr => by simpa using (List.mem_filter.mp hr).2
        have hrf0 : ∀ r ∈ (dedup (cond :: setsOf body)).filter (fl.contains ·), fl.contains r = true :=
          fun r hr => by simpa using (List.mem_filter.mp hr).2
        have hout0 : ∀ r, r ∉ (dedup (cond :: setsOf body)).filter (!fl.contains ·) →
            r ∉ (dedup (cond :: setsOf body)).filter (fl.contains ·) → r ∉ setsOf body := by
          intro r h1 h2 hm
          have hd : r ∈ dedup (cond :: setsOf body) := mem_dedup.mpr (List.mem_cons_of_mem _ hm)
          cases hc : fl.contains r
          · exact h1 (List.mem_filter.mpr ⟨hd, by simpa using hc⟩)
          · exact h2 (List.mem_filter.mpr ⟨hd, by simpa using hc⟩)
        simp only [codeOf, codeOfT]
        generalize (dedup (cond :: setsOf body)).filter (!fl.contains ·) = rv at hrv0 hout0 ⊢
        generalize (dedup (cond :: setsOf body)).filter (fl.contains ·) = rf at hrf0 hout0 ⊢
        -- A lane state the loop reaches: its registers outside the carries the
        -- lane's before the loop, its flags bits.
        let Fr (s1 : LSt) : Prop :=
          (∀ r, r ∉ rv → r ∉ rf → s1.regs r = st.regs r) ∧ ∀ r, fl.contains r = true → s1.regs r ≤ 1
        let R (cs : List V) (wi : World) (k : Nat) : Prop := ∃ s1 : LSt, cs = carried s1.regs rv rf ∧ wi = s1.w ∧
          Fr s1 ∧ laneRun coop smv k (.uwhile cond body) s1 = some st'
        have huniq : ∀ s1 s2 : LSt, carried s1.regs rv rf = carried s2.regs rv rf → s1.w = s2.w → Fr s1 → Fr s2 →
            s1 = s2 := by
          intro s1 s2 hc hw h1 h2
          obtain ⟨hv, hf⟩ := carried_inj hc
          cases s1; cases s2
          simp only at hw; subst hw
          congr 1
          funext r
          by_cases hrv : r ∈ rv
          · exact hv r hrv
          · by_cases hrf : r ∈ rf
            · exact hf r hrf
            · exact (h1.1 r hrv hrf).trans (h2.1 r hrv hrf).symm
        refine wpE_bind (wpE_pack hE hrf0 fun init Γp hxp hinit => ?_)
        have hxB : ∀ cs, Extends Γp (bindAt Γp Γp.size cs) := fun cs => (Extends.refl _).bindAt_of (Nat.le_refl _) cs
        refine wpE_bind (wpE_wloopC (fun cs wi => ∃ k, k ≤ fu ∧ R cs wi k) (fun cs wi => least (R cs wi) fu)
          (fun cs wi a Γ1 w1 => Extends (bindAt Γp Γp.size cs) Γ1 ∧ ∃ c s1,
            a = (contIf .ne c no, carriesFrom Γp.size (carryTys rv rf), ()) ∧
            Γ1[c]? = some (boolV (s1.regs cond == 1)) ∧ Γ1[no]? = some (.sc .i8 0) ∧ w1 = wi ∧
            cs = carried s1.regs rv rf ∧ wi = s1.w ∧ Fr s1 ∧ ∃ k, k ≤ fu ∧
              laneRun coop smv k (.uwhile cond body) s1 = some st')
          (fun vs w => vs = carried st'.regs rv rf ∧ w = st'.w ∧ Fr st') ?_ ?_ ?_ ?_ ?_ ?_)
        · exact ⟨_, hinit, fu, Nat.le_refl _, st, rfl, rfl, ⟨fun _ _ _ => rfl, hE.bit⟩, hr⟩
        · rintro wi cs ⟨k, hk, s1, rfl, rfl, hF, hrun⟩
          have hU := unpack_ok (n := Γp.size) hE hxp hxp.1 hrv0 hrf0 hF.2 hF.1
          refine wpE_bind (wpE_getFlag hU fun c Γ1 hx1 hc => ?_)
          exact wpE_pure ⟨hx1, hx1, c, s1, rfl, hc, (hS.extend (hxp.trans ((hxB _).trans hx1))).no, rfl, rfl, rfl, hF,
            k, hk, hrun⟩
        · rintro cs wi a Γ1 w1 ⟨-, c, s1, rfl, hc, hno, -⟩
          exact ⟨_, _, eval_ne_no hc hno⟩
        · rintro cs wi a Γ1 w1 t f ⟨hxe, c, s1, rfl, hc, hno, rfl, rfl, rfl, hF, k, hk, hrun⟩ hev hexit
          simp only [contIf] at hev hexit
          rw [eval_ne_no hc hno] at hev
          have hcf : (s1.regs cond == 1) = false := by
            cases hb : s1.regs cond == 1 <;> simp [hb, boolV] at hev ⊢
            obtain ⟨-, rfl⟩ := hev; simp at hexit
          rw [laneRun_while_false coop smv hcf k] at hrun
          cases hrun
          have hrd := carries_read (Γ := Γp) (n := Γp.size) (xs := carried st'.regs rv rf) (carryTys rv rf) 0
            (by simp [carried_length])
          simp only [Nat.add_zero, List.drop_zero] at hrd
          exact ⟨_, mapM_extends (hxe.push_of _) hrd, rfl, rfl, hF⟩
        · rintro lp cs wi a Γ1 wb t f ⟨hxe, c, s1, rfl, hc, hno, rfl, rfl, rfl, hF, k, hk, hrun⟩ _ hev hcont
          simp only [contIf] at hev hcont
          rw [eval_ne_no hc hno] at hev
          have hct : (s1.regs cond == 1) = true := by
            cases hb : s1.regs cond == 1 <;> simp [hb, boolV] at hev ⊢
            obtain ⟨-, rfl⟩ := hev; simp at hcont
          obtain ⟨hR0, hmin⟩ := least_spec (P := R (carried s1.regs rv rf) s1.w) ⟨s1, rfl, rfl, hF, hrun⟩ hk
          generalize hk0e : least (R (carried s1.regs rv rf) s1.w) fu = k0 at hR0 hmin
          obtain ⟨s1', hcs, hw, hF', hrun0⟩ := hR0
          rw [huniq s1' s1 hcs.symm hw.symm hF' hF] at hrun0
          have hk0 : k0 ≤ k := hmin k ⟨s1, rfl, rfl, hF, hrun⟩
          cases k0 with
          | zero => simp [laneRun, hct] at hrun0
          | succ k' =>
          simp only [laneRun, hct, if_true, bind, Option.bind] at hrun0
          split at hrun0
          · cases hrun0
          rename_i s2 hb2
          have hfr2 : ∀ r, r ∉ rv → r ∉ rf → s2.regs r = st.regs r := fun r h1 h2 =>
            (laneRun_frame coop smv hcframe body k' s1 s2 hb2 r (hout0 r h1 h2)).trans (hF.1 r h1 h2)
          have hxe' : Extends Γ (Γ1.push (.sc t f)) := (hxp.trans ((hxB _).trans hxe)).push_of _
          have hn' : Γ.size ≤ Γ1.size + 1 := by simpa using hxe'.1
          have hU := unpack_ok hE hxe' hn' hrv0 hrf0 hF.2 hF.1
          refine wpC_bind_ret (ih.2 ⟨rv, rf, lp, .skip⟩ k' _ _ s1 s2 _ rfl (CodeOk.skip coop fl sm lane no smv lv)
            hwb (fun r h => by simp [assigns] at h) hrf0 hU (hS.extend (hxe'.bindAt hn' _))
            (by rw [laneRun_seq_skip]; exact hb2) fun hbit => ?_)
          have hR2 : R (carried s2.regs rv rf) s2.w k' := ⟨s2, rfl, rfl, ⟨hfr2, hbit⟩, hrun0⟩
          refine ⟨carried_length _ _ _, ⟨k', by omega, hR2⟩, ?_⟩
          have := (least_spec hR2 (show k' ≤ fu by omega)).2 k' hR2
          show least (R (carried s2.regs rv rf) s2.w) fu < least (R (carried s1.regs rv rf) s1.w) fu
          rw [hk0e]; omega
        · rintro Γb vs w' nE hxb hle ⟨rfl, rfl, hF⟩
          exact wpE_pure (hQ _ _ ((hxp.trans hxb).bindAt (Nat.le_trans hxp.1 hle) _)
            (unpack_ok hE (hxp.trans hxb) (Nat.le_trans hxp.1 hle) hrv0 hrf0 hF.2 hF.1))
      refine ⟨hA, tail_of_code hA fun t e => ?_⟩
      simp only [codeOf, codeOfT, Prog.bind_assoc]
      rfl

/-- **The code a statement writes ends as the lane's run of it**, its
    environment holding the lane's registers after, given that the
    cooperative statements' do (`hcoop`). -/
theorem codeOf_lane (coop : WStmt → LSt → Option LSt) (fl : List String) (sm lane : Slot .i64) (no : Slot .i8)
    (smv lv : UInt64)
    (hcoop : ∀ (s : WStmt) (e : RegEnv Slot) (Γ : Env) (st st' : LSt) (Q : RegEnv Slot → Env → World → Prop),
      (s matches .coopCopy .. | .coopWindow .. | .coopExtendStep ..) → Wf fl s →
      EnvOk fl e st.regs Γ → Slots sm lane no smv lv Γ → coop s st = some st' →
      (∀ e' Γ', Extends Γ Γ' → EnvOk fl e' st'.regs Γ' → Q e' Γ' st'.w) →
      wpE cfg (codeOf fl sm lane no s e) Q Γ st.w)
    (hcframe : ∀ s st st', coop s st = some st' → ∀ r, r ∉ setsOf s → st'.regs r = st.regs r) :
    ∀ (s : WStmt), CodeOk cfg coop fl sm lane no smv lv s := fun s =>
  (code_lane coop fl sm lane no smv lv hcoop hcframe s).1

end

end Lz4Code
