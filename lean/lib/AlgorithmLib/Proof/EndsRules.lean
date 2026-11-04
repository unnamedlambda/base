module
public import AlgorithmLib.Proof.Ends
meta import AlgorithmLib.Proof.Ends
public import AlgorithmLib.Proof.LibSpecFloat
meta import AlgorithmLib.Proof.LibSpecFloat
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Proof.EndsRules` — one ending rule for each construct of a body

`wpE cfg p Q Γ w` says `p`, run from `Γ` and `w`, ends, answering, in `Q`:
`Proof.Ends`'s `PE` from that one state. A statement about every state a
precondition allows is one about each (`PE.of_wpE`), so its rules read as the
weakest precondition's (`Proof.Rules`), each run ending: sequencing
(`wpE_bind`), an operation (`wpE_op`, `wpE_val`), a store (`wpE_store`), a
call to the program's own functions (`wpE_callLocal`, `wpE_callLocalVoid`), a
branch (`wpE_ifteT`) and a loop by invariant and measure (`wpE_wloopN`).

A program of operations alone (`OpsOnly`) ends wherever its weakest
precondition holds with faults ruled out (`wpE_of_wp`): every operation must
then answer. So each operation's ending rule is its `wp` rule
(`wpE_iconst32u`, `wpE_loadF32`, …).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.HProg.Contracts

namespace AlgorithmLib.LibSpec

/-- **`p`, run from `Γ` and `w`, ends**, answering, in `Q`. -/
def wpE {α : Type} (cfg : Cfg) (p : Prog Slot Lvl α) (Q : α → Env → World → Prop) (Γ : Env) (w : World) :
    Prop :=
  PE cfg (At Γ w) p Q

section
variable {α β : Type} {cfg : Cfg} {Γ : Env} {w : World}

/-- What ends from each state a precondition allows ends from all of them:
    the code before runs from each starting state to one state, deterministically. -/
theorem PE.of_wpE {P : Env → World → Prop} {p : Prog Slot Lvl α} {Q : α → Env → World → Prop}
    (h : ∀ Γ w, P Γ w → wpE cfg p Q Γ w) : PE cfg P p Q := by
  intro s P0 hf hs Γi wi h0
  obtain ⟨k, Γ1, w1, hr, hn, hP⟩ := hs Γi wi h0
  exact h Γ1 w1 hP s (At Γi wi) hf (fun Γ w ⟨e1, e2⟩ => by
    subst e1 e2; exact ⟨k, Γ1, w1, hr, hn, rfl, rfl⟩) Γi wi ⟨rfl, rfl⟩

theorem wpE_mono {p : Prog Slot Lvl α} {Q Q' : α → Env → World → Prop}
    (hQ : ∀ a Γ w, Q a Γ w → Q' a Γ w) (h : wpE cfg p Q Γ w) : wpE cfg p Q' Γ w :=
  PE.conseq h (fun _ _ h => h) hQ

theorem wpE_bind {p : Prog Slot Lvl α} {f : α → Prog Slot Lvl β} {Q : β → Env → World → Prop}
    (h : wpE cfg p (fun a => wpE cfg (f a) Q) Γ w) : wpE cfg (p >>= f) Q Γ w :=
  PE.bind h fun _ => PE.of_wpE fun _ _ h => h

theorem wpE_ret {a : α} {Q : α → Env → World → Prop} (h : Q a Γ w) : wpE cfg (.ret a) Q Γ w :=
  PE.ret fun _ _ ⟨e1, e2⟩ => e1 ▸ e2 ▸ h

theorem wpE_pure {a : α} {Q : α → Env → World → Prop} (h : Q a Γ w) :
    wpE cfg (pure a : Prog Slot Lvl α) Q Γ w :=
  wpE_ret h

theorem wpE_op {ty} {o : Op' Slot ty} {k : Slot ty → Prog Slot Lvl α} {Q : α → Env → World → Prop} {v : V}
    (he : evalOp w.mem Γ o.erase = some v) (h : wpE cfg (k Γ.size) Q (Γ.push v) w) :
    wpE cfg (.op o k) Q Γ w :=
  PE.op (fun r => PE.of_wpE fun Γ' w' ⟨Γ0, v', e, hr, ⟨e1, e2⟩, hv⟩ => by
      subst e1 e2 e hr; rw [he] at hv; cases hv; exact h)
    (fun _ _ ⟨e1, e2⟩ => ⟨v, e1 ▸ e2 ▸ he⟩)

theorem wpE_val {ty} {o : Op' Slot ty} {v : V} {Q : Slot ty → Env → World → Prop}
    (he : evalOp w.mem Γ o.erase = some v) (h : Q Γ.size (Γ.push v) w) : wpE cfg (op o) Q Γ w :=
  wpE_op he (wpE_ret h)

theorem wpE_store {ty} {v : Slot ty} {a : Slot .i64} {k : Prog Slot Lvl α} {Q : α → Env → World → Prop}
    {w' : World} (hs : runStmt cfg Γ w (.store ty v a) = .ok Γ w') (h : wpE cfg k Q Γ w') :
    wpE cfg (.store v a k) Q Γ w :=
  PE.store (R := At Γ w') (fun n Γ0 w0 ⟨hn, e1, e2⟩ => by subst e1 e2; exact ⟨_, _, hs, hn, rfl, rfl⟩) h

theorem wpE_istore8 {ty} {v : Slot ty} {a : Slot .i64} {hi : ty.isInt = true} {k : Prog Slot Lvl α}
    {Q : α → Env → World → Prop} {w' : World} (hs : runStmt cfg Γ w (.istore8 v a) = .ok Γ w')
    (h : wpE cfg k Q Γ w') : wpE cfg (.istore8 v a hi k) Q Γ w :=
  PE.istore8 (R := At Γ w') (fun n Γ0 w0 ⟨hn, e1, e2⟩ => by subst e1 e2; exact ⟨_, _, hs, hn, rfl, rfl⟩) h

theorem wpE_callLocal {ps : List ClifTy} {res : Option ClifTy} {r : LocalRef ps res} {args : Vals Slot ps}
    {k : ResV Slot res → Prog Slot Lvl α} {Q : α → Env → World → Prop} {vs : List V} {v : V} {w' : World}
    (hres : res.isSome = true) (hvs : args.slots.mapM (fun r => Γ[r]?) = some vs)
    (hc : callOf cfg.locals r.callee vs (obsCall w r.callee vs) = some (some v, w'))
    (h : wpE cfg (k (resSlot res Γ.size)) Q (Γ.push v) w') : wpE cfg (.callLocal r args k) Q Γ w :=
  PE.callLocal hres (R := fun n Γ' w'' => n = Γ.size ∧ Γ' = Γ.push v ∧ w'' = w')
    (fun n Γ0 w0 ⟨hn, e1, e2⟩ => by
      subst e1 e2
      exact ⟨_, _, by rw [runStmt_call]; simp only [hvs, hc], by simp [hn], hn.symm, rfl, rfl⟩)
    (fun n => PE.of_wpE fun Γ' w'' ⟨e1, e2, e3⟩ => by subst e1 e2 e3; exact h)

theorem wpE_callLocalVoid {ps : List ClifTy} {res : Option ClifTy} {r : LocalRef ps res} {args : Vals Slot ps}
    {k : ResV Slot res → Prog Slot Lvl α} {Q : α → Env → World → Prop} {vs : List V} {x : Option V}
    {w' : World} (hres : res.isSome = false) (hvs : args.slots.mapM (fun r => Γ[r]?) = some vs)
    (hc : callOf cfg.locals r.callee vs (obsCall w r.callee vs) = some (x, w'))
    (h : wpE cfg (k (resSlot res 0)) Q Γ w') : wpE cfg (.callLocal r args k) Q Γ w :=
  PE.callLocalVoid hres (R := At Γ w')
    (fun n Γ0 w0 ⟨hn, e1, e2⟩ => by
      subst e1 e2
      exact ⟨_, _, by rw [runStmt_callVoid]; simp only [hvs, hc], hn, rfl, rfl⟩) h

theorem wpE_params {tys : List ClifTy} {k : Vals Slot tys → Prog Slot Lvl α} {Q : α → Env → World → Prop}
    (h : wpE cfg (k (carriesFrom 0 tys)) Q Γ w) : wpE cfg (.params tys k) Q Γ w :=
  PE.params h

/-- Reading the entry block's parameters spends nothing. -/
theorem wpE_entryParams {tys : List ClifTy} {f : Vals Slot tys → Prog Slot Lvl α}
    {Q : α → Env → World → Prop} (h : wpE cfg (f (carriesFrom 0 tys)) Q Γ w) :
    wpE cfg (entryParams tys >>= f) Q Γ w :=
  PE.params h

/-- **A body that ends ends as the code it ships**: `PE` of a body is `Ends` of
    `emit` of it. -/
theorem emit_ends {P : Env → World → Prop} {p : Body}
    {Q : Unit → Env → World → Prop} {params : List ClifTy} {P0 : Env → World → Prop}
    (h : PE cfg P (fnBody params p) Q)
    (hf : Fine (emitGo (fnBody params p) ⟨params.length, 0, [], [], [], none⟩).2)
    (h0 : ∀ Γ w, P0 Γ w → Γ.size = params.length ∧ P Γ w) :
    Ends cfg P0 (emit p params) (fun Γ w => Q () Γ w) := by
  have h1 := h ⟨params.length, 0, [], [], [], none⟩ P0 hf (endsE_nil h0)
  show Ends cfg P0 (emitGo (fnBody params p) ⟨params.length, 0, [], [], [], none⟩).2.flush.pieces.reverse _
  rw [St.out_eq_flush]
  exact Ends.conseq h1 (fun _ _ h => h) (fun Γ w ⟨_, _, _, hq⟩ => hq)

/-- A program of operations alone. -/
inductive OpsOnly : {α : Type} → Prog Slot Lvl α → Prop
  | ret {α : Type} (a : α) : OpsOnly (.ret a)
  | op {α : Type} {ty} (o : Op' Slot ty) (k : Slot ty → Prog Slot Lvl α) :
      (∀ r, OpsOnly (k r)) → OpsOnly (.op o k)

/-- What may end a region besides finishing: nothing, and no fault. -/
def J0 : Post := { ok := fun _ _ => True, faultOk := false }

/-- **A program of operations alone ends** wherever its weakest precondition
    holds with faults ruled out: every operation then answers. -/
theorem wpE_of_wp {p : Prog Slot Lvl α} (hp : OpsOnly p) :
    ∀ {Q : α → Env → World → Prop} {Γ : Env} {w : World}, wp cfg 0 J0 p Q Γ w → wpE cfg p Q Γ w := by
  induction hp with
  | ret a => intro Q Γ w h; exact wpE_ret h
  | op o k _ ih =>
      intro Q Γ w h
      rw [wp_op] at h
      obtain ⟨hn, hs⟩ := h
      cases he : evalOp w.mem Γ o.erase with
      | none => exact absurd (hn he) (by simp [J0])
      | some v => exact wpE_op he (ih _ (hs v he))

syntax "ops_only" : tactic
macro_rules
  | `(tactic| ops_only) => `(tactic| repeat (first | exact OpsOnly.ret _ | (refine OpsOnly.op _ _ ?_; intro _)))

end

/-! ## Operations -/

section
variable {cfg : Cfg} {Γ : Env} {w : World}

theorem wpE_iconst32u {k : Int} {Q : Slot .i32 → Env → World → Prop} (hk : 0 ≤ k ∧ k < 2 ^ 32 := by decide)
    (h : Q Γ.size (Γ.push (.sc .i32 (s32 (UInt32.ofNat k.toNat)))) w) : wpE cfg (iconst32 k) Q Γ w :=
  wpE_of_wp (by ops_only) (wp_iconst32u (d := 0) (J := J0) hk h)

theorem wpE_iconst64 {k : Int} {Q : Slot .i64 → Env → World → Prop}
    (h : Q Γ.size (Γ.push (.sc .i64 (c64 k))) w) : wpE cfg (iconst64 k) Q Γ w :=
  wpE_of_wp (by ops_only) (wp_iconst64 (d := 0) (J := J0) h)

theorem wpE_fconst32 {bits : UInt64} {Q : Slot .f32 → Env → World → Prop}
    (h : Q Γ.size (Γ.push (.sc .f32 (bits &&& widthMask .f32))) w) : wpE cfg (fconst .f32 bits) Q Γ w :=
  wpE_of_wp (by ops_only) (wp_fconst32 (d := 0) (J := J0) h)

theorem wpE_loadF32 {a : Slot .i64} {p v : UInt64} {Q : Slot .f32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 p)) (hl : w.mem.load p 4 = some v)
    (h : Q Γ.size (Γ.push (.sc .f32 v)) w) : wpE cfg (loadF32 a) Q Γ w :=
  wpE_of_wp (by ops_only) (wp_loadF32 (d := 0) (J := J0) ha hl h)

theorem wpE_iaddImm {a : Slot .i64} {x : UInt64} {imm : Int} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x))
    (h : Q (Γ.push (.sc .i64 (c64 imm))).size ((Γ.push (.sc .i64 (c64 imm))).push (.sc .i64 (x + c64 imm))) w) :
    wpE cfg (iaddImm a imm) Q Γ w :=
  wpE_of_wp (by ops_only) (wp_iaddImm (d := 0) (J := J0) ha h)

theorem wpE_iadd {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : Q Γ.size (Γ.push (.sc .i64 (x + y))) w) : wpE cfg (iadd a b) Q Γ w :=
  wpE_of_wp (by ops_only) (wp_iadd (d := 0) (J := J0) ha hb h)

theorem wpE_iadd32 {a b : Slot .i32} {x y : UInt32} {Q : Slot .i32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 (s32 x))) (hb : Γ[b]? = some (.sc .i32 (s32 y)))
    (h : Q Γ.size (Γ.push (.sc .i32 (s32 (x + y)))) w) : wpE cfg (iadd a b) Q Γ w :=
  wpE_of_wp (by ops_only) (wp_iadd32 (d := 0) (J := J0) ha hb h)

theorem wpE_imul32 {a b : Slot .i32} {x y : UInt32} {Q : Slot .i32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 (s32 x))) (hb : Γ[b]? = some (.sc .i32 (s32 y)))
    (h : Q Γ.size (Γ.push (.sc .i32 (s32 (x * y)))) w) : wpE cfg (imul a b) Q Γ w :=
  wpE_of_wp (by ops_only) (wp_imul32 (d := 0) (J := J0) ha hb h)

theorem wpE_band32 {a b : Slot .i32} {x y : UInt32} {Q : Slot .i32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 (s32 x))) (hb : Γ[b]? = some (.sc .i32 (s32 y)))
    (h : Q Γ.size (Γ.push (.sc .i32 (s32 (x &&& y)))) w) : wpE cfg (band a b) Q Γ w :=
  wpE_of_wp (by ops_only) (wp_band32 (d := 0) (J := J0) ha hb h)

theorem wpE_ushrImm32 {a : Slot .i32} {x : UInt32} {imm : Int} {Q : Slot .i32 → Env → World → Prop}
    (hi : 0 ≤ imm ∧ imm < 32 := by decide) (ha : Γ[a]? = some (.sc .i32 (s32 x)))
    (h : Q (Γ.push (.sc .i64 (c64 imm))).size
      ((Γ.push (.sc .i64 (c64 imm))).push (.sc .i32 (s32 (x >>> UInt32.ofNat imm.toNat)))) w) :
    wpE cfg (ushrImm a imm) Q Γ w :=
  wpE_of_wp (by ops_only) (wp_ushrImm32 (d := 0) (J := J0) hi ha h)

theorem wpE_uext32 {a : Slot .i32} {x : UInt32} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 (s32 x))) (h : Q Γ.size (Γ.push (.sc .i64 (s32 x))) w) :
    wpE cfg (uextend64 a) Q Γ w :=
  wpE_of_wp (by ops_only) (wp_uext32 (d := 0) (J := J0) ha h)

theorem wpE_ishlImm {a : Slot .i64} {x : UInt64} {imm : Int} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x))
    (h : Q (Γ.push (.sc .i64 (c64 imm))).size
      ((Γ.push (.sc .i64 (c64 imm))).push (.sc .i64 (x <<< (c64 imm % 64)))) w) :
    wpE cfg (ishlImm a imm) Q Γ w :=
  wpE_of_wp (by ops_only) (wp_ishlImm (d := 0) (J := J0) ha h)

theorem wpE_bitcastI32 {a : Slot .f32} {x : UInt32} {Q : Slot .i32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f32 (s32 x))) (h : Q Γ.size (Γ.push (.sc .i32 (s32 x))) w) :
    wpE cfg (bitcast .i32 a) Q Γ w :=
  wpE_of_wp (by ops_only) (wp_bitcastI32 (d := 0) (J := J0) ha h)

theorem wpE_bitcastF32 {a : Slot .i32} {x : UInt32} {Q : Slot .f32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 (s32 x))) (h : Q Γ.size (Γ.push (.sc .f32 (s32 x))) w) :
    wpE cfg (bitcast .f32 a) Q Γ w :=
  wpE_of_wp (by ops_only) (wp_bitcastF32 (d := 0) (J := J0) ha h)

theorem wpE_fadd32 {a b : Slot .f32} {x y : UInt64} {Q : Slot .f32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f32 x)) (hb : Γ[b]? = some (.sc .f32 y))
    (h : Q Γ.size (Γ.push (.sc .f32 (w.mem.fl .f32 (ofF32 (f32 x + f32 y))))) w) : wpE cfg (fadd a b) Q Γ w :=
  wpE_of_wp (by ops_only) (wp_fadd32 (d := 0) (J := J0) ha hb h)

theorem wpE_fmul32 {a b : Slot .f32} {x y : UInt64} {Q : Slot .f32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f32 x)) (hb : Γ[b]? = some (.sc .f32 y))
    (h : Q Γ.size (Γ.push (.sc .f32 (w.mem.fl .f32 (ofF32 (f32 x * f32 y))))) w) : wpE cfg (fmul a b) Q Γ w :=
  wpE_of_wp (by ops_only) (wp_fmul32 (d := 0) (J := J0) ha hb h)

theorem wpE_fdiv32 {a b : Slot .f32} {x y : UInt64} {Q : Slot .f32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f32 x)) (hb : Γ[b]? = some (.sc .f32 y))
    (h : Q Γ.size (Γ.push (.sc .f32 (w.mem.fl .f32 (ofF32 (f32 x / f32 y))))) w) : wpE cfg (fdiv a b) Q Γ w :=
  wpE_of_wp (by ops_only) (wp_fdiv32 (d := 0) (J := J0) ha hb h)

theorem wpE_fneg32 {a : Slot .f32} {x : UInt64} {Q : Slot .f32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f32 x)) (h : Q Γ.size (Γ.push (.sc .f32 (x ^^^ 0x80000000))) w) :
    wpE cfg (fneg a) Q Γ w :=
  wpE_of_wp (by ops_only) (wp_fneg32 (d := 0) (J := J0) ha h)

theorem wpE_sqrt32 {a : Slot .f32} {x : UInt64} {Q : Slot .f32 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f32 x)) (h : Q Γ.size (Γ.push (.sc .f32 (w.mem.fl .f32 (funOp .sqrt .f32 x) &&& widthMask .f32))) w) :
    wpE cfg (sqrt a) Q Γ w :=
  wpE_of_wp (by ops_only) (wp_sqrt32 (d := 0) (J := J0) ha h)

theorem wpE_fcmp32 {c : FloatCC} {a b : Slot .f32} {x y : UInt64} {Q : Slot .i8 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .f32 x)) (hb : Γ[b]? = some (.sc .f32 y))
    (h : Q Γ.size (Γ.push (boolV (fcmpBits c .f32 x y))) w) : wpE cfg (fcmp c a b) Q Γ w :=
  wpE_of_wp (by ops_only) (wp_fcmp32 (d := 0) (J := J0) ha hb h)

theorem wpE_selectF32 {c : Slot .i8} {a b : Slot .f32} {p : Bool} {x y : UInt64}
    {Q : Slot .f32 → Env → World → Prop}
    (hc : Γ[c]? = some (boolV p)) (ha : Γ[a]? = some (.sc .f32 x)) (hb : Γ[b]? = some (.sc .f32 y))
    (h : Q Γ.size (Γ.push (.sc .f32 (if p then x else y))) w) : wpE cfg (select c a b) Q Γ w :=
  wpE_of_wp (by ops_only) (wp_selectF32 (d := 0) (J := J0) hc ha hb h)

theorem wpE_storeF32 {v : Slot .f32} {a : Slot .i64} {x p : UInt64} {m : Mem} {Q : Unit → Env → World → Prop}
    (hv : Γ[v]? = some (.sc .f32 x)) (ha : Γ[a]? = some (.sc .i64 p)) (hs : w.mem.store p 4 x = some m)
    (h : Q () Γ { obsStore w p 4 x with mem := m }) : wpE cfg (store v a) Q Γ w :=
  wpE_store (by simp [runStmt, Sem.get, hv, ha, hs, tyBytes, ClifTy.width]) (wpE_ret h)

end

/-! ## Branches and loops -/

section
variable {cfg : Cfg} {Γ : Env} {w : World}

theorem OpsOnly.bind {α β : Type} {p : Prog Slot Lvl α} {f : α → Prog Slot Lvl β} (hp : OpsOnly p)
    (hf : ∀ a, OpsOnly (f a)) : OpsOnly (p >>= f) := by
  induction hp with
  | ret a => exact hf a
  | op o k _ ih => exact OpsOnly.op o _ fun r => ih r

/-- **A branch on two integer slots ends** where the arm the test picks ends
    in `Qj` with its exports in scope. -/
theorem wpE_ifteT {t : ClifTy} {jTys : List ClifTy} {cc : ICmpCond} {a b : Slot t} {x y : UInt64}
    {thn els : Prog Slot Lvl (Vals Slot jTys)} {Q : Vals Slot jTys → Env → World → Prop}
    (Qj : Vals Slot jTys → Env → World → Prop) (hti : t.isInt = true)
    (ha : Γ[a]? = some (.sc t x)) (hb : Γ[b]? = some (.sc t y))
    (ht : cmpInt cc t x y = true → wpE cfg thn Qj (Γ.push (boolV true)) w)
    (he : cmpInt cc t x y = false → ∀ ne, (Γ.push (boolV false)).size ≤ ne →
      wpE cfg els Qj (bindAt (Γ.push (boolV false)) ne []) w)
    (hj : ∀ jv Γ' w', Qj jv Γ' w' → ∃ vs, jv.slots.mapM (fun r => Γ'[r]?) = some vs)
    (hk : ∀ nJ Γ' jv vs w', Γ'.size ≤ nJ → jv.slots.mapM (fun r => Γ'[r]?) = some vs → Qj jv Γ' w' →
      Q (carriesFrom nJ jTys) (bindAt Γ' nJ vs) w') :
    wpE cfg (ifte cc a b thn els) Q Γ w := by
  have hev : evalOp w.mem Γ (.icmp cc a b) = some (boolV (cmpInt cc t x y)) := by
    simp [evalOp, Sem.get, ha, hb, zipIntCmp, hti]
  unfold wpE ifte
  refine PE.ite Qj Qj ?_ ?_ ?_ ?_ hj hj
  · refine PE.of_wpE fun Γ' w' ⟨Γ0, t', f, e, ⟨e1, e2⟩, hc, hf⟩ => ?_
    subst e1 e2 e
    rw [hev] at hc
    have hb' : V.sc t' f = boolV (cmpInt cc t x y) := (Option.some.inj hc).symm
    cases hcmp : cmpInt cc t x y
    · rw [hcmp] at hb'; simp [boolV] at hb'; obtain ⟨rfl, rfl⟩ := hb'; simp at hf
    · rw [hb', hcmp]; exact ht hcmp
  · intro ne
    refine PE.of_wpE fun Γ' w' ⟨Γ0, t', f, hne, e, ⟨e1, e2⟩, hc, hf⟩ => ?_
    subst e1 e2 e
    rw [hev] at hc
    have hb' : V.sc t' f = boolV (cmpInt cc t x y) := (Option.some.inj hc).symm
    cases hcmp : cmpInt cc t x y
    · rw [hb', hcmp]; exact he hcmp ne (by simpa using hne)
    · rw [hcmp] at hb'; simp [boolV] at hb'; obtain ⟨rfl, rfl⟩ := hb'; simp at hf
  · intro nJ
    exact PE.ret fun Γ' w' ⟨Γ'', jv, vs, hle, e, hvs, hq⟩ => by
      subst e; exact hk nJ Γ'' jv vs w' hle hvs (hq.elim id id)
  · rintro Γ' w' ⟨rfl, rfl⟩
    rw [hev]
    unfold boolV
    exact ⟨_, _, rfl⟩

/-- **A loop of any carries ends**, by an invariant of the carries and the
    world and a measure of them each trip lowers. The head runs with the
    carries bound after the loop's environment `Γ`, and its test evaluates;
    where it leaves, the values it leaves with are in scope and `X` holds of
    them; otherwise the body ends with next carries in scope that meet the
    invariant and lower the measure from where the trip began. What follows
    sees the values left with. -/
theorem wpE_wloopN {tys exitTys : List ClifTy} {β : Type} {init : Vals Slot tys}
    {head : Vals Slot tys → Prog Slot Lvl (Cond Slot × Vals Slot exitTys × β)}
    {body : Vals Slot tys → β → Prog Slot Lvl (Vals Slot tys)}
    {Q : Vals Slot exitTys → Env → World → Prop}
    (Inv : List V → World → Prop) (μ : List V → World → Nat)
    (Hd : List V → World → Cond Slot × Vals Slot exitTys × β → Env → World → Prop)
    (X : List V → World → Prop)
    (h0 : ∃ cs, init.slots.mapM (fun r => Γ[r]?) = some cs ∧ Inv cs w)
    (hh : ∀ (wi : World) (cs : List V), Inv cs wi →
      wpE cfg (head (carriesFrom Γ.size tys))
        (fun a Γ1 w1 => Extends (bindAt Γ Γ.size cs) Γ1 ∧ Hd cs wi a Γ1 w1) (bindAt Γ Γ.size cs) wi)
    (hc : ∀ cs wi a Γ1 w1, Hd cs wi a Γ1 w1 →
      ∃ t f, evalOp w1.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) = some (.sc t f))
    (hx : ∀ cs wi a Γ1 w1 t f, Hd cs wi a Γ1 w1 → evalOp w1.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) = some (.sc t f) →
      ((f != 0) == a.1.exitOnTrue) = true →
      ∃ vs, a.2.1.slots.mapM (fun r => (Γ1.push (.sc t f))[r]?) = some vs ∧ X vs w1)
    (hb : ∀ cs wi a Γ1 wb t f, Hd cs wi a Γ1 wb → Extends (bindAt Γ Γ.size cs) Γ1 →
      evalOp wb.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) = some (.sc t f) → ((f != 0) == a.1.exitOnTrue) = false →
      wpE cfg (body (carriesFrom (Γ1.size + 1) tys) a.2.2)
        (fun next Γ3 w3 => ∃ nx, next.slots.mapM (fun r => Γ3[r]?) = some nx ∧ Inv nx w3 ∧ μ nx w3 < μ cs wi)
        (bindAt (Γ1.push (.sc t f)) (Γ1.size + 1) cs) wb)
    (hk : ∀ Γb vs w' nE, Extends Γ Γb → Γ.size ≤ nE → X vs w' →
      Q (carriesFrom nE exitTys) (bindAt Γb nE vs) w') :
    wpE cfg (wloop init head body) Q Γ w := by
  unfold wpE wloop wloopL
  refine PE.loop (fun Γ0 cs w' => Γ0 = Γ ∧ Inv cs w') (fun _ cs w => μ cs w)
    (fun Γ0 cs wi a Γ1 w1 => Γ0 = Γ ∧ Extends (bindAt Γ Γ.size cs) Γ1 ∧ Hd cs wi a Γ1 w1)
    (fun Γ0 Γb vs w' => Γ0 = Γ ∧ Extends Γ Γb ∧ X vs w') ?_ ?_ ?_ ?_ ?_ ?_
  · rintro Γ' w' ⟨rfl, rfl⟩
    obtain ⟨cs, hcs, hI⟩ := h0
    exact ⟨cs, hcs, rfl, hI⟩
  · intro lv n0 Γ0 cs w0
    refine PE.of_wpE fun Γh wh ⟨hn, e, ew, e0, hinv⟩ => ?_
    subst e0 hn e ew
    exact wpE_mono (fun a Γ1 w1 ⟨hE, hd⟩ => ⟨rfl, hE, hd⟩) (hh wh cs hinv)
  · rintro Γ0 cs wi a Γ1 w1 ⟨-, -, hd⟩
    exact hc cs wi a Γ1 w1 hd
  · rintro Γ0 cs wi a Γ1 w1 t f ⟨e0, hE, hd⟩ hev hxx
    subst e0
    obtain ⟨vs, hvs, hX⟩ := hx cs wi a Γ1 w1 t f hd hev hxx
    exact ⟨vs, hvs, rfl, (((Extends.refl _).bindAt_of (Nat.le_refl _) cs).trans hE).push_of _, hX⟩
  · intro lv nb Γ0 cs wi a
    refine PE.of_wpE fun Γb wb ⟨Γ1, t, f, hnb, e, ⟨e0, hE, hd⟩, hev, hxx⟩ => ?_
    subst e0 hnb e
    exact wpE_mono (fun next Γ3 w3 ⟨nx, hnx, hI, hμ⟩ => ⟨nx, hnx, ⟨rfl, hI⟩, hμ⟩)
      (hb cs wi a Γ1 wb t f hd hE hev hxx)
  · intro nE
    exact PE.ret fun Γ' w' ⟨Γ0, Γb, vs, hle, e, e0, hE, hX⟩ => by
      subst e e0; exact hk Γb vs w' nE hE hle hX

end

end AlgorithmLib.LibSpec
