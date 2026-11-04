module
public import AlgorithmLib.Proof.Loops
meta import AlgorithmLib.Proof.Loops
public import AlgorithmLib.Proof.ProgHoare
meta import AlgorithmLib.Proof.ProgHoare
public import AlgorithmLib.Host.Heap
meta import AlgorithmLib.Host.Heap
public import AlgorithmLib.Surface.LibBase
meta import AlgorithmLib.Surface.LibBase
public import AlgorithmLib.Host.Summary
meta import AlgorithmLib.Host.Summary
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Proof.Rules` — one rule for each construct of a body

The weakest-precondition steps every proof over a body uses, whatever it
states: slots (`get_push`, `get_bindAt`, the `slot` tactics), values
(`ofInt64`, `c64_nat`), one rule per operation (`wp_iconst64`, `wp_iadd`,
`wp_load64`, `wp_storeI64`, `wp_val`, `wp_icmp`, …), branches and their
joins (`wp_ifte`, `wp_ifteT`, `wp_ifteV`, `join1`), and loops (`wp_forLoop`,
`wp_forLoopAcc`, `wp_wloop1`, `wp_wloopN`). `Proof.LibSpec` states library
functions with them and `Proof.Typestate` typestates.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.HProg.Contracts

namespace AlgorithmLib.LibSpec

/-- The world after an eight-byte store of `x` at `p` left memory `m`. -/
abbrev stored (w : World) (p : UInt64) (x : UInt64) (m : Mem) : World := { obsStore w p 8 x with mem := m }

-- ---------------------------------------------------------------------------
-- Slots
-- ---------------------------------------------------------------------------

theorem get_push {Γ : Env} {i : Nat} {v x : V} (h : Γ[i]? = some x) : (Γ.push v)[i]? = some x := by
  have hi : i < Γ.size := by
    rcases Nat.lt_or_ge i Γ.size with hi | hi
    · exact hi
    · rw [Array.getElem?_eq_none hi] at h; cases h
  rw [Array.getElem?_push_lt hi, ← h, Array.getElem?_eq_getElem hi]

theorem get_push_size {Γ : Env} {v : V} : (Γ.push v)[Γ.size]? = some v := by simp

theorem get_bindAt {Γ : Env} {n i : Nat} {vs : List V} {x : V} (hn : Γ.size ≤ n) (h : Γ[i]? = some x) :
    (bindAt Γ n vs)[i]? = some x := by
  have hi : i < Γ.size := by
    rcases Nat.lt_or_ge i Γ.size with hi | hi
    · exact hi
    · rw [Array.getElem?_eq_none hi] at h; cases h
  rw [bindAt_lt (by omega) hi, ← h]

syntax "slot_walk" : tactic
macro_rules
  | `(tactic| slot_walk) => `(tactic| first
      | with_reducible assumption
      | with_reducible exact get_push_size
      | (with_reducible apply get_push; slot_walk)
      | (with_reducible refine get_bindAt (by with_reducible assumption) ?_; slot_walk)
      | (with_reducible refine Extends.get (by with_reducible assumption) ?_; slot_walk))

/-- An environment built by pushes and joins holds what the one it was built
    on holds. -/
syntax "ext_env" : tactic
macro_rules
  | `(tactic| ext_env) => `(tactic| first
      | with_reducible assumption
      | with_reducible exact Extends.refl _
      | (with_reducible refine Extends.push_of ?_ _; ext_env)
      | (with_reducible refine Extends.bindAt_of ?_ (by with_reducible assumption) _; ext_env))

/-- Read a slot of an environment built by pushes and joins: the slot's name
    is the size of the environment it was pushed onto, so the walk compares
    names as written. -/
macro "slot" : tactic => `(tactic| ((try simp only [Vals.head, Vals.tail, Vals.snd, Vals.thd, Vals.fth,
  Vals.fif, Nat.zero_add, Nat.reduceAdd]); slot_walk))

-- ---------------------------------------------------------------------------
-- Values
-- ---------------------------------------------------------------------------

theorem ofInt64 (k : Int) : ofInt .i64 k = .sc .i64 (c64 k) := norm64 _

theorem c64_nat {k : Nat} (hk : k < 2 ^ 64) : c64 (k : Int) = UInt64.ofNat k := by
  unfold c64
  congr 1
  show ((k : Int) % ((18446744073709551616 : Nat) : Int)).toNat = k
  omega

-- ---------------------------------------------------------------------------
-- One step of a body
-- ---------------------------------------------------------------------------

section steps
variable {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}

theorem wp_iconst64 {k : Int} {Q : Slot .i64 → Env → World → Prop}
    (h : Q Γ.size (Γ.push (.sc .i64 (c64 k))) w) : wp cfg d J (iconst64 k) Q Γ w := by
  simp only [iconst64, iconst, op, wp_op, wp_ret, Op'.erase, evalOp, ofInt64, Option.some.injEq, forall_eq',
    reduceCtorEq, false_implies, true_and]
  exact h

theorem wp_iadd {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : Q Γ.size (Γ.push (.sc .i64 (x + y))) w) : wp cfg d J (iadd a b) Q Γ w := by
  simp [iadd, op, wp_op, wp_ret, Op'.erase, evalOp, bin, Sem.get, ha, hb, norm64]
  exact ⟨fun h => absurd h (by decide), fun _ => h⟩

theorem wp_iaddImm {a : Slot .i64} {x : UInt64} {imm : Int} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x))
    (h : Q (Γ.push (.sc .i64 (c64 imm))).size ((Γ.push (.sc .i64 (c64 imm))).push (.sc .i64 (x + c64 imm))) w) :
    wp cfg d J (iaddImm a imm) Q Γ w := by
  unfold iaddImm
  rw [wp_bind]
  exact wp_iconst64 (wp_iadd (get_push ha) get_push_size h)

/-- The address of the field at `off`. -/
theorem wp_at {a : Slot .i64} {x : UInt64} {off : Nat} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x))
    (h : Q (Γ.push (.sc .i64 (UInt64.ofNat off))).size
      ((Γ.push (.sc .i64 (UInt64.ofNat off))).push (.sc .i64 (x + UInt64.ofNat off))) w)
    (hoff : off < 2 ^ 64 := by decide) :
    wp cfg d J (Lib.at_ a off) Q Γ w := by
  unfold Lib.at_
  refine wp_iaddImm ha ?_
  rw [c64_nat hoff]
  exact h

theorem wp_load64 {a : Slot .i64} {p v : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 p)) (hl : w.mem.load p 8 = some v)
    (h : Q Γ.size (Γ.push (.sc .i64 v)) w) : wp cfg d J (load64 a) Q Γ w := by
  simp [load64, load, op, wp_op, wp_ret, Op'.erase, evalOp, Sem.get, ha, hl, tyBytes, ClifTy.lanes, ClifTy.width]
  exact h

theorem wp_storeI64 {v a : Slot .i64} {x p : UInt64} {m : Mem} {Q : Unit → Env → World → Prop}
    (hv : Γ[v]? = some (.sc .i64 x)) (ha : Γ[a]? = some (.sc .i64 p)) (hs : w.mem.store p 8 x = some m)
    (h : Q () Γ (stored w p x m)) : wp cfg d J (storeI64 v a) Q Γ w := by
  simp [storeI64, storeW, wp_store, wp_ret, runStmt, Sem.get, hv, ha, hs, tyBytes, ClifTy.width]
  exact h

/-- A branch on two slots of an integer type `t`, both arms ending in `Qj`. -/
theorem wp_ifteT {t : ClifTy} {jTys : List ClifTy} {cc : ICmpCond} {a b : Slot t} {x y : UInt64}
    {thn els : Prog Slot Lvl (Vals Slot jTys)} {Q : Vals Slot jTys → Env → World → Prop}
    (Qj : Vals Slot jTys → Env → World → Prop) (hti : t.isInt = true)
    (ha : Γ[a]? = some (.sc t x)) (hb : Γ[b]? = some (.sc t y))
    (ht : cmpInt cc t x y = true → wp cfg d J thn Qj (Γ.push (boolV true)) w)
    (he : cmpInt cc t x y = false → ∀ ne, (Γ.push (boolV false)).size ≤ ne →
      wp cfg d J els Qj (bindAt (Γ.push (boolV false)) ne []) w)
    (hk : ∀ nJ Γ' jv vs w', Γ'.size ≤ nJ → jv.slots.mapM (fun r => Γ'[r]?) = some vs → Qj jv Γ' w' →
      Q (carriesFrom nJ jTys) (bindAt Γ' nJ vs) w') :
    wp cfg d J (ifte cc a b thn els) Q Γ w := by
  simp only [ifte, wp_ite, wp_ret]
  have hev : evalOp w.mem Γ (.icmp cc a b) = some (boolV (cmpInt cc t x y)) := by
    simp [evalOp, Sem.get, ha, hb, zipIntCmp, hti]
  refine ⟨Qj, Qj, ⟨?_, ?_, ?_⟩, fun nJ Γ' jv vs w' hle hvs hq => hk nJ Γ' jv vs w' hle hvs (hq.elim id id)⟩
  · intro t' f hc hf
    rw [hev] at hc
    have hb' : V.sc t' f = boolV (cmpInt cc t x y) := (Option.some.inj hc).symm
    cases hcmp : cmpInt cc t x y
    · rw [hcmp] at hb'; simp [boolV] at hb'; obtain ⟨rfl, rfl⟩ := hb'; simp at hf
    · rw [hb', hcmp]; exact ht hcmp
  · intro ne t' f hne hc hf
    rw [hev] at hc
    have hb' : V.sc t' f = boolV (cmpInt cc t x y) := (Option.some.inj hc).symm
    cases hcmp : cmpInt cc t x y
    · rw [hb', hcmp]; exact he hcmp ne (by simpa using hne)
    · rw [hcmp] at hb'; simp [boolV] at hb'; obtain ⟨rfl, rfl⟩ := hb'; simp at hf
  · rw [hev]; nofun

/-- A branch on two 64-bit slots, both arms ending in `Qj`. -/
theorem wp_ifte {jTys : List ClifTy} {cc : ICmpCond} {a b : Slot .i64} {x y : UInt64}
    {thn els : Prog Slot Lvl (Vals Slot jTys)} {Q : Vals Slot jTys → Env → World → Prop}
    (Qj : Vals Slot jTys → Env → World → Prop)
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (ht : cmpInt cc .i64 x y = true → wp cfg d J thn Qj (Γ.push (boolV true)) w)
    (he : cmpInt cc .i64 x y = false → ∀ ne, (Γ.push (boolV false)).size ≤ ne →
      wp cfg d J els Qj (bindAt (Γ.push (boolV false)) ne []) w)
    (hk : ∀ nJ Γ' jv vs w', Γ'.size ≤ nJ → jv.slots.mapM (fun r => Γ'[r]?) = some vs → Qj jv Γ' w' →
      Q (carriesFrom nJ jTys) (bindAt Γ' nJ vs) w') :
    wp cfg d J (ifte cc a b thn els) Q Γ w :=
  wp_ifteT Qj rfl ha hb ht he hk

/-- The one value a join of one slot carries. -/
theorem join1 {t : ClifTy} {Γ' : Env} {jv : Vals Slot [t]} {vs : List V} {x : V}
    (hvs : jv.slots.mapM (fun r => Γ'[r]?) = some vs) (hx : Γ'[jv.head]? = some x) : vs = [x] := by
  cases jv with
  | cons v t =>
    cases t
    simp only [Vals.slots, Vals.head] at hvs hx
    rw [List.mapM_cons, hx] at hvs
    simp at hvs
    exact hvs.symm

end steps

/-- Sequence the statements of the body at the head of the goal. -/
macro "wp_seq" : tactic => `(tactic| simp only [wp_bind, wp_pure, wp_ret])

/-- Close what evaluating an operation leaves: its side conditions. -/
macro "op_fin" : tactic => `(tactic| first
  | done
  | rfl
  | decide
  | (constructor <;> first | rfl | decide)
  | (intro h; exact absurd h (by decide)))

section ops
variable {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}

/-- An operation that evaluates. -/
theorem wp_val {ty : ClifTy} {o : Op' Slot ty} {v : V} {Q : Slot ty → Env → World → Prop}
    (he : evalOp w.mem Γ o.erase = some v) (h : Q Γ.size (Γ.push v) w) : wp cfg d J (op o) Q Γ w := by
  simp only [op, wp_op, wp_ret, he]
  exact ⟨nofun, fun v' hv => by cases hv; exact h⟩

theorem wp_isub {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : Q Γ.size (Γ.push (.sc .i64 (x - y))) w) : wp cfg d J (isub a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, bin, Sem.get, ha, hb, norm64] <;> op_fin) h

theorem wp_imul {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : Q Γ.size (Γ.push (.sc .i64 (x * y))) w) : wp cfg d J (imul a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, bin, Sem.get, ha, hb, norm64] <;> op_fin) h

theorem wp_band {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : Q Γ.size (Γ.push (.sc .i64 (x &&& y))) w) : wp cfg d J (band a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, zipIntBits, Sem.get, ha, hb, norm64] <;> op_fin) h

theorem wp_bor {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : Q Γ.size (Γ.push (.sc .i64 (x ||| y))) w) : wp cfg d J (bor a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, zipIntBits, Sem.get, ha, hb, norm64] <;> op_fin) h

theorem wp_bxor {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : Q Γ.size (Γ.push (.sc .i64 (x ^^^ y))) w) : wp cfg d J (bxor a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, zipIntBits, Sem.get, ha, hb, norm64] <;> op_fin) h

theorem wp_ishlImm {a : Slot .i64} {x : UInt64} {imm : Int} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x))
    (h : Q (Γ.push (.sc .i64 (c64 imm))).size
      ((Γ.push (.sc .i64 (c64 imm))).push (.sc .i64 (x <<< (c64 imm % 64)))) w) :
    wp cfg d J (ishlImm a imm) Q Γ w := by
  unfold ishlImm
  rw [wp_bind]
  apply wp_iconst64
  exact wp_val (by simp [Op'.erase, evalOp, shiftBin, Sem.get, get_push ha, norm64] <;> op_fin) h

theorem wp_ushrImm {a : Slot .i64} {x : UInt64} {imm : Int} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x))
    (h : Q (Γ.push (.sc .i64 (c64 imm))).size
      ((Γ.push (.sc .i64 (c64 imm))).push (.sc .i64 (x >>> (c64 imm % 64)))) w) :
    wp cfg d J (ushrImm a imm) Q Γ w := by
  unfold ushrImm
  rw [wp_bind]
  apply wp_iconst64
  exact wp_val (by simp [Op'.erase, evalOp, shiftBin, Sem.get, get_push ha, norm64, and_wide64] <;> op_fin) h

theorem wp_umax {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : Q Γ.size (Γ.push (.sc .i64 (if x ≤ y then y else x))) w) : wp cfg d J (umax a b) Q Γ w := by
  refine wp_val (o := .ibin .umax a b (by decide)) (v := .sc .i64 (if x ≤ y then y else x)) ?_ h
  simp only [Op'.erase, evalOp, bin, Sem.get, ha, hb, ibinOp]
  simp only [Option.bind_eq_bind, Option.bind_some, BEq.rfl, Bool.true_and]
  rw [if_pos (by rfl), norm64, and_wide64, and_wide64]
  congr 2
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by have := x.toNat_lt; have := y.toNat_lt; omega)]
  split
  · rename_i hle; exact Nat.max_eq_right (UInt64.le_iff_toNat_le.mp hle)
  · rename_i hle; exact Nat.max_eq_left (by have := (not_congr UInt64.le_iff_toNat_le).mp hle; omega)

/-- A comparison, as a value. -/
theorem wp_icmp {cc : ICmpCond} {a b : Slot .i64} {x y : UInt64} {Q : Slot .i8 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : Q Γ.size (Γ.push (boolV (cmpInt cc .i64 x y))) w) : wp cfg d J (icmp cc a b) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, zipIntCmp, Sem.get, ha, hb] <;> op_fin) h

theorem wp_bor8 {a b : Slot .i8} {p q : Bool} {Q : Slot .i8 → Env → World → Prop}
    (ha : Γ[a]? = some (boolV p)) (hb : Γ[b]? = some (boolV q))
    (h : Q Γ.size (Γ.push (boolV (p || q))) w) : wp cfg d J (bor a b) Q Γ w :=
  wp_val (by cases p <;> cases q <;> simp [Op'.erase, evalOp, zipIntBits, Sem.get, ha, hb, boolV] <;> op_fin) h

theorem wp_select {c : Slot .i8} {a b : Slot .i64} {p : Bool} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (hc : Γ[c]? = some (boolV p)) (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : Q Γ.size (Γ.push (.sc .i64 (if p then x else y))) w) : wp cfg d J (select c a b) Q Γ w :=
  wp_val (by cases p <;> simp [Op'.erase, evalOp, Sem.get, ha, hb, hc, boolV, Sem.isTrue, V.ty] <;> op_fin) h

theorem wp_uextend64 {a : Slot .i32} {x : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i32 x))
    (h : Q Γ.size (Γ.push (.sc .i64 (x &&& 0xFFFFFFFF))) w) : wp cfg d J (uextend64 a) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, un, Sem.get, ha] <;> op_fin) h

theorem wp_iconst8 {k : Int} {Q : Slot .i8 → Env → World → Prop}
    (h : Q Γ.size (Γ.push (ofInt .i8 k)) w) : wp cfg d J (iconst .i8 k) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp] <;> op_fin) h

theorem wp_iconst32 {k : Int} {Q : Slot .i32 → Env → World → Prop}
    (h : Q Γ.size (Γ.push (ofInt .i32 k)) w) : wp cfg d J (iconst32 k) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp] <;> op_fin) h

theorem wp_uload8 {a : Slot .i64} {p b : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 p)) (hl : w.mem.load p 1 = some b)
    (h : Q Γ.size (Γ.push (.sc .i64 b)) w) : wp cfg d J (uload8_64 a) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, Sem.get, ha, hl, norm64] <;> op_fin) h

theorem wp_umin {a b : Slot .i64} {x y : UInt64} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : Q Γ.size (Γ.push (.sc .i64 (if x ≤ y then x else y))) w) : wp cfg d J (umin a b) Q Γ w := by
  refine wp_val (o := .ibin .umin a b (by decide)) (v := .sc .i64 (if x ≤ y then x else y)) ?_ h
  simp only [Op'.erase, evalOp, bin, Sem.get, ha, hb, ibinOp]
  simp only [Option.bind_eq_bind, Option.bind_some, BEq.rfl, Bool.true_and]
  by_cases hxy : x ≤ y
  · have : (x &&& widthMask .i64).toNat ≤ (y &&& widthMask .i64).toNat := by
      rw [show widthMask .i64 = -1 from rfl, UInt64.and_neg_one, UInt64.and_neg_one]
      exact UInt64.le_iff_toNat_le.mp hxy
    simp only [ClifTy.isInt, if_true, Nat.min_eq_left this, if_pos hxy, norm, Option.some.injEq, V.sc.injEq,
      true_and]
    rw [show widthMask .i64 = -1 from rfl, UInt64.and_neg_one, UInt64.and_neg_one, UInt64.ofNat_toNat]
  · have : (y &&& widthMask .i64).toNat ≤ (x &&& widthMask .i64).toNat := by
      rw [show widthMask .i64 = -1 from rfl, UInt64.and_neg_one, UInt64.and_neg_one]
      have := UInt64.lt_iff_toNat_lt.mp (UInt64.not_le.mp hxy); omega
    simp only [ClifTy.isInt, if_true, Nat.min_eq_right this, if_neg hxy, norm, Option.some.injEq, V.sc.injEq,
      true_and]
    rw [show widthMask .i64 = -1 from rfl, UInt64.and_neg_one, UInt64.and_neg_one, UInt64.ofNat_toNat]

/-- A byte swap, by what `Sem.iunOp` answers for it. -/
theorem wp_bswap {a : Slot .i64} {x : UInt64} {v : V} {Q : Slot .i64 → Env → World → Prop}
    (ha : Γ[a]? = some (.sc .i64 x)) (hv : iunOp .bswap .i64 x = some v)
    (h : Q Γ.size (Γ.push v) w) : wp cfg d J (bswap a) Q Γ w :=
  wp_val (by simp [Op'.erase, evalOp, un, Sem.get, ha, hv, ClifTy.isInt]) h

end ops

-- ---------------------------------------------------------------------------
-- Loops
-- ---------------------------------------------------------------------------

/-- The first value a join binds, whatever the environment's size. -/
theorem bindAt_head (Γ : Env) (n : Nat) (x : V) (xs : List V) : (bindAt Γ n (x :: xs))[n]? = some x := by
  rw [bindAt_get?, if_neg (Nat.lt_irrefl n)]; simp

theorem bindAt_second (Γ : Env) (n : Nat) (x y : V) (xs : List V) :
    (bindAt Γ n (x :: y :: xs))[n + 1]? = some y := by
  rw [bindAt_get?, if_neg (by omega)]; simp

/-- The third value a join binds. -/
theorem bindAt_third (Γ : Env) (n : Nat) (x y z : V) (xs : List V) :
    (bindAt Γ n (x :: y :: z :: xs))[n + 2]? = some z := by
  rw [bindAt_get?, if_neg (by omega)]; simp

theorem ofNat_lt_iff {i : Nat} {n : UInt64} (hi : i ≤ n.toNat) : UInt64.ofNat i < n ↔ i < n.toNat := by
  rw [UInt64.lt_iff_toNat_lt, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by have := n.toNat_lt; omega)]

section loops
variable {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}

/-- **A counted loop threading an accumulator**: `Inv i a` of the trips made,
    the accumulator and the world, kept by a trip. -/
theorem wp_forLoopAcc {ta : ClifTy} {limit : Slot .i64} {acc0 : Slot ta} {n a0 : UInt64}
    {body : Slot .i64 → Slot ta → Prog Slot Lvl (Slot ta)} {Q : Slot ta → Env → World → Prop}
    (Inv : Nat → UInt64 → World → Prop)
    (hl : Γ[limit]? = some (.sc .i64 n)) (ha : Γ[acc0]? = some (.sc ta a0)) (h0 : Inv 0 a0 w)
    (hb : ∀ (J' : Post) (Γ' : Env) (wi : World) (i : Nat) (a : UInt64) (si : Slot .i64) (sa : Slot ta), Extends Γ Γ' →
      i < n.toNat → Inv i a wi → Γ'[si]? = some (.sc .i64 (UInt64.ofNat i)) → Γ'[sa]? = some (.sc ta a) →
      wp cfg (d + 1) J' (body si sa)
        (fun r Γ'' w'' => Extends Γ' Γ'' ∧ ∃ a', Γ''[r]? = some (.sc ta a') ∧ Inv (i + 1) a' w'') Γ' wi)
    (hk : ∀ Γ' w' r a, Extends Γ Γ' → Γ'[r]? = some (.sc ta a) → Inv n.toNat a w' → Q r Γ' w') :
    wp cfg d J (forLoopAcc limit acc0 body) Q Γ w := by
  unfold forLoopAcc wloop2 wloop wloopL
  rw [wp_bind]
  apply wp_iconst64
  have hl' : (Γ.push (.sc .i64 (c64 0)))[limit]? = some (.sc .i64 n) := get_push hl
  have ha' : (Γ.push (.sc .i64 (c64 0)))[acc0]? = some (.sc ta a0) := get_push ha
  have hc' : (Γ.push (.sc .i64 (c64 0)))[Γ.size]? = some (.sc .i64 (UInt64.ofNat 0)) := get_push_size
  have hext : Extends Γ (Γ.push (.sc .i64 (c64 0))) := (Extends.refl Γ).push_of _
  generalize Γ.push (.sc .i64 (c64 0)) = Γc at hl' ha' hc' hext ⊢
  rw [wp_bind, wp_loop]
  refine ⟨fun Γ0 cs w' => Γ0 = Γc ∧ ∃ i a, i ≤ n.toNat ∧ cs = [.sc .i64 (UInt64.ofNat i), .sc ta a] ∧ Inv i a w',
    fun Γ0 cs a Γ1 w1 => Γ0 = Γc ∧ Γ1 = bindAt Γc Γc.size cs ∧
      a = (contIfULt (ty := .i64) Γc.size limit, %[Γc.size + 1], ()) ∧
      ∃ i acc, i ≤ n.toNat ∧ cs = [.sc .i64 (UInt64.ofNat i), .sc ta acc] ∧ Inv i acc w1,
    fun Γ0 Γb vs w' => Γ0 = Γc ∧ Extends Γc Γb ∧ ∃ a, vs = [.sc ta a] ∧ Inv n.toNat a w', ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro cs hcs
    simp only [Vals.slots, List.mapM_cons, List.mapM_nil, hc', ha', Option.pure_def, Option.bind_eq_bind,
      Option.bind_some, Option.some.injEq] at hcs
    exact ⟨rfl, 0, a0, Nat.zero_le _, hcs.symm, h0⟩
  · rintro n0 Γ0 cs wh hn0 ⟨rfl, hI⟩
    subst hn0
    simp only [wp_pure]
    exact ⟨by first | rfl | trivial, by first | rfl | trivial, by first | rfl | trivial, hI⟩
  · rintro Γ0 cs a Γ1 w1 t f vs ⟨rfl, rfl, rfl, i, acc, hi, rfl, hinv⟩ hev hx hvs
    have hev' : evalOp w1.mem (bindAt Γ0 Γ0.size [.sc .i64 (UInt64.ofNat i), .sc ta acc])
        (.icmp .ult Γ0.size limit) = some (boolV (cmpInt .ult .i64 (UInt64.ofNat i) n)) := by
      simp [evalOp, Sem.get, bindAt_head, get_bindAt (Nat.le_refl _) hl', zipIntCmp]
      rfl
    simp only [contIfULt, contIf] at hev hx hvs
    rw [hev'] at hev
    have hnlt : ¬ i < n.toNat := by
      intro hlt
      have : cmpInt .ult .i64 (UInt64.ofNat i) n = true := by simpa [cmpInt] using (ofNat_lt_iff hi).mpr hlt
      rw [this] at hev
      simp [boolV] at hev
      obtain ⟨rfl, rfl⟩ := hev
      simp at hx
    have hin : i = n.toNat := by omega
    subst hin
    refine ⟨rfl, ((Extends.refl _).bindAt_of (Nat.le_refl _) _).push_of _, acc, ?_, hinv⟩
    simp only [Vals.slots, List.mapM_cons, List.mapM_nil, get_push (bindAt_second _ _ _ _ _), Option.pure_def,
      Option.bind_eq_bind, Option.bind_some, Option.some.injEq] at hvs
    exact hvs.symm
  · rintro nb Γ0 cs a Γ1 t f wb hnb ⟨rfl, rfl, rfl, i, acc, hi, rfl, hinv⟩ hev hx
    have hev' : evalOp wb.mem (bindAt Γ0 Γ0.size [.sc .i64 (UInt64.ofNat i), .sc ta acc])
        (.icmp .ult Γ0.size limit) = some (boolV (cmpInt .ult .i64 (UInt64.ofNat i) n)) := by
      simp [evalOp, Sem.get, bindAt_head, get_bindAt (Nat.le_refl _) hl', zipIntCmp]
      rfl
    simp only [contIfULt, contIf] at hev hx
    rw [hev'] at hev
    have hlt : i < n.toNat := by
      rcases Nat.lt_or_ge i n.toNat with hlt | hge
      · exact hlt
      · have : cmpInt .ult .i64 (UInt64.ofNat i) n = false := by
          simp only [cmpInt]
          exact decide_eq_false fun h => absurd ((ofNat_lt_iff hi).mp h) (by omega)
        rw [this] at hev
        simp [boolV] at hev
        obtain ⟨rfl, rfl⟩ := hev
        simp at hx
    simp only [wp_bind, wp_pure, carriesFrom, Vals.head, Vals.snd, Vals.tail]
    have hsz : ((bindAt Γ0 Γ0.size [V.sc .i64 (UInt64.ofNat i), .sc ta acc]).push (.sc t f)).size ≤ nb := by
      rw [Array.size_push]; omega
    have hE : Extends Γ (bindAt ((bindAt Γ0 Γ0.size [V.sc .i64 (UInt64.ofNat i), .sc ta acc]).push (.sc t f)) nb
        [.sc .i64 (UInt64.ofNat i), .sc ta acc]) :=
      (((hext.bindAt_of (Nat.le_refl _) _).push_of _).bindAt_of hsz _)
    refine wp_mono cfg _ _ _ _ _ ?_ _ _ (hb _ _ wb i acc nb (nb + 1) hE hlt hinv (bindAt_head _ _ _ _)
      (bindAt_second _ _ _ _ _))
    rintro r Γ'' w'' ⟨hE2, a', hr, hinv'⟩
    refine wp_iaddImm (hE2.get (bindAt_head _ _ _ _)) ?_
    intro nx hnx
    simp only [Vals.slots, List.mapM_cons, List.mapM_nil, get_push_size, get_push (get_push hr), Option.pure_def,
      Option.bind_eq_bind, Option.bind_some, Option.some.injEq] at hnx
    refine ⟨by first | rfl | trivial, i + 1, a', hlt, ?_, hinv'⟩
    rw [← hnx, UInt64.ofNat_add, show c64 1 = 1 from by decide]
    rfl
  · rintro nE Γ0 Γb vs w' hle ⟨rfl, hEb, a, rfl, hinv⟩
    simp only [wp_pure, wp_ret, carriesFrom, Vals.head]
    exact hk _ _ _ _ ((hext.trans hEb).bindAt (Nat.le_trans hext.size hle) _) (bindAt_head _ _ _ _) hinv
  · rintro Γ0 cs a Γ1 w1 ⟨rfl, rfl, rfl, i, acc, hi, rfl, hinv⟩ hev
    have hev' : evalOp w1.mem (bindAt Γ0 Γ0.size [.sc .i64 (UInt64.ofNat i), .sc ta acc])
        (.icmp .ult Γ0.size limit) = some (boolV (cmpInt .ult .i64 (UInt64.ofNat i) n)) := by
      simp [evalOp, Sem.get, bindAt_head, get_bindAt (Nat.le_refl _) hl', zipIntCmp]
      rfl
    simp only [contIfULt, contIf] at hev
    rw [hev'] at hev
    cases hev

/-- **A counted loop**: `Inv i` of the trips made and the world, kept by a trip. -/
theorem wp_forLoop {limit : Slot .i64} {n : UInt64} {body : Slot .i64 → Prog Slot Lvl Unit}
    {Q : Unit → Env → World → Prop} (Inv : Nat → World → Prop)
    (hl : Γ[limit]? = some (.sc .i64 n)) (h0 : Inv 0 w)
    (hb : ∀ (J' : Post) (Γ' : Env) (wi : World) (i : Nat) (si : Slot .i64), Extends Γ Γ' → i < n.toNat → Inv i wi →
      Γ'[si]? = some (.sc .i64 (UInt64.ofNat i)) →
      wp cfg (d + 1) J' (body si) (fun _ Γ'' w'' => Extends Γ' Γ'' ∧ Inv (i + 1) w'') Γ' wi)
    (hk : ∀ Γ' w', Extends Γ Γ' → Inv n.toNat w' → Q () Γ' w') :
    wp cfg d J (forLoop limit body) Q Γ w := by
  unfold forLoop wloop1 wloop wloopL
  rw [wp_bind]
  apply wp_iconst64
  have hl' : (Γ.push (.sc .i64 (c64 0)))[limit]? = some (.sc .i64 n) := get_push hl
  have hc' : (Γ.push (.sc .i64 (c64 0)))[Γ.size]? = some (.sc .i64 (UInt64.ofNat 0)) := get_push_size
  have hext : Extends Γ (Γ.push (.sc .i64 (c64 0))) := (Extends.refl Γ).push_of _
  generalize Γ.push (.sc .i64 (c64 0)) = Γc at hl' hc' hext ⊢
  rw [wp_bind, wp_loop]
  refine ⟨fun Γ0 cs w' => Γ0 = Γc ∧ ∃ i, i ≤ n.toNat ∧ cs = [.sc .i64 (UInt64.ofNat i)] ∧ Inv i w',
    fun Γ0 cs a Γ1 w1 => Γ0 = Γc ∧ Γ1 = bindAt Γc Γc.size cs ∧
      a = (contIfULt (ty := .i64) Γc.size limit, %[], ()) ∧
      ∃ i, i ≤ n.toNat ∧ cs = [.sc .i64 (UInt64.ofNat i)] ∧ Inv i w1,
    fun Γ0 Γb vs w' => Γ0 = Γc ∧ Extends Γc Γb ∧ vs = [] ∧ Inv n.toNat w', ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro cs hcs
    simp only [Vals.slots, List.mapM_cons, List.mapM_nil, hc', Option.pure_def, Option.bind_eq_bind,
      Option.bind_some, Option.some.injEq] at hcs
    exact ⟨rfl, 0, Nat.zero_le _, hcs.symm, h0⟩
  · rintro n0 Γ0 cs wh hn0 ⟨rfl, hI⟩
    subst hn0
    simp only [wp_pure]
    exact ⟨by first | rfl | trivial, by first | rfl | trivial, by first | rfl | trivial, hI⟩
  · rintro Γ0 cs a Γ1 w1 t f vs ⟨rfl, rfl, rfl, i, hi, rfl, hinv⟩ hev hx hvs
    have hev' : evalOp w1.mem (bindAt Γ0 Γ0.size [.sc .i64 (UInt64.ofNat i)])
        (.icmp .ult Γ0.size limit) = some (boolV (cmpInt .ult .i64 (UInt64.ofNat i) n)) := by
      simp [evalOp, Sem.get, bindAt_head, get_bindAt (Nat.le_refl _) hl', zipIntCmp]
      rfl
    simp only [contIfULt, contIf] at hev hx hvs
    rw [hev'] at hev
    have hnlt : ¬ i < n.toNat := by
      intro hlt
      have : cmpInt .ult .i64 (UInt64.ofNat i) n = true := by simpa [cmpInt] using (ofNat_lt_iff hi).mpr hlt
      rw [this] at hev
      simp [boolV] at hev
      obtain ⟨rfl, rfl⟩ := hev
      simp at hx
    have hin : i = n.toNat := by omega
    subst hin
    refine ⟨rfl, ((Extends.refl _).bindAt_of (Nat.le_refl _) _).push_of _, ?_, hinv⟩
    simp only [Vals.slots, List.mapM_nil, Option.pure_def, Option.some.injEq] at hvs
    exact hvs.symm
  · rintro nb Γ0 cs a Γ1 t f wb hnb ⟨rfl, rfl, rfl, i, hi, rfl, hinv⟩ hev hx
    have hev' : evalOp wb.mem (bindAt Γ0 Γ0.size [.sc .i64 (UInt64.ofNat i)])
        (.icmp .ult Γ0.size limit) = some (boolV (cmpInt .ult .i64 (UInt64.ofNat i) n)) := by
      simp [evalOp, Sem.get, bindAt_head, get_bindAt (Nat.le_refl _) hl', zipIntCmp]
      rfl
    simp only [contIfULt, contIf] at hev hx
    rw [hev'] at hev
    have hlt : i < n.toNat := by
      rcases Nat.lt_or_ge i n.toNat with hlt | hge
      · exact hlt
      · have : cmpInt .ult .i64 (UInt64.ofNat i) n = false := by
          simp only [cmpInt]
          exact decide_eq_false fun h => absurd ((ofNat_lt_iff hi).mp h) (by omega)
        rw [this] at hev
        simp [boolV] at hev
        obtain ⟨rfl, rfl⟩ := hev
        simp at hx
    simp only [wp_bind, wp_pure, carriesFrom, Vals.head]
    have hsz : ((bindAt Γ0 Γ0.size [V.sc .i64 (UInt64.ofNat i)]).push (.sc t f)).size ≤ nb := by
      rw [Array.size_push]; omega
    have hE : Extends Γ (bindAt ((bindAt Γ0 Γ0.size [V.sc .i64 (UInt64.ofNat i)]).push (.sc t f)) nb
        [.sc .i64 (UInt64.ofNat i)]) :=
      (((hext.bindAt_of (Nat.le_refl _) _).push_of _).bindAt_of hsz _)
    refine wp_mono cfg _ _ _ _ _ ?_ _ _ (hb _ _ wb i nb hE hlt hinv (bindAt_head _ _ _ _))
    rintro r Γ'' w'' ⟨hE2, hinv'⟩
    refine wp_iaddImm (hE2.get (bindAt_head _ _ _ _)) ?_
    intro nx hnx
    simp only [Vals.slots, List.mapM_cons, List.mapM_nil, get_push_size, Option.pure_def,
      Option.bind_eq_bind, Option.bind_some, Option.some.injEq] at hnx
    refine ⟨by first | rfl | trivial, i + 1, hlt, ?_, hinv'⟩
    rw [← hnx, UInt64.ofNat_add, show c64 1 = 1 from by decide]
    rfl
  · rintro nE Γ0 Γb vs w' hle ⟨rfl, hEb, rfl, hinv⟩
    simp only [wp_pure, wp_ret]
    exact hk _ _ ((hext.trans hEb).bindAt (Nat.le_trans hext.size hle) _) hinv
  · rintro Γ0 cs a Γ1 w1 ⟨rfl, rfl, rfl, i, hi, rfl, hinv⟩ hev
    have hev' : evalOp w1.mem (bindAt Γ0 Γ0.size [.sc .i64 (UInt64.ofNat i)])
        (.icmp .ult Γ0.size limit) = some (boolV (cmpInt .ult .i64 (UInt64.ofNat i) n)) := by
      simp [evalOp, Sem.get, bindAt_head, get_bindAt (Nat.le_refl _) hl', zipIntCmp]
      rfl
    simp only [contIfULt, contIf] at hev
    rw [hev'] at hev
    cases hev

end loops

section joins
variable {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}

/-- A branch on two 64-bit slots whose arms each answer one 64-bit value
    `R` holds of, and what follows it. -/
theorem wp_ifteV {cc : ICmpCond} {a b : Slot .i64} {x y : UInt64}
    {thn els : Prog Slot Lvl (Vals Slot [.i64])} {Q : Vals Slot [.i64] → Env → World → Prop}
    (R : UInt64 → World → Prop)
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (ht : cmpInt cc .i64 x y = true →
      wp cfg d J thn (fun jv Γ' w' => Extends Γ Γ' ∧ ∃ v, Γ'[jv.head]? = some (.sc .i64 v) ∧ R v w')
        (Γ.push (boolV true)) w)
    (he : cmpInt cc .i64 x y = false → ∀ ne, (Γ.push (boolV false)).size ≤ ne →
      wp cfg d J els (fun jv Γ' w' => Extends Γ Γ' ∧ ∃ v, Γ'[jv.head]? = some (.sc .i64 v) ∧ R v w')
        (bindAt (Γ.push (boolV false)) ne []) w)
    (hk : ∀ Γ' w' v (nJ : Nat), Extends Γ Γ' → Γ'[nJ]? = some (.sc .i64 v) → R v w' →
      Q (carriesFrom nJ [.i64]) Γ' w') :
    wp cfg d J (ifte cc a b thn els) Q Γ w := by
  refine wp_ifte (fun jv Γ' w' => Extends Γ Γ' ∧ ∃ v, Γ'[jv.head]? = some (.sc .i64 v) ∧ R v w') ha hb ht he ?_
  rintro nJ Γ' jv vs w' hle hvs ⟨hE, v, hv, hr⟩
  have hj := join1 hvs hv
  subst hj
  exact hk _ _ v nJ (hE.bindAt_of hle _) (bindAt_head _ _ _ _) hr

/-- **A loop of one 64-bit carry**, left where its head says: `Inv` of the
    carry and the world at each trip's start, `Hd` what the head leaves with
    its condition and exit values, `X` of the exit values. -/
theorem wp_wloop1 {exitTys : List ClifTy} {β : Type} {init : Slot .i64} {x0 : UInt64}
    {head : Slot .i64 → Prog Slot Lvl (Cond Slot × Vals Slot exitTys × β)}
    {body : Slot .i64 → β → Prog Slot Lvl (Vals Slot [.i64])}
    {Q : Vals Slot exitTys → Env → World → Prop}
    (Inv : UInt64 → World → Prop)
    (Hd : UInt64 → Cond Slot × Vals Slot exitTys × β → Env → World → Prop)
    (X : List V → World → Prop)
    (hi : Γ[init]? = some (.sc .i64 x0)) (h0 : Inv x0 w)
    (hh : ∀ (J' : Post) (Γ' : Env) (wi : World) (x : UInt64) (si : Slot .i64), Extends Γ Γ' → Inv x wi →
      Γ'[si]? = some (.sc .i64 x) →
      wp cfg (d + 1) J' (head si) (fun a Γ1 w1 => Extends Γ' Γ1 ∧ Hd x a Γ1 w1) Γ' wi)
    (hx : ∀ x a Γ1 w1 t f vs, Hd x a Γ1 w1 → evalOp w1.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) = some (.sc t f) →
      ((f != 0) == a.1.exitOnTrue) = true →
      a.2.1.slots.mapM (fun r => (Γ1.push (.sc t f))[r]?) = some vs → X vs w1)
    (hb : ∀ (J' : Post) x a Γ1 wb t f Γ2 (si : Slot .i64), Hd x a Γ1 wb → Extends Γ Γ1 →
      evalOp wb.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) = some (.sc t f) → ((f != 0) == a.1.exitOnTrue) = false →
      Extends Γ1 Γ2 → Γ2[si]? = some (.sc .i64 x) →
      wp cfg (d + 1) J' (body si a.2.2)
        (fun next Γ3 w3 => ∃ x', Γ3[next.head]? = some (.sc .i64 x') ∧ Inv x' w3) Γ2 wb)
    (hf : ∀ x a Γ1 w1, Hd x a Γ1 w1 → evalOp w1.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) ≠ none)
    (hk : ∀ Γb vs w' nE, Extends Γ Γb → Γ.size ≤ nE → X vs w' → Q (carriesFrom nE exitTys) (bindAt Γb nE vs) w') :
    wp cfg d J (wloop1 init head body) Q Γ w := by
  unfold wloop1 wloop wloopL
  rw [wp_loop]
  refine ⟨fun Γ0 cs w' => Γ0 = Γ ∧ ∃ x, cs = [.sc .i64 x] ∧ Inv x w',
    fun Γ0 cs a Γ1 w1 => Γ0 = Γ ∧ Extends Γ Γ1 ∧ ∃ x, cs = [.sc .i64 x] ∧ Hd x a Γ1 w1,
    fun Γ0 Γb vs w' => Γ0 = Γ ∧ Extends Γ Γb ∧ X vs w', ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro cs hcs
    simp only [Vals.slots, List.mapM_cons, List.mapM_nil, hi, Option.pure_def, Option.bind_eq_bind,
      Option.bind_some, Option.some.injEq] at hcs
    exact ⟨rfl, x0, hcs.symm, h0⟩
  · rintro n0 Γ0 cs wh hn0 ⟨rfl, x, rfl, hinv⟩
    subst hn0
    simp only [carriesFrom, Vals.head]
    refine wp_mono cfg _ _ _ _ _ ?_ _ _ (hh _ _ wh x Γ0.size ((Extends.refl _).bindAt_of (Nat.le_refl _) _) hinv
      (bindAt_head _ _ _ _))
    rintro a Γ1 w1 ⟨hE, hd⟩
    exact ⟨by first | rfl | trivial, Extends.trans ((Extends.refl _).bindAt_of (Nat.le_refl _) _) hE, x,
      by first | rfl | trivial, hd⟩
  · rintro Γ0 cs a Γ1 w1 t f vs ⟨rfl, hE, x, rfl, hd⟩ hev hxx hvs
    exact ⟨rfl, hE.push_of _, hx x a Γ1 w1 t f vs hd hev hxx hvs⟩
  · rintro nb Γ0 cs a Γ1 t f wb hnb ⟨rfl, hE, x, rfl, hd⟩ hev hxx
    subst hnb
    simp only [carriesFrom, Vals.head]
    refine wp_mono cfg _ _ _ _ _ ?_ _ _ (hb _ x a Γ1 wb t f _ (Γ1.size + 1) hd hE hev hxx
      (((Extends.refl _).push_of _).bindAt_of (by rw [Array.size_push]; omega) _) (bindAt_head _ _ _ _))
    rintro next Γ3 w3 ⟨x', hx', hinv⟩ nx hnx
    have hj := join1 hnx hx'
    subst hj
    exact ⟨by first | rfl | trivial, x', by first | rfl | trivial, hinv⟩
  · rintro nE Γ0 Γb vs w' hle ⟨rfl, hE, hX⟩
    simp only [wp_ret]
    exact hk _ _ _ _ hE hle hX
  · rintro Γ0 cs a Γ1 w1 ⟨rfl, hE, x, rfl, hd⟩ hev
    exact absurd hev (hf x a Γ1 w1 hd)

/-- The carries a loop binds, read back from where it bound them. -/
theorem bindAt_carry {Γ : Env} {n : Nat} {cs : List V} (hn : Γ.size ≤ n) (j : Nat) :
    (bindAt Γ n cs)[n + j]? = cs[j]? := by
  rw [bindAt_get?, if_neg (by omega)]; simp

/-- A body's next carries, read back once: what holds of the ones read holds
    of whatever reading gives. -/
theorem wloop_next {α : Type} {m : Option α} {P : α → Prop} (h : ∃ nx, m = some nx ∧ P nx) :
    ∀ nx, m = some nx → P nx := by
  obtain ⟨nx, hm, hp⟩ := h
  intro nx' h'
  rw [hm] at h'
  cases h'
  exact hp

/-- **A loop of any carries left where its head says**, by an invariant of
    the carries and the world. The head runs with the carries bound after the
    loop's environment `Γ`; where its condition leaves, `X` holds of the
    values it leaves with; otherwise the body runs with them bound after the
    flag, and its next carries meet the invariant. What follows sees the
    values left with. -/
theorem wp_wloopN {tys exitTys : List ClifTy} {β : Type} {init : Vals Slot tys}
    {head : Vals Slot tys → Prog Slot Lvl (Cond Slot × Vals Slot exitTys × β)}
    {body : Vals Slot tys → β → Prog Slot Lvl (Vals Slot tys)}
    {Q : Vals Slot exitTys → Env → World → Prop}
    (Inv : List V → World → Prop)
    (Hd : List V → Cond Slot × Vals Slot exitTys × β → Env → World → Prop)
    (X : List V → World → Prop)
    (h0 : ∀ cs, init.slots.mapM (fun r => Γ[r]?) = some cs → Inv cs w)
    (hh : ∀ (J' : Post) (wi : World) (cs : List V), Inv cs wi →
      wp cfg (d + 1) J' (head (carriesFrom Γ.size tys))
        (fun a Γ1 w1 => Extends (bindAt Γ Γ.size cs) Γ1 ∧ Hd cs a Γ1 w1) (bindAt Γ Γ.size cs) wi)
    (hx : ∀ cs a Γ1 w1 t f vs, Hd cs a Γ1 w1 → evalOp w1.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) = some (.sc t f) →
      ((f != 0) == a.1.exitOnTrue) = true →
      a.2.1.slots.mapM (fun r => (Γ1.push (.sc t f))[r]?) = some vs → X vs w1)
    (hb : ∀ (J' : Post) cs a Γ1 wb t f, Hd cs a Γ1 wb → Extends (bindAt Γ Γ.size cs) Γ1 →
      evalOp wb.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) = some (.sc t f) → ((f != 0) == a.1.exitOnTrue) = false →
      wp cfg (d + 1) J' (body (carriesFrom (Γ1.size + 1) tys) a.2.2)
        (fun next Γ3 w3 => ∀ nx, next.slots.mapM (fun r => Γ3[r]?) = some nx → Inv nx w3)
        (bindAt (Γ1.push (.sc t f)) (Γ1.size + 1) cs) wb)
    (hf : ∀ cs a Γ1 w1, Hd cs a Γ1 w1 → evalOp w1.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) ≠ none)
    (hk : ∀ Γb vs w' nE, Extends Γ Γb → Γ.size ≤ nE → X vs w' → Q (carriesFrom nE exitTys) (bindAt Γb nE vs) w') :
    wp cfg d J (wloop init head body) Q Γ w := by
  unfold wloop wloopL
  rw [wp_loop]
  refine ⟨fun Γ0 cs w' => Γ0 = Γ ∧ Inv cs w',
    fun Γ0 cs a Γ1 w1 => Γ0 = Γ ∧ Extends (bindAt Γ Γ.size cs) Γ1 ∧ Hd cs a Γ1 w1,
    fun Γ0 Γb vs w' => Γ0 = Γ ∧ Extends Γ Γb ∧ X vs w', ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro cs hcs
    exact ⟨rfl, h0 cs hcs⟩
  · rintro n0 Γ0 cs wh hn0 ⟨rfl, hinv⟩
    subst hn0
    refine wp_mono cfg _ _ _ _ _ ?_ _ _ (hh _ wh cs hinv)
    rintro a Γ1 w1 ⟨hE, hd⟩
    exact ⟨rfl, hE, hd⟩
  · rintro Γ0 cs a Γ1 w1 t f vs ⟨rfl, hE, hd⟩ hev hxx hvs
    exact ⟨rfl, (((Extends.refl _).bindAt_of (Nat.le_refl _) cs).trans hE).push_of _, hx cs a Γ1 w1 t f vs hd hev hxx hvs⟩
  · rintro nb Γ0 cs a Γ1 t f wb hnb ⟨rfl, hE, hd⟩ hev hxx
    subst hnb
    refine wp_mono cfg _ _ _ _ _ ?_ _ _ (hb _ cs a Γ1 wb t f hd hE hev hxx)
    rintro next Γ3 w3 h nx hnx
    exact ⟨rfl, h nx hnx⟩
  · rintro nE Γ0 Γb vs w' hle ⟨rfl, hE, hX⟩
    simp only [wp_ret]
    exact hk _ _ _ _ hE hle hX
  · rintro Γ0 cs a Γ1 w1 ⟨rfl, hE, hd⟩ hev
    exact absurd hev (hf cs a Γ1 w1 hd)

end joins

end AlgorithmLib.LibSpec
