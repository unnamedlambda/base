module
public import AlgorithmLib.Proof.ProgHoare
public import AlgorithmLib.Host.DevSpec
public import AlgorithmLib.Host.Lifecycle
public import AlgorithmLib.Host.ExtLife
public import AlgorithmLib.Host.FfiAnswers
public import AlgorithmLib.Host.Summary
meta import AlgorithmLib.Proof.ProgHoare
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Proof.Typestate` — generators proven without proof text

A generator's specification is computed, not written: `wp` is the weakest
precondition of any body, and `wp_sound` makes it a triple, so a generator is
safe when its weakest precondition holds. What a generator writer would
otherwise have to supply is the loop invariants and the facts each call
needs. Here both come from one predicate on the world, the *typestate* `W`:

* **Loops.** `wp_forLoop` takes `W` as the loop's invariant: a body that
  keeps `W` keeps it for any number of trips.
* **Calls.** `Moves cfg W f vs W'` is a call's contract in the typestate's
  terms: from a world in `W`, the call answers into a world in `W'`.
  `moves_of_pre` derives it for any entry point from its precondition in the
  contract table and what its answers leave. The standard typestates are lists
  of facts (`TState`): lifecycle parts, rooms (a region at least so large) and
  cells (what eight bytes hold). For them the table computes where a call
  lands (`after`), so `prog_keeps` has only the precondition left, which the
  facts give: the parts and cells by rewriting, the memory a call touches from
  the rooms. `Live` is one: the device calls' typestate.
* **Stores.** A store (of any width, or a byte) keeps every part and room and
  the cells it cannot reach,
  and a known eight-byte value stored to a cell becomes that cell's fact
  (`wp_store_state`).
* **Values.** What a slot holds is not a type index but a fact about the
  environment: a constant's slot holds that constant, an operation over known
  operands that reads no memory holds what it computes (`evalOp_renumber*`), a
  load from a cell the typestate names holds the cell's value, and the arguments a body starts with are known where the theorem says
  what they are (`safe_of_wp_entry`). An environment only grows (`Ext`), so what
  was bound before a loop or a call is still there after it. A generator reads
  a context handle back from its slot, and the typestate knows what it reads.

`prog_vc W` proves a body's weakest precondition from these, one construct at
a time from the front of the body, and `safe_of_wp` carries it to what ships.
It keeps every environment a variable and every slot's value a fact carried
forward only when next used, and it closes the proof in segments, so its work
is linear in the body. A program whose loop changes the typestate, or whose
safety depends on values the typestate does not name, needs an invariant of
its own: that is `wp_loop` with its witnesses, by hand.

Decisions the generator makes about facts (whether a typestate names a cell,
whether a loop's end names every fact its entry did) are made by the kernel,
which computes with numbers as numbers.
-/

open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare
open AlgorithmLib.HProg.DevSpec

namespace AlgorithmLib.HProg.Sem

local instance : LawfulBEq ClifTy where
  eq_of_beq {a b} h := by cases a <;> cases b <;> first | rfl | exact absurd h (by decide)
  rfl {a} := by cases a <;> decide

/-- Each slot holds a scalar of its type, in order. -/
def ArgsTy (Γ : Env) : List R → List ClifTy → Prop
  | [], [] => True
  | r :: rs, t :: ts => (∃ x, Γ[r]? = some (.sc t x)) ∧ ArgsTy Γ rs ts
  | _, _ => False

def isF (t : ClifTy) : Bool := t == .f32 || t == .f64

/-- Two operands of one integer type: that type. -/
def sameInt : List ClifTy → Option ClifTy
  | [t, u] => if t == u && t.isInt then some t else none
  | _ => none

/-- Two operands of one float type: that type. -/
def sameF : List ClifTy → Option ClifTy
  | [t, u] => if t == u && isF t then some t else none
  | _ => none

/-- The scalar type an operation answers over scalar operands of types `ts`, for
    the operations that answer whatever the operands' bits; `none` for the rest. -/
def tyOp : Op → List ClifTy → Option ClifTy
  | .iadd _ _, ts | .isub _ _, ts | .imul _ _, ts => sameInt ts
  | .band _ _, ts | .bandNot _ _, ts | .bor _ _, ts | .bxor _ _, ts => sameInt ts
  | .icmp _ _ _, ts => (sameInt ts).map fun _ => .i8
  | .fadd _ _, ts | .fsub _ _, ts | .fmul _ _, ts => sameF ts
  | .fmax _ _, ts | .fmin _ _, ts => sameF ts
  | .fcmp _ _ _, ts => (sameF ts).map fun _ => .i8
  | .ishl _ _, [t, u] | .ushr _ _, [t, u] | .ishift _ _ _, [t, u] =>
      if t.isInt && u.isInt then some t else none
  | .ineg _, [t] | .ctz _, [t] | .popcnt _, [t] => if t.isInt then some t else none
  | .ireduce32 _, [t] => if t.isInt && decide (t.width > 32) then some .i32 else none
  | .uextend64 _, [t] | .sextend64 _, [t] =>
      if t.isInt && decide (t.width < 64) then some .i64 else none
  | .fneg _, [t] | .fun1 _ _, [t] => if isF t then some t else none
  | .fpromote _, [t] => if t == .f32 then some .f64 else none
  | .fcvtFromSint ty _, [t] => if t.isInt && isF ty then some ty else none
  | .bitcast ty _, [t] => if t.width == ty.width then some ty else none
  | .iun k _, [t] => if k.admits t then some t else none
  | .iext k ty _, [t] => if k.admits t ty then some ty else none
  | .select _ _ _, [c, a, b] => if c.isInt && a == b then some a else none
  | _, _ => none

theorem sameInt_eq {ts : List ClifTy} {T : ClifTy} (h : sameInt ts = some T) :
    ts = [T, T] ∧ T.isInt = true := by
  unfold sameInt at h; split at h
  · simp only [Option.ite_none_right_eq_some, Option.some.injEq, Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨rfl, h2⟩, rfl⟩ := h; exact ⟨rfl, h2⟩
  · cases h

theorem sameF_eq {ts : List ClifTy} {T : ClifTy} (h : sameF ts = some T) :
    ts = [T, T] ∧ (T = .f32 ∨ T = .f64) := by
  unfold sameF at h; split at h
  · simp only [Option.ite_none_right_eq_some, Option.some.injEq, Bool.and_eq_true, beq_iff_eq,
      isF, Bool.or_eq_true] at h
    obtain ⟨⟨rfl, h2⟩, rfl⟩ := h; exact ⟨rfl, h2⟩
  · cases h

theorem args2 {Γ : Env} {a b : R} {t u : ClifTy} (h : ArgsTy Γ [a, b] [t, u]) :
    (∃ x, Γ[a]? = some (.sc t x)) ∧ ∃ y, Γ[b]? = some (.sc u y) := ⟨h.1, h.2.1⟩

theorem bin_sound {Γ : Env} {a b : R} {ts : List ClifTy} {T : ClifTy}
    {f : ClifTy → UInt64 → UInt64 → Option V} (hargs : ArgsTy Γ [a, b] ts) (h : sameInt ts = some T)
    (hf : ∀ x y, ∃ z, f T x y = some (.sc T z)) : ∃ z, bin Γ a b f = some (.sc T z) := by
  obtain ⟨rfl, hi⟩ := sameInt_eq h
  obtain ⟨⟨x, hx⟩, ⟨y, hy⟩⟩ := args2 hargs
  simpa [bin, get, hx, hy, hi] using hf x y

theorem zipIntBits_sound {Γ : Env} {a b : R} {ts : List ClifTy} {T : ClifTy}
    {f : ClifTy → UInt64 → UInt64 → UInt64} (hargs : ArgsTy Γ [a, b] ts) (h : sameInt ts = some T) :
    ∃ z, (do zipIntBits (← get Γ a) (← get Γ b) f) = some (.sc T z) := by
  obtain ⟨rfl, hi⟩ := sameInt_eq h
  obtain ⟨⟨x, hx⟩, ⟨y, hy⟩⟩ := args2 hargs
  simp [get, hx, hy, zipIntBits, hi, norm]

theorem icmp_sound {Γ : Env} {a b : R} {ts : List ClifTy} {T : ClifTy} {c : ICmpCond}
    (hargs : ArgsTy Γ [a, b] ts) (h : (sameInt ts).map (fun _ => ClifTy.i8) = some T) :
    ∃ z, (do zipIntCmp c (← get Γ a) (← get Γ b)) = some (.sc T z) := by
  obtain ⟨U, hU, rfl⟩ := Option.map_eq_some_iff.mp h
  obtain ⟨rfl, hi⟩ := sameInt_eq hU
  obtain ⟨⟨x, hx⟩, ⟨y, hy⟩⟩ := args2 hargs
  simp [get, hx, hy, zipIntCmp, hi, boolV]

theorem zipF_sound {Γ : Env} {a b : R} {ts : List ClifTy} {T : ClifTy} {f g}
    (hargs : ArgsTy Γ [a, b] ts) (h : sameF ts = some T) :
    ∃ z, (do zipF (← get Γ a) (← get Γ b) f g) = some (.sc T z) := by
  obtain ⟨rfl, hT⟩ := sameF_eq h
  obtain ⟨⟨x, hx⟩, ⟨y, hy⟩⟩ := args2 hargs
  rcases hT with rfl | rfl <;> simp [get, hx, hy, zipF]

theorem zipBits_sound {Γ : Env} {a b : R} {ts : List ClifTy} {T : ClifTy} {f}
    (hargs : ArgsTy Γ [a, b] ts) (h : sameF ts = some T) :
    ∃ z, (do zipBits (← get Γ a) (← get Γ b) f) = some (.sc T z) := by
  obtain ⟨rfl, hT⟩ := sameF_eq h
  obtain ⟨⟨x, hx⟩, ⟨y, hy⟩⟩ := args2 hargs
  rcases hT with rfl | rfl <;> simp [get, hx, hy, zipBits, zipBitsIf, ClifTy.isFloat]

theorem fcmp_sound {Γ : Env} {a b : R} {ts : List ClifTy} {T : ClifTy} {c : FloatCC}
    (hargs : ArgsTy Γ [a, b] ts) (h : (sameF ts).map (fun _ => ClifTy.i8) = some T) :
    ∃ z, evalOp m Γ (.fcmp c a b) = some (.sc T z) := by
  obtain ⟨U, hU, rfl⟩ := Option.map_eq_some_iff.mp h
  obtain ⟨rfl, hT⟩ := sameF_eq hU
  obtain ⟨⟨x, hx⟩, ⟨y, hy⟩⟩ := args2 hargs
  rcases hT with rfl | rfl <;> simp [evalOp, get, hx, hy, boolV]

theorem norm_sc (t : ClifTy) (x : UInt64) : ∃ z, norm t x = .sc t z := ⟨_, rfl⟩
theorem ofInt_sc (t : ClifTy) (i : Int) : ∃ z, ofInt t i = .sc t z := ⟨_, rfl⟩

theorem args1 {Γ : Env} {a : R} {ts : List ClifTy} (h : ArgsTy Γ [a] ts) :
    ∃ t, ts = [t] ∧ ∃ x, Γ[a]? = some (.sc t x) := by
  match ts, h with
  | [t], ⟨hx, _⟩ => exact ⟨t, rfl, hx⟩

theorem args2' {Γ : Env} {a b : R} {ts : List ClifTy} (h : ArgsTy Γ [a, b] ts) :
    ∃ t u, ts = [t, u] ∧ (∃ x, Γ[a]? = some (.sc t x)) ∧ ∃ y, Γ[b]? = some (.sc u y) := by
  match ts, h with
  | [t, u], ⟨hx, hy, _⟩ => exact ⟨t, u, rfl, hx, hy⟩

theorem args3 {Γ : Env} {a b c : R} {ts : List ClifTy} (h : ArgsTy Γ [a, b, c] ts) :
    ∃ t u v, ts = [t, u, v] ∧ (∃ x, Γ[a]? = some (.sc t x)) ∧ (∃ y, Γ[b]? = some (.sc u y)) ∧
      ∃ z, Γ[c]? = some (.sc v z) := by
  rcases ts with _ | ⟨t, _ | ⟨u, _ | ⟨v, _ | ⟨_, _⟩⟩⟩⟩ <;> simp only [ArgsTy, and_false, and_true] at h
  exact ⟨t, u, v, rfl, h.1, h.2.1, h.2.2⟩

set_option linter.unusedSimpArgs false in
/-- **An operation over scalars of the types `tyOp` asks answers a scalar of the
    type it names.** One lemma per shape; the cases share their tactic text, so
    a lemma one case needs is unused in another. -/
theorem tyOp_sound {m : Mem} {Γ : Env} {o : Op} {ts : List ClifTy} {T : ClifTy}
    (hargs : ArgsTy Γ o.regs ts) (h : tyOp o ts = some T) : ∃ x, evalOp m Γ o = some (.sc T x) := by
  cases o with
  | iadd a b => exact bin_sound hargs h fun _ _ => ⟨_, rfl⟩
  | isub a b => exact bin_sound hargs h fun _ _ => ⟨_, rfl⟩
  | imul a b => exact bin_sound hargs h fun _ _ => ⟨_, rfl⟩
  | band a b => exact zipIntBits_sound hargs h
  | bandNot a b => exact zipIntBits_sound hargs h
  | bor a b => exact zipIntBits_sound hargs h
  | bxor a b => exact zipIntBits_sound hargs h
  | icmp c a b => exact icmp_sound hargs h
  | fadd a b => exact zipF_sound hargs h
  | fsub a b => exact zipF_sound hargs h
  | fmul a b => exact zipF_sound hargs h
  | fmax a b => exact zipBits_sound hargs h
  | fmin a b => exact zipBits_sound hargs h
  | fcmp c a b => exact fcmp_sound hargs h
  | ishl a b =>
      obtain ⟨t, u, rfl, ⟨x, hx⟩, ⟨y, hy⟩⟩ := args2' (by simpa only [Op.regs] using hargs)
      simp only [tyOp, Option.ite_none_right_eq_some, Option.some.injEq, Bool.and_eq_true] at h
      obtain ⟨⟨h1, h2⟩, rfl⟩ := h
      simp [evalOp, shiftBin, get, hx, hy, h1, h2, norm]
  | ushr a b =>
      obtain ⟨t, u, rfl, ⟨x, hx⟩, ⟨y, hy⟩⟩ := args2' (by simpa only [Op.regs] using hargs)
      simp only [tyOp, Option.ite_none_right_eq_some, Option.some.injEq, Bool.and_eq_true] at h
      obtain ⟨⟨h1, h2⟩, rfl⟩ := h
      simp [evalOp, shiftBin, get, hx, hy, h1, h2, norm]
  | ishift k a b =>
      obtain ⟨t, u, rfl, ⟨x, hx⟩, ⟨y, hy⟩⟩ := args2' (by simpa only [Op.regs] using hargs)
      simp only [tyOp, Option.ite_none_right_eq_some, Option.some.injEq, Bool.and_eq_true] at h
      obtain ⟨⟨h1, h2⟩, rfl⟩ := h
      simp [evalOp, shiftBin, get, hx, hy, h1, h2, norm]
      cases k <;> exact ⟨_, rfl⟩
  | ineg a =>
      obtain ⟨t, rfl, x, hx⟩ := args1 (by simpa only [Op.regs] using hargs)
      simp only [tyOp, Option.ite_none_right_eq_some, Option.some.injEq, Bool.and_eq_true, isF,
        Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at h
      first
        | (obtain ⟨hc, rfl⟩ := h
           first
             | (simp [evalOp, un, get, hx, hc, norm, ofInt]; done)
             | (simp [evalOp, un, get, hx, hc, norm, ofInt, iunOp]; split <;> simp_all; done)
             | (cases t <;> simp_all [evalOp, get, mapFBits, ofInt, norm, IExt.admits, ClifTy.isInt, ClifTy.width]; done)
             | (cases t <;> simp_all [evalOp, get, mapFBits, ofInt, norm, IExt.admits, ClifTy.isInt, ClifTy.width] <;> split <;> simp_all))
        | (obtain ⟨rfl, rfl⟩ := h; simp [evalOp, get, hx])
  | ctz a =>
      obtain ⟨t, rfl, x, hx⟩ := args1 (by simpa only [Op.regs] using hargs)
      simp only [tyOp, Option.ite_none_right_eq_some, Option.some.injEq, Bool.and_eq_true, isF,
        Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at h
      first
        | (obtain ⟨hc, rfl⟩ := h
           first
             | (simp [evalOp, un, get, hx, hc, norm, ofInt]; done)
             | (simp [evalOp, un, get, hx, hc, norm, ofInt, iunOp]; split <;> simp_all; done)
             | (cases t <;> simp_all [evalOp, get, mapFBits, ofInt, norm, IExt.admits, ClifTy.isInt, ClifTy.width]; done)
             | (cases t <;> simp_all [evalOp, get, mapFBits, ofInt, norm, IExt.admits, ClifTy.isInt, ClifTy.width] <;> split <;> simp_all))
        | (obtain ⟨rfl, rfl⟩ := h; simp [evalOp, get, hx])
  | popcnt a =>
      obtain ⟨t, rfl, x, hx⟩ := args1 (by simpa only [Op.regs] using hargs)
      simp only [tyOp, Option.ite_none_right_eq_some, Option.some.injEq, Bool.and_eq_true, isF,
        Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at h
      first
        | (obtain ⟨hc, rfl⟩ := h
           first
             | (simp [evalOp, un, get, hx, hc, norm, ofInt]; done)
             | (simp [evalOp, un, get, hx, hc, norm, ofInt, iunOp]; split <;> simp_all; done)
             | (cases t <;> simp_all [evalOp, get, mapFBits, ofInt, norm, IExt.admits, ClifTy.isInt, ClifTy.width]; done)
             | (cases t <;> simp_all [evalOp, get, mapFBits, ofInt, norm, IExt.admits, ClifTy.isInt, ClifTy.width] <;> split <;> simp_all))
        | (obtain ⟨rfl, rfl⟩ := h; simp [evalOp, get, hx])
  | ireduce32 a =>
      obtain ⟨t, rfl, x, hx⟩ := args1 (by simpa only [Op.regs] using hargs)
      simp only [tyOp, Option.ite_none_right_eq_some, Option.some.injEq, Bool.and_eq_true, isF,
        Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at h
      first
        | (obtain ⟨hc, rfl⟩ := h
           first
             | (simp [evalOp, un, get, hx, hc, norm, ofInt]; done)
             | (simp [evalOp, un, get, hx, hc, norm, ofInt, iunOp]; split <;> simp_all; done)
             | (cases t <;> simp_all [evalOp, get, mapFBits, ofInt, norm, IExt.admits, ClifTy.isInt, ClifTy.width]; done)
             | (cases t <;> simp_all [evalOp, get, mapFBits, ofInt, norm, IExt.admits, ClifTy.isInt, ClifTy.width] <;> split <;> simp_all))
        | (obtain ⟨rfl, rfl⟩ := h; simp [evalOp, get, hx])
  | uextend64 a =>
      obtain ⟨t, rfl, x, hx⟩ := args1 (by simpa only [Op.regs] using hargs)
      simp only [tyOp, Option.ite_none_right_eq_some, Option.some.injEq, Bool.and_eq_true, isF,
        Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at h
      first
        | (obtain ⟨hc, rfl⟩ := h
           first
             | (simp [evalOp, un, get, hx, hc, norm, ofInt]; done)
             | (simp [evalOp, un, get, hx, hc, norm, ofInt, iunOp]; split <;> simp_all; done)
             | (cases t <;> simp_all [evalOp, get, mapFBits, ofInt, norm, IExt.admits, ClifTy.isInt, ClifTy.width]; done)
             | (cases t <;> simp_all [evalOp, get, mapFBits, ofInt, norm, IExt.admits, ClifTy.isInt, ClifTy.width] <;> split <;> simp_all))
        | (obtain ⟨rfl, rfl⟩ := h; simp [evalOp, get, hx])
  | sextend64 a =>
      obtain ⟨t, rfl, x, hx⟩ := args1 (by simpa only [Op.regs] using hargs)
      simp only [tyOp, Option.ite_none_right_eq_some, Option.some.injEq, Bool.and_eq_true, isF,
        Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at h
      first
        | (obtain ⟨hc, rfl⟩ := h
           first
             | (simp [evalOp, un, get, hx, hc, norm, ofInt]; done)
             | (simp [evalOp, un, get, hx, hc, norm, ofInt, iunOp]; split <;> simp_all; done)
             | (cases t <;> simp_all [evalOp, get, mapFBits, ofInt, norm, IExt.admits, ClifTy.isInt, ClifTy.width]; done)
             | (cases t <;> simp_all [evalOp, get, mapFBits, ofInt, norm, IExt.admits, ClifTy.isInt, ClifTy.width] <;> split <;> simp_all))
        | (obtain ⟨rfl, rfl⟩ := h; simp [evalOp, get, hx])
  | fneg a =>
      obtain ⟨t, rfl, x, hx⟩ := args1 (by simpa only [Op.regs] using hargs)
      simp only [tyOp, Option.ite_none_right_eq_some, Option.some.injEq, Bool.and_eq_true, isF,
        Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at h
      first
        | (obtain ⟨hc, rfl⟩ := h
           first
             | (simp [evalOp, un, get, hx, hc, norm, ofInt]; done)
             | (simp [evalOp, un, get, hx, hc, norm, ofInt, iunOp]; split <;> simp_all; done)
             | (cases t <;> simp_all [evalOp, get, mapFBits, ofInt, norm, IExt.admits, ClifTy.isInt, ClifTy.width]; done)
             | (cases t <;> simp_all [evalOp, get, mapFBits, ofInt, norm, IExt.admits, ClifTy.isInt, ClifTy.width] <;> split <;> simp_all))
        | (obtain ⟨rfl, rfl⟩ := h; simp [evalOp, get, hx])
  | fun1 k a =>
      obtain ⟨t, rfl, x, hx⟩ := args1 (by simpa only [Op.regs] using hargs)
      simp only [tyOp, Option.ite_none_right_eq_some, Option.some.injEq, Bool.and_eq_true, isF,
        Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at h
      first
        | (obtain ⟨hc, rfl⟩ := h
           first
             | (simp [evalOp, un, get, hx, hc, norm, ofInt]; done)
             | (simp [evalOp, un, get, hx, hc, norm, ofInt, iunOp]; split <;> simp_all; done)
             | (cases t <;> simp_all [evalOp, get, mapFBits, ofInt, norm, IExt.admits, ClifTy.isInt, ClifTy.width]; done)
             | (cases t <;> simp_all [evalOp, get, mapFBits, ofInt, norm, IExt.admits, ClifTy.isInt, ClifTy.width] <;> split <;> simp_all))
        | (obtain ⟨rfl, rfl⟩ := h; simp [evalOp, get, hx])
  | fpromote a =>
      obtain ⟨t, rfl, x, hx⟩ := args1 (by simpa only [Op.regs] using hargs)
      simp only [tyOp, Option.ite_none_right_eq_some, Option.some.injEq, Bool.and_eq_true, isF,
        Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at h
      first
        | (obtain ⟨hc, rfl⟩ := h
           first
             | (simp [evalOp, un, get, hx, hc, norm, ofInt]; done)
             | (simp [evalOp, un, get, hx, hc, norm, ofInt, iunOp]; split <;> simp_all; done)
             | (cases t <;> simp_all [evalOp, get, mapFBits, ofInt, norm, IExt.admits, ClifTy.isInt, ClifTy.width]; done)
             | (cases t <;> simp_all [evalOp, get, mapFBits, ofInt, norm, IExt.admits, ClifTy.isInt, ClifTy.width] <;> split <;> simp_all))
        | (obtain ⟨rfl, rfl⟩ := h; simp [evalOp, get, hx])
  | fcvtFromSint ty a =>
      obtain ⟨t, rfl, x, hx⟩ := args1 (by simpa only [Op.regs] using hargs)
      simp only [tyOp, Option.ite_none_right_eq_some, Option.some.injEq, Bool.and_eq_true, isF,
        Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at h
      first
        | (obtain ⟨hc, rfl⟩ := h
           first
             | (simp [evalOp, un, get, hx, hc, norm, ofInt]; done)
             | (simp [evalOp, un, get, hx, hc, norm, ofInt, iunOp]; split <;> simp_all; done)
             | (cases t <;> simp_all [evalOp, get, mapFBits, ofInt, norm, IExt.admits, ClifTy.isInt, ClifTy.width]; done)
             | (cases t <;> simp_all [evalOp, get, mapFBits, ofInt, norm, IExt.admits, ClifTy.isInt, ClifTy.width] <;> split <;> simp_all))
        | (obtain ⟨rfl, rfl⟩ := h; simp [evalOp, get, hx])
  | bitcast ty a =>
      obtain ⟨t, rfl, x, hx⟩ := args1 (by simpa only [Op.regs] using hargs)
      simp only [tyOp, Option.ite_none_right_eq_some, Option.some.injEq, Bool.and_eq_true, isF,
        Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at h
      first
        | (obtain ⟨hc, rfl⟩ := h
           first
             | (simp [evalOp, un, get, hx, hc, norm, ofInt]; done)
             | (simp [evalOp, un, get, hx, hc, norm, ofInt, iunOp]; split <;> simp_all; done)
             | (cases t <;> simp_all [evalOp, get, mapFBits, ofInt, norm, IExt.admits, ClifTy.isInt, ClifTy.width]; done)
             | (cases t <;> simp_all [evalOp, get, mapFBits, ofInt, norm, IExt.admits, ClifTy.isInt, ClifTy.width] <;> split <;> simp_all))
        | (obtain ⟨rfl, rfl⟩ := h; simp [evalOp, get, hx])
  | iun k a =>
      obtain ⟨t, rfl, x, hx⟩ := args1 (by simpa only [Op.regs] using hargs)
      simp only [tyOp, Option.ite_none_right_eq_some, Option.some.injEq] at h
      obtain ⟨hc, rfl⟩ := h
      simp only [IUn.admits, Bool.and_eq_true, Bool.or_eq_true, bne_iff_ne, ne_eq,
        decide_eq_true_eq] at hc
      cases k <;> simp_all [evalOp, un, get, iunOp, norm, ofInt]
      exact hc.2.resolve_left (by decide)
  | iext k ty a =>
      obtain ⟨t, rfl, x, hx⟩ := args1 (by simpa only [Op.regs] using hargs)
      simp only [tyOp, Option.ite_none_right_eq_some, Option.some.injEq, Bool.and_eq_true, isF,
        Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at h
      first
        | (obtain ⟨hc, rfl⟩ := h
           first
             | (simp [evalOp, un, get, hx, hc, norm, ofInt]; done)
             | (simp [evalOp, un, get, hx, hc, norm, ofInt, iunOp]; split <;> simp_all; done)
             | (cases t <;> simp_all [evalOp, get, mapFBits, ofInt, norm, IExt.admits, ClifTy.isInt, ClifTy.width]; done)
             | (cases t <;> simp_all [evalOp, get, mapFBits, ofInt, norm, IExt.admits, ClifTy.isInt, ClifTy.width] <;> split <;> simp_all))
        | (obtain ⟨rfl, rfl⟩ := h; simp [evalOp, get, hx])
  | select c a b =>
      obtain ⟨tc, ta, tb, rfl, ⟨x, hx⟩, ⟨y, hy⟩, ⟨z, hz⟩⟩ := args3 (by simpa only [Op.regs] using hargs)
      simp only [tyOp, Option.ite_none_right_eq_some, Option.some.injEq, Bool.and_eq_true,
        beq_iff_eq] at h
      obtain ⟨⟨h1, rfl⟩, rfl⟩ := h
      by_cases hc : isTrue (V.sc tc x) <;> simp [evalOp, get, hx, hy, hz, h1, V.ty, hc]
  | _ => simp [tyOp] at h

end AlgorithmLib.HProg.Sem

namespace AlgorithmLib.Prog

/-- Binding past everything an environment extends keeps the extension. -/
theorem bindAt_append {Γ δ : Env} {n : Nat} (vs : List V) (h : Γ.size ≤ n) :
    bindAt (Γ ++ δ) n vs = Γ ++ bindAt δ (n - Γ.size) vs := by
  apply Array.toList_inj.mp
  simp only [bindAt, Array.take, Array.toList_append, Array.toList_extract, Array.toList_replicate,
    Array.size_append, List.append_assoc, List.extract_eq_drop_take, List.drop_zero, List.take_append,
    Array.length_toList, Nat.sub_zero]
  rw [List.take_of_length_le (by simp; omega)]
  have : n - (Γ.size + δ.size) = n - Γ.size - δ.size := by omega
  rw [this]

/-- A constant binds its value: nothing about the world is asked. -/
theorem wp_iconst {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {ty : ClifTy} {k : Int} {h : ty.isInt = true}
    (c : Slot ty → Prog Slot Lvl α) (Q : α → Env → World → Prop) :
    wp cfg d J (.op (.iconst ty k h) c) Q = fun Γ w => wp cfg d J (c Γ.size) Q (Γ.push (ofInt ty k)) w := by
  funext Γ w
  simp only [wp_op, Op'.erase, evalOp, Option.some.injEq, forall_eq', reduceCtorEq, false_implies, true_and]

/-- Code a generator unrolls over a list is the code for each element in turn. -/
theorem wp_forM_cons {cfg : Cfg} {d : Nat} {J : Post} {β : Type} (a : β) (as : List β)
    (f : β → Prog Slot Lvl PUnit) (Q : PUnit → Env → World → Prop) :
    wp cfg d J (as.cons a |>.forM f) Q = wp cfg d J (f a) (fun _ => wp cfg d J (as.forM f) Q) :=
  wp_bind cfg (f a) (fun _ => as.forM f) d J Q

theorem wp_forM_nil {cfg : Cfg} {d : Nat} {J : Post} {β : Type} (f : β → Prog Slot Lvl PUnit)
    (Q : PUnit → Env → World → Prop) : wp cfg d J (([] : List β).forM f) Q = Q ⟨⟩ := rfl

/-- **A counted loop, by the typestate.** A body that keeps `W` from any
    extension of the environment the loop starts from keeps it for every trip,
    and what follows starts from an extension of that environment in `W`. The
    counter, the test and the increment are the loop's own. -/
theorem wp_forLoop {cfg : Cfg} {d : Nat} {J : Post} {n : Slot .i64} {body : Slot .i64 → Prog Slot Lvl Unit}
    {R : Unit → Env → World → Prop} {Γ : Env} {w : World} (W : World → Prop) (hW : W w)
    (hbody : ∀ d' J', J'.faultOk = true → ∀ i δ wb, W wb → wp cfg d' J' (body i) (fun _ _ w' => W w') (Γ ++ δ) wb)
    (hk : ∀ δ w', W w' → R () (Γ ++ δ) w') (hJ : J.faultOk = true) :
    wp cfg d J (forLoop n body) R Γ w := by
  simp only [forLoop, wloop1, wloop, wloopL, iconst64, iconst, iaddImm, op, wp_bind, wp_op, wp_ret, wp_pure,
    loopJ_faultOk, and_of_faultOk hJ, and_of_faultOk' hJ]
  intro v0 _
  rw [wp_loop]
  refine ⟨fun Γ0 _ w => (∃ δ, Γ0 = Γ ++ δ) ∧ W w,
          fun Γ0 _ _ Γh wh => (∃ δ, Γ0 = Γ ++ δ) ∧ (∃ δ, Γh = Γ ++ δ) ∧ W wh,
          fun Γ0 Γb _ w => (∃ δ, Γ0 = Γ ++ δ) ∧ (∃ δ, Γb = Γ ++ δ) ∧ W w, ?_, ?_, ?_, ?_, ?_,
          fun _ _ _ _ _ _ _ => hJ⟩
  · intro cs _; exact ⟨⟨#[v0], by simp⟩, hW⟩
  · rintro n0 Γ0 cs wh rfl ⟨⟨δ, rfl⟩, hw⟩
    simp only [wp_pure]
    exact ⟨⟨δ, rfl⟩, ⟨_, bindAt_append cs (by simp)⟩, hw⟩
  · rintro Γ0 cs a Γ1 w1 t f vs ⟨h0, ⟨δ, rfl⟩, hw⟩ _ _ _
    exact ⟨h0, ⟨δ.push (.sc t f), by simp⟩, hw⟩
  · rintro nb Γ0 cs a Γ1 t f wb rfl ⟨h0, ⟨δ, rfl⟩, hw⟩ _ _
    simp only [wp_bind]
    have e : (Γ ++ δ).push (.sc t f) = Γ ++ δ.push (.sc t f) := by simp
    rw [e, bindAt_append cs (by simp; omega)]
    refine wp_mono cfg _ _ _ _ _ ?_ _ _ (hbody _ _ (by exact hJ) _ _ wb hw)
    intro _ Γ2 w2 hw2
    simp only [wp_op, wp_ret, wp_pure, iadd, op, loopJ_faultOk, and_of_faultOk hJ, and_of_faultOk' hJ]
    intro v _ v' _ nx _
    exact ⟨h0, hw2⟩
  · rintro nE Γ0 Γb vs w' hle ⟨⟨δ0, rfl⟩, ⟨δ, rfl⟩, hw⟩
    simp only [wp_ret]
    rw [bindAt_append vs (by simp at hle; omega)]
    exact hk _ _ hw

/-- **A counted loop threading an accumulator, by a typestate.** The
    typestate is the invariant; the accumulator may be anything, so the body
    may compute it however it likes, and what follows the loop gets whichever
    value it ends with. -/
theorem wp_forLoopAcc {cfg : Cfg} {d : Nat} {J : Post} {t : ClifTy} {n : Slot .i64} {acc0 : Slot t}
    {body : Slot .i64 → Slot t → Prog Slot Lvl (Slot t)}
    {R : Slot t → Env → World → Prop} {Γ : Env} {w : World} (W : World → Prop) (hW : W w)
    (hbody : ∀ d' J', J'.faultOk = true → ∀ i a δ wb, W wb → wp cfg d' J' (body i a) (fun _ _ w' => W w') (Γ ++ δ) wb)
    (hk : ∀ a δ w', W w' → R a (Γ ++ δ) w') (hJ : J.faultOk = true) :
    wp cfg d J (forLoopAcc n acc0 body) R Γ w := by
  simp only [forLoopAcc, wloop2, wloop, wloopL, iconst64, iconst, iaddImm, op, wp_bind, wp_op, wp_ret,
    wp_pure, loopJ_faultOk, and_of_faultOk hJ, and_of_faultOk' hJ]
  intro v0 _
  rw [wp_loop]
  refine ⟨fun Γ0 _ w => (∃ δ, Γ0 = Γ ++ δ) ∧ W w,
          fun Γ0 _ _ Γh wh => (∃ δ, Γ0 = Γ ++ δ) ∧ (∃ δ, Γh = Γ ++ δ) ∧ W wh,
          fun Γ0 Γb _ w => (∃ δ, Γ0 = Γ ++ δ) ∧ (∃ δ, Γb = Γ ++ δ) ∧ W w, ?_, ?_, ?_, ?_, ?_,
    fun _ _ _ _ _ _ _ => hJ⟩
  · intro cs _; exact ⟨⟨#[v0], by simp⟩, hW⟩
  · rintro n0 Γ0 cs wh rfl ⟨⟨δ, rfl⟩, hw⟩
    simp only [wp_pure]
    exact ⟨⟨δ, rfl⟩, ⟨_, bindAt_append cs (by simp)⟩, hw⟩
  · rintro Γ0 cs a Γ1 w1 t' f vs ⟨h0, ⟨δ, rfl⟩, hw⟩ _ _ _
    exact ⟨h0, ⟨δ.push (.sc t' f), by simp⟩, hw⟩
  · rintro nb Γ0 cs a Γ1 t' f wb rfl ⟨h0, ⟨δ, rfl⟩, hw⟩ _ _
    simp only [wp_bind]
    have e : (Γ ++ δ).push (.sc t' f) = Γ ++ δ.push (.sc t' f) := by simp
    rw [e, bindAt_append cs (by simp; omega)]
    refine wp_mono cfg _ _ _ _ _ ?_ _ _ (hbody _ _ (by exact hJ) _ _ _ wb hw)
    intro _ Γ2 w2 hw2
    simp only [wp_op, wp_ret, wp_pure, iadd, op, loopJ_faultOk, and_of_faultOk hJ, and_of_faultOk' hJ]
    intro v _ v' _ nx _
    exact ⟨h0, hw2⟩
  · rintro nE Γ0 Γb vs w' hle ⟨⟨δ0, rfl⟩, ⟨δ, rfl⟩, hw⟩
    simp only [wp_ret]
    rw [bindAt_append vs (by simp at hle; omega)]
    exact hk _ _ _ hw

/-- **A call moves a typestate**: from any world in `W`, the call with these
    arguments answers, into a world in `W'`. -/
def Moves (cfg : Cfg) (W : World → Prop) (f : Ffi) (vs : List V) (W' : World → Prop) : Prop :=
  ∀ w, W w → ∃ r w', callOf cfg.locals (.ffi f) vs (obsCall w (.ffi f) vs) = some (r, w') ∧ W' w'

/-- A call that keeps its typestate. -/
abbrev Keeps (cfg : Cfg) (W : World → Prop) (f : Ffi) (vs : List V) : Prop := Moves cfg W f vs W

/-- **A call, by its contract.** Whatever the argument slots hold, the call
    moves `W` to `W'`; a call that answers nothing leaves the environment as it
    was. -/
theorem wp_ffiVoid_moves {cfg : Cfg} {d : Nat} {J : Post} {f : Ffi} {args : Vals Slot f.params}
    {Q : Unit → Env → World → Prop} {Γ : Env} {w : World} (W W' : World → Prop)
    (hres : f.result.isSome = false)
    (hK : ∀ vs, args.slots.mapM (fun r => Γ[r]?) = some vs → Moves cfg W f vs W') (hW : W w)
    (hQ : ∀ w', W' w' → Q () Γ w') :
    wp cfg d J (ffiVoid f args) Q Γ w := by
  simp only [ffiVoid, ffi, wp_bind, wp_call, wp_ret, wp_pure]
  intro vs hvs
  obtain ⟨r, w', hc, hw'⟩ := hK vs hvs w hW
  exact ⟨r, w', hc, by simp only [hres, Bool.false_eq_true, if_false]; exact hQ w' hw'⟩

/-- The same for a call whose answer is dropped: it still binds the next slot. -/
theorem wp_ffiVoid_moves_res {cfg : Cfg} {d : Nat} {J : Post} {f : Ffi} {args : Vals Slot f.params}
    {Q : Unit → Env → World → Prop} {Γ : Env} {w : World} (W W' : World → Prop)
    (hres : f.result.isSome = true)
    (hK : ∀ vs, args.slots.mapM (fun r => Γ[r]?) = some vs → Moves cfg W f vs W') (hW : W w)
    (hQ : ∀ v w', W' w' → Q () (Γ.push v) w') :
    wp cfg d J (ffiVoid f args) Q Γ w := by
  simp only [ffiVoid, ffi, wp_bind, wp_call, wp_ret, wp_pure]
  intro vs hvs
  obtain ⟨r, w', hc, hw'⟩ := hK vs hvs w hW
  exact ⟨r, w', hc, by simp only [hres, if_true]; exact fun v _ => hQ v w' hw'⟩

/-- **A call moves a typestate by its contract.** The call's precondition
    holds in every world of `W`, and every answer the contract gives from there
    lands in `W'`. The call itself is `callBits` on its argument bits
    (`callOf_ffi`), and `pre_safe` says it answers. -/
theorem moves_of_pre (cfg : Cfg) {W W' : World → Prop} {f : Ffi} {vs : List V} {bits : List UInt64}
    (hf : f ≠ .threadSpawn) (hb : vs.mapM asBits = some bits)
    (hobs : ∀ w, W w → W (obsCall w (.ffi f) vs)) (hz : ∀ w, W w → w.mem.frozen = false)
    (hpre : ∀ w, W w → Contracts.Pre f bits w)
    (hpost : ∀ w r w', W w → callBits f bits w = some (r, w') → W' w') : Moves cfg W f vs W' := by
  intro w hw
  have hw1 := hobs w hw
  rw [Contracts.callOf_ffi hf _ _ (hz _ hw1), hb]
  obtain ⟨⟨r, w'⟩, hr⟩ := Option.isSome_iff_exists.mp (Contracts.pre_safe f bits _ (hpre _ hw1))
  exact ⟨r, w', hr, hpost _ _ _ hw1 hr⟩

open Contracts (Part Fact TState after after_sound afterStore afterStore_sound)

theorem Part.get_obsCall (p : Part) (w : World) (c : Callee) (vs : List V) :
    p.get (obsCall w c vs) = p.get w := by
  cases p <;> rfl

theorem TState.holds_obsCall {S : TState} {w : World} {c : Callee} {vs : List V} (h : S.holds w) :
    S.holds (obsCall w c vs) := by
  intro x hx
  have := h x hx
  cases x with
  | part p b => exact (Part.get_obsCall p w c vs).trans this
  | room _ _ => exact this
  | cell _ _ => exact this
  | held k _ => cases k <;> exact this
  | opened k _ => cases k <;> exact this
  | htVals _ => exact this
  | cstr _ => exact this
  | devSeq => exact this
  | oracles => exact this
  | cstrIn _ _ => exact this
  | roomArg _ _ => exact this
  | devBuf _ _ => exact this
  | pinnedUsed _ => exact this

/-- **The standard typestates move by the table.** From a typestate that has
    memory unfrozen, a call whose precondition the facts give lands in what
    the contract table says it leaves, `after f bits S`, computed as `S'`. -/
theorem moves_state (cfg : Cfg) (S S' : TState) {f : Ffi} {vs : List V} {bits : List UInt64}
    (hz : S.contains (.part .frozen false) = true) (hf : f ≠ .threadSpawn) (hb : vs.mapM asBits = some bits)
    (hS' : after f bits S = S') (hpre : ∀ w, S.holds w → Contracts.Pre f bits w) :
    Moves cfg S.holds f vs S'.holds := by
  subst hS'
  exact moves_of_pre cfg hf hb (fun _ h => TState.holds_obsCall h)
    (fun _ h => Contracts.holds_of_mem h hz) hpre (fun _ _ _ hS h => after_sound hS h)

/-- **The same, landing in fewer facts**: for a call whose table entry only
    keeps facts, any of them it keeps. What a call keeps is computed; a fact
    whose keeping does not compute, as over a length the program was answered,
    is dropped. -/
theorem moves_state_sub (cfg : Cfg) (S S' : TState) {f : Ffi} {vs : List V} {bits : List UInt64}
    (hz : S.contains (.part .frozen false) = true) (hf : f ≠ .threadSpawn) (hb : vs.mapM asBits = some bits)
    (ha : after f bits S = S.filter (Contracts.keeps f bits)) (hsub : S'.Sublist S)
    (hk : S'.all (Contracts.keeps f bits) = true) (hpre : ∀ w, S.holds w → Contracts.Pre f bits w) :
    Moves cfg S.holds f vs S'.holds := by
  intro w hw
  obtain ⟨r, w', hc, hw'⟩ := moves_state cfg S _ hz hf hb rfl hpre w hw
  refine ⟨r, w', hc, fun x hx => hw' x ?_⟩
  rw [ha]
  exact List.mem_filter.mpr ⟨hsub.subset hx, List.all_eq_true.mp hk x hx⟩

/-- **What a call's answer is known to be**, whatever the world: a bound a
    program may count by. `fileRead` answers `-1` or at most the bytes it was
    asked for; `windowPoll` answers `-1` or at most the events it has room for.
    Every other answer is known by its type alone. -/
def answerOk : Ffi → List UInt64 → UInt64 → Prop
  | .fileRead, [_, _, _, _, size], x => x.toNat = 18446744073709551615 ∨ (0 < size.toNat → x.toNat ≤ size.toNat)
  | .windowPoll, [_, _, maxEvents], x => x.toNat = 4294967295 ∨ x.toNat ≤ (asI32 maxEvents).toNat
  | _, _, _ => True

theorem ofInt_nat_val {n : Nat} (h : n < 2 ^ 64) :
    UInt64.ofNat ((n : Int).emod (1 <<< 64)).toNat = UInt64.ofNat n := by
  have e : (n : Int).emod ((1 <<< 64 : Nat) : Int) = n := by
    rw [Nat.one_shiftLeft]
    exact Int.emod_eq_of_lt (by omega) (by omega)
  rw [e, Int.toNat_natCast]

/-- An answer written as a count below the width, read back as its bits. -/
theorem ofInt_toNat {ty t : ClifTy} {n k : Nat} {x : UInt64} (h : ofInt ty (n : Int) = .sc t x)
    (hm : (widthMask ty).toNat = 2 ^ k - 1) (hk : k ≤ 64) (hn : n < 2 ^ k) : x.toNat = n := by
  have h64 : n < 2 ^ 64 := Nat.lt_of_lt_of_le hn (Nat.pow_le_pow_right (by decide) hk)
  simp only [ofInt, norm, ofInt_nat_val h64, V.sc.injEq] at h
  obtain ⟨-, rfl⟩ := h
  have e2 : (UInt64.ofNat n).toNat = n := by simp; omega
  rw [UInt64.toNat_and, hm, e2, Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hn]

theorem ofInt_neg_one_i64 {t : ClifTy} {x : UInt64} (h : ofInt .i64 (-1) = .sc t x) :
    x = 0xFFFFFFFFFFFFFFFF := by
  have e : ofInt .i64 (-1) = .sc .i64 0xFFFFFFFFFFFFFFFF := by rfl
  rw [e, V.sc.injEq] at h; exact h.2.symm

theorem ofInt_neg_one_i32 {t : ClifTy} {x : UInt64} (h : ofInt .i32 (-1) = .sc t x) : x = 0xFFFFFFFF := by
  have e : ofInt .i32 (-1) = .sc .i32 0xFFFFFFFF := by rfl
  rw [e, V.sc.injEq] at h; exact h.2.symm

theorem answerOk_sound {f : Ffi} {bits : List UInt64} {w : World} {t : ClifTy} {x : UInt64} {w' : World}
    (h : Sem.callBits f bits w = some (some (V.sc t x), w')) : answerOk f bits x := by
  cases f
  case fileRead =>
    change ffiFileRead bits w = _ at h
    rcases bits with _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨size, _ | ⟨_, _⟩⟩⟩⟩⟩⟩ <;>
      try (simp [ffiFileRead] at h; done)
    · show _ ∨ _
      simp only [ffiFileRead] at h
      obtain ⟨path, -, h⟩ := Option.bind_eq_some_iff.mp h
      split at h
      · simp only [Option.some.injEq, Prod.mk.injEq] at h
        exact .inl (by rw [ofInt_neg_one_i64 h.1]; rfl)
      · obtain ⟨m, -, h⟩ := Option.bind_eq_some_iff.mp h
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        right; intro hs
        have hs' : (size == 0) = false := by
          simp only [beq_eq_false_iff_ne, ne_eq]; intro e; subst e; simp at hs
        simp only [hs', Bool.false_eq_true, if_false] at h
        have := size.toNat_lt
        rw [ofInt_toNat h.1 (k := 64) (by decide) (by decide) (by omega)]
        omega
  case windowPoll =>
    change ffiWindowPoll bits w = _ at h
    rcases bits with _ | ⟨_, _ | ⟨_, _ | ⟨maxEvents, _ | ⟨_, _⟩⟩⟩⟩ <;>
      try (simp [ffiWindowPoll] at h; done)
    · show _ ∨ _
      simp only [ffiWindowPoll] at h
      split at h
      · simp only [cudaFail, Option.some.injEq, Prod.mk.injEq] at h
        exact .inl (by rw [ofInt_neg_one_i32 h.1]; rfl)
      · split at h
        · cases h
        · simp only [cudaFail, Option.some.injEq, Prod.mk.injEq] at h
          exact .inl (by rw [ofInt_neg_one_i32 h.1]; rfl)
        · obtain ⟨m, -, h⟩ := Option.bind_eq_some_iff.mp h
          simp only [Option.some.injEq, Prod.mk.injEq] at h
          have hm : (asI32 maxEvents).toNat < 2 ^ 31 := by
            have hb : (maxEvents &&& 4294967295).toNat < 2 ^ 32 := by
              rw [UInt64.toNat_and]; exact Nat.lt_of_le_of_lt Nat.and_le_right (by decide)
            have hh : ((1 : UInt64) <<< UInt64.ofNat (32 - 1)).toNat = 2147483648 := by decide
            unfold asI32 signed
            simp only [ClifTy.width, widthMask]
            split
            · omega
            · split <;> rename_i hc <;> simp only [UInt64.le_iff_toNat_le, hh] at hc <;> omega
          right
          rw [ofInt_toNat h.1 (k := 32) (by decide) (by decide) (by omega)]
          omega
  all_goals trivial

/-- A typestate asks less once some of its facts are dropped. -/
theorem TState.holds_filter {S : TState} {w : World} (p : Fact → Bool) (h : S.holds w) :
    TState.holds (S.filter p) w := fun x hx => h x (List.mem_filter.mp hx).1

/-- Whether a typestate names every fact another does. -/
def TState.sub (S S' : TState) : Bool := S.all (S'.contains ·)

/-- A typestate asks less than one that names every fact it names. -/
theorem TState.holds_sub {S S' : TState} (h : TState.sub S S' = true) {w : World}
    (hS : S'.holds w) : S.holds w := by
  intro x hx
  exact hS x (by have := List.all_eq_true.mp h x hx; simpa using this)

/-- Each fact of a list is one of `S'`'s. -/
def TState.AllIn (S' : TState) : List Fact → Prop
  | [] => True
  | x :: xs => x ∈ S' ∧ TState.AllIn S' xs

/-- A typestate asks less than one whose facts include each of its own, found
    by place where a fact names a value the kernel cannot compare. -/
theorem TState.holds_allIn {S S' : TState} (h : TState.AllIn S' S) {w : World}
    (hS : S'.holds w) : S.holds w := by
  intro x hx
  induction S with
  | nil => cases hx
  | cons y ys ih =>
      rcases List.mem_cons.mp hx with rfl | hx
      · exact hS _ h.1
      · exact ih h.2 hx

/-- The typestate ordinary device calls need: host memory is not frozen, the
    context is live, and no stream is being captured. -/
abbrev Live : World → Prop :=
  TState.holds [.part .frozen false, .part .cuda true, .part .capturing false]

theorem tyBytes_le (ty : ClifTy) : tyBytes ty ≤ 2 ^ 64 := by cases ty <;> decide

/-- `n` bytes at `a` lie inside one region outside pinned memory, within the
    span an address names, with memory not frozen: a load or a store of them
    finds them, and so does one of any part of them. -/
def Fits (m : Mem) (a : UInt64) (n : Nat) : Prop :=
  ∃ r off, decodeAddr a = some (r, off) ∧ r ≠ .pinned ∧ off + n ≤ m.sizes r ∧ m.frozen = false ∧
    off + n ≤ 2 ^ 36

theorem Fits.load {m : Mem} {a : UInt64} {n : Nat} (h : Fits m a n) : ∃ x, m.load a n = some x := by
  obtain ⟨r, off, hd, hp, hs, hz, -⟩ := h
  exact Option.isSome_iff_exists.mp (Contracts.load_isSome hd hp hs hz)

theorem Fits.store {m : Mem} {a : UInt64} {n : Nat} (h : Fits m a n) (v : UInt64) :
    ∃ m', m.store a n v = some m' := by
  obtain ⟨r, off, hd, hp, hs, hz, -⟩ := h
  exact Option.isSome_iff_exists.mp (Contracts.store_isSome hd hp hs hz v)

theorem Fits.mono {m : Mem} {a : UInt64} {n n' : Nat} (hn : n ≤ n') (h : Fits m a n') : Fits m a n := by
  obtain ⟨r, off, hd, hp, hs, hz, hsp⟩ := h
  exact ⟨r, off, hd, hp, by omega, hz, by omega⟩

/-- Part of bytes that fit fits. -/
theorem Fits.part {m : Mem} {a : UInt64} {n i k : Nat} (h : Fits m a n) (hik : i + k ≤ n) (hk : 0 < k) :
    Fits m (a + UInt64.ofNat i) k := by
  obtain ⟨r, off, hd, hp, hs, hz, hsp⟩ := h
  have hd' := Static.decodeAddr_add (i := i) hd (by simp only [regionSpan, UInt64.reduceToNat]; omega)
  exact ⟨r, off + i, hd', hp, by omega, hz, by omega⟩

/-- Bytes a `room` fact covers. -/
theorem fits_of_state {S : TState} {w : World} {a : UInt64} {n : Nat} (hS : S.holds w)
    (h : Contracts.roomAt S a n = true) : Fits w.mem a n := by
  obtain ⟨r, off, hd, hp, hs, hz, hsp⟩ := Contracts.roomAt_spec hS h
  exact ⟨r, off, hd, hp, hs, hz, hsp⟩

/-- Bytes within a length the program was handed. -/
theorem fits_of_roomArg {S : TState} {w : World} {a x : UInt64} {n : Nat} {r : Region} {off : Nat}
    (hS : S.holds w) (hm : Contracts.Fact.roomArg r x ∈ S) (hz : S.contains (.part .frozen false) = true)
    (hd : decodeAddr a = some (r, off)) (hp : r ≠ .pinned) (hb : off + n ≤ x.toNat) :
    Fits w.mem a n :=
  ⟨r, off, hd, hp, by have := (hS _ hm).1; omega, Contracts.holds_of_mem hS hz,
    by have := (hS _ hm).2; omega⟩

/-- Bytes at a value past an address, within a length the program was handed. -/
theorem fits_of_roomArg_add {S : TState} {w : World} {b x X : UInt64} {n : Nat} {r : Region} {off : Nat}
    (hS : S.holds w) (hm : Contracts.Fact.roomArg r X ∈ S) (hz : S.contains (.part .frozen false) = true)
    (hd : decodeAddr b = some (r, off)) (hp : r ≠ .pinned) (hb : off + x.toNat + n ≤ X.toNat)
    (hn : 0 < n) : Fits w.mem (b + x) n := by
  have hx := hS _ hm
  have hd' : decodeAddr (b + x) = some (r, off + x.toNat) := by
    have := Static.decodeAddr_add (i := x.toNat) hd (by simp only [regionSpan, UInt64.reduceToNat]; have := hx.2; omega)
    rwa [UInt64.ofNat_toNat] at this
  exact fits_of_roomArg hS hm hz hd' hp (by omega)

/-- Bytes at a value past an address, within a room the typestate names. -/
theorem fits_of_room_add {S : TState} {w : World} {b x : UInt64} {n N : Nat} {r : Region} {off : Nat}
    (hS : S.holds w) (hm : Contracts.Fact.room r N ∈ S) (hz : S.contains (.part .frozen false) = true)
    (hd : decodeAddr b = some (r, off)) (hp : r ≠ .pinned) (hb : off + x.toNat + n ≤ N)
    (hN : N ≤ 2 ^ 36) (hn : 0 < n) : Fits w.mem (b + x) n := by
  have hx : N ≤ w.mem.sizes r := hS _ hm
  have hd' : decodeAddr (b + x) = some (r, off + x.toNat) := by
    have := Static.decodeAddr_add (i := x.toNat) hd (by simp only [regionSpan, UInt64.reduceToNat]; omega)
    rwa [UInt64.ofNat_toNat] at this
  exact ⟨r, off + x.toNat, hd', hp, by omega, Contracts.holds_of_mem hS hz, by omega⟩

/-- A place a literal past a sum: the same place as the literal added first. -/
theorem fits_regroup {m : Mem} {b x c : UInt64} {n : Nat} (h : Fits m (b + c + x) n) :
    Fits m (b + x + c) n := by
  rwa [UInt64.add_assoc, UInt64.add_comm x c, ← UInt64.add_assoc]

theorem store_parts {w : World} {m : Mem} {a : UInt64} {n : Nat} {b : UInt64} (hz : m.frozen = w.mem.frozen)
    (p : Part) : p.get { obsStore w a n b with mem := m } = p.get w := by
  cases p <;> first | rfl | exact hz

/-- **A store, by the typestate.** When the address and the value are both
    known, the typestate keeps what the store cannot reach and learns the cell
    it writes; otherwise it keeps every fact but the cells. -/
theorem wp_store_state {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty : ClifTy} {v : Slot ty} {a : Slot .i64} {k : Prog Slot Lvl α}
    (S : TState) (st : Option (UInt64 × UInt64))
    (hst : ∀ x y, st = some (x, y) → (∃ t, Γ[a]? = some (.sc t x)) ∧ ∃ t, Γ[v]? = some (.sc t y))
    (hS : S.holds w) (hJ : J.faultOk = true ∨ ∃ x y, st = some (x, y) ∧ Fits w.mem x (tyBytes ty))
    (hk : ∀ w', (afterStore (st.map (·.1)) (st.map (·.2)) (tyBytes ty) S).holds w' → wp cfg d J k Q Γ w') :
    wp cfg d J (.store v a k) Q Γ w := by
  rw [wp_store]
  refine ⟨fun m hf => ?_, ?_⟩
  · rcases hJ with hJ | ⟨x, y, rfl, hsp⟩
    · exact hJ
    obtain ⟨⟨t, ha⟩, ⟨t', hv⟩⟩ := hst x y rfl
    obtain ⟨m', hm⟩ := hsp.store y
    simp only [runStmt, Sem.get, ha, hv, hm, reduceCtorEq] at hf
  intro w' hrun
  apply hk
  have hn := tyBytes_le ty
  simp only [runStmt, Sem.get] at hrun
  cases hv : Γ[v]? with
  | none => rw [hv] at hrun; simp only [reduceCtorEq] at hrun
  | some val =>
  cases ha : Γ[a]? with
  | none => rw [hv, ha] at hrun; simp only [reduceCtorEq] at hrun
  | some av =>
  cases av with
  | vec _ _ => rw [hv, ha] at hrun; simp only [reduceCtorEq] at hrun
  | sc ta addr =>
  cases val with
  | sc tb b =>
      rw [hv, ha] at hrun
      simp only at hrun
      cases hm : w.mem.store addr (tyBytes ty) b with
      | none => rw [hm] at hrun; cases hrun
      | some m =>
          rw [hm] at hrun; cases hrun
          refine afterStore_sound hS (store_parts (Contracts.store_frozen hm)) (fun k _ => by cases k <;> rfl) rfl ⟨rfl, rfl, rfl⟩ (Contracts.store_sizes_eq hm)
            (fun _ => hm) ?_ ?_ hn
          · intro x hx
            cases st with
            | none => cases hx
            | some q =>
                obtain ⟨x', y'⟩ := q
                simp only [Option.map_some, Option.some.injEq] at hx; subst hx
                obtain ⟨⟨t, ht⟩, -⟩ := hst x' y' rfl
                rw [ha] at ht; cases ht; rfl
          · intro y hy
            cases st with
            | none => cases hy
            | some q =>
                obtain ⟨x', y'⟩ := q
                simp only [Option.map_some, Option.some.injEq] at hy; subst hy
                obtain ⟨-, ⟨t, ht⟩⟩ := hst x' y' rfl
                rw [hv] at ht; cases ht; rfl
  | vec tv ls =>
      rw [hv, ha] at hrun
      simp only at hrun
      split at hrun
      · rename_i m hm
        cases hrun
        cases st with
        | some q =>
            obtain ⟨x', y'⟩ := q
            obtain ⟨-, ⟨t', ht⟩⟩ := hst x' y' rfl
            rw [hv] at ht; cases ht
        | none =>
            rw [← Array.foldlM_toList] at hm
            obtain ⟨hsz, hz⟩ := Contracts.foldl_stores_sizes _
              (fun _ _ _ h => ⟨Contracts.store_sizes_eq h, Contracts.store_frozen h⟩) _ _ _ hm
            exact Contracts.nonCell_kept hS (store_parts hz) (fun k _ => by cases k <;> rfl) rfl ⟨rfl, rfl, rfl⟩ hsz
      · cases hrun

theorem wp_store_known {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty : ClifTy} {v : Slot ty} {a : Slot .i64} {k : Prog Slot Lvl α}
    (S S' : TState) {t t' : ClifTy} {x y : UInt64} (ha : Γ[a]? = some (.sc t x))
    (hv : Γ[v]? = some (.sc t' y)) (hS : S.holds w) (hJ : J.faultOk = true ∨ Fits w.mem x (tyBytes ty))
    (hS' : afterStore (some x) (some y) (tyBytes ty) S = S')
    (hk : ∀ w', S'.holds w' → wp cfg d J k Q Γ w') : wp cfg d J (.store v a k) Q Γ w :=
  wp_store_state S (some (x, y)) (fun _ _ h => by cases h; exact ⟨⟨t, ha⟩, ⟨t', hv⟩⟩) hS
    (hJ.imp_right fun h => ⟨x, y, rfl, h⟩)
    (by subst hS'; exact hk)

theorem afterStore_forget {S : TState} {w : World} {a v : UInt64} {n : Nat}
    (h : (afterStore (some a) (some v) n S).holds w) : (afterStore (some a) none n S).holds w := by
  unfold afterStore at h ⊢
  dsimp only at h ⊢
  split at h
  · exact fun x hx => h x (List.mem_cons_of_mem _ hx)
  · exact h

/-- A store to a known address of a value the typestate does not name: the
    cells it cannot reach survive. -/
theorem wp_store_addr {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty : ClifTy} {v : Slot ty} {a : Slot .i64} {k : Prog Slot Lvl α}
    (S S' : TState) {t t' : ClifTy} {x y : UInt64} (ha : Γ[a]? = some (.sc t x))
    (hv : Γ[v]? = some (.sc t' y)) (hS : S.holds w) (hJ : J.faultOk = true ∨ Fits w.mem x (tyBytes ty))
    (hS' : afterStore (some x) none (tyBytes ty) S = S')
    (hk : ∀ w', S'.holds w' → wp cfg d J k Q Γ w') : wp cfg d J (.store v a k) Q Γ w :=
  wp_store_known S _ ha hv hS hJ rfl (fun w' h => hk w' (by subst hS'; exact afterStore_forget h))

theorem all_cons_true {β : Type} {f : β → Bool} {a : β} {l : List β} (h1 : f a = true)
    (h2 : l.all f = true) : (a :: l).all f = true := by
  simp only [List.all_cons, h1, h2, Bool.and_self]

/-- A store to an address known only by its bounds: the facts it provably
    cannot reach survive. -/
theorem wp_store_sub {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty : ClifTy} {v : Slot ty} {a : Slot .i64} {k : Prog Slot Lvl α}
    (S S' : TState) {t t' : ClifTy} {x y : UInt64} (ha : Γ[a]? = some (.sc t x))
    (hv : Γ[v]? = some (.sc t' y)) (hS : S.holds w) (hJ : J.faultOk = true ∨ Fits w.mem x (tyBytes ty))
    (hsub : S'.Sublist S) (hkeep : S'.all (Contracts.storeKeeps x (tyBytes ty)) = true)
    (hk : ∀ w', S'.holds w' → wp cfg d J k Q Γ w') : wp cfg d J (.store v a k) Q Γ w :=
  wp_store_addr S _ ha hv hS hJ rfl fun w' h => hk w' fun f hf => h f (by
    show f ∈ S.filter (Contracts.storeKeeps x (tyBytes ty))
    exact List.mem_filter.mpr ⟨hsub.subset hf, List.all_eq_true.mp hkeep f hf⟩)

theorem wp_store_unknown {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty : ClifTy} {v : Slot ty} {a : Slot .i64} {k : Prog Slot Lvl α}
    (S S' : TState) (hS : S.holds w) (hJ : J.faultOk = true) (hS' : afterStore none none (tyBytes ty) S = S')
    (hk : ∀ w', S'.holds w' → wp cfg d J k Q Γ w') : wp cfg d J (.store v a k) Q Γ w :=
  wp_store_state S none (fun _ _ h => by cases h) hS (.inl hJ) (by subst hS'; exact hk)

/-- Lanes stored one after another keep every region's size and the frozen flag. -/
theorem foldlM_lane_stores_inv {addr : UInt64} {lw : Nat} : ∀ (l : List (UInt64 × Nat)) (m0 m : Mem),
    l.foldlM (fun mm (p : UInt64 × Nat) => mm.store (addr + UInt64.ofNat (p.2 * lw)) lw p.1) m0 = some m →
    m.sizes = m0.sizes ∧ m.frozen = m0.frozen
  | [], m0, m, h => by simp at h; subst h; exact ⟨rfl, rfl⟩
  | p :: l, m0, m, h => by
      simp only [List.foldlM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
      obtain ⟨m1, h1, h2⟩ := h
      obtain ⟨hs, hz⟩ := foldlM_lane_stores_inv l m1 m h2
      exact ⟨hs.trans (Contracts.store_sizes_eq h1), hz.trans (Contracts.store_frozen h1)⟩

/-- **A store under default flags, by the typestate**: its width is the
    value's own. -/
theorem wp_storeUnaligned_state {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty : ClifTy} {v : Slot ty} {a : Slot .i64} {k : Prog Slot Lvl α}
    (S : TState) (st : Option (UInt64 × ClifTy × UInt64))
    (hst : ∀ x t' y, st = some (x, t', y) → (∃ t, Γ[a]? = some (.sc t x)) ∧ Γ[v]? = some (.sc t' y))
    (hS : S.holds w) (hJ : J.faultOk = true ∨ ∃ x t' y, st = some (x, t', y) ∧ Fits w.mem x (tyBytes t'))
    (hk : ∀ w', (afterStore (st.map (·.1)) (st.map (·.2.2)) (tyBytes ((st.map (·.2.1)).getD .i8)) S).holds w' →
      wp cfg d J k Q Γ w') :
    wp cfg d J (.storeUnaligned v a k) Q Γ w := by
  rw [wp_storeUnaligned]
  refine ⟨fun m hf => ?_, ?_⟩
  · rcases hJ with hJ | ⟨x, t', y, rfl, hsp⟩
    · exact hJ
    obtain ⟨⟨t, ha⟩, hv⟩ := hst x t' y rfl
    obtain ⟨m', hm⟩ := hsp.store y
    simp only [runStmt, Sem.get, ha, hv, hm, reduceCtorEq] at hf
  intro w' hrun
  apply hk
  simp only [runStmt, Sem.get] at hrun
  cases hv : Γ[v]? with
  | none => rw [hv] at hrun; simp only [reduceCtorEq] at hrun
  | some val =>
  cases ha : Γ[a]? with
  | none => rw [hv, ha] at hrun; simp only [reduceCtorEq] at hrun
  | some av =>
  cases av with
  | vec _ _ => rw [hv, ha] at hrun; simp only [reduceCtorEq] at hrun
  | sc ta addr =>
  cases val with
  | vec tv ls =>
      -- lane by lane: every fact but the cells is kept
      rw [hv, ha] at hrun
      simp only at hrun
      split at hrun
      · rename_i m hm
        cases hrun
        cases st with
        | none =>
            rw [← Array.foldlM_toList] at hm
            obtain ⟨hs, hz⟩ := foldlM_lane_stores_inv _ _ _ hm
            exact Contracts.nonCell_kept hS (store_parts hz) (fun k _ => by cases k <;> rfl) rfl ⟨rfl, rfl, rfl⟩ hs
        | some q =>
            obtain ⟨x, t', y⟩ := q
            obtain ⟨-, hv'⟩ := hst x t' y rfl
            rw [hv] at hv'; cases hv'
      · cases hrun
  | sc tb b =>
      rw [hv, ha] at hrun
      simp only at hrun
      cases hm : w.mem.store addr (tyBytes tb) b with
      | none => rw [hm] at hrun; cases hrun
      | some m =>
          rw [hm] at hrun; cases hrun
          cases st with
          | none =>
              exact Contracts.nonCell_kept hS (store_parts (Contracts.store_frozen hm)) (fun k _ => by cases k <;> rfl) rfl ⟨rfl, rfl, rfl⟩
                (Contracts.store_sizes_eq hm)
          | some q =>
              obtain ⟨x, t', y⟩ := q
              obtain ⟨⟨t0, ht⟩, hv'⟩ := hst x t' y rfl
              rw [ha] at ht; cases ht
              rw [hv] at hv'; cases hv'
              exact afterStore_sound hS (store_parts (Contracts.store_frozen hm)) (fun k _ => by cases k <;> rfl) rfl ⟨rfl, rfl, rfl⟩ (Contracts.store_sizes_eq hm)
                (fun _ => hm) (fun _ h => by cases h; rfl) (fun _ h => by cases h; rfl) (tyBytes_le _)

theorem wp_storeU_known {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty : ClifTy} {v : Slot ty} {a : Slot .i64} {k : Prog Slot Lvl α}
    (S S' : TState) {t t' : ClifTy} {x y : UInt64} (ha : Γ[a]? = some (.sc t x))
    (hv : Γ[v]? = some (.sc t' y)) (hS : S.holds w) (hJ : J.faultOk = true ∨ Fits w.mem x (tyBytes t'))
    (hS' : afterStore (some x) (some y) (tyBytes t') S = S')
    (hk : ∀ w', S'.holds w' → wp cfg d J k Q Γ w') : wp cfg d J (.storeUnaligned v a k) Q Γ w :=
  wp_storeUnaligned_state S (some (x, t', y)) (fun _ _ _ h => by cases h; exact ⟨⟨t, ha⟩, hv⟩) hS
    (hJ.imp_right fun h => ⟨x, t', y, rfl, h⟩)
    (by subst hS'; exact hk)

/-- A store under default flags to a known address of a value the typestate
    does not name: the cells its width cannot reach survive. -/
theorem wp_storeU_addr {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty : ClifTy} {v : Slot ty} {a : Slot .i64} {k : Prog Slot Lvl α}
    (S S' : TState) {t t' : ClifTy} {x y : UInt64} (ha : Γ[a]? = some (.sc t x))
    (hv : Γ[v]? = some (.sc t' y)) (hS : S.holds w) (hJ : J.faultOk = true ∨ Fits w.mem x (tyBytes t'))
    (hS' : afterStore (some x) none (tyBytes t') S = S')
    (hk : ∀ w', S'.holds w' → wp cfg d J k Q Γ w') : wp cfg d J (.storeUnaligned v a k) Q Γ w :=
  wp_storeU_known S _ ha hv hS hJ rfl (fun w' h => hk w' (by subst hS'; exact afterStore_forget h))

/-- A store under default flags to a known address of a value whose type is
    not known: whatever its width, it is at most sixteen bytes, and the cells
    beyond those survive. -/
theorem wp_storeU_wide {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty : ClifTy} {v : Slot ty} {a : Slot .i64} {k : Prog Slot Lvl α}
    (S S' : TState) {t t' : ClifTy} {x y : UInt64} (ha : Γ[a]? = some (.sc t x))
    (hv : Γ[v]? = some (.sc t' y)) (hS : S.holds w) (hJ : J.faultOk = true ∨ Fits w.mem x 16)
    (hS' : afterStore (some x) none 16 S = S')
    (hk : ∀ w', S'.holds w' → wp cfg d J k Q Γ w') : wp cfg d J (.storeUnaligned v a k) Q Γ w :=
  wp_storeU_addr S _ ha hv hS (hJ.imp_right (Fits.mono (by cases t' <;> decide))) rfl fun w' h => hk w' (by
    subst hS'
    intro f hf
    have hf' : f ∈ S.filter (Contracts.storeKeeps x 16) := hf
    obtain ⟨hm, hkp⟩ := List.mem_filter.mp hf'
    exact h f (List.mem_filter.mpr ⟨hm, Contracts.storeKeeps_mono hkp (by cases t' <;> decide)⟩))

/-- A store under default flags to an address known by its bounds: the facts
    sixteen bytes from it provably miss survive. -/
theorem wp_storeU_sub {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty : ClifTy} {v : Slot ty} {a : Slot .i64} {k : Prog Slot Lvl α}
    (S S' : TState) {t t' : ClifTy} {x y : UInt64} (ha : Γ[a]? = some (.sc t x))
    (hv : Γ[v]? = some (.sc t' y)) (hS : S.holds w) (hJ : J.faultOk = true ∨ Fits w.mem x 16)
    (hsub : S'.Sublist S) (hkeep : S'.all (Contracts.storeKeeps x 16) = true)
    (hk : ∀ w', S'.holds w' → wp cfg d J k Q Γ w') : wp cfg d J (.storeUnaligned v a k) Q Γ w :=
  wp_storeU_wide S _ ha hv hS hJ rfl fun w' h => hk w' fun f hf => h f (by
    show f ∈ S.filter (Contracts.storeKeeps x 16)
    exact List.mem_filter.mpr ⟨hsub.subset hf, List.all_eq_true.mp hkeep f hf⟩)

theorem wp_storeU_unknown {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty : ClifTy} {v : Slot ty} {a : Slot .i64} {k : Prog Slot Lvl α}
    (S S' : TState) (hS : S.holds w) (hJ : J.faultOk = true) (hS' : afterStore none none 1 S = S')
    (hk : ∀ w', S'.holds w' → wp cfg d J k Q Γ w') : wp cfg d J (.storeUnaligned v a k) Q Γ w :=
  wp_storeUnaligned_state S none (fun _ _ _ h => by cases h) hS (.inl hJ) (by subst hS'; exact hk)

/-- **A byte store, by the typestate**: it writes one byte, so it keeps every
    cell it does not reach when its address is known, and every fact but the
    cells otherwise. -/
theorem wp_istore8_state {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty : ClifTy} {v : Slot ty} {a : Slot .i64} {hi : ty.isInt = true}
    {k : Prog Slot Lvl α} (S : TState) (ao : Option UInt64)
    (ha : ∀ x, ao = some x → ∃ t, Γ[a]? = some (.sc t x)) (hS : S.holds w)
    (hJ : J.faultOk = true ∨ ∃ x, ao = some x ∧ Fits w.mem x 1)
    (hk : ∀ w', (afterStore ao none 1 S).holds w' → wp cfg d J k Q Γ w') :
    wp cfg d J (.istore8 v a hi k) Q Γ w := by
  rw [wp_istore8]
  refine ⟨fun m hf => ?_, ?_⟩
  · rcases hJ with hJ | ⟨x, rfl, hsp⟩
    · exact hJ
    obtain ⟨t, ha⟩ := ha x rfl
    simp only [runStmt, Sem.get, ha] at hf
    cases hv : Γ[v]? with
    | none => simp [hv] at hf
    | some val =>
      cases val with
      | vec _ _ => simp [hv] at hf
      | sc _ b =>
        obtain ⟨m', hm⟩ := hsp.store (b &&& 0xff)
        simp [hv, hm] at hf
  intro w' hrun
  apply hk
  simp only [runStmt, Sem.get] at hrun
  cases hv : Γ[v]? with
  | none => rw [hv] at hrun; simp only [reduceCtorEq] at hrun
  | some val =>
  cases ha' : Γ[a]? with
  | none => rw [hv, ha'] at hrun; simp only [reduceCtorEq] at hrun
  | some av =>
  cases av with
  | vec _ _ => rw [hv, ha'] at hrun; simp only [reduceCtorEq] at hrun
  | sc ta addr =>
  cases val with
  | vec _ _ => rw [hv, ha'] at hrun; simp only [reduceCtorEq] at hrun
  | sc tb b =>
      rw [hv, ha'] at hrun
      simp only at hrun
      cases hm : w.mem.store addr 1 (b &&& 0xff) with
      | none => rw [hm] at hrun; simp only [reduceCtorEq] at hrun
      | some m =>
          rw [hm] at hrun; cases hrun
          exact afterStore_sound hS (store_parts (Contracts.store_frozen hm)) (fun k _ => by cases k <;> rfl) rfl ⟨rfl, rfl, rfl⟩ (Contracts.store_sizes_eq hm)
            (fun _ => hm) (fun x hx => by
              obtain ⟨t, ht⟩ := ha x hx
              rw [ha'] at ht; cases ht; rfl) (fun _ h => by cases h) (by decide)

theorem wp_istore8_known {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty : ClifTy} {v : Slot ty} {a : Slot .i64} {hi : ty.isInt = true}
    {k : Prog Slot Lvl α} (S S' : TState) {t : ClifTy} {x : UInt64} (ha : Γ[a]? = some (.sc t x))
    (hS : S.holds w) (hJ : J.faultOk = true ∨ Fits w.mem x 1) (hS' : afterStore (some x) none 1 S = S')
    (hk : ∀ w', S'.holds w' → wp cfg d J k Q Γ w') : wp cfg d J (.istore8 v a hi k) Q Γ w :=
  wp_istore8_state S (some x) (fun _ h => by cases h; exact ⟨t, ha⟩) hS (hJ.imp_right fun h => ⟨x, rfl, h⟩) (by subst hS'; exact hk)

/-- A store to any address keeps every fact but the cells. -/
theorem afterStore_nonCell {S : TState} {w : World} {a : UInt64} {n : Nat}
    (h : (afterStore (some a) none n S).holds w) : TState.holds (S.filter (!·.isCell)) w := by
  intro f hf
  obtain ⟨hm, hc⟩ := List.mem_filter.mp hf
  apply h
  refine List.mem_filter.mpr ⟨hm, ?_⟩
  cases f <;> simp_all [Contracts.storeKeeps, Contracts.Fact.isCell]

/-- **A byte store to an address known by its bounds**: it fits where the post
    allows no fault, and keeps every fact but the cells, without asking which
    cells it may reach. -/
theorem wp_istore8_sym {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty : ClifTy} {v : Slot ty} {a : Slot .i64} {hi : ty.isInt = true}
    {k : Prog Slot Lvl α} (S S' : TState) {t : ClifTy} {x : UInt64} (ha : Γ[a]? = some (.sc t x))
    (hS : S.holds w) (hJ : J.faultOk = true ∨ Fits w.mem x 1) (hS' : S.filter (!·.isCell) = S')
    (hk : ∀ w', S'.holds w' → wp cfg d J k Q Γ w') : wp cfg d J (.istore8 v a hi k) Q Γ w :=
  wp_istore8_known S _ ha hS hJ rfl fun w' h => hk w' (hS' ▸ afterStore_nonCell h)

theorem wp_istore8_unknown {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty : ClifTy} {v : Slot ty} {a : Slot .i64} {hi : ty.isInt = true}
    {k : Prog Slot Lvl α} (S S' : TState) (hS : S.holds w) (hJ : J.faultOk = true) (hS' : afterStore none none 1 S = S')
    (hk : ∀ w', S'.holds w' → wp cfg d J k Q Γ w') : wp cfg d J (.istore8 v a hi k) Q Γ w :=
  wp_istore8_state S none (fun _ h => by cases h) hS (.inl hJ) (by subst hS'; exact hk)

theorem evalOp_iadd {m : Mem} {Γ : Env} {a b : R} {t : ClifTy} {x y : UInt64}
    (ha : Γ[a]? = some (.sc t x)) (hb : Γ[b]? = some (.sc t y)) (ht : t.isInt = true) :
    evalOp m Γ (.iadd a b) = some (norm t (x + y)) := by
  show bin Γ a b _ = _
  simp only [bin, Sem.get, ha, hb]
  have : (t == t) = true := by cases t <;> rfl
  simp [ht, this]

/-- A 64-bit addition over values known symbolically: their sum, as a term. -/
theorem evalOp_iadd64 {m : Mem} {Γ : Env} {a b : R} {x y : UInt64}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) :
    evalOp m Γ (.iadd a b) = some (.sc .i64 (x + y)) := by
  rw [evalOp_iadd ha hb rfl]
  simp only [norm, widthMask]
  congr 2
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_and]
  have := (x + y).toNat_lt
  show (x + y).toNat &&& (2^64 - 1) = (x + y).toNat
  rw [Nat.and_two_pow_sub_one_eq_mod]
  omega

theorem norm64 (z : UInt64) : norm .i64 z = .sc .i64 z := by
  simp only [norm, widthMask]
  congr 1
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_and]
  have := z.toNat_lt
  show z.toNat &&& (2^64 - 1) = z.toNat
  rw [Nat.and_two_pow_sub_one_eq_mod]
  omega

/-- The lesser of two 64-bit values is at most each. -/
theorem umin_bounds (x y : UInt64) :
    (UInt64.ofNat (min x.toNat y.toNat)).toNat ≤ x.toNat ∧ (UInt64.ofNat (min x.toNat y.toNat)).toNat ≤ y.toNat := by
  have hx := x.toNat_lt
  have hy := y.toNat_lt
  rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega)]
  omega

/-- The lesser of two 64-bit values known symbolically, as a term. -/
theorem evalOp_umin64 {m : Mem} {Γ : Env} {a b : R} {x y : UInt64}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) :
    evalOp m Γ (.ibin .umin a b) = some (.sc .i64 (UInt64.ofNat (min x.toNat y.toNat))) := by
  show bin Γ a b _ = _
  simp only [bin, Sem.get, ha, hb]
  simp [ibinOp, ClifTy.isInt, norm64]
  have hw : (widthMask ClifTy.i64).toNat = 2 ^ 64 - 1 := rfl
  refine ⟨rfl, ?_⟩
  rw [hw, Nat.and_two_pow_sub_one_eq_mod, Nat.and_two_pow_sub_one_eq_mod,
    Nat.mod_eq_of_lt x.toNat_lt, Nat.mod_eq_of_lt y.toNat_lt]

/-- The leading zeros of a 64-bit value, as `clz` counts them. -/
def clz64 (x : UInt64) : UInt64 :=
  UInt64.ofNat ((bitsDown 64 x.toNat).takeWhile (· == false)).length

/-- A 64-bit value's bits in reverse order, as `bitrev` lays them. -/
def bitrev64 (x : UInt64) : UInt64 :=
  UInt64.ofNat ((List.range 64).foldl
    (fun acc i => if (x &&& widthMask .i64).toNat.testBit i then acc ||| (1 <<< (64 - 1 - i)) else acc) 0)

theorem and_widthMask64 (x : UInt64) : x &&& widthMask .i64 = x := by
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_and]
  show x.toNat &&& (2 ^ 64 - 1) = x.toNat
  rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt (UInt64.toNat_lt x)]

theorem toNat_and_widthMask64 (x : UInt64) : x.toNat &&& (widthMask .i64).toNat = x.toNat := by
  rw [← UInt64.toNat_and, and_widthMask64]

theorem evalOp_clz64 {m : Mem} {Γ : Env} {a : R} {x : UInt64} (ha : Γ[a]? = some (.sc .i64 x)) :
    evalOp m Γ (.iun .clz a) = some (.sc .i64 (clz64 x)) := by
  show un Γ a _ _ = _
  simp only [un, Sem.get, ha]
  simp [iunOp, ClifTy.isInt, ClifTy.width, toNat_and_widthMask64, norm64, clz64]

theorem evalOp_bitrev64 {m : Mem} {Γ : Env} {a : R} {x : UInt64} (ha : Γ[a]? = some (.sc .i64 x)) :
    evalOp m Γ (.iun .bitrev a) = some (.sc .i64 (bitrev64 x)) := by
  show un Γ a _ _ = _
  simp only [un, Sem.get, ha]
  simp [iunOp, ClifTy.isInt, ClifTy.width, norm64, bitrev64]

theorem bitsDown_getElem (n k : Nat) (hk : k < (bitsDown 64 n).length) :
    (bitsDown 64 n)[k] = n.testBit (63 - k) := by
  simp [bitsDown] at hk ⊢

/-- With `c` leading zeros a value is below `2^(64-c)`, and at least
    `2^(63-c)` where it has a one. -/
theorem clz_bounds (n : Nat) (hn : n < 2^64) :
    n < 2^(64 - ((bitsDown 64 n).takeWhile (· == false)).length) ∧
    (((bitsDown 64 n).takeWhile (· == false)).length < 64 →
      2^(63 - ((bitsDown 64 n).takeWhile (· == false)).length) ≤ n) := by
  generalize hT : (bitsDown 64 n).takeWhile (· == false) = T
  generalize hD : (bitsDown 64 n).dropWhile (· == false) = D
  have hsplit : T ++ D = bitsDown 64 n := by rw [← hT, ← hD]; exact List.takeWhile_append_dropWhile
  have hlen : (bitsDown 64 n).length = 64 := by simp [bitsDown]
  have hlen' : T.length + D.length = 64 := by rw [← List.length_append, hsplit, hlen]
  have hall : ∀ x ∈ T, (x == false) = true := by
    have := List.all_takeWhile (p := (· == false)) (l := bitsDown 64 n)
    rw [hT, List.all_eq_true] at this; exact this
  refine ⟨?_, ?_⟩
  · apply Nat.lt_pow_two_of_testBit
    intro i hi
    by_cases h64 : i < 64
    · have hk : 63 - i < T.length := by omega
      have e1 : (bitsDown 64 n)[63 - i]'(by omega) = T[63 - i] := by
        simp only [← hsplit]; rw [List.getElem_append_left hk]
      have h1 := hall _ (List.getElem_mem hk)
      rw [← e1, bitsDown_getElem] at h1
      have e : 63 - (63 - i) = i := by omega
      rw [e] at h1; simpa using h1
    · exact Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le hn (Nat.pow_le_pow_right (by decide) (by omega)))
  · intro hc
    have hne : D ≠ [] := by intro h; rw [h] at hlen'; simp at hlen'; omega
    have hh := List.head_dropWhile_not (· == false) (l := bitsDown 64 n) (by rw [hD]; exact hne)
    have e1 : (bitsDown 64 n)[T.length]'(by omega)
        = D[0]'(by cases D with | nil => exact absurd rfl hne | cons => simp) := by
      simp only [← hsplit]; rw [List.getElem_append_right (by omega)]; simp
    have e2 : ((bitsDown 64 n).dropWhile (· == false)).head (by rw [hD]; exact hne)
        = D[0]'(by cases D with | nil => exact absurd rfl hne | cons => simp) := by
      subst hD; simp [List.head_eq_getElem]
    rw [e2, ← e1, bitsDown_getElem] at hh
    have : n.testBit (63 - T.length) = true := by simpa using hh
    exact Nat.ge_two_pow_of_testBit this

theorem bitrev64_zero : bitrev64 0 = 0 := by decide

theorem clz64_toNat (n : UInt64) :
    (clz64 n).toNat = ((bitsDown 64 n.toNat).takeWhile (· == false)).length := by
  have h := (List.takeWhile_prefix (· == false) (l := bitsDown 64 n.toNat)).length_le
  have h2 : (bitsDown 64 n.toNat).length = 64 := by simp [bitsDown]
  simp only [clz64, UInt64.toNat_ofNat']
  apply Nat.mod_eq_of_lt; omega

/-- A bit-reversed index, shifted down by one past the leading zeros of a bound
    above the index, stays below the bound: it is the index's low bits
    reversed, as many as the bound's highest bit is high. Stated as the
    arithmetic reads it, where the `+ 1` may be a computed one. -/
theorem bitrev_index_lt (j n c : UInt64) (hc : c = 1) :
    n.toNat ≤ j.toNat ∨ (bitrev64 j >>> ((clz64 n + c) % 64)).toNat < n.toNat := by
  subst hc
  rcases Nat.lt_or_ge j.toNat n.toNat with hj | hj
  · right
    have hb := clz_bounds n.toNat n.toNat_lt
    have hz := clz64_toNat n
    generalize ((bitsDown 64 n.toNat).takeWhile (· == false)).length = cv at hb hz
    obtain ⟨hlt, hge⟩ := hb
    have hc1 : cv < 64 := by
      apply Nat.lt_of_not_le; intro h
      have : 64 - cv = 0 := by omega
      rw [this] at hlt; omega
    by_cases h63 : cv = 63
    · subst h63
      have hn : n.toNat < 2 := by simpa using hlt
      have hj0 : j = 0 := by apply UInt64.toNat_inj.mp; simp; omega
      subst hj0; rw [bitrev64_zero]; simp; omega
    · have hs : ((clz64 n + 1) % 64).toNat = cv + 1 := by
        simp [UInt64.toNat_add, UInt64.toNat_mod, hz]; omega
      rw [UInt64.toNat_shiftRight, hs, Nat.mod_eq_of_lt (show cv + 1 < 64 by omega)]
      have hr := (bitrev64 j).toNat_lt
      have h2 : (bitrev64 j).toNat >>> (cv + 1) < 2^(63 - cv) := by
        rw [Nat.shiftRight_eq_div_pow]
        apply Nat.div_lt_of_lt_mul
        rw [← Nat.pow_add]; have : cv + 1 + (63 - cv) = 64 := by omega
        rw [this]; exact hr
      have := hge hc1; omega
  · left; exact hj

theorem evalOp_ireduce32_64 {m : Mem} {Γ : Env} {a : R} {x : UInt64} (ha : Γ[a]? = some (.sc .i64 x)) :
    evalOp m Γ (.ireduce32 a) = some (.sc .i32 (x &&& widthMask .i32)) := by
  show un Γ a _ _ = _
  simp only [un, Sem.get, ha]
  rfl

theorem evalOp_isub64 {m : Mem} {Γ : Env} {a b : R} {x y : UInt64}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) :
    evalOp m Γ (.isub a b) = some (.sc .i64 (x - y)) := by
  show bin Γ a b _ = _
  simp only [bin, Sem.get, ha, hb]
  have e1 : (ClifTy.i64 == ClifTy.i64) = true := rfl
  have e2 : ClifTy.i64.isInt = true := rfl
  simp [e1, e2, norm64]

theorem evalOp_imul64 {m : Mem} {Γ : Env} {a b : R} {x y : UInt64}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) :
    evalOp m Γ (.imul a b) = some (.sc .i64 (x * y)) := by
  show bin Γ a b _ = _
  simp only [bin, Sem.get, ha, hb]
  have e1 : (ClifTy.i64 == ClifTy.i64) = true := rfl
  have e2 : ClifTy.i64.isInt = true := rfl
  simp [e1, e2, norm64]

theorem evalOp_udiv64 {m : Mem} {Γ : Env} {a b : R} {x y : UInt64}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) (hy : y ≠ 0) :
    evalOp m Γ (.udiv a b) = some (.sc .i64 (x / y)) := by
  show bin Γ a b _ = _
  simp only [bin, Sem.get, ha, hb]
  have e1 : (ClifTy.i64 == ClifTy.i64) = true := rfl
  have e2 : ClifTy.i64.isInt = true := rfl
  simp [e1, e2, norm64, hy]

theorem evalOp_select64 {m : Mem} {Γ : Env} {c a b : R} {tc : ClifTy} {z x y : UInt64}
    (hc : Γ[c]? = some (.sc tc z)) (htc : tc.isInt = true)
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) :
    evalOp m Γ (.select c a b) = some (.sc .i64 (if Sem.isTrue (.sc tc z) then x else y)) := by
  simp only [evalOp, Sem.get, hc, ha, hb, Option.bind_eq_bind, Option.bind_some, htc]
  split
  · split <;> simp_all
  · rename_i h; exact absurd h (by simp only [V.ty]; decide)

theorem evalOp_icmp64 {m : Mem} {Γ : Env} {c : ICmpCond} {a b : R} {x y : UInt64}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) :
    evalOp m Γ (.icmp c a b) = some (.sc .i8 (if cmpInt c .i64 x y then 1 else 0)) := by
  simp only [evalOp, Sem.get, ha, hb, Option.bind_eq_bind, Option.bind_some, zipIntCmp, boolV]
  rfl

theorem evalOp_bandNot64 {m : Mem} {Γ : Env} {a b : R} {x y : UInt64}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) :
    evalOp m Γ (.bandNot a b) = some (.sc .i64 (x &&& ~~~y)) := by
  simp [evalOp, Sem.get, ha, hb, zipIntBits, norm64, ClifTy.isInt]; decide

theorem evalOp_ishl64 {m : Mem} {Γ : Env} {a b : R} {x y : UInt64} {tb : ClifTy}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc tb y)) (htb : tb.isInt = true) :
    evalOp m Γ (.ishl a b) = some (.sc .i64 (x <<< (y % 64))) := by
  show shiftBin Γ a b _ = _
  simp only [shiftBin, Sem.get, ha, hb]
  have e2 : ClifTy.i64.isInt = true := rfl
  simp [htb, e2, norm64, ClifTy.width]

theorem evalOp_ushr64 {m : Mem} {Γ : Env} {a b : R} {x y : UInt64} {tb : ClifTy}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc tb y)) (htb : tb.isInt = true) :
    evalOp m Γ (.ushr a b) = some (.sc .i64 (x >>> (y % 64))) := by
  show shiftBin Γ a b _ = _
  simp only [shiftBin, Sem.get, ha, hb]
  have e2 : ClifTy.i64.isInt = true := rfl
  have hm : x &&& (18446744073709551615 : UInt64) = x := by
    apply UInt64.toNat_inj.mp
    rw [UInt64.toNat_and]
    have := x.toNat_lt
    show x.toNat &&& (2^64 - 1) = x.toNat
    rw [Nat.and_two_pow_sub_one_eq_mod]
    omega
  simp [htb, e2, norm64, ClifTy.width, widthMask, hm]

theorem evalOp_band64 {m : Mem} {Γ : Env} {a b : R} {x y : UInt64}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) :
    evalOp m Γ (.band a b) = some (.sc .i64 (x &&& y)) := by
  simp [evalOp, Sem.get, ha, hb, zipIntBits, norm64, ClifTy.isInt]; decide

theorem evalOp_bor64 {m : Mem} {Γ : Env} {a b : R} {x y : UInt64}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) :
    evalOp m Γ (.bor a b) = some (.sc .i64 (x ||| y)) := by
  simp [evalOp, Sem.get, ha, hb, zipIntBits, norm64, ClifTy.isInt]; decide

theorem evalOp_bxor64 {m : Mem} {Γ : Env} {a b : R} {x y : UInt64}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) :
    evalOp m Γ (.bxor a b) = some (.sc .i64 (x ^^^ y)) := by
  simp [evalOp, Sem.get, ha, hb, zipIntBits, norm64, ClifTy.isInt]; decide

/-- An integer operation that asks its operands be of one type: when one is a
    64-bit value, so is the other, or the operation is undefined. -/
theorem left_i64 {m : Mem} {Γ : Env} {a b : R} {t : ClifTy} {x y : UInt64} {v : V} (o : Op)
    (ho : o = .iadd a b ∨ o = .isub a b ∨ o = .imul a b ∨ o = .band a b ∨ o = .bor a b ∨ o = .bxor a b)
    (ha : Γ[a]? = some (.sc t x)) (hb : Γ[b]? = some (.sc .i64 y)) (h : evalOp m Γ o = some v) :
    t = .i64 := by
  rcases ho with rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp only [evalOp, bin, zipIntBits, Sem.get, ha, hb, Option.bind_eq_bind, Option.bind_some] at h <;>
    (split at h <;> (try cases h) <;> (cases t <;> first | rfl | (rename_i hc; exact absurd hc (by decide))))

/-- `left_i64`, the value of a type not known on the right. -/
theorem right_i64 {m : Mem} {Γ : Env} {a b : R} {t : ClifTy} {x y : UInt64} {v : V} (o : Op)
    (ho : o = .iadd a b ∨ o = .isub a b ∨ o = .imul a b ∨ o = .band a b ∨ o = .bor a b ∨ o = .bxor a b)
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc t y)) (h : evalOp m Γ o = some v) :
    t = .i64 := by
  rcases ho with rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp only [evalOp, bin, zipIntBits, Sem.get, ha, hb, Option.bind_eq_bind, Option.bind_some] at h <;>
    (split at h <;> (try cases h) <;> (cases t <;> first | rfl | (rename_i hc; exact absurd hc (by decide))))

theorem evalOp_load {m : Mem} {Γ : Env} {a : R} {op : LoadOp} {t : ClifTy} {x v : UInt64}
    (hk : op.kind = .plain) (hl0 : op.ty.lanes = none)
    (ha : Γ[a]? = some (.sc t x)) (hl : m.load x (tyBytes op.ty) = some v) :
    evalOp m Γ (.load op a) = some (.sc op.ty v) := by
  unfold evalOp
  simp only [Sem.get, ha]
  simp [hk, hl0, hl]

/-- Whether an operation reads memory: only a load does. -/
def opReadsMem : Op → Bool
  | .load _ _ => true
  | _ => false

/-- An operation with its operands renamed to the slots `0, 1, 2`, in the order
    `Op.regs` lists them. -/
def renumberOp : Op → Op
  | .iconst ty k => .iconst ty k
  | .fconst ty b => .fconst ty b
  | .iadd _ _ => .iadd 0 1
  | .isub _ _ => .isub 0 1
  | .imul _ _ => .imul 0 1
  | .udiv _ _ => .udiv 0 1
  | .ineg _ => .ineg 0
  | .ishl _ _ => .ishl 0 1
  | .ushr _ _ => .ushr 0 1
  | .band _ _ => .band 0 1
  | .bandNot _ _ => .bandNot 0 1
  | .bor _ _ => .bor 0 1
  | .bxor _ _ => .bxor 0 1
  | .ireduce32 _ => .ireduce32 0
  | .uextend64 _ => .uextend64 0
  | .sextend64 _ => .sextend64 0
  | .icmp c _ _ => .icmp c 0 1
  | .select _ _ _ => .select 0 1 2
  | .bitselect _ _ _ => .bitselect 0 1 2
  | .ibin k _ _ => .ibin k 0 1
  | .ishift k _ _ => .ishift k 0 1
  | .iun k _ => .iun k 0
  | .fbin k _ _ => .fbin k 0 1
  | .fun1 k _ => .fun1 k 0
  | .fconv k t _ => .fconv k t 0
  | .fma _ _ _ => .fma 0 1 2
  | .iext k t _ => .iext k t 0
  | .ctz _ => .ctz 0
  | .popcnt _ => .popcnt 0
  | .fadd _ _ => .fadd 0 1
  | .fsub _ _ => .fsub 0 1
  | .fmul _ _ => .fmul 0 1
  | .fmax _ _ => .fmax 0 1
  | .fmin _ _ => .fmin 0 1
  | .fneg _ => .fneg 0
  | .fpromote _ => .fpromote 0
  | .fcmp c _ _ => .fcmp c 0 1
  | .fcvtFromSint ty _ => .fcvtFromSint ty 0
  | .fcvtToUint ty _ => .fcvtToUint ty 0
  | .splat ty _ => .splat ty 0
  | .extractlane _ l => .extractlane 0 l
  | .vhighBits _ => .vhighBits 0
  | .bitcast ty _ => .bitcast ty 0
  | .load op _ => .load op 0

/-! **An operation's value is a function of its operands' values.** An
    operation that reads no memory computes, in any environment, what it
    computes renumbered over just its operands' values and with any memory — so
    once those values are known, its own is computed rather than proved case by
    case. One lemma per operand count. -/

theorem evalOp_renumber0 {m m' : Mem} {Γ : Env} {o : Op}
    (hr : o.regs = []) (hl : opReadsMem o = false) :
    evalOp m Γ o = evalOp m' #[] (renumberOp o) := by
  cases o <;> simp_all [Op.regs, opReadsMem, renumberOp, evalOp]

theorem evalOp_renumber1 {m m' : Mem} {Γ : Env} {o : Op} {a : R} {x : V}
    (hr : o.regs = [a]) (hl : opReadsMem o = false) (ha : Γ[a]? = some x) :
    evalOp m Γ o = evalOp m' #[x] (renumberOp o) := by
  cases o <;> simp only [Op.regs, List.cons.injEq, List.ne_cons_self, and_true, List.cons_ne_nil] at hr <;>
    (try subst hr) <;> simp_all [opReadsMem, renumberOp, evalOp, un, Sem.get]

theorem evalOp_renumber2 {m m' : Mem} {Γ : Env} {o : Op} {a b : R} {x y : V}
    (hr : o.regs = [a, b]) (hl : opReadsMem o = false)
    (ha : Γ[a]? = some x) (hb : Γ[b]? = some y) :
    evalOp m Γ o = evalOp m' #[x, y] (renumberOp o) := by
  cases o <;> simp only [Op.regs, List.cons.injEq, List.cons_ne_nil, and_false, List.nil_eq] at hr <;>
    (try obtain ⟨rfl, rfl⟩ := hr) <;> simp_all [opReadsMem, renumberOp, evalOp, bin, shiftBin, Sem.get]

theorem evalOp_renumber3 {m m' : Mem} {Γ : Env} {o : Op} {a b c : R} {x y z : V}
    (hr : o.regs = [a, b, c]) (hl : opReadsMem o = false)
    (ha : Γ[a]? = some x) (hb : Γ[b]? = some y) (hc : Γ[c]? = some z) :
    evalOp m Γ o = evalOp m' #[x, y, z] (renumberOp o) := by
  cases o <;> simp only [Op.regs, List.cons.injEq, List.cons_ne_nil, and_false, List.nil_eq] at hr <;>
    (try obtain ⟨rfl, rfl, rfl⟩ := hr) <;> simp_all [opReadsMem, renumberOp, evalOp, Sem.get]

theorem entry_get {Γ arr : Env} {i : Nat} {v : V} (h : Γ = arr) (hi : arr[i]? = some v) : Γ[i]? = some v :=
  h ▸ hi

/-- **A body that keeps a typestate never misuses a call**: if from every world
    in `W` the body's weakest precondition holds, no run of what `emit` ships
    misuses anything, for any input. -/
theorem safe_of_wp {cfg : Cfg} {p : Body} {params : List ClifTy} (W : World → Prop)
    (hwp : ∀ Γ w, W w → wp cfg 0 { ok := fun _ _ => True } p (fun _ _ _ => True) Γ w)
    (hf : Fine (emitGo p ⟨params.length, 0, [], [], [], none⟩).2)
    {args : List V} {w : World} (hargs : args.length = params.length) (hw : W w) (m : String) :
    Sem.run cfg args w (emit p params) ≠ .misuse m :=
  run_safe (emit_triple (P := fun _ w => W w)
    (PT.conseq (wp_sound cfg 0 _ p _) (fun Γ w h => hwp Γ w h) (fun _ _ _ h => h)) hf
    (fun Γ w' ⟨e1, e2⟩ => by subst e1 e2; exact ⟨by simp [hargs], hw⟩)) m

/-- The arguments the runtime hands an entry point: the arena's, the data's and
    the output's bases, and the data's and the output's lengths as the caller
    gives them. -/
def runArgs (dataLen outLen : UInt64) : List V :=
  [.sc .i64 (regionBase .arena), .sc .i64 (regionBase .data), .sc .i64 dataLen,
   .sc .i64 (regionBase .out), .sc .i64 outLen]

/-- The same, for a body whose safety depends on what it is handed: the
    environment it starts from holds the arguments, so a body may use what it
    knows of them, such as which region a pointer points into. -/
theorem safe_of_wp_entry {cfg : Cfg} {p : Body} {params : List ClifTy} (W : World → Prop) (args : List V)
    (hwp : ∀ Γ w, Γ = args.toArray → W w → wp cfg 0 { ok := fun _ _ => True } p (fun _ _ _ => True) Γ w)
    (hf : Fine (emitGo p ⟨params.length, 0, [], [], [], none⟩).2)
    {w : World} (hargs : args.length = params.length) (hw : W w) (m : String) :
    Sem.run cfg args w (emit p params) ≠ .misuse m :=
  run_safe (emit_triple (P := fun Γ w => Γ = args.toArray ∧ W w)
    (PT.conseq (wp_sound cfg 0 _ p _) (fun Γ w h => hwp Γ w h.1 h.2) (fun _ _ _ h => h)) hf
    (fun Γ w' ⟨e1, e2⟩ => by subst e1 e2; exact ⟨by simp [hargs], rfl, hw⟩)) m

/-- `safe_of_wp_entry` under a post that allows no fault: every load and store
    the body makes finds its memory, and every call it makes answers. -/
theorem sound_of_wp_entry {cfg : Cfg} {p : Body} {params : List ClifTy} (W : World → Prop) (args : List V)
    (hwp : ∀ Γ w, Γ = args.toArray → W w →
      wp cfg 0 { ok := fun _ _ => True, faultOk := false } p (fun _ _ _ => True) Γ w)
    (hf : Fine (emitGo p ⟨params.length, 0, [], [], [], none⟩).2)
    {w : World} (hargs : args.length = params.length) (hw : W w) (m : String) :
    Sem.run cfg args w (emit p params) ≠ .misuse m ∧ Sem.run cfg args w (emit p params) ≠ .fault m :=
  run_sound (emit_triple (P := fun Γ w => Γ = args.toArray ∧ W w)
    (PT.conseq (wp_sound cfg 0 _ p _) (fun Γ w h => hwp Γ w h.1 h.2) (fun _ _ _ h => h)) hf
    (fun Γ w' ⟨e1, e2⟩ => by subst e1 e2; exact ⟨by simp [hargs], rfl, hw⟩)) rfl m

/-- `safe_of_wp_entry`, for a body that answers a status. -/
theorem safe_of_wp_entry_ans {cfg : Cfg} {α : Type} {p : Prog Slot Lvl α} {params : List ClifTy}
    (W : World → Prop) (args : List V)
    (hwp : ∀ Γ w, Γ = args.toArray → W w → wp cfg 0 { ok := fun _ _ => True } p (fun _ _ _ => True) Γ w)
    (hf : Fine (emitGo p ⟨params.length, 0, [], [], [], none⟩).2)
    {w : World} (hargs : args.length = params.length) (hw : W w) (m : String) :
    Sem.run cfg args w (emitAns p params) ≠ .misuse m :=
  run_safe (emitAns_triple (P := fun Γ w => Γ = args.toArray ∧ W w)
    (PT.conseq (wp_sound cfg 0 _ p _) (fun Γ w h => hwp Γ w h.1 h.2) (fun _ _ _ h => h)) hf
    (fun Γ w' ⟨e1, e2⟩ => by subst e1 e2; exact ⟨by simp [hargs], rfl, hw⟩)) m

/-- `sound_of_wp_entry`, for a body that answers a status. -/
theorem sound_of_wp_entry_ans {cfg : Cfg} {α : Type} {p : Prog Slot Lvl α} {params : List ClifTy}
    (W : World → Prop) (args : List V)
    (hwp : ∀ Γ w, Γ = args.toArray → W w →
      wp cfg 0 { ok := fun _ _ => True, faultOk := false } p (fun _ _ _ => True) Γ w)
    (hf : Fine (emitGo p ⟨params.length, 0, [], [], [], none⟩).2)
    {w : World} (hargs : args.length = params.length) (hw : W w) (m : String) :
    Sem.run cfg args w (emitAns p params) ≠ .misuse m ∧ Sem.run cfg args w (emitAns p params) ≠ .fault m :=
  run_sound (emitAns_triple (P := fun Γ w => Γ = args.toArray ∧ W w)
    (PT.conseq (wp_sound cfg 0 _ p _) (fun Γ w h => hwp Γ w h.1 h.2) (fun _ _ _ h => h)) hf
    (fun Γ w' ⟨e1, e2⟩ => by subst e1 e2; exact ⟨by simp [hargs], rfl, hw⟩)) rfl m

-- ---------------------------------------------------------------------------
-- Environments as variables
--
-- The condition generator never carries an environment as a term. Each
-- binding leaves a fresh environment `Γ'` with two facts: `Ext Γ Γ'`, and what
-- `Γ'` holds at the new slot. Proving the rest for every such `Γ'` proves it
-- for the one the model builds, and a slot's value is found by carrying its
-- fact forward along `Ext`, so no lookup walks a chain of pushes.
-- ---------------------------------------------------------------------------

/-- `Γ'` keeps every slot of `Γ`. -/
def Ext (Γ Γ' : Env) : Prop := Γ.size ≤ Γ'.size ∧ ∀ i, i < Γ.size → Γ'[i]? = Γ[i]?

theorem Ext.push (Γ : Env) (v : V) : Ext Γ (Γ.push v) :=
  ⟨by simp, fun i hi => by simp [Array.getElem?_push, show i ≠ Γ.size by omega]⟩

theorem Ext.append (Γ δ : Env) : Ext Γ (Γ ++ δ) :=
  ⟨by simp, fun i hi => Array.getElem?_append_left hi⟩

/-- A slot's value, carried to an environment that keeps it. -/
theorem Ext.fact {Γ Γ' : Env} {s : Nat} {c : V} (h : Ext Γ Γ') (hs : Γ[s]? = some c) : Γ'[s]? = some c := by
  have hi : s < Γ.size := by
    rcases Nat.lt_or_ge s Γ.size with hi | hi
    · exact hi
    · rw [Array.getElem?_eq_none hi] at hs; cases hs
  rw [h.2 s hi, hs]

theorem lookup_nil (Γ : Env) : ([] : List Nat).mapM (fun r => Γ[r]?) = some [] := rfl

theorem lookup_cons {Γ : Env} {s : Nat} {c : V} {ss : List Nat} {cs : List V} (h : Γ[s]? = some c)
    (hs : ss.mapM (fun r => Γ[r]?) = some cs) : (s :: ss).mapM (fun r => Γ[r]?) = some (c :: cs) := by
  simp [List.mapM_cons, h, hs]

section vars
variable {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop} {Γ : Env} {w : World}

/-- The operations whose value, when they have one, is a scalar. -/
def _root_.AlgorithmLib.HProg.Op.scalarResult : Op → Bool
  | .iconst .. | .iadd .. | .isub .. | .imul .. | .udiv .. | .ishl .. | .ushr .. | .ineg .. | .ctz ..
  | .popcnt .. | .ireduce32 .. | .uextend64 .. | .sextend64 .. | .ibin .. | .ishift .. | .iun .. => true
  | .load op _ => op.ty.lanes.isNone
  | _ => false

theorem evalOp_scalar {m : Mem} {Γ : Env} {o : Op} {v : V} (hs : o.scalarResult = true)
    (h : evalOp m Γ o = some v) : ∃ t x, v = .sc t x := by
  cases o <;> simp only [Op.scalarResult] at hs <;> (try cases hs)
  all_goals simp only [evalOp, bin, shiftBin, un] at h
  all_goals decode_inv
  all_goals first
    | exact ⟨_, _, rfl⟩
    | (simp only [ibinOp, iunOp] at h; split at h <;> decode_inv <;> exact ⟨_, _, rfl⟩)
    | (simp only [ishiftOp]; split <;> exact ⟨_, _, rfl⟩)
    | (simp_all)

theorem wp_op_sc_var {ty} {o : Op' Slot ty} {k : Slot ty → Prog Slot Lvl α} (hs : o.erase.scalarResult = true)
    (hJ : J.faultOk = true)
    (h : ∀ t x, evalOp w.mem Γ o.erase = some (.sc t x) → ∀ Γ', Ext Γ Γ' → Γ'[Γ.size]? = some (.sc t x) →
      wp cfg d J (k Γ.size) Q Γ' w) :
    wp cfg d J (.op o k) Q Γ w := ⟨fun _ => hJ, fun v hv => by
  obtain ⟨t, x, rfl⟩ := evalOp_scalar hs hv
  exact h t x hv _ (Ext.push Γ _) (by simp)⟩

/-- `wp_op_sc_var`, where the post allows no fault: the operation answers. -/
theorem wp_op_sc_ok {ty} {o : Op' Slot ty} {k : Slot ty → Prog Slot Lvl α} (hs : o.erase.scalarResult = true)
    (hJ : J.faultOk = true ∨ evalOp w.mem Γ o.erase ≠ none)
    (h : ∀ t x, evalOp w.mem Γ o.erase = some (.sc t x) → ∀ Γ', Ext Γ Γ' → Γ'[Γ.size]? = some (.sc t x) →
      wp cfg d J (k Γ.size) Q Γ' w) :
    wp cfg d J (.op o k) Q Γ w := ⟨fun hn => hJ.elim id fun h' => absurd hn h', fun v hv => by
  obtain ⟨t, x, rfl⟩ := evalOp_scalar hs hv
  exact h t x hv _ (Ext.push Γ _) (by simp)⟩

/-- An operation that answers a scalar of type `T` when it answers, and answers
    where the post allows no fault: its value, of that type. -/
theorem wp_op_scT {ty} {o : Op' Slot ty} {k : Slot ty → Prog Slot Lvl α} (T : ClifTy)
    (hsc : ∀ v, evalOp w.mem Γ o.erase = some v → ∃ x, v = .sc T x)
    (hJ : J.faultOk = true ∨ evalOp w.mem Γ o.erase ≠ none)
    (h : ∀ x, evalOp w.mem Γ o.erase = some (.sc T x) → ∀ Γ', Ext Γ Γ' → Γ'[Γ.size]? = some (.sc T x) →
      wp cfg d J (k Γ.size) Q Γ' w) :
    wp cfg d J (.op o k) Q Γ w := ⟨fun hn => hJ.elim id fun h' => absurd hn h', fun v hv => by
  obtain ⟨x, rfl⟩ := hsc v hv
  exact h x hv _ (Ext.push Γ _) (by simp)⟩

/-- An operation `tyOp` types, over operands of the types it asks: its value, of
    the type it names. -/
theorem wp_op_ty {ty} {o : Op' Slot ty} {k : Slot ty → Prog Slot Lvl α} (T : ClifTy)
    (hty : ∃ x, evalOp w.mem Γ o.erase = some (.sc T x))
    (h : ∀ x, evalOp w.mem Γ o.erase = some (.sc T x) → ∀ Γ', Ext Γ Γ' → Γ'[Γ.size]? = some (.sc T x) →
      wp cfg d J (k Γ.size) Q Γ' w) :
    wp cfg d J (.op o k) Q Γ w := by
  obtain ⟨x, hx⟩ := hty
  refine wp_op_scT T (fun v hv => ?_) (Or.inr (by simp [hx])) h
  rw [hx] at hv; cases hv; exact ⟨x, rfl⟩

/-- A scalar load answers a value of the type it loads. -/
theorem evalOp_load_ty {m : Mem} {Γ : Env} {op : LoadOp} {a : R} {t : ClifTy} {x : UInt64}
    (hl : op.ty.lanes = none) (h : evalOp m Γ (.load op a) = some (.sc t x)) : t = op.ty := by
  simp only [evalOp] at h
  revert h
  generalize Sem.get Γ a = g
  rcases g with _ | ⟨_, addr⟩ | _ <;> simp only [Option.bind, bind, reduceCtorEq, false_imp_iff]
  cases op.kind <;> simp only [hl] <;> split <;> simp [norm, ofInt] <;> intros <;> simp_all

theorem wp_op_load_var {ty} {o : Op' Slot ty} {k : Slot ty → Prog Slot Lvl α} {op : LoadOp} {a : R}
    {t : ClifTy} (ho : o.erase = .load op a) (hl : op.ty.lanes = none) (ht : op.ty = t)
    (hJ : J.faultOk = true)
    (h : ∀ x, evalOp w.mem Γ o.erase = some (.sc t x) → ∀ Γ', Ext Γ Γ' → Γ'[Γ.size]? = some (.sc t x) →
      wp cfg d J (k Γ.size) Q Γ' w) :
    wp cfg d J (.op o k) Q Γ w := ⟨fun _ => hJ, fun v hv => by
  have hs : o.erase.scalarResult = true := by rw [ho]; simp [Op.scalarResult, hl]
  obtain ⟨t', x, rfl⟩ := evalOp_scalar hs hv
  have := evalOp_load_ty hl (ho ▸ hv)
  subst ht; subst this
  exact h x hv _ (Ext.push Γ _) (by simp)⟩

/-- How many bytes a scalar load reads. -/
def loadBytes (op : LoadOp) : Nat :=
  match op.kind with
  | .uload8 | .sload8 => 1
  | .uload16 | .sload16 => 2
  | .uload32 | .sload32 => 4
  | .plain => tyBytes op.ty

/-- A scalar load from an address whose bytes are there answers. -/
theorem evalOp_load_some {m : Mem} {Γ : Env} {op : LoadOp} {a : R} {t : ClifTy} {x : UInt64}
    (hl : op.ty.lanes = none) (ha : Γ[a]? = some (.sc t x)) (h : ∃ b, m.load x (loadBytes op) = some b) :
    evalOp m Γ (.load op a) ≠ none := by
  obtain ⟨b, hb⟩ := h
  intro hn
  simp only [evalOp, Sem.get, ha] at hn
  unfold loadBytes at hb
  cases hk : op.kind <;> simp_all

/-- A scalar load from a known address, by the typestate: where the post
    allows no fault, its bytes are there. -/
theorem wp_op_load_at {ty} {o : Op' Slot ty} {k : Slot ty → Prog Slot Lvl α} {op : LoadOp} {a : R}
    {t : ClifTy} (ho : o.erase = .load op a) (hl : op.ty.lanes = none) (ht : op.ty = t)
    {t' : ClifTy} {x : UInt64} (ha : Γ[a]? = some (.sc t' x))
    (hJ : J.faultOk = true ∨ Fits w.mem x (loadBytes op))
    (h : ∀ x, evalOp w.mem Γ o.erase = some (.sc t x) → ∀ Γ', Ext Γ Γ' → Γ'[Γ.size]? = some (.sc t x) →
      wp cfg d J (k Γ.size) Q Γ' w) :
    wp cfg d J (.op o k) Q Γ w := ⟨fun hn => hJ.elim id fun hsp =>
    absurd (ho ▸ hn) (evalOp_load_some hl ha hsp.load), fun v hv => by
  have hs : o.erase.scalarResult = true := by rw [ho]; simp [Op.scalarResult, hl]
  obtain ⟨t', x, rfl⟩ := evalOp_scalar hs hv
  have := evalOp_load_ty hl (ho ▸ hv)
  subst ht; subst this
  exact h x hv _ (Ext.push Γ _) (by simp)⟩

/-- A byte read is below 256. -/
theorem load1_lt {m : Mem} {a b : UInt64} (h : m.load a 1 = some b) : b.toNat < 256 := by
  unfold Mem.load at h
  rcases hd : decodeAddr a with _ | ⟨r, off⟩ <;> simp only [hd, bind, Option.bind, reduceCtorEq] at h
  split at h
  · cases h
  · cases h
    simp only [List.range_one, List.foldr_cons, List.foldr_nil, UInt64.zero_shiftLeft, UInt64.zero_or,
      UInt8.toNat_toUInt64]
    exact UInt8.toNat_lt _

/-- So is a byte load's answer, zero-extended. -/
theorem evalOp_uload8_lt {m : Mem} {Γ : Env} {op : LoadOp} {a : R} {t : ClifTy} {x : UInt64}
    (hk : op.kind = .uload8) (h : evalOp m Γ (.load op a) = some (.sc t x)) : x.toNat < 256 := by
  simp only [evalOp] at h
  revert h
  generalize Sem.get Γ a = g
  rcases g with _ | ⟨_, addr⟩ | _ <;> simp only [Option.bind, bind, reduceCtorEq, false_imp_iff, hk]
  cases hl : m.load addr 1 with
  | none => simp
  | some b =>
    simp only [norm, Option.some.injEq, V.sc.injEq]
    rintro ⟨-, rfl⟩
    exact Nat.lt_of_le_of_lt (by rw [UInt64.toNat_and]; exact Nat.and_le_left) (load1_lt hl)

/-- A byte load zero-extended, from a known address: `wp_op_load_at`, its
    answer below 256. -/
theorem wp_op_uload8_at {ty} {o : Op' Slot ty} {k : Slot ty → Prog Slot Lvl α} {op : LoadOp} {a : R}
    {t : ClifTy} (ho : o.erase = .load op a) (hl : op.ty.lanes = none) (ht : op.ty = t)
    (hk8 : op.kind = .uload8)
    {t' : ClifTy} {x : UInt64} (ha : Γ[a]? = some (.sc t' x))
    (hJ : J.faultOk = true ∨ Fits w.mem x (loadBytes op))
    (h : ∀ x, evalOp w.mem Γ o.erase = some (.sc t x) → x.toNat < 256 → ∀ Γ', Ext Γ Γ' →
      Γ'[Γ.size]? = some (.sc t x) → wp cfg d J (k Γ.size) Q Γ' w) :
    wp cfg d J (.op o k) Q Γ w :=
  wp_op_load_at ho hl ht ha hJ fun x hv => h x hv (evalOp_uload8_lt hk8 (ho ▸ hv))

theorem wp_op_var {ty} {o : Op' Slot ty} {k : Slot ty → Prog Slot Lvl α} (hJ : J.faultOk = true)
    (h : ∀ v, evalOp w.mem Γ o.erase = some v → ∀ Γ', Ext Γ Γ' → Γ'[Γ.size]? = some v →
      wp cfg d J (k Γ.size) Q Γ' w) :
    wp cfg d J (.op o k) Q Γ w := ⟨fun _ => hJ, fun v hv => h v hv _ (Ext.push Γ v) (by simp)⟩

theorem wp_iconst_var {ty} {k : Int} {hk : ty.isInt = true} {c : Slot ty → Prog Slot Lvl α}
    (h : ∀ Γ', Ext Γ Γ' → Γ'[Γ.size]? = some (ofInt ty k) → wp cfg d J (c Γ.size) Q Γ' w) :
    wp cfg d J (.op (.iconst ty k hk) c) Q Γ w := by
  rw [wp_iconst]; exact h _ (Ext.push Γ _) (by simp)

/-- A constant computed by a term, taken as the number it computes. -/
theorem wp_iconst_lit {ty} {k k' : Int} {hk : ty.isInt = true} {c : Slot ty → Prog Slot Lvl α}
    (he : k = k')
    (h : ∀ Γ', Ext Γ Γ' → Γ'[Γ.size]? = some (ofInt ty k') → wp cfg d J (c Γ.size) Q Γ' w) :
    wp cfg d J (.op (.iconst ty k hk) c) Q Γ w := by
  subst he; exact wp_iconst_var h

theorem wp_forLoop_var {n : Slot .i64} {body : Slot .i64 → Prog Slot Lvl Unit}
    {R : Unit → Env → World → Prop} (W : World → Prop) (hW : W w)
    (hbody : ∀ d' J', J'.faultOk = true → ∀ i Γb wb, Ext Γ Γb → W wb → wp cfg d' J' (body i) (fun _ _ w' => W w') Γb wb)
    (hk : ∀ Γ' w', Ext Γ Γ' → W w' → R () Γ' w') (hJ : J.faultOk = true) :
    wp cfg d J (forLoop n body) R Γ w :=
  wp_forLoop W hW (fun d' J' hJ' i δ wb h => hbody d' J' hJ' i _ wb (Ext.append Γ δ) h)
    (fun δ w' h => hk _ w' (Ext.append Γ δ) h) hJ

theorem wp_forLoopAcc_var {t : ClifTy} {n : Slot .i64} {acc0 : Slot t}
    {body : Slot .i64 → Slot t → Prog Slot Lvl (Slot t)} {R : Slot t → Env → World → Prop}
    (W : World → Prop) (hW : W w)
    (hbody : ∀ d' J', J'.faultOk = true → ∀ i a Γ' wb, Ext Γ Γ' → W wb → wp cfg d' J' (body i a) (fun _ _ w' => W w') Γ' wb)
    (hk : ∀ a Γ' w', Ext Γ Γ' → W w' → R a Γ' w') (hJ : J.faultOk = true) :
    wp cfg d J (forLoopAcc n acc0 body) R Γ w :=
  wp_forLoopAcc W hW (fun d' J' hJ' i a δ wb h => hbody d' J' hJ' i a _ wb (Ext.append Γ δ) h)
    (fun a δ w' h => hk a _ w' (Ext.append Γ δ) h) hJ

theorem wp_ffiVoid_var {f : Ffi} {args : Vals Slot f.params} {Q : Unit → Env → World → Prop}
    (W W' : World → Prop) (hres : f.result.isSome = false) {cs : List V}
    (hcs : args.slots.mapM (fun r => Γ[r]?) = some cs) (hK : Moves cfg W f cs W') (hW : W w)
    (hQ : ∀ w', W' w' → Q () Γ w') :
    wp cfg d J (ffiVoid f args) Q Γ w :=
  wp_ffiVoid_moves W W' hres (fun vs hvs => by rw [hcs] at hvs; cases hvs; exact hK) hW hQ

theorem wp_ffiVoid_res_var {f : Ffi} {args : Vals Slot f.params} {Q : Unit → Env → World → Prop}
    (W W' : World → Prop) (hres : f.result.isSome = true) {cs : List V}
    (hcs : args.slots.mapM (fun r => Γ[r]?) = some cs) (hK : Moves cfg W f cs W') (hW : W w)
    (hQ : ∀ v w' Γ', Ext Γ Γ' → Γ'[Γ.size]? = some v → W' w' → Q () Γ' w') :
    wp cfg d J (ffiVoid f args) Q Γ w :=
  wp_ffiVoid_moves_res W W' hres (fun vs hvs => by rw [hcs] at hvs; cases hvs; exact hK) hW
    (fun v w' h => hQ v w' _ (Ext.push Γ v) (by simp) h)

/-- **A call whose answer is bound**, by its contract: what follows gets the
    answer's slot, holding whatever the call answered. -/
theorem wp_call_res_var {f : Ffi} {args : Vals Slot f.params} {k : ResV Slot f.result → Prog Slot Lvl α}
    {Q : α → Env → World → Prop}
    (W W' : World → Prop) (hres : f.result.isSome = true) {cs : List V}
    (hcs : args.slots.mapM (fun r => Γ[r]?) = some cs) (hK : Moves cfg W f cs W') (hW : W w)
    (hQ : ∀ v w' Γ', Ext Γ Γ' → Γ'[Γ.size]? = some v → W' w' →
      wp cfg d J (k (resSlot f.result Γ.size)) Q Γ' w') :
    wp cfg d J (.call f args k) Q Γ w := by
  rw [wp_call]
  intro vs hvs
  rw [hcs] at hvs; cases hvs
  obtain ⟨r, w', hc, hw'⟩ := hK w hW
  exact ⟨r, w', hc, by simp only [hres, if_true]; exact fun v _ => hQ v w' _ (Ext.push Γ v) (by simp) hw'⟩

/-- The same, knowing the answer is a scalar: every entry point but the thread
    spawn answers one, so what follows may pass it on with known bits. -/
theorem wp_call_res_sc {f : Ffi} {args : Vals Slot f.params} {k : ResV Slot f.result → Prog Slot Lvl α}
    {Q : α → Env → World → Prop}
    (W W' : World → Prop) {t0 : ClifTy} (ht : f.result = some t0) (hf : f ≠ .threadSpawn) {cs : List V}
    (hcs : args.slots.mapM (fun r => Γ[r]?) = some cs) (hK : Moves cfg W f cs W') (hW : W w)
    (hQ : ∀ x w' Γ', Ext Γ Γ' → Γ'[Γ.size]? = some (.sc t0 x) → W' w' →
      wp cfg d J (k (resSlot f.result Γ.size)) Q Γ' w') :
    wp cfg d J (.call f args k) Q Γ w := by
  rw [wp_call]
  intro vs hvs
  rw [hcs] at hvs; cases hvs
  obtain ⟨r, w', hc, hw'⟩ := hK w hW
  refine ⟨r, w', hc, ?_⟩
  simp only [ht, Option.isSome_some, if_true]
  intro v hv; subst hv
  obtain ⟨x, rfl⟩ := Contracts.callOf_ffi_scalar hf hc
  have ht' : f.result.getD .i64 = t0 := by rw [ht]; rfl
  rw [ht']
  exact hQ x w' _ (Ext.push Γ _) (by simp) hw'

/-- A call that answers went through the table on its argument bits. -/
theorem callOf_ffi_bits {f : Ffi} (hf : f ≠ .threadSpawn) {lc : Locals} {vs : List V} {w : World}
    {v : V} {w' : World} (h : callOf lc (.ffi f) vs w = some (some v, w')) :
    ∃ bits, vs.mapM asBits = some bits ∧ Sem.callBits f bits w = some (some v, w') := by
  have hs : (f == .threadSpawn) = false := by simpa using hf
  simp only [callOf, hs, if_false, Bool.false_eq_true, callImport, Contracts.ofCname_cname, bind, Option.bind,
    callFfi] at h
  split at h
  · cases h
  · split at h
    · cases h
    · rename_i bits hb
      exact ⟨bits, hb, h⟩

/-- The same, knowing what the table says of the answer (`answerOk`): a bound
    what follows may count by. -/
theorem wp_call_res_ans {f : Ffi} {args : Vals Slot f.params} {k : ResV Slot f.result → Prog Slot Lvl α}
    {Q : α → Env → World → Prop}
    (W W' : World → Prop) {t0 : ClifTy} (ht : f.result = some t0) (hf : f ≠ .threadSpawn) {cs : List V}
    {bits : List UInt64} (hb : cs.mapM asBits = some bits)
    (hcs : args.slots.mapM (fun r => Γ[r]?) = some cs) (hK : Moves cfg W f cs W') (hW : W w)
    (hQ : ∀ x w' Γ', Ext Γ Γ' → Γ'[Γ.size]? = some (.sc t0 x) → answerOk f bits x → W' w' →
      wp cfg d J (k (resSlot f.result Γ.size)) Q Γ' w') :
    wp cfg d J (.call f args k) Q Γ w := by
  rw [wp_call]
  intro vs hvs
  rw [hcs] at hvs; cases hvs
  obtain ⟨r, w', hc, hw'⟩ := hK w hW
  refine ⟨r, w', hc, ?_⟩
  simp only [ht, Option.isSome_some, if_true]
  intro v hv; subst hv
  obtain ⟨x, rfl⟩ := Contracts.callOf_ffi_scalar hf hc
  obtain ⟨bits', hb', hc'⟩ := callOf_ffi_bits hf hc
  rw [hb] at hb'; cases hb'
  have ht' : f.result.getD .i64 = t0 := by rw [ht]; rfl
  rw [ht'] at hc' ⊢
  exact hQ x w' _ (Ext.push Γ _) (by simp) (answerOk_sound hc') hw'

/-- The same for a call that answers nothing. -/
theorem wp_call_var {f : Ffi} {args : Vals Slot f.params} {k : ResV Slot f.result → Prog Slot Lvl α}
    {Q : α → Env → World → Prop}
    (W W' : World → Prop) (hres : f.result.isSome = false) {cs : List V}
    (hcs : args.slots.mapM (fun r => Γ[r]?) = some cs) (hK : Moves cfg W f cs W') (hW : W w)
    (hQ : ∀ w', W' w' → wp cfg d J (k (resSlot f.result 0)) Q Γ w') :
    wp cfg d J (.call f args k) Q Γ w := by
  rw [wp_call]
  intro vs hvs
  rw [hcs] at hvs; cases hvs
  obtain ⟨r, w', hc, hw'⟩ := hK w hW
  exact ⟨r, w', hc, by simp only [hres, Bool.false_eq_true, if_false]; exact hQ w' hw'⟩

/-- An operation whose value is known binds it. -/
theorem wp_op_val {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty : ClifTy} {o : Op' Slot ty} {k : Slot ty → Prog Slot Lvl α} (c : V)
    (hc : evalOp w.mem Γ o.erase = some c)
    (h : ∀ Γ', Ext Γ Γ' → Γ'[Γ.size]? = some c → wp cfg d J (k Γ.size) Q Γ' w) :
    wp cfg d J (.op o k) Q Γ w := by
  rw [wp_op]
  refine ⟨fun hn => (by rw [hc] at hn; cases hn), ?_⟩
  intro v hv
  rw [hc] at hv; cases hv
  exact h _ (Ext.push Γ _) (by simp)

/-- An operation over a value of a type not known and a 64-bit one: where it
    is defined, it is the 64-bit operation. -/
theorem wp_op_left64 {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty : ClifTy} {o : Op' Slot ty} {k : Slot ty → Prog Slot Lvl α}
    {a b : R} {t : ClifTy} {x y : UInt64} (r : UInt64)
    (ho : o.erase = .iadd a b ∨ o.erase = .isub a b ∨ o.erase = .imul a b ∨ o.erase = .band a b ∨
      o.erase = .bor a b ∨ o.erase = .bxor a b)
    (ha : Γ[a]? = some (.sc t x)) (hb : Γ[b]? = some (.sc .i64 y))
    (hc : Γ[a]? = some (.sc .i64 x) → evalOp w.mem Γ o.erase = some (.sc .i64 r))
    (hJ : J.faultOk = true)
    (h : ∀ Γ', Ext Γ Γ' → Γ'[Γ.size]? = some (.sc .i64 r) → wp cfg d J (k Γ.size) Q Γ' w) :
    wp cfg d J (.op o k) Q Γ w := by
  rw [wp_op]
  refine ⟨fun _ => hJ, ?_⟩
  intro v hv
  obtain rfl := left_i64 o.erase ho ha hb hv
  rw [hc ha] at hv; cases hv
  exact h _ (Ext.push Γ _) (by simp)

/-- `wp_op_left64`, the value of a type not known on the right. -/
theorem wp_op_right64 {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty : ClifTy} {o : Op' Slot ty} {k : Slot ty → Prog Slot Lvl α}
    {a b : R} {t : ClifTy} {x y : UInt64} (r : UInt64)
    (ho : o.erase = .iadd a b ∨ o.erase = .isub a b ∨ o.erase = .imul a b ∨ o.erase = .band a b ∨
      o.erase = .bor a b ∨ o.erase = .bxor a b)
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc t y))
    (hc : Γ[b]? = some (.sc .i64 y) → evalOp w.mem Γ o.erase = some (.sc .i64 r))
    (hJ : J.faultOk = true)
    (h : ∀ Γ', Ext Γ Γ' → Γ'[Γ.size]? = some (.sc .i64 r) → wp cfg d J (k Γ.size) Q Γ' w) :
    wp cfg d J (.op o k) Q Γ w := by
  rw [wp_op]
  refine ⟨fun _ => hJ, ?_⟩
  intro v hv
  obtain rfl := right_i64 o.erase ho ha hb hv
  rw [hc hb] at hv; cases hv
  exact h _ (Ext.push Γ _) (by simp)

/-- A 64-bit widening of an integer: its bits under the source's mask, where
    it is defined. -/
theorem wp_op_uext64 {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty : ClifTy} {o : Op' Slot ty} {k : Slot ty → Prog Slot Lvl α}
    {a : R} {t : ClifTy} {x : UInt64} (ho : o.erase = .uextend64 a) (ha : Γ[a]? = some (.sc t x))
    (hJ : J.faultOk = true ∨ (t.isInt && decide (t.width < 64)) = true)
    (h : ∀ Γ', Ext Γ Γ' → Γ'[Γ.size]? = some (.sc .i64 (x &&& widthMask t)) → wp cfg d J (k Γ.size) Q Γ' w) :
    wp cfg d J (.op o k) Q Γ w := by
  rw [wp_op]
  refine ⟨fun hn => hJ.elim id fun hi => ?_, ?_⟩
  · rw [ho] at hn
    simp only [evalOp, un, Sem.get, ha, Option.bind_eq_bind, Option.bind_some, hi, if_true] at hn
    cases hn
  intro v hv
  rw [ho] at hv
  simp only [evalOp, un, Sem.get, ha, Option.bind_eq_bind, Option.bind_some] at hv
  split at hv
  · cases hv; exact h _ (Ext.push Γ _) (by simp)
  · cases hv

/-- A 64-bit sign extension of an integer, where it is defined. -/
theorem wp_op_sext64 {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty : ClifTy} {o : Op' Slot ty} {k : Slot ty → Prog Slot Lvl α}
    {a : R} {t : ClifTy} {x : UInt64} (ho : o.erase = .sextend64 a) (ha : Γ[a]? = some (.sc t x))
    (hJ : J.faultOk = true ∨ (t.isInt && decide (t.width < 64)) = true)
    (h : ∀ Γ', Ext Γ Γ' → Γ'[Γ.size]? = some (ofInt .i64 (signed t x)) → wp cfg d J (k Γ.size) Q Γ' w) :
    wp cfg d J (.op o k) Q Γ w := by
  rw [wp_op]
  refine ⟨fun hn => hJ.elim id fun hi => ?_, ?_⟩
  · rw [ho] at hn
    simp only [evalOp, un, Sem.get, ha, Option.bind_eq_bind, Option.bind_some, hi, if_true] at hn
    cases hn
  intro v hv
  rw [ho] at hv
  simp only [evalOp, un, Sem.get, ha, Option.bind_eq_bind, Option.bind_some] at hv
  split at hv
  · cases hv; exact h _ (Ext.push Γ _) (by simp)
  · cases hv

end vars

-- ---------------------------------------------------------------------------
-- The engine's own libraries, and branches on what they answer
-- ---------------------------------------------------------------------------

section ext
variable {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop} {Γ : Env} {w : World}

open Contracts (Held extNeeds extAfter extKeeps roomAt) in
/-- **A call of the engine's own libraries whose answer is bound**, by the
    typestate: the handles it is handed are held, its buffers have room, and
    what follows gets the answer's slot, a scalar, and the typestate
    `extAfter` computes from it. -/
theorem wp_ext_res {e : IR.Ext} {args : Vals Slot e.sig.1} {k : ResV Slot e.sig.2 → Prog Slot Lvl α}
    (S S' : TState) {cs : List V} {bits : List UInt64} {hs : List (Held × UInt64)} {rs : List (UInt64 × Nat)}
    (hres : e.sig.2.isSome = true) (ha : e.aside = true)
    (hcs : args.slots.mapM (fun r => Γ[r]?) = some cs) (hb : cs.mapM asBits = some bits)
    (hn : extNeeds e bits = some (hs, rs)) (hS' : S.filter Contracts.Fact.plain = S')
    (hS : S.holds w) (hh : ∀ p ∈ hs, Fact.held p.1 p.2 ∈ S)
    (hr : rs.all (fun q => roomAt S' q.1 q.2) = true)
    (hQ : ∀ t x w' Γ', Ext Γ Γ' → Γ'[Γ.size]? = some (.sc t x) →
      TState.holds (extAfter e bits (some x) S) w' → wp cfg d J (k (resSlot e.sig.2 Γ.size)) Q Γ' w') :
    wp cfg d J (.callLocal ⟨.ext e⟩ args k) Q Γ w := by
  rw [wp_callLocal]
  intro vs hvs
  rw [hcs] at hvs; cases hvs
  have hS0 := TState.holds_obsCall (c := .ext e) (vs := cs) hS
  have hr' : rs.all (fun q => roomAt S q.1 q.2) = true := by
    refine List.all_eq_true.mpr fun q hq => Contracts.roomAt_sub (S' := S') ?_ (List.all_eq_true.mp hr q hq)
    intro x hx; rw [← hS'] at hx; exact (List.mem_filter.mp hx).1
  have hpre := Contracts.extNeeds_sound hn hS0 hh hr'
  have hsome := ExtContracts.ext_pre_safe e bits _ hpre
  rw [Contracts.extCall_bits (cs := e.args bits) (cs' := cs) ha _ (ExtContracts.mapM_asBits_zip _ _ (Contracts.extNeeds_length hn)) hb]
    at hsome
  obtain ⟨⟨r, w'⟩, hc⟩ := Option.isSome_iff_exists.mp hsome
  refine callL_of_some hc ?_
  simp only [hres, if_true]
  intro v hv; subst hv
  obtain ⟨t, x, rfl⟩ := Contracts.lib_scalar ha hc hb
  exact hQ t x w' _ (Ext.push Γ _) (by simp) (Contracts.extAfter_sound hS0 hb hc)

open Contracts (Held extNeeds extAfter extKeeps roomAt) in
/-- The same for a call that answers nothing. -/
theorem wp_ext_void {e : IR.Ext} {args : Vals Slot e.sig.1} {k : ResV Slot e.sig.2 → Prog Slot Lvl α}
    (S S' : TState) {cs : List V} {bits : List UInt64} {hs : List (Held × UInt64)} {rs : List (UInt64 × Nat)}
    (hres : e.sig.2.isSome = false) (ha : e.aside = true)
    (hcs : args.slots.mapM (fun r => Γ[r]?) = some cs) (hb : cs.mapM asBits = some bits)
    (hn : extNeeds e bits = some (hs, rs)) (hS' : S.filter Contracts.Fact.plain = S')
    (hS : S.holds w) (hh : ∀ p ∈ hs, Fact.held p.1 p.2 ∈ S)
    (hr : rs.all (fun q => roomAt S' q.1 q.2) = true)
    (hQ : ∀ w', TState.holds (extAfter e bits none S) w' → wp cfg d J (k (resSlot e.sig.2 0)) Q Γ w') :
    wp cfg d J (.callLocal ⟨.ext e⟩ args k) Q Γ w := by
  rw [wp_callLocal]
  intro vs hvs
  rw [hcs] at hvs; cases hvs
  have hS0 := TState.holds_obsCall (c := .ext e) (vs := cs) hS
  have hr' : rs.all (fun q => roomAt S q.1 q.2) = true := by
    refine List.all_eq_true.mpr fun q hq => Contracts.roomAt_sub (S' := S') ?_ (List.all_eq_true.mp hr q hq)
    intro x hx; rw [← hS'] at hx; exact (List.mem_filter.mp hx).1
  have hpre := Contracts.extNeeds_sound hn hS0 hh hr'
  have hsome := ExtContracts.ext_pre_safe e bits _ hpre
  rw [Contracts.extCall_bits (cs := e.args bits) (cs' := cs) ha _ (ExtContracts.mapM_asBits_zip _ _ (Contracts.extNeeds_length hn)) hb]
    at hsome
  obtain ⟨⟨r, w'⟩, hc⟩ := Option.isSome_iff_exists.mp hsome
  refine callL_of_some hc ?_
  simp only [hres, Bool.false_eq_true, if_false]
  have hafter := Contracts.extAfter_sound hS0 hb hc
  refine hQ w' ?_
  have hop : e.opens = none := by
    cases e <;> simp only [Ext.aside, reduceCtorEq] at ha
    all_goals rename_i f; cases f <;> first | rfl | (simp [Ext.sig] at hres)
  unfold Contracts.extAfter at hafter ⊢
  simp only [hop] at hafter ⊢
  exact hafter

/-- A call to function `i` of the program that answers nothing, where the
    caller holds the callee's summary: from a world its typestate `S0` holds,
    the call ends as the callee does, which is neither a misuse nor a fault,
    and where it answers, what follows starts in the callee's `S'`. -/
theorem wp_local_void {ps : List ClifTy} {res : Option ClifTy} {i : Nat} {args : Vals Slot ps}
    {k : ResV Slot res → Prog Slot Lvl α} (S0 S' : TState) {cs : List V}
    (hres : res.isSome = false) (hcs : args.slots.mapM (fun r => Γ[r]?) = some cs)
    (hsum : Summary cfg.locals i cs S0.holds S'.holds) (hS : S0.holds w)
    (hQ : ∀ w', S'.holds w' → wp cfg d J (k (resSlot res 0)) Q Γ w') :
    wp cfg d J (.callLocal ⟨.local i⟩ args k) Q Γ w := by
  rw [wp_callLocal]
  intro vs hvs
  rw [hcs] at hvs; cases hvs
  have h := hsum _ (TState.holds_obsCall (c := .local i) (vs := cs) hS)
  simp only [callOf, failOf]
  revert h
  cases cfg.locals i cs (obsCall w (.local i) cs) with
  | error f =>
      intro h
      simp only [Except.toOption]
      exact ⟨fun _ => ⟨h.1, fun m hm => absurd (h.2 m hm) Bool.false_ne_true⟩, fun _ _ hc => by cases hc⟩
  | ok p =>
      obtain ⟨x, w'⟩ := p
      intro h
      simp only [Except.toOption, reduceCtorEq, false_implies, Option.some.injEq, Prod.mk.injEq, true_and]
      rintro x' w'' ⟨rfl, rfl⟩
      simp only [hres, Bool.false_eq_true, if_false]
      exact hQ _ h

/-- A summary of a body from its condition: where the condition holds from
    every world `W` holds, with the post that the world left holds `W'`, the
    emitted body run on `args` ends in neither misuse nor fault, and where it
    answers it leaves a world `W'` holds. `summary_succ` takes it. -/
theorem triple_of_wp_entry {cfg : Cfg} {p : Body} {params : List ClifTy} (W W' : World → Prop) (args : List V)
    (hwp : ∀ Γ w, Γ = args.toArray → W w →
      wp cfg 0 { ok := fun _ _ => True, faultOk := false } p (fun _ _ w => W' w) Γ w)
    (hf : Fine (emitGo p ⟨params.length, 0, [], [], [], none⟩).2)
    (hargs : args.length = params.length) (w : World) (hw : W w) :
    Triple cfg (At args.toArray w) (emit p params) { ok := fun _ w' => W' w', faultOk := false } :=
  emit_triple (P := fun Γ w => Γ = args.toArray ∧ W w)
    (PT.conseq (wp_sound cfg 0 _ p _) (fun Γ w h => hwp Γ w h.1 h.2) (fun _ _ _ h => h)) hf
    (fun Γ w' ⟨e1, e2⟩ => by subst e1 e2; exact ⟨by simp [hargs], rfl, hw⟩)

/-- The branch an environment takes keeps what it had. -/
theorem Ext.bindAt_push (Γ : Env) (v : V) {ne : Nat} (h : Γ.size + 1 ≤ ne) : Ext Γ (bindAt (Γ.push v) ne []) := by
  have e : Γ.push v = Γ ++ #[v] := by simp
  rw [e, bindAt_append _ (by omega)]
  exact Ext.append _ _

theorem Ext.refl (Γ : Env) : Ext Γ Γ := ⟨Nat.le_refl _, fun _ _ => rfl⟩

theorem Ext.trans {Γ Γ' Γ'' : Env} (h : Ext Γ Γ') (h' : Ext Γ' Γ'') : Ext Γ Γ'' :=
  ⟨Nat.le_trans h.1 h'.1, fun i hi => by rw [h'.2 i (Nat.lt_of_lt_of_le hi h.1), h.2 i hi]⟩

/-- Binding carries past what an environment keeps still keeps it. -/
theorem Ext.carry {Γ Γb : Env} {n : Nat} (vs : List V) (h : Ext Γ Γb) (hn : Γ.size ≤ n) :
    Ext Γ (bindAt Γb n vs) := by
  refine ⟨by simp [bindAt]; omega, fun i hi => ?_⟩
  have hb : i < Γb.size := Nat.lt_of_lt_of_le hi h.1
  rw [← h.2 i hi]
  simp only [bindAt]
  rw [Array.getElem?_append_left (by simp; omega), Array.getElem?_append_left (by simp; omega)]
  simp only [Array.take, Array.getElem?_extract]
  rw [if_pos (by simp only [Nat.sub_zero, Nat.lt_min]; omega), Nat.zero_add]

/-- An unsigned comparison that held: the first operand was below the second. -/
theorem icmp_ult_true {m : Mem} {Γ : Env} {a b : R} {x y : UInt64} {t : ClifTy} {f : UInt64}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y))
    (h : evalOp m Γ (.icmp .ult a b) = some (.sc t f)) (hf : (f != 0) = true) : x < y := by
  simp only [evalOp, Sem.get, ha, hb, Option.bind_eq_bind, Option.bind_some, zipIntCmp] at h
  simp only [show (ClifTy.i64 == ClifTy.i64 && ClifTy.i64.isInt) = true from rfl, if_true, Option.some.injEq,
    boolV, cmpInt, V.sc.injEq] at h
  obtain ⟨-, rfl⟩ := h
  split at hf
  · simpa using ‹decide (x < y) = true›
  · simp at hf

/-- A loop's carries sit from the slot they are bound at. -/
theorem bindAt_get_carry (Γ : Env) (n : Nat) (cs : List V) (i : Nat) :
    (bindAt Γ n cs)[n + i]? = cs[i]? := by
  have hs : (Γ.take n ++ Array.replicate (n - Γ.size) (default : V)).size = n := by
    simp [Array.take, Array.size_extract]; omega
  simp only [bindAt]
  rw [Array.getElem?_append_right (by omega)]
  simp [hs]

/-- A comparison of two 64-bit values answers. -/
theorem icmp_i64_some {m : Mem} {Γ : Env} {a b : R} {x y : UInt64} (c : ICmpCond)
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) :
    evalOp m Γ (.icmp c a b) ≠ none := by
  simp [evalOp, Sem.get, ha, hb, zipIntCmp]
  decide

/-- **A counted loop threading a bounded accumulator.** A limit of at most `N`
    trips, an accumulator starting at most `A0` and each trip adding at most
    `S` leave it at most `A0 + N * S`, when that fits: the invariant is
    `acc ≤ A0 + i * S` of the trip count `i`. The body sees its trip below `N`
    and room for `S` more without wrapping. -/
theorem wp_forLoopAcc_bnd {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}
    {n acc0 : Slot .i64} {body : Slot .i64 → Slot .i64 → Prog Slot Lvl (Slot .i64)}
    {R : Slot .i64 → Env → World → Prop} (W : World → Prop) (hW : W w) (N A0 S : Nat) {nv a0 : UInt64}
    (hn : Γ[n]? = some (.sc .i64 nv)) (hN : nv.toNat ≤ N) (ha : Γ[acc0]? = some (.sc .i64 a0))
    (hA : a0.toNat ≤ A0) (hfit : A0 + N * S < 2 ^ 64)
    (hbody : ∀ d' (J' : Post), J'.faultOk = J.faultOk → ∀ i a Γb wb x y, Ext Γ Γb → W wb → Γb[i]? = some (.sc .i64 x) →
      Γb[a]? = some (.sc .i64 y) → x.toNat < N → y.toNat + S < 2 ^ 64 → y.toNat ≤ A0 + x.toNat * S →
      wp cfg d' J' (body i a)
        (fun r Γ2 w2 => Ext Γb Γ2 ∧ W w2 ∧ ∃ z, Γ2[r]? = some (.sc .i64 z) ∧ z.toNat ≤ y.toNat + S) Γb wb)
    (hk : ∀ r Γ' w' z, Ext Γ Γ' → W w' → Γ'[r]? = some (.sc .i64 z) → z.toNat ≤ A0 + N * S → R r Γ' w') :
    wp cfg d J (forLoopAcc n acc0 body) R Γ w := by
  simp only [forLoopAcc, wloop2, wloop, wloopL, iconst64, iconst, iaddImm, op, wp_bind, wp_op, wp_ret,
    wp_pure, Op'.erase, evalOp, reduceCtorEq, false_implies, true_and, forall_eq', Option.some.injEq]
  rw [show ofInt ClifTy.i64 0 = V.sc .i64 0 from rfl]
  have hsz : acc0 < Γ.size := by
    rcases Nat.lt_or_ge acc0 Γ.size with h | h
    · exact h
    · rw [Array.getElem?_eq_none h] at ha; cases ha
  have hnsz : n < Γ.size := by
    rcases Nat.lt_or_ge n Γ.size with h | h
    · exact h
    · rw [Array.getElem?_eq_none h] at hn; cases hn
  let Γp := Γ.push (V.sc .i64 0)
  have hEp : Ext Γ Γp := Ext.push Γ _
  rw [wp_loop]
  refine ⟨fun Γ0 cs w => Γ0 = Γp ∧ W w ∧ ∃ i a, cs = [.sc .i64 i, .sc .i64 a] ∧ i.toNat ≤ N ∧
      a.toNat ≤ A0 + i.toNat * S,
    fun Γ0 cs a' Γh wh => (Γ0 = Γp ∧ W wh ∧ ∃ i a, cs = [.sc .i64 i, .sc .i64 a] ∧ i.toNat ≤ N ∧
      a.toNat ≤ A0 + i.toNat * S) ∧ Γh = bindAt Γ0 Γ0.size cs ∧
      a' = (contIfULt Γ0.size n, ((Γ0.size + 1 : Slot .i64) ::ᵥ Vals.nil), ()),
    fun Γ0 Γb vs w => Γ0 = Γp ∧ W w ∧ Ext Γ Γb ∧ ∃ z, vs = [.sc .i64 z] ∧ z.toNat ≤ A0 + N * S,
    ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro cs hcs
    have ha' : Γ[acc0] = V.sc .i64 a0 := by
      rw [Array.getElem?_eq_getElem hsz] at ha; exact Option.some.inj ha
    simp only [Vals.slots, List.mapM_cons, List.mapM_nil, Array.getElem?_push_size,
      Array.getElem?_push_lt hsz, ha', Option.bind_eq_bind, Option.bind_some, Option.pure_def,
      Option.some.injEq] at hcs
    subst hcs
    exact ⟨rfl, hW, 0, a0, rfl, by simp, by simpa using hA⟩
  · rintro n0 Γ0 cs wh rfl hI
    simp only [wp_pure]
    refine ⟨hI, ?_, ?_⟩ <;> first | trivial | rfl
  · rintro Γ0 cs a Γ1 w1 t f vs ⟨⟨rfl, hw, i, av, rfl, hi, hav⟩, rfl, rfl⟩ _ _ hvs
    refine ⟨rfl, hw, ?_, av, ?_, ?_⟩
    · exact (hEp.carry (n := Γp.size) _ (by simp [Γp])).trans (Ext.push _ _)
    · have hsz2 : (bindAt Γp Γp.size [V.sc .i64 i, V.sc .i64 av]).size = Γp.size + 2 := by
        simp [bindAt, Array.take]
      have h1 : (Array.push (bindAt Γp Γp.size [V.sc .i64 i, V.sc .i64 av]) (V.sc t f))[Γp.size + 1]? =
          some (.sc .i64 av) := by
        rw [Array.getElem?_push, if_neg (by omega), bindAt_get_carry]; rfl
      simp only [Vals.slots, List.mapM_cons, List.mapM_nil, h1, Option.bind_eq_bind, Option.bind_some,
        Option.pure_def, Option.some.injEq] at hvs
      exact hvs.symm
    · have := Nat.mul_le_mul_right S hi
      omega
  · rintro nb Γ0 cs a Γ1 t f wb hnb ⟨⟨rfl, hw, i, av, rfl, hi, hav⟩, rfl, rfl⟩ heval hcont
    have hf : (f != 0) = true := by simpa [contIfULt, contIf] using hcont
    simp only [contIfULt, contIf] at heval
    have hE1 : Ext Γ (bindAt Γp Γp.size [.sc .i64 i, .sc .i64 av]) := hEp.carry _ (by simp [Γp])
    have hi0 : (bindAt Γp Γp.size [.sc .i64 i, .sc .i64 av])[Γp.size]? = some (.sc .i64 i) := by
      simpa using bindAt_get_carry Γp Γp.size [.sc .i64 i, .sc .i64 av] 0
    have hn1 := hE1.fact hn
    have hlt : i < nv := icmp_ult_true hi0 hn1 heval hf
    have hiN : i.toNat < N := by have := UInt64.lt_iff_toNat_lt.mp hlt; omega
    have hfit' : av.toNat + S < 2 ^ 64 := by
      have := Nat.mul_le_mul_right S (show i.toNat + 1 ≤ N by omega)
      rw [Nat.add_mul] at this; omega
    have hEb : Ext Γ (bindAt ((bindAt Γp Γp.size [.sc .i64 i, .sc .i64 av]).push (.sc t f)) nb
        [.sc .i64 i, .sc .i64 av]) :=
      (hE1.trans (Ext.push _ _)).carry _ (by have := hE1.1; simp at hnb ⊢; omega)
    have hib := bindAt_get_carry ((bindAt Γp Γp.size [.sc .i64 i, .sc .i64 av]).push (.sc t f)) nb
      [.sc .i64 i, .sc .i64 av] 0
    have hab := bindAt_get_carry ((bindAt Γp Γp.size [.sc .i64 i, .sc .i64 av]).push (.sc t f)) nb
      [.sc .i64 i, .sc .i64 av] 1
    simp only [Nat.add_zero, List.getElem?_cons_zero, List.getElem?_cons_succ] at hib hab
    simp only [wp_bind]
    refine wp_mono cfg _ _ _ _ _ ?_ _ _ (hbody _ _ (by simp) nb (nb + 1) _ wb i av hEb hw hib hab hiN hfit' hav)
    rintro r Γ2 w2 ⟨hE2, hw2, z, hz, hzb⟩
    have hnb2 : (Γ2.push (.sc .i64 1))[nb]? = some (.sc .i64 i) := (Ext.push Γ2 _).fact (hE2.fact hib)
    have hc1 : (Γ2.push (.sc .i64 1))[Γ2.size]? = some (.sc .i64 1) := by simp
    have hv'e := evalOp_iadd64 (m := w2.mem) hnb2 hc1
    simp only [wp_op, wp_ret, wp_pure, iadd, op]
    refine ⟨fun h => (by simp [Op'.erase, evalOp] at h), fun v hv => ?_⟩
    have hv1 : v = .sc .i64 1 := by simp only [Op'.erase, evalOp] at hv; cases hv; rfl
    subst hv1
    rw [show (Op'.iadd (carriesFrom nb [ClifTy.i64, ClifTy.i64]).head (Array.size Γ2) iconst64._proof_1).erase =
      Op.iadd nb Γ2.size from rfl, hv'e]
    refine ⟨fun h => (by cases h), fun v' hv' nx hnx => ?_⟩
    cases hv'
    have hr : ((Γ2.push (.sc .i64 1)).push (.sc .i64 (i + 1)))[r]? = some (.sc .i64 z) :=
      (Ext.push _ _).fact ((Ext.push _ _).fact hz)
    simp only [Vals.slots, List.mapM_cons, List.mapM_nil, Array.getElem?_push_size, hr, Option.bind_eq_bind,
      Option.bind_some, Option.pure_def, Option.some.injEq] at hnx
    subst hnx
    have hnv := nv.toNat_lt
    have hlt' := UInt64.lt_iff_toNat_lt.mp hlt
    have hi1 : (i + 1).toNat = i.toNat + 1 := by rw [UInt64.toNat_add]; simp; omega
    refine ⟨trivial, hw2, i + 1, z, rfl, by omega, ?_⟩
    rw [hi1, Nat.add_mul]; omega
  · rintro nE Γ0 Γb vs w' hle ⟨rfl, hw, hEb, z, rfl, hz⟩
    simp only [wp_ret]
    refine hk nE (bindAt Γb nE [.sc .i64 z]) w' z
      (hEb.carry _ (by have := hEp.1; simp [Γp] at hle this ⊢; omega)) hw ?_ hz
    simpa using bindAt_get_carry Γb nE [.sc .i64 z] 0
  · rintro Γ0 cs a Γ1 w1 ⟨⟨rfl, -, i, av, rfl, -, -⟩, rfl, rfl⟩ hnone
    have hE1 : Ext Γ (bindAt Γp Γp.size [.sc .i64 i, .sc .i64 av]) := hEp.carry _ (by simp [Γp])
    have hi0 : (bindAt Γp Γp.size [.sc .i64 i, .sc .i64 av])[Γp.size]? = some (.sc .i64 i) := by
      simpa using bindAt_get_carry Γp Γp.size [.sc .i64 i, .sc .i64 av] 0
    exact absurd hnone (icmp_i64_some _ hi0 (hE1.fact hn))

/-- `wp_forLoopAcc_idx`, the value a 64-bit one throughout. -/
theorem wp_forLoopAcc_idx64 {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}
    {n acc0 : Slot .i64} {body : Slot .i64 → Slot .i64 → Prog Slot Lvl (Slot .i64)}
    {R : Slot .i64 → Env → World → Prop} (W : World → Prop) (hW : W w) (N : Nat) {nv a0 : UInt64}
    (hn : Γ[n]? = some (.sc .i64 nv)) (hN : nv.toNat ≤ N) (ha : Γ[acc0]? = some (.sc .i64 a0))
    (hbody : ∀ d' (J' : Post), J'.faultOk = J.faultOk → ∀ i a Γb wb x y, Ext Γ Γb → W wb → Γb[i]? = some (.sc .i64 x) →
      Γb[a]? = some (.sc .i64 y) → x.toNat < N →
      wp cfg d' J' (body i a)
        (fun r Γ2 w2 => Ext Γb Γ2 ∧ W w2 ∧ ∃ z, Γ2[r]? = some (.sc .i64 z)) Γb wb)
    (hk : ∀ r Γ' w' z, Ext Γ Γ' → W w' → Γ'[r]? = some (.sc .i64 z) → R r Γ' w') :
    wp cfg d J (forLoopAcc n acc0 body) R Γ w := by
  simp only [forLoopAcc, wloop2, wloop, wloopL, iconst64, iconst, iaddImm, op, wp_bind, wp_op, wp_ret,
    wp_pure, Op'.erase, evalOp, reduceCtorEq, false_implies, true_and, forall_eq', Option.some.injEq]
  rw [show ofInt ClifTy.i64 0 = V.sc .i64 0 from rfl]
  have hsz : acc0 < Γ.size := by
    rcases Nat.lt_or_ge acc0 Γ.size with h | h
    · exact h
    · rw [Array.getElem?_eq_none h] at ha; cases ha
  have hnsz : n < Γ.size := by
    rcases Nat.lt_or_ge n Γ.size with h | h
    · exact h
    · rw [Array.getElem?_eq_none h] at hn; cases hn
  let Γp := Γ.push (V.sc .i64 0)
  have hEp : Ext Γ Γp := Ext.push Γ _
  rw [wp_loop]
  refine ⟨fun Γ0 cs w => Γ0 = Γp ∧ W w ∧ ∃ i a, cs = [.sc .i64 i, .sc .i64 a] ∧ i.toNat ≤ N,
    fun Γ0 cs a' Γh wh => (Γ0 = Γp ∧ W wh ∧ ∃ i a, cs = [.sc .i64 i, .sc .i64 a] ∧ i.toNat ≤ N) ∧
      Γh = bindAt Γ0 Γ0.size cs ∧
      a' = (contIfULt Γ0.size n, ((Γ0.size + 1 : Slot .i64) ::ᵥ Vals.nil), ()),
    fun Γ0 Γb vs w => Γ0 = Γp ∧ W w ∧ Ext Γ Γb ∧ ∃ z, vs = [.sc .i64 z],
    ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro cs hcs
    have ha' : Γ[acc0] = V.sc .i64 a0 := by
      rw [Array.getElem?_eq_getElem hsz] at ha; exact Option.some.inj ha
    simp only [Vals.slots, List.mapM_cons, List.mapM_nil, Array.getElem?_push_size,
      Array.getElem?_push_lt hsz, ha', Option.bind_eq_bind, Option.bind_some, Option.pure_def,
      Option.some.injEq] at hcs
    subst hcs
    exact ⟨rfl, hW, 0, a0, rfl, by simp⟩
  · rintro n0 Γ0 cs wh rfl hI
    simp only [wp_pure]
    refine ⟨hI, ?_, ?_⟩ <;> first | trivial | rfl
  · rintro Γ0 cs a Γ1 w1 t f vs ⟨⟨rfl, hw, i, av, rfl, hi⟩, rfl, rfl⟩ _ _ hvs
    refine ⟨rfl, hw, ?_, av, ?_⟩
    · exact (hEp.carry (n := Γp.size) _ (by simp [Γp])).trans (Ext.push _ _)
    · have hsz2 : (bindAt Γp Γp.size [V.sc .i64 i, V.sc .i64 av]).size = Γp.size + 2 := by
        simp [bindAt, Array.take]
      have h1 : (Array.push (bindAt Γp Γp.size [V.sc .i64 i, V.sc .i64 av]) (V.sc t f))[Γp.size + 1]? =
          some (.sc .i64 av) := by
        rw [Array.getElem?_push, if_neg (by omega), bindAt_get_carry]; rfl
      simp only [Vals.slots, List.mapM_cons, List.mapM_nil, h1, Option.bind_eq_bind, Option.bind_some,
        Option.pure_def, Option.some.injEq] at hvs
      exact hvs.symm
  · rintro nb Γ0 cs a Γ1 t f wb hnb ⟨⟨rfl, hw, i, av, rfl, hi⟩, rfl, rfl⟩ heval hcont
    have hf : (f != 0) = true := by simpa [contIfULt, contIf] using hcont
    simp only [contIfULt, contIf] at heval
    have hE1 : Ext Γ (bindAt Γp Γp.size [.sc .i64 i, .sc .i64 av]) := hEp.carry _ (by simp [Γp])
    have hi0 : (bindAt Γp Γp.size [.sc .i64 i, .sc .i64 av])[Γp.size]? = some (.sc .i64 i) := by
      simpa using bindAt_get_carry Γp Γp.size [.sc .i64 i, .sc .i64 av] 0
    have hn1 := hE1.fact hn
    have hlt : i < nv := icmp_ult_true hi0 hn1 heval hf
    have hiN : i.toNat < N := by have := UInt64.lt_iff_toNat_lt.mp hlt; omega
    have hEb : Ext Γ (bindAt ((bindAt Γp Γp.size [.sc .i64 i, .sc .i64 av]).push (.sc t f)) nb
        [.sc .i64 i, .sc .i64 av]) :=
      (hE1.trans (Ext.push _ _)).carry _ (by have := hE1.1; simp at hnb ⊢; omega)
    have hib := bindAt_get_carry ((bindAt Γp Γp.size [.sc .i64 i, .sc .i64 av]).push (.sc t f)) nb
      [.sc .i64 i, .sc .i64 av] 0
    have hab := bindAt_get_carry ((bindAt Γp Γp.size [.sc .i64 i, .sc .i64 av]).push (.sc t f)) nb
      [.sc .i64 i, .sc .i64 av] 1
    simp only [Nat.add_zero, List.getElem?_cons_zero, List.getElem?_cons_succ] at hib hab
    simp only [wp_bind]
    refine wp_mono cfg _ _ _ _ _ ?_ _ _ (hbody _ _ (by simp) nb (nb + 1) _ wb i av hEb hw hib hab hiN)
    rintro r Γ2 w2 ⟨hE2, hw2, z, hz⟩
    have hnb2 : (Γ2.push (.sc .i64 1))[nb]? = some (.sc .i64 i) := (Ext.push Γ2 _).fact (hE2.fact hib)
    have hc1 : (Γ2.push (.sc .i64 1))[Γ2.size]? = some (.sc .i64 1) := by simp
    have hv'e := evalOp_iadd64 (m := w2.mem) hnb2 hc1
    simp only [wp_op, wp_ret, wp_pure, iadd, op]
    refine ⟨fun h => (by simp [Op'.erase, evalOp] at h), fun v hv => ?_⟩
    have hv1 : v = .sc .i64 1 := by simp only [Op'.erase, evalOp] at hv; cases hv; rfl
    subst hv1
    rw [show (Op'.iadd (carriesFrom nb [ClifTy.i64, ClifTy.i64]).head (Array.size Γ2) iconst64._proof_1).erase =
      Op.iadd nb Γ2.size from rfl, hv'e]
    refine ⟨fun h => (by cases h), fun v' hv' nx hnx => ?_⟩
    cases hv'
    have hr : ((Γ2.push (.sc .i64 1)).push (.sc .i64 (i + 1)))[r]? = some (.sc .i64 z) :=
      (Ext.push _ _).fact ((Ext.push _ _).fact hz)
    simp only [Vals.slots, List.mapM_cons, List.mapM_nil, Array.getElem?_push_size, hr, Option.bind_eq_bind,
      Option.bind_some, Option.pure_def, Option.some.injEq] at hnx
    subst hnx
    have hnv := nv.toNat_lt
    have hlt' := UInt64.lt_iff_toNat_lt.mp hlt
    have hi1 : (i + 1).toNat = i.toNat + 1 := by rw [UInt64.toNat_add]; simp; omega
    exact ⟨trivial, hw2, i + 1, z, rfl, by omega⟩
  · rintro nE Γ0 Γb vs w' hle ⟨rfl, hw, hEb, z, rfl⟩
    simp only [wp_ret]
    refine hk nE (bindAt Γb nE [.sc .i64 z]) w' z
      (hEb.carry _ (by have := hEp.1; simp [Γp] at hle this ⊢; omega)) hw ?_
    simpa using bindAt_get_carry Γb nE [.sc .i64 z] 0
  · rintro Γ0 cs a Γ1 w1 ⟨⟨rfl, -, i, av, rfl, -⟩, rfl, rfl⟩ hnone
    have hE1 : Ext Γ (bindAt Γp Γp.size [.sc .i64 i, .sc .i64 av]) := hEp.carry _ (by simp [Γp])
    have hi0 : (bindAt Γp Γp.size [.sc .i64 i, .sc .i64 av])[Γp.size]? = some (.sc .i64 i) := by
      simpa using bindAt_get_carry Γp Γp.size [.sc .i64 i, .sc .i64 av] 0
    exact absurd hnone (icmp_i64_some _ hi0 (hE1.fact hn))

/-- **A counted loop threading a value, whose body sees its trip's bound**:
    `wp_forLoopAcc_bnd` with nothing asked of the value. -/
theorem wp_forLoopAcc_idx {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}
    {n acc0 : Slot .i64} {body : Slot .i64 → Slot .i64 → Prog Slot Lvl (Slot .i64)}
    {R : Slot .i64 → Env → World → Prop} (W : World → Prop) (hW : W w) (N : Nat) {nv : UInt64} {a0 : V}
    (hn : Γ[n]? = some (.sc .i64 nv)) (hN : nv.toNat ≤ N) (ha : Γ[acc0]? = some a0)
    (hbody : ∀ d' (J' : Post), J'.faultOk = J.faultOk → ∀ i a Γb wb x, Ext Γ Γb → W wb → Γb[i]? = some (.sc .i64 x) → x.toNat < N →
      wp cfg d' J' (body i a) (fun r Γ2 w2 => Ext Γb Γ2 ∧ W w2 ∧ ∃ z, Γ2[r]? = some z) Γb wb)
    (hk : ∀ r Γ' w', Ext Γ Γ' → W w' → R r Γ' w') :
    wp cfg d J (forLoopAcc n acc0 body) R Γ w := by
  simp only [forLoopAcc, wloop2, wloop, wloopL, iconst64, iconst, iaddImm, op, wp_bind, wp_op, wp_ret,
    wp_pure, Op'.erase, evalOp, reduceCtorEq, false_implies, true_and, forall_eq', Option.some.injEq]
  rw [show ofInt ClifTy.i64 0 = V.sc .i64 0 from rfl]
  have hsz : acc0 < Γ.size := by
    rcases Nat.lt_or_ge acc0 Γ.size with h | h
    · exact h
    · rw [Array.getElem?_eq_none h] at ha; cases ha
  have hnsz : n < Γ.size := by
    rcases Nat.lt_or_ge n Γ.size with h | h
    · exact h
    · rw [Array.getElem?_eq_none h] at hn; cases hn
  let Γp := Γ.push (V.sc .i64 0)
  have hEp : Ext Γ Γp := Ext.push Γ _
  rw [wp_loop]
  refine ⟨fun Γ0 cs w => Γ0 = Γp ∧ W w ∧ ∃ i a, cs = [.sc .i64 i, a] ∧ i.toNat ≤ N,
    fun Γ0 cs a' Γh wh => (Γ0 = Γp ∧ W wh ∧ ∃ i a, cs = [.sc .i64 i, a] ∧ i.toNat ≤ N) ∧
      Γh = bindAt Γ0 Γ0.size cs ∧
      a' = (contIfULt Γ0.size n, ((Γ0.size + 1 : Slot .i64) ::ᵥ Vals.nil), ()),
    fun Γ0 Γb vs w => Γ0 = Γp ∧ W w ∧ Ext Γ Γb ∧ ∃ z, vs = [z],
    ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro cs hcs
    have ha' : Γ[acc0] = a0 := by
      rw [Array.getElem?_eq_getElem hsz] at ha; exact Option.some.inj ha
    simp only [Vals.slots, List.mapM_cons, List.mapM_nil, Array.getElem?_push_size,
      Array.getElem?_push_lt hsz, ha', Option.bind_eq_bind, Option.bind_some, Option.pure_def,
      Option.some.injEq] at hcs
    subst hcs
    exact ⟨rfl, hW, 0, a0, rfl, by simp⟩
  · rintro n0 Γ0 cs wh rfl hI
    simp only [wp_pure]
    refine ⟨hI, ?_, ?_⟩ <;> first | trivial | rfl
  · rintro Γ0 cs a Γ1 w1 t f vs ⟨⟨rfl, hw, i, av, rfl, hi⟩, rfl, rfl⟩ _ _ hvs
    refine ⟨rfl, hw, ?_, av, ?_⟩
    · exact (hEp.carry (n := Γp.size) _ (by simp [Γp])).trans (Ext.push _ _)
    · have hsz2 : (bindAt Γp Γp.size [V.sc .i64 i, av]).size = Γp.size + 2 := by
        simp [bindAt, Array.take]
      have h1 : (Array.push (bindAt Γp Γp.size [V.sc .i64 i, av]) (V.sc t f))[Γp.size + 1]? =
          some av := by
        rw [Array.getElem?_push, if_neg (by omega), bindAt_get_carry]; rfl
      simp only [Vals.slots, List.mapM_cons, List.mapM_nil, h1, Option.bind_eq_bind, Option.bind_some,
        Option.pure_def, Option.some.injEq] at hvs
      exact hvs.symm
  · rintro nb Γ0 cs a Γ1 t f wb hnb ⟨⟨rfl, hw, i, av, rfl, hi⟩, rfl, rfl⟩ heval hcont
    have hf : (f != 0) = true := by simpa [contIfULt, contIf] using hcont
    simp only [contIfULt, contIf] at heval
    have hE1 : Ext Γ (bindAt Γp Γp.size [.sc .i64 i, av]) := hEp.carry _ (by simp [Γp])
    have hi0 : (bindAt Γp Γp.size [.sc .i64 i, av])[Γp.size]? = some (.sc .i64 i) := by
      simpa using bindAt_get_carry Γp Γp.size [.sc .i64 i, av] 0
    have hn1 := hE1.fact hn
    have hlt : i < nv := icmp_ult_true hi0 hn1 heval hf
    have hiN : i.toNat < N := by have := UInt64.lt_iff_toNat_lt.mp hlt; omega
    have hEb : Ext Γ (bindAt ((bindAt Γp Γp.size [.sc .i64 i, av]).push (.sc t f)) nb
        [.sc .i64 i, av]) :=
      (hE1.trans (Ext.push _ _)).carry _ (by have := hE1.1; simp at hnb ⊢; omega)
    have hib := bindAt_get_carry ((bindAt Γp Γp.size [.sc .i64 i, av]).push (.sc t f)) nb
      [.sc .i64 i, av] 0
    have hab := bindAt_get_carry ((bindAt Γp Γp.size [.sc .i64 i, av]).push (.sc t f)) nb
      [.sc .i64 i, av] 1
    simp only [Nat.add_zero, List.getElem?_cons_zero, List.getElem?_cons_succ] at hib hab
    simp only [wp_bind]
    refine wp_mono cfg _ _ _ _ _ ?_ _ _ (hbody _ _ (by simp) nb (nb + 1) _ wb i hEb hw hib hiN)
    rintro r Γ2 w2 ⟨hE2, hw2, z, hz⟩
    have hnb2 : (Γ2.push (.sc .i64 1))[nb]? = some (.sc .i64 i) := (Ext.push Γ2 _).fact (hE2.fact hib)
    have hc1 : (Γ2.push (.sc .i64 1))[Γ2.size]? = some (.sc .i64 1) := by simp
    have hv'e := evalOp_iadd64 (m := w2.mem) hnb2 hc1
    simp only [wp_op, wp_ret, wp_pure, iadd, op]
    refine ⟨fun h => (by simp [Op'.erase, evalOp] at h), fun v hv => ?_⟩
    have hv1 : v = .sc .i64 1 := by simp only [Op'.erase, evalOp] at hv; cases hv; rfl
    subst hv1
    rw [show (Op'.iadd (carriesFrom nb [ClifTy.i64, ClifTy.i64]).head (Array.size Γ2) iconst64._proof_1).erase =
      Op.iadd nb Γ2.size from rfl, hv'e]
    refine ⟨fun h => (by cases h), fun v' hv' nx hnx => ?_⟩
    cases hv'
    have hr : ((Γ2.push (.sc .i64 1)).push (.sc .i64 (i + 1)))[r]? = some z :=
      (Ext.push _ _).fact ((Ext.push _ _).fact hz)
    simp only [Vals.slots, List.mapM_cons, List.mapM_nil, Array.getElem?_push_size, hr, Option.bind_eq_bind,
      Option.bind_some, Option.pure_def, Option.some.injEq] at hnx
    subst hnx
    have hnv := nv.toNat_lt
    have hlt' := UInt64.lt_iff_toNat_lt.mp hlt
    have hi1 : (i + 1).toNat = i.toNat + 1 := by rw [UInt64.toNat_add]; simp; omega
    exact ⟨trivial, hw2, i + 1, z, rfl, by omega⟩
  · rintro nE Γ0 Γb vs w' hle ⟨rfl, hw, hEb, z, rfl⟩
    simp only [wp_ret]
    exact hk nE (bindAt Γb nE [z]) w'
      (hEb.carry _ (by have := hEp.1; simp [Γp] at hle this ⊢; omega)) hw
  · rintro Γ0 cs a Γ1 w1 ⟨⟨rfl, -, i, av, rfl, -⟩, rfl, rfl⟩ hnone
    have hE1 : Ext Γ (bindAt Γp Γp.size [.sc .i64 i, av]) := hEp.carry _ (by simp [Γp])
    have hi0 : (bindAt Γp Γp.size [.sc .i64 i, av])[Γp.size]? = some (.sc .i64 i) := by
      simpa using bindAt_get_carry Γp Γp.size [.sc .i64 i, av] 0
    exact absurd hnone (icmp_i64_some _ hi0 (hE1.fact hn))

/-- **A counted loop whose body sees its trip's bound.** A limit of at most
    `N` trips: the body sees its trip below `N`, and ends in an environment
    keeping the one it started from, so the trip is still there to count. -/
theorem wp_forLoop_bnd {cfg : Cfg} {d : Nat} {J : Post} {Γ : Env} {w : World}
    {n : Slot .i64} {body : Slot .i64 → Prog Slot Lvl Unit}
    {R : Unit → Env → World → Prop} (W : World → Prop) (hW : W w) (N : Nat) {nv : UInt64}
    (hn : Γ[n]? = some (.sc .i64 nv)) (hN : nv.toNat ≤ N)
    (hbody : ∀ d' (J' : Post), J'.faultOk = J.faultOk → ∀ i Γb wb x, Ext Γ Γb → W wb → Γb[i]? = some (.sc .i64 x) → x.toNat < N →
      x.toNat < nv.toNat → wp cfg d' J' (body i) (fun _ Γ2 w2 => Ext Γb Γ2 ∧ W w2) Γb wb)
    (hk : ∀ Γ' w', Ext Γ Γ' → W w' → R () Γ' w') :
    wp cfg d J (forLoop n body) R Γ w := by
  simp only [forLoop, wloop1, wloop, wloopL, iconst64, iconst, iaddImm, op, wp_bind, wp_op, wp_ret,
    wp_pure, Op'.erase, evalOp, reduceCtorEq, false_implies, true_and, forall_eq', Option.some.injEq]
  rw [show ofInt ClifTy.i64 0 = V.sc .i64 0 from rfl]
  let Γp := Γ.push (V.sc .i64 0)
  have hEp : Ext Γ Γp := Ext.push Γ _
  rw [wp_loop]
  refine ⟨fun Γ0 cs w => Γ0 = Γp ∧ W w ∧ ∃ i, cs = [.sc .i64 i] ∧ i.toNat ≤ N,
    fun Γ0 cs a' Γh wh => (Γ0 = Γp ∧ W wh ∧ ∃ i, cs = [.sc .i64 i] ∧ i.toNat ≤ N) ∧
      Γh = bindAt Γ0 Γ0.size cs ∧ a' = (contIfULt Γ0.size n, Vals.nil, ()),
    fun Γ0 Γb vs w => Γ0 = Γp ∧ W w ∧ Ext Γ Γb ∧ vs = [],
    ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro cs hcs
    simp only [Vals.slots, List.mapM_cons, List.mapM_nil, Array.getElem?_push_size,
      Option.bind_eq_bind, Option.bind_some, Option.pure_def, Option.some.injEq] at hcs
    subst hcs
    exact ⟨rfl, hW, 0, rfl, by simp⟩
  · rintro n0 Γ0 cs wh rfl hI
    simp only [wp_pure]
    refine ⟨hI, ?_, ?_⟩ <;> first | trivial | rfl
  · rintro Γ0 cs a Γ1 w1 t f vs ⟨⟨rfl, hw, i, rfl, hi⟩, rfl, rfl⟩ _ _ hvs
    refine ⟨rfl, hw, ?_, ?_⟩
    · exact (hEp.carry (n := Γp.size) _ (by simp [Γp])).trans (Ext.push _ _)
    · simpa [Vals.slots] using hvs.symm
  · rintro nb Γ0 cs a Γ1 t f wb hnb ⟨⟨rfl, hw, i, rfl, hi⟩, rfl, rfl⟩ heval hcont
    have hf : (f != 0) = true := by simpa [contIfULt, contIf] using hcont
    simp only [contIfULt, contIf] at heval
    have hE1 : Ext Γ (bindAt Γp Γp.size [.sc .i64 i]) := hEp.carry _ (by simp [Γp])
    have hi0 : (bindAt Γp Γp.size [.sc .i64 i])[Γp.size]? = some (.sc .i64 i) := by
      simpa using bindAt_get_carry Γp Γp.size [.sc .i64 i] 0
    have hn1 := hE1.fact hn
    have hlt : i < nv := icmp_ult_true hi0 hn1 heval hf
    have hiN : i.toNat < N := by have := UInt64.lt_iff_toNat_lt.mp hlt; omega
    have hEb : Ext Γ (bindAt ((bindAt Γp Γp.size [.sc .i64 i]).push (.sc t f)) nb [.sc .i64 i]) :=
      (hE1.trans (Ext.push _ _)).carry _ (by have := hE1.1; simp at hnb ⊢; omega)
    have hib := bindAt_get_carry ((bindAt Γp Γp.size [.sc .i64 i]).push (.sc t f)) nb [.sc .i64 i] 0
    simp only [Nat.add_zero, List.getElem?_cons_zero] at hib
    simp only [wp_bind]
    refine wp_mono cfg _ _ _ _ _ ?_ _ _ (hbody _ _ (by simp) nb _ wb i hEb hw hib hiN (UInt64.lt_iff_toNat_lt.mp hlt))
    rintro _ Γ2 w2 ⟨hE2, hw2⟩
    have hnb2 : (Γ2.push (.sc .i64 1))[nb]? = some (.sc .i64 i) := (Ext.push Γ2 _).fact (hE2.fact hib)
    have hc1 : (Γ2.push (.sc .i64 1))[Γ2.size]? = some (.sc .i64 1) := by simp
    have hv'e := evalOp_iadd64 (m := w2.mem) hnb2 hc1
    simp only [wp_op, wp_ret, wp_pure, iadd, op]
    refine ⟨fun h => (by simp [Op'.erase, evalOp] at h), fun v hv => ?_⟩
    have hv1 : v = .sc .i64 1 := by simp only [Op'.erase, evalOp] at hv; cases hv; rfl
    subst hv1
    rw [show (Op'.iadd (carriesFrom nb [ClifTy.i64]).head (Array.size Γ2) iconst64._proof_1).erase =
      Op.iadd nb Γ2.size from rfl, hv'e]
    refine ⟨fun h => (by cases h), fun v' hv' nx hnx => ?_⟩
    cases hv'
    simp only [Vals.slots, List.mapM_cons, List.mapM_nil, Array.getElem?_push_size, Option.bind_eq_bind,
      Option.bind_some, Option.pure_def, Option.some.injEq] at hnx
    subst hnx
    have hnv := nv.toNat_lt
    have hlt' := UInt64.lt_iff_toNat_lt.mp hlt
    have hi1 : (i + 1).toNat = i.toNat + 1 := by rw [UInt64.toNat_add]; simp; omega
    exact ⟨trivial, hw2, i + 1, rfl, by omega⟩
  · rintro nE Γ0 Γb vs w' hle ⟨rfl, hw, hEb, rfl⟩
    simp only [wp_ret]
    exact hk _ w' (hEb.carry _ (by have := hEp.1; simp [Γp] at hle this ⊢; omega)) hw
  · rintro Γ0 cs a Γ1 w1 ⟨⟨rfl, -, i, rfl, -⟩, rfl, rfl⟩ hnone
    have hE1 : Ext Γ (bindAt Γp Γp.size [.sc .i64 i]) := hEp.carry _ (by simp [Γp])
    have hi0 : (bindAt Γp Γp.size [.sc .i64 i])[Γp.size]? = some (.sc .i64 i) := by
      simpa using bindAt_get_carry Γp Γp.size [.sc .i64 i] 0
    exact absurd hnone (icmp_i64_some _ hi0 (hE1.fact hn))

/-- Where a loop entered at `Γ` in `W` may go: back to its head in `W`, or out
    from an environment keeping `Γ`, in `W`; farther out as `J` says. -/
abbrev loopTo (J : Post) (Γ : Env) (W : World → Prop) (exitTys tys : List ClifTy) : Post :=
  loopJ J Γ (fun Γ0 _ w => Γ0 = Γ ∧ W w) (fun Γ0 Γb _ w => Γ0 = Γ ∧ Ext Γ Γb ∧ W w) exitTys tys

/-- **A loop, by the typestate.** A head and a body that keep `W` from any
    environment keeping the one the loop starts in keep it for every trip, and
    what follows starts from such an environment in `W`. The carries may be
    anything. A jump back to the head or out of the loop is taken in `W`. -/
theorem wp_loop_vc {tys exitTys : List ClifTy} {γ : Type} {init : Vals Slot tys}
    {head : Lvl exitTys tys → Vals Slot tys → Prog Slot Lvl (Cond Slot × Vals Slot exitTys × γ)}
    {body : Lvl exitTys tys → Vals Slot tys → γ → Prog Slot Lvl (Vals Slot tys)}
    {k : Vals Slot exitTys → Prog Slot Lvl α} (W : World → Prop) (hW : W w)
    (hhead : ∀ n0 Γh wh, Ext Γ Γh → W wh →
      wp cfg (d + 1) (loopTo J Γ W exitTys tys) (head d (carriesFrom n0 tys))
        (fun _ Γ1 w1 => Ext Γh Γ1 ∧ W w1) Γh wh)
    (hbody : ∀ nb a Γb wb, Ext Γ Γb → W wb →
      wp cfg (d + 1) (loopTo J Γ W exitTys tys) (body d (carriesFrom nb tys) a) (fun _ _ w2 => W w2) Γb wb)
    (hk : ∀ nE Γ' w', Ext Γ Γ' → W w' → wp cfg d J (k (carriesFrom nE exitTys)) Q Γ' w') (hJ : J.faultOk = true) :
    wp cfg d J (.loop init head body k) Q Γ w := by
  rw [wp_loop]
  refine ⟨fun Γ0 _ w => Γ0 = Γ ∧ W w, fun Γ0 _ _ Γ1 w1 => Γ0 = Γ ∧ Ext Γ Γ1 ∧ W w1,
    fun Γ0 Γb _ w => Γ0 = Γ ∧ Ext Γ Γb ∧ W w, fun _ _ => ⟨rfl, hW⟩, ?_, ?_, ?_, ?_,
    fun _ _ _ _ _ _ _ => hJ⟩
  · rintro n0 Γ0 cs wh rfl ⟨rfl, hw⟩
    have he := Ext.carry (n := Γ0.size) cs (Ext.refl Γ0) (Nat.le_refl _)
    exact wp_mono cfg _ _ _ _ _ (fun _ _ _ h => ⟨rfl, he.trans h.1, h.2⟩) _ _ (hhead _ _ _ he hw)
  · rintro Γ0 cs a Γ1 w1 t f vs ⟨rfl, he, hw⟩ _ _ _
    exact ⟨rfl, he.trans (Ext.push _ _), hw⟩
  · rintro nb Γ0 cs a Γ1 t f wb rfl ⟨rfl, he, hw⟩ _ _
    have he' := Ext.carry (n := Γ1.size + 1) cs (he.trans (Ext.push Γ1 (.sc t f))) (by have := he.1; simp; omega)
    exact wp_mono cfg _ _ _ _ _ (fun _ _ _ h _ _ => ⟨rfl, h⟩) _ _ (hbody _ _ _ _ he' hw)
  · rintro nE Γ0 Γb vs w' hle ⟨rfl, he, hw⟩
    exact hk _ _ _ (Ext.carry vs he hle) hw

/-- Where a bottom-tested loop entered at `Γ` in `W` may go, from a trip
    entered at `Γ0`: back to its head, or out from an environment keeping `Γ`,
    each from `Γ0` keeping `Γ` and in `W`; farther out as `J` says. -/
abbrev dloopTo (J : Post) (Γ Γ0 : Env) (W : World → Prop) (exitTys tys : List ClifTy) : Post :=
  loopJ J Γ0 (fun Γ0 _ w => Ext Γ Γ0 ∧ W w) (fun Γ0 Γb _ w => Ext Γ Γ0 ∧ Ext Γ Γb ∧ W w) exitTys tys

/-- **A bottom-tested loop, by the typestate.** A body that keeps `W` from any
    environment keeping the one the loop starts in keeps it for every trip, and
    what follows starts from such an environment in `W`. The carries, and the
    guard and back-edge tests, may be anything. -/
theorem wp_dloop_vc {tys : List ClifTy} {tb : ClifTy} {init : Vals Slot tys} {cc : ICmpCond} {cb : Slot tb}
    {guardIdx : Option Nat} {hg : guardIdx.all (fun i => (tys[i]?).getD default == tb) = true}
    {contOnTrue : Bool} {exitIdx : List Nat}
    {body : Lvl (idxTys tys exitIdx) tys → Vals Slot tys → Prog Slot Lvl (Slot tb × Vals Slot tys)}
    {k : Vals Slot (idxTys tys exitIdx) → Prog Slot Lvl α} (W : World → Prop) (hW : W w)
    (hbody : ∀ Γ0 nb Γb wb, Ext Γ Γ0 → Ext Γ Γb → W wb →
      wp cfg (d + 1) (dloopTo J Γ Γ0 W (idxTys tys exitIdx) tys) (body d (carriesFrom nb tys))
        (fun _ Γ2 w2 => Ext Γb Γ2 ∧ W w2) Γb wb)
    (hk : ∀ nE Γ' w', Ext Γ Γ' → W w' → wp cfg d J (k (carriesFrom nE (idxTys tys exitIdx))) Q Γ' w') (hJ : J.faultOk = true) :
    wp cfg d J (.dloop init cc cb guardIdx hg contOnTrue exitIdx body k) Q Γ w := by
  rw [wp_dloop]
  refine ⟨fun Γ0 _ w => Ext Γ Γ0 ∧ W w, fun Γ0 _ _ Γ2 w2 => Ext Γ Γ0 ∧ Ext Γ Γ2 ∧ W w2,
    fun Γ0 Γb _ w => Ext Γ Γ0 ∧ Ext Γ Γb ∧ W w, ⟨fun _ _ _ => hJ, ?_⟩, ?_, ?_, ?_,
    fun _ _ _ _ _ _ _ => hJ⟩
  · intro Γ' w' cs hg' _
    cases guardIdx with
    | none => obtain ⟨rfl, rfl⟩ := hg'; exact ⟨Ext.refl _, hW⟩
    | some _ =>
        obtain ⟨Γ1, v, rfl, ⟨rfl, rfl⟩, _⟩ := hg'
        exact fun _ _ _ => ⟨fun _ => ⟨Ext.push _ _, hW⟩,
          fun _ _ _ => ⟨Ext.push _ _, Ext.push _ _, hW⟩⟩
  · rintro n0 Γ0 cs wb rfl ⟨he0, hw⟩
    have heb := Ext.carry (n := Γ0.size) cs (Ext.refl Γ0) (Nat.le_refl _)
    exact wp_mono cfg _ _ _ _ _ (fun _ _ _ h => ⟨he0, (he0.trans heb).trans h.1, h.2⟩) _ _
      (hbody _ _ _ _ he0 (he0.trans heb) hw)
  · rintro Γ0 cs a Γ2 w2 t f next ⟨he0, he2, hw⟩ _ _
    exact ⟨fun _ => ⟨he0, hw⟩, fun _ _ _ => ⟨he0, he2.trans (Ext.push _ _), hw⟩⟩
  · rintro nE Γ0 Γb outs w' hle ⟨he0, heb, hw⟩
    exact hk _ _ _ (Ext.carry outs heb (Nat.le_trans he0.1 hle)) hw

theorem mapM_length {tys : List ClifTy} {args : Vals Slot tys} {Γ : Env} {vs : List V}
    (h : args.slots.mapM (fun r => Γ[r]?) = some vs) : vs.length = tys.length := by
  induction args generalizing vs with
  | nil => simp [Vals.slots] at h; subst h; rfl
  | cons r rs ih =>
      simp only [Vals.slots, List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff,
        Option.pure_def, Option.some.injEq] at h
      obtain ⟨_, _, _, h2, rfl⟩ := h
      simp [ih h2]

/-- **A jump out of a loop**, by where it goes: the answers it carries are as
    many as the way out takes. -/
theorem wp_br_vc {ex ca} {l : Lvl ex ca} {args : Vals Slot ex}
    (h : ∀ vs, args.slots.mapM (fun r => Γ[r]?) = some vs → vs.length = ex.length →
      J.brk (labelDepth d l) Γ vs w) : wp cfg d J (.br l args) Q Γ w :=
  fun vs hvs => h vs hvs (mapM_length hvs)

/-- **A jump back to a loop's head**, by where it goes. -/
theorem wp_cont_vc {ex ca} {l : Lvl ex ca} {args : Vals Slot ca}
    (h : ∀ vs, args.slots.mapM (fun r => Γ[r]?) = some vs → vs.length = ca.length →
      J.cont (labelDepth d l) vs w) : wp cfg d J (.cont l args) Q Γ w :=
  fun vs hvs => h vs hvs (mapM_length hvs)

/-- What a comparison of `c` coming out `ok` tells: `R`. -/
def Refines (m : Mem) (Γ : Env) (c : Cond Slot) (ok : Bool) (R : Prop) : Prop :=
  ∀ t f, evalOp m Γ (.icmp c.cc c.a c.b) = some (.sc t f) → (f != 0) = ok → R

theorem refines_true {m : Mem} {c : Cond Slot} {ok : Bool} : Refines m Γ c ok True := fun _ _ _ _ => trivial

/-- Two slots hold integers of one type: a comparison of them answers. -/
def IcmpOk (Γ : Env) (a b : R) : Prop :=
  ∃ t x y, Γ[a]? = some (.sc t x) ∧ Γ[b]? = some (.sc t y) ∧ t.isInt = true

theorem IcmpOk.of {Γ : Env} {a b : R} {t : ClifTy} {x y : UInt64} (ha : Γ[a]? = some (.sc t x))
    (hb : Γ[b]? = some (.sc t y)) (ht : t.isInt = true) : IcmpOk Γ a b := ⟨t, x, y, ha, hb, ht⟩

theorem IcmpOk.some {m : Mem} {Γ : Env} {a b : R} (h : IcmpOk Γ a b) (cc : ICmpCond) :
    evalOp m Γ (.icmp cc a b) ≠ none := by
  obtain ⟨t, x, y, ha, hb, ht⟩ := h
  have : (t == t) = true := by cases t <;> rfl
  simp [evalOp, Sem.get, ha, hb, zipIntCmp, this, ht]

/-- A value of type `t`: a scalar of it, or a vector of it with its lanes. -/
def TyV (t : ClifTy) (v : V) : Prop :=
  match t.lanes with
  | none => ∃ x, v = .sc t x
  | some (_, n) => ∃ xs, v = .vec t xs ∧ xs.size = n

/-- Slot `r` of `Γ` holds a value of type `t`. -/
def TyAt (Γ : Env) (r : R) (t : ClifTy) : Prop :=
  match t.lanes with
  | none => ∃ x, Γ[r]? = some (.sc t x)
  | some (_, n) => ∃ xs, Γ[r]? = some (.vec t xs) ∧ xs.size = n

theorem TyV.sc {t : ClifTy} {x : UInt64} (hl : t.lanes = none) : TyV t (.sc t x) := by
  unfold TyV; rw [hl]; exact ⟨x, rfl⟩

theorem TyV.vec {t lane : ClifTy} {n : Nat} {xs : Array UInt64} (hl : t.lanes = some (lane, n))
    (hn : xs.size = n) : TyV t (.vec t xs) := by
  unfold TyV; rw [hl]; exact ⟨xs, rfl, hn⟩

theorem tyAt_of {Γ : Env} {r : R} {t : ClifTy} {v : V} (hv : TyV t v) (h : Γ[r]? = some v) : TyAt Γ r t := by
  unfold TyV at hv; unfold TyAt
  cases hl : t.lanes with
  | none => rw [hl] at hv; obtain ⟨x, rfl⟩ := hv; exact ⟨x, h⟩
  | some p => obtain ⟨_, n⟩ := p; rw [hl] at hv; obtain ⟨xs, rfl, hn⟩ := hv; exact ⟨xs, h, hn⟩

theorem TyAt.get {Γ : Env} {r : R} {t : ClifTy} (h : TyAt Γ r t) : ∃ v, Γ[r]? = some v ∧ TyV t v := by
  unfold TyAt at h; unfold TyV
  cases hl : t.lanes with
  | none => rw [hl] at h; obtain ⟨x, hx⟩ := h; exact ⟨_, hx, x, rfl⟩
  | some p => obtain ⟨_, n⟩ := p; rw [hl] at h; obtain ⟨xs, hx, hn⟩ := h; exact ⟨_, hx, xs, rfl, hn⟩

section VecTy

local instance : LawfulBEq ClifTy where
  eq_of_beq {a b} h := by cases a <;> cases b <;> first | rfl | exact absurd h (by decide)
  rfl {a} := by cases a <;> decide

/-- Operands of their types, scalars or vectors. -/
def ArgsV (Γ : Env) : List R → List ClifTy → Prop
  | [], [] => True
  | r :: rs, t :: ts => TyAt Γ r t ∧ ArgsV Γ rs ts
  | _, _ => False

/-- Two vectors of one type. -/
def vSame : List ClifTy → Option ClifTy
  | [t, u] => if t == u && t.lanes.isSome then some t else none
  | _ => none

/-- Two vectors of one type with float lanes. -/
def vSameF : List ClifTy → Option ClifTy
  | [t, u] => match t.lanes with
    | some (l, _) => if t == u && l.isFloat then some t else none
    | none => none
  | _ => none

/-- Three vectors of one type: `bitselect`'s mask and its two choices. -/
def vSame3 : List ClifTy → Option ClifTy
  | [t, u, v] => if t == u && t == v && t.lanes.isSome then some t else none
  | _ => none

/-- Two `f32x4`s. -/
def vF4 : List ClifTy → Option ClifTy
  | [t, u] => if t == .f32x4 && u == .f32x4 then some .f32x4 else none
  | _ => none

/-- The type an operation answers over vectors, or that makes or reads one. -/
def tyVOp (o : Op) (ts : List ClifTy) : Option ClifTy :=
  match o with
  | .fadd _ _ | .fsub _ _ | .fmul _ _ => vF4 ts
  | .fmax _ _ | .fmin _ _ => vSameF ts
  | .band _ _ | .bandNot _ _ | .bor _ _ | .bxor _ _ | .icmp _ _ _ => vSame ts
  | .bitselect _ _ _ => vSame3 ts
  | .fneg _ => if ts == [.f32x4] then some .f32x4 else none
  | .splat ty _ => match ty.lanes, ts with
    | some (l, _), [t] => if t == l then some ty else none
    | _, _ => none
  | .extractlane _ i => match ts with
    | [t] => match t.lanes with
      | some (l, n) => if i < n then some l else none
      | none => none
    | _ => none
  | .vhighBits _ => match ts with
    | [t] => if t.lanes.isSome then some .i32 else none
    | _ => none
  | _ => none

theorem lane_scalar {t l : ClifTy} {n : Nat} (h : t.lanes = some (l, n)) : l.lanes = none := by
  cases t <;> simp [ClifTy.lanes] at h <;> obtain ⟨rfl, rfl⟩ := h <;> rfl

theorem tyAt_vec {Γ : Env} {r : R} {t l : ClifTy} {n : Nat} (hl : t.lanes = some (l, n)) (h : TyAt Γ r t) :
    ∃ xs, Γ[r]? = some (.vec t xs) ∧ xs.size = n := by
  unfold TyAt at h; rw [hl] at h; exact h

theorem tyAt_sc {Γ : Env} {r : R} {t : ClifTy} (hl : t.lanes = none) (h : TyAt Γ r t) :
    ∃ x, Γ[r]? = some (.sc t x) := by
  unfold TyAt at h; rw [hl] at h; exact h

theorem vSame_eq {ts : List ClifTy} {T : ClifTy} (h : vSame ts = some T) :
    ts = [T, T] ∧ ∃ l n, T.lanes = some (l, n) := by
  rcases ts with _ | ⟨t, _ | ⟨u, _ | ⟨_, _⟩⟩⟩ <;> simp [vSame] at h
  obtain ⟨⟨rfl, h2⟩, rfl⟩ := h
  refine ⟨rfl, ?_⟩
  cases hl : t.lanes with
  | none => rw [hl] at h2; cases h2
  | some p => exact ⟨p.1, p.2, rfl⟩

theorem vSameF_eq {ts : List ClifTy} {T : ClifTy} (h : vSameF ts = some T) :
    ts = [T, T] ∧ ∃ l n, T.lanes = some (l, n) ∧ l.isFloat = true := by
  rcases ts with _ | ⟨t, _ | ⟨u, _ | ⟨_, _⟩⟩⟩ <;> simp only [vSameF, reduceCtorEq] at h
  split at h
  · rename_i l n hl
    simp only [Option.ite_none_right_eq_some, Option.some.injEq, Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨rfl, h2⟩, rfl⟩ := h
    exact ⟨rfl, l, n, hl, h2⟩
  · cases h

theorem vSame3_eq {ts : List ClifTy} {T : ClifTy} (h : vSame3 ts = some T) :
    ts = [T, T, T] ∧ ∃ l n, T.lanes = some (l, n) := by
  rcases ts with _ | ⟨t, _ | ⟨u, _ | ⟨v, _ | ⟨_, _⟩⟩⟩⟩ <;> simp [vSame3] at h
  obtain ⟨⟨⟨rfl, rfl⟩, h2⟩, rfl⟩ := h
  refine ⟨rfl, ?_⟩
  cases hl : t.lanes with
  | none => rw [hl] at h2; cases h2
  | some p => exact ⟨p.1, p.2, rfl⟩

theorem vF4_eq {ts : List ClifTy} {T : ClifTy} (h : vF4 ts = some T) : ts = [.f32x4, .f32x4] ∧ T = .f32x4 := by
  rcases ts with _ | ⟨t, _ | ⟨u, _ | ⟨_, _⟩⟩⟩ <;> simp [vF4] at h
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h; exact ⟨rfl, rfl⟩

theorem args2V {Γ : Env} {a b : R} {t u : ClifTy} (h : ArgsV Γ [a, b] [t, u]) : TyAt Γ a t ∧ TyAt Γ b u :=
  ⟨h.1, h.2.1⟩

theorem zipIntBits_vec {xs ys : Array UInt64} {T l : ClifTy} {n : Nat} (hl : T.lanes = some (l, n))
    (hx : xs.size = n) (hy : ys.size = n) (f : ClifTy → UInt64 → UInt64 → UInt64) :
    ∃ v, zipIntBits (.vec T xs) (.vec T ys) f = some v ∧ TyV T v := by
  refine ⟨_, ?_, TyV.vec hl (xs := xs.zipWith (fun x y => f l x y &&& widthMask l) ys) (by simp [hx, hy])⟩
  simp [zipIntBits, hl, hx, hy]

theorem zipIntCmp_vec {xs ys : Array UInt64} {T l : ClifTy} {n : Nat} (hl : T.lanes = some (l, n))
    (hx : xs.size = n) (hy : ys.size = n) (c : ICmpCond) :
    ∃ v, zipIntCmp c (.vec T xs) (.vec T ys) = some v ∧ TyV T v := by
  refine ⟨_, ?_, TyV.vec hl (xs := xs.zipWith (fun x y => if cmpInt c l x y then widthMask l else 0) ys)
    (by simp [hx, hy])⟩
  simp [zipIntCmp, hl, hx, hy]

theorem zipAnyBits_vec {xs ys : Array UInt64} {T l : ClifTy} {n : Nat} (hl : T.lanes = some (l, n))
    (hx : xs.size = n) (hy : ys.size = n) (f : ClifTy → UInt64 → UInt64 → UInt64) :
    zipAnyBits (.vec T xs) (.vec T ys) f = some (.vec T (xs.zipWith (f l) ys)) := by
  simp [zipAnyBits, zipBitsIf, hl, hx, hy]

theorem zipBits_vec {xs ys : Array UInt64} {T l : ClifTy} {n : Nat} (hl : T.lanes = some (l, n))
    (hf : l.isFloat = true) (hx : xs.size = n) (hy : ys.size = n) (f : ClifTy → UInt64 → UInt64 → UInt64) :
    ∃ v, zipBits (.vec T xs) (.vec T ys) f = some v ∧ TyV T v := by
  refine ⟨_, ?_, TyV.vec hl (xs := xs.zipWith (f l) ys) (by simp [hx, hy])⟩
  simp [zipBits, zipBitsIf, hl, hx, hy, hf]

theorem zipF_vec {xs ys : Array UInt64} (hx : xs.size = 4) (hy : ys.size = 4)
    (f : Float32 → Float32 → Float32) (g : Float → Float → Float) :
    ∃ v, zipF (.vec .f32x4 xs) (.vec .f32x4 ys) f g = some v ∧ TyV .f32x4 v := by
  refine ⟨_, ?_, TyV.vec (t := .f32x4) (lane := .f32) (n := 4) rfl (xs := xs.zipWith (fun x y => ofF32 (f (f32 x) (f32 y))) ys)
    (by simp [hx, hy])⟩
  simp [zipF, hx, hy]

theorem tyVOp_sound {m : Mem} {Γ : Env} {o : Op} {ts : List ClifTy} {T : ClifTy}
    (h : ArgsV Γ o.regs ts) (hT : tyVOp o ts = some T) : ∃ v, evalOp m Γ o = some v ∧ TyV T v := by
  cases o <;> simp only [tyVOp, reduceCtorEq] at hT
  case fadd a b | fsub a b | fmul a b =>
    obtain ⟨rfl, rfl⟩ := vF4_eq hT
    obtain ⟨ha, hb⟩ := args2V h
    obtain ⟨xs, hxa, hxs⟩ := tyAt_vec (t := .f32x4) (l := .f32) (n := 4) rfl ha
    obtain ⟨ys, hya, hys⟩ := tyAt_vec (t := .f32x4) (l := .f32) (n := 4) rfl hb
    simp only [evalOp, Sem.get, hxa, hya, Option.bind_eq_bind, Option.bind_some]
    exact zipF_vec hxs hys _ _
  case fmax a b | fmin a b =>
    obtain ⟨rfl, l, n, hl, hf⟩ := vSameF_eq hT
    obtain ⟨ha, hb⟩ := args2V h
    obtain ⟨xs, hxa, hxs⟩ := tyAt_vec hl ha
    obtain ⟨ys, hya, hys⟩ := tyAt_vec hl hb
    simp only [evalOp, Sem.get, hxa, hya, Option.bind_eq_bind, Option.bind_some]
    exact zipBits_vec hl hf hxs hys _
  case band a b | bandNot a b | bor a b | bxor a b =>
    obtain ⟨rfl, l, n, hl⟩ := vSame_eq hT
    obtain ⟨ha, hb⟩ := args2V h
    obtain ⟨xs, hxa, hxs⟩ := tyAt_vec hl ha
    obtain ⟨ys, hya, hys⟩ := tyAt_vec hl hb
    simp only [evalOp, Sem.get, hxa, hya, Option.bind_eq_bind, Option.bind_some]
    exact zipIntBits_vec hl hxs hys _
  case icmp c a b =>
    obtain ⟨rfl, l, n, hl⟩ := vSame_eq hT
    obtain ⟨ha, hb⟩ := args2V h
    obtain ⟨xs, hxa, hxs⟩ := tyAt_vec hl ha
    obtain ⟨ys, hya, hys⟩ := tyAt_vec hl hb
    simp only [evalOp, Sem.get, hxa, hya, Option.bind_eq_bind, Option.bind_some]
    exact zipIntCmp_vec hl hxs hys _
  case bitselect c a b =>
    obtain ⟨rfl, l, n, hl⟩ := vSame3_eq hT
    obtain ⟨xs, hxc, hxs⟩ := tyAt_vec hl h.1
    obtain ⟨ys, hya, hys⟩ := tyAt_vec hl h.2.1
    obtain ⟨zs, hzb, hzs⟩ := tyAt_vec hl h.2.2.1
    simp only [evalOp, Sem.get, hxc, hya, hzb, Option.bind_eq_bind, Option.bind_some,
      zipAnyBits_vec hl hxs hys, zipAnyBits_vec hl hxs hzs]
    exact ⟨_, zipAnyBits_vec hl (by simp [hxs, hys]) (by simp [hxs, hzs]) _, TyV.vec hl (by simp [hxs, hys, hzs])⟩
  case fneg a =>
    split at hT
    · rename_i hts
      simp only [beq_iff_eq] at hts; subst hts
      cases hT
      obtain ⟨xs, hxa, hxs⟩ := tyAt_vec (t := .f32x4) (l := .f32) (n := 4) rfl h.1
      refine ⟨_, ?_, TyV.vec (t := .f32x4) (lane := .f32) (n := 4) rfl (xs := xs.map (· ^^^ 0x80000000))
        (by simp [hxs])⟩
      simp [evalOp, Sem.get, hxa]
    · cases hT
  case splat ty a =>
    split at hT
    · rename_i l n t hl
      split at hT
      · rename_i htl
        simp only [beq_iff_eq] at htl; subst htl
        cases hT
        obtain ⟨x, hxa⟩ := tyAt_sc (lane_scalar hl) h.1
        refine ⟨_, ?_, TyV.vec hl (xs := Array.replicate n (x &&& widthMask t)) (by simp)⟩
        simp [evalOp, Sem.get, hxa, hl]
      · cases hT
    · cases hT
  case extractlane a i =>
    split at hT
    · rename_i t
      split at hT
      · rename_i l n hl
        split at hT
        · rename_i hi
          cases hT
          obtain ⟨xs, hxa, hxs⟩ := tyAt_vec hl h.1
          refine ⟨.sc T (xs[i]'(hxs ▸ hi)), ?_, TyV.sc (lane_scalar hl)⟩
          simp [evalOp, Sem.get, hxa, hl, hxs, hi]
        · cases hT
      · cases hT
    · cases hT
  case vhighBits a =>
    split at hT
    · rename_i t
      split at hT
      · rename_i hs
        cases hT
        obtain ⟨⟨l, n⟩, hl⟩ := Option.isSome_iff_exists.mp hs
        obtain ⟨xs, hxa, -⟩ := tyAt_vec hl h.1
        cases t <;> simp [ClifTy.lanes] at hl <;> simp [evalOp, Sem.get, hxa, TyV, ClifTy.lanes]
      · cases hT
    · cases hT

/-- Lanes loaded one after another: as many as asked, when every one is there. -/
theorem foldlM_lanes_size {m : Mem} {addr : UInt64} {w : Nat} : ∀ (l : List Nat) (acc r : Array UInt64),
    l.foldlM (fun acc i => (m.load (addr + UInt64.ofNat (i * w)) w).bind fun b => pure (acc.push b)) acc = some r →
    r.size = acc.size + l.length
  | [], acc, r, h => by simp only [List.foldlM_nil, Option.pure_def, Option.some.injEq] at h; subst h; simp
  | i :: l, acc, r, h => by
      simp only [List.foldlM_cons, Option.bind_eq_bind, Option.pure_def] at h
      cases hb : m.load (addr + UInt64.ofNat (i * w)) w with
      | none => rw [hb] at h; simp only [Option.bind_none, reduceCtorEq] at h
      | some b =>
          rw [hb] at h; simp only [Option.bind_some] at h
          have := foldlM_lanes_size l _ _ h
          simp only [Array.size_push, List.length_cons] at this ⊢; omega

theorem foldlM_lanes_some {m : Mem} {addr : UInt64} {w : Nat} (P : Nat → Prop)
    (hf : ∀ i, P i → ∃ b, m.load (addr + UInt64.ofNat (i * w)) w = some b) :
    ∀ (l : List Nat) (acc : Array UInt64), (∀ i ∈ l, P i) →
    ∃ r, l.foldlM (fun acc i => (m.load (addr + UInt64.ofNat (i * w)) w).bind fun b => pure (acc.push b)) acc = some r
  | [], acc, _ => ⟨acc, rfl⟩
  | i :: l, acc, hl => by
      obtain ⟨b, hb⟩ := hf i (hl i (by simp))
      simp only [List.foldlM_cons, Option.bind_eq_bind, Option.pure_def, hb, Option.bind_some]
      exact foldlM_lanes_some P hf l _ fun j hj => hl j (by simp [hj])

/-- A vector load answers a vector of its type, and answers where its bytes fit. -/
theorem evalOp_vload {m : Mem} {Γ : Env} {op : LoadOp} {a : R} {lane : ClifTy} {n : Nat} {t' : ClifTy}
    {x : UInt64} (hk : op.kind = .plain) (hl : op.ty.lanes = some (lane, n)) (ha : Γ[a]? = some (.sc t' x)) :
    (∀ v, evalOp m Γ (.load op a) = some v → TyV op.ty v) ∧
    (Fits m x (n * tyBytes lane) → evalOp m Γ (.load op a) ≠ none) := by
  have hw : 0 < tyBytes lane := by
    cases h : op.ty <;> rw [h] at hl <;> simp [ClifTy.lanes] at hl <;> obtain ⟨rfl, rfl⟩ := hl <;> decide
  simp only [evalOp, Sem.get, ha, hk, hl, Option.bind_eq_bind, Option.bind_some]
  refine ⟨fun v hv => ?_, fun hf => ?_⟩
  · obtain ⟨r, hr, hv⟩ := Option.bind_eq_some_iff.mp hv
    cases hv
    exact TyV.vec hl (by simpa using foldlM_lanes_size _ _ _ hr)
  · obtain ⟨r, hr⟩ := foldlM_lanes_some (m := m) (addr := x) (w := tyBytes lane) (fun i => i < n)
      (fun i hi => (hf.part (i := i * tyBytes lane) (k := tyBytes lane)
        (by have := Nat.mul_le_mul_right (tyBytes lane) (Nat.succ_le_of_lt hi); rw [Nat.succ_mul] at this; omega)
        hw).load)
      (List.range n) #[] (fun i hi => by simpa using hi)
    rw [hr]; simp

/-- **A vector load from a known address**: where the post allows no fault, its
    bytes fit; it answers a vector of its type. -/
theorem wp_op_vload_at {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty} {o : Op' Slot ty} {k : Slot ty → Prog Slot Lvl α} {op : LoadOp} {a : R}
    {lane : ClifTy} {n : Nat} (ho : o.erase = .load op a) (hk : op.kind = .plain)
    (hl : op.ty.lanes = some (lane, n)) {t' : ClifTy} {x : UInt64} (ha : Γ[a]? = some (.sc t' x))
    (hJ : J.faultOk = true ∨ Fits w.mem x (n * tyBytes lane))
    (h : ∀ v, evalOp w.mem Γ o.erase = some v → TyV op.ty v → ∀ Γ', Ext Γ Γ' → Γ'[Γ.size]? = some v →
      wp cfg d J (k Γ.size) Q Γ' w) :
    wp cfg d J (.op o k) Q Γ w :=
  have hv := evalOp_vload (m := w.mem) hk hl ha
  ⟨fun hn => hJ.elim id fun hf => absurd (ho ▸ hn) (hv.2 hf),
   fun v hev => h v hev (hv.1 v (ho ▸ hev)) _ (Ext.push Γ v) (by simp)⟩

/-- **An operation `tyVOp` types**, over operands of the types it asks: it
    answers a value of the type it names. -/
theorem wp_op_tyV {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty} {o : Op' Slot ty} {k : Slot ty → Prog Slot Lvl α} (T : ClifTy)
    (hty : ∃ v, evalOp w.mem Γ o.erase = some v ∧ TyV T v)
    (h : ∀ v, evalOp w.mem Γ o.erase = some v → TyV T v → ∀ Γ', Ext Γ Γ' → Γ'[Γ.size]? = some v →
      wp cfg d J (k Γ.size) Q Γ' w) :
    wp cfg d J (.op o k) Q Γ w :=
  by
  obtain ⟨v', hv', ht⟩ := hty
  refine ⟨fun hn => (by rw [hv'] at hn; cases hn), fun v hev => ?_⟩
  rw [hv'] at hev; cases hev
  exact h v' hv' ht _ (Ext.push Γ _) (by simp)

/-- A scalar answer, typed: of its constructor. -/
theorem tyV_sc_ex {e : Option V} {T : ClifTy} (hl : T.lanes = none) (h : ∃ v, e = some v ∧ TyV T v) :
    ∃ x, e = some (.sc T x) := by
  obtain ⟨v, hv, ht⟩ := h
  unfold TyV at ht; rw [hl] at ht
  obtain ⟨x, rfl⟩ := ht
  exact ⟨x, hv⟩

end VecTy

/-- Values of the types a loop carries, one each. -/
def CarryOk : List ClifTy → List V → Prop
  | [], [] => True
  | t :: ts, v :: vs => TyV t v ∧ CarryOk ts vs
  | _, _ => False

/-- The slots of `vals` hold values of their types. -/
def SlotsOk (Γ : Env) {tys : List ClifTy} (vals : Vals Slot tys) : Prop :=
  ∃ vs, vals.slots.mapM (fun r => Γ[r]?) = some vs ∧ CarryOk tys vs

theorem SlotsOk.nil : SlotsOk Γ (Vals.nil : Vals Slot []) := ⟨[], by simp [Vals.slots], trivial⟩

theorem SlotsOk.cons {t : ClifTy} {ts : List ClifTy} {s : Slot t} {rest : Vals Slot ts} {v : V}
    (h : Γ[s]? = some v) (hv : TyV t v) (hr : SlotsOk Γ rest) : SlotsOk Γ (Vals.cons s rest) := by
  obtain ⟨vs, hvs, hc⟩ := hr
  refine ⟨v :: vs, ?_, hv, hc⟩
  simp [Vals.slots, List.mapM_cons, h, hvs]

theorem mapM_ext {Γ' : Env} (he : Ext Γ Γ') : ∀ {rs : List Nat} {vs : List V},
    rs.mapM (fun r => Γ[r]?) = some vs → rs.mapM (fun r => Γ'[r]?) = some vs
  | [], _, h => h
  | r :: rs, vs, h => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h ⊢
      obtain ⟨v, hv, vs', hvs', rfl⟩ := h
      exact ⟨v, he.fact hv, vs', mapM_ext he hvs', rfl⟩

theorem SlotsOk.ext {Γ' : Env} {tys : List ClifTy} {vals : Vals Slot tys} (he : Ext Γ Γ') (h : SlotsOk Γ vals) :
    SlotsOk Γ' vals := let ⟨vs, hvs, hc⟩ := h; ⟨vs, mapM_ext he hvs, hc⟩

theorem SlotsOk.carry {tys : List ClifTy} {vals : Vals Slot tys} {vs : List V} (h : SlotsOk Γ vals)
    (hvs : vals.slots.mapM (fun r => Γ[r]?) = some vs) : CarryOk tys vs := by
  obtain ⟨vs', h', hc⟩ := h; rw [h'] at hvs; cases hvs; exact hc

/-- Each carry slot from `n` holds a value of its type. -/
def CarryAt (Γ : Env) : Nat → List ClifTy → Prop
  | _, [] => True
  | n, t :: ts => TyAt Γ n t ∧ CarryAt Γ (n + 1) ts

theorem carryAt_of {G : Env} : ∀ {tys : List ClifTy} {cs : List V} {n : Nat},
    CarryOk tys cs → (∀ i, G[n + i]? = cs[i]?) → CarryAt G n tys
  | [], _, _, _, _ => trivial
  | t :: ts, v :: vs, n, ⟨hv, hr⟩, hg => by
      refine ⟨tyAt_of hv (by simpa using hg 0), carryAt_of hr fun i => ?_⟩
      have := hg (i + 1)
      rw [show n + (i + 1) = n + 1 + i by omega] at this
      simpa using this
  | _ :: _, [], _, h, _ => h.elim

theorem carryOk_get : ∀ {tys : List ClifTy} {vs : List V} {i : Nat} {v : V},
    CarryOk tys vs → vs[i]? = some v → TyV ((tys[i]?).getD default) v
  | [], [], _, _, _, h => by simp at h
  | t :: ts, u :: us, 0, v, ⟨hu, _⟩, h => by simp at h; subst h; simpa using hu
  | _ :: ts, _ :: us, i + 1, v, ⟨_, hr⟩, h => by simpa using carryOk_get (i := i) hr (by simpa using h)
  | [], _ :: _, _, _, h, _ => h.elim
  | _ :: _, [], _, _, h, _ => h.elim

theorem carryOk_idx {tys : List ClifTy} {vs : List V} (h : CarryOk tys vs) : ∀ {idx : List Nat} {outs : List V},
    idx.mapM (fun i => vs[i]?) = some outs → CarryOk (idxTys tys idx) outs
  | [], outs, ho => by simp at ho; subst ho; trivial
  | i :: is, outs, ho => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at ho
      obtain ⟨v, hv, os, hos, rfl⟩ := ho
      exact ⟨carryOk_get h hv, carryOk_idx h hos⟩

/-- A loop's guard test, where it has one, compares integers of one type. -/
def GuardOk (Γ : Env) {tys : List ClifTy} {tb : ClifTy} : Option Nat → Vals Slot tys → Slot tb → Prop
  | none, _, _ => True
  | some gi, init, cb => IcmpOk Γ ((init.slots[gi]?).getD 0) cb

/-- `dloopTo`, the carries of their types at each trip. -/
abbrev dloopToT (J : Post) (Γ Γ0 : Env) (W : World → Prop) (exitTys tys : List ClifTy) : Post :=
  loopJ J Γ0 (fun Γ0 cs w => Ext Γ Γ0 ∧ W w ∧ CarryOk tys cs)
    (fun Γ0 Γb outs w => Ext Γ Γ0 ∧ Ext Γ Γb ∧ W w ∧ CarryOk exitTys outs) exitTys tys

/-- **A bottom-tested loop, by the typestate, its carries of their types.** A body
    that keeps `W`, ends in values of the carries' types and in a back-edge test
    over integers of one type, from any environment keeping the one the loop
    starts in whose carries are of their types, keeps `W` every trip; where the
    post allows no fault, the guard test's operands are integers of one type. -/
theorem wp_dloop_vcT {tys : List ClifTy} {tb : ClifTy} {init : Vals Slot tys} {cc : ICmpCond} {cb : Slot tb}
    {guardIdx : Option Nat} {hg : guardIdx.all (fun i => (tys[i]?).getD default == tb) = true}
    {contOnTrue : Bool} {exitIdx : List Nat}
    {body : Lvl (idxTys tys exitIdx) tys → Vals Slot tys → Prog Slot Lvl (Slot tb × Vals Slot tys)}
    {k : Vals Slot (idxTys tys exitIdx) → Prog Slot Lvl α} (W : World → Prop) (hW : W w)
    (hinit : SlotsOk Γ init)
    (hgd : J.faultOk = true ∨ GuardOk Γ guardIdx init cb)
    (hbody : ∀ Γ0 nb Γb wb, Ext Γ Γ0 → Ext Γ Γb → W wb → CarryAt Γb nb tys →
      wp cfg (d + 1) (dloopToT J Γ Γ0 W (idxTys tys exitIdx) tys) (body d (carriesFrom nb tys))
        (fun a Γ2 w2 => Ext Γb Γ2 ∧ W w2 ∧ (J.faultOk = true ∨ IcmpOk Γ2 a.1 cb) ∧ SlotsOk Γ2 a.2) Γb wb)
    (hk : ∀ nE Γ' w', Ext Γ Γ' → W w' → CarryAt Γ' nE (idxTys tys exitIdx) →
      wp cfg d J (k (carriesFrom nE (idxTys tys exitIdx))) Q Γ' w') :
    wp cfg d J (.dloop init cc cb guardIdx hg contOnTrue exitIdx body k) Q Γ w := by
  rw [wp_dloop]
  refine ⟨fun Γ0 cs w => Ext Γ Γ0 ∧ W w ∧ CarryOk tys cs,
    fun Γ0 _ a Γ2 w2 => Ext Γ Γ0 ∧ Ext Γ Γ2 ∧ W w2 ∧ (J.faultOk = true ∨ IcmpOk Γ2 a.1 cb) ∧ SlotsOk Γ2 a.2,
    fun Γ0 Γb outs w => Ext Γ Γ0 ∧ Ext Γ Γb ∧ W w ∧ CarryOk (idxTys tys exitIdx) outs,
    ⟨fun gi hgi hn => hgd.elim id fun h => by subst hgi; exact absurd hn (IcmpOk.some h cc), ?_⟩,
    ?_, ?_, ?_, ?_⟩
  · intro Γ' w' cs hg' hcs
    cases guardIdx with
    | none =>
        obtain ⟨rfl, rfl⟩ := hg'
        exact ⟨Ext.refl _, hW, hinit.carry hcs⟩
    | some _ =>
        obtain ⟨Γ1, v, rfl, ⟨rfl, rfl⟩, _⟩ := hg'
        have hc := (hinit.ext (Ext.push Γ1 v)).carry hcs
        exact fun _ _ _ => ⟨fun _ => ⟨Ext.push _ _, hW, hc⟩,
          fun _ outs ho => ⟨Ext.push _ _, Ext.push _ _, hW, carryOk_idx hc ho⟩⟩
  · rintro n0 Γ0 cs wb rfl ⟨he0, hw, hc⟩
    have heb := Ext.carry (n := Γ0.size) cs (Ext.refl Γ0) (Nat.le_refl _)
    have hca : CarryAt (bindAt Γ0 Γ0.size cs) Γ0.size tys :=
      carryAt_of hc fun i => bindAt_get_carry Γ0 Γ0.size cs i
    exact wp_mono cfg _ _ _ _ _ (fun a _ _ h => ⟨he0, (he0.trans heb).trans h.1, h.2.1, h.2.2.1, h.2.2.2⟩) _ _
      (hbody _ _ _ _ he0 (he0.trans heb) hw hca)
  · rintro Γ0 cs a Γ2 w2 t f next ⟨he0, he2, hw, -, hs⟩ _ hnx
    have hc := (hs.ext (Ext.push Γ2 (.sc t f))).carry hnx
    exact ⟨fun _ => ⟨he0, hw, hc⟩, fun _ outs ho => ⟨he0, he2.trans (Ext.push _ _), hw, carryOk_idx hc ho⟩⟩
  · rintro nE Γ0 Γb outs w' hle ⟨he0, heb, hw, ho⟩
    exact hk _ _ _ (Ext.carry outs heb (Nat.le_trans he0.1 hle)) hw
      (carryAt_of ho fun i => bindAt_get_carry Γb nE outs i)
  · rintro Γ0 cs a Γ2 w2 ⟨-, -, -, hi, -⟩ hn
    exact hi.elim id fun h => absurd hn (h.some cc)

/-- `loopTo`, the carries of their types at each trip. -/
abbrev loopToT (J : Post) (Γ : Env) (W : World → Prop) (exitTys tys : List ClifTy) : Post :=
  loopJ J Γ (fun Γ0 cs w => Γ0 = Γ ∧ W w ∧ CarryOk tys cs)
    (fun Γ0 Γb vs w => Γ0 = Γ ∧ Ext Γ Γb ∧ W w ∧ CarryOk exitTys vs) exitTys tys

/-- **A loop, by the typestate, its carries of their types.** A head that keeps
    `W`, ends in a test over integers of one type and in exit values of their
    types, and a body that keeps `W` and ends in values of the carries' types,
    each from any environment keeping the one the loop starts in whose carries
    are of their types, keep `W` every trip; what follows starts from such an
    environment in `W`, the exit values of their types. -/
theorem wp_loop_vcT {tys exitTys : List ClifTy} {γ : Type} {init : Vals Slot tys}
    {head : Lvl exitTys tys → Vals Slot tys → Prog Slot Lvl (Cond Slot × Vals Slot exitTys × γ)}
    {body : Lvl exitTys tys → Vals Slot tys → γ → Prog Slot Lvl (Vals Slot tys)}
    {k : Vals Slot exitTys → Prog Slot Lvl α} (W : World → Prop) (hW : W w)
    (hinit : SlotsOk Γ init)
    (hhead : ∀ n0 Γh wh, Ext Γ Γh → W wh → CarryAt Γh n0 tys →
      wp cfg (d + 1) (loopToT J Γ W exitTys tys) (head d (carriesFrom n0 tys))
        (fun a Γ1 w1 => Ext Γh Γ1 ∧ W w1 ∧ (J.faultOk = true ∨ IcmpOk Γ1 a.1.a a.1.b) ∧
          SlotsOk Γ1 a.2.1) Γh wh)
    (hbody : ∀ nb a Γb wb, Ext Γ Γb → W wb → CarryAt Γb nb tys →
      wp cfg (d + 1) (loopToT J Γ W exitTys tys) (body d (carriesFrom nb tys) a)
        (fun next Γ2 w2 => W w2 ∧ SlotsOk Γ2 next) Γb wb)
    (hk : ∀ nE Γ' w', Ext Γ Γ' → W w' → CarryAt Γ' nE exitTys →
      wp cfg d J (k (carriesFrom nE exitTys)) Q Γ' w') :
    wp cfg d J (.loop init head body k) Q Γ w := by
  rw [wp_loop]
  refine ⟨fun Γ0 cs w => Γ0 = Γ ∧ W w ∧ CarryOk tys cs,
    fun Γ0 cs a Γ1 w1 => Γ0 = Γ ∧ CarryOk tys cs ∧ Ext Γ Γ1 ∧ W w1 ∧
      (J.faultOk = true ∨ IcmpOk Γ1 a.1.a a.1.b) ∧ SlotsOk Γ1 a.2.1,
    fun Γ0 Γb vs w => Γ0 = Γ ∧ Ext Γ Γb ∧ W w ∧ CarryOk exitTys vs,
    fun _ hcs => ⟨rfl, hW, hinit.carry hcs⟩, ?_, ?_, ?_, ?_, ?_⟩
  · rintro n0 Γ0 cs wh rfl ⟨rfl, hw, hc⟩
    have he := Ext.carry (n := Γ0.size) cs (Ext.refl Γ0) (Nat.le_refl _)
    have hca : CarryAt (bindAt Γ0 Γ0.size cs) Γ0.size tys :=
      carryAt_of hc fun i => bindAt_get_carry Γ0 Γ0.size cs i
    exact wp_mono cfg _ _ _ _ _ (fun _ _ _ h => ⟨rfl, hc, he.trans h.1, h.2⟩) _ _ (hhead _ _ _ he hw hca)
  · rintro Γ0 cs a Γ1 w1 t f vs ⟨rfl, -, he, hw, -, hs⟩ _ _ hvs
    exact ⟨rfl, he.trans (Ext.push _ _), hw, (hs.ext (Ext.push Γ1 (.sc t f))).carry hvs⟩
  · rintro nb Γ0 cs a Γ1 t f wb rfl ⟨rfl, hc, he, hw, -⟩ _ _
    have he' := Ext.carry (n := Γ1.size + 1) cs (he.trans (Ext.push Γ1 (.sc t f))) (by have := he.1; simp; omega)
    have hca : CarryAt (bindAt (Γ1.push (.sc t f)) (Γ1.size + 1) cs) (Γ1.size + 1) tys :=
      carryAt_of hc fun i => bindAt_get_carry _ _ cs i
    exact wp_mono cfg _ _ _ _ _ (fun _ _ _ h _ hnx => ⟨rfl, h.1, h.2.carry hnx⟩) _ _
      (hbody _ _ _ _ he' hw hca)
  · rintro nE Γ0 Γb vs w' hle ⟨rfl, he, hw, hc⟩
    exact hk _ _ _ (Ext.carry vs he hle) hw (carryAt_of hc fun i => bindAt_get_carry Γb nE vs i)
  · rintro Γ0 cs a Γ1 w1 ⟨-, -, -, -, hi, -⟩ hn
    exact hi.elim id fun h => absurd hn (h.some a.1.cc)

theorem TyAt.ext {Γ Γ' : Env} {r : R} {t : ClifTy} (he : Ext Γ Γ') (h : TyAt Γ r t) : TyAt Γ' r t := by
  obtain ⟨v, hv, ht⟩ := h.get; exact tyAt_of ht (he.fact hv)

theorem carryAt_one {Γ : Env} {r : R} {t : ClifTy} (h : TyAt Γ r t) : CarryAt Γ r [t] := ⟨h, trivial⟩

/-- `wp_loop_vcT` where the head hands its body one value it computed: the
    body is told the value's type, as a carry's. -/
theorem wp_loop_vcTA {tys exitTys : List ClifTy} {u : ClifTy} {init : Vals Slot tys}
    {head : Lvl exitTys tys → Vals Slot tys → Prog Slot Lvl (Cond Slot × Vals Slot exitTys × Slot u)}
    {body : Lvl exitTys tys → Vals Slot tys → Slot u → Prog Slot Lvl (Vals Slot tys)}
    {k : Vals Slot exitTys → Prog Slot Lvl α} (W : World → Prop) (hW : W w)
    (hinit : SlotsOk Γ init)
    (hhead : ∀ n0 Γh wh, Ext Γ Γh → W wh → CarryAt Γh n0 tys →
      wp cfg (d + 1) (loopToT J Γ W exitTys tys) (head d (carriesFrom n0 tys))
        (fun a Γ1 w1 => Ext Γh Γ1 ∧ W w1 ∧ (J.faultOk = true ∨ IcmpOk Γ1 a.1.a a.1.b) ∧
          SlotsOk Γ1 a.2.1 ∧ CarryAt Γ1 a.2.2 [u]) Γh wh)
    (hbody : ∀ nb a Γb wb, Ext Γ Γb → W wb → CarryAt Γb nb tys → CarryAt Γb a [u] →
      wp cfg (d + 1) (loopToT J Γ W exitTys tys) (body d (carriesFrom nb tys) a)
        (fun next Γ2 w2 => W w2 ∧ SlotsOk Γ2 next) Γb wb)
    (hk : ∀ nE Γ' w', Ext Γ Γ' → W w' → CarryAt Γ' nE exitTys →
      wp cfg d J (k (carriesFrom nE exitTys)) Q Γ' w') :
    wp cfg d J (.loop init head body k) Q Γ w := by
  rw [wp_loop]
  refine ⟨fun Γ0 cs w => Γ0 = Γ ∧ W w ∧ CarryOk tys cs,
    fun Γ0 cs a Γ1 w1 => Γ0 = Γ ∧ CarryOk tys cs ∧ Ext Γ Γ1 ∧ W w1 ∧
      (J.faultOk = true ∨ IcmpOk Γ1 a.1.a a.1.b) ∧ SlotsOk Γ1 a.2.1 ∧ CarryAt Γ1 a.2.2 [u],
    fun Γ0 Γb vs w => Γ0 = Γ ∧ Ext Γ Γb ∧ W w ∧ CarryOk exitTys vs,
    fun _ hcs => ⟨rfl, hW, hinit.carry hcs⟩, ?_, ?_, ?_, ?_, ?_⟩
  · rintro n0 Γ0 cs wh rfl ⟨rfl, hw, hc⟩
    have he := Ext.carry (n := Γ0.size) cs (Ext.refl Γ0) (Nat.le_refl _)
    have hca : CarryAt (bindAt Γ0 Γ0.size cs) Γ0.size tys :=
      carryAt_of hc fun i => bindAt_get_carry Γ0 Γ0.size cs i
    exact wp_mono cfg _ _ _ _ _ (fun _ _ _ h => ⟨rfl, hc, he.trans h.1, h.2⟩) _ _ (hhead _ _ _ he hw hca)
  · rintro Γ0 cs a Γ1 w1 t f vs ⟨rfl, -, he, hw, -, hs, -⟩ _ _ hvs
    exact ⟨rfl, he.trans (Ext.push _ _), hw, (hs.ext (Ext.push Γ1 (.sc t f))).carry hvs⟩
  · rintro nb Γ0 cs a Γ1 t f wb rfl ⟨rfl, hc, he, hw, -, -, ha⟩ _ _
    have h1 : Ext Γ1 (bindAt (Γ1.push (.sc t f)) (Γ1.size + 1) cs) :=
      Ext.carry (n := Γ1.size + 1) cs (Ext.push Γ1 (.sc t f)) (by simp)
    have he' := he.trans h1
    have hca : CarryAt (bindAt (Γ1.push (.sc t f)) (Γ1.size + 1) cs) (Γ1.size + 1) tys :=
      carryAt_of hc fun i => bindAt_get_carry _ _ cs i
    exact wp_mono cfg _ _ _ _ _ (fun _ _ _ h _ hnx => ⟨rfl, h.1, h.2.carry hnx⟩) _ _
      (hbody _ _ _ _ he' hw hca (carryAt_one (ha.1.ext h1)))
  · rintro nE Γ0 Γb vs w' hle ⟨rfl, he, hw, hc⟩
    exact hk _ _ _ (Ext.carry vs he hle) hw (carryAt_of hc fun i => bindAt_get_carry Γb nE vs i)
  · rintro Γ0 cs a Γ1 w1 ⟨-, -, -, -, hi, -⟩ hn
    exact hi.elim id fun h => absurd hn (h.some a.1.cc)

section LoopN

local instance : LawfulBEq ClifTy where
  eq_of_beq {a b} h := by cases a <;> cases b <;> first | rfl | exact absurd h (by decide)
  rfl {a} := by cases a <;> decide

/-- What a test's outcome tells of its operands' values, its fields apart. -/
def CondOutR (Γ : Env) (cc : ICmpCond) (a b : R) (exitOnTrue : Bool) (Pc Px : Prop) : Prop :=
  ∀ t x y, Γ[a]? = some (.sc t x) → Γ[b]? = some (.sc t y) →
    (cmpInt cc t x y = !exitOnTrue → Pc) ∧ (cmpInt cc t x y = exitOnTrue → Px)

theorem condOutR_of {Γ : Env} {cc : ICmpCond} {a b : R} {e : Bool} {Pc Px : Prop} {t : ClifTy} {x y : UInt64}
    (ha : Γ[a]? = some (.sc t x)) (hb : Γ[b]? = some (.sc t y))
    (h1 : cmpInt cc t x y = !e → Pc) (h2 : cmpInt cc t x y = e → Px) : CondOutR Γ cc a b e Pc Px := by
  intro t' x' y' ha' hb'
  rw [ha] at ha'; rw [hb] at hb'
  cases ha'; cases hb'
  exact ⟨h1, h2⟩

/-- What a loop test's outcome tells of its operands' values: `Pc` where the
    loop goes on, `Px` where it leaves. -/
abbrev CondOut (Γ : Env) (c : Cond Slot) (Pc Px : Prop) : Prop :=
  CondOutR Γ c.cc c.a c.b c.exitOnTrue Pc Px

/-- A comparison that answers a scalar compared two integers of one type, and
    answers whether they compare. -/
theorem icmp_sc_inv {m : Mem} {Γ : Env} {cc : ICmpCond} {a b : R} {t : ClifTy} {f : UInt64}
    (h : evalOp m Γ (.icmp cc a b) = some (.sc t f)) :
    ∃ ty x y, Γ[a]? = some (.sc ty x) ∧ Γ[b]? = some (.sc ty y) ∧ (f != 0) = cmpInt cc ty x y := by
  simp only [evalOp, Sem.get, Option.bind_eq_bind] at h
  cases ha : Γ[a]? with
  | none => rw [ha] at h; simp at h
  | some va =>
  cases hb : Γ[b]? with
  | none => rw [ha, hb] at h; simp at h
  | some vb =>
  rw [ha, hb] at h
  simp only [Option.bind_some] at h
  cases va with
  | vec _ _ => cases vb <;> simp [zipIntCmp] at h <;> split at h <;> simp_all
  | sc ta x =>
  cases vb with
  | vec _ _ => simp [zipIntCmp] at h
  | sc tb y =>
  simp only [zipIntCmp] at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, beq_iff_eq] at hc
    obtain ⟨rfl, -⟩ := hc
    simp only [Option.some.injEq, boolV, V.sc.injEq] at h
    obtain ⟨-, rfl⟩ := h
    exact ⟨ta, x, y, rfl, rfl, by cases cmpInt cc ta x y <;> rfl⟩
  · cases h

/-- The counter carry `j` of `cs` holds a 64-bit value of which `P` holds. -/
def CtrOk (cs : List V) (j : Nat) (P : UInt64 → Prop) : Prop := ∃ x, cs[j]? = some (.sc .i64 x) ∧ P x

theorem mapM_get {Γ : Env} : ∀ {rs : List Nat} {vs : List V}, rs.mapM (fun r => Γ[r]?) = some vs →
    ∀ i : Nat, vs[i]? = (rs[i]?).bind (fun r => Γ[r]?)
  | [], vs, h, i => by simp at h; subst h; simp
  | r :: rs, vs, h, i => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h
      obtain ⟨v, hv, vs', hvs', rfl⟩ := h
      cases i with
      | zero => simp [hv]
      | succ i => simpa using mapM_get hvs' i

/-- Slots `nb..` of `Γb` hold what slots `n0..` of `Γh` hold, one per type. -/
def SameFrom (Γb : Env) (nb : Nat) (Γh : Env) (n0 : Nat) : List ClifTy → Prop
  | [] => True
  | _ :: ts => Γb[nb]? = Γh[n0]? ∧ SameFrom Γb (nb + 1) Γh (n0 + 1) ts

theorem sameFrom_of {Γb Γh : Env} : ∀ {tys : List ClifTy} {nb n0 : Nat},
    (∀ i : Nat, Γb[nb + i]? = Γh[n0 + i]?) → SameFrom Γb nb Γh n0 tys
  | [], _, _, _ => trivial
  | _ :: _, nb, n0, h => ⟨by simpa using h 0, sameFrom_of fun i => by
      have := h (i + 1)
      rwa [show nb + (i + 1) = nb + 1 + i by omega, show n0 + (i + 1) = n0 + 1 + i by omega] at this⟩

/-- Slots `n..` of `Γ'` hold what slots `rs` of `Γ1` hold. -/
def SameAt (Γ' : Env) (n : Nat) (Γ1 : Env) : List Nat → Prop
  | [] => True
  | r :: rs => Γ'[n]? = Γ1[r]? ∧ SameAt Γ' (n + 1) Γ1 rs

theorem sameAt_of {Γ' Γ1 : Env} : ∀ {rs : List Nat} {n : Nat},
    (∀ i : Nat, i < rs.length → Γ'[n + i]? = (rs[i]?).bind (fun r => Γ1[r]?)) → SameAt Γ' n Γ1 rs
  | [], _, _ => trivial
  | r :: rs, n, h => ⟨by simpa using h 0 (by simp), sameAt_of fun i hi => by
      have := h (i + 1) (by simp; omega)
      rwa [show n + (i + 1) = n + 1 + i by omega] at this⟩

/-- Slots an environment holds, it holds after a push. -/
theorem mapM_push {Γ : Env} {v : V} : ∀ {rs : List Nat} {vs : List V},
    rs.mapM (fun r => Γ[r]?) = some vs → rs.mapM (fun r => (Γ.push v)[r]?) = some vs
  | [], _, h => h
  | r :: rs, vs, h => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h ⊢
      obtain ⟨v', hv, vs', hvs', rfl⟩ := h
      exact ⟨v', (Ext.push Γ v).fact hv, vs', mapM_push hvs', rfl⟩

/-- `loopToT`, a counter carry of which `P` holds, and the loop's exit as what
    follows it. -/
abbrev loopToN (J : Hoare.Post) (Γ : Env) (W : World → Prop) {α : Type} {exitTys : List ClifTy}
    (k : Vals Slot exitTys → Prog Slot Lvl α) (Q : α → Env → World → Prop) (cfg : Cfg) (d : Nat)
    (tys : List ClifTy) (j : Nat) (P : UInt64 → Prop) : Hoare.Post :=
  loopJ J Γ (fun Γ0 cs w => Γ0 = Γ ∧ W w ∧ CarryOk tys cs ∧ CtrOk cs j P)
    (fun Γ0 Γb vs w => Γ0 = Γ ∧ ∀ nE, Γ.size ≤ nE → wp cfg d J (k (carriesFrom nE exitTys)) Q (bindAt Γb nE vs) w)
    exitTys tys

/-- **A loop, its head first.** The counter, carry `j`, holds a value of which `P`
    holds at every trip. The head ends in a test of integers of one type and in
    exit values of their types; where the test goes on, the body, from the
    head's environment and given the head's carries, ends in carries of their
    types, its counter of which `P` holds; where it leaves, what follows the
    loop, given the exit values. Both are asked where the head ends, so each
    knows what the head computed and how its test came out. -/
theorem wp_loop_vcN {cfg : Cfg} {d : Nat} {J : Hoare.Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {tys exitTys : List ClifTy} {γ : Type} {init : Vals Slot tys}
    {head : Lvl exitTys tys → Vals Slot tys → Prog Slot Lvl (Cond Slot × Vals Slot exitTys × γ)}
    {body : Lvl exitTys tys → Vals Slot tys → γ → Prog Slot Lvl (Vals Slot tys)}
    {k : Vals Slot exitTys → Prog Slot Lvl α} (W : World → Prop) (hW : W w)
    (j : Nat) (P : UInt64 → Prop) (hjt : j < tys.length)
    (hinit : SlotsOk Γ init) {x0 : UInt64}
    (hx0 : Γ[(init.slots[j]?).getD 0]? = some (.sc .i64 x0)) (hP0 : P x0)
    (hhead : ∀ n0 Γh wh x, Ext Γ Γh → W wh → CarryAt Γh n0 tys → Γh[n0 + j]? = some (.sc .i64 x) → P x →
      wp cfg (d + 1) (loopToN J Γ W k Q cfg d tys j P) (head d (carriesFrom n0 tys))
        (fun a Γ1 w1 => Ext Γh Γ1 ∧ (J.faultOk = true ∨ IcmpOk Γ1 a.1.a a.1.b) ∧ SlotsOk Γ1 a.2.1 ∧
          CondOut Γ1 a.1
            (∀ nb Γb, Ext Γ1 Γb → CarryAt Γb nb tys → SameFrom Γb nb Γh n0 tys →
              wp cfg (d + 1) (loopToN J Γ W k Q cfg d tys j P) (body d (carriesFrom nb tys) a.2.2)
                (fun next Γ2 w2 => W w2 ∧ SlotsOk Γ2 next ∧
                  ∃ y, Γ2[(next.slots[j]?).getD 0]? = some (.sc .i64 y) ∧ P y) Γb w1)
            (∀ nE Γ', Ext Γ Γ' → CarryAt Γ' nE exitTys → SameAt Γ' nE Γ1 a.2.1.slots →
              wp cfg d J (k (carriesFrom nE exitTys)) Q Γ' w1)) Γh wh) :
    wp cfg d J (.loop init head body k) Q Γ w := by
  rw [wp_loop]
  refine ⟨fun Γ0 cs w => Γ0 = Γ ∧ W w ∧ CarryOk tys cs ∧ CtrOk cs j P,
    fun Γ0 cs a Γ1 w1 => Γ0 = Γ ∧ CarryOk tys cs ∧ Ext (bindAt Γ0 Γ0.size cs) Γ1 ∧
      (J.faultOk = true ∨ IcmpOk Γ1 a.1.a a.1.b) ∧ SlotsOk Γ1 a.2.1 ∧
      CondOut Γ1 a.1
        (∀ nb Γb, Ext Γ1 Γb → CarryAt Γb nb tys → SameFrom Γb nb (bindAt Γ0 Γ0.size cs) Γ0.size tys →
          wp cfg (d + 1) (loopToN J Γ W k Q cfg d tys j P) (body d (carriesFrom nb tys) a.2.2)
            (fun next Γ2 w2 => W w2 ∧ SlotsOk Γ2 next ∧
              ∃ y, Γ2[(next.slots[j]?).getD 0]? = some (.sc .i64 y) ∧ P y) Γb w1)
        (∀ nE Γ', Ext Γ Γ' → CarryAt Γ' nE exitTys → SameAt Γ' nE Γ1 a.2.1.slots →
          wp cfg d J (k (carriesFrom nE exitTys)) Q Γ' w1),
    fun Γ0 Γb vs w => Γ0 = Γ ∧ ∀ nE, Γ.size ≤ nE →
      wp cfg d J (k (carriesFrom nE exitTys)) Q (bindAt Γb nE vs) w,
    fun cs hcs => ⟨rfl, hW, hinit.carry hcs, x0, ?_, hP0⟩, ?_, ?_, ?_, ?_, ?_⟩
  · rw [mapM_get hcs j]
    have hj : j < init.slots.length := by rw [Vals.slots_length]; exact hjt
    rw [List.getElem?_eq_getElem hj] at hx0 ⊢
    exact hx0
  · rintro n0 Γ0 cs wh rfl ⟨rfl, hw, hc, x, hxj, hPx⟩
    have he := Ext.carry (n := Γ0.size) cs (Ext.refl Γ0) (Nat.le_refl _)
    have hca : CarryAt (bindAt Γ0 Γ0.size cs) Γ0.size tys :=
      carryAt_of hc fun i => bindAt_get_carry Γ0 Γ0.size cs i
    have hxs : (bindAt Γ0 Γ0.size cs)[Γ0.size + j]? = some (.sc .i64 x) := by
      rw [bindAt_get_carry]; exact hxj
    exact wp_mono cfg _ _ _ _ _ (fun _ _ _ h => ⟨rfl, hc, h⟩) _ _ (hhead _ _ _ x he hw hca hxs hPx)
  · rintro Γ0 cs a Γ1 w1 t f vs ⟨rfl, -, he, -, hs, hco⟩ hev hout hvs
    obtain ⟨ty, xa, ya, ha, hb, hf⟩ := icmp_sc_inv hev
    have hk := (hco ty xa ya ha hb).2 (by
      rw [← hf]; revert hout; cases (f != 0) <;> cases a.1.exitOnTrue <;> simp)
    obtain ⟨vs1, hvs1, hc1⟩ := hs
    have hv : vs = vs1 := by rw [mapM_push hvs1] at hvs; exact (Option.some.inj hvs).symm
    subst hv
    refine ⟨rfl, fun nE hle => hk nE _ ?_ ?_ ?_⟩
    · exact Ext.carry vs (((Ext.carry cs (Ext.refl Γ0) (Nat.le_refl _)).trans he).trans (Ext.push Γ1 _)) hle
    · exact carryAt_of hc1 fun i => bindAt_get_carry _ nE vs i
    · refine sameAt_of fun i _ => ?_
      rw [bindAt_get_carry, mapM_get hvs1 i]
  · rintro nb Γ0 cs a Γ1 t f wb rfl ⟨rfl, hc, he, -, -, hco⟩ hev hout
    obtain ⟨ty, xa, ya, ha, hb, hf⟩ := icmp_sc_inv hev
    have hbody := (hco ty xa ya ha hb).1 (by
      rw [← hf]; revert hout; cases (f != 0) <;> cases a.1.exitOnTrue <;> simp)
    have heb : Ext Γ1 (bindAt (Γ1.push (.sc t f)) (Γ1.size + 1) cs) :=
      Ext.carry cs ((Ext.refl Γ1).trans (Ext.push Γ1 _)) (by simp)
    refine wp_mono cfg _ _ _ _ _ (fun next Γ2 w2 h nx hnx => ⟨rfl, h.1, h.2.1.carry hnx, ?_⟩) _ _
      (hbody _ _ heb (carryAt_of hc fun i => bindAt_get_carry _ _ cs i)
        (sameFrom_of fun i => by rw [bindAt_get_carry, bindAt_get_carry]))
    obtain ⟨y, hy, hPy⟩ := h.2.2
    refine ⟨y, ?_, hPy⟩
    rw [mapM_get hnx j]
    have hj : j < next.slots.length := by rw [Vals.slots_length]; exact hjt
    rw [List.getElem?_eq_getElem hj] at hy ⊢
    exact hy
  · rintro nE Γ0 Γb vs w' hle ⟨rfl, hk⟩
    exact hk nE hle
  · rintro Γ0 cs a Γ1 w1 ⟨-, -, -, hi, -⟩ hn
    exact hi.elim id fun h => absurd hn (h.some a.1.cc)

theorem mapM_getF {f : Nat → Option V} : ∀ {rs : List Nat} {vs : List V}, rs.mapM f = some vs →
    ∀ i : Nat, vs[i]? = (rs[i]?).bind f
  | [], vs, h, i => by simp at h; subst h; simp
  | r :: rs, vs, h, i => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h
      obtain ⟨v, hv, vs', hvs', rfl⟩ := h
      cases i with
      | zero => simp [hv]
      | succ i => simpa using mapM_getF hvs' i

theorem mapM_lenF {f : Nat → Option V} : ∀ {rs : List Nat} {vs : List V}, rs.mapM f = some vs →
    vs.length = rs.length
  | [], vs, h => by simp at h; subst h; rfl
  | r :: rs, vs, h => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h
      obtain ⟨v, hv, vs', hvs', rfl⟩ := h
      simp [mapM_lenF hvs']

/-- The values a loop leaves with, picked by index from carries read at slots
    `rs` of `Γ2`, are what those slots hold. -/
theorem sameAt_idx {Γ' Γ2 : Env} {n : Nat} {rs : List Nat} {vs1 : List V} {idx : List Nat} {outs : List V}
    (h1 : rs.mapM (fun r => Γ2[r]?) = some vs1) (h2 : idx.mapM (fun i => vs1[i]?) = some outs)
    (h3 : ∀ i : Nat, Γ'[n + i]? = outs[i]?) : SameAt Γ' n Γ2 (idx.map (fun i => (rs[i]?).getD 0)) := by
  refine sameAt_of fun i hi => ?_
  rw [List.length_map] at hi
  have hlen := mapM_lenF h2
  have hio : i < outs.length := by omega
  rw [h3, mapM_getF h2 i, List.getElem?_map, List.getElem?_eq_getElem hi]
  simp only [Option.map_some, Option.bind_some]
  have hsome : (vs1[idx[i]]?).isSome := by
    have := mapM_getF h2 i
    rw [List.getElem?_eq_getElem hi, Option.bind_some, List.getElem?_eq_getElem hio] at this
    rw [← this]; rfl
  rw [mapM_get h1 (idx[i])] at hsome ⊢
  cases hr : rs[idx[i]]? with
  | none => rw [hr] at hsome; simp at hsome
  | some r => rfl

theorem condOutR_use {m : Mem} {Γ : Env} {cc : ICmpCond} {a b : R} {e : Bool} {Pc Px : Prop} {t : ClifTy}
    {f : UInt64} (h : CondOutR Γ cc a b e Pc Px) (hev : evalOp m Γ (.icmp cc a b) = some (.sc t f)) :
    (((f != 0) == e) = false → Pc) ∧ (((f != 0) == e) = true → Px) := by
  obtain ⟨ty, xa, ya, ha, hb, hf⟩ := icmp_sc_inv hev
  have := h ty xa ya ha hb
  refine ⟨fun hout => this.1 ?_, fun hout => this.2 ?_⟩ <;>
    (rw [← hf]; revert hout; cases (f != 0) <;> cases e <;> simp)

/-- `dloopToT`, a counter carry of which `P` holds, and the loop's exit as what
    follows it. -/
abbrev dloopToN (J : Hoare.Post) (Γ Γ0 : Env) (W : World → Prop) {α : Type} {exitTys : List ClifTy}
    (k : Vals Slot exitTys → Prog Slot Lvl α) (Q : α → Env → World → Prop) (cfg : Cfg) (d : Nat)
    (tys : List ClifTy) (j : Nat) (P : UInt64 → Prop) : Hoare.Post :=
  loopJ J Γ0 (fun Γ0 cs w => Ext Γ Γ0 ∧ W w ∧ CarryOk tys cs ∧ CtrOk cs j P)
    (fun Γ0 Γb outs w => Ext Γ Γ0 ∧ ∀ nE, Γ0.size ≤ nE →
      wp cfg d J (k (carriesFrom nE exitTys)) Q (bindAt Γb nE outs) w)
    exitTys tys

/-- **A bottom-tested loop, its test asked where each trip ends.** The counter,
    carry `j`, holds a value of which `P` holds at every trip. The guard, where
    there is one, and each trip's back-edge test compare integers of one type;
    where the test goes on, the counter the trip passes on has `P`; where it
    leaves, what follows the loop, given the exit values, is asked there, so it
    knows what the trip computed and how its test came out. -/
theorem wp_dloop_vcN {cfg : Cfg} {d : Nat} {J : Hoare.Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {tys : List ClifTy} {tb : ClifTy} {init : Vals Slot tys} {cc : ICmpCond}
    {cb : Slot tb} {guardIdx : Option Nat} {hg : guardIdx.all (fun i => (tys[i]?).getD default == tb) = true}
    {contOnTrue : Bool} {exitIdx : List Nat}
    {body : Lvl (idxTys tys exitIdx) tys → Vals Slot tys → Prog Slot Lvl (Slot tb × Vals Slot tys)}
    {k : Vals Slot (idxTys tys exitIdx) → Prog Slot Lvl α} (W : World → Prop) (hW : W w)
    (j : Nat) (P : UInt64 → Prop) (hjt : j < tys.length)
    (hinit : SlotsOk Γ init) {x0 : UInt64}
    (hx0 : Γ[(init.slots[j]?).getD 0]? = some (.sc .i64 x0))
    (hP0 : guardIdx = none → P x0)
    (hguard : ∀ gi, guardIdx = some gi →
      (J.faultOk = true ∨ IcmpOk Γ ((init.slots[gi]?).getD 0) cb) ∧
      CondOutR Γ cc ((init.slots[gi]?).getD 0) cb (!contOnTrue) (P x0)
        (∀ nE Γ', Ext Γ Γ' → CarryAt Γ' nE (idxTys tys exitIdx) →
          SameAt Γ' nE Γ (exitIdx.map (fun i => (init.slots[i]?).getD 0)) →
          wp cfg d J (k (carriesFrom nE (idxTys tys exitIdx))) Q Γ' w))
    (hbody : ∀ Γ0 n0 Γb wb x, Ext Γ Γ0 → Ext Γ Γb → W wb → CarryAt Γb n0 tys →
      Γb[n0 + j]? = some (.sc .i64 x) → P x →
      wp cfg (d + 1) (dloopToN J Γ Γ0 W k Q cfg d tys j P) (body d (carriesFrom n0 tys))
        (fun a Γ2 w2 => Ext Γb Γ2 ∧ W w2 ∧ (J.faultOk = true ∨ IcmpOk Γ2 a.1 cb) ∧ SlotsOk Γ2 a.2 ∧
          CondOutR Γ2 cc a.1 cb (!contOnTrue)
            (∃ y, Γ2[(a.2.slots[j]?).getD 0]? = some (.sc .i64 y) ∧ P y)
            (∀ nE Γ', Ext Γ Γ' → CarryAt Γ' nE (idxTys tys exitIdx) →
              SameAt Γ' nE Γ2 (exitIdx.map (fun i => (a.2.slots[i]?).getD 0)) →
              wp cfg d J (k (carriesFrom nE (idxTys tys exitIdx))) Q Γ' w2)) Γb wb) :
    wp cfg d J (.dloop init cc cb guardIdx hg contOnTrue exitIdx body k) Q Γ w := by
  have hj : j < init.slots.length := by rw [Vals.slots_length]; exact hjt
  rw [wp_dloop]
  refine ⟨fun Γ0 cs w => Ext Γ Γ0 ∧ W w ∧ CarryOk tys cs ∧ CtrOk cs j P,
    fun Γ0 cs a Γ2 w2 => Ext Γ Γ0 ∧ CarryOk tys cs ∧ Ext (bindAt Γ0 Γ0.size cs) Γ2 ∧ W w2 ∧
      (J.faultOk = true ∨ IcmpOk Γ2 a.1 cb) ∧ SlotsOk Γ2 a.2 ∧
      CondOutR Γ2 cc a.1 cb (!contOnTrue)
        (∃ y, Γ2[(a.2.slots[j]?).getD 0]? = some (.sc .i64 y) ∧ P y)
        (∀ nE Γ', Ext Γ Γ' → CarryAt Γ' nE (idxTys tys exitIdx) →
          SameAt Γ' nE Γ2 (exitIdx.map (fun i => (a.2.slots[i]?).getD 0)) →
          wp cfg d J (k (carriesFrom nE (idxTys tys exitIdx))) Q Γ' w2),
    fun Γ0 Γb outs w => Ext Γ Γ0 ∧ ∀ nE, Γ0.size ≤ nE →
      wp cfg d J (k (carriesFrom nE (idxTys tys exitIdx))) Q (bindAt Γb nE outs) w,
    ⟨fun gi hgi hn => (hguard gi hgi).1.elim id fun h => absurd hn (h.some cc), ?_⟩, ?_, ?_, ?_, ?_⟩
  · intro Γ' w' cs hpre hcs
    have hc0 := hx0
    rw [List.getElem?_eq_getElem hj] at hc0
    cases guardIdx with
    | none =>
        obtain ⟨rfl, rfl⟩ := hpre
        refine ⟨Ext.refl _, hW, hinit.carry hcs, x0, ?_, hP0 rfl⟩
        rw [mapM_get hcs j, List.getElem?_eq_getElem hj]; exact hc0
    | some gi =>
        obtain ⟨Γ1, v, rfl, ⟨rfl, rfl⟩, hev⟩ := hpre
        obtain ⟨vs1, hvs1, hc1⟩ := hinit
        have hcv : cs = vs1 := by rw [mapM_push hvs1] at hcs; exact (Option.some.inj hcs).symm
        subst hcv
        intro t f hl
        have hv : v = .sc t f := by simpa using hl
        subst hv
        have hco := condOutR_use (hguard gi rfl).2 hev
        refine ⟨fun hout => ⟨Ext.push _ _, hW, hc1, x0, ?_, hco.1 (by revert hout; cases (f != 0) <;> cases contOnTrue <;> simp)⟩,
          fun hout outs houts => ⟨Ext.push _ _, fun nE hle => (hco.2 (by revert hout; cases (f != 0) <;> cases contOnTrue <;> simp)) nE _
            (Ext.carry outs (Ext.push _ _) (by simp at hle; omega)) (carryAt_of (carryOk_idx hc1 houts)
              fun i => bindAt_get_carry _ nE outs i)
            (sameAt_idx hvs1 houts fun i => bindAt_get_carry _ nE outs i)⟩⟩
        rw [mapM_get hvs1 j, List.getElem?_eq_getElem hj]; exact hc0
  · rintro n0 Γ0 cs wb rfl ⟨he0, hw, hc, x, hxj, hPx⟩
    have heb := Ext.carry (n := Γ0.size) cs (Ext.refl Γ0) (Nat.le_refl _)
    have hca : CarryAt (bindAt Γ0 Γ0.size cs) Γ0.size tys :=
      carryAt_of hc fun i => bindAt_get_carry Γ0 Γ0.size cs i
    have hxs : (bindAt Γ0 Γ0.size cs)[Γ0.size + j]? = some (.sc .i64 x) := by
      rw [bindAt_get_carry]; exact hxj
    exact wp_mono cfg _ _ _ _ _ (fun _ _ _ h => ⟨he0, hc, h⟩) _ _
      (hbody _ _ _ _ x he0 (he0.trans heb) hw hca hxs hPx)
  · rintro Γ0 cs a Γ2 w2 t f next ⟨he0, hc, he2, hw, -, hs, hco⟩ hev hnx
    have hco := condOutR_use hco hev
    obtain ⟨vs1, hvs1, hc1⟩ := hs
    have hn : next = vs1 := by rw [mapM_push hvs1] at hnx; exact (Option.some.inj hnx).symm
    subst hn
    refine ⟨fun hout => ⟨he0, hw, hc1, ?_⟩, fun hout outs houts => ⟨he0, fun nE hle => ?_⟩⟩
    · obtain ⟨y, hy, hPy⟩ := hco.1 (by revert hout; cases (f != 0) <;> cases contOnTrue <;> simp)
      refine ⟨y, ?_, hPy⟩
      have hj' : j < a.2.slots.length := by rw [Vals.slots_length]; exact hjt
      rw [mapM_get hvs1 j, List.getElem?_eq_getElem hj']
      rw [List.getElem?_eq_getElem hj'] at hy; exact hy
    · have hle' : Γ.size ≤ nE := Nat.le_trans he0.1 hle
      exact (hco.2 (by revert hout; cases (f != 0) <;> cases contOnTrue <;> simp)) nE _
        (Ext.carry outs (((he0.trans (Ext.carry cs (Ext.refl Γ0) (Nat.le_refl _))).trans he2).trans
          (Ext.push _ _)) hle')
        (carryAt_of (carryOk_idx hc1 houts) fun i => bindAt_get_carry _ nE outs i)
        (sameAt_idx hvs1 houts fun i => bindAt_get_carry _ nE outs i)
  · rintro nE Γ0 Γb outs w' hle ⟨-, hk⟩
    exact hk nE hle
  · rintro Γ0 cs a Γ2 w2 ⟨-, -, -, -, hi, -⟩ hn
    exact hi.elim id fun h => absurd hn (h.some cc)

theorem condOutR_mono {Γ : Env} {cc : ICmpCond} {a b : R} {e : Bool} {Pc Px Pc' Px' : Prop}
    (h : CondOutR Γ cc a b e Pc Px) (hc : Pc → Pc') (hx : Px → Px') : CondOutR Γ cc a b e Pc' Px' :=
  fun t x y ha hb => ⟨fun h1 => hc ((h t x y ha hb).1 h1), fun h2 => hx ((h t x y ha hb).2 h2)⟩

theorem sameAt_get {Γ' Γ1 : Env} : ∀ {rs : List Nat} {n i r : Nat}, SameAt Γ' n Γ1 rs → rs[i]? = some r →
    Γ'[n + i]? = Γ1[r]?
  | [], _, _, _, _, h => by simp at h
  | _ :: _, _, 0, _, h, hr => by simp at hr; subst hr; simpa using h.1
  | _ :: _, n, i + 1, _, h, hr => by
      have := sameAt_get h.2 (by simpa using hr)
      rwa [show n + 1 + i = n + (i + 1) by omega] at this

theorem sameAt_head {Γ' Γ1 : Env} {n r : Nat} {rs : List Nat} (h : SameAt Γ' n Γ1 (r :: rs)) :
    Γ'[n]? = Γ1[r]? := h.1
theorem sameAt_tail {Γ' Γ1 : Env} {n r : Nat} {rs : List Nat} (h : SameAt Γ' n Γ1 (r :: rs)) :
    SameAt Γ' (n + 1) Γ1 rs := h.2
theorem sameFrom_head {Γb Γh : Env} {nb n0 : Nat} {t : ClifTy} {ts : List ClifTy}
    (h : SameFrom Γb nb Γh n0 (t :: ts)) : Γb[nb]? = Γh[n0]? := h.1
theorem sameFrom_tail {Γb Γh : Env} {nb n0 : Nat} {t : ClifTy} {ts : List ClifTy}
    (h : SameFrom Γb nb Γh n0 (t :: ts)) : SameFrom Γb (nb + 1) Γh (n0 + 1) ts := h.2

/-- `loopToT`, a counter carry of which `P` holds, and the loop's exit, by the
    head or by a jump, with its exit counter of which `Px` holds. -/
abbrev loopToX (J : Hoare.Post) (Γ : Env) (W : World → Prop) (exitTys tys : List ClifTy) (j : Nat)
    (P : UInt64 → Prop) (jE : Nat) (Px : UInt64 → Prop) : Hoare.Post :=
  loopJ J Γ (fun Γ0 cs w => Γ0 = Γ ∧ W w ∧ CarryOk tys cs ∧ CtrOk cs j P)
    (fun Γ0 Γb vs w => Γ0 = Γ ∧ Ext Γ Γb ∧ W w ∧ CarryOk exitTys vs ∧ CtrOk vs jE Px) exitTys tys

/-- A jump's exit counter, from the slot the jump carries it from. -/
theorem ctrOk_of_mapM {Γ : Env} {rs : List Nat} {vs : List V} {jE : Nat} {Px : UInt64 → Prop}
    (hvs : rs.mapM (fun r => Γ[r]?) = some vs) (hj : jE < rs.length)
    (h : ∃ y, Γ[(rs[jE]?).getD 0]? = some (.sc .i64 y) ∧ Px y) : CtrOk vs jE Px := by
  obtain ⟨y, hy, hp⟩ := h
  refine ⟨y, ?_, hp⟩
  rw [mapM_get hvs jE, List.getElem?_eq_getElem hj]
  rw [List.getElem?_eq_getElem hj] at hy
  exact hy

/-- **A loop, its head first, its exit told.** As `wp_loop_vcN`, but what
    follows the loop is asked once, from any environment whose exit counter,
    exit `jE`, has `Px`: the head's test that leaves, and every jump out, show
    `Px` of the counter they leave with. -/
theorem wp_loop_vcX {cfg : Cfg} {d : Nat} {J : Hoare.Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {tys exitTys : List ClifTy} {γ : Type} {init : Vals Slot tys}
    {head : Lvl exitTys tys → Vals Slot tys → Prog Slot Lvl (Cond Slot × Vals Slot exitTys × γ)}
    {body : Lvl exitTys tys → Vals Slot tys → γ → Prog Slot Lvl (Vals Slot tys)}
    {k : Vals Slot exitTys → Prog Slot Lvl α} (W : World → Prop) (hW : W w)
    (j : Nat) (P : UInt64 → Prop) (hjt : j < tys.length) (jE : Nat) (Px : UInt64 → Prop)
    (hjE : jE < exitTys.length)
    (hinit : SlotsOk Γ init) {x0 : UInt64}
    (hx0 : Γ[(init.slots[j]?).getD 0]? = some (.sc .i64 x0)) (hP0 : P x0)
    (hhead : ∀ n0 Γh wh x, Ext Γ Γh → W wh → CarryAt Γh n0 tys → Γh[n0 + j]? = some (.sc .i64 x) → P x →
      wp cfg (d + 1) (loopToX J Γ W exitTys tys j P jE Px) (head d (carriesFrom n0 tys))
        (fun a Γ1 w1 => Ext Γh Γ1 ∧ W w1 ∧ (J.faultOk = true ∨ IcmpOk Γ1 a.1.a a.1.b) ∧ SlotsOk Γ1 a.2.1 ∧
          CondOut Γ1 a.1
            (∀ nb Γb, Ext Γ1 Γb → CarryAt Γb nb tys → SameFrom Γb nb Γh n0 tys →
              wp cfg (d + 1) (loopToX J Γ W exitTys tys j P jE Px) (body d (carriesFrom nb tys) a.2.2)
                (fun next Γ2 w2 => W w2 ∧ SlotsOk Γ2 next ∧
                  ∃ y, Γ2[(next.slots[j]?).getD 0]? = some (.sc .i64 y) ∧ P y) Γb w1)
            (∃ y, Γ1[(a.2.1.slots[jE]?).getD 0]? = some (.sc .i64 y) ∧ Px y)) Γh wh)
    (hk : ∀ nE Γ' w' y, Ext Γ Γ' → W w' → CarryAt Γ' nE exitTys → Γ'[nE + jE]? = some (.sc .i64 y) →
      Px y → wp cfg d J (k (carriesFrom nE exitTys)) Q Γ' w') :
    wp cfg d J (.loop init head body k) Q Γ w := by
  rw [wp_loop]
  refine ⟨fun Γ0 cs w => Γ0 = Γ ∧ W w ∧ CarryOk tys cs ∧ CtrOk cs j P,
    fun Γ0 cs a Γ1 w1 => Γ0 = Γ ∧ CarryOk tys cs ∧ Ext (bindAt Γ0 Γ0.size cs) Γ1 ∧ W w1 ∧
      (J.faultOk = true ∨ IcmpOk Γ1 a.1.a a.1.b) ∧ SlotsOk Γ1 a.2.1 ∧
      CondOut Γ1 a.1
        (∀ nb Γb, Ext Γ1 Γb → CarryAt Γb nb tys → SameFrom Γb nb (bindAt Γ0 Γ0.size cs) Γ0.size tys →
          wp cfg (d + 1) (loopToX J Γ W exitTys tys j P jE Px) (body d (carriesFrom nb tys) a.2.2)
            (fun next Γ2 w2 => W w2 ∧ SlotsOk Γ2 next ∧
              ∃ y, Γ2[(next.slots[j]?).getD 0]? = some (.sc .i64 y) ∧ P y) Γb w1)
        (∃ y, Γ1[(a.2.1.slots[jE]?).getD 0]? = some (.sc .i64 y) ∧ Px y),
    fun Γ0 Γb vs w => Γ0 = Γ ∧ Ext Γ Γb ∧ W w ∧ CarryOk exitTys vs ∧ CtrOk vs jE Px,
    fun cs hcs => ⟨rfl, hW, hinit.carry hcs, x0, ?_, hP0⟩, ?_, ?_, ?_, ?_, ?_⟩
  · rw [mapM_get hcs j]
    have hj : j < init.slots.length := by rw [Vals.slots_length]; exact hjt
    rw [List.getElem?_eq_getElem hj] at hx0 ⊢
    exact hx0
  · rintro n0 Γ0 cs wh rfl ⟨rfl, hw, hc, x, hxj, hPx⟩
    have he := Ext.carry (n := Γ0.size) cs (Ext.refl Γ0) (Nat.le_refl _)
    have hca : CarryAt (bindAt Γ0 Γ0.size cs) Γ0.size tys :=
      carryAt_of hc fun i => bindAt_get_carry Γ0 Γ0.size cs i
    have hxs : (bindAt Γ0 Γ0.size cs)[Γ0.size + j]? = some (.sc .i64 x) := by
      rw [bindAt_get_carry]; exact hxj
    exact wp_mono cfg _ _ _ _ _ (fun _ _ _ h => ⟨rfl, hc, h⟩) _ _ (hhead _ _ _ x he hw hca hxs hPx)
  · rintro Γ0 cs a Γ1 w1 t f vs ⟨rfl, -, he, hw, -, hs, hco⟩ hev hout hvs
    obtain ⟨ty, xa, ya, ha, hb, hf⟩ := icmp_sc_inv hev
    have hx := (hco ty xa ya ha hb).2 (by
      rw [← hf]; revert hout; cases (f != 0) <;> cases a.1.exitOnTrue <;> simp)
    obtain ⟨vs1, hvs1, hc1⟩ := hs
    have hv : vs = vs1 := by rw [mapM_push hvs1] at hvs; exact (Option.some.inj hvs).symm
    subst hv
    have hE : Ext Γ0 (Γ1.push (.sc t f)) :=
      ((Ext.carry cs (Ext.refl Γ0) (Nat.le_refl _)).trans he).trans (Ext.push Γ1 _)
    exact ⟨rfl, hE, hw, hc1, ctrOk_of_mapM hvs1 (by rw [Vals.slots_length]; exact hjE) hx⟩
  · rintro nb Γ0 cs a Γ1 t f wb rfl ⟨rfl, hc, he, -, -, -, hco⟩ hev hout
    obtain ⟨ty, xa, ya, ha, hb, hf⟩ := icmp_sc_inv hev
    have hbody := (hco ty xa ya ha hb).1 (by
      rw [← hf]; revert hout; cases (f != 0) <;> cases a.1.exitOnTrue <;> simp)
    have heb : Ext Γ1 (bindAt (Γ1.push (.sc t f)) (Γ1.size + 1) cs) :=
      Ext.carry cs ((Ext.refl Γ1).trans (Ext.push Γ1 _)) (by simp)
    refine wp_mono cfg _ _ _ _ _ (fun next Γ2 w2 h nx hnx => ⟨rfl, h.1, h.2.1.carry hnx, ?_⟩) _ _
      (hbody _ _ heb (carryAt_of hc fun i => bindAt_get_carry _ _ cs i)
        (sameFrom_of fun i => by rw [bindAt_get_carry, bindAt_get_carry]))
    obtain ⟨y, hy, hPy⟩ := h.2.2
    refine ⟨y, ?_, hPy⟩
    rw [mapM_get hnx j]
    have hj : j < next.slots.length := by rw [Vals.slots_length]; exact hjt
    rw [List.getElem?_eq_getElem hj] at hy ⊢
    exact hy
  · rintro nE Γ0 Γb vs w' hle ⟨rfl, he, hw, hc, y, hy, hpy⟩
    exact hk nE _ w' y (Ext.carry vs he hle) hw (carryAt_of hc fun i => bindAt_get_carry Γb nE vs i)
      (by rw [bindAt_get_carry]; exact hy) hpy
  · rintro Γ0 cs a Γ1 w1 ⟨-, -, -, -, hi, -⟩ hn
    exact hi.elim id fun h => absurd hn (h.some a.1.cc)

/-- **A bottom-tested loop, its counter's exit told.** As `wp_dloop_vcN`, but
    what follows the loop is asked once, from any environment whose exit
    counter, carry `jE` of the exits, has `Px`: the guard and every trip's test
    that leaves show `Px` of the counter they leave with. -/
theorem wp_dloop_vcX {cfg : Cfg} {d : Nat} {J : Hoare.Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {tys : List ClifTy} {tb : ClifTy} {init : Vals Slot tys} {cc : ICmpCond}
    {cb : Slot tb} {guardIdx : Option Nat} {hg : guardIdx.all (fun i => (tys[i]?).getD default == tb) = true}
    {contOnTrue : Bool} {exitIdx : List Nat}
    {body : Lvl (idxTys tys exitIdx) tys → Vals Slot tys → Prog Slot Lvl (Slot tb × Vals Slot tys)}
    {k : Vals Slot (idxTys tys exitIdx) → Prog Slot Lvl α} (W : World → Prop) (hW : W w)
    (j : Nat) (P Px : UInt64 → Prop) (hjt : j < tys.length) (jE : Nat) (hjE : exitIdx[jE]? = some j)
    (hinit : SlotsOk Γ init) {x0 : UInt64}
    (hx0 : Γ[(init.slots[j]?).getD 0]? = some (.sc .i64 x0))
    (hP0 : guardIdx = none → P x0)
    (hguard : ∀ gi, guardIdx = some gi →
      (J.faultOk = true ∨ IcmpOk Γ ((init.slots[gi]?).getD 0) cb) ∧
      CondOutR Γ cc ((init.slots[gi]?).getD 0) cb (!contOnTrue) (P x0) (Px x0))
    (hbody : ∀ Γ0 n0 Γb wb x, Ext Γ Γ0 → Ext Γ Γb → W wb → CarryAt Γb n0 tys →
      Γb[n0 + j]? = some (.sc .i64 x) → P x →
      wp cfg (d + 1) (dloopToN J Γ Γ0 W k Q cfg d tys j P) (body d (carriesFrom n0 tys))
        (fun a Γ2 w2 => Ext Γb Γ2 ∧ W w2 ∧ (J.faultOk = true ∨ IcmpOk Γ2 a.1 cb) ∧ SlotsOk Γ2 a.2 ∧
          CondOutR Γ2 cc a.1 cb (!contOnTrue)
            (∃ y, Γ2[(a.2.slots[j]?).getD 0]? = some (.sc .i64 y) ∧ P y)
            (∃ y, Γ2[(a.2.slots[j]?).getD 0]? = some (.sc .i64 y) ∧ Px y)) Γb wb)
    (hk : ∀ nE Γ' w' y, Ext Γ Γ' → W w' → CarryAt Γ' nE (idxTys tys exitIdx) →
      Γ'[nE + jE]? = some (.sc .i64 y) → Px y →
      wp cfg d J (k (carriesFrom nE (idxTys tys exitIdx))) Q Γ' w') :
    wp cfg d J (.dloop init cc cb guardIdx hg contOnTrue exitIdx body k) Q Γ w :=
  wp_dloop_vcN W hW j P hjt hinit hx0 hP0
    (fun gi hgi => ⟨(hguard gi hgi).1, condOutR_mono (hguard gi hgi).2 id fun hp nE Γ' he hc hs =>
      hk nE Γ' w x0 he hW hc (by
        rw [sameAt_get hs (r := (init.slots[j]?).getD 0) (by rw [List.getElem?_map, hjE]; rfl)]
        exact hx0) hp⟩)
    (fun Γ0 n0 Γb wb x h0 hb hw hc hx hp => wp_mono cfg _ _ _ _ _ (fun a Γ2 w2 h =>
      ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, condOutR_mono h.2.2.2.2 id fun ⟨y, hy, hpy⟩ nE Γ' he hc hs =>
        hk nE Γ' w2 y he h.2.1 hc (by
          rw [sameAt_get hs (r := (a.2.slots[j]?).getD 0) (by rw [List.getElem?_map, hjE]; rfl)]
          exact hy) hpy⟩) _ _ (hbody Γ0 n0 Γb wb x h0 hb hw hc hx hp))

/-- **What follows a loop left by a jump**, from the environment the jump
    leaves: one keeping the loop's, its exit slots holding what the jump
    carried from the slots `rs` of the trip's. -/
theorem wp_bindAt_vc {cfg : Cfg} {d : Nat} {J : Hoare.Post} {α : Type} {Q : α → Env → World → Prop}
    {p : Prog Slot Lvl α} {Γ Γb : Env} {w : World} {nE : Nat} {rs : List Nat} {vs : List V}
    (he : Ext Γ Γb) (hle : Γ.size ≤ nE) (hvs : rs.mapM (fun r => Γb[r]?) = some vs)
    (h : ∀ Γ', Ext Γ Γ' → SameAt Γ' nE Γb rs → wp cfg d J p Q Γ' w) :
    wp cfg d J p Q (bindAt Γb nE vs) w :=
  h _ (Ext.carry vs he hle) (sameAt_of fun i _ => by rw [bindAt_get_carry, mapM_get hvs i])

/-- What the generator's first pass puts for a loop counter's invariant, to
    see what a trip passes on: a goal no hypothesis closes. -/
@[irreducible] def ProbeP (x : UInt64) : Prop := x = x

end LoopN

/-- **A branch**, by the condition generator: each arm from its own facts,
    `Rt` where the comparison held and `Re` where it did not, and ending in
    `W`; what follows the join from an environment keeping this one, in `W`. -/
theorem wp_ite_vc {jTys : List ClifTy} {c : Cond Slot} {thn els : Prog Slot Lvl (Vals Slot jTys)}
    {k : Vals Slot jTys → Prog Slot Lvl α} (W : World → Prop) (Rt Re : Prop)
    (hRt : Refines w.mem Γ c true Rt) (hRe : Refines w.mem Γ c false Re)
    (hT : ∀ Γ', Ext Γ Γ' → Rt → wp cfg d J thn (fun _ Γe we => Ext Γ Γe ∧ W we) Γ' w)
    (hE : ∀ Γ', Ext Γ Γ' → Re → wp cfg d J els (fun _ Γe we => Ext Γ Γe ∧ W we) Γ' w)
    (hK : ∀ nJ Γ' w', Ext Γ Γ' → W w' → wp cfg d J (k (carriesFrom nJ jTys)) Q Γ' w')
    (hJ : J.faultOk = true ∨ IcmpOk Γ c.a c.b) :
    wp cfg d J (.ite c thn els k) Q Γ w := by
  rw [wp_ite]
  exact ⟨fun _ Γe we => Ext Γ Γe ∧ W we, fun _ Γe we => Ext Γ Γe ∧ W we,
    ⟨fun t f h hf => hT _ (Ext.push Γ _) (hRt t f h hf),
     fun ne t f hne h hf => hE _ (Ext.bindAt_push Γ _ hne) (hRe t f h hf),
     fun hn => hJ.elim id fun h => absurd hn (h.some _)⟩,
    fun nJ Γ' _ vs w' hle _ hq => by
      have h := hq.elim id id
      exact hK nJ _ w' (Ext.carry vs h.1 (Nat.le_trans h.1.1 hle)) h.2⟩

/-- **A comparison of two values**, as each arm finds it. -/
theorem refine_cmp {m : Mem} {Γ : Env} {c : Cond Slot} {t t' : ClifTy} {x z : UInt64}
    (ha : Γ[c.a]? = some (.sc t x)) (hb : Γ[c.b]? = some (.sc t' z)) (ok : Bool) :
    Refines m Γ c ok (cmpInt c.cc t x z = ok) := by
  intro tt f h hf
  simp only [evalOp, Sem.get, ha, hb, Option.bind_eq_bind, Option.bind_some, zipIntCmp] at h
  split at h
  · simp only [Option.some.injEq, boolV, V.sc.injEq] at h
    obtain ⟨-, rfl⟩ := h
    cases hc : cmpInt c.cc t x z <;> simp only [hc, if_true, if_false, Bool.false_eq_true] at hf ⊢ <;>
      first | exact hf | (subst hf; rfl) | (rw [← hf]; rfl)
  · cases h

/-- **A handle compared with null and found not to be it is held.** -/
theorem refine_ne {m : Mem} {c : Cond Slot} {ty ty' : ClifTy} {x z : UInt64} {S : TState}
    {k : Contracts.Held} (hc : c.cc = .ne) (ha : Γ[c.a]? = some (.sc ty x)) (hb : Γ[c.b]? = some (.sc ty' z))
    (hz : z = 0) (hS : S.holds w) (hm : Fact.opened k x ∈ S) :
    Refines m Γ c true (TState.holds (.held k x :: S) w) := by
  intro t f h hf
  rw [hc] at h
  subst hz
  refine Contracts.held_of_opened hS hm fun hx => ?_
  subst hx
  simp only [evalOp, Sem.get, ha, hb, Option.bind_eq_bind, Option.bind_some, zipIntCmp] at h
  split at h
  · simp only [Option.some.injEq, boolV, V.sc.injEq, cmpInt] at h
    obtain ⟨-, rfl⟩ := h
    simp at hf
  · cases h

/-- The same where the program asked whether the handle is null, and it was not. -/
theorem refine_eq {m : Mem} {c : Cond Slot} {ty ty' : ClifTy} {x z : UInt64} {S : TState}
    {k : Contracts.Held} (hc : c.cc = .eq) (ha : Γ[c.a]? = some (.sc ty x)) (hb : Γ[c.b]? = some (.sc ty' z))
    (hz : z = 0) (hS : S.holds w) (hm : Fact.opened k x ∈ S) :
    Refines m Γ c false (TState.holds (.held k x :: S) w) := by
  intro t f h hf
  rw [hc] at h
  subst hz
  refine Contracts.held_of_opened hS hm fun hx => ?_
  subst hx
  simp only [evalOp, Sem.get, ha, hb, Option.bind_eq_bind, Option.bind_some, zipIntCmp] at h
  split at h
  · simp only [Option.some.injEq, boolV, V.sc.injEq, cmpInt] at h
    obtain ⟨-, rfl⟩ := h
    simp at hf
  · cases h

end ext

/-- A length the caller handed over is at most a region's span. -/
theorem roomArg_toNat_le {S : Contracts.TState} {w : World} {r : Region} {x : UInt64} (hS : S.holds w)
    (hm : Contracts.Fact.roomArg r x ∈ S) : x.toNat ≤ 2 ^ 36 := (hS _ hm).2

/-- A 64-bit value below `2^63` is itself, signed. -/
theorem signed_i64_of_lt {x : UInt64} (h : x.toNat < 2 ^ 63) : signed .i64 x = x.toNat := by
  have hm : x &&& widthMask .i64 = x := by
    apply UInt64.toNat_inj.mp
    rw [UInt64.toNat_and]
    show x.toNat &&& (2 ^ 64 - 1) = x.toNat
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt (UInt64.toNat_lt x)]
  simp only [signed, ClifTy.width, hm]
  rw [if_pos (by decide), if_neg]
  intro hge
  have : (0x8000000000000000 : UInt64).toNat ≤ x.toNat := UInt64.le_iff_toNat_le.mp hge
  simp at this; omega

/-- Bytes at an address past a region's base, within a length the program was
    handed for it. -/
theorem fits_of_roomArg_sub {S : Contracts.TState} {w : World} {a X : UInt64} {n : Nat} {r : Region}
    (hS : S.holds w) (hm : Contracts.Fact.roomArg r X ∈ S) (hz : S.contains (.part .frozen false) = true)
    (hp : r ≠ .pinned) (hle : (regionBase r).toNat ≤ a.toNat)
    (hb : a.toNat - (regionBase r).toNat + n ≤ X.toNat) (hn : 0 < n) : Fits w.mem a n := by
  have hx := hS _ hm
  have hk : a.toNat - (regionBase r).toNat < regionSpan.toNat := by
    simp only [regionSpan, UInt64.reduceToNat]; have := hx.2; omega
  have hd := Static.decodeAddr_base r _ hk
  have hsub : (a - regionBase r).toNat = a.toNat - (regionBase r).toNat :=
    UInt64.toNat_sub_of_le _ _ (UInt64.le_iff_toNat_le.mpr hle)
  rw [← hsub, UInt64.ofNat_toNat, UInt64.add_comm, UInt64.sub_add_cancel] at hd
  rw [hsub] at hd
  exact ⟨r, _, hd, hp, by have := hx.1; omega, Contracts.holds_of_mem hS hz, by have := hx.2; omega⟩

/-- Bytes at an address past a region's base, within a room the typestate
    names. -/
theorem fits_of_room_sub {S : Contracts.TState} {w : World} {a : UInt64} {n N : Nat} {r : Region}
    (hS : S.holds w) (hm : Contracts.Fact.room r N ∈ S) (hz : S.contains (.part .frozen false) = true)
    (hp : r ≠ .pinned) (hle : (regionBase r).toNat ≤ a.toNat)
    (hb : a.toNat - (regionBase r).toNat + n ≤ N) (hN : N ≤ 2 ^ 36) (hn : 0 < n) : Fits w.mem a n := by
  have hx : N ≤ w.mem.sizes r := hS _ hm
  have hk : a.toNat - (regionBase r).toNat < regionSpan.toNat := by
    simp only [regionSpan, UInt64.reduceToNat]; omega
  have hd := Static.decodeAddr_base r _ hk
  have hsub : (a - regionBase r).toNat = a.toNat - (regionBase r).toNat :=
    UInt64.toNat_sub_of_le _ _ (UInt64.le_iff_toNat_le.mpr hle)
  rw [← hsub, UInt64.ofNat_toNat, UInt64.add_comm, UInt64.sub_add_cancel] at hd
  rw [hsub] at hd
  exact ⟨r, _, hd, hp, by omega, Contracts.holds_of_mem hS hz, by omega⟩

/-- A mask clearing the low `k` bits rounds down to a multiple of `2^k`. -/
theorem toNat_and_mask {x : UInt64} {k : Nat} (hk : k < 64) :
    (x &&& UInt64.ofNat (2 ^ 64 - 2 ^ k)).toNat = x.toNat / 2 ^ k * 2 ^ k := by
  rw [UInt64.toNat_and, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by have := Nat.one_le_two_pow (n := k); omega)]
  apply Nat.eq_of_testBit_eq
  intro i
  rw [Nat.testBit_and, ← Nat.shiftLeft_eq, ← Nat.shiftRight_eq_div_pow, Nat.testBit_shiftLeft,
    Nat.testBit_shiftRight]
  have hx := UInt64.toNat_lt x
  by_cases hi : k ≤ i
  · have hsub : (2 ^ 64 - 2 ^ k).testBit i = decide (i < 64) := by
      rw [show 2 ^ 64 - 2 ^ k = 2 ^ k * (2 ^ (64 - k) - 1) by
        rw [Nat.mul_sub, Nat.mul_one, ← Nat.pow_add, Nat.add_sub_cancel' (by omega)]]
      rw [Nat.mul_comm, ← Nat.shiftLeft_eq, Nat.testBit_shiftLeft, Nat.testBit_two_pow_sub_one]
      by_cases h64 : i < 64 <;> simp [hi, h64] <;> omega
    rw [hsub, Nat.add_sub_cancel' hi]
    by_cases h64 : i < 64
    · simp [hi, h64]
    · simp only [h64, decide_false, Bool.and_false, hi, decide_true, Bool.true_and]
      exact (Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le hx (Nat.pow_le_pow_right (by decide) (by omega)))).symm
  · have : (2 ^ 64 - 2 ^ k).testBit i = false := by
      rw [show 2 ^ 64 - 2 ^ k = 2 ^ k * (2 ^ (64 - k) - 1) by
        rw [Nat.mul_sub, Nat.mul_one, ← Nat.pow_add, Nat.add_sub_cancel' (by omega)]]
      rw [Nat.mul_comm, ← Nat.shiftLeft_eq, Nat.testBit_shiftLeft]
      simp [hi]
    simp [this, hi]

/-- A fit asks only a region's size and whether memory is frozen. -/
theorem Fits.congr {m m' : Mem} {a : UInt64} {n : Nat} (hs : m'.sizes = m.sizes) (hz : m'.frozen = m.frozen)
    (h : Fits m a n) : Fits m' a n := by
  obtain ⟨r, off, hd, hp, hsz, hf, hsp⟩ := h
  exact ⟨r, off, hd, hp, by rw [hs]; exact hsz, by rw [hz]; exact hf, hsp⟩

/-- Lanes that each fit store. -/
theorem foldlM_lane_stores_some {addr : UInt64} {lw : Nat} : ∀ (l : List (UInt64 × Nat)) (m0 : Mem),
    (∀ p ∈ l, Fits m0 (addr + UInt64.ofNat (p.2 * lw)) lw) →
    ∃ m, l.foldlM (fun mm (p : UInt64 × Nat) => mm.store (addr + UInt64.ofNat (p.2 * lw)) lw p.1) m0 = some m
  | [], m0, _ => ⟨m0, rfl⟩
  | p :: l, m0, h => by
      obtain ⟨m1, h1⟩ := (h p (by simp)).store p.1
      simp only [List.foldlM_cons, h1, Option.bind_eq_bind, Option.bind_some]
      exact foldlM_lane_stores_some l m1 fun q hq =>
        (h q (by simp [hq])).congr (Contracts.store_sizes_eq h1) (Contracts.store_frozen h1)

/-- **A vector store, by the typestate**: where the post allows no fault, its
    bytes fit; it keeps every fact but the cells. -/
theorem wp_store_vec {cfg : Cfg} {d : Nat} {J : Hoare.Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty : ClifTy} {v : Slot ty} {a : Slot .i64} {k : Prog Slot Lvl α}
    (S : Contracts.TState) {t' : ClifTy} {x : UInt64} {T lane : ClifTy} {n : Nat} {ls : Array UInt64}
    (ha : Γ[a]? = some (.sc t' x)) (hv : Γ[v]? = some (.vec T ls)) (hl : T.lanes = some (lane, n))
    (hn : ls.size = n) (hS : S.holds w) (hJ : J.faultOk = true ∨ Fits w.mem x (n * tyBytes lane))
    (hk : ∀ w', Contracts.TState.holds (S.filter (!·.isCell)) w' → wp cfg d J k Q Γ w') :
    wp cfg d J (.store v a k) Q Γ w := by
  have hw : 0 < tyBytes lane := by
    cases T <;> simp [ClifTy.lanes] at hl <;> obtain ⟨rfl, rfl⟩ := hl <;> decide
  have hlw : ((some (lane, n)).map (·.1.width)).getD 8 / 8 = tyBytes lane := rfl
  subst hn
  rw [wp_store]
  refine ⟨fun m hf => ?_, fun w' hrun => ?_⟩
  · rcases hJ with hJ | hfit
    · exact hJ
    simp only [runStmt, Sem.get, ha, hv, hl] at hf
    rw [hlw] at hf
    obtain ⟨mm, hmm⟩ := foldlM_lane_stores_some (addr := x) (lw := tyBytes lane) ls.zipIdx.toList w.mem
      (fun p hp => by
        obtain ⟨-, hi, -⟩ := Array.mem_zipIdx (Array.mem_toList_iff.mp hp)
        exact hfit.part (by
          have := Nat.mul_le_mul_right (tyBytes lane) (Nat.succ_le_of_lt hi)
          rw [Nat.succ_mul] at this; simp at this; omega) hw)
    rw [Array.foldlM_toList] at hmm
    rw [hmm] at hf; cases hf
  · simp only [runStmt, Sem.get, ha, hv, hl] at hrun
    rw [hlw] at hrun
    cases hmm : ls.zipIdx.foldlM (fun mm (p : UInt64 × Nat) =>
        mm.store (x + UInt64.ofNat (p.2 * tyBytes lane)) (tyBytes lane) p.1) w.mem with
    | none => rw [hmm] at hrun; cases hrun
    | some m =>
        rw [hmm] at hrun; cases hrun
        rw [← Array.foldlM_toList] at hmm
        obtain ⟨hs, hz⟩ := foldlM_lane_stores_inv _ _ _ hmm
        exact hk _ (Contracts.nonCell_kept hS (store_parts hz) (fun kk _ => by cases kk <;> rfl) rfl
          ⟨rfl, rfl, rfl⟩ hs)

/-- `wp_store_vec`, under default flags: a vector stores lane by lane all the
    same. -/
theorem wp_storeU_vec {cfg : Cfg} {d : Nat} {J : Hoare.Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty : ClifTy} {v : Slot ty} {a : Slot .i64} {k : Prog Slot Lvl α}
    (S : Contracts.TState) {t' : ClifTy} {x : UInt64} {T lane : ClifTy} {n : Nat} {ls : Array UInt64}
    (ha : Γ[a]? = some (.sc t' x)) (hv : Γ[v]? = some (.vec T ls)) (hl : T.lanes = some (lane, n))
    (hn : ls.size = n) (hS : S.holds w) (hJ : J.faultOk = true ∨ Fits w.mem x (n * tyBytes lane))
    (hk : ∀ w', Contracts.TState.holds (S.filter (!·.isCell)) w' → wp cfg d J k Q Γ w') :
    wp cfg d J (.storeUnaligned v a k) Q Γ w := by
  have hw : 0 < tyBytes lane := by
    cases T <;> simp [ClifTy.lanes] at hl <;> obtain ⟨rfl, rfl⟩ := hl <;> decide
  have hlw : ((some (lane, n)).map (·.1.width)).getD 8 / 8 = tyBytes lane := rfl
  subst hn
  rw [wp_storeUnaligned]
  refine ⟨fun m hf => ?_, fun w' hrun => ?_⟩
  · rcases hJ with hJ | hfit
    · exact hJ
    simp only [runStmt, Sem.get, ha, hv, hl] at hf
    rw [hlw] at hf
    obtain ⟨mm, hmm⟩ := foldlM_lane_stores_some (addr := x) (lw := tyBytes lane) ls.zipIdx.toList w.mem
      (fun p hp => by
        obtain ⟨-, hi, -⟩ := Array.mem_zipIdx (Array.mem_toList_iff.mp hp)
        exact hfit.part (by
          have := Nat.mul_le_mul_right (tyBytes lane) (Nat.succ_le_of_lt hi)
          rw [Nat.succ_mul] at this; simp at this; omega) hw)
    rw [Array.foldlM_toList] at hmm
    rw [hmm] at hf; cases hf
  · simp only [runStmt, Sem.get, ha, hv, hl] at hrun
    rw [hlw] at hrun
    cases hmm : ls.zipIdx.foldlM (fun mm (p : UInt64 × Nat) =>
        mm.store (x + UInt64.ofNat (p.2 * tyBytes lane)) (tyBytes lane) p.1) w.mem with
    | none => rw [hmm] at hrun; cases hrun
    | some m =>
        rw [hmm] at hrun; cases hrun
        rw [← Array.foldlM_toList] at hmm
        obtain ⟨hs, hz⟩ := foldlM_lane_stores_inv _ _ _ hmm
        exact hk _ (Contracts.nonCell_kept hS (store_parts hz) (fun kk _ => by cases kk <;> rfl) rfl
          ⟨rfl, rfl, rfl⟩ hs)

/-- `wp_store_vec`, the vector known by its type: the typestate after it given
    as its cells dropped. -/
theorem wp_store_tyv {cfg : Cfg} {d : Nat} {J : Hoare.Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty : ClifTy} {v : Slot ty} {a : Slot .i64} {k : Prog Slot Lvl α}
    (S S' : Contracts.TState) {t' : ClifTy} {x : UInt64} {T lane : ClifTy} {n : Nat} {c : V}
    (ha : Γ[a]? = some (.sc t' x)) (hv : Γ[v]? = some c) (hty : TyV T c) (hl : T.lanes = some (lane, n))
    (hS : S.holds w) (hJ : J.faultOk = true ∨ Fits w.mem x (n * tyBytes lane))
    (hS' : S.filter (!·.isCell) = S')
    (hk : ∀ w', Contracts.TState.holds S' w' → wp cfg d J k Q Γ w') :
    wp cfg d J (.store v a k) Q Γ w := by
  unfold TyV at hty; rw [hl] at hty
  obtain ⟨xs, rfl, hn⟩ := hty
  exact wp_store_vec S ha hv hl hn hS hJ fun w' h => hk w' (hS' ▸ h)

/-- `wp_store_tyv`, under default flags. -/
theorem wp_storeU_tyv {cfg : Cfg} {d : Nat} {J : Hoare.Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {ty : ClifTy} {v : Slot ty} {a : Slot .i64} {k : Prog Slot Lvl α}
    (S S' : Contracts.TState) {t' : ClifTy} {x : UInt64} {T lane : ClifTy} {n : Nat} {c : V}
    (ha : Γ[a]? = some (.sc t' x)) (hv : Γ[v]? = some c) (hty : TyV T c) (hl : T.lanes = some (lane, n))
    (hS : S.holds w) (hJ : J.faultOk = true ∨ Fits w.mem x (n * tyBytes lane))
    (hS' : S.filter (!·.isCell) = S')
    (hk : ∀ w', Contracts.TState.holds S' w' → wp cfg d J k Q Γ w') :
    wp cfg d J (.storeUnaligned v a k) Q Γ w := by
  unfold TyV at hty; rw [hl] at hty
  obtain ⟨xs, rfl, hn⟩ := hty
  exact wp_storeU_vec S ha hv hl hn hS hJ fun w' h => hk w' (hS' ▸ h)

/-- `wp_ite_vc`, the values each arm ends in of their types, and so the join's. -/
theorem wp_ite_vcT {cfg : Cfg} {d : Nat} {J : Hoare.Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {jTys : List ClifTy} {c : Cond Slot} {thn els : Prog Slot Lvl (Vals Slot jTys)}
    {k : Vals Slot jTys → Prog Slot Lvl α} (W : World → Prop) (Rt Re : Prop)
    (hRt : Refines w.mem Γ c true Rt) (hRe : Refines w.mem Γ c false Re)
    (hT : ∀ Γ', Ext Γ Γ' → Rt → wp cfg d J thn (fun jv Γe we => Ext Γ Γe ∧ W we ∧ SlotsOk Γe jv) Γ' w)
    (hE : ∀ Γ', Ext Γ Γ' → Re → wp cfg d J els (fun jv Γe we => Ext Γ Γe ∧ W we ∧ SlotsOk Γe jv) Γ' w)
    (hK : ∀ nJ Γ' w', Ext Γ Γ' → W w' → CarryAt Γ' nJ jTys → wp cfg d J (k (carriesFrom nJ jTys)) Q Γ' w')
    (hJ : J.faultOk = true ∨ IcmpOk Γ c.a c.b) :
    wp cfg d J (.ite c thn els k) Q Γ w := by
  rw [wp_ite]
  exact ⟨fun jv Γe we => Ext Γ Γe ∧ W we ∧ SlotsOk Γe jv, fun jv Γe we => Ext Γ Γe ∧ W we ∧ SlotsOk Γe jv,
    ⟨fun t f h hf => hT _ (Ext.push Γ _) (hRt t f h hf),
     fun ne t f hne h hf => hE _ (Ext.bindAt_push Γ _ hne) (hRe t f h hf),
     fun hn => hJ.elim id fun h => absurd hn (h.some _)⟩,
    fun nJ Γ' jv vs w' hle hvs hq => by
      have h := hq.elim id id
      exact hK nJ _ w' (Ext.carry vs h.1 (Nat.le_trans h.1.1 hle)) h.2.1
        (carryAt_of (h.2.2.carry hvs) fun i => bindAt_get_carry Γ' nJ vs i)⟩


/-- **A branch joining a 64-bit value, the value told**: as `wp_ite_vcT`, and
    each arm ends with its first joined value of which `A` holds, so what
    follows knows it of the value it joins. -/
theorem wp_ite_vcJ {cfg : Cfg} {d : Nat} {J : Hoare.Post} {α : Type} {Q : α → Env → World → Prop}
    {Γ : Env} {w : World} {jTys : List ClifTy} {c : Cond Slot} {thn els : Prog Slot Lvl (Vals Slot jTys)}
    {k : Vals Slot jTys → Prog Slot Lvl α} (W : World → Prop) (Rt Re : Prop) (A : UInt64 → Prop)
    (h0 : 0 < jTys.length)
    (hRt : Refines w.mem Γ c true Rt) (hRe : Refines w.mem Γ c false Re)
    (hT : ∀ Γ', Ext Γ Γ' → Rt → wp cfg d J thn (fun jv Γe we => Ext Γ Γe ∧ W we ∧ SlotsOk Γe jv ∧
      ∃ y, Γe[(jv.slots[0]?).getD 0]? = some (.sc .i64 y) ∧ A y) Γ' w)
    (hE : ∀ Γ', Ext Γ Γ' → Re → wp cfg d J els (fun jv Γe we => Ext Γ Γe ∧ W we ∧ SlotsOk Γe jv ∧
      ∃ y, Γe[(jv.slots[0]?).getD 0]? = some (.sc .i64 y) ∧ A y) Γ' w)
    (hK : ∀ nJ Γ' w' y, Ext Γ Γ' → W w' → CarryAt Γ' nJ jTys → Γ'[nJ + 0]? = some (.sc .i64 y) → A y →
      wp cfg d J (k (carriesFrom nJ jTys)) Q Γ' w')
    (hJ : J.faultOk = true ∨ IcmpOk Γ c.a c.b) :
    wp cfg d J (.ite c thn els k) Q Γ w := by
  rw [wp_ite]
  let R := fun (jv : Vals Slot jTys) (Γe : Env) (we : World) => Ext Γ Γe ∧ W we ∧ SlotsOk Γe jv ∧
      ∃ y, Γe[(jv.slots[0]?).getD 0]? = some (.sc .i64 y) ∧ A y
  exact ⟨R, R,
    ⟨fun t f h hf => hT _ (Ext.push Γ _) (hRt t f h hf),
     fun ne t f hne h hf => hE _ (Ext.bindAt_push Γ _ hne) (hRe t f h hf),
     fun hn => hJ.elim id fun h => absurd hn (h.some _)⟩,
    fun nJ Γ' jv vs w' hle hvs hq => by
      have h : R jv Γ' w' := hq.elim id id
      obtain ⟨he, hw, hs, y, hy, hA⟩ := h
      have hj : 0 < jv.slots.length := by rw [Vals.slots_length]; exact h0
      refine hK nJ _ w' y (Ext.carry vs he (Nat.le_trans he.1 hle)) hw
        (carryAt_of (hs.carry hvs) fun i => bindAt_get_carry Γ' nJ vs i) ?_ hA
      rw [bindAt_get_carry, mapM_get hvs 0, List.getElem?_eq_getElem hj, Option.bind_some]
      rw [List.getElem?_eq_getElem hj] at hy
      exact hy⟩

/-- A mask that clears the low `k` bits, given by its value. -/
theorem toNat_and_lowMask {x m : UInt64} (k : Nat) (hk : k < 64) (hm : m.toNat = 2 ^ 64 - 2 ^ k) :
    (x &&& m).toNat = x.toNat / 2 ^ k * 2 ^ k := by
  have : m = UInt64.ofNat (2 ^ 64 - 2 ^ k) := by
    apply UInt64.toNat_inj.mp
    rw [hm, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by have := Nat.one_le_two_pow (n := k); omega)]
  subst this; exact toNat_and_mask hk

-- ---------------------------------------------------------------------------
-- The condition generator
--
-- `prog_vc W` walks a body from the front, one construct at a time: a bind
-- puts its first part in front, a constant binds its value, an operation binds
-- whatever it evaluates to, a counted loop and a call go by the typestate, and
-- anything else is unfolded one definition. Each step looks at the head of the
-- program only, so the work is linear in the program.
-- ---------------------------------------------------------------------------

section vc
variable {cfg : Cfg} {d : Nat} {J : Post} {α β : Type} {Q : α → Env → World → Prop} {Γ : Env} {w : World}

theorem wp_bind_of {p : Prog Slot Lvl β} {f : β → Prog Slot Lvl α}
    (h : wp cfg d J p (fun a => wp cfg d J (f a) Q) Γ w) : wp cfg d J (p >>= f) Q Γ w := by
  rw [wp_bind]; exact h
theorem wp_ret_of {a : α} (h : Q a Γ w) : wp cfg d J (.ret a) Q Γ w := h
theorem wp_pure_of {a : α} (h : Q a Γ w) : wp cfg d J (pure a) Q Γ w := h
theorem wp_params_of {tys} {k : Vals Slot tys → Prog Slot Lvl α}
    (h : wp cfg d J (k (carriesFrom 0 tys)) Q Γ w) : wp cfg d J (.params tys k) Q Γ w := h
theorem wp_forM_cons_of {γ : Type} {a : γ} {as : List γ} {f : γ → Prog Slot Lvl PUnit}
    {Q : PUnit → Env → World → Prop} (h : wp cfg d J (f a) (fun _ => wp cfg d J (as.forM f) Q) Γ w) :
    wp cfg d J ((a :: as).forM f) Q Γ w := by rw [wp_forM_cons]; exact h
theorem wp_forM_nil_of {γ : Type} {f : γ → Prog Slot Lvl PUnit} {Q : PUnit → Env → World → Prop}
    (h : Q ⟨⟩ Γ w) : wp cfg d J (([] : List γ).forM f) Q Γ w := h

theorem wp_forIn_cons_of {γ δ : Type} {a : γ} {as : List γ} {b : δ} {f : γ → δ → Prog Slot Lvl (ForInStep δ)}
    {Q : δ → Env → World → Prop}
    (h : wp cfg d J (f a b) (fun r => wp cfg d J (match r with
      | .done b => pure b
      | .yield b => forIn as b f) Q) Γ w) :
    wp cfg d J (forIn (a :: as) b f) Q Γ w := by
  rw [List.forIn_cons]; exact wp_bind_of h
theorem wp_forIn_nil_of {γ δ : Type} {b : δ} {f : γ → δ → Prog Slot Lvl (ForInStep δ)}
    {Q : δ → Env → World → Prop} (h : Q b Γ w) : wp cfg d J (forIn ([] : List γ) b f) Q Γ w := by
  rw [List.forIn_nil]; exact h

end vc

theorem toNat_and_le_left (a b : UInt64) : (a &&& b).toNat ≤ a.toNat := by
  rw [UInt64.toNat_and]; exact Nat.and_le_left

theorem toNat_and_le_right (a b : UInt64) : (a &&& b).toNat ≤ b.toNat := by
  rw [UInt64.toNat_and]; exact Nat.and_le_right

/-- For each `(a &&& b).toNat` the goal and its hypotheses name, the fact that
    it is at most `a.toNat`, and for each value's `toNat`, that it is below
    `2 ^ 64`: linear arithmetic reads both as atoms. -/
elab "and_facts" : tactic => do
  let g ← Lean.Elab.Tactic.getMainGoal
  g.withContext do
  let mut ts : Array Lean.Expr := #[(← Lean.instantiateMVars (← g.getType))]
  for d in ← Lean.getLCtx do
    unless d.isImplementationDetail do ts := ts.push (← Lean.instantiateMVars d.type)
  let mut found : Array Lean.Expr := #[]
  let hits ← IO.mkRef (#[] : Array Lean.Expr)
  let vals ← IO.mkRef (#[] : Array Lean.Expr)
  for t in ts do
    t.forEach fun x => do
      if x.isAppOfArity ``UInt64.toNat 1 then
        let y := x.getArg! 0
        if y.hasFVar then vals.modify (·.push y)
      -- a mask anywhere, a sum's operand among them: simplification puts
      -- its number apart
      if x.isAppOfArity ``HAnd.hAnd 6 && (x.getArg! 0).isConstOf ``UInt64 then hits.modify (·.push x)
  for e in ← hits.get do
    unless found.contains e do found := found.push e
  let mut vs : Array Lean.Expr := #[]
  for e in ← vals.get do
    unless vs.contains e do vs := vs.push e
  let mut g := g
  for e in found do
    let pf ← Lean.Meta.mkAppM ``toNat_and_le_left #[e.getArg! 4, e.getArg! 5]
    let (_, g') ← g.note `hand pf
    g := g'
    let pf ← Lean.Meta.mkAppM ``toNat_and_le_right #[e.getArg! 4, e.getArg! 5]
    let (_, g') ← g.note `hand pf
    g := g'
    -- a mask that clears the low bits: the value rounded down
    let b := e.getArg! 5
    -- a literal only, read off as written: computing one would walk the
    -- proof a computed literal carries
    let litOf (b : Lean.Expr) : Option Nat :=
      if b.isAppOfArity ``OfNat.ofNat 3 then (b.getArg! 1).rawNatLit?
      else if b.isAppOfArity ``UInt64.ofBitVec 1 && (b.getArg! 0).isAppOfArity ``BitVec.ofFin 2 &&
          ((b.getArg! 0).getArg! 1).isAppOfArity ``Fin.mk 3 then
        (((b.getArg! 0).getArg! 1).getArg! 1).rawNatLit? <|> (((b.getArg! 0).getArg! 1).getArg! 1).nat?
      else none
    -- a literal, or the complement of one
    let lit? : Option Nat :=
      if b.isAppOfArity ``Complement.complement 3 then (litOf (b.getArg! 2)).map (2 ^ 64 - 1 - ·)
      else litOf b
    if !b.hasFVar && !b.hasMVar then
      if let some v := lit? then
        let dd := 2 ^ 64 - v
        if 0 < dd && v < 2 ^ 64 && dd &&& (dd - 1) == 0 then
          let kk := Nat.log2 dd
          let hk ← Lean.Meta.mkDecideProof (← Lean.Meta.mkAppM ``LT.lt #[Lean.mkNatLit kk, Lean.mkNatLit 64])
          let hm ← Lean.Meta.mkDecideProof (← Lean.Meta.mkEq (← Lean.Meta.mkAppM ``UInt64.toNat #[b])
            (← Lean.Meta.mkAppM ``HSub.hSub #[← Lean.Meta.mkAppM ``HPow.hPow #[Lean.mkNatLit 2, Lean.mkNatLit 64],
              ← Lean.Meta.mkAppM ``HPow.hPow #[Lean.mkNatLit 2, Lean.mkNatLit kk]]))
          let pm ← Lean.Meta.mkAppOptM ``toNat_and_lowMask #[e.getArg! 4, b, Lean.mkNatLit kk, hk, hm]
          let (_, g') ← g.note `hmask pm
          g := g'
  -- and that a value's number is below `2 ^ 64`, which omega does not know
  for e in vs do
    let (_, g') ← g.note `hlt (← Lean.Meta.mkAppM ``UInt64.toNat_lt #[e])
    g := g'
  Lean.Elab.Tactic.replaceMainGoal [g]

/-- Room at an address a value past a known one: the region's room covers
    the offset and the length. -/
theorem roomAt_add {S : Contracts.TState} {b x : UInt64} {n K : Nat} {r : Region} {off : Nat}
    (hm : Contracts.Fact.room r K ∈ S) (hz : S.contains (.part .frozen false) = true)
    (hd : decodeAddr b = some (r, off)) (hp : r ≠ .pinned) (hb : off + x.toNat + n ≤ K)
    (hs : off + x.toNat + n < 2 ^ 36) : Contracts.roomAt S (b + x) n = true := by
  have hd' : decodeAddr (b + x) = some (r, off + x.toNat) := by
    have := Static.decodeAddr_add (i := x.toNat) hd (by simp only [regionSpan, UInt64.reduceToNat]; omega)
    rwa [UInt64.ofNat_toNat] at this
  unfold Contracts.roomAt
  rw [hd']
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.any_eq_true]
  refine ⟨⟨⟨hp, by omega⟩, hz⟩, _, hm, ?_⟩
  simp only [Bool.and_eq_true, decide_eq_true_eq, true_and]
  omega

/-- What a window presents from sees the same device a program opened. -/
theorem pump_gpu_live (w : World) : w.pump.gpu.live = w.gpu.live := by
  rw [(Contracts.pump_life w).2.2.2.2.1]

/-- **Room for page-locked memory**, from a bound on what it spans. -/
theorem pinnedRoom_of_state {S : Contracts.TState} {w : World} {n : Nat} {size : UInt64} (hS : S.holds w)
    (hm : Contracts.Fact.pinnedUsed n ∈ S) (hb : n + 63 + size.toNat ≤ regionSpan.toNat) :
    Contracts.PinnedRoom w size := by
  have hn : w.mem.sizes .pinned ≤ n := hS _ hm
  unfold Contracts.PinnedRoom
  have := Nat.mod_lt (64 - w.mem.pinned.size % 64) (by decide : 64 > 0)
  change w.mem.pinned.size ≤ n at hn
  omega

/-- `readable_upto_of_state`, for a write. -/
theorem writable_upto_of_state {S : Contracts.TState} {w : World} {a : UInt64} {n K : Nat} (hS : S.holds w)
    (h : Contracts.roomAt S a K = true) (hn : n ≤ K) : Writable w.mem a n :=
  Contracts.writable_of_state hS (Contracts.roomAt_mono h hn)

/-- Only the hypotheses linear arithmetic reads: comparisons and equations of
    numbers, tests' outcomes, and what a call's answer is known to be. The
    rest, an environment's slots and the typestate among them, is cleared, so
    the simplification that follows does not walk it. -/
elab "arith_ctx" : tactic => do
  let g ← Lean.Elab.Tactic.getMainGoal
  g.withContext do
  let rec arith (t : Lean.Expr) : Bool :=
    if t.isAppOfArity ``And 2 || t.isAppOfArity ``Or 2 || t.isAppOfArity ``Iff 2 then
      arith (t.getArg! 0) && arith (t.getArg! 1)
    else if t.isAppOfArity ``Not 1 then arith (t.getArg! 0)
    else if t.isAppOfArity ``LE.le 4 || t.isAppOfArity ``LT.lt 4 || t.isAppOfArity ``GE.ge 4 ||
      t.isAppOfArity ``GT.gt 4 then true
    else if t.isAppOfArity ``Eq 3 || t.isAppOfArity ``Ne 3 then
      let ty := t.getArg! 0
      ty.isConstOf ``Nat || ty.isConstOf ``UInt64 || ty.isConstOf ``Bool || ty.isConstOf ``Int
    else if t.isAppOf ``answerOk then true
    else if t.isForall then arith t.bindingDomain! && arith t.bindingBody!
    else false
  let mut drop := #[]
  -- the arithmetic hypotheses, and the variables each names
  let mut cands : Array (Lean.FVarId × Array Lean.FVarId) := #[]
  for d in ← Lean.getLCtx do
    if d.isImplementationDetail then continue
    unless ← Lean.Meta.isProp d.type do continue
    let ty ← Lean.instantiateMVars d.type
    if arith ty then cands := cands.push (d.fvarId, (Lean.collectFVars {} ty).fvarIds)
    else drop := drop.push d.fvarId
  -- only those that reach the goal through the variables they share: the
  -- rest is about other values, and linear arithmetic would split on it
  let mut reach : Std.HashSet Lean.FVarId :=
    Std.HashSet.ofArray (Lean.collectFVars {} (← Lean.instantiateMVars (← g.getType))).fvarIds
  let mut kept : Std.HashSet Lean.FVarId := {}
  let mut changed := true
  while changed do
    changed := false
    for (h, vs) in cands do
      if kept.contains h then continue
      if vs.any reach.contains then
        kept := kept.insert h
        for v in vs do reach := reach.insert v
        changed := true
  -- a goal naming no variable, or one still to be filled, keeps them all
  let goalTy ← Lean.instantiateMVars (← g.getType)
  if goalTy.hasFVar && !goalTy.hasMVar then
    for (h, _) in cands do
      unless kept.contains h do drop := drop.push h
  Lean.Elab.Tactic.replaceMainGoal [← g.tryClearMany drop]

/-- A bound on computed values, by linear arithmetic over what the context
    says of them: a sum's value is its operands' when it does not wrap. -/
macro "bound_close" : tactic => `(tactic| (arith_ctx; first
  | omega
  | (simp only [UInt64.toNat_add, UInt64.toNat_ofNat, UInt64.reduceToNat, Nat.reducePow, Nat.reduceMod]
     first | done | omega)
  | (simp only [UInt64.toNat_add, UInt64.toNat, BitVec.toNat, Fin.val_mk, Nat.reducePow, Nat.reduceMod,
       UInt64.toBitVec] at *
     first | done | omega)
  | (and_facts
     try simp only [answerOk] at *
     -- signed comparisons of values below `2 ^ 63`, as their numbers: before
     -- the numbers are taken apart, so the bounds read as written
     try simp (disch := omega) only [cmpInt, signed_i64_of_lt] at *
     simp only [cmpInt, decide_eq_true_eq, decide_eq_false_iff_not, beq_iff_eq, beq_eq_false_iff_ne,
       bne_iff_ne, ne_eq, ← UInt64.toNat_inj, UInt64.le_iff_toNat_le,
       UInt64.lt_iff_toNat_lt, ge_iff_le, gt_iff_lt, Nat.not_le, Nat.not_lt, UInt64.toNat_add,
       UInt64.toNat_sub, UInt64.toNat_mul, UInt64.toNat_div, UInt64.toNat_shiftLeft, UInt64.toNat_shiftRight, UInt64.toNat_mod,
       UInt64.toNat_ofNat, UInt64.reduceToNat, Contracts.toNat_mk_lit, Nat.shiftLeft_eq,
       Nat.shiftRight_eq_div_pow,
       UInt64.reduceMod, Nat.reducePow, Nat.reduceMod, Nat.reduceMul] at *
     -- sums that provably do not wrap lose their `% 2 ^ 64`: omega's
     -- elimination misses bounds through a modulus
     try simp (disch := omega) only [Nat.mod_eq_of_lt] at *
     first
       | done
       | omega
       | (try simp only [UInt64.toNat, BitVec.toNat, Fin.val_mk] at *
          try simp only [Nat.reducePow, Nat.reduceMod, Nat.reduceMul, Nat.shiftRight_eq_div_pow] at *
          first | done | omega))))

/-- Discharge a call's `Moves` from the contract table. A typestate of the
    program's own adds a rule. -/
syntax "prog_keeps" : tactic

/-- A decidable proposition the kernel computes true, where the proposition
    may name the values a program was handed: `decide` refuses a goal with free
    variables, though what it asks need not look at them. -/
syntax "kdecide" : tactic

/-- Memory a call reads, `n` bytes where `n` is known by a bound: room for
    as much as the typestate names there, and `n` within it. -/
syntax "readable_bound" : tactic

/-- Memory a call reads or writes by a length the program was handed: the
    region the address is in holds as many bytes as a value the typestate
    names, and the access lies within them. -/
syntax "room_arg" : tactic
syntax "fact_mem" : tactic

/-- Memory a call reads or writes at a value past an address known outright:
    the region's room covers the offset and the length. -/
syntax "room_add" : tactic

/-- Room for page-locked memory, from the typestate's bound on it. -/
syntax "pinned_room" : tactic

/-- A product fits buffers whose sizes are written as numbers: the fit is then
    arithmetic on them. -/
macro "gemm_fits_lit" : tactic => `(tactic| (
  intro _ _ _ hA hB hC
  simp only [AlgorithmLib.HProg.Sem.sgemmFits, AlgorithmLib.HProg.Sem.gemmStridedFits,
    AlgorithmLib.HProg.Sem.colMajorSpan, hA, hB, hC]
  kdecide))

/-- A strided product's fit, from its buffers' sizes: each dimension, stride
    and span computed out over the bounds the context gives. -/
macro "gemm_fits" : tactic => `(tactic| (
  intro _ _ _ hA hB hC
  simp (config := { decide := true })
    (disch := (simp only [UInt64.toNat_mul, UInt64.toNat_shiftLeft, UInt64.reduceToNat, Nat.shiftLeft_eq,
      Nat.reduceMod, Contracts.toNat_mk_lit] at *; omega))
    only [Contracts.toNat_mk_lit, sgemmFits, gemmStridedFits, colMajorSpan, hA, hB, hC, Contracts.asI32_low, Contracts.asI32_small,
      Contracts.asI64_low,
      Int.toNat_natCast, Bool.and_eq_true, decide_eq_true_eq, Bool.or_eq_true, if_true, if_false]
  simp only [UInt64.toNat_mul, UInt64.toNat_shiftLeft, UInt64.reduceToNat, Nat.shiftLeft_eq, Nat.reduceMod,
    Nat.reduceSub, Nat.reduceMul, Nat.reduceAdd, Nat.reducePow, Nat.mul_zero, Nat.zero_add, false_or, or_false,
    true_and, and_true, Contracts.toNat_mk_lit] at *
  (repeat' split) <;> omega))

/-- The arithmetic a product's fit comes to: each dimension cut to 32 bits
    read back as itself, each size in bytes of four without wrapping, by the
    bounds the context gives. -/
macro "gemm_close" : tactic => `(tactic| (
  simp (config := { decide := true }) (disch := omega) only [Contracts.asI32_low, Int.toNat_natCast,
    Contracts.mul_shl2, Contracts.shl2_low, Int.natCast_pos, if_true, if_false]
  first | omega | (simp only [Nat.mul_comm]; omega)))

/-- Fails unless the goal, as written, is an application of one of `cs`: a
    guard for rules whose unification would otherwise unfold an unrelated
    goal. -/
elab "goal_head " cs:ident+ : tactic => do
  let mut ns : Array Lean.Name := #[]
  for c in cs do ns := ns.push (← Lean.Elab.realizeGlobalConstNoOverloadWithInfo c)
  let t ← Lean.instantiateMVars (← Lean.Elab.Tactic.getMainTarget)
  let f := t.cleanupAnnotations.getAppFn.consumeMData
  unless ns.any f.isConstOf do throwError "goal_head: {f} is not one of {ns}"

/-- Arithmetic on addresses: a sum of a literal and a scaled index, as numbers,
    with the width of the access written out. -/
macro "addr_close" : tactic => `(tactic| first
  | (simp only [tyBytes, ClifTy.width, Nat.reduceDiv, UInt64.toNat_add, UInt64.toNat_mul,
      Contracts.toNat_mk_lit, Nat.reducePow]; omega)
  | bound_close)

/-- A store keeps one fact, by the bounds of its address. -/
macro "store_keep_one" : tactic => `(tactic| first
  | (apply Contracts.storeKeeps_cell_of (by kdecide) <;> addr_close)
  | (apply Contracts.storeKeeps_cstrIn_of; addr_close)
  | (simp only [Contracts.storeKeeps]; done))

/-- Fails unless the goal, as written, says a device's capture is `none`: a
    guard for the rule that proves it, whose unification with another equation
    would unfold that equation's reader. -/
elab "goal_capture" : tactic => do
  let t := (← Lean.instantiateMVars (← Lean.Elab.Tactic.getMainTarget)).cleanupAnnotations
  let ok := t.isAppOfArity ``Eq 3 && (t.getArg! 1).consumeMData.isAppOf ``Dev.capture &&
    (t.getArg! 2).consumeMData.isAppOf ``Option.none
  unless ok do throwError "goal_capture: not a capture"

/-- A contract's precondition, from the facts a typestate names: memory it
    may touch from its rooms and cells, and everything else from its parts and
    cells by rewriting. -/
macro "pre_state" : tactic => `(tactic| (
  intro w hS
  try simp only [Contracts.Pre]
  repeat' apply And.intro
  all_goals first
    | (goal_head AlgorithmLib.HProg.DevSpec.CtxOk; exact Contracts.ctxOk_of_state hS (by kdecide))
    | (goal_capture; exact Contracts.capture_of_state hS (by kdecide))
    | (goal_head AlgorithmLib.HProg.Contracts.VendorKeeps; exact Contracts.vendor_of_state hS (by kdecide))
    | (rw [if_neg (by kdecide)]; refine Contracts.readable_of_state hS ?_; kdecide)
    | (goal_head AlgorithmLib.HProg.Contracts.PathOk; refine Contracts.pathOk_of_state hS ?_; kdecide)
    | (goal_head AlgorithmLib.HProg.Contracts.GemmOk; refine Contracts.gemmOk_of_state hS (by kdecide) (by fact_mem) (by fact_mem) (by fact_mem) ?_; first | gemm_fits_lit | gemm_fits)
    | (goal_head AlgorithmLib.HProg.Contracts.SgemvOk; refine Contracts.sgemvOk_of_state hS (by kdecide) (by fact_mem) (by fact_mem) (by fact_mem) ?_ ?_ ?_ ?_ ?_
        <;> first | kdecide | gemm_close)
    | (goal_head AlgorithmLib.HProg.Contracts.CStr Eq; refine Contracts.cstr_of_state hS ?_; kdecide)
    | (goal_head AlgorithmLib.HProg.Contracts.CStr Eq; refine Contracts.cstrIn_of_state hS ?_; kdecide)
    | (goal_head AlgorithmLib.HProg.Contracts.Slot8; refine Contracts.slot8_of_state hS ?_; kdecide)
    | (refine Contracts.store_of_state hS ?_ _; kdecide)
    | (goal_head AlgorithmLib.HProg.Contracts.Load8; refine Contracts.load8_of_cell hS ?_; kdecide)
    | (goal_head AlgorithmLib.HProg.Contracts.Load8; refine Contracts.load8_of_state hS ?_; kdecide)
    | (refine Contracts.load_of_state hS ?_; kdecide)
    | (goal_head AlgorithmLib.HProg.DevSpec.Readable; refine Contracts.readable_of_state hS ?_; kdecide)
    | (goal_head AlgorithmLib.HProg.DevSpec.Writable; refine Contracts.writable_of_state hS ?_; kdecide)
    | (refine Contracts.htLookup_of_state hS ?_; kdecide)
    | (refine Contracts.cstr_imp_of_state hS ?_; kdecide)
    | (goal_head AlgorithmLib.HProg.Contracts.PathOk; refine Contracts.pathOk_of_state hS ?_; kdecide)
    | (refine Contracts.pathOk_imp_of_state hS ?_; kdecide)
    | (refine Contracts.readTo_of_state hS ?_; kdecide)
    | (refine Contracts.fileRead_room_of_state hS ?_; kdecide)
    | (refine Contracts.writable_upto_imp_of_state hS ?_; kdecide)
    | (refine Contracts.ready_of_state hS ?_ _ _; kdecide)
    | (refine Contracts.keepsSize_of_state hS ?_; kdecide)
    | (refine Contracts.vendorKeeps_of_state hS ?_; kdecide)
    | (refine Contracts.bindsReady_of_state hS ?_ ?_ <;> kdecide)
    | (intro _ _ _; refine ⟨?_, ?_⟩ <;> first
        | (goal_head AlgorithmLib.HProg.DevSpec.Readable; refine Contracts.readable_of_state hS ?_; kdecide)
        | (goal_head AlgorithmLib.HProg.DevSpec.Writable; refine Contracts.writable_of_state hS ?_; kdecide)
        | (refine Contracts.ready_of_state hS ?_ _ _; kdecide)
        | room_arg)
    | (intro _ _; refine ⟨?_, ?_⟩ <;> first
        | (goal_head AlgorithmLib.HProg.DevSpec.Readable; refine Contracts.readable_of_state hS ?_; kdecide)
        | (goal_head AlgorithmLib.HProg.DevSpec.Writable; refine Contracts.writable_of_state hS ?_; kdecide)
        | (refine Contracts.ready_of_state hS ?_ _ _; kdecide)
        | (goal_head AlgorithmLib.HProg.Contracts.CStr Eq; refine Contracts.cstr_of_state hS ?_; kdecide)
        | (goal_head AlgorithmLib.HProg.Contracts.CStr Eq; refine Contracts.cstrIn_of_state hS ?_; kdecide)
        | (refine Contracts.readBinds_of_state hS ?_; kdecide)
        | room_arg)
    | readable_bound
    | room_arg
    | (rw [if_neg (by intro h0; have h1 := congrArg UInt64.toNat h0; bound_close)]; readable_bound)
    | (intro _; refine ⟨?_, ?_⟩ <;> first
        | (goal_head AlgorithmLib.HProg.DevSpec.Readable; refine Contracts.readable_of_state hS ?_; kdecide)
        | (goal_head AlgorithmLib.HProg.Contracts.PathOk; refine Contracts.pathOk_of_state hS ?_; kdecide))
    | (intro _ h; exact h.elim)
    | (intro _ h; exact absurd h (by kdecide))
    | (intro _; goal_head AlgorithmLib.HProg.DevSpec.Writable; refine Contracts.pollRoom_of_state hS ?_; kdecide)
    | (intros; first
        | (goal_head AlgorithmLib.HProg.DevSpec.Readable; refine Contracts.readable_of_state hS ?_; kdecide)
        | (goal_head AlgorithmLib.HProg.DevSpec.Writable; refine Contracts.writable_of_state hS ?_; kdecide))
    | (intros; goal_head AlgorithmLib.HProg.DevSpec.Readable AlgorithmLib.HProg.DevSpec.Writable; readable_bound)
    | (intros; goal_head AlgorithmLib.HProg.DevSpec.Readable AlgorithmLib.HProg.DevSpec.Writable; room_add)
    | (goal_head AlgorithmLib.HProg.Contracts.PinnedRoom; pinned_room)
    | (rw [if_neg (by intro h0; have h1 := congrArg UInt64.toNat h0; bound_close)]; room_add)
    | (have hw := hS
       simp only [Contracts.TState.holds, List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq,
         Contracts.Fact.holds, Contracts.Part.get] at hw
       clear hS
       simp_all (config := { decide := true }) only [Contracts.Pre, Contracts.LmdbOk, Contracts.WinOk,
        Contracts.GpuOk, Contracts.ThreadOk, Contracts.LmdbSlotOk, Contracts.WinSlotOk,
        Contracts.ThreadSlotOk, Contracts.Load8, CtxOk, cudaCtx, true_and, and_true, or_true, true_or,
        and_self, not_false_eq_true, Option.isSome_eq_false_iff, Option.isNone_iff_eq_none,
        Option.isSome_some, Option.some.injEq, forall_eq', beq_self_eq_true, pump_gpu_live])
    | (have hw := hS
       simp only [Contracts.TState.holds, List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq,
         Contracts.Fact.holds, Contracts.Part.get] at hw
       clear hS; simp_all [Array.isEmpty_iff])))

open Lean Meta Elab Tactic

/-- A term with each written-out list's length put as its number: an offset
    the program counts off a list of slots names those slots, which its value
    does not depend on. -/
meta def closeLengths (e : Expr) : Expr :=
  e.replace fun x =>
    if x.isAppOfArity ``List.length 2 then
      match x.appArg!.listLit? with
      | some (_, l) => some (mkNatLit l.length)
      | none => none
    else none

/-- A call's argument bits, computed: `vs.mapM asBits = some ?bits` with the
    bits as a list of literals. -/
elab "bits_rfl" : tactic => do
  let g ← getMainGoal
  let T ← instantiateMVars (← g.getType)
  let some (_, lhs, rhs) := T.eq? | throwError "bits_rfl: not an equation"
  -- a term over answers is kept whole, each value's bits read off it, and the
  -- kernel checks the equation
  if lhs.hasFVar then
    if let some (_, cs) := lhs.appArg!.listLit? then
      let mut bs := #[]
      for c in cs do
        let c ← whnf c
        unless c.isAppOfArity ``V.sc 2 do throwError "bits_rfl: arguments are not all scalars"
        let b := closeLengths (c.getArg! 1)
        bs := bs.push (← if b.hasFVar then pure b else withTransparency .all (Meta.reduce b (skipTypes := false)))
      let r ← mkAppM ``Option.some #[← mkListLit (mkConst ``UInt64) bs.toList]
      unless ← isDefEq rhs r do throwError "bits_rfl: bits do not match"
      g.assign (← mkExpectedTypeHint (← mkEqRefl lhs) T)
      return
  let r ← withTransparency .all (Meta.reduce lhs (skipTypes := false))
  unless r.isAppOfArity ``Option.some 2 do throwError "bits_rfl: arguments are not all scalars"
  unless ← isDefEq rhs r do throwError "bits_rfl: bits do not match"
  g.assign (← mkEqRefl lhs)

/-- An equation both sides of which compute to one value, left for the kernel
    to check: the kernel computes with numbers as numbers, where checking it
    here would unfold arithmetic one step at a time. -/
elab "kernel_rfl" : tactic => do
  let g ← getMainGoal
  let T ← instantiateMVars (← g.getType)
  let some (_, lhs, rhs) := T.eq? | throwError "kernel_rfl: not an equation"
  if rhs.hasMVar then throwError "kernel_rfl: the right side is not known"
  g.assign (← mkExpectedTypeHint (← mkEqRefl lhs) T)

macro_rules | `(tactic| prog_keeps) => `(tactic| (
  refine moves_state _ _ _ (by kdecide) (by decide) (by bits_rfl) (by first | kdecide | kernel_rfl | rfl) ?_
  pre_state))

/-- `prog_keeps`, taking the computed typestate on trust for the kernel to
    check when the lemma is added: where the kernel cannot compute it (a
    string, say), adding the lemma fails and `prog_keeps` is run instead. -/
syntax "prog_keeps_fast" : tactic

macro_rules | `(tactic| prog_keeps_fast) => `(tactic| (
  refine moves_state _ _ _ (by kdecide) (by decide) (by bits_rfl) (by first | kernel_rfl | kdecide | rfl) ?_
  pre_state))



/-- What the condition generator knows along one path of a body. -/
structure VCKnow where
  /-- Each environment variable's predecessor, with the `Ext` between them. -/
  parent : Std.HashMap FVarId (Expr × Expr) := {}
  /-- Each slot's value where it was last looked up: the environment, the value,
      and a proof that the environment holds the value there. -/
  facts : Std.HashMap Expr (Expr × Expr × Expr) := {}
  /-- A proof of the typestate, by world variable. -/
  holds : Std.HashMap FVarId Expr := {}
  /-- Each call obligation proved so far, closed, by its statement. -/
  moves : Std.HashMap Expr Expr := {}
  /-- Each closed expression computed out so far: typestates, and what calls
      leave of them. -/
  outs : Std.HashMap Expr Expr := {}
  /-- Each value known only by its type: the type, and a proof `TyV t v`. -/
  tys : Std.HashMap Expr (Expr × Expr) := {}
  /-- Each slot's facts from before it was last looked up elsewhere: an
      environment beside the one a lookup moved the fact to still finds it. -/
  older : Std.HashMap Expr (List (Expr × Expr × Expr)) := {}

/-- A slot's fact, the one it replaces kept among the older ones. -/
meta def VCKnow.addFact (k : VCKnow) (s : Expr) (f : Expr × Expr × Expr) : VCKnow :=
  match k.facts[s]? with
  | some old => if old.1 == f.1 then { k with facts := k.facts.insert s f } else
      { k with facts := k.facts.insert s f, older := k.older.insert s (old :: (k.older[s]?.getD [])) }
  | none => { k with facts := k.facts.insert s f }

/-- The facts a typestate names, if it is a list of facts. -/
meta def stateOf? (T : Expr) : MetaM (Option Expr) := do
  let T ← whnfR T
  if T.isAppOfArity ``Contracts.TState.holds 2 then return some (T.getArg! 0) else return none

/-- The typestate a proof establishes, as a predicate on worlds. -/
meta def typestateOf (W hW : Expr) : MetaM Expr := do
  let T ← instantiateMVars (← inferType hW)
  match ← stateOf? T with
  | some S => return mkApp (mkConst ``Contracts.TState.holds) S
  | none => return W

/-- An expression computed out, as the kernel would. -/
meta def evalOut (e : Expr) : MetaM Expr :=
  withTransparency .all (Meta.reduce e (skipTypes := false))

/-- A closed integer term's value, by running the compiled code for it. -/
meta unsafe def evalIntImpl (e : Expr) : MetaM Int := evalExpr Int (mkConst ``Int) e

@[implemented_by evalIntImpl]
meta opaque evalInt (e : Expr) : MetaM Int

/-- The string operations that only build a string: the kernel computes a
    term through them without forcing the string, unless something measures it
    (a region's name beside its offset, say). -/
meta def stringBuilds : List Name :=
  [`String.append, `String.mk, `List.asString, `String.singleton, `String.push,
   `String.ofList, `String.toString, `String.instAppend, `instToStringString]

/-- Whether a closed term computes through what measures a string, or a
    definition by well-founded recursion, following definitions up to a budget:
    the kernel computes neither at any size. -/
meta def kernelSlow (e : Expr) : MetaM Bool := do
  let env ← getEnv
  let mut seen : NameSet := {}
  let mut todo := e.getUsedConstants.toList
  let mut fuel := 4000
  while fuel > 0 do
    let c :: rest := todo | return false
    todo := rest
    if seen.contains c then continue
    seen := seen.insert c
    fuel := fuel - 1
    if stringBuilds.contains c then continue
    if c.getPrefix == `String || c == ``WellFounded.fix || c == ``Acc.rec then return true
    -- the kernel computes numbers as numbers, whatever their definitions
    if c.getRoot == `Nat || c.getRoot == `Int then continue
    if let some (.defnInfo d) := env.find? c then
      todo := d.value.getUsedConstants.toList ++ todo
  return true

/-- The kernel's weak head normal form: it computes with numbers as numbers. -/
meta def kernelWhnf (e : Expr) : MetaM Expr := do
  ofExceptKernelException (Kernel.whnf (← getEnv) (← getLCtx) e)

/-- A fact as the next step reads it: its constructor, and each argument
    computed out on its own. -/
meta def factOut (x : Expr) : MetaM Expr := do
  let hd ← kernelWhnf x
  return mkAppN hd.getAppFn (← hd.getAppArgs.mapM fun a => do
    if a.isSort || (← inferType a).isSort || a.hasFVar then pure a else evalOut a)

/-- A list of facts computed by the kernel, one cell at a time, each fact's
    arguments then computed out on their own. -/
meta partial def kernelList (e : Expr) : MetaM (Option Expr) := do
  let r ← try kernelWhnf e catch _ => return none
  if r.isAppOfArity ``List.nil 1 then return some r
  unless r.isAppOfArity ``List.cons 3 do return none
  let hd ← factOut (r.getArg! 1)
  let some tl ← kernelList (r.getArg! 2) | return none
  return some (mkApp3 r.getAppFn (r.getArg! 0) hd tl)

/-- A closed list of booleans, by running the compiled code for it. -/
meta unsafe def evalBoolsImpl (e : Expr) : MetaM (List Bool) :=
  evalExpr (List Bool) (mkApp (mkConst ``List [levelZero]) (mkConst ``Bool)) e

@[implemented_by evalBoolsImpl]
meta opaque evalBools (e : Expr) : MetaM (List Bool)

/-- A fact as the predicates of what a step keeps read it: closed, its
    variables read as zero where none of them reads the argument (a region's
    length as handed, a value a cell holds, a device buffer, a page-locked
    count). None where a variable is anywhere else. -/
meta def keepsClosed? (x : Expr) : MetaM (Option Expr) := do
  unless x.hasFVar do return some x
  let zeroable : List (Name × List Nat) :=
    [(``Contracts.Fact.roomArg, [1]), (``Contracts.Fact.cell, [1]),
     (``Contracts.Fact.devBuf, [0, 1]), (``Contracts.Fact.pinnedUsed, [0])]
  let some c := x.getAppFn.constName? | return none
  let some (_, ix) := zeroable.find? (·.1 == c) | return none
  let mut args := x.getAppArgs
  for i in [0:args.size] do
    unless args[i]!.hasFVar do continue
    unless ix.contains i do return none
    args := args.set! i (← mkNumeral (← inferType args[i]!) 0)
  return some (mkAppN x.getAppFn args)

/-- `S.filter p` for a list of facts written out, `p` closed: which facts `p`
    keeps, by running it on each, each kept fact computed out as the kernel's
    walk would leave it. None where a fact's variables might decide it. -/
meta def filterOut? (p S : Expr) : MetaM (Option Expr) := do
  if p.hasFVar || p.hasMVar then return none
  let some (α, xs) := S.listLit? | return none
  let mut cs := #[]
  for x in xs do
    let some c ← keepsClosed? x | return none
    cs := cs.push c
  let mask ← try evalBools (← mkAppM ``List.map #[p, ← mkListLit α cs.toList]) catch _ => return none
  unless mask.length == xs.length do return none
  let kept ← ((xs.zip mask).filterMap fun (x, b) => if b then some x else none).mapM factOut
  return some (← mkListLit α kept)

/-- The calls whose `after` adds or replaces facts; every other call's is
    the facts it keeps. -/
meta def afterRewrites : List Name :=
  [``Ffi.cudaInit, ``Ffi.cudaCleanup, ``Ffi.lmdbInit, ``Ffi.gpuInit, ``Ffi.windowInit, ``Ffi.gpuCleanup,
   ``Ffi.threadInit, ``Ffi.htInit, ``Ffi.htInsert, ``Ffi.htCleanup, ``Ffi.lmdbCleanup, ``Ffi.windowCleanup,
   ``Ffi.threadCleanup, ``Ffi.threadJoin, ``Ffi.cudaPinnedAlloc]

/-- What a call or a store leaves of a typestate written out, without the
    kernel walking the list a cell at a time: the facts it keeps found by
    running what decides each, and for a store of a known eight-byte value to a
    cell, that cell. The kernel checks the list where it stands for the
    expression; a wrong one fails there. -/
meta def fastOut? (e : Expr) : MetaM (Option Expr) := do
  let e ← instantiateMVars e
  if e.isAppOfArity ``Contracts.after 3 then
    let f := e.getArg! 0
    let some c := f.getAppFn.constName? | return none
    if afterRewrites.contains c then return none
    return ← filterOut? (← mkAppM ``Contracts.keeps #[f, e.getArg! 1]) (e.getArg! 2)
  if e.isAppOfArity ``Contracts.afterStore 4 then
    let a := e.getArg! 0
    let v := e.getArg! 1
    let n := e.getArg! 2
    unless a.isAppOfArity ``Option.some 2 do return none
    let a := a.appArg!
    if a.hasFVar || n.hasFVar then return none
    let some k ← filterOut? (← mkAppM ``Contracts.storeKeeps #[a, n]) (e.getArg! 3) | return none
    if v.isAppOfArity ``Option.none 1 then return some k
    unless v.isAppOfArity ``Option.some 2 do return none
    let [cellIt] ← try evalBools (← mkListLit (mkConst ``Bool)
        [← mkAppM ``and #[← mkAppM ``BEq.beq #[n, mkNatLit 8], ← mkAppM ``Contracts.cellOk #[a]]]) catch _ => return none
      | return none
    if cellIt then return some (← mkAppM ``List.cons #[← factOut (← mkAppM ``Contracts.Fact.cell #[a, v.appArg!]), k])
    return some k
  if e.isAppOfArity ``List.filter 3 then
    return ← filterOut? (e.getArg! 1) (e.getArg! 2)
  return none

/-- A computed typestate, as a list the next step can read, when it computes
    to one. The two are equal by computation, which the kernel checks where the
    list stands for the expression; checking it here would unfold arithmetic
    one step at a time. -/
meta def stateOut (e : Expr) : MetaM Expr := do
  if let some r ← fastOut? e then return r
  if let some r ← kernelList e then return r
  -- over a term with variables, reducing here unfolds without end
  if e.hasFVar then return e
  let r ← evalOut e
  return if r.listLit?.isSome then r else e

/-- A computed typestate, with its equation to the term it is computed from:
    the kernel checks it, where a check here would unfold the term fact by
    fact. -/
meta def stateOutHint (e : Expr) : MetaM (Expr × Expr) := do
  let r ← stateOut e
  return (r, ← mkExpectedTypeHint (← mkEqRefl r) (← mkEq e r))

/-- A list of facts found within another, both computed out: the proof walks
    the two lists, which the kernel checks. -/
elab "sublist_lits" : tactic => do
  let g ← getMainGoal
  let T ← instantiateMVars (← g.getType)
  unless T.isAppOfArity ``List.Sublist 3 do throwError "sublist_lits: not a sublist"
  let some a ← kernelList (T.getArg! 1) | throwError "sublist_lits: no list"
  let some b ← kernelList (T.getArg! 2) | throwError "sublist_lits: no list"
  let some (α, xs) := a.listLit? | throwError "sublist_lits: no list"
  let some (_, ys) := b.listLit? | throwError "sublist_lits: no list"
  let rec go : List Expr → List Expr → MetaM Expr
    | [], [] => mkAppOptM ``List.Sublist.slnil #[α]
    | x :: xs, y :: ys => do
        if x == y then mkAppM ``List.Sublist.cons₂ #[y, ← go xs ys]
        else mkAppM ``List.Sublist.cons #[y, ← go (x :: xs) ys]
    | [], y :: ys => do mkAppM ``List.Sublist.cons #[y, ← go [] ys]
    | _ :: _, [] => throwError "sublist_lits: not found in order"
  g.assign (← mkExpectedTypeHint (← go xs ys) T)

/-- A call's `Moves` obligation into the facts it is computed to keep. -/
macro "prog_keeps_sub" : tactic => `(tactic| (
  refine moves_state_sub _ _ _ (by kdecide) (by decide) (by bits_rfl) (by kernel_rfl) (by sublist_lits)
    (by kdecide) ?_
  pre_state))

/-- A closed expression computed out, once per condition generator run. -/
meta def VCKnow.out (k : VCKnow) (e : Expr) : MetaM (Expr × VCKnow) := do
  let e ← instantiateMVars e
  if let some r := k.outs[e]? then return (r, k)
  let r ← stateOut e
  return (r, { k with outs := k.outs.insert e r })

/-- `s.f` where `s` reduces to a structure written out: its field `f`, as written. -/
meta def projField? (a : Expr) : MetaM (Option Expr) := do
  match a with
  | .proj _ i s =>
      let s ← whnfD s
      let some c ← isConstructorApp? s | return none
      return s.getAppArgs[c.numParams + i]?
  | _ =>
      let .const f _ := a.getAppFn | return none
      let some info ← getProjectionFnInfo? f | return none
      unless a.getAppNumArgs == info.numParams + 1 do return none
      let s ← whnfD a.appArg!
      let some c ← isConstructorApp? s | return none
      return s.getAppArgs[c.numParams + info.i]?

/-- A value list, its head put as a constructor: through fields of structures
    written out, but no further. -/
meta partial def valsWhnf (v : Expr) : MetaM Expr := do
  let v ← whnfR v
  if v.isAppOf ``Vals.cons || v.isAppOf ``Vals.nil then return v
  if let some f ← projField? v then return ← valsWhnf f
  let v' ← whnfCore v
  if v' != v then return ← valsWhnf v'
  return v

/-- The slot at place `i` of a value list written out. -/
meta def valsSlotAt? (v : Expr) (i : Nat) : MetaM (Option Expr) := do
  let mut v ← valsWhnf v
  for _ in [0:i] do
    unless v.isAppOfArity ``Vals.cons 5 do return none
    v ← valsWhnf v.appArg!
  unless v.isAppOfArity ``Vals.cons 5 do return none
  return some (v.getArg! 3)

/-- A slot number, computed out when it names no variable, so a slot read by
    the program and the same slot named by a fact are found as one. -/
meta def normSlot (s : Expr) : MetaM Expr := do
  -- the slot at a place of a value list written out: that slot
  if s.isAppOfArity ``Option.getD 3 then
    let o := s.getArg! 1
    if o.isAppOfArity ``GetElem?.getElem? 7 && (o.getArg! 5).isAppOf ``Vals.slots then
      if let some i ← (evalNat (← instantiateMVars (o.getArg! 6))).run then
        if let some sl ← valsSlotAt? (o.getArg! 5).appArg! i then return ← normSlot sl
  -- An answer's slot, as a call's continuation names it.
  if s.isAppOfArity ``resSlot 2 then
    if (← evalOut (s.getArg! 0)).isAppOf ``Option.some then return ← normSlot (s.getArg! 1)
  -- a field of a structure of slots a body built: the slot it names
  if let some c := s.getAppFn.constName? then
    if (← getProjectionFnInfo? c).isSome then
      let s' ← withReducible (whnf s)
      if s' != s then return ← normSlot s'
  -- a loop's carry, as the body names it: its place past the first
  if s.find? (fun e => e.isAppOf ``carriesFrom) |>.isSome then
    if let some c := s.getAppFn.constName? then
      unless c == ``HAdd.hAdd do
        let s' ← whnf s
        if s' != s then return ← normSlot s'
  -- a place past a slot not yet known: that slot, and a number past it
  if s.hasFVar || s.hasMVar then
    let rec split (e : Expr) (fuel : Nat) : MetaM (Expr × Nat) := do
      if fuel == 0 then return (e, 0)
      let e := e.consumeMData
      if e.isAppOfArity ``HAdd.hAdd 6 then
        if let some n := (e.getArg! 5).nat? then
          let (b, k) ← split (e.getArg! 4) (fuel - 1); return (b, k + n)
        if let some n := (e.getArg! 5).rawNatLit? then
          let (b, k) ← split (e.getArg! 4) (fuel - 1); return (b, k + n)
      if e.isAppOfArity ``Nat.succ 1 then
        let (b, k) ← split e.appArg! (fuel - 1); return (b, k + 1)
      if e.isAppOfArity ``Nat.add 2 then
        if let some n := (e.getArg! 1).nat? <|> (e.getArg! 1).rawNatLit? then
          let (b, k) ← split (e.getArg! 0) (fuel - 1); return (b, k + n)
      return (e, 0)
    let (b, k) ← split s 64
    if k == 0 then return b
    return ← mkAppM ``HAdd.hAdd #[b, mkNatLit k]
  let r ← evalOut s
  match ← (evalNat r).run with
  | some n => return mkRawNatLit n
  | none => return s

/-- The bits of a list of argument values, when each is a known scalar. -/
meta def bitsOf? (vs : Expr) : MetaM (Option Expr) := do
  -- a term over answers is kept whole: computing it out unfolds without end
  if let some (_, cs) := vs.listLit? then
    if cs.any (·.hasFVar) then
      let mut bs := #[]
      for c in cs do
        let c ← whnf c
        unless c.isAppOfArity ``V.sc 2 do return none
        let b := closeLengths (c.getArg! 1)
        bs := bs.push (← if b.hasFVar then pure b else evalOut b)
      return some (← mkListLit (mkConst ``UInt64) bs.toList)
  let r ← evalOut (← mkAppM ``List.mapM #[mkConst ``asBits, vs])
  if r.isAppOfArity ``Option.some 2 then return some r.appArg! else return none

/-- A value known to be a scalar: its type and bits. -/
meta def scalarOf? (c : Expr) : MetaM (Option (Expr × Expr)) := do
  let c ← whnf c
  unless c.isAppOfArity ``V.sc 2 do return none
  let b := closeLengths (c.getArg! 1)
  return some (c.getArg! 0, ← if b.hasFVar then pure b else evalOut b)

/-- A scalar's type and bits, the bits computed only when they are closed:
    a term over answers stays as it is. -/
meta def scalarTerm? (c : Expr) : MetaM (Option (Expr × Expr)) := do
  let c ← whnf c
  unless c.isAppOfArity ``V.sc 2 do return none
  let b := closeLengths (c.getArg! 1)
  let b ← if b.hasFVar then pure b else evalOut b
  return some (← whnf (c.getArg! 0), b)

/-- Record what a new hypothesis says: an `Ext`, a slot's value, or the
    typestate of a world. -/
meta def VCKnow.learn (k : VCKnow) (W : Expr) (h : FVarId) : MetaM VCKnow := do
  let T ← instantiateMVars (← h.getType)
  if T.isAppOfArity ``Ext 2 && T.appArg!.isFVar then
    return { k with parent := k.parent.insert T.appArg!.fvarId! (T.appFn!.appArg!, mkFVar h) }
  if let some (_, lhs, rhs) := T.eq? then
    if lhs.isAppOfArity ``GetElem?.getElem? 7 && rhs.isAppOfArity ``Option.some 2 then
      return (k.addFact (← normSlot (lhs.getArg! 6)) (lhs.getArg! 5, rhs.appArg!, mkFVar h))
    -- The environment a body starts from: each argument at its slot.
    if lhs.isFVar && (← whnfR (← inferType lhs)).isAppOfArity ``Array 1 then
      let arr ← whnf rhs
      if arr.isAppOfArity ``List.toArray 2 || arr.isAppOfArity ``Array.mk 2 then
        if let some xs := (← evalOut arr.appArg!).listLit? then
          let mut k := k
          for x in xs.2, i in [0:xs.2.length] do
            let hi ← mkEqRefl (← mkAppM ``Option.some #[x])
            let pf ← mkAppOptM ``entry_get #[lhs, rhs, mkRawNatLit i, x, mkFVar h, hi]
            k := (k.addFact (mkRawNatLit i) (lhs, x, pf))
          return k
  if T.isAppOfArity ``TyV 2 then
    return { k with tys := k.tys.insert T.appArg! (T.appFn!.appArg!, mkFVar h) }
  if T.isApp && T.appArg!.isFVar && (T.appFn! == W || (← stateOf? T).isSome) then
    return { k with holds := k.holds.insert T.appArg!.fvarId! (mkFVar h) }
  return k

/-- A value's type and a proof `TyV t v`: a scalar's by its constructor, a
    vector's as it was learned. -/
meta def VCKnow.tyV? (k : VCKnow) (c : Expr) : MetaM (Option (Expr × Expr)) := do
  if let some pr := k.tys[c]? then return some pr
  let c' ← whnfR c
  -- a vector written out: its lanes counted
  if c'.isAppOfArity ``V.vec 2 && !c'.hasFVar then
    let t ← whnf (c'.getArg! 0)
    let l ← whnfD (← mkAppM ``ClifTy.lanes #[t])
    unless l.isAppOfArity ``Option.some 2 do return none
    let hl ← mkExpectedTypeHint (← mkEqRefl l) (← mkEq (← mkAppM ``ClifTy.lanes #[t]) l)
    let n := (← whnf l.appArg!).getArg! 3
    let sz ← evalOut (← mkAppM ``Array.size #[c'.getArg! 1])
    unless ← isDefEq sz n do return none
    let hn ← mkExpectedTypeHint (← mkEqRefl n) (← mkEq (← mkAppM ``Array.size #[c'.getArg! 1]) n)
    return some (t, ← mkAppM ``TyV.vec #[hl, hn])
  -- a scalar, its constructor found as `scalarTerm?` finds it; the proof
  -- names the value as it was given
  let c' ← if c'.isAppOfArity ``V.sc 2 then pure c' else whnf c
  unless c'.isAppOfArity ``V.sc 2 do return none
  let t ← whnf (c'.getArg! 0)
  unless t.isConst do return none
  let none_ ← mkAppOptM ``Option.none #[← mkAppM ``Prod #[mkConst ``ClifTy, mkConst ``Nat]]
  unless (← whnfD (← mkAppM ``ClifTy.lanes #[t])).isAppOf ``Option.none do return none
  let hl ← mkExpectedTypeHint (← mkEqRefl none_) (← mkEq (← mkAppM ``ClifTy.lanes #[t]) none_)
  return some (t, ← mkExpectedTypeHint (← mkAppOptM ``TyV.sc #[t, c'.getArg! 1, hl]) (← mkAppM ``TyV #[t, c]))

/-- A slot's value in an environment: the fact where it was last known, carried
    forward along `Ext`, and remembered where it is found. -/
meta def VCKnow.lookup (k : VCKnow) (s env : Expr) : MetaM (Option (Expr × Expr × VCKnow)) := do
  let s ← normSlot s
  let some top := k.facts[s]? | return none
  -- the fact as last found, then as found before: a lookup moves the fact to
  -- where it was looked up, and an environment beside that one finds it
  -- among the older ones
  for (e0, c, pf0) in top :: (k.older[s]?.getD []) do
    let mut chain := #[]
    let mut cur := env
    let mut ok := true
    while cur != e0 do
      unless cur.isFVar do ok := false; break
      let some (prev, h) := k.parent[cur.fvarId!]? | ok := false; break
      chain := chain.push (prev, cur, h)
      cur := prev
    unless ok do continue
    let mut pf := pf0
    for (prev, nxt, h) in chain.reverse do
      pf := mkAppN (mkConst ``Ext.fact) #[prev, nxt, s, c, h, pf]
    if chain.isEmpty then return some (c, pf, k)
    return some (c, pf, k.addFact s (env, c, pf))
  return none

/-- That `env` keeps `root`: the `Ext` steps between them, composed. -/
meta def VCKnow.extProof (k : VCKnow) (root env : Expr) : MetaM (Option Expr) := do
  let mut pf ← mkAppM ``Ext.refl #[env]
  let mut cur := env
  while cur != root do
    unless cur.isFVar do return none
    let some (prev, h) := k.parent[cur.fvarId!]? | return none
    pf ← mkAppM ``Ext.trans #[h, pf]
    cur := prev
  return some pf

/-- How a proposition is decided: an equation of booleans by `Bool`'s own
    equality, the usual goal, since a search for the instance walks the whole
    term, a typestate of a hundred facts in it; anything else by the search. -/
meta def decInst (p : Expr) : MetaM Expr :=
  match p.eq? with
  | some (ty, a, b) =>
      if ty.isConstOf ``Bool then pure (mkApp2 (mkConst ``instDecidableEqBool) a b)
      else synthInstance (mkApp (mkConst ``Decidable) p)
  | none => synthInstance (mkApp (mkConst ``Decidable) p)

/-- A proof of a closed decidable proposition, when the kernel computes it to
    true. The kernel, not `Meta.reduce`, decides: it computes with numbers as
    numbers, where unfolding an instance for 64-bit equality does not end. -/
meta def decideTrue? (p : Expr) : MetaM (Option Expr) := do
  let p ← instantiateMVars p
  if p.hasMVar then return none
  let inst ← decInst p
  let d := mkApp2 (mkConst ``Decidable.decide) p inst
  match Kernel.isDefEq (← getEnv) (← getLCtx) d (mkConst ``Bool.true) with
  | .ok true => return some (mkApp3 (mkConst ``of_decide_eq_true) p inst (← mkEqRefl (mkConst ``Bool.true)))
  | _ => return none

/-- A goal that the post allows a fault, `J.faultOk = true`: by computing it
    or from the context. -/
meta partial def closeFaultOk (m : MVarId) (t : Expr) (fuel : Nat := 8) : MetaM Bool := do
  let some (_, a, b) := t.eq? | return false
  unless a.isAppOfArity ``Post.faultOk 1 && b.isConstOf ``Bool.true do return false
  -- a loop's post allows a fault exactly when the post around the loop does:
  -- that one asked, never the loop's, which holds the rest of the body
  let mut Jo := a.appArg!
  for _ in [0:64] do
    let J' ← whnfR Jo
    unless J'.isAppOf ``loopJ do break
    Jo := J'.getAppArgs[0]!
  if Jo != a.appArg! then
    let a' ← mkAppM ``Post.faultOk #[Jo]
    let t' ← mkEq a' b
    let m' ← mkFreshExprMVar t'
    unless ← closeFaultOk m'.mvarId! t' fuel do return false
    m.assign (← mkExpectedTypeHint (← instantiateMVars m') t)
    return true
  -- a post written out: its flag, computed
  let J := a.appArg!
  if !J.hasFVar && !J.hasMVar then
    let v ← whnfD a
    if v.isConstOf ``Bool.true then m.assign (← mkEqRefl b); return true
    if v.isConstOf ``Bool.false then return false
  if ← isDefEq a b then
    m.assign (← mkEqRefl b); return true
  -- a hypothesis that says so, found by its shape
  let found ← m.withContext do
    for d in ← getLCtx do
      if d.isImplementationDetail then continue
      let some (_, l, r) := d.type.eq? | continue
      unless r.isConstOf ``Bool.true && l.isAppOfArity ``Post.faultOk 1 do continue
      if l == a then return some d.toExpr
    return none
  if let some h := found then m.assign h; return true
  if fuel == 0 then return false
  -- a loop's body: its post allows a fault exactly when the loop's does
  m.withContext do
  for d in ← getLCtx do
    if d.isImplementationDetail then continue
    let some (_, l, r) := (← instantiateMVars d.type).eq? | continue
    unless l.isAppOfArity ``Post.faultOk 1 && r.isAppOfArity ``Post.faultOk 1 do continue
    unless ← isDefEq l a do continue
    let t' ← mkEq r b
    let m' ← mkFreshExprMVar t'
    if ← closeFaultOk m'.mvarId! t' (fuel - 1) then
      m.assign (← mkEqTrans d.toExpr m'); return true
  return false

/-- The value a top-tested loop's head hands its body, where it is one scalar
    slot: its type. -/
meta def loopHandsSlot? (p : Expr) : Option Expr :=
  let args := p.getAppArgs
  if args.size < 6 then none else
  let β := args[args.size - 6]!
  if β.isAppOfArity ``Slot 1 then some β.appArg! else none

/-- The typed rule for a top-tested loop: the one telling the body the type
    of a value its head hands it, where there is one. -/
meta def loopVcTFor (p : Expr) : Name :=
  if (loopHandsSlot? p).isSome then ``wp_loop_vcTA else ``wp_loop_vcT

/-- Whether the post allows a fault where goal `g` stands: by computing it, or
    from `g`'s context. Nothing is assigned. -/
meta def faultAllowed (g : MVarId) (J : Expr) : MetaM Bool := g.withContext do
  -- a first pass that only looks at what a loop's trip computes
  if (← getOptions).getBool `vcProbe false then return true
  let l ← mkEq (← mkAppM ``Post.faultOk #[J]) (mkConst ``Bool.true)
  let m ← mkFreshExprMVar l
  let st ← saveState
  let ok ← closeFaultOk m.mvarId! l
  st.restore
  return ok

/-- A fault goal a rule leaves, where the post allows one: `J.faultOk = true`,
    or the left of `J.faultOk = true ∨ _`; or the right, where it is a closed
    `b = true` the kernel computes. `false` otherwise. -/
meta def closeFault (m : MVarId) : MetaM Bool := do
  let t ← instantiateMVars (← m.getType)
  if ← closeFaultOk m t then return true
  unless t.isAppOfArity ``Or 2 do return false
  let l := t.getArg! 0
  let r := t.getArg! 1
  let ml ← mkFreshExprMVar l
  if ← closeFaultOk ml.mvarId! l then
    m.assign (mkApp3 (mkConst ``Or.inl) l r (← instantiateMVars ml)); return true
  -- or the right, a closed test of a type the kernel computes
  let some (ty, _, b) := (← instantiateMVars r).eq? | return false
  unless ty.isConstOf ``Bool && b.isConstOf ``Bool.true do return false
  let some pr ← decideTrue? r | return false
  m.assign (mkApp3 (mkConst ``Or.inr) l r pr); return true

/-- Apply a lemma to a goal, giving some of its arguments by binder name; the
    arguments neither given nor fixed by the goal are the new goals. -/
meta def applyNamed (g : MVarId) (lem : Name) (given : List (Name × Expr)) : MetaM (List MVarId) := do
  let c ← mkConstWithFreshMVarLevels lem
  let ty ← inferType c
  let (ms, _, concl) ← forallMetaTelescope ty
  let mut names := #[]
  let mut t := ty
  while t.isForall do
    names := names.push t.bindingName!
    t := t.bindingBody!
  unless ← isDefEq concl (← g.getType) do
    throwError "prog_vc: {lem} does not fit{indentExpr (← g.getType)}"
  for (n, v) in given do
    let some i := names.idxOf? n | throwError "prog_vc: {lem} has no argument {n}"
    unless ← isDefEq ms[i]! v do
      throwError "prog_vc: argument {n} of {lem} does not fit{indentExpr v}"
  g.assign (mkAppN c ms)
  -- that a fault is allowed, or cannot happen: closed here, where the post
  -- and the typestate are known, so no caller counts it among the rule's goals
  ms.toList.map (·.mvarId!) |>.filterM fun m => do
    if ← m.isAssigned then return false
    if ← closeFault m then return false
    -- a fault goal left open names the rule that asked it
    let t ← instantiateMVars (← m.getType)
    let isFault := (t.eq?.any fun (_, a, _) => a.isAppOfArity ``Post.faultOk 1) ||
      (t.isAppOfArity ``Or 2 && ((t.getArg! 0).eq?.any fun (_, a, _) => a.isAppOfArity ``Post.faultOk 1))
    if isFault then m.setTag (Name.mkSimple s!"fault@{lem.toString}")
    return true

elab_rules : tactic
  | `(tactic| kdecide) => do
    let g ← getMainGoal
    g.withContext do
    let p ← instantiateMVars (← g.getType)
    let inst ← decInst p
    let d := mkApp2 (mkConst ``Decidable.decide) p inst
    let ok ← ofExceptKernelException (Kernel.isDefEq (← getEnv) (← getLCtx) d (mkConst ``Bool.true))
    unless ok do throwError "kdecide: the kernel computes false"
    g.assign (mkApp3 (mkConst ``of_decide_eq_true) p inst (← mkEqRefl (mkConst ``Bool.true)))

/-- The hypothesis that a typestate holds, the last one in scope, and its
    list of facts. -/
meta def stateHyp? : MetaM (Option (LocalDecl × Expr)) := do
  -- from the latest back: the walk forward instantiated every hypothesis, and a
  -- body's context grows by a few with each operation
  (← getLCtx).findDeclRevM? fun d => do
    if d.isImplementationDetail then return none
    return (← stateOf? (← instantiateMVars d.type)).map (d, ·)

elab_rules : tactic
  | `(tactic| room_arg) => do
    let g ← getMainGoal
    g.withContext do
    let t ← instantiateMVars (← g.getType)
    let isFits := t.isAppOfArity ``Fits 3
    let isRead := t.isAppOfArity ``Readable 3
    unless isFits || isRead || t.isAppOfArity ``Writable 3 do throwError "room_arg: not a read or a write"
    let a := t.getArg! 1
    let some (d, S) ← stateHyp? | throwError "room_arg: no typestate"
    -- an address known outright, or a value past one
    -- an address over values not known is not computed: the term would unfold
    let dec ← if a.hasFVar then pure (mkConst ``Option.none) else evalOut (← mkAppM ``decodeAddr #[a])
    let (lem, dec, extra) ← if dec.isAppOfArity ``Option.some 2 then
        pure (if isFits then ``fits_of_roomArg else if isRead then ``Contracts.readable_of_roomArg
          else ``Contracts.writable_of_roomArg, dec, [])
      else if a.isAppOfArity ``HAdd.hAdd 6 && !(a.getArg! 4).hasFVar then do
        let dec ← evalOut (← mkAppM ``decodeAddr #[a.getArg! 4])
        pure (if isFits then ``fits_of_roomArg_add else if isRead then ``Contracts.readable_of_roomArg_add
          else ``Contracts.writable_of_roomArg_add, dec, [(`b, a.getArg! 4), (`x, a.getArg! 5)])
      else pure (if isFits then ``fits_of_roomArg else ``Contracts.readable_of_roomArg, dec, [])
    unless dec.isAppOfArity ``Option.some 2 do throwError "room_arg: the address decodes nowhere"
    let pr := dec.appArg!
    let r := pr.getArg! 2
    let off := pr.getArg! 3
    let some facts ← kernelList S | throwError "room_arg: no list of facts"
    let mut x? := none
    let mut room? := none
    let mut l := facts
    while l.isAppOfArity ``List.cons 3 do
      let f := l.getArg! 1
      if f.isAppOfArity ``Contracts.Fact.roomArg 2 then
        if ← isDefEq (f.getArg! 0) r then x? := some (f.getArg! 1)
      if f.isAppOfArity ``Contracts.Fact.room 2 then
        if ← isDefEq (f.getArg! 0) r then room? := some (f.getArg! 1)
      l := l.getArg! 2
    -- a value past an address inside a room of known size
    let args ← match x?, room? with
      | some x, _ => pure (lem, [(`hS, d.toExpr), (`r, r), (`off, off), (if extra.isEmpty then `x else `X, x)] ++ extra)
      | none, some N =>
        unless isFits && !extra.isEmpty do throwError "room_arg: no length names the region"
        pure (``fits_of_room_add, [(`hS, d.toExpr), (`r, r), (`off, off), (`N, N)] ++ extra)
      | none, none => throwError "room_arg: no length names the region"
    let gs ← applyNamed g args.1 args.2
    -- a room's size computed by a term: its number, for the arithmetic
    let sizeLit? : Option (Expr × Expr) ← match room? with
      | some N => if N.hasFVar || N.hasMVar || N.rawNatLit?.isSome || N.nat?.isSome then pure none else
          match (← evalOut N).rawNatLit? with
          | some v => pure (some (N, mkNatLit v))
          | none => pure none
      | none => pure none
    for g' in gs do
      let ty ← instantiateMVars (← g'.getType)
      let g' ← match sizeLit? with
        | some (N, v) =>
            if ty.isAppOfArity ``LE.le 4 || ty.isAppOfArity ``LT.lt 4 then
              g'.replaceTargetDefEq (ty.replace fun e => if e == N then some v else none)
            else pure g'
        | none => pure g'
      let tac ← if ty.isAppOfArity ``Membership.mem 5 then `(tactic| first | simp | kdecide)
        else if ty.isAppOfArity ``LE.le 4 || ty.isAppOfArity ``LT.lt 4 then `(tactic| bound_close)
        else `(tactic| kdecide)
      let rest ← Tactic.run g' (withoutRecover <| evalTactic tac)
      unless rest.isEmpty do throwError "room_arg: {ty} left"

/-- Bytes at an address past a region's base by a bound the context gives:
    a length the program was handed for the region, or its room, covers them. -/
elab "room_sub" : tactic => do
  let g ← getMainGoal
  g.withContext do
  let t ← instantiateMVars (← g.getType)
  unless t.isAppOfArity ``Fits 3 do throwError "room_sub: not a fit"
  let some (d, S) ← stateHyp? | throwError "room_sub: no typestate"
  let some facts ← kernelList S | throwError "room_sub: no list of facts"
  -- each region a fact names, with its base as a number
  let mut cands : Array (Name × List (Name × Expr) × Expr × Nat) := #[]
  let mut l := facts
  while l.isAppOfArity ``List.cons 3 do
    let f := l.getArg! 1
    l := l.getArg! 2
    let (lem, args) ← if f.isAppOfArity ``Contracts.Fact.roomArg 2 then
        pure (``fits_of_roomArg_sub, [(`hS, d.toExpr), (`r, f.getArg! 0), (`X, f.getArg! 1)])
      else if f.isAppOfArity ``Contracts.Fact.room 2 then
        pure (``fits_of_room_sub, [(`hS, d.toExpr), (`r, f.getArg! 0), (`N, f.getArg! 1)])
      else continue
    let rb ← mkAppM ``UInt64.toNat #[← mkAppM ``regionBase #[f.getArg! 0]]
    let some v := (← evalOut rb).rawNatLit? | continue
    cands := cands.push (lem, args, rb, v)
  -- the regions whose base the context names first: the address is past it
  -- (a walk over every hypothesis: only where there is an order to find)
  let namedR ← IO.mkRef (#[] : Array Nat)
  if cands.size > 1 then
    for dcl in ← getLCtx do
      if dcl.isImplementationDetail then continue
      (← instantiateMVars dcl.type).forEach fun e => do
        if let some v := e.rawNatLit? then if v ≥ 2 ^ 32 then namedR.modify (·.push v)
  let named ← namedR.get
  let order := cands.filter (fun c => named.contains c.2.2.2) ++ cands.filter (fun c => !named.contains c.2.2.2)
  for (lem, args, rb, v) in order do
    let st ← saveState
    let ok ← try
        let gs ← applyNamed g lem args
        let mut ok := true
        for g' in gs do
          let ty ← instantiateMVars (← g'.getType)
          -- the base as its number, for the arithmetic
          let ty' := ty.replace fun e => if e == rb then some (mkNatLit v) else none
          -- and a room's size computed by a term, as its number
          let ty' ← match args.lookup `N with
            | some N => if N.hasFVar || N.rawNatLit?.isSome then pure ty' else
                match (← evalOut N).rawNatLit? with
                | some n => pure (ty'.replace fun e => if e == N then some (mkNatLit n) else none)
                | none => pure ty'
            | none => pure ty'
          let g' ← if ty' != ty then g'.replaceTargetDefEq ty' else pure g'
          let tac ← if ty.isAppOfArity ``Membership.mem 5 then `(tactic| first | simp | kdecide)
            else if ty.isAppOfArity ``LE.le 4 || ty.isAppOfArity ``LT.lt 4 then `(tactic| bound_close)
            else `(tactic| kdecide)
          let rest ← Tactic.run g' (withoutRecover (evalTactic tac))
          unless rest.isEmpty do ok := false; break
        pure ok
      catch _ => pure false
    if ok then return
    st.restore
  throwError "room_sub: no room covers it"

/-- Bytes a load or store reaches are there: a `room` fact covers them, or a
    length the program was handed does. -/
elab "fits_close" : tactic => do
  let g ← getMainGoal
  g.withContext do
  let t ← instantiateMVars (← g.getType)
  unless t.isAppOfArity ``Fits 3 do throwError "fits_close: not a fit"
  -- the width as a number, for the arithmetic
  let t := mkApp3 (mkConst ``Fits) (t.getArg! 0) (t.getArg! 1) (← evalOut (t.getArg! 2))
  let g ← g.replaceTargetDefEq t
  replaceMainGoal [g]
  let some (d, S) ← stateHyp? | throwError "fits_close: no typestate"
  -- by the kernel only at an address it can compute: over a variable it
  -- unfolds the arithmetic without end
  if !(t.getArg! 1).hasFVar then
    if let some pr ← decideTrue? (← mkEq (← mkAppM ``Contracts.roomAt #[S, t.getArg! 1, t.getArg! 2])
        (mkConst ``Bool.true)) then
      let pf ← mkAppM ``fits_of_state #[d.toExpr, pr]
      if ← isDefEq (← inferType pf) t then
        g.assign pf; return
  evalTactic (← `(tactic| first | room_arg | room_sub))

/-- A goal `Fits _ _ _`, or `J.faultOk = true ∨ Fits _ _ _`, by `fits_close`;
    `false`, with nothing changed, when it does not close. -/
meta def closeFits (m : MVarId) : TacticM Bool := m.withContext do
  let t ← instantiateMVars (← m.getType)
  let (goal, wrap) ← if t.isAppOfArity ``Or 2 && (t.getArg! 1).isAppOfArity ``Fits 3 then do
      let r ← mkFreshExprMVar (t.getArg! 1)
      pure (r.mvarId!, some r)
    else if t.isAppOfArity ``Fits 3 then pure (m, none) else return false
  -- `b + x + c`, the literals apart: `b + c + x`, a literal past a place
  let goal ← do
    let ft ← instantiateMVars (← goal.getType)
    let a := ft.getArg! 1
    if a.isAppOfArity ``HAdd.hAdd 6 && (a.getArg! 4).isAppOfArity ``HAdd.hAdd 6 then
      let c := a.getArg! 5
      let b := (a.getArg! 4).getArg! 4
      let x := (a.getArg! 4).getArg! 5
      if !c.hasFVar && !b.hasFVar && x.hasFVar then
        let ng ← mkFreshExprMVar (← mkAppM ``Fits #[ft.getArg! 0, ← mkAppM ``HAdd.hAdd #[← mkAppM ``HAdd.hAdd #[b, c], x], ft.getArg! 2])
        goal.assign (← mkAppM ``fits_regroup #[ng])
        pure ng.mvarId!
      else pure goal
    else pure goal
  let st ← saveState
  let ok ← tryCatchRuntimeEx (do pure (← Tactic.run goal (withoutRecover <| evalTactic (← `(tactic| fits_close)))).isEmpty)
    (fun _ => pure false)
  unless ok do st.restore; return false
  if let some r := wrap then
    m.assign (mkApp3 (mkConst ``Or.inr) (t.getArg! 0) (t.getArg! 1) (← instantiateMVars r))
  return true

/-- A proof that the first element of a written-out list that `p` picks is
    in it, built by its place. -/
meta def memLit? (α : Expr) (xs : List Expr) (p : Expr → MetaM Bool) : MetaM (Option Expr) := do
  let rec go : List Expr → MetaM (Option Expr)
    | [] => pure none
    | x :: rest => do
        if ← p x then return some (← mkAppOptM ``List.Mem.head #[α, x, ← mkListLit α rest])
        let some h ← go rest | return none
        return some (← mkAppM ``List.Mem.tail #[x, h])
  go xs

elab_rules : tactic
  | `(tactic| fact_mem) => do
    let g ← getMainGoal
    g.withContext do
    let t ← instantiateMVars (← g.getType)
    unless t.isAppOfArity ``Membership.mem 5 do throwError "fact_mem: not a membership"
    let S := t.getArg! 3
    let x := t.getArg! 4
    let some Sv ← kernelList S | throwError "fact_mem: no list of facts"
    let some (α, facts) := Sv.listLit? | throwError "fact_mem: no list of facts"
    let some pf ← memLit? α facts (fun f => isDefEq f x) | throwError "fact_mem: {x} not found"
    g.assign (← mkExpectedTypeHint pf (← instantiateMVars t))

elab_rules : tactic
  | `(tactic| readable_bound) => do
    let g ← getMainGoal
    g.withContext do
    let t ← instantiateMVars (← g.getType)
    let lem ← if t.isAppOfArity ``Readable 3 then pure ``Contracts.readable_upto_of_state
      else if t.isAppOfArity ``Writable 3 then pure ``writable_upto_of_state
      else throwError "readable_bound: not a read or a write"
    let a := t.getArg! 1
    let some (d, S) ← stateHyp? | throwError "readable_bound: no typestate"
    let K ← evalOut (← mkAppM ``Contracts.roomLeft #[S, a])
    let gs ← applyNamed g lem [(`hS, d.toExpr), (`K, K)]
    for g' in gs do
      let ty ← instantiateMVars (← g'.getType)
      let tac ← if ty.isAppOfArity ``Eq 3 then `(tactic| kdecide) else `(tactic| bound_close)
      let rest ← Tactic.run g' (withoutRecover <| evalTactic tac)
      unless rest.isEmpty do throwError "readable_bound: {ty} left"

elab_rules : tactic
  | `(tactic| room_add) => do
    let g ← getMainGoal
    g.withContext do
    let t ← instantiateMVars (← g.getType)
    let isRead := t.isAppOfArity ``Readable 3
    unless isRead || t.isAppOfArity ``Writable 3 do throwError "room_add: not a read or a write"
    let a := t.getArg! 1
    let n := t.getArg! 2
    unless a.isAppOfArity ``HAdd.hAdd 6 do throwError "room_add: not an address past another"
    let b := a.getArg! 4
    if b.hasFVar then throwError "room_add: the base is not known"
    let some (d, S) ← stateHyp? | throwError "room_add: no typestate"
    let dec ← evalOut (← mkAppM ``decodeAddr #[b])
    unless dec.isAppOfArity ``Option.some 2 do throwError "room_add: the base decodes nowhere"
    let pr := dec.appArg!
    let r := pr.getArg! 2
    let off := pr.getArg! 3
    let some facts ← kernelList S | throwError "room_add: no list of facts"
    let mut K? := none
    let mut l := facts
    while l.isAppOfArity ``List.cons 3 do
      let f := l.getArg! 1
      if f.isAppOfArity ``Contracts.Fact.room 2 then
        if ← isDefEq (f.getArg! 0) r then K? := some (f.getArg! 1)
      l := l.getArg! 2
    let some K := K? | throwError "room_add: no room names the region"
    let hroom ← mkFreshExprMVar (← mkEq (← mkAppM ``Contracts.roomAt #[S, a, n]) (mkConst ``Bool.true))
    let gs ← applyNamed hroom.mvarId! ``roomAt_add [(`K, K), (`r, r), (`off, off)]
    for g' in gs do
      let ty ← instantiateMVars (← g'.getType)
      let tac ← if ty.isAppOfArity ``Membership.mem 5 then `(tactic| fact_mem)
        else if ty.isAppOfArity ``Eq 3 then `(tactic| kdecide)
        else if ty.isAppOfArity ``Not 1 || ty.isAppOfArity ``Ne 3 then `(tactic| decide)
        else `(tactic| bound_close)
      let rest ← Tactic.run g' (withoutRecover <| evalTactic tac)
      unless rest.isEmpty do throwError "room_add: {ty} left"
    let lem := if isRead then ``Contracts.readable_of_state else ``Contracts.writable_of_state
    g.assign (← mkAppM lem #[d.toExpr, ← instantiateMVars hroom])

elab_rules : tactic
  | `(tactic| pinned_room) => do
    let g ← getMainGoal
    g.withContext do
    let some (d, S) ← stateHyp? | throwError "pinned_room: no typestate"
    let some facts ← kernelList S | throwError "pinned_room: no list of facts"
    let some (_, fs) := facts.listLit? | throwError "pinned_room: no list of facts"
    let some f := fs.find? (·.isAppOfArity ``Contracts.Fact.pinnedUsed 1) | throwError "pinned_room: no bound"
    let gs ← applyNamed g ``pinnedRoom_of_state [(`hS, d.toExpr), (`n, f.appArg!)]
    for g' in gs do
      let ty ← instantiateMVars (← g'.getType)
      let tac ← if ty.isAppOfArity ``Membership.mem 5 then `(tactic| fact_mem) else `(tactic| first | kdecide | bound_close)
      let rest ← Tactic.run g' (withoutRecover <| evalTactic tac)
      unless rest.isEmpty do throwError "pinned_room: {ty} left"

/-- What a weakest-precondition goal is about: its program. -/
meta def wpProg? (e : Expr) : Option Expr :=
  if e.isAppOfArity ``AlgorithmLib.Prog.wp 8 then some (e.getArg! 4) else none

/-- The slots a value list names, as a list of expressions. -/
meta partial def slotList (vs : Expr) : MetaM (Option (List Expr)) := do
  let l ← whnf (← mkAppM ``Vals.slots #[vs])
  go l
where
  go (l : Expr) : MetaM (Option (List Expr)) := do
    let l ← whnf l
    if l.isAppOfArity ``List.nil 1 then return some []
    if l.isAppOfArity ``List.cons 3 then
      let some rest ← go (l.getArg! 2) | return none
      return some (l.getArg! 1 :: rest)
    return none

/-- A call's `Moves` obligation, proved by `prog_keeps` where it depends on
    nothing of the path: closed over the variables it names, in an empty
    context, as a lemma of its own, and remembered, so a call made again with
    the same arguments in the same typestate costs nothing. -/
meta def movesProof (k : VCKnow) (T : Expr) (sub : Bool := false) : TacticM (Option (Expr × VCKnow)) := do
  let T ← instantiateMVars T
  let fvs ← sortFVarIds (collectFVars {} T).fvarIds
  -- what the context bounds of those values goes along, and the comparisons
  -- the branches the call sits in made: a size the call is handed may be
  -- known only by a bound
  let mut hs := #[]
  for d in ← getLCtx do
    if d.isImplementationDetail then continue
    let ty ← instantiateMVars d.type
    let rec isBound (ty : Expr) : Bool :=
      ty.isAppOfArity ``LE.le 4 || ty.isAppOfArity ``LT.lt 4 ||
        (ty.isAppOfArity ``Eq 3 && (ty.getArg! 1).isAppOf ``cmpInt) ||
        (ty.isAppOfArity ``And 2 && isBound (ty.getArg! 0) && isBound (ty.getArg! 1))
    unless isBound ty do continue
    let tfs := (collectFVars {} ty).fvarIds
    if tfs.isEmpty || !tfs.all fvs.contains then continue
    hs := hs.push d.fvarId
  let fvs := fvs ++ hs
  let xs := fvs.map mkFVar
  let Tc ← mkForallFVars xs T
  if let some pf := k.moves[Tc]? then return some (mkAppN pf xs, k)
  let m ← withLCtx {} {} (mkFreshExprMVar Tc)
  let (_, g1) ← m.mvarId!.introN fvs.size
  -- on a budget: an obligation the table's rules cannot meet may otherwise
  -- send a rule computing out a length of millions
  let st ← saveState
  let rest ← tryCatchRuntimeEx
    (withCurrHeartbeats <| withTheReader Core.Context (fun c => { c with maxHeartbeats := 400000000 }) do
      Tactic.run g1 (withoutRecover <| evalTactic (← if sub then `(tactic| prog_keeps_sub) else `(tactic| prog_keeps_fast))))
    (fun _ => do st.restore; pure [g1])
  unless rest.isEmpty do return none
  -- added straight to the kernel: a check here would unfold arithmetic the
  -- kernel computes
  let name ← Term.mkAuxName `prog_vc_moves
  let added ← try addDecl (.thmDecl { name, levelParams := [], type := Tc, value := ← instantiateMVars m }); pure true
    catch _ => pure false
  unless added do
    -- the fast rule left the kernel an equation it cannot compute: check it here
    st.restore
    let m ← withLCtx {} {} (mkFreshExprMVar Tc)
    let (_, g1) ← m.mvarId!.introN fvs.size
    let rest ← tryCatchRuntimeEx
      (withCurrHeartbeats <| withTheReader Core.Context (fun c => { c with maxHeartbeats := 400000000 }) do
        Tactic.run g1 (withoutRecover <| evalTactic (← `(tactic| prog_keeps))))
      (fun _ => do st.restore; pure [g1])
    unless rest.isEmpty do return none
    addDecl (.thmDecl { name, levelParams := [], type := Tc, value := ← instantiateMVars m })
    return some (mkAppN (mkConst name) xs, { k with moves := k.moves.insert Tc (mkConst name) })
  let pf := mkConst name
  return some (mkAppN pf xs, { k with moves := k.moves.insert Tc pf })

/-- Whether a call's frame, or a rule of what it keeps, reads an argument the
    program was answered: an address or a length the kernel would count out
    against a constant one step at a time. -/
meta def framesAnswered (f bits : Expr) : MetaM Bool := do
  let some (_, bs) := bits.listLit? | return bits.hasFVar
  let fr ← evalOut (← mkAppM ``HProg.frame #[f])
  let nat (e : Expr) : MetaM (Option Nat) := do return (← evalOut e).rawNatLit?
  let mut idxs : List Nat := []
  for a in fr.getAppArgs do
    if (← inferType a).isConstOf ``Nat then
      if let some i ← nat a then idxs := i :: idxs
  -- the rules beside the frame: a read by its size, an insertion by its
  -- value's length, an init by its slot
  if f.isConstOf ``IR.Ffi.fileRead then idxs := [0, 2, 4] ++ idxs
  if f.isConstOf ``IR.Ffi.htInsert then idxs := 4 :: idxs
  if f.isConstOf ``IR.Ffi.lmdbInit || f.isConstOf ``IR.Ffi.htInit then idxs := 0 :: idxs
  return idxs.any fun i => (bs[i]?.map (·.hasFVar)).getD false

/-- The facts a call is computed to keep, for a call whose table entry only
    keeps facts: where what it leaves does not compute, as over a length the
    program was answered, each fact whose keeping computes is kept and the rest
    dropped. -/
meta def keptOf? (f bits S : Expr) (blindOnly : Bool := false) : MetaM (Option Expr) := do
  let some Sv ← kernelList S | return none
  let some (α, facts) := Sv.listLit? | return none
  let kf ← mkAppM ``Contracts.keeps #[f, bits]
  let aft := mkApp3 (mkConst ``Contracts.after) f bits S
  let filt ← mkAppM ``List.filter #[kf, S]
  unless (Kernel.isDefEq (← getEnv) (← getLCtx) aft filt) matches .ok true do return none
  -- Over arguments the program was answered, a fact whose keeping reads them
  -- is dropped rather than computed: the kernel would count a length out one
  -- step at a time. The kinds kept read only which call it is.
  let symbolic := blindOnly
  let blind : Name → Bool := fun n =>
    n == ``Contracts.Fact.part || n == ``Contracts.Fact.room || n == ``Contracts.Fact.held ||
    n == ``Contracts.Fact.opened || n == ``Contracts.Fact.devSeq || n == ``Contracts.Fact.oracles ||
    n == ``Contracts.Fact.roomArg || n == ``Contracts.Fact.devBuf || n == ``Contracts.Fact.pinnedUsed ||
    (n == ``Contracts.Fact.htVals && !f.isConstOf ``IR.Ffi.htInsert)
  let mut kept := #[]
  for x in facts do
    if symbolic && !blind (x.getAppFn.constName?.getD .anonymous) then continue
    if (Kernel.isDefEq (← getEnv) (← getLCtx) (mkApp kf x) (mkConst ``Bool.true)) matches .ok true then
      kept := kept.push x
  return some (← mkListLit α kept.toList)

/-- A call by its contract: look up the argument values, discharge `Keeps` with
    `prog_keeps`, and go on after the call. -/
meta def vcCall (W : Expr) (k : VCKnow) (g : MVarId) (t p : Expr) (raw : Bool := false) :
    TacticM (Option (List (MVarId × VCKnow × Nat))) := do
  let cfg := t.getArg! 0
  let env := t.getArg! 6
  let w := t.getArg! 7
  -- `ffiVoid f args`, or `Prog.call f args k` with the answer bound
  let f := p.getArg! (if raw then 3 else 2)
  let args := p.getArg! (if raw then 4 else 3)
  let some ss ← slotList args | return none
  let mut k := k
  let mut cs := #[]
  let mut pfs := #[]
  let mut found := #[]
  for sl in ss do
    let some (c, pf, _) ← k.lookup sl env | return none
    cs := cs.push c
    pfs := pfs.push pf
    unless pf.isFVar do
      found := found.push (sl, c, pfs.size - 1, { userName := `hslot, type := ← inferType pf, value := pf })
  let vTy := mkConst ``V
  let natTy := mkConst ``Nat
  let csE ← mkListLit vTy cs.toList
  unless w.isFVar do return none
  let some hW := k.holds[w.fvarId!]? | return none
  -- A list of facts moves by the contract table; a typestate of the program's
  -- own is kept. The goal is left as it was when the call's obligation fails.
  let (W0, W1, sub) ← do
    match ← stateOf? (← instantiateMVars (← inferType hW)) with
    | some S =>
        let some bits ← bitsOf? csE | return none
        let holds := mkConst ``Contracts.TState.holds
        -- over an address or a length the program was answered, only what the
        -- call keeps without reading it
        if ← framesAnswered f bits then
          let some S'' ← keptOf? f bits S (blindOnly := true) | return none
          pure (mkApp holds S, mkApp holds S'', true)
        else
        let (S', k') ← k.out (mkApp3 (mkConst ``Contracts.after) f bits S)
        k := k'
        if S'.listLit?.isSome then pure (mkApp holds S, mkApp holds S', false)
        else
          let some S'' ← keptOf? f bits S | return none
          pure (mkApp holds S, mkApp holds S'', true)
    | none => pure (W, W, false)
  let some (gK, k') ← movesProof k (← mkAppM ``Moves #[cfg, W0, f, csE, W1]) sub | return none
  k := k'
  -- What was found is bound here, in one group, so the next lookup starts from it.
  let (hs, g) ← g.assertHypotheses (found.map (·.2.2.2))
  for (sl, c, i, _) in found, h in hs do
    pfs := pfs.set! i (mkFVar h)
    k := (k.addFact (← normSlot sl) (env, c, mkFVar h))
  for sl in ss, pf in pfs, c in cs do
    k := (k.addFact (← normSlot sl) (env, c, pf))
  let mut hcs := mkApp (mkConst ``lookup_nil) env
  for i in (List.range ss.length).reverse do
    let rest := ss.drop (i + 1)
    let restCs := cs.toList.drop (i + 1)
    hcs := mkAppN (mkConst ``lookup_cons)
      #[env, ss[i]!, cs[i]!, ← mkListLit natTy rest, ← mkListLit vTy restCs, pfs[i]!, hcs]
  g.withContext do
  let isSome ← whnf (← mkAppM ``Option.isSome #[← mkAppM ``Ffi.result #[f]])
  let spawn := f.isConstOf ``Ffi.threadSpawn
  let (lem, b) := if isSome.isConstOf ``Bool.true then
      (if raw then (if spawn then ``wp_call_res_var else ``wp_call_res_sc) else ``wp_ffiVoid_res_var,
        mkConst ``Bool.true)
    else (if raw then ``wp_call_var else ``wp_ffiVoid_var, mkConst ``Bool.false)
  let mut given := [(`W, W0), (`W', W1), (`hcs, hcs), (`hK, gK), (`hW, hW)]
  if lem == ``wp_call_res_sc then
    -- the answer's type, as the signature gives it
    let res ← whnf (← mkAppM ``Ffi.result #[f])
    unless res.isAppOfArity ``Option.some 2 do return none
    let t0 ← whnf res.appArg!
    let ht ← mkExpectedTypeHint (← mkEqRefl res)
      (← mkEq (← mkAppM ``Ffi.result #[f]) (← mkAppM ``Option.some #[t0]))
    given := given ++ [(`t0, t0), (`ht, ht),
      (`hf, ← mkDecideProof (mkNot (← mkEq f (mkConst ``Ffi.threadSpawn))))]
  else given := (`hres, ← mkEqRefl b) :: given
  -- what the table says of the answer, when the argument bits are known
  let mut lem := lem
  if lem == ``wp_call_res_sc then
    if let some bits ← bitsOf? csE then
      let hbT ← mkEq (← mkAppM ``List.mapM #[mkConst ``asBits, csE]) (← mkAppM ``Option.some #[bits])
      let hb ← mkExpectedTypeHint (← mkEqRefl (← mkAppM ``Option.some #[bits])) hbT
      lem := ``wp_call_res_ans
      given := given ++ [(`bits, bits), (`hb, hb)]
  let gs ← applyNamed g lem given
  return some (gs.map (·, k, hs.size))

/-- Bind slot facts found along the way as hypotheses of the goal, in one
    group, so their proofs are not rebuilt and the next lookup starts here. -/
meta def bindFound (g : MVarId) (k : VCKnow) (env : Expr) (found : Array (Expr × Expr × Expr)) :
    MetaM (MVarId × Array Expr × VCKnow × Nat) := do
  let mut hyps := #[]
  let mut idx := #[]
  for (_, _, pf) in found, i in [0:found.size] do
    unless pf.isFVar do
      hyps := hyps.push { userName := `hslot, type := ← inferType pf, value := pf : Hypothesis }
      idx := idx.push i
  let (hs, g) ← g.assertHypotheses hyps
  let mut pfs := found.map (·.2.2)
  for i in idx, h in hs do
    pfs := pfs.set! i (mkFVar h)
  let mut k := k
  for (sl, c, _) in found, pf in pfs do
    k := (k.addFact (← normSlot sl) (env, c, pf))
  return (g, pfs, k, hs.size)

/-- The typestate a goal's world is known in, as a list of facts, with its proof. -/
meta def stateAt? (k : VCKnow) (w : Expr) : MetaM (Option (Expr × Expr)) := do
  unless w.isFVar do return none
  let some hW := k.holds[w.fvarId!]? | return none
  let some S ← stateOf? (← instantiateMVars (← inferType hW)) | return none
  return some (S, hW)

/-- Whether a body uses a construct whose name holds `word`, read off its
    syntax, the helpers that wrap one included. -/
meta def bodyUses (word : String) (body : Expr) : MetaM Bool := do
  let env ← getEnv
  let isStore : Name → Bool
    | .str _ s => (s.toLower.splitOn word).length > 1
    | _ => false
  -- a definition that builds a program is looked into: a helper stores if
  -- anything it is made of does
  let buildsProg (t : Expr) : Bool := t.getForallBody.getAppFn.isConstOf ``Prog
  let mut seen : NameSet := {}
  let mut todo := #[body]
  let mut fuel := 4000
  while h : todo.size > 0 do
    let e := todo.back
    todo := todo.pop
    for c in e.getUsedConstants do
      if isStore c then return true
      if seen.contains c then continue
      seen := seen.insert c
      if fuel == 0 then return true
      fuel := fuel - 1
      if let some (.defnInfo d) := env.find? c then
        if buildsProg d.type then todo := todo.push d.value
  return false

/-- Whether a loop body stores to memory: any constant whose name says it
    stores. -/
meta def bodyStores (body : Expr) : MetaM Bool := bodyUses "store" body

/-- Whether a body calls out of the program, read off its syntax as
    `bodyStores` reads its stores. -/
meta def bodyCalls (body : Expr) : MetaM Bool := do
  return (← bodyUses "ffi" body) || (← bodyUses "call" body)

/-- A loop's head and body together, for what they store and insert. -/
meta def loopParts (p : Expr) : Expr :=
  let args := p.getAppArgs
  .app args[args.size - 3]! args[args.size - 2]!

/-- The elements of a value list written out. -/
meta partial def valsElems (e : Expr) : List Expr :=
  if e.isAppOfArity ``Vals.cons 5 then e.getArg! 3 :: valsElems (e.getArg! 4) else []

/-- The value lengths a loop body inserts into the hash table, where each is
    a slot known here, holding a constant; `none` when some insert's is not. -/
meta def insertLens (k : VCKnow) (env body : Expr) : MetaM (Option (List Nat)) := do
  let mut lens := []
  let mut rest := body
  let isInsert (x : Expr) : Bool :=
    x.isAppOfArity ``ffiVoid 4 && (x.getArg! 2).isConstOf ``Ffi.htInsert
  while true do
    let some x := rest.find? isInsert | break
    let some sl := (valsElems (x.getArg! 3))[4]? | return none
    if sl.hasLooseBVars then return none
    let some (c, _, _) ← k.lookup sl env | return none
    let some (_, bits) ← scalarOf? c | return none
    let some n ← (evalNat (← evalOut (← mkAppM ``UInt64.toNat #[bits]))).run | return none
    lens := n :: lens
    rest := rest.replace fun y => if isInsert y then some (mkConst ``Unit.unit) else none
  return some lens

/-- A loop's invariant and its proof where the loop is entered: the typestate
    there, less its cells when the body stores, and with its bound on the hash
    table's values widened to cover every value the body inserts. -/
meta def loopState (k : VCKnow) (env W hW body : Expr) (keepCells : Bool := false)
    (keepOnly : Option Expr := none) : MetaM (Expr × Expr) := do
  let T ← instantiateMVars (← inferType hW)
  let body ← instantiateMVars body
  match ← stateOf? T with
  | some S =>
      let holds := mkConst ``Contracts.TState.holds
      let stores ← bodyStores body
      let (S, hW) ← if stores && !keepCells then
          let p ← withLocalDeclD `x (mkConst ``Contracts.Fact) fun x => do
            mkLambdaFVars #[x] (← mkAppM ``not #[← mkAppM ``Contracts.Fact.isCell #[x]])
          let S' ← stateOut (← mkAppM ``List.filter #[p, S])
          pure (S', ← mkExpectedTypeHint (← mkAppM ``TState.holds_filter #[p, hW]) (mkApp2 holds S' T.appArg!))
        else match keepOnly with
          | some L =>
              -- only the cells a trip of the body leaves as they were
              let p ← withLocalDeclD `x (mkConst ``Contracts.Fact) fun x => do
                mkLambdaFVars #[x] (← mkAppM ``or #[← mkAppM ``not #[← mkAppM ``Contracts.Fact.isCell #[x]],
                  ← mkAppM ``List.contains #[L, x]])
              let S' ← stateOut (← mkAppM ``List.filter #[p, S])
              pure (S', ← mkExpectedTypeHint (← mkAppM ``TState.holds_filter #[p, hW]) (mkApp2 holds S' T.appArg!))
          | none => pure (S, hW)
      let inserts := (body.find? fun x => x.isConstOf ``Ffi.htInsert).isSome
      let (S, hW) ← if inserts then
          match ← insertLens k env body with
          | some lens =>
              let b := mkNatLit (lens.foldl max 0)
              let S' ← stateOut (← mkAppM ``List.map #[← mkAppM ``Contracts.Fact.widen #[b], S])
              pure (S', ← mkExpectedTypeHint (← mkAppM ``Contracts.TState.holds_widen #[b, hW])
                (mkApp2 holds S' T.appArg!))
          | none =>
              let p ← withLocalDeclD `x (mkConst ``Contracts.Fact) fun x => do
                mkLambdaFVars #[x] (← mkAppM ``not #[← mkAppM ``Contracts.Fact.isHtVals #[x]])
              let S' ← stateOut (← mkAppM ``List.filter #[p, S])
              pure (S', ← mkExpectedTypeHint (← mkAppM ``TState.holds_filter #[p, hW])
                (mkApp2 holds S' T.appArg!))
        else pure (S, hW)
      if S == (← stateOf? T).getD S && (!stores || keepCells) && keepOnly.isNone && !inserts then
        return (← typestateOf W hW, hW)
      return (mkApp holds S, hW)
  | none => return (W, hW)

/-- `p₁ ∨ p₂ ∨ …`, right-nested. -/
meta def mkListOr : List Expr → MetaM Expr
  | [] => pure (mkConst ``False)
  | [p] => pure p
  | p :: ps => do mkAppM ``Or #[p, ← mkListOr ps]

/-- The value of an operation that reads no memory, when each of its operands'
    values is known: `evalOp_renumber*` turns it into the operation over just
    those values, which computes. An operation whose value does not compute —
    float arithmetic the kernel cannot evaluate, or operands the operation
    refuses — is left unknown. -/
meta def vcOpComputed (k : VCKnow) (g : MVarId) (env w oe : Expr) :
    TacticM (Option (List (MVarId × VCKnow × Nat))) := do
  -- Only `Op.regs` is unfolded: the operands stay as the program wrote them,
  -- which is how the slots they name are found among the known facts.
  let regs ← whnf (← mkAppM ``Op.regs #[oe])
  let some (_, rs) := regs.listLit? | return none
  let lem ← match rs.length with
    | 0 => pure ``evalOp_renumber0
    | 1 => pure ``evalOp_renumber1
    | 2 => pure ``evalOp_renumber2
    | 3 => pure ``evalOp_renumber3
    | _ => return none
  let mut known := k
  let mut found := #[]
  for r in rs do
    let some (x, pf, k') ← known.lookup r env | return none
    known := k'
    found := found.push (r, x, pf)
  let xs := found.toList.map (closeLengths ·.2.1)
  -- Only values written out compute: evaluating over a value the program
  -- was answered can unfold without end. A 64-bit sum over such values is
  -- kept as a term, so bounds on its operands carry to it.
  if xs.any (·.hasFVar) then
    let some opName := oe.getAppFn.constName? | return none
    -- an integer sign-extended to 64 bits: a 64-bit value
    if opName == ``Op.sextend64 then
      let [x0] := xs | return none
      let some (_, _) ← scalarTerm? x0 | return none
      let (g, pfs, k, n) ← bindFound g known env found
      return ← g.withContext do
        let gs ← applyNamed g ``wp_op_sext64 [(`ho, ← mkEqRefl oe), (`ha, pfs[0]!)]
        return some (gs.map (·, k, n))
    -- an integer widened to 64 bits: its bits under the source's mask
    if opName == ``Op.uextend64 then
      let [x0] := xs | return none
      let some (_, _) ← scalarTerm? x0 | return none
      let (g, pfs, k, n) ← bindFound g known env found
      return ← g.withContext do
        let gs ← applyNamed g ``wp_op_uext64 [(`ho, ← mkEqRefl oe), (`ha, pfs[0]!)]
        return some (gs.map (·, k, n))
    -- a 64-bit term cut to 32 bits stays a term: its low bits
    if opName == ``Op.ireduce32 then
      let [x0] := xs | return none
      let some (t0, b0) ← scalarTerm? x0 | return none
      unless t0.isConstOf ``ClifTy.i64 do return none
      let (g, pfs, k, n) ← bindFound g known env found
      return ← g.withContext do
        let mem ← mkAppM ``World.mem #[w]
        let hc ← mkAppOptM ``evalOp_ireduce32_64 #[mem, env, rs[0]!, b0, pfs[0]!]
        let v ← mkAppM ``HAnd.hAnd #[b0, ← mkAppM ``widthMask #[mkConst ``ClifTy.i32]]
        let c ← mkAppM ``V.sc #[mkConst ``ClifTy.i32, v]
        let gs ← applyNamed g ``wp_op_val [(`c, c), (`hc, hc)]
        return some (gs.map (·, k, n))
    -- a comparison of two 64-bit values: a byte, one when it holds
    if opName == ``Op.icmp then
      let [x0, x1] := xs | return none
      let some (t0, b0) ← scalarTerm? x0 | return none
      let some (t1, b1) ← scalarTerm? x1 | return none
      unless t0.isConstOf ``ClifTy.i64 && t1.isConstOf ``ClifTy.i64 do return none
      let (g, pfs, k, n) ← bindFound g known env found
      return ← g.withContext do
        let mem ← mkAppM ``World.mem #[w]
        let hc ← mkAppOptM ``evalOp_icmp64 #[mem, env, oe.getArg! 0, rs[0]!, rs[1]!, b0, b1, pfs[0]!, pfs[1]!]
        let c := (← instantiateMVars (← inferType hc)).appArg!.appArg!
        let gs ← applyNamed g ``wp_op_val [(`c, c), (`hc, hc)]
        return some (gs.map (·, k, n))
    -- the lesser of two 64-bit values: a term, as a sum is
    let isUmin ← if opName == ``Op.ibin then pure ((← whnf (oe.getArg! 0)).isConstOf ``IBin.umin)
      else pure false
    if isUmin then
      let [x0, x1] := xs | return none
      let some (t0, b0) ← scalarTerm? x0 | return none
      let some (t1, b1) ← scalarTerm? x1 | return none
      unless t0.isConstOf ``ClifTy.i64 && t1.isConstOf ``ClifTy.i64 do return none
      -- its bounds, for the arithmetic: a rewrite of its number would be
      -- tried at every number of every goal
      let (_, g) ← g.withContext do g.note `hmin (← mkAppM ``umin_bounds #[b0, b1])
      let (g, pfs, k, n) ← bindFound g known env found
      return ← g.withContext do
        let mem ← mkAppM ``World.mem #[w]
        let hc ← mkAppOptM ``evalOp_umin64 #[mem, env, rs[0]!, rs[1]!, b0, b1, pfs[0]!, pfs[1]!]
        let c := (← instantiateMVars (← inferType hc)).appArg!.appArg!
        let gs ← applyNamed g ``wp_op_val [(`c, c), (`hc, hc)]
        return some (gs.map (·, k, n))
    -- a choice between two 64-bit values on an integer condition
    if opName == ``Op.select then
      let [xc, xa, xb] := xs | return none
      let some (tc, bc) ← scalarTerm? xc | return none
      let some (ta, ba) ← scalarTerm? xa | return none
      let some (tb, bb) ← scalarTerm? xb | return none
      unless ta.isConstOf ``ClifTy.i64 && tb.isConstOf ``ClifTy.i64 && !tc.hasFVar do return none
      unless (← evalOut (← mkAppM ``ClifTy.isInt #[tc])).isConstOf ``Bool.true do return none
      let (g, pfs, k, n) ← bindFound g known env found
      return ← g.withContext do
        let mem ← mkAppM ``World.mem #[w]
        let htc ← mkExpectedTypeHint (← mkEqRefl (mkConst ``Bool.true))
          (← mkEq (← mkAppM ``ClifTy.isInt #[tc]) (mkConst ``Bool.true))
        let hc ← mkAppOptM ``evalOp_select64 #[mem, env, rs[0]!, rs[1]!, rs[2]!, tc, bc, ba, bb,
          pfs[0]!, htc, pfs[1]!, pfs[2]!]
        let c := (← instantiateMVars (← inferType hc)).appArg!.appArg!
        let gs ← applyNamed g ``wp_op_val [(`c, c), (`hc, hc)]
        return some (gs.map (·, k, n))
    -- a 64-bit value's leading zeros or reversed bits: a term
    if opName == ``Op.iun then
      let kind ← whnf (oe.getArg! 0)
      let lem? := if kind.isConstOf ``IUn.clz then some ``evalOp_clz64
        else if kind.isConstOf ``IUn.bitrev then some ``evalOp_bitrev64 else none
      let some lem := lem? | return none
      let [x0] := xs | return none
      let some (t0, b0) ← scalarTerm? x0 | return none
      unless t0.isConstOf ``ClifTy.i64 do return none
      let (g, pfs, k, n) ← bindFound g known env found
      return ← g.withContext do
        let mem ← mkAppM ``World.mem #[w]
        let hc ← mkAppOptM lem #[mem, env, rs[0]!, b0, pfs[0]!]
        let c := (← instantiateMVars (← inferType hc)).appArg!.appArg!
        let gs ← applyNamed g ``wp_op_val [(`c, c), (`hc, hc)]
        return some (gs.map (·, k, n))
    unless oe.getAppNumArgs == 2 do return none
    let [x0, x1] := xs | return none
    let some (t0, b0) ← scalarTerm? x0 | return none
    let some (t1, b1) ← scalarTerm? x1 | return none
    let bin2 (f : Name) : Expr → Expr → MetaM Expr := fun a b => mkAppM f #[a, b]
    let arith : Option (Name × (Expr → Expr → MetaM Expr)) :=
      if opName == ``Op.iadd then some (``evalOp_iadd64, bin2 ``HAdd.hAdd)
      else if opName == ``Op.isub then some (``evalOp_isub64, bin2 ``HSub.hSub)
      else if opName == ``Op.imul then some (``evalOp_imul64, bin2 ``HMul.hMul)
      else if opName == ``Op.band then some (``evalOp_band64, bin2 ``HAnd.hAnd)
      else if opName == ``Op.bor then some (``evalOp_bor64, bin2 ``HOr.hOr)
      else if opName == ``Op.bxor then some (``evalOp_bxor64, bin2 ``HXor.hXor)
      else if opName == ``Op.bandNot then
        some (``evalOp_bandNot64, fun a b => do mkAppM ``HAnd.hAnd #[a, ← mkAppM ``Complement.complement #[b]])
      else none
    -- a quotient by a constant other than zero
    if opName == ``Op.udiv then
      unless t0.isConstOf ``ClifTy.i64 && t1.isConstOf ``ClifTy.i64 && !b1.hasFVar do return none
      let some hy ← decideTrue? (mkNot (← mkEq b1 (← mkAppOptM ``OfNat.ofNat
          #[mkConst ``UInt64, mkRawNatLit 0, none]))) | return none
      let (g, pfs, k, n) ← bindFound g known env found
      return ← g.withContext do
        let mem ← mkAppM ``World.mem #[w]
        let hc ← mkAppOptM ``evalOp_udiv64 #[mem, env, rs[0]!, rs[1]!, b0, b1, pfs[0]!, pfs[1]!, hy]
        let c ← mkAppM ``V.sc #[mkConst ``ClifTy.i64, ← mkAppM ``HDiv.hDiv #[b0, b1]]
        let gs ← applyNamed g ``wp_op_val [(`c, c), (`hc, hc)]
        return some (gs.map (·, k, n))
    let shift : Option (Name × Name) :=
      if opName == ``Op.ishl then some (``evalOp_ishl64, ``HShiftLeft.hShiftLeft)
      else if opName == ``Op.ushr then some (``evalOp_ushr64, ``HShiftRight.hShiftRight) else none
    if arith.isNone && shift.isNone then return none
    -- a 64-bit value with one whose type is not known: 64-bit where defined
    if arith.isSome && opName != ``Op.bandNot && t0.isConstOf ``ClifTy.i64 && !t1.isConstOf ``ClifTy.i64 then
      let some (lem, f) := arith | return none
      let ops := [``Op.iadd, ``Op.isub, ``Op.imul, ``Op.band, ``Op.bor, ``Op.bxor]
      let some idx := ops.idxOf? opName | return none
      let (g, pfs, k, n) ← bindFound g known env found
      return ← g.withContext do
        let mem ← mkAppM ``World.mem #[w]
        let lhsB := (← instantiateMVars (← inferType pfs[1]!)).getArg! 1
        let hc ← withLocalDeclD `h (← mkEq lhsB (← mkAppM ``Option.some
            #[← mkAppM ``V.sc #[mkConst ``ClifTy.i64, b1]])) fun h => do
          mkLambdaFVars #[h] (← mkAppOptM lem #[mem, env, rs[0]!, rs[1]!, b0, b1, pfs[0]!, h])
        let eqs ← ops.mapM fun c => mkEq oe (mkApp2 (mkConst c) rs[0]! rs[1]!)
        let mut ho ← mkEqRefl oe
        let mut i := idx
        if idx < ops.length - 1 then
          ho ← mkAppOptM ``Or.inl #[eqs[idx]!, ← mkListOr (eqs.drop (idx + 1)), ho]
        while i > 0 do
          i := i - 1
          ho ← mkAppOptM ``Or.inr #[eqs[i]!, ← mkListOr (eqs.drop (i + 1)), ho]
        let gs ← applyNamed g ``wp_op_right64 [(`r, ← f b0 b1), (`ho, ho), (`ha, pfs[0]!),
          (`hb, pfs[1]!), (`hc, hc)]
        return some (gs.map (·, k, n))
    if arith.isSome && !t1.isConstOf ``ClifTy.i64 then return none
    -- a value whose type is not known, with a 64-bit one: 64-bit where defined
    if !t0.isConstOf ``ClifTy.i64 then
      if opName == ``Op.bandNot then return none
      let some (lem, f) := arith | return none
      let ops := [``Op.iadd, ``Op.isub, ``Op.imul, ``Op.band, ``Op.bor, ``Op.bxor]
      let some idx := ops.idxOf? opName | return none
      let (g, pfs, k, n) ← bindFound g known env found
      return ← g.withContext do
        let mem ← mkAppM ``World.mem #[w]
        let lhsA := (← instantiateMVars (← inferType pfs[0]!)).getArg! 1
        let hc ← withLocalDeclD `h (← mkEq lhsA (← mkAppM ``Option.some
            #[← mkAppM ``V.sc #[mkConst ``ClifTy.i64, b0]])) fun h => do
          mkLambdaFVars #[h] (← mkAppOptM lem #[mem, env, rs[0]!, rs[1]!, b0, b1, h, pfs[1]!])
        -- which of the six the operation is, as a proof
        let eqs ← ops.mapM fun c => mkEq oe (mkApp2 (mkConst c) rs[0]! rs[1]!)
        let mut ho ← mkEqRefl oe
        let mut i := idx
        if idx < ops.length - 1 then
          ho ← mkAppOptM ``Or.inl #[eqs[idx]!, ← mkListOr (eqs.drop (idx + 1)), ho]
        while i > 0 do
          i := i - 1
          ho ← mkAppOptM ``Or.inr #[eqs[i]!, ← mkListOr (eqs.drop (i + 1)), ho]
        let gs ← applyNamed g ``wp_op_left64 [(`r, ← f b0 b1), (`ho, ho), (`ha, pfs[0]!),
          (`hb, pfs[1]!), (`hc, hc)]
        return some (gs.map (·, k, n))
    unless t0.isConstOf ``ClifTy.i64 do return none
    -- a shift amount of any integer type, known to be one
    if shift.isSome && (t1.hasFVar || !(← evalOut (← mkAppM ``ClifTy.isInt #[t1])).isConstOf ``Bool.true) then
      return none
    -- a bit-reversed index shifted down past a bound's leading zeros and one:
    -- below the bound, for the arithmetic
    let mut g1 := g
    if opName == ``Op.ushr && b0.isAppOfArity ``bitrev64 1 && b1.isAppOfArity ``HAdd.hAdd 6
        && (b1.getArg! 4).isAppOfArity ``clz64 1 then
      let one ← mkAppOptM ``OfNat.ofNat #[mkConst ``UInt64, mkRawNatLit 1, none]
      if let some hc ← decideTrue? (← mkEq (b1.getArg! 5) one) then
        let pf ← g.withContext <|
          mkAppM ``bitrev_index_lt #[b0.appArg!, (b1.getArg! 4).appArg!, b1.getArg! 5, hc]
        let (_, g') ← g.withContext (g.note `hrev pf)
        g1 := g'
    let (g, pfs, k, n) ← bindFound g1 known env found
    return ← g.withContext do
      let mem ← mkAppM ``World.mem #[w]
      let (hc, v) ← match arith with
        | some (lem, f) => do
            pure (← mkAppOptM lem #[mem, env, rs[0]!, rs[1]!, b0, b1, pfs[0]!, pfs[1]!], ← f b0 b1)
        | none => do
            let some (lem, f) := shift | throwError "vcOpComputed: no rule"
            let isInt ← mkAppM ``ClifTy.isInt #[t1]
            let htb ← mkExpectedTypeHint (← mkEqRefl (mkConst ``Bool.true)) (← mkEq isInt (mkConst ``Bool.true))
            let amt ← mkAppM ``HMod.hMod #[b1, ← mkAppOptM ``OfNat.ofNat #[mkConst ``UInt64, mkRawNatLit 64, none]]
            pure (← mkAppOptM lem #[mem, env, rs[0]!, rs[1]!, b0, b1, t1, pfs[0]!, pfs[1]!, htb],
              ← mkAppM f #[b0, amt])
      let c ← mkAppM ``V.sc #[mkConst ``ClifTy.i64, v]
      let gs ← applyNamed g ``wp_op_val [(`c, c), (`hc, hc)]
      return some (gs.map (·, k, n))
  let delta ← mkAppM ``List.toArray #[← mkListLit (mkConst ``V) xs]
  let dflt ← mkAppOptM ``Inhabited.default #[mkConst ``Mem, none]
  let rhs ← mkAppM ``evalOp #[dflt, delta, ← mkAppM ``renumberOp #[oe]]
  let r ← evalOut rhs
  unless r.isAppOfArity ``Option.some 2 do return none
  let (g, pfs, k, n) ← bindFound g known env found
  g.withContext do
  let mem ← mkAppM ``World.mem #[w]
  let hr ← mkExpectedTypeHint (← mkEqRefl regs) (← mkEq (← mkAppM ``Op.regs #[oe]) regs)
  let fls := mkConst ``Bool.false
  let hl ← mkExpectedTypeHint (← mkEqRefl fls) (← mkEq (← mkAppM ``opReadsMem #[oe]) fls)
  let hre ← mkAppOptM lem (#[some mem, some dflt, some env, some oe] ++ rs.toArray.map some ++
    xs.toArray.map some ++ #[some hr, some hl] ++ pfs.map some)
  let hc ← mkEqTrans hre (← mkExpectedTypeHint (← mkEqRefl r) (← mkEq rhs r))
  let gs ← applyNamed g ``wp_op_val [(`c, r.appArg!), (`hc, hc)]
  return some (gs.map (·, k, n))

/-- An operation whose value follows from what is known: any operation over
    known operands that reads no memory, or an eight-byte load from a cell the
    typestate names. -/
meta def vcOpVal (k : VCKnow) (g : MVarId) (t p : Expr) : TacticM (Option (List (MVarId × VCKnow × Nat))) := do
  let env := t.getArg! 6
  let w := t.getArg! 7
  let oe ← whnf (← mkAppM ``Op'.erase #[p.getArg! 4])
  unless oe.isAppOfArity ``Op.load 2 do return ← vcOpComputed k g env w oe
  let op := oe.getArg! 0
  let some (x, pfx, k) ← k.lookup (oe.getArg! 1) env | return none
  let some (tx, bx) ← scalarOf? x | return none
  let some (S, hW) ← stateAt? k w | return none
  let kind ← mkAppM ``LoadOp.kind #[op]
  unless (← evalOut kind).isConstOf ``LoadKind.plain do return none
  let hk ← mkEqRefl (mkConst ``LoadKind.plain)
  let opTy ← mkAppM ``LoadOp.ty #[op]
  let lanesV ← evalOut (← mkAppM ``ClifTy.lanes #[opTy])
  unless lanesV.isAppOf ``Option.none do return none
  let hl0 ← mkEqRefl lanesV
  let nb ← (evalNat (← evalOut (← mkAppM ``tyBytes #[opTy]))).run
  -- four bytes at a cell, or four past it: the low or the high half of its value
  if nb == some 4 then
    let (Sv, k) ← k.out S
    let some facts := Sv.listLit? | return none
    let u64 := mkConst ``UInt64
    for f in facts.2 do
      unless f.isAppOfArity ``Contracts.Fact.cell 2 do continue
      let v := f.getArg! 1
      if v.hasFVar then continue
      let ca ← evalOut (f.getArg! 0)
      let lo ← isDefEq ca bx
      let hi ← if lo then pure false else isDefEq (← evalOut (← mkAppM ``HAdd.hAdd #[ca, ← mkNumeral u64 4])) bx
      unless lo || hi do continue
      let some hc ← decideTrue? (← mkEq (← mkAppM ``List.contains #[S, f]) (mkConst ``Bool.true)) | continue
      let hl? : Option Expr ← if lo then pure (some (← mkAppM ``Contracts.TState.holds_cell_half #[hW, hc])) else do
        let dec ← evalOut (← mkAppM ``decodeAddr #[ca])
        if !dec.isAppOfArity ``Option.some 2 then pure none else
        let off := dec.appArg!.getArg! 3
        let some hd ← decideTrue? (← mkEq (← mkAppM ``decodeAddr #[ca]) dec) | pure none
        let some hi ← decideTrue? (← mkAppM ``LT.lt #[← mkAppM ``HAdd.hAdd #[off, mkNatLit 4],
          ← mkAppM ``UInt64.toNat #[mkConst ``regionSpan]]) | pure none
        pure (some (← mkAppM ``Contracts.TState.holds_cell_hi #[hW, hc, hd, hi]))
      let some hl := hl? | continue
      let mask ← mkNumeral u64 0xFFFFFFFF
      let sh ← mkNumeral u64 32
      let v4 ← evalOut (← if lo then mkAppM ``HAnd.hAnd #[v, mask] else mkAppM ``HShiftRight.hShiftRight #[v, sh])
      let mem ← mkAppM ``World.mem #[w]
      let hl ← mkExpectedTypeHint hl
        (← mkEq (← mkAppM ``Mem.load #[mem, bx, ← mkAppM ``tyBytes #[opTy]]) (← mkAppM ``Option.some #[v4]))
      let c ← evalOut (mkApp2 (mkConst ``V.sc) opTy v4)
      let (g, pfs, k, n) ← bindFound g k env #[(oe.getArg! 1, x, pfx)]
      return ← g.withContext do
        let hcE ← mkAppOptM ``evalOp_load #[mem, env, oe.getArg! 1, op, tx, bx, v4, hk, hl0, pfs[0]!, hl]
        let gs ← applyNamed g ``wp_op_val [(`c, c), (`hc, hcE)]
        return some (gs.map (·, k, n))
    return none
  unless nb == some 8 do return none
  let (Sv, k) ← k.out S
  let some facts := Sv.listLit? | return none
  let mut hit := none
  for f in facts.2 do
    if f.isAppOfArity ``Contracts.Fact.cell 2 then
      if ← isDefEq (← evalOut (f.getArg! 0)) bx then
        hit := some (f.getArg! 1)
        break
  let some v := hit | return none
  let cell := mkApp2 (mkConst ``Contracts.Fact.cell) bx v
  -- a cell holding a value the program was answered is found by its place:
  -- deciding membership would compare that value with itself
  let hl? ← do
    if let some hc ← decideTrue? (← mkEq (← mkAppM ``List.contains #[S, cell]) (mkConst ``Bool.true)) then
      pure (some (← mkAppM ``Contracts.TState.holds_cell #[hW, hc]))
    else if let some hm ← memLit? facts.1 facts.2 (fun f => pure (f.isAppOfArity ``Contracts.Fact.cell 2 &&
        (f.getArg! 1) == v)) then
      let hm ← mkExpectedTypeHint hm (← mkAppM ``Membership.mem #[S, cell])
      pure (some (← mkAppM ``Contracts.TState.holds_cell_mem #[hW, hm]))
    else pure none
  let some hl := hl? | return none
  let c ← evalOut (mkApp2 (mkConst ``V.sc) opTy v)
  let (g, pfs, k, n) ← bindFound g k env #[(oe.getArg! 1, x, pfx)]
  g.withContext do
  let mem ← mkAppM ``World.mem #[w]
  let hcE ← mkAppOptM ``evalOp_load #[mem, env, oe.getArg! 1, op, tx, bx, v, hk, hl0, pfs[0]!, hl]
  let gs ← applyNamed g ``wp_op_val [(`c, c), (`hc, hcE)]
  return some (gs.map (·, k, n))

/-- A store to an address known only by its bounds, by the typestate: the
    facts the store provably cannot reach, each shown by the address's bounds. -/
meta def vcStoreSub (g : MVarId) (env S hW ty a v : Expr)
    (q : Expr × Expr × Expr × Expr × Expr × Expr × Expr × Expr × VCKnow) (unaligned : Bool := false) :
    TacticM (Option (List (MVarId × VCKnow × Nat))) := do
  let (xa, xv, ta, ba, tv, bv, pfa, pfv, k) := q
  -- an unaligned store is as wide as its value, sixteen bytes at most
  let n := if unaligned then mkNatLit 16 else mkApp (mkConst ``tyBytes) ty
  let some Sv ← kernelList S | return none
  let some (α, facts) := Sv.listLit? | return none
  let r ← g.withContext do
    let mut kept := #[]
    let mut pfs := #[]
    for x in facts do
      let fn := x.getAppFn.constName?.getD .anonymous
      if fn == ``Contracts.Fact.cstr then continue
      let goal ← mkEq (← mkAppM ``Contracts.storeKeeps #[ba, n, x]) (mkConst ``Bool.true)
      let m ← mkFreshExprMVar goal
      let tac ← if fn == ``Contracts.Fact.cell || fn == ``Contracts.Fact.cstrIn then `(tactic| store_keep_one)
        else `(tactic| (simp only [Contracts.storeKeeps]; done))
      let st ← saveState
      -- each fact on a small budget: one the arithmetic cannot show missed
      -- soon is dropped, which only costs a fact
      let ok ← tryCatchRuntimeEx
          (withCurrHeartbeats <| withTheReader Core.Context (fun c => { c with maxHeartbeats := 20000000 }) do
            let rest ← Term.withoutErrToSorry (Tactic.run m.mvarId! (withoutRecover <| evalTactic tac))
            let pf ← instantiateMVars m
            pure (rest.isEmpty && !pf.hasSorry && !pf.hasExprMVar))
          (fun _ => pure false)
      if ok then
        kept := kept.push x
        let pf ← instantiateMVars m
        pfs := pfs.push pf
      else st.restore
    let S' ← mkListLit α kept.toList
    let hsubT ← mkAppM ``List.Sublist #[S', S]
    let hsub ← mkFreshExprMVar hsubT
    let r1 ← Tactic.run hsub.mvarId! (withoutRecover <| evalTactic (← `(tactic| sublist_lits)))
    unless r1.isEmpty do throwError "vcStore: sublist"
    let kf ← mkAppM ``Contracts.storeKeeps #[ba, n]
    let nil ← mkListLit α []
    let mut hkeep ← mkExpectedTypeHint (← mkEqRefl (mkConst ``Bool.true))
      (← mkEq (← mkAppM ``List.all #[nil, kf]) (mkConst ``Bool.true))
    let mut tail := nil
    for (x, pf) in (kept.zip pfs).reverse do
      hkeep ← mkAppOptM ``all_cons_true #[α, kf, x, tail, pf, hkeep]
      tail ← mkAppOptM ``List.cons #[α, x, tail]
    pure (S', hsub, hkeep)
  let (S', hsub, hkeep) := r
  let (g, pfs, k, nh) ← bindFound g k env #[(a, xa, pfa), (v, xv, pfv)]
  g.withContext do
  let gs ← applyNamed g (if unaligned then ``wp_storeU_sub else ``wp_store_sub)
    [(`S, S), (`S', S'), (`t, ta), (`x, ba), (`t', tv), (`y, bv), (`ha, pfs[0]!), (`hv, pfs[1]!),
     (`hS, hW), (`hsub, ← instantiateMVars hsub), (`hkeep, ← instantiateMVars hkeep)]
  return some (gs.map (·, k, nh))

/-- A store under default flags of a value whose type is not known: at most
    sixteen bytes wide, so the cells beyond them are kept. -/
meta def vcStoreWide (g : MVarId) (k : VCKnow) (env S hW a v xa xv ta ba tv bv pfa pfv : Expr) :
    TacticM (Option (List (MVarId × VCKnow × Nat))) := do
  let u64 := mkConst ``UInt64
  let (S', hS'h) ← stateOutHint (← mkAppM ``afterStore #[← mkAppM ``Option.some #[ba],
    ← mkAppOptM ``Option.none #[u64], mkNatLit 16, S])
  let (g, pfs, k, nh) ← bindFound g k env #[(a, xa, pfa), (v, xv, pfv)]
  g.withContext do
  let gs ← applyNamed g ``wp_storeU_wide
    [(`S, S), (`S', S'), (`t, ta), (`x, ba), (`t', tv), (`y, bv), (`ha, pfs[0]!), (`hv, pfs[1]!), (`hS, hW),
     (`hS', hS'h)]
  return some (gs.map (·, k, nh))

/-- A scalar load under a post that allows no fault, from an address whose
    bytes `fits_close` shows are there; `none` otherwise. -/
meta def vcLoadAt (k : VCKnow) (g : MVarId) (t oe ho hl ht : Expr) :
    TacticM (Option (List (MVarId × VCKnow × Nat))) := do
  let J := t.getArg! 2
  if ← faultAllowed g J then return none
  let env := t.getArg! 6
  let some (xa, pfa, k) ← k.lookup (oe.getArg! 1) env | return none
  let some (ta, ba) ← scalarTerm? xa | return none
  let hJT ← mkAppM ``Or #[← mkEq (← mkAppM ``Post.faultOk #[J]) (mkConst ``Bool.true),
    ← mkAppM ``Fits #[← mkAppM ``World.mem #[t.getArg! 7], ba, ← mkAppM ``loadBytes #[oe.getArg! 0]]]
  let hJ ← mkFreshExprMVar hJT
  unless ← closeFits hJ.mvarId! do return none
  -- a byte zero-extended is below 256, so a slot it indexes is bounded
  let u8 := (← whnfD (← mkAppM ``LoadOp.kind #[oe.getArg! 0])).isConstOf ``LoadKind.uload8
  let gs ← if u8 then
      applyNamed g ``wp_op_uload8_at [(`ho, ho), (`hl, hl), (`ht, ht), (`hk8, ← mkEqRefl (mkConst ``LoadKind.uload8)),
        (`t', ta), (`x, ba), (`ha, pfa), (`hJ, ← instantiateMVars hJ)]
    else applyNamed g ``wp_op_load_at [(`ho, ho), (`hl, hl), (`ht, ht),
      (`t', ta), (`x, ba), (`ha, pfa), (`hJ, ← instantiateMVars hJ)]
  return some (gs.map (·, k, 0))

/-- A store by the typestate: what it keeps, and the cell it writes when its
    address and value are known. -/
meta def vcStore (k : VCKnow) (g : MVarId) (t p : Expr) (unaligned : Bool) :
    TacticM (Option (List (MVarId × VCKnow × Nat))) := do
  let env := t.getArg! 6
  let some (S, hW) ← stateAt? k (t.getArg! 7) | return none
  let ty := p.getArg! 2
  let v := p.getArg! 4
  let a := p.getArg! 5
  let u64 := mkConst ``UInt64
  -- a first pass only walks on: the store forgets what it may reach
  if (← getOptions).getBool `vcProbe false then
    let n := if unaligned then mkNatLit 1 else mkApp (mkConst ``tyBytes) ty
    let none_ ← mkAppOptM ``Option.none #[u64]
    let (S', hS'h) ← stateOutHint (← mkAppM ``afterStore #[none_, none_, n, S])
    let gs ← applyNamed g (if unaligned then ``wp_storeU_unknown else ``wp_store_unknown)
      [(`S, S), (`S', S'), (`hS, hW), (`hS', hS'h)]
    return some (gs.map (·, k, 0))
  -- a vector, known by its type, where the post allows no fault: its lanes
  -- fit; the cells go
  if !(← faultAllowed g (t.getArg! 2)) then
    if let some (xv, pfv, k1) ← k.lookup v env then
      if let some (T, hty) ← k1.tyV? xv then
        let l ← whnfD (← mkAppM ``ClifTy.lanes #[T])
        if l.isAppOfArity ``Option.some 2 then
          if let some (xa, pfa, k2) ← k1.lookup a env then
            if let some (ta, ba) ← scalarOf? xa then
              let hl ← mkExpectedTypeHint (← mkEqRefl l) (← mkEq (← mkAppM ``ClifTy.lanes #[T]) l)
              let ha ← mkExpectedTypeHint pfa (← mkEq (← inferType pfa >>= fun ty => pure (ty.eq?.get!.2.1))
                (← mkAppM ``Option.some #[mkApp2 (mkConst ``V.sc) ta ba]))
              let p' ← withLocalDeclD `f (mkConst ``Contracts.Fact) fun f => do
                mkLambdaFVars #[f] (← mkAppM ``not #[← mkAppM ``Contracts.Fact.isCell #[f]])
              let (S', hS'h) ← stateOutHint (← mkAppM ``List.filter #[p', S])
              let gs ← applyNamed g (if unaligned then ``wp_storeU_tyv else ``wp_store_tyv)
                [(`S, S), (`S', S'), (`ha, ha), (`hv, pfv), (`hty, hty),
                (`hl, hl), (`hS, hW), (`hS', hS'h)]
              return some (gs.map (·, k2, 0))
  let known : Option (Expr × Expr × Expr × Expr × Expr × Expr × Expr × Expr × VCKnow) ← do
    let some (xa, pfa, k) ← k.lookup a env | pure none
    let some (xv, pfv, k) ← k.lookup v env | pure none
    let some (ta, ba) ← scalarOf? xa | pure none
    let some (tv, bv) ← scalarOf? xv | pure none
    pure (some (xa, xv, ta, ba, tv, bv, pfa, pfv, k))
  -- an address known only by its bounds keeps the facts it provably misses
  let sub ← match known with
    | some q => if q.2.2.2.1.hasFVar then vcStoreSub g env S hW ty a v q unaligned else pure none
    | none => pure none
  if sub.isSome then return sub
  let known := known.filter fun q => !q.2.2.2.1.hasFVar
  match known with
    -- a cell names only values written out, since the typestate is decided on:
    -- a value that is not keeps the cells the store cannot reach
    | some (xa, xv, ta, ba, tv, bv, pfa, pfv, k) =>
      if bv.hasFVar && unaligned && tv.hasFVar then
        return (← vcStoreWide g k env S hW a v xa xv ta ba tv bv pfa pfv)
      if bv.hasFVar then
        let n := mkApp (mkConst ``tyBytes) (if unaligned then tv else ty)
        let (S', hS'h) ← stateOutHint (← mkAppM ``afterStore #[← mkAppM ``Option.some #[ba],
          ← mkAppOptM ``Option.none #[u64], n, S])
        let (g, pfs, k, nh) ← bindFound g k env #[(a, xa, pfa), (v, xv, pfv)]
        g.withContext do
        let gs ← applyNamed g (if unaligned then ``wp_storeU_addr else ``wp_store_addr)
          [(`S, S), (`S', S'), (`t, ta), (`x, ba), (`t', tv), (`y, bv), (`ha, pfs[0]!), (`hv, pfs[1]!), (`hS, hW),
           (`hS', hS'h)]
        return some (gs.map (·, k, nh))
      else
        let n := mkApp (mkConst ``tyBytes) (if unaligned then tv else ty)
        let (S', hS'h) ← stateOutHint (← mkAppM ``afterStore #[← mkAppM ``Option.some #[ba], ← mkAppM ``Option.some #[bv], n, S])
        let (g, pfs, k, nh) ← bindFound g k env #[(a, xa, pfa), (v, xv, pfv)]
        g.withContext do
        let gs ← applyNamed g (if unaligned then ``wp_storeU_known else ``wp_store_known)
          [(`S, S), (`S', S'), (`t, ta), (`x, ba), (`t', tv), (`y, bv), (`ha, pfs[0]!), (`hv, pfs[1]!), (`hS, hW),
           (`hS', hS'h)]
        return some (gs.map (·, k, nh))
    | none =>
        let n := if unaligned then mkNatLit 1 else mkApp (mkConst ``tyBytes) ty
        let none_ ← mkAppOptM ``Option.none #[u64]
        let (S', hS'h) ← stateOutHint (← mkAppM ``afterStore #[none_, none_, n, S])
        let gs ← applyNamed g (if unaligned then ``wp_storeU_unknown else ``wp_store_unknown)
          [(`S, S), (`S', S'), (`hS, hW), (`hS', hS'h)]
        return some (gs.map (·, k, 0))

/-- A byte store by the typestate. -/
meta def vcIstore8 (k : VCKnow) (g : MVarId) (t p : Expr) :
    TacticM (Option (List (MVarId × VCKnow × Nat))) := do
  let env := t.getArg! 6
  let some (S, hW) ← stateAt? k (t.getArg! 7) | return none
  let a := p.getArg! 5
  let none_ ← mkAppOptM ``Option.none #[mkConst ``UInt64]
  let known : Option (Expr × Expr × Expr × Expr × VCKnow) ← do
    let some (xa, pfa, k) ← k.lookup a env | pure none
    let some (ta, ba) ← scalarOf? xa | pure none
    -- an address known only by its bounds, where the post allows a fault:
    -- the store forgets the cells, as it costs nothing to say so
    if ba.hasFVar && (← faultAllowed g (t.getArg! 2)) then pure none
    else pure (some (xa, ta, ba, pfa, k))
  match known with
    | some (xa, ta, ba, pfa, k) =>
        -- an address over values not known: the cells go, as computing which
        -- it may reach would unfold the arithmetic
        if ba.hasFVar then
          let p' ← withLocalDeclD `f (mkConst ``Contracts.Fact) fun f => do
            mkLambdaFVars #[f] (← mkAppM ``not #[← mkAppM ``Contracts.Fact.isCell #[f]])
          let (S', hS'h) ← stateOutHint (← mkAppM ``List.filter #[p', S])
          let (g, pfs, k, nh) ← bindFound g k env #[(a, xa, pfa)]
          return ← g.withContext do
            let gs ← applyNamed g ``wp_istore8_sym
              [(`S, S), (`S', S'), (`t, ta), (`x, ba), (`ha, pfs[0]!), (`hS, hW), (`hS', hS'h)]
            return some (gs.map (·, k, nh))
        let (S', hS'h) ← stateOutHint (← mkAppM ``afterStore #[← mkAppM ``Option.some #[ba], none_, mkNatLit 1, S])
        let (g, pfs, k, nh) ← bindFound g k env #[(a, xa, pfa)]
        g.withContext do
        let gs ← applyNamed g ``wp_istore8_known
          [(`S, S), (`S', S'), (`t, ta), (`x, ba), (`ha, pfs[0]!), (`hS, hW), (`hS', hS'h)]
        return some (gs.map (·, k, nh))
    | none =>
        let (S', hS'h) ← stateOutHint (← mkAppM ``afterStore #[none_, none_, mkNatLit 1, S])
        let gs ← applyNamed g ``wp_istore8_unknown [(`S, S), (`S', S'), (`hS, hW), (`hS', hS'h)]
        return some (gs.map (·, k, 0))

/-- Close a goal with a tactic, or leave it as it was. -/
meta def closeWith (g : MVarId) (tac : TSyntax `tactic) : TacticM (List MVarId) := do
  let st ← saveState
  -- a failure is a failure: not logged and the goal admitted; a limit the
  -- attempt reached, too
  tryCatchRuntimeEx (Tactic.run g (withoutRecover (evalTactic tac))) fun _ => do
    st.restore
    return [g]

/-- **A call of the engine's own libraries**, by the typestate: look up the
    arguments, compute their bits and what the call needs, and go on in what
    it leaves. The handles it needs are found among the facts by `simp`, and
    the room it needs is decided on the facts that name no handle. -/
meta def vcExt (k : VCKnow) (g : MVarId) (t p : Expr) : TacticM (Option (List (MVarId × VCKnow × Nat))) := do
  let env := t.getArg! 6
  let w := t.getArg! 7
  let r ← whnf (p.getArg! 5)
  unless r.isAppOfArity ``LocalRef.mk 3 do return none
  let callee ← whnf (r.getArg! 2)
  unless callee.isAppOfArity ``Callee.ext 1 do return none
  let e := callee.appArg!
  unless (← evalOut (← mkAppM ``Ext.aside #[e])).isConstOf ``Bool.true do return none
  let some (S, hW) ← stateAt? k w | return none
  let some ss ← slotList (p.getArg! 6) | return none
  let mut kk := k
  let mut found := #[]
  for sl in ss do
    let some (c, pf, k') ← kk.lookup sl env | return none
    kk := k'
    found := found.push (sl, c, pf)
  let cs := found.toList.map (·.2.1)
  let vTy := mkConst ``V
  let natTy := mkConst ``Nat
  let some bits ← bitsOf? (← mkListLit vTy cs) | return none
  let needsE ← mkAppM ``Contracts.extNeeds #[e, bits]
  let needs ← evalOut needsE
  unless needs.isAppOfArity ``Option.some 2 do return none
  let (Sv, k1) ← kk.out S
  let plain := mkConst ``Contracts.Fact.plain
  let (S', k2) ← k1.out (← mkAppM ``List.filter #[plain, Sv])
  let (g, pfs, kb, nh) ← bindFound g k2 env found
  g.withContext do
  let mut hcs := mkApp (mkConst ``lookup_nil) env
  for i in (List.range ss.length).reverse do
    hcs := mkAppN (mkConst ``lookup_cons)
      #[env, ss[i]!, cs[i]!, ← mkListLit natTy (ss.drop (i + 1)), ← mkListLit vTy (cs.drop (i + 1)), pfs[i]!, hcs]
  let res := (← evalOut (← mkAppM ``Option.isSome #[← mkAppM ``Prod.snd #[← mkAppM ``Ext.sig #[e]]])).isConstOf
    ``Bool.true
  let holds := mkConst ``Contracts.TState.holds
  let hWv ← mkExpectedTypeHint hW (mkApp2 holds Sv w)
  let hn ← mkExpectedTypeHint (← mkEqRefl needs) (← mkEq needsE needs)
  let hS' ← mkExpectedTypeHint (← mkEqRefl S') (← mkEq (← mkAppM ``List.filter #[plain, Sv]) S')
  let gs ← applyNamed g (if res then ``wp_ext_res else ``wp_ext_void)
    [(`S, Sv), (`S', S'), (`hres, ← mkEqRefl (mkConst (if res then ``Bool.true else ``Bool.false))),
     (`ha, ← mkEqRefl (mkConst ``Bool.true)), (`hcs, hcs), (`hn, hn), (`hS', hS'), (`hS, hWv)]
  let [gb, gh, gr, gq] := gs | return none
  let rb ← closeWith gb (← `(tactic| bits_rfl))
  let rh ← closeWith gh (← `(tactic| simp))
  let rr ← closeWith gr (← `(tactic| kdecide))
  return some ((rb ++ rh ++ rr).map (·, kb, 0) ++ [(gq, kb, nh)])

/-- A call to one of the program's own functions that answers nothing, by the
    summary the caller's hypotheses give for that function on those arguments
    (`wp_local_void`): the typestate before must give the summary's, and what
    follows starts in the one the summary leaves. -/
meta def vcLocal (k : VCKnow) (g : MVarId) (t p : Expr) : TacticM (Option (List (MVarId × VCKnow × Nat))) := do
  let env := t.getArg! 6
  let r ← whnf (p.getArg! 5)
  unless r.isAppOfArity ``LocalRef.mk 3 do return none
  unless (← whnf (r.getArg! 1)).isAppOf ``Option.none do return none
  let callee ← whnf (r.getArg! 2)
  unless callee.isAppOfArity ``Callee.local 1 do return none
  let i := callee.appArg!
  let some ss ← slotList (p.getArg! 6) | return none
  let mut kk := k
  let mut found := #[]
  for sl in ss do
    let some (c, pf, k') ← kk.lookup sl env | return none
    kk := k'
    found := found.push (sl, c, pf)
  let cs := found.toList.map (·.2.1)
  let vTy := mkConst ``V
  let natTy := mkConst ``Nat
  let csE ← mkListLit vTy cs
  let mut hsum := none
  for d in ← getLCtx do
    if d.isImplementationDetail then continue
    let ty ← instantiateMVars d.type
    unless ty.isAppOfArity ``Summary 5 do continue
    if ← withNewMCtxDepth (isDefEq (ty.getArg! 1) i <&&> isDefEq (ty.getArg! 2) csE) then
      hsum := some d.toExpr
      break
  let some hs := hsum | return none
  let (g, pfs, kb, nh) ← bindFound g kk env found
  g.withContext do
  let mut hcs := mkApp (mkConst ``lookup_nil) env
  for j in (List.range ss.length).reverse do
    hcs := mkAppN (mkConst ``lookup_cons)
      #[env, ss[j]!, cs[j]!, ← mkListLit natTy (ss.drop (j + 1)), ← mkListLit vTy (cs.drop (j + 1)), pfs[j]!, hcs]
  let gs ← applyNamed g ``wp_local_void
    [(`hres, ← mkEqRefl (mkConst ``Bool.false)), (`hcs, hcs), (`hsum, hs)]
  let [gS, gq] := gs | return none
  return some [(gS, kb, 0), (gq, kb, nh)]

/-- A slot fact restated for the slot as a program names it: the kernel
    computes the one to the other. -/
meta def atSlot (pf s : Expr) : MetaM Expr := do
  let some (_, lhs, rhs) := (← instantiateMVars (← inferType pf)).eq? | return pf
  unless lhs.isAppOfArity ``GetElem?.getElem? 7 do return pf
  mkExpectedTypeHint pf (← mkEq (mkAppN lhs.getAppFn (lhs.getAppArgs.set! 6 s)) rhs)

/-- What a branch on a handle compared with null teaches: in the arm where
    it is not null, a handle the typestate holds as opened is held. The arm,
    `true` for the one taken when the comparison holds, the typestate there,
    and the handle's kind and facts. -/
meta def refineOf? (k : VCKnow) (env w cc a b : Expr) :
    TacticM (Option (Bool × Expr × Expr × Expr × Expr × Expr × Expr)) := do
  let isNe := cc.isConstOf ``ICmpCond.ne
  unless isNe || cc.isConstOf ``ICmpCond.eq do return none
  let some (S, hW) ← stateAt? k w | return none
  let some (xa, pfa, k1) ← k.lookup a env | return none
  let some (_, pfb, _) ← k1.lookup b env | return none
  let some (_, x) ← scalarOf? xa | return none
  let (Sv, _) ← k.out S
  let some facts := Sv.listLit? | return none
  let mut kind := none
  for f in facts.2 do
    if f.isAppOfArity ``Contracts.Fact.opened 2 then
      if ← isDefEq (f.getArg! 1) x then
        kind := some (f.getArg! 0)
        break
  let some kd := kind | return none
  let holds := mkConst ``Contracts.TState.holds
  let held ← mkAppM ``Contracts.Fact.held #[kd, x]
  let R := mkApp2 holds (← mkAppM ``List.cons #[held, Sv]) w
  let hWv ← mkExpectedTypeHint hW (mkApp2 holds Sv w)
  return some (isNe, R, kd, ← atSlot pfa a, ← atSlot pfb b, hWv, Sv)

/-- The typestate that asks nothing. -/
meta def mkConstTrueFun : MetaM Expr :=
  withLocalDeclD `w (mkConst ``World) fun w => mkLambdaFVars #[w] (mkConst ``True)

/-- **A branch**: each arm from what is known here, the arm where a handle
    was found not null holding it, and what follows the join from what both
    arms end in. -/
meta def vcIte (k : VCKnow) (g : MVarId) (t p : Expr) (level : Nat := 2) (join? : Option Expr := none) :
    TacticM (Option (List (MVarId × VCKnow × Nat))) := do
  let env := t.getArg! 6
  let w := t.getArg! 7
  let c ← whnf (p.getArg! 4)
  let trueE := mkConst ``True
  let refine ← if c.isAppOfArity ``Cond.mk 6 then
      refineOf? k env w (← whnf (c.getArg! 2)) (c.getArg! 3) (c.getArg! 4)
    else pure none
  -- Otherwise a comparison of two values known as terms: each arm learns it.
  let cmp : Option (Expr × Expr × Expr × Expr × Expr × Expr) ←
    if refine.isSome || !c.isAppOfArity ``Cond.mk 6 then pure none else do
      let some (xa, pfa, _) ← k.lookup (c.getArg! 3) env | pure none
      let some (xb, pfb, _) ← k.lookup (c.getArg! 4) env | pure none
      let some (ta, a) ← scalarTerm? xa | pure none
      let some (tb, b) ← scalarTerm? xb | pure none
      pure (some (ta, tb, a, b, pfa, pfb))
  let cmpProp (ok : Bool) : MetaM Expr := do
    let some (ta, _, a, b, _, _) := cmp | pure trueE
    mkEq (mkApp4 (mkConst ``cmpInt) (← whnf (c.getArg! 2)) ta a b) (toExpr ok)
  let (Rt, Re) ← match refine with
    | some (true, R, _, _, _, _, _) => pure (R, trueE)
    | some (false, R, _, _, _, _, _) => pure (trueE, R)
    | none => pure (← cmpProp true, ← cmpProp false)
  -- What the join keeps: at level 4 the typestate the branch is entered in;
  -- at level 0 that less its cells when an arm stores; at level 1 only its
  -- parts and rooms, which an arm that closes a handle keeps too; at level 2
  -- nothing; `join?`, when given, is what both arms were found to end in.
  let hW? := if w.isFVar then k.holds[w.fvarId!]? else none
  let W0 ← if let some L := join? then pure (mkApp (mkConst ``Contracts.TState.holds) L) else
    match hW?, level with
    | some hW, 4 => pure (← loopState k env (← mkConstTrueFun) hW p (keepCells := true)).1
    | some hW, 0 => pure (← loopState k env (← mkConstTrueFun) hW p).1
    | some hW, 1 => do
        let T ← instantiateMVars (← inferType hW)
        match ← stateOf? T with
        | some S =>
            let pr ← withLocalDeclD `x (mkConst ``Contracts.Fact) fun x => do
              mkLambdaFVars #[x] (← mkAppM ``and #[← mkAppM ``Contracts.Fact.plain #[x],
                ← mkAppM ``not #[← mkAppM ``Contracts.Fact.isCell #[x]]])
            pure (mkApp (mkConst ``Contracts.TState.holds) (← stateOut (← mkAppM ``List.filter #[pr, S])))
        | none => mkConstTrueFun
    | _, _ => mkConstTrueFun
  -- under a post that allows no fault, the comparison answers: its operands
  -- are integers of one type
  let J := t.getArg! 2
  let lax ← faultAllowed g J
  -- in a first pass, which takes every post to allow a fault, the comparison
  -- is still given where its operands are known, and an arm is still walked
  -- where they are not
  let probe := (← getOptions).getBool `vcProbe false
  let strict : List (Name × Expr) ← if lax && !probe then pure [] else do
    let ab : Option (Expr × Expr) := match refine, cmp with
      | some (_, _, _, pfa, pfb, _, _), _ => some (pfa, pfb)
      | none, some (_, _, _, _, pfa, pfb) => some (pfa, pfb)
      | none, none => none
    let some (pfa, pfb) := ab | pure []
    let ok ← try some <$> mkAppM ``IcmpOk.of #[pfa, pfb, ← mkEqRefl (mkConst ``Bool.true)]
      catch _ => pure none
    let some ok := ok | pure []
    let l ← mkEq (← mkAppM ``Post.faultOk #[J]) (mkConst ``Bool.true)
    pure [(`hJ, mkApp3 (mkConst ``Or.inr) l (← inferType ok) ok)]
  -- where no fault is allowed, the values each arm joins with, of their types;
  -- and where each arm returns one 64-bit value known here, which it is
  -- the value an arm ends returning: followed through what it does first,
  -- by each construct's continuation, and known here only if computed here
  let rec finalRet (e : Expr) (fuel : Nat) : MetaM (Option Expr) := do
    if fuel == 0 then return none
    let e ← whnf e
    if e.isAppOfArity ``Prog.ret 4 then return some e.appArg!
    unless e.getAppFn.isConst && e.getAppNumArgs > 0 do return none
    let kont := e.appArg!
    if kont.isLambda then
      lambdaTelescope kont fun xs b => do
        let some r ← finalRet b (fuel - 1) | return none
        if xs.any (fun x => r.containsFVar x.fvarId!) then return none
        return some r
    else finalRet kont (fuel - 1)
  let armVal (arm : Expr) : MetaM (Option Expr) := do
    let some rv ← tryCatchRuntimeEx (finalRet arm 16) (fun _ => pure none) | return none
    let vs ← valsWhnf rv
    unless vs.isAppOfArity ``Vals.cons 5 do return none
    let some (c, _, _) ← k.lookup (vs.getArg! 3) env | return none
    let some (t, v) ← scalarTerm? c | return none
    unless t.isConstOf ``ClifTy.i64 do return none
    return some v
  let np := p.getAppNumArgs
  let joinA ← if lax then pure none else do
    let some a ← armVal (p.getArg! (np - 3)) | pure none
    let some b ← armVal (p.getArg! (np - 2)) | pure none
    let A ← withLocalDeclD `y (mkConst ``UInt64) fun y => do
      mkLambdaFVars #[y] (← mkAppM ``Or #[← mkAppM ``And #[Rt, ← mkEq y a], ← mkAppM ``And #[Re, ← mkEq y b]])
    pure (some A)
  let gs ← match joinA with
    | some A => applyNamed g ``wp_ite_vcJ ([(`W, W0), (`Rt, Rt), (`Re, Re), (`A, A),
        (`h0, ← mkDecideProof (← mkAppM ``LT.lt #[mkNatLit 0, ← mkAppM ``List.length #[p.getArg! 2]]))] ++ strict)
    | none => applyNamed g (if lax then ``wp_ite_vc else ``wp_ite_vcT) ([(`W, W0), (`Rt, Rt), (`Re, Re)] ++ strict)
  let (gRt, gRe, gT, gE, gK, extra) ← match gs with
    | [gRt, gRe, gT, gE, gK] => pure (gRt, gRe, gT, gE, gK, [])
    | [gRt, gRe, gT, gE, gK, gJ] => if probe then pure (gRt, gRe, gT, gE, gK, [gJ]) else return none
    | _ => return none
  let trivialTac ← `(tactic| exact refines_true)
  let mut rest := extra.toArray
  for (gR, isThen) in [(gRt, true), (gRe, false)] do
    match refine with
    | some (ne, _, kd, pfa, pfb, hWv, _) =>
        if ne == isThen then
          let hs ← applyNamed gR (if ne then ``refine_ne else ``refine_eq)
            [(`hc, ← mkEqRefl (mkConst (if ne then ``ICmpCond.ne else ``ICmpCond.eq))),
             (`ha, pfa), (`hb, pfb), (`hS, hWv), (`k, kd)]
          for h in hs do
            let ty ← instantiateMVars (← h.getType)
            let tac ← if ty.isAppOfArity ``Eq 3 then `(tactic| kdecide) else `(tactic| simp)
            rest := rest ++ (← closeWith h tac)
        else rest := rest ++ (← closeWith gR trivialTac)
    | none =>
        match cmp with
        | some (_, _, _, _, pfa, pfb) =>
            let gs ← applyNamed gR ``refine_cmp [(`ha, pfa), (`hb, pfb), (`ok, toExpr isThen)]
            rest := rest ++ gs.toArray
        | none => rest := rest ++ (← closeWith gR trivialTac)
  return some (rest.toList.map (·, k, 0) ++ [(gT, k, 0), (gE, k, 0), (gK, k, 0)])

/-- `Or.inr` of a proof that operation `oe` answers in goal `g`'s environment,
    from what is known of its operands: their values' types. -/
meta def evalsOk (k : VCKnow) (g : MVarId) (t oe : Expr) : TacticM (Option Expr) := g.withContext do
  let env := t.getArg! 6
  let mem ← mkAppM ``World.mem #[t.getArg! 7]
  let mut pfs : Array Expr := #[]
  for a in oe.getAppArgs do
    unless (← whnfR (← inferType a)).isConstOf ``Nat do continue
    if let some (_, pf, _) ← k.lookup a env then pfs := pfs.push pf
  if pfs.isEmpty then return none
  let ev ← mkAppM ``evalOp #[mem, env, oe]
  let vTy := (← whnfR (← inferType ev)).appArg!
  let ne ← mkAppM ``Ne #[ev, ← mkAppOptM ``Option.none #[vTy]]
  let m ← mkFreshExprMVar ne
  let st ← saveState
  let ok ← tryCatchRuntimeEx (do
      let tys ← pfs.mapM (fun e => Meta.inferType e)
      let (_, m'') ← m.mvarId!.assertHypotheses (← (pfs.zip tys).mapIdxM fun i (pf, ty) =>
        pure { userName := Name.mkSimple s!"hop{i}", type := ty, value := pf })
      let rest ← Term.withoutErrToSorry (Tactic.run m'' (withoutRecover <| evalTactic (← `(tactic| first
        | (simp only [evalOp, Sem.get, un, bin, zipIntCmp, ishiftOp, ibinOp, iunOp, Option.bind_eq_bind,
            Option.bind_some, *] <;> simp)
        | (simp [evalOp, Sem.get, *]; decide)))))
      let pf ← instantiateMVars m
      pure (rest.isEmpty && !pf.hasSorry && !pf.hasExprMVar)) (fun _ => pure false)
  unless ok do st.restore; return none
  let J := t.getArg! 2
  let l ← mkEq (← mkAppM ``Post.faultOk #[J]) (mkConst ``Bool.true)
  return some (mkApp3 (mkConst ``Or.inr) l ne (← instantiateMVars m))

/-- An operation over scalars `tyOp` types, where no fault is allowed: each
    operand's fact, the type `tyOp` names for their types, and `tyOp_sound`. -/
meta def opByType (k : VCKnow) (g : MVarId) (t oe : Expr) : TacticM (Option (List (MVarId × VCKnow × Nat))) :=
    g.withContext do
  let env := t.getArg! 6
  let mem ← mkAppM ``World.mem #[t.getArg! 7]
  let o ← whnfD oe
  let mut regs ← whnfD (← mkAppM ``Op.regs #[o])
  let mut tys : Array Expr := #[]
  let mut exs : Array Expr := #[]
  let mut k := k
  while regs.isAppOfArity ``List.cons 3 do
    let r := regs.getArg! 1
    regs ← whnfD (regs.getArg! 2)
    let some (c, pf, k') ← k.lookup (← normSlot r) env | return none
    k := k'
    let c ← whnfR c
    unless c.isAppOfArity ``V.sc 2 do return none
    let ta ← whnf (c.getArg! 0)
    unless ta.isConst do return none
    let x := c.getArg! 1
    let lam ← withLocalDeclD `x (mkConst ``UInt64) fun xv => do
      mkLambdaFVars #[xv] (← mkEq (← mkAppM ``GetElem?.getElem? #[env, r])
        (← mkAppM ``Option.some #[mkApp2 (mkConst ``V.sc) ta xv]))
    exs := exs.push (← mkAppOptM ``Exists.intro #[none, lam, x, ← mkExpectedTypeHint pf (lam.beta #[x])])
    tys := tys.push ta
  unless regs.isAppOf ``List.nil do return none
  let ts ← mkListLit (mkConst ``ClifTy) tys.toList
  let r ← whnfD (← mkAppM ``tyOp #[o, ts])
  unless r.isAppOfArity ``Option.some 2 do return none
  let T ← whnf r.appArg!
  let mut acc := mkConst ``True.intro
  for e in exs.reverse do
    acc ← mkAppM ``And.intro #[e, acc]
  let hargs ← mkExpectedTypeHint acc (← mkAppM ``ArgsTy #[env, ← mkAppM ``Op.regs #[o], ts])
  let hT ← mkExpectedTypeHint (← mkEqRefl (← mkAppM ``Option.some #[T]))
    (← mkEq (← mkAppM ``tyOp #[o, ts]) (← mkAppM ``Option.some #[T]))
  let hty ← mkAppOptM ``tyOp_sound #[mem, env, o, ts, T, hargs, hT]
  let hty ← mkExpectedTypeHint hty (← mkAppM ``Exists #[← withLocalDeclD `x (mkConst ``UInt64) fun xv => do
    mkLambdaFVars #[xv] (← mkEq (← mkAppM ``evalOp #[mem, env, oe])
      (← mkAppM ``Option.some #[mkApp2 (mkConst ``V.sc) T xv]))])
  let gs ← applyNamed g ``wp_op_ty [(`T, T), (`hty, hty)]
  return some (gs.map (·, k, 0))

/-- An operation over vectors, or that makes or reads one, where no fault is
    allowed: the type it answers, by `tyVOp` over its operands' types. -/
meta def opByTypeV (k : VCKnow) (g : MVarId) (t oe : Expr) : TacticM (Option (List (MVarId × VCKnow × Nat))) :=
    g.withContext do
  let env := t.getArg! 6
  let mem ← mkAppM ``World.mem #[t.getArg! 7]
  let o ← whnfD oe
  let mut regs ← whnfD (← mkAppM ``Op.regs #[o])
  let mut tys : Array Expr := #[]
  let mut pfs : Array Expr := #[]
  let mut k := k
  while regs.isAppOfArity ``List.cons 3 do
    let r := regs.getArg! 1
    regs ← whnfD (regs.getArg! 2)
    let some (c, pf, k') ← k.lookup (← normSlot r) env | return none
    k := k'
    let some (ty, hv) ← k.tyV? c | return none
    let ty ← whnf ty
    unless ty.isConst do return none
    let some pa ← (try some <$> mkAppM ``tyAt_of #[hv, pf] catch _ => pure none) | return none
    pfs := pfs.push pa
    tys := tys.push ty
  unless regs.isAppOf ``List.nil do return none
  let ts ← mkListLit (mkConst ``ClifTy) tys.toList
  let r ← whnfD (← mkAppM ``tyVOp #[o, ts])
  unless r.isAppOfArity ``Option.some 2 do return none
  let T ← whnf r.appArg!
  let mut acc := mkConst ``True.intro
  for e in pfs.reverse do
    acc ← mkAppM ``And.intro #[e, acc]
  let hargs ← mkExpectedTypeHint acc (← mkAppM ``ArgsV #[env, ← mkAppM ``Op.regs #[o], ts])
  let hT ← mkExpectedTypeHint (← mkEqRefl (← mkAppM ``Option.some #[T]))
    (← mkEq (← mkAppM ``tyVOp #[o, ts]) (← mkAppM ``Option.some #[T]))
  let hty ← mkAppOptM ``tyVOp_sound #[mem, env, o, ts, T, hargs, hT]
  let hty ← mkExpectedTypeHint hty (← mkAppM ``Exists #[← withLocalDeclD `v (mkConst ``V) fun vv => do
    mkLambdaFVars #[vv] (← mkAppM ``And #[← mkEq (← mkAppM ``evalOp #[mem, env, oe]) (← mkAppM ``Option.some #[vv]),
      ← mkAppM ``TyV #[T, vv]])])
  let lanes ← whnfD (← mkAppM ``ClifTy.lanes #[T])
  -- a scalar out of a vector: bound as a scalar, so the operations over it answer
  let gs ← if lanes.isAppOf ``Option.none then do
      let hl ← mkExpectedTypeHint (← mkEqRefl lanes) (← mkEq (← mkAppM ``ClifTy.lanes #[T]) lanes)
      applyNamed g ``wp_op_ty [(`T, T), (`hty, ← mkAppM ``tyV_sc_ex #[hl, hty])]
    else applyNamed g ``wp_op_tyV [(`T, T), (`hty, hty)]
  return some (gs.map (·, k, 0))

/-- A vector load from a known address, where no fault is allowed: its bytes
    fit, and it answers a vector of its type. -/
meta def vcVLoadAt (k : VCKnow) (g : MVarId) (t oe ho : Expr) :
    TacticM (Option (List (MVarId × VCKnow × Nat))) := do
  let J := t.getArg! 2
  if ← faultAllowed g J then return none
  let op := oe.getArg! 0
  let kind ← whnfD (← mkAppM ``LoadOp.kind #[op])
  unless kind.isConstOf ``LoadKind.plain do return none
  let hk ← mkExpectedTypeHint (← mkEqRefl kind) (← mkEq (← mkAppM ``LoadOp.kind #[op]) kind)
  let opTy ← mkAppM ``LoadOp.ty #[op]
  let lanes ← evalOut (← mkAppM ``ClifTy.lanes #[opTy])
  unless lanes.isAppOfArity ``Option.some 2 do return none
  let pr ← whnf lanes.appArg!
  unless pr.isAppOfArity ``Prod.mk 4 do return none
  let hl ← mkExpectedTypeHint (← mkEqRefl lanes) (← mkEq (← mkAppM ``ClifTy.lanes #[opTy]) lanes)
  let env := t.getArg! 6
  let some (xa, pfa, k) ← k.lookup (oe.getArg! 1) env | return none
  let some (ta, ba) ← scalarTerm? xa | return none
  let nb ← mkAppM ``HMul.hMul #[pr.getArg! 3, ← mkAppM ``tyBytes #[pr.getArg! 2]]
  let hJT ← mkAppM ``Or #[← mkEq (← mkAppM ``Post.faultOk #[J]) (mkConst ``Bool.true),
    ← mkAppM ``Fits #[← mkAppM ``World.mem #[t.getArg! 7], ba, nb]]
  let hJ ← mkFreshExprMVar hJT
  unless ← closeFits hJ.mvarId! do return none
  let gs ← applyNamed g ``wp_op_vload_at [(`ho, ho), (`hk, hk), (`hl, hl), (`t', ta), (`x, ba), (`ha, pfa),
    (`hJ, ← instantiateMVars hJ)]
  return some (gs.map (·, k, 0))

/-- An operation over scalars, where no fault is allowed: the type of scalar it
    answers, among its operands' types, `i8` and the types it names, with that it
    answers, both by computing it from what is known of the operands. -/
meta def opTyped (k : VCKnow) (g : MVarId) (t oe : Expr) : TacticM (Option (List (MVarId × VCKnow × Nat))) :=
    g.withContext do
  -- its slots as the facts name them: a carry by its place, not its projection
  let oe ← do
    let args ← oe.getAppArgs.mapM fun a => do
      if (← whnfR (← inferType a)).isConstOf ``Nat then normSlot a else pure a
    pure (mkAppN oe.getAppFn args)
  let env := t.getArg! 6
  let mem ← mkAppM ``World.mem #[t.getArg! 7]
  let mut pfs : Array Expr := #[]
  let mut cands : Array Expr := #[]
  for a in oe.getAppArgs do
    let aty ← whnfR (← inferType a)
    if aty.isConstOf ``ClifTy then cands := cands.push (← whnf a); continue
    unless aty.isConstOf ``Nat do continue
    let some (c, pf, _) ← k.lookup a env | return none
    pfs := pfs.push pf
    if let some (ta, _) ← scalarTerm? c then cands := cands.push ta
  cands := cands.push (mkConst ``ClifTy.i8)
  let ev ← mkAppM ``evalOp #[mem, env, oe]
  let vTy := (← whnfR (← inferType ev)).appArg!
  -- each operand's fact with its bits hidden: only its type is asked about, and
  -- the bits of a float constant do not compute
  let hid ← pfs.mapM fun pf => do
    let ty ← whnfR (← inferType pf)
    let some (_, lhs, rhs) := ty.eq? | return (pf, ty, false)
    let c ← whnf rhs.appArg!
    unless c.isAppOfArity ``V.sc 2 do return (pf, ty, false)
    let lam ← withLocalDeclD `x (mkConst ``UInt64) fun x => do
      mkLambdaFVars #[x] (← mkEq lhs (← mkAppM ``Option.some #[mkApp2 (mkConst ``V.sc) (c.getArg! 0) x]))
    let exT ← mkAppM ``Exists #[lam]
    let pf' ← mkAppOptM ``Exists.intro #[none, lam, c.getArg! 1, ← mkExpectedTypeHint pf (lam.beta #[c.getArg! 1])]
    return (pf', exT, true)
  let run (goal : Expr) (tac : TSyntax `tactic) : TacticM (Option Expr) := do
    let m ← mkFreshExprMVar goal
    let st ← saveState
    let ok ← tryCatchRuntimeEx (do
        let (fvs, m') ← m.mvarId!.assertHypotheses (hid.mapIdx fun i (pf, ty, e) =>
          { userName := Name.mkSimple (if e then s!"hex{i}" else s!"hop{i}"), type := ty, value := pf })
        -- a proof of its own: its context may be taken apart
        let mut m' := m'
        for (_, _, e) in hid, fv in fvs, i in [0:hid.size] do
          if e then
            let #[sg] ← m'.cases fv | throwError "opTyped: cases"
            let #[_, h] := sg.fields | throwError "opTyped: fields"
            m' ← sg.mvarId.rename h.fvarId! (Name.mkSimple s!"hop{i}")
        let rest ← Term.withoutErrToSorry (Tactic.run m' (withoutRecover <| evalTactic tac))
        let pf ← instantiateMVars m
        pure (rest.isEmpty && !pf.hasSorry && !pf.hasExprMVar)) (fun _ => pure false)
    unless ok do st.restore; return none
    return some (← instantiateMVars m)
  let ne ← mkAppM ``Ne #[ev, ← mkAppOptM ``Option.none #[vTy]]
  -- the simp set: the semantics, and the operands' facts by name
  let lemma (n : Name) : TSyntax `Lean.Parser.Tactic.simpLemma :=
    ⟨mkNode ``Lean.Parser.Tactic.simpLemma #[mkNullNode, mkNullNode, mkIdent n]⟩
  let args : Array (TSyntax `Lean.Parser.Tactic.simpLemma) :=
    #[lemma ``evalOp, lemma ``Sem.get, lemma ``zipF, lemma ``zipBits, lemma ``zipBitsIf, lemma ``boolV,
      lemma ``ClifTy.isInt, lemma ``ClifTy.isFloat]
      ++ (List.range pfs.size).toArray.map (fun i => lemma (Name.mkSimple s!"hop{i}"))
  let some hne ← run ne (← `(tactic| simp [$args,*])) | return none
  let J := t.getArg! 2
  let l ← mkEq (← mkAppM ``Post.faultOk #[J]) (mkConst ``Bool.true)
  let hJ := mkApp3 (mkConst ``Or.inr) l ne hne
  for T in cands do
    let hscT ← withLocalDeclD `v vTy fun v => do
      let lhs ← mkEq ev (← mkAppM ``Option.some #[v])
      let rhs ← withLocalDeclD `x (mkConst ``UInt64) fun x => do
        mkLambdaFVars #[x] (← mkEq v (mkApp2 (mkConst ``V.sc) T x))
      mkForallFVars #[v] (← mkArrow lhs (← mkAppM ``Exists #[rhs]))
    let tac ← `(tactic| intro v hv <;> simp [$args,*] at hv <;>
      first | exact ⟨_, hv.symm⟩ | exact ⟨_, hv.2.symm⟩ | exact ⟨_, hv.1.symm⟩)
    let some hsc ← run hscT tac | continue
    let gs ← applyNamed g ``wp_op_scT [(`T, T), (`hsc, hsc), (`hJ, hJ)]
    return some (gs.map (·, k, 0))
  return none

/-- A goal about the types of values: `SlotsOk` of slots written out, `IcmpOk`
    of two slots, a loop's `GuardOk`, or `J.faultOk = true ∨` one of them
    where no fault is allowed. The goals left, or `none`. -/
meta partial def closeTyped (k : VCKnow) (g : MVarId) (t : Expr) :
    TacticM (Option (List (MVarId × VCKnow × Nat))) := g.withContext do
  if t.isAppOfArity ``Or 2 then
    let r := t.getArg! 1
    unless r.isAppOf ``SlotsOk || r.isAppOf ``IcmpOk || r.isAppOf ``GuardOk do return none
    let gr ← mkFreshExprMVar r
    g.assign (mkApp3 (mkConst ``Or.inr) (t.getArg! 0) r gr)
    return some [(gr.mvarId!, k, 0)]
  if t.isAppOf ``GuardOk then
    let args := t.getAppArgs
    let gi ← whnfR args[args.size - 3]!
    if gi.isAppOf ``Option.none then g.assign (mkConst ``True.intro); return some []
    unless gi.isAppOfArity ``Option.some 2 do return none
    let some n ← (evalNat (← instantiateMVars gi.appArg!)).run | return none
    -- the guard's slot, read off the initial values written out
    let mut vals ← whnfR args[args.size - 2]!
    for _ in [0:n] do
      unless vals.isAppOf ``Vals.cons do return none
      vals ← whnfR vals.appArg!
    unless vals.isAppOf ``Vals.cons do return none
    let a := vals.getArg! (vals.getAppNumArgs - 2)
    let g' ← g.replaceTargetDefEq (← mkAppM ``IcmpOk #[args[0]!, a, args[args.size - 1]!])
    return some [(g', k, 0)]
  if t.isAppOfArity ``IcmpOk 3 then
    let env := t.getArg! 0
    -- a slot the head names through its test, `(cond, exits, x).1.a`: the slot
    let look (k : VCKnow) (a : Expr) := do
      if let some r ← k.lookup a env then return some r
      let some a' ← projField? a | return none
      k.lookup a' env
    let some (_, pfa, k) ← look k (t.getArg! 1) | return none
    let some (_, pfb, _) ← look k (t.getArg! 2) | return none
    let some pf ← (try some <$> mkAppM ``IcmpOk.of #[pfa, pfb, ← mkEqRefl (mkConst ``Bool.true)]
      catch _ => pure none) | return none
    unless ← isDefEq (← inferType pf) t do return none
    g.assign pf; return some []
  -- the values a jump carries, of their types: from where it read them
  if t.isAppOfArity ``CarryOk 2 then
    let vs := t.getArg! 1
    for d in ← getLCtx do
      if d.isImplementationDetail then continue
      let ty ← instantiateMVars d.type
      let some (_, lhs, rhs) := ty.eq? | continue
      unless rhs.isAppOfArity ``Option.some 2 && rhs.appArg! == vs do continue
      unless lhs.isAppOf ``List.mapM do continue
      let sl := lhs.appArg!
      unless sl.isAppOf ``Vals.slots do continue
      let f := lhs.appFn!.appArg!
      unless f.isLambda do continue
      let body := f.bindingBody!
      unless body.isAppOfArity ``GetElem?.getElem? 7 do continue
      let env := body.getArg! 5
      if env.hasLooseBVars then continue
      let vals := sl.appArg!
      let gS ← mkFreshExprMVar (← mkAppM ``SlotsOk #[env, vals])
      let pf ← try mkAppM ``SlotsOk.carry #[gS, d.toExpr] catch _ => continue
      unless ← isDefEq (← inferType pf) t do continue
      g.assign pf
      return some [(gS.mvarId!, k, 0)]
    return none
  -- the value a loop's head hands its body, of its type
  if t.isAppOfArity ``CarryAt 3 then
    let env := t.getArg! 0
    let l ← whnf (t.getArg! 2)
    unless l.isAppOfArity ``List.cons 3 && (← whnf (l.getArg! 2)).isAppOfArity ``List.nil 1 do return none
    let some (c, pf, k) ← k.lookup (t.getArg! 1) env | return none
    let some (_, hv) ← k.tyV? c | return none
    let some pf' ← (try some <$> (do mkAppM ``carryAt_one #[← mkAppM ``tyAt_of #[hv, pf]])
      catch _ => pure none) | return none
    unless ← isDefEq (← inferType pf') t do return none
    g.assign pf'; return some []
  if t.isAppOf ``SlotsOk then
    let env := t.getArg! 0
    let vals ← whnf t.appArg!
    if vals.isAppOf ``Vals.nil then
      g.assign (← mkAppOptM ``SlotsOk.nil #[env]); return some []
    unless vals.isAppOf ``Vals.cons do return none
    -- a slot a function of the place gives, as `Vals.ofFn` puts it: applied
    let mut sl := vals.getArg! (vals.getAppNumArgs - 2)
    while sl.isHeadBetaTarget do sl := sl.headBeta
    let rest := vals.appArg!
    let some (c, pf, k) ← k.lookup sl env | return none
    let some (_, hv) ← k.tyV? c | return none
    let gr ← mkFreshExprMVar (← mkAppM ``SlotsOk #[env, rest])
    let some pf' ← (try some <$> mkAppOptM ``SlotsOk.cons #[none, none, none, sl, rest, none, pf, hv, gr]
      catch _ => pure none) | return none
    unless ← isDefEq (← inferType pf') t do return none
    g.assign pf'
    return some [(gr.mvarId!, k, 0)]
  return none

/-- A loop body's carries of their types, `CarryAt Γ n tys` proved by `pf`:
    one fact per carry, `Γ[n + i]? = some (.sc t x)`, its value named by
    `Classical.choose` so the context is left as it is. -/
meta partial def carryFacts (k : VCKnow) (pf : Expr) : MetaM VCKnow := do
  let T ← whnfD (← instantiateMVars (← inferType pf))
  unless T.isAppOfArity ``And 2 do return k
  let ex ← whnfD (T.getArg! 0)
  unless ex.isAppOfArity ``Exists 2 do return k
  let hex ← mkExpectedTypeHint (← mkAppM ``And.left #[pf]) ex
  let hx ← mkAppM ``Classical.choose_spec #[hex]
  let fact ← whnfR (← inferType hx)
  -- a vector's: its slot, and its lanes
  let (fact, hx, hn?) ← if fact.isAppOfArity ``And 2 then
      pure (← whnfR (fact.getArg! 0), ← mkAppM ``And.left #[hx], some (← mkAppM ``And.right #[hx]))
    else pure (fact, hx, none)
  let some (_, lhs, rhs) := fact.eq? | return k
  unless lhs.isAppOfArity ``GetElem?.getElem? 7 && rhs.isAppOfArity ``Option.some 2 do return k
  let c := rhs.appArg!
  let mut k := (k.addFact (← normSlot (lhs.getArg! 6)) (lhs.getArg! 5, c, hx))
  if let some hn := hn? then
    let c' ← whnfR c
    if c'.isAppOfArity ``V.vec 2 then
      let t := c'.getArg! 0
      let l ← whnfD (← mkAppM ``ClifTy.lanes #[t])
      if l.isAppOfArity ``Option.some 2 then
        let hl ← mkExpectedTypeHint (← mkEqRefl l) (← mkEq (← mkAppM ``ClifTy.lanes #[t]) l)
        if let some pv ← (try some <$> mkAppM ``TyV.vec #[hl, hn] catch _ => pure none) then
          k := { k with tys := k.tys.insert c (t, pv) }
  carryFacts k (← mkAppM ``And.right #[pf])

/-- The elements of a list, its spine put as constructors. -/
meta def listElems? (l : Expr) : MetaM (Option (List Expr)) := do
  let mut l ← whnf l
  let mut out := #[]
  for _ in [0:256] do
    if l.isAppOfArity ``List.nil 1 then return some out.toList
    unless l.isAppOfArity ``List.cons 3 do return none
    out := out.push (l.getArg! 1).headBeta
    l ← whnf (l.getArg! 2)
  return none

/-- What `SameAt` or `SameFrom` says: each slot of the new environment holds
    what the slot of the old one it names holds, as known there. -/
meta def sameFacts (k : VCKnow) (h : Expr) : MetaM VCKnow := do
  let T ← instantiateMVars (← inferType h)
  if T.isAppOfArity ``SameAt 4 then
    let Γ' := T.getArg! 0
    let Γ1 := T.getArg! 2
    let some rs ← listElems? (T.getArg! 3) | return k
    let mut k := k
    let mut hc ← mkExpectedTypeHint h
      (mkApp4 (mkConst ``SameAt) Γ' (T.getArg! 1) Γ1 (← mkListLit (mkConst ``Nat) rs))
    let mut n := T.getArg! 1
    for r in rs do
      let hd ← mkAppM ``sameAt_head #[hc]
      if let some (c, pf, k') ← k.lookup r Γ1 then
        if let some pf' ← (try some <$> mkAppM ``Eq.trans #[hd, pf] catch _ => pure none) then
          k := (k'.addFact (← normSlot n) (Γ', c, pf'))
      hc ← mkAppM ``sameAt_tail #[hc]
      n ← mkAppM ``HAdd.hAdd #[n, mkNatLit 1]
    return k
  if T.isAppOfArity ``SameFrom 5 then
    let Γb := T.getArg! 0
    let Γh := T.getArg! 2
    let some ts ← listElems? (T.getArg! 4) | return k
    let mut k := k
    let mut hc ← mkExpectedTypeHint h
      (mkApp5 (mkConst ``SameFrom) Γb (T.getArg! 1) Γh (T.getArg! 3) (← mkListLit (mkConst ``ClifTy) ts))
    let mut nb := T.getArg! 1
    let mut n0 := T.getArg! 3
    for _ in ts do
      let hd ← mkAppM ``sameFrom_head #[hc]
      if let some (c, pf, k') ← k.lookup n0 Γh then
        if let some pf' ← (try some <$> mkAppM ``Eq.trans #[hd, pf] catch _ => pure none) then
          k := (k'.addFact (← normSlot nb) (Γb, c, pf'))
      hc ← mkAppM ``sameFrom_tail #[hc]
      nb ← mkAppM ``HAdd.hAdd #[nb, mkNatLit 1]
      n0 ← mkAppM ``HAdd.hAdd #[n0, mkNatLit 1]
    return k
  return k

/-- What a test's outcome tells of its operands, `CondOutR`: the operands
    looked up, a goal for each outcome, the comparison of their values given. -/
meta def closeCondOut (k : VCKnow) (g : MVarId) (t : Expr) :
    TacticM (Option (List (MVarId × VCKnow × Nat))) := g.withContext do
  let t ← if t.isAppOf ``CondOut then whnfR t else pure t
  unless t.isAppOfArity ``CondOutR 7 do return none
  let env := t.getArg! 0
  let look (k : VCKnow) (a : Expr) : MetaM (Option (Expr × Expr × VCKnow)) := do
    if let some r ← k.lookup a env then return some r
    let some a' ← projField? a | return none
    k.lookup a' env
  let some (ca, pfa, k) ← look k (t.getArg! 2) | return none
  let some (cb, pfb, k) ← look k (t.getArg! 3) | return none
  let some (ta, xa) ← scalarTerm? ca | return none
  let some (_, yb) ← scalarTerm? cb | return none
  let cc ← whnf (t.getArg! 1)
  let e ← whnf (t.getArg! 4)
  let ne ← whnf (mkApp (mkConst ``not) e)
  let cmp := mkAppN (mkConst ``cmpInt) #[cc, ta, xa, yb]
  let g1 ← mkFreshExprMVar (← mkArrow (← mkEq cmp ne) (t.getArg! 5))
  let g2 ← mkFreshExprMVar (← mkArrow (← mkEq cmp e) (t.getArg! 6))
  let restate (pf a x : Expr) : MetaM Expr := do
    let some (_, lhs, _) := (← instantiateMVars (← inferType pf)).eq? | throwError "closeCondOut: not a slot"
    let lhs := mkAppN lhs.getAppFn (lhs.getAppArgs.set! 6 a)
    mkExpectedTypeHint pf (← mkEq lhs (← mkAppM ``Option.some #[mkApp2 (mkConst ``V.sc) ta x]))
  let ha ← restate pfa (t.getArg! 2) xa
  let hb ← restate pfb (t.getArg! 3) yb
  let some pf ← (try some <$> mkAppOptM ``condOutR_of #[env, cc, t.getArg! 2, t.getArg! 3, e, t.getArg! 5,
      t.getArg! 6, ta, xa, yb, ha, hb, g1, g2] catch _ => pure none) | return none
  unless ← isDefEq (← inferType pf) t do return none
  g.assign pf
  return some [(g1.mvarId!, k, 0), (g2.mvarId!, k, 0)]

/-- One step of the condition generator on one goal: `none` when no rule
    applies. -/
meta def vcStep (W : Expr) (k : VCKnow) (g : MVarId) : TacticM (Option (List (MVarId × VCKnow × Nat))) :=
  g.withContext do
  let t := (← instantiateMVars (← g.getType)).headBeta
  if t.isForall then
    -- All the binders at once: one binder group, not one per hypothesis.
    let (hs, g') ← g.intros
    -- a property given as a function, applied: what it says of its argument
    let mut g' := g'
    for h in hs do
      let ty ← g'.withContext do instantiateMVars (← h.getType)
      if ty.isHeadBetaTarget then g' ← g'.replaceLocalDeclDefEq h ty.headBeta
    -- a loop body's carries: one fact per carry; the slots one environment
    -- holds as another does: the other's facts
    let k' ← g'.withContext (hs.foldlM (fun k h => do
      let T ← instantiateMVars (← h.getType)
      if T.isAppOf ``CarryAt then carryFacts k (mkFVar h)
      else if T.isAppOf ``SameAt || T.isAppOf ``SameFrom then sameFacts k (mkFVar h)
      else k.learn W h) k)
    return some [(g', k', hs.size)]
  let some p := wpProg? t | do
    if t.isConstOf ``True then
      g.assign (mkConst ``True.intro); return some []
    -- a first pass asks nothing of addresses or numbers: those goals stay
    if (← getOptions).getBool `vcProbe false then
      if t.isAppOf ``Or || t.isAppOf ``Fits || t.isAppOf ``LE.le || t.isAppOf ``LT.lt ||
          t.isAppOf ``Readable || t.isAppOf ``Writable || t.isAppOf ``Not then return none
      if let some (ty, _, _) := t.eq? then
        if ty.isConstOf ``Nat || ty.isConstOf ``Bool then return none
    -- that a fault is allowed where the program runs, or cannot happen
    if ← closeFault g then return some []
    -- that a fault is allowed, where it is not: nothing else proves it
    if let some (_, a, _) := t.eq? then
      if a.isAppOfArity ``Post.faultOk 1 then return none
    if ← closeFits g then return some []
    -- what a loop's trip ends in: its value, as the context has it
    if t.isAppOfArity ``Exists 2 then
      let st ← saveState
      let rest ← try Tactic.run g (withoutRecover <| evalTactic (← `(tactic| exact ⟨_, by assumption⟩)))
        catch _ => pure [g]
      if rest.isEmpty then return some []
      st.restore
    -- a loop's guard, and the values a trip passes back, of their types
    if let some r ← closeTyped k g t then return some r
    -- the counter a jump out of a loop carries: from the slot it carries it from
    if t.isAppOfArity ``CtrOk 3 then
      let vs := t.getArg! 0
      let mut hvs? := none
      for d in ← getLCtx do
        if d.isImplementationDetail then continue
        let ty ← instantiateMVars d.type
        if let some (_, l, r) := ty.eq? then
          if r.isAppOfArity ``Option.some 2 && r.appArg! == vs && l.isAppOf ``List.mapM then hvs? := some d.toExpr
      if let some hvs := hvs? then
        let gs ← applyNamed g ``ctrOk_of_mapM [(`hvs, hvs)]
        let mut out := []
        for g' in gs do
          let ty ← instantiateMVars (← g'.getType)
          if ty.isAppOfArity ``LT.lt 4 then
            unless (← closeWith g' (← `(tactic| first | decide | simp [Vals.slots_length]))).isEmpty do return none
          else out := out ++ [(g', k, 0)]
        return some out
    -- what a loop's test tells of its operands
    if t.isAppOf ``CondOut || t.isAppOf ``CondOutR then
      if let some r ← closeCondOut k g t then return some r
    -- what a loop's head ends in: an environment keeping the one it began in,
    -- and the typestate
    if t.isAppOfArity ``And 2 then
      let gl ← mkFreshExprMVar (t.getArg! 0)
      let gr ← mkFreshExprMVar (t.getArg! 1)
      g.assign (← mkAppM ``And.intro #[gl, gr])
      return some [(gl.mvarId!, k, 0), (gr.mvarId!, k, 0)]
    -- where a jump goes: what the loop it leaves or returns to asks
    if t.isAppOf ``Post.brk || t.isAppOf ``Post.cont then
      let t' ← whnf t
      if t' != t then return some [(← g.replaceTargetDefEq t', k, 0)]
    if let some (ty, a, b) := t.eq? then
      if a == b then g.assign (← mkEqRefl a); return some []
      -- arithmetic on numbers: a counter's invariant
      if ty.isConstOf ``Nat then
        if !t.hasFVar then
          if let some pr ← decideTrue? t then g.assign pr; return some []
        if (← closeWith g (← `(tactic| bound_close))).isEmpty then return some []
    if t.isAppOfArity ``Ext 2 then
      let some pf ← k.extProof (t.getArg! 0) (t.getArg! 1) | return none
      g.assign pf; return some []
    -- a value a slot holds, with what is asked of it
    if t.isAppOfArity ``Exists 2 then
      let lam := t.appArg!
      if lam.isLambda && lam.bindingBody!.isAppOfArity ``And 2 then
        let c1 := lam.bindingBody!.getArg! 0
        if let some (_, lhs, _) := c1.eq? then
          if lhs.isAppOfArity ``GetElem?.getElem? 7 then
            let env := lhs.getArg! 5
            let sl := lhs.getArg! 6
            if !env.hasLooseBVars && !sl.hasLooseBVars then
              if let some (c, pf, k') ← k.lookup sl env then
                if let some (_, b) ← scalarTerm? c then
                  let gP ← mkFreshExprMVar ((lam.bindingBody!.getArg! 1).instantiate1 b)
                  let pf' ← mkExpectedTypeHint pf (c1.instantiate1 b)
                  g.assign (← mkAppOptM ``Exists.intro #[none, lam, b, ← mkAppM ``And.intro #[pf', gP]])
                  return some [(gP.mvarId!, k', 0)]
    -- which arm a joined value came from: its test, and the value
    if t.isAppOfArity ``Or 2 && (t.getArg! 0).isAppOfArity ``And 2 then
      for (side, inl) in [(t.getArg! 0, true), (t.getArg! 1, false)] do
        unless side.isAppOfArity ``And 2 do continue
        let some (_, l, r) := (side.getArg! 1).eq? | continue
        unless l == r do continue
        let test := side.getArg! 0
        let some h ← (do
            if test.isConstOf ``True then return some (mkConst ``True.intro)
            for d in ← getLCtx do
              if d.isImplementationDetail then continue
              if (← instantiateMVars d.type) == test then return some d.toExpr
            return none) | continue
        let pr ← mkAppM ``And.intro #[h, ← mkEqRefl l]
        g.assign (if inl then mkApp3 (mkConst ``Or.inl) (t.getArg! 0) (t.getArg! 1) pr
          else mkApp3 (mkConst ``Or.inr) (t.getArg! 0) (t.getArg! 1) pr)
        return some []
    -- a disjunction of comparisons of numbers: linear arithmetic takes it whole
    if t.isAppOfArity ``Or 2 && ((t.getArg! 0).isAppOfArity ``LE.le 4 || (t.getArg! 0).isAppOfArity ``LT.lt 4) then
      -- a side the kernel computes true
      for (sd, inl) in [(t.getArg! 1, false), (t.getArg! 0, true)] do
        unless sd.hasFVar do
          if let some pr ← decideTrue? sd then
            g.assign (if inl then mkApp3 (mkConst ``Or.inl) (t.getArg! 0) (t.getArg! 1) pr
              else mkApp3 (mkConst ``Or.inr) (t.getArg! 0) (t.getArg! 1) pr)
            return some []
      if (← closeWith g (← `(tactic| bound_close))).isEmpty then return some []
    if t.isAppOfArity ``LE.le 4 || t.isAppOfArity ``LT.lt 4 then
      if !t.hasFVar then
        if let some pr ← decideTrue? t then g.assign pr; return some []
      if (← closeWith g (← `(tactic| bound_close))).isEmpty then return some []
    if t.isApp && t.appArg!.isFVar then
      if let some h := k.holds[t.appArg!.fvarId!]? then
        let T ← instantiateMVars (← inferType h)
        if ← isDefEq t T then
          g.assign h; return some []
        -- A loop's typestate from what the body ended in: it names less.
        if let (some S, some S') := (← stateOf? t, ← stateOf? T) then
          if let some hsub ← decideTrue? (← mkEq (← mkAppM ``TState.sub #[S, S']) (mkConst ``Bool.true)) then
            g.assign (← mkAppM ``TState.holds_sub #[hsub, h]); return some []
          -- facts that name a value: each found by place
          if let (some Sv, some S'v) := (← kernelList S, ← kernelList S') then
            if let (some (α, xs), some (_, ys)) := (Sv.listLit?, S'v.listLit?) then
              let mut acc := mkConst ``True.intro
              let mut ok := true
              for x in xs.reverse do
                let some m ← memLit? α ys (fun y => isDefEq y x) | ok := false; break
                acc ← mkAppM ``And.intro #[m, acc]
              if ok then
                let hall ← mkExpectedTypeHint acc (← mkAppM ``TState.AllIn #[S', S])
                g.assign (← mkAppM ``TState.holds_allIn #[hall, h]); return some []
          return none
    if (← g.assumptionCore) then return some []
    return none
  let p := (← instantiateMVars p).headBeta
  -- the environment a jump out of a loop leaves: named, keeping the loop's,
  -- its exit slots holding what the jump carried
  let envE := t.getArg! 6
  if envE.isAppOfArity ``bindAt 3 then
    let Γb := envE.getArg! 0
    let nE := envE.getArg! 1
    let vs := envE.getArg! 2
    let mut hvs? := none
    let mut hle? := none
    for d in ← getLCtx do
      if d.isImplementationDetail then continue
      let ty ← instantiateMVars d.type
      if let some (_, l, r) := ty.eq? then
        if r.isAppOfArity ``Option.some 2 && r.appArg! == vs && l.isAppOf ``List.mapM then hvs? := some d.toExpr
      if ty.isAppOfArity ``LE.le 4 && ty.getArg! 3 == nE && (ty.getArg! 2).isAppOfArity ``Array.size 2 then
        hle? := some (d.toExpr, (ty.getArg! 2).appArg!)
    if let (some hvs, some (hle, Γ)) := (hvs?, hle?) then
      if let some he ← k.extProof Γ Γb then
        let gs ← applyNamed g ``wp_bindAt_vc [(`he, he), (`hle, hle), (`hvs, hvs)]
        return some (gs.map (·, k, 0))
  -- a function applied where it is written, under an annotation
  let fn := p.getAppFn.consumeMData
  if fn.isLambda && p.getAppNumArgs > 0 then
    let p' := fn.betaRev p.getAppRevArgs
    return some [(← g.replaceTargetDefEq (mkAppN t.getAppFn (t.getAppArgs.set! 4 p')), k, 0)]
  -- a value the body names: the body with the value in its place
  if p.isLet then
    let p' := p.letBody!.instantiate1 p.letValue!
    return some [(← g.replaceTargetDefEq (mkAppN t.getAppFn (t.getAppArgs.set! 4 p')), k, 0)]
  let ap (n : Name) : TacticM (Option (List (MVarId × VCKnow × Nat))) := do
    return some ((← applyNamed g n []).map (·, k, 0))
  match p.getAppFn.constName? with
  | some ``Bind.bind => ap ``wp_bind_of
  | some ``Pure.pure => ap ``wp_pure_of
  | some ``Prog.ret => ap ``wp_ret_of
  | some ``Prog.params => ap ``wp_params_of
  | some ``Prog.op =>
      if (p.getArg! 4).isAppOf ``Op'.iconst then
        -- a constant computed by a term (a string's length, say) is put as the
        -- number it computes: reducing the term here would walk it out
        let kE := (p.getArg! 4).getArg! 2
        if kE.int?.isSome || kE.nat?.isSome || kE.hasFVar || kE.hasMVar then ap ``wp_iconst_var
        else
          let kv ← try some <$> evalInt kE catch _ => pure none
          match kv with
          | none => ap ``wp_iconst_var
          | some v =>
              let k' := toExpr v
              let eqT ← mkEq kE k'
              -- the kernel does not compute a string's size: that one is
              -- checked natively
              let he ← if ← kernelSlow kE then
                  let m ← mkFreshExprMVar eqT
                  let rest ← Tactic.run m.mvarId! (withoutRecover <| evalTactic (← `(tactic| native_decide)))
                  unless rest.isEmpty do throwError "vcStep: constant {kE} is not {v}"
                  instantiateMVars m
                else mkExpectedTypeHint (← mkEqRefl k') eqT
              return some ((← applyNamed g ``wp_iconst_lit [(`k', k'), (`he, he)]).map (·, k, 0))
      else match ← vcOpVal k g t p with
        | some r => return some r
        | none =>
            -- an operation that answers a scalar binds one, so its bits can be passed on
            let oe ← whnf (← mkAppM ``Op'.erase #[p.getArg! 4])
            -- a scalar load binds a value of the type it loads, so sums over it stay terms
            if oe.isAppOfArity ``Op.load 2 then
              let op := oe.getArg! 0
              let opTy ← mkAppM ``LoadOp.ty #[op]
              let lanes ← evalOut (← mkAppM ``ClifTy.lanes #[opTy])
              if lanes.isAppOf ``Option.none then
                let tv ← evalOut opTy
                let ho ← mkExpectedTypeHint (← mkEqRefl oe) (← mkEq (← mkAppM ``Op'.erase #[p.getArg! 4]) oe)
                let hl ← mkExpectedTypeHint (← mkEqRefl lanes) (← mkEq (← mkAppM ``ClifTy.lanes #[opTy]) lanes)
                let ht ← mkExpectedTypeHint (← mkEqRefl tv) (← mkEq opTy tv)
                -- where the post allows no fault: the bytes at a known address are there
                if let some r ← vcLoadAt k g t oe ho hl ht then return some r
                return some ((← applyNamed g ``wp_op_load_var [(`ho, ho), (`hl, hl), (`ht, ht)]).map (·, k, 0))
              -- a vector load binds a vector of its type, its lanes counted
              let ho ← mkExpectedTypeHint (← mkEqRefl oe) (← mkEq (← mkAppM ``Op'.erase #[p.getArg! 4]) oe)
              if let some r ← vcVLoadAt k g t oe ho then return some r
            let sc ← evalOut (← mkAppM ``Op.scalarResult #[← mkAppM ``Op'.erase #[p.getArg! 4]])
            if sc.isConstOf ``Bool.true then
              -- where the post allows no fault: the operation answers, by its operands' types
              if !(← faultAllowed g (t.getArg! 2)) then
                if let some r ← opByType k g t oe then return some r
                if let some r ← opByTypeV k g t oe then return some r
                if let some r ← opTyped k g t oe then return some r
                if let some hJ ← evalsOk k g t oe then
                  return some ((← applyNamed g ``wp_op_sc_ok [(`hs, ← mkEqRefl (mkConst ``Bool.true)),
                    (`hJ, hJ)]).map (·, k, 0))
              return some ((← applyNamed g ``wp_op_sc_var [(`hs, ← mkEqRefl (mkConst ``Bool.true))]).map (·, k, 0))
            -- where the post allows no fault: a scalar of a type the operands fix
            if !(← faultAllowed g (t.getArg! 2)) then
              if let some r ← opByType k g t oe then return some r
              if let some r ← opByTypeV k g t oe then return some r
              if let some r ← opTyped k g t oe then return some r
            ap ``wp_op_var
  | some ``Prog.store => vcStore k g t p false
  | some ``Prog.storeUnaligned => vcStore k g t p true
  | some ``Prog.istore8 => vcIstore8 k g t p
  | some ``forLoop =>
      let w := t.getArg! 7
      unless w.isFVar do return none
      let some hW := k.holds[w.fvarId!]? | return none
      -- The loop's invariant is the typestate it is entered in, less its cells
      -- when the body stores: a store may overwrite a cell, and what the body
      -- needs of the values it read before the loop is in the environment.
      let (W0, hW0) ← loopState k (t.getArg! 6) W hW (p.getArg! 3)
      return some ((← applyNamed g ``wp_forLoop_var [(`W, W0), (`hW, hW0)]).map (·, k, 0))
  | some ``Prog.br => ap ``wp_br_vc
  | some ``Prog.cont => ap ``wp_cont_vc
  | some ``Prog.loop =>
      let w := t.getArg! 7
      unless w.isFVar do return none
      let some hW := k.holds[w.fvarId!]? | return none
      let (W0, hW0) ← loopState k (t.getArg! 6) W hW (loopParts p)
      let lem ← if ← faultAllowed g (t.getArg! 2) then pure ``wp_loop_vc else pure (loopVcTFor p)
      return some ((← applyNamed g lem [(`W, W0), (`hW, hW0)]).map (·, k, 0))
  | some ``Prog.dloop =>
      let w := t.getArg! 7
      unless w.isFVar do return none
      let some hW := k.holds[w.fvarId!]? | return none
      let (W0, hW0) ← loopState k (t.getArg! 6) W hW (p.getArg! (p.getAppNumArgs - 2))
      -- where the post allows no fault: the carries of their types, so the
      -- tests over them answer
      let lem := if ← faultAllowed g (t.getArg! 2) then ``wp_dloop_vc else ``wp_dloop_vcT
      return some ((← applyNamed g lem [(`W, W0), (`hW, hW0)]).map (·, k, 0))
  | some ``ffiVoid => vcCall W k g t p
  | some ``Prog.callLocal =>
      if let some r ← vcLocal k g t p then return some r
      vcExt k g t p
  | some ``Prog.ite => vcIte k g t p
  | some ``Prog.call => vcCall W k g t p (raw := true)
  | some ``forLoopAcc =>
      let w := t.getArg! 7
      unless w.isFVar do return none
      let some hW := k.holds[w.fvarId!]? | return none
      let (W0, hW) ← loopState k (t.getArg! 6) W hW (p.getArg! 5)
      return some ((← applyNamed g ``wp_forLoopAcc_var [(`W, W0), (`hW, hW)]).map (·, k, 0))
  | some ``List.forM =>
      let l ← whnf (p.getArg! 3)
      if l.isAppOf ``List.cons then ap ``wp_forM_cons_of
      else if l.isAppOf ``List.nil then ap ``wp_forM_nil_of
      else return none
  | _ =>
      -- `have x := v; b`, which a dropped answer leaves: substitute it
      let zeta? : Option Expr :=
        if p.isLet then some (p.letBody!.instantiate1 p.letValue!)
        else if p.isAppOfArity ``letFun 4 then some ((p.getArg! 3).beta #[p.getArg! 2])
        else none
      if let some p' := zeta? then
        return some [(← g.replaceTargetDefEq (mkAppN t.getAppFn (t.getAppArgs.set! 4 p')), k, 0)]
      -- a `for` over a list written out: one trip at a time
      if p.getAppFn.isProj || p.isAppOf ``ForIn.forIn then
        let args := p.getAppArgs
        if args.size ≥ 3 then
          let coll := args[args.size - 3]!
          let l ← evalOut coll
          -- a list computed, not written out: put it written out, once, so
          -- each trip does not compute the rest of it again; the kernel
          -- checks the two agree
          if coll.listLit?.isNone && l.listLit?.isSome && !coll.hasFVar && !coll.hasMVar then
            let hEq ← mkExpectedTypeHint (← mkEqRefl l) (← mkEq coll l)
            let p' := mkAppN p.getAppFn (args.set! (args.size - 3) l)
            let t' := mkAppN t.getAppFn (t.getAppArgs.set! 4 p')
            let motive ← withLocalDeclD `c (← inferType coll) fun c => do
              mkLambdaFVars #[c] (mkAppN t.getAppFn (t.getAppArgs.set! 4 (mkAppN p.getAppFn (args.set! (args.size - 3) c))))
            let g' ← g.replaceTargetEq t' (← mkCongrArg motive hEq)
            return some [(g', k, 0)]
          let lem := if l.isAppOf ``List.cons then some ``wp_forIn_cons_of
            else if l.isAppOf ``List.nil then some ``wp_forIn_nil_of else none
          if let some lem := lem then
            let st ← saveState
            try return some ((← applyNamed g lem []).map (·, k, 0))
            catch _ => st.restore
      -- a test the generator makes on values written out: the arm it takes
      if p.isAppOfArity ``Decidable.rec 5 && !(p.getArg! 4).hasFVar && !(p.getArg! 4).hasMVar then
        let r ← whnf (p.getArg! 4)
        let arm? := if r.isAppOfArity ``Decidable.isTrue 2 then some ((p.getArg! 3).beta #[r.appArg!])
          else if r.isAppOfArity ``Decidable.isFalse 2 then some ((p.getArg! 2).beta #[r.appArg!]) else none
        if let some p' := arm? then
          return some [(← g.replaceTargetDefEq (mkAppN t.getAppFn (t.getAppArgs.set! 4 p'.headBeta)), k, 0)]
      -- a method of an instance: its body
      if let .proj sn i c := p.getAppFn then
        let fn ← whnfCore (.proj sn i (← whnf c))
        unless fn.isProj do
          let p' := (mkAppN fn p.getAppArgs).headBeta
          return some [(← g.replaceTargetDefEq (mkAppN t.getAppFn (t.getAppArgs.set! 4 p')), k, 0)]
      -- a match on a value already built: take its arm
      if let .reduced p' ← reduceMatcher? p then
        return some [(← g.replaceTargetDefEq (mkAppN t.getAppFn (t.getAppArgs.set! 4 p')), k, 0)]
      let some p' ← unfoldDefinition? p | return none
      return some [(← g.replaceTargetDefEq (mkAppN t.getAppFn (t.getAppArgs.set! 4 p')), k, 0)]

/-- A bound on a loop's limit: the limit itself when it is written out, the
    first constant a comparison on the path bounds it by, or else the limit's
    own value. -/
meta def limitOf? (nb : Expr) : TacticM (Option (Expr × Expr)) := do
  let nNat ← mkAppM ``UInt64.toNat #[nb]
  if !nb.hasFVar then
    -- a numeral as omega reads it, not a raw literal
    let some Nv := (← evalOut nNat).rawNatLit? | return none
    let N := mkNatLit Nv
    return some (N, ← mkExpectedTypeHint (← mkAppM ``Nat.le_refl #[N]) (← mkAppM ``LE.le #[nNat, N]))
  let mut cands : Array Nat := #[]
  -- the limit computed at the compared value first: a limit that grows with
  -- the value it is computed from is bounded by it there
  let mut direct : Array Nat := #[]
  for d in ← getLCtx do
    if d.isImplementationDetail then continue
    let ty ← instantiateMVars d.type
    unless ty.isAppOfArity ``Eq 3 && (ty.getArg! 1).isAppOfArity ``cmpInt 4 do continue
    let c := (ty.getArg! 1).getArg! 3
    if c.hasFVar then continue
    if let some v := (← evalOut (← mkAppM ``UInt64.toNat #[c])).rawNatLit? then
      cands := cands.push v
    let x := (ty.getArg! 1).getArg! 2
    if x.isFVar && nb.containsFVar x.fvarId! then
      let atC ← instantiateMVars (nb.replaceFVar x c)
      unless atC.hasFVar do
        if let some v := (← evalOut (← mkAppM ``UInt64.toNat #[atC])).rawNatLit? then
          direct := direct.push v
  let proves (v : Nat) : TacticM (Option Expr) := do
    let st ← saveState
    let hN ← mkFreshExprMVar (← mkAppM ``LE.le #[nNat, mkNatLit v])
    if (← closeWith hN.mvarId! (← `(tactic| bound_close))).isEmpty then
      return some (← instantiateMVars hN)
    st.restore
    return none
  for v in direct.qsort (· < ·) do
    if let some h ← proves v then return some (mkNatLit v, h)
  for v in cands.qsort (· < ·) do
    let some h ← proves v | continue
    -- the least bound that follows, searched below the one found: a limit
    -- computed from the compared value is often far below it
    let mut hi := v
    let mut best := h
    let mut lo := 0
    while lo < hi do
      let mid := (lo + hi) / 2
      match ← proves mid with
      | some h' => hi := mid; best := h'
      | none => lo := mid + 1
    return some (mkNatLit hi, best)
  -- no constant bounds it: the limit is its own bound, so a trip still knows
  -- it is below the value it was counted to (a length the program was handed)
  return some (nNat, ← mkAppM ``Nat.le_refl #[nNat])

/-- How a fit in 64 bits is decided: by the kernel when it is closed; by
    linear arithmetic over a start the program was answered, where the kernel
    would count `2 ^ 64` out one step at a time. -/
meta def fitTac (e : Expr) : TacticM (TSyntax `tactic) :=
  if e.hasFVar then `(tactic| bound_close) else `(tactic| first | kdecide | bound_close)

/-- Run `x` as a first pass that only looks at what a loop's trip computes:
    every post allows a fault, so nothing is asked of addresses. -/
meta def withProbe {β : Type} (x : TacticM β) : TacticM β := do
  -- on a budget of its own: a trip that walks far is not a count over memory,
  -- and its first pass would cost as much as the proof
  let hb ← IO.getNumHeartbeats
  let budget (c : Core.Context) : Core.Context :=
    { c with initHeartbeats := hb, maxHeartbeats := 20000 * 1000 }
  withTheReader Core.Context (fun c => budget { c with options := c.options.setBool `vcProbe true }) x

/-- A counted loop's bound on its accumulator: the limit's value `N`, the
    accumulator's start `A0`, and the first step `S` a trip may add that the
    body proves, with `run` proving the body. The goal after the loop, or
    `none` when no step is proved. -/
meta def vcAccBoundWith (run : MVarId → VCKnow → TacticM (Array MVarId)) (W : Expr) (k : VCKnow)
    (g : MVarId) (t p : Expr) : TacticM (Option MVarId) := do
  -- a first pass walks on by the plain rule
  if (← getOptions).getBool `vcProbe false then return none
  let env := t.getArg! 6
  let w := t.getArg! 7
  unless w.isFVar do return none
  let some hW := k.holds[w.fvarId!]? | return none
  let some (nc, pfn, k1) ← k.lookup (p.getArg! 3) env | return none
  let some (_, nb) ← scalarTerm? nc | return none
  let some (N, hN) ← limitOf? nb | return none
  let some (ac, pfa, k2) ← k1.lookup (p.getArg! 4) env | return none
  let some (_, ab) ← scalarTerm? ac | return none
  let natTy := mkConst ``Nat
  let A0 ← if ab.hasFVar then mkAppM ``UInt64.toNat #[ab] else evalOut (← mkAppM ``UInt64.toNat #[ab])
  let hA ← mkExpectedTypeHint (← mkAppM ``Nat.le_refl #[A0]) (← mkAppM ``LE.le #[← mkAppM ``UInt64.toNat #[ab], A0])
  let two64 ← mkAppM ``HPow.hPow #[mkNatLit 2, mkNatLit 64]
  for keep in [true, false] do
    let (W0, hW0) ← loopState k2 env W hW (p.getArg! 5) (keepCells := keep)
    let st0 ← saveState
    -- the rule at one step, the state restored when the body does not prove it
    let attempt (step : Nat) : TacticM (Option MVarId) := do
      let S := mkNatLit step
      let fitE ← mkAppM ``LT.lt #[← mkAppM ``HAdd.hAdd #[A0, ← mkAppM ``HMul.hMul #[N, S]], two64]
      let hfit ← mkFreshExprMVar fitE
      unless (← closeWith hfit.mvarId! (← fitTac fitE)).isEmpty do
        st0.restore; return none
      let gs ← try applyNamed g ``wp_forLoopAcc_bnd [(`W, W0), (`hW, hW0), (`N, N), (`A0, A0), (`S, S),
          (`hn, pfn), (`hN, hN), (`ha, pfa), (`hA, hA), (`hfit, hfit)]
        catch _ => pure []
      let [gb, gk] := gs | do st0.restore; return none
      if (← run gb k2).isEmpty then return some gk
      st0.restore; return none
    -- the step a trip adds, as a first pass sees it: the body's answer the
    -- value plus a constant
    let probed : Option Nat ← do
      let Sm ← mkFreshExprMVar (mkConst ``Nat)
      let fitE ← mkAppM ``LT.lt #[← mkAppM ``HAdd.hAdd #[A0, ← mkAppM ``HMul.hMul #[N, Sm]], two64]
      let hfit ← mkFreshExprMVar fitE
      let r ← tryCatchRuntimeEx (do
          let gs ← applyNamed g ``wp_forLoopAcc_bnd [(`W, W0), (`hW, hW0), (`N, N), (`A0, A0), (`S, Sm),
            (`hn, pfn), (`hN, hN), (`ha, pfa), (`hA, hA), (`hfit, hfit)]
          let [gb, _] := gs | pure none
          let left ← withProbe (run gb k2)
          let mut found := none
          for lg in left do
            let lt ← instantiateMVars (← lg.getType)
            unless lt.isAppOfArity ``LE.le 4 do continue
            -- this loop's own bound, not an inner one's
            unless ((lt.getArg! 3).find? (· == Sm)).isSome do continue
            let lhs := lt.getArg! 2
            unless lhs.isAppOfArity ``UInt64.toNat 1 do continue
            -- an answer bounded where it was computed: by the bound's constant
            let fromHyp ← lg.withContext do
              for d in ← getLCtx do
                let ty ← instantiateMVars d.type
                unless ty.isAppOfArity ``LE.le 4 && ty.getArg! 2 == lhs do continue
                let rhs := ty.getArg! 3
                unless rhs.isAppOfArity ``HAdd.hAdd 6 do continue
                let c := rhs.getArg! 5
                if c.hasFVar || c.hasMVar then continue
                if let some v ← (evalNat c).run then return some v
              return none
            if let some v := fromHyp then
              if 0 < v then found := some v; break
            -- constants added one after another: their sum
            let mut e := lhs.getArg! 0
            let mut tot := 0
            let mut ok := true
            while e.isAppOfArity ``HAdd.hAdd 6 do
              let c := e.getArg! 5
              if c.hasFVar || c.hasMVar then ok := false; break
              match (← evalOut (← mkAppM ``UInt64.toNat #[c])).rawNatLit? with
              | some v => tot := tot + v
              | none => ok := false; break
              e := e.getArg! 4
            if ok && 0 < tot then found := some tot; break
          pure found) (fun _ => pure none)
      st0.restore
      pure r
    if let some step := probed then
      if let some gk ← attempt step then return some gk
    -- the steps whose trips fit in 64 bits: a prefix, the fit falling with the step
    let mut fits : Array Nat := #[]
    for step in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 65536] do
      let fitE ← mkAppM ``LT.lt #[← mkAppM ``HAdd.hAdd #[A0, ← mkAppM ``HMul.hMul #[N, mkNatLit step]], two64]
      let hfit ← mkFreshExprMVar fitE
      let ok := (← closeWith hfit.mvarId! (← fitTac fitE)).isEmpty
      st0.restore
      if ok then fits := fits.push step else break
    -- a body that proves a step proves every wider one that fits: try the
    -- widest, and when it holds, search down for the narrowest
    let some widest := fits.back? | continue
    let some gTop ← attempt widest | continue
    let mut best ← saveState
    let mut bestGk := gTop
    let mut lo := 0
    let mut hi := fits.size - 1
    while lo < hi do
      let mid := (lo + hi) / 2
      st0.restore
      match ← attempt fits[mid]! with
      | some gk => hi := mid; best ← saveState; bestGk := gk
      | none => lo := mid + 1
    best.restore
    return some bestGk
  -- no step the body proves: the trip's bound alone, the value a 64-bit one
  -- when the body keeps it one
  for lem in [``wp_forLoopAcc_idx64, ``wp_forLoopAcc_idx] do
   for keep in [true, false] do
    let (W0, hW0) ← loopState k2 env W hW (p.getArg! 5) (keepCells := keep)
    let st ← saveState
    let gs ← try applyNamed g lem [(`W, W0), (`hW, hW0), (`N, N), (`hn, pfn), (`hN, hN), (`ha, pfa)]
      catch _ => do st.restore; continue
    let [gb, gk] := gs | do st.restore; continue
    if (← run gb k2).isEmpty then return some gk
    st.restore
  let _ := natTy
  return none

/-- The cells a goal `TState.holds S w` left by a trip of a loop body has
    to go on with: those of the latest typestate its context holds of `w`. -/
meta def endCells? (g : MVarId) : MetaM (Option (List Expr)) := g.withContext do
  let t ← instantiateMVars (← g.getType)
  unless t.isAppOfArity ``Contracts.TState.holds 2 do return none
  let w := t.appArg!
  let mut best : Option Expr := none
  for d in (← getLCtx) do
    if d.isImplementationDetail then continue
    let ty ← instantiateMVars d.type
    if ty.isAppOfArity ``Contracts.TState.holds 2 && ty.appArg! == w then
      best := some (ty.getArg! 0)
  let some S := best | return none
  let some Sv ← kernelList S | return none
  let some (_, facts) := Sv.listLit? | return none
  return some (facts.filter fun x =>
    x.isAppOf ``Contracts.Fact.cell || x.isAppOf ``Contracts.Fact.cstr || x.isAppOf ``Contracts.Fact.cstrIn)

/-- A counted loop of a known trip count whose body stores: the rule that
    shows the body its trip's bound, with every cell kept, when the body
    proves it. -/
meta def vcForBoundWith (run : MVarId → VCKnow → TacticM (Array MVarId)) (W : Expr) (k : VCKnow)
    (g : MVarId) (t p : Expr) : TacticM (Option MVarId) := do
  -- a first pass walks on by the plain rule
  if (← getOptions).getBool `vcProbe false then return none
  let env := t.getArg! 6
  let w := t.getArg! 7
  unless w.isFVar do return none
  let some hW := k.holds[w.fvarId!]? | return none
  let some (nc, pfn, k1) ← k.lookup (p.getArg! 2) env | return none
  let some (_, nb) ← scalarTerm? nc | return none
  let some (N, hN) ← limitOf? nb | return none
  -- every cell, then only those every trip leaves as they were
  let mut keepOnly : Option Expr := none
  for _ in [0:4] do
    let (W0, hW0) ← loopState k1 env W hW (p.getArg! 3) (keepCells := true) (keepOnly := keepOnly)
    let st ← saveState
    let gs ← try applyNamed g ``wp_forLoop_bnd [(`W, W0), (`hW, hW0), (`N, N), (`hn, pfn), (`hN, hN)]
      catch _ => do st.restore; return none
    let [gb, gk] := gs | do st.restore; return none
    let left ← run gb k1
    if left.isEmpty then return some gk
    let mut keep : Option (List Expr) := none
    for lg in left do
      let some ends ← endCells? lg | st.restore; return none
      keep := some (match keep with | none => ends | some l => l.filter ends.contains)
    st.restore
    let some kept := keep | return none
    let L ← mkListLit (mkConst ``Contracts.Fact) kept
    if keepOnly == some L then return none
    keepOnly := some L
  return none

/-- The facts a goal `TState.holds S w` left at the end of an arm has to go on
    with: those of the latest typestate its context holds of `w`, less any
    that name the arm's own values. -/
meta def endFacts? (g : MVarId) (outer : LocalContext) : MetaM (Option (List Expr)) := g.withContext do
  let t ← instantiateMVars (← g.getType)
  unless t.isAppOfArity ``Contracts.TState.holds 2 do return none
  let w := t.appArg!
  let mut best : Option Expr := none
  for d in (← getLCtx) do
    if d.isImplementationDetail then continue
    let ty ← instantiateMVars d.type
    if ty.isAppOfArity ``Contracts.TState.holds 2 && ty.appArg! == w then
      best := some (ty.getArg! 0)
  let some S := best | return none
  let some Sv ← kernelList S | return none
  let some (_, facts) := Sv.listLit? | return none
  return some (facts.filter fun f => !f.hasAnyFVar (!outer.contains ·))

/-- The step a loop body adds to its first carry, read off its syntax: the one
    amount it adds to that carry with `iaddImm`, when there is one only. -/
meta def stepOf? (bodyFn : Expr) (slotVal : Expr → MetaM (Option Nat) := fun _ => pure none) :
    MetaM (Option Nat) := do
  forallTelescope (← inferType bodyFn) fun xs _ => do
    unless xs.size ≥ 2 do return none
    let cs := xs[1]!
    let b ← Core.betaReduce (mkAppN bodyFn xs)
    let hits ← IO.mkRef (#[] : Array Expr)
    let slots ← IO.mkRef (#[] : Array Expr)
    b.forEach fun e => do
      -- under a binder a term may name its variable: not one to look up
      if e.isAppOfArity ``iaddImm 4 && (e.getArg! 2).isAppOf ``Vals.head && (e.getArg! 2).appArg! == cs then
        hits.modify (·.push (e.getArg! 3))
      -- or a value made before the loop added to it
      if e.isAppOfArity ``iadd 6 && (e.getArg! 3).isAppOf ``Vals.head && (e.getArg! 3).appArg! == cs &&
          !(e.getArg! 4).hasLooseBVars && !(e.getArg! 4).hasAnyFVar (fun f => xs.any (·.fvarId! == f)) then
        slots.modify (·.push (e.getArg! 4))
    let mut steps : Array Int := #[]
    -- a sum whose other side is not known here may not be the step: passed
    -- over; and only where no constant is added
    let hs ← hits.get
    let ss ← slots.get
    for sl in (if hs.isEmpty then ss else #[]) do
      let some v ← slotVal sl | continue
      unless steps.contains (v : Int) do steps := steps.push v
    for k in ← hits.get do
      -- an amount a binder names (a trip of an unrolled loop): no one step
      if k.hasFVar || k.hasMVar || k.hasLooseBVars then return none
      let v ← try evalInt k catch _ => return none
      unless steps.contains v do steps := steps.push v
    match steps.toList with
    | [v] => return if v > 0 then some v.toNat else none
    | _ => return none

/-- A 64-bit value's number, computed out when it names no variable. -/
meta def natOf (x : Expr) : MetaM Expr := do
  let n ← mkAppM ``UInt64.toNat #[x]
  if x.hasFVar || x.hasMVar then return n
  match (← evalOut n).rawNatLit? with
  | some v => return mkNatLit v
  | none => return n

/-- A counter's invariant: from `lo`, by `s` a trip, while `s` more stay at most
    `hi`, on the grid `lo` starts; or, unaligned, below `hi`. -/
meta def ctrPred (lo hi : Expr) (s : Nat) (aligned : Bool) : MetaM Expr :=
  withLocalDeclD `x (mkConst ``UInt64) fun x => do
    let xn ← mkAppM ``UInt64.toNat #[x]
    let c1 ← mkAppM ``LE.le #[lo, xn]
    unless aligned do return ← mkLambdaFVars #[x] (← mkAppM ``And #[c1, ← mkAppM ``LT.lt #[xn, hi]])
    let c2 ← mkAppM ``LE.le #[← mkAppM ``HAdd.hAdd #[xn, mkNatLit s], hi]
    if s == 1 then return ← mkLambdaFVars #[x] (← mkAppM ``And #[c1, c2])
    let c3 ← mkEq (← mkAppM ``HMod.hMod #[← mkAppM ``HSub.hSub #[xn, lo], mkNatLit s]) (mkNatLit 0)
    mkLambdaFVars #[x] (← mkAppM ``And #[c1, ← mkAppM ``And #[c2, c3]])

/-- Where a counter leaves: at or past `lo`, and at most `hi` when `hi` is
    past `lo`. -/
meta def ctrExit (lo hi : Expr) (s : Nat) (aligned : Bool) : MetaM Expr :=
  withLocalDeclD `x (mkConst ``UInt64) fun x => do
    let xn ← mkAppM ``UInt64.toNat #[x]
    let c12 ← mkAppM ``And #[← mkAppM ``LE.le #[lo, xn],
      ← mkAppM ``LE.le #[xn, ← mkAppM ``HAdd.hAdd #[lo, ← mkAppM ``HSub.hSub #[hi, lo]]]]
    -- and on the grid it started on, where it steps by more than one
    if !aligned || s == 1 then return ← mkLambdaFVars #[x] c12
    let c3 ← mkEq (← mkAppM ``HMod.hMod #[← mkAppM ``HSub.hSub #[xn, lo], mkNatLit s]) (mkNatLit 0)
    mkLambdaFVars #[x] (← mkAppM ``And #[c12, c3])

/-- The lengths the program was handed, each at most a region's span, put in
    the context once, for the arithmetic on addresses past them. -/
meta def noteRoomBounds (g : MVarId) (hW : Expr) : MetaM MVarId := g.withContext do
  let some S ← stateOf? (← instantiateMVars (← inferType hW)) | return g
  let some facts ← kernelList S | return g
  let some (α, fs) := facts.listLit? | return g
  let mut g := g
  for f in fs do
    unless f.isAppOfArity ``Contracts.Fact.roomArg 2 && (f.getArg! 1).hasFVar do continue
    let X := f.getArg! 1
    let ty ← mkAppM ``LE.le #[← mkAppM ``UInt64.toNat #[X], ← mkAppM ``HPow.hPow #[mkNatLit 2, mkNatLit 36]]
    let known ← g.withContext do
      return (← getLCtx).any fun d => !d.isImplementationDetail && d.type == ty
    if known then continue
    let some hm ← memLit? α fs (fun y => pure (y == f)) | continue
    let hm ← mkExpectedTypeHint hm (← mkAppM ``Membership.mem #[S, f])
    let pf ← g.withContext do mkAppM ``roomArg_toNat_le #[hW, hm]
    let (_, g') ← g.note `hlen pf (some ty)
    g := g'
  return g

/-- **A bottom-tested loop counting to a bound**, where the post allows no
    fault: the guard's carry is the counter, compared below the bound. A first
    pass over a trip finds the step it adds; the counter's invariant is then
    that it stays on its grid and a step short of the bound, and what follows
    the loop knows where it left. The goals left, or `none`. -/
meta def dloopX? (run : MVarId → VCKnow → TacticM (Array MVarId)) (W : Expr) (k : VCKnow) (g : MVarId)
    (t p : Expr) : TacticM (Option (List MVarId)) := g.withContext do
  if ← faultAllowed g (t.getArg! 2) then return none
  let w := t.getArg! 7
  unless w.isFVar do return none
  let some hW := k.holds[w.fvarId!]? | return none
  let env := t.getArg! 6
  let n := p.getAppNumArgs
  let body := p.getArg! (n - 2)
  let exitIdx := p.getArg! (n - 3)
  let contOnTrue ← whnf (p.getArg! (n - 4))
  let guardIdx ← whnf (p.getArg! (n - 6))
  let cb := p.getArg! (n - 7)
  let cc ← whnf (p.getArg! (n - 8))
  let init := p.getArg! (n - 9)
  let tys := p.getArg! (n - 12)
  -- the counter: the guard's carry, or the first where there is no guard and
  -- every trip runs from it
  let j ← if guardIdx.isAppOfArity ``Option.some 2 then
      match ← (evalNat (← instantiateMVars guardIdx.appArg!)).run with
      | some j => pure j
      | none => return none
    else pure 0
  unless cc.isConstOf ``ICmpCond.ult && contOnTrue.isConstOf ``Bool.true do return none
  let some (_, ex) := (← evalOut exitIdx).listLit? | return none
  let mut jE? := none
  for e in ex, i in [0:ex.length] do
    if (← (evalNat e).run) == some j then jE? := some i; break
  let some s0 ← valsSlotAt? init j | return none
  let some (c0, pf0, k1) ← k.lookup s0 env | return none
  let some (t0, x0) ← scalarTerm? c0 | return none
  unless t0.isConstOf ``ClifTy.i64 do return none
  let some (cbv, _, k1) ← k1.lookup cb env | return none
  let some (_, hiv) ← scalarTerm? cbv | return none
  let lo ← natOf x0
  let hi ← natOf hiv
  let (W0, hW0) ← loopState k1 env W hW body
  let hjt ← mkDecideProof (← mkAppM ``LT.lt #[mkNatLit j, ← mkAppM ``List.length #[tys]])
  -- the counter's place among the exits, where it leaves: what follows is
  -- then asked once, of where it left; else at each exit, in place
  let exitArgs ← match jE? with
    | some jE => pure [(`jE, mkNatLit jE), (`hjE, ← mkDecideProof (← mkEq
        (← mkAppM ``GetElem?.getElem? #[exitIdx, mkNatLit jE]) (← mkAppM ``Option.some #[mkNatLit j])))]
    | none => pure []
  let rule := if jE?.isSome then ``wp_dloop_vcX else ``wp_dloop_vcN
  let slotJ ← mkAppM ``Option.getD #[← mkAppM ``GetElem?.getElem? #[← mkAppM ``Vals.slots #[init], mkNatLit j],
    mkNatLit 0]
  let hx0 ← mkExpectedTypeHint pf0 (← mkEq (← mkAppM ``GetElem?.getElem? #[env, slotJ])
    (← mkAppM ``Option.some #[mkApp2 (mkConst ``V.sc) (mkConst ``ClifTy.i64) x0]))
  let common := [(`W, W0), (`hW, hW0), (`j, mkNatLit j), (`hjt, hjt), (`hx0, hx0)] ++ exitArgs
  -- the goals the rule leaves: the guard's case put for its index, the
  -- impossible case of no guard closed
  let prep (gs : List MVarId) : TacticM (List MVarId) := do
    let mut out := []
    for g' in gs do
      let ty ← instantiateMVars (← g'.getType)
      let guardCase := ty.isForall && ty.bindingDomain!.isConstOf ``Nat && ty.bindingBody!.isForall &&
        ((ty.bindingBody!.bindingDomain!.eq?.map (·.2.2.isAppOf ``Option.some)).getD false)
      if guardCase then
        out := out ++ (← Tactic.run g' (withoutRecover <| evalTactic (← `(tactic| intro gi hgi; cases hgi))))
      else if ty.isForall && (ty.bindingDomain!.eq?.map (·.2.2.isAppOf ``Option.none)).getD false then
        -- with a guard, it is not absent; without one, the counter's start
        -- has the invariant
        out := out ++ (← Tactic.run g' (withoutRecover <| evalTactic (← `(tactic| intro h; try cases h))))
      else out := out ++ [g']
    return out
  -- the step a trip adds to the counter: as written, else by a first pass
  let st ← saveState
  let probe := mkConst ``ProbeP
  let written ← if j == 0 then stepOf? body else pure none
  let step? : Option Nat ← if written.isSome then pure written else (tryCatchRuntimeEx (do
      let gs ← applyNamed g rule (common ++ [(`P, probe)] ++ (if jE?.isSome then [(`Px, probe)] else []))
      let some gb := (if jE?.isSome then gs.dropLast else gs).getLast? | pure none
      let left ← withProbe (run gb k1)
      let mut found := none
      for lg in left do
        let lt ← instantiateMVars (← lg.getType)
        unless lt.isAppOfArity ``ProbeP 1 do continue
        let b := lt.appArg!
        unless b.isAppOfArity ``HAdd.hAdd 6 && !(b.getArg! 5).hasFVar do continue
        if let some v := (← evalOut (← mkAppM ``UInt64.toNat #[b.getArg! 5])).rawNatLit? then
          if 0 < v then found := some v; break
      pure found
      ) (fun _ => pure none))
  st.restore
  let some step := step? | return none
  -- the bound on the counter's grid, where it is; the lengths first
  let g ← noteRoomBounds g hW
  let (g, aligned) ← g.withContext do
    if step == 1 then return (g, true)
    let ty ← mkEq (← mkAppM ``HMod.hMod #[← mkAppM ``HSub.hSub #[hi, lo], mkNatLit step]) (mkNatLit 0)
    let m ← mkFreshExprMVar ty
    unless (← closeWith m.mvarId! (← `(tactic| bound_close))).isEmpty do return (g, false)
    let (_, g') ← g.note `hgrid (← instantiateMVars m) (some ty)
    return (g', true)
  g.withContext do
  let P ← ctrPred lo hi step aligned
  let Px ← ctrExit lo hi step aligned
  let gs ← applyNamed g rule (common ++ [(`P, P)] ++ (if jE?.isSome then [(`Px, Px)] else []))
  return some (← prep gs)

/-- A head-tested loop's counter invariant: from `lo`, by `s` a trip, at most
    `hi`, on the grid `lo` starts. -/
meta def ctrPredH (lo hi : Expr) (s : Nat) (below : Bool) : MetaM Expr :=
  withLocalDeclD `x (mkConst ``UInt64) fun x => do
    let xn ← mkAppM ``UInt64.toNat #[x]
    let c1 ← mkAppM ``LE.le #[lo, xn]
    -- at most the bound; or, where the bound may be below the start, still
    -- at the start
    let c2 ← if below then mkAppM ``Or #[← mkAppM ``LE.le #[xn, hi], ← mkEq xn lo]
      else mkAppM ``LE.le #[xn, hi]
    if s == 1 then return ← mkLambdaFVars #[x] (← mkAppM ``And #[c1, c2])
    let c3 ← mkEq (← mkAppM ``HMod.hMod #[← mkAppM ``HSub.hSub #[xn, lo], mkNatLit s]) (mkNatLit 0)
    mkLambdaFVars #[x] (← mkAppM ``And #[c1, ← mkAppM ``And #[c2, c3]])

/-- **A head-tested loop counting to a bound**, where the post allows no
    fault: carry 0 is the counter, and the head's test goes on while it is
    below a bound fixed before the loop. A first pass over a trip finds the
    test and the step; the counter's invariant is that it stays on its grid
    and at most the bound. The goals left, or `none`. -/
meta def loopN? (run : MVarId → VCKnow → TacticM (Array MVarId)) (W : Expr) (k : VCKnow) (g : MVarId)
    (t p : Expr) : TacticM (Option (List MVarId)) := g.withContext do
  if ← faultAllowed g (t.getArg! 2) then return none
  let w := t.getArg! 7
  unless w.isFVar do return none
  let some hW := k.holds[w.fvarId!]? | return none
  let env := t.getArg! 6
  let n := p.getAppNumArgs
  let init := p.getArg! (n - 4)
  let tys := p.getArg! (n - 8)
  let j := 0
  -- only a head that answers its test outright, of the first carry: any other
  -- would be walked by the first pass for nothing
  let headFn := p.getArg! (n - 3)
  let test? : Option (Expr × Expr × Expr × Option Nat) ← forallTelescope (← inferType headFn) fun xs _ => do
    unless xs.size == 2 do return none
    let r ← whnf (mkAppN headFn xs).headBeta
    unless r.isAppOf ``Prog.ret do return none
    let tup ← whnf r.appArg!
    unless tup.isAppOfArity ``Prod.mk 4 do return none
    let cond ← whnf (tup.getArg! 2)
    let some ca ← projField? (← mkAppM ``Cond.a #[cond]) | return none
    unless ca.isAppOf ``Vals.head && ca.appArg! == xs[1]! do return none
    let some cb ← projField? (← mkAppM ``Cond.b #[cond]) | return none
    if cb.containsFVar xs[1]!.fvarId! then return none
    let some cc ← projField? (← mkAppM ``Cond.cc #[cond]) | return none
    let some e ← projField? (← mkAppM ``Cond.exitOnTrue #[cond]) | return none
    -- the counter's place among the values the loop leaves with, if it leaves
    let rest ← whnf (tup.getArg! 3)
    let mut jE := none
    if rest.isAppOfArity ``Prod.mk 4 then
      let mut v ← valsWhnf (rest.getArg! 2)
      let mut i := 0
      while v.isAppOfArity ``Vals.cons 5 do
        let sl := v.getArg! 3
        if sl.isAppOf ``Vals.head && sl.appArg! == xs[1]! then jE := some i; break
        v ← valsWhnf v.appArg!
        i := i + 1
    return some (← whnf cc, cb, ← whnf e, jE)
  let some (tcc, tb, texit, jE?) := test? | return none
  -- a trip that calls out of the program is not a count over memory
  if ← bodyCalls (loopParts p) then return none
  let some s0 ← valsSlotAt? init j | return none
  let some (c0, pf0, k1) ← k.lookup s0 env | return none
  let some (t0, x0) ← scalarTerm? c0 | return none
  unless t0.isConstOf ``ClifTy.i64 do return none
  let (W0, hW0) ← loopState k1 env W hW (loopParts p)
  let hjt ← mkDecideProof (← mkAppM ``LT.lt #[mkNatLit j, ← mkAppM ``List.length #[tys]])
  let slotJ ← mkAppM ``Option.getD #[← mkAppM ``GetElem?.getElem? #[← mkAppM ``Vals.slots #[init], mkNatLit j],
    mkNatLit 0]
  let hx0 ← mkExpectedTypeHint pf0 (← mkEq (← mkAppM ``GetElem?.getElem? #[env, slotJ])
    (← mkAppM ``Option.some #[mkApp2 (mkConst ``V.sc) (mkConst ``ClifTy.i64) x0]))
  let common := [(`W, W0), (`hW, hW0), (`j, mkNatLit j), (`hjt, hjt), (`hx0, hx0)]
  let outer ← getLCtx
  -- the step a trip adds, and the test the head makes: as written, else by a
  -- first pass
  let st ← saveState
  let written : Option (Nat × Expr × Expr × Bool) ← do
    let some step ← stepOf? (p.getArg! (n - 2)) (fun sl => do
        let some (c, _, _) ← k1.lookup sl env | return none
        let some (_, v) ← scalarTerm? c | return none
        if v.hasFVar then return none
        return (← evalOut (← mkAppM ``UInt64.toNat #[v])).rawNatLit?) | pure none
    let some (cbv, _, _) ← k1.lookup tb env | pure none
    let some (_, hiv) ← scalarTerm? cbv | pure none
    -- going on while below: `<`, or not `≥`, or not equal
    let rv := mkConst (if texit.isConstOf ``Bool.true then ``Bool.false else ``Bool.true)
    let goesOn := (tcc.isConstOf ``ICmpCond.ult || tcc.isConstOf ``ICmpCond.slt) && rv.isConstOf ``Bool.true ||
      (tcc.isConstOf ``ICmpCond.uge || tcc.isConstOf ``ICmpCond.sge) && rv.isConstOf ``Bool.false ||
      tcc.isConstOf ``ICmpCond.ne && rv.isConstOf ``Bool.true || tcc.isConstOf ``ICmpCond.eq && rv.isConstOf ``Bool.false
    if !goesOn then pure none
    else pure (some (step, hiv, tcc, tcc.isConstOf ``ICmpCond.slt || tcc.isConstOf ``ICmpCond.sge))
  let found? : Option (Nat × Expr × Expr × Bool) ← if written.isSome then pure written else (tryCatchRuntimeEx (do
      let gs ← applyNamed g ``wp_loop_vcN (common ++ [(`P, mkConst ``ProbeP)])
      let some gh := gs.getLast? | pure none
      let left ← withProbe (run gh k1)
      let mut found := none
      for lg in left do
        let lt ← instantiateMVars (← lg.getType)
        unless lt.isAppOfArity ``ProbeP 1 do continue
        let b := lt.appArg!
        unless b.isAppOfArity ``HAdd.hAdd 6 && !(b.getArg! 5).hasFVar && (b.getArg! 4).isFVar do continue
        let x := b.getArg! 4
        let some v := (← evalOut (← mkAppM ``UInt64.toNat #[b.getArg! 5])).rawNatLit? | continue
        -- the test, as the trip's context has it: of the counter, against a
        -- value from before the loop
        let test ← lg.withContext do
          let mut r := none
          for d in ← getLCtx do
            let some (_, l, rv) := (← instantiateMVars d.type).eq? | continue
            unless l.isAppOfArity ``cmpInt 4 && l.getArg! 2 == x do continue
            let hiv := l.getArg! 3
            if hiv.hasAnyFVar (fun f => !outer.contains f) then continue
            r := some (l.getArg! 0, hiv, rv)
          pure r
        let some (cc, hiv, rv) := test | continue
        let cc ← whnf cc
        let rv ← whnf rv
        -- going on while below: `<`, or not `≥`, or not equal
        let goesOn := (cc.isConstOf ``ICmpCond.ult || cc.isConstOf ``ICmpCond.slt) && rv.isConstOf ``Bool.true ||
          (cc.isConstOf ``ICmpCond.uge || cc.isConstOf ``ICmpCond.sge) && rv.isConstOf ``Bool.false ||
          cc.isConstOf ``ICmpCond.ne && rv.isConstOf ``Bool.true || cc.isConstOf ``ICmpCond.eq && rv.isConstOf ``Bool.false
        unless goesOn do continue
        let signed := cc.isConstOf ``ICmpCond.slt || cc.isConstOf ``ICmpCond.sge
        if 0 < v then found := some (v, hiv, cc, signed); break
      pure found
      ) (fun _ => pure none))
  st.restore
  let some (step, hiv, _, signed) := found? | return none
  let lo ← natOf x0
  let hi ← natOf hiv
  let g ← noteRoomBounds g hW
  -- the bound on the counter's grid, and below `2 ^ 63` where the test is signed
  let mut g := g
  for (nm, ty, need) in [(`hgrid, ← mkEq (← mkAppM ``HMod.hMod #[← mkAppM ``HSub.hSub #[hi, lo], mkNatLit step])
        (mkNatLit 0), step != 1), (`hsign, ← mkAppM ``LT.lt #[hi, ← mkAppM ``HPow.hPow #[mkNatLit 2, mkNatLit 63]],
        signed)] do
    unless need do continue
    let some g' ← g.withContext do
        let m ← mkFreshExprMVar ty
        unless (← closeWith m.mvarId! (← `(tactic| bound_close))).isEmpty do return none
        let (_, g') ← g.note nm (← instantiateMVars m) (some ty)
        return some g'
      | return none
    g := g'
  g.withContext do
  -- whether the bound may be below the start: a loop that may not run
  let below ← do
    let m ← mkFreshExprMVar (← mkAppM ``LE.le #[lo, hi])
    pure !(← closeWith m.mvarId! (← `(tactic| bound_close))).isEmpty
  let P ← ctrPredH lo hi step below
  -- where the counter leaves: what follows is asked once, of where it left
  match jE? with
  | some jE =>
      let exitTys := p.getArg! (n - 7)
      let hjE ← mkDecideProof (← mkAppM ``LT.lt #[mkNatLit jE, ← mkAppM ``List.length #[exitTys]])
      let Px ← ctrExit lo hi step true
      let gs ← applyNamed g ``wp_loop_vcX (common ++ [(`P, P), (`jE, mkNatLit jE), (`Px, Px), (`hjE, hjE)])
      return some gs
  | none =>
      let gs ← applyNamed g ``wp_loop_vcN (common ++ [(`P, P)])
      return some gs

/-- The condition generator's work: a goal, with what is known along its path
    and how many binders it sits under since its segment began; or the close of
    a segment. -/
inductive VCTask where
  | goal (g : MVarId) (k : VCKnow) (depth : Nat)
  | close (seg g : MVarId)

/-- How many binders a segment of the proof may open before the rest of the
    body is proved as a lemma of its own. A proof nested one binder per call
    costs the square of its depth to assemble; segments keep that bounded. -/
meta def vcSegment : Nat := 48

/-- Run the condition generator on a goal; the goals no rule covers remain. -/
meta partial def vcRun (W : Expr) (g0 : MVarId) (k0 : VCKnow := {}) : TacticM (Array MVarId) := do
  let mut work : List VCTask := [.goal g0 k0 0]
  let mut stuck := #[]
  while true do
    let task :: rest := work | break
    work := rest
    match task with
    | .close seg g =>
        let v ← instantiateMVars (mkMVar seg)
        if v.hasExprMVar then g.assign v
        -- `zetaDelta` spares the closure a type check of the whole segment;
        -- the goals it closes bind no lets, and the kernel checks the lemma
        else g.withContext do
          g.assign (← mkAuxTheorem (← instantiateMVars (← g.getType)) v (zetaDelta := true))
    | .goal g k depth =>
        if ← g.isAssigned then continue
        if depth ≥ vcSegment then
          let t := (← instantiateMVars (← g.getType)).headBeta
          if (wpProg? t).isSome then
            let seg ← g.withContext (mkFreshExprSyntheticOpaqueMVar t)
            work := .goal seg.mvarId! k 0 :: .close seg.mvarId! g :: work
            continue
        let t := (← instantiateMVars (← g.getType)).headBeta
        -- A counted loop threading a 64-bit accumulator: try the rule that
        -- bounds it, for each step the body may add, and keep the first the
        -- body proves.
        if let some p := wpProg? t then
          let p := (← instantiateMVars p).headBeta
          if p.isAppOf ``forLoopAcc then
            if (← g.withContext (whnf (p.getArg! 2))).isConstOf ``ClifTy.i64 then
              if let some gk ← g.withContext (vcAccBoundWith (fun gb kb => vcRun W gb kb) W k g t p) then
                work := .goal gk k depth :: work
                continue
          if p.isAppOfArity ``forLoop 4 then
            if ← bodyStores (← instantiateMVars (p.getArg! 3)) then
              if let some gk ← g.withContext (vcForBoundWith (fun gb kb => vcRun W gb kb) W k g t p) then
                work := .goal gk k depth :: work
                continue
        -- A bottom-tested loop counting to a bound, where no fault is allowed:
        -- the counter's invariant, its step found by a first pass.
        if let some p := wpProg? t then
          let p := (← instantiateMVars p).headBeta
          if p.isAppOf ``Prog.loop then
            if let some gs ← g.withContext (loopN? (fun gb kb => vcRun W gb kb) W k g t p) then
              work := gs.map (fun g' => VCTask.goal g' k depth) ++ work
              continue
          if p.isAppOf ``Prog.dloop then
            if let some gs ← g.withContext (dloopX? (fun gb kb => vcRun W gb kb) W k g t p) then
              work := gs.map (fun g' => VCTask.goal g' k depth) ++ work
              continue
        -- A loop whose body stores: try the invariant that keeps every cell,
        -- and keep it when the body proves it; drop the cells otherwise.
        if let some p := wpProg? t then
          let p := (← instantiateMVars p).headBeta
          let lax ← if p.isAppOf ``Prog.loop || p.isAppOf ``Prog.dloop then faultAllowed g (t.getArg! 2)
            else pure true
          let lem? := if p.isAppOf ``forLoop then some (``wp_forLoop_var, p.getArg! 3)
            else if p.isAppOf ``forLoopAcc then some (``wp_forLoopAcc_var, p.getArg! 5)
            else if p.isAppOf ``Prog.loop then
              some (if lax then ``wp_loop_vc else loopVcTFor p, loopParts p)
            else if p.isAppOf ``Prog.dloop then
              some (if lax then ``wp_dloop_vc else ``wp_dloop_vcT, p.getArg! (p.getAppNumArgs - 2))
            else none
          if let some (lem, body) := lem? then
            let w := t.getArg! 7
            if w.isFVar && (← bodyStores (← instantiateMVars body)) then
              if let some hW := k.holds[w.fvarId!]? then
                -- keep every cell, then only those every trip leaves as they
                -- were, until the body proves what it is given
                let mut keepOnly : Option Expr := none
                let mut found : Option MVarId := none
                for _ in [0:4] do
                  let st ← saveState
                  let ok : Except MessageData (MVarId ⊕ Expr) ← g.withContext do
                    let (W0, hW0) ← loopState k (t.getArg! 6) W hW body (keepCells := true) (keepOnly := keepOnly)
                    let gs ← applyNamed g lem [(`W, W0), (`hW, hW0)]
                    let some gk := gs.getLast? | return .error m!"the loop rule left no goals"
                    let mut left := #[]
                    for gb in gs.dropLast do
                      left := left ++ (← vcRun W gb k)
                    if left.isEmpty then return .ok (.inl gk)
                    let mut keep : Option (List Expr) := none
                    for lg in left do
                      let some ends ← endCells? lg | return .error m!"a trip ends short of a typestate"
                      keep := some (match keep with | none => ends | some l => l.filter ends.contains)
                    let some kept := keep | return .error m!"no trip"
                    return .ok (.inr (← mkListLit (mkConst ``Contracts.Fact) kept))
                  match ok with
                  | .ok (.inl gk) => found := some gk; break
                  | .ok (.inr L) =>
                      st.restore
                      if keepOnly == some L then break
                      keepOnly := some L
                  | .error _ => st.restore; break
                if let some gk := found then
                  work := .goal gk k depth :: work
                  continue
        -- A branch: try a join that keeps the typestate, then one that keeps
        -- its parts and rooms; keep the first both arms prove.
        if let some p := wpProg? t then
          if (← instantiateMVars p).headBeta.isAppOf ``Prog.ite then
            let p' := (← instantiateMVars p).headBeta
            -- a join the rest of the body needs nothing at: an arm that ends
            -- where it returns, under a post that asks nothing
            let kE := p'.getArg! 5
            let kBody ← g.withContext do
              let ty ← whnf (← inferType kE)
              forallTelescope ty fun xs _ => do
                pure (← whnfR (mkAppN kE xs)).headBeta
            let post ← instantiateMVars (t.getArg! 5)
            let postTrue ← lambdaTelescope post fun _ b => pure b.isTrue
            let free := (kBody.isAppOf ``Prog.ret || kBody.isAppOf ``Pure.pure) && postTrue
            let mut done := false
            -- after the typestate the branch is entered in, the facts both
            -- arms end in, when that is where they fall short
            let mut joinFacts : Option Expr := none
            for level in (if free then [2] else [4, 3, 0, 1]) do
              if level == 3 && joinFacts.isNone then continue
              let st ← saveState
              let ok : Except (Option Expr) (List (MVarId × VCKnow × Nat)) ← g.withContext do
                let some gs ← vcIte k g t (← instantiateMVars p).headBeta (if level == 3 then 0 else level)
                    (join? := if level == 3 then joinFacts else none) | return .error none
                let some gk := gs.getLast? | return .error none
                let mut left := #[]
                for (ga, ka, _) in gs.dropLast do
                  left := left ++ (← vcRun W ga ka)
                if left.isEmpty then return .ok [gk]
                unless level == 4 do return .error none
                let mut keep : Option (List Expr) := none
                for lg in left do
                  let some ends ← endFacts? lg (← getLCtx) | return .error none
                  keep := some (match keep with | none => ends | some l => l.filter ends.contains)
                let some kept := keep | return .error none
                return .error (some (← mkListLit (mkConst ``Contracts.Fact) kept))
              match ok with
              | .ok [(gk, kk, dd)] =>
                  work := .goal gk kk (depth + dd) :: work
                  done := true
                  break
              | .error j => st.restore; if level == 4 then joinFacts := j
              | _ => st.restore
            if done then continue
        match ← vcStep W k g with
        | some gs => work := gs.map (fun (g', k', dd) => VCTask.goal g' k' (depth + dd)) ++ work
        | none => stuck := stuck.push g
  return stuck

/-- **Discharge a body's weakest precondition from a typestate.** The goal is
    `∀ Γ w, W w → wp …`. The typestate is every counted loop's invariant and
    every call's contract; everything else is computed, from the front of the
    body, one construct at a time. What no rule covers is left as a goal. -/
elab "prog_vc " Wstx:term : tactic => withoutRecover do
  let W ← elabTerm Wstx none
  replaceMainGoal (← vcRun W (← getMainGoal)).toList

end AlgorithmLib.Prog
