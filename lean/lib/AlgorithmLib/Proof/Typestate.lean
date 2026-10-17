module
public import AlgorithmLib.Proof.ProgHoare
public import AlgorithmLib.Host.DevSpec
public import AlgorithmLib.Host.Lifecycle
public import AlgorithmLib.Host.ExtLife
public import AlgorithmLib.Host.FfiAnswers
public import AlgorithmLib.Host.Summary
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
  lands (`after`), so `moves_state` has only the precondition left, which the
  facts give. `Live` is one: the device calls' typestate.
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

A body's weakest precondition is proven from these one construct at a time;
`Check` decides no-misuse of a shipped body by computation instead.

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
    (hf : f.engine = true) (hb : vs.mapM asBits = some bits)
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
  | htAt _ _ _ => exact HtSpec.HtObj.of_mem this rfl rfl
  | cstr _ => exact this
  | devSeq => exact this
  | oracles => exact this
  | cstrIn _ _ => exact this
  | roomArg _ _ => exact this
  | devBuf _ _ => exact this
  | pinnedUsed _ => exact this
  | streams _ | events _ | evLive _ | readyAll _ | evAll _ => exact this

/-- **The standard typestates move by the table.** From a typestate that has
    memory unfrozen, a call whose precondition the facts give lands in what
    the contract table says it leaves, `after f bits S`, computed as `S'`. -/
theorem moves_state (cfg : Cfg) (S S' : TState) {f : Ffi} {vs : List V} {bits : List UInt64}
    (hz : S.contains (.part .frozen false) = true) (hf : f.engine = true) (hb : vs.mapM asBits = some bits)
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
    (hz : S.contains (.part .frozen false) = true) (hf : f.engine = true) (hb : vs.mapM asBits = some bits)
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

export AlgorithmLib.HProg.Contracts (Fits)

/-- Bytes a `room` fact covers. -/
theorem fits_of_state {S : TState} {w : World} {a : UInt64} {n : Nat} (hS : S.holds w)
    (h : Contracts.roomAt S a n = true) : Fits w.mem a n :=
  Contracts.roomAt_spec hS h

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
    have := Sem.decodeAddr_add (i := x.toNat) hd (by simp only [regionSpan, UInt64.reduceToNat]; have := hx.2; omega)
    rwa [UInt64.ofNat_toNat] at this
  exact fits_of_roomArg hS hm hz hd' hp (by omega)

/-- Bytes at a value past an address, within a room the typestate names. -/
theorem fits_of_room_add {S : TState} {w : World} {b x : UInt64} {n N : Nat} {r : Region} {off : Nat}
    (hS : S.holds w) (hm : Contracts.Fact.room r N ∈ S) (hz : S.contains (.part .frozen false) = true)
    (hd : decodeAddr b = some (r, off)) (hp : r ≠ .pinned) (hb : off + x.toNat + n ≤ N)
    (hN : N ≤ 2 ^ 36) (hn : 0 < n) : Fits w.mem (b + x) n := by
  have hx : N ≤ w.mem.sizes r := hS _ hm
  have hd' : decodeAddr (b + x) = some (r, off + x.toNat) := by
    have := Sem.decodeAddr_add (i := x.toNat) hd (by simp only [regionSpan, UInt64.reduceToNat]; omega)
    rwa [UInt64.ofNat_toNat] at this
  exact ⟨r, off + x.toNat, hd', hp, by omega, Contracts.holds_of_mem hS hz, by omega⟩

/-- A place a literal past a sum: the same place as the literal added first. -/
theorem fits_regroup {m : Mem} {b x c : UInt64} {n : Nat} (h : Fits m (b + c + x) n) :
    Fits m (b + x + c) n := by
  rwa [UInt64.add_assoc, UInt64.add_comm x c, ← UInt64.add_assoc]

theorem store_parts {w : World} {m : Mem} {a : UInt64} {n : Nat} {b : UInt64}
    (hz : m.frozen = w.mem.frozen ∧ m.busy = w.mem.busy)
    (p : Part) : p.get { obsStore w a n b with mem := m } = p.get w := by
  cases p <;> first | rfl | exact hz.1 | (show m.busy.isEmpty = w.mem.busy.isEmpty; rw [hz.2])

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
          refine afterStore_sound hS (store_parts ⟨Contracts.store_frozen hm, Contracts.store_busy hm⟩) (fun k _ => by cases k <;> rfl) rfl ⟨rfl, rfl, rfl⟩ (Contracts.store_sizes_eq hm)
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
              (fun _ _ _ h => ⟨Contracts.store_sizes_eq h, Contracts.store_frozen h, Contracts.store_busy h⟩) _ _ _ hm
            exact Contracts.nonCell_kept hS (store_parts hz) (fun k _ => by cases k <;> rfl) ⟨rfl, rfl, rfl⟩ hsz
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

/-- Lanes stored one after another keep every region's size, the frozen flag and the copies in flight. -/
theorem foldlM_lane_stores_inv {addr : UInt64} {lw : Nat} : ∀ (l : List (UInt64 × Nat)) (m0 m : Mem),
    l.foldlM (fun mm (p : UInt64 × Nat) => mm.store (addr + UInt64.ofNat (p.2 * lw)) lw p.1) m0 = some m →
    m.sizes = m0.sizes ∧ m.frozen = m0.frozen ∧ m.busy = m0.busy
  | [], m0, m, h => by simp at h; subst h; exact ⟨rfl, rfl, rfl⟩
  | p :: l, m0, m, h => by
      simp only [List.foldlM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
      obtain ⟨m1, h1, h2⟩ := h
      obtain ⟨hs, hz, hb⟩ := foldlM_lane_stores_inv l m1 m h2
      exact ⟨hs.trans (Contracts.store_sizes_eq h1), hz.trans (Contracts.store_frozen h1),
        hb.trans (Contracts.store_busy h1)⟩

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
            exact Contracts.nonCell_kept hS (store_parts hz) (fun k _ => by cases k <;> rfl) ⟨rfl, rfl, rfl⟩ hs
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
              exact Contracts.nonCell_kept hS (store_parts ⟨Contracts.store_frozen hm, Contracts.store_busy hm⟩) (fun k _ => by cases k <;> rfl) ⟨rfl, rfl, rfl⟩
                (Contracts.store_sizes_eq hm)
          | some q =>
              obtain ⟨x, t', y⟩ := q
              obtain ⟨⟨t0, ht⟩, hv'⟩ := hst x t' y rfl
              rw [ha] at ht; cases ht
              rw [hv] at hv'; cases hv'
              exact afterStore_sound hS (store_parts ⟨Contracts.store_frozen hm, Contracts.store_busy hm⟩) (fun k _ => by cases k <;> rfl) rfl ⟨rfl, rfl, rfl⟩ (Contracts.store_sizes_eq hm)
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
  wp_storeU_addr S _ ha hv hS (hJ.imp_right (AlgorithmLib.HProg.Contracts.Fits.mono (by cases t' <;> decide))) rfl fun w' h => hk w' (by
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
          exact afterStore_sound hS (store_parts ⟨Contracts.store_frozen hm, Contracts.store_busy hm⟩) (fun k _ => by cases k <;> rfl) rfl ⟨rfl, rfl, rfl⟩ (Contracts.store_sizes_eq hm)
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
  simp [evalOp, Sem.get, ha, hb, zipIntBits, norm64, ClifTy.isInt]

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
  simp [evalOp, Sem.get, ha, hb, zipIntBits, norm64, ClifTy.isInt]

theorem evalOp_bor64 {m : Mem} {Γ : Env} {a b : R} {x y : UInt64}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) :
    evalOp m Γ (.bor a b) = some (.sc .i64 (x ||| y)) := by
  simp [evalOp, Sem.get, ha, hb, zipIntBits, norm64, ClifTy.isInt]

theorem evalOp_bxor64 {m : Mem} {Γ : Env} {a b : R} {x y : UInt64}
    (ha : Γ[a]? = some (.sc .i64 x)) (hb : Γ[b]? = some (.sc .i64 y)) :
    evalOp m Γ (.bxor a b) = some (.sc .i64 (x ^^^ y)) := by
  simp [evalOp, Sem.get, ha, hb, zipIntBits, norm64, ClifTy.isInt]

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
    (W W' : World → Prop) {t0 : ClifTy} (ht : f.result = some t0) (hf : f.engine = true) {cs : List V}
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
theorem callOf_ffi_bits {f : Ffi} (hf : f.engine = true) {lc : Locals} {vs : List V} {w : World}
    {v : V} {w' : World} (h : callOf lc (.ffi f) vs w = some (some v, w')) :
    ∃ bits, vs.mapM asBits = some bits ∧ Sem.callBits f bits w = some (some v, w') := by
  have hf' : (f == .threadSpawn) = false ∧ f.inLib = false := by
    simpa [Ffi.engine, bne, Bool.and_eq_true] using hf
  have hs := hf'.1
  have hl := hf'.2
  simp only [callOf, hs, hl, if_false, Bool.false_eq_true, callImport, Contracts.ofCname_cname, bind, Option.bind,
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
    (W W' : World → Prop) {t0 : ClifTy} (ht : f.result = some t0) (hf : f.engine = true) {cs : List V}
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
  have hd := Sem.decodeAddr_base r _ hk
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
  have hd := Sem.decodeAddr_base r _ hk
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
        exact hk _ (Contracts.nonCell_kept hS (store_parts hz) (fun kk _ => by cases kk <;> rfl)
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
        exact hk _ (Contracts.nonCell_kept hS (store_parts hz) (fun kk _ => by cases kk <;> rfl)
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
-- One construct at a time
--
-- The weakest precondition of a body, from its front: a bind puts its first
-- part in front, and a loop over a list of steps takes one.
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

/-- Room at an address a value past a known one: the region's room covers
    the offset and the length. -/
theorem roomAt_add {S : Contracts.TState} {b x : UInt64} {n K : Nat} {r : Region} {off : Nat}
    (hm : Contracts.Fact.room r K ∈ S) (hz : S.contains (.part .frozen false) = true)
    (hd : decodeAddr b = some (r, off)) (hp : r ≠ .pinned) (hb : off + x.toNat + n ≤ K)
    (hs : off + x.toNat + n < 2 ^ 36) : Contracts.roomAt S (b + x) n = true := by
  have hd' : decodeAddr (b + x) = some (r, off + x.toNat) := by
    have := Sem.decodeAddr_add (i := x.toNat) hd (by simp only [regionSpan, UInt64.reduceToNat]; omega)
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

end AlgorithmLib.Prog
