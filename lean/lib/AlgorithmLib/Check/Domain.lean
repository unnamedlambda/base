module
public import AlgorithmLib.Check.Lin
import all Init.Data.Repr
@[expose] public section

/-!
# `Check.Domain` — what the checker knows of a value, and of an operation

An abstract value (`AV`) describes the values a slot may hold: one it knows, an
integer in a range, an address into a region at an offset in a range, a length
the caller passed (no more than its region holds), the outcome of a comparison
between two slots, a scalar of a known type, or anything. `AV.Desc Γ w` is what
each describes, in the environment and world it is read in.

`absOp` is what an operation leaves in its slot, and **`absOp_sound`** says it
is true: wherever the abstract state describes the concrete one, the
operation is defined and its value is one the result describes. Each case is
one transfer function; whatever none of them covers is typed by `tyOp` or
refused.
-/

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare
  AlgorithmLib.HProg.DevSpec AlgorithmLib.Prog

namespace AlgorithmLib.Check

variable {vals : List UInt64}

open AlgorithmLib.HProg.Contracts (Fact TState roomAt roomAt_spec holds_of_mem)

/-- How far into its region an address may point: the span of a region. -/
abbrev span : Nat := 2 ^ 36

/-- A bound against a region's size: `x * s + d` bytes of region `r`. -/
abbrev Bound := Region × Nat × Nat

/-- The bytes a bound names lie within its region, outside pinned memory. -/
@[reducible] def Fit (w : World) : Option Bound → Nat → Prop
  | none, _ => True
  | some (r, s, d), x => r ≠ .pinned ∧ x * s + d ≤ w.mem.sizes r

theorem Fit.transport {w w' : World} {f : Option Bound} {x : Nat} (h : Fit w f x)
    (hw : ∀ r, r ≠ .pinned → w'.mem.sizes r = w.mem.sizes r) : Fit w' f x := by
  match f, h with
  | none, _ => trivial
  | some (r, s, d), ⟨hp, hle⟩ => exact ⟨hp, by rw [hw r hp]; exact hle⟩

/-- What the checker knows of an integer: its type, a range its bits lie in as a
    number, a divisor (`0`: the value is zero), and a bound against a region. -/
structure Num where
  t : ClifTy
  lo : Nat
  hi : Nat
  al : Nat
  fit : Option Bound
  deriving Repr, Inhabited, DecidableEq

/-- The numbers a `Num` describes. -/
@[reducible] def Num.Holds (n : Num) (w : World) (x : Nat) : Prop :=
  n.lo ≤ x ∧ x ≤ n.hi ∧ n.al ∣ x ∧ Fit w n.fit x

/-- What the checker knows of a slot. -/
inductive AV where
  | known (v : V)
  /-- An integer scalar of type `n.t` whose bits, as a number, `n` describes. -/
  | int (n : Num)
  /-- The address `regionBase r + off`, `off` a number `n` describes, no
      further than the region's span: an address may sit just past its region. -/
  | ptr (r : Region) (n : Num)
  /-- The flag `icmp c` leaves on slots `a` and `b`. -/
  | cmp (c : ICmpCond) (a b : R)
  /-- A value of type `t`, contents unknown: a scalar, or a vector with its
      lanes. -/
  | sc (t : ClifTy)
  /-- A 64-bit handle of kind `k` held open; where `held` is false, that or
      null: what an open answers. -/
  | hand (k : Contracts.Held) (held : Bool)
  | any
  deriving Repr, Inhabited

/-- A length no larger than region `r` holds: what the caller passes. -/
def AV.len (r : Region) : AV := .int ⟨.i64, 0, span, 1, some (r, 1, 0)⟩

/-- The values an abstract value describes, read in `Γ` and `w`. -/
def AV.Desc (Γ : Env) (w : World) : AV → V → Prop
  | .known v', v => v = v'
  | .int n, v => ∃ x, v = .sc n.t x ∧ n.Holds w x.toNat
  | .ptr r n, v => ∃ off, v = .sc .i64 (regionBase r + UInt64.ofNat off) ∧ n.Holds w off ∧ n.hi ≤ span
  | .cmp c a b, v => ∃ t x y, Γ[a]? = some (.sc t x) ∧ Γ[b]? = some (.sc t y) ∧ t.isInt = true ∧
      v = boolV (cmpInt c t x y)
  | .sc t, v => TyV t v
  | .hand k held, v => ∃ x, v = .sc .i64 x ∧ ((held = false ∧ x = 0) ∨ k.live w x = true)
  | .any, _ => True

/-- The scalar type a value is known to have. -/
def AV.scTy : AV → Option ClifTy
  | .known (.sc t _) => some t
  | .int n => some n.t
  | .ptr .. => some .i64
  | .cmp .. => some .i8
  | .sc t => if t.lanes.isNone then some t else none
  | .hand .. => some .i64
  | _ => none

/-- The type a value is known to have, scalar or vector. -/
def AV.ty : AV → Option ClifTy
  | .sc t => some t
  | .known (.vec t xs) => if t.lanes.map (·.2) = some xs.size then some t else none
  | a => a.scTy.filter (·.lanes.isNone)

/-- The bits of a value known exactly. -/
def AV.bits : AV → Option UInt64
  | .known (.sc _ x) => some x
  | _ => none

def AV.isKnown : AV → Bool
  | .known _ => true
  | _ => false

/-- The slots a value's description reads. -/
def AV.refs : AV → List R
  | .cmp _ a b => [a, b]
  | _ => []

/-- The same value, described without reading other slots: what may be moved
    to another environment. -/
def AV.portable : AV → AV
  | .cmp .. => .sc .i8
  | v => v

theorem AV.scTy_desc {Γ : Env} {w : World} {av : AV} {v : V} {t : ClifTy} (hd : av.Desc Γ w v)
    (h : av.scTy = some t) : ∃ x, v = .sc t x := by
  cases av with
  | known u =>
      simp only [Desc] at hd; rw [hd]
      cases u with
      | sc t' x => simp only [scTy, Option.some.injEq] at h; subst h; exact ⟨x, rfl⟩
      | vec _ _ => cases h
  | int n =>
      simp only [scTy, Option.some.injEq] at h; subst h
      obtain ⟨x, rfl, -⟩ := hd; exact ⟨x, rfl⟩
  | ptr r n =>
      simp only [scTy, Option.some.injEq] at h; subst h
      obtain ⟨off, rfl, -⟩ := hd; exact ⟨_, rfl⟩
  | cmp c a b =>
      simp only [scTy, Option.some.injEq] at h; subst h
      obtain ⟨_, _, _, -, -, -, rfl⟩ := hd; exact ⟨_, rfl⟩
  | sc t' =>
      simp only [scTy] at h
      split at h
      · rename_i hl
        cases h
        simp only [Desc, TyV] at hd
        cases e : t.lanes with
        | none => rw [e] at hd; exact hd
        | some _ => rw [e] at hl; cases hl
      · cases h
  | hand k b =>
      simp only [scTy, Option.some.injEq] at h; subst h
      obtain ⟨x, rfl, -⟩ := hd; exact ⟨x, rfl⟩
  | any => cases h

theorem AV.ty_desc {Γ : Env} {w : World} {av : AV} {v : V} {t : ClifTy} (hd : av.Desc Γ w v)
    (h : av.ty = some t) : TyV t v := by
  have viaSc : av.scTy.filter (·.lanes.isNone) = some t → TyV t v := fun h => by
    obtain ⟨hs, hl⟩ := Option.filter_eq_some_iff.mp h
    obtain ⟨x, rfl⟩ := AV.scTy_desc hd hs
    exact TyV.sc (Option.isNone_iff_eq_none.mp hl)
  cases av with
  | sc t' => simp only [ty, Option.some.injEq] at h; subst h; exact hd
  | known u =>
      cases u with
      | vec t' xs =>
          simp only [ty] at h
          split at h
          · rename_i hn
            cases h
            simp only [Desc] at hd; subst hd
            cases e : t.lanes with
            | none => rw [e] at hn; cases hn
            | some p =>
                rw [e] at hn
                simp only [Option.map_some, Option.some.injEq] at hn
                exact TyV.vec e hn.symm
          · cases h
      | sc _ _ => exact viaSc h
  | _ => exact viaSc h

theorem AV.bits_desc {Γ : Env} {w : World} {av : AV} {v : V} {x : UInt64} (hd : av.Desc Γ w v)
    (h : av.bits = some x) : ∃ t, v = .sc t x := by
  cases av with
  | known u =>
      simp only [Desc] at hd; rw [hd]
      cases u with
      | sc t y => simp only [bits, Option.some.injEq] at h; subst h; exact ⟨t, rfl⟩
      | vec _ _ => cases h
  | _ => cases h

theorem AV.portable_desc {Γ Γ' : Env} {w : World} {av : AV} {v : V} (hd : av.Desc Γ w v) :
    av.portable.Desc Γ' w v := by
  cases av with
  | cmp c a b => obtain ⟨_, _, _, -, -, -, rfl⟩ := hd; exact ⟨_, rfl⟩
  | _ => exact hd

/-- A description reads only the slots `refs` names, and the world only
    through region sizes other than the pinned one's and the handles held. -/
theorem AV.desc_transport {Γ Γ' : Env} {w w' : World} {av : AV} {v : V} (hd : av.Desc Γ w v)
    (hΓ : ∀ a ∈ av.refs, Γ'[a]? = Γ[a]?) (hw : ∀ r, r ≠ .pinned → w'.mem.sizes r = w.mem.sizes r)
    (hh : ∀ (k : Contracts.Held) b, av = .hand k b → ∀ x, k.live w x = true → k.live w' x = true) :
    av.Desc Γ' w' v := by
  cases av with
  | int n =>
      obtain ⟨x, rfl, h1, h2, h3, h4⟩ := hd
      exact ⟨x, rfl, h1, h2, h3, h4.transport hw⟩
  | ptr r n =>
      obtain ⟨off, rfl, ⟨h1, h2, h3, h4⟩, h5⟩ := hd
      exact ⟨off, rfl, ⟨h1, h2, h3, h4.transport hw⟩, h5⟩
  | cmp c a b =>
      obtain ⟨t, x, y, ha, hb, ht, rfl⟩ := hd
      exact ⟨t, x, y, by rw [hΓ a (by simp [refs])]; exact ha, by rw [hΓ b (by simp [refs])]; exact hb, ht, rfl⟩
  | known _ => exact hd
  | sc _ => exact hd
  | hand k b =>
      obtain ⟨x, rfl, h⟩ := hd
      exact ⟨x, rfl, h.imp id (hh k b rfl x)⟩
  | any => exact hd

-- ---------------------------------------------------------------------------
-- States
-- ---------------------------------------------------------------------------

/-- What a slot holds before anything is put in it. -/
class Blank (α : Type) where
  blank : α

instance : Blank AV := ⟨.any⟩

instance : Blank Sym := ⟨{}⟩

/-- What the checker knows of each slot, by index: slot `0` at the root, slot
    `n + 1` in the left subtree at `n / 2` when `n` is even and in the right
    one when it is odd. Slots filled in order make a balanced tree, so a slot
    is read in steps logarithmic in how many there are. -/
inductive Slots (α : Type) where
  | leaf
  | node (l : Slots α) (x : α) (r : Slots α)
  deriving Inhabited

namespace Slots

variable {α : Type} [Blank α]

def get : Slots α → Nat → α
  | .leaf, _ => Blank.blank
  | .node _ x _, 0 => x
  | .node l _ r, n + 1 => if n % 2 = 0 then l.get (n / 2) else r.get (n / 2)

/-- `v` at slot `n` and nothing else; `f` bounds how deep `n` sits. -/
def single : Nat → Nat → α → Slots α
  | _, 0, v => .node .leaf v .leaf
  | 0, _ + 1, _ => .leaf
  | f + 1, n + 1, v =>
      if n % 2 = 0 then .node (single f (n / 2) v) Blank.blank .leaf
      else .node .leaf Blank.blank (single f (n / 2) v)

def set : Slots α → Nat → α → Slots α
  | .leaf, n, v => single n n v
  | .node l _ r, 0, v => .node l v r
  | .node l x r, n + 1, v =>
      if n % 2 = 0 then .node (l.set (n / 2) v) x r else .node l x (r.set (n / 2) v)

def map (f : α → α) : Slots α → Slots α
  | .leaf => .leaf
  | .node l x r => .node (l.map f) (f x) (r.map f)

def zip (f : α → α → α) : Slots α → Slots α → Slots α
  | .leaf, b => b.map (f Blank.blank)
  | .node l x r, .leaf => (Slots.node l x r).map (f · Blank.blank)
  | .node l x r, .node l' x' r' => .node (zip f l l') (f x x') (zip f r r')

theorem get_single : ∀ (f n : Nat) (v : α) (m : Nat), n ≤ f →
    (single f n v).get m = if m = n then v else Blank.blank
  | _, 0, v, 0, _ => by simp [single, get]
  | _, 0, v, m + 1, _ => by simp only [single, get]; split <;> simp [get]
  | 0, _ + 1, _, _, h => by omega
  | f + 1, n + 1, v, m, h => by
      simp only [single]
      split
      · rename_i hn
        cases m with
        | zero => simp [get]
        | succ m =>
            simp only [get]
            split
            · rename_i hm
              rw [get_single f (n / 2) v (m / 2) (by omega)]
              by_cases e : m = n
              · subst e; simp
              · rw [if_neg (by omega), if_neg (by omega)]
            · rw [if_neg (by omega)]
      · rename_i hn
        cases m with
        | zero => simp [get]
        | succ m =>
            simp only [get]
            split
            · rw [if_neg (by omega)]
            · rename_i hm
              rw [get_single f (n / 2) v (m / 2) (by omega)]
              by_cases e : m = n
              · subst e; simp
              · rw [if_neg (by omega), if_neg (by omega)]

theorem get_set : ∀ (t : Slots α) (i : Nat) (v : α) (j : Nat), (t.set i v).get j = if j = i then v else t.get j
  | .leaf, i, v, j => by
      simp only [set]
      rw [get_single i i v j (Nat.le_refl _)]
      split <;> rfl
  | .node l x r, 0, v, j => by
      cases j with
      | zero => simp [set, get]
      | succ j => simp [set, get]
  | .node l x r, i + 1, v, j => by
      simp only [set]
      split
      · rename_i hi
        cases j with
        | zero => simp [get]
        | succ j =>
            simp only [get]
            split
            · rename_i hj
              rw [get_set l (i / 2) v (j / 2)]
              by_cases e : j = i
              · subst e; simp
              · rw [if_neg (by omega), if_neg (by omega)]
            · rw [if_neg (by omega)]
      · rename_i hi
        cases j with
        | zero => simp [get]
        | succ j =>
            simp only [get]
            split
            · rw [if_neg (by omega)]
            · rename_i hj
              rw [get_set r (i / 2) v (j / 2)]
              by_cases e : j = i
              · subst e; simp
              · rw [if_neg (by omega), if_neg (by omega)]

theorem get_map {f : α → α} (hf : f Blank.blank = Blank.blank) :
    ∀ (t : Slots α) (i : Nat), (t.map f).get i = f (t.get i)
  | .leaf, _ => by simp [map, get, hf]
  | .node l x r, 0 => rfl
  | .node l x r, i + 1 => by
      simp only [map, get]
      split
      · exact get_map hf l _
      · exact get_map hf r _

theorem get_zip {f : α → α → α} (hf : f Blank.blank Blank.blank = Blank.blank) :
    ∀ (a b : Slots α) (i : Nat), (zip f a b).get i = f (a.get i) (b.get i)
  | .leaf, b, i => by simp only [zip]; rw [get_map hf]; rfl
  | .node l x r, .leaf, i => by
      simp only [zip]; rw [get_map (f := fun a => f a Blank.blank) hf]; cases i <;> rfl
  | .node l x r, .node l' x' r', 0 => rfl
  | .node l x r, .node l' x' r', i + 1 => by
      simp only [zip, get]
      split
      · exact get_zip hf l l' _
      · exact get_zip hf r r' _

end Slots

/-- A number computed from the program's parameters, the numbers its
    typestate names but does not fix, as 64-bit arithmetic computes it. -/
inductive PExpr where
  | lit (c : UInt64)
  | param (i : Nat)
  | mul (a b : PExpr)
  | shl (a : PExpr) (k : Nat)
  | trunc32 (a : PExpr)
  deriving DecidableEq, Repr, Inhabited

def PExpr.eval (vals : List UInt64) : PExpr → UInt64
  | .lit c => c
  | .param i => vals.getD i 0
  | .mul a b => a.eval vals * b.eval vals
  | .shl a k => a.eval vals <<< UInt64.ofNat k
  | .trunc32 a => a.eval vals &&& 0xFFFFFFFF

/-- An argument a call is handed: bits known, or bits computed from the
    parameters. -/
inductive PArg where
  | known (x : UInt64)
  | sym (e : PExpr)
  deriving DecidableEq, Repr, Inhabited

def PArg.bits (vals : List UInt64) : PArg → UInt64
  | .known x => x
  | .sym e => e.eval vals

/-- The device buffers sized by the parameters, as facts. -/
def bufFacts (bufs : List (UInt64 × PExpr)) (vals : List UInt64) : TState :=
  bufs.map fun b => .devBuf b.1 (b.2.eval vals)

/-- A call whose precondition the parameters decide, written out for a proof
    to show: entry point `f` handed `args`, in a typestate of `facts` and the
    device buffers `bufs` sized by the parameters. -/
structure Obl where
  f : Ffi
  args : List PArg
  facts : TState
  bufs : List (UInt64 × PExpr)
  deriving DecidableEq, Repr

/-- An obligation shown: for the parameters' values, every world its typestate
    describes meets the call's precondition. -/
def Obl.Holds (o : Obl) (vals : List UInt64) : Prop :=
  ∀ w, TState.holds (o.facts ++ bufFacts o.bufs vals) w → Contracts.Pre o.f (o.args.map (PArg.bits vals)) w

/-- What the checker knows of the program's parameters, never their values:
    the slots holding a number computed from them, the cells holding one, the
    device buffers sized by them, and the calls whose preconditions they
    decide. In `collect`, a run that only gathers those calls. -/
structure Par where
  slots : List (R × ClifTy × PExpr) := []
  cells : List (UInt64 × Nat) := []
  bufs : List (UInt64 × PExpr) := []
  obls : List Obl := []
  collect : Bool := false
  deriving Inhabited, DecidableEq

/-- The cells and buffers the record names, as facts at the parameters'
    values. -/
def Par.facts (P : Par) (vals : List UInt64) : TState :=
  P.cells.map (fun c => .cell c.1 (vals.getD c.2 0)) ++ bufFacts P.bufs vals

/-- What the record says of an environment and a world, at the parameters'
    values `vals`. -/
def Par.At (P : Par) (vals : List UInt64) (Γ : Env) (w : World) : Prop :=
  P.collect = false ∧
  (∀ s ∈ P.slots, Γ[s.1]? = some (.sc s.2.1 (s.2.2.eval vals))) ∧
  (P.facts vals).holds w ∧
  (∀ o ∈ P.obls, (o.Holds vals))

/-- The record with what it says of the world forgotten. -/
def Par.clearWorld (P : Par) : Par := { P with cells := [], bufs := [] }

theorem Par.At.clearWorld {P : Par} {vals : List UInt64} {Γ : Env} {w : World} (h : P.At vals Γ w)
    (w' : World) : P.clearWorld.At vals Γ w' :=
  ⟨h.1, h.2.1, fun _ hx => by simp [Par.clearWorld, Par.facts, bufFacts] at hx, h.2.2.2⟩

/-- The record keeping, of the cells and buffers it names, those `k` keeps:
    a cell or a buffer read by where it is, whatever its value. -/
def Par.keepWorld (P : Par) (k : Fact → Bool) : Par :=
  { P with cells := P.cells.filter (fun c => k (.cell c.1 0)),
           bufs := P.bufs.filter (fun b => k (.devBuf b.1 0)) }

/-- What the record keeps holds in a world where what `k` keeps does. -/
theorem Par.At.keepWorld {P : Par} {vals : List UInt64} {Γ : Env} {w w' : World} (h : P.At vals Γ w)
    {k : Fact → Bool} (hc : ∀ a v, k (.cell a v) = k (.cell a 0)) (hb : ∀ b n, k (.devBuf b n) = k (.devBuf b 0))
    (hk : ∀ x, k x = true → x.holds w → x.holds w') : (P.keepWorld k).At vals Γ w' := by
  refine ⟨h.1, h.2.1, fun x hx => ?_, h.2.2.2⟩
  simp only [Par.keepWorld, Par.facts, bufFacts, List.mem_append, List.mem_map, List.mem_filter] at hx
  rcases hx with ⟨c, ⟨hcm, hkc⟩, rfl⟩ | ⟨d, ⟨hdm, hkd⟩, rfl⟩
  · exact hk _ (by rw [hc]; exact hkc) (h.2.2.1 _ (List.mem_append_left _ (List.mem_map_of_mem hcm)))
  · exact hk _ (by rw [hb]; exact hkd) (h.2.2.1 _ (List.mem_append_right _ (List.mem_map_of_mem hdm)))

/-- What two records both say: nothing but the calls gathered, where the run
    gathers them. -/
def Par.join (P Q : Par) : Par :=
  { collect := P.collect && Q.collect, obls := if P.collect && Q.collect then P.obls ++ Q.obls else [] }

theorem Par.join_at {P : Par} (Q : Par) {vals : List UInt64} {Γ : Env} {w : World} (hc : P.collect = false) :
    (P.join Q).At vals Γ w :=
  ⟨by simp [Par.join, hc], fun _ e => by simp [Par.join] at e,
    fun _ e => by simp [Par.join, Par.facts, bufFacts] at e,
    fun _ e => by simp only [Par.join, hc, Bool.false_and, Bool.false_eq_true, if_false, List.not_mem_nil] at e⟩

theorem Par.At.join_left {P : Par} {vals : List UInt64} {Γ : Env} {w : World} (h : P.At vals Γ w) (Q : Par) :
    (P.join Q).At vals Γ w :=
  Par.join_at Q h.1

theorem Par.At.join_right {Q : Par} {vals : List UInt64} {Γ : Env} {w : World} (h : Q.At vals Γ w) (P : Par) :
    (P.join Q).At vals Γ w := by
  have e : P.join Q = Q.join P := by simp [Par.join, Bool.and_comm, h.1]
  rw [e]; exact Par.join_at P h.1

/-- The record with the slots from `n` on forgotten. -/
def Par.cut (P : Par) (n : Nat) : Par := { P with slots := P.slots.filter (·.1 < n) }

theorem Par.At.cut {P : Par} {vals : List UInt64} {Γ Γ' : Env} {w : World} (h : P.At vals Γ w) {n : Nat}
    (hΓ : ∀ r, r < n → Γ'[r]? = Γ[r]?) : (P.cut n).At vals Γ' w := by
  refine ⟨h.1, fun s hs => ?_, h.2.2.1, h.2.2.2⟩
  obtain ⟨hs, hn⟩ := List.mem_filter.mp hs
  rw [hΓ _ (by simpa using hn)]
  exact h.2.1 s hs

/-- What the record says of slots holds as more are appended. -/
theorem Par.At.extend {P : Par} {vals : List UInt64} {Γ Γ' : Env} {w : World} (h : P.At vals Γ w)
    (hΓ : ∀ r, r < Γ.size → Γ'[r]? = Γ[r]?) : P.At vals Γ' w := by
  refine ⟨h.1, fun s hs => ?_, h.2.2.1, h.2.2.2⟩
  have e := h.2.1 s hs
  have hlt : s.1 < Γ.size := by
    rcases Nat.lt_or_ge s.1 Γ.size with h' | h'
    · exact h'
    · rw [Array.getElem?_eq_none h'] at e; cases e
  rw [hΓ _ hlt, e]

/-- The number slot `r` holds, where it holds one computed from the
    parameters. -/
def Par.slotOf (P : Par) (r : R) : Option (ClifTy × PExpr) := (P.slots.find? (·.1 == r)).map (·.2)

theorem Par.At.slotOf {P : Par} {vals : List UInt64} {Γ : Env} {w : World} (h : P.At vals Γ w) {r : R}
    {t : ClifTy} {e : PExpr} (he : P.slotOf r = some (t, e)) : Γ[r]? = some (.sc t (e.eval vals)) := by
  unfold Par.slotOf at he
  obtain ⟨s, hs, hse⟩ := Option.map_eq_some_iff.mp he
  have hm := List.mem_of_find?_eq_some hs
  have hr : s.1 = r := by simpa using List.find?_some hs
  rw [← hr, h.2.1 s hm, hse]

/-- The checker's state: what it knows of each of `len` slots, and of the
    world; beyond each slot's range, the linear facts `syms` holds; and what
    it knows of the program's parameters. -/
structure AS where
  slots : Slots AV
  len : Nat
  facts : TState
  syms : Slots Sym
  par : Par := {}

def AS.get (A : AS) (r : R) : AV := if r < A.len then A.slots.get r else .any

/-- What the checker knows of a slot beyond its range. -/
def AS.sym (A : AS) (r : R) : Sym := if r < A.len then A.syms.get r else {}

/-- The same state, knowing one more fact of the same world. -/
def AS.addFact (A : AS) (f : Fact) : AS := { A with facts := f :: A.facts }

/-- The same slots, with other facts, and what the parameters' record says of
    the world forgotten. -/
def AS.withFacts (A : AS) (S : TState) : AS := { A with facts := S, par := A.par.clearWorld }

/-- A slot's linear facts hold, and read only slots before `n` and sizes of
    regions other than the pinned one. -/
def Sym.FitsAt (σ : Sym) (Γ : Env) (w : World) (x : Int) (n : Nat) : Prop :=
  σ.Holds Γ w x ∧ (∀ y ∈ σ.slots, y < n) ∧ ∀ g ∈ σ.regions, g ≠ .pinned

/-- A slot's linear facts hold of its bits. -/
abbrev Sym.Fits (σ : Sym) (Γ : Env) (w : World) (r n : Nat) : Prop := σ.FitsAt Γ w (slotVal Γ r) n

/-- The environments and worlds an abstract state describes. -/
def AS.Holds (A : AS) (vals : List UInt64) (Γ : Env) (w : World) : Prop :=
  (Γ.size = A.len ∧ (∀ i, i < A.len → (A.syms.get i).Fits Γ w i A.len) ∧ A.par.At vals Γ w) ∧
    (∀ i, i < A.len → ∃ v, Γ[i]? = some v ∧ (A.get i).Desc Γ w v) ∧
    A.facts.holds w

theorem AS.get_out {A : AS} {r : R} (hr : ¬ r < A.len) : A.get r = .any := by
  unfold AS.get; rw [if_neg hr]

theorem AS.get_lt_scTy {A : AS} {r : R} {t : ClifTy} (h : (A.get r).scTy = some t) : r < A.len := by
  rcases Nat.lt_or_ge r A.len with hl | hl
  · exact hl
  · rw [AS.get_out (Nat.not_lt_of_ge hl)] at h; cases h

theorem AS.get_lt_known {A : AS} {r : R} (hk : (A.get r).isKnown = true) : r < A.len := by
  rcases Nat.lt_or_ge r A.len with hl | hl
  · exact hl
  · rw [AS.get_out (Nat.not_lt_of_ge hl)] at hk; cases hk

theorem AS.Holds.get {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {r : R} (hr : r < A.len) :
    ∃ v, Γ[r]? = some v ∧ (A.get r).Desc Γ w v := h.2.1 r hr

/-- A slot known to be a scalar holds one. -/
theorem AS.Holds.getSc {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {r : R} {t : ClifTy}
    (ht : (A.get r).scTy = some t) : ∃ x, Γ[r]? = some (.sc t x) ∧ (A.get r).Desc Γ w (.sc t x) := by
  obtain ⟨v, hv, hd⟩ := h.get (AS.get_lt_scTy ht)
  obtain ⟨x, rfl⟩ := AV.scTy_desc hd ht
  exact ⟨x, hv, hd⟩

/-- What the checker knows after a slot is appended, with what it knows of
    it beyond its range. -/
def AS.pushS (A : AS) (av : AV) (σ : Sym) : AS :=
  ⟨A.slots.set A.len av, A.len + 1, A.facts, A.syms.set A.len σ, A.par⟩

/-- What the checker knows after a slot is appended. -/
def AS.push (A : AS) (av : AV) : AS := A.pushS av {}

/-- Slots appended in order, each with its linear facts. -/
def AS.pushAll (B : AS) (xs : List (AV × Sym)) : AS := xs.foldl (fun B (av, σ) => B.pushS av σ) B

/-- A state of the slots a list describes, in order. -/
def AS.ofList (l : List AV) (S : TState) : AS := (⟨.leaf, 0, S, .leaf, {}⟩ : AS).pushAll (l.map (·, {}))

theorem AS.get_pushS_lt {A : AS} {av : AV} {σ : Sym} {i : R} (hi : i < A.len) :
    (A.pushS av σ).get i = A.get i := by
  show (if i < A.len + 1 then (A.slots.set A.len av).get i else .any) =
    (if i < A.len then A.slots.get i else .any)
  rw [if_pos (Nat.lt_succ_of_lt hi), if_pos hi, Slots.get_set, if_neg (Nat.ne_of_lt hi)]

theorem AS.get_pushS_eq {A : AS} {av : AV} {σ : Sym} : (A.pushS av σ).get A.len = av := by
  show (if A.len < A.len + 1 then (A.slots.set A.len av).get A.len else .any) = av
  rw [if_pos (Nat.lt_succ_self _), Slots.get_set, if_pos rfl]

theorem AS.get_push_lt {A : AS} {av : AV} {i : R} (hi : i < A.len) : (A.push av).get i = A.get i :=
  AS.get_pushS_lt hi

theorem AS.get_push_eq {A : AS} {av : AV} : (A.push av).get A.len = av := AS.get_pushS_eq

theorem slotVal_push {Γ : Env} {v : V} {r : R} (hr : r < Γ.size) : slotVal (Γ.push v) r = slotVal Γ r := by
  unfold slotVal; rw [Array.getElem?_push_lt hr, Array.getElem?_eq_getElem hr]

theorem Lin.eval_congr {Γ Γ' : Env} {w w' : World} {L : Lin}
    (hΓ : ∀ x ∈ L.slots, slotVal Γ' x = slotVal Γ x) (hw : ∀ g ∈ L.regions, w'.mem.sizes g = w.mem.sizes g) :
    L.eval Γ' w' = L.eval Γ w := by
  obtain ⟨t, k⟩ := L
  simp only [Lin.eval]
  congr 1
  induction t with
  | nil => rfl
  | cons p t ih =>
      obtain ⟨a, c⟩ := p
      simp only [Lin.slots, Lin.regions, List.filterMap_cons] at hΓ hw ih ⊢
      simp only [Lin.evalT]
      congr 1
      · cases a with
        | slot r => simp only [Atom.val]; rw [hΓ r (by simp)]
        | size g => simp only [Atom.val]; rw [hw g (by simp)]
        | iun k r => simp only [Atom.val]; rw [hΓ r (by simp)]
      · exact ih (fun x hx => hΓ x (by cases a <;> simp_all)) (fun g hg => hw g (by cases a <;> simp_all))

/-- Facts read in an environment and world that agree on what they read hold
    there too. -/
theorem Sym.fitsAt_congr {σ : Sym} {Γ Γ' : Env} {w w' : World} {x : Int} {n m : Nat} (h : σ.FitsAt Γ w x n)
    (hΓ : ∀ y, y < n → slotVal Γ' y = slotVal Γ y)
    (hw : ∀ g, g ≠ .pinned → w'.mem.sizes g = w.mem.sizes g) (hnm : n ≤ m) : σ.FitsAt Γ' w' x m := by
  obtain ⟨⟨hd, hu⟩, hs, hg⟩ := h
  have he : ∀ L : Lin, (∀ y ∈ L.slots, y ∈ σ.slots) → (∀ g ∈ L.regions, g ∈ σ.regions) →
      L.eval Γ' w' = L.eval Γ w := fun L h1 h2 =>
    Lin.eval_congr (fun y hy => hΓ y (hs y (h1 y hy))) (fun g hg' => hw g (hg g (h2 g hg')))
  refine ⟨⟨fun D hD => ?_, fun p hp => ?_⟩, fun y hy => Nat.lt_of_lt_of_le (hs y hy) hnm, hg⟩
  · rw [he D (fun y hy => by simp [Sym.slots, hD, hy]) (fun g hg' => by simp [Sym.regions, hD, hg'])]
    exact hd D hD
  · rw [he p.2 (fun y hy => by
        simp only [Sym.slots, List.mem_append, List.mem_flatten, List.mem_map]
        exact .inr ⟨_, ⟨p, hp, rfl⟩, hy⟩) (fun g hg' => by
        simp only [Sym.regions, List.mem_append, List.mem_flatten, List.mem_map]
        exact .inr ⟨_, ⟨p, hp, rfl⟩, hg'⟩)]
    exact hu p hp

/-- A slot's linear facts hold of the same bits wherever the slots and sizes
    they read are the same. -/
theorem Sym.fits_congr {σ : Sym} {Γ Γ' : Env} {w w' : World} {r n m : Nat} (h : σ.Fits Γ w r n)
    (hΓ : ∀ x, x < n → slotVal Γ' x = slotVal Γ x) (hr : slotVal Γ' r = slotVal Γ r)
    (hw : ∀ g, g ≠ .pinned → w'.mem.sizes g = w.mem.sizes g) (hnm : n ≤ m) : σ.Fits Γ' w' r m := by
  show σ.FitsAt Γ' w' (slotVal Γ' r) m
  rw [hr]
  exact Sym.fitsAt_congr h hΓ hw hnm

/-- Fewer facts fit where more do. -/
theorem Sym.fits_sub {σ τ : Sym} {Γ : Env} {w : World} {x : Int} {m : Nat} (h : σ.FitsAt Γ w x m)
    (hd : ∀ D, τ.dfn = some D → σ.dfn = some D) (hu : ∀ p ∈ τ.ubs, p ∈ σ.ubs) : τ.FitsAt Γ w x m := by
  obtain ⟨⟨h1, h2⟩, h3, h4⟩ := h
  refine ⟨⟨fun D e => h1 D (hd D e), fun p hp => h2 p (hu p hp)⟩, fun y hy => h3 y ?_, fun g hg => h4 g ?_⟩
  · simp only [Sym.slots, List.mem_append, List.mem_flatten, List.mem_map] at hy ⊢
    rcases hy with hy | ⟨_, ⟨p, hp, rfl⟩, hy⟩
    · left; cases e : τ.dfn with
      | none => rw [e] at hy; simp at hy
      | some D => rw [e] at hy; rw [hd D e]; exact hy
    · exact .inr ⟨_, ⟨p, hu p hp, rfl⟩, hy⟩
  · simp only [Sym.regions, List.mem_append, List.mem_flatten, List.mem_map] at hg ⊢
    rcases hg with hg | ⟨_, ⟨p, hp, rfl⟩, hg⟩
    · left; cases e : τ.dfn with
      | none => rw [e] at hg; simp at hg
      | some D => rw [e] at hg; rw [hd D e]; exact hg
    · exact .inr ⟨_, ⟨p, hu p hp, rfl⟩, hg⟩

/-- Facts no stronger than ones that fit, fit: each bound one with the same
    multiple and terms, and a constant no smaller. -/
theorem Sym.fits_weaken {σ τ : Sym} {Γ : Env} {w : World} {x : Int} {m : Nat} (h : σ.FitsAt Γ w x m)
    (hd : ∀ D, τ.dfn = some D → σ.dfn = some D)
    (hu : ∀ p ∈ τ.ubs, ∃ p' ∈ σ.ubs, p'.1 = p.1 ∧ p'.2.terms = p.2.terms ∧ p'.2.k ≤ p.2.k) :
    τ.FitsAt Γ w x m := by
  obtain ⟨⟨h1, h2⟩, h3, h4⟩ := h
  refine ⟨⟨fun D e => h1 D (hd D e), fun p hp => ?_⟩, fun y hy => h3 y ?_, fun g hg => h4 g ?_⟩
  · obtain ⟨p', hp', e1, e2, e3⟩ := hu p hp
    have := h2 p' hp'
    rw [e1] at this
    simp only [Lin.eval, e2] at this ⊢
    omega
  · simp only [Sym.slots, List.mem_append, List.mem_flatten, List.mem_map] at hy ⊢
    rcases hy with hy | ⟨_, ⟨p, hp, rfl⟩, hy⟩
    · left; cases e : τ.dfn with
      | none => rw [e] at hy; simp at hy
      | some D => rw [e] at hy; rw [hd D e]; exact hy
    · obtain ⟨p', hp', -, e2, -⟩ := hu p hp
      exact .inr ⟨_, ⟨p', hp', rfl⟩, by unfold Lin.slots at hy ⊢; rw [e2]; exact hy⟩
  · simp only [Sym.regions, List.mem_append, List.mem_flatten, List.mem_map] at hg ⊢
    rcases hg with hg | ⟨_, ⟨p, hp, rfl⟩, hg⟩
    · left; cases e : τ.dfn with
      | none => rw [e] at hg; simp at hg
      | some D => rw [e] at hg; rw [hd D e]; exact hg
    · obtain ⟨p', hp', -, e2, -⟩ := hu p hp
      exact .inr ⟨_, ⟨p', hp', rfl⟩, by unfold Lin.regions at hg ⊢; rw [e2]; exact hg⟩

/-- Facts that read only slots before `k` fit below `k`. -/
theorem Sym.fits_bound {σ : Sym} {Γ : Env} {w : World} {x : Int} {m k : Nat} (h : σ.FitsAt Γ w x m)
    (hk : ∀ y ∈ σ.slots, y < k) : σ.FitsAt Γ w x k :=
  ⟨h.1, hk, h.2.2⟩

theorem Sym.fitsAt_blank (Γ : Env) (w : World) (x : Int) (n : Nat) : ({} : Sym).FitsAt Γ w x n :=
  ⟨Sym.holds_blank Γ w _, fun _ h => by simp [Sym.slots] at h, fun _ h => by simp [Sym.regions] at h⟩

theorem Sym.fits_blank (Γ : Env) (w : World) (r n : Nat) : ({} : Sym).Fits Γ w r n :=
  Sym.fitsAt_blank Γ w _ n

/-- The state with another record of the parameters. -/
def AS.withPar (A : AS) (P : Par) : AS := { A with par := P }

theorem AS.Holds.withPar {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {P : Par} (hP : P.At vals Γ w) :
    (A.withPar P).Holds vals Γ w :=
  ⟨⟨h.1.1, h.1.2.1, hP⟩, h.2.1, h.2.2⟩

/-- The same slots, with other facts, and of the parameters' cells and
    buffers those `k` keeps. -/
def AS.withFactsK (A : AS) (S : TState) (k : Fact → Bool) : AS := (A.withFacts S).withPar (A.par.keepWorld k)

/-- The state knowing slot `r` holds a number computed from the parameters,
    where `x` gives one. -/
def AS.parSlot (A : AS) (r : R) : Option (ClifTy × PExpr) → AS
  | some (t, e) => A.withPar { A.par with slots := (r, t, e) :: A.par.slots }
  | none => A

theorem AS.Holds.parSlot {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w)
    {r : R} {x : Option (ClifTy × PExpr)}
    (hx : ∀ t e, x = some (t, e) → Γ[r]? = some (.sc t (e.eval vals))) : (A.parSlot r x).Holds vals Γ w := by
  cases x with
  | none => exact h
  | some p =>
    obtain ⟨t, e⟩ := p
    have hP := h.1.2.2
    refine h.withPar ⟨hP.1, fun s hs => ?_, hP.2.2.1, hP.2.2.2⟩
    rcases List.mem_cons.mp hs with rfl | hs
    · exact hx t e rfl
    · exact hP.2.1 s hs

theorem AS.Holds.pushS {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {av : AV} {v : V} {σ : Sym}
    (hv : av.Desc (Γ.push v) w v) (hσ : σ.Fits (Γ.push v) w A.len (A.len + 1)) :
    (A.pushS av σ).Holds vals (Γ.push v) w := by
  obtain ⟨⟨hs, hy, hp⟩, he, hf⟩ := h
  refine ⟨⟨by simp [AS.pushS, hs], fun i hi => ?_, hp.extend fun r hr => by
    rw [Array.getElem?_push, if_neg (by omega)]⟩, fun i hi => ?_, hf⟩
  · simp only [AS.pushS] at hi ⊢
    rw [Slots.get_set]
    split
    · rename_i e; subst e; exact hσ
    · rename_i hne
      have hi' : i < A.len := by omega
      exact Sym.fits_congr (hy i hi') (fun x hx => slotVal_push (Γ := Γ) (v := v) (hs ▸ hx))
        (slotVal_push (Γ := Γ) (v := v) (hs ▸ hi')) (fun _ _ => rfl) (Nat.le_succ _)
  simp only [AS.pushS] at hi
  by_cases hlt : i < A.len
  · obtain ⟨u, hu, hd⟩ := he i hlt
    refine ⟨u, by rw [Array.getElem?_push, if_neg (by omega)]; exact hu, ?_⟩
    rw [AS.get_pushS_lt hlt]
    refine AV.desc_transport hd (fun a ha => ?_) (fun _ _ => rfl) (fun _ _ _ _ h => h)
    rcases Nat.lt_or_ge a Γ.size with ha' | ha'
    · rw [Array.getElem?_push, if_neg (by omega)]
    · -- a slot past the environment reads nothing, so no description names one
      exfalso
      revert hd ha
      cases A.get i with
      | cmp c x y =>
          intro hd ha
          obtain ⟨_, _, _, hx, hy, -⟩ := hd
          simp only [AV.refs, List.mem_cons, List.not_mem_nil, or_false] at ha
          rcases ha with rfl | rfl
          · rw [Array.getElem?_eq_none (by omega)] at hx; cases hx
          · rw [Array.getElem?_eq_none (by omega)] at hy; cases hy
      | _ => intro _ ha; simp [AV.refs] at ha
  · have hi' : i = A.len := by omega
    subst hi'
    refine ⟨v, by rw [Array.getElem?_push, if_pos (by omega)], ?_⟩
    rw [AS.get_pushS_eq]; exact hv

theorem AS.Holds.push {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {av : AV} {v : V}
    (hv : av.Desc (Γ.push v) w v) : (A.push av).Holds vals (Γ.push v) w :=
  h.pushS hv (Sym.fits_blank _ _ _ _)

/-- The same environment, a world the new facts describe, reached by a step that
    keeps the region sizes and each handle a slot is described as. -/
theorem AS.Holds.worldP {A : AS} {Γ : Env} {w w' : World} (h : A.Holds vals Γ w) {S : TState}
    (hS : S.holds w') (hw : ∀ r, r ≠ .pinned → w'.mem.sizes r = w.mem.sizes r)
    (hh : ∀ i (k : Contracts.Held) b, A.get i = .hand k b → ∀ x, k.live w x = true → k.live w' x = true) :
    (A.withFacts S).Holds vals Γ w' :=
  ⟨⟨h.1.1, fun i hi => Sym.fits_congr (h.1.2.1 i hi) (fun _ _ => rfl) rfl hw (Nat.le_refl _),
    h.1.2.2.clearWorld w'⟩, fun i hi => by
    obtain ⟨v, hv, hd⟩ := h.2.1 i hi
    exact ⟨v, hv, AV.desc_transport hd (fun _ _ => rfl) hw (fun k b e => hh i k b e)⟩, hS⟩

/-- The same, reached by a step that keeps every handle held. -/
theorem AS.Holds.world {A : AS} {Γ : Env} {w w' : World} (h : A.Holds vals Γ w) {S : TState}
    (hS : S.holds w') (hw : ∀ r, r ≠ .pinned → w'.mem.sizes r = w.mem.sizes r)
    (hh : ∀ (k : Contracts.Held) x, k.live w x = true → k.live w' x = true) :
    (A.withFacts S).Holds vals Γ w' :=
  h.worldP hS hw fun _ k _ _ => hh k

/-- Whether a value is described as a handle. -/
def AV.isHand : AV → Bool
  | .hand .. => true
  | _ => false

/-- A value described without the handle it may be: what a step that may close
    any handle leaves of it. -/
def AV.forget : AV → AV
  | .hand .. => .sc .i64
  | a => a

theorem AV.forget_desc {Γ : Env} {w : World} {av : AV} {v : V} (hd : av.Desc Γ w v) : av.forget.Desc Γ w v := by
  cases av with
  | hand k b => obtain ⟨x, rfl, -⟩ := hd; exact TyV.sc (t := .i64) rfl
  | _ => exact hd

/-- A description that is no handle reads the world only through region sizes. -/
theorem AV.desc_transport_nh {Γ Γ' : Env} {w w' : World} {av : AV} {v : V} (hd : av.Desc Γ w v)
    (hΓ : ∀ a ∈ av.refs, Γ'[a]? = Γ[a]?) (hw : ∀ r, r ≠ .pinned → w'.mem.sizes r = w.mem.sizes r)
    (hn : av.isHand = false) : av.Desc Γ' w' v := by
  cases av with
  | int n =>
      obtain ⟨x, rfl, h1, h2, h3, h4⟩ := hd
      exact ⟨x, rfl, h1, h2, h3, h4.transport hw⟩
  | ptr r n =>
      obtain ⟨off, rfl, ⟨h1, h2, h3, h4⟩, h5⟩ := hd
      exact ⟨off, rfl, ⟨h1, h2, h3, h4.transport hw⟩, h5⟩
  | cmp c a b =>
      obtain ⟨t, x, y, ha, hb, ht, rfl⟩ := hd
      exact ⟨t, x, y, by rw [hΓ a (by simp [refs])]; exact ha, by rw [hΓ b (by simp [refs])]; exact hb, ht, rfl⟩
  | known _ => exact hd
  | sc _ => exact hd
  | hand k b => cases hn
  | any => exact hd

/-- The state with every handle forgotten. -/
def AS.forgetHands (A : AS) : AS := { A with slots := A.slots.map AV.forget }

/-- No slot is described as a handle. -/
def AS.NoHands (A : AS) : Prop := ∀ i, (A.get i).isHand = false

theorem AS.forgetHands_get (A : AS) (i : R) : A.forgetHands.get i = (A.get i).forget := by
  unfold AS.forgetHands AS.get
  dsimp only
  split
  · exact Slots.get_map (f := AV.forget) rfl _ _
  · rfl

theorem AS.forgetHands_noHands (A : AS) : A.forgetHands.NoHands := fun i => by
  rw [AS.forgetHands_get]; cases A.get i <;> rfl

theorem AS.Holds.forgetHands {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) : A.forgetHands.Holds vals Γ w :=
  ⟨h.1, fun i hi => by
    obtain ⟨v, hv, hd⟩ := h.2.1 i hi
    exact ⟨v, hv, by rw [AS.forgetHands_get]; exact AV.forget_desc hd⟩, h.2.2⟩

/-- The same environment, a world the new facts describe, reached by a step that
    keeps the region sizes, from a state that names no handle. -/
theorem AS.Holds.worldNH {A : AS} {Γ : Env} {w w' : World} (h : A.Holds vals Γ w) (hn : A.NoHands) {S : TState}
    (hS : S.holds w') (hw : ∀ r, r ≠ .pinned → w'.mem.sizes r = w.mem.sizes r) :
    (A.withFacts S).Holds vals Γ w' :=
  h.worldP hS hw fun i k b e => by have := hn i; rw [e] at this; cases this

/-- A value described without the handles of kind `k` it may be. -/
def AV.forgetK (k : Contracts.Held) : AV → AV
  | .hand k' b => if k' = k then .sc .i64 else .hand k' b
  | a => a

theorem AV.forgetK_desc {Γ : Env} {w : World} {k : Contracts.Held} {av : AV} {v : V} (hd : av.Desc Γ w v) :
    (av.forgetK k).Desc Γ w v := by
  cases av with
  | hand k' b =>
      simp only [AV.forgetK]
      split
      · obtain ⟨x, rfl, -⟩ := hd; exact TyV.sc (t := .i64) rfl
      · exact hd
  | _ => exact hd

/-- The state with the handles of the kind a step may close forgotten. -/
def AS.forgetKind (A : AS) : Option Contracts.Held → AS
  | some k => { A with slots := A.slots.map (AV.forgetK k) }
  | none => A

theorem AS.forgetKind_get (A : AS) (k : Contracts.Held) (i : R) :
    (A.forgetKind (some k)).get i = (A.get i).forgetK k := by
  unfold AS.forgetKind AS.get
  dsimp only
  split
  · exact Slots.get_map (f := AV.forgetK k) rfl _ _
  · rfl

theorem AS.Holds.forgetKind {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) (kc : Option Contracts.Held) :
    (A.forgetKind kc).Holds vals Γ w := by
  cases kc with
  | none => exact h
  | some k =>
      exact ⟨h.1, fun i hi => by
        obtain ⟨v, hv, hd⟩ := h.2.1 i hi
        exact ⟨v, hv, by rw [AS.forgetKind_get]; exact AV.forgetK_desc hd⟩, h.2.2⟩

theorem AS.forgetKind_hand {A : AS} {kc : Option Contracts.Held} {i : R} {k : Contracts.Held} {b : Bool}
    (h : (A.forgetKind kc).get i = .hand k b) : kc ≠ some k := by
  intro e; subst e
  rw [AS.forgetKind_get] at h
  cases e' : A.get i <;> rw [e'] at h <;> simp only [AV.forgetK, reduceCtorEq] at h
  split at h
  · cases h
  · rename_i hne; cases h; exact hne rfl

-- ---------------------------------------------------------------------------
-- Operations
-- ---------------------------------------------------------------------------

/-- A known slot's value; any value for the rest. -/
def concV : AV → V
  | .known v => v
  | _ => .sc .i64 0

/-- A known slot holds the value it is known to. -/
theorem AS.Holds.known {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {r : R}
    (hk : (A.get r).isKnown = true) : Γ[r]? = some (concV (A.get r)) := by
  obtain ⟨u, hu, hd⟩ := h.get (AS.get_lt_known hk)
  rw [hu]
  generalize A.get r = x at hd hk
  cases x with
  | known v => simp only [AV.Desc] at hd; rw [hd]; rfl
  | _ => cases hk

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

/-- An operation reading its operands from slots `0`, `1`, `2` in the order
    it takes them: what it computes on an environment of just its operands. -/
def _root_.AlgorithmLib.HProg.Op.positional : Op → Op
  | .iconst t k => .iconst t k
  | .fconst t b => .fconst t b
  | .iadd _ _ => .iadd 0 1 | .isub _ _ => .isub 0 1 | .imul _ _ => .imul 0 1
  | .udiv _ _ => .udiv 0 1 | .ishl _ _ => .ishl 0 1 | .ushr _ _ => .ushr 0 1
  | .band _ _ => .band 0 1 | .bandNot _ _ => .bandNot 0 1 | .bor _ _ => .bor 0 1
  | .bxor _ _ => .bxor 0 1 | .icmp c _ _ => .icmp c 0 1 | .fadd _ _ => .fadd 0 1
  | .fsub _ _ => .fsub 0 1 | .fmul _ _ => .fmul 0 1 | .fmax _ _ => .fmax 0 1
  | .fmin _ _ => .fmin 0 1 | .fcmp c _ _ => .fcmp c 0 1 | .ibin k _ _ => .ibin k 0 1
  | .ishift k _ _ => .ishift k 0 1 | .fbin k _ _ => .fbin k 0 1
  | .ineg _ => .ineg 0 | .ctz _ => .ctz 0 | .popcnt _ => .popcnt 0
  | .ireduce32 _ => .ireduce32 0 | .uextend64 _ => .uextend64 0 | .sextend64 _ => .sextend64 0
  | .fneg _ => .fneg 0 | .fpromote _ => .fpromote 0 | .vhighBits _ => .vhighBits 0
  | .fcvtFromSint t _ => .fcvtFromSint t 0 | .fcvtToUint t _ => .fcvtToUint t 0
  | .splat t _ => .splat t 0 | .extractlane _ n => .extractlane 0 n | .bitcast t _ => .bitcast t 0
  | .iun k _ => .iun k 0 | .fun1 k _ => .fun1 k 0 | .fconv k t _ => .fconv k t 0
  | .iext k t _ => .iext k t 0 | .load op _ => .load op 0
  | .select _ _ _ => .select 0 1 2 | .bitselect _ _ _ => .bitselect 0 1 2 | .fma _ _ _ => .fma 0 1 2

/-- An operation computes the same on its operands alone, read positionally. -/
theorem evalOp_positional (m : Mem) (Γ Γ' : Env) (o : Op)
    (h : ∀ i (hi : i < o.regs.length), Γ'[i]? = Γ[o.regs[i]]?) : evalOp m Γ' o.positional = evalOp m Γ o := by
  cases o with
  | iconst | fconst => rfl
  | iadd a b | isub a b | imul a b | udiv a b | ishl a b | ushr a b | band a b | bandNot a b
  | bor a b | bxor a b | icmp _ a b | fadd a b | fsub a b | fmul a b | fmax a b | fmin a b
  | fcmp _ a b | ibin _ a b | ishift _ a b | fbin _ a b =>
    have h0 := h 0 (by simp [Op.regs]); have h1 := h 1 (by simp [Op.regs])
    simp only [Op.regs, List.getElem_cons_zero, List.getElem_cons_succ] at h0 h1
    simp only [Op.positional, evalOp, bin, shiftBin, Sem.get, h0, h1]
  | ineg a | ctz a | popcnt a | ireduce32 a | uextend64 a | sextend64 a | fneg a | fpromote a
  | vhighBits a | fcvtFromSint _ a | fcvtToUint _ a | splat _ a | extractlane a _ | bitcast _ a
  | iun _ a | fun1 _ a | fconv _ _ a | iext _ _ a | load _ a =>
    have h0 := h 0 (by simp [Op.regs])
    simp only [Op.regs, List.getElem_cons_zero] at h0
    simp only [Op.positional, evalOp, un, Sem.get, h0]
  | select c a b | bitselect c a b | fma c a b =>
    have h0 := h 0 (by simp [Op.regs]); have h1 := h 1 (by simp [Op.regs])
    have h2 := h 2 (by simp [Op.regs])
    simp only [Op.regs, List.getElem_cons_zero, List.getElem_cons_succ] at h0 h1 h2
    simp only [Op.positional, evalOp, Sem.get, h0, h1, h2]

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

/-- Whether the typestate gives region `r` at least `n` bytes and memory is not
    frozen: what an access below `n` into `r` needs. -/
def roomIn (S : TState) (r : Region) (n : Nat) : Bool :=
  decide (r ≠ .pinned) && S.contains (.part .frozen false) &&
    S.any fun | .room r' k => decide (r' = r) && decide (n ≤ k) | _ => false

theorem roomIn_spec {S : TState} {w : World} {r : Region} {n : Nat} (hS : S.holds w)
    (h : roomIn S r n = true) : r ≠ .pinned ∧ w.mem.frozen = false ∧ n ≤ w.mem.sizes r := by
  unfold roomIn at h
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.any_eq_true] at h
  obtain ⟨⟨hp, hz⟩, x, hx, hr⟩ := h
  refine ⟨hp, holds_of_mem hS hz, ?_⟩
  cases x with
  | room r' k =>
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hr
      obtain ⟨rfl, hk⟩ := hr
      exact Nat.le_trans hk (hS _ hx)
  | _ => cases hr

theorem ptr_decode {r : Region} {off hi : Nat} (h1 : off ≤ hi) (h2 : hi < span) :
    decodeAddr (regionBase r + UInt64.ofNat off) = some (r, off) :=
  Static.decodeAddr_base r off (by rw [Static.regionSpan_toNat]; unfold span at h2; omega)

/-- An address in a region's span, or just past it, as a number. -/
theorem addr_toNat (r : Region) {off : Nat} (h : off ≤ span) :
    (regionBase r + UInt64.ofNat off).toNat = (regionBase r).toNat + off := by
  have hb := Static.regionBase_toNat r
  rw [Static.regionSpan_toNat] at hb
  unfold span at h
  rw [UInt64.toNat_add, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (a := off) (by omega), Nat.mod_eq_of_lt (by omega)]

/-- The bytes `[x, x + k)` lie in one region: by the room the typestate gives
    it, or by the bound an address carries. -/
def memN (S : TState) (p : AV) (k : Nat) : Bool :=
  match p with
  | .known (.sc _ x) => roomAt S x k
  | .ptr r n => decide (n.hi + k ≤ span) && decide (n.hi < span) &&
      (roomIn S r (n.hi + k) ||
        (match n.fit with
         | some (r', 1, d) => decide (r' = r) && decide (k ≤ d) && decide (r ≠ .pinned) &&
             S.contains (.part .frozen false)
         | _ => false))
  | _ => false

/-- What `memN` gives: the region, the offset, and room for every access below
    `k` bytes. -/
theorem memN_spec {Γ : Env} {w : World} {S : TState} {p : AV} {x : UInt64} {k : Nat} (hS : S.holds w)
    (hd : ∃ t, p.Desc Γ w (.sc t x)) (h : memN S p k = true) :
    ∃ r off, decodeAddr x = some (r, off) ∧ r ≠ .pinned ∧ off + k ≤ w.mem.sizes r ∧
      w.mem.frozen = false ∧ off + k ≤ 2 ^ 36 := by
  obtain ⟨t, hd⟩ := hd
  unfold memN at h
  split at h
  · rename_i t' x'
    simp only [AV.Desc, V.sc.injEq] at hd
    obtain ⟨-, rfl⟩ := hd
    exact roomAt_spec hS h
  · rename_i r n
    simp only [Bool.and_eq_true, decide_eq_true_eq, Bool.or_eq_true] at h
    obtain ⟨⟨hs, h3⟩, hr⟩ := h
    obtain ⟨off, e, ⟨h1, h2, -, hf⟩, -⟩ := hd
    simp only [V.sc.injEq] at e
    obtain ⟨-, rfl⟩ := e
    rcases hr with hr | hr
    · obtain ⟨hp, hz, hsz⟩ := roomIn_spec hS hr
      exact ⟨r, off, ptr_decode h2 h3, hp, by omega, hz, by unfold span at hs; omega⟩
    · split at hr
      · rename_i r' d hfit
        simp only [Bool.and_eq_true, decide_eq_true_eq] at hr
        obtain ⟨⟨⟨rfl, hkd⟩, hp⟩, hz⟩ := hr
        rw [hfit] at hf
        obtain ⟨-, hle⟩ := hf
        exact ⟨r', off, ptr_decode h2 h3, hp, by omega, holds_of_mem hS hz, by unfold span at hs; omega⟩
      · cases hr
  · cases h

theorem memN_fits {Γ : Env} {w : World} {S : TState} {p : AV} {x : UInt64} {k : Nat} (hS : S.holds w)
    (hd : ∃ t, p.Desc Γ w (.sc t x)) (h : memN S p k = true) : Fits w.mem x k :=
  memN_spec hS hd h

/-- An address `memN` accepts is a scalar. -/
theorem memN_sc {Γ : Env} {w : World} {S : TState} {p : AV} {u : V} {k : Nat} (hd : p.Desc Γ w u)
    (h : memN S p k = true) : ∃ t x, u = .sc t x := by
  unfold memN at h
  split at h
  · simp only [AV.Desc] at hd; exact ⟨_, _, hd⟩
  · obtain ⟨off, rfl, -⟩ := hd; exact ⟨_, _, rfl⟩
  · cases h

/-- How many bytes a load reads. -/
def loadBytes (op : LoadOp) : Nat :=
  match op.kind with
  | .uload8 | .sload8 => 1
  | .uload16 | .sload16 => 2
  | .uload32 | .sload32 => 4
  | .plain => match op.ty.lanes with
    | some (l, n) => n * tyBytes l
    | none => tyBytes op.ty

/-- A load whose bytes fit answers a value of its type: a scalar, or a vector
    read plainly. -/
theorem evalOp_load_fits {m : Mem} {Γ : Env} {op : LoadOp} {a : R} {t : ClifTy} {x : UInt64}
    (hok : (op.kind == .plain || op.ty.lanes.isNone) = true) (ha : Γ[a]? = some (.sc t x))
    (hf : Fits m x (loadBytes op)) : ∃ v, evalOp m Γ (.load op a) = some v ∧ TyV op.ty v := by
  cases hl : op.ty.lanes with
  | some p =>
      obtain ⟨l, n⟩ := p
      have hk : op.kind = .plain := by
        rw [hl] at hok; cases e : op.kind <;> rw [e] at hok <;> first | rfl | cases hok
      have hv := evalOp_vload (m := m) hk hl ha
      have hf' : Fits m x (n * tyBytes l) := by unfold loadBytes at hf; rw [hk, hl] at hf; exact hf
      cases e : evalOp m Γ (.load op a) with
      | none => exact absurd e (hv.2 hf')
      | some v => exact ⟨v, rfl, hv.1 v e⟩
  | none =>
      obtain ⟨b, hb⟩ := hf.load
      unfold loadBytes at hb
      cases hk : op.kind <;> simp only [hk, hl] at hb <;>
        exact ⟨_, by simp [evalOp, Sem.get, ha, hk, hl, hb] <;> rfl,
          by first | exact TyV.sc hl | (unfold norm; exact TyV.sc hl) | (unfold ofInt; exact TyV.sc hl)⟩

/-- A read of a cell the typestate names: the whole of it as an `i64`, or
    either half as an `i32`. -/
def cellLoad (S : TState) (op : LoadOp) (addr : AV) : Option AV :=
  match op.kind, op.ty, addr with
  | .plain, .i64, .known (.sc _ x) => (cellAt S x).map fun v => .known (.sc .i64 v)
  | .plain, .i32, .known (.sc _ x) =>
      match cellAt S x with
      | some v => some (.known (.sc .i32 (v &&& 0xFFFFFFFF)))
      | none =>
          match decodeAddr (x - 4), cellAt S (x - 4) with
          | some (_, off), some v => if off + 4 < 2 ^ 36 then some (.known (.sc .i32 (v >>> 32))) else none
          | _, _ => none
  | _, _, _ => none


theorem widthMask_int {t : ClifTy} (h : t.isInt = true) : (widthMask t).toNat = 2 ^ t.width - 1 := by
  cases t <;> first | rfl | cases h

theorem norm_bound {t : ClifTy} (ht : t.isInt = true) (z : UInt64) :
    ∃ y, norm t z = .sc t y ∧ y.toNat ≤ z.toNat ∧ y.toNat ≤ 2 ^ t.width - 1 := by
  refine ⟨z &&& widthMask t, rfl, ?_, ?_⟩
  · rw [UInt64.toNat_and]; exact Nat.and_le_left
  · rw [UInt64.toNat_and, ← widthMask_int ht]; exact Nat.and_le_right

/-- Bytes put together little-endian, eight at most, make a number below
    `2 ^ (8 * n)`. -/
theorem foldr_bytes_lt (g : Nat → UInt64) (hg : ∀ i, (g i).toNat < 256) :
    ∀ (l : List Nat), l.length ≤ 8 →
      (l.foldr (fun i (acc : UInt64) => (acc <<< 8) ||| g i) (0 : UInt64)).toNat < 2 ^ (8 * l.length)
  | [], _ => by simp
  | i :: l, h => by
      have hl : l.length ≤ 7 := by simp at h; omega
      have ih := foldr_bytes_lt g hg l (by omega)
      simp only [List.foldr_cons, List.length_cons]
      generalize l.foldr (fun i (acc : UInt64) => (acc <<< 8) ||| g i) (0 : UInt64) = a at ih
      have ha : a.toNat < 2 ^ 56 := Nat.lt_of_lt_of_le ih (Nat.pow_le_pow_right (by decide) (by omega))
      rw [UInt64.toNat_or, UInt64.toNat_shiftLeft]
      have h8 : (8 : UInt64).toNat % 64 = 8 := by decide
      rw [h8, Nat.shiftLeft_eq, Nat.mod_eq_of_lt (by omega), ← Nat.shiftLeft_eq,
        ← Nat.shiftLeft_add_eq_or_of_lt (hg i), Nat.shiftLeft_eq]
      have := hg i
      have e : 2 ^ (8 * (l.length + 1)) = 2 ^ (8 * l.length) * 2 ^ 8 := by
        rw [Nat.mul_succ, Nat.pow_add]
      rw [e]
      have : (a.toNat + 1) * 2 ^ 8 ≤ 2 ^ (8 * l.length) * 2 ^ 8 := Nat.mul_le_mul_right _ ih
      omega

/-- A load of `n` bytes, eight at most, reads a number below `2 ^ (8 * n)`. -/
theorem Mem.load_lt {m : Mem} {a b : UInt64} {n : Nat} (h : m.load a n = some b) (hn : n ≤ 8) :
    b.toNat < 2 ^ (8 * n) := by
  unfold Mem.load at h
  rcases hd : decodeAddr a with _ | ⟨r, off⟩ <;> simp only [hd, bind, Option.bind, reduceCtorEq] at h
  split at h
  · cases h
  · cases h
    have := foldr_bytes_lt (fun i => ((m.region r).get! (off + i)).toUInt64)
      (fun i => by simp only [UInt8.toNat_toUInt64]; exact UInt8.toNat_lt _) (List.range n) (by simp; omega)
    simpa using this

/-- The largest number a load of `op` may leave, where it leaves an integer: a
    zero-extending load the bytes it reads, the rest their type's width. -/
def loadHi (op : LoadOp) : Nat :=
  match op.kind with
  | .uload8 => min 255 (2 ^ op.ty.width - 1)
  | .uload16 => min 65535 (2 ^ op.ty.width - 1)
  | .uload32 => min (2 ^ 32 - 1) (2 ^ op.ty.width - 1)
  | .plain => 2 ^ (8 * tyBytes op.ty) - 1
  | _ => 2 ^ op.ty.width - 1

theorem int_lanes {t : ClifTy} (ht : t.isInt = true) : t.lanes = none := by
  cases t <;> first | rfl | cases ht

theorem int_bytes {t : ClifTy} (ht : t.isInt = true) : tyBytes t ≤ 8 := by
  cases t <;> first | decide | cases ht

/-- A load of an integer type leaves a number no larger than `loadHi`. -/
theorem evalOp_load_hi {m : Mem} {Γ : Env} {op : LoadOp} {a : R} {v : V} (ht : op.ty.isInt = true)
    (h : evalOp m Γ (.load op a) = some v) : ∃ x, v = .sc op.ty x ∧ x.toNat ≤ loadHi op := by
  have hl := int_lanes ht
  have hb := int_bytes ht
  have unext : ∀ (addr : UInt64) (n : Nat) (b : UInt64), n ≤ 8 → m.load addr n = some b →
      ∀ c, c = min (2 ^ (8 * n) - 1) (2 ^ op.ty.width - 1) → ∃ x, norm op.ty b = .sc op.ty x ∧ x.toNat ≤ c :=
    fun addr n b hn hld c hc => by
      obtain ⟨y, e, h1, h2⟩ := norm_bound ht b
      have := Mem.load_lt hld hn
      exact ⟨y, e, by rw [hc]; omega⟩
  have sext : ∀ (i : Int), ∃ x, ofInt op.ty i = .sc op.ty x ∧ x.toNat ≤ 2 ^ op.ty.width - 1 := fun i => by
    obtain ⟨y, e, -, h2⟩ := norm_bound ht (UInt64.ofNat (i.emod (1 <<< 64)).toNat)
    exact ⟨y, e, h2⟩
  simp only [evalOp, Sem.get] at h
  cases hg : Γ[a]? with
  | none => rw [hg] at h; simp at h
  | some u =>
      rw [hg] at h
      cases u with
      | vec _ _ => simp at h
      | sc t' addr =>
          simp only [Option.bind, bind] at h
          unfold loadHi
          cases hk : op.kind <;> rw [hk] at h <;> dsimp only at h ⊢
          · rw [hl] at h
            dsimp only at h
            cases hld : m.load addr (tyBytes op.ty) with
            | none => rw [hld] at h; cases h
            | some b =>
                rw [hld] at h; cases h
                have := Mem.load_lt hld hb
                exact ⟨b, rfl, by omega⟩
          all_goals first
            | (cases hld : m.load addr 1 with
                | none => rw [hld] at h; cases h
                | some b =>
                    rw [hld] at h; cases h
                    first
                      | exact unext addr 1 b (by decide) hld _ rfl
                      | exact sext _)
            | (cases hld : m.load addr 2 with
                | none => rw [hld] at h; cases h
                | some b =>
                    rw [hld] at h; cases h
                    first
                      | exact unext addr 2 b (by decide) hld _ rfl
                      | exact sext _)
            | (cases hld : m.load addr 4 with
                | none => rw [hld] at h; cases h
                | some b =>
                    rw [hld] at h; cases h
                    first
                      | exact unext addr 4 b (by decide) hld _ rfl
                      | exact sext _)

/-- What a load whose bytes are there leaves. -/
def loadAV (op : LoadOp) : AV := if op.ty.isInt then .int ⟨op.ty, 0, loadHi op, 1, none⟩ else .sc op.ty

/-- A load the checker reads whole: a plain one, or one of a scalar. -/
def loadKindOk (op : LoadOp) : Bool := op.kind == .plain || op.ty.lanes.isNone

/-- What a load of `op` reads from an address the checker knows of. -/
def absLoad (S : TState) (op : LoadOp) (addr : AV) : Option AV :=
  match cellLoad S op addr with
  | some av => some av
  | none => if loadKindOk op && memN S addr (loadBytes op) then some (loadAV op) else none

theorem cellLoad_sound {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {op : LoadOp} {a : R}
    {av : AV} (hl : cellLoad A.facts op (A.get a) = some av) :
    ∃ v, evalOp w.mem Γ (.load op a) = some v ∧ ∀ Γ', av.Desc Γ' w v := by
  unfold cellLoad at hl
  split at hl
  · rename_i t x hk hty hkn
    obtain ⟨v, hcell, rfl⟩ := Option.map_eq_some_iff.mp hl
    have hr := AS.get_lt_known (r := a) (by rw [hkn]; rfl)
    obtain ⟨u, hu, hd⟩ := h.get hr
    rw [hkn] at hd; simp only [AV.Desc] at hd; subst hd
    have hload : w.mem.load x (tyBytes op.ty) = some v := by
      rw [hty]; exact h.2.2 _ (cellAt_mem hcell)
    exact ⟨_, evalOp_load hk (by rw [hty]; rfl) hu hload, fun _ => by simp only [AV.Desc, hty]⟩
  · rename_i t x hk hty hkn
    have hr := AS.get_lt_known (r := a) (by rw [hkn]; rfl)
    obtain ⟨u, hu, hd⟩ := h.get hr
    rw [hkn] at hd; simp only [AV.Desc] at hd; subst hd
    split at hl
    · rename_i v hcell
      cases hl
      have hload : w.mem.load x (tyBytes op.ty) = some (v &&& 0xFFFFFFFF) := by
        rw [hty]; exact Contracts.TState.holds_cell_half h.2.2 (List.contains_iff_mem.mpr (cellAt_mem hcell))
      exact ⟨_, evalOp_load hk (by rw [hty]; rfl) hu hload, fun _ => by simp only [AV.Desc, hty]⟩
    · split at hl
      · rename_i r off v hdec hcell
        split at hl
        · rename_i hoff
          cases hl
          have hx : x - 4 + 4 = x := by rw [UInt64.sub_add_cancel]
          have hload : w.mem.load x (tyBytes op.ty) = some (v >>> 32) := by
            rw [hty, ← hx]
            exact Contracts.TState.holds_cell_hi h.2.2 (List.contains_iff_mem.mpr (cellAt_mem hcell)) hdec
              (by rw [Static.regionSpan_toNat]; exact hoff)
          exact ⟨_, evalOp_load hk (by rw [hty]; rfl) hu hload, fun _ => by simp only [AV.Desc, hty]⟩
        · cases hl
      · cases hl
  · cases hl

/-- **A load whose bytes are there** is defined, and leaves what `loadAV` says. -/
theorem load_of_fits {Γ : Env} {w : World} {op : LoadOp} {a : R} {t : ClifTy} {x : UInt64}
    (hok : loadKindOk op = true) (hu : Γ[a]? = some (.sc t x)) (hf : Fits w.mem x (loadBytes op)) :
    ∃ v, evalOp w.mem Γ (.load op a) = some v ∧ ∀ Γ', (loadAV op).Desc Γ' w v := by
  obtain ⟨v, hv, ht⟩ := evalOp_load_fits hok hu hf
  refine ⟨v, hv, fun _ => ?_⟩
  unfold loadAV
  split
  · rename_i hi
    obtain ⟨y, rfl, hy⟩ := evalOp_load_hi hi hv
    exact ⟨y, rfl, Nat.zero_le _, hy, Nat.one_dvd _, trivial⟩
  · exact ht

theorem absLoad_sound {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {op : LoadOp} {a : R}
    {av : AV} (hl : absLoad A.facts op (A.get a) = some av) :
    ∃ v, evalOp w.mem Γ (.load op a) = some v ∧ ∀ Γ', av.Desc Γ' w v := by
  unfold absLoad at hl
  split at hl
  · rename_i av' hc; cases hl; exact cellLoad_sound h hc
  · split at hl
    · rename_i hc
      cases hl
      simp only [Bool.and_eq_true] at hc
      obtain ⟨hok, hm⟩ := hc
      have hr : a < A.len := by
        rcases Nat.lt_or_ge a A.len with h' | h'
        · exact h'
        · rw [AS.get_out (Nat.not_lt_of_ge h')] at hm; simp [memN] at hm
      obtain ⟨u, hu, hd⟩ := h.get hr
      obtain ⟨t, x, rfl⟩ := memN_sc hd hm
      exact load_of_fits hok hu (memN_fits h.2.2 ⟨t, hd⟩ hm)
    · cases hl

-- ---------------------------------------------------------------------------
-- Integers
-- ---------------------------------------------------------------------------

/-- What the checker knows of a slot as an integer. -/
def AV.num : AV → Option Num
  | .known (.sc t x) => if t.isInt then some ⟨t, x.toNat, x.toNat, x.toNat, none⟩ else none
  | .int n => some n
  | .sc t => if t.isInt then some ⟨t, 0, 2 ^ 64 - 1, 1, none⟩ else none
  | .cmp .. => some ⟨.i8, 0, 1, 1, none⟩
  | _ => none

theorem AV.num_desc {Γ : Env} {w : World} {av : AV} {v : V} {n : Num}
    (hd : av.Desc Γ w v) (h : av.num = some n) : ∃ x, v = .sc n.t x ∧ n.Holds w x.toNat := by
  cases av with
  | known u =>
      simp only [Desc] at hd; rw [hd]
      cases u with
      | sc t' x =>
          simp only [num] at h
          split at h
          · cases h; exact ⟨x, rfl, Nat.le_refl _, Nat.le_refl _, Nat.dvd_refl _, trivial⟩
          · cases h
      | vec _ _ => cases h
  | int n' => simp only [num, Option.some.injEq] at h; subst h; exact hd
  | sc t' =>
      simp only [num] at h
      split at h
      · rename_i hi
        cases h
        have hl : ∀ s : ClifTy, s.isInt = true → s.lanes = none := by
          intro s hs; cases s <;> first | rfl | cases hs
        simp only [Desc, TyV, hl _ hi] at hd
        obtain ⟨x, rfl⟩ := hd
        exact ⟨x, rfl, Nat.zero_le _, by show x.toNat ≤ 2 ^ 64 - 1; have := x.toNat_lt; omega, Nat.one_dvd _, trivial⟩
      · cases h
  | cmp c a b =>
      simp only [num, Option.some.injEq] at h; subst h
      obtain ⟨_, _, _, -, -, -, rfl⟩ := hd
      unfold boolV
      split <;> exact ⟨_, rfl, by decide, by decide, Nat.one_dvd _, trivial⟩
  | _ => cases h

/-- A number known exactly. -/
def Num.exact (n : Num) : Option Nat := if n.lo = n.hi then some n.lo else none

theorem Num.exact_eq {n : Num} {w : World} {x k : Nat} (hx : n.Holds w x) (h : n.exact = some k) : x = k := by
  unfold Num.exact at h
  split at h
  · cases h; obtain ⟨h1, h2, -⟩ := hx; omega
  · cases h

-- Bounds

theorem Fit.le {w : World} {f : Option Bound} {x y : Nat} (h : Fit w f x) (hy : y ≤ x) : Fit w f y := by
  match f, h with
  | none, _ => trivial
  | some (r, s, d), ⟨hp, hle⟩ => exact ⟨hp, by have := Nat.mul_le_mul_right s hy; omega⟩

theorem Fit.or {w : World} {f g : Option Bound} {x : Nat} (hf : Fit w f x) (hg : Fit w g x) :
    Fit w (f.or g) x := by
  cases f with
  | none => exact hg
  | some _ => exact hf

/-- The bound of a value after at most `h` is added to it. -/
def fitAdd (f : Option Bound) (h : Nat) : Option Bound :=
  match f with
  | some (r, s, d) => if h * s ≤ d then some (r, s, d - h * s) else none
  | none => none

theorem fitAdd_sound {w : World} {f : Option Bound} {x y h : Nat} (hf : Fit w f x) (hy : y ≤ h) :
    Fit w (fitAdd f h) (x + y) := by
  match f, hf with
  | none, _ => trivial
  | some (r, s, d), ⟨hp, hle⟩ =>
      simp only [fitAdd]
      split
      · refine ⟨hp, ?_⟩
        have := Nat.mul_le_mul_right s hy
        rw [Nat.add_mul]; omega
      · trivial

/-- The bound of a value after at least `l` is taken from it. -/
def fitSub (f : Option Bound) (l : Nat) : Option Bound := f.map fun (r, s, d) => (r, s, d + l * s)

theorem fitSub_sound {w : World} {f : Option Bound} {x y l : Nat} (hf : Fit w f x) (hl : l ≤ y)
    (hy : y ≤ x) : Fit w (fitSub f l) (x - y) := by
  match f, hf with
  | none, _ => trivial
  | some (r, s, d), ⟨hp, hle⟩ =>
      refine ⟨hp, ?_⟩
      have h1 := Nat.mul_le_mul_right s hl
      have h2 := Nat.mul_le_mul_right s hy
      show (x - y) * s + (d + l * s) ≤ _
      rw [Nat.sub_mul]; omega

/-- The bound of a value no more than a `k`-th of one bounded. -/
def fitScale (f : Option Bound) (k : Nat) : Option Bound := f.map fun (r, s, d) => (r, s * k, d)

theorem fitScale_sound {w : World} {f : Option Bound} {x z k : Nat} (hf : Fit w f x) (hz : z * k ≤ x) :
    Fit w (fitScale f k) z := by
  match f, hf with
  | none, _ => trivial
  | some (r, s, d), ⟨hp, hle⟩ =>
      refine ⟨hp, ?_⟩
      show z * (s * k) + d ≤ _
      have := Nat.mul_le_mul_right s hz
      rw [Nat.mul_comm s k, ← Nat.mul_assoc]; omega

/-- The bound of a value `k` times one bounded, where `k` divides the scale. -/
def fitMul (f : Option Bound) (k : Nat) : Option Bound :=
  match f with
  | some (r, s, d) => if 0 < k ∧ s % k = 0 then some (r, s / k, d) else none
  | none => none

theorem fitMul_sound {w : World} {f : Option Bound} {x k : Nat} (hf : Fit w f x) :
    Fit w (fitMul f k) (x * k) := by
  match f, hf with
  | none, _ => trivial
  | some (r, s, d), ⟨hp, hle⟩ =>
      simp only [fitMul]
      split
      · rename_i hc
        refine ⟨hp, ?_⟩
        rw [Nat.mul_assoc, Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hc.2)]; exact hle
      · trivial

/-- A bound both of two hold. -/
def fitJoin : Option Bound → Option Bound → Option Bound
  | some (r, s, d), some (r', s', d') => if r = r' ∧ s = s' then some (r, s, min d d') else none
  | _, _ => none

theorem fitJoin_left {w : World} {f : Option Bound} (g : Option Bound) {x : Nat} (hf : Fit w f x) :
    Fit w (fitJoin f g) x := by
  match f, g, hf with
  | none, _, _ => trivial
  | some _, none, _ => trivial
  | some (r, s, d), some (r', s', d'), ⟨hp, hle⟩ =>
      simp only [fitJoin]
      split
      · exact ⟨hp, by have := Nat.min_le_left d d'; omega⟩
      · trivial

theorem fitJoin_right {w : World} (f : Option Bound) {g : Option Bound} {x : Nat} (hg : Fit w g x) :
    Fit w (fitJoin f g) x := by
  match f, g, hg with
  | none, _, _ => trivial
  | some _, none, _ => trivial
  | some (r, s, d), some (r', s', d'), ⟨hp, hle⟩ =>
      simp only [fitJoin]
      split
      · rename_i hc; obtain ⟨rfl, rfl⟩ := hc
        exact ⟨hp, by have := Nat.min_le_right d d'; omega⟩
      · trivial

-- Arithmetic on what is known of numbers

/-- A sum that cannot wrap. -/
def Num.add (a b : Num) : Option Num :=
  if a.t = b.t ∧ a.t.isInt ∧ a.hi + b.hi < 2 ^ a.t.width then
    some ⟨a.t, a.lo + b.lo, a.hi + b.hi, Nat.gcd a.al b.al, (fitAdd a.fit b.hi).or (fitAdd b.fit a.hi)⟩
  else none

theorem Num.add_sound {a b n : Num} {w : World} {x y : Nat} (hx : a.Holds w x) (hy : b.Holds w y)
    (h : a.add b = some n) : n.t = a.t ∧ a.t = b.t ∧ a.t.isInt ∧ x + y < 2 ^ a.t.width ∧ n.Holds w (x + y) := by
  unfold Num.add at h
  split at h
  · rename_i hc
    obtain ⟨ht, hi, hw⟩ := hc
    cases h
    obtain ⟨hx1, hx2, hx3, hx4⟩ := hx
    obtain ⟨hy1, hy2, hy3, hy4⟩ := hy
    refine ⟨rfl, ht, hi, by (try dsimp only); omega, by (try dsimp only); omega, by (try dsimp only); omega,
      Nat.dvd_add (Nat.dvd_trans (Nat.gcd_dvd_left _ _) hx3) (Nat.dvd_trans (Nat.gcd_dvd_right _ _) hy3),
      Fit.or (fitAdd_sound hx4 hy2) (by rw [Nat.add_comm]; exact fitAdd_sound hy4 hx2)⟩
  · cases h

/-- A difference that cannot wrap. -/
def Num.sub (a b : Num) : Option Num :=
  if a.t = b.t ∧ a.t.isInt ∧ b.hi ≤ a.lo ∧ a.hi < 2 ^ a.t.width then
    some ⟨a.t, a.lo - b.hi, a.hi - b.lo, Nat.gcd a.al b.al, fitSub a.fit b.lo⟩
  else none

theorem Num.sub_sound {a b n : Num} {w : World} {x y : Nat} (hx : a.Holds w x) (hy : b.Holds w y)
    (h : a.sub b = some n) : n.t = a.t ∧ a.t = b.t ∧ a.t.isInt ∧ y ≤ x ∧ x < 2 ^ a.t.width ∧
      n.Holds w (x - y) := by
  unfold Num.sub at h
  split at h
  · rename_i hc
    obtain ⟨ht, hi, hle, hw⟩ := hc
    cases h
    obtain ⟨hx1, hx2, hx3, hx4⟩ := hx
    obtain ⟨hy1, hy2, hy3, hy4⟩ := hy
    refine ⟨rfl, ht, hi, by (try dsimp only); omega, by (try dsimp only); omega, by (try dsimp only); omega, by (try dsimp only); omega,
      Nat.dvd_sub (Nat.dvd_trans (Nat.gcd_dvd_left _ _) hx3) (Nat.dvd_trans (Nat.gcd_dvd_right _ _) hy3),
      fitSub_sound hx4 hy1 (by (try dsimp only); omega)⟩
  · cases h

/-- A product that cannot overflow. -/
def Num.mul (a b : Num) : Option Num :=
  if a.t = b.t ∧ a.t.isInt ∧ a.hi * b.hi < 2 ^ a.t.width then
    some ⟨a.t, a.lo * b.lo, a.hi * b.hi, a.al * b.al,
      (match b.exact with | some k => fitMul a.fit k | none => none).or
      (match a.exact with | some k => fitMul b.fit k | none => none)⟩
  else none

theorem Num.mul_sound {a b n : Num} {w : World} {x y : Nat} (hx : a.Holds w x) (hy : b.Holds w y)
    (h : a.mul b = some n) : n.t = a.t ∧ a.t = b.t ∧ a.t.isInt ∧ x * y < 2 ^ a.t.width ∧
      n.Holds w (x * y) := by
  unfold Num.mul at h
  split at h
  · rename_i hc
    obtain ⟨ht, hi, hw⟩ := hc
    cases h
    have hm := Nat.mul_le_mul hx.2.1 hy.2.1
    refine ⟨rfl, ht, hi, by (try dsimp only); omega, Nat.mul_le_mul hx.1 hy.1, hm, Nat.mul_dvd_mul hx.2.2.1 hy.2.2.1, ?_⟩
    apply Fit.or
    · split
      · rename_i k hk; rw [Num.exact_eq hy hk]; exact fitMul_sound hx.2.2.2
      · trivial
    · split
      · rename_i k hk; rw [Num.exact_eq hx hk, Nat.mul_comm]; exact fitMul_sound hy.2.2.2
      · trivial
  · cases h

/-- A left shift by `p` that cannot overflow. -/
def Num.shl (a : Num) (p : Nat) : Option Num :=
  if a.t.isInt ∧ p < a.t.width ∧ a.hi * 2 ^ p < 2 ^ a.t.width then
    some ⟨a.t, a.lo * 2 ^ p, a.hi * 2 ^ p, a.al * 2 ^ p, fitMul a.fit (2 ^ p)⟩
  else none

theorem Num.shl_sound {a n : Num} {w : World} {x p : Nat} (hx : a.Holds w x) (h : a.shl p = some n) :
    n.t = a.t ∧ a.t.isInt ∧ p < a.t.width ∧ x * 2 ^ p < 2 ^ a.t.width ∧ n.Holds w (x * 2 ^ p) := by
  unfold Num.shl at h
  split at h
  · rename_i hc
    obtain ⟨hi, hp, hw⟩ := hc
    cases h
    have hm := Nat.mul_le_mul_right (2 ^ p) hx.2.1
    exact ⟨rfl, hi, hp, by (try dsimp only); omega, Nat.mul_le_mul_right _ hx.1, hm, Nat.mul_dvd_mul_right hx.2.2.1 _,
      fitMul_sound hx.2.2.2⟩
  · cases h

/-- An unsigned right shift by `p` of the value's bits within its width. -/
def Num.shr (a : Num) (p : Nat) : Num :=
  if a.hi < 2 ^ a.t.width then
    ⟨a.t, a.lo / 2 ^ p, a.hi / 2 ^ p, if a.al % 2 ^ p = 0 then a.al / 2 ^ p else 1, fitScale a.fit (2 ^ p)⟩
  else ⟨a.t, 0, (2 ^ a.t.width - 1) / 2 ^ p, 1, fitScale a.fit (2 ^ p)⟩

theorem Num.shr_sound {a : Num} {w : World} {x p : Nat} (hx : a.Holds w x) :
    (a.shr p).t = a.t ∧ (a.shr p).Holds w (x % 2 ^ a.t.width / 2 ^ p) := by
  obtain ⟨hx1, hx2, hx3, hx4⟩ := hx
  have hfit : Fit w (fitScale a.fit (2 ^ p)) (x % 2 ^ a.t.width / 2 ^ p) :=
    fitScale_sound hx4 (Nat.le_trans (Nat.div_mul_le_self _ _) (Nat.mod_le _ _))
  unfold Num.shr
  split
  · rename_i hw
    have hm : x % 2 ^ a.t.width = x := Nat.mod_eq_of_lt (by omega)
    rw [hm] at hfit ⊢
    refine ⟨rfl, Nat.div_le_div_right hx1, Nat.div_le_div_right hx2, ?_, hfit⟩
    split
    · rename_i hd; exact Nat.div_dvd_div (Nat.dvd_of_mod_eq_zero hd) hx3
    · exact Nat.one_dvd _
  · refine ⟨rfl, Nat.zero_le _, Nat.div_le_div_right ?_, Nat.one_dvd _, hfit⟩
    have := Nat.mod_lt x (Nat.two_pow_pos a.t.width); omega

/-- A value no larger than either of two: what a bitwise and or an unsigned
    minimum leaves. -/
def Num.below (a b : Num) : Option Num :=
  if a.t = b.t ∧ a.t.isInt then some ⟨a.t, 0, min a.hi b.hi, 1, a.fit.or b.fit⟩ else none

theorem Num.below_sound {a b n : Num} {w : World} {x y z : Nat} (hx : a.Holds w x) (hy : b.Holds w y)
    (h : a.below b = some n) (hzx : z ≤ x) (hzy : z ≤ y) :
    n.t = a.t ∧ a.t = b.t ∧ a.t.isInt ∧ n.Holds w z := by
  unfold Num.below at h
  split at h
  · rename_i hc
    cases h
    exact ⟨rfl, hc.1, hc.2, Nat.zero_le _, Nat.le_min.mpr ⟨by have := hx.2.1; omega, by have := hy.2.1; omega⟩,
      Nat.one_dvd _, Fit.or (hx.2.2.2.le hzx) (hy.2.2.2.le hzy)⟩
  · cases h

/-- The largest power of two dividing `k`, no larger than `2 ^ f`. -/
def pow2Of : Nat → Nat → Nat
  | 0, _ => 1
  | f + 1, k => if k % 2 = 0 then 2 * pow2Of f (k / 2) else 1

theorem pow2Of_spec : ∀ (f k : Nat), ∃ j, pow2Of f k = 2 ^ j ∧ 2 ^ j ∣ k
  | 0, k => ⟨0, rfl, Nat.one_dvd _⟩
  | f + 1, k => by
      simp only [pow2Of]
      split
      · rename_i h
        obtain ⟨j, e, hd⟩ := pow2Of_spec f (k / 2)
        refine ⟨j + 1, by rw [e, Nat.pow_succ, Nat.mul_comm], ?_⟩
        rw [Nat.pow_succ]
        obtain ⟨c, hc⟩ := hd
        refine ⟨c, ?_⟩
        have h2 := Nat.div_add_mod k 2
        rw [Nat.mul_comm (2 ^ j) 2, Nat.mul_assoc, ← hc]
        omega
      · exact ⟨0, rfl, Nat.one_dvd _⟩

/-- What `band` leaves: no more than either operand, and a multiple of every
    power of two both divide either of. -/
def Num.band (a b : Num) : Option Num :=
  if a.t = b.t ∧ a.t.isInt then
    some ⟨a.t, 0, min a.hi b.hi, max (pow2Of 64 a.al) (pow2Of 64 b.al), a.fit.or b.fit⟩
  else none

/-- Bits kept from a multiple of a power of two make one. -/
theorem and_pow_dvd {x y : Nat} {j : Nat} (h : 2 ^ j ∣ x ∨ 2 ^ j ∣ y) : 2 ^ j ∣ (x &&& y) := by
  apply Nat.dvd_of_mod_eq_zero
  rw [Nat.and_mod_two_pow]
  rcases h with h | h
  · rw [Nat.mod_eq_zero_of_dvd h, Nat.zero_and]
  · rw [Nat.mod_eq_zero_of_dvd h, Nat.and_zero]

/-- A quotient by a divisor at least `b.lo`, at least one. -/
def Num.udiv (a b : Num) : Option Num :=
  if a.t = b.t ∧ a.t.isInt ∧ 1 ≤ b.lo then some ⟨a.t, 0, a.hi / b.lo, 1, fitScale a.fit b.lo⟩ else none

theorem Num.udiv_sound {a b n : Num} {w : World} {x y z : Nat} (hx : a.Holds w x) (hy : b.Holds w y)
    (h : a.udiv b = some n) (hz : z ≤ x / y) :
    n.t = a.t ∧ a.t = b.t ∧ a.t.isInt ∧ 1 ≤ y ∧ n.Holds w z := by
  unfold Num.udiv at h
  split at h
  · rename_i hc
    obtain ⟨ht, hi, h1⟩ := hc
    cases h
    have hy1 := hy.1
    have hq : x / y ≤ a.hi / b.lo :=
      Nat.le_trans (Nat.div_le_div_right hx.2.1) (Nat.div_le_div_left hy1 (by (try dsimp only); omega))
    refine ⟨rfl, ht, hi, by (try dsimp only); omega, Nat.zero_le _, by (try dsimp only); omega, Nat.one_dvd _, fitScale_sound hx.2.2.2 ?_⟩
    calc z * b.lo ≤ x / y * y := Nat.mul_le_mul hz hy1
      _ ≤ x := Nat.div_mul_le_self _ _
  · cases h

/-- The unsigned maximum. -/
def Num.umax (a b : Num) : Option Num :=
  if a.t = b.t ∧ a.t.isInt then some ⟨a.t, 0, min (max a.hi b.hi) (2 ^ a.t.width - 1), 1, none⟩ else none

/-- A remainder by a divisor known nonzero within its width. -/
def Num.urem (a b : Num) : Option Num :=
  if a.t = b.t ∧ a.t.isInt ∧ 1 ≤ b.lo ∧ b.hi < 2 ^ a.t.width then some ⟨a.t, 0, b.hi - 1, 1, a.fit⟩ else none

/-- A value widened to `i64` from its type's width. -/
def Num.uext (a : Num) : Option Num :=
  if a.t.isInt ∧ a.t.width < 64 then
    if a.hi < 2 ^ a.t.width then some ⟨.i64, a.lo, a.hi, a.al, a.fit⟩
    else some ⟨.i64, 0, min a.hi (2 ^ a.t.width - 1), 1, a.fit⟩
  else none

theorem Num.uext_sound {a n : Num} {w : World} {x : Nat} (hx : a.Holds w x) (h : a.uext = some n) :
    n.t = .i64 ∧ a.t.isInt ∧ a.t.width < 64 ∧ n.Holds w (x % 2 ^ a.t.width) := by
  unfold Num.uext at h
  split at h
  · rename_i hc
    have hlt := Nat.mod_lt x (Nat.two_pow_pos a.t.width)
    have hle := Nat.mod_le x (2 ^ a.t.width)
    split at h
    · rename_i hw; cases h
      rw [Nat.mod_eq_of_lt (by have := hx.2.1; omega)]
      exact ⟨rfl, hc.1, hc.2, hx⟩
    · cases h
      exact ⟨rfl, hc.1, hc.2, Nat.zero_le _, Nat.le_min.mpr ⟨by have := hx.2.1; omega, by (try dsimp only); omega⟩,
        Nat.one_dvd _, hx.2.2.2.le hle⟩
  · cases h

/-- What two numbers both are. -/
def Num.join (a b : Num) : Option Num :=
  if a.t = b.t then some ⟨a.t, min a.lo b.lo, max a.hi b.hi, Nat.gcd a.al b.al, fitJoin a.fit b.fit⟩ else none

theorem Num.join_left {a b n : Num} {w : World} {x : Nat} (hx : a.Holds w x) (h : a.join b = some n) :
    n.t = a.t ∧ n.Holds w x := by
  unfold Num.join at h
  split at h
  · cases h
    exact ⟨rfl, by dsimp only; have := hx.1; have := Nat.min_le_left a.lo b.lo; omega,
      by dsimp only; have := hx.2.1; have := Nat.le_max_left a.hi b.hi; omega,
      Nat.dvd_trans (Nat.gcd_dvd_left _ _) hx.2.2.1, fitJoin_left _ hx.2.2.2⟩
  · cases h

theorem Num.join_right {a b n : Num} {w : World} {y : Nat} (hy : b.Holds w y) (h : a.join b = some n) :
    n.t = b.t ∧ n.Holds w y := by
  unfold Num.join at h
  split at h
  · rename_i ht
    cases h
    exact ⟨ht, by dsimp only; have := hy.1; have := Nat.min_le_right a.lo b.lo; omega,
      by dsimp only; have := hy.2.1; have := Nat.le_max_right a.hi b.hi; omega,
      Nat.dvd_trans (Nat.gcd_dvd_right _ _) hy.2.2.1, fitJoin_right _ hy.2.2.2⟩
  · cases h

-- ---------------------------------------------------------------------------
-- Addresses
-- ---------------------------------------------------------------------------

/-- A known address read as the region and offset it names. -/
def AV.asPtr : AV → AV
  | .known (.sc .i64 x) =>
      match decodeAddr x with
      | some (r, off) => if off < span then .ptr r ⟨.i64, off, off, off, none⟩ else .known (.sc .i64 x)
      | none => .known (.sc .i64 x)
  | a => a

theorem AV.asPtr_desc {Γ : Env} {w : World} {a : AV} {v : V} (hd : a.Desc Γ w v) : a.asPtr.Desc Γ w v := by
  unfold AV.asPtr
  split
  · rename_i x
    simp only [AV.Desc] at hd
    subst hd
    split
    · rename_i r off hdec
      split
      · rename_i hs
        refine ⟨off, ?_, ⟨Nat.le_refl _, Nat.le_refl _, Nat.dvd_refl _, trivial⟩, Nat.le_of_lt hs⟩
        rw [(Static.decodeAddr_some hdec).1]
      · rfl
    · rfl
  · exact hd

/-- An address moved by a number of type `i64`, where it stays in its span. -/
def addPtr (r : Region) (n b : Num) : Option AV :=
  match n.add b with
  | some m => if n.t = .i64 ∧ m.hi ≤ span then some (.ptr r m) else none
  | none => none

/-- What `iadd` leaves: an address moved by a number, or the sum of two
    numbers that cannot wrap. -/
def addA (a b : AV) : Option AV :=
  match a.asPtr with
  | .ptr r n => b.num.bind (addPtr r n)
  | _ => match b.asPtr with
    | .ptr r n => a.num.bind (addPtr r n)
    | _ => do let n ← (← a.num).add (← b.num); pure (.int n)

/-- What a comparison leaves: the flag of the two slots, where both are
    integers of one type. -/
def cmpA (A : AS) (c : ICmpCond) (a b : R) : Option AV :=
  match (A.get a).scTy, (A.get b).scTy with
  | some t, some t' => if t = t' ∧ t.isInt then some (.cmp c a b) else none
  | _, _ => none

theorem argsTy_of {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) :
    ∀ {rs : List R} {ts : List ClifTy}, rs.mapM (fun r => (A.get r).scTy) = some ts → ArgsTy Γ rs ts
  | [], ts, e => by simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at e; subst e; trivial
  | r :: rs, ts, e => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at e
      obtain ⟨t, ht, ts', hts, rfl⟩ := e
      obtain ⟨x, hx, -⟩ := h.getSc ht
      exact ⟨⟨x, hx⟩, argsTy_of h hts⟩

theorem argsV_of {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) :
    ∀ {rs : List R} {ts : List ClifTy}, rs.mapM (fun r => (A.get r).ty) = some ts → ArgsV Γ rs ts
  | [], ts, e => by simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at e; subst e; trivial
  | r :: rs, ts, e => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at e
      obtain ⟨t, ht, ts', hts, rfl⟩ := e
      have hr : r < A.len := by
        rcases Nat.lt_or_ge r A.len with h' | h'
        · exact h'
        · rw [AS.get_out (Nat.not_lt_of_ge h')] at ht; cases ht
      obtain ⟨v, hv, hd⟩ := h.get hr
      exact ⟨tyAt_of (AV.ty_desc hd ht) hv, argsV_of h hts⟩

-- ---------------------------------------------------------------------------
-- The transfer functions are sound
-- ---------------------------------------------------------------------------

theorem clifTy_beq_self (t : ClifTy) : (t == t) = true := by cases t <;> rfl

theorem width_int {t : ClifTy} (h : t.isInt = true) : t.width ≤ 64 := by
  cases t <;> first | (show _ ≤ 64; decide) | cases h

theorem pow_width_le {t : ClifTy} (h : t.isInt = true) : 2 ^ t.width ≤ 2 ^ 64 :=
  Nat.pow_le_pow_right (by decide) (width_int h)

/-- An integer below its type's width is its own normal form. -/
theorem norm_small {t : ClifTy} (h : t.isInt = true) {z : UInt64} (hz : z.toNat < 2 ^ t.width) :
    norm t z = .sc t z := by
  unfold norm
  congr 1
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_and, widthMask_int h, Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt hz]

theorem norm_i64 (z : UInt64) : norm .i64 z = .sc .i64 z :=
  norm_small (t := .i64) rfl z.toNat_lt

/-- The normal form of a value is no larger than it. -/
theorem mask_toNat {t : ClifTy} (ht : t.isInt = true) (x : UInt64) :
    (x &&& widthMask t).toNat = x.toNat % 2 ^ t.width := by
  rw [UInt64.toNat_and, widthMask_int ht, Nat.and_two_pow_sub_one_eq_mod]

/-- An address moved by an amount that keeps it in its span. -/
theorem addr_add (r : Region) (off : Nat) (y : UInt64) :
    regionBase r + UInt64.ofNat off + y = regionBase r + UInt64.ofNat (off + y.toNat) := by
  rw [UInt64.ofNat_add, UInt64.ofNat_toNat, UInt64.add_assoc]

theorem bin_eval {Γ : Env} {a b : R} {t : ClifTy} {x y : UInt64} (ha : Γ[a]? = some (.sc t x))
    (hb : Γ[b]? = some (.sc t y)) (hi : t.isInt = true) (f : ClifTy → UInt64 → UInt64 → Option V) :
    bin Γ a b f = f t x y := by
  simp [bin, Sem.get, ha, hb, clifTy_beq_self, hi]

/-- Two operands the checker knows as numbers. -/
theorem num2 {Γ : Env} {w : World} {ga gb : AV} {a b : R} {u u' : V} {na nb : Num}
    (ha : Γ[a]? = some u) (hb : Γ[b]? = some u') (hda : ga.Desc Γ w u) (hdb : gb.Desc Γ w u')
    (hna : ga.num = some na) (hnb : gb.num = some nb) :
    ∃ x y, Γ[a]? = some (.sc na.t x) ∧ Γ[b]? = some (.sc nb.t y) ∧ na.Holds w x.toNat ∧ nb.Holds w y.toNat := by
  obtain ⟨x, rfl, hx⟩ := AV.num_desc hda hna
  obtain ⟨y, rfl, hy⟩ := AV.num_desc hdb hnb
  exact ⟨x, y, ha, hb, hx, hy⟩

theorem addPtr_sound {r : Region} {n b : Num} {av : AV} {u : V} {y : UInt64}
    {Γ : Env} {w : World} (hda : (AV.ptr r n).Desc Γ w u) (hy : b.Holds w y.toNat)
    (h : addPtr r n b = some av) :
    b.t = .i64 ∧ ∃ off, u = .sc .i64 (regionBase r + UInt64.ofNat off) ∧
      ∀ Γ' : Env, av.Desc Γ' w (.sc .i64 (regionBase r + UInt64.ofNat (off + y.toNat))) := by
  unfold addPtr at h
  split at h
  · rename_i m hm
    split at h
    · rename_i hc
      obtain ⟨ht, hs⟩ := hc
      cases h
      obtain ⟨off, rfl, hx, -⟩ := hda
      obtain ⟨hmt, hnb, -, -, hh⟩ := Num.add_sound hx hy hm
      exact ⟨by rw [← hnb, ht], off, rfl, fun _ => ⟨off + y.toNat, rfl, hh, hs⟩⟩
    · cases h
  · cases h

theorem addA_sound {Γ : Env} {w : World} {ga gb av : AV} {a b : R} {u u' : V}
    (ha : Γ[a]? = some u) (hb : Γ[b]? = some u') (hda : ga.Desc Γ w u) (hdb : gb.Desc Γ w u')
    (h : addA ga gb = some av) (m : Mem) :
    ∃ v, evalOp m Γ (.iadd a b) = some v ∧ ∀ Γ' : Env, av.Desc Γ' w v := by
  unfold addA at h
  split at h
  · rename_i r n heq
    have hda' := AV.asPtr_desc hda; rw [heq] at hda'
    obtain ⟨nb, hnb, hp⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨y, rfl, hy⟩ := AV.num_desc hdb hnb
    obtain ⟨ht, off, rfl, hd⟩ := addPtr_sound hda' hy hp
    rw [ht] at hb
    refine ⟨_, ?_, hd⟩
    simp only [evalOp]; rw [bin_eval ha hb rfl]; simp only [norm_i64, addr_add]
  · split at h
    · rename_i r n heq
      have hdb' := AV.asPtr_desc hdb; rw [heq] at hdb'
      obtain ⟨na, hna, hp⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨x, rfl, hx⟩ := AV.num_desc hda hna
      obtain ⟨ht, off, rfl, hd⟩ := addPtr_sound hdb' hx hp
      rw [ht] at ha
      refine ⟨_, ?_, hd⟩
      simp only [evalOp]; rw [bin_eval ha hb rfl]; simp only [norm_i64, UInt64.add_comm x, addr_add]
    · simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at h
      obtain ⟨na, hna, nb, hnb, n, hn, rfl⟩ := h
      obtain ⟨x, y, hx', hy', hx, hy⟩ := num2 ha hb hda hdb hna hnb
      obtain ⟨hnt, hab, hi, hw, hh⟩ := Num.add_sound hx hy hn
      rw [← hab] at hy'
      have hsum : (x + y).toNat = x.toNat + y.toNat := by
        rw [UInt64.toNat_add]; exact Nat.mod_eq_of_lt (by have := pow_width_le hi; omega)
      refine ⟨_, by simp only [evalOp]; rw [bin_eval hx' hy' hi], fun _ => ⟨x + y, ?_, by rw [hsum]; exact hh⟩⟩
      rw [hnt]; exact norm_small hi (by omega)

/-- A two-operand number transfer function, sound for an operation. -/
def numBin (f : Num → Num → Option Num) (ga gb : AV) : Option AV := do
  let n ← f (← ga.num) (← gb.num); pure (.int n)

theorem numBin_some {f : Num → Num → Option Num} {ga gb av : AV} (h : numBin f ga gb = some av) :
    ∃ na nb n, ga.num = some na ∧ gb.num = some nb ∧ f na nb = some n ∧ av = .int n := by
  simp only [numBin, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at h
  obtain ⟨na, hna, nb, hnb, n, hn, rfl⟩ := h
  exact ⟨na, nb, n, hna, hnb, hn, rfl⟩

theorem subA_sound {Γ : Env} {w : World} {ga gb av : AV} {a b : R} {u u' : V}
    (ha : Γ[a]? = some u) (hb : Γ[b]? = some u') (hda : ga.Desc Γ w u) (hdb : gb.Desc Γ w u')
    (h : numBin Num.sub ga gb = some av) (m : Mem) :
    ∃ v, evalOp m Γ (.isub a b) = some v ∧ ∀ Γ' : Env, av.Desc Γ' w v := by
  obtain ⟨na, nb, n, hna, hnb, hn, rfl⟩ := numBin_some h
  obtain ⟨x, y, hx', hy', hx, hy⟩ := num2 ha hb hda hdb hna hnb
  obtain ⟨hnt, hab, hi, hle, hw, hh⟩ := Num.sub_sound hx hy hn
  rw [← hab] at hy'
  have hyx : y ≤ x := UInt64.le_iff_toNat_le.mpr hle
  have hsub : (x - y).toNat = x.toNat - y.toNat := UInt64.toNat_sub_of_le x y hyx
  refine ⟨_, by simp only [evalOp]; rw [bin_eval hx' hy' hi], fun _ => ⟨x - y, ?_, by rw [hsub]; exact hh⟩⟩
  rw [hnt]; exact norm_small hi (by omega)

theorem mulA_sound {Γ : Env} {w : World} {ga gb av : AV} {a b : R} {u u' : V}
    (ha : Γ[a]? = some u) (hb : Γ[b]? = some u') (hda : ga.Desc Γ w u) (hdb : gb.Desc Γ w u')
    (h : numBin Num.mul ga gb = some av) (m : Mem) :
    ∃ v, evalOp m Γ (.imul a b) = some v ∧ ∀ Γ' : Env, av.Desc Γ' w v := by
  obtain ⟨na, nb, n, hna, hnb, hn, rfl⟩ := numBin_some h
  obtain ⟨x, y, hx', hy', hx, hy⟩ := num2 ha hb hda hdb hna hnb
  obtain ⟨hnt, hab, hi, hw, hh⟩ := Num.mul_sound hx hy hn
  rw [← hab] at hy'
  have hprod : (x * y).toNat = x.toNat * y.toNat := by
    rw [UInt64.toNat_mul]; exact Nat.mod_eq_of_lt (by have := pow_width_le hi; omega)
  refine ⟨_, by simp only [evalOp]; rw [bin_eval hx' hy' hi], fun _ => ⟨x * y, ?_, by rw [hprod]; exact hh⟩⟩
  rw [hnt]; exact norm_small hi (by omega)

/-- What a shift by a known amount leaves. -/
def shlA (ga gb : AV) : Option AV := do
  let a ← ga.num
  let b ← gb.num
  let k ← b.exact
  if b.t.isInt then pure (.int (← a.shl k)) else none

/-- The most a shift right by an amount in `b`'s range leaves of `a`: halved
    at least `b.lo` times where the amount is below the width. -/
def shrHi (a b : Num) : Nat :=
  if b.hi < a.t.width then min a.hi (2 ^ a.t.width - 1) / 2 ^ b.lo else min a.hi (2 ^ a.t.width - 1)

def ushrA (ga gb : AV) : Option AV := do
  let a ← ga.num
  let b ← gb.num
  if a.t.isInt ∧ b.t.isInt then
    match b.exact with
    | some k => pure (.int (a.shr (k % a.t.width)))
    | none => pure (.int ⟨a.t, 0, shrHi a b, 1, a.fit⟩)
  else none

theorem shlA_sound {Γ : Env} {w : World} {ga gb av : AV} {a b : R} {u u' : V}
    (ha : Γ[a]? = some u) (hb : Γ[b]? = some u') (hda : ga.Desc Γ w u) (hdb : gb.Desc Γ w u')
    (h : shlA ga gb = some av) (m : Mem) :
    ∃ v, evalOp m Γ (.ishl a b) = some v ∧ ∀ Γ' : Env, av.Desc Γ' w v := by
  simp only [shlA, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨na, hna, nb, hnb, k, hk, h⟩ := h
  split at h
  · rename_i hbi
    simp only [Option.pure_def, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.some.injEq] at h
    obtain ⟨n, hn, rfl⟩ := h
    obtain ⟨x, y, hx', hy', hx, hy⟩ := num2 ha hb hda hdb hna hnb
    have hyk := Num.exact_eq hy hk
    obtain ⟨hnt, hi, hkw, hw, hh⟩ := Num.shl_sound (p := k) hx hn
    have h64 := width_int hi
    have hpw := pow_width_le hi
    have hmod : y % UInt64.ofNat na.t.width = y := by
      apply UInt64.toNat_inj.mp
      rw [UInt64.toNat_mod, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (show na.t.width < 2 ^ 64 by omega)]
      exact Nat.mod_eq_of_lt (by omega)
    have hshl : (x <<< y).toNat = x.toNat * 2 ^ k := by
      rw [UInt64.toNat_shiftLeft, Nat.shiftLeft_eq, Nat.mod_eq_of_lt (by omega : y.toNat < 64), hyk,
        Nat.mod_eq_of_lt (by omega)]
    refine ⟨.sc na.t (x <<< y), ?_, fun _ => ⟨x <<< y, by rw [hnt], by rw [hshl]; exact hh⟩⟩
    have hv : norm na.t (x <<< (y % UInt64.ofNat na.t.width)) = .sc na.t (x <<< y) := by
      rw [hmod]; exact norm_small hi (by rw [hshl]; omega)
    simp [evalOp, shiftBin, Sem.get, hx', hy', hi, hbi, hv]
  · cases h

theorem ushrA_sound {Γ : Env} {w : World} {ga gb av : AV} {a b : R} {u u' : V}
    (ha : Γ[a]? = some u) (hb : Γ[b]? = some u') (hda : ga.Desc Γ w u) (hdb : gb.Desc Γ w u')
    (h : ushrA ga gb = some av) (m : Mem) :
    ∃ v, evalOp m Γ (.ushr a b) = some v ∧ ∀ Γ' : Env, av.Desc Γ' w v := by
  simp only [ushrA, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨na, hna, nb, hnb, h⟩ := h
  split at h
  · rename_i hc
    obtain ⟨hi, hbi⟩ := hc
    obtain ⟨x, y, hx', hy', hx, hy⟩ := num2 ha hb hda hdb hna hnb
    have h64 := width_int hi
    have hwpos : 0 < na.t.width := by cases e : na.t <;> rw [e] at hi <;> first | decide | cases hi
    have hmx : (x &&& widthMask na.t).toNat ≤ x.toNat := by rw [UInt64.toNat_and]; exact Nat.and_le_left
    have hmk : (x &&& widthMask na.t).toNat ≤ 2 ^ na.t.width - 1 := by
      rw [UInt64.toNat_and, ← widthMask_int hi]; exact Nat.and_le_right
    have hle : ((x &&& widthMask na.t) >>> (y % UInt64.ofNat na.t.width)).toNat ≤
        (x &&& widthMask na.t).toNat := by
      rw [UInt64.toNat_shiftRight]; exact Nat.shiftRight_le _ _
    have hlt : ((x &&& widthMask na.t) >>> (y % UInt64.ofNat na.t.width)).toNat < 2 ^ na.t.width := by
      have := Nat.two_pow_pos na.t.width; omega
    have hev : evalOp m Γ (.ushr a b) =
        some (.sc na.t ((x &&& widthMask na.t) >>> (y % UInt64.ofNat na.t.width))) := by
      simp [evalOp, shiftBin, Sem.get, hx', hy', hi, hbi, norm_small hi hlt]
    split at h
    · rename_i k hk
      simp only [Option.pure_def, Option.some.injEq] at h
      subst h
      have hyk := Num.exact_eq hy hk
      have hp : k % na.t.width < 64 := by have := Nat.mod_lt k hwpos; omega
      have hmod : (y % UInt64.ofNat na.t.width).toNat = k % na.t.width := by
        rw [UInt64.toNat_mod, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (show na.t.width < 2 ^ 64 by omega), hyk]
      have hval : ((x &&& widthMask na.t) >>> (y % UInt64.ofNat na.t.width)).toNat =
          x.toNat % 2 ^ na.t.width / 2 ^ (k % na.t.width) := by
        rw [UInt64.toNat_shiftRight, hmod, Nat.mod_eq_of_lt hp, Nat.shiftRight_eq_div_pow, mask_toNat hi]
      obtain ⟨hnt, hh⟩ := Num.shr_sound (p := k % na.t.width) hx
      exact ⟨_, hev, fun _ => ⟨_, by rw [hnt], by rw [hval]; exact hh⟩⟩
    · simp only [Option.pure_def, Option.some.injEq] at h
      subst h
      have hxh := hx.2.1
      have hxm : (x &&& widthMask na.t).toNat ≤ min na.hi (2 ^ na.t.width - 1) := Nat.le_min.mpr ⟨by omega, by omega⟩
      refine ⟨_, hev, fun _ => ⟨_, rfl, Nat.zero_le _, ?_, Nat.one_dvd _, hx.2.2.2.le (by omega)⟩⟩
      unfold shrHi
      split
      · rename_i hbw
        have hy1 := hy.1
        have hy2 := hy.2.1
        have hmod : (y % UInt64.ofNat na.t.width).toNat = y.toNat := by
          rw [UInt64.toNat_mod, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (show na.t.width < 2 ^ 64 by omega),
            Nat.mod_eq_of_lt (by omega)]
        rw [UInt64.toNat_shiftRight, hmod, Nat.mod_eq_of_lt (show y.toNat < 64 by omega), Nat.shiftRight_eq_div_pow]
        exact Nat.le_trans (Nat.div_le_div_left (Nat.pow_le_pow_right (by decide) hy1) (Nat.two_pow_pos _))
          (Nat.div_le_div_right hxm)
      · exact Nat.le_trans hle hxm
  · cases h

theorem bandA_sound {Γ : Env} {w : World} {ga gb av : AV} {a b : R} {u u' : V}
    (ha : Γ[a]? = some u) (hb : Γ[b]? = some u') (hda : ga.Desc Γ w u) (hdb : gb.Desc Γ w u')
    (h : numBin Num.band ga gb = some av) (m : Mem) :
    ∃ v, evalOp m Γ (.band a b) = some v ∧ ∀ Γ' : Env, av.Desc Γ' w v := by
  obtain ⟨na, nb, n, hna, hnb, hn, rfl⟩ := numBin_some h
  obtain ⟨x, y, hx', hy', hx, hy⟩ := num2 ha hb hda hdb hna hnb
  unfold Num.band at hn
  split at hn
  · rename_i hc
    obtain ⟨hab, hi⟩ := hc
    cases hn
    rw [← hab] at hy'
    refine ⟨.sc na.t ((x &&& y) &&& widthMask na.t), ?_, fun _ => ⟨_, rfl, ?_⟩⟩
    · simp [evalOp, Sem.get, hx', hy', zipIntBits, clifTy_beq_self, hi, norm]
    · have hz1 : ((x &&& y) &&& widthMask na.t).toNat ≤ (x &&& y).toNat := by
        rw [UInt64.toNat_and]; exact Nat.and_le_left
      have hzx : (x &&& y).toNat ≤ x.toNat := by rw [UInt64.toNat_and]; exact Nat.and_le_left
      have hzy : (x &&& y).toNat ≤ y.toNat := by rw [UInt64.toNat_and]; exact Nat.and_le_right
      refine ⟨Nat.zero_le _, Nat.le_min.mpr ⟨by have := hx.2.1; omega, by have := hy.2.1; omega⟩, ?_,
        Fit.or (hx.2.2.2.le (by omega)) (hy.2.2.2.le (by omega))⟩
      have key : ∀ (c : Nat), c ∣ na.al ∨ c ∣ nb.al → (∃ j, c = 2 ^ j) →
          c ∣ ((x &&& y) &&& widthMask na.t).toNat := by
        rintro c hc ⟨j, rfl⟩
        rw [UInt64.toNat_and, UInt64.toNat_and]
        apply and_pow_dvd
        left
        apply and_pow_dvd
        rcases hc with hc | hc
        · exact .inl (Nat.dvd_trans hc hx.2.2.1)
        · exact .inr (Nat.dvd_trans hc hy.2.2.1)
      obtain ⟨j1, e1, d1⟩ := pow2Of_spec 64 na.al
      obtain ⟨j2, e2, d2⟩ := pow2Of_spec 64 nb.al
      rcases Nat.le_total (pow2Of 64 na.al) (pow2Of 64 nb.al) with hle | hle
      · rw [Nat.max_eq_right hle]; exact key _ (.inr (e2 ▸ d2)) ⟨j2, e2⟩
      · rw [Nat.max_eq_left hle]; exact key _ (.inl (e1 ▸ d1)) ⟨j1, e1⟩
  · cases hn

theorem and_high_mask {x k : Nat} (hx : x < 2 ^ 64) (hk : k ≤ 64) :
    x &&& (2 ^ 64 - 2 ^ k) = x / 2 ^ k * 2 ^ k := by
  apply Nat.eq_of_testBit_eq
  intro i
  have hM : 2 ^ 64 - 2 ^ k = (2 ^ (64 - k) - 1) * 2 ^ k := by
    rw [Nat.sub_mul, Nat.one_mul, ← Nat.pow_add, Nat.sub_add_cancel hk]
  rw [Nat.testBit_and, hM, Nat.testBit_mul_two_pow, Nat.testBit_mul_two_pow, Nat.testBit_two_pow_sub_one,
    Nat.testBit_div_two_pow]
  by_cases hik : k ≤ i
  · rw [Nat.sub_add_cancel hik]
    by_cases hi : i < 64
    · simp [hik, show i - k < 64 - k by omega]
    · have : x.testBit i = false :=
        Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le hx (Nat.pow_le_pow_right (by decide) (by omega)))
      simp [this]
  · simp [hik]

theorem not_low_mask {k : Nat} (hk : k ≤ 64) : ~~~(UInt64.ofNat (2 ^ k - 1)) = UInt64.ofNat (2 ^ 64 - 2 ^ k) := by
  apply UInt64.toNat_inj.mp
  have h1 : 1 ≤ 2 ^ k := Nat.one_le_two_pow
  have h2 : 2 ^ k ≤ 2 ^ 64 := Nat.pow_le_pow_right (by decide) hk
  rw [UInt64.toNat_not, UInt64.toNat_ofNat', UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega),
    Nat.mod_eq_of_lt (by omega)]
  simp only [UInt64.size]; omega

/-- `k` where `m`'s bits are the ones from bit `k` up: `x &&& m` rounds `x`
    down to a multiple of `2 ^ k`. -/
def highMaskK (m : UInt64) : Option Nat := (List.range 65).find? fun k => m.toNat == 2 ^ 64 - 2 ^ k

/-- `k` where `m`'s bits are the ones below bit `k`: `x &&& ~~~m` rounds `x`
    down to a multiple of `2 ^ k`. -/
def lowMaskK (m : UInt64) : Option Nat := (List.range 65).find? fun k => m.toNat == 2 ^ k - 1

theorem maskK_spec {m : UInt64} {k : Nat} {f : Nat → Nat}
    (h : (List.range 65).find? (fun k => m.toNat == f k) = some k) (hf : f k < 2 ^ 64) :
    k ≤ 64 ∧ m = UInt64.ofNat (f k) := by
  have hm := List.mem_range.mp (List.mem_of_find?_eq_some h)
  have he' : (m.toNat == f k) = true := List.find?_some (p := fun k => m.toNat == f k) h
  have he := beq_iff_eq.mp he'
  refine ⟨by omega, UInt64.toNat_inj.mp ?_⟩
  rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt hf, he]

theorem highMaskK_spec {m : UInt64} {k : Nat} (h : highMaskK m = some k) :
    k ≤ 64 ∧ m = UInt64.ofNat (2 ^ 64 - 2 ^ k) :=
  maskK_spec (f := fun k => 2 ^ 64 - 2 ^ k) h
    (by show 2 ^ 64 - 2 ^ k < 2 ^ 64; have : 1 ≤ 2 ^ k := Nat.one_le_two_pow; omega)

theorem lowMaskK_spec {m : UInt64} {k : Nat} (h : lowMaskK m = some k) :
    k ≤ 64 ∧ ~~~m = UInt64.ofNat (2 ^ 64 - 2 ^ k) := by
  have hk : k ≤ 64 := by
    have := List.mem_range.mp (List.mem_of_find?_eq_some h); omega
  have hlt : 2 ^ k - 1 < 2 ^ 64 := by
    have : 2 ^ k ≤ 2 ^ 64 := Nat.pow_le_pow_right (by decide) hk
    have : 1 ≤ 2 ^ k := Nat.one_le_two_pow
    omega
  obtain ⟨-, e⟩ := maskK_spec (f := fun k => 2 ^ k - 1) h hlt
  exact ⟨hk, by rw [e]; exact not_low_mask hk⟩

/-- A 64-bit number rounded down to a multiple of `2 ^ k`. -/
def Num.roundDown (a : Num) (k : Nat) : Num :=
  ⟨.i64, a.lo / 2 ^ k * 2 ^ k, a.hi / 2 ^ k * 2 ^ k, 2 ^ k, a.fit⟩

def roundA (ga : AV) (k : Nat) : Option AV :=
  match ga.num with
  | some n => if n.t = .i64 then some (.int (n.roundDown k)) else none
  | none => none

theorem roundA_sound {Γ : Env} {w : World} {ga av : AV} {u : V} {k : Nat} (hk : k ≤ 64)
    (hda : ga.Desc Γ w u) (h : roundA ga k = some av) :
    ∃ x, u = .sc .i64 x ∧ ∀ Γ' : Env, av.Desc Γ' w (.sc .i64 (x &&& UInt64.ofNat (2 ^ 64 - 2 ^ k))) := by
  unfold roundA at h
  split at h
  · rename_i n hn
    split at h
    · rename_i ht
      cases h
      obtain ⟨x, rfl, hx⟩ := AV.num_desc hda hn
      refine ⟨x, by rw [ht], fun _ => ⟨_, rfl, ?_⟩⟩
      have e : (x &&& UInt64.ofNat (2 ^ 64 - 2 ^ k)).toNat = x.toNat / 2 ^ k * 2 ^ k := by
        rw [UInt64.toNat_and, UInt64.toNat_ofNat',
          Nat.mod_eq_of_lt (by have : 1 ≤ 2 ^ k := Nat.one_le_two_pow; omega)]
        exact and_high_mask x.toNat_lt hk
      rw [e]
      exact ⟨Nat.mul_le_mul_right _ (Nat.div_le_div_right hx.1),
        Nat.mul_le_mul_right _ (Nat.div_le_div_right hx.2.1), Nat.dvd_mul_left _ _,
        hx.2.2.2.le (Nat.div_mul_le_self _ _)⟩
    · cases h
  · cases h

/-- `band` with a mask that rounds down, and otherwise by ranges. -/
def bandA (ga gb : AV) : Option AV :=
  (match gb with
   | .known (.sc .i64 m) => (highMaskK m).bind (roundA ga)
   | _ => none).or (numBin Num.band ga gb)

/-- `bandNot` with a mask of low bits: a rounding down. -/
def bandNotA (ga gb : AV) : Option AV :=
  match gb with
  | .known (.sc .i64 m) => (lowMaskK m).bind (roundA ga)
  | _ => none

theorem and_wide (z : UInt64) : z &&& widthMask .i64 = z := by
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_and, show (widthMask .i64).toNat = 2 ^ 64 - 1 from rfl,
    Nat.and_two_pow_sub_one_of_lt_two_pow z.toNat_lt]

theorem bandA_sound' {Γ : Env} {w : World} {ga gb av : AV} {a b : R} {u u' : V}
    (ha : Γ[a]? = some u) (hb : Γ[b]? = some u') (hda : ga.Desc Γ w u) (hdb : gb.Desc Γ w u')
    (h : bandA ga gb = some av) (m : Mem) :
    ∃ v, evalOp m Γ (.band a b) = some v ∧ ∀ Γ' : Env, av.Desc Γ' w v := by
  unfold bandA at h
  rcases Option.or_eq_some_iff.mp h with h | ⟨-, h⟩
  · split at h
    · rename_i mk
      obtain ⟨k, hk, hr⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨hk64, rfl⟩ := highMaskK_spec hk
      obtain ⟨x, rfl, hd⟩ := roundA_sound hk64 hda hr
      simp only [AV.Desc] at hdb
      subst hdb
      refine ⟨_, ?_, hd⟩
      simp [evalOp, Sem.get, ha, hb, zipIntBits, norm, and_wide, clifTy_beq_self, ClifTy.isInt]
    · cases h
  · exact bandA_sound ha hb hda hdb h m

theorem bandNotA_sound {Γ : Env} {w : World} {ga gb av : AV} {a b : R} {u u' : V}
    (ha : Γ[a]? = some u) (hb : Γ[b]? = some u') (hda : ga.Desc Γ w u) (hdb : gb.Desc Γ w u')
    (h : bandNotA ga gb = some av) (m : Mem) :
    ∃ v, evalOp m Γ (.bandNot a b) = some v ∧ ∀ Γ' : Env, av.Desc Γ' w v := by
  unfold bandNotA at h
  split at h
  · rename_i mk
    obtain ⟨k, hk, hr⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨hk64, e⟩ := lowMaskK_spec hk
    obtain ⟨x, rfl, hd⟩ := roundA_sound hk64 hda hr
    simp only [AV.Desc] at hdb
    subst hdb
    refine ⟨_, ?_, hd⟩
    simp [evalOp, Sem.get, ha, hb, zipIntBits, norm, and_wide, e, clifTy_beq_self, ClifTy.isInt]
  · cases h

theorem udivA_sound {Γ : Env} {w : World} {ga gb av : AV} {a b : R} {u u' : V}
    (ha : Γ[a]? = some u) (hb : Γ[b]? = some u') (hda : ga.Desc Γ w u) (hdb : gb.Desc Γ w u')
    (h : numBin Num.udiv ga gb = some av) (m : Mem) :
    ∃ v, evalOp m Γ (.udiv a b) = some v ∧ ∀ Γ' : Env, av.Desc Γ' w v := by
  obtain ⟨na, nb, n, hna, hnb, hn, rfl⟩ := numBin_some h
  obtain ⟨x, y, hx', hy', hx, hy⟩ := num2 ha hb hda hdb hna hnb
  have hi : na.t.isInt = true := by
    unfold Num.udiv at hn; split at hn
    · rename_i hc; exact hc.2.1
    · cases hn
  obtain ⟨z, hz, hz1, -⟩ := norm_bound hi (x / y)
  rw [UInt64.toNat_div] at hz1
  obtain ⟨hnt, hab, -, hy1, hh⟩ := Num.udiv_sound hx hy hn hz1
  rw [← hab] at hy'
  have hy0 : (y == 0) = false := by
    simp only [beq_eq_false_iff_ne, ne_eq]; intro e; subst e; simp at hy1
  refine ⟨.sc na.t z, ?_, fun _ => ⟨z, by rw [hnt], hh⟩⟩
  simp only [evalOp]; rw [bin_eval hx' hy' hi]; simp [hy0, hz]

/-- The unsigned minimum, maximum, and remainder by a divisor known nonzero. -/
def ibinA (k : IBin) (ga gb : AV) : Option AV :=
  match k with
  | .umin => numBin Num.below ga gb
  | .umax => numBin Num.umax ga gb
  | .urem => numBin Num.urem ga gb
  | _ => none

theorem ibinA_sound {Γ : Env} {w : World} {ga gb av : AV} {k : IBin} {a b : R} {u u' : V}
    (ha : Γ[a]? = some u) (hb : Γ[b]? = some u') (hda : ga.Desc Γ w u) (hdb : gb.Desc Γ w u')
    (h : ibinA k ga gb = some av) (m : Mem) :
    ∃ v, evalOp m Γ (.ibin k a b) = some v ∧ ∀ Γ' : Env, av.Desc Γ' w v := by
  cases k <;> simp only [ibinA, reduceCtorEq] at h
  case umin =>
    obtain ⟨na, nb, n, hna, hnb, hn, rfl⟩ := numBin_some h
    obtain ⟨x, y, hx', hy', hx, hy⟩ := num2 ha hb hda hdb hna hnb
    have hi : na.t.isInt = true := by
      unfold Num.below at hn; split at hn
      · rename_i hc; exact hc.2
      · cases hn
    have hab : na.t = nb.t := by
      unfold Num.below at hn; split at hn
      · rename_i hc; exact hc.1
      · cases hn
    rw [← hab] at hy'
    have hm : min (x &&& widthMask na.t).toNat (y &&& widthMask na.t).toNat < 2 ^ 64 := by
      have := (x &&& widthMask na.t).toNat_lt; omega
    obtain ⟨z, hz, hz1, -⟩ := norm_bound hi
      (UInt64.ofNat (min (x &&& widthMask na.t).toNat (y &&& widthMask na.t).toNat))
    rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt hm, mask_toNat hi, mask_toNat hi] at hz1
    have hzx : z.toNat ≤ x.toNat := by have := Nat.mod_le x.toNat (2 ^ na.t.width); omega
    have hzy : z.toNat ≤ y.toNat := by have := Nat.mod_le y.toNat (2 ^ na.t.width); omega
    obtain ⟨hnt, -, -, hh⟩ := Num.below_sound hx hy hn hzx hzy
    refine ⟨.sc na.t z, ?_, fun _ => ⟨z, by rw [hnt], hh⟩⟩
    simp only [evalOp]; rw [bin_eval hx' hy' hi]; simp only [ibinOp, hz]
  case umax =>
    obtain ⟨na, nb, n, hna, hnb, hn, rfl⟩ := numBin_some h
    obtain ⟨x, y, hx', hy', hx, hy⟩ := num2 ha hb hda hdb hna hnb
    unfold Num.umax at hn
    split at hn
    · rename_i hc
      obtain ⟨hab, hi⟩ := hc
      cases hn
      rw [← hab] at hy'
      have hm : max (x &&& widthMask na.t).toNat (y &&& widthMask na.t).toNat < 2 ^ 64 := by
        have := (x &&& widthMask na.t).toNat_lt; have := (y &&& widthMask na.t).toNat_lt; omega
      obtain ⟨z, hz, hz1, hz2⟩ := norm_bound hi
        (UInt64.ofNat (max (x &&& widthMask na.t).toNat (y &&& widthMask na.t).toNat))
      rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt hm, mask_toNat hi, mask_toNat hi] at hz1
      have := Nat.mod_le x.toNat (2 ^ na.t.width); have := Nat.mod_le y.toNat (2 ^ na.t.width)
      have := hx.2.1; have := hy.2.1
      refine ⟨.sc na.t z, ?_, fun _ => ⟨z, rfl, Nat.zero_le _, by (try dsimp only); omega, Nat.one_dvd _, trivial⟩⟩
      simp only [evalOp]; rw [bin_eval hx' hy' hi]; simp only [ibinOp, hz]
    · cases hn
  case urem =>
    obtain ⟨na, nb, n, hna, hnb, hn, rfl⟩ := numBin_some h
    obtain ⟨x, y, hx', hy', hx, hy⟩ := num2 ha hb hda hdb hna hnb
    unfold Num.urem at hn
    split at hn
    · rename_i hc
      obtain ⟨hab, hi, hl2, hh2⟩ := hc
      cases hn
      rw [← hab] at hy'
      have hyeq : (y &&& widthMask na.t).toNat = y.toNat := by
        rw [mask_toNat hi]; exact Nat.mod_eq_of_lt (by have := hy.2.1; omega)
      have hy1 := hy.1
      have hy0 : ((y &&& widthMask na.t).toNat == 0) = false := by
        simp only [beq_eq_false_iff_ne, ne_eq]; omega
      have hlt := Nat.mod_lt (x &&& widthMask na.t).toNat (show 0 < (y &&& widthMask na.t).toNat by (try dsimp only); omega)
      have hm : (x &&& widthMask na.t).toNat % (y &&& widthMask na.t).toNat < 2 ^ 64 := by
        have := (y &&& widthMask na.t).toNat_lt; omega
      obtain ⟨z, hz, hz1, -⟩ := norm_bound hi
        (UInt64.ofNat ((x &&& widthMask na.t).toNat % (y &&& widthMask na.t).toNat))
      rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt hm] at hz1
      have hzx : z.toNat ≤ x.toNat := by
        have := Nat.mod_le (x &&& widthMask na.t).toNat (y &&& widthMask na.t).toNat
        rw [mask_toNat hi] at this hz1; have := Nat.mod_le x.toNat (2 ^ na.t.width); omega
      have := hy.2.1
      refine ⟨.sc na.t z, ?_, fun _ => ⟨z, rfl, Nat.zero_le _, by (try dsimp only); omega, Nat.one_dvd _, hx.2.2.2.le hzx⟩⟩
      simp only [evalOp]; rw [bin_eval hx' hy' hi]; simp only [ibinOp, hy0, hz]; rfl
    · cases hn

/-- What `uextend64` leaves. -/
def uextA (ga : AV) : Option AV := do pure (.int (← (← ga.num).uext))

theorem uextA_sound {Γ : Env} {w : World} {ga av : AV} {a : R} {u : V}
    (ha : Γ[a]? = some u) (hda : ga.Desc Γ w u) (h : uextA ga = some av) (m : Mem) :
    ∃ v, evalOp m Γ (.uextend64 a) = some v ∧ ∀ Γ' : Env, av.Desc Γ' w v := by
  simp only [uextA, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at h
  obtain ⟨na, hna, n, hn, rfl⟩ := h
  obtain ⟨x, rfl, hx⟩ := AV.num_desc hda hna
  obtain ⟨hnt, hi, hw, hh⟩ := Num.uext_sound hx hn
  have hok : (na.t.isInt && decide (na.t.width < 64)) = true := by
    simp only [Bool.and_eq_true, decide_eq_true_eq]; exact ⟨hi, hw⟩
  refine ⟨_, ?_, fun _ => ⟨x &&& widthMask na.t, by rw [hnt], by rw [mask_toNat hi]; exact hh⟩⟩
  simp only [evalOp, un, Sem.get, ha, Option.bind_eq_bind, Option.bind_some, hok, if_true]

theorem cmpA_sound {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {c : ICmpCond} {a b : R} {av : AV}
    (hc : cmpA A c a b = some av) (m : Mem) :
    ∃ v, evalOp m Γ (.icmp c a b) = some v ∧ ∀ v0, av.Desc (Γ.push v0) w v := by
  unfold cmpA at hc
  split at hc
  · rename_i t t' hta htb
    split at hc
    · rename_i hcond
      obtain ⟨rfl, hi⟩ := hcond
      cases hc
      obtain ⟨x, hx, -⟩ := h.getSc hta
      obtain ⟨y, hy, -⟩ := h.getSc htb
      have hxs : a < Γ.size := by
        rcases Nat.lt_or_ge a Γ.size with h' | h'
        · exact h'
        · rw [Array.getElem?_eq_none h'] at hx; cases hx
      have hys : b < Γ.size := by
        rcases Nat.lt_or_ge b Γ.size with h' | h'
        · exact h'
        · rw [Array.getElem?_eq_none h'] at hy; cases hy
      refine ⟨boolV (cmpInt c t x y), ?_, fun v0 => ⟨t, x, y, ?_, ?_, hi, rfl⟩⟩
      · simp [evalOp, Sem.get, hx, hy, zipIntCmp, clifTy_beq_self, hi]
      · rw [Array.getElem?_push_lt hxs, ← Array.getElem?_eq_getElem]; exact hx
      · rw [Array.getElem?_push_lt hys, ← Array.getElem?_eq_getElem]; exact hy
    · cases hc
  · cases hc

/-- The special transfer function an operation has, if any. -/
def special (A : AS) : Op → Option AV
  | .iadd a b => addA (A.get a) (A.get b)
  | .isub a b => numBin Num.sub (A.get a) (A.get b)
  | .imul a b => numBin Num.mul (A.get a) (A.get b)
  | .ishl a b => shlA (A.get a) (A.get b)
  | .ushr a b => ushrA (A.get a) (A.get b)
  | .band a b => bandA (A.get a) (A.get b)
  | .bandNot a b => bandNotA (A.get a) (A.get b)
  | .udiv a b => numBin Num.udiv (A.get a) (A.get b)
  | .ibin k a b => ibinA k (A.get a) (A.get b)
  | .uextend64 a => uextA (A.get a)
  | .icmp c a b => cmpA A c a b
  | _ => none

/-- A transfer function over two ranges answers nothing where either operand is
    unknown. -/
theorem ranged_ne_any {f : AV → AV → Option AV} {ga gb av : AV}
    (h1 : ∀ g, f .any g = none) (h2 : ∀ g, f g .any = none) (h : f ga gb = some av) :
    ga ≠ .any ∧ gb ≠ .any := by
  constructor <;> intro e <;> subst e
  · rw [h1] at h; cases h
  · rw [h2] at h; cases h

/-- An operation on floating-point numbers: the kernel cannot compute these
    (`Float32` is opaque to it), so the checker does not fold them. -/
def floaty : Op → Bool
  | .fadd .. | .fsub .. | .fmul .. | .fmax .. | .fmin .. | .fcmp .. | .fbin .. | .fneg _
  | .fpromote _ | .fcvtFromSint .. | .fcvtToUint .. | .fun1 .. | .fconv .. | .fma .. => true
  | _ => false

/-- What an operation leaves in its slot, `none` where the checker cannot
    show it is defined: computed where every operand is known and the
    operation is not `floaty`, by a transfer function where one applies, and
    otherwise typed by `tyOp`. -/
def absOp (A : AS) (o : Op) : Option AV :=
  match o with
  | .load op a => absLoad A.facts op (A.get a)
  -- a float constant's bits, which a generator may have written as a `Float`
  -- the kernel cannot read, are never an address or a length
  | .fconst t _ => if t.lanes.isNone then some (.sc t) else none
  | o =>
      if !floaty o && o.regs.all (fun r => (A.get r).isKnown) then
        (evalOp default (o.regs.map fun r => concV (A.get r)).toArray o.positional).map .known
      else
        match special A o with
        | some av => some av
        | none =>
          match o.regs.mapM (fun r => (A.get r).scTy) >>= tyOp o with
          | some T => if T.lanes.isNone then some (.sc T) else none
          | none => (o.regs.mapM (fun r => (A.get r).ty) >>= tyVOp o).map .sc

theorem AS.Holds.get' {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {r : R} (hr : A.get r ≠ .any) :
    ∃ u, Γ[r]? = some u ∧ (A.get r).Desc Γ w u :=
  h.get (by
    rcases Nat.lt_or_ge r A.len with h' | h'
    · exact h'
    · exact absurd (AS.get_out (Nat.not_lt_of_ge h')) hr)

theorem special_bin {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {f : AV → AV → Option AV}
    {o : R → R → Op} {a b : R} {av : AV}
    (h1 : ∀ g, f .any g = none) (h2 : ∀ g, f g .any = none) (hs : f (A.get a) (A.get b) = some av)
    (snd : ∀ {ga gb av : AV} {u u' : V}, Γ[a]? = some u → Γ[b]? = some u' → ga.Desc Γ w u → gb.Desc Γ w u' →
      f ga gb = some av → ∃ v, evalOp w.mem Γ (o a b) = some v ∧ ∀ Γ' : Env, av.Desc Γ' w v) :
    ∃ v, evalOp w.mem Γ (o a b) = some v ∧ ∀ v0, av.Desc (Γ.push v0) w v := by
  obtain ⟨hna, hnb⟩ := ranged_ne_any h1 h2 hs
  obtain ⟨u, hu, hdu⟩ := h.get' hna
  obtain ⟨u', hu', hdu'⟩ := h.get' hnb
  obtain ⟨v, hv, hd⟩ := snd hu hu' hdu hdu' hs
  exact ⟨v, hv, fun _ => hd _⟩

theorem special_sound {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {o : Op} {av : AV}
    (hs : special A o = some av) (m : Mem) :
    ∃ v, evalOp m Γ o = some v ∧ ∀ v0, av.Desc (Γ.push v0) w v := by
  have nb : ∀ (f : Num → Num → Option Num) g, numBin f .any g = none := fun _ _ => rfl
  have nb' : ∀ (f : Num → Num → Option Num) g, numBin f g .any = none := fun f g => by
    simp [numBin, AV.num]
  cases o
  case iadd a b =>
    exact special_bin (o := .iadd) h (fun g => by simp [addA, AV.asPtr, AV.num]; split <;> simp_all)
      (fun g => by simp [addA, AV.asPtr, AV.num]; split <;> simp_all [AV.num]) hs
      (fun ha hb hda hdb hf => addA_sound ha hb hda hdb hf m)
  case isub a b =>
    exact special_bin (o := .isub) h (nb _) (nb' _) hs (fun ha hb hda hdb hf => subA_sound ha hb hda hdb hf m)
  case imul a b =>
    exact special_bin (o := .imul) h (nb _) (nb' _) hs (fun ha hb hda hdb hf => mulA_sound ha hb hda hdb hf m)
  case ishl a b =>
    exact special_bin (o := .ishl) h (fun _ => rfl) (fun _ => by simp [shlA, AV.num]) hs
      (fun ha hb hda hdb hf => shlA_sound ha hb hda hdb hf m)
  case ushr a b =>
    exact special_bin (o := .ushr) h (fun _ => rfl) (fun _ => by simp [ushrA, AV.num]) hs
      (fun ha hb hda hdb hf => ushrA_sound ha hb hda hdb hf m)
  case band a b =>
    exact special_bin (o := .band) h (fun g => by simp only [bandA]; split <;> simp [roundA, AV.num, numBin])
      (fun g => by simp [bandA, numBin, AV.num]) hs (fun ha hb hda hdb hf => bandA_sound' ha hb hda hdb hf m)
  case bandNot a b =>
    exact special_bin (o := .bandNot) h (fun g => by simp only [bandNotA]; split <;> simp [roundA, AV.num])
      (fun g => by simp [bandNotA]) hs (fun ha hb hda hdb hf => bandNotA_sound ha hb hda hdb hf m)
  case udiv a b =>
    exact special_bin (o := .udiv) h (nb _) (nb' _) hs (fun ha hb hda hdb hf => udivA_sound ha hb hda hdb hf m)
  case ibin k a b =>
    exact special_bin (o := .ibin k) h (fun _ => by cases k <;> rfl)
      (fun g => by cases k <;> simp [ibinA, numBin, AV.num]) hs
      (fun ha hb hda hdb hf => ibinA_sound ha hb hda hdb hf m)
  case uextend64 a =>
    simp only [special] at hs
    have hna : A.get a ≠ .any := by intro e; rw [e] at hs; cases hs
    obtain ⟨u, hu, hdu⟩ := h.get' hna
    obtain ⟨v, hv, hd⟩ := uextA_sound hu hdu hs m
    exact ⟨v, hv, fun _ => hd _⟩
  case icmp c a b =>
    simp only [special] at hs
    exact cmpA_sound h hs m
  all_goals simp [special] at hs

/-- **What `absOp` says of an operation is true of it**: in every state the
    abstract one describes, the operation is defined, and the value it leaves is
    one the result describes in the environment it is pushed onto. -/
theorem absOp_sound {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) {o : Op} {av : AV}
    (ho : absOp A o = some av) : ∃ v, evalOp w.mem Γ o = some v ∧ av.Desc (Γ.push v) w v := by
  unfold absOp at ho
  split at ho
  · obtain ⟨v, hv, hd⟩ := absLoad_sound h ho
    exact ⟨v, hv, hd _⟩
  · rename_i t b
    split at ho
    · rename_i hl
      cases ho
      exact ⟨_, rfl, TyV.sc (Option.isNone_iff_eq_none.mp hl)⟩
    · cases ho
  · rename_i hnl _
    split at ho
    · rename_i hall
      obtain ⟨v, hv, rfl⟩ := Option.map_eq_some_iff.mp ho
      refine ⟨v, ?_, rfl⟩
      rw [evalOp_congr w.mem default Γ Γ o (fun op p e => hnl op p e) (fun _ _ => rfl),
        ← evalOp_positional default Γ _ o (fun i hi => ?_)]
      · exact hv
      · have hk := List.all_eq_true.mp (Bool.and_eq_true_iff.mp hall).2 _ (List.getElem_mem hi)
        rw [List.getElem?_toArray, List.getElem?_map, List.getElem?_eq_getElem hi, Option.map_some,
          h.known hk]
    · split at ho
      · rename_i av' hsp
        cases ho
        obtain ⟨v, hv, hd⟩ := special_sound h hsp w.mem
        exact ⟨v, hv, hd v⟩
      · split at ho
        · rename_i T hT
          split at ho
          · rename_i hl
            cases ho
            simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at hT
            obtain ⟨ts, hts, hT⟩ := hT
            obtain ⟨x, hx⟩ := tyOp_sound (m := w.mem) (argsTy_of h hts) hT
            exact ⟨_, hx, TyV.sc (Option.isNone_iff_eq_none.mp hl)⟩
          · cases ho
        · obtain ⟨T, hT, rfl⟩ := Option.map_eq_some_iff.mp ho
          simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at hT
          obtain ⟨ts, hts, hT⟩ := hT
          exact tyVOp_sound (argsV_of h hts) hT

end AlgorithmLib.Check
