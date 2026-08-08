import AlgorithmLib.ML.Kernels
import AlgorithmLib.ML.Rewrite

namespace AlgorithmLib.ML

/-- A scheduling choice for a strided warp reduction. -/
inductive Sched where
  /-- Quad loads: four contiguous elements per lane per step.  Fewer, wider
      memory transactions; requires the four to be adjacent. -/
  | vec4
  /-- Scalar loads: one element per lane per step.  Slower per element, but the
      addressing is unconstrained — which is what a transposed (column) walk
      needs. -/
  | strided
  /-- Blocked scalar loads: lane `l` walks the contiguous run `l*(n/32)…`
      instead of interleaving with its neighbours.  Same kernel and same fold
      as `.strided`; only the walk differs, so the warp's 32 loads land in 32
      different segments rather than one. -/
  | blocked
  deriving DecidableEq, Repr

/-- How many loop trips this schedule needs to cover `n` elements with one
    warp. -/
def Sched.trips : Sched → Nat → Nat
  | .vec4,    n => n / 128
  | .strided, n => n / 32
  | .blocked, n => n / 32

/-- Elements consumed per lane per step. -/
def Sched.width : Sched → Nat
  | .vec4    => 4
  | .strided => 1
  | .blocked => 1

/-- **The kernel for a choice.**  Both arms are schemas that were already
    proven; this is a menu over them, not new machinery. -/
def Sched.realize : Sched → Buf → Buf → IdxE → IdxE → Buf → IdxE → Nat → EWStmt
  | .vec4,    bA, bB, ixA, ixB, out, oi, n => warpDotV4  bA bB ixA ixB out oi (n / 128)
  | .strided, bA, bB, ixA, ixB, out, oi, n => dotStrided bA bB ixA ixB out oi (n / 32)
  | .blocked, bA, bB, ixA, ixB, out, oi, n => dotStrided bA bB ixA ixB out oi (n / 32)

/-- **What that choice computes**, per lane, before the butterfly.

    The two arms are deliberately different functions.  `.vec4` adds four
    products into the accumulator per step, in load order; `.strided` adds one.
    Read this as the *definition* of the schedule's numerics. -/
def Sched.fold : Sched → (Nat → Float32) → (Nat → Float32) →
    (Nat → Lane → Nat) → (Nat → Lane → Nat) → Nat → Lane → Float32
  | .vec4,    memA, memB, fA, fB, n => dotLane        memA memB fA fB (n / 128)
  | .strided, memA, memB, fA, fB, n => dotStridedLane memA memB fA fB (n / 32)
  | .blocked, memA, memB, fA, fB, n => dotStridedLane memA memB fA fB (n / 32)

/-- **The addressing each schedule emits**: the index lane `l` reads on trip
    `i` of a walk starting at `base`.  `.vec4` gives each lane four contiguous
    elements from `4·l` and strides the warp by 128; `.strided` gives it one at
    `l` and strides by 32.  A schedule is a fold *and* a walk, and the walk is
    what has to be shown to cover the input.

    `base` is added last, which is where a row walk over a row-major matrix puts
    it — so a kernel's `IdxE` evaluates to this on the nose. -/
def Sched.idx : Sched → Nat → Nat → Nat → Lane → Nat
  | .vec4,    _, base, i, l => i * 128 + l.val * 4 + base
  | .strided, _, base, i, l => i * 32 + l.val + base
  | .blocked, n, base, i, l => l.val * (n / 32) + i + base

/-- **Every schedule realizes its own fold — exactly.**

    One `cases`, two existing theorems.  A new schedule is a new constructor
    plus a new arm here; nothing else in the stack changes, which is what makes
    this a menu rather than a rewrite of the compiler. -/
theorem Sched.realize_spec (s : Sched) (bA bB : Buf) (ixA ixB : IdxE) (out : Buf)
    (oi : IdxE) (n cta : Nat) (st : WSt) :
    (((s.realize bA bB ixA ixB out oi n).elabIn cta).run st).mem out
        (oi.eval cta 0 ⟨0, by decide⟩)
      = bflyFold (s.fold (st.mem bA) (st.mem bB)
          (fun i l => ixA.eval cta i l) (fun i l => ixB.eval cta i l) n)
          ⟨0, by decide⟩ := by
  cases s with
  | vec4    => exact warpDotV4_spec  bA bB ixA ixB out oi (n / 128) cta st
  | strided => exact dotStrided_spec bA bB ixA ixB out oi (n / 32)  cta st
  | blocked => exact dotStrided_spec bA bB ixA ixB out oi (n / 32)  cta st

/-- The PTX for a choice. -/
def Sched.ptx (s : Sched) (bA bB : Buf) (ixA ixB : IdxE) (out : Buf) (oi : IdxE)
    (n nbuf : Nat) : String :=
  emitProvenKernelN "main" nbuf 0 (s.realize bA bB ixA ixB out oi n)

/-- **Every schedule computes the flat dot product of the first `n` elements.**

    Each arm discharges the same two obligations at its own walk: that the walk
    covers `0 … n-1` once, and that its fold shape reduces to one addition per
    element.  `Law.laneRegroup` then supplies the reassociation, and only that.
    The `128 ∣ n` side condition is the quad schedule's tile: `n/128` trips of
    128 elements have to be all of them. -/
theorem Sched.fold_eq_flatSum (h : AllHold [Law.laneRegroup]) (s : Sched)
    (memA memB : Nat → Float32) (bA bB n : Nat) (hn : n % 128 = 0) :
    bflyFold (s.fold memA memB (s.idx n bA) (s.idx n bB) n) ⟨0, by decide⟩
      = (List.range n).foldl
          (fun a i => NumOps.add a (NumOps.mul (memA (i + bA)) (memB (i + bB))))
          (NumOps.ofNat 0) := by
  cases s with
  | vec4 =>
      have e : (n/128) * 128 = n := by omega
      have hq := quad_eq_flatSum_at h memA memB bA bB (n/128)
      rw [e] at hq
      exact hq
  | strided =>
      have e : (n/32) * 32 = n := by omega
      have hs := strided_eq_flatSum_at h memA memB bA bB (n/32)
      rw [e] at hs
      exact hs
  | blocked =>
      have e : (n/32) * 32 = n := by omega
      have hb := blocked_eq_flatSum_at h memA memB bA bB (n/32)
      rw [e] at hb
      exact hb

/-- **Any two schedules agree**, each at the addressing it actually emits.

    This is the theorem a tuning API needs: choosing `.vec4` over `.strided` for
    speed does not change the answer.  It is uniform in `s` and `t`, so a new
    constructor buys agreement with every existing schedule by proving one arm
    of `fold_eq_flatSum` — its coverage proof and its fold shape — and nothing
    else in the stack moves. -/
theorem sched_agree_at_idx (h : AllHold [Law.laneRegroup]) (s t : Sched)
    (memA memB : Nat → Float32) (bA bB n : Nat) (hn : n % 128 = 0) :
    bflyFold (s.fold memA memB (s.idx n bA) (s.idx n bB) n) ⟨0, by decide⟩
      = bflyFold (t.fold memA memB (t.idx n bA) (t.idx n bB) n) ⟨0, by decide⟩ :=
  (Sched.fold_eq_flatSum h s memA memB bA bB n hn).trans
    (Sched.fold_eq_flatSum h t memA memB bA bB n hn).symm

/-- **The stronger claim, which is false**: that two schedules agree at *any*
    shared index functions.  They do not — at a common `fA` the quad arm reads
    `fA i l … fA i l + 3` and the strided arm reads only `fA i l`, so the two
    cover different sets.  Stated so that `sched_agree_at_idx`'s restriction to
    `Sched.idx` is visible as a restriction rather than a detail. -/
def SchedAgree (s t : Sched) : Prop :=
  ∀ (memA memB : Nat → Float32) (fA fB : Nat → Lane → Nat) (n : Nat) (l : Lane),
    bflyFold (s.fold memA memB fA fB n) l = bflyFold (t.fold memA memB fA fB n) l

/-- Reflexivity is all of `SchedAgree` that holds; agreement between *distinct*
    schedules is `sched_agree_at_idx`, at their own walks and under its law. -/
theorem schedAgree_refl (s : Sched) : SchedAgree s s := fun _ _ _ _ _ _ => rfl

/-- The menu, for a tool or a docstring to enumerate. -/
def Sched.all : List Sched := [.vec4, .strided, .blocked]

/-- A one-line description of each choice. -/
def Sched.describe : Sched → String
  | .vec4    => "ld.global.v4.f32, 4 contiguous elements per lane per step, n/128 trips"
  | .strided => "ld.global.f32, 1 element per lane per step, n/32 trips, free addressing"
  | .blocked => "ld.global.f32, lane l walks the contiguous run l*(n/32)…, n/32 trips"

end AlgorithmLib.ML
