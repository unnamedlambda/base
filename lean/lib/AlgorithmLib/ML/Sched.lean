import AlgorithmLib.ML.Kernels

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
  deriving DecidableEq, Repr

/-- How many loop trips this schedule needs to cover `n` elements with one
    warp. -/
def Sched.trips : Sched → Nat → Nat
  | .vec4,    n => n / 128
  | .strided, n => n / 32

/-- Elements consumed per lane per step. -/
def Sched.width : Sched → Nat
  | .vec4    => 4
  | .strided => 1

/-- **The kernel for a choice.**  Both arms are schemas that were already
    proven; this is a menu over them, not new machinery. -/
def Sched.realize : Sched → Buf → Buf → IdxE → IdxE → Buf → IdxE → Nat → EWStmt
  | .vec4,    bA, bB, ixA, ixB, out, oi, n => warpDotV4  bA bB ixA ixB out oi (n / 128)
  | .strided, bA, bB, ixA, ixB, out, oi, n => dotStrided bA bB ixA ixB out oi (n / 32)

/-- **What that choice computes**, per lane, before the butterfly.

    The two arms are deliberately different functions.  `.vec4` adds four
    products into the accumulator per step, in load order; `.strided` adds one.
    Read this as the *definition* of the schedule's numerics. -/
def Sched.fold : Sched → (Nat → Float32) → (Nat → Float32) →
    (Nat → Lane → Nat) → (Nat → Lane → Nat) → Nat → Lane → Float32
  | .vec4,    memA, memB, fA, fB, n => dotLane        memA memB fA fB (n / 128)
  | .strided, memA, memB, fA, fB, n => dotStridedLane memA memB fA fB (n / 32)

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

/-- The PTX for a choice. -/
def Sched.ptx (s : Sched) (bA bB : Buf) (ixA ixB : IdxE) (out : Buf) (oi : IdxE)
    (n nbuf : Nat) : String :=
  emitProvenKernelN "main" nbuf 0 (s.realize bA bB ixA ixB out oi n)

/-- **What it would cost to call two schedules interchangeable.**

    Not a theorem — a proposition, and a false one at `Float32`, for the same
    reason `SumAssoc` is: regrouping a sum changes it.  It is stated so that a
    claim of schedule-independence has somewhere to point, and so that the
    absence of a proof is visible.

    Its shape is deliberately the strongest thing a tuning API might want: that
    switching schedules never changes a result. -/
def SchedAgree (s t : Sched) : Prop :=
  ∀ (memA memB : Nat → Float32) (fA fB : Nat → Lane → Nat) (n : Nat) (l : Lane),
    bflyFold (s.fold memA memB fA fB n) l = bflyFold (t.fold memA memB fA fB n) l

/-- Reflexivity is all that is provable, and stating it is the point: this is
    the *only* instance of `SchedAgree` in the stack. -/
theorem schedAgree_refl (s : Sched) : SchedAgree s s := fun _ _ _ _ _ _ => rfl

/-- The menu, for a tool or a docstring to enumerate. -/
def Sched.all : List Sched := [.vec4, .strided]

/-- A one-line description of each choice. -/
def Sched.describe : Sched → String
  | .vec4    => "ld.global.v4.f32, 4 contiguous elements per lane per step, n/128 trips"
  | .strided => "ld.global.f32, 1 element per lane per step, n/32 trips, free addressing"

end AlgorithmLib.ML
