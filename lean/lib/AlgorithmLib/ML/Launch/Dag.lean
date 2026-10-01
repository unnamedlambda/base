module
public import AlgorithmLib.ML.Launch.Interchange
meta import AlgorithmLib.ML.Launch.Interchange
public import AlgorithmLib.Host.DevProg
meta import AlgorithmLib.Host.DevProg
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# Launches ordered by a graph

A `Pipeline` runs its stages in the one order it lists. What a host issues on
several streams, or a captured graph replays, is only partially ordered: a
`StageDag` is stages with the edges they must keep, and it is a device program
(`Device.denote_linearisation`) over the warp machine's memory. So a schedule
is a choice only where it cannot matter: when stages no path orders commute ---
which `Interchange`'s footprint lemmas establish --- every order that keeps the
edges computes the same memory.

A `Pipeline` is the special case whose edges chain its stages, and its
`denote` is the device program's in list order.
-/

namespace AlgorithmLib.ML

open AlgorithmLib.Device

/-- Stages, and the order some of them must keep. -/
structure StageDag where
  stages : List StageSpec
  edges : List (Nat × Nat)

/-- What stage `i` does to memory. -/
noncomputable def StageDag.stepAt (D : StageDag) (i : Nat) :
    (Buf → Nat → Float32) → (Buf → Nat → Float32) :=
  match D.stages[i]? with
  | some S => S.step
  | none => id

/-- A pipeline: its stages, each before the next. -/
def Pipeline.toDag (P : Pipeline) : StageDag :=
  { stages := P.stages, edges := (List.range (P.stages.length - 1)).map fun i => (i, i + 1) }

theorem denote_range_stages (L : List StageSpec) (m : Buf → Nat → Float32) :
    L.foldl (fun mm S => S.step mm) m
      = Device.denote (StageDag.stepAt { stages := L, edges := [] }) (List.range L.length) m := by
  have hmap : (List.range L.length).map (StageDag.stepAt { stages := L, edges := [] })
      = L.map StageSpec.step := by
    apply List.ext_getElem
    · simp
    · intro i h1 h2
      have hi : i < L.length := by simpa using h1
      simp [StageDag.stepAt, List.getElem?_eq_getElem hi]
  have e1 : Device.denote (StageDag.stepAt { stages := L, edges := [] }) (List.range L.length) m
      = ((List.range L.length).map (StageDag.stepAt { stages := L, edges := [] })).foldl
          (fun mm g => g mm) m := by
    simp [Device.denote, List.foldl_map]
  have e2 : L.foldl (fun mm S => S.step mm) m = (L.map StageSpec.step).foldl (fun mm g => g mm) m := by
    simp [List.foldl_map]
  rw [e1, e2, hmap]

/-- **A pipeline's denotation is its device program's, in list order.** -/
theorem Pipeline.denote_eq_dag (P : Pipeline) (m : Buf → Nat → Float32) :
    P.denote m = Device.denote P.toDag.stepAt (List.range P.stages.length) m := by
  rw [Pipeline.denote, denote_range_stages]
  rfl

/-- **Any two schedules of a race-free stage graph leave the same memory.** -/
theorem StageDag.schedules_agree (D : StageDag)
    (hrf : ∀ u v S T, u ≠ v → ¬ Reach D.edges u v → ¬ Reach D.edges v u →
      D.stages[u]? = some S → D.stages[v]? = some T → Commute S.step T.step)
    (o₁ o₂ : List Nat) (hp : o₁.Perm o₂) (hnd : o₁.Nodup)
    (h₁ : Resp D.edges o₁) (h₂ : Resp D.edges o₂) (m : Buf → Nat → Float32) :
    Device.denote D.stepAt o₁ m = Device.denote D.stepAt o₂ m := by
  refine Device.denote_linearisation D.edges D.stepAt ?_ o₁ o₂ hp hnd h₁ h₂ m
  intro u v huv hu hv mm
  unfold StageDag.stepAt
  cases hs : D.stages[u]? with
  | none => rfl
  | some S =>
    cases ht : D.stages[v]? with
    | none => rfl
    | some T => exact hrf u v S T huv hu hv hs ht mm

end AlgorithmLib.ML
