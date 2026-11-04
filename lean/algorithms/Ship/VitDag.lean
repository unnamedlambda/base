import AlgorithmLib.Check.DrvState
import AlgorithmLib.MLHost.TapeRun
import Vit.Algorithm
import Ship.VitState

/-!
# The calls the ViT's ranges and captures make, as data

The steps a range issues in the schedule's order (`VitCalls.vSteps`), and
the calls a capture of a range makes over the pooled streams (`vCapCalls`):
the fork onto the pooled streams, each unit's waits on the events its
predecessors on other streams recorded, its launch or product on its
stream, the event it records, and the join back. Each pooled stream's
cuBLAS handle is on that stream (`vBs`).

A plain file, not a module, as the Ship files it reads are.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
  AlgorithmLib.ML

namespace VitCalls

/-- **The steps `vIssue lo hi` takes**, in the schedule's order: a group's
    kernel, by its index in the layout, or a contraction's product. -/
def vSteps (lo hi : Nat) : List (Sum Nat ML.Gemm) :=
  ((List.range Vit.VNUNIT).zip Vit.vUnits).filterMap fun (k, u) =>
    if lo ≤ Vit.vUnitLo u && Vit.vUnitHi u < hi then
      if Vit.vIsGemm k then
        ((Vit.vUnitOps (Vit.vUnitArr.getD k [])).head? >>= ML.TOp.vendorGemm).map .inr
      else some (.inl (Vit.vKernIx.getD k 0))
    else none

end VitCalls

namespace VitDag

/-- The capture stream, a pooled stream, its cuBLAS handle, an event. -/
def S0 : UInt64 := drvHandle kStream 0
def pool (s : Nat) : UInt64 := drvHandle kStream (s + 1)
def hdl (s : Nat) : UInt64 := drvHandle kBlas (s + 1)
def ev (e : Nat) : UInt64 := drvHandle kEvent e

/-- **The calls a capture of `[lo, hi)` makes after it begins** (`vFork`,
    `vIssueDag`, `vJoin`), as `ML.CapCall`s. -/
def vCapCalls (lo hi : Nat) : List CapCall :=
  let d := Vit.vDagIn lo hi
  ([.record (ev 0) S0] ++ (List.range Vit.VNSTRM).map fun s => .wait (pool s) (ev 0)) ++
  ((List.range Vit.VNUNIT).zip Vit.vUnits).flatMap (fun (k, u) =>
    if lo ≤ Vit.vUnitLo u && Vit.vUnitHi u < hi then
      let s := Vit.vStrm.getD k 0
      (d.ewait.getD k []).map (fun e => .wait (pool s) (ev (Vit.vDagEvent e))) ++
      (if Vit.vIsGemm k then
         match (Vit.vUnitOps (Vit.vUnitArr.getD k [])).head? >>= TOp.vendorGemm with
         | some g => [.gemm g (hdl s)]
         | none => []
       else [.launch (Vit.vKernIx.getD k 0) (pool s)]) ++
      (match d.erec.getD k none with
       | none => []
       | some e => [.record (ev (Vit.vDagEvent e)) (pool s)])
    else []) ++
  (List.range Vit.VNSTRM).flatMap fun s => [.record (ev (Vit.VNSTRM + s)) (pool s), .wait S0 (ev (Vit.VNSTRM + s))]

/-- The stream each pooled cuBLAS handle is on. -/
def vBs (h : UInt64) : Option UInt64 :=
  ((List.range Vit.VNSTRM).find? fun s => hdl s == h).map pool

theorem blas_on : (List.range Vit.VNSTRM).all (fun s =>
    (VitState.after 0 0).contains (.blasOn (s + 1) (pool s))) = true := by native_decide

/-- **Each pooled handle is on its stream**, in what `main` leaves. -/
theorem vBs_ok {w : World} (h : VitState.Run w) :
    ∀ hb s, vBs hb = some s → w.dev.blasParty? hb = w.dev.streamParty? s := by
  intro hb s e
  obtain ⟨i, hi, rfl⟩ := Option.map_eq_some_iff.mp e
  have hmem := List.mem_of_find?_eq_some hi
  have heq : hdl i = hb := by simpa using List.find?_some hi
  subst heq
  have hfact : w.dev.blas[i + 1]? = some (some (pool i)) := by
    have := List.all_eq_true.mp blas_on i hmem
    rw [List.contains_iff_mem] at this
    exact h _ this
  have hlt : i + 1 < 2 ^ 48 := by
    have := List.mem_range.mp hmem
    simp only [Vit.VNSTRM] at this; omega
  simp only [Dev.blasParty?, hdl, handleOf_drvHandle (by decide : kBlas < 2 ^ 8) hlt, Option.bind_eq_bind,
    Option.bind_some, hfact, Option.join_some]

end VitDag
