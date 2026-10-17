import AlgorithmLib.Proof.LibSpec
import AlgorithmLib.Surface.Link

/-!
# `Lib.Cuda`'s functions, stated over the driver

A linked program's CUDA calls are to functions of `Surface.LibCuda`, written
over the driver called directly. Each function's statement here is over the
driver's own model (`Sem.cudaDrv`), not over the engine entry point it
replaces: the two act on the same device state but differ, the driver's
`cuCtxSynchronize` waiting for every stream where `cudaSync` is stated to wait
for the default one, so a program proven as it is linked reads its calls by
these.

`sync_spec` is `cudaSync` handed a context: where the driver answers
`cuCtxSynchronize`, `0` for its success and `-1` for any other code, and the
world the driver leaves (`drvSync`). `sync_null` is the call handed no
context. `sync_calls` is the call from a linked program.

Every statement here is of partial correctness.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.HProg.Contracts

namespace CudaSpec
open AlgorithmLib.LibSpec AlgorithmLib.LibCuda AlgorithmLib.Lib

theorem sync_fine : Fine (emitGo sync ⟨1, 0, [], [], [], none⟩).2 := ⟨by decide +kernel, by decide +kernel⟩

/-- **`cudaSync`** handed no context answers `-1` and changes nothing. -/
theorem sync_null (cfg : Cfg) (w : World) :
    Ans cfg [.sc .i64 0] w sync [.i64] (fun x w' => x = c64 (-1) ∧ w' = w) := by
  refine ans_of_wp ?_ sync_fine rfl
  rw [fnBody_of (by decide +kernel)]
  simp only [sync, withCtx, entryParams, wp_bind, wp_params, carriesFrom, wp_ret]
  have h0 : (#[V.sc .i64 0] : Env)[0]? = some (.sc .i64 0) := rfl
  apply wp_iconst64
  refine wp_failIf (x := 0) (y := c64 0) (by slot) (by slot) (fun _ => ⟨rfl, rfl⟩) ?_
  intro h
  exact absurd h (by decide)

/-- What the driver answers to `cuCtxSynchronize` and leaves. -/
abbrev drvSync (w : World) : Option (Option V × World) :=
  extCall (.cuda .ctxSynchronize) [] (obsCall w (.ext (.cuda .ctxSynchronize)) [])

theorem ofInt32_sc (k : Int) : ∃ rc, ofInt .i32 k = .sc .i32 rc := ⟨_, rfl⟩

/-- The driver answers `cuCtxSynchronize` with a result code. -/
theorem drvSync_rc {w w' : World} {v : V} (h : drvSync w = some (some v, w')) : ∃ rc, v = .sc .i32 rc := by
  simp only [drvSync, extCall, List.mapM_nil, Option.pure_def, Option.bind_eq_bind, Option.bind_some, cudaDrv,
    cudaStep, devOnly, cuRes, Option.map_some] at h
  split at h
  · cases h
    exact ofInt32_sc _
  · split at h
    · cases h
    · split at h
      · cases h
      · cases h
        exact ofInt32_sc _

/-- **`cudaSync`** handed a context is the driver's `cuCtxSynchronize`, where
    the driver answers it: `0` for its success, `-1` for any other code, and
    the world the driver leaves. -/
theorem sync_spec (cfg : Cfg) (w : World) (s : UInt64) (hs : s ≠ 0) (hdef : (drvSync w).isSome = true) :
    Ans cfg [.sc .i64 s] w sync [.i64] (fun x w' =>
      ∃ rc, drvSync w = some (some (.sc .i32 rc), w') ∧ x = if rc = 0 then 0 else c64 (-1)) := by
  refine ans_of_wp ?_ sync_fine rfl
  rw [fnBody_of (by decide +kernel)]
  simp only [sync, withCtx, entryParams, wp_bind, wp_params, carriesFrom, wp_ret]
  have h0 : (#[V.sc .i64 s] : Env)[0]? = some (.sc .i64 s) := rfl
  apply wp_iconst64
  refine wp_failIf (x := s) (y := c64 0) (by slot) (by slot) ?_ ?_
  · intro h
    exact absurd (by simpa [cmpInt, show c64 0 = 0 from by decide] using h) hs
  intro _ Γ' hext
  simp only [cu, Prog.ext, AlgorithmLib.Prog.callLocal, wp_bind]
  refine wp_extRc (cs := []) rfl hdef ?_
  intro v w' hv
  simp only [wp_ret]
  obtain ⟨rc, rfl⟩ := drvSync_rc hv
  exact wp_status get_push_size ⟨rc, hv, rfl⟩

/-- A linked program's call to `cudaSync`, at every call depth. -/
theorem sync_calls {fns : Nat → Option Fn} {env : FnEnv} {steps i : Nat}
    (hf : fns i = some (implFn [.i64] (.status sync))) (w : World) (s : UInt64) (hs : s ≠ 0)
    (hdef : (drvSync w).isSome = true) (k : Nat) :
    CallsA (termLocals fns env steps k) i [.sc .i64 s] w (fun x w' =>
      ∃ rc, drvSync w = some (some (.sc .i32 rc), w') ∧ x = if rc = 0 then 0 else c64 (-1)) :=
  Ans.calls hf (fun cfg => sync_spec cfg w s hs hdef) k

end CudaSpec

#print axioms CudaSpec.sync_null
#print axioms CudaSpec.sync_calls
