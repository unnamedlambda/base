import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.DrvState
import AlgorithmLib.Proof.Typestate
import Demo.Matmul
import Demo.Scene
import Demo.BlackHole

/-!
# The CUDA apps neither misuse a call nor fault

Each app's `main`, proven by the condition generator from the typestate it
starts in, for any lengths of data and output the caller hands over: kernels and cuBLAS that keep buffer sizes (`oracles`), the arena
the artifact asks for, and the driver loaded and untouched with each
module's PTX and each entry point's name in the arena, in the driver's
standard layout (`Drv.App.loadTS`).

Not faulting means every load and store finds the bytes it reaches,
and every operation answers.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace CudaAppsSafe

-- Matmul, at the shapes it ships with

abbrev MatmulMain : World → Prop := Contracts.TState.holds
  ((Matmul.app Matmul.M Matmul.K Matmul.N).loadTS ++ [.oracles])

ship_entry Matmul : Matmul.code Matmul.M Matmul.K Matmul.N under MatmulMain

-- Scene, at the scene it ships with

abbrev SceneMain : World → Prop := Contracts.TState.holds
  ((Scene.app Scene.defaultScene).loadTS ++ [.oracles])

/-- The signature `SceneMain` states of the renderer's kernel. -/
theorem scene_sig : HProg.Sem.ptxParamSizes (Scene.ptxSource Scene.defaultScene) "main" = some [8] := by
  native_decide

ship_entry Scene : Scene.code Scene.defaultScene under SceneMain

-- BlackHole, at the scene it ships with

abbrev BlackHoleMain : World → Prop := Contracts.TState.holds
  ((BlackHole.app BlackHole.defaultBlackHole).loadTS ++ [.oracles])

/-- The signatures `BlackHoleMain` states of the renderer's two entry points. -/
theorem blackHole_sigs :
    HProg.Sem.ptxParamSizes (BlackHole.ptxSource BlackHole.defaultBlackHole) "render_hdr" = some [8] ∧
      HProg.Sem.ptxParamSizes (BlackHole.ptxSource BlackHole.defaultBlackHole) "composite_bloom" = some [8, 8] := by
  native_decide

ship_entry BlackHole : BlackHole.code BlackHole.defaultBlackHole under BlackHoleMain

end CudaAppsSafe

#print axioms CudaAppsSafe.Matmul_no_misuse
#print axioms CudaAppsSafe.Scene_no_misuse
#print axioms CudaAppsSafe.BlackHole_no_misuse
