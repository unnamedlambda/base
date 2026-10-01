import AlgorithmLib.Proof.Typestate
import Demo.Matmul
import Demo.Scene
import Demo.BlackHole

/-!
# The CUDA apps neither misuse a call nor fault

Each app's `main`, proven by the condition generator from the typestate it
starts in, for any lengths of data and output the caller hands over: memory
not frozen, kernels and cuBLAS that keep buffer sizes (`oracles`), the arena
the artifact asks for, and each kernel's text inside the field the layout
gives it (`cstrIn`).

Not faulting means every load and store finds the bytes it reaches,
and every operation answers.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace CudaAppsSafe

-- Matmul, at the shapes it ships with

abbrev MatmulMain : World → Prop := Contracts.TState.holds
  [.part .cudaDevice true, .part .frozen false, .part .cuda false, .part .capturing false, .oracles,
   .room .arena (Matmul.DATA_OFF + (Matmul.M * Matmul.K + Matmul.K * Matmul.N + Matmul.M * Matmul.N) * 4),
   .cstrIn (regionBase .arena + UInt64.ofNat Matmul.PTX_OFF) Matmul.PTX_REGION]

theorem Matmul_fine :
    Fine (emitGo (Matmul.code Matmul.M Matmul.K Matmul.N : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem Matmul_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : MatmulMain w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (Matmul.code Matmul.M Matmul.K Matmul.N : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (Matmul.code Matmul.M Matmul.K Matmul.N : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry MatmulMain (runArgs dataLen outLen) (by prog_vc MatmulMain) Matmul_fine rfl h m

-- Scene, at the scene it ships with

abbrev SceneMain : World → Prop := Contracts.TState.holds
  [.part .cudaDevice true, .part .frozen false, .part .cuda false, .part .capturing false, .oracles,
   .room .arena (Scene.pixelsOff + Scene.pixelBytes Scene.defaultScene),
   .cstrIn (regionBase .arena + UInt64.ofNat Scene.ptxOff) Scene.ptxRegion]

theorem Scene_fine :
    Fine (emitGo (Scene.code Scene.defaultScene : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem Scene_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : SceneMain w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (Scene.code Scene.defaultScene : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (Scene.code Scene.defaultScene : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry SceneMain (runArgs dataLen outLen) (by prog_vc SceneMain) Scene_fine rfl h m

-- BlackHole, at the scene it ships with

abbrev BlackHoleMain : World → Prop := Contracts.TState.holds
  [.part .cudaDevice true, .part .frozen false, .part .cuda false, .part .capturing false, .oracles,
   .room .arena (BlackHole.pixelsOff + BlackHole.pixelBytes BlackHole.defaultBlackHole),
   .cstrIn (regionBase .arena + UInt64.ofNat BlackHole.ptxOff) BlackHole.ptxRegion,
   .cstrIn (regionBase .arena + UInt64.ofNat BlackHole.nameOffRender) BlackHole.nameRegion,
   .cstrIn (regionBase .arena + UInt64.ofNat BlackHole.nameOffComposite) BlackHole.nameRegion]

theorem BlackHole_fine :
    Fine (emitGo (BlackHole.code BlackHole.defaultBlackHole : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem BlackHole_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : BlackHoleMain w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (BlackHole.code BlackHole.defaultBlackHole : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (BlackHole.code BlackHole.defaultBlackHole : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry BlackHoleMain (runArgs dataLen outLen) (by prog_vc BlackHoleMain) BlackHole_fine rfl h m

end CudaAppsSafe

#print axioms CudaAppsSafe.Matmul_no_misuse
#print axioms CudaAppsSafe.Scene_no_misuse
#print axioms CudaAppsSafe.BlackHole_no_misuse
