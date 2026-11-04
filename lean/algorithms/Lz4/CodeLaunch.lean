module
public import Lz4.CodeGrid
public import AlgorithmLib.Sound.PtxLaunch
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lz4.CodeLaunch` — a launch of the compressor, on the Code and the PTX

The compressor's blocks, from the world a launch starts from
(`launchWorld`), run in order (`grid_runs`): a launch of the Code
(`LaunchRuns`), every byte outside shared memory the launch's with each
block's byte stores made in block order. The PTX entry the kernel compiles to
runs the same launch to the same device memory (`launch_sound_bits`), so the
two agree (`launch_runs`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Prog
  AlgorithmLib.LZ4WarpDSL AlgorithmLib.LibSpec AlgorithmLib.Kernel

open AlgorithmLib.ML.LaneWarp (NotArena Writable)

namespace Lz4Code

/-- The world a launch starts its blocks from: device memory `bufs`, nothing
    else, the hardware's NaN rule. -/
abbrev launchWorld (bufs : Nat → ByteArray) (nan : ClifTy → UInt64 → UInt64) : World :=
  { mem := { arena := .empty, data := .empty, out := .empty, bufs, nan } }

set_option maxHeartbeats 2000000 in
/-- **A launch of the compressor runs, on the Code and on the PTX it compiles
    to**: `numBlocks` blocks of 32 threads on device memory `bufs` leave
    `bufs'`, every byte outside shared memory the launch's with each block's
    byte stores made in block order; the PTX entry the kernel compiles to
    (`he`), run on hardware with the GPU's float rules (`HwOk`), leaves the
    same `bufs'`. -/
theorem launch_runs (hw : Hw) (hhw : PtxSound.HwOk hw) (name : String) {e : PtxIr.Entry}
    {numBlocks inStride outStride lenOff hashLog f : Nat} {inP outA : UInt64}
    (he : (compressorKernel name numBlocks inStride outStride lenOff hashLog).entry = .ok e)
    (I O : Nat → UInt64 → Prop) (G : Nat → Array UInt8) (bufs : Nat → ByteArray)
    (hnb : numBlocks * 32 + 32 < 2 ^ 32) (h5 : 5 ≤ hashLog) (hh : 2 * 2 ^ hashLog + 64 < 2 ^ 36) (hf : f ≤ 2 ^ 64)
    (hfine : Fine (emitGo (fnBody (idParams ++ [.i64, .i64]) (ML.Body.body 2 (compressorS numBlocks inStride outStride lenOff hashLog)))
      ⟨(idParams ++ [ClifTy.i64, .i64]).length, 0, [], [], [], none⟩).2)
    (hflags : ∀ r, r ∈ ["inBase","outBase","tbl","op","searchPos","litAnchor"] →
      (flagRegs (compressorBodyWStmt inStride lenOff hashLog)).contains r = false)
    (hIna : ∀ c a, I c a → NotArena a) (hOna : ∀ c a, O c a → NotArena a)
    (hok : ∀ c, c < numBlocks → okOf (I c) (O c) (okLZ4 (I c)) f (compressorBodyWStmt inStride lenOff hashLog)
      (startWS numBlocks inStride outStride hashLog c inP (G c)))
    (hI : ∀ c, c < numBlocks → ∀ a, I c a →
      (launchWorld bufs hw.nan).mem.load a 1 = some ((G c).getD a.toNat 0).toUInt64)
    (hO : ∀ c, c < numBlocks → ∀ a, O c a → Writable (launchWorld bufs hw.nan).mem a 1)
    (hdis : ∀ c c', c' < c → c < numBlocks → ∀ a, O c' a → ¬ I c a) :
    ∃ u : World, LaunchRuns (compressorKernel name numBlocks inStride outStride lenOff hashLog) numBlocks 32
        [.sc .i64 inP, .sc .i64 outA] bufs u.mem.bufs hw ∧
      RunsLaunch (PtxRun.runner e hw) (compressorKernel name numBlocks inStride outStride lenOff hashLog).smem numBlocks 32
        [.sc .i64 inP, .sc .i64 outA] bufs u.mem.bufs hw.nan ∧
      ∀ a, NotArena a → u.mem.load a 1 = replay ((List.range numBlocks).flatMap fun c =>
        blockStores f (compressorBodyWStmt inStride lenOff hashLog)
          (startWS numBlocks inStride outStride hashLog c inP (G c))) a ((launchWorld bufs hw.nan).mem.load a 1) := by
  obtain ⟨u, hu, hum, -⟩ := grid_runs (outA := outA) hw (fun k v => by
      have h := hhw.approx k v
      rw [← h, UInt64.toNat_and]; exact Nat.lt_of_le_of_lt Nat.and_le_right (by simp [widthMask])) name I O G
    (launchWorld bufs hw.nan) hnb h5 hh hf hfine hflags hIna hOna hok hI hO hdis rfl numBlocks (Nat.le_refl _)
  have hL : LaunchRuns (compressorKernel name numBlocks inStride outStride lenOff hashLog) numBlocks 32
      [.sc .i64 inP, .sc .i64 outA] bufs u.mem.bufs hw := ⟨u, hu, rfl⟩
  have hmask : ∀ x : UInt64, x &&& 18446744073709551615 = x := fun x => by
    apply UInt64.toNat_inj.mp
    rw [UInt64.toNat_and, show (18446744073709551615 : UInt64).toNat = 2 ^ 64 - 1 from rfl,
      Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt x.toNat_lt]
  refine ⟨u, hL, ?_, hum⟩
  exact PtxSound.launch_sound_bits _ e he hw hhw numBlocks 32 [inP, outA] rfl
    (fun i h1 h2 => by
      rcases i with _ | _ | i
      · simp [compressorKernel, widthMask, hmask]
      · simp [compressorKernel, widthMask, hmask]
      · exact absurd h2 (by simp))
    (fun _ => Nat.le_refl _) bufs u.mem.bufs hL

end Lz4Code
