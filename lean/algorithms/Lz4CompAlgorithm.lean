import AlgorithmLib
import AlgorithmLib.LZ4SimtSerialize
import AlgorithmLib.LZ4WarpKernel
import AlgorithmLib.LZ4CompTop

set_option maxRecDepth 8192

open Lean (Json)
open AlgorithmLib
open AlgorithmLib.PTX

namespace Algorithm

def corpusBytes : Nat := 209715200   -- 3200 * 65536 (fixed corpus prefix)
def rLaunches  : Nat := 20

def rPTX_OFF   : Nat := 0x100


-- ── Warp-cooperative compressor: 1 warp/block, 4096-entry shared-mem hash table.
-- Parameterized by block-size log2 (pos+1 must fit u16: blkLog ≤ 16). --
structure WP where
  blkLog : Nat
def wHashLog     : Nat := 12                  -- 4096-entry shared table (8 KiB/warp)

def WP.inStride  (w : WP) : Nat := 2 ^ w.blkLog
def WP.numBlk    (w : WP) : Nat := corpusBytes / w.inStride
def WP.lenOff    (w : WP) : Nat := w.inStride + w.inStride / 16 + 256
def WP.outStride (w : WP) : Nat := w.lenOff + 8  -- compressed | u32 length (no global table)
def wTableBytes  : Nat := (2 ^ wHashLog) * 2
def wSmem        : Nat := wTableBytes
def wBlockDim    : Nat := AlgorithmLib.LZ4Simt.modelBlockDim
def WP.gridX     (w : WP) : Nat := w.numBlk
def WP.totIn     (w : WP) : Nat := w.numBlk * w.inStride
def WP.totOut    (w : WP) : Nat := w.numBlk * w.outStride
-- `9*inStride` of headroom past the last warp's stride: the body simulation's
-- budget is 9 bytes per remaining input byte, which `hbuf` needs for every warp.
def WP.outAlloc  (w : WP) : Nat := w.totOut + 9 * w.inStride

/-- The kernel this artifact ships.  `warpKernelDSLStr` serializes THIS, and
    `ShippedCorrect` is stated about THIS — one definition, so the proven object and
    the executed object cannot be different programs. -/
def WP.kernel (w : WP) : Array AlgorithmLib.LZ4Simt.SInstr :=
  AlgorithmLib.LZ4WarpDSL.warpKernelDSL w.numBlk w.inStride w.outStride w.lenOff wHashLog

def warpKernelDSLStr (w : WP) : String :=
  AlgorithmLib.LZ4Simt.serializeKernel w.kernel wSmem

-- ── Payload layout, derived from the serialized kernel ────────────────────────
-- Offsets are relative to the PTX's actual length, so there is no fixed slot to
-- overflow (Nat subtraction would truncate silently rather than fail).
/-- The bytes at `rPTX_OFF`.  `ptxLen` is this list's length and `warpPayloadDSL`
    emits this same list, so offsets and image cannot disagree. -/
def WP.ptxBytes  (w : WP) : List UInt8 := (warpKernelDSLStr w).toUTF8.toList ++ [0]
def WP.ptxLen    (w : WP) : Nat := w.ptxBytes.length
/-- 16-byte-align the binding table so the i64 output columns stay aligned. -/
def WP.bindOff   (w : WP) : Nat := (rPTX_OFF + w.ptxLen + 15) / 16 * 16
def WP.rowOff    (w : WP) : Nat := w.bindOff + 0x40
def WP.passOff   (w : WP) : Nat := w.rowOff + 0x08
def WP.launOff   (w : WP) : Nat := w.rowOff + 0x10
def WP.bytesOff  (w : WP) : Nat := w.rowOff + 0x18
def WP.instrOff  (w : WP) : Nat := w.rowOff + 0x20
def WP.outstrOff (w : WP) : Nat := w.rowOff + 0x28
def WP.lenoffOff (w : WP) : Nat := w.rowOff + 0x30
def WP.numblkOff (w : WP) : Nat := w.rowOff + 0x38
def WP.memSize   (w : WP) : Nat := w.rowOff + 0x40

def compSchema (w : WP) : List Json :=
  [Output.schema
    [ Output.column "pass" .i64 w.passOff,
      Output.column "launches" .i64 w.launOff,
      Output.column "bytes_per_launch" .i64 w.bytesOff,
      Output.column "in_stride" .i64 w.instrOff,
      Output.column "out_stride" .i64 w.outstrOff,
      Output.column "len_off" .i64 w.lenoffOff,
      Output.column "num_blk" .i64 w.numblkOff ]
    w.rowOff]

open AlgorithmLib.IR in
def warpClif (w : WP) : String := buildProgram do
  let cuda ← declareCudaFFI
  let ptr ← entryBlock
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let dataLen ← load64 (← absAddr ptr 0x20)
  let outPtr ← load64 (← absAddr ptr 0x28)
  cudaInit cuda ptr
  let inBuf ← cudaCreateBuffer cuda ptr dataLen
  let outBuf ← cudaCreateBuffer cuda ptr (← iconst64 w.outAlloc)
  let _ ← cudaUploadRaw cuda ptr inBuf dataPtr dataLen
  let g ← iconst32 w.gridX
  let bk ← iconst32 wBlockDim
  let one32 ← iconst32 1
  let nbufs ← ireduce32 (← iconst64 2)
  let ptxOff ← iconst64 rPTX_OFF
  let bindOff ← iconst64 w.bindOff
  let _ ← forLoopAcc .i64 .i64 (← iconst64 rLaunches) (← iconst64 0) (fun _ acc => do
    let _ ← cudaLaunch cuda ptr ptxOff nbufs bindOff g one32 one32 bk one32 one32
    pure acc)
  let _ ← cudaDownloadRawOffset cuda ptr outBuf (← iconst64 0) outPtr (← iconst64 w.totOut)
  cudaCleanup cuda ptr
  storeAt ptr w.rowOff (← iconst64 1)
  storeAt ptr w.passOff (← iconst64 1)
  storeAt ptr w.launOff (← iconst64 rLaunches)
  storeAt ptr w.bytesOff (← iconst64 w.totIn)
  storeAt ptr w.instrOff (← iconst64 w.inStride)
  storeAt ptr w.outstrOff (← iconst64 w.outStride)
  storeAt ptr w.lenoffOff (← iconst64 w.lenOff)
  storeAt ptr w.numblkOff (← iconst64 w.numBlk)
  ret




def warpPayloadDSL (w : WP) : List UInt8 :=
  zeros rPTX_OFF ++
  (w.ptxBytes ++ zeros (w.bindOff - rPTX_OFF - w.ptxLen)) ++
  (uint32ToBytes 0 ++ uint32ToBytes 1)

/-- The binding table lands at `bindOff` and the image fits `memSize`: `bindOff`
    is `rPTX_OFF + ptxLen` rounded up, so the padding is alignment only. -/
theorem payload_length (w : WP) : (warpPayloadDSL w).length = w.bindOff + 8 := by
  have hdef : w.ptxLen = w.ptxBytes.length := rfl
  have halign : rPTX_OFF + w.ptxLen ≤ w.bindOff := by
    simp only [WP.bindOff, rPTX_OFF]; omega
  simp only [warpPayloadDSL, List.length_append, zeros, List.length_replicate,
    uint32ToBytes, List.length_cons, List.length_nil]
  omega

theorem payload_fits (w : WP) : (warpPayloadDSL w).length ≤ w.memSize := by
  rw [payload_length]; simp only [WP.memSize, WP.rowOff]; omega

def warpArtifactDSL (name : String) (blkLog : Nat) :=
  let w : WP := ⟨blkLog⟩
  AlgorithmLib.toJsonArtifact name
    { cranelift_ir := warpClif w,
      memory_size := w.memSize,
      initial_memory := warpPayloadDSL w }
    { fn_idx := AlgorithmLib.IR.mainFnIdx, output := compSchema w }

-- ── The shipped claim ─────────────────────────────────────────────────────────
-- `ShippedCorrect b` is the correctness statement for the artifact
-- `warpArtifactDSL _ b` emits, in that artifact's own geometry.  Every numeric
-- side condition is discharged inside the proof; the remaining hypotheses are the
-- artifact's contract.  `inpAll.length = totIn` is NOT enforced at run time.
section ShippedClaim
open AlgorithmLib.LZ4WarpDSL

/-- Correctness of the artifact `warpArtifactDSL _ b` emits, for warp `w`, with the
    two device allocations placed ANYWHERE satisfying the stated layout contract —
    the runtime makes two independent `cudaCreateBuffer` calls, so their addresses
    are not related. -/
def ShippedCorrect (b : Nat) : Prop :=
  ∀ (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8),
    w < (WP.mk b).numBlk →
    -- layout contract on the two buffers
    inPtr + w * (WP.mk b).inStride < 2 ^ 40 →
    outPtr + w * (WP.mk b).outStride + 9 * (WP.mk b).inStride < 2 ^ 32 →
    outPtr + w * (WP.mk b).outStride + 9 * (WP.mk b).inStride ≤ gm.size →
    inPtr + w * (WP.mk b).inStride + (WP.mk b).inStride
      ≤ outPtr + w * (WP.mk b).outStride →
    outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff + 3 < 2 ^ 64 →
    outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff + 4 ≤ gm.size →
    ∃ (n : Nat) (ss' : AlgorithmLib.LZ4Simt.SState) (k : Nat),
      AlgorithmLib.LZ4Simt.SReaches (WP.mk b).kernel n
        (AlgorithmLib.LZ4Simt.initSt w inPtr outPtr gm smemB) ss' ∧
      ss'.pc = 272 ∧ 0 < k ∧ k ≤ (WP.mk b).lenOff ∧
      AlgorithmLib.readU32LE ss'.gmem
        (outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff) = k ∧
      AlgorithmLib.LZ4Imp.decompress
        ((List.range k).map (fun i => ss'.gmem.getD
          (outPtr + w * (WP.mk b).outStride + i) 0))
        (WP.mk b).inStride
        = some (gmemInpAt gm (inPtr + w * (WP.mk b).inStride) (WP.mk b).inStride) ∧
      -- warp `w` writes ONLY its own stride, so the per-warp results compose
      (∀ j, j < outPtr + w * (WP.mk b).outStride ∨
            outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff + 4 ≤ j →
        ss'.gmem.getD j 0 = gm.getD j 0)

theorem shipped32_correct : ShippedCorrect 15 := by
  intro w inPtr outPtr gm smemB hw hib40 htop hbuf hdisj hlOtop hlOfit
  have hw64 : w * 32 + 32 < 2 ^ 64 := by
    have h1 : w * 32 ≤ (WP.mk 15).numBlk * 32 := Nat.mul_le_mul_right 32 (Nat.le_of_lt hw)
    have h2 : (WP.mk 15).numBlk * 32 + 32 < 2 ^ 64 := by decide
    omega
  exact warpKernelDSL_tail_roundtrips
    (WP.mk 15).numBlk (WP.mk 15).inStride (WP.mk 15).outStride (WP.mk 15).lenOff wHashLog
    w inPtr outPtr gm smemB
    (by decide) (by decide) hw (by decide) (by decide) (by decide)
    hw64 hib40 htop hbuf hdisj (Nat.le_refl _) hlOtop hlOfit

theorem shipped64_correct : ShippedCorrect 16 := by
  intro w inPtr outPtr gm smemB hw hib40 htop hbuf hdisj hlOtop hlOfit
  have hw64 : w * 32 + 32 < 2 ^ 64 := by
    have h1 : w * 32 ≤ (WP.mk 16).numBlk * 32 := Nat.mul_le_mul_right 32 (Nat.le_of_lt hw)
    have h2 : (WP.mk 16).numBlk * 32 + 32 < 2 ^ 64 := by decide
    omega
  exact warpKernelDSL_tail_roundtrips
    (WP.mk 16).numBlk (WP.mk 16).inStride (WP.mk 16).outStride (WP.mk 16).lenOff wHashLog
    w inPtr outPtr gm smemB
    (by decide) (by decide) hw (by decide) (by decide) (by decide)
    hw64 hib40 htop hbuf hdisj (Nat.le_refl _) hlOtop hlOfit

/-- info: 'Algorithm.shipped32_correct' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms shipped32_correct

/-- info: 'Algorithm.shipped64_correct' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms shipped64_correct


/-- The buffer-placement contract the runtime must satisfy: the two allocations
    are far enough apart and large enough, for every warp of the launch. -/
def LayoutOK (b : Nat) (inPtr outPtr : Nat) (gm : Array UInt8) : Prop :=
  -- The WHOLE input buffer precedes the WHOLE output buffer.  Per-warp
  -- disjointness is not enough: warp `w'`'s output must also miss warp `w`'s
  -- INPUT slice, or warps could clobber each other's source data.
  inPtr + (WP.mk b).numBlk * (WP.mk b).inStride ≤ outPtr ∧
  ∀ w, w < (WP.mk b).numBlk →
    inPtr + w * (WP.mk b).inStride < 2 ^ 40 ∧
    outPtr + w * (WP.mk b).outStride + 9 * (WP.mk b).inStride < 2 ^ 32 ∧
    outPtr + w * (WP.mk b).outStride + 9 * (WP.mk b).inStride ≤ gm.size ∧
    outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff + 3 < 2 ^ 64 ∧
    outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff + 4 ≤ gm.size

/-- **Disjointness / data-race-freedom.**  Distinct warps write disjoint ranges:
    warp `w` touches only `[outPtr + w*outStride, … + lenOff + 4)`, and consecutive
    strides are `outStride = lenOff + 8` apart, leaving 4 bytes of clearance. -/
theorem warp_regions_disjoint (b outPtr w w' : Nat) (hne : w ≠ w') (j : Nat)
    (hj : outPtr + w * (WP.mk b).outStride ≤ j)
    (hj2 : j < outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff + 4) :
    j < outPtr + w' * (WP.mk b).outStride ∨
      outPtr + w' * (WP.mk b).outStride + (WP.mk b).lenOff + 4 ≤ j := by
  have hstride : (WP.mk b).outStride = (WP.mk b).lenOff + 8 := rfl
  rcases Nat.lt_or_ge w w' with h | h
  · have h1 : (w + 1) * (WP.mk b).outStride ≤ w' * (WP.mk b).outStride :=
      Nat.mul_le_mul_right _ h
    have he : (w + 1) * (WP.mk b).outStride = w * (WP.mk b).outStride + (WP.mk b).outStride :=
      Nat.succ_mul w _
    left; omega
  · have hlt : w' < w := by omega
    have h1 : (w' + 1) * (WP.mk b).outStride ≤ w * (WP.mk b).outStride :=
      Nat.mul_le_mul_right _ hlt
    have he : (w' + 1) * (WP.mk b).outStride = w' * (WP.mk b).outStride + (WP.mk b).outStride :=
      Nat.succ_mul w' _
    right; omega

def LaunchAgreesPerWarp (b : Nat) (inPtr outPtr : Nat) (gm : Array UInt8)
    (smemB : List UInt8) (gfinal : Array UInt8) : Prop :=
  ∀ w, w < (WP.mk b).numBlk → ∀ (ss' : AlgorithmLib.LZ4Simt.SState),
    (∃ n, AlgorithmLib.LZ4Simt.SReaches (WP.mk b).kernel n
            (AlgorithmLib.LZ4Simt.initSt w inPtr outPtr gm smemB) ss') →
    ss'.pc = 272 →
    ∀ j, outPtr + w * (WP.mk b).outStride ≤ j →
         j < outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff + 4 →
      gfinal.getD j 0 = ss'.gmem.getD j 0

/-- **Whole-launch correctness.**  Given the memory-model assumption, EVERY block
    decodes correctly out of the FINAL memory — not merely each warp in isolation.
    This is what the host's download relies on. -/
theorem launch_correct (b : Nat) (inPtr outPtr : Nat) (gm : Array UInt8)
    (smemB : List UInt8) (gfinal : Array UInt8)
    (hcorrect : ShippedCorrect b)
    (hlayout : LayoutOK b inPtr outPtr gm)
    (hSC : LaunchAgreesPerWarp b inPtr outPtr gm smemB gfinal) :
    ∀ w, w < (WP.mk b).numBlk →
      ∃ k, 0 < k ∧ k ≤ (WP.mk b).lenOff ∧
        AlgorithmLib.readU32LE gfinal
          (outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff) = k ∧
        AlgorithmLib.LZ4Imp.decompress
          ((List.range k).map (fun i => gfinal.getD
            (outPtr + w * (WP.mk b).outStride + i) 0))
          (WP.mk b).inStride
          = some (gmemInpAt gm (inPtr + w * (WP.mk b).inStride) (WP.mk b).inStride) := by
  intro w hw
  obtain ⟨hglob, hper⟩ := hlayout
  obtain ⟨l1, l2, l3, l5, l6⟩ := hper w hw
  -- per-warp input/output disjointness follows from the whole-buffer version
  have l4 : inPtr + w * (WP.mk b).inStride + (WP.mk b).inStride
      ≤ outPtr + w * (WP.mk b).outStride := by
    have h1 : (w + 1) * (WP.mk b).inStride ≤ (WP.mk b).numBlk * (WP.mk b).inStride :=
      Nat.mul_le_mul_right _ hw
    have h2 : (w + 1) * (WP.mk b).inStride
        = w * (WP.mk b).inStride + (WP.mk b).inStride := Nat.succ_mul w _
    omega
  obtain ⟨n, ss', k, hreach, hpc272, hk0, hkle, hlenf, hdec, _hconf⟩ :=
    hcorrect w inPtr outPtr gm smemB hw l1 l2 l3 l4 l5 l6
  have hagree := hSC w hw ss' ⟨n, hreach⟩ hpc272
  refine ⟨k, hk0, hkle, ?_, ?_⟩
  · rw [← hlenf]
    unfold AlgorithmLib.readU32LE
    rw [hagree _ (by omega) (by omega), hagree _ (by omega) (by omega),
      hagree _ (by omega) (by omega), hagree _ (by omega) (by omega)]
  · rw [← hdec]
    congr 1
    apply List.map_congr_left
    intro i hi
    rw [List.mem_range] at hi
    exact hagree _ (by omega) (by omega)

/-- info: 'Algorithm.launch_correct' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms launch_correct

end ShippedClaim

-- ── Seam guards: launch facts the roundtrip theorem cannot see ────────────────
section SeamGuards

/-- The corpus divides into whole blocks, so `inpAll.length = totIn` is coherent. -/
example : (WP.mk 15).totIn = corpusBytes := by decide
example : (WP.mk 16).totIn = corpusBytes := by decide

end SeamGuards

end Algorithm

def main (args : List String) : IO Unit := do
  let outDir ← AlgorithmLib.requireOutputDir args
  AlgorithmLib.emitArtifacts outDir #[
    Algorithm.warpArtifactDSL "lz4_comp_warpdsl" 15,     -- proven kernel, 32 KiB blocks
    Algorithm.warpArtifactDSL "lz4_comp_warpdsl64" 16]   -- proven kernel, 64 KiB blocks
