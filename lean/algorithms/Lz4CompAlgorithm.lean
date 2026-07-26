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

/-- The warp-dependent side conditions, from `w < numBlk` plus three numeric facts.
    Symbolic in `w`, so it covers the last warp as well as the first. -/
private theorem guard_sidecond (nb iS oS lO iT w : Nat)
    (hw : w < nb) (hoS : oS = lO + 8)
    (hN1 : nb * iS < 2 ^ 40)
    (hN2 : iT + nb * oS + 9 * iS < 2 ^ 32)
    (hN4 : nb * 32 + 32 < 2 ^ 64) :
    w * 32 + 32 < 2 ^ 64
    ∧ w * iS < 2 ^ 40
    ∧ iT + w * oS + 9 * iS < 2 ^ 32
    ∧ w * oS + 9 * iS ≤ nb * oS + 9 * iS
    ∧ iT + w * oS + lO + 3 < 2 ^ 64
    ∧ w * oS + lO + 4 ≤ nb * oS + 9 * iS := by
  have h1 : w * 32 ≤ nb * 32 := Nat.mul_le_mul_right 32 (Nat.le_of_lt hw)
  have h2 : w * iS ≤ nb * iS := Nat.mul_le_mul_right iS (Nat.le_of_lt hw)
  have h3 : w * oS ≤ nb * oS := Nat.mul_le_mul_right oS (Nat.le_of_lt hw)
  have h4 : (w + 1) * oS ≤ nb * oS := Nat.mul_le_mul_right oS hw
  have h5 : (w + 1) * oS = w * oS + oS := Nat.succ_mul w oS
  refine ⟨by omega, by omega, by omega, by omega, by omega, by omega⟩

/-- Whole-kernel guarantee for warp `w`, assuming only the host's allocation. -/
def ShippedCorrect (b : Nat) : Prop :=
  ∀ (w : Nat) (inpAll outB smemB : List UInt8),
    inpAll.length = (WP.mk b).totIn →
    w < (WP.mk b).numBlk →
    outB.length = (WP.mk b).outAlloc →
    ∃ (n : Nat) (ss' : AlgorithmLib.LZ4Simt.SState) (k : Nat),
      AlgorithmLib.LZ4Simt.SReaches
        (WP.mk b).kernel n
        (AlgorithmLib.LZ4Simt.initSt w inpAll outB smemB) ss' ∧
      ss'.pc = 272 ∧ 0 < k ∧ k ≤ (WP.mk b).lenOff ∧
      AlgorithmLib.readU32LE ss'.gmem
        ((WP.mk b).totIn + w * (WP.mk b).outStride + (WP.mk b).lenOff) = k ∧
      AlgorithmLib.LZ4Imp.decompress
        ((List.range k).map (fun i => ss'.gmem.getD
          ((WP.mk b).totIn + w * (WP.mk b).outStride + i) 0))
        (WP.mk b).inStride
        = some (blockAt inpAll w (WP.mk b).inStride)

theorem shipped32_correct : ShippedCorrect 15 := by
  intro w inpAll outB smemB hiT hw hout
  obtain ⟨a1, a2, a3, a4, a5, a6⟩ :=
    guard_sidecond (WP.mk 15).numBlk (WP.mk 15).inStride (WP.mk 15).outStride
      (WP.mk 15).lenOff (WP.mk 15).totIn w hw rfl (by decide) (by decide) (by decide)
  exact warpKernelDSL_tail_roundtrips
    (WP.mk 15).numBlk (WP.mk 15).inStride (WP.mk 15).outStride (WP.mk 15).lenOff wHashLog
    w (WP.mk 15).totIn inpAll outB smemB
    hiT (Nat.le_refl _) (by decide) (by decide) hw (by decide) (by decide) (by decide)
    a1 a2 a3 (by rw [hout]; exact a4) (Nat.le_refl _) (by rw [hout] at *; exact a5)
    (by rw [hout]; exact a6)

theorem shipped64_correct : ShippedCorrect 16 := by
  intro w inpAll outB smemB hiT hw hout
  obtain ⟨a1, a2, a3, a4, a5, a6⟩ :=
    guard_sidecond (WP.mk 16).numBlk (WP.mk 16).inStride (WP.mk 16).outStride
      (WP.mk 16).lenOff (WP.mk 16).totIn w hw rfl (by decide) (by decide) (by decide)
  exact warpKernelDSL_tail_roundtrips
    (WP.mk 16).numBlk (WP.mk 16).inStride (WP.mk 16).outStride (WP.mk 16).lenOff wHashLog
    w (WP.mk 16).totIn inpAll outB smemB
    hiT (Nat.le_refl _) (by decide) (by decide) hw (by decide) (by decide) (by decide)
    a1 a2 a3 (by rw [hout]; exact a4) (Nat.le_refl _) (by rw [hout] at *; exact a5)
    (by rw [hout]; exact a6)

/-- info: 'Algorithm.shipped32_correct' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms shipped32_correct

/-- info: 'Algorithm.shipped64_correct' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms shipped64_correct

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
