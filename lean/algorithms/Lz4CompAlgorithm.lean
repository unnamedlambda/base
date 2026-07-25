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
def rBIND_OFF  : Nat := 0x8100
def rROW_OFF   : Nat := 0x8140
def rPASS_OFF  : Nat := rROW_OFF + 0x08
def rLAUN_OFF  : Nat := rROW_OFF + 0x10
def rBYTES_OFF : Nat := rROW_OFF + 0x18
def rINSTR_OFF : Nat := rROW_OFF + 0x20
def rOUTSTR_OFF: Nat := rROW_OFF + 0x28
def rLENOFF_OFF: Nat := rROW_OFF + 0x30
def rNUMBLK_OFF: Nat := rROW_OFF + 0x38
def rMEM_SIZE  : Nat := 0x9000

def compSchema : List Json :=
  [Output.schema
    [ Output.column "pass" .i64 rPASS_OFF,
      Output.column "launches" .i64 rLAUN_OFF,
      Output.column "bytes_per_launch" .i64 rBYTES_OFF,
      Output.column "in_stride" .i64 rINSTR_OFF,
      Output.column "out_stride" .i64 rOUTSTR_OFF,
      Output.column "len_off" .i64 rLENOFF_OFF,
      Output.column "num_blk" .i64 rNUMBLK_OFF ]
    rROW_OFF]

-- ── Warp-cooperative compressor: 1 warp/block, 4096-entry shared-mem hash table.
-- Parameterized by block-size log2 (pos+1 must fit u16: blkLog ≤ 16). --
structure WP where
  blkLog : Nat
def wHashLog     : Nat := 12                  -- 4096-entry shared table (8 KiB/warp)
def wWarpsPerCTA : Nat := 4
def WP.inStride  (w : WP) : Nat := 2 ^ w.blkLog
def WP.numBlk    (w : WP) : Nat := corpusBytes / w.inStride
def WP.lenOff    (w : WP) : Nat := w.inStride + w.inStride / 16 + 256
def WP.outStride (w : WP) : Nat := w.lenOff + 8  -- compressed | u32 length (no global table)
def wTableBytes  : Nat := (2 ^ wHashLog) * 2
def wSmem        : Nat := wWarpsPerCTA * wTableBytes
def wBlockDim    : Nat := wWarpsPerCTA * 32
def WP.gridX     (w : WP) : Nat := (w.numBlk + wWarpsPerCTA - 1) / wWarpsPerCTA
def WP.totIn     (w : WP) : Nat := w.numBlk * w.inStride
def WP.totOut    (w : WP) : Nat := w.numBlk * w.outStride

open AlgorithmLib.IR in
def warpClif (w : WP) : String := buildProgram do
  let cuda ← declareCudaFFI
  let ptr ← entryBlock
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let dataLen ← load64 (← absAddr ptr 0x20)
  let outPtr ← load64 (← absAddr ptr 0x28)
  cudaInit cuda ptr
  let inBuf ← cudaCreateBuffer cuda ptr dataLen
  let outBuf ← cudaCreateBuffer cuda ptr (← iconst64 w.totOut)
  let _ ← cudaUploadRaw cuda ptr inBuf dataPtr dataLen
  let g ← iconst32 w.gridX
  let bk ← iconst32 wBlockDim
  let one32 ← iconst32 1
  let nbufs ← ireduce32 (← iconst64 2)
  let ptxOff ← iconst64 rPTX_OFF
  let bindOff ← iconst64 rBIND_OFF
  let _ ← forLoopAcc .i64 .i64 (← iconst64 rLaunches) (← iconst64 0) (fun _ acc => do
    let _ ← cudaLaunch cuda ptr ptxOff nbufs bindOff g one32 one32 bk one32 one32
    pure acc)
  let _ ← cudaDownloadRaw cuda ptr outBuf outPtr (← iconst64 w.totOut)
  cudaCleanup cuda ptr
  storeAt ptr rROW_OFF (← iconst64 1)
  storeAt ptr rPASS_OFF (← iconst64 1)
  storeAt ptr rLAUN_OFF (← iconst64 rLaunches)
  storeAt ptr rBYTES_OFF (← iconst64 w.totIn)
  storeAt ptr rINSTR_OFF (← iconst64 w.inStride)
  storeAt ptr rOUTSTR_OFF (← iconst64 w.outStride)
  storeAt ptr rLENOFF_OFF (← iconst64 w.lenOff)
  storeAt ptr rNUMBLK_OFF (← iconst64 w.numBlk)
  ret

-- The shipped kernel = the serialized PROOF OBJECT `warpKernelDSL`: what runs on
-- the GPU is exactly the program `warpKernelDSL_prologue_roundtrips` reasons over
-- (option-A all-b64 serialization; see LZ4SimtSerialize).
def warpKernelDSLStr (w : WP) : String :=
  AlgorithmLib.LZ4Simt.serializeKernel
    (AlgorithmLib.LZ4WarpDSL.warpKernelDSL w.numBlk w.inStride w.outStride w.lenOff wHashLog)
    wSmem

def warpPayloadDSL (w : WP) : List UInt8 :=
  let ptxBytes := (warpKernelDSLStr w).toUTF8.toList ++ [0]
  let bind := uint32ToBytes 0 ++ uint32ToBytes 1
  zeros rPTX_OFF ++
  (ptxBytes ++ zeros (rBIND_OFF - rPTX_OFF - ptxBytes.length)) ++ bind

def warpArtifactDSL (name : String) (blkLog : Nat) :=
  let w : WP := ⟨blkLog⟩
  AlgorithmLib.toJsonArtifact name
    { cranelift_ir := warpClif w,
      memory_size := rMEM_SIZE,
      initial_memory := warpPayloadDSL w }
    { fn_idx := AlgorithmLib.IR.mainFnIdx, output := compSchema }

-- Kernel-checked instantiation guards: the roundtrip theorems cover the EXACT
-- parameters of both shipped artifacts, for EVERY warp `w < numBlk` of the launch
-- and ARBITRARY launch memory (`outB` is whatever the previous launch left;
-- `smemB` is uninitialized shared memory).
-- If theorem and shipped geometry ever drift apart, these fail to elaborate.
section Guards
open AlgorithmLib.LZ4WarpDSL

private def guardBody (b : Nat) : Prop :=
  ∀ (w : Nat) (inpAll outB smemB : List UInt8),
    inpAll.length = (WP.mk b).numBlk * (WP.mk b).inStride →
    w < (WP.mk b).numBlk →
    w * 32 + 32 < 2 ^ 64 →
    w * (WP.mk b).inStride < 2 ^ 40 →
    (WP.mk b).numBlk * (WP.mk b).inStride + w * (WP.mk b).outStride
      + 9 * (WP.mk b).inStride < 2 ^ 32 →
    w * (WP.mk b).outStride + 9 * (WP.mk b).inStride ≤ outB.length →
    ∃ (n : Nat) (ss' : AlgorithmLib.LZ4Simt.SState) (k : Nat),
      AlgorithmLib.LZ4Simt.SReaches
        (AlgorithmLib.LZ4WarpDSL.warpKernelDSL (WP.mk b).numBlk (WP.mk b).inStride
          (WP.mk b).outStride (WP.mk b).lenOff wHashLog) n
        (AlgorithmLib.LZ4Simt.initSt w inpAll outB smemB) ss' ∧
      AlgorithmLib.LZ4Imp.decompress
        ((List.range k).map (fun i => ss'.gmem.getD
          ((WP.mk b).numBlk * (WP.mk b).inStride + w * (WP.mk b).outStride + i) 0))
        (WP.mk b).inStride
        = some (blockAt inpAll w (WP.mk b).inStride)

example : guardBody 15 := fun w inpAll outB smemB hiT hw hw64 hib40 htop hbuf =>
  warpKernelDSL_prologue_roundtrips
    (WP.mk 15).numBlk (WP.mk 15).inStride (WP.mk 15).outStride (WP.mk 15).lenOff wHashLog
    w ((WP.mk 15).numBlk * (WP.mk 15).inStride) inpAll outB smemB
    hiT (Nat.le_refl _) (by decide) (by decide) hw (by decide) (by decide) (by decide)
    hw64 hib40 htop hbuf

example : guardBody 16 := fun w inpAll outB smemB hiT hw hw64 hib40 htop hbuf =>
  warpKernelDSL_prologue_roundtrips
    (WP.mk 16).numBlk (WP.mk 16).inStride (WP.mk 16).outStride (WP.mk 16).lenOff wHashLog
    w ((WP.mk 16).numBlk * (WP.mk 16).inStride) inpAll outB smemB
    hiT (Nat.le_refl _) (by decide) (by decide) hw (by decide) (by decide) (by decide)
    hw64 hib40 htop hbuf

-- The whole-kernel theorem (through the length-store tail): terminal state,
-- `clen ≤ lenOff`, the stored u32 IS `clen`, and the `clen`-prefix decodes back.
private def guardTail (b : Nat) : Prop :=
  ∀ (w : Nat) (inpAll outB smemB : List UInt8),
    inpAll.length = (WP.mk b).numBlk * (WP.mk b).inStride →
    w < (WP.mk b).numBlk →
    w * 32 + 32 < 2 ^ 64 →
    w * (WP.mk b).inStride < 2 ^ 40 →
    (WP.mk b).numBlk * (WP.mk b).inStride + w * (WP.mk b).outStride
      + 9 * (WP.mk b).inStride < 2 ^ 32 →
    w * (WP.mk b).outStride + 9 * (WP.mk b).inStride ≤ outB.length →
    (WP.mk b).numBlk * (WP.mk b).inStride + w * (WP.mk b).outStride
      + (WP.mk b).lenOff + 3 < 2 ^ 64 →
    w * (WP.mk b).outStride + (WP.mk b).lenOff + 4 ≤ outB.length →
    ∃ (n : Nat) (ss' : AlgorithmLib.LZ4Simt.SState) (k : Nat),
      AlgorithmLib.LZ4Simt.SReaches
        (AlgorithmLib.LZ4WarpDSL.warpKernelDSL (WP.mk b).numBlk (WP.mk b).inStride
          (WP.mk b).outStride (WP.mk b).lenOff wHashLog) n
        (AlgorithmLib.LZ4Simt.initSt w inpAll outB smemB) ss' ∧
      ss'.pc = 272 ∧ k ≤ (WP.mk b).lenOff ∧
      AlgorithmLib.readU32LE ss'.gmem
        ((WP.mk b).numBlk * (WP.mk b).inStride + w * (WP.mk b).outStride + (WP.mk b).lenOff) = k ∧
      AlgorithmLib.LZ4Imp.decompress
        ((List.range k).map (fun i => ss'.gmem.getD
          ((WP.mk b).numBlk * (WP.mk b).inStride + w * (WP.mk b).outStride + i) 0))
        (WP.mk b).inStride
        = some (blockAt inpAll w (WP.mk b).inStride)

example : guardTail 15 :=
  fun w inpAll outB smemB hiT hw hw64 hib40 htop hbuf hlOtop hlOfit =>
  warpKernelDSL_tail_roundtrips
    (WP.mk 15).numBlk (WP.mk 15).inStride (WP.mk 15).outStride (WP.mk 15).lenOff wHashLog
    w ((WP.mk 15).numBlk * (WP.mk 15).inStride) inpAll outB smemB
    hiT (Nat.le_refl _) (by decide) (by decide) hw (by decide) (by decide) (by decide)
    hw64 hib40 htop hbuf (Nat.le_refl _) hlOtop hlOfit

example : guardTail 16 :=
  fun w inpAll outB smemB hiT hw hw64 hib40 htop hbuf hlOtop hlOfit =>
  warpKernelDSL_tail_roundtrips
    (WP.mk 16).numBlk (WP.mk 16).inStride (WP.mk 16).outStride (WP.mk 16).lenOff wHashLog
    w ((WP.mk 16).numBlk * (WP.mk 16).inStride) inpAll outB smemB
    hiT (Nat.le_refl _) (by decide) (by decide) hw (by decide) (by decide) (by decide)
    hw64 hib40 htop hbuf (Nat.le_refl _) hlOtop hlOfit

end Guards

end Algorithm

def main (args : List String) : IO Unit := do
  let outDir ← AlgorithmLib.requireOutputDir args
  AlgorithmLib.emitArtifacts outDir #[
    Algorithm.warpArtifactDSL "lz4_comp_warpdsl" 15,     -- proven kernel, 32 KiB blocks
    Algorithm.warpArtifactDSL "lz4_comp_warpdsl64" 16]   -- proven kernel, 64 KiB blocks
