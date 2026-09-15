import Lz4Kernel
import AlgorithmLib.Gen
import AlgorithmLib.LZ4Suite
import AlgorithmLib.LZ4SimtSerialize
import AlgorithmLib.LZ4WarpKernel
import AlgorithmLib.LZ4CompTop
import ShipScan


open Lean (Json)
open AlgorithmLib
open AlgorithmLib.PTX

namespace Algorithm

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

-- The host program as a *builder*, so `Clif`'s scanners can read the blocks it
-- emits.  `warpClif` is this printed; the two cannot drift because there is only
-- one of them.
-- Every store offset below is derived from `bo`, the binding-table offset, which
-- at the call site is `WP.bindOff` — a value that can only be computed by
-- serializing the whole PTX kernel.  Holding it as a PARAMETER makes that
-- dependency explicit and was an attempt to let `Lz4Host`'s recovery theorems
-- reduce symbolically; measured, it does not (reducing the builder's state
-- monad is itself the cost), so those theorems use `native_decide`.  Kept
-- because the separation is worth stating.  `warpCode` instantiates it at
-- `w.bindOff`, so the program that ships is byte-identical.
open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.Prog

/-- The host program, as a term.

    `bo` is the binding-table offset, which can only be computed by serializing
    the whole PTX kernel; holding it as a parameter keeps that dependency
    explicit. `warpCode` instantiates it at `w.bindOff`, and that instance is
    what ships. -/
def warpCodeAt (w : WP) (bo : Nat) : Prog V L Unit :=
  do
    let ptr ← basePtr
    let dataPtr ← load64 (← absAddr ptr 0x18)
    let dataLen ← load64 (← absAddr ptr 0x20)
    let outPtr ← load64 (← absAddr ptr 0x28)
    cudaInit ptr
    -- ONE allocation: input at offset 0, output immediately after it.  Both
    -- kernel parameters are bound to this buffer and the kernel derives its
    -- output base as `in_ptr + totIn`, so the placement contract `LayoutOK`
    -- asks for holds by construction instead of relating two independent
    -- allocations.
    let inBuf ← cudaCreateBuffer ptr (← iconst64 (w.outOff + w.outAlloc))
    let _ ← cudaUploadRawOffset ptr inBuf (← iconst64 0) dataPtr dataLen
    let g ← iconst32 w.gridX
    let bk ← iconst32 wBlockDim
    let one32 ← iconst32 1
    let nbufs ← ireduce32 (← iconst64 2)
    let ptxOff ← iconst64 rPTX_OFF
    let bindOff ← iconst64 bo
    let _ ← forLoopAcc (← iconst64 rLaunches) (← iconst64 0) (fun _ acc => do
      let _ ← cudaLaunch ptr ptxOff nbufs bindOff g one32 one32 bk one32 one32
      pure acc)
    let _ ← cudaDownloadRawOffset ptr inBuf (← iconst64 w.outOff) outPtr
              (← iconst64 w.totOut)
    cudaCleanup ptr
    storeAt ptr (bo + 0x40) (← iconst64 1)
    storeAt ptr (bo + 0x48) (← iconst64 1)
    storeAt ptr (bo + 0x50) (← iconst64 rLaunches)
    storeAt ptr (bo + 0x58) (← iconst64 w.totIn)
    storeAt ptr (bo + 0x60) (← iconst64 w.inStride)
    storeAt ptr (bo + 0x68) (← iconst64 w.outStride)
    storeAt ptr (bo + 0x70) (← iconst64 w.lenOff)
    storeAt ptr (bo + 0x78) (← iconst64 w.numBlk)

def warpCode (w : WP) : Prog V L Unit := warpCodeAt w w.bindOff

/-- The emitted function, which every host theorem is now stated over. -/
def warpFn (w : WP) : FuncData :=
  Prog.stateOf 1 (warpCode w)

def warpClif (w : WP) : Except String Program :=
  Prog.program [.ok noopFunction, Prog.compileProg 1 (warpCode w)]

def warpPayloadDSL (w : WP) : List UInt8 :=
  zeros rPTX_OFF ++
  (w.ptxBytes ++ zeros (w.bindOff - rPTX_OFF - w.ptxLen)) ++
  (uint32ToBytes 0 ++ uint32ToBytes 0)

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

open AlgorithmLib.IR AlgorithmLib.HProg in
def warpArtifactDSL (name : String) (blkLog : Nat) : Except String Lean.Json := do
  let w : WP := ⟨blkLog⟩
  return AlgorithmLib.toJsonArtifact name
    { clif := ← warpClif w,
      memory_size := w.memSize,
      initial_memory := warpPayloadDSL w }
    { fn_idx := AlgorithmLib.IR.mainFnIdx, output := compSchema w }

end Algorithm

def main (args : List String) : IO Unit := do
  let outDir ← AlgorithmLib.requireOutputDir args
  AlgorithmLib.emitArtifacts outDir #[
    ← AlgorithmLib.Prog.orDie (Algorithm.warpArtifactDSL "lz4_comp_warpdsl" 15),
    ← AlgorithmLib.Prog.orDie (Algorithm.warpArtifactDSL "lz4_comp_warpdsl64" 16)]

#eval ShipScan.check "Lz4CompAlgorithm"
