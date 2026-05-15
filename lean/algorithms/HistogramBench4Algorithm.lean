import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace HistogramBench4

/-
  Multi-threaded histogram: 4 workers, each with its own 4KB-aligned histogram.
  Payload: "input_path\0output_path\0"
  Orchestrator (fn2) spawns 4 workers (fn3), joins, merges, writes.
  Worker (fn3) receives 48-byte descriptor, runs 4x-unrolled scan.
-/

def INPUT_PATH_OFF  : Nat := 0x0100
def OUTPUT_PATH_OFF : Nat := 0x0200
def THREAD_CTX_OFF  : Nat := 0x0300
def HIST_REGION_OFF : Nat := 0x0400
def BINS            : Nat := 256
def WORKERS         : Nat := 4
def HIST_STRIDE     : Nat := 4096
def RESULT_OFF      : Nat := HIST_REGION_OFF + WORKERS * HIST_STRIDE  -- 0x4400
def RESULT_SIZE     : Nat := BINS * 8
def HANDLES_OFF     : Nat := 19456
def DESCS_OFF       : Nat := 19520
def DESC_SIZE       : Nat := 48
def DATA_OFF        : Nat := 19712
def MAX_DATA_BYTES  : Nat := 64 * 1024 * 1024
def MEM_SIZE        : Nat := DATA_OFF + MAX_DATA_BYTES

set_option maxRecDepth 2048 in
def orchFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let fnThInit    ← declareFFI "cl_thread_init"    [.i64]           none
  let fnThSpawn   ← declareFFI "cl_thread_spawn"   [.i64, .i64, .i64] (some .i64)
  let fnThJoin    ← declareFFI "cl_thread_join"    [.i64, .i64]     (some .i64)
  let fnThCleanup ← declareFFI "cl_thread_cleanup" [.i64]           none
  let fnRead      ← declareFileRead
  let fnWrite     ← declareFileWrite
  let dataPtr     ← load64 (← absAddr ptr 0x18)
  let zero        ← iconst64 0

  let cpIn     ← declareBlock [.i64]
  let cpOut1   ← declareBlock [.i64]
  let cpOut    ← declareBlock [.i64, .i64]
  let readBlk  ← declareBlock []
  let spawnLp  ← declareBlock [.i64, .i64, .i64]  -- (worker, n, chunk)
  let joinLp   ← declareBlock [.i64]              -- (worker)
  let mergeBin ← declareBlock [.i64]              -- (bin)
  let mergeWk  ← declareBlock [.i64, .i64, .i64] -- (wk, sum, bin)
  let storeBin ← declareBlock [.i64, .i64]        -- (sum, bin)
  let writeOut ← declareBlock []

  jump cpIn.ref [zero]

  startBlock cpIn
  let si1 := cpIn.param 0
  let ch1  ← uload8_64 (← iadd dataPtr si1)
  istore8 ch1 (← iadd (← absAddr ptr INPUT_PATH_OFF) si1)
  let si1' ← iaddImm si1 1
  brif (← icmpImm .eq ch1 0) cpOut1.ref [si1'] cpIn.ref [si1']

  startBlock cpOut1
  jump cpOut.ref [cpOut1.param 0, zero]

  startBlock cpOut
  let si3 := cpOut.param 0; let di3 := cpOut.param 1
  let ch3  ← uload8_64 (← iadd dataPtr si3)
  istore8 ch3 (← iadd (← absAddr ptr OUTPUT_PATH_OFF) di3)
  let si3' ← iaddImm si3 1; let di3' ← iaddImm di3 1
  brif (← icmpImm .eq ch3 0) readBlk.ref [] cpOut.ref [si3', di3']

  startBlock readBlk
  let fileSize ← readFile ptr fnRead INPUT_PATH_OFF DATA_OFF
  let n        ← ushrImm fileSize 2          -- n = bytes / 4
  let nPlus    ← iaddImm n 3
  let workers4 ← iconst64 WORKERS
  let chunk    ← udiv nPlus workers4
  let ctxSlot  ← absAddr ptr THREAD_CTX_OFF
  callVoid fnThInit [ctxSlot]
  let ctxPtr   ← load64 ctxSlot
  let fnIdx    ← iconst64 3   -- worker function index
  jump spawnLp.ref [zero, n, chunk]

  -- Spawn loop
  startBlock spawnLp
  let wk := spawnLp.param 0; let nn := spawnLp.param 1; let ch := spawnLp.param 2
  let descOff  ← iadd (← absAddr ptr DESCS_OFF) (← imul wk (← iconst64 DESC_SIZE))
  -- Fill descriptor
  storeI64 ptr descOff
  storeI64 (← iconst64 DATA_OFF) (← iaddImm descOff 8)
  let start ← imul wk ch
  storeI64 start (← iaddImm descOff 16)
  let rem   ← isub nn start
  let cnt   ← select' (← icmp .ult rem ch) rem ch
  storeI64 cnt (← iaddImm descOff 24)
  let histOff ← iaddImm (← imul wk (← iconst64 HIST_STRIDE)) HIST_REGION_OFF
  storeI64 histOff (← iaddImm descOff 32)
  storeI64 (← iconst64 BINS) (← iaddImm descOff 40)
  -- Spawn and store handle
  let handle ← call fnThSpawn [ctxPtr, fnIdx, descOff]
  let hAddr  ← iadd (← absAddr ptr HANDLES_OFF) (← ishlImm wk 3)
  storeI64 handle hAddr
  let wk' ← iaddImm wk 1
  brif (← icmp .ult wk' workers4) spawnLp.ref [wk', nn, ch] joinLp.ref [zero]

  -- Join loop
  startBlock joinLp
  let wk2 := joinLp.param 0
  let hAddr2 ← iadd (← absAddr ptr HANDLES_OFF) (← ishlImm wk2 3)
  let handle2 ← load64 hAddr2
  let _ ← call fnThJoin [ctxPtr, handle2]
  let wk2' ← iaddImm wk2 1
  brif (← icmp .ult wk2' workers4) joinLp.ref [wk2'] mergeBin.ref [zero]

  -- Merge bins outer loop
  startBlock mergeBin
  let bin := mergeBin.param 0
  jump mergeWk.ref [zero, zero, bin]

  startBlock mergeWk
  let mwk := mergeWk.param 0; let msum := mergeWk.param 1; let mbin := mergeWk.param 2
  let histBase ← iaddImm (← imul mwk (← iconst64 HIST_STRIDE)) HIST_REGION_OFF
  let binAddr  ← iadd ptr (← iadd histBase (← ishlImm mbin 3))
  let cnt2     ← load64 binAddr
  let msum'    ← iadd msum cnt2
  let mwk'     ← iaddImm mwk 1
  brif (← icmp .ult mwk' workers4) mergeWk.ref [mwk', msum', mbin]
                                     storeBin.ref [msum', mbin]

  startBlock storeBin
  let msum2 := storeBin.param 0; let mbin2 := storeBin.param 1
  let resAddr ← iadd ptr (← iaddImm (← ishlImm mbin2 3) RESULT_OFF)
  storeI64 msum2 resAddr
  let mbin2' ← iaddImm mbin2 1
  brif (← icmp .ult mbin2' (← iconst64 BINS)) mergeBin.ref [mbin2'] writeOut.ref []

  startBlock writeOut
  let _ ← call fnWrite [ptr, ← iconst64 OUTPUT_PATH_OFF, ← iconst64 RESULT_OFF,
                         zero, ← iconst64 RESULT_SIZE]
  callVoid fnThCleanup [← absAddr ptr THREAD_CTX_OFF]
  ret

def workerFn : IRBuilder Unit := do
  let desc ← entryBlock
  let zero ← iconst64 0

  -- Load descriptor fields
  let base      ← load64 desc
  let dataOff   ← load64 (← iaddImm desc 8)
  let dataStart ← load64 (← iaddImm desc 16)
  let dataCnt   ← load64 (← iaddImm desc 24)
  let histOff   ← load64 (← iaddImm desc 32)
  let bins      ← load64 (← iaddImm desc 40)

  let dataBase  ← iadd base dataOff
  let dataPtr2  ← iadd dataBase (← ishlImm dataStart 2)
  let histPtr   ← iadd base histOff
  let histEnd   ← iadd histPtr (← ishlImm bins 3)
  let dataEnd   ← iadd dataPtr2 (← ishlImm dataCnt 2)
  let cnt4      ← bandImm dataCnt (-4)
  let dataEnd4  ← iadd dataPtr2 (← ishlImm cnt4 2)

  let zeroLp  ← declareBlock [.i64]
  let scan4L  ← declareBlock [.i64]
  let scan4B  ← declareBlock [.i64]
  let scalarL ← declareBlock [.i64]
  let scalarB ← declareBlock [.i64]
  let done    ← declareBlock []

  jump zeroLp.ref [histPtr]

  startBlock zeroLp
  let hp := zeroLp.param 0
  storeI64 zero hp
  storeI64 zero (← iaddImm hp 8)
  storeI64 zero (← iaddImm hp 16)
  storeI64 zero (← iaddImm hp 24)
  storeI64 zero (← iaddImm hp 32)
  storeI64 zero (← iaddImm hp 40)
  storeI64 zero (← iaddImm hp 48)
  storeI64 zero (← iaddImm hp 56)
  let hp' ← iaddImm hp 64
  brif (← icmp .ult hp' histEnd) zeroLp.ref [hp'] scan4L.ref [dataPtr2]

  startBlock scan4L
  let dp := scan4L.param 0
  brif (← icmp .ult dp dataEnd4) scan4B.ref [dp] scalarL.ref [dataEnd4]

  startBlock scan4B
  let dp2 := scan4B.param 0
  let v0  ← uload32_64 dp2
  let a0  ← iadd histPtr (← ishlImm v0 3)
  let c0  ← load64 a0; storeI64 (← iaddImm c0 1) a0
  let v1  ← uload32_64 (← iaddImm dp2 4)
  let a1  ← iadd histPtr (← ishlImm v1 3)
  let c1  ← load64 a1; storeI64 (← iaddImm c1 1) a1
  let v2  ← uload32_64 (← iaddImm dp2 8)
  let a2  ← iadd histPtr (← ishlImm v2 3)
  let c2  ← load64 a2; storeI64 (← iaddImm c2 1) a2
  let v3  ← uload32_64 (← iaddImm dp2 12)
  let a3  ← iadd histPtr (← ishlImm v3 3)
  let c3  ← load64 a3; storeI64 (← iaddImm c3 1) a3
  jump scan4L.ref [← iaddImm dp2 16]

  startBlock scalarL
  let dp3 := scalarL.param 0
  brif (← icmp .ult dp3 dataEnd) scalarB.ref [dp3] done.ref []

  startBlock scalarB
  let dp4 := scalarB.param 0
  let vS  ← uload32_64 dp4
  let aS  ← iadd histPtr (← ishlImm vS 3)
  let cS  ← load64 aS; storeI64 (← iaddImm cS 1) aS
  jump scalarL.ref [← iaddImm dp4 4]

  startBlock done
  ret

def clifIR : String :=
  noopFunction ++ "\n" ++ noopAt 1 ++ "\n" ++
  buildFunction 2 orchFn ++ "\n" ++ buildFunction 3 workerFn

def artifacts : Array Json :=
  #[toJsonEntry "hist4_algorithm" {
    cranelift_ir := clifIR,
    memory_size := MEM_SIZE,
    context_offset := 0
  } {
    fn_idx := u32 2
  }]

end HistogramBench4
