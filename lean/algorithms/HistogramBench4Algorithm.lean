import Lean
import AlgorithmLib.Gen
import LayoutScan

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg

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

open AlgorithmLib.Prog


abbrev fnThInit : Ffi := .threadInit
abbrev fnThSpawn : Ffi := .threadSpawn
abbrev fnThJoin : Ffi := .threadJoin
abbrev fnThCleanup : Ffi := .threadCleanup
abbrev fnRead : Ffi := .fileRead
abbrev fnWrite : Ffi := .fileWrite

/-- The orchestrator: copy both paths, read, spawn `WORKERS`, join them, and
    merge their per-worker histograms bin by bin. -/
def orchCode : Prog V L Unit := do
  let ptr ← basePtr
  let dataPtr ← dataPtr
  let zero    ← iconst64 0

  let inEnd ← dwloop %[zero] .eq zero (contOnTrue := false) [0]
    (body := fun c => do
      let si := c.head
      let ch ← uload8_64 (← iadd dataPtr si)
      istore8 ch (← iadd (← absAddr ptr INPUT_PATH_OFF) si)
      let si' ← iaddImm si 1
      return (ch, %[si']))

  let _ ← dwloop %[inEnd.head, zero] .eq zero (contOnTrue := false) []
    (body := fun c => do
      let si := c.head; let di := c.snd
      let ch ← uload8_64 (← iadd dataPtr si)
      istore8 ch (← iadd (← absAddr ptr OUTPUT_PATH_OFF) di)
      let si' ← iaddImm si 1
      let di' ← iaddImm di 1
      return (ch, %[si', di']))

  let fileSize ← ffi fnRead
    %[ptr, ← iconst64 INPUT_PATH_OFF, ← iconst64 DATA_OFF, zero, zero]
  let n        ← ushrImm fileSize 2          -- n = bytes / 4
  let nPlus    ← iaddImm n 3
  let workers4 ← iconst64 WORKERS
  let chunk    ← udiv nPlus workers4
  let ctxSlot  ← absAddr ptr THREAD_CTX_OFF
  ffiVoid fnThInit %[ctxSlot]
  let ctxPtr   ← load64 ctxSlot
  let fnIdx    ← iconst64 3   -- worker function index

  -- Spawn. `n` and `chunk` do not change, so they stay in scope.
  let _ ← dwloop %[zero] .ult workers4 (contOnTrue := true) []
    (body := fun c => do
      let wk := c.head
      let descOff  ← iadd (← absAddr ptr DESCS_OFF) (← imul wk (← iconst64 DESC_SIZE))
      store ptr descOff
      store (← iconst64 DATA_OFF) (← iaddImm descOff 8)
      let start ← imul wk chunk
      store start (← iaddImm descOff 16)
      let rem   ← isub n start
      let cnt   ← select (← icmp .ult rem chunk) rem chunk
      store cnt (← iaddImm descOff 24)
      let histOff ← iaddImm (← imul wk (← iconst64 HIST_STRIDE)) HIST_REGION_OFF
      store histOff (← iaddImm descOff 32)
      store (← iconst64 BINS) (← iaddImm descOff 40)
      let handle ← ffi fnThSpawn %[ctxPtr, fnIdx, descOff]
      let hAddr  ← iadd (← absAddr ptr HANDLES_OFF) (← ishlImm wk 3)
      store handle hAddr
      let wk' ← iaddImm wk 1
      return (wk', %[wk']))

  -- Join
  let _ ← dwloop %[zero] .ult workers4 (contOnTrue := true) []
    (body := fun c => do
      let wk := c.head
      let hAddr ← iadd (← absAddr ptr HANDLES_OFF) (← ishlImm wk 3)
      let handle ← load64 hAddr
      let _ ← ffi fnThJoin %[ctxPtr, handle]
      let wk' ← iaddImm wk 1
      return (wk', %[wk']))

  -- Merge: one bin a trip on the outside, one worker a trip within
  let _ ← dwloop %[zero] .ult (← iconst64 BINS) (contOnTrue := true) []
    (body := fun cb => do
      let mbin := cb.head
      let acc ← dwloop %[zero, zero] .ult workers4 (contOnTrue := true) [1]
        (body := fun c => do
          let mwk := c.head; let msum := c.snd
          let histBase ← iaddImm (← imul mwk (← iconst64 HIST_STRIDE)) HIST_REGION_OFF
          let binAddr  ← iadd ptr (← iadd histBase (← ishlImm mbin 3))
          let cnt2     ← load64 binAddr
          let msum'    ← iadd msum cnt2
          let mwk'     ← iaddImm mwk 1
          return (mwk', %[mwk', msum']))
      let resAddr ← iadd ptr (← iaddImm (← ishlImm mbin 3) RESULT_OFF)
      store (acc.head) resAddr
      let mbin' ← iaddImm mbin 1
      return (mbin', %[mbin']))

  let _ ← ffi fnWrite %[ptr, ← iconst64 OUTPUT_PATH_OFF, ← iconst64 RESULT_OFF,
                        zero, ← iconst64 RESULT_SIZE]
  ffiVoid fnThCleanup %[← absAddr ptr THREAD_CTX_OFF]

/-- One worker: zero its own histogram, then count its slice. -/
def workerCode : Prog V L Unit := do
  -- A worker is handed the one pointer `cl_thread_spawn` was given: its
  -- descriptor, not the arena base.
  let desc := (← entryParams [.i64]).head
  let zero ← iconst64 0

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
  let cnt4      ← band dataCnt (← iconst64 (-4))
  let dataEnd4  ← iadd dataPtr2 (← ishlImm cnt4 2)

  let _ ← dwloop %[histPtr] .ult histEnd (contOnTrue := true) []
    (body := fun c => do
      let hp := c.head
      store zero hp
      for k in [1:8] do store zero (← iaddImm hp (8 * k))
      let hp' ← iaddImm hp 64
      return (hp', %[hp']))

  let mid ← ifte .ult dataPtr2 dataEnd4
    (thn := do
      let _ ← wloop1 dataPtr2
        (head := fun dp => return (contIfULt dp dataEnd4, %[], ()))
        (body := fun dp _ => do
          for k in [0:4] do
            let v ← uload32_64 (← iaddImm dp (4 * k))
            let a ← iadd histPtr (← ishlImm v 3)
            let c ← load64 a
            store (← iaddImm c 1) a
          return %[← iaddImm dp 16])
      pure %[dataEnd4])
    (els := pure %[dataPtr2])

  let _ ← wloop1 (mid.head)
    (head := fun dp => return (contIfULt dp dataEnd, %[], ()))
    (body := fun dp _ => do
      let v ← uload32_64 dp
      let a ← iadd histPtr (← ishlImm v 3)
      let c ← load64 a
      store (← iaddImm c 1) a
      return %[← iaddImm dp 4])

def clifIR : Except String (List FuncData) :=
  Prog.program
    [.ok noopFunction,
     .ok (noopAt 1),
     Prog.entry "main" (Prog.compileProg 2 orchCode),
     Prog.compileProg 3 workerCode [.i64]]

/-- Every byte of shared memory this program names.

    The offsets are assigned by hand, and a collision between two of them is
    invisible at every other layer: both stores succeed and the second wins.
    `0x00`-`0x18` are the context slots the FFI's init calls fill, so naming
    those is what stops an offset being placed where one of those calls will
    overwrite it. -/
def memMap : AlgorithmLib.Layout.RegionMap :=
  [⟨"input_path",  INPUT_PATH_OFF, OUTPUT_PATH_OFF - INPUT_PATH_OFF⟩,
   ⟨"output_path", OUTPUT_PATH_OFF, THREAD_CTX_OFF - OUTPUT_PATH_OFF⟩,
   ⟨"thread_ctx",  THREAD_CTX_OFF, HIST_REGION_OFF - THREAD_CTX_OFF⟩,
   ⟨"hist",        HIST_REGION_OFF, WORKERS * HIST_STRIDE⟩,
   ⟨"result",      RESULT_OFF, RESULT_SIZE⟩,
   ⟨"handles",     HANDLES_OFF, DESCS_OFF - HANDLES_OFF⟩,
   ⟨"descs",       DESCS_OFF, WORKERS * DESC_SIZE⟩,
   ⟨"data",        DATA_OFF, MAX_DATA_BYTES⟩]


#eval LayoutScan.check "HistogramBench4Algorithm" [``memMap]
theorem memMap_ok : AlgorithmLib.Layout.RegionMap.okB memMap = true := by decide

theorem memMap_within :
    AlgorithmLib.Layout.RegionMap.withinB MEM_SIZE memMap = true := by decide

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[artifactEntry "hist4_algorithm" {
    functions := clif,
    required_memory := MEM_SIZE
  }]

end HistogramBench4
