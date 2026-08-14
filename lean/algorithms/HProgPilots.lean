import Lean
import AlgorithmLib.Gen
import HistogramBench1Algorithm
import ClampSumBenchAlgorithm
import CudaRmsNormPersistAlgorithm

/-!
# Three generators written as `HProg` terms

One per shape the library has to carry:

* `hist1` — loops, an FFI read and write, byte and word memory traffic.
* `clampSum` — `f32x4` and `f64` carried through loop parameters, so every
  annotation the term holds is a type the checker had to agree with.
* `rmsNorm` — a four-function program with `i32`/`i64` callee signatures and a
  branch, written against the same `declareCudaFFI` the original uses.

Each is checked by `decide` (scoping and types) and `rfl` (the FFI calls it
performs, in order), and each emits an artifact alongside the generator it
mirrors so `base/tests/hprog_diff.rs` can compare the two by behavior.
-/

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg

namespace HProgPilots

-- ---------------------------------------------------------------------------
-- Pilot 1: the histogram
-- ---------------------------------------------------------------------------

namespace Hist

open HistogramBench1 (INPUT_PATH_OFF OUTPUT_PATH_OFF HIST_OFF HIST_BYTES DATA_OFF MEM_SIZE)

/-- The standard table, which is what `HistogramBench1.orchFn` is checked
    against too. -/
def env : FnEnv := env% [.cuda, .fileIO]

def fnRead : Nat := IR.FFI.std.fileRead.id
def fnWrite : Nat := IR.FFI.std.fileWrite.id

open Sur in
/-- `HistogramBench1.orchFn`, as a term. -/
def code : Code := clif%(env, ptrParams) do
  let dataPtr ← load64 (← absAddr basePtr 0x18)
  let zeroI ← iconst64 0
  -- copy the input path until NUL
  let inExit ← wloop1 zeroI
    (head := fun si => do
      let ch ← uload8_64 (← iadd dataPtr si)
      istore8 ch (← iadd (← absAddr basePtr INPUT_PATH_OFF) si)
      let si' ← iadd si (← iconst64 1)
      return (exitIfEq ch (← iconst64 0), [si'], si'))
    (body := fun _ si' => return [si'])
  -- copy the output path until NUL
  let _ ← wloop2 (inExit.headD 0) zeroI
    (head := fun si di => do
      let ch ← uload8_64 (← iadd dataPtr si)
      istore8 ch (← iadd (← absAddr basePtr OUTPUT_PATH_OFF) di)
      let si' ← iadd si (← iconst64 1)
      let di' ← iadd di (← iconst64 1)
      return (exitIfEq ch (← iconst64 0), ([] : List R), (si', di')))
    (body := fun _ _ c => return [c.1, c.2])
  -- read the file, then derive the element count and the histogram bounds
  let fileSize ← call fnRead
    [basePtr, ← iconst64 INPUT_PATH_OFF, ← iconst64 DATA_OFF, zeroI, zeroI]
  let n ← ushr fileSize (← iconst64 2)
  let histPtr ← absAddr basePtr HIST_OFF
  let histEnd ← iadd histPtr (← iconst64 HIST_BYTES)
  -- zero the histogram, eight words per iteration
  let _ ← wloop1 histPtr
    (head := fun hp => do
      store zeroI hp
      for off in [8, 16, 24, 32, 40, 48, 56] do
        store zeroI (← iadd hp (← iconst64 off))
      let hp' ← iadd hp (← iconst64 64)
      return (contIfULt hp' histEnd, ([] : List R), hp'))
    (body := fun _ hp' => return [hp'])
  -- scan bounds
  let dataPtr2 ← absAddr basePtr DATA_OFF
  let dataEnd ← iadd dataPtr2 (← ishl n (← iconst64 2))
  let n4 ← band n (← iconst64 (-4))
  let dataEnd4 ← iadd dataPtr2 (← ishl n4 (← iconst64 2))
  -- four elements per iteration, then the scalar tail
  let bump := fun (histPtr one dp : R) (off : Int) => do
    let v ← uload32_64 (← iadd dp (← iconst64 off))
    let a ← iadd histPtr (← ishl v (← iconst64 3))
    store (← iadd (← load64 a) one) a
  let scanExit ← wloop1 dataPtr2
    (head := fun dp => return (contIfULt dp dataEnd4, [dataEnd4], ()))
    (body := fun dp _ => do
      let one ← iconst64 1
      for off in [0, 4, 8, 12] do bump histPtr one dp off
      return [← iadd dp (← iconst64 16)])
  let _ ← wloop1 (scanExit.headD 0)
    (head := fun dp => return (contIfULt dp dataEnd, ([] : List R), ()))
    (body := fun dp _ => do
      bump histPtr (← iconst64 1) dp 0
      return [← iadd dp (← iconst64 4)])
  -- write the histogram out
  let _ ← call fnWrite
    [basePtr, ← iconst64 OUTPUT_PATH_OFF, ← iconst64 HIST_OFF, zeroI,
     ← iconst64 HIST_BYTES]
  return ()

/-- Every reference names a slot in scope, at the type its use demands. -/
theorem code_wf : wf env ptrParams code = true := by decide

/-- The program reads one file and then writes one, and does nothing else
    across the FFI. -/
theorem code_calls : callsOf code = [fnRead, fnWrite] := rfl

def program : Program :=
  IR.program [noopFunction, noopAt 1, compileFn 2 code env]

end Hist

-- ---------------------------------------------------------------------------
-- Pilot 2: the clamped sum — floats and vectors through loop carries
-- ---------------------------------------------------------------------------

namespace ClampSum

open ClampSumBench (MEM_SIZE)

/-- No FFI: the whole computation is loads, arithmetic and one store. -/
def env : FnEnv := { sigs := [], fns := [] }

open Sur in
/-- `ClampSumBench.mainFn`, as a term. The four vector accumulators and the
    `f64` tail accumulator are ordinary binders; their types reach the emitted
    block parameters because the surface tracked them. -/
def code : Code := clif%(env, ptrParams) do
  let dataPtr ← load64 (← absAddr basePtr 0x18)
  let dataLen ← load64 (← absAddr basePtr 0x20)
  let outPtr ← load64 (← absAddr basePtr 0x28)
  let two ← iconst64 2
  let n ← ushr dataLen two
  let mainEnd ← ishl (← ushr n (← iconst64 4)) (← iconst64 6)
  let simdEnd ← ishl (← ushr n two) (← iconst64 4)
  let scEnd ← ishl n two
  let hi ← fconst32 0.5
  let lo ← fneg hi
  let hiV ← splat .f32x4 hi
  let loV ← splat .f32x4 lo
  let zeroV ← splat .f32x4 (← fconst32 f32Zero)
  let i0 ← iconst64 0
  let clamp := fun (v hiB loB acc : R) => do fadd acc (← fmax (← fmin v hiB) loB)
  -- four vectors per iteration, into four accumulators
  let wide ← wloop [i0, zeroV, zeroV, zeroV, zeroV]
    (head := fun cs => return (exitIfSGe (cs.headD 0) mainEnd, cs, ()))
    (body := fun cs _ => do
      let i := cs.headD 0
      let off ← iadd dataPtr i
      let mut acc := #[]
      for (a, k) in (cs.drop 1).zipIdx do
        let v ← loadF32x4 (← iadd off (← iconst64 (16 * k)))
        acc := acc.push (← clamp v hiV loV a)
      return (← iadd i (← iconst64 64)) :: acc.toList)
  let acc ← fadd (← fadd (wide.getD 1 0) (wide.getD 2 0))
                 (← fadd (wide.getD 3 0) (wide.getD 4 0))
  -- one vector per iteration
  let simd ← wloop2 (wide.headD 0) acc
    (head := fun i v => return (exitIfSGe i simdEnd, [i, v], ()))
    (body := fun i v _ => do
      let x ← loadF32x4 (← iadd dataPtr i)
      return [← iadd i (← iconst64 16), ← clamp x hiV loV v])
  -- horizontal reduction to f64
  let vec := simd.getD 1 0
  let mut lanes := #[]
  for k in [0, 1, 2, 3] do lanes := lanes.push (← fpromote (← extractlane vec k))
  let sum64 ← fadd (← fadd (lanes.getD 0 0) (lanes.getD 1 0))
                   (← fadd (lanes.getD 2 0) (lanes.getD 3 0))
  -- one element per iteration
  let tail ← wloop2 (simd.headD 0) sum64
    (head := fun i s => return (exitIfSGe i scEnd, [s], ()))
    (body := fun i s _ => do
      let x ← loadF32 (← iadd dataPtr i)
      let c ← fmax (← fmin x hi) lo
      return [← iadd i (← iconst64 4), ← fadd s (← fpromote c)])
  store (tail.headD 0) outPtr

/-- Scoping and types, including the `f32x4` and `f64` carried across every
    loop boundary. -/
theorem code_wf : wf env ptrParams code = true := by decide

/-- The computation crosses no FFI boundary at all. -/
theorem code_calls : callsOf code = [] := rfl

def program : Program :=
  IR.program [noopFunction, compileFn 1 code env]

end ClampSum

-- ---------------------------------------------------------------------------
-- Pilot 3: RMSNorm's host side — four functions, `i32` callees, a branch
-- ---------------------------------------------------------------------------

namespace RmsNorm

open CudaRmsNormPersist (PTX_SOURCE_OFF BIND_DESC_OFF MEM_SIZE N_OFF BUF0_OFF BUF1_OFF)

/-- The same `declareCudaFFI` the original calls, so the signatures the checker
    holds these bodies to are the runtime's own. -/
def cudaEnv : CudaSetup × FnEnv := (IR.FFI.std.cuda, env% [.cuda, .fileIO])
def cuda : CudaSetup := cudaEnv.1
def env : FnEnv := cudaEnv.2

/-- The CUDA context pointer lives at a fixed slot in shared memory. -/
def CTX_OFF : Int := 0x10

open Sur in
/-- Initialize CUDA, allocate the two device buffers, upload `N` and the
    weights. -/
def loadCode : Code := clif%(env, ptrParams) do
  let dataPtr ← load64 (← absAddr basePtr 0x18)
  callVoid cuda.fnInit.id [← absAddr basePtr CTX_OFF]
  let ctxPtr ← load64 (← absAddr basePtr CTX_OFF)
  let n ← load64 dataPtr
  store n (← absAddr basePtr N_OFF)
  let nBytes ← ishl n (← iconst64 2)
  let buf0Sz ← iadd (← iadd nBytes nBytes) (← iconst64 8)
  let buf1Sz ← ishl n (← iconst64 2)
  let buf0 ← call cuda.fnCreateBuffer.id [ctxPtr, buf0Sz]
  let buf1 ← call cuda.fnCreateBuffer.id [ctxPtr, buf1Sz]
  store buf0 (← absAddr basePtr BUF0_OFF)
  store buf1 (← absAddr basePtr BUF1_OFF)
  let _ ← call cuda.fnUploadOffset.id
    [ctxPtr, buf0, ← iconst64 0, ← absAddr basePtr N_OFF, ← iconst64 8]
  let _ ← call cuda.fnUploadOffset.id
    [ctxPtr, buf0, ← iadd nBytes (← iconst64 8),
     ← iadd dataPtr (← iconst64 8), nBytes]
  return ()

open Sur in
/-- Upload the input vector ahead of a launch. -/
def prepCode : Code := clif% do
  let dataPtr ← load64 (← absAddr basePtr 0x18)
  let n ← load64 (← absAddr basePtr N_OFF)
  let buf0 ← load32 (← absAddr basePtr BUF0_OFF)
  let ctxPtr ← load64 (← absAddr basePtr CTX_OFF)
  let _ ← call cuda.fnUploadOffset.id
    [ctxPtr, buf0, ← iconst64 8, dataPtr, ← ishl n (← iconst64 2)]
  return ()

open Sur in
/-- Launch, synchronize, and download only when the caller asked for output. -/
def inferCode : Code := clif% do
  let outPtr ← load64 (← absAddr basePtr 0x28)
  let outLen ← load64 (← absAddr basePtr 0x30)
  let ctxPtr ← load64 (← absAddr basePtr CTX_OFF)
  let nBufs ← iconst32 2
  let one32 ← iconst32 1
  let blk256 ← iconst32 256
  let _ ← call cuda.fnLaunch.id
    [ctxPtr, ← absAddr basePtr PTX_SOURCE_OFF, nBufs,
     ← absAddr basePtr BIND_DESC_OFF,
     one32, one32, one32, blk256, one32, one32]
  let _ ← call cuda.fnSync.id [ctxPtr]
  let zeroI ← iconst64 0
  when .ne outLen zeroI do
    let buf1 ← load32 (← absAddr basePtr BUF1_OFF)
    let _ ← call cuda.fnDownload.id [ctxPtr, buf1, outPtr, outLen]
    return ()

theorem load_wf : wf env ptrParams loadCode = true := by decide
theorem prep_wf : wf env ptrParams prepCode = true := by decide
theorem infer_wf : wf env ptrParams inferCode = true := by decide

/-- Loading allocates both buffers before either upload. -/
theorem load_calls :
    callsOf loadCode =
      [cuda.fnInit.id, cuda.fnCreateBuffer.id, cuda.fnCreateBuffer.id,
       cuda.fnUploadOffset.id, cuda.fnUploadOffset.id] := rfl

/-- Inference launches, synchronizes, and downloads at most once. -/
theorem infer_calls :
    callsOf inferCode = [cuda.fnLaunch.id, cuda.fnSync.id, cuda.fnDownload.id] := rfl

def program : Program :=
  IR.program
    [noopFunction,
     compileFn 1 loadCode,
     compileFn 2 prepCode,
     compileFn 3 inferCode]

end RmsNorm

-- ---------------------------------------------------------------------------
-- Nesting, and terms with an open parameter
-- ---------------------------------------------------------------------------

namespace Nested

def env : FnEnv := Hist.env

open Sur in
/-- Σ over `i < 3`, `j < 4` of `4i + j`, by a loop inside a loop, then a branch
    on the total, written to a file. The inner body reads the outer body's
    carry and the prologue's constants, which is the cross-scope dataflow
    nesting has to carry. -/
def code : Code := clif%(env, ptrParams) do
  let dataPtr ← load64 (← absAddr basePtr 0x18)
  let zeroI ← iconst64 0
  let _ ← wloop1 zeroI
    (head := fun si => do
      let ch ← uload8_64 (← iadd dataPtr si)
      istore8 ch (← iadd (← absAddr basePtr 0x200) si)
      let si' ← iadd si (← iconst64 1)
      return (exitIfEq ch (← iconst64 0), ([] : List R), si'))
    (body := fun _ si' => return [si'])
  let three ← iconst64 3
  let four ← iconst64 4
  let one ← iconst64 1
  let outer ← wloop2 zeroI zeroI
    (head := fun i acc => return (contIfULt i three, [acc], ()))
    (body := fun i acc _ => do
      let inner ← wloop2 zeroI acc
        (head := fun j a => return (contIfULt j four, [a], ()))
        (body := fun j a _ => do
          let t ← iadd (← ishl i (← iconst64 2)) j
          return [← iadd j one, ← iadd a t])
      return [← iadd i one, inner.headD 0])
  let acc := outer.headD 0
  store acc (← absAddr basePtr 0x400)
  -- branch on the total; `acc < 100`, so the then-arm's export reaches the join
  let branched ← ifte .ult acc (← iconst64 100)
    (thn := do pure [← iadd acc (← iconst64 1000)])
    (els := pure [acc])
  store (branched.headD 0) (← absAddr basePtr 0x408)
  let _ ← call Hist.fnWrite
    [basePtr, ← iconst64 0x200, ← iconst64 0x400, zeroI, ← iconst64 16]
  return ()

theorem code_wf : wf env ptrParams code = true := by decide

theorem code_calls : callsOf code = [Hist.fnWrite] := rfl

def program : Program :=
  IR.program [noopFunction, noopAt 1, compileFn 2 code env]

end Nested

-- ---------------------------------------------------------------------------
-- Leaving a loop early
-- ---------------------------------------------------------------------------

namespace Early

def env : FnEnv := Hist.env

open Sur in
/-- A scan that stops at the first zero byte and reports where it stopped, and
    a nested pair of loops the inner one leaves outright.

    Every rule `br` adds is exercised: nothing follows it, an arm that leaves
    means the join carries only what the other arm exports, both arms leaving
    means there is no join and the body has no back edge, and a depth above zero
    names an outer loop. `wf` below is what says so in the kernel — the corpus
    says the same thing by running it. -/
def code : Code := clif%(env, ptrParams) do
  let dataPtr ← load64 (← absAddr basePtr 0x18)
  let one ← iconst64 1
  let zero ← iconst64 0
  -- stop at the first zero byte, or at 64, whichever comes first
  let stopped ← wloop1 zero
    (head := fun i => return (exitIfSGe i (← iconst64 64), [i], ()))
    (body := fun i _ => do
      let ch ← uload8_64 (← iadd dataPtr i)
      when .eq ch zero (brk [i])
      return [← iadd i one])
  store (stopped.headD 0) (← absAddr basePtr 0x400)
  -- both arms leave, so the inner loop's body ends without a back edge, and the
  -- deeper `brk` names the outer loop from inside the inner one
  let picked ← wloop2 zero zero
    (head := fun i acc => return (exitIfSGe i (← iconst64 8), [acc], ()))
    (body := fun i acc _ => do
      let inner ← wloop1 zero
        (head := fun j => return (exitIfSGe j (← iconst64 8), [j], ()))
        (body := fun j _ => do
          let _ ← ifte .sge (← iadd i j) (← iconst64 6)
            (thn := do brkTo 1 [← iadd acc one]; pure [])
            (els := do brk [j]; pure [])
          return [j])
      return [← iadd i one, ← iadd acc (inner.headD 0)])
  store (picked.headD 0) (← absAddr basePtr 0x408)
  return ()

theorem code_wf : wf env ptrParams code = true := by decide

def program : Program := IR.program [noopFunction, compileFn 1 code env]

end Early

-- ---------------------------------------------------------------------------
-- A term with an open parameter
-- ---------------------------------------------------------------------------

open Sur in
/-- Slot numbering never depends on an immediate's *value*, so the builder
    reduces symbolically with `k` still open. -/
def openFrag (k : Int) : Code :=
  HProg.Sur.build (env := { sigs := [], fns := [] }) do
    let a ← Sur.iconst .i64 k
    let b ← Sur.iadd Sur.basePtr a
    let c ← Sur.load64 b
    Sur.store c b

/-- One characterization lemma per parameterized fragment, by `rfl` and generic
    in the parameter; every proof after it works on the literal.

    Reducing the slot map descends by halving the slot number, and `Nat.div` is
    irreducible, so the elaborator needs telling to unfold it. The kernel checks
    the same `rfl` either way — reducibility is an elaboration setting and the
    proof term this produces is the one a bare `rfl` would. -/
theorem openFrag_eq (k : Int) :
    openFrag k = [.straight [.op (.iconst .i64 k), .op (.iadd 0 1),
                             .op (.load { ty := .i64 } 2), .store .i64 3 2]] := by
  with_unfolding_all rfl



/-- `compile_sound` for the histogram, executed rather than proved: the term
    and the function `compileFn` produced perform the same observations, in the
    same order, and leave the same memory.

    Stronger than the corpus's version of this check, because the trace here
    contains FFI *calls* with their arguments, not only stores — so it pins the
    order of the two file operations across three loops and a branch. -/
def histCompileSound : Except String (Nat × Nat) :=
  let arena := ByteArray.mk (Array.replicate 0x8000 0)
  let payload := ByteArray.mk
    (("in.bin".toUTF8.toList ++ [0] ++ "out.bin".toUTF8.toList ++ [0]).toArray)
  let input := ByteArray.mk (((List.range 64).flatMap fun v =>
    [UInt8.ofNat (v % 256), 0, 0, 0]).toArray)
  let m0 : Sem.Mem :=
    { arena, data := ByteArray.mk (Array.replicate 64 0),
      out := ByteArray.mk (Array.replicate 8 0) }
  match (List.range payload.size).foldlM
      (fun (mm : Sem.Mem) i => mm.store (Sem.addrOf .data i) 1 (payload.get! i).toUInt64) m0 with
  | none => .error "could not place the payload"
  | some m1 =>
    match m1.store (Sem.addrOf .arena 0x18) 8 (Sem.regionBase .data) with
    | none => .error "could not place the data pointer"
    | some m =>
      let w : Sem.World := { mem := m, fs := { files := [("in.bin", input)] } }
      let ptr : Sem.V := .sc .i64 (Sem.regionBase .arena)
      match Sem.run { env := Hist.env } [ptr] w Hist.code,
            Blocks.run Hist.env (compileFn 2 Hist.code) [ptr] w with
      | .stuck e, _ => .error s!"term: {e}"
      | _, .stuck e => .error s!"blocks: {e}"
      | .ok tObs tw, .ok bObs bw =>
          if tObs != bObs then .error s!"traces differ: {tObs.length} vs {bObs.length}"
          else if (tw.fs.get "out.bin") != (bw.fs.get "out.bin") then
            .error "output files differ"
          else .ok (tObs.length,
            (tObs.filter fun o => match o with | .call .. => true | _ => false).length)

end HProgPilots

open AlgorithmLib in
/-- Each pilot's artifact next to the one its generator emits, so the runtime
    can be asked whether they behave the same. -/
def main (args : List String) : IO Unit := do
  let dir ← requireOutputDir args
  emitArtifacts dir <|
    HistogramBench1.artifacts ++
    ClampSumBench.artifacts ++
    CudaRmsNormPersist.artifacts ++
    #[toJsonEntry "hist1_hprog" {
        clif := HProgPilots.Hist.program,
        memory_size := HistogramBench1.MEM_SIZE
      } { fn_idx := u32 2 },
      toJsonEntry "clamp_sum_hprog" {
        clif := HProgPilots.ClampSum.program,
        memory_size := ClampSumBench.MEM_SIZE
      } { fn_idx := u32 1 },
      toJsonArtifact "cuda_rmsnorm_hprog" {
        clif := HProgPilots.RmsNorm.program,
        memory_size := CudaRmsNormPersist.MEM_SIZE,
        initial_memory := CudaRmsNormPersist.buildInitialMemory
      } { fn_idx := u32 1 } [
        ("prep", { fn_idx := u32 2 }),
        ("infer", { fn_idx := u32 3 })
      ],
      toJsonEntry "nested_hprog" {
        clif := HProgPilots.Nested.program,
        memory_size := 0x100000
      } { fn_idx := u32 2 }]
  -- Each pilot reports the FFI it actually assumes, derived from its own term.
  for (nm, e, c) in [("hist1", HProgPilots.Hist.env, HProgPilots.Hist.code),
                     ("clamp_sum", HProgPilots.ClampSum.env, HProgPilots.ClampSum.code),
                     ("rmsnorm.infer", HProgPilots.RmsNorm.env, HProgPilots.RmsNorm.inferCode),
                     ("nested", HProgPilots.Nested.env, HProgPilots.Nested.code)] do
    if !footprintComplete e c then
      throw (IO.userError s!"{nm} calls a symbol with no declared frame")
    IO.println s!"  {nm}: {footprintReport e c}"
  match HProgPilots.histCompileSound with
  | .error e => throw (IO.userError s!"histogram compile_sound: {e}")
  | .ok (n, calls) =>
      IO.println s!"compile_sound (executed): histogram term and compiled form agree \
                   — {n} observations, {calls} of them FFI calls"

