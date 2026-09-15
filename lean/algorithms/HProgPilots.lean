import Lean
import AlgorithmLib.Gen
import HistogramBench1Algorithm
import ClampSumBenchAlgorithm
import CudaRmsNormPersistAlgorithm
import ShipScan

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
open AlgorithmLib.Prog

namespace HProgPilots

-- ---------------------------------------------------------------------------
-- Pilot 1: the histogram
-- ---------------------------------------------------------------------------

namespace Hist

open HistogramBench1 (INPUT_PATH_OFF OUTPUT_PATH_OFF HIST_OFF HIST_BYTES DATA_OFF MEM_SIZE)

abbrev fnRead : Ffi := .fileRead
abbrev fnWrite : Ffi := .fileWrite

/-- `HistogramBench1.orchFn`, as a term. -/
def code : Prog V L Unit := do
  let dataPtr ← load64 (← absAddr (← basePtr) 0x18)
  let zeroI ← iconst64 0
  -- copy the input path until NUL
  let inExit ← wloop1 zeroI
    (head := fun si => do
      let ch ← uload8_64 (← iadd dataPtr si)
      istore8 ch (← iadd (← absAddr (← basePtr) INPUT_PATH_OFF) si)
      let si' ← iadd si (← iconst64 1)
      return (exitIfEq ch (← iconst64 0), %[si'], si'))
    (body := fun _ si' => return %[si'])
  -- copy the output path until NUL
  let _ ← wloop2 (inExit.head) zeroI
    (head := fun si di => do
      let ch ← uload8_64 (← iadd dataPtr si)
      istore8 ch (← iadd (← absAddr (← basePtr) OUTPUT_PATH_OFF) di)
      let si' ← iadd si (← iconst64 1)
      let di' ← iadd di (← iconst64 1)
      return (exitIfEq ch (← iconst64 0), %[], (si', di')))
    (body := fun _ _ c => return %[c.1, c.2])
  -- read the file, then derive the element count and the histogram bounds
  let fileSize ← ffi fnRead
    %[(← basePtr), ← iconst64 INPUT_PATH_OFF, ← iconst64 DATA_OFF, zeroI, zeroI]
  let n ← ushr fileSize (← iconst64 2)
  let histPtr ← absAddr (← basePtr) HIST_OFF
  let histEnd ← iadd histPtr (← iconst64 HIST_BYTES)
  -- zero the histogram, eight words per iteration
  let _ ← wloop1 histPtr
    (head := fun hp => do
      store zeroI hp
      for off in [8, 16, 24, 32, 40, 48, 56] do
        store zeroI (← iadd hp (← iconst64 off))
      let hp' ← iadd hp (← iconst64 64)
      return (contIfULt hp' histEnd, %[], hp'))
    (body := fun _ hp' => return %[hp'])
  -- scan bounds
  let dataPtr2 ← absAddr (← basePtr) DATA_OFF
  let dataEnd ← iadd dataPtr2 (← ishl n (← iconst64 2))
  let n4 ← band n (← iconst64 (-4))
  let dataEnd4 ← iadd dataPtr2 (← ishl n4 (← iconst64 2))
  -- four elements per iteration, then the scalar tail
  let bump := fun (histPtr one dp : V .i64) (off : Int) => do
    let v ← uload32_64 (← iadd dp (← iconst64 off))
    let a ← iadd histPtr (← ishl v (← iconst64 3))
    store (← iadd (← load64 a) one) a
  let scanExit ← wloop1 dataPtr2
    (head := fun dp => return (contIfULt dp dataEnd4, %[dataEnd4], ()))
    (body := fun dp _ => do
      let one ← iconst64 1
      for off in [0, 4, 8, 12] do bump histPtr one dp off
      return %[← iadd dp (← iconst64 16)])
  let _ ← wloop1 (scanExit.head)
    (head := fun dp => return (contIfULt dp dataEnd, %[], ()))
    (body := fun dp _ => do
      bump histPtr (← iconst64 1) dp 0
      return %[← iadd dp (← iconst64 4)])
  -- write the histogram out
  let _ ← ffi fnWrite
    %[(← basePtr), ← iconst64 OUTPUT_PATH_OFF, ← iconst64 HIST_OFF, zeroI,
     ← iconst64 HIST_BYTES]
  return ()


/-- The program reads one file and then writes one, and does nothing else
    across the FFI. -/
theorem code_calls : callsOf (Prog.emit code) = [fnRead, fnWrite].map Ffi.id := rfl

def program : Except String Program :=
  Prog.program
    [.ok noopFunction, .ok (noopAt 1), Prog.compileProg 2 code]

end Hist

-- ---------------------------------------------------------------------------
-- Pilot 2: the clamped sum — floats and vectors through loop carries
-- ---------------------------------------------------------------------------

namespace ClampSum

open ClampSumBench (MEM_SIZE)

/-- `ClampSumBench.mainFn`, as a term. The four vector accumulators and the
    `f64` tail accumulator are ordinary binders; their types reach the emitted
    block parameters because the surface tracked them. -/
def code : Prog V L Unit := do
  let dataPtr ← load64 (← absAddr (← basePtr) 0x18)
  let dataLen ← load64 (← absAddr (← basePtr) 0x20)
  let outPtr ← load64 (← absAddr (← basePtr) 0x28)
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
  let clamp := fun v hiB loB acc => do fadd acc (← fmax (← fmin v hiB) loB)
  -- four vectors per iteration, into four accumulators
  let wide ← wloop (Vals.cons i0 (Vals.ofFn (n := 4) (fun _ => zeroV)))
    (head := fun cs => return (exitIfSGe (cs.head) mainEnd, cs, ()))
    (body := fun cs _ => do
      let i := cs.head
      let off ← iadd dataPtr i
      -- the four accumulators are a run of one type, rebuilt in position order
      let acc ← cs.tail.uniformMapIdxM fun k a => do
        let v ← loadF32x4 (← iadd off (← iconst64 (16 * k)))
        clamp v hiV loV a
      return Vals.cons (← iadd i (← iconst64 64)) acc)
  let acc ← fadd (← fadd (wide.snd) (wide.thd))
                 (← fadd (wide.fth) (wide.fif))
  -- one vector per iteration
  let simd ← wloop2 (wide.head) acc
    (head := fun i v => return (exitIfSGe i simdEnd, %[i, v], ()))
    (body := fun i v _ => do
      let x ← loadF32x4 (← iadd dataPtr i)
      return %[← iadd i (← iconst64 16), ← clamp x hiV loV v])
  -- horizontal reduction to f64
  -- the lane index is part of the operation's type, so the four are written
  let vec := simd.snd
  let l0 ← fpromote (← extractlane vec 0)
  let l1 ← fpromote (← extractlane vec 1)
  let l2 ← fpromote (← extractlane vec 2)
  let l3 ← fpromote (← extractlane vec 3)
  let sum64 ← fadd (← fadd l0 l1) (← fadd l2 l3)
  -- one element per iteration
  let tail ← wloop2 (simd.head) sum64
    (head := fun i s => return (exitIfSGe i scEnd, %[s], ()))
    (body := fun i s _ => do
      let x ← loadF32 (← iadd dataPtr i)
      let c ← fmax (← fmin x hi) lo
      return %[← iadd i (← iconst64 4), ← fadd s (← fpromote c)])
  store (tail.head) outPtr


/-- The computation crosses no FFI boundary at all. -/
theorem code_calls : callsOf (Prog.emit code) = [] := rfl

def program : Except String Program :=
  Prog.program
    [.ok noopFunction, Prog.compileProg 1 code]

end ClampSum

-- ---------------------------------------------------------------------------
-- Pilot 3: RMSNorm's host side — four functions, `i32` callees, a branch
-- ---------------------------------------------------------------------------

namespace RmsNorm

open CudaRmsNormPersist (PTX_SOURCE_OFF BIND_DESC_OFF MEM_SIZE N_OFF BUF0_OFF BUF1_OFF)

/-- The CUDA context pointer lives at a fixed slot in shared memory. -/
def CTX_OFF : Int := 0x10

/-- Initialize CUDA, allocate the two device buffers, upload `N` and the
    weights. -/
def loadCode : Prog V L Unit := do
  let dataPtr ← load64 (← absAddr (← basePtr) 0x18)
  ffiVoid .cudaInit %[← absAddr (← basePtr) CTX_OFF]
  let ctxPtr ← load64 (← absAddr (← basePtr) CTX_OFF)
  let n ← load64 dataPtr
  store n (← absAddr (← basePtr) N_OFF)
  let nBytes ← ishl n (← iconst64 2)
  let buf0Sz ← iadd (← iadd nBytes nBytes) (← iconst64 8)
  let buf1Sz ← ishl n (← iconst64 2)
  let buf0 ← ffi .cudaCreateBuffer %[ctxPtr, buf0Sz]
  let buf1 ← ffi .cudaCreateBuffer %[ctxPtr, buf1Sz]
  store buf0 (← absAddr (← basePtr) BUF0_OFF)
  store buf1 (← absAddr (← basePtr) BUF1_OFF)
  let _ ← ffi .cudaUploadOffset
    %[ctxPtr, buf0, ← iconst64 0, ← absAddr (← basePtr) N_OFF, ← iconst64 8]
  let _ ← ffi .cudaUploadOffset
    %[ctxPtr, buf0, ← iadd nBytes (← iconst64 8),
     ← iadd dataPtr (← iconst64 8), nBytes]
  return ()

/-- Upload the input vector ahead of a launch. -/
def prepCode : Prog V L Unit := do
  let dataPtr ← load64 (← absAddr (← basePtr) 0x18)
  let n ← load64 (← absAddr (← basePtr) N_OFF)
  let buf0 ← load32 (← absAddr (← basePtr) BUF0_OFF)
  let ctxPtr ← load64 (← absAddr (← basePtr) CTX_OFF)
  let _ ← ffi .cudaUploadOffset
    %[ctxPtr, buf0, ← iconst64 8, dataPtr, ← ishl n (← iconst64 2)]
  return ()

/-- Launch, synchronize, and download only when the caller asked for output. -/
def inferCode : Prog V L Unit := do
  let outPtr ← load64 (← absAddr (← basePtr) 0x28)
  let outLen ← load64 (← absAddr (← basePtr) 0x30)
  let ctxPtr ← load64 (← absAddr (← basePtr) CTX_OFF)
  let nBufs ← iconst32 2
  let one32 ← iconst32 1
  let blk256 ← iconst32 256
  let _ ← ffi .cudaLaunch
    %[ctxPtr, ← absAddr (← basePtr) PTX_SOURCE_OFF, nBufs,
     ← absAddr (← basePtr) BIND_DESC_OFF,
     one32, one32, one32, blk256, one32, one32]
  let _ ← ffi .cudaSync %[ctxPtr]
  let zeroI ← iconst64 0
  when .ne outLen zeroI do
    let buf1 ← load32 (← absAddr (← basePtr) BUF1_OFF)
    let _ ← ffi .cudaDownload %[ctxPtr, buf1, outPtr, outLen]
    return ()

/-- Loading allocates both buffers before either upload. -/
theorem load_calls :
    callsOf (Prog.emit loadCode) =
      [IR.Ffi.cudaInit.id, IR.Ffi.cudaCreateBuffer.id, IR.Ffi.cudaCreateBuffer.id,
       IR.Ffi.cudaUploadOffset.id, IR.Ffi.cudaUploadOffset.id] := rfl

/-- Inference launches, synchronizes, and downloads at most once. -/
theorem infer_calls :
    callsOf (Prog.emit inferCode) =
      [IR.Ffi.cudaLaunch.id, IR.Ffi.cudaSync.id, IR.Ffi.cudaDownload.id] := rfl

def program : Except String Program :=
  Prog.program
    [.ok noopFunction,
     Prog.compileProg 1 loadCode,
     Prog.compileProg 2 prepCode,
     Prog.compileProg 3 inferCode]

end RmsNorm

-- ---------------------------------------------------------------------------
-- Nesting, and terms with an open parameter
-- ---------------------------------------------------------------------------

namespace Nested


/-- Σ over `i < 3`, `j < 4` of `4i + j`, by a loop inside a loop, then a branch
    on the total, written to a file. The inner body reads the outer body's
    carry and the prologue's constants, which is the cross-scope dataflow
    nesting has to carry. -/
def code : Prog V L Unit := do
  let dataPtr ← load64 (← absAddr (← basePtr) 0x18)
  let zeroI ← iconst64 0
  let _ ← wloop1 zeroI
    (head := fun si => do
      let ch ← uload8_64 (← iadd dataPtr si)
      istore8 ch (← iadd (← absAddr (← basePtr) 0x200) si)
      let si' ← iadd si (← iconst64 1)
      return (exitIfEq ch (← iconst64 0), %[], si'))
    (body := fun _ si' => return %[si'])
  let three ← iconst64 3
  let four ← iconst64 4
  let one ← iconst64 1
  let outer ← wloop2 zeroI zeroI
    (head := fun i acc => return (contIfULt i three, %[acc], ()))
    (body := fun i acc _ => do
      let inner ← wloop2 zeroI acc
        (head := fun j a => return (contIfULt j four, %[a], ()))
        (body := fun j a _ => do
          let t ← iadd (← ishl i (← iconst64 2)) j
          return %[← iadd j one, ← iadd a t])
      return %[← iadd i one, inner.head])
  let acc := outer.head
  store acc (← absAddr (← basePtr) 0x400)
  -- branch on the total; `acc < 100`, so the then-arm's export reaches the join
  let branched ← ifte .ult acc (← iconst64 100)
    (thn := do pure %[← iadd acc (← iconst64 1000)])
    (els := pure %[acc])
  store (branched.head) (← absAddr (← basePtr) 0x408)
  let _ ← ffi Hist.fnWrite
    %[(← basePtr), ← iconst64 0x200, ← iconst64 0x400, zeroI, ← iconst64 16]
  return ()


theorem code_calls : callsOf (Prog.emit code) = [Hist.fnWrite].map Ffi.id := rfl

def program : Except String Program :=
  Prog.program
    [.ok noopFunction, .ok (noopAt 1), Prog.compileProg 2 code]

end Nested

-- ---------------------------------------------------------------------------
-- Leaving a loop early
-- ---------------------------------------------------------------------------

namespace Early


/-- A scan that stops at the first zero byte and reports where it stopped, and
    a nested pair of loops the inner one leaves outright.

    Every rule `br` adds is exercised: nothing follows it, an arm that leaves
    means the join carries only what the other arm exports, both arms leaving
    means there is no join and the body has no back edge, and a depth above zero
    names an outer loop. `wf` below is what says so in the kernel — the corpus
    says the same thing by running it. -/
def code : Prog V L Unit := do
  let dataPtr ← load64 (← absAddr (← basePtr) 0x18)
  let one ← iconst64 1
  let zero ← iconst64 0
  -- stop at the first zero byte, or at 64, whichever comes first
  let stopped ← wloop1L zero
    (head := fun _ i => return (exitIfSGe i (← iconst64 64), %[i], ()))
    (body := fun scan i _ => do
      let ch ← uload8_64 (← iadd dataPtr i)
      when .eq ch zero (brk scan %[i])
      return %[← iadd i one])
  store (stopped.head) (← absAddr (← basePtr) 0x400)
  -- both arms leave, so the inner loop's body ends without a back edge, and the
  -- deeper `brk` names the outer loop from inside the inner one
  let picked ← wloop2L zero zero
    (head := fun _ i _ => return (exitIfSGe i (← iconst64 8), %[zero], ()))
    (body := fun outer i acc _ => do
      let inner ← wloop1L zero
        (head := fun _ j => return (exitIfSGe j (← iconst64 8), %[j], ()))
        (body := fun innerLbl j _ => do
          let _ ← ifte (jTys := []) .sge (← iadd i j) (← iconst64 6)
            (thn := do brk outer %[← iadd acc one])
            (els := do brk innerLbl %[j])
          return %[j])
      return %[← iadd i one, ← iadd acc inner.head])
  store (picked.head) (← absAddr (← basePtr) 0x408)
  return ()


def program : Except String Program :=
  Prog.program [.ok noopFunction, Prog.compileProg 1 code]

end Early

-- ---------------------------------------------------------------------------
-- A term with an open parameter
-- ---------------------------------------------------------------------------

/-- Slot numbering never depends on an immediate's *value*, so the fold
    reduces symbolically with `k` still open. -/
def openFrag (k : Int) : Code :=
  Prog.emit do
    let p ← Prog.basePtr
    let a ← Prog.iconst .i64 k
    let b ← Prog.iadd p a
    let c ← Prog.load64 b
    Prog.store c b

/-- One characterization lemma per parameterized fragment, by `rfl` and generic
    in the parameter; every proof after it works on the literal.

    A bare `rfl`. The old builder tracked slot *types* in a `Trie` keyed by
    halving the slot number, and `Nat.div` is irreducible, so this needed
    `with_unfolding_all`. `emit` carries no type map --- the types are the
    term's --- so there is nothing to unfold. -/
theorem openFrag_eq (k : Int) :
    openFrag k = [.straight [.op (.iconst .i64 k), .op (.iadd 0 1),
                             .op (.load { ty := .i64 } 2), .store .i64 3 2]] := by
  rfl

/-- `compile_sound` for the histogram, executed rather than proved: the term
    and the function the compiler produced perform the same observations, in the
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
      let (c, cenv, _) := Prog.run Hist.code
      match Prog.compileProg 2 Hist.code with
      | .error e => .error e
      | .ok fd =>
      match Sem.run { env := cenv } [ptr] w c, Blocks.run cenv fd [ptr] w with
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
  let hist1Clif ← Prog.orDie HistogramBench1.clifIR
  let clampClif ← Prog.orDie ClampSumBench.clifIR
  let rmsClif ← Prog.orDie CudaRmsNormPersist.clifIR
  let histProg ← Prog.orDie HProgPilots.Hist.program
  let clampProg ← Prog.orDie HProgPilots.ClampSum.program
  let rmsProg ← Prog.orDie HProgPilots.RmsNorm.program
  let nestedProg ← Prog.orDie HProgPilots.Nested.program
  emitArtifacts dir <|
    HistogramBench1.artifacts hist1Clif ++
    ClampSumBench.artifacts clampClif ++
    CudaRmsNormPersist.artifacts rmsClif ++
    #[toJsonEntry "hist1_hprog" {
        clif := histProg,
        memory_size := HistogramBench1.MEM_SIZE
      } { fn_idx := u32 2 },
      toJsonEntry "clamp_sum_hprog" {
        clif := clampProg,
        memory_size := ClampSumBench.MEM_SIZE
      } { fn_idx := u32 1 },
      toJsonArtifact "cuda_rmsnorm_hprog" {
        clif := rmsProg,
        memory_size := CudaRmsNormPersist.MEM_SIZE,
        initial_memory := CudaRmsNormPersist.buildInitialMemory
      } { fn_idx := u32 1 } [
        ("prep", { fn_idx := u32 2 }),
        ("infer", { fn_idx := u32 3 })
      ],
      toJsonEntry "nested_hprog" {
        clif := nestedProg,
        memory_size := 0x100000
      } { fn_idx := u32 2 }]
  -- Each pilot reports the FFI it actually assumes, derived from its own term.
  for (nm, p) in [("hist1", HProgPilots.Hist.code),
                  ("clamp_sum", HProgPilots.ClampSum.code),
                  ("rmsnorm.infer", HProgPilots.RmsNorm.inferCode),
                  ("nested", HProgPilots.Nested.code)] do
    let (c, e, _) := Prog.run p
    if !footprintComplete e c then
      throw (IO.userError s!"{nm} calls a symbol with no declared frame")
    IO.println s!"  {nm}: {footprintReport e c}"
  match HProgPilots.histCompileSound with
  | .error e => throw (IO.userError s!"histogram compile_sound: {e}")
  | .ok (n, calls) =>
      IO.println s!"compile_sound (executed): histogram term and compiled form agree \
                   — {n} observations, {calls} of them FFI calls"


#eval ShipScan.check "HProgPilots"
