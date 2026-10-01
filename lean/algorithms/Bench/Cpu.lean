module
public import Lean
public import AlgorithmLib.Gen
meta import AlgorithmLib.Gen
public import Scan.Ship
meta import Scan.Ship
public import Bench.CpuAsm
meta import Bench.CpuAsm
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.Prog

namespace CpuBench

/-!
  # The CPU benchmark: common workloads as CLIF and as carried machine code

  Each workload leaves an `i64` answer at `outPtr`; `benchmarks/cpu` checks it
  against the same kernel in Rust, then times both.

  * `<w>_clif`: the kernel as CLIF shaped as LLVM's loop is.
  * `<w>_asm`, where Cranelift cannot reach LLVM: LLVM's code (`CpuBenchAsm`),
    mapped once by `asm_load` and called directly, falling back to the CLIF
    when the body did not load.

  Every kernel takes four `i64`s, as a native call does. A kernel writes only
  from `outPtr + 64`.
-/

-- ---------------------------------------------------------------------------
-- The machine code and where it sits
-- ---------------------------------------------------------------------------

/-- Every body this artifact carries (from `CpuBenchAsm`), in slot order. All
    need AVX. -/
def natives : List X86.Body := [CpuBenchAsm.poly, CpuBenchAsm.stream]

#guard natives.length ≤ 32

/-- Each body's slot in `natives`. -/
def POLY_SLOT : Nat := 0
def STREAM_SLOT : Nat := 1

/-- `0x00`--`0x40` are the runtime's context slots. From `SLOT_OFF`, one `i64`
    per body: where `asm_load` mapped it, or 0. At `NAME_OFF`, `"avx"`,
    NUL-terminated, for `cl_cpu_has`. At `POLY_OFF`, poly's five
    coefficients, each as four lanes. From `CODE_OFF`, the bodies, each on a
    64-byte boundary. -/
def SLOT_OFF : Nat := 0x100
def NAME_OFF : Nat := 0x200
def POLY_OFF : Nat := 0x240
def CODE_OFF : Nat := 0x2C0

/-- The layout above holds only while each part fits before what follows it. -/
example : SLOT_OFF + 8 * 32 ≤ NAME_OFF ∧ NAME_OFF + 4 ≤ POLY_OFF ∧
    POLY_OFF + 16 * 5 ≤ CODE_OFF ∧ CODE_OFF % 64 = 0 := by
  decide

/-- A body's bytes, as `X86.assemble` makes them. -/
def bodyBytes (b : X86.Body) : List UInt8 :=
  match X86.assemble b with
  | .ok bs => bs
  | .error e => panic! e

-- Every body assembles: an instruction `X86` does not encode, or a label a
-- body does not define, stops the build here rather than the generator.
#guard natives.all fun b => (X86.assemble b).isOk

def align64 (n : Nat) : Nat := (n + 63) / 64 * 64

/-- Where each body starts. -/
def codeOffs : List Nat :=
  (natives.foldl (fun (acc : List Nat × Nat) b =>
    (acc.1 ++ [acc.2], align64 (acc.2 + (bodyBytes b).length))) ([], CODE_OFF)).1

def MEM_SIZE : Nat :=
  align64 (natives.foldl (fun acc b => align64 (acc + (bodyBytes b).length)) CODE_OFF)

def u64le (x : UInt64) : List UInt8 :=
  (List.range 8).map fun i => (x >>> (8 * i).toUInt64).toUInt8

/-- poly's coefficients, lowest degree first, as the Rust source has them. -/
def POLY : List Float := [1.0, -0.5, 0.25, -0.125, 0.0625]

/-- Each coefficient as an `f32x4` of four copies. -/
def polyBytes : List UInt8 :=
  POLY.flatMap fun c => (List.replicate 4 ((u64le c.toFloat32.toBits.toUInt64).take 4)).flatten

def writeAt (m : List UInt8) (off : Nat) (bs : List UInt8) : List UInt8 :=
  m.take off ++ bs ++ m.drop (off + bs.length)

def initialMemory : List UInt8 :=
  let withCode := (natives.zip codeOffs).foldl (fun m (b, off) => writeAt m off (bodyBytes b))
    (writeAt (zeros MEM_SIZE) NAME_OFF ("avx".toUTF8.toList ++ [0]))
  writeAt withCode POLY_OFF polyBytes

/-- Map every body, on an x86-64 CPU with AVX. The caller's output gets how
    many bodies there are at 0 and each slot at `8 * (1 + slot)`, so it can see
    that all loaded. -/
def asmLoad : Prog V L Unit := do
  let base ← basePtr
  let out ← outPtr
  let one ← iconst32 1
  storeI64 (← iconst64 natives.length) out
  when .eq (← nativeArch) one do
    when .eq (← cpuHas (← iaddImm base NAME_OFF)) one do
      for (b, off, k) in natives.zip (codeOffs.zip (List.range natives.length)) do
        let addr ← nativeLoad (← iaddImm base off) (← iconst64 (bodyBytes b).length)
        storeI64 addr (← iaddImm base (SLOT_OFF + 8 * k))
  for k in List.range natives.length do
    storeI64 (← load64 (← iaddImm base (SLOT_OFF + 8 * k))) (← iaddImm out (8 * (k + 1)))

/-- Run body `slot` on `a b c d` if `asm_load` mapped it, else `fallback`. -/
def viaNative (slot : Nat) (a b c d : V .i64) (fallback : Prog V L (V .i64)) :
    Prog V L (V .i64) := do
  let fn ← load64 (← iaddImm (← basePtr) (SLOT_OFF + 8 * slot))
  let r ← ifte (jTys := [.i64]) .eq fn (← iconst64 0)
    (thn := do return %[← fallback])
    (els := do return %[← nativeCall fn a b c d])
  return r.head

-- ---------------------------------------------------------------------------
-- Loop shapes shared by the kernels
-- ---------------------------------------------------------------------------

/-- The four arguments every kernel and every body takes. -/
abbrev Args (V : ClifTy → Type) := V .i64 × V .i64 × V .i64 × V .i64

/-- `f` at `q = p, p + stride, ...` while `q < e`, rotated: one guard, then the
    test at the bottom. One accumulator; answers where `q` stopped and it. -/
def ptrLoop (p e : V .i64) (stride : Int) (acc0 : V .i64)
    (f : V .i64 → V .i64 → Prog V L (V .i64)) : Prog V L (V .i64 × V .i64) := do
  let r ← dwloop %[p, acc0] .ult e (contOnTrue := true) [0, 1] (guardIdx := some 0)
    (body := fun c => do
      let acc' ← f c.head c.snd
      let q' ← iaddImm c.head stride
      return (q', %[q', acc']))
  return (r.head, r.snd)

/-- Sixteen bytes that need not be aligned. -/
def loadU (a : V .i64) : Prog V L (V .i8x16) := load { ty := .i8x16 } a

-- ---------------------------------------------------------------------------
-- histogram: 256 counters in the output
-- ---------------------------------------------------------------------------

def histFinish (h : V .i64) : Prog V L (V .i64) := do
  forLoopAcc (← iconst64 256) (← iconst64 0) fun j s => do
    iadd s (← imul (← load64 (← iadd h (← ishlImm j 3))) (← iaddImm j 1))

def histZero (h : V .i64) : Prog V L Unit := do
  let zero ← iconst64 0
  forLoop (← iconst64 256) fun j => do storeI64 zero (← iadd h (← ishlImm j 3))

def histBump (h b : V .i64) : Prog V L Unit := do
  let slot ← iadd h (← ishlImm b 3)
  storeI64 (← iaddImm (← load64 slot) 1) slot

/-- Eight bytes a trip, then the rest one at a time: what LLVM does. -/
def histUnr8 (a : Args V) : Prog V L (V .i64) := do
  let (p, n, h, _) := a
  histZero h
  let zero ← iconst64 0
  let end8 ← iadd p (← band n (← iconst64 (-8)))
  let (q, _) ← ptrLoop p end8 8 zero fun q acc => do
    for k in List.range 8 do histBump h (← uload8_64 (← iaddImm q k))
    return acc
  let _ ← ptrLoop q (← iadd p n) 1 zero fun q acc => do
    histBump h (← uload8_64 q)
    return acc
  histFinish h

-- ---------------------------------------------------------------------------
-- mandel: iteration counts over a 128 x 128 grid of the Mandelbrot set
-- ---------------------------------------------------------------------------

def MANDEL : Nat := 128

/-- `u` steps a trip, each with its escape test; the count is tested once a
    trip, `u` dividing 256. -/
def mandelUnr (u : Nat) (_ : Args V) : Prog V L (V .i64) := do
  let d ← fconst64 (3.0 / 128.0)
  let four ← fconst64 4.0
  let zero ← fconst64 0.0
  let none8 ← iconst .i8 0
  let side ← iconst64 MANDEL
  let zero64 ← iconst64 0
  let rows ← dwloop %[zero64, zero64] .ult side (contOnTrue := true) [1] (body := fun cy => do
    let py := cy.head
    let ci ← fadd (← fmul (← fcvtFromSint .f64 py) d) (← fconst64 (-1.5))
    let cols ← dwloop %[zero64, cy.snd] .ult side (contOnTrue := true) [1] (body := fun cx => do
      let px := cx.head
      let total := cx.snd
      let cr ← fadd (← fmul (← fcvtFromSint .f64 px) d) (← fconst64 (-2.0))
      let r ← dwloopL %[← iconst64 0, zero, zero] .ult (← iconst64 256) (contOnTrue := true) [0]
        (body := fun l c => do
          let mut zr := c.snd
          let mut zi := c.thd
          for j in List.range u do
            let zr2 ← fmul zr zr
            let zi2 ← fmul zi zi
            when .ne (← fcmp .gt (← fadd zr2 zi2) four) none8 (brk l %[← iaddImm c.head (Int.ofNat j)])
            zi ← fadd (← fmul (← fadd zr zr) zi) ci
            zr ← fadd (← fsub zr2 zi2) cr
          let k' ← iaddImm c.head (Int.ofNat u)
          return (k', %[k', zr, zi]))
      let px' ← iaddImm px 1
      return (px', %[px', ← iadd total r.head]))
    let py' ← iaddImm py 1
    return (py', %[py', cols.head]))
  return rows.head

-- ---------------------------------------------------------------------------
-- chase: once round a permutation's cycle, summing the slots
-- ---------------------------------------------------------------------------

/-- From slot 0, `n` steps, each load's address the last load's value;
    rotated, as LLVM's loop is. -/
def chaseLoop (a : Args V) : Prog V L (V .i64) := do
  let (p, n, _, _) := a
  let zero ← iconst64 0
  let r ← dwloop %[zero, zero, zero] .ult n (contOnTrue := true) [2] (guardIdx := some 0)
    (body := fun c => do
      let at' ← uload32_64 (← iadd p (← ishlImm c.snd 2))
      let k' ← iaddImm c.head 1
      return (k', %[k', at', ← iadd c.thd at']))
  return r.head

-- ---------------------------------------------------------------------------
-- poly: a degree-4 polynomial at every f32, the XOR of the results' bits
-- ---------------------------------------------------------------------------

/-- The polynomial at the four lanes of `x`, from coefficients `cs` (lowest
    degree first), by Horner as the Rust source writes it. -/
def polyVec (cs : List (V .f32x4)) (x : V .f32x4) : Prog V L (V .f32x4) := do
  match cs.reverse with
  | [] => return x
  | top :: rest => rest.foldlM (fun y c => do fadd (← fmul y x) c) top

/-- Four vectors a trip, bounded by an end pointer, rotated; then one vector a
    trip for the rest. `n` is a multiple of 4, as the input is. The
    coefficients are read from memory once, before the loop. -/
def polySimd (a : Args V) : Prog V L (V .i64) := do
  let (p, n, coef, _) := a
  let cs ← (List.range POLY.length).mapM fun d => do loadF32x4 (← iaddImm coef (16 * Int.ofNat d))
  let h0 ← splat .f32x4 (← fconst32 0.0)
  let end16 ← iadd p (← ishlImm (← band n (← iconst64 (-16))) 2)
  let endN ← iadd p (← ishlImm n 2)
  let r ← dwloop %[p, h0] .ult end16 (contOnTrue := true) [0, 1] (guardIdx := some 0)
    (body := fun c => do
      let mut h := c.snd
      for k in [0, 16, 32, 48] do
        h ← bxor h (← polyVec cs (← loadF32x4 (← iaddImm c.head k)))
      let q' ← iaddImm c.head 64
      return (q', %[q', h]))
  let t ← dwloop %[r.head, r.snd] .ult endN (contOnTrue := true) [1] (guardIdx := some 0)
    (body := fun c => do
      let h ← bxor c.snd (← polyVec cs (← loadF32x4 c.head))
      let q' ← iaddImm c.head 16
      return (q', %[q', h]))
  let h : V .f32x4 := t.head
  let l0 ← bitcast .i32 (← extractlane h 0)
  let l1 ← bitcast .i32 (← extractlane h 1)
  let l2 ← bitcast .i32 (← extractlane h 2)
  let l3 ← bitcast .i32 (← extractlane h 3)
  uextend64 (← bxor (← bxor l0 l1) (← bxor l2 l3))

-- ---------------------------------------------------------------------------
-- stream: 32 MB copied, far past the caches
-- ---------------------------------------------------------------------------

/-- `b[n/2] ^ b[n-1]` of the u64s copied to `o`. -/
def copyAnswer (n o : V .i64) : Prog V L (V .i64) := do
  let mid ← load64 (← iadd o (← ishlImm (← ushrImm n 1) 3))
  let last ← load64 (← iadd o (← ishlImm (← iaddImm n (-1)) 3))
  bxor mid last

/-- 64 bytes a trip in four `i8x16`s, then the rest a u64 at a time. -/
def copySimd (a : Args V) : Prog V L (V .i64) := do
  let (p, n, o, _) := a
  let delta ← isub o p
  let zero ← iconst64 0
  let bytes ← ishlImm n 3
  let end64 ← iadd p (← band bytes (← iconst64 (-64)))
  let endN ← iadd p bytes
  let (q, _) ← ptrLoop p end64 64 zero fun q acc => do
    let d ← iadd q delta
    for k in [0, 16, 32, 48] do
      storeUnaligned (← loadU (← iaddImm q k)) (← iaddImm d k)
    return acc
  let _ ← ptrLoop q endN 8 zero fun q acc => do
    storeI64 (← load64 q) (← iadd q delta)
    return acc
  copyAnswer n o

-- ---------------------------------------------------------------------------
-- Inputs and entries
-- ---------------------------------------------------------------------------

/-- Where a kernel may write: the caller's output from `outPtr + 64`. -/
def scratch : Prog V L (V .i64) := do iaddImm (← outPtr) 64

/-- The input as `count` elements of `2^shift` bytes, the scratch third. -/
def array (shift : Nat) : Prog V L (Args V) := do
  return (← dataPtr, ← ushrImm (← dataLen) shift, ← scratch, ← iconst64 0)

/-- The input as `count` elements of `2^shift` bytes, a table in the
    artifact's memory at `off` third. -/
def arrayWith (shift off : Nat) : Prog V L (Args V) := do
  return (← dataPtr, ← ushrImm (← dataLen) shift, ← iaddImm (← basePtr) off, ← iconst64 0)

/-- No input. -/
def nothing : Prog V L (Args V) := do
  let z ← iconst64 0
  return (z, z, z, z)

abbrev Kernel := Args Slot → Prog Slot Lvl (Slot .i64)

structure Workload where
  name : String
  input : Prog Slot Lvl (Args Slot)
  clif : Kernel
  native : Option Nat := none

/-- An entry that leaves `kernel`'s answer at `outPtr`. -/
def answer (kernel : Prog V L (V .i64)) : Prog V L Unit := do
  storeI64 (← kernel) (← outPtr)

/-- A workload's entries, in function order from `idx`: `<name>_clif`, and
    `<name>_asm` when it carries machine code, falling back to the CLIF. -/
def Workload.fns (w : Workload) (idx : Nat) : List (Except String FuncData) :=
  [Prog.entry s!"{w.name}_clif" (Prog.compileProg idx (do answer (w.clif (← w.input))))]
  ++ match w.native with
    | some slot =>
      [Prog.entry s!"{w.name}_asm" (Prog.compileProg (idx + 1) (do
        let args ← w.input
        let (a, b, c, d) := args
        answer (viaNative slot a b c d (w.clif args))))]
    | none => []

def Workload.count (w : Workload) : Nat := if w.native.isSome then 2 else 1

def workloads : List Workload :=
  [{ name := "histogram", input := array 0, clif := histUnr8 },
   { name := "mandel", input := nothing, clif := mandelUnr 8 },
   { name := "chase", input := array 2, clif := chaseLoop },
   { name := "poly", input := arrayWith 2 POLY_OFF, clif := polySimd, native := some POLY_SLOT },
   -- copy far past the caches, where LLVM's non-temporal stores write a line
   -- without reading it; CLIF's stores all read it first
   { name := "stream", input := array 3, clif := copySimd, native := some STREAM_SLOT }]

/-- The functions before the workloads' entries: `noop` and `asm_load`. -/
def FIRST : Nat := 2

def clifIR : Except String (List FuncData) :=
  let wfns := (workloads.foldl (fun (acc : List (Except String FuncData) × Nat) w =>
    (acc.1 ++ w.fns acc.2, acc.2 + w.count)) ([], FIRST)).1
  Prog.program <|
    [Prog.entry "noop" (.ok noopFunction),
     Prog.entry "asm_load" (Prog.compileProg 1 asmLoad)]
    ++ wfns

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[artifactEntry "cpu_bench" {
    functions := clif,
    required_memory := MEM_SIZE,
    initial_memory := initialMemory
  }]

end CpuBench

def Bench.Cpu.main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  let clif ← Prog.orDie CpuBench.clifIR
  emitArtifacts outDir (CpuBench.artifacts clif)

#eval ShipScan.check "Bench.Cpu" `Bench.Cpu.main