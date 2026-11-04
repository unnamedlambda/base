module
public import Scan.Ship
meta import Scan.Ship
public import AlgorithmLib.Surface.FFI
meta import AlgorithmLib.Surface.FFI
public import AlgorithmLib.Kernel.Block
meta import AlgorithmLib.Kernel.Block
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
public import AlgorithmLib.Surface.Drv
meta import AlgorithmLib.Surface.Drv
public import AlgorithmLib.Host.Driver
meta import AlgorithmLib.Host.Driver
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

set_option maxRecDepth 4096
open Lean (Json toJson)
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg

namespace Matmul

-- ===========================================================================
-- Dependent-type interface for matrix multiplication.
--
-- The Matrix type carries its shape at the type level. Incompatible shapes
-- are rejected at Lean elaboration time, before any PTX or CLIF is generated.
--
--   matmul {m k n} (A : Matrix m k) (B : Matrix k n) : Artifact × UInt32
-- ===========================================================================

structure Matrix (m n : Nat) : Type where
  mk ::

def mkMatrix (m n : Nat) : Matrix m n := Matrix.mk

-- ---------------------------------------------------------------------------
-- Deterministic PRNG: xorshift32
-- ---------------------------------------------------------------------------

def xorshift32 (s : UInt32) : UInt32 :=
  let s := s ^^^ (s <<< 13)
  let s := s ^^^ (s >>> 17)
  s ^^^ (s <<< 5)

-- f32 bit pattern in ±[0.5, 1.0):
--   sign bit from MSB, fixed biased exponent 126 (= 2^-1), 23-bit mantissa
def randF32Bits (s : UInt32) : UInt32 :=
  let sign := s &&& 0x80000000
  let mantissa := s &&& 0x007FFFFF
  let exp : UInt32 := (126 : UInt32) <<< 23
  sign ||| exp ||| mantissa

-- Tail-recursive matrix byte generator.
-- Pre-allocates a ByteArray and writes little-endian f32 bit patterns.
def genMatrixBytesBA (count : Nat) (seed : UInt32) : ByteArray :=
  let rec go : Nat → UInt32 → ByteArray → ByteArray
    | 0, _, acc => acc
    | Nat.succ n, s, acc =>
      let s' := xorshift32 s
      let bits := randF32Bits s'
      let v := bits.toNat
      let acc := acc.push (UInt8.ofNat (v &&& 0xFF))
      let acc := acc.push (UInt8.ofNat ((v >>> 8) &&& 0xFF))
      let acc := acc.push (UInt8.ofNat ((v >>> 16) &&& 0xFF))
      let acc := acc.push (UInt8.ofNat ((v >>> 24) &&& 0xFF))
      go n s' acc
  go count seed (ByteArray.mk (Array.mkEmpty (count * 4)))

def genMatrixBytes (count : Nat) (seed : UInt32) : List UInt8 :=
  (genMatrixBytesBA count seed).toList

namespace Body
open AlgorithmLib.Prog AlgorithmLib.Kernel AlgorithmLib.Kernel.Block

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

/-- **Tiled GEMM, `C = A·B`, a 16×16 tile of `C` a block.** `M`, `K`, `N`
    are the first three words at `par`. A block is 256 threads, thread `t` at
    `(tx, ty) = (t mod 16, t / 16)` within the tile, and block `b` takes tile
    `(b mod gx, b / gx)` for `gx = ⌈N / 16⌉`. For each 16 columns of `A`
    (rows of `B`) every thread stages one element of each tile in shared
    memory, zero past the matrix's edge, then accumulates its row of the `A`
    tile against its column of the `B` tile by fused multiply-adds. -/
def matmulBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64])
  let cta := ps.head; let tid := ps.snd; let sm := (ps.drop 3).head
  let pA := (ps.drop 4).head; let pB := (ps.drop 5).head
  let pC := (ps.drop 6).head; let par := (ps.drop 7).head
  let m ← load32 par
  let k ← load32 (← at_ par 4)
  let n ← load32 (← at_ par 8)
  let gx ← ushrImm (← iadd n (← c32 15)) 4
  let tx ← band tid (← c32 15)
  let ty ← ushrImm tid 4
  let by_ ← udiv cta gx
  let row ← iadd (← ishlImm by_ 4) ty
  let col ← iadd (← ishlImm (← isub cta (← imul by_ gx)) 4) tx
  let aT ← at4 sm tid
  let bT ← at_ aT 1024
  let aRow ← at4 sm (← ishlImm ty 4)
  let bCol ← at4 (← at_ sm 1024) tx
  let rowIn ← icmp .ult row m
  let colIn ← icmp .ult col n
  let zero ← f32c 0
  let r ← wloop %[← c32 0, zero] (fun cs => pure (contIfULt cs.head k, %[cs.snd], ()))
    (fun cs _ => do
      let t := cs.head
      let ka ← iadd t tx
      let okA ← band rowIn (← icmp .ult ka k)
      let ia ← select okA (← iadd (← imul row k) ka) (← c32 0)
      store (← select okA (← loadF32 (← at4 pA ia)) zero) aT
      let kb ← iadd t ty
      let okB ← band (← icmp .ult kb k) colIn
      let ib ← select okB (← iadd (← imul kb n) col) (← c32 0)
      store (← select okB (← loadF32 (← at4 pB ib)) zero) bT
      callLocalVoid barrier %[]
      let mut acc := cs.snd
      for j in List.range 16 do
        acc ← fma (← loadF32 (← at_ aRow (4 * j))) (← loadF32 (← at_ bCol (64 * j))) acc
      callLocalVoid barrier %[]
      pure %[← iadd t (← c32 16), acc])
  when .ne (← band rowIn colIn) (← iconst .i8 0) do
    store r.head (← at4 pC (← iadd (← imul row n) col))

end

def kernel : Kernel :=
  { name := "main", params := [.i64, .i64, .i64, .i64],
    code := emit matmulBody (idParams ++ [.i64, .i64, .i64, .i64]), smem := 2048 }

end Body

/-- The kernel's module, or empty where the printer refuses it. -/
def ptxSource : String :=
  match Body.kernel.ptx with
  | .ok s => s
  | .error _ => ""

-- ---------------------------------------------------------------------------
-- Layout constants (fixed region offsets; data region base depends on m,k,n)
-- ---------------------------------------------------------------------------

def PTX_OFF : Nat := 0x0100
def PTX_REGION : Nat := 8192
def BIND_OFF : Nat := PTX_OFF + PTX_REGION            -- 0x2100
def PARAMS_OFF : Nat := BIND_OFF + 16                  -- 0x2110
def OUTPUT_FN_OFF : Nat := PARAMS_OFF + 16             -- 0x2120
def OUTPUT_FN_REGION : Nat := 256
def DATA_OFF : Nat := 0x3000

-- ---------------------------------------------------------------------------
-- Payload: PTX source, bind desc, params, output filename, random A & B data
-- ---------------------------------------------------------------------------

/-- Where the driver's state starts (`Drv.App`). -/
def ORG : Nat := 0x2300

/-- The module prints and fits its region with its terminating zero. -/
theorem ptx_fits : 0 < ptxSource.utf8ByteSize ∧ ptxSource.utf8ByteSize < PTX_REGION := by
  native_decide

/-- The signature the program's typestate states of its kernel: `main` takes
    four pointers. -/
theorem matmul_sig : AlgorithmLib.HProg.Sem.ptxParamSizes ptxSource "main" = some [8, 8, 8, 8] := by native_decide

/-- **The program in the driver's standard layout**: `A`, `B`, `C` and the
    dimensions as four allocations, and one kernel over them, a block of 256
    threads per 16×16 tile of `C`. -/
def app (m k n : Nat) : Drv.App :=
  { bufs := [m * k * 4, k * n * 4, m * n * 4, 16],
    kernels := [⟨PTX_OFF, PTX_REGION, [0, 1, 2, 3], ((n + 15) / 16 * ((m + 15) / 16), 1, 1), (256, 1, 1), []⟩],
    memSize := DATA_OFF + (m * k + k * n + m * n) * 4, org := ORG }

def buildPayload (m k n : Nat) : List UInt8 :=
  -- params [M, K, N, pad]
  let params :=
    uint32ToBytes (UInt32.ofNat m) ++
    uint32ToBytes (UInt32.ofNat k) ++
    uint32ToBytes (UInt32.ofNat n) ++
    uint32ToBytes 0
  -- Random matrices: distinct seeds so A ≠ B
  (app m k n).initialMemory [ptxSource]
    [(PARAMS_OFF, params), (OUTPUT_FN_OFF, stringToBytes "matmul_output.bin"),
     (DATA_OFF, genMatrixBytes (m * k) 0xC0FFEE01),
     (DATA_OFF + m * k * 4, genMatrixBytes (k * n) 0xDEADBEEF)]

-- ---------------------------------------------------------------------------
-- CLIF orchestrator: bring the driver up, make the four allocations, upload,
-- launch, download, free, write the output file. No timestep loop — one
-- kernel, one result.
-- ---------------------------------------------------------------------------

open AlgorithmLib.Prog


def code (m k n : Nat) : Prog V L Unit :=
  let aBytes := m * k * 4
  let bBytes := k * n * 4
  let cBytes := m * n * 4
  let aOff := DATA_OFF
  let bOff := aOff + aBytes
  let cOff := bOff + bBytes
  let a := app m k n
  do
    let base ← basePtr
    a.setup base
    a.upload base 0 (← iaddImm base aOff) aBytes
    a.upload base 1 (← iaddImm base bOff) bBytes
    a.upload base 3 (← iaddImm base PARAMS_OFF) 16
    a.run base 0
    let cOffV ← iaddImm base cOff
    a.download base cOffV 2 cBytes
    for i in List.range 4 do
      Drv.free base (a.bufSlot i)

    -- Write output file: bytes [cOff .. cOff + cBytes)
    let fnOffV ← iconst64 OUTPUT_FN_OFF
    let _ ← replaceFileAt base fnOffV (← iconst64 OUTPUT_FN_REGION) (← iconst64 cOff) (← iconst64 cBytes)


def clifIrSource (m k n : Nat) : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 (code m k n))]

-- ---------------------------------------------------------------------------
-- Monomorphic builder: takes concrete dims, returns (Artifact, Algorithm).
-- ---------------------------------------------------------------------------

def buildMatmulConfig (m k n : Nat) (clif : List FuncData) : Artifact × UInt32 :=
  let payload := buildPayload m k n
  let memSize := payload.length
  let cfg : Artifact := {
    functions := clif,
    required_memory := memSize,
    initial_memory := payload
  }
  let alg : UInt32 := IR.mainFnIdx
  (cfg, alg)

-- ---------------------------------------------------------------------------
-- Typed matmul: the dependent-type surface.
-- Passing incompatible shapes fails at Lean elaboration time.
-- ---------------------------------------------------------------------------

def matmul {m k n : Nat} (_A : Matrix m k) (_B : Matrix k n)
    (clif : List FuncData) : Artifact × UInt32 :=
  buildMatmulConfig m k n clif

-- ---------------------------------------------------------------------------
-- Demo: shapes fixed at Lean compile time.
-- Change M, K, N to generate a different sized matmul (type-checked).
-- ---------------------------------------------------------------------------

def M : Nat := 1024
def K : Nat := 512
def N : Nat := 256

def A : Matrix M K := mkMatrix M K
def B : Matrix K N := mkMatrix K N

/-- What ships, at the dimensions the types above fix. -/
def result : Except String (Artifact × UInt32) := do
  return matmul A B (← clifIrSource M K N)   -- : Matrix M N (erased)

-- Uncomment to see the dependent-type check in action:
--
--   def bad : Artifact × UInt32 := matmul A (mkMatrix 128 128)
--   -- error: type mismatch
--   --   mkMatrix 128 128 has type Matrix 128 128
--   --   but is expected to have type Matrix K ?n

end Matmul

def Demo.Matmul.main (args : List String) : IO Unit := do
  let (cfg, alg) ← Prog.orDie Matmul.result
  let outDir ← requireOutputDir args
  emitArtifacts outDir #[artifactEntry "matmul_app" cfg]

#eval ShipScan.check "Demo.Matmul" `Demo.Matmul.main
