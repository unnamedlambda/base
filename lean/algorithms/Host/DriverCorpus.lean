module
public import Lean
public import Scan.Ship
meta import Scan.Ship
public import AlgorithmLib.Host.Sem
meta import AlgorithmLib.Host.Sem
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
public import Host.VendorRef
meta import Host.VendorRef
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic

@[expose] public section

/-!
# The CUDA driver, checked against the device

`Sem.cudaDrv` states what each driver call a program makes directly does. This
body makes them in one sequence — context, allocation, the three copies and a
fill, two modules, launches on the default stream and a created one, events,
page-locked memory and copies on a stream ordered by an event, cuBLAS on that
stream (a matrix-vector product, products single and batched, `f32` and
`bf16`, a batch over device arrays of pointers, and two it refuses), a launch
and a product captured into a graph that runs twice, frees — storing every
result code and every byte it copies back in the output buffer;
`base/tests/hprog_driver_corpus.rs` runs the artifact on the GPU and compares.

`abiCheck` is the other half: a C++ file whose `static_assert`s hold only if
every driver and cuBLAS function's signature here matches the headers, which
the same test compiles.

Handles and device pointers differ between the model and the machine, and
nothing here stores one: what is compared is what a program can observe of
them, the codes and the bytes.

The kernels are pinned by the interpreter's oracle to what they compute: `main`
adds one to each byte of its buffer and `addk` adds its scalar parameter, which
is what checks that a parameter's value reaches the kernel through
`kernelParams` at the size its PTX declares.
-/

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.Prog
open HProgVendorRef

namespace HProgDriverCorpus

def ptxHeader : String := ".version 7.0\n.target sm_60\n.address_size 64\n\n"

/-- Every thread adds one to the byte at its index. -/
def addOnePtx : String :=
  ptxHeader ++
  ".visible .entry main(.param .u64 p0)\n{\n" ++
  "  .reg .u32 %r<2>;\n  .reg .u64 %rd<4>;\n  .reg .u16 %h<3>;\n" ++
  "  ld.param.u64 %rd1, [p0];\n  cvta.to.global.u64 %rd1, %rd1;\n" ++
  "  mov.u32 %r1, %tid.x;\n  cvt.u64.u32 %rd2, %r1;\n  add.u64 %rd3, %rd1, %rd2;\n" ++
  "  ld.global.u8 %h1, [%rd3];\n  add.u16 %h2, %h1, 1;\n  st.global.u8 [%rd3], %h2;\n" ++
  "  ret;\n}\n"

/-- Every thread adds its scalar parameter to the byte at its index. -/
def addKPtx : String :=
  ptxHeader ++
  ".visible .entry addk(.param .u64 p0, .param .u32 p1)\n{\n" ++
  "  .reg .u32 %r<3>;\n  .reg .u64 %rd<4>;\n  .reg .u16 %h<4>;\n" ++
  "  ld.param.u64 %rd1, [p0];\n  ld.param.u32 %r2, [p1];\n  cvta.to.global.u64 %rd1, %rd1;\n" ++
  "  mov.u32 %r1, %tid.x;\n  cvt.u64.u32 %rd2, %r1;\n  add.u64 %rd3, %rd1, %rd2;\n" ++
  "  ld.global.u8 %h1, [%rd3];\n  cvt.u16.u32 %h3, %r2;\n  add.u16 %h2, %h1, %h3;\n" ++
  "  st.global.u8 [%rd3], %h2;\n  ret;\n}\n"

/-- What the kernels compute, as the interpreter's oracle. -/
def oracle (l : Sem.Launch) (ins : List ByteArray) : List ByteArray :=
  if l.entry == "main" && l.kernel == addOnePtx then ins.map (fun b => ⟨b.data.map (· + 1)⟩)
  else if l.entry == "addk" && l.kernel == addKPtx then
    let k := ((l.args.getD 1 0) &&& 0xff).toUInt8
    ins.map (fun b => ⟨b.data.map (· + k)⟩)
  else ins

-- The arena: slots the driver writes handles into, the kernel parameter
-- array, host data, and the PTX and names as C strings.
def S_DEV : Nat := 0x00
def S_CTX : Nat := 0x08
def S_A : Nat := 0x10
def S_B : Nat := 0x18
def S_Z : Nat := 0x20
def S_MOD : Nat := 0x28
def S_MOD2 : Nat := 0x30
def S_F : Nat := 0x38
def S_K : Nat := 0x40
def S_G : Nat := 0x48
def S_STREAM : Nat := 0x50
def S_KVAL : Nat := 0x58
def S_FREE : Nat := 0x60
def S_TOTAL : Nat := 0x68
def S_E1 : Nat := 0x70
def S_E2 : Nat := 0x78
def S_H : Nat := 0x140
def S_MS : Nat := 0x148
def S_BLAS : Nat := 0x150
def S_GA : Nat := 0x158
def S_GX : Nat := 0x160
def S_GY : Nat := 0x168
def S_IN : Nat := 0x170
def S_OUTD : Nat := 0x178
def PARAMS : Nat := 0x80
def S_GRAPH : Nat := 0x90
def S_EXEC : Nat := 0x98
def S_PARR : Nat := 0xA0
def S_C2 : Nat := 0xA8
/-- A batch's three device arrays of pointers, staged in host memory. -/
def HOSTARR : Nat := 0x1A0
def SRC : Nat := 0x100
def NAME_MAIN : Nat := 0x180
def NAME_ADDK : Nat := 0x188
def NAME_NONE : Nat := 0x190
def PTX1 : Nat := 0x200
def PTX2 : Nat := 0x600
-- cuBLAS: the host scalars, then `sgemv`'s `A`, `x`, `y`, then every product's
-- inputs as one block uploaded to one allocation: two batches of a 2 × 3
-- `f32` `A`, two of a 3 × 2 (or, read transposed, 2 × 3) `B`, and a `bf16` pair.
def ONE : Nat := 0xA00
def ZERO : Nat := 0xA04
def GA : Nat := 0xA10
def GX : Nat := 0xA40
def GY : Nat := 0xA50
def INB : Nat := 0xA60
def MEM : Nat := 0xB00

def gA : List Float := [1, 2, 0, 3,  2, 1, 1, 0,  0, 3, 2, 1]
def gX : List Float := [1, 2, 3]
def gY : List Float := [5, 0, 1, 2]
def sA : List Float := [1, 2, 0, 1, 3, 1,  2, 0, 1, 1, 0, 3]
def sB : List Float := [1, 0, 2, 2, 1, 1,  0, 1, 1, 3, 2, 0]
def hA : List Float := [1, 2, 0, 1, 3, 1]
def hB : List Float := [2, 1, 1, 0, 1, 3]
/-- Where each operand starts in the input allocation, in bytes. -/
def IN_SA : Nat := 0
def IN_SB : Nat := 48
def IN_HA : Nat := 96
def IN_HB : Nat := 108
def IN_LEN : Nat := 128
def OUTD_LEN : Nat := 128

def image : List UInt8 :=
  let pad (xs : List UInt8) (n : Nat) := xs ++ List.replicate (n - xs.length) 0
  let cstr (s : String) := s.toUTF8.toList ++ [0]
  let src := (List.range 64).map (fun i => UInt8.ofNat (3 * i + 1))
  let x := pad (List.replicate SRC 0 ++ src) NAME_MAIN ++ cstr "main"
  let x := pad x NAME_ADDK ++ cstr "addk"
  let x := pad x NAME_NONE ++ cstr "nothere"
  let x := pad x PTX1 ++ cstr addOnePtx
  let x := pad x PTX2 ++ cstr addKPtx
  let x := pad x ONE ++ f32Bytes [1] ++ f32Bytes [0]
  let x := pad x GA ++ f32Bytes gA
  let x := pad x GX ++ f32Bytes gX
  let x := pad x GY ++ f32Bytes gY
  let x := pad x (INB + IN_SA) ++ f32Bytes sA
  let x := pad x (INB + IN_SB) ++ f32Bytes sB
  let x := pad x (INB + IN_HA) ++ bf16Bytes hA
  let x := pad x (INB + IN_HB) ++ bf16Bytes hB
  pad x MEM

/-- The output buffer: result codes from 0, four bytes each; the bytes the
    downloads write from `DATA`. -/
def NCODES : Nat := 104
def DATA : Nat := 4 * NCODES
/-- After the five 64-byte downloads: the products' outputs, then `sgemv`'s `y`. -/
def DATA_OUTD : Nat := DATA + 5 * 64
def DATA_GY : Nat := DATA_OUTD + OUTD_LEN
/-- After the graph ran twice: `A`, and the products again. -/
def DATA_A2 : Nat := DATA_GY + 16
def DATA_OUTD2 : Nat := DATA_A2 + 64
/-- The pointer-array batch's second output, in an allocation of its own. -/
def DATA_C2 : Nat := DATA_OUTD2 + OUTD_LEN
def OUT : Nat := DATA_C2 + 16

def cu (f : CudaFn) (args : Vals V (Ext.cuda f).sig.1) : Prog V L (V .i32) :=
  ext (.cuda f) args

def blas (f : CublasFn) (args : Vals V (Ext.cublas f).sig.1) : Prog V L (V .i32) :=
  ext (.cublas f) args

set_option maxRecDepth 8000 in
def body : Prog V L Unit := do
  let base ← basePtr
  let out ← outPtr
  let at_ (off : Nat) : Prog V L (V .i64) := absAddr base off
  let slot (off : Nat) : Prog V L (V .i64) := do load64 (← at_ off)
  let codes : List (Prog V L (V .i32)) :=
    [ do cu .init %[← iconst32 0]
    , do cu .deviceGet %[← at_ S_DEV, ← iconst32 0]
    , do cu .deviceGet %[← at_ S_Z, ← iconst32 7]
    , do cu .primaryCtxRetain %[← at_ S_CTX, ← load32 (← at_ S_DEV)]
    , do cu .ctxSetCurrent %[← slot S_CTX]
    , do cu .memGetInfo %[← at_ S_FREE, ← at_ S_TOTAL]
    , do cu .memAlloc %[← at_ S_A, ← iconst64 64]
    , do cu .memAlloc %[← at_ S_B, ← iconst64 64]
    , do cu .memAlloc %[← at_ S_Z, ← iconst64 0]
    , do cu .memcpyHtoD %[← slot S_A, ← at_ SRC, ← iconst64 64]
    , do cu .memsetD8 %[← slot S_B, ← iconst32 7, ← iconst64 64]
    , do cu .memsetD8 %[← iadd (← slot S_B) (← iconst64 8), ← iconst32 9, ← iconst64 4]
    , do cu .memcpyDtoD %[← iadd (← slot S_B) (← iconst64 32), ← slot S_A, ← iconst64 16]
    , do cu .moduleLoadData %[← at_ S_MOD, ← at_ PTX1]
    , do cu .moduleLoadData %[← at_ S_MOD2, ← at_ PTX2]
    , do cu .moduleGetFunction %[← at_ S_F, ← slot S_MOD, ← at_ NAME_MAIN]
    , do cu .moduleGetFunction %[← at_ S_G, ← slot S_MOD, ← at_ NAME_NONE]
    , do cu .moduleGetFunction %[← at_ S_K, ← slot S_MOD2, ← at_ NAME_ADDK]
    -- `main` on A: one parameter, the pointer in slot A.
    , do storeI64 (← at_ S_A) (← at_ PARAMS)
         cu .launchKernel %[← slot S_F, ← iconst32 1, ← iconst32 1, ← iconst32 1,
           ← iconst32 64, ← iconst32 1, ← iconst32 1, ← iconst32 0, ← iconst64 0,
           ← at_ PARAMS, ← iconst64 0]
    -- `addk` on B with k = 5: the pointer, then the `u32`.
    , do storeI64 (← iconst64 5) (← at_ S_KVAL)
         storeI64 (← at_ S_B) (← at_ PARAMS)
         storeI64 (← at_ S_KVAL) (← at_ (PARAMS + 8))
         cu .launchKernel %[← slot S_K, ← iconst32 1, ← iconst32 1, ← iconst32 1,
           ← iconst32 64, ← iconst32 1, ← iconst32 1, ← iconst32 0, ← iconst64 0,
           ← at_ PARAMS, ← iconst64 0]
    , do cu .ctxSynchronize %[]
    , do cu .memcpyDtoH %[← iadd out (← iconst64 DATA), ← slot S_A, ← iconst64 64]
    , do cu .memcpyDtoH %[← iadd out (← iconst64 (DATA + 64)), ← slot S_B, ← iconst64 64]
    -- A launch on a created stream, which the host waits for before reading.
    , do cu .streamCreate %[← at_ S_STREAM, ← iconst32 1]
    , do storeI64 (← at_ S_A) (← at_ PARAMS)
         cu .launchKernel %[← slot S_F, ← iconst32 1, ← iconst32 1, ← iconst32 1,
           ← iconst32 64, ← iconst32 1, ← iconst32 1, ← iconst32 0, ← slot S_STREAM,
           ← at_ PARAMS, ← iconst64 0]
    , do cu .streamSynchronize %[← slot S_STREAM]
    , do cu .memcpyDtoH %[← iadd out (← iconst64 (DATA + 128)), ← slot S_A, ← iconst64 64]
    , do cu .memcpyDtoH %[← iadd out (← iconst64 (DATA + 192)), ← iadd (← slot S_B) (← iconst64 30),
           ← iconst64 20]
    -- Events, page-locked memory and copies on the stream: B comes down into
    -- page-locked H, goes up again into A, and the default stream reads A
    -- only once it has waited for an event recorded after that upload.
    , do cu .eventCreate %[← at_ S_E1, ← iconst32 0]
    , do cu .eventCreate %[← at_ S_E2, ← iconst32 0]
    , do cu .eventElapsedTime %[← at_ S_MS, ← slot S_E1, ← slot S_E2]
    , do cu .memAllocHost %[← at_ S_H, ← iconst64 64]
    , do cu .eventRecord %[← slot S_E1, ← slot S_STREAM]
    , do cu .memcpyDtoHAsync %[← slot S_H, ← slot S_B, ← iconst64 64, ← slot S_STREAM]
    , do cu .eventRecord %[← slot S_E2, ← slot S_STREAM]
    , do cu .eventSynchronize %[← slot S_E2]
    , do cu .eventElapsedTime %[← at_ S_MS, ← slot S_E1, ← slot S_E2]
    , do cu .memcpyHtoDAsync %[← slot S_A, ← slot S_H, ← iconst64 64, ← slot S_STREAM]
    , do cu .eventRecord %[← slot S_E1, ← slot S_STREAM]
    , do cu .streamWaitEvent %[← iconst64 0, ← slot S_E1, ← iconst32 0]
    , do cu .memcpyDtoH %[← iadd out (← iconst64 (DATA + 256)), ← slot S_A, ← iconst64 64]
    , do cu .eventSynchronize %[← slot S_E1]
    , do cu .memFreeHost %[← slot S_H]
    , do cu .eventDestroy %[← slot S_E1]
    , do cu .eventDestroy %[← slot S_E2]
    -- cuBLAS on the stream. The operands are uploaded and the outputs zeroed
    -- on the default stream, which the host waits for first.
    , do cu .memAlloc %[← at_ S_GA, ← iconst64 48]
    , do cu .memAlloc %[← at_ S_GX, ← iconst64 12]
    , do cu .memAlloc %[← at_ S_GY, ← iconst64 16]
    , do cu .memAlloc %[← at_ S_IN, ← iconst64 IN_LEN]
    , do cu .memAlloc %[← at_ S_OUTD, ← iconst64 OUTD_LEN]
    , do cu .memcpyHtoD %[← slot S_GA, ← at_ GA, ← iconst64 48]
    , do cu .memcpyHtoD %[← slot S_GX, ← at_ GX, ← iconst64 12]
    , do cu .memcpyHtoD %[← slot S_GY, ← at_ GY, ← iconst64 16]
    , do cu .memcpyHtoD %[← slot S_IN, ← at_ INB, ← iconst64 IN_LEN]
    , do cu .memsetD8 %[← slot S_OUTD, ← iconst32 0, ← iconst64 OUTD_LEN]
    , do cu .ctxSynchronize %[]
    , do blas .create %[← at_ S_BLAS]
    , do blas .setStream %[← slot S_BLAS, ← slot S_STREAM]
    -- y = A x + y, A 4 × 3
    , do blas .sgemv %[← slot S_BLAS, ← iconst32 0, ← iconst32 4, ← iconst32 3, ← at_ ONE,
           ← slot S_GA, ← iconst32 4, ← slot S_GX, ← iconst32 1, ← at_ ONE, ← slot S_GY, ← iconst32 1]
    -- C0 = A0 B0, 2 × 3 by 3 × 2
    , do blas .sgemm %[← slot S_BLAS, ← iconst32 0, ← iconst32 0, ← iconst32 2, ← iconst32 2,
           ← iconst32 3, ← at_ ONE, ← iadd (← slot S_IN) (← iconst64 IN_SA), ← iconst32 2,
           ← iadd (← slot S_IN) (← iconst64 IN_SB), ← iconst32 3, ← at_ ZERO, ← slot S_OUTD, ← iconst32 2]
    -- refused: an unknown operation, and a leading dimension shorter than A
    , do blas .sgemm %[← slot S_BLAS, ← iconst32 5, ← iconst32 0, ← iconst32 2, ← iconst32 2,
           ← iconst32 3, ← at_ ONE, ← slot S_IN, ← iconst32 2, ← slot S_IN, ← iconst32 3,
           ← at_ ZERO, ← slot S_OUTD, ← iconst32 2]
    , do blas .sgemm %[← slot S_BLAS, ← iconst32 0, ← iconst32 0, ← iconst32 2, ← iconst32 2,
           ← iconst32 3, ← at_ ONE, ← slot S_IN, ← iconst32 1, ← slot S_IN, ← iconst32 3,
           ← at_ ZERO, ← slot S_OUTD, ← iconst32 2]
    -- two batches, B read transposed (stored 2 × 3), into C at 16
    , do blas .sgemmStridedBatched %[← slot S_BLAS, ← iconst32 0, ← iconst32 1, ← iconst32 2,
           ← iconst32 2, ← iconst32 3, ← at_ ONE, ← iadd (← slot S_IN) (← iconst64 IN_SA), ← iconst32 2,
           ← iconst64 6, ← iadd (← slot S_IN) (← iconst64 IN_SB), ← iconst32 2, ← iconst64 6, ← at_ ZERO,
           ← iadd (← slot S_OUTD) (← iconst64 16), ← iconst32 2, ← iconst64 4, ← iconst32 2]
    -- bf16 in, f32 out, into C at 48
    , do blas .gemmEx %[← slot S_BLAS, ← iconst32 0, ← iconst32 0, ← iconst32 2, ← iconst32 2,
           ← iconst32 3, ← at_ ONE, ← iadd (← slot S_IN) (← iconst64 IN_HA), ← iconst32 14, ← iconst32 2,
           ← iadd (← slot S_IN) (← iconst64 IN_HB), ← iconst32 14, ← iconst32 3, ← at_ ZERO,
           ← iadd (← slot S_OUTD) (← iconst64 48), ← iconst32 0, ← iconst32 2, ← iconst32 68,
           ← iconst32 (-1)]
    -- the same, A read transposed (stored 3 × 2), one batch, into C at 64
    , do blas .gemmStridedBatchedEx %[← slot S_BLAS, ← iconst32 1, ← iconst32 0, ← iconst32 2,
           ← iconst32 2, ← iconst32 3, ← at_ ONE, ← iadd (← slot S_IN) (← iconst64 IN_HA), ← iconst32 14,
           ← iconst32 3, ← iconst64 6, ← iadd (← slot S_IN) (← iconst64 IN_HB), ← iconst32 14,
           ← iconst32 3, ← iconst64 6, ← at_ ZERO, ← iadd (← slot S_OUTD) (← iconst64 64), ← iconst32 0,
           ← iconst32 2, ← iconst64 4, ← iconst32 1, ← iconst32 68, ← iconst32 (-1)]
    -- A batch over device arrays of pointers: both members' A and B from the
    -- inputs, C into the products at 96 and into an allocation of its own.
    -- The arrays are uploaded on the default stream, which the host waits for.
    , do cu .memAlloc %[← at_ S_PARR, ← iconst64 48]
    , do cu .memAlloc %[← at_ S_C2, ← iconst64 16]
    , do let ptrs := [(S_IN, IN_SA), (S_IN, IN_SA + 24), (S_IN, IN_SB), (S_IN, IN_SB + 24),
           (S_OUTD, 96), (S_C2, 0)]
         for ((sl, off), i) in ptrs.zipIdx do
           storeI64 (← iadd (← slot sl) (← iconst64 off)) (← at_ (HOSTARR + 8 * i))
         cu .memcpyHtoD %[← slot S_PARR, ← at_ HOSTARR, ← iconst64 48]
    , do blas .sgemmBatched %[← slot S_BLAS, ← iconst32 0, ← iconst32 0, ← iconst32 2, ← iconst32 2,
           ← iconst32 3, ← at_ ONE, ← slot S_PARR, ← iconst32 2, ← iadd (← slot S_PARR) (← iconst64 16),
           ← iconst32 3, ← at_ ZERO, ← iadd (← slot S_PARR) (← iconst64 32), ← iconst32 2, ← iconst32 2]
    , do cu .streamSynchronize %[← slot S_STREAM]
    , do cu .memcpyDtoH %[← iadd out (← iconst64 DATA_OUTD), ← slot S_OUTD, ← iconst64 OUTD_LEN]
    , do cu .memcpyDtoH %[← iadd out (← iconst64 DATA_C2), ← slot S_C2, ← iconst64 16]
    , do cu .memcpyDtoH %[← iadd out (← iconst64 DATA_GY), ← slot S_GY, ← iconst64 16]
    -- A graph: `main` on A and C = A0 B0 into the products at 80, captured on
    -- the stream, instantiated, and launched twice.
    , do cu .beginCapture %[← slot S_STREAM, ← iconst32 2]
    , do storeI64 (← at_ S_A) (← at_ PARAMS)
         cu .launchKernel %[← slot S_F, ← iconst32 1, ← iconst32 1, ← iconst32 1,
           ← iconst32 64, ← iconst32 1, ← iconst32 1, ← iconst32 0, ← slot S_STREAM,
           ← at_ PARAMS, ← iconst64 0]
    , do blas .sgemm %[← slot S_BLAS, ← iconst32 0, ← iconst32 0, ← iconst32 2, ← iconst32 2,
           ← iconst32 3, ← at_ ONE, ← iadd (← slot S_IN) (← iconst64 IN_SA), ← iconst32 2,
           ← iadd (← slot S_IN) (← iconst64 IN_SB), ← iconst32 3, ← at_ ZERO,
           ← iadd (← slot S_OUTD) (← iconst64 80), ← iconst32 2]
    , do cu .endCapture %[← slot S_STREAM, ← at_ S_GRAPH]
    , do cu .graphInstantiate %[← at_ S_EXEC, ← slot S_GRAPH, ← iconst64 0]
    , do cu .graphLaunch %[← slot S_EXEC, ← slot S_STREAM]
    , do cu .graphLaunch %[← slot S_EXEC, ← slot S_STREAM]
    , do cu .streamSynchronize %[← slot S_STREAM]
    , do cu .memcpyDtoH %[← iadd out (← iconst64 DATA_A2), ← slot S_A, ← iconst64 64]
    , do cu .memcpyDtoH %[← iadd out (← iconst64 DATA_OUTD2), ← slot S_OUTD, ← iconst64 OUTD_LEN]
    , do cu .graphExecDestroy %[← slot S_EXEC]
    , do cu .graphDestroy %[← slot S_GRAPH]
    , do blas .destroy %[← slot S_BLAS]
    , do cu .memFree %[← slot S_GA]
    , do cu .memFree %[← slot S_GX]
    , do cu .memFree %[← slot S_GY]
    , do cu .memFree %[← slot S_IN]
    , do cu .memFree %[← slot S_OUTD]
    , do cu .memFree %[← slot S_PARR]
    , do cu .memFree %[← slot S_C2]
    , do cu .streamDestroy %[← slot S_STREAM]
    , do cu .memFree %[← slot S_A]
    , do cu .memFree %[← slot S_B]
    , do cu .moduleUnload %[← slot S_MOD]
    , do cu .moduleUnload %[← slot S_MOD2]
    , do cu .primaryCtxRelease %[← load32 (← at_ S_DEV)]
    -- the device ordinal the driver wrote, and the presence probe
    , do load32 (← at_ S_DEV)
    , do libPresent .cuda ]
  for (c, k) in codes.zipIdx do
    let r ← c
    storeI32 r (← iadd out (← iconst64 (4 * k)))

def checked : Except String Code := Prog.emitChecked body
def code : Code := Prog.emit body
def program : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 body)]
def env : FnEnv := (Prog.run body).2.1

def entryArgVals : List Sem.V :=
  [.sc .i64 (Sem.regionBase .arena), .sc .i64 (Sem.regionBase .data),
   .sc .i64 8, .sc .i64 (Sem.regionBase .out), .sc .i64 OUT.toUInt64]

def startWorld : Sem.World :=
  { mem := { arena := ⟨image.toArray⟩, data := ByteArray.mk (Array.replicate 8 0),
             out := ByteArray.mk (Array.replicate OUT 0) },
    kernel := oracle, vendor := vendorRef }

def expected : Except String ByteArray :=
  match Sem.run { env } entryArgVals startWorld code with
  | .stuck why | .misuse why | .fault why => .error why
  | .ok _ w => .ok w.mem.out

/-- The names of the codes, in the order the body stores them. -/
def codeNames : List String :=
  [ "cuInit", "cuDeviceGet 0", "cuDeviceGet 7 is an invalid device", "retain the primary context",
    "make it current", "cuMemGetInfo", "alloc A", "alloc B", "alloc 0 is an invalid value",
    "HtoD A", "memset B", "memset B+8", "DtoD B+32 from A", "load module main", "load module addk",
    "get main", "get a name the module lacks", "get addk", "launch main on A", "launch addk 5 on B",
    "ctx synchronize", "DtoH A", "DtoH B", "create a non-blocking stream", "launch main on the stream",
    "stream synchronize", "DtoH A again", "DtoH B+30",
    "create event 1", "create event 2", "elapsed before recording is an invalid handle",
    "alloc page-locked H", "record 1 on the stream", "DtoH async B into H", "record 2 on the stream",
    "synchronize event 2", "elapsed between 1 and 2", "HtoD async H into A", "record 1 again",
    "default stream waits on 1", "DtoH A on the default stream", "synchronize event 1", "free H",
    "destroy event 1", "destroy event 2",
    "alloc GA", "alloc GX", "alloc GY", "alloc IN", "alloc OUTD", "HtoD GA", "HtoD GX", "HtoD GY",
    "HtoD IN", "zero OUTD", "ctx synchronize before cuBLAS", "cublasCreate", "cublasSetStream",
    "sgemv", "sgemm", "sgemm with an unknown operation", "sgemm with lda too short",
    "sgemmStridedBatched", "gemmEx bf16", "gemmStridedBatchedEx bf16 transposed",
    "alloc the pointer arrays", "alloc C2", "HtoD the pointer arrays", "sgemmBatched",
    "synchronize the stream after cuBLAS", "DtoH the products", "DtoH C2", "DtoH y",
    "begin capture", "capture main on A", "capture sgemm", "end capture", "instantiate",
    "launch the graph", "launch the graph again", "synchronize after the graphs", "DtoH A after",
    "DtoH the products after", "destroy the instantiated graph", "destroy the graph", "cublasDestroy",
    "free GA", "free GX", "free GY", "free IN", "free OUTD", "free the pointer arrays", "free C2",
    "destroy the stream", "free A", "free B",
    "unload main", "unload addk", "release the primary context", "the device ordinal written",
    "cuda present" ]

/-- A C++ translation unit that compiles only if every driver and cuBLAS
    function's signature in `Ext.sig` matches `cuda.h` and `cublas_v2.h`: as
    many parameters, each of the width the table says, and a 32-bit result.
    An `i32` also stands for a narrower integer, which the C calling
    convention passes promoted. Where the header declares C++ overloads too,
    `Ext.cParams` selects the C declaration. -/
def abiCheck : String :=
  let width : ClifTy → String
    | .i32 => "32" | .i64 => "64" | _ => "0"
  let header :=
    "#include <cuda.h>\n#include <cublas_v2.h>\n#include <type_traits>\n\n" ++
    "template <class T> constexpr int w() {\n" ++
    "  return std::is_floating_point_v<T> ? -int(sizeof(T) * 8) : int(sizeof(T) * 8);\n}\n" ++
    "template <int P, class T> constexpr bool fits() {\n" ++
    "  return P == 32 ? (w<T>() > 0 && w<T>() <= 32) : w<T>() == P;\n}\n" ++
    "template <int... P> struct L {};\n" ++
    "template <class R, class... A, int... P>\nconstexpr bool same(R (*)(A...), L<P...>) {\n" ++
    "  if constexpr (sizeof...(A) != sizeof...(P)) return false;\n" ++
    "  else return (fits<P, A>() && ...) && fits<32, R>();\n}\n\n"
  let exts := CudaFn.all.map Ext.cuda ++ CublasFn.all.map Ext.cublas
  header ++ String.join (exts.map fun e =>
    let ps := ", ".intercalate (e.sig.1.map width)
    let fn := match e.cParams with
      | some cs => "static_cast<cublasStatus_t (*)(" ++ ", ".intercalate cs ++ ")>(&" ++ e.symbol ++ ")"
      | none => "&" ++ e.symbol
    "static_assert(same(" ++ fn ++ ", L<" ++ ps ++ ">{}), \"" ++ e.symbol ++ "\");\n")

end HProgDriverCorpus

open AlgorithmLib in
def Host.DriverCorpus.main (args : List String) : IO Unit := do
  let dir ← requireOutputDir args
  match HProgDriverCorpus.checked with
  | .error e => throw (IO.userError s!"the driver corpus body is not well-formed: {e}")
  | .ok _ => pure ()
  if HProgDriverCorpus.codeNames.length > HProgDriverCorpus.NCODES then
    throw (IO.userError "more driver results than NCODES leaves room for")
  match HProgDriverCorpus.expected with
  | .error e => throw (IO.userError s!"interpreting the driver corpus: {e}")
  | .ok bytes =>
      let clif ← AlgorithmLib.Prog.orDie HProgDriverCorpus.program
      emitArtifacts dir #[artifactEntry "hprog_driver_corpus" {
        functions := clif, required_memory := HProgDriverCorpus.MEM,
        initial_memory := HProgDriverCorpus.image
      }]
      let sideDir := System.FilePath.mk dir / "hprog_driver_corpus"
      IO.FS.createDirAll sideDir
      IO.FS.writeFile (sideDir / "expected.json")
        (Lean.Json.mkObj [("expected", Lean.toJson (bytes.toList.map (·.toNat))),
                          ("names", Lean.toJson HProgDriverCorpus.codeNames),
                          ("data", Lean.toJson HProgDriverCorpus.DATA)]).compress
      IO.FS.writeFile (sideDir / "abi_check.cpp") HProgDriverCorpus.abiCheck
      IO.println s!"driver corpus: {bytes.size} expected bytes"

#eval ShipScan.check "Host.DriverCorpus" `Host.DriverCorpus.main
