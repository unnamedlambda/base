import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace CudaRmsNormPersist

def PTX_SOURCE_OFF : Nat := 0x0100
def BIND_DESC_OFF  : Nat := 0x1100
def MEM_SIZE       : Nat := 0x1200
def TIMEOUT_MS     : Nat := 30000

-- App fields in shared memory (after 0x38 reserved header)
def N_OFF      : Nat := 0x38   -- i64: element count N
def BUF0_OFF   : Nat := 0x40   -- i32: buf0 id (x + weights)
def BUF1_OFF   : Nat := 0x44   -- i32: buf1 id (output)

def ptxSource : String :=
  ".version 8.0\n" ++
  ".target sm_86\n" ++
  ".address_size 64\n" ++
  "\n" ++
  ".shared .align 4 .b8 _smem[36];\n" ++
  "\n" ++
  ".visible .entry main(\n" ++
  "    .param .u64 buf0,\n" ++
  "    .param .u64 buf1\n" ++
  ")\n" ++
  "{\n" ++
  "    .reg .pred   %p;\n" ++
  "    .reg .u32    %r<12>;\n" ++
  "    .reg .u64    %rd<10>;\n" ++
  "    .reg .f32    %f<10>;\n" ++
  "\n" ++
  "    ld.param.u64  %rd0, [buf0];\n" ++
  "    ld.param.u64  %rd1, [buf1];\n" ++
  "    ld.global.u32 %r0, [%rd0];\n" ++
  "    mov.u32       %r1, %tid.x;\n" ++
  "    shr.u32       %r2, %r1, 5;\n" ++
  "    and.b32       %r3, %r1, 31;\n" ++
  "    add.u64       %rd2, %rd0, 8;\n" ++
  "    cvt.u64.u32   %rd3, %r0;\n" ++
  "    shl.b64       %rd3, %rd3, 2;\n" ++
  "    add.u64       %rd4, %rd2, %rd3;\n" ++
  "    mov.f32       %f0, 0f00000000;\n" ++
  "    mov.u32       %r4, %r1;\n" ++
  "loop1:\n" ++
  "    setp.ge.u32   %p, %r4, %r0;\n" ++
  "    @%p bra       done1;\n" ++
  "    cvt.u64.u32   %rd5, %r4;\n" ++
  "    shl.b64       %rd5, %rd5, 2;\n" ++
  "    add.u64       %rd6, %rd2, %rd5;\n" ++
  "    ld.global.f32 %f1, [%rd6];\n" ++
  "    fma.rn.f32    %f0, %f1, %f1, %f0;\n" ++
  "    add.u32       %r4, %r4, 256;\n" ++
  "    bra           loop1;\n" ++
  "done1:\n" ++
  "    shfl.sync.bfly.b32 %f2, %f0, 16, 31, 0xffffffff;\n" ++
  "    add.f32       %f0, %f0, %f2;\n" ++
  "    shfl.sync.bfly.b32 %f2, %f0, 8, 31, 0xffffffff;\n" ++
  "    add.f32       %f0, %f0, %f2;\n" ++
  "    shfl.sync.bfly.b32 %f2, %f0, 4, 31, 0xffffffff;\n" ++
  "    add.f32       %f0, %f0, %f2;\n" ++
  "    shfl.sync.bfly.b32 %f2, %f0, 2, 31, 0xffffffff;\n" ++
  "    add.f32       %f0, %f0, %f2;\n" ++
  "    shfl.sync.bfly.b32 %f2, %f0, 1, 31, 0xffffffff;\n" ++
  "    add.f32       %f0, %f0, %f2;\n" ++
  "    setp.ne.u32   %p, %r3, 0;\n" ++
  "    @%p bra       skip1;\n" ++
  "    mov.u32       %r5, _smem;\n" ++
  "    shl.b32       %r6, %r2, 2;\n" ++
  "    add.u32       %r5, %r5, %r6;\n" ++
  "    st.shared.f32 [%r5], %f0;\n" ++
  "skip1:\n" ++
  "    bar.sync      0;\n" ++
  "    setp.ne.u32   %p, %r1, 0;\n" ++
  "    @%p bra       skip2;\n" ++
  "    mov.u32       %r5, _smem;\n" ++
  "    ld.shared.f32 %f0, [%r5+0];\n" ++
  "    ld.shared.f32 %f1, [%r5+4];\n" ++
  "    add.f32       %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r5+8];\n" ++
  "    add.f32       %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r5+12];\n" ++
  "    add.f32       %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r5+16];\n" ++
  "    add.f32       %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r5+20];\n" ++
  "    add.f32       %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r5+24];\n" ++
  "    add.f32       %f0, %f0, %f1;\n" ++
  "    ld.shared.f32 %f1, [%r5+28];\n" ++
  "    add.f32       %f0, %f0, %f1;\n" ++
  "    cvt.rn.f32.u32 %f1, %r0;\n" ++
  "    div.rn.f32    %f0, %f0, %f1;\n" ++
  "    mov.f32       %f1, 0f3727c5ac;\n" ++
  "    add.f32       %f0, %f0, %f1;\n" ++
  "    rsqrt.approx.f32 %f0, %f0;\n" ++
  "    mov.u32       %r5, _smem;\n" ++
  "    st.shared.f32 [%r5+32], %f0;\n" ++
  "skip2:\n" ++
  "    bar.sync      0;\n" ++
  "    mov.u32       %r5, _smem;\n" ++
  "    ld.shared.f32 %f3, [%r5+32];\n" ++
  "    mov.u32       %r4, %r1;\n" ++
  "loop2:\n" ++
  "    setp.ge.u32   %p, %r4, %r0;\n" ++
  "    @%p bra       done2;\n" ++
  "    cvt.u64.u32   %rd5, %r4;\n" ++
  "    shl.b64       %rd5, %rd5, 2;\n" ++
  "    add.u64       %rd6, %rd2, %rd5;\n" ++
  "    ld.global.f32 %f4, [%rd6];\n" ++
  "    add.u64       %rd7, %rd4, %rd5;\n" ++
  "    ld.global.f32 %f5, [%rd7];\n" ++
  "    mul.f32       %f4, %f4, %f3;\n" ++
  "    mul.f32       %f4, %f4, %f5;\n" ++
  "    add.u64       %rd8, %rd1, %rd5;\n" ++
  "    st.global.f32 [%rd8], %f4;\n" ++
  "    add.u32       %r4, %r4, 256;\n" ++
  "    bra           loop2;\n" ++
  "done2:\n" ++
  "    ret;\n" ++
  "}\n"

/-
  Load: init CUDA, read N and weights from data, alloc 2 GPU bufs,
  upload N + weights into buf0.
  Shared memory app fields: N_OFF (i64), BUF0_OFF (i32), BUF1_OFF (i32)
-/
def loadFn : IRBuilder Unit := do
  let ptr  ← entryBlock
  let cuda ← declareCudaFFI
  let dataPtr ← load64 (← absAddr ptr 0x18)

  cudaInit cuda ptr 0x10
  let ctxPtr ← load64 (← absAddr ptr 0x10)

  -- Read N from data[0], store at N_OFF
  let n    ← load64 dataPtr
  storeI64 n (← absAddr ptr N_OFF)

  -- buf0 size = N*4 (input x) + 8 (N header) + N*4 (weights) = 8 + N*8
  let nBytes ← ishlImm n 2
  let buf0Sz ← iaddImm (← iadd nBytes nBytes) 8
  -- buf1 size = N*4 (output)
  let buf1Sz ← ishlImm n 2

  let buf0 ← call cuda.fnCreateBuffer [ctxPtr, buf0Sz]
  let buf1 ← call cuda.fnCreateBuffer [ctxPtr, buf1Sz]
  storeI32 buf0 (← absAddr ptr BUF0_OFF)
  storeI32 buf1 (← absAddr ptr BUF1_OFF)

  -- Upload N (8 bytes) to buf0 at offset 0
  let nAddr  ← absAddr ptr N_OFF
  let _ ← call cuda.fnUploadOffset [ctxPtr, buf0, ← iconst64 0, nAddr, ← iconst64 8]

  -- Upload weights (data[1..N], N*4 bytes) to buf0 at offset 8 + N*4
  let wSrc ← iaddImm dataPtr 8
  let wOff ← iaddImm nBytes 8
  let _ ← call cuda.fnUploadOffset [ctxPtr, buf0, wOff, wSrc, nBytes]
  ret

/-
  Prep: upload input x (data_ptr, N*4 bytes) to buf0 at offset 8.
-/
def prepFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let cuda    ← declareCudaFFI
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let n       ← load64 (← absAddr ptr N_OFF)
  let buf0    ← load32 (← absAddr ptr BUF0_OFF)
  let ctxPtr  ← load64 (← absAddr ptr 0x10)
  let nBytes  ← ishlImm n 2
  let _ ← call cuda.fnUploadOffset [ctxPtr, buf0, ← iconst64 8, dataPtr, nBytes]
  ret

/-
  Infer: launch kernel (1 block, 256 threads), sync, optionally download buf1 to out_ptr.
-/
def inferFn : IRBuilder Unit := do
  let ptr    ← entryBlock
  let cuda   ← declareCudaFFI
  let outPtr ← load64 (← absAddr ptr 0x28)
  let outLen ← load64 (← absAddr ptr 0x30)
  let ctxPtr ← load64 (← absAddr ptr 0x10)
  let nBufs  ← iconst32 2
  let one32  ← iconst32 1
  let blk256 ← iconst32 256

  let skipDl     ← declareBlock []
  let doDownload ← declareBlock []

  let _ ← cudaLaunch cuda ptr (← iconst64 PTX_SOURCE_OFF) nBufs
             (← iconst64 BIND_DESC_OFF) one32 one32 one32 blk256 one32 one32
  let _ ← cudaSync cuda ptr 0x10
  brif (← icmpImm .eq outLen 0) skipDl.ref [] doDownload.ref []

  startBlock doDownload
  let buf1 ← load32 (← absAddr ptr BUF1_OFF)
  let _ ← call cuda.fnDownload [ctxPtr, buf1, outPtr, outLen]
  ret

  startBlock skipDl
  ret

def clifIR : String :=
  noopFunction ++ "\n" ++
  buildFunction 1 loadFn ++ "\n" ++
  buildFunction 2 prepFn ++ "\n" ++
  buildFunction 3 inferFn

def ptxBytes : List UInt8 := ptxSource.toUTF8.toList ++ [0]
def bindDesc : List UInt8 := [0, 0, 0, 0, 1, 0, 0, 0]

def buildInitialMemory : List UInt8 :=
  let reserved := zeros 0x0100
  let ptx := ptxBytes ++ zeros (BIND_DESC_OFF - PTX_SOURCE_OFF - ptxBytes.length)
  let bind := bindDesc ++ zeros (MEM_SIZE - BIND_DESC_OFF - bindDesc.length)
  reserved ++ ptx ++ bind

def actions (src : UInt32) : List Action :=
  [{ kind := .ClifCall, dst := 0, src := src, offset := 0, size := 0 }]

def buildConfig : BaseConfig := {
  cranelift_ir := clifIR,
  memory_size := MEM_SIZE,
  context_offset := 0,
  initial_memory := buildInitialMemory
}

def loadAlgorithm : Algorithm := { actions := actions 1, cranelift_units := 0, timeout_ms := some TIMEOUT_MS }
def prepAlgorithm : Algorithm := { actions := actions 2, cranelift_units := 0, timeout_ms := some TIMEOUT_MS }
def inferAlgorithm : Algorithm := { actions := actions 3, cranelift_units := 0, timeout_ms := some TIMEOUT_MS }

def artifacts : Array Json :=
  #[
    toJsonEntry "cuda_rmsnorm_load" buildConfig loadAlgorithm,
    toJsonEntry "cuda_rmsnorm_prep" buildConfig prepAlgorithm,
    toJsonEntry "cuda_rmsnorm_infer" buildConfig inferAlgorithm,
  ]

end CudaRmsNormPersist
