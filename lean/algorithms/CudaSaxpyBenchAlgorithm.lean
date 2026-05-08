import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace CudaSaxpyBench

/-
  CUDA SAXPY: y[i] = 2.0 * x[i] + y[i]
  Payload: [x floats: N][y floats: N], Output: [y result floats: N]
-/

def PTX_SOURCE_OFF : Nat := 0x0100
def BIND_DESC_OFF  : Nat := 0x1100
def MEM_SIZE       : Nat := 0x1200
def TIMEOUT_MS     : Nat := 120000

def ptxSource : String :=
  ".version 7.0\n" ++
  ".target sm_50\n" ++
  ".address_size 64\n" ++
  "\n" ++
  ".visible .entry main(\n" ++
  "    .param .u64 x_ptr,\n" ++
  "    .param .u64 y_ptr\n" ++
  ")\n" ++
  "{\n" ++
  "    .reg .u32 %r0, %r1;\n" ++
  "    .reg .u64 %rx, %ry, %off;\n" ++
  "    .reg .f32 %fx, %fy, %fa;\n" ++
  "\n" ++
  "    mov.u32 %r0, %ctaid.x;\n" ++
  "    mov.u32 %r1, %tid.x;\n" ++
  "    mad.lo.u32 %r0, %r0, 256, %r1;\n" ++
  "\n" ++
  "    cvt.u64.u32 %off, %r0;\n" ++
  "    shl.b64 %off, %off, 2;\n" ++
  "\n" ++
  "    ld.param.u64 %rx, [x_ptr];\n" ++
  "    ld.param.u64 %ry, [y_ptr];\n" ++
  "\n" ++
  "    add.u64 %rx, %rx, %off;\n" ++
  "    add.u64 %ry, %ry, %off;\n" ++
  "\n" ++
  "    ld.global.f32 %fx, [%rx];\n" ++
  "    ld.global.f32 %fy, [%ry];\n" ++
  "    mov.f32 %fa, 0f40000000;\n" ++
  "    fma.rn.f32 %fy, %fa, %fx, %fy;\n" ++
  "    st.global.f32 [%ry], %fy;\n" ++
  "\n" ++
  "    ret;\n" ++
  "}\n"

def mainFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let dataLen ← load64 (← absAddr ptr 0x20)
  let outPtr  ← load64 (← absAddr ptr 0x28)

  let fnInit         ← declareFFI "cl_cuda_init"          [.i64]                    none
  let fnCreateBuffer ← declareFFI "cl_cuda_create_buffer" [.i64, .i64]             (some .i32)
  let fnUploadPtr    ← declareFFI "cl_cuda_upload_ptr"    [.i64, .i32, .i64, .i64] (some .i32)
  let fnDownloadPtr  ← declareFFI "cl_cuda_download_ptr"  [.i64, .i32, .i64, .i64] (some .i32)
  let fnLaunch       ← declareFFI "cl_cuda_launch"
    [.i64, .i64, .i32, .i64, .i32, .i32, .i32, .i32, .i32, .i32] (some .i32)
  let fnCleanup      ← declareFFI "cl_cuda_cleanup"       [.i64]                    none

  let ctxSlotPtr ← absAddr ptr 0x10   -- ContextSlots.cuda
  callVoid fnInit [ctxSlotPtr]
  let ctxPtr ← load64 ctxSlotPtr

  -- buf_size = data_len / 2 (each of x and y is half)
  let bufSize ← ushrImm dataLen 1
  let xBufId  ← call fnCreateBuffer [ctxPtr, bufSize]
  let yBufId  ← call fnCreateBuffer [ctxPtr, bufSize]

  -- Upload x from data_ptr, y from data_ptr + buf_size
  let _ ← call fnUploadPtr [ctxPtr, xBufId, dataPtr, bufSize]
  let yDataPtr ← iadd dataPtr bufSize
  let _ ← call fnUploadPtr [ctxPtr, yBufId, yDataPtr, bufSize]

  -- Grid: ceil(N / 256) where N = buf_size / 4
  let bigN  ← ireduce32 (← ushrImm bufSize 2)
  let c255  ← iconst32 255
  let nPlus ← iadd bigN c255
  let c256  ← iconst32 256
  let gridX ← udiv nPlus c256
  let one   ← iconst32 1

  let ptxAddr  ← absAddr ptr PTX_SOURCE_OFF
  let two      ← iconst32 2
  let bindAddr ← absAddr ptr BIND_DESC_OFF
  let _ ← call fnLaunch [ctxPtr, ptxAddr, two, bindAddr,
                          gridX, one, one, c256, one, one]

  let _ ← call fnDownloadPtr [ctxPtr, yBufId, outPtr, bufSize]

  callVoid fnCleanup [ctxSlotPtr]
  ret

def clifIR : String := buildProgram mainFn

def ptxBytes : List UInt8 := ptxSource.toUTF8.toList ++ [0]
def bindDesc : List UInt8 := [0, 0, 0, 0, 1, 0, 0, 0]

def buildInitialMemory : List UInt8 :=
  let reserved := zeros 0x0100
  let ptx := ptxBytes ++ zeros (BIND_DESC_OFF - PTX_SOURCE_OFF - ptxBytes.length)
  let bind := bindDesc ++ zeros (MEM_SIZE - BIND_DESC_OFF - bindDesc.length)
  reserved ++ ptx ++ bind

def artifacts : Array Json :=
  #[toJsonEntry "cuda_saxpy_algorithm" {
    cranelift_ir := clifIR,
    memory_size := MEM_SIZE,
    context_offset := 0,
    initial_memory := buildInitialMemory
  } {
    actions := mkCallActions 1,
    cranelift_units := 0,
    timeout_ms := some TIMEOUT_MS
  }]

end CudaSaxpyBench
