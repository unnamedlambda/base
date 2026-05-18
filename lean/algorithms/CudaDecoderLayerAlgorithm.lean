import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.PTX

namespace CudaDecoderLayer

/-!
  Persistent single-token decoder-layer benchmark.

  Fixed dimensions:
    d_model = 896
    d_ff    = 4864

  Load payload layout:
    rms1[d_model]
    wq[d_model, d_model]
    wk[d_model, d_model]
    wv[d_model, d_model]
    wo[d_model, d_model]
    rms2[d_model]
    wg[d_ff, d_model]
    wu[d_ff, d_model]
    wd[d_model, d_ff]

  Infer payload layout:
    x[d_model]

  Output:
    y[d_model]

  Attention is simplified to the seq_len=1 case, so the attention output is v.
  We still compute q and k so the projection cost matches a real decoder layer.
-/

def D_MODEL : Nat := 896
def D_FF    : Nat := 4864

def D_MODEL_BYTES : Nat := D_MODEL * 4
def D_FF_BYTES    : Nat := D_FF * 4
def W_DM_DM_BYTES : Nat := D_MODEL * D_MODEL * 4
def W_FF_DM_BYTES : Nat := D_FF * D_MODEL * 4
def W_DM_FF_BYTES : Nat := D_MODEL * D_FF * 4

def PTX_RMS_OFF : Nat := 0x0800
def PTX_SILU_OFF : Nat := 0x1C00
def PTX_ADDRMS_OFF : Nat := 0x2800
def PTX_ADD_OFF : Nat := 0x3800
def MEM_SIZE : Nat := 0x4800

-- App fields: buf IDs stored as i32 (4 bytes each), starting at 0x38
-- (beyond the 56-byte IoOffsets at 0x00-0x37)
def BUF_X_OFF      : Nat := 0x38
def BUF_XN1_OFF    : Nat := 0x3C
def BUF_Q_OFF      : Nat := 0x40
def BUF_K_OFF      : Nat := 0x44
def BUF_V_OFF      : Nat := 0x48
def BUF_O_OFF      : Nat := 0x4C
def BUF_XN2_OFF    : Nat := 0x50
def BUF_G_OFF      : Nat := 0x54
def BUF_U_OFF      : Nat := 0x58
def BUF_A_OFF      : Nat := 0x5C
def BUF_D_OFF      : Nat := 0x60
def BUF_RMS1_OFF   : Nat := 0x64
def BUF_WQ_OFF     : Nat := 0x68
def BUF_WK_OFF     : Nat := 0x6C
def BUF_WV_OFF     : Nat := 0x70
def BUF_WO_OFF     : Nat := 0x74
def BUF_RMS2_OFF   : Nat := 0x78
def BUF_WG_OFF     : Nat := 0x7C
def BUF_WU_OFF     : Nat := 0x80
def BUF_WD_OFF     : Nat := 0x84

def BIND_RMS1_OFF  : Nat := 0x100
def BIND_ADDRMS_OFF : Nat := 0x110
def BIND_SILU_OFF  : Nat := 0x130
def BIND_ADD2_OFF  : Nat := 0x140

def ptxRmsNorm : String := buildModule 36 [{ name := "main", params := ["x_ptr", "w_ptr", "y_ptr"], body := do
  let xPtr ← ldParam "x_ptr"
  let wPtr ← ldParam "w_ptr"
  let yPtr ← ldParam "y_ptr"
  let nReg ← freshR; movRC nReg D_MODEL
  let (tid, warpId, laneId) ← getWarpIds
  rmsNormBody xPtr wPtr yPtr tid warpId laneId nReg 0x44600000 ""
  ptxRet }]

def ptxSiluGate : String := buildModule 0 [{ name := "main", params := ["gate_ptr", "up_ptr", "out_ptr"], body := do
  let gatePtr ← ldParam "gate_ptr"
  let upPtr   ← ldParam "up_ptr"
  let outPtr  ← ldParam "out_ptr"
  let nReg ← freshR; movRC nReg D_FF
  let (gid, _) ← gridStrideSetup nReg "done"
  let gAddr ← elemAddr gatePtr gid
  let uAddr ← elemAddr upPtr gid
  let oAddr ← elemAddr outPtr gid
  let g  ← freshF; ldGlobalF g gAddr
  let u  ← freshF; ldGlobalF u uAddr
  let ng ← freshF; negF ng g
  let l  ← freshF; movFC l f32_log2e
  mulF ng ng l; ex2 ng ng
  let one ← freshF; movFC one f32_1
  addF ng ng one; rcp ng ng
  mulF g g ng; mulF g g u
  stGlobalF oAddr g
  label "done"; ptxRet }]

def ptxResidualAdd : String := buildModule 0 [{ name := "main", params := ["x_ptr", "add_ptr"], body := do
  let xPtr   ← ldParam "x_ptr"
  let addPtr ← ldParam "add_ptr"
  let nReg ← freshR; movRC nReg D_MODEL
  let (gid, _) ← gridStrideSetup nReg "done"
  let xAddr ← elemAddr xPtr gid
  let aAddr ← elemAddr addPtr gid
  let xi ← freshF; ldGlobalF xi xAddr
  let ai ← freshF; ldGlobalF ai aAddr
  addF xi xi ai; stGlobalF xAddr xi
  label "done"; ptxRet }]

def ptxAddRmsNorm : String := buildModule 36 [{ name := "main", params := ["x_ptr", "add_ptr", "w_ptr", "y_ptr"], body := do
  let xPtr   ← ldParam "x_ptr"
  let addPtr ← ldParam "add_ptr"
  let wPtr   ← ldParam "w_ptr"
  let yPtr   ← ldParam "y_ptr"
  let nReg ← freshR; movRC nReg D_MODEL
  let (tid, warpId, laneId) ← getWarpIds
  let acc ← freshF; movFC acc f32_0
  let tmp ← freshF
  strideLoop tid nReg 256 "loop1" "done1" fun i => do
    let xAddr ← elemAddr xPtr i
    let aAddr ← elemAddr addPtr i
    let xi ← freshF; ldGlobalF xi xAddr
    let ai ← freshF; ldGlobalF ai aAddr
    addF xi xi ai; fmaRn acc xi xi acc
  warpReduceSum acc tmp
  lane0WriteSmem laneId warpId "skip1" fun wAddr => stSharedFD wAddr acc
  thread0Op tid "skip2" do
    let sBase ← smemBase
    let total ← freshF
    crossWarp8 total tmp sBase 0 addF
    let nf ← freshF; movFC nf 0x44600000
    divRn total total nf
    let eps ← freshF; movFC eps f32_eps
    addF total total eps; rsqrt total total
    stSharedF sBase 32 total
  let sBase2 ← smemBase
  let scale ← freshF; ldSharedF scale sBase2 32
  strideLoop tid nReg 256 "loop2" "done2" fun j => do
    let xAddr ← elemAddr xPtr j
    let aAddr ← elemAddr addPtr j
    let wAddr ← elemAddr wPtr j
    let yAddr ← elemAddr yPtr j
    let xi ← freshF; ldGlobalF xi xAddr
    let ai ← freshF; ldGlobalF ai aAddr
    addF xi xi ai
    stGlobalF xAddr xi
    let wi ← freshF; ldGlobalF wi wAddr
    mulF xi xi scale; mulF xi xi wi
    stGlobalF yAddr xi
  ptxRet }]

def loadFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let cuda    ← declareCudaFFI
  let dataPtr ← load64 (← absAddr ptr 0x18)
  cudaInit cuda ptr 0x10
  let ctxPtr  ← load64 (← absAddr ptr 0x10)

  let dmBytes  ← iconst64 D_MODEL_BYTES
  let ffBytes  ← iconst64 D_FF_BYTES
  let wdmBytes ← iconst64 W_DM_DM_BYTES
  let wffBytes ← iconst64 W_FF_DM_BYTES
  let wdfBytes ← iconst64 W_DM_FF_BYTES

  -- activation buffers
  let bufX   ← call cuda.fnCreateBuffer [ctxPtr, dmBytes]
  let bufXn1 ← call cuda.fnCreateBuffer [ctxPtr, dmBytes]
  let bufQ   ← call cuda.fnCreateBuffer [ctxPtr, dmBytes]
  let bufK   ← call cuda.fnCreateBuffer [ctxPtr, dmBytes]
  let bufV   ← call cuda.fnCreateBuffer [ctxPtr, dmBytes]
  let bufO   ← call cuda.fnCreateBuffer [ctxPtr, dmBytes]
  let bufXn2 ← call cuda.fnCreateBuffer [ctxPtr, dmBytes]
  let bufG   ← call cuda.fnCreateBuffer [ctxPtr, ffBytes]
  let bufU   ← call cuda.fnCreateBuffer [ctxPtr, ffBytes]
  let bufA   ← call cuda.fnCreateBuffer [ctxPtr, ffBytes]
  let bufD   ← call cuda.fnCreateBuffer [ctxPtr, dmBytes]
  -- weight buffers
  let bufRms1 ← call cuda.fnCreateBuffer [ctxPtr, dmBytes]
  let bufWq   ← call cuda.fnCreateBuffer [ctxPtr, wdmBytes]
  let bufWk   ← call cuda.fnCreateBuffer [ctxPtr, wdmBytes]
  let bufWv   ← call cuda.fnCreateBuffer [ctxPtr, wdmBytes]
  let bufWo   ← call cuda.fnCreateBuffer [ctxPtr, wdmBytes]
  let bufRms2 ← call cuda.fnCreateBuffer [ctxPtr, dmBytes]
  let bufWg   ← call cuda.fnCreateBuffer [ctxPtr, wffBytes]
  let bufWu   ← call cuda.fnCreateBuffer [ctxPtr, wffBytes]
  let bufWd   ← call cuda.fnCreateBuffer [ctxPtr, wdfBytes]

  storeI32 bufX   (← absAddr ptr BUF_X_OFF)
  storeI32 bufXn1 (← absAddr ptr BUF_XN1_OFF)
  storeI32 bufQ   (← absAddr ptr BUF_Q_OFF)
  storeI32 bufK   (← absAddr ptr BUF_K_OFF)
  storeI32 bufV   (← absAddr ptr BUF_V_OFF)
  storeI32 bufO   (← absAddr ptr BUF_O_OFF)
  storeI32 bufXn2 (← absAddr ptr BUF_XN2_OFF)
  storeI32 bufG   (← absAddr ptr BUF_G_OFF)
  storeI32 bufU   (← absAddr ptr BUF_U_OFF)
  storeI32 bufA   (← absAddr ptr BUF_A_OFF)
  storeI32 bufD   (← absAddr ptr BUF_D_OFF)
  storeI32 bufRms1 (← absAddr ptr BUF_RMS1_OFF)
  storeI32 bufWq  (← absAddr ptr BUF_WQ_OFF)
  storeI32 bufWk  (← absAddr ptr BUF_WK_OFF)
  storeI32 bufWv  (← absAddr ptr BUF_WV_OFF)
  storeI32 bufWo  (← absAddr ptr BUF_WO_OFF)
  storeI32 bufRms2 (← absAddr ptr BUF_RMS2_OFF)
  storeI32 bufWg  (← absAddr ptr BUF_WG_OFF)
  storeI32 bufWu  (← absAddr ptr BUF_WU_OFF)
  storeI32 bufWd  (← absAddr ptr BUF_WD_OFF)

  -- upload weights (rms1, wq, wk, wv, wo, rms2, wg, wu, wd)
  let _ ← call cuda.fnUpload [ctxPtr, bufRms1, dataPtr, dmBytes]
  let p1 ← iaddImm dataPtr D_MODEL_BYTES
  let _ ← call cuda.fnUpload [ctxPtr, bufWq, p1, wdmBytes]
  let p2 ← iaddImm p1 W_DM_DM_BYTES
  let _ ← call cuda.fnUpload [ctxPtr, bufWk, p2, wdmBytes]
  let p3 ← iaddImm p2 W_DM_DM_BYTES
  let _ ← call cuda.fnUpload [ctxPtr, bufWv, p3, wdmBytes]
  let p4 ← iaddImm p3 W_DM_DM_BYTES
  let _ ← call cuda.fnUpload [ctxPtr, bufWo, p4, wdmBytes]
  let p5 ← iaddImm p4 W_DM_DM_BYTES
  let _ ← call cuda.fnUpload [ctxPtr, bufRms2, p5, dmBytes]
  let p6 ← iaddImm p5 D_MODEL_BYTES
  let _ ← call cuda.fnUpload [ctxPtr, bufWg, p6, wffBytes]
  let p7 ← iaddImm p6 W_FF_DM_BYTES
  let _ ← call cuda.fnUpload [ctxPtr, bufWu, p7, wffBytes]
  let p8 ← iaddImm p7 W_FF_DM_BYTES
  let _ ← call cuda.fnUpload [ctxPtr, bufWd, p8, wdfBytes]
  ret

def prepFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let cuda    ← declareCudaFFI
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let ctxPtr  ← load64 (← absAddr ptr 0x10)
  let bufX    ← load32 (← absAddr ptr BUF_X_OFF)
  let dmBytes ← iconst64 D_MODEL_BYTES
  let _ ← call cuda.fnUpload [ctxPtr, bufX, dataPtr, dmBytes]
  ret

def inferFn : IRBuilder Unit := do
  let ptr    ← entryBlock
  let cuda   ← declareCudaFFI
  let blas   ← declareCuBlasFFI
  let ctxPtr ← load64 (← absAddr ptr 0x10)

  -- load all 20 buf IDs
  let bufX    ← load32 (← absAddr ptr BUF_X_OFF)
  let bufXn1  ← load32 (← absAddr ptr BUF_XN1_OFF)
  let bufQ    ← load32 (← absAddr ptr BUF_Q_OFF)
  let bufK    ← load32 (← absAddr ptr BUF_K_OFF)
  let bufV    ← load32 (← absAddr ptr BUF_V_OFF)
  let bufO    ← load32 (← absAddr ptr BUF_O_OFF)
  let bufXn2  ← load32 (← absAddr ptr BUF_XN2_OFF)
  let bufG    ← load32 (← absAddr ptr BUF_G_OFF)
  let bufU    ← load32 (← absAddr ptr BUF_U_OFF)
  let bufA    ← load32 (← absAddr ptr BUF_A_OFF)
  let bufD    ← load32 (← absAddr ptr BUF_D_OFF)
  let bufRms1 ← load32 (← absAddr ptr BUF_RMS1_OFF)
  let bufWq   ← load32 (← absAddr ptr BUF_WQ_OFF)
  let bufWk   ← load32 (← absAddr ptr BUF_WK_OFF)
  let bufWv   ← load32 (← absAddr ptr BUF_WV_OFF)
  let bufWo   ← load32 (← absAddr ptr BUF_WO_OFF)
  let bufRms2 ← load32 (← absAddr ptr BUF_RMS2_OFF)
  let bufWg   ← load32 (← absAddr ptr BUF_WG_OFF)
  let bufWu   ← load32 (← absAddr ptr BUF_WU_OFF)
  let bufWd   ← load32 (← absAddr ptr BUF_WD_OFF)

  let one32   ← iconst32 1
  let three32 ← iconst32 3
  let two32   ← iconst32 2
  let four32  ← iconst32 4
  let blk256  ← iconst32 256
  let dm32    ← iconst32 D_MODEL
  let ff32    ← iconst32 D_FF
  let alpha   ← iconst32 0x3f800000
  let zero32  ← iconst32 0

  -- rms1: normalize x with rms1 weights → xn1
  storeI32 bufX   (← absAddr ptr BIND_RMS1_OFF)
  storeI32 bufRms1 (← absAddr ptr (BIND_RMS1_OFF + 4))
  storeI32 bufXn1 (← absAddr ptr (BIND_RMS1_OFF + 8))
  let _ ← cudaLaunch cuda ptr (← iconst64 PTX_RMS_OFF) three32
             (← iconst64 BIND_RMS1_OFF) one32 one32 one32 blk256 one32 one32

  -- attention projections: q = WQ @ xn1, k = WK @ xn1, v = WV @ xn1, o = WO @ v
  let _ ← call blas.fnSgemv [ctxPtr, one32, dm32, dm32, alpha, bufWq, bufXn1, zero32, bufQ]
  let _ ← call blas.fnSgemv [ctxPtr, one32, dm32, dm32, alpha, bufWk, bufXn1, zero32, bufK]
  let _ ← call blas.fnSgemv [ctxPtr, one32, dm32, dm32, alpha, bufWv, bufXn1, zero32, bufV]
  let _ ← call blas.fnSgemv [ctxPtr, one32, dm32, dm32, alpha, bufWo, bufV,   zero32, bufO]

  -- residual add: x += o
  storeI32 bufX (← absAddr ptr BIND_ADDRMS_OFF)
  storeI32 bufO (← absAddr ptr (BIND_ADDRMS_OFF + 4))
  let _ ← cudaLaunch cuda ptr (← iconst64 PTX_ADD_OFF) two32
             (← iconst64 BIND_ADDRMS_OFF) four32 one32 one32 blk256 one32 one32

  -- rms2: normalize x with rms2 weights → xn2
  storeI32 bufX    (← absAddr ptr (BIND_ADDRMS_OFF + 16))
  storeI32 bufRms2 (← absAddr ptr (BIND_ADDRMS_OFF + 20))
  storeI32 bufXn2  (← absAddr ptr (BIND_ADDRMS_OFF + 24))
  let _ ← cudaLaunch cuda ptr (← iconst64 PTX_RMS_OFF) three32
             (← iconst64 (BIND_ADDRMS_OFF + 16)) one32 one32 one32 blk256 one32 one32

  -- FFN: gate = WG @ xn2, up = WU @ xn2
  let _ ← call blas.fnSgemv [ctxPtr, one32, dm32, ff32, alpha, bufWg, bufXn2, zero32, bufG]
  let _ ← call blas.fnSgemv [ctxPtr, one32, dm32, ff32, alpha, bufWu, bufXn2, zero32, bufU]

  -- SiLU-gate: a = silu(g) * u
  storeI32 bufG (← absAddr ptr BIND_SILU_OFF)
  storeI32 bufU (← absAddr ptr (BIND_SILU_OFF + 4))
  storeI32 bufA (← absAddr ptr (BIND_SILU_OFF + 8))
  let nineteen32 ← iconst32 19
  let _ ← cudaLaunch cuda ptr (← iconst64 PTX_SILU_OFF) three32
             (← iconst64 BIND_SILU_OFF) nineteen32 one32 one32 blk256 one32 one32

  -- down projection: d = WD @ a
  let _ ← call blas.fnSgemv [ctxPtr, one32, ff32, dm32, alpha, bufWd, bufA, zero32, bufD]

  -- residual add: x += d
  storeI32 bufX (← absAddr ptr BIND_ADD2_OFF)
  storeI32 bufD (← absAddr ptr (BIND_ADD2_OFF + 4))
  let _ ← cudaLaunch cuda ptr (← iconst64 PTX_ADD_OFF) two32
             (← iconst64 BIND_ADD2_OFF) four32 one32 one32 blk256 one32 one32
  ret

def finalizeFn : IRBuilder Unit := do
  let ptr    ← entryBlock
  let cuda   ← declareCudaFFI
  let outPtr ← load64 (← absAddr ptr 0x28)
  let outLen ← load64 (← absAddr ptr 0x30)
  let ctxPtr ← load64 (← absAddr ptr 0x10)
  let bufX   ← load32 (← absAddr ptr BUF_X_OFF)

  let skipDl     ← declareBlock []
  let doDownload ← declareBlock []

  let _ ← cudaSync cuda ptr 0x10
  brif (← icmpImm .eq outLen 0) skipDl.ref [] doDownload.ref []

  startBlock doDownload
  let _ ← call cuda.fnDownload [ctxPtr, bufX, outPtr, outLen]
  ret

  startBlock skipDl
  ret

def STACK16_DEPTH : Nat := 16
def STACK32_DEPTH : Nat := 32

def clifIR : String :=
  noopFunction ++ "\n" ++
  buildFunction 1 loadFn ++ "\n" ++
  buildFunction 2 prepFn ++ "\n" ++
  buildFunction 3 inferFn ++ "\n" ++
  buildFunction 4 finalizeFn ++
  clifSequenceWrapper 5 [3, 4] ++
  clifSequenceWrapper 6 (List.replicate STACK16_DEPTH 3 ++ [4]) ++
  clifSequenceWrapper 7 (List.replicate STACK32_DEPTH 3 ++ [4])

def ptxRmsBytes : List UInt8 := ptxRmsNorm.toUTF8.toList ++ [0]
def ptxSiluBytes : List UInt8 := ptxSiluGate.toUTF8.toList ++ [0]
def ptxAddRmsBytes : List UInt8 := ptxAddRmsNorm.toUTF8.toList ++ [0]
def ptxAddBytes : List UInt8 := ptxResidualAdd.toUTF8.toList ++ [0]

def buildInitialMemory : List UInt8 :=
  let pre := zeros PTX_RMS_OFF
  let rms := ptxRmsBytes ++ zeros (PTX_SILU_OFF - PTX_RMS_OFF - ptxRmsBytes.length)
  let silu := ptxSiluBytes ++ zeros (PTX_ADDRMS_OFF - PTX_SILU_OFF - ptxSiluBytes.length)
  let addrms := ptxAddRmsBytes ++ zeros (PTX_ADD_OFF - PTX_ADDRMS_OFF - ptxAddRmsBytes.length)
  let add := ptxAddBytes ++ zeros (MEM_SIZE - PTX_ADD_OFF - ptxAddBytes.length)
  pre ++ rms ++ silu ++ addrms ++ add

def buildSetup : Setup := {
  cranelift_ir := clifIR,
  memory_size := MEM_SIZE,
  initial_memory := buildInitialMemory
}

def loadAlgorithm   : Algorithm := { fn_idx := u32 1 }
def prepAlgorithm   : Algorithm := { fn_idx := u32 2 }
def inferAlgorithm  : Algorithm := { fn_idx := u32 5 }
def stack16Algorithm : Algorithm := { fn_idx := u32 6 }
def stack32Algorithm : Algorithm := { fn_idx := u32 7 }

def artifacts : Array Json :=
  #[
    toJsonArtifact "cuda_decoder" buildSetup loadAlgorithm [
      ("prep",    prepAlgorithm),
      ("infer",   inferAlgorithm),
      ("stack16", stack16Algorithm),
      ("stack32", stack32Algorithm)
    ]
  ]

end CudaDecoderLayer
