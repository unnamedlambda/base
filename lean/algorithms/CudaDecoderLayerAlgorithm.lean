import Lean
import Std
import AlgorithmLib.Gen
import AlgorithmLib.ProgCuda
import AlgorithmLib.ProgCuda
import LayoutScan

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
-- (past the 56-byte reserved header at 0x00-0x37)
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

open AlgorithmLib.Prog


/-- The CUDA context pointer lives at a fixed slot in shared memory. -/
def CTX_OFF : Nat := 0x10

def loadCode : Prog V L Unit := do
  let ptr ← basePtr
  let dataPtr ← dataPtr
  cudaInit ptr CTX_OFF
  let ctxPtr  ← load64 (← absAddr ptr CTX_OFF)

  let dmBytes  ← iconst64 D_MODEL_BYTES
  let ffBytes  ← iconst64 D_FF_BYTES
  let wdmBytes ← iconst64 W_DM_DM_BYTES
  let wffBytes ← iconst64 W_FF_DM_BYTES
  let wdfBytes ← iconst64 W_DM_FF_BYTES

  -- activation buffers
  let bufX   ← ffi .cudaCreateBuffer %[ctxPtr, dmBytes]
  let bufXn1 ← ffi .cudaCreateBuffer %[ctxPtr, dmBytes]
  let bufQ   ← ffi .cudaCreateBuffer %[ctxPtr, dmBytes]
  let bufK   ← ffi .cudaCreateBuffer %[ctxPtr, dmBytes]
  let bufV   ← ffi .cudaCreateBuffer %[ctxPtr, dmBytes]
  let bufO   ← ffi .cudaCreateBuffer %[ctxPtr, dmBytes]
  let bufXn2 ← ffi .cudaCreateBuffer %[ctxPtr, dmBytes]
  let bufG   ← ffi .cudaCreateBuffer %[ctxPtr, ffBytes]
  let bufU   ← ffi .cudaCreateBuffer %[ctxPtr, ffBytes]
  let bufA   ← ffi .cudaCreateBuffer %[ctxPtr, ffBytes]
  let bufD   ← ffi .cudaCreateBuffer %[ctxPtr, dmBytes]
  -- weight buffers
  let bufRms1 ← ffi .cudaCreateBuffer %[ctxPtr, dmBytes]
  let bufWq   ← ffi .cudaCreateBuffer %[ctxPtr, wdmBytes]
  let bufWk   ← ffi .cudaCreateBuffer %[ctxPtr, wdmBytes]
  let bufWv   ← ffi .cudaCreateBuffer %[ctxPtr, wdmBytes]
  let bufWo   ← ffi .cudaCreateBuffer %[ctxPtr, wdmBytes]
  let bufRms2 ← ffi .cudaCreateBuffer %[ctxPtr, dmBytes]
  let bufWg   ← ffi .cudaCreateBuffer %[ctxPtr, wffBytes]
  let bufWu   ← ffi .cudaCreateBuffer %[ctxPtr, wffBytes]
  let bufWd   ← ffi .cudaCreateBuffer %[ctxPtr, wdfBytes]

  store bufX   (← absAddr ptr BUF_X_OFF)
  store bufXn1 (← absAddr ptr BUF_XN1_OFF)
  store bufQ   (← absAddr ptr BUF_Q_OFF)
  store bufK   (← absAddr ptr BUF_K_OFF)
  store bufV   (← absAddr ptr BUF_V_OFF)
  store bufO   (← absAddr ptr BUF_O_OFF)
  store bufXn2 (← absAddr ptr BUF_XN2_OFF)
  store bufG   (← absAddr ptr BUF_G_OFF)
  store bufU   (← absAddr ptr BUF_U_OFF)
  store bufA   (← absAddr ptr BUF_A_OFF)
  store bufD   (← absAddr ptr BUF_D_OFF)
  store bufRms1 (← absAddr ptr BUF_RMS1_OFF)
  store bufWq  (← absAddr ptr BUF_WQ_OFF)
  store bufWk  (← absAddr ptr BUF_WK_OFF)
  store bufWv  (← absAddr ptr BUF_WV_OFF)
  store bufWo  (← absAddr ptr BUF_WO_OFF)
  store bufRms2 (← absAddr ptr BUF_RMS2_OFF)
  store bufWg  (← absAddr ptr BUF_WG_OFF)
  store bufWu  (← absAddr ptr BUF_WU_OFF)
  store bufWd  (← absAddr ptr BUF_WD_OFF)

  -- upload weights (rms1, wq, wk, wv, wo, rms2, wg, wu, wd)
  let _ ← ffi .cudaUpload %[ctxPtr, bufRms1, dataPtr, dmBytes]
  let p1 ← iaddImm dataPtr D_MODEL_BYTES
  let _ ← ffi .cudaUpload %[ctxPtr, bufWq, p1, wdmBytes]
  let p2 ← iaddImm p1 W_DM_DM_BYTES
  let _ ← ffi .cudaUpload %[ctxPtr, bufWk, p2, wdmBytes]
  let p3 ← iaddImm p2 W_DM_DM_BYTES
  let _ ← ffi .cudaUpload %[ctxPtr, bufWv, p3, wdmBytes]
  let p4 ← iaddImm p3 W_DM_DM_BYTES
  let _ ← ffi .cudaUpload %[ctxPtr, bufWo, p4, wdmBytes]
  let p5 ← iaddImm p4 W_DM_DM_BYTES
  let _ ← ffi .cudaUpload %[ctxPtr, bufRms2, p5, dmBytes]
  let p6 ← iaddImm p5 D_MODEL_BYTES
  let _ ← ffi .cudaUpload %[ctxPtr, bufWg, p6, wffBytes]
  let p7 ← iaddImm p6 W_FF_DM_BYTES
  let _ ← ffi .cudaUpload %[ctxPtr, bufWu, p7, wffBytes]
  let p8 ← iaddImm p7 W_FF_DM_BYTES
  let _ ← ffi .cudaUpload %[ctxPtr, bufWd, p8, wdfBytes]

def prepCode : Prog V L Unit := do
  let ptr ← basePtr
  let dataPtr ← dataPtr
  let ctxPtr  ← load64 (← absAddr ptr CTX_OFF)
  let bufX    ← load32 (← absAddr ptr BUF_X_OFF)
  let dmBytes ← iconst64 D_MODEL_BYTES
  let _ ← ffi .cudaUpload %[ctxPtr, bufX, dataPtr, dmBytes]

/-! ### The kernels, as records the launch sites read

    Each bind table and the launch that reads it now take their length from
    the same `params`, so a table and the arity handed to the driver cannot
    disagree.  `addK` appears at two sites over two tables — same kernel, same
    declared shape, different scratch. -/

/-- Normalize a `D_MODEL` row by its RMS: `x`, the weights, the output. -/
def rmsK : AlgorithmLib.Kernel := {
  name   := "rms"
  params := [{ shape := [.sta D_MODEL], ro := true,  name := "x" },
             { shape := [.sta D_MODEL], ro := true,  name := "w" },
             { shape := [.sta D_MODEL], ro := false, name := "xn" }]
  geom   := AlgorithmLib.Kernel.Geom.static 1 1 1 256 1 1
  ptxOff := PTX_RMS_OFF
}

/-- Residual add, in place: `x += y`. -/
def addK : AlgorithmLib.Kernel := {
  name   := "add"
  params := [{ shape := [.sta D_MODEL], ro := false, name := "x" },
             { shape := [.sta D_MODEL], ro := true,  name := "y" }]
  geom   := AlgorithmLib.Kernel.Geom.static 4 1 1 256 1 1
  ptxOff := PTX_ADD_OFF
}

/-- SiLU gate: `a = silu(g) * u`, over the FFN width. -/
def siluK : AlgorithmLib.Kernel := {
  name   := "silu"
  params := [{ shape := [.sta D_FF], ro := true,  name := "g" },
             { shape := [.sta D_FF], ro := true,  name := "u" },
             { shape := [.sta D_FF], ro := false, name := "a" }]
  geom   := AlgorithmLib.Kernel.Geom.static 19 1 1 256 1 1
  ptxOff := PTX_SILU_OFF
}

def inferCode : Prog V L Unit := do
  let ptr ← basePtr
  let ctxPtr ← load64 (← absAddr ptr CTX_OFF)

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
  let dm32    ← iconst32 D_MODEL
  let ff32    ← iconst32 D_FF
  let alpha   ← iconst32 0x3f800000
  let zero32  ← iconst32 0

  -- rms1: normalize x with rms1 weights → xn1
  kernelLaunchAt rmsK ptr BIND_RMS1_OFF [bufX, bufRms1, bufXn1]

  -- attention projections: q = WQ @ xn1, k = WK @ xn1, v = WV @ xn1, o = WO @ v
  let _ ← ffi .cublasSgemv %[ctxPtr, one32, dm32, dm32, alpha, bufWq, bufXn1, zero32, bufQ]
  let _ ← ffi .cublasSgemv %[ctxPtr, one32, dm32, dm32, alpha, bufWk, bufXn1, zero32, bufK]
  let _ ← ffi .cublasSgemv %[ctxPtr, one32, dm32, dm32, alpha, bufWv, bufXn1, zero32, bufV]
  let _ ← ffi .cublasSgemv %[ctxPtr, one32, dm32, dm32, alpha, bufWo, bufV,   zero32, bufO]

  -- residual add: x += o
  kernelLaunchAt addK ptr BIND_ADDRMS_OFF [bufX, bufO]

  -- rms2: normalize x with rms2 weights → xn2
  kernelLaunchAt rmsK ptr (BIND_ADDRMS_OFF + 16) [bufX, bufRms2, bufXn2]

  -- FFN: gate = WG @ xn2, up = WU @ xn2
  let _ ← ffi .cublasSgemv %[ctxPtr, one32, dm32, ff32, alpha, bufWg, bufXn2, zero32, bufG]
  let _ ← ffi .cublasSgemv %[ctxPtr, one32, dm32, ff32, alpha, bufWu, bufXn2, zero32, bufU]

  -- SiLU-gate: a = silu(g) * u
  kernelLaunchAt siluK ptr BIND_SILU_OFF [bufG, bufU, bufA]

  -- down projection: d = WD @ a
  let _ ← ffi .cublasSgemv %[ctxPtr, one32, ff32, dm32, alpha, bufWd, bufA, zero32, bufD]

  -- residual add: x += d
  kernelLaunchAt addK ptr BIND_ADD2_OFF [bufX, bufD]

/-- Finalize: sync, then download only if the caller asked for output. -/
def finalizeCode : Prog V L Unit := do
  let ptr    := (← basePtr)
  let outPtr ← outPtr
  let outLen ← outLen
  let ctxPtr ← load64 (← absAddr ptr CTX_OFF)
  let bufX   ← load32 (← absAddr ptr BUF_X_OFF)

  let _ ← cudaSync ptr CTX_OFF
  let _ ← ifte .eq outLen (← iconst64 0)
    (thn := pure %[])
    (els := do
      let _ ← ffi .cudaDownload %[ctxPtr, bufX, outPtr, outLen]
      pure %[])
  return ()


-- The launch helpers add definitional layers the body checks reduce through.
set_option maxRecDepth 4000


def STACK16_DEPTH : Nat := 16
def STACK32_DEPTH : Nat := 32

def clifIR : Except String Program :=
  Prog.program
    [.ok noopFunction,
     Prog.compileProg 1 loadCode,
     Prog.compileProg 2 prepCode,
     Prog.compileProg 3 inferCode,
     Prog.compileProg 4 finalizeCode,
     Prog.compileProg 5 (Prog.sequenceWrapper [3, 4]),
     Prog.compileProg 6 (Prog.sequenceWrapper (List.replicate STACK16_DEPTH 3 ++ [4])),
     Prog.compileProg 7 (Prog.sequenceWrapper (List.replicate STACK32_DEPTH 3 ++ [4]))]

def ptxRmsBytes : List UInt8 := ptxRmsNorm.toUTF8.toList ++ [0]
def ptxSiluBytes : List UInt8 := ptxSiluGate.toUTF8.toList ++ [0]
def ptxAddRmsBytes : List UInt8 := ptxAddRmsNorm.toUTF8.toList ++ [0]
def ptxAddBytes : List UInt8 := ptxResidualAdd.toUTF8.toList ++ [0]

/-- Every byte of shared memory this program names, and how much of it it uses.

    The offsets are assigned by hand, and a collision between two of them is
    invisible at every other layer: both stores succeed, the second one wins,
    and the kernel reads whichever ran last. Sizes are what the code actually
    writes -- one `i32` per buffer slot, and a bind table as wide as the launch
    that reads it -- so a region that grows past its neighbour is a failed
    proof rather than a corrupted field.

    `0x00`-`0x18` are the context slots the runtime fills, and `0x18`-`0x38`
    the input and output descriptors it writes; naming them is what stops a
    future offset being placed where the runtime will overwrite it. -/
def memMap : AlgorithmLib.Layout.RegionMap :=
  [⟨"ctx_ht",       ContextSlots.ht, 8⟩,
   ⟨"ctx_wgpu",     ContextSlots.wgpu, 8⟩,
   ⟨"ctx_cuda",     CTX_OFF, 8⟩,
   ⟨"buf_x",        BUF_X_OFF, 4⟩,
   ⟨"buf_xn1",      BUF_XN1_OFF, 4⟩,
   ⟨"buf_q",        BUF_Q_OFF, 4⟩,
   ⟨"buf_k",        BUF_K_OFF, 4⟩,
   ⟨"buf_v",        BUF_V_OFF, 4⟩,
   ⟨"buf_o",        BUF_O_OFF, 4⟩,
   ⟨"buf_xn2",      BUF_XN2_OFF, 4⟩,
   ⟨"buf_g",        BUF_G_OFF, 4⟩,
   ⟨"buf_u",        BUF_U_OFF, 4⟩,
   ⟨"buf_a",        BUF_A_OFF, 4⟩,
   ⟨"buf_d",        BUF_D_OFF, 4⟩,
   ⟨"buf_rms1",     BUF_RMS1_OFF, 4⟩,
   ⟨"buf_wq",       BUF_WQ_OFF, 4⟩,
   ⟨"buf_wk",       BUF_WK_OFF, 4⟩,
   ⟨"buf_wv",       BUF_WV_OFF, 4⟩,
   ⟨"buf_wo",       BUF_WO_OFF, 4⟩,
   ⟨"buf_rms2",     BUF_RMS2_OFF, 4⟩,
   ⟨"buf_wg",       BUF_WG_OFF, 4⟩,
   ⟨"buf_wu",       BUF_WU_OFF, 4⟩,
   ⟨"buf_wd",       BUF_WD_OFF, 4⟩,
   -- Each bind table is as wide as the furthest slot its launch stores into.
   ⟨"bind_rms1",    BIND_RMS1_OFF, 12⟩,
   ⟨"bind_addrms",  BIND_ADDRMS_OFF, 28⟩,
   ⟨"bind_silu",    BIND_SILU_OFF, 12⟩,
   ⟨"bind_add2",    BIND_ADD2_OFF, 8⟩,
   ⟨"ptx_rms",      PTX_RMS_OFF, PTX_SILU_OFF - PTX_RMS_OFF⟩,
   ⟨"ptx_silu",     PTX_SILU_OFF, PTX_ADDRMS_OFF - PTX_SILU_OFF⟩,
   ⟨"ptx_addrms",   PTX_ADDRMS_OFF, PTX_ADD_OFF - PTX_ADDRMS_OFF⟩,
   ⟨"ptx_add",      PTX_ADD_OFF, MEM_SIZE - PTX_ADD_OFF⟩]


#eval LayoutScan.check "CudaDecoderLayerAlgorithm" [``memMap]
theorem memMap_ok : AlgorithmLib.Layout.RegionMap.okB memMap = true := by decide

theorem memMap_within :
    AlgorithmLib.Layout.RegionMap.withinB MEM_SIZE memMap = true := by decide

/-- **Each kernel's PTX fits the slot it is placed in.**

    `buildInitialMemory` pads each slot with `zeros (next - this - length)`, and
    that subtraction is on `Nat`: a PTX text longer than its slot gives a pad of
    zero rather than a negative one, so the overrun is silently written over the
    following kernel and every offset after it shifts. The failure is a device
    that launches garbage, which is a long way from the edit that caused it. -/
theorem ptx_fits :
    ptxRmsBytes.length ≤ PTX_SILU_OFF - PTX_RMS_OFF
      ∧ ptxSiluBytes.length ≤ PTX_ADDRMS_OFF - PTX_SILU_OFF
      ∧ ptxAddRmsBytes.length ≤ PTX_ADD_OFF - PTX_ADDRMS_OFF
      ∧ ptxAddBytes.length ≤ MEM_SIZE - PTX_ADD_OFF := by native_decide

def buildInitialMemory : List UInt8 :=
  let pre := zeros PTX_RMS_OFF
  let rms := ptxRmsBytes ++ zeros (PTX_SILU_OFF - PTX_RMS_OFF - ptxRmsBytes.length)
  let silu := ptxSiluBytes ++ zeros (PTX_ADDRMS_OFF - PTX_SILU_OFF - ptxSiluBytes.length)
  let addrms := ptxAddRmsBytes ++ zeros (PTX_ADD_OFF - PTX_ADDRMS_OFF - ptxAddRmsBytes.length)
  let add := ptxAddBytes ++ zeros (MEM_SIZE - PTX_ADD_OFF - ptxAddBytes.length)
  pre ++ rms ++ silu ++ addrms ++ add

def buildSetup (clif : Program) : Setup := {
  clif,
  memory_size := MEM_SIZE,
  initial_memory := buildInitialMemory
}

def loadAlgorithm   : Algorithm := { fn_idx := u32 1 }
def prepAlgorithm   : Algorithm := { fn_idx := u32 2 }
def inferAlgorithm  : Algorithm := { fn_idx := u32 5 }
def stack16Algorithm : Algorithm := { fn_idx := u32 6 }
def stack32Algorithm : Algorithm := { fn_idx := u32 7 }

def artifacts (clif : Program) : Array Json :=
  #[
    toJsonArtifact "cuda_decoder" (buildSetup clif) loadAlgorithm [
      ("prep",    prepAlgorithm),
      ("infer",   inferAlgorithm),
      ("stack16", stack16Algorithm),
      ("stack32", stack32Algorithm)
    ]
  ]

end CudaDecoderLayer
