import Lean
import Std
import AlgorithmLib
import AlgorithmLib.Cuda
import Qwen2Common

set_option maxRecDepth 4096

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.PTX
open AlgorithmLib.Tensor
open Qwen2Common

namespace Qwen2

/-- loadInitFn (fn_1): in-memory init.  Shared prefix allocates pinned scratch
    + activation/embed/lm_head/rope buffers and streams those weights from disk.
    No per-layer alloc here — that's done by `loadLayerFn`. -/
def loadInitFn : IRBuilder Unit := do
  let ptr        ← entryBlock
  let cuda       ← declareCudaFFI
  let fnFileRead ← declareFFI "cl_file_read_to_ptr" [.i64, .i64, .i64, .i64] (some .i64)
  let fnSinf     ← declareFFI "cl_sinf" [.f32]       (some .f32)
  let fnCosf     ← declareFFI "cl_cosf" [.f32]       (some .f32)
  let fnPowf     ← declareFFI "cl_powf" [.f32, .f32] (some .f32)
  loadInitCommon cuda fnFileRead fnSinf fnCosf fnPowf ptr
  ret

/-- loadLayerFn (fn_2+l): create per-layer GPU buffers and stream-upload weights
    via the pinned scratch buffer.  Stores 14 buffer IDs in layer slot. -/
def loadLayerFn (l : Nat) : IRBuilder Unit := do
  let ptr    ← entryBlock
  let cuda   ← declareCudaFFI
  let fnFileRead ← declareFFI "cl_file_read_to_ptr" [.i64, .i64, .i64, .i64] (some .i64)
  let ctxPtr    ← load64 (← absAddr ptr 0x10)
  let pathPtr   ← load64 (← absAddr ptr WEIGHTS_PATH_PTR_OFF)
  let pinnedPtr ← load64 (← absAddr ptr PINNED_HOST_PTR_OFF)

  let layerOff := FILE_LAYER_OFF l

  -- Byte sizes
  let dBytes  ← iconst64 D_BYTES
  let kvBytes ← iconst64 KV_BYTES
  let wqBytes ← iconst64 WQ_BYTES
  let wkBytes ← iconst64 WK_BYTES
  let wgBytes ← iconst64 WG_BYTES
  let kvCacheBytes ← iconst64 KV_CACHE_BYTES

  -- Create weight buffers — shape in the type, byte size in the runtime arg.
  let bufRmsAttn : VecD     ← Tensor.create cuda ptr dBytes
  let bufWq      : MatDD    ← Tensor.create cuda ptr wqBytes
  let bufBq      : VecD     ← Tensor.create cuda ptr dBytes
  let bufWk      : MatKVD   ← Tensor.create cuda ptr wkBytes
  let bufBk      : VecKV    ← Tensor.create cuda ptr kvBytes
  let bufWv      : MatKVD   ← Tensor.create cuda ptr wkBytes
  let bufBv      : VecKV    ← Tensor.create cuda ptr kvBytes
  let bufWo      : MatDD    ← Tensor.create cuda ptr wqBytes
  let bufRmsFfn  : VecD     ← Tensor.create cuda ptr dBytes
  let bufWg      : MatDffD  ← Tensor.create cuda ptr wgBytes
  let bufWu      : MatDffD  ← Tensor.create cuda ptr wgBytes
  let bufWd      : MatDDff  ← Tensor.create cuda ptr wgBytes
  let bufKCache  : KVCache  ← Tensor.create cuda ptr kvCacheBytes
  let bufVCache  : KVCache  ← Tensor.create cuda ptr kvCacheBytes

  -- Store buffer IDs into layer `l`'s slot (cell base = ptr + LAYER_BUFS_BASE + l*STRIDE).
  let cellBase ← absAddr ptr (LAYER_BUFS_BASE + l * LAYER_BUF_STRIDE)
  LayerSlot.rmsAttn.store cellBase bufRmsAttn
  LayerSlot.wq.store      cellBase bufWq
  LayerSlot.bq.store      cellBase bufBq
  LayerSlot.wk.store      cellBase bufWk
  LayerSlot.bk.store      cellBase bufBk
  LayerSlot.wv.store      cellBase bufWv
  LayerSlot.bv.store      cellBase bufBv
  LayerSlot.wo.store      cellBase bufWo
  LayerSlot.rmsFfn.store  cellBase bufRmsFfn
  LayerSlot.wg.store      cellBase bufWg
  LayerSlot.wu.store      cellBase bufWu
  LayerSlot.wd.store      cellBase bufWd
  LayerSlot.kCache.store  cellBase bufKCache
  LayerSlot.vCache.store  cellBase bufVCache

  -- Stream-upload weight tensors through pinned scratch
  let up {s : Shape} (t : Tensor s) (fileOff size : Nat) : IRBuilder Unit :=
    uploadFromFile cuda fnFileRead ctxPtr pathPtr pinnedPtr t (layerOff + fileOff) size
  up bufRmsAttn LF_RMS_ATTN D_BYTES
  up bufWq      LF_WQ       WQ_BYTES
  up bufBq      LF_BQ       D_BYTES
  up bufWk      LF_WK       WK_BYTES
  up bufBk      LF_BK       KV_BYTES
  up bufWv      LF_WV       WK_BYTES
  up bufBv      LF_BV       KV_BYTES
  up bufWo      LF_WO       WQ_BYTES
  up bufRmsFfn  LF_RMS_FFN  D_BYTES
  up bufWg      LF_WG       WG_BYTES
  up bufWu      LF_WU       WG_BYTES
  up bufWd      LF_WD       WG_BYTES
  ret

/-- loadFinalizeFn (fn_26): sync GPU then free the pinned scratch buffer. -/
def loadFinalizeFn : IRBuilder Unit := do
  let ptr      ← entryBlock
  let cuda     ← declareCudaFFI
  let ctxPtr   ← load64 (← absAddr ptr 0x10)
  let pinnedId ← load32 (← absAddr ptr PINNED_ID_OFF)
  let _ ← cudaSync cuda ptr 0x10
  let _ ← call cuda.fnPinnedFree [ctxPtr, pinnedId]
  ret

/-- inferLayerFn (fn_28): runs one transformer layer — calls attn then ffn. -/
def inferLayerFn : IRBuilder Unit := do
  let ptr    ← entryBlock
  let fnAttn ← declareColocatedFFI "fn_29" [.i64] none
  let fnFfn  ← declareColocatedFFI "fn_30" [.i64] none
  callVoid fnAttn [ptr]
  callVoid fnFfn  [ptr]
  ret

/-- Compute the per-layer slot base address for the current `LAYER_IDX_OFF`. -/
private def currentLayerSlot (ptr : Val) : IRBuilder Val := do
  let layerIdx ← load64 (← absAddr ptr LAYER_IDX_OFF)
  let stride64 ← iconst64 LAYER_BUF_STRIDE
  let base64   ← iconst64 LAYER_BUFS_BASE
  let slotOff  ← imul layerIdx stride64
  let slotBase ← iadd base64 slotOff
  iadd ptr slotBase

/-- inferLayerAttnFn (fn_29): attention sub-layer.  Reads per-layer slot from
    `LAYER_BUFS_BASE + layerIdx * STRIDE`. -/
def inferLayerAttnFn : IRBuilder Unit := do
  let ptr       ← entryBlock
  let cuda      ← declareCudaFFI
  let blas      ← declareCuBlasFFI
  let slotBaseA ← currentLayerSlot ptr
  attnBody cuda blas ptr slotBaseA

/-- inferLayerFfnFn (fn_30): FFN sub-layer.  Same slot lookup as attn. -/
def inferLayerFfnFn : IRBuilder Unit := do
  let ptr       ← entryBlock
  let cuda      ← declareCudaFFI
  let blas      ← declareCuBlasFFI
  let slotBaseA ← currentLayerSlot ptr
  ffnBody cuda blas ptr slotBaseA

-- ── CLIF IR ──────────────────────────────────────────────────────────────────

def clifIR : String :=
  noopFunction ++ "\n" ++
  buildFunction 1 loadInitFn ++ "\n" ++
  (List.range N_LAYERS).foldl
    (fun acc l => acc ++ buildFunction (2 + l) (loadLayerFn l) ++ "\n") "" ++
  buildFunction 26 loadFinalizeFn ++ "\n" ++
  buildFunction 27 inferFn ++ "\n" ++
  buildFunction 28 inferLayerFn ++ "\n" ++
  buildFunction 29 inferLayerAttnFn ++ "\n" ++
  buildFunction 30 inferLayerFfnFn ++ "\n" ++
  buildFunction 31 inferFinalFn ++ "\n" ++
  buildFunction 32 loadTokenizerFn ++ "\n" ++
  buildFunction 33 tokenizeInitFn ++ "\n" ++
  buildFunction 34 tokenizeBpeFn ++ "\n" ++
  buildFunction 35 detokenizeFn ++ "\n" ++
  buildFunction 36 cliFn ++ "\n" ++
  buildFunction 37 parseArgsFn ++
  -- fn38: orchestrator wrapper — parse args (37), load weights (1..26),
  --       load tokenizer (32), server (36 — runs forever).
  clifSequenceWrapper 38
    (37 :: (List.range 26).map (fun i => i + 1) ++ [32, 36])

-- ── Initial memory ───────────────────────────────────────────────────────────

def buildInitialMemory : List UInt8 :=
  let sysBlock := systemTokenBytes ++ zeros (PTX_EMBED_OFF - SYSTEM_TOKENS_OFF - systemTokenBytes.length)
  zeros SYSTEM_TOKENS_OFF ++ sysBlock ++ buildInitialMemoryTail

-- ── Algorithm definition ─────────────────────────────────────────────────────

def buildSetup : Setup := {
  cranelift_ir := clifIR,
  memory_size := MEM_SIZE,
  initial_memory := buildInitialMemory
}

/-- Single end-to-end algorithm: parse args → load weights → load tokenizer → server.
    `data` must be `weights_path\0tokenizer_path\0`. The orchestrator is `fn38`,
    a CLIF wrapper that calls each step in sequence (see `clifIR`). -/
def qwen2Algorithm : Algorithm := { fn_idx := u32 38 }

end Qwen2

def main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  emitArtifacts outDir #[
    toJsonEntry "qwen2" Qwen2.buildSetup Qwen2.qwen2Algorithm
  ]
