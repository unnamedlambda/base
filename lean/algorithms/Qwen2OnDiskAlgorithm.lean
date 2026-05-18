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

namespace Qwen2OnDisk

-- ── On-disk-specific layout ──────────────────────────────────────────────────

/-- The on-disk variant keeps just ONE layer's worth of weight buffers + ONE
    pair of K/V cache buffers in VRAM, reused across all 24 layers.  The slot
    follows the same `LayerSlot` shape as the in-memory variant's per-layer
    cells, so `attnBody`/`ffnBody` work unchanged when handed this address.

    Slot layout (56 bytes, mirrors `LayerSlot`):
      0  rmsAttn / 4 wq / 8 bq / 12 wk / 16 bk / 20 wv / 24 bv / 28 wo
      32 rmsFfn / 36 wg / 40 wu / 44 wd / 48 kCache / 52 vCache
    Lives at 0x0660, fitting between `RUNNING_POS_OFF` and `SYSTEM_TOKENS_OFF`. -/
def WORKING_SET_BASE : Nat := 0x0660

/-- KV cache backing file path bytes live in initial_memory at this fixed
    offset so streamLayer/kvLoad/kvSave can resolve it cheaply.  Park it in
    the free slot between `WORKING_SET_BASE` (0x0660..0x0697) and
    `SYSTEM_TOKENS_OFF` (0x0700) — 104 bytes available, path is 23. -/
def KV_CACHE_PATH_OFF : Nat := 0x06A0

/-- Hardcoded for now; future work could parameterize via parseArgsFn. -/
def kvCachePathBytes : List UInt8 :=
  "/tmp/qwen2_kvcache.bin".toUTF8.toList ++ [0]

-- ── Init / per-layer / finalize ──────────────────────────────────────────────

/-- loadInitFn (fn_1): shared prefix, then allocate the working-set slot
    (12 weight buffers + shared K/V cache pair) reused across all layers. -/
def loadInitFn : IRBuilder Unit := do
  let ptr        ← entryBlock
  let cuda       ← declareCudaFFI
  let fnFileRead ← declareFFI "cl_file_read_to_ptr" [.i64, .i64, .i64, .i64] (some .i64)
  let fnSinf     ← declareFFI "cl_sinf" [.f32]       (some .f32)
  let fnCosf     ← declareFFI "cl_cosf" [.f32]       (some .f32)
  let fnPowf     ← declareFFI "cl_powf" [.f32, .f32] (some .f32)
  loadInitCommon cuda fnFileRead fnSinf fnCosf fnPowf ptr

  -- Working-set weight buffers — one set, reused across all 24 layers.
  -- streamLayerFn rewrites their contents per-layer from disk.
  let wsBaseA ← absAddr ptr WORKING_SET_BASE
  let dBytes  ← iconst64 D_BYTES
  let bufRmsAttn : VecD    ← Tensor.create cuda ptr dBytes
  let bufWq      : MatDD   ← Tensor.create cuda ptr (← iconst64 WQ_BYTES)
  let bufBq      : VecD    ← Tensor.create cuda ptr dBytes
  let bufWk      : MatKVD  ← Tensor.create cuda ptr (← iconst64 WK_BYTES)
  let bufBk      : VecKV   ← Tensor.create cuda ptr (← iconst64 KV_BYTES)
  let bufWv      : MatKVD  ← Tensor.create cuda ptr (← iconst64 WK_BYTES)
  let bufBv      : VecKV   ← Tensor.create cuda ptr (← iconst64 KV_BYTES)
  let bufWo      : MatDD   ← Tensor.create cuda ptr (← iconst64 WQ_BYTES)
  let bufRmsFfn  : VecD    ← Tensor.create cuda ptr dBytes
  let bufWg      : MatDffD ← Tensor.create cuda ptr (← iconst64 WG_BYTES)
  let bufWu      : MatDffD ← Tensor.create cuda ptr (← iconst64 WG_BYTES)
  let bufWd      : MatDDff ← Tensor.create cuda ptr (← iconst64 WG_BYTES)
  LayerSlot.rmsAttn.store wsBaseA bufRmsAttn
  LayerSlot.wq.store      wsBaseA bufWq
  LayerSlot.bq.store      wsBaseA bufBq
  LayerSlot.wk.store      wsBaseA bufWk
  LayerSlot.bk.store      wsBaseA bufBk
  LayerSlot.wv.store      wsBaseA bufWv
  LayerSlot.bv.store      wsBaseA bufBv
  LayerSlot.wo.store      wsBaseA bufWo
  LayerSlot.rmsFfn.store  wsBaseA bufRmsFfn
  LayerSlot.wg.store      wsBaseA bufWg
  LayerSlot.wu.store      wsBaseA bufWu
  LayerSlot.wd.store      wsBaseA bufWd

  -- Shared K/V cache buffers — stored at the kCache/vCache offsets within the
  -- same working-set slot, so attnLoadBufs unifies with the in-memory variant.
  let kvCacheBytes ← iconst64 KV_CACHE_BYTES
  let bufKvK : KVCache ← Tensor.create cuda ptr kvCacheBytes
  let bufKvV : KVCache ← Tensor.create cuda ptr kvCacheBytes
  LayerSlot.kCache.store wsBaseA bufKvK
  LayerSlot.vCache.store wsBaseA bufKvV
  ret

/-- loadLayerFn (fn_2+l): no-op.  Per-layer state is streamed from disk on
    demand by `streamLayerFn` + `kvLoadLayerFn` just before each layer runs. -/
def loadLayerFn (_l : Nat) : IRBuilder Unit := do
  let _ptr ← entryBlock
  ret

/-- loadFinalizeFn (fn_26): sync GPU.  Pinned scratch stays alive for the
    program's lifetime; streamLayerFn re-uses it on every layer. -/
def loadFinalizeFn : IRBuilder Unit := do
  let ptr  ← entryBlock
  let cuda ← declareCudaFFI
  let _ ← cudaSync cuda ptr 0x10
  ret

-- ── inferLayerFn: stream weights/KV → attn → ffn → save KV ───────────────────

/-- inferLayerFn (fn_28): 5-step layer pipeline.
    1) stream weights from disk into the working set,
    2) stream K/V history from disk into the shared cache,
    3) attention (which also writes the new K/V slot via kvStoreKernel),
    4) FFN,
    5) save new K/V slot back to disk for next-token retrieval. -/
def inferLayerFn : IRBuilder Unit := do
  let ptr      ← entryBlock
  let fnStream ← declareColocatedFFI "fn_38" [.i64] none
  let fnKvLoad ← declareColocatedFFI "fn_39" [.i64] none
  let fnAttn   ← declareColocatedFFI "fn_29" [.i64] none
  let fnFfn    ← declareColocatedFFI "fn_30" [.i64] none
  let fnKvSave ← declareColocatedFFI "fn_40" [.i64] none
  callVoid fnStream [ptr]
  callVoid fnKvLoad [ptr]
  callVoid fnAttn   [ptr]
  callVoid fnFfn    [ptr]
  callVoid fnKvSave [ptr]
  ret

/-- inferLayerAttnFn (fn_29): attention sub-layer.  Slot base = working set. -/
def inferLayerAttnFn : IRBuilder Unit := do
  let ptr       ← entryBlock
  let cuda      ← declareCudaFFI
  let blas      ← declareCuBlasFFI
  let slotBaseA ← absAddr ptr WORKING_SET_BASE
  attnBody cuda blas ptr slotBaseA

/-- inferLayerFfnFn (fn_30): FFN sub-layer.  Slot base = working set. -/
def inferLayerFfnFn : IRBuilder Unit := do
  let ptr       ← entryBlock
  let cuda      ← declareCudaFFI
  let blas      ← declareCuBlasFFI
  let slotBaseA ← absAddr ptr WORKING_SET_BASE
  ffnBody cuda blas ptr slotBaseA

-- ── streamLayerFn / kvLoadLayerFn / kvSaveLayerFn ────────────────────────────

/-- streamLayerFn (fn_38): pull this layer's 12 weight tensors from disk into
    the GPU working-set buffers.  One big file-read + 12 H→D uploads. -/
def streamLayerFn : IRBuilder Unit := do
  let ptr        ← entryBlock
  let cuda       ← declareCudaFFI
  let fnFileRead ← declareFFI "cl_file_read_to_ptr" [.i64, .i64, .i64, .i64] (some .i64)
  let ctxPtr    ← load64 (← absAddr ptr 0x10)
  let pathPtr   ← load64 (← absAddr ptr WEIGHTS_PATH_PTR_OFF)
  let pinnedPtr ← load64 (← absAddr ptr PINNED_HOST_PTR_OFF)
  let layerIdx  ← load64 (← absAddr ptr LAYER_IDX_OFF)
  -- File offset: EMBED_BYTES + layerIdx * LAYER_BYTES
  let layerBytes64 ← iconst64 LAYER_BYTES
  let layerSpan    ← imul layerIdx layerBytes64
  let embedBytes64 ← iconst64 EMBED_BYTES
  let baseOff      ← iadd embedBytes64 layerSpan
  let wsBaseA      ← absAddr ptr WORKING_SET_BASE
  -- ONE big read pulls the entire layer (~57 MB) into pinned scratch in one
  -- syscall.  The on-disk layout matches the pinned scratch layout 1:1.
  let _ ← call fnFileRead [pathPtr, pinnedPtr, baseOff, layerBytes64]
  -- 12 synchronous H→D uploads.  Sync semantics keep the pinned scratch safe
  -- to reuse on the next streamLayerFn call without an extra device sync.
  let upOne (bufId : Val) (scratchOff size : Nat) : IRBuilder Unit := do
    let pinnedAt ← iaddImm pinnedPtr scratchOff
    let size64   ← iconst64 size
    let _ ← call cuda.fnUpload [ctxPtr, bufId, pinnedAt, size64]
  let tRms  ← LayerSlot.rmsAttn.load wsBaseA; upOne tRms.buf  LF_RMS_ATTN D_BYTES
  let tWq   ← LayerSlot.wq.load      wsBaseA; upOne tWq.buf   LF_WQ       WQ_BYTES
  let tBq   ← LayerSlot.bq.load      wsBaseA; upOne tBq.buf   LF_BQ       D_BYTES
  let tWk   ← LayerSlot.wk.load      wsBaseA; upOne tWk.buf   LF_WK       WK_BYTES
  let tBk   ← LayerSlot.bk.load      wsBaseA; upOne tBk.buf   LF_BK       KV_BYTES
  let tWv   ← LayerSlot.wv.load      wsBaseA; upOne tWv.buf   LF_WV       WK_BYTES
  let tBv   ← LayerSlot.bv.load      wsBaseA; upOne tBv.buf   LF_BV       KV_BYTES
  let tWo   ← LayerSlot.wo.load      wsBaseA; upOne tWo.buf   LF_WO       WQ_BYTES
  let tRmsF ← LayerSlot.rmsFfn.load  wsBaseA; upOne tRmsF.buf LF_RMS_FFN  D_BYTES
  let tWg   ← LayerSlot.wg.load      wsBaseA; upOne tWg.buf   LF_WG       WG_BYTES
  let tWu   ← LayerSlot.wu.load      wsBaseA; upOne tWu.buf   LF_WU       WG_BYTES
  let tWd   ← LayerSlot.wd.load      wsBaseA; upOne tWd.buf   LF_WD       WG_BYTES
  ret

/-- kvLoadLayerFn (fn_39): stream this layer's K/V cache history from disk into
    the shared K/V VRAM buffers.  Skips when pos==0 (no history yet). -/
def kvLoadLayerFn : IRBuilder Unit := do
  let ptr        ← entryBlock
  let cuda       ← declareCudaFFI
  let fnFileRead ← declareFFI "cl_file_read_to_ptr" [.i64, .i64, .i64, .i64] (some .i64)
  let ctxPtr     ← load64 (← absAddr ptr 0x10)
  let pinnedPtr  ← load64 (← absAddr ptr PINNED_HOST_PTR_OFF)
  let pos        ← load64 (← absAddr ptr POS_SLOT_OFF)
  let layerIdx   ← load64 (← absAddr ptr LAYER_IDX_OFF)
  let wsBaseA    ← absAddr ptr WORKING_SET_BASE
  let tK         ← LayerSlot.kCache.load wsBaseA
  let tV         ← LayerSlot.vCache.load wsBaseA
  let kvPath     ← absAddr ptr KV_CACHE_PATH_OFF
  let kvBytes64  ← iconst64 KV_CACHE_BYTES
  let perLayer64 ← iconst64 (2 * KV_CACHE_BYTES)
  let kFileOff   ← imul layerIdx perLayer64
  let vFileOff   ← iadd kFileOff kvBytes64
  let zero64     ← iconst64 0
  let posIsZero  ← icmp .eq pos zero64
  let doLoad     ← declareBlock []
  let doneBlk    ← declareBlock []
  brif posIsZero doneBlk.ref [] doLoad.ref []
  startBlock doLoad
  let _ ← call fnFileRead    [kvPath, pinnedPtr, kFileOff, kvBytes64]
  let _ ← call cuda.fnUpload [ctxPtr, tK.buf, pinnedPtr, kvBytes64]
  let _ ← call fnFileRead    [kvPath, pinnedPtr, vFileOff, kvBytes64]
  let _ ← call cuda.fnUpload [ctxPtr, tV.buf, pinnedPtr, kvBytes64]
  jump doneBlk.ref []
  startBlock doneBlk
  ret

/-- kvSaveLayerFn (fn_40): write this layer's newly-computed K/V slot at the
    current position back to disk so the next token can stream it back in. -/
def kvSaveLayerFn : IRBuilder Unit := do
  let ptr         ← entryBlock
  let cuda        ← declareCudaFFI
  let fnFileWrite ← declareFFI "cl_file_write_from_ptr" [.i64, .i64, .i64, .i64] (some .i64)
  let ctxPtr    ← load64 (← absAddr ptr 0x10)
  let pinnedPtr ← load64 (← absAddr ptr PINNED_HOST_PTR_OFF)
  let pos       ← load64 (← absAddr ptr POS_SLOT_OFF)
  let layerIdx  ← load64 (← absAddr ptr LAYER_IDX_OFF)
  let wsBaseA   ← absAddr ptr WORKING_SET_BASE
  let tK        ← LayerSlot.kCache.load wsBaseA
  let tV        ← LayerSlot.vCache.load wsBaseA
  let kvPath    ← absAddr ptr KV_CACHE_PATH_OFF
  let perLayer64  ← iconst64 (2 * KV_CACHE_BYTES)
  let kvBytes64   ← iconst64 KV_CACHE_BYTES
  let kFileOff    ← imul layerIdx perLayer64
  let vFileOff    ← iadd kFileOff kvBytes64
  let slotBytes64 ← iconst64 (HEAD_DIM * 4)
  let posByteOff  ← imul pos slotBytes64
  -- Per kv-head: slot byte-offset within K (or V) buffer is
  --   h * MAX_SEQ * HEAD_DIM * 4 + pos * HEAD_DIM * 4
  for h in List.range N_KV do
    let headByteOff   := h * MAX_SEQ * HEAD_DIM * 4
    let headByteOff64 ← iconst64 headByteOff
    let slotOff       ← iadd headByteOff64 posByteOff
    let kSlotFile     ← iadd kFileOff slotOff
    let vSlotFile     ← iadd vFileOff slotOff
    let _ ← call cuda.fnDownloadOffset [ctxPtr, tK.buf, slotOff, pinnedPtr, slotBytes64]
    let _ ← call fnFileWrite           [kvPath, pinnedPtr, kSlotFile, slotBytes64]
    let _ ← call cuda.fnDownloadOffset [ctxPtr, tV.buf, slotOff, pinnedPtr, slotBytes64]
    let _ ← call fnFileWrite           [kvPath, pinnedPtr, vSlotFile, slotBytes64]
  ret

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
  buildFunction 37 parseArgsFn ++ "\n" ++
  buildFunction 38 streamLayerFn ++ "\n" ++
  buildFunction 39 kvLoadLayerFn ++ "\n" ++
  buildFunction 40 kvSaveLayerFn ++
  -- fn41: orchestrator wrapper — parse args (37), load weights (1..26),
  --       load tokenizer (32), server (36 — runs forever).
  clifSequenceWrapper 41
    (37 :: (List.range 26).map (fun i => i + 1) ++ [32, 36])

-- ── Initial memory ───────────────────────────────────────────────────────────

def buildInitialMemory : List UInt8 :=
  let kvPath   := kvCachePathBytes  ++ zeros (SYSTEM_TOKENS_OFF - KV_CACHE_PATH_OFF - kvCachePathBytes.length)
  let sysBlock := systemTokenBytes  ++ zeros (PTX_EMBED_OFF - SYSTEM_TOKENS_OFF - systemTokenBytes.length)
  zeros KV_CACHE_PATH_OFF ++ kvPath ++ sysBlock ++ buildInitialMemoryTail

-- ── Algorithm definition ─────────────────────────────────────────────────────

def buildSetup : Setup := {
  cranelift_ir := clifIR,
  memory_size := MEM_SIZE,
  context_offset := 0,
  initial_memory := buildInitialMemory
}

/-- Orchestrator at `fn41` runs the full pipeline (see `clifIR`). -/
def qwen2OnDiskAlgorithm : Algorithm := { fn_idx := u32 41 }

end Qwen2OnDisk

def main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  emitArtifacts outDir #[
    toJsonEntry "qwen2_on_disk" Qwen2OnDisk.buildSetup Qwen2OnDisk.qwen2OnDiskAlgorithm
  ]
