import Lean
import Std
import AlgorithmLib.Gen
import AlgorithmLib.Cuda
import AlgorithmLib.ProgCuda
import Qwen2Common
import LayoutScan
import ShipScan


open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.PTX
open AlgorithmLib.Tensor
open Qwen2Common
open AlgorithmLib.Prog

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
def loadInitFn : Prog V L Unit := do
  let ptr ← basePtr
  loadInitCommon ptr

  -- Working-set weight buffers — one set, reused across all 24 layers.
  -- streamLayerFn rewrites their contents per-layer from disk.
  let wsBaseA ← absAddr ptr WORKING_SET_BASE
  let dBytes  ← iconst64 D_BYTES
  let bufRmsAttn : VecD V    ← tensorCreate ptr dBytes
  let bufWq      : MatDD V   ← tensorCreate ptr (← iconst64 WQ_BYTES)
  let bufBq      : VecD V    ← tensorCreate ptr dBytes
  let bufWk      : MatKVD V  ← tensorCreate ptr (← iconst64 WK_BYTES)
  let bufBk      : VecKV V   ← tensorCreate ptr (← iconst64 KV_BYTES)
  let bufWv      : MatKVD V  ← tensorCreate ptr (← iconst64 WK_BYTES)
  let bufBv      : VecKV V   ← tensorCreate ptr (← iconst64 KV_BYTES)
  let bufWo      : MatDD V   ← tensorCreate ptr (← iconst64 WQ_BYTES)
  let bufRmsFfn  : VecD V    ← tensorCreate ptr dBytes
  let bufWg      : MatDffD V ← tensorCreate ptr (← iconst64 WG_BYTES)
  let bufWu      : MatDffD V ← tensorCreate ptr (← iconst64 WG_BYTES)
  let bufWd      : MatDDff V ← tensorCreate ptr (← iconst64 WG_BYTES)
  slotStore LayerSlot.rmsAttn wsBaseA bufRmsAttn
  slotStore LayerSlot.wq      wsBaseA bufWq
  slotStore LayerSlot.bq      wsBaseA bufBq
  slotStore LayerSlot.wk      wsBaseA bufWk
  slotStore LayerSlot.bk      wsBaseA bufBk
  slotStore LayerSlot.wv      wsBaseA bufWv
  slotStore LayerSlot.bv      wsBaseA bufBv
  slotStore LayerSlot.wo      wsBaseA bufWo
  slotStore LayerSlot.rmsFfn  wsBaseA bufRmsFfn
  slotStore LayerSlot.wg      wsBaseA bufWg
  slotStore LayerSlot.wu      wsBaseA bufWu
  slotStore LayerSlot.wd      wsBaseA bufWd

  -- Shared K/V cache buffers — stored at the kCache/vCache offsets within the
  -- same working-set slot, so attnLoadBufs unifies with the in-memory variant.
  let kvCacheBytes ← iconst64 KV_CACHE_BYTES
  let bufKvK : KVCache V ← tensorCreate ptr kvCacheBytes
  let bufKvV : KVCache V ← tensorCreate ptr kvCacheBytes
  slotStore LayerSlot.kCache wsBaseA bufKvK
  slotStore LayerSlot.vCache wsBaseA bufKvV

/-- loadLayerFn (fn_2+l): no-op.  Per-layer state is streamed from disk on
    demand by `streamLayerFn` + `kvLoadLayerFn` just before each layer runs. -/
def loadLayerFn (_l : Nat) : Prog V L Unit := pure ()

/-- loadFinalizeFn (fn_26): sync GPU.  Pinned scratch stays alive for the
    program's lifetime; streamLayerFn re-uses it on every layer. -/
def loadFinalizeFn : Prog V L Unit := do
  let ptr ← basePtr
  let _ ← cudaSync ptr 0x10

-- ── inferLayerFn: stream weights/KV → attn → ffn → save KV ───────────────────

/-- inferLayerFn (fn_28): 5-step layer pipeline.
    1) stream weights from disk into the working set,
    2) stream K/V history from disk into the shared cache,
    3) attention (which also writes the new K/V slot via kvStoreKernel),
    4) FFN,
    5) save new K/V slot back to disk for next-token retrieval. -/
def inferLayerFn : Prog V L Unit := do
  callLocalVoid Qwen2Common.q.fnStream (← entryArgs)
  callLocalVoid Qwen2Common.q.fnKvLoad (← entryArgs)
  callLocalVoid Qwen2Common.q.fnAttn   (← entryArgs)
  callLocalVoid Qwen2Common.q.fnFfn    (← entryArgs)
  callLocalVoid Qwen2Common.q.fnKvSave (← entryArgs)

/-- inferLayerAttnFn (fn_29): attention sub-layer.  Slot base = working set. -/
def inferLayerAttnFn : Prog V L Unit := do
  let ptr ← basePtr
  let slotBaseA ← absAddr ptr WORKING_SET_BASE
  attnBody ptr slotBaseA

/-- inferLayerFfnFn (fn_30): FFN sub-layer.  Slot base = working set. -/
def inferLayerFfnFn : Prog V L Unit := do
  let ptr ← basePtr
  let slotBaseA ← absAddr ptr WORKING_SET_BASE
  ffnBody ptr slotBaseA

-- ── streamLayerFn / kvLoadLayerFn / kvSaveLayerFn ────────────────────────────

/-- streamLayerFn (fn_38): pull this layer's 12 weight tensors from disk into
    the GPU working-set buffers.  One big file-read + 12 H→D uploads. -/
def streamLayerFn : Prog V L Unit := do
  let ptr ← basePtr
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
  let _ ← ffi .fileReadToPtr %[pathPtr, pinnedPtr, baseOff, layerBytes64]
  -- 12 synchronous H→D uploads.  Sync semantics keep the pinned scratch safe
  -- to reuse on the next streamLayerFn call without an extra device sync.
  let upOne (bufId : V .i32) (scratchOff size : Nat) : Prog V L Unit := do
    let pinnedAt ← iaddImm pinnedPtr scratchOff
    let size64   ← iconst64 size
    let _ ← ffi .cudaUpload %[ctxPtr, bufId, pinnedAt, size64]
  let tRms  ← slotLoad LayerSlot.rmsAttn wsBaseA; upOne tRms.buf  LF_RMS_ATTN D_BYTES
  let tWq   ← slotLoad LayerSlot.wq      wsBaseA; upOne tWq.buf   LF_WQ       WQ_BYTES
  let tBq   ← slotLoad LayerSlot.bq      wsBaseA; upOne tBq.buf   LF_BQ       D_BYTES
  let tWk   ← slotLoad LayerSlot.wk      wsBaseA; upOne tWk.buf   LF_WK       WK_BYTES
  let tBk   ← slotLoad LayerSlot.bk      wsBaseA; upOne tBk.buf   LF_BK       KV_BYTES
  let tWv   ← slotLoad LayerSlot.wv      wsBaseA; upOne tWv.buf   LF_WV       WK_BYTES
  let tBv   ← slotLoad LayerSlot.bv      wsBaseA; upOne tBv.buf   LF_BV       KV_BYTES
  let tWo   ← slotLoad LayerSlot.wo      wsBaseA; upOne tWo.buf   LF_WO       WQ_BYTES
  let tRmsF ← slotLoad LayerSlot.rmsFfn  wsBaseA; upOne tRmsF.buf LF_RMS_FFN  D_BYTES
  let tWg   ← slotLoad LayerSlot.wg      wsBaseA; upOne tWg.buf   LF_WG       WG_BYTES
  let tWu   ← slotLoad LayerSlot.wu      wsBaseA; upOne tWu.buf   LF_WU       WG_BYTES
  let tWd   ← slotLoad LayerSlot.wd      wsBaseA; upOne tWd.buf   LF_WD       WG_BYTES

/-- kvLoadLayerFn (fn_39): stream this layer's K/V cache history from disk into
    the shared K/V VRAM buffers.  Skips when pos==0 (no history yet). -/
def kvLoadLayerFn : Prog V L Unit := do
  let ptr ← basePtr
  let ctxPtr     ← load64 (← absAddr ptr 0x10)
  let pinnedPtr  ← load64 (← absAddr ptr PINNED_HOST_PTR_OFF)
  let pos        ← load64 (← absAddr ptr POS_SLOT_OFF)
  let layerIdx   ← load64 (← absAddr ptr LAYER_IDX_OFF)
  let wsBaseA    ← absAddr ptr WORKING_SET_BASE
  let tK         ← slotLoad LayerSlot.kCache wsBaseA
  let tV         ← slotLoad LayerSlot.vCache wsBaseA
  let kvPath     ← absAddr ptr KV_CACHE_PATH_OFF
  let kvBytes64  ← iconst64 KV_CACHE_BYTES
  let perLayer64 ← iconst64 (2 * KV_CACHE_BYTES)
  let kFileOff   ← imul layerIdx perLayer64
  let vFileOff   ← iadd kFileOff kvBytes64
  let zero64     ← iconst64 0
  -- position zero has nothing cached yet
  when .ne pos zero64 (do
    let _ ← ffi .fileReadToPtr %[kvPath, pinnedPtr, kFileOff, kvBytes64]
    let _ ← ffi .cudaUpload %[ctxPtr, tK.buf, pinnedPtr, kvBytes64]
    let _ ← ffi .fileReadToPtr %[kvPath, pinnedPtr, vFileOff, kvBytes64]
    let _ ← ffi .cudaUpload %[ctxPtr, tV.buf, pinnedPtr, kvBytes64]
    pure ())

/-- kvSaveLayerFn (fn_40): write this layer's newly-computed K/V slot at the
    current position back to disk so the next token can stream it back in. -/
def kvSaveLayerFn : Prog V L Unit := do
  let ptr ← basePtr
  let ctxPtr    ← load64 (← absAddr ptr 0x10)
  let pinnedPtr ← load64 (← absAddr ptr PINNED_HOST_PTR_OFF)
  let pos       ← load64 (← absAddr ptr POS_SLOT_OFF)
  let layerIdx  ← load64 (← absAddr ptr LAYER_IDX_OFF)
  let wsBaseA   ← absAddr ptr WORKING_SET_BASE
  let tK        ← slotLoad LayerSlot.kCache wsBaseA
  let tV        ← slotLoad LayerSlot.vCache wsBaseA
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
    let _ ← ffi .cudaDownloadOffset %[ctxPtr, tK.buf, slotOff, pinnedPtr, slotBytes64]
    let _ ← ffi .fileWriteFromPtr           %[kvPath, pinnedPtr, kSlotFile, slotBytes64]
    let _ ← ffi .cudaDownloadOffset %[ctxPtr, tV.buf, slotOff, pinnedPtr, slotBytes64]
    let _ ← ffi .fileWriteFromPtr           %[kvPath, pinnedPtr, vSlotFile, slotBytes64]

-- ── CLIF IR ──────────────────────────────────────────────────────────────────

/-- The bodies this artifact ships, in the order their function indices run.
    `clifIR` numbers them from this list, so an index cannot drift from the body
    it names. -/
def shippedBodies : List Prog.Body :=
  [loadInitFn]
  ++ (List.range N_LAYERS).map loadLayerFn
  ++ [loadFinalizeFn, inferFn, inferLayerFn, inferLayerAttnFn, inferLayerFfnFn,
      inferFinalFn, loadTokenizerFn, tokenizeInitFn, tokenizeBpeFn, detokenizeFn,
      cliFn, parseArgsFn, streamLayerFn, kvLoadLayerFn, kvSaveLayerFn]

/-- The orchestrator's callees: parse args (37), load the weights (1..26),
    load the tokenizer (32), then serve (36 --- which runs forever). -/
def wrapperCallees : List Nat :=
  37 :: (List.range 26).map (fun i => i + 1) ++ [32, 36]

def clifIR : Except String (List FuncData) :=
  Prog.program <|
    (.ok noopFunction :: shippedBodies.zipIdx.map
      (fun p => Prog.compileProg (p.2 + 1) p.1))
    ++ [Prog.entry "main" (Prog.compileProg 41 (Prog.sequenceWrapper wrapperCallees))]

-- ── Initial memory ───────────────────────────────────────────────────────────

def buildInitialMemory : List UInt8 :=
  let kvPath   := kvCachePathBytes  ++ zeros (SYSTEM_TOKENS_OFF - KV_CACHE_PATH_OFF - kvCachePathBytes.length)
  let sysBlock := systemTokenBytes  ++ zeros (PTX_EMBED_OFF - SYSTEM_TOKENS_OFF - systemTokenBytes.length)
  zeros KV_CACHE_PATH_OFF ++ kvPath ++ sysBlock ++ buildInitialMemoryTail

-- ── Algorithm definition ─────────────────────────────────────────────────────

def buildSetup (clif : List FuncData) : Artifact := {
  functions := clif,
  memory_size := MEM_SIZE,
  initial_memory := buildInitialMemory
}

/-- Orchestrator at `fn41` runs the full pipeline (see `clifIR`). -/
def qwen2OnDiskAlgorithm : UInt32 := 41

-- ── The memory map, as data ──────────────────────────────────────────────────

/-- The shared map plus this variant's two extra regions.  Both live in the
    0x06xx band, which is the most crowded part of the layout — `WORKING_SET_BASE`
    sits between `running_pos` and `system_tokens`, and the KV-cache path string
    between that and `system_tokens` again.  Checking them here means the
    in-memory variant cannot claim that space without breaking this build. -/
def memMapOnDisk : AlgorithmLib.Layout.RegionMap :=
  Qwen2Common.memMap ++
    [ ⟨"working_set",   WORKING_SET_BASE, LAYER_BUF_STRIDE⟩,
      ⟨"kv_cache_path", KV_CACHE_PATH_OFF, kvCachePathBytes.length⟩ ]


#eval LayoutScan.check "Qwen2OnDiskAlgorithm" [``memMapOnDisk]
theorem memMapOnDisk_ok : memMapOnDisk.okB = true := by native_decide

theorem memMapOnDisk_within :
    AlgorithmLib.Layout.RegionMap.withinB MEM_SIZE memMapOnDisk = true := by
  native_decide

end Qwen2OnDisk

def main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  let clif ← Prog.orDie Qwen2OnDisk.clifIR
  emitArtifacts outDir #[
    toJsonArtifact "qwen2_on_disk" (Qwen2OnDisk.buildSetup clif)
  ]

#eval ShipScan.check "Qwen2OnDiskAlgorithm"
