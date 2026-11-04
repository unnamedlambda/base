import AlgorithmLib.Check.Entry
import AlgorithmLib.Proof.Typestate

/-!
# A program calling the CUDA driver directly neither misuses it nor faults

The driver's own contract is the one the checker reads (`Check.CudaRule`):
the program brings the driver and the device's primary context up, makes an
allocation, copies the caller's data to the device and back to the caller's
output, and frees the allocation. Where each pointer comes from is in the
typestate: the context the retain stored, and the allocation the next
`cuMemAlloc` answers, since the typestate counts them.

The theorem states the platform it needs: the driver loaded (`has cuda`),
nothing captured, no worker and no copy in flight, the default stream's
order, the context not yet retained and no allocation made.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace DriverChecks

variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

/-- Where the context and the allocation's pointer are kept. -/
def SLOT_CTX : Nat := 0x40
def SLOT_BUF : Nat := 0x48

/-- How many bytes go to the device and back. -/
def N : Nat := 4096

def roundTrip : Prog V L Unit := do
  let base ← basePtr
  let src ← dataPtr
  let out ← outPtr
  let z32 ← iconst32 0
  let _ ← ext (.cuda .init) %[z32]
  let _ ← ext (.cuda .primaryCtxRetain) %[← iaddImm base SLOT_CTX, z32]
  let _ ← ext (.cuda .ctxSetCurrent) %[← load64 (← iaddImm base SLOT_CTX)]
  let _ ← ext (.cuda .memAlloc) %[← iaddImm base SLOT_BUF, ← iconst64 N]
  let _ ← ext (.cuda .memcpyHtoD) %[← load64 (← iaddImm base SLOT_BUF), src, ← iconst64 N]
  let _ ← ext (.cuda .memcpyDtoH) %[out, ← load64 (← iaddImm base SLOT_BUF), ← iconst64 N]
  let _ ← ext (.cuda .memFree) %[← load64 (← iaddImm base SLOT_BUF)]

abbrev Start : World → Prop := Contracts.TState.holds
  [.part (.has .cuda) true, .part .capturing false, .part .frozen false, .part .idle true, .devSeq,
   .retained 0, .nbufs 0, .room .arena 4096, .room .data N, .room .out N]

ship_entry roundTrip : roundTrip under Start

#print axioms roundTrip_no_misuse

/-- A kernel's module: one entry point taking one pointer. -/
def KPTX : String :=
  ".version 7.0\n.target sm_50\n.address_size 64\n.visible .entry k(\n.param .u64 p0\n)\n{\nret;\n}\n"

/-- The signature the typestates below state of `KPTX`: its entry `k` takes
    one eight-byte parameter. -/
theorem kptx_sig : ptxParamSizes KPTX "k" = some [8] := by decide

def SLOT_MOD : Nat := 0x50
def SLOT_FN : Nat := 0x58
def SLOT_ARG : Nat := 0x60
def SLOT_PARAMS : Nat := 0x68
def NAME_OFF : Nat := 0x80
def PTX_OFF : Nat := 0x100

/-- Bring the driver up, load the module, look the entry point up, make an
    allocation, and launch the kernel on it on the default stream. -/
def launchOnce : Prog V L Unit := do
  let base ← basePtr
  let z32 ← iconst32 0
  let one ← iconst32 1
  let _ ← ext (.cuda .init) %[z32]
  let _ ← ext (.cuda .primaryCtxRetain) %[← iaddImm base SLOT_CTX, z32]
  let _ ← ext (.cuda .ctxSetCurrent) %[← load64 (← iaddImm base SLOT_CTX)]
  let _ ← ext (.cuda .moduleLoadData) %[← iaddImm base SLOT_MOD, ← iaddImm base PTX_OFF]
  let _ ← ext (.cuda .moduleGetFunction) %[← iaddImm base SLOT_FN, ← load64 (← iaddImm base SLOT_MOD),
    ← iaddImm base NAME_OFF]
  let _ ← ext (.cuda .memAlloc) %[← iaddImm base SLOT_BUF, ← iconst64 N]
  storeI64 (← load64 (← iaddImm base SLOT_BUF)) (← iaddImm base SLOT_ARG)
  storeI64 (← iaddImm base SLOT_ARG) (← iaddImm base SLOT_PARAMS)
  let _ ← ext (.cuda .launchKernel) %[← load64 (← iaddImm base SLOT_FN), one, one, one, ← iconst32 32, one, one,
    z32, ← iconst64 0, ← iaddImm base SLOT_PARAMS, ← iconst64 0]

abbrev LaunchStart : World → Prop := Contracts.TState.holds
  [.part (.has .cuda) true, .part .capturing false, .part .frozen false, .part .idle true, .devSeq, .oracles,
   .retained 0, .nbufs 0, .nmods 0, .nfuncs 0, .room .arena 4096,
   .text (regionBase .arena + UInt64.ofNat NAME_OFF) (some "k") 2 [],
   .text (regionBase .arena + UInt64.ofNat PTX_OFF) none 128 [("k", [8])]]

ship_entry launchOnce : launchOnce under LaunchStart

#print axioms launchOnce_no_misuse

def SLOT_STREAM : Nat := 0x70
def SLOT_EVENT : Nat := 0x78

/-- Make a stream and an event, record the event on the stream, make the
    default stream wait for it, wait for the stream from the host, and destroy
    both. -/
def streamEvent : Prog V L Unit := do
  let base ← basePtr
  let z32 ← iconst32 0
  let _ ← ext (.cuda .init) %[z32]
  let _ ← ext (.cuda .primaryCtxRetain) %[← iaddImm base SLOT_CTX, z32]
  let _ ← ext (.cuda .ctxSetCurrent) %[← load64 (← iaddImm base SLOT_CTX)]
  let _ ← ext (.cuda .streamCreate) %[← iaddImm base SLOT_STREAM, ← iconst32 1]
  let _ ← ext (.cuda .eventCreate) %[← iaddImm base SLOT_EVENT, z32]
  let _ ← ext (.cuda .eventRecord) %[← load64 (← iaddImm base SLOT_EVENT), ← load64 (← iaddImm base SLOT_STREAM)]
  let _ ← ext (.cuda .streamWaitEvent) %[← iconst64 0, ← load64 (← iaddImm base SLOT_EVENT), z32]
  let _ ← ext (.cuda .streamSynchronize) %[← load64 (← iaddImm base SLOT_STREAM)]
  let _ ← ext (.cuda .eventDestroy) %[← load64 (← iaddImm base SLOT_EVENT)]
  let _ ← ext (.cuda .streamDestroy) %[← load64 (← iaddImm base SLOT_STREAM)]

abbrev StreamStart : World → Prop := Contracts.TState.holds
  [.part (.has .cuda) true, .part .capturing false, .part .frozen false, .part .idle true, .devSeq,
   .retained 0, .streams [], .events 0, .room .arena 4096]

ship_entry streamEvent : streamEvent under StreamStart

#print axioms streamEvent_no_misuse

def SLOT_BUF2 : Nat := 0x88
def SLOT_FREE : Nat := 0x90
def SLOT_TOTAL : Nat := 0x98

/-- Ask how much device memory is free, copy one allocation to another on
    the device, and free both. -/
def deviceCopy : Prog V L Unit := do
  let base ← basePtr
  let z32 ← iconst32 0
  let _ ← ext (.cuda .init) %[z32]
  let _ ← ext (.cuda .primaryCtxRetain) %[← iaddImm base SLOT_CTX, z32]
  let _ ← ext (.cuda .ctxSetCurrent) %[← load64 (← iaddImm base SLOT_CTX)]
  let _ ← ext (.cuda .memGetInfo) %[← iaddImm base SLOT_FREE, ← iaddImm base SLOT_TOTAL]
  let _ ← ext (.cuda .memAlloc) %[← iaddImm base SLOT_BUF, ← iconst64 N]
  let _ ← ext (.cuda .memAlloc) %[← iaddImm base SLOT_BUF2, ← iconst64 N]
  let _ ← ext (.cuda .memcpyDtoD) %[← load64 (← iaddImm base SLOT_BUF2), ← load64 (← iaddImm base SLOT_BUF),
    ← iconst64 N]
  let _ ← ext (.cuda .memFree) %[← load64 (← iaddImm base SLOT_BUF)]
  let _ ← ext (.cuda .memFree) %[← load64 (← iaddImm base SLOT_BUF2)]

ship_entry deviceCopy : deviceCopy under Start

#print axioms deviceCopy_no_misuse

/-- Launch the kernel on a stream of its own, record an event after it, and
    make the default stream wait for the event before freeing the allocation
    the kernel was handed. -/
def streamLaunch : Prog V L Unit := do
  let base ← basePtr
  let z32 ← iconst32 0
  let one ← iconst32 1
  let _ ← ext (.cuda .init) %[z32]
  let _ ← ext (.cuda .primaryCtxRetain) %[← iaddImm base SLOT_CTX, z32]
  let _ ← ext (.cuda .ctxSetCurrent) %[← load64 (← iaddImm base SLOT_CTX)]
  let _ ← ext (.cuda .moduleLoadData) %[← iaddImm base SLOT_MOD, ← iaddImm base PTX_OFF]
  let _ ← ext (.cuda .moduleGetFunction) %[← iaddImm base SLOT_FN, ← load64 (← iaddImm base SLOT_MOD),
    ← iaddImm base NAME_OFF]
  let _ ← ext (.cuda .memAlloc) %[← iaddImm base SLOT_BUF, ← iconst64 N]
  let _ ← ext (.cuda .streamCreate) %[← iaddImm base SLOT_STREAM, one]
  let _ ← ext (.cuda .eventCreate) %[← iaddImm base SLOT_EVENT, z32]
  storeI64 (← load64 (← iaddImm base SLOT_BUF)) (← iaddImm base SLOT_ARG)
  storeI64 (← iaddImm base SLOT_ARG) (← iaddImm base SLOT_PARAMS)
  let _ ← ext (.cuda .launchKernel) %[← load64 (← iaddImm base SLOT_FN), one, one, one, ← iconst32 32, one, one,
    z32, ← load64 (← iaddImm base SLOT_STREAM), ← iaddImm base SLOT_PARAMS, ← iconst64 0]
  let _ ← ext (.cuda .eventRecord) %[← load64 (← iaddImm base SLOT_EVENT), ← load64 (← iaddImm base SLOT_STREAM)]
  let _ ← ext (.cuda .streamWaitEvent) %[← iconst64 0, ← load64 (← iaddImm base SLOT_EVENT), z32]
  let _ ← ext (.cuda .streamSynchronize) %[← iconst64 0]

abbrev StreamLaunchStart : World → Prop := Contracts.TState.holds
  [.part (.has .cuda) true, .part .capturing false, .part .frozen false, .part .idle true, .devSeq, .oracles,
   .retained 0, .nbufs 0, .nmods 0, .nfuncs 0, .streams [], .events 0, .room .arena 4096,
   .text (regionBase .arena + UInt64.ofNat NAME_OFF) (some "k") 2 [],
   .text (regionBase .arena + UInt64.ofNat PTX_OFF) none 128 [("k", [8])]]

ship_entry streamLaunch : streamLaunch under StreamLaunchStart

#print axioms streamLaunch_no_misuse

def SLOT_A : Nat := 0xa0
def SLOT_B : Nat := 0xa8
def SLOT_C : Nat := 0xb0
def SLOT_BLAS : Nat := 0xb8
def ALPHA_OFF : Nat := 0xc0
def BETA_OFF : Nat := 0xc4

/-- `C = A B` for 4×4 single-precision matrices through cuBLAS on the default
    stream, then wait for it and free the three allocations. -/
def blasProduct : Prog V L Unit := do
  let base ← basePtr
  let z32 ← iconst32 0
  let four ← iconst32 4
  let bytes ← iconst64 64
  let _ ← ext (.cuda .init) %[z32]
  let _ ← ext (.cuda .primaryCtxRetain) %[← iaddImm base SLOT_CTX, z32]
  let _ ← ext (.cuda .ctxSetCurrent) %[← load64 (← iaddImm base SLOT_CTX)]
  let _ ← ext (.cuda .memAlloc) %[← iaddImm base SLOT_A, bytes]
  let _ ← ext (.cuda .memAlloc) %[← iaddImm base SLOT_B, bytes]
  let _ ← ext (.cuda .memAlloc) %[← iaddImm base SLOT_C, bytes]
  let _ ← ext (.cublas .create) %[← iaddImm base SLOT_BLAS]
  let _ ← ext (.cublas .sgemm) %[← load64 (← iaddImm base SLOT_BLAS), z32, z32, four, four, four,
    ← iaddImm base ALPHA_OFF, ← load64 (← iaddImm base SLOT_A), four, ← load64 (← iaddImm base SLOT_B), four,
    ← iaddImm base BETA_OFF, ← load64 (← iaddImm base SLOT_C), four]
  let _ ← ext (.cuda .ctxSynchronize) %[]
  let _ ← ext (.cublas .destroy) %[← load64 (← iaddImm base SLOT_BLAS)]
  let _ ← ext (.cuda .memFree) %[← load64 (← iaddImm base SLOT_A)]
  let _ ← ext (.cuda .memFree) %[← load64 (← iaddImm base SLOT_B)]
  let _ ← ext (.cuda .memFree) %[← load64 (← iaddImm base SLOT_C)]

abbrev BlasStart : World → Prop := Contracts.TState.holds
  [.part (.has .cuda) true, .part (.has .cublas) true, .part .capturing false, .part .frozen false,
   .part .idle true, .devSeq, .oracles, .retained 0, .nbufs 0, .nblas 0, .room .arena 4096]

ship_entry blasProduct : blasProduct under BlasStart

#print axioms blasProduct_no_misuse

def SLOT_GRAPH : Nat := 0xc8
def SLOT_EXEC : Nat := 0xd0

/-- Capture the kernel's launch on a stream of its own as a graph,
    instantiate it, upload it and replay it twice, then wait for the stream
    and destroy both graphs. -/
def graphReplay : Prog V L Unit := do
  let base ← basePtr
  let z32 ← iconst32 0
  let one ← iconst32 1
  let _ ← ext (.cuda .init) %[z32]
  let _ ← ext (.cuda .primaryCtxRetain) %[← iaddImm base SLOT_CTX, z32]
  let _ ← ext (.cuda .ctxSetCurrent) %[← load64 (← iaddImm base SLOT_CTX)]
  let _ ← ext (.cuda .moduleLoadData) %[← iaddImm base SLOT_MOD, ← iaddImm base PTX_OFF]
  let _ ← ext (.cuda .moduleGetFunction) %[← iaddImm base SLOT_FN, ← load64 (← iaddImm base SLOT_MOD),
    ← iaddImm base NAME_OFF]
  let _ ← ext (.cuda .memAlloc) %[← iaddImm base SLOT_BUF, ← iconst64 N]
  let _ ← ext (.cuda .streamCreate) %[← iaddImm base SLOT_STREAM, one]
  storeI64 (← load64 (← iaddImm base SLOT_BUF)) (← iaddImm base SLOT_ARG)
  storeI64 (← iaddImm base SLOT_ARG) (← iaddImm base SLOT_PARAMS)
  let _ ← ext (.cuda .beginCapture) %[← load64 (← iaddImm base SLOT_STREAM), ← iconst32 2]
  let _ ← ext (.cuda .launchKernel) %[← load64 (← iaddImm base SLOT_FN), one, one, one, ← iconst32 32, one, one,
    z32, ← load64 (← iaddImm base SLOT_STREAM), ← iaddImm base SLOT_PARAMS, ← iconst64 0]
  let _ ← ext (.cuda .endCapture) %[← load64 (← iaddImm base SLOT_STREAM), ← iaddImm base SLOT_GRAPH]
  let _ ← ext (.cuda .graphInstantiate) %[← iaddImm base SLOT_EXEC, ← load64 (← iaddImm base SLOT_GRAPH),
    ← iconst64 0]
  let _ ← ext (.cuda .graphUpload) %[← load64 (← iaddImm base SLOT_EXEC), ← load64 (← iaddImm base SLOT_STREAM)]
  let _ ← ext (.cuda .graphLaunch) %[← load64 (← iaddImm base SLOT_EXEC), ← load64 (← iaddImm base SLOT_STREAM)]
  let _ ← ext (.cuda .graphLaunch) %[← load64 (← iaddImm base SLOT_EXEC), ← load64 (← iaddImm base SLOT_STREAM)]
  let _ ← ext (.cuda .streamSynchronize) %[← load64 (← iaddImm base SLOT_STREAM)]
  let _ ← ext (.cuda .graphExecDestroy) %[← load64 (← iaddImm base SLOT_EXEC)]
  let _ ← ext (.cuda .graphDestroy) %[← load64 (← iaddImm base SLOT_GRAPH)]

abbrev GraphStart : World → Prop := Contracts.TState.holds
  [.part (.has .cuda) true, .part .capturing false, .part .frozen false, .part .idle true, .devSeq, .oracles,
   .retained 0, .nbufs 0, .nmods 0, .nfuncs 0, .streams [], .events 0, .ngraphs 0, .nexecs 0,
   .room .arena 4096,
   .text (regionBase .arena + UInt64.ofNat NAME_OFF) (some "k") 2 [],
   .text (regionBase .arena + UInt64.ofNat PTX_OFF) none 128 [("k", [8])]]

ship_entry graphReplay : graphReplay under GraphStart

#print axioms graphReplay_no_misuse

def SLOT_STREAM2 : Nat := 0x200
def SLOT_EV0 : Nat := 0x208
def SLOT_EV1 : Nat := 0x210

set_option maxRecDepth 2000 in
/-- **Two streams captured as one graph.** The capture begins on one stream;
    an event recorded there and waited on by a second stream brings that
    stream into the capture. The kernel's launch goes on the first stream and
    a cuBLAS product on the second, each ordered against what the capture has
    recorded by the allocations it touches; an event recorded on the second
    and waited on by the first joins it back before the capture ends. The
    graph, holding both, is then replayed twice. -/
def forkJoin : Prog V L Unit := do
  let base ← basePtr
  let z32 ← iconst32 0
  let one ← iconst32 1
  let four ← iconst32 4
  let bytes ← iconst64 64
  let _ ← ext (.cuda .init) %[z32]
  let _ ← ext (.cuda .primaryCtxRetain) %[← iaddImm base SLOT_CTX, z32]
  let _ ← ext (.cuda .ctxSetCurrent) %[← load64 (← iaddImm base SLOT_CTX)]
  let _ ← ext (.cuda .moduleLoadData) %[← iaddImm base SLOT_MOD, ← iaddImm base PTX_OFF]
  let _ ← ext (.cuda .moduleGetFunction) %[← iaddImm base SLOT_FN, ← load64 (← iaddImm base SLOT_MOD),
    ← iaddImm base NAME_OFF]
  let _ ← ext (.cuda .memAlloc) %[← iaddImm base SLOT_BUF, ← iconst64 N]
  let _ ← ext (.cuda .memAlloc) %[← iaddImm base SLOT_A, bytes]
  let _ ← ext (.cuda .memAlloc) %[← iaddImm base SLOT_B, bytes]
  let _ ← ext (.cuda .memAlloc) %[← iaddImm base SLOT_C, bytes]
  let _ ← ext (.cuda .streamCreate) %[← iaddImm base SLOT_STREAM, one]
  let _ ← ext (.cuda .streamCreate) %[← iaddImm base SLOT_STREAM2, one]
  let _ ← ext (.cuda .eventCreate) %[← iaddImm base SLOT_EV0, z32]
  let _ ← ext (.cuda .eventCreate) %[← iaddImm base SLOT_EV1, z32]
  let _ ← ext (.cublas .create) %[← iaddImm base SLOT_BLAS]
  let _ ← ext (.cublas .setStream) %[← load64 (← iaddImm base SLOT_BLAS), ← load64 (← iaddImm base SLOT_STREAM2)]
  storeI64 (← load64 (← iaddImm base SLOT_BUF)) (← iaddImm base SLOT_ARG)
  storeI64 (← iaddImm base SLOT_ARG) (← iaddImm base SLOT_PARAMS)
  let _ ← ext (.cuda .beginCapture) %[← load64 (← iaddImm base SLOT_STREAM), ← iconst32 2]
  let _ ← ext (.cuda .eventRecord) %[← load64 (← iaddImm base SLOT_EV0), ← load64 (← iaddImm base SLOT_STREAM)]
  let _ ← ext (.cuda .streamWaitEvent) %[← load64 (← iaddImm base SLOT_STREAM2),
    ← load64 (← iaddImm base SLOT_EV0), z32]
  let _ ← ext (.cuda .launchKernel) %[← load64 (← iaddImm base SLOT_FN), one, one, one, ← iconst32 32, one, one,
    z32, ← load64 (← iaddImm base SLOT_STREAM), ← iaddImm base SLOT_PARAMS, ← iconst64 0]
  let _ ← ext (.cublas .sgemm) %[← load64 (← iaddImm base SLOT_BLAS), z32, z32, four, four, four,
    ← iaddImm base ALPHA_OFF, ← load64 (← iaddImm base SLOT_A), four, ← load64 (← iaddImm base SLOT_B), four,
    ← iaddImm base BETA_OFF, ← load64 (← iaddImm base SLOT_C), four]
  let _ ← ext (.cuda .eventRecord) %[← load64 (← iaddImm base SLOT_EV1), ← load64 (← iaddImm base SLOT_STREAM2)]
  let _ ← ext (.cuda .streamWaitEvent) %[← load64 (← iaddImm base SLOT_STREAM),
    ← load64 (← iaddImm base SLOT_EV1), z32]
  let _ ← ext (.cuda .endCapture) %[← load64 (← iaddImm base SLOT_STREAM), ← iaddImm base SLOT_GRAPH]
  let _ ← ext (.cuda .graphInstantiate) %[← iaddImm base SLOT_EXEC, ← load64 (← iaddImm base SLOT_GRAPH),
    ← iconst64 0]
  let _ ← ext (.cuda .graphUpload) %[← load64 (← iaddImm base SLOT_EXEC), ← load64 (← iaddImm base SLOT_STREAM)]
  let _ ← ext (.cuda .graphLaunch) %[← load64 (← iaddImm base SLOT_EXEC), ← load64 (← iaddImm base SLOT_STREAM)]
  let _ ← ext (.cuda .graphLaunch) %[← load64 (← iaddImm base SLOT_EXEC), ← load64 (← iaddImm base SLOT_STREAM)]
  let _ ← ext (.cuda .streamSynchronize) %[← load64 (← iaddImm base SLOT_STREAM)]
  let _ ← ext (.cuda .graphExecDestroy) %[← load64 (← iaddImm base SLOT_EXEC)]
  let _ ← ext (.cuda .graphDestroy) %[← load64 (← iaddImm base SLOT_GRAPH)]

abbrev ForkStart : World → Prop := Contracts.TState.holds
  [.part (.has .cuda) true, .part (.has .cublas) true, .part .capturing false, .part .frozen false,
   .part .idle true, .devSeq, .oracles,
   .retained 0, .nbufs 0, .nmods 0, .nfuncs 0, .nblas 0, .streams [], .events 0, .ngraphs 0, .nexecs 0,
   .room .arena 4096,
   .text (regionBase .arena + UInt64.ofNat NAME_OFF) (some "k") 2 [],
   .text (regionBase .arena + UInt64.ofNat PTX_OFF) none 128 [("k", [8])]]

ship_entry forkJoin : forkJoin under ForkStart

#print axioms forkJoin_no_misuse

def SLOT_N : Nat := 0xd8

/-- Copy the caller's `n` floats into an allocation of `n` floats, `n` a
    number the program keeps in memory: only when the caller handed over
    that many bytes. -/
def sizedUpload : Prog V L Unit := do
  let base ← basePtr
  let n ← load64 (← iaddImm base SLOT_N)
  let bytes ← imul n (← iconst64 4)
  Prog.when .ule bytes (← dataLen) do
    let _ ← ext (.cuda .memcpyHtoD) %[← load64 (← iaddImm base SLOT_BUF), ← dataPtr, bytes]

abbrev SizedStart (dl n : UInt64) : World → Prop := Contracts.TState.holds
  [.part (.has .cuda) true, .part .capturing false, .part .frozen false, .part .idle true, .part .cuda true,
   .devSeq, .oracles, .room .arena 4096, .roomArg .data dl,
   .cell (regionBase .arena + UInt64.ofNat SLOT_BUF) (devPtr 0),
   .cell (regionBase .arena + UInt64.ofNat SLOT_N) n, .alloc 0 ((n * 4).toNat)]

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem sizedUpload_no_misuse (cfg : Cfg) (dataLen outLen n : UInt64) (w : World)
    (h : SizedStart dataLen n w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (sizedUpload : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (sizedUpload : Prog Slot Lvl Unit)) ≠ .fault m := by
  check_body
  -- the copy's bytes, at `n`'s value, lie in the allocation of `n` floats
  have hp : devPtrOf 4503599627370496 = some (0, 0) := by decide
  simp [Check.Obl.Holds, Check.vendorVal, Check.cudaVal, Check.argAt, Check.allocFor, Check.allocFacts,
    Check.PExpr.eval, Check.PArg.bits, hp]

#print axioms sizedUpload_no_misuse

/-- Copy as many bytes as the caller's output holds out of an allocation of
    `n` floats: only when they fit in it. The checker keeps the comparison,
    so the copy's obligation is shown where it holds. -/
def sizedDownload : Prog V L Unit := do
  let base ← basePtr
  let n ← load64 (← iaddImm base SLOT_N)
  let bytes ← imul n (← iconst64 4)
  let ol ← outLen
  Prog.when .ule ol bytes do
    let _ ← ext (.cuda .memcpyDtoH) %[← outPtr, ← load64 (← iaddImm base SLOT_BUF), ol]

abbrev SizedOut (ol n : UInt64) : World → Prop := Contracts.TState.holds
  [.part (.has .cuda) true, .part .capturing false, .part .frozen false, .part .idle true, .part .cuda true,
   .devSeq, .oracles, .room .arena 4096, .roomArg .out ol,
   .cell (regionBase .arena + UInt64.ofNat SLOT_BUF) (devPtr 0),
   .cell (regionBase .arena + UInt64.ofNat SLOT_N) n, .alloc 0 ((n * 4).toNat)]

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem sizedDownload_no_misuse (cfg : Cfg) (dataLen outLen n : UInt64) (w : World)
    (h : SizedOut outLen n w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (sizedDownload : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (sizedDownload : Prog Slot Lvl Unit)) ≠ .fault m := by
  check_body
  -- the copy's bytes, no more than the allocation's, lie in it
  have hp : devPtrOf 4503599627370496 = some (0, 0) := by decide
  intro hc
  simp [Check.PCond.Holds, Check.RelHolds, Check.PExpr.eval] at hc
  simp [Check.vendorVal, Check.cudaVal, Check.argAt, Check.allocFor, Check.allocFacts,
    Check.PExpr.eval, Check.PArg.bits, hp, hc]

#print axioms sizedDownload_no_misuse

end DriverChecks
