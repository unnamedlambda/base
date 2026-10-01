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
# The device contracts, checked against the device

`Sem.callFfi` states what the CUDA entry points do: the context, allocation,
the copies, freeing, synchronising, launching, streams, events and cuBLAS. Like every other
contract there it is a transcription of `base/src/ffi/cuda/`, and this is what
checks it. One body exercises each entry point on its success and its refusal
paths, storing every result code and every byte it copies back in the output
buffer; `base/tests/hprog_cuda_corpus.rs` runs the artifact on the GPU and
compares the output with what the interpreter computed here.

The launch runs a real kernel — every thread adds one to its byte — and the
interpreter's kernel oracle is that function, so the check covers the launch's
plumbing (text, bindings, grid) and the ordering of launches and copies, with
the one thing the model does not compute pinned to a known answer.
-/

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.Prog
open HProgVendorRef

namespace HProgCudaCorpus

/-- Every thread adds one to the byte at its index. -/
def addOnePtx : String :=
  ".version 7.0\n.target sm_60\n.address_size 64\n\n" ++
  ".visible .entry main(.param .u64 p0)\n{\n" ++
  "  .reg .u32 %r<2>;\n  .reg .u64 %rd<4>;\n  .reg .u16 %h<3>;\n" ++
  "  ld.param.u64 %rd1, [p0];\n  cvta.to.global.u64 %rd1, %rd1;\n" ++
  "  mov.u32 %r1, %tid.x;\n  cvt.u64.u32 %rd2, %r1;\n  add.u64 %rd3, %rd1, %rd2;\n" ++
  "  ld.global.u8 %h1, [%rd3];\n  add.u16 %h2, %h1, 1;\n  st.global.u8 [%rd3], %h2;\n" ++
  "  ret;\n}\n"

/-- The same kernel with its entry point named `addone`, for the named launch. -/
def addOneNamedPtx : String := addOnePtx.replace ".entry main(" ".entry addone("

/-- What the kernels do, as the interpreter's oracle. -/
def addOne (l : Sem.Launch) (ins : List ByteArray) : List ByteArray :=
  if (l.kernel == addOnePtx && l.entry == "main") || (l.kernel == addOneNamedPtx && l.entry == "addone")
  then ins.map (fun b => ⟨b.data.map (· + 1)⟩) else ins

-- The arena: context slots, the bytes to upload, two binding slots, the kernel.
def SRC : Nat := 0x100
def BIND : Nat := 0x140
def BIND_FREED : Nat := 0x148
def KERNEL : Nat := 0x180
def MEM : Nat := 0x800

def srcBytes : List UInt8 := (List.range 64).map (fun i => (i * 3 + 1).toUInt8)

-- The matrices, from 0x300: a 4 × 3 `A` with `x` and `y` for `sgemv`, and two
-- batches of 2 × 3 by 3 × 2 for `sgemm`.
def GA : Nat := 0x300
def GX : Nat := 0x330
def GY : Nat := 0x340
def SA : Nat := 0x380
def SB : Nat := 0x3b0

def gA : List Float := [1, 2, 0, 3,  2, 1, 1, 0,  0, 3, 2, 1]
def gX : List Float := [1, 2, 3]
def gY : List Float := [5, 0, 1, 2]
def sA : List Float := [1, 2, 0, 1, 3, 1,  2, 0, 1, 1, 0, 3]
def sB : List Float := [1, 0, 2, 2, 1, 1,  0, 1, 1, 3, 2, 0]

-- The named kernel and the bf16 operands, from 0x500.
def KERNEL_NAMED : Nat := 0x500
def ENTRY_NAME : Nat := 0x4f0
def HA : Nat := 0x680
def HB : Nat := 0x6a0
def hA : List Float := [1, 2, 0, 1, 3, 1]
def hB : List Float := [2, 1, 1, 0, 1, 3]

def image : List UInt8 :=
  let pad (xs : List UInt8) (n : Nat) := xs ++ List.replicate (n - xs.length) 0
  let withKernel := pad (pad (List.replicate SRC 0 ++ srcBytes) KERNEL ++ (stringToBytes addOnePtx ++ [0])) GA
  let withGemv := pad (pad (pad withKernel GA ++ f32Bytes gA) GX ++ f32Bytes gX) GY ++ f32Bytes gY
  let withSgemm := pad (pad withGemv SA ++ f32Bytes sA) SB ++ f32Bytes sB
  let withNamed := pad (pad withSgemm ENTRY_NAME ++ (stringToBytes "addone" ++ [0])) KERNEL_NAMED
    ++ (stringToBytes addOneNamedPtx ++ [0])
  pad (pad (pad withNamed HA ++ bf16Bytes hA) HB ++ bf16Bytes hB) MEM

/-- Where each result lands in the output buffer. Codes are `i32`s from 0;
    downloaded bytes from 128. -/
def OUT : Nat := 912

set_option maxRecDepth 8000 in
def body : Prog V L Unit := do
  let ptr ← basePtr
  let out ← outPtr
  let code (k : Nat) (r : V .i32) : Prog V L Unit := do
    storeI32 r (← iadd out (← iconst64 (4 * k)))
  cudaInit ptr
  let c ← cudaCtxPtr ptr
  let b0 ← cudaCreateBuffer ptr (← iconst64 64)
  code 0 b0
  code 1 (← cudaCreateBuffer ptr (← iconst64 0))
  let b2 ← cudaCreateBuffer ptr (← iconst64 16)
  code 2 b2
  code 3 (← cudaUpload ptr b0 (← iconst64 SRC) (← iconst64 64))
  code 4 (← cudaUpload ptr (← iconst32 99) (← iconst64 SRC) (← iconst64 64))
  code 5 (← cudaUploadOffset ptr b2 (← iconst64 8) (← iconst64 SRC) (← iconst64 8))
  code 6 (← cudaUploadOffset ptr b2 (← iconst64 12) (← iconst64 SRC) (← iconst64 8))
  storeI32 b0 (← iadd ptr (← iconst64 BIND))
  let one ← iconst32 1
  code 7 (← cudaLaunch ptr (← iconst64 KERNEL) one (← iconst64 BIND)
            one one one (← iconst32 64) one one)
  -- Downloads go straight to the caller's output buffer.
  let dst0 ← iadd out (← iconst64 128)
  code 8 (← ffi .cudaDownload %[c, b0, dst0, ← iconst64 64])
  let dst2 ← iadd out (← iconst64 192)
  code 9 (← ffi .cudaDownloadOffset %[c, b2, ← iconst64 0, dst2, ← iconst64 16])
  code 10 (← ffi .cudaDownloadOffset %[c, b2, ← iconst64 12, dst2, ← iconst64 8])
  code 11 (← cudaFreeBuffer ptr b2)
  code 12 (← cudaFreeBuffer ptr b2)
  code 13 (← ffi .cudaDownload %[c, b2, dst2, ← iconst64 16])
  storeI32 b2 (← iadd ptr (← iconst64 BIND_FREED))
  code 14 (← cudaLaunch ptr (← iconst64 KERNEL) one (← iconst64 BIND_FREED)
             one one one (← iconst32 64) one one)
  code 15 (← cudaSync ptr)
  -- sgemv: y ← A·x + y
  let up (off n : Nat) : Prog V L (V .i32) := do
    let b ← cudaCreateBuffer ptr (← iconst64 n)
    let _ ← cudaUpload ptr b (← iconst64 off) (← iconst64 n)
    pure b
  let bA ← up GA 48
  let bX ← up GX 12
  let bY ← up GY 16
  let f1 ← iconst32 0x3f800000
  let z ← iconst32 0
  code 16 (← cublasSgemv ptr z (← iconst32 4) (← iconst32 3) f1 bA bX f1 bY)
  code 17 (← cublasSgemv ptr z (← iconst32 4) (← iconst32 3) f1 (← iconst32 99) bX f1 bY)
  code 18 (← ffi .cudaDownload %[c, bY, ← iadd out (← iconst64 256), ← iconst64 16])
  -- sgemm, two batches of C (2 × 2) ← A (2 × 3) · B (3 × 2)
  let bSA ← up SA 48
  let bSB ← up SB 48
  let bSC ← cudaCreateBuffer ptr (← iconst64 32)
  code 19 (← cublasSgemmStridedBatched ptr z z (← iconst32 2) (← iconst32 2) (← iconst32 3) f1 bSA
             (← iconst64 6) bSB (← iconst64 6) z bSC (← iconst64 4) (← iconst32 2))
  code 20 (← cublasSgemmStridedBatched ptr z z z (← iconst32 2) (← iconst32 3) f1 bSA
             (← iconst64 6) bSB (← iconst64 6) z bSC (← iconst64 4) (← iconst32 2))
  code 21 (← ffi .cudaDownload %[c, bSC, ← iadd out (← iconst64 272), ← iconst64 32])
  -- the named launch, on a fresh buffer holding the first 64 source bytes
  let bN ← up SRC 64
  storeI32 bN (← iadd ptr (← iconst64 BIND))
  code 22 (← cudaLaunchNamed ptr (← iconst64 KERNEL_NAMED) (← iconst64 ENTRY_NAME) one
             (← iconst64 BIND) one one one (← iconst32 64) one one)
  -- C (2 × 2, f32) ← A (2 × 3, bf16) · B (3 × 2, bf16)
  let bHA ← up HA 12
  let bHB ← up HB 12
  let bHC ← cudaCreateBuffer ptr (← iconst64 16)
  code 23 (← ffi .cublasGemmExBf16 %[c, z, z, ← iconst32 2, ← iconst32 2, ← iconst32 3, f1,
             bHA, bHB, z, bHC, ← iconst64 0, ← iconst64 0, ← iconst64 0, z, z, z])
  code 24 (← ffi .cudaDownload %[c, bHC, ← iadd out (← iconst64 304), ← iconst64 16])
  code 25 (← ffi .cudaDownload %[c, bN, ← iadd out (← iconst64 320), ← iconst64 64])
  -- pinned host memory: fill it through its pointer, upload from it, launch,
  -- download back into it, and copy it out
  let pid ← ffi .cudaPinnedAlloc %[c, ← iconst64 64]
  code 26 pid
  let pp ← ffi .cudaPinnedPtr %[c, pid]
  for k in List.range 8 do
    storeI64 (← load64 (← iadd ptr (← iconst64 (SRC + 8 * k)))) (← iadd pp (← iconst64 (8 * k)))
  let bP ← cudaCreateBuffer ptr (← iconst64 64)
  code 27 (← ffi .cudaUpload %[c, bP, pp, ← iconst64 64])
  storeI32 bP (← iadd ptr (← iconst64 BIND))
  code 28 (← cudaLaunch ptr (← iconst64 KERNEL) one (← iconst64 BIND)
             one one one (← iconst32 64) one one)
  code 29 (← ffi .cudaDownload %[c, bP, pp, ← iconst64 64])
  for k in List.range 8 do
    storeI64 (← load64 (← iadd pp (← iconst64 (8 * k)))) (← iadd out (← iconst64 (384 + 8 * k)))
  storeI64 (← ffi .cudaPinnedPtrAt %[c, pid, ← iconst64 60, ← iconst64 8]) (← iadd out (← iconst64 448))
  code 30 (← ffi .cudaPinnedFree %[c, pid])
  code 31 (← ffi .cudaPinnedFree %[c, pid])
  storeI64 (← ffi .cudaPinnedPtr %[c, pid]) (← iadd out (← iconst64 456))
  -- how much device memory there is cannot be compared, only that there is some
  let total ← ffi .cudaMemInfoTotal %[c]
  let some' ← icmp .sgt total (← iconst64 0)
  storeI64 (← select some' (← iconst64 1) (← iconst64 0)) (← iadd out (← iconst64 464))
  -- On a created stream: sgemv, a named launch, the pointer-array batch, and
  -- the time between two events once the host has waited for them. Codes from
  -- byte 480, downloads from 528.
  let code2 (k : Nat) (r : V .i32) : Prog V L Unit := do
    storeI32 r (← iadd out (← iconst64 (480 + 4 * k)))
  let s ← ffi .cudaStreamCreate %[c]
  let e1 ← ffi .cudaEventCreate %[c]
  let e2 ← ffi .cudaEventCreate %[c]
  let e3 ← ffi .cudaEventCreate %[c]
  code2 0 (← ffi .cudaEventRecord %[c, e1, s])
  let bY2 ← up GY 16
  code2 1 (← ffi .cublasSgemvOnStream %[c, z, ← iconst32 4, ← iconst32 3, f1, bA, bX, f1, bY2, s])
  let bN2 ← up SRC 64
  storeI32 bN2 (← iadd ptr (← iconst64 BIND))
  let at_ (o : Nat) : Prog V L (V .i64) := do iadd ptr (← iconst64 o)
  code2 2 (← ffi .cudaLaunchNamedOnStream %[c, ← at_ KERNEL_NAMED, ← at_ ENTRY_NAME, one,
             ← at_ BIND, one, one, one, ← iconst32 64, one, one, s])
  let arr (n : Nat) : Prog V L (V .i32) := do cudaCreateBuffer ptr (← iconst64 (8 * n))
  let aArr ← arr 2
  let bArr ← arr 2
  let cArr ← arr 2
  let bC1 ← cudaCreateBuffer ptr (← iconst64 16)
  let bC2 ← cudaCreateBuffer ptr (← iconst64 16)
  let _ ← ffi .cublasPtrArray %[c, aArr, z, bSA, ← iconst64 0]
  let _ ← ffi .cublasPtrArray %[c, aArr, one, bSA, ← iconst64 6]
  let _ ← ffi .cublasPtrArray %[c, bArr, z, bSB, ← iconst64 0]
  let _ ← ffi .cublasPtrArray %[c, bArr, one, bSB, ← iconst64 6]
  let _ ← ffi .cublasPtrArray %[c, cArr, z, bC1, ← iconst64 0]
  code2 3 (← ffi .cublasPtrArray %[c, cArr, one, bC2, ← iconst64 0])
  code2 4 (← ffi .cublasPtrArray %[c, cArr, ← iconst32 2, bC2, ← iconst64 0])
  code2 5 (← ffi .cublasSgemmBatchedOnStream %[c, z, z, ← iconst32 2, ← iconst32 2, ← iconst32 3, f1,
             aArr, bArr, z, cArr, ← iconst32 2, s])
  let _ ← ffi .cudaEventRecord %[c, e2, s]
  code2 6 (← ffi .cudaStreamSync %[c, s])
  let el ← ffi .cudaEventElapsedMsBits %[c, e1, e2]
  let answered ← icmp .ne el (← iconst32 (-1))
  storeI32 (← select answered one z) (← iadd out (← iconst64 (480 + 4 * 7)))
  code2 8 (← ffi .cudaEventElapsedMsBits %[c, e1, e3])
  let _ ← ffi .cudaDownload %[c, bY2, ← iadd out (← iconst64 528), ← iconst64 16]
  let _ ← ffi .cudaDownload %[c, bN2, ← iadd out (← iconst64 544), ← iconst64 64]
  let _ ← ffi .cudaDownload %[c, bC1, ← iadd out (← iconst64 608), ← iconst64 16]
  let _ ← ffi .cudaDownload %[c, bC2, ← iadd out (← iconst64 624), ← iconst64 16]
  -- the strided-batched bf16 gemmEx: both batches read the same A and B
  let bHC2 ← cudaCreateBuffer ptr (← iconst64 32)
  code2 9 (← ffi .cublasGemmStridedBatchedExBf16 %[c, z, z, ← iconst32 2, ← iconst32 2, ← iconst32 3,
             f1, bHA, ← iconst64 0, bHB, ← iconst64 0, z, bHC2, ← iconst64 4, ← iconst32 2,
             ← iconst64 0, ← iconst64 0, ← iconst64 0, z, z, z])
  code2 10 (← ffi .cublasGemmStridedBatchedExBf16 %[c, z, z, ← iconst32 2, ← iconst32 2, ← iconst32 3,
             f1, bHA, ← iconst64 0, bHB, ← iconst64 0, z, bHC2, ← iconst64 4, z,
             ← iconst64 0, ← iconst64 0, ← iconst64 0, z, z, z])
  let _ ← ffi .cudaDownload %[c, bHC2, ← iadd out (← iconst64 640), ← iconst64 32]
  -- Asynchronous copies on the stream: up from pinned memory, a patch from
  -- pageable memory, the named launch, and down into pinned and into pageable
  -- memory. The pinned bytes are the host's again only after the stream sync.
  -- Codes from byte 800, downloads from 672.
  let code3 (k : Nat) (r : V .i32) : Prog V L Unit := do
    storeI32 r (← iadd out (← iconst64 (800 + 4 * k)))
  let pid2 ← ffi .cudaPinnedAlloc %[c, ← iconst64 64]
  let pp2 ← ffi .cudaPinnedPtr %[c, pid2]
  let pid3 ← ffi .cudaPinnedAlloc %[c, ← iconst64 64]
  let pp3 ← ffi .cudaPinnedPtr %[c, pid3]
  for k in List.range 8 do
    storeI64 (← load64 (← iadd ptr (← iconst64 (SRC + 8 * k)))) (← iadd pp2 (← iconst64 (8 * k)))
  let bQ ← cudaCreateBuffer ptr (← iconst64 64)
  let n64 ← iconst64 64
  code3 0 (← ffi .cudaUploadAsync %[c, bQ, pp2, n64, s])
  -- An upload only reads its source: the host may read it while it is in flight.
  storeI64 (← load64 pp2) (← iadd out (← iconst64 840))
  code3 1 (← ffi .cudaUploadOffsetAsync %[c, bQ, ← iconst64 60, ← at_ SRC, ← iconst64 4, s])
  storeI32 bQ (← at_ BIND)
  code3 2 (← ffi .cudaLaunchNamedOnStream %[c, ← at_ KERNEL_NAMED, ← at_ ENTRY_NAME, one,
             ← at_ BIND, one, one, one, ← iconst32 64, one, one, s])
  code3 3 (← ffi .cudaDownloadAsync %[c, bQ, pp3, n64, s])
  code3 4 (← ffi .cudaDownloadAsync %[c, bQ, ← iadd out (← iconst64 736), n64, s])
  -- A download on the same stream into the upload's source is ordered after it.
  code3 8 (← ffi .cudaDownloadAsync %[c, bQ, pp2, n64, s])
  -- A length other than the buffer's is refused, both ways.
  code3 9 (← ffi .cudaUpload %[c, b0, ← iadd ptr (← iconst64 SRC), ← iconst64 32])
  code3 10 (← ffi .cudaDownload %[c, b0, ← iadd out (← iconst64 128), ← iconst64 32])
  code3 5 (← ffi .cudaStreamSync %[c, s])
  for k in List.range 8 do
    storeI64 (← load64 (← iadd pp2 (← iconst64 (8 * k)))) (← iadd out (← iconst64 (848 + 8 * k)))
  for k in List.range 8 do
    storeI64 (← load64 (← iadd pp3 (← iconst64 (8 * k)))) (← iadd out (← iconst64 (672 + 8 * k)))
  code3 6 (← ffi .cudaUploadAsync %[c, bQ, pp2, ← iconst64 65, s])
  code3 7 (← ffi .cudaDownloadAsync %[c, bQ, pp3, n64, ← iconst32 99])
  let _ ← ffi .cudaPinnedFree %[c, pid2]
  let _ ← ffi .cudaPinnedFree %[c, pid3]
  cudaCleanup ptr

def code : Code := Prog.emit body
def checked : Except String Code := Prog.emitChecked body
def program : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 body)]
def env : FnEnv := (Prog.run body).2.1

def entryArgVals : List Sem.V :=
  [.sc .i64 (Sem.regionBase .arena), .sc .i64 (Sem.regionBase .data),
   .sc .i64 8, .sc .i64 (Sem.regionBase .out), .sc .i64 OUT.toUInt64]

/-- The world the run starts from: the artifact's image in the arena, and the
    kernel oracle that is the kernel's own meaning. -/
def startWorld : Sem.World :=
  { mem := { arena := ⟨image.toArray⟩, data := ByteArray.mk (Array.replicate 8 0),
             out := ByteArray.mk (Array.replicate OUT 0) },
    kernel := addOne, vendor := vendorRef, memInfo := (1, 2) }

def expected : Except String ByteArray :=
  match Sem.run { env } entryArgVals startWorld code with
  | .stuck why | .misuse why | .fault why => .error why
  | .ok _ w => .ok w.mem.out

/-- The hazard the busy ranges exist for: the host reads pinned bytes a download
    on a stream is still writing. The model refuses it; with the stream synced
    first, the same read is fine. -/
def hazard (syncFirst : Bool) : Prog V L Unit := do
  let ptr ← basePtr
  let out ← outPtr
  cudaInit ptr
  let c ← cudaCtxPtr ptr
  let s ← ffi .cudaStreamCreate %[c]
  let pid ← ffi .cudaPinnedAlloc %[c, ← iconst64 64]
  let pp ← ffi .cudaPinnedPtr %[c, pid]
  let b ← cudaCreateBuffer ptr (← iconst64 64)
  let _ ← ffi .cudaDownloadAsync %[c, b, pp, ← iconst64 64, s]
  if syncFirst then
    let _ ← ffi .cudaStreamSync %[c, s]
  storeI64 (← load64 pp) out
  cudaCleanup ptr

def hazardStuck (syncFirst : Bool) : Bool :=
  match Sem.run { env := (Prog.run (hazard syncFirst)).2.1 } entryArgVals startWorld
      (Prog.emit (hazard syncFirst)) with
  | .stuck _ | .fault _ => true
  | _ => false

#guard hazardStuck false
#guard !hazardStuck true

end HProgCudaCorpus

open AlgorithmLib in
def Host.CudaCorpus.main (args : List String) : IO Unit := do
  let dir ← requireOutputDir args
  match HProgCudaCorpus.checked with
  | .error e => throw (IO.userError s!"the CUDA corpus body is not well-formed: {e}")
  | .ok _ => pure ()
  match HProgCudaCorpus.expected with
  | .error e => throw (IO.userError s!"interpreting the CUDA corpus: {e}")
  | .ok bytes =>
      let clif ← AlgorithmLib.Prog.orDie HProgCudaCorpus.program
      emitArtifacts dir #[artifactEntry "hprog_cuda_corpus" {
        functions := clif, required_memory := HProgCudaCorpus.MEM,
        initial_memory := HProgCudaCorpus.image
      }]
      let sideDir := System.FilePath.mk dir / "hprog_cuda_corpus"
      IO.FS.createDirAll sideDir
      IO.FS.writeFile (sideDir / "expected.json")
        (Lean.Json.mkObj [("expected", Lean.toJson (bytes.toList.map (·.toNat)))]).compress
      IO.println s!"CUDA corpus: {bytes.size} expected bytes"

#eval ShipScan.check "Host.CudaCorpus" `Host.CudaCorpus.main
