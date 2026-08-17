import AlgorithmLib.Cuda
import AlgorithmLib.HProgFFI

/-!
# The typed CUDA layer, over `HProg.Sur`

Not re-exported from the root module: `Cuda.lean` imports the root, so a
generator that wants this surface imports `AlgorithmLib.HProgCuda` directly.

`Cuda.lean` carries the descriptions — `Kernel`, `Kernel.Geom`, `Tensor s`,
`BufferSlot s`. A `Geom` is six `Nat`s and a `Tensor` is a slot number, so
nothing here re-declares them; what lives here is the *call* side.

The elaboration obligations come across unchanged. `launch2/3` still check
each argument's shape against the kernel's declared parameters by `rfl`, and
still derive the arity `launchAt` needs from that check, so a mismatch is a
build error on the generator rather than a wrong grid on the device.
-/

namespace AlgorithmLib.HProg.Sur

open AlgorithmLib.IR
open AlgorithmLib.Tensor (Dim Shape)

/-- **Write the bind descriptor.**  Buffers are named by slot, one `i32` per
    binding at `bindOff + 4i`, which is the table the kernel's parameters were
    emitted against.  The obligation says the table is as long as the kernel
    declares, discharged by `rfl` at the call site. -/
def kernelBindAt
    (k : AlgorithmLib.Kernel) (ptr : R)
    (bindOff : Nat) (bufs : List R)
    (_harity : bufs.length = k.params.length := by rfl) : M Unit := do
  for (b, i) in bufs.zip (List.range bufs.length) do
    storeUnaligned b (← iaddImm ptr (bindOff + i * 4))

/-- **Issue the launch against a table already written.**

    Generators bind once — where the buffers are made — and launch many times,
    so the two halves sit in different functions and no scan within either can
    relate them.  Taking the declared arity from `k.params.length` here, as
    `kernelBindAt` takes the table's length from the same `k`, makes the launch
    agree with its table across that boundary by construction. -/
def kernelRelaunch (k : AlgorithmLib.Kernel) (ptr : R) (bindOff : Nat) : M Unit := do
  let arity32 ← iconst32 k.params.length
  let ptxOff64  ← iconst64 k.ptxOff
  let bindOff64 ← iconst64 bindOff
  let gx ← iconst32 k.geom.gridX
  let gy ← iconst32 k.geom.gridY
  let gz ← iconst32 k.geom.gridZ
  let bx ← iconst32 k.geom.blockX
  let by_ ← iconst32 k.geom.blockY
  let bz ← iconst32 k.geom.blockZ
  let _ ← cudaLaunch ptr ptxOff64 arity32 bindOff64 gx gy gz bx by_ bz
  pure ()

/-- Bind and launch from one site. -/
def kernelLaunchAt
    (k : AlgorithmLib.Kernel) (ptr : R)
    (bindOff : Nat) (bufs : List R)
    (harity : bufs.length = k.params.length := by rfl) : M Unit := do
  kernelBindAt k ptr bindOff bufs harity
  kernelRelaunch k ptr bindOff

/-- Bindings, shape checked against `k.params`. -/
def launch2 {s1 s2 : Shape}
    (k : AlgorithmLib.Kernel) (ptr : R) (bindOff : Nat)
    (t1 : Tensor s1) (t2 : Tensor s2)
    (hsh : k.params.map Kernel.ParamSpec.shape = [s1, s2] := by rfl) : M Unit :=
  kernelLaunchAt k ptr bindOff [t1.slot, t2.slot]
    (by simpa using (congrArg List.length hsh).symm)

def launch3 {s1 s2 s3 : Shape}
    (k : AlgorithmLib.Kernel) (ptr : R) (bindOff : Nat)
    (t1 : Tensor s1) (t2 : Tensor s2) (t3 : Tensor s3)
    (hsh : k.params.map Kernel.ParamSpec.shape = [s1, s2, s3] := by rfl) : M Unit :=
  kernelLaunchAt k ptr bindOff [t1.slot, t2.slot, t3.slot]
    (by simpa using (congrArg List.length hsh).symm)


/-- Read the typed handle out of its layout slot. -/
def slotLoad {s : Shape} (b : AlgorithmLib.Tensor.BufferSlot s) (ptr : R) : M (Tensor s) := do
  let v ← load32 (← iaddImm ptr b.fld.offset)
  return ⟨v⟩

/-- Write a typed handle into its slot; a shape mismatch is a type error. -/
def slotStore {s : Shape} (b : AlgorithmLib.Tensor.BufferSlot s) (ptr : R) (t : Tensor s) :
    M Unit := do
  storeUnaligned t.slot (← iaddImm ptr b.fld.offset)

/-- Allocate a buffer of shape `s`; `bytes` is the runtime size. -/
def tensorCreate {s : Shape} (ptr bytes : R) : M (Tensor s) := do
  let buf ← cudaCreateBuffer ptr bytes
  return ⟨buf⟩


/-- Copy `bytes` from the host buffer at `hostPtr` into `t`. -/
def tensorUpload {s : Shape} (ptr : R) (t : Tensor s)
    (hostPtr bytes : R) : M Unit := do
  let ctxPtr ← cudaCtxPtr ptr
  let _ ← call IR.Ffi.cudaUpload.id [ctxPtr, t.slot, hostPtr, bytes]
  pure ()

/-- Copy `bytes` out of `t` into the host buffer at `hostPtr`. -/
def tensorDownload {s : Shape} (ptr : R) (t : Tensor s)
    (hostPtr bytes : R) : M Unit := do
  let ctxPtr ← cudaCtxPtr ptr
  let _ ← call IR.Ffi.cudaDownload.id [ctxPtr, t.slot, hostPtr, bytes]
  pure ()

/-- `y = A · x`, with `A` row-major so the call transposes — the PyTorch
    convention the weights are stored in. -/
def cublasLinear {inN outN : Nat} (ptr : R)
    (a : Tensor [.sta outN, .sta inN])
    (x : Tensor [.sta inN])
    (y : Tensor [.sta outN]) : M Unit := do
  let trans ← iconst32 1
  let m32   ← iconst32 inN
  let n32   ← iconst32 outN
  let alpha ← iconst32 0x3F800000   -- 1.0
  let beta  ← iconst32 0            -- 0.0
  let _ ← cublasSgemv ptr trans m32 n32 alpha a.slot x.slot beta y.slot
  pure ()


/-- GQA attention scores: for each `(kv, i)`,
    `scores[kv, i, :seqLen] = alpha * K[kv, :seqLen, :] @ Q[kv, i]`. -/
def attnScoresQK {nKV gqaRatio headDim maxSeq : Nat}
    (ptr : R)
    (alphaBits seqLen32 seqLen64 : R)
    (k : Tensor [.sta nKV, .sta maxSeq, .sta headDim])
    (q : Tensor [.sta nKV, .sta gqaRatio, .sta headDim])
    (scores : Tensor [.sta nKV, .sta gqaRatio, .dyn]) : M Unit := do
  let zero32   ← iconst32 0
  let one32    ← iconst32 1
  let k32      ← iconst32 headDim
  let gqaR32   ← iconst32 gqaRatio
  let strideK  ← iconst64 (maxSeq * headDim)
  let strideQ  ← iconst64 (gqaRatio * headDim)
  let gqaR64   ← iconst64 gqaRatio
  let strideC  ← imul gqaR64 seqLen64
  let nKV32    ← iconst32 nKV
  let _ ← cublasSgemmStridedBatched ptr one32 zero32
    seqLen32 gqaR32 k32 alphaBits
    k.slot strideK q.slot strideQ zero32 scores.slot strideC nKV32
  pure ()

/-- GQA V-mix: for each `(kv, i)`,
    `out[kv, i] = V[kv, :seqLen, :]^T @ probs[kv, i, :seqLen]`. -/
def attnMixV {nKV gqaRatio headDim maxSeq : Nat}
    (ptr : R)
    (alphaBits seqLen32 seqLen64 : R)
    (v : Tensor [.sta nKV, .sta maxSeq, .sta headDim])
    (probs : Tensor [.sta nKV, .sta gqaRatio, .dyn])
    (out : Tensor [.sta nKV, .sta gqaRatio, .sta headDim]) : M Unit := do
  let zero32   ← iconst32 0
  let hd32     ← iconst32 headDim
  let gqaR32   ← iconst32 gqaRatio
  let strideV  ← iconst64 (maxSeq * headDim)
  let gqaR64   ← iconst64 gqaRatio
  let strideP  ← imul gqaR64 seqLen64
  let strideC  ← iconst64 (gqaRatio * headDim)
  let nKV32    ← iconst32 nKV
  let _ ← cublasSgemmStridedBatched ptr zero32 zero32
    hd32 gqaR32 seqLen32 alphaBits
    v.slot strideV probs.slot strideP zero32 out.slot strideC nKV32
  pure ()

end AlgorithmLib.HProg.Sur
