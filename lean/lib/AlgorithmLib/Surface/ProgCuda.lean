import AlgorithmLib.Surface.Cuda
import AlgorithmLib.Surface.ProgFFI

/-!
# The typed CUDA layer, over `Prog`

`Cuda.lean` carries the descriptions --- `Kernel`, `Kernel.Geom`, `Shape`,
`BufferSlot s`. A `Geom` is six `Nat`s and a `BufferSlot` is a layout offset,
so nothing here re-declares them; what lives here is the *call* side.

The elaboration obligations come across unchanged. `launch2/3` still check each
argument's shape against the kernel's declared parameters by `rfl`, and still
derive the arity `launchAt` needs from that check, so a mismatch is a build
error on the generator rather than a wrong grid on the device.

What is new is that a buffer handle carries the surface's own value type. A
`Tsr V s` holds a `V .i32`, which is what a buffer id is, so a handle cannot be
passed where a byte count belongs and a handle from one body cannot reach
another.
-/

namespace AlgorithmLib.Prog

open AlgorithmLib.IR
open AlgorithmLib.Tensor (Dim Shape)


/-- Phantom-typed tensor handle: the value holding a CUDA buffer id, with the
    shape entirely at the type level. -/
structure Tsr (V : ClifTy → Type) (s : Shape) where
  buf : V .i32

/-- The same buffer under a shape with the same element count. No instruction
    is emitted; the obligation closes by `decide` when both shapes are
    fully static. -/
def Tsr.reshape {V} {s1 s2 : Shape} (t : Tsr V s1)
    (_h : Tensor.Shape.staticElems? s1 = Tensor.Shape.staticElems? s2 := by decide) :
    Tsr V s2 := ⟨t.buf⟩

/-- **Write the bind descriptor.** Buffers are named by id, one `i32` per
    binding at `bindOff + 4i`, which is the table the kernel's parameters were
    emitted against. The obligation says the table is as long as the kernel
    declares, discharged by `rfl` at the call site. -/
def kernelBindAt (k : AlgorithmLib.Kernel) (ptr : V .i64) (bindOff : Nat)
    (bufs : List (V .i32))
    (_harity : bufs.length = k.params.length := by rfl) : Prog V L Unit := do
  for (b, i) in bufs.zip (List.range bufs.length) do
    storeUnaligned b (← iaddImm ptr (bindOff + i * 4))

/-- **Issue the launch against a table already written.**

    Generators bind once --- where the buffers are made --- and launch many
    times, so the two halves sit in different functions and no scan within
    either can relate them. Taking the declared arity from `k.params.length`
    here, as `kernelBindAt` takes the table's length from the same `k`, makes
    the launch agree with its table across that boundary by construction. -/
def kernelRelaunch (k : AlgorithmLib.Kernel) (ptr : V .i64) (bindOff : Nat) :
    Prog V L Unit := do
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

/-- **Issue the launch with the block count supplied at run time.**

    For a kernel over a length the generator does not know --- declare its
    block shape with `Geom.perLaunch`, and pass the count here. Everything the
    arity check rests on is unchanged: the buffer count still comes from
    `k.params`, so only the grid is a register. -/
def kernelRelaunchN (k : AlgorithmLib.Kernel) (ptr : V .i64) (bindOff : Nat)
    (gridX : V .i32) : Prog V L Unit := do
  let arity32 ← iconst32 k.params.length
  let ptxOff64  ← iconst64 k.ptxOff
  let bindOff64 ← iconst64 bindOff
  let gy ← iconst32 k.geom.gridY
  let gz ← iconst32 k.geom.gridZ
  let bx ← iconst32 k.geom.blockX
  let by_ ← iconst32 k.geom.blockY
  let bz ← iconst32 k.geom.blockZ
  let _ ← cudaLaunch ptr ptxOff64 arity32 bindOff64 gridX gy gz bx by_ bz
  pure ()

/-- Bind and launch over a run-time block count, from one site. -/
def kernelLaunchAtN (k : AlgorithmLib.Kernel) (ptr : V .i64) (bindOff : Nat)
    (bufs : List (V .i32)) (gridX : V .i32)
    (harity : bufs.length = k.params.length := by rfl) : Prog V L Unit := do
  kernelBindAt k ptr bindOff bufs harity
  kernelRelaunchN k ptr bindOff gridX

/-- Bind and launch from one site. -/
def kernelLaunchAt (k : AlgorithmLib.Kernel) (ptr : V .i64) (bindOff : Nat)
    (bufs : List (V .i32))
    (harity : bufs.length = k.params.length := by rfl) : Prog V L Unit := do
  kernelBindAt k ptr bindOff bufs harity
  kernelRelaunch k ptr bindOff

/-- Bindings, shape checked against `k.params`. -/
def launch2 {s1 s2 : Shape} (k : AlgorithmLib.Kernel) (ptr : V .i64) (bindOff : Nat)
    (t1 : Tsr V s1) (t2 : Tsr V s2)
    (hsh : k.params.map Kernel.ParamSpec.shape = [s1, s2] := by rfl) : Prog V L Unit :=
  kernelLaunchAt k ptr bindOff [t1.buf, t2.buf]
    (by simpa using (congrArg List.length hsh).symm)

def launch3 {s1 s2 s3 : Shape} (k : AlgorithmLib.Kernel) (ptr : V .i64) (bindOff : Nat)
    (t1 : Tsr V s1) (t2 : Tsr V s2) (t3 : Tsr V s3)
    (hsh : k.params.map Kernel.ParamSpec.shape = [s1, s2, s3] := by rfl) :
    Prog V L Unit :=
  kernelLaunchAt k ptr bindOff [t1.buf, t2.buf, t3.buf]
    (by simpa using (congrArg List.length hsh).symm)

/-- Read the typed handle out of its layout slot. -/
def slotLoad {s : Shape} (b : AlgorithmLib.Tensor.BufferSlot s) (ptr : V .i64) :
    Prog V L (Tsr V s) := do
  let v ← load32 (← iaddImm ptr b.fld.offset)
  return ⟨v⟩

/-- Write a typed handle into its slot; a shape mismatch is a type error. -/
def slotStore {s : Shape} (b : AlgorithmLib.Tensor.BufferSlot s) (ptr : V .i64)
    (t : Tsr V s) : Prog V L Unit := do
  storeUnaligned t.buf (← iaddImm ptr b.fld.offset)

/-- Allocate a buffer of shape `s`; `bytes` is the runtime size. -/
def tensorCreate {s : Shape} (ptr bytes : V .i64) : Prog V L (Tsr V s) := do
  let buf ← cudaCreateBuffer ptr bytes
  return ⟨buf⟩

/-- Copy `bytes` from the host buffer at `hostPtr` into `t`. -/
def tensorUpload {s : Shape} (ptr : V .i64) (t : Tsr V s) (hostPtr bytes : V .i64) :
    Prog V L Unit := do
  let c ← cudaCtxPtr ptr
  let _ ← ffi .cudaUpload %[c, t.buf, hostPtr, bytes]
  pure ()

/-- Copy `bytes` out of `t` into the host buffer at `hostPtr`. -/
def tensorDownload {s : Shape} (ptr : V .i64) (t : Tsr V s) (hostPtr bytes : V .i64) :
    Prog V L Unit := do
  let c ← cudaCtxPtr ptr
  let _ ← ffi .cudaDownload %[c, t.buf, hostPtr, bytes]
  pure ()

/-- `y = A · x`, with `A` row-major so the call transposes --- the PyTorch
    convention the weights are stored in. -/
def cublasLinear {inN outN : Nat} (ptr : V .i64)
    (a : Tsr V [.sta outN, .sta inN]) (x : Tsr V [.sta inN]) (y : Tsr V [.sta outN]) :
    Prog V L Unit := do
  let trans ← iconst32 1
  let m32   ← iconst32 inN
  let n32   ← iconst32 outN
  let alpha ← iconst32 0x3F800000   -- 1.0
  let beta  ← iconst32 0            -- 0.0
  let _ ← cublasSgemv ptr trans m32 n32 alpha a.buf x.buf beta y.buf
  pure ()

/-- GQA attention scores: for each `(kv, i)`,
    `scores[kv, i, :seqLen] = alpha * K[kv, :seqLen, :] @ Q[kv, i]`. -/
def attnScoresQK {nKV gqaRatio headDim maxSeq : Nat} (ptr : V .i64)
    (alphaBits seqLen32 : V .i32) (seqLen64 : V .i64)
    (k : Tsr V [.sta nKV, .sta maxSeq, .sta headDim])
    (q : Tsr V [.sta nKV, .sta gqaRatio, .sta headDim])
    (scores : Tsr V [.sta nKV, .sta gqaRatio, .dyn]) : Prog V L Unit := do
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
    k.buf strideK q.buf strideQ zero32 scores.buf strideC nKV32
  pure ()

/-- GQA V-mix: for each `(kv, i)`,
    `out[kv, i] = V[kv, :seqLen, :]^T @ probs[kv, i, :seqLen]`. -/
def attnMixV {nKV gqaRatio headDim maxSeq : Nat} (ptr : V .i64)
    (alphaBits seqLen32 : V .i32) (seqLen64 : V .i64)
    (v : Tsr V [.sta nKV, .sta maxSeq, .sta headDim])
    (probs : Tsr V [.sta nKV, .sta gqaRatio, .dyn])
    (out : Tsr V [.sta nKV, .sta gqaRatio, .sta headDim]) : Prog V L Unit := do
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
    v.buf strideV probs.buf strideP zero32 out.buf strideC nKV32
  pure ()

end AlgorithmLib.Prog
