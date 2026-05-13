import AlgorithmLib

namespace AlgorithmLib

open AlgorithmLib.IR
open AlgorithmLib.PTX

/-!
  Layer 2: typed CUDA tensor + kernel surface.

  Sits between Layer 3 (model code — e.g. Qwen2) and Layer 1 (PTX.lean,
  register-level emission). Provides:

  - `Dim`, `Shape`            — tensor shapes with mixed static/dynamic dims
  - `Tensor s`                — phantom-typed handle to a CUDA buffer
  - `Kernel`                  — declarative kernel: bind shapes + PTX body +
                                 launch geometry + assigned PTX offset.
  - typed launchers           — `launch3 k cuda ptr bindOff t1 t2 t3` etc.

  All build-time machinery. Emitted PTX bytes and `cudaLaunch` calls match
  what the hand-written form produces — the typing is a compile-time
  discipline.
-/

namespace Tensor

/-- A tensor dimension: static compile-time `Nat` or runtime placeholder. -/
inductive Dim where
  | sta : Nat → Dim
  | dyn : Dim
  deriving BEq, Repr

/-- Tensor shape — list of dims, leftmost is outermost. -/
abbrev Shape := List Dim

/-- Total element count if shape is fully static. -/
def Shape.staticElems? : Shape → Option Nat
  | []              => some 1
  | .sta n :: rest  => Option.map (· * n) (Shape.staticElems? rest)
  | .dyn   :: _     => none

/-- Static byte size assuming f32 elements (4 bytes). -/
def Shape.staticBytesF32? (s : Shape) : Option Nat :=
  (Shape.staticElems? s).map (· * 4)

/-- Pretty-printed shape for diagnostics. -/
def Shape.render : Shape → String
  | []   => "[]"
  | dims =>
    let r : Dim → String
      | .sta n => toString n
      | .dyn   => "?"
    "[" ++ String.intercalate ", " (dims.map r) ++ "]"

/-- Phantom-typed tensor handle. `buf` is a CUDA buffer id (i32 at runtime);
    `shape` is fully type-level. -/
structure _root_.AlgorithmLib.Tensor (s : Shape) where
  buf : Val

end Tensor

open Tensor (Dim Shape)

namespace Kernel

/-- Launch geometry — block dims are static `Nat`s; grid dims are produced
    by `IRBuilder` so they can depend on runtime values. -/
structure Geom where
  gridX  : IRBuilder Val
  gridY  : IRBuilder Val := iconst32 1
  gridZ  : IRBuilder Val := iconst32 1
  blockX : Nat := 256
  blockY : Nat := 1
  blockZ : Nat := 1

/-- A fully static geometry — grid + block are compile-time `Nat`s. -/
def Geom.static (gx : Nat) (gy : Nat := 1) (gz : Nat := 1)
    (bx : Nat := 256) (by_ : Nat := 1) (bz : Nat := 1) : Geom :=
  { gridX  := iconst32 gx
    gridY  := iconst32 gy
    gridZ  := iconst32 gz
    blockX := bx, blockY := by_, blockZ := bz }

/-- One kernel parameter — shape + role + the `ldParam` name. -/
structure ParamSpec where
  shape : Shape
  ro    : Bool := false
  name  : String

end Kernel

/-- A declarative CUDA kernel — PTX body + binding shapes + launch geometry
    + assigned PTX offset in shared memory. The bind descriptor offset is
    *per call site* (passed to `launchAt`), so the same kernel can be
    launched from many sites with different scratch areas. -/
structure Kernel where
  name      : String
  params    : List Kernel.ParamSpec
  smemBytes : Nat := 0
  body      : PTX Unit
  geom      : Kernel.Geom
  ptxOff    : Nat                          -- where its PTX null-term string lives

namespace Kernel

def Kernel.arity (k : _root_.AlgorithmLib.Kernel) : Nat := k.params.length

/-- Render the PTX module string for this kernel. -/
def ptxSource (k : _root_.AlgorithmLib.Kernel) : String :=
  buildModuleWith
    { smemSize := k.smemBytes }
    [{ name := k.name, params := k.params.map (·.name), body := k.body }]

/-- Null-terminated UTF-8 bytes ready for embedding at `k.ptxOff`. -/
def ptxBytes (k : _root_.AlgorithmLib.Kernel) : List UInt8 :=
  (ptxSource k).toUTF8.toList ++ [0]

end Kernel

namespace Tensor

/-- Untyped launch — writes `bufs` to bind descriptor at `bindOff` and
    issues the cudaLaunch. Checks arity. Used internally; callers normally
    use the typed `launch1/2/3/4` wrappers. -/
def Kernel.launchAt
    (k : _root_.AlgorithmLib.Kernel) (cuda : CudaSetup) (ptr : Val)
    (bindOff : Nat) (bufs : List Val) : IRBuilder Unit := do
  let expected := k.params.length
  if bufs.length ≠ expected then
    panic! s!"kernel {k.name}: launched with {bufs.length} bufs, expected {expected}"
  -- Write bind descriptor: one i32 per buffer at bindOff + 4*i
  for (b, i) in bufs.zip (List.range bufs.length) do
    storeI32 b (← iaddImm ptr (bindOff + i * 4))
  let arity32 ← iconst32 expected
  let ptxOff64  ← iconst64 k.ptxOff
  let bindOff64 ← iconst64 bindOff
  let gx ← k.geom.gridX
  let gy ← k.geom.gridY
  let gz ← k.geom.gridZ
  let bx ← iconst32 k.geom.blockX
  let by_ ← iconst32 k.geom.blockY
  let bz ← iconst32 k.geom.blockZ
  let _ ← cudaLaunch cuda ptr ptxOff64 arity32 bindOff64 gx gy gz bx by_ bz
  pure ()

/-- Typed 1-binding launch. -/
def launch1 {s1 : Shape}
    (k : _root_.AlgorithmLib.Kernel) (cuda : CudaSetup) (ptr : Val) (bindOff : Nat)
    (t1 : Tensor s1) : IRBuilder Unit :=
  Kernel.launchAt k cuda ptr bindOff [t1.buf]

/-- Typed 2-binding launch. -/
def launch2 {s1 s2 : Shape}
    (k : _root_.AlgorithmLib.Kernel) (cuda : CudaSetup) (ptr : Val) (bindOff : Nat)
    (t1 : Tensor s1) (t2 : Tensor s2) : IRBuilder Unit :=
  Kernel.launchAt k cuda ptr bindOff [t1.buf, t2.buf]

/-- Typed 3-binding launch. -/
def launch3 {s1 s2 s3 : Shape}
    (k : _root_.AlgorithmLib.Kernel) (cuda : CudaSetup) (ptr : Val) (bindOff : Nat)
    (t1 : Tensor s1) (t2 : Tensor s2) (t3 : Tensor s3) : IRBuilder Unit :=
  Kernel.launchAt k cuda ptr bindOff [t1.buf, t2.buf, t3.buf]

/-- Typed 4-binding launch. -/
def launch4 {s1 s2 s3 s4 : Shape}
    (k : _root_.AlgorithmLib.Kernel) (cuda : CudaSetup) (ptr : Val) (bindOff : Nat)
    (t1 : Tensor s1) (t2 : Tensor s2) (t3 : Tensor s3) (t4 : Tensor s4) : IRBuilder Unit :=
  Kernel.launchAt k cuda ptr bindOff [t1.buf, t2.buf, t3.buf, t4.buf]

end Tensor

-- ---------------------------------------------------------------------------
-- BufferSlot: shape-typed Layout slot for a CUDA buffer id.
--
-- Closes the loop between Layout (where buffers are allocated) and Tensor
-- (where shapes are tracked).  A `BufferSlot s` is an `Fld .i32` whose
-- contents — a CUDA buffer id — point to memory holding a `Tensor s`.
-- Loading the slot returns a typed `Tensor s` directly; no `⟨buf⟩` casts.
-- ---------------------------------------------------------------------------

namespace Tensor

/-- A shape-typed memory slot for a CUDA buffer id. -/
structure BufferSlot (s : Shape) where
  fld : Layout.Fld .i32

/-- Allocate a typed buffer slot in the current layout. -/
def slotOf (s : Shape) : Layout.LayoutBuilder (BufferSlot s) := do
  let fld ← Layout.field .i32
  return ⟨fld⟩

/-- Construct a typed buffer slot at a fixed, externally-known offset.
    Shape is inferred from the expected return type.  Useful for incremental
    migration: keep an existing offset constant while gaining typed load/store. -/
def slotOfAt {s : Shape} (offset : Nat) : BufferSlot s := ⟨{ offset := offset }⟩

/-- Reshape a tensor to a different shape with the same total element count.
    No runtime cost; the proof obligation closes by `decide` when both shapes
    are fully-static and evaluate to the same product. -/
def reshape {s1 s2 : Shape} (t : Tensor s1)
    (_h : Shape.staticElems? s1 = Shape.staticElems? s2 := by decide) : Tensor s2 :=
  ⟨t.buf⟩

/-- Load the typed `Tensor s` from this slot. -/
def BufferSlot.load (b : BufferSlot s) (ptr : Val) : IRBuilder (Tensor s) := do
  let v ← load32 (← iaddImm ptr b.fld.offset)
  return ⟨v⟩

/-- Store a typed `Tensor s` into the slot.  Shape mismatch is a type error. -/
def BufferSlot.store (b : BufferSlot s) (ptr : Val) (t : Tensor s) : IRBuilder Unit := do
  storeI32 t.buf (← iaddImm ptr b.fld.offset)

/-- Allocate a CUDA buffer with shape `s`, returning a typed `Tensor s`.
    `bytes` is the runtime size in bytes; for fully-static shapes use
    `createStatic` which derives the size from the type. -/
def Tensor.create {s : Shape} (cuda : CudaSetup) (ptr : Val) (bytes : Val) :
    IRBuilder (Tensor s) := do
  let buf ← cudaCreateBuffer cuda ptr bytes
  return ⟨buf⟩

/-- Allocate a CUDA buffer with a fully-static shape.  Byte size is computed
    from the shape at elaboration time (`f32` elements assumed).  If `s`
    contains any `.dyn` dimension, this fails to elaborate. -/
def Tensor.createStatic (cuda : CudaSetup) (ptr : Val) (s : Shape)
    (_h : Shape.staticElems? s = some n := by decide) : IRBuilder (Tensor s) := do
  let bytes ← iconst64 (n * 4)
  let buf ← cudaCreateBuffer cuda ptr bytes
  return ⟨buf⟩

end Tensor

-- ---------------------------------------------------------------------------
-- Typed cuBLAS wrappers.
--
-- Hide the cuBLAS column-major + transpose flag mess behind a logical
-- "y = A @ x" interface where `A` has row-major shape [outN, inN].
-- ---------------------------------------------------------------------------

namespace CuBlas

/-- Linear projection: `y = A @ x` where `A : [outN, inN]`, `x : [inN]`, `y : [outN]`.
    Shape errors are elaboration errors.  Uses cuBLAS sgemv with trans=1
    (since weights are stored row-major as PyTorch convention). -/
def linear {inN outN : Nat} (cublas : CuBlasSetup) (ptr : Val)
    (a : Tensor [.sta outN, .sta inN])
    (x : Tensor [.sta inN])
    (y : Tensor [.sta outN]) : IRBuilder Unit := do
  let trans ← iconst32 1
  let m32   ← iconst32 inN
  let n32   ← iconst32 outN
  let alpha ← iconst32 0x3F800000   -- 1.0
  let beta  ← iconst32 0            -- 0.0
  let _ ← cublasSgemv cublas ptr trans m32 n32 alpha a.buf x.buf beta y.buf
  pure ()

/-- Linear projection with explicit alpha/beta scalars (raw f32 bit patterns). -/
def linearAB {inN outN : Nat} (cublas : CuBlasSetup) (ptr : Val)
    (alphaBits betaBits : Val)
    (a : Tensor [.sta outN, .sta inN])
    (x : Tensor [.sta inN])
    (y : Tensor [.sta outN]) : IRBuilder Unit := do
  let trans ← iconst32 1
  let m32   ← iconst32 inN
  let n32   ← iconst32 outN
  let _ ← cublasSgemv cublas ptr trans m32 n32 alphaBits a.buf x.buf betaBits y.buf
  pure ()

/-- Per-head attention scores: for each `h ∈ [0, nQ)`,
    `scores[h, :seqLen] = alpha * K[h, :seqLen, :] @ Q[h]`.

    Shapes (statically checked across all three tensors):
    - `K` : `[nQ, maxSeq, headDim]`
    - `Q` : `[nQ, headDim]`
    - `scores` : `[nQ, .dyn]` (dynamic seq dim) -/
def attnScoresQK {nQ headDim maxSeq : Nat}
    (cublas : CuBlasSetup) (ptr : Val)
    (alphaBits : Val) (seqLen32 seqLen64 : Val)
    (k : Tensor [.sta nQ, .sta maxSeq, .sta headDim])
    (q : Tensor [.sta nQ, .sta headDim])
    (scores : Tensor [.sta nQ, .dyn]) : IRBuilder Unit := do
  let one32   ← iconst32 1
  let zero32  ← iconst32 0
  let k32     ← iconst32 headDim
  let strideK ← iconst64 (maxSeq * headDim)
  let strideQ ← iconst64 headDim
  let nQ32    ← iconst32 nQ
  let _ ← cublasSgemmStridedBatched cublas ptr one32 zero32
    seqLen32 one32 k32 alphaBits
    k.buf strideK q.buf strideQ zero32 scores.buf seqLen64 nQ32
  pure ()

/-- Per-head V-mix: for each `h ∈ [0, nQ)`,
    `out[h] = V[h, :seqLen, :]^T @ probs[h, :seqLen]`.

    Shapes (statically checked):
    - `V`     : `[nQ, maxSeq, headDim]`
    - `probs` : `[nQ, .dyn]`
    - `out`   : `[nQ, headDim]` -/
def attnMixV {nQ headDim maxSeq : Nat}
    (cublas : CuBlasSetup) (ptr : Val)
    (alphaBits : Val) (seqLen32 seqLen64 : Val)
    (v : Tensor [.sta nQ, .sta maxSeq, .sta headDim])
    (probs : Tensor [.sta nQ, .dyn])
    (out : Tensor [.sta nQ, .sta headDim]) : IRBuilder Unit := do
  let zero32   ← iconst32 0
  let one32    ← iconst32 1
  let hd32     ← iconst32 headDim
  let hd64     ← iconst64 headDim
  let strideV  ← iconst64 (maxSeq * headDim)
  let nQ32     ← iconst32 nQ
  let _ ← cublasSgemmStridedBatched cublas ptr zero32 zero32
    hd32 one32 seqLen32 alphaBits
    v.buf strideV probs.buf seqLen64 zero32 out.buf hd64 nQ32
  pure ()

end CuBlas

-- ---------------------------------------------------------------------------
-- Typed CUDA host/device transfer wrappers.
-- ---------------------------------------------------------------------------

namespace Tensor

/-- Upload `bytes` bytes from host buffer `hostPtr` into typed tensor `t`. -/
def upload {s} (cuda : CudaSetup) (ptr : Val) (t : Tensor s)
    (hostPtr bytes : Val) : IRBuilder Unit := do
  let ctxPtr ← cudaCtxPtr ptr
  let _ ← call cuda.fnUpload [ctxPtr, t.buf, hostPtr, bytes]
  pure ()

/-- Download `bytes` bytes from typed tensor `t` into host buffer `hostPtr`. -/
def download {s} (cuda : CudaSetup) (ptr : Val) (t : Tensor s)
    (hostPtr bytes : Val) : IRBuilder Unit := do
  let ctxPtr ← cudaCtxPtr ptr
  let _ ← call cuda.fnDownload [ctxPtr, t.buf, hostPtr, bytes]
  pure ()

end Tensor

end AlgorithmLib

-- ---------------------------------------------------------------------------
-- Layout extensions: indexed array field for slot arrays.
-- ---------------------------------------------------------------------------

namespace AlgorithmLib.Layout

/-- A 1-D array field — `count` cells, each `cellSize` bytes.  Provides
    runtime-indexed offset access (`cellOffset`) and a static
    `cellOffsetStatic` for compile-time-known indices. -/
structure ArrayFld (cellSize : Nat) (count : Nat) where
  offset : Nat
  deriving Repr

/-- Allocate an `ArrayFld` (`count` cells of `cellSize` bytes each). -/
def arrayField (cellSize : Nat) (count : Nat) : LayoutBuilder (ArrayFld cellSize count) :=
  modifyGet fun s =>
    let totalBytes := cellSize * count
    let f : ArrayFld cellSize count := { offset := s.cursor }
    let anyFld : AnyFld := { offset := s.cursor, ty := .bytes totalBytes }
    (f, { fields := s.fields ++ [anyFld], cursor := s.cursor + totalBytes })

/-- Byte offset of cell `i` from the base of shared memory, computed at runtime. -/
def ArrayFld.cellOffset (a : ArrayFld cs n) (i : AlgorithmLib.IR.Val) :
    AlgorithmLib.IR.IRBuilder AlgorithmLib.IR.Val := do
  let stride ← AlgorithmLib.IR.iconst64 cs
  let scaled ← AlgorithmLib.IR.imul i stride
  AlgorithmLib.IR.iaddImm scaled a.offset

/-- Absolute address of cell `i`: `ptr + cellOffset(i)`. -/
def ArrayFld.cellAddr (a : ArrayFld cs n) (ptr : AlgorithmLib.IR.Val) (i : AlgorithmLib.IR.Val) :
    AlgorithmLib.IR.IRBuilder AlgorithmLib.IR.Val := do
  let off ← a.cellOffset i
  AlgorithmLib.IR.iadd ptr off

/-- Static-index cell offset (compile-time `Nat` index). -/
def ArrayFld.cellOffsetStatic (a : ArrayFld cs n) (i : Nat) : Nat :=
  a.offset + i * cs

end AlgorithmLib.Layout
