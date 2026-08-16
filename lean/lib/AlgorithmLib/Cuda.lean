import AlgorithmLib.IR
import AlgorithmLib.Layout
import AlgorithmLib.PTX

namespace AlgorithmLib

open AlgorithmLib.IR
open AlgorithmLib.PTX

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

/-- Phantom-typed tensor handle. `buf` is the slot holding a CUDA buffer id
    (i32 at runtime); `shape` is fully type-level.

    The slot is a bare `Nat`, so a tensor belongs to no particular surface: a
    handle carrying a builder's own value type could only be launched by the
    builder that produced it. -/
structure _root_.AlgorithmLib.Tensor (s : Shape) where
  buf : Nat

/-- The handle as an `HProg` slot. -/
def _root_.AlgorithmLib.Tensor.slot {s : Shape} (t : Tensor s) : Nat := t.buf

end Tensor

open Tensor (Dim Shape)

namespace Kernel

/-- Launch geometry — six compile-time `Nat`s.

    Grid dimensions are *data*, not a builder action: every geometry in this
    development is static, and a `Kernel` whose geometry named a monad could
    only be launched by that monad. A runtime-derived grid, if one is ever
    needed, belongs in a constructor beside these rather than in the field
    type. -/
structure Geom where
  gridX  : Nat
  gridY  : Nat := 1
  gridZ  : Nat := 1
  blockX : Nat := 256
  blockY : Nat := 1
  blockZ : Nat := 1

/-- A fully static geometry — grid + block are compile-time `Nat`s. -/
def Geom.static (gx : Nat) (gy : Nat := 1) (gz : Nat := 1)
    (bx : Nat := 256) (by_ : Nat := 1) (bz : Nat := 1) : Geom :=
  { gridX := gx, gridY := gy, gridZ := gz
    blockX := bx, blockY := by_, blockZ := bz }

/-- **A grid derived from the work it covers.**

    `Geom.static` takes a block count as a bare number: nothing relates the
    grid to the element count the kernel is proven over, so a wrong count
    leaves every kernel theorem intact while the launch reduces the wrong set.
    These two constructors carry that relation as an elaboration obligation.

    `covering n blocks lanes` — `blocks × lanes` work items, one per lane. -/
def Geom.covering (n blocks lanes : Nat) (h : blocks * lanes = n := by decide) : Geom :=
  Geom.static blocks 1 1 lanes 1 1

/-- `sweeping n lanes trips` — one block, whose `lanes` threads each loop
    `trips` times.  The reduction kernels: the grid is 1 and the coverage lives
    entirely in the trip count. -/
def Geom.sweeping (n lanes trips : Nat) (h : lanes * trips = n := by decide) : Geom :=
  Geom.static 1 1 1 lanes 1 1

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
  /-- The raw PTX builder.  Only used when `ptxText` is absent, so a kernel on
      the proven lowering supplies nothing here.

      It defaults to empty rather than being required because keeping a dead
      hand-written body next to a proven one is an active hazard, not
      documentation: it looks like the kernel, is never emitted, is never
      checked against anything, and drifts.  The declarative record that
      matters — name, params, geometry, slot — is still right here; the *shape*
      is documented by the `EWStmt` that actually ships. -/
  body      : PTX Unit := pure ()
  geom      : Kernel.Geom
  ptxOff    : Nat                          -- where its PTX null-term string lives
  /-- Emit this text instead of rendering `body`.  Set when the kernel has been
      migrated onto the proven lowering, whose PTX comes from
      `ML.emitProvenKernel` rather than from the raw builder. -/
  ptxText   : Option String := none

namespace Kernel

def Kernel.arity (k : _root_.AlgorithmLib.Kernel) : Nat := k.params.length

/-- Render the PTX module string for this kernel. -/
def ptxSource (k : _root_.AlgorithmLib.Kernel) : String :=
  match k.ptxText with
  | some t => t
  | none =>
    buildModuleWith
      { smemSize := k.smemBytes }
      [{ name := k.name, params := k.params.map (·.name), body := k.body }]

/-- Null-terminated UTF-8 bytes ready for embedding at `k.ptxOff`. -/
def ptxBytes (k : _root_.AlgorithmLib.Kernel) : List UInt8 :=
  (ptxSource k).toUTF8.toList ++ [0]

end Kernel

namespace Tensor


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


end Tensor

-- ---------------------------------------------------------------------------
-- Typed cuBLAS wrappers.
--
-- Hide the cuBLAS column-major + transpose flag mess behind a logical
-- "y = A @ x" interface where `A` has row-major shape [outN, inN].
-- ---------------------------------------------------------------------------

namespace CuBlas


end CuBlas

-- ---------------------------------------------------------------------------
-- Typed CUDA host/device transfer wrappers.
-- ---------------------------------------------------------------------------

namespace Tensor


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

/-- Static-index cell offset (compile-time `Nat` index). -/
def ArrayFld.cellOffsetStatic (a : ArrayFld cs n) (i : Nat) : Nat :=
  a.offset + i * cs

end AlgorithmLib.Layout
