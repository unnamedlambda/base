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

end AlgorithmLib
