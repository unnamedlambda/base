module
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# Typed handles

A resource id is an `i32` and a context is an `i64`, so at the value level a
stream, an event, a pinned allocation and a byte count are interchangeable ---
and passing one where another belongs is a wrong answer on the device, not an
error at build time. Each resource here is its own type over the surface's
value type, and each wrapper takes and returns the types it means: a
`streamWaitEvent` takes a `Strm` and an `Evt`, in that order, and nothing else
elaborates.

The wrappers emit exactly the calls the untyped ones do; a handle's `id` is
the value, for a body that needs the number itself. Device buffers keep
`Tsr` (`ProgCuda`), which also carries their shape.
-/

namespace AlgorithmLib.Prog

open AlgorithmLib.IR

/-- The families that hold a context in a slot. -/
inductive Family where
  | cuda | wgpu | ht | lmdb | window

/-- A family's context, loaded from its slot: the value its calls take first. -/
structure Ctx (V : ClifTy → Type) (fam : Family) where
  ptr : V .i64

/-- A created CUDA stream. -/
structure Strm (V : ClifTy → Type) where
  id : V .i32

/-- A created CUDA event. -/
structure Evt (V : ClifTy → Type) where
  id : V .i32

/-- A captured CUDA graph. -/
structure Graph (V : ClifTy → Type) where
  id : V .i32

/-- A pinned host allocation. -/
structure Pinned (V : ClifTy → Type) where
  id : V .i32

/-- A wgpu buffer. -/
structure GBuf (V : ClifTy → Type) where
  id : V .i32

/-- A wgpu compute pipeline. -/
structure GPipe (V : ClifTy → Type) where
  id : V .i32

/-- An open LMDB environment. -/
structure DbEnv (V : ClifTy → Type) where
  id : V .i32

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

def cudaCtx (ptr : V .i64) (slotOffset : Nat := ContextSlots.cuda) : Prog V L (Ctx V .cuda) := do
  pure ⟨← cudaCtxPtr ptr slotOffset⟩

def wgpuCtx (ptr : V .i64) (slotOffset : Nat := ContextSlots.wgpu) : Prog V L (Ctx V .wgpu) := do
  pure ⟨← gpuCtxPtr ptr slotOffset⟩

-- CUDA streams, events and graphs

def Ctx.streamCreate (c : Ctx V .cuda) : Prog V L (Strm V) := do
  pure ⟨← ffi .cudaStreamCreate %[c.ptr]⟩

def Ctx.streamSync (c : Ctx V .cuda) (s : Strm V) : Prog V L (V .i32) :=
  ffi .cudaStreamSync %[c.ptr, s.id]

def Ctx.streamDestroy (c : Ctx V .cuda) (s : Strm V) : Prog V L (V .i32) :=
  ffi .cudaStreamDestroy %[c.ptr, s.id]

def Ctx.eventCreate (c : Ctx V .cuda) : Prog V L (Evt V) := do
  pure ⟨← ffi .cudaEventCreate %[c.ptr]⟩

def Ctx.eventRecord (c : Ctx V .cuda) (e : Evt V) (s : Strm V) : Prog V L (V .i32) :=
  ffi .cudaEventRecord %[c.ptr, e.id, s.id]

def Ctx.streamWaitEvent (c : Ctx V .cuda) (s : Strm V) (e : Evt V) : Prog V L (V .i32) :=
  ffi .cudaStreamWaitEvent %[c.ptr, s.id, e.id]

def Ctx.eventDestroy (c : Ctx V .cuda) (e : Evt V) : Prog V L (V .i32) :=
  ffi .cudaEventDestroy %[c.ptr, e.id]

def Ctx.beginCapture (c : Ctx V .cuda) (s : Strm V) : Prog V L (V .i32) :=
  ffi .cudaGraphBeginCapture %[c.ptr, s.id]

def Ctx.endCapture (c : Ctx V .cuda) (s : Strm V) : Prog V L (Graph V) := do
  pure ⟨← ffi .cudaGraphEndCapture %[c.ptr, s.id]⟩

def Ctx.graphUpload (c : Ctx V .cuda) (g : Graph V) (s : Strm V) : Prog V L (V .i32) :=
  ffi .cudaGraphUpload %[c.ptr, g.id, s.id]

def Ctx.graphLaunch (c : Ctx V .cuda) (g : Graph V) (s : Strm V) : Prog V L (V .i32) :=
  ffi .cudaGraphLaunch %[c.ptr, g.id, s.id]

def Ctx.graphDestroy (c : Ctx V .cuda) (g : Graph V) : Prog V L (V .i32) :=
  ffi .cudaGraphDestroy %[c.ptr, g.id]

/-- A launch on a stream: the kernel text and bind table as host pointers. -/
def Ctx.launchOn (c : Ctx V .cuda) (kernel : V .i64) (nBufs : V .i32) (bind : V .i64)
    (gx gy gz bx by_ bz : V .i32) (s : Strm V) : Prog V L (V .i32) :=
  ffi .cudaLaunchOnStream %[c.ptr, kernel, nBufs, bind, gx, gy, gz, bx, by_, bz, s.id]

-- pinned host memory

def Ctx.pinnedAlloc (c : Ctx V .cuda) (size : V .i64) : Prog V L (Pinned V) := do
  pure ⟨← ffi .cudaPinnedAlloc %[c.ptr, size]⟩

def Ctx.pinnedPtr (c : Ctx V .cuda) (p : Pinned V) : Prog V L (V .i64) :=
  ffi .cudaPinnedPtr %[c.ptr, p.id]

def Ctx.pinnedFree (c : Ctx V .cuda) (p : Pinned V) : Prog V L (V .i32) :=
  ffi .cudaPinnedFree %[c.ptr, p.id]

-- wgpu

def Ctx.gbufCreate (c : Ctx V .wgpu) (size : V .i64) : Prog V L (GBuf V) := do
  pure ⟨← ffi .gpuCreateBuffer %[c.ptr, size]⟩

def Ctx.pipeCreate (c : Ctx V .wgpu) (shader bind : V .i64) (nBindings : V .i32) :
    Prog V L (GPipe V) := do
  pure ⟨← ffi .gpuCreatePipeline %[c.ptr, shader, bind, nBindings]⟩

def Ctx.dispatch (c : Ctx V .wgpu) (p : GPipe V) (x y z : V .i32) : Prog V L (V .i32) :=
  ffi .gpuDispatch %[c.ptr, p.id, x, y, z]

def Ctx.gbufDownload (c : Ctx V .wgpu) (b : GBuf V) (dst size : V .i64) : Prog V L (V .i32) :=
  ffi .gpuDownload %[c.ptr, b.id, dst, size]

-- LMDB

def Ctx.dbOpen (c : Ctx V .lmdb) (path : V .i64) (mapMb : V .i32) : Prog V L (DbEnv V) := do
  pure ⟨← ffi .lmdbOpen %[c.ptr, path, mapMb]⟩

def Ctx.dbBegin (c : Ctx V .lmdb) (e : DbEnv V) : Prog V L (V .i32) :=
  ffi .lmdbBeginWriteTxn %[c.ptr, e.id]

def Ctx.dbPut (c : Ctx V .lmdb) (e : DbEnv V) (key : V .i64) (keyLen : V .i32)
    (val : V .i64) (valLen : V .i32) : Prog V L (V .i32) :=
  ffi .lmdbPut %[c.ptr, e.id, key, keyLen, val, valLen]

def Ctx.dbCommit (c : Ctx V .lmdb) (e : DbEnv V) : Prog V L (V .i32) :=
  ffi .lmdbCommitWriteTxn %[c.ptr, e.id]

def Ctx.dbScan (c : Ctx V .lmdb) (e : DbEnv V) (key : V .i64) (keyLen maxEntries : V .i32)
    (result room : V .i64) : Prog V L (V .i32) :=
  ffi .lmdbCursorScan %[c.ptr, e.id, key, keyLen, maxEntries, result, room]

end

end AlgorithmLib.Prog
