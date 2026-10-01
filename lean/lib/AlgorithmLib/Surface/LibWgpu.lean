module
public import AlgorithmLib.Surface.LibBase
meta import AlgorithmLib.Surface.LibBase
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lib.Wgpu` — the engine's wgpu entry points, as CLIF over `Ext.wgpu`

The `gpu*` entry points, stated in `Host.Ffi` — buffers and compute pipelines
by `i32` id, a dispatch that waits until the next dispatch or read, and a read
that waits for the queue — as functions of the program itself, over wgpu
called directly (`Ext.wgpu`).

**The state.** `gpuInit` allocates one block and writes its address where the
engine wrote its context pointer. The block holds the instance, adapter,
device and queue; a table of buffers, each with the staging buffer a read maps;
a table of pipelines, each with its bind group; the encoder the last dispatch
was recorded in; and scratch for the one handle a pipeline layout is made of.
A table doubles when full, and an id is an index into it.

**Errors.** A pipeline is made inside an error scope, so a shader wgpu rejects
answers `-1` and leaves nothing behind. An error outside a scope — a copy or
write the program should not have made — is reported by the engine and the
program goes on.

**Without wgpu or a GPU** `gpuInit` stores null, and every call on that context
answers `-1`, as on any null context.

**Limits.** One thread per context: the state is not locked.
-/

namespace AlgorithmLib.LibWgpu

open AlgorithmLib.IR
open AlgorithmLib.Prog
open AlgorithmLib.Lib

variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

def INST : Nat := 0x00
def ADAPTER : Nat := 0x08
def DEVICE : Nat := 0x10
def QUEUE : Nat := 0x18
/-- The buffers — array, length, capacity; an entry is the buffer, its
    staging buffer and its size. -/
def BUFS : Nat := 0x20
def BUF_W : Nat := 24
/-- The pipelines; an entry is the pipeline and its bind group. -/
def PIPES : Nat := 0x38
def PIPE_W : Nat := 16
/-- The encoder the last dispatch was recorded in, `0` when none waits. -/
def PENDING : Nat := 0x50
/-- `"main"`. -/
def MAIN : Nat := 0x58
def SCRATCH : Nat := 0x60
def SIZE : Nat := 0x68

/-- `WGPUBufferUsage`: storage, copied to and from; and mapped for reading,
    copied to. -/
def USAGE_STORAGE : Int := 128 + 8 + 4
def USAGE_STAGING : Int := 1 + 8

def wg (f : WgpuFn) (args : Vals V (Ext.wgpu f).sig.1) : Prog V L (ResV V (Ext.wgpu f).sig.2) :=
  ext (.wgpu f) args

def rel (o : WgObj) (h : V .i64) : Prog V L Unit := do
  let _ ← wg (.release o) %[h]
  pure ()

-- ---------------------------------------------------------------------------
-- Tables
-- ---------------------------------------------------------------------------

/-- The address of entry `id` of the table at `t`, or `0` when `id` is
    negative or past the end. -/
def entry (s : V .i64) (t width : Nat) (id : V .i32) : Prog V L (V .i64) := do
  let i ← sextend64 id
  let len ← load64 (← at_ s (t + 8))
  -- a negative id is past every length, compared unsigned
  let r ← ifte (jTys := [.i64]) .uge i len (do pure %[← iconst64 0]) (do
    pure %[← iadd (← load64 (← at_ s t)) (← imul i (← iconst64 width))])
  pure r.head

/-- Append an entry to the table at `t`: its index, or `-1` when the table
    could not grow. -/
def push (s : V .i64) (t width : Nat) : Prog V L (V .i64) := do
  let lenA ← at_ s (t + 8)
  let capA ← at_ s (t + 16)
  let len ← load64 lenA
  let cap ← load64 capA
  let z ← iconst64 0
  let room ← ifte (jTys := [.i64]) .ult len cap (do pure %[← iconst64 1]) (do
    let nc ← select (← icmp .eq cap z) (← iconst64 8) (← ishlImm cap 1)
    let na ← calloc (← imul nc (← iconst64 width))
    let r ← ifte (jTys := [.i64]) .eq na z (do pure %[z]) (do
      let old ← load64 (← at_ s t)
      when .ne old z (do
        let _ ← memcpy na old (← imul cap (← iconst64 width))
        free old)
      storeI64 na (← at_ s t)
      storeI64 nc capA
      pure %[← iconst64 1])
    pure %[r.head])
  let r ← ifte (jTys := [.i64]) .eq room.head z (do pure %[← iconst64 (-1)]) (do
    storeI64 (← iaddImm len 1) lenA
    pure %[len])
  pure r.head

def entryAt (s : V .i64) (t width : Nat) (i : V .i64) : Prog V L (V .i64) := do
  iadd (← load64 (← at_ s t)) (← imul i (← iconst64 width))

-- ---------------------------------------------------------------------------
-- The queue
-- ---------------------------------------------------------------------------

/-- Submit the encoder a dispatch waits in, if one does. -/
def flush (s : V .i64) : Prog V L Unit := do
  let pa ← at_ s PENDING
  let enc ← load64 pa
  let z ← iconst64 0
  when .ne enc z (do
    let cb ← wg .encoderFinish %[enc]
    let _ ← wg .queueSubmit %[← load64 (← at_ s QUEUE), cb]
    storeI64 z pa)

-- ---------------------------------------------------------------------------
-- The entry points
-- ---------------------------------------------------------------------------

/-- An instance, its high-performance adapter, a device and its queue; null
    without wgpu or a GPU. -/
def init : Body := do
  let (slot ::ᵥ .nil) ← entryParams [.i64]
  let z ← iconst64 0
  storeI64 z slot
  when .ne (← libPresent .wgpu) (← iconst32 0) (do
    let inst ← wg .createInstance %[]
    let ad ← wg .requestAdapter %[inst]
    let _ ← ifte (jTys := []) .eq ad z (do rel .instance inst; pure %[]) (do
      let dev ← wg .requestDevice %[ad]
      let _ ← ifte (jTys := []) .eq dev z (do rel .adapter ad; rel .instance inst; pure %[]) (do
        let s ← calloc (← iconst64 SIZE)
        let _ ← ifte (jTys := []) .eq s z
          (do rel .device dev; rel .adapter ad; rel .instance inst; pure %[]) (do
          storeI64 inst (← at_ s INST)
          storeI64 ad (← at_ s ADAPTER)
          storeI64 dev (← at_ s DEVICE)
          storeI64 (← wg .deviceGetQueue %[dev]) (← at_ s QUEUE)
          -- "main"
          storeI32 (← iconst32 0x6e69616d) (← at_ s MAIN)
          storeI64 s slot
          pure %[])
        pure %[])
      pure %[])
    pure ())

/-- A zeroed buffer of `size` bytes: its id, or `-1`. -/
def createBuffer : StatusBody := do
  let (s ::ᵥ size ::ᵥ .nil) ← entryParams [.i64, .i64]
  let z ← iconst64 0
  failIf .sle size z (do
    failIf .eq s z (do
      let dev ← load64 (← at_ s DEVICE)
      let b ← wg .deviceCreateBuffer %[dev, ← iconst64 USAGE_STORAGE, size]
      let st ← wg .deviceCreateBuffer %[dev, ← iconst64 USAGE_STAGING, size]
      let i ← push s BUFS BUF_W
      let r ← ifte (jTys := [.i64]) .eq i (← iconst64 (-1))
        (do rel .buffer b; rel .buffer st; pure %[← iconst64 (-1)]) (do
        let e ← entryAt s BUFS BUF_W i
        storeI64 b e
        storeI64 st (← at_ e 8)
        storeI64 size (← at_ e 16)
        pure %[i])
      pure r.head))

/-- A compute pipeline of the NUL-terminated WGSL at `shader`, entry point
    `main`, with `n` bindings, each an `i32` buffer id and an `i32` read-only
    flag: its id, or `-1` when a binding names no buffer or wgpu rejects the
    shader. -/
def createPipeline : StatusBody := do
  let (s ::ᵥ shader ::ᵥ binds ::ᵥ n ::ᵥ .nil) ← entryParams [.i64, .i64, .i64, .i32]
  let z ← iconst64 0
  failIf .slt n (← iconst32 0) (do
    failIf .eq s z (do
      let n64 ← sextend64 n
      let nb ← load64 (← at_ s (BUFS + 8))
      let bad ← forLoopAcc n64 z fun i acc => do
        let id ← sextend64 (← load32 (← iadd binds (← ishlImm i 3)))
        -- a negative id is past every length, compared unsigned
        bor acc (← uextend64 (← icmp .uge id nb))
      failIf .ne bad z (do
        let dev ← load64 (← at_ s DEVICE)
        let scope ← wg .devicePushErrorScope %[dev]
        let module ← wg .deviceCreateShaderModule %[dev, shader, ← strlen shader]
        -- the layout's entries, a word each: the binding, and whether it is
        -- read-only
        let ents ← calloc (← imul (← iaddImm n64 1) (← iconst64 16))
        forLoop n64 fun i => do
          let e ← iadd ents (← ishlImm i 3)
          storeI32 (← ireduce32 i) e
          let ro ← load32 (← iadd binds (← iaddImm (← ishlImm i 3) 4))
          storeI32 (← select (← icmp .ne ro (← iconst32 0)) (← iconst32 1) (← iconst32 0)) (← at_ e 4)
        let bgl ← wg .deviceCreateBindGroupLayout %[dev, ents, n64]
        let slot ← at_ s SCRATCH
        storeI64 bgl slot
        let pl ← wg .deviceCreatePipelineLayout %[dev, slot, ← iconst64 1]
        let pipe ← wg .deviceCreateComputePipeline %[dev, pl, module, ← at_ s MAIN, ← iconst64 4]
        -- the group's entries, two words each: the binding and the buffer
        forLoop n64 fun i => do
          let e ← iadd ents (← ishlImm i 4)
          storeI64 i e
          let id ← sextend64 (← load32 (← iadd binds (← ishlImm i 3)))
          storeI64 (← load64 (← entryAt s BUFS BUF_W id)) (← at_ e 8)
        let bg ← wg .deviceCreateBindGroup %[dev, bgl, ents, n64]
        free ents
        rel .shaderModule module
        rel .bindGroupLayout bgl
        rel .pipelineLayout pl
        let err ← wg .errorScopePop %[scope]
        let r ← ifte (jTys := [.i64]) .ne err (← iconst32 0)
          (do rel .computePipeline pipe; rel .bindGroup bg; pure %[← iconst64 (-1)]) (do
          let i ← push s PIPES PIPE_W
          let r ← ifte (jTys := [.i64]) .eq i (← iconst64 (-1))
            (do rel .computePipeline pipe; rel .bindGroup bg; pure %[i]) (do
            let e ← entryAt s PIPES PIPE_W i
            storeI64 pipe e
            storeI64 bg (← at_ e 8)
            pure %[i])
          pure %[r.head])
        pure r.head)))

/-- Whether `size` is not a multiple of four. -/
def unaligned (size : V .i64) : Prog V L (V .i8) := do
  icmp .ne (← band size (← iconst64 3)) (← iconst64 0)

/-- `size` bytes from `src` into buffer `buf`, landing before any later
    submit: `0`, or `-1` — also for a size past the buffer or not a multiple
    of four, which wgpu would reject outside any error scope. -/
def upload : StatusBody := do
  let (s ::ᵥ buf ::ᵥ src ::ᵥ size ::ᵥ .nil) ← entryParams [.i64, .i32, .i64, .i64]
  let z ← iconst64 0
  let bad ← or8 (← anyNeg32 [buf]) (← or8 (← icmp .sle size z) (← icmp .eq src z))
  failUnless0 bad (do
    failIf .eq s z (do
      let e ← entry s BUFS BUF_W buf
      failIf .eq e z (do
        let fit ← or8 (← icmp .ugt size (← load64 (← at_ e 16))) (← unaligned size)
        failUnless0 fit (do
          let _ ← wg .queueWriteBuffer %[← load64 (← at_ s QUEUE), ← load64 e, z, src, size]
          pure z))))

/-- Record a dispatch of pipeline `pipe` over `x × y × z` workgroups, after
    submitting the one before it. -/
def dispatch : StatusBody := do
  let (s ::ᵥ pipe ::ᵥ gx ::ᵥ gy ::ᵥ gz ::ᵥ .nil) ← entryParams [.i64, .i32, .i32, .i32, .i32]
  let z ← iconst64 0
  failUnless0 (← or8 (← anyNeg32 [pipe]) (← anyNonPos [gx, gy, gz])) (do
    failIf .eq s z (do
      let e ← entry s PIPES PIPE_W pipe
      failIf .eq e z (do
        flush s
        let enc ← wg .deviceCreateCommandEncoder %[← load64 (← at_ s DEVICE)]
        let pass ← wg .encoderBeginComputePass %[enc]
        let _ ← wg .computePassSetPipeline %[pass, ← load64 e]
        let _ ← wg .computePassSetBindGroup %[pass, ← iconst32 0, ← load64 (← at_ e 8)]
        let _ ← wg .computePassDispatch %[pass, gx, gy, gz]
        let _ ← wg .computePassEnd %[pass]
        storeI64 enc (← at_ s PENDING)
        pure z)))

/-- Copy `size` bytes of the buffer at entry `e` from `off` to its staging
    buffer, submit that behind whatever waits, and read them back to `dst`. -/
def readBack (s e off dst size : V .i64) : Prog V L (V .i64) := do
  let z ← iconst64 0
  let pa ← at_ s PENDING
  let p ← load64 pa
  let dev ← load64 (← at_ s DEVICE)
  let enc ← ifte (jTys := [.i64]) .eq p z
    (do pure %[← wg .deviceCreateCommandEncoder %[dev]]) (do pure %[p])
  let enc := enc.head
  let st ← load64 (← at_ e 8)
  let _ ← wg .encoderCopyBufferToBuffer %[enc, ← load64 e, off, st, z, size]
  storeI64 enc pa
  flush s
  sextend64 (← wg .bufferRead %[dev, st, z, dst, size])

/-- The whole of buffer `buf`, `size` bytes, to `dst`: `0`, or `-1` — also
    when `size` is not the buffer's, or not a multiple of four. -/
def download : StatusBody := do
  let (s ::ᵥ buf ::ᵥ dst ::ᵥ size ::ᵥ .nil) ← entryParams [.i64, .i32, .i64, .i64]
  let z ← iconst64 0
  let bad ← or8 (← anyNeg32 [buf]) (← or8 (← icmp .sle size z) (← icmp .eq dst z))
  failUnless0 bad (do
    failIf .eq s z (do
      let e ← entry s BUFS BUF_W buf
      failIf .eq e z (do
        let fit ← or8 (← icmp .ne size (← load64 (← at_ e 16))) (← unaligned size)
        failUnless0 fit (readBack s e z dst size))))

/-- `size` bytes of buffer `buf` from `off` to `dst`: `0`, or `-1` — also
    when they run past the buffer, or `off` or `size` is not a multiple of
    four. Both are below `2^63`, so their sum does not wrap. -/
def downloadPtr : StatusBody := do
  let (s ::ᵥ buf ::ᵥ off ::ᵥ dst ::ᵥ size ::ᵥ .nil) ← entryParams [.i64, .i32, .i64, .i64, .i64]
  let z ← iconst64 0
  let bad ← or8 (← or8 (← anyNeg32 [buf]) (← anyNeg64 [off]))
    (← or8 (← icmp .sle size z) (← icmp .eq dst z))
  failUnless0 bad (do
    failIf .eq s z (do
      let e ← entry s BUFS BUF_W buf
      failIf .eq e z (do
        let past ← icmp .ugt (← iadd off size) (← load64 (← at_ e 16))
        let fit ← or8 past (← or8 (← unaligned off) (← unaligned size))
        failUnless0 fit (readBack s e off dst size))))

/-- Everything the context made, released, and the context with it. Work
    recorded and not submitted is dropped. -/
def cleanup : Body := do
  let (slot ::ᵥ .nil) ← entryParams [.i64]
  let s ← load64 slot
  let z ← iconst64 0
  when .ne s z (do
    let p ← load64 (← at_ s PENDING)
    when .ne p z (rel .commandEncoder p)
    forLoop (← load64 (← at_ s (BUFS + 8))) fun i => do
      let e ← entryAt s BUFS BUF_W i
      rel .buffer (← load64 e)
      rel .buffer (← load64 (← at_ e 8))
    free (← load64 (← at_ s BUFS))
    forLoop (← load64 (← at_ s (PIPES + 8))) fun i => do
      let e ← entryAt s PIPES PIPE_W i
      rel .computePipeline (← load64 e)
      rel .bindGroup (← load64 (← at_ e 8))
    free (← load64 (← at_ s PIPES))
    rel .queue (← load64 (← at_ s QUEUE))
    rel .device (← load64 (← at_ s DEVICE))
    rel .adapter (← load64 (← at_ s ADAPTER))
    rel .instance (← load64 (← at_ s INST))
    free s)
  storeI64 z slot

/-- The function implementing a wgpu entry point. -/
def implOf : Ffi → Option Impl
  | .gpuInit => some (.void init)
  | .gpuCreateBuffer => some (.status createBuffer)
  | .gpuCreatePipeline => some (.status createPipeline)
  | .gpuUpload | .gpuUploadPtr => some (.status upload)
  | .gpuDispatch => some (.status dispatch)
  | .gpuDownload => some (.status download)
  | .gpuDownloadPtr => some (.status downloadPtr)
  | .gpuCleanup => some (.void cleanup)
  | _ => none

end AlgorithmLib.LibWgpu
