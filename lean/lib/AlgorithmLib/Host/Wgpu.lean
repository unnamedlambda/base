module
public import AlgorithmLib.Host.Ffi
meta import AlgorithmLib.Host.Ffi
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# wgpu, called directly

What each `WgpuFn` does to the objects the program has made (`World.wg`).

**Decoding, then acting.** Every call is read in two steps. `wgDecode` reads
the arguments, and the arrays and bytes they point at, and checks what WebGPU
requires of them: each handle names an object the program holds, of the kind
the parameter takes, in the state the call needs (an encoder with no pass open,
a pass not yet ended, a surface configured). What decodes is a `WgCall`, and
`wgEffect` carries it out and always answers. So whatever the model refuses, it
refuses while decoding, and the precondition of a call is that it decodes.

**Consuming.** Ending a pass, finishing an encoder, submitting a command
buffer, presenting a texture and popping an error scope use their object up:
the program no longer holds it, and releasing it again does not decode.

**What runs.** Commands wait in their encoder, and a submit runs them in
order: copies move bytes, a dispatch gives every buffer bound to it what the
world's `shader` oracle computes (a read-only binding keeps its own), and a
draw puts the buffer bound at index 0 on the texture it targets. A present
shows that buffer. Everything happens at the submit, which is as early as any
program can observe it: reading a buffer back waits for the queue.

**Errors.** A shader the device does not accept (`World.wgslOk`) makes an
invalid module and pipeline, and records an error in the innermost error scope
of the device, which the program pops by the scope's handle, innermost first.
An invalid object is not usable: using one is undefined here.

**Not modelled**: textures and samplers in bind groups, blending, depth and
stencil, a view other than a texture's default, and a buffer mapped at
creation, none of which the calls can ask for. Objects of different devices
are not told apart; running out of memory is not modelled.
-/

namespace AlgorithmLib.HProg.Sem

open AlgorithmLib.IR
open AlgorithmLib.HProg

/-- A decoded call: what to do to wgpu's objects. Each but a pop and a read
    may use objects up (`frees`). -/
inductive WgCall where
  /-- Make an object, after replacing others (an encoder a pass locks). -/
  | make (d : WgData) (changes : List (Nat × WgData)) (frees : List Nat := [])
  /-- Replace objects, answering nothing. -/
  | update (changes : List (Nat × WgData)) (frees : List Nat := [])
  | submit (cb : Nat)
  | write (buf off : Nat) (bytes : ByteArray)
  /-- `n` bytes of a buffer from `off`, to `dst`. -/
  | read (buf off n : Nat) (dst : UInt64)
  /-- Pop scope `sc` of device `dev`: whether it caught an error. -/
  | pop (dev sc : Nat) (caught : Bool) (rest : List Bool)
  | acquire (surf : Nat)
  | present (surf tex : Nat)
  /-- Answer `h`, changing nothing: no object to answer. -/
  | none_

-- ---------------------------------------------------------------------------
-- Reading arguments
-- ---------------------------------------------------------------------------

/-- The most entries an array argument may hold in the model. -/
def WG_MAX : Nat := 1024

/-- The `n` bytes at `a` as UTF-8. -/
def wgText (m : Mem) (a n : UInt64) : Option String := do
  String.fromUTF8? (← readBytes m a n.toNat)

/-- The `n` entries of an array at `a`, `n` at most `WG_MAX`. -/
def wgArray {α : Type} (a : UInt64) (n : UInt64) (stride : Nat) (f : UInt64 → Option α) :
    Option (List α) :=
  if n.toNat > WG_MAX then none
  else (List.range n.toNat).mapM fun k => f (a + UInt64.ofNat (stride * k))

/-- A `WGPUTextureFormat` the calls take: RGBA8 and BGRA8, linear or sRGB. -/
def wgFormatOk (f : UInt64) : Bool :=
  let f := f.toUInt32
  f == 0x16 || f == 0x17 || f == 0x1B || f == 0x1C

/-- A device's error scopes, the innermost one marked as having caught an
    error. With none pushed the error is uncaptured, which the model drops. -/
def markErr : List Bool → List Bool
  | [] => []
  | _ :: rest => true :: rest

def WgState.device? (s : WgState) (h : UInt64) : Option (Nat × List Bool) := do
  let (i, d) ← s.get? .device h
  match d with
  | .device sc => some (i, sc)
  | _ => none

/-- An error recorded on device `h` when `ok` is false. -/
def WgState.errUnless (s : WgState) (h : UInt64) (ok : Bool) : Option (List (Nat × WgData)) := do
  let (i, sc) ← s.device? h
  some (if ok then [] else [(i, .device (markErr sc))])

-- ---------------------------------------------------------------------------
-- Decoding, one function each
-- ---------------------------------------------------------------------------

section
variable (w : World)

def wgDRelease (o : WgObj) : List UInt64 → Option WgCall
  | [h] => do let (i, _) ← w.wg.get? o h; some (.update [] [i])
  | _ => none

/-- A high-performance adapter where the world has one (`wgAdapter`), and
    `0` where it has none. -/
def wgDRequestAdapter : List UInt64 → Option WgCall
  | [inst] => do
      let _ ← w.wg.get? .instance inst
      if w.wgAdapter then some (.make .adapter []) else some .none_
  | _ => none

def wgDRequestDevice : List UInt64 → Option WgCall
  | [a] => do let _ ← w.wg.get? .adapter a; some (.make (.device []) [])
  | _ => none

def wgDGetQueue : List UInt64 → Option WgCall
  | [d] => do let _ ← w.wg.device? d; some (.make .queue [])
  | _ => none

/-- Zeroed contents, as WebGPU initializes them. `MAP_READ` goes only with
    `COPY_DST`. -/
def wgDCreateBuffer : List UInt64 → Option WgCall
  | [d, usage, size] => do
      let _ ← w.wg.device? d
      if usage == 0 || usage ≥ 0x400 || (usage &&& 1 != 0 && usage &&& ~~~9 != 0)
          || size ≥ 0x8000000000000000 then none
      else some (.make (.buffer usage (ByteArray.mk (Array.replicate size.toNat 0))) [])
  | _ => none

/-- WGSL, `n` bytes of UTF-8 at `src`. -/
def wgDCreateShader : List UInt64 → Option WgCall
  | [d, src, n] => do
      let text ← wgText w.mem src n
      let ok := w.wgslOk text
      let ch ← w.wg.errUnless d ok
      some (.make (.shader text ok) ch)
  | _ => none

/-- `n` entries at `ents`, each a word of `(binding, read-only)` as two
    `u32`s: storage buffers, seen by compute. -/
def wgDCreateBgl : List UInt64 → Option WgCall
  | [d, ents, n] => do
      let _ ← w.wg.device? d
      let es ← wgArray ents n 8 fun a => do
        let b ← w.mem.load a 4
        let ro ← w.mem.load (a + 4) 4
        some (b.toNat, ro != 0)
      if (es.map (·.1)).Nodup then some (.make (.bgl (some es)) []) else none
  | _ => none

/-- `n` group layouts, a handle a word, at `arr`. -/
def wgDCreatePlayout : List UInt64 → Option WgCall
  | [d, arr, n] => do
      let _ ← w.wg.device? d
      let gs ← wgArray arr n 8 fun a => do
        let h ← w.mem.load a 8
        let (_, l) ← w.wg.get? .bindGroupLayout h
        match l with
        | .bgl (some es) => some es
        | _ => none
      some (.make (.playout gs) [])
  | _ => none

/-- An explicit layout, the entry point named by `n` bytes at `entry`; the
    pipeline is valid when its module is. -/
def wgDCreateCpipe : List UInt64 → Option WgCall
  | [d, lh, mh, entry, n] => do
      let (_, l) ← w.wg.get? .pipelineLayout lh
      let gs ← match l with | .playout gs => some gs | _ => none
      let (_, md) ← w.wg.get? .shaderModule mh
      let (src, ok) ← match md with | .shader src ok => some (src, ok) | _ => none
      let _ ← wgText w.mem entry n
      let ch ← w.wg.errUnless d ok
      some (.make (.cpipe src gs ok) ch)
  | _ => none

/-- A whole storage buffer bound at `binding`: an entry of two words. -/
def wgBgEntry (s : WgState) (m : Mem) (e : UInt64) : Option (Nat × Nat) := do
  let b ← m.load e 8
  let bh ← m.load (e + 8) 8
  let (bi, bd) ← s.get? .buffer bh
  let usage ← match bd with | .buffer u _ => some u | _ => none
  if usage &&& 128 == 0 || b ≥ 0x100000000 then none else some (b.toNat, bi)

/-- The bindings must be exactly the layout's, unless the layout is one a
    render pipeline derived. -/
def wgDCreateBgroup : List UInt64 → Option WgCall
  | [d, lh, ents, n] => do
      let _ ← w.wg.device? d
      let (_, l) ← w.wg.get? .bindGroupLayout lh
      let es ← wgArray ents n 16 (wgBgEntry w.wg w.mem)
      let fits : Bool := match l with
        | .bgl (some ls) => decide (es.map (·.1)).Nodup && es.length == ls.length &&
            es.all (fun (b, _) => ls.any (·.1 == b))
        | _ => decide (es.map (·.1)).Nodup
      if fits then some (.make (.bgroup es) []) else none
  | _ => none

def wgDCreateEncoder : List UInt64 → Option WgCall
  | [d] => do let _ ← w.wg.device? d; some (.make (.encoder [] true) [])
  | _ => none

/-- A layout derived from the shaders, the entry points named by the bytes
    at `vs` and `fs`, one color target of a format the calls take. -/
def wgDCreateRpipe : List UInt64 → Option WgCall
  | [d, mh, vs, vn, fs, fn, fmt] => do
      let (_, m) ← w.wg.get? .shaderModule mh
      let ok ← match m with | .shader _ ok => some ok | _ => none
      let _ ← wgText w.mem vs vn
      let _ ← wgText w.mem fs fn
      if !wgFormatOk fmt then none else
      let ch ← w.wg.errUnless d ok
      some (.make (.rpipe ok) ch)
  | _ => none

/-- Validation errors only: a scope, one deeper than the device's last. -/
def wgDPushScope : List UInt64 → Option WgCall
  | [d] => do
      let (i, sc) ← w.wg.device? d
      some (.make (.scope i (sc.length + 1)) [(i, .device (false :: sc))])
  | _ => none

/-- The innermost scope of its device. -/
def wgDPopScope : List UInt64 → Option WgCall
  | [h] => do
      let (si, sd) ← w.wg.get? .errorScope h
      let (dev, level) ← match sd with | .scope dev level => some (dev, level) | _ => none
      match w.wg.data? dev with
      | some (.device (e :: rest)) =>
          if rest.length + 1 == level then some (.pop dev si e rest) else none
      | _ => none
  | _ => none

def wgDRpipeBgl : List UInt64 → Option WgCall
  | [p, idx] => do
      let (_, pd) ← w.wg.get? .renderPipeline p
      match pd with
      | .rpipe true => if idx.toUInt32 == 0 then some (.make (.bgl none) []) else none
      | _ => none
  | _ => none

/-- The encoder is locked while the pass is open. -/
def wgDBeginCpass : List UInt64 → Option WgCall
  | [e] => do
      let (ei, ed) ← w.wg.get? .commandEncoder e
      let cmds ← match ed with | .encoder cmds true => some cmds | _ => none
      some (.make (.cpass ei none [] false) [(ei, .encoder cmds false)])
  | _ => none

def wgDCpassSetPipe : List UInt64 → Option WgCall
  | [p, q] => do
      let (pi, pd) ← w.wg.get? .computePass p
      let (qi, qd) ← w.wg.get? .computePipeline q
      match pd, qd with
      | .cpass e _ b false, .cpipe _ _ true => some (.update [(pi, .cpass e (some qi) b false)])
      | _, _ => none
  | _ => none

def wgDCpassSetGroup : List UInt64 → Option WgCall
  | [p, idx, g] => do
      let (pi, pd) ← w.wg.get? .computePass p
      let (gi, _) ← w.wg.get? .bindGroup g
      match pd with
      | .cpass e q b false =>
          let k := idx.toUInt32.toNat
          some (.update [(pi, .cpass e q ((k, gi) :: b.filter (·.1 != k)) false)])
      | _ => none
  | _ => none

/-- Every group the pipeline's layout names is bound, to a group with that
    layout's bindings; each count at most 65535. -/
def wgDCpassDispatch : List UInt64 → Option WgCall
  | [p, x, y, z] => do
      let (_, pd) ← w.wg.get? .computePass p
      let (e, q, b) ← match pd with | .cpass e (some q) b false => some (e, q, b) | _ => none
      let gs ← match w.wg.data? q with | some (.cpipe _ gs true) => some gs | _ => none
      let cmds ← match w.wg.data? e with | some (.encoder cmds false) => some cmds | _ => none
      let bound := gs.zipIdx.all fun (ls, k) =>
        match (b.lookup k).bind w.wg.data? with
        | some (.bgroup es) => es.length == ls.length && ls.all (fun (bn, _) => es.any (·.1 == bn))
        | _ => false
      let xyz := [x.toUInt32.toUInt64, y.toUInt32.toUInt64, z.toUInt32.toUInt64]
      if !bound || xyz.any (· > 65535) then none
      else some (.update [(e, .encoder (cmds ++ [.dispatch q b xyz]) false)])
  | _ => none

/-- The pass used up; its encoder unlocked. -/
def wgDCpassEnd : List UInt64 → Option WgCall
  | [p] => do
      let (pi, pd) ← w.wg.get? .computePass p
      match pd with
      | .cpass e _ _ false =>
          match w.wg.data? e with
          | some (.encoder cmds false) => some (.update [(e, .encoder cmds true)] [pi])
          | _ => none
      | _ => none
  | _ => none

/-- Between two buffers, the source `COPY_SRC` and the destination `COPY_DST`,
    at offsets and a size that are multiples of 4, inside both. -/
def wgDCopy : List UInt64 → Option WgCall
  | [e, src, so, dst, dof, n] => do
      let (ei, ed) ← w.wg.get? .commandEncoder e
      let cmds ← match ed with | .encoder cmds true => some cmds | _ => none
      let (si, sd) ← w.wg.get? .buffer src
      let (di, dd) ← w.wg.get? .buffer dst
      match sd, dd with
      | .buffer su sb, .buffer du db =>
          if su &&& 4 == 0 || du &&& 8 == 0 || si == di || so.toNat % 4 != 0 || dof.toNat % 4 != 0
              || n.toNat % 4 != 0 || so.toNat + n.toNat > sb.size || dof.toNat + n.toNat > db.size then none
          else some (.update [(ei, .encoder (cmds ++ [.copy si so.toNat di dof.toNat n.toNat]) true)])
      | _, _ => none
  | _ => none

/-- The encoder used up: its commands, as a command buffer. -/
def wgDFinish : List UInt64 → Option WgCall
  | [e] => do
      let (ei, ed) ← w.wg.get? .commandEncoder e
      match ed with
      | .encoder cmds true => some (.make (.cmdbuf cmds false) [(ei, .encoder cmds false)] [ei])
      | _ => none
  | _ => none

/-- One color attachment, a view, cleared or loaded, and stored. -/
def wgDBeginRpass : List UInt64 → Option WgCall
  | [e, vh, _] => do
      let (ei, ed) ← w.wg.get? .commandEncoder e
      let cmds ← match ed with | .encoder cmds true => some cmds | _ => none
      let (vi, _) ← w.wg.get? .textureView vh
      some (.make (.rpass ei vi none none false) [(ei, .encoder cmds false)])
  | _ => none

def wgDRpassSetPipe : List UInt64 → Option WgCall
  | [p, q] => do
      let (pi, pd) ← w.wg.get? .renderPass p
      let (qi, qd) ← w.wg.get? .renderPipeline q
      match pd, qd with
      | .rpass e v _ g false, .rpipe true => some (.update [(pi, .rpass e v (some qi) g false)])
      | _, _ => none
  | _ => none

def wgDRpassSetGroup : List UInt64 → Option WgCall
  | [p, idx, g] => do
      let (pi, pd) ← w.wg.get? .renderPass p
      let (gi, _) ← w.wg.get? .bindGroup g
      if idx.toUInt32 != 0 then none else
      match pd with
      | .rpass e v q _ false => some (.update [(pi, .rpass e v q (some gi) false)])
      | _ => none
  | _ => none

def wgDRpassDraw : List UInt64 → Option WgCall
  | [p, _, _, _, _] => do
      let (_, pd) ← w.wg.get? .renderPass p
      match pd with
      | .rpass e v (some _) g false =>
          match w.wg.data? e with
          | some (.encoder cmds false) => some (.update [(e, .encoder (cmds ++ [.draw v g]) false)])
          | _ => none
      | _ => none
  | _ => none

/-- The pass used up; its encoder unlocked. -/
def wgDRpassEnd : List UInt64 → Option WgCall
  | [p] => do
      let (pi, pd) ← w.wg.get? .renderPass p
      match pd with
      | .rpass e _ _ _ false =>
          match w.wg.data? e with
          | some (.encoder cmds false) => some (.update [(e, .encoder cmds true)] [pi])
          | _ => none
      | _ => none
  | _ => none

/-- A command buffer not yet submitted, used up. -/
def wgDSubmit : List UInt64 → Option WgCall
  | [q, cb] => do
      let _ ← w.wg.get? .queue q
      let (ci, cd) ← w.wg.get? .commandBuffer cb
      match cd with
      | .cmdbuf _ false => some (.submit ci)
      | _ => none
  | _ => none

/-- Into a `COPY_DST` buffer, at an offset and a size that are multiples of 4,
    inside it. -/
def wgDWriteBuffer : List UInt64 → Option WgCall
  | [q, b, off, src, n] => do
      let _ ← w.wg.get? .queue q
      let (bi, bd) ← w.wg.get? .buffer b
      match bd with
      | .buffer u bs =>
          if u &&& 8 == 0 || off.toNat % 4 != 0 || n.toNat % 4 != 0 || off.toNat + n.toNat > bs.size then none
          else do
            let bytes ← readBytes w.mem src n.toNat
            some (.write bi off.toNat bytes)
      | _ => none
  | _ => none

/-- `n` bytes of a `MAP_READ` buffer from `off`, to `dst`: an offset a
    multiple of 8, a size a positive multiple of 4, inside the buffer, and
    room at `dst`, as wgpu maps a range. -/
def wgDRead : List UInt64 → Option WgCall
  | [d, b, off, dst, n] => do
      let _ ← w.wg.device? d
      let (bi, bd) ← w.wg.get? .buffer b
      match bd with
      | .buffer u bs =>
          if u &&& 1 == 0 || off.toNat % 8 != 0 || n == 0 || n.toNat % 4 != 0
              || off.toNat + n.toNat > bs.size then none
          else do
            let _ ← copyIn w.mem dst (ByteArray.mk (Array.replicate n.toNat 0))
            some (.read bi off.toNat n.toNat dst)
      | _ => none
  | _ => none

/-- Of a window the window library holds open. -/
def wgDCreateSurface : List UInt64 → Option WgCall
  | [inst, win] => do
      let _ ← w.wg.get? .instance inst
      let open_ := (List.range w.wl.windows.size).any fun k =>
        wlWinAddr k == win && (w.wl.windows[k]?.map (·.live)).getD false
      if open_ then some (.make (.surface false none) []) else none
  | _ => none

/-- On a device, in a format the calls take, at a nonzero size, while no
    texture of it is outstanding. -/
def wgDConfigure : List UInt64 → Option WgCall
  | [sh, dh, fmt, wd, ht] => do
      let (si, sd) ← w.wg.get? .surface sh
      let _ ← w.wg.device? dh
      match sd with
      | .surface _ none =>
          if !wgFormatOk fmt || asI32 wd ≤ 0 || asI32 ht ≤ 0 then none
          else some (.update [(si, .surface true none)])
      | _ => none
  | _ => none

/-- The surface configured, its last texture presented. -/
def wgDAcquire : List UInt64 → Option WgCall
  | [sh] => do
      let (si, sd) ← w.wg.get? .surface sh
      match sd with
      | .surface true none => some (.acquire si)
      | _ => none
  | _ => none

/-- The texture its surface holds, used up. -/
def wgDPresent : List UInt64 → Option WgCall
  | [th] => do
      let (ti, td) ← w.wg.get? .texture th
      let si ← match td with | .texture si _ => some si | _ => none
      match w.wg.data? si with
      | some (.surface true (some t)) => if t == ti then some (.present si ti) else none
      | _ => none
  | _ => none

/-- The texture's default view. -/
def wgDCreateView : List UInt64 → Option WgCall
  | [t] => do
      let (ti, _) ← w.wg.get? .texture t
      some (.make (.view ti) [])
  | _ => none

end

/-- **What a call decodes to**, or `none` where WebGPU does not define it or
    the model does not state it. -/
def wgDecode (f : WgpuFn) (bits : List UInt64) (w : World) : Option WgCall :=
  match f with
  | .createInstance => if bits.isEmpty then some (.make .instance []) else none
  | .requestAdapter => wgDRequestAdapter w bits
  | .requestDevice => wgDRequestDevice w bits
  | .release o => wgDRelease w o bits
  | .deviceGetQueue => wgDGetQueue w bits
  | .deviceCreateBuffer => wgDCreateBuffer w bits
  | .deviceCreateShaderModule => wgDCreateShader w bits
  | .deviceCreateBindGroupLayout => wgDCreateBgl w bits
  | .deviceCreatePipelineLayout => wgDCreatePlayout w bits
  | .deviceCreateComputePipeline => wgDCreateCpipe w bits
  | .deviceCreateBindGroup => wgDCreateBgroup w bits
  | .deviceCreateCommandEncoder => wgDCreateEncoder w bits
  | .deviceCreateRenderPipeline => wgDCreateRpipe w bits
  | .devicePushErrorScope => wgDPushScope w bits
  | .errorScopePop => wgDPopScope w bits
  | .renderPipelineGetBindGroupLayout => wgDRpipeBgl w bits
  | .encoderBeginComputePass => wgDBeginCpass w bits
  | .computePassSetPipeline => wgDCpassSetPipe w bits
  | .computePassSetBindGroup => wgDCpassSetGroup w bits
  | .computePassDispatch => wgDCpassDispatch w bits
  | .computePassEnd => wgDCpassEnd w bits
  | .encoderCopyBufferToBuffer => wgDCopy w bits
  | .encoderFinish => wgDFinish w bits
  | .encoderBeginRenderPass => wgDBeginRpass w bits
  | .renderPassSetPipeline => wgDRpassSetPipe w bits
  | .renderPassSetBindGroup => wgDRpassSetGroup w bits
  | .renderPassDraw => wgDRpassDraw w bits
  | .renderPassEnd => wgDRpassEnd w bits
  | .queueSubmit => wgDSubmit w bits
  | .queueWriteBuffer => wgDWriteBuffer w bits
  | .bufferRead => wgDRead w bits
  | .instanceCreateSurface => wgDCreateSurface w bits
  | .surfaceConfigure => wgDConfigure w bits
  | .surfaceGetCurrentTexture => wgDAcquire w bits
  | .surfacePresent => wgDPresent w bits
  | .textureCreateView => wgDCreateView w bits

-- ---------------------------------------------------------------------------
-- Acting
-- ---------------------------------------------------------------------------

def WgState.setAll (s : WgState) (ch : List (Nat × WgData)) : WgState :=
  ch.foldl (fun s (i, d) => s.set i d) s

/-- Objects the program no longer holds. -/
def WgState.freeAll (s : WgState) (is : List Nat) : WgState :=
  is.foldl (fun s i => { objs := s.objs.modify i fun (_, d) => (false, d) }) s

def WgState.bytes (s : WgState) (i : Nat) : ByteArray :=
  match s.data? i with
  | some (.buffer _ bs) => bs
  | _ => ByteArray.empty

def WgState.putBytes (s : WgState) (i : Nat) (bs : ByteArray) : WgState :=
  match s.data? i with
  | some (.buffer u _) => s.set i (.buffer u bs)
  | _ => s

/-- The buffers a dispatch binds, in its layout's order: `(buffer, read-only)`. -/
def WgState.binds (s : WgState) (gs : List (List (Nat × Bool))) (b : List (Nat × Nat)) :
    List (Nat × Bool) :=
  gs.zipIdx.flatMap fun (ls, k) =>
    match (b.lookup k).bind s.data? with
    | some (.bgroup es) => ls.filterMap fun (bn, ro) => (es.lookup bn).map (·, ro)
    | _ => []

/-- One command, run. -/
def runWgCmd (sh : Dispatch → List ByteArray → List ByteArray) (s : WgState) : WgCmd → WgState
  | .copy src so dst dof n =>
      s.putBytes dst (overwrite (s.bytes dst) dof ((s.bytes src).extract so (so + n)))
  | .dispatch q b xyz =>
      match s.data? q with
      | some (.cpipe src gs _) =>
          let bs := s.binds gs b
          let ins := bs.map fun (i, _) => s.bytes i
          let outs := sh ⟨src, bs, xyz⟩ ins
          bs.zipIdx.foldl (fun s ((i, ro), k) =>
            if ro then s
            else s.putBytes i (exactly (outs.getD k ByteArray.empty) (ins.getD k ByteArray.empty).size)) s
      | _ => s
  | .draw v g =>
      match s.data? v, g.bind s.data? with
      | some (.view t), some (.bgroup es) =>
          match s.data? t, es.lookup 0 with
          | some (.texture si _), some bi => s.set t (.texture si (some (s.bytes bi)))
          | _, _ => s
      | _, _ => s

/-- A command buffer run, and used up. -/
def WgState.run (sh : Dispatch → List ByteArray → List ByteArray) (s : WgState) (ci : Nat) : WgState :=
  match s.data? ci with
  | some (.cmdbuf cmds _) => ((cmds.foldl (runWgCmd sh) s).set ci (.cmdbuf cmds true)).freeAll [ci]
  | _ => s

/-- **What a decoded call does to wgpu.** It always answers, and it leaves
    program memory alone: the one call that writes it, a buffer read, does so
    in `wgpuCall`, at the address it was handed. -/
def wgEffect (c : WgCall) (w : World) : Option V × World :=
  match c with
  | .make d ch fr =>
      let (h, s) := ((w.wg.setAll ch).freeAll fr).push d
      (some (.sc .i64 h), { w with wg := s })
  | .update ch fr => (none, { w with wg := (w.wg.setAll ch).freeAll fr })
  | .submit ci => (none, { w with wg := w.wg.run w.shader ci })
  | .write i off bs => (none, { w with wg := w.wg.putBytes i (overwrite (w.wg.bytes i) off bs) })
  | .read _ _ _ _ => (some (ofInt .i32 0), w)
  | .pop dev si e rest =>
      (some (ofInt .i32 (if e then -1 else 0)),
        { w with wg := (w.wg.set dev (.device rest)).freeAll [si] })
  | .acquire si =>
      let (h, s) := w.wg.push (.texture si none)
      (some (.sc .i64 h), { w with wg := s.set si (.surface true (some w.wg.objs.size)) })
  | .present si t =>
      let shown := match w.wg.data? t with
        | some (.texture _ (some bs)) => [bs]
        | _ => []
      (none, { w with wg := (w.wg.set si (.surface true none)).freeAll [t], shown := w.shown ++ shown })
  | .none_ => (some (.sc .i64 0), w)

/-- **What a wgpu call does**: decoded, then carried out; a buffer read's
    bytes land where the caller asked. -/
def wgpuCall (f : WgpuFn) (bits : List UInt64) (w : World) : Option (Option V × World) :=
  (wgDecode f bits w).map fun c =>
    let e := wgEffect c w
    match f, bits with
    | .bufferRead, [_, b, off, dst, n] =>
        let bytes := match w.wg.get? .buffer b with
          | some (bi, _) => (w.wg.bytes bi).extract off.toNat (off.toNat + n.toNat)
          | none => ByteArray.empty
        (e.1, { e.2 with mem := (copyIn w.mem dst bytes).getD w.mem })
    | _, _ => e

theorem wgEffect_mem (c : WgCall) (w : World) : (wgEffect c w).2.mem = w.mem := by
  cases c <;> rfl

end AlgorithmLib.HProg.Sem
