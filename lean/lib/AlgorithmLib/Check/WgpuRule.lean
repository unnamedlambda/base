module
public import AlgorithmLib.Check.CudaRule
meta import AlgorithmLib.Check.CudaRule
import all Init.Data.Repr
@[expose] public section

/-!
# `Check.WgpuRule` — wgpu, called directly

What the checker takes of a call into wgpu (`Ext.wgpu`): whether the call
decodes (`Sem.wgDecode`), read off the typestate, and what it leaves of it.

**Objects made once** — the instance, adapter, device and queue, buffers,
shaders, layouts, pipelines, bind groups, a surface — are named by index
(`wgo`), and the typestate counts what wgpu has made (`nwg`), so the handle of
the next is a number the checker knows.

**Objects made afresh** each time round a loop — an encoder, a pass, a
command buffer, a surface's texture and its view — are the one live object of
their kind (`wg1`), and a handle to one is a value the checker describes as
that one (`AV.hand (.wg k)`). A typestate need not count them, so a loop that
makes and uses them up goes round with the same facts.

**A lost device.** Once wgpu's handles run out (`WG_OBJS`) the device is lost
and every call does nothing, so every fact about wgpu's objects holds and every
call is defined: the rows read the typestate as of a device that is not.

**Shaders.** A shader module is made only from text the typestate says the
device accepts (`wgText … true`): what a theorem states of its platform.

A call this file has no row for is refused, an error scope's among them.
-/

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace AlgorithmLib.Check

open AlgorithmLib.HProg.Contracts (Fact TState)

/-- How many objects the typestate says wgpu has made. -/
def nwgOf (S : TState) : Option Nat :=
  S.findSome? fun | .nwg n => some n | _ => none

/-- The shape the typestate gives live object `i`. -/
def wgoOf (S : TState) (i : Nat) : Option WgShape :=
  S.findSome? fun | .wgo i' s => if i' == i then some s else none | _ => none

/-- What the typestate says of the live objects of kind `k`: `some none` that
    there are none, `some (some s)` that there is at most one, of shape `s`. -/
def wg1Of (S : TState) (k : WgObj) : Option (Option WgShape) :=
  S.findSome? fun | .wg1 k' s => if decide (k' = k) then some s else none | _ => none

/-- An object of kind `k` made once, by its handle's bits: its index and
    shape. -/
def wgoAt (S : TState) (h : UInt64) (k : WgObj) : Option (Nat × WgShape) := do
  let i ← wgIndex h
  let s ← wgoOf S i
  if decide (s.kind = k) then some (i, s) else none

/-- An object of kind `k` made once, by a value. -/
def wgoArg (S : TState) (a : AV) (k : WgObj) : Option (Nat × WgShape) := do
  wgoAt S (← a.bits) k

/-- The one object of kind `k`, by a value the checker describes as that one:
    its shape. -/
def wg1Arg (S : TState) (a : AV) (k : WgObj) : Option WgShape :=
  match a, wg1Of S k with
  | .hand (.wg k') true, some (some s) => if decide (k' = k) then some s else none
  | _, _ => none

/-- A device, by a value: its index and error scopes. -/
def wgDev (S : TState) (a : AV) : Option (Nat × List Bool) :=
  match wgoArg S a .device with
  | some (i, .device sc) => some (i, sc)
  | _ => none

/-- Whether the typestate says the `n` bytes at `a` are UTF-8. -/
def wgTextAt (S : TState) (a : UInt64) (n : Nat) : Bool :=
  S.any fun | .wgText a' n' _ => a' == a && n' == n | _ => false

/-- **Whether a fact survives a wgpu step** that touches the objects at
    indices `Ti` and those of kinds `TK`, changes which objects of kinds `K`
    are live, and adds `g` objects (`none`: a number not known). -/
def wgKeeps (Ti : List Nat) (TK K : List WgObj) (g : Option Nat) : Fact → Bool
  | .nwg _ => g == some 0
  | .wgo i sh => !Ti.contains i && !TK.any (fun t => decide (t = sh.kind)) &&
      sh.reads.all (fun r => !K.any (fun k => decide (k = r)))
  | .wg1 k none => !K.any (fun k' => decide (k' = k))
  | .wg1 k (some sh) => !K.any (fun k' => decide (k' = k)) && !TK.any (fun t => decide (t = k)) &&
      sh.reads.all (fun r => !K.any (fun k' => decide (k' = r)))
  | .held (.wg k) _ | .opened (.wg k) _ => !K.any (fun k' => decide (k' = k))
  | _ => true

/-- **An object made once**, of shape `sh`: at the count the typestate keeps,
    so its handle is known. -/
def wgMake (S : TState) (sh : WgShape) : Option (TState × AV) := do
  let n ← nwgOf S
  if n < WG_OBJS then
    some (.nwg (n + 1) :: .wgo n sh :: S.filter (wgKeeps [] [] [sh.kind] (some 1)),
      .known (.sc .i64 (wgHandle n)))
  else none

/-- The count after a call that makes one object, where the typestate knows
    the count before: an object made afresh each time round a loop is counted
    too, so a program may make objects once after making some afresh. -/
def wgBump (S : TState) : TState :=
  match nwgOf S with
  | some n => [.nwg (n + 1)]
  | none => []

/-- What a step that changes nothing the typestate keeps of wgpu leaves. -/
def wgSame (S : TState) : TState := S.filter (wgKeeps [] [] [] (some 0))

/-- The `n` entries of an array at `a`, each read off the typestate's cells
    by `f` from its address. -/
def wgCells {α : Type} (a n : UInt64) (stride : Nat) (f : UInt64 → Option α) : Option (List α) :=
  if n.toNat > WG_MAX then none
  else (List.range n.toNat).mapM fun k => f (a + UInt64.ofNat (stride * k))

/-- A bind group layout's entries, as the typestate's cells hold them. -/
def wgBglEntries (S : TState) (ents n : UInt64) : Option (List (Nat × Bool)) :=
  wgCells ents n 8 fun a => do
    let e ← cellOf S a
    some (e.toUInt32.toNat, (e >>> 32).toUInt32 != 0)

/-- A pipeline layout's groups, as the typestate's cells name layouts. -/
def wgGroups (S : TState) (arr n : UInt64) : Option (List (List (Nat × Bool))) :=
  wgCells arr n 8 fun a => do
    let h ← cellOf S a
    match wgoAt S h .bindGroupLayout with
    | some (_, .bgl (some es)) => some es
    | _ => none

/-- A bind group's entries, as the typestate's cells name storage buffers. -/
def wgBgEntries (S : TState) (ents n : UInt64) : Option (List (Nat × Nat)) :=
  wgCells ents n 16 fun e => do
    let b ← cellOf S e
    let bh ← cellOf S (e + 8)
    match wgoAt S bh .buffer with
    | some (bi, .buffer u _) => if u &&& 128 == 0 || b ≥ 0x100000000 then none else some (b.toNat, bi)
    | _ => none

/-- Every group a pipeline's layout names is bound, to a group with that
    layout's bindings, as the typestate has them. -/
def wgBound (S : TState) (gs : List (List (Nat × Bool))) (b : List (Nat × Nat)) : Bool :=
  gs.zipIdx.all fun (ls, k) =>
    match (b.lookup k).bind (wgoOf S) with
    | some (.bgroup es) => es.length == ls.length && ls.all (fun (bn, _) => es.any (·.1 == bn))
    | _ => false

/-- The value of a call that answers nothing. -/
def wgNone : AV := .any

def wgR_createInstance (S : TState) : List AV → Option (TState × AV)
  | [] => wgMake S .instance
  | _ => none

def wgR_requestAdapter (S : TState) : List AV → Option (TState × AV)
  | [i] => do
      let _ ← wgoArg S i .instance
      if S.contains (.part .gpuAdapter true) then wgMake S .adapter else none
  | _ => none

def wgR_requestDevice (S : TState) : List AV → Option (TState × AV)
  | [a] => do
      let _ ← wgoArg S a .adapter
      wgMake S (.device [])
  | _ => none

def wgR_deviceGetQueue (S : TState) : List AV → Option (TState × AV)
  | [d] => do
      let _ ← wgDev S d
      wgMake S .queue
  | _ => none

def wgR_deviceCreateBuffer (S : TState) : List AV → Option (TState × AV)
  | [d, u, sz] => do
      let _ ← wgDev S d
      let usage ← u.bits
      let size ← sz.bits
      if usage == 0 || usage ≥ 0x400 || (usage &&& 1 != 0 && usage &&& ~~~9 != 0)
          || size ≥ 0x8000000000000000 then none
      else wgMake S (.buffer usage size.toNat)
  | _ => none

def wgR_deviceCreateShaderModule (S : TState) : List AV → Option (TState × AV)
  | [d, src, n] => do
      let _ ← wgDev S d
      let a ← src.bits
      let k ← n.bits
      if S.contains (.wgText a k.toNat true) then wgMake S (.shader true) else none
  | _ => none

def wgR_deviceCreateBindGroupLayout (S : TState) : List AV → Option (TState × AV)
  | [d, ents, n] => do
      let _ ← wgDev S d
      let es ← wgBglEntries S (← ents.bits) (← n.bits)
      if (es.map (·.1)).Nodup then wgMake S (.bgl (some es)) else none
  | _ => none

def wgR_deviceCreatePipelineLayout (S : TState) : List AV → Option (TState × AV)
  | [d, arr, n] => do
      let _ ← wgDev S d
      let gs ← wgGroups S (← arr.bits) (← n.bits)
      wgMake S (.playout gs)
  | _ => none

def wgR_deviceCreateComputePipeline (S : TState) : List AV → Option (TState × AV)
  | [d, lh, mh, entry, n] => do
      let (_, l) ← wgoArg S lh .pipelineLayout
      let (_, m) ← wgoArg S mh .shaderModule
      let _ ← wgDev S d
      match l, m with
      | .playout gs, .shader true =>
          if wgTextAt S (← entry.bits) (← n.bits).toNat then wgMake S (.cpipe gs true) else none
      | _, _ => none
  | _ => none

def wgR_deviceCreateBindGroup (S : TState) : List AV → Option (TState × AV)
  | [d, lh, ents, n] => do
      let _ ← wgDev S d
      let (_, l) ← wgoArg S lh .bindGroupLayout
      let es ← wgBgEntries S (← ents.bits) (← n.bits)
      let fits : Bool := match l with
        | .bgl (some ls) => decide (es.map (·.1)).Nodup && es.length == ls.length &&
            es.all (fun (b, _) => ls.any (·.1 == b))
        | _ => decide (es.map (·.1)).Nodup
      if fits then wgMake S (.bgroup es) else none
  | _ => none

def wgR_deviceCreateCommandEncoder (S : TState) : List AV → Option (TState × AV)
  | [d] => do
      let _ ← wgDev S d
      if wg1Of S .commandEncoder == some none then some (.wg1 .commandEncoder (some (.encoder true)) :: (wgBump S ++ S.filter (wgKeeps [] [] [.commandEncoder] none)),
          .hand (.wg .commandEncoder) true)
      else none
  | _ => none

def wgR_deviceCreateRenderPipeline (S : TState) : List AV → Option (TState × AV)
  | [d, mh, vs, vn, fs, fn, fmt] => do
      let (_, m) ← wgoArg S mh .shaderModule
      let _ ← wgDev S d
      if m == .shader true && wgTextAt S (← vs.bits) (← vn.bits).toNat &&
          wgTextAt S (← fs.bits) (← fn.bits).toNat && wgFormatOk (← fmt.bits) then
        wgMake S (.rpipe true)
      else none
  | _ => none

def wgR_renderPipelineGetBindGroupLayout (S : TState) : List AV → Option (TState × AV)
  | [p, idx] => do
      let (_, s) ← wgoArg S p .renderPipeline
      if s == .rpipe true && (← idx.bits).toUInt32 == 0 then wgMake S (.bgl none) else none
  | _ => none

def wgR_encoderBeginComputePass (S : TState) : List AV → Option (TState × AV)
  | [e] => do
      let s ← wg1Arg S e .commandEncoder
      if s == .encoder true && wg1Of S .computePass == some none then
        some (.wg1 .computePass (some (.cpass none [])) :: .wg1 .commandEncoder (some (.encoder false)) ::
          (wgBump S ++ S.filter (wgKeeps [] [.commandEncoder] [.computePass] none)), .hand (.wg .computePass) true)
      else none
  | _ => none

def wgR_computePassSetPipeline (S : TState) : List AV → Option (TState × AV)
  | [p, q] => do
      let s ← wg1Arg S p .computePass
      let (qi, qs) ← wgoArg S q .computePipeline
      match s, qs with
      | .cpass _ b, .cpipe _ true => some (.wg1 .computePass (some (.cpass (some qi) b)) :: S.filter (wgKeeps [] [.computePass] [] (some 0)), wgNone)
      | _, _ => none
  | _ => none

def wgR_computePassSetBindGroup (S : TState) : List AV → Option (TState × AV)
  | [p, idx, g] => do
      let s ← wg1Arg S p .computePass
      let (gi, _) ← wgoArg S g .bindGroup
      let k := (← idx.bits).toUInt32.toNat
      match s with
      | .cpass q b => some (.wg1 .computePass (some (.cpass q ((k, gi) :: b.filter (·.1 != k)))) ::
          S.filter (wgKeeps [] [.computePass] [] (some 0)), wgNone)
      | _ => none
  | _ => none

def wgR_computePassDispatch (S : TState) : List AV → Option (TState × AV)
  | [p, x, y, z] => do
      let s ← wg1Arg S p .computePass
      match s with
      | .cpass (some q) b =>
          match wgoOf S q, wg1Of S .commandEncoder with
          | some (.cpipe gs true), some (some (.encoder false)) =>
              if wgBound S gs b && [← x.num, ← y.num, ← z.num].all (·.hi ≤ 65535) then some (wgSame S, wgNone)
              else none
          | _, _ => none
      | _ => none
  | _ => none

def wgR_computePassEnd (S : TState) : List AV → Option (TState × AV)
  | [p] => do
      let s ← wg1Arg S p .computePass
      match s, wg1Of S .commandEncoder with
      | .cpass _ _, some (some (.encoder false)) =>
          some (.wg1 .computePass none :: .wg1 .commandEncoder (some (.encoder true)) ::
            S.filter (wgKeeps [] [.commandEncoder, .computePass] [.computePass] (some 0)), wgNone)
      | _, _ => none
  | _ => none

def wgR_encoderCopyBufferToBuffer (S : TState) : List AV → Option (TState × AV)
  | [e, src, so, dst, dof, n] => do
      let s ← wg1Arg S e .commandEncoder
      let (si, ss) ← wgoArg S src .buffer
      let (di, ds) ← wgoArg S dst .buffer
      let so ← so.bits
      let dof ← dof.bits
      let nn ← n.num
      match s, ss, ds with
      | .encoder true, .buffer su sn, .buffer du dn =>
          if su &&& 4 == 0 || du &&& 8 == 0 || si == di || so.toNat % 4 != 0 || dof.toNat % 4 != 0
              || nn.al % 4 != 0 || so.toNat + nn.hi > sn || dof.toNat + nn.hi > dn then none
          else some (wgSame S, wgNone)
      | _, _, _ => none
  | _ => none

def wgR_encoderFinish (S : TState) : List AV → Option (TState × AV)
  | [e] => do
      let s ← wg1Arg S e .commandEncoder
      if s == .encoder true && wg1Of S .commandBuffer == some none then
        some (.wg1 .commandBuffer (some .cmdbuf) :: .wg1 .commandEncoder none ::
          (wgBump S ++ S.filter (wgKeeps [] [.commandEncoder, .commandEncoder] [.commandEncoder, .commandBuffer] none)),
          .hand (.wg .commandBuffer) true)
      else none
  | _ => none

def wgR_encoderBeginRenderPass (S : TState) : List AV → Option (TState × AV)
  | [e, v, _] => do
      let s ← wg1Arg S e .commandEncoder
      let _ ← wg1Arg S v .textureView
      if s == .encoder true && wg1Of S .renderPass == some none then
        some (.wg1 .renderPass (some (.rpass none none)) :: .wg1 .commandEncoder (some (.encoder false)) ::
          (wgBump S ++ S.filter (wgKeeps [] [.commandEncoder] [.renderPass] none)), .hand (.wg .renderPass) true)
      else none
  | _ => none

def wgR_renderPassSetPipeline (S : TState) : List AV → Option (TState × AV)
  | [p, q] => do
      let s ← wg1Arg S p .renderPass
      let (qi, qs) ← wgoArg S q .renderPipeline
      match s, qs with
      | .rpass _ g, .rpipe true => some (.wg1 .renderPass (some (.rpass (some qi) g)) :: S.filter (wgKeeps [] [.renderPass] [] (some 0)), wgNone)
      | _, _ => none
  | _ => none

def wgR_renderPassSetBindGroup (S : TState) : List AV → Option (TState × AV)
  | [p, idx, g] => do
      let s ← wg1Arg S p .renderPass
      let (gi, _) ← wgoArg S g .bindGroup
      match s with
      | .rpass q _ =>
          if (← idx.bits).toUInt32 == 0 then some (.wg1 .renderPass (some (.rpass q (some gi))) ::
            S.filter (wgKeeps [] [.renderPass] [] (some 0)), wgNone)
          else none
      | _ => none
  | _ => none

def wgR_renderPassDraw (S : TState) : List AV → Option (TState × AV)
  | [p, _, _, _, _] => do
      let s ← wg1Arg S p .renderPass
      match s, wg1Of S .commandEncoder with
      | .rpass (some _) _, some (some (.encoder false)) => some (wgSame S, wgNone)
      | _, _ => none
  | _ => none

def wgR_renderPassEnd (S : TState) : List AV → Option (TState × AV)
  | [p] => do
      let s ← wg1Arg S p .renderPass
      match s, wg1Of S .commandEncoder with
      | .rpass _ _, some (some (.encoder false)) =>
          some (.wg1 .renderPass none :: .wg1 .commandEncoder (some (.encoder true)) ::
            S.filter (wgKeeps [] [.commandEncoder, .renderPass] [.renderPass] (some 0)), wgNone)
      | _, _ => none
  | _ => none

def wgR_queueSubmit (S : TState) : List AV → Option (TState × AV)
  | [q, cb] => do
      let _ ← wgoArg S q .queue
      let s ← wg1Arg S cb .commandBuffer
      if s == .cmdbuf then
        some (.wg1 .commandBuffer none :: S.filter (wgKeeps [] [.commandBuffer] [.commandBuffer] (some 0)), wgNone)
      else none
  | _ => none

def wgR_queueWriteBuffer (S : TState) : List AV → Option (TState × AV)
  | [q, b, off, src, n] => do
      let _ ← wgoArg S q .queue
      let (_, bs) ← wgoArg S b .buffer
      let off ← off.bits
      let nn ← n.num
      match bs with
      | .buffer u sz =>
          if u &&& 8 == 0 || off.toNat % 4 != 0 || nn.al % 4 != 0 || off.toNat + nn.hi > sz
              || !memB S src n then none
          else some (wgSame S, wgNone)
      | _ => none
  | _ => none

def wgR_bufferRead (S : TState) : List AV → Option (TState × AV)
  | [d, b, off, dst, n] => do
      let _ ← wgDev S d
      let (_, bs) ← wgoArg S b .buffer
      let off ← off.bits
      let nn ← n.num
      match bs with
      | .buffer u sz =>
          if u &&& 1 == 0 || off.toNat % 8 != 0 || nn.lo == 0 || nn.al % 4 != 0
              || off.toNat + nn.hi > sz || !memB S dst n then none
          else some (afterCopy dst.bits nn.hi S, .known (ofInt .i32 0))
      | _ => none
  | _ => none

def wgR_instanceCreateSurface (S : TState) : List AV → Option (TState × AV)
  | [i, win] => do
      let _ ← wgoArg S i .instance
      match win with
      | .hand .window true => if wg1Of S .texture == some none then wgMake S (.surface false) else none
      | _ => none
  | _ => none

def wgR_surfaceConfigure (S : TState) : List AV → Option (TState × AV)
  | [sh, d, fmt, wd, ht] => do
      let (si, ss) ← wgoArg S sh .surface
      let _ ← wgDev S d
      match ss with
      | .surface _ =>
          let wn ← wd.num
          let hn ← ht.num
          if wg1Of S .texture == some none && wgFormatOk (← fmt.bits) &&
              decide (1 ≤ wn.lo) && decide (wn.hi < 2 ^ 31) && decide (1 ≤ hn.lo) && decide (hn.hi < 2 ^ 31) then
            some (.wgo si (.surface true) :: S.filter (wgKeeps [] [.surface] [] (some 0)), wgNone)
          else none
      | _ => none
  | _ => none

def wgR_surfaceGetCurrentTexture (S : TState) : List AV → Option (TState × AV)
  | [sh] => do
      let (si, ss) ← wgoArg S sh .surface
      if ss == .surface true && wg1Of S .texture == some none then
        some (.wg1 .texture (some (.texture si)) :: .wgo si (.surface true) ::
          (wgBump S ++ S.filter (wgKeeps [] [.surface] [.texture] none)), .hand (.wg .texture) true)
      else none
  | _ => none

def wgR_textureCreateView (S : TState) : List AV → Option (TState × AV)
  | [t] => do
      let _ ← wg1Arg S t .texture
      if wg1Of S .textureView == some none then some (.wg1 .textureView (some .view) :: (wgBump S ++ S.filter (wgKeeps [] [] [.textureView] none)),
        .hand (.wg .textureView) true) else none
  | _ => none

def wgR_surfacePresent (S : TState) : List AV → Option (TState × AV)
  | [t] => do
      let s ← wg1Arg S t .texture
      match s with
      | .texture si =>
          if wgoOf S si == some (.surface true) && wg1Of S .textureView == some none then
            some (.wgo si (.surface true) :: .wg1 .texture none ::
              S.filter (wgKeeps [] [.surface, .texture] [.texture] (some 0)), wgNone)
          else none
      | _ => none
  | _ => none

def wgR_release (S : TState) (o : WgObj) : List AV → Option (TState × AV)
  | [h] =>
      match wgoArg S h o, wg1Arg S h o with
      | some (i, _), _ => some (S.filter (wgKeeps [i] [] [o] (some 0)), wgNone)
      | none, some _ => some (.wg1 o none :: S.filter (wgKeeps [] [o] [o] (some 0)), wgNone)
      | none, none => none
  | _ => none

/-- **A wgpu call, decided**: the typestate after it and its answer, where the
    typestate shows the call decodes; `none` otherwise. -/
def wgRow (S : TState) : WgpuFn → List AV → Option (TState × AV)
  | .createInstance => wgR_createInstance S
  | .requestAdapter => wgR_requestAdapter S
  | .requestDevice => wgR_requestDevice S
  | .deviceGetQueue => wgR_deviceGetQueue S
  | .deviceCreateBuffer => wgR_deviceCreateBuffer S
  | .deviceCreateShaderModule => wgR_deviceCreateShaderModule S
  | .deviceCreateBindGroupLayout => wgR_deviceCreateBindGroupLayout S
  | .deviceCreatePipelineLayout => wgR_deviceCreatePipelineLayout S
  | .deviceCreateComputePipeline => wgR_deviceCreateComputePipeline S
  | .deviceCreateBindGroup => wgR_deviceCreateBindGroup S
  | .deviceCreateCommandEncoder => wgR_deviceCreateCommandEncoder S
  | .deviceCreateRenderPipeline => wgR_deviceCreateRenderPipeline S
  | .renderPipelineGetBindGroupLayout => wgR_renderPipelineGetBindGroupLayout S
  | .encoderBeginComputePass => wgR_encoderBeginComputePass S
  | .computePassSetPipeline => wgR_computePassSetPipeline S
  | .computePassSetBindGroup => wgR_computePassSetBindGroup S
  | .computePassDispatch => wgR_computePassDispatch S
  | .computePassEnd => wgR_computePassEnd S
  | .encoderCopyBufferToBuffer => wgR_encoderCopyBufferToBuffer S
  | .encoderFinish => wgR_encoderFinish S
  | .encoderBeginRenderPass => wgR_encoderBeginRenderPass S
  | .renderPassSetPipeline => wgR_renderPassSetPipeline S
  | .renderPassSetBindGroup => wgR_renderPassSetBindGroup S
  | .renderPassDraw => wgR_renderPassDraw S
  | .renderPassEnd => wgR_renderPassEnd S
  | .queueSubmit => wgR_queueSubmit S
  | .queueWriteBuffer => wgR_queueWriteBuffer S
  | .bufferRead => wgR_bufferRead S
  | .instanceCreateSurface => wgR_instanceCreateSurface S
  | .surfaceConfigure => wgR_surfaceConfigure S
  | .surfaceGetCurrentTexture => wgR_surfaceGetCurrentTexture S
  | .textureCreateView => wgR_textureCreateView S
  | .surfacePresent => wgR_surfacePresent S
  | .release o => wgR_release S o
  | _ => fun _ => none

/-- The kinds of object a wgpu call may make or end: those of which the
    one live object may change, so what the checker forgets of the values it
    describes as that one. -/
def wgKills : WgpuFn → List WgObj
  | .createInstance => [.instance]
  | .requestAdapter => [.adapter]
  | .requestDevice => [.device]
  | .deviceGetQueue => [.queue]
  | .deviceCreateBuffer => [.buffer]
  | .deviceCreateShaderModule => [.shaderModule]
  | .deviceCreateBindGroupLayout | .renderPipelineGetBindGroupLayout => [.bindGroupLayout]
  | .deviceCreatePipelineLayout => [.pipelineLayout]
  | .deviceCreateComputePipeline => [.computePipeline]
  | .deviceCreateBindGroup => [.bindGroup]
  | .deviceCreateCommandEncoder => [.commandEncoder]
  | .deviceCreateRenderPipeline => [.renderPipeline]
  | .devicePushErrorScope | .errorScopePop => [.errorScope]
  | .encoderBeginComputePass => [.computePass]
  | .computePassEnd => [.computePass]
  | .encoderBeginRenderPass => [.renderPass]
  | .renderPassEnd => [.renderPass]
  | .encoderFinish => [.commandEncoder, .commandBuffer]
  | .queueSubmit => [.commandBuffer]
  | .instanceCreateSurface => [.surface]
  | .surfaceGetCurrentTexture | .surfacePresent => [.texture]
  | .textureCreateView => [.textureView]
  | .release o => [o]
  | _ => []

end AlgorithmLib.Check
