module
public import Lean
public import Scan.Ship
meta import Scan.Ship
public import AlgorithmLib.Host.Sem
meta import AlgorithmLib.Host.Sem
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic

@[expose] public section

/-!
# wgpu, checked against wgpu

`Sem.wgpuCall` states what each wgpu call a program makes does. This body
makes them in one sequence — an instance, adapter, device and queue; a storage
buffer written and a staging buffer; a compute pipeline made inside an error
scope; a dispatch, a copy, a submit and the bytes read back; a shader wgpu
rejects, caught by a scope — and stores every answer and the bytes read.
`base/tests/hprog_wgpu_corpus.rs` runs it and compares.

The world's shader oracle computes what the one shader here does, `2x + 1` on
each `u32`; which WGSL wgpu rejects is its `wgslOk`.
-/

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.Prog

namespace HProgWgpuCorpus

def SRC : String :=
  "@group(0) @binding(0) var<storage, read_write> data: array<u32>;\n" ++
  "@compute @workgroup_size(64)\n" ++
  "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n" ++
  "  let i = gid.x;\n" ++
  "  if (i < arrayLength(&data)) { data[i] = data[i] * 2u + 1u; }\n" ++
  "}\n"
def BAD : String := "fn main( {"

def A_SRC : Nat := 0x000
def A_BAD : Nat := 0x200
def A_IN : Nat := 0x240
/-- The layout's one entry, a word: binding 0, written. -/
def A_BGLE : Nat := 0x300
/-- The one group layout a pipeline layout is made of. -/
def A_SLOT : Nat := 0x310
/-- The group's one entry, two words: binding 0, the buffer. -/
def A_BGE : Nat := 0x320
def A_MAIN : Nat := 0x540
def MEM : Nat := 0x600

def N : Nat := 16
def READ : Nat := 64
def OUT : Nat := READ + 4 * N

def image : List UInt8 :=
  let pad (xs : List UInt8) (n : Nat) := xs ++ List.replicate (n - xs.length) 0
  let src := pad (SRC.toUTF8.toList ++ [0]) A_BAD
  let bad := pad (BAD.toUTF8.toList ++ [0]) (A_IN - A_BAD)
  let inp := (List.range N).flatMap fun i => [(i + 1).toUInt8, 0, 0, 0]
  let main := "main".toUTF8.toList
  pad (src ++ bad ++ pad inp (A_MAIN - A_IN) ++ main) MEM

def wg (f : WgpuFn) (args : Vals V (Ext.wgpu f).sig.1) : Prog V L (ResV V (Ext.wgpu f).sig.2) :=
  ext (.wgpu f) args

def rel (o : WgObj) (h : V .i64) : Prog V L Unit := do
  let _ ← wg (.release o) %[h]
  pure ()

/-- `1` when `cc a b`, `0` otherwise. -/
def flag64 (cc : ICmpCond) (a b : V .i64) : Prog V L (V .i32) := do
  ireduce32 (← uextend64 (← icmp cc a b))

/-- A compute pipeline of the `len` bytes of WGSL at `src`, over the
    one-binding layout. -/
def pipeline (base dev bgl : V .i64) (src len : Nat) : Prog V L (V .i64 × V .i64 × V .i64) := do
  let module ← wg .deviceCreateShaderModule %[dev, ← absAddr base src, ← iconst64 len]
  let slot ← absAddr base A_SLOT
  storeI64 bgl slot
  let pl ← wg .deviceCreatePipelineLayout %[dev, slot, ← iconst64 1]
  let pipe ← wg .deviceCreateComputePipeline %[dev, pl, module, ← absAddr base A_MAIN, ← iconst64 4]
  pure (module, pl, pipe)

set_option maxRecDepth 8000 in
def body : Prog V L Unit := do
  let base ← basePtr
  let out ← outPtr
  let z ← iconst64 0
  let put (k : Nat) (v : V .i32) : Prog V L Unit := do storeI32 v (← iadd out (← iconst64 (4 * k)))
  put 0 (← libPresent .wgpu)
  let inst ← wg .createInstance %[]
  let ad ← wg .requestAdapter %[inst]
  put 1 (← flag64 .ne ad z)
  let dev ← wg .requestDevice %[ad]
  put 2 (← flag64 .ne dev z)
  let q ← wg .deviceGetQueue %[dev]
  -- a storage buffer and its staging buffer
  let size ← iconst64 (4 * N)
  let b ← wg .deviceCreateBuffer %[dev, ← iconst64 (128 + 8 + 4), size]
  let st ← wg .deviceCreateBuffer %[dev, ← iconst64 (1 + 8), size]
  put 3 (← flag64 .ne b z)
  let _ ← wg .queueWriteBuffer %[q, b, z, ← absAddr base A_IN, size]
  -- the layout: binding 0, a storage buffer written
  let be ← absAddr base A_BGLE
  storeI64 z be
  let bgl ← wg .deviceCreateBindGroupLayout %[dev, be, ← iconst64 1]
  let scope ← wg .devicePushErrorScope %[dev]
  let (module, pl, pipe) ← pipeline base dev bgl A_SRC SRC.utf8ByteSize
  put 4 (← wg .errorScopePop %[scope])
  -- the bind group: the whole buffer at binding 0
  let ge ← absAddr base A_BGE
  storeI64 z ge
  storeI64 b (← iaddImm ge 8)
  let bg ← wg .deviceCreateBindGroup %[dev, bgl, ge, ← iconst64 1]
  -- dispatch, copy to staging, submit, read back
  let enc ← wg .deviceCreateCommandEncoder %[dev]
  let pass ← wg .encoderBeginComputePass %[enc]
  let _ ← wg .computePassSetPipeline %[pass, pipe]
  let _ ← wg .computePassSetBindGroup %[pass, ← iconst32 0, bg]
  let _ ← wg .computePassDispatch %[pass, ← iconst32 1, ← iconst32 1, ← iconst32 1]
  let _ ← wg .computePassEnd %[pass]
  let _ ← wg .encoderCopyBufferToBuffer %[enc, b, z, st, z, size]
  let _ ← wg .queueSubmit %[q, ← wg .encoderFinish %[enc]]
  put 5 (← wg .bufferRead %[dev, st, z, ← iaddImm out READ, size])
  -- a shader wgpu rejects, caught by a scope
  let scope ← wg .devicePushErrorScope %[dev]
  let (bmod, bpl, bpipe) ← pipeline base dev bgl A_BAD BAD.utf8ByteSize
  put 6 (← wg .errorScopePop %[scope])
  -- what is held is released once; what a call used up is not
  rel .computePipeline bpipe
  rel .pipelineLayout bpl
  rel .shaderModule bmod
  rel .bindGroup bg
  rel .computePipeline pipe
  rel .pipelineLayout pl
  rel .shaderModule module
  rel .bindGroupLayout bgl
  rel .buffer st
  rel .buffer b
  rel .queue q
  rel .device dev
  rel .adapter ad
  rel .instance inst
  put 7 (← iconst32 1)

def codeNames : List String :=
  [ "wgpu present", "an adapter", "a device", "a buffer", "the pipeline's scope caught nothing",
    "the read back", "the rejected shader's scope caught it", "released" ]

def checked : Except String Code := Prog.emitChecked body
def code : Code := Prog.emit body
def program : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 body)]
def env : FnEnv := (Prog.run body).2.1

def entryArgVals : List Sem.V :=
  [.sc .i64 (Sem.regionBase .arena), .sc .i64 (Sem.regionBase .data),
   .sc .i64 8, .sc .i64 (Sem.regionBase .out), .sc .i64 OUT.toUInt64]

/-- What the one shader here computes: `2x + 1` on each `u32`. -/
def shader (d : Sem.Dispatch) (bs : List ByteArray) : List ByteArray :=
  if d.shader != SRC then bs else bs.map fun b =>
    ⟨(List.range (b.size / 4)).foldl (fun (a : Array UInt8) i =>
      let x := (List.range 4).foldl (fun acc j => acc + (b.get! (4 * i + j)).toNat * 256 ^ j) 0
      let y := (2 * x + 1) % 2 ^ 32
      a ++ #[(y % 256).toUInt8, (y / 256 % 256).toUInt8, (y / 65536 % 256).toUInt8,
             (y / 16777216 % 256).toUInt8]) #[]⟩

def startWorld : Sem.World :=
  { mem := { arena := ⟨image.toArray⟩, data := ByteArray.mk (Array.replicate 8 0),
             out := ByteArray.mk (Array.replicate OUT 0) },
    shader, wgslOk := fun s => s == SRC }

def expected : Except String ByteArray :=
  match Sem.run { env } entryArgVals startWorld code with
  | .stuck why | .misuse why | .fault why => .error why
  | .ok _ w => .ok w.mem.out

end HProgWgpuCorpus

open AlgorithmLib in
def Host.WgpuCorpus.main (args : List String) : IO Unit := do
  let dir ← requireOutputDir args
  match HProgWgpuCorpus.checked with
  | .error e => throw (IO.userError s!"the wgpu corpus body is not well-formed: {e}")
  | .ok _ => pure ()
  match HProgWgpuCorpus.expected with
  | .error e => throw (IO.userError s!"interpreting the wgpu corpus: {e}")
  | .ok bytes =>
      let clif ← AlgorithmLib.Prog.orDie HProgWgpuCorpus.program
      emitArtifacts dir #[artifactEntry "hprog_wgpu_corpus" {
        functions := clif, required_memory := HProgWgpuCorpus.MEM,
        initial_memory := HProgWgpuCorpus.image }]
      let sideDir := System.FilePath.mk dir / "hprog_wgpu_corpus"
      IO.FS.createDirAll sideDir
      IO.FS.writeFile (sideDir / "expected.json")
        (Lean.Json.mkObj [("expected", Lean.toJson (bytes.toList.map (·.toNat))),
                          ("names", Lean.toJson HProgWgpuCorpus.codeNames),
                          ("read", Lean.toJson HProgWgpuCorpus.READ)]).compress
      IO.println s!"wgpu corpus: {bytes.size} expected bytes"

#eval ShipScan.check "Host.WgpuCorpus" `Host.WgpuCorpus.main
