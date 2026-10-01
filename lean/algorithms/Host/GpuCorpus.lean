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
# The wgpu contracts, checked against the device

What `Sem.callFfi` says the nine wgpu entry points do, run on the GPU through
`base/tests/hprog_gpu_corpus.rs` and compared byte for byte with what the
interpreter computed here. The shader adds one to every `u32` of the buffer
bound to it, and so does the interpreter's shader oracle.

One case is here for the order more than for the values: a dispatch, then an
upload, then a download. The runtime records a dispatch and submits it only at
the next dispatch or download, while `queue.write_buffer` lands before that
submit, so the dispatch sees the data uploaded *after* it. The model says so,
and this is what checks that it says so correctly.

The last cases hand uploads and downloads sizes the buffer does not have: past
its end, not the whole buffer for a download, or not whole words. Each answers
`-1` and changes nothing.
-/

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.Prog

namespace HProgGpuCorpus

def addOneWgsl : String :=
  "@group(0) @binding(0) var<storage, read_write> a: array<u32>;\n" ++
  "@compute @workgroup_size(64)\n" ++
  "fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n" ++
  "  if (id.x < arrayLength(&a)) { a[id.x] = a[id.x] + 1u; }\n}\n"

/-- The shader, as the interpreter's oracle: every `u32` of a writable binding
    plus one. -/
def addOne (d : Sem.Dispatch) (ins : List ByteArray) : List ByteArray :=
  if d.shader != addOneWgsl then ins
  else ins.map fun b =>
    (List.range (b.size / 4)).foldl (fun b i =>
      let v : UInt32 := (List.range 4).foldr (fun j acc => (acc <<< 8) ||| (b.get! (4 * i + j)).toUInt32) 0
      let v := v + 1
      (List.range 4).foldl (fun b j => b.set! (4 * i + j) ((v >>> (8 * j.toUInt32)).toUInt8)) b) b

def SRC : Nat := 0x100
def SRC2 : Nat := 0x140
def BIND : Nat := 0x180
def BAD_BIND : Nat := 0x190
def SHADER : Nat := 0x200
def MEM : Nat := 0x800
def OUT : Nat := 320

def src1 : List UInt8 := (List.range 64).map (fun i => (i * 5 + 3).toUInt8)
def src2 : List UInt8 := (List.range 64).map (fun i => (200 - i).toUInt8)

def image : List UInt8 :=
  let pad (xs : List UInt8) (n : Nat) := xs ++ List.replicate (n - xs.length) 0
  let a := pad (pad (List.replicate SRC 0 ++ src1) SRC2 ++ src2) BAD_BIND
  pad (pad (a ++ [99, 0, 0, 0, 0, 0, 0, 0]) SHADER ++ (stringToBytes addOneWgsl ++ [0])) MEM

def body : Prog V L Unit := do
  let ptr ← basePtr
  let out ← outPtr
  let code (k : Nat) (r : V .i32) : Prog V L Unit := do
    storeI32 r (← iadd out (← iconst64 (4 * k)))
  gpuInit ptr
  let c ← gpuCtxPtr ptr
  let b0 ← gpuCreateBuffer ptr (← iconst64 64)
  code 0 b0
  code 1 (← gpuCreateBuffer ptr (← iconst64 0))
  code 2 (← gpuUpload ptr b0 (← iconst64 SRC) (← iconst64 64))
  code 3 (← gpuUpload ptr (← iconst32 99) (← iconst64 SRC) (← iconst64 64))
  storeI32 b0 (← iadd ptr (← iconst64 BIND))
  storeI32 (← iconst32 0) (← iadd ptr (← iconst64 (BIND + 4)))
  let one ← iconst32 1
  let p ← gpuCreatePipeline ptr (← iconst64 SHADER) (← iconst64 BIND) one
  code 4 p
  code 5 (← gpuCreatePipeline ptr (← iconst64 SHADER) (← iconst64 BAD_BIND) one)
  code 6 (← gpuDispatch ptr p one one one)
  code 7 (← gpuDispatch ptr (← iconst32 99) one one one)
  code 8 (← ffi .gpuDownload %[c, b0, ← iadd out (← iconst64 128), ← iconst64 64])
  -- The dispatch waits; the upload after it lands first.
  code 9 (← gpuDispatch ptr p one one one)
  code 10 (← gpuUpload ptr b0 (← iconst64 SRC2) (← iconst64 64))
  code 11 (← ffi .gpuDownload %[c, b0, ← iadd out (← iconst64 192), ← iconst64 64])
  code 12 (← ffi .gpuDownloadPtr %[c, b0, ← iconst64 16, ← iadd out (← iconst64 256), ← iconst64 16])
  -- Sizes the buffer does not have, refused before wgpu sees them.
  let dst ← iadd out (← iconst64 288)
  code 13 (← gpuUpload ptr b0 (← iconst64 SRC) (← iconst64 128))
  code 14 (← gpuUpload ptr b0 (← iconst64 SRC) (← iconst64 6))
  code 15 (← ffi .gpuDownload %[c, b0, dst, ← iconst64 32])
  code 16 (← ffi .gpuDownload %[c, b0, dst, ← iconst64 68])
  code 17 (← ffi .gpuDownloadPtr %[c, b0, ← iconst64 56, dst, ← iconst64 16])
  code 18 (← ffi .gpuDownloadPtr %[c, b0, ← iconst64 2, dst, ← iconst64 8])
  code 19 (← ffi .gpuDownloadPtr %[c, b0, ← iconst64 0, dst, ← iconst64 6])
  gpuCleanup ptr

def code : Code := Prog.emit body
def checked : Except String Code := Prog.emitChecked body
def program : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 body)]
def env : FnEnv := (Prog.run body).2.1

def entryArgVals : List Sem.V :=
  [.sc .i64 (Sem.regionBase .arena), .sc .i64 (Sem.regionBase .data),
   .sc .i64 0, .sc .i64 (Sem.regionBase .out), .sc .i64 OUT.toUInt64]

def startWorld : Sem.World :=
  { mem := { arena := ⟨image.toArray⟩, data := ByteArray.empty,
             out := ByteArray.mk (Array.replicate OUT 0) },
    shader := addOne }

def expected : Except String ByteArray :=
  match Sem.run { env } entryArgVals startWorld code with
  | .stuck why | .misuse why | .fault why => .error why
  | .ok _ w => .ok w.mem.out

end HProgGpuCorpus

open AlgorithmLib in
def Host.GpuCorpus.main (args : List String) : IO Unit := do
  let dir ← requireOutputDir args
  match HProgGpuCorpus.checked with
  | .error e => throw (IO.userError s!"the wgpu corpus body is not well-formed: {e}")
  | .ok _ => pure ()
  match HProgGpuCorpus.expected with
  | .error e => throw (IO.userError s!"interpreting the wgpu corpus: {e}")
  | .ok bytes =>
      let clif ← AlgorithmLib.Prog.orDie HProgGpuCorpus.program
      emitArtifacts dir #[artifactEntry "hprog_gpu_corpus" {
        functions := clif, required_memory := HProgGpuCorpus.MEM,
        initial_memory := HProgGpuCorpus.image
      }]
      let sideDir := System.FilePath.mk dir / "hprog_gpu_corpus"
      IO.FS.createDirAll sideDir
      IO.FS.writeFile (sideDir / "expected.json")
        (Lean.Json.mkObj [("expected", Lean.toJson (bytes.toList.map (·.toNat)))]).compress
      IO.println s!"wgpu corpus: {bytes.size} expected bytes"

#eval ShipScan.check "Host.GpuCorpus" `Host.GpuCorpus.main
