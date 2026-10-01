module
public import Lean
public import AlgorithmLib.Surface.Link
meta import AlgorithmLib.Surface.Link
public import Scan.Ship
meta import Scan.Ship
public import Host.CudaCorpus
meta import Host.CudaCorpus
public import AlgorithmLib.Surface.Handles
meta import AlgorithmLib.Surface.Handles
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic

@[expose] public section

/-!
# Streams, events and graphs, checked against the device

What `Sem.callFfi` says created streams, events and captured graphs do, run on
the GPU through `base/tests/hprog_stream_corpus.rs` and compared with the
interpreter byte for byte. Two streams ordered by an event; a graph captured
across a fork into a second stream and a join back, replayed twice. Every
conflicting pair of launches here is ordered, as it must be: a program the race
tracker refuses is one the interpreter is stuck on, and the generator would not
ship it. The tracker's refusals are checked in `Host/RaceChecks.lean`.
-/

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.Prog

namespace HProgStreamCorpus

open HProgCudaCorpus (SRC KERNEL MEM image addOne)
open HProgVendorRef (vendorRef)

/-- Two binding slots: `X`, then `Y`. -/
def BA : Nat := 0x700
def BB : Nat := 0x704
def OUT : Nat := 320

set_option maxRecDepth 8000 in
def body : Prog V L Unit := do
  let ptr ← basePtr
  let out ← outPtr
  let code (k : Nat) (r : V .i32) : Prog V L Unit := do
    storeI32 r (← iadd out (← iconst64 (4 * k)))
  cudaInit ptr
  let c ← cudaCtx ptr
  let one ← iconst32 1
  let w64 ← iconst32 64
  let launchOn (bind : Nat) (s : Strm V) : Prog V L (V .i32) := do
    c.launchOn (← iadd ptr (← iconst64 KERNEL)) one (← iadd ptr (← iconst64 bind))
      one one one w64 one one s
  let s1 ← c.streamCreate
  code 0 s1.id
  let s2 ← c.streamCreate
  code 1 s2.id
  let bX ← cudaCreateBuffer ptr (← iconst64 64)
  let _ ← cudaUpload ptr bX (← iconst64 SRC) (← iconst64 64)
  let bY ← cudaCreateBuffer ptr (← iconst64 64)
  let _ ← cudaUpload ptr bY (← iconst64 SRC) (← iconst64 64)
  storeI32 bX (← iadd ptr (← iconst64 BA))
  storeI32 bY (← iadd ptr (← iconst64 BB))
  -- stream 1 then stream 2 on X, ordered by an event
  code 2 (← launchOn BA s1)
  let e ← c.eventCreate
  code 3 e.id
  code 4 (← c.eventRecord e s1)
  code 5 (← c.streamWaitEvent s2 e)
  code 6 (← launchOn BA s2)
  code 7 (← c.streamSync s2)
  code 8 (← ffi .cudaDownload %[c.ptr, bX, ← iadd out (← iconst64 128), ← iconst64 64])
  -- a graph: stream 1 on X, forked to stream 2 on Y, joined back
  let e2 ← c.eventCreate
  let e3 ← c.eventCreate
  code 9 (← c.beginCapture s1)
  code 10 (← launchOn BA s1)
  code 11 (← c.eventRecord e2 s1)
  code 12 (← c.streamWaitEvent s2 e2)
  code 13 (← launchOn BB s2)
  code 14 (← c.eventRecord e3 s2)
  code 15 (← c.streamWaitEvent s1 e3)
  let g ← c.endCapture s1
  code 16 g.id
  code 17 (← c.graphUpload g s1)
  code 18 (← c.graphLaunch g s1)
  code 19 (← c.graphLaunch g s1)
  code 20 (← c.streamSync s1)
  code 21 (← ffi .cudaDownload %[c.ptr, bX, ← iadd out (← iconst64 192), ← iconst64 64])
  code 22 (← ffi .cudaDownload %[c.ptr, bY, ← iadd out (← iconst64 256), ← iconst64 64])
  code 23 (← c.streamDestroy s2)
  code 24 (← c.streamDestroy s2)
  code 25 (← c.graphDestroy g)
  code 26 (← c.eventDestroy e)
  code 27 (← launchOn BA ⟨← iconst32 99⟩)
  cudaCleanup ptr

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
    kernel := addOne, vendor := vendorRef }

def expected : Except String ByteArray :=
  match Sem.run { env } entryArgVals startWorld code with
  | .stuck why | .misuse why | .fault why => .error why
  | .ok _ w => .ok w.mem.out

end HProgStreamCorpus

open AlgorithmLib in
def Host.StreamCorpus.main (args : List String) : IO Unit := do
  let dir ← requireOutputDir args
  match HProgStreamCorpus.checked with
  | .error e => throw (IO.userError s!"the stream corpus body is not well-formed: {e}")
  | .ok _ => pure ()
  match HProgStreamCorpus.expected with
  | .error e => throw (IO.userError s!"interpreting the stream corpus: {e}")
  | .ok bytes =>
      let clif ← AlgorithmLib.Prog.orDie HProgStreamCorpus.program
      emitArtifacts dir #[artifactEntry "hprog_stream_corpus" {
        functions := clif, required_memory := HProgCudaCorpus.MEM,
        initial_memory := HProgCudaCorpus.image
      }]
      let sideDir := System.FilePath.mk dir / "hprog_stream_corpus"
      IO.FS.createDirAll sideDir
      IO.FS.writeFile (sideDir / "expected.json")
        (Lean.Json.mkObj [("expected", Lean.toJson (bytes.toList.map (·.toNat)))]).compress
      IO.println s!"stream corpus: {bytes.size} expected bytes"

#eval ShipScan.check "Host.StreamCorpus" `Host.StreamCorpus.main
