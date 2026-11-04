module
public import Scan.Ship
meta import Scan.Ship
public import AlgorithmLib.Surface.Drv
public import AlgorithmLib.Surface.Link
meta import AlgorithmLib.Surface.Link
meta import AlgorithmLib.Surface.Drv
public import Lz4.Kernel
meta import Lz4.Kernel
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open Lean (Json)
open AlgorithmLib
open AlgorithmLib.PTX

namespace Lz4Ship

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.Prog

/-- **The compressor in the driver's standard layout.** One allocation holds
    the input at offset 0 and the output immediately after it, and the kernel
    is handed it for both its parameters: it derives its output base as
    `in_ptr + totIn`, so the placement contract `LayoutOK` asks for holds by
    construction instead of relating two allocations. The layout's state sits
    past the kernel's text, at `org`; the text lies within the room before it.

    `org` is `bindOff`, which can only be computed by serializing the whole
    kernel; holding it as a parameter keeps that dependency explicit, and
    `lzApp` instantiates it. -/
def lzAppAt (w : WP) (org : Nat) : Drv.App :=
  { bufs := [w.outOff + w.outAlloc],
    kernels := [{ ptxOff := rPTX_OFF, ptxLen := org - rPTX_OFF, args := [0, 0], grid := (w.gridX, 1, 1),
                  block := (wBlockDim, 1, 1) }],
    memSize := org + 0x140, org := org }

def lzApp (w : WP) : Drv.App := lzAppAt w w.bindOff

/-- Bring the driver up, make the allocation, load the kernel and write its
    parameters: once, before any compression. -/
def loadCodeAt (w : WP) (org : Nat) : Prog V L Unit := do
  (lzAppAt w org).setup (← basePtr)

def loadCode (w : WP) : Prog V L Unit := loadCodeAt w w.bindOff

/-- **One compression.** The caller's data is uploaded to the allocation's
    start (a length past the input's room is refused), the kernel launched
    `rLaunches` times, and the blocks copied back only into an output with room
    for all of them; after them, the geometry this program was built for, so a
    host checks its own against it and prices a run by launches x bytes per
    launch. A buffer with no room past the blocks is not written past them. -/
def warpCodeAt (w : WP) (org : Nat) : Prog V L Unit := do
  let base ← basePtr
  let dataPtr ← dataPtr
  let dataLen ← dataLen
  let outPtr ← outPtr
  let outLen ← outLen
  when .ule dataLen (← iconst64 w.outOff) do
    let _ ← ext (.cuda .memcpyHtoD) %[← load64 (← iaddImm base ((lzAppAt w org).bufSlot 0)), dataPtr, dataLen]
  let _ ← forLoopAcc (← iconst64 rLaunches) (← iconst64 0) (fun _ acc => do
    (lzAppAt w org).run base 0
    pure acc)
  when .uge outLen (← iconst64 w.totOut) do
    let src ← iaddImm (← load64 (← iaddImm base ((lzAppAt w org).bufSlot 0))) w.outOff
    let _ ← ext (.cuda .memcpyDtoH) %[outPtr, src, ← iconst64 w.totOut]
  when .uge outLen (← iconst64 (w.totOut + 48)) do
    storeAt outPtr w.totOut (← iconst64 rLaunches)
    storeAt outPtr (w.totOut + 8) (← iconst64 w.totIn)
    storeAt outPtr (w.totOut + 16) (← iconst64 w.inStride)
    storeAt outPtr (w.totOut + 24) (← iconst64 w.outStride)
    storeAt outPtr (w.totOut + 32) (← iconst64 w.lenOff)
    storeAt outPtr (w.totOut + 40) (← iconst64 w.numBlk)

def warpCode (w : WP) : Prog V L Unit := warpCodeAt w w.bindOff

def warpClif (w : WP) : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "load" (Prog.compileProg 1 (loadCode w)),
    Prog.entry "main" (Prog.compileProg 2 (warpCode w))]

/-- The image the artifact embeds: the kernel's text at `rPTX_OFF`, the
    layout's constants past it. -/
def warpPayloadDSL (w : WP) : List UInt8 := (lzApp w).initialMemory [warpKernelDSLStr w]

theorem payload_fits (w : WP) : (warpPayloadDSL w).length ≤ w.memSize :=
  Nat.le_of_eq (Drv.App.initialMemory_length _ _ _)

open AlgorithmLib.IR AlgorithmLib.HProg in
def warpArtifactDSL (name : String) (blkLog : Nat) : Except String ArtifactEntry := do
  let w : WP := ⟨blkLog⟩
  let _ ← w.code.ptx
  let functions ← warpClif w
  return AlgorithmLib.artifactEntry name
    { functions, required_memory := w.memSize, initial_memory := warpPayloadDSL w }

end Lz4Ship

def Lz4.Comp.main (args : List String) : IO Unit := do
  let outDir ← AlgorithmLib.requireOutputDir args
  AlgorithmLib.emitArtifacts outDir #[
    ← AlgorithmLib.Prog.orDie (Lz4Ship.warpArtifactDSL "lz4_comp_warpdsl" 15),
    ← AlgorithmLib.Prog.orDie (Lz4Ship.warpArtifactDSL "lz4_comp_warpdsl64" 16)]

#eval ShipScan.check "Lz4.Comp" `Lz4.Comp.main
