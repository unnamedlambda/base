module
public import Scan.Ship
meta import Scan.Ship
public import AlgorithmLib.Surface.Link
meta import AlgorithmLib.Surface.Link
public import Vit.Algorithm
meta import Vit.Algorithm
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open AlgorithmLib AlgorithmLib.ML

/-!
  # What the ViT artifact ships

  Its own module because the well-formedness of its bodies is decided
  here, and that decision is the longest single step in building this
  development. Nothing that reads the tape needs it, so keeping it beside
  them rather than under them keeps it off the critical path.
-/

namespace Vit
open AlgorithmLib.IR

/-- Every emitted body, with the CLIF function index it ships as and the name a
    host calls it by.

    The bodies are named here rather than at the `compileProg` calls so that
    the list is the one place a function index is chosen. -/
def vBodies : List (Nat × String × Prog.Body) :=
  [ (1,  "main",        vLoadFn)
  , (2,  "run",         vRangeFn 0 VFWD_N)
  , (3,  "fetch",       vFetchFn VOUT (SQ * NC * 4))
  , (4,  "step",        vRangeFn 0 VSTEP_N)
  , (5,  "seed",        vSeedFn)
  , (6,  "fetchAny",    vFetchAnyFn)
  -- The backward and the updates, from where the seed was uploaded. Not the
  -- whole step: `dL/dlogits` is written between the forward and the
  -- backward, so a graph spanning both would replay against the seed it
  -- captured.
  , (7,  "bwd",         vRangeFn VFWD_N VBWD_N)
  , (8,  "sgd",         vRangeFn VBWD_N VSTEP_N)
  , (9,  "reload",      vReloadFn)
  , (10, "capture",     vCaptureAt 0 VFWD_N VGRAPH_FWD VEXEC_FWD)
  , (11, "replay",      vReplayAt VEXEC_FWD 1)
  , (12, "captureStep", vCaptureAt VFWD_N VSTEP_N VGRAPH_STEP VEXEC_STEP)
  , (13, "replayStep",  vReplayAt VEXEC_STEP 1) ]

def vClifIR : Except String (List FuncData) := Prog.program (.ok noopFunction ::
  vBodies.map (fun (i, name, body) => Prog.entry name (Prog.compileProg i body)))

/-- Where each input's gradient landed, one `u32` per input, `0` for the
    constants and the patch embedding that are not trained.  A host reads this
    out of the artifact rather than reproducing the reverse pass's allocation. -/
def vGradMapBytes : List UInt8 :=
  (List.range VBASE).flatMap (fun i => u32le ((vGradOf i).getD 0))

def vInitialMemory : List UInt8 :=
  vApp.initialMemory (vKernUnits.map fun k => vPtx.getD (vSlotIx k) "")
    [(VHOST_LEN_OFF, u32le VHOST_BYTES), (VMASK_OFF, vMaskBytes),
     (VSIZES_OFF, vBufBytes.flatMap fun n => u32le n ++ [0, 0, 0, 0]), (VGMAP_OFF, vGradMapBytes)]

def vSetup (clif : List FuncData) : Artifact := {
  functions := clif
  required_memory := VMEM_SIZE
  initial_memory := vInitialMemory
}

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[ artifactEntry "vit_block" (vSetup clif) ]

end Vit


/-- The shape of what was written.

    These print from `main` rather than from an `#eval`: every one of them
    forces the whole tape, the schedule and the emitted PTX, and an `#eval` runs
    that in the elaborator — a second full generation whose only product is the
    text below.  Measured, it doubled the edit-to-artifact cycle from 65s to
    128s, while elaborating the definitions alone takes 2.2s. -/
def report : IO Unit := do
  for (nm, t) in [("forward", Vit.vTape), ("backward", Vit.vBwd),
                  ("update", Vit.vSgd), ("step", Vit.vAll)] do
    let v := (t.filter Vit.vIsVendor).length
    IO.println s!"[vit] {nm}: {t.length} ops ({v} cuBLAS, {t.length - v} proven)"
  IO.println s!"[vit] distinct kernels {Vit.VNSLOT}, {Vit.VNUNIT} launches for {Vit.vAll.length} operations, widest group binds {Vit.VCELLS}"
  IO.println s!"[vit] buffers {Vit.VNBUF}, device {(Vit.vBufBytes.foldl (· + ·) 0) / 1048576} MiB"
  IO.println s!"[vit] host in {Vit.VHOST_BYTES} B, artifact mem {Vit.VMEM_SIZE / 1048576} MiB"
  IO.println s!"[vit] trained parameters {Vit.vSgd.length} of {Vit.VBASE - 5}"
  let ls : List Nat := Vit.vPtx.map (fun t => t.toUTF8.toList.length + 1)
  IO.println s!"[vit] PTX: max kernel {ls.foldl max 0} B, all slots {Vit.VPTX_BYTES / 1024} KiB"
  let d := Vit.vLevel.foldl max 0
  let sp := (List.range Vit.VNSTRM).map (fun s =>
    (Vit.vStrm.toList.filter (· == s)).length)
  IO.println s!"[vit] schedule: depth {d} of {Vit.VNUNIT} launches, {Vit.VNSTRM} streams, per-stream {sp}"
  let sizes := Vit.vUnitsRaw.map List.length
  let hist := (List.range 8).map (fun k => (k+1, (sizes.filter (· == k+1)).length))
  IO.println s!"[vit] group sizes {hist}"
  IO.println s!"[vit] group breaks {Vit.vBreaks}"
  IO.println s!"[vit] ops per grid {Vit.vGridHist}"
  IO.println s!"[vit] warps {Vit.vWarpCensus}"
  IO.println s!"[vit] traffic {Vit.vTraffic}"
  IO.println s!"[vit] fusion ceiling {Vit.vFuseCeiling}"
  IO.println s!"[vit] register-forwardable {Vit.vRegForward}"
  IO.println s!"[vit] refused because {Vit.vRefuseCensus}"
  IO.println s!"[vit] sinks accepted {Vit.vSinkApplied} of {Vit.vSinkList.length}"
  IO.println s!"[vit] hoists accepted {Vit.vHoistApplied} of {Vit.vHoistList.length}"
  IO.println s!"[vit] hoistable but declined {(Vit.vHoistTargets false [Vit.vTape.length, Vit.vTape.length + Vit.vBwd.length] Vit.vAll).length} of which taken {Vit.vHoistList.length} intermediates"
  IO.println s!"[vit] fusion sites {Vit.vFuseCensus}"
  IO.println s!"[vit] chains {Vit.vChainCensus}"
  IO.println s!"[vit] contractions (tA,tB,m,n,k) {Vit.vGemmCensus}"
  IO.println s!"[vit] horizontal {Vit.vHoriz}"
  IO.println s!"[vit] horizontal contiguity {Vit.vHorizContig}"
  IO.println s!"[vit] self-contained {Vit.vSelfContained}"
  IO.println s!"[vit] closed {Vit.vHorizClosed.2}"
  IO.println s!"[vit] concat {Vit.vGemmConcat}"
  IO.println s!"[vit] dependence check {Vit.vDepsCheck}"
  IO.println s!"[vit] readers {Vit.vReaderCensus}"
  IO.println s!"[vit] fusion plan {Vit.vFuseReport}"
  IO.println s!"[vit] rejected because {Vit.vSelReasons}"
  IO.println s!"[vit] fusion safety: shift refusals {Vit.vShiftFailures}, unshiftable reads {Vit.vUnshiftableReads}"
  IO.println s!"[vit] same check, one edge dropped {Vit.vDepsCheckWith true}"
  IO.println s!"[vit] grid pairs at a break {Vit.vGridPairs}"
  let szN := Vit.vUnitsNorm.map List.length
  IO.println s!"[vit] row-chunked elementwise would give {Vit.vUnitsNorm.length} launches, sizes {(List.range 10).map (fun k => (k+1, (szN.filter (· == k+1)).length))}"
  for (nm, d) in [("forward", Vit.vDagFwd), ("step", Vit.vDagStep)] do
    IO.println s!"[vit] {nm} schedule: {d.nev} events, {(d.ewait.toList.map List.length).foldl (· + ·) 0} waits"

def Vit.Ship.main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  let clif ← Prog.orDie Vit.vClifIR
  emitArtifacts outDir (Vit.artifacts clif)
  report

#eval ShipScan.check "Vit.Ship" `Vit.Ship.main
