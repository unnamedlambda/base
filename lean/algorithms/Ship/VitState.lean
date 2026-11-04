import AlgorithmLib.Check.DrvState
import Vit.Algorithm

/-!
# What the ViT's main leaves

The standard layout's state after `main` (`Drv.App.runTS`) but the default
stream's being the only one on record (`devSeq`): `main` runs a product on
each pooled stream. Past the layout, the capture stream and the pooled
streams, a cuBLAS handle on each pooled one, the events, and the host knowing
of every device access (`Pool`). `VitSafe.main_reaches` proves `main` leaves
it; the other entries start from it.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace VitState
open Contracts (Fact TState)

/-- What `main` makes past the standard layout: the capture stream and the
    pooled streams in their slots, a cuBLAS handle on each pooled stream, the
    events, everything the host issued done, and `graphs` graphs and `execs`
    instantiated ones made. -/
def Pool (graphs execs : Nat) : TState :=
  [.ngraphs graphs, .nexecs execs, .streams (List.replicate (Vit.VNSTRM + 1) true), .events Vit.VNEVENT,
   .nblas (Vit.VNSTRM + 1), .cell (arenaAt Vit.VSTREAM_OFF) (drvHandle kStream 0), .readyAll hostParty] ++
  (List.range Vit.VNSTRM).flatMap (fun s =>
    [.cell (arenaAt (Vit.vPoolOff s)) (drvHandle kStream (s + 1)),
     .cell (arenaAt (Vit.vHandleOff s)) (drvHandle kBlas (s + 1)),
     .blasOn (s + 1) (drvHandle kStream (s + 1))]) ++
  (List.range Vit.VNEVENT).flatMap (fun e =>
    [.evLive e, .cell (arenaAt (Vit.vEventOff e)) (drvHandle kEvent e)])

/-- The kernels' text, each kernel's in order, as the image places it. -/
def ptxs : List String := Vit.vKernUnits.map fun k => Vit.vPtx.getD (Vit.vSlotIx k) ""

/-- Which text and entry point each function runs. -/
def texts : TState :=
  (List.range Vit.vApp.kernels.length).map fun j => .fnText j (ptxs.getD j "") (Vit.vApp.entry j)

/-- What `main` leaves, with `graphs` graphs and `execs` instantiated ones
    made: kernels and cuBLAS keeping the sizes of the buffers they write, the
    caller's data holding the inputs, and each function the text and entry
    point it runs. -/
def after (graphs execs : Nat) : TState :=
  Vit.vApp.runTS.filter (· != .devSeq) ++ [.oracles, .room .data Vit.VHOST_BYTES] ++ Pool graphs execs ++ texts

/-- What `main` leaves (`VitSafe.main_reaches`): what the ranges and the
    captures start from. -/
abbrev Run : World → Prop := TState.holds (after 0 0)

/-- What `main` starts from: the driver loaded and untouched (`loadTS`),
    kernels and cuBLAS keeping the sizes of the buffers they write, and the
    caller's data holding the inputs. -/
abbrev Load : World → Prop :=
  TState.holds (Vit.vApp.loadTS VitState.ptxs ++ [.oracles, .room .data Vit.VHOST_BYTES])

/-- What `capture` leaves: the forward's graph instantiated, its handle in
    its slot. -/
abbrev Replay : World → Prop :=
  TState.holds (after 1 1 ++ [.exec 0, .cell (arenaAt Vit.VEXEC_FWD) (drvHandle kExec 0)])

end VitState
