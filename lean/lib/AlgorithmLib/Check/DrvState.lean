module
public import AlgorithmLib.Surface.Drv
meta import AlgorithmLib.Surface.Drv
public import AlgorithmLib.Host.Lifecycle
meta import AlgorithmLib.Host.Lifecycle
import all Init.Data.Repr
@[expose] public section

/-!
# What a program in the standard layout starts from

The typestates a theorem about a program in `Drv.App`'s layout states, in
the driver's own terms. `loadTS` is what `main` starts from: the driver
loaded and untouched, the entry points' names and each module's PTX in the
arena, and the claim each PTX makes that each entry point in it takes one
pointer per allocation its kernel is handed. `runTS` is what `main` leaves, which the
other entry points start from (`runTSWith`, where `main` sizes allocations
by what the caller hands it): the context current, every allocation live
with its pointer in its slot, every function looked up with its handle in
its slot, every parameter block's cells and array written, and for a
program that calls cuBLAS, its handle in its slot on the default stream.
-/

namespace AlgorithmLib.Drv.App

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Contracts (Fact TState)

/-- The arena address `off` bytes in. -/
abbrev at_ (off : Nat) : UInt64 := regionBase .arena + UInt64.ofNat off

/-- The parameter sizes a kernel's entry point declares. -/
def sizes (k : KSpec) : List Nat := k.args.map fun _ => 8

/-- What `main` starts from. -/
def loadTS (a : App) : TState :=
  [.part (.has .cuda) true, .part .capturing false, .part .idle true, .devSeq,
   .retained 0, .nbufs 0, .nmods 0, .nfuncs 0, .room .arena a.memSize,
   .text (at_ a.NAME) (some "main") 5 []] ++
  (if a.blas then [.part (.has .cublas) true, .nblas 0] else []) ++
  (a.indexed.filterMap fun (j, _) =>
    if a.entry j == "main" then none else some (.text (at_ (a.nameOff j)) (some (a.entry j)) 32 [])) ++
  a.indexed.filterMap fun (j, k) =>
    if a.modOwner j == j then
      some (.text (at_ k.ptxOff) none k.ptxLen
        (a.indexed.filterMap fun (j', k') => if a.modOwner j' == j then some (a.entry j', sizes k') else none))
    else none

/-- What the other entry points start from: what `main` leaves, its
    allocations of the sizes `szs`. -/
def runTSWith (a : App) (szs : List Nat) : TState :=
  [.part (.has .cuda) true, .part .capturing false, .part .idle true, .part .cuda true,
   .devSeq, .oracles, .room .arena a.memSize] ++
  (if a.blas then [.part (.has .cublas) true, .blasOn 0 0, .cell (at_ a.BLAS) (drvHandle kBlas 0)] else []) ++
  szs.zipIdx.flatMap (fun (n, i) =>
    [.alloc i n, .cell (at_ (a.bufSlot i)) (devPtr i)]) ++
  (a.indexed.flatMap fun (j, k) => [.fn j (sizes k), .cell (at_ (a.fnSlot j)) (drvHandle kFunc j)]) ++
  ((List.range a.blocks.length).zip a.blocks).flatMap fun (b, args) =>
    ((List.range args.length).zip args).flatMap fun (p, i) =>
      [.cell (at_ (a.blockOff b + 8 * p)) (devPtr i), .cell (at_ (a.blockOff b + a.arrOff + 8 * p)) (at_ (a.blockOff b + 8 * p))]

/-- What the other entry points start from: what `main` leaves, its
    allocations of the sizes `bufs` gives. -/
def runTS (a : App) : TState := a.runTSWith a.bufs

end AlgorithmLib.Drv.App
