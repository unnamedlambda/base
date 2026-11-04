import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.DrvState
import AlgorithmLib.Proof.Typestate
import Warp.MlpCifar
import Warp.MlpCifarBodies
import AlgorithmLib.MLHost.TapeRun

/-!
# The mixture-of-experts dispatch neither misuses a call nor faults

`main` brings the CUDA driver up, makes the 123 allocations, loads the
seventeen kernels and writes their parameters in the driver's standard layout
(`Drv.App`), and uploads the shared inputs and every expert's weights from the
caller's data; every other entry starts from what it leaves (`Drv.App.runTS`)
but the cells `bindExperts` and a rebinding launch rewrite (`moved`): the
slots of the six buffers whose pointers it moves and the parameter cells of
every block that binds one. So each entry is proven whichever experts were
chosen. `bindExperts` calls nothing: it needs only the arena and the caller's
choice, read through the length it is handed (`roomArg`); an upload asks for
the bytes it reads from the caller's data, a fetch for those it writes to the
caller's output.

Not faulting means every load and store finds the bytes it reaches,
and every operation answers.

`main` is checked by compiled evaluation (`ship_entry`), so its theorems
also rest on the compiler.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace MoeSafe

abbrev Load : World → Prop := Contracts.TState.holds
  (MlpCifar.mApp.loadTS ++ [.room .data MlpCifar.MHOST_BYTES])

/-- The arena offsets `bindExperts` and a rebinding launch write: the slots of
    the buffers whose pointers it moves, and the parameter cells of every
    block that binds one. -/
def moved : List Nat :=
  ((List.range (3 * MlpCifar.MUSED)).map fun t => MlpCifar.mApp.bufSlot (MlpCifar.MSLOT0 + t)) ++
  ((List.range MlpCifar.mApp.blocks.length).zip MlpCifar.mApp.blocks).flatMap fun (b, args) =>
    if args.any (fun x => MlpCifar.MSLOT0 ≤ x && x < MlpCifar.MBASE) then
      (List.range args.length).map fun p => MlpCifar.mApp.blockOff b + 8 * p
    else []

/-- What the entries after `bindExperts` start from: what `main` leaves but
    the cells it moves. -/
def runState (d o : Nat) : Contracts.TState :=
  (MlpCifar.mApp.runTS.filter fun
      | .cell a _ => !(moved.any fun off => a == arenaAt off)
      | _ => true) ++ [.room .data d, .room .out o]

abbrev Run (d o : Nat) : World → Prop := Contracts.TState.holds (runState d o)

/-- The arena's size, which the kernel computes from the layout. -/
theorem mem_size : MlpCifar.MMEM_SIZE = 565248 := by decide +kernel

/-- What `bindExperts` reads and writes: the binding table in the arena, and
    the caller's choice through the length it hands over. It calls nothing.
    The arena's size is its number: an address over the caller's choice is
    bounded by arithmetic, which reads a size only as a number. -/
abbrev Bind (dl : UInt64) : World → Prop := Contracts.TState.holds
  [.room .arena 565248, .roomArg .data dl]

ship_entry main : MlpCifar.mLoadFn under Load

ship_entry runRouter : (MlpCifar.mRunRange 0 MlpCifar.mRouterTape.length) under ((Run 0 0))

/-- The kernels `runRouter` launches, in order. -/
def routerSteps : List Nat := List.range MlpCifar.mRouterTape.length

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
/-- Every run of `runRouter` that ends launches the router's kernels in
    order on the default stream, then waits, the device changed only by
    those calls. -/
theorem runRouter_calls (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run 0 0 w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w
      (emit (MlpCifar.mRunRange 0 MlpCifar.mRouterTape.length : Prog Slot Lvl Unit)) = .ok obs w') :
    (MlpCifar.mApp.runCalls routerSteps).Covers w obs ∧ Check.DevPath w w' := by
  check_calls

theorem bytes_le : MlpCifar.moeLayout.bytes.all (· ≤ 2 ^ 34) = true := by native_decide
theorem router_ok : ML.kernelsOk MlpCifar.moeLayout MlpCifar.mApp MlpCifar.moeUnit routerSteps = true := by
  native_decide
theorem run_ready : (ML.readyFacts MlpCifar.mApp).all (runState 0 0).contains = true := by native_decide
theorem router_units : routerSteps.flatMap MlpCifar.moeUnit = MlpCifar.mRouterTape := by
  exact ML.flatMap_range_take MlpCifar.mRouterTape MlpCifar.mExpertTape

/-- **The words every run of `runRouter` leaves are the router's tape's**:
    from what `main` leaves, whichever experts were bound, on a device honest
    about each launch (`TapeLayout.KernelHonest`), every word inside each
    allocation, at its size, is as the router's operations leave it from the
    memory the run started with. -/
theorem runRouter_tape (hw : Kernel.Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run 0 0 w)
    {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w
      (emit (MlpCifar.mRunRange 0 MlpCifar.mRouterTape.length : Prog Slot Lvl Unit)) = .ok obs w')
    (hHon : ∀ j ∈ routerSteps, MlpCifar.moeLayout.KernelHonest MlpCifar.mApp w j (MlpCifar.moeUnit j) hw) :
    MlpCifar.moeLayout.Agree (Device.devMem w'.dev)
      (MlpCifar.mRouterTape.foldl (fun M op => ML.TOp.denV hw op M) (ML.wordView (Device.devMem w.dev))) := by
  rw [← router_units]
  exact ML.kernels_tape _ _ bytes_le hw hw32 _ _ router_ok (ML.ready_of h run_ready) hr
    (runRouter_calls cfg dataLen outLen w h hr) hHon

ship_entry fetchGate : (MlpCifar.mFetchFn MlpCifar.MGATE (MlpCifar.NE * 4)) under ((Run 0 (MlpCifar.NE * 4)))

theorem bindExperts_fine : Fine (emitGo (fnBody ptrParams (MlpCifar.mBindExperts : Prog Slot Lvl Unit)) ⟨4, 0, [], [], [], none⟩).2 :=
  ⟨by native_decide, by native_decide⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem bindExperts_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : Contracts.TState.holds [.room .arena MlpCifar.MMEM_SIZE, .roomArg .data dataLen] w)
    (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.mBindExperts : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (MlpCifar.mBindExperts : Prog Slot Lvl Unit)) ≠ .fault m := by
  rw [mem_size] at h
  check_body

theorem uploadGates_fine : Fine (emitGo (fnBody ptrParams ((MlpCifar.mUploadFn 3 128) : Prog Slot Lvl Unit)) ⟨4, 0, [], [], [], none⟩).2 :=
  ⟨by native_decide, by native_decide⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem uploadGates_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : (Run 128 0) w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit ((MlpCifar.mUploadFn 3 128) : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit ((MlpCifar.mUploadFn 3 128) : Prog Slot Lvl Unit)) ≠ .fault m := by
  check_body

ship_entry runExperts : (MlpCifar.mRunRange MlpCifar.mRouterTape.length MlpCifar.moeTape.length) under ((Run 0 0))

ship_entry fetchOut : (MlpCifar.mFetchFn MlpCifar.MOUT (MlpCifar.MD * 4)) under ((Run 0 (MlpCifar.MD * 4)))

end MoeSafe

#print axioms MoeSafe.main_no_misuse
#print axioms MoeSafe.runRouter_no_misuse
#print axioms MoeSafe.runRouter_tape
#print axioms MoeSafe.fetchGate_no_misuse
#print axioms MoeSafe.bindExperts_no_misuse
#print axioms MoeSafe.uploadGates_no_misuse
#print axioms MoeSafe.runExperts_no_misuse
#print axioms MoeSafe.fetchOut_no_misuse
