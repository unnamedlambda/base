import AlgorithmLib.Check.Entry
import AlgorithmLib.Check.Seq
import AlgorithmLib.Proof.Typestate
import AlgorithmLib.Surface.ProgFFI

/-!
# A program's own calls, by their callees' summaries

A wrapper that calls two of the program's functions neither misuses a call nor
faults where its hypotheses give each callee's `Summary`: the generator reads
each call by the summary it finds for that function on those arguments
(`wp_local_void`). A callee's summary comes from its own condition with the
post that the world it leaves holds the summary's typestate
(`triple_of_wp_entry`, then `summary_succ`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace LocalCalls

abbrev SW : World → Prop := Contracts.TState.holds [.room .arena 4096]

def wrap : Body := sequenceWrapper [5, 6]

theorem wrap_fine : Fine (emitGo (fnBody ptrParams (wrap : Prog Slot Lvl Unit)) ⟨4, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxHeartbeats 400000 in
theorem wrap_no_misuse (cfg : Cfg) (dl ol : UInt64)
    (h5 : Summary cfg.locals 5 (runArgs dl ol) SW SW) (h6 : Summary cfg.locals 6 (runArgs dl ol) SW SW)
    (w : World) (hw : SW w) (m : String) :
    Sem.run cfg (runArgs dl ol) w (emit (wrap : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dl ol) w (emit (wrap : Prog Slot Lvl Unit)) ≠ .fault m := by
  rw [show (emit (wrap : Prog Slot Lvl Unit)) = [.straight (.op .memory :: Check.seqCalls [5, 6])] from rfl]
  exact Check.seqCalls_run (cs := [5, 6]) ⟨_, h5, _, h6, trivial⟩ hw m

/-- A leaf that stores into the arena: its summary, from its condition. -/
def leaf : Body := do
  let p ← basePtr
  let v ← iconst64 7
  storeI64 v (← absAddr p 8)

theorem leaf_fine : Fine (emitGo (fnBody ptrParams (leaf : Prog Slot Lvl Unit)) ⟨4, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxHeartbeats 400000 in
theorem leaf_triple (cfg : Cfg) (dl ol : UInt64) (w : World) (hw : SW w) :
    Hoare.Triple cfg (Hoare.At (runArgs dl ol).toArray w) (emit (leaf : Prog Slot Lvl Unit))
      { ok := fun _ w' => SW w', faultOk := false } := by
  check_reach

end LocalCalls

#print axioms LocalCalls.wrap_no_misuse
#print axioms LocalCalls.leaf_triple
