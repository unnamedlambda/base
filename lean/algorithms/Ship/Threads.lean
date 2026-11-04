import AlgorithmLib.Check.Entry
import AlgorithmLib.Proof.Typestate
import Demo.ParSum

/-!
# A program that starts threads never misuses a call

`Demo.ParSum`'s `main`, whose input the caller writes into the arena, starts
four threads on its function `u0:2`, each owning a 1024-byte quarter, and
joins them. Both
sides are proven by the checker: `main` never misuses a call and never faults,
from the typestate that says no thread runs yet and which quarter each start
owns; and `u0:2`, run in a thread's view of any 1024 bytes (`Thread.view`),
never faults either. A thread's own loads and stores stay in its quarter
(`Confined.load`, `Confined.store`), and a join hands back what it left
(`ffiThreadJoin_honest`). What `main` computes is `Ship.ParSumSpec`.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace ShipThreads

/-- The quarters a start of `u0:2` owns: quarter `k`, for `k` below four,
    at its address. -/
def lends : List (Nat × UInt64 × Nat × Nat) :=
  [0, 1, 2, 3].map fun k => (2, regionBase .arena + UInt64.ofNat (1024 * k), 1024 * k, 1024)

/-- No thread runs yet, the world lends what `lends` says and makes threads,
    the arena has room, and the caller's data and output have the lengths it
    passes. -/
abbrev MainRun (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.threads 0 ParSum.MEM [] lends true, .room .arena ParSum.MEM, .roomArg .data dl, .roomArg .out ol]

set_option maxRecDepth 20000 in
set_option maxHeartbeats 1000000 in
theorem main_safe (cfg : Cfg) (dl ol : UInt64) (w : World) (h : MainRun dl ol w) (m : String) :
    Sem.run cfg (runArgs dl ol) w (emit ParSum.code) ≠ .misuse m ∧
    Sem.run cfg (runArgs dl ol) w (emit ParSum.code) ≠ .fault m := by
  check_body

/-- The thread's function as the program ships it. -/
def workerCode : Code := (runAns ParSum.worker [.i64]).2.1

/-- A thread's view of its bytes says it owns them. -/
theorem view_owns (t : Thread) (bs : ByteArray) (hb : bs.size = t.len) (h36 : t.off + t.len ≤ 2 ^ 36) :
    (Contracts.Fact.owns t.off t.len).holds (t.view bs) := by
  refine ⟨?_, h36, ?_⟩
  · show t.off + t.len ≤ (ByteArray.mk (Array.replicate t.off 0) ++ bs).size
    rw [ByteArray.size_append, hb]
    show t.off + t.len ≤ (Array.replicate t.off (0 : UInt8)).size + t.len
    rw [Array.size_replicate]; exact Nat.le_refl _
  · simp [Mem.lentClear, Thread.view]

set_option maxRecDepth 20000 in
/-- **Thread `k`'s function**, run where it owns its quarter, never misuses
    a call and never faults. -/
theorem worker_safe (cfg : Cfg) (k : Nat) (hk : k < 4) (w : World)
    (h : Contracts.TState.holds [.owns (1024 * k) 1024] w) (m : String) :
    Sem.run cfg [.sc .i64 (regionBase .arena + UInt64.ofNat (1024 * k))] w workerCode ≠ .misuse m ∧
    Sem.run cfg [.sc .i64 (regionBase .arena + UInt64.ofNat (1024 * k))] w workerCode ≠ .fault m := by
  obtain rfl | rfl | rfl | rfl : k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 := by omega
  all_goals check_body

/-- **The thread `main` starts on quarter `k`**, run in its view of any
    1024 bytes, never misuses a call and never faults. -/
theorem worker_in_view (cfg : Cfg) (k : Nat) (hk : k < 4) (bs : ByteArray) (hb : bs.size = 1024) (m : String) :
    let t : Thread := ⟨2, regionBase .arena + UInt64.ofNat (1024 * k), 1024 * k, 1024⟩
    Sem.run cfg [.sc .i64 t.arg] (t.view bs) workerCode ≠ .misuse m ∧
    Sem.run cfg [.sc .i64 t.arg] (t.view bs) workerCode ≠ .fault m :=
  worker_safe cfg k hk _ (fun f hf => by
    simp only [List.mem_singleton] at hf
    subst hf
    exact view_owns ⟨2, regionBase .arena + UInt64.ofNat (1024 * k), 1024 * k, 1024⟩ bs hb (show 1024 * k + 1024 ≤ 2 ^ 36 by omega)) m

end ShipThreads

#print axioms ShipThreads.main_safe
#print axioms ShipThreads.worker_in_view
