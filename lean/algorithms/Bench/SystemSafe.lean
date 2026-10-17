import AlgorithmLib.Check.Entry
import Bench.System
import AlgorithmLib.Proof.Typestate
import Host.StaticChecks

/-!
# The system benchmarks never misuse a library

Each workload of `system_bench`, proven by the condition generator from the
typestate it starts in.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds `Nat.mul` one step at a time and
overflows the stack.
-/

open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.Prog

namespace SystemBench


open AlgorithmLib.HProg AlgorithmLib.HProg.Sem

/-- Where `ht` may start: no worker running and no copy in flight, the arena
    holding the context slot and the key and value scratch, the input its
    count, the output its sum. -/
abbrev HtStart : World → Prop :=
  Contracts.TState.holds [.part .frozen false, .part .idle true, .room .arena 0xC0, .room .data 8,
    .room .out 8]

theorem ht_fine : Fine (emitGo (ht (V := Slot) (L := Lvl)) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨rfl, by decide⟩

set_option maxRecDepth 20000 in
/-- **`ht`, linked with `Lib.Ht`, never misuses a call**, however many
    entries the input asks for: each lookup's value fits the eight bytes it is
    read into, because every value inserted is eight bytes. Its calls to the
    hash table are calls to the library's functions, read by their statements
    (`Check.Linked`). -/
theorem ht_no_misuse (cfg : Cfg) [Check.Linked true cfg] (dataLen outLen : UInt64) (w : World) (h : HtStart w)
    (m : String) :
    run cfg (runArgs dataLen outLen) w (emit ht) ≠ .misuse m := by
  check_body

/-- Where `wc` may start: memory not frozen, the input a C string (the file's
    path), the arena holding the largest file it reads, the output its count. -/
abbrev WcStart : World → Prop :=
  Contracts.TState.holds [.part .frozen false, .cstr (regionBase .data), .room .arena (BUF + WC_MAX),
    .room .out 8]

theorem wc_fine : Fine (emitGo (wc (V := Slot) (L := Lvl)) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨rfl, by decide⟩

set_option maxRecDepth 20000 in
/-- **`wc` never misuses the file adapter**, whatever file its input names:
    the file is read into room for as much as it asks for. -/
theorem wc_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : WcStart w) (m : String) :
    run cfg (runArgs dataLen outLen) w (emit wc) ≠ .misuse m := by
  check_body

/-- The arguments `kv` is run with: the output has room for the scan. -/
def kvArgs : List V :=
  [.sc .i64 (regionBase .arena), .sc .i64 (regionBase .data), .sc .i64 64, .sc .i64 (regionBase .out),
   .sc .i64 0x100000]

/-- Where `kv` may start: memory not frozen, no LMDB context live, the input
    its count and then a C string (the directory), the arena holding the slot
    and the key and value scratch, the output room for the scan. -/
abbrev KvStart : World → Prop :=
  Contracts.TState.holds [.part .frozen false, .part .lmdb false, .cstr (regionBase .data + 8),
    .room .arena 0xC0, .room .data 8, .room .out 0x100000]

theorem kv_fine : Fine (emitGo (kv (V := Slot) (L := Lvl)) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨rfl, by decide⟩

set_option maxRecDepth 20000 in
/-- **`kv` never misuses the ordered store**, whatever the directory already
    holds: the scan writes only the rows that fit in the room it is given. -/
theorem kv_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : KvStart w) (m : String) :
    run cfg kvArgs w (emit kv) ≠ .misuse m := by
  check_body

end SystemBench
