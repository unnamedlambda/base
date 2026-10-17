import AlgorithmLib.Surface.Link
import AlgorithmLib.Host.Program
import AlgorithmLib.Proof.LibSpec

/-!
# The hash table, run both ways

One driver over the table, run in the semantics twice against the functions
of `Surface.LibHt`: calling them as the entry points they implement, the way a
program is written, and calling them as the program's own functions, the way
it is linked. It makes two handles, inserts keys
until the entry array, the index and the arena have each grown, rewrites
values at their old length and at a longer one, looks up keys present and
absent, increments counters, walks the entries in order and cleans up. Both
runs leave the same answers, and no allocation live.

A failed allocation is not reached here: the model's heap is full only at
64 GiB of pinned memory. What the library does then is `Ship.HtSpec`'s to
prove.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace HtDiff

variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

/-- The table's functions, as a driver calls them. Every answer is 64 bits,
    a status in its low 32. -/
structure Api (V : ClifTy → Type) (L : List ClifTy → List ClifTy → Type) where
  init : V .i64 → Prog V L Unit
  cleanup : V .i64 → Prog V L Unit
  create : V .i64 → Prog V L (V .i64)
  insert : V .i64 → V .i64 → V .i32 → V .i64 → V .i32 → Prog V L Unit
  lookup : V .i64 → V .i64 → V .i32 → V .i64 → Prog V L (V .i64)
  increment : V .i64 → V .i64 → V .i32 → V .i64 → Prog V L (V .i64)
  count : V .i64 → Prog V L (V .i64)
  getEntry : V .i64 → V .i32 → V .i64 → V .i64 → Prog V L (V .i64)

/-- The table called by its entry points, each a call to the library's
    function at the entry point's place (`Ffi.libSlot`). -/
def entry : Api V L where
  init s := ffiVoid .htInit %[s]
  cleanup s := ffiVoid .htCleanup %[s]
  create c := do uextend64 (← ffi .htCreate %[c])
  insert c k kl v vl := ffiVoid .htInsert %[c, k, kl, v, vl]
  lookup c k kl o := do uextend64 (← ffi .htLookup %[c, k, kl, o])
  increment c k kl a := ffi .htIncrement %[c, k, kl, a]
  count c := do uextend64 (← ffi .htCount %[c])
  getEntry c i ko vo := do uextend64 (← ffi .htGetEntry %[c, i, ko, vo])

def lo32 (x : V .i64) : Prog V L (V .i64) := do band x (← iconst64 0xFFFFFFFF)

def lib : Api V L where
  init s := callLocalVoid (⟨.local 1⟩ : LocalRef [.i64] none) %[s]
  cleanup s := callLocalVoid (⟨.local 2⟩ : LocalRef [.i64] none) %[s]
  create c := do lo32 (← callLocal (⟨.local 3⟩ : LocalRef [.i64] (some .i64)) %[c])
  insert c k kl v vl := callLocalVoid (⟨.local 4⟩ : LocalRef [.i64, .i64, .i32, .i64, .i32] none) %[c, k, kl, v, vl]
  lookup c k kl o := do lo32 (← callLocal (⟨.local 5⟩ : LocalRef [.i64, .i64, .i32, .i64] (some .i64)) %[c, k, kl, o])
  increment c k kl a := callLocal (⟨.local 6⟩ : LocalRef [.i64, .i64, .i32, .i64] (some .i64)) %[c, k, kl, a]
  count c := do lo32 (← callLocal (⟨.local 7⟩ : LocalRef [.i64] (some .i64)) %[c])
  getEntry c i ko vo := do lo32 (← callLocal (⟨.local 8⟩ : LocalRef [.i64, .i32, .i64, .i64] (some .i64)) %[c, i, ko, vo])

/-- The library's function for an entry point, as one of the program's: the
    function `Ship.HtSpec`'s `_calls` theorems speak of. -/
def fnOf (f : Ffi) : Option Fn := (LibHt.implOf f).map (LibSpec.implFn f.params)

/-- The program's functions: the library's, at the places a linked program
    calls them and at the places their entry points name. -/
def fns : Nat → Option Fn
  | 1 => fnOf .htInit | 2 => fnOf .htCleanup | 3 => fnOf .htCreate | 4 => fnOf .htInsert
  | 5 => fnOf .htLookup | 6 => fnOf .htIncrement | 7 => fnOf .htCount | 8 => fnOf .htGetEntry
  | i => LibSpec.libFnsOf LibHt.implOf i

-- arena: slot 0, key 16, value 32..96, out 128..192, kout 256, vout 320, results from 512
def driver (n : Nat) (api : Api V L) : Prog V L Unit := do
  let base ← basePtr
  let slot ← absAddr base 0
  let key ← absAddr base 16
  let val ← absAddr base 32
  let out ← absAddr base 128
  let kout ← absAddr base 256
  let vout ← absAddr base 320
  let res ← absAddr base 512
  let k8 ← iconst32 8
  let mix ← iconst64 31
  let mul ← iconst64 0x9E3779B97F4A7C15
  -- before a context: every accessor refuses
  let a0 ← api.count (← iconst64 0)
  api.init slot
  let c ← load64 slot
  let a1 ← api.count c
  let a2 ← api.create c
  let a3 ← api.create c
  -- n keys, each a 40-byte value whose first word is the key's index
  forLoop (← iconst64 n) fun i => do
    storeI64 (← imul (← iaddImm i 1) mul) key
    storeI64 i val
    storeI64 (← bxor i mul) (← iaddImm val 32)
    api.insert c key k8 val (← iconst32 40)
  -- every third key again: the same length, then a longer one
  let third ← iconst64 (n / 3)
  forLoop third fun j => do
    let i ← imul j (← iconst64 3)
    storeI64 (← imul (← iaddImm i 1) mul) key
    storeI64 (← iaddImm i 1000) val
    api.insert c key k8 val (← iconst32 40)
    storeI64 (← iaddImm i 2000) val
    api.insert c key k8 val (← iconst32 56)
  -- look every key up, and some that are absent
  let acc ← forLoopAcc (← iconst64 (n + 20)) a0 fun i acc => do
    storeI64 (← imul (← iaddImm i 1) mul) key
    let r ← api.lookup c key k8 out
    let acc ← iadd (← imul acc mix) r
    iadd (← imul acc mix) (← load64 out)
  -- counters, new and old
  let acc ← forLoopAcc (← iconst64 40) acc fun i acc => do
    storeI64 (← imul (← iaddImm (← imul i (← iconst64 5)) 1) mul) key
    let r ← api.increment c key k8 (← iaddImm i 7)
    iadd (← imul acc mix) r
  -- the entries in order
  let cnt ← api.count c
  let acc ← forLoopAcc (← iaddImm cnt 3) acc fun i acc => do
    let r ← api.getEntry c (← ireduce32 i) kout vout
    let acc ← iadd (← imul acc mix) r
    let acc ← iadd (← imul acc mix) (← load64 kout)
    iadd (← imul acc mix) (← load64 vout)
  storeI64 acc res
  storeI64 a1 (← iaddImm res 8)
  storeI64 a2 (← iaddImm res 16)
  storeI64 a3 (← iaddImm res 24)
  storeI64 cnt (← iaddImm res 32)
  api.cleanup slot
  storeI64 (← load64 slot) (← iaddImm res 40)

def world0 : World :=
  { mem := { arena := ByteArray.mk (Array.replicate 1024 0), data := ByteArray.mk (Array.replicate 8 0),
             out := ByteArray.mk (Array.replicate 64 0) } }

def args : List Sem.V :=
  [.sc .i64 (Sem.regionBase .arena), .sc .i64 (Sem.regionBase .data), .sc .i64 8,
   .sc .i64 (Sem.regionBase .out), .sc .i64 64]

def results (n : Nat) (api : Api Slot Lvl) : String :=
  let body : Body := driver n api
  let env := (Prog.run body).2.1
  match Sem.run { env, locals := termLocals fns env 100000000 2 } args world0 (Prog.emit body) with
  | .ok _ w => toString ((List.range 6).map fun i => w.mem.load (Sem.regionBase .arena + UInt64.ofNat (512 + 8 * i)) 8)
      ++ s!" heap={w.heap.length} live={w.mem.pinnedLive.length}"
  | .stuck m => "stuck: " ++ m
  | .misuse m => "misuse: " ++ m
  | .fault m => "fault: " ++ m

end HtDiff

#guard HtDiff.results 200 HtDiff.entry == HtDiff.results 200 HtDiff.lib
#guard (HtDiff.results 200 HtDiff.lib).endsWith "heap=0 live=0"
#guard !(HtDiff.results 200 HtDiff.lib).startsWith "stuck" && !(HtDiff.results 200 HtDiff.lib).startsWith "misuse" &&
  !(HtDiff.results 200 HtDiff.lib).startsWith "fault"
