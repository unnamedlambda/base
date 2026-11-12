module
public import Lean
public import Scan.Ship
meta import Scan.Ship
public import AlgorithmLib.Surface.Link
meta import AlgorithmLib.Surface.Link
public import AlgorithmLib.Surface.Drv
meta import AlgorithmLib.Surface.Drv
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
public import AlgorithmLib.Kernel.Block
meta import AlgorithmLib.Kernel.Block
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# base's analytics engine: ClickBench's numeric queries on the GPU

`benchmarks/analytics` measures these beside DuckDB on the first 10 million
rows of ClickBench's `hits`. The columns live on the device; each query is
its own pipeline of kernels (as a compiling engine makes one), every block
folding its stride of rows into a partial and one block folding the partials,
with no atomic operation, so an answer is a function of the rows alone.

Its functions (the caller's data, then what each does):

* `load`: `n` (`u64`, below `2^31`), then the columns `AdvEngineID`
  (`i16`), `ResolutionWidth` (`i16`), `UserID` (`i64`), `EventDate` (`u16`),
  `RegionID` (`i32`), each `n` long, padded to 8 bytes: up to the device;
  nothing is taken from an input too short to hold them.
* `q1`: `COUNT(*) WHERE AdvEngineID <> 0`: the count.
* `q2`: `SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth)`: the two sums and
  `n`, the caller dividing.
* `q3`: `AVG(UserID)`: the sums of the high and the low 32 bits of every
  `UserID` (so the total is exact, as DuckDB's 128-bit sum is) and `n`.
* `q6`: `MIN(EventDate), MAX(EventDate)`.
* `q7`: `AdvEngineID, COUNT(*) WHERE AdvEngineID <> 0 GROUP BY AdvEngineID`:
  256 counts, one an `AdvEngineID`, the caller ordering them; count 0 holds
  the rows whose `AdvEngineID` is 256 or more, which the caller refuses.

Every function writes its answer as `i64` words and ends synchronised.
-/

set_option maxRecDepth 8000

namespace Analytics

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.Kernel AlgorithmLib.Kernel.Block

/-- Threads a block, and their warps. -/
def T : Nat := 256
def WARPS : Nat := 8

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

/-- How a fold combines two partials: sum, unsigned minimum, unsigned maximum. -/
inductive Op | sum | min | max

def Op.apply (o : Op) (a b : V .i64) : Prog V L (V .i64) :=
  match o with
  | .sum => iadd a b
  | .min => umin a b
  | .max => umax a b

/-- **`x` folded over the block**: each thread's word in shared memory at
    `sm + off`, then halves folded into the lower until thread 0 holds the
    whole, a barrier between steps. -/
def blockFold (o : Op) (tid : V .i32) (sm : V .i64) (off : Nat) (x : V .i64) : Prog V L (V .i64) := do
  let slot ← iadd (← iaddImm sm off) (← ishlImm (← ext tid) 3)
  storeI64 x slot
  for s in [128, 64, 32, 16, 8, 4, 2, 1] do
    callLocalVoid barrier %[]
    when .ult tid (← c32 s) do
      storeI64 (← o.apply (← load64 slot) (← load64 (← iaddImm slot (8 * s)))) slot
  callLocalVoid barrier %[]
  load64 (← iaddImm sm off)

/-- The rows thread `t` of `stride` takes: `t, t + stride, …` below `n`,
    carrying `k` words. -/
def rows (start n stride : V .i32) (init : Vals V (List.replicate k .i64))
    (step : V .i32 → Vals V (List.replicate k .i64) → Prog V L (Vals V (List.replicate k .i64))) :
    Prog V L (Vals V (List.replicate k .i64)) := do
  let r ← wloop (%[start] |>.append init) (fun cs => pure (contIfULt cs.head n, cs.tail, ()))
    (fun cs _ => do pure (%[← iadd cs.head stride] |>.append (← step cs.head cs.tail)))
  pure r

/-- The block's first thread, and this thread's place in the grid. -/
def place (cta tid : V .i32) : Prog V L (V .i32) := do iadd (← imul cta (← c32 T)) tid

/-- **Rows where an `i16` column is not 0, counted**, a partial a block.
    `meta`: `n`, the grid's threads. -/
def countNeBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64])
  let cta := ps.head; let tid := ps.snd; let sm := (ps.drop 3).head
  let col := (ps.drop 4).head; let out := (ps.drop 5).head; let mt := (ps.drop 6).head
  let n ← load32 mt
  let stride ← load32 (← at_ mt 4)
  let r ← rows (k := 1) (← place cta tid) n stride %[← iconst64 0] fun i cs => do
    let v ← sextend64 (← load_i16 (← iadd col (← ishlImm (← ext i) 1)))
    pure %[← iadd cs.head (← select (← icmp .ne v (← iconst64 0)) (← iconst64 1) (← iconst64 0))]
  let tot ← blockFold .sum tid sm 0 r.head
  when .eq tid (← c32 0) do storeI64 tot (← iadd out (← ishlImm (← ext cta) 3))

/-- **Two `i16` columns summed**, a partial each a block. `meta`: `n`, the
    grid's threads. -/
def sum2Body : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64, .i64])
  let cta := ps.head; let tid := ps.snd; let sm := (ps.drop 3).head
  let a := (ps.drop 4).head; let b := (ps.drop 5).head
  let outA := (ps.drop 6).head; let outB := (ps.drop 7).head; let mt := (ps.drop 8).head
  let n ← load32 mt
  let stride ← load32 (← at_ mt 4)
  let zero ← iconst64 0
  let r ← rows (k := 2) (← place cta tid) n stride %[zero, zero] fun i cs => do
    let o ← ishlImm (← ext i) 1
    pure %[← iadd cs.head (← sextend64 (← load_i16 (← iadd a o))), ← iadd cs.snd (← sextend64 (← load_i16 (← iadd b o)))]
  let ta ← blockFold .sum tid sm 0 r.head
  let tb ← blockFold .sum tid sm (8 * T) r.snd
  when .eq tid (← c32 0) do
    let at_ ← ishlImm (← ext cta) 3
    storeI64 ta (← iadd outA at_)
    storeI64 tb (← iadd outB at_)

/-- **An `i64` column summed exactly**: its high 32 bits (signed) and its low
    32 bits (unsigned) summed apart, neither overflowing below `2^31` rows,
    a partial each a block. `meta`: `n`, the grid's threads. -/
def sumSplitBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64])
  let cta := ps.head; let tid := ps.snd; let sm := (ps.drop 3).head
  let col := (ps.drop 4).head; let outHi := (ps.drop 5).head; let outLo := (ps.drop 6).head
  let mt := (ps.drop 7).head
  let n ← load32 mt
  let stride ← load32 (← at_ mt 4)
  let zero ← iconst64 0
  let r ← rows (k := 2) (← place cta tid) n stride %[zero, zero] fun i cs => do
    let x ← load64 (← iadd col (← ishlImm (← ext i) 3))
    pure %[← iadd cs.head (← sextend64 (← ireduce32 (← ushrImm x 32))), ← iadd cs.snd (← band x (← iconst64 0xFFFFFFFF))]
  let th ← blockFold .sum tid sm 0 r.head
  let tl ← blockFold .sum tid sm (8 * T) r.snd
  when .eq tid (← c32 0) do
    let at_ ← ishlImm (← ext cta) 3
    storeI64 th (← iadd outHi at_)
    storeI64 tl (← iadd outLo at_)

/-- **A `u16` column's minimum and maximum**, a partial each a block (the
    minimum of no rows `2^16`, the maximum 0). `meta`: `n`, the grid's
    threads. -/
def minMaxBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64])
  let cta := ps.head; let tid := ps.snd; let sm := (ps.drop 3).head
  let col := (ps.drop 4).head; let outMin := (ps.drop 5).head; let outMax := (ps.drop 6).head
  let mt := (ps.drop 7).head
  let n ← load32 mt
  let stride ← load32 (← at_ mt 4)
  let r ← rows (k := 2) (← place cta tid) n stride %[← iconst64 0x10000, ← iconst64 0] fun i cs => do
    let v ← uload16_64 (← iadd col (← ishlImm (← ext i) 1))
    pure %[← umin cs.head v, ← umax cs.snd v]
  let tmin ← blockFold .min tid sm 0 r.head
  let tmax ← blockFold .max tid sm (8 * T) r.snd
  when .eq tid (← c32 0) do
    let at_ ← ishlImm (← ext cta) 3
    storeI64 tmin (← iadd outMin at_)
    storeI64 tmax (← iadd outMax at_)

/-- **The partials folded**, one block: `meta`: the partials' count, the fold
    (`0` sum, `1` minimum, `2` maximum). -/
def finishBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64])
  let tid := ps.snd; let sm := (ps.drop 3).head
  let part := (ps.drop 4).head; let out := (ps.drop 5).head; let mt := (ps.drop 6).head
  let cnt ← load32 mt
  let op ← load32 (← at_ mt 4)
  let isMin ← icmp .eq op (← c32 1)
  let isMax ← icmp .eq op (← c32 2)
  let ident ← select isMin (← iconst64 0x7FFFFFFFFFFFFFFF) (← iconst64 0)
  let fold (a b : V .i64) : Prog V L (V .i64) := do
    select isMin (← umin a b) (← select isMax (← umax a b) (← iadd a b))
  let r ← rows (k := 1) tid cnt (← c32 T) %[ident] fun i cs => do
    pure %[← fold cs.head (← load64 (← iadd part (← ishlImm (← ext i) 3)))]
  -- the block fold by the requested op: a sum's tree folds the selected values
  let slot ← iadd sm (← ishlImm (← ext tid) 3)
  storeI64 r.head slot
  for s in [128, 64, 32, 16, 8, 4, 2, 1] do
    callLocalVoid barrier %[]
    when .ult tid (← c32 s) do
      storeI64 (← fold (← load64 slot) (← load64 (← iaddImm slot (8 * s)))) slot
  callLocalVoid barrier %[]
  when .eq tid (← c32 0) do storeI64 (← load64 sm) out

/-- **An `i16` column's nonzero values counted by value**, 256 counts a
    block: each warp counts 32 rows at a time into its own histogram in
    shared memory (a value's count added by the lowest lane holding it), the
    warps' histograms summed at the end. A value of 256 or more is counted
    at 0. Every thread of the block makes the same trips, so its ballots and
    barriers meet.
    `meta`: `n`, the grid's threads. -/
def hist8Body : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64])
  let cta := ps.head; let tid := ps.snd; let sm := (ps.drop 3).head
  let col := (ps.drop 4).head; let out := (ps.drop 5).head; let mt := (ps.drop 6).head
  let n ← load32 mt
  let stride ← load32 (← at_ mt 4)
  let lane ← band tid (← c32 31)
  let warp ← ushrImm tid 5
  stridedDo tid (← c32 (WARPS * 256)) T fun j => do storeI32 (← c32 0) (← at4 sm j)
  callLocalVoid barrier %[]
  let hist ← iadd sm (← ishlImm (← ext (← imul warp (← c32 256))) 2)
  let zero ← c32 0
  -- the block's rows a trip, so every warp makes the same trips and meets the
  -- same barriers; a row past `n` is no row
  let _ ← wloop %[← imul cta (← c32 T)] (fun cs => pure (contIfULt cs.head n, %[], ()))
    (fun cs _ => do
      let i ← iadd cs.head tid
      let inN ← icmp .ult i n
      let v ← sextend64 (← load_i16 (← iadd col (← ishlImm (← ext (← select inN i zero)) 1)))
      let v32 ← ireduce32 v
      let nz ← icmp .ne (← band (← bit inN) (← bit (← icmp .ne v32 zero))) zero
      let small ← icmp .ult v32 (← c32 256)
      let bin ← select small v32 zero
      let m ← peers bin nz
      when .ne (← bit nz) zero do
        when .eq (← ctz m) lane do
          let slot ← at4 hist bin
          storeI32 (← iadd (← load32 slot) (← popcnt m)) slot
      callLocalVoid barrier %[]
      pure %[← iadd cs.head stride])
  callLocalVoid barrier %[]
  let mut sum ← c32 0
  for w in List.range WARPS do
    sum ← iadd sum (← load32 (← at4 sm (← iadd tid (← c32 (256 * w)))))
  storeI64 (← uextend64 sum) (← iadd out (← ishlImm (← ext (← iadd (← imul cta (← c32 256)) tid)) 3))

/-- **The blocks' histograms summed**, a thread a value: `meta`: blocks. -/
def histFinishBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64])
  let tid := ps.snd
  let part := (ps.drop 4).head; let out := (ps.drop 5).head; let mt := (ps.drop 6).head
  let blocks ← load32 mt
  let r ← rows (k := 1) (← c32 0) blocks (← c32 1) %[← iconst64 0] fun b cs => do
    pure %[← iadd cs.head (← load64 (← iadd part (← ishlImm (← ext (← iadd (← imul b (← c32 256)) tid)) 3)))]
  storeI64 r.head (← iadd out (← ishlImm (← ext tid) 3))

end

def k (name : String) (body : Prog Slot Lvl Unit) (np : Nat) (smem : Nat := 0) : Kernel :=
  { name, params := List.replicate np .i64, code := emit body (idParams ++ List.replicate np .i64), smem }

/-- The kernels, each its own module, by index. -/
def kernels : List Kernel :=
  [k "an_count_ne" countNeBody 3 (8 * T), k "an_sum2" sum2Body 5 (16 * T), k "an_sum_split" sumSplitBody 4 (16 * T),
   k "an_min_max" minMaxBody 4 (16 * T), k "an_finish" finishBody 3 (8 * T), k "an_hist8" hist8Body 3 (4 * 256 * WARPS),
   k "an_hist_finish" histFinishBody 3]

/-! ## The layout -/

def ADV : Nat := 0
def RES : Nat := 1
def USER : Nat := 2
def DATE : Nat := 3
def REGION : Nat := 4
def PA : Nat := 5
def PB : Nat := 6
def RESULT : Nat := 7
/-- The result's second word on: a view, where a query's second fold writes. -/
def RESULT2 : Nat := 8
def METAD : Nat := 9
def M (j : Nat) : Nat := 10 + j

def kCOUNT : Nat := 0
def kSUM2 : Nat := 1
def kSPLIT : Nat := 2
def kMINMAX : Nat := 3
def kFIN_A : Nat := 4
def kFIN_B : Nat := 5
def kHIST : Nat := 6
def kHFIN : Nat := 7
def NSPEC : Nat := 8

/-- Each spec: its kernel, the allocations it is handed, threads a block.
    The two folds of a query's partials are `kFIN_A` (into the result's
    first word) and `kFIN_B` (its second). -/
def specs : List (Nat × List Nat × Nat) :=
  [(0, [ADV, PA, M kCOUNT], T), (1, [ADV, RES, PA, PB, M kSUM2], T), (2, [USER, PA, PB, M kSPLIT], T),
   (3, [DATE, PA, PB, M kMINMAX], T), (4, [PA, RESULT, M kFIN_A], T), (4, [PB, RESULT2, M kFIN_B], T),
   (5, [ADV, PA, M kHIST], T), (6, [PA, RESULT, M kHFIN], 256)]

def STATE_OFF : Nat := 0x4000
def META_OFF : Nat := 0x5000
def MODTAB_OFF : Nat := 0x5400
def PTX_OFF : Nat := 0x5800

def S_N : Nat := 0
def S_INIT : Nat := 1
def capOf (b : Nat) : Nat := 8 + b

/-- The grid every row kernel runs on: enough blocks for 8 rows a thread, at
    most `MAX_BLOCKS`, so the partials fit one fold. -/
def MAX_BLOCKS : Nat := 2048

def app (memSize : Nat := PTX_OFF) : Drv.App :=
  { bufs := List.replicate (M NSPEC) 0,
    kernels := specs.map fun (i, args, blk) => ⟨PTX_OFF + i, 0, args, (1, 1, 1), (blk, 1, 1), []⟩,
    memSize, cells := 8,
    entries := specs.map fun (i, _, _) => ((kernels[i]?.map (·.name)).getD "") }

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

def word (base : V .i64) (i : Nat) : Prog V L (V .i64) := iaddImm base (STATE_OFF + 8 * i)
def getW (base : V .i64) (i : Nat) : Prog V L (V .i64) := do load64 (← word base i)
def setW (base : V .i64) (i : Nat) (v : V .i64) : Prog V L Unit := do storeI64 v (← word base i)
def ptrOf (a : Drv.App) (base : V .i64) (b : Nat) : Prog V L (V .i64) := do load64 (← iaddImm base (a.bufSlot b))
def setPtr (a : Drv.App) (base : V .i64) (b : Nat) (p : V .i64) : Prog V L Unit := do
  storeI64 p (← iaddImm base (a.bufSlot b))

def ensure (a : Drv.App) (base : V .i64) (b : Nat) (bytes : V .i64) : Prog V L Unit := do
  let cap ← getW base (capOf b)
  when .ugt bytes cap do
    when .ne cap (← iconst64 0) do
      let _ ← ext (.cuda .memFree) %[← ptrOf a base b]
    let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot b), bytes]
    setW base (capOf b) bytes

def metaW (base : V .i64) (j w : Nat) (v : V .i32) : Prog V L Unit := do
  storeI32 v (← iaddImm base (META_OFF + 64 * j + 4 * w))

def upMetaAll (a : Drv.App) (base : V .i64) : Prog V L Unit := do
  let _ ← ext (.cuda .memcpyHtoD) %[← ptrOf a base METAD, ← iaddImm base META_OFF, ← iconst64 (64 * NSPEC)]

def rebindAll (a : Drv.App) (base : V .i64) : Prog V L Unit := do
  for j in List.range NSPEC do a.rebind base j

def up (a : Drv.App) (base : V .i64) : Prog V L Unit := do
  when .eq (← getW base S_INIT) (← iconst64 0) do
    Drv.up base a.CTX
    for (j, kk) in a.indexed do
      if a.modOwner j == j then
        let off ← load64 (← iaddImm base (MODTAB_OFF + 8 * (kk.ptxOff - PTX_OFF)))
        let _ ← ext (.cuda .moduleLoadData) %[← iaddImm base (a.modSlot j), ← iadd base off]
      Drv.getFunction base (a.kern j kk)
    let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot METAD), ← iconst64 (64 * NSPEC)]
    for j in List.range NSPEC do
      setPtr a base (M j) (← iaddImm (← ptrOf a base METAD) (64 * j))
    setW base S_INIT (← iconst64 1)

/-- The device's status once it has finished: the call's. -/
def finish : Prog V L (V .i64) := do uextend64 (← ext (.cuda .ctxSynchronize) %[])

/-- The row kernels' grid: blocks, and the threads they make. -/
def grid (base : V .i64) : Prog V L (V .i32 × V .i32) := do
  let n ← getW base S_N
  let b ← umin (← udiv (← iaddImm n (8 * T - 1)) (← iconst64 (8 * T))) (← iconst64 MAX_BLOCKS)
  let b32 ← ireduce32 (← umax b (← iconst64 1))
  pure (b32, ← imul b32 (← iconst32 T))

/-- The `words` of the result, down to the caller's output. -/
def answer (a : Drv.App) (base : V .i64) (words : Nat) : Prog V L (V .i64) := do
  when .ule (← iconst64 (8 * words)) (← outLen) do
    let _ ← ext (.cuda .memcpyDtoH) %[← outPtr, ← ptrOf a base RESULT, ← iconst64 (8 * words)]
  finish

def loadCode : Prog V L (V .i64) := do
  let a := app
  let base ← basePtr
  up a base
  let dp ← dataPtr
  let n ← load64 dp
  let pad8 (x : V .i64) : Prog V L (V .i64) := do band (← iaddImm x 7) (← iconst64 (-8))
  let widths : List (Nat × Nat) := [(ADV, 2), (RES, 2), (USER, 8), (DATE, 2), (REGION, 4)]
  -- the columns are taken only when the input holds them all and the rows
  -- count in the kernels' 32 bits
  let mut total ← iconst64 8
  for (_, w) in widths do total ← iadd total (← pad8 (← imul n (← iconst64 w)))
  let fits ← band (← select (← icmp .ule total (← dataLen)) (← iconst64 1) (← iconst64 0))
    (← select (← icmp .ult n (← iconst64 (2 ^ 31))) (← iconst64 1) (← iconst64 0))
  when .ne fits (← iconst64 0) do
    let mut at_ ← iaddImm dp 8
    for (b, w) in widths do
      let bytes ← pad8 (← imul n (← iconst64 w))
      ensure a base b (← umax bytes (← iconst64 8))
      rebindAll a base
      let _ ← ext (.cuda .memcpyHtoD) %[← ptrOf a base b, at_, bytes]
      at_ ← iadd at_ bytes
    for b in [PA, PB] do ensure a base b (← iconst64 (8 * 256 * MAX_BLOCKS))
    ensure a base RESULT (← iconst64 (8 * 256))
    setPtr a base RESULT2 (← iaddImm (← ptrOf a base RESULT) 8)
    rebindAll a base
    setW base S_N n
  finish

/-- A query's row kernel `j` over the grid, then its partials folded by
    `fa` into the result's first word and, where it has two, by `fb` into
    its second. -/
def scalarQuery (a : Drv.App) (base : V .i64) (j : Nat) (fa : Nat) (fb : Option Nat) : Prog V L Unit := do
  let (blocks, threads) ← grid base
  metaW base j 0 (← ireduce32 (← getW base S_N)); metaW base j 1 threads
  metaW base kFIN_A 0 blocks; metaW base kFIN_A 1 (← iconst32 fa)
  if let some f := fb then
    metaW base kFIN_B 0 blocks; metaW base kFIN_B 1 (← iconst32 f)
  upMetaAll a base
  a.runX base j blocks
  a.runX base kFIN_A (← iconst32 1)
  if fb.isSome then
    a.runX base kFIN_B (← iconst32 1)

def q1Code : Prog V L (V .i64) := do
  let base ← basePtr
  scalarQuery app base kCOUNT 0 none
  answer app base 1

def q2Code : Prog V L (V .i64) := do
  let base ← basePtr
  scalarQuery app base kSUM2 0 (some 0)
  answer app base 2

def q3Code : Prog V L (V .i64) := do
  let base ← basePtr
  scalarQuery app base kSPLIT 0 (some 0)
  answer app base 2

def q6Code : Prog V L (V .i64) := do
  let base ← basePtr
  scalarQuery app base kMINMAX 1 (some 2)
  answer app base 2

def q7Code : Prog V L (V .i64) := do
  let a := app
  let base ← basePtr
  let (blocks, threads) ← grid base
  metaW base kHIST 0 (← ireduce32 (← getW base S_N)); metaW base kHIST 1 threads
  metaW base kHFIN 0 blocks
  upMetaAll a base
  a.runX base kHIST blocks
  a.runX base kHFIN (← iconst32 1)
  answer a base 256

end

def clifIR : Except String (List FuncData) :=
  Prog.program [.ok noopFunction,
    Prog.entry "load" (Prog.compileProgStatus 1 loadCode),
    Prog.entry "q1" (Prog.compileProgStatus 2 q1Code),
    Prog.entry "q2" (Prog.compileProgStatus 3 q2Code),
    Prog.entry "q3" (Prog.compileProgStatus 4 q3Code),
    Prog.entry "q6" (Prog.compileProgStatus 5 q6Code),
    Prog.entry "q7" (Prog.compileProgStatus 6 q7Code)]

/-- The arena's image: the driver's state and each kernel's module. -/
def setup (clif : List FuncData) : Except String Artifact := do
  let texts ← kernels.mapM Kernel.ptx
  let blobs := texts.map fun t => t.toUTF8.toList ++ [0]
  let offs := (blobs.foldl (fun (acc : List Nat × Nat) b => (acc.1 ++ [acc.2], acc.2 + b.length)) ([], PTX_OFF)).1
  let total := PTX_OFF + (blobs.map List.length).sum
  let a := app total
  let u64 (v : Nat) : List UInt8 := (List.range 8).map fun i => UInt8.ofNat (v / 256 ^ i % 256)
  let image := a.initialMemory [] ([(MODTAB_OFF, offs.flatMap u64)] ++ offs.zip blobs)
  pure { functions := clif, required_memory := a.memSize, initial_memory := image }

/-- The driver's state ends before the host's words, the staged parameters
    and the module table fit their regions, and no spec has more arguments
    than a parameter block holds. -/
theorem layout_fits : decide (app.stateEnd ≤ STATE_OFF) = true ∧ STATE_OFF + 8 * capOf (M NSPEC) ≤ META_OFF ∧
    META_OFF + 64 * NSPEC ≤ MODTAB_OFF ∧ MODTAB_OFF + 8 * kernels.length ≤ PTX_OFF ∧
    (specs.all fun (_, args, _) => decide (args.length ≤ 8)) = true := by
  refine ⟨?_, by decide, by decide, by native_decide, by decide⟩
  native_decide

end Analytics

open AlgorithmLib in
def Bench.Analytics.main (args : List String) : IO Unit := do
  let dir ← requireOutputDir args
  let clif ← Prog.orDie Analytics.clifIR
  let art ← Prog.orDie (Analytics.setup clif)
  emitArtifacts dir #[artifactEntry "analytics" art]

#eval ShipScan.check "Bench.Analytics" `Bench.Analytics.main
