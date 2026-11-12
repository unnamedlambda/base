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
public import Ann.Kernels
meta import Ann.Kernels
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# base's sort and top-k engine

`benchmarks/sort` measures these beside CUB's radix sort and PyTorch's
`topk`, on data already on the device. The sort's kernels are here: one
histogram of all four passes' digits and its scan, then each pass one sweep
(a tile's prefix found by looking back over the tiles before, as CUB's
one-sweep sort does). The top-k is the ANN engine's row select
(`Ann.Kernels`) and a gather of the selected keys.

Its functions (the caller's data, then what each does):

* `load`: `n` (`u64`), then `n` keys and `n` values (`u32` each): up to the
  device.
* `sort`: the pairs sorted by key, stably: four passes of 8 bits, the keys
  and values moving between two pairs of buffers and ending where they began.
* `fetch`: the output gets the sorted keys, then their values.
* `tload`: `rows`, `cols` (`u64`), then the `rows × cols` floats: up to the
  device.
* `topk`: `k` (`u64`): each row's `k` smallest, in no order, their columns
  and their keys.
* `tfetch`: the output gets the keys (`rows × k` floats), then their columns
  (`rows × k` words).

Every call ends synchronised, so a caller timing one times the device's work.
-/

set_option maxRecDepth 8000

namespace SortEngine

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.Kernel AlgorithmLib.Kernel.Block

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

/-- **The selected keys**, a thread an output: key `i` of `rows × k` is row
    `i / k`'s column `sel[i]`. `meta`: `rows · k`, `k`, `cols`. -/
def gatherBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64])
  let b := ps.head; let tid := ps.snd
  let x := (ps.drop 4).head; let sel := (ps.drop 5).head; let out := (ps.drop 6).head
  let mt := (ps.drop 7).head
  let total ← load32 mt
  let k ← load32 (← at_ mt 4)
  let cols ← load32 (← at_ mt 8)
  let i ← iadd (← imul b (← c32 256)) tid
  when .ult i total do
    let row ← udiv i k
    let col ← load32 (← at4 sel i)
    let src ← iadd x (← ishlImm (← iadd (← imul (← ext row) (← ext cols)) (← ext col)) 2)
    store (← loadF32 src) (← at4 out i)

/-- **An exclusive scan of 256 words**, one a thread of the block: each
    warp's inclusive scan by shuffles, the warps' totals through shared memory
    at `sm + off`, then each thread's warp base. Answers this thread's
    exclusive prefix and the total. -/
def blockScan (tid : V .i32) (sm : V .i64) (off : Nat) (x : V .i32) : Prog V L (V .i32 × V .i32) := do
  let lane ← band tid (← c32 31)
  let w ← ushrImm tid 5
  let inc ← AnnKernels.laneScan lane x
  let tot ← at_ sm off
  when .eq lane (← c32 31) do storeI32 inc (← at4 tot w)
  callLocalVoid barrier %[]
  let mut base ← c32 0
  let mut all ← c32 0
  for k in List.range 8 do
    let t ← load32 (← at4 tot (← c32 k))
    base ← iadd base (← select (← icmp .ult (← c32 k) w) t (← c32 0))
    all ← iadd all t
  callLocalVoid barrier %[]
  pure (← iadd base (← isub inc x), all)

/-- Keys a block of the histogram counts at a time: its 128 threads' 32. -/
def HKEYS : Nat := 4096

/-- Blocks the histogram takes, at most: each counts its chunks' digits. -/
def UPG : Nat := 1024

/-- **The four passes' histograms**: block `g` (128 threads) counts the
    digits `(key >> 8p) & 255` of every pass `p` of chunks `g, g + G, …` of
    `HKEYS` keys in shared memory, by shared adds, then adds count `v` of
    pass `p` to `hist[p · 256 + v]` (atomically: the blocks' sums commute).
    `meta`: `n`, `chunks`, `G`. -/
def histBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64])
  let g := ps.head; let tid := ps.snd; let sm := (ps.drop 3).head
  let keys := (ps.drop 4).head; let hist := (ps.drop 5).head; let mt := (ps.drop 6).head
  let n ← load32 mt
  let chunks ← load32 (← at_ mt 4)
  let grid ← load32 (← at_ mt 8)
  stridedDo tid (← c32 1024) 128 fun i => do storeI32 (← c32 0) (← at4 sm i)
  callLocalVoid barrier %[]
  let one ← c32 1
  let _ ← wloop %[g] (fun cs => pure (contIfULt cs.head chunks, %[], ()))
    (fun cs _ => do
      let first ← iadd (← imul cs.head (← c32 HKEYS)) tid
      -- the thread's keys, their loads together, then their digits counted
      let mut ks := #[]
      for r in List.range (HKEYS / 128) do
        let i ← iadd first (← c32 (128 * r))
        let ok ← icmp .ult i n
        ks := ks.push (ok, ← load32 (← at4 keys (← select ok i (← c32 0))))
      for (ok, key) in ks do
        when .ne (← bit ok) (← c32 0) do
          for p in List.range 4 do
            let d ← band (← ushrImm key (8 * p)) (← c32 255)
            callLocalVoid sharedAdd %[← at4 sm (← iadd d (← c32 (256 * p))), one]
      pure %[← iadd cs.head grid])
  callLocalVoid barrier %[]
  stridedDo tid (← c32 1024) 128 fun i => do
    let _ ← atomicRmw .add (← at4 hist i) (← load32 (← at4 sm i)) .relaxed

/-- **Each pass's digit bases**, one block: `base[p · 256 + v]`, the keys of
    the digit values below `v` in pass `p`, the histogram scanned. -/
def baseBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64])
  let tid := ps.snd; let sm := (ps.drop 3).head
  let hist := (ps.drop 4).head; let base := (ps.drop 5).head
  for p in List.range 4 do
    let v ← iadd tid (← c32 (256 * p))
    let (pre, _) ← blockScan tid sm 0 (← load32 (← at4 hist v))
    storeI32 pre (← at4 base v)

/-- Keys a lane of the tile scatter holds: a warp's run of `32 · ITEMS`. -/
def ITEMS : Nat := 12

/-- Keys a tile of the scatter holds: its 256 threads' items. -/
def TILE : Nat := 256 * ITEMS

/-- Shared memory of the tile scatter: the tile's keys, then its values, in
    digit order, each warp's count of each digit, each digit's start in the
    tile and its place in the output less that start, the scan's warp
    totals, the tile's number. -/
def SK : Nat := 0
def SCNT : Nat := 4 * TILE
def STDS : Nat := SCNT + 4 * 8 * 256
def SOUT : Nat := STDS + 4 * 256
def SSCAN : Nat := SOUT + 4 * 256
def STILE : Nat := SSCAN + 4 * 8
def STHIST : Nat := STILE + 4
def SCATTER_SMEM : Nat := STHIST + 4 * 256

/-- **One pass of the sort**, in one sweep: a block (256 threads) takes the
    next tile by the pass's ticket, tile `b` the `TILE` keys at `b · TILE`,
    warp `w` the run of `32 · ITEMS` at `w · 32 · ITEMS`, a lane its `ITEMS`
    keys `32` apart. It puts the tile in shared memory in digit order (stable:
    a key's place its digit's start in the tile, the keys of that digit in the
    warps before its own, then those its warp met before it: in its earlier
    rounds, and before it in its round), finds each digit's keys in the tiles
    before by looking back over their statuses, then writes the tile out in
    that order, so threads next to each other write places next to each
    other: a key `i` places at `base[d] + before[d] + i - start[d]`. A warp
    counts its digits in its own counters, a round's count of a digit by its
    lowest lane. `meta`: `n`, `shift`, the pass's tag, its ticket. -/
def scatterBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ List.replicate 7 .i64)
  let tid := ps.snd; let sm := (ps.drop 3).head
  let kin := (ps.drop 4).head; let vin := (ps.drop 5).head
  let kout := (ps.drop 6).head; let vout := (ps.drop 7).head
  let bases := (ps.drop 8).head; let look := (ps.drop 9).head
  let mt := (ps.drop 10).head
  let n ← load32 mt
  let shift ← load32 (← at_ mt 4)
  let tag ← load32 (← at_ mt 8)
  let base ← iadd bases (← ext (← ishlImm (← ushrImm shift 3) 10))
  let lane ← band tid (← c32 31)
  let w ← ushrImm tid 5
  let ltMask ← isub (← ishl (← c32 1) lane) (← c32 1)
  let sk ← at_ sm SK
  let cnt ← at_ sm SCNT
  let tds ← at_ sm STDS
  let outAt ← at_ sm SOUT
  let thist ← at_ sm STHIST
  for k in List.range 8 do storeI32 (← c32 0) (← at4 cnt (← iadd (← c32 (256 * k)) tid))
  storeI32 (← c32 0) (← at4 thist tid)
  -- the tile: the next by the pass's ticket, so a tile's predecessors are
  -- held by blocks already running
  when .eq tid (← c32 0) do
    storeI32 (← atomicRmw .add (← at_ mt 12) (← c32 1) .relaxed) (← at_ sm STILE)
  callLocalVoid barrier %[]
  let b ← load32 (← at_ sm STILE)
  let tile ← imul b (← c32 TILE)
  let digit (k : V .i32) : Prog V L (V .i32) := do band (← ushr k shift) (← c32 255)
  let run ← iadd tile (← iadd (← imul w (← c32 (32 * ITEMS))) lane)
  let mut items := #[]
  for r in List.range ITEMS do
    let g ← iadd run (← c32 (32 * r))
    let ok ← icmp .ult g n
    let gc ← select ok g (← c32 0)
    items := items.push (ok, ← load32 (← at4 kin gc), r)
  -- the tile's count of each digit, by shared adds, published before the
  -- ranking, so a tile after looking back finds it there. A status is the
  -- count under the pass's tag (`2 · tag + 1`, this tile's count; `2 · tag +
  -- 2`, every count up to it), so what an earlier pass or sort left is not
  -- mistaken for this pass's.
  let one ← c32 1
  for (ok, key, _) in items do
    when .ne (← bit ok) (← c32 0) do callLocalVoid sharedAdd %[← at4 thist (← digit key), one]
  callLocalVoid barrier %[]
  let tcount ← load32 (← at4 thist tid)
  let stat (j : V .i32) : Prog V L (V .i64) := do
    iadd look (← ext (← ishlImm (← iadd (← ishlImm j 8) tid) 3))
  let aggTag ← iadd (← ishlImm tag 1) (← c32 1)
  let prefTag ← iadd (← ishlImm tag 1) (← c32 2)
  let status (tg x : V .i32) : Prog V L (V .i64) := do bor (← ishlImm (← ext tg) 32) (← ext x)
  atomicStore (← status (← select (← icmp .eq b (← c32 0)) prefTag aggTag) tcount) (← stat b) .relaxed
  -- each key's place among its warp's keys of its digit
  let wcnt ← at4 cnt (← imul w (← c32 256))
  let mut ranks := #[]
  for (ok, key, _) in items do
    let d ← digit key
    let m ← peers d ok
    let lead ← band ok (← icmp .eq (← ctz m) lane)
    let slot ← at4 wcnt d
    let before ← ifte (jTys := [.i32]) .ne (← bit lead) (← c32 0)
      (thn := do
        let c ← load32 slot
        storeI32 (← iadd c (← popcnt m)) slot
        pure %[c])
      (els := do pure %[← c32 0])
    let bs ← callLocal shflIdx %[before.head, ← band (← ctz m) (← c32 31)]
    ranks := ranks.push (← iadd bs (← popcnt (← band m ltMask)))
    callLocalVoid barrier %[]
  -- each warp's counts become the counts of the warps before it, and the
  -- tile's count of each digit is their sum
  let mut acc ← c32 0
  for k in List.range 8 do
    let slot ← at4 cnt (← iadd (← c32 (256 * k)) tid)
    let t ← load32 slot
    storeI32 acc slot
    acc ← iadd acc t
  -- each digit's start in the tile
  let (start, _) ← blockScan tid sm SSCAN acc
  storeI32 start (← at4 tds tid)
  callLocalVoid barrier %[]
  let places ← (items.zip ranks).mapM fun ((ok, key, _), rank) => do
    let d ← select ok (← digit key) (← c32 0)
    iadd (← iadd (← load32 (← at4 tds d)) (← load32 (← at4 wcnt d))) rank
  -- the values, now the ranking no longer holds registers
  let vals ← items.mapM fun (ok, _, r) => do
    load32 (← at4 vin (← select ok (← iadd run (← c32 (32 * r))) (← c32 0)))
  -- the tile's keys in digit order, then out in that order (threads next to
  -- each other writing places next to each other)
  for ((ok, key, _), pos) in items.zip places do
    when .ne (← bit ok) (← c32 0) do storeI32 key (← at4 sk pos)
  -- the digit's keys in the tiles before, looked back for once the values'
  -- loads are under way and the keys are in place (the tiles before have
  -- published their counts by then): each status waited for, its count
  -- added, until one that has all before it
  let excl ← ifte (jTys := [.i32]) .eq b (← c32 0)
    (thn := do pure %[← c32 0])
    (els := do
      let r ← wloopL %[← isub b (← c32 1), ← c32 0]
        (fun _ c => pure (contIfULt c.head b, %[c.snd], ()))
        (fun lbl c _ => do
          let v ← atomicLoad .i64 (← stat c.head) .relaxed
          let hi ← ireduce32 (← ushrImm v 32)
          let sum ← iadd c.snd (← ireduce32 v)
          when .eq hi prefTag do brk lbl %[sum]
          let isAgg ← icmp .eq hi aggTag
          pure %[← select isAgg (← isub c.head (← c32 1)) c.head, ← select isAgg sum c.snd])
      pure %[r.head])
  atomicStore (← status prefTag (← iadd excl.head tcount)) (← stat b) .relaxed
  storeI32 (← isub (← iadd (← load32 (← at4 base tid)) excl.head) start) (← at4 outAt tid)
  callLocalVoid barrier %[]
  let mut outs := #[]
  for r in List.range ITEMS do
    let i ← iadd tid (← c32 (256 * r))
    let ok ← icmp .ult (← iadd tile i) n
    let key ← load32 (← at4 sk i)
    let o ← iadd (← load32 (← at4 outAt (← digit key))) i
    when .ne (← bit ok) (← c32 0) do storeI32 key (← at4 kout o)
    outs := outs.push (ok, o)
  callLocalVoid barrier %[]
  for (((ok, _, _), pos), val) in (items.zip places).zip vals do
    when .ne (← bit ok) (← c32 0) do storeI32 val (← at4 sk pos)
  callLocalVoid barrier %[]
  for (r, ok, o) in (List.range ITEMS).zip outs.toList do
    when .ne (← bit ok) (← c32 0) do storeI32 (← load32 (← at4 sk (← iadd tid (← c32 (256 * r))))) (← at4 vout o)

end

def hist : Kernel := AnnKernels.k "sort_hist" histBody 3 (4 * 1024)
def gather : Kernel := AnnKernels.k "sort_gather" gatherBody 4
def bases : Kernel := AnnKernels.k "sort_bases" baseBody 2 (4 * 8)
def scatter : Kernel := AnnKernels.k "sort_scatter" scatterBody 7 SCATTER_SMEM

/-- The kernels, each its own module, by index. -/
def kernels : List Kernel := [hist, AnnKernels.rowSelect, gather, bases, scatter]

/-! ## The layout -/

def K0 : Nat := 0
def V0 : Nat := 1
def K1 : Nat := 2
def V1 : Nat := 3
def HIST : Nat := 4
def X : Nat := 5
def SEL : Nat := 6
def OUTV : Nat := 7
/-- Each pass's digit bases, and each tile's status of each digit. -/
def BASE : Nat := 8
def LOOK : Nat := 9
def METAD : Nat := 10
def M (j : Nat) : Nat := 11 + j

/-- The specs: the histogram of the four passes and their bases, a scatter
    for each pass (the passes alternate the buffer pairs, each its own shift
    and ticket), the row select and the gather. -/
def kHIST : Nat := 0
def kBASE : Nat := 1
def kSCAT (p : Nat) : Nat := 2 + p
def kSEL : Nat := 6
def kGATH : Nat := 7
def NSPEC : Nat := 8

def STATE_OFF : Nat := 0x4000
def META_OFF : Nat := 0x5000
def MODTAB_OFF : Nat := 0x5400
def PTX_OFF : Nat := 0x5800

/-- State words: the pairs, the matrix's rows and columns, the `k` last run,
    whether the driver is up, the sorts run (each pass's tag). -/
def S_N : Nat := 0
def S_ROWS : Nat := 1
def S_COLS : Nat := 2
def S_K : Nat := 3
def S_INIT : Nat := 4
def S_SORTS : Nat := 5
def capOf (b : Nat) : Nat := 8 + b

def specs : List (Nat × List Nat × Nat) :=
  let src (p : Nat) := if p % 2 == 0 then (K0, V0, K1, V1) else (K1, V1, K0, V0)
  [(0, [K0, HIST, M kHIST], 128), (3, [HIST, BASE], 256)]
  ++ (List.range 4).map (fun p => let (k, v, k', v') := src p; (4, [k, v, k', v', BASE, LOOK, M (kSCAT p)], 256))
  ++ [(1, [X, SEL, M kSEL], 256), (2, [X, SEL, OUTV, M kGATH], 256)]

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

/-- Allocation `b` at least `bytes` long (its contents are not kept). -/
def ensure (a : Drv.App) (base : V .i64) (b : Nat) (bytes : V .i64) : Prog V L Unit := do
  let cap ← getW base (capOf b)
  when .ugt bytes cap do
    when .ne cap (← iconst64 0) do
      let _ ← ext (.cuda .memFree) %[← ptrOf a base b]
    let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot b), bytes]
    setW base (capOf b) bytes

def metaW (base : V .i64) (j w : Nat) (v : V .i32) : Prog V L Unit := do
  storeI32 v (← iaddImm base (META_OFF + 64 * j + 4 * w))

/-- Every kernel's meta words, by one copy. -/
def upMetaAll (a : Drv.App) (base : V .i64) : Prog V L Unit := do
  let _ ← ext (.cuda .memcpyHtoD) %[← ptrOf a base METAD, ← iaddImm base META_OFF, ← iconst64 (64 * NSPEC)]

def rebindAll (a : Drv.App) (base : V .i64) : Prog V L Unit := do
  for j in List.range NSPEC do a.rebind base j

def cdiv (x : V .i64) (y : Nat) : Prog V L (V .i32) := do
  ireduce32 (← udiv (← iaddImm x (y - 1)) (← iconst64 y))

def lo32 (x : V .i64) : Prog V L (V .i32) := ireduce32 x

/-- Bring the driver up, load the modules, make the meta block. Once. -/
def up (a : Drv.App) (base : V .i64) : Prog V L Unit := do
  when .eq (← getW base S_INIT) (← iconst64 0) do
    Drv.up base a.CTX
    for (j, k) in a.indexed do
      if a.modOwner j == j then
        let off ← load64 (← iaddImm base (MODTAB_OFF + 8 * (k.ptxOff - PTX_OFF)))
        let _ ← ext (.cuda .moduleLoadData) %[← iaddImm base (a.modSlot j), ← iadd base off]
      Drv.getFunction base (a.kern j k)
    let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot METAD), ← iconst64 (64 * NSPEC)]
    for j in List.range NSPEC do
      setPtr a base (M j) (← iaddImm (← ptrOf a base METAD) (64 * j))
    setW base S_INIT (← iconst64 1)

/-- The device's status once it has finished: the call's. -/
def finish : Prog V L (V .i64) := do uextend64 (← ext (.cuda .ctxSynchronize) %[])

def loadCode : Prog V L (V .i64) := do
  let a := app
  let base ← basePtr
  up a base
  let dp ← dataPtr
  -- taken only where the input holds `n` and its keys and values, `n` in the
  -- kernels' 32 bits
  let n ← ifte (jTys := [.i64]) .uge (← dataLen) (← iconst64 8)
    (thn := do pure %[← load64 dp]) (els := do pure %[← iconst64 (-1)])
  let ok ← band (← select (← icmp .ult n.head (← iconst64 (2 ^ 31))) (← iconst64 1) (← iconst64 0))
    (← select (← icmp .ule (← iaddImm (← ishlImm n.head 3) 8) (← dataLen)) (← iconst64 1) (← iconst64 0))
  let st ← ifte (jTys := [.i64]) .ne ok (← iconst64 0)
    (thn := do
      let n := n.head
      let bytes ← ishlImm (← umax n (← iconst64 1)) 2
      for b in [K0, V0, K1, V1] do ensure a base b bytes
      ensure a base HIST (← iconst64 4096)
      ensure a base BASE (← iconst64 4096)
      -- the tiles' statuses, cleared once so none reads as a pass's own
      let lb ← imul (← umax (← udiv (← iaddImm n (TILE - 1)) (← iconst64 TILE))
        (← iconst64 1)) (← iconst64 2048)
      ensure a base LOOK lb
      let _ ← ext (.cuda .memsetD8) %[← ptrOf a base LOOK, ← iconst32 0, lb]
      rebindAll a base
      setW base S_N n
      let _ ← ext (.cuda .memcpyHtoD) %[← ptrOf a base K0, ← iaddImm dp 8, ← ishlImm n 2]
      let _ ← ext (.cuda .memcpyHtoD) %[← ptrOf a base V0, ← iadd (← iaddImm dp 8) (← ishlImm n 2), ← ishlImm n 2]
      pure %[← finish])
    (els := do pure %[← iconst64 1])
  pure st.head

def sortCode : Prog V L (V .i64) := do
  let a := app
  let base ← basePtr
  let n ← getW base S_N
  let n32 ← lo32 n
  let blocks ← cdiv n TILE
  let chunks ← cdiv n HKEYS
  let grid ← umin chunks (← iconst32 UPG)
  metaW base kHIST 0 n32; metaW base kHIST 1 chunks; metaW base kHIST 2 grid
  -- each pass's tag, below 2^30 so its statuses' tags fit 32 bits; its ticket
  -- starts at nought with the meta block's copy
  let sorts ← getW base S_SORTS
  setW base S_SORTS (← iaddImm sorts 1)
  for p in List.range 4 do
    let tag ← band (← lo32 (← iaddImm (← ishlImm sorts 2) (p + 1))) (← iconst32 0x3FFFFFFF)
    metaW base (kSCAT p) 0 n32; metaW base (kSCAT p) 1 (← iconst32 (8 * p)); metaW base (kSCAT p) 2 tag
    metaW base (kSCAT p) 3 (← iconst32 0)
  upMetaAll a base
  let _ ← ext (.cuda .memsetD8) %[← ptrOf a base HIST, ← iconst32 0, ← iconst64 4096]
  a.runX base kHIST grid
  a.runX base kBASE (← iconst32 1)
  for p in List.range 4 do a.runX base (kSCAT p) blocks
  finish

def fetchCode : Prog V L (V .i64) := do
  let a := app
  let base ← basePtr
  let n ← getW base S_N
  let out ← outPtr
  when .ule (← ishlImm n 3) (← outLen) do
    let _ ← ext (.cuda .memcpyDtoH) %[out, ← ptrOf a base K0, ← ishlImm n 2]
    let _ ← ext (.cuda .memcpyDtoH) %[← iadd out (← ishlImm n 2), ← ptrOf a base V0, ← ishlImm n 2]
  finish

def tloadCode : Prog V L (V .i64) := do
  let a := app
  let base ← basePtr
  up a base
  let dp ← dataPtr
  -- taken only where the input holds the shape and the matrix, its size in the
  -- kernels' 32 bits
  let shape ← ifte (jTys := [.i64, .i64]) .uge (← dataLen) (← iconst64 16)
    (thn := do pure %[← load64 dp, ← load64 (← iaddImm dp 8)])
    (els := do pure %[← iconst64 (2 ^ 31), ← iconst64 1])
  let rows := shape.head
  let cols := shape.snd
  -- each below 2^31 first, so their product cannot wrap
  let small ← band (← select (← icmp .ult rows (← iconst64 (2 ^ 31))) (← iconst64 1) (← iconst64 0))
    (← select (← icmp .ult cols (← iconst64 (2 ^ 31))) (← iconst64 1) (← iconst64 0))
  let cells ← imul rows cols
  let ok ← band small (← band (← select (← icmp .ult cells (← iconst64 (2 ^ 31))) (← iconst64 1) (← iconst64 0))
    (← select (← icmp .ule (← iaddImm (← ishlImm cells 2) 16) (← dataLen)) (← iconst64 1) (← iconst64 0)))
  let st ← ifte (jTys := [.i64]) .ne ok (← iconst64 0)
    (thn := do
      ensure a base X (← ishlImm (← umax (← imul rows cols) (← iconst64 1)) 2)
      rebindAll a base
      setW base S_ROWS rows
      setW base S_COLS cols
      let _ ← ext (.cuda .memcpyHtoD) %[← ptrOf a base X, ← iaddImm dp 16, ← ishlImm (← imul rows cols) 2]
      pure %[← finish])
    (els := do pure %[← iconst64 1])
  pure st.head

def topkCode : Prog V L (V .i64) := do
  let a := app
  let base ← basePtr
  -- answered only for a `k` of at least one, after it is clamped to the row
  -- (the selection finds the `k`-th smallest), read from an input that holds it
  let k0 ← ifte (jTys := [.i64]) .uge (← dataLen) (← iconst64 8)
    (thn := do pure %[← load64 (← dataPtr)]) (els := do pure %[← iconst64 0])
  let k ← umin k0.head (← getW base S_COLS)
  let st ← ifte (jTys := [.i64]) .uge k (← iconst64 1)
    (thn := do
      let rows ← getW base S_ROWS
      let total ← imul rows k
      ensure a base SEL (← ishlImm (← umax total (← iconst64 1)) 2)
      ensure a base OUTV (← ishlImm (← umax total (← iconst64 1)) 2)
      rebindAll a base
      setW base S_K k
      metaW base kSEL 0 (← lo32 (← getW base S_COLS)); metaW base kSEL 1 (← lo32 k)
      metaW base kGATH 0 (← lo32 total); metaW base kGATH 1 (← lo32 k); metaW base kGATH 2 (← lo32 (← getW base S_COLS))
      upMetaAll a base
      a.runX base kSEL (← lo32 rows)
      a.runX base kGATH (← cdiv total 256)
      pure %[← finish])
    (els := do pure %[← iconst64 1])
  pure st.head

def tfetchCode : Prog V L (V .i64) := do
  let a := app
  let base ← basePtr
  let total ← imul (← getW base S_ROWS) (← getW base S_K)
  let out ← outPtr
  when .ule (← ishlImm total 3) (← outLen) do
    let _ ← ext (.cuda .memcpyDtoH) %[out, ← ptrOf a base OUTV, ← ishlImm total 2]
    let _ ← ext (.cuda .memcpyDtoH) %[← iadd out (← ishlImm total 2), ← ptrOf a base SEL, ← ishlImm total 2]
  finish

end

def clifIR : Except String (List FuncData) :=
  Prog.program [.ok noopFunction,
    Prog.entry "load" (Prog.compileProgStatus 1 loadCode),
    Prog.entry "sort" (Prog.compileProgStatus 2 sortCode),
    Prog.entry "fetch" (Prog.compileProgStatus 3 fetchCode),
    Prog.entry "tload" (Prog.compileProgStatus 4 tloadCode),
    Prog.entry "topk" (Prog.compileProgStatus 5 topkCode),
    Prog.entry "tfetch" (Prog.compileProgStatus 6 tfetchCode)]

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
theorem layout_fits : decide (app.stateEnd ≤ STATE_OFF) = true ∧ STATE_OFF + 8 * (capOf (M NSPEC)) ≤ META_OFF ∧
    META_OFF + 64 * NSPEC ≤ MODTAB_OFF ∧ MODTAB_OFF + 8 * kernels.length ≤ PTX_OFF ∧
    (specs.all fun (_, args, _) => decide (args.length ≤ 8)) = true := by
  refine ⟨?_, by decide, by decide, by native_decide, by decide⟩
  native_decide

end SortEngine

open AlgorithmLib in
def Ann.Sort.main (args : List String) : IO Unit := do
  let dir ← requireOutputDir args
  let clif ← Prog.orDie SortEngine.clifIR
  let art ← Prog.orDie (SortEngine.setup clif)
  emitArtifacts dir #[artifactEntry "sort" art]

#eval ShipScan.check "Ann.Sort" `Ann.Sort.main
