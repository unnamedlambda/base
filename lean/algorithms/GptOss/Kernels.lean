module
public import AlgorithmLib.Surface.Drv
meta import AlgorithmLib.Surface.Drv
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
public import GptOss.Attention
meta import GptOss.Attention
public import AlgorithmLib.Kernel.Block
meta import AlgorithmLib.Kernel.Block
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
  # The MXFP4 expert kernels

  Three kernels, and one decoding routine they share.

  gpt-oss ships its experts in MXFP4: two four-bit codes to a byte, one shared
  power-of-two scale per thirty-two of them. A decode step reads four experts
  per layer and nothing else of comparable size, so what these kernels cost is
  what a token costs.

  ## Why these are here and not in `ML/`

  The machine `ML/` proves kernels in addresses buffers by element and holds
  `Float32`: no byte load, no shift, no mask (`ML/QuantMX.lean` says so in its
  own header, having otherwise specified this exact format completely). A
  kernel that unpacks nibbles is therefore outside it. These are written as
  kernel bodies (`Body`), which the proven PTX compiler compiles, but nothing
  states what they compute.

  That is a real gap and it is on the ledger (`GptOssScan.openObligations`,
  the first three entries). What closes it is extending the machine with a byte
  load and the shifts, after which `mxDot_spec` — already proven — hands the
  existing `warpDotV4` theorems a dequantised buffer and asks for no new kernel
  proof. Until then these are checked, not proven, and the check is bit-level:
  every one of the 256 byte values against `fp4Val`.

  ## What they must not become

  The obvious way to make an MXFP4 kernel provable today is to dequantise into
  an f32 buffer and hand that to a proven GEMV. That kernel exists here —
  `mxfp4DequantF32` — and prefill uses it, because prefill is compute-bound on
  this hardware and can afford the extra traffic.

  Decode cannot. Dequantising 12.6 MiB of expert to 101 MiB of f32 and reading
  it back is eight times the memory traffic on the one path that is bound by
  exactly that. So decode fuses: the nibbles are unpacked in registers, between
  the load and the multiply, and the f32 weight never reaches memory. The
  proof obligation is the price, and it is paid on the ledger rather than in
  the emitted code.

  ## The decode, exactly

  E2M1 is sign(1) exponent(2) mantissa(1), bias 1. Rather than a lookup table,
  the `Float32` is assembled from its parts:

  * `e = 0` is the subnormal case: the value is `0.0` or `0.5`, chosen by `m`.
  * `e > 0` gives exponent field `e - 1 + 127 = e + 126` and mantissa `m << 22`.

  which reproduces the eight magnitudes `QuantMX.fp4Mag` lists — `0, 0.5, 1,
  1.5, 2, 3, 4, 6` — and costs six integer instructions with no memory
  reference. The E8M0 scale is the same trick and cheaper: a byte `s` is
  `2^(s-127)`, which is the `Float32` whose exponent field is `s` and whose
  mantissa is zero, so the scale is `s << 23` read as a float.
-/

set_option maxRecDepth 8000

namespace GptOssKernels


/-! ## Geometry -/

/-- Hidden width. -/
def H : Nat := 2880
/-- Expert intermediate width. -/
def I : Nat := 2880
/-- Elements per microscaling block, and the warp width. -/
def G : Nat := 32

/-- Packed bytes in one weight row of `H` columns: two codes to a byte. -/
def rowBytesH : Nat := H / 2
/-- Scale bytes in one weight row of `H` columns: one per block. -/
def rowScalesH : Nat := H / G
/-- The same for a row of `I` columns (the down projection contracts over `I`). -/
def rowBytesI : Nat := I / 2
def rowScalesI : Nat := I / G

/-- Warps to a block, each taking one output row.

    One warp per block leaves the card two thirds idle: sm_86 will hold 48
    warps on an SM but only 16 blocks, so single-warp blocks cap occupancy at a
    third and there is nothing left to hide a load's latency behind.

    Eight, not four, because the activation is staged in shared memory per
    block: one staged copy then serves eight rows instead of four, and eight
    warps still fit the shared-memory budget at full warp occupancy. -/
def warpsPerCta : Nat := 8

/-- Output rows a warp of the down projection takes.

    The gate/up kernel gets two for free -- gate and up are two rows over one
    activation -- and that is most of why it runs closer to bandwidth. Down has
    one row per output and has to be told to take several.

    What it buys is the staged activation: every CTA stages `I` floats, so at
    one row per warp the staging traffic equals the weight traffic and the
    kernel reads twice what it came for. `rowsPerWarpDown` divides that, and
    divides the shared reads per `fma` with it. Four rather than more because
    the hoisted loads are `rowsPerWarpDown * nChunks` registers live at once,
    and past four that costs occupancy faster than it saves traffic. -/
def rowsPerWarpDown : Nat := 4

/-- Output rows a warp of the gate/up kernel takes.

    Each is two contractions -- gate and up -- so this warp runs
    `2 * rowsPerWarpGateUp` of them over one staged activation. -/
def rowsPerWarpGateUp : Nat := 2


/-- gpt-oss's clamped SwiGLU: the gate is clamped above, the linear branch on
    both sides, and the linear branch is offset by one.

    `alpha` and `limit` are the published constants. They appear as raw
    `Float32` bit patterns because that is the only way to name a float exactly
    in this builder; the decimal is in the comment beside each. -/
def swigluAlphaBits : UInt32 := 0x3FD9DB23   -- 1.702
def swigluLimitBits : UInt32 := 0x40E00000   -- 7.0
def negLimitBits    : UInt32 := 0xC0E00000   -- -7.0
def oneBits         : UInt32 := 0x3F800000   -- 1.0

/-- Experts in a layer, and how many a token routes to.  Named here as well as
    in the algorithm because the router kernel is emitted for these counts: a
    kernel is generated for a geometry, and this is part of one. -/
def NE : Nat := 32
def TOPK : Nat := 4

/-- The sixteen values a code can take, as `Float32` bit patterns in code
    order: `QuantMX.fp4Val` tabulated. -/
def fp4Table : List UInt32 :=
  [0x00000000, 0x3F000000, 0x3F800000, 0x3FC00000,
   0x40000000, 0x40400000, 0x40800000, 0x40C00000,
   0x80000000, 0xBF000000, 0xBF800000, 0xBFC00000,
   0xC0000000, 0xC0400000, 0xC0800000, 0xC0C00000]

/-- Floats of shared memory the table occupies: **one copy per lane**.

    A sixteen-entry table is the obvious way to replace `Body.fp4`'s thirteen
    integer instructions with one load, and the obvious layout does not work:
    thirty-two lanes decoding thirty-two unrelated codes read sixteen scattered
    entries, and the bank conflicts cost more than the arithmetic they replace
    — measured at 208 us an expert against 190 for the arithmetic form.

    Replicating the table once per lane fixes the conflict. Entry
    `(code, lane)` sits at index `lane + 32*code`, so lane `L` always reads
    bank `L` whatever code it is decoding, and thirty-two lanes decoding
    thirty-two different codes still take one cycle.

    It is worth 3%, not the sixfold the instruction count suggests, and the
    whole sequence of attempts is worth recording because the arithmetic
    predicted the wrong winner three times out of four:

    | change | us/expert |
    |---|---|
    | one packed byte a lane | 231 |
    | four bytes a lane | 190 |
    | four warps a block | 190 |
    | sixteen-entry shared table | 208 (reverted) |
    | activation staged in shared memory, eight warps | 154 |
    | table replicated per lane | 149 |

    Only two of those mattered, and the one that mattered most — staging the
    activation — was invisible until the others had been tried and had not
    worked. What it addressed was traffic nobody had counted: every warp read
    the whole activation for its own row, so the activation crossed the cache
    thousands of times per launch against the weights' once. -/
def fp4TableFloats : Nat := 16 * 32

/-- Floats of shared memory the staged activation occupies: one per element,
    plus one pad per thirty-two.

    The pad is what makes the reads conflict-free. A lane owns eight
    consecutive elements, so without padding lanes four apart would land on the
    same bank and every read would serialise eight ways. Shifting each group of
    thirty-two along by one float sends lane `L` to bank `(8L + L/4 + j) mod 32`,
    which is a permutation of the thirty-two banks. -/
def smemFloats : Nat := H + H / G

/-- Shared bytes the module declares: the replicated table, then the staged
    activation after it. -/
def smemBytes : Nat := (fp4TableFloats + smemFloats) * 4

/-- Byte offset of the staged activation within the shared block. -/
def xSmemOff : Nat := fp4TableFloats * 4

/-! ## Fused decode attention

    The four-stage attention path -- scores, softmax, narrow, mix -- costs more
    at depth than everything else in a token put together, and neither of its
    two problems is reachable by tuning.

    The scores contraction has `n = GQA`, which is eight. cuBLAS tiles that
    dimension sixty-four wide, so it discards most of every tile: measured at a
    hundred thousand keys it runs six times off the bandwidth its own operands
    would need. And the four stages round-trip a scores buffer that is
    `NQ * seqLen` floats -- twenty-five megabytes at that length -- written by
    the first, rewritten by the second and third, and read by the fourth. That
    buffer exists because they are four kernels and for no other reason.

    So they become one. A warp takes one key head and one tile of the sequence,
    and for each key in the tile it forms the eight scores that head serves,
    folds them straight into a running maximum and sum, and accumulates the
    value row against them. The scores never reach memory at all.

    **Online softmax.** Holding `m` (the largest score seen) and `l` (the sum of
    `exp(score - m)`), a new score `x` gives `m' = max(m, x)` and rescales both
    the sum and the accumulator by `exp(m - m')` before adding the new term.
    That is exact, not an approximation: it is the same sum, factored so that
    nothing is exponentiated at more than zero and no pass over the scores is
    needed to find their maximum first.

    **Why tiles and a combine pass.** One warp per key head would be eight warps
    for the whole card. Splitting the sequence gives `NKV * nTiles` of them, and
    the partial results merge exactly -- two `(m, l, acc)` triples combine by
    the same rescaling the loop already does. The learned sink is added by the
    combine, once, because it belongs to the head and not to a tile.
-/

/-- The fewest keys a tile is ever given.

    Below this the tiles stop buying parallelism -- a warp still walks them one
    key at a time -- and start costing merge. -/
def ATT_TILE_MIN : Nat := 32

/-- The most tiles a layer is ever split into, and so the height of the
    partials buffer whatever the context.

    Measured: at 98304 keys, 768 tiles beat 192, which beat 24. More tiles keep
    winning until the merge, which reads every tile, becomes the cost -- so the
    tile size is chosen at run time to land here and the merge is bounded. -/
def ATT_TILES_MAX : Nat := 768

/-- Head width of the model these kernels are generated for. -/
def HD_K : Nat := 64
/-- Query heads that share one key head. -/
def GQA_K : Nat := 8
/-- Key heads. -/
def NKV_K : Nat := 8

/-- Floats a tile's partial result occupies: the running maximum, the running
    sum, and one accumulator per element of the head. -/
def attPartialFloats : Nat := 2 + HD_K

/-- `1/sqrt(64)`, the attention scale, as a `Float32` bit pattern. -/
def attScaleBits : UInt32 := 0x3E000000

-- ---------------------------------------------------------------------------
-- Kernels as bodies
-- ---------------------------------------------------------------------------

namespace Body
open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.Prog AlgorithmLib.Kernel AlgorithmLib.Kernel.Block

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

def u8 (a : V .i64) : Prog V L (V .i32) := load { kind := .uload8, ty := .i32 } a

/-- Stage the per-lane FP4 table at `sm` and the `k` floats at `hv` after it,
    padded (element `e` at `e + e/32`); every thread of the block takes part,
    and a barrier follows each. Returns the activation's base.

    The table holds a copy of each code's value for each lane, at a stride of
    128 bytes a code, so every lane reads its own bank. The activation is
    staged because every warp contracts a different row against the same
    activation: read from global memory, it crosses the cache far more often
    than the weights do, which is what the kernel was measured spending its
    time on. -/
def stage (tid : V .i32) (sm hv : V .i64) (k : Nat) : Prog V L (V .i64) := do
  -- the table: 512 entries, two per thread of the 256
  for rep in [0, 1] do
    let i ← iadd tid (← c32 (256 * rep))
    let code ← ushrImm i 5
    let mut v ← fconst .f32 0
    for (bits, j) in fp4Table.zipIdx do
      let isJ ← icmp .eq code (← c32 j)
      v ← select isJ (← fconst .f32 bits.toUInt64) v
    store v (← iadd sm (← ext (← ishlImm i 2)))
  callLocalVoid barrier %[]
  let xs ← at_ sm xSmemOff
  for rep in List.range ((k + 255) / 256) do
    let i ← iadd tid (← c32 (256 * rep))
    when .ult i (← c32 k) do
      let v ← loadF32 (← iadd hv (← ext (← ishlImm i 2)))
      let idx ← iadd i (← ushrImm i 5)
      store v (← iadd xs (← ext (← ishlImm idx 2)))
  callLocalVoid barrier %[]
  pure xs

/-- **One warp's MXFP4 contractions**: each row `(blocks, scales)` against
    the staged activation, over `nBlocks` blocks of 32: chunks of eight
    blocks a warp with every load hoisted, the remaining
    blocks one at a time, then a butterfly. Every lane ends with every sum.

    **Four packed bytes to a lane.** One byte a lane, one block a warp, issues
    thirty-two one-byte loads for sixteen distinct bytes, and runs at a sixth
    of what memory can do. A `u32` a lane covers 128 contiguous bytes a warp,
    eight blocks; a lane's eight codes lie in one block, so it reads one scale,
    and code `j` is the nibble at bit `4j`. At this model's widths there are
    ninety blocks: eleven chunks and two left over.

    **Straight-line, loads first.** The block count is known when the body is
    built, so the chunks are emitted flat with all their weight words, then
    all their scale bytes, loaded before any arithmetic: issuing them together
    is the only way a warp has more than one request in flight. The cost is
    two registers a chunk. -/
def dotRows (sm xs : V .i64) (lane : V .i32) (rows : List (V .i64 × V .i64)) (nBlocks : Nat) :
    Prog V L (List (V .f32)) := do
  let tab ← iadd sm (← ext (← ishlImm lane 2))
  let lookup (nib : V .i32) : Prog V L (V .f32) := do
    loadF32 (← iadd tab (← ext (← ishlImm nib 7)))
  let e8m0 (sb : V .i32) : Prog V L (V .f32) := do bitcast .f32 (← ishlImm sb 23)
  let zero ← fconst .f32 0
  let mut accs : List (V .f32) := rows.map fun _ => zero
  -- whole chunks: eight blocks a warp, four bytes a lane
  let nChunks := nBlocks / 8
  let wLane ← ext (← imul lane (← c32 4))
  let sLane ← ushrImm lane 2
  let sLane64 ← ext sLane
  let rowBases ← rows.mapM fun (wb, sb) => do pure (← iadd wb wLane, ← iadd sb sLane64)
  let xLane ← iadd (← imul lane (← c32 8)) sLane
  let xBase0 ← iadd xs (← ext (← ishlImm xLane 2))
  let packed ← (List.range nChunks).mapM fun c =>
    rowBases.mapM fun (wb, _) => do load32 (← at_ wb (c * 128))
  let scaleB ← (List.range nChunks).mapM fun c =>
    rowBases.mapM fun (_, sb) => do u8 (← at_ sb (c * 8))
  for (pws, c) in packed.zipIdx do
    let scs ← (scaleB.getD c []).mapM e8m0
    for j in [0, 1, 2, 3, 4, 5, 6, 7] do
      let xv ← loadF32 (← at_ xBase0 (c * 264 * 4 + 4 * j))
      let mut next : List (V .f32) := []
      for ((pw, acc), r) in (pws.zip accs).zipIdx do
        let nib ← band (← ushrImm pw (4 * j)) (← c32 0xF)
        let w ← fmul (← lookup nib) (scs.getD r xv)
        next := next ++ [← fma w xv acc]
      accs := next
  -- the tail: whole blocks one at a time
  let tailStart := nChunks * 8
  if tailStart < nBlocks then do
    let halfLane ← ext (← ushrImm lane 1)
    let isOdd ← icmp .eq (← band lane (← c32 1)) (← c32 1)
    let rowBasesT ← rows.mapM fun (wb, sb) => do pure (← iadd wb halfLane, sb)
    let xBaseT ← iadd xs (← ext (← ishlImm lane 2))
    let tail := (List.range (nBlocks - tailStart)).map (· + tailStart)
    let bytes ← tail.mapM fun blk => rowBasesT.mapM fun (wb, _) => do u8 (← at_ wb (blk * 16))
    let scaleT ← tail.mapM fun blk => rowBasesT.mapM fun (_, sb) => do u8 (← at_ sb blk)
    for (bs, k) in bytes.zipIdx do
      let blk := tailStart + k
      let xv ← loadF32 (← at_ xBaseT (blk * 33 * 4))
      let mut next : List (V .f32) := []
      for ((b32, acc), r) in (bs.zip accs).zipIdx do
        let nib ← select isOdd (← ushrImm b32 4) (← band b32 (← c32 0xF))
        let w ← fmul (← lookup nib) (← e8m0 ((scaleT.getD k []).getD r b32))
        next := next ++ [← fma w xv acc]
      accs := next
  -- reduce across the warp
  let mut sums : List (V .f32) := []
  for acc in accs do
    let mut a := acc
    for m in [16, 8, 4, 2, 1] do
      let o ← callLocal shflBfly %[← bitcast .i32 a, ← c32 m]
      a ← fadd a (← bitcast .f32 o)
    sums := sums ++ [a]
  pure sums

/-- A list's elements two at a time. -/
def pairs {α} : List α → List (α × α)
  | a :: b :: r => (a, b) :: pairs r
  | _ => []

/-- Row `r`'s weight blocks and scales. -/
def rowAt (blocks scales : V .i64) (rowBytes rowScales : Nat) (r : V .i32) :
    Prog V L (V .i64 × V .i64) := do
  pure (← iadd blocks (← imul (← ext r) (← c64n rowBytes)),
        ← iadd scales (← imul (← ext r) (← c64n rowScales)))

/-- **The down projection, plus its bias.** One CTA of `warpsPerCta` warps,
    each taking `rowsPerWarpDown` consecutive output rows `r < H` and
    contracting over `I`; lane 0 adds the bias. -/
def downGemvBias : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64, .i64])
  let cta := ps.head; let tid := ps.snd; let sm := (ps.drop 3).head
  let blocks := (ps.drop 4).head; let scales := (ps.drop 5).head
  let bias := (ps.drop 6).head; let hv := (ps.drop 7).head; let out := (ps.drop 8).head
  let wid ← ushrImm tid 5
  let lane ← band tid (← c32 31)
  let xs ← stage tid sm hv I
  let warp ← iadd (← imul cta (← c32 warpsPerCta)) wid
  let row0 ← imul warp (← c32 rowsPerWarpDown)
  let rowIx ← (List.range rowsPerWarpDown).mapM fun k => do iadd row0 (← c32 k)
  let rows ← rowIx.mapM (rowAt blocks scales rowBytesI rowScalesI)
  let sums ← dotRows sm xs lane rows (I / G)
  when .eq lane (← c32 0) do
    for (s, k) in sums.zipIdx do
      let off ← ext (← ishlImm (rowIx.getD k row0) 2)
      let bv ← loadF32 (← iadd bias off)
      store (← fadd s bv) (← iadd out off)

/-- **Gate and up, contracted and combined in one launch.** Each warp takes
    `rowsPerWarpGateUp` output rows `g < I`; row `g` of the weights is the
    gate and row `I + g` the up projection, both contracted in one traversal
    of the activation. Lane 0 adds the biases and applies gpt-oss's clamped
    SwiGLU: the gate clamped above at `limit`, the linear branch to
    `[-limit, limit]` and offset by one, and the sigmoid of `alpha·gate` as
    `1 / (1 + 2^(-alpha·gate·log2 e))` by the hardware's approximations. -/
def gateUpSwiglu : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64, .i64])
  let cta := ps.head; let tid := ps.snd; let sm := (ps.drop 3).head
  let blocks := (ps.drop 4).head; let scales := (ps.drop 5).head
  let bias := (ps.drop 6).head; let hv := (ps.drop 7).head; let out := (ps.drop 8).head
  let wid ← ushrImm tid 5
  let lane ← band tid (← c32 31)
  let xs ← stage tid sm hv H
  let warp ← iadd (← imul cta (← c32 warpsPerCta)) wid
  let row0 ← imul warp (← c32 rowsPerWarpGateUp)
  let outRows ← (List.range rowsPerWarpGateUp).mapM fun k => do
    let g ← iadd row0 (← c32 k)
    pure (g, ← iadd g (← c32 I))
  let rows ← (outRows.flatMap fun (g, u) => [g, u]).mapM (rowAt blocks scales rowBytesH rowScalesH)
  let sums ← dotRows sm xs lane rows (H / G)
  let f (b : UInt32) : Prog V L (V .f32) := fconst .f32 b.toUInt64
  when .eq lane (← c32 0) do
    for ((gRow, uRow), (gs, us)) in outRows.zip (pairs sums) do
      let biased (acc : V .f32) (r : V .i32) : Prog V L (V .f32) := do
        fadd acc (← loadF32 (← iadd bias (← ext (← ishlImm r 2))))
      let g ← fmin (← biased gs gRow) (← f swigluLimitBits)
      let u ← fmax (← fmin (← biased us uRow)
        (← f swigluLimitBits)) (← f negLimitBits)
      let t ← fmul (← fmul g (← f swigluAlphaBits)) (← f log2eBits)
      let e ← callLocal ex2Approx %[← fsub (← f 0) t]
      let sig ← callLocal rcpApprox %[← fadd e (← f oneBits)]
      let y ← fmul (← fmul g sig) (← fadd u (← f oneBits))
      store y (← iadd out (← ext (← ishlImm gRow 2)))

/-- The thread's index in the grid, blocks of `32 · warpsPerCta`. -/
def gidx (cta tid : V .i32) : Prog V L (V .i32) := do
  iadd (← imul cta (← c32 (32 * warpsPerCta))) tid

/-- **The gated sum of the chosen experts**: `out[i] := Σ_j y_j[i] · gates[j]`
    for `i < H`, a thread an element, the slots added in order by fused
    multiply-adds. The slots are summed here rather than accumulated in place
    by the down projection: four read-modify-write passes over the row would
    cost three extra traversals to save one launch. -/
def moeCombine : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64, .i64, .i64])
  let ys := [(ps.drop 4).head, (ps.drop 5).head, (ps.drop 6).head, (ps.drop 7).head]
  let gates := (ps.drop 8).head; let out := (ps.drop 9).head
  let i ← gidx ps.head ps.snd
  when .ult i (← c32 H) do
    let off ← ext (← ishlImm i 2)
    let mut acc ← fconst .f32 0
    for (y, j) in ys.zipIdx do
      acc ← fma (← loadF32 (← iadd y off)) (← loadF32 (← at_ gates (4 * j))) acc
    store acc (← iadd out off)

/-- A float's bfloat16, rounded to nearest with ties to even: adding `0x7FFF`
    plus the kept bit's value before the shift. Truncating instead biases every
    value toward zero, and a bias does not cancel across a dot product. -/
def rneBf16 (x : V .f32) : Prog V L (V .i32) := do
  let u ← bitcast .i32 x
  let r ← iadd (← band (← ushrImm u 16) (← c32 1)) (← c32 0x7FFF)
  ushrImm (← iadd u r) 16

/-- **Narrow a row to bf16, a pair of elements a thread**, packed into one
    word; the word at `pairs` is how many pairs, so one kernel serves every width.
    cuBLAS takes no bf16 matrix against an f32 vector, so an activation
    meeting a bf16 weight is narrowed first. -/
def narrowBf16 : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64])
  let inp := (ps.drop 4).head; let out := (ps.drop 5).head; let pairs := (ps.drop 6).head
  let i ← gidx ps.head ps.snd
  when .ult i (← load32 pairs) do
    let a ← iadd inp (← ext (← ishlImm i 3))
    let lo ← rneBf16 (← loadF32 a)
    let hi ← rneBf16 (← loadF32 (← at_ a 4))
    storeI32 (← bor lo (← ishlImm hi 16)) (← iadd out (← ext (← ishlImm i 2)))

/-- **bf16 to f32**, a packed pair a thread: exact, since a bf16 is the f32
    with its low sixteen bits clear. The word at `pairs` is how many pairs. -/
def widenBf16 : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64])
  let inp := (ps.drop 4).head; let out := (ps.drop 5).head; let pairs := (ps.drop 6).head
  let i ← gidx ps.head ps.snd
  when .ult i (← load32 pairs) do
    let packed ← load32 (← iadd inp (← ext (← ishlImm i 2)))
    let a ← iadd out (← ext (← ishlImm i 3))
    store (← bitcast .f32 (← ishlImm packed 16)) a
    store (← bitcast .f32 (← band packed (← c32 0xFFFF0000))) (← at_ a 4)

/-- **The router's decision**: the `TOPK` largest of `NE` logits, their
    indices to `chosen` and the softmax of their logits to `gates` (the other
    gates zero). Threads below `NE` stage the logits in shared memory and zero
    the gates; lane 0 then takes the largest still standing `TOPK` times,
    striking each out with `-inf`, the first index winning a tie, and weights
    them by `2^((l - max)·log2 e)` over their sum, by the hardware's
    approximations. One lane serially: a hundred and twenty-eight comparisons,
    which saves the host round trip that choosing the experts would cost. -/
def routerTop : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64])
  let tid := ps.snd; let sm := (ps.drop 3).head
  let logits := (ps.drop 4).head; let gates := (ps.drop 5).head; let chosen := (ps.drop 6).head
  let at4 (b : V .i64) (i : V .i32) : Prog V L (V .i64) := do iadd b (← ext (← ishlImm i 2))
  when .ult tid (← c32 NE) do
    store (← loadF32 (← at4 logits tid)) (← at4 sm tid)
    store (← fconst .f32 0) (← at4 gates tid)
  callLocalVoid barrier %[]
  when .eq tid (← c32 0) do
    let ninf ← fconst .f32 0xFF800000
    let mut tops : List (V .f32) := []
    for j in List.range TOPK do
      let mut best := ninf
      let mut bi ← c32 0
      for i in List.range NE do
        let v ← loadF32 (← at_ sm (4 * i))
        let gt ← fcmp .gt v best
        best ← fmax best v
        bi ← select gt (← c32 i) bi
      storeI32 bi (← at_ chosen (4 * j))
      store ninf (← at4 sm bi)
      tops := tops ++ [best]
    let mx := tops.headD ninf
    let log2e ← f32c log2eBits
    let mut es : List (V .f32) := []
    let mut sum ← fconst .f32 0
    for t in tops do
      let e : V .f32 ← callLocal ex2Approx %[← fmul (← fsub t mx) log2e]
      es := es ++ [e]
      sum ← fadd sum e
    let inv ← callLocal rcpApprox %[sum]
    for (e, k) in es.zipIdx do
      store (← fmul e inv) (← at_ gates (4 * k))

/-- The larger of `(best, bi)` and `(v, i)`: `best`'s index when equal. -/
def better (i : V .i32) (v best : V .f32) (bi : V .i32) : Prog V L (Vals V [.f32, .i32]) := do
  let gt ← fcmp .gt v best
  pure %[← fmax best v, ← select gt i bi]

/-- One lane's stride over `i = tid, tid + 32, … < n`, carrying a best value
    and its index from `(-∞, 0)` through `step`. -/
def laneMax (tid n : V .i32) (step : V .i32 → V .f32 → V .i32 → Prog V L (Vals V [.f32, .i32])) :
    Prog V L (Vals V [.f32, .i32]) := do
  let b0 ← f32c negInfBits
  let i0 ← c32 0
  wloop %[tid, b0, i0] (fun cs => pure (contIfULt cs.head n, %[cs.snd, cs.thd], ()))
    (fun cs _ => do
      let r ← step cs.head cs.snd cs.thd
      pure %[← iadd cs.head (← c32 32), r.head, r.snd])

/-- The 32 lanes' bests to shared memory (value at `4·tid`, index at
    `128 + 4·tid`), and lane 0 writes the index of the largest, the lowest
    lane's on a tie. -/
def pickLanes (tid : V .i32) (sm : V .i64) (r : Vals V [.f32, .i32]) (out : V .i64) :
    Prog V L Unit := do
  let va ← iadd sm (← ext (← ishlImm tid 2))
  store r.head va
  storeI32 r.snd (← at_ va 128)
  callLocalVoid barrier %[]
  when .eq tid (← c32 0) do
    let mut bv ← loadF32 sm
    let mut bx ← load32 (← at_ sm 128)
    for k in List.range' 1 31 do
      let b ← better (← load32 (← at_ sm (128 + 4 * k))) (← loadF32 (← at_ sm (4 * k))) bv bx
      bv := b.head; bx := b.snd
    storeI32 bx out

/-- **The greedy token**: the index of the largest of the `n` logits, `n` the
    word at `meta`, the lowest index on a tie. One warp: each lane strides the
    row keeping its best, and lane 0 picks among the thirty-two. -/
def argmaxBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64])
  let tid := ps.snd; let sm := (ps.drop 3).head
  let logits := (ps.drop 4).head; let out := (ps.drop 5).head; let m := (ps.drop 6).head
  let n ← load32 m
  let r ← laneMax tid n fun i best bi => do
    better i (← loadF32 (← iadd logits (← ext (← ishlImm i 2)))) best bi
  pickLanes tid sm r out

/-- **A sample from the tempered softmax**: `argmax_i (logit_i / T + g_i)`
    with `g_i` standard Gumbel is a draw from `softmax(logit / T)`, so sampling
    costs what the argmax costs, with no normalising constant and no sort.

    The words at `meta` are `n`, a seed, `1/T` and `ln(min_p)`. min-p keeps a
    token whose probability is at least `min_p` times the most likely one's,
    which in tempered space is the threshold `v ≥ vmax + ln(min_p)`: one more
    sweep, for the largest. The noise is a hash of `(i, seed)`, a uniform `u`
    in `(0, 1)` from its top 24 bits offset by half an ulp, and
    `g = -ln(-ln u)` by the hardware's `lg2`; a sample is a function of its
    inputs and the seed. Greedy decoding (`1/T = 0`) is the argmax kernel's. -/
def sampleBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64])
  let tid := ps.snd; let sm := (ps.drop 3).head
  let logits := (ps.drop 4).head; let out := (ps.drop 5).head; let m := (ps.drop 6).head
  let n ← load32 m
  let seed ← load32 (← at_ m 4)
  let invT ← loadF32 (← at_ m 8)
  let lnMinP ← loadF32 (← at_ m 12)
  let tempered (i : V .i32) : Prog V L (V .f32) := do
    fmul (← loadF32 (← iadd logits (← ext (← ishlImm i 2)))) invT
  -- the largest tempered logit
  let ninf ← f32c negInfBits
  let mx ← wloop %[tid, ninf] (fun cs => pure (contIfULt cs.head n, %[cs.snd], ()))
    (fun cs _ => do pure %[← iadd cs.head (← c32 32), ← fmax cs.snd (← tempered cs.head)])
  store mx.head (← iadd sm (← ext (← ishlImm tid 2)))
  callLocalVoid barrier %[]
  let mut vmax := ninf
  for k in List.range 32 do
    vmax ← fmax vmax (← loadF32 (← at_ sm (4 * k)))
  callLocalVoid barrier %[]
  let thresh ← fadd vmax lnMinP
  let ln2 ← fconst .f32 0x3F317218
  let r ← laneMax tid n fun i best bi => do
    let v ← tempered i
    let keep ← fcmp .ge v thresh
    ifte (jTys := [.f32, .i32]) .ne keep (← iconst .i8 0)
      (do
        let h1 ← imul (← bxor i seed) (← c32 0x9E3779B1)
        let h2 ← bxor h1 (← ushrImm h1 15)
        let h3 ← imul h2 (← c32 0x85EBCA6B)
        let h4 ← bxor h3 (← ushrImm h3 13)
        let u ← fmul (← fadd (← fcvtFromUint .f32 (← ushrImm h4 8)) (← fconst .f32 0x3F000000))
          (← fconst .f32 0x33800000)
        let l1 : V .f32 ← callLocal lg2Approx %[u]
        let g1 ← fneg (← fmul l1 ln2)
        let l2 : V .f32 ← callLocal lg2Approx %[g1]
        let g ← fneg (← fmul l2 ln2)
        better i (← fadd v g) best bi)
      (pure %[best, bi])
  callLocalVoid barrier %[]
  pickLanes tid sm r out

/-- An FP4 (E2M1) code's float, from its low four bits: the sign bit 3, the
    exponent bits 1–2 and the mantissa bit 0; exponent zero is `0` or `0.5`.
    Integer instructions and a reinterpretation, no memory and no table. -/
def fp4 (nib : V .i32) : Prog V L (V .f32) := do
  let m ← band nib (← c32 1)
  let e ← band (← ushrImm nib 1) (← c32 3)
  let sgn ← ishlImm (← band (← ushrImm nib 3) (← c32 1)) 31
  let normal ← bor (← ishlImm (← iadd e (← c32 126)) 23) (← ishlImm m 22)
  let sub ← imul m (← c32 0x3F000000)
  let mag ← select (← icmp .eq e (← c32 0)) sub normal
  bitcast .f32 (← bor mag sgn)

/-- **A packed row dequantised to `Float32`**: one CTA a row, one lane an
    element of a block, the word at `meta` the blocks a row. The prefill
    path's kernel: it lands `mxDeq` of its input, after which a GEMM reads
    ordinary floats. The codes are decoded arithmetically, with no table and
    no shared memory. -/
def dequantBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64])
  let row := ps.head; let lane := ps.snd
  let blocks := (ps.drop 4).head; let scales := (ps.drop 5).head
  let out := (ps.drop 6).head; let m := (ps.drop 7).head
  let nBlocks ← load32 m
  let isOdd ← icmp .eq (← band lane (← c32 1)) (← c32 1)
  let halfLane ← ushrImm lane 1
  let wBase ← iadd blocks (← ext (← imul row (← imul nBlocks (← c32 16))))
  let sBase ← iadd scales (← ext (← imul row nBlocks))
  let oBase ← iadd out (← ishlImm (← ext (← imul row (← imul nBlocks (← c32 32)))) 2)
  let i0 ← c32 0
  let _ ← wloop %[i0] (fun cs => pure (contIfULt cs.head nBlocks, %[], ()))
    (fun cs _ => do
      let blk := cs.head
      let b ← u8 (← iadd wBase (← ext (← iadd (← imul blk (← c32 16)) halfLane)))
      let nib ← select isOdd (← ushrImm b 4) (← band b (← c32 0xF))
      -- an E8M0 scale is a float's exponent field: 2^(s-127), zero at 0,
      -- NaN at 255 (which the converter rejects)
      let sc ← bitcast .f32 (← ishlImm (← u8 (← iadd sBase (← ext blk))) 23)
      let w ← fmul (← fp4 nib) sc
      store w (← iadd oBase (← ishlImm (← ext (← iadd (← imul blk (← c32 32)) lane)) 2))
      pure %[← iadd blk (← c32 1)])

/-- The two bf16 in a loaded word, as `Float32`: bf16 is the top sixteen bits
    of the `Float32` it rounds to, so the low half shifts up and the high half
    is masked in place. -/
def unpack (w : V .i32) : Prog V L (V .f32 × V .f32) := do
  pure (← bitcast .f32 (← ishlImm w 16), ← bitcast .f32 (← band w (← c32 0xFFFF0000)))

/-- A float through the warp's butterfly at `mask`, added to itself. -/
def bflyAdd (a : V .f32) (mask : Nat) : Prog V L (V .f32) := do
  let o ← callLocal shflBfly %[← bitcast .i32 a, ← c32 mask]
  fadd a (← bitcast .f32 o)

/-- **One tile of the sequence, one key head, scores never stored.**

    Grid is one CTA per tile, block is `NKV_K` warps, and warp `w` takes key
    head `w`. A lane owns two elements of the head -- `2L` and `2L+1` -- so the
    key row is one coalesced word per lane.

    Per key: form the eight scores this head serves (each a dot over the head,
    so a butterfly across the warp), fold them into the running `(m, l)`, and
    accumulate the value row. The next key's words are loaded before this
    one's are used, which keeps two rows in flight on a serial walk; the read
    one past the last key lands in the row of slack the caches carry.

    Writes `(m, l, acc[HD_K])` per `(tile, head, query)` for the combine pass;
    the sink belongs to the head and is applied there, once. -/
def attnTileBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64, .i64])
  let cta := ps.head; let tid := ps.snd
  let kp := (ps.drop 4).head; let vp := (ps.drop 5).head; let qp := (ps.drop 6).head
  let pp := (ps.drop 7).head; let mp := (ps.drop 8).head
  let wid ← ushrImm tid 5
  let lane ← band tid (← c32 31)
  let seqLen ← load32 (← at_ mp (4 * GptOssAttention.M_SEQ))
  let tileSz ← load32 (← at_ mp (4 * GptOssAttention.M_TILESZ))
  let kvStride ← load32 (← at_ mp (4 * GptOssAttention.M_KVSTRIDE))
  -- this warp's slice: tile `cta`, key head `wid`
  let s0 ← imul cta tileSz
  let sEnd ← umin (← iadd s0 tileSz) seqLen
  -- the eight queries this key head serves, two elements to a lane
  let qWord ← iadd (← imul wid (← c32 (GQA_K * HD_K / 2))) lane
  let qBase ← iadd qp (← imul (← ext qWord) (← c64n 4))
  let qs ← (List.range GQA_K).mapM fun g => do
    unpack (← load32 (← at_ qBase (g * (HD_K / 2) * 4)))
  -- the walk's addresses advance by one key row a step
  let kWord ← iadd (← iadd (← imul wid kvStride) lane) (← imul s0 (← c32 (HD_K / 2)))
  let kOff ← imul (← ext kWord) (← c64n 4)
  let kAddr0 ← iadd kp kOff
  let vAddr0 ← iadd vp kOff
  let kw0 ← load32 kAddr0
  let vw0 ← load32 vAddr0
  let ninf ← fconst .f32 0xFF800000
  let zero ← fconst .f32 0
  -- running (m, l, a0, a1) a query, four floats to a query
  let st0 : List (V .f32) := (List.range GQA_K).flatMap fun _ => [ninf, zero, zero, zero]
  let carry0 := Vals.append %[s0, kAddr0, vAddr0, kw0, vw0]
    (Vals.ofFn (n := 4 * GQA_K) fun i => st0.getD i.val zero)
  let fin ← wloop carry0 (fun cs => pure (contIfULt cs.head sEnd, cs, ()))
    (fun cs _ => do
      let sIx := cs.head; let kAddr := cs.snd; let vAddr := cs.thd
      let kw := cs.fth; let vw := cs.fif
      let st := Vals.uniformToList (cs.drop 5 : Vals V (List.replicate (4 * GQA_K) .f32))
      let (k0, k1) ← unpack kw
      let (v0, v1) ← unpack vw
      let kAddr' ← at_ kAddr (HD_K / 2 * 4)
      let vAddr' ← at_ vAddr (HD_K / 2 * 4)
      let kw' ← load32 kAddr'
      let vw' ← load32 vAddr'
      -- one partial dot a query, then one butterfly for all of them
      let mut dots : List (V .f32) := []
      for (q0, q1) in qs do
        dots := dots ++ [← fma q1 k1 (← fmul q0 k0)]
      for mask in [16, 8, 4, 2, 1] do
        let mut next : List (V .f32) := []
        for d in dots do
          next := next ++ [← bflyAdd d mask]
        dots := next
      -- fold each score into its running (m, l) and rescale the accumulator
      let scale ← fconst .f32 attScaleBits.toUInt64
      let mut st' : List (V .f32) := []
      for (sc0, g) in dots.zipIdx do
        let m := st.getD (4 * g) zero; let l := st.getD (4 * g + 1) zero
        let a0 := st.getD (4 * g + 2) zero; let a1 := st.getD (4 * g + 3) zero
        let sc ← fmul sc0 scale
        let mNew ← fmax m sc
        let corr ← expA (← fsub m mNew)
        let pr ← expA (← fsub sc mNew)
        let l' ← fma l corr pr
        let a0' ← fma pr v0 (← fmul a0 corr)
        let a1' ← fma pr v1 (← fmul a1 corr)
        st' := st' ++ [mNew, l', a0', a1']
      pure (Vals.append %[← iadd sIx (← c32 1), kAddr', vAddr', kw', vw']
        (Vals.ofFn (n := 4 * GQA_K) fun i => st'.getD i.val zero)))
  let st := Vals.uniformToList (fin.drop 5 : Vals V (List.replicate (4 * GQA_K) .f32))
  -- this warp's partials: one block of `attPartialFloats` a query
  let pBlk ← imul (← iadd (← imul cta (← c32 NKV_K)) wid) (← c32 GQA_K)
  for g in List.range GQA_K do
    let ix ← iadd pBlk (← c32 g)
    let addr ← iadd pp (← imul (← ext ix) (← c64n (attPartialFloats * 4)))
    when .eq lane (← c32 0) do
      store (st.getD (4 * g) zero) addr
      store (st.getD (4 * g + 1) zero) (← at_ addr 4)
    let aAddr ← iadd addr (← ishlImm (← ext lane) 3)
    store (st.getD (4 * g + 2) zero) (← at_ aAddr 8)
    store (st.getD (4 * g + 3) zero) (← at_ aAddr 12)

/-- **The tiles, merged, and the sink applied once.**

    One CTA of one warp per query head. Two partial results `(m, l, acc)` merge
    the way the tile loop folds a single key: rescale both by `exp(m - max)`
    and add. The learned sink joins the denominator here and nowhere else.
    The tile count comes from the meta buffer, which the layer uploads anyway. -/
def attnCombineBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64])
  let head := ps.head; let tid := ps.snd
  let pp := (ps.drop 4).head; let sp := (ps.drop 5).head
  let op := (ps.drop 6).head; let np := (ps.drop 7).head
  let lane ← band tid (← c32 31)
  let nTiles ← load32 (← at_ np (4 * GptOssAttention.M_NTILES))
  -- this head's partials within a tile's block; the tile stride is every head's
  let hBase ← iadd pp (← imul (← ext head) (← c64n (attPartialFloats * 4)))
  let tStride := NKV_K * GQA_K * attPartialFloats * 4
  -- pass one: the largest maximum any tile saw
  let ninf ← fconst .f32 0xFF800000
  let r1 ← wloop %[← c32 0, hBase, ninf] (fun cs => pure (contIfULt cs.head nTiles, %[cs.thd], ()))
    (fun cs _ => do
      let m ← loadF32 cs.snd
      pure %[← iadd cs.head (← c32 1), ← at_ cs.snd tStride, ← fmax cs.thd m])
  -- the sink shares the denominator, so it belongs in the maximum too
  let sink ← loadF32 (← iadd sp (← ishlImm (← ext head) 2))
  let mAll ← fmax r1.head sink
  -- pass two: rescale every tile onto that maximum and add
  let zero ← fconst .f32 0
  let addr2 ← iadd hBase (← ishlImm (← ext lane) 3)
  let r2 ← wloop %[← c32 0, hBase, addr2, zero, zero, zero]
    (fun cs => pure (contIfULt cs.head nTiles, %[(cs.drop 3).head, (cs.drop 4).head, (cs.drop 5).head], ()))
    (fun cs _ => do
      let m ← loadF32 cs.snd
      let l ← loadF32 (← at_ cs.snd 4)
      let w ← expA (← fsub m mAll)
      let lAll ← fma l w (cs.drop 3).head
      let t0 ← loadF32 (← at_ cs.thd 8)
      let t1 ← loadF32 (← at_ cs.thd 12)
      pure %[← iadd cs.head (← c32 1), ← at_ cs.snd tStride, ← at_ cs.thd tStride, lAll,
        ← fma t0 w (cs.drop 4).head, ← fma t1 w (cs.drop 5).head])
  -- the sink's own term, and then the division softmax would have done
  let lAll ← fadd r2.head (← expA (← fsub sink mAll))
  let inv ← callLocal rcpApprox %[lAll]
  let oAddr ← iadd op (← ishlImm (← ext (← iadd (← imul head (← c32 HD_K)) (← ishlImm lane 1))) 2)
  store (← fmul r2.snd inv) oAddr
  store (← fmul r2.thd inv) (← at_ oAddr 4)

end

/-- **The down projection, plus its bias**, as a kernel. -/
def downGemv : Kernel :=
  { name := "mxfp4_down_gemv_bias", params := [.i64, .i64, .i64, .i64, .i64],
    code := emit downGemvBias (idParams ++ [.i64, .i64, .i64, .i64, .i64]), smem := smemBytes }

/-- **Gate and up with the clamped SwiGLU**, as a kernel. -/
def gateUp : Kernel :=
  { name := "mxfp4_gate_up_swiglu_gemv", params := [.i64, .i64, .i64, .i64, .i64],
    code := emit gateUpSwiglu (idParams ++ [.i64, .i64, .i64, .i64, .i64]), smem := smemBytes }

def moeCombine4 : Kernel :=
  { name := "mxfp4_moe_combine4", params := [.i64, .i64, .i64, .i64, .i64, .i64],
    code := emit moeCombine (idParams ++ [.i64, .i64, .i64, .i64, .i64, .i64]) }

def narrow : Kernel :=
  { name := "narrow_bf16", params := [.i64, .i64, .i64],
    code := emit narrowBf16 (idParams ++ [.i64, .i64, .i64]) }

def widen : Kernel :=
  { name := "widen_bf16", params := [.i64, .i64, .i64],
    code := emit widenBf16 (idParams ++ [.i64, .i64, .i64]) }

def routerTop4 : Kernel :=
  { name := "router_top4", params := [.i64, .i64, .i64],
    code := emit routerTop (idParams ++ [.i64, .i64, .i64]), smem := NE * 4 }

def argmax : Kernel :=
  { name := "argmax_logits", params := [.i64, .i64, .i64],
    code := emit argmaxBody (idParams ++ [.i64, .i64, .i64]), smem := 256 }

def sample : Kernel :=
  { name := "sample_logits", params := [.i64, .i64, .i64],
    code := emit sampleBody (idParams ++ [.i64, .i64, .i64]), smem := 256 }

def dequant : Kernel :=
  { name := "mxfp4_dequant_f32", params := [.i64, .i64, .i64, .i64],
    code := emit dequantBody (idParams ++ [.i64, .i64, .i64, .i64]) }

/-- One tile of the sequence for each key head, as a kernel: blocks of
    `NKV_K` warps. Its partials' addresses are `mul.wide` products: with
    64-bit ones `ptxas` takes the tile loop's control off the uniform datapath
    and the loop runs 8% slower. -/
def attnTile : Kernel :=
  { name := "gptoss_attn_tile", params := [.i64, .i64, .i64, .i64, .i64],
    code := emit attnTileBody (idParams ++ [.i64, .i64, .i64, .i64, .i64]), mulWide := true }

/-- The tiles merged, one warp a query head, as a kernel. -/
def attnCombine : Kernel :=
  { name := "gptoss_attn_combine", params := [.i64, .i64, .i64, .i64],
    code := emit attnCombineBody (idParams ++ [.i64, .i64, .i64, .i64]), warp := true }

/-- A body kernel as its own module, entry named `main`. -/
def moduleOf (k : Kernel) : String :=
  match Kernel.ptx { k with name := "main" } with
  | .ok s => s
  | .error _ => ""

/-- The body kernels as one module, for the standalone harnesses. -/
def moduleText : String :=
  match Kernel.module [gateUp, downGemv, moeCombine4, narrow, widen, routerTop4, argmax, sample, dequant,
      attnTile, attnCombine] with
  | .ok s => s
  | .error _ => ""

end Body

/-- The kernels a decode step launches, in slot order. -/
def gptossPtx : List String :=
  [Body.moduleOf Body.gateUp, Body.moduleOf Body.downGemv, Body.moduleOf Body.moeCombine4,
   Body.moduleOf Body.routerTop4]

end GptOssKernels

/-! `Body.moduleText` is emitted by `tools/dump_gptoss_kernels.lean` rather
    than by a `main` here: a `lean_exe` in this package is a *generator*, and
    every generator is expected to produce an artifact. -/
