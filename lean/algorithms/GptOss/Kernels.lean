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

/-- An expert's six pieces, in the order the converter writes a row of
    `experts.bin` and a slot of the device pool holds them: gate/up's blocks,
    scales and bias, then down's. -/
def expertPieces : List Nat :=
  [ 2 * I * rowBytesH, 2 * I * rowScalesH, 2 * I * 4
  , H * rowBytesI, H * rowScalesI, H * 4 ]
def EXPERT_BYTES : Nat := expertPieces.foldl (· + ·) 0
def expertPieceOff (t : Nat) : Nat := ((expertPieces.take t).foldl (· + ·) 0)

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

/-- The fewest keys a tile is ever given, and what its length is a multiple
    of: one sub-tile of the tensor cores' work (`DT_KEYS`). -/
def ATT_TILE_MIN : Nat := 32

/-- Tiles a layer is aimed at: a block a tile and key head, `NKV_K` a tile,
    so 392 blocks of one warp, fourteen to each of the card's 28
    multiprocessors (as many as their shared memory holds), one wave. -/
def ATT_TILES_AIM : Nat := 49

/-- Tile places the combine reads, two a lane: at least the most tiles a
    layer is split into (`ATT_TILES_AIM`). -/
def COMB_TILES : Nat := 64

/-- The height of the partials buffer: at least the most tiles a layer is
    split into (`ATT_TILES_AIM`, or fewer). -/
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
  -- a lane's eight codes share a block, so their products are summed first
  -- and the block's scale applied once; and code `j`'s table offset,
  -- `nibble · 128`, is one shift and one mask of the word
  let tabAt (pw : V .i32) (j : Nat) : Prog V L (V .f32) := do
    let sh ← if 4 * j ≥ 7 then ushrImm pw (4 * j - 7) else ishlImm pw (7 - 4 * j)
    loadF32 (← iadd tab (← ext (← band sh (← c32 0x780))))
  for (pws, c) in packed.zipIdx do
    let scs ← (scaleB.getD c []).mapM e8m0
    let mut parts : List (V .f32) := pws.map fun _ => zero
    for j in [0, 1, 2, 3, 4, 5, 6, 7] do
      let xv ← loadF32 (← at_ xBase0 (c * 264 * 4 + 4 * j))
      let mut next : List (V .f32) := []
      for (pw, part) in pws.zip parts do
        next := next ++ [← fma (← tabAt pw j) xv part]
      parts := next
    let mut next : List (V .f32) := []
    for ((part, acc), r) in (parts.zip accs).zipIdx do
      next := next ++ [← fma part (scs.getD r zero) acc]
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
def downCore (cta tid : V .i32) (sm blocks scales bias hv out : V .i64) : Prog V L Unit := do
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

/-- **Where a launch over all `TOPK` experts finds its expert.** CTA `cta`
    takes the router's `j = cta / per`th choice and is CTA `cta - j·per` of
    it; expert `e = chosen[j]` of layer `mt[layerWord]` sits in slot
    `slotOf[layer · NE + e]` of the device pool, `EXPERT_BYTES` a slot, so its
    piece `t` is at `pool + slot·EXPERT_BYTES + expertPieceOff t`. One launch
    of `TOPK · per` CTAs rather than `TOPK` of `per`: four times the CTAs in
    flight, and a launch's fixed costs paid once. Everything the lookup needs
    is on the device, so the host need not wait for the router to launch it;
    the weights are reached from the pool parameter, so their loads are
    read-only global ones. -/
def expertAt (pool slotOf chosen mt : V .i64) (layerWord : Nat) (cta : V .i32) (per : Nat) :
    Prog V L (V .i32 × V .i32 × V .i64) := do
  let j ← udiv cta (← c32 per)
  let jc ← isub cta (← imul j (← c32 per))
  let e ← load32 (← iadd chosen (← ext (← ishlImm j 2)))
  let layer ← load32 (← at_ mt (4 * layerWord))
  let key ← iadd (← imul layer (← c32 NE)) e
  let slot ← load32 (← iadd slotOf (← ext (← ishlImm key 2)))
  pure (jc, j, ← iadd pool (← imul (← ext slot) (← c64n EXPERT_BYTES)))

/-- The down projection of one launch's experts: entry params
    `(pool, slotOf, chosen, meta, hid, y)` (`expertAt`), expert `j`'s input at
    `hid + 4·I·j` and its output at `y + 4·H·j`. -/
def downAll (layerWord : Nat) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64, .i64, .i64])
  let (cta, j, row) ← expertAt (ps.drop 4).head (ps.drop 5).head (ps.drop 6).head
    (ps.drop 7).head layerWord ps.head (H / (warpsPerCta * rowsPerWarpDown))
  let slice (b : V .i64) (n : Nat) : Prog V L (V .i64) := do
    iadd b (← ext (← imul j (← c32 n)))
  downCore cta ps.snd (ps.drop 3).head (← iaddImm row (expertPieceOff 3))
    (← iaddImm row (expertPieceOff 4)) (← iaddImm row (expertPieceOff 5))
    (← slice (ps.drop 8).head (4 * I)) (← slice (ps.drop 9).head (4 * H))

def downGemvBias : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64, .i64])
  downCore ps.head ps.snd (ps.drop 3).head (ps.drop 4).head (ps.drop 5).head
    (ps.drop 6).head (ps.drop 7).head (ps.drop 8).head

/-- **Gate and up, contracted and combined in one launch.** Each warp takes
    `rowsPerWarpGateUp` output rows `g < I`; row `g` of the weights is the
    gate and row `I + g` the up projection, both contracted in one traversal
    of the activation. Lane 0 adds the biases and applies gpt-oss's clamped
    SwiGLU: the gate clamped above at `limit`, the linear branch to
    `[-limit, limit]` and offset by one, and the sigmoid of `alpha·gate` as
    `1 / (1 + 2^(-alpha·gate·log2 e))` by the hardware's approximations. -/
def gateUpCore (cta tid : V .i32) (sm blocks scales bias hv out : V .i64) : Prog V L Unit := do
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

def gateUpSwiglu : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64, .i64])
  gateUpCore ps.head ps.snd (ps.drop 3).head (ps.drop 4).head (ps.drop 5).head
    (ps.drop 6).head (ps.drop 7).head (ps.drop 8).head

/-- Gate and up of one launch's experts: entry params `(pool, slotOf, chosen,
    meta, x, hid)` (`expertAt`), expert `j`'s output at `hid + 4·I·j`. -/
def gateUpAll (layerWord : Nat) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64, .i64, .i64])
  let (cta, j, row) ← expertAt (ps.drop 4).head (ps.drop 5).head (ps.drop 6).head
    (ps.drop 7).head layerWord ps.head (I / (warpsPerCta * rowsPerWarpGateUp))
  gateUpCore cta ps.snd (ps.drop 3).head row (← iaddImm row (expertPieceOff 1))
    (← iaddImm row (expertPieceOff 2)) (ps.drop 8).head
    (← iadd (ps.drop 9).head (← ext (← imul j (← c32 (4 * I)))))

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

/-- Threads of the head's kernels (the argmax and the sampler): one block. -/
def HEAD_T : Nat := 1024

/-- One thread's stride over `i = tid, tid + HEAD_T, … < n`, carrying a best
    value and its index from `(-∞, 0)` through `step`. -/
def laneMax (tid n : V .i32) (step : V .i32 → V .f32 → V .i32 → Prog V L (Vals V [.f32, .i32])) :
    Prog V L (Vals V [.f32, .i32]) := do
  let b0 ← f32c negInfBits
  let i0 ← c32 0
  wloop %[tid, b0, i0] (fun cs => pure (contIfULt cs.head n, %[cs.snd, cs.thd], ()))
    (fun cs _ => do
      let r ← step cs.head cs.snd cs.thd
      pure %[← iadd cs.head (← c32 HEAD_T), r.head, r.snd])

/-- The best `(value, index)` across a warp, by butterfly: the larger value,
    the lower index when they are equal. Every lane ends with it. -/
def warpBest (r : Vals V [.f32, .i32]) : Prog V L (Vals V [.f32, .i32]) := do
  let mut v := r.head; let mut ix := r.snd
  for m in [16, 8, 4, 2, 1] do
    let ov ← bitcast .f32 (← callLocal shflBfly %[← bitcast .i32 v, ← c32 m])
    let oi ← callLocal shflBfly %[ix, ← c32 m]
    let gt ← fcmp .gt ov v
    let eq ← fcmp .eq ov v
    let low ← select (← icmp .ult oi ix) oi ix
    ix ← select gt oi (← select eq low ix)
    v ← fmax v ov
  pure %[v, ix]

/-- The largest value across the block's `HEAD_T` threads, to every thread:
    a butterfly a warp, the warps' through shared memory at `sm`. -/
def blockMaxF (tid : V .i32) (sm : V .i64) (x : V .f32) : Prog V L (V .f32) := do
  let mut v := x
  for m in [16, 8, 4, 2, 1] do
    v ← fmax v (← bitcast .f32 (← callLocal shflBfly %[← bitcast .i32 v, ← c32 m]))
  when .eq (← band tid (← c32 31)) (← c32 0) do
    store v (← iadd sm (← ext (← ishlImm (← ushrImm tid 5) 2)))
  callLocalVoid barrier %[]
  let mut w ← loadF32 sm
  for k in List.range' 1 (HEAD_T / 32 - 1) do
    w ← fmax w (← loadF32 (← at_ sm (4 * k)))
  callLocalVoid barrier %[]
  pure w

/-- The block's best: each warp's (`warpBest`) to shared memory (value at
    `4·warp`, index at `128 + 4·warp`), then the first warp's over those, and
    thread 0 writes the index of the largest, the lowest index on a tie. -/
def pickLanes (tid : V .i32) (sm : V .i64) (r : Vals V [.f32, .i32]) (out : V .i64) :
    Prog V L Unit := do
  let w ← warpBest r
  let lane ← band tid (← c32 31)
  when .eq lane (← c32 0) do
    let va ← iadd sm (← ext (← ishlImm (← ushrImm tid 5) 2))
    store w.head va
    storeI32 w.snd (← at_ va 128)
  callLocalVoid barrier %[]
  when .ult tid (← c32 32) do
    let va ← iadd sm (← ext (← ishlImm lane 2))
    let b ← warpBest %[← loadF32 va, ← load32 (← at_ va 128)]
    when .eq tid (← c32 0) do storeI32 b.snd out

/-- **The greedy token**: the index of the largest of the `n` logits, `n` the
    word at `meta`, the lowest index on a tie. One block of `HEAD_T`: each
    thread strides the row keeping its best, and `pickLanes` reduces them. -/
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
    (fun cs _ => do pure %[← iadd cs.head (← c32 HEAD_T), ← fmax cs.snd (← tempered cs.head)])
  let vmax ← blockMaxF tid sm mx.head
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

/-- Keys a tile is walked in, a sub-tile of the tensor cores' work. -/
def DT_KEYS : Nat := 32
/-- Words a staged dimension row of values takes in shared memory: its 16 key
    pairs (f16) and 8, so a half warp's 64-bit fragment loads meet in no bank;
    a row's columns xor twice its eighth, `row / 8`, and a row's words are
    paired (`8 s + 4 h + u` at `8 s + 2 u + h`) so a lane's two words of an
    `mma` fragment are adjacent. -/
def DT_VSTR : Nat := 24
/-- Words a lane reads of a sub-tile: one of each key's values, and its eight
    fragment words of each of four key groups. -/
def DT_WORDS : Nat := DT_KEYS + 2 * DT_KEYS

/-- **One tile of the sequence, one key head, on the tensor cores.**

    A block is one warp, block `cta` taking key head `cta mod NKV_K` and tile
    `cta / NKV_K`. Lane `l` holds row `l / 4` of the `mma`s, query `l / 4` of
    the eight its key head serves (the fragments' other eight rows repeat
    them, and their results are not read). `DT_KEYS` keys at a time: the
    scores are the bf16 `mma`'s of the queries (bf16, as narrowed) against the
    keys, read from the cache straight into the fragments; the softmax is
    online and in base 2 (scores scaled by `log₂ e / 8`), a row's four lanes
    combining by butterfly; the values are staged in shared memory as f16 and
    the probabilities, rounded to f16, multiply them by the f16 `mma` into
    f16 sums, scaled each sub-tile by their row's factor.

    Writes `(m, l, acc[HD_K])` per `(tile, head, query)` for the combine pass,
    `m` back in natural units; the sink belongs to the head and is applied
    there, once. -/
def attnTileBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64, .i64])
  let cta := ps.head; let tid := ps.snd; let sm := (ps.drop 3).head
  let kp := (ps.drop 4).head; let vp := (ps.drop 5).head; let qp := (ps.drop 6).head
  let pp := (ps.drop 7).head; let mp := (ps.drop 8).head
  let lane ← band tid (← c32 31)
  let g ← ushrImm lane 2
  let t ← band lane (← c32 3)
  let tw ← ishlImm t 1
  let seqLen ← load32 (← at_ mp (4 * GptOssAttention.M_SEQ))
  let tileSz ← load32 (← at_ mp (4 * GptOssAttention.M_TILESZ))
  let kvStride ← load32 (← at_ mp (4 * GptOssAttention.M_KVSTRIDE))
  let tile ← udiv cta (← c32 NKV_K)
  let kh ← isub cta (← imul tile (← c32 NKV_K))
  let s0 ← imul tile tileSz
  let sEnd ← umin (← iadd s0 tileSz) seqLen
  let lastKey ← isub sEnd (← c32 1)
  let word (base : V .i64) (ix : V .i32) : Prog V L (V .i32) := do
    load32 (← iadd base (← ishlImm (← ext ix) 2))
  -- query `g`'s words `8 s + t` and `8 s + 4 + t`, a fragment's two halves
  let qRow ← iadd qp (← ishlImm (← ext (← imul (← iadd (← imul kh (← c32 GQA_K)) g) (← c32 (HD_K / 2)))) 2)
  let qf ← (List.range (HD_K / 16)).mapM fun st => do
    pure (← word qRow (← iadd (← c32 (8 * st)) t), ← word qRow (← iadd (← c32 (8 * st + 4)) t))
  let kB ← iadd kp (← ishlImm (← ext (← imul kh kvStride)) 2)
  let vB ← iadd vp (← ishlImm (← ext (← imul kh kvStride)) 2)
  let zero ← fconst .f32 0
  let one ← fconst .f32 0x3F800000
  let ninf ← fconst .f32 0xFF800000
  let z64 ← iconst64 0
  let z32 ← c32 0
  let cScale ← fconst .f32 (1.4426950408889634 / 8 : Float).toFloat32.toBits.toUInt64
  let ex2 (x : V .f32) : Prog V L (V .f32) := do callLocal ex2Approx %[x]
  let lo32 (x : V .i64) : Prog V L (V .f32) := do bitcast .f32 (← ireduce32 x)
  let hi32 (x : V .i64) : Prog V L (V .f32) := do bitcast .f32 (← ireduce32 (← ushrImm x 32))
  let packH (x y : V .f32) : Prog V L (V .i32) := do
    callLocal cvtHR %[← bitcast .i32 x, ← bitcast .i32 y, z32]
  let bfToH (w : V .i32) : Prog V L (V .i32) := do
    callLocal cvtHR %[← ishlImm w 16, ← band w (← c32 0xFFFF0000), z32]
  let quad (x : V .f32) (f : V .f32 → V .f32 → Prog V L (V .f32)) : Prog V L (V .f32) := do
    let x1 ← f x (← bitcast .f32 (← callLocal shflBfly %[← bitcast .i32 x, ← c32 1]))
    f x1 (← bitcast .f32 (← callLocal shflBfly %[← bitcast .i32 x1, ← c32 2]))
  let pairOf (w : V .i32) : Prog V L (V .i32) := do
    bor (← band w (← c32 0xFFFFFFF8)) (← bor (← ishlImm (← band w (← c32 3)) 1) (← band (← ushrImm w 2) (← c32 1)))
  -- a sub-tile's words this lane reads: its value words (dimensions `2 l`,
  -- `2 l + 1` of each key), then its keys' fragment words; a key past the
  -- tile read as its last and weighted zero
  let fetch (sb : V .i32) : Prog V L (List (V .i32)) := do
    let keyAt (i : V .i32) : Prog V L (V .i32) := do umin (← iadd sb i) lastKey
    let mut ws : List (V .i32) := []
    for mm in List.range DT_KEYS do
      ws := ws ++ [← word vB (← iadd (← imul (← keyAt (← c32 mm)) (← c32 (HD_K / 2))) lane)]
    for kg in List.range (DT_KEYS / 8) do
      let kRow ← imul (← keyAt (← iadd (← c32 (8 * kg)) g)) (← c32 (HD_K / 2))
      for stp in List.range (HD_K / 16) do
        ws := ws ++ [← word kB (← iadd kRow (← iadd (← c32 (8 * stp)) t)),
          ← word kB (← iadd kRow (← iadd (← c32 (8 * stp + 4)) t))]
    pure ws
  let first ← fetch s0
  -- the running maximum and sum of row `g`, eight f16 sums of values, and the
  -- sub-tile's words, each trip fetching the next one's before its own work
  let st ← wloop (Vals.append (Vals.append %[s0, ninf, zero] (Vals.ofFn (n := 8) fun _ => z64))
      (Vals.ofFn (n := DT_WORDS) fun q => first.getD q.val z32))
    (fun cs => do pure (contIfULt cs.head sEnd, cs, ()))
    (fun cs _ => do
      let sb := cs.head
      let m := (cs.drop 1).head; let l := (cs.drop 2).head
      let os : List (V .i64) := [cs.get 3, cs.get 4, cs.get 5, cs.get 6, cs.get 7, cs.get 8, cs.get 9, cs.get 10]
      let cur := Vals.uniformToList (cs.drop 11 : Vals V (List.replicate DT_WORDS .i32))
      let next ← fetch (← iadd sb (← c32 DT_KEYS))
      -- the values, dimension by dimension: lane `dp`'s word of dimensions
      -- `2 dp`, `2 dp + 1` of every key pair, transposed
      for mm in List.range (DT_KEYS / 2) do
        let w0 ← bfToH (cur.getD (2 * mm) z32)
        let w1 ← bfToH (cur.getD (2 * mm + 1) z32)
        let lo ← bor (← band w0 (← c32 0xFFFF)) (← ishlImm w1 16)
        let hi ← bor (← ushrImm w0 16) (← band w1 (← c32 0xFFFF0000))
        let mx ← bxor (← pairOf (← c32 mm)) (← ishlImm (← band (← ushrImm lane 2) (← c32 7)) 1)
        let at0 ← iadd (← imul (← ishlImm lane 1) (← c32 DT_VSTR)) mx
        storeI32 lo (← at4 sm at0)
        storeI32 hi (← at4 sm (← iadd at0 (← c32 DT_VSTR)))
      callLocalVoid barrier %[]
      -- the scores: keys `8 kg + 2 t`, `+ 1` of row `g`
      let mut sv : List (V .f32) := []
      for kg in List.range (DT_KEYS / 8) do
        let mut acc := z64
        for stp in List.range (HD_K / 16) do
          let (q0, q1) := qf.getD stp (z32, z32)
          let b0 := cur.getD (DT_KEYS + 8 * kg + 2 * stp) z32
          let b1 := cur.getD (DT_KEYS + 8 * kg + 2 * stp + 1) z32
          acc ← callLocal mmaBfLoR %[q0, q0, q1, q1, b0, b1, acc]
        sv := sv ++ [← lo32 acc, ← hi32 acc]
      -- scaled to base 2, a key past the tile `-inf`
      let sv' ← (List.range sv.length).zip sv |>.mapM fun (i, x) => do
        let k ← iadd sb (← iadd (← c32 (8 * (i / 2) + i % 2)) tw)
        select (← icmp .ult k sEnd) (← fmul x cScale) ninf
      let mut tmax := ninf
      for x in sv' do tmax ← fmax tmax x
      let mNew ← fmax m (← quad tmax fmax)
      let dead ← fcmp .eq mNew ninf
      let corr ← select dead one (← ex2 (← fsub m mNew))
      let mUse ← select dead zero mNew
      let pr ← sv'.mapM fun x => do ex2 (← fsub x mUse)
      let mut tsum := zero
      for x in pr do tsum ← fadd tsum x
      let lNew ← fma l corr (← quad tsum fadd)
      -- the sums so far scaled by the row's factor, then the tile's values
      let cc ← packH corr corr
      let mut os' : List (V .i64) := []
      for o in os do
        let lo : V .i32 ← callLocal hfma2R %[← ireduce32 o, cc, z32]
        os' := os' ++ [← uextend64 lo]
      for c in List.range (DT_KEYS / 16) do
        let a0 ← packH (pr.getD (4 * c) zero) (pr.getD (4 * c + 1) zero)
        let a2 ← packH (pr.getD (4 * c + 2) zero) (pr.getD (4 * c + 3) zero)
        let mut nx : List (V .i64) := []
        for ng in List.range (HD_K / 8) do
          let vRow ← imul (← iadd (← c32 (8 * ng)) g) (← c32 DT_VSTR)
          let bb ← load64 (← at4 sm (← iadd vRow (← bxor (← iadd (← c32 (8 * c)) tw) (← c32 (2 * ng)))))
          let o : V .i64 ← callLocal mmaHR %[a0, z32, a2, z32, ← ireduce32 bb, ← ireduce32 (← ushrImm bb 32),
            os'.getD ng z64]
          nx := nx ++ [o]
        os' := nx
      callLocalVoid barrier %[]
      pure (Vals.append (Vals.append %[← iadd sb (← c32 DT_KEYS), mNew, lNew]
        (Vals.ofFn (n := 8) fun q => os'.getD q.val z64)) (Vals.ofFn (n := DT_WORDS) fun q => next.getD q.val z32)))
  -- an f16 (a word's low half) as the f32 of the same value
  let f32OfH (h : V .i32) : Prog V L (V .f32) := do
    let sign ← ishlImm (← band h (← c32 0x8000)) 16
    let e ← band (← ushrImm h 10) (← c32 31)
    let mt ← band h (← c32 1023)
    let normal ← bor (← ishlImm (← iadd e (← c32 112)) 23) (← ishlImm mt 13)
    let special ← bor (← c32 0x7F800000) (← ishlImm mt 13)
    let sub ← bitcast .i32 (← fmul (← fcvtFromSint .f32 mt) (← fconst .f32 0x33800000))
    let mag ← select (← icmp .eq e (← c32 0)) sub (← select (← icmp .eq e (← c32 31)) special normal)
    bitcast .f32 (← bor sign mag)
  -- this row's partial: one block of `attPartialFloats` a query
  let os : List (V .i64) := [st.get 3, st.get 4, st.get 5, st.get 6, st.get 7, st.get 8, st.get 9, st.get 10]
  let ix ← iadd (← imul (← iadd (← imul tile (← c32 NKV_K)) kh) (← c32 GQA_K)) g
  let addr ← iadd pp (← imul (← ext ix) (← c64n (attPartialFloats * 4)))
  when .eq t (← c32 0) do
    store (← fmul (st.drop 1).head (← fconst .f32 0x3F317218)) addr
    store (st.drop 2).head (← at_ addr 4)
  for ng in List.range (HD_K / 8) do
    let w ← ireduce32 (os.getD ng z64)
    let aAddr ← iadd addr (← ishlImm (← ext (← iadd (← c32 (8 * ng)) tw)) 2)
    store (← f32OfH (← band w (← c32 0xFFFF))) (← at_ aAddr 8)
    store (← f32OfH (← ushrImm w 16)) (← at_ aAddr 12)

/-- **The tiles, merged, and the sink applied once.**

    One CTA of one warp per query head. Partial results `(m, l, acc)` merge
    the way the tile loop folds a single key: each rescaled by
    `exp(m - max)` and added. Lane `L` takes the maxima and sums of tiles `L`
    and `L + 32`, combined by butterfly; the values of every tile place are
    loaded at once, a place past the last tile weighted zero. The learned
    sink joins the denominator here and nowhere else. The tile count comes
    from the meta buffer, which the layer uploads anyway. -/
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
  let ninf ← fconst .f32 0xFF800000
  let zero ← fconst .f32 0
  let last ← isub nTiles (← c32 1)
  -- tile `i`'s partial (past the last, the last's)
  let at_tile (i : V .i32) : Prog V L (V .i64) := do
    iadd hBase (← imul (← ext (← umin i last)) (← c64n tStride))
  let bfly (x : V .f32) (f : V .f32 → V .f32 → Prog V L (V .f32)) : Prog V L (V .f32) := do
    let mut v := x
    for m in [16, 8, 4, 2, 1] do
      v ← f v (← bitcast .f32 (← callLocal shflBfly %[← bitcast .i32 v, ← c32 m]))
    pure v
  -- lane `L` takes tiles `L` and `L + 32` (a layer has at most `COMB_TILES`)
  let tA := lane
  let tB ← iadd lane (← c32 32)
  let okA ← icmp .ult tA nTiles
  let okB ← icmp .ult tB nTiles
  let pA ← at_tile tA
  let pB ← at_tile tB
  let mA ← select okA (← loadF32 pA) ninf
  let mB ← select okB (← loadF32 pB) ninf
  -- the largest maximum any tile saw; the sink shares the denominator, so it
  -- belongs in the maximum too
  let sink ← loadF32 (← iadd sp (← ishlImm (← ext head) 2))
  let mAll ← fmax (← bfly (← fmax mA mB) fmax) sink
  -- every tile rescaled onto that maximum and added
  let wA ← select okA (← expA (← fsub mA mAll)) zero
  let wB ← select okB (← expA (← fsub mB mAll)) zero
  let lSum ← bfly (← fma (← loadF32 (← at_ pB 4)) wB (← fmul (← loadF32 (← at_ pA 4)) wA)) fadd
  let mut a0 := zero
  let mut a1 := zero
  for i in List.range COMB_TILES do
    let w ← bitcast .f32 (← callLocal shflIdx %[← bitcast .i32 (if i < 32 then wA else wB), ← c32 (i % 32)])
    let p ← iadd (← at_tile (← c32 i)) (← ishlImm (← ext lane) 3)
    a0 ← fma (← loadF32 (← at_ p 8)) w a0
    a1 ← fma (← loadF32 (← at_ p 12)) w a1
  let r2 := (lSum, a0, a1)
  -- the sink's own term, and then the division softmax would have done
  let lAll ← fadd r2.1 (← expA (← fsub sink mAll))
  let inv ← callLocal rcpApprox %[lAll]
  let oAddr ← iadd op (← ishlImm (← ext (← iadd (← imul head (← c32 HD_K)) (← ishlImm lane 1))) 2)
  store (← fmul r2.2.1 inv) oAddr
  store (← fmul r2.2.2 inv) (← at_ oAddr 4)

end

/-- Rows a warp of the Q8_0 GEMV computes: the activation's words are loaded
    once for all of them. -/
def rowsPerWarpQ8 : Nat := 2

/-- **A Q8_0 matrix-vector product**: `out[r] = Σ_k W[r, k] · x[k]`, `W`'s
    int8 values row by row and its f32 scales one a block of 32 (llama.cpp's
    Q8_0, repacked aligned by `prepare_q8.py`), `x` and `out` f32, for a width
    `cols` fixed when the kernel is made. A warp takes `rowsPerWarpQ8` rows;
    lane `l` takes the words `l, l + 32, …` of each (four values a word, so a
    warp step reads 128 bytes a row, side by side), the trips unrolled, then
    a butterfly sums the lanes. `w` holds the `rows × cols` values and then,
    straight after them, the scales, as `prepare_q8.py` lays a tensor out. -/
def q8GemvBody (rows cols : Nat) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64])
  let cta := ps.head; let tid := ps.snd
  let w := (ps.drop 4).head; let x := (ps.drop 5).head; let out := (ps.drop 6).head
  let sc ← iaddImm w (rows * cols)
  let rows ← c32 rows
  let lane ← band tid (← c32 31)
  let warp ← iadd (← imul cta (← c32 warpsPerCta)) (← ushrImm tid 5)
  let r0 ← imul warp (← c32 rowsPerWarpQ8)
  let words := cols / 4
  let zero ← f32c 0
  -- each row's start, clamped to a real row (a row past the end is computed and not written)
  let rowsAt ← (List.range rowsPerWarpQ8).mapM fun k => do
    let r ← iadd r0 (← c32 k)
    let rc ← select (← icmp .ult r rows) r (← c32 0)
    pure (r, ← iadd w (← ext (← imul rc (← c32 cols))), ← iadd sc (← ext (← ishlImm (← imul rc (← c32 (cols / 32))) 2)))
  let mut accs : List (V .f32) := List.replicate rowsPerWarpQ8 zero
  for t in List.range ((words + 31) / 32) do
    let j ← iadd lane (← c32 (32 * t))
    let live ← icmp .ult j (← c32 words)
    let jc ← select live j (← c32 0)
    let xa ← iadd x (← ext (← ishlImm jc 4))
    let xs ← (List.range 4).mapM fun i => do loadF32 (← at_ xa (4 * i))
    let mut next : List (V .f32) := []
    for ((_, wr, sr), acc) in rowsAt.zip accs do
      let q ← load32 (← iadd wr (← ext (← ishlImm jc 2)))
      let s ← loadF32 (← iadd sr (← ext (← ishlImm (← ushrImm jc 3) 2)))
      let mut dot ← f32c 0
      for (i, xi) in (List.range 4).zip xs do
        let b ← sshr (← ishlImm q (24 - 8 * i)) (← c32 24)
        dot ← fma (← fcvtFromSint .f32 b) xi dot
      next := next ++ [← fadd acc (← select live (← fmul s dot) zero)]
    accs := next
  for ((r, _, _), acc) in rowsAt.zip accs do
    let mut v := acc
    for m in [16, 8, 4, 2, 1] do v ← bflyAdd v m
    when .eq lane (← c32 0) do
      when .ult r rows do store v (← iadd out (← ext (← ishlImm r 2)))

/-- **The down projection, plus its bias**, as a kernel. -/
def downGemv : Kernel :=
  { name := "mxfp4_down_gemv_bias", params := [.i64, .i64, .i64, .i64, .i64],
    code := emit downGemvBias (idParams ++ [.i64, .i64, .i64, .i64, .i64]), smem := smemBytes }

/-- **The down projection of all `TOPK` experts**, one launch; the layer is
    meta word `layerWord`. -/
def downAllK (layerWord : Nat) : Kernel :=
  { name := "mxfp4_down_gemv_bias_all", params := List.replicate 6 .i64,
    code := emit (downAll layerWord) (idParams ++ List.replicate 6 .i64), smem := smemBytes }

/-- **Gate and up of all `TOPK` experts**, one launch; the layer is meta word
    `layerWord`. -/
def gateUpAllK (layerWord : Nat) : Kernel :=
  { name := "mxfp4_gate_up_swiglu_gemv_all", params := List.replicate 6 .i64,
    code := emit (gateUpAll layerWord) (idParams ++ List.replicate 6 .i64), smem := smemBytes }

/-- **The gated sum of the chosen experts**, their outputs side by side:
    entry params `(y, gates, out)`, expert `j`'s at `y + 4·H·j`. -/
def moeCombineAllBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64])
  let y := (ps.drop 4).head; let gates := (ps.drop 5).head; let out := (ps.drop 6).head
  let i ← gidx ps.head ps.snd
  when .ult i (← c32 H) do
    let off ← ext (← ishlImm i 2)
    let mut acc ← fconst .f32 0
    for j in List.range TOPK do
      acc ← fma (← loadF32 (← at_ (← iadd y off) (4 * H * j))) (← loadF32 (← at_ gates (4 * j))) acc
    store acc (← iadd out off)

def moeCombineAll : Kernel :=
  { name := "mxfp4_moe_combine_all", params := [.i64, .i64, .i64],
    code := emit moeCombineAllBody (idParams ++ [.i64, .i64, .i64]) }

/-- **Gate and up with the clamped SwiGLU**, as a kernel. -/
def gateUp : Kernel :=
  { name := "mxfp4_gate_up_swiglu_gemv", params := [.i64, .i64, .i64, .i64, .i64],
    code := emit gateUpSwiglu (idParams ++ [.i64, .i64, .i64, .i64, .i64]), smem := smemBytes }

def moeCombine4 : Kernel :=
  { name := "mxfp4_moe_combine4", params := [.i64, .i64, .i64, .i64, .i64, .i64],
    code := emit moeCombine (idParams ++ [.i64, .i64, .i64, .i64, .i64, .i64]) }

/-- **The Q8_0 matrix-vector product** of a `rows × cols` matrix, as a kernel:
    `rows / (warpsPerCta · rowsPerWarpQ8)` blocks of `32 · warpsPerCta`. -/
def q8Gemv (rows cols : Nat) : Kernel :=
  { name := s!"q8_0_gemv_{rows}x{cols}", params := [.i64, .i64, .i64],
    code := emit (q8GemvBody rows cols) (idParams ++ [.i64, .i64, .i64]) }

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

/-- One tile of the sequence and one key head, as a kernel: a block of one
    warp, `DT_VSTR · HD_K` words of shared memory for the staged values. -/
def attnTile : Kernel :=
  { name := "gptoss_attn_tile", params := [.i64, .i64, .i64, .i64, .i64],
    code := emit attnTileBody (idParams ++ [.i64, .i64, .i64, .i64, .i64]), warp := true,
    smem := DT_VSTR * HD_K * 4 }

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
