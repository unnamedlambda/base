import AlgorithmLib.PTX
import AlgorithmLib.ML.QuantMX

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
  kernel that unpacks nibbles is therefore outside it, and these are written in
  the string builder, which has no semantics attached.

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

namespace GptOssKernels

open AlgorithmLib.PTX

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

    A sixteen-entry table is the obvious way to replace `decodeFp4`'s thirteen
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

/-- The padded shared index of element `e`: `e + e/32`. -/
def padIdx (e : Nat) : Nat := e + e / G

/-! ## The shared decoding fragment -/

/-- Decode one four-bit code into a `Float32`, in registers.

    `nib` holds the code in its low four bits; anything above them is ignored.
    Six integer instructions and one reinterpretation, no memory reference and
    no table. -/
def decodeFp4 (out : Reg .f32) (nib : Reg .u32) : PTX Unit := do
  let m ← freshR; andR m nib 1                 -- mantissa bit
  let e ← freshR; shrR e nib 1; andR e e 3     -- exponent field, two bits
  let s ← freshR; shrR s nib 3; andR s s 1     -- sign bit
  -- Normal case: exponent field e-1+127 = e+126, mantissa in bit 22.
  let ef ← freshR; addRI ef e 126; shlR ef ef 23
  let mf ← freshR; shlR mf m 22
  let normal ← freshR; orRR normal ef mf
  -- Subnormal case (e = 0): 0.0 or 0.5, chosen by the mantissa bit.
  let sub ← freshR; mulLoRI sub m 0x3F000000   -- m = 0 -> 0, m = 1 -> 0.5
  let isSub ← freshP; setpEqI isSub e 0
  let mag ← freshR; selpR mag sub normal isSub
  -- Sign, applied to the assembled magnitude.
  let sgn ← freshR; shlR sgn s 31
  let bits ← freshR; orRR bits mag sgn
  bitsToF out bits

/-- Decode an E8M0 scale byte into `2^(s-127)`.

    A `Float32` with exponent field `s` and zero mantissa *is* that power of
    two, so the byte only has to be shifted into place. `s = 0` yields zero
    (the exponent field is reserved for subnormals) and `s = 255` yields a NaN;
    the converter rejects the latter, which is why nothing here tests for it. -/
def decodeE8M0 (out : Reg .f32) (sb : Reg .u32) : PTX Unit := do
  let bits ← freshR; shlR bits sb 23
  bitsToF out bits

/-- Fill the per-lane replicated table, once per block, and fence.

    512 entries written by the block's threads striding together; each thread
    picks its entry's value with a chain of comparisons that runs once and
    never again. -/
def initFp4Table (tid : Reg .u32) (base : Reg .u32) : PTX Unit := do
  let i ← freshR; movR i tid
  let loop := "FP4TAB_LOOP"
  let done := "FP4TAB_DONE"
  label loop
  do
    let p ← freshP; setpGeI p i fp4TableFloats; braIf p done
    let code ← freshR; shrR code i 5
    let off ← freshR; shlR off i 2
    let addr ← freshR; addR addr base off
    for (bits, k) in fp4Table.zipIdx do
      let isMine ← freshP; setpEqI isMine code k
      let lbl := s!"FP4TAB_E{k}"
      braIfNot isMine lbl
      let v ← freshF; movFI v (.bits bits)
      stSharedFD addr v
      label lbl
    addRI i i (32 * warpsPerCta)
    bra loop
  label done
  barSync

/-- Look a code up in this lane's own copy of the table.

    `laneOff` is `lane * 4`, computed once per thread; the code selects the
    copy at a stride of 128 bytes, which is what keeps every lane on its own
    bank. -/
def lookupFp4 (out : Reg .f32) (nib : Reg .u32) (tabBase laneOff : Reg .u32) : PTX Unit := do
  let codeOff ← freshR; shlR codeOff nib 7
  let addr ← freshR; addR addr tabBase laneOff; addR addr addr codeOff
  ldSharedFD out addr

/-- Copy the activation into shared memory, padded, once per block.

    Every warp contracts a different weight row against the *same* activation,
    so without this each of the eight warps reads all of `x` from the cache and
    the activation crosses it far more often than the weights do — which is
    what the measurements said the kernel was actually spending its time on,
    once the weight loads were widened and occupancy ruled out.

    Cooperative: the block's threads stride through the vector together, and
    each writes element `e` at padded index `e + e/32`. The barrier is
    block-wide because every warp reads what every other warp wrote. -/
def stageActivation (tid : Reg .u32) (src : Reg .u64) (dst : Reg .u32)
    (n : Nat) (tag : String) : PTX Unit := do
  let i ← freshR; movR i tid
  let loop := "STAGE_" ++ tag
  let done := "STAGEDONE_" ++ tag
  label loop
  do
    let p ← freshP; setpGeI p i n; braIf p done
    let gOff ← freshRd; mulWideRI gOff i 4
    let gAddr ← freshRd; addRd gAddr src gOff
    let v ← freshF; ldGlobalF v gAddr
    let pad ← freshR; shrR pad i 5
    let idx ← freshR; addR idx i pad
    let sOff ← freshR; shlR sOff idx 2
    let sAddr ← freshR; addR sAddr dst sOff
    stSharedFD sAddr v
    addRI i i (32 * warpsPerCta)
    bra loop
  label done
  barSync

/-- One lane's contribution to a dot product over `nBlocks` microscaling blocks.

    **Four packed bytes to a lane, not one.** The obvious mapping — lane `L`
    takes element `blk*32 + L`, one block to a warp — issues thirty-two
    one-byte loads to fetch the sixteen distinct bytes of a block, and the
    kernel then runs at a sixth of what the memory system can do: it is bound
    by load instructions, not by bytes. Measured, before and after.

    So a lane loads a `u32` instead. Thirty-two lanes then cover 128 contiguous
    bytes — one transaction — which is 256 codes, so a warp crosses eight
    microscaling blocks per iteration and a lane owns eight consecutive
    elements of them. Because eight divides thirty-two, a lane's eight elements
    always lie in one block, so it still reads exactly one scale.

    The nibble order falls out for free: within a little-endian `u32`, code `j`
    is at bit `4j`, so the eight codes are successive nibbles of one register.

    `nBlocks` need not be a multiple of eight; the blocks past the last whole
    chunk are taken one at a time by the tail, which is the chunk-and-remainder
    shape used elsewhere in this development. At this model's widths there are
    ninety blocks: eleven chunks and two left over.

    `acc` is accumulated across iterations; the caller reduces across lanes.
    `tag` distinguishes this expansion's labels from another's — PTX labels are
    per-module, and the gate/up kernel expands this twice. -/
def dotRowMx (acc : Reg .f32) (wBase sBase : Reg .u64)
    (xSmem tabBase laneOff lane : Reg .u32) (nBlocks : Nat) (tag : String) : PTX Unit := do
  let nChunks := nBlocks / 8
  let tailStart := nChunks * 8
  -- ── whole chunks: eight blocks a warp, four bytes a lane ──
  if nChunks > 0 then do
    let wLaneOff ← freshR; mulLoRI wLaneOff lane 4          -- byte within the chunk
    let sLane ← freshR; shrR sLane lane 2                   -- which of the 8 blocks
    -- this lane's first element within a chunk, in PADDED shared floats:
    -- 8L for the element, L/4 for the pads crossed on the way there.
    let xLaneOff ← freshR; mulLoRI xLaneOff lane 8; addR xLaneOff xLaneOff sLane
    let chunk ← freshR; movRC chunk 0
    let cLoop := "MXDOT_CHUNK_" ++ tag
    let cDone := "MXDOT_CHUNKDONE_" ++ tag
    label cLoop
    do
      let p ← freshP; setpGeI p chunk nChunks; braIf p cDone
      -- 128 packed bytes a chunk; this lane's four of them
      let wOff ← freshR; mulLoRI wOff chunk 128; addR wOff wOff wLaneOff
      let wOff64 ← freshRd; cvtU64 wOff64 wOff
      let wAddr ← freshRd; addRd wAddr wBase wOff64
      let packed ← freshR; ldGlobalU packed wAddr
      -- the one scale this lane's eight elements share
      let sOff ← freshR; mulLoRI sOff chunk 8; addR sOff sOff sLane
      let sOff64 ← freshRd; cvtU64 sOff64 sOff
      let sAddr ← freshRd; addRd sAddr sBase sOff64
      let sByte ← freshRd; ldGlobalU8 sByte sAddr
      let sByte32 ← freshR; cvtU32of64 sByte32 sByte
      let sc ← freshF; decodeE8M0 sc sByte32
      -- this lane's eight activations, from the staged copy.  A chunk spans
      -- 256 elements and therefore 264 padded floats; the eight are contiguous
      -- because eight divides thirty-two, so no pad falls between them.
      let xIdx ← freshR; mulLoRI xIdx chunk 264; addR xIdx xIdx xLaneOff
      let xOff ← freshR; shlR xOff xIdx 2
      let xAddr ← freshR; addR xAddr xSmem xOff
      for j in [0, 1, 2, 3, 4, 5, 6, 7] do
        let nib ← freshR; shrR nib packed (4 * j); andR nib nib 0xF
        let w ← freshF; lookupFp4 w nib tabBase laneOff
        mulF w w sc
        let xv ← freshF; ldSharedF xv xAddr (4 * j)
        fmaRn acc w xv acc
      addRI chunk chunk 1
      bra cLoop
    label cDone
  -- ── the tail: whole blocks that did not fill a chunk, one at a time ──
  if tailStart < nBlocks then do
    let halfLane ← freshR; shrR halfLane lane 1
    let oddLane ← freshR; andR oddLane lane 1
    let isOdd ← freshP; setpEqI isOdd oddLane 1
    let blk ← freshR; movRC blk tailStart
    let loop := "MXDOT_TAIL_" ++ tag
    let done := "MXDOT_TAILDONE_" ++ tag
    label loop
    do
      let p ← freshP; setpGeI p blk nBlocks; braIf p done
      let bOff ← freshR; mulLoRI bOff blk 16; addR bOff bOff halfLane
      let bOff64 ← freshRd; cvtU64 bOff64 bOff
      let wAddr ← freshRd; addRd wAddr wBase bOff64
      let byte ← freshRd; ldGlobalU8 byte wAddr
      let byte32 ← freshR; cvtU32of64 byte32 byte
      let hi ← freshR; shrR hi byte32 4
      let lo ← freshR; andR lo byte32 0xF
      let nib ← freshR; selpR nib hi lo isOdd
      let w ← freshF; lookupFp4 w nib tabBase laneOff
      let sOff64 ← freshRd; cvtU64 sOff64 blk
      let sAddr ← freshRd; addRd sAddr sBase sOff64
      let sByte ← freshRd; ldGlobalU8 sByte sAddr
      let sByte32 ← freshR; cvtU32of64 sByte32 sByte
      let sc ← freshF; decodeE8M0 sc sByte32
      mulF w w sc
      -- padded index: blk*32 + lane, plus the blk pads crossed = blk*33 + lane
      let xIdx ← freshR; mulLoRI xIdx blk 33; addR xIdx xIdx lane
      let xOff ← freshR; shlR xOff xIdx 2
      let xAddr ← freshR; addR xAddr xSmem xOff
      let xv ← freshF; ldSharedFD xv xAddr
      fmaRn acc w xv acc
      addRI blk blk 1
      bra loop
    label done

/-- Sum a value across the thirty-two lanes of a warp, leaving the total in
    every lane. The butterfly order is the one the rest of this development
    commits to, so a host-side reference can reproduce it exactly. -/
def warpSum (acc : Reg .f32) : PTX Unit := do
  for mask in [16, 8, 4, 2, 1] do
    let t ← freshF
    shflBfly t acc mask
    addF acc acc t

/-! ## The kernels -/

/-- **Gate and up, contracted and combined in one launch.**

    One CTA of one warp per output row `r < I`. The gate row is `r` and the up
    row is `I + r`, which is the layout the converter writes: the published
    checkpoint interleaves the two, and de-interleaving them offline costs
    nothing at run time and saves an address here.

    The epilogue is gpt-oss's clamped SwiGLU, fused rather than left to a
    second pass: the two contracted values are already in registers, and
    spilling them to memory to read them back would double this kernel's
    traffic for one multiply. -/
def gateUpSwigluGemv : KernelSpec where
  name := "mxfp4_gate_up_swiglu_gemv"
  params := ["p_blocks", "p_scales", "p_bias", "p_x", "p_out"]
  body := do
    let blocks ← freshRd; ldParam64 blocks "p_blocks"
    let scales ← freshRd; ldParam64 scales "p_scales"
    let bias ← freshRd; ldParam64 bias "p_bias"
    let x ← freshRd; ldParam64 x "p_x"
    let out ← freshRd; ldParam64 out "p_out"
    let (tid, wid, lane) ← getWarpIds
    let tabBase ← smemBase
    initFp4Table tid tabBase
    let laneOff ← freshR; shlR laneOff lane 2
    let xSmem ← freshR; addRI xSmem tabBase xSmemOff
    stageActivation tid x xSmem H "GU"
    -- %ctaid.x is a special register and cannot be an ALU operand; move first.
    let cta ← freshR; movR cta ctaX
    let row ← freshR; mulLoRI row cta warpsPerCta; addR row row wid
    -- gate: row `row`
    let gwOff ← freshRd; mulWideRI gwOff row rowBytesH
    let gwBase ← freshRd; addRd gwBase blocks gwOff
    let gsOff ← freshRd; mulWideRI gsOff row rowScalesH
    let gsBase ← freshRd; addRd gsBase scales gsOff
    let gAcc ← freshF; movFI gAcc (.bits 0)
    dotRowMx gAcc gwBase gsBase xSmem tabBase laneOff lane (H / G) "GATE"
    warpSum gAcc
    -- up: row `I + row`
    let upRow ← freshR; addRI upRow row I
    let uwOff ← freshRd; mulWideRI uwOff upRow rowBytesH
    let uwBase ← freshRd; addRd uwBase blocks uwOff
    let usOff ← freshRd; mulWideRI usOff upRow rowScalesH
    let usBase ← freshRd; addRd usBase scales usOff
    let uAcc ← freshF; movFI uAcc (.bits 0)
    dotRowMx uAcc uwBase usBase xSmem tabBase laneOff lane (H / G) "UP"
    warpSum uAcc
    -- lane 0 adds the biases, applies the activation, and stores
    let isLane0 ← freshP; setpEqI isLane0 lane 0
    let skip := "GU_SKIP"
    braIfNot isLane0 skip
    let gbOff ← freshRd; mulWideRI gbOff row 4
    let gbAddr ← freshRd; addRd gbAddr bias gbOff
    let gb ← freshF; ldGlobalF gb gbAddr
    addF gAcc gAcc gb
    let ubOff ← freshRd; mulWideRI ubOff upRow 4
    let ubAddr ← freshRd; addRd ubAddr bias ubOff
    let ub ← freshF; ldGlobalF ub ubAddr
    addF uAcc uAcc ub
    -- clamped SwiGLU: gate clamped above, linear clamped both sides and offset
    minFI gAcc gAcc (.bits swigluLimitBits)
    minFI uAcc uAcc (.bits swigluLimitBits)
    maxFI uAcc uAcc (.bits negLimitBits)
    -- sigmoid(alpha*g) = 1 / (1 + exp2(-alpha*g*log2 e))
    let t ← freshF; mulFI t gAcc (.bits swigluAlphaBits)
    mulFI t t (.bits f32_log2e)
    let negT ← freshF; movFI negT (.bits 0); subF negT negT t
    let e ← freshF; ex2 e negT
    addFI e e (.bits oneBits)
    let sig ← freshF; rcp sig e
    mulF gAcc gAcc sig
    addFI uAcc uAcc (.bits oneBits)
    mulF gAcc gAcc uAcc
    let oOff ← freshRd; mulWideRI oOff row 4
    let oAddr ← freshRd; addRd oAddr out oOff
    stGlobalF oAddr gAcc
    label skip
    ptxRet

/-- **The down projection, plus its bias.**

    One CTA of one warp per output row `r < H`, contracting over `I`. Same
    shape as the gate/up kernel with a plainer epilogue. -/
def downGemvBias : KernelSpec where
  name := "mxfp4_down_gemv_bias"
  params := ["p_blocks", "p_scales", "p_bias", "p_h", "p_out"]
  body := do
    let blocks ← freshRd; ldParam64 blocks "p_blocks"
    let scales ← freshRd; ldParam64 scales "p_scales"
    let bias ← freshRd; ldParam64 bias "p_bias"
    let hv ← freshRd; ldParam64 hv "p_h"
    let out ← freshRd; ldParam64 out "p_out"
    let (tid, wid, lane) ← getWarpIds
    let tabBase ← smemBase
    initFp4Table tid tabBase
    let laneOff ← freshR; shlR laneOff lane 2
    let xSmem ← freshR; addRI xSmem tabBase xSmemOff
    stageActivation tid hv xSmem I "DN"
    let cta ← freshR; movR cta ctaX
    let row ← freshR; mulLoRI row cta warpsPerCta; addR row row wid
    let wOff ← freshRd; mulWideRI wOff row rowBytesI
    let wBase ← freshRd; addRd wBase blocks wOff
    let sOff ← freshRd; mulWideRI sOff row rowScalesI
    let sBase ← freshRd; addRd sBase scales sOff
    let acc ← freshF; movFI acc (.bits 0)
    dotRowMx acc wBase sBase xSmem tabBase laneOff lane (I / G) "DOWN"
    warpSum acc
    let isLane0 ← freshP; setpEqI isLane0 lane 0
    let skip := "DN_SKIP"
    braIfNot isLane0 skip
    let bOff ← freshRd; mulWideRI bOff row 4
    let bAddr ← freshRd; addRd bAddr bias bOff
    let bv ← freshF; ldGlobalF bv bAddr
    addF acc acc bv
    let oOff ← freshRd; mulWideRI oOff row 4
    let oAddr ← freshRd; addRd oAddr out oOff
    stGlobalF oAddr acc
    label skip
    ptxRet

/-- **Dequantise a packed row to `Float32`.**

    The prefill path's kernel, and the one a proof could reach today: it lands
    `mxDeq` of its input in a buffer, after which a proven GEMM reads ordinary
    floats. Decode does not use it — eight times the traffic on the path that
    is bound by traffic — but prefill is compute-bound here and can spend it.

    One CTA per row, one lane per element of a block, `nBlocks` blocks. -/
def dequantF32 : KernelSpec where
  name := "mxfp4_dequant_f32"
  params := ["p_blocks", "p_scales", "p_out", "p_meta"]
  body := do
    let blocks ← freshRd; ldParam64 blocks "p_blocks"
    let scales ← freshRd; ldParam64 scales "p_scales"
    let out ← freshRd; ldParam64 out "p_out"
    let metaP ← freshRd; ldParam64 metaP "p_meta"
    -- meta[0] = blocks per row, so one kernel serves both widths
    let nBlocks ← freshR; ldGlobalU nBlocks metaP
    let row ← freshR; movR row ctaX
    let lane ← freshR; movR lane tidX
    let halfLane ← freshR; shrR halfLane lane 1
    let oddLane ← freshR; andR oddLane lane 1
    let isOdd ← freshP; setpEqI isOdd oddLane 1
    -- row bases, in bytes: nBlocks*16 packed, nBlocks scales, nBlocks*32 floats
    let rowB ← freshR; mulLoRI rowB nBlocks 16
    let wOffR ← freshR; mulLoR wOffR row rowB
    let wOff ← freshRd; cvtU64 wOff wOffR
    let wBase ← freshRd; addRd wBase blocks wOff
    let sOffR ← freshR; mulLoR sOffR row nBlocks
    let sOff ← freshRd; cvtU64 sOff sOffR
    let sBase ← freshRd; addRd sBase scales sOff
    let oElemR ← freshR; mulLoRI oElemR nBlocks 32
    let oIdxR ← freshR; mulLoR oIdxR row oElemR
    let oOff ← freshRd; mulWideRI oOff oIdxR 4
    let oBase ← freshRd; addRd oBase out oOff
    let blk ← freshR; movRC blk 0
    let loop := "DQ_LOOP"
    let done := "DQ_DONE"
    label loop
    do
      let p ← freshP; setpGe p blk nBlocks; braIf p done
      let bOff ← freshR; mulLoRI bOff blk 16; addR bOff bOff halfLane
      let bOff64 ← freshRd; cvtU64 bOff64 bOff
      let wAddr ← freshRd; addRd wAddr wBase bOff64
      let byte ← freshRd; ldGlobalU8 byte wAddr
      let byte32 ← freshR; cvtU32of64 byte32 byte
      let hi ← freshR; shrR hi byte32 4
      let lo ← freshR; andR lo byte32 0xF
      let nib ← freshR; selpR nib hi lo isOdd
      -- the arithmetic decode, not the table: this kernel is the prefill
      -- path, it is not what a token's latency rides on, and keeping it free of
      -- shared state is what makes it the one a proof can reach first.
      let w ← freshF; decodeFp4 w nib
      let sOff64 ← freshRd; cvtU64 sOff64 blk
      let sAddr ← freshRd; addRd sAddr sBase sOff64
      let sByte ← freshRd; ldGlobalU8 sByte sAddr
      let sByte32 ← freshR; cvtU32of64 sByte32 sByte
      let sc ← freshF; decodeE8M0 sc sByte32
      mulF w w sc
      let eIdx ← freshR; mulLoRI eIdx blk 32; addR eIdx eIdx lane
      let eOff ← freshRd; mulWideRI eOff eIdx 4
      let eAddr ← freshRd; addRd eAddr oBase eOff
      stGlobalF eAddr w
      addRI blk blk 1
      bra loop
    label done
    ptxRet

/-- **The gated sum of the chosen experts.**

    Four slot outputs, four gates, one row of the residual stream. The slots
    are summed here rather than accumulated in place by the down projection
    because a slot's output is worth reading once and the alternative — four
    read-modify-write passes over the same row — costs three extra traversals
    to save one launch.

    Threads map to elements; `H` is not a multiple of the block, so the tail is
    masked rather than padded. -/
def moeCombine4 : KernelSpec where
  name := "mxfp4_moe_combine4"
  params := ["p_y0", "p_y1", "p_y2", "p_y3", "p_gates", "p_out"]
  body := do
    let gates ← freshRd; ldParam64 gates "p_gates"
    let out ← freshRd; ldParam64 out "p_out"
    let tid ← freshR; movR tid tidX
    let cta ← freshR; movR cta ctaX
    let i ← freshR; mulLoRI i cta (32 * warpsPerCta); addR i i tid
    let inRange ← freshP; setpLtI inRange i H
    let skip := "COMBINE_SKIP"
    braIfNot inRange skip
    let off ← freshRd; mulWideRI off i 4
    let acc ← freshF; movFI acc (.bits 0)
    for (nm, j) in [("p_y0", 0), ("p_y1", 1), ("p_y2", 2), ("p_y3", 3)] do
      let yb ← freshRd; ldParam64 yb nm
      let ya ← freshRd; addRd ya yb off
      let yv ← freshF; ldGlobalF yv ya
      let ga ← freshRd; addRdI ga gates (4 * j)
      let gv ← freshF; ldGlobalF gv ga
      fmaRn acc yv gv acc
    let oa ← freshRd; addRd oa out off
    stGlobalF oa acc
    label skip
    ptxRet

/-- **`Float32` to bfloat16, rounding to nearest with ties to even.**

    bf16 *is* the top sixteen bits of a `Float32`, so narrowing is a shift —
    but a shift alone truncates, and truncation biases every weight and every
    activation toward zero. The bias is systematic rather than random, so it
    does not cancel across a 2880-term dot product; it accumulates. Adding
    `0x7FFF` plus the retained bit's own value before the shift rounds to
    nearest and breaks ties toward the even mantissa, which is what the
    converter does on the host side and what this must agree with. -/
def rneBf16 (d : Reg .u32) (s : Reg .f32) : PTX Unit := do
  let u ← freshR; fToBits u s
  let r ← freshR; shrR r u 16; andR r r 1
  addRI r r 0x7FFF
  addR u u r
  shrR d u 16

/-- **Narrow an activation row to bf16, a pair of elements per thread.**

    cuBLAS refuses a bf16 matrix against an f32 vector — the mixed pair is not
    among the combinations `cublasGemmEx` accepts — so an activation meeting a
    bf16 weight has to be narrowed first. It is cheap: the row is a few
    thousand elements against a weight matrix of millions.

    Two elements per thread, packed into one `u32` and stored once. The
    alternative is a `st.global.u16` per element, which is the same bytes in
    twice the transactions; the widths this runs at — `H`, `QKV`, `NQ·HD` —
    are all even, so the pairing needs no tail. -/
def narrowBf16 : KernelSpec where
  name := "narrow_bf16"
  params := ["p_in", "p_out", "p_meta"]
  body := do
    let inp ← freshRd; ldParam64 inp "p_in"
    let out ← freshRd; ldParam64 out "p_out"
    let metaP ← freshRd; ldParam64 metaP "p_meta"
    -- meta[0] = pairs to narrow, so one kernel serves every width
    let nPairs ← freshR; ldGlobalU nPairs metaP
    let tid ← freshR; movR tid tidX
    let cta ← freshR; movR cta ctaX
    let i ← freshR; mulLoRI i cta (32 * warpsPerCta); addR i i tid
    let inRange ← freshP; setpLt inRange i nPairs
    let skip := "NARROW_SKIP"
    braIfNot inRange skip
    let off ← freshRd; mulWideRI off i 8
    let a ← freshRd; addRd a inp off
    let x0 ← freshF; ldGlobalF x0 a
    let x1 ← freshF; ldGlobalFO x1 a 4
    let lo ← freshR; rneBf16 lo x0
    let hi ← freshR; rneBf16 hi x1
    shlR hi hi 16
    let packed ← freshR; orRR packed lo hi
    let oOff ← freshRd; mulWideRI oOff i 4
    let oa ← freshRd; addRd oa out oOff
    stGlobalU32 oa packed
    label skip
    ptxRet


/-- **bf16 to f32, the other direction.**

    The embedding table is bf16 and the first RMSNorm reads f32, so without
    this the caller has to widen a row and hand it in — the one gather the
    engine could not do for itself. One thread per packed pair, the same shape
    `narrow_bf16` has, and the widening is exact: bf16 is f32 with the low
    sixteen mantissa bits cleared, so shifting them back is lossless and no
    rounding decision arises. -/
def widenBf16 : KernelSpec where
  name := "widen_bf16"
  params := ["p_in", "p_out", "p_meta"]
  body := do
    let inp ← freshRd; ldParam64 inp "p_in"
    let out ← freshRd; ldParam64 out "p_out"
    let metaP ← freshRd; ldParam64 metaP "p_meta"
    let nPairs ← freshR; ldGlobalU nPairs metaP
    let tid ← freshR; movR tid tidX
    let cta ← freshR; movR cta ctaX
    let i ← freshR; mulLoRI i cta (32 * warpsPerCta); addR i i tid
    let inRange ← freshP; setpLt inRange i nPairs
    let skip := "WIDEN_SKIP"
    braIfNot inRange skip
    let off ← freshRd; mulWideRI off i 4
    let a ← freshRd; addRd a inp off
    let packed ← freshR; ldGlobalU packed a
    let lo ← freshR; shlR lo packed 16
    let hi ← freshR; andR hi packed 0xFFFF0000
    let f0 ← freshF; bitsToF f0 lo
    let f1 ← freshF; bitsToF f1 hi
    let oOff ← freshRd; mulWideRI oOff i 8
    let oa ← freshRd; addRd oa out oOff
    stGlobalF oa f0
    stGlobalFO oa 4 f1
    label skip
    ptxRet


/-- **The router's decision, taken on the device.**

    Four experts out of thirty-two, and the gates that weight them. Small
    enough that it is written serially in one lane: thirty-two logits, four
    passes, a hundred and twenty-eight comparisons — against a decode step that
    spends fourteen milliseconds in the experts themselves. What it costs is
    nothing; what it *saves* is the reason it exists.

    Without it the layer is two entry points with a host round trip between
    them, because which experts run is not known until the router has spoken.
    With it, a layer is one call, and a token is one call, and the loop over
    layers can live in the host program instead of above it.

    The chosen ids go to a small buffer the host program reads back — four
    integers, not a row — because binding a slot means rewriting a buffer id
    and buffer ids live on the host side. That is the whole remaining traffic.

    Shared memory holds a scratch copy of the row so that a chosen logit can be
    struck out with `-inf` and the next pass ignores it, which is what makes
    four independent maxima out of one array. -/
def routerTop4 : KernelSpec where
  name := "router_top4"
  params := ["p_logits", "p_gates", "p_chosen"]
  body := do
    let logits ← freshRd; ldParam64 logits "p_logits"
    let gates ← freshRd; ldParam64 gates "p_gates"
    let chosen ← freshRd; ldParam64 chosen "p_chosen"
    let tid ← freshR; movR tid tidX
    let sm ← smemBase
    -- stage the row: lane i holds logit i, and zero the gate slots beside it
    let inRow ← freshP; setpLtI inRow tid NE
    let skipStage := "RT_SKIP_STAGE"
    braIfNot inRow skipStage
    do
      let off ← freshRd; mulWideRI off tid 4
      let a ← freshRd; addRd a logits off
      let v ← freshF; ldGlobalF v a
      let sa ← freshR; shlR sa tid 2; addR sa sa sm
      stSharedFD sa v
      let ga ← freshRd; addRd ga gates off
      let z ← freshF; movFI z (.bits 0)
      stGlobalF ga z
    label skipStage
    barSync
    let isLane0 ← freshP; setpEqI isLane0 tid 0
    let done := "RT_DONE"
    braIfNot isLane0 done
    -- four passes, each taking the largest logit still standing
    let j ← freshR; movRC j 0
    let outer := "RT_OUTER"
    let outerEnd := "RT_OUTER_END"
    label outer
    do
      let p ← freshP; setpGeI p j TOPK; braIf p outerEnd
      let best ← freshF; movFI best (.bits 0xFF800000)   -- -inf
      let bi ← freshR; movRC bi 0
      let i ← freshR; movRC i 0
      let inner := "RT_INNER"
      let innerEnd := "RT_INNER_END"
      label inner
      do
        let q ← freshP; setpGeI q i NE; braIf q innerEnd
        let sa ← freshR; shlR sa i 2; addR sa sa sm
        let v ← freshF; ldSharedFD v sa
        -- `max.f32` gives the value; the index needs the predicate, and
        -- `selp.b32` on the bit pattern is how a float choice becomes an
        -- integer one without a branch.
        let gt ← freshP; setpGtF gt v best
        maxF best best v
        let ni ← freshR; selpR ni i bi gt
        movR bi ni
        addRI i i 1
        bra inner
      label innerEnd
      -- record the winner, then strike it out so the next pass cannot see it
      let coff ← freshRd; mulWideRI coff j 4
      let ca ← freshRd; addRd ca chosen coff
      stGlobalU32 ca bi
      let ga ← freshRd; addRd ga gates coff
      stGlobalF ga best
      let sa ← freshR; shlR sa bi 2; addR sa sa sm
      let ninf ← freshF; movFI ninf (.bits 0xFF800000)
      stSharedFD sa ninf
      addRI j j 1
      bra outer
    label outerEnd
    -- the gates: a softmax over the four, stabilised by the first, which is
    -- the largest because the passes above took them in order
    let mx ← freshF; ldGlobalF mx gates
    let sum ← freshF; movFI sum (.bits 0)
    let log2e := FImm.bits 0x3FB8AA3B
    for k in List.range TOPK do
      let ga ← freshRd; addRdI ga gates (4 * k)
      let v ← freshF; ldGlobalF v ga
      subF v v mx
      mulFI v v log2e
      ex2 v v
      stGlobalF ga v
      addF sum sum v
    let inv ← freshF; rcp inv sum
    for k in List.range TOPK do
      let ga ← freshRd; addRdI ga gates (4 * k)
      let v ← freshF; ldGlobalF v ga
      mulF v v inv
      stGlobalF ga v
    label done
    ptxRet


/-- **The greedy token: the largest logit's index.**

    Two hundred thousand logits, one block. Each lane strides the row keeping
    its own best, writes the pair to shared memory, and lane 0 picks among
    thirty-two — which is the whole reduction, because thirty-two is small and
    a tree would cost more in code than in cycles.

    The index is what is wanted, so the value is only ever a means; ties go to
    the lower index, which is what `>` rather than `>=` in the scan gives and
    what every reference implementation does. -/
def argmaxLogits : KernelSpec where
  name := "argmax_logits"
  params := ["p_logits", "p_out", "p_meta"]
  body := do
    let logits ← freshRd; ldParam64 logits "p_logits"
    let out ← freshRd; ldParam64 out "p_out"
    let metaP ← freshRd; ldParam64 metaP "p_meta"
    let n ← freshR; ldGlobalU n metaP
    let tid ← freshR; movR tid tidX
    let sm ← smemBase
    let best ← freshF; movFI best (.bits 0xFF800000)
    let bi ← freshR; movRC bi 0
    let i ← freshR; movR i tid
    let loop := "AM_LOOP"
    let ldone := "AM_DONE"
    label loop
    do
      let p ← freshP; setpGe p i n; braIf p ldone
      let off ← freshRd; mulWideRI off i 4
      let a ← freshRd; addRd a logits off
      let v ← freshF; ldGlobalF v a
      let gt ← freshP; setpGtF gt v best
      maxF best best v
      let ni ← freshR; selpR ni i bi gt
      movR bi ni
      addRI i i 32
      bra loop
    label ldone
    -- lane `tid` parks its pair: value at `tid`, index at `32 + tid`
    let va ← freshR; shlR va tid 2; addR va va sm
    stSharedFD va best
    let ia ← freshR; addRI ia va 128
    stSharedU32D ia bi
    barSync
    let isLane0 ← freshP; setpEqI isLane0 tid 0
    let fin := "AM_FIN"
    braIfNot isLane0 fin
    do
      let bv ← freshF; ldSharedFD bv sm
      let bx ← freshR; ldSharedU32D bx (← do let t ← freshR; addRI t sm 128; pure t)
      let k ← freshR; movRC k 1
      let sl := "AM_SEL"
      let slEnd := "AM_SEL_END"
      label sl
      do
        let p ← freshP; setpGeI p k 32; braIf p slEnd
        let ka ← freshR; shlR ka k 2; addR ka ka sm
        let v ← freshF; ldSharedFD v ka
        let ja ← freshR; addRI ja ka 128
        let jx ← freshR; ldSharedU32D jx ja
        let gt ← freshP; setpGtF gt v bv
        maxF bv bv v
        let nx ← freshR; selpR nx jx bx gt
        movR bx nx
        addRI k k 1
        bra sl
      label slEnd
      stGlobalU32 out bx
    label fin
    ptxRet


/-- **A sample from the tempered softmax, in one pass.**

    `argmax_i (logit_i / T + g_i)` with `g_i` independent standard Gumbel is
    exactly a draw from `softmax(logit / T)` — the Gumbel-max trick — so
    sampling costs what the argmax costs and needs no normalising constant, no
    prefix scan and no sort over two hundred thousand logits.

    The noise is generated from the index rather than stored: a hash of
    `(i, seed)` gives a uniform, and `g = -ln(-ln u)` follows. That keeps the
    kernel a pure function of its inputs, which is what lets a caller reproduce
    a sample by passing the same seed.

    `meta` is `[n, seed, 1/T as bits]`. A zero `1/T` would be a division by
    zero at temperature infinity; the caller sends greedy to the argmax kernel
    instead, which is the same thing at `T = 0` and cheaper to reason about. -/
def sampleLogits : KernelSpec where
  name := "sample_logits"
  params := ["p_logits", "p_out", "p_meta"]
  body := do
    let logits ← freshRd; ldParam64 logits "p_logits"
    let out ← freshRd; ldParam64 out "p_out"
    let metaP ← freshRd; ldParam64 metaP "p_meta"
    let n ← freshR; ldGlobalU n metaP
    let seed ← freshR; ldGlobalUO seed metaP 4
    let invT ← freshF; ldGlobalFO invT metaP 8
    let tid ← freshR; movR tid tidX
    let sm ← smemBase
    let best ← freshF; movFI best (.bits 0xFF800000)
    let bi ← freshR; movRC bi 0
    let i ← freshR; movR i tid
    let ln2 := FImm.bits 0x3F317218          -- ln 2
    let loop := "SM_LOOP"
    let ldone := "SM_DONE"
    label loop
    do
      let p ← freshP; setpGe p i n; braIf p ldone
      let off ← freshRd; mulWideRI off i 4
      let a ← freshRd; addRd a logits off
      let v ← freshF; ldGlobalF v a
      mulF v v invT
      -- a hash of the index, mixed with the seed
      let h ← freshR; xorRR h i seed
      mulLoRI h h 0x9E3779B1
      let t ← freshR; shrR t h 15; xorRR h h t
      mulLoRI h h 0x85EBCA6B
      shrR t h 13; xorRR h h t
      -- u in (0, 1): 24 bits, offset half an ulp so it is never zero
      shrR h h 8
      let uf ← freshF; cvtF32 uf h
      addFI uf uf (.bits 0x3F000000)         -- + 0.5
      mulFI uf uf (.bits 0x33800000)         -- * 2^-24
      -- g = -ln(-ln u) = -ln2 * lg2( -ln2 * lg2 u )
      let g ← freshF; lg2 g uf
      mulFI g g ln2
      negF g g
      lg2 g g
      mulFI g g ln2
      negF g g
      addF v v g
      let gt ← freshP; setpGtF gt v best
      maxF best best v
      let ni ← freshR; selpR ni i bi gt
      movR bi ni
      addRI i i 32
      bra loop
    label ldone
    let va ← freshR; shlR va tid 2; addR va va sm
    stSharedFD va best
    let ia ← freshR; addRI ia va 128
    stSharedU32D ia bi
    barSync
    let isLane0 ← freshP; setpEqI isLane0 tid 0
    let fin := "SM_FIN"
    braIfNot isLane0 fin
    do
      let bv ← freshF; ldSharedFD bv sm
      let bx ← freshR; ldSharedU32D bx (← do let t ← freshR; addRI t sm 128; pure t)
      let k ← freshR; movRC k 1
      let sl := "SM_SEL"
      let slEnd := "SM_SEL_END"
      label sl
      do
        let p ← freshP; setpGeI p k 32; braIf p slEnd
        let ka ← freshR; shlR ka k 2; addR ka ka sm
        let v ← freshF; ldSharedFD v ka
        let ja ← freshR; addRI ja ka 128
        let jx ← freshR; ldSharedU32D jx ja
        let gt ← freshP; setpGtF gt v bv
        maxF bv bv v
        let nx ← freshR; selpR nx jx bx gt
        movR bx nx
        addRI k k 1
        bra sl
      label slEnd
      stGlobalU32 out bx
    label fin
    ptxRet

/-- One kernel as its own module, entry named `main`.

    The launch primitive loads a module and runs the entry called `main`, so a
    module carries one kernel and a slot carries one module. -/
def moduleFor (k : KernelSpec) : String :=
  buildModuleWith { smemSize := smemBytes } [{ k with name := "main" }]

/-- The kernels a decode step launches, in slot order. -/
def gptossPtx : List String :=
  [moduleFor gateUpSwigluGemv, moduleFor downGemvBias, moduleFor moeCombine4,
   moduleFor routerTop4]

/-- The three kernels as one module, for the standalone bit-level harness. -/
def moduleText : String :=
  buildModule smemBytes [gateUpSwigluGemv, downGemvBias, dequantF32, moeCombine4]

end GptOssKernels

/-! The module text is emitted by a driver rather than by a `main` here: a
    `lean_exe` in this package is a *generator*, and every generator is expected
    to produce an artifact.  These three kernels become one in M3; until then
    the harness asks for the text directly. -/
