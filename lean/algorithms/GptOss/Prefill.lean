module
public import AlgorithmLib.Surface.Drv
meta import AlgorithmLib.Surface.Drv
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
public import GptOss.Kernels
meta import GptOss.Kernels
public import AlgorithmLib.Kernel.Block
meta import AlgorithmLib.Kernel.Block
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
  # Prefill kernels: many tokens against one weight

  A decode step multiplies one token by every weight it reaches, and reads the
  weight to do it: memory decides its speed. A prompt is many tokens through
  the same weights, so a weight read once can serve all of them, and the
  arithmetic decides instead. These kernels are written for that.

  ## Weights against int8 tokens, by the tensor cores

  An MXFP4 code is an e2m1 value: doubled, every one is an integer from -12 to
  12, and its block's e8m0 scale is a power of two; a Q8_0 value is an int8
  with an f32 scale a block. A token's activations, quantised to int8 a block
  of thirty-two at a time with one float scale, make each block's product an
  exact integer sum, which int8 `mma` (`Intr.mma`) computes for eight rows and
  four tokens at a time; it is scaled once a block by the weight's scale
  (halved, for MXFP4's doubling) and the activation's.

  ## A layer

  `GptOssDecode.dPrefillChunkM` runs a chunk of a prompt through these: the
  norm, the Q8 projections, the rotation, attention over the cache and the
  chunk, the router, the experts grouped by expert (one launch each for all
  of a layer's gate/up and down products), and the gated sum.
-/

namespace GptOssPrefill

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.Prog AlgorithmLib.Kernel AlgorithmLib.Kernel.Block

/-- Columns a weight row contracts over, and its blocks of thirty-two. -/
def K : Nat := 2880

/-- A block's tile: `TR` weight rows by `TT` tokens. -/
def TR : Nat := 128
def TT : Nat := 32

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

/-- **Four codes to four int8 values**, the low sixteen bits of `cw` (code
    `k` at bits `4k`) to byte `k` of the result, each its e2m1 value doubled:
    the nibbles spread a byte apart, a magnitude `m` maps to
    `m + [m ≥ 4] · (m mod 4 + 2 · [m mod 4 = 3])` (so `0 1 2 3 4 6 8 12`), and a
    byte with its sign bit is negated as `(0x80 - v) xor 0x80`, which borrows
    from no other byte since `v ≤ 12`. All four bytes at once. -/
def unpack4 (cw : V .i32) : Prog V L (V .i32) := do
  let x0 ← band cw (← c32 0xFFFF)
  let x1 ← band (← bor x0 (← ishlImm x0 8)) (← c32 0x00FF00FF)
  let x ← band (← bor x1 (← ishlImm x1 4)) (← c32 0x0F0F0F0F)
  let m ← band x (← c32 0x07070707)
  let ones ← c32 0x01010101
  let b1 ← band (← ushrImm m 2) ones
  let b2 ← isub (← ishlImm b1 8) b1
  let low2 ← band m (← c32 0x03030303)
  let both ← band (← band m (← ushrImm m 1)) ones
  let v ← iadd m (← band (← iadd low2 (← ishlImm both 1)) b2)
  let s1 ← band (← ushrImm x 3) ones
  let smask ← isub (← ishlImm s1 8) s1
  let h ← c32 0x80808080
  let negv ← bxor (← isub h v) h
  bor (← band negv smask) (← band v (← bxor smask (← c32 0xFFFFFFFF)))

/-- Words a tile row of the `mma` core occupies in shared memory: eight, and
    four more, so the eight rows (or four tokens) and four words a fragment
    load reads fall on distinct banks. -/
def MSTRIDE : Nat := 12

/-- How a weight matrix holds its values: MXFP4 (two codes a byte, an e8m0
    scale a block) or Q8_0 (an int8 a value, an f32 scale a block). -/
inductive Fmt | mx | q8
  deriving DecidableEq

/-- Blocks of thirty-two columns a stage takes between barriers, for a width
    of `k` columns: three where they divide its blocks, else two. -/
def ksOf (k : Nat) : Nat := if (k / 32) % 3 == 0 then 3 else 2

/-- A stage's tiles, `ks` of each kind: weights, tokens, their scales. -/
def WTILE : Nat := TR * MSTRIDE * 4
def XTILE : Nat := TT * MSTRIDE * 4
def M_W : Nat := 0
def M_X (ks : Nat) : Nat := M_W + ks * WTILE
def M_SW (ks : Nat) : Nat := M_X ks + ks * XTILE
def M_SX (ks : Nat) : Nat := M_SW ks + ks * TR * 4
def msmem (ks : Nat) : Nat := M_SX ks + ks * TT * 4

/-- **The tile's sums by int8 `mma`**: for tile rows `r < TR` and tile tokens
    `t < TT`, `Σ_c W[wRow r][c] · x[tokRow t][c]` over `k` columns, `W` in
    format `fmt` (its values at `wb`, its scales at `ws`, a row each), `x` as
    int8 (`xq`) with a float scale a block (`xsc`). Warp `w` takes rows `32 · (w mod 4) …` and tokens
    `16 · (w / 4) …`; each `mma` is eight of its rows (lane `l`'s row
    `l / 4`) against four of its tokens (lane `l` passing token `(l / 4) / 2`,
    receiving token `l mod 4`), sixteen columns at a time, two a block. Thread
    `tid`'s sum `4 · i + j` is tile row `32 · (w mod 4) + 8 i + (l / 4)` and tile
    token `16 · (w / 4) + 4 j + l mod 4`. `tokRow` answers a row in range for
    every tile token. -/
def mmaCore (fmt : Fmt) (k : Nat) (tid : V .i32) (sm wb ws xq xsc : V .i64) (wRow tokRow : V .i32 → Prog V L (V .i32)) :
    Prog V L (List (V .f32)) := do
  let KS := ksOf k
  let kb := k / 32
  let lane ← band tid (← c32 31)
  let w ← ushrImm tid 5
  let wr ← band w (← c32 3)
  let wc ← ushrImm w 2
  let g ← ushrImm lane 2
  let t ← band lane (← c32 3)
  let zero ← fconst .f32 0
  -- the global rows this thread stages, fixed for the whole contraction
  -- a thread's share of a block's weight tile: two code words of MXFP4 (a
  -- row's 16 bytes in four), or four value words of Q8 (its 32 in eight)
  let perRow := if fmt == .mx then 4 else 8
  let cRows ← (List.range (TR * perRow / 256)).mapM fun rep => do
    let q ← iadd tid (← c32 (256 * rep))
    pure (q, ← wRow (← udiv q (← c32 perRow)))
  let xRow ← tokRow (← ushrImm tid 3)
  let sRow ← wRow (← band tid (← c32 (TR - 1)))
  let sTok ← tokRow (← band tid (← c32 (TT - 1)))
  -- the fragments this lane reads: its rows' and its tokens' first words
  let aBase ← (List.range 4).mapM fun i => do
    at4 (← at_ sm M_W) (← imul (← iadd (← iadd (← ishlImm wr 5) (← c32 (8 * i))) g) (← c32 MSTRIDE))
  let bBase ← (List.range 4).mapM fun j => do
    at4 (← at_ sm (M_X KS)) (← imul (← iadd (← iadd (← ishlImm wc 4) (← c32 (4 * j))) (← ushrImm g 1)) (← c32 MSTRIDE))
  let sBase ← (List.range 4).mapM fun i => do
    at4 (← at_ sm (M_SW KS)) (← iadd (← iadd (← ishlImm wr 5) (← c32 (8 * i))) g)
  let xBase ← (List.range 4).mapM fun j => do
    at4 (← at_ sm (M_SX KS)) (← iadd (← iadd (← ishlImm wc 4) (← c32 (4 * j))) t)
  -- `KS` blocks a stage: a thread's code words (eight bytes of codes each,
  -- four a block's row), token words and scales for all of them, then one
  -- barrier
  let urem' (q : V .i32) (n : Nat) : Prog V L (V .i32) := do band q (← c32 (n - 1))
  let put (b0 : V .i32) : Prog V L Unit := do
    for kbi in List.range KS do
      let b ← iadd b0 (← c32 kbi)
      let rowBytes := if fmt == .mx then k / 2 else k
      let blkBytes := if fmt == .mx then 16 else 32
      let codes ← cRows.mapM fun (q, row) => do
        load32 (← iadd wb (← ext (← iadd (← imul row (← c32 rowBytes))
          (← iadd (← imul b (← c32 blkBytes)) (← ishlImm (← urem' q perRow) 2)))))
      let tok ← load32 (← iadd xq (← ext (← iadd (← imul xRow (← c32 k))
        (← iadd (← imul b (← c32 32)) (← ishlImm (← band tid (← c32 7)) 2)))))
      let sw ← if fmt == .mx then do
          let e ← GptOssKernels.Body.u8 (← iadd ws (← ext (← iadd (← imul sRow (← c32 kb)) b)))
          fmul (← bitcast .f32 (← ishlImm e 23)) (← fconst .f32 0x3F000000)
        else loadF32 (← at4 ws (← iadd (← imul sRow (← c32 kb)) b))
      let f ← loadF32 (← at4 xsc (← iadd (← imul sTok (← c32 kb)) b))
      let wTile ← at_ sm (M_W + kbi * WTILE)
      for (code, (q, _)) in codes.zip cRows do
        if fmt == .mx then
          let dst ← at4 wTile (← iadd (← imul (← ushrImm q 2) (← c32 MSTRIDE)) (← ishlImm (← band q (← c32 3)) 1))
          storeI32 (← unpack4 code) dst
          storeI32 (← unpack4 (← ushrImm code 16)) (← at_ dst 4)
        else
          storeI32 code (← at4 wTile (← iadd (← imul (← ushrImm q 3) (← c32 MSTRIDE)) (← band q (← c32 7))))
      storeI32 tok (← at4 (← at_ sm (M_X KS + kbi * XTILE)) (← iadd (← imul (← ushrImm tid 3) (← c32 MSTRIDE)) (← band tid (← c32 7))))
      when .ult tid (← c32 TR) do
        store sw (← at4 (← at_ sm (M_SW KS + 4 * TR * kbi)) tid)
      when .ult (← isub tid (← c32 TR)) (← c32 TT) do
        store f (← at4 (← at_ sm (M_SX KS + 4 * TT * kbi)) (← isub tid (← c32 TR)))
  let tw ← ishlImm t 2
  let st ← wloop (Vals.append %[← c32 0] (Vals.ofFn (n := 16) fun _ => zero))
    (fun cs => do pure (contIfULt cs.head (← c32 kb), cs, ()))
    (fun cs _ => do
      let b := cs.head
      put b
      callLocalVoid barrier %[]
      let zi ← c32 0
      let mut accs := Vals.uniformToList (cs.drop 1 : Vals V (List.replicate 16 .f32))
      for kbi in List.range KS do
        let frag (a : V .i64) (tile : Nat) : Prog V L (V .i32 × V .i32) := do
          let p ← iadd (← at_ a tile) (← ext tw)
          pure (← load32 p, ← load32 (← at_ p 16))
        let as ← aBase.mapM fun a => frag a (kbi * WTILE)
        let bs ← bBase.mapM fun a => frag a (kbi * XTILE)
        let sws ← sBase.mapM fun a => do loadF32 (← at_ a (4 * TR * kbi))
        let sxs ← xBase.mapM fun a => do loadF32 (← at_ a (4 * TT * kbi))
        let mut next : List (V .f32) := []
        for ((a0, a1), i) in as.zipIdx do
          for ((b0, b1), j) in bs.zipIdx do
            let p0 : V .i32 ← callLocal mmaR %[a0, b0, zi]
            let p1 : V .i32 ← callLocal mmaR %[a1, b1, p0]
            let sc ← fmul (sws.getD i zero) (sxs.getD j zero)
            next := next ++ [← fma (← fcvtFromSint .f32 p1) sc (accs.getD (4 * i + j) zero)]
        accs := next
      callLocalVoid barrier %[]
      pure (Vals.append %[← iadd b (← c32 KS)] (Vals.ofFn (n := 16) fun q => accs.getD q.val zero)))
  pure (Vals.uniformToList (st.drop 1 : Vals V (List.replicate 16 .f32)))

/-- Thread `tid`'s sum `4 · i + j` of `mmaCore`: its tile row and tile token. -/
def mmaAt (tid : V .i32) (i j : Nat) : Prog V L (V .i32 × V .i32) := do
  let lane ← band tid (← c32 31)
  let w ← ushrImm tid 5
  let row ← iadd (← iadd (← ishlImm (← band w (← c32 3)) 5) (← c32 (8 * i))) (← ushrImm lane 2)
  let tok ← iadd (← iadd (← ishlImm (← ushrImm w 2) 4) (← c32 (4 * j))) (← band lane (← c32 3))
  pure (row, tok)

/-- **Q8_0 rows against int8 tokens, plus a bias**: `out[t][r] = bias[r] +
    Σ_c W[r][c] · x[t][c]` for the `rows × cols` matrix `w` (its int8 values
    row by row, then its f32 scales, as `prepare_q8.py` lays one out) and `T`
    tokens (`meta`: `T`, a multiple of the tile), `x` as int8 a token with one
    float scale a block. Block `cta` takes token tile `cta mod (T / TT)` and
    row tile `cta / (T / TT)`; a row past `rows` is computed and not written. -/
def q8GemmBody (rows cols : Nat) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ List.replicate 6 .i64)
  let cta := ps.head; let tid := ps.snd; let sm := (ps.drop 3).head
  let w := (ps.drop 4).head; let xq := (ps.drop 5).head; let xsc := (ps.drop 6).head
  let out := (ps.drop 7).head; let bias := (ps.drop 8).head; let mt := (ps.drop 9).head
  let ws ← iaddImm w (rows * cols)
  let nTT ← udiv (← load32 mt) (← c32 TT)
  let rt ← udiv cta nTT
  let tt ← isub cta (← imul rt nTT)
  let row0 ← imul rt (← c32 TR)
  let tok0 ← imul tt (← c32 TT)
  let wRow (r : V .i32) : Prog V L (V .i32) := do
    let row ← iadd row0 r
    select (← icmp .ult row (← c32 rows)) row (← c32 (rows - 1))
  let accs ← mmaCore .q8 cols tid sm w ws xq xsc wRow (fun t => iadd tok0 t)
  let zero ← fconst .f32 0
  for i in List.range 4 do
    let (r, _) ← mmaAt tid i 0
    let row ← iadd row0 r
    when .ult row (← c32 rows) do
      let b ← loadF32 (← at4 bias row)
      for j in List.range 4 do
        let (_, t) ← mmaAt tid i j
        store (← fadd (accs.getD (4 * i + j) zero) b) (← at4 out (← iadd (← imul (← iadd tok0 t) (← c32 rows)) row))

/-- Experts a layer has, and the expert intermediate width. -/
def NE : Nat := GptOssKernels.NE
def I : Nat := GptOssKernels.I

/-- **Which expert's tile a block computes** from the layer's tile prefix
    `tileOff` (`NE + 1` words; expert `e` has tiles `tileOff[e] … tileOff[e+1]`):
    the expert whose range holds token tile `gt`, and the tile's index within
    it. -/
def tileExpert (tileOff : V .i64) (gt : V .i32) : Prog V L (V .i32 × V .i32) := do
  let mut e ← c32 0
  for k in List.range NE do
    let past ← icmp .ule (← load32 (← at4 tileOff (← c32 (k + 1)))) gt
    e ← iadd e (← select past (← c32 1) (← c32 0))
  pure (e, ← isub gt (← load32 (← at4 tileOff e)))

/-- **Gate and up of every token routed to every expert, and the SwiGLU**,
    one launch a layer. The routed (token, expert) pairs are in expert order:
    pair `p` is token `pairTok[p]`, expert `e`'s pairs `eOff[e] … eOff[e+1]`, in
    tiles of `TT` (`tileOff`, the `NE + 1` words after `eOff`'s); a block takes row tile `cta mod (I / 64)`, its 64 gate rows
    and their 64 up rows, and token tile `cta / (I / 64)` of all the layer's.
    The expert's weights sit in pool slot `slotOf[layer · NE + e]` (`meta`:
    `layer`), its activations' int8 rows at `xq`, scales at `xsc`; pair `p`'s
    SwiGLU outputs go to `hid + 4 · I · p`, gpt-oss's: the gate clamped above,
    the linear branch clamped both sides and offset by one, the gate's
    sigmoid by the hardware's approximations. -/
def moeGateUpBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ List.replicate 8 .i64)
  let cta := ps.head; let tid := ps.snd; let sm := (ps.drop 3).head
  let pool := (ps.drop 4).head; let slotOf := (ps.drop 5).head; let mt := (ps.drop 6).head
  let xq := (ps.drop 7).head; let xsc := (ps.drop 8).head; let pairTok := (ps.drop 9).head
  let eOff := (ps.drop 10).head; let hid := (ps.drop 11).head
  let tileOff ← iaddImm eOff (4 * (NE + 1))
  let nRT ← c32 (I / 64)
  let gt ← udiv cta nRT
  let rt ← isub cta (← imul gt nRT)
  -- a block past the layer's tiles has nothing to do (the grid is the most
  -- tiles any routing can make)
  when .ult gt (← load32 (← at4 tileOff (← c32 NE))) do
    let (e, inE) ← tileExpert tileOff gt
    let layer ← load32 mt
    let slot ← load32 (← at4 slotOf (← iadd (← imul layer (← c32 NE)) e))
    let wb ← iadd pool (← imul (← ext slot) (← c64n GptOssKernels.EXPERT_BYTES))
    let ws ← iaddImm wb (GptOssKernels.expertPieceOff 1)
    let bias ← iaddImm wb (GptOssKernels.expertPieceOff 2)
    let p0 ← iadd (← load32 (← at4 eOff e)) (← imul inE (← c32 TT))
    let pEnd ← load32 (← at4 eOff (← iadd e (← c32 1)))
    let g0 ← imul rt (← c32 64)
    -- tile row `32 w + 8 i + g`: gate row `g0 + 16 w + 8 (i mod 2) + g` for
    -- `i < 2`, its up row for `i ≥ 2`, so a lane holds each gate with its up
    let wRow (r : V .i32) : Prog V L (V .i32) := do
      let ri ← band (← ushrImm r 3) (← c32 3)
      let go ← iadd (← iadd (← ishlImm (← ushrImm r 5) 4) (← ishlImm (← band ri (← c32 1)) 3)) (← band r (← c32 7))
      iadd (← iadd g0 go) (← select (← icmp .uge ri (← c32 2)) (← c32 I) (← c32 0))
    let tokRow (t : V .i32) : Prog V L (V .i32) := do
      let p ← iadd p0 t
      let pc ← select (← icmp .ult p pEnd) p (← isub pEnd (← c32 1))
      load32 (← at4 pairTok pc)
    let accs ← mmaCore .mx K tid sm wb ws xq xsc wRow tokRow
    let zero ← fconst .f32 0
    let f (b : UInt32) : Prog V L (V .f32) := fconst .f32 b.toUInt64
    for i in List.range 2 do
      let (r, _) ← mmaAt tid i 0
      let g ← wRow r
      let bg ← loadF32 (← at4 bias g)
      let bu ← loadF32 (← at4 bias (← iadd g (← c32 I)))
      for j in List.range 4 do
        let (_, tk) ← mmaAt tid i j
        let p ← iadd p0 tk
        when .ult p pEnd do
          let gv ← fmin (← fadd (accs.getD (4 * i + j) zero) bg) (← f GptOssKernels.swigluLimitBits)
          let uv ← fmax (← fmin (← fadd (accs.getD (4 * (i + 2) + j) zero) bu)
            (← f GptOssKernels.swigluLimitBits)) (← f GptOssKernels.negLimitBits)
          let t ← fmul (← fmul gv (← f GptOssKernels.swigluAlphaBits)) (← f log2eBits)
          let ex ← callLocal ex2Approx %[← fsub (← f 0) t]
          let sig ← callLocal rcpApprox %[← fadd ex (← f GptOssKernels.oneBits)]
          let y ← fmul (← fmul gv sig) (← fadd uv (← f GptOssKernels.oneBits))
          store y (← at4 hid (← iadd (← imul p (← c32 I)) g))

/-- Hidden width. -/
def H : Nat := GptOssKernels.H
def TOPK : Nat := GptOssKernels.TOPK

/-- **Rows to int8, a block of thirty-two at a time**: block `q` of the
    `n · 32` (`meta`: `n`), one a thread, scaled by its largest magnitude over
    127 and rounded half away from zero; its int8 values to `outq + 32 q`, its
    scale (the largest magnitude over 127, 0 for a block of zeros) to
    `outs + 4 q`. -/
def quantRowsBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ List.replicate 4 .i64)
  let inp := (ps.drop 4).head; let outq := (ps.drop 5).head
  let outs := (ps.drop 6).head; let mt := (ps.drop 7).head
  let q ← iadd (← imul ps.head (← c32 256)) ps.snd
  when .ult q (← load32 mt) do
    let src ← iadd inp (← imul (← ext q) (← c64n 128))
    let xs ← (List.range 32).mapM fun k => do loadF32 (← at_ src (4 * k))
    let mut amax ← fconst .f32 0
    for x in xs do amax ← fmax amax (← fabs x)
    let pos ← fcmp .gt amax (← fconst .f32 0)
    let inv ← select pos (← fdiv (← fconst .f32 0x42FE0000) amax) (← fconst .f32 0)
    let half ← fconst .f32 0x3F000000
    let nhalf ← fconst .f32 0xBF000000
    let dst ← iadd outq (← imul (← ext q) (← c64n 32))
    for w in List.range 8 do
      let mut word ← c32 0
      for k in List.range 4 do
        let x := xs.getD (4 * w + k) amax
        let y ← fmul x inv
        let r ← fadd y (← select (← fcmp .ge y (← fconst .f32 0)) half nhalf)
        let b ← band (← fconv .toSint .i32 r) (← c32 0xFF)
        word ← bor word (← ishlImm b (8 * k))
      storeI32 word (← at_ dst (4 * w))
    store (← fmul amax (← fconst .f32 0x3C010204)) (← at4 outs q)

/-- **Down of every routed pair**, one launch a layer: pair `p`'s SwiGLU
    output, quantised (`hq`, `hs`), against its expert's down rows, plus their
    bias, to `y + 4 · H · p`. Pairs as `moeGateUpBody` has them; a block takes
    row tile `cta mod ⌈H / TR⌉` (the last one part full) and token tile
    `cta / ⌈H / TR⌉`. -/
def moeDownBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ List.replicate 7 .i64)
  let cta := ps.head; let tid := ps.snd; let sm := (ps.drop 3).head
  let pool := (ps.drop 4).head; let slotOf := (ps.drop 5).head; let mt := (ps.drop 6).head
  let hq := (ps.drop 7).head; let hs := (ps.drop 8).head
  let eOff := (ps.drop 9).head; let y := (ps.drop 10).head
  let tileOff ← iaddImm eOff (4 * (NE + 1))
  let nRT ← c32 ((H + TR - 1) / TR)
  let gt ← udiv cta nRT
  let rt ← isub cta (← imul gt nRT)
  -- a block past the layer's tiles has nothing to do (the grid is the most
  -- tiles any routing can make)
  when .ult gt (← load32 (← at4 tileOff (← c32 NE))) do
    let (e, inE) ← tileExpert tileOff gt
    let layer ← load32 mt
    let slot ← load32 (← at4 slotOf (← iadd (← imul layer (← c32 NE)) e))
    let base ← iadd pool (← imul (← ext slot) (← c64n GptOssKernels.EXPERT_BYTES))
    let wb ← iaddImm base (GptOssKernels.expertPieceOff 3)
    let ws ← iaddImm base (GptOssKernels.expertPieceOff 4)
    let bias ← iaddImm base (GptOssKernels.expertPieceOff 5)
    let p0 ← iadd (← load32 (← at4 eOff e)) (← imul inE (← c32 TT))
    let pEnd ← load32 (← at4 eOff (← iadd e (← c32 1)))
    let row0 ← imul rt (← c32 TR)
    let wRow (r : V .i32) : Prog V L (V .i32) := do
      let row ← iadd row0 r
      select (← icmp .ult row (← c32 H)) row (← c32 (H - 1))
    let tokRow (t : V .i32) : Prog V L (V .i32) := do
      let p ← iadd p0 t
      select (← icmp .ult p pEnd) p (← isub pEnd (← c32 1))
    let accs ← mmaCore .mx K tid sm wb ws hq hs wRow tokRow
    let zero ← fconst .f32 0
    for i in List.range 4 do
      let (r, _) ← mmaAt tid i 0
      let row ← iadd row0 r
      when .ult row (← c32 H) do
        let b ← loadF32 (← at4 bias row)
        for j in List.range 4 do
          let (_, tk) ← mmaAt tid i j
          let p ← iadd p0 tk
          when .ult p pEnd do
            store (← fadd (accs.getD (4 * i + j) zero) b) (← at4 y (← iadd (← imul p (← c32 H)) row))

/-- **The router's decision for every token**: token `t`'s `TOPK` largest of
    its `NE` logits (`logits + 4 · NE · t`), their experts to `chosen + 4 · TOPK
    · t` and the softmax of their logits to `gw` alike, as `routerTop` decides
    one token's; a thread a token (`meta`: `T`). -/
def routerTopTBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ List.replicate 4 .i64)
  let logits := (ps.drop 4).head; let chosen := (ps.drop 5).head
  let gw := (ps.drop 6).head; let mt := (ps.drop 7).head
  let t ← iadd (← imul ps.head (← c32 256)) ps.snd
  when .ult t (← load32 mt) do
    let row ← at4 logits (← imul t (← c32 NE))
    let ls ← (List.range NE).mapM fun i => do loadF32 (← at_ row (4 * i))
    let ninf ← fconst .f32 0xFF800000
    let mut taken : List (V .i32) := []
    let mut tops : List (V .f32) := []
    for _ in List.range TOPK do
      let mut best := ninf
      let mut bi ← c32 NE
      for (v, i) in ls.zipIdx do
        -- an expert already chosen is out
        let mut out ← c32 0
        for c in taken do out ← bor out (← select (← icmp .eq c (← c32 i)) (← c32 1) (← c32 0))
        let gt ← fcmp .gt v best
        let ok ← band (← select gt (← c32 1) (← c32 0)) (← isub (← c32 1) out)
        let take ← icmp .ne ok (← c32 0)
        best ← select take v best
        bi ← select take (← c32 i) bi
      taken := taken ++ [bi]
      tops := tops ++ [best]
    let mx := tops.headD ninf
    let log2e ← fconst .f32 log2eBits.toUInt64
    let mut es : List (V .f32) := []
    let mut sum ← fconst .f32 0
    for v in tops do
      let e : V .f32 ← callLocal ex2Approx %[← fmul (← fsub v mx) log2e]
      es := es ++ [e]
      sum ← fadd sum e
    let inv ← callLocal rcpApprox %[sum]
    let at_t (b : V .i64) : Prog V L (V .i64) := do at4 b (← imul t (← c32 TOPK))
    for ((e, c), k) in (es.zip taken).zipIdx do
      storeI32 c (← at_ (← at_t chosen) (4 * k))
      store (← fmul e inv) (← at_ (← at_t gw) (4 * k))

/-- **The routed pairs in expert order**, one block: from the `T · TOPK`
    choices (`meta`: `T`; choice `c = TOPK · t + j` is token `t`'s `j`th),
    each expert's count and its pairs' start (`eOff`, `NE + 1` words), its
    tiles' start (`tileOff`, the `NE + 1` words after, `TT` pairs a tile), each pair's token
    (`pairTok`, an expert's pairs in token order) and each choice's pair
    (`pos`). A thread an expert walks every choice twice: to count, then to
    place. -/
def groupBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ List.replicate 5 .i64)
  let tid := ps.snd; let sm := (ps.drop 3).head
  let chosen := (ps.drop 4).head; let eOff := (ps.drop 5).head
  let pairTok := (ps.drop 6).head; let pos := (ps.drop 7).head; let mt := (ps.drop 8).head
  let tileOff ← iaddImm eOff (4 * (NE + 1))
  let n ← imul (← load32 mt) (← c32 TOPK)
  when .ult tid (← c32 NE) do
    let cnt ← wloop %[← c32 0, ← c32 0] (fun cs => do pure (contIfULt cs.head n, cs, ()))
      (fun cs _ => do
        let hit ← icmp .eq (← load32 (← at4 chosen cs.head)) tid
        pure %[← iadd cs.head (← c32 1), ← iadd cs.snd (← select hit (← c32 1) (← c32 0))])
    storeI32 cnt.snd (← at4 sm tid)
  callLocalVoid barrier %[]
  when .ult tid (← c32 NE) do
    let mut start ← c32 0
    let mut tstart ← c32 0
    for k in List.range NE do
      let ck ← load32 (← at_ sm (4 * k))
      let before ← icmp .ult (← c32 k) tid
      start ← iadd start (← select before ck (← c32 0))
      tstart ← iadd tstart (← select before (← udiv (← iadd ck (← c32 (TT - 1))) (← c32 TT)) (← c32 0))
    storeI32 start (← at4 eOff tid)
    storeI32 tstart (← at4 tileOff tid)
    let mine ← load32 (← at4 sm tid)
    when .eq tid (← c32 (NE - 1)) do
      storeI32 (← iadd start mine) (← at4 eOff (← c32 NE))
      storeI32 (← iadd tstart (← udiv (← iadd mine (← c32 (TT - 1))) (← c32 TT))) (← at4 tileOff (← c32 NE))
    let _ ← wloop %[← c32 0, start] (fun cs => do pure (contIfULt cs.head n, cs, ()))
      (fun cs _ => do
        let c := cs.head
        let hit ← icmp .eq (← load32 (← at4 chosen c)) tid
        when .ne (← select hit (← c32 1) (← c32 0)) (← c32 0) do
          storeI32 (← udiv c (← c32 TOPK)) (← at4 pairTok cs.snd)
          storeI32 cs.snd (← at4 pos c)
        pure %[← iadd c (← c32 1), ← iadd cs.snd (← select hit (← c32 1) (← c32 0))])
    pure ()

/-- **Each token's experts, gated, added to it**: `x[t][i] += Σ_j gw[t][j] ·
    y[pos[t][j]][i]`, a thread an element (`meta`: `T`), the choices added in
    order. -/
def combineBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ List.replicate 5 .i64)
  let x := (ps.drop 4).head; let y := (ps.drop 5).head; let gw := (ps.drop 6).head
  let pos := (ps.drop 7).head; let mt := (ps.drop 8).head
  let g ← iadd (← imul ps.head (← c32 256)) ps.snd
  let t ← udiv g (← c32 H)
  let i ← isub g (← imul t (← c32 H))
  when .ult t (← load32 mt) do
    let mut acc ← fconst .f32 0
    for j in List.range TOPK do
      let c ← iadd (← imul t (← c32 TOPK)) (← c32 j)
      let p ← load32 (← at4 pos c)
      let w ← loadF32 (← at4 gw c)
      acc ← fma w (← loadF32 (← at4 y (← iadd (← imul p (← c32 H)) i))) acc
    let xa ← at4 x g
    store (← fadd (← loadF32 xa) acc) xa

/-! ## A layer's other kernels, for many tokens

Each takes the chunk's `T` tokens at positions `start …` (`meta`: `T`,
`start`, then what the kernel says), its rows side by side. -/

/-- Head width, query and key heads, and where keys and values sit in a
    token's packed QKV row. -/
def HD : Nat := 64
def NQ : Nat := 64
def NKV : Nat := 8
def QW : Nat := NQ * HD
def KBASE : Nat := QW
def VBASE : Nat := QW + NKV * HD
def QKVW : Nat := QW + 2 * NKV * HD

/-- **RMSNorm of every row**: a warp a row (`meta`: `T`), each lane the
    squares of every thirty-second element, the warp's sum by butterfly, then
    `x · w / √(Σx² / H + ε)`. -/
def rmsTBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ List.replicate 4 .i64)
  let x := (ps.drop 4).head; let w := (ps.drop 5).head; let out := (ps.drop 6).head; let mt := (ps.drop 7).head
  let lane ← band ps.snd (← c32 31)
  let t ← iadd (← imul ps.head (← c32 8)) (← ushrImm ps.snd 5)
  when .ult t (← load32 mt) do
    let row ← at4 x (← imul t (← c32 H))
    let mut acc ← fconst .f32 0
    for i in List.range (H / 32) do
      let v ← loadF32 (← at4 row (← iadd lane (← c32 (32 * i))))
      acc ← fma v v acc
    for m in [16, 8, 4, 2, 1] do
      let o ← callLocal shflBfly %[← bitcast .i32 acc, ← c32 m]
      acc ← fadd acc (← bitcast .f32 o)
    let sc : V .f32 ← callLocal rsqrtApprox %[← fadd (← fmul acc (← fl (1.0 / 2880.0))) (← fl 0.00001)]
    let orow ← at4 out (← imul t (← c32 H))
    for i in List.range (H / 32) do
      let ix ← iadd lane (← c32 (32 * i))
      let v ← loadF32 (← at4 row ix)
      store (← fmul (← fmul v (← loadF32 (← at4 w ix))) sc) (← at4 orow ix)

/-- **The rotation of every token's query and key heads**: a thread a pair
    (`meta`: `T`, `start`), token `t` at position `start + t`, the table's
    sines then cosines a position each, as the decode's rotation reads it. -/
def ropeTBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ List.replicate 3 .i64)
  let qkv := (ps.drop 4).head; let tab := (ps.drop 5).head; let mt := (ps.drop 6).head
  let g ← iadd (← imul ps.head (← c32 256)) ps.snd
  let perTok := (NQ + NKV) * (HD / 2)
  let t ← udiv g (← c32 perTok)
  let r ← isub g (← imul t (← c32 perTok))
  when .ult t (← load32 mt) do
    let h ← udiv r (← c32 (HD / 2))
    let lane ← isub r (← imul h (← c32 (HD / 2)))
    let pos ← iadd (← load32 (← at_ mt 4)) t
    let base ← iadd (← imul t (← c32 QKVW)) (← iadd (← imul h (← c32 HD)) lane)
    let lo ← at4 qkv base
    let hi ← at4 qkv (← iadd base (← c32 (HD / 2)))
    let ti ← iadd (← imul pos (← c32 (HD / 2))) lane
    let sn ← loadF32 (← at4 tab ti)
    let cs ← loadF32 (← at4 tab (← iadd ti (← c32 (GptOssAttention.ROPE_N * (HD / 2)))))
    let a ← loadF32 lo
    let b ← loadF32 hi
    store (← fsub (← fmul a cs) (← fmul b sn)) lo
    store (← fadd (← fmul a sn) (← fmul b cs)) hi

/-- **Every token's keys and values, narrowed into the chunk's buffers**: a
    thread a word (two bf16, rounded to nearest even) of one head of one token
    (`meta`: `T`); token `t`'s key head `k` at `kc + 4 · (t · NKV + k) · HD / 2`,
    its values alike in `vc`. -/
def kvNarrowBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ List.replicate 4 .i64)
  let qkv := (ps.drop 4).head; let kc := (ps.drop 5).head; let vc := (ps.drop 6).head; let mt := (ps.drop 7).head
  let g ← iadd (← imul ps.head (← c32 256)) ps.snd
  let perTok := NKV * HD / 2
  let t ← udiv g (← c32 perTok)
  let r ← isub g (← imul t (← c32 perTok))
  when .ult t (← load32 mt) do
    let row ← iadd (← imul t (← c32 QKVW)) (← ishlImm r 1)
    for (base, dst) in [(KBASE, kc), (VBASE, vc)] do
      let a ← at4 qkv (← iadd row (← c32 base))
      let lo ← GptOssKernels.Body.rneBf16 (← loadF32 a)
      let hi ← GptOssKernels.Body.rneBf16 (← loadF32 (← at_ a 4))
      storeI32 (← bor lo (← ishlImm hi 16)) (← at4 dst g)

/-- **The chunk's keys and values into the cache**: a thread a word (`meta`:
    `T`, `start`, the window (unread), `cap` the cache's slots, `stride` its
    words a head, as `attnTBody` reads it too), token
    `t` to slot `(start + t) mod cap` of each head, as the decode's store puts a
    token: head `k`'s slot `s` at word `k · stride + s · HD / 2`. Launched after
    the chunk's attention, which a ring would otherwise lose keys under. -/
def kvStoreTBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ List.replicate 5 .i64)
  let kc := (ps.drop 4).head; let vc := (ps.drop 5).head
  let kcache := (ps.drop 6).head; let vcache := (ps.drop 7).head; let mt := (ps.drop 8).head
  let g ← iadd (← imul ps.head (← c32 256)) ps.snd
  let perTok := NKV * HD / 2
  let t ← udiv g (← c32 perTok)
  let r ← isub g (← imul t (← c32 perTok))
  when .ult t (← load32 mt) do
    let pos ← iadd (← load32 (← at_ mt 4)) t
    let cap ← load32 (← at_ mt 12)
    let slot ← isub pos (← imul (← udiv pos cap) cap)
    let k ← udiv r (← c32 (HD / 2))
    let wd ← isub r (← imul k (← c32 (HD / 2)))
    let dst ← iadd (← imul k (← load32 (← at_ mt 16))) (← iadd (← imul slot (← c32 (HD / 2))) wd)
    storeI32 (← load32 (← at4 kc g)) (← at4 kcache dst)
    storeI32 (← load32 (← at4 vc g)) (← at4 vcache dst)

/-- Tokens a block of the prefill attention takes, and keys it stages at once. -/
def AT_TOK : Nat := 16
def AT_KEYS : Nat := 32

set_option maxRecDepth 8000 in
/-- **Attention of every token**, causal, a key window where the layer slides,
    the learned sink in the denominator. A block takes key head `kh = cta mod
    NKV` and tokens `AT_TOK · (cta / NKV) …`; thread `i < 128` the query head
    `GQA · kh + i mod 8` of token `i / 8` (`meta`: `T`, `start`, `window` (0
    for none), `cap`, `stride`). Token `t` at position `p = start + t` sees the
    keys at `max(0, p - window + 1) … p`, scores `q · k / 8`; a key before
    `start` is the cache's (slot `j mod cap`), one at or after it the chunk's
    (`kc`, `vc`, as `kvNarrowBody` writes them). Keys are staged `AT_KEYS` at a
    time; the softmax is online, and the sink joins once at the end, as the
    decode's combine applies it. -/
def attnTBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ List.replicate 8 .i64)
  let cta := ps.head; let tid := ps.snd; let sm := (ps.drop 3).head
  let qkv := (ps.drop 4).head; let kc := (ps.drop 5).head; let vc := (ps.drop 6).head
  let kcache := (ps.drop 7).head; let vcache := (ps.drop 8).head; let sinks := (ps.drop 9).head
  let out := (ps.drop 10).head; let mt := (ps.drop 11).head
  let T ← load32 mt
  let start ← load32 (← at_ mt 4)
  let window ← load32 (← at_ mt 8)
  let cap ← load32 (← at_ mt 12)
  let stride ← load32 (← at_ mt 16)
  let kh ← isub cta (← imul (← udiv cta (← c32 NKV)) (← c32 NKV))
  let t0 ← imul (← udiv cta (← c32 NKV)) (← c32 AT_TOK)
  let tl ← ushrImm tid 3
  let t ← iadd t0 tl
  let tc ← select (← icmp .ult t T) t (← isub T (← c32 1))
  let head ← iadd (← imul kh (← c32 8)) (← band tid (← c32 7))
  let p ← iadd start tc
  -- the block's key range: from its first token's window to its last token
  let tLast ← isub (← select (← icmp .ult (← iadd t0 (← c32 AT_TOK)) T) (← iadd t0 (← c32 AT_TOK)) T) (← c32 1)
  let pLo ← iadd start t0
  let lo0 ← select (← icmp .ne window (← c32 0))
    (← select (← icmp .uge pLo window) (← isub (← iadd pLo (← c32 1)) window) (← c32 0)) (← c32 0)
  let hi0 ← iadd (← iadd start tLast) (← c32 1)
  let myLo ← select (← icmp .ne window (← c32 0))
    (← select (← icmp .uge p window) (← isub (← iadd p (← c32 1)) window) (← c32 0)) (← c32 0)
  -- this thread's query, scaled by 1/8
  let qrow ← at4 qkv (← iadd (← imul tc (← c32 QKVW)) (← imul head (← c32 HD)))
  let eighth ← fl 0.125
  let q ← (List.range HD).mapM fun d => do fmul (← loadF32 (← at_ qrow (4 * d))) eighth
  let zero ← fconst .f32 0
  let ninf ← fconst .f32 0xFF800000
  let ks ← at_ sm 0
  let vs ← at_ sm (AT_KEYS * HD * 4)
  let nTiles ← udiv (← iadd (← isub hi0 lo0) (← c32 (AT_KEYS - 1))) (← c32 AT_KEYS)
  let st ← wloop (Vals.append %[← c32 0, ninf, zero] (Vals.ofFn (n := HD) fun _ => zero))
    (fun cs => do pure (contIfULt cs.head nTiles, cs, ()))
    (fun cs _ => do
      let j0 ← iadd lo0 (← imul cs.head (← c32 AT_KEYS))
      -- stage the tile's keys and values as floats: 32 keys of 32 words each
      for rep in List.range (AT_KEYS * HD / 2 / 128) do
        let wi ← iadd tid (← c32 (128 * rep))
        let kj ← ushrImm wi 5
        let wd ← band wi (← c32 31)
        let j ← iadd j0 kj
        let jc ← select (← icmp .ult j hi0) j lo0
        let fromCache ← icmp .ult jc start
        let slot ← isub jc (← imul (← udiv jc cap) cap)
        let cIx ← iadd (← imul kh stride) (← iadd (← imul slot (← c32 (HD / 2))) wd)
        let chunkT ← select fromCache (← c32 0) (← isub jc start)
        let kIx ← iadd (← imul (← iadd (← imul chunkT (← c32 NKV)) kh) (← c32 (HD / 2))) wd
        for (cache, chunk, dst) in [(kcache, kc, ks), (vcache, vc, vs)] do
          let wcache ← load32 (← at4 cache (← select fromCache cIx (← c32 0)))
          let wchunk ← load32 (← at4 chunk (← select fromCache (← c32 0) kIx))
          let wv ← select fromCache wcache wchunk
          let lo ← bitcast .f32 (← ishlImm wv 16)
          let hi ← bitcast .f32 (← band wv (← c32 0xFFFF0000))
          let d ← at4 dst (← iadd (← imul kj (← c32 HD)) (← ishlImm wd 1))
          store lo d
          store hi (← at_ d 4)
      callLocalVoid barrier %[]
      let inner ← wloop (Vals.append %[← c32 0, (cs.drop 1).head, (cs.drop 2).head]
          (Vals.ofFn (n := HD) fun q => (Vals.uniformToList (cs.drop 3 : Vals V (List.replicate HD .f32))).getD q.val zero))
        (fun ic => do pure (contIfULt ic.head (← c32 AT_KEYS), ic, ()))
        (fun ic _ => do
          let j ← iadd j0 ic.head
          let m := (ic.drop 1).head
          let l := (ic.drop 2).head
          let acc := Vals.uniformToList (ic.drop 3 : Vals V (List.replicate HD .f32))
          let seen ← band (← select (← icmp .uge j myLo) (← c32 1) (← c32 0))
            (← select (← icmp .ule j p) (← c32 1) (← c32 0))
          let kr ← at4 ks (← imul ic.head (← c32 HD))
          let mut s := zero
          for (qd, d) in q.zipIdx do
            s ← fma qd (← loadF32 (← at_ kr (4 * d))) s
          let live ← icmp .ne seen (← c32 0)
          let sx ← select live s ninf
          let mNew ← fmax m sx
          let corr ← expA (← fsub m mNew)
          let pr ← select live (← expA (← fsub sx mNew)) zero
          let lNew ← fma l corr pr
          let vr ← at4 vs (← imul ic.head (← c32 HD))
          let mut nacc : List (V .f32) := []
          for (a, d) in acc.zipIdx do
            nacc := nacc ++ [← fma (← loadF32 (← at_ vr (4 * d))) pr (← fmul a corr)]
          -- a key not seen yet leaves the state as it was (no `-inf - -inf`)
          let mK ← select live mNew m
          let lK ← select live lNew l
          let accK ← (acc.zip nacc).mapM fun (a, b) => select live b a
          pure (Vals.append %[← iadd ic.head (← c32 1), mK, lK] (Vals.ofFn (n := HD) fun q => accK.getD q.val zero)))
      callLocalVoid barrier %[]
      pure (Vals.append %[← iadd cs.head (← c32 1), (inner.drop 1).head, (inner.drop 2).head]
        (Vals.ofFn (n := HD) fun q => (Vals.uniformToList (inner.drop 3 : Vals V (List.replicate HD .f32))).getD q.val zero)))
  when .ult t T do
    let m := (st.drop 1).head
    let l := (st.drop 2).head
    let acc := Vals.uniformToList (st.drop 3 : Vals V (List.replicate HD .f32))
    let sink ← loadF32 (← at4 sinks head)
    let mAll ← fmax m sink
    let w ← expA (← fsub m mAll)
    let lAll ← fadd (← fmul l w) (← expA (← fsub sink mAll))
    let inv : V .f32 ← callLocal rcpApprox %[lAll]
    let scale ← fmul w inv
    let orow ← at4 out (← iadd (← imul t (← c32 QW)) (← imul head (← c32 HD)))
    for (a, d) in acc.zipIdx do
      store (← fmul a scale) (← at_ orow (4 * d))

/-- **The router's logits for every token**: a warp a (token, expert)
    (`meta`: `T`), `bias[e] + Σ_c W[e][c] · x[t][c]`, lanes every
    thirty-second column, the warp's sum by butterfly. -/
def routerLogitsBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ List.replicate 5 .i64)
  let x := (ps.drop 4).head; let w := (ps.drop 5).head; let bias := (ps.drop 6).head
  let out := (ps.drop 7).head; let mt := (ps.drop 8).head
  let lane ← band ps.snd (← c32 31)
  let g ← iadd (← imul ps.head (← c32 8)) (← ushrImm ps.snd 5)
  let t ← udiv g (← c32 NE)
  let e ← isub g (← imul t (← c32 NE))
  when .ult t (← load32 mt) do
    let xr ← at4 x (← imul t (← c32 H))
    let wr ← at4 w (← imul e (← c32 H))
    let mut acc ← fconst .f32 0
    for i in List.range (H / 32) do
      let ix ← iadd lane (← c32 (32 * i))
      acc ← fma (← loadF32 (← at4 xr ix)) (← loadF32 (← at4 wr ix)) acc
    for m in [16, 8, 4, 2, 1] do
      let o ← callLocal shflBfly %[← bitcast .i32 acc, ← c32 m]
      acc ← fadd acc (← bitcast .f32 o)
    when .eq lane (← c32 0) do
      store (← fadd acc (← loadF32 (← at4 bias e))) (← at4 out g)

/-- **`x += y` over `n` floats** (`meta`: `n`), a thread a float. -/
def addRowsBody : Prog V L Unit := do
  let ps ← entryParams (idParams ++ List.replicate 3 .i64)
  let x := (ps.drop 4).head; let y := (ps.drop 5).head; let mt := (ps.drop 6).head
  let i ← iadd (← imul ps.head (← c32 256)) ps.snd
  when .ult i (← load32 mt) do
    let a ← at4 x i
    store (← fadd (← loadF32 a) (← loadF32 (← at4 y i))) a

end

def moeGateUp : Kernel :=
  { name := "moe_gate_up_swiglu", params := List.replicate 8 .i64,
    code := emit moeGateUpBody (idParams ++ List.replicate 8 .i64), smem := msmem (ksOf K) }

def quantRows : Kernel :=
  { name := "quant_rows_int8", params := List.replicate 4 .i64,
    code := emit quantRowsBody (idParams ++ List.replicate 4 .i64) }

def moeDown : Kernel :=
  { name := "moe_down", params := List.replicate 7 .i64,
    code := emit moeDownBody (idParams ++ List.replicate 7 .i64), smem := msmem (ksOf K) }

def routerTopT : Kernel :=
  { name := "router_top4_tokens", params := List.replicate 4 .i64,
    code := emit routerTopTBody (idParams ++ List.replicate 4 .i64) }

def group : Kernel :=
  { name := "moe_group", params := List.replicate 5 .i64,
    code := emit groupBody (idParams ++ List.replicate 5 .i64), smem := 4 * NE }

def combine : Kernel :=
  { name := "moe_combine_tokens", params := List.replicate 5 .i64,
    code := emit combineBody (idParams ++ List.replicate 5 .i64) }

def q8Gemm (rows cols : Nat) : Kernel :=
  { name := "q8_gemm_bias", params := List.replicate 6 .i64,
    code := emit (q8GemmBody rows cols) (idParams ++ List.replicate 6 .i64), smem := msmem (ksOf cols) }

def rmsT : Kernel :=
  { name := "rms_rows", params := List.replicate 4 .i64,
    code := emit rmsTBody (idParams ++ List.replicate 4 .i64) }

def ropeT : Kernel :=
  { name := "rope_rows", params := List.replicate 3 .i64,
    code := emit ropeTBody (idParams ++ List.replicate 3 .i64) }

def kvNarrow : Kernel :=
  { name := "kv_narrow_rows", params := List.replicate 4 .i64,
    code := emit kvNarrowBody (idParams ++ List.replicate 4 .i64) }

def kvStoreT : Kernel :=
  { name := "kv_store_rows", params := List.replicate 5 .i64,
    code := emit kvStoreTBody (idParams ++ List.replicate 5 .i64) }

def attnT : Kernel :=
  { name := "attention_rows", params := List.replicate 8 .i64,
    code := emit attnTBody (idParams ++ List.replicate 8 .i64), smem := 2 * AT_KEYS * HD * 4 }

def routerLogits : Kernel :=
  { name := "router_logits_rows", params := List.replicate 5 .i64,
    code := emit routerLogitsBody (idParams ++ List.replicate 5 .i64) }

def addRows : Kernel :=
  { name := "add_rows", params := List.replicate 3 .i64,
    code := emit addRowsBody (idParams ++ List.replicate 3 .i64) }

end GptOssPrefill
