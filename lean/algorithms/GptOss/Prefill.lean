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
  exact integer sum, which the wide int8 `mma` (`Intr.mmaLo`, `Intr.mmaHi`)
  computes for sixteen rows and eight tokens a block at a time; it is scaled
  once a block by the weight's scale (halved, for MXFP4's doubling) and the
  activation's.

  ## A layer

  `GptOssDecode.dPrefillChunkM` runs a chunk of a prompt through these: the
  norm, the Q8 projections, the rotation, attention over the cache and the
  chunk (on the bf16 tensor cores, `Intr.mmaBfLo`, `Intr.mmaBfHi`), the router, the experts grouped by expert (one launch each for all
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
    a magnitude's value from the table `0 1 2 3 4 6 8 12` and from its
    negation, by one `prmt` each, then byte `k` from the one its sign picks. -/
def unpack4 (cw : V .i32) : Prog V L (V .i32) := do
  let sel ← band cw (← c32 0x7777)
  let pos : V .i32 ← callLocal prmtR %[← c32 0x03020100, ← c32 0x0C080604, sel]
  let neg : V .i32 ← callLocal prmtR %[← c32 0xFDFEFF00, ← c32 0xF4F8FAFC, sel]
  let pick ← bor (← ushrImm (← band cw (← c32 0x8888)) 1) (← c32 0x3210)
  callLocal prmtR %[pos, neg, pick]

/-- Words a tile row of the `mma` core occupies in shared memory: its eight,
    word `w` at `2 · (w mod 4) + w / 4`, so the two a lane passes a block's two
    `mma`s (words `t` and `t + 4`) are one 64-bit load, and the four rows a
    half-warp loads fall on distinct banks. -/
def MSTRIDE : Nat := 8

/-- How a weight matrix holds its values: MXFP4 (two codes a byte, an e8m0
    scale a block) or Q8_0 (an int8 a value, an f32 scale a block). -/
inductive Fmt | mx | q8
  deriving DecidableEq

/-- Blocks of thirty-two columns a stage takes between barriers, for a width
    of `k` columns: three where they divide its blocks, else two. -/
def ksOf (k : Nat) : Nat := if (k / 32) % 3 == 0 then 3 else 2

/-- Tokens a block's tile takes where each warp takes `nj` groups of eight. -/
def TTn (nj : Nat) : Nat := 16 * nj

/-- A stage's tiles, `ks` of each kind: weights, tokens, their scales. -/
def WTILE : Nat := TR * MSTRIDE * 4
def XTILE (nj : Nat) : Nat := TTn nj * MSTRIDE * 4
def M_W : Nat := 0
def M_X (ks : Nat) : Nat := M_W + ks * WTILE
def M_SW (ks nj : Nat) : Nat := M_X ks + ks * XTILE nj
def M_SX (ks nj : Nat) : Nat := M_SW ks nj + ks * TR * 4
def msmem (ks nj : Nat) : Nat := M_SX ks nj + ks * TTn nj * 4

/-- **The tile's sums by int8 `mma`**: for tile rows `r < TR` and tile tokens
    `t < TT`, `Σ_c W[wRow r][c] · x[tokRow t][c]` over `k` columns, `W` in
    format `fmt` (its values at `wb`, its scales at `ws`, a row each), `x` as
    int8 (`xq`) with a float scale a block (`xsc`), the tile `TR` rows by
    `TTn nj` tokens. Warp `w` takes rows `32 · (w mod 4) …` and tokens
    `8 nj · (w / 4) …`; each wide `mma` is sixteen
    of its rows against eight of its tokens, a whole block, lane `l` passing
    its rows `l / 4` and `l / 4 + 8`'s words and token `l / 4`'s, and receiving
    row `l / 4` (`mmaLo`) or eight past (`mmaHi`) at tokens `2 · (l mod 4)` and
    the next. Thread `tid`'s sum `2 nj · i + j` is tile row
    `32 · (w mod 4) + 8 i + (l / 4)` and tile token
    `8 nj · (w / 4) + 8 · (j / 2) + 2 · (l mod 4) + j mod 2` (`mmaAt`). `tokRow`
    answers a row in range for every tile token. -/
def mmaCore (fmt : Fmt) (nj : Nat) (k : Nat) (tid : V .i32) (sm wb ws xq xsc : V .i64) (wRow tokRow : V .i32 → Prog V L (V .i32)) :
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
  let TT := TTn nj
  -- a thread's share of a block's token tile: a word a 32 tokens
  let xRows ← (List.range (TT / 32)).mapM fun rep => do
    let q ← iadd tid (← c32 (256 * rep))
    pure (q, ← tokRow (← ushrImm q 3))
  let sRow ← wRow (← band tid (← c32 (TR - 1)))
  let sTok ← tokRow (← band tid (← c32 (TT - 1)))
  -- the bytes into a stage of the words this lane reads: its rows' and its
  -- tokens' fragments (its two words of each), its rows' and tokens' scales;
  -- offsets, so that each address is formed from `sm` where it is read
  let tw ← ishlImm t 3
  let aOff ← (List.range 4).mapM fun i => do
    iadd (← ishlImm (← imul (← iadd (← iadd (← ishlImm wr 5) (← c32 (8 * i))) g) (← c32 MSTRIDE)) 2) tw
  let bOff ← (List.range nj).mapM fun j => do
    iadd (← ishlImm (← imul (← iadd (← iadd (← imul wc (← c32 (8 * nj))) (← c32 (8 * j))) g) (← c32 MSTRIDE)) 2) tw
  let sOff ← ishlImm (← iadd (← ishlImm wr 5) (← ishlImm g 2)) 2
  let xOff ← ishlImm (← iadd (← imul wc (← c32 (8 * nj))) (← imul t (← c32 (2 * nj)))) 2
  -- `KS` blocks a stage: a thread's code words (eight bytes of codes each,
  -- four a block's row), token words and the raw words of their scales,
  -- loaded a stage ahead of the one the warps multiply
  let urem' (q : V .i32) (n : Nat) : Prog V L (V .i32) := do band q (← c32 (n - 1))
  -- where a row's word `w` sits
  let wpos (w : V .i32) : Prog V L (V .i32) := do
    iadd (← ishlImm (← band w (← c32 3)) 1) (← ushrImm w 2)
  let nC := cRows.length
  let nX := xRows.length
  let NP := KS * (nC + nX + 2)
  let fetch (b0 : V .i32) : Prog V L (List (V .i32)) := do
    let mut ws' : List (V .i32) := []
    for kbi in List.range KS do
      let b ← iadd b0 (← c32 kbi)
      let rowBytes := if fmt == .mx then k / 2 else k
      let blkBytes := if fmt == .mx then 16 else 32
      let codes ← cRows.mapM fun (q, row) => do
        load32 (← iadd wb (← ext (← iadd (← imul row (← c32 rowBytes))
          (← iadd (← imul b (← c32 blkBytes)) (← ishlImm (← urem' q perRow) 2)))))
      let toks ← xRows.mapM fun (q, xRow) => do
        load32 (← iadd xq (← ext (← iadd (← imul xRow (← c32 k))
          (← iadd (← imul b (← c32 32)) (← ishlImm (← band q (← c32 7)) 2)))))
      let sw ← if fmt == .mx then
          GptOssKernels.Body.u8 (← iadd ws (← ext (← iadd (← imul sRow (← c32 kb)) b)))
        else load32 (← at4 ws (← iadd (← imul sRow (← c32 kb)) b))
      let f ← load32 (← at4 xsc (← iadd (← imul sTok (← c32 kb)) b))
      ws' := ws' ++ codes ++ toks ++ [sw, f]
    pure ws'
  let stash (raw : List (V .i32)) : Prog V L Unit := do
    for kbi in List.range KS do
      let part := (raw.drop (kbi * (nC + nX + 2))).take (nC + nX + 2)
      let zi ← c32 0
      let codes := part.take nC
      let toks := (part.drop nC).take nX
      let sw ← if fmt == .mx then do
          fmul (← bitcast .f32 (← ishlImm (part.getD (nC + nX) zi) 23)) (← fconst .f32 0x3F000000)
        else bitcast .f32 (part.getD (nC + nX) zi)
      let f ← bitcast .f32 (part.getD (nC + nX + 1) zi)
      let wTile ← at_ sm (M_W + kbi * WTILE)
      for (code, (q, _)) in codes.zip cRows do
        if fmt == .mx then
          -- words `2c` and `2c + 1` of the row, `c = q mod 4`
          let c ← band q (← c32 3)
          let pos ← iadd (← ishlImm (← band c (← c32 1)) 2) (← ushrImm c 1)
          let dst ← at4 wTile (← iadd (← imul (← ushrImm q 2) (← c32 MSTRIDE)) pos)
          storeI32 (← unpack4 code) dst
          storeI32 (← unpack4 (← ushrImm code 16)) (← at_ dst 8)
        else
          storeI32 code (← at4 wTile (← iadd (← imul (← ushrImm q 3) (← c32 MSTRIDE)) (← wpos (← band q (← c32 7)))))
      for (tok, (q, _)) in toks.zip xRows do
        storeI32 tok (← at4 (← at_ sm (M_X KS + kbi * XTILE nj))
          (← iadd (← imul (← ushrImm q 3) (← c32 MSTRIDE)) (← wpos (← band q (← c32 7)))))
      -- scales lane by lane: row `32 a + 8 i + g` at `32 a + 4 g + i`, token
      -- `8 nj a + 8 j + 2 t + e` at `8 nj a + 2 nj t + 2 j + e`
      when .ult tid (← c32 TR) do
        let at' ← bor (← band tid (← c32 (TR - 32)))
          (← bor (← ishlImm (← band tid (← c32 7)) 2) (← band (← ushrImm tid 3) (← c32 3)))
        store sw (← at4 (← at_ sm (M_SW KS nj + 4 * TR * kbi)) at')
      let tt ← isub tid (← c32 TR)
      when .ult tt (← c32 TT) do
        let u ← band tt (← c32 (8 * nj - 1))
        let at' ← iadd (← isub tt u)
          (← iadd (← imul (← band (← ushrImm u 1) (← c32 3)) (← c32 (2 * nj)))
            (← iadd (← ishlImm (← ushrImm u 3) 1) (← band u (← c32 1))))
        store f (← at4 (← at_ sm (M_SX KS nj + 4 * TT * kbi)) at')
  -- the loop carries the stage, the sums' bits and the next stage's words
  let zbits ← bitcast .i32 zero
  let NA := 8 * nj
  let init := [← c32 0] ++ List.replicate NA zbits ++ (← fetch (← c32 0))
  let st ← wloop (Vals.ofFn (n := NP + NA + 1) fun q => init.getD q.val zbits)
    (fun cs => do pure (contIfULt cs.head (← c32 kb), cs, ()))
    (fun cs _ => do
      let all := Vals.uniformToList cs
      let b := all.getD 0 zbits
      -- the next stage's words, or this one's again past the last, asked for
      -- before the barrier so that they arrive while this stage multiplies
      let nb ← iadd b (← c32 KS)
      let pre ← fetch (← select (← icmp .ult nb (← c32 kb)) nb b)
      stash (all.drop (NA + 1))
      callLocalVoid barrier %[]
      let zi ← c32 0
      let mut accs ← ((all.drop 1).take NA).mapM fun x => bitcast .f32 x
      for kbi in List.range KS do
        let halves (x : V .i64) : Prog V L (V .i32 × V .i32) := do
          pure (← ireduce32 x, ← ireduce32 (← ushrImm x 32))
        let word2 (o : V .i32) (at' : Nat) : Prog V L (V .i64) := do
          load64 (← at_ (← iadd sm (← ext o)) at')
        let floats (o : V .i32) (at' n : Nat) : Prog V L (List (V .f32)) := do
          let ws ← (List.range (n / 2)).mapM fun p => do halves (← word2 o (at' + 8 * p))
          (ws.flatMap fun (x, y) => [x, y]).mapM fun x => bitcast .f32 x
        -- a row's two words of the block (`t`, `t + 4`), rows `g` and `g + 8`
        -- of each sixteen; a token's two, tokens `g` of each eight
        let as ← aOff.mapM fun o => do halves (← word2 o (M_W + kbi * WTILE))
        let bs ← bOff.mapM fun o => do halves (← word2 o (M_X KS + kbi * XTILE nj))
        let sws ← floats sOff (M_SW KS nj + 4 * TR * kbi) 4
        let sxs ← floats xOff (M_SX KS nj + 4 * TT * kbi) (2 * nj)
        let zp : V .i32 × V .i32 := (zi, zi)
        let mut next : List (V .f32) := List.replicate NA zero
        for i in List.range 2 do
          let (r0, r2) := as.getD (2 * i) zp
          let (r1, r3) := as.getD (2 * i + 1) zp
          for j in List.range nj do
            let (q0, q1) := bs.getD j zp
            for h in List.range 2 do
              let pr : V .i64 ← callLocal (if h == 0 then mmaLoR else mmaHiR) %[r0, r1, r2, r3, q0, q1]
              let (s0, s1) ← halves pr
              for (sum, e) in [(s0, 0), (s1, 1)] do
                let idx := 2 * nj * (2 * i + h) + (2 * j + e)
                let sc ← fmul (sws.getD (2 * i + h) zero) (sxs.getD (2 * j + e) zero)
                next := next.set idx (← fma (← fcvtFromSint .f32 sum) sc (accs.getD idx zero))
        accs := next
      callLocalVoid barrier %[]
      let bits ← accs.mapM fun x => bitcast .i32 x
      let out := [nb] ++ bits ++ pre
      pure (Vals.ofFn (n := NP + NA + 1) fun q => out.getD q.val zbits))
  ((Vals.uniformToList st).drop 1 |>.take NA).mapM fun x => bitcast .f32 x

/-- Thread `tid`'s sum `2 nj · i + j` of `mmaCore`: its tile row and tile
    token. -/
def mmaAt (nj : Nat) (tid : V .i32) (i j : Nat) : Prog V L (V .i32 × V .i32) := do
  let lane ← band tid (← c32 31)
  let w ← ushrImm tid 5
  let row ← iadd (← iadd (← ishlImm (← band w (← c32 3)) 5) (← c32 (8 * i))) (← ushrImm lane 2)
  let tok ← iadd (← iadd (← imul (← ushrImm w 2) (← c32 (8 * nj))) (← c32 (8 * (j / 2) + j % 2)))
    (← ishlImm (← band lane (← c32 3)) 1)
  pure (row, tok)

/-- Token groups of eight a warp of a Q8 projection over `cols` columns
    takes: two (a 32-token tile) where a stage takes three blocks, four (64
    tokens) where it takes two, so that a barrier covers as much work. -/
def q8nj (cols : Nat) : Nat := if ksOf cols == 3 then 2 else 4

/-- **Q8_0 rows against int8 tokens, plus a bias**: `out[t][r] = bias[r] +
    Σ_c W[r][c] · x[t][c]` for the `rows × cols` matrix `w` (its int8 values
    row by row, then its f32 scales, as `prepare_q8.py` lays one out) and `T`
    tokens (`meta`: `T`), `x` as int8 a token with one float scale a block.
    Block `cta` takes token tile `cta mod ⌈T / TTn (q8nj cols)⌉` and row tile
    `cta / ⌈T / TTn (q8nj cols)⌉`; a row past `rows` or a token past `T` is
    computed (on the last row or token) and not written. -/
def q8GemmBody (rows cols : Nat) : Prog V L Unit := do
  let ps ← entryParams (idParams ++ List.replicate 6 .i64)
  let cta := ps.head; let tid := ps.snd; let sm := (ps.drop 3).head
  let w := (ps.drop 4).head; let xq := (ps.drop 5).head; let xsc := (ps.drop 6).head
  let out := (ps.drop 7).head; let bias := (ps.drop 8).head; let mt := (ps.drop 9).head
  let ws ← iaddImm w (rows * cols)
  let T ← load32 mt
  let nj := q8nj cols
  let TTq := TTn nj
  let nTT ← udiv (← iadd T (← c32 (TTq - 1))) (← c32 TTq)
  let rt ← udiv cta nTT
  let tt ← isub cta (← imul rt nTT)
  let row0 ← imul rt (← c32 TR)
  let tok0 ← imul tt (← c32 TTq)
  let wRow (r : V .i32) : Prog V L (V .i32) := do
    let row ← iadd row0 r
    select (← icmp .ult row (← c32 rows)) row (← c32 (rows - 1))
  let tokRow (t : V .i32) : Prog V L (V .i32) := do
    let tok ← iadd tok0 t
    select (← icmp .ult tok T) tok (← isub T (← c32 1))
  let accs ← mmaCore .q8 nj cols tid sm w ws xq xsc wRow tokRow
  let zero ← fconst .f32 0
  for i in List.range 4 do
    let (r, _) ← mmaAt nj tid i 0
    let row ← iadd row0 r
    when .ult row (← c32 rows) do
      let b ← loadF32 (← at4 bias row)
      for j in List.range (2 * nj) do
        let (_, t) ← mmaAt nj tid i j
        let tok ← iadd tok0 t
        when .ult tok T do
          store (← fadd (accs.getD (2 * nj * i + j) zero) b) (← at4 out (← iadd (← imul tok (← c32 rows)) row))

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
    let accs ← mmaCore .mx 2 K tid sm wb ws xq xsc wRow tokRow
    let zero ← fconst .f32 0
    let f (b : UInt32) : Prog V L (V .f32) := fconst .f32 b.toUInt64
    for i in List.range 2 do
      let (r, _) ← mmaAt 2 tid i 0
      let g ← wRow r
      let bg ← loadF32 (← at4 bias g)
      let bu ← loadF32 (← at4 bias (← iadd g (← c32 I)))
      for j in List.range 4 do
        let (_, tk) ← mmaAt 2 tid i j
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
    let accs ← mmaCore .mx 2 K tid sm wb ws hq hs wRow tokRow
    let zero ← fconst .f32 0
    for i in List.range 4 do
      let (r, _) ← mmaAt 2 tid i 0
      let row ← iadd row0 r
      when .ult row (← c32 H) do
        let b ← loadF32 (← at4 bias row)
        for j in List.range 4 do
          let (_, tk) ← mmaAt 2 tid i j
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
    words a head, as `attnFBody` reads it too), token
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

/-- Words a staged key row takes in shared memory (its 32, and 4 so that the
    eight keys a fragment load reads fall on distinct banks), and a staged
    dimension row of values (its 16 key pairs, and 4). -/
def KSTR : Nat := 36
def VSTR : Nat := 20

set_option maxRecDepth 8000 in
/-- **Attention of every token, on the bf16 tensor cores**: causal, a key
    window where the layer slides, the learned sink in the denominator
    (`meta`: `T`, `start`, `window` (0 for none), `cap`, `stride`). Token `t`
    at position `p = start + t` sees the keys at `max(0, p - window + 1) … p`,
    scores `q · k / 8`; a key before `start` is the cache's (slot `j mod cap`),
    one at or after it the chunk's (`kc`, `vc`, as `kvNarrowBody` writes them).
    A block of 256 threads takes key head `kh = cta mod NKV` and tokens
    `AT_TOK · (cta / NKV) …`. Warp `w` takes sixteen rows, the eight query
    heads of key head `kh` for tokens `2 w` (rows `0 … 7`) and `2 w + 1` (rows
    `8 … 15`); lane `l` holds rows `l / 4` and `l / 4 + 8`. The queries,
    scaled by 1/8 and rounded to bf16, stay in the lanes; each tile of
    `AT_KEYS` keys is staged as bf16 (keys row by row, values dimension by
    dimension), its scores are the bf16 `mma`'s, the softmax is online in the
    lanes (a row's four lanes combining by butterfly), and the probabilities,
    rounded to bf16, multiply the values by the bf16 `mma` again. -/
def attnFBody : Prog V L Unit := do
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
  let lane ← band tid (← c32 31)
  let w ← ushrImm tid 5
  let g ← ushrImm lane 2
  let t ← band lane (← c32 3)
  let head ← iadd (← imul kh (← c32 8)) g
  let tokA ← iadd t0 (← ishlImm w 1)
  let tokB ← iadd tokA (← c32 1)
  let clampT (x : V .i32) : Prog V L (V .i32) := do
    select (← icmp .ult x T) x (← isub T (← c32 1))
  let tA ← clampT tokA
  let tB ← clampT tokB
  let pA ← iadd start tA
  let pB ← iadd start tB
  let loOf (p : V .i32) : Prog V L (V .i32) := do
    select (← icmp .ne window (← c32 0))
      (← select (← icmp .uge p window) (← isub (← iadd p (← c32 1)) window) (← c32 0)) (← c32 0)
  let loA ← loOf pA
  let loB ← loOf pB
  -- the block's key range: from its first token's window to its last token
  let tLast ← isub (← select (← icmp .ult (← iadd t0 (← c32 AT_TOK)) T) (← iadd t0 (← c32 AT_TOK)) T) (← c32 1)
  let lo0 ← loOf (← iadd start t0)
  let hi0 ← iadd (← iadd start tLast) (← c32 1)
  -- f32 to bf16, rounding to nearest even, and two into a word
  let bfOf (x : V .f32) : Prog V L (V .i32) := do
    let b ← bitcast .i32 x
    ushrImm (← iadd (← iadd b (← c32 0x7FFF)) (← band (← ushrImm b 16) (← c32 1))) 16
  let pack (x y : V .f32) : Prog V L (V .i32) := do bor (← bfOf x) (← ishlImm (← bfOf y) 16)
  -- the queries' fragments: for each sixteen dimensions, rows `g` (token A)
  -- and `g + 8` (token B) at dimensions `2 t, 2 t + 1` and eight past
  let eighth ← fl 0.125
  let qA ← at4 qkv (← iadd (← imul tA (← c32 QKVW)) (← imul head (← c32 HD)))
  let qB ← at4 qkv (← iadd (← imul tB (← c32 QKVW)) (← imul head (← c32 HD)))
  let tw ← ishlImm t 1
  let qpair (row : V .i64) (d0 : Nat) : Prog V L (V .i32) := do
    let a ← at4 row (← iadd (← c32 d0) tw)
    pack (← fmul (← loadF32 a) eighth) (← fmul (← loadF32 (← at_ a 4)) eighth)
  let qf ← (List.range (HD / 16)).mapM fun st => do
    pure [← qpair qA (16 * st), ← qpair qB (16 * st), ← qpair qA (16 * st + 8), ← qpair qB (16 * st + 8)]
  let zero ← fconst .f32 0
  let one ← fconst .f32 0x3F800000
  let ninf ← fconst .f32 0xFF800000
  let z64 ← iconst64 0
  let lo32 (x : V .i64) : Prog V L (V .f32) := do bitcast .f32 (← ireduce32 x)
  let hi32 (x : V .i64) : Prog V L (V .f32) := do bitcast .f32 (← ireduce32 (← ushrImm x 32))
  let join (a b : V .f32) : Prog V L (V .i64) := do
    bor (← uextend64 (← bitcast .i32 a)) (← ishlImm (← uextend64 (← bitcast .i32 b)) 32)
  let quad (x : V .f32) (f : V .f32 → V .f32 → Prog V L (V .f32)) : Prog V L (V .f32) := do
    let x1 ← f x (← bitcast .f32 (← callLocal shflBfly %[← bitcast .i32 x, ← c32 1]))
    f x1 (← bitcast .f32 (← callLocal shflBfly %[← bitcast .i32 x1, ← c32 2]))
  let nTiles ← udiv (← iadd (← isub hi0 lo0) (← c32 (AT_KEYS - 1))) (← c32 AT_KEYS)
  let st ← wloop (Vals.append %[← c32 0, ninf, ninf, zero, zero] (Vals.ofFn (n := 16) fun _ => z64))
    (fun cs => do pure (contIfULt cs.head nTiles, cs, ()))
    (fun cs _ => do
      let mA := (cs.drop 1).head; let mB := (cs.drop 2).head
      let lA := (cs.drop 3).head; let lB := (cs.drop 4).head
      let os := Vals.uniformToList (cs.drop 5 : Vals V (List.replicate 16 .i64))
      let j0 ← iadd lo0 (← imul cs.head (← c32 AT_KEYS))
      -- a key's word `wd` (its dimensions `2 wd`, `2 wd + 1`), from the cache
      -- or the chunk
      -- (a cache that wraps is the sliding layers' ring, of `CAP_SWA` slots, a
      -- power of two; a full layer's holds every position)
      let kvWord (cache chunk : V .i64) (j wd : V .i32) : Prog V L (V .i32) := do
        let jc ← select (← icmp .ult j hi0) j lo0
        let fromCache ← icmp .ult jc start
        let slot ← select (← icmp .ult jc cap) jc (← band jc (← isub cap (← c32 1)))
        let cIx ← iadd (← imul kh stride) (← iadd (← imul slot (← c32 (HD / 2))) wd)
        let chunkT ← select fromCache (← c32 0) (← isub jc start)
        let kIx ← iadd (← imul (← iadd (← imul chunkT (← c32 NKV)) kh) (← c32 (HD / 2))) wd
        load32 (← select fromCache (← at4 cache cIx) (← at4 chunk kIx))
      -- the keys, row by row
      for rep in List.range (AT_KEYS * HD / 2 / 256) do
        let wi ← iadd tid (← c32 (256 * rep))
        let kj ← ushrImm wi 5
        let wd ← band wi (← c32 31)
        let wv ← kvWord kcache kc (← iadd j0 kj) wd
        storeI32 wv (← at4 sm (← iadd (← imul kj (← c32 KSTR)) wd))
      -- the values, dimension by dimension: two keys' words, transposed
      for rep in List.range (AT_KEYS / 2 * HD / 2 / 256) do
        let ci ← iadd tid (← c32 (256 * rep))
        let m ← ushrImm ci 5
        let dp ← band ci (← c32 31)
        let j ← iadd j0 (← ishlImm m 1)
        let w0 ← kvWord vcache vc j dp
        let w1 ← kvWord vcache vc (← iadd j (← c32 1)) dp
        let lo ← bor (← band w0 (← c32 0xFFFF)) (← ishlImm w1 16)
        let hi ← bor (← ushrImm w0 16) (← band w1 (← c32 0xFFFF0000))
        let at0 ← iadd (← c32 (AT_KEYS * KSTR)) (← iadd (← imul (← ishlImm dp 1) (← c32 VSTR)) m)
        storeI32 lo (← at4 sm at0)
        storeI32 hi (← at4 sm (← iadd at0 (← c32 VSTR)))
      callLocalVoid barrier %[]
      -- the scores, eight keys a group, sixteen dimensions an `mma`
      let mut sA : List (V .f32) := []
      let mut sB : List (V .f32) := []
      for kg in List.range (AT_KEYS / 8) do
        let mut accA := z64
        let mut accB := z64
        for stp in List.range (HD / 16) do
          let a := qf.getD stp []
          let kRow ← imul (← iadd (← c32 (8 * kg)) g) (← c32 KSTR)
          let b0 ← load32 (← at4 sm (← iadd kRow (← iadd (← c32 (8 * stp)) t)))
          let b1 ← load32 (← at4 sm (← iadd kRow (← iadd (← c32 (8 * stp + 4)) t)))
          let z ← c32 0
          let (q0, q1, q2, q3) := (a.getD 0 z, a.getD 1 z, a.getD 2 z, a.getD 3 z)
          accA ← callLocal mmaBfLoR %[q0, q1, q2, q3, b0, b1, accA]
          accB ← callLocal mmaBfHiR %[q0, q1, q2, q3, b0, b1, accB]
        sA := sA ++ [← lo32 accA, ← hi32 accA]
        sB := sB ++ [← lo32 accB, ← hi32 accB]
      -- the keys each row sees, the rest `-inf`
      let seen (p lo : V .i32) (i : Nat) : Prog V L (V .i32) := do
        let j ← iadd j0 (← iadd (← c32 (8 * (i / 2) + i % 2)) tw)
        band (← band (← select (← icmp .ult j hi0) (← c32 1) (← c32 0))
          (← select (← icmp .uge j lo) (← c32 1) (← c32 0))) (← select (← icmp .ule j p) (← c32 1) (← c32 0))
      let maskRow (sv : List (V .f32)) (p lo : V .i32) : Prog V L (List (V .f32) × List (V .i32)) := do
        let vs ← (List.range sv.length).mapM fun i => seen p lo i
        let ms ← (sv.zip vs).mapM fun (x, v) => do select (← icmp .ne v (← c32 0)) x ninf
        pure (ms, vs)
      let (sA', vA) ← maskRow sA pA loA
      let (sB', vB) ← maskRow sB pB loB
      -- the online softmax of a row: its new maximum, the old sum's factor,
      -- the tile's probabilities and the new sum
      let soft (sv : List (V .f32)) (vs : List (V .i32)) (m l : V .f32) :
          Prog V L (V .f32 × V .f32 × List (V .f32) × V .f32) := do
        let mut tmax := ninf
        for x in sv do tmax ← fmax tmax x
        let rmax ← quad tmax fmax
        let mNew ← fmax m rmax
        let dead ← fcmp .eq mNew ninf
        let alive ← fcmp .ne mNew ninf
        let corr ← select dead one (← expA (← fsub m mNew))
        let pr ← (sv.zip vs).mapM fun (x, v) => do
          select (← band (← icmp .ne v (← c32 0)) alive) (← expA (← fsub x mNew)) zero
        let mut tsum := zero
        for x in pr do tsum ← fadd tsum x
        let rsum ← quad tsum fadd
        pure (mNew, corr, pr, ← fma l corr rsum)
      let (mA', corrA, prA, lA') ← soft sA' vA mA lA
      let (mB', corrB, prB, lB') ← soft sB' vB mB lB
      -- the sums so far, scaled by their factor, then the tile's values
      let mut os' : List (V .i64) := []
      for (o, idx) in os.zipIdx do
        let corr := if idx % 2 == 0 then corrA else corrB
        os' := os' ++ [← join (← fmul (← lo32 o) corr) (← fmul (← hi32 o) corr)]
      for c in List.range (AT_KEYS / 16) do
        let a0 ← pack (prA.getD (4 * c) zero) (prA.getD (4 * c + 1) zero)
        let a1 ← pack (prB.getD (4 * c) zero) (prB.getD (4 * c + 1) zero)
        let a2 ← pack (prA.getD (4 * c + 2) zero) (prA.getD (4 * c + 3) zero)
        let a3 ← pack (prB.getD (4 * c + 2) zero) (prB.getD (4 * c + 3) zero)
        let mut nx : List (V .i64) := []
        for ng in List.range (HD / 8) do
          let vRow ← iadd (← c32 (AT_KEYS * KSTR)) (← imul (← iadd (← c32 (8 * ng)) g) (← c32 VSTR))
          let b0 ← load32 (← at4 sm (← iadd vRow (← iadd (← c32 (8 * c)) t)))
          let b1 ← load32 (← at4 sm (← iadd vRow (← iadd (← c32 (8 * c + 4)) t)))
          let oA : V .i64 ← callLocal mmaBfLoR %[a0, a1, a2, a3, b0, b1, os'.getD (2 * ng) z64]
          let oB : V .i64 ← callLocal mmaBfHiR %[a0, a1, a2, a3, b0, b1, os'.getD (2 * ng + 1) z64]
          nx := nx ++ [oA, oB]
        os' := nx
      callLocalVoid barrier %[]
      pure (Vals.append %[← iadd cs.head (← c32 1), mA', mB', lA', lB']
        (Vals.ofFn (n := 16) fun q => os'.getD q.val z64)))
  -- the sink, and each row's dimensions `8 ng + 2 t`, `+ 1` out
  let sink ← loadF32 (← at4 sinks head)
  let os := Vals.uniformToList (st.drop 5 : Vals V (List.replicate 16 .i64))
  for (tok, m, l, half) in [(tokA, (st.drop 1).head, (st.drop 3).head, 0),
      (tokB, (st.drop 2).head, (st.drop 4).head, 1)] do
    when .ult tok T do
      let mAll ← fmax m sink
      let wgt ← expA (← fsub m mAll)
      let lAll ← fadd (← fmul l wgt) (← expA (← fsub sink mAll))
      let inv : V .f32 ← callLocal rcpApprox %[lAll]
      let scale ← fmul wgt inv
      let orow ← at4 out (← iadd (← imul tok (← c32 QW)) (← imul head (← c32 HD)))
      for ng in List.range (HD / 8) do
        let o := os.getD (2 * ng + half) z64
        let at' ← at4 orow (← iadd (← c32 (8 * ng)) tw)
        store (← fmul (← lo32 o) scale) at'
        store (← fmul (← hi32 o) scale) (← at_ at' 4)

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

/-- The `mma` kernels' launch bounds: blocks of 256 threads, two to a
    multiprocessor, so their registers stay within 128 a thread. -/
def MMA_BOUNDS : Option (Nat × Nat) := some (256, 2)

def moeGateUp : Kernel :=
  { name := "moe_gate_up_swiglu", params := List.replicate 8 .i64,
    code := emit moeGateUpBody (idParams ++ List.replicate 8 .i64), smem := msmem (ksOf K) 2, bounds := MMA_BOUNDS }

def quantRows : Kernel :=
  { name := "quant_rows_int8", params := List.replicate 4 .i64,
    code := emit quantRowsBody (idParams ++ List.replicate 4 .i64) }

def moeDown : Kernel :=
  { name := "moe_down", params := List.replicate 7 .i64,
    code := emit moeDownBody (idParams ++ List.replicate 7 .i64), smem := msmem (ksOf K) 2, bounds := MMA_BOUNDS }

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
    code := emit (q8GemmBody rows cols) (idParams ++ List.replicate 6 .i64), smem := msmem (ksOf cols) (q8nj cols),
    bounds := some (256, if q8nj cols == 2 then 2 else 1) }

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

def attnF : Kernel :=
  { name := "attention_rows_bf16", params := List.replicate 8 .i64,
    code := emit attnFBody (idParams ++ List.replicate 8 .i64), smem := (AT_KEYS * KSTR + HD * VSTR) * 4,
    bounds := some (256, 2) }

def routerLogits : Kernel :=
  { name := "router_logits_rows", params := List.replicate 5 .i64,
    code := emit routerLogitsBody (idParams ++ List.replicate 5 .i64) }

def addRows : Kernel :=
  { name := "add_rows", params := List.replicate 3 .i64,
    code := emit addRowsBody (idParams ++ List.replicate 3 .i64) }

end GptOssPrefill
