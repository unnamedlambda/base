import AlgorithmLib.Layout
import AlgorithmLib.PTX
import AlgorithmLib.ML.Bind
import AlgorithmLib.ML.Ptx
import AlgorithmLib.ML.PtxPrint
import AlgorithmLib.ML.Fuse
import AlgorithmLib.ML.Machine
import AlgorithmLib.ML.Compose
import AlgorithmLib.ML.Kernels
import AlgorithmLib.ML.Frontend
import AlgorithmLib.ML.Schedule
import AlgorithmLib.ML.LocalBind
import AlgorithmLib.ML.BufsOf
import AlgorithmLib.IR
open AlgorithmLib AlgorithmLib.ML

/-! DeiT-Tiny's twelve blocks as one `Ten` term, at the padded geometry. -/
namespace Vit

/-- **Query rows: the tokens, padded only to an even count.**

    A grid, so nothing forces it to a multiple of 32 — only `SK` below is
    reduced over.  It is even so that `vWarpsOf` can still pack two chunks per
    block, and no larger, because every launch in the step but the attention's
    two contractions is this many rows tall. -/
def SQ  : Nat := 200     -- 1 class token + 196 patches + 3 padding, even

/-- **Key rows: the same tokens, padded to a multiple of 32.**

    Softmax reduces *along* a row of scores, and a row pass walks a row of
    width `w` in `w/32` lane-strided trips, so the key count is a width and has
    to be 32-aligned where the query count does not. -/
def SK  : Nat := 224
def DM  : Nat := 192
def NH  : Nat := 3
def HD  : Nat := 64
def DFF : Nat := 768
def NC  : Nat := 128     -- 102 classes, padded to a multiple of 32
def EPS : Float32 := 0.000001

/-- Per-layer parameter block: 22 buffers, laid out in this order.

    Each of `wq`, `wk`, `wv` is **one** `DM x DM` matrix holding all three
    heads, not three `HD x DM` ones.  Their **biases stay per head**, because a
    head's bias is added while its share of the projection is being sliced out —
    one pass, reading the window and that head's own vector.  The bytes are
    the same bytes in the same order — three contiguous equal regions merged
    into one is a no-op on the host blob — but on the device it is one buffer,
    which is what lets the three per-head projections be one contraction three
    times taller.  Measured, that is 1.8-3.7x faster than issuing them
    separately and faster than batching them. -/
def pBase (i : Nat) : Nat := 9 + 22 * i

def addW : WFExp := .add (.reg 1) (.reg 2)
def mulW : WFExp := .mul (.reg 1) (.reg 2)
def add2 : Expr 2 := .add (.var ⟨0, by decide⟩) (.var ⟨1, by decide⟩)

/-- `1/c`, the scale a summed row needs to become an average. -/
def INV_DM : Float32 := 1.0 / 192.0

/-- `(x - mean)·rsqrt(var + eps)·gamma + beta`, from row passes only.

    The variance comes from `rowsq`, which sums a row's squares in one pass.
    Squaring into a tensor and reducing that is two passes over the whole
    tensor, and — the reason it matters — the elementwise square is the one step
    of a norm that is *not* one warp per row, so it is what stops the norm being
    a single kernel.  `rowsq` sums rather than averages, hence `invC`. -/
def layerNorm {v : Nat → Nat → Type} {r c : Nat} (invC : Float32)
    (gamma beta onesN : Ten v 1 c) (x : Ten v r c) : Ten v r c :=
  -- `x` is read twice below, so it must be bound: a `Ten` is a tree, and an
  -- unbound repeat re-emits the whole subterm rather than reusing its buffer.
  .letT x (fun xv =>
  .letT (.zipS (.add (.reg 1) (.neg (.reg 2))) (.var xv) (.rowB (.var xv) onesN)) (fun ctr =>
    let nrm : Ten v r c :=
      .zipS (.mul (.reg 1) (.rsqrt (.add (.mul (.reg 2) (.lit invC)) (.lit EPS))))
        (.var ctr) (.rowsq (.var ctr))
    .zipB addW (.zipB mulW nrm gamma) beta))

/-- Row-wise softmax with the exponential folded into the shift.

    `Ten.softmaxRow` shifts and then exponentiates as a separate elementwise
    pass; one row pass does both, and — as with the norm's square — that
    elementwise pass is the step that is not one warp per row. -/
def softmaxRowFused {v : Nat → Nat → Type} {r c : Nat}
    (ones : Ten v 1 c) (t : Ten v r c) : Ten v r c :=
  .letT t (fun tb =>
    .letT (Ten.rowMax AlgorithmLib.ML.softmaxFloor (.var tb)) (fun mx =>
      .letT (.zipS (.exp (.add (.reg 1) (.neg (.reg 2)))) (.var tb) (.var mx)) (fun e =>
        .zipS (.mul (.reg 1) (.inv (.reg 2))) (.var e) (Ten.rowB (.var e) ones))))

/-- tanh-GELU, `0.5x(1+tanh(c(x+0.044715x³)))`, as a row pass: `Expr` literals
    are `Nat` and these constants are not.  A declared approximation to the
    exact erf form, measured at 3.4e-3 relative on DeiT-Tiny's logits. -/
def geluW : WFExp :=
  let x := WFExp.reg 1
  let z := WFExp.mul (.lit (0.7978845608028654 : Float32))
             (.add x (.mul (.lit (0.044715 : Float32)) (.mul x (.mul x x))))
  -- `0.5(1+tanh z)` written as `e^z/(e^z + e^-z)`, which is the same function
  -- and the only arrangement of it that survives float32 in **both** tails.
  --
  -- Anything phrased in `e^2z` overflows for `z > 44`, and anything in `e^-2z`
  -- overflows for `z < -44`; DeiT's first block reaches ±12 pre-activation,
  -- which is `z` of ±73, so both happen.  Overflow is not itself the failure —
  -- the value survives it — but the derivative does not: `sderiv` of `inv a` is
  -- `-da·inv(a)²`, and at `a = inf` that is `inf · 0 = NaN`.
  --
  -- Here `e^±z` alone stays in range wherever `e^±2z` would not, so no
  -- intermediate is ever infinite and the derivative is finite throughout.  The
  -- arrangement holds while `|z| < 88`, i.e. pre-activations within about ±13.
  let a := WFExp.exp z
  let b := WFExp.exp (.neg z)
  .mul x (.mul a (.inv (.add a b)))

/-- **All heads' share of one projection, in one contraction.**

    `w` is the whole `DM x DM` matrix and `n1` the normed stream, so this is the
    contraction timm issues and not the three-times-narrower one per head. -/
def proj {v : Nat → Nat → Type} (w : Ten v DM DM) (n1 : Ten v SQ DM) : Ten v SQ DM :=
  .mv Backend.proven w n1

/-- The same projection, landing in a tensor as tall as the key padding.

    Only the keys and the values need it: they are indexed *along* a score row,
    so they must reach `SK`, while the `SQ - 197` rows past the real tokens are
    never written and read as the zeros the buffer was allocated with.  The mask
    is what makes those rows harmless — a padded key scores `-1e30`, so its
    softmax weight is exactly zero. -/
def projK {v : Nat → Nat → Type} (w : Ten v DM DM) (n1 : Ten v SQ DM) : Ten v SK DM :=
  .mvAt SK Backend.proven w n1

/-- **One head's share of a projection, with its bias.**

    A column window of a row-major buffer is strided and everything below takes
    whole pointers, so the split is a real pass — but it is *one* pass, reading
    the window at the projection's pitch and the bias at the head's own, and
    three of them cost far less than the two extra contractions they replace. -/
def slice {v : Nat → Nat → Type} {r : Nat}
    (t : Ten v r DM) (bias : Ten v 1 HD) (h : Nat) : Ten v r HD :=
  .cols (HD * h) addW t bias

/-- One head, over projections already sliced. -/
def head {v : Nat → Nat → Type} (onesSK mask : Ten v 1 SK)
    (qh : Ten v SQ HD) (kh vh : Ten v SK HD) : Ten v SQ HD :=
  .letT (.zipB addW
          (.ew1 (.mul (.var ⟨0, by decide⟩) (.rsqrt (.lit HD)))
            (.mv Backend.proven kh qh)) mask) (fun sc =>
    .mvT Backend.proven vh (softmaxRowFused onesSK (.var sc)))

def block {v : Nat → Nat → Type} (i : Nat)
    (onesN : Ten v 1 DM) (onesF : Ten v 1 DFF) (onesSK mask : Ten v 1 SK)
    (x : Ten v SQ DM) : Ten v SQ DM :=
  let p := pBase i
  .letT x (fun xb =>
    .letT (layerNorm INV_DM (.inp p) (.inp (p+1)) onesN (.var xb)) (fun n1 =>
      -- Each projection is bound once, for every head at once.  Without `letT`
      -- a `Ten` is a tree and the three heads would re-emit the contraction.
      .letT (proj (.inp (p+2)) (.var n1)) (fun q =>
      .letT (projK (.inp (p+6)) (.var n1)) (fun k =>
      .letT (projK (.inp (p+10)) (.var n1)) (fun val =>
      -- The nine slices are bound here rather than inside a head, so that the
      -- six that are `SK` rows tall are emitted together.  A launch joins the
      -- kernel beside it only at an equal grid, and a head's own order would
      -- alternate `SQ` and `SK` six times a block: same work, 240 fewer groups.
      .letT (slice (.var q) (.inp (p+3)) 0) (fun q0 =>
      .letT (slice (.var q) (.inp (p+4)) 1) (fun q1 =>
      .letT (slice (.var q) (.inp (p+5)) 2) (fun q2 =>
      .letT (slice (.var k) (.inp (p+7)) 0) (fun k0 =>
      .letT (slice (.var k) (.inp (p+8)) 1) (fun k1 =>
      .letT (slice (.var k) (.inp (p+9)) 2) (fun k2 =>
      .letT (slice (.var val) (.inp (p+11)) 0) (fun v0 =>
      .letT (slice (.var val) (.inp (p+12)) 1) (fun v1 =>
      .letT (slice (.var val) (.inp (p+13)) 2) (fun v2 =>
      -- heads 1.. are summed onto head 0; `List.range NH` here would count
      -- head 0 twice, and a `Ten` is a tree, so the duplicate re-emits.
      -- One contraction over the whole projection matrix, not three over its
      -- column blocks summed: `Σ_h Wo_h · hd_h = [Wo_0|Wo_1|Wo_2]·[hd_0;hd_1;hd_2]`.
      let hd := fun (qh : Ten v SQ HD) (kh vh : Ten v SK HD) =>
        head onesSK mask qh kh vh
      let heads : Ten v SQ DM :=
        Ten.cat3 (by decide) (hd (.var q0) (.var k0) (.var v0))
          (hd (.var q1) (.var k1) (.var v1)) (hd (.var q2) (.var k2) (.var v2)) mask
      let att := .mv Backend.proven (.inp (p+14)) heads
      .letT (.ew2 add2 (.var xb) (.zipB addW att (.inp (p+15)))) (fun xr =>
        .letT (layerNorm INV_DM (.inp (p+16)) (.inp (p+17)) onesN (.var xr)) (fun n2 =>
          .ew2 add2 (.var xr)
            (.zipB addW
              (.mv Backend.proven (.inp (p+20))
                (.zipB geluW (.zipB addW (.mv Backend.proven (.inp (p+18)) (.var n2))
                                (.inp (p+19))) onesF))
              (.inp (p+21)))))))))))))))))))

/-- Twelve blocks, a final LayerNorm, and the classifier on the class token. -/
def model (n : Nat) : TenProg SQ NC := fun v =>
  let onesN : Ten v 1 DM := .inp 0
  let onesF : Ten v 1 DFF := .inp 1
  let onesSK : Ten v 1 SK := .inp 2
  let mask : Ten v 1 SK := .inp 3
  let toks : Ten v SQ DM := .inp 4
  let x := (List.range n).foldl (fun a i => block i onesN onesF onesSK mask a) toks
  .zipB addW (.mv Backend.proven (.inp 6) (layerNorm INV_DM (.inp 5) (.inp 7) onesN x)) (.inp 8)

end Vit


-- ---------------------------------------------------------------------------
-- The artifact: one block, end to end
-- ---------------------------------------------------------------------------

namespace Vit
open AlgorithmLib.IR

/-- Structural equality for the emittable statement, so groups can be keyed on
    what they are rather than on the text they render to. -/
instance : Hashable Float32 := ⟨fun f => hash (toString f)⟩
deriving instance BEq, Hashable for AlgorithmLib.ML.IdxE
deriving instance BEq, Hashable for AlgorithmLib.ML.WFExp
deriving instance BEq, Hashable for AlgorithmLib.ML.EWStmt

/-- Blocks in this artifact.  One, so the whole pipeline can be checked against
    timm's block 0 before the geometry grows. -/
def NL : Nat := 12

/-- Inputs: nine globals, then thirty parameters per block. -/
def VBASE : Nat := 9 + 22 * NL

def perLayerBytes : List Nat :=
  [DM*4, DM*4]                                        -- norm1 weight, bias
  ++ [DM*DM*4] ++ List.replicate 3 (HD*4)             -- wq (all heads), bq per head
  ++ [DM*DM*4] ++ List.replicate 3 (HD*4)             -- wk, bk
  ++ [DM*DM*4] ++ List.replicate 3 (HD*4)             -- wv, bv
  ++ [DM*DM*4]                                        -- wo, all heads
  ++ [DM*4]                                           -- proj bias
  ++ [DM*4, DM*4]                                     -- norm2 weight, bias
  ++ [DFF*DM*4, DFF*4, DM*DFF*4, DM*4]                -- mlp

def vInBytes : List Nat :=
  [DM*4, DFF*4, SK*4, SK*4, SQ*DM*4, DM*4, NC*DM*4, DM*4, NC*4]
  ++ (List.range NL).flatMap (fun _ => perLayerBytes)

/-- Contractions go to cuBLAS; the row passes, softmax and GELU stay proven.
    `impl` is a schedule field, so this is a lowering choice and not an edit to
    the model — `TOp.retarget` cannot touch a denotation.

    `fuse` is empty because fusion does not happen here.  A schedule-level fusion
    would run before the backward is derived, and the shapes it emits —
    `ziprow3` and upward — are shapes `TOp.grad` has no reverse rule for, so the
    fused tape would not differentiate.  Fusing after the derivation has neither
    problem and reaches the backward too, which is where most of the traffic is:
    `vFuseRun` and `vFuseRun2` take 304 sites across the whole step. -/
def vSched : AlgorithmLib.ML.TenSchedule :=
  { impl := .vendor .cublasSgemm }

def vTape : List AlgorithmLib.ML.TOp := ((model NL).compile VBASE vSched).2
def vBufs : List (Ref × Nat) := vTape.map (AlgorithmLib.ML.TOp.outSize SQ)

/-- One past the last buffer the forward writes. -/
def VFWDBUF : Nat := vBufs.foldl (fun a p => max a (p.1 + 1)) VBASE

/-- The buffer the last forward operation writes: the logits. -/
def VOUT : Ref := (vBufs.getLast? |>.map Prod.fst).getD 0

/-- **What gets a gradient**: every parameter, and nothing that is a constant
    of the geometry.  Buffers 0-3 are the ones vectors and the padding mask, 4
    is the patch embedding the host supplies; 5 upward are trained. -/
def vNeeds (r : Ref) : Bool := decide (5 ≤ r)

/-- `dL/dlogits`, uploaded by the host.  The loss is cross-entropy on the class
    token, whose gradient is `softmax(logits) - onehot` — arithmetic on 102
    numbers, computed where the label lives rather than made into a kernel. -/
def VSEED : Ref := VFWDBUF

/-- **The backward pass, derived from the forward tape.**

    `ones` is buffer 1, the `DFF`-long vector: it must be at least as long as
    the widest row a broadcast adjoint reduces over, which is `DFF` and not
    `SQ`.  Nothing here is a written gradient formula — every adjoint comes
    from the forward operation it belongs to, and the activation derivatives
    from `Expr.sderiv`, which `grad_hasDerivAt` ties to the analytic one.

    The batch is `SQ`: a token is what this model contracts over in parallel,
    and every operation on the tape carries `SQ` as its own batch.  It sizes the
    grid of the adds that sum a value's contributions, so a value used once is
    right at any setting and only a fan-out reveals a wrong one — which is what
    `n1`, read by three projections in each of three heads, did. -/
def vBwd : List AlgorithmLib.ML.TOp :=
  (AlgorithmLib.ML.Ten.backwardFrom vNeeds 1 SQ (VFWDBUF + 1)
    [(VOUT, VSEED)] vTape true).getD []

/-- Where each buffer's gradient landed.  The reverse pass allocates as it
    goes, so this is read out of the derivation rather than assumed. -/
def vGradCoT : AlgorithmLib.ML.CoT :=
  (AlgorithmLib.ML.Ten.backwardCoT vNeeds 1 SQ (VFWDBUF + 1)
    [(VOUT, VSEED)] vTape true).getD []

def vGradOf (r : Ref) : Option Ref := (vGradCoT.find? (fun p => p.1 == r)).map Prod.snd

/-- `w := w - lr·dw`, in place.  `LR_RECIP` is `1/lr` because `Expr.lit` is a
    `Nat`; at 1e-3 that is the usual fine-tuning rate for a pretrained ViT. -/
def LR_RECIP : Nat := 1000

def sgdSpec : Expr 2 :=
  .add (.var ⟨0, by decide⟩)
       (.neg (.mul (.inv (.lit LR_RECIP)) (.var ⟨1, by decide⟩)))

/-- One in-place update per trained parameter, derived from where the backward
    put that parameter's gradient.  A parameter the reverse pass produced no
    gradient for gets no step, rather than a silent zero. -/
def vSgd : List AlgorithmLib.ML.TOp :=
  (List.range VBASE).filterMap (fun i =>
    if 5 ≤ i then
      (vGradOf i).map (fun g => .upd2 sgdSpec i g (vInBytes.getD i 0 / 128))
    else none)

/-- Rows to re-chunk elementwise passes to, or `0` to leave them element-shaped.

    Measured both ways: at `SQ` the tape needs 1517 launches instead of 1928 and
    its critical path is 460 deep instead of 594, and the step takes exactly as
    long.  Fewer kernels is not faster here, because a row pass has one warp per
    row and re-chunking an elementwise pass to rows divides its warp count by the
    row width.  See `group.cu`: at equal traffic, 224 warps doing eight passes
    reach 144 GB/s where 1792 warps doing one each reach 293. -/
def VROWCHUNK : Nat := SQ

/-- **How many rows to chunk an elementwise pass into.**

    `VROWCHUNK` where the addresses divide that way, because then the pass shares
    a chunk shape with the row passes beside it and can join their kernel.  Where
    they do not, its *own* grid, which sounds like a no-op and is not: at
    `rows = grid` the width is 32 and `.rowOf 32 0` evaluates to `cta*32 + o`,
    which is `elemIx` exactly — the same addresses, the same schedule, but
    written as a row pass, and a row pass has a `BCast` with somewhere to put a
    base offset.  That is what lets anything downstream read it as a slice of a
    fused buffer. -/
def vRowsFor (op : AlgorithmLib.ML.TOp) : Nat :=
  let g := AlgorithmLib.ML.TOp.gridOf op
  if VROWCHUNK != 0 && (32 * g) % VROWCHUNK == 0 then VROWCHUNK else g

/-- The whole training step as one sequence: forward, backward, update, with
    every elementwise pass written as a row pass.

    `TOp.atRows_den` says the function is unchanged; what changes is which
    operations share a chunk shape — 1928 launches became 1517 — and whether an
    operand can be read at an offset, which is what buffer fusion needs.

    Applied here rather than to `vTape`, because `TOp.grad` has no reverse rule
    for a row pass and the backward is derived from the forward above. -/
def vAll : List AlgorithmLib.ML.TOp :=
  (vTape ++ vBwd ++ vSgd).map (fun op => op.atRows (vRowsFor op))

def vAllBufs : List (Ref × Nat) := vAll.map (AlgorithmLib.ML.TOp.outSize SQ)
def VNBUF : Nat :=
  max (VFWDBUF + 1) (vAllBufs.foldl (fun a p => max a (p.1 + 1)) VBASE)

def vBufBytes : List Nat :=
  vInBytes ++ (List.range (VNBUF - VBASE)).map (fun k =>
    let need := vAllBufs.foldl (fun a p => if p.1 == VBASE + k then max a p.2 else a) 0
    -- `VSEED` is written by the host, not by any operation, so its size comes
    -- from the shape it seeds rather than from the tape.
    max 128 (if VBASE + k == VSEED then SQ * NC * 4 else need))

/-- **The buffer the host does not supply.**  Everything else comes from the
    blob; the mask comes from the artifact. -/
def VMASK_BUF : Ref := 3

/-- How many bytes of the host blob a buffer takes.  Zero for the mask, so the
    blob a host packs has no entry for it and cannot disagree about one. -/
def vHostBytesOf (i : Nat) : Nat :=
  if i == VMASK_BUF then 0 else vInBytes.getD i 0

def vHostIn : AlgorithmLib.Layout.RegionMap :=
  (List.range VBASE).map (fun i =>
    ⟨s!"in{i}", ((List.range i).map vHostBytesOf).foldl (· + ·) 0, vHostBytesOf i⟩)

def VHOST_BYTES : Nat := AlgorithmLib.Layout.RegionMap.total vHostIn

def u32le (v : Nat) : List UInt8 :=
  [UInt8.ofNat (v % 256), UInt8.ofNat (v / 256 % 256),
   UInt8.ofNat (v / 65536 % 256), UInt8.ofNat (v / 16777216 % 256)]

def F32_ONE : Nat := 0x3F800000
def F32_ZERO : Nat := 0

/-- **Real tokens**: one class token and 196 patches.  `SQ - TOK` query rows and
    `SK - TOK` key columns past this are padding. -/
def TOK : Nat := 197

/-- **What a padded key scores.**  Added to every score row, so a padded key's
    exponential underflows to zero: it takes no weight in the softmax, and the
    value it would have been multiplied by never reaches the output.

    It is the artifact's own constant, laid out into `initial_memory` and
    uploaded from there — not a number a host script has to agree about. -/
def VMASK_VAL : Float32 := -1e30
def VMASK_BITS : Nat := 0xF149F2CA

/-- The mask, one `f32` per key: zero on a real token, the floor on a padded
    one.  Derived from `TOK` and `SK`, so a geometry change moves it. -/
def vMaskWords : List Nat :=
  (List.range SK).map (fun j => if j < TOK then F32_ZERO else VMASK_BITS)

def vMaskBytes : List UInt8 := vMaskWords.flatMap u32le

/-- A contraction as a cuBLAS call.  cuBLAS is column-major and a row-major
    `(r x c)` is a column-major `(c x r)` of the same bytes, so each of the
    three contractions becomes one transpose choice.  The FFI derives
    `lda`/`ldb`/`ldc` from `m`/`n`/`k` and the transposes. -/
def vGemmOf : AlgorithmLib.ML.TOp →
    Option (Nat × Nat × Nat × Nat × Nat × Ref × Ref × Ref)
  | .mv    (.vendor _) w x o b i ow _ => some (1, 0, ow, b,  i,  w, x, o)
  | .mvT   (.vendor _) w d o b i ow => some (0, 0, i,  b,  ow, w, d, o)
  | .outer (.vendor _) d x o b i ow => some (0, 1, i,  ow, b,  x, d, o)
  | _ => none

/-- **Warps per block, per operation.**

    A block of one warp is 32 threads, which caps occupancy near a third of the
    card and left a pure streaming pass at 83 GB/s of 360.  The chunk indices a
    launch must cover are the same either way, so this packs `W` of them per
    block and divides the grid by `W`.

    `W` is the largest power of two that divides the grid, capped at eight, so
    the launch covers exactly the chunks the kernel expects and no guard is
    needed.  None of these kernels use shared memory, which is the condition
    `PTX_WARP_SCRATCH` names. -/
def vWarpsOf (g : Nat) : Nat :=
  if g % 2 = 0 then 2 else 1

/-- **The warp count divides the grid it is chosen for.**

    The obligation the emitter never states and nothing else checks: a warp
    takes one chunk, so a count that does not divide the grid leaves the last
    warps addressing rows no block owns.  That is not a slow kernel, it is a
    wrong one — four warps everywhere left the forward bit-identical and
    silently zeroed every reduction gradient, because the reductions are the
    launches whose grids four does not divide.

    Stated about `vWarpsOf` rather than about this tape, so it holds at every
    grid any future schedule asks for and an edit that breaks it fails the
    build. -/
theorem vWarpsOf_divides (g : Nat) : g % vWarpsOf g = 0 := by
  unfold vWarpsOf
  split
  · assumption
  · omega

/-- **Which loads take the read-only path.**  `.nc` is sound for any buffer a
    kernel does not store to, but the library is explicit that it is not always
    faster: a buffer read once and never revisited gains nothing from a cache.
    Nearly every buffer here is written by one kernel and read by the next, which
    is exactly that case — so it was measured rather than assumed: `.none` costs
    **13.51 ms a step against `.all`'s 12.76**, so the read-only path wins here
    and the intuition above is wrong for this tape. -/
def vRO : AlgorithmLib.ML.ROPolicy := .all

/-- How many times fusing would duplicate the producer.

    `WFExp.fuseA` substitutes the producer at *every* `.reg 1` of the consumer,
    so a consumer reading its first operand twice emits the producer twice.
    That trades a load for arithmetic, which is a different bargain from the one
    the traffic census is counting, so those sites are not taken. -/
def wfRegOnes : AlgorithmLib.ML.WFExp → Nat
  | .reg r    => if r == 1 then 1 else 0
  | .lit _    => 0
  | .add a b  => wfRegOnes a + wfRegOnes b
  | .mul a b  => wfRegOnes a + wfRegOnes b
  | .maxW a b => wfRegOnes a + wfRegOnes b
  | .geF a b  => wfRegOnes a + wfRegOnes b
  | .neg a    => wfRegOnes a
  | .inv a    => wfRegOnes a
  | .exp a    => wfRegOnes a
  | .ex2 a    => wfRegOnes a
  | .rsqrt a  => wfRegOnes a

/-- **Intermediates whose producer and consumer the guard would fuse if they
    were adjacent.**

    Exactly one writer and exactly one reader, a row pass at both ends, and
    something unrelated in between.  Both ends are required to sit in the same
    capture segment: a hoist moves the reader up to the writer and everything it
    passes stays between them, so an intermediate whose ends straddle the
    forward/backward join is left alone rather than moved across it. -/
def vHoistTargets (narrow : Bool) (bnd : List Nat) (t : List AlgorithmLib.ML.TOp) :
    List AlgorithmLib.ML.Buf := Id.run do
  let arr := t.toArray
  let seg := fun (i : Nat) => (bnd.filter (fun b => b ≤ i)).length
  let mut nwrite : Array Nat := Array.replicate VNBUF 0
  let mut writer : Array Int := Array.replicate VNBUF (-1)
  let mut nread : Array Nat := Array.replicate VNBUF 0
  let mut reader : Array Int := Array.replicate VNBUF (-1)
  for i in [0:t.length] do
    match arr[i]? with
    | none => pure ()
    | some op =>
      let o := (AlgorithmLib.ML.TOp.outSize SQ op).1
      nwrite := nwrite.set! o (nwrite.getD o 0 + 1)
      if writer.getD o (-1) < 0 then writer := writer.set! o (Int.ofNat i)
      for r in AlgorithmLib.ML.TOp.reads op do
        nread := nread.set! r (nread.getD r 0 + 1)
        reader := reader.set! r (Int.ofNat i)
  let mut out : List AlgorithmLib.ML.Buf := []
  for r in [0:VNBUF] do
    if VBASE ≤ r && nwrite.getD r 0 == 1 && nread.getD r 0 == 1
        && writer.getD r (-1) ≥ 0 then
      let w := (writer.getD r 0).toNat
      let c := (reader.getD r 0).toNat
      if w + 1 < c && seg w == seg c then
        match arr[w]?, arr[c]? with
        | some pw, some pc =>
          let okw := match pw with
            | AlgorithmLib.ML.TOp.ziprow ..  => !narrow
            | AlgorithmLib.ML.TOp.ziprow3 .. => true
            | _ => false
          let okc := match pc with
            | AlgorithmLib.ML.TOp.ziprow .. => true
            | _ => false
          if okw && okc then out := out ++ [r]
        | _, _ => pure ()
  pure out

/-- **Which intermediates are hoisted before fusing: none.**

    Taking them all is sound and it does everything it was meant to — 36 more
    pairs fuse, 7 MiB stops moving, the launch count falls by 36 and the
    dependence depth falls from 376 to 364 — and the step gets *slower*, 4.91 ms
    to 5.04.  Every static measure improved and the wall clock disagreed.

    Taking only the 133 whose producer is a three-operand pass — the shape the
    four-operand rule fuses — changes *nothing at all*: same sites, same
    traffic.  So adjacency is not what holds those back; the guard refuses them
    for a reason `vRegForward` cannot see, since that census asks only for one
    writer, one reader and one launch, which is far weaker than what
    `fuseNormAt` checks.

    The machinery is a menu entry, not a missing optimisation: `applyHoists_den`
    holds on every buffer with no side condition, so turning this back on is
    changing a list. -/
def vHoistList : List AlgorithmLib.ML.Buf := []

/-- How many of the named hoists the exchange guard actually accepts, each
    judged on the tape as it stands.  A refusal is silent in `applyHoists`, so
    without this a list of a hundred targets and a tape that never moved look
    the same. -/
def vHoistApplied : Nat :=
  (vHoistList.filter (fun b => (AlgorithmLib.ML.hoistTo SQ b vAll).isSome)).length

/-- The intermediates whose *producer* is sent down to its consumer instead.
    Sound the same way and by the same primitive; what differs is which side has
    to be able to pass the operations in between. -/
def vSinkList : List AlgorithmLib.ML.Buf :=
  vHoistTargets true [vTape.length, vTape.length + vBwd.length] vAll

/-- How many sinks the exchange guard accepts, each judged on `vAll`. -/
def vSinkApplied : Nat :=
  (vSinkList.filter (fun b => (AlgorithmLib.ML.sinkTo SQ b vAll).isSome)).length

def vHoisted : List AlgorithmLib.ML.TOp :=
  AlgorithmLib.ML.applySinks SQ vSinkList
    (AlgorithmLib.ML.applyHoists SQ vHoistList vAll)

/-- **The tape with each consumer turned toward what the operation before it
    wrote.**  Bit-identical to `vAll` on every buffer (`orientTape_den`); what it
    changes is which slot holds the produced value, and the fusion guard only
    consumes it from the first.  On this tape the second slot holds it slightly
    more often than the first — 134 adjacent pairs against 110 — so orienting is
    worth more than the guard it feeds. -/
def vOriented : List AlgorithmLib.ML.TOp :=
  AlgorithmLib.ML.orientTape (fun op => (AlgorithmLib.ML.TOp.outSize SQ op).1) vHoisted

/-- **Every row-pass pair the library will fuse, on the tape as shipped.**

    Read on the oriented, re-chunked tape rather than on the term's own: the
    guard matches `.ziprow`, `atRows` rewrites every elementwise pass into one,
    and `orientTape` puts the produced value in the slot the guard consumes. -/
def vFuseAll : List (Nat × AlgorithmLib.ML.Buf) := AlgorithmLib.ML.fuseTargets vOriented

/-- **The sites worth taking on a tape, and why the others are not.**

    Two conditions.  A pair straddling the forward/backward join would fuse two
    operations the capture ranges hold apart, and those ranges are counts of
    operations per segment, so sites are kept inside one segment and the counts
    stay derivable.  And a consumer that reads the produced value more than once
    would emit the producer's whole expression that many times, which is
    arithmetic the pair did not do. -/
def vSitesOf (bnd : List Nat) (t : List AlgorithmLib.ML.TOp) :
    List AlgorithmLib.ML.Buf :=
  let arr : Array AlgorithmLib.ML.TOp := t.toArray
  (AlgorithmLib.ML.fuseTargets t).filterMap (fun p =>
    if bnd.any (fun b => p.1 + 1 == b) then none else
    match (arr[p.1 + 1]? : Option AlgorithmLib.ML.TOp) with
    | some (AlgorithmLib.ML.TOp.ziprow _ _ _ f2 _ _ _ _ _ _) =>
        if wfRegOnes f2 ≤ 1 then some p.2 else none
    -- A reduction reads the produced value exactly once per element, so there
    -- is no expression to emit twice and nothing to count.
    | some (AlgorithmLib.ML.TOp.rowdot ..) => some p.2
    | _ => none)

/-- Where each removed temporary was written, in the tape the round ran on.  A
    fusion replaces two operations by one at the producer's position, so a kill
    below a boundary moves that boundary down by exactly one. -/
def vKillPosIn (t : List AlgorithmLib.ML.TOp) (killed : List AlgorithmLib.ML.Buf) :
    List Nat :=
  let arr : Array AlgorithmLib.ML.TOp := t.toArray
  killed.filterMap (fun tm =>
    (List.range t.length).find? (fun i =>
      match (arr[i]? : Option AlgorithmLib.ML.TOp) with
      | some op => (AlgorithmLib.ML.TOp.outSize SQ op).1 == tm
      | none    => false))

/-- **Why the four-operand rule refuses a pair the census says it should take.**

    `vRegForward` asks only for one writer, one reader and one launch.
    `fuseNormAt` asks for much more, and this is which of those conditions the
    remaining three-operand-into-row-pass sites fail — counted in MiB, since one
    refusal on a wide row is worth a hundred on a scalar. -/
def vRefuseCensus : List (String × Nat) := Id.run do
  let arr := vOriented.toArray
  let mut nwrite : Array Nat := Array.replicate VNBUF 0
  let mut writer : Array Int := Array.replicate VNBUF (-1)
  let mut nread : Array Nat := Array.replicate VNBUF 0
  let mut reader : Array Int := Array.replicate VNBUF (-1)
  for i in [0:vOriented.length] do
    match arr[i]? with
    | none => pure ()
    | some op =>
      let o := (AlgorithmLib.ML.TOp.outSize SQ op).1
      nwrite := nwrite.set! o (nwrite.getD o 0 + 1)
      if writer.getD o (-1) < 0 then writer := writer.set! o (Int.ofNat i)
      for r in AlgorithmLib.ML.TOp.reads op do
        nread := nread.set! r (nread.getD r 0 + 1)
        reader := reader.set! r (Int.ofNat i)
  let mut acc : List (String × Nat) := []
  for r in [0:VNBUF] do
    if VBASE ≤ r && nwrite.getD r 0 == 1 && nread.getD r 0 == 1
        && writer.getD r (-1) ≥ 0 then
      match arr[(writer.getD r 0).toNat]?, arr[(reader.getD r 0).toNat]? with
      | some (AlgorithmLib.ML.TOp.ziprow3 _ _ _ _ f1 _ _ _ nP offP w rows),
        some (AlgorithmLib.ML.TOp.ziprow a2 _ o2 f2 mP _ nC offC w2 rows2) =>
        let kb := 2 * vBufBytes.getD r 0 / 1024
        let why :=
          if a2 != r then "value not in the consumer's first slot"
          else if w2 != w then "windows differ in width"
          else if rows2 != rows then "different row counts"
          else if mP != AlgorithmLib.ML.BCast.rowOf nP offP then
            "not read at the producer's own window"
          else if !f1.tripleOnly then "producer's expression is not three-operand"
          else if !f2.pairOnly then "consumer's expression is not two-operand"
          else if o2 == r then "consumer writes what it consumes"
          else if !(offP + w ≤ nP && offC + w ≤ nC && 0 < nP) then "window out of range"
          else "would fuse if adjacent"
        acc := (if (acc.find? (fun q => q.1 == why)).isSome
                then acc.map (fun q => if q.1 == why then (q.1, q.2 + kb) else q)
                else acc ++ [(why, kb)])
      | _, _ => pure ()
  pure ((acc.toArray.qsort (fun x y => x.2 > y.2)).toList.map
    (fun q => (q.1, q.2 / 1024)))

def vFuseSites : List AlgorithmLib.ML.Buf :=
  vSitesOf [vTape.length, vTape.length + vBwd.length] vOriented

/-- **The first round: `vAll` with every accepted pair fused.**

    Named by the buffer each one removes rather than by position, because a
    fusion shortens the tape and every later position moves — `FuseSite.killing`
    is stable under that and `FuseSite.at` is not. -/
def vFuseRun : List AlgorithmLib.ML.Buf × List AlgorithmLib.ML.TOp :=
  AlgorithmLib.ML.applyFuse [] (vFuseSites.map AlgorithmLib.ML.FuseSite.killing) vOriented

def vKilled1 : List AlgorithmLib.ML.Buf := vFuseRun.1
def vT1 : List AlgorithmLib.ML.TOp := vFuseRun.2

def vKillPos1 : List Nat := vKillPosIn vOriented vKilled1
def VFWD_1 : Nat :=
  vTape.length - (vKillPos1.filter (fun i => i < vTape.length)).length
def VBWD_1 : Nat :=
  (vTape.length + vBwd.length)
    - (vKillPos1.filter (fun i => i < vTape.length + vBwd.length)).length

/-- **The second round.**

    The first turns each fused pair into a `ziprow3`, and a `ziprow3` feeding a
    third row pass is a site the pair rule cannot take — which is most of what a
    chain longer than two is.

    The round runs on the first round's output as it stands.  Re-orienting is
    sound and would expose the second-slot consumers as well, but it rewrites
    the operand order of passes that were not going to fuse either way, and
    those rewrites change enough kernel texts to cost more in grouping than the
    extra sites are worth: measured 5.09 ms against 4.86. -/
def vOriented2 : List AlgorithmLib.ML.TOp :=
  AlgorithmLib.ML.applyHoists SQ vHoistList vT1

def vFuseSites2 : List AlgorithmLib.ML.Buf := vSitesOf [VFWD_1, VBWD_1] vOriented2

def vFuseRun2 : List AlgorithmLib.ML.Buf × List AlgorithmLib.ML.TOp :=
  AlgorithmLib.ML.applyFuse [] (vFuseSites2.map AlgorithmLib.ML.FuseSite.killing) vOriented2

def vKilled2 : List AlgorithmLib.ML.Buf := vFuseRun2.1

/-- Every temporary either round stopped writing. -/
def vKilled : List AlgorithmLib.ML.Buf := vKilled1 ++ vKilled2

/-- **The tape as shipped.**  `vit_fusion_sound` is the statement that this
    computes what `vAll` does. -/
def vFused : List AlgorithmLib.ML.TOp := vFuseRun2.2

def vKillPos2 : List Nat := vKillPosIn vOriented2 vKilled2

/-- The forward/backward/update boundaries, in the shipped tape's own indexing.
    The capture ranges slice on these, so they are counted rather than assumed. -/
def VFWD_N : Nat := VFWD_1 - (vKillPos2.filter (fun i => i < VFWD_1)).length
def VBWD_N : Nat := VBWD_1 - (vKillPos2.filter (fun i => i < VBWD_1)).length
def VSTEP_N : Nat := vFused.length

/-- **The shipped tape computes what the model's tape computes.**

    Two steps, neither of which invokes a law.  `orientTape_den` says commuting a
    row pass's operands changes nothing at all; `applyFuse_den` says fusing
    changes nothing on any buffer except the temporaries it stopped writing.
    Together: every buffer the tape produces holds what `vAll` would have left
    there, from any starting memory.

    The conclusion is conditional on `b ∉ vKilled`, which is the whole content of
    a fusion — so the two theorems below are what stop it being vacuous. -/
theorem vit_fusion_sound (m : AlgorithmLib.ML.Buf → Nat → Float32)
    (b : AlgorithmLib.ML.Buf) (hb : b ∉ vKilled) :
    (vFused.foldl (fun mm o => AlgorithmLib.ML.TOp.den o mm) m) b
      = (vAll.foldl (fun mm o => AlgorithmLib.ML.TOp.den o mm) m) b := by
  have h1 : b ∉ vKilled1 := fun h => hb (List.mem_append_left _ h)
  have h2 : b ∉ vKilled2 := fun h => hb (List.mem_append_right _ h)
  refine ((AlgorithmLib.ML.applyFuse_den []
      (vFuseSites2.map AlgorithmLib.ML.FuseSite.killing) vOriented2 m b h2).trans
    ((congrFun (AlgorithmLib.ML.applyHoists_den SQ vHoistList vT1 m) b).trans ?_))
  refine ((AlgorithmLib.ML.applyFuse_den []
      (vFuseSites.map AlgorithmLib.ML.FuseSite.killing) vOriented m b h1).trans ?_)
  refine ((congrFun (AlgorithmLib.ML.orientTape_den
      (fun op => (AlgorithmLib.ML.TOp.outSize SQ op).1) vHoisted m) b).trans ?_)
  exact (congrFun (AlgorithmLib.ML.applySinks_den SQ vSinkList
      (AlgorithmLib.ML.applyHoists SQ vHoistList vAll) m) b).trans
    (congrFun (AlgorithmLib.ML.applyHoists_den SQ vHoistList vAll m) b)

/-- **The shipped tape computes what the term's own lowering computes.**

    One step further left than `vit_fusion_sound`: past the re-chunking, onto
    `vTape ++ vBwd ++ vSgd`, which is not written here at all — `vTape` is
    `(model NL).compile`, `vBwd` is `Ten.backwardFrom` applied to it, and `vSgd`
    is one `upd2` per parameter where the cotangent map left that parameter's
    gradient.  No operation of the backward or the update is authored.

    So the chain from what the artifact launches back to the `Ten` term is:
    fusion (twice) and orientation by `applyFuse_den`/`orientTape_den`, sinking
    by `applySinks_den`, re-chunking by `atRows_map_den`, and then a tape that
    is a function of the term.  What it does *not* yet reach is the launches
    themselves — see `VitScan.notYetStated`. -/
theorem vit_lowering_sound (m : AlgorithmLib.ML.Buf → Nat → Float32)
    (b : AlgorithmLib.ML.Buf) (hb : b ∉ vKilled) :
    (vFused.foldl (fun mm o => AlgorithmLib.ML.TOp.den o mm) m) b
      = ((vTape ++ vBwd ++ vSgd).foldl (fun mm o => AlgorithmLib.ML.TOp.den o mm) m) b :=
  (vit_fusion_sound m b hb).trans
    (congrFun (AlgorithmLib.ML.TOp.atRows_map_den vRowsFor (vTape ++ vBwd ++ vSgd) m) b)

/-- **…and the forward half of that tape is the term's lowering, not a copy of
    it.**  `vTape` is *defined* as the compilation, so this is `rfl` — which is
    the point: there is no second description of the model to drift from. -/
theorem vit_tape_is_the_term :
    vTape = ((model NL).compile VBASE vSched).2 := rfl

/-- Buffers a contraction fills only partly: the output ref, the rows written,
    the rows allocated.  `mvAt` is the only way to make one. -/
def vPadded : List (Ref × Nat × Nat) :=
  vAll.filterMap (fun op =>
    match op with
    | .mv _ _ _ o b _ _ bA => if bA > b then some (o, b, bA) else none
    | _ => none)

/-- **Nothing on the tape writes a padded buffer's tail.**

    `SK - SQ` rows of each key and value projection are never computed, and the
    model is correct only if reading them yields a *finite* number — the mask
    then sends the score they produce to `-1e30`, so softmax gives them weight
    exactly zero.  A device buffer is allocated zeroed, so what has to hold is
    that no later operation puts anything else there.

    Stated as: a partly-filled buffer has exactly one writer, the contraction
    that fills it.  Then the tail is untouched from allocation onwards, for the
    whole run and not merely the first step.  Checked over the *shipped* tape —
    after fusion, sinking and re-chunking — because those are the passes that
    could introduce a second writer. -/
def vPadTailUnwritten : Bool :=
  vPadded.all (fun p =>
    (vAll.filter (fun op => (AlgorithmLib.ML.TOp.outSize SQ op).1 == p.1)).length == 1)

/-- A one-word name for what an operation is, so a census can say which shapes
    a chain is actually made of. -/
def vKindOf : AlgorithmLib.ML.TOp → String
  | .ziprow _ _ _ _ _ _ _ _ _ _      => "ziprow"
  | .ziprow3 _ _ _ _ _ _ _ _ _ _ _ _ => "ziprow3"
  | .ziprow4 _ _ _ _ _ _ _ _ _ _ _ _ _ _ => "ziprow4"
  | .rowdot4 _ _ _ _ _ _ _ _ _ _ _ _ => "rowdot4"
  | .rowdot _ _ _ _ _ _ _            => "rowdot"
  | .rowsq _ _ _ _                   => "rowsq"
  | .rowmax _ _ _ _ _                => "rowmax"
  | .ew1 _ _ _ _                     => "ew1"
  | .ew2 _ _ _ _ _                   => "ew2"
  | .ew3 _ _ _ _ _ _                 => "ew3"
  | .ew4 _ _ _ _ _ _ _               => "ew4"
  | .upd2 _ _ _ _                    => "upd2"
  | .smce _ _ _ _ _                  => "smce"
  | .mv _ _ _ _ _ _ _ _              => "mv"
  | .mvT _ _ _ _ _ _ _               => "mvT"
  | .outer _ _ _ _ _ _ _             => "outer"

/-- **What the chains are actually made of.**

    Every adjacent pair where the second reads what the first wrote, keyed by
    the two shapes and by which operand slot carries the value.  This is the
    specification for what a fused row block has to cover: the pair fusion in
    `Fuse.lean` handles exactly `ziprow -> ziprow` in slot 0, and this says how
    much of the tape that is. -/
def vChainCensus : List (String × Nat) := Id.run do
  let arr := vAll.toArray
  let mut acc : List (String × Nat) := []
  let mut bytes := 0
  let mut n := 0
  for i in [0:vAll.length - 1] do
    match arr[i]?, arr[i+1]? with
    | some a, some b =>
      let o := (AlgorithmLib.ML.TOp.outSize SQ a).1
      let rs := AlgorithmLib.ML.TOp.reads b
      match rs.idxOf? o with
      | none => pure ()
      | some slot =>
        n := n + 1
        bytes := bytes + vBufBytes.getD o 0
        let key := s!"{vKindOf a}->{vKindOf b}@{slot}:{vBufBytes.getD o 0 / 1024}K"
        if (acc.find? (fun p => p.1 == key)).isSome then
          acc := acc.map (fun p => if p.1 == key then (p.1, p.2 + 1) else p)
        else acc := acc ++ [(key, 1)]
    | _, _ => pure ()
  let top := ((acc.toArray.qsort (fun x y => x.2 > y.2)).toList.take 12)
  pure ([("adjacent producer-consumer pairs", n),
         ("their intermediates, MiB (x2 to move)", bytes / 1048576)] ++ top)

def vFuseCensus : List (String × Nat) :=
  [ ("sites the guard accepts", vFuseAll.length)
  , ("taken", vFuseSites.length)
  , ("dropped: producer would be emitted twice", vFuseAll.length - vFuseSites.length) ]

/-- The tape as an array, converted once.  Every group asks for its members by
    position, and `List.toArray` inside that lookup is a copy of the whole tape
    per question. -/
def vAllArr : Array AlgorithmLib.ML.TOp := vFused.toArray

/-- **A chunk-local operation**: every address it touches belongs to the chunk
    its own `cta` selects, so the warp that writes a value is the warp that
    reads it back.  These are the operations whose statements may be run one
    after another by the same warp with no barrier between them.

    Two families qualify, and the grid keeps them apart: the row passes, whose
    chunk is a row indexed by `.ctaId`, and the elementwise passes, whose chunk
    is 32 elements at `elemIx`. -/
def vChunkLocal (op : AlgorithmLib.ML.TOp) : Bool :=
  match op with
  | .rowsq _ _ _ _ => true
  | .rowdot _ _ _ _ _ _ _ => true
  | .rowmax _ _ _ _ _ => true
  | .ziprow _ _ _ _ _ _ _ _ _ _ => true
  | .ziprow3 _ _ _ _ _ _ _ _ _ _ _ _ => true
  | .ziprow4 _ _ _ _ _ _ _ _ _ _ _ _ _ _ => true
  | .rowdot4 _ _ _ _ _ _ _ _ _ _ _ _ => true
  | .ew1 _ _ _ _ => true
  | .ew2 _ _ _ _ _ => true
  | .ew3 _ _ _ _ _ _ => true
  | .ew4 _ _ _ _ _ _ _ => true
  -- in place, but still one chunk per `cta`: a warp updates the slice of a
  -- parameter its own index selects and reads nothing else
  | .upd2 _ _ _ _ => true
  | _ => false

/-- Buffers an operation reads **across** rows: a gain, a mask, a table.  Those
    reads are row-independent, so a group may only contain one if no earlier
    member of that group writes the buffer — otherwise a warp would read a row
    another warp is responsible for. -/
def vSharedReads (op : AlgorithmLib.ML.TOp) : List Ref :=
  let sh := fun (m : AlgorithmLib.ML.BCast) (r : Ref) =>
    match m with
    | .sharedAt _ => [r]
    | .constAt _  => [r]
    | _           => []
  match op with
  | .rowdot a b _ mA mB _ _ => sh mA a ++ sh mB b
  | .ziprow a b _ _ mA mB _ _ _ _ => sh mA a ++ sh mB b
  | .ziprow3 a b c _ _ mA mB mC _ _ _ _ => sh mA a ++ sh mB b ++ sh mC c
  | .ziprow4 a b c d _ _ mA mB mC mD _ _ _ _ =>
      sh mA a ++ sh mB b ++ sh mC c ++ sh mD d
  | .rowdot4 a b c d _ _ mA mB mC mD _ _ =>
      sh mA a ++ sh mB b ++ sh mC c ++ sh mD d
  | _ => []

/-- **Consecutive row-local operations over the same rows, as one kernel.**

    A measurement, not a guess: at these shapes a kernel costs about 0.93 us
    before it moves a byte, and a 338 KB row pass moves its bytes in 1.0 us — so
    five passes are five times the fixed cost and one pass worth of bandwidth.
    Sequencing their statements pays that cost once.

    A group never crosses a segment boundary, because the forward and the
    backward are captured as separate graphs. -/
def vUnitsRaw : List (List Nat) := Id.run do
  let a := vAllArr
  let bounds : List Nat := [0, VFWD_N, VBWD_N, a.size]
  let mut out : List (List Nat) := []
  for (lo, hi) in bounds.zip (bounds.drop 1) do
    let mut cur : List Nat := []
    let mut curGrid := 0
    let mut written : List Ref := []
    for i in [lo:hi] do
      match a[i]? with
      | none => pure ()
      | some op =>
        let g := AlgorithmLib.ML.TOp.gridOf op
        let joins := vChunkLocal op && !cur.isEmpty && g == curGrid
                       && (vSharedReads op).all (fun r => !written.contains r)
        if joins then
          cur := cur ++ [i]
          written := (AlgorithmLib.ML.TOp.outSize SQ op).1 :: written
        else
          if !cur.isEmpty then out := out ++ [cur]
          if vChunkLocal op then
            cur := [i]; curGrid := g
            written := [(AlgorithmLib.ML.TOp.outSize SQ op).1]
          else
            cur := []; written := []; out := out ++ [[i]]
    if !cur.isEmpty then out := out ++ [cur]
  pure out

def vUnitOps (u : List Nat) : List AlgorithmLib.ML.TOp :=
  u.filterMap (fun i => vAllArr[i]?)

/-- Why consecutive operations land in different kernels, counted over the whole
    tape.  Each kernel costs about a microsecond before it moves a byte, so the
    census of what ends a group is the census of what to relax. -/
def vBreaks : List (String × Nat) := Id.run do
  let a := vAllArr
  let bounds : List Nat := [0, VFWD_N, VBWD_N, a.size]
  let mut nVendor := 0
  let mut nAfter := 0
  let mut nGrid := 0
  let mut nShared := 0
  for (lo, hi) in bounds.zip (bounds.drop 1) do
    let mut cur : List Nat := []
    let mut curGrid := 0
    let mut written : List Ref := []
    for i in [lo:hi] do
      match a[i]? with
      | none => pure ()
      | some op =>
        let g := AlgorithmLib.ML.TOp.gridOf op
        let isLoc := vChunkLocal op
        let joins := isLoc && !cur.isEmpty && g == curGrid
                       && (vSharedReads op).all (fun r => !written.contains r)
        if joins then
          cur := cur ++ [i]
          written := (AlgorithmLib.ML.TOp.outSize SQ op).1 :: written
        else
          if !isLoc then nVendor := nVendor + 1
          else if cur.isEmpty then nAfter := nAfter + 1
          else if g == curGrid then nShared := nShared + 1
          else nGrid := nGrid + 1
          if isLoc then
            cur := [i]; curGrid := g
            written := [(AlgorithmLib.ML.TOp.outSize SQ op).1]
          else
            cur := []; written := []
  pure [("not chunk-local", nVendor), ("first local after a non-local", nAfter),
        ("grid mismatch", nGrid), ("reads what the group wrote", nShared)]

/-- Which pairs of grids meet at a grid-mismatch break, commonest first.  A pair
    whose two grids cover the same tensor is a group the chunk shape split, not
    the data. -/
def vGridPairs : List ((Nat × Nat) × Nat) := Id.run do
  let a := vAllArr
  let bounds : List Nat := [0, VFWD_N, VBWD_N, a.size]
  let mut ps : List (Nat × Nat) := []
  for (lo, hi) in bounds.zip (bounds.drop 1) do
    let mut cur : List Nat := []
    let mut curGrid := 0
    for i in [lo:hi] do
      match a[i]? with
      | none => pure ()
      | some op =>
        let g := AlgorithmLib.ML.TOp.gridOf op
        let isLoc := vChunkLocal op
        if isLoc && !cur.isEmpty && g == curGrid then
          cur := cur ++ [i]
        else
          if isLoc && !cur.isEmpty then ps := ps ++ [(curGrid, g)]
          if isLoc then cur := [i]; curGrid := g else cur := []
  let ds := ps.eraseDups
  pure (((ds.map (fun p => (p, (ps.filter (· == p)).length))).toArray.qsort
    (fun x y => x.2 > y.2)).toList.take 10)

/-- What the grouping would be if an elementwise pass over a `SQ`-row tensor were
    chunked by row rather than by 32 elements.  Both cover the same addresses, so
    this is a chunk-shape question and not a semantic one; the count says whether
    answering it is worth the rewrite. -/
def vGridNorm (op : AlgorithmLib.ML.TOp) : Nat :=
  let g := AlgorithmLib.ML.TOp.gridOf op
  match op with
  | .ew1 .. | .ew2 .. | .ew3 .. => if (32 * g) % SQ == 0 then SQ else g
  | _ => g

def vUnitsNorm : List (List Nat) := Id.run do
  let a := vAllArr
  let bounds : List Nat := [0, VFWD_N, VBWD_N, a.size]
  let mut out : List (List Nat) := []
  for (lo, hi) in bounds.zip (bounds.drop 1) do
    let mut cur : List Nat := []
    let mut curGrid := 0
    let mut written : List Ref := []
    for i in [lo:hi] do
      match a[i]? with
      | none => pure ()
      | some op =>
        let g := vGridNorm op
        if vChunkLocal op && !cur.isEmpty && g == curGrid
             && (vSharedReads op).all (fun r => !written.contains r) then
          cur := cur ++ [i]
          written := (AlgorithmLib.ML.TOp.outSize SQ op).1 :: written
        else
          if !cur.isEmpty then out := out ++ [cur]
          if vChunkLocal op then
            cur := [i]; curGrid := g
            written := [(AlgorithmLib.ML.TOp.outSize SQ op).1]
          else
            cur := []; written := []; out := out ++ [[i]]
    if !cur.isEmpty then out := out ++ [cur]
  pure out

/-- A row pass whose every operand walks its own row is a pure map: it reads and
    writes one address per element and shares nothing along the row.  Those are
    the ones that could run one element per lane instead of one row per warp,
    which is `w` times the warps. -/
def vPureMap (op : AlgorithmLib.ML.TOp) : Bool :=
  let isRow : AlgorithmLib.ML.BCast → Bool
    | .rowOf _ _ => true
    | _          => false
  match op with
  | .ziprow _ _ _ _ mA mB _ _ _ _ => isRow mA && isRow mB
  | .ziprow3 _ _ _ _ _ mA mB mC _ _ _ _ => isRow mA && isRow mB && isRow mC
  | .ziprow4 _ _ _ _ _ _ mA mB mC mD _ _ _ _ =>
      isRow mA && isRow mB && isRow mC && isRow mD
  | .rowdot4 _ _ _ _ _ _ mA mB mC mD _ _ =>
      isRow mA && isRow mB && isRow mC && isRow mD
  | _ => false

/-- Warps the tape asks for, split by whether more of them are available. -/
def vWarpCensus : List (String × Nat × Nat) :=
  let rows := vAll.filter (fun op => AlgorithmLib.ML.TOp.gridOf op == SQ)
  let pure := rows.filter vPureMap
  let red := rows.filter (fun op => !vPureMap op)
  [("row ops, pure map", pure.length, pure.foldl (fun a op => a + AlgorithmLib.ML.TOp.gridOf op) 0),
   ("row ops, shares along the row", red.length, red.foldl (fun a op => a + AlgorithmLib.ML.TOp.gridOf op) 0),
   ("all ops", vAll.length, vAll.foldl (fun a op => a + AlgorithmLib.ML.TOp.gridOf op) 0)]

/-- How many operations sit at each distinct grid, largest first. -/
def vGridHist : List (Nat × Nat) :=
  let gs := (vAll.map AlgorithmLib.ML.TOp.gridOf)
  let ds := gs.eraseDups
  ((ds.map (fun g => (g, (gs.filter (· == g)).length))).toArray.qsort
    (fun x y => x.2 > y.2)).toList.take 12

/-- **What the contractions are**, by shape, commonest first.

    Every one is a separate `cublasSgemmStridedBatched` at `batchCount = 1`, and
    they are about half the tape's launches.  A shape appearing `c` times is `c`
    calls that a batched call could be one of — so this census is the price list
    for `CuBlasBatchedIsSomeReassoc`, and reading it is how to tell whether the
    remaining launches are a handful of large calls or a crowd of tiny ones. -/
def vGemmCensus : List ((Nat × Nat × Nat × Nat × Nat) × Nat) := Id.run do
  let ks := vAll.filterMap (fun op =>
    (vGemmOf op).map (fun g => (g.1, g.2.1, g.2.2.1, g.2.2.2.1, g.2.2.2.2.1)))
  let ds := ks.eraseDups
  pure (((ds.map (fun k => (k, (ks.filter (· == k)).length))).toArray.qsort
    (fun x y => x.2 > y.2)).toList.take 10)

/-- The buffers a group's kernel binds: every member's, in first-use order. -/
def vUnitBufs (u : List Nat) : List AlgorithmLib.ML.Buf :=
  ((vUnitOps u).flatMap (AlgorithmLib.ML.TOp.bufs SQ)).eraseDups

/-- The group's statement: each member's, renamed onto the group's own table.
    `.seq` is the whole of the fusion — no barrier, because each warp keeps to
    its own row throughout, which `vRowLocal` is what decides. -/
def vUnitStmt (u : List Nat) : AlgorithmLib.ML.EWStmt :=
  let bs := vUnitBufs u
  (vUnitOps u).foldl
    (fun acc op => .seq acc ((AlgorithmLib.ML.TOp.stmt SQ op).renameBuf
                              (AlgorithmLib.ML.compactMap bs)))
    .skip

def vUnitGrid (u : List Nat) : Nat :=
  match (vUnitOps u).head? with
  | some op => AlgorithmLib.ML.TOp.gridOf op
  | none => 1

/-- One group's kernel text.  Renaming onto the group's own table is what makes
    the text depend on the shape and not on where in the model the group sits —
    which is what makes twelve identical blocks twelve references to one
    kernel. -/
def vTextOf (u : List Nat) : String :=
  match (vUnitOps u).head? with
  | some op =>
    if (vGemmOf op).isSome then "" else
      AlgorithmLib.ML.emitProvenKernelN "main"
        (vUnitBufs u).length 0 (vUnitStmt u) vRO (vWarpsOf (vUnitGrid u))
  | none => ""

/-- **Distinct kernels, and which one each group launches.**

    Emitting one slot per group would cost a slot per block per pass; the blocks
    are identical, so the texts are.  One pass, keyed on the text. -/
def vSlotMapRaw : Array Nat × Array String := Id.run do
  let mut seen : Std.HashMap (Nat × Nat × AlgorithmLib.ML.EWStmt) Nat := {}
  let mut seenText : Std.HashMap String Nat := {}
  let mut ix : Array Nat := #[]
  let mut uniq : Array String := #[]
  for u in vUnitsRaw do
    -- Key on the statement, not on the text.  Emitting is what `expandEW`
    -- unrolls, so keying on the text emits every group and keeps a
    -- thirtieth of them; the statement decides the text and costs nothing.
    -- A contraction group renders no kernel, so it needs no statement built:
    -- `TOp.stmt` for a contraction is the widest schema there is.
    if ((vUnitOps u).head? >>= vGemmOf).isSome then
      match seenText[""]? with
      | some j => ix := ix.push j
      | none =>
          let j := uniq.size
          seenText := seenText.insert "" j
          uniq := uniq.push ""
          ix := ix.push j
    else
    let k := ((vUnitBufs u).length, vWarpsOf (vUnitGrid u), vUnitStmt u)
    match seen[k]? with
    | some j => ix := ix.push j
    | none =>
        -- Two statements can still render alike, so the text decides the slot;
        -- the statement only decides whether emitting is needed at all.
        let t := vTextOf u
        match seenText[t]? with
        | some j => seen := seen.insert k j; ix := ix.push j
        | none =>
            let j := uniq.size
            seen := seen.insert k j
            seenText := seenText.insert t j
            uniq := uniq.push t
            ix := ix.push j
  pure (ix, uniq)

/-- The slot group `k` launches from. -/
def vSlotIxRaw (k : Nat) : Nat := vSlotMapRaw.1.getD k 0

/-- The groups, as an array, and what each reads and writes.  A group's
    dependences are its members' — the ones inside it are already met by the
    order the one kernel performs them in. -/
def vUnitArrRaw : Array (List Nat) := vUnitsRaw.toArray
def VNUNITRAW : Nat := vUnitArrRaw.size

def vUnitWrites (u : List Nat) : List Ref :=
  ((vUnitOps u).map (fun op => (AlgorithmLib.ML.TOp.outSize SQ op).1)).eraseDups
def vUnitReads (u : List Nat) : List Ref :=
  ((vUnitOps u).flatMap AlgorithmLib.ML.TOp.reads).eraseDups

/-- **Bytes each launch touches**, split by who performs it.

    The step's time tracks traffic much more closely than it tracks launch
    count — three separate launch-count reductions were each worth nothing
    measurable — so this is the census the schedule work is actually against.
    A buffer is counted once per launch that binds it, which is what a launch
    reads or writes if it touches all of what it binds. -/
def vTraffic : List (String × Nat) := Id.run do
  let mut pv := 0; let mut vn := 0
  let mut np := 0; let mut nv := 0
  for i in [0:vUnitsRaw.length] do
    let u := vUnitArrRaw.getD i []
    let b := (vUnitBufs u).foldl (fun a r => a + vBufBytes.getD r 0) 0
    if ((vUnitOps u).head? >>= vGemmOf).isSome then
      vn := vn + b; nv := nv + 1
    else
      pv := pv + b; np := np + 1
  pure [("proven launches", np), ("their MiB", pv / 1048576),
        ("contraction launches", nv), ("their MiB", vn / 1048576)]

/-- **What vertical fusion could remove, at its ceiling.**

    An intermediate is a buffer one operation writes and exactly one reads.
    Fusing that pair removes both the store and the load, so the ceiling on
    what *any* amount of vertical fusion can save is twice the bytes of every
    such buffer — no schedule, no proof obligation, and no dependence on which
    pairs a particular fusion guard accepts.

    Counted against `vTraffic`'s total, this is the question worth asking
    before writing a reverse rule: whether the term that binds is one fusion
    can reach at all.

    Buffers a contraction writes are excluded on both sides: cuBLAS takes whole
    pointers, so nothing can be fused into or out of one. -/
def vFuseCeiling : List (String × Nat) := Id.run do
  let mut writer : Array Int := Array.replicate VNBUF (-1)
  let mut nread : Array Nat := Array.replicate VNBUF 0
  let mut reader : Array Int := Array.replicate VNBUF (-1)
  for (op, i) in vFused.zip (List.range vFused.length) do
    let o := (AlgorithmLib.ML.TOp.outSize SQ op).1
    if writer.getD o (-1) < 0 then writer := writer.set! o (Int.ofNat i)
    for r in AlgorithmLib.ML.TOp.reads op do
      nread := nread.set! r (nread.getD r 0 + 1)
      reader := reader.set! r (Int.ofNat i)
  let mut single := 0
  let mut bytes := 0
  let mut vend := 0
  let arr := vAllArr
  for r in [0:VNBUF] do
    if VBASE ≤ r && nread.getD r 0 == 1 && writer.getD r (-1) ≥ 0 then
      let w := (writer.getD r 0).toNat
      let c := (reader.getD r 0).toNat
      let wv := (arr[w]?.bind vGemmOf).isSome
      let cv := (arr[c]?.bind vGemmOf).isSome
      if wv || cv then vend := vend + vBufBytes.getD r 0
      else
        single := single + 1
        bytes := bytes + vBufBytes.getD r 0
  -- Of that ceiling, how much is between two operations the schedule already
  -- puts in one kernel.  That part needs no new grouping at all -- only that a
  -- `.seq` forward the value in a register instead of storing and reloading it.
  let mut inUnit := 0
  let mut unitOf : Array Int := Array.replicate vAllArr.size (-1)
  for (u, k) in vUnitsRaw.zip (List.range vUnitsRaw.length) do
    for i in u do unitOf := unitOf.set! i (Int.ofNat k)
  for r in [0:VNBUF] do
    if VBASE ≤ r && nread.getD r 0 == 1 && writer.getD r (-1) ≥ 0 then
      let w := (writer.getD r 0).toNat
      let c := (reader.getD r 0).toNat
      let wv := (arr[w]?.bind vGemmOf).isSome
      let cv := (arr[c]?.bind vGemmOf).isSome
      if !wv && !cv && unitOf.getD w (-1) == unitOf.getD c (-2) then
        inUnit := inUnit + vBufBytes.getD r 0
  let tot := (List.range VNBUF).foldl (fun a r => a + vBufBytes.getD r 0) 0
  pure [("write-once read-once buffers", single),
        ("their MiB", bytes / 1048576),
        ("ceiling: MiB fusion could stop moving", 2 * bytes / 1048576),
        ("of that, already in one kernel: MiB", 2 * inUnit / 1048576),
        ("MiB in such buffers a contraction touches", vend / 1048576),
        ("all device MiB", tot / 1048576)]

/-- **The traffic a wider fusion rule would remove, by the rule it needs.**

    `vFuseCeiling` says how many bytes are stored and reloaded between two
    operations the schedule already runs in one kernel.  This says which pair of
    shapes carries them, so the next constructor is chosen by what it is worth
    rather than by which pair is easiest to write.

    Read on the shipped tape, so a pair the guard has already fused is absent
    rather than counted twice, and in MiB rather than in pairs — a hundred
    scalar intermediates are worth nothing and one row is worth a kilobyte
    each way. -/
def vRegForward : List (String × Nat) := Id.run do
  let arr := vAllArr
  let mut writer : Array Int := Array.replicate VNBUF (-1)
  let mut nread : Array Nat := Array.replicate VNBUF 0
  let mut reader : Array Int := Array.replicate VNBUF (-1)
  for i in [0:arr.size] do
    match arr[i]? with
    | none => pure ()
    | some op =>
      let o := (AlgorithmLib.ML.TOp.outSize SQ op).1
      if writer.getD o (-1) < 0 then writer := writer.set! o (Int.ofNat i)
      for r in AlgorithmLib.ML.TOp.reads op do
        nread := nread.set! r (nread.getD r 0 + 1)
        reader := reader.set! r (Int.ofNat i)
  let mut unitOf : Array Int := Array.replicate arr.size (-1)
  for (u, k) in vUnitsRaw.zip (List.range vUnitsRaw.length) do
    for i in u do unitOf := unitOf.set! i (Int.ofNat k)
  let mut acc : List (String × Nat) := []
  let mut tot := 0
  for r in [0:VNBUF] do
    if VBASE ≤ r && nread.getD r 0 == 1 && writer.getD r (-1) ≥ 0 then
      let w := (writer.getD r 0).toNat
      let c := (reader.getD r 0).toNat
      match arr[w]?, arr[c]? with
      | some a, some b =>
        if (vGemmOf a).isNone && (vGemmOf b).isNone
            && unitOf.getD w (-1) == unitOf.getD c (-2) then
          let slot := ((AlgorithmLib.ML.TOp.reads b).idxOf? r).getD 9
          let key := s!"{vKindOf a}->{vKindOf b}@{slot}"
          let kb := 2 * vBufBytes.getD r 0 / 1024
          tot := tot + kb
          if (acc.find? (fun p => p.1 == key)).isSome then
            acc := acc.map (fun p => if p.1 == key then (p.1, p.2 + kb) else p)
          else acc := acc ++ [(key, kb)]
      | _, _ => pure ()
  let top := ((acc.toArray.qsort (fun x y => x.2 > y.2)).toList.take 10).map
    (fun p => (p.1, p.2 / 1024))
  pure ([("MiB a register could carry", tot / 1024)] ++ top)

/-- The first and last tape positions a group covers, so a capture over a range
    of the tape can ask which groups fall inside it. -/
def vUnitLo (u : List Nat) : Nat := u.foldl min (u.head?.getD 0)
def vUnitHi (u : List Nat) : Nat := u.foldl max 0

/-- **How many streams the tape is issued across.**  A capture on one stream
    records a chain as long as the tape, so the graph's serial length is the
    launch count rather than the dependence depth. -/
def VNSTRM : Nat := 18

/-- **What every operation must follow**: the last writer of each buffer it
    reads, the last writer of the buffer it writes, and every reader of that
    buffer since — read-after-write, write-after-write and write-after-read.

    This is the whole of what the tape's order is for.  Two operations with no
    path between them here may be performed in either order or at once, which
    is what lets a capture record branches instead of a chain. -/
def vPredsRaw : Array (List Nat) := Id.run do
  let mut wr : Std.HashMap Nat Nat := {}
  let mut rd : Std.HashMap Nat (List Nat) := {}
  let mut out : Array (List Nat) := #[]
  for i in [0:VNUNITRAW] do
    let u := vUnitArrRaw.getD i []
    let ws := vUnitWrites u
    let rs := (vUnitReads u).filter (fun r => !(ws.contains r))
    let mut p : List Nat := []
    for r in rs do
      match wr[r]? with | some j => p := j :: p | none => pure ()
    for o in ws do
      p := p ++ rd.getD o []
      match wr[o]? with | some j => p := j :: p | none => pure ()
    out := out.push ((p.filter (fun j => j != i)).eraseDups)
    for r in rs do rd := rd.insert r (i :: rd.getD r [])
    for o in ws do
      wr := wr.insert o i
      rd := rd.insert o []
  pure out

/-- Longest path to each operation, which is the depth a chain cannot beat. -/
def vLevelRaw : Array Nat := Id.run do
  let mut lvs : Array Nat := #[]
  for i in [0:vPredsRaw.size] do
    let mut m := 0
    for j in vPredsRaw.getD i [] do m := max m (lvs.getD j 0)
    lvs := lvs.push (m + 1)
  pure lvs

/-- Where each buffer starts on the device: the running sum of the sizes. -/
def vBufAddr : Array Nat := Id.run do
  let mut a : Array Nat := #[]
  let mut off := 0
  for s in vBufBytes do
    a := a.push off
    off := off + s
  pure a

/-- **What makes two launches interchangeable**: the same dependence level and
    the same work.  For a proven kernel that is its slot, since the slot *is* the
    emitted text.  For a contraction it is the gemm's shape — every contraction
    shares slot zero, because a contraction renders no kernel, so keying on the
    slot would call two differently-shaped gemms one group. -/
def vHKey (i : Nat) : Nat × Nat × Nat × Nat × Nat :=
  let lv := vLevelRaw.getD i 0
  match (vUnitOps (vUnitArrRaw.getD i [])).head? >>= vGemmOf with
  | some (ta, tb, m, n, k, _, _, _) => (lv, 1, ta * 2 + tb, m, n * 100000 + k)
  | none => (lv, 0, vSlotIxRaw i, 0, 0)

/-- **Does the dependence graph actually order every conflicting pair?**

    The schedule issues `VNUNITRAW` launches across sixteen streams and synchronises
    only along `vPredsRaw`.  Everything that makes that equal to the tape's own order
    rests on `vPredsRaw` containing an edge for *every* read-after-write,
    write-after-write and write-after-read — and nothing states that it does.
    This is the claim under the whole session's speedup, so it is checked rather
    than assumed: for each unit and each buffer it touches, the most recent
    conflicting unit before it must be reachable through `vPredsRaw`.

    Reported as counts rather than a `Bool` so a failure says how many and where,
    instead of `false`. -/
def vDepsCheckWith (drop : Bool) : List (String × Nat) := Id.run do
  -- Plain arrays, no hashing: a `Std.HashMap` here would put
  -- `System.Platform.getNumBits` in the closure and widen the declared surface
  -- for a diagnostic, which is the wrong trade.
  let mut wrote : Array Nat := (List.replicate VNBUF vUnitsRaw.length).toArray
  let mut readBy : Array (List Nat) := (List.replicate VNBUF ([] : List Nat)).toArray
  let mut pairs := 0
  let mut missing := 0
  for i in [0:vUnitsRaw.length] do
    let u := vUnitArrRaw.getD i []
    let rs := vUnitReads u
    let ws := vUnitWrites u
    -- `drop` removes one edge per unit: the check must notice, or it is checking
    -- nothing.  Instantiating a guard at a value that should fail it is the only
    -- way to tell a real one from a decorative one.
    let ps := let q := vPredsRaw.getD i []; if drop then q.drop 1 else q
    let ordered := fun (j : Nat) => ps.contains j
    for r in rs do
      let w := wrote.getD r vUnitsRaw.length
      if w < vUnitsRaw.length then
        pairs := pairs + 1
        if !(ordered w) then missing := missing + 1
    for w in ws do
      let p := wrote.getD w vUnitsRaw.length
      if p < vUnitsRaw.length then
        pairs := pairs + 1
        if !(ordered p) then missing := missing + 1
      for rd in readBy.getD w [] do
        if rd != i then
          pairs := pairs + 1
          if !(ordered rd) then missing := missing + 1
    for r in rs do readBy := readBy.set! r ((readBy.getD r []) ++ [i])
    for w in ws do
      wrote := wrote.set! w i
      readBy := readBy.set! w []
  pure [("conflicting pairs", pairs), ("NOT ordered by vPredsRaw", missing)]

def vDepsCheck : List (String × Nat) := vDepsCheckWith false

/-- **…and it does.**  Build-enforced, so a scheduler change that drops an edge
    fails the build instead of producing a race that shows up as a wrong number
    once in a while.  This is the claim the sixteen-stream schedule rests on and
    the first thing this artifact states about itself. -/
def vDepsSound : Bool := (vDepsCheck.getD 1 ("", 0)).2 == 0

/-- Whether an operand may carry a byte offset, so a call can address a slice of
    a buffer.  For a contraction this is a pointer argument and not a change to
    the matrix it contracts, so `Law.cublasIsMatvec` is untouched — unlike
    batching, which `VendorKernel.assumes` deliberately leaves lawless. -/
def VOFFSETS : Nat := 1

/-- **Contractions that concatenate rather than batch.**

    `Law.cublasIsMatvec` is stated for every `rows`, so `P` matvecs sharing an
    input and differing only in their matrix are one *taller* matvec under the
    law already assumed — row `i` of the tall call is the fold row `i` of member
    `p` was.  That is a different move from `cublasSgemmStridedBatched`, which
    `VendorKernel.assumes` deliberately gives `[]`: batching would trade the
    stack's main equation for a lawless assumption, concatenating does not.

    So the question worth counting is how many contraction groups share their
    input operand. -/
def vGemmConcat : List (String × Nat) := Id.run do
  let keys := (List.range vUnitsRaw.length).map vHKey
  let ds := (keys.eraseDups).filter (fun k =>
    k.2.1 == 1 && (keys.filter (· == k)).length ≥ 2)
  let mut share := 0
  let mut saved := 0
  let mut tot := 0
  for k in ds do
    let mem := (List.range vUnitsRaw.length).filter (fun i => keys.getD i (0,0,0,0,0) == k)
    let ins := mem.filterMap (fun i =>
      ((vUnitOps (vUnitArrRaw.getD i [])).head? >>= vGemmOf).map (fun g => g.2.2.2.2.2.2.1))
    tot := tot + 1
    if ins.eraseDups.length == 1 then
      share := share + 1
      saved := saved + (mem.length - 1)
  pure [("contraction groups of 2+", tot), ("sharing one input", share),
        ("launches concatenating removes", saved)]

/-- **Material for horizontal fusion**: launches at one dependence level running
    the same kernel.  Merging `m` of them into one launch over `m` times the grid
    is what makes a kernel both bigger and wider, which is the pair of properties
    the measurements say matters.  It only stays affine in the chunk index if the
    buffers each launch binds are contiguous and in step, so that is counted
    separately from the opportunity itself. -/
def vHoriz : List (String × Nat) := Id.run do
  let keys := (List.range vUnitsRaw.length).map vHKey
  let ds := keys.eraseDups
  let sizes := ds.map (fun k => (keys.filter (· == k)).length)
  -- how many launches would disappear if every such group became one launch
  let saved := sizes.foldl (fun a m => a + (m - 1)) 0
  let big := (sizes.filter (· ≥ 2)).length
  let widest := sizes.foldl max 0
  pure [("groups of 2+", big), ("launches they would remove", saved),
        ("widest group", widest), ("launches now", vUnitsRaw.length)]

/-- An operation cuBLAS performs: it takes whole pointers, so it can neither
    read nor write a slice of a buffer. -/
def vIsVendor (o : AlgorithmLib.ML.TOp) : Bool :=
  match o with
  | .mv (.vendor _) _ _ _ _ _ _ _ => true
  | .mvT (.vendor _) _ _ _ _ _ _ => true
  | .outer (.vendor _) _ _ _ _ _ _ => true
  | _ => false

/-- The mergeable groups, as lists of unit indices. -/
def vHorizGroups : List (List Nat) :=
  let keys : List (Nat × Nat × Nat × Nat × Nat) :=
    (List.range vUnitsRaw.length).map vHKey
  let ds := (keys.eraseDups).filter (fun k => (keys.filter (· == k)).length ≥ 2)
  ds.map (fun k => (List.range vUnitsRaw.length).filter (fun i => keys.getD i (0,0,0,0,0) == k))


-- ---------------------------------------------------------------------------
-- Batched contractions: the launches, after grouping
-- ---------------------------------------------------------------------------

/-- **The narrowest contraction group to batch**, or `0` to batch none.

    A batch of `P` same-shaped contractions is one launch where the tape has
    `P`.  Back to back on the card it is a large win — `batch.cu`, `P separate
    | batched`, at this tape's four shapes:

    * qkv, `P=9`:            64.65 us | 26.85 us
    * scores, `P=3`:         16.27 us | 12.22 us
    * attn x values, `P=3`:  23.36 us | 30.45 us   <- batching loses
    * out projection, `P=3`: 15.51 us | 12.22 us

    The pointer-array and strided forms land within 1% of each other at every
    shape, which is why this uses the one that needs no adjacency: merging
    buffers to get a uniform stride would buy nothing.

    **In the schedule it is worth nothing, and mostly worse.** The step, same
    tape, same card:

    * batch nothing:           5.74 ms
    * batch the `P=9` groups:  5.79 ms
    * batch all 192 groups:    6.59 ms

    The reason is that the tape is issued across sixteen streams, so `P`
    separate contractions already overlap; a batch cannot overlap with itself,
    and it has to beat `P` *concurrent* launches rather than `P` sequential
    ones.  This is the fourth measurement in a row saying launch count is not
    what binds here.

    It also costs bits: a batched forward is not bit-equal to the unbatched one
    (`check_replay.py`), which is what moved `Law.cublasBatchedIsSomeReassoc`
    from the closed form to the weak one.

    So the default is off, and this stays because being able to *choose* it,
    with the price in launches, in bits and in laws all stated, is the whole
    point of a schedule menu. -/
def VBATCHMIN : Nat := 0



/-- **Contractions one batched call performs.**

    `vHKey` already keys a contraction on its dependence level and its gemm
    shape, so a group of it is a set of same-shaped contractions with no path
    between them — exactly what `cl_cublas_sgemm_batched_on_stream` takes.  A
    per-head projection is this: `head` builds q, k and v for each of three
    heads from one normed stream, so nine contractions of one shape sit at one
    level and are issued one at a time.

    Nothing about the members' *buffers* is required.  The pointer-array form
    names each member by pointer, so the members need share no allocation and
    each member's output stays a whole matrix of its own — which is why this
    needs neither a merged buffer nor a strided view of anyone's output, and
    why nothing downstream has to learn to read a slice. -/
def vBatchGroups : List (List Nat) :=
  if VBATCHMIN == 0 then [] else
  vHorizGroups.filter (fun g =>
    g.length ≥ VBATCHMIN && g.all (fun u =>
      match vUnitOps (vUnitArrRaw.getD u []) with
      | [op] => (vGemmOf op).isSome
      | _    => false))

/-- **The launches.**  A batched group is one entry holding all its members'
    operations; every other unit is itself.

    Collapsing here — before the dependence graph, not after it — is what keeps
    the schedule honest: `vPreds`, `vLevel` and `vDag` are computed over what
    actually launches, so a stream assignment or an event can never be derived
    for a launch that was then merged away.

    **The order is recomputed, not inherited.**  Merging `P` launches into one
    position moves the others, and a unit indexed between two members can depend
    on the earlier one — the bias add after head 0's projection does exactly
    that — so keeping the tape's order would run it too early.

    Sorting by dependence level is a topological order of the collapsed graph,
    and the collapsed graph is acyclic for a reason worth stating: an edge
    strictly raises the level, and `vHKey` keys a group on its level, so every
    member of a group sits at one level.  A cycle would need that level to be
    both above and below another group's.  Ties keep the tape's order, so the
    schedule is the tape's wherever the levels do not force otherwise. -/
def vUnits : List (List Nat) := Id.run do
  let mut inG : Array Bool := Array.replicate vUnitsRaw.length false
  for g in vBatchGroups do
    for u in g do inG := inG.set! u true
  let mut nodes : Array (Nat × Nat × List Nat) := #[]
  for g in vBatchGroups do
    let lo := (g.head?).getD 0
    nodes := nodes.push (vLevelRaw.getD lo 0, lo, g)
  for i in [0:vUnitsRaw.length] do
    if !(inG.getD i false) then
      nodes := nodes.push (vLevelRaw.getD i 0, i, [i])
  let sorted := nodes.qsort (fun x y => x.1 < y.1 || (x.1 == y.1 && x.2.1 < y.2.1))
  pure (sorted.toList.map (fun n => n.2.2.flatMap (fun u => vUnitArrRaw.getD u [])))

def vUnitArr : Array (List Nat) := vUnits.toArray
def VNUNIT : Nat := vUnitArr.size

/-- How many contractions the launch at `k` performs: one, unless it is a
    batch. -/
def vBatchOf (k : Nat) : Nat :=
  let ops := vUnitOps (vUnitArr.getD k [])
  if ops.length ≥ 2 && ops.all (fun op => (vGemmOf op).isSome) then ops.length else 1

/-- The distinct kernels and each launch's slot, over the launches rather than
    over the pre-batch units.  A contraction renders no kernel either way, so
    batching changes which index a slot is filed under and not the set of
    texts. -/
def vSlotMap : Array Nat × Array String := Id.run do
  let mut seen : Std.HashMap (Nat × Nat × AlgorithmLib.ML.EWStmt) Nat := {}
  let mut seenText : Std.HashMap String Nat := {}
  let mut ix : Array Nat := #[]
  let mut uniq : Array String := #[]
  for u in vUnits do
    if ((vUnitOps u).head? >>= vGemmOf).isSome then
      match seenText[""]? with
      | some j => ix := ix.push j
      | none =>
          let j := uniq.size
          seenText := seenText.insert "" j
          uniq := uniq.push ""
          ix := ix.push j
    else
    let k := ((vUnitBufs u).length, vWarpsOf (vUnitGrid u), vUnitStmt u)
    match seen[k]? with
    | some j => ix := ix.push j
    | none =>
        let t := vTextOf u
        match seenText[t]? with
        | some j => seen := seen.insert k j; ix := ix.push j
        | none =>
            let j := uniq.size
            seen := seen.insert k j
            seenText := seenText.insert t j
            uniq := uniq.push t
            ix := ix.push j
  pure (ix, uniq)

def vSlotIx (k : Nat) : Nat := vSlotMap.1.getD k 0

def vPtx : List String := vSlotMap.2.toList
def VNSLOT : Nat := vPtx.length

/-- Slots are packed to the text they hold, aligned to 256 bytes.  A uniform
    stride would have to be the widest kernel's, and the widest here is thirty
    times the median — the adjoint of a wide contraction unrolls further than a
    row pass does. -/
def vSlotSizes : List Nat :=
  vPtx.map (fun t => ((t.toUTF8.toList.length + 1) + 255) / 256 * 256)
def VPTX_BYTES : Nat := vSlotSizes.foldl (· + ·) 0

-- ---------------------------------------------------------------------------
-- Seam guards on the kernels that are actually emitted
-- ---------------------------------------------------------------------------

/-! A launch here is a *group* of operations sharing one buffer table, not a
    single operation, so `TOp.localStmt`'s guards do not describe what is
    printed.  `TOp.groupStmt` is what is printed, and `TOp.groupRegsOk` is the
    same argument at group scale: registers take the group's maximum, because
    each member numbers its own from zero, and instructions its sum.

    Only the two arithmetic facts are decided over the tape.  Everything that
    would have to build an instruction list is generic. -/

/-- The dependences of the launches, by the definition `vPredsRaw` uses. -/
def vPreds : Array (List Nat) := Id.run do
  let mut wr : Std.HashMap Nat Nat := {}
  let mut rd : Std.HashMap Nat (List Nat) := {}
  let mut out : Array (List Nat) := #[]
  for i in [0:VNUNIT] do
    let u := vUnitArr.getD i []
    let ws := vUnitWrites u
    let rs := (vUnitReads u).filter (fun r => !(ws.contains r))
    let mut p : List Nat := []
    for r in rs do
      match wr[r]? with | some j => p := j :: p | none => pure ()
    for o in ws do
      p := p ++ rd.getD o []
      match wr[o]? with | some j => p := j :: p | none => pure ()
    out := out.push ((p.filter (fun j => j != i)).eraseDups)
    for r in rs do rd := rd.insert r (i :: rd.getD r [])
    for o in ws do
      wr := wr.insert o i
      rd := rd.insert o []
  pure out

def vLevel : Array Nat := Id.run do
  let mut lvs : Array Nat := #[]
  for i in [0:vPreds.size] do
    let mut m := 0
    for j in vPreds.getD i [] do m := max m (lvs.getD j 0)
    lvs := lvs.push (m + 1)
  pure lvs

/-- **Pointer arrays.**  Three per batched launch — one for each operand — each
    holding one device pointer per member.  They are filled once, after the
    buffers exist and before anything launches, because a device pointer does
    not move. -/
def vBatchUnits : List Nat :=
  (List.range VNUNIT).filter (fun k => vBatchOf k ≥ 2)

def VNPARR : Nat := 3 * vBatchUnits.length

/-- Which of the three arrays of batched launch `k` a slot holds. -/
def vParrIx (k : Nat) : Nat :=
  ((vBatchUnits.zip (List.range vBatchUnits.length)).find? (fun p => p.1 == k)).map
    Prod.snd |>.getD 0


/-- **Which mergeable groups need no operand offset at all.**

    A proven kernel's output address is affine in its chunk index, so merging `P`
    members into one launch over `P` times the grid writes them stacked *by
    construction* — no offset is involved on the output side.  The question is
    only the inputs: a buffer every member reads is already shared, and a buffer
    each member reads its own copy of is fine exactly when those copies are
    themselves the stacked output of one other merged group.

    A group failing this reads something a vendor call produced one buffer at a
    time, and that is where an operand offset becomes unavoidable. -/
def vSelfContained : List (String × Nat) := Id.run do
  let gs := vHorizGroups
  let NONE := gs.length          -- a group index no group has
  let mut owner : Array Nat := (List.replicate vUnitsRaw.length NONE).toArray
  for gi in [0:gs.length] do
    for u in gs.getD gi [] do owner := owner.set! u gi
  let mut writer : Std.HashMap Ref Nat := {}
  for i in [0:vUnitsRaw.length] do
    for r in vUnitWrites (vUnitArrRaw.getD i []) do writer := writer.insert r i
  let mut ok := 0
  let mut saved := 0
  let mut vendorFed := 0
  for gi in [0:gs.length] do
    let mem := gs.getD gi []
    let readsOf := fun i => vUnitReads (vUnitArrRaw.getD i [])
    let mut good := true
    let mut byVendor := false
    match mem with
    | [] => pure ()
    | m0 :: _ =>
      for j in [0:(readsOf m0).length] do
        let per := mem.map (fun i => (readsOf i).getD j 0)
        if per.eraseDups.length == 1 then pure ()      -- shared by every member
        else
          -- each member has its own: they must be one other group's stacked output
          let owners := (per.map (fun r =>
            match writer[r]? with
            | some u => owner.getD u NONE
            | none   => NONE)).eraseDups
          match owners with
          | [o] =>
            if o == NONE then
              good := false
              byVendor := true
          | _ => good := false
      if good then
        ok := ok + 1
        saved := saved + (mem.length - 1)
      else if byVendor then
        vendorFed := vendorFed + 1
  pure [("mergeable groups", gs.length), ("need no operand offset", ok),
        ("launches those remove", saved),
        ("blocked by a vendor-written operand", vendorFed)]

/-- **The merges that need no operand offset, closed under both directions.**

    Buffers are separate allocations referenced by handle, so a merged group's
    `P` outputs have to *become one buffer* rather than land next to each other.
    That is free for the merged launch itself — its chunk index already runs the
    whole range — but it means every reader of a member's output must also be a
    merged launch reading the whole buffer, and symmetrically every per-member
    input must be a merged group's whole output.

    So the mergeable set is a greatest fixed point, not a filter: start from the
    groups whose inputs come from one other group and drop any whose producer or
    consumer is not itself kept, until nothing more drops.  What survives can be
    merged with the layout alone; what drops needs an operand offset. -/
def vHorizClosed : List Nat × List (String × Nat) := Id.run do
  let gs := vHorizGroups
  let NONE := gs.length
  let mut owner : Array Nat := (List.replicate vUnitsRaw.length NONE).toArray
  for gi in [0:gs.length] do
    for u in gs.getD gi [] do owner := owner.set! u gi
  let mut writer : Std.HashMap Ref Nat := {}
  let mut readers : Std.HashMap Ref (List Nat) := {}
  for i in [0:vUnitsRaw.length] do
    for r in vUnitWrites (vUnitArrRaw.getD i []) do writer := writer.insert r i
    for r in vUnitReads (vUnitArrRaw.getD i []) do
      readers := readers.insert r ((readers.getD r []) ++ [i])
  -- A contraction group is mergeable only by *concatenation*, which needs its
  -- members to share an input; batching them instead would trade
  -- `Law.cublasIsMatvec` for the lawless `cublasSgemmStridedBatched`, so a
  -- group that does not share an input is not a merge this development may make.
  let concatable := fun (gi : Nat) =>
    let mem := gs.getD gi []
    match mem.head? >>= (fun i => (vUnitOps (vUnitArrRaw.getD i [])).head?) >>= vGemmOf with
    | none => true
    | some _ =>
      let ins := mem.filterMap (fun i =>
        ((vUnitOps (vUnitArrRaw.getD i [])).head? >>= vGemmOf).map (fun g => g.2.2.2.2.2.2.1))
      ins.eraseDups.length == 1
  let mut keep : Array Bool :=
    ((List.range gs.length).map concatable).toArray
  let mut changed := true
  let mut rounds := 0
  while changed && rounds < 40 do
    changed := false
    rounds := rounds + 1
    for gi in [0:gs.length] do
      if keep.getD gi false then
        let mem := gs.getD gi []
        let readsOf := fun i => vUnitReads (vUnitArrRaw.getD i [])
        let mut good := true
        match mem with
        | [] => good := false
        | m0 :: _ =>
          -- inputs: shared, or one kept group's whole output at the same width
          for j in [0:(readsOf m0).length] do
            let per := mem.map (fun i => (readsOf i).getD j 0)
            if per.eraseDups.length == 1 then pure ()
            else
              let os := (per.map (fun r =>
                match writer[r]? with
                | some u => owner.getD u NONE
                | none   => NONE)).eraseDups
              match os with
              | [o] =>
                if o == NONE then
                  -- not a merged group: fine when every producer can write at an
                  -- offset into the fused buffer.  A proven kernel can already
                  -- (`.rowOf n k`); a contraction can once its operands carry a
                  -- byte offset, which changes the pointer and not the matrix,
                  -- so `Law.cublasIsMatvec` still says what it computes.
                  if per.any (fun r =>
                       match writer[r]? with
                       | some _ => VOFFSETS == 0
                       | none   => true) then good := false
                else if VOFFSETS == 0 && (!(keep.getD o false)
                     || (gs.getD o []).length != mem.length) then good := false
              | _ => if VOFFSETS == 0 then good := false
          -- outputs: every reader of a member's write is in this same group's
          -- consumer, and that consumer is kept and equally wide
          for i in mem do
            for r in vUnitWrites (vUnitArrRaw.getD i []) do
              for c in readers.getD r [] do
                let o := owner.getD c NONE
                if o == NONE then
                  if VOFFSETS == 0 && (vUnitOps (vUnitArrRaw.getD c [])).any vIsVendor then
                    good := false
                else if VOFFSETS == 0 && (!(keep.getD o false)
                     || (gs.getD o []).length != mem.length) then good := false
        if !good then
          keep := keep.set! gi false
          changed := true
  let kept := (List.range gs.length).filter (fun gi => keep.getD gi false)
  let saved := kept.foldl (fun a gi => a + ((gs.getD gi []).length - 1)) 0
  pure (kept, [("mergeable groups", gs.length), ("closed under both directions", kept.length),
               ("launches they remove", saved), ("fixed-point rounds", rounds)])

/-- **Can this operation read an operand at an offset into a bigger buffer?**

    A merged group writes its fused output as one buffer and needs no offset for
    that — its chunk index already runs the whole range.  What decides whether
    the merge is possible is the *readers*: a row pass reaches its operands
    through a `BCast`, and `.rowOf s k` evaluates to `cta * s + k + o`, so `k`
    shifts the base and a slice is expressible.  An elementwise pass reads at the
    bare element address with nowhere to put a `k`, and `rowsq`/`rowmax` walk a
    fixed `a * n + t`.  A contraction can since its operands carry `off_a`/`off_b`.

    This is what the landed FFI offsets bought and what they did not. -/
def vCanReadOffset (op : AlgorithmLib.ML.TOp) : Bool :=
  match op with
  | .ziprow .. | .ziprow3 .. | .ziprow4 .. | .rowdot .. | .rowdot4 .. => true
  | .mv .. | .mvT .. | .outer .. => true
  | _ => false

/-- Of the launches that read what a merged group would fuse, how many could
    address their slice. -/
def vReaderCensus : List (String × Nat) := Id.run do
  let gs := vHorizGroups
  let mut writer : Array Nat := (List.replicate VNBUF vUnitsRaw.length).toArray
  for i in [0:vUnitsRaw.length] do
    for r in vUnitWrites (vUnitArrRaw.getD i []) do writer := writer.set! r i
  let mut inGroup : Array Bool := (List.replicate vUnitsRaw.length false).toArray
  for g in gs do
    for u in g do inGroup := inGroup.set! u true
  -- every buffer a group member writes would become a slice of a fused buffer
  let mut fused : Array Bool := (List.replicate VNBUF false).toArray
  for g in gs do
    if g.length ≥ 2 then
      for u in g do
        for w in vUnitWrites (vUnitArrRaw.getD u []) do fused := fused.set! w true
  let mut ok := 0
  let mut bad := 0
  let mut merged := 0
  for i in [0:vUnitsRaw.length] do
    let u := vUnitArrRaw.getD i []
    if (vUnitReads u).any (fun r => fused.getD r false) then
      if inGroup.getD i false then merged := merged + 1
      else if (vUnitOps u).all vCanReadOffset then ok := ok + 1
      else bad := bad + 1
  pure [("readers of a fused buffer", ok + bad + merged),
        ("themselves merged", merged),
        ("can address a slice", ok),
        ("CANNOT -- block the merge", bad)]

/-- **The groups actually selected for one launch.**

    Conservative on purpose, because every condition here is one the emission
    does not have to re-check: proven kernels only (a contraction merges by a
    different route), every member binding the same number of buffers, and each
    bind position either shared by every member or held by distinct buffers of
    equal size.  Equal size is what makes member `p`'s slice start at exactly
    `p * size`, which is what makes the merged launch's chunk index land on it
    with no offset at all.

    A group is dropped if anything outside it reads a member's output through
    addressing that cannot shift — that reader would have to find its slice, and
    `BCast.shift` refuses rather than approximating. -/
def vMergeSel : List (List Nat) := Id.run do
  let mut writer : Array Nat := (List.replicate VNBUF vUnitsRaw.length).toArray
  for i in [0:vUnitsRaw.length] do
    for r in vUnitWrites (vUnitArrRaw.getD i []) do writer := writer.set! r i
  let mut inG : Array Bool := (List.replicate vUnitsRaw.length false).toArray
  for g in vHorizGroups do
    for u in g do inG := inG.set! u true
  let mut out : List (List Nat) := []
  for g in vHorizGroups do
    if g.length ≥ 2 && g.all (fun u => !((vUnitOps (vUnitArrRaw.getD u [])).any vIsVendor)) then
      let bss := g.map (fun u => vUnitBufs (vUnitArrRaw.getD u []))
      let L := (bss.head?.getD []).length
      let shapeOk := bss.all (fun b => b.length == L) &&
        (List.range L).all (fun j =>
          let col := bss.map (fun b => b.getD j 0)
          col.eraseDups.length == 1 ||
            (col.eraseDups.length == col.length &&
             (col.map (fun r => vBufBytes.getD r 0)).eraseDups.length == 1))
      -- Everything outside the group that reads a buffer the fusion *moves*
      -- must be able to find its slice.  The moved ones are every bind position
      -- held by distinct buffers, minus the first, which stays put at offset
      -- zero -- inputs as much as outputs, which is what an earlier version of
      -- this condition missed: it looked only at what members wrote, and the
      -- `vUnshiftableReads` counter reported 85 addresses that would have been
      -- silently wrong.
      let moved : List Ref := (List.range L).flatMap (fun j =>
        let col := bss.map (fun b => b.getD j 0)
        if col.eraseDups.length == col.length then col.drop 1 else [])
      let readersOk := (List.range vUnitsRaw.length).all (fun c =>
        g.contains c ||
        !((vUnitReads (vUnitArrRaw.getD c [])).any (fun r => moved.contains r)) ||
        (vUnitOps (vUnitArrRaw.getD c [])).all vCanReadOffset)
      if shapeOk && readersOk then out := out ++ [g]
  pure out

/-- Why a mergeable group was not selected, counted.  Whether the remaining work
    is loosening a condition or another library change depends entirely on which
    line here is large. -/
def vSelReasons : List (String × Nat) := Id.run do
  let mut vendor := 0
  let mut shape := 0
  let mut readers := 0
  let mut ok := 0
  for g in vHorizGroups do
    if g.length < 2 then pure ()
    else if g.any (fun u => (vUnitOps (vUnitArrRaw.getD u [])).any vIsVendor) then
      vendor := vendor + 1
    else
      let bss := g.map (fun u => vUnitBufs (vUnitArrRaw.getD u []))
      let L := (bss.head?.getD []).length
      let shapeOk := bss.all (fun b => b.length == L) &&
        (List.range L).all (fun j =>
          let col := bss.map (fun b => b.getD j 0)
          col.eraseDups.length == 1 ||
            (col.eraseDups.length == col.length &&
             (col.map (fun r => vBufBytes.getD r 0)).eraseDups.length == 1))
      let moved : List Ref := (List.range L).flatMap (fun j =>
        let col := bss.map (fun b => b.getD j 0)
        if col.eraseDups.length == col.length then col.drop 1 else [])
      let readersOk := (List.range vUnitsRaw.length).all (fun c =>
        g.contains c ||
        !((vUnitReads (vUnitArrRaw.getD c [])).any (fun r => moved.contains r)) ||
        (vUnitOps (vUnitArrRaw.getD c [])).all vCanReadOffset)
      if !shapeOk then shape := shape + 1
      else if !readersOk then readers := readers + 1
      else ok := ok + 1
  pure [("contains a contraction", vendor), ("bind shape mismatch", shape),
        ("a reader cannot shift", readers), ("selected", ok)]

/-- Where each buffer lands once the selected groups are fused: the buffer that
    absorbs it, and how many elements into that buffer it starts. -/
def vFuseMap : Array (Ref × Nat) := Id.run do
  let mut mp : Array (Ref × Nat) := ((List.range VNBUF).map (fun r => (r, 0))).toArray
  for g in vMergeSel do
    let bss := g.map (fun u => vUnitBufs (vUnitArrRaw.getD u []))
    let L := (bss.head?.getD []).length
    for j in [0:L] do
      let col := bss.map (fun b => b.getD j 0)
      if col.eraseDups.length == col.length then
        let base := col.getD 0 0
        let sz := (vBufBytes.getD base 0) / 4
        for p in [0:col.length] do
          mp := mp.set! (col.getD p 0) (base, p * sz)
  pure mp

def vFuseReport : List (String × Nat) :=
  let fused := (List.range VNBUF).filter (fun r =>
    let (b, _) := vFuseMap.getD r (r, 0); b != r)
  [ ("groups selected", vMergeSel.length)
  , ("launches they remove", vMergeSel.foldl (fun a g => a + (g.length - 1)) 0)
  , ("buffers absorbed into another", fused.length)
  , ("largest fused buffer, KiB",
      ((List.range VNBUF).map (fun b =>
        ((List.range VNBUF).filter (fun r => (vFuseMap.getD r (r,0)).1 == b)).length
          * vBufBytes.getD b 0)).foldl max 0 / 1024) ]

/-- Where a buffer ended up. -/
def vFuseOf (r : Ref) : Ref × Nat := vFuseMap.getD r (r, 0)

/-- Which units are members of a selected merge group. -/
def vInMerge : Array Bool := Id.run do
  let mut a : Array Bool := (List.replicate vUnitsRaw.length false).toArray
  for g in vMergeSel do
    for u in g do a := a.set! u true
  pure a

/-- **An operation as it addresses the fused buffers.**

    A member of a merged group needs no shift: the merged launch runs its chunk
    index over the whole fused range, so at `cta = p*grid + r` the addressing it
    already has lands on member `p`'s slice.  Everything *else* reading a fused
    buffer is reaching into the middle of it and shifts by the offset.

    `shift` refuses on `.scalar`, and `vShiftFailures` counts the refusals rather
    than letting `getD` paper over one — a silent wrong address here is a wrong
    number, not a crash. -/
def vShiftB (member : Bool) (m : AlgorithmLib.ML.BCast) (r : Ref) :
    AlgorithmLib.ML.BCast :=
  if member then m else (m.shift (vFuseOf r).2).getD m

def vOpFused (member : Bool) (op : AlgorithmLib.ML.TOp) : AlgorithmLib.ML.TOp :=
  let f := fun r => (vFuseOf r).1
  match op with
  | .ziprow a b o fx mA mB n k w rows =>
      .ziprow (f a) (f b) (f o) fx (vShiftB member mA a) (vShiftB member mB b) n k w rows
  | .ziprow3 a b c o fx mA mB mC n k w rows =>
      .ziprow3 (f a) (f b) (f c) (f o) fx (vShiftB member mA a) (vShiftB member mB b)
        (vShiftB member mC c) n k w rows
  | .ziprow4 a b c d o fx mA mB mC mD n k w rows =>
      .ziprow4 (f a) (f b) (f c) (f d) (f o) fx (vShiftB member mA a)
        (vShiftB member mB b) (vShiftB member mC c) (vShiftB member mD d) n k w rows
  | .rowdot4 a b c d o fx mA mB mC mD n rows =>
      .rowdot4 (f a) (f b) (f c) (f d) (f o) fx (vShiftB member mA a)
        (vShiftB member mB b) (vShiftB member mC c) (vShiftB member mD d) n rows
  | .rowdot a b o mA mB n rows =>
      .rowdot (f a) (f b) (f o) (vShiftB member mA a) (vShiftB member mB b) n rows
  | .ew1 fx a o g => .ew1 fx (f a) (f o) g
  | .ew2 fx a b o g => .ew2 fx (f a) (f b) (f o) g
  | .ew3 fx a b c o g => .ew3 fx (f a) (f b) (f c) (f o) g
  | .ew4 fx a b c d o g => .ew4 fx (f a) (f b) (f c) (f d) (f o) g
  | .upd2 fx a b g => .upd2 fx (f a) (f b) g
  | .rowsq a o n rows => .rowsq (f a) (f o) n rows
  | .rowmax a o n rows i => .rowmax (f a) (f o) n rows i
  | .smce a b c o g => .smce (f a) (f b) (f c) (f o) g
  | .mv bk a b o bt i ow bA => .mv bk (f a) (f b) (f o) bt i ow bA
  | .mvT bk a b o bt i ow => .mvT bk (f a) (f b) (f o) bt i ow
  | .outer bk a b o bt i ow => .outer bk (f a) (f b) (f o) bt i ow

/-- Shifts `BCast.shift` refused: every one is an address the fusion would get
    wrong, so this must be zero for the plan to be usable. -/
def vShiftFailures : Nat := Id.run do
  let mut n := 0
  for i in [0:vUnitsRaw.length] do
    if !(vInMerge.getD i false) then
      for op in vUnitOps (vUnitArrRaw.getD i []) do
        let chk := fun (m : AlgorithmLib.ML.BCast) (r : Ref) =>
          if (vFuseOf r).2 != 0 && (m.shift (vFuseOf r).2).isNone then 1 else 0
        match op with
        | .ziprow a b _ _ mA mB _ _ _ _ => n := n + chk mA a + chk mB b
        | .ziprow3 a b c _ _ mA mB mC _ _ _ _ => n := n + chk mA a + chk mB b + chk mC c
        | .ziprow4 a b c d _ _ mA mB mC mD _ _ _ _ =>
            n := n + chk mA a + chk mB b + chk mC c + chk mD d
        | .rowdot4 a b c d _ _ mA mB mC mD _ _ =>
            n := n + chk mA a + chk mB b + chk mC c + chk mD d
        | .rowdot a b _ mA mB _ _ => n := n + chk mA a + chk mB b
        | _ => pure ()
  pure n

/-- An operation outside every merged group that would have to read a fused
    buffer through addressing with no offset at all. -/
def vUnshiftableReads : Nat := Id.run do
  let mut n := 0
  for i in [0:vUnitsRaw.length] do
    if !(vInMerge.getD i false) then
      for op in vUnitOps (vUnitArrRaw.getD i []) do
        if !vCanReadOffset op then
          for r in AlgorithmLib.ML.TOp.reads op do
            if (vFuseOf r).2 != 0 then n := n + 1
  pure n

/-- Of the mergeable groups, how many are already laid out so that member `p`'s
    data sits exactly `p` buffer-lengths along: then the `p` buffers are one
    bigger buffer, the merged launch's address stays affine in the chunk index,
    and no operand needs a byte offset. -/
def vHorizContig : List (String × Nat) := Id.run do
  let keys := (List.range vUnitsRaw.length).map vHKey
  let ds := (keys.eraseDups).filter (fun k => (keys.filter (· == k)).length ≥ 2)
  let mut ok := 0
  let mut saved := 0
  let mut savedAll := 0
  for k in ds do
    let members := (List.range vUnitsRaw.length).filter (fun i => keys.getD i (0,0,0,0,0) == k)
    let bufsOf := fun i => vUnitBufs (vUnitsRaw.getD i [])
    savedAll := savedAll + (members.length - 1)
    match members with
    | [] => pure ()
    | m0 :: _ =>
      let b0 := bufsOf m0
      let stacked := (List.range members.length).all (fun p =>
        let bp := bufsOf (members.getD p m0)
        bp.length == b0.length &&
        (List.range b0.length).all (fun j =>
          let r0 := b0.getD j 0
          let rp := bp.getD j 0
          let sz := vBufBytes.getD r0 0
          -- either every member reads the same buffer, or member p's is exactly
          -- p lengths along from member 0's
          (rp == r0) || (vBufAddr.getD rp 0 == vBufAddr.getD r0 0 + p * sz)))
      if stacked then
        ok := ok + 1
        saved := saved + (members.length - 1)
  pure [("mergeable groups", ds.length), ("stacked already", ok),
        ("launches those would remove", saved),
        ("launches all groups would remove", savedAll)]


/-- **The tape dealt onto `VNSTRM` streams, with the events that keep it
    equivalent to the tape's own order.**

    An operation continues its latest predecessor's stream when that stream is
    otherwise idle, which costs no event at all; failing that it goes to the
    least loaded stream that is not already occupied at this depth.  A
    predecessor left on another stream then needs one edge — unless the two
    streams have already been synchronised at or past that point, which
    `cov` tracks, since ordering after a later operation orders after every
    earlier one on the same stream.

    Returns the stream per operation, the event each operation records (when
    some later operation on another stream needs it), the events each waits on
    before launching, and how many events that is. -/
structure VDag where
  strm : Array Nat
  erec : Array (Option Nat)
  ewait : Array (List Nat)
  /-- The operation each event is recorded after.  A range that begins past it
      must drop the wait: the fork already put the pool behind everything the
      capture stream had done, and waiting on an event no operation in the
      capture records is what invalidates one. -/
  eown : Array Nat
  nev : Nat

def vDag : VDag := Id.run do
  let n := VNUNIT
  let mut strm : Array Nat := Array.replicate n 0
  let mut lastOn : Array Int := Array.replicate VNSTRM (-1)
  let mut lastLv : Array Nat := Array.replicate VNSTRM 0
  let mut load : Array Nat := Array.replicate VNSTRM 0
  for i in [0:n] do
    let ps := vPreds.getD i []
    let cand : Int := ps.foldl (fun m j => max m (Int.ofNat j)) (-1)
    let mut s := 0
    let mut chosen := false
    if cand ≥ 0 then
      let sc := strm.getD cand.toNat 0
      if lastOn.getD sc (-1) == cand then
        s := sc; chosen := true
    if !chosen then
      let mut best : Option Nat := none
      for t in [0:VNSTRM] do
        if lastLv.getD t 0 < vLevel.getD i 0 then
          match best with
          | none => best := some t
          | some b => if load.getD t 0 < load.getD b 0 then best := some t
      s := match best with
           | some t => t
           | none => if cand ≥ 0 then strm.getD cand.toNat 0 else 0
    strm := strm.set! i s
    lastOn := lastOn.set! s (Int.ofNat i)
    lastLv := lastLv.set! s (vLevel.getD i 0)
    load := load.set! s (load.getD s 0 + 1)
  -- **What each stream is already ordered after**, as one index per stream: a
  -- vector clock.  Waiting for an operation orders you after everything that
  -- operation was itself ordered after, so an edge is emitted only when the
  -- predecessor is not already implied — which is most of them.
  let mut clock : Array (Array Int) := Array.replicate VNSTRM (Array.replicate VNSTRM (-1))
  let mut snap : Array (Array Int) := Array.replicate n #[]
  let mut erec : Array (Option Nat) := Array.replicate n none
  let mut ewait : Array (List Nat) := Array.replicate n []
  let mut eown : Array Nat := #[]
  let mut nev := 0
  for i in [0:n] do
    let si := strm.getD i 0
    -- Latest predecessor first: ordering after it orders after every earlier
    -- one on that same stream, so the earlier ones then need no edge.
    for p in ((vPreds.getD i []).toArray.qsort (fun a b => a > b)).toList do
      let sp := strm.getD p 0
      if sp != si && Int.ofNat p > (clock.getD si #[]).getD sp (-1) then
        let mut e := nev
        match erec.getD p none with
        | some j => e := j
        | none =>
          erec := erec.set! p (some nev)
          eown := eown.push p
          nev := nev + 1
        ewait := ewait.set! i (e :: ewait.getD i [])
        let sp' := snap.getD p #[]
        let mut ci := clock.getD si #[]
        for x in [0:VNSTRM] do
          ci := ci.set! x (max (ci.getD x (-1)) (sp'.getD x (-1)))
        clock := clock.set! si ci
    clock := clock.set! si ((clock.getD si #[]).set! si (Int.ofNat i))
    snap := snap.set! i (clock.getD si #[])
  pure { strm, erec, ewait, eown, nev }

def VHOST_LEN_OFF : Nat := 0x0080
/-- The stream a capture runs on, and the graph each capture leaves. -/
def VSTREAM_OFF : Nat := 0x0090
def VGRAPH_OFF : Nat := 0x0094
def VGRAPH_STEP_OFF : Nat := 0x0098

/-- The pooled streams, one `u32` each. -/
def VPOOL_OFF : Nat := 0x00A0
def vPoolOff (s : Nat) : Nat := VPOOL_OFF + 4 * s

/-- The forward and the backward-with-updates, each captured as the graph their
    dependences allow rather than as the chain their listing is.

    Derived from where the pool ends, because a fixed address here is one the
    pool grows into: at sixteen streams the pool reaches `0xE0`, and a graph
    handle written at `0xB8` overwrites a stream.  It aliased quietly for a
    while — a graph id and a stream id are both small integers, so the launches
    still went somewhere — which is exactly why it is derived now. -/
def VGRAPH_DFWD_OFF : Nat := VPOOL_OFF + 4 * VNSTRM
def VGRAPH_DSTEP_OFF : Nat := VGRAPH_DFWD_OFF + 4
/-- Graph slots for the two single-class captures that split the step's time. -/
def VGRAPH_BLAS_OFF : Nat := VGRAPH_DSTEP_OFF + 4
def VGRAPH_ROW_OFF : Nat := VGRAPH_BLAS_OFF + 4

def VPTX_OFF : Nat := 0x0100
def vSlotOff (i : Nat) : Nat := VPTX_OFF + (vSlotSizes.take i).foldl (· + ·) 0
def VBIND_OFF : Nat := VPTX_OFF + VPTX_BYTES
def vBindOff (i : Nat) : Nat := VBIND_OFF + 4 * i
/-- The pointer arrays' buffer ids, one slot each, three per batched launch. -/
def VPARR_OFF : Nat := VBIND_OFF + 4 * VNBUF
def vParrOff (i : Nat) : Nat := VPARR_OFF + 4 * i
def VLOCAL_OFF : Nat := VPARR_OFF + 4 * VNPARR
/-- Events: one to fork the pool and one per stream to join it, then one for
    each operation the schedule has to publish to another stream. -/
def VNEVENT : Nat := 2 * VNSTRM + vDag.nev
/-- Where the schedule's own events start. -/
def vDagEvent (e : Nat) : Nat := 2 * VNSTRM + e
/-- Slots in the per-launch buffer table: as many as the widest group binds.
    A single operation never needed more than five; a fused group does, and
    eight was the fixed size that used to be enough. -/
def VLOCAL_SLOTS : Nat :=
  ((List.range VNUNIT).filter (fun k => ((vUnitOps (vUnitArr.getD k [])).head?
      >>= vGemmOf).isNone)).foldl
    (fun m k => max m (vUnitBufs (vUnitArr.getD k [])).length) 8
def VEVENT_OFF : Nat := VLOCAL_OFF + 4 * VLOCAL_SLOTS
def vEventOff (e : Nat) : Nat := VEVENT_OFF + 4 * e
/-- The padding mask, `SK` floats, shipped in the artifact itself.

    It goes before the gradient map, not after: the map's own contract is that
    it is the *last* region, and a host finds it by counting back from the end
    of memory. -/
def VMASK_OFF : Nat := VEVENT_OFF + 4 * VNEVENT

/-- Where each input's gradient landed, as one `u32` per input, so a host reads
    the derivation's own allocation instead of guessing at it.  It is the last
    region, which is the whole of how a host finds it: `4 * VBASE` bytes ending
    `0x100` before the end of memory. -/
def VGMAP_OFF : Nat := VMASK_OFF + 4 * SK
def VMEM_SIZE : Nat := VGMAP_OFF + 4 * VBASE + 0x100

def vSlotBytes (t : String) : List UInt8 :=
  let b := t.toUTF8.toList ++ [0]
  b ++ zeros (((b.length + 255) / 256 * 256) - b.length)
