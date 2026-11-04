module
public import Lean
public import Std
public import AlgorithmLib.ML.Kernel.Library
meta import AlgorithmLib.ML.Kernel.Library
public import AlgorithmLib.ML.Ptx.Body
meta import AlgorithmLib.ML.Ptx.Body
public import AlgorithmLib.ML.Machine.Geometry
meta import AlgorithmLib.ML.Machine.Geometry
public import AlgorithmLib.Surface.FFI
meta import AlgorithmLib.Surface.FFI
public import AlgorithmLib.Core.IR
meta import AlgorithmLib.Core.IR
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open Lean AlgorithmLib AlgorithmLib.IR AlgorithmLib.ML

/-!
  # gpt-oss-20b attention, in the proven machine

  The other half of a layer. Where the experts needed a new instruction set —
  nibbles, shifts, an assembled `Float32` — attention needs none: it is
  reductions, a rotation and a gather, which is exactly what `ML/` proves
  kernels in. So these are written as `EWStmt` and carry the soundness
  theorems every `EWStmt` kernel does, one line each.

  Three things here are not in that stack, and each is small:

  * **Grouped queries.** Eight query heads share a key head. That is an
    addressing fact and it lives entirely in the batched contraction's strides,
    so no kernel changes at all.

  * **Attention sinks.** Each query head has a learned logit that competes in
    the softmax and receives none of the output — it lets a head decline to
    attend. Two edits: the row maximum takes the sink into account, and the
    denominator gains `exp(sink − max)`. The normalising pass is untouched,
    which is the point: a sink is a term in the denominator, not a key.

  * **The sliding window.** Half the layers see only the last 128 positions.
    The obvious implementation offsets into the cache and shortens the
    contraction, which makes the offset a run-time quantity in every launch
    that touches it. This does something cheaper: the sliding layers get a
    128-entry **ring** cache, and position `p` writes slot `p mod 128`. The
    window is then always the whole cache, so no offset and no mask arises —
    and it is sound because softmax and the value mix are both symmetric in the
    keys, so the order the ring leaves them in cannot matter. It also costs 64×
    less memory on half the layers than a linear cache would.

  ## RoPE and the tables

  The rotation is the standard RoPE kernel at this geometry. YaRN changes which angles
  the table holds and nothing about how they are applied, so the scaling is the
  converter's business — it is arithmetic on the host, done once — and the
  kernel reads a table either way.
-/

namespace GptOssAttention

/-! ## Geometry -/

def H : Nat := 2880
def HD : Nat := 64
def HALF : Nat := HD / 2
def NQ : Nat := 64
def NKV : Nat := 8
/-- Query heads per key head. -/
def GQA : Nat := NQ / NKV

/-- Where K and V begin inside the packed QKV row, in elements. -/
def QO : Nat := NQ * HD
def KO : Nat := QO + NKV * HD
def QKV : Nat := QO + 2 * NKV * HD

/-- Cache depth of a sliding layer: one window, and a property of the
    checkpoint rather than of a run. -/
def CAP_SWA : Nat := 128

/-- The most positions a full-attention layer can be asked to hold.

    Not the depth it *is* given -- that is chosen when the engine starts and
    carried in the meta buffer, because the key cache is the largest thing on
    the card after the experts and how much of it to buy is the caller's
    decision. This is the ceiling: the published rotation tables have exactly
    this many rows, so no position beyond it can be encoded at all. -/
def CAP_MAX : Nat := 131072

/-- Positions the rotation table covers.  The whole published height: the table
    is 33.5 MiB at that size, which is nothing beside a key cache, and holding
    all of it means the rotation kernel's cosine base stays a constant. -/
def ROPE_N : Nat := CAP_MAX

/-! ## The meta buffer

    One integer buffer carries everything a launch needs to know about *this*
    token. It is uploaded once per step and read by three kernels, which is why
    the slots are named here rather than at each use. -/

/-- The absolute position, for the rotation. -/
def M_POS : Nat := 1
/-- Keys the softmax and the mix run over: `min(pos+1, cap)`. -/
def M_SEQ : Nat := 2
def M_CHUNKS : Nat := 3
def M_TAIL : Nat := 4
def M_REM : Nat := 5
/-- The cache entry this token owns: `pos mod cap`. -/
def M_SLOT : Nat := 6
/-- One key head's stride in the cache, `cap * HD`.

    In the meta rather than emitted as a literal because the two kinds of layer
    have different depths and the full one's is chosen at start-up. The slot
    beside it is already a memory read, so this costs the cache store one more
    load and buys a cache whose size is not a property of the artifact. -/
def M_KVSTRIDE : Nat := 7
/-- Tiles the fused attention splits this layer's keys into.

    In the meta because the meta is already uploaded once a layer: given its own
    buffer it needed an upload of its own, and twenty-four synchronous
    four-byte copies a token cost more than the fusion saved at short
    context. -/
def M_NTILES : Nat := 8
/-- Keys in one of those tiles.

    A run-time value because the right answer moves with the length: a tile
    that keeps the card busy at a hundred thousand keys leaves it two thirds
    idle at one thousand, and a tile small enough for one thousand makes the
    merge, which is linear in tiles, the cost at a hundred thousand. -/
def M_TILESZ : Nat := 9

/-! ## RMSNorm -/

def strideIx32 : IdxE := .add (.mul .loopI (.lit 32)) .laneId

def rmsEps : Float32 := 0.00001
def hFloat : Float32 := 2880.0

/-- `%fw0 = Σ x²`, in every lane. -/
def rmsReduce : EWStmt :=
  .seq (.seq (.setR 0 (.lit (NumOps.ofNat 0)))
             (.forN (H / 32)
               (.seq (.loadIdx 2 0 strideIx32)
                     (.setR 0 (.add (.reg 0) (.mul (.reg 2) (.reg 2)))))))
       (.seq (warpRoundE 16) (.seq (warpRoundE 8) (.seq (warpRoundE 4)
         (.seq (warpRoundE 2) (warpRoundE 1)))))

def rmsScale : WFExp :=
  .rsqrt (.add (.mul (.reg 0) (.inv (.lit hFloat))) (.lit rmsEps))

def rmsStoreCompute : EWStmt :=
  .seq (.seq (.loadIdx 2 0 strideIx32) (.loadIdx 3 1 strideIx32))
       (.setR 4 (.mul (.mul (.reg 2) (.reg 3)) (.reg 1)))

/-- One warp, ninety strided steps. The reduction leaves the total in every
    lane, so the scale needs no broadcast and the store pass follows it with no
    barrier between. -/
def rmsKernelEW : EWStmt :=
  .seq (.seq rmsReduce (.setR 1 rmsScale))
       (.forN (H / 32) (storeBody 2 4 strideIx32 rmsStoreCompute))

/-- The RMSNorm as the decode launches it: one block of one warp. -/
def rmsKernel : Kernel.Kernel := Body.kernel "rmsNorm" 3 0 rmsKernelEW

/-! ## Elementwise add

    Bias and residual are the same kernel: the output buffer is also the first
    input, so it accumulates in place. -/

def twoIn : Fin 2 → Buf := fun i => if i.val = 0 then 0 else 1
def addSpec : Expr 2 := .add (.var ⟨0, by decide⟩) (.var ⟨1, by decide⟩)

def addKernelEW : EWStmt := (mapKernel addSpec twoIn 0).ew

/-- The add as the decode launches it, one warp a block. -/
def addKernel : Kernel.Kernel := Body.kernel "addEW" 2 0 addKernelEW

/-! ## The rotation

    Buffer `0` is the packed QKV row and `base` says whether this launch turns
    the query heads or the key heads. Buffer `1` is the meta, buffer `2` the
    table — sines first, then cosines, both indexed by the position the meta
    publishes. The pair a lane owns is `base + head·HD + lane` and that plus
    `HALF`; both are read before either is written, which is what makes the
    in-place rotation sound. -/

def ropeTblIx : IdxE := .add (.mul (.ldIdx 1 (.lit M_POS)) (.lit HALF)) .laneId
def ropeCosIx : IdxE := .add (.lit (ROPE_N * HALF)) ropeTblIx
def ropeLoIx (base : Nat) : IdxE :=
  .add (.lit base) (.add (.mul .ctaId (.lit HD)) .laneId)
def ropeHiIx (base : Nat) : IdxE := .add (ropeLoIx base) (.lit HALF)

def ropeBodyEW (base : Nat) : EWStmt :=
  .seq (.seq (.seq (.loadIdx 2 2 ropeTblIx) (.loadIdx 3 2 ropeCosIx))
             (.seq (.loadIdx 4 0 (ropeLoIx base)) (.loadIdx 5 0 (ropeHiIx base))))
       (.seq (.setR 6 (.add (.mul (.reg 4) (.reg 3))
                            (.neg (.mul (.reg 5) (.reg 2)))))
             (.setR 7 (.add (.mul (.reg 4) (.reg 2))
                            (.mul (.reg 5) (.reg 3)))))

def ropeEW (base : Nat) : EWStmt :=
  .seq (ropeBodyEW base)
       (.seq (.storeLane 0 (ropeLoIx base) 6) (.storeLane 0 (ropeHiIx base) 7))

/-- The rotation as the decode launches it: one block of one warp per head. -/
def ropeKernel (base : Nat) : Kernel.Kernel := Body.kernel "rope" 3 0 (ropeEW base)

/-- The rotations that ship, as (base, heads): the query launch turns heads at
    the front of the packed row, the key launch those behind them. Two emitted
    kernels rather than one reading a base from memory, because a base in
    memory is a base every soundness statement then has to quantify over — and
    the two are the same eleven instructions. -/
def ropeShipped : List (Nat × Nat) := [(0, NQ), (QO, NKV)]

/-! ## The cache store

    A move whose destination advances with the token, so it is written directly
    rather than compiled from an expression: one block per key head, `HD/32`
    steps per warp. The destination slot is `M_SLOT`, which is the position for
    a linear cache and the position modulo the window for a ring — the kernel
    cannot tell the difference, and that is what makes one kernel serve both. -/

/-- Elements a head occupies in the cache.

    `HD` when the cache holds `Float32`, and `HD / 2` when it holds bf16 -- the
    store moves 32-bit words and two bf16 sit in one, so a bf16 cache is the
    same kernel over half as many of them. The move does not read what it
    carries, so the pair travelling in one word is not something it has to know
    about; only how many words a head is. -/
def HDW : Nat := HD / 2

def kvElem : IdxE := .add (.mul .loopI (.lit 32)) .laneId
def kvSrcIx (base hd : Nat) : IdxE :=
  .add (.lit base) (.add (.mul .ctaId (.lit hd)) kvElem)
def kvDstIx (hd : Nat) : IdxE :=
  .add (.add (.mul .ctaId (.ldIdx 2 (.lit M_KVSTRIDE)))
             (.mul (.ldIdx 2 (.lit M_SLOT)) (.lit hd)))
       kvElem

def kvStoreEW (base hd : Nat) : EWStmt :=
  .forN (hd / 32) (.seq (.loadIdx 0 0 (kvSrcIx base hd)) (.storeLane 1 (kvDstIx hd) 0))

/-- The cache store as the decode launches it: one block of one warp per key
    head, over the packed row, the cache and the meta. -/
def kvStoreKernel (base hd : Nat) : Kernel.Kernel := Body.kernel "kvStore" 3 0 (kvStoreEW base hd)

/-- The cache stores that ship, as (base, words a head): keys and values, over
    the decode's bf16 cache and the per-piece slices' `Float32` one. -/
def kvShipped : List (Nat × Nat) := [(QO / 2, HDW), (KO / 2, HDW), (QO, HD), (KO, HD)]

/-- The destination, in closed form: head, slot, element. -/
theorem kvDst_eval (hd cta j : Nat) (l : Lane) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) :
    (kvDstIx hd).eval cta j l ir im
      = cta * im 2 M_KVSTRIDE + im 2 M_SLOT * hd + (j * 32 + l.val) := rfl

/-- **The cache store lands the right element at the right address**, for every
    `(loop, lane)` the kernel visits — not merely "some lane wrote it". -/
theorem kvStore_writes {R : Type} [NumOps R] [NumLit R] (base hd : Nat) (cta : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) (st : WSt R) (i0 : Nat) (l0 : Lane) (hi0 : i0 < hd / 32) :
    (((kvStoreEW base hd).run st cta 0 ir im)).mem 1
        ((kvDstIx hd).eval cta i0 l0 ir im)
      = st.mem 0 ((kvSrcIx base hd).eval cta i0 l0 ir im) := by
  refine storeLoop_at 1 0 (kvDstIx hd) (.loadIdx 0 0 (kvSrcIx base hd)) cta ir im st
    (fun j l => st.mem 0 ((kvSrcIx base hd).eval cta j l ir im))
    ((kvDstIx hd).eval cta i0 l0 ir im)
    (st.mem 0 ((kvSrcIx base hd).eval cta i0 l0 ir im)) []
    (fun _ _ => rfl) (fun _ _ r' h => absurd h (by simp))
    (fun j s hinv _ l => by
      show s.mem 0 ((kvSrcIx base hd).eval cta j l ir im) = _
      rw [hinv 0 (by decide)])
    (List.range (hd / 32)) st (fun _ _ => rfl)
    (fun r' h => absurd h (by simp)) ?_ ?_
  · intro j hj l hl
    rw [kvDst_eval, kvDst_eval] at hl
    have hlt : l.val < 32 := l.isLt
    have hlt0 : l0.val < 32 := l0.isLt
    have hj2 : j = i0 ∧ l.val = l0.val := by omega
    have : l = l0 := Fin.ext hj2.2
    rw [hj2.1, this]
  · exact Or.inl ⟨i0, List.mem_range.mpr hi0, l0, rfl⟩

/-! ## Softmax with a sink

    A dynamic-length softmax, which reads its trip counts from integer
    memory so that one emitted kernel serves every sequence length, plus the
    sink. Buffer `3` holds one learned logit per query head and the block *is*
    the query head, so the sink's address is the block index and nothing else.

    The sink enters twice and only twice:

    * the row maximum becomes `max(max_j s_j, sink)`, so the exponentials stay
      bounded when the sink dominates — which is exactly the case a head that
      wants to attend to nothing produces;
    * the denominator gains `exp(sink − max)`.

    The normalising pass is untouched. That asymmetry is the definition: the
    sink absorbs probability mass and contributes no value, so it belongs in
    the sum and not in the store. -/

def seqLenIx : IdxE := .ldIdx 1 (.lit M_SEQ)
def rowBaseIx : IdxE := .mul .ctaId seqLenIx
def smIx : IdxE := .add rowBaseIx (.add (.mul .loopI (.lit 32)) .laneId)
def smTailIx : IdxE := .add rowBaseIx (.add (.ldIdx 1 (.lit M_TAIL)) .loopI)

/-- This block's sink: the block index, since a block is a query head. -/
def sinkIx : IdxE := .ctaId

def maxF : WFExp := .maxW (.reg 0) (.reg 2)
def sumF : WFExp := .add (.reg 0) (.exp (.add (.reg 2) (.neg (.reg 5))))

def smSumBfly : EWStmt :=
  .seq (warpRoundE 16) (.seq (warpRoundE 8) (.seq (warpRoundE 4)
    (.seq (warpRoundE 2) (warpRoundE 1))))

def smMaxReduce : EWStmt :=
  chunkRemReduce 0 2 0 maxF smIx smTailIx (.lit (-100000000.0))
    1 M_CHUNKS M_REM warpReduceMaxE

/-- Pass 1, and the first sink edit: `%fw5 ← max(row max, sink)`. -/
def smMax : EWStmt :=
  .seq (.seq smMaxReduce (.loadIdx 6 3 sinkIx)) (.setR 5 (.maxW (.reg 0) (.reg 6)))

def smSum : EWStmt :=
  chunkRemReduce 0 2 0 sumF smIx smTailIx (.lit (NumOps.ofNat 0))
    1 M_CHUNKS M_REM smSumBfly

/-- The second sink edit: one more term in the denominator. The sink is
    re-loaded rather than kept across the sum, because the reduction's register
    discipline is stated about the accumulator and its load slot and says
    nothing about `%fw6`; a load is one instruction and an assumption is not. -/
def smSink : EWStmt :=
  .seq (.loadIdx 6 3 sinkIx)
       (.setR 0 (.add (.reg 0) (.exp (.add (.reg 6) (.neg (.reg 5))))))

def normCompute (ix : IdxE) : EWStmt :=
  .seq (.loadIdx 2 0 ix)
       (.setR 4 (.mul (.exp (.add (.reg 2) (.neg (.reg 5)))) (.reg 3)))

def normStep (ix : IdxE) : EWStmt := storeBody 2 4 ix (normCompute ix)

/-- A remainder step: every lane computes the element at the same index, and
    lane 0 stores it, so each element has one writer. -/
def normStep0 (ix : IdxE) : EWStmt := .seq (normCompute ix) (.storeLane0 2 ix 4)

def smNorm : EWStmt :=
  .seq (.setR 3 (.inv (.reg 0)))
       (.seq (.forM 1 M_CHUNKS (normStep smIx))
             (.forM 1 M_REM (normStep0 smTailIx)))

def sinkSoftmaxEW : EWStmt := .seq (.seq (.seq smMax smSum) smSink) smNorm

/-- The softmax as it is launched: one block of one warp per query head, over
    the scores, the meta, the probabilities and the sinks. -/
def sinkKernel : Kernel.Kernel := Body.kernel "sinkSoftmax" 4 0 sinkSoftmaxEW

/-- **The sink is in the denominator and out of the numerator.**

    The claim the two edits amount to, checked at the statement the kernel
    prints: of its four buffers, the only one it stores to is `2`, the
    probabilities. Buffer `3` is the
    sinks and buffer `0` the scores, so neither the sink nor a score can be
    written back however the passes are read — the sink reaches the output only
    through the denominator.

    A head whose sink dominates therefore emits probabilities summing to less
    than one, which is the behaviour the model defines and the whole reason the
    term exists. -/
theorem sink_not_stored :
    (List.range 4).all (fun b => !(sinkSoftmaxEW.WritesBufB b) || b == 2) = true := by decide

end GptOssAttention
