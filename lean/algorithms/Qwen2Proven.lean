import Lean
import Std
import AlgorithmLib

open Lean AlgorithmLib AlgorithmLib.IR AlgorithmLib.ML

namespace Qwen2Proven

def D      : Nat := 896
def D_FF   : Nat := 4864
def KV_DIM : Nat := 128

/-! Both operands live in their own buffer; each lane reads index
    `ctaid·32 + laneid` from each — the library's `elemIx`. -/

def twoIn : Fin 2 → Buf := fun i => if i.val = 0 then 0 else 1

def twoIx : Fin 2 → IdxE := fun _ => elemIx

-- ---------------------------------------------------------------------------
-- residual add / bias add:  out[i] = x[i] + a[i]
-- ---------------------------------------------------------------------------

/-- The spec.  Two inputs, one output, no `exp` — so its lowering theorem is
    unconditional. -/
def addSpec : Expr 2 := .add (.var ⟨0, by decide⟩) (.var ⟨1, by decide⟩)

/-- Output buffer `0`: `residualAdd` writes back into `x`, which is exactly
    what `x_buf` was in the hand-written kernel. -/
def addKernel : EWStmt :=
  compileWKernel twoIn 0 twoIx addSpec elemIx (2 + slots addSpec + 1)

def ptxAdd : String := emitProvenKernel "main" addKernel

/-- **The residual add is a pipeline stage.**

    It reads and writes buffer `0`, which `mapStage` forbids.  `mapStageIP`
    admits it: a block reads `out` only at the address it writes, and that is an
    address it owns, so no other block of the launch can have disturbed it.
    Same kernel, same emitted PTX — the `rfl` below is the check. -/
def addStage (grid : Nat) : StageSpec := mapStageIP addSpec twoIn 0 grid

example (grid : Nat) : (addStage grid).ew = addKernel := rfl

/-- **The emitted add kernel runs its spec, from raw launch.** -/
theorem add_ptx_exact (cta : Nat) (m : MState) :
    ∃ k m', steps cta (flatKernel (expandEW addKernel)) k (0, m)
          = some ((flatKernel (expandEW addKernel)).length, m')
      ∧ m'.toWSt = ((expandEW addKernel).elabIn cta).run m.toWSt :=
  flatKernel_sound_idxFree cta (expandEW addKernel) (expandEW_expFree addKernel)
    (expandEW_idxFree addKernel
      (compileWKernel_idxFree twoIn 0 twoIx addSpec elemIx _
        (fun _ => ⟨⟨trivial, trivial⟩, trivial⟩) ⟨⟨trivial, trivial⟩, trivial⟩))
    (expandEW_flat addKernel (compileWKernel_flat _ _ _ _ _ _)) m

/-- **Per lane, the kernel computes `x + a` from memory.**  The addressing, the
    two-buffer prologue and the store are all discharged. -/
theorem add_computes (st : WSt) (l : Lane) :
    ((compileW (fun i => .reg i.val) 2 addSpec).2).eval
        (runW (compileW (fun i => .reg i.val) 2 addSpec).1
          (runW (loadSeq twoIn twoIx) st)) l
      = NumOps.add (st.mem 0 (elemIx.eval 0 0 l)) (st.mem 1 (elemIx.eval 0 0 l)) :=
  compileWKernel_correct twoIn twoIx addSpec st l

-- ---------------------------------------------------------------------------
-- silu-gate:  out[i] = silu(g[i]) * u[i]
-- ---------------------------------------------------------------------------

/-- `silu x = x · (1 + e^{-x})⁻¹` — written from the same `Transformer.silu`
    the MLP demo uses, so the two share one definition of the activation. -/
def siluGateSpec : Expr 2 :=
  .mul (Transformer.silu (.var ⟨0, by decide⟩)) (.var ⟨1, by decide⟩)

/-! The same specs, function-first — `rfl`, so the surface is not a fork. -/
example : addSpec = ofFn 2 (fun a b => a + b) := rfl
example : siluGateSpec = ofFn 2 (fun a b => silu a * b) := rfl

def siluKernel : EWStmt :=
  compileWKernel twoIn 2 twoIx siluGateSpec elemIx (2 + slots siluGateSpec + 1)

def ptxSilu : String := emitProvenKernelN "main" 3 0 siluKernel

/-- **The emitted silu-gate kernel runs its compiled statement, unconditionally.**

    Unconditional because the *printed* object is `expandEW siluKernel`, whose
    `exp` has already been rewritten to `ex2`. -/
theorem silu_ptx_exact (cta : Nat) (m : MState) :
    ∃ k m', steps cta (flatKernel (expandEW siluKernel)) k (0, m)
          = some ((flatKernel (expandEW siluKernel)).length, m')
      ∧ m'.toWSt = ((expandEW siluKernel).elabIn cta).run m.toWSt :=
  flatKernel_sound_idxFree cta (expandEW siluKernel) (expandEW_expFree siluKernel)
    (expandEW_idxFree siluKernel
      (compileWKernel_idxFree twoIn 2 twoIx siluGateSpec elemIx _
        (fun _ => ⟨⟨trivial, trivial⟩, trivial⟩) ⟨⟨trivial, trivial⟩, trivial⟩))
    (expandEW_flat siluKernel (compileWKernel_flat _ _ _ _ _ _)) m

/-- …and it runs the *spec* under the one declared identity, `e^x = 2^(x·log₂e)` —
    the same hypothesis the MLP demo carries, measured there at 3.27e-7. -/
theorem silu_ptx_runs_kernel (h : ExpIsEx2) (cta : Nat) (m : MState) :
    ∃ k m', steps cta (flatKernel (expandEW siluKernel)) k (0, m)
          = some ((flatKernel (expandEW siluKernel)).length, m')
      ∧ m'.toWSt = (siluKernel.elabIn cta).run m.toWSt := by
  obtain ⟨k, m', hs, hw⟩ := silu_ptx_exact cta m
  exact ⟨k, m', hs, by rw [hw]; exact expandEW_run h siluKernel cta 0 m.toWSt⟩

/-- **Per lane, the kernel computes `silu(g)·u` from memory.** -/
theorem silu_computes (st : WSt) (l : Lane) :
    ((compileW (fun i => .reg i.val) 2 siluGateSpec).2).eval
        (runW (compileW (fun i => .reg i.val) 2 siluGateSpec).1
          (runW (loadSeq twoIn twoIx) st)) l
      = denote (fun i => st.mem (twoIn i) ((twoIx i).eval 0 0 l)) siluGateSpec :=
  compileWKernel_correct twoIn twoIx siluGateSpec st l

/-! ### Where the elementwise kernels store

    `compileWKernel_correct` proves what a lane *computes*; a kernel that
    computes perfectly and stores to the wrong slot satisfies it.
    `compileWKernel_stores` closes that, and every `compileWKernel` kernel gets
    it by instantiation — the only obligation is that the output address
    distinguishes lanes, which for `elemIx` is one `omega`. -/

/-- **The add kernel's result lands where it should.** -/
theorem add_stores (st : WSt) (l0 : Lane) :
    ((addKernel.elabIn 0).run st).mem 0 (elemIx.eval 0 0 l0)
      = denote (fun i => st.mem (twoIn i) ((twoIx i).eval 0 0 l0)) addSpec :=
  compileWKernel_stores twoIn 0 twoIx addSpec elemIx _ st l0
    (fun l h => by rw [elemIx_inj l l0 h])

/-- **…and the silu-gate kernel's.** -/
theorem silu_stores (st : WSt) (l0 : Lane) :
    ((siluKernel.elabIn 0).run st).mem 2 (elemIx.eval 0 0 l0)
      = denote (fun i => st.mem (twoIn i) ((twoIx i).eval 0 0 l0)) siluGateSpec :=
  compileWKernel_stores twoIn 2 twoIx siluGateSpec elemIx _ st l0
    (fun l h => by rw [elemIx_inj l l0 h])

-- ---------------------------------------------------------------------------
-- embedding lookup:  out[i] = embed[tok · D + i],  tok read from memory
-- ---------------------------------------------------------------------------

/-! The first *shipped* kernel whose address is data-dependent.  The token id is
    not a function of `(lane, loop, block)` — it is read from the meta buffer at
    run time, which is precisely what `IdxE.ldIdx` was added for and what no
    affine address could express.

    Consequently this kernel is **not** `IdxFree`, so it uses `flatKernel_sound`
    directly rather than the address-free specialisation.  Its precondition is
    `IdxBelow`, discharged because the gathered index's own offset is a
    literal. -/

/-- `embed[ meta[0] · D + (ctaid·32 + laneid) ]`. -/
def embedIx : IdxE := .add (.mul (.ldIdx 1 (.lit 0)) (.lit D)) elemIx

def embedIn : Fin 1 → Buf := fun _ => 0
def embedInIx : Fin 1 → IdxE := fun _ => embedIx

/-- The spec is the identity on the gathered element: a row copy. -/
def embedSpec : Expr 1 := .var ⟨0, by decide⟩

def embedKernelEW : EWStmt :=
  compileWKernel embedIn 2 embedInIx embedSpec elemIx (1 + slots embedSpec + 1)

def ptxEmbed : String := emitProvenKernelN "main" 3 0 embedKernelEW

theorem embed_idxBelow : (expandEW embedKernelEW).IdxBelow 3 :=
  expandEW_idxBelow 3 embedKernelEW (by decide)

/-- **The emitted gather runs its compiled statement, from raw launch.**

    Note the shape of the conclusion: it elaborates against the machine's own
    integer memory `m.imem`, because the address genuinely depends on it.  That
    is the difference between this and every affine kernel in the stack. -/
theorem embed_ptx_exact (cta : Nat) (m : MState) :
    ∃ k m', steps cta (flatKernel (expandEW embedKernelEW)) k (0, m)
          = some ((flatKernel (expandEW embedKernelEW)).length, m')
      ∧ m'.toWSt
        = ((expandEW embedKernelEW).elabAt cta 0
            (SI.stepL cta emitPrologue m).ir m.imem).run m.toWSt :=
  flatKernel_sound cta (expandEW embedKernelEW) (expandEW_expFree embedKernelEW)
    embed_idxBelow (expandEW_flat embedKernelEW (compileWKernel_flat _ _ _ _ _ _)) m

/-- **The gather's result lands where it should.**  Same theorem, and the
    data-dependent *source* address costs nothing here — only the destination
    has to distinguish lanes. -/
theorem embed_stores (st : WSt) (l0 : Lane) :
    ((embedKernelEW.elabIn 0).run st).mem 2 (elemIx.eval 0 0 l0)
      = denote (fun i => st.mem (embedIn i) ((embedInIx i).eval 0 0 l0)) embedSpec :=
  compileWKernel_stores embedIn 2 embedInIx embedSpec elemIx _ st l0
    (fun l h => by rw [elemIx_inj l l0 h])

-- ---------------------------------------------------------------------------
-- KV-cache store:  kCache[h, pos, e] = kCur[h, e],  pos read from memory
-- ---------------------------------------------------------------------------

/-! Not a `compileW` kernel: there is no arithmetic to compile, only a move
    whose *destination* is data-dependent (`pos` advances every token).  So it
    is written directly in `EWStmt`, the way `warpDotV4` is — the emittable
    language is meant to be written by hand when the kernel is a data movement
    rather than an expression.

    One block per KV head, a loop of `HEAD_DIM/32` steps per warp. -/

def HEAD_DIM : Nat := 64
def MAX_SEQ  : Nat := 2048

/-- Element within this head: `loop·32 + lane`. -/
def kvElem : IdxE := .add (.mul .loopI (.lit 32)) .laneId

/-- Source: `kCur[ctaid · HEAD_DIM + e]`. -/
def kvSrcIx : IdxE := .add (.mul .ctaId (.lit HEAD_DIM)) kvElem

/-- Destination: `kCache[ctaid · MAX_SEQ · HEAD_DIM + pos · HEAD_DIM + e]`,
    where `pos` is `meta[1]` — read at run time. -/
def kvDstIx : IdxE :=
  .add (.add (.mul .ctaId (.lit (MAX_SEQ * HEAD_DIM)))
             (.mul (.ldIdx 2 (.lit 1)) (.lit HEAD_DIM)))
       kvElem

def kvStoreEW : EWStmt :=
  .forN (HEAD_DIM / 32) (.seq (.loadIdx 0 0 kvSrcIx) (.storeLane 1 kvDstIx 0))

def ptxKVStore : String := emitProvenKernelN "main" 3 0 kvStoreEW

theorem kvStore_idxBelow : (expandEW kvStoreEW).IdxBelow 3 :=
  expandEW_idxBelow 3 kvStoreEW (by decide)

/-- **The emitted KV store runs its statement, from raw launch.**  Both the
    strided source address and the position-dependent destination are covered. -/
theorem kvStore_ptx_exact (cta : Nat) (m : MState) :
    ∃ k m', steps cta (flatKernel (expandEW kvStoreEW)) k (0, m)
          = some ((flatKernel (expandEW kvStoreEW)).length, m')
      ∧ m'.toWSt
        = ((expandEW kvStoreEW).elabAt cta 0
            (SI.stepL cta emitPrologue m).ir m.imem).run m.toWSt :=
  flatKernel_sound cta (expandEW kvStoreEW) (expandEW_expFree kvStoreEW)
    kvStore_idxBelow (expandEW_flat kvStoreEW (by decide)) m

/-! ### What the KV store writes

    The first *store* conformance in the stack, and the one that closes a gap
    open since `compileWKernel_correct`: that theorem proves the **value** a
    kernel computes and says nothing about **where it lands**.

    `storeLoop_at` supplies the read-back; all this instance has to do is show
    the destination address determines the `(loop, lane)` that wrote it, which
    is `omega` once the address is unfolded. -/

/-- The destination address, in closed form.  `rfl` — but naming it is what
    lets `omega` see the arithmetic. -/
theorem kvDst_eval (cta j : Nat) (l : Lane) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) :
    kvDstIx.eval cta j l ir im
      = cta * (MAX_SEQ * HEAD_DIM) + im 2 1 * HEAD_DIM + (j * 32 + l.val) := rfl

/-- **The KV store lands the right element at the right address.**

    For every `(loop, lane)` the kernel visits, the destination slot ends up
    holding the source element — not merely "some lane computed it". -/
theorem kvStore_writes (cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (st : WSt) (i0 : Nat) (l0 : Lane) (hi0 : i0 < HEAD_DIM / 32) :
    ((kvStoreEW.elabAt cta 0 ir im).run st).mem 1 (kvDstIx.eval cta i0 l0 ir im)
      = st.mem 0 (kvSrcIx.eval cta i0 l0 ir im) := by
  refine storeLoop_at 1 0 kvDstIx (.loadIdx 0 0 kvSrcIx) cta ir im st
    (fun j l => st.mem 0 (kvSrcIx.eval cta j l ir im))
    (kvDstIx.eval cta i0 l0 ir im) (st.mem 0 (kvSrcIx.eval cta i0 l0 ir im)) []
    (fun _ _ => rfl) (fun _ _ r' h => absurd h (by simp))
    (fun j s hinv _ l => by
      show s.mem 0 (kvSrcIx.eval cta j l ir im) = _
      rw [hinv 0 (by decide)])
    (List.range (HEAD_DIM / 32)) st (fun _ _ => rfl)
    (fun r' h => absurd h (by simp)) ?_ ?_
  · -- every (loop, lane) writing this address writes this value
    intro j hj l hl
    rw [kvDst_eval, kvDst_eval] at hl
    have hlt : l.val < 32 := l.isLt
    have hlt0 : l0.val < 32 := l0.isLt
    have hj2 : j = i0 ∧ l.val = l0.val := by omega
    have : l = l0 := Fin.ext hj2.2
    rw [hj2.1, this]
  · exact Or.inl ⟨i0, List.mem_range.mpr hi0, l0, rfl⟩

-- ---------------------------------------------------------------------------
-- RMSNorm:  y[i] = x[i] · w[i] · rsqrt(Σx²/D + eps)
-- ---------------------------------------------------------------------------

/-! The hand-written kernel used **eight warps and shared memory**: a per-warp
    butterfly, then lane 0 of each warp writes a slot, then thread 0 sums the
    eight and broadcasts through shared memory.  Three synchronisation points
    for one scalar.

    One warp needs none of that.  The butterfly already leaves the total *in
    every lane*, so the scale is computed redundantly per lane and phase 2
    proceeds immediately — no shared memory, no barrier, and the reduction is
    the proven `warpRoundE` sequence from `Schema.lean`.

    Trip count `D/32 = 28`: each lane owns a strided slice.

    One deliberate numerical change: the original divided by `D` with `div.rn`;
    the spec language expresses division as multiplication by `inv`, so this
    computes `Σx² · (1/D)` — two roundings instead of one.  It is exact with
    respect to *its own* spec, which is what the theorem states, and the model's
    output is unchanged (verified end to end). -/

/-- Strided element for a single warp: `loop·32 + lane`. -/
def strideIx32 : IdxE := .add (.mul .loopI (.lit 32)) .laneId

/-- `1e-5`, the RMSNorm epsilon, and `896.0` — the same constants the
    hand-written kernel folded in as bit patterns. -/
def rmsEps : Float32 := 0.00001
def dFloat : Float32 := 896.0

/-- Phase 1: `%fw0 = Σ x²` in every lane, via a strided sweep and the proven
    five-round butterfly. -/
def rmsReduce : EWStmt :=
  .seq (.seq (.setR 0 (.lit (NumOps.ofNat 0)))
             (.forN (D / 32)
               (.seq (.loadIdx 2 0 strideIx32)
                     (.setR 0 (.add (.reg 0) (.mul (.reg 2) (.reg 2)))))))
       (.seq (warpRoundE 16) (.seq (warpRoundE 8) (.seq (warpRoundE 4)
         (.seq (warpRoundE 2) (warpRoundE 1)))))

/-- Phase 2: `%fw1 = rsqrt(Σx²·(1/D) + eps)`, then the normalised store. -/
def rmsScale : WFExp :=
  .rsqrt (.add (.mul (.reg 0) (.inv (.lit dFloat))) (.lit rmsEps))

/-- The store pass's per-iteration computation, with the store split off so it
    can be read back.  Same emitted instructions — `seq` is list append. -/
def rmsStoreCompute : EWStmt :=
  .seq (.seq (.loadIdx 2 0 strideIx32) (.loadIdx 3 1 strideIx32))
       (.setR 4 (.mul (.mul (.reg 2) (.reg 3)) (.reg 1)))

def rmsKernelEW : EWStmt :=
  .seq (.seq rmsReduce (.setR 1 rmsScale))
       (.forN (D / 32) (storeBody 2 4 strideIx32 rmsStoreCompute))

/-! ### Where RMSNorm stores

    The first store pass whose computation reads a **loop-invariant register**:
    `%fw1` holds the scale, computed once before the loop.  `storeLoop_at`'s
    `keep` list is what carries that across iterations — the loop writes `%fw2`,
    `%fw3` and `%fw4`, so `%fw1` still holds what phase 2 put there. -/

/-- What lands at element `loop·32 + lane`: `x · w · scale`. -/
def rmsStoreVal (mem : Buf → Nat → Float32) (scale : Lane → Float32) (cta : Nat)
    (j : Nat) (l : Lane) : Float32 :=
  NumOps.mul (NumOps.mul (mem 0 (strideIx32.eval cta j l)) (mem 1 (strideIx32.eval cta j l)))
             (scale l)

theorem strideIx32_eval (cta j : Nat) (l : Lane) :
    strideIx32.eval cta j l = j * 32 + l.val := rfl

set_option maxRecDepth 20000 in
/-- **RMSNorm's normalised output lands at the right address.** -/
theorem rms_stores (cta : Nat) (st : WSt) (j0 : Nat) (l0 : Lane)
    (hj0 : j0 < D / 32) :
    (((EWStmt.forN (D / 32) (storeBody 2 4 strideIx32 rmsStoreCompute)).elabIn cta).run st).mem
        2 (strideIx32.eval cta j0 l0)
      = rmsStoreVal st.mem (st.regs 1) cta j0 l0 := by
  refine storeLoop_at 2 4 strideIx32 rmsStoreCompute cta _ _ st
    (fun j l => rmsStoreVal st.mem (st.regs 1) cta j l)
    (strideIx32.eval cta j0 l0) (rmsStoreVal st.mem (st.regs 1) cta j0 l0) [1]
    (fun _ _ => rfl) (fun _ s r' hr' => by
      have : r' = 1 := by simp at hr'; omega
      subst this; rfl)
    (fun j s hinv hkeep l => by
      show NumOps.mul (NumOps.mul (s.mem 0 (strideIx32.eval cta j l))
                                  (s.mem 1 (strideIx32.eval cta j l))) (s.regs 1 l) = _
      rw [hinv 0 (by decide), hinv 1 (by decide), hkeep 1 (by simp)]
      rfl)
    (List.range (D / 32)) st (fun _ _ => rfl) (fun _ _ => rfl) ?_ ?_
  · intro j hj l hl
    rw [strideIx32_eval, strideIx32_eval] at hl
    have h1 : l.val < 32 := l.isLt
    have h2 : l0.val < 32 := l0.isLt
    have hjl : j = j0 ∧ l.val = l0.val := by omega
    have : l = l0 := Fin.ext hjl.2
    rw [hjl.1, this]
  · exact Or.inl ⟨j0, List.mem_range.mpr hj0, l0, rfl⟩

def ptxRmsNorm : String := emitProvenKernelN "main" 3 0 rmsKernelEW

/-- **The emitted RMSNorm runs its statement, from raw launch, unconditionally.**

    No `exp`, no data-dependent address — so this is the strongest form: the
    address-free specialisation with no hypothesis at all. -/
theorem rms_ptx_exact (cta : Nat) (m : MState) :
    ∃ k m', steps cta (flatKernel (expandEW rmsKernelEW)) k (0, m)
          = some ((flatKernel (expandEW rmsKernelEW)).length, m')
      ∧ m'.toWSt = ((expandEW rmsKernelEW).elabIn cta).run m.toWSt :=
  flatKernel_sound_idxFree cta (expandEW rmsKernelEW) (expandEW_expFree rmsKernelEW)
    (expandEW_idxFree rmsKernelEW (by decide))
    (expandEW_flat rmsKernelEW (by decide)) m

/-! ### What the reduction computes

    The theorems above say the PTX performs the `EWStmt`.  They do **not** say
    the `EWStmt` computes RMSNorm — and that gap is not academic: the softmax
    draft in this file once carried two bugs (a 32× index overrun and an
    accumulator kept in the butterfly's own scratch register) that every
    execution theorem would have happily certified.

    The reduction is where the content is, and `sweep_fold`/`denote_sweepFoldE`
    from `Schema.lean` supply it generically — the same lemmas softmax and
    argmax use.  Note the spec is the *two-level* fold the hardware actually
    performs, sequential within a lane and butterfly across lanes: committing to
    that order is what makes this an exact `Float32` equality. -/

/-- `Σ x²` over this lane's strided slice, in load order. -/
def rmsLaneFold (mem : Nat → Float32) (cta : Nat) (l : Lane) : Float32 :=
  sweepFold mem (fun i l' => strideIx32.eval cta i l')
    (fun _ a x => NumOps.add a (NumOps.mul x x)) (NumOps.ofNat 0) (D / 32) l

/-- **Phase 1 computes the two-level sum of squares, in every lane.** -/
theorem rms_reduce_spec (cta : Nat) (st : WSt) (l : Lane) :
    ((rmsReduce.elabIn cta).run st).regs 0 l
      = bflyFold (rmsLaneFold (st.mem 0) cta) l := by
  have hinner : ((((EWStmt.setR 0 (.lit (NumOps.ofNat 0))).seq
        (EWStmt.forN (D / 32) (sweepBody 0 2 0
          (.add (.reg 0) (.mul (.reg 2) (.reg 2))) strideIx32))).elabAt cta 0).run st).regs 0
      = rmsLaneFold (st.mem 0) cta := by
    funext l'
    show (((EWStmt.forN (D / 32) (sweepBody 0 2 0 _ strideIx32)).elabAt cta 0).run
            ((WStmt.setR 0 (.lit (NumOps.ofNat 0))).run st)).regs 0 l' = _
    rw [sweepN_spec 0 2 0 (by decide) (.add (.reg 0) (.mul (.reg 2) (.reg 2))) strideIx32
          (fun _ a x => NumOps.add a (NumOps.mul x x)) (D / 32) cta
          [] (fun _ h => absurd h (by simp))
          ((WStmt.setR 0 (.lit (NumOps.ofNat 0))).run st) (fun _ _ _ => rfl) l']
    simp only [wrun_setR, WSt.mem_setReg, WSt.regs_setReg_same, WFExp.eval]
    rfl
  show ((warpReduceSum 0 1).run
          ((((EWStmt.setR 0 (.lit (NumOps.ofNat 0))).seq
              (EWStmt.forN (D / 32) (sweepBody 0 2 0
                (.add (.reg 0) (.mul (.reg 2) (.reg 2))) strideIx32))).elabAt cta 0).run st)).regs 0 l
        = _
  rw [warpReduceSum_spec 0 1 (by decide), hinner]

/-- The same fold as a spec-language term: `ae i` is whatever the model says
    lives at slot `i` of the input buffer. -/
def rmsSumSqE {Γ : Nat} (ae : Nat → Expr Γ) (cta : Nat) : Expr Γ :=
  bflyStepE 1 (bflyStepE 2 (bflyStepE 4 (bflyStepE 8 (bflyStepE 16
    (sweepFoldE ae (fun i l => strideIx32.eval cta i l)
      (fun _ a x => .add a (.mul x x)) (.lit 0) (D / 32)))))) ⟨0, by decide⟩

/-- **RMSNorm's reduction computes its spec.**

    `ha` is the layout link: buffer slot `i` holds what `ae i` denotes.  Given
    it, the value the kernel leaves in `%fw0` is exactly `denote env` of the
    spec term — for every block, exactly, at `Float32`. -/
theorem rms_reduce_implements {Γ : Nat} (cta : Nat) (st : WSt)
    (env : Fin Γ → Float32) (ae : Nat → Expr Γ)
    (ha : ∀ i, denote env (ae i) = st.mem 0 i) :
    ((rmsReduce.elabIn cta).run st).regs 0 ⟨0, by decide⟩
      = denote env (rmsSumSqE ae cta) := by
  rw [rms_reduce_spec cta st ⟨0, by decide⟩]
  -- the per-lane bridge, then the butterfly on both sides
  have hfold : ∀ l : Lane,
      denote env (sweepFoldE ae (fun i l' => strideIx32.eval cta i l')
          (fun _ a x => Expr.add a (.mul x x)) (.lit 0) (D / 32) l)
        = rmsLaneFold (st.mem 0) cta l := by
    intro l
    rw [denote_sweepFoldE env ae (fun i l' => strideIx32.eval cta i l')
          (fun _ a x => Expr.add a (.mul x x))
          (fun _ a x => NumOps.add a (NumOps.mul x x)) (st.mem 0) ha
          (fun _ a x => by simp only [denote_add, denote_mul]) (.lit 0) (D / 32) l]
    rfl
  show _ = denote env (bflyStepE 1 (bflyStepE 2 (bflyStepE 4 (bflyStepE 8
    (bflyStepE 16 (sweepFoldE ae (fun i l' => strideIx32.eval cta i l')
      (fun _ a x => Expr.add a (.mul x x)) (.lit 0) (D / 32)))))) ⟨0, by decide⟩)
  simp only [bflyStepE, denote_add, bflyFold, bflyStep, hfold]

-- ---------------------------------------------------------------------------
-- RoPE:  rotate (v[lo], v[hi]) by the angle at (pos, freq)
-- ---------------------------------------------------------------------------

/-! No trigonometry is needed here, despite appearances: `sin` and `cos` come
    from a table the host precomputes, so the kernel only gathers and rotates.
    That is why RoPE migrates without extending `NumOps` — a gap I had expected
    and that turned out not to exist.

    The geometry already matched: `HEAD_DIM/2 = 32` is exactly one warp, so the
    lane *is* the frequency index and the block *is* the head.

    Both table reads are data-dependent (`pos` advances each token), so this is
    another `ldIdx` kernel.  One numerical difference from the hand-written
    version: that one used `fma`, which rounds once; the spec language has no
    fused multiply-add, so this rounds twice. -/

def HALF : Nat := HEAD_DIM / 2

/-- `pos · 32 + lane` — where this lane's angle lives in the table. -/
def ropeTblIx : IdxE := .add (.mul (.ldIdx 1 (.lit 1)) (.lit HALF)) .laneId
/-- Cosines follow the whole sine block. -/
def ropeCosIx : IdxE := .add (.lit (MAX_SEQ * HALF)) ropeTblIx
/-- The pair this lane rotates: `head · HEAD_DIM + freq` and `+ HALF`. -/
def ropeLoIx : IdxE := .add (.mul .ctaId (.lit HEAD_DIM)) .laneId
def ropeHiIx : IdxE := .add ropeLoIx (.lit HALF)

/-- `%fw6 = lo·cos − hi·sin`, `%fw7 = lo·sin + hi·cos`.  Both are computed
    before either is stored — the rotation reads both halves, which is why the
    stores are split off below: they can then be read back independently.  The
    emitted instruction list is unchanged, since `emitEW` on a `seq` is list
    append and append is associative. -/
def ropeBodyEW : EWStmt :=
  .seq (.seq (.seq (.loadIdx 2 2 ropeTblIx) (.loadIdx 3 2 ropeCosIx))
             (.seq (.loadIdx 4 0 ropeLoIx) (.loadIdx 5 0 ropeHiIx)))
       (.seq (.setR 6 (.add (.mul (.reg 4) (.reg 3))
                            (.neg (.mul (.reg 5) (.reg 2)))))
             (.setR 7 (.add (.mul (.reg 4) (.reg 2))
                            (.mul (.reg 5) (.reg 3)))))

def ropeEW : EWStmt :=
  .seq ropeBodyEW (.seq (.storeLane 0 ropeLoIx 6) (.storeLane 0 ropeHiIx 7))

/-- **RoPE is provably *not* stage-eligible.**

    It loads buffer 0 (`ropeLoIx`, `ropeHiIx`) and stores buffer 0 — so a
    `StageSpec`'s `valOnly`, which requires the output to be independent of the
    output buffer's prior contents, is false of it.  Its exclusion from the
    pipeline abstraction was a docstring until `A47` G8; this is the check.

    The exclusion is *correct*, not a limitation: RoPE's soundness rests on all
    four loads preceding either store, which is a fact about intra-kernel order
    that no inter-kernel frame condition can express. -/
theorem rope_not_idempotent : ropeEW.IdempotentEligibleB 0 = false := by decide

/-! ### Where RoPE stores

    Unlike every other pass, RoPE **writes the buffer it reads** — the rotated
    pair replaces the original in place.  That is sound because all four loads
    happen before either store, and it is why this one is not a `storeLoop_at`
    instance: it is two `storeLane`s whose address sets must be shown disjoint.

    `ropeHiIx = ropeLoIx + 32` and a lane index is `< 32`, so neither store can
    land on the other's slots — one `omega` each. -/

/-- `%fw6`: `lo·cos − hi·sin`. -/
def ropeLoVal (mem : Buf → Nat → Float32) (cta : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) (l : Lane) : Float32 :=
  NumOps.add (NumOps.mul (mem 0 (ropeLoIx.eval cta 0 l ir im))
                         (mem 2 (ropeCosIx.eval cta 0 l ir im)))
             (NumOps.neg (NumOps.mul (mem 0 (ropeHiIx.eval cta 0 l ir im))
                                     (mem 2 (ropeTblIx.eval cta 0 l ir im))))

/-- `%fw7`: `lo·sin + hi·cos`. -/
def ropeHiVal (mem : Buf → Nat → Float32) (cta : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) (l : Lane) : Float32 :=
  NumOps.add (NumOps.mul (mem 0 (ropeLoIx.eval cta 0 l ir im))
                         (mem 2 (ropeTblIx.eval cta 0 l ir im)))
             (NumOps.mul (mem 0 (ropeHiIx.eval cta 0 l ir im))
                         (mem 2 (ropeCosIx.eval cta 0 l ir im)))

theorem ropeBody_regs6 (cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (st : WSt) (l : Lane) :
    ((ropeBodyEW.elabAt cta 0 ir im).run st).regs 6 l = ropeLoVal st.mem cta ir im l := rfl

theorem ropeBody_regs7 (cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (st : WSt) (l : Lane) :
    ((ropeBodyEW.elabAt cta 0 ir im).run st).regs 7 l = ropeHiVal st.mem cta ir im l := rfl

theorem ropeLo_eval (cta : Nat) (l : Lane) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) :
    ropeLoIx.eval cta 0 l ir im = cta * HEAD_DIM + l.val := rfl

theorem ropeHi_eval (cta : Nat) (l : Lane) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) :
    ropeHiIx.eval cta 0 l ir im = cta * HEAD_DIM + l.val + HALF := rfl

/-- **The low half lands at the low address.** -/
theorem rope_stores_lo (cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (st : WSt) (l0 : Lane) :
    ((ropeEW.elabAt cta 0 ir im).run st).mem 0 (ropeLoIx.eval cta 0 l0 ir im)
      = ropeLoVal st.mem cta ir im l0 := by
  show ((WStmt.storeLane 0 (fun l => ropeHiIx.eval cta 0 l ir im) 7).run
          ((WStmt.storeLane 0 (fun l => ropeLoIx.eval cta 0 l ir im) 6).run
            ((ropeBodyEW.elabAt cta 0 ir im).run st))).mem 0
              (ropeLoIx.eval cta 0 l0 ir im) = _
  rw [storeLane_two_first 0 (fun l => ropeLoIx.eval cta 0 l ir im)
        (fun l => ropeHiIx.eval cta 0 l ir im) 6 7 _ l0
        (fun l => by
          have h1 : l.val < 32 := l.isLt
          have h2 : l0.val < 32 := l0.isLt
          show cta * HEAD_DIM + l.val + HALF ≠ cta * HEAD_DIM + l0.val
          simp only [HALF, HEAD_DIM]
          omega)
        (fun l h => by
          have hb : (cta * HEAD_DIM + l.val : Nat) = cta * HEAD_DIM + l0.val := h
          have : l = l0 := Fin.ext (by omega)
          rw [this])]
  exact ropeBody_regs6 cta ir im st l0

/-- **…and the high half at the high address.** -/
theorem rope_stores_hi (cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (st : WSt) (l0 : Lane) :
    ((ropeEW.elabAt cta 0 ir im).run st).mem 0 (ropeHiIx.eval cta 0 l0 ir im)
      = ropeHiVal st.mem cta ir im l0 := by
  show ((WStmt.storeLane 0 (fun l => ropeHiIx.eval cta 0 l ir im) 7).run
          ((WStmt.storeLane 0 (fun l => ropeLoIx.eval cta 0 l ir im) 6).run
            ((ropeBodyEW.elabAt cta 0 ir im).run st))).mem 0
              (ropeHiIx.eval cta 0 l0 ir im) = _
  rw [storeLane_two_second 0 (fun l => ropeLoIx.eval cta 0 l ir im)
        (fun l => ropeHiIx.eval cta 0 l ir im) 6 7 _ l0
        (fun l h => by
          have hb : (cta * HEAD_DIM + l.val + HALF : Nat)
              = cta * HEAD_DIM + l0.val + HALF := h
          have : l = l0 := Fin.ext (by omega)
          rw [this])]
  exact ropeBody_regs7 cta ir im st l0

def ptxRope : String := emitProvenKernelN "main" 3 0 ropeEW

theorem rope_idxBelow : (expandEW ropeEW).IdxBelow 3 :=
  expandEW_idxBelow 3 ropeEW (by decide)

/-- **The emitted RoPE runs its statement, from raw launch.**  Q and K use the
    same kernel — the only difference is which buffer is bound and how many
    heads are launched. -/
theorem rope_ptx_exact (cta : Nat) (m : MState) :
    ∃ k m', steps cta (flatKernel (expandEW ropeEW)) k (0, m)
          = some ((flatKernel (expandEW ropeEW)).length, m')
      ∧ m'.toWSt
        = ((expandEW ropeEW).elabAt cta 0
            (SI.stepL cta emitPrologue m).ir m.imem).run m.toWSt :=
  flatKernel_sound cta (expandEW ropeEW) (expandEW_expFree ropeEW) rope_idxBelow
    (expandEW_flat ropeEW (by decide)) m

-- ---------------------------------------------------------------------------
-- softmax:  p[i] = exp(s[i] − max) / Σ exp(s[j] − max)
-- ---------------------------------------------------------------------------

/-! The kernel the whole dynamic-loop design was for.  Its trip count is the
    sequence length, which is not known until the token is being generated — so
    it is `EWStmt.forM`, reading the bound from integer memory where it is
    lane-uniform by construction.

    Three passes, and **no shared memory and no barrier**: as with RMSNorm, the
    butterfly leaves the result in every lane, so the eight-warp hand-off the
    hand-written kernel needs simply does not arise.

    ## The tail, without a mask

    A warp is 32 lanes and `seqLen` is not a multiple of 32.  The usual fix is a
    predicated accumulate, which needs a float select this instruction set does
    not have.  Instead each pass is split in two:

    * a **chunk loop** over `⌊seqLen/32⌋` full 32-lane groups — every lane holds
      a distinct element, reduced by a butterfly, and no lane is out of range;
    * a **remainder loop** over the last `seqLen mod 32` elements, run *after*
      the butterfly, in which every lane reads the **same** element.  Each lane
      therefore folds in the identical remainder, so the accumulator stays equal
      across the warp without a second reduction — and each element is counted
      exactly once.

    That is exact for every sequence length, costs at most 31 extra iterations,
    and needs no padding, no layout change and no new opcode.  In the store pass
    the remainder lanes write the same value to the same address, which is what
    `stG`'s fold over lanes already denotes.

    The host publishes `seqLen`, `⌊seqLen/32⌋`, `32·⌊seqLen/32⌋` and
    `seqLen mod 32`; the scores for head `ctaid` begin at `ctaid · seqLen`,
    a product of a block index and a memory-read value — expressible only
    because `IdxE` has `ldIdx`.

    Proven all the way to the PTX floor: `flatEW_sound` compiles `forM` to the
    seven-instruction memory-bounded shape — `mov` the address, `ld.global.u32`
    the bound, then init/test/exit/body/increment/back-edge — and the exit
    branch is never divergent because `imem` has no lane index. -/

/-- Meta-buffer slots the host publishes for the dynamic loop. -/
def SEQ_SLOT    : Nat := 2   -- seqLen
def CHUNKS_SLOT : Nat := 3   -- ⌊seqLen / 32⌋
def TAIL_SLOT   : Nat := 4   -- 32 · ⌊seqLen / 32⌋
def REM_SLOT    : Nat := 5   -- seqLen mod 32

/-- `seqLen`, as an address component. -/
def seqLenIx : IdxE := .ldIdx 1 (.lit SEQ_SLOT)

/-- This head's row: `ctaid · seqLen`. -/
def rowBaseIx : IdxE := .mul .ctaId seqLenIx

/-- Chunk loop: element `loop·32 + lane` of the row. -/
def smIx : IdxE := .add rowBaseIx (.add (.mul .loopI (.lit 32)) .laneId)

/-- Remainder loop: element `32·⌊seqLen/32⌋ + loop`, the *same* one in every
    lane — which is what makes the un-reduced fold below exact. -/
def smTailIx : IdxE := .add rowBaseIx (.add (.ldIdx 1 (.lit TAIL_SLOT)) .loopI)

/-- `%fw0 ← max(%fw0, s[ix])` — `sweepBody` at the max combiner. -/
private def maxF : WFExp := .maxW (.reg 0) (.reg 2)

/-- `%fw0 ← %fw0 + exp(s[ix] − %fw5)` — `sweepBody` at the sum combiner.  It
    reads `%fw5`, which is why the sweep frame has to name kept registers. -/
private def sumF : WFExp := .add (.reg 0) (.exp (.add (.reg 2) (.neg (.reg 5))))

/-- The sum butterfly, as the schema's reduction argument. -/
def smSumBfly : EWStmt :=
  .seq (warpRoundE 16) (.seq (warpRoundE 8) (.seq (warpRoundE 4)
    (.seq (warpRoundE 2) (warpRoundE 1))))

/-- `probs[ix] ← exp(s[ix] − %fw5) · %fw3`, the body both store sweeps share.
    The computation is split from the store so it can be read back; the emitted
    instructions are unchanged, `seq` being list append. -/
def normCompute (ix : IdxE) : EWStmt :=
  .seq (.loadIdx 2 0 ix)
       (.setR 4 (.mul (.exp (.add (.reg 2) (.neg (.reg 5)))) (.reg 3)))

def normStep (ix : IdxE) : EWStmt := storeBody 2 4 ix (normCompute ix)

/-- Pass 1: the row maximum, in every lane.  `%fw5` keeps it — *not* `%fw1`,
    which every butterfly round below uses as its shuffle temporary. -/
def smMaxReduce : EWStmt :=
  chunkRemReduce 0 2 0 maxF smIx smTailIx (.lit (-100000000.0))
    1 CHUNKS_SLOT REM_SLOT warpReduceMaxE

/-- **The shipped pass *is* the library wrapper** — definitionally.

    `A38` cites this equality as its evidence that `maxReduceEW` and
    `sumReduceEW` are the shapes the stack actually uses rather than shapes
    invented beside them.  It was cited but **never written down** (`A47` G22):
    the ledger asserted a build-checked `rfl` that was not in the build.

    It is here now, so the claim and the artifact are the same object. -/
example : smMaxReduce
    = maxReduceEW 0 smIx smTailIx 1 CHUNKS_SLOT REM_SLOT (-100000000.0) := rfl

def smMax : EWStmt := .seq smMaxReduce (.setR 5 (.reg 0))

/-- Pass 2: `Σ exp(s − max)`, in every lane — the *same* schema, at `add`. -/
def smSum : EWStmt :=
  chunkRemReduce 0 2 0 sumF smIx smTailIx (.lit (NumOps.ofNat 0))
    1 CHUNKS_SLOT REM_SLOT smSumBfly

/-- **The sum pass is deliberately *not* `sumReduceEW`.**

    Its combiner is `acc + exp(x − max)` (`sumF`), fusing the exponential into
    the reduction and reading the row maximum from `%fw5` — which is why it uses
    the general `chunkRemReduce` with `keep = [5]` rather than the plain-sum
    wrapper.  Writing `smSum = sumReduceEW …` fails to typecheck, and that is
    the correct outcome: the shapes genuinely differ.

    Recorded because `A38` was loose about it, implying both softmax passes were
    wrapper instances when only the maximum is (`A47` G22). -/
example : smSumBfly
    = (.seq (warpRoundE 16) (.seq (warpRoundE 8) (.seq (warpRoundE 4)
        (.seq (warpRoundE 2) (warpRoundE 1))))) := rfl

/-- Pass 3: normalise and store.  `%fw5` is the max, `%fw0` the sum. -/
def smNorm : EWStmt :=
  .seq (.setR 3 (.inv (.reg 0)))
       (.seq (.forM 1 CHUNKS_SLOT (normStep smIx))
             (.forM 1 REM_SLOT (normStep smTailIx)))

def softmaxEW : EWStmt := .seq (.seq smMax smSum) smNorm

/-! ### Where softmax stores

    The chunk pass is the usual shape.  The **remainder** pass is the one that
    justifies stating `storeLoop_at` with agreement rather than injectivity:
    every lane reads and writes the *same* element there, so its address map is
    emphatically not injective — but all 32 lanes store the same value, so the
    agreement hypothesis holds and the theorem applies unchanged.  An
    injectivity-based statement would have excluded the kernel we ship. -/

/-- `exp(s[ix] − max) · invSum`, the value both store passes write. -/
def smNormVal (mem : Nat → Float32) (mx inv : Lane → Float32) (a : Nat) (l : Lane) :
    Float32 :=
  NumOps.mul (NumOps.exp (NumOps.add (mem a) (NumOps.neg (mx l)))) (inv l)

/-- **The chunk store pass lands its value at the right address.** -/
theorem softmax_stores_chunk (cta i : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) (st : WSt) (j0 : Nat) (l0 : Lane)
    (hj0 : j0 < im 1 CHUNKS_SLOT) :
    (((EWStmt.forM 1 CHUNKS_SLOT (normStep smIx)).elabAt cta i ir im).run st).mem
        2 (smIx.eval cta j0 l0 ir im)
      = smNormVal (st.mem 0) (st.regs 5) (st.regs 3) (smIx.eval cta j0 l0 ir im) l0 := by
  refine storeLoop_at 2 4 smIx (normCompute smIx) cta ir im st
    (fun j l => smNormVal (st.mem 0) (st.regs 5) (st.regs 3) (smIx.eval cta j l ir im) l)
    (smIx.eval cta j0 l0 ir im)
    (smNormVal (st.mem 0) (st.regs 5) (st.regs 3) (smIx.eval cta j0 l0 ir im) l0) [5, 3]
    (fun _ _ => rfl)
    (fun _ s r' hr' => by
      have : r' = 5 ∨ r' = 3 := by simp at hr'; omega
      rcases this with h | h <;> (subst h; rfl))
    (fun j s hinv hkeep l => by
      show NumOps.mul (NumOps.exp (NumOps.add (s.mem 0 (smIx.eval cta j l ir im))
              (NumOps.neg (s.regs 5 l)))) (s.regs 3 l) = _
      rw [hinv 0 (by decide), hkeep 5 (by simp), hkeep 3 (by simp)]
      rfl)
    (List.range (im 1 CHUNKS_SLOT)) st (fun _ _ => rfl) (fun _ _ => rfl) ?_ ?_
  · -- distinct (chunk, lane) pairs address distinct elements
    intro j hj l hl
    have hb : (rowBaseIx.eval cta j l ir im + (j * 32 + l.val) : Nat)
        = rowBaseIx.eval cta j0 l0 ir im + (j0 * 32 + l0.val) := hl
    have hrow : rowBaseIx.eval cta j l ir im = rowBaseIx.eval cta j0 l0 ir im := rfl
    rw [hrow] at hb
    have h1 : l.val < 32 := l.isLt
    have h2 : l0.val < 32 := l0.isLt
    have hjl : j = j0 ∧ l.val = l0.val := by omega
    have : l = l0 := Fin.ext hjl.2
    rw [hjl.1, this]
  · exact Or.inl ⟨j0, List.mem_range.mpr hj0, l0, rfl⟩

/-- **The remainder store pass lands its value too** — with every lane writing
    the same address, which is exactly the case agreement covers and injectivity
    would not. -/
theorem softmax_stores_tail (cta i : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) (st : WSt) (j0 : Nat) (l0 : Lane)
    (hj0 : j0 < im 1 REM_SLOT) (huni : ∀ l : Lane, st.regs 5 l = st.regs 5 l0)
    (huni3 : ∀ l : Lane, st.regs 3 l = st.regs 3 l0) :
    (((EWStmt.forM 1 REM_SLOT (normStep smTailIx)).elabAt cta i ir im).run st).mem
        2 (smTailIx.eval cta j0 l0 ir im)
      = smNormVal (st.mem 0) (st.regs 5) (st.regs 3) (smTailIx.eval cta j0 l0 ir im) l0 := by
  refine storeLoop_at 2 4 smTailIx (normCompute smTailIx) cta ir im st
    (fun j l => smNormVal (st.mem 0) (st.regs 5) (st.regs 3) (smTailIx.eval cta j l ir im) l)
    (smTailIx.eval cta j0 l0 ir im)
    (smNormVal (st.mem 0) (st.regs 5) (st.regs 3) (smTailIx.eval cta j0 l0 ir im) l0) [5, 3]
    (fun _ _ => rfl)
    (fun _ s r' hr' => by
      have : r' = 5 ∨ r' = 3 := by simp at hr'; omega
      rcases this with h | h <;> (subst h; rfl))
    (fun j s hinv hkeep l => by
      show NumOps.mul (NumOps.exp (NumOps.add (s.mem 0 (smTailIx.eval cta j l ir im))
              (NumOps.neg (s.regs 5 l)))) (s.regs 3 l) = _
      rw [hinv 0 (by decide), hkeep 5 (by simp), hkeep 3 (by simp)]
      rfl)
    (List.range (im 1 REM_SLOT)) st (fun _ _ => rfl) (fun _ _ => rfl) ?_ ?_
  · -- the address does not mention the lane, so all 32 lanes agree
    intro j hj l hl
    have hb : (rowBaseIx.eval cta j l ir im + (im 1 TAIL_SLOT + j) : Nat)
        = rowBaseIx.eval cta j0 l0 ir im + (im 1 TAIL_SLOT + j0) := hl
    have hrow : rowBaseIx.eval cta j l ir im = rowBaseIx.eval cta j0 l0 ir im := rfl
    rw [hrow] at hb
    have hjj : j = j0 := by omega
    show smNormVal (st.mem 0) (st.regs 5) (st.regs 3) (smTailIx.eval cta j l ir im) l = _
    have haddr : smTailIx.eval cta j0 l ir im = smTailIx.eval cta j0 l0 ir im := rfl
    rw [hjj]
    simp only [smNormVal, haddr, huni l, huni3 l]
  · exact Or.inl ⟨j0, List.mem_range.mpr hj0, l0, rfl⟩

/-! ### What the two reductions compute

    Both passes are instances of `chunkRemReduce_spec`, so neither needed a
    proof of its own — the chunk sweep, the butterfly and the un-reduced
    remainder fold are established once in `Schema.lean` and applied twice
    here.  The trip counts are values in `im`, so **one statement covers every
    sequence length**.

    This is the layer that was missing when the draft shipped a 32× index
    overrun and an accumulator in the butterfly's scratch register: every
    execution theorem held, because none of them mentioned what the kernel was
    supposed to compute. -/

/-- Pass 1's result: chunk sweep per lane, max butterfly, then the shared
    remainder fold — in that order. -/
def smMaxFold (mem : Nat → Float32) (cta : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) (l : Lane) : Float32 :=
  chunkRemFold mem (fun _ a x => NumOps.max a x) (fun _ => (-100000000.0))
    (fun j l' => smIx.eval cta j l' ir im) (fun j l' => smTailIx.eval cta j l' ir im)
    (im 1 CHUNKS_SLOT) (im 1 REM_SLOT)
    (bflyFoldOp (fun a b => NumOps.max a b)) l

/-- **Pass 1 leaves the row maximum in `%fw5`, in every lane.** -/
theorem softmax_max_spec (cta i : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (st : WSt) (l : Lane) :
    ((smMax.elabAt cta i ir im).run st).regs 5 l = smMaxFold (st.mem 0) cta ir im l := by
  show ((smMaxReduce.elabAt cta i ir im).run st).regs 0 l = _
  exact chunkRemReduce_spec 0 2 0 (by decide) maxF smIx smTailIx (.lit (-100000000.0))
    1 CHUNKS_SLOT REM_SLOT cta i warpReduceMaxE ir im
    (fun _ a x => NumOps.max a x) (bflyFoldOp (fun a b => NumOps.max a b))
    [] (fun _ h => absurd h (by simp)) st (fun _ => (-100000000.0)) (fun _ => rfl)
    (fun _ _ _ => rfl)
    (fun st' => warpReduceMaxE_spec cta i ir im st')
    (fun st' => warpReduceMaxE_mem cta i ir im st')
    (fun _ r h => absurd h (by simp)) l

/-- Pass 2's result: the same schema at `add`, with `%fw5` (the row maximum)
    read by the combiner and therefore *kept* across the butterfly. -/
def smSumFold (mem : Nat → Float32) (mx : Lane → Float32) (cta : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (l : Lane) : Float32 :=
  chunkRemFold mem
    (fun l' a x => NumOps.add a (NumOps.exp (NumOps.add x (NumOps.neg (mx l')))))
    (fun _ => NumOps.ofNat 0)
    (fun j l' => smIx.eval cta j l' ir im) (fun j l' => smTailIx.eval cta j l' ir im)
    (im 1 CHUNKS_SLOT) (im 1 REM_SLOT)
    (bflyFoldOp (fun a b => NumOps.add a b)) l

/-- **Pass 2 leaves `Σ exp(s − max)` in `%fw0`, in every lane.**

    `keep = [5]` is the whole subtlety: the combiner reads the row maximum, the
    butterfly between the two sweeps clobbers `%fw1`, and the frame has to
    permit the second while guaranteeing the first. -/
theorem softmax_sum_spec (cta i : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (st : WSt) (l : Lane) :
    ((smSum.elabAt cta i ir im).run st).regs 0 l
      = smSumFold (st.mem 0) (st.regs 5) cta ir im l :=
  chunkRemReduce_spec 0 2 0 (by decide) sumF smIx smTailIx (.lit (NumOps.ofNat 0))
    1 CHUNKS_SLOT REM_SLOT cta i smSumBfly ir im
    (fun l' a x => NumOps.add a (NumOps.exp (NumOps.add x (NumOps.neg (st.regs 5 l')))))
    (bflyFoldOp (fun a b => NumOps.add a b))
    [5] (fun r h => by simp at h; omega) st (fun _ => NumOps.ofNat 0) (fun _ => rfl)
    (fun st' hfr l' => by
      show NumOps.add (st'.regs 0 l') (NumOps.exp (NumOps.add (st'.regs 2 l')
              (NumOps.neg (st'.regs 5 l')))) = _
      rw [hfr.2 5 (by simp)])
    (fun st' => warpReduceSumE_spec cta i ir im st')
    (fun st' => warpReduceSumE_mem cta i ir im st')
    (fun st' r h => by
      have : r = 5 := by simp at h; omega
      subst this
      exact warpReduceSumE_frame cta i ir im 5 (by decide) (by decide) st') l

-- ---------------------------------------------------------------------------
-- argmax:  the token id of the largest logit
-- ---------------------------------------------------------------------------

/-! The last kernel, and the one that needed the instruction set widened.

    Its arithmetic mixes an element's **value** with the element's **index**,
    and until `EWStmt.cvtIF` there was no way to put an index where a float
    expression could see it.  That one instruction — `cvt.rn.f32.u32` — is the
    whole extension, and it is exact here: `Float32` represents every integer
    below 2^24, and `VOCAB = 151936` is far below it.

    The output is a float, converted to a token id by the host.  That is *not*
    a workaround for a missing store: it is what keeps `MState.imem` read-only,
    and read-only integer memory is precisely what buys the frame-freedom that
    makes `ldIdx` and `forM` cheap.  Paying for mutable integer memory to serve
    one kernel would have been a bad trade.

    ## First-occurrence argmax without a select

    Two passes, both `chunkRemReduce`-free because `VOCAB` is static and
    `151936 = 4748 · 32` exactly — no remainder, so a plain sweep suffices.

    1. `%fw5 ← max_i v_i`, the usual sweep and butterfly.
    2. `%fw0 ← −min_i (i + BIG·(1 − ge(v_i, max)))`.  `geF` is `1.0` exactly at
       the maxima, so non-maxima are pushed above `BIG` and the minimum picks
       the *first* index achieving the max — matching the hand-written kernel's
       strict `>` update.  `min` is `−max(−·)`, so the same proven butterfly
       serves, now at its third operation. -/

def VOCAB : Nat := 151936

/-- `BIG` exceeds any index, and `BIG + VOCAB < 2^24`, so every value the
    second pass forms is still an exact `Float32` integer. -/
def BIG : Float32 := 1048576.0

/-- Element `loop·32 + lane` of the logits. -/
def vocabIx : IdxE := .add (.mul .loopI (.lit 32)) .laneId

/-- Pass 1: the row maximum, in every lane, then saved to `%fw5`. -/
def amMax : EWStmt :=
  .seq (.seq (.seq (.setR 0 (.lit (-100000000.0)))
                   (.forN (VOCAB / 32) (sweepBody 0 2 0 (.maxW (.reg 0) (.reg 2)) vocabIx)))
             warpReduceMaxE)
       (.setR 5 (.reg 0))

/-- `%fw0 ← max(%fw0, −(idx + BIG·(1 − ge(v, max))))` — the negated candidate,
    so the shared `max` butterfly computes a minimum. -/
def amStep : WFExp :=
  .maxW (.reg 0)
        (.neg (.add (.reg 3)
                    (.mul (.lit BIG)
                          (.add (.lit 1.0) (.neg (.geF (.reg 2) (.reg 5)))))))

/-- Pass 2's body: materialise the index as a float, load the value, combine. -/
def amBody : EWStmt :=
  .seq (.seq (.cvtIF 3 vocabIx) (.loadIdx 2 0 vocabIx)) (.setR 0 amStep)

/-- Pass 2, then undo the negation and store from lane 0. -/
def argmaxEW : EWStmt :=
  .seq (.seq amMax
             (.seq (.seq (.setR 0 (.lit (-100000000.0)))
                         (.forN (VOCAB / 32) amBody))
                   warpReduceMaxE))
       (.seq (.setR 4 (.neg (.reg 0))) (.storeLane0 1 (.lit 0) 4))

def ptxArgmax : String := emitProvenKernelN "main" 2 0 argmaxEW

/-! ### What argmax computes

    `A47` G15: this kernel shipped with `argmax_ptx_exact` — the PTX performs
    the statement — and **no theorem saying the statement computes anything**.
    That is precisely the gap `A36` exists to name: execution correctness
    certified the buggy softmax draft for as long as nothing said what softmax
    *meant*.

    The candidate a lane forms for element `e` is `−(e + BIG·(1 − [v ≥ m]))`:
    the negated index when the value ties the row maximum `m`, and a very
    negative number otherwise.  Maximising that and negating gives the
    **smallest index achieving the maximum** — which is why the kernel can reuse
    the `max` butterfly for what is really a minimum. -/

/-- The candidate value for element at address `a`, given the row maximum. -/
def amCand (mem : Nat → Float32) (m : Float32) (a : Nat) : Float32 :=
  NumOps.neg (NumOps.add (NumOps.ofNat a)
    (NumOps.mul BIG (NumOps.add 1.0 (NumOps.neg (NumOps.ifGe (mem a) m 1.0 0.0)))))

/-- **The second pass folds the candidates, in load order.**

    Registers `2` and `3` are scratch and `5` holds the row maximum, which the
    body reads but never writes — the frame condition that makes the fold's
    value depend only on the entry state. -/
theorem amBody_fold (cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (st0 : WSt) :
    ∀ (L : List Nat) (st : WSt), st.regs 5 = st0.regs 5 → st.mem = st0.mem →
      ∀ (l : Lane),
      ((L.foldl (fun s j => (amBody.elabAt cta j ir im).run s) st).regs 0 l)
        = L.foldl
            (fun acc j => NumOps.max acc
              (amCand (st0.mem 0) (st0.regs 5 l) (vocabIx.eval cta j l ir im)))
            (st.regs 0 l) := by
  intro L
  induction L with
  | nil => intro _ _ _ _; rfl
  | cons j L ih =>
      intro st h5 hm l
      have hstep : (amBody.elabAt cta j ir im).run st
          = (st.setReg 3 (fun l' => NumOps.ofNat (vocabIx.eval cta j l' ir im))
              |>.setReg 2 (fun l' => st.mem 0 (vocabIx.eval cta j l' ir im))
              |>.setReg 0 (fun l' => WFExp.eval _ l' amStep)) := rfl
      rw [List.foldl_cons]
      refine (ih _ ?_ ?_ l).trans ?_
      · show ((amBody.elabAt cta j ir im).run st).regs 5 = _
        rw [hstep]
        rw [WSt.regs_setReg_other _ 0 5 _ (by decide),
            WSt.regs_setReg_other _ 2 5 _ (by decide)]
        rw [WSt.regs_setReg_other _ 3 5 _ (by decide)]
        exact h5
      · show ((amBody.elabAt cta j ir im).run st).mem = _
        rw [hstep]
        rw [WSt.mem_setReg, WSt.mem_setReg]
        rw [WSt.mem_setReg]
        exact hm
      · rw [List.foldl_cons]
        congr 1
        -- the state the combiner sees: index in %fw3, value in %fw2
        have hpre : (EWStmt.elabAt cta j ir im
              ((EWStmt.cvtIF 3 vocabIx).seq (EWStmt.loadIdx 2 0 vocabIx))).run st
            = (st.setReg 3 (fun l' => NumOps.ofNat (vocabIx.eval cta j l' ir im))).setReg 2
                (fun l' => st.mem 0 (vocabIx.eval cta j l' ir im)) := rfl
        show ((amBody.elabAt cta j ir im).run st).regs 0 l = _
        rw [hstep, WSt.regs_setReg_same]
        show NumOps.max _ _ = NumOps.max _ _
        congr 1
        rw [hpre]
        unfold amCand
        simp only [WFExp.eval, WSt.regs_setReg_same,
                   WSt.regs_setReg_other _ 2 3 _ (by decide),
                   WSt.regs_setReg_other _ 2 5 _ (by decide),
                   WSt.regs_setReg_other _ 3 5 _ (by decide), h5, hm]

/-- The scan-and-store half of argmax, as its own statement.

    Stated separately because the whole kernel's state term is large enough
    that elaborating a `show` about it exhausts the stack — and because this is
    the half with the content.  Composition with pass 1 is definitional. -/
def amScanStore (K : Nat) : EWStmt :=
  .seq (.seq (.seq (.setR 0 (.lit (-100000000.0)))
                   (.forN K amBody))
             warpReduceMaxE)
       (.seq (.setR 4 (.neg (.reg 0))) (.storeLane0 1 (.lit 0) 4))

/-- **The scan half stores the negated butterfly-max of the candidates.** -/
theorem amScanStore_stores (K cta : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) (s : WSt) :
    (((amScanStore K).elabAt cta 0 ir im).run s).mem 1 0
      = NumOps.neg (bflyFoldOp (fun a c => NumOps.max a c)
          (fun l => (List.range K).foldl
            (fun acc j => NumOps.max acc
              (amCand (s.mem 0) (s.regs 5 l) (vocabIx.eval cta j l ir im)))
            (-100000000.0 : Float32))
          ⟨0, by decide⟩) := by
  have key : ∀ s' : WSt,
      ((WStmt.storeLane0 1 0 4).run ((WStmt.setR 4 (.neg (.reg 0))).run s')).mem 1 0
        = NumOps.neg (s'.regs 0 ⟨0, by decide⟩) := by
    intro s'
    rw [wrun_storeLane0, WSt.mem_store1_same, wrun_setR, WSt.regs_setReg_same]
    rfl
  show ((WStmt.storeLane0 1 0 4).run
      ((WStmt.setR 4 (.neg (.reg 0))).run
        ((warpReduceMaxE.elabAt cta 0 ir im).run
          ((WStmt.forN K (fun j => amBody.elabAt cta j ir im)).run
            ((WStmt.setR 0 (.lit (-100000000.0))).run s))))).mem 1 0 = _
  rw [key]
  show NumOps.neg _ = NumOps.neg _
  congr 1
  rw [warpReduceMaxE_spec cta 0 ir im]
  show bflyFoldOp _ _ _ = bflyFoldOp _ _ _
  congr 1
  funext l
  show ((WStmt.forN K (fun j => amBody.elabAt cta j ir im)).run
      ((WStmt.setR 0 (.lit (-100000000.0))).run s)).regs 0 l = _
  rw [wrun_forN, amBody_fold cta ir im ((WStmt.setR 0 (.lit (-100000000.0))).run s)
      (List.range K) _ rfl rfl l]
  simp only [wrun_setR, WSt.regs_setReg_same, WFExp.eval, WSt.mem_setReg,
             WSt.regs_setReg_other _ 0 5 _ (by decide)]

theorem argmax_stores (cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (st : WSt) :
    ((argmaxEW.elabAt cta 0 ir im).run st).mem 1 0
      = NumOps.neg (bflyFoldOp (fun a c => NumOps.max a c)
          (fun l => (List.range (VOCAB / 32)).foldl
            (fun acc j => NumOps.max acc
              (amCand (((amMax.elabAt cta 0 ir im).run st).mem 0)
                (((amMax.elabAt cta 0 ir im).run st).regs 5 l)
                (vocabIx.eval cta j l ir im)))
            (-100000000.0 : Float32))
          ⟨0, by decide⟩) :=
  amScanStore_stores (VOCAB / 32) cta ir im ((amMax.elabAt cta 0 ir im).run st)

theorem argmax_idxBelow : (expandEW argmaxEW).IdxBelow 3 :=
  expandEW_idxBelow 3 argmaxEW (by decide)

/-- **The emitted argmax runs its statement, from raw launch.**  The last of the
    eleven kernels to reach the PTX floor. -/
theorem argmax_ptx_exact (cta : Nat) (m : MState) :
    ∃ k m', steps cta (flatKernel (expandEW argmaxEW)) k (0, m)
          = some ((flatKernel (expandEW argmaxEW)).length, m')
      ∧ m'.toWSt
        = ((expandEW argmaxEW).elabAt cta 0
            (SI.stepL cta emitPrologue m).ir m.imem).run m.toWSt :=
  flatKernel_sound cta (expandEW argmaxEW) (expandEW_expFree argmaxEW)
    argmax_idxBelow (expandEW_flat argmaxEW (by decide)) m

/-- **The softmax kernel runs its statement on the structured machine**, for any
    sequence length the meta buffer names.  Every construct it uses — the
    dynamic loop, the gathered stride, `max`, `exp` — is covered. -/
theorem softmax_structured (cta lr n i : Nat) (m : MState)
    (hf : (expandEW softmaxEW).ExpFree) (h2 : 2 ≤ n) (hlr : lr < n)
    (hb : (expandEW softmaxEW).IdxBelow n) (hinv : MInv cta i lr m) :
    (SI.stepL cta (emitEW lr n (expandEW softmaxEW)) m).toWSt
      = ((expandEW softmaxEW).elabAt cta i m.ir m.imem).run m.toWSt :=
  emitEW_sound cta (expandEW softmaxEW) hf lr n i m h2 hlr hb hinv

def ptxSoftmax : String := emitProvenKernelN "main" 3 0 softmaxEW

-- ---------------------------------------------------------------------------
-- Seam guards for the shipped model
-- ---------------------------------------------------------------------------

theorem qwen2_bufs_bound :
    addKernel.BufBelow 2
      ∧ siluKernel.BufBelow 3
      ∧ embedKernelEW.BufBelow 3
      ∧ kvStoreEW.BufBelow 3
      ∧ rmsKernelEW.BufBelow 3
      ∧ ropeEW.BufBelow 3
      ∧ argmaxEW.BufBelow 2
      ∧ softmaxEW.BufBelow 3 := by decide

/-- **Every branch in every shipped kernel resolves**, and **nothing
    unrenderable reaches the printer.**  Together with `programText_label`,
    that is the whole of what the trusted printer is held to. -/
theorem qwen2_emit_ok :
    FlatTargetsOkB (flatKernel (expandEW addKernel)) = true
      ∧ FlatTargetsOkB (flatKernel (expandEW siluKernel)) = true
      ∧ FlatTargetsOkB (flatKernel (expandEW embedKernelEW)) = true
      ∧ FlatTargetsOkB (flatKernel (expandEW kvStoreEW)) = true
      ∧ FlatTargetsOkB (flatKernel (expandEW rmsKernelEW)) = true
      ∧ FlatTargetsOkB (flatKernel (expandEW ropeEW)) = true
      ∧ FlatTargetsOkB (flatKernel (expandEW argmaxEW)) = true
      ∧ FlatTargetsOkB (flatKernel (expandEW softmaxEW)) = true
      ∧ FlatPrintableB (flatKernel (expandEW addKernel)) = true
      ∧ FlatPrintableB (flatKernel (expandEW siluKernel)) = true
      ∧ FlatPrintableB (flatKernel (expandEW embedKernelEW)) = true
      ∧ FlatPrintableB (flatKernel (expandEW kvStoreEW)) = true
      ∧ FlatPrintableB (flatKernel (expandEW rmsKernelEW)) = true
      ∧ FlatPrintableB (flatKernel (expandEW ropeEW)) = true
      ∧ FlatPrintableB (flatKernel (expandEW argmaxEW)) = true
      ∧ FlatPrintableB (flatKernel (expandEW softmaxEW)) = true := by
  native_decide

/-- **Every shipped kernel writes the buffer it claims.**

    The necessary condition for being a `StageSpec`: a wrong buffer number
    fails here rather than producing a stage whose `out` nothing writes. -/
theorem qwen2_stage_eligible :
    siluKernel.StageEligibleB 2 = true
      ∧ embedKernelEW.StageEligibleB 2 = true
      ∧ rmsKernelEW.StageEligibleB 2 = true
      ∧ softmaxEW.StageEligibleB 2 = true
      ∧ argmaxEW.StageEligibleB 1 = true
      ∧ kvStoreEW.StageEligibleB 1 = true := by decide

/-- **Six of the eight are idempotent; the residual add and RoPE are not.**

    `addKernel` writes buffer `0` and also reads it — `out[i] = out[i] + b[i]`
    — and RoPE likewise rotates in place.  Running such a block twice lands
    `out[i] + 2·b[i]`, so `StageSpec.Idempotent` is false of them.

    That is a statement about *re-running*, not about being a stage.  A grid
    runs each block once, so both are perfectly good stages and `runGrid_value`
    covers them; what they cannot join is the arbitrary-list and permutation
    results, which tolerate a block appearing twice. `compileWKernel` loads
    every input before storing any, so the in-place update is sound within a
    block, and `add_stores` proves it. -/
theorem qwen2_idempotent :
    siluKernel.IdempotentEligibleB 2 = true
      ∧ embedKernelEW.IdempotentEligibleB 2 = true
      ∧ rmsKernelEW.IdempotentEligibleB 2 = true
      ∧ softmaxEW.IdempotentEligibleB 2 = true
      ∧ argmaxEW.IdempotentEligibleB 1 = true
      ∧ kvStoreEW.IdempotentEligibleB 1 = true := by decide

/-- The two in-place kernels, named rather than discovered later. -/
theorem qwen2_in_place :
    addKernel.IdempotentEligibleB 0 = false
      ∧ ropeEW.IdempotentEligibleB 0 = false := by decide

theorem softmax_idxBelow : (expandEW softmaxEW).IdxBelow 3 :=
  expandEW_idxBelow 3 softmaxEW (by decide)

/-- **The emitted softmax runs its statement, from raw launch — including the
    dynamic loop.**

    The three sweeps are `forM`s whose trip count is read from the meta buffer,
    so this one theorem covers *every* sequence length at once: the bound is a
    value in the machine state, not a parameter of the proof.  That is the whole
    reason the loop reads memory rather than a register — `imem` has no lane
    index, so the guard is uniform and the exit branch cannot diverge. -/
theorem softmax_ptx_exact (cta : Nat) (m : MState) :
    ∃ k m', steps cta (flatKernel (expandEW softmaxEW)) k (0, m)
          = some ((flatKernel (expandEW softmaxEW)).length, m')
      ∧ m'.toWSt
        = ((expandEW softmaxEW).elabAt cta 0
            (SI.stepL cta emitPrologue m).ir m.imem).run m.toWSt :=
  flatKernel_sound cta (expandEW softmaxEW) (expandEW_expFree softmaxEW)
    softmax_idxBelow (expandEW_flat softmaxEW (by decide)) m

-- ---------------------------------------------------------------------------
-- The kernels as pipeline stages
-- ---------------------------------------------------------------------------

/-! Everything above proves what a kernel *does*: which PTX it lowers to, what
    it leaves in a register, where its store lands.  None of it says the kernel
    is a **stage** — a memory-to-memory function that can be composed with the
    next one.  `StageSpec` is that interface, and until `StageFrame.lean` the
    only kernels that could meet it were the ones a schedule generated.

    These are the hand-written ones, and they are the whole model.  Each is
    `stageOfEW` applied to the kernel's own store theorem. -/

namespace Stage

open AlgorithmLib.ML

/-- The address set one warp covers in `n / 32` strided iterations: `[0, n)`,
    written as the image of the loop so it matches `strideIx32` syntactically.

    Independent of `cta` on purpose.  These are the single-block kernels: the
    warp sweeps the whole row itself, so every hypothetical block would claim
    the same addresses.  Safety comes from the grid being 1, which is a fact
    about the launch — see `StageSpec.Exclusive`. -/
def strided (n : Nat) : Nat → Nat → Prop :=
  fun _ a => ∃ (j : Nat) (l : Lane), j < n / 32 ∧ j * 32 + l.val = a

/-- The lane that owns address `a` in a strided sweep. -/
def laneOf (a : Nat) : Lane := ⟨a % 32, Nat.mod_lt _ (by decide)⟩

theorem strided_lane {n cta a : Nat} (h : strided n cta a) :
    ∃ (j : Nat), j < n / 32 ∧ j * 32 + (laneOf a).val = a := by
  obtain ⟨j, l, hj, hl⟩ := h
  refine ⟨j, hj, ?_⟩
  have hlt : l.val < 32 := l.isLt
  show j * 32 + a % 32 = a
  omega

theorem laneOf_eq {a : Nat} (j : Nat) (l : Lane) (h : j * 32 + l.val = a) :
    l = laneOf a := by
  have hlt : l.val < 32 := l.isLt
  exact Fin.ext (by show l.val = a % 32; omega)

-- ── RMSNorm ────────────────────────────────────────────────────────────────

/-- The normalisation factor, in the lane that computes it.  Not claimed to be
    lane-uniform: the butterfly makes it so, but the stage does not need that,
    and proving it would be work for nothing — the address determines the lane,
    so the value is pinned either way. -/
def rmsScaleVal (mem : Buf → Nat → Float32) (cta : Nat) (l : Lane) : Float32 :=
  NumOps.rsqrt (NumOps.add (NumOps.mul (bflyFold (rmsLaneFold (mem 0) cta) l)
                                       (NumOps.inv dFloat)) rmsEps)

/-- What RMSNorm leaves at address `a`: `x[a] · w[a] · scale`. -/
def rmsVal (mem : Buf → Nat → Float32) (cta a : Nat) : Float32 :=
  NumOps.mul (NumOps.mul (mem 0 a) (mem 1 a)) (rmsScaleVal mem cta (laneOf a))

/-- The reduce-then-scale prefix, split out so the store loop can be read back
    against the state it actually starts from. -/
def rmsPrefix : EWStmt := .seq rmsReduce (.setR 1 rmsScale)

theorem rmsKernel_split : rmsKernelEW
    = .seq rmsPrefix (.forN (D / 32) (storeBody 2 4 strideIx32 rmsStoreCompute)) := rfl

/-- The prefix touches no memory. -/
theorem rms_prefix_mem (cta : Nat) (st : WSt) :
    ((rmsPrefix.elabIn cta).run st).mem = st.mem := by
  have hw : rmsPrefix.wbufs = [] := rfl
  funext b
  refine EWStmt.run_otherBuf b cta _ _ rmsPrefix (fun c hc => ?_) 0 st
  rw [hw] at hc
  exact absurd hc (by simp)

/-- …and leaves the scale in `%fw1`. -/
theorem rms_prefix_scale (cta : Nat) (st : WSt) (l : Lane) :
    ((rmsPrefix.elabIn cta).run st).regs 1 l = rmsScaleVal st.mem cta l := by
  have h0 : ((rmsReduce.elabIn cta).run st).regs 0 l
      = bflyFold (rmsLaneFold (st.mem 0) cta) l := rms_reduce_spec cta st l
  show (WSt.setReg ((rmsReduce.elabIn cta).run st) 1
          (fun l' => WFExp.eval ((rmsReduce.elabIn cta).run st) l' rmsScale)).regs 1 l = _
  rw [WSt.regs_setReg_same]
  show NumOps.rsqrt (NumOps.add (NumOps.mul
        (((rmsReduce.elabIn cta).run st).regs 0 l) (NumOps.inv dFloat)) rmsEps) = _
  rw [h0]
  rfl

set_option maxRecDepth 8000 in
/-- **RMSNorm as a stage.**

    Grid 1: one warp sweeps the whole row, which is why its addressing ignores
    `ctaid` and why exclusivity is the launch's job rather than the kernel's. -/
def rmsStage : StageSpec :=
  stageOfEW rmsKernelEW 2 1 (strided D) rmsVal (fun _ _ => 0) (fun _ _ => 0)
    (by decide)
    (fun cta =>
      ⟨EWStmt.storesWithin_of_notWrites 2 _ cta _ _ rmsPrefix 0 (by decide),
       fun j hj =>
         ⟨EWStmt.storesWithin_of_notWrites 2 _ cta _ _ rmsStoreCompute j (by decide),
          fun _ l => ⟨j, l, hj, rfl⟩⟩⟩)
    (by
      intro cta st a hdom
      obtain ⟨j0, hj0, ha⟩ := strided_lane hdom
      have hix : strideIx32.eval cta j0 (laneOf a) = a := by
        rw [strideIx32_eval]; exact ha
      show (((EWStmt.forN (D / 32) (storeBody 2 4 strideIx32 rmsStoreCompute)).elabIn cta).run
              ((rmsPrefix.elabIn cta).run st)).mem 2 a = _
      rw [← hix, rms_stores cta _ j0 (laneOf a) hj0]
      show NumOps.mul (NumOps.mul
              (((rmsPrefix.elabIn cta).run st).mem 0 (strideIx32.eval cta j0 (laneOf a)))
              (((rmsPrefix.elabIn cta).run st).mem 1 (strideIx32.eval cta j0 (laneOf a))))
              (((rmsPrefix.elabIn cta).run st).regs 1 (laneOf a)) = _
      rw [rms_prefix_mem cta st, rms_prefix_scale cta st (laneOf a), hix]
      rfl)
    (valOnly_of_indep rmsVal (fun m m' cta a hb => by
      have h0 : m 0 = m' 0 := hb 0 (by decide)
      have h1 : m 1 = m' 1 := hb 1 (by decide)
      show NumOps.mul (NumOps.mul (m 0 a) (m 1 a))
             (NumOps.rsqrt (NumOps.add (NumOps.mul
               (bflyFold (rmsLaneFold (m 0) cta) (laneOf a)) (NumOps.inv dFloat)) rmsEps))
         = NumOps.mul (NumOps.mul (m' 0 a) (m' 1 a))
             (NumOps.rsqrt (NumOps.add (NumOps.mul
               (bflyFold (rmsLaneFold (m' 0) cta) (laneOf a)) (NumOps.inv dFloat)) rmsEps))
      rw [h0, h1]) (strided D))

/-- Exclusivity, from the grid rather than from disjoint ownership. -/
theorem rmsStage_exclusive : rmsStage.Exclusive :=
  StageSpec.exclusive_of_grid_one rfl

-- ── SiLU gate, residual add, bias add ──────────────────────────────────────

/-! These three are `compileWKernel` output, so the library's `mapStage` and
    `mapStageIP` already build them.  Stated here so the model's stage list is
    in one place, and so the `ew` fields are checked against the *shipped*
    kernels rather than against a re-derivation. -/

/-- `out[i] = silu(g[i]) · u[i]`, out of place. -/
def siluStage (grid : Nat) : StageSpec :=
  mapStage siluGateSpec twoIn 2 grid (by decide)

theorem siluStage_ew (grid : Nat) : (siluStage grid).ew = siluKernel := rfl
theorem addStage_ew (grid : Nat) : (addStage grid).ew = addKernel := rfl

theorem siluStage_exclusive (grid : Nat) : (siluStage grid).Exclusive :=
  mapStage_exclusive _ _ _ _ _

theorem addStage_exclusive (grid : Nat) : (addStage grid).Exclusive := by
  refine StageSpec.Exclusive.ofUnbounded ?_
  intro cta cta' a h h'
  obtain ⟨l, hl⟩ := h
  obtain ⟨l', hl'⟩ := h'
  exact elemIx_blocks_disjoint cta cta' l l' (by rw [hl, hl'])

-- ── RoPE ───────────────────────────────────────────────────────────────────

/-! The in-place case, and the first stage whose addressing depends on a value
    read at run time (the token position, `im 1 1`).  Both are exactly what
    `StageSpec`'s weaker-than-idempotent `valOnly` and its `imem` field exist
    for: RoPE reads the buffer it writes, but only at the two addresses its own
    block owns. -/

/-- Block `cta` owns its head: `[cta·HEAD_DIM, cta·HEAD_DIM + HEAD_DIM)`. -/
def headDom : Nat → Nat → Prop :=
  fun cta a => cta * HEAD_DIM ≤ a ∧ a < cta * HEAD_DIM + HEAD_DIM

/-- The lane that owns `a` — the same expression in both halves, since
    `HALF = 32` and a lane index is taken mod 32. -/
def ropeLane (cta a : Nat) : Lane := ⟨(a - cta * HEAD_DIM) % 32, Nat.mod_lt _ (by decide)⟩

def ropeVal (im : Buf → Nat → Nat) (mem : Buf → Nat → Float32) (cta a : Nat) : Float32 :=
  if a < cta * HEAD_DIM + HALF
  then ropeLoVal mem cta (fun _ _ => 0) im (ropeLane cta a)
  else ropeHiVal mem cta (fun _ _ => 0) im (ropeLane cta a)

/-- **RoPE as a stage, at a given token position.**

    `im` is the launch's integer memory — the position lives at `meta[1]`.  A
    stage pinned to `im = 0` would be a model of the first token only, and every
    theorem about it would still hold. -/
def ropeStage (im : Buf → Nat → Nat) (grid : Nat) : StageSpec :=
  stageOfEW ropeEW 0 grid headDom (ropeVal im) (fun _ _ => 0) im
    (by decide)
    (fun cta =>
      ⟨EWStmt.storesWithin_of_notWrites 0 _ cta _ _ ropeBodyEW 0 (by decide),
       ⟨fun _ l => by
          have hl : l.val < 32 := l.isLt
          show cta * HEAD_DIM ≤ cta * HEAD_DIM + l.val
             ∧ cta * HEAD_DIM + l.val < cta * HEAD_DIM + HEAD_DIM
          simp only [HEAD_DIM]; omega,
        fun _ l => by
          have hl : l.val < 32 := l.isLt
          show cta * HEAD_DIM ≤ cta * HEAD_DIM + l.val + HALF
             ∧ cta * HEAD_DIM + l.val + HALF < cta * HEAD_DIM + HEAD_DIM
          simp only [HEAD_DIM, HALF]; omega⟩⟩)
    (by
      intro cta st a hdom
      obtain ⟨hlo, hhi⟩ := hdom
      have hlane : (ropeLane cta a).val = (a - cta * HEAD_DIM) % 32 := rfl
      by_cases hc : a < cta * HEAD_DIM + HALF
      · have hix : ropeLoIx.eval cta 0 (ropeLane cta a) (fun _ _ => 0) im = a := by
          rw [ropeLo_eval, hlane]
          simp only [HALF, HEAD_DIM] at hc hlo ⊢
          omega
        show ((ropeEW.elabAt cta 0 (fun _ _ => 0) im).run st).mem 0 a = _
        rw [← hix, rope_stores_lo cta (fun _ _ => 0) im st (ropeLane cta a), hix]
        show _ = ropeVal im st.mem cta a
        rw [ropeVal, if_pos hc]
      · have hix : ropeHiIx.eval cta 0 (ropeLane cta a) (fun _ _ => 0) im = a := by
          rw [ropeHi_eval, hlane]
          simp only [HALF, HEAD_DIM] at hc hhi ⊢
          omega
        show ((ropeEW.elabAt cta 0 (fun _ _ => 0) im).run st).mem 0 a = _
        rw [← hix, rope_stores_hi cta (fun _ _ => 0) im st (ropeLane cta a), hix]
        show _ = ropeVal im st.mem cta a
        rw [ropeVal, if_neg hc])
    (by
      intro m m' cta a hdom hb hout
      have h2 : m 2 = m' 2 := hb 2 (by decide)
      have hL : m 0 (ropeLoIx.eval cta 0 (ropeLane cta a) (fun _ _ => 0) im)
              = m' 0 (ropeLoIx.eval cta 0 (ropeLane cta a) (fun _ _ => 0) im) := by
        refine hout _ ?_
        have hl : (ropeLane cta a).val < 32 := (ropeLane cta a).isLt
        show cta * HEAD_DIM ≤ _ ∧ _
        rw [ropeLo_eval]; simp only [HEAD_DIM]; omega
      have hH : m 0 (ropeHiIx.eval cta 0 (ropeLane cta a) (fun _ _ => 0) im)
              = m' 0 (ropeHiIx.eval cta 0 (ropeLane cta a) (fun _ _ => 0) im) := by
        refine hout _ ?_
        have hl : (ropeLane cta a).val < 32 := (ropeLane cta a).isLt
        show cta * HEAD_DIM ≤ _ ∧ _
        rw [ropeHi_eval]; simp only [HEAD_DIM, HALF]; omega
      show (if a < cta * HEAD_DIM + HALF then _ else _) = (if _ then _ else _)
      by_cases hc : a < cta * HEAD_DIM + HALF
      · rw [if_pos hc, if_pos hc]
        show NumOps.add (NumOps.mul _ _) (NumOps.neg (NumOps.mul _ _))
           = NumOps.add (NumOps.mul _ _) (NumOps.neg (NumOps.mul _ _))
        rw [hL, hH, h2]
      · rw [if_neg hc, if_neg hc]
        show NumOps.add (NumOps.mul _ _) (NumOps.mul _ _)
           = NumOps.add (NumOps.mul _ _) (NumOps.mul _ _)
        rw [hL, hH, h2])

-- ── Binding the FFN half into one numbering ────────────────────────────────

/-! Every kernel above is a stage over the buffers *it* names — RMSNorm writes
    its slot `2`, and so does the SiLU gate.  They are not the same memory, and
    until `Bind.lean` that made a `Pipeline` of them inexpressible however many
    stages existed.

    Here is the FFN half of a layer in one numbering.  The point is not the
    particular numbers; it is that the three kernels now live in a single buffer
    space, so composing them is a `List` rather than a new proof. -/

-- ── The renaming: a recovered handle ↦ an abstract buffer ──────────────────

/-!
  **Where the buffer numbers come from.**

  They used to be chosen.  `B_X := 0` for the residual stream in the
  feed-forward half, `B_AX := 10` for the residual stream in the attention
  half — two numbers for one tensor, with nothing in the development able to
  notice, because nothing related either number to the program.  Composing the
  two halves then produced a `layerPlan` whose attention output (buffer 10) was
  not the input its feed-forward half read (buffer 0): a plan that type-checks,
  proves `run = denote`, and describes a layer that does not exist.  The same
  slip put the output projection in buffer 25 while the residual add read
  buffer 12.

  `Clif.bindsOf` recovers what the host actually stored into each launch's
  pointer array — `near k` for a handle loaded from slot `k` of the descriptor
  pointer, `far b k` for one reached through the per-layer base, whose SSA id
  `b` is carried so that a layer weight at offset `0` is not confused with the
  descriptor's own slot `0`.  So the numbering is now the image of a single
  function on those recovered handles, and a bind is derived from the array
  rather than written next to it.

  Two numbers for one tensor is no longer expressible: `bufOf` is a function.
-/

/-- **The renaming.**  The one table in this file that is chosen rather than
    derived — and it is a renaming, not a claim about memory: injectivity is
    all any theorem below asks of it.  Every entry corresponds to a handle that
    appears in some launch's recovered bind array (`bufOf_total`). -/
def bufTable : List (AlgorithmLib.Clif.BufDesc × Buf) :=
  -- activations and tables, at fixed slots of the descriptor pointer
  [ (.near 72,   1)    -- hidden state / residual stream
  , (.near 76,   2)    -- hdNorm — the normalised activation, and Wo's output
  , (.near 80,   3)    -- q
  , (.near 84,   4)    -- kCur
  , (.near 88,   5)    -- vCur
  , (.near 92,   6)    -- attnOut, reused as the FFN down-projection's output
  , (.near 96,   7)    -- ffGate
  , (.near 100,  8)    -- ffUp
  , (.near 104,  9)    -- ffAct
  , (.near 108, 10)    -- embedding table
  , (.near 116, 11)    -- logits
  , (.near 120, 12)    -- final norm weight
  , (.near 124, 13)    -- scores
  , (.near 128, 14)    -- probabilities
  , (.near 132, 15)    -- meta [token, pos, seqLen, chunks, tail, rem]
  , (.near 1592, 16)   -- RoPE sin/cos table
  -- per-layer weights, reached through the layer base
  , (.far 8 0,  17)    -- attention norm weight
  , (.far 8 8,  18)    -- q bias
  , (.far 8 16, 19)    -- k bias
  , (.far 8 24, 20)    -- v bias
  , (.far 8 32, 21)    -- feed-forward norm weight
  , (.far 8 48, 22)    -- key cache
  , (.far 8 52, 23)    -- value cache
  -- the projection matrices.  These never appear in a bind array — every one
  -- is a vendor GEMV's `A` argument — but they are buffers a `DeclaredStep`
  -- names, so they are numbered by the same function rather than by hand.
  , (.far 8 4,  24)    -- Wq
  , (.far 8 12, 25)    -- Wk
  , (.far 8 20, 26)    -- Wv
  , (.far 8 28, 27)    -- Wo
  , (.far 8 36, 28)    -- W_gate
  , (.far 8 40, 29)    -- W_up
  , (.far 8 44, 30)    -- W_down
  , (.near 112, 31) ]  -- the LM head

/-- A handle not in the table — in practice a slot the kernel does not use.
    `addKernel` and `argmaxKernel` bind two buffers, so slot `2` of their
    `bind3` is this: `0`, distinct from every real buffer, which is what keeps
    the bind injective. -/
def bufOf (d : AlgorithmLib.Clif.BufDesc) : Buf :=
  match bufTable.find? (fun p => p.1 == d) with
  | some p => p.2
  | none   => 0

/-- **A launch's bind, derived from the array the scan recovered.**

    The whole point: a table entry supplies the *recovered* descriptor list and
    the stage's renaming is computed from it, so the two cannot disagree.  A
    two-buffer kernel's third slot reads as `opaque`, hence `0`. -/
def bindOf (bs : List AlgorithmLib.Clif.BufDesc) : Buf → Buf :=
  bind3 (bufOf (bs.getD 0 .opaque)) (bufOf (bs.getD 1 .opaque))
        (bufOf (bs.getD 2 .opaque))

/-- Names for the numbers, so the plans below read as tensors rather than as
    integers.  Each is *defined* as the image of a recovered handle. -/
def B_X    : Buf := bufOf (.near 72)     -- residual stream, in place
def B_XN   : Buf := bufOf (.near 76)     -- normalised activation
def B_Q    : Buf := bufOf (.near 80)
def B_K    : Buf := bufOf (.near 84)
def B_V    : Buf := bufOf (.near 88)
def B_AO   : Buf := bufOf (.near 92)     -- attention output / FFN output
def B_GATE : Buf := bufOf (.near 96)
def B_UP   : Buf := bufOf (.near 100)
def B_ACT  : Buf := bufOf (.near 104)
def B_EMB  : Buf := bufOf (.near 108)
def B_LOG  : Buf := bufOf (.near 116)
def B_FNW  : Buf := bufOf (.near 120)
def B_SC   : Buf := bufOf (.near 124)
def B_PR   : Buf := bufOf (.near 128)
def B_META : Buf := bufOf (.near 132)
def B_TBL  : Buf := bufOf (.near 1592)
def B_ANW  : Buf := bufOf (.far 8 0)
def B_BQ   : Buf := bufOf (.far 8 8)
def B_BK   : Buf := bufOf (.far 8 16)
def B_BV   : Buf := bufOf (.far 8 24)
def B_NW   : Buf := bufOf (.far 8 32)    -- feed-forward norm weight
def B_KC   : Buf := bufOf (.far 8 48)
def B_VC   : Buf := bufOf (.far 8 52)
def B_WQ   : Buf := bufOf (.far 8 4)
def B_WK   : Buf := bufOf (.far 8 12)
def B_WV   : Buf := bufOf (.far 8 20)
def B_WO   : Buf := bufOf (.far 8 28)
def B_WG   : Buf := bufOf (.far 8 36)
def B_WU   : Buf := bufOf (.far 8 40)
def B_WD   : Buf := bufOf (.far 8 44)
def B_LMH  : Buf := bufOf (.near 112)

/-- Every buffer the plans below name. -/
def namedBufs : List Buf :=
  [ B_X, B_XN, B_Q, B_K, B_V, B_AO, B_GATE, B_UP, B_ACT, B_EMB, B_LOG
  , B_FNW, B_SC, B_PR, B_META, B_TBL, B_ANW, B_BQ, B_BK, B_BV, B_NW
  , B_KC, B_VC, B_WQ, B_WK, B_WV, B_WO, B_WG, B_WU, B_WD, B_LMH ]

/-- **Every named buffer is a different buffer.**  The renaming is injective on
    the handles that actually occur, which is what stops two tensors collapsing
    into one — the failure this section replaced. -/
theorem bufOf_injective_on_binds :
    (namedBufs).eraseDups.length = 31 := by decide

/-- …and none of them is the spare that fills an unused slot. -/
theorem bufOf_spare_unused :
    (namedBufs).all (fun b => b ≠ bufOf .opaque) = true := by decide

-- ── The feed-forward half's binds, from its recovered arrays ───────────────

/-- The three arrays `Clif.bindsOf` reads out of `inferLayerFfnFn`, in order. -/
def BS_FFN_NORM : List AlgorithmLib.Clif.BufDesc := [.near 72, .far 8 32, .near 76]
def BS_FFN_SILU : List AlgorithmLib.Clif.BufDesc := [.near 96, .near 100, .near 104]
def BS_FFN_ADD  : List AlgorithmLib.Clif.BufDesc := [.near 72, .near 92]

/-- RMSNorm's bind: it reads `x`/`w` and writes `xNorm`. -/
def bindFfnNorm : Buf → Buf := bindOf BS_FFN_NORM

/-- The gate's bind: slots `0`/`1` are the two projections, the output is the
    activation.  A different bind of the *same* kernel — which is the whole
    point of the bind being a parameter. -/
def bindFfnSilu : Buf → Buf := bindOf BS_FFN_SILU

/-- The residual add is in place: slot `0` is `x` itself, slot `1` the FFN
    output.  The kernel binds only two buffers, so slot `2` is the spare. -/
def bindFfnAdd : Buf → Buf := bindOf BS_FFN_ADD

theorem bindFfnNorm_inj : BufInj bindFfnNorm :=
  bind3_inj (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
theorem bindFfnSilu_inj : BufInj bindFfnSilu :=
  bind3_inj (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
theorem bindFfnAdd_inj : BufInj bindFfnAdd :=
  bind3_inj (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)

/-- The three FFN kernels, in the model's numbering.  `by decide` discharges the
    integer-memory obligation because none of these three reads integer memory —
    RoPE, the KV write and the embed gather would each need a real one. -/
def ffnNormStage : StageSpec :=
  rmsStage.rename bindFfnNorm bindFfnNorm_inj (fun _ _ => 0) (by intro b hb; cases hb)

def ffnSiluStage : StageSpec :=
  (siluStage (D_FF / 32)).rename bindFfnSilu bindFfnSilu_inj (fun _ _ => 0) (by intro b hb; cases hb)

def ffnAddStage : StageSpec :=
  (addStage (D / 32)).rename bindFfnAdd bindFfnAdd_inj (fun _ _ => 0) (by intro b hb; cases hb)

/-- **The three write three different buffers.**  The check the whole binding
    exercise exists for: before it, all three claimed slot `2` or slot `0` and
    the question could not even be asked. -/
theorem ffn_outputs_distinct :
    ffnNormStage.out = B_XN ∧ ffnSiluStage.out = B_ACT ∧ ffnAddStage.out = B_X := by
  refine ⟨rfl, rfl, rfl⟩

/-- Ownership survives the bind. -/
theorem ffnNormStage_exclusive : ffnNormStage.Exclusive :=
  StageSpec.rename_exclusive _ _ _ _ _ rmsStage_exclusive
theorem ffnSiluStage_exclusive : ffnSiluStage.Exclusive :=
  StageSpec.rename_exclusive _ _ _ _ _ (siluStage_exclusive _)
theorem ffnAddStage_exclusive : ffnAddStage.Exclusive :=
  StageSpec.rename_exclusive _ _ _ _ _ (addStage_exclusive _)

-- ── The feed-forward half, as a plan ───────────────────────────────────────

/-! Six steps: three proven kernels and three vendor GEMVs.  This is the first
    time in the project that a *sequence* of shipped kernels has a single
    memory-to-memory meaning — everything before was per-kernel.

    The three GEMVs are `DeclaredStep`s, not omissions: what lands in `y` is
    assumed (NVIDIA specifies no fold order, so no exact `Float32` claim is
    available), while *where* it can land is proven by the step's `frame`
    field.  `ffnPlan_declaredCount` says the number is three and
    `ffnPlan_declaredNames` says which — the gap is a value, not a caveat. -/

noncomputable def ffnPlan : Plan where
  steps :=
    [ .proven ffnNormStage                              -- xNorm = rms(x) * w
    , .declared (cublasStep B_WG B_XN B_GATE D_FF D)     -- gate = W_gate · xNorm
    , .declared (cublasStep B_WU B_XN B_UP D_FF D)       -- up   = W_up   · xNorm
    , .proven ffnSiluStage                              -- act  = silu(gate) * up
    , .declared (cublasStep B_WD B_ACT B_AO D D_FF)      -- ffn  = W_down · act
    , .proven ffnAddStage ]                             -- x   += ffn

theorem ffnPlan_declaredCount : ffnPlan.declaredCount = 3 := rfl

theorem ffnPlan_declaredNames :
    ffnPlan.declaredNames
      = ["cl_cublas_sgemv", "cl_cublas_sgemv", "cl_cublas_sgemv"] := rfl

/-- Six steps, in the order the host launches them. -/
theorem ffnPlan_length : ffnPlan.steps.length = 6 := rfl

theorem ffnPlan_exclusive : ffnPlan.Exclusive := by
  intro S hS
  rcases List.mem_cons.mp hS with h | hS
  · rw [PStep.proven.inj h]; exact ffnNormStage_exclusive
  rcases List.mem_cons.mp hS with h | hS
  · exact absurd h (by simp)
  rcases List.mem_cons.mp hS with h | hS
  · exact absurd h (by simp)
  rcases List.mem_cons.mp hS with h | hS
  · rw [PStep.proven.inj h]; exact ffnSiluStage_exclusive
  rcases List.mem_cons.mp hS with h | hS
  · exact absurd h (by simp)
  rcases List.mem_cons.mp hS with h | hS
  · rw [PStep.proven.inj h]; exact ffnAddStage_exclusive
  · exact absurd hS (by simp)

/-- **What the feed-forward half does to memory.**

    One equation for six launches.  `hR` is the single named assumption the
    whole plan carries — that the runtime's vendor GEMV honours the declared
    step — and it is discharged per primitive by the FFI contract, not here. -/
theorem ffn_computes (R : Realisation) (hR : Honours R) (st : WSt) :
    (ffnPlan.run R st).mem = ffnPlan.denote st.mem :=
  Plan.run_denote R hR ffnPlan ffnPlan_exclusive st

/-- **Non-vacuity.**  `ffn_computes` assumes `Honours R`; if no realisation
    could satisfy it the theorem would be true and empty, which is exactly the
    failure mode `A47` found in the first chaining theorem.  Here is a witness. -/
noncomputable def idealR : Realisation := fun d st => { st with mem := d.step st.mem }

theorem idealR_honours : Honours idealR := fun _ _ => rfl

theorem ffn_computes_nonvacuous (st : WSt) :
    (ffnPlan.run idealR st).mem = ffnPlan.denote st.mem :=
  ffn_computes idealR idealR_honours st

-- ── Softmax ────────────────────────────────────────────────────────────────

/-! The last per-token kernel, and the one that needed a new fact rather than
    new plumbing: its remainder pass has **all 32 lanes write one address**, so
    the kernel is correct only if all 32 agree on the row maximum and the
    reciprocal sum.  `bflyFoldOp_const` supplies that from `Law.combinerComm`
    alone — commutativity, not associativity, so it is a law that is *true* at
    `Float32` rather than one that is measurably false. -/

/-- **What the host owes the kernel.**  Four meta slots describe one sequence
    length, and softmax reads all four.  Nothing in the kernel checks they agree,
    and a stage cannot be stated without them: with `TAIL ≠ CHUNKS·32` the two
    store passes overlap and the second silently wins. -/
structure SmMeta (im : Buf → Nat → Nat) : Prop where
  tail : im 1 TAIL_SLOT = im 1 CHUNKS_SLOT * 32
  row  : im 1 CHUNKS_SLOT * 32 + im 1 REM_SLOT ≤ im 1 SEQ_SLOT
  pos  : 0 < im 1 SEQ_SLOT

/-- Block `cta` owns its row: the first `CHUNKS·32 + REM` elements of it. -/
def smDom (im : Buf → Nat → Nat) : Nat → Nat → Prop :=
  fun cta a => cta * im 1 SEQ_SLOT ≤ a
             ∧ a < cta * im 1 SEQ_SLOT + (im 1 CHUNKS_SLOT * 32 + im 1 REM_SLOT)

/-- The row maximum, read at lane `0`. -/
def smxMax (im : Buf → Nat → Nat) (mem : Buf → Nat → Float32) (cta : Nat) : Float32 :=
  smMaxFold (mem 0) cta (fun _ _ => 0) im ⟨0, by decide⟩

/-- …and the reciprocal of `Σ exp(s − max)`. -/
def smxInv (im : Buf → Nat → Nat) (mem : Buf → Nat → Float32) (cta : Nat) : Float32 :=
  NumOps.inv (smSumFold (mem 0) (fun _ => smxMax im mem cta) cta (fun _ _ => 0) im
    ⟨0, by decide⟩)

def smxVal (im : Buf → Nat → Nat) (mem : Buf → Nat → Float32) (cta a : Nat) : Float32 :=
  NumOps.mul (NumOps.exp (NumOps.add (mem 0 a) (NumOps.neg (smxMax im mem cta))))
             (smxInv im mem cta)

/-- The remainder sweep's address does not depend on the lane — which is exactly
    why every lane writes the same slot, and why the two folds below are
    lane-uniform once their butterfly seed is. -/
theorem smTailIx_lane_free (cta j : Nat) (l l' : Lane)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) :
    smTailIx.eval cta j l ir im = smTailIx.eval cta j l' ir im := rfl

/-- **The row maximum is the same in every lane.** -/
theorem smMaxFold_const (h : AllHold [Law.combinerComm]) (mem : Nat → Float32)
    (cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (l l' : Lane) :
    smMaxFold mem cta ir im l = smMaxFold mem cta ir im l' := by
  show (List.range (im 1 REM_SLOT)).foldl
        (fun a j => NumOps.max a (mem (smTailIx.eval cta j l ir im)))
        (bflyFoldOp (fun a b => NumOps.max a b) _ l)
     = (List.range (im 1 REM_SLOT)).foldl
        (fun a j => NumOps.max a (mem (smTailIx.eval cta j l' ir im)))
        (bflyFoldOp (fun a b => NumOps.max a b) _ l')
  rw [bfly_lane_uniform_max h _ l l']
  rfl

/-- **…and so is the sum**, once the maximum it subtracts is. -/
theorem smSumFold_const (h : AllHold [Law.combinerComm]) (mem : Nat → Float32)
    (mx : Lane → Float32) (hmx : ∀ p q : Lane, mx p = mx q)
    (cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (l l' : Lane) :
    smSumFold mem mx cta ir im l = smSumFold mem mx cta ir im l' := by
  show (List.range (im 1 REM_SLOT)).foldl
        (fun a j => NumOps.add a (NumOps.exp (NumOps.add
          (mem (smTailIx.eval cta j l ir im)) (NumOps.neg (mx l)))))
        (bflyFoldOp (fun a b => NumOps.add a b) _ l)
     = (List.range (im 1 REM_SLOT)).foldl
        (fun a j => NumOps.add a (NumOps.exp (NumOps.add
          (mem (smTailIx.eval cta j l' ir im)) (NumOps.neg (mx l')))))
        (bflyFoldOp (fun a b => NumOps.add a b) _ l')
  rw [hmx l l', bfly_lane_uniform_add h _ l l']
  rfl

/-- The state the two store passes start from: both reductions done, `%fw5`
    holding the row maximum and `%fw3` the reciprocal sum. -/
def smQ (cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt) : WSt :=
  ((EWStmt.setR 3 (.inv (.reg 0))).elabAt cta 0 ir im).run
    ((smSum.elabAt cta 0 ir im).run ((smMax.elabAt cta 0 ir im).run st))

theorem smMax_mem (cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt) :
    ((smMax.elabAt cta 0 ir im).run st).mem = st.mem := by
  funext b
  refine EWStmt.run_otherBuf b cta ir im smMax (fun c hc => ?_) 0 st
  rw [show smMax.wbufs = [] from rfl] at hc
  exact absurd hc (by simp)

theorem smQ_mem (cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt) :
    (smQ cta ir im st).mem = st.mem := by
  funext b
  show ((smSum.elabAt cta 0 ir im).run ((smMax.elabAt cta 0 ir im).run st)).mem b = st.mem b
  rw [EWStmt.run_otherBuf b cta ir im smSum
        (fun c hc => by rw [show smSum.wbufs = [] from rfl] at hc; exact absurd hc (by simp))
        0 _]
  exact congrFun (smMax_mem cta ir im st) b

theorem smQ_regs5 (cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (st : WSt) (l : Lane) :
    (smQ cta ir im st).regs 5 l = smMaxFold (st.mem 0) cta ir im l := by
  show (WSt.setReg ((smSum.elabAt cta 0 ir im).run ((smMax.elabAt cta 0 ir im).run st))
          3 _).regs 5 l = _
  rw [WSt.regs_setReg_other _ 3 5 _ (by decide),
      EWStmt.run_otherReg 5 cta ir im smSum (by decide) 0 _]
  exact softmax_max_spec cta 0 ir im st l

theorem smQ_regs3 (cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (st : WSt) (l : Lane) :
    (smQ cta ir im st).regs 3 l
      = NumOps.inv (smSumFold (st.mem 0)
          (((smMax.elabAt cta 0 ir im).run st).regs 5) cta ir im l) := by
  show (WSt.setReg ((smSum.elabAt cta 0 ir im).run ((smMax.elabAt cta 0 ir im).run st))
          3 (fun l' => NumOps.inv (((smSum.elabAt cta 0 ir im).run
            ((smMax.elabAt cta 0 ir im).run st)).regs 0 l'))).regs 3 l = _
  rw [WSt.regs_setReg_same,
      softmax_sum_spec cta 0 ir im ((smMax.elabAt cta 0 ir im).run st) l,
      smMax_mem cta ir im st]

/-- **Both quantities are lane-uniform** — the fact the remainder pass needs. -/
theorem smQ_regs5_const (h : AllHold [Law.combinerComm]) (cta : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt) :
    (smQ cta ir im st).regs 5 = fun _ => smxMax im st.mem cta := by
  funext l
  rw [smQ_regs5, smMaxFold_const h (st.mem 0) cta ir im l ⟨0, by decide⟩]
  rfl

theorem smQ_regs3_const (h : AllHold [Law.combinerComm]) (cta : Nat)
    (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) (st : WSt) :
    (smQ cta ir im st).regs 3 = fun _ => smxInv im st.mem cta := by
  have hmx : ((smMax.elabAt cta 0 ir im).run st).regs 5
      = fun _ => smxMax im st.mem cta := by
    funext l
    rw [softmax_max_spec cta 0 ir im st l,
        smMaxFold_const h (st.mem 0) cta ir im l ⟨0, by decide⟩]
    rfl
  funext l
  rw [smQ_regs3, hmx,
      smSumFold_const h (st.mem 0) (fun _ => smxMax im st.mem cta)
        (fun _ _ => rfl) cta ir im l ⟨0, by decide⟩]
  rfl

/-- Addresses the remainder pass writes. -/
def smTailAddr (im : Buf → Nat → Nat) (cta a : Nat) : Prop :=
  ∃ j, j < im 1 REM_SLOT ∧ a = cta * im 1 SEQ_SLOT + (im 1 TAIL_SLOT + j)

theorem smTailLoop_storesWithin (cta : Nat) (ir : Nat → Lane → Nat)
    (im : Buf → Nat → Nat) :
    (EWStmt.forM 1 REM_SLOT (normStep smTailIx)).StoresWithin 2 (smTailAddr im cta)
      cta ir im 0 :=
  fun j hj =>
    ⟨EWStmt.storesWithin_of_notWrites 2 _ cta ir im (normCompute smTailIx) j (by decide),
     fun _ l => ⟨j, hj, rfl⟩⟩

/-- Softmax's whole `StoresWithin` obligation, under the host's meta contract. -/
theorem softmax_storesWithin {im : Buf → Nat → Nat} (hm : SmMeta im) (cta : Nat)
    (ir : Nat → Lane → Nat) :
    softmaxEW.StoresWithin 2 (smDom im cta) cta ir im 0 := by
  refine ⟨EWStmt.storesWithin_of_notWrites 2 _ cta ir im (EWStmt.seq smMax smSum) 0 (by decide),
          EWStmt.storesWithin_of_notWrites 2 _ cta ir im
            (EWStmt.setR 3 (.inv (.reg 0))) 0 (by decide), ?_, ?_⟩
  · -- the chunk pass
    intro j hj
    refine ⟨EWStmt.storesWithin_of_notWrites 2 _ cta ir im (normCompute smIx) j (by decide),
            fun _ l => ?_⟩
    have hl : l.val < 32 := l.isLt
    have hjc : j + 1 ≤ im 1 CHUNKS_SLOT := hj
    have hb : (j + 1) * 32 ≤ im 1 CHUNKS_SLOT * 32 := Nat.mul_le_mul_right 32 hjc
    show cta * im 1 SEQ_SLOT ≤ cta * im 1 SEQ_SLOT + (j * 32 + l.val)
       ∧ cta * im 1 SEQ_SLOT + (j * 32 + l.val)
           < cta * im 1 SEQ_SLOT + (im 1 CHUNKS_SLOT * 32 + im 1 REM_SLOT)
    rw [Nat.succ_mul] at hb
    omega
  · -- the remainder pass
    intro j hj
    refine ⟨EWStmt.storesWithin_of_notWrites 2 _ cta ir im (normCompute smTailIx) j (by decide),
            fun _ l => ?_⟩
    have ht := hm.tail
    show cta * im 1 SEQ_SLOT ≤ cta * im 1 SEQ_SLOT + (im 1 TAIL_SLOT + j)
       ∧ cta * im 1 SEQ_SLOT + (im 1 TAIL_SLOT + j)
           < cta * im 1 SEQ_SLOT + (im 1 CHUNKS_SLOT * 32 + im 1 REM_SLOT)
    omega

/-- The chunk pass touches neither the score buffer nor the two registers the
    remainder pass still needs. -/
theorem smChunkLoop_mem0 (cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (s : WSt) :
    (((EWStmt.forM 1 CHUNKS_SLOT (normStep smIx)).elabAt cta 0 ir im).run s).mem 0
      = s.mem 0 :=
  EWStmt.run_otherBuf 0 cta ir im (EWStmt.forM 1 CHUNKS_SLOT (normStep smIx))
    (by decide) 0 s

theorem smChunkLoop_regs5 (cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (s : WSt) :
    (((EWStmt.forM 1 CHUNKS_SLOT (normStep smIx)).elabAt cta 0 ir im).run s).regs 5
      = s.regs 5 :=
  EWStmt.run_otherReg 5 cta ir im (EWStmt.forM 1 CHUNKS_SLOT (normStep smIx))
    (by decide) 0 s

theorem smChunkLoop_regs3 (cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (s : WSt) :
    (((EWStmt.forM 1 CHUNKS_SLOT (normStep smIx)).elabAt cta 0 ir im).run s).regs 3
      = s.regs 3 :=
  EWStmt.run_otherReg 3 cta ir im (EWStmt.forM 1 CHUNKS_SLOT (normStep smIx))
    (by decide) 0 s

set_option maxHeartbeats 1000000 in
/-- **Softmax lands its value at every address of the row.** -/
theorem softmax_value (h : AllHold [Law.combinerComm]) {im : Buf → Nat → Nat}
    (hm : SmMeta im) (cta : Nat) (ir : Nat → Lane → Nat) (st : WSt) (a : Nat)
    (hdom : smDom im cta a) :
    ((softmaxEW.elabAt cta 0 ir im).run st).mem 2 a = smxVal im st.mem cta a := by
  obtain ⟨hlo, hhi⟩ := hdom
  have ht := hm.tail
  -- the state both store passes start from
  have hrun : (softmaxEW.elabAt cta 0 ir im).run st
      = ((EWStmt.forM 1 REM_SLOT (normStep smTailIx)).elabAt cta 0 ir im).run
          (((EWStmt.forM 1 CHUNKS_SLOT (normStep smIx)).elabAt cta 0 ir im).run
            (smQ cta ir im st)) := rfl
  have hQm := smQ_mem cta ir im st
  have hQ5 := smQ_regs5_const h cta ir im st
  have hQ3 := smQ_regs3_const h cta ir im st
  rw [hrun]
  by_cases hc : a < cta * im 1 SEQ_SLOT + im 1 CHUNKS_SLOT * 32
  · -- a chunk address: the remainder pass does not touch it
    have hna : ¬ smTailAddr im cta a := by
      rintro ⟨j, hj, rfl⟩
      omega
    rw [EWStmt.run_otherAddr 2 (smTailAddr im cta) cta ir im a hna
          (EWStmt.forM 1 REM_SLOT (normStep smTailIx)) 0 _
          (smTailLoop_storesWithin cta ir im)]
    -- decompose the address into (loop, lane)
    have ho : a - cta * im 1 SEQ_SLOT < im 1 CHUNKS_SLOT * 32 := by omega
    have hlt : (a - cta * im 1 SEQ_SLOT) % 32 < 32 := Nat.mod_lt _ (by decide)
    have hj0 : (a - cta * im 1 SEQ_SLOT) / 32 < im 1 CHUNKS_SLOT := by omega
    have hix : smIx.eval cta ((a - cta * im 1 SEQ_SLOT) / 32)
                 ⟨(a - cta * im 1 SEQ_SLOT) % 32, hlt⟩ ir im = a := by
      show cta * im 1 SEQ_SLOT
             + ((a - cta * im 1 SEQ_SLOT) / 32 * 32 + (a - cta * im 1 SEQ_SLOT) % 32) = a
      omega
    have hs := softmax_stores_chunk cta 0 ir im (smQ cta ir im st)
      ((a - cta * im 1 SEQ_SLOT) / 32) ⟨(a - cta * im 1 SEQ_SLOT) % 32, hlt⟩ hj0
    rw [hix] at hs
    rw [hs, hQm, hQ5, hQ3]
    rfl
  · -- a remainder address: every lane wrote it, and they agreed
    have hj0 : a - (cta * im 1 SEQ_SLOT + im 1 TAIL_SLOT) < im 1 REM_SLOT := by omega
    have hix : smTailIx.eval cta (a - (cta * im 1 SEQ_SLOT + im 1 TAIL_SLOT))
                 ⟨0, by decide⟩ ir im = a := by
      show cta * im 1 SEQ_SLOT
             + (im 1 TAIL_SLOT + (a - (cta * im 1 SEQ_SLOT + im 1 TAIL_SLOT))) = a
      omega
    have h5 := smChunkLoop_regs5 cta ir im (smQ cta ir im st)
    have h3 := smChunkLoop_regs3 cta ir im (smQ cta ir im st)
    have hs := softmax_stores_tail cta 0 ir im
      (((EWStmt.forM 1 CHUNKS_SLOT (normStep smIx)).elabAt cta 0 ir im).run
        (smQ cta ir im st))
      (a - (cta * im 1 SEQ_SLOT + im 1 TAIL_SLOT)) ⟨0, by decide⟩ hj0
      (fun l => by rw [h5, hQ5]) (fun l => by rw [h3, hQ3])
    rw [hix] at hs
    rw [hs, h5, h3, hQ5, hQ3, smChunkLoop_mem0 cta ir im (smQ cta ir im st),
        congrFun hQm 0]
    rfl

/-- **Softmax as a stage**, at a given sequence length.

    Three parameters, each a real dependency rather than bookkeeping: the
    launch's integer memory `im`, the host's meta contract `hm`, and
    `Law.combinerComm` — without which the remainder pass's 32 lanes are not
    known to write the same value. -/
noncomputable def softmaxStage (h : AllHold [Law.combinerComm]) {im : Buf → Nat → Nat}
    (hm : SmMeta im) (grid : Nat) : StageSpec :=
  stageOfEW softmaxEW 2 grid (smDom im) (smxVal im) (fun _ _ => 0) im
    (by decide)
    (fun cta => softmax_storesWithin hm cta (fun _ _ => 0))
    (fun cta st a hdom => softmax_value h hm cta (fun _ _ => 0) st a hdom)
    (valOnly_of_indep (smxVal im) (fun m m' cta a hb => by
      have h0 : m 0 = m' 0 := hb 0 (by decide)
      show NumOps.mul (NumOps.exp (NumOps.add (m 0 a) (NumOps.neg (smxMax im m cta))))
             (smxInv im m cta) = _
      unfold smxInv smxMax
      rw [h0]
      rfl) (smDom im))

/-- Rows do not overlap, so softmax is exclusive outright. -/
theorem softmaxStage_exclusive (h : AllHold [Law.combinerComm]) {im : Buf → Nat → Nat}
    (hm : SmMeta im) (grid : Nat) : (softmaxStage h hm grid).Exclusive := by
  refine StageSpec.Exclusive.ofUnbounded ?_
  intro cta cta' a hd hd'
  obtain ⟨h1, h2⟩ := hd
  obtain ⟨h1', h2'⟩ := hd'
  have hr := hm.row
  have hp := hm.pos
  rcases Nat.lt_or_ge cta cta' with hlt | hge
  · exfalso
    have hstep : (cta + 1) * im 1 SEQ_SLOT ≤ cta' * im 1 SEQ_SLOT :=
      Nat.mul_le_mul_right (im 1 SEQ_SLOT) hlt
    rw [Nat.succ_mul] at hstep
    omega
  · rcases Nat.eq_or_lt_of_le hge with heq | hgt
    · exact heq.symm
    · exfalso
      have hstep : (cta' + 1) * im 1 SEQ_SLOT ≤ cta * im 1 SEQ_SLOT :=
        Nat.mul_le_mul_right (im 1 SEQ_SLOT) hgt
      rw [Nat.succ_mul] at hstep
      omega

-- ── Argmax ─────────────────────────────────────────────────────────────────

/-! One block, one warp, one output word.  Two things had to be true before it
    could be a stage.

    First, `val` may depend only on *memory*, while pass 2 reads the row maximum
    out of a register pass 1 left behind — so pass 1 needs a spec, which is
    `amMax_regs5` below, the `max` analogue of `rms_reduce_spec`.

    Second, that spec has to be *stated generically in the trip count*.  The
    earlier attempt fixed it at `VOCAB / 32 = 4748` and aborted the elaborator;
    the identical RMSNorm proof goes through at `D / 32 = 28`.  Keeping `K` a
    variable until the last step costs nothing and removes the cliff — worth
    recording, because the failure looked like a gap in the kernel and was a
    gap in how the lemma was phrased. -/

/-- Pass 1's sweep, at an arbitrary trip count. -/
def amSweepK (K : Nat) : EWStmt :=
  .seq (.setR 0 (.lit (-100000000.0)))
       (.forN K (sweepBody 0 2 0 (.maxW (.reg 0) (.reg 2)) vocabIx))

/-- The per-lane running maximum the sweep leaves in `%fw0`. -/
def amLaneMaxK (K : Nat) (mem : Nat → Float32) (cta : Nat) (l : Lane) : Float32 :=
  (List.range K).foldl
    (fun acc j => NumOps.max acc (mem (vocabIx.eval cta j l)))
    (-100000000.0 : Float32)

theorem amSweepK_regs0 (K cta : Nat) (st : WSt) :
    (((amSweepK K).elabAt cta 0 (fun _ _ => 0) (fun _ _ => 0)).run st).regs 0
      = amLaneMaxK K (st.mem 0) cta := by
  funext l
  show (((EWStmt.forN K (sweepBody 0 2 0 (.maxW (.reg 0) (.reg 2)) vocabIx)).elabAt cta 0).run
          ((WStmt.setR 0 (.lit (-100000000.0))).run st)).regs 0 l = _
  rw [sweepN_spec 0 2 0 (by decide) (.maxW (.reg 0) (.reg 2)) vocabIx
        (fun _ a x => NumOps.max a x) K cta
        [] (fun _ h => absurd h (by simp))
        ((WStmt.setR 0 (.lit (-100000000.0))).run st) (fun _ _ _ => rfl) l]
  simp only [wrun_setR, WSt.mem_setReg, WSt.regs_setReg_same, WFExp.eval]
  rfl

theorem amMax_split : amMax = .seq (.seq (amSweepK (VOCAB / 32)) warpReduceMaxE)
    (.setR 5 (.reg 0)) := rfl

/-- The row maximum, as a function of memory. -/
def amRowMax (mem : Nat → Float32) (cta : Nat) (l : Lane) : Float32 :=
  bflyFoldOp (fun a b => NumOps.max a b) (amLaneMaxK (VOCAB / 32) mem cta) l

/-- Elaboration distributes over the two `seq`s — structural, so it is cheap.
    Stated rather than left to `show`, because a `show` here asks the unifier to
    whnf a `forN` of 4748 iterations and it does not come back. -/
theorem amMax_elab (cta : Nat) :
    amMax.elabAt cta 0 (fun _ _ => 0) (fun _ _ => 0)
      = WStmt.seq (WStmt.seq
            ((amSweepK (VOCAB / 32)).elabAt cta 0 (fun _ _ => 0) (fun _ _ => 0))
            (warpReduceMaxE.elabAt cta 0 (fun _ _ => 0) (fun _ _ => 0)))
          (WStmt.setR 5 (.reg 0)) := rfl

/-- **Pass 1 leaves the row maximum in `%fw5`.** -/
theorem amMax_regs5 (cta : Nat) (st : WSt) (l : Lane) :
    ((amMax.elabAt cta 0 (fun _ _ => 0) (fun _ _ => 0)).run st).regs 5 l
      = amRowMax (st.mem 0) cta l := by
  rw [amMax_elab cta, wrun_seq, wrun_seq, wrun_setR, WSt.regs_setReg_same]
  show ((warpReduceMaxE.elabAt cta 0 (fun _ _ => 0) (fun _ _ => 0)).run
          (((amSweepK (VOCAB / 32)).elabAt cta 0 (fun _ _ => 0)
            (fun _ _ => 0)).run st)).regs 0 l = _
  rw [warpReduceMaxE_spec cta 0 _ _ _, amSweepK_regs0 (VOCAB / 32) cta st]
  rfl

/-- Pass 1 touches no memory, so pass 2 reads the entry contents. -/
theorem amMax_mem (cta : Nat) (st : WSt) :
    ((amMax.elabAt cta 0 (fun _ _ => 0) (fun _ _ => 0)).run st).mem = st.mem := by
  funext b
  refine EWStmt.run_otherBuf b cta _ _ amMax (fun c hc => ?_) 0 st
  rw [show amMax.wbufs = [] from rfl] at hc
  exact absurd hc (by simp)

/-- What argmax leaves at `out[0]`: the smallest index achieving the maximum,
    obtained by maximising a negated candidate. -/
def argmaxVal (mem : Buf → Nat → Float32) (cta : Nat) (_a : Nat) : Float32 :=
  NumOps.neg (bflyFoldOp (fun x c => NumOps.max x c)
    (fun l => (List.range (VOCAB / 32)).foldl
      (fun acc j => NumOps.max acc
        (amCand (mem 0) (amRowMax (mem 0) cta l) (vocabIx.eval cta j l)))
      (-100000000.0 : Float32))
    ⟨0, by decide⟩)

/-- **Argmax as a stage.**  Grid 1, one output address. -/
def argmaxStage : StageSpec :=
  stageOfEW argmaxEW 1 1 (fun _ a => a = 0) argmaxVal (fun _ _ => 0) (fun _ _ => 0)
    (by decide)
    (fun cta =>
      ⟨EWStmt.storesWithin_of_notWrites 1 _ cta _ _
         (EWStmt.seq amMax (.seq (.seq (.setR 0 (.lit (-100000000.0)))
            (.forN (VOCAB / 32) amBody)) warpReduceMaxE)) 0 (by decide),
       ⟨EWStmt.storesWithin_of_notWrites 1 _ cta _ _
          (EWStmt.setR 4 (.neg (.reg 0))) 0 (by decide),
        fun _ => rfl⟩⟩)
    (by
      intro cta st a hdom
      subst hdom
      rw [argmax_stores cta (fun _ _ => 0) (fun _ _ => 0) st]
      show _ = argmaxVal st.mem cta 0
      unfold argmaxVal
      refine congrArg NumOps.neg (congrArg (fun f => bflyFoldOp _ f ⟨0, by decide⟩)
        (funext fun l => ?_))
      rw [amMax_mem cta st, amMax_regs5 cta st l])
    (valOnly_of_indep argmaxVal (fun m m' cta a hb => by
      have h0 : m 0 = m' 0 := hb 0 (by decide)
      show argmaxVal m cta a = argmaxVal m' cta a
      unfold argmaxVal amRowMax
      rw [h0]) (fun _ a => a = 0))

theorem argmaxStage_exclusive : argmaxStage.Exclusive :=
  StageSpec.exclusive_of_grid_one rfl

-- ── KV-cache store ─────────────────────────────────────────────────────────

/-! A pure scatter: the *destination* is data-dependent (`pos = im 2 1`), which
    is the second reason `StageSpec` needed an `imem` field.  The address is
    injective in `(cta, loop, lane)` for a fixed position, so the element that
    lands at `a` is recoverable from `a` alone — which is what `val` needs. -/

/-- Where block `cta` writes: its head's slice of the row at `pos`. -/
def kvDom (im : Buf → Nat → Nat) : Nat → Nat → Prop :=
  fun cta a => ∃ e, e < HEAD_DIM
    ∧ a = cta * (MAX_SEQ * HEAD_DIM) + im 2 1 * HEAD_DIM + e

/-- The element index recovered from a destination address. -/
def kvElemOf (im : Buf → Nat → Nat) (cta a : Nat) : Nat :=
  a - cta * (MAX_SEQ * HEAD_DIM) - im 2 1 * HEAD_DIM

def kvVal (im : Buf → Nat → Nat) (mem : Buf → Nat → Float32) (cta a : Nat) : Float32 :=
  mem 0 (cta * HEAD_DIM + kvElemOf im cta a)

/-- **The KV write as a stage, at a given token position.** -/
def kvStoreStage (im : Buf → Nat → Nat) (grid : Nat) : StageSpec :=
  stageOfEW kvStoreEW 1 grid (kvDom im) (kvVal im) (fun _ _ => 0) im
    (by decide)
    (fun cta j hj =>
      ⟨EWStmt.storesWithin_of_notWrites 1 _ cta _ _ (EWStmt.loadIdx 0 0 kvSrcIx) j (by decide),
       fun _ l => by
         have hl : l.val < 32 := l.isLt
         have hj2 : j < 2 := hj
         exact ⟨j * 32 + l.val, by simp only [HEAD_DIM]; omega, by rw [kvDst_eval]⟩⟩)
    (by
      intro cta st a hdom
      obtain ⟨e, he, ha⟩ := hdom
      have he32 : e / 32 < HEAD_DIM / 32 := by simp only [HEAD_DIM] at he ⊢; omega
      have hlt : e % 32 < 32 := Nat.mod_lt _ (by decide)
      have hem : e / 32 * 32 + e % 32 = e := by omega
      have hix : kvDstIx.eval cta (e / 32) ⟨e % 32, hlt⟩ (fun _ _ => 0) im = a := by
        rw [kvDst_eval, ha]
        show cta * (MAX_SEQ * HEAD_DIM) + im 2 1 * HEAD_DIM + (e / 32 * 32 + e % 32)
           = cta * (MAX_SEQ * HEAD_DIM) + im 2 1 * HEAD_DIM + e
        rw [hem]
      have hw := kvStore_writes cta (fun _ _ => 0) im st (e / 32) ⟨e % 32, hlt⟩ he32
      rw [hix] at hw
      show ((kvStoreEW.elabAt cta 0 (fun _ _ => 0) im).run st).mem 1 a = _
      rw [hw]
      have hk : kvElemOf im cta a = e := by
        show a - cta * (MAX_SEQ * HEAD_DIM) - im 2 1 * HEAD_DIM = e
        rw [ha]; omega
      show st.mem 0 (cta * HEAD_DIM + (e / 32 * 32 + e % 32))
         = st.mem 0 (cta * HEAD_DIM + kvElemOf im cta a)
      rw [hem, hk])
    (by
      intro m m' cta a _ hb _
      show m 0 _ = m' 0 _
      rw [hb 0 (by decide)])

/-- KV heads sit `MAX_SEQ · HEAD_DIM` apart, far wider than the `HEAD_DIM`
    slice each writes. -/
theorem kvStoreStage_exclusive (im : Buf → Nat → Nat) (grid : Nat) :
    (kvStoreStage im grid).Exclusive := by
  refine StageSpec.Exclusive.ofUnbounded ?_
  intro cta cta' a h h'
  obtain ⟨e, he, ha⟩ := h
  obtain ⟨e', he', ha'⟩ := h'
  simp only [HEAD_DIM, MAX_SEQ] at he ha he' ha'
  omega

-- ── Embedding lookup ───────────────────────────────────────────────────────

/-- **The embedding gather as a stage, at a given token id.**

    The token id is `im 1 0`, read at run time — so like RoPE and the KV write,
    the stage is indexed by the launch's integer state rather than pretending
    it is zero. -/
def embedStage (im : Buf → Nat → Nat) (grid : Nat) : StageSpec :=
  gatherStage embedSpec embedIn embedInIx 2 grid (fun _ _ => 0) im (by decide)

theorem embedStage_ew (im : Buf → Nat → Nat) (grid : Nat) :
    (embedStage im grid).ew = embedKernelEW := rfl

theorem embedStage_exclusive (im : Buf → Nat → Nat) (grid : Nat) :
    (embedStage im grid).Exclusive := gatherStage_exclusive _ _ _ _ _ _ _ _

/-- Heads do not overlap, so RoPE is exclusive outright — no appeal to the
    grid, unlike RMSNorm. -/
theorem ropeStage_exclusive (im : Buf → Nat → Nat) (grid : Nat) :
    (ropeStage im grid).Exclusive := by
  refine StageSpec.Exclusive.ofUnbounded ?_
  intro cta cta' a h h'
  obtain ⟨h1, h2⟩ := h
  obtain ⟨h1', h2'⟩ := h'
  simp only [HEAD_DIM] at h1 h2 h1' h2'
  omega


-- ── Binding the attention half ─────────────────────────────────────────────

/-! Sixteen launches, ten of them proven kernels.  Ten kernels which all name
    buffers `0`/`1`/`2` internally end up writing ten different places, and the
    four with data-dependent addressing (RoPE twice, the KV write twice) carry
    the launch's integer memory through the same bind.

    Each stage's *local* integer memory is defined as the global one seen
    through its own bind, so `StageSpec.rename`'s `him` obligation is `rfl` and
    the meta buffer cannot drift away from the float buffers.

    The buffer numbers are the feed-forward half's — `bufOf` is one function, so
    the residual stream `B_X` that attention writes is the one the FFN reads,
    and the `hdNorm` that RMSNorm produces is the one `Wo` writes into.  Both
    identities were false while the two halves had their own numbering. -/

/-- The ten arrays `Clif.bindsOf` reads out of `inferLayerAttnFn`, in order.
    Every one of them is a value the scan recovered from the emitted stores;
    the binds below are computed from them. -/
def BS_A_NORM  : List AlgorithmLib.Clif.BufDesc := [.near 72, .far 8 0, .near 76]
def BS_A_BIASQ : List AlgorithmLib.Clif.BufDesc := [.near 80, .far 8 8]
def BS_A_BIASK : List AlgorithmLib.Clif.BufDesc := [.near 84, .far 8 16]
def BS_A_BIASV : List AlgorithmLib.Clif.BufDesc := [.near 88, .far 8 24]
def BS_A_ROPEQ : List AlgorithmLib.Clif.BufDesc := [.near 80, .near 132, .near 1592]
def BS_A_ROPEK : List AlgorithmLib.Clif.BufDesc := [.near 84, .near 132, .near 1592]
def BS_A_KVK   : List AlgorithmLib.Clif.BufDesc := [.near 84, .far 8 48, .near 132]
def BS_A_KVV   : List AlgorithmLib.Clif.BufDesc := [.near 88, .far 8 52, .near 132]
def BS_A_SOFT  : List AlgorithmLib.Clif.BufDesc := [.near 124, .near 132, .near 128]
def BS_A_ADD   : List AlgorithmLib.Clif.BufDesc := [.near 72, .near 76]

/-- The binds, each derived from its launch's recovered array.  A slot the
    kernel does not use maps to `bufOf .opaque`, distinct from every real
    buffer, which is what keeps each map injective. -/
def bAnorm : Buf → Buf := bindOf BS_A_NORM
def bBiasQ : Buf → Buf := bindOf BS_A_BIASQ
def bBiasK : Buf → Buf := bindOf BS_A_BIASK
def bBiasV : Buf → Buf := bindOf BS_A_BIASV
def bRopeQ : Buf → Buf := bindOf BS_A_ROPEQ
def bRopeK : Buf → Buf := bindOf BS_A_ROPEK
def bKvK   : Buf → Buf := bindOf BS_A_KVK
def bKvV   : Buf → Buf := bindOf BS_A_KVV
def bSoft  : Buf → Buf := bindOf BS_A_SOFT
def bAadd  : Buf → Buf := bindOf BS_A_ADD

/-- **The meta buffer really is one buffer.**

    RoPE, the KV write and softmax each read it, and under the old numbering
    each read a *different* spare (`903`, `905`, `907`) — so the position a
    RoPE launch rotated by was formally unrelated to the position the KV write
    stored at.  Deriving the bind from `near 132` in all five places makes them
    the same buffer, and this is the statement that says so. -/
theorem meta_is_shared :
    bRopeQ 1 = B_META ∧ bRopeK 1 = B_META ∧ bKvK 2 = B_META
      ∧ bKvV 2 = B_META ∧ bSoft 1 = B_META := by
  refine ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- **…and so does the residual stream, across the two halves.**  The identity
    `layerPlan` needs and did not have. -/
theorem residual_is_shared : bAnorm 0 = bindFfnNorm 0 ∧ bAadd 0 = bindFfnAdd 0 := by
  refine ⟨rfl, rfl⟩

theorem bAnorm_inj : BufInj bAnorm := bind3_inj (by decide) (by decide) (by decide)
  (by decide) (by decide) (by decide)
theorem bBiasQ_inj : BufInj bBiasQ := bind3_inj (by decide) (by decide) (by decide)
  (by decide) (by decide) (by decide)
theorem bBiasK_inj : BufInj bBiasK := bind3_inj (by decide) (by decide) (by decide)
  (by decide) (by decide) (by decide)
theorem bBiasV_inj : BufInj bBiasV := bind3_inj (by decide) (by decide) (by decide)
  (by decide) (by decide) (by decide)
theorem bRopeQ_inj : BufInj bRopeQ := bind3_inj (by decide) (by decide) (by decide)
  (by decide) (by decide) (by decide)
theorem bRopeK_inj : BufInj bRopeK := bind3_inj (by decide) (by decide) (by decide)
  (by decide) (by decide) (by decide)
theorem bKvK_inj : BufInj bKvK := bind3_inj (by decide) (by decide) (by decide)
  (by decide) (by decide) (by decide)
theorem bKvV_inj : BufInj bKvV := bind3_inj (by decide) (by decide) (by decide)
  (by decide) (by decide) (by decide)
theorem bSoft_inj : BufInj bSoft := bind3_inj (by decide) (by decide) (by decide)
  (by decide) (by decide) (by decide)
theorem bAadd_inj : BufInj bAadd := bind3_inj (by decide) (by decide) (by decide)
  (by decide) (by decide) (by decide)

variable (gim : Buf → Nat → Nat)

/-- The ten proven steps of the attention half, in the model's numbering. -/
def aNormStage : StageSpec :=
  rmsStage.rename bAnorm bAnorm_inj gim (by intro b hb; cases hb)
def aBiasQStage : StageSpec :=
  (addStage (D / 32)).rename bBiasQ bBiasQ_inj gim (by intro b hb; cases hb)
def aBiasKStage : StageSpec :=
  (addStage (KV_DIM / 32)).rename bBiasK bBiasK_inj gim (by intro b hb; cases hb)
def aBiasVStage : StageSpec :=
  (addStage (KV_DIM / 32)).rename bBiasV bBiasV_inj gim (by intro b hb; cases hb)
def aRopeQStage : StageSpec :=
  (ropeStage (fun b => gim (bRopeQ b)) 14).rename bRopeQ bRopeQ_inj gim (fun _ _ => rfl)
def aRopeKStage : StageSpec :=
  (ropeStage (fun b => gim (bRopeK b)) 2).rename bRopeK bRopeK_inj gim (fun _ _ => rfl)
def aKvKStage : StageSpec :=
  (kvStoreStage (fun b => gim (bKvK b)) 2).rename bKvK bKvK_inj gim (fun _ _ => rfl)
def aKvVStage : StageSpec :=
  (kvStoreStage (fun b => gim (bKvV b)) 2).rename bKvV bKvV_inj gim (fun _ _ => rfl)
noncomputable def aSoftStage (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) : StageSpec :=
  (softmaxStage h hm 14).rename bSoft bSoft_inj gim (fun _ _ => rfl)
def aAddStage : StageSpec :=
  (addStage (D / 32)).rename bAadd bAadd_inj gim (by intro b hb; cases hb)

/-- **Where each of the ten proven attention kernels lands.**

    Not ten *distinct* buffers, and the theorem says so: `B_Q` and `B_K` each
    appear twice, because the bias add and RoPE both update those tensors in
    place.  That is the correct answer, and it is the reason the list is stated
    rather than a distinctness claim asserted — an in-place pair sharing a
    buffer is exactly what `StageSpec`'s `valOnly` (weaker than `Idempotent`)
    was built to allow.

    What the bind machinery bought is the rest of it: `addStage` appears three
    times, `ropeStage` and `kvStoreStage` twice each, and each instance writes
    the buffer its own launch bound — a distinction that could not be *stated*
    before `StageSpec.rename`, because all of them named slot `0` or `1`. -/
theorem attn_outputs_distinct (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) :
    [ (aNormStage gim).out, (aBiasQStage gim).out, (aBiasKStage gim).out
    , (aBiasVStage gim).out, (aRopeQStage gim).out, (aRopeKStage gim).out
    , (aKvKStage gim).out, (aKvVStage gim).out, (aSoftStage gim h hm).out
    , (aAddStage gim).out ]
      = [B_XN, B_Q, B_K, B_V, B_Q, B_K, B_KC, B_VC, B_PR, B_X] := rfl

-- ── The prologue and the sampling tail ─────────────────────────────────────

/-! Two more launches sit outside the layer loop: the embedding gather that
    starts a token, and the argmax that ends it.  Both had `StageSpec`s and
    neither was in a plan, so the eleven staged kernels covered nine of the
    program's launch sites.  These close that. -/

/-- The arrays the scan reads out of `inferFn` and `inferFinalFn`. -/
def BS_EMBED  : List AlgorithmLib.Clif.BufDesc := [.near 108, .near 132, .near 72]
def BS_F_NORM : List AlgorithmLib.Clif.BufDesc := [.near 72, .near 120, .near 76]
def BS_ARGMAX : List AlgorithmLib.Clif.BufDesc := [.near 116, .near 132]

def bEmbed  : Buf → Buf := bindOf BS_EMBED
def bFnorm  : Buf → Buf := bindOf BS_F_NORM
def bArgmax : Buf → Buf := bindOf BS_ARGMAX

theorem bEmbed_inj : BufInj bEmbed := bind3_inj (by decide) (by decide) (by decide)
  (by decide) (by decide) (by decide)
theorem bFnorm_inj : BufInj bFnorm := bind3_inj (by decide) (by decide) (by decide)
  (by decide) (by decide) (by decide)
theorem bArgmax_inj : BufInj bArgmax := bind3_inj (by decide) (by decide) (by decide)
  (by decide) (by decide) (by decide)

/-- The embedding gather reads the token id out of integer memory, so its bind
    must carry the launch's integer state — the same `rfl` obligation RoPE and
    the KV write discharge. -/
def eEmbedStage : StageSpec :=
  (embedStage (fun b => gim (bEmbed b)) (D / 32)).rename bEmbed bEmbed_inj gim
    (fun _ _ => rfl)

def fNormStage : StageSpec :=
  rmsStage.rename bFnorm bFnorm_inj gim (by intro b hb; cases hb)

def fArgmaxStage : StageSpec :=
  argmaxStage.rename bArgmax bArgmax_inj gim (by intro b hb; cases hb)

/-- **What the host→device upload leaves behind.**  `cl_cuda_upload_ptr` writes
    the meta buffer from host memory; nothing in this model knows the host
    side, so it is a declared step like a vendor GEMV — recorded, framed, and
    counted rather than dropped. -/
opaque uploadedValue (dstB : Buf) (a : Nat) : Float32

noncomputable def uploadStep (dstB : Buf) : DeclaredStep where
  name  := "cl_cuda_upload_ptr"
  why   := "host→device copy; the source is host memory, which this model does \
            not describe. Framed, so it cannot touch any other buffer."
  out   := dstB
  step  := fun mem b a => if b = dstB then uploadedValue dstB a else mem b a
  frame := by
    intro mem b hb
    funext a
    simp [hb]

/-- **The prologue: upload, then gather the token's embedding.** -/
noncomputable def entryPlan : Plan where
  steps := [ .declared (uploadStep B_META), .proven (eEmbedStage gim) ]

/-- **The sampling tail: final norm, LM head, argmax.** -/
noncomputable def finalPlan : Plan where
  steps :=
    [ .proven (fNormStage gim)
    , .declared (cublasStep B_LMH B_XN B_LOG VOCAB D)
    , .proven (fArgmaxStage gim) ]

theorem entryPlan_exclusive : (entryPlan gim).Exclusive := by
  intro S hS
  simp only [entryPlan, List.mem_cons, List.not_mem_nil, or_false, reduceCtorEq,
             PStep.proven.injEq, false_or, or_false] at hS
  rcases hS with rfl
  exact StageSpec.rename_exclusive _ _ _ _ _ (embedStage_exclusive _ _)

theorem finalPlan_exclusive : (finalPlan gim).Exclusive := by
  intro S hS
  simp only [finalPlan, List.mem_cons, List.not_mem_nil, or_false, reduceCtorEq,
             PStep.proven.injEq, false_or, or_false] at hS
  rcases hS with rfl | rfl
  · exact StageSpec.rename_exclusive _ _ _ _ _ rmsStage_exclusive
  · exact StageSpec.rename_exclusive _ _ _ _ _ argmaxStage_exclusive

theorem entry_computes (R : Realisation) (hR : Honours R) (st : WSt) :
    ((entryPlan gim).run R st).mem = (entryPlan gim).denote st.mem :=
  Plan.run_denote R hR (entryPlan gim) (entryPlan_exclusive gim) st

theorem final_computes (R : Realisation) (hR : Honours R) (st : WSt) :
    ((finalPlan gim).run R st).mem = (finalPlan gim).denote st.mem :=
  Plan.run_denote R hR (finalPlan gim) (finalPlan_exclusive gim) st

/-- **The attention half, as a plan.**

    Sixteen steps: ten proven, six vendor.  The four `sgemv`s are the Q/K/V and
    output projections; the two batched `sgemm`s are the score and output
    contractions. -/
noncomputable def attnPlan (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) : Plan where
  steps :=
    [ .proven (aNormStage gim)
    , .declared (cublasStep B_WQ B_XN B_Q D D)               -- Wq · xNorm
    , .declared (cublasStep B_WK B_XN B_K KV_DIM D)          -- Wk · xNorm
    , .declared (cublasStep B_WV B_XN B_V KV_DIM D)          -- Wv · xNorm
    , .proven (aBiasQStage gim)
    , .proven (aBiasKStage gim)
    , .proven (aBiasVStage gim)
    , .proven (aRopeQStage gim)
    , .proven (aRopeKStage gim)
    , .proven (aKvKStage gim)
    , .proven (aKvVStage gim)
    , .declared (sgemmBatchedStep B_KC B_Q B_SC MAX_SEQ HEAD_DIM)
    , .proven (aSoftStage gim h hm)
    , .declared (sgemmBatchedStep B_VC B_PR B_AO D MAX_SEQ)
    , .declared (cublasStep B_WO B_AO B_XN D D)              -- Wo · attnOut → hdNorm
    , .proven (aAddStage gim) ]

theorem attnPlan_length (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) : (attnPlan gim h hm).steps.length = 16 := rfl

theorem attnPlan_declaredCount (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) : (attnPlan gim h hm).declaredCount = 6 := rfl

theorem attnPlan_declaredNames (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) :
    (attnPlan gim h hm).declaredNames
      = ["cl_cublas_sgemv", "cl_cublas_sgemv", "cl_cublas_sgemv",
         "cl_cublas_sgemm_strided_batched", "cl_cublas_sgemm_strided_batched",
         "cl_cublas_sgemv"] := rfl

theorem attnPlan_exclusive (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) : (attnPlan gim h hm).Exclusive := by
  intro S hS
  simp only [attnPlan, List.mem_cons, List.not_mem_nil, or_false, reduceCtorEq,
             PStep.proven.injEq, false_or, or_false] at hS
  rcases hS with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact StageSpec.rename_exclusive _ _ _ _ _ rmsStage_exclusive
  · exact StageSpec.rename_exclusive _ _ _ _ _ (addStage_exclusive _)
  · exact StageSpec.rename_exclusive _ _ _ _ _ (addStage_exclusive _)
  · exact StageSpec.rename_exclusive _ _ _ _ _ (addStage_exclusive _)
  · exact StageSpec.rename_exclusive _ _ _ _ _ (ropeStage_exclusive _ _)
  · exact StageSpec.rename_exclusive _ _ _ _ _ (ropeStage_exclusive _ _)
  · exact StageSpec.rename_exclusive _ _ _ _ _ (kvStoreStage_exclusive _ _)
  · exact StageSpec.rename_exclusive _ _ _ _ _ (kvStoreStage_exclusive _ _)
  · exact StageSpec.rename_exclusive _ _ _ _ _ (softmaxStage_exclusive _ _ _)
  · exact StageSpec.rename_exclusive _ _ _ _ _ (addStage_exclusive _)

/-- **What the attention half does to memory.**

    One equation for sixteen launches.  Two named assumptions appear in the
    type and nowhere else: `Honours R` for the six vendor calls, and
    `Law.combinerComm` for softmax's remainder pass. -/
theorem attn_computes (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) (R : Realisation) (hR : Honours R) (st : WSt) :
    ((attnPlan gim h hm).run R st).mem = (attnPlan gim h hm).denote st.mem :=
  Plan.run_denote R hR (attnPlan gim h hm) (attnPlan_exclusive gim h hm) st

-- ── A whole layer ──────────────────────────────────────────────────────────

/-- **One transformer layer: attention then feed-forward, twenty-two steps.**

    `Plan.denote` is a left fold, so appending the two halves is composition of
    their memory transformations — which is the sentence the whole stage,
    binding and plan apparatus exists to make expressible. -/
noncomputable def layerPlan (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) : Plan where
  steps := (attnPlan gim h hm).steps ++ ffnPlan.steps

theorem layerPlan_length (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) :
    (layerPlan gim h hm).steps.length = 22 := rfl

/-- **Nine of a layer's twenty-two steps are vendor calls; thirteen are proven
    kernels.**  The gap is a number. -/
theorem layerPlan_declaredCount (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) :
    (layerPlan gim h hm).declaredCount = 9 := rfl

theorem layerPlan_declaredNames (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) :
    (layerPlan gim h hm).declaredNames
      = ["cl_cublas_sgemv", "cl_cublas_sgemv", "cl_cublas_sgemv",
         "cl_cublas_sgemm_strided_batched", "cl_cublas_sgemm_strided_batched",
         "cl_cublas_sgemv", "cl_cublas_sgemv", "cl_cublas_sgemv",
         "cl_cublas_sgemv"] := rfl

theorem layerPlan_exclusive (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) : (layerPlan gim h hm).Exclusive := by
  intro S hS
  rcases List.mem_append.mp hS with hS | hS
  · exact attnPlan_exclusive gim h hm S hS
  · exact ffnPlan_exclusive S hS

/-- **What one transformer layer does to memory.** -/
theorem layer_computes (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) (R : Realisation) (hR : Honours R) (st : WSt) :
    ((layerPlan gim h hm).run R st).mem = (layerPlan gim h hm).denote st.mem :=
  Plan.run_denote R hR (layerPlan gim h hm) (layerPlan_exclusive gim h hm) st

/-- Non-vacuity, again: the layer theorem's hypotheses are satisfiable. -/
theorem layer_computes_nonvacuous (h : AllHold [Law.combinerComm])
    (hm : SmMeta (fun b => gim (bSoft b))) (st : WSt) :
    ((layerPlan gim h hm).run idealR st).mem = (layerPlan gim h hm).denote st.mem :=
  layer_computes gim h hm idealR idealR_honours st

end Stage

end Qwen2Proven
