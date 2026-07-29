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
theorem rope_not_stage_eligible : ropeEW.StageEligibleB 0 = false := by decide

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

/-- **Which shipped kernels could be pipeline stages, and which could not.**

    Writing this check immediately falsified a claim: the residual add is
    **in-place**.  `addKernel` writes buffer `0` and reads buffers `0` and `1`
    — `out[i] = out[i] + b[i]` — so like RoPE it cannot be a `StageSpec`, for
    the same honest reason: `valOnly` is false of it.

    Both remain correct *as kernels*: `compileWKernel` loads every input before
    it stores anything, so within one launch the in-place update is sound and
    `add_stores` proves it.  What they cannot do is participate in an
    abstraction whose whole content is "this stage's output does not depend on
    its output buffer's previous contents".

    So the shipped set is **six eligible, two not**, and the two are exactly the
    two that update in place.  That is a fact about the model, not a gap. -/
theorem qwen2_stage_eligible :
    siluKernel.StageEligibleB 2 = true
      ∧ embedKernelEW.StageEligibleB 2 = true
      ∧ rmsKernelEW.StageEligibleB 2 = true
      ∧ softmaxEW.StageEligibleB 2 = true
      ∧ argmaxEW.StageEligibleB 1 = true
      ∧ kvStoreEW.StageEligibleB 1 = true := by decide

/-- The two in-place kernels, named rather than discovered later. -/
theorem qwen2_in_place :
    addKernel.StageEligibleB 0 = false ∧ ropeEW.StageEligibleB 0 = false := by
  decide

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

end Qwen2Proven
