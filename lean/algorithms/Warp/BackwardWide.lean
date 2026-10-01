import Lean
import Std
import AlgorithmLib.Gen
import AlgorithmLib.ML
import AlgorithmLib.Surface.ProgCuda
import Scan.Layout
import Scan.Ship

open Lean AlgorithmLib AlgorithmLib.IR AlgorithmLib.ML AlgorithmLib.Host

namespace BackwardWide

-- `wf` is run on every body this file ships; the launch-sequence
-- bodies are long enough that the default recursion budget does not reach the
-- end of one.
set_option maxRecDepth 100000

/-- Qwen2-0.5B's hidden size.  `896 = 28 · 32`, so the warp sweep divides. -/
def N : Nat := 896

/-- **The launch geometry, with its coverage obligation discharged.**

    One block per output, each warp folding `trips · 32` elements.  `covers`
    is the seam guard: it is what makes "this kernel reduces all `N` elements"
    a checked fact rather than a comment.  Both the trip count and the grid
    below are *read off this record*, so there is one source, not two
    expressions that must agree. -/
def geom : ReduceGeom := { n := N, outs := N, trips := 28 }

/-- Iterations per warp — derived, not restated. -/
def K : Nat := geom.trips

/-- Blocks — likewise. -/
def GRID : Nat := geom.outs

def adjB : Buf := 0        -- upstream gradient, one per output row
def wB   : Buf := 1        -- weight matrix, row-major `W[i·N + j]`
def dxB  : Buf := 2        -- result: ∂L/∂xⱼ
def xB   : Buf := 3        -- the layer's input activations
def dwB  : Buf := 4        -- result: ∂L/∂Wᵢⱼ
def zB   : Buf := 5        -- pre-activations `z = W·x`
def dyB  : Buf := 6        -- upstream gradient at the layer's *output*

def gamB : Buf := 7        -- RMSNorm gain γ
def tB   : Buf := 8        -- scratch: `t = dy ⊙ γ`
def qB   : Buf := 9        -- scratch: `Q = Σx²`   (one float)
def sB   : Buf := 10       -- scratch: `S = Σ tᵢxᵢ` (one float)
def dxrB : Buf := 11       -- result: RMSNorm's ∂L/∂xⱼ

def yB   : Buf := 12       -- forward activations `y = silu(z)`
def ysB  : Buf := 13       -- the training target `y*`

def NBUF : Nat := 14

/-- Lane `l`, iteration `t` handles row `i = t·32 + l`. -/
def rowIx : IdxE := .add (.mul .loopI (.lit 32)) .laneId

/-- …reading `adj[i]`… -/
def adjIx : IdxE := rowIx

/-- …and `W[i·N + j]`, where `j = ctaid`.  This is the transposed walk: the
    stride between successive `i` is `N`, which is exactly why the vectorised
    schema does not apply and `dotStrided` does. -/
def wIx : IdxE := .add (.mul rowIx (.lit N)) .ctaId

/-- **The kernel is an instance of the proven strided schema** — no new kernel
    code and no new proof, only buffers, addressing and a trip count. -/
def kernel : EWStmt := dotStrided adjB wB adjIx wIx dxB .ctaId K

def ptx : String := emitProvenKernelN "main" NBUF 0 kernel

-- ── the weight gradient: an outer product ───────────────────────────────────

/-! `∂L/∂Wᵢⱼ = adjᵢ · xⱼ`.  One warp per row `i = ctaid`, walking `j` in the
    same strided pattern — so `adj[i]` is lane-uniform, `x[j]` is the strided
    read, and the destination is `stride32 (ctaid·N)`.

    This is `zipPassEW`, the two-buffer store pass, and like the dot it is a
    *schema instance*: the PTX does not grow with `N`. -/

/-- Row base of the destination: `ctaid · N`. -/
def dwBase : IdxE := .mul .ctaId (.lit N)

/-- `adj[i]` — lane- and loop-uniform, since `i = ctaid`. -/
def adjRowIx : IdxE := .ctaId

/-- `x[j]`, `j = loop·32 + lane`. -/
def xIx : IdxE := stride32 (.lit 0)

/-- `%fw1 · %fw2` — the outer product's combiner. -/
def outerF : WFExp := .mul (.reg 1) (.reg 2)

def dwKernel : EWStmt :=
  zipPassEW adjB xB dwB 1 2 0 outerF adjRowIx xIx (stride32 dwBase) K

def ptxDw : String := emitProvenKernelN "main" NBUF 0 dwKernel

-- ── the activation's backward: the derivative comes from the spec ───────────

/-! `ds = dy · silu'(z)`.  The point is that `silu'` is not written here — it is
    `sderiv` applied to the **same** `Transformer.silu` the forward kernel is
    proven against.  Differentiating the spec is what the whole stack is for,
    and at this level it costs one line.

    Two inputs at the same address, so this is `mapKernel`: zero obligations. -/

/-- `dy · ∂silu(z)/∂z`, with the derivative taken symbolically. -/
def siluBwdSpec : Expr 2 :=
  .mul (.var ⟨1, by decide⟩)
       (sderiv (Transformer.silu (.var ⟨0, by decide⟩)) ⟨0, by decide⟩)

/-- Input 0 is `z`, input 1 is `dy`; the result is the layer's adjoint. -/
def siluBwdIn : Fin 2 → Buf := fun i => if i.val = 0 then zB else dyB

def siluBwd : MapKernel 2 := mapKernel siluBwdSpec siluBwdIn adjB

def ptxSiluBwd : String := siluBwd.ptx NBUF

/-- The elementwise passes' geometry — one element per lane, checked to cover
    exactly `N`. -/
def egeom : MapGeom := MapGeom.simple N 28

/-- Blocks for the elementwise pass — derived from the geometry. -/
def EGRID : Nat := egeom.grid

-- ── RMSNorm's backward: four schema instances, no new machinery ─────────────

/-! `yᵢ = xᵢ·γᵢ·r` with `r = rsqrt(Q/n + ε)` and `Q = Σx²`, so

      ∂L/∂xⱼ = tⱼ·r − (xⱼ·r³/n)·S,   t = dy ⊙ γ,   S = Σᵢ tᵢxᵢ

  The `S` term is what makes this not elementwise: every output depends on a
  reduction over the whole row.  Four passes, each an instance of something
  already proven:

  | pass | schema |
  |---|---|
  | `t = dy ⊙ γ` | `mapKernel` |
  | `Q = Σ xᵢ·xᵢ` | `dotStrided` |
  | `S = Σ tᵢ·xᵢ` | `dotStrided` |
  | the epilogue | `mapKernelAt` — per-element `t`,`x`; **broadcast** `Q`,`S` |

  The epilogue is why `mapKernelAt` exists: injectivity is required of the
  destination, not the sources, so reading one scalar in every lane is free. -/

/-- `t = dy ⊙ γ`. -/
def tSpec : Expr 2 := .mul (.var ⟨0, by decide⟩) (.var ⟨1, by decide⟩)
def tIn : Fin 2 → Buf := fun i => if i.val = 0 then dyB else gamB
def tKernel : MapKernel 2 := mapKernel tSpec tIn tB
def ptxT : String := tKernel.ptx NBUF

/-- `Q = Σ xᵢ²` and `S = Σ tᵢxᵢ`, one warp each. -/
def qKernel : EWStmt := (sumSqKernel xB xIx qB (.lit 0) K).ew
def sKernel : EWStmt := (dotKernel tB xB xIx xIx sB (.lit 0) K).ew
def ptxQ : String := emitProvenKernelN "main" NBUF 0 qKernel
def ptxS : String := emitProvenKernelN "main" NBUF 0 sKernel

/-- The epilogue's spec: inputs are `t`, `x`, `Q`, `S` in that order.  `r` is
    bound with `letE` so the `rsqrt` is evaluated once, not four times. -/
def dxrSpec : Expr 4 :=
  letIn (.rsqrt (.add (.mul (.var ⟨2, by decide⟩) (.inv (.lit N)))
                      (.inv (.lit 1000000))))
    (fun r =>
      .add (.mul (.var ⟨0, by decide⟩) r)
           (.neg (.mul (.mul (.mul (.var ⟨1, by decide⟩) (.mul r (.mul r r)))
                             (.inv (.lit N)))
                       (.var ⟨3, by decide⟩))))

def dxrIn : Fin 4 → Buf := fun i =>
  if i.val = 0 then tB else if i.val = 1 then xB else if i.val = 2 then qB else sB

/-- `t` and `x` per element; `Q` and `S` broadcast from slot 0. -/
def dxrIx : Fin 4 → IdxE := fun i =>
  if i.val = 0 then elemIx else if i.val = 1 then elemIx else .lit 0

def dxrKernel : MapKernel 4 := mapKernelAt dxrSpec dxrIn dxrIx dxrB
def ptxDxr : String := dxrKernel.ptx NBUF

-- ── the forward half, and an optimiser: enough to actually train ────────────

/-! Everything above computes a gradient and checks it.  Training needs three
    more passes and a loop:

      z = W·x     y = silu(z)     dy = y − y*     W ← W − lr·dW

    Each is an instance of a schema that is already proven, so the training
    step adds launches rather than machinery.  The loss `½‖y − y*‖²` is not a
    kernel: it is `N` floats the host reads back, and reading it is what makes
    the demo a measurement rather than an assertion. -/

/-- The forward walk: row `ctaid` of `W`, dotted with `x`.  Rows are contiguous,
    so this is the *untransposed* walk — the same schema as the backward matvec
    at a different address expression. -/
def wFwdIx : IdxE := .add (.mul .ctaId (.lit N)) rowIx

/-- Quad addressing for the same walk: lane `l` takes four contiguous weights
    per trip, `N/128` trips. -/
def wFwdIx4 : IdxE :=
  .add (.mul .ctaId (.lit N)) (.add (.mul .loopI (.lit 128)) (.mul .laneId (.lit 4)))
def xIx4 : IdxE := .add (.mul .loopI (.lit 128)) (.mul .laneId (.lit 4))

/-- `z = W·x`, one warp per output row.

    **A tuning choice, not a rewrite.**  The backward matvec walks a column, so
    its four elements are `N` floats apart and only `dotStrided` applies.  The
    forward walks a *row*, so the quad schema does apply — same reduction, same
    committed butterfly, wider loads.  Picking it here is one constructor of
    `Sched`; `sched_agree_at_idx` is what says the choice does not change the
    answer beyond the declared law. -/
def fwdKernel : EWStmt := warpDotV4 wB xB wFwdIx4 xIx4 zB .ctaId (N / 128)
def ptxFwd : String := emitProvenKernelN "main" NBUF 0 fwdKernel

/-- `y = silu(z)` — the *same* `Transformer.silu` whose `sderiv` the backward
    kernel above uses.  One spec, differentiated for the backward pass and
    evaluated for the forward one. -/
def ySpec : Expr 1 := Transformer.silu (.var ⟨0, by decide⟩)
def yKernel : MapKernel 1 := mapKernel ySpec (fun _ => zB) yB
def ptxY : String := yKernel.ptx NBUF

/-- `dy = y − y*`, the gradient of `½‖y − y*‖²` with respect to `y`. -/
def dySpec : Expr 2 := .add (.var ⟨0, by decide⟩) (.neg (.var ⟨1, by decide⟩))
def dyIn : Fin 2 → Buf := fun i => if i.val = 0 then yB else ysB
def dyKernel : MapKernel 2 := mapKernel dySpec dyIn dyB
def ptxDy : String := dyKernel.ptx NBUF

/-- **The whole activation half of the step, in one pass.**

    `y = silu(z)`, `dy = y − y*` and `adj = dy·silu'(z)` are three elementwise
    passes over `N` floats — three launches whose combined traffic is 14 KB.  At
    that size a launch costs more than the work, so they are one kernel:

      adj = (silu(z) − y*) · silu'(z)

    Still one `mapKernel`, still one spec, and `silu'` is still `sderiv` of the
    same `Transformer.silu` evaluated in the first factor.  Fusing here costs no
    proof because the schema never cared how big the expression was. -/
def adjSpec : Expr 2 :=
  .mul (.add (Transformer.silu (.var ⟨0, by decide⟩)) (.neg (.var ⟨1, by decide⟩)))
       (sderiv (Transformer.silu (.var ⟨0, by decide⟩)) ⟨0, by decide⟩)
def adjIn : Fin 2 → Buf := fun i => if i.val = 0 then zB else ysB
def adjKernel : MapKernel 2 := mapKernel adjSpec adjIn adjB
def ptxAdj : String := adjKernel.ptx NBUF

/-- The learning rate, as an exact binary fraction so the update is a `Float32`
    identity rather than a rounding of a decimal. -/
def LR_RECIP : Nat := 1024

/-- `W ← W − lr·dW`, one element per lane over all `N²` weights.

    In place, deliberately: an optimiser step is the one kernel whose whole job
    is to overwrite its input, so it is *not* claimed `StageEligible` below —
    that guard asks a kernel not to read what it writes, which is exactly what
    an update does.  Correctness is unaffected: each lane touches one weight. -/
def sgdSpec : Expr 2 :=
  .add (.var ⟨0, by decide⟩)
       (.neg (.mul (.inv (.lit LR_RECIP)) (.var ⟨1, by decide⟩)))
def sgdIn : Fin 2 → Buf := fun i => if i.val = 0 then wB else dwB
def sgdKernel : MapKernel 2 := mapKernel sgdSpec sgdIn wB
def ptxSgd : String := sgdKernel.ptx NBUF

/-- The optimiser's geometry: one weight per lane, checked to cover `N²`. -/
def wgeom : MapGeom := MapGeom.simple (N * N) (N * N / 32)
def WGRID : Nat := wgeom.grid

-- ── memory layout ───────────────────────────────────────────────────────────

/-- Where the expected host-input size is published, so the Rust side can
    assert its packing against the Lean layout instead of duplicating it. -/
def HOST_LEN_OFF : Nat := 0x0080

def PTX_OFF    : Nat := 0x0100
def PTX_DW_OFF : Nat := 0x1400
def PTX_SB_OFF : Nat := 0x2600
def PTX_T_OFF  : Nat := 0x3E00
def PTX_Q_OFF  : Nat := 0x5000
def PTX_S_OFF  : Nat := 0x6200
def PTX_DXR_OFF: Nat := 0x7400
def PTX_FWD_OFF: Nat := 0x8C00
def PTX_Y_OFF  : Nat := 0xA400
def PTX_DY_OFF : Nat := 0xBC00
def PTX_SGD_OFF: Nat := 0xD400
def PTX_ADJ_OFF: Nat := 0xEC00
def BIND_OFF   : Nat := 0x10400
def ADJ_ID     : Nat := 0x0040
def W_ID       : Nat := 0x0044
def DX_ID      : Nat := 0x0048
def X_ID       : Nat := 0x004C
def DW_ID      : Nat := 0x0050
def Z_ID       : Nat := 0x0054
def DY_ID      : Nat := 0x0058
def GAM_ID     : Nat := 0x005C
def T_ID       : Nat := 0x0060
def Y_ID       : Nat := 0x0070
def YS_ID      : Nat := 0x0074
def Q_ID       : Nat := 0x0064
def S_ID       : Nat := 0x0068
def DXR_ID     : Nat := 0x006C
def MEM_SIZE   : Nat := 0x10500

/-- **Every byte this file names.**

    The fourteen buffer-id words are written here and read at every launch, so
    a map that lists only the PTX slots would call itself disjoint while two
    of them shared a word.  They are named one by one rather than as one
    block, because that is the collision worth catching. -/
def memMap : AlgorithmLib.Layout.RegionMap :=
      [⟨"ctx_cuda", AlgorithmLib.ContextSlots.cuda, 8⟩,
       ⟨"adj_id", ADJ_ID, 4⟩, ⟨"w_id", W_ID, 4⟩,
       ⟨"dx_id", DX_ID, 4⟩,   ⟨"x_id", X_ID, 4⟩,
       ⟨"dw_id", DW_ID, 4⟩,   ⟨"z_id", Z_ID, 4⟩,
       ⟨"dy_id", DY_ID, 4⟩,   ⟨"gam_id", GAM_ID, 4⟩,
       ⟨"t_id", T_ID, 4⟩,     ⟨"q_id", Q_ID, 4⟩,
       ⟨"s_id", S_ID, 4⟩,     ⟨"dxr_id", DXR_ID, 4⟩,
       ⟨"y_id", Y_ID, 4⟩,     ⟨"ys_id", YS_ID, 4⟩,
       ⟨"host_len", HOST_LEN_OFF, 8⟩,
       ⟨"ptx", PTX_OFF, PTX_DW_OFF - PTX_OFF⟩,
       ⟨"ptxDw", PTX_DW_OFF, PTX_SB_OFF - PTX_DW_OFF⟩,
       ⟨"ptxSiluBwd", PTX_SB_OFF, PTX_T_OFF - PTX_SB_OFF⟩,
       ⟨"ptxT", PTX_T_OFF, PTX_Q_OFF - PTX_T_OFF⟩,
       ⟨"ptxQ", PTX_Q_OFF, PTX_S_OFF - PTX_Q_OFF⟩,
       ⟨"ptxS", PTX_S_OFF, PTX_DXR_OFF - PTX_S_OFF⟩,
       ⟨"ptxDxr", PTX_DXR_OFF, PTX_FWD_OFF - PTX_DXR_OFF⟩,
       ⟨"ptxFwd", PTX_FWD_OFF, PTX_Y_OFF - PTX_FWD_OFF⟩,
       ⟨"ptxY", PTX_Y_OFF, PTX_DY_OFF - PTX_Y_OFF⟩,
       ⟨"ptxDy", PTX_DY_OFF, PTX_SGD_OFF - PTX_DY_OFF⟩,
       ⟨"ptxSgd", PTX_SGD_OFF, PTX_ADJ_OFF - PTX_SGD_OFF⟩,
       ⟨"ptxAdj", PTX_ADJ_OFF, BIND_OFF - PTX_ADJ_OFF⟩,
       ⟨"bind", BIND_OFF, 4 * NBUF⟩]


#eval LayoutScan.check "Warp.BackwardWide" [``memMap]
theorem bwdMap_ok :
    memMap.okB = true ∧ memMap.withinB MEM_SIZE = true := by decide

/-- **Seam guard: every kernel's buffers are inside the binding table.**

    `emitProvenKernelN` declares `NBUF` parameters; these are the checks that no
    kernel names a buffer beyond them.  A `decide`, not a theorem — the printer
    is trusted (see the ledger's `A46`), and this is what keeps its input
    well-formed. -/
theorem bwd_bufs_bound :
    kernel.BufBelow NBUF ∧ dwKernel.BufBelow NBUF ∧ siluBwd.ew.BufBelow NBUF
      ∧ tKernel.ew.BufBelow NBUF ∧ qKernel.BufBelow NBUF ∧ sKernel.BufBelow NBUF
      ∧ dxrKernel.ew.BufBelow NBUF := by decide

/-- **Seam guard: the launch covers exactly the intended elements.**

    Stated separately from `geom.covers` so that the numbers the *launch* uses
    are the ones checked: `GRID` blocks, `K` trips, `N` elements. -/
theorem bwd_geometry : K * 32 = N ∧ GRID = N ∧ EGRID * 32 = N := by decide

/-! **Seam guard: every branch in the emitted text resolves.**

    `flatKernel` emits absolute instruction indices as branch targets and the
    printer labels instruction `i` as `L{i}` (`programText_label`).  This checks
    the other half — that no target points past the program.  Together: every
    `bra` in the PTX names the instruction the machine model jumps to.

    That is not printer correctness, and `A46` does not claim it; it is the one
    failure this seam is actually exposed to. -/
/-! Checked one kernel at a time.  `native_decide` compiles the expression it
    decides, so a seven-way conjunction is one compilation unit carrying seven
    PTX emissions — measured at several GB, enough to exhaust the machine.
    Seven separate units prove exactly the same thing at a seventh the peak.
    The conjunction is reassembled below so nothing downstream changes. -/

theorem bwd_targets_ok_kernel : FlatTargetsOkB (flatKernel (expandEW kernel)) = true := by
  native_decide
theorem bwd_targets_ok_dw : FlatTargetsOkB (flatKernel (expandEW dwKernel)) = true := by
  native_decide
theorem bwd_targets_ok_silu : FlatTargetsOkB (flatKernel (expandEW siluBwd.ew)) = true := by
  native_decide
theorem bwd_targets_ok_dxr : FlatTargetsOkB (flatKernel (expandEW dxrKernel.ew)) = true := by
  native_decide
theorem bwd_targets_ok_t : FlatTargetsOkB (flatKernel (expandEW tKernel.ew)) = true := by
  native_decide
theorem bwd_targets_ok_q : FlatTargetsOkB (flatKernel (expandEW qKernel)) = true := by
  native_decide
theorem bwd_targets_ok_sK : FlatTargetsOkB (flatKernel (expandEW sKernel)) = true := by
  native_decide

theorem bwd_targets_ok :
    FlatTargetsOkB (flatKernel (expandEW kernel)) = true
      ∧ FlatTargetsOkB (flatKernel (expandEW dwKernel)) = true
      ∧ FlatTargetsOkB (flatKernel (expandEW siluBwd.ew)) = true
      ∧ FlatTargetsOkB (flatKernel (expandEW dxrKernel.ew)) = true
      ∧ FlatTargetsOkB (flatKernel (expandEW tKernel.ew)) = true
      ∧ FlatTargetsOkB (flatKernel (expandEW qKernel)) = true
      ∧ FlatTargetsOkB (flatKernel (expandEW sKernel)) = true :=
  ⟨bwd_targets_ok_kernel, bwd_targets_ok_dw, bwd_targets_ok_silu, bwd_targets_ok_dxr,
   bwd_targets_ok_t, bwd_targets_ok_q, bwd_targets_ok_sK⟩

/-- **Seam guard: every shipped kernel is stage-eligible.**

    A stage's value may not depend on its own output buffer's prior contents,
    so a kernel that reads what it writes cannot be one.  Checked rather than
    assumed (`A47` G8) — and the check is the reason RoPE, which rotates in
    place, is correctly outside this abstraction. -/
theorem bwd_stage_eligible :
    kernel.StageEligibleB dxB = true
      ∧ dwKernel.StageEligibleB dwB = true
      ∧ siluBwd.ew.StageEligibleB adjB = true
      ∧ tKernel.ew.StageEligibleB tB = true
      ∧ dxrKernel.ew.StageEligibleB dxrB = true
      ∧ qKernel.StageEligibleB qB = true
      ∧ sKernel.StageEligibleB sB = true := by decide

/-- **Seam guard: nothing unrenderable reaches the printer.**

    `siText` turns `.loop` and `.ext` into comments, so either one arriving
    would produce a silently wrong kernel instead of an error.  Checked, not
    assumed (`A47` G6). -/
theorem bwd_printable_kernel : FlatPrintableB (flatKernel (expandEW kernel)) = true := by
  native_decide
theorem bwd_printable_dw : FlatPrintableB (flatKernel (expandEW dwKernel)) = true := by
  native_decide
theorem bwd_printable_silu : FlatPrintableB (flatKernel (expandEW siluBwd.ew)) = true := by
  native_decide
theorem bwd_printable_dxr : FlatPrintableB (flatKernel (expandEW dxrKernel.ew)) = true := by
  native_decide
theorem bwd_printable_t : FlatPrintableB (flatKernel (expandEW tKernel.ew)) = true := by
  native_decide
theorem bwd_printable_q : FlatPrintableB (flatKernel (expandEW qKernel)) = true := by
  native_decide
theorem bwd_printable_sK : FlatPrintableB (flatKernel (expandEW sKernel)) = true := by
  native_decide

theorem bwd_printable :
    FlatPrintableB (flatKernel (expandEW kernel)) = true
      ∧ FlatPrintableB (flatKernel (expandEW dwKernel)) = true
      ∧ FlatPrintableB (flatKernel (expandEW siluBwd.ew)) = true
      ∧ FlatPrintableB (flatKernel (expandEW dxrKernel.ew)) = true
      ∧ FlatPrintableB (flatKernel (expandEW tKernel.ew)) = true
      ∧ FlatPrintableB (flatKernel (expandEW qKernel)) = true
      ∧ FlatPrintableB (flatKernel (expandEW sKernel)) = true :=
  ⟨bwd_printable_kernel, bwd_printable_dw, bwd_printable_silu, bwd_printable_dxr,
   bwd_printable_t, bwd_printable_q, bwd_printable_sK⟩

theorem bwdPtx_fits_a : ptx.toUTF8.toList.length + 1 ≤ PTX_DW_OFF - PTX_OFF := by
  native_decide
theorem bwdPtx_fits_b : ptxDw.toUTF8.toList.length + 1 ≤ PTX_SB_OFF - PTX_DW_OFF := by
  native_decide
theorem bwdPtx_fits_c : ptxSiluBwd.toUTF8.toList.length + 1 ≤ PTX_T_OFF - PTX_SB_OFF := by
  native_decide
theorem bwdPtx_fits_d : ptxT.toUTF8.toList.length + 1 ≤ PTX_Q_OFF - PTX_T_OFF := by
  native_decide
theorem bwdPtx_fits_e : ptxQ.toUTF8.toList.length + 1 ≤ PTX_S_OFF - PTX_Q_OFF := by
  native_decide
theorem bwdPtx_fits_f : ptxS.toUTF8.toList.length + 1 ≤ PTX_DXR_OFF - PTX_S_OFF := by
  native_decide
theorem bwdPtx_fits_g : ptxDxr.toUTF8.toList.length + 1 ≤ PTX_FWD_OFF - PTX_DXR_OFF := by
  native_decide

/-- The four training kernels fit their slots too. -/
theorem trainPtx_fits :
    (ptxFwd.toUTF8.toList.length + 1 ≤ PTX_Y_OFF - PTX_FWD_OFF)
      ∧ (ptxY.toUTF8.toList.length + 1 ≤ PTX_DY_OFF - PTX_Y_OFF)
      ∧ (ptxDy.toUTF8.toList.length + 1 ≤ PTX_SGD_OFF - PTX_DY_OFF)
      ∧ (ptxSgd.toUTF8.toList.length + 1 ≤ PTX_ADJ_OFF - PTX_SGD_OFF)
      ∧ (ptxAdj.toUTF8.toList.length + 1 ≤ BIND_OFF - PTX_ADJ_OFF) := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> native_decide

theorem bwdPtx_fits :
    (ptx.toUTF8.toList.length + 1 ≤ PTX_DW_OFF - PTX_OFF)
      ∧ (ptxDw.toUTF8.toList.length + 1 ≤ PTX_SB_OFF - PTX_DW_OFF)
      ∧ (ptxSiluBwd.toUTF8.toList.length + 1 ≤ PTX_T_OFF - PTX_SB_OFF)
      ∧ (ptxT.toUTF8.toList.length + 1 ≤ PTX_Q_OFF - PTX_T_OFF)
      ∧ (ptxQ.toUTF8.toList.length + 1 ≤ PTX_S_OFF - PTX_Q_OFF)
      ∧ (ptxS.toUTF8.toList.length + 1 ≤ PTX_DXR_OFF - PTX_S_OFF)
      ∧ (ptxDxr.toUTF8.toList.length + 1 ≤ PTX_FWD_OFF - PTX_DXR_OFF) :=
  ⟨bwdPtx_fits_a, bwdPtx_fits_b, bwdPtx_fits_c, bwdPtx_fits_d,
   bwdPtx_fits_e, bwdPtx_fits_f, bwdPtx_fits_g⟩

/-! ### The host input layout — one source, checked

    The host packs six arrays and `loadFn` uploads them at matching offsets.
    Previously both sides hand-wrote `k * N * 4`, with nothing relating them,
    and `zeros (A - B - len)` is Nat subtraction that saturates silently — the
    family of the worst LZ4 layout bug.

    Now the layout is one list.  `packedB` checks it is gapless, non-overlapping
    **and in the stated order** (`okB` would catch neither a gap nor a
    reordering), the uploader reads its offsets out of that same list, and the
    total is written into the memory image so the host can assert its packing
    against it at run time. -/
def hostIn : AlgorithmLib.Layout.RegionMap :=
  [⟨"adj", 0,         N * 4⟩,
   ⟨"x",   N * 4,     N * 4⟩,
   ⟨"z",   2 * N * 4, N * 4⟩,
   ⟨"dy",  3 * N * 4, N * 4⟩,
   ⟨"gam", 4 * N * 4, N * 4⟩,
   ⟨"W",   5 * N * 4, N * N * 4⟩,
   ⟨"ystar", 5 * N * 4 + N * N * 4, N * 4⟩]

/-- **Seam guard: the host layout is packed, in order, with no gaps.** -/
theorem hostIn_packed : AlgorithmLib.Layout.RegionMap.packedB 0 hostIn = true := by
  decide

/-- Bytes the host must supply — derived from the same list. -/
def HOST_BYTES : Nat := AlgorithmLib.Layout.RegionMap.total hostIn

/-- Offset of input `i`, read out of the layout rather than recomputed. -/
def hostOff (i : Nat) : Nat := AlgorithmLib.Layout.RegionMap.offAt hostIn i

/-! ### The binding table, from one list

    `EWStmt.BufBelow` checks the *kernels* name no buffer past `NBUF`.  It says
    nothing about whether the CLIF actually **writes** `NBUF` ids.  Independent
    `storeI32` calls beside an independent `NBUF` would leave those two free to
    disagree; the table is one list, the uploader indexes it, and its length is
    checked against `NBUF`. -/
def bindSlots : List String :=
  ["adj", "W", "dx", "x", "dW", "z", "dy", "gam", "t", "Q", "S", "dxr", "y", "ystar"]

/-- **Seam guard: the binding table has exactly `NBUF` entries.** -/
theorem bind_count : bindSlots.length = NBUF := by decide

/-- Byte offset of binding slot `i`. -/
def bindOff (i : Nat) : Nat := BIND_OFF + 4 * i

open AlgorithmLib.Prog

/-- **The backward kernels, as one record per PTX slot.**

    All five launches read the same fourteen-entry table at `BIND_OFF`; they
    differ in which program they run and over how many blocks.  The table is
    written once in `load` and each launch takes its arity from `params` here,
    so a stage added without a matching handle cannot elaborate. -/
def bwdK (off g : Nat) : AlgorithmLib.Kernel :=
  { name   := "main"
    params := [{ shape := [.sta N],        ro := true,  name := "adj" },
               { shape := [.sta N, .sta N], ro := true,  name := "w" },
               { shape := [.sta N],        ro := false, name := "dx" },
               { shape := [.sta N],        ro := true,  name := "x" },
               { shape := [.sta N, .sta N], ro := false, name := "dw" },
               { shape := [.sta N],        ro := true,  name := "z" },
               { shape := [.sta N],        ro := true,  name := "dy" },
               { shape := [.sta N],        ro := true,  name := "gam" },
               { shape := [.sta N],        ro := false, name := "t" },
               { shape := [.sta 1],        ro := false, name := "q" },
               { shape := [.sta 1],        ro := false, name := "s" },
               { shape := [.sta N],        ro := false, name := "dxr" },
               { shape := [.sta N],        ro := false, name := "y" },
               { shape := [.sta N],        ro := true,  name := "ys" }]
    geom   := AlgorithmLib.Kernel.Geom.static g 1 1 32 1 1
    ptxOff := off }

def loadFn : Prog V L Unit := do
  let ptr ← basePtr
  let dataPtr ← dataPtr
  cudaInit ptr
  let ctxPtr ← cudaCtxPtr ptr
  let adjBytes ← iconst64 (N * 4)
  let wBytes ← iconst64 (N * N * 4)
  let dxBytes ← iconst64 (N * 4)
  let adjId ← cudaCreateBuffer ptr adjBytes
  store adjId (← absAddr ptr ADJ_ID)
  let wId ← cudaCreateBuffer ptr wBytes
  store wId (← absAddr ptr W_ID)
  let dxId ← cudaCreateBuffer ptr dxBytes
  store dxId (← absAddr ptr DX_ID)
  let xId ← cudaCreateBuffer ptr dxBytes
  store xId (← absAddr ptr X_ID)
  let dwId ← cudaCreateBuffer ptr wBytes
  store dwId (← absAddr ptr DW_ID)
  let zId ← cudaCreateBuffer ptr dxBytes
  store zId (← absAddr ptr Z_ID)
  let dyId ← cudaCreateBuffer ptr dxBytes
  store dyId (← absAddr ptr DY_ID)
  let gamId ← cudaCreateBuffer ptr dxBytes
  store gamId (← absAddr ptr GAM_ID)
  let tId ← cudaCreateBuffer ptr dxBytes
  store tId (← absAddr ptr T_ID)
  let four ← iconst64 4
  let qId ← cudaCreateBuffer ptr four
  store qId (← absAddr ptr Q_ID)
  let sId ← cudaCreateBuffer ptr four
  store sId (← absAddr ptr S_ID)
  let dxrId ← cudaCreateBuffer ptr dxBytes
  store dxrId (← absAddr ptr DXR_ID)
  let yId ← cudaCreateBuffer ptr dxBytes
  store yId (← absAddr ptr Y_ID)
  let ysId ← cudaCreateBuffer ptr dxBytes
  store ysId (← absAddr ptr YS_ID)
  -- host buffer holds `adj`, then `x`, then `W`, contiguously
  let _ ← ffi .cudaUpload %[ctxPtr, adjId, dataPtr, adjBytes]
  let xSrc ← iaddImm dataPtr (hostOff 1)
  let _ ← ffi .cudaUpload %[ctxPtr, xId, xSrc, dxBytes]
  let zSrc ← iaddImm dataPtr (hostOff 2)
  let _ ← ffi .cudaUpload %[ctxPtr, zId, zSrc, dxBytes]
  let dySrc ← iaddImm dataPtr (hostOff 3)
  let _ ← ffi .cudaUpload %[ctxPtr, dyId, dySrc, dxBytes]
  let gamSrc ← iaddImm dataPtr (hostOff 4)
  let _ ← ffi .cudaUpload %[ctxPtr, gamId, gamSrc, dxBytes]
  let ysSrc ← iaddImm dataPtr (hostOff 6)
  let _ ← ffi .cudaUpload %[ctxPtr, ysId, ysSrc, dxBytes]
  let wSrc ← iaddImm dataPtr (hostOff 5)
  let _ ← ffi .cudaUpload %[ctxPtr, wId, wSrc, wBytes]
  kernelBindAt (bwdK PTX_OFF GRID) ptr BIND_OFF
    [adjId, wId, dxId, xId, dwId, zId, dyId, gamId, tId, qId, sId, dxrId, yId, ysId]

def runFn : Prog V L Unit := do
  let ptr ← basePtr
  kernelRelaunch (bwdK PTX_OFF GRID) ptr BIND_OFF
  let _ ← cudaSync ptr

/-- The activation backward: one element per lane, `N/32` blocks. -/
def runSiluBwdFn : Prog V L Unit := do
  let ptr ← basePtr
  kernelRelaunch (bwdK PTX_SB_OFF EGRID) ptr BIND_OFF
  let _ ← cudaSync ptr

/-- A launch of the kernel at `off` over `g` blocks of one warp. -/
def launchAt (off g : Nat) : Prog V L Unit := do
  let ptr ← basePtr
  kernelRelaunch (bwdK off g) ptr BIND_OFF
  let _ ← cudaSync ptr

def runTFn : Prog V L Unit := launchAt PTX_T_OFF EGRID
def runQFn : Prog V L Unit := launchAt PTX_Q_OFF 1
def runSFn : Prog V L Unit := launchAt PTX_S_OFF 1
def runDxrFn : Prog V L Unit := launchAt PTX_DXR_OFF EGRID

/-- The training step's four launches.  `runSgd` covers all `N²` weights. -/
def runFwdFn : Prog V L Unit := launchAt PTX_FWD_OFF GRID
def runYFn : Prog V L Unit := launchAt PTX_Y_OFF EGRID
def runDyFn : Prog V L Unit := launchAt PTX_DY_OFF EGRID
def runSgdFn : Prog V L Unit := launchAt PTX_SGD_OFF WGRID
def runAdjFn : Prog V L Unit := launchAt PTX_ADJ_OFF EGRID

/-- Fetch the forward activations, so the host can compute the loss. -/
def fetchYFn : Prog V L Unit := do
  let ptr ← basePtr
  let ctxPtr ← cudaCtxPtr ptr
  let outPtr ← outPtr
  let yId ← load32 (← absAddr ptr Y_ID)
  let dxBytes ← iconst64 (N * 4)
  let _ ← ffi .cudaDownload %[ctxPtr, yId, outPtr, dxBytes]

/-- Fetch RMSNorm's `dx`. -/
def fetchDxrFn : Prog V L Unit := do
  let ptr ← basePtr
  let ctxPtr ← cudaCtxPtr ptr
  let outPtr ← outPtr
  let dxrId ← load32 (← absAddr ptr DXR_ID)
  let dxBytes ← iconst64 (N * 4)
  let _ ← ffi .cudaDownload %[ctxPtr, dxrId, outPtr, dxBytes]

/-- **Fill the launch argument array from the buffer table.**

    `Clif.bindsOf` recovers a launch's buffer-pointer array from the stores
    preceding it, so a function that only launches recovers `bufs := none` and
    realises no stage.  Re-establishing the array here changes no handle — it is
    the same one `loadFn` wrote — and makes the run function *say* what it
    launches over.

    Before **every** launch, not once: `Clif.stepMem` maps a call to the empty
    store map, because a call could write anything. -/
def bindPass (ptr : V .i64) : Prog V L Unit := do
  for i in List.range NBUF do
    let id ← load32 (← absAddr ptr (bindOff i))
    store id (← absAddr ptr (bindOff i))

/-- **The backward pass as one function**, so the launch sequence the pipeline
    claims is a sequence *some emitted program actually performs*.

    The three stages otherwise run as three separate host calls, and a pipeline
    is a claim about an order — an order no single program exhibited. -/
def runBwdAllFn : Prog V L Unit := do
  let ptr ← basePtr
  for (off, g) in [(PTX_SB_OFF, EGRID), (PTX_OFF, GRID), (PTX_DW_OFF, GRID)] do
    bindPass ptr
    kernelRelaunch (bwdK off g) ptr BIND_OFF
  let _ ← cudaSync ptr

/-- The weight gradient, same geometry: one warp per row. -/
def runDwFn : Prog V L Unit := do
  let ptr ← basePtr
  kernelRelaunch (bwdK PTX_DW_OFF GRID) ptr BIND_OFF
  let _ ← cudaSync ptr

def fetchFn : Prog V L Unit := do
  let ptr ← basePtr
  let ctxPtr ← cudaCtxPtr ptr
  let outPtr ← outPtr
  let dxId ← load32 (← absAddr ptr DX_ID)
  let dxBytes ← iconst64 (N * 4)
  let _ ← ffi .cudaDownload %[ctxPtr, dxId, outPtr, dxBytes]

/-- Fetch the weight gradient — `N·N` floats. -/
def fetchDwFn : Prog V L Unit := do
  let ptr ← basePtr
  let ctxPtr ← cudaCtxPtr ptr
  let outPtr ← outPtr
  let dwId ← load32 (← absAddr ptr DW_ID)
  let wBytes ← iconst64 (N * N * 4)
  let _ ← ffi .cudaDownload %[ctxPtr, dwId, outPtr, wBytes]

def clifIR : Except String (List FuncData) :=
  Prog.program
    [.ok noopFunction,
     Prog.entry "main" (Prog.compileProg 1 loadFn),
     Prog.entry "run" (Prog.compileProg 2 runFn),
     Prog.entry "fetch" (Prog.compileProg 3 fetchFn),
     Prog.entry "runDw" (Prog.compileProg 4 runDwFn),
     Prog.entry "fetchDw" (Prog.compileProg 5 fetchDwFn),
     Prog.entry "runSiluBwd" (Prog.compileProg 6 runSiluBwdFn),
     Prog.entry "runT" (Prog.compileProg 7 runTFn),
     Prog.entry "runQ" (Prog.compileProg 8 runQFn),
     Prog.entry "runS" (Prog.compileProg 9 runSFn),
     Prog.entry "runDxr" (Prog.compileProg 10 runDxrFn),
     Prog.entry "fetchDxr" (Prog.compileProg 11 fetchDxrFn),
     Prog.entry "runFwd" (Prog.compileProg 12 runFwdFn),
     Prog.entry "runY" (Prog.compileProg 13 runYFn),
     Prog.entry "runDy" (Prog.compileProg 14 runDyFn),
     Prog.entry "runSgd" (Prog.compileProg 15 runSgdFn),
     Prog.entry "fetchY" (Prog.compileProg 16 fetchYFn),
     Prog.entry "runAdj" (Prog.compileProg 17 runAdjFn),
     Prog.entry "runBwdAll" (Prog.compileProg 18 runBwdAllFn)]

/-- A `Nat` as four little-endian bytes. -/
def u32le (v : Nat) : List UInt8 :=
  [UInt8.ofNat (v % 256), UInt8.ofNat (v / 256 % 256),
   UInt8.ofNat (v / 65536 % 256), UInt8.ofNat (v / 16777216 % 256)]

def initialMemory : List UInt8 :=
  let p := ptx.toUTF8.toList ++ [0]
  let q := ptxDw.toUTF8.toList ++ [0]
  let r := ptxSiluBwd.toUTF8.toList ++ [0]
  let t := ptxT.toUTF8.toList ++ [0]
  let u := ptxQ.toUTF8.toList ++ [0]
  let v := ptxS.toUTF8.toList ++ [0]
  let w := ptxDxr.toUTF8.toList ++ [0]
  let f := ptxFwd.toUTF8.toList ++ [0]
  let g := ptxY.toUTF8.toList ++ [0]
  let h := ptxDy.toUTF8.toList ++ [0]
  let k := ptxSgd.toUTF8.toList ++ [0]
  let a := ptxAdj.toUTF8.toList ++ [0]
  zeros HOST_LEN_OFF ++ u32le HOST_BYTES
    ++ zeros (PTX_OFF - HOST_LEN_OFF - 4)
    ++ p ++ zeros (PTX_DW_OFF - PTX_OFF - p.length)
    ++ q ++ zeros (PTX_SB_OFF - PTX_DW_OFF - q.length)
    ++ r ++ zeros (PTX_T_OFF - PTX_SB_OFF - r.length)
    ++ t ++ zeros (PTX_Q_OFF - PTX_T_OFF - t.length)
    ++ u ++ zeros (PTX_S_OFF - PTX_Q_OFF - u.length)
    ++ v ++ zeros (PTX_DXR_OFF - PTX_S_OFF - v.length)
    ++ w ++ zeros (PTX_FWD_OFF - PTX_DXR_OFF - w.length)
    ++ f ++ zeros (PTX_Y_OFF - PTX_FWD_OFF - f.length)
    ++ g ++ zeros (PTX_DY_OFF - PTX_Y_OFF - g.length)
    ++ h ++ zeros (PTX_SGD_OFF - PTX_DY_OFF - h.length)
    ++ k ++ zeros (PTX_ADJ_OFF - PTX_SGD_OFF - k.length)
    ++ a ++ zeros (MEM_SIZE - PTX_ADJ_OFF - a.length)

def setup (clif : List FuncData) : Artifact := {
  functions := clif,
  required_memory := MEM_SIZE
  initial_memory := initialMemory
}

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[ artifactEntry "backward_wide" (setup clif) ]

end BackwardWide

def Warp.BackwardWide.main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  let clif ← Prog.orDie BackwardWide.clifIR
  emitArtifacts outDir (BackwardWide.artifacts clif)

namespace BackwardWide

/-- **The backward matvec computes its spec** — one application of the strided
    schema theorem, at width 896.

    The right-hand side is the *committed* order: sequential within a lane, then
    a five-round butterfly across lanes.  That is the same order the forward
    GEMV is proven against (`gemv_computes_spec`), which is what makes both
    exact `Float32` equalities rather than reassociation claims.  No `SumAssoc`,
    no `ZeroTermFree`, no hypothesis at all. -/
theorem bwd_computes_spec (cta : Nat) (st : WSt) {Γ : Nat} (env : Fin Γ → Float32)
    (ae be : Nat → Expr Γ)
    (ha : ∀ i, denote env (ae i) = st.mem adjB i)
    (hb : ∀ i, denote env (be i) = st.mem wB i) :
    ((kernel.elabIn cta).run st).mem dxB cta
      = denote env (dotStridedE ae be
          (fun i l => adjIx.eval cta i l) (fun i l => wIx.eval cta i l) K) :=
  dotStrided_implements adjB wB adjIx wIx dxB .ctaId K cta st env ae be ha hb

/-- **And the emitted PTX runs it**, from instruction 0, over real branches,
    with no hypothesis.  Proven object = executed object, at model width. -/
theorem bwd_ptx_computes_spec (cta : Nat) (m : MState) {Γ : Nat} (env : Fin Γ → Float32)
    (ae be : Nat → Expr Γ)
    (ha : ∀ i, denote env (ae i) = m.mem adjB i)
    (hb : ∀ i, denote env (be i) = m.mem wB i) :
    ∃ k m', steps cta (flatKernel (expandEW kernel)) k (0, m)
          = some ((flatKernel (expandEW kernel)).length, m')
      ∧ m'.mem dxB cta
          = denote env (dotStridedE ae be
              (fun i l => adjIx.eval cta i l) (fun i l => wIx.eval cta i l) K) := by
  obtain ⟨k, m', hs, hw⟩ :=
    flatKernel_sound_idxFree cta (expandEW kernel) (expandEW_expFree kernel)
      (expandEW_idxFree kernel (by decide))
      (expandEW_flat kernel (by decide)) m
  refine ⟨k, m', hs, ?_⟩
  have hm : m'.mem dxB cta = (((expandEW kernel).elabIn cta).run m.toWSt).mem dxB cta :=
    congrArg (fun st => st.mem dxB cta) hw
  have hid : (expandEW kernel).elabIn cta = kernel.elabIn cta := rfl
  rw [hm, hid]
  exact bwd_computes_spec cta m.toWSt env ae be ha hb

/-- **The weight gradient lands where it should, and is what it should be.**

    At every `(iteration, lane)` the row visits, `dW[i·N + j] = adj[i] · x[j]`
    read from the *entry* memory.  Injectivity of the destination, the kept
    registers, and the read-back are all discharged by `zipPass_spec`; what is
    left here is a `decide` on the base address and two buffer disequalities. -/
theorem dW_stores (cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (st : WSt) (j0 : Nat) (l0 : Lane) (hj0 : j0 < K) :
    (((dwKernel.elabAt cta 0 ir im).run st).mem dwB
        ((stride32 dwBase).eval cta j0 l0 ir im))
      = NumOps.mul (st.mem adjB (adjRowIx.eval cta j0 l0 ir im))
                   (st.mem xB (xIx.eval cta j0 l0 ir im)) :=
  zipPass_spec adjB xB dwB 1 2 0 (by decide) outerF
    dwBase (by decide) adjRowIx xIx K cta (by decide) (by decide) ir im st
    (fun a b => NumOps.mul a b) (fun _ _ => rfl) j0 l0 hj0

/-- **The activation backward stores the spec's derivative.**

    `mapKernel`'s store theorem, at this instance.  The right-hand side is
    `denote` of `dy · sderiv(silu z)` — so what the GPU writes is the symbolic
    derivative of the *same* activation the forward kernel computes, with no
    hand-written backward formula anywhere in the chain. -/
theorem siluBwd_stores (st : WSt) (l : Lane) :
    ((siluBwd.ew.elabIn 0).run st).mem adjB (elemIx.eval 0 0 l)
      = denote (fun i => st.mem (siluBwdIn i) (elemIx.eval 0 0 l)) siluBwdSpec :=
  siluBwd.stores st l

/-- …and the emitted PTX runs it, from raw launch. -/
theorem siluBwd_ptx_exact (cta : Nat) (m : MState) :
    ∃ k m', steps cta (flatKernel (expandEW siluBwd.ew)) k (0, m)
          = some ((flatKernel (expandEW siluBwd.ew)).length, m')
      ∧ m'.toWSt = ((expandEW siluBwd.ew).elabIn cta).run m.toWSt :=
  mapKernel_ptx_exact siluBwdSpec siluBwdIn adjB cta m

/-! ### The training step's four kernels, each against its spec -/

/-- **The forward matvec computes its spec** — the same strided schema as the
    backward one, at the untransposed address expression.  Exact `Float32`, no
    hypothesis: forward and backward are proven against the *same* committed
    fold order, which is what lets a gradient be checked against a loss that
    actually moved. -/
theorem fwd_computes_spec (cta : Nat) (st : WSt) :
    ((fwdKernel.elabIn cta).run st).mem zB cta
      = bflyFold (dotLane (st.mem wB) (st.mem xB)
          (fun i l => wFwdIx4.eval cta i l) (fun i l => xIx4.eval cta i l) (N / 128))
          ⟨0, by decide⟩ :=
  warpDotV4_spec wB xB wFwdIx4 xIx4 zB .ctaId (N / 128) cta st

/-- `y = silu(z)`, from the same `Transformer.silu` the backward differentiates. -/
theorem y_stores (st : WSt) (l : Lane) :
    ((yKernel.ew.elabIn 0).run st).mem yB (elemIx.eval 0 0 l)
      = denote (fun _ => st.mem zB (elemIx.eval 0 0 l)) ySpec :=
  yKernel.stores st l

/-- `dy = y − y*`: the loss gradient, as one elementwise pass. -/
theorem dy_stores (st : WSt) (l : Lane) :
    ((dyKernel.ew.elabIn 0).run st).mem dyB (elemIx.eval 0 0 l)
      = denote (fun i => st.mem (dyIn i) (elemIx.eval 0 0 l)) dySpec :=
  dyKernel.stores st l

/-- **The optimiser step computes its spec** — `W ← W − lr·dW`, per weight. -/
theorem sgd_stores (st : WSt) (l : Lane) :
    ((sgdKernel.ew.elabIn 0).run st).mem wB (elemIx.eval 0 0 l)
      = denote (fun i => st.mem (sgdIn i) (elemIx.eval 0 0 l)) sgdSpec :=
  sgdKernel.stores st l

/-- **The fused activation pass stores its spec** — the three-in-one kernel is
    held to the same standard as the three it replaces. -/
theorem adj_stores (st : WSt) (l : Lane) :
    ((adjKernel.ew.elabIn 0).run st).mem adjB (elemIx.eval 0 0 l)
      = denote (fun i => st.mem (adjIn i) (elemIx.eval 0 0 l)) adjSpec :=
  adjKernel.stores st l

theorem adj_ptx_exact (cta : Nat) (m : MState) :
    ∃ k m', steps cta (flatKernel (expandEW adjKernel.ew)) k (0, m)
          = some ((flatKernel (expandEW adjKernel.ew)).length, m')
      ∧ m'.toWSt = ((expandEW adjKernel.ew).elabIn cta).run m.toWSt :=
  mapKernel_ptx_exact adjSpec adjIn adjB cta m

theorem train_ptx_exact (cta : Nat) (m : MState) :
    (∃ k m', steps cta (flatKernel (expandEW yKernel.ew)) k (0, m)
        = some ((flatKernel (expandEW yKernel.ew)).length, m')
      ∧ m'.toWSt = ((expandEW yKernel.ew).elabIn cta).run m.toWSt)
    ∧ (∃ k m', steps cta (flatKernel (expandEW dyKernel.ew)) k (0, m)
        = some ((flatKernel (expandEW dyKernel.ew)).length, m')
      ∧ m'.toWSt = ((expandEW dyKernel.ew).elabIn cta).run m.toWSt)
    ∧ (∃ k m', steps cta (flatKernel (expandEW sgdKernel.ew)) k (0, m)
        = some ((flatKernel (expandEW sgdKernel.ew)).length, m')
      ∧ m'.toWSt = ((expandEW sgdKernel.ew).elabIn cta).run m.toWSt) :=
  ⟨mapKernel_ptx_exact ySpec (fun _ => zB) yB cta m,
   mapKernel_ptx_exact dySpec dyIn dyB cta m,
   mapKernel_ptx_exact sgdSpec sgdIn wB cta m⟩

/-- **RMSNorm's epilogue stores its spec, with broadcast inputs.**

    The interesting part is `dxrIx`: `t` and `x` are read per element, `Q` and
    `S` from slot 0 in every lane.  `mapKernelAt` demands injectivity of the
    *destination* only, so the broadcast costs no obligation. -/
theorem rmsBwd_stores (st : WSt) (l : Lane) :
    ((dxrKernel.ew.elabIn 0).run st).mem dxrB (elemIx.eval 0 0 l)
      = denote (fun i => st.mem (dxrIn i) ((dxrIx i).eval 0 0 l)) dxrSpec :=
  dxrKernel.stores st l

theorem rmsBwd_ptx_exact (cta : Nat) (m : MState) :
    ∃ k m', steps cta (flatKernel (expandEW dxrKernel.ew)) k (0, m)
          = some ((flatKernel (expandEW dxrKernel.ew)).length, m')
      ∧ m'.toWSt = ((expandEW dxrKernel.ew).elabIn cta).run m.toWSt :=
  mapKernelAt_ptx_exact dxrSpec dxrIn dxrIx (by decide) dxrB cta m

/-- `S = Σᵢ tᵢ·xᵢ` computes the committed-order fold — the reduction the
    epilogue's broadcast reads. -/
theorem rmsBwd_S_spec (cta : Nat) (st : WSt) {Γ : Nat} (env : Fin Γ → Float32)
    (ae be : Nat → Expr Γ)
    (ha : ∀ i, denote env (ae i) = st.mem tB i)
    (hb : ∀ i, denote env (be i) = st.mem xB i) :
    ((sKernel.elabIn cta).run st).mem sB 0
      = denote env (dotStridedE ae be
          (fun i l => xIx.eval cta i l) (fun i l => xIx.eval cta i l) K) :=
  dotStrided_implements tB xB xIx xIx sB (.lit 0) K cta st env ae be ha hb

/-- **And the emitted PTX runs the outer-product kernel**, from raw launch. -/
theorem dW_ptx_exact (cta : Nat) (m : MState) :
    ∃ k m', steps cta (flatKernel (expandEW dwKernel)) k (0, m)
          = some ((flatKernel (expandEW dwKernel)).length, m')
      ∧ m'.toWSt = ((expandEW dwKernel).elabIn cta).run m.toWSt :=
  flatKernel_sound_idxFree cta (expandEW dwKernel) (expandEW_expFree dwKernel)
    (expandEW_idxFree dwKernel (by decide))
    (expandEW_flat dwKernel (by decide)) m

/-! ## The three kernels are pipeline stages

    Not "resemble" — **are**, definitionally.  Each `rfl` below is the check
    that `Pipeline.lean` abstracts the kernels this file actually ships rather
    than an idealised cousin of them. -/

/-- The activation backward is a map stage. -/
def sbStage : StageSpec :=
  mapStage siluBwdSpec siluBwdIn adjB EGRID (by decide)

/-- The transposed matvec is a reduction stage. -/
def dxStage : StageSpec :=
  reduceStage adjB wB adjIx wIx dxB K GRID (by decide) (by decide)

/-- The weight gradient is an outer-product stage. -/
def dwStage : StageSpec :=
  outerStage adjB xB dwB N K GRID (by decide) (by decide) (by decide)

-- ---------------------------------------------------------------------------
-- The backward pass as a pipeline value
-- ---------------------------------------------------------------------------

/-- **The launch sequence, as data.**  `bwd_chain` below states what running
    `sbStage` then `dxStage` computes, but it does so by writing the composite
    into its own conclusion — so it describes one pipeline and cannot be reused.
    This is the same sequence as a value, which `Pipeline.run_denote` covers
    generically at any length. -/
def bwdPipeline : Pipeline := ⟨[sbStage, dxStage]⟩

theorem bwdPipeline_exclusive : bwdPipeline.Exclusive := by
  intro S hS
  rcases List.mem_cons.mp hS with h | h
  · subst h; exact mapStage_exclusive _ _ _ _ _
  · rcases List.mem_cons.mp h with h' | h'
    · subst h'; exact reduceStage_exclusive _ _ _ _ _ _ _ _ _
    · exact absurd h' (by simp)

/-- **The two-stage backward pass computes its denotation.**  No composite is
    written here: `Pipeline.denote` derives it from the stage list. -/
theorem bwdPipeline_runs (st : WSt) :
    (bwdPipeline.run st).mem = bwdPipeline.denote st.mem :=
  bwdPipeline.run_denote bwdPipeline_exclusive st

/-- **Adding the weight-gradient stage needs no new proof.**  The three-stage
    pipeline is covered by the same theorem — which is the point of reifying
    the sequence rather than proving one composition at a time. -/
def bwdPipelineFull : Pipeline := ⟨[sbStage, dxStage, dwStage]⟩

theorem bwdPipelineFull_exclusive : bwdPipelineFull.Exclusive := by
  intro S hS
  rcases List.mem_cons.mp hS with h | h
  · subst h; exact mapStage_exclusive _ _ _ _ _
  · rcases List.mem_cons.mp h with h' | h'
    · subst h'; exact reduceStage_exclusive _ _ _ _ _ _ _ _ _
    · rcases List.mem_cons.mp h' with h'' | h''
      · subst h''; exact outerStage_exclusive _ _ _ _ _ _ (by decide) _ _
      · exact absurd h'' (by simp)

-- ---------------------------------------------------------------------------
-- The host program that launches them
-- ---------------------------------------------------------------------------

-- ---------------------------------------------------------------------------
-- The shipped function launches exactly this pipeline
-- ---------------------------------------------------------------------------

/-! `bwdDriver` below is a *model* of a host program: nothing emits it.
    `runBwdAllFn` is the shipped function, and it is what carries the three
    stages in one program, so it is the one a `Pipeline` claim can be about.

    What follows recovers the launches from `runBwdAllFn`'s own instruction
    stream and matches them against `bwdPipelineFull`. -/

def ROOT : Nat := 0

/-- The bind array as `Clif.bindsOf` recovers it — entry `i` is the handle at
    layout slot `bindOff i`, which is buffer `i`. -/
def bwdBufs : List AlgorithmLib.Clif.BufDesc :=
  (List.range NBUF).map (fun i => .near (Int.ofNat (bindOff i)))

def bwdRec (off g : Nat) : AlgorithmLib.Clif.LaunchRec :=
  { fnName    := "cl_cuda_launch"
    kernelOff := some (Int.ofNat off)
    nBufs     := some (Int.ofNat NBUF)
    bindOff   := some (Int.ofNat BIND_OFF)
    gridX     := some (Int.ofNat g)
    blockX    := some 32 }

/-- The three launches, in pipeline order. -/
def bwdAllOps : List DeviceOp :=
  [(bwdRec PTX_SB_OFF EGRID, { bufs := some bwdBufs }),
   (bwdRec PTX_OFF GRID,     { bufs := some bwdBufs }),
   (bwdRec PTX_DW_OFF GRID,  { bufs := some bwdBufs })]

/-- **Which slot holds which stage**, for the shipped function.  One shared
    pointer array, so the slot is what tells the three apart. -/
def bwdAllTable : List KernelBinding :=
  [ ⟨PTX_SB_OFF, BIND_OFF, bwdBufs, sbStage⟩
  , ⟨PTX_OFF,    BIND_OFF, bwdBufs, dxStage⟩
  , ⟨PTX_DW_OFF, BIND_OFF, bwdBufs, dwStage⟩ ]

/-- **Seam guard: the emitted CLIF performs these launches**, in this order,
    over these buffers, at these grids.

    Stated over the compiled body rather than the term it was written as, so
    what the generator is written in does not enter the claim --- only what
    was emitted. -/
theorem bwdAll_ops_are :
    AlgorithmLib.Clif.deviceOpsOf ROOT (Prog.stateOf 18 runBwdAllFn) = bwdAllOps := by
  native_decide

/-- …and those launches are the proven three-stage pipeline. -/
theorem bwdAll_realises :
    pipelineOf? bwdAllTable none bwdAllOps = some bwdPipelineFull := rfl

/-- **The shipped backward pass computes its denotation.**

    Unlike `bwd_host_computes` below, the launch sequence here is read out of a
    function that is actually built into the artifact. -/
theorem bwdAll_host_computes (st : WSt) :
    pipelineOf? bwdAllTable none (AlgorithmLib.Clif.deviceOpsOf ROOT (Prog.stateOf 18 runBwdAllFn))
        = some bwdPipelineFull
      ∧ (bwdPipelineFull.run st).mem = bwdPipelineFull.denote st.mem :=
  ⟨by rw [bwdAll_ops_are]; exact bwdAll_realises,
   bwdPipelineFull.run_denote bwdPipelineFull_exclusive st⟩

/-- **Which PTX slot holds which stage.**  The single place the host-to-kernel
    correspondence is asserted; everything below is derived from it. -/
def bwdTable : List KernelBinding :=
  [ ⟨0,  100, [.near 0x40, .near 0x48],              sbStage⟩
  , ⟨8,  108, [.near 0x48, .near 0x50, .near 0x58], dxStage⟩
  , ⟨16, 116, [.near 0x48, .near 0x60, .near 0x68], dwStage⟩ ]

/-- The slots each launch stores into its pointer array — the `ExternArg` side
    of the same three entries.  `HStmt.binds` maps these through
    `ExternArg.toBuf`, so a disagreement with `bwdTable` above makes the
    realisation theorems fail rather than quietly match. -/
def sbBinds : List ExternArg := [.slot 0x40, .slot 0x48]
def dxBinds : List ExternArg := [.slot 0x48, .slot 0x50, .slot 0x58]
def dwBinds : List ExternArg := [.slot 0x48, .slot 0x60, .slot 0x68]

/-- **The driver, as a host program.**  Three launches in pipeline order, with
    the geometry each stage declares — not a grid number written twice. -/
def bwdDriver : HStmt :=
  .seq (.launch ⟨0, 2, 100, EGRID, 32, sbBinds⟩)
    (.seq (.launch ⟨8, 3, 108, GRID, 32, dxBinds⟩)
          (.launch ⟨16, 3, 116, GRID, 32, dwBinds⟩))

/-- The emitted host code — a real instruction list, compiled by `flatHI`. -/
def bwdCode : List HI := code (.ffi .cudaLaunch) ⟨0⟩ 1 0 bwdDriver

/-- **The driver's launch sequence is the backward pipeline.**  Decided, not
    assumed: a drifted slot or a grid disagreeing with the stage's own `grid`
    field makes this `none` and the proof fails. -/
theorem bwdDriver_realises :
    pipelineOf? bwdTable none bwdDriver.deviceOps = some bwdPipelineFull := rfl

/-- **The whole host side, end to end, on the kernels this file ships.**

    Executing the emitted CLIF instruction by instruction performs three
    launches; under `bwdTable` those launches *are* `bwdPipelineFull`; and that
    pipeline computes its denotation.  Every hypothesis is discharged at a
    concrete value, so nothing here is vacuous.

    The remaining trusted step is `bwdTable` itself — that the PTX at slot `0`
    is `sbStage.ew` compiled, and so on.  `sb_ptx_exact`, `dx_ptx_exact` and
    `dW_ptx_exact` above are what make each of those three claims checkable. -/
theorem bwd_host_computes (st : WSt) :
    ∃ k c', hsteps 0 bwdCode k
              ⟨0, AlgorithmLib.Clif.Env.empty, [], fun _ => 0, [], []⟩ = some c'
      ∧ pipelineOf? bwdTable none (c'.trace.zip c'.btrace) = some bwdPipelineFull
      ∧ (bwdPipelineFull.run st).mem = bwdPipelineFull.denote st.mem :=
  host_computes_denote (.ffi .cudaLaunch) ⟨0⟩ rfl bwdCode bwdDriver 1 0
    AlgorithmLib.Clif.Env.empty [] (fun _ => 0) (by decide) rfl (by decide)
    (AlgorithmLib.Host.FarOk.of_noBases (by decide))
    (fun x hx b hb => AlgorithmLib.Host.noBases_primDests (by decide) x hx b hb)
    (fun j _ => by rw [Nat.zero_add]; rfl)
    bwdTable bwdPipelineFull none bwdDriver_realises bwdPipelineFull_exclusive st

-- ---------------------------------------------------------------------------
-- The same driver with a vendor call in it
-- ---------------------------------------------------------------------------

/-!
  The shipped inference model does not consist only of kernels this development
  compiled: about 99.9% of Qwen2's arithmetic goes through `cl_cublas_sgemv`,
  whose fold order NVIDIA does not specify.  What follows is the same driver
  with such a call in the middle of it — a `DeclaredStep`, sitting *in* the
  sequence rather than excluded from it, so the composition covers every device
  write and the number of assumed steps is a value.
-/

def bwdBlasRef : Callee := .ffi .cublasSgemv

/-- Buffer-handle slots, as a generator would lay them out. -/
def SLOT_W : Nat := 0x100
def SLOT_X : Nat := 0x108
def SLOT_Z : Nat := 0x110
def SLOT_Q : Nat := 0x118

/-- `cl_cublas_sgemv(ctx, trans, m, n, alpha, A, x, beta, y)` — the real
    signature.  The scalars are immediates; the three buffers are handles loaded
    from their slots, which is what makes this call site distinguishable from
    another of the same shape. -/
def sgemvArgs (aSlot xSlot ySlot : Nat) : List ExternArg :=
  [ .const 0, .const (Int.ofNat N), .const (Int.ofNat K), .const 1
  , .slot aSlot, .slot xSlot, .const 0, .slot ySlot ]

def blasCall (aSlot xSlot ySlot : Nat) : HStmt :=
  .extern { name := "cl_cublas_sgemv", fn := bwdBlasRef
            argv := sgemvArgs aSlot xSlot ySlot }

/-- Two vendor calls of **identical shape**, differing only in which buffers
    they touch — the situation Qwen2 is in with `Wq`/`Wk`/`Wv`. -/
noncomputable def zStep : DeclaredStep := cublasStep wB xB zB N K
noncomputable def qStep : DeclaredStep := cublasStep wB xB qB N K

noncomputable def bwdDeclared : List DeclaredBinding :=
  [ { name := "cl_cublas_sgemv", args := (sgemvArgs SLOT_W SLOT_X SLOT_Z).map ExternArg.toBuf
      decl := zStep }
  , { name := "cl_cublas_sgemv", args := (sgemvArgs SLOT_W SLOT_X SLOT_Q).map ExternArg.toBuf
      decl := qStep } ]

/-- Launch, two same-shape vendor calls, launch, launch. -/
def bwdDriverBlas : HStmt :=
  .seq (.launch ⟨0, 2, 100, EGRID, 32, sbBinds⟩)
    (.seq (blasCall SLOT_W SLOT_X SLOT_Z)
      (.seq (blasCall SLOT_W SLOT_X SLOT_Q)
        (.seq (.launch ⟨8, 3, 108, GRID, 32, dxBinds⟩)
              (.launch ⟨16, 3, 116, GRID, 32, dwBinds⟩))))

def bwdCodeBlas : List HI := code (.ffi .cudaLaunch) ⟨0⟩ 1 0 bwdDriverBlas

/-- **The plan: three proven steps and two declared ones, in host order.**

    The two declared steps are *different* — `zStep` writes `zB`, `qStep` writes
    `qB` — and the only thing separating their two launch records is the slot
    each output handle was loaded from.  Keying on the primitive's name alone
    would collapse them. -/
noncomputable def bwdPlan : Plan :=
  ⟨[ .proven sbStage, .declared zStep, .declared qStep,
     .proven dxStage, .proven dwStage ]⟩

/-- The device-write sequence realises it — vendor calls included, in position,
    and told apart by their arguments. -/
theorem bwdDriverBlas_realises :
    planOf? bwdTable bwdDeclared none bwdDriverBlas.deviceOps = some bwdPlan := rfl

/-- **How much of this plan is assumed: two steps.**  A number, not a caveat. -/
theorem bwdPlan_declaredCount : bwdPlan.declaredCount = 2 := rfl

/-- …and they name themselves. -/
theorem bwdPlan_declaredNames :
    bwdPlan.declaredNames = ["cl_cublas_sgemv", "cl_cublas_sgemv"] := rfl

theorem bwdPlan_exclusive : bwdPlan.Exclusive := by
  intro S hS
  simp only [bwdPlan, List.mem_cons, List.not_mem_nil, or_false, reduceCtorEq,
             false_or, PStep.proven.injEq] at hS
  rcases hS with h | h | h
  · subst h; exact mapStage_exclusive _ _ _ _ _
  · subst h; exact reduceStage_exclusive _ _ _ _ _ _ _ _ _
  · subst h; exact outerStage_exclusive _ _ _ _ _ _ (by decide) _ _

/-- **Proven all the way, with the gap named and counted.**

    Executing the emitted CLIF performs five device writes; under the kernel and
    declared tables those *are* `bwdPlan`; and the plan computes its denotation.
    What it rests on is `Honours R` — one hypothesis, covering exactly the two
    declared steps, which `bwdPlan_declaredNames` identifies.  Everything else
    in the chain is proven. -/
theorem bwd_host_computes_plan (R : Realisation) (hR : Honours R) (st : WSt) :
    ∃ k c', hsteps 0 bwdCodeBlas k
              ⟨0, AlgorithmLib.Clif.Env.empty, [], fun _ => 0, [], []⟩ = some c'
      ∧ planOf? bwdTable bwdDeclared none (c'.trace.zip c'.btrace) = some bwdPlan
      ∧ (bwdPlan.run R st).mem = bwdPlan.denote st.mem :=
  host_computes_plan (.ffi .cudaLaunch) ⟨0⟩ rfl bwdCodeBlas bwdDriverBlas 1 0
    AlgorithmLib.Clif.Env.empty [] (fun _ => 0) (by decide) rfl (by decide)
    (AlgorithmLib.Host.FarOk.of_noBases (by decide))
    (fun x hx b hb => AlgorithmLib.Host.noBases_primDests (by decide) x hx b hb)
    (fun j _ => by rw [Nat.zero_add]; rfl)
    bwdTable bwdDeclared bwdPlan none bwdDriverBlas_realises R hR
    bwdPlan_exclusive st


theorem bwdPipelineFull_runs (st : WSt) :
    (bwdPipelineFull.run st).mem = bwdPipelineFull.denote st.mem :=
  bwdPipelineFull.run_denote bwdPipelineFull_exclusive st

example : siluBwd.ew = sbStage.ew := rfl
example : kernel    = dxStage.ew := rfl
example : dwKernel  = dwStage.ew := rfl

/-- **Delta-only unfoldings of the two stages.**

    These exist so `bwd_chain` can be proven by *rewriting* rather than by
    definitional unification.  Unifying `map_then_reduce`'s conclusion against
    `dxStage.run (sbStage.run st)` directly makes the elaborator `whnf` its way
    through `StageSpec.run`, which is a fold over `GRID = 896` blocks: measured
    at **over 9 GB**, and with `maxRecDepth` raised it stack-overflows instead.
    Neither `rfl` below mentions `.run`, so `rw` replaces the stages
    syntactically and the unification that follows is immediate. -/
theorem dxStage_def : dxStage
    = reduceStage adjB wB adjIx wIx dxB K GRID (by decide) (by decide) := rfl

theorem sbStage_def : sbStage
    = mapStage siluBwdSpec siluBwdIn adjB EGRID (by decide) := rfl

theorem bwd_chain (st : WSt) (cta : Nat) (hlt : cta < GRID) :
    (dxStage.run (sbStage.run st)).mem dxB cta
      = bflyFold (dotStridedLane
          (fun a => denote (fun i => st.mem (siluBwdIn i) a) siluBwdSpec)
          (st.mem wB)
          (fun i l => adjIx.eval cta i l) (fun i l => wIx.eval cta i l) K)
          ⟨0, by decide⟩ := by
  rw [dxStage_def, sbStage_def]
  exact map_then_reduce siluBwdSpec siluBwdIn adjB wB dxB adjIx wIx
    (by decide) (by decide) (by decide) (by decide) K EGRID GRID st cta hlt
    (fun i hi l => ⟨i, by simpa [K, geom, EGRID, egeom, MapGeom.simple] using hi, l, rfl⟩)


end BackwardWide

#eval ShipScan.check "Warp.BackwardWide" `Warp.BackwardWide.main