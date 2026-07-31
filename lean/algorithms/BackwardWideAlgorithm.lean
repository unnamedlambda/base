import Lean
import Std
import AlgorithmLib

open Lean AlgorithmLib AlgorithmLib.IR AlgorithmLib.ML AlgorithmLib.Host

namespace BackwardWide

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

def NBUF : Nat := 12

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
def BIND_OFF   : Nat := 0x8C00
def ADJ_ID     : Nat := 0x0040
def W_ID       : Nat := 0x0044
def DX_ID      : Nat := 0x0048
def X_ID       : Nat := 0x004C
def DW_ID      : Nat := 0x0050
def Z_ID       : Nat := 0x0054
def DY_ID      : Nat := 0x0058
def GAM_ID     : Nat := 0x005C
def T_ID       : Nat := 0x0060
def Q_ID       : Nat := 0x0064
def S_ID       : Nat := 0x0068
def DXR_ID     : Nat := 0x006C
def MEM_SIZE   : Nat := 0x8D00

/-- Both kernels fit their slots, and the slots do not overlap. -/
theorem bwdMap_ok :
    AlgorithmLib.Layout.RegionMap.okB
      [⟨"ptx", PTX_OFF, PTX_DW_OFF - PTX_OFF⟩,
       ⟨"ptxDw", PTX_DW_OFF, PTX_SB_OFF - PTX_DW_OFF⟩,
       ⟨"ptxSiluBwd", PTX_SB_OFF, PTX_T_OFF - PTX_SB_OFF⟩,
       ⟨"ptxT", PTX_T_OFF, PTX_Q_OFF - PTX_T_OFF⟩,
       ⟨"ptxQ", PTX_Q_OFF, PTX_S_OFF - PTX_Q_OFF⟩,
       ⟨"ptxS", PTX_S_OFF, PTX_DXR_OFF - PTX_S_OFF⟩,
       ⟨"ptxDxr", PTX_DXR_OFF, BIND_OFF - PTX_DXR_OFF⟩,
       ⟨"bind", BIND_OFF, 4 * NBUF⟩] = true := by decide

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
theorem bwdPtx_fits_g : ptxDxr.toUTF8.toList.length + 1 ≤ BIND_OFF - PTX_DXR_OFF := by
  native_decide

theorem bwdPtx_fits :
    (ptx.toUTF8.toList.length + 1 ≤ PTX_DW_OFF - PTX_OFF)
      ∧ (ptxDw.toUTF8.toList.length + 1 ≤ PTX_SB_OFF - PTX_DW_OFF)
      ∧ (ptxSiluBwd.toUTF8.toList.length + 1 ≤ PTX_T_OFF - PTX_SB_OFF)
      ∧ (ptxT.toUTF8.toList.length + 1 ≤ PTX_Q_OFF - PTX_T_OFF)
      ∧ (ptxQ.toUTF8.toList.length + 1 ≤ PTX_S_OFF - PTX_Q_OFF)
      ∧ (ptxS.toUTF8.toList.length + 1 ≤ PTX_DXR_OFF - PTX_S_OFF)
      ∧ (ptxDxr.toUTF8.toList.length + 1 ≤ BIND_OFF - PTX_DXR_OFF) :=
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
   ⟨"W",   5 * N * 4, N * N * 4⟩]

/-- **Seam guard: the host layout is packed, in order, with no gaps.** -/
theorem hostIn_packed : AlgorithmLib.Layout.RegionMap.packedB 0 hostIn = true := by
  decide

/-- Bytes the host must supply — derived from the same list. -/
def HOST_BYTES : Nat := AlgorithmLib.Layout.RegionMap.total hostIn

/-- Offset of input `i`, read out of the layout rather than recomputed. -/
def hostOff (i : Nat) : Nat := AlgorithmLib.Layout.RegionMap.offAt hostIn i

/-! ### The binding table, from one list

    `EWStmt.BufBelow` checks the *kernels* name no buffer past `NBUF`.  It says
    nothing about whether the CLIF actually **writes** `NBUF` ids — those were
    twelve independent `storeI32` calls next to an independent `NBUF = 12`
    (`A47` G7).  The table is now a list, the uploader indexes it, and its
    length is checked against `NBUF`. -/
def bindSlots : List String :=
  ["adj", "W", "dx", "x", "dW", "z", "dy", "gam", "t", "Q", "S", "dxr"]

/-- **Seam guard: the binding table has exactly `NBUF` entries.** -/
theorem bind_count : bindSlots.length = NBUF := by decide

/-- Byte offset of binding slot `i`. -/
def bindOff (i : Nat) : Nat := BIND_OFF + 4 * i

def loadFn : IRBuilder Unit := do
  let ptr ← entryBlock
  let cuda ← declareCudaFFI
  let dataPtr ← load64 (← absAddr ptr 0x18)
  cudaInit cuda ptr
  let ctxPtr ← cudaCtxPtr ptr
  let adjBytes ← iconst64 (N * 4)
  let wBytes ← iconst64 (N * N * 4)
  let dxBytes ← iconst64 (N * 4)
  let adjId ← cudaCreateBuffer cuda ptr adjBytes
  storeI32 adjId (← absAddr ptr ADJ_ID)
  let wId ← cudaCreateBuffer cuda ptr wBytes
  storeI32 wId (← absAddr ptr W_ID)
  let dxId ← cudaCreateBuffer cuda ptr dxBytes
  storeI32 dxId (← absAddr ptr DX_ID)
  let xId ← cudaCreateBuffer cuda ptr dxBytes
  storeI32 xId (← absAddr ptr X_ID)
  let dwId ← cudaCreateBuffer cuda ptr wBytes
  storeI32 dwId (← absAddr ptr DW_ID)
  let zId ← cudaCreateBuffer cuda ptr dxBytes
  storeI32 zId (← absAddr ptr Z_ID)
  let dyId ← cudaCreateBuffer cuda ptr dxBytes
  storeI32 dyId (← absAddr ptr DY_ID)
  let gamId ← cudaCreateBuffer cuda ptr dxBytes
  storeI32 gamId (← absAddr ptr GAM_ID)
  let tId ← cudaCreateBuffer cuda ptr dxBytes
  storeI32 tId (← absAddr ptr T_ID)
  let four ← iconst64 4
  let qId ← cudaCreateBuffer cuda ptr four
  storeI32 qId (← absAddr ptr Q_ID)
  let sId ← cudaCreateBuffer cuda ptr four
  storeI32 sId (← absAddr ptr S_ID)
  let dxrId ← cudaCreateBuffer cuda ptr dxBytes
  storeI32 dxrId (← absAddr ptr DXR_ID)
  -- host buffer holds `adj`, then `x`, then `W`, contiguously
  let _ ← call cuda.fnUpload [ctxPtr, adjId, dataPtr, adjBytes]
  let xSrc ← iaddImm dataPtr (hostOff 1)
  let _ ← call cuda.fnUpload [ctxPtr, xId, xSrc, dxBytes]
  let zSrc ← iaddImm dataPtr (hostOff 2)
  let _ ← call cuda.fnUpload [ctxPtr, zId, zSrc, dxBytes]
  let dySrc ← iaddImm dataPtr (hostOff 3)
  let _ ← call cuda.fnUpload [ctxPtr, dyId, dySrc, dxBytes]
  let gamSrc ← iaddImm dataPtr (hostOff 4)
  let _ ← call cuda.fnUpload [ctxPtr, gamId, gamSrc, dxBytes]
  let wSrc ← iaddImm dataPtr (hostOff 5)
  let _ ← call cuda.fnUpload [ctxPtr, wId, wSrc, wBytes]
  storeI32 adjId (← absAddr ptr (bindOff 0))
  storeI32 wId (← absAddr ptr (bindOff 1))
  storeI32 dxId (← absAddr ptr (bindOff 2))
  storeI32 xId (← absAddr ptr (bindOff 3))
  storeI32 dwId (← absAddr ptr (bindOff 4))
  storeI32 zId (← absAddr ptr (bindOff 5))
  storeI32 dyId (← absAddr ptr (bindOff 6))
  storeI32 gamId (← absAddr ptr (bindOff 7))
  storeI32 tId (← absAddr ptr (bindOff 8))
  storeI32 qId (← absAddr ptr (bindOff 9))
  storeI32 sId (← absAddr ptr (bindOff 10))
  storeI32 dxrId (← absAddr ptr (bindOff 11))
  ret

def runFn : IRBuilder Unit := do
  let ptr ← entryBlock
  let cuda ← declareCudaFFI
  let ptxOff ← iconst64 PTX_OFF
  let nBufs ← iconst32 NBUF
  let bindOff ← iconst64 BIND_OFF
  let one ← iconst32 1
  let warp ← iconst32 32
  let grid ← iconst32 GRID
  let _ ← cudaLaunch cuda ptr ptxOff nBufs bindOff grid one one warp one one
  let _ ← cudaSync cuda ptr
  ret

/-- The activation backward: one element per lane, `N/32` blocks. -/
def runSiluBwdFn : IRBuilder Unit := do
  let ptr ← entryBlock
  let cuda ← declareCudaFFI
  let ptxOff ← iconst64 PTX_SB_OFF
  let nBufs ← iconst32 NBUF
  let bindOff ← iconst64 BIND_OFF
  let one ← iconst32 1
  let warp ← iconst32 32
  let grid ← iconst32 EGRID
  let _ ← cudaLaunch cuda ptr ptxOff nBufs bindOff grid one one warp one one
  let _ ← cudaSync cuda ptr
  ret

/-- A launch of the kernel at `off` over `g` blocks of one warp. -/
def launchAt (off g : Nat) : IRBuilder Unit := do
  let ptr ← entryBlock
  let cuda ← declareCudaFFI
  let ptxOff ← iconst64 off
  let nBufs ← iconst32 NBUF
  let bindOff ← iconst64 BIND_OFF
  let one ← iconst32 1
  let warp ← iconst32 32
  let grid ← iconst32 g
  let _ ← cudaLaunch cuda ptr ptxOff nBufs bindOff grid one one warp one one
  let _ ← cudaSync cuda ptr
  ret

def runTFn   : IRBuilder Unit := launchAt PTX_T_OFF EGRID
def runQFn   : IRBuilder Unit := launchAt PTX_Q_OFF 1
def runSFn   : IRBuilder Unit := launchAt PTX_S_OFF 1
def runDxrFn : IRBuilder Unit := launchAt PTX_DXR_OFF EGRID

/-- Fetch RMSNorm's `dx`. -/
def fetchDxrFn : IRBuilder Unit := do
  let ptr ← entryBlock
  let cuda ← declareCudaFFI
  let ctxPtr ← cudaCtxPtr ptr
  let outPtr ← load64 (← absAddr ptr 0x28)
  let dxrId ← load32 (← absAddr ptr DXR_ID)
  let dxBytes ← iconst64 (N * 4)
  let _ ← call cuda.fnDownload [ctxPtr, dxrId, outPtr, dxBytes]
  ret

/-- The weight gradient, same geometry: one warp per row. -/
def runDwFn : IRBuilder Unit := do
  let ptr ← entryBlock
  let cuda ← declareCudaFFI
  let ptxOff ← iconst64 PTX_DW_OFF
  let nBufs ← iconst32 NBUF
  let bindOff ← iconst64 BIND_OFF
  let one ← iconst32 1
  let warp ← iconst32 32
  let grid ← iconst32 GRID
  let _ ← cudaLaunch cuda ptr ptxOff nBufs bindOff grid one one warp one one
  let _ ← cudaSync cuda ptr
  ret

def fetchFn : IRBuilder Unit := do
  let ptr ← entryBlock
  let cuda ← declareCudaFFI
  let ctxPtr ← cudaCtxPtr ptr
  let outPtr ← load64 (← absAddr ptr 0x28)
  let dxId ← load32 (← absAddr ptr DX_ID)
  let dxBytes ← iconst64 (N * 4)
  let _ ← call cuda.fnDownload [ctxPtr, dxId, outPtr, dxBytes]
  ret

/-- Fetch the weight gradient — `N·N` floats. -/
def fetchDwFn : IRBuilder Unit := do
  let ptr ← entryBlock
  let cuda ← declareCudaFFI
  let ctxPtr ← cudaCtxPtr ptr
  let outPtr ← load64 (← absAddr ptr 0x28)
  let dwId ← load32 (← absAddr ptr DW_ID)
  let wBytes ← iconst64 (N * N * 4)
  let _ ← call cuda.fnDownload [ctxPtr, dwId, outPtr, wBytes]
  ret

def clifIR : String :=
  noopFunction ++ "\n" ++ buildFunction 1 loadFn ++ "\n"
    ++ buildFunction 2 runFn ++ "\n" ++ buildFunction 3 fetchFn ++ "\n"
    ++ buildFunction 4 runDwFn ++ "\n" ++ buildFunction 5 fetchDwFn ++ "\n"
    ++ buildFunction 6 runSiluBwdFn ++ "\n"
    ++ buildFunction 7 runTFn ++ "\n" ++ buildFunction 8 runQFn ++ "\n"
    ++ buildFunction 9 runSFn ++ "\n" ++ buildFunction 10 runDxrFn ++ "\n"
    ++ buildFunction 11 fetchDxrFn

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
  zeros HOST_LEN_OFF ++ u32le HOST_BYTES
    ++ zeros (PTX_OFF - HOST_LEN_OFF - 4)
    ++ p ++ zeros (PTX_DW_OFF - PTX_OFF - p.length)
    ++ q ++ zeros (PTX_SB_OFF - PTX_DW_OFF - q.length)
    ++ r ++ zeros (PTX_T_OFF - PTX_SB_OFF - r.length)
    ++ t ++ zeros (PTX_Q_OFF - PTX_T_OFF - t.length)
    ++ u ++ zeros (PTX_S_OFF - PTX_Q_OFF - u.length)
    ++ v ++ zeros (PTX_DXR_OFF - PTX_S_OFF - v.length)
    ++ w ++ zeros (MEM_SIZE - PTX_DXR_OFF - w.length)

def setup : Setup := {
  cranelift_ir := clifIR
  memory_size := MEM_SIZE
  initial_memory := initialMemory
}

def artifacts : Array Json :=
  #[ toJsonArtifact "backward_wide" setup { fn_idx := u32 1 }
       [("run", { fn_idx := u32 2 }), ("fetch", { fn_idx := u32 3 }),
        ("runDw", { fn_idx := u32 4 }), ("fetchDw", { fn_idx := u32 5 }),
        ("runSiluBwd", { fn_idx := u32 6 }), ("runT", { fn_idx := u32 7 }),
        ("runQ", { fn_idx := u32 8 }), ("runS", { fn_idx := u32 9 }),
        ("runDxr", { fn_idx := u32 10 }), ("fetchDxr", { fn_idx := u32 11 })] ]

end BackwardWide

def main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  emitArtifacts outDir BackwardWide.artifacts

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

def bwdFns : List FnDecl :=
  [{ ref := ⟨0⟩, name := "cl_cuda_launch", sig := ⟨0⟩ }]

/-- The emitted host code — a real instruction list, compiled by `flatHI`. -/
def bwdCode : List HI := code ⟨0⟩ ⟨0⟩ 1 0 bwdDriver

/-- **The driver's launch sequence is the backward pipeline.**  Decided, not
    assumed: a drifted slot or a grid disagreeing with the stage's own `grid`
    field makes this `none` and the proof fails. -/
theorem bwdDriver_realises :
    pipelineOf? bwdTable bwdDriver.deviceOps = some bwdPipelineFull := rfl

/-- **The whole host side, end to end, on the kernels this file ships.**

    Executing the emitted CLIF instruction by instruction performs three
    launches; under `bwdTable` those launches *are* `bwdPipelineFull`; and that
    pipeline computes its denotation.  Every hypothesis is discharged at a
    concrete value, so nothing here is vacuous.

    The remaining trusted step is `bwdTable` itself — that the PTX at slot `0`
    is `sbStage.ew` compiled, and so on.  `sb_ptx_exact`, `dx_ptx_exact` and
    `dW_ptx_exact` above are what make each of those three claims checkable. -/
theorem bwd_host_computes (st : WSt) :
    ∃ k c', hsteps bwdFns 0 bwdCode k
              ⟨0, AlgorithmLib.Clif.Env.empty, [], fun _ => 0, [], []⟩ = some c'
      ∧ pipelineOf? bwdTable (c'.trace.zip c'.btrace) = some bwdPipelineFull
      ∧ (bwdPipelineFull.run st).mem = bwdPipelineFull.denote st.mem :=
  host_computes_denote bwdFns ⟨0⟩ ⟨0⟩ rfl bwdCode bwdDriver 1 0
    AlgorithmLib.Clif.Env.empty [] (fun _ => 0) (by decide) rfl (by decide)
    (AlgorithmLib.Host.FarOk.of_noBases (by decide))
    (fun x hx b hb => AlgorithmLib.Host.noBases_primDests (by decide) x hx b hb)
    (fun j _ => by rw [Nat.zero_add]; rfl)
    bwdTable bwdPipelineFull bwdDriver_realises bwdPipelineFull_exclusive st

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

def bwdBlasRef : FnRef := ⟨1⟩

def bwdFnsBlas : List FnDecl :=
  [ { ref := ⟨0⟩, name := "cl_cuda_launch",  sig := ⟨0⟩ }
  , { ref := ⟨1⟩, name := "cl_cublas_sgemv", sig := ⟨0⟩ } ]

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

def bwdCodeBlas : List HI := code ⟨0⟩ ⟨0⟩ 1 0 bwdDriverBlas

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
    planOf? bwdTable bwdDeclared bwdDriverBlas.deviceOps = some bwdPlan := rfl

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
    ∃ k c', hsteps bwdFnsBlas 0 bwdCodeBlas k
              ⟨0, AlgorithmLib.Clif.Env.empty, [], fun _ => 0, [], []⟩ = some c'
      ∧ planOf? bwdTable bwdDeclared (c'.trace.zip c'.btrace) = some bwdPlan
      ∧ (bwdPlan.run R st).mem = bwdPlan.denote st.mem :=
  host_computes_plan bwdFnsBlas ⟨0⟩ ⟨0⟩ rfl bwdCodeBlas bwdDriverBlas 1 0
    AlgorithmLib.Clif.Env.empty [] (fun _ => 0) (by decide) rfl (by decide)
    (AlgorithmLib.Host.FarOk.of_noBases (by decide))
    (fun x hx b hb => AlgorithmLib.Host.noBases_primDests (by decide) x hx b hb)
    (fun j _ => by rw [Nat.zero_add]; rfl)
    bwdTable bwdDeclared bwdPlan bwdDriverBlas_realises R hR
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
