import Lean
import Std
import AlgorithmLib

open Lean AlgorithmLib AlgorithmLib.IR AlgorithmLib.ML

namespace GradWarp

/-- Inputs per lane, and therefore outputs per lane: `∂out/∂xₖ` for each `k`. -/
def G : Nat := 4

def GRID : Nat := 16384
def LANES : Nat := GRID * 32
def NIN : Nat := LANES * G
def NOUT : Nat := LANES * G

/-- `v₀ = x₀ · x₁` -/
def b0 : Expr (G + 0) := .mul (.var ⟨0, by decide⟩) (.var ⟨1, by decide⟩)
/-- `v₁ = v₀ · exp x₂`, referring to `v₀` by variable. -/
def b1 : Expr (G + 1) := .mul (.var ⟨4, by decide⟩) (.exp (.var ⟨2, by decide⟩))
/-- `out = v₁ + x₃ · v₀` — `v₀` read a second time. -/
def outE : Expr (G + 2) :=
  .add (.var ⟨5, by decide⟩) (.mul (.var ⟨3, by decide⟩) (.var ⟨4, by decide⟩))

def tele : Tele G 2 := (Tele.nil.cons b0).cons b1

/-- The gradient, as one program with the intermediates shared. -/
def gp : VProg G G := gradProg tele outE

def depFn : Nat → Nat := fun i => if i = 0 then 0 else 2

/-- The narrowed gradient — the one that is linear in depth. -/
def gpD : VProg G G := gradProgD tele outE depFn

/-- Lane `l` of block `c` owns inputs `[(c*32+l)*G, +G)`. -/
def inIx : Fin G → IdxE := fun i =>
  .add (.mul (.add (.mul .ctaId (.lit 32)) .laneId) (.lit G)) (.lit i.val)

/-- …and writes its `G` partials to the matching slots. -/
def outIx : Fin G → IdxE := fun i =>
  .add (.mul (.add (.mul .ctaId (.lit 32)) .laneId) (.lit G)) (.lit i.val)

def inBuf : Buf := 0
def outBuf : Buf := 1

/-- Scratch register for the store, above everything the program allocates. -/
def resReg : Nat := G + teleSlots gp.binds + slots (gp.outs ⟨0, by decide⟩) + 8

/-- **The kernel**: load the inputs, emit the shared telescope once, then
    compute and store each partial. -/
def kernel : EWStmt := compileVWKernel (fun _ => inBuf) outBuf inIx gp outIx resReg

def ptx : String := emitProvenKernel "main" kernel

/-- Scratch register for the narrowed kernel's store. -/
def resRegD : Nat := G + teleSlots gpD.binds + slots (gpD.outs ⟨0, by decide⟩) + 8

/-- The same pipeline, from the **narrowed** program.  Nothing about the
    lowering changes — that is the point of `VProg` being the interface. -/
def kernelD : EWStmt := compileVWKernel (fun _ => inBuf) outBuf inIx gpD outIx resRegD

def ptxD : String := emitProvenKernel "main" kernelD

/-! ### Seam guards (`A47` G2) -/
theorem grad_guards :
    kernel.BufBelow 2 ∧ kernelD.BufBelow 2
      ∧ kernel.StageEligibleB outBuf = true ∧ kernelD.StageEligibleB outBuf = true
      ∧ FlatTargetsOkB (flatKernel (expandEW kernel)) = true
      ∧ FlatTargetsOkB (flatKernel (expandEW kernelD)) = true
      ∧ FlatPrintableB (flatKernel (expandEW kernel)) = true
      ∧ FlatPrintableB (flatKernel (expandEW kernelD)) = true := by
  refine ⟨by decide, by decide, by decide, by decide, ?_, ?_, ?_, ?_⟩ <;> native_decide

/-- One lane per input group. -/
theorem grad_geometry : GRID * 32 = LANES ∧ NIN = LANES * G := by decide

-- ── memory layout ───────────────────────────────────────────────────────────

def PTX_OFF   : Nat := 0x0100
/-- The narrowed kernel's PTX, in its own slot so both can be launched. -/
def PTX_D_OFF : Nat := 0x18000
def BIND_OFF  : Nat := 0x30000
def IN_ID     : Nat := 0x0040
def OUT_ID    : Nat := 0x0044
def MEM_SIZE  : Nat := 0x30100

def loadFn : IRBuilder Unit := do
  let ptr ← entryBlock
  let cuda ← declareCudaFFI
  let dataPtr ← load64 (← absAddr ptr 0x18)
  cudaInit cuda ptr
  let ctxPtr ← cudaCtxPtr ptr
  let inBytes ← iconst64 (NIN * 4)
  let inId ← cudaCreateBuffer cuda ptr inBytes
  storeI32 inId (← absAddr ptr IN_ID)
  let outBytes ← iconst64 (NOUT * 4)
  let outId ← cudaCreateBuffer cuda ptr outBytes
  storeI32 outId (← absAddr ptr OUT_ID)
  let _ ← call cuda.fnUpload [ctxPtr, inId, dataPtr, inBytes]
  storeI32 inId  (← absAddr ptr BIND_OFF)
  storeI32 outId (← absAddr ptr (BIND_OFF + 4))
  ret

def runFn : IRBuilder Unit := do
  let ptr ← entryBlock
  let cuda ← declareCudaFFI
  let ptxOff ← iconst64 PTX_OFF
  let nBufs ← iconst32 2
  let bindOff ← iconst64 BIND_OFF
  let one ← iconst32 1
  let warp ← iconst32 32
  let grid ← iconst32 GRID
  let _ ← cudaLaunch cuda ptr ptxOff nBufs bindOff grid one one warp one one
  let _ ← cudaSync cuda ptr
  ret

/-- The same launch, from the narrowed kernel's PTX.  Everything else — the
    buffers, the binding table, the geometry — is identical, which is the point:
    only the *program* differs. -/
def runDFn : IRBuilder Unit := do
  let ptr ← entryBlock
  let cuda ← declareCudaFFI
  let ptxOff ← iconst64 PTX_D_OFF
  let nBufs ← iconst32 2
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
  let outId ← load32 (← absAddr ptr OUT_ID)
  let outBytes ← iconst64 (NOUT * 4)
  let _ ← call cuda.fnDownload [ctxPtr, outId, outPtr, outBytes]
  ret

def clifIR : String :=
  noopFunction ++ "\n" ++ buildFunction 1 loadFn ++ "\n"
    ++ buildFunction 2 runFn ++ "\n" ++ buildFunction 3 fetchFn ++ "\n"
    ++ buildFunction 4 runDFn

def initialMemory : List UInt8 :=
  let ptxBytes := ptx.toUTF8.toList ++ [0]
  let ptxDBytes := ptxD.toUTF8.toList ++ [0]
  zeros PTX_OFF ++ ptxBytes ++ zeros (PTX_D_OFF - PTX_OFF - ptxBytes.length)
    ++ ptxDBytes ++ zeros (MEM_SIZE - PTX_D_OFF - ptxDBytes.length)

/-- Both kernels fit their slots, and the slots do not overlap. -/
theorem gradMap_ok :
    AlgorithmLib.Layout.RegionMap.okB
      [⟨"ptx", PTX_OFF, PTX_D_OFF - PTX_OFF⟩,
       ⟨"ptxD", PTX_D_OFF, BIND_OFF - PTX_D_OFF⟩,
       ⟨"bind", BIND_OFF, 8⟩] = true := by decide

theorem gradPtx_fits :
    (ptx.toUTF8.toList.length + 1 ≤ PTX_D_OFF - PTX_OFF)
      ∧ (ptxD.toUTF8.toList.length + 1 ≤ BIND_OFF - PTX_D_OFF) := by native_decide

def setup : Setup := {
  cranelift_ir := clifIR
  memory_size := MEM_SIZE
  initial_memory := initialMemory
}

def artifacts : Array Json :=
  #[ toJsonArtifact "grad_warp" setup { fn_idx := u32 1 }
       [("run", { fn_idx := u32 2 }), ("fetch", { fn_idx := u32 3 }),
        ("runD", { fn_idx := u32 4 })] ]

end GradWarp

def main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  emitArtifacts outDir GradWarp.artifacts

namespace GradWarp

/-- **Every output of the gradient program is the spec's derivative.**

    `gradProg_correct` at this instance: output `k`, read from memory, is
    `sderiv` of the whole let-bound expression with respect to input `k`.  No
    `NumLaws` — this is an exact `Float32` equality, not a real-number one. -/
theorem grad_outputs_are_derivatives (env : Fin G → Float32) (k : Fin G) :
    denote env (gp.get k) = denote env (sderiv (tele.bind outE) k) :=
  gradProg_correct tele outE env k

/-- **The narrowed gradient computes the same derivatives — under its two
    declared propositions, and no others.**

    `ZeroTermFree` licenses dropping a term that is semantically zero;
    `ZeroLaws` is what `sderiv_notUses` needs to turn "this variable does not
    occur" into "its partial is zero".  Both hold in ℝ and in `Int`
    (`zeroLaws_Int` is proven); for `Float32` they fail only where an infinity
    or a negative zero is involved, so for finite weights the narrowed sweep
    differs from `sderiv` in at most the sign bit of an exact zero.

    That is a much tighter approximation than `ExpIsEx2`, which this same demo
    already measures at 1.04e-5 — and unlike the sparsity optimisation every AD
    framework performs silently, it is written down. -/
theorem gradD_outputs_are_derivatives (hz : ZeroTermFree Float32)
    (hzl : ZeroLaws Float32) (env : Fin G → Float32) (k : Fin G) :
    denote env (gpD.get k) = denote env (sderiv (tele.bind outE) k) := by
  refine gradProgD_correct tele outE env depFn hz ?_ k
  intro i h q hq
  match i, h with
  | 0, _ => exact absurd hq (by simp [depFn])
  | 1, _ =>
      refine sderiv_notUses hzl _ q _ ?_
      have h2 : q.val < 2 := by simpa [depFn] using hq
      show NotUses q (Expr.mul (.var ⟨4, by decide⟩) (.exp (.var ⟨2, by decide⟩)))
      refine ⟨fun hc => ?_, fun hc => ?_⟩ <;>
        · have hv := congrArg Fin.val hc
          simp at hv
          omega

/-- **The compiled kernel computes those derivatives, per lane, in a register.**

    The telescope is emitted once and shared by all `G` outputs — the property
    that makes this quadratic rather than exponential — and output `k` still
    denotes `gp.get k`.

    Not a claim about memory: the statement below has no store in it.  It said
    "from memory" until `A47` G14, which was false — the multi-output path had
    no memory-level theorem at all, because that one needs distinct outputs to
    write distinct addresses and nothing had stated it.  `grad_stores` does. -/
theorem grad_kernel_computes (k : Fin G) (st : WSt) (l : Lane) :
    ((compileW (teleEnv (fun q => .reg q.val) G gp.binds) (G + teleSlots gp.binds)
        (gp.outs k)).2).eval
        (runW (compileW (teleEnv (fun q => .reg q.val) G gp.binds)
                (G + teleSlots gp.binds) (gp.outs k)).1
          (runW (compileTele (fun q => .reg q.val) G gp.binds)
            (runW (loadSeq (fun _ => inBuf) inIx) st))) l
      = denote (fun q => st.mem inBuf ((inIx q).eval 0 0 l)) (gp.get k) :=
  compileVWKernel_value (fun _ => inBuf) inIx gp k st l

/-- The emission order, split at each output — and the output does **not**
    reappear later.  That second half is what makes the store obligation
    dischargeable: without it, "no later output writes my address" is false at
    the output itself. -/
theorem grad_split (i : Fin G) :
    ∃ pre post, List.finRange G = pre ++ i :: post ∧ i ∉ post := by
  match i with
  | ⟨0, h⟩ =>
      refine ⟨[], [⟨1, by decide⟩, ⟨2, by decide⟩, ⟨3, by decide⟩], rfl, ?_⟩
      intro hc; simp at hc
  | ⟨1, h⟩ =>
      refine ⟨[⟨0, by decide⟩], [⟨2, by decide⟩, ⟨3, by decide⟩], rfl, ?_⟩
      intro hc; simp at hc; rcases hc with h1 | h1 <;> exact absurd (congrArg Fin.val h1) (by decide)
  | ⟨2, h⟩ =>
      refine ⟨[⟨0, by decide⟩, ⟨1, by decide⟩], [⟨3, by decide⟩], rfl, ?_⟩
      intro hc; simp at hc
  | ⟨3, h⟩ =>
      refine ⟨[⟨0, by decide⟩, ⟨1, by decide⟩, ⟨2, by decide⟩], [], rfl, ?_⟩
      intro hc; simp at hc

/-- **The gradient lands in memory, not just in a register.**

    `grad_kernel_computes` says what a lane computes; this says where it goes —
    the statement the multi-output path was missing until `A47` G14, and the one
    a reader would assume from the demo's name.

    The obligation is real and here it is discharged: output `q` at lane `l`
    addresses `(c·32+l)·G + q`, injective in `(l, q)` because `q < G`.  So no
    later output overwrites an earlier one — the failure the register-level
    theorem could never have detected. -/
theorem grad_stores (k : Fin G) (st : WSt) (l0 : Lane) :
    ∃ s0 : WSt,
      ((kernel.elabIn 0).run st).mem outBuf ((outIx k).eval 0 0 l0)
        = ((compileW (teleEnv (fun q => .reg q.val) G gp.binds) (G + teleSlots gp.binds)
            (gp.outs k)).2).eval
            (runW (compileW (teleEnv (fun q => .reg q.val) G gp.binds)
              (G + teleSlots gp.binds) (gp.outs k)).1 s0) l0 := by
  obtain ⟨pre, post, hsplit, hnot⟩ := grad_split k
  refine compileVWKernel_stores (fun _ => inBuf) outBuf inIx gp outIx resReg k st l0
    pre post hsplit ?_ ?_
  · intro j hj l hc
    have hv : (0 * 32 + l.val) * G + j.val = (0 * 32 + l0.val) * G + k.val := hc
    have hjk : j.val ≠ k.val := fun he => hnot (Fin.ext he ▸ hj)
    have h1 : j.val < G := j.isLt
    have h2 : k.val < G := k.isLt
    have h3 : l.val < 32 := l.isLt
    have h4 : l0.val < 32 := l0.isLt
    show False
    simp only [G] at *
    omega
  · intro s' l hl
    have hv : (0 * 32 + l.val) * G + k.val = (0 * 32 + l0.val) * G + k.val := hl
    have h2 : k.val < G := k.isLt
    have h3 : l.val < 32 := l.isLt
    have h4 : l0.val < 32 := l0.isLt
    have : l = l0 := Fin.ext (by simp only [G] at *; omega)
    rw [this]

/-- **The emitted PTX runs the gradient kernel — exactly, from raw launch.**

    This is the same floor the forward kernels stand on (`mlp_ptx_exact`,
    `gemv_ptx_computes_spec`): the printed instruction list, executed by a
    program counter over real branches from instruction 0, performs the
    compiled statement for every block, unconditionally.  Without this the
    gradient's chain stopped at the warp machine; with it, proven object =
    executed object holds for training exactly as for inference. -/
theorem grad_ptx_exact (cta : Nat) (m : MState) :
    ∃ k m', steps cta (flatKernel (expandEW kernel)) k (0, m)
          = some ((flatKernel (expandEW kernel)).length, m')
      ∧ m'.toWSt = ((expandEW kernel).elabIn cta).run m.toWSt :=
  flatKernel_sound_idxFree cta (expandEW kernel) (expandEW_expFree kernel)
    (expandEW_idxFree kernel
      (compileVWKernel_idxFree (fun _ => inBuf) outBuf inIx gp outIx resReg
        (fun _ => ⟨⟨⟨⟨trivial, trivial⟩, trivial⟩, trivial⟩, trivial⟩)
        (fun _ => ⟨⟨⟨⟨trivial, trivial⟩, trivial⟩, trivial⟩, trivial⟩)))
    (expandEW_flat kernel (compileVWKernel_flat _ _ _ _ _ _)) m

theorem gradD_stores (k : Fin G) (st : WSt) (l0 : Lane) :
    ∃ s0 : WSt,
      ((kernelD.elabIn 0).run st).mem outBuf ((outIx k).eval 0 0 l0)
        = ((compileW (teleEnv (fun q => .reg q.val) G gpD.binds) (G + teleSlots gpD.binds)
            (gpD.outs k)).2).eval
            (runW (compileW (teleEnv (fun q => .reg q.val) G gpD.binds)
              (G + teleSlots gpD.binds) (gpD.outs k)).1 s0) l0 := by
  obtain ⟨pre, post, hsplit, hnot⟩ := grad_split k
  refine compileVWKernel_stores (fun _ => inBuf) outBuf inIx gpD outIx resRegD k st l0
    pre post hsplit ?_ ?_
  · intro j hj l hc
    have hv : (0 * 32 + l.val) * G + j.val = (0 * 32 + l0.val) * G + k.val := hc
    have hjk : j.val ≠ k.val := fun he => hnot (Fin.ext he ▸ hj)
    have h1 : j.val < G := j.isLt
    have h2 : k.val < G := k.isLt
    have h3 : l.val < 32 := l.isLt
    have h4 : l0.val < 32 := l0.isLt
    show False
    simp only [G] at *
    omega
  · intro s' l hl
    have hv : (0 * 32 + l.val) * G + k.val = (0 * 32 + l0.val) * G + k.val := hl
    have h2 : k.val < G := k.isLt
    have h3 : l.val < 32 := l.isLt
    have h4 : l0.val < 32 := l0.isLt
    have : l = l0 := Fin.ext (by simp only [G] at *; omega)
    rw [this]

/-- **And the narrowed kernel's PTX runs it, from raw launch.** -/
theorem gradD_ptx_exact (cta : Nat) (m : MState) :
    ∃ k m', steps cta (flatKernel (expandEW kernelD)) k (0, m)
          = some ((flatKernel (expandEW kernelD)).length, m')
      ∧ m'.toWSt = ((expandEW kernelD).elabIn cta).run m.toWSt :=
  flatKernel_sound_idxFree cta (expandEW kernelD) (expandEW_expFree kernelD)
    (expandEW_idxFree kernelD
      (compileVWKernel_idxFree (fun _ => inBuf) outBuf inIx gpD outIx resRegD
        (fun _ => ⟨⟨⟨⟨trivial, trivial⟩, trivial⟩, trivial⟩, trivial⟩)
        (fun _ => ⟨⟨⟨⟨trivial, trivial⟩, trivial⟩, trivial⟩, trivial⟩)))
    (expandEW_flat kernelD (compileVWKernel_flat _ _ _ _ _ _)) m

/-- The same, discharging the `exp` in the spec via the one declared identity. -/
theorem grad_ptx_runs_kernel (h : ExpIsEx2) (cta : Nat) (m : MState) :
    ∃ k m', steps cta (flatKernel (expandEW kernel)) k (0, m)
          = some ((flatKernel (expandEW kernel)).length, m')
      ∧ m'.toWSt = (kernel.elabIn cta).run m.toWSt := by
  obtain ⟨k, m', hs, hw⟩ := grad_ptx_exact cta m
  exact ⟨k, m', hs, by rw [hw]; exact expandEW_run h kernel cta 0 m.toWSt⟩

end GradWarp
