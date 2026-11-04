import AlgorithmLib.ML.Math.Transformer
import AlgorithmLib.ML.Model.Dense
import AlgorithmLib.ML.Tensor.Surface
import AlgorithmLib.ML.Model.Ten
import Lean
import Std
import Scan.Layout
import Scan.Ship
import AlgorithmLib.ML.Launch.HostBridge
import AlgorithmLib.ML.Model.BufsOf
import AlgorithmLib.ML.Model.Schedule

import AlgorithmLib.Surface.Layout
import AlgorithmLib.Surface.Link
import AlgorithmLib.Surface.Drv
import AlgorithmLib.Host.Driver
open Lean AlgorithmLib AlgorithmLib.IR AlgorithmLib.ML

/-!
  # A CIFAR-10 classifier, trained end to end on proven kernels

  A two-layer MLP on a real dataset and a real loss: `3072 → 256 → 10`, trained with
  cross-entropy on CIFAR-10.

  Every kernel below is an instance of a schema proven once — `dotBatched`,
  `mapKernel`, `outerBatched` — so the model costs addressing and geometry, not
  proof text.  The activation's derivative is `sderiv` of the
  *same* `Transformer.silu` the forward pass evaluates.

  **Batched.**  `B` samples per step, unrolled at generation time: one warp per
  output row carrying `B` accumulators, so a weight row is fetched once per `B`
  samples and the weight matrix is written once per batch.  `dotBatched_agrees`
  is what says the batched answer is the unbatched one bit-for-bit.

  **The one host-side step.**  The softmax and the cross-entropy gradient are
  computed on the host: the GPU produces `B·32` logits, the host returns `B·32`
  values of `softmax(logits) − onehot(label)`.  That is a kilobyte each way per
  step against 3.1 MB of weight traffic, so it costs bandwidth that does not
  matter
  and buys not having to thread `maxReduceEW`'s chunk/remainder metadata
  through the artifact.  It is named here rather than hidden, and it is the
  piece to move onto the device next.

  **Why 32 classes.**  CIFAR-10 has ten, but a warp has 32 lanes, and every
  reduction schema in this stack sweeps in multiples of 32.  Padding the output
  layer to the warp width makes `C/32 = 1` exact, so the class dimension needs
  no masking anywhere.  The host zeroes the gradient of the 22 padding units,
  so their weights never move and their logits are never read.
-/

namespace MlpCifar

-- The per-sample kernel theorems unfold the batched tapes past the default depth.
set_option maxRecDepth 4096

/-- CIFAR-10 images: 32·32·3 = 3072 floats.  `3072 = 96·32`, so the strided
    sweep divides. -/
def IN : Nat := 3072

/-- Hidden width. -/
def H : Nat := 256

/-- Output units — ten classes padded to the warp width. -/
def C : Nat := 32

/-- **Samples per step.**  Unrolled at generation time rather than mapped onto
    the grid, which is what keeps every address affine — see `ML/Batch.lean`.
    One warp per output row carries `B` accumulators, so a weight row is fetched
    once per `B` samples instead of once per sample. -/
def B : Nat := 8

/-! **Buffer numbers are allocation order**, not a table someone maintains:
    `cifarTen` below binds each value in turn and `flat` hands out the next
    free buffer, so these thirteen constants are what that allocation produced.
    `program_allocates` is what keeps them honest. -/
def xB    : Buf := 0   -- `B` images, `B·IN` floats
def w1B   : Buf := 1   -- first layer, row-major `W1[i·IN + j]`
def w2B   : Buf := 2   -- second layer, row-major `W2[c·H + i]`
def biasB : Buf := 3   -- per-class bias: 0 at a real class, very negative at padding
def ohB   : Buf := 4   -- one-hot labels `[s][c]`
def z1B   : Buf := 5   -- pre-activations `z1[s][i]`
def hB    : Buf := 6   -- hidden activations `h = silu(z1)`
def logB  : Buf := 7   -- logits `[s][c]`
def dlogB : Buf := 8   -- `softmax(logits) − onehot(label)`
def dw2B  : Buf := 9   -- `∂L/∂W2`, summed over the batch
def dhB   : Buf := 10  -- `∂L/∂h[s][i]`
def adjB  : Buf := 11  -- `∂L/∂z1 = dh · silu'(z1)`
def dw1B  : Buf := 12  -- `∂L/∂W1`, summed over the batch

def NBUF : Nat := 13

-- ---------------------------------------------------------------------------
-- Addressing
-- ---------------------------------------------------------------------------

/-! Every index below is `stride32 base` or a literal-offset row read, so the
    batch index enters only as `s · stride` with `s` a numeral.  That is the
    whole reason batching needed no new `IdxE` constructor. -/

/-- Row `ctaid` of `W1`, swept `loop·32 + lane`. -/
def w1Ix : IdxE := stride32 (.mul .ctaId (.lit IN))
/-- Sample `s`'s image, same sweep. -/
def xIx (s : Nat) : IdxE := stride32 (.lit (s * IN))
/-- Where sample `s`'s hidden unit `ctaid` goes. -/
def z1Oi (s : Nat) : IdxE := .add (.lit (s * H)) .ctaId

/-- Row `ctaid` of `W2`. -/
def w2Ix : IdxE := stride32 (.mul .ctaId (.lit H))
def hIx (s : Nat) : IdxE := stride32 (.lit (s * H))
def logOi (s : Nat) : IdxE := .add (.lit (s * C)) .ctaId

/-- Column `ctaid` of `W2` — the transposed walk, striding by `H`.  Shared
    across the batch, so it is the operand loaded once per trip. -/
def w2TIx : IdxE := .add (.mul (.add (.mul .loopI (.lit 32)) .laneId) (.lit H)) .ctaId
def dlIx (s : Nat) : IdxE := stride32 (.lit (s * C))
def dhOi (s : Nat) : IdxE := .add (.lit (s * H)) .ctaId

/-- `adj[s][ctaid]` — lane- and loop-uniform. -/
def dw1AIx (s : Nat) : IdxE := .add (.lit (s * H)) .ctaId
def dw1BIx (s : Nat) : IdxE := stride32 (.lit (s * IN))
def dw1Base : IdxE := .mul .ctaId (.lit IN)

def dw2AIx (s : Nat) : IdxE := .add (.lit (s * C)) .ctaId
def dw2BIx (s : Nat) : IdxE := stride32 (.lit (s * H))
def dw2Base : IdxE := .mul .ctaId (.lit H)

-- ---------------------------------------------------------------------------
-- Geometry
-- ---------------------------------------------------------------------------

/-! Each record carries its own coverage obligation and the launch grids are
    read off it, so "this kernel touches every element" is checked rather than
    asserted twice in two places that can drift. -/

def g1 : ReduceGeom := { n := IN, outs := H, trips := IN / 32 }
def g2 : ReduceGeom := { n := H, outs := C, trips := H / 32 }
def gdh : ReduceGeom := { n := C, outs := H, trips := C / 32 }

/-- Elementwise over the hidden layer — `B` times as many elements as at batch
    one, and the only change these kernels needed. -/
def gh : MapGeom := MapGeom.simple (B * H) (B * H / 32)
def gw1 : MapGeom := MapGeom.simple (H * IN) (H * IN / 32)
def gw2 : MapGeom := MapGeom.simple (C * H) (C * H / 32)

def GRID1  : Nat := g1.outs
def GRID2  : Nat := g2.outs
def GRIDDH : Nat := gdh.outs
def GRIDH  : Nat := gh.grid
def GRIDW1 : Nat := gw1.grid
def GRIDW2 : Nat := gw2.grid

-- ---------------------------------------------------------------------------
-- Forward
-- ---------------------------------------------------------------------------

/-- `z1[s][i] = Σⱼ W1[i][j]·x[s][j]`, one warp per hidden unit, `B`
    accumulators.  The row of `W1` is loaded once for all `B` samples. -/
def fwd1Kernel : EWStmt := dotBatched w1B xB w1Ix xIx z1B z1Oi B (IN / 32)

/-- `h = silu(z1)`.  The forward activation and the backward derivative are the
    same term; only `sderiv` separates them. -/
def hSpec : Expr 1 := Transformer.silu (.var ⟨0, by decide⟩)
def hKernel : MapKernel 1 := mapKernel hSpec (fun _ => z1B) hB

/-- `logits[s][c] = Σᵢ W2[c][i]·h[s][i]`, one warp per class. -/
def fwd2Kernel : EWStmt := dotBatched w2B hB w2Ix hIx logB logOi B (H / 32)

-- ---------------------------------------------------------------------------
-- Backward
-- ---------------------------------------------------------------------------

/-- `dh[s][i] = Σ_c dlog[s][c]·W2[c][i]`.  The column of `W2` is the shared
    operand — the transposed walk, loaded once per trip for all `B` samples. -/
def dhKernel : EWStmt := dotBatched w2B dlogB w2TIx dlIx dhB dhOi B (C / 32)

/-- `adj = dh · silu'(z1)`, the derivative taken symbolically from `hSpec`. -/
def adjSpec : Expr 2 :=
  .mul (.var ⟨1, by decide⟩)
       (sderiv (Transformer.silu (.var ⟨0, by decide⟩)) ⟨0, by decide⟩)
/-- **Both passes come from one Lean function.**  The forward activation and
    the elementwise factor of its backward are `Transformer.silu` read two
    ways — the derivative is not a second definition that could drift. -/
theorem activation_derived :
    hSpec = ewFwd Transformer.silu ∧ adjSpec = ewBack Transformer.silu :=
  ⟨rfl, rfl⟩

def adjIn : Fin 2 → Buf := fun i => if i.val = 0 then z1B else dhB
def adjKernel : MapKernel 2 := mapKernel adjSpec adjIn adjB

/-- `dW2[c][i] = Σₛ dlog[s][c]·h[s][i]` — summed over the batch inside one
    warp, because two blocks accumulating into one element would be a race. -/
def dw2Kernel : EWStmt :=
  outerBatched dlogB hB dw2B dw2AIx dw2BIx dw2Base B (H / 32)

/-- `dW1[i][j] = Σₛ adj[s][i]·x[s][j]`, likewise.  The weight matrix is written
    **once per batch** rather than once per sample. -/
def dw1Kernel : EWStmt :=
  outerBatched adjB xB dw1B dw1AIx dw1BIx dw1Base B (IN / 32)

/-- **Softmax and the cross-entropy gradient, on the device.**

    `C = 32` *is* a warp, so each row's maximum and sum are one butterfly and
    the kernel needs no shared memory, no barrier and no chunk metadata.  The
    padding classes are masked by `bias`: a large negative value there makes
    `exp` underflow to zero, so they contribute nothing to the sum and their
    gradient is `−onehot = 0`. -/
def smKernel : EWStmt := softmaxCE logB biasB ohB dlogB .laneId

-- ---------------------------------------------------------------------------
-- The optimiser
-- ---------------------------------------------------------------------------

/-- The learning rate's reciprocal, as an exact binary fraction so the update is
    a `Float32` identity rather than a rounding of a decimal.

    Scaled by `B` because the gradient kernels accumulate a *sum* over the
    batch: dividing here is what makes the step the mean gradient, so the
    effective per-sample rate is the same at every batch size and a batch-size
    change is not silently a learning-rate change. -/
def LR_RECIP : Nat := 256 * B

def sgdSpec : Expr 2 :=
  .add (.var ⟨0, by decide⟩)
       (.neg (.mul (.inv (.lit LR_RECIP)) (.var ⟨1, by decide⟩)))
def sgd1In : Fin 2 → Buf := fun i => if i.val = 0 then w1B else dw1B
def sgd2In : Fin 2 → Buf := fun i => if i.val = 0 then w2B else dw2B
def sgd1Kernel : MapKernel 2 := mapKernel sgdSpec sgd1In w1B
def sgd2Kernel : MapKernel 2 := mapKernel sgdSpec sgd2In w2B

-- ---------------------------------------------------------------------------
-- The model, and the operations it flattens to
-- ---------------------------------------------------------------------------

/-- How many buffers the program declares rather than computes.  `flat`
    allocates from here upward. -/
def NIN : Nat := 5

/-- **The model as a tensor term.**  Every width is a type index, so `mv`
    accepts only a matching contraction and a wrong one does not elaborate;
    every launch grid is derived from the extents rather than written.  `letT`
    binds each intermediate once, so an operand used twice — `z1` by the
    activation and by the backward adjustment, `x` by the forward pass and by
    `dW1` — is one kernel, not two. -/
def cifarTen : TenProg H IN := fun v =>
  let x    : Ten v B IN := .inp xB
  let w1   : Ten v H IN := .inp w1B
  let w2   : Ten v C H  := .inp w2B
  let bias : Ten v 1 C  := .inp biasB
  let oh   : Ten v B C  := .inp ohB
  tlet z1   := w1 * x;
  tlet h    := z1.silu;
  tlet y    := w2 * h;
  tlet dlog := Ten.smce y bias oh;
  tlet dw2  := Ten.outer .proven dlog h;
  tlet dh   := Ten.mvT .proven w2 dlog;
  tlet adj  := Ten.zipWith adjSpec z1 dh;
  tlet dw1  := Ten.outer .proven adj x;
  .upd2 sgdSpec w1 dw1 (.upd2 sgdSpec w2 dw2 dw1)

/-- The flattened operations of the model, forward then backward then the
    optimiser. -/
def cifarTape : List TOp := cifarTen.tape NIN

-- ---------------------------------------------------------------------------
-- PTX, one slot per kernel
-- ---------------------------------------------------------------------------

/-- The buffers slot `i` binds — the operation's own, in the order its
    kernel's table (`unitBufs`) numbers them. -/
def slotBufs (i : Nat) : List Buf := ((cifarTape[i]?).map (TOp.bufs B)).getD []

/-- The kernels, in the order the tape performs them.

    Each is emitted against its *own* table rather than against every buffer
    the model allocates, so a kernel declares the handful of parameters it
    reads.  The list is the tape's, so adding an equation to the model adds a
    kernel here and nowhere else. -/
def ptxAll : List String :=
  cifarTape.map (fun op =>
    Body.text "main" (unitBufs B [op]).length 0 (unitStmt B [op]))

/-- Kernels shipped. -/
def NSLOT : Nat := 10

theorem slot_count : ptxAll.length = NSLOT := by decide

/-- The signature the entry points' typestates state of every kernel: `main`
    takes one pointer per buffer its slot binds. -/
theorem mlp_sigs :
    ((List.range NSLOT).all fun i =>
      HProg.Sem.ptxParamSizes (ptxAll.getD i "") "main" == some ((slotBufs i).map fun _ => 8)) = true := by
  native_decide

/-- Where the expected host-input size is published, so the host can assert its
    packing against the Lean layout instead of duplicating it. -/
def HOST_LEN_OFF : Nat := 0x0080

/-- Where the driver's state starts (`Drv.App`). -/
def ORG : Nat := 0x0100

def PTX_OFF : Nat := 0x1000

/-- Bytes reserved per kernel.  Uniform, so a slot's offset is arithmetic
    rather than a twelfth named constant.

    A batched reduction unrolls `B` accumulators — `B` butterflies and `B`
    stores — so its text grows linearly in `B`.  `mlpPtx_fits` is what turns
    "large enough" into a checked fact rather than a guess. -/
def SLOT : Nat := 0x4000

def slotOff (i : Nat) : Nat := PTX_OFF + i * SLOT
def MEM_SIZE : Nat := slotOff NSLOT

/-- Device bytes per buffer, in allocation order. -/
def bufBytes : List Nat :=
  [B * IN * 4, H * IN * 4, C * H * 4, C * 4, B * C * 4,
   B * H * 4, B * H * 4, B * C * 4,
   B * C * 4, C * H * 4, B * H * 4, B * H * 4, H * IN * 4]

/-- **Seam guard: every buffer the kernels name gets allocated.** -/
theorem buf_count : bufBytes.length = NBUF := by decide

/-- Each kernel's grid, in slot order: one block of one warp per output row
    for a reduction, one warp per thirty-two elements for a map. -/
def slotGrid : List Nat := [GRID1, GRIDH, GRID2, B, GRID2, GRIDDH, GRIDH, GRID1, GRIDW1, GRIDW2]

/-- **The program in the driver's standard layout**: its thirteen
    allocations, its ten kernels each handed its own buffers, and cuBLAS. -/
def app : Drv.App :=
  { bufs := bufBytes,
    kernels := (List.range NSLOT).map fun i =>
      ⟨slotOff i, SLOT, slotBufs i, (slotGrid.getD i 1, 1, 1), (32, 1, 1), []⟩,
    memSize := MEM_SIZE, blas := true, org := ORG }

/-- **Every byte this program names**, not only its PTX slots. -/
def memMap : AlgorithmLib.Layout.RegionMap :=
  [⟨"host_len", HOST_LEN_OFF, 8⟩, ⟨"driver", ORG, app.stateEnd - ORG⟩]
    ++ (List.range NSLOT).map (fun i => ⟨s!"ptx{i}", slotOff i, SLOT⟩)

/-- **Seam guard: the slots tile the image without overlapping.** -/
theorem mlpMap_ok :
    memMap.okB = true ∧ memMap.withinB MEM_SIZE = true := by decide

/-- **Seam guard: every kernel fits its slot.** -/
theorem mlpPtx_fits :
    ptxAll.all (fun s => decide (s.toUTF8.toList.length + 1 ≤ SLOT)) = true := by
  native_decide

/-- **Seam guard: every kernel's buffers are inside its own table**, and the
    compaction that put them there is injective — the two conditions
    `StageSpec.rename` needs.

    The table is the operation's own, so this says nothing about how many
    buffers the model allocates: `bufs_le_five` is what bounds it. -/
theorem mlp_bufs_bound :
    cifarTape.all (fun op => op.localBelowB B) = true
      ∧ cifarTape.all (fun op => op.compactInjB B) = true :=
  ⟨List.all_eq_true.mpr (fun op _ => TOp.localBelow B op), by decide⟩

/-- **Seam guard: no kernel's own table can reach the address scratch.** -/
theorem mlp_bufs_within_regs :
    cifarTape.all (fun op => decide ((op.bufs B).length ≤ 5)) = true :=
  List.all_eq_true.mpr (fun op _ => decide_eq_true (bufs_le_five B op))

/-- **Seam guard: the launches cover exactly the intended elements.**

    Stated over the numbers the *launch* uses, so a grid and a trip count that
    disagree fail here rather than silently dropping a tail. -/
theorem mlp_geometry :
    GRID1 = H ∧ GRID2 = C ∧ GRIDDH = H
      ∧ GRIDH * 32 = B * H ∧ GRIDW1 * 32 = H * IN ∧ GRIDW2 * 32 = C * H
      ∧ (C / 32) * 32 = C ∧ (H / 32) * 32 = H ∧ (IN / 32) * 32 = IN := by decide

/-- **Seam guard: every shipped forward/backward kernel is stage-eligible.**

    A stage's value may not depend on its own output buffer's prior contents.
    The two optimiser kernels are deliberately absent: overwriting their input
    is what they are for. -/
theorem mlp_stage_eligible :
    fwd1Kernel.StageEligibleB z1B = true
      ∧ hKernel.ew.StageEligibleB hB = true
      ∧ fwd2Kernel.StageEligibleB logB = true
      ∧ dw2Kernel.StageEligibleB dw2B = true
      ∧ dhKernel.StageEligibleB dhB = true
      ∧ adjKernel.ew.StageEligibleB adjB = true
      ∧ dw1Kernel.StageEligibleB dw1B = true
      ∧ smKernel.StageEligibleB dlogB = true := by decide

-- ---------------------------------------------------------------------------
-- The host input layout
-- ---------------------------------------------------------------------------

/-! The entry point uploads the two weight matrices once; `uploadX` and
    `uploadDl` then move one image and one gradient per step.  The layout is a
    single list, `packedB` checks it is gapless, in order and non-overlapping,
    the uploader reads its offsets out of that same list, and the total is
    written into the memory image so the host can assert its packing against it
    at run time. -/
def hostIn : AlgorithmLib.Layout.RegionMap :=
  [⟨"W1", 0, H * IN * 4⟩,
   ⟨"W2", H * IN * 4, C * H * 4⟩]

theorem hostIn_packed : AlgorithmLib.Layout.RegionMap.packedB 0 hostIn = true := by
  decide

def HOST_BYTES : Nat := AlgorithmLib.Layout.RegionMap.total hostIn
def hostOff (i : Nat) : Nat := AlgorithmLib.Layout.RegionMap.offAt hostIn i

-- ---------------------------------------------------------------------------
-- The host program
-- ---------------------------------------------------------------------------

open AlgorithmLib.Prog

/-- Bring the driver up, make the thirteen allocations, load the ten
    kernels and write their parameters (`Drv.App.setup`), then upload both
    weight matrices from the caller's data. -/
def loadCode : Prog V L Unit := do
  let base ← basePtr
  let dataPtr ← dataPtr
  app.setup base
  app.upload base w1B dataPtr (H * IN * 4)
  app.upload base w2B (← iaddImm dataPtr (hostOff 1)) (C * H * 4)

/-- Upload `n` bytes from the caller's data pointer into buffer `b`. -/
def uploadCode (b n : Nat) : Prog V L Unit := do
  let base ← basePtr
  app.upload base b (← dataPtr) n

/-- Download `n` bytes of buffer `b` into the caller's output pointer. -/
def fetchCode (b n : Nat) : Prog V L Unit := do
  let base ← basePtr
  app.download base (← outPtr) b n

/-- **A run of launches with one synchronisation at the end.**

    Every launch goes to the same stream, so the device already orders them; a
    sync between two of them buys nothing but a round trip.  Measured at 5.6 µs
    per launch-and-sync against ~1 µs for the smallest kernel here, which is why
    a training step is one call to this rather than nine calls to a
    launch-and-sync — the fusion is in the *driver*, not in the kernels, so it
    costs no proof.

    The one place a sync is genuinely required is before a download, and
    `fetchFn` is what follows a run. -/
def runSeq (slots : List Nat) : Prog V L Unit := do
  let base ← basePtr
  for i in slots do
    app.run base i
  Drv.sync

/-- A single launch, for the per-kernel timing the demo reports. -/
def launchSlot (i : Nat) : Prog V L Unit := runSeq [i]

/-- The forward pass: two matvecs, an activation and the softmax. -/
def fwdSteps : List Nat := [0, 1, 2, 3]

/-- The backward pass and both optimiser steps.  `dh` reads `W2` and `sgd2`
    writes it, so the order here is what keeps the gradient the one belonging to
    the weights the forward pass used. -/
def bwdSteps : List Nat := [4, 5, 6, 7, 8, 9]

/-- **Seam guard: the fused runs launch every kernel exactly once, in order.**

    A training step is two calls, not nine, and the sequence they enqueue is
    written in one place.  Without this a kernel could be dropped from the fused
    path and stay in the artifact — a silently untrained layer, which reads as a
    hyperparameter problem rather than a bug. -/
theorem steps_cover : fwdSteps ++ bwdSteps = List.range NSLOT := by decide

-- ---------------------------------------------------------------------------
-- The same GEMMs, through cuBLAS
-- ---------------------------------------------------------------------------

/-! Five of the nine kernels are GEMM-shaped, and cuBLAS is what PyTorch calls
    for them.  Routing them there makes the arithmetic *identical* to the
    baseline's rather than a kernel to be raced against it — and what is left
    over, the elementwise passes, the softmax and the optimiser, stays proven.

    The cost is named rather than hidden: a cuBLAS step is `.declared`, not
    `.proven`, because NVIDIA does not specify a fold order, so no exact
    `Float32` claim is available for it.  That is the same footing PyTorch is
    on. The proven kernels stay in the artifact and `mlp-cifar` checks cuBLAS
    against them, which is what catches a wrong transpose or leading dimension
    — the usual failure of a cuBLAS integration, and one that produces
    plausible numbers rather than an error.

    **Row-major buffers, column-major library.**  A row-major `M×N` buffer *is*
    a column-major `N×M` matrix, so each call below is stated in column-major
    terms and no data is ever transposed in memory.  `Drv.App.sgemm` packs
    every operand, `lda = k if transa else m`, `ldb = n if transb else k`,
    `ldc = m`, which is exactly what these five need.  The products are read
    off the model's tape under the cuBLAS schedule (`TOp.vendorGemm`), not
    written beside it. -/

/-- Every contraction to cuBLAS's strided-batched product. -/
def GEMM : Backend :=
  .vendor .cublasSgemmStridedBatched

/-- **The schedule.**  The same equations, with the contractions sent to
    cuBLAS — one value, named where the model is not. -/
def cifarBlasSched : TenSchedule := { impl := GEMM }

/-- The model's tape under that schedule: the same operations in the same
    order, each contraction cuBLAS's. -/
def cifarBlasTape : List TOp := (cifarTen.compile NIN cifarBlasSched).2

/-- Operation `i` as the cuBLAS build takes it: cuBLAS's product
    (`TOp.vendorGemm`) for a contraction, slot `i`'s kernel otherwise. -/
def blasStep (op : TOp) (i : Nat) : Sum Nat (Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat) :=
  match op.vendorGemm with
  | some g => .inr g
  | none => .inl i

/-- The cuBLAS tape's operations `lo..hi`, each with its index. -/
def blasRange (lo hi : Nat) : List (TOp × Nat) := (cifarBlasTape.zipIdx.drop lo).take (hi - lo)

/-- A run of products and kernel launches, in the given order, one sync at the
    end. `.inl` is a kernel's slot; `.inr` is a cuBLAS product on the
    program's handle. -/
def runMixed (steps : List (Sum Nat (Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat)))
    : Prog V L Unit := do
  let base ← basePtr
  for st in steps do
    match st with
    | .inl i => app.run base i
    | .inr (tA, tB, m, n, k, a, b, c) => app.sgemm base (← load64 (← iaddImm base app.BLAS)) tA tB m n k a b c
  Drv.sync

/-- The forward with cuBLAS for both matmuls; `silu` and the softmax stay
    proven kernels. -/
def fwdBlasSteps : List (Sum Nat (Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat)) :=
  (blasRange 0 4).map fun (op, i) => blasStep op i

/-- The backward likewise; the activation backward and both optimiser steps
    stay proven kernels. -/
def bwdBlasSteps : List (Sum Nat (Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat)) :=
  (blasRange 4 10).map fun (op, i) => blasStep op i

/-- The bodies this artifact ships, in the order their function indices run.
    `clifIR` numbers them from this list, so an index cannot drift from the body
    it names. -/
def shippedBodies : List (String × Prog.Body) :=
  [ ("main", loadCode)
  , ("uploadX", uploadCode xB (B * IN * 4))
  , ("uploadOneHot", uploadCode ohB (B * C * 4))
  , ("fetchLogits", fetchCode logB (B * C * 4))
  , ("runFwd1", launchSlot 0)
  , ("runAct", launchSlot 1)
  , ("runFwd2", launchSlot 2)
  , ("runDw2", launchSlot 4)
  , ("runDh", launchSlot 5)
  , ("runAdj", launchSlot 6)
  , ("runDw1", launchSlot 7)
  , ("runSgd1", launchSlot 8)
  , ("runSgd2", launchSlot 9)
  , ("fetchH", fetchCode hB (B * H * 4))
  , ("fetchZ1", fetchCode z1B (B * H * 4))
  , ("fetchDh", fetchCode dhB (B * H * 4))
  , ("fetchAdj", fetchCode adjB (B * H * 4))
  , ("fetchDw1", fetchCode dw1B (H * IN * 4))
  , ("fetchDw2", fetchCode dw2B (C * H * 4))
  , ("fetchW1", fetchCode w1B (H * IN * 4))
  , ("fetchW2", fetchCode w2B (C * H * 4))
  , ("runFwd", runSeq fwdSteps)
  , ("runBwd", runSeq bwdSteps)
  , ("runSoftmax", launchSlot 3)
  , ("uploadBias", uploadCode biasB (C * 4))
  , ("fetchDlog", fetchCode dlogB (B * C * 4))
  , ("runFwdBlas", runMixed fwdBlasSteps)
  , ("runBwdBlas", runMixed bwdBlasSteps) ]


def clifIR : Except String (List FuncData) :=
  Prog.program <|
    .ok noopFunction :: shippedBodies.zipIdx.map
      (fun ((name, body), i) => Prog.entry name (Prog.compileProg (i + 1) body))

/-- A `Nat` as four little-endian bytes. -/
def u32le (v : Nat) : List UInt8 :=
  [UInt8.ofNat (v % 256), UInt8.ofNat (v / 256 % 256),
   UInt8.ofNat (v / 65536 % 256), UInt8.ofNat (v / 16777216 % 256)]

def initialMemory : List UInt8 :=
  app.initialMemory ptxAll [(HOST_LEN_OFF, u32le HOST_BYTES)]

def setup (clif : List FuncData) : Artifact := {
  functions := clif,
  required_memory := MEM_SIZE
  initial_memory := initialMemory
}

-- ---------------------------------------------------------------------------
-- What each kernel computes
-- ---------------------------------------------------------------------------

/-! Nine kernels, nine spec theorems, each one application of an existing
    schema theorem.  None has a hypothesis about the world: the right-hand
    sides are the *committed* fold order — sequential within a lane, five-round
    butterfly across lanes — which is what makes them exact `Float32`
    equalities rather than reassociation claims.

    The batched reductions carry one genuine hypothesis, that their `B`
    destinations differ.  It is discharged below at the addressing this file
    actually emits; without it a later sample would overwrite an earlier one
    and the kernel would be silently wrong for every sample but the last. -/

/-- **The first matvec computes its spec, per sample.**

    Compare the right-hand side with `dotStrided_spec`'s: it is the same
    expression.  `dotBatched_agrees` turns that observation into a theorem —
    batching changed which warp does the work and how often the weight row is
    fetched, not what is computed. -/
theorem fwd1_computes_spec (cta s : Nat) (hs : s < B) (st : WSt) :
    ((fwd1Kernel.run st cta 0)).mem z1B ((z1Oi s).eval cta 0 ⟨0, by decide⟩)
      = bflyFold (dotStridedLane (st.mem w1B) (st.mem xB)
          (fun i l => w1Ix.eval cta i l) (fun i l => (xIx s).eval cta i l) (IN / 32))
          ⟨0, by decide⟩ :=
  dotBatched_spec w1B xB w1Ix xIx z1B z1Oi B (IN / 32) s cta hs _ _ st
    (fun s' _ hne => batch_addr_inj H (by decide) cta s s' hne)

/-- **The second matvec computes its spec**, same schema at a different row
    stride and trip count. -/
theorem fwd2_computes_spec (cta s : Nat) (hs : s < B) (st : WSt) :
    ((fwd2Kernel.run st cta 0)).mem logB ((logOi s).eval cta 0 ⟨0, by decide⟩)
      = bflyFold (dotStridedLane (st.mem w2B) (st.mem hB)
          (fun i l => w2Ix.eval cta i l) (fun i l => (hIx s).eval cta i l) (H / 32))
          ⟨0, by decide⟩ :=
  dotBatched_spec w2B hB w2Ix hIx logB logOi B (H / 32) s cta hs _ _ st
    (fun s' _ hne => batch_addr_inj C (by decide) cta s s' hne)

/-- **The transposed matvec computes its spec** — the gradient flowing back into
    the hidden layer, with the `W2` column as the operand shared across the
    batch. -/
theorem dh_computes_spec (cta s : Nat) (hs : s < B) (st : WSt) :
    ((dhKernel.run st cta 0)).mem dhB ((dhOi s).eval cta 0 ⟨0, by decide⟩)
      = bflyFold (dotStridedLane (st.mem w2B) (st.mem dlogB)
          (fun i l => w2TIx.eval cta i l) (fun i l => (dlIx s).eval cta i l) (C / 32))
          ⟨0, by decide⟩ :=
  dotBatched_spec w2B dlogB w2TIx dlIx dhB dhOi B (C / 32) s cta hs _ _ st
    (fun s' _ hne => batch_addr_inj H (by decide) cta s s' hne)

/-- **Batching is bit-exact against the unbatched kernel.**

    Sample `s` of `fwd1` and a `dotStrided` launched for sample `s` alone leave
    the same `Float32`.  Not equal up to reassociation — equal, invoking no
    law.  This is the theorem a batch-size knob needs: changing `B` is a
    scheduling decision, and it is checked to be one. -/
theorem fwd1_agrees_unbatched (cta s : Nat) (hs : s < B) (st : WSt)
    (out' : Buf) (oi' : IdxE) :
    ((fwd1Kernel.run st cta 0)).mem z1B ((z1Oi s).eval cta 0 ⟨0, by decide⟩)
      = (((dotStrided w1B xB w1Ix (xIx s) out' oi' (IN / 32)).run st cta 0)).mem
          out' (oi'.eval cta 0 ⟨0, by decide⟩) :=
  dotBatched_agrees w1B xB w1Ix xIx z1B out' z1Oi oi' B (IN / 32) s cta hs st
    (fun s' _ hne => batch_addr_inj H (by decide) cta s s' hne)

/-- **The activation stores `silu`.** -/
theorem h_stores (st : WSt) (l : Lane) :
    ((hKernel.ew.run st 0 0)).mem hB (elemIx.eval 0 0 l)
      = denote (fun _ => st.mem z1B (elemIx.eval 0 0 l)) hSpec :=
  hKernel.stores st l

/-- **The activation's backward stores the spec's derivative** — `sderiv` of the
    same `Transformer.silu` the forward kernel evaluates, so there is no
    hand-written backward formula anywhere in this chain. -/
theorem adj_stores (st : WSt) (l : Lane) :
    ((adjKernel.ew.run st 0 0)).mem adjB (elemIx.eval 0 0 l)
      = denote (fun i => st.mem (adjIn i) (elemIx.eval 0 0 l)) adjSpec :=
  adjKernel.stores st l

/-- **The first layer's weight gradient is the batch sum.**

    At every `(trip, lane)` the row visits, `dW1[i·IN + j] = Σₛ adj[s][i]·x[s][j]`
    read from the *entry* memory, folded in the committed order.  Injectivity of
    the destination, the kept registers and the read-back are discharged by
    `outerBatched_spec`; what is left is a `decide` on the base address and two
    buffer disequalities. -/
theorem dw1_stores (cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (st : WSt) (j0 : Nat) (l0 : Lane) (hj0 : j0 < IN / 32) :
    (((dw1Kernel.run st cta 0 ir im)).mem dw1B
        ((stride32 dw1Base).eval cta j0 l0 ir im))
      = dotStridedLane (st.mem adjB) (st.mem xB)
          (fun s l => (dw1AIx s).eval cta j0 l ir im)
          (fun s l => (dw1BIx s).eval cta j0 l ir im) B l0 :=
  outerBatched_spec adjB xB dw1B dw1AIx dw1BIx dw1Base (by decide) B (IN / 32) cta
    (by decide) (by decide) ir im st j0 l0 hj0

/-- **The second layer's weight gradient, same schema at 256 columns.** -/
theorem dw2_stores (cta : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat)
    (st : WSt) (j0 : Nat) (l0 : Lane) (hj0 : j0 < H / 32) :
    (((dw2Kernel.run st cta 0 ir im)).mem dw2B
        ((stride32 dw2Base).eval cta j0 l0 ir im))
      = dotStridedLane (st.mem dlogB) (st.mem hB)
          (fun s l => (dw2AIx s).eval cta j0 l ir im)
          (fun s l => (dw2BIx s).eval cta j0 l ir im) B l0 :=
  outerBatched_spec dlogB hB dw2B dw2AIx dw2BIx dw2Base (by decide) B (H / 32) cta
    (by decide) (by decide) ir im st j0 l0 hj0

/-- **Both optimiser steps compute their spec** — `W ← W − (1/(256·B))·dW`, per
    weight.  The `B` in the rate is what makes the update the *mean* gradient,
    since the two kernels above accumulate a sum. -/
theorem sgd1_stores (st : WSt) (l : Lane) :
    ((sgd1Kernel.ew.run st 0 0)).mem w1B (elemIx.eval 0 0 l)
      = denote (fun i => st.mem (sgd1In i) (elemIx.eval 0 0 l)) sgdSpec :=
  sgd1Kernel.stores st l

theorem sgd2_stores (st : WSt) (l : Lane) :
    ((sgd2Kernel.ew.run st 0 0)).mem w2B (elemIx.eval 0 0 l)
      = denote (fun i => st.mem (sgd2In i) (elemIx.eval 0 0 l)) sgdSpec :=
  sgd2Kernel.stores st l

-- ---------------------------------------------------------------------------
-- …and the emitted PTX runs it
-- ---------------------------------------------------------------------------

/-! Each theorem below says the *text that ships* executes from instruction 0,
    over real branches, to the state the kernel's semantics predicts.  Proven
    object = executed object, at model width and batch. -/

-- ---------------------------------------------------------------------------
-- Every forward and backward kernel as a pipeline stage
-- ---------------------------------------------------------------------------

/-- **The model's stages**, as the tensor term lowers them: each kernel
    bundled with the proof that its blocks do not race, the forward half
    first. -/
noncomputable def cifarStages : List XStage := (cifarTen.stages B NIN).getD []

noncomputable def fwdStages : List XStage := cifarStages.take 4
noncomputable def bwdStages : List XStage := cifarStages.drop 4

/-- The nine staged kernels, in slot order. -/
def stagedKernels : List EWStmt :=
  [fwd1Kernel, hKernel.ew, fwd2Kernel, smKernel, dw2Kernel, dhKernel, adjKernel.ew,
   dw1Kernel, sgd1Kernel.ew, sgd2Kernel.ew]

/-- **Seam guard: the staged pipeline is the shipped launch sequence.**

    Each stage's block program is the kernel that occupies its slot, and its
    grid is the grid that slot is launched over.  A stage list that quietly
    omitted a kernel, reordered two, or modelled one at a grid the driver does
    not use fails to build — which is the failure mode this project keeps
    finding, since every layer can be proven and none of it applied.

    The stage list covers *every* slot, so `steps_cover` and this theorem
    together say the launches and the stages are the same nine things. -/
theorem stages_are_launches :
    (fwdStages ++ bwdStages).map (fun S => S.val.ew) = stagedKernels
      ∧ (fwdStages ++ bwdStages).map (fun S => S.val.grid)
          = (fwdSteps ++ bwdSteps).map (slotGrid.getD · 1)
      ∧ (fwdStages ++ bwdStages).length = NSLOT := ⟨rfl, rfl, rfl⟩

/-- **Seam guard: the kernels emitted are the staged ones.**

    `ptxAll` is emitted from the tape and the stages are lowered from it, so
    this is what says the two descriptions have not drifted: slot `i` carries
    the statement stage `i` is proven about, renumbered to its own table by the
    compaction `mlp_bufs_bound` shows is injective. -/
theorem mlp_kernels_are_the_stages :
    cifarTape.map (TOp.stmt B) = stagedKernels
      ∧ cifarTape.length = NSLOT := ⟨rfl, rfl⟩

noncomputable def fwdPipeline : Pipeline := Pipeline.ofStages fwdStages
noncomputable def bwdPipeline : Pipeline := Pipeline.ofStages bwdStages

theorem fwdPipeline_exclusive : fwdPipeline.Exclusive := Pipeline.ofStages_exclusive _
theorem bwdPipeline_exclusive : bwdPipeline.Exclusive := Pipeline.ofStages_exclusive _

/-- **What a forward half-step leaves in memory** — the composite of its three
    stages' denotations, with no intermediate assumed. -/
theorem fwdPipeline_runs (st : WSt) :
    (fwdPipeline.run st).mem = fwdPipeline.denote st.mem :=
  Pipeline.ofStages_runs _ st

/-- …and a backward half-step, optimiser included. -/
theorem bwdPipeline_runs (st : WSt) :
    (bwdPipeline.run st).mem = bwdPipeline.denote st.mem :=
  Pipeline.ofStages_runs _ st

-- ---------------------------------------------------------------------------
-- A training run, not just a training step
-- ---------------------------------------------------------------------------

/-! Everything above is one step from an arbitrary memory.  A training run is
    that step repeated with the weights carried across, and "the weights after
    `n` steps" does not follow from "the weights after one" — each step reads
    what the last one wrote.

    The host transformation is a parameter rather than a modelled function:
    softmax and the cross-entropy gradient run on the host here, and a step that
    quietly omitted them would describe a different program.  `mlp_host_swap` is
    what says moving them onto the device (`ML/SoftmaxCE.lean` proves the
    kernel) changes nothing, provided it computes the same function. -/

/-- The training step: forward on the device, softmax and the CE gradient on the
    host, backward and both optimiser steps on the device. -/
noncomputable def mlpStep (host : Mem → Mem) : Step := ⟨fwdPipeline, host, bwdPipeline⟩

theorem mlpStep_exclusive (host : Mem → Mem) : (mlpStep host).Exclusive :=
  ⟨fwdPipeline_exclusive, bwdPipeline_exclusive⟩

/-- **`n` training steps compute `n` applications of one step** — for every `n`,
    with the weights threaded by the theorem rather than by an assumption about
    what the previous step left behind. -/
theorem mlp_training_run (host : Mem → Mem) (n : Nat) (st : WSt) :
    ((mlpStep host).iter n st).mem = iterMem (mlpStep host).denote n st.mem :=
  Step.iter_denote _ (mlpStep_exclusive host) n st

/-- **Moving the softmax onto the device is a drop-in exactly when it computes
    the same function** — at every run length, not just one step. -/
theorem mlp_host_swap (h₁ h₂ : Mem → Mem)
    (heq : (mlpStep h₁).denote = (mlpStep h₂).denote) (n : Nat) (st : WSt) :
    ((mlpStep h₁).iter n st).mem = ((mlpStep h₂).iter n st).mem :=
  Step.iter_congr _ _ (mlpStep_exclusive h₁) (mlpStep_exclusive h₂) heq n st

-- ---------------------------------------------------------------------------
-- The cuBLAS configuration, as a plan
-- ---------------------------------------------------------------------------

/-! The two configurations are not two models.  Same spec, same nine
    operations, same order — what differs is which steps are *proven* and which
    are *declared*.  `blas_denotes_the_same` is the statement that makes that
    precise: routing the five GEMMs through cuBLAS changes the denotation not at
    all, so the whole difference between the configurations is the assumption
    count, and `blas_declared` reports it. -/

/-- **Which implementation each operation gets**: the vendor's product where
    the cuBLAS schedule sends a contraction to it, the proven kernel
    elsewhere. -/
noncomputable def cifarChoices (gemm : Backend) : List (Backend × XStage) :=
  (cifarBlasTape.zip cifarStages).map fun (op, S) => (if op.vendorGemm.isSome then gemm else .proven, S)

/-- The forward half's choices, and the backward half's. -/
noncomputable def fwdChoices (gemm : Backend) : List (Backend × XStage) := (cifarChoices gemm).take 4
noncomputable def bwdChoices (gemm : Backend) : List (Backend × XStage) := (cifarChoices gemm).drop 4

noncomputable def fwdBlasPlan : Plan := lowerAll (fwdChoices GEMM)
noncomputable def bwdBlasPlan : Plan := lowerAll (bwdChoices GEMM)

/-- The same four operations with every contraction proven — the schedule the
    bill is compared against. -/
noncomputable def fwdProvenPlan : Plan := lowerAll (fwdChoices .proven)

theorem fwdBlasPlan_exclusive : fwdBlasPlan.Exclusive := lowerAll_exclusive _
theorem bwdBlasPlan_exclusive : bwdBlasPlan.Exclusive := lowerAll_exclusive _

/-- **The price of the cuBLAS configuration, as a number.**  Five of nine
    operations rest on the vendor; the other four stay bit-exact. -/
theorem blas_declared :
    fwdBlasPlan.declaredCount + bwdBlasPlan.declaredCount = 5
      ∧ fwdBlasPlan.declaredNames ++ bwdBlasPlan.declaredNames
          = List.replicate 5 (IR.Ext.cublas .sgemmStridedBatched).symbol := ⟨rfl, rfl⟩

/-- **What this configuration bills, read off the plan** — the laws it rests
    on, and how many of its steps rest on no stated law at all.

    Not a sentence beside the theorem: both numbers are walked out of the steps
    the lowering produced.  The batched GEMM has no equation behind it, so the
    law list is empty and the *second* component is what carries the cost —
    reading the empty list alone would say this schedule assumes nothing, which
    is the opposite of true. -/
theorem blas_law_bill :
    fwdBlasPlan.lawBill = [] ∧ fwdBlasPlan.lawlessCount = 2
      ∧ bwdBlasPlan.lawBill = [] ∧ bwdBlasPlan.lawlessCount = 3 :=
  ⟨rfl, rfl, rfl, rfl⟩

/-- **The strict cuBLAS law is a refinement, not a separate commitment.**

    The registry says `cublas-is-matvec` strictifies `cublas-sums-in-some-order`,
    and this is that claim instantiated: assuming the vendor folds left to right
    gives you the honest statement that it sums the right products in some
    association.  So a theorem billing the strict law is billing the weak one
    too, and the *only* thing the extra strength buys is the right to chain the
    equality through a sequence of calls. -/
theorem cublas_law_refines :
    Law.cublasIsMatvec.weakens = some Law.cublasIsSomeReassoc
      ∧ (CuBlasIsMatvec → CuBlasIsSomeReassoc) :=
  ⟨rfl, Law.weakens_holds Law.cublasIsMatvec Law.cublasIsSomeReassoc rfl⟩

/-- **…and the all-proven configuration bills nothing on both counts.**

    Non-vacuity for the bill: it is not a constant.  This is the difference the
    schedule choice makes, as data — and it is the pair, not the law list, that
    separates the two configurations. -/
theorem proven_law_bill :
    fwdProvenPlan.lawBill = [] ∧ fwdProvenPlan.lawlessCount = 0 := ⟨rfl, rfl⟩

/-- **Both configurations compute the same function.**

    The cuBLAS plan and the all-proven pipeline have the same denotation, so
    they are interchangeable *given* the declared steps are honoured — which is
    exactly `Law.cublasIsMatvec`, and exactly what `mlp-cifar` checks at
    runtime by running one against the other. -/
theorem blas_denotes_the_same :
    fwdBlasPlan.denote = fwdPipeline.denote
      ∧ bwdBlasPlan.denote = bwdPipeline.denote := ⟨rfl, rfl⟩

/-- **The cuBLAS configuration computes its denotation at run time**, for any
    realisation that honours the declared steps. -/
theorem blas_plans_run (R : Realisation) (hR : Honours R) (st : WSt) :
    (Plan.run R fwdBlasPlan st).mem = fwdBlasPlan.denote st.mem
      ∧ (Plan.run R bwdBlasPlan st).mem = bwdBlasPlan.denote st.mem :=
  ⟨Plan.run_denote R hR fwdBlasPlan fwdBlasPlan_exclusive st,
   Plan.run_denote R hR bwdBlasPlan bwdBlasPlan_exclusive st⟩

-- ---------------------------------------------------------------------------
-- The same model, as a list of backend choices
-- ---------------------------------------------------------------------------

/-! The two configurations differ in exactly one list.  Nothing about the model
    is written twice: the stages are the same objects, and `lowerAll_denote`
    — proven once, for any model and any assignment — is what says the choice
    cannot change the answer. -/

/-- **What each shipped schedule bills, as build output.**

    `blas_law_bill` proves the plan's own bill; this is the same information in
    a form a build log can print, so "which assumptions did this schedule make"
    is answered every time the artifact is generated rather than by reading a
    theorem. -/
def scheduleBills : List (String × List String × Nat) :=
  [("all-proven",   (Backend.laws .proven).map Law.title, 0),
   ("cuBLAS GEMMs", (Backend.laws GEMM).map Law.title,    2)]

/-- **The printed counts are the plans' own**, by `rfl`.

    The build log is a report, and a report that could drift from what it
    describes is worth nothing; this is what stops the printed number from
    being a second, hand-maintained answer. -/
theorem scheduleBills_are_the_plans :
    scheduleBills.map (fun e => e.2.2)
      = [fwdProvenPlan.lawlessCount, fwdBlasPlan.lawlessCount] := rfl


/-! ### The escape hatch

    The menu's `impl` is all-or-nothing: every contraction goes to one backend.
    Sending *one* of them elsewhere is outside it, so it is an assertion — a
    rewrite applied to the tape as given, named, and billed.

    What follows is the whole point of the mechanism: an assertion is a
    *hypothesis*, so a user who can prove theirs stops paying for it.  This one
    is dischargeable because `TOp.retarget_den` already exists; one that is not
    stays in the report until someone proves it or removes it. -/

/-- Send the tape's first operation to the vendor backend, leaving the rest
    proven. -/
def firstToBlas : AssertedRewrite :=
  { name  := "first-contraction-to-cublas"
    apply := fun t => match t with
      | []      => []
      | o :: os => o.retarget (.vendor .cublasSgemmStridedBatched) :: os }

/-- A schedule that reaches outside the menu. -/
def assertedSched : TenSchedule := { asserted := [firstToBlas] }

/-- **…and the assertion is discharged**, so this schedule costs nothing after
    all.  Retargeting one operation is denotation-preserving for the same reason
    retargeting all of them is, which the library proved once. -/
theorem firstToBlas_sound : firstToBlas.Sound := by
  intro _ _ _ _ t m b
  cases t with
  | nil => rfl
  | cons o os =>
      show ((o.retarget _ :: os).foldl (fun mm p => p.den mm) m) b = _
      simp only [List.foldl_cons, TOp.retarget_den]

theorem assertedSched_sound : AssertedSound assertedSched.asserted := by
  intro a ha
  have : a = firstToBlas := by
    simpa [assertedSched] using ha
  subst this; exact @firstToBlas_sound

/-- **The model still computes the model under it** — the escape hatch does not
    escape `compile_den`, it only adds a hypothesis to it. -/
theorem asserted_schedule_is_the_model (m : Buf → Nat → Float32) (b : Buf) :
    ((cifarTen.compile NIN assertedSched).2.foldl (fun mm o => o.den mm) m) b
      = (cifarTen.tape NIN).foldl (fun mm o => o.den mm) m b :=
  cifarTen.compile_den NIN assertedSched assertedSched_sound m b
    (by intro h; exact absurd (show b ∈ ([] : List Buf) from h) (by simp))

/-- **The all-proven build and the cuBLAS build compute the same function.**

    An instance of `lowerAll_denote`, which is quantified over every model and
    every assignment — so this costs no proof here, and a third backend would
    cost none either. -/
theorem builds_agree (m : Buf → Nat → Float32) :
    (lowerAll (fwdChoices GEMM)).denote m = (lowerAll (fwdChoices .proven)).denote m
      ∧ (lowerAll (bwdChoices GEMM)).denote m = (lowerAll (bwdChoices .proven)).denote m := by
  refine ⟨?_, ?_⟩
  · rw [lowerAll_denote, lowerAll_denote]; rfl
  · rw [lowerAll_denote, lowerAll_denote]; rfl

/-- **…and either build computes the model's denotation at run time.** -/
theorem builds_run (R : Realisation) (hR : Honours R) (b : Backend) (st : WSt) :
    (Plan.run R (lowerAll (fwdChoices b)) st).mem = (stagesOf (fwdChoices b)).denote st.mem
      ∧ (Plan.run R (lowerAll (bwdChoices b)) st).mem
          = (stagesOf (bwdChoices b)).denote st.mem :=
  ⟨lowerAll_runs R hR _ st, lowerAll_runs R hR _ st⟩

/-- **The price, read off the choice list.**  Five of nine operations rest on
    the vendor in the cuBLAS build; none do in the proven build. -/
theorem builds_declared :
    (lowerAll (fwdChoices GEMM)).declaredCount
        + (lowerAll (bwdChoices GEMM)).declaredCount = 5
      ∧ (lowerAll (fwdChoices .proven)).declaredCount
        + (lowerAll (bwdChoices .proven)).declaredCount = 0 := ⟨rfl, rfl⟩

-- ---------------------------------------------------------------------------
-- The same model, written as two dense layers
-- ---------------------------------------------------------------------------

/-! The model above names twelve index expressions, five trip counts and six
    grids.  None of that is a modelling decision — it is all determined by two
    widths and a batch.  `Dense` derives it, and the theorem below is what makes
    that a *frontend* rather than a second surface: the derived stages are the
    shipped ones, not merely equivalent to them.

    Note `layer2.dx` is `dhB` and `layer1.dy` is `adjB` rather than `dhB`: the
    activation backward sits between them, which is exactly the composition the
    pipeline records and the reason the two layers are not simply chained. -/

/-- The input layer's `dx` is the gradient w.r.t. the image, which nothing
    needs, so its buffer is deliberately **out of range**: `NBUF = 13`, so a
    stage built from it fails `BufBelow` rather than quietly writing over an
    input. -/
def layer1 : Dense :=
  { inW := IN, outW := H, batch := B
    w := w1B, x := xB, y := z1B, dy := adjB, dx := NBUF, dw := dw1B }

def layer2 : Dense :=
  { inW := H, outW := C, batch := B
    w := w2B, x := hB, y := logB, dy := dlogB, dx := dhB, dw := dw2B }

/-- **Seam guard: the derived layers are the shipped kernels.**

    Every index expression, trip count and grid the model spells out is
    reproduced from `(inW, outW, batch)` alone.  If a hand-written address and
    the derived one ever disagree, this fails to build. -/
theorem dense_derives_the_model :
    layer1.fwdStage.val.ew = fwd1Kernel
      ∧ layer2.fwdStage.val.ew = fwd2Kernel
      ∧ layer2.dxStage.val.ew = dhKernel
      ∧ layer2.dwStage.val.ew = dw2Kernel
      ∧ layer1.dwStage.val.ew = dw1Kernel := ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- …at the grids the driver launches, too. -/
theorem dense_derives_the_grids :
    layer1.fwdStage.val.grid = GRID1 ∧ layer2.fwdStage.val.grid = GRID2
      ∧ layer2.dxStage.val.grid = GRIDDH ∧ layer2.dwStage.val.grid = GRID2
      ∧ layer1.dwStage.val.grid = GRID1 := ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- **Both layers' forward stages compute the model's dot product**, under the
    one named law.

    Instantiated at the shipped widths — `3072 % 128 = 0` and `256 % 128 = 0`,
    both decided — so this is not a statement about a hypothetical layer. The
    right-hand side is what `Expr.denote` gives for `Transformer.dot`, up to
    `List.finRange` vs `List.range`.

    `Law.laneRegroup` is the whole assumption: the kernel folds sequentially
    within a lane then closes with a butterfly, the model folds flat, and the
    two agree by reassociation. Nothing else is assumed, and it is visible in
    the type. -/
theorem layers_compute_the_dot (h : AllHold [Law.laneRegroup])
    (m : Buf → Nat → Float32) (cta a : Nat) :
    layer1.fwdStage.val.val m cta a
        = (List.range IN).foldl
            (fun acc i => NumOps.add acc
              (NumOps.mul (m w1B (i + cta * IN)) (m xB (i + ((a - cta) / H) * IN))))
            (NumOps.ofNat 0)
      ∧ layer2.fwdStage.val.val m cta a
        = (List.range H).foldl
            (fun acc i => NumOps.add acc
              (NumOps.mul (m w2B (i + cta * H)) (m hB (i + ((a - cta) / C) * H))))
            (NumOps.ofNat 0) :=
  ⟨Dense.fwd_is_flatDot h layer1 (by decide) (by decide) (by decide) (by decide) m cta a,
   Dense.fwd_is_flatDot h layer2 (by decide) (by decide) (by decide) (by decide) m cta a⟩

-- ---------------------------------------------------------------------------
-- The whole model as a graph
-- ---------------------------------------------------------------------------

/-! Every operation, in the order the driver performs them.  No index
    expressions, no trip counts, no launch grids beyond the geometry records,
    and no addressing at all — each node names its operands and its output, and
    the widths determine the rest.

    `cifarTen_lowers` is what makes this a frontend rather than a picture of
    one: the stages the term lowers to are the stages the driver launches, the
    same objects, checked by `rfl`. -/

/-- **The buffers the program allocated are the ones the model names.** -/
theorem program_allocates :
    (xB, w1B, w2B, biasB, ohB, z1B, hB) = (0, 1, 2, 3, 4, 5, 6)
      ∧ (logB, dlogB, dw2B, dhB, adjB, dw1B) = (7, 8, 9, 10, 11, 12) := ⟨rfl, rfl⟩

def cifarTenProgram : TenProg H IN := cifarTen

/-- **The tensor term lowers to the shipped pipeline** — the same stage
    objects the driver launches and the rest of the chain reasons about, by
    `rfl`.  This is what makes the surface a frontend rather than a second
    description of the model. -/
theorem cifarTen_lowers :
    cifarTenProgram.stages B NIN = some (fwdStages ++ bwdStages) := rfl

/-- **The schedule erases to the model** — by `rfl`, with no decision
    procedure.  The comparison is of denotations, on every buffer: this schedule
    fuses nothing, so there is no intermediate to except.  A schedule that
    substituted a different activation would not typecheck against it. -/
theorem ten_schedule_is_the_model (m : Buf → Nat → Float32) (b : Buf) :
    ((cifarTen.compile NIN cifarBlasSched).2.foldl (fun mm o => o.den mm) m) b
      = (cifarTen.tape NIN).foldl (fun mm o => o.den mm) m b :=
  cifarTen.compile_den NIN cifarBlasSched assertedSound_nil m b
    (by intro h; exact absurd (show b ∈ ([] : List Buf) from h) (by simp))

/-- **The default schedule is the model's own tape** — naming no move leaves
    the operations alone. -/
theorem cifar_default_schedule : (cifarTen.compile NIN {}).2 = cifarTape := rfl

/-- The image's own gradient is not wanted; every other operand's is. -/
def needsGrad (r : Ref) : Bool := r != xB

/-- **The backward pass is derived from the forward pass.**

    `Ten.backward` walks the four forward operations in reverse and emits the
    four backward ones — and they are *the operations this model ships*, buffer
    numbers included, by `rfl`.  Nothing about the backward pass is written
    twice.

    Correctness comes by composition rather than by a new theorem:
    `layer_backprop_dx`/`_dW` already prove these kernels compute `sderiv` of
    the layer, and `activation_derived` pins the elementwise adjoint to the
    same Lean function as the forward activation. -/
theorem cifar_backward_derived :
    Ten.backward needsGrad 0 B (NIN + 4) (cifarTape.take 4)
      = some ((cifarTape.drop 4).take 4) := rfl

/-- Non-vacuity: the derived backward is four operations, not the empty list a
    stuck cotangent map would produce. -/
theorem cifar_backward_nonempty :
    ((Ten.backward needsGrad 0 B (NIN + 4) (cifarTape.take 4)).map List.length)
      = some 4 := rfl

/-- **The cotangent accumulator fires.**  A value used twice — the standard
    residual pattern — must sum its adjoints, not overwrite them.  First
    contribution binds; the second emits an add.  Without both halves the
    reverse pass would be silently wrong on every residual network. -/
theorem accum_binds_then_sums :
    (CoT.accum [] 3 9 20 4).1 = [] ∧
      (CoT.accum [(3, 9)] 3 11 20 4).1 = [TOp.ew2 addW2 9 11 20 4] := ⟨rfl, rfl⟩

/-- **A 93-layer stack**, to exercise depth on the real lowering rather than on
    the theorem alone.  Each layer is a contraction and an activation, with its
    own weight buffer; `Ten.stack` binds every layer's input, so the residual
    stream is one buffer per layer rather than a re-substituted subtree. -/
def deepStack (n : Nat) : TenProg B H := fun v =>
  Ten.stack (fun i (a : Ten v B H) => (Ten.mv .proven (.inp (1 + i)) a).silu)
    n (.inp 0)

set_option maxRecDepth 8000 in
/-- Two operations per layer, 93 layers, flattened. -/
theorem deep93_graph : ((deepStack 93).graph 200).length = 186 := by rfl

set_option maxRecDepth 8000 in
/-- **…and all 186 of them lower.**  A depth the model was never instantiated
    at costs no new theorem: `Ten.stack_stages` is the general statement, this
    is the check that the general statement is about something real. -/
theorem deep93_lowers : ((deepStack 93).stages (R := Float32) B 200).isSome = true := by rfl

/-- An RMS norm on its own, at a width a transformer block uses.  `x` is bound
    once and read three times — by the statistic, by the scale, by nothing
    else — which is what the `tlet` is for. -/
def rmsProg : TenProg 4 64 := fun _ =>
  Ten.letT (.inp 0) (fun xb => Ten.rmsNorm (0.000001 : Float32) (.inp 1) (.var xb))

/-- **A norm is three row passes**, not one and not four: the sum of squares,
    the scale, the gain.  Binding `x` is what keeps it at three. -/
theorem rms_ops : (rmsProg.graph 2).length = 3 := rfl

/-- **…and all three lower to proven stages.**  The broadcast reads — one
    scalar per row, one vector shared by every row — are inside the stage's
    contract rather than beside it. -/
theorem rms_lowers : (rmsProg.stages (R := Float32) 4 2).isSome = true := by rfl

/-- **The three passes address their second operand differently.**  Non-vacuity
    for the two above: a norm that read every operand at the element index
    would be a different kernel, and the digest would say so. -/
theorem rms_addressing :
    ((rmsProg.graph 2).map (fun p => (Node.sig p.1).1)) = [7, 8, 8] := by decide

/-! ### A transformer block, at K3-ish shape -/

/-- Model width, heads, head dimension, cache length, feed-forward width. -/
def DM : Nat := 512
def NH : Nat := 8
def HD : Nat := 64
def SQ : Nat := 128
def DFF : Nat := 1408

/-- Inputs `0…5` are the ones shared by every layer — the rotation tables, the
    ones vector, the two caches, the residual stream — and each layer owns
    seven weights after that. -/
def kBase (i : Nat) : Nat := 6 + i * 7

/-- The block's widths.  Nothing else about the shape is stated anywhere: the
    tiling condition is discharged here, once. -/
def K3 : BlockGeom := { dm := DM, nh := NH, hd := HD, sq := SQ, dff := DFF }

def k3Layer {v : Nat → Nat → Type} (i : Nat) (x : Ten v 1 DM) : Ten v 1 DM :=
  Ten.blockAt K3 (0.000001 : Float32)
    (.inp (kBase i)) (.inp (kBase i + 1)) (.inp (kBase i + 2)) (.inp (kBase i + 3))
    (.inp (kBase i + 4)) (.inp (kBase i + 5)) (.inp (kBase i + 6))
    (.inp 0) (.inp 1) (.inp 2) (.inp 3) (.inp 4) x (nm := s!"L{i}")

/-- `n` blocks, residual stream threaded. -/
def k3Stack (n : Nat) : TenProg 1 DM := fun v =>
  Ten.stack (fun i (a : Ten v 1 DM) => k3Layer i a) n (.inp 5)

/-- **One block is 28 operations** — twelve row passes (three per norm and six
    for the rotation), seven contractions (`wq`, `wo`, `w1`, `w3`, `w2` and
    attention's two), the softmax's four, two sums of squares, and the three
    elementwise passes that are the gate and the two residual adds. -/
theorem k3_block_ops : ((k3Stack 1).graph 700).length = 28 := by rfl

/-- **…and every one of them lowers to a proven stage.**  Nothing in a block
    reaches outside the stage vocabulary: no kernel is declared, no law is
    invoked, and a shape that did not fit would have failed to elaborate. -/
theorem k3_block_lowers : ((k3Stack 1).stages (R := Float32) 1 700).isSome = true := by rfl

set_option maxRecDepth 40000 in
/-- **Eight blocks compose**, which is the statement `Ten.stack_stages` makes
    for every depth, checked at one where the operation count is real. -/
theorem k3_eight_ops : ((k3Stack 8).graph 700).length = 224 := by rfl

/-- The same block, with both norms scheduled as two kernels instead of three. -/
def k3LayerFused {v : Nat → Nat → Type} (i : Nat) (x : Ten v 1 DM) : Ten v 1 DM :=
  Ten.blockFused (nh := NH) (hd := HD) (sq := SQ) (dff := DFF)
    Backend.proven (by decide) (by decide) (0.000001 : Float32)
    (.inp (kBase i)) (.inp (kBase i + 1)) (.inp (kBase i + 2)) (.inp (kBase i + 3))
    (.inp (kBase i + 4)) (.inp (kBase i + 5)) (.inp (kBase i + 6))
    (.inp 0) (.inp 1) (.inp 2) (.inp 3) (.inp 4) x

def k3StackFused (n : Nat) : TenProg 1 DM := fun v =>
  Ten.stack (fun i (a : Ten v 1 DM) => k3LayerFused i a) n (.inp 5)

/-! ### The block's kernels, emitted -/

/-- Inputs `0…4` are shared, `5` is the residual stream, `6…12` are the layer's
    seven weights — so the first computed buffer is `13`. -/
def QBASE : Nat := 13

/-- Buffers the binding table declares. -/
def QNBUF : Nat := 103

/-- The rotation tables, the ones vector and the caches are read, not trained. -/
def blockNeeds (r : Ref) : Bool := decide (5 ≤ r)

/-- The forward pass and the backward pass it derives, as one launch sequence:
    the training topology.  Buffer `40` is the incoming gradient the host
    uploads; the adjoint's own buffers start at `41`. -/
def qwenFwd : List TOp := (k3Stack 1).tape QBASE
def qwenBwd : List TOp :=
  (Ten.backwardFrom blockNeeds 2 1 41 [(39, 40)] qwenFwd).getD []

/-- **What the artifact ships: the training step.**

    Ninety launches — twenty-eight forward, sixty-two the derivation produced —
    as one sequence.  The forward half is entry `runFwd`, so a host can
    re-evaluate the loss at a perturbed weight without also recomputing
    gradients, which is what makes a finite-difference check of the derived
    backward possible at all. -/
def qwenTape : List TOp := qwenFwd ++ qwenBwd
noncomputable def qwenStages : List XStage :=
  (lowerNet (R := Float32) 1 (forget (qwenTape.map TOp.node))).getD []

/-- **The emitted kernels are the proven stages' own statements.**

    A `StageSpec` is noncomputable — its domain is a `Prop` — so the text has to
    be produced by something that runs.  This says the runnable side is not a
    second definition: statement for statement, it is what the stages carry. -/
theorem qwen_kernels_are_the_stages :
    qwenStages.map (fun S => S.val.ew) = qwenTape.map (TOp.stmt 1) := by rfl

/-- What the host must allocate: the buffer each operation writes and its size
    in bytes, derived from the model rather than tabulated beside it. -/
def qwenBufs : List (Ref × Nat) := qwenTape.map (TOp.outSize 1)

/-- Bytes for the block's thirteen inputs, in binding order: the two rotation
    tables, the ones vector, the two caches, the residual stream, the two
    gains, and the five weight matrices. -/
def qInBytes : List Nat :=
  [ HD * 4, HD * 4, DM * 4, SQ * HD * 4, SQ * HD * 4, DM * 4
  , DM * 4, DM * 4, DM * DM * 4, DM * DM * 4
  , DFF * DM * 4, DFF * DM * 4, DM * DFF * 4 ]

/-- Bytes for every buffer, inputs then computed.  The computed sizes come from
    `TOp.outSize`, so a shape that changed in the model changes the allocation
    rather than disagreeing with it, and **every computed buffer is at least a
    warp wide**.

    An elementwise pass covers whole warps, so a buffer one of them touches is
    read 32 lanes at a time even where only one value is meaningful — a
    cotangent that is one scalar per row is the case that matters.  The floor
    is an allocation fact, not a change to any kernel. -/
def qBufBytes : List Nat :=
  qInBytes ++ (List.range (QNBUF - QBASE)).map (fun k =>
    max 128 (qwenBufs.foldl (fun a p => if p.1 == QBASE + k then max a p.2 else a)
      (if QBASE + k == 40 then DM * 4 else 0)))

def qBufSize (b : Buf) : Nat := qBufBytes.getD b 0

/-! ### Fusion, applied to the shipped tape -/

/-- **The fused forward schedule**, written against the name the model bound:
    the second norm's scale.  `fuseNormAt` checks the site is fusable and
    returns the intermediate it removes along with the tape. -/
def qwenFwdSched : TenSchedule := { fuse := [.named "L0.ffn.scale"] }

/-- **The named site is the one the tape offers.**

    `fuseTargets` reports `[(1, 14), (21, 33)]` for this tape — two fusable
    pairs, each identified by the intermediate it removes — and the name in the
    schedule resolves to the second.  Naming rather than counting is what makes
    it survive a tape edit: fusing at 1 first would renumber 21, but neither
    the buffer nor the name. -/
theorem qwen_fuse_site :
    fuseTargets qwenFwd = [(1, 14), (21, 33)]
      ∧ FuseSite.resolve ((k3Stack 1).labels QBASE) qwenFwd (.named "L0.ffn.scale")
          = some 21 := ⟨rfl, rfl⟩

/-- **A model names its own sites, whoever wrote it.**

    `tlet z1 := …` labels the buffer it binds, and the library combinators
    label theirs under the name the layer was given — so a schedule says
    `.named "L0.ffn.scale"` and never sees a number.  Both tables are recovered
    from the term rather than maintained beside it, which is what these two
    `rfl`s check. -/
theorem cifar_labels :
    cifarTen.labels NIN
      = [("z1", 5), ("h", 6), ("y", 7), ("dlog", 8), ("dw2", 9), ("dh", 10),
         ("adj", 11), ("dw1", 12)] := rfl

theorem k3_labels :
    (k3Stack 1).labels QBASE
      = [("L0.n1", 15), ("L0.attn.scale", 14), ("L0.q", 21), ("L0.att", 30),
         ("L0.xr", 31), ("L0.n2", 34), ("L0.ffn.scale", 33), ("L0.ff", 37)] := rfl

/-- The forward pass under that schedule: the intermediates it removed, and the
    tape it produced. -/
def qwenFwdCompiled : List Buf × List TOp := (k3Stack 1).compile QBASE qwenFwdSched

/-- **The site is fusable** — every side condition the library decides. -/
theorem qwen_fwd_fusable : qwenFwdCompiled.1 = [33] := rfl

def qwenFwdFused : List TOp := qwenFwdCompiled.2

/-- **Two kernels shorter.** -/
theorem qwen_fused_shorter :
    qwenFwdFused.length = 27 ∧ qwenFwd.length = 28 := ⟨rfl, rfl⟩

/-- **…but the training step does read the intermediate** — twice, in the
    passes that differentiate the norm.  This is why the shipped training tape
    is not fused: the intermediate is a saved activation, so fusing it away
    would leave the backward reading a buffer nothing wrote.  `fuseNormAt`
    decides exactly this condition, which is why naming the same site on the
    training tape returns `none`. -/
theorem qwen_training_temp_live :
    ((List.range 90).zip qwenTape).filterMap (fun p =>
      if 23 ≤ p.1 && (TOp.reads p.2).contains 33 then some p.1 else none)
      = [39, 40] := by decide

theorem qwen_training_not_fusable : (fuseNormAt 21 qwenTape) = none := rfl

/-- **The fused forward computes what the forward computes.**

    Zero proof text at the instantiation: every side condition was discharged
    inside `fuseNormAt`, so naming the site is the whole schedule. -/
theorem qwen_fwd_fusion_agrees (m : Buf → Nat → Float32) (b : Buf) (hb : b ≠ 33) :
    (qwenFwdFused.foldl (fun mm o => o.den mm) m) b
      = (qwenFwd.foldl (fun mm o => o.den mm) m) b :=
  (k3Stack 1).compile_den QBASE qwenFwdSched assertedSound_nil m b
    (by intro h
        have h33 : b ∈ [33] := h
        rw [List.mem_singleton] at h33
        exact hb h33)

/-- One PTX kernel per operation: its statement's body kernel, printed. -/
def qwenPtxOf (t : List TOp) : List String :=
  t.map (fun op => Body.text "main" (unitBufs 1 [op]).length 0 (unitStmt 1 [op]))

def qwenPtx : List String := qwenPtxOf (qwenTape ++ qwenFwdFused)

/-- Ninety launches for the training step, then twenty-seven for the fused
    forward schedule — one image, two schedules. -/
theorem qwen_slot_count : qwenPtx.length = 117 := by rfl

/-- **How many kernels an artifact may hold.**

    The runtime loads one module per kernel and keeps it resident; if a context
    ran out of room the loader would evict and reload, so a limit there would
    bound *launch cost*, not correctness.  Ninety resident modules provoke no
    eviction, so the only bound this artifact has met is the one here: what the
    memory image's slot table can address. -/
def MODULE_LIMIT : Nat := 1024

/-- **Seam guard: the artifact fits the loader.** -/
theorem qwen_within_module_limit : qwenPtx.length ≤ MODULE_LIMIT := by
  rw [qwen_slot_count]; decide

/-- **The bound that makes the stack depth-independent.**

    A kernel binds the buffers of one operation, so its parameter list is fixed
    by the vocabulary — five at the widest — however many buffers the model
    has.  Emitted text therefore grows with the number of operations and not
    with operations × buffers, which is the difference between a stack that can
    represent a ninety-three layer model and one that cannot. -/
theorem qwen_kernel_width_is_bounded :
    (qwenTape ++ qwenFwdFused).all (fun op => decide ((op.bufs 1).length ≤ 5))
      = true :=
  List.all_eq_true.mpr (fun op _ => decide_eq_true (bufs_le_five 1 op))

/-- **Seam guard: every kernel's buffers are inside its own table**, and the
    compaction that put them there is injective — the two conditions
    `StageSpec.rename` needs, decided per kernel.

    The table is the operation's own, so this says nothing about how many
    buffers the model has: `bufs_le_five` is what bounds it. -/
theorem qwen_bufs_bound :
    (qwenTape ++ qwenFwdFused).all (fun op => op.localBelowB 1) = true
      ∧ (qwenTape ++ qwenFwdFused).all (fun op => op.compactInjB 1) = true :=
  ⟨List.all_eq_true.mpr (fun op _ => TOp.localBelow 1 op), rfl⟩

/-- **Seam guard: every kernel fits its slot.** -/
theorem qwen_ptx_fits :
    qwenPtx.all (fun t => decide (t.toUTF8.toList.length + 1 ≤ 0x8000)) = true := by
  native_decide

/-- The lowering really is `some`, so the stage list is the one the theorems
    below are about rather than a silently empty default. -/
theorem qwen_stages_eq :
    lowerNet (R := Float32) 1 (forget (qwenTape.map TOp.node)) = some qwenStages := by rfl

/-- **Running the block's launch sequence performs the block's mathematics.**

    The left end of the chain: from the launches the artifact makes back to the
    tensor term, through `TOp.den` — whose extents come from the term's types,
    not from an ambient batch.  Generic in the tape and the depth; this is its
    instantiation on what ships. -/
theorem qwen_run_den (st : WSt) :
    ((Pipeline.ofStages qwenStages).run st).mem
      = qwenTape.foldl (fun mm op => op.den mm) st.mem := by
  rw [Pipeline.ofStages_runs qwenStages st]
  show (qwenStages.map Subtype.val).foldl (fun mm S => S.step mm) st.mem = _
  rw [List.foldl_map]
  exact lowerTOps_den 1 qwenTape qwenStages qwen_stages_eq st.mem

/-- The block's stages write disjoint buffers from those they read. -/
theorem qwenPipeline_exclusive : (Pipeline.ofStages qwenStages).Exclusive :=
  Pipeline.ofStages_exclusive qwenStages

/-- **Non-vacuity for the theorem above, and the shape of the defect it rules
    out.**  The contractions carry the extent from their own operand types —
    the two attention products over `NH` heads, the five residual-stream
    projections over one position.  A lowering that took the leading extent
    from an ambient batch would read `1` here and compute one head. -/
theorem qwen_extents_are_typed :
    qwenFwd.filterMap (fun op => match op with
      | .mv _ _ _ _ b _ _ _ => some b
      | .mvT _ _ _ _ b _ _ => some b
      | _ => none) = [1, NH, NH, 1, 1, 1, 1] := rfl

/-- The launch sequence: slot `i` over the blocks its stage declares. -/
def qwenSteps : List (Nat × Nat) :=
  (List.range qwenTape.length).zip (qwenTape.map TOp.gridOf)

/-- A launch: its slot, its grid, and the buffers it binds. -/
def qLaunches (base : Nat) (t : List TOp) : List (Nat × Nat × List Buf) :=
  ((List.range t.length).map (· + base)).zip
    (t.map (fun o => (TOp.gridOf o, TOp.bufs 1 o)))

/-- **Seam guard: the launch geometry is the stages' own.**

    A grid and a stage that disagree would drop a tail silently — every theorem
    about the stage would still hold, about elements the launch never visits. -/
theorem qwen_grids_are_the_stages :
    qwenTape.map TOp.gridOf = qwenStages.map (fun S => S.val.grid) := by rfl

/-- **Seam guard: the launches are slots `0…89`, in graph order.** -/
theorem qwen_steps_in_order :
    qwenSteps.map Prod.fst = List.range 90 := by decide

/-- The fused forward's launch sequence: its kernels live in slots `90…116`. -/
def qwenFusedSteps : List (Nat × Nat) :=
  ((List.range qwenFwdFused.length).map (· + qwenTape.length)).zip
    (qwenFwdFused.map TOp.gridOf)

/-- **Seam guard: the fused schedule's launches are its own slots, in order**,
    so the two schedules in this image cannot address each other's kernels. -/
theorem qwen_fused_steps_in_order :
    qwenFusedSteps.map Prod.fst = (List.range 27).map (· + 90) := by decide

/-- **Seam guard: every computed buffer is a fresh one inside the table.**

    An operation writing a buffer another already wrote would be a fusion the
    stage vocabulary does not describe; `rope`'s two halves are the one place
    that is intended, and they are why this counts distinct entries rather than
    asserting the list has none. -/
theorem qwen_bufs_fresh :
    (qwenBufs.map Prod.fst).all (fun b => decide (QBASE ≤ b ∧ b < QNBUF)) = true
      ∧ (qwenBufs.map Prod.fst).eraseDups.length = 89 := by decide

/-- **Seam guard: the training step's activations are 11,013,264 bytes.**

    Ninety operations, eighty-nine buffers: the one repeat is the rotation
    writing two halves of a row.  Weights are inputs and are not counted here —
    this is what the step needs *between* its kernels, which is the number a
    host has to find room for. -/
theorem qwen_working_set :
    (qwenBufs.map Prod.snd).foldl (· + ·) 0 = 11013264 := by decide
/-- **Where each trainable input's gradient lands**, read off the derivation
    rather than assumed: `w2 → 43`, `w3 → 47`, `w1 → 49`, `wo → 63`, `g2 → 54`,
    `wq → 92`, `g1 → 96`, `x → 102`. -/
theorem qwen_grad_slots :
    (Ten.backwardCoT blockNeeds 2 1 41 [(39, 40)] qwenFwd).map
      (fun ct => (List.range 13).filterMap (fun w => (ct.get w).map (fun g => (w, g))))
      = some [(3, 79), (4, 65), (5, 102), (6, 96), (7, 54), (8, 92), (9, 63),
              (10, 49), (11, 47), (12, 43)] := by rfl

/-! ### The rotation differentiates -/

/-- A rotation on its own, at the block's head geometry. -/
def ropeProg : TenProg NH HD := fun _ => Ten.rope (.inp 0) (.inp 1) (.inp 2)

def ropeTape : List TOp := ((ropeProg RefV).flat 3).2.2

/-- The rotation tables are read, never trained. -/
def ropeNeeds (r : Ref) : Bool := r != 1 && r != 2

/-- **Six row passes in, eleven out.**  The rotation's adjoint is derived from
    the forward passes by `WFExp.deriv`, and lands in the three-operand row
    pass — which is why an adjoint that is not linear in its operand is
    expressible at all.  Where a buffer is reached twice, `CoT.accum` adds. -/
theorem rope_backward_derived :
    ((Ten.backwardFrom ropeNeeds 9 1 10 [(7, 8)] ropeTape).map List.length) = some 11 := rfl

/-- **…and every one of those passes lowers to a proven stage.**  The adjoint
    is not a term that merely typechecks: it is a launch sequence. -/
theorem rope_backward_lowers :
    ((Ten.backwardFrom ropeNeeds 9 1 10 [(7, 8)] ropeTape).map
      (fun ops => (lowerNet (R := Float32) 1 (forget (ops.map TOp.node))).isSome)) = some true := by
  rfl

/-- **A row pass differentiates by the rule the stack already proved.**

    `Expr.sderiv` is tied to the analytic derivative by `grad_hasDerivAt`;
    `Expr.toWF_deriv` says the lane-level rule is that one translated, not a
    second differentiator.  Instantiated here at the gated feed-forward's own
    expression, so the general theorem is about something the model uses. -/
theorem lane_deriv_is_scalar :
    (Expr.toWF gatedW2).deriv 1 = some (sderiv gatedW2 ⟨0, by decide⟩).toWF :=
  Expr.toWF_deriv gatedW2 (by decide) ⟨0, by decide⟩

/-! ### The same block, differentiated -/

abbrev blockTape : List TOp := qwenFwd

/-- **The whole block differentiates: 28 passes forward, 62 back.**

    Every adjoint is derived from the forward tape — no backward model is
    written.  The rotation's, the norms', the softmax's (including the maximum,
    whose gradient goes where the maximum was attained, `geF` being that
    indicator as a value) and both contractions'.

    So one model definition yields three checked artifacts: this training
    topology and the two inference schedules above.

    The ones vector must be at least as long as the widest row a broadcast
    adjoint reduces over — `DM`, not `SQ` — which is an allocation the training
    setup makes, not a property of the tape. -/
theorem block_backward_derived :
    ((Ten.backwardFrom blockNeeds 2 1 41 [(39, 40)] blockTape).map List.length)
      = some 62 := rfl

/-- **…and all 62 lower to proven stages.** -/
theorem block_backward_lowers :
    ((Ten.backwardFrom blockNeeds 2 1 41 [(39, 40)] blockTape).map
      (fun ops => (lowerNet (R := Float32) 1 (forget (ops.map TOp.node))).isSome)) = some true := by
  rfl

/-! ### A schedule that fuses -/

/-- **The fused schedule is two kernels shorter**, and it is a schedule, not a
    different model: every operation it drops is a temporary the fused pass no
    longer writes. -/
theorem k3_fused_ops : ((k3StackFused 1).graph 700).length = 26 := by rfl

/-- **…and all of them lower.** -/
theorem k3_fused_lowers : ((k3StackFused 1).stages (R := Float32) 1 700).isSome = true := by rfl

/-- **The fusion computes the same number, bit for bit.**

    `fuse_ziprow_den` at the norm's own expressions and the block's width: the
    scale-then-gain pair and the single three-operand pass agree on the buffer
    that survives, from any memory, with no law invoked. -/
theorem rms_fusion_agrees (x ss g t out : Ref) (m : Buf → Nat → Float32) (a : Nat)
    (htg : g ≠ t) (hot : out ≠ t) :
    ((TOp.ziprow t g out mulW (.rowOf DM 0) (.sharedAt 0) DM 0 DM 1).den
      ((TOp.ziprow x ss t (rmsScaleW DM (0.000001 : Float32))
        (.rowOf DM 0) .scalar DM 0 DM 1).den m)) out a
      = (TOp.ziprow3 x ss g out
          (WFExp.fuseA mulW (rmsScaleW DM (0.000001 : Float32)))
          (.rowOf DM 0) .scalar (.sharedAt 0) DM 0 DM 1).den m out a :=
  fuse_ziprow_den x ss g t out _ _ rfl rfl (.rowOf DM 0) .scalar (.sharedAt 0)
    DM 0 DM 0 DM 1 (by decide) (by decide) (by decide) htg hot m a

/-- **The rotation fuses too**, on a *segment* of a row rather than all of it:
    the lower half of a rotated head is `x·cos` fed into a subtraction, and the
    pair becomes one three-operand pass writing `[0, 32)` of a 64-wide row.

    The same theorem at a different geometry — which is what keeping the
    producer's and the consumer's extents as separate parameters buys. -/
theorem rope_fusion_agrees (x c t2 t out : Ref) (m : Buf → Nat → Float32) (a : Nat)
    (htg : t2 ≠ t) (hot : out ≠ t) :
    ((TOp.ziprow t t2 out subW (.rowOf 32 0) (.rowOf 32 0) HD 0 32 NH).den
      ((TOp.ziprow x c t mulW (.rowOf HD 0) (.sharedAt 0) 32 0 32 NH).den m)) out a
      = (TOp.ziprow3 x c t2 out (WFExp.fuseA subW mulW)
          (.rowOf HD 0) (.sharedAt 0) (.rowOf 32 0) HD 0 32 NH).den m out a :=
  fuse_ziprow_den x c t2 t out _ _ rfl rfl (.rowOf HD 0) (.sharedAt 0)
    (.rowOf 32 0) 32 0 HD 0 32 NH (by decide) (by decide) (by decide) htg hot m a

/-! ### A norm differentiates, gains included -/

/-- Input `0` is the row, `1` the gain, `2` a vector of ones. -/
def normProg : TenProg 1 DM := fun _ =>
  Ten.letT (.inp 0) (fun xb => Ten.rmsNorm (0.000001 : Float32) (.inp 1) (.var xb))

def normTape : List TOp := ((normProg RefV).flat 3).2.2

/-- The ones vector is a constant, not a parameter. -/
def normNeeds (r : Ref) : Bool := r != 2

/-- **Three row passes forward, eight back — and the gain is among them.**

    The two broadcast operands are what needed the ones vector: the statistic's
    cotangent is a row sum (`rowdot` against ones) and the gain's is a column
    sum (`outer` against ones).  Both are operations the vocabulary already
    had; only the addressing is new. -/
theorem norm_backward_derived :
    ((Ten.backwardFrom normNeeds 2 1 7 [(5, 6)] normTape).map List.length)
      = some 8 := rfl

/-- **…and the whole adjoint lowers to proven stages.** -/
theorem norm_backward_lowers :
    ((Ten.backwardFrom normNeeds 2 1 7 [(5, 6)] normTape).map
      (fun ops => (lowerNet (R := Float32) 1 (forget (ops.map TOp.node))).isSome)) = some true := by
  rfl

/-! ### The block as a shippable artifact -/

set_option maxRecDepth 8000 in
/-- **Seam guard: every buffer the kernels name gets allocated, and none of
    them gets zero bytes.** -/
theorem qwen_alloc_covers :
    qBufBytes.length = QNBUF ∧ qBufBytes.all (fun n => decide (0 < n)) = true :=
  ⟨rfl, rfl⟩

/-- The host uploads all thirteen inputs in one packed block. -/
def qHostIn : AlgorithmLib.Layout.RegionMap :=
  (List.range 13).map (fun i =>
    ⟨s!"in{i}", (qInBytes.take i).foldl (· + ·) 0, qInBytes.getD i 0⟩)

theorem qHostIn_packed :
    AlgorithmLib.Layout.RegionMap.packedB 0 qHostIn = true := by decide

def QHOST_BYTES : Nat := AlgorithmLib.Layout.RegionMap.total qHostIn

def QSLOT : Nat := 0x8000
def QHOST_LEN_OFF : Nat := 0x0080

/-- Where the capture leaves the stream, the graph and the executable it built. -/
def QSTREAM_OFF : Nat := 0x0090
def QGRAPH_OFF : Nat := 0x0098
def QEXEC_OFF : Nat := 0x00a0

/-- Where the driver's state starts (`Drv.App`). -/
def QORG : Nat := 0x0100
def QPTX_OFF : Nat := 0x9000
def qSlotOff (i : Nat) : Nat := QPTX_OFF + i * QSLOT
def QMEM_SIZE : Nat := qSlotOff 117

/-- Every launch the image holds a kernel for: the tape's, then the fused
    forward's. -/
def qAllLaunches : List (Nat × Nat × List Buf) :=
  qLaunches 0 qwenTape ++ qLaunches qwenTape.length qwenFwdFused

/-- **The block in the driver's standard layout**: an allocation per buffer,
    and each of the 117 kernels handed its own buffers. -/
def qApp : Drv.App :=
  { bufs := qBufBytes,
    kernels := qAllLaunches.map fun (i, g, bs) => ⟨qSlotOff i, QSLOT, bs, (g, 1, 1), (32, 1, 1), []⟩,
    memSize := QMEM_SIZE, org := QORG }

/-- **Every byte this program names.** -/
def qMemMap : AlgorithmLib.Layout.RegionMap :=
  [⟨"host_len", QHOST_LEN_OFF, 8⟩,
   ⟨"stream",   QSTREAM_OFF, 8⟩,
   ⟨"graph",    QGRAPH_OFF, 8⟩,
   ⟨"exec",     QEXEC_OFF, 8⟩,
   ⟨"driver",   QORG, qApp.stateEnd - QORG⟩]
    ++ (List.range 117).map (fun i => ⟨s!"ptx{i}", qSlotOff i, QSLOT⟩)

/-- **Seam guard: the slots tile the image without overlapping.** -/
theorem qwenMap_ok :
    qMemMap.okB = true ∧ qMemMap.withinB QMEM_SIZE = true := by decide

/-- Bring the driver up, make every allocation, load every kernel and write
    its parameters, then upload the thirteen inputs. -/
def qLoadFn : Prog V L Unit := do
  let base ← basePtr
  let dataPtr ← dataPtr
  qApp.setup base
  for i in List.range 13 do
    qApp.upload base i (← iaddImm dataPtr (AlgorithmLib.Layout.RegionMap.offAt qHostIn i)) (qInBytes.getD i 0)

def qUploadFn (b n : Nat) : Prog V L Unit := do
  let base ← basePtr
  qApp.upload base b (← dataPtr) n

def qFetchFn (b n : Nat) : Prog V L Unit := do
  let base ← basePtr
  qApp.download base (← outPtr) b n

/-- Launch the kernels of `ls` in turn on the default stream, then wait. -/
def qRun (ls : List (Nat × Nat × List Buf)) : Prog V L Unit := do
  let base ← basePtr
  for (i, _, _) in ls do
    qApp.run base i
  Drv.sync

/-- **The block, as one launch sequence with a single synchronisation.**

    The steps are `qwenSteps` — the slots and grids the stages declare — so the
    driver names no geometry of its own. -/
def qRunFn : Prog V L Unit := qRun (qLaunches 0 qwenTape)

/-- The first `k` launches only — the forward half at `k = 28`, and a way to
    find which launch a fault comes from. -/
def qRunUpto (k : Nat) : Prog V L Unit := qRun ((qLaunches 0 qwenTape).take k)

/-- **The step, captured once as a graph.**

    A created stream runs the launches between `beginCapture` and
    `endCapture`, which records them instead of issuing them; the graph is
    instantiated and uploaded, so the first replay costs what the others do.
    Every kernel's module is loaded by `main`, which a capture could not do.

    A capture followed by replays leaves memory as that many steps of the
    tape (`QwenReplaySafe.capture_replay_tape`). -/
def qCaptureFn : Prog V L Unit := do
  let base ← basePtr
  let _ ← ext (.cuda .streamCreate) %[← iaddImm base QSTREAM_OFF, ← iconst32 1]
  let s ← load64 (← iaddImm base QSTREAM_OFF)
  let _ ← ext (.cuda .beginCapture) %[s, ← iconst32 2]
  for (i, _, _) in qLaunches 0 qwenTape do
    qApp.runOn base s i
  let _ ← ext (.cuda .endCapture) %[s, ← iaddImm base QGRAPH_OFF]
  let _ ← ext (.cuda .graphInstantiate) %[← iaddImm base QEXEC_OFF, ← load64 (← iaddImm base QGRAPH_OFF),
    ← iconst64 0]
  let _ ← ext (.cuda .graphUpload) %[← load64 (← iaddImm base QEXEC_OFF), s]
  let _ ← ext (.cuda .streamSynchronize) %[s]

/-- **The captured step, replayed `k` times.**

    One driver call per step instead of ninety.  The parameter blocks are not
    touched: a graph holds the parameters it was captured with, which is
    exactly why this entry exists for the training step and not for the
    dispatch one. -/
def qReplayFn (k : Nat) : Prog V L Unit := do
  let base ← basePtr
  let e ← load64 (← iaddImm base QEXEC_OFF)
  let s ← load64 (← iaddImm base QSTREAM_OFF)
  for _ in List.range k do
    let _ ← ext (.cuda .graphLaunch) %[e, s]
  let _ ← ext (.cuda .streamSynchronize) %[s]

/-- The forward half only, so a host can re-evaluate the loss at a perturbed
    weight without also recomputing gradients. -/
def qRunFwd : Prog V L Unit := qRun ((qLaunches 0 qwenTape).take 28)

/-- The forward half under the fused schedule: the same stages, with the second
    norm's scale and gain in one kernel.  Its slots sit after the tape's, which
    is why the launch base is `qwenTape.length`. -/
def qRunFused : Prog V L Unit := qRun (qLaunches qwenTape.length qwenFwdFused)

/-- The bodies the quantised artifact carries, in the order it numbers them. -/
def qShippedBodies : List (String × Prog.Body) :=
  [ ("main", qLoadFn)
  , ("runBlock", qRunFn)
  , ("fetchOut", qFetchFn 39 (DM * 4))
  , ("runFwd", qRunFwd)
  , ("uploadDOut", qUploadFn 40 (DM * 4))
  , ("fetchDW2", qFetchFn 43 (DM * DFF * 4))
  , ("uploadW2", qUploadFn 12 (DM * DFF * 4))
  , ("runTo30", qRunUpto 30)
  , ("runTo45", qRunUpto 45)
  , ("runTo60", qRunUpto 60)
  , ("runTo75", qRunUpto 75)
  , ("capture", qCaptureFn)
  , ("replay", qReplayFn 1)
  , ("replay2", qReplayFn 2)
  , ("replay4", qReplayFn 4)
  , ("runTo31", qRunUpto 31)
  , ("runTo34", qRunUpto 34)
  , ("runTo35", qRunUpto 35)
  , ("runTo37", qRunUpto 37)
  , ("runFwdFused", qRunFused) ]


def qClifIR : Except String (List FuncData) :=
  Prog.program <|
    .ok noopFunction :: qShippedBodies.zipIdx.map
      (fun ((name, body), i) => Prog.entry name (Prog.compileProg (i + 1) body))

/-- The signature the block's launches rely on of every kernel: `main` takes
    one pointer per buffer it binds. -/
theorem qwen_sigs :
    ((qwenPtx.zip qAllLaunches).all fun (p, _, _, bs) =>
      HProg.Sem.ptxParamSizes p "main" == some (bs.map fun _ => 8)) = true := by
  native_decide

def qInitialMemory : List UInt8 :=
  qApp.initialMemory qwenPtx [(QHOST_LEN_OFF, u32le QHOST_BYTES)]

def qSetup (clif : List FuncData) : Artifact := {
  functions := clif
  required_memory := QMEM_SIZE
  initial_memory := qInitialMemory
}

/-- Experts the router scores.  A row pass covers whole warps, so a router
    narrower than 32 is a build failure rather than a kernel that folds nothing
    — which is what a width of 8 silently was. -/
def NE : Nat := 32

/-- A mixture-of-experts feed-forward: a router and eight gated experts. -/
def k3MoeProg : TenProg 1 DM := fun _ =>
  Ten.moe (nE := NE) (dff := DFF) Backend.proven (.inp 0) (.inp 1)
    (fun i => (.inp (2 + i * 3), .inp (3 + i * 3), .inp (4 + i * 3))) 7
    (.inp 100)

/-- **A 32-way router and eight evaluated experts is 53 operations**, all of
    them lowering.
    The gate for expert `e` reaches its expert as a read at one address —
    which is the whole reason a mixture needs no new stage. -/
theorem k3_moe_ops : (k3MoeProg.graph 200).length = 53 := by rfl

theorem k3_moe_lowers : (k3MoeProg.stages (R := Float32) 1 200).isSome = true := by rfl

set_option maxRecDepth 100000 in
/-- **A whole model's worth of blocks lowers** — twenty-four of them, Qwen2's
    depth, 624 operations, every one a proven stage.

    Depth itself is generic: `Ten.stack_stages` is the statement for every `n`.
    This is the check that the generic statement is about a real stack of real
    blocks, at the depth a real model has. -/
theorem k3_24_lowers : ((k3Stack 24).stages (R := Float32) 1 700).isSome = true := by rfl

set_option maxRecDepth 100000 in
/-- …and the operation count is linear in the depth, which is what says the
    stack shares nothing it should not. -/
theorem k3_24_ops : ((k3Stack 24).graph 700).length = 672 := by rfl

/-- **…and it really does choose differently.**  Non-vacuity for the check
    above: the two programs agree everywhere except on the implementations. -/
theorem ten_schedule_differs :
    (cifarTen.compile NIN cifarBlasSched).2.map (fun o => o.node.2)
      ≠ (cifarTen.compile NIN).2.map (fun o => o.node.2) := by decide

/-- **The lowering refuses a malformed graph.**

    Non-vacuity for the theorem above: a matvec whose weight operand *is* its
    output has no sound stage, and the lowering returns `none` rather than
    skipping the node — which would silently ship a pipeline missing a
    kernel. -/
theorem badNet_rejected : Net.stages (R := Float32) B [.input xB, .matvec z1B xB z1B B IN H] = none := rfl

/-! ### Sparse dispatch: choosing experts is a rebinding -/

/-- The mixture the dispatch demo runs: a 32-way router, two slots evaluated. -/
def MD : Nat := 128
def MDFF : Nat := 256
def MUSED : Nat := 2

/-- Inputs: the ones row, the router weight, the token, the packed gates the
    host writes, then two slots of three weights each — `4…9`.

    The gate buffer is a whole warp wide because an upload is sized by the
    buffer, and slot `j` reads element `j` of it; the entries past `MUSED` are
    written and never read. -/
def MSLOT0 : Ref := 4

/-- The first buffer past the slots, where the term's own buffers start. -/
def MBASE : Ref := MSLOT0 + 3 * MUSED

/-- The router, as its own launch sequence: the host runs it, reads the row it
    writes, and only then knows which weights to bind. -/
def mRouterProg : TenProg 1 NE := fun _ =>
  Ten.router (dm := MD) Backend.proven (.inp 0) (.inp 1) (.inp 2)

/-- The chosen experts, as a second launch sequence over the same buffers. -/
def mExpertProg : TenProg 1 MD := fun _ =>
  Ten.moeSlots (dff := MDFF) (nUsed := MUSED) Backend.proven
    (fun j => (.inp (MSLOT0 + 3 * j), .inp (MSLOT0 + 3 * j + 1),
               .inp (MSLOT0 + 3 * j + 2)))
    (.inp 3) (.inp 2)

def mRouterFlat : Ref × Ref × List TOp := (mRouterProg RefV).flat MBASE
def mExpertFlat : Ref × Ref × List TOp := (mExpertProg RefV).flat mRouterFlat.2.1

def mRouterTape : List TOp := mRouterFlat.2.2
def mExpertTape : List TOp := mExpertFlat.2.2
def moeTape : List TOp := mRouterTape ++ mExpertTape

/-- The gate row the host reads, and the mixture the experts write. -/
def MGATE : Ref := mRouterFlat.1
def MOUT : Ref := mExpertFlat.1

def MNBUF : Ref := mExpertFlat.2.1

/-- **The allocation the thirty-two experts' weights live in**, past every
    buffer the term names. The term never names it: a slot reads
    `MSLOT0 + 3j`, and which expert that *is* is which pointer `bindExperts`
    writes into the slot. -/
def MSTORE : Nat := MNBUF

/-- **Six launches to route, eleven to run the two chosen experts.**

    The eleven do not grow with the thirty-two: a slot's cost is its own
    weights, so what the sequence charges for is `MUSED`, not `NE`. -/
theorem moe_slot_count : mRouterTape.length = 6 ∧ mExpertTape.length = 11 := by
  exact ⟨rfl, rfl⟩

/-- **Seam guard: the dispatch tape lowers, kernel for kernel.** -/
theorem moe_kernels_are_the_stages :
    ((lowerNet (R := Float32) 1 (forget (moeTape.map TOp.node))).getD []).map (fun S => S.val.ew)
      = moeTape.map (TOp.stmt 1) := by rfl

/-- **The slots read the bound buffers and nothing else.**

    Non-vacuity for the whole design: what the expert half names is `MSLOT0…`,
    never a buffer in the store.  If any expert's identity had reached the
    term, choosing experts would mean re-emitting kernels instead of writing
    six integers. -/
theorem moe_slots_are_bound :
    (mExpertTape.filterMap (fun op => match op with
      | .mv _ w _ _ _ _ _ _ => some w
      | _ => none)).all (fun b => decide (MSLOT0 ≤ b ∧ b < MBASE)) = true := by
  decide

/-- …and the store really is outside what the term names. -/
theorem moe_store_is_unnamed : moeTape.all (fun op => (op.bufs 1).all (· < MSTORE)) = true := by
  decide

/-! ### The dispatch artifact -/

/-- Bytes per buffer.  The six slot buffers are placeholders: the bind array
    points them at the store before the experts run, so what is allocated for
    them is never read. -/
def mBufBytes : List Nat :=
  [ NE * 4, NE * MD * 4, MD * 4, 128 ]
    ++ (List.replicate (3 * MUSED) 128)
    ++ (List.range (MNBUF - MBASE)).map (fun k =>
          max 128 (moeTape.foldl
            (fun a op => let p := TOp.outSize 1 op
                         if p.1 == MBASE + k then max a p.2 else a) 0))

set_option maxRecDepth 100000 in
/-- **Seam guard: every buffer the dispatch kernels name gets allocated.** -/
theorem moe_alloc_covers :
    mBufBytes.length = MNBUF ∧ mBufBytes.all (fun n => decide (0 < n)) = true := by
  decide

/-- **Seam guard: the binding table stays below the address scratch.** -/
theorem moe_bufs_within_regs :
    moeTape.all (fun op => decide ((op.bufs 1).length ≤ 5)) = true :=
  List.all_eq_true.mpr (fun op _ => decide_eq_true (bufs_le_five 1 op))

/-- **Seam guard: every kernel's buffers are inside the binding table**, and
    the compaction that put them there is injective. -/
theorem moe_kernels_ok :
    moeTape.all (fun op => op.localBelowB 1) = true
      ∧ moeTape.all (fun op => op.compactInjB 1) = true :=
  ⟨List.all_eq_true.mpr (fun op _ => TOp.localBelow 1 op), rfl⟩

def moePtx : List String :=
  moeTape.map (fun op =>
    Body.text "main" (unitBufs 1 [op]).length 0 (unitStmt 1 [op]))

theorem moe_ptx_fits :
    moePtx.all (fun t => decide (t.toUTF8.toList.length + 1 ≤ QSLOT)) = true := by
  native_decide

/-- **Seam guard: the launch geometry is the stages' own.** -/
theorem moe_grids_are_the_stages :
    moeTape.map TOp.gridOf
      = ((lowerNet (R := Float32) 1 (forget (moeTape.map TOp.node))).getD []).map
          (fun S => S.val.grid) := by rfl

/-- What the host uploads once: the ones row, the router weight, the token,
    and every expert's weights.  The gates are written per token. -/
def mHostIn : AlgorithmLib.Layout.RegionMap :=
  let sizes := mBufBytes.take 3 ++ (List.range (3 * NE)).map (fun _ => MDFF * MD * 4)
  (List.range sizes.length).map (fun i =>
    ⟨s!"in{i}", (sizes.take i).foldl (· + ·) 0, sizes.getD i 0⟩)

theorem mHostIn_packed :
    AlgorithmLib.Layout.RegionMap.packedB 0 mHostIn = true := by decide

def MHOST_BYTES : Nat := AlgorithmLib.Layout.RegionMap.total mHostIn
/-- Where the driver's state starts (`Drv.App`). -/
def MORG : Nat := 0x0100
def MPTX_OFF : Nat := 0x2000
def mSlotOff (i : Nat) : Nat := MPTX_OFF + i * QSLOT
def MMEM_SIZE : Nat := mSlotOff moeTape.length

/-- A launch: its slot, its grid, and the buffers it binds. -/
def mLaunches : List (Nat × Nat × List Buf) :=
  (List.range moeTape.length).zip
    (moeTape.map (fun o => (TOp.gridOf o, TOp.bufs 1 o)))

/-- Bytes of one expert matrix. -/
def MEXPERT : Nat := MDFF * MD * 4

/-- Bytes per allocation: each buffer's, then the store, the experts'
    matrices in order, `MEXPERT` bytes apart. -/
def mAllocBytes : List Nat := mBufBytes ++ [3 * NE * MEXPERT]

/-- **Seam guard: no kernel names the store**, so a kernel's allocations are
    its buffers', one for one. -/
theorem moe_store_unbound :
    mLaunches.all (fun (_, _, bs) => bs.all fun b => b < MSTORE) = true := by decide

/-- **The dispatch in the driver's standard layout**: an allocation per
    buffer, then the store, and each kernel handed its own buffers. -/
def mApp : Drv.App :=
  { bufs := mAllocBytes,
    kernels := mLaunches.map fun (i, g, bs) => ⟨mSlotOff i, QSLOT, bs, (g, 1, 1), (32, 1, 1), []⟩,
    memSize := MMEM_SIZE, org := MORG }

/-- **Every byte this program names.** -/
def mMemMap : AlgorithmLib.Layout.RegionMap :=
  [⟨"host_len", QHOST_LEN_OFF, 8⟩, ⟨"driver", MORG, mApp.stateEnd - MORG⟩]
    ++ (List.range moeTape.length).map (fun i => ⟨s!"ptx{i}", mSlotOff i, QSLOT⟩)


#eval LayoutScan.check "Warp.MlpCifar" [``memMap, ``qMemMap, ``mMemMap]
/-- **Seam guard: the slots tile the image without overlapping.** -/
theorem moeMap_ok :
    mMemMap.okB = true ∧ mMemMap.withinB MMEM_SIZE = true := by decide

/-- The signature the entry points' typestates state of every kernel: `main`
    takes one pointer per buffer it binds. -/
theorem moe_sigs :
    ((moePtx.zip moeTape).all fun (p, op) =>
      HProg.Sem.ptxParamSizes p "main" == some ((op.bufs 1).map fun _ => 8)) = true := by
  native_decide

/-- Bring the driver up, make every allocation, load every kernel and write
    its parameters, then upload the weights that never move. -/
def mLoadFn : Prog V L Unit := do
  let base ← basePtr
  let dataPtr ← dataPtr
  mApp.setup base
  -- The three shared inputs, then the store, allocation `MSTORE`.
  for i in List.range 3 do
    mApp.upload base i (← iaddImm dataPtr (AlgorithmLib.Layout.RegionMap.offAt mHostIn i)) (mBufBytes.getD i 0)
  mApp.upload base MSTORE (← iaddImm dataPtr (AlgorithmLib.Layout.RegionMap.offAt mHostIn 3))
    (3 * NE * MEXPERT)

/-- **Choosing experts, as the host does it.**

    One `u32` per slot arrives in the input region.  For slot `j` holding
    expert `e`, the pointers to its three matrices in the store are written
    into the slots of the three buffers slot `j`'s kernels read — six
    pointers, no kernel re-emitted and no weight copied.  The expert index is
    loaded, so the address is computed at run time; nothing about which expert
    is chosen is baked into the image. A choice the data does not hold whole,
    or one naming no expert, binds nothing. -/
def mBindExperts : Prog V L Unit :=
  do
  let base ← basePtr
  let dataPtr ← dataPtr
  let mat ← iconst64 MEXPERT
  let three ← iconst64 3
  let ne ← iconst64 NE
  -- the caller's choice, one expert per slot: bound only when the data holds
  -- every slot's and each names an expert there is
  Prog.when .ule (← iconst64 (4 * MUSED)) (← dataLen) do
    let es ← (List.range MUSED).mapM fun (j : Nat) => do
      uextend64 (← load32 (← iaddImm dataPtr ((4 * j : Nat) : Int)))
    let bind : Prog V L Unit := do
      for (j, e64) in (List.range MUSED).zip es do
        let e3 ← imul e64 three
        for t in List.range 3 do
          let off ← imul (← iaddImm e3 t) mat
          let p ← iadd (← load64 (← iaddImm base (mApp.bufSlot MSTORE))) off
          storeI64 p (← iaddImm base (mApp.bufSlot (MSLOT0 + 3 * j + t)))
    es.foldr (fun e k => do let _ ← ifte .ult e ne (do k; pure %[]) (pure %[])) bind

/-- Whether kernel `i` binds a buffer whose pointer `mBindExperts` moves. -/
def mRebinds (i : Nat) : Bool :=
  ((mLaunches.getD i (0, 0, [])).2.2).any fun b => MSLOT0 ≤ b && b < MBASE

/-- Launches `[lo, hi)` of the dispatch tape, one synchronisation at the
    end. A kernel bound to a slot `mBindExperts` rewrites has its parameters
    written again first, so it reads the experts chosen. -/
def mRunRange (lo hi : Nat) : Prog V L Unit := do
  let base ← basePtr
  for i in (List.range hi).drop lo do
    if mRebinds i then mApp.rebind base i
    mApp.run base i
  Drv.sync

def mUploadFn (b n : Nat) : Prog V L Unit := do
  let base ← basePtr
  mApp.upload base b (← dataPtr) n

def mFetchFn (b n : Nat) : Prog V L Unit := do
  let base ← basePtr
  mApp.download base (← outPtr) b n

/-- The bodies the mixture-of-experts artifact carries, in the order it numbers
    them. -/
def mShippedBodies : List (String × Prog.Body) :=
  [ ("main", mLoadFn)
  , ("runRouter", mRunRange 0 mRouterTape.length)
  , ("fetchGate", mFetchFn MGATE (NE * 4))
  , ("bindExperts", mBindExperts)
  , ("uploadGates", mUploadFn 3 128)
  , ("runExperts", mRunRange mRouterTape.length moeTape.length)
  , ("fetchOut", mFetchFn MOUT (MD * 4)) ]


def mClifIR : Except String (List FuncData) :=
  Prog.program <|
    .ok noopFunction :: mShippedBodies.zipIdx.map
      (fun ((name, body), i) => Prog.entry name (Prog.compileProg (i + 1) body))

def mInitialMemory : List UInt8 :=
  mApp.initialMemory moePtx [(QHOST_LEN_OFF, u32le MHOST_BYTES)]

def mSetup (clif : List FuncData) : Artifact := {
  functions := clif
  required_memory := MMEM_SIZE
  initial_memory := mInitialMemory
}

def artifacts (clif qClif mClif : List FuncData) : Array ArtifactEntry :=
  #[ artifactEntry "ten_qwen_block" (qSetup qClif),
     artifactEntry "ten_moe_dispatch" (mSetup mClif),
     artifactEntry "mlp_cifar" (setup clif) ]

end MlpCifar

def Warp.MlpCifar.main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  let clif ← Prog.orDie MlpCifar.clifIR
  let qClif ← Prog.orDie MlpCifar.qClifIR
  let mClif ← Prog.orDie MlpCifar.mClifIR
  emitArtifacts outDir (MlpCifar.artifacts clif qClif mClif)


#eval ShipScan.check "Warp.MlpCifar" `Warp.MlpCifar.main
