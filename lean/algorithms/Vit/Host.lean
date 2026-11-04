module
public import Vit.Body
meta import Vit.Body
public import Vit.Guards
meta import Vit.Guards
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# A group's launch, through the driver, over the whole device memory

`vit_drv_launch` is the host triple for launching a group's kernel: from a
world where `cuLaunchKernel` answers, its parameters pointing at the group's
buffers, on a device honest that the launch leaves the words the group's
statement kernel leaves (`TapeLayout.machine`), every word of device memory
afterwards is what the group's operations, folded in order over the whole
memory, leave there (`TapeLayout.unit_drv_launch`). Over buffers of the
layout's sizes each group's kernel runs (`vit_runs`).

The schedule's units compute the tape (`vit_range_computes_tape_with`), each
contraction as the vendor computes it (`vit_gemm_words`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Contracts
  AlgorithmLib.HProg.DevSpec AlgorithmLib.Device AlgorithmLib.Kernel AlgorithmLib.ML AlgorithmLib.ML.BodyProof
  AlgorithmLib.ML.BodyWarp AlgorithmLib.ML.BodyCheck AlgorithmLib.ML.GroupLink

namespace Vit

/-- **A group's statement kernel runs** over buffers of at least the sizes
    the tape allocates: a run of its launch exists (`BodyRuns.launch_ends`,
    its obligations the ones `vit_static` and `vit_fine` decide). -/
theorem vit_runs (u : List Nat) (hu : u ∈ vProvenUnits) (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    (bufs : Nat → ByteArray)
    (hsz : ∀ b, b < (vUnitBufs u).length → (vLayout.sizes (vUnitOps u)).getD b 0 ≤ (bufs b).size) :
    vLayout.Runs (vUnitOps u) hw bufs :=
  TapeLayout.unit_runs (vit_unit_ok u hu) hw hw32 bufs hsz

/-- **A group's launch through the driver leaves the whole device memory as
    its operations do.** From a world where `cuLaunchKernel` answers with its
    parameters pointing at the group's buffers, each holding whole words, at
    least as large as the layout makes them, on a device honest about the
    launch, every word of every buffer afterwards is what the group's
    operations, folded in order at `F32` over the whole memory's words, leave
    there. -/
theorem vit_drv_launch (u : List Nat) (hu : u ∈ vProvenUnits) (hw : Hw)
    (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    {w : World} {hf gx gy gz bx by_ bz shmem hstream params : UInt64}
    {fi mi p : Nat} {entry ptx : String} {sizes : List Nat} {args : List UInt64}
    (hl : w.dev.live = true) (hcap : w.dev.capture = none) (hk : KeepsSize w.kernel)
    (h1 : handleOf kFunc hf = some fi) (h2 : w.dev.funcs[fi]? = some (mi, entry))
    (h3 : w.dev.modules[mi]? = some (some ptx)) (h4 : w.dev.streamParty? hstream = some p)
    (h5 : ptxParamSizes ptx entry = some sizes) (h6 : kernelArgs w.mem params sizes = some args)
    (hdims : (DrvLaunch.dimsOf gx gy gz bx by_ bz).any (· == 0) = false)
    (hr : ∀ a ∈ args, ∀ id off b, w.dev.range? a 0 = some (id, off, b) → Ready w.dev.race p id true)
    (hns : DrvLaunch.bindsOf w.dev args = vUnitBufs u)
    (hsz : ∀ g ∈ vUnitBufs u, vBufBytes.getD g 0 ≤ (devMem w.dev g).size)
    (h4s : ∀ g ∈ vUnitBufs u, (devMem w.dev g).size % 4 = 0)
    (hM : Honest w.kernel ⟨ptx, entry, vUnitBufs u, DrvLaunch.dimsOf gx gy gz bx by_ bz, args,
      w.dev.writesOf (w.dev.kernelWrites ptx entry) args⟩ (vLayout.machine (vUnitOps u) hw)) :
    ∃ r w', cudaDrv .launchKernel [hf, gx, gy, gz, bx, by_, bz, shmem, hstream, params, 0] w
        = some (r, w') ∧
      ∀ g j, j < 2 ^ 32 → 4 * j + 4 ≤ (devMem w.dev g).size →
        loadWord (devMem w'.dev g) j =
          ((vUnitOps u).foldl (fun m op => @TOp.den F32 (F32.ops hw) _ op m) (wordView (devMem w.dev)) g j).bits :=
  TapeLayout.unit_drv_launch (vit_unit_ok u hu) hw hw32 hl hcap hk h1 h2 h3 h4 h5 h6 hdims hr hns hsz h4s hM

/-- **The schedule computes the tape with its contractions the vendor's**:
    each unit's operations folded unit after unit, its contractions read as
    the vendor computes them and every other operation as its denotation at
    `F32`, leave memory as the tape `vFused` read that way does. -/
theorem vit_schedule_computes_vendor_tape (hw : Hw) (m : Buf → Nat → F32) :
    (vUnits.flatMap vUnitOps).foldl (fun mm o => TOp.denV hw o mm) m = vFused.foldl (fun mm o => TOp.denV hw o mm) m :=
  @vit_schedule_computes_tape_with F32 (F32.ops hw) _ _ (denV_frame hw SQ) m

/-- The units a range issues: those whose operations all lie in it. -/
def vRangeUnits (lo hi : Nat) : List (List Nat) :=
  vUnits.filter fun u => lo ≤ vUnitLo u && vUnitHi u < hi

/-- **The forward's units, in the schedule's order, run the tape's first
    `VFWD_N` operations**, each past only operations it cannot observe. -/
theorem vit_fwd_pull :
    pull SQ ((vFused.take VFWD_N).zipIdx.map fun p => (p.2, p.1)) (vRangeUnits 0 VFWD_N).flatten = true := by
  native_decide

theorem filterMap_congr_mem {α β : Type} {f g : α → Option β} : ∀ {l : List α}, (∀ x ∈ l, f x = g x) →
    l.filterMap f = l.filterMap g
  | [], _ => rfl
  | x :: xs, h => by
      simp only [List.filterMap_cons, h x List.mem_cons_self,
        filterMap_congr_mem (fun y hy => h y (List.mem_cons_of_mem _ hy))]

theorem le_foldl_max : ∀ (u : List Nat) (acc i : Nat), i ∈ u → i ≤ u.foldl max acc
  | [], _, _, h => by cases h
  | x :: xs, acc, i, h => by
      simp only [List.foldl_cons]
      rcases List.mem_cons.mp h with rfl | h
      · exact Nat.le_trans (Nat.le_max_right acc i) (foldl_max_ge xs _)
      · exact le_foldl_max xs _ i h
where
  foldl_max_ge : ∀ (xs : List Nat) (a : Nat), a ≤ xs.foldl max a
    | [], _ => Nat.le_refl _
    | y :: ys, a => Nat.le_trans (Nat.le_max_left a y) (foldl_max_ge ys _)

/-- **A range's units, in the schedule's order, compute the tape's first `hi`
    operations**, under any meaning of them that keeps the frame, where the
    units bring each of those operations to the front past only ones it
    cannot observe (`pull`). -/
theorem vit_range_computes_tape_with {R : Type} [NumOps R] [NumLit R] {S : TOp → (Buf → Nat → R) → Buf → Nat → R}
    (hS : TOp.Frame SQ S) (hi : Nat)
    (hp : pull SQ ((vFused.take hi).zipIdx.map fun p => (p.2, p.1)) (vRangeUnits 0 hi).flatten = true)
    (m : Buf → Nat → R) :
    ((vRangeUnits 0 hi).flatMap vUnitOps).foldl (fun mm o => S o mm) m
      = (vFused.take hi).foldl (fun mm o => S o mm) m := by
  have e : (vRangeUnits 0 hi).flatMap vUnitOps = (vRangeUnits 0 hi).flatten.filterMap ((vFused.take hi)[·]?) := by
    rw [List.filterMap_flatten]
    unfold List.flatMap
    refine congrArg List.flatten (List.map_congr_left fun u hu => ?_)
    simp only [vRangeUnits, List.mem_filter, Bool.and_eq_true, decide_eq_true_eq] at hu
    simp only [vUnitOps, vAllArr]
    refine filterMap_congr_mem fun i hi' => ?_
    have hlt : i < hi := Nat.lt_of_le_of_lt (le_foldl_max u 0 i hi') hu.2.2
    simp [List.getElem?_take_of_lt hlt]
  rw [e]
  exact pull_tape_with hS (vFused.take hi) _ hp m

/-- **The forward's units, in the schedule's order, compute the tape's first
    `VFWD_N` operations**, under any meaning of them that keeps the frame. -/
theorem vit_fwd_computes_tape_with {R : Type} [NumOps R] [NumLit R] {S : TOp → (Buf → Nat → R) → Buf → Nat → R}
    (hS : TOp.Frame SQ S) (m : Buf → Nat → R) :
    ((vRangeUnits 0 VFWD_N).flatMap vUnitOps).foldl (fun mm o => S o mm) m
      = (vFused.take VFWD_N).foldl (fun mm o => S o mm) m :=
  vit_range_computes_tape_with hS VFWD_N vit_fwd_pull m

/-- **A training step's units, in the schedule's order, bring each of the
    tape's operations to the front past only ones it cannot observe.** -/
theorem vit_step_pull :
    pull SQ ((vFused.take VSTEP_N).zipIdx.map fun p => (p.2, p.1)) (vRangeUnits 0 VSTEP_N).flatten = true := by
  native_decide

/-- **A training step's units, in the schedule's order, compute the whole
    tape**, under any meaning of its operations that keeps the frame. -/
theorem vit_step_computes_tape_with {R : Type} [NumOps R] [NumLit R] {S : TOp → (Buf → Nat → R) → Buf → Nat → R}
    (hS : TOp.Frame SQ S) (m : Buf → Nat → R) :
    ((vRangeUnits 0 VSTEP_N).flatMap vUnitOps).foldl (fun mm o => S o mm) m
      = (vFused.take VSTEP_N).foldl (fun mm o => S o mm) m :=
  vit_range_computes_tape_with hS VSTEP_N vit_step_pull m

/-- **A ViT contraction's product, on a device honest about its products,
    leaves every word as the tape read with its contractions the vendor's
    does**: the output's words the vendor's GEMM of its operands' words,
    every other word as it was. -/
theorem vit_gemm_words (hw : Hw) {v : VendorCall → List ByteArray → ByteArray} (hv : VendorGemmHonest v)
    (hvk : Contracts.VendorKeeps v) {op : TOp} {ta tb m n k A B C : Nat}
    (hg : TOp.vendorGemm op = some (ta, tb, m, n, k, A, B, C)) (hmk : m * k ≤ 2 ^ 32) (hkn : k * n ≤ 2 ^ 32)
    (mem : DevMem) :
    ∀ b j, j < 2 ^ 32 →
      loadWord ((vendorStep v (stridedCall 0x3f800000 0 0 0 0 (ta, tb, m, n, k, A, B, C)) [A, B, C] C).run mem b) j
        = (TOp.denV hw op (wordView mem) b j).bits :=
  gemm_words hw hv hvk hg hmk hkn mem

/-- Every buffer of the layout is under `2^34` bytes. -/
theorem vBufBytes_le : vLayout.bytes.all (· ≤ 2 ^ 34) = true := by native_decide

end Vit
