import Lz4Launches
import AlgorithmLib.LZ4Concurrent

set_option maxRecDepth 1000000

/-!
  # `LaunchAgreesPerWarp`, derived rather than assumed

  `Algorithm.launch_correct` takes `LaunchAgreesPerWarp` — that the final memory
  of the real launch agrees, on each warp's output range, with what that warp
  would have written running alone.  It was assumed, and described as the place
  where the platform's memory model enters.

  Half of that was true and half was not.  `AlgorithmLib.LZ4Simt.schedule_completes`
  proves the composition outright: for *any* interleaving of race-free warps, the
  memory that comes out is each warp's own solo result on that warp's region and
  the initial memory everywhere else.  What genuinely belongs to the platform is
  only that the hardware realises *some* interleaving — PTX's documented DRF-SC
  guarantee — and that is a statement about GPUs, not about this kernel.

  What this file does is instantiate that theorem at the compressor: the regions
  are the warps' output strides, and their disjointness is already proven
  (`warp_regions_disjoint`).  What is left is `KernelConfined` below: that each
  warp, running alone, writes only inside its own stride and never reads another
  warp's.  Both are per-STEP properties, and the existing simulation establishes
  only the end-to-end frame, so neither is discharged here — but they are now
  named, checkable properties of this kernel rather than an assumption about
  hardware.
-/

namespace Lz4Interleave

open Algorithm
open AlgorithmLib.LZ4Simt

/-- Warp `w`'s output range: the bytes `ShippedCorrect` says it writes. -/
def outRegion (b outPtr : Nat) (w : Nat) (j : Nat) : Prop :=
  outPtr + w * (WP.mk b).outStride ≤ j ∧
  j < outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff + 4

/-- The regions are pairwise disjoint — this is `warp_regions_disjoint`, in the
    shape the interleaving theorem wants. -/
theorem outRegion_disjoint (b outPtr : Nat) (w w' : Fin (WP.mk b).numBlk) (j : Nat)
    (hne : w ≠ w') (h : outRegion b outPtr w.val j) : ¬ outRegion b outPtr w'.val j := by
  obtain ⟨h1, h2⟩ := h
  intro ⟨h3, h4⟩
  have hv : w.val ≠ w'.val := fun e => hne (Fin.ext e)
  rcases warp_regions_disjoint b outPtr w.val w'.val hv j h1 h2 with h5 | h5 <;> omega

/-- **The two per-step obligations.**  Everything else the interleaving argument
    needs is proven; these are what remain, and they are properties of the
    kernel running ALONE — exactly what the single-warp machine can express.

    * `writes` strengthens the frame clause of `ShippedCorrect` from "the final
      memory differs only inside the stride" to "no intermediate step ever
      writes outside it".  Nothing in the current simulation states the
      intermediate form.
    * `reads` is the read-confinement property.  `LZ4Simt.Reads` now makes it
      expressible — `ldgo` is the only instruction that reads global memory —
      and all five `ldgo` sites in the kernel use `inBase`-derived address
      registers, which is why it is believed.  Believed, not proven. -/
structure KernelConfined (b : Nat) (inPtr outPtr : Nat) (gm : Array UInt8)
    (smemB : List UInt8) : Prop where
  writes : ∀ (w : Fin (WP.mk b).numBlk) (k j : Nat),
    WritesAct (WP.mk b).kernel
      (siter (WP.mk b).kernel k (initSt w.val inPtr outPtr gm smemB)) j →
    outRegion b outPtr w.val j
  reads : ∀ (w w' : Fin (WP.mk b).numBlk) (k j : Nat), w ≠ w' →
    Reads (WP.mk b).kernel
      (siter (WP.mk b).kernel k (initSt w.val inPtr outPtr gm smemB)) j →
    ¬ outRegion b outPtr w'.val j

/-- The per-warp initial local states of a launch. -/
def launchInit (b : Nat) (inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) :
    Fin (WP.mk b).numBlk → LState :=
  fun w => (initSt w.val inPtr outPtr gm smemB).loc

theorem launchInit_withMem (b inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (w : Fin (WP.mk b).numBlk) :
    (launchInit b inPtr outPtr gm smemB w).withMem gm
      = initSt w.val inPtr outPtr gm smemB := rfl

/-- **Race-freedom for the compressor**, from the one structure above plus the
    already-proven disjointness. -/
theorem raceFree_of_confined (b inPtr outPtr : Nat) (gm : Array UInt8)
    (smemB : List UInt8) (hc : KernelConfined b inPtr outPtr gm smemB) :
    RaceFree (WP.mk b).kernel (launchInit b inPtr outPtr gm smemB) gm
      (fun w => outRegion b outPtr w.val) where
  disjoint := fun w w' j hne h => outRegion_disjoint b outPtr w w' j hne h
  writes := fun w k j hw => hc.writes w k j hw
  reads := fun w w' k j hne hr => hc.reads w w' k j hne hr

/-- **The payoff.**  For any schedule that runs every warp to completion, the
    memory the launch leaves is exactly what `LaunchAgreesPerWarp` and
    `LaunchFrame` assert — now a consequence of race-freedom rather than an
    assumption about the hardware. -/
theorem launch_agrees (b inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hc : KernelConfined b inPtr outPtr gm smemB)
    (sched : List (Fin (WP.mk b).numBlk))
    (fin : Fin (WP.mk b).numBlk → SState) (need : Fin (WP.mk b).numBlk → Nat)
    (hreach : ∀ w, siter (WP.mk b).kernel (need w) (initSt w.val inPtr outPtr gm smemB) = fin w)
    (hhalt : ∀ w, Halted (WP.mk b).kernel (fin w))
    (hlong : ∀ w, need w ≤ schedCount sched w) :
    let gfinal := (crun (WP.mk b).kernel sched ⟨launchInit b inPtr outPtr gm smemB, gm⟩).gmem
    (∀ (w : Fin (WP.mk b).numBlk) j, outRegion b outPtr w.val j →
        gfinal.getD j 0 = (fin w).gmem.getD j 0) ∧
    (∀ j, (∀ w : Fin (WP.mk b).numBlk, ¬ outRegion b outPtr w.val j) →
        gfinal.getD j 0 = gm.getD j 0) ∧
    gfinal.size = gm.size :=
  schedule_completes (raceFree_of_confined b inPtr outPtr gm smemB hc) sched fin need
    hreach hhalt hlong

/-- **`LaunchFrame` is derived too.**  `Lz4Launches.LaunchFrame` — that a launch
    resizes nothing and changes nothing outside the union of the warps' output
    ranges — is exactly the second and third conclusions of `schedule_completes`,
    so it needs no assumption at all beyond race-freedom.  Only the *agreement*
    half still has to identify the terminal state. -/
theorem launchFrame_of_confined (b inPtr outPtr : Nat) (gm : Array UInt8)
    (smemB : List UInt8) (hc : KernelConfined b inPtr outPtr gm smemB)
    (sched : List (Fin (WP.mk b).numBlk))
    (fin : Fin (WP.mk b).numBlk → SState) (need : Fin (WP.mk b).numBlk → Nat)
    (hreach : ∀ w, siter (WP.mk b).kernel (need w) (initSt w.val inPtr outPtr gm smemB) = fin w)
    (hhalt : ∀ w, Halted (WP.mk b).kernel (fin w))
    (hlong : ∀ w, need w ≤ schedCount sched w) :
    Lz4Launches.LaunchFrame b outPtr gm
      (crun (WP.mk b).kernel sched ⟨launchInit b inPtr outPtr gm smemB, gm⟩).gmem := by
  obtain ⟨_, hout, hsize⟩ :=
    schedule_completes (raceFree_of_confined b inPtr outPtr gm smemB hc) sched fin need
      hreach hhalt hlong
  refine ⟨hsize, ?_⟩
  intro j hj
  refine hout j ?_
  intro w
  rcases hj w.val w.isLt with h | h
  · exact fun hr => absurd hr.1 (by omega)
  · exact fun hr => absurd hr.2 (by omega)

-- ── Identifying the terminal state ───────────────────────────────────────────

/-- The shipped kernel ends `lbl "OOB"; ret`. -/
theorem tail32 : (WP.mk 15).kernel[272]? = some (.lbl "OOB") ∧
    (WP.mk 15).kernel[273]? = some .ret := by
  constructor <;> decide

/-- `pc = 272` is one step from halting, and that step touches no memory — so the
    state `ShippedCorrect` hands back and the state a schedule runs to have the
    same global memory. -/
theorem halt_after_272 (st : SState) (hpc : st.pc = 272) :
    (sstep (WP.mk 15).kernel st).gmem = st.gmem ∧
    (sstep (WP.mk 15).kernel st).pc = 273 ∧
    Halted (WP.mk 15).kernel (sstep (WP.mk 15).kernel st) := by
  obtain ⟨h272, h273⟩ := tail32
  have hstep : sstep (WP.mk 15).kernel st = st.setPc 273 := by
    rw [sstep_eq_instr, hpc, h272]
    show sstepInstr (WP.mk 15).kernel (.lbl "OOB") st = st.setPc 273
    rw [sstepInstr, hpc]
  refine ⟨by rw [hstep]; rfl, by rw [hstep]; rfl, ?_⟩
  unfold Halted
  rw [hstep, sstep_eq_instr]
  have hp : (st.setPc 273).pc = 273 := rfl
  rw [hp, h273]
  rfl

/-- **`pc = 272` happens at most once.**  After it the machine is at `ret`, which
    is a fixpoint, so a state with that pc is reached at exactly one step count —
    which is what lets an arbitrary reachable `pc = 272` state be identified with
    the one a schedule ran to. -/
theorem pc272_unique (init : SState) (n m : Nat)
    (hn : (siter (WP.mk 15).kernel n init).pc = 272)
    (hm : (siter (WP.mk 15).kernel m init).pc = 272) :
    siter (WP.mk 15).kernel n init = siter (WP.mk 15).kernel m init := by
  rcases Nat.lt_trichotomy n m with h | h | h
  · exfalso
    obtain ⟨_, hpc273, hhalt⟩ := halt_after_272 _ hn
    have hreach : siter (WP.mk 15).kernel (n + 1) init = sstep (WP.mk 15).kernel
        (siter (WP.mk 15).kernel n init) := siter_succ _ _ _
    have := siter_of_halted_ge (WP.mk 15).kernel init _ (n + 1) hreach hhalt m (by omega)
    rw [this, hpc273] at hm
    exact absurd hm (by decide)
  · rw [h]
  · exfalso
    obtain ⟨_, hpc273, hhalt⟩ := halt_after_272 _ hm
    have hreach : siter (WP.mk 15).kernel (m + 1) init = sstep (WP.mk 15).kernel
        (siter (WP.mk 15).kernel m init) := siter_succ _ _ _
    have := siter_of_halted_ge (WP.mk 15).kernel init _ (m + 1) hreach hhalt n (by omega)
    rw [this, hpc273] at hn
    exact absurd hn (by decide)

/-- **`LaunchAgreesPerWarp` is derived.**  The last assumption `launch_correct`
    carried about the memory model is now a theorem: for a schedule that runs
    every warp one step past `pc = 272`, the memory it leaves agrees, on each
    warp's output range, with the state `ShippedCorrect` hands back — for *every*
    reachable such state, since `pc272_unique` says there is only one.

    What is left to the platform is that the hardware realises *some* schedule.
    What is left to this kernel is `Lz4Sites.RegConfined`. -/
theorem launchAgrees_of_confined (inPtr outPtr : Nat) (gm : Array UInt8)
    (smemB : List UInt8) (hc : KernelConfined 15 inPtr outPtr gm smemB)
    (sched : List (Fin (WP.mk 15).numBlk)) (nAt : Fin (WP.mk 15).numBlk → Nat)
    (hAt : ∀ w, (siter (WP.mk 15).kernel (nAt w)
      (initSt w.val inPtr outPtr gm smemB)).pc = 272)
    (hlong : ∀ w, nAt w + 1 ≤ schedCount sched w) :
    Algorithm.LaunchAgreesPerWarp 15 inPtr outPtr gm smemB
      (crun (WP.mk 15).kernel sched ⟨launchInit 15 inPtr outPtr gm smemB, gm⟩).gmem := by
  intro w hw ss' hreach hpc272 j hj1 hj2
  let wf : Fin (WP.mk 15).numBlk := ⟨w, hw⟩
  -- the warp's halted state, one step past `pc = 272`
  have hstep := halt_after_272 _ (hAt wf)
  obtain ⟨hmem, _, hhalt⟩ := hstep
  obtain ⟨hagree, _, _⟩ :=
    launch_agrees 15 inPtr outPtr gm smemB hc sched
      (fun x => sstep (WP.mk 15).kernel (siter (WP.mk 15).kernel (nAt x)
        (initSt x.val inPtr outPtr gm smemB)))
      (fun x => nAt x + 1)
      (fun x => siter_succ _ _ _)
      (fun x => (halt_after_272 _ (hAt x)).2.2)
      hlong
  have hgm : (sstep (WP.mk 15).kernel (siter (WP.mk 15).kernel (nAt wf)
      (initSt wf.val inPtr outPtr gm smemB))).gmem.getD j 0 = ss'.gmem.getD j 0 := by
    obtain ⟨n, hn⟩ := hreach
    have hiter : siter (WP.mk 15).kernel n (initSt w inPtr outPtr gm smemB) = ss' :=
      (sreaches_iff_siter _ n _ _).mp hn
    have huniq := pc272_unique (initSt w inPtr outPtr gm smemB) (nAt wf) n (hAt wf)
      (by rw [hiter]; exact hpc272)
    rw [hmem, huniq, hiter]
  rw [← hgm]
  exact hagree wf j ⟨hj1, hj2⟩

/-- info: 'Lz4Interleave.launch_agrees' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms launch_agrees

end Lz4Interleave
