import Lz4Launches
import AlgorithmLib.LZ4Concurrent

set_option maxRecDepth 1000000

/-!
  # `LaunchAgreesPerWarp`, derived rather than assumed

  `Algorithm.launch_correct` takes `LaunchAgreesPerWarp` — that the final memory
  of the real launch agrees, on each warp's output range, with what that warp
  would have written running alone.  It is where the platform's memory model
  would enter, if it had to.

  Almost none of it does.  `AlgorithmLib.LZ4Simt.schedule_completes`
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

/-- **The shape of the kernel's tail**: `pc = 272` is `lbl "OOB"` and `pc = 273`
    is `ret`.  Everything below identifies the terminal state from these two
    facts alone, so it holds at any geometry that can produce them. -/
def TailOOB (b : Nat) : Prop :=
  (WP.mk b).kernel[272]? = some (.lbl "OOB") ∧ (WP.mk b).kernel[273]? = some .ret

theorem tail32 : TailOOB 15 := ⟨by decide, by decide⟩
theorem tail64 : TailOOB 16 := ⟨by decide, by decide⟩

/-- `pc = 272` is one step from halting, and that step touches no memory — so the
    state `ShippedCorrect` hands back and the state a schedule runs to have the
    same global memory. -/
theorem halt_after_272 (b : Nat) (ht : TailOOB b) (st : SState) (hpc : st.pc = 272) :
    (sstep (WP.mk b).kernel st).gmem = st.gmem ∧
    (sstep (WP.mk b).kernel st).pc = 273 ∧
    Halted (WP.mk b).kernel (sstep (WP.mk b).kernel st) := by
  obtain ⟨h272, h273⟩ := ht
  have hstep : sstep (WP.mk b).kernel st = st.setPc 273 := by
    rw [sstep_eq_instr, hpc, h272]
    show sstepInstr (WP.mk b).kernel (.lbl "OOB") st = st.setPc 273
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
theorem pc272_unique (b : Nat) (ht : TailOOB b) (init : SState) (n m : Nat)
    (hn : (siter (WP.mk b).kernel n init).pc = 272)
    (hm : (siter (WP.mk b).kernel m init).pc = 272) :
    siter (WP.mk b).kernel n init = siter (WP.mk b).kernel m init := by
  rcases Nat.lt_trichotomy n m with h | h | h
  · exfalso
    obtain ⟨_, hpc273, hhalt⟩ := halt_after_272 b ht _ hn
    have hreach : siter (WP.mk b).kernel (n + 1) init = sstep (WP.mk b).kernel
        (siter (WP.mk b).kernel n init) := siter_succ _ _ _
    have := siter_of_halted_ge (WP.mk b).kernel init _ (n + 1) hreach hhalt m (by omega)
    rw [this, hpc273] at hm
    exact absurd hm (by decide)
  · rw [h]
  · exfalso
    obtain ⟨_, hpc273, hhalt⟩ := halt_after_272 b ht _ hm
    have hreach : siter (WP.mk b).kernel (m + 1) init = sstep (WP.mk b).kernel
        (siter (WP.mk b).kernel m init) := siter_succ _ _ _
    have := siter_of_halted_ge (WP.mk b).kernel init _ (m + 1) hreach hhalt n (by omega)
    rw [this, hpc273] at hn
    exact absurd hn (by decide)

/-- **`LaunchAgreesPerWarp` is derived.**  The last assumption `launch_correct`
    carried about the memory model is now a theorem: for a schedule that runs
    every warp one step past `pc = 272`, the memory it leaves agrees, on each
    warp's output range, with the state `ShippedCorrect` hands back — for *every*
    reachable such state, since `pc272_unique` says there is only one.

    What is left to the platform is that the hardware realises *some* schedule.
    What is left to this kernel is `Lz4Sites.RegConfined`. -/
theorem launchAgrees_of_confined (b : Nat) (ht : TailOOB b) (inPtr outPtr : Nat)
    (gm : Array UInt8)
    (smemB : List UInt8) (hc : KernelConfined b inPtr outPtr gm smemB)
    (sched : List (Fin (WP.mk b).numBlk)) (nAt : Fin (WP.mk b).numBlk → Nat)
    (hAt : ∀ w, (siter (WP.mk b).kernel (nAt w)
      (initSt w.val inPtr outPtr gm smemB)).pc = 272)
    (hlong : ∀ w, nAt w + 1 ≤ schedCount sched w) :
    Algorithm.LaunchAgreesPerWarp b inPtr outPtr gm smemB
      (crun (WP.mk b).kernel sched ⟨launchInit b inPtr outPtr gm smemB, gm⟩).gmem := by
  intro w hw ss' hreach hpc272 j hj1 hj2
  let wf : Fin (WP.mk b).numBlk := ⟨w, hw⟩
  -- the warp's halted state, one step past `pc = 272`
  have hstep := halt_after_272 b ht _ (hAt wf)
  obtain ⟨hmem, _, hhalt⟩ := hstep
  obtain ⟨hagree, _, _⟩ :=
    launch_agrees b inPtr outPtr gm smemB hc sched
      (fun x => sstep (WP.mk b).kernel (siter (WP.mk b).kernel (nAt x)
        (initSt x.val inPtr outPtr gm smemB)))
      (fun x => nAt x + 1)
      (fun x => siter_succ _ _ _)
      (fun x => (halt_after_272 b ht _ (hAt x)).2.2)
      hlong
  have hgm : (sstep (WP.mk b).kernel (siter (WP.mk b).kernel (nAt wf)
      (initSt wf.val inPtr outPtr gm smemB))).gmem.getD j 0 = ss'.gmem.getD j 0 := by
    obtain ⟨n, hn⟩ := hreach
    have hiter : siter (WP.mk b).kernel n (initSt w inPtr outPtr gm smemB) = ss' :=
      (sreaches_iff_siter _ n _ _).mp hn
    have huniq := pc272_unique b ht (initSt w inPtr outPtr gm smemB) (nAt wf) n (hAt wf)
      (by rw [hiter]; exact hpc272)
    rw [hmem, huniq, hiter]
  rw [← hgm]
  exact hagree wf j ⟨hj1, hj2⟩

/-- info: 'Lz4Interleave.launch_agrees' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms launch_agrees

-- ── From `LayoutOK` to a whole run ───────────────────────────────────────────

/-- Every warp reaches `pc = 272` at some step count, as a *function* of the
    warp.  `ShippedCorrect` gives it one warp at a time; choice does the rest. -/
theorem exists_nAt (b inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hcorrect : ShippedCorrect b) (hlayout : LayoutOK b inPtr outPtr gm) :
    ∃ nAt : Fin (WP.mk b).numBlk → Nat,
      ∀ w : Fin (WP.mk b).numBlk,
        (siter (WP.mk b).kernel (nAt w)
          (initSt w.val inPtr outPtr gm smemB)).pc = 272 := by
  have h : ∀ w : Fin (WP.mk b).numBlk, ∃ n,
      (siter (WP.mk b).kernel n (initSt w.val inPtr outPtr gm smemB)).pc = 272 := by
    intro w
    obtain ⟨hglob, hper⟩ := hlayout
    obtain ⟨l1, l2, l3, l5, l6⟩ := hper w.val w.isLt
    obtain ⟨n, ss', k, hreach, hpc, -⟩ :=
      hcorrect w.val inPtr outPtr gm smemB w.isLt hglob l1 l2 l3 l5 l6
    exact ⟨n, by rw [(sreaches_iff_siter _ n _ _).mp hreach]; exact hpc⟩
  exact ⟨fun w => Classical.choose (h w), fun w => Classical.choose_spec (h w)⟩

/-- The schedule gives every warp at least as many steps as it needs to finish.

    Existential per warp, not universal over schedules: "every schedule is long
    enough" is false for the empty one and would make everything below vacuous.
    `schedComplete_exists` proves this form inhabited. -/
def SchedComplete (b inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (sched : List (Fin (WP.mk b).numBlk)) : Prop :=
  ∀ w : Fin (WP.mk b).numBlk, ∃ n,
    (siter (WP.mk b).kernel n (initSt w.val inPtr outPtr gm smemB)).pc = 272 ∧
    n + 1 ≤ schedCount sched w

-- how many steps a schedule gives one warp, over the shapes a schedule is built from

theorem schedCount_append {n : Nat} (a b : List (Fin n)) (w : Fin n) :
    schedCount (a ++ b) w = schedCount a w + schedCount b w := by
  induction a with
  | nil => simp [schedCount]
  | cons x xs ih =>
      rw [List.cons_append, schedCount_cons, schedCount_cons, ih]
      split <;> omega

theorem schedCount_replicate {n : Nat} (m : Nat) (v w : Fin n) :
    schedCount (List.replicate m v) w = if v = w then m else 0 := by
  induction m with
  | zero => by_cases h : v = w <;> simp [h, schedCount]
  | succ k ih =>
      rw [List.replicate_succ, schedCount_cons, ih]
      by_cases h : v = w <;> simp [h]

/-- A schedule that runs each warp `f w` times gives it at least that many steps. -/
theorem schedCount_flatMap_ge {n : Nat} (L : List (Fin n)) (f : Fin n → Nat) (w : Fin n)
    (hw : w ∈ L) : f w ≤ schedCount (L.flatMap (fun v => List.replicate (f v) v)) w := by
  induction L with
  | nil => cases hw
  | cons x xs ih =>
      rw [List.flatMap_cons, schedCount_append, schedCount_replicate]
      rcases List.mem_cons.mp hw with h | h
      · subst h; simp
      · have := ih h; split <;> omega

/-- **A long-enough schedule exists** — one more step per warp than `exists_nAt`
    says it needs.  Without this the hypothesis could be unsatisfiable. -/
theorem schedComplete_exists (b inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hcorrect : ShippedCorrect b) (hlayout : LayoutOK b inPtr outPtr gm) :
    ∃ sched, SchedComplete b inPtr outPtr gm smemB sched := by
  obtain ⟨nAt, hAt⟩ := exists_nAt b inPtr outPtr gm smemB hcorrect hlayout
  refine ⟨(List.finRange (WP.mk b).numBlk).flatMap
    (fun v => List.replicate (nAt v + 1) v), fun w => ⟨nAt w, hAt w, ?_⟩⟩
  exact schedCount_flatMap_ge _ (fun v => nAt v + 1) w (List.mem_finRange w)

/-- The memory a launch under `sched` leaves behind. -/
def gmemAfter (b inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (sched : List (Fin (WP.mk b).numBlk)) : Array UInt8 :=
  (crun (WP.mk b).kernel sched ⟨launchInit b inPtr outPtr gm smemB, gm⟩).gmem

/-- **Both halves of one launch, from `KernelConfined` alone**: the two
    components of `LaunchesTo`, for the memory a schedule actually produces. -/
theorem oneLaunch_ok (b : Nat) (ht : TailOOB b) (inPtr outPtr : Nat)
    (gm : Array UInt8) (smemB : List UInt8)
    (hc : KernelConfined b inPtr outPtr gm smemB)
    (sched : List (Fin (WP.mk b).numBlk))
    (hsched : SchedComplete b inPtr outPtr gm smemB sched) :
    LaunchAgreesPerWarp b inPtr outPtr gm smemB (gmemAfter b inPtr outPtr gm smemB sched) ∧
    Lz4Launches.LaunchFrame b outPtr gm (gmemAfter b inPtr outPtr gm smemB sched) := by
  have hAt : ∀ w, (siter (WP.mk b).kernel (Classical.choose (hsched w))
      (initSt w.val inPtr outPtr gm smemB)).pc = 272 :=
    fun w => (Classical.choose_spec (hsched w)).1
  have hlong : ∀ w, Classical.choose (hsched w) + 1 ≤ schedCount sched w :=
    fun w => (Classical.choose_spec (hsched w)).2
  refine ⟨launchAgrees_of_confined b ht inPtr outPtr gm smemB hc sched
      (fun w => Classical.choose (hsched w)) hAt hlong, ?_⟩
  exact launchFrame_of_confined b inPtr outPtr gm smemB hc sched
    (fun x => sstep (WP.mk b).kernel (siter (WP.mk b).kernel (Classical.choose (hsched x))
      (initSt x.val inPtr outPtr gm smemB)))
    (fun x => Classical.choose (hsched x) + 1)
    (fun x => siter_succ _ _ _)
    (fun x => (halt_after_272 b ht _ (hAt x)).2.2)
    hlong

/-- **`LayoutOK` supplies every hypothesis `KernelConfined` needs.**  Four are
    its own per-warp clauses; the fifth — warp `w`'s input slice ending before its
    output range — follows from the placement equation. -/
theorem confineHyps_of_layoutOK (b inPtr outPtr : Nat) (gm : Array UInt8)
    (hlayout : LayoutOK b inPtr outPtr gm) :
    outPtr = inPtr + ((WP.mk b).numBlk * (WP.mk b).inStride + copySlack) ∧
    (∀ w, w < (WP.mk b).numBlk → inPtr + w * (WP.mk b).inStride < 2 ^ 40) ∧
    (∀ w, w < (WP.mk b).numBlk →
      outPtr + w * (WP.mk b).outStride + 9 * (WP.mk b).inStride < 2 ^ 32) ∧
    (∀ w, w < (WP.mk b).numBlk →
      outPtr + w * (WP.mk b).outStride + 9 * (WP.mk b).inStride ≤ gm.size) ∧
    (∀ w, w < (WP.mk b).numBlk →
      inPtr + w * (WP.mk b).inStride + (WP.mk b).inStride
        ≤ outPtr + w * (WP.mk b).outStride) := by
  obtain ⟨hglob, hper⟩ := hlayout
  refine ⟨hglob, fun w hw => (hper w hw).1, fun w hw => (hper w hw).2.1,
    fun w hw => (hper w hw).2.2.1, fun w hw => ?_⟩
  have h1 : (w + 1) * (WP.mk b).inStride ≤ (WP.mk b).numBlk * (WP.mk b).inStride :=
    Nat.mul_le_mul_right _ hw
  have h2 : (w + 1) * (WP.mk b).inStride
      = w * (WP.mk b).inStride + (WP.mk b).inStride := Nat.succ_mul w _
  omega

/-- **A run of `n` launches, with no hypothesis but `LayoutOK`.**

    Each launch picks a schedule for the memory it starts from;
    `layoutOK_preserved` carries the contract across so `KernelConfined` can be
    re-established each time.

    No schedule is assumed to exist or to be long enough — complete schedules are
    constructed and `launch_agrees` holds for all of them, so the hardware owes
    only that it realises some interleaving (trusted base row 6). -/
theorem launchesTo_of_layout (b : Nat) (ht : TailOOB b) (inPtr outPtr : Nat)
    (smemB : List UInt8)
    (hcorrect : ShippedCorrect b)
    (hconf : ∀ g, LayoutOK b inPtr outPtr g → KernelConfined b inPtr outPtr g smemB) :
    ∀ (n : Nat) (gm : Array UInt8), LayoutOK b inPtr outPtr gm →
      ∃ gfinal, Lz4Launches.LaunchesTo b inPtr outPtr smemB n gm gfinal ∧
        LayoutOK b inPtr outPtr gfinal := by
  intro n
  induction n with
  | zero => exact fun gm hlay => ⟨gm, rfl, hlay⟩
  | succ m ih =>
      intro gm hlay
      obtain ⟨s, hs⟩ := schedComplete_exists b inPtr outPtr gm smemB hcorrect hlay
      obtain ⟨hagree, hframe⟩ := oneLaunch_ok b ht inPtr outPtr gm smemB (hconf gm hlay) s hs
      obtain ⟨gfinal, hrest, hlayF⟩ :=
        ih (gmemAfter b inPtr outPtr gm smemB s)
          (Lz4Launches.layoutOK_preserved b inPtr outPtr gm _ hlay hframe)
      exact ⟨gfinal, ⟨gmemAfter b inPtr outPtr gm smemB s, hagree, hframe, hrest⟩, hlayF⟩

end Lz4Interleave
