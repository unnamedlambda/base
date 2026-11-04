module
public import AlgorithmLib.MLHost.UnitWords
meta import AlgorithmLib.MLHost.UnitWords
public import AlgorithmLib.MLHost.AppDag
meta import AlgorithmLib.MLHost.AppDag
public import AlgorithmLib.Check.Entry
meta import AlgorithmLib.Check.Entry
public import AlgorithmLib.Check.DrvState
meta import AlgorithmLib.Check.DrvState
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# A run's calls, as steps of device memory

A run that ends having made calls a list of steps' records admits, into the
driver and cuBLAS, the device changed only as `DevPath` allows, took those
steps (`lit_steps`). What the entries of a program in the driver's standard
layout start from (`Drv.App.runTS`) has the driver and cuBLAS loaded, no
capture in progress, kernels that keep their sizes, and every allocation live
at its size (`runTS_facts`). So where its groups are ones the step lemmas read,
every such run leaves device memory as its groups' operations, in order,
leave it (`run_tape`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Contracts
  AlgorithmLib.Device AlgorithmLib.Kernel AlgorithmLib.Check

namespace AlgorithmLib.ML

/-- Every step's record is a call into the driver or cuBLAS. -/
theorem record_ext (a : Drv.App) (h s0 s1 : UInt64) :
    ∀ (ss : List DevCall), (ss.map (DevCall.record a h s0 s1)).all
      (fun r => match r.callee with | .ext _ => true | _ => false) = true := by
  intro ss
  refine List.all_eq_true.mpr fun r hr => ?_
  obtain ⟨s, -, rfl⟩ := List.mem_map.mp hr
  cases s with
  | kernel j =>
      cases hk : a.kernels[j]? <;> simp [DevCall.record, Drv.App.launchRec, hk]
  | gemm g => rfl
  | sync => rfl

/-- **A run through calls a list of steps' records admits is those steps**:
    from a world with the driver loaded, and cuBLAS where a step is a product,
    no capture in progress and kernels that keep their sizes, device memory at the end is the steps', in
    order, from device memory at the start. -/
theorem lit_steps {a : Drv.App} {h s0 s1 : UInt64} {ss : List DevCall} {cfg : Cfg} {args : List V} {c : Code}
    {w w' : World} {obs : List Obs}
    (hr : Sem.run cfg args w c = .ok obs w')
    (hc : (Tr.lit (ss.map (DevCall.record a h s0 s1))).Covers w obs ∧ DevPath w w')
    (hok : ss.all (DevCall.okB a) = true)
    (hcu : w.has (Ext.cuda .launchKernel).lib = true)
    (hbl : ∀ g, DevCall.gemm g ∈ ss → w.has (Ext.cublas .sgemmStridedBatched).lib = true)
    (hcap : w.dev.capture = none) (hks : KeepsSize w.kernel) :
    DevCall.Steps a w s0 s1 ss (devMem w.dev) (devMem w'.dev) := by
  obtain ⟨cs, -, hh, hx⟩ := extRun_of hr hc (fun _ hh' => hh'.lit_ext (record_ext a h s0 s1 ss))
  cases hh with
  | lit hm =>
      exact (extRun_steps hcu hcap hks hx ss hm
        (fun s hs => DevCall.okB_ok (List.all_eq_true.mp hok s hs)) hbl).1

theorem mem_zipIdx_getD : ∀ (l : List Nat) (k i : Nat), i < l.length → (l.getD i 0, k + i) ∈ l.zipIdx k
  | [], _, _, h => absurd h (Nat.not_lt_zero _)
  | x :: xs, k, 0, _ => by simp [List.zipIdx_cons]
  | x :: xs, k, i + 1, h => by
      rw [List.zipIdx_cons]
      refine List.mem_cons_of_mem _ ?_
      have := mem_zipIdx_getD xs (k + 1) i (by simpa using h)
      rw [show k + 1 + i = k + (i + 1) by omega] at this
      simpa using this

/-- **A world a run of a program in the driver's standard layout can start
    from**: the driver and, where the program uses it, cuBLAS loaded, no
    capture in progress, kernels that keep the sizes of what they write, and
    every allocation live at its size. -/
structure Ready (a : Drv.App) (w : World) : Prop where
  cuda : w.has (Ext.cuda .launchKernel).lib = true
  capture : w.dev.capture = none
  keeps : KeepsSize w.kernel
  blas : a.blas = true → w.has (Ext.cublas .sgemmStridedBatched).lib = true
  allocs : ∀ i, i < a.bufs.length → ∃ b, (w.dev.bufs[i]?).join = some b ∧ b.size = a.bufs.getD i 0

/-- **What a program's other entries start from is ready** (`Drv.App.runTS`). -/
theorem runTS_ready {a : Drv.App} {extra : TState} {w : World} (h : TState.holds (a.runTS ++ extra) w) :
    Ready a w := by
  have hm : ∀ f, f ∈ a.runTS → f.holds w := fun f hf => h f (List.mem_append_left _ hf)
  refine ⟨hm (.part (.has .cuda) true) (by simp [Drv.App.runTS, Drv.App.runTSWith]), ?_,
    (hm .oracles (by simp [Drv.App.runTS, Drv.App.runTSWith])).1,
    fun hb => hm (.part (.has .cublas) true) (by simp [Drv.App.runTS, Drv.App.runTSWith, hb]), fun i hi => ?_⟩
  · have : w.dev.capture.isSome = false := hm (.part .capturing false) (by simp [Drv.App.runTS, Drv.App.runTSWith])
    cases hc : w.dev.capture with
    | none => rfl
    | some _ => rw [hc] at this; cases this
  · refine hm (.alloc i (a.bufs.getD i 0)) ?_
    have hz : Fact.alloc i (a.bufs.getD i 0) ∈ a.bufs.zipIdx.flatMap (fun (n, i) =>
        [Fact.alloc i n, Fact.cell (arenaAt (a.bufSlot i)) (devPtr i)]) :=
      List.mem_flatMap.mpr ⟨_, by simpa using mem_zipIdx_getD a.bufs 0 i hi, by simp⟩
    simp only [Drv.App.runTS, Drv.App.runTSWith, List.mem_append]
    exact Or.inl (Or.inl (Or.inr hz))

/-- The facts a world's typestate must hold for it to be ready for a run. -/
def readyFacts (a : Drv.App) : TState :=
  [.part (.has .cuda) true, .part .capturing false, .oracles] ++
  (if a.blas then [.part (.has .cublas) true] else []) ++
  (List.range a.bufs.length).map fun i => .alloc i (a.bufs.getD i 0)

/-- **A world whose typestate holds the ready facts is ready.** -/
theorem ready_of {a : Drv.App} {S : TState} {w : World} (h : TState.holds S w)
    (hs : (readyFacts a).all S.contains = true) : Ready a w := by
  have hm : ∀ f ∈ readyFacts a, f.holds w := fun f hf =>
    h f (List.contains_iff_mem.mp (List.all_eq_true.mp hs f hf))
  refine ⟨hm (.part (.has .cuda) true) (by simp [readyFacts]), ?_, (hm .oracles (by simp [readyFacts])).1,
    fun hb => hm (.part (.has .cublas) true) (by simp [readyFacts, hb]),
    fun i hi => hm (.alloc i (a.bufs.getD i 0)) (by simp [readyFacts]; exact ⟨i, hi, rfl, rfl⟩)⟩
  have : w.dev.capture.isSome = false := hm (.part .capturing false) (by simp [readyFacts])
  cases hc : w.dev.capture with
  | none => rfl
  | some _ => rw [hc] at this; cases this

theorem runCalls_lit (a : Drv.App) (h s0 s1 : UInt64) (js : List Nat) :
    a.runCalls js = .lit ((js.map DevCall.kernel ++ [DevCall.sync]).map (DevCall.record a h s0 s1)) := by
  simp only [Drv.App.runCalls, List.map_append, List.map_map]
  rfl

/-- **Every run of an entry leaves device memory as the tape says**: a
    program in the driver's standard layout, calling cuBLAS where a step is
    a product, run from a world ready for it (`Ready`), where every run
    that ends makes steps `ps`' records then waits, each a step the device
    reads, each group one the step lemmas read and each launch handed exactly
    its group's buffers over its group's grid, on a device honest about each
    group's launch and about each product, leaves every word inside each
    buffer, at the size the layout makes it, as the groups' operations, in
    order, leave it from the memory the run started with, each contraction
    read as the vendor computes it. -/
theorem run_tape (L : TapeLayout) (a : Drv.App) (hB : L.bytes.all (· ≤ 2 ^ 34) = true)
    (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (ps : List (DevCall × List TOp))
    {cfg : Cfg} {args : List V} {c : Code} {w w' : World} {obs : List Obs}
    (hR : Ready a w) (hr : Sem.run cfg args w c = .ok obs w')
    (hc : (Tr.lit ((ps.map Prod.fst ++ [DevCall.sync]).map (DevCall.record a (drvHandle kBlas 0) 0x3f800000 0))).Covers w obs ∧
      DevPath w w')
    (hok : (ps.map Prod.fst ++ [DevCall.sync]).all (DevCall.okB a) = true)
    (hst : ps.all L.stepOk = true) (hU : ∀ p ∈ ps, ∀ j, p.1 = .kernel j → L.UnitOk p.2)
    (hbo : ps.all (L.bindsOk a) = true) (hsz : L.sizesOk a ps = true)
    (hblas : ∀ p ∈ ps, ∀ g, p.1 = .gemm g → a.blas = true)
    (hv : ∀ p ∈ ps, ∀ g, p.1 = .gemm g → VendorGemmHonest w.vendor ∧ VendorKeeps w.vendor)
    (hHon : ∀ p ∈ ps, ∀ j, p.1 = .kernel j → L.KernelHonest a w j p.2 hw) :
    L.Agree (devMem w'.dev) ((ps.flatMap Prod.snd).foldl (fun M op => TOp.denV hw op M) (wordView (devMem w.dev))) := by
  obtain ⟨hcu, hcap, hks, hbl, halloc⟩ := hR
  have hbl' : ∀ g, DevCall.gemm g ∈ ps.map Prod.fst ++ [DevCall.sync] →
      w.has (Ext.cublas .sgemmStridedBatched).lib = true := fun g hg => by
    rcases List.mem_append.mp hg with hg | hg
    · obtain ⟨p, hp, e⟩ := List.mem_map.mp hg
      exact hbl (hblas p hp g e)
    · cases List.mem_singleton.mp hg
  exact TapeLayout.steps_tape hB a hw hw32 ps (lit_steps hr hc hok hcu hbl' hcap hks) hks hv hst hU hbo hsz
    halloc hHon (fun _ _ _ => rfl)

/-- What a run launching kernels `js` in order, kernel `j` group `unit j`'s,
    needs decided: each kernel one the device reads and its group one the
    step lemmas read, handed exactly its group's buffers over its group's
    grid, every buffer at least the layout's size. -/
def kernelsOk (L : TapeLayout) (a : Drv.App) (unit : Nat → List TOp) (js : List Nat) : Bool :=
  js.all (fun j => DevCall.okB a (.kernel j) && L.unitOkB (unit j) && L.bindsOk a (.kernel j, unit j)) &&
    L.sizesOk a (js.map fun j => (DevCall.kernel j, unit j))

/-- **Every run launching kernels in order leaves device memory as the tape
    says**: a program in the driver's standard layout, run from a world
    ready for it, where every run that ends launches
    kernels `js` in order then waits (`Drv.App.runCalls`), kernel `j` group
    `unit j`'s, with what `kernelsOk` decides, on a device honest about each
    launch, leaves every word inside each buffer, at the size the layout makes
    it, as the groups' operations, in order, leave it from the memory the run
    started with. -/
theorem kernels_tape (L : TapeLayout) (a : Drv.App) (hB : L.bytes.all (· ≤ 2 ^ 34) = true)
    (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (unit : Nat → List TOp) (js : List Nat)
    (hk : kernelsOk L a unit js = true)
    {cfg : Cfg} {args : List V} {c : Code} {w w' : World} {obs : List Obs}
    (hR : Ready a w) (hr : Sem.run cfg args w c = .ok obs w')
    (hc : (a.runCalls js).Covers w obs ∧ DevPath w w')
    (hHon : ∀ j ∈ js, L.KernelHonest a w j (unit j) hw) :
    L.Agree (devMem w'.dev) ((js.flatMap unit).foldl (fun M op => TOp.denV hw op M) (wordView (devMem w.dev))) := by
  simp only [kernelsOk, Bool.and_eq_true, List.all_eq_true] at hk
  obtain ⟨hks, hsz⟩ := hk
  have hmem : ∀ p ∈ js.map (fun j => (DevCall.kernel j, unit j)), ∃ j ∈ js, p = (.kernel j, unit j) :=
    fun p hp => by obtain ⟨j, hj, e⟩ := List.mem_map.mp hp; exact ⟨j, hj, e.symm⟩
  have hfst : (js.map fun j => (DevCall.kernel j, unit j)).map Prod.fst = js.map DevCall.kernel := by
    rw [List.map_map]; rfl
  have e := run_tape L a hB hw hw32 (js.map fun j => (DevCall.kernel j, unit j)) hR hr
    (by rw [hfst, ← runCalls_lit]; exact hc)
    (by
      rw [hfst]
      refine List.all_eq_true.mpr fun s hs => ?_
      rcases List.mem_append.mp hs with hs | hs
      · obtain ⟨j, hj, rfl⟩ := List.mem_map.mp hs
        exact (hks j hj).1.1
      · rw [List.mem_singleton.mp hs]; rfl)
    (List.all_eq_true.mpr fun p hp => by obtain ⟨j, -, rfl⟩ := hmem p hp; rfl)
    (fun p hp j' hj' => by
      obtain ⟨j, hj, rfl⟩ := hmem p hp
      exact L.unitOk_of (hks j hj).1.2)
    (List.all_eq_true.mpr fun p hp => by obtain ⟨j, hj, rfl⟩ := hmem p hp; exact (hks j hj).2)
    hsz
    (fun p hp g hg => by obtain ⟨j, -, rfl⟩ := hmem p hp; cases hg)
    (fun p hp g hg => by obtain ⟨j, -, rfl⟩ := hmem p hp; cases hg)
    (fun p hp j' hj' => by
      obtain ⟨j, hj, rfl⟩ := hmem p hp
      cases hj'
      exact hHon _ hj)
  rwa [List.flatMap_map] at e

theorem flatMap_range'_take {α : Type} (u : List α) :
    ∀ (t s : List α) (k : Nat), s.length = k →
      (List.range' k t.length).flatMap (fun i => ((s ++ t ++ u).drop i).take 1) = t
  | [], _, _, _ => rfl
  | x :: xs, s, k, hs => by
      rw [List.length_cons, List.range'_succ, List.flatMap_cons]
      have e := flatMap_range'_take u xs (s ++ [x]) (k + 1) (by simp [hs])
      simp only [List.append_assoc, List.singleton_append, List.cons_append, List.nil_append] at e ⊢
      rw [e, ← hs, List.drop_left' rfl]
      rfl

/-- **One operation a slot, in order**: slots `0` to `t.length - 1` of the
    tape `t ++ u`, each the operation at its index, are `t`. -/
theorem flatMap_range_take {α : Type} (t u : List α) :
    (List.range t.length).flatMap (fun i => ((t ++ u).drop i).take 1) = t := by
  rw [List.range_eq_range']
  simpa using flatMap_range'_take u t [] 0 rfl

/-- A step as a run takes it: a kernel's launch or a product. -/
def DevCall.ofStep : Sum Nat Gemm → DevCall
  | .inl j => .kernel j
  | .inr g => .gemm g

/-- **Taking steps in order, products at a batch of one on handle `h`
    reading scalars `s0` and `s1`, then waiting**, as steps. -/
theorem callsWith_lit (a : Drv.App) (h s0 s1 : UInt64) (ss : List (Sum Nat Gemm)) :
    a.callsWith (a.stridedRec h [s0, s1]) ss =
      .lit ((ss.map DevCall.ofStep ++ [DevCall.sync]).map (DevCall.record a h s0 s1)) := by
  simp only [Drv.App.callsWith, List.map_append, List.map_map]
  congr 2
  refine List.map_congr_left fun s _ => ?_
  cases s <;> rfl

/-- What a run taking steps `ps` then waiting needs decided: each step one
    the device reads and the step lemmas read, each launch handed exactly its
    group's buffers over its group's grid, every buffer at least the layout's
    size. -/
def stepsOk (L : TapeLayout) (a : Drv.App) (ps : List (DevCall × List TOp)) : Bool :=
  (ps.map Prod.fst ++ [DevCall.sync]).all (DevCall.okB a) && ps.all L.stepOk && ps.all (L.bindsOk a) &&
    L.sizesOk a ps

/-- Each launch's group is one the step lemmas read, decided. -/
def unitsOkB (L : TapeLayout) (ps : List (DevCall × List TOp)) : Bool :=
  ps.all fun p => match p.1 with
    | .kernel _ => L.unitOkB p.2
    | _ => true

theorem unitsOk_of {L : TapeLayout} {ps : List (DevCall × List TOp)} (h : unitsOkB L ps = true) :
    ∀ p ∈ ps, ∀ j, p.1 = .kernel j → L.UnitOk p.2 := fun p hp j hj => by
  have := List.all_eq_true.mp h p hp
  simp only [hj] at this
  exact L.unitOk_of this

/-- **Every run taking launches and products in order leaves device memory
    as the tape says**: a program in the driver's standard layout, which calls
    cuBLAS, run from a world ready for it, where every run that ends takes
    steps `ss` in order, products at a batch of one with `α = 1` and `β = 0`
    on the default handle, then waits (`Drv.App.callsWith`), the steps `ps`
    pairing them with their groups, with what `stepsOk` decides, each group
    one the step lemmas read, on a device honest about each launch and each
    product, leaves every word inside each buffer, at the size the layout
    makes it, as the groups' operations, in order, leave it from the memory
    the run started with, each contraction read as the vendor computes it. -/
theorem calls_tape (L : TapeLayout) (a : Drv.App) (hblas : a.blas = true) (hB : L.bytes.all (· ≤ 2 ^ 34) = true)
    (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (ss : List (Sum Nat Gemm))
    (ps : List (DevCall × List TOp)) (hps : ps.map Prod.fst = ss.map DevCall.ofStep) (hk : stepsOk L a ps = true)
    (hU : ∀ p ∈ ps, ∀ j, p.1 = .kernel j → L.UnitOk p.2)
    {cfg : Cfg} {args : List V} {c : Code} {w w' : World} {obs : List Obs}
    (hR : Ready a w) (hr : Sem.run cfg args w c = .ok obs w')
    (hc : (a.callsWith (a.stridedRec (drvHandle kBlas 0) [0x3f800000, 0]) ss).Covers w obs ∧ DevPath w w')
    (hv : VendorGemmHonest w.vendor) (hvk : VendorKeeps w.vendor)
    (hHon : ∀ p ∈ ps, ∀ j, p.1 = .kernel j → L.KernelHonest a w j p.2 hw) :
    L.Agree (devMem w'.dev) ((ps.flatMap Prod.snd).foldl (fun M op => TOp.denV hw op M) (wordView (devMem w.dev))) := by
  simp only [stepsOk, Bool.and_eq_true] at hk
  obtain ⟨⟨⟨hok, hst⟩, hbo⟩, hsz⟩ := hk
  exact run_tape L a hB hw hw32 ps hR hr (by rw [hps, ← callsWith_lit]; exact hc) hok hst hU hbo hsz
    (fun _ _ _ _ => hblas) (fun _ _ _ _ => ⟨hv, hvk⟩) hHon

/-! ## A captured run, replayed -/

theorem mem_flatten_replicate {α : Type} {x : α} {k : Nat} {l : List α} (h : x ∈ (List.replicate k l).flatten) :
    x ∈ l := by
  obtain ⟨l', hl', hx⟩ := List.mem_flatten.mp h
  rw [List.eq_of_mem_replicate hl'] at hx
  exact hx

theorem foldl_flatten_replicate {α β : Type} (f : β → α → β) (l : List α) :
    ∀ (k : Nat) (b : β), ((List.replicate k l).flatten).foldl f b = Nat.repeat (fun b => l.foldl f b) k b
  | 0, _ => rfl
  | k + 1, b => by
      rw [List.replicate_succ', List.flatten_append, List.flatten_singleton, List.foldl_append,
        foldl_flatten_replicate f l k b]
      rfl

theorem captureRecs_ext (a : Drv.App) (js : List Nat) (ps pg pe : UInt64) :
    (captureRecs a js ps pg pe).all (fun r => match r.callee with | .ext _ => true | _ => false) = true := by
  refine List.all_eq_true.mpr fun r hr => ?_
  simp only [captureRecs, List.mem_append, List.mem_cons, List.mem_map, List.not_mem_nil, or_false] at hr
  rcases hr with ((rfl | rfl) | ⟨j, -, rfl⟩) | rfl | rfl | rfl | rfl
  case inl.inr => cases hk : a.kernels[j]? <;> simp [Drv.App.launchRec, hk]
  all_goals rfl

/-- **A run that ends meets a triple's post**: what a triple says of every
    run of a body from a state, the run that ended leaves. -/
theorem triple_run {cfg : Cfg} {args : List V} {w : World} {c : Code} {Q : Hoare.Post} {obs : List Obs}
    {w' : World} (ht : Hoare.Triple cfg (Hoare.At args.toArray w) c Q) (hr : Sem.run cfg args w c = .ok obs w') :
    ∃ r, Q.ok r w' := by
  have hs := ht cfg.steps args.toArray w ⟨rfl, rfl⟩
  unfold Sem.run at hr
  cases hrc : runCode cfg.steps cfg args.toArray w c with
  | ok Γ w'' =>
      rw [hrc] at hr hs
      simp only [Outcome.ok.injEq] at hr
      obtain ⟨-, rfl⟩ := hr
      exact ⟨Γ, hs⟩
  | _ => rw [hrc] at hr; cases hr

theorem mapM_kernels_congr {f g : DevCall → Option DevOp} (h : ∀ j, f (.kernel j) = g (.kernel j)) :
    ∀ js : List Nat, (js.map DevCall.kernel).mapM f = (js.map DevCall.kernel).mapM g
  | [] => rfl
  | j :: js => by simp only [List.map_cons, List.mapM_cons, h j, mapM_kernels_congr h js]

theorem flatMap_replicate_units (unit : Nat → List TOp) (js : List Nat) :
    ∀ k, ((List.replicate k (js.map fun j => (DevCall.kernel j, unit j))).flatten).flatMap Prod.snd =
      (List.replicate k (js.flatMap unit)).flatten
  | 0 => rfl
  | k + 1 => by
      rw [List.replicate_succ', List.flatten_append, List.flatMap_append, flatMap_replicate_units unit js k,
        List.replicate_succ', List.flatten_append, List.flatten_singleton, List.flatten_singleton, List.flatMap_map]

/-- **Every run of a capture, then of its replays, leaves device memory as
    the tape `k` times over says**: a program in the driver's standard
    layout, from a world ready for it with no stream, graph or executable
    graph yet, every run that ends of an entry whose calls capture launches
    of kernels `js` on its first stream (`captureRecs`), with what
    `kernelsOk` decides, then every run that ends of an entry whose calls
    replay that graph `k` times and wait (`Drv.App.replayCalls`), on a
    device honest about each launch, leaves every word inside each buffer,
    at the size the layout makes it, as the kernels' groups' operations, `k`
    times over, leave it from the memory the capture started with. -/
theorem capture_replay_tape (L : TapeLayout) (a : Drv.App) (hB : L.bytes.all (· ≤ 2 ^ 34) = true)
    (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (unit : Nat → List TOp) (js : List Nat)
    (hk : kernelsOk L a unit js = true) (k : Nat) {ps pg pe : UInt64}
    {cfg : Cfg} {args : List V} {c : Code} {w0 w1 : World} {obs1 : List Obs}
    (hR : Ready a w0) (hs : w0.dev.streams.size = 0) (hg : w0.dev.graphs.size = 0) (he : w0.dev.execs.size = 0)
    (hr1 : Sem.run cfg args w0 c = .ok obs1 w1)
    (hc1 : (Tr.lit (captureRecs a js ps pg pe)).Covers w0 obs1 ∧ DevPath w0 w1)
    {cfg' : Cfg} {args' : List V} {c' : Code} {w2 : World} {obs2 : List Obs}
    (hr2 : Sem.run cfg' args' w1 c' = .ok obs2 w2)
    (hc2 : (Drv.App.replayCalls (drvHandle kExec 0) (drvHandle kStream 0) k).Covers w1 obs2 ∧ DevPath w1 w2)
    (hHon : ∀ j ∈ js, L.KernelHonest a w1 j (unit j) hw) :
    L.Agree (devMem w2.dev)
      (Nat.repeat (fun M => (js.flatMap unit).foldl (fun M op => TOp.denV hw op M) M) k (wordView (devMem w0.dev))) := by
  simp only [kernelsOk, Bool.and_eq_true, List.all_eq_true] at hk
  obtain ⟨hks, hsz⟩ := hk
  -- the capture leaves executable graph 0 holding the launches' operations
  obtain ⟨cs1, -, hh1, hx1⟩ := extRun_of hr1 hc1 (fun _ hh => hh.lit_ext (captureRecs_ext a js ps pg pe))
  have hm1 : recsMatch (captureRecs a js ps pg pe) cs1 := by cases hh1 with | lit hm => exact hm
  obtain ⟨ops, hops, hE, hcap1, hsame, hfix⟩ := extRun_capture hR.cuda hR.capture hs hg he
    (fun j hj => (hks j hj).1.1) hx1 hm1
  -- the replays take the launches' steps k times over
  have hext2 : (List.replicate k (graphRec (drvHandle kExec 0) (drvHandle kStream 0)) ++
      [streamSyncRec (drvHandle kStream 0)]).all (fun r => match r.callee with | .ext _ => true | _ => false) = true := by
    refine List.all_eq_true.mpr fun r hr => ?_
    rcases List.mem_append.mp hr with h | h
    · rw [List.eq_of_mem_replicate h]; rfl
    · rw [List.mem_singleton.mp h]; rfl
  obtain ⟨cs2, -, hh2, hx2⟩ := extRun_of hr2 hc2 (fun _ hh => hh.lit_ext hext2)
  have hm2 : recsMatch (List.replicate k (graphRec (drvHandle kExec 0) (drvHandle kStream 0)) ++
      [streamSyncRec (drvHandle kStream 0)]) cs2 := by cases hh2 with | lit hm => exact hm
  have hops' : (js.map DevCall.kernel).mapM (DevCall.op? a w1 0x3f800000 0) = some ops.toArray.toList := by
    rw [mapM_kernels_congr (g := DevCall.op? a w0 0 0) (fun j => by rw [op?_congr hsame]; rfl) js, hops]
  have hks1 : KeepsSize w1.kernel := by rw [hfix.1]; exact hR.keeps
  obtain ⟨hsteps, -⟩ := replay_steps (has_cuda (has_fixed hfix hR.cuda)) hcap1 hks1 hE (by decide) hops' k hx2 hm2
  -- the steps keep memory in step with the tape
  let ps : List (DevCall × List TOp) := (List.replicate k (js.map fun j => (DevCall.kernel j, unit j))).flatten
  have hmem : ∀ p ∈ ps, ∃ j ∈ js, p = (.kernel j, unit j) := fun p hp => by
    obtain ⟨j, hj, e⟩ := List.mem_map.mp (mem_flatten_replicate hp)
    exact ⟨j, hj, e.symm⟩
  have hfst : ps.map Prod.fst = (List.replicate k (js.map DevCall.kernel)).flatten := by
    simp only [ps, List.map_flatten, List.map_replicate, List.map_map]
    rfl
  have hall : ∀ i, i < a.bufs.length → ∃ b, (w1.dev.bufs[i]?).join = some b ∧ b.size = a.bufs.getD i 0 := by
    intro i hi; rw [hsame.bufs]; exact hR.allocs i hi
  have e := TapeLayout.steps_tape hB a hw hw32 ps
    (by rw [hfst]; exact .snoc hsteps (rfl : devMem w2.dev = devMem w2.dev)) hks1
    (fun p hp g hg => by obtain ⟨j, -, rfl⟩ := hmem p hp; cases hg)
    (List.all_eq_true.mpr fun p hp => by obtain ⟨j, -, rfl⟩ := hmem p hp; rfl)
    (fun p hp j' hj' => by obtain ⟨j, hj, rfl⟩ := hmem p hp; exact L.unitOk_of (hks j hj).1.2)
    (List.all_eq_true.mpr fun p hp => by obtain ⟨j, hj, rfl⟩ := hmem p hp; exact (hks j hj).2)
    (List.all_eq_true.mpr fun p hp => by
      obtain ⟨j, hj, rfl⟩ := hmem p hp
      exact List.all_eq_true.mp hsz _ (List.mem_map.mpr ⟨j, hj, rfl⟩))
    hall
    (fun p hp j' hj' => by
      obtain ⟨j, hj, rfl⟩ := hmem p hp
      cases hj'
      exact hHon _ hj) (fun _ _ _ => rfl)
  rw [devMem_congr hsame.bufs] at e
  rw [flatMap_replicate_units, foldl_flatten_replicate] at e
  exact e

/-! ## A range run, then captured over many streams, replayed -/

theorem dagRecs_ext (a : Drv.App) (sc : List UInt64) (cc : List CapCall) (s pg pe : UInt64) :
    (dagRecs a sc cc s pg pe).all (fun r => match r.callee with | .ext _ => true | _ => false) = true := by
  refine List.all_eq_true.mpr fun r hr => ?_
  simp only [dagRecs, List.mem_append, List.mem_cons, List.mem_map, List.not_mem_nil, or_false] at hr
  rcases hr with (rfl | ⟨c, -, rfl⟩) | rfl | rfl | rfl | rfl
  case inl.inr =>
    cases c with
    | launch j s => cases hk : a.kernels[j]? <;> simp [CapCall.call, Drv.App.launchRec, hk]
    | _ => rfl
  all_goals rfl

theorem flatMap_flatten_replicate {α β : Type} (f : α → List β) (l : List α) :
    ∀ k, ((List.replicate k l).flatten).flatMap f = (List.replicate k (l.flatMap f)).flatten
  | 0 => rfl
  | k + 1 => by
      rw [List.replicate_succ', List.flatten_append, List.flatMap_append, flatMap_flatten_replicate f l k,
        List.replicate_succ', List.flatten_append, List.flatten_singleton, List.flatten_singleton]

theorem stepsOk_parts {L : TapeLayout} {a : Drv.App} {ps : List (DevCall × List TOp)} (hk : stepsOk L a ps = true) :
    (ps.map Prod.fst ++ [DevCall.sync]).all (DevCall.okB a) = true ∧ ps.all L.stepOk = true ∧
      ps.all (L.bindsOk a) = true ∧ L.sizesOk a ps = true := by
  simp only [stepsOk, Bool.and_eq_true] at hk
  exact ⟨hk.1.1.1, hk.1.1.2, hk.1.2, hk.2⟩

/-- **Every run of a range then its capture over many streams, then of its
    replays, leaves device memory as the tape says**: a program in the
    driver's standard layout, from a world ready for it with no graph or
    executable graph yet, every run that ends of an entry whose calls take
    steps `ps` and wait for the default stream, then capture on stream `s` calls `cc` that
    `capSim` reads, from `s` alone joined, as steps `qs` (`dagRecs`), each
    with what `stepsOk` decides, then every run that ends of an entry whose
    calls replay that graph `k` times on `s` and wait, on a device honest
    about each launch and each product, leaves every word inside each
    buffer, at the size the layout makes it, as `ps`' operations and then
    `qs`' `k` times over leave it from the memory the first run started
    with. -/
theorem dag_replay_tape (L : TapeLayout) (a : Drv.App) (hblas : a.blas = true) (hB : L.bytes.all (· ≤ 2 ^ 34) = true)
    (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (ps qs : List (DevCall × List TOp))
    (hk : stepsOk L a ps = true) (hU : ∀ p ∈ ps, ∀ j, p.1 = .kernel j → L.UnitOk p.2)
    (hkq : stepsOk L a qs = true) (hUq : ∀ p ∈ qs, ∀ j, p.1 = .kernel j → L.UnitOk p.2)
    (cc : List CapCall) (bs : UInt64 → Option UInt64) {s pg pe : UInt64}
    (hsim : capSim a bs [s] [] cc = some (qs.map Prod.fst)) (k : Nat)
    {cfg : Cfg} {args : List V} {c : Code} {w0 w1 : World} {obs1 : List Obs}
    (hR : Ready a w0) (hbs : ∀ h s', bs h = some s' → w0.dev.blasParty? h = w0.dev.streamParty? s')
    (hg : w0.dev.graphs.size = 0) (he : w0.dev.execs.size = 0)
    (hr1 : Sem.run cfg args w0 c = .ok obs1 w1)
    (hc1 : (Tr.lit ((ps.map Prod.fst).map (DevCall.record a (drvHandle kBlas 0) 0x3f800000 0) ++
      [wordRec .streamSynchronize [0]] ++ dagRecs a [0x3f800000, 0] cc s pg pe)).Covers w0 obs1 ∧ DevPath w0 w1)
    {cfg' : Cfg} {args' : List V} {c' : Code} {w2 : World} {obs2 : List Obs}
    (hr2 : Sem.run cfg' args' w1 c' = .ok obs2 w2)
    (hc2 : (Drv.App.replayCalls (drvHandle kExec 0) s k).Covers w1 obs2 ∧ DevPath w1 w2)
    (hv : VendorGemmHonest w0.vendor) (hvk : VendorKeeps w0.vendor)
    (hHon : ∀ p ∈ ps, ∀ j, p.1 = .kernel j → L.KernelHonest a w0 j p.2 hw)
    (hHon1 : ∀ p ∈ qs, ∀ j, p.1 = .kernel j → L.KernelHonest a w1 j p.2 hw) :
    L.Agree (devMem w2.dev)
      (Nat.repeat (fun M => (qs.flatMap Prod.snd).foldl (fun M op => TOp.denV hw op M) M) k
        ((ps.flatMap Prod.snd).foldl (fun M op => TOp.denV hw op M) (wordView (devMem w0.dev)))) := by
  obtain ⟨hcu, hcap, hks, hbl0, halloc⟩ := hR
  have hbl : w0.has (Ext.cublas .sgemmStridedBatched).lib = true := hbl0 hblas
  obtain ⟨hok, hst, hbo, hsz⟩ := stepsOk_parts hk
  obtain ⟨hokq, hstq, hboq, hszq⟩ := stepsOk_parts hkq
  -- the first run: the range's steps, then the capture
  have hext : ((ps.map Prod.fst).map (DevCall.record a (drvHandle kBlas 0) 0x3f800000 0) ++
      [wordRec .streamSynchronize [0]] ++ dagRecs a [0x3f800000, 0] cc s pg pe).all
        (fun r => match r.callee with | .ext _ => true | _ => false) = true := by
    rw [List.all_append, List.all_append]
    exact Bool.and_eq_true_iff.mpr ⟨Bool.and_eq_true_iff.mpr ⟨record_ext _ _ _ _ _, rfl⟩, dagRecs_ext _ _ _ _ _ _⟩
  obtain ⟨cs1, -, hh1, hx1⟩ := extRun_of hr1 hc1 (fun _ hh => hh.lit_ext hext)
  have hm1 : recsMatch ((ps.map Prod.fst).map (DevCall.record a (drvHandle kBlas 0) 0x3f800000 0) ++
      [wordRec .streamSynchronize [0]] ++ dagRecs a [0x3f800000, 0] cc s pg pe) cs1 := by
    cases hh1 with | lit hm => exact hm
  obtain ⟨cps, cd, rfl, hmps, hmd⟩ := recsMatch_split hm1
  obtain ⟨cp, csy, rfl, hmp, hmsy⟩ := recsMatch_split hmps
  obtain ⟨wm, hxps, hxd⟩ := hx1.split _ _ rfl
  obtain ⟨wp, hxp, hxsy⟩ := hxps.split _ _ rfl
  obtain ⟨hsp, hsame0, hfp⟩ := extRun_steps hcu hcap hks hxp _ hmp
    (fun s hs => DevCall.okB_ok (List.all_eq_true.mp hok s (List.mem_append_left _ hs))) (fun _ _ => hbl)
  have hA0 := TapeLayout.steps_tape hB a hw hw32 ps (w' := wp) (.snoc hsp rfl) hks
    (fun _ _ _ _ => ⟨hv, hvk⟩) hst hU hbo hsz halloc hHon (fun _ _ _ => rfl)
  -- the host waits for the default stream
  obtain ⟨x1, y1, vs1, r1, ex1, fx1, -, hx1', ey1, fy1⟩ := hxsy.one_word hmsy
  obtain ⟨⟨rc1, hd1⟩, -, f1⟩ := streamSyncRec_done (has_cuda (has_fixed (hfp.trans fx1) hcu))
    (by rw [ex1, hsame0.capture]; exact hcap) hx1'
  have hdm : wm.dev = { wp.dev with race := rc1 } := by rw [ey1, hd1, ex1]
  have hfm : Fixed w0 wm := hfp.trans (fx1.trans (f1.2.trans fy1))
  have hsame : DevSame w0.dev wm.dev := by
    rw [hdm]
    exact ⟨hsame0.live, hsame0.streams, hsame0.capture, hsame0.events, hsame0.modules, hsame0.funcs,
      hsame0.drvInit, hsame0.retained, hsame0.size, hsame0.blas, hsame0.graphs, hsame0.execs, hsame0.sizes,
      hsame0.kernelWrites⟩
  have hA1 : L.Agree (devMem wm.dev) ((ps.flatMap Prod.snd).foldl (fun M op => TOp.denV hw op M)
      (wordView (devMem w0.dev))) := by
    rw [hdm]; exact hA0
  -- the capture leaves executable graph 0 holding `qs`' operations
  have hbsm : ∀ h s', bs h = some s' → wm.dev.blasParty? h = wm.dev.streamParty? s' := fun h s' e => by
    rw [blasParty_congr hsame.streams hsame.blas, streamParty_congr hsame.streams]; exact hbs h s' e
  obtain ⟨ops, hops, hE, hcap1, hsame1, hf1⟩ := extRun_dagCapture (has_fixed hfm hcu) (has_fixed hfm hbl) hbsm
    (by rw [hsame.capture]; exact hcap) (by rw [hsame.graphs]; exact hg) (by rw [hsame.execs]; exact he) hsim hxd hmd
  -- the replays take `qs`' steps `k` times over
  have hext2 : (List.replicate k (graphRec (drvHandle kExec 0) s) ++ [streamSyncRec s]).all
      (fun r => match r.callee with | .ext _ => true | _ => false) = true := by
    refine List.all_eq_true.mpr fun r hr => ?_
    rcases List.mem_append.mp hr with h | h
    · rw [List.eq_of_mem_replicate h]; rfl
    · rw [List.mem_singleton.mp h]; rfl
  obtain ⟨cs2, -, hh2, hx2⟩ := extRun_of hr2 hc2 (fun _ hh => hh.lit_ext hext2)
  have hm2 : recsMatch (List.replicate k (graphRec (drvHandle kExec 0) s) ++ [streamSyncRec s]) cs2 := by
    cases hh2 with | lit hm => exact hm
  have hfix1 : Fixed w0 w1 := hfm.trans hf1
  have hks1 : KeepsSize w1.kernel := by rw [hfix1.1]; exact hks
  have hops' : (qs.map Prod.fst).mapM (DevCall.op? a w1 0x3f800000 0) = some ops.toArray.toList := by
    rw [show DevCall.op? a w1 0x3f800000 0 = DevCall.op? a wm 0x3f800000 0 from funext (op?_congr hsame1)]
    simpa using hops
  obtain ⟨hsteps, -⟩ := replay_steps (has_cuda (has_fixed hfix1 hcu)) hcap1 hks1 hE (by decide) hops' k hx2 hm2
  -- the replays' steps keep memory in step with the tape, from where the first run left it
  have hdm1 : devMem w1.dev = devMem wm.dev := devMem_congr hsame1.bufs
  have hall : ∀ i, i < a.bufs.length → ∃ b, (w1.dev.bufs[i]?).join = some b ∧ b.size = a.bufs.getD i 0 := by
    intro i hi
    obtain ⟨b, hb, hbs'⟩ := halloc i hi
    have hz := hsame.sizes i
    simp only [Dev.get?, Int.natCast_nonneg, decide_true, Int.toNat_natCast] at hz
    have hz' : ((wm.dev.bufs[i]?).join).map ByteArray.size = some b.size := by
      have : ((wm.dev.bufs[(i : Int).toNat]?).join).map ByteArray.size = ((w0.dev.bufs[(i : Int).toNat]?).join).map ByteArray.size := by
        simpa [Dev.get?] using hsame.sizes i
      simpa [hb] using this
    obtain ⟨b', hb', hbs2⟩ := Option.map_eq_some_iff.mp hz'
    exact ⟨b', by rw [hsame1.bufs]; exact hb', by rw [hbs2, hbs']⟩
  let rs : List (DevCall × List TOp) := (List.replicate k qs).flatten
  have hrs : rs.map Prod.fst = (List.replicate k (qs.map Prod.fst)).flatten := by
    simp only [rs, List.map_flatten, List.map_replicate]
  have hA2 : L.Agree (devMem w1.dev) ((ps.flatMap Prod.snd).foldl (fun M op => TOp.denV hw op M)
      (wordView (devMem w0.dev))) := by rw [hdm1]; exact hA1
  have e := TapeLayout.steps_tape hB a hw hw32 rs
    (by rw [hrs]; exact .snoc hsteps (rfl : devMem w2.dev = devMem w2.dev)) hks1
    (fun _ _ _ _ => by rw [hfix1.2.1]; exact ⟨hv, hvk⟩)
    (List.all_eq_true.mpr fun p hp => List.all_eq_true.mp hstq p (mem_flatten_replicate hp))
    (fun p hp => hUq p (mem_flatten_replicate hp))
    (List.all_eq_true.mpr fun p hp => List.all_eq_true.mp hboq p (mem_flatten_replicate hp))
    (by
      simp only [TapeLayout.sizesOk, List.all_eq_true] at hszq ⊢
      exact fun p hp => hszq p (mem_flatten_replicate hp))
    hall (fun p hp => hHon1 p (mem_flatten_replicate hp)) hA2
  rw [flatMap_flatten_replicate, foldl_flatten_replicate] at e
  exact e

end AlgorithmLib.ML
