module
public import AlgorithmLib.ML.Launch.HostBridge
meta import AlgorithmLib.ML.Launch.HostBridge
public import AlgorithmLib.Check.Trace
meta import AlgorithmLib.Check.Trace
public import AlgorithmLib.Check.DrvState
meta import AlgorithmLib.Check.DrvState
import all Init.Data.Repr
@[expose] public section

/-!
# The launches a run makes, read from its calls

A program in the driver's standard layout (`Drv.App`) launches kernel `j`
by calling `cuLaunchKernel` with `j`'s function handle, its grid and block,
and its parameter array, and a launch is observed with the parameter values
it reads. `appLaunchOfCall` reads a launch off such a call: the kernel its
handle names, handed the allocations the observed values point at.
`Drv.App.runCalls a js` is the call pattern of launching the kernels `js`
in order on the default stream, each reading pointers to the allocations
it is declared to be handed, and then waiting; where the checker shows a
body's runs make exactly those calls (`check_calls`), `launchesOf_runCalls`
reads the launches those runs made as `js.map (appLaunch a)`: the pipeline
a theorem about the kernels states is the one the shipped code runs, read
from its calls rather than written down.
-/

namespace AlgorithmLib.ML

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Check

/-- The call launching kernel `j` of a program in the standard layout on the
    default stream: its function handle, grid, block, no shared memory, the
    default stream, its parameter array, and no extra arguments, observed
    reading the pointers to the allocations it is handed. -/
def _root_.AlgorithmLib.Drv.App.launchRec (a : Drv.App) (j : Nat) : CallRec :=
  match a.kernels[j]? with
  | some k =>
      let (gx, gy, gz) := k.grid
      let (bx, by_, bz) := k.block
      { callee := .ext (.cuda .launchKernel),
        args := [.is (drvHandle kFunc j), .is (UInt64.ofNat gx), .is (UInt64.ofNat gy), .is (UInt64.ofNat gz),
                 .is (UInt64.ofNat bx), .is (UInt64.ofNat by_), .is (UInt64.ofNat bz), .is 0, .is 0,
                 .is (Drv.App.at_ (a.paramsOff j)), .is 0] ++ k.args.map fun i => .is (devPtr i) }
  | none => { callee := .ext (.cuda .launchKernel), args := [] }

/-- **Launching kernels `js` in order, then waiting**: the calls a body that
    does that makes. -/
def _root_.AlgorithmLib.Drv.App.runCalls (a : Drv.App) (js : List Nat) : Tr :=
  .lit (js.map a.launchRec ++ [⟨.ext (.cuda .ctxSynchronize), []⟩])

/-- The device write a call is, for a program in the standard layout: a
    `cuLaunchKernel` handed kernel `j`'s function handle is kernel `j`'s
    launch, handed the buffers the parameter values it was observed reading
    name; a `cublasSgemm` is a product and a `cuGraphLaunch` a replay, each
    handed what its words name; any other call writes nothing a pipeline
    reads. -/
def appLaunchOfCall (a : Drv.App) : Callee × List V → List DeviceOp
  | (.ext (.cublas .sgemm), vs) => [gemmOp a (vs.map fun x => (asBits x).getD 0)]
  | (.ext (.cuda .graphLaunch), vs) => [graphOp a (vs.map fun x => (asBits x).getD 0)]
  | (.ext (.cuda .launchKernel), v :: rest) =>
      match asBits v with
      | some h => match handleOf kFunc h with
        | some j => if j < a.kernels.length then
            [appLaunchWith a j ((rest.drop 10).map fun x => wordBuf a ((asBits x).getD 0))] else []
        | none => []
      | none => []
  | _ => []

/-- The launches a sequence of calls makes, in order. -/
def appLaunchesOf (a : Drv.App) (cs : List (Callee × List V)) : List DeviceOp := cs.flatMap (appLaunchOfCall a)

/-- The call `Drv.App.sgemm` makes for `g`. -/
def _root_.AlgorithmLib.Drv.App.gemmRec (a : Drv.App) (g : Gemm) : CallRec :=
  { callee := .ext (.cublas .sgemm), args := (a.gemmWords g).map .is }

/-- The call a step makes. -/
def _root_.AlgorithmLib.Drv.App.stepRec (a : Drv.App) : Sum Nat Gemm → CallRec
  | .inl j => a.launchRec j
  | .inr g => a.gemmRec g

/-- **Taking `ss` in order, then waiting**: the calls a body that does that
    makes. -/
def _root_.AlgorithmLib.Drv.App.mixedCalls (a : Drv.App) (ss : List (Sum Nat Gemm)) : Tr :=
  .lit (ss.map a.stepRec ++ [⟨.ext (.cuda .ctxSynchronize), []⟩])

theorem handleOf_drvHandle {kind j : Nat} (hk : kind < 2 ^ 8) (hj : j < 2 ^ 48) :
    handleOf kind (drvHandle kind j) = some j := by
  unfold handleOf drvHandle
  have : (UInt64.ofNat (2 ^ 56 + kind * 2 ^ 48 + j + 1)).toNat = 2 ^ 56 + kind * 2 ^ 48 + j + 1 := by
    simp; omega
  simp only [this]
  split
  · congr 1; omega
  · rename_i h; simp at h; omega

theorem devPtrOf_devPtr {i : Nat} (hi : i < 2 ^ 16) : devPtrOf (devPtr i) = some (i, 0) := by
  unfold devPtrOf devPtr
  have : (UInt64.ofNat (2 ^ 52 + i * 2 ^ 36)).toNat = 2 ^ 52 + i * 2 ^ 36 := by simp; omega
  simp only [this]
  have h1 : 2 ^ 52 ≤ 2 ^ 52 + i * 2 ^ 36 := by omega
  have h2 : 2 ^ 52 + i * 2 ^ 36 < 2 ^ 53 := by omega
  simp only [h1, h2, decide_true, Bool.and_self, if_true, Nat.add_sub_cancel_left, Option.some.injEq,
    Prod.mk.injEq]
  exact ⟨Nat.mul_div_cancel i (by decide), Nat.mul_mod_left i _⟩

theorem argsOk_drop : ∀ {xs ys : List ArgPat} {vs : List V}, argsOk (xs ++ ys) vs →
    argsOk ys (vs.drop xs.length)
  | [], _, _, h => h
  | _ :: xs, ys, _ :: vs, ⟨_, h⟩ => by
      show argsOk ys (vs.drop xs.length)
      exact argsOk_drop h
  | _ :: _, _, [], h => h.elim

theorem wordBuf_of {a : Drv.App} {x : UInt64} {i off : Nat} (h : devPtrOf x = some (i, off)) (h1 : i < a.bufs.length) :
    wordBuf a x = .near (Int.ofNat (a.bufSlot i)) := by
  unfold wordBuf
  rw [h]
  simp only [h1, if_true]

theorem wordBuf_devPtr {a : Drv.App} {i : Nat} (h1 : i < a.bufs.length) (h2 : i < 2 ^ 16) :
    wordBuf a (devPtr i) = .near (Int.ofNat (a.bufSlot i)) :=
  wordBuf_of (devPtrOf_devPtr h2) h1

/-- Pointers to the starts of allocations name them. -/
theorem words_bufs {a : Drv.App} {ks : List Nat} {ws : List V}
    (hk : ∀ i ∈ ks, i < a.bufs.length ∧ i < 2 ^ 16) (h : argsOk (ks.map fun i => ArgPat.is (devPtr i)) ws) :
    ws.map (fun x => wordBuf a ((asBits x).getD 0)) = ks.map fun i => BufDesc.near (Int.ofNat (a.bufSlot i)) := by
  induction ks generalizing ws with
  | nil =>
      cases ws with
      | nil => rfl
      | cons _ _ => exact h.elim
  | cons i ks ih =>
      cases ws with
      | nil => exact h.elim
      | cons w ws =>
          obtain ⟨hw, h⟩ := h
          have hi := hk i List.mem_cons_self
          have hw' : asBits w = some (devPtr i) := hw
          rw [List.map_cons, List.map_cons, hw', Option.getD_some, wordBuf_devPtr hi.1 hi.2,
            ih (fun j hj => hk j (List.mem_cons_of_mem _ hj)) h]

theorem appLaunchOfCall_launch {a : Drv.App} {j : Nat} (hj : j < a.kernels.length) (hjb : j < 2 ^ 48)
    (hb : ∀ i ∈ (a.kernels[j]'hj).args, i < a.bufs.length ∧ i < 2 ^ 16) {x : Callee × List V}
    (hm : (a.launchRec j).Matches x) : appLaunchOfCall a x = [appLaunch a j] := by
  obtain ⟨c, vs⟩ := x
  unfold Drv.App.launchRec at hm
  have hk := List.getElem?_eq_getElem hj
  rw [hk] at hm
  obtain ⟨hc, ha⟩ := hm
  simp only at hc ha
  subst hc
  match vs, ha with
  | v :: rest, ⟨hv, hrest⟩ =>
      simp only [argOk] at hv
      have hw := argsOk_drop (xs := [_, _, _, _, _, _, _, _, _, _]) hrest
      simp only [List.length_cons, List.length_nil] at hw
      simp only [appLaunchOfCall, hv, handleOf_drvHandle (by decide : kFunc < 2 ^ 8) (by omega), hj, if_true,
        appLaunch, appBufs, hk, Option.map_some, Option.getD_some, words_bufs hb hw]

theorem appLaunchOfCall_sync {a : Drv.App} {x : Callee × List V}
    (hm : (⟨.ext (.cuda .ctxSynchronize), []⟩ : CallRec).Matches x) : appLaunchOfCall a x = [] := by
  obtain ⟨c, vs⟩ := x
  obtain ⟨hc, -⟩ := hm
  simp only at hc
  subst hc
  rfl

/-- What a kernel of a program in the standard layout must be for its
    observed launches to be read: an index a handle can carry, handed
    allocations whose pointers name them. -/
def _root_.AlgorithmLib.Drv.App.readable (a : Drv.App) (j : Nat) : Bool :=
  decide (j < a.kernels.length) && decide (j < 2 ^ 48) &&
    (a.kernels[j]?).all fun k => k.args.all fun i => decide (i < a.bufs.length) && decide (i < 2 ^ 16)

theorem readable_spec {a : Drv.App} {j : Nat} (h : a.readable j = true) :
    ∃ hj : j < a.kernels.length, j < 2 ^ 48 ∧ ∀ i ∈ (a.kernels[j]'hj).args, i < a.bufs.length ∧ i < 2 ^ 16 := by
  simp only [Drv.App.readable, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨hj, hjb⟩, hk⟩ := h
  refine ⟨hj, hjb, fun i hi => ?_⟩
  rw [List.getElem?_eq_getElem hj] at hk
  simp only [Option.all_some, List.all_eq_true, Bool.and_eq_true, decide_eq_true_eq] at hk
  exact hk i hi

theorem appLaunchesOf_launches {a : Drv.App} :
    ∀ {js : List Nat} {cs : List (Callee × List V)}, (∀ j ∈ js, a.readable j = true) →
      recsMatch (js.map a.launchRec ++ [⟨.ext (.cuda .ctxSynchronize), []⟩]) cs →
      appLaunchesOf a cs = js.map (appLaunch a)
  | [], [x], _, ⟨hm, _⟩ => by simp [appLaunchesOf, appLaunchOfCall_sync hm]
  | [], [], _, h => h.elim
  | [], _ :: _ :: _, _, ⟨_, h⟩ => h.elim
  | j :: js, x :: cs, hj, ⟨hm, h⟩ => by
      have e := appLaunchesOf_launches (js := js) (fun i hi => hj i (List.mem_cons_of_mem _ hi)) h
      simp only [appLaunchesOf, List.flatMap_cons, List.map_cons] at e ⊢
      obtain ⟨h1, h2, h3⟩ := readable_spec (hj j List.mem_cons_self)
      rw [appLaunchOfCall_launch h1 h2 h3 hm, e]; rfl
  | _ :: _, [], _, h => h.elim

/-- **The launches a run of `runCalls` makes are its kernels' launches.** -/
theorem launchesOf_runCalls {a : Drv.App} {js : List Nat}
    (hj : js.all a.readable = true)
    {cs : List (Callee × List V)} (h : (a.runCalls js).Has cs) : appLaunchesOf a cs = js.map (appLaunch a) := by
  cases h with
  | lit hm => exact appLaunchesOf_launches (fun j hm' => List.all_eq_true.mp hj j hm') hm

/-- Values matched word for word are those words. -/
theorem words_exact : ∀ {ws : List UInt64} {vs : List V}, argsOk (ws.map ArgPat.is) vs →
    vs.map (fun x => (asBits x).getD 0) = ws
  | [], [], _ => rfl
  | [], _ :: _, h => h.elim
  | _ :: _, [], h => h.elim
  | w :: ws, v :: vs, ⟨hv, h⟩ => by
      have hv' : asBits v = some w := hv
      rw [List.map_cons, hv', Option.getD_some, words_exact h]

theorem appLaunchOfCall_gemm {a : Drv.App} {g : Gemm} {x : Callee × List V}
    (hm : (a.gemmRec g).Matches x) : appLaunchOfCall a x = [appGemm a g] := by
  obtain ⟨c, vs⟩ := x
  obtain ⟨hc, ha⟩ := hm
  simp only [Drv.App.gemmRec] at hc ha
  subst hc
  simp only [appLaunchOfCall, appGemm, words_exact ha]

/-- What a step must be for its observed call to be read: a readable kernel,
    or a product. -/
def _root_.AlgorithmLib.Drv.App.stepReadable (a : Drv.App) : Sum Nat Gemm → Bool
  | .inl j => a.readable j
  | .inr _ => true

theorem appLaunchesOf_steps {a : Drv.App} :
    ∀ {ss : List (Sum Nat Gemm)} {cs : List (Callee × List V)}, (∀ s ∈ ss, a.stepReadable s = true) →
      recsMatch (ss.map a.stepRec ++ [⟨.ext (.cuda .ctxSynchronize), []⟩]) cs →
      appLaunchesOf a cs = ss.map (appStep a)
  | [], [x], _, ⟨hm, _⟩ => by simp [appLaunchesOf, appLaunchOfCall_sync hm]
  | [], [], _, h => h.elim
  | [], _ :: _ :: _, _, ⟨_, h⟩ => h.elim
  | s :: ss, x :: cs, hs, ⟨hm, h⟩ => by
      have e := appLaunchesOf_steps (ss := ss) (fun i hi => hs i (List.mem_cons_of_mem _ hi)) h
      simp only [appLaunchesOf, List.flatMap_cons, List.map_cons] at e ⊢
      have hr := hs s List.mem_cons_self
      cases s with
      | inl j =>
          obtain ⟨h1, h2, h3⟩ := readable_spec hr
          rw [appLaunchOfCall_launch h1 h2 h3 hm, e]; rfl
      | inr g => rw [appLaunchOfCall_gemm hm, e]; rfl
  | _ :: _, [], _, h => h.elim

/-- **The device writes a run of `mixedCalls` makes are its steps' records.** -/
theorem launchesOf_mixedCalls {a : Drv.App} {ss : List (Sum Nat Gemm)}
    (hs : ss.all a.stepReadable = true)
    {cs : List (Callee × List V)} (h : (a.mixedCalls ss).Has cs) : appLaunchesOf a cs = ss.map (appStep a) := by
  cases h with
  | lit hm => exact appLaunchesOf_steps (fun s hm' => List.all_eq_true.mp hs s hm') hm

/-- **Replaying executable graph `e` on stream `s` `k` times, then waiting for
    the stream**: the calls a body that does that makes. -/
def _root_.AlgorithmLib.Drv.App.replayCalls (e s : UInt64) (k : Nat) : Tr :=
  .lit (List.replicate k ⟨.ext (.cuda .graphLaunch), [.is e, .is s]⟩ ++
    [⟨.ext (.cuda .streamSynchronize), [.is s]⟩])

theorem appLaunchesOf_replays {a : Drv.App} {e s : UInt64} :
    ∀ {k : Nat} {cs : List (Callee × List V)},
      recsMatch (List.replicate k ⟨.ext (.cuda .graphLaunch), [.is e, .is s]⟩ ++
        [⟨.ext (.cuda .streamSynchronize), [.is s]⟩]) cs →
      appLaunchesOf a cs = List.replicate k (graphOp a [e, s])
  | 0, [⟨c, vs⟩], ⟨⟨hc, _⟩, _⟩ => by
      simp only at hc; subst hc; rfl
  | 0, [], h => h.elim
  | 0, _ :: _ :: _, ⟨_, h⟩ => h.elim
  | k + 1, ⟨c, vs⟩ :: cs, ⟨⟨hc, ha⟩, h⟩ => by
      simp only at hc ha; subst hc
      have e' := appLaunchesOf_replays (a := a) (k := k) h
      simp only [appLaunchesOf, List.flatMap_cons, List.replicate_succ] at e' ⊢
      rw [e']
      simp only [appLaunchOfCall, words_exact (ws := [e, s]) ha, List.singleton_append]
  | _ + 1, [], h => h.elim

/-- **The device writes a run of `replayCalls` makes are its replays.** -/
theorem launchesOf_replayCalls {a : Drv.App} {e s : UInt64} {k : Nat}
    {cs : List (Callee × List V)} (h : (Drv.App.replayCalls e s k).Has cs) :
    appLaunchesOf a cs = List.replicate k (graphOp a [e, s]) := by
  cases h with
  | lit hm => exact appLaunchesOf_replays hm

end AlgorithmLib.ML
