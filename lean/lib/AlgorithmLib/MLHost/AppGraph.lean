module
public import AlgorithmLib.MLHost.AppRun
meta import AlgorithmLib.MLHost.AppRun
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# A replayed graph, as steps of device memory

An executable graph holds device operations, each a kernel's launch or a
vendor routine, in the order a capture recorded them, and launching it runs
them in that order (`Dev.runGraph`). So what launching it does to device
memory is those operations' steps in order (`runGraph_mem`), and where a
graph holds the operations steps `ss` stand for (`DevCall.op?`), a run that
replays it `k` times and waits for the stream takes those steps `k` times
over (`extRun_replay`, `replay_steps`).
-/

namespace AlgorithmLib.ML

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Contracts
  AlgorithmLib.Check AlgorithmLib.Device

/-- What a device operation does to device memory: a launch's step or a
    vendor routine's. -/
def opStep (k : Launch → List ByteArray → List ByteArray) (v : VendorCall → List ByteArray → ByteArray) :
    DevMem → DevOp → DevMem
  | m, .launch l ids => (launchStep k l ids).run m
  | m, .vendor c ins out => (vendorStep v c ins out).run m

theorem foldlM_ops_mem {w : World} (hk : KeepsSize w.kernel) :
    ∀ (ops : List DevOp) (d d' : Dev),
      ops.foldlM (fun d op => match op with
        | .launch l ids => d.runLaunch w.kernel l ids
        | .vendor c ins out => d.runVendor w.vendor c ins out) d = some d' →
      devMem d' = ops.foldl (opStep w.kernel w.vendor) (devMem d) ∧ DevSame d d'
  | [], d, d', h => by
      simp only [List.foldlM_nil, pure, Option.some.injEq] at h
      subst h
      exact ⟨rfl, DevSame.refl _⟩
  | op :: ops, d, d', h => by
      simp only [List.foldlM_cons, Option.bind_eq_bind] at h
      obtain ⟨d1, h1, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨e, hs⟩ := foldlM_ops_mem hk ops d1 d' h
      cases op with
      | launch l ids =>
          refine ⟨by rw [e, List.foldl_cons, runLaunch_mem _ _ _ _ _ h1]; rfl,
            (DevSame.of_bufsOnly (runLaunch_bufsOnly hk h1)).trans hs⟩
      | vendor c ins out =>
          refine ⟨by rw [e, List.foldl_cons, runVendor_mem _ _ _ _ _ _ h1]; rfl,
            (DevSame.of_bufsOnly (runVendor_bufsOnly h1)).trans hs⟩

/-- **Launching a graph is its operations' steps**, in the order it holds
    them, the device otherwise as it was. -/
theorem runGraph_mem {d d' : Dev} {w : World} {ops : Array DevOp} {p : Nat} (hk : KeepsSize w.kernel)
    (h : d.runGraph w ops p = some d') :
    devMem d' = ops.toList.foldl (opStep w.kernel w.vendor) (devMem d) ∧ DevSame d d' := by
  unfold Dev.runGraph at h
  simp only [Option.bind_eq_bind] at h
  obtain ⟨r, -, h⟩ := Option.bind_eq_some_iff.mp h
  rw [← Array.foldlM_toList] at h
  obtain ⟨e, hs⟩ := foldlM_ops_mem hk ops.toList _ d' h
  exact ⟨e, (DevSame.race d r).trans hs⟩

/-- The record of launching executable graph `e` on stream `s`. -/
def graphRec (e s : UInt64) : CallRec := ⟨.ext (.cuda .graphLaunch), [.is e, .is s]⟩

/-- The record of waiting for stream `s`. -/
def streamSyncRec (s : UInt64) : CallRec := ⟨.ext (.cuda .streamSynchronize), [.is s]⟩

theorem replayCalls_lit (e s : UInt64) (k : Nat) :
    Drv.App.replayCalls e s k = .lit (List.replicate k (graphRec e s) ++ [streamSyncRec s]) := rfl

/-- **A launch of graph `ei` that answered is its operations' steps.** -/
theorem graphRec_done {w1 w2 : World} {vs : List V} {r : Option V} {ei : Nat} {s : UInt64}
    {ops : Array DevOp} (hhas : w1.has (Ext.cuda .graphLaunch).lib = true) (hcap : w1.dev.capture = none)
    (hks : KeepsSize w1.kernel) (hE : w1.dev.execs[ei]? = some (some ops)) (hei : ei < 2 ^ 48)
    (hm : (graphRec (drvHandle kExec ei) s).Matches (.ext (.cuda .graphLaunch), vs ++ callWords w1 (.ext (.cuda .graphLaunch)) vs))
    (hx : extCall (.cuda .graphLaunch) vs (obsCall w1 (.ext (.cuda .graphLaunch)) vs) = some (r, w2)) :
    devMem w2.dev = ops.toList.foldl (opStep w1.kernel w1.vendor) (devMem w1.dev) ∧ CallKeeps w1 w2 := by
  obtain ⟨-, ha⟩ := hm
  obtain ⟨bits, hbits, hcd⟩ := extCall_cuda (w := obsCall w1 _ vs) hhas hx
  have hcap' : (obsCall w1 (.ext (.cuda .graphLaunch)) vs).dev.capture = none := hcap
  rw [cudaDrv_step hcap'] at hcd
  have hl : vs.length = 2 := by
    rw [← mapM_asBits_length hbits]
    simp only [cudaStep] at hcd
    split at hcd
    · rfl
    · cases hcd
  obtain ⟨h2, -⟩ := argsOk_split (os := [_, _]) (by simpa using hl) ha
  rw [mapM_asBits_of_argsOk (ws := [drvHandle kExec ei, s]) h2] at hbits
  cases hbits
  simp only [cudaStep] at hcd
  split at hcd
  · cases hcd
  · simp only [Option.bind_eq_bind, handleOf_drvHandle (by decide : kExec < 2 ^ 8) hei, Option.bind_some] at hcd
    have hE' : (obsCall w1 (.ext (.cuda .graphLaunch)) vs).dev.execs[ei]? = some (some ops) := hE
    rw [hE', Option.join_some, Option.bind_some] at hcd
    obtain ⟨p, -, hcd⟩ := Option.bind_eq_some_iff.mp hcd
    obtain ⟨d, hkd, rfl⟩ := devOnly_eq hcd
    obtain ⟨d0, hd0, hkd⟩ := Option.bind_eq_some_iff.mp hkd
    simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hkd
    obtain ⟨-, rfl⟩ := hkd
    obtain ⟨e, hs⟩ := runGraph_mem (w := obsCall w1 _ vs) hks hd0
    exact ⟨e, hs, Fixed.refl _⟩

/-- **A wait for a stream leaves device memory as it was.** -/
theorem streamSyncRec_done {w1 w2 : World} {vs : List V} {r : Option V}
    (hhas : w1.has (Ext.cuda .streamSynchronize).lib = true) (hcap : w1.dev.capture = none)
    (hx : extCall (.cuda .streamSynchronize) vs (obsCall w1 (.ext (.cuda .streamSynchronize)) vs) = some (r, w2)) :
    (∃ rc, w2.dev = { w1.dev with race := rc }) ∧ devMem w2.dev = devMem w1.dev ∧ CallKeeps w1 w2 := by
  obtain ⟨bits, -, hcd⟩ := extCall_cuda (w := obsCall w1 _ vs) hhas hx
  have hcap' : (obsCall w1 (.ext (.cuda .streamSynchronize)) vs).dev.capture = none := hcap
  rw [cudaDrv_step hcap'] at hcd
  simp only [cudaStep] at hcd
  split at hcd
  · split at hcd
    · cases hcd
    · simp only [Option.bind_eq_bind] at hcd
      obtain ⟨p, -, hcd⟩ := Option.bind_eq_some_iff.mp hcd
      split at hcd
      · cases hcd
      · obtain ⟨d, hkd, rfl⟩ := devOnly_eq hcd
        simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hkd
        obtain ⟨-, rfl⟩ := hkd
        exact ⟨⟨_, rfl⟩, rfl, DevSame.race _ _, Fixed.refl _⟩
  · cases hcd

theorem replicate_snoc_inv {α : Type} {x y : α} : ∀ {k : Nat} {rs0 : List α},
    List.replicate k x = rs0 ++ [y] → ∃ k', k = k' + 1 ∧ rs0 = List.replicate k' x ∧ y = x
  | 0, rs0, h => by cases rs0 <;> simp at h
  | k + 1, rs0, h => by
      rw [List.replicate_succ'] at h
      obtain ⟨e1, e2⟩ := List.append_inj' h rfl
      exact ⟨k, rfl, e1.symm, by simpa using e2.symm⟩

/-- **Launching graph `ei` `k` times is its operations' steps `k` times
    over**, the device otherwise as it was. -/
theorem extRun_graphs {w0 : World} {ei : Nat} {s : UInt64} {ops : Array DevOp}
    (hcu : w0.has (Ext.cuda .graphLaunch).lib = true) (hcap : w0.dev.capture = none)
    (hks : KeepsSize w0.kernel) (hE : w0.dev.execs[ei]? = some (some ops)) (hei : ei < 2 ^ 48) :
    ∀ {cs : List (Callee × List V)} {w : World}, ExtRun w0 cs w →
      ∀ k, recsMatch (List.replicate k (graphRec (drvHandle kExec ei) s)) cs →
        devMem w.dev = Nat.repeat (fun m => ops.toList.foldl (opStep w0.kernel w0.vendor) m) k (devMem w0.dev) ∧
          CallKeeps w0 w := by
  intro cs w hr
  induction hr with
  | nil hd ho hf =>
      intro k hm
      have : k = 0 := by
        cases k with
        | zero => rfl
        | succ k => simp [List.replicate_succ] at hm; exact absurd hm (by simp [recsMatch])
      subst this
      exact ⟨by rw [hd]; rfl, ⟨by rw [hd]; exact DevSame.refl _, hf⟩⟩
  | host _ hd ho hf _ ih =>
      intro k hm
      obtain ⟨e, hk⟩ := ih k hm
      exact ⟨by rw [hd]; exact e, hk.trans ⟨by rw [hd]; exact DevSame.refl _, hf⟩⟩
  | @call w1 w2 cs e vs r _ hx ho ih =>
      intro k hm
      obtain ⟨rs0, r0, hrs, hm0, hr0⟩ := recsMatch_snoc hm
      obtain ⟨k', rfl, rfl, rfl⟩ := replicate_snoc_inv hrs
      obtain ⟨e0, hk0⟩ := ih k' hm0
      have he : Callee.ext e = .ext (.cuda .graphLaunch) := hr0.1.symm
      injection he with he
      subst he
      have hcap1 : w1.dev.capture = none := by rw [hk0.1.capture]; exact hcap
      have hks1 : KeepsSize w1.kernel := by rw [hk0.2.1]; exact hks
      have hE1 : w1.dev.execs[ei]? = some (some ops) := by rw [hk0.1.execs]; exact hE
      have hcu1 : w1.has (Ext.cuda .graphLaunch).lib = true := by
        unfold World.has at hcu ⊢; rw [hk0.2.2.2]; exact hcu
      obtain ⟨e1, hk1⟩ := graphRec_done hcu1 hcap1 hks1 hE1 hei hr0 hx
      refine ⟨?_, hk0.trans hk1⟩
      rw [e1, e0, hk0.2.1, hk0.2.2.1]
      rfl

/-- **Replaying graph `ei` `k` times, then waiting for the stream, is its
    operations' steps `k` times over**, the device otherwise as it was. -/
theorem extRun_replay {w0 : World} {ei : Nat} {s : UInt64} {ops : Array DevOp}
    (hcu : w0.has (Ext.cuda .graphLaunch).lib = true) (hcap : w0.dev.capture = none)
    (hks : KeepsSize w0.kernel) (hE : w0.dev.execs[ei]? = some (some ops)) (hei : ei < 2 ^ 48) (k : Nat) :
    ∀ {cs : List (Callee × List V)} {w : World}, ExtRun w0 cs w →
      recsMatch (List.replicate k (graphRec (drvHandle kExec ei) s) ++ [streamSyncRec s]) cs →
        devMem w.dev = Nat.repeat (fun m => ops.toList.foldl (opStep w0.kernel w0.vendor) m) k (devMem w0.dev) ∧
          CallKeeps w0 w := by
  intro cs w hr
  induction hr with
  | nil _ _ _ =>
      intro hm
      exact absurd (recsMatch_nil_right hm) (by simp)
  | host _ hd ho hf _ ih =>
      intro hm
      obtain ⟨e, hk⟩ := ih hm
      exact ⟨by rw [hd]; exact e, hk.trans ⟨by rw [hd]; exact DevSame.refl _, hf⟩⟩
  | @call w1 w2 cs e vs r hr1 hx ho _ =>
      intro hm
      obtain ⟨rs0, r0, hrs, hm0, hr0⟩ := recsMatch_snoc hm
      obtain ⟨rfl, e2⟩ := List.append_inj' hrs.symm rfl
      injection e2 with e2
      subst e2
      obtain ⟨e0, hk0⟩ := extRun_graphs hcu hcap hks hE hei hr1 k hm0
      have he : Callee.ext e = .ext (.cuda .streamSynchronize) := hr0.1.symm
      injection he with he
      subst he
      have hcap1 : w1.dev.capture = none := by rw [hk0.1.capture]; exact hcap
      have hcu1 : w1.has (Ext.cuda .streamSynchronize).lib = true := by
        unfold World.has at hcu ⊢; rw [hk0.2.2.2]; exact hcu
      obtain ⟨-, e1, hk1⟩ := streamSyncRec_done hcu1 hcap1 hx
      exact ⟨by rw [e1, e0], hk0.trans hk1⟩

/-- **The device operation step `s` stands for in a graph**, read against
    the world `w0` the run started from: a kernel's launch as `DevCall.Step`
    states it, a product as the vendor's routine with scalars `s0`, `s1`. A
    wait is no operation of a graph. -/
def DevCall.op? (a : Drv.App) (w0 : World) (s0 s1 : UInt64) : DevCall → Option DevOp
  | .kernel j => do
      let k ← a.kernels[j]?
      let (mi, entry) ← w0.dev.funcs[j]?
      let ptx ← (w0.dev.modules[mi]?).join
      pure (.launch ⟨ptx, entry, DrvLaunch.bindsOf w0.dev (recLaunchArgs k), recDims k, recLaunchArgs k,
        w0.dev.writesOf (w0.dev.kernelWrites ptx entry) (recLaunchArgs k)⟩ (DrvLaunch.bindsOf w0.dev (recLaunchArgs k)))
  | .gemm g => some (.vendor (stridedCall s0 s1 0 0 0 g) [g.2.2.2.2.2.1, g.2.2.2.2.2.2.1, g.2.2.2.2.2.2.2]
      g.2.2.2.2.2.2.2)
  | .sync => none

theorem DevCall.Steps.append {a : Drv.App} {w0 : World} {s0 s1 : UInt64} {ss : List DevCall} {m m1 : DevMem}
    (h1 : DevCall.Steps a w0 s0 s1 ss m m1) :
    ∀ {tt : List DevCall} {m2 : DevMem}, DevCall.Steps a w0 s0 s1 tt m1 m2 → DevCall.Steps a w0 s0 s1 (ss ++ tt) m m2 := by
  intro tt m2 h2
  induction h2 with
  | nil => simpa using h1
  | snoc _ hs ih => rw [← List.append_assoc]; exact .snoc (ih h1) hs

theorem DevCall.Steps.single {a : Drv.App} {w0 : World} {s0 s1 : UInt64} {s : DevCall} {m m' : DevMem}
    (h : DevCall.Step a w0 s0 s1 s m m') : DevCall.Steps a w0 s0 s1 [s] m m' := by
  simpa using DevCall.Steps.snoc (.nil m) h

/-- **The operations steps stand for are those steps**: run in order, from
    any device memory, they leave what the steps leave. -/
theorem ops_steps (a : Drv.App) (w0 : World) (s0 s1 : UInt64) :
    ∀ (ss : List DevCall) (ops : List DevOp) (m : DevMem), ss.mapM (DevCall.op? a w0 s0 s1) = some ops →
      DevCall.Steps a w0 s0 s1 ss m (ops.foldl (opStep w0.kernel w0.vendor) m)
  | [], ops, m, h => by
      simp only [List.mapM_nil, pure, Option.some.injEq] at h
      subst h
      exact .nil m
  | s :: ss, ops, m, h => by
      simp only [List.mapM_cons, Option.bind_eq_bind] at h
      obtain ⟨op, hop, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨ops', hops', h⟩ := Option.bind_eq_some_iff.mp h
      simp only [pure, Option.some.injEq] at h
      subst h
      rw [List.foldl_cons]
      refine (DevCall.Steps.single ?_).append (ops_steps a w0 s0 s1 ss ops' _ hops')
      cases s with
      | kernel j =>
          simp only [DevCall.op?, Option.bind_eq_bind] at hop
          obtain ⟨k, hk, hop⟩ := Option.bind_eq_some_iff.mp hop
          obtain ⟨⟨mi, entry⟩, hf, hop⟩ := Option.bind_eq_some_iff.mp hop
          obtain ⟨ptx, hp, hop⟩ := Option.bind_eq_some_iff.mp hop
          simp only [pure, Option.some.injEq] at hop
          subst hop
          exact ⟨k, mi, entry, ptx, hk, hf, hp, rfl⟩
      | gemm g =>
          simp only [DevCall.op?, Option.some.injEq] at hop
          subst hop
          rfl
      | sync => simp [DevCall.op?] at hop

theorem repeat_steps {a : Drv.App} {w0 : World} {s0 s1 : UInt64} {ss : List DevCall} {f : DevMem → DevMem}
    (hf : ∀ m, DevCall.Steps a w0 s0 s1 ss m (f m)) :
    ∀ (k : Nat) (m : DevMem), DevCall.Steps a w0 s0 s1 (List.replicate k ss).flatten m (Nat.repeat f k m)
  | 0, m => .nil m
  | k + 1, m => by
      rw [List.replicate_succ', List.flatten_append, List.flatten_singleton]
      exact (repeat_steps hf k m).append (hf _)

/-- **Replaying a graph of steps' operations `k` times is those steps `k`
    times over**: from a world with the driver loaded, no capture in
    progress, kernels that keep their sizes, and executable graph `ei`
    holding the operations steps `ss` stand for, a run that replays it `k`
    times on a stream and waits for the stream takes `ss` `k` times over,
    the device otherwise as it was. -/
theorem replay_steps {a : Drv.App} {s0 s1 : UInt64} {w0 : World} {ei : Nat} {s : UInt64} {ss : List DevCall}
    {ops : Array DevOp} (hcu : w0.has (Ext.cuda .graphLaunch).lib = true) (hcap : w0.dev.capture = none)
    (hks : KeepsSize w0.kernel) (hE : w0.dev.execs[ei]? = some (some ops)) (hei : ei < 2 ^ 48)
    (hops : ss.mapM (DevCall.op? a w0 s0 s1) = some ops.toList) (k : Nat)
    {cs : List (Callee × List V)} {w : World} (hr : ExtRun w0 cs w)
    (hm : recsMatch (List.replicate k (graphRec (drvHandle kExec ei) s) ++ [streamSyncRec s]) cs) :
    DevCall.Steps a w0 s0 s1 (List.replicate k ss).flatten (devMem w0.dev) (devMem w.dev) ∧ CallKeeps w0 w := by
  obtain ⟨e, hk⟩ := extRun_replay hcu hcap hks hE hei k hr hm
  rw [e]
  exact ⟨repeat_steps (fun m => ops_steps a w0 s0 s1 ss _ m hops) k _, hk⟩

/-- **A launch record's call, answered while its stream captures, records
    the launch**: the capture's operations gain the launch kernel `j`'s step
    stands for (`DevCall.op?`), and the device is otherwise as it was but for
    the capture's race tracker. -/
theorem launchRec_captured {a : Drv.App} {j : Nat} {k : Drv.KSpec} {s : UInt64} (hk : a.kernels[j]? = some k)
    (hj : j < 2 ^ 48) {w1 w2 : World} {vs : List V} {r : Option V} {c : Capture} {p : Nat}
    (hhas : w1.has (Ext.cuda .launchKernel).lib = true) (hc : w1.dev.capture = some c)
    (hp : w1.dev.streamParty? s = some p) (hcp : w1.dev.capturing p = true)
    (hdims : (recDims k).any (· == 0) = false)
    (hm : (a.launchRec j s).Matches (.ext (.cuda .launchKernel), vs ++ callWords w1 (.ext (.cuda .launchKernel)) vs))
    (hx : extCall (.cuda .launchKernel) vs (obsCall w1 (.ext (.cuda .launchKernel)) vs) = some (r, w2)) :
    ∃ op r', DevCall.op? a w1 0 0 (.kernel j) = some op ∧
      w2.dev = { w1.dev with capture := some { c with race := r', ops := c.ops.push op } } ∧ Fixed w1 w2 := by
  obtain ⟨-, ha⟩ := hm
  simp only [Drv.App.launchRec, hk] at ha
  have hw1 : (obsCall w1 (.ext (.cuda .launchKernel)) vs).has (Ext.cuda .launchKernel).lib = true := hhas
  obtain ⟨bits, hbits, hcd⟩ := extCall_cuda hw1 hx
  simp only [cudaDrv, CudaFn.underCapture, Bool.not_true, Bool.and_false, Bool.false_eq_true, if_false] at hcd
  have hl : vs.length = 11 := by
    rw [← mapM_asBits_length hbits]
    exact cudaStep_launch_length hcd
  obtain ⟨h11, hcw⟩ := argsOk_split (os := [_, _, _, _, _, _, _, _, _, _, _]) (by simpa using hl) ha
  have hb := mapM_asBits_of_argsOk (ws := [drvHandle kFunc j, _, _, _, _, _, _, 0, s, _, 0]) h11
  rw [hbits] at hb
  cases hb
  simp only [cudaStep] at hcd
  split at hcd
  · cases hcd
  simp only [Option.bind_eq_bind] at hcd
  obtain ⟨fi, h1, hcd⟩ := Option.bind_eq_some_iff.mp hcd
  rw [handleOf_drvHandle (by decide : kFunc < 2 ^ 8) hj] at h1
  cases h1
  obtain ⟨⟨mi, entry⟩, h2, hcd⟩ := Option.bind_eq_some_iff.mp hcd
  obtain ⟨ptx, h3, hcd⟩ := Option.bind_eq_some_iff.mp hcd
  obtain ⟨p', h4, hcd⟩ := Option.bind_eq_some_iff.mp hcd
  have hp' : p' = p := by
    have : (obsCall w1 (.ext (.cuda .launchKernel)) vs).dev.streamParty? s = some p := hp
    rw [this] at h4; exact (Option.some.inj h4).symm
  subst hp'
  simp only [recDims, DrvLaunch.dimsOf] at hdims
  rw [if_neg (by rw [hdims]; exact Bool.false_ne_true)] at hcd
  obtain ⟨sizes, h5, hcd⟩ := Option.bind_eq_some_iff.mp hcd
  obtain ⟨args, h6, hcd⟩ := Option.bind_eq_some_iff.mp hcd
  obtain ⟨d, hkd, rfl⟩ := devOnly_eq hcd
  obtain ⟨d0, hd0, hkd⟩ := Option.bind_eq_some_iff.mp hkd
  simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hkd
  obtain ⟨-, rfl⟩ := hkd
  -- the launch is recorded, not run
  have hc' : (obsCall w1 (.ext (.cuda .launchKernel)) vs).dev.capture = some c := hc
  have hcp' : (obsCall w1 (.ext (.cuda .launchKernel)) vs).dev.capturing p' = true := hcp
  unfold Dev.devOp at hd0
  rw [hc'] at hd0
  simp only [hcp', if_true, Option.bind_eq_bind] at hd0
  obtain ⟨r', -, hd0⟩ := Option.bind_eq_some_iff.mp hd0
  simp only [Option.some.injEq] at hd0
  subst hd0
  -- the parameter values read are the ones the record names
  have hlw : launchWords w1.dev w1.mem (drvHandle kFunc j) (arenaAt (a.paramsOff j)) = some args := by
    simp only [launchWords, handleOf_drvHandle (by decide : kFunc < 2 ^ 8) hj, Option.bind_eq_bind,
      Option.bind_some]
    have h2' : w1.dev.funcs[j]? = some (mi, entry) := h2
    have h3' : (w1.dev.modules[mi]?).join = some ptx := h3
    rw [h2', Option.bind_some]
    simp only [h3', Option.bind_some]
    rw [h5, Option.bind_some]
    exact h6
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, rfl⟩ := eq_of_length_eleven vs hl
  obtain ⟨hv0, -, -, -, -, -, -, -, -, hv9, -⟩ := h11
  have hv0' : asBits v0 = some (drvHandle kFunc j) := hv0
  have hv9' : asBits v9 = some (arenaAt (a.paramsOff j)) := hv9
  have hcw' : callWords w1 (.ext (.cuda .launchKernel)) [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10]
      = args.map (V.sc .i64) := by
    show (match asBits v0, asBits v9 with
      | some f, some p => ((launchWords w1.dev w1.mem f p).getD []).map (V.sc .i64)
      | _, _ => []) = _
    simp only [hv0', hv9', hlw, Option.getD_some]
  rw [hcw'] at hcw
  have e := words_exact (ws := recLaunchArgs k) (by simpa [recLaunchArgs, List.map_map] using hcw)
  have e' : args = recLaunchArgs k := by
    simpa [List.map_map, Function.comp_def, asBits] using e
  subst e'
  refine ⟨_, r', ?_, rfl, Fixed.refl _⟩
  have h2' : w1.dev.funcs[j]? = some (mi, entry) := h2
  have h3' : (w1.dev.modules[mi]?).join = some ptx := h3
  simp only [DevCall.op?, hk, h2', h3', Option.bind_eq_bind, Option.bind_some, pure, recDims, DrvLaunch.dimsOf,
    DrvLaunch.bindsOf]
  rfl

/-! ## A run, cut in two, and runs of one call -/

theorem _root_.AlgorithmLib.Check.ExtRun.refl (w : World) : ExtRun w [] w :=
  .nil rfl rfl (Fixed.refl _) (MemLogged.refl _)

/-- **A run through calls `cs₁ ++ cs₂` passes through a world after `cs₁`.** -/
theorem _root_.AlgorithmLib.Check.ExtRun.split {w0 : World} :
    ∀ {cs : List (Callee × List V)} {w : World}, ExtRun w0 cs w →
      ∀ cs1 cs2, cs = cs1 ++ cs2 → ∃ w1, ExtRun w0 cs1 w1 ∧ ExtRun w1 cs2 w := by
  intro cs w h
  induction h with
  | nil hd ho hf hm =>
      intro cs1 cs2 e
      obtain ⟨rfl, rfl⟩ := List.append_eq_nil_iff.mp e.symm
      exact ⟨w0, ExtRun.refl _, .nil hd ho hf hm⟩
  | @call w1 w2 cs e vs r h1 hx ho ih =>
      intro cs1 cs2 he
      rcases List.eq_nil_or_concat cs2 with rfl | ⟨cs2', x, rfl⟩
      · rw [List.append_nil] at he
        subst he
        exact ⟨w2, .call h1 hx ho, ExtRun.refl _⟩
      · rw [List.concat_eq_append, ← List.append_assoc] at he
        obtain ⟨e1, e2⟩ := List.append_inj' he rfl
        obtain ⟨w', ha, hb⟩ := ih cs1 cs2' e1
        cases e2
        exact ⟨w', ha, by rw [List.concat_eq_append]; exact .call hb hx ho⟩
  | host _ hd ho hf hm ih =>
      intro cs1 cs2 he
      obtain ⟨w', ha, hb⟩ := ih cs1 cs2 he
      exact ⟨w', ha, .host hb hd ho hf hm⟩

/-- A run through no calls leaves the device and the oracles as they were,
    and host memory but where it observed stores. -/
theorem _root_.AlgorithmLib.Check.ExtRun.nil_inv {w0 w : World} :
    ExtRun w0 [] w → w.dev = w0.dev ∧ Fixed w0 w ∧ MemLogged w0 w := by
  intro h
  generalize e : ([] : List (Callee × List V)) = cs at h
  induction h with
  | nil hd _ hf hm => exact ⟨hd, hf, hm⟩
  | call => simp at e
  | host _ hd _ hf hm ih =>
      obtain ⟨d, f, m⟩ := ih e
      exact ⟨hd.trans d, f.trans hf, m.trans hm⟩

/-- **A run through one call is that call**, between runs through none. -/
theorem _root_.AlgorithmLib.Check.ExtRun.one_inv {w0 w : World} {x : Callee × List V} (h : ExtRun w0 [x] w) :
    ∃ w1 w2 e vs r, w1.dev = w0.dev ∧ Fixed w0 w1 ∧ x = (.ext e, vs ++ callWords w1 (.ext e) vs) ∧
      extCall e vs (obsCall w1 (.ext e) vs) = some (r, w2) ∧ w.dev = w2.dev ∧ Fixed w2 w ∧
      MemLogged w0 w1 ∧ MemLogged w2 w ∧ w2.obs = .call (.ext e) (vs ++ callWords w1 (.ext e) vs) :: w1.obs := by
  generalize ex : [x] = cs at h
  induction h with
  | nil => simp at ex
  | @call w1 w2 cs e vs r h1 hx ho _ =>
      obtain ⟨rfl, rfl⟩ := List.append_eq_singleton_iff.mp ex.symm |>.resolve_right (by simp) |>.imp id
        (fun h => by simpa using h)
      obtain ⟨d, f, m⟩ := h1.nil_inv
      exact ⟨w1, w2, e, vs, r, d, f, rfl, hx, rfl, Fixed.refl _, m, MemLogged.refl _, ho⟩
  | host _ hd _ hf hm ih =>
      obtain ⟨w1, w2, e, vs, r, a, b, c, d, f, g, m1, m2, o⟩ := ih ex
      exact ⟨w1, w2, e, vs, r, a, b, c, d, hd.trans f, g.trans hf, m1, m2.trans hm, o⟩

/-- **A driver run only adds to what was observed.** -/
theorem _root_.AlgorithmLib.Check.ExtRun.obs {w0 w : World} {cs : List (Callee × List V)} (h : ExtRun w0 cs w) :
    ∃ ds, w.obs = ds ++ w0.obs := by
  induction h with
  | nil _ _ _ hm => obtain ⟨ds, e, -⟩ := hm; exact ⟨ds, e⟩
  | call _ _ ho ih => obtain ⟨ds, e⟩ := ih; exact ⟨_ :: ds, by rw [ho, e]; rfl⟩
  | host _ _ _ _ hm ih =>
      obtain ⟨ds, e⟩ := ih
      obtain ⟨d2, e2, -⟩ := hm
      exact ⟨d2 ++ ds, by rw [e2, e, List.append_assoc]⟩

/-- **What a launch's operation reads of the device**: its functions,
    modules, allocations and what each kernel is said to write. -/
structure OpSame (d d' : Dev) : Prop where
  funcs : d'.funcs = d.funcs
  modules : d'.modules = d.modules
  bufs : d'.bufs = d.bufs
  kernelWrites : d'.kernelWrites = d.kernelWrites

theorem OpSame.refl (d : Dev) : OpSame d d := ⟨rfl, rfl, rfl, rfl⟩

theorem OpSame.trans {a b c : Dev} (h1 : OpSame a b) (h2 : OpSame b c) : OpSame a c :=
  ⟨h2.funcs.trans h1.funcs, h2.modules.trans h1.modules, h2.bufs.trans h1.bufs,
    h2.kernelWrites.trans h1.kernelWrites⟩

theorem DevSame.opSame {d d' : Dev} (h : DevSame d d') (hb : d'.bufs = d.bufs) : OpSame d d' :=
  ⟨h.funcs, h.modules, hb, h.kernelWrites⟩

theorem range_congr {d d' : Dev} (hb : d'.bufs = d.bufs) : d'.range? = d.range? := by
  funext p n
  simp only [Dev.range?, hb]

theorem op?_congr {a : Drv.App} {w w' : World} {s0 s1 : UInt64} (h : OpSame w.dev w'.dev) (st : DevCall) :
    DevCall.op? a w' s0 s1 st = DevCall.op? a w s0 s1 st := by
  cases st with
  | kernel j =>
      simp only [DevCall.op?, h.funcs, h.modules, DrvLaunch.bindsOf, Dev.writesOf, range_congr h.bufs,
        h.kernelWrites]
  | gemm g => rfl
  | sync => rfl

/-- **A record of plain words, matched, is that call.** -/
theorem cudaRec_call {f : CudaFn} {ws : List UInt64} {w1 w2 : World} {vs : List V} {r : Option V}
    (hhas : w1.has (Ext.cuda f).lib = true)
    (hm : (⟨.ext (.cuda f), ws.map ArgPat.is⟩ : CallRec).Matches (.ext (.cuda f), vs ++ callWords w1 (.ext (.cuda f)) vs))
    (hl : ∀ bits x, cudaDrv f bits (obsCall w1 (.ext (.cuda f)) vs) = some x → bits.length = ws.length)
    (hx : extCall (.cuda f) vs (obsCall w1 (.ext (.cuda f)) vs) = some (r, w2)) :
    cudaDrv f ws (obsCall w1 (.ext (.cuda f)) vs) = some (r, w2) := by
  have hw : (obsCall w1 (.ext (.cuda f)) vs).has (Ext.cuda f).lib = true := hhas
  obtain ⟨bits, hbits, hcd⟩ := extCall_cuda hw hx
  have hlen : vs.length = (ws.map ArgPat.is).length := by
    rw [← mapM_asBits_length hbits, hl _ _ hcd, List.length_map]
  obtain ⟨h1, -⟩ := argsOk_split (os' := []) hlen (by simpa using hm.2)
  rw [mapM_asBits_of_argsOk h1] at hbits
  cases hbits
  exact hcd

/-! ## The calls a capture makes -/

/-- The record of a call handed plain words. -/
def wordRec (f : CudaFn) (ws : List UInt64) : CallRec := ⟨.ext (.cuda f), ws.map ArgPat.is⟩

theorem cudaStep_len {f : CudaFn} {bits : List UInt64} {w : World} {x : Option V × World} {n : Nat}
    (hn : ∀ bits : List UInt64, bits.length ≠ n → cudaStep f bits w = none)
    (h : cudaStep f bits w = some x) : bits.length = n := by
  by_cases hne : bits.length = n
  · exact hne
  · rw [hn bits hne] at h
    cases h

/-- **Making a stream** adds a live stream, the device otherwise as it was
    but for the race tracker. -/
theorem streamCreate_done {w1 w2 : World} {vs : List V} {r : Option V} {ps : UInt64}
    (hhas : w1.has (Ext.cuda .streamCreate).lib = true) (hcap : w1.dev.capture = none)
    (hm : (wordRec .streamCreate [ps, 1]).Matches (.ext (.cuda .streamCreate), vs ++ callWords w1 (.ext (.cuda .streamCreate)) vs))
    (hx : extCall (.cuda .streamCreate) vs (obsCall w1 (.ext (.cuda .streamCreate)) vs) = some (r, w2)) :
    ∃ rc, w2.dev = { w1.dev with streams := w1.dev.streams.push true, race := rc } ∧ Fixed w1 w2 := by
  have hcap' : (obsCall w1 (.ext (.cuda .streamCreate)) vs).dev.capture = none := hcap
  have hc := cudaRec_call hhas hm (fun bits x h => by
    rw [cudaDrv_step hcap'] at h
    exact cudaStep_len (fun bits hne => by
      simp only [cudaStep]; split
      · simp at hne
      · rfl) h) hx
  rw [cudaDrv_step hcap'] at hc
  simp only [cudaStep] at hc
  split at hc
  · cases hc
  simp only [Option.bind_eq_bind] at hc
  obtain ⟨m, -, hc⟩ := Option.bind_eq_some_iff.mp hc
  simp only [Option.some.injEq, Prod.mk.injEq] at hc
  obtain ⟨-, rfl⟩ := hc
  exact ⟨_, rfl, Fixed.refl _⟩

/-- **Beginning a capture on a stream** starts one there with nothing
    recorded, the device otherwise as it was. -/
theorem beginCapture_done {w1 w2 : World} {vs : List V} {r : Option V} {s : UInt64}
    (hhas : w1.has (Ext.cuda .beginCapture).lib = true) (hcap : w1.dev.capture = none)
    (hm : (wordRec .beginCapture [s, 2]).Matches (.ext (.cuda .beginCapture), vs ++ callWords w1 (.ext (.cuda .beginCapture)) vs))
    (hx : extCall (.cuda .beginCapture) vs (obsCall w1 (.ext (.cuda .beginCapture)) vs) = some (r, w2)) :
    ∃ p, w1.dev.streamParty? s = some p ∧ w2.dev = { w1.dev with capture := some ⟨p, [], {}, #[]⟩ } ∧
      Fixed w1 w2 := by
  have hcap' : (obsCall w1 (.ext (.cuda .beginCapture)) vs).dev.capture = none := hcap
  have hc := cudaRec_call hhas hm (fun bits x h => by
    rw [cudaDrv_step hcap'] at h
    exact cudaStep_len (fun bits hne => by
      simp only [cudaStep]; split
      · simp at hne
      · rfl) h) hx
  rw [cudaDrv_step hcap'] at hc
  simp only [cudaStep] at hc
  split at hc
  · cases hc
  simp only [Option.bind_eq_bind] at hc
  obtain ⟨p, hp, hc⟩ := Option.bind_eq_some_iff.mp hc
  split at hc
  · cases hc
  obtain ⟨d, hkd, rfl⟩ := devOnly_eq hc
  simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hkd
  obtain ⟨-, rfl⟩ := hkd
  exact ⟨p, hp, rfl, Fixed.refl _⟩

/-- **Ending a capture that every joined stream rejoined** makes its graph
    of the operations it recorded, and ends the capture. -/
theorem endCapture_done {w1 w2 : World} {vs : List V} {r : Option V} {s pg : UInt64} {c : Capture} {p : Nat}
    (hhas : w1.has (Ext.cuda .endCapture).lib = true) (hc : w1.dev.capture = some c)
    (hp : w1.dev.streamParty? s = some p) (ho : c.origin = p) (hrj : c.rejoined = true)
    (hm : (wordRec .endCapture [s, pg]).Matches (.ext (.cuda .endCapture), vs ++ callWords w1 (.ext (.cuda .endCapture)) vs))
    (hx : extCall (.cuda .endCapture) vs (obsCall w1 (.ext (.cuda .endCapture)) vs) = some (r, w2)) :
    w2.dev = { w1.dev.endCapture with graphs := w1.dev.graphs.push (some c.ops) } ∧ Fixed w1 w2 := by
  have hdrv : ∀ bits, cudaDrv .endCapture bits (obsCall w1 (.ext (.cuda .endCapture)) vs) =
      cudaStep .endCapture bits (obsCall w1 (.ext (.cuda .endCapture)) vs) := fun bits => by
    simp [cudaDrv, CudaFn.underCapture]
  have hcd := cudaRec_call hhas hm (fun bits x h => by
    rw [hdrv] at h
    exact cudaStep_len (fun bits hne => by
      simp only [cudaStep]; split
      · simp at hne
      · rfl) h) hx
  rw [hdrv] at hcd
  simp only [cudaStep] at hcd
  split at hcd
  · cases hcd
  simp only [Option.bind_eq_bind] at hcd
  have hp' : (obsCall w1 (.ext (.cuda .endCapture)) vs).dev.streamParty? s = some p := hp
  have hc' : (obsCall w1 (.ext (.cuda .endCapture)) vs).dev.capture = some c := hc
  rw [hp', Option.bind_some, hc', Option.bind_some] at hcd
  rw [if_neg (by simp [ho]), if_neg (by simp [hrj])] at hcd
  obtain ⟨m, -, hcd⟩ := Option.bind_eq_some_iff.mp hcd
  simp only [Option.some.injEq, Prod.mk.injEq] at hcd
  obtain ⟨-, rfl⟩ := hcd
  exact ⟨rfl, Fixed.refl _⟩

/-- **Instantiating graph `gi`** makes an executable graph of its
    operations, the device otherwise as it was. -/
theorem graphInstantiate_done {w1 w2 : World} {vs : List V} {r : Option V} {pe : UInt64} {gi : Nat}
    {ops : Array DevOp} (hhas : w1.has (Ext.cuda .graphInstantiate).lib = true) (hcap : w1.dev.capture = none)
    (hgi : gi < 2 ^ 48) (hg : w1.dev.graphs[gi]? = some (some ops))
    (hm : (wordRec .graphInstantiate [pe, drvHandle kGraph gi, 0]).Matches
      (.ext (.cuda .graphInstantiate), vs ++ callWords w1 (.ext (.cuda .graphInstantiate)) vs))
    (hx : extCall (.cuda .graphInstantiate) vs (obsCall w1 (.ext (.cuda .graphInstantiate)) vs) = some (r, w2)) :
    w2.dev = { w1.dev with execs := w1.dev.execs.push (some ops) } ∧ Fixed w1 w2 := by
  have hcap' : (obsCall w1 (.ext (.cuda .graphInstantiate)) vs).dev.capture = none := hcap
  have hcd := cudaRec_call hhas hm (fun bits x h => by
    rw [cudaDrv_step hcap'] at h
    exact cudaStep_len (fun bits hne => by
      simp only [cudaStep]; split
      · simp at hne
      · rfl) h) hx
  rw [cudaDrv_step hcap'] at hcd
  simp only [cudaStep] at hcd
  split at hcd
  · cases hcd
  have hg' : (obsCall w1 (.ext (.cuda .graphInstantiate)) vs).dev.graphs[gi]? = some (some ops) := hg
  simp only [Option.bind_eq_bind, handleOf_drvHandle (by decide : kGraph < 2 ^ 8) hgi, Option.bind_some, hg',
    Option.join_some] at hcd
  obtain ⟨m, -, hcd⟩ := Option.bind_eq_some_iff.mp hcd
  simp only [Option.some.injEq, Prod.mk.injEq] at hcd
  obtain ⟨-, rfl⟩ := hcd
  exact ⟨rfl, Fixed.refl _⟩

/-- **Uploading an executable graph** changes nothing observable. -/
theorem graphUpload_done {w1 w2 : World} {vs : List V} {r : Option V} {he s : UInt64}
    (hhas : w1.has (Ext.cuda .graphUpload).lib = true) (hcap : w1.dev.capture = none)
    (hm : (wordRec .graphUpload [he, s]).Matches (.ext (.cuda .graphUpload), vs ++ callWords w1 (.ext (.cuda .graphUpload)) vs))
    (hx : extCall (.cuda .graphUpload) vs (obsCall w1 (.ext (.cuda .graphUpload)) vs) = some (r, w2)) :
    w2.dev = w1.dev ∧ Fixed w1 w2 := by
  have hcap' : (obsCall w1 (.ext (.cuda .graphUpload)) vs).dev.capture = none := hcap
  have hcd := cudaRec_call hhas hm (fun bits x h => by
    rw [cudaDrv_step hcap'] at h
    exact cudaStep_len (fun bits hne => by
      simp only [cudaStep]; split
      · simp at hne
      · rfl) h) hx
  rw [cudaDrv_step hcap'] at hcd
  simp only [cudaStep] at hcd
  split at hcd
  · cases hcd
  simp only [Option.bind_eq_bind] at hcd
  obtain ⟨ei, -, hcd⟩ := Option.bind_eq_some_iff.mp hcd
  obtain ⟨_, -, hcd⟩ := Option.bind_eq_some_iff.mp hcd
  obtain ⟨_, -, hcd⟩ := Option.bind_eq_some_iff.mp hcd
  simp only [Option.some.injEq, Prod.mk.injEq] at hcd
  obtain ⟨-, rfl⟩ := hcd
  exact ⟨rfl, Fixed.refl _⟩

/-! ## A capture's run -/

theorem recsMatch_split : ∀ {a b : List CallRec} {cs : List (Callee × List V)}, recsMatch (a ++ b) cs →
    ∃ s t, cs = s ++ t ∧ recsMatch a s ∧ recsMatch b t
  | [], _, cs, h => ⟨[], cs, rfl, trivial, h⟩
  | _ :: _, _, x :: _, ⟨hr, h⟩ => by
      obtain ⟨s, t, rfl, h1, h2⟩ := recsMatch_split h
      exact ⟨x :: s, t, rfl, ⟨hr, h1⟩, h2⟩
  | _ :: _, _, [], h => h.elim

theorem devMem_congr {d d' : Dev} (hb : d'.bufs = d.bufs) : devMem d' = devMem d := by
  funext i; simp only [devMem, hb]

theorem dev_capture_eta {d : Dev} {c : Capture} (h : d.capture = some c) :
    { d with capture := some { c with race := c.race, ops := c.ops ++ (([] : List DevOp)).toArray } } = d := by
  cases d; cases c; simp_all

/-- **Launches made while their stream captures are recorded**: a run
    through launch records of kernels `js` on stream `s`, from a world
    where a capture began on `s`, adds to the capture the operations those
    kernels' steps stand for, in order, the device otherwise as it was. -/
theorem extRun_captured {a : Drv.App} {s : UInt64} {w1 : World} {c : Capture} {p : Nat}
    (hcu : w1.has (Ext.cuda .launchKernel).lib = true) (hc : w1.dev.capture = some c)
    (hp : w1.dev.streamParty? s = some p) (ho : c.origin = p) :
    ∀ {cs : List (Callee × List V)} {w : World}, ExtRun w1 cs w →
      ∀ js : List Nat, (∀ j ∈ js, DevCall.okB a (.kernel j) = true) →
      recsMatch (js.map fun j => a.launchRec j s) cs →
      ∃ r ops, (js.map DevCall.kernel).mapM (DevCall.op? a w1 0 0) = some ops ∧
        w.dev = { w1.dev with capture := some { c with race := r, ops := c.ops ++ ops.toArray } } ∧ Fixed w1 w := by
  intro cs w hr
  induction hr with
  | nil hd ho' hf =>
      intro js _ hm
      have hjs : js = [] := by simpa using recsMatch_nil_right hm
      subst hjs
      exact ⟨c.race, [], rfl, by rw [hd, dev_capture_eta hc], hf⟩
  | host _ hd ho' hf _ ih =>
      intro js hok hm
      obtain ⟨r, ops, hops, hdev, hfx⟩ := ih js hok hm
      exact ⟨r, ops, hops, by rw [hd, hdev], hfx.trans hf⟩
  | @call wa wb cs e vs r0 _ hx _ ih =>
      intro js hok hm
      obtain ⟨rs0, rl, hrs, hm0, hrl⟩ := recsMatch_snoc hm
      obtain ⟨l1, l2, rfl, e1, e2⟩ := List.map_eq_append_iff.mp hrs
      obtain ⟨j, rfl, rfl⟩ := List.map_eq_singleton_iff.mp e2
      obtain ⟨r, ops, hops, hdev, hfx⟩ := ih l1 (fun j hj => hok j (List.mem_append_left _ hj)) (e1 ▸ hm0)
      have he : Callee.ext e = .ext (.cuda .launchKernel) := by
        have := hrl.1
        simp only [Drv.App.launchRec] at this
        split at this <;> exact this.symm
      injection he with he
      subst he
      obtain ⟨k, hk, hj, hdims⟩ := DevCall.okB_ok (hok j (List.mem_append_right _ (List.mem_singleton_self _)))
      have hcu' : wa.has (Ext.cuda .launchKernel).lib = true := by
        unfold World.has at hcu ⊢; rw [hfx.2.2]; exact hcu
      have hca : wa.dev.capture = some { c with race := r, ops := c.ops ++ ops.toArray } := by rw [hdev]
      have hpa : wa.dev.streamParty? s = some p := by rw [hdev]; exact hp
      have hcpa : wa.dev.capturing p = true := by simp [Dev.capturing, hca, ho]
      obtain ⟨op, r', hop, hdev', hfx'⟩ := launchRec_captured hk hj hcu' hca hpa hcpa hdims hrl hx
      have hsame : OpSame w1.dev wa.dev := by rw [hdev]; exact ⟨rfl, rfl, rfl, rfl⟩
      rw [op?_congr hsame] at hop
      refine ⟨r', ops ++ [op], ?_, ?_, hfx.trans hfx'⟩
      · rw [List.map_append, List.mapM_append, hops]
        simp [hop]
      · have ha : (c.ops ++ ops.toArray).push op = c.ops ++ (ops ++ [op]).toArray := by
          apply Array.toList_inj.mp
          simp
        rw [hdev', hdev, ← ha]

/-- A run through one call a record of plain words admits. -/
theorem _root_.AlgorithmLib.Check.ExtRun.one_word {w0 w : World} {f : CudaFn} {ws : List UInt64}
    {cs : List (Callee × List V)} (h : ExtRun w0 cs w) (hm : recsMatch [wordRec f ws] cs) :
    ∃ w1 w2 vs r, w1.dev = w0.dev ∧ Fixed w0 w1 ∧
      (wordRec f ws).Matches (.ext (.cuda f), vs ++ callWords w1 (.ext (.cuda f)) vs) ∧
      extCall (.cuda f) vs (obsCall w1 (.ext (.cuda f)) vs) = some (r, w2) ∧ w.dev = w2.dev ∧ Fixed w2 w := by
  have hcs : ∃ x, cs = [x] ∧ (wordRec f ws).Matches x := by
    cases cs with
    | nil => exact hm.elim
    | cons x t =>
        cases t with
        | nil => exact ⟨x, rfl, hm.1⟩
        | cons _ _ => exact hm.2.elim
  obtain ⟨x, rfl, hx⟩ := hcs
  obtain ⟨w1, w2, e, vs, r, h1, h2, hxe, h4, h5, h6, -, -, -⟩ := h.one_inv
  subst hxe
  have he : Callee.ext (Ext.cuda f) = .ext e := hx.1
  injection he with he
  subst he
  exact ⟨w1, w2, vs, r, h1, h2, hx, h4, h5, h6⟩

theorem has_cuda {w : World} {f g : CudaFn} (h : w.has (Ext.cuda f).lib = true) : w.has (Ext.cuda g).lib = true := h

theorem has_fixed {w w' : World} {l : Lib} (hf : Fixed w w') (h : w.has l = true) : w'.has l = true := by
  unfold World.has at h ⊢; rw [hf.2.2]; exact h

/-- The records of capturing launches of kernels `js` on the program's first
    stream: making the stream, its handle stored at `ps`, beginning the
    capture, the launches, ending it, its graph's handle stored at `pg`,
    instantiating the graph, its handle stored at `pe`, uploading it, and
    waiting for the stream. -/
def captureRecs (a : Drv.App) (js : List Nat) (ps pg pe : UInt64) : List CallRec :=
  [wordRec .streamCreate [ps, 1], wordRec .beginCapture [drvHandle kStream 0, 2]] ++
    js.map (fun j => a.launchRec j (drvHandle kStream 0)) ++
    [wordRec .endCapture [drvHandle kStream 0, pg], wordRec .graphInstantiate [pe, drvHandle kGraph 0, 0],
     wordRec .graphUpload [drvHandle kExec 0, drvHandle kStream 0], wordRec .streamSynchronize [drvHandle kStream 0]]

theorem streamParty_first {d : Dev} (hs : d.streams.size = 0) (r : Race) :
    ({ d with streams := d.streams.push true, race := r } : Dev).streamParty? (drvHandle kStream 0) =
      some (streamParty 0) := by
  have he : d.streams = #[] := Array.eq_empty_of_size_eq_zero hs
  unfold Dev.streamParty?
  rw [if_neg (by decide), handleOf_drvHandle (by decide) (by decide)]
  simp [he]

/-- **A capture's run leaves its graph holding the launches' operations**:
    from a world with the driver loaded, no capture in progress and no
    stream, graph or executable graph yet, a run through a capture of
    launches of kernels `js`, each one the device reads, leaves executable
    graph `0` holding the operations those kernels' steps stand for, in
    order, read against the world the run started from, no capture in
    progress, and the device's functions, modules, allocations and their
    contents and the oracles as they were. -/
theorem extRun_capture {a : Drv.App} {js : List Nat} {ps pg pe : UInt64} {w0 : World}
    (hcu : w0.has (Ext.cuda .launchKernel).lib = true) (hcap : w0.dev.capture = none)
    (hs : w0.dev.streams.size = 0) (hg : w0.dev.graphs.size = 0) (he : w0.dev.execs.size = 0)
    (hok : ∀ j ∈ js, DevCall.okB a (.kernel j) = true)
    {cs : List (Callee × List V)} {w : World} (hr : ExtRun w0 cs w)
    (hm : recsMatch (captureRecs a js ps pg pe) cs) :
    ∃ ops, (js.map DevCall.kernel).mapM (DevCall.op? a w0 0 0) = some ops ∧
      w.dev.execs[0]? = some (some ops.toArray) ∧ w.dev.capture = none ∧ OpSame w0.dev w.dev ∧ Fixed w0 w := by
  unfold captureRecs at hm
  obtain ⟨c12L, c4567, rfl, hm12L, hm4567⟩ := recsMatch_split hm
  obtain ⟨c12, cL, rfl, hm12, hmL⟩ := recsMatch_split hm12L
  obtain ⟨c1, c2, rfl, hm1, hm2⟩ := recsMatch_split (a := [_]) (b := [_]) hm12
  obtain ⟨c4, c567, rfl, hm4, hm567⟩ := recsMatch_split (a := [_]) hm4567
  obtain ⟨c5, c67, rfl, hm5, hm67⟩ := recsMatch_split (a := [_]) hm567
  obtain ⟨c6, c7, rfl, hm6, hm7⟩ := recsMatch_split (a := [_]) hm67
  obtain ⟨wL, hrL, hr4567⟩ := hr.split _ _ rfl
  obtain ⟨w2, hr12, hrL⟩ := hrL.split _ _ rfl
  obtain ⟨w1, hr1, hr2⟩ := hr12.split _ _ rfl
  obtain ⟨w4, hr4, hr567⟩ := hr4567.split _ _ rfl
  obtain ⟨w5, hr5, hr67⟩ := hr567.split _ _ rfl
  obtain ⟨w6, hr6, hr7⟩ := hr67.split _ _ rfl
  -- the stream
  obtain ⟨x1, y1, vs1, r1, ex1, fx1, hm1', hx1, ey1, fy1⟩ := hr1.one_word hm1
  obtain ⟨rc1, hd1, f1⟩ := streamCreate_done (has_cuda (has_fixed fx1 hcu)) (by rw [ex1]; exact hcap) hm1' hx1
  have hd1' : w1.dev = { w0.dev with streams := w0.dev.streams.push true, race := rc1 } := by rw [ey1, hd1, ex1]
  have fw1 : Fixed w0 w1 := fx1.trans (f1.trans fy1)
  have hp1 : w1.dev.streamParty? (drvHandle kStream 0) = some (streamParty 0) := by
    rw [hd1']; exact streamParty_first hs rc1
  -- the capture begins
  obtain ⟨x2, y2, vs2, r2, ex2, fx2, hm2', hx2, ey2, fy2⟩ := hr2.one_word hm2
  obtain ⟨p, hp, hd2, f2⟩ := beginCapture_done (has_cuda (has_fixed (fw1.trans fx2) hcu))
    (by rw [ex2, hd1']; exact hcap) hm2' hx2
  rw [ex2, hp1] at hp
  cases hp
  have hd2' : w2.dev = { w1.dev with capture := some ⟨streamParty 0, [], {}, #[]⟩ } := by rw [ey2, hd2, ex2]
  have fw2 : Fixed w0 w2 := fw1.trans (fx2.trans (f2.trans fy2))
  -- the launches are recorded
  obtain ⟨rL, ops, hops, hdL, fL⟩ := extRun_captured (has_cuda (has_fixed fw2 hcu))
    (c := ⟨streamParty 0, [], {}, #[]⟩) (p := streamParty 0) (by rw [hd2']) (by rw [hd2']; exact hp1) rfl
    hrL js hok hmL
  have hsame2 : OpSame w0.dev w2.dev := by rw [hd2', hd1']; exact ⟨rfl, rfl, rfl, rfl⟩
  have hop : DevCall.op? a w2 0 0 = DevCall.op? a w0 0 0 := funext (op?_congr hsame2)
  rw [hop] at hops
  have fwL : Fixed w0 wL := fw2.trans fL
  -- the capture ends in a graph
  obtain ⟨x4, y4, vs4, r4, ex4, fx4, hm4', hx4, ey4, fy4⟩ := hr4.one_word hm4
  obtain ⟨hd4, f4⟩ := endCapture_done
    (c := { (⟨streamParty 0, [], {}, #[]⟩ : Capture) with race := rL, ops := #[] ++ ops.toArray })
    (p := streamParty 0) (has_cuda (has_fixed (fwL.trans fx4) hcu))
    (by rw [ex4, hdL]) (by rw [ex4, hdL, hd2']; exact hp1) rfl (by simp [Capture.rejoined]) hm4' hx4
  have hd4' : w4.dev = { wL.dev.endCapture with graphs := wL.dev.graphs.push (some (#[] ++ ops.toArray)) } := by
    rw [ey4, hd4, ex4]
  have hgL : wL.dev.graphs = #[] := by rw [hdL, hd2', hd1']; exact Array.eq_empty_of_size_eq_zero hg
  have heL : wL.dev.execs = #[] := by rw [hdL, hd2', hd1']; exact Array.eq_empty_of_size_eq_zero he
  have hg4 : w4.dev.graphs[0]? = some (some (#[] ++ ops.toArray)) := by
    rw [hd4']; simp [hgL]
  have hcap4 : w4.dev.capture = none := by rw [hd4']; rfl
  have fw4 : Fixed w0 w4 := fwL.trans (fx4.trans (f4.trans fy4))
  -- the graph is instantiated
  obtain ⟨x5, y5, vs5, r5, ex5, fx5, hm5', hx5, ey5, fy5⟩ := hr5.one_word hm5
  obtain ⟨hd5, f5⟩ := graphInstantiate_done (has_cuda (has_fixed (fw4.trans fx5) hcu)) (by rw [ex5]; exact hcap4)
    (by decide) (by rw [ex5]; exact hg4) hm5' hx5
  have hd5' : w5.dev = { w4.dev with execs := w4.dev.execs.push (some (#[] ++ ops.toArray)) } := by
    rw [ey5, hd5, ex5]
  have fw5 : Fixed w0 w5 := fw4.trans (fx5.trans (f5.trans fy5))
  -- uploaded
  obtain ⟨x6, y6, vs6, r6, ex6, fx6, hm6', hx6, ey6, fy6⟩ := hr6.one_word hm6
  obtain ⟨hd6, f6⟩ := graphUpload_done (has_cuda (has_fixed (fw5.trans fx6) hcu))
    (by rw [ex6, hd5']; exact hcap4) hm6' hx6
  have hd6' : w6.dev = w5.dev := by rw [ey6, hd6, ex6]
  have fw6 : Fixed w0 w6 := fw5.trans (fx6.trans (f6.trans fy6))
  -- and waited for
  obtain ⟨x7, y7, vs7, r7, ex7, fx7, hm7', hx7, ey7, fy7⟩ := hr7.one_word hm7
  obtain ⟨⟨rc7, hd7⟩, -, f7⟩ := streamSyncRec_done (has_cuda (has_fixed (fw6.trans fx7) hcu))
    (by rw [ex7, hd6', hd5']; exact hcap4) hx7
  have hd7' : w.dev = { w6.dev with race := rc7 } := by rw [ey7, hd7, ex7]
  have fw7 : Fixed w0 w := fw6.trans (fx7.trans (f7.2.trans fy7))
  refine ⟨ops, hops, ?_, ?_, ?_, fw7⟩
  · rw [hd7', hd6', hd5', hd4']
    simp [Dev.endCapture, heL]
  · rw [hd7', hd6', hd5', hcap4]
  · rw [hd7', hd6', hd5', hd4', hdL, hd2', hd1']
    exact ⟨rfl, rfl, rfl, rfl⟩

end AlgorithmLib.ML
