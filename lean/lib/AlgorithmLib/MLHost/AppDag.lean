module
public import AlgorithmLib.MLHost.AppGraph
meta import AlgorithmLib.MLHost.AppGraph
@[expose] public section

/-!
# A capture over many streams, as steps

A capture that forks onto other streams records each launch and each product
issued on a stream that has joined it, and nothing for the events that order
them. Which streams have joined is read off the calls: a stream joins by
waiting on an event recorded inside the capture, and an event holds such a
point once a stream that has joined records it. `capSim` reads a list of
such calls (`CapCall`) that way and gives the steps the recorded operations
stand for, failing on any call it cannot place; `extRun_dag` proves that a
run through those calls records exactly those steps' operations.
-/

namespace AlgorithmLib.ML

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Contracts
  AlgorithmLib.Check AlgorithmLib.Device

/-! ## The calls, one at a time -/

/-- **An event recorded on a stream that has joined the capture** holds a
    point of the capture, the device otherwise as it was but for the
    capture's race tracker. -/
theorem eventRecord_captured {w1 w2 : World} {vs : List V} {r : Option V} {e s : UInt64} {c : Capture} {p : Nat}
    (hhas : w1.has (Ext.cuda .eventRecord).lib = true) (hc : w1.dev.capture = some c)
    (hp : w1.dev.streamParty? s = some p) (hcp : w1.dev.capturing p = true)
    (hm : (wordRec .eventRecord [e, s]).Matches (.ext (.cuda .eventRecord), vs ++ callWords w1 (.ext (.cuda .eventRecord)) vs))
    (hx : extCall (.cuda .eventRecord) vs (obsCall w1 (.ext (.cuda .eventRecord)) vs) = some (r, w2)) :
    ∃ id rc clk, (∃ x, w1.dev.eventOf? e = some (id, x)) ∧
      w2.dev = { w1.dev with
                 capture := some { c with race := rc },
                 events := w1.dev.events.set! id (some (some (clk, true))) } ∧ Fixed w1 w2 := by
  have hdrv : ∀ bits, cudaDrv .eventRecord bits (obsCall w1 (.ext (.cuda .eventRecord)) vs) =
      cudaStep .eventRecord bits (obsCall w1 (.ext (.cuda .eventRecord)) vs) := fun bits => by
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
  obtain ⟨⟨id, ev⟩, hev, hcd⟩ := Option.bind_eq_some_iff.mp hcd
  have hp' : (obsCall w1 (.ext (.cuda .eventRecord)) vs).dev.streamParty? s = some p := hp
  rw [hp', Option.bind_some] at hcd
  obtain ⟨d, hkd, rfl⟩ := devOnly_eq hcd
  simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hkd
  obtain ⟨-, rfl⟩ := hkd
  have hid : ∃ x, w1.dev.eventOf? e = some (id, x) := ⟨ev, hev⟩
  have hc' : (obsCall w1 (.ext (.cuda .eventRecord)) vs).dev.capture = some c := hc
  have hcp' : (obsCall w1 (.ext (.cuda .eventRecord)) vs).dev.capturing p = true := hcp
  refine ⟨id, (c.race.issue p).1, (c.race.issue p).2, hid, ?_, Fixed.refl _⟩
  show Dev.recordAt (obsCall w1 (.ext (.cuda .eventRecord)) vs).dev id p = _
  unfold Dev.recordAt
  rw [hc']
  simp only [hcp', if_true]
  rfl

/-- **A stream waiting on a point of the capture joins it**, the device
    otherwise as it was but for the capture's race tracker. -/
theorem streamWait_captured {w1 w2 : World} {vs : List V} {r : Option V} {e s : UInt64} {c : Capture} {id : Nat}
    {clk : Clock} (hhas : w1.has (Ext.cuda .streamWaitEvent).lib = true) (hc : w1.dev.capture = some c)
    (he : w1.dev.eventOf? e = some (id, some (clk, true)))
    (hm : (wordRec .streamWaitEvent [s, e, 0]).Matches
      (.ext (.cuda .streamWaitEvent), vs ++ callWords w1 (.ext (.cuda .streamWaitEvent)) vs))
    (hx : extCall (.cuda .streamWaitEvent) vs (obsCall w1 (.ext (.cuda .streamWaitEvent)) vs) = some (r, w2)) :
    ∃ p rc js, w1.dev.streamParty? s = some p ∧
      w2.dev = { w1.dev with capture := some { c with race := rc, joined := js } } ∧
      (c.origin == p || js.contains p) = true ∧ (∀ q, c.joined.contains q = true → js.contains q = true) ∧
      Fixed w1 w2 := by
  have hdrv : ∀ bits, cudaDrv .streamWaitEvent bits (obsCall w1 (.ext (.cuda .streamWaitEvent)) vs) =
      cudaStep .streamWaitEvent bits (obsCall w1 (.ext (.cuda .streamWaitEvent)) vs) := fun bits => by
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
  obtain ⟨p, hp, hcd⟩ := Option.bind_eq_some_iff.mp hcd
  have he' : (obsCall w1 (.ext (.cuda .streamWaitEvent)) vs).dev.eventOf? e = some (id, some (clk, true)) := he
  rw [he', Option.bind_some] at hcd
  obtain ⟨d', hw, hcd⟩ := Option.bind_eq_some_iff.mp hcd
  obtain ⟨d, hkd, rfl⟩ := devOnly_eq hcd
  simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hkd
  obtain ⟨-, rfl⟩ := hkd
  have hc' : (obsCall w1 (.ext (.cuda .streamWaitEvent)) vs).dev.capture = some c := hc
  simp only [Dev.waitOn, hc', Option.some.injEq] at hw
  subst hw
  refine ⟨p, _, if c.origin == p || c.joined.contains p then c.joined else p :: c.joined, hp, rfl, ?_, ?_, Fixed.refl _⟩
  · by_cases h : (c.origin == p || c.joined.contains p) = true
    · rw [if_pos h]; exact h
    · rw [if_neg h]; simp
  · intro q hq
    split
    · exact hq
    · simp only [List.contains_cons, hq, Bool.or_true]

/-- **A strided product's call on a handle whose stream has joined the
    capture** records the vendor's operation for it, with the scalars the
    record names, the device otherwise as it was but for the capture's race
    tracker. -/
theorem strided_captured {a : Drv.App} {h s0 s1 : UInt64} {ta tb m n k A B C : Nat} {w1 w2 : World}
    {vs : List V} {r : Option V} {c : Capture} {p : Nat}
    (hhas : w1.has (Ext.cublas .sgemmStridedBatched).lib = true) (hc : w1.dev.capture = some c)
    (hp : w1.dev.blasParty? h = some p) (hcp : w1.dev.capturing p = true)
    (hv : gemmInvalid (UInt64.ofNat ta) (UInt64.ofNat tb) (UInt64.ofNat m) (UInt64.ofNat n) (UInt64.ofNat k)
      (UInt64.ofNat (if ta = 1 then k else m)) (UInt64.ofNat (if tb = 1 then n else k)) (UInt64.ofNat m) 1 = false)
    (hb16 : A < 2 ^ 16 ∧ B < 2 ^ 16 ∧ C < 2 ^ 16)
    (hm : (a.stridedRec h [s0, s1] (ta, tb, m, n, k, A, B, C)).Matches
      (.ext (.cublas .sgemmStridedBatched), vs ++ callWords w1 (.ext (.cublas .sgemmStridedBatched)) vs))
    (hx : extCall (.cublas .sgemmStridedBatched) vs (obsCall w1 (.ext (.cublas .sgemmStridedBatched)) vs) = some (r, w2)) :
    ∃ rc, w2.dev = { w1.dev with
                     capture := some { c with
                       race := rc,
                       ops := c.ops.push (.vendor (stridedCall s0 s1 0 0 0 (ta, tb, m, n, k, A, B, C)) [A, B, C] C) } } ∧
      Fixed w1 w2 := by
  obtain ⟨-, ha⟩ := hm
  simp only [Drv.App.stridedRec, List.map_append] at ha
  obtain ⟨bits, hbits, hcd⟩ := extCall_cublas (w := obsCall w1 _ vs) hhas hx
  have hl : vs.length = 18 := by rw [← mapM_asBits_length hbits]; exact cublas_strided_length hcd
  obtain ⟨h18, hsc⟩ := argsOk_split (by rw [hl]; rfl) ha
  rw [mapM_asBits_of_argsOk h18] at hbits
  cases hbits
  simp only [Drv.App.stridedWords] at hcd
  simp only [cublasCall] at hcd
  split at hcd
  · cases hcd
  unfold gemmCall at hcd
  simp only [Option.bind_eq_bind] at hcd
  have hp' : (obsCall w1 (.ext (.cublas .sgemmStridedBatched)) vs).dev.blasParty? h = some p := hp
  rw [hp', Option.bind_some] at hcd
  rw [if_neg (by rw [hv]; exact Bool.false_ne_true)] at hcd
  split at hcd
  · cases hcd
  obtain ⟨⟨ia, oa⟩, hA, hcd⟩ := Option.bind_eq_some_iff.mp hcd
  obtain ⟨⟨ib, ob⟩, hB, hcd⟩ := Option.bind_eq_some_iff.mp hcd
  obtain ⟨⟨ic, oc⟩, hC, hcd⟩ := Option.bind_eq_some_iff.mp hcd
  split at hcd
  · cases hcd
  obtain ⟨alpha, hal, hcd⟩ := Option.bind_eq_some_iff.mp hcd
  obtain ⟨beta, hbe, hcd⟩ := Option.bind_eq_some_iff.mp hcd
  obtain ⟨d, hkd, rfl⟩ := devOnly_eq hcd
  obtain ⟨d0, hd0, hkd⟩ := Option.bind_eq_some_iff.mp hkd
  simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hkd
  obtain ⟨-, rfl⟩ := hkd
  -- the scalars and operands are the record's
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, rfl⟩ :=
    eq_of_length_eighteen vs hl
  have hcw := callWords_strided (w := w1) (mapM_asBits_of_argsOk h18) rfl
  simp only [Drv.App.stridedWords, List.getD_eq_getElem?_getD, List.getElem?_cons_succ, List.getElem?_cons_zero,
    Option.getD_some] at hcw
  have hal' : w1.mem.load (regionBase .arena + UInt64.ofNat a.ONE) 4 = some alpha := hal
  have hbe' : w1.mem.load (regionBase .arena + UInt64.ofNat a.ZERO) 4 = some beta := hbe
  rw [hal', hbe', Option.getD_some, Option.getD_some] at hcw
  rw [hcw] at hsc
  obtain ⟨h0, h1, -⟩ := hsc
  have e0 : alpha = s0 := Option.some.inj h0
  have e1 : beta = s1 := Option.some.inj h1
  subst e0 e1
  obtain ⟨rfl, rfl⟩ := operand_devPtr hb16.1 hA
  obtain ⟨rfl, rfl⟩ := operand_devPtr hb16.2.1 hB
  obtain ⟨rfl, rfl⟩ := operand_devPtr hb16.2.2 hC
  -- the operation is recorded, not run
  have hc' : (obsCall w1 (.ext (.cublas .sgemmStridedBatched)) [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17]).dev.capture = some c := hc
  have hcp' : (obsCall w1 (.ext (.cublas .sgemmStridedBatched)) [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17]).dev.capturing p = true := hcp
  unfold Dev.devOp at hd0
  rw [hc'] at hd0
  simp only [hcp', if_true, Option.bind_eq_bind] at hd0
  obtain ⟨rc, -, hd0⟩ := Option.bind_eq_some_iff.mp hd0
  simp only [Option.some.injEq] at hd0
  subst hd0
  exact ⟨rc, rfl, Fixed.refl _⟩

/-! ## The calls, read in order -/

/-- A call a capture over many streams makes: a launch of kernel `j` on
    stream `s`, a strided product `g` on cuBLAS handle `h`, recording event
    `e` on stream `s`, or stream `s` waiting on event `e`. -/
inductive CapCall where
  | launch (j : Nat) (s : UInt64)
  | gemm (g : Gemm) (h : UInt64)
  | record (e s : UInt64)
  | wait (s e : UInt64)

/-- The record of a call, a product's observed reading scalars `sc`. -/
def CapCall.call (a : Drv.App) (sc : List UInt64) : CapCall → CallRec
  | .launch j s => a.launchRec j s
  | .gemm g h => a.stridedRec h sc g
  | .record e s => wordRec .eventRecord [e, s]
  | .wait s e => wordRec .streamWaitEvent [s, e, 0]

/-- **One call of a capture, read**, with `J` the streams known to have
    joined it and `E` the events known to hold a point of it, each cuBLAS
    handle's stream as `bs` says: a launch or product on a joined stream is
    its step; recording on a joined stream gives the event a point; waiting
    on such an event joins the stream. Anything else is not read. -/
def capStep (a : Drv.App) (bs : UInt64 → Option UInt64) (J E : List UInt64) :
    CapCall → Option (List UInt64 × List UInt64 × List DevCall)
  | .launch j s => if J.contains s && DevCall.okB a (.kernel j) then some (J, E, [.kernel j]) else none
  | .gemm g h => match bs h with
      | some s => if J.contains s && DevCall.okB a (.gemm g) then some (J, E, [.gemm g]) else none
      | none => none
  | .record e s => if J.contains s then some (J, e :: E, []) else none
  | .wait s e => if E.contains e then some (s :: J, E, []) else none

/-- **The steps a capture's calls record**, its calls read in order
    (`capStep`). -/
def capSim (a : Drv.App) (bs : UInt64 → Option UInt64) : List UInt64 → List UInt64 → List CapCall → Option (List DevCall)
  | _, _, [] => some []
  | J, E, c :: cs => match capStep a bs J E c with
      | some (J', E', ds) => (capSim a bs J' E' cs).map (ds ++ ·)
      | none => none

/-- What holds of a world partway through a capture that began from `w0`:
    a capture from party `o` holding `base` and then `ops`; what launches,
    products and handles read as `w0` had them; the streams `J` joined; the
    events `E` holding points of the capture. -/
structure CapInv (w0 : World) (o : Nat) (base : Array DevOp) (J E : List UInt64) (ops : List DevOp)
    (w : World) : Prop where
  cap : ∃ c, w.dev.capture = some c ∧ c.origin = o ∧ c.ops = base ++ ops.toArray
  same : OpSame w0.dev w.dev
  streams : w.dev.streams = w0.dev.streams
  blas : w.dev.blas = w0.dev.blas
  graphs : w.dev.graphs = w0.dev.graphs
  execs : w.dev.execs = w0.dev.execs
  joined : ∀ s ∈ J, ∃ p, w0.dev.streamParty? s = some p ∧ w.dev.capturing p = true
  evs : ∀ e ∈ E, ∃ id clk, w.dev.eventOf? e = some (id, some (clk, true))
  fixed : Fixed w0 w

theorem streamParty_congr {d d' : Dev} (h : d'.streams = d.streams) : d'.streamParty? = d.streamParty? := by
  funext s; simp only [Dev.streamParty?, h]

theorem blasParty_congr {d d' : Dev} (h1 : d'.streams = d.streams) (h2 : d'.blas = d.blas) :
    d'.blasParty? = d.blasParty? := by
  funext s; simp only [Dev.blasParty?, h2, streamParty_congr h1]

theorem capturing_of {d : Dev} {c : Capture} (h : d.capture = some c) (p : Nat) :
    d.capturing p = (c.origin == p || c.joined.contains p) := by
  simp [Dev.capturing, h]

theorem eventOf_set {d : Dev} {id : Nat} {v : Option (Clock × Bool)} {e : UInt64} {id' : Nat}
    {x : Option (Clock × Bool)} (h : d.eventOf? e = some (id', x)) :
    ({ d with events := d.events.set! id (some v) } : Dev).eventOf? e = some (id', if id = id' then v else x) := by
  unfold Dev.eventOf? at h ⊢
  simp only [Option.bind_eq_bind] at h ⊢
  obtain ⟨i, hi, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨ev, hev, h⟩ := Option.bind_eq_some_iff.mp h
  simp only [Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  rw [hi, Option.bind_some]
  have hlt : i < d.events.size := by
    rcases hx : d.events[i]? with _ | _
    · rw [hx] at hev; cases hev
    · exact (Array.getElem?_eq_some_iff.mp hx).1
  simp only [Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds]
  by_cases hid : id = i
  · subst hid; simp [hlt]
  · simp only [hid, if_false]
    rw [hev]; rfl

theorem eventOf_congr {d d' : Dev} (h : d'.events = d.events) : d'.eventOf? = d.eventOf? := by
  funext e; simp only [Dev.eventOf?, h]

theorem capturing_mono {d d' : Dev} {c c' : Capture} (hc : d.capture = some c) (hc' : d'.capture = some c')
    (ho : c'.origin = c.origin) (hj : ∀ q, c.joined.contains q = true → c'.joined.contains q = true) :
    ∀ p, d.capturing p = true → d'.capturing p = true := by
  intro p h
  rw [capturing_of hc] at h
  rw [capturing_of hc', ho]
  simp only [Bool.or_eq_true] at h ⊢
  exact h.imp id (hj p)

/-- The invariant past a call that changed only the capture and the events. -/
theorem CapInv.next {w0 : World} {o : Nat} {base : Array DevOp} {J E J' E' : List UInt64} {ops ops' : List DevOp}
    {w1 w2 : World} (hI : CapInv w0 o base J E ops w1) {cc' : Capture} {ev : Array (Option (Option (Clock × Bool)))}
    (hd : w2.dev = { w1.dev with capture := some cc', events := ev }) (ho : cc'.origin = o)
    (hops : cc'.ops = base ++ ops'.toArray)
    (hJ : ∀ s ∈ J', ∃ p, w0.dev.streamParty? s = some p ∧ w2.dev.capturing p = true)
    (hE : ∀ e ∈ E', ∃ id clk, w2.dev.eventOf? e = some (id, some (clk, true))) (hf : Fixed w1 w2) :
    CapInv w0 o base J' E' ops' w2 where
  cap := ⟨cc', by rw [hd], ho, hops⟩
  same := by
    rw [hd]
    exact ⟨hI.same.funcs, hI.same.modules, hI.same.bufs, hI.same.kernelWrites⟩
  streams := by rw [hd]; exact hI.streams
  blas := by rw [hd]; exact hI.blas
  graphs := by rw [hd]; exact hI.graphs
  execs := by rw [hd]; exact hI.execs
  joined := hJ
  evs := hE
  fixed := hI.fixed.trans hf

theorem CapInv.congr {w0 : World} {o : Nat} {base : Array DevOp} {J E : List UInt64} {ops : List DevOp}
    {w w' : World} (hI : CapInv w0 o base J E ops w) (hd : w'.dev = w.dev) (hf : Fixed w w') :
    CapInv w0 o base J E ops w' where
  cap := by rw [hd]; exact hI.cap
  same := by rw [hd]; exact hI.same
  streams := by rw [hd]; exact hI.streams
  blas := by rw [hd]; exact hI.blas
  graphs := by rw [hd]; exact hI.graphs
  execs := by rw [hd]; exact hI.execs
  joined := by rw [hd]; exact hI.joined
  evs := by rw [hd]; exact hI.evs
  fixed := hI.fixed.trans hf

theorem push_ops (base : Array DevOp) (ops : List DevOp) (op : DevOp) :
    (base ++ ops.toArray).push op = base ++ (ops ++ [op]).toArray := by
  apply Array.toList_inj.mp
  simp

/-- **A call of a capture, read by `capStep`, records the operations of the
    steps it reads**, and keeps the invariant with what it reads next. -/
theorem capStep_sound {a : Drv.App} {bs : UInt64 → Option UInt64} {s0 s1 : UInt64} {w0 : World} {o : Nat}
    {base : Array DevOp} {J E J' E' : List UInt64} {ops : List DevOp} {ds : List DevCall} {w1 w2 : World}
    {c : CapCall} {e : Ext} {vs : List V} {r : Option V}
    (hcu : w0.has (Ext.cuda .launchKernel).lib = true)
    (hbl : w0.has (Ext.cublas .sgemmStridedBatched).lib = true)
    (hbs : ∀ h s, bs h = some s → w0.dev.blasParty? h = w0.dev.streamParty? s)
    (hI : CapInv w0 o base J E ops w1)
    (hm : (c.call a [s0, s1]).Matches (.ext e, vs ++ callWords w1 (.ext e) vs))
    (hx : extCall e vs (obsCall w1 (.ext e) vs) = some (r, w2))
    (hs : capStep a bs J E c = some (J', E', ds)) :
    ∃ ops1, ds.mapM (DevCall.op? a w0 s0 s1) = some ops1 ∧ CapInv w0 o base J' E' (ops ++ ops1) w2 := by
  obtain ⟨cc, hcc, hco, hcops⟩ := hI.cap
  have hcu1 : w1.has (Ext.cuda .launchKernel).lib = true := has_fixed hI.fixed hcu
  have hp1 : w1.dev.streamParty? = w0.dev.streamParty? := streamParty_congr hI.streams
  cases c with
  | launch j s =>
      simp only [capStep] at hs
      split at hs
      · rename_i hc
        simp only [Option.some.injEq, Prod.mk.injEq] at hs
        obtain ⟨rfl, rfl, rfl⟩ := hs
        simp only [Bool.and_eq_true] at hc
        obtain ⟨hJs, hok⟩ := hc
        obtain ⟨k, hk, hj, hdims⟩ := DevCall.okB_ok hok
        have he : Callee.ext e = .ext (.cuda .launchKernel) := by
          have := hm.1
          simp only [CapCall.call, Drv.App.launchRec, hk] at this
          exact this.symm
        injection he with he
        subst he
        obtain ⟨p, hp0, hcp⟩ := hI.joined s (List.contains_iff_mem.mp hJs)
        obtain ⟨op, r', hop, hd2, hf⟩ := launchRec_captured hk hj hcu1 hcc (by rw [hp1]; exact hp0) hcp hdims hm hx
        have hop' : DevCall.op? a w0 s0 s1 (.kernel j) = some op := by
          rw [← op?_congr hI.same]; exact hop
        refine ⟨[op], by simp [hop'], hI.next (cc' := { cc with race := r', ops := cc.ops.push op })
          (ev := w1.dev.events) hd2 hco (by rw [hcops]; exact push_ops _ _ _) ?_ ?_ hf⟩
        · intro s' hs'
          obtain ⟨p', h1, h2⟩ := hI.joined s' hs'
          exact ⟨p', h1, capturing_mono hcc (congrArg Dev.capture hd2) rfl (fun _ h => h) p' h2⟩
        · intro e' he'
          rw [hd2]; exact hI.evs e' he'
      · cases hs
  | gemm g h =>
      obtain ⟨ta, tb, m, n, k, A, B, C⟩ := g
      simp only [capStep] at hs
      split at hs
      · rename_i s hbsh
        split at hs
        · rename_i hc
          simp only [Option.some.injEq, Prod.mk.injEq] at hs
          obtain ⟨rfl, rfl, rfl⟩ := hs
          simp only [Bool.and_eq_true] at hc
          obtain ⟨hJs, hok⟩ := hc
          obtain ⟨hv, hb1, hb2, hb3⟩ := DevCall.okB_ok hok
          have he : Callee.ext e = .ext (.cublas .sgemmStridedBatched) := hm.1.symm
          injection he with he
          subst he
          obtain ⟨p, hp0, hcp⟩ := hI.joined s (List.contains_iff_mem.mp hJs)
          have hbp : w1.dev.blasParty? h = some p := by
            rw [blasParty_congr hI.streams hI.blas, hbs h s hbsh]; exact hp0
          obtain ⟨r', hd2, hf⟩ := strided_captured (has_fixed hI.fixed hbl) hcc hbp hcp hv ⟨hb1, hb2, hb3⟩ hm hx
          refine ⟨[_], rfl, hI.next (ev := w1.dev.events) hd2 hco (by rw [hcops]; exact push_ops _ _ _) ?_ ?_ hf⟩
          · intro s' hs'
            obtain ⟨p', h1, h2⟩ := hI.joined s' hs'
            exact ⟨p', h1, capturing_mono hcc (congrArg Dev.capture hd2) rfl (fun _ h => h) p' h2⟩
          · intro e' he'
            rw [hd2]; exact hI.evs e' he'
        · cases hs
      · cases hs
  | record ev s =>
      simp only [capStep] at hs
      split at hs
      · rename_i hJs
        simp only [Option.some.injEq, Prod.mk.injEq] at hs
        obtain ⟨rfl, rfl, rfl⟩ := hs
        have he : Callee.ext e = .ext (.cuda .eventRecord) := hm.1.symm
        injection he with he
        subst he
        obtain ⟨p, hp0, hcp⟩ := hI.joined s (List.contains_iff_mem.mp hJs)
        obtain ⟨id, rc, clk, ⟨x, hx0⟩, hd2, hf⟩ := eventRecord_captured (has_cuda hcu1) hcc
          (by rw [hp1]; exact hp0) hcp hm hx
        refine ⟨[], rfl, ?_⟩
        rw [List.append_nil]
        refine hI.next hd2 hco hcops ?_ ?_ hf
        · intro s' hs'
          obtain ⟨p', h1, h2⟩ := hI.joined s' hs'
          exact ⟨p', h1, capturing_mono hcc (congrArg Dev.capture hd2) rfl (fun _ h => h) p' h2⟩
        · intro e' he'
          rw [hd2]
          rcases List.mem_cons.mp he' with rfl | he'
          · refine ⟨id, clk, ?_⟩
            rw [eventOf_set (d := { w1.dev with capture := some { cc with race := rc } }) hx0]
            simp
          · obtain ⟨id', clk', h'⟩ := hI.evs e' he'
            rw [eventOf_set (d := { w1.dev with capture := some { cc with race := rc } }) h']
            split
            · exact ⟨id', clk, rfl⟩
            · exact ⟨id', clk', rfl⟩
      · cases hs
  | wait s ev =>
      simp only [capStep] at hs
      split at hs
      · rename_i hEe
        simp only [Option.some.injEq, Prod.mk.injEq] at hs
        obtain ⟨rfl, rfl, rfl⟩ := hs
        have he : Callee.ext e = .ext (.cuda .streamWaitEvent) := hm.1.symm
        injection he with he
        subst he
        obtain ⟨id, clk, hev⟩ := hI.evs ev (List.contains_iff_mem.mp hEe)
        obtain ⟨p, rc, js, hp, hd2, hjp, hmono, hf⟩ := streamWait_captured (has_cuda hcu1) hcc hev hm hx
        refine ⟨[], rfl, ?_⟩
        rw [List.append_nil]
        have hcc2 : w2.dev.capture = some { cc with race := rc, joined := js } := by rw [hd2]
        refine hI.next (ev := w1.dev.events) hd2 hco hcops ?_ ?_ hf
        · intro s' hs'
          rcases List.mem_cons.mp hs' with rfl | hs'
          · refine ⟨p, by rw [← hp1]; exact hp, ?_⟩
            rw [capturing_of hcc2]; exact hjp
          · obtain ⟨p', h1, h2⟩ := hI.joined s' hs'
            exact ⟨p', h1, capturing_mono hcc hcc2 rfl hmono p' h2⟩
        · intro e' he'
          rw [hd2]; exact hI.evs e' he'
      · cases hs

/-- **A run through a capture's calls records the operations of the steps
    `capSim` reads off them**, in order, keeping the invariant. -/
theorem extRun_dag {a : Drv.App} {bs : UInt64 → Option UInt64} {s0 s1 : UInt64} {w0 : World} {o : Nat}
    {base : Array DevOp}
    (hcu : w0.has (Ext.cuda .launchKernel).lib = true)
    (hbl : w0.has (Ext.cublas .sgemmStridedBatched).lib = true)
    (hbs : ∀ h s, bs h = some s → w0.dev.blasParty? h = w0.dev.streamParty? s) :
    ∀ (cs : List CapCall) (J E : List UInt64) (ops : List DevOp) (ds : List DevCall) {w1 w : World}
      {xs : List (Callee × List V)}, CapInv w0 o base J E ops w1 → ExtRun w1 xs w →
      recsMatch (cs.map (CapCall.call a [s0, s1])) xs → capSim a bs J E cs = some ds →
      ∃ ops' J' E', ds.mapM (DevCall.op? a w0 s0 s1) = some ops' ∧ CapInv w0 o base J' E' (ops ++ ops') w
  | [], J, E, ops, ds, w1, w, xs, hI, hr, hm, hs => by
      simp only [capSim, Option.some.injEq] at hs
      subst hs
      have hxs : xs = [] := by cases xs <;> simp_all [recsMatch]
      subst hxs
      obtain ⟨hd, hf, -⟩ := hr.nil_inv
      exact ⟨[], J, E, rfl, by rw [List.append_nil]; exact hI.congr hd hf⟩
  | c :: cs, J, E, ops, ds, w1, w, xs, hI, hr, hm, hs => by
      cases xs with
      | nil => exact hm.elim
      | cons x xs =>
          obtain ⟨hm1, hmr⟩ := hm
          simp only [capSim] at hs
          split at hs
          · rename_i J1 E1 d1 hst
            obtain ⟨ds', hs', rfl⟩ := Option.map_eq_some_iff.mp hs
            obtain ⟨w2, hr1, hr2⟩ := hr.split [x] xs rfl
            obtain ⟨wa, wb, e, vs, r, hda, hfa, rfl, hx, hdb, hfb, -, -, -⟩ := hr1.one_inv
            have hIa := hI.congr hda hfa
            obtain ⟨ops1, hops1, hIb⟩ := capStep_sound hcu hbl hbs hIa hm1 hx hst
            obtain ⟨ops2, J2, E2, hops2, hI2⟩ :=
              extRun_dag hcu hbl hbs cs J1 E1 (ops ++ ops1) ds' (hIb.congr hdb hfb) hr2 hmr hs'
            refine ⟨ops1 ++ ops2, J2, E2, ?_, by rw [← List.append_assoc]; exact hI2⟩
            rw [List.mapM_append, hops1, hops2]
            rfl
          · cases hs

/-! ## A capture's run, over many streams -/

/-- **Ending a capture** makes its graph of the operations it recorded, or,
    a joined stream not rejoined, makes none; either way the capture ends. -/
theorem endCapture_ends {w1 w2 : World} {vs : List V} {r : Option V} {s pg : UInt64} {c : Capture} {p : Nat}
    (hhas : w1.has (Ext.cuda .endCapture).lib = true) (hc : w1.dev.capture = some c)
    (hp : w1.dev.streamParty? s = some p) (ho : c.origin = p)
    (hm : (wordRec .endCapture [s, pg]).Matches (.ext (.cuda .endCapture), vs ++ callWords w1 (.ext (.cuda .endCapture)) vs))
    (hx : extCall (.cuda .endCapture) vs (obsCall w1 (.ext (.cuda .endCapture)) vs) = some (r, w2)) :
    (w2.dev = { w1.dev.endCapture with graphs := w1.dev.graphs.push (some c.ops) } ∨ w2.dev = w1.dev.endCapture) ∧
      Fixed w1 w2 := by
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
  rw [if_neg (by simp [ho])] at hcd
  split at hcd
  · obtain ⟨d, hkd, rfl⟩ := devOnly_eq hcd
    simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hkd
    obtain ⟨-, rfl⟩ := hkd
    exact ⟨.inr rfl, Fixed.refl _⟩
  · obtain ⟨m, -, hcd⟩ := Option.bind_eq_some_iff.mp hcd
    simp only [Option.some.injEq, Prod.mk.injEq] at hcd
    obtain ⟨-, rfl⟩ := hcd
    exact ⟨.inl rfl, Fixed.refl _⟩

/-- **Instantiating graph `gi` answers only for a graph there is.** -/
theorem graphInstantiate_some {w1 w2 : World} {vs : List V} {r : Option V} {pe : UInt64} {gi : Nat}
    (hhas : w1.has (Ext.cuda .graphInstantiate).lib = true) (hcap : w1.dev.capture = none) (hgi : gi < 2 ^ 48)
    (hm : (wordRec .graphInstantiate [pe, drvHandle kGraph gi, 0]).Matches
      (.ext (.cuda .graphInstantiate), vs ++ callWords w1 (.ext (.cuda .graphInstantiate)) vs))
    (hx : extCall (.cuda .graphInstantiate) vs (obsCall w1 (.ext (.cuda .graphInstantiate)) vs) = some (r, w2)) :
    ∃ ops, w1.dev.graphs[gi]? = some (some ops) := by
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
  simp only [Option.bind_eq_bind, handleOf_drvHandle (by decide : kGraph < 2 ^ 8) hgi, Option.bind_some] at hcd
  obtain ⟨ops, hg, -⟩ := Option.bind_eq_some_iff.mp hcd
  refine ⟨ops, ?_⟩
  have h : (w1.dev.graphs[gi]?).join = some ops := hg
  exact Option.join_eq_some_iff.mp h

/-- **Waiting for the device** changes only the race tracker. -/
theorem ctxSync_done {w1 w2 : World} {vs : List V} {r : Option V}
    (hhas : w1.has (Ext.cuda .ctxSynchronize).lib = true) (hcap : w1.dev.capture = none)
    (hm : (wordRec .ctxSynchronize []).Matches
      (.ext (.cuda .ctxSynchronize), vs ++ callWords w1 (.ext (.cuda .ctxSynchronize)) vs))
    (hx : extCall (.cuda .ctxSynchronize) vs (obsCall w1 (.ext (.cuda .ctxSynchronize)) vs) = some (r, w2)) :
    ∃ rc, w2.dev = { w1.dev with race := rc } ∧ Fixed w1 w2 := by
  have hcap' : (obsCall w1 (.ext (.cuda .ctxSynchronize)) vs).dev.capture = none := hcap
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
  obtain ⟨d, hkd, rfl⟩ := devOnly_eq hcd
  simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hkd
  obtain ⟨-, rfl⟩ := hkd
  exact ⟨_, rfl, Fixed.refl _⟩

/-- The records of capturing, on stream `s`, the calls `cc`, then ending
    the capture into the graph whose handle is stored at `pg`,
    instantiating it, its handle stored at `pe`, uploading it, and waiting
    for the stream. -/
def dagRecs (a : Drv.App) (sc : List UInt64) (cc : List CapCall) (s pg pe : UInt64) : List CallRec :=
  [wordRec .beginCapture [s, 2]] ++ cc.map (CapCall.call a sc) ++
    [wordRec .endCapture [s, pg], wordRec .graphInstantiate [pe, drvHandle kGraph 0, 0],
     wordRec .graphUpload [drvHandle kExec 0, s], wordRec .streamSynchronize [s]]

/-- **A capture over many streams leaves its graph holding the operations
    of the steps `capSim` reads off its calls**: from a world with the
    driver and cuBLAS loaded, no capture in progress and no graph or
    executable graph yet, a run through a capture begun on stream `s` of
    calls `cc` that `capSim` reads, from `s` alone joined, as steps `ds`,
    leaves executable graph `0` holding those steps' operations, in order,
    no capture in progress, and the device's functions, modules,
    allocations and their contents and the oracles as they were. -/
theorem extRun_dagCapture {a : Drv.App} {bs : UInt64 → Option UInt64} {s0 s1 : UInt64} {cc : List CapCall}
    {ds : List DevCall} {s pg pe : UInt64} {w0 : World}
    (hcu : w0.has (Ext.cuda .launchKernel).lib = true)
    (hbl : w0.has (Ext.cublas .sgemmStridedBatched).lib = true)
    (hbs : ∀ h s, bs h = some s → w0.dev.blasParty? h = w0.dev.streamParty? s)
    (hcap : w0.dev.capture = none) (hg : w0.dev.graphs.size = 0) (he : w0.dev.execs.size = 0)
    (hsim : capSim a bs [s] [] cc = some ds)
    {cs : List (Callee × List V)} {w : World} (hr : ExtRun w0 cs w)
    (hm : recsMatch (dagRecs a [s0, s1] cc s pg pe) cs) :
    ∃ ops, ds.mapM (DevCall.op? a w0 s0 s1) = some ops ∧
      w.dev.execs[0]? = some (some ops.toArray) ∧ w.dev.capture = none ∧ OpSame w0.dev w.dev ∧ Fixed w0 w := by
  unfold dagRecs at hm
  obtain ⟨c1L, c4567, rfl, hm1L, hm4567⟩ := recsMatch_split hm
  obtain ⟨c1, cL, rfl, hm1, hmL⟩ := recsMatch_split (a := [_]) hm1L
  obtain ⟨c4, c567, rfl, hm4, hm567⟩ := recsMatch_split (a := [_]) hm4567
  obtain ⟨c5, c67, rfl, hm5, hm67⟩ := recsMatch_split (a := [_]) hm567
  obtain ⟨c6, c7, rfl, hm6, hm7⟩ := recsMatch_split (a := [_]) hm67
  obtain ⟨wL, hr1L, hr4567⟩ := hr.split _ _ rfl
  obtain ⟨w1, hr1, hrL⟩ := hr1L.split _ _ rfl
  obtain ⟨w4, hr4, hr567⟩ := hr4567.split _ _ rfl
  obtain ⟨w5, hr5, hr67⟩ := hr567.split _ _ rfl
  obtain ⟨w6, hr6, hr7⟩ := hr67.split _ _ rfl
  -- the capture begins
  obtain ⟨x1, y1, vs1, r1, ex1, fx1, hm1', hx1, ey1, fy1⟩ := hr1.one_word hm1
  obtain ⟨p, hp, hd1, f1⟩ := beginCapture_done (has_cuda (has_fixed fx1 hcu)) (by rw [ex1]; exact hcap) hm1' hx1
  rw [ex1] at hp
  have hd1' : w1.dev = { w0.dev with capture := some ⟨p, [], {}, #[]⟩ } := by rw [ey1, hd1, ex1]
  have fw1 : Fixed w0 w1 := fx1.trans (f1.trans fy1)
  have hI1 : CapInv w0 p #[] [s] [] [] w1 := by
    refine ⟨⟨_, by rw [hd1'], rfl, rfl⟩, by rw [hd1']; exact ⟨rfl, rfl, rfl, rfl⟩, by rw [hd1'], by rw [hd1'],
      by rw [hd1'], by rw [hd1'], ?_, fun _ h => absurd h (List.not_mem_nil), fw1⟩
    intro s' hs'
    rw [List.mem_singleton.mp hs']
    exact ⟨p, hp, by rw [capturing_of (c := ⟨p, [], {}, #[]⟩) (by rw [hd1'])]; simp⟩
  -- the calls are recorded
  obtain ⟨ops, J', E', hops, hIL⟩ := extRun_dag hcu hbl hbs cc [s] [] [] ds hI1 hrL hmL hsim
  obtain ⟨c, hcL, hco, hcops⟩ := hIL.cap
  -- the capture ends, and makes its graph, since graph 0 is instantiated
  obtain ⟨x4, y4, vs4, r4, ex4, fx4, hm4', hx4, ey4, fy4⟩ := hr4.one_word hm4
  have hpL : wL.dev.streamParty? s = some p := by rw [streamParty_congr hIL.streams]; exact hp
  obtain ⟨hd4, f4⟩ := endCapture_ends (has_cuda (has_fixed (hIL.fixed.trans fx4) hcu)) (by rw [ex4]; exact hcL)
    (by rw [ex4]; exact hpL) hco hm4' hx4
  have hgL : wL.dev.graphs = #[] := by rw [hIL.graphs]; exact Array.eq_empty_of_size_eq_zero hg
  have heL : wL.dev.execs = #[] := by rw [hIL.execs]; exact Array.eq_empty_of_size_eq_zero he
  have fw4 : Fixed w0 w4 := hIL.fixed.trans (fx4.trans (f4.trans fy4))
  have hcap4 : w4.dev.capture = none := by rw [ey4]; rcases hd4 with h | h <;> rw [h] <;> rfl
  obtain ⟨x5, y5, vs5, r5, ex5, fx5, hm5', hx5, ey5, fy5⟩ := hr5.one_word hm5
  obtain ⟨g0, hg0⟩ := graphInstantiate_some (has_cuda (has_fixed (fw4.trans fx5) hcu)) (by rw [ex5]; exact hcap4)
    (by decide) hm5' hx5
  have hg4 : w4.dev.graphs[0]? = some (some c.ops) := by
    rw [ex5] at hg0
    rw [ey4]
    rcases hd4 with h | h
    · rw [h, ex4]; simp [Dev.endCapture, hgL]
    · rw [ey4, h, ex4] at hg0; simp [Dev.endCapture, hgL] at hg0
  -- the graph is instantiated
  obtain ⟨hd5, f5⟩ := graphInstantiate_done (has_cuda (has_fixed (fw4.trans fx5) hcu)) (by rw [ex5]; exact hcap4)
    (by decide) (by rw [ex5]; exact hg4) hm5' hx5
  have hd5' : w5.dev = { w4.dev with execs := w4.dev.execs.push (some c.ops) } := by rw [ey5, hd5, ex5]
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
  have hops' : c.ops = ops.toArray := by rw [hcops]; simp
  have hsame4 : OpSame w0.dev w4.dev := by
    rw [ey4]
    rcases hd4 with h | h <;> rw [h, ex4] <;>
      exact ⟨hIL.same.funcs, hIL.same.modules, hIL.same.bufs, hIL.same.kernelWrites⟩
  refine ⟨ops, hops, ?_, ?_, ?_, fw7⟩
  · rw [hd7', hd6', hd5', ← hops']
    have he4 : w4.dev.execs = #[] := by
      rw [ey4]; rcases hd4 with h | h <;> rw [h, ex4] <;> exact heL
    simp [he4]
  · rw [hd7', hd6', hd5', hcap4]
  · rw [hd7', hd6', hd5']
    exact ⟨hsame4.funcs, hsame4.modules, hsame4.bufs, hsame4.kernelWrites⟩

end AlgorithmLib.ML
