module
public import AlgorithmLib.Host.RaceRefine
meta import AlgorithmLib.Host.RaceRefine
public import AlgorithmLib.Host.Ffi
meta import AlgorithmLib.Host.Ffi
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The World's device state always holds a tracker the proof covers

`TrackOk` says the device's race tracker, and the tracker of a capture in
progress, are states `CReached` covers --- so `CReached.ordered` and the
refinement apply to them --- and that every event's clock is a snapshot of the
history it belongs to (or empty, once its capture has ended). **`callBits_track`**:
every contract keeps it. So what the World's device contracts accept, outside a
capture and inside one, is what the abstract tracker accepts.
-/

namespace AlgorithmLib.Device

open AlgorithmLib.HProg.Sem

/-- Every event recorded with flag `fl` holds a clock of `evs`, or none. -/
def EvOk (evs : List Clock) (fl : Bool) (d : Dev) : Prop :=
  ∀ (i : Nat) (clk : Clock), d.events[i]? = some (some (some (clk, fl))) → clk = #[] ∨ clk ∈ evs

structure TrackOk (d : Dev) : Prop where
  outer : ∃ evs, CReached d.race evs ∧ EvOk evs false d
  inner : match d.capture with
    | none => EvOk [] true d
    | some c => ∃ evs, CReached c.race evs ∧ EvOk evs true d

theorem TrackOk.congr {d d' : Dev} (h : TrackOk d) (hr : d'.race = d.race)
    (hc : d'.capture = d.capture) (he : d'.events = d.events) : TrackOk d' := by
  obtain ⟨⟨evs, h1, h2⟩, h3⟩ := h
  refine ⟨⟨evs, hr ▸ h1, fun i clk hi => h2 i clk (he ▸ hi)⟩, ?_⟩
  rw [hc]
  cases hcap : d.capture with
  | none => rw [hcap] at h3; exact fun i clk hi => h3 i clk (he ▸ hi)
  | some c =>
      rw [hcap] at h3
      obtain ⟨evs', h4, h5⟩ := h3
      exact ⟨evs', h4, fun i clk hi => h5 i clk (he ▸ hi)⟩

theorem EvOk.mono {evs evs' : List Clock} {fl : Bool} {d : Dev} (h : EvOk evs fl d)
    (hs : ∀ c ∈ evs, c ∈ evs') : EvOk evs' fl d := fun i clk hi =>
  (h i clk hi).imp id (hs clk)

theorem Race.op_nil (r : Race) (p : Nat) : r.op p [] [] = some (r.issue p).1 := by
  simp [Race.op, Race.access]

theorem Race.issue_clock (r : Race) (p : Nat) : (r.issue p).2 = (r.issue p).1.clock p := by
  simp [Race.issue, Race.clock_setClock]

theorem put_track {d : Dev} (h : TrackOk d) (id : Nat) (b : Option ByteArray) :
    TrackOk (d.put id b) := h.congr rfl rfl rfl

theorem putAll_meta : ∀ (ws : List (Nat × ByteArray)) (d : Dev),
    let d' := ws.foldl (fun d (x : Nat × ByteArray) => d.put x.1 (some x.2)) d
    d'.race = d.race ∧ d'.capture = d.capture ∧ d'.events = d.events
  | [], _ => ⟨rfl, rfl, rfl⟩
  | p :: ws, d => putAll_meta ws (d.put p.1 (some p.2))

theorem runLaunch_meta {d d' : Dev} {k : Launch → List ByteArray → List ByteArray} {l : Launch}
    {ids : List Nat} (h : d.runLaunch k l ids = some d') :
    d'.race = d.race ∧ d'.capture = d.capture ∧ d'.events = d.events := by
  unfold Dev.runLaunch at h
  obtain ⟨ins, -, h⟩ := Option.bind_eq_some_iff.mp h
  dsimp only at h
  split at h
  · cases h
  · simp only [Option.some.injEq] at h
    subst h
    exact putAll_meta _ d

theorem runVendor_meta {d d' : Dev} {v : VendorCall → List ByteArray → ByteArray} {c : VendorCall}
    {ins : List Nat} {out : Nat} (h : d.runVendor v c ins out = some d') :
    d'.race = d.race ∧ d'.capture = d.capture ∧ d'.events = d.events := by
  unfold Dev.runVendor at h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
  dsimp only at h
  split at h
  · cases h
  · simp only [Option.some.injEq] at h; subst h; exact ⟨rfl, rfl, rfl⟩

/-- **One device operation keeps the tracker covered**: on a capturing party it
    is a move of the capture's tracker, otherwise of the device's. -/
theorem devOp_track {d d' : Dev} {w : World} {p : Nat} {op : DevOp} {rs ws : List Nat}
    (h : d.devOp w p op rs ws = some d') (ht : TrackOk d) : TrackOk d' := by
  unfold Dev.devOp at h
  have run_ok : ∀ {d'}, Dev.devOp.run d w p op rs ws = some d' → TrackOk d' := by
    intro d' h
    unfold Dev.devOp.run at h
    obtain ⟨r, hr, h⟩ := Option.bind_eq_some_iff.mp h
    have ht1 : TrackOk { d with race := r } := by
      obtain ⟨⟨evs, h1, h2⟩, h3⟩ := ht
      exact ⟨⟨evs, .op p rs ws h1 hr, h2⟩, h3⟩
    split at h
    · obtain ⟨e1, e2, e3⟩ := runLaunch_meta h
      exact ht1.congr e1 e2 e3
    · obtain ⟨e1, e2, e3⟩ := runVendor_meta h
      exact ht1.congr e1 e2 e3
  split at h
  · rename_i c hcap
    split at h
    · obtain ⟨r, hr, h⟩ := Option.bind_eq_some_iff.mp h
      simp only [Option.some.injEq] at h
      subst h
      obtain ⟨ho, hi⟩ := ht
      rw [hcap] at hi
      obtain ⟨evs, h1, h2⟩ := hi
      exact ⟨ho, ⟨evs, .op p rs ws h1 hr, h2⟩⟩
    · exact run_ok h
  · exact run_ok h

theorem syncWrite_track {d d' : Dev} {id : Nat} {b : ByteArray} (h : d.syncWrite id b = some d')
    (ht : TrackOk d) : TrackOk d' := by
  unfold Dev.syncWrite at h
  obtain ⟨r, hr, h⟩ := Option.bind_eq_some_iff.mp h
  simp only [Option.some.injEq] at h
  subst h
  obtain ⟨⟨evs, h1, h2⟩, h3⟩ := ht
  exact ⟨⟨evs, .learnParty hostParty defaultParty (.op defaultParty [] [id] h1 hr), h2⟩, h3⟩

theorem syncRead_track {d d' : Dev} {id : Nat} (h : d.syncRead id = some d') (ht : TrackOk d) :
    TrackOk d' := by
  unfold Dev.syncRead at h
  obtain ⟨r, hr, h⟩ := Option.bind_eq_some_iff.mp h
  simp only [Option.some.injEq] at h
  subst h
  obtain ⟨⟨evs, h1, h2⟩, h3⟩ := ht
  exact ⟨⟨evs, .learnParty hostParty defaultParty (.op defaultParty [id] [] h1 hr), h2⟩, h3⟩

theorem devOnly_track {w : World} {k : Option (Option V × Dev)} {r : Option V} {w' : World}
    (h : devOnly w k = some (r, w')) (hk : ∀ r d, k = some (r, d) → TrackOk d) : TrackOk w'.dev := by
  unfold devOnly at h
  obtain ⟨⟨r0, d⟩, hkd, he⟩ := Option.map_eq_some_iff.mp h
  simp only [Prod.mk.injEq] at he
  obtain ⟨-, rfl⟩ := he
  exact hk _ _ hkd

theorem track_fresh {d : Dev} (hr : d.race = {}) (hc : d.capture = none) (he : d.events = #[]) :
    TrackOk d := by
  refine ⟨⟨[], hr ▸ .init, fun i clk h => by rw [he] at h; simp at h⟩, ?_⟩
  rw [hc]; exact fun i clk h => by rw [he] at h; simp at h

theorem memInfoOf_track {w : World} {ctx : UInt64} {pick : UInt64 × UInt64 → UInt64}
    {r : Option V} {d : Dev} (hk : memInfoOf w ctx pick = some (r, d)) (ht : TrackOk w.dev) :
    TrackOk d := by
  unfold memInfoOf at hk
  obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
  split at hk <;> (simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht)

/-- `EvOk` over an events array. -/
theorem evOk_iff (evs : List Clock) (fl : Bool) (d : Dev) :
    EvOk evs fl d ↔ ∀ (i : Nat) (clk : Clock), d.events[i]? = some (some (some (clk, fl))) →
      clk = #[] ∨ clk ∈ evs := Iff.rfl

theorem evOk_set {evs : List Clock} {fl : Bool} {d d' : Dev} (h : EvOk evs fl d) (e : Nat)
    (v : Option (Option (Clock × Bool)))
    (hv : ∀ clk, v = some (some (clk, fl)) → clk = #[] ∨ clk ∈ evs)
    (he : d'.events = d.events.set! e v) : EvOk evs fl d' := by
  intro i clk hi
  rw [he, Array.set!_eq_setIfInBounds, Array.getElem?_setIfInBounds] at hi
  split at hi
  · split at hi
    · simp only [Option.some.injEq] at hi; exact hv clk hi
    · cases hi
  · exact h i clk hi

theorem evOk_push {evs : List Clock} {fl : Bool} {d d' : Dev} (h : EvOk evs fl d)
    (he : d'.events = d.events.push (some none)) : EvOk evs fl d' := by
  intro i clk hi
  rw [he, Array.getElem?_push] at hi
  split at hi
  · simp at hi
  · exact h i clk hi

theorem evOk_endCapture_false {evs : List Clock} {d : Dev} (h : EvOk evs false d) :
    EvOk evs false d.endCapture := by
  intro i clk hi
  simp only [Dev.endCapture, Array.getElem?_map] at hi
  cases e : d.events[i]? with
  | none => rw [e] at hi; cases hi
  | some x =>
    rw [e] at hi
    simp only [Option.map_some, Option.some.injEq] at hi
    split at hi
    · simp at hi
    · subst hi; exact h i clk e

theorem evOk_endCapture_true {d : Dev} : EvOk [] true d.endCapture := by
  intro i clk hi
  simp only [Dev.endCapture, Array.getElem?_map] at hi
  cases e : d.events[i]? with
  | none => rw [e] at hi; cases hi
  | some x =>
    rw [e] at hi
    simp only [Option.map_some, Option.some.injEq] at hi
    split at hi
    · simp only [Option.some.injEq, Prod.mk.injEq] at hi; exact Or.inl hi.1.symm
    · rename_i hne; subst hi; exact absurd rfl (hne clk)

theorem track_race {d : Dev} (ht : TrackOk d) {r : Race}
    (hr : ∀ evs, CReached d.race evs → CReached r evs) (d' : Dev) (h1 : d'.race = r)
    (h2 : d'.capture = d.capture) (h3 : d'.events = d.events) : TrackOk d' := by
  obtain ⟨⟨evs, c1, c2⟩, c3⟩ := ht
  refine ⟨⟨evs, h1 ▸ hr evs c1, fun i clk hi => c2 i clk (h3 ▸ hi)⟩, ?_⟩
  rw [h2]
  cases hcap : d.capture with
  | none => rw [hcap] at c3; exact fun i clk hi => c3 i clk (h3 ▸ hi)
  | some c =>
      rw [hcap] at c3
      obtain ⟨evs', c4, c5⟩ := c3
      exact ⟨evs', c4, fun i clk hi => c5 i clk (h3 ▸ hi)⟩

theorem foldlM_ops_meta (w : World) : ∀ (ops : List DevOp) (d d' : Dev),
    ops.foldlM (fun d op => match op with
      | .launch l ids => d.runLaunch w.kernel l ids
      | .vendor c ins out => d.runVendor w.vendor c ins out) d = some d' →
    d'.race = d.race ∧ d'.capture = d.capture ∧ d'.events = d.events
  | [], d, d', h => by simp at h; subst h; exact ⟨rfl, rfl, rfl⟩
  | op :: ops, d, d', h => by
      simp only [List.foldlM_cons] at h
      obtain ⟨d1, h1, h2⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨e1, e2, e3⟩ := foldlM_ops_meta w ops d1 d' h2
      have : d1.race = d.race ∧ d1.capture = d.capture ∧ d1.events = d.events := by
        cases op with
        | launch l ids => exact runLaunch_meta h1
        | vendor c ins out => exact runVendor_meta h1
      exact ⟨e1.trans this.1, e2.trans this.2.1, e3.trans this.2.2⟩

/-- Recording an event on the device's tracker: an issue, then the event holds
    the party's new clock. -/
theorem track_record_outer {d : Dev} (ht : TrackOk d) (hcap : d.capture = none ∨ True) (p e : Nat)
    (d' : Dev) (h1 : d'.race = (d.race.issue p).1) (h2 : d'.capture = d.capture)
    (h3 : d'.events = d.events.set! e (some (some ((d.race.issue p).2, false)))) : TrackOk d' := by
  obtain ⟨⟨evs, c1, c2⟩, c3⟩ := ht
  have hc : CReached (d.race.issue p).1 evs := by
    have := CReached.op p [] [] c1 (Race.op_nil d.race p); exact this
  refine ⟨⟨((d.race.issue p).1.clock p) :: evs, by rw [h1]; exact .record p hc, ?_⟩, ?_⟩
  · refine evOk_set (EvOk.mono c2 (fun c h => List.mem_cons_of_mem _ h)) e _ ?_ h3
    intro clk hv
    simp only [Option.some.injEq, Prod.mk.injEq] at hv
    exact Or.inr (by rw [← hv.1, Race.issue_clock]; exact List.mem_cons_self ..)
  · rw [h2]
    have hv : ∀ (evs : List Clock) clk,
        (some (some ((d.race.issue p).2, false)) : Option (Option (Clock × Bool)))
          = some (some (clk, true)) → clk = #[] ∨ clk ∈ evs := fun _ _ h => by simp at h
    cases hcap' : d.capture with
    | none => rw [hcap'] at c3; exact evOk_set c3 e _ (hv _) h3
    | some c =>
        rw [hcap'] at c3
        obtain ⟨evs', c4, c5⟩ := c3
        exact ⟨evs', c4, evOk_set c5 e _ (hv _) h3⟩

/-- Recording an event inside a capture: an issue on the capture's tracker. -/
theorem track_record_inner {d : Dev} (ht : TrackOk d) {c : Capture} (hcap : d.capture = some c)
    (p e : Nat) (d' : Dev) (h1 : d'.race = d.race)
    (h2 : d'.capture = some { c with race := (c.race.issue p).1 })
    (h3 : d'.events = d.events.set! e (some (some ((c.race.issue p).2, true)))) : TrackOk d' := by
  obtain ⟨⟨evs, c1, c2⟩, c3⟩ := ht
  rw [hcap] at c3
  obtain ⟨evs', c4, c5⟩ := c3
  have hc : CReached (c.race.issue p).1 evs' := CReached.op p [] [] c4 (Race.op_nil c.race p)
  refine ⟨⟨evs, h1 ▸ c1, evOk_set c2 e _ (fun _ h => by simp at h) h3⟩, ?_⟩
  rw [h2]
  refine ⟨((c.race.issue p).1.clock p) :: evs', .record p hc, ?_⟩
  refine evOk_set (EvOk.mono c5 (fun c h => List.mem_cons_of_mem _ h)) e _ ?_ h3
  intro clk hv
  simp only [Option.some.injEq, Prod.mk.injEq] at hv
  exact Or.inr (by rw [← hv.1, Race.issue_clock]; exact List.mem_cons_self ..)

theorem event_clock {d : Dev} {evs : List Clock} {fl : Bool} (h : EvOk evs fl d) {eid : Int}
    {clk : Clock} (hev : d.event? eid = some (some (clk, fl))) : clk = #[] ∨ clk ∈ evs := by
  simp only [Dev.event?] at hev
  split at hev
  · cases hev
  · rw [Option.join_eq_some_iff] at hev
    exact h _ clk hev

theorem sgemvOn_track {w : World} {p : Nat} {trans m n alpha a x beta y : UInt64} {r : Option V}
    {d : Dev} (hk : sgemvOn w p trans m n alpha a x beta y = some (r, d)) (ht : TrackOk w.dev) :
    TrackOk d := by
  unfold sgemvOn at hk
  split at hk
  · split at hk
    · cases hk
    · dsimp only at hk
      obtain ⟨d', hop, hk⟩ := Option.bind_eq_some_iff.mp hk
      simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact devOp_track hop ht
  · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht

theorem sgemmOn_track {w : World} {p : Nat}
    {ta tb m n k alpha a sa b sb beta c sc batch oa ob oc la lb lc : UInt64} {r : Option V} {d : Dev}
    (hk : sgemmOn w p ta tb m n k alpha a sa b sb beta c sc batch oa ob oc la lb lc = some (r, d))
    (ht : TrackOk w.dev) : TrackOk d := by
  unfold sgemmOn at hk
  split at hk
  · split at hk
    · cases hk
    · dsimp only at hk
      obtain ⟨d', hop, hk⟩ := Option.bind_eq_some_iff.mp hk
      simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact devOp_track hop ht
  · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht

theorem devOps_fold_track {α : Type} (w : World) (p : Nat) (f : α → DevOp) (rs ws : α → List Nat) :
    ∀ (xs : List α) (d d' : Dev), xs.foldlM (fun d x => d.devOp w p (f x) (rs x) (ws x)) d = some d' →
      TrackOk d → TrackOk d'
  | [], d, d', h, ht => by simp at h; subst h; exact ht
  | x :: xs, d, d', h, ht => by
      simp only [List.foldlM_cons] at h
      obtain ⟨d1, h1, h2⟩ := Option.bind_eq_some_iff.mp h
      exact devOps_fold_track w p f rs ws xs d1 d' h2 (devOp_track h1 ht)

theorem asyncCopy_track {w : World} {p : Nat} {hostAddr : UInt64} {n : Nat} {wr : Bool}
    {rs ws : List Nat} {k : Race → Option (Dev × Mem)} {r : Option V} {w' : World}
    (h : asyncCopy w p hostAddr n wr rs ws k = some (r, w')) (ht : TrackOk w.dev)
    (hk : ∀ r' d m, w.dev.race.op p rs ws = some r' → k r' = some (d, m) →
      TrackOk { w.dev with race := r' } → TrackOk d) : TrackOk w'.dev := by
  unfold asyncCopy at h
  split at h
  · cases h
  · obtain ⟨r', hr, h⟩ := Option.bind_eq_some_iff.mp h
    obtain ⟨⟨d, m⟩, hkd, h⟩ := Option.bind_eq_some_iff.mp h
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨-, rfl⟩ := h
    refine hk r' d m hr hkd ?_
    obtain ⟨⟨evs, h1, h2⟩, h3⟩ := ht
    exact ⟨⟨evs, .op p rs ws h1 hr, h2⟩, h3⟩

/-- The entry points whose contracts never touch the device: libm, LMDB and
    threads. -/
def devUntouched : List IR.Ffi :=
  [.sinf, .cosf, .powf, .lmdbInit, .lmdbCleanup, .lmdbOpen, .lmdbBeginWriteTxn, .lmdbPut, .lmdbCommitWriteTxn, .lmdbCursorScan, .threadInit, .threadCleanup, .threadJoin]

/-- Follow a contract that never touches the device to each of its answers. -/
macro "dev_keep" h:ident : tactic => `(tactic| (
  try simp only [bind, Option.bind, cudaFail, lmdbI32] at $h:ident
  repeat' (first
    | (cases $h:ident; done)
    | (injection $h:ident with h1; injection h1 with h2 h3; subst h3; rfl)
    | split at $h:ident
    | (dsimp only at $h:ident))))

open Lean Elab Tactic Meta in
/-- Unfold the function the left side of `h` applies. -/
elab "unfold_head " h:ident : tactic => withMainContext do
  let d ← getLocalDeclFromUserName h.getId
  let T ← instantiateMVars d.type
  let some (_, lhs, _) := T.eq? | throwError "unfold_head: not an equation"
  let some c := lhs.getAppFn.constName? | throwError "unfold_head: no head constant"
  evalTactic (← `(tactic| unfold $(mkIdent c):ident at $h:ident))

set_option maxHeartbeats 4000000 in
/-- **A call whose contract never touches the device leaves it as it was.** -/
theorem callBits_dev_same (f : IR.Ffi) (hf : f ∈ devUntouched) {bits : List UInt64} {w : World}
    {r : Option V} {w' : World} (h : callBits f bits w = some (r, w')) : w'.dev = w.dev := by
  cases f <;> first
    | exact absurd hf (by decide)
    | (dsimp only [callBits] at h; unfold_head h; dev_keep h)

/-- Follow a device contract to each of its answers, leaving the steps it took
    among the hypotheses. -/
macro "dev_walk " h:ident : tactic => `(tactic| (
  repeat' (first
    | (cases $h:ident; done)
    | (simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at $h:ident
       obtain ⟨-, h3⟩ := $h:ident; subst h3)
    | (have h9 := Option.bind_eq_some_iff.mp $h:ident; clear $h:ident; obtain ⟨_, _, $h:ident⟩ := h9)
    | split at $h:ident
    | (dsimp only at $h:ident))))

/-- Close `TrackOk` at one answer of a contract, through the steps it took. -/
macro "track_close" : tactic => `(tactic| (
  repeat' (first
    | assumption
    | refine devOp_track (by assumption) ?_
    | refine syncWrite_track (by assumption) ?_
    | refine syncRead_track (by assumption) ?_
    | refine memInfoOf_track (by assumption) ?_
    | refine sgemvOn_track (by assumption) ?_
    | refine sgemmOn_track (by assumption) ?_)))

set_option maxHeartbeats 8000000 in
/-- **Every contract keeps the tracker consistent.** Whatever a call does to the
    device, the concrete race state it leaves is one the abstract tracker
    reaches, and every recorded event clock is one of its snapshots. -/
theorem callBits_track (f : IR.Ffi) {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (h : callBits f bits w = some (r, w')) (ht : TrackOk w.dev) : TrackOk w'.dev := by
  by_cases hu : f ∈ devUntouched
  · rw [callBits_dev_same f hu h]; exact ht
  cases f with
  | _ =>
    first
      | exact absurd (by decide) hu
      | (cases h; done)
      | (dsimp only [callBits] at h; unfold_head h
         first
           | (refine devOnly_track h ?_; intro r d hk; dev_walk hk <;> (track_close; done))
           | (dev_walk h <;> (track_close; done)))

end AlgorithmLib.Device
