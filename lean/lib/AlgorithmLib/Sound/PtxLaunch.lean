module
public import AlgorithmLib.Sound.PtxThread
public import AlgorithmLib.Sound.PtxCoalesce
public import AlgorithmLib.Kernel.Warp
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Sound.PtxLaunch` — a kernel's launch, as PTX, leaves what its body's does

A launch runs its blocks in order, and a block its threads in order, each with
the answers the laws give back from what the threads recorded
(`Kernel.BlockRuns`). The entry a kernel compiles to runs the same way, its
threads PTX's (`RunsBlock (PtxRun.runner e hw)`). The two record their collectives alike as the
laws read them (`calls_view`) and the same shared stores (`stores_view`), so
the laws give the same answers, and each thread ends as its body's does
(`thread_sound`). A block's PTX threads run at one fuel, which more fuel
keeps (`thread_mono`).

**`launch_sound_bits`**: a launch of a kernel on parameter bits cut to their
types runs as a launch of the entry it compiles to, leaving device memory the
same.
-/

namespace AlgorithmLib.Kernel.PtxSound

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem Ptx

/-- A collective as the laws read it: a ballot's arguments as whether each is
    not zero. -/
def callView (c : Intr × List V) : Intr × List V :=
  (c.1, if c.1 == .ballot then c.2.map ballotBit else c.2)

theorem calls_view_list : ∀ l : List Obs, (l.filterMap collOf).map callView = (l.filterMap viewObs).filterMap collOf
  | [] => rfl
  | o :: l => by
      have ih := calls_view_list l
      cases o with
      | call c args =>
          cases c with
          | intr i =>
              by_cases hc : i.isColl = true
              · simp [collOf, viewObs, hc, ih, callView]
              · simp [collOf, viewObs, hc, ih]
          | _ => simp [collOf, viewObs, ih]
      | _ => simp [collOf, viewObs, ih]

/-- **Related worlds record their collectives alike**, as the laws read them. -/
theorem calls_view (bs ps : Nat) (bw pw : World) (h : WRel bs ps bw pw) :
    (collCalls bw bs).map callView = (collCalls pw ps).map callView := by
  unfold collCalls
  rw [calls_view_list, calls_view_list, h.2.2.2]

theorem shStep_view : ∀ (l : List Obs) (st : Nat × List (Nat × UInt64 × Nat × UInt64)),
    l.foldl shStep st = (l.filterMap viewObs).foldl shStep st
  | [], _ => rfl
  | o :: l, st => by
      cases o with
      | call c args =>
          cases c with
          | intr i =>
              by_cases hc : i.isColl = true
              · cases i <;> simp_all [viewObs, shStep, Intr.isColl] <;>
                  exact shStep_view l _
              · cases i <;> simp_all [viewObs, shStep, Intr.isColl] <;>
                  exact shStep_view l _
          | _ => simp only [List.foldl_cons, List.filterMap_cons, viewObs]; rw [shStep_view l]; rfl
      | store a n v => simp only [List.foldl_cons, List.filterMap_cons, viewObs]; exact shStep_view l _

/-- **Related worlds made the same shared stores.** -/
theorem stores_view (bs ps : Nat) (bw pw : World) (h : WRel bs ps bw pw) :
    sharedStores bw bs = sharedStores pw ps := by
  unfold sharedStores
  rw [shStep_view, shStep_view (pw.obs.reverse.drop ps), h.2.2.2]

/-- What a block's threads recorded, as the laws read it. -/
def callsView (calls : Nat → List (Intr × List V)) : Nat → List (Intr × List V) :=
  fun t => (calls t).map callView

theorem laneArg_view (calls : Nat → List (Intr × List V)) (warp lane k : Nat) (c : Intr) :
    laneArg (callsView calls) warp lane k c =
      (laneArg calls warp lane k c).map (fun a => if c == .ballot then a.map ballotBit else a) := by
  unfold laneArg callsView
  simp only [List.getElem?_map]
  cases h : (calls (32 * warp + lane))[k]? with
  | none => rfl
  | some x =>
      obtain ⟨c', a⟩ := x
      simp only [Option.map_some, callView]
      by_cases hc : (c' == c) = true
      · have : c' = c := by revert hc; cases c' <;> cases c <;> decide
        subst this; simp [hc, Except.map]
      · simp [hc, Except.map]

theorem bit_view (v : V) : ((asBits (ballotBit v)).getD 0 != 0) = ((asBits v).getD 0 != 0) := by
  cases v with
  | sc t b => by_cases h : b = 0 <;> simp [ballotBit, asBits, h]
  | vec t xs => simp [ballotBit, asBits]

theorem answerStep_view (calls : Nat → List (Intr × List V)) (snaps : Array ByteArray) (smem0 : ByteArray)
    (warp lane : Nat) (st : Nat × Array Answer) (x : Intr × List V) (k : Nat) :
    answerStep (callsView calls) snaps smem0 warp lane st (callView x, k) =
      answerStep calls snaps smem0 warp lane st (x, k) := by
  obtain ⟨n, out⟩ := st
  obtain ⟨c, args⟩ := x
  cases c with
  | barrier => rfl
  | shflBfly =>
      simp only [callView, show (Intr.shflBfly == Intr.ballot) = false from rfl, Bool.false_eq_true, if_false]
      match args with
      | [a, .sc ty m] =>
          simp only [answerStep, laneArg_view, show (Intr.shflBfly == Intr.ballot) = false from rfl,
            Bool.false_eq_true, if_false]
          cases laneArg calls warp ((lane ^^^ m.toNat) % 32) k .shflBfly <;> rfl
      | [] => rfl
      | [_] => rfl
      | [_, .vec _ _] => rfl
      | _ :: b :: _ :: _ => cases b <;> rfl
  | shflIdx =>
      simp only [callView, show (Intr.shflIdx == Intr.ballot) = false from rfl, Bool.false_eq_true, if_false]
      match args with
      | [a, .sc ty m] =>
          simp only [answerStep, laneArg_view, show (Intr.shflIdx == Intr.ballot) = false from rfl,
            Bool.false_eq_true, if_false]
          cases laneArg calls warp (m.toNat % 32) k .shflIdx <;> rfl
      | [] => rfl
      | [_] => rfl
      | [_, .vec _ _] => rfl
      | _ :: b :: _ :: _ => cases b <;> rfl
  | ballot =>
      simp only [callView, show (Intr.ballot == Intr.ballot) = true from rfl, if_true]
      unfold answerStep
      simp only [laneArg_view, show (Intr.ballot == Intr.ballot) = true from rfl, if_true]
      congr 1
      apply congrArg (fun f => (List.range 32).foldlM f (0 : UInt32))
      funext acc l
      cases laneArg calls warp l k .ballot with
      | error e => rfl
      | ok a =>
          simp only [Except.map, bind, Except.bind]
          cases a with
          | nil => rfl
          | cons v rest => cases rest with
            | nil => simp [bit_view]
            | cons _ _ => rfl
  | ex2Approx => rfl
  | rcpApprox => rfl
  | lg2Approx => rfl
  | sinApprox => rfl
  | cosApprox => rfl
  | sqrtApprox => rfl
  | rsqrtApprox => rfl
  | dp4a => rfl

/-- **The laws read the records only as their view.** -/
theorem answers_view (threads : Nat) (smem0 : ByteArray) (calls : Nat → List (Intr × List V))
    (stores : Nat → List (Nat × UInt64 × Nat × UInt64)) :
    answers threads smem0 (callsView calls) stores = answers threads smem0 calls stores := by
  have hsn : snapshots threads smem0 (callsView calls) stores = snapshots threads smem0 calls stores := by
    unfold snapshots callsView
    have : ∀ t, (((calls t).map callView).filter (·.1 == .barrier)).length = ((calls t).filter (·.1 == .barrier)).length := by
      intro t; rw [List.filter_map, List.length_map]; rfl
    simp only [this]
  unfold answers
  rw [hsn]
  congr 1
  apply congrArg (fun f => (List.range threads).mapM f)
  funext t
  unfold answersOf
  congr 1
  show List.foldlM _ _ ((callsView calls t).zipIdx) = _
  simp only [callsView]
  rw [show ((calls t).map callView).zipIdx = ((calls t).zipIdx).map (fun p => (callView p.1, p.2)) by
    apply List.ext_getElem (by simp); intro i h1 h2; simp [List.getElem_zipIdx]]
  rw [List.foldlM_map]
  congr 1
  funext st p
  exact answerStep_view calls _ smem0 (t / 32) (t % 32) st p.1 p.2

theorem norm32_u32 (x : UInt32) : x.toUInt64 &&& widthMask .i32 = x.toUInt64 := by
  apply UInt64.toNat_inj.mp
  rw [toNat_wm .i32 (by simp)]
  exact Nat.mod_eq_of_lt (by simpa using x.toNat_lt)

theorem lane32_ok (a : List V) : ∃ z, lane32 a = .sc .i32 z ∧ z &&& widthMask .i32 = z := by
  unfold lane32
  split
  · exact ⟨_, rfl, norm32_u32 _⟩
  · exact ⟨0, rfl, rfl⟩

/-- A kernel the compiler takes asks for less shared memory than a GPU addresses. -/
theorem entry_smem (k : Kernel) (e : PtxIr.Entry) (he : k.entry = .ok e) : k.smem < 2 ^ 32 := by
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, h⟩ := entry_facts k e he
  exact h

/-- **The laws answer values a thread can take.** -/
theorem answersOf_ok (calls : Nat → List (Intr × List V)) (snaps : Array ByteArray) (smem0 : ByteArray) (t : Nat)
    (hs0 : smem0.size < 2 ^ 32) (hsn : ∀ B ∈ snaps, B.size = smem0.size)
    (as : List Answer) (h : answersOf calls snaps smem0 t = .ok as) : AnsOk as := by
  refine ⟨?_, fun k b hk => by rw [Kernel.answersOf_smem_size hsn h k b hk]; exact hs0⟩
  unfold answersOf at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  rename_i r hr
  obtain ⟨_, out⟩ := r
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  obtain ⟨hsz, _, h3⟩ := foldlM_zipIdx_push _
    (fun _ (_ : Intr × List V) y => ∀ v, y = Answer.val v → ∃ z, v = .sc .i32 z ∧ z &&& widthMask .i32 = z)
    (fun s out x k r hr => by
      obtain ⟨c, args⟩ := x
      unfold answerStep at hr
      dsimp only at hr
      split at hr
      · simp only [pure, Except.pure, Except.ok.injEq] at hr; subst hr
        exact ⟨_, rfl, fun v hv => by cases hv⟩
      · simp only [bind, Except.bind] at hr
        split at hr
        · cases hr
        simp only [pure, Except.pure, Except.ok.injEq] at hr; subst hr
        exact ⟨_, rfl, fun v hv => by cases hv; exact lane32_ok _⟩
      · simp only [bind, Except.bind] at hr
        split at hr
        · cases hr
        simp only [pure, Except.pure, Except.ok.injEq] at hr; subst hr
        exact ⟨_, rfl, fun v hv => by cases hv; exact lane32_ok _⟩
      · simp only [bind, Except.bind] at hr
        split at hr
        · cases hr
        simp only [pure, Except.pure, Except.ok.injEq] at hr; subst hr
        exact ⟨_, rfl, fun v hv => by cases hv; exact ⟨_, rfl, norm32_u32 _⟩⟩
      · cases hr)
    (calls t) 0 0 #[] _ hr
  intro k v hk
  simp only [Array.size_empty, Nat.zero_add] at h3
  have hk' : k < (calls t).length := by
    have := (List.getElem?_eq_some_iff.mp hk).1
    simp at hsz this; omega
  obtain ⟨x, hx⟩ : ∃ x, (calls t)[k]? = some x := ⟨_, List.getElem?_eq_getElem hk'⟩
  obtain ⟨y, hy, hP⟩ := h3 k x hx
  have hk2 : out[k]? = some (Answer.val v) := by simpa using hk
  have : y = Answer.val v := by
    rw [hk2] at hy; exact (Option.some.inj hy).symm
  exact hP v this

theorem mapM_ok_mem {α β : Type} (f : α → Except String β) :
    ∀ (l : List α) (r : List β), l.mapM f = .ok r → ∀ y ∈ r, ∃ x, f x = .ok y
  | [], r, h => by simp [pure, Except.pure] at h; subst h; simp
  | a :: l, r, h => by
      simp only [List.mapM_cons, bind, Except.bind] at h
      split at h
      · cases h
      rename_i b hb
      split at h
      · cases h
      rename_i rs hrs
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      intro y hy
      rcases List.mem_cons.mp hy with rfl | hy
      · exact ⟨a, hb⟩
      · exact mapM_ok_mem f l rs hrs y hy

theorem answers_ok (threads : Nat) (smem0 : ByteArray) (calls : Nat → List (Intr × List V))
    (stores : Nat → List (Nat × UInt64 × Nat × UInt64)) (ans : Nat → List Answer)
    (hs0 : smem0.size < 2 ^ 32) (h : answers threads smem0 calls stores = .ok ans) (t : Nat) :
    AnsOk (ans t) := by
  unfold answers at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  rename_i per hper
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  cases hg : per[t]? with
  | none => simp [List.getD_eq_getElem?_getD, hg]; exact ⟨fun k v hk => by simp at hk, fun k b hk => by simp at hk⟩
  | some as =>
      simp only [List.getD_eq_getElem?_getD, hg, Option.getD_some]
      have hm := List.mem_of_getElem? hg
      obtain ⟨t', ht'⟩ := mapM_ok_mem _ _ _ hper as hm
      exact answersOf_ok calls _ smem0 t' hs0
        (Kernel.snapshots_size _ _ _ _) as ht'

open PtxRun in
/-- **More fuel ends a run as less did.** -/
theorem go_succ (c : PtxRun.Ctx) : ∀ (fuel pc : Nat) (ts : PtxRun.TS) (r : Unit × PtxRun.TS),
    (go c fuel pc).run ts = .ok r → (go c (fuel + 1) pc).run ts = .ok r
  | 0, _, _, _, h => by simp [go, StateT.run, throw, throwThe, MonadExceptOf.throw, StateT.lift] at h; cases h
  | fuel + 1, pc, ts, r, h => by
      have ih := go_succ c fuel
      unfold go at h ⊢
      cases hb : c.body[pc]? with
      | none => simp only [hb] at h ⊢; exact h
      | some l =>
          simp only [hb] at h ⊢
          cases l with
          | label _ => exact ih _ _ _ h
          | ins g i =>
              simp only [StateT.run, bind, StateT.bind] at h ⊢
              cases hg : guardOk c g ts with
              | error e => rw [hg] at h; cases h
              | ok gr =>
                  simp only [hg] at h ⊢
                  obtain ⟨b, ts1⟩ := gr
                  simp only [Except.bind] at h ⊢
                  cases b with
                  | false => exact ih _ _ _ h
                  | true =>
                      simp only [if_true, StateT.bind] at h ⊢
                      cases he : exec c i ts1 with
                      | error e => rw [he] at h; cases h
                      | ok er =>
                          obtain ⟨n, ts2⟩ := er
                          simp only [he] at h ⊢
                          cases n with
                          | step => exact ih _ _ _ h
                          | jump j => exact ih _ _ _ h
                          | done => exact h

open PtxRun in
theorem go_mono (c : PtxRun.Ctx) (fuel pc : Nat) (ts : PtxRun.TS) (r : Unit × PtxRun.TS)
    (h : (go c fuel pc).run ts = .ok r) : ∀ m, (go c (fuel + m) pc).run ts = .ok r
  | 0 => h
  | m + 1 => go_succ c _ _ _ _ (go_mono c fuel pc ts r h m)

/-- **A thread that ends with some fuel ends as it did with any more.** -/
theorem thread_mono (e : PtxIr.Entry) (ans : List Answer) (args : List V) (w : World) (hw : Hw) (prov : Bool)
    (F F' : Nat) (u : World) (hF : F ≤ F') (h : PtxRun.thread e ans args w hw F prov = .ok u) :
    PtxRun.thread e ans args w hw F' prov = .ok u := by
  unfold PtxRun.thread at h ⊢
  simp only [bind, Except.bind] at h ⊢
  split at h
  · cases h
  rename_i r hr
  obtain ⟨m, rfl⟩ := Nat.exists_eq_add_of_le hF
  rw [go_mono _ _ _ _ _ hr m]
  exact h

/-! ## A launch of the PTX

A launch of the PTX is `Kernel.Run`'s launch with each thread `PtxRun.thread`
(`RunsLaunch (PtxRun.runner e hw)`), as the Code's is with each thread its
body (`LaunchRuns`). -/

/-- Two worlds the same apart from their observations. -/
def Same (a b : World) : Prop := { a with obs := [] } = { b with obs := [] }

theorem same_mem {a b : World} (h : Same a b) : a.mem = b.mem := by
  unfold Same at h; have h' := congrArg World.mem h; exact h'

theorem same_arena {a b : World} (h : Same a b) (A : ByteArray) :
    Same { a with mem := { a.mem with arena := A } } { b with mem := { b.mem with arena := A } } := by
  unfold Same at h ⊢
  exact congrArg (fun x : World => { x with mem := { x.mem with arena := A } }) h

/-- Arguments a thread of `k` can be run on: its bits, each cut to its type's
    width, the shared base at its place. -/
def ArgsOk (k : Kernel) (args : List V) : Prop :=
  ∃ xs : List UInt64, args = threadArgsOf k xs ∧ xs.length = 4 + k.params.length ∧
    (∀ i (hi : i < (idParams ++ k.params).length) (hx : i < xs.length),
      xs[i] &&& widthMask (idParams ++ k.params)[i] = xs[i]) ∧ xs[3]? = some (addrOf .arena 0) ∧
    (k.warp = true → xs.getD 2 0 ≤ 32)

/-- **A pass of the Code's threads is one of the PTX's**: from worlds the same
    apart from their observations and calls alike as the laws read them, the
    PTX's pass ends, at every fuel from some on, with the same stores, its
    world the same apart from its observations and its calls alike. -/
theorem pass_sim (k : Kernel) (e : PtxIr.Entry) (he : k.entry = .ok e) (hw : Hw) (hhw : HwOk hw)
    (smem0 : ByteArray) (hsm : smem0.size < 2 ^ 32) (args : Nat → List V)
    (hargs : ∀ t, ArgsOk k (args t)) (ans : Nat → List Answer) (steps : Nat) :
    ∀ (l : List Nat) (bw pw bw' : World) (bc pc bc' : Nat → List (Intr × List V))
      (st st' : Nat → List (Nat × UInt64 × Nat × UInt64)),
      (∀ t ∈ l, AnsOk (ans t)) → Same bw pw → callsView bc = callsView pc → pw.mem.nan = hw.nan →
      l.foldlM (passStep (k.run hw steps) smem0 args ans) (bw, bc, st) = .ok (bw', bc', st') →
      ∃ F, ∀ F', F ≤ F' → ∃ pw' pc',
        l.foldlM (passStep (PtxRun.runner e hw F') smem0 args ans) (pw, pc, st) = .ok (pw', pc', st') ∧
        Same bw' pw' ∧ callsView bc' = callsView pc' ∧ pw'.mem.nan = hw.nan
  | [], bw, pw, bw', bc, pc, bc', st, st', _, hs, hc, hpn, h => by
      simp only [List.foldlM_nil, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl⟩ := h
      exact ⟨0, fun _ _ => ⟨pw, pc, rfl, hs, hc, hpn⟩⟩
  | t :: l, bw, pw, bw', bc, pc, bc', st, st', hans, hs, hc, hpn, h => by
      simp only [List.foldlM_cons, bind, Except.bind] at h
      split at h
      · cases h
      rename_i r hr
      simp only [passStep, bind, Except.bind] at hr
      split at hr
      · cases hr
      rename_i u hu
      simp only [pure, Except.pure, Except.ok.injEq] at hr
      subst hr
      obtain ⟨xs, hxs, hlen, hnorm, h3, hwarp⟩ := hargs t
      rw [hxs] at hu
      have hs0 := same_arena hs smem0
      obtain ⟨F1, u', hu', hrel⟩ := thread_sound k e he (ans t) xs _ u _ hw false hs0 hlen hnorm h3
        (hans t (List.mem_cons_self ..)) hhw (by exact hpn) (by exact hsm) hwarp steps hu
      have hcv : callsView (fun t' => if t' = t then collCalls u bw.obs.length else bc t') =
          callsView (fun t' => if t' = t then collCalls u' pw.obs.length else pc t') := by
        funext t'
        unfold callsView
        by_cases ht : t' = t
        · simp only [ht, if_true]; exact calls_view _ _ _ _ hrel
        · simp only [ht, if_false]; exact congrFun hc t'
      have hst : sharedStores u bw.obs.length = sharedStores u' pw.obs.length := stores_view _ _ _ _ hrel
      obtain ⟨F2, hF2⟩ := pass_sim k e he hw hhw smem0 hsm args hargs ans steps l _ _ bw' _ _ bc' _ st'
        (fun t' ht' => hans t' (List.mem_cons_of_mem _ ht')) hrel.1 hcv
        (by rw [thread_nan _ _ _ _ _ _ _ _ hu']; exact hpn) h
      refine ⟨max F1 F2, fun F' hF' => ?_⟩
      obtain ⟨pw', pc', hp, hs', hc', hn'⟩ := hF2 F' (Nat.le_trans (Nat.le_max_right _ _) hF')
      refine ⟨pw', pc', ?_, hs', hc', hn'⟩
      have hth : PtxRun.runner e hw F' false (ans t) (args t) { pw with mem := { pw.mem with arena := smem0 } } = .ok u' := by
        rw [hxs]
        exact thread_mono e _ _ _ hw false F1 F' u' (Nat.le_trans (Nat.le_max_left _ _) hF') hu'
      simp only [List.foldlM_cons, bind, Except.bind, passStep, hth, pure, Except.pure]
      rw [← hst]
      exact hp

/-- **A block of the Code is one of the PTX**: from worlds the same apart from
    their observations, the PTX's block ends in one the same as the Code's
    apart from its observations. -/
theorem block_sound (k : Kernel) (e : PtxIr.Entry) (he : k.entry = .ok e) (hw : Hw) (hhw : HwOk hw)
    (b bd : Nat) (params : List V) (hargs : ∀ t, ArgsOk k (threadArgs b bd params t))
    (w u pw : World) (hs : Same w pw) (hpn : pw.mem.nan = hw.nan) (h : BlockRuns k b bd params hw w u) :
    ∃ u', RunsBlock (PtxRun.runner e hw) k.smem b bd params pw u' ∧ Same u u' ∧ u'.mem.nan = hw.nan := by
  obtain ⟨steps, ans, w', calls, stores, ans', hf, ha, hset, rfl⟩ := h
  have hsm : (ByteArray.mk (Array.replicate k.smem 0)).size < 2 ^ 32 := by
    simpa [ByteArray.size] using entry_smem k e he
  have hans : ∀ t ∈ List.range bd, AnsOk (ans t) := fun t ht => by
    rw [← hset t (List.mem_range.mp ht)]; exact answers_ok _ _ _ _ _ hsm ha t
  have hs0 := same_arena hs (ByteArray.mk (Array.replicate k.smem 0))
  obtain ⟨F, hF⟩ := pass_sim k e he hw hhw _ hsm _ hargs ans steps (List.range bd) _ _ w' _ _ calls _ stores
    hans hs0 rfl hpn hf
  obtain ⟨pw', pc', hp, hs', hc', hn'⟩ := hF F (Nat.le_refl _)
  refine ⟨{ pw' with mem := { pw'.mem with arena := pw.mem.arena } }, ⟨F, ans, pw', pc', stores, ans', hp, ?_, hset, rfl⟩, ?_, hn'⟩
  · rw [← answers_view, ← hc', answers_view]; exact ha
  · rw [← same_mem hs]; exact same_arena hs' _

/-- The first `c` blocks of the Code run as those of the PTX. -/
theorem blocks_sound (k : Kernel) (e : PtxIr.Entry) (he : k.entry = .ok e) (hw : Hw) (hhw : HwOk hw)
    (bd : Nat) (params : List V)
    (hargs : ∀ b t, ArgsOk k (threadArgs b bd params t)) :
    ∀ (c : Nat) (w u pw : World), Same w pw → pw.mem.nan = hw.nan → BlocksRun k bd params hw c w u →
      ∃ u', RunsBlocks (PtxRun.runner e hw) k.smem bd params c pw u' ∧ Same u u' ∧ u'.mem.nan = hw.nan
  | 0, w, u, pw, hs, hpn, h => ⟨pw, rfl, by cases h; exact hs, hpn⟩
  | c + 1, w, u, pw, hs, hpn, h => by
      obtain ⟨v, hv, hb⟩ := h
      obtain ⟨v', hv', hsv, hvn⟩ := blocks_sound k e he hw hhw bd params hargs c w v pw hs hpn hv
      obtain ⟨u', hu', hsu, hun⟩ := block_sound k e he hw hhw c bd params (hargs c) v u v' hsv hvn hb
      exact ⟨u', ⟨v', hv', hu'⟩, hsu, hun⟩

/-- **A launch of the Code is one of the PTX it compiles to**, leaving device
    memory the same. -/
theorem launch_sound (k : Kernel) (e : PtxIr.Entry) (he : k.entry = .ok e) (hw : Hw) (hhw : HwOk hw)
    (grid bd : Nat) (params : List V)
    (hargs : ∀ b t, ArgsOk k (threadArgs b bd params t)) (bufs bufs' : Nat → ByteArray)
    (h : LaunchRuns k grid bd params bufs bufs' hw) :
    RunsLaunch (PtxRun.runner e hw) k.smem grid bd params bufs bufs' hw.nan := by
  obtain ⟨u, hu, rfl⟩ := h
  obtain ⟨u', hu', hs, -⟩ := blocks_sound k e he hw hhw bd params hargs grid _ u _ rfl rfl hu
  exact ⟨u', hu', by rw [same_mem hs]⟩

/-- A launch's parameters: bits `ps`, each of the kernel's parameter type at
    its place. -/
def paramsOf (k : Kernel) (ps : List UInt64) : List V :=
  (k.params.zip ps).map fun (t, x) => .sc t x

/-- **A thread of a launch is run on arguments it can be run on** when the
    launch's parameter bits are cut to their types' widths. -/
theorem threadArgs_ok (k : Kernel) (ps : List UInt64) (hps : ps.length = k.params.length)
    (hn : ∀ i (h1 : i < k.params.length) (h2 : i < ps.length), ps[i] &&& widthMask k.params[i] = ps[i])
    (b bd t : Nat) (hbd : k.warp = true → bd ≤ 32) : ArgsOk k (threadArgs b bd (paramsOf k ps) t) := by
  refine ⟨[(UInt32.ofNat b).toUInt64, (UInt32.ofNat t).toUInt64, (UInt32.ofNat bd).toUInt64, addrOf .arena 0] ++ ps,
    ?_, by simp [hps]; omega, ?_, rfl, ?_⟩
  · simp [threadArgs, threadArgsOf, paramsOf, idParams]
  · intro i hi hx
    rcases i with _ | _ | _ | _ | i
    · simp [idParams, norm32_u32]
    · simp [idParams, norm32_u32]
    · simp [idParams, norm32_u32]
    · simp [idParams]; decide
    · simp only [idParams, List.cons_append, List.nil_append, List.getElem_cons_succ]
      exact hn i (by simp [idParams] at hi; omega) (by simp at hx; omega)
  · intro hw
    have := hbd hw
    simp only [List.cons_append, List.getD_cons_succ, List.getD_cons_zero]
    rw [UInt64.le_iff_toNat_le, UInt32.toNat_toUInt64, UInt32.toNat_ofNat_of_lt' (show bd < 2 ^ 32 by omega)]
    exact this

/-- **A launch of the Code on parameter bits cut to their types is one of the
    PTX it compiles to**, leaving device memory the same, in blocks of at most
    32 threads where each is one warp. -/
theorem launch_sound_bits (k : Kernel) (e : PtxIr.Entry) (he : k.entry = .ok e) (hw : Hw) (hhw : HwOk hw)
    (grid bd : Nat) (ps : List UInt64) (hps : ps.length = k.params.length)
    (hn : ∀ i (h1 : i < k.params.length) (h2 : i < ps.length), ps[i] &&& widthMask k.params[i] = ps[i])
    (hbd : k.warp = true → bd ≤ 32)
    (bufs bufs' : Nat → ByteArray) (h : LaunchRuns k grid bd (paramsOf k ps) bufs bufs' hw) :
    RunsLaunch (PtxRun.runner e hw) k.smem grid bd (paramsOf k ps) bufs bufs' hw.nan :=
  launch_sound k e he hw hhw grid bd _ (fun b t => threadArgs_ok k ps hps hn b bd t hbd) bufs bufs' h

/-! ## The entry that ships -/

/-- A kernel's shipped entry is its compiled entry coalesced. -/
theorem shipped_ok (k : Kernel) (e : PtxIr.Entry) (he : k.entry = .ok e) : k.shipped = .ok (Coalesce.run e) := by
  simp [Kernel.shipped, he, Except.map]

/-- **The shipped entry runs as the compiled one does**, on a GPU: coalescing
    its registers changes no thread's end (`PtxCoalesce.thread_run`). -/
theorem runner_shipped (e : PtxIr.Entry) (hw : Hw) (hhw : HwOk hw) :
    PtxRun.runner (Coalesce.run e) hw = PtxRun.runner e hw := by
  funext steps
  exact PtxCoalesce.runner_run e hw hhw.approx steps

end AlgorithmLib.Kernel.PtxSound
