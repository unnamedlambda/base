module
public import AlgorithmLib.Check.WgpuRule
meta import AlgorithmLib.Check.WgpuRule
public import AlgorithmLib.Check.CudaSound
meta import AlgorithmLib.Check.CudaSound
public import AlgorithmLib.Host.WgpuLaws
meta import AlgorithmLib.Host.WgpuLaws
import all Init.Data.Repr
@[expose] public section

/-!
# `Check.WgpuSound` — the wgpu rows are sound

`wgRow_sound`: a call a wgpu row accepts decodes, and the world it leaves
satisfies the typestate the row gives, with the answer the row describes; the
rest of the world is kept but for the values the call may end
(`CuKeptBut`). Each row is a step of `WgStep` (`Host.WgpuLaws`), and what it
keeps of the typestate is what `wgKeeps` keeps of that step
(`wgKeeps_sound`).
-/

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace AlgorithmLib.Check

open AlgorithmLib.HProg.Contracts (Fact TState holds_of_mem Held)

/-- What a call keeps of the rest of the world, but the handles of kinds `ks`. -/
structure CuKeptBut (ks : List Held) (w w' : World) : Prop where
  sizes : ∀ r, r ≠ .pinned → w'.mem.sizes r = w.mem.sizes r
  held : ∀ (k : Held) x, k ∉ ks → k.live w x = true → k.live w' x = true
  blk : ∀ x n, Contracts.Blk w x n → Contracts.Blk w' x n
  obs : w'.obs = w.obs

theorem CuKept.but {w w' : World} (h : CuKept w w') (ks : List Held) : CuKeptBut ks w w' :=
  ⟨h.sizes, fun k x _ hl => h.held k x hl, h.blk, h.obs⟩

-- ---------------------------------------------------------------------------
-- Reading the typestate
-- ---------------------------------------------------------------------------

theorem nwgOf_holds {S : TState} {w : World} {n : Nat} (hS : S.holds w) (h : nwgOf S = some n) :
    w.wg.objs.size = n := by
  unfold nwgOf at h
  obtain ⟨x, hx, hk⟩ := List.exists_of_findSome?_eq_some h
  cases x <;> simp only [reduceCtorEq] at hk
  cases hk
  exact hS _ hx

theorem wgoOf_holds {S : TState} {w : World} {i : Nat} {s : WgShape} (hS : S.holds w) (hl : w.wg.lost = false)
    (h : wgoOf S i = some s) : ∃ d, w.wg.objs[i]? = some (true, d) ∧ w.wg.Fits d s := by
  unfold wgoOf at h
  obtain ⟨x, hx, hk⟩ := List.exists_of_findSome?_eq_some h
  cases x <;> simp only [reduceCtorEq] at hk
  rename_i i' s'
  split at hk
  · rename_i e; simp only [beq_iff_eq] at e; subst e; cases hk
    rcases hS _ hx with h | h
    · rw [hl] at h; cases h
    · exact h
  · cases hk

theorem wg1Of_mem {S : TState} {k : WgObj} {o : Option WgShape} (h : wg1Of S k = some o) : Fact.wg1 k o ∈ S := by
  unfold wg1Of at h
  obtain ⟨x, hx, hk⟩ := List.exists_of_findSome?_eq_some h
  cases x <;> simp only [reduceCtorEq] at hk
  rename_i k' o'
  split at hk
  · rename_i e; simp only [decide_eq_true_eq] at e; subst e; cases hk; exact hx
  · cases hk

theorem wg1Of_none {S : TState} {w : World} {k : WgObj} (hS : S.holds w) (hl : w.wg.lost = false)
    (h : wg1Of S k = some none) : w.wg.NoneOf k :=
  (hS _ (wg1Of_mem h)).resolve_left (by rw [hl]; nofun)

theorem wg1Of_some {S : TState} {w : World} {k : WgObj} {s : WgShape} (hS : S.holds w) (hl : w.wg.lost = false)
    (h : wg1Of S k = some (some s)) {i : Nat} {d : WgData} (hi : w.wg.objs[i]? = some (true, d)) (hk : d.kind = k) :
    w.wg.isOne k i = true ∧ w.wg.Fits d s :=
  (hS _ (wg1Of_mem h)).resolve_left (by rw [hl]; nofun) i d hi hk

/-- An object made once, named by its handle: wgpu finds it. -/
theorem wgoAt_holds {S : TState} {w : World} {h : UInt64} {k : WgObj} {i : Nat} {s : WgShape} (hS : S.holds w)
    (hl : w.wg.lost = false) (e : wgoAt S h k = some (i, s)) :
    ∃ d, w.wg.get? k h = some (i, d) ∧ w.wg.objs[i]? = some (true, d) ∧ w.wg.Fits d s ∧ s.kind = k := by
  unfold wgoAt at e
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at e
  obtain ⟨i', hi, s', hs, e⟩ := e
  split at e
  · rename_i hk
    cases e
    simp only [decide_eq_true_eq] at hk
    obtain ⟨d, hd, hf⟩ := wgoOf_holds hS hl hs
    refine ⟨d, ?_, hd, hf, hk⟩
    have hkd : d.kind = k := hf.kind.trans hk
    simp [WgState.get?, hi, hd, hkd]
  · cases e

theorem wgoArg_holds {S : TState} {w : World} {Γ : Env} {a : AV} {x : UInt64} {k : WgObj} {i : Nat}
    {s : WgShape} (hS : S.holds w) (hl : w.wg.lost = false) (hd : ∃ t, a.Desc Γ w (.sc t x))
    (e : wgoArg S a k = some (i, s)) :
    ∃ d, w.wg.get? k x = some (i, d) ∧ w.wg.objs[i]? = some (true, d) ∧ w.wg.Fits d s ∧ s.kind = k := by
  unfold wgoArg at e
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at e
  obtain ⟨h, hb, e⟩ := e
  obtain rfl := known_bits hd hb
  exact wgoAt_holds hS hl e

/-- The one object of kind `k`, named by a value the checker describes as that
    one: wgpu finds it. -/
theorem wg1Arg_holds {S : TState} {w : World} {Γ : Env} {a : AV} {x : UInt64} {k : WgObj} {s : WgShape}
    (hS : S.holds w) (hl : w.wg.lost = false) (hd : ∃ t, a.Desc Γ w (.sc t x)) (e : wg1Arg S a k = some s) :
    ∃ i d, w.wg.get? k x = some (i, d) ∧ wgIndex x = some i ∧ w.wg.objs[i]? = some (true, d) ∧
      d.kind = k ∧ w.wg.isOne k i = true ∧ w.wg.Fits d s := by
  unfold wg1Arg at e
  split at e
  · rename_i k' s' hw
    split at e
    · rename_i hk
      simp only [decide_eq_true_eq] at hk
      subst hk; cases e
      obtain ⟨t, hd⟩ := hd
      obtain ⟨x', e, hl'⟩ := hd
      simp only [V.sc.injEq] at e
      obtain ⟨-, rfl⟩ := e
      rcases hl' with ⟨h, -⟩ | hl'
      · cases h
      · simp only [Held.live, hl, Bool.false_or] at hl'
        split at hl'
        · rename_i i hi
          obtain ⟨⟨d, hdi, hkd⟩, -⟩ := WgState.isOne_iff.mp hl'
          obtain ⟨-, hf⟩ := wg1Of_some hS hl hw hdi hkd
          refine ⟨i, d, ?_, hi, hdi, hkd, hl', hf⟩
          simp [WgState.get?, hi, hdi, hkd]
        · cases hl'
    · cases e
  · cases e

theorem wgDev_holds {S : TState} {w : World} {Γ : Env} {a : AV} {x : UInt64} {i : Nat} {sc : List Bool}
    (hS : S.holds w) (hl : w.wg.lost = false) (hd : ∃ t, a.Desc Γ w (.sc t x)) (e : wgDev S a = some (i, sc)) :
    w.wg.device? x = some (i, sc) ∧ w.wg.objs[i]? = some (true, .device sc) := by
  unfold wgDev at e
  split at e
  · rename_i i' sc' h
    cases e
    obtain ⟨d, hg, hdi, hf, -⟩ := wgoArg_holds hS hl hd h
    have : d = .device sc := by cases d <;> first | exact hf.elim | (cases hf; rfl)
    subst this
    exact ⟨by simp [WgState.device?, hg], hdi⟩
  · cases e

/-- The text the typestate says is at `a`, by `n` bytes. -/
theorem wgText_of {S : TState} {w : World} {a : UInt64} {n : Nat} {sh : Bool} (hS : S.holds w)
    (h : Fact.wgText a n sh ∈ S) :
    ∃ t, wgText w.mem a (UInt64.ofNat n) = some t ∧ (sh = true → w.wgslOk t = true) :=
  (hS _ h).2.2

theorem wgTextAt_holds {S : TState} {w : World} {a : UInt64} {n : Nat} (hS : S.holds w)
    (h : wgTextAt S a n = true) : ∃ t, wgText w.mem a (UInt64.ofNat n) = some t := by
  unfold wgTextAt at h
  obtain ⟨x, hx, hk⟩ := List.any_eq_true.mp h
  cases x <;> simp only [Bool.false_eq_true] at hk
  rename_i a' n' sh
  simp only [Bool.and_eq_true, beq_iff_eq] at hk
  obtain ⟨rfl, rfl⟩ := hk
  obtain ⟨t, ht, -⟩ := wgText_of hS hx
  exact ⟨t, ht⟩

-- ---------------------------------------------------------------------------
-- What a step keeps
-- ---------------------------------------------------------------------------

/-- A fact that reads nothing of wgpu holds where only wgpu's objects
    changed. -/
theorem holds_wg {x : Fact} {w : World} {s : WgState} (h : x.holds w)
    (hx : ∀ k a, x ≠ .held (.wg k) a ∧ x ≠ .opened (.wg k) a) (hw : x.isWg = false) :
    x.holds { w with wg := s } := by
  cases x with
  | part p b => cases p <;> exact h
  | held k a =>
      cases k with
      | wg k => exact absurd rfl (hx k a).1
      | _ => exact h
  | opened k a =>
      cases k with
      | wg k => exact absurd rfl (hx k a).2
      | _ => exact h
  | htAt _ _ _ => exact HtSpec.HtObj.of_mem h rfl rfl
  | nwg _ | wgo _ _ | wg1 _ _ => simp [Fact.isWg] at hw
  | _ => exact h

/-- **What `wgKeeps` keeps survives the step.** -/
theorem wgKeeps_sound {Ti : List Nat} {TK K : List WgObj} {g : Option Nat} {n : Nat} {w : World} {s' : WgState}
    {x : Fact} (hst : WgStep w.wg s' Ti TK K n) (hg : ∀ m, g = some m → n = m)
    (hx : x.holds w) (hk : wgKeeps Ti TK K g x = true) : x.holds { w with wg := s' } := by
  have notIn : ∀ {k : WgObj} {L : List WgObj}, L.any (fun k' => decide (k' = k)) = false → k ∉ L := by
    intro k L h hm
    have := List.any_eq_false.mp h k hm
    simp at this
  have readsOk : ∀ {sh : WgShape}, sh.reads.all (fun r => !K.any (fun k => decide (k = r))) = true →
      ∀ k ∈ sh.reads, KindSame w.wg s' k := by
    intro sh h k hk
    have := List.all_eq_true.mp h k hk
    simp only [Bool.not_eq_true'] at this
    exact hst.kinds k (notIn this)
  cases x with
  | nwg m =>
      simp only [wgKeeps, beq_iff_eq] at hk
      have := hg 0 hk
      show s'.objs.size = m
      rw [hst.size, this]; exact hx
  | wgo i sh =>
      simp only [wgKeeps, Bool.and_eq_true, Bool.not_eq_true'] at hk
      obtain ⟨⟨hi, hT⟩, hr⟩ := hk
      rcases hx with hlost | ⟨d, hd, hf⟩
      · exact .inl (hst.lost.trans hlost)
      refine .inr ?_
      have hkd := hf.kind
      have hi' : i ∉ Ti := fun hm => by rw [List.contains_iff_mem.mpr hm] at hi; cases hi
      obtain ⟨d', hd', hs⟩ := hst.same i true d hd hi' (by rw [hkd]; exact notIn hT)
      exact ⟨d', hd', (WgState.Fits.congr (readsOk hr)).mp ((hs.2 _ _).mp hf)⟩
  | wg1 k o =>
      cases o with
      | none =>
          simp only [wgKeeps, Bool.not_eq_true'] at hk
          rcases hx with hlost | hx
          · exact .inl (hst.lost.trans hlost)
          exact .inr (((hst.kinds k (notIn hk)).noneOf).mp hx)
      | some sh =>
          simp only [wgKeeps, Bool.and_eq_true, Bool.not_eq_true'] at hk
          obtain ⟨⟨hK, hT⟩, hr⟩ := hk
          have hks := hst.kinds k (notIn hK)
          rcases hx with hlost | hx
          · exact .inl (hst.lost.trans hlost)
          refine .inr fun i d hi hkd => ?_
          obtain ⟨d0, hd0, hk0⟩ := (hks i).mpr ⟨d, hi, hkd⟩
          obtain ⟨h1, h2⟩ := hx i d0 hd0 hk0
          have hiT : i ∉ Ti := by
            intro hm
            rcases hst.touched i hm true d0 hd0 with e | e
            · exact notIn hT (hk0 ▸ e)
            · exact notIn hK (hk0 ▸ e)
          obtain ⟨d', hd', hs⟩ := hst.same i true d0 hd0 hiT (by rw [hk0]; exact notIn hT)
          rw [hi] at hd'; cases hd'
          exact ⟨(hks.isOne i) ▸ h1, (WgState.Fits.congr (readsOk hr)).mp ((hs.2 _ _).mp h2)⟩
  | held k a =>
      cases k with
      | wg k =>
          simp only [wgKeeps, Bool.not_eq_true'] at hk
          have hks := hst.kinds k (notIn hk)
          simp only [Fact.holds, Held.live, Bool.or_eq_true] at hx ⊢
          rw [hst.lost]
          rcases hx with hx | hx
          · exact .inl hx
          refine .inr ?_
          split at hx
          · rw [← hks.isOne]; exact hx
          · cases hx
      | _ => exact hx
  | opened k a =>
      cases k with
      | wg k =>
          simp only [wgKeeps, Bool.not_eq_true'] at hk
          have hks := hst.kinds k (notIn hk)
          intro h0
          have hx := hx h0
          simp only [Held.live, Bool.or_eq_true] at hx ⊢
          rw [hst.lost]
          rcases hx with hx | hx
          · exact .inl hx
          refine .inr ?_
          split at hx
          · rw [← hks.isOne]; exact hx
          · cases hx
      | _ => exact hx
  | _ => exact holds_wg hx (fun _ _ => ⟨nofun, nofun⟩) rfl

/-- What `wgKeeps` keeps of a typestate survives the step. -/
theorem wgKeeps_filter {Ti : List Nat} {TK K : List WgObj} {g : Option Nat} {n : Nat} {w : World} {s' : WgState}
    {S : TState} (hS : S.holds w) (hst : WgStep w.wg s' Ti TK K n) (hg : ∀ m, g = some m → n = m) :
    TState.holds (S.filter (wgKeeps Ti TK K g)) { w with wg := s' } := fun x hx => by
  obtain ⟨hm, hk⟩ := List.mem_filter.mp hx
  exact wgKeeps_sound hst hg (hS x hm) hk


-- ---------------------------------------------------------------------------
-- What a row gives
-- ---------------------------------------------------------------------------

/-- **What a wgpu row promises**: the call answers, the world it leaves
    satisfies the typestate the row gives, its answer is one the row
    describes, and the rest of the world is kept but for the values of the
    kinds it may end or make. -/
def WgOk (f : WgpuFn) (bits : List UInt64) (w : World) (S' : TState) (av : AV) : Prop :=
  ∃ r w', wgpuCall f bits w = some (r, w') ∧ S'.holds w' ∧ (∀ (Γ : Env) v, r = some v → av.Desc Γ w' v) ∧
    CuKeptBut ((wgKills f).map .wg) w w'

/-- A call but a buffer read, on a device not lost, is the effect of what it
    decodes to. -/
theorem wgpuCall_of {f : WgpuFn} {bits : List UInt64} {w : World} {c : WgCall} (hl : w.wg.lost = false)
    (hd : wgDecode f bits w = some c) (hf : f ≠ .bufferRead) : wgpuCall f bits w = some (wgEffect c w) := by
  unfold wgpuCall
  rw [hl, if_neg (by simp), hd]
  simp only [Option.map_some, Option.some.injEq]
  split
  · exact absurd rfl hf
  · rfl

/-- A step on wgpu's objects alone keeps the rest of the world, but the values
    of the kinds whose live objects it changes. -/
theorem wgKept {w : World} {s' : WgState} {Ti : List Nat} {TK K : List WgObj} {g : Nat} {f : WgpuFn}
    (hst : WgStep w.wg s' Ti TK K g) (hK : ∀ k ∈ K, k ∈ wgKills f) :
    CuKeptBut ((wgKills f).map .wg) w { w with wg := s' } where
  sizes _ _ := rfl
  held k x hk hl := by
    cases k with
    | wg k =>
        have hk' : k ∉ K := fun e => hk (List.mem_map_of_mem (hK k e))
        simp only [Held.live, Bool.or_eq_true] at hl ⊢
        rw [hst.lost]
        rcases hl with hl | hl
        · exact .inl hl
        refine .inr ?_
        split at hl
        · rw [← (hst.kinds k hk').isOne]; exact hl
        · cases hl
    | _ => exact hl
  blk _ _ hb := hb
  obs := rfl

/-- Losing the device keeps the rest of the world. -/
theorem wgKept_lose {w : World} (d : WgData) (f : WgpuFn) :
    CuKeptBut ((wgKills f).map .wg) w { w with wg := (w.wg.lose d).2 } where
  sizes _ _ := rfl
  held k x _ hl := by
    cases k with
    | wg k => simp [Held.live, WgState.lose]
    | _ => exact hl
  blk _ _ hb := hb
  obs := rfl

/-- The count `wgBump` says holds after a step that makes one object. -/
theorem wgBump_holds {S : TState} {w w' : World} (hS : S.holds w)
    (hsz : w'.wg.objs.size = w.wg.objs.size + 1) : ∀ x ∈ wgBump S, x.holds w' := by
  unfold wgBump
  split
  · rename_i n hn
    intro x hx
    simp only [List.mem_singleton] at hx
    subst hx
    show w'.wg.objs.size = n + 1
    rw [hsz, nwgOf_holds hS hn]
  · intro x hx; cases hx

/-- Every fact but a count of wgpu's objects holds once the device is lost. -/
theorem holds_lose {x : Fact} {w : World} (hx : x.holds w) (hn : ∀ m, x ≠ .nwg m) (d : WgData) :
    x.holds { w with wg := (w.wg.lose d).2 } := by
  cases x with
  | nwg m => exact absurd rfl (hn m)
  | wgo _ _ => exact .inl rfl
  | wg1 _ o => cases o <;> exact .inl rfl
  | held k a => cases k with
    | wg k => simp [Fact.holds, Held.live, WgState.lose]
    | _ => exact hx
  | opened k a => cases k with
    | wg k => intro _; simp [Held.live, WgState.lose]
    | _ => exact hx
  | _ => exact holds_wg hx (fun _ _ => ⟨nofun, nofun⟩) rfl

theorem wgKept_same (f : WgpuFn) (w : World) : CuKeptBut ((wgKills f).map .wg) w w :=
  ⟨fun _ _ => rfl, fun _ _ _ h => h, fun _ _ h => h, rfl⟩

/-- The one live object of kind `k` once one is added to a state with none. -/
theorem isOne_push {s : WgState} {d : WgData} {k : WgObj} (hn : s.NoneOf k) (hk : d.kind = k) :
    (s.push d).2.isOne k s.objs.size = true := by
  rw [WgState.isOne_iff]
  have hget : ∀ j, (s.push d).2.objs[j]? = if j = s.objs.size then some (true, d) else s.objs[j]? := by
    intro j; simp only [WgState.push]; rw [Array.getElem?_push]
  refine ⟨⟨d, by rw [hget, if_pos rfl], hk⟩, fun j ⟨d', hd', hk'⟩ => ?_⟩
  rw [hget] at hd'
  split at hd'
  · assumption
  · exact absurd hk' (hn j d' hd')

/-- The one object of kind `k`, made where there was none, of shape `sh`. -/
theorem wg1_push {w : World} {s : WgState} {d : WgData} {k : WgObj} {sh : WgShape} (hn : s.NoneOf k)
    (hk : d.kind = k) (hf : (s.push d).2.Fits d sh) : (Fact.wg1 k (some sh)).holds { w with wg := (s.push d).2 } := by
  refine .inr fun i d' hi hk' => ?_
  have h1 := isOne_push hn hk
  obtain ⟨-, hu⟩ := WgState.isOne_iff.mp h1
  obtain rfl := hu i ⟨d', hi, hk'⟩
  have : d' = d := by
    simp only [WgState.push] at hi
    rw [Array.getElem?_push, if_pos rfl] at hi; cases hi; rfl
  subst this
  exact ⟨h1, hf⟩

theorem wgHandle_lt {n : Nat} (h : n < WG_OBJS) : wgIndex (wgHandle n) = some n :=
  wgIndex_handle (by unfold WG_OBJS at h; omega)

/-- A handle to the one object of kind `k`. -/
theorem hand_desc {Γ : Env} {w : World} {k : WgObj} {x : UInt64} (h : Held.live (.wg k) w x = true) :
    (AV.hand (.wg k) true).Desc Γ w (.sc .i64 x) :=
  ⟨x, rfl, .inr h⟩

/-- The one object of kind `k` made by a push is what its handle names. -/
theorem live_push {s : WgState} {d : WgData} {k : WgObj} {w : World} (hn : s.NoneOf k) (hk : d.kind = k)
    (hs : s.objs.size < WG_OBJS) : Held.live (.wg k) { w with wg := (s.push d).2 } (s.push d).1 = true := by
  simp only [Held.live, WgState.push, wgHandle_lt hs, Bool.or_eq_true]
  exact .inr (isOne_push hn hk)

-- ---------------------------------------------------------------------------
-- Objects made once
-- ---------------------------------------------------------------------------

/-- **An object made once**: a call that decodes to making `d`, of shape `sh`,
    leaves what `wgMake` says. -/
theorem wgMake_ok {S S' : TState} {av : AV} {w : World} {f : WgpuFn} {bits : List UInt64} {d : WgData}
    {sh : WgShape} (hS : S.holds w) (hl : w.wg.lost = false) (h : wgMake S sh = some (S', av))
    (hdec : wgDecode f bits w = some (.make d [] [])) (hf : f ≠ .bufferRead) (hk : d.kind = sh.kind)
    (hfit : (w.wg.push d).2.Fits d sh) (hkill : sh.kind ∈ wgKills f) : WgOk f bits w S' av := by
  unfold wgMake at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨n, hn, h⟩ := h
  split at h
  · rename_i hlt
    cases h
    have hsz := nwgOf_holds hS hn
    have hlt' : w.wg.objs.size < WG_OBJS := by rw [hsz]; exact hlt
    have hst := WgStep.push w.wg d
    refine ⟨_, _, wgpuCall_of hl hdec hf, ?_, fun Γ v hv => ?_, ?_⟩
    · simp only [wgEffect, WgState.setAll, WgState.freeAll, List.foldl_nil, if_pos hlt']
      refine TState.holds_cons ?_ (TState.holds_cons ?_ (wgKeeps_filter hS (hk ▸ hst) (fun m e => by cases e; rfl)))
      · show (w.wg.push d).2.objs.size = n + 1
        rw [hst.size, hsz]
      · refine .inr ⟨d, ?_, hfit⟩
        simp only [WgState.push]; rw [Array.getElem?_push, if_pos hsz.symm]
    · simp only [wgEffect, WgState.setAll, WgState.freeAll, List.foldl_nil, if_pos hlt'] at hv
      cases hv
      show _ = _
      simp [WgState.push, hsz]
    · simp only [wgEffect, WgState.setAll, WgState.freeAll, List.foldl_nil, if_pos hlt']
      exact wgKept hst (fun k e => by simp only [List.mem_singleton] at e; rw [e, hk]; exact hkill)
  · cases h


theorem mapM_of {α β : Type} {f g : α → Option β} :
    ∀ {l : List α} {r : List β}, l.mapM f = some r → (∀ x y, f x = some y → g x = some y) → l.mapM g = some r
  | [], r, h, _ => h
  | x :: xs, r, h, hfg => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h ⊢
      obtain ⟨y, hy, ys, hys, rfl⟩ := h
      exact ⟨y, hfg x y hy, ys, mapM_of hys hfg, rfl⟩

theorem wgArray_mono {α : Type} {a n : UInt64} {stride : Nat} {f g : UInt64 → Option α} {r : List α}
    (h : wgArray a n stride f = some r) (hfg : ∀ x y, f x = some y → g x = some y) :
    wgArray a n stride g = some r := by
  unfold wgArray at h; unfold wgArray
  split at h
  · cases h
  · rw [if_neg (by assumption)]; exact mapM_of h (fun _ y hy => hfg _ y hy)

theorem cellOf_load {S : TState} {w : World} {a v : UInt64} (hS : S.holds w) (h : cellOf S a = some v) :
    w.mem.load a 8 = some v :=
  hS _ (cellOf_mem h)

theorem wgText_ofNat {m : Mem} {a k : UInt64} {t : String} (h : wgText m a (UInt64.ofNat k.toNat) = some t) :
    wgText m a k = some t := by
  rwa [UInt64.ofNat_toNat] at h

-- ---------------------------------------------------------------------------
-- The rows for objects made once
-- ---------------------------------------------------------------------------

section
variable {S S' : TState} {w : World} {Γ : Env} {bits : List UInt64} {av : AV}

theorem row_createInstance (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [] bits)
    (h : wgRow S .createInstance [] = some (S', av)) : WgOk .createInstance bits w S' av := by
  obtain rfl := ArgBits.nil_inv hb
  exact wgMake_ok (d := .instance) hS hl h (by simp [wgDecode]) (by decide) rfl (by simp [WgState.Fits])
    (by simp [wgKills, WgShape.kind])

theorem row_requestAdapter {i : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [i] bits)
    (h : wgRow S .requestAdapter [i] = some (S', av)) : WgOk .requestAdapter bits w S' av := by
  simp only [wgRow, wgR_requestAdapter, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨⟨ii, si⟩, hi, h⟩ := h
  split at h
  · rename_i hc
    obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
    obtain rfl := ArgBits.nil_inv hb
    obtain ⟨d, hg, -⟩ := wgoArg_holds hS hl hx hi
    have ha : w.gpuAdapter = true := holds_of_mem hS hc
    exact wgMake_ok (d := .adapter) hS hl h (by simp [wgDecode, wgDRequestAdapter, hg, ha]) (by decide) rfl
      (by simp [WgState.Fits]) (by simp [wgKills, WgShape.kind])
  · cases h

theorem row_requestDevice {a : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [a] bits)
    (h : wgRow S .requestDevice [a] = some (S', av)) : WgOk .requestDevice bits w S' av := by
  simp only [wgRow, wgR_requestDevice, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨⟨ii, si⟩, hi, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  obtain ⟨d, hg, -⟩ := wgoArg_holds hS hl hx hi
  exact wgMake_ok (d := .device []) hS hl h (by simp [wgDecode, wgDRequestDevice, hg]) (by decide) rfl
    (by simp [WgState.Fits]) (by simp [wgKills, WgShape.kind])

theorem row_deviceGetQueue {a : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [a] bits)
    (h : wgRow S .deviceGetQueue [a] = some (S', av)) : WgOk .deviceGetQueue bits w S' av := by
  simp only [wgRow, wgR_deviceGetQueue, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨⟨ii, sc⟩, hi, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  obtain ⟨hg, -⟩ := wgDev_holds hS hl hx hi
  exact wgMake_ok (d := .queue) hS hl h (by simp [wgDecode, wgDGetQueue, hg]) (by decide) rfl
    (by simp [WgState.Fits]) (by simp [wgKills, WgShape.kind])

theorem row_deviceCreateBuffer {a u sz : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [a, u, sz] bits)
    (h : wgRow S .deviceCreateBuffer [a, u, sz] = some (S', av)) : WgOk .deviceCreateBuffer bits w S' av := by
  simp only [wgRow, wgR_deviceCreateBuffer, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨⟨ii, sc⟩, hi, usage, hu, size, hsz, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨y, b, rfl, hy, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨z, b, rfl, hz, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  obtain rfl := known_bits hy hu
  obtain rfl := known_bits hz hsz
  obtain ⟨hg, -⟩ := wgDev_holds hS hl hx hi
  split at h
  · cases h
  · rename_i hc
    exact wgMake_ok (d := .buffer y (ByteArray.mk (Array.replicate z.toNat 0))) hS hl h (by simp only [wgDecode, wgDCreateBuffer, hg, Option.bind_eq_bind, Option.bind_some]; rw [if_neg hc])
      (by decide) rfl (by simp [WgState.Fits, ByteArray.size]) (by simp [wgKills, WgShape.kind])

theorem row_deviceCreateShaderModule {a src n : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [a, src, n] bits)
    (h : wgRow S .deviceCreateShaderModule [a, src, n] = some (S', av)) :
    WgOk .deviceCreateShaderModule bits w S' av := by
  simp only [wgRow, wgR_deviceCreateShaderModule, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨⟨ii, sc⟩, hi, sa, hsa, k, hk, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨y, b, rfl, hy, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨z, b, rfl, hz, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  obtain rfl := known_bits hy hsa
  obtain rfl := known_bits hz hk
  obtain ⟨hg, -⟩ := wgDev_holds hS hl hx hi
  split at h
  · rename_i hc
    obtain ⟨t, ht, hok⟩ := wgText_of hS (List.contains_iff_mem.mp hc)
    have ht' := wgText_ofNat ht
    have hok' := hok rfl
    exact wgMake_ok (d := .shader t true) hS hl h
      (by simp [wgDecode, wgDCreateShader, ht', hok', WgState.errUnless, hg]) (by decide) rfl
      (by simp [WgState.Fits]) (by simp [wgKills, WgShape.kind])
  · cases h

theorem row_deviceCreateBindGroupLayout {a ents n : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [a, ents, n] bits)
    (h : wgRow S .deviceCreateBindGroupLayout [a, ents, n] = some (S', av)) :
    WgOk .deviceCreateBindGroupLayout bits w S' av := by
  simp only [wgRow, wgR_deviceCreateBindGroupLayout, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨⟨ii, sc⟩, hi, ea, hea, k, hk, es, hes, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨y, b, rfl, hy, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨z, b, rfl, hz, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  obtain rfl := known_bits hy hea
  obtain rfl := known_bits hz hk
  obtain ⟨hg, -⟩ := wgDev_holds hS hl hx hi
  have harr := wgArray_mono (g := wgBglEntry w.mem) hes (fun a v hv => by
    simp only [wgBglEntry, Option.bind_eq_bind, Option.bind_eq_some_iff] at hv ⊢
    obtain ⟨e, he, hv⟩ := hv
    exact ⟨e, cellOf_load hS he, hv⟩)
  split at h
  · rename_i hnd
    exact wgMake_ok (d := .bgl (some es)) hS hl h (by simp [wgDecode, wgDCreateBgl, hg, harr, hnd]) (by decide) rfl
      (by simp [WgState.Fits]) (by simp [wgKills, WgShape.kind])
  · cases h

theorem row_deviceCreatePipelineLayout {a arr n : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [a, arr, n] bits)
    (h : wgRow S .deviceCreatePipelineLayout [a, arr, n] = some (S', av)) :
    WgOk .deviceCreatePipelineLayout bits w S' av := by
  simp only [wgRow, wgR_deviceCreatePipelineLayout, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨⟨ii, sc⟩, hi, ea, hea, k, hk, gs, hgs, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨y, b, rfl, hy, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨z, b, rfl, hz, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  obtain rfl := known_bits hy hea
  obtain rfl := known_bits hz hk
  obtain ⟨hg, -⟩ := wgDev_holds hS hl hx hi
  have harr := wgArray_mono (g := wgGroupAt w.wg w.mem) hgs (fun a v hv => by
    simp only [wgGroupAt, Option.bind_eq_bind, Option.bind_eq_some_iff] at hv ⊢
    obtain ⟨h, hh, hv⟩ := hv
    split at hv
    · rename_i j es he
      cases hv
      obtain ⟨d, hgd, -, hf, -⟩ := wgoAt_holds hS hl he
      have : d = .bgl (some v) := by cases d <;> first | exact hf.elim | (cases hf; rfl)
      subst this
      exact ⟨h, cellOf_load hS hh, (j, .bgl (some v)), hgd, rfl⟩
    · cases hv)
  exact wgMake_ok (d := .playout gs) hS hl h (by simp [wgDecode, wgDCreatePlayout, hg, harr]) (by decide) rfl
    (by simp [WgState.Fits]) (by simp [wgKills, WgShape.kind])

end

section
variable {S S' : TState} {w : World} {Γ : Env} {bits : List UInt64} {av : AV}

theorem fits_eq_playout {s : WgState} {d : WgData} {gs : List (List (Nat × Bool))} (h : s.Fits d (.playout gs)) :
    d = .playout gs := by
  cases d <;> first | exact h.elim | (cases h; rfl)

theorem fits_eq_bgl {s : WgState} {d : WgData} {e : Option (List (Nat × Bool))} (h : s.Fits d (.bgl e)) :
    d = .bgl e := by
  cases d <;> first | exact h.elim | (cases h; rfl)

theorem fits_eq_rpipe {s : WgState} {d : WgData} {v : Bool} (h : s.Fits d (.rpipe v)) : d = .rpipe v := by
  cases d <;> first | exact h.elim | (cases h; rfl)

theorem fits_shader {s : WgState} {d : WgData} {v : Bool} (h : s.Fits d (.shader v)) : ∃ src, d = .shader src v := by
  cases d <;> first | exact h.elim | (cases h; exact ⟨_, rfl⟩)

theorem fits_buffer {s : WgState} {d : WgData} {u : UInt64} {n : Nat} (h : s.Fits d (.buffer u n)) :
    ∃ bs, d = .buffer u bs ∧ bs.size = n := by
  cases d <;> first | exact h.elim | (obtain ⟨rfl, h⟩ := h; exact ⟨_, rfl, h⟩)

theorem errUnless_ok {s : WgState} {h : UInt64} {i : Nat} {sc : List Bool} (hd : s.device? h = some (i, sc)) :
    s.errUnless h true = some [] := by
  simp [WgState.errUnless, hd]

theorem row_deviceCreateComputePipeline {a lh mh entry n : AV} (hS : S.holds w) (hl : w.wg.lost = false)
    (hb : ArgBits Γ w [a, lh, mh, entry, n] bits)
    (h : wgRow S .deviceCreateComputePipeline [a, lh, mh, entry, n] = some (S', av)) :
    WgOk .deviceCreateComputePipeline bits w S' av := by
  simp only [wgRow, wgR_deviceCreateComputePipeline, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨⟨li, l⟩, hL, ⟨mi, m⟩, hm, ⟨ii, sc⟩, hi, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨y, b, rfl, hy, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨z, b, rfl, hz, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨u, b, rfl, hu, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨v, b, rfl, hv, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  obtain ⟨hg, -⟩ := wgDev_holds hS hl hx hi
  obtain ⟨dl, hgl, -, hfl, hkl⟩ := wgoArg_holds hS hl hy hL
  obtain ⟨dm, hgm, -, hfm, hkm⟩ := wgoArg_holds hS hl hz hm
  cases l <;> simp only [WgShape.kind, reduceCtorEq] at hkl
  cases m <;> simp only [WgShape.kind, reduceCtorEq] at hkm
  rename_i gs v
  obtain rfl := fits_eq_playout hfl
  obtain ⟨src, rfl⟩ := fits_shader hfm
  cases v
  · simp at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨ea, hea, k, hk, h⟩ := h
  obtain rfl := known_bits hu hea
  obtain rfl := known_bits hv hk
  split at h
  · rename_i htx
    obtain ⟨t, ht⟩ := wgTextAt_holds hS htx
    exact wgMake_ok (d := .cpipe src gs true) hS hl h
      (by simp [wgDecode, wgDCreateCpipe, hgl, hgm, wgText_ofNat ht, errUnless_ok hg]) (by decide) rfl
      (by simp [WgState.Fits]) (by simp [wgKills, WgShape.kind])
  · cases h

/-- A bind group entry the typestate's cells name is one wgpu reads. -/
theorem bgEntry_of {S : TState} {w : World} (hS : S.holds w) (hl : w.wg.lost = false) {e : UInt64} {v : Nat × Nat}
    (hv : (do
      let b ← cellOf S e
      let bh ← cellOf S (e + 8)
      match wgoAt S bh .buffer with
      | some (bi, .buffer u _) => if u &&& 128 == 0 || b ≥ 0x100000000 then none else some (b.toNat, bi)
      | _ => none) = some v) :
    wgBgEntry w.wg w.mem e = some v := by
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at hv
  obtain ⟨b0, hb0, bh, hbh, hv⟩ := hv
  split at hv
  · rename_i bi u0 n0 hw
    obtain ⟨d, hgd, -, hfd, -⟩ := wgoAt_holds hS hl hw
    obtain ⟨bs, rfl, -⟩ := fits_buffer hfd
    unfold wgBgEntry
    rw [cellOf_load hS hb0, cellOf_load hS hbh]
    simp only [Option.bind_eq_bind, Option.bind_some, hgd]
    exact hv
  · cases hv

theorem row_deviceCreateBindGroup {a lh ents n : AV} (hS : S.holds w) (hl : w.wg.lost = false)
    (hb : ArgBits Γ w [a, lh, ents, n] bits)
    (h : wgRow S .deviceCreateBindGroup [a, lh, ents, n] = some (S', av)) :
    WgOk .deviceCreateBindGroup bits w S' av := by
  simp only [wgRow, wgR_deviceCreateBindGroup, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨⟨ii, sc⟩, hi, ⟨li, l⟩, hL, ea, hea, k, hk, es, hes, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨y, b, rfl, hy, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨z, b, rfl, hz, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨u, b, rfl, hu, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  obtain rfl := known_bits hz hea
  obtain rfl := known_bits hu hk
  obtain ⟨hg, -⟩ := wgDev_holds hS hl hx hi
  obtain ⟨dl, hgl, -, hfl, hkl⟩ := wgoArg_holds hS hl hy hL
  have harr := wgArray_mono (g := wgBgEntry w.wg w.mem) hes (fun _ _ hv => bgEntry_of hS hl hv)
  cases l <;> simp only [WgShape.kind, reduceCtorEq] at hkl
  rename_i e
  obtain rfl := fits_eq_bgl hfl
  cases e with
  | none =>
      simp only at h
      split at h
      · rename_i hfits
        exact wgMake_ok (d := .bgroup es) hS hl h
          (by simp only [decide_eq_true_eq] at hfits; simp [wgDecode, wgDCreateBgroup, hg, hgl, harr, hfits])
          (by decide) rfl (by simp [WgState.Fits]) (by simp [wgKills, WgShape.kind])
      · cases h
  | some ls =>
      simp only at h
      split at h
      · rename_i hfits
        exact wgMake_ok (d := .bgroup es) hS hl h
          (by simp only [Bool.and_eq_true, decide_eq_true_eq] at hfits
              simp [wgDecode, wgDCreateBgroup, hg, hgl, harr, hfits])
          (by decide) rfl (by simp [WgState.Fits]) (by simp [wgKills, WgShape.kind])
      · cases h

theorem row_deviceCreateRenderPipeline {a mh vs vn fs fn fmt : AV} (hS : S.holds w) (hl : w.wg.lost = false)
    (hb : ArgBits Γ w [a, mh, vs, vn, fs, fn, fmt] bits)
    (h : wgRow S .deviceCreateRenderPipeline [a, mh, vs, vn, fs, fn, fmt] = some (S', av)) :
    WgOk .deviceCreateRenderPipeline bits w S' av := by
  simp only [wgRow, wgR_deviceCreateRenderPipeline, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨⟨mi, m⟩, hm, ⟨ii, sc⟩, hi, va, hva, vk, hvk, fa, hfa, fk, hfk, ft, hft, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨y, b, rfl, hy, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨x1, b, rfl, hx1, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨x2, b, rfl, hx2, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨x3, b, rfl, hx3, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨x4, b, rfl, hx4, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨x5, b, rfl, hx5, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  obtain rfl := known_bits hx1 hva
  obtain rfl := known_bits hx2 hvk
  obtain rfl := known_bits hx3 hfa
  obtain rfl := known_bits hx4 hfk
  obtain rfl := known_bits hx5 hft
  obtain ⟨hg, -⟩ := wgDev_holds hS hl hx hi
  obtain ⟨dm, hgm, -, hfm, -⟩ := wgoArg_holds hS hl hy hm
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, beq_iff_eq] at hc
    obtain ⟨⟨⟨rfl, h1⟩, h2⟩, h3⟩ := hc
    obtain ⟨src, rfl⟩ := fits_shader hfm
    obtain ⟨t1, ht1⟩ := wgTextAt_holds hS h1
    obtain ⟨t2, ht2⟩ := wgTextAt_holds hS h2
    exact wgMake_ok (d := .rpipe true) hS hl h
      (by simp [wgDecode, wgDCreateRpipe, hgm, wgText_ofNat ht1, wgText_ofNat ht2, h3, errUnless_ok hg])
      (by decide) rfl (by simp [WgState.Fits]) (by simp [wgKills, WgShape.kind])
  · cases h

theorem row_renderPipelineGetBindGroupLayout {p idx : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [p, idx] bits)
    (h : wgRow S .renderPipelineGetBindGroupLayout [p, idx] = some (S', av)) :
    WgOk .renderPipelineGetBindGroupLayout bits w S' av := by
  simp only [wgRow, wgR_renderPipelineGetBindGroupLayout, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨⟨pi, ps⟩, hp, k, hk, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨y, b, rfl, hy, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  obtain rfl := known_bits hy hk
  obtain ⟨d, hgd, -, hfd, -⟩ := wgoArg_holds hS hl hx hp
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, beq_iff_eq] at hc
    obtain ⟨rfl, h0⟩ := hc
    obtain rfl := fits_eq_rpipe hfd
    exact wgMake_ok (d := .bgl none) hS hl h (by simp [wgDecode, wgDRpipeBgl, hgd, h0]) (by decide) rfl
      (by simp [WgState.Fits]) (by simp [wgKills, WgShape.kind])
  · cases h

theorem row_instanceCreateSurface {i win : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [i, win] bits)
    (h : wgRow S .instanceCreateSurface [i, win] = some (S', av)) :
    WgOk .instanceCreateSurface bits w S' av := by
  simp only [wgRow, wgR_instanceCreateSurface, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨⟨ii, si⟩, hi, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨y, b, rfl, hy, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  obtain ⟨d, hgd, -⟩ := wgoArg_holds hS hl hx hi
  split at h
  · split at h
    · rename_i hn
      have hnone := wg1Of_none hS hl (by simpa using hn)
      obtain ⟨t, hw⟩ := hy
      obtain ⟨y', e, hw'⟩ := hw
      simp only [V.sc.injEq] at e
      obtain ⟨-, rfl⟩ := e
      rcases hw' with ⟨h0, -⟩ | hw'
      · cases h0
      · obtain ⟨j, hj, hlj⟩ := Contracts.window_live_iff.mp hw'
        have hopen : ((List.range w.wl.windows.size).any fun k =>
            wlWinAddr k == y && (w.wl.windows[k]?.map (·.live)).getD false) = true := by
          have hjlt : j < w.wl.windows.size := by
            cases e : w.wl.windows[j]? with
            | none => rw [e] at hlj; cases hlj
            | some _ => exact (Array.getElem?_eq_some_iff.mp e).1
          exact List.any_eq_true.mpr ⟨j, List.mem_range.mpr hjlt, by simp [hj, hlj]⟩
        refine wgMake_ok (d := .surface false none) hS hl h (by simp [wgDecode, wgDCreateSurface, hgd, hopen])
          (by decide) rfl ?_ (by simp [wgKills, WgShape.kind])
        refine ⟨rfl, ?_⟩
        exact ((WgStep.push w.wg (.surface false none)).kinds .texture (by simp [WgData.kind])).noneOf.mp hnone
    · cases h
  · cases h

end


-- ---------------------------------------------------------------------------
-- Objects made afresh, and changes
-- ---------------------------------------------------------------------------

/-- **The one object of kind `k` made**, after the changes the call makes to
    the others: the call answers it, or loses the device, and leaves what the
    row says either way. -/
theorem wgMakeOne_ok {S N : TState} {w : World} {f : WgpuFn} {bits : List UInt64} {d : WgData}
    {ch : List (Nat × WgData)} {fr : List Nat} {Ti : List Nat} {TK K : List WgObj} {k : WgObj} {sh : WgShape}
    (hS : S.holds w) (hl : w.wg.lost = false) (hdec : wgDecode f bits w = some (.make d ch fr)) (hf : f ≠ .bufferRead)
    (hst : WgStep w.wg ((w.wg.setAll ch).freeAll fr) Ti TK K 0)
    (hnone : ((w.wg.setAll ch).freeAll fr).NoneOf k) (hk : d.kind = k)
    (hfit : (((w.wg.setAll ch).freeAll fr).push d).2.Fits d sh)
    (hN : N.holds { w with wg := (w.wg.setAll ch).freeAll fr })
    (hNk : ∀ x ∈ N, wgKeeps [] [] [k] (some 1) x = true)
    (hNw : ∀ x ∈ N, ∃ k' o, x = .wg1 k' o)
    (hkill : ∀ k' ∈ K ++ [k], k' ∈ wgKills f) :
    WgOk f bits w (.wg1 k (some sh) :: (N ++ (wgBump S ++ S.filter (wgKeeps Ti TK (K ++ [k]) none)))) (.hand (.wg k) true) := by
  generalize hs1 : (w.wg.setAll ch).freeAll fr = s1 at hst hnone hfit hN
  have hsz : s1.objs.size = w.wg.objs.size := by rw [hst.size]; rfl
  refine ⟨_, _, wgpuCall_of hl hdec hf, ?_⟩
  simp only [wgEffect, hs1]
  split
  · rename_i hlt
    have hpush := WgStep.push s1 d
    have h2 := hst.trans hpush
    rw [List.append_nil, List.append_nil, hk] at h2
    refine ⟨?_, fun Γ v hv => ?_, wgKept h2 hkill⟩
    · refine TState.holds_cons (wg1_push hnone hk hfit) fun x hx => ?_
      rcases List.mem_append.mp hx with hx | hx
      · exact wgKeeps_sound (w := { w with wg := s1 }) (hk ▸ hpush) (fun m e => by cases e; rfl) (hN x hx) (hNk x hx)
      rcases List.mem_append.mp hx with hx | hx
      · exact wgBump_holds hS (by rw [hpush.size, hsz]) x hx
      · exact wgKeeps_filter hS h2 (fun _ e => by cases e) x hx
    · cases hv
      exact hand_desc (live_push hnone hk (by rw [hsz]; exact hlt))
  · refine ⟨TState.holds_cons (.inl rfl) fun x hx => ?_, fun Γ v hv => ?_, wgKept_lose d f⟩
    · rcases List.mem_append.mp hx with hx | hx
      · obtain ⟨k', o, rfl⟩ := hNw x hx
        cases o <;> exact .inl rfl
      rcases List.mem_append.mp hx with hx | hx
      · exact wgBump_holds hS (by simp [WgState.lose]) x hx
      · obtain ⟨hm, hkx⟩ := List.mem_filter.mp hx
        exact holds_lose (hS x hm) (fun m e => by subst e; simp [wgKeeps] at hkx) d
    · cases hv
      exact hand_desc (by simp [Held.live, WgState.lose])

/-- **A change to objects**, none made: the call answers nothing and leaves
    what the row says. -/
theorem wgUpdate_ok {S N : TState} {w : World} {f : WgpuFn} {bits : List UInt64}
    {ch : List (Nat × WgData)} {fr : List Nat} {Ti : List Nat} {TK K : List WgObj}
    (hS : S.holds w) (hl : w.wg.lost = false) (hdec : wgDecode f bits w = some (.update ch fr)) (hf : f ≠ .bufferRead)
    (hst : WgStep w.wg ((w.wg.setAll ch).freeAll fr) Ti TK K 0)
    (hN : N.holds { w with wg := (w.wg.setAll ch).freeAll fr })
    (hkill : ∀ k' ∈ K, k' ∈ wgKills f) :
    WgOk f bits w (N ++ S.filter (wgKeeps Ti TK K (some 0))) wgNone := by
  refine ⟨_, _, wgpuCall_of hl hdec hf, fun x hx => ?_, fun Γ v hv => by simp [wgEffect] at hv, wgKept hst hkill⟩
  rcases List.mem_append.mp hx with hx | hx
  · exact hN x hx
  · exact wgKeeps_filter hS hst (fun m e => by cases e; rfl) x hx

theorem setAll_one (s : WgState) (i : Nat) (d : WgData) : s.setAll [(i, d)] = s.set i d := rfl
theorem setAll_nil (s : WgState) : s.setAll [] = s := rfl
theorem freeAll_nil (s : WgState) : s.freeAll [] = s := rfl

/-- The objects of kind `k` live after setting one of kind `k'` to data of
    its kind are those before. -/
theorem kindSame_set {s : WgState} {i : Nat} {d : WgData} {k : WgObj}
    (hk : ∀ l d0, s.objs[i]? = some (l, d0) → d0.kind = d.kind) : KindSame s (s.set i d) k :=
  (WgStep.set hk rfl).kinds k (fun h => nomatch h)

/-- The one object of kind `k` in `s`, its data changed to `d` of that kind, is
    the one in the state after, of whatever shape `d` is. -/
theorem wg1_set {w : World} {i : Nat} {d0 d : WgData} {k : WgObj} {sh : WgShape}
    (hone : w.wg.isOne k i = true) (hi : w.wg.objs[i]? = some (true, d0)) (hk : d.kind = k)
    (hf : (w.wg.set i d).Fits d sh) : (Fact.wg1 k (some sh)).holds { w with wg := w.wg.set i d } := by
  refine .inr ?_
  have hk0 : d0.kind = k := by obtain ⟨⟨d', hd', hk'⟩, -⟩ := WgState.isOne_iff.mp hone; rw [hi] at hd'; cases hd'; exact hk'
  have hks : KindSame w.wg (w.wg.set i d) k :=
    kindSame_set (fun l d1 h => by rw [hi] at h; cases h; rw [hk0, hk])
  intro j d' hj hkj
  have hone' : (w.wg.set i d).isOne k j = true := by
    obtain ⟨-, hu⟩ := WgState.isOne_iff.mp hone
    obtain rfl := hu j ((hks j).mpr ⟨d', hj, hkj⟩)
    rw [← hks.isOne]; exact hone
  obtain ⟨-, hu⟩ := WgState.isOne_iff.mp hone
  obtain rfl := hu j ((hks j).mpr ⟨d', hj, hkj⟩)
  rw [WgState.set_get, if_pos rfl, hi] at hj
  simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at hj
  obtain ⟨-, rfl⟩ := hj
  exact ⟨hone', hf⟩


/-- Using up the one object of kind `k` leaves none. -/
theorem noneOf_free {s : WgState} {i : Nat} {k : WgObj} (hone : s.isOne k i = true) : (s.freeAll [i]).NoneOf k := by
  intro j d hj hk
  obtain ⟨-, hu⟩ := WgState.isOne_iff.mp hone
  rw [WgState.free_get] at hj
  split at hj
  · simp only [Option.map_eq_some_iff, Prod.mk.injEq] at hj
    obtain ⟨_, -, h, -⟩ := hj; cases h
  · exact absurd (hu j ⟨d, hj, hk⟩) (by omega)

theorem kind_of_one {s : WgState} {k : WgObj} {i : Nat} (h : s.isOne k i = true) :
    ∀ l d, s.objs[i]? = some (l, d) → d.kind = k := by
  obtain ⟨⟨d', hd', hk'⟩, -⟩ := WgState.isOne_iff.mp h
  intro l d hd; rw [hd'] at hd; cases hd; exact hk'

theorem wgKeeps_mem {Ti : List Nat} {TK K : List WgObj} {g : Option Nat} {N : TState}
    (h : N.all (wgKeeps Ti TK K g) = true) : ∀ x ∈ N, wgKeeps Ti TK K g x = true :=
  fun x hx => List.all_eq_true.mp h x hx

section
variable {S S' : TState} {w : World} {Γ : Env} {bits : List UInt64} {av : AV}

theorem fits_encoder {s : WgState} {d : WgData} {o : Bool} (h : s.Fits d (.encoder o)) : ∃ cmds, d = .encoder cmds o := by
  cases d <;> first | exact h.elim | (cases h; exact ⟨_, rfl⟩)

theorem fits_cpass {s : WgState} {d : WgData} {q : Option Nat} {b : List (Nat × Nat)} (h : s.Fits d (.cpass q b)) :
    ∃ e, d = .cpass e q b false ∧ s.isOne .commandEncoder e = true := by
  cases d <;> first | exact h.elim | (obtain ⟨he, rfl, rfl, rfl⟩ := h; exact ⟨_, rfl, he⟩)

theorem fits_rpass {s : WgState} {d : WgData} {q g : Option Nat} (h : s.Fits d (.rpass q g)) :
    ∃ e v, d = .rpass e v q g false ∧ s.isOne .commandEncoder e = true ∧ s.isOne .textureView v = true := by
  cases d <;> first | exact h.elim | (obtain ⟨he, hv, rfl, rfl, rfl⟩ := h; exact ⟨_, _, rfl, he, hv⟩)

theorem fits_cmdbuf {s : WgState} {d : WgData} (h : s.Fits d .cmdbuf) : ∃ cmds, d = .cmdbuf cmds false := by
  cases d <;> first | exact h.elim | (cases h; exact ⟨_, rfl⟩)

theorem fits_cpipe {s : WgState} {d : WgData} {gs : List (List (Nat × Bool))} {v : Bool} (h : s.Fits d (.cpipe gs v)) :
    ∃ src, d = .cpipe src gs v := by
  cases d <;> first | exact h.elim | (obtain ⟨rfl, rfl⟩ := h; exact ⟨_, rfl⟩)

theorem row_deviceCreateCommandEncoder {a : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [a] bits)
    (h : wgRow S .deviceCreateCommandEncoder [a] = some (S', av)) :
    WgOk .deviceCreateCommandEncoder bits w S' av := by
  simp only [wgRow, wgR_deviceCreateCommandEncoder, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨⟨ii, sc⟩, hi, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  obtain ⟨hg, -⟩ := wgDev_holds hS hl hx hi
  split at h
  · rename_i hn
    cases h
    exact wgMakeOne_ok (f := .deviceCreateCommandEncoder) (d := .encoder [] true) (ch := []) (fr := [])
      (k := .commandEncoder) (sh := .encoder true) (N := []) (Ti := []) (TK := []) (K := []) hS hl
      (by simp [wgDecode, wgDCreateEncoder, hg])
      (by decide) (WgStep.refl w.wg) (wg1Of_none hS hl (by simpa using hn)) rfl (by simp [WgState.Fits])
      (fun _ h => by cases h) (fun _ h => by cases h) (fun _ h => by cases h) (by simp [wgKills])
  · cases h

theorem row_encoderBeginComputePass {e : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [e] bits)
    (h : wgRow S .encoderBeginComputePass [e] = some (S', av)) :
    WgOk .encoderBeginComputePass bits w S' av := by
  simp only [wgRow, wgR_encoderBeginComputePass, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨se, hse, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, beq_iff_eq] at hc
    obtain ⟨rfl, hn⟩ := hc
    cases h
    obtain ⟨i, d, hgd, -, hdi, hkd, hone, hf⟩ := wg1Arg_holds hS hl hx hse
    obtain ⟨cmds, rfl⟩ := fits_encoder hf
    have hst := WgStep.set (s := w.wg) (i := i) (d := .encoder cmds false) (k := .commandEncoder) (kind_of_one hone) rfl
    have hnone := ((hst.kinds .computePass (fun h => nomatch h)).noneOf).mp (wg1Of_none hS hl hn)
    have hks := (hst.kinds .commandEncoder (fun h => nomatch h)).trans
      ((WgStep.push (w.wg.set i (.encoder cmds false)) (.cpass i none [] false)).kinds .commandEncoder (by simp [WgData.kind]))
    exact wgMakeOne_ok (f := .encoderBeginComputePass) (d := .cpass i none [] false) (ch := [(i, .encoder cmds false)])
      (fr := []) (k := .computePass) (sh := .cpass none []) (N := [.wg1 .commandEncoder (some (.encoder false))]) hS hl
      (by simp [wgDecode, wgDBeginCpass, hgd]) (by decide) hst hnone rfl
      (by simp only [WgState.Fits, and_true, setAll_one, setAll_nil, freeAll_nil]; rw [← hks.isOne]; exact hone)
      (TState.holds_cons (wg1_set hone hdi rfl (by simp [WgState.Fits])) (fun _ h => by cases h))
      (wgKeeps_mem (by decide)) (fun x hx => by simp only [List.mem_singleton] at hx; exact ⟨_, _, hx⟩)
      (by simp [wgKills])
  · cases h

theorem row_encoderBeginRenderPass {e v ld : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [e, v, ld] bits)
    (h : wgRow S .encoderBeginRenderPass [e, v, ld] = some (S', av)) :
    WgOk .encoderBeginRenderPass bits w S' av := by
  simp only [wgRow, wgR_encoderBeginRenderPass, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨se, hse, sv, hsv, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨y, b, rfl, hy, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨z, b, rfl, hz, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, beq_iff_eq] at hc
    obtain ⟨rfl, hn⟩ := hc
    cases h
    obtain ⟨i, d, hgd, -, hdi, hkd, hone, hf⟩ := wg1Arg_holds hS hl hx hse
    obtain ⟨vi, dv, hgv, -, hdv, hkv, honev, -⟩ := wg1Arg_holds hS hl hy hsv
    obtain ⟨cmds, rfl⟩ := fits_encoder hf
    have hst := WgStep.set (s := w.wg) (i := i) (d := .encoder cmds false) (k := .commandEncoder) (kind_of_one hone) rfl
    have hnone := ((hst.kinds .renderPass (fun h => nomatch h)).noneOf).mp (wg1Of_none hS hl hn)
    have hpush := WgStep.push (w.wg.set i (.encoder cmds false)) (.rpass i vi none none false)
    have hks := (hst.kinds .commandEncoder (fun h => nomatch h)).trans (hpush.kinds .commandEncoder (by simp [WgData.kind]))
    have hkv' := (hst.kinds .textureView (fun h => nomatch h)).trans (hpush.kinds .textureView (by simp [WgData.kind]))
    exact wgMakeOne_ok (f := .encoderBeginRenderPass) (d := .rpass i vi none none false) (ch := [(i, .encoder cmds false)])
      (fr := []) (k := .renderPass) (sh := .rpass none none) (N := [.wg1 .commandEncoder (some (.encoder false))]) hS hl
      (by simp [wgDecode, wgDBeginRpass, hgd, hgv]) (by decide) hst hnone rfl
      (by simp only [WgState.Fits, and_true, setAll_one, setAll_nil, freeAll_nil]; rw [← hks.isOne, ← hkv'.isOne]; exact ⟨hone, honev⟩)
      (TState.holds_cons (wg1_set hone hdi rfl (by simp [WgState.Fits])) (fun _ h => by cases h))
      (wgKeeps_mem (by decide)) (fun x hx => by simp only [List.mem_singleton] at hx; exact ⟨_, _, hx⟩)
      (by simp [wgKills])
  · cases h

theorem row_textureCreateView {t : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [t] bits)
    (h : wgRow S .textureCreateView [t] = some (S', av)) : WgOk .textureCreateView bits w S' av := by
  simp only [wgRow, wgR_textureCreateView, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨st, hst', h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  split at h
  · rename_i hn
    cases h
    obtain ⟨i, d, hgd, -, hdi, hkd, hone, -⟩ := wg1Arg_holds hS hl hx hst'
    have hpush := WgStep.push w.wg (.view i)
    exact wgMakeOne_ok (f := .textureCreateView) (d := .view i) (ch := []) (fr := []) (k := .textureView) (sh := .view)
      (N := []) (Ti := []) (TK := []) (K := []) hS hl
      (by simp [wgDecode, wgDCreateView, hgd]) (by decide) (WgStep.refl w.wg) (wg1Of_none hS hl (by simpa using hn)) rfl
      (by simp only [WgState.Fits, setAll_nil, freeAll_nil]; rw [← (hpush.kinds .texture (by simp [WgData.kind])).isOne]; exact hone)
      (fun _ h => by cases h) (fun _ h => by cases h) (fun _ h => by cases h) (by simp [wgKills])
  · cases h

theorem row_encoderFinish {e : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [e] bits)
    (h : wgRow S .encoderFinish [e] = some (S', av)) : WgOk .encoderFinish bits w S' av := by
  simp only [wgRow, wgR_encoderFinish, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨se, hse, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, beq_iff_eq] at hc
    obtain ⟨rfl, hn⟩ := hc
    cases h
    obtain ⟨i, d, hgd, -, hdi, hkd, hone, hf⟩ := wg1Arg_holds hS hl hx hse
    obtain ⟨cmds, rfl⟩ := fits_encoder hf
    have hset := WgStep.set (s := w.wg) (i := i) (d := .encoder cmds false) (k := .commandEncoder) (kind_of_one hone) rfl
    have hone' : (w.wg.set i (.encoder cmds false)).isOne .commandEncoder i = true := by
      rw [← (hset.kinds .commandEncoder (fun h => nomatch h)).isOne]; exact hone
    have hfree := (WgStep.free (kind_of_one hone')).forgetIdx (kind_of_one hone')
    have hst := hset.trans hfree
    have hnone := ((hst.kinds .commandBuffer (by simp)).noneOf).mp (wg1Of_none hS hl hn)
    exact wgMakeOne_ok (f := .encoderFinish) (d := .cmdbuf cmds false) (ch := [(i, .encoder cmds false)]) (fr := [i])
      (k := .commandBuffer) (sh := .cmdbuf) (N := [.wg1 .commandEncoder none]) hS hl
      (by simp [wgDecode, wgDFinish, hgd]) (by decide) hst hnone rfl (by simp [WgState.Fits])
      (TState.holds_cons (.inr <| noneOf_free hone') (fun _ h => by cases h))
      (wgKeeps_mem (by decide)) (fun x hx => by simp only [List.mem_singleton] at hx; exact ⟨_, _, hx⟩)
      (by simp [wgKills])
  · cases h

end


section
variable {S S' : TState} {w : World} {Γ : Env} {bits : List UInt64} {av : AV}

theorem shape_kind {s : WgState} {d : WgData} {sh : WgShape} {k : WgObj} (hf : s.Fits d sh) (hk : d.kind = k) :
    sh.kind = k := hf.kind ▸ hk

/-- The one object of kind `k` set to data `d` keeps every object of another
    kind `k'` the one of its kind where it was. -/
theorem isOne_set_other {s : WgState} {i : Nat} {d : WgData} {k : WgObj} (hone : s.isOne k i = true)
    (hd : d.kind = k) (k' : WgObj) (j : Nat) : (s.set i d).isOne k' j = s.isOne k' j :=
  ((WgStep.set (kind_of_one hone) hd).kinds k' (fun h => nomatch h)).isOne j |>.symm

theorem row_computePassSetPipeline {p q : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [p, q] bits)
    (h : wgRow S .computePassSetPipeline [p, q] = some (S', av)) : WgOk .computePassSetPipeline bits w S' av := by
  simp only [wgRow, wgR_computePassSetPipeline, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨s0, hs0, ⟨qi, qs⟩, hq, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨y, b, rfl, hy, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  obtain ⟨i, d, hgd, -, hdi, hkd, hone, hf⟩ := wg1Arg_holds hS hl hx hs0
  have hks := shape_kind hf hkd
  cases s0 <;> simp only [WgShape.kind, reduceCtorEq] at hks
  rename_i q0 bb
  obtain ⟨dq, hgq, -, hfq, hkq⟩ := wgoArg_holds hS hl hy hq
  cases qs <;> simp only [WgShape.kind, reduceCtorEq] at hkq
  rename_i gs v
  cases v
  · simp at h
  simp only [Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  obtain ⟨e, rfl, hone_e⟩ := fits_cpass hf
  obtain ⟨src, rfl⟩ := fits_cpipe hfq
  have hst := WgStep.set (s := w.wg) (i := i) (d := .cpass e (some qi) bb false) (kind_of_one hone) rfl
  exact wgUpdate_ok (f := .computePassSetPipeline) (ch := [(i, .cpass e (some qi) bb false)]) (fr := [])
    (N := [.wg1 .computePass (some (.cpass (some qi) bb))]) hS hl
    (by simp [wgDecode, wgDCpassSetPipe, hgd, hgq]) (by decide) hst
    (TState.holds_cons (wg1_set hone hdi rfl (by simp only [WgState.Fits, and_true, true_and]; rw [isOne_set_other hone rfl]; exact hone_e)) (fun _ h => by cases h))
    (fun _ h => by cases h)

theorem row_computePassSetBindGroup {p idx g : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [p, idx, g] bits)
    (h : wgRow S .computePassSetBindGroup [p, idx, g] = some (S', av)) :
    WgOk .computePassSetBindGroup bits w S' av := by
  simp only [wgRow, wgR_computePassSetBindGroup, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨s0, hs0, ⟨gi, gs⟩, hg, k, hk, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨y, b, rfl, hy, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨z, b, rfl, hz, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  obtain rfl := known_bits hy hk
  obtain ⟨i, d, hgd, -, hdi, hkd, hone, hf⟩ := wg1Arg_holds hS hl hx hs0
  have hks := shape_kind hf hkd
  cases s0 <;> simp only [WgShape.kind, reduceCtorEq] at hks
  rename_i q0 bb
  obtain ⟨dg, hgg, -⟩ := wgoArg_holds hS hl hz hg
  simp only [Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  obtain ⟨e, rfl, hone_e⟩ := fits_cpass hf
  have hst := WgStep.set (s := w.wg) (i := i)
    (d := .cpass e q0 ((y.toUInt32.toNat, gi) :: bb.filter (·.1 != y.toUInt32.toNat)) false) (kind_of_one hone) rfl
  exact wgUpdate_ok (f := .computePassSetBindGroup)
    (ch := [(i, .cpass e q0 ((y.toUInt32.toNat, gi) :: bb.filter (·.1 != y.toUInt32.toNat)) false)]) (fr := [])
    (N := [.wg1 .computePass (some (.cpass q0 ((y.toUInt32.toNat, gi) :: bb.filter (·.1 != y.toUInt32.toNat))))]) hS hl
    (by simp [wgDecode, wgDCpassSetGroup, hgd, hgg]) (by decide) hst
    (TState.holds_cons (wg1_set hone hdi rfl (by simp only [WgState.Fits, and_true, true_and]; rw [isOne_set_other hone rfl]; exact hone_e)) (fun _ h => by cases h))
    (fun _ h => by cases h)

theorem row_renderPassSetPipeline {p q : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [p, q] bits)
    (h : wgRow S .renderPassSetPipeline [p, q] = some (S', av)) : WgOk .renderPassSetPipeline bits w S' av := by
  simp only [wgRow, wgR_renderPassSetPipeline, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨s0, hs0, ⟨qi, qs⟩, hq, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨y, b, rfl, hy, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  obtain ⟨i, d, hgd, -, hdi, hkd, hone, hf⟩ := wg1Arg_holds hS hl hx hs0
  have hks := shape_kind hf hkd
  cases s0 <;> simp only [WgShape.kind, reduceCtorEq] at hks
  rename_i q0 g0
  obtain ⟨dq, hgq, -, hfq, hkq⟩ := wgoArg_holds hS hl hy hq
  cases qs <;> simp only [WgShape.kind, reduceCtorEq] at hkq
  rename_i v
  cases v
  · simp at h
  simp only [Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  obtain ⟨e, vv, rfl, hone_e, hone_v⟩ := fits_rpass hf
  obtain rfl := fits_eq_rpipe hfq
  have hst := WgStep.set (s := w.wg) (i := i) (d := .rpass e vv (some qi) g0 false) (kind_of_one hone) rfl
  exact wgUpdate_ok (f := .renderPassSetPipeline) (ch := [(i, .rpass e vv (some qi) g0 false)]) (fr := [])
    (N := [.wg1 .renderPass (some (.rpass (some qi) g0))]) hS hl
    (by simp [wgDecode, wgDRpassSetPipe, hgd, hgq]) (by decide) hst
    (TState.holds_cons (wg1_set hone hdi rfl (by simp only [WgState.Fits, and_true, true_and]; rw [isOne_set_other hone rfl, isOne_set_other hone rfl]; exact ⟨hone_e, hone_v⟩)) (fun _ h => by cases h))
    (fun _ h => by cases h)

theorem row_renderPassSetBindGroup {p idx g : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [p, idx, g] bits)
    (h : wgRow S .renderPassSetBindGroup [p, idx, g] = some (S', av)) :
    WgOk .renderPassSetBindGroup bits w S' av := by
  simp only [wgRow, wgR_renderPassSetBindGroup, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨s0, hs0, ⟨gi, gs⟩, hg, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨y, b, rfl, hy, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨z, b, rfl, hz, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  obtain ⟨i, d, hgd, -, hdi, hkd, hone, hf⟩ := wg1Arg_holds hS hl hx hs0
  have hks := shape_kind hf hkd
  cases s0 <;> simp only [WgShape.kind, reduceCtorEq] at hks
  rename_i q0 g0
  obtain ⟨dg, hgg, -⟩ := wgoArg_holds hS hl hz hg
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨k, hk, h⟩ := h
  obtain rfl := known_bits hy hk
  split at h
  · rename_i h0
    simp only [beq_iff_eq] at h0
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    obtain ⟨e, vv, rfl, hone_e, hone_v⟩ := fits_rpass hf
    have hst := WgStep.set (s := w.wg) (i := i) (d := .rpass e vv q0 (some gi) false) (kind_of_one hone) rfl
    exact wgUpdate_ok (f := .renderPassSetBindGroup) (ch := [(i, .rpass e vv q0 (some gi) false)]) (fr := [])
      (N := [.wg1 .renderPass (some (.rpass q0 (some gi)))]) hS hl
      (by simp [wgDecode, wgDRpassSetGroup, hgd, hgg, h0]) (by decide) hst
      (TState.holds_cons (wg1_set hone hdi rfl (by simp only [WgState.Fits, and_true, true_and]; rw [isOne_set_other hone rfl, isOne_set_other hone rfl]; exact ⟨hone_e, hone_v⟩)) (fun _ h => by cases h))
      (fun _ h => by cases h)
  · cases h

/-- Ending the one pass of kind `kp`, on the one encoder: the encoder open
    again, the pass used up. -/
theorem passEnd_ok {S : TState} {w : World} {f : WgpuFn} {bits : List UInt64} {kp : WgObj} {pi e : Nat}
    {cmds : List WgCmd} (hS : S.holds w) (hl : w.wg.lost = false) (hdec : wgDecode f bits w = some (.update [(e, .encoder cmds true)] [pi]))
    (hf : f ≠ .bufferRead) (honeP : w.wg.isOne kp pi = true) (honeE : w.wg.isOne .commandEncoder e = true)
    (hde : w.wg.objs[e]? = some (true, .encoder cmds false)) (hne : kp ≠ .commandEncoder)
    (hkill : kp ∈ wgKills f) :
    WgOk f bits w (.wg1 kp none :: .wg1 .commandEncoder (some (.encoder true)) ::
      S.filter (wgKeeps [] [.commandEncoder, kp] [kp] (some 0))) wgNone := by
  have hset := WgStep.set (s := w.wg) (i := e) (d := .encoder cmds true) (kind_of_one honeE) rfl
  have honeP' : (w.wg.set e (.encoder cmds true)).isOne kp pi = true := by
    rw [isOne_set_other honeE rfl]; exact honeP
  have hfree := WgStep.free (kind_of_one honeP')
  have hst := hset.trans (hfree.forgetIdx (kind_of_one honeP'))
  refine wgUpdate_ok (N := [.wg1 kp none, .wg1 .commandEncoder (some (.encoder true))]) hS hl hdec hf hst ?_
    (by intro k hk; simp only [List.nil_append, List.mem_singleton] at hk; subst hk; exact hkill)
  refine TState.holds_cons (.inr <| noneOf_free honeP') (TState.holds_cons ?_ (fun _ h => by cases h))
  have h1 := wg1_set (w := w) (d := .encoder cmds true) (sh := .encoder true) honeE hde rfl (by simp [WgState.Fits])
  exact wgKeeps_sound (w := { w with wg := w.wg.set e (.encoder cmds true) }) (g := some 0) hfree
    (fun m hm => by cases hm; rfl) h1 (by simp [wgKeeps, hne, WgShape.reads])

theorem row_computePassEnd {p : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [p] bits)
    (h : wgRow S .computePassEnd [p] = some (S', av)) : WgOk .computePassEnd bits w S' av := by
  simp only [wgRow, wgR_computePassEnd, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨s0, hs0, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  obtain ⟨i, d, hgd, -, hdi, hkd, hone, hf⟩ := wg1Arg_holds hS hl hx hs0
  have hks := shape_kind hf hkd
  cases s0 <;> simp only [WgShape.kind, reduceCtorEq] at hks
  rename_i q0 bb
  obtain ⟨e, rfl, hone_e⟩ := fits_cpass hf
  split at h
  · rename_i hw
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    obtain ⟨⟨de, hde, hke⟩, -⟩ := WgState.isOne_iff.mp hone_e
    obtain ⟨-, hfe⟩ := wg1Of_some hS hl hw hde hke
    obtain ⟨cmds, rfl⟩ := fits_encoder hfe
    exact passEnd_ok hS hl (by simp [wgDecode, wgDCpassEnd, hgd, WgState.data?, hde]) (by decide) hone hone_e hde
      (by decide) (by simp [wgKills])
  · cases h

theorem row_renderPassEnd {p : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [p] bits)
    (h : wgRow S .renderPassEnd [p] = some (S', av)) : WgOk .renderPassEnd bits w S' av := by
  simp only [wgRow, wgR_renderPassEnd, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨s0, hs0, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  obtain ⟨i, d, hgd, -, hdi, hkd, hone, hf⟩ := wg1Arg_holds hS hl hx hs0
  have hks := shape_kind hf hkd
  cases s0 <;> simp only [WgShape.kind, reduceCtorEq] at hks
  rename_i q0 g0
  obtain ⟨e, vv, rfl, hone_e, -⟩ := fits_rpass hf
  split at h
  · rename_i hw
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    obtain ⟨⟨de, hde, hke⟩, -⟩ := WgState.isOne_iff.mp hone_e
    obtain ⟨-, hfe⟩ := wg1Of_some hS hl hw hde hke
    obtain ⟨cmds, rfl⟩ := fits_encoder hfe
    exact passEnd_ok hS hl (by simp [wgDecode, wgDRpassEnd, hgd, WgState.data?, hde]) (by decide) hone hone_e hde
      (by decide) (by simp [wgKills])
  · cases h

end


section
variable {S S' : TState} {w : World} {Γ : Env} {bits : List UInt64} {av : AV}

/-- Every group the pipeline names is bound as the typestate says, so as wgpu
    reads it. -/
theorem wgBound_holds {S : TState} {w : World} (hS : S.holds w) (hl : w.wg.lost = false) {gs : List (List (Nat × Bool))}
    {b : List (Nat × Nat)} (h : wgBound S gs b = true) :
    (gs.zipIdx.all fun (ls, k) =>
      match (b.lookup k).bind w.wg.data? with
      | some (.bgroup es) => es.length == ls.length && ls.all (fun (bn, _) => es.any (·.1 == bn))
      | _ => false) = true := by
  unfold wgBound at h
  rw [List.all_eq_true] at h ⊢
  intro p hp
  have := h p hp
  obtain ⟨ls, k⟩ := p
  dsimp only at this ⊢
  cases hlk : b.lookup k with
  | none => simp [hlk] at this
  | some gi =>
      simp only [hlk, Option.bind_some] at this ⊢
      split at this
      · rename_i es hw
        obtain ⟨d, hd, hf⟩ := wgoOf_holds hS hl hw
        have : d = .bgroup es := by cases d <;> first | exact hf.elim | (cases hf; rfl)
        subst this
        simp only [WgState.data?, hd, Option.map_some]
        exact this
      · cases this

/-- A value inside a range the checker knows. -/
theorem num_holds {Γ : Env} {w : World} {a : AV} {x : UInt64} {n : Num} (hd : ∃ t, a.Desc Γ w (.sc t x))
    (h : a.num = some n) : n.lo ≤ x.toNat ∧ x.toNat ≤ n.hi ∧ n.al ∣ x.toNat := by
  obtain ⟨t, hd⟩ := hd
  obtain ⟨x', e, h1, h2, h3, -⟩ := AV.num_desc hd h
  simp only [V.sc.injEq] at e
  obtain ⟨-, rfl⟩ := e
  exact ⟨h1, h2, h3⟩

theorem mod4_of_al {x al : Nat} (hd : al ∣ x) (ha : al % 4 = 0) : x % 4 = 0 := by
  obtain ⟨k, rfl⟩ := hd
  obtain ⟨j, hj⟩ : 4 ∣ al := Nat.dvd_of_mod_eq_zero ha
  subst hj
  simp [Nat.mul_assoc, Nat.mul_mod_right]

theorem or_tail2 {A f g f' g' : Bool} (hc : ¬ (A || f' || g') = true) (hf : f = false) (hg : g = false) :
    ¬ (A || f || g) = true := by
  subst hf hg; cases A <;> simp_all

theorem or_tail3 {A f g h f' g' h' : Bool} (hc : ¬ (A || f' || g' || h') = true) (hf : f = false) (hg : g = false)
    (hh : h = false) : ¬ (A || f || g || h) = true := by
  subst hf hg hh; cases A <;> simp_all

/-- What a copy of at most `m` bytes keeps, one of `n ≤ m` keeps. -/
theorem afterCopy_mono {S : TState} {a : Option UInt64} {n m : Nat} (hn : n ≤ m) :
    ∀ x ∈ afterCopy a m S, x ∈ afterCopy a n S := by
  intro x hx
  unfold afterCopy at hx ⊢
  split at hx
  · obtain ⟨hm, hk⟩ := List.mem_filter.mp hx
    refine List.mem_filter.mpr ⟨hm, ?_⟩
    simp only [copyKeeps, Bool.and_eq_true] at hk ⊢
    exact ⟨Contracts.storeKeeps_mono hk.1 hn, hk.2⟩
  · exact hx

theorem toU32_small {x : UInt64} (h : x.toNat ≤ 65535) : ¬ (x.toUInt32.toUInt64 > 65535) := by
  simp only [GT.gt, UInt64.lt_iff_toNat_lt, UInt64.toNat_ofNat, UInt32.toNat_toUInt64, UInt64.toNat_toUInt32]
  omega

theorem row_computePassDispatch {p gx gy gz : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [p, gx, gy, gz] bits)
    (h : wgRow S .computePassDispatch [p, gx, gy, gz] = some (S', av)) : WgOk .computePassDispatch bits w S' av := by
  simp only [wgRow, wgR_computePassDispatch, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨s0, hs0, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨y1, b, rfl, hy1, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨y2, b, rfl, hy2, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨y3, b, rfl, hy3, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  obtain ⟨i, d, hgd, -, hdi, hkd, hone, hf⟩ := wg1Arg_holds hS hl hx hs0
  have hks := shape_kind hf hkd
  cases s0 <;> simp only [WgShape.kind, reduceCtorEq] at hks
  rename_i q0 bb
  obtain ⟨e, rfl, hone_e⟩ := fits_cpass hf
  cases q0 with
  | none => simp at h
  | some q =>
  simp only at h
  split at h
  · rename_i gs hq hw
    simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
    obtain ⟨n1, hn1, n2, hn2, n3, hn3, h⟩ := h
    split at h
    · rename_i hc
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      simp only [Bool.and_eq_true] at hc
      obtain ⟨hbd, hall⟩ := hc
      have h1 : n1.hi ≤ 65535 := of_decide_eq_true (List.all_eq_true.mp hall n1 (by simp))
      have h2 : n2.hi ≤ 65535 := of_decide_eq_true (List.all_eq_true.mp hall n2 (by simp))
      have h3 : n3.hi ≤ 65535 := of_decide_eq_true (List.all_eq_true.mp hall n3 (by simp))
      have hxyz := And.intro (toU32_small (Nat.le_trans (num_holds hy1 hn1).2.1 h1))
        (And.intro (toU32_small (Nat.le_trans (num_holds hy2 hn2).2.1 h2)) (toU32_small (Nat.le_trans (num_holds hy3 hn3).2.1 h3)))
      obtain ⟨dq, hdq, hfq⟩ := wgoOf_holds hS hl hq
      obtain ⟨src, rfl⟩ := fits_cpipe hfq
      obtain ⟨⟨de, hde, hke⟩, -⟩ := WgState.isOne_iff.mp hone_e
      obtain ⟨-, hfe⟩ := wg1Of_some hS hl hw hde hke
      obtain ⟨cmds, rfl⟩ := fits_encoder hfe
      have hbound := wgBound_holds hS hl hbd
      have hst : WgStep w.wg ((w.wg.setAll [(e, .encoder (cmds ++ [.dispatch q bb
          [y1.toUInt32.toUInt64, y2.toUInt32.toUInt64, y3.toUInt32.toUInt64]]) false)]).freeAll []) [] [] [] 0 :=
        WgStep.setSame (fun l d0 h0 => by rw [hde] at h0; cases h0; exact ⟨rfl, fun s sh => by cases sh <;> simp [WgState.Fits]⟩)
      exact wgUpdate_ok (f := .computePassDispatch)
        (ch := [(e, .encoder (cmds ++ [.dispatch q bb [y1.toUInt32.toUInt64, y2.toUInt32.toUInt64, y3.toUInt32.toUInt64]]) false)])
        (fr := []) (N := []) hS hl
        (by simp [wgDecode, wgDCpassDispatch, hgd, WgState.data?, hdq, hde]
            refine ⟨by simpa [List.all_eq_true] using hbound, ?_⟩
            simp only [GT.gt] at hxyz
            simpa using hxyz) (by decide) hst
        (fun _ h => by cases h) (fun _ h => by cases h)
    · cases h
  · cases h

theorem row_encoderCopyBufferToBuffer {e src so dst dof n : AV} (hS : S.holds w) (hl : w.wg.lost = false)
    (hb : ArgBits Γ w [e, src, so, dst, dof, n] bits)
    (h : wgRow S .encoderCopyBufferToBuffer [e, src, so, dst, dof, n] = some (S', av)) :
    WgOk .encoderCopyBufferToBuffer bits w S' av := by
  simp only [wgRow, wgR_encoderCopyBufferToBuffer, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨s0, hs0, ⟨si, ss⟩, hsrc, ⟨di, ds⟩, hdst, k1, hk1, k2, hk2, nn, hnn, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨x1, b, rfl, hx1, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨x2, b, rfl, hx2, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨x3, b, rfl, hx3, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨x4, b, rfl, hx4, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨x5, b, rfl, hx5, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  obtain rfl := known_bits hx2 hk1
  obtain rfl := known_bits hx4 hk2
  obtain ⟨-, hnh, hna⟩ := num_holds hx5 hnn
  obtain ⟨i, d, hgd, -, hdi, hkd, hone, hf⟩ := wg1Arg_holds hS hl hx hs0
  have hks := shape_kind hf hkd
  cases s0 <;> simp only [WgShape.kind, reduceCtorEq] at hks
  rename_i o
  cases o
  · simp at h
  obtain ⟨cmds, rfl⟩ := fits_encoder hf
  obtain ⟨dsrc, hgs, -, hfs, hks'⟩ := wgoArg_holds hS hl hx1 hsrc
  obtain ⟨ddst, hgdst, -, hfd, hkd'⟩ := wgoArg_holds hS hl hx3 hdst
  cases ss <;> simp only [WgShape.kind, reduceCtorEq] at hks'
  cases ds <;> simp only [WgShape.kind, reduceCtorEq] at hkd'
  rename_i su sn du dn
  obtain ⟨sb, rfl, rfl⟩ := fits_buffer hfs
  obtain ⟨db, rfl, rfl⟩ := fits_buffer hfd
  simp only at h
  split at h
  · cases h
  · rename_i hc
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    have hst : WgStep w.wg ((w.wg.setAll [(i, .encoder (cmds ++ [.copy si x2.toNat di x4.toNat x5.toNat]) true)]).freeAll [])
        [] [] [] 0 := WgStep.setSame (fun l d0 h0 => by rw [hdi] at h0; cases h0; exact ⟨rfl, fun s sh => by cases sh <;> simp [WgState.Fits]⟩)
    exact wgUpdate_ok (f := .encoderCopyBufferToBuffer)
      (ch := [(i, .encoder (cmds ++ [.copy si x2.toNat di x4.toNat x5.toNat]) true)]) (fr := []) (N := []) hS hl
      (by
        simp only [wgDecode, wgDCopy, hgd, hgs, hgdst, Option.bind_eq_bind, Option.bind_some]
        have hc' := hc
        simp only [Bool.or_eq_true, not_or, bne_iff_ne, ne_eq, Decidable.not_not, decide_eq_true_eq, Nat.not_lt]
          at hc'
        rw [if_neg (or_tail3 hc (by simp [mod4_of_al hna hc'.1.1.2]) (by simp; omega) (by simp; omega))])
      (by decide) hst
      (fun _ h => by cases h) (fun _ h => by cases h)

theorem row_renderPassDraw {p a1 a2 a3 a4 : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [p, a1, a2, a3, a4] bits)
    (h : wgRow S .renderPassDraw [p, a1, a2, a3, a4] = some (S', av)) : WgOk .renderPassDraw bits w S' av := by
  simp only [wgRow, wgR_renderPassDraw, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨s0, hs0, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨x1, b, rfl, -, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨x2, b, rfl, -, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨x3, b, rfl, -, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨x4, b, rfl, -, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  obtain ⟨i, d, hgd, -, hdi, hkd, hone, hf⟩ := wg1Arg_holds hS hl hx hs0
  have hks := shape_kind hf hkd
  cases s0 <;> simp only [WgShape.kind, reduceCtorEq] at hks
  rename_i q0 g0
  obtain ⟨e, vv, rfl, hone_e, -⟩ := fits_rpass hf
  cases q0 with
  | none => simp at h
  | some q =>
  split at h
  · rename_i hw
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    obtain ⟨⟨de, hde, hke⟩, -⟩ := WgState.isOne_iff.mp hone_e
    obtain ⟨-, hfe⟩ := wg1Of_some hS hl hw hde hke
    obtain ⟨cmds, rfl⟩ := fits_encoder hfe
    have hst : WgStep w.wg ((w.wg.setAll [(e, .encoder (cmds ++ [.draw vv g0]) false)]).freeAll []) [] [] [] 0 :=
      WgStep.setSame (fun l d0 h0 => by rw [hde] at h0; cases h0; exact ⟨rfl, fun s sh => by cases sh <;> simp [WgState.Fits]⟩)
    exact wgUpdate_ok (f := .renderPassDraw) (ch := [(e, .encoder (cmds ++ [.draw vv g0]) false)]) (fr := [])
      (N := []) hS hl (by simp [wgDecode, wgDRpassDraw, hgd, WgState.data?, hde]) (by decide) hst
      (fun _ h => by cases h) (fun _ h => by cases h)
  · cases h

theorem row_surfaceConfigure {sh dv fmt wd ht : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [sh, dv, fmt, wd, ht] bits)
    (h : wgRow S .surfaceConfigure [sh, dv, fmt, wd, ht] = some (S', av)) : WgOk .surfaceConfigure bits w S' av := by
  simp only [wgRow, wgR_surfaceConfigure, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨⟨si, ss⟩, hs, ⟨ii, sc⟩, hi, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨y, b, rfl, hy, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨x1, b, rfl, hx1, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨x2, b, rfl, hx2, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨x3, b, rfl, hx3, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  obtain ⟨d, hgd, hdi, hf, hks⟩ := wgoArg_holds hS hl hx hs
  obtain ⟨hg, -⟩ := wgDev_holds hS hl hy hi
  cases ss <;> simp only [WgShape.kind, reduceCtorEq] at hks
  rename_i c
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨wn, hwn, hn', hhn, f1, hf1, h⟩ := h
  obtain rfl := known_bits hx1 hf1
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hc
    obtain ⟨⟨⟨⟨⟨hn, hfmt⟩, hw1⟩, hw2⟩, hh1⟩, hh2⟩ := hc
    obtain ⟨hwl, hwh, -⟩ := num_holds hx2 hwn
    obtain ⟨hhl, hhh, -⟩ := num_holds hx3 hhn
    have hw : 0 < asI32 x2 := by rw [Contracts.asI32_small (by omega)]; omega
    have hh : 0 < asI32 x3 := by rw [Contracts.asI32_small (by omega)]; omega
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    have hnone := wg1Of_none hS hl hn
    cases d <;> simp only [WgState.Fits] at hf <;> try exact hf.elim
    rename_i c0 cur
    obtain ⟨rfl, hcur⟩ := hf
    have hcur' : cur = none := by
      cases cur with
      | none => rfl
      | some t =>
          obtain ⟨⟨dt, hdt, hkt⟩, -⟩ := WgState.isOne_iff.mp hcur
          exact absurd hkt (hnone t dt hdt)
    subst hcur'
    have hst := WgStep.set (s := w.wg) (i := si) (d := .surface true none) (k := .surface)
      (fun l d0 h0 => by rw [hdi] at h0; cases h0; rfl) rfl
    exact wgUpdate_ok (f := .surfaceConfigure) (ch := [(si, .surface true none)]) (fr := [])
      (N := [.wgo si (.surface true)]) hS hl
      (by simp [wgDecode, wgDConfigure, hgd, hg, hfmt, Int.not_le.mpr hw, Int.not_le.mpr hh]) (by decide) hst
      (TState.holds_cons (show (Fact.wgo si (.surface true)).holds { w with wg := w.wg.set si (.surface true none) } from
        .inr ⟨.surface true none, by rw [WgState.set_get, if_pos rfl, hdi]; rfl,
          rfl, ((hst.kinds .texture (fun h => nomatch h)).noneOf).mp hnone⟩) (fun _ h => by cases h))
      (fun _ h => by cases h)
  · cases h

theorem row_release {o : WgObj} {hh : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [hh] bits)
    (h : wgRow S (.release o) [hh] = some (S', av)) : WgOk (.release o) bits w S' av := by
  simp only [wgRow, wgR_release] at h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  cases hp : wgoArg S hh o with
  | some p =>
      obtain ⟨i, s0⟩ := p
      simp only [hp, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      obtain ⟨d, hgd, hdi, hf, hk⟩ := wgoArg_holds hS hl hx hp
      have hkd : d.kind = o := hf.kind.trans hk
      have hst := WgStep.free (s := w.wg) (i := i) (k := o) (fun l d0 h0 => by rw [hdi] at h0; cases h0; exact hkd)
      exact wgUpdate_ok (f := .release o) (ch := []) (fr := [i]) (N := []) hS hl (by simp [wgDecode, wgDRelease, hgd])
        (by intro e; cases e) hst (fun _ h => by cases h) (by simp [wgKills])
  | none =>
      cases hq : wg1Arg S hh o with
      | none => simp [hp, hq] at h
      | some s0 =>
          simp only [hp, hq, Option.some.injEq, Prod.mk.injEq] at h
          obtain ⟨rfl, rfl⟩ := h
          obtain ⟨i, d, hgd, -, hdi, hkd, hone, -⟩ := wg1Arg_holds hS hl hx hq
          have hst := (WgStep.free (s := w.wg) (i := i) (k := o) (kind_of_one hone)).forgetIdx (kind_of_one hone)
          exact wgUpdate_ok (f := .release o) (ch := []) (fr := [i]) (N := [.wg1 o none]) hS hl
            (by simp [wgDecode, wgDRelease, hgd]) (by intro e; cases e) hst
            (TState.holds_cons (.inr <| noneOf_free hone) (fun _ h => by cases h)) (by simp [wgKills])

end


section
variable {S S' : TState} {w : World} {Γ : Env} {bits : List UInt64} {av : AV}

/-- No fact reads what a window has been shown. -/
theorem holds_shown {x : Fact} {w : World} (h : x.holds w) (l : List ByteArray) : x.holds { w with shown := l } := by
  cases x with
  | part p b => cases p <;> exact h
  | held k a => cases k <;> exact h
  | opened k a => cases k <;> exact h
  | htAt _ _ _ => exact HtSpec.HtObj.of_mem h rfl rfl
  | wg1 _ o => cases o <;> exact h
  | _ => exact h

theorem row_queueSubmit {q cb : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [q, cb] bits)
    (h : wgRow S .queueSubmit [q, cb] = some (S', av)) : WgOk .queueSubmit bits w S' av := by
  simp only [wgRow, wgR_queueSubmit, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨⟨qi, qs⟩, hq, s0, hs0, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨y, b, rfl, hy, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  split at h
  · rename_i hc
    simp only [beq_iff_eq] at hc; subst hc
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    obtain ⟨dq, hgq, -⟩ := wgoArg_holds hS hl hx hq
    obtain ⟨ci, d, hgc, -, hdc, hkc, hone, hf⟩ := wg1Arg_holds hS hl hy hs0
    obtain ⟨cmds, rfl⟩ := fits_cmdbuf hf
    have hdec : wgDecode .queueSubmit [x, y] w = some (.submit ci) := by simp [wgDecode, wgDSubmit, hgq, hgc]
    have hrun : w.wg.run w.shader ci =
        ((cmds.foldl (runWgCmd w.shader) w.wg).set ci (.cmdbuf cmds true)).freeAll [ci] := by
      simp [WgState.run, WgState.data?, hdc]
    have h1 := WgStep.cmds w.shader w.wg cmds
    obtain ⟨p1, hp1⟩ : ∃ p, (cmds.foldl (runWgCmd w.shader) w.wg).objs[ci]? = some p :=
      ⟨_, Array.getElem?_eq_getElem (by rw [h1.size, Nat.add_zero]; exact (Array.getElem?_eq_some_iff.mp hdc).1)⟩
    have hk1 : ∀ l d0, (cmds.foldl (runWgCmd w.shader) w.wg).objs[ci]? = some (l, d0) → d0.kind = .commandBuffer :=
      fun l d0 h0 => (h1.kindKeep ci true l _ d0 hdc h0).symm.trans hkc
    have h2 := WgStep.set (d := .cmdbuf cmds true) hk1 rfl
    have hone2 : ((cmds.foldl (runWgCmd w.shader) w.wg).set ci (.cmdbuf cmds true)).isOne .commandBuffer ci = true := by
      rw [← ((h1.trans h2).kinds .commandBuffer (by simp)).isOne]; exact hone
    have h3 := (WgStep.free (kind_of_one hone2)).forgetIdx (kind_of_one hone2)
    have hst := ((h1.trans h2).trans h3).weaken (TK' := [.commandBuffer]) (K' := [.commandBuffer])
      (by simp) (by simp)
    rw [← hrun] at hst
    refine ⟨_, _, wgpuCall_of hl hdec (by decide), ?_, fun Γ v hv => by simp [wgEffect] at hv, wgKept hst (by simp [wgKills])⟩
    simp only [wgEffect]
    refine TState.holds_cons ?_ (wgKeeps_filter hS hst (fun m e => by cases e; rfl))
    refine .inr ?_
    show (w.wg.run w.shader ci).NoneOf .commandBuffer
    rw [hrun]; exact noneOf_free hone2
  · cases h

/-- A row's condition: the model's refusal false, and the memory it reads there. -/
theorem or_not_false {a b : Bool} (h : ¬((a || !b) = true)) : ¬(a = true) ∧ b = true := by
  cases a <;> cases b <;> simp_all

theorem row_queueWriteBuffer {q bf off src n : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [q, bf, off, src, n] bits)
    (h : wgRow S .queueWriteBuffer [q, bf, off, src, n] = some (S', av)) : WgOk .queueWriteBuffer bits w S' av := by
  simp only [wgRow, wgR_queueWriteBuffer, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨⟨qi, qs⟩, hq, ⟨bi, bs⟩, hbf, k1, hk1, nn, hnn, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨y, b, rfl, hy, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨x1, b, rfl, hx1, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨x2, b, rfl, hx2, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨x3, b, rfl, hx3, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  obtain rfl := known_bits hx1 hk1
  obtain ⟨-, hnh, hna⟩ := num_holds hx3 hnn
  obtain ⟨dq, hgq, -⟩ := wgoArg_holds hS hl hx hq
  obtain ⟨d, hgd, hdi, hf, hkb⟩ := wgoArg_holds hS hl hy hbf
  cases bs <;> simp only [WgShape.kind, reduceCtorEq] at hkb
  rename_i u sz
  obtain ⟨bb, rfl, rfl⟩ := fits_buffer hf
  simp only at h
  split at h
  · cases h
  · rename_i hc
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    obtain ⟨hc1, hm⟩ := or_not_false hc
    obtain ⟨r, o, hd, hp, hs, hn⟩ := memB_spec hS hx2 hx3 hm
    obtain ⟨bytes, hbytes⟩ := Option.isSome_iff_exists.mp (Contracts.copyOut_isSome hd hp hs hn)
    have hc' := hc1
    simp only [Bool.or_eq_true, not_or, bne_iff_ne, ne_eq, Decidable.not_not, decide_eq_true_eq, Nat.not_lt]
      at hc'
    have hdec : wgDecode .queueWriteBuffer [x, y, x1, x2, x3] w = some (.write bi x1.toNat bytes) := by
      simp only [wgDecode, wgDWriteBuffer, hgq, hgd, Option.bind_eq_bind, Option.bind_some]
      rw [if_neg (or_tail2 hc1 (by simp [mod4_of_al hna hc'.1.2]) (by simp; omega))]
      simp [readBytes, hbytes]
    have hst := WgStep.putBytes w.wg bi (overwrite (w.wg.bytes bi) x1.toNat bytes)
    refine ⟨_, _, wgpuCall_of hl hdec (by decide), ?_, fun Γ v hv => by simp [wgEffect] at hv, wgKept hst (by simp)⟩
    exact wgKeeps_filter hS hst (fun m e => by cases e; rfl)

theorem row_bufferRead {dv bf off dst n : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [dv, bf, off, dst, n] bits)
    (h : wgRow S .bufferRead [dv, bf, off, dst, n] = some (S', av)) : WgOk .bufferRead bits w S' av := by
  simp only [wgRow, wgR_bufferRead, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨⟨ii, sc⟩, hi, ⟨bi, bs⟩, hbf, k1, hk1, nn, hnn, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨y, b, rfl, hy, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨x1, b, rfl, hx1, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨x2, b, rfl, hx2, hb⟩ := ArgBits.cons_inv hb
  obtain ⟨x3, b, rfl, hx3, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  obtain rfl := known_bits hx1 hk1
  obtain ⟨hnl, hnh, hna⟩ := num_holds hx3 hnn
  obtain ⟨hg, -⟩ := wgDev_holds hS hl hx hi
  obtain ⟨d, hgd, hdi, hf, hkb⟩ := wgoArg_holds hS hl hy hbf
  cases bs <;> simp only [WgShape.kind, reduceCtorEq] at hkb
  rename_i u sz
  obtain ⟨bb, rfl, rfl⟩ := fits_buffer hf
  simp only at h
  split at h
  · cases h
  · rename_i hc
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    obtain ⟨hc1, hm⟩ := or_not_false hc
    obtain ⟨r, o, hd, hp, hs, hn⟩ := memB_spec hS hx2 hx3 hm
    have hc' := hc1
    simp only [Bool.or_eq_true, not_or, bne_iff_ne, ne_eq, Decidable.not_not, decide_eq_true_eq, Nat.not_lt,
      beq_iff_eq] at hc'
    have hz0 : (ByteArray.mk (Array.replicate x3.toNat 0)).size = x3.toNat := by simp [ByteArray.size]
    obtain ⟨m0, hm0⟩ := Option.isSome_iff_exists.mp
      (Contracts.copyIn_isSome (ByteArray.mk (Array.replicate x3.toNat 0)) hd hp (by rw [hz0]; exact hs)
        (by rw [hz0]; exact hn))
    have hdec : wgDecode .bufferRead [x, y, x1, x2, x3] w = some (.read bi x1.toNat x3.toNat x2) := by
      simp only [wgDecode, wgDRead, hg, hgd, Option.bind_eq_bind, Option.bind_some]
      have h0 : x3 ≠ 0 := by intro e; subst e; simp at hnl; omega
      rw [if_neg (or_tail3 hc1 (by simpa using h0) (by simp [mod4_of_al hna hc'.1.2]) (by simp; omega))]
      simp [hm0]
    have hfit : x1.toNat + x3.toNat ≤ bb.size := by omega
    have hsz : ((w.wg.bytes bi).extract x1.toNat (x1.toNat + x3.toNat)).size = x3.toNat := by
      simp only [WgState.bytes, WgState.data?, hdi, Option.map_some, ByteArray.size_extract]
      omega
    obtain ⟨m, hm'⟩ := Option.isSome_iff_exists.mp
      (Contracts.copyIn_isSome ((w.wg.bytes bi).extract x1.toNat (x1.toNat + x3.toNat)) hd hp (by rw [hsz]; exact hs)
        (by rw [hsz]; exact hn))
    have hcall : wgpuCall .bufferRead [x, y, x1, x2, x3] w = some (some (ofInt .i32 0), { w with mem := m }) := by
      simp [wgpuCall, hl, hdec, wgEffect, hgd, hm']
    obtain ⟨hS', hk⟩ := afterCopy_sound hS hm' (ao := dst.bits) (fun _ e => (known_bits hx2 e).symm) (by rw [hsz]; omega)
    rw [hsz] at hS'
    refine ⟨_, _, hcall, fun z hz => hS' z (afterCopy_mono hnh z hz), fun Γ v hv => by cases hv; rfl, hk.but _⟩

theorem row_surfaceGetCurrentTexture {sh : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [sh] bits)
    (h : wgRow S .surfaceGetCurrentTexture [sh] = some (S', av)) : WgOk .surfaceGetCurrentTexture bits w S' av := by
  simp only [wgRow, wgR_surfaceGetCurrentTexture, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨⟨si, ss⟩, hs, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, beq_iff_eq] at hc
    obtain ⟨rfl, hn⟩ := hc
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    have hnone := wg1Of_none hS hl hn
    obtain ⟨d, hgd, hdi, hf, -⟩ := wgoArg_holds hS hl hx hs
    cases d <;> simp only [WgState.Fits] at hf <;> try exact hf.elim
    rename_i c0 cur
    obtain ⟨rfl, hcur⟩ := hf
    have hcur' : cur = none := by
      cases cur with
      | none => rfl
      | some t =>
          obtain ⟨⟨dt, hdt, hkt⟩, -⟩ := WgState.isOne_iff.mp hcur
          exact absurd hkt (hnone t dt hdt)
    subst hcur'
    have hdec : wgDecode .surfaceGetCurrentTexture [x] w = some (.acquire si) := by
      simp [wgDecode, wgDAcquire, hgd]
    have hsi : si < w.wg.objs.size := (Array.getElem?_eq_some_iff.mp hdi).1
    refine ⟨_, _, wgpuCall_of hl hdec (by decide), ?_⟩
    simp only [wgEffect]
    split
    · rename_i hlt
      have hpush := WgStep.push w.wg (.texture si none)
      have hset := WgStep.set (s := (w.wg.push (.texture si none)).2) (i := si)
        (d := .surface true (some w.wg.objs.size)) (k := .surface)
        (fun l d0 h0 => by
          simp only [WgState.push] at h0; rw [Array.getElem?_push, if_neg (by omega), hdi] at h0; cases h0; rfl) rfl
      have hst := hpush.trans hset
      have hone : (w.wg.push (.texture si none)).2.isOne .texture w.wg.objs.size = true := isOne_push hnone (by rfl)
      have hone' := (hset.kinds .texture (fun h => nomatch h)).isOne w.wg.objs.size ▸ hone
      have hget : ((w.wg.push (.texture si none)).2.set si (.surface true (some w.wg.objs.size))).objs[w.wg.objs.size]? =
          some (true, .texture si none) := by
        rw [WgState.set_get, if_neg (by omega)]; simp [WgState.push]
      have hst' := hst.weaken (TK' := [.surface]) (K' := [.texture]) (by simp) (by simp [WgData.kind])
      refine ⟨TState.holds_cons ?_ (TState.holds_cons ?_ fun x hx => ?_), fun Γ v hv => ?_,
        wgKept hst' (by simp [wgKills])⟩
      · refine .inr fun j d hj hk => ?_
        obtain ⟨-, hu⟩ := WgState.isOne_iff.mp hone'
        obtain rfl := hu j ⟨d, hj, hk⟩
        rw [hget] at hj; cases hj
        exact ⟨hone', rfl⟩
      · exact .inr ⟨.surface true (some w.wg.objs.size), by
          rw [WgState.set_get, if_pos rfl]; simp [WgState.push, Array.getElem?_push, show si ≠ w.wg.objs.size by omega, hdi],
          rfl, hone'⟩
      · rcases List.mem_append.mp hx with hx | hx
        · exact wgBump_holds hS (by show (WgState.set _ _ _).objs.size = _; rw [hset.size, hpush.size]) x hx
        · exact wgKeeps_filter hS hst' (fun _ e => by cases e) x hx
      · cases hv
        refine hand_desc ?_
        simp only [Held.live, WgState.push, WgState.set, wgHandle_lt hlt, Bool.or_eq_true]
        exact .inr hone'
    · refine ⟨TState.holds_cons (.inl rfl) (TState.holds_cons (.inl rfl) fun x hx => ?_), fun Γ v hv => ?_,
        wgKept_lose _ _⟩
      · rcases List.mem_append.mp hx with hx | hx
        · exact wgBump_holds hS (by simp [WgState.lose]) x hx
        obtain ⟨hm, hkx⟩ := List.mem_filter.mp hx
        exact holds_lose (hS x hm) (fun m e => by subst e; simp [wgKeeps] at hkx) _
      · cases hv
        exact hand_desc (by simp [Held.live, WgState.lose])
  · cases h

theorem row_surfacePresent {t : AV} (hS : S.holds w) (hl : w.wg.lost = false) (hb : ArgBits Γ w [t] bits)
    (h : wgRow S .surfacePresent [t] = some (S', av)) : WgOk .surfacePresent bits w S' av := by
  simp only [wgRow, wgR_surfacePresent, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨s0, hs0, h⟩ := h
  obtain ⟨x, b, rfl, hx, hb⟩ := ArgBits.cons_inv hb
  obtain rfl := ArgBits.nil_inv hb
  obtain ⟨ti, d, hgd, -, hdi, hkd, hone, hf⟩ := wg1Arg_holds hS hl hx hs0
  have hks := shape_kind hf hkd
  cases s0 <;> simp only [WgShape.kind, reduceCtorEq] at hks
  rename_i si
  cases d <;> simp only [WgState.Fits] at hf <;> try exact hf.elim
  rename_i si' dr
  have e : si = si' := hf.symm
  subst e
  simp only at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, beq_iff_eq] at hc
    obtain ⟨hsurf, hnv⟩ := hc
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    obtain ⟨ds, hds, hfs⟩ := wgoOf_holds hS hl hsurf
    cases ds <;> simp only [WgState.Fits] at hfs <;> try exact hfs.elim
    rename_i c cur
    obtain ⟨rfl, hcur⟩ := hfs
    have hcur' : cur = some ti := by
      cases cur with
      | none => exact absurd rfl (hcur ti _ hdi)
      | some x0 =>
          obtain ⟨-, hu⟩ := WgState.isOne_iff.mp hcur
          rw [hu ti ⟨_, hdi, rfl⟩]
    subst hcur'
    have hdec : wgDecode .surfacePresent [x] w = some (.present si ti) := by
      simp [wgDecode, wgDPresent, hgd, WgState.data?, hds]
    have hset := WgStep.set (s := w.wg) (i := si) (d := .surface true none) (k := .surface)
      (fun l d0 h0 => by rw [hds] at h0; cases h0; rfl) rfl
    have hone2 : (w.wg.set si (.surface true none)).isOne .texture ti = true := by
      rw [← (hset.kinds .texture (fun h => nomatch h)).isOne]; exact hone
    have hfree := (WgStep.free (kind_of_one hone2)).forgetIdx (kind_of_one hone2)
    have hst := hset.trans hfree
    have hne : si ≠ ti := by rintro rfl; rw [hds] at hdi; cases hdi
    refine ⟨_, _, wgpuCall_of hl hdec (by decide), ?_, fun Γ v hv => by simp [wgEffect] at hv, ?_⟩
    · simp only [wgEffect]
      have hW : TState.holds (.wgo si (.surface true) :: .wg1 .texture none ::
          S.filter (wgKeeps [] [.surface, .texture] [.texture] (some 0)))
          { w with wg := (w.wg.set si (.surface true none)).freeAll [ti] } := by
        refine TState.holds_cons ?_ (TState.holds_cons (.inr <| noneOf_free hone2) (wgKeeps_filter hS hst (fun m e => by cases e; rfl)))
        refine .inr ⟨.surface true none, ?_, rfl, ?_⟩
        · rw [WgState.free_get, if_neg hne.symm, WgState.set_get, if_pos rfl, hds]; rfl
        · exact noneOf_free hone2
      exact fun y hy => holds_shown (hW y hy) _
    · have hk := wgKept hst (f := .surfacePresent) (by simp [wgKills])
      exact ⟨hk.sizes, fun k y hk' hl => by have := hk.held k y hk' hl; cases k <;> exact this, hk.blk, hk.obs⟩
  · cases h

end


-- ---------------------------------------------------------------------------
-- A lost device
-- ---------------------------------------------------------------------------

/-- A fact about one of wgpu's objects: what holds of a lost device. -/
def _root_.AlgorithmLib.HProg.Contracts.Fact.isWgObj : Fact → Bool
  | .wgo _ _ | .wg1 _ _ => true
  | _ => false

/-- Whether a call makes an object: it answers a handle. -/
def wgCreates (f : WgpuFn) : Bool := (Ext.wgpu f).sig.2 == some .i64

/-- **What every row leaves**, read off the row alone: facts about wgpu's
    objects, the typestate's own facts (a count of objects only where the call
    makes none), and a count one more where it makes one; and an answer of the
    kind the call gives. A lost device leaves every such typestate holding. -/
structure RowShape (f : WgpuFn) (S S' : TState) (av : AV) : Prop where
  facts : ∀ x ∈ S', (∃ i sh, x = .wgo i sh) ∨ (∃ k o, x = .wg1 k o) ∨
    (x ∈ S ∧ ∀ m, x = .nwg m → wgCreates f = false) ∨
    (∃ n, x = .nwg (n + 1) ∧ Fact.nwg n ∈ S ∧ wgCreates f = true)
  av : av = .any ∨ (∃ k, av = .hand (.wg k) true ∧ wgCreates f = true) ∨
    (∃ n, av = .known (.sc .i64 (wgHandle n)) ∧ Fact.nwg n ∈ S ∧ wgCreates f = true) ∨
    (av = .known (ofInt .i32 0) ∧ (Ext.wgpu f).sig.2 = some .i32)

theorem nwgOf_mem {S : TState} {n : Nat} (h : nwgOf S = some n) : Fact.nwg n ∈ S := by
  unfold nwgOf at h
  obtain ⟨x, hx, hk⟩ := List.exists_of_findSome?_eq_some h
  cases x <;> simp only [reduceCtorEq] at hk
  cases hk; exact hx

/-- **A row on a lost device**: every call does nothing, so what the row
    leaves holds. -/
theorem wgRow_lost {S S' : TState} {w : World} {f : WgpuFn} {bits : List UInt64} {av : AV}
    (hS : S.holds w) (hl : w.wg.lost = true) (hr : RowShape f S S' av) : WgOk f bits w S' av := by
  have hcall : wgpuCall f bits w = some (wgLostCall f w) := by simp [wgpuCall, hl]
  refine ⟨_, _, hcall, ?_⟩
  try unfold wgLostCall
  split
  · rename_i t hsig
    split
    · rename_i ht
      subst ht
      have hc : wgCreates f = true := by simp [wgCreates, hsig]
      have hsz : (w.wg.lose .instance).2.objs.size = w.wg.objs.size + 1 := by simp [WgState.lose]
      refine ⟨fun x hx => ?_, fun Γ v hv => ?_, wgKept_lose _ _⟩
      · rcases hr.facts x hx with ⟨i, sh, rfl⟩ | ⟨k, o, rfl⟩ | ⟨hm, hn⟩ | ⟨n, rfl, hm, -⟩
        · exact .inl rfl
        · cases o <;> exact .inl rfl
        · exact holds_lose (hS x hm) (fun m e => by rw [hn m e] at hc; cases hc) _
        · show (w.wg.lose .instance).2.objs.size = n + 1
          rw [hsz, (hS _ hm : w.wg.objs.size = n)]
      · cases hv
        rcases hr.av with rfl | ⟨k, rfl, -⟩ | ⟨n, rfl, hm, -⟩ | ⟨-, hs⟩
        · trivial
        · exact hand_desc (by simp [Held.live, WgState.lose])
        · show _ = _
          have hn : w.wg.objs.size = n := hS _ hm
          simp only [WgState.lose, hn]
        · rw [hsig] at hs; cases hs
    · rename_i ht
      have hc : wgCreates f = false := by simp [wgCreates, hsig, ht]
      refine ⟨fun x hx => ?_, fun Γ v hv => ?_, wgKept_same _ _⟩
      · rcases hr.facts x hx with ⟨i, sh, rfl⟩ | ⟨k, o, rfl⟩ | ⟨hm, -⟩ | ⟨n, -, -, hc'⟩
        · exact .inl hl
        · cases o <;> exact .inl hl
        · exact hS x hm
        · rw [hc] at hc'; cases hc'
      · cases hv
        rcases hr.av with rfl | ⟨k, -, hc'⟩ | ⟨n, -, -, hc'⟩ | ⟨rfl, hs⟩
        · trivial
        · rw [hc] at hc'; cases hc'
        · rw [hc] at hc'; cases hc'
        · rw [hsig] at hs; cases hs; rfl
  · rename_i hsig
    have hc : wgCreates f = false := by simp [wgCreates, hsig]
    refine ⟨fun x hx => ?_, fun Γ v hv => (by cases hv), wgKept_same _ _⟩
    rcases hr.facts x hx with ⟨i, sh, rfl⟩ | ⟨k, o, rfl⟩ | ⟨hm, -⟩ | ⟨n, -, -, hc'⟩
    · exact .inl hl
    · cases o <;> exact .inl hl
    · exact hS x hm
    · rw [hc] at hc'; cases hc'

/-- What `wgMake` leaves has the shape of a row. -/
theorem shape_make {S S' : TState} {av : AV} {f : WgpuFn} {sh : WgShape} (h : wgMake S sh = some (S', av))
    (hc : wgCreates f = true) : RowShape f S S' av := by
  unfold wgMake at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨n, hn, h⟩ := h
  split at h
  · cases h
    refine ⟨fun x hx => ?_, .inr (.inr (.inl ⟨n, rfl, nwgOf_mem hn, hc⟩))⟩
    simp only [List.mem_cons, List.mem_filter] at hx
    rcases hx with rfl | rfl | ⟨hm, hk⟩
    · exact .inr (.inr (.inr ⟨n, rfl, nwgOf_mem hn, hc⟩))
    · exact .inl ⟨_, _, rfl⟩
    · refine .inr (.inr (.inl ⟨hm, fun m e => ?_⟩))
      subst e; simp [wgKeeps] at hk
  · cases h

/-- New facts about wgpu's objects before a filter of the typestate have the
    shape of a row. -/
theorem shape_emit {S N : TState} {av : AV} {f : WgpuFn} {Ti : List Nat} {TK K : List WgObj} {g : Option Nat}
    (hN : N.all Fact.isWgObj = true)
    (hg : g = some 0 → wgCreates f = false)
    (hav : av = .any ∨ (∃ k, av = .hand (.wg k) true ∧ wgCreates f = true)) :
    RowShape f S (N ++ S.filter (wgKeeps Ti TK K g)) av := by
  refine ⟨fun x hx => ?_, hav.imp_right .inl⟩
  rcases List.mem_append.mp hx with hx | hx
  · have := List.all_eq_true.mp hN x hx
    cases x <;> simp only [Fact.isWgObj, Bool.false_eq_true] at this
    · exact .inl ⟨_, _, rfl⟩
    · exact .inr (.inl ⟨_, _, rfl⟩)
  · obtain ⟨hm, hk⟩ := List.mem_filter.mp hx
    refine .inr (.inr (.inl ⟨hm, fun m e => ?_⟩))
    subst e
    simp only [wgKeeps, beq_iff_eq] at hk
    exact hg hk

/-- What a row that makes one object afresh leaves has the shape of a row. -/
theorem shape_bump {S N : TState} {av : AV} {f : WgpuFn} {Ti : List Nat} {TK K : List WgObj}
    (hN : N.all Fact.isWgObj = true) (hc : wgCreates f = true)
    (hav : av = .any ∨ (∃ k, av = .hand (.wg k) true ∧ wgCreates f = true)) :
    RowShape f S (N ++ (wgBump S ++ S.filter (wgKeeps Ti TK K none))) av := by
  refine ⟨fun x hx => ?_, hav.imp_right .inl⟩
  rcases List.mem_append.mp hx with hx | hx
  · have := List.all_eq_true.mp hN x hx
    cases x <;> simp only [Fact.isWgObj, Bool.false_eq_true] at this
    · exact .inl ⟨_, _, rfl⟩
    · exact .inr (.inl ⟨_, _, rfl⟩)
  rcases List.mem_append.mp hx with hx | hx
  · unfold wgBump at hx
    split at hx
    · rename_i n hn
      simp only [List.mem_singleton] at hx
      exact .inr (.inr (.inr ⟨n, hx, nwgOf_mem hn, hc⟩))
    · cases hx
  · obtain ⟨hm, hk⟩ := List.mem_filter.mp hx
    refine .inr (.inr (.inl ⟨hm, fun m e => ?_⟩))
    subst e
    simp [wgKeeps] at hk

/-- What a buffer read leaves has the shape of a row. -/
theorem shape_copy {S : TState} {a : Option UInt64} {n : Nat} :
    RowShape .bufferRead S (afterCopy a n S) (.known (ofInt .i32 0)) := by
  refine ⟨fun x hx => .inr (.inr (.inl ⟨?_, fun _ _ => rfl⟩)), .inr (.inr (.inr ⟨rfl, rfl⟩))⟩
  unfold afterCopy at hx
  split at hx <;> exact (List.mem_filter.mp hx).1

set_option hygiene false in
/-- Peel a row's binds, matches and conditions down to what it answers, and
    show that has the shape of a row. -/
macro "wg_shape" : tactic => `(tactic| (
  repeat' (first
    | (cases h; done)
    | (simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨rfl, rfl⟩ := h)
    | (simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h)
    | (rcases h with ⟨_, h⟩)
    | split at h)
  all_goals first
    | exact shape_make h rfl
    | exact shape_copy
    | exact shape_emit (N := []) (by simp [Fact.isWgObj]) (by simp [wgCreates, Ext.sig])
        (by simp [wgCreates, wgNone, Ext.sig])
    | exact shape_emit (N := [_]) (by simp [Fact.isWgObj]) (by simp [wgCreates, Ext.sig])
        (by simp [wgCreates, wgNone, Ext.sig])
    | exact shape_emit (N := [_, _]) (by simp [Fact.isWgObj]) (by simp [wgCreates, Ext.sig])
        (by simp [wgCreates, wgNone, Ext.sig])
    | exact shape_bump (N := [_]) (by simp [Fact.isWgObj]) (by simp [wgCreates, Ext.sig])
        (by simp [wgCreates, wgNone, Ext.sig])
    | exact shape_bump (N := [_, _]) (by simp [Fact.isWgObj]) (by simp [wgCreates, Ext.sig])
        (by simp [wgCreates, wgNone, Ext.sig])
    | exact shape_bump (N := [_, _, _]) (by simp [Fact.isWgObj]) (by simp [wgCreates, Ext.sig])
        (by simp [wgCreates, wgNone, Ext.sig])))

/-- **Every row leaves the shape of a row.** -/
theorem wgRow_shape {S S' : TState} {f : WgpuFn} {as : List AV} {av : AV}
    (h : wgRow S f as = some (S', av)) : RowShape f S S' av := by
  cases f <;> simp only [wgRow] at h
  all_goals first
    | (cases h; done)
    | (simp only [wgR_createInstance, wgR_requestAdapter, wgR_requestDevice, wgR_deviceGetQueue,
        wgR_deviceCreateBuffer, wgR_deviceCreateShaderModule, wgR_deviceCreateBindGroupLayout,
        wgR_deviceCreatePipelineLayout, wgR_deviceCreateComputePipeline, wgR_deviceCreateBindGroup,
        wgR_deviceCreateCommandEncoder, wgR_deviceCreateRenderPipeline, wgR_renderPipelineGetBindGroupLayout,
        wgR_encoderBeginComputePass, wgR_computePassSetPipeline, wgR_computePassSetBindGroup,
        wgR_computePassDispatch, wgR_computePassEnd, wgR_encoderCopyBufferToBuffer, wgR_encoderFinish,
        wgR_encoderBeginRenderPass, wgR_renderPassSetPipeline, wgR_renderPassSetBindGroup, wgR_renderPassDraw,
        wgR_renderPassEnd, wgR_queueSubmit, wgR_queueWriteBuffer, wgR_bufferRead, wgR_instanceCreateSurface,
        wgR_surfaceConfigure, wgR_surfaceGetCurrentTexture, wgR_textureCreateView, wgR_surfacePresent,
        wgR_release, wgSame] at h
       wg_shape)


/-- **The wgpu rows are sound.** -/
theorem wgRow_sound {S S' : TState} {w : World} {Γ : Env} {f : WgpuFn} {as : List AV} {bits : List UInt64}
    {av : AV} (hS : S.holds w) (hb : ArgBits Γ w as bits) (h : wgRow S f as = some (S', av)) :
    WgOk f bits w S' av := by
  cases hl : w.wg.lost
  case true => exact wgRow_lost hS hl (wgRow_shape h)
  cases f with
  | createInstance =>
      match as, hb, h with
      | [], hb, h => exact row_createInstance hS hl hb h
      | _ :: _, _, h => simp [wgRow, wgR_createInstance] at h
  | requestAdapter =>
      match as, hb, h with
      | [a0], hb, h => exact row_requestAdapter hS hl hb h
      | [], _, h => simp [wgRow, wgR_requestAdapter] at h
      | _ :: _ :: _, _, h => simp [wgRow, wgR_requestAdapter] at h
  | requestDevice =>
      match as, hb, h with
      | [a0], hb, h => exact row_requestDevice hS hl hb h
      | [], _, h => simp [wgRow, wgR_requestDevice] at h
      | _ :: _ :: _, _, h => simp [wgRow, wgR_requestDevice] at h
  | deviceGetQueue =>
      match as, hb, h with
      | [a0], hb, h => exact row_deviceGetQueue hS hl hb h
      | [], _, h => simp [wgRow, wgR_deviceGetQueue] at h
      | _ :: _ :: _, _, h => simp [wgRow, wgR_deviceGetQueue] at h
  | deviceCreateBuffer =>
      match as, hb, h with
      | [a0, a1, a2], hb, h => exact row_deviceCreateBuffer hS hl hb h
      | [], _, h => simp [wgRow, wgR_deviceCreateBuffer] at h
      | [_], _, h => simp [wgRow, wgR_deviceCreateBuffer] at h
      | [_, _], _, h => simp [wgRow, wgR_deviceCreateBuffer] at h
      | _ :: _ :: _ :: _ :: _, _, h => simp [wgRow, wgR_deviceCreateBuffer] at h
  | deviceCreateShaderModule =>
      match as, hb, h with
      | [a0, a1, a2], hb, h => exact row_deviceCreateShaderModule hS hl hb h
      | [], _, h => simp [wgRow, wgR_deviceCreateShaderModule] at h
      | [_], _, h => simp [wgRow, wgR_deviceCreateShaderModule] at h
      | [_, _], _, h => simp [wgRow, wgR_deviceCreateShaderModule] at h
      | _ :: _ :: _ :: _ :: _, _, h => simp [wgRow, wgR_deviceCreateShaderModule] at h
  | deviceCreateBindGroupLayout =>
      match as, hb, h with
      | [a0, a1, a2], hb, h => exact row_deviceCreateBindGroupLayout hS hl hb h
      | [], _, h => simp [wgRow, wgR_deviceCreateBindGroupLayout] at h
      | [_], _, h => simp [wgRow, wgR_deviceCreateBindGroupLayout] at h
      | [_, _], _, h => simp [wgRow, wgR_deviceCreateBindGroupLayout] at h
      | _ :: _ :: _ :: _ :: _, _, h => simp [wgRow, wgR_deviceCreateBindGroupLayout] at h
  | deviceCreatePipelineLayout =>
      match as, hb, h with
      | [a0, a1, a2], hb, h => exact row_deviceCreatePipelineLayout hS hl hb h
      | [], _, h => simp [wgRow, wgR_deviceCreatePipelineLayout] at h
      | [_], _, h => simp [wgRow, wgR_deviceCreatePipelineLayout] at h
      | [_, _], _, h => simp [wgRow, wgR_deviceCreatePipelineLayout] at h
      | _ :: _ :: _ :: _ :: _, _, h => simp [wgRow, wgR_deviceCreatePipelineLayout] at h
  | deviceCreateComputePipeline =>
      match as, hb, h with
      | [a0, a1, a2, a3, a4], hb, h => exact row_deviceCreateComputePipeline hS hl hb h
      | [], _, h => simp [wgRow, wgR_deviceCreateComputePipeline] at h
      | [_], _, h => simp [wgRow, wgR_deviceCreateComputePipeline] at h
      | [_, _], _, h => simp [wgRow, wgR_deviceCreateComputePipeline] at h
      | [_, _, _], _, h => simp [wgRow, wgR_deviceCreateComputePipeline] at h
      | [_, _, _, _], _, h => simp [wgRow, wgR_deviceCreateComputePipeline] at h
      | _ :: _ :: _ :: _ :: _ :: _ :: _, _, h => simp [wgRow, wgR_deviceCreateComputePipeline] at h
  | deviceCreateBindGroup =>
      match as, hb, h with
      | [a0, a1, a2, a3], hb, h => exact row_deviceCreateBindGroup hS hl hb h
      | [], _, h => simp [wgRow, wgR_deviceCreateBindGroup] at h
      | [_], _, h => simp [wgRow, wgR_deviceCreateBindGroup] at h
      | [_, _], _, h => simp [wgRow, wgR_deviceCreateBindGroup] at h
      | [_, _, _], _, h => simp [wgRow, wgR_deviceCreateBindGroup] at h
      | _ :: _ :: _ :: _ :: _ :: _, _, h => simp [wgRow, wgR_deviceCreateBindGroup] at h
  | deviceCreateCommandEncoder =>
      match as, hb, h with
      | [a0], hb, h => exact row_deviceCreateCommandEncoder hS hl hb h
      | [], _, h => simp [wgRow, wgR_deviceCreateCommandEncoder] at h
      | _ :: _ :: _, _, h => simp [wgRow, wgR_deviceCreateCommandEncoder] at h
  | deviceCreateRenderPipeline =>
      match as, hb, h with
      | [a0, a1, a2, a3, a4, a5, a6], hb, h => exact row_deviceCreateRenderPipeline hS hl hb h
      | [], _, h => simp [wgRow, wgR_deviceCreateRenderPipeline] at h
      | [_], _, h => simp [wgRow, wgR_deviceCreateRenderPipeline] at h
      | [_, _], _, h => simp [wgRow, wgR_deviceCreateRenderPipeline] at h
      | [_, _, _], _, h => simp [wgRow, wgR_deviceCreateRenderPipeline] at h
      | [_, _, _, _], _, h => simp [wgRow, wgR_deviceCreateRenderPipeline] at h
      | [_, _, _, _, _], _, h => simp [wgRow, wgR_deviceCreateRenderPipeline] at h
      | [_, _, _, _, _, _], _, h => simp [wgRow, wgR_deviceCreateRenderPipeline] at h
      | _ :: _ :: _ :: _ :: _ :: _ :: _ :: _ :: _, _, h => simp [wgRow, wgR_deviceCreateRenderPipeline] at h
  | renderPipelineGetBindGroupLayout =>
      match as, hb, h with
      | [a0, a1], hb, h => exact row_renderPipelineGetBindGroupLayout hS hl hb h
      | [], _, h => simp [wgRow, wgR_renderPipelineGetBindGroupLayout] at h
      | [_], _, h => simp [wgRow, wgR_renderPipelineGetBindGroupLayout] at h
      | _ :: _ :: _ :: _, _, h => simp [wgRow, wgR_renderPipelineGetBindGroupLayout] at h
  | encoderBeginComputePass =>
      match as, hb, h with
      | [a0], hb, h => exact row_encoderBeginComputePass hS hl hb h
      | [], _, h => simp [wgRow, wgR_encoderBeginComputePass] at h
      | _ :: _ :: _, _, h => simp [wgRow, wgR_encoderBeginComputePass] at h
  | computePassSetPipeline =>
      match as, hb, h with
      | [a0, a1], hb, h => exact row_computePassSetPipeline hS hl hb h
      | [], _, h => simp [wgRow, wgR_computePassSetPipeline] at h
      | [_], _, h => simp [wgRow, wgR_computePassSetPipeline] at h
      | _ :: _ :: _ :: _, _, h => simp [wgRow, wgR_computePassSetPipeline] at h
  | computePassSetBindGroup =>
      match as, hb, h with
      | [a0, a1, a2], hb, h => exact row_computePassSetBindGroup hS hl hb h
      | [], _, h => simp [wgRow, wgR_computePassSetBindGroup] at h
      | [_], _, h => simp [wgRow, wgR_computePassSetBindGroup] at h
      | [_, _], _, h => simp [wgRow, wgR_computePassSetBindGroup] at h
      | _ :: _ :: _ :: _ :: _, _, h => simp [wgRow, wgR_computePassSetBindGroup] at h
  | computePassDispatch =>
      match as, hb, h with
      | [a0, a1, a2, a3], hb, h => exact row_computePassDispatch hS hl hb h
      | [], _, h => simp [wgRow, wgR_computePassDispatch] at h
      | [_], _, h => simp [wgRow, wgR_computePassDispatch] at h
      | [_, _], _, h => simp [wgRow, wgR_computePassDispatch] at h
      | [_, _, _], _, h => simp [wgRow, wgR_computePassDispatch] at h
      | _ :: _ :: _ :: _ :: _ :: _, _, h => simp [wgRow, wgR_computePassDispatch] at h
  | computePassEnd =>
      match as, hb, h with
      | [a0], hb, h => exact row_computePassEnd hS hl hb h
      | [], _, h => simp [wgRow, wgR_computePassEnd] at h
      | _ :: _ :: _, _, h => simp [wgRow, wgR_computePassEnd] at h
  | encoderCopyBufferToBuffer =>
      match as, hb, h with
      | [a0, a1, a2, a3, a4, a5], hb, h => exact row_encoderCopyBufferToBuffer hS hl hb h
      | [], _, h => simp [wgRow, wgR_encoderCopyBufferToBuffer] at h
      | [_], _, h => simp [wgRow, wgR_encoderCopyBufferToBuffer] at h
      | [_, _], _, h => simp [wgRow, wgR_encoderCopyBufferToBuffer] at h
      | [_, _, _], _, h => simp [wgRow, wgR_encoderCopyBufferToBuffer] at h
      | [_, _, _, _], _, h => simp [wgRow, wgR_encoderCopyBufferToBuffer] at h
      | [_, _, _, _, _], _, h => simp [wgRow, wgR_encoderCopyBufferToBuffer] at h
      | _ :: _ :: _ :: _ :: _ :: _ :: _ :: _, _, h => simp [wgRow, wgR_encoderCopyBufferToBuffer] at h
  | encoderFinish =>
      match as, hb, h with
      | [a0], hb, h => exact row_encoderFinish hS hl hb h
      | [], _, h => simp [wgRow, wgR_encoderFinish] at h
      | _ :: _ :: _, _, h => simp [wgRow, wgR_encoderFinish] at h
  | encoderBeginRenderPass =>
      match as, hb, h with
      | [a0, a1, a2], hb, h => exact row_encoderBeginRenderPass hS hl hb h
      | [], _, h => simp [wgRow, wgR_encoderBeginRenderPass] at h
      | [_], _, h => simp [wgRow, wgR_encoderBeginRenderPass] at h
      | [_, _], _, h => simp [wgRow, wgR_encoderBeginRenderPass] at h
      | _ :: _ :: _ :: _ :: _, _, h => simp [wgRow, wgR_encoderBeginRenderPass] at h
  | renderPassSetPipeline =>
      match as, hb, h with
      | [a0, a1], hb, h => exact row_renderPassSetPipeline hS hl hb h
      | [], _, h => simp [wgRow, wgR_renderPassSetPipeline] at h
      | [_], _, h => simp [wgRow, wgR_renderPassSetPipeline] at h
      | _ :: _ :: _ :: _, _, h => simp [wgRow, wgR_renderPassSetPipeline] at h
  | renderPassSetBindGroup =>
      match as, hb, h with
      | [a0, a1, a2], hb, h => exact row_renderPassSetBindGroup hS hl hb h
      | [], _, h => simp [wgRow, wgR_renderPassSetBindGroup] at h
      | [_], _, h => simp [wgRow, wgR_renderPassSetBindGroup] at h
      | [_, _], _, h => simp [wgRow, wgR_renderPassSetBindGroup] at h
      | _ :: _ :: _ :: _ :: _, _, h => simp [wgRow, wgR_renderPassSetBindGroup] at h
  | renderPassDraw =>
      match as, hb, h with
      | [a0, a1, a2, a3, a4], hb, h => exact row_renderPassDraw hS hl hb h
      | [], _, h => simp [wgRow, wgR_renderPassDraw] at h
      | [_], _, h => simp [wgRow, wgR_renderPassDraw] at h
      | [_, _], _, h => simp [wgRow, wgR_renderPassDraw] at h
      | [_, _, _], _, h => simp [wgRow, wgR_renderPassDraw] at h
      | [_, _, _, _], _, h => simp [wgRow, wgR_renderPassDraw] at h
      | _ :: _ :: _ :: _ :: _ :: _ :: _, _, h => simp [wgRow, wgR_renderPassDraw] at h
  | renderPassEnd =>
      match as, hb, h with
      | [a0], hb, h => exact row_renderPassEnd hS hl hb h
      | [], _, h => simp [wgRow, wgR_renderPassEnd] at h
      | _ :: _ :: _, _, h => simp [wgRow, wgR_renderPassEnd] at h
  | queueSubmit =>
      match as, hb, h with
      | [a0, a1], hb, h => exact row_queueSubmit hS hl hb h
      | [], _, h => simp [wgRow, wgR_queueSubmit] at h
      | [_], _, h => simp [wgRow, wgR_queueSubmit] at h
      | _ :: _ :: _ :: _, _, h => simp [wgRow, wgR_queueSubmit] at h
  | queueWriteBuffer =>
      match as, hb, h with
      | [a0, a1, a2, a3, a4], hb, h => exact row_queueWriteBuffer hS hl hb h
      | [], _, h => simp [wgRow, wgR_queueWriteBuffer] at h
      | [_], _, h => simp [wgRow, wgR_queueWriteBuffer] at h
      | [_, _], _, h => simp [wgRow, wgR_queueWriteBuffer] at h
      | [_, _, _], _, h => simp [wgRow, wgR_queueWriteBuffer] at h
      | [_, _, _, _], _, h => simp [wgRow, wgR_queueWriteBuffer] at h
      | _ :: _ :: _ :: _ :: _ :: _ :: _, _, h => simp [wgRow, wgR_queueWriteBuffer] at h
  | bufferRead =>
      match as, hb, h with
      | [a0, a1, a2, a3, a4], hb, h => exact row_bufferRead hS hl hb h
      | [], _, h => simp [wgRow, wgR_bufferRead] at h
      | [_], _, h => simp [wgRow, wgR_bufferRead] at h
      | [_, _], _, h => simp [wgRow, wgR_bufferRead] at h
      | [_, _, _], _, h => simp [wgRow, wgR_bufferRead] at h
      | [_, _, _, _], _, h => simp [wgRow, wgR_bufferRead] at h
      | _ :: _ :: _ :: _ :: _ :: _ :: _, _, h => simp [wgRow, wgR_bufferRead] at h
  | instanceCreateSurface =>
      match as, hb, h with
      | [a0, a1], hb, h => exact row_instanceCreateSurface hS hl hb h
      | [], _, h => simp [wgRow, wgR_instanceCreateSurface] at h
      | [_], _, h => simp [wgRow, wgR_instanceCreateSurface] at h
      | _ :: _ :: _ :: _, _, h => simp [wgRow, wgR_instanceCreateSurface] at h
  | surfaceConfigure =>
      match as, hb, h with
      | [a0, a1, a2, a3, a4], hb, h => exact row_surfaceConfigure hS hl hb h
      | [], _, h => simp [wgRow, wgR_surfaceConfigure] at h
      | [_], _, h => simp [wgRow, wgR_surfaceConfigure] at h
      | [_, _], _, h => simp [wgRow, wgR_surfaceConfigure] at h
      | [_, _, _], _, h => simp [wgRow, wgR_surfaceConfigure] at h
      | [_, _, _, _], _, h => simp [wgRow, wgR_surfaceConfigure] at h
      | _ :: _ :: _ :: _ :: _ :: _ :: _, _, h => simp [wgRow, wgR_surfaceConfigure] at h
  | surfaceGetCurrentTexture =>
      match as, hb, h with
      | [a0], hb, h => exact row_surfaceGetCurrentTexture hS hl hb h
      | [], _, h => simp [wgRow, wgR_surfaceGetCurrentTexture] at h
      | _ :: _ :: _, _, h => simp [wgRow, wgR_surfaceGetCurrentTexture] at h
  | textureCreateView =>
      match as, hb, h with
      | [a0], hb, h => exact row_textureCreateView hS hl hb h
      | [], _, h => simp [wgRow, wgR_textureCreateView] at h
      | _ :: _ :: _, _, h => simp [wgRow, wgR_textureCreateView] at h
  | surfacePresent =>
      match as, hb, h with
      | [a0], hb, h => exact row_surfacePresent hS hl hb h
      | [], _, h => simp [wgRow, wgR_surfacePresent] at h
      | _ :: _ :: _, _, h => simp [wgRow, wgR_surfacePresent] at h
  | release o =>
      match as, hb, h with
      | [a0], hb, h => exact row_release hS hl hb h
      | [], _, h => simp [wgRow, wgR_release] at h
      | _ :: _ :: _, _, h => simp [wgRow, wgR_release] at h
  | _ => simp [wgRow] at h

end AlgorithmLib.Check
