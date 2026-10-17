module
public import AlgorithmLib.Check.PreSound
public import AlgorithmLib.Check.Exec
@[expose] public section

/-!
# `Check.Sound` — what the checker accepts does not misuse

**`absStmt_sound`**: where the checker steps a statement from one abstract
state to another, the statement, run from any state the first describes, does
not misuse, does not fault, and leaves a state the second describes.
-/

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare
  AlgorithmLib.HProg.DevSpec AlgorithmLib.Prog

namespace AlgorithmLib.Check

variable {vals : List UInt64}

open AlgorithmLib.HProg.Contracts (Fact TState holds_of_mem Part afterStore afterStore_sound storeKeeps
  storeKeeps_sound storeKeeps_cell_of storeKeeps_cstrIn_of cellOk cellOk_spec nonCell_kept store_sizes_eq
  store_frozen callBits_rooms after streamsOf_holds eventsOf_holds readyIn)

-- ---------------------------------------------------------------------------
-- Addresses as numbers
-- ---------------------------------------------------------------------------

/-- An address a region decodes, as a number: its base plus the offset. -/
theorem decode_toNat {a : UInt64} {r : Region} {off : Nat} (h : decodeAddr a = some (r, off)) :
    a.toNat = (regionBase r).toNat + off ∧ (regionBase r).toNat + off < 2 ^ 39 := by
  obtain ⟨rfl, ho⟩ := Sem.decodeAddr_some h
  have hb := Sem.regionBase_toNat r
  rw [Sem.regionSpan_toNat] at ho hb
  refine ⟨?_, by omega⟩
  rw [UInt64.toNat_add, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (a := off) (by omega),
    Nat.mod_eq_of_lt (by omega)]

theorem Fits.toNat_le {m : Mem} {a : UInt64} {n : Nat} (h : Fits m a n) : a.toNat + n ≤ 2 ^ 40 := by
  obtain ⟨r, off, hd, -, -, -, hs⟩ := h
  have := decode_toNat hd
  omega

theorem cellOk_toNat {c : UInt64} (h : cellOk c = true) : c.toNat + 8 ≤ 2 ^ 64 := by
  obtain ⟨r, off, hd, -⟩ := cellOk_spec h
  have := decode_toNat hd
  omega

/-- The numbers `storeSpan` gives bound the bytes a store at an address it
    describes writes. -/
theorem storeSpan_spec {Γ : Env} {w : World} {S : TState} {av : AV} {n lo hi : Nat} {t : ClifTy}
    {x : UInt64} (hd : av.Desc Γ w (.sc t x)) (h : storeSpan S av n = some (lo, hi)) :
    lo ≤ x.toNat ∧ x.toNat + n ≤ hi := by
  unfold storeSpan at h
  split at h
  · cases av with
    | known u =>
        cases u with
        | sc t' x' =>
            simp only [Option.some.injEq, Prod.mk.injEq] at h
            obtain ⟨rfl, rfl⟩ := h
            simp only [AV.Desc, V.sc.injEq] at hd
            obtain ⟨-, rfl⟩ := hd
            omega
        | vec _ _ => simp at h
    | ptr rg m =>
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        obtain ⟨off, e, ⟨h1, h2, -⟩, h3⟩ := hd
        simp only [V.sc.injEq] at e
        obtain ⟨-, rfl⟩ := e
        have := addr_toNat rg (show off ≤ span by omega)
        omega
    | _ => simp at h
  · cases h

/-- A store inside `[lo, hi)` keeps what `rangeKeeps lo hi` keeps. -/
theorem rangeKeeps_store {lo hi n : Nat} {a : UInt64} {x : Fact} (hx : rangeKeeps lo hi x = true)
    (hlo : lo ≤ a.toNat) (hhi : a.toNat + n ≤ hi) (hw : a.toNat + n ≤ 2 ^ 64) :
    storeKeeps a n x = true := by
  cases x with
  | cell c v =>
      simp only [rangeKeeps, Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq] at hx
      exact storeKeeps_cell_of hx.1 (by omega) hw (cellOk_toNat hx.1)
  | cstr _ => cases hx
  | cstrIn c k =>
      simp only [rangeKeeps, Bool.or_eq_true, decide_eq_true_eq] at hx
      exact storeKeeps_cstrIn_of (by omega)
  | _ => rfl

theorem afterStore_ne8 {a : UInt64} {vo : Option UInt64} {n : Nat} {S : TState} (hn : n ≠ 8) :
    afterStore (some a) vo n S = afterStore (some a) none n S := by
  cases vo with
  | none => rfl
  | some v => simp [afterStore, hn]

-- ---------------------------------------------------------------------------
-- States
-- ---------------------------------------------------------------------------

theorem AS.Holds.desc {A : AS} {Γ : Env} {w : World} (hA : A.Holds vals Γ w) {r : R} {u : V}
    (hr : Γ[r]? = some u) : (A.get r).Desc Γ w u := by
  have hl : r < A.len := by
    rw [← hA.1.1]
    rcases Nat.lt_or_ge r Γ.size with h' | h'
    · exact h'
    · rw [Array.getElem?_eq_none h'] at hr; cases hr
  obtain ⟨u', hu', hd⟩ := hA.get hl
  rw [hr] at hu'
  cases hu'
  exact hd

theorem AS.lt_of_ne_any {A : AS} {r : R} (h : A.get r ≠ .any) : r < A.len := by
  rcases Nat.lt_or_ge r A.len with h' | h'
  · exact h'
  · exact absurd (AS.get_out (Nat.not_lt_of_ge h')) h

/-- The facts of a state, replaced by ones true of a world that keeps every
    region's size. -/
theorem AS.Holds.keep {A : AS} {Γ : Env} {w w' : World} (h : A.Holds vals Γ w) {S : TState}
    (hS : S.holds w') (hw : w'.mem.sizes = w.mem.sizes)
    (hh : ∀ (k : Contracts.Held) x, k.live w x = true → k.live w' x = true) : (A.withFacts S).Holds vals Γ w' :=
  h.world hS (fun r _ => congrFun hw r) hh

theorem AS.Holds.obsCall {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) (c : Callee) (vs : List V) :
    A.Holds vals Γ (obsCall w c vs) := by
  obtain ⟨⟨hs, hy, hp⟩, hd, hf⟩ := h
  refine ⟨⟨hs, fun i hi => Sym.fits_congr (hy i hi) (fun _ _ => rfl) rfl (fun _ _ => rfl) (Nat.le_refl _),
    hp.1, hp.2.1, AlgorithmLib.Prog.TState.holds_obsCall hp.2.2.1, hp.2.2.2⟩,
    fun i hi => ?_, AlgorithmLib.Prog.TState.holds_obsCall hf⟩
  obtain ⟨v, hv, hd⟩ := hd i hi
  exact ⟨v, hv, AV.desc_transport hd (fun _ _ => rfl) (fun _ _ => rfl) (fun k _ _ _ hk => by cases k <;> exact hk)⟩

theorem descAll_of {A : AS} {Γ : Env} {w : World} (hA : A.Holds vals Γ w) :
    ∀ {args : List R} {vs : List V}, args.mapM (fun r => Γ[r]?) = some vs → DescAll Γ w (args.map A.get) vs
  | [], vs, h => by
      simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h
      subst h; trivial
  | r :: rs, vs, h => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h
      obtain ⟨v, hv, vs', hvs, rfl⟩ := h
      exact ⟨hA.desc hv, descAll_of hA hvs⟩

-- ---------------------------------------------------------------------------
-- Stores
-- ---------------------------------------------------------------------------

/-- What an accepted store needs of the state: its address a scalar whose
    bytes fit, its value in scope. -/
theorem absStore_pre {A A' : AS} {v a : R} {n : Nat} (h : absStore A v a n = .ok A') {Γ : Env}
    {w : World} (hA : A.Holds vals Γ w) :
    ∃ k t x val, storeBytes A v n = some k ∧ Γ[a]? = some (.sc t x) ∧ Γ[v]? = some val ∧
      Fits w.mem x k := by
  unfold absStore at h
  cases hk : storeBytes A v n with
  | none => rw [hk] at h; cases h
  | some k =>
      rw [hk] at h
      dsimp only at h
      cases hs : storeSpan A.facts (A.get a) k with
      | none => rw [hs] at h; split at h <;> first | contradiction | cases h
      | some sp =>
          have hm : memN A.facts (A.get a) k = true := by
            unfold storeSpan at hs; split at hs
            · assumption
            · cases hs
          have hna : A.get a ≠ .any := by intro e; rw [e] at hm; cases hm
          obtain ⟨u, hu, hdu⟩ := hA.get' hna
          obtain ⟨t, x, rfl⟩ := memN_sc hdu hm
          have hnv : A.get v ≠ .any := by
            intro e
            simp only [storeBytes, e, AV.scTy, AV.ty] at hk
            cases hk
          obtain ⟨val, hv, -⟩ := hA.get' hnv
          exact ⟨k, t, x, val, rfl, hu, hv, memN_fits hA.2.2 ⟨t, hdu⟩ hm⟩

/-- **A scalar store**: the state `absStore` leaves describes the world the
    store leaves. -/
theorem absStore_scalar {A A' : AS} {v a : R} {n : Nat} (h : absStore A v a n = .ok A') {Γ : Env}
    {w : World} (hA : A.Holds vals Γ w) {t tv : ClifTy} {x b b' c : UInt64} {m : Mem} {k : Nat}
    (ha : Γ[a]? = some (.sc t x)) (hsc : (A.get v).scTy = some tv) (hv : Γ[v]? = some (.sc tv b))
    (hm : w.mem.store x n b' = some m) (hb : n = 8 → b' = b) :
    A'.Holds vals Γ { obsStore w x k c with mem := m } := by
  have hfit : Fits w.mem x n := by
    obtain ⟨k', t', x', val, hk', ha', -, hf⟩ := absStore_pre h hA
    simp only [storeBytes, hsc, Option.some.injEq] at hk'
    rw [ha] at ha'; cases ha'; subst hk'; exact hf
  have hx := Fits.toNat_le hfit
  have hn : n ≤ 2 ^ 64 := by omega
  have hsz := store_sizes_eq hm
  have hp := fun p => store_parts (a := x) (n := k) (b := c) (w := w) (store_frozen hm) p
  have hda : (A.get a).Desc Γ w (.sc t x) := hA.desc ha
  unfold absStore at h
  simp only [storeBytes, hsc] at h
  generalize hsp : storeSpan A.facts (A.get a) n = sp at h
  generalize hga : A.get a = ga at h hsp hda
  have hpar : ∀ {lo hi : Nat}, lo ≤ x.toNat → x.toNat + n ≤ hi →
      (A.par.keepWorld (rangeKeeps lo hi)).At vals Γ { obsStore w x k c with mem := m } := fun hlo hhi =>
    hA.1.2.2.keepWorld (fun _ _ => rfl) (fun _ _ => rfl) fun f hf hx =>
      storeKeeps_sound (S := [f]) (fun g hg => by rw [List.mem_singleton.mp hg]; exact hx) hp
        (fun kk _ => by cases kk <;> rfl) rfl ⟨rfl, rfl, rfl⟩ hsz hm hn f
        (List.mem_filter.mpr ⟨List.mem_singleton_self _, rangeKeeps_store hf hlo hhi (by omega)⟩)
  split at h
  next t0 x0 lo hi _ =>
    cases h
    obtain ⟨hlo, hhi⟩ := storeSpan_spec hda hsp
    simp only [AV.Desc, V.sc.injEq] at hda
    obtain ⟨-, rfl⟩ := hda
    refine AS.Holds.withPar (hA.keep ?_ hsz (fun kk _ hk => by cases kk <;> exact hk)) (hpar hlo hhi)
    by_cases h8 : n = 8
    · exact afterStore_sound hA.2.2 hp (fun kk _ => by cases kk <;> rfl) rfl ⟨rfl, rfl, rfl⟩ hsz
        (fun _ => hm) (fun _ e => (Option.some.inj e).symm)
        (fun y hy => by
          obtain ⟨t', e⟩ := AV.bits_desc (hA.desc hv) hy
          cases e; exact (hb h8).symm) hn
    · rw [afterStore_ne8 h8]
      exact afterStore_sound hA.2.2 hp (fun kk _ => by cases kk <;> rfl) rfl ⟨rfl, rfl, rfl⟩ hsz
        (fun _ => hm) (fun _ e => (Option.some.inj e).symm) (fun _ e => by cases e) hn
  next lo hi _ =>
    cases h
    obtain ⟨hlo, hhi⟩ := storeSpan_spec hda hsp
    refine AS.Holds.withPar (hA.keep ?_ hsz (fun kk _ hk => by cases kk <;> exact hk)) (hpar hlo hhi)
    have hk := storeKeeps_sound hA.2.2 hp (fun kk _ => by cases kk <;> rfl) rfl ⟨rfl, rfl, rfl⟩ hsz hm hn
    intro f hf
    obtain ⟨hf1, hf2⟩ := List.mem_filter.mp hf
    exact hk f (List.mem_filter.mpr ⟨hf1, rangeKeeps_store hf2 hlo hhi (by omega)⟩)
  all_goals contradiction

/-- **A vector store**: every fact but the cells survives. -/
theorem absStore_vec {A A' : AS} {v a : R} {n : Nat} (h : absStore A v a n = .ok A') {Γ : Env}
    {w : World} (hA : A.Holds vals Γ w) (hsc : (A.get v).scTy = none) {x c : UInt64} {m : Mem} {k : Nat}
    (hsz : m.sizes = w.mem.sizes) (hz : m.frozen = w.mem.frozen) :
    A'.Holds vals Γ { obsStore w x k c with mem := m } := by
  unfold absStore at h
  cases hk : storeBytes A v n with
  | none => rw [hk] at h; cases h
  | some k' =>
      rw [hk] at h
      dsimp only at h
      rw [hsc] at h
      split at h <;> first
        | contradiction
        | (cases h
           refine AS.Holds.withPar (hA.keep (nonCell_kept hA.2.2 (store_parts hz) (fun kk _ => by cases kk <;> rfl)
             rfl ⟨rfl, rfl, rfl⟩ hsz) hsz (fun kk _ hk => by cases kk <;> exact hk))
             (hA.1.2.2.keepWorld (fun _ _ => rfl) (fun _ _ => rfl) fun f hf hx =>
               nonCell_kept (S := [f]) (fun g hg => by rw [List.mem_singleton.mp hg]; exact hx) (store_parts hz)
                 (fun kk _ => by cases kk <;> rfl) rfl ⟨rfl, rfl, rfl⟩ hsz f
                 (List.mem_filter.mpr ⟨List.mem_singleton_self _, hf⟩)))

/-- Lanes inside bytes that fit all store. -/
theorem lanes_store {m0 : Mem} {x : UInt64} {lane : ClifTy} {ls : Array UInt64}
    (hfit : Fits m0 x (ls.size * tyBytes lane)) (hw : 0 < tyBytes lane) :
    ∃ m, ls.zipIdx.toList.foldlM (fun mm (p : UInt64 × Nat) =>
      mm.store (x + UInt64.ofNat (p.2 * tyBytes lane)) (tyBytes lane) p.1) m0 = some m :=
  foldlM_lane_stores_some _ _ fun p hp => by
    obtain ⟨-, hi, -⟩ := Array.mem_zipIdx (Array.mem_toList_iff.mp hp)
    exact hfit.part (by
      have := Nat.mul_le_mul_right (tyBytes lane) (Nat.succ_le_of_lt hi)
      rw [Nat.succ_mul] at this; simp at this; omega) hw

theorem lanes_pos {T lane : ClifTy} {n : Nat} (hl : T.lanes = some (lane, n)) : 0 < tyBytes lane := by
  cases T <;> simp [ClifTy.lanes] at hl <;> obtain ⟨rfl, rfl⟩ := hl <;> decide

/-- What a store's value is, where `absStore` accepts it: a scalar of the
    type the state gives it, or a vector whose lanes the bytes it was given
    room for hold. -/
theorem store_val {A : AS} {Γ : Env} {w : World} {v : R} {n k : Nat} {val : V}
    (hd : (A.get v).Desc Γ w val) (hk : storeBytes A v n = some k) :
    (∃ tv b, (A.get v).scTy = some tv ∧ val = .sc tv b ∧ k = n) ∨
    (∃ T lane cnt ls, (A.get v).scTy = none ∧ T.lanes = some (lane, cnt) ∧ val = .vec T ls ∧
      ls.size = cnt ∧ k = cnt * tyBytes lane) := by
  cases hsc : (A.get v).scTy with
  | some tv =>
      obtain ⟨b, rfl⟩ := AV.scTy_desc hd hsc
      simp only [storeBytes, hsc, Option.some.injEq] at hk
      exact .inl ⟨tv, b, rfl, rfl, hk.symm⟩
  | none =>
      simp only [storeBytes, hsc] at hk
      cases hT : (A.get v).ty with
      | none => rw [hT] at hk; cases hk
      | some T =>
          simp only [hT] at hk
          have hty := AV.ty_desc hd hT
          cases hl : T.lanes with
          | none => rw [hl] at hk; cases hk
          | some p =>
              obtain ⟨lane, cnt⟩ := p
              rw [hl] at hk
              simp only [Option.map_some, Option.some.injEq] at hk
              simp only [TyV, hl] at hty
              obtain ⟨ls, rfl, hn⟩ := hty
              exact .inr ⟨T, lane, cnt, ls, rfl, hl, rfl, hn, hk.symm⟩

theorem store_sound {cfg : Cfg} {A A' : AS} {ty : ClifTy} {v a : R} {fo : Bool}
    (h : absStore A v a (tyBytes ty) = .ok A') : Stmt1 cfg (A.Holds vals) (.store ty v a) (A'.Holds vals) fo := by
  intro Γ w hA
  obtain ⟨k, t, x, val, hk, ha, hv, hfit⟩ := absStore_pre h hA
  rw [runStmt_store, hv, ha]
  rcases store_val (hA.desc hv) hk with ⟨tv, b, hsc, rfl, rfl⟩ | ⟨T, lane, cnt, ls, hsc, hl, rfl, hn, rfl⟩
  · obtain ⟨m, hm⟩ := hfit.store b
    simp only [hm]
    exact absStore_scalar h hA ha hsc hv hm (fun _ => rfl)
  · have hlw : ((some (lane, cnt)).map (·.1.width)).getD 8 / 8 = tyBytes lane := rfl
    simp only [hl]
    rw [hlw]
    subst hn
    obtain ⟨mm, hmm⟩ := lanes_store hfit (lanes_pos hl)
    obtain ⟨hs, hz⟩ := foldlM_lane_stores_inv _ _ _ hmm
    rw [Array.foldlM_toList] at hmm
    simp only [hmm]
    exact absStore_vec h hA hsc hs hz

theorem storeU_sound {cfg : Cfg} {A A' : AS} {v a : R} {fo : Bool}
    (h : absStore A v a (tyBytes ((A.get v).scTy.getD .i64)) = .ok A') :
    Stmt1 cfg (A.Holds vals) (.storeUnaligned v a) (A'.Holds vals) fo := by
  intro Γ w hA
  obtain ⟨k, t, x, val, hk, ha, hv, hfit⟩ := absStore_pre h hA
  rw [runStmt_storeUnaligned, hv, ha]
  rcases store_val (hA.desc hv) hk with ⟨tv, b, hsc, rfl, rfl⟩ | ⟨T, lane, cnt, ls, hsc, hl, rfl, hn, rfl⟩
  · simp only [hsc, Option.getD_some] at hfit h
    obtain ⟨m, hm⟩ := hfit.store b
    simp only [hm]
    exact absStore_scalar h hA ha hsc hv hm (fun _ => rfl)
  · have hlw : ((some (lane, cnt)).map (·.1.width)).getD 8 / 8 = tyBytes lane := rfl
    simp only [hl]
    rw [hlw]
    subst hn
    obtain ⟨mm, hmm⟩ := lanes_store hfit (lanes_pos hl)
    obtain ⟨hs, hz⟩ := foldlM_lane_stores_inv _ _ _ hmm
    rw [Array.foldlM_toList] at hmm
    simp only [hmm]
    exact absStore_vec h hA hsc hs hz

theorem istore8_sound {cfg : Cfg} {A A' : AS} {v a : R} {tv : ClifTy} {fo : Bool}
    (hsc : (A.get v).scTy = some tv)
    (h : absStore A v a 1 = .ok A') : Stmt1 cfg (A.Holds vals) (.istore8 v a) (A'.Holds vals) fo := by
  intro Γ w hA
  obtain ⟨k, t, x, val, hk, ha, hv, hfit⟩ := absStore_pre h hA
  rw [runStmt_istore8, hv, ha]
  rcases store_val (hA.desc hv) hk with ⟨tv', b, hsc', rfl, rfl⟩ | ⟨_, _, _, _, hsc', -⟩
  · rw [hsc] at hsc'
    cases hsc'
    obtain ⟨m, hm⟩ := hfit.store (b &&& 0xff)
    simp only [hm]
    exact absStore_scalar h hA ha hsc hv hm (fun e => by cases e)
  · rw [hsc] at hsc'; cases hsc'


/-- Whatever a store writes, every fact but the cells survives it. -/
theorem wild_holds {A : AS} {Γ : Env} {w : World} (hA : A.Holds vals Γ w) {x c : UInt64} {k : Nat} {m : Mem}
    (hsz : m.sizes = w.mem.sizes) (hz : m.frozen = w.mem.frozen) :
    (A.withFacts (A.facts.filter (!·.isCell))).Holds vals Γ { obsStore w x k c with mem := m } :=
  hA.keep (nonCell_kept hA.2.2 (store_parts hz) (fun kk _ => by cases kk <;> rfl) rfl ⟨rfl, rfl, rfl⟩ hsz) hsz
    (fun kk _ hk => by cases kk <;> exact hk)

theorem wild_store {cfg : Cfg} {A : AS} {ty : ClifTy} {v a : R} :
    Stmt1 cfg (A.Holds vals) (.store ty v a) ((A.withFacts (A.facts.filter (!·.isCell))).Holds vals) true := by
  intro Γ w hA
  rw [runStmt_store]
  split
  · dsimp only
    split
    · split
      · rename_i m hm; exact wild_holds hA (store_sizes_eq hm) (store_frozen hm)
      · rfl
    · split
      · rename_i m hm
        rw [← Array.foldlM_toList] at hm
        obtain ⟨hs, hz⟩ := Contracts.foldl_stores_sizes _ (fun _ _ _ h => ⟨store_sizes_eq h, store_frozen h⟩) _ _ _ hm
        exact wild_holds hA hs hz
      · rfl
  · trivial

theorem wild_storeU {cfg : Cfg} {A : AS} {v a : R} :
    Stmt1 cfg (A.Holds vals) (.storeUnaligned v a) ((A.withFacts (A.facts.filter (!·.isCell))).Holds vals) true := by
  intro Γ w hA
  rw [runStmt_storeUnaligned]
  split
  · split
    · rename_i m hm; exact wild_holds hA (store_sizes_eq hm) (store_frozen hm)
    · rfl
  · dsimp only
    split
    · rename_i m hm
      rw [← Array.foldlM_toList] at hm
      obtain ⟨hs, hz⟩ := Contracts.foldl_stores_sizes _ (fun _ _ _ h => ⟨store_sizes_eq h, store_frozen h⟩) _ _ _ hm
      exact wild_holds hA hs hz
    · rfl
  · trivial

theorem wild_istore8 {cfg : Cfg} {A : AS} {v a : R} :
    Stmt1 cfg (A.Holds vals) (.istore8 v a) ((A.withFacts (A.facts.filter (!·.isCell))).Holds vals) true := by
  intro Γ w hA
  rw [runStmt_istore8]
  split
  · split
    · rename_i m hm; exact wild_holds hA (store_sizes_eq hm) (store_frozen hm)
    · rfl
  · trivial

theorem storeBytes_any {A : AS} {v a : R} {n : Nat} (hm : (storeBytes A v n).any (memRel A a) = true) :
    ∃ k, storeBytes A v n = some k ∧ memRel A a k = true := by
  cases hb : storeBytes A v n with
  | none => simp [hb] at hm
  | some k => exact ⟨k, rfl, by simpa [hb] using hm⟩

/-- A store the linear facts place inside a region keeps every fact but the
    cells, and does not fault. -/
theorem rel_store {cfg : Cfg} {A : AS} {ty : ClifTy} {v a : R} {fo : Bool}
    (hm : (storeBytes A v (tyBytes ty)).any (memRel A a) = true) :
    Stmt1 cfg (A.Holds vals) (.store ty v a) ((A.withFacts (A.facts.filter (!·.isCell))).Holds vals) fo := by
  intro Γ w hA
  obtain ⟨k, hk, hmk⟩ := storeBytes_any hm
  cases hv : Γ[v]? with
  | none => rw [runStmt_store, hv]; trivial
  | some val =>
    cases ha : Γ[a]? with
    | none => rw [runStmt_store, hv, ha]; trivial
    | some av =>
      cases av with
      | vec _ _ => rw [runStmt_store, hv, ha]; trivial
      | sc t x =>
        have hfit := memRel_fits hA hmk ha
        rw [runStmt_store, hv, ha]
        rcases store_val (hA.desc hv) hk with ⟨tv, b, hsc, rfl, rfl⟩ | ⟨T, lane, cnt, ls, hsc, hl, rfl, hn, rfl⟩
        · obtain ⟨m, hm⟩ := hfit.store b
          simp only [hm]
          exact wild_holds hA (store_sizes_eq hm) (store_frozen hm)
        · have hlw : ((some (lane, cnt)).map (·.1.width)).getD 8 / 8 = tyBytes lane := rfl
          simp only [hl]
          rw [hlw]
          subst hn
          obtain ⟨mm, hmm⟩ := lanes_store hfit (lanes_pos hl)
          obtain ⟨hs, hz⟩ := foldlM_lane_stores_inv _ _ _ hmm
          rw [Array.foldlM_toList] at hmm
          simp only [hmm]
          exact wild_holds hA hs hz

theorem rel_storeU {cfg : Cfg} {A : AS} {v a : R} {fo : Bool}
    (hm : (storeBytes A v (tyBytes ((A.get v).scTy.getD .i64))).any (memRel A a) = true) :
    Stmt1 cfg (A.Holds vals) (.storeUnaligned v a) ((A.withFacts (A.facts.filter (!·.isCell))).Holds vals) fo := by
  intro Γ w hA
  obtain ⟨k, hk, hmk⟩ := storeBytes_any hm
  cases hv : Γ[v]? with
  | none => rw [runStmt_storeUnaligned, hv]; trivial
  | some val =>
    cases ha : Γ[a]? with
    | none => rw [runStmt_storeUnaligned, hv, ha]; cases val <;> trivial
    | some av =>
      cases av with
      | vec _ _ => rw [runStmt_storeUnaligned, hv, ha]; cases val <;> trivial
      | sc t x =>
        have hfit := memRel_fits hA hmk ha
        rw [runStmt_storeUnaligned, hv, ha]
        rcases store_val (hA.desc hv) hk with ⟨tv, b, hsc, rfl, rfl⟩ | ⟨T, lane, cnt, ls, hsc, hl, rfl, hn, rfl⟩
        · simp only [hsc, Option.getD_some] at hfit
          obtain ⟨m, hm⟩ := hfit.store b
          simp only [hm]
          exact wild_holds hA (store_sizes_eq hm) (store_frozen hm)
        · have hlw : ((some (lane, cnt)).map (·.1.width)).getD 8 / 8 = tyBytes lane := rfl
          simp only [hl]
          rw [hlw]
          subst hn
          obtain ⟨mm, hmm⟩ := lanes_store hfit (lanes_pos hl)
          obtain ⟨hs, hz⟩ := foldlM_lane_stores_inv _ _ _ hmm
          rw [Array.foldlM_toList] at hmm
          simp only [hmm]
          exact wild_holds hA hs hz

theorem rel_istore8 {cfg : Cfg} {A : AS} {v a : R} {fo : Bool}
    (hm : (storeBytes A v 1).any (memRel A a) = true) :
    Stmt1 cfg (A.Holds vals) (.istore8 v a) ((A.withFacts (A.facts.filter (!·.isCell))).Holds vals) fo := by
  intro Γ w hA
  obtain ⟨k, hk, hmk⟩ := storeBytes_any hm
  cases hv : Γ[v]? with
  | none => rw [runStmt_istore8, hv]; trivial
  | some val =>
    cases val with
    | vec _ _ => rw [runStmt_istore8, hv]; trivial
    | sc tv b =>
      cases ha : Γ[a]? with
      | none => rw [runStmt_istore8, hv, ha]; trivial
      | some av =>
        cases av with
        | vec _ _ => rw [runStmt_istore8, hv, ha]; trivial
        | sc t x =>
          have hfit := memRel_fits hA hmk ha
          rw [runStmt_istore8, hv, ha]
          have hk1 : k = 1 := by
            rcases store_val (hA.desc hv) hk with ⟨_, _, _, -, e⟩ | ⟨_, _, _, _, -, -, e, -⟩
            · exact e
            · cases e
          subst hk1
          obtain ⟨m, hm⟩ := hfit.store (b &&& 0xff)
          simp only [hm]
          exact wild_holds hA (store_sizes_eq hm) (store_frozen hm)

theorem storeF_sound {cfg : Cfg} {A A' : AS} {ty : ClifTy} {v a : R} {fo : Bool}
    (h : absStoreF fo A v a (tyBytes ty) = .ok A') : Stmt1 cfg (A.Holds vals) (.store ty v a) (A'.Holds vals) fo := by
  unfold absStoreF at h
  split at h
  · rename_i A1 h1; cases h; exact store_sound h1
  · split at h
    · rename_i hc; cases h
      rcases Bool.or_eq_true_iff.mp hc with hm | hfo
      · exact rel_store hm
      · subst hfo; exact wild_store
    · cases h

theorem storeUF_sound {cfg : Cfg} {A A' : AS} {v a : R} {fo : Bool}
    (h : absStoreF fo A v a (tyBytes ((A.get v).scTy.getD .i64)) = .ok A') :
    Stmt1 cfg (A.Holds vals) (.storeUnaligned v a) (A'.Holds vals) fo := by
  unfold absStoreF at h
  split at h
  · rename_i A1 h1; cases h; exact storeU_sound h1
  · split at h
    · rename_i hc; cases h
      rcases Bool.or_eq_true_iff.mp hc with hm | hfo
      · exact rel_storeU hm
      · subst hfo; exact wild_storeU
    · cases h

theorem istore8F_sound {cfg : Cfg} {A A' : AS} {v a : R} {tv : ClifTy} {fo : Bool}
    (hsc : (A.get v).scTy = some tv) (h : absStoreF fo A v a 1 = .ok A') :
    Stmt1 cfg (A.Holds vals) (.istore8 v a) (A'.Holds vals) fo := by
  unfold absStoreF at h
  split at h
  · rename_i A1 h1; cases h; exact istore8_sound hsc h1
  · split at h
    · rename_i hc; cases h
      rcases Bool.or_eq_true_iff.mp hc with hm | hfo
      · exact rel_istore8 hm
      · subst hfo; exact wild_istore8
    · cases h
-- ---------------------------------------------------------------------------
-- Calls
-- ---------------------------------------------------------------------------

theorem ffi_result_lanes (f : Ffi) : (f.result.getD .i64).lanes = none := by
  cases f <;> rfl

theorem ffiCall_ok {A B : AS} {f : Ffi} {args : List R} (h : A.ffiCall f args = .ok B) :
    f ≠ .threadSpawn ∧ A.facts.contains (.part .frozen false) = true ∧
      (args.map A.get).all (fun a => a.scTy.isSome) = true ∧
      ∃ P, ffiPre A f args = some P ∧
        B = (A.forgetHands.withFacts (postA f (args.map A.get) A.facts)).withPar
          (P.keepWorld (keepA f (args.map A.get))) := by
  unfold AS.ffiCall at h
  by_cases hc : (f != .threadSpawn && A.facts.contains (.part .frozen false) &&
      (args.map A.get).all (·.scTy.isSome)) = true
  · rw [if_pos hc] at h
    simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at hc
    obtain ⟨⟨hf, hz⟩, hsc⟩ := hc
    cases hP : ffiPre A f args with
    | none => rw [hP] at h; cases h
    | some P => rw [hP] at h; cases h; exact ⟨hf, hz, hsc, P, rfl, rfl⟩
  · rw [if_neg hc] at h; cases h

theorem parArg_bits {A : AS} {Γ : Env} {w : World} {vals : List UInt64} (hA : A.Holds vals Γ w)
    (hP : A.par.At vals Γ w) {r : R} {p : PArg} (hp : parArg A r = some p) {v : V} (hv : Γ[r]? = some v)
    {x : UInt64} (hx : asBits v = some x) : p.bits vals = x := by
  unfold parArg at hp
  split at hp
  · rename_i t e he
    cases hp
    rw [hP.slotOf he] at hv; cases hv
    simp only [asBits, Option.some.injEq] at hx
    exact hx
  · obtain ⟨y, hy, rfl⟩ := Option.map_eq_some_iff.mp hp
    cases v with
    | sc t z =>
        simp only [asBits, Option.some.injEq] at hx
        subst hx
        exact (known_bits ⟨t, hA.desc hv⟩ hy).symm
    | vec _ _ => cases hx

/-- The arguments' bits, as the parameters' record reads them. -/
theorem parArgs_bits {A : AS} {Γ : Env} {w : World} {vals : List UInt64} (hA : A.Holds vals Γ w)
    (hP : A.par.At vals Γ w) :
    ∀ {args : List R} {pas : List PArg} {vs : List V} {bits : List UInt64},
      args.mapM (parArg A) = some pas → args.mapM (fun r => Γ[r]?) = some vs → vs.mapM asBits = some bits →
      pas.map (PArg.bits vals) = bits
  | [], pas, vs, bits, h1, h2, h3 => by
      simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h1 h2
      subst h1 h2
      simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h3
      subst h3; rfl
  | r :: rs, pas, vs, bits, h1, h2, h3 => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h1 h2
      obtain ⟨p, hp, ps, hps, rfl⟩ := h1
      obtain ⟨v, hv, vs', hvs, rfl⟩ := h2
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h3
      obtain ⟨x, hx, xs, hxs, rfl⟩ := h3
      simp only [List.map_cons, parArg_bits hA hP hp hv hx, parArgs_bits hA hP hps hvs hxs]

/-- **A precondition `ffiPre` shows** holds, and the record it leaves is the
    state's own. -/
theorem ffiPre_pre {A : AS} {f : Ffi} {args : List R} {P : Par} (hP : ffiPre A f args = some P)
    {Γ : Env} {w : World} {vs : List V} {bits : List UInt64} (hA : A.Holds vals Γ w)
    (hvs : args.mapM (fun r => Γ[r]?) = some vs) (hb : vs.mapM asBits = some bits) :
    Contracts.Pre f bits w ∧ P = A.par := by
  have hv := hA.1.2.2
  unfold ffiPre at hP
  by_cases hp : preB f (args.map A.get) A.facts (relRb A args) = true
  · rw [if_pos hp] at hP
    cases hP
    exact ⟨preB_sound hA.2.2 (argBits_of (descAll_of hA hvs) hb) hp (relRb_sound hA hvs hb), rfl⟩
  · rw [if_neg hp] at hP
    cases hpas : args.mapM (parArg A) with
    | none => rw [hpas] at hP; cases hP
    | some pas =>
      rw [hpas] at hP
      dsimp only at hP
      by_cases hc : A.par.obls.contains (oblOf A f pas) = true
      · rw [if_pos hc] at hP
        cases hP
        refine ⟨?_, rfl⟩
        have hpre := hv.2.2.2 _ (List.contains_iff_mem.mp hc) w (fun x hx =>
          (List.mem_append.mp hx).elim (hA.2.2 x) fun hx => hv.2.2.1 x (List.mem_append_right _ hx))
        simp only [oblOf] at hpre
        rw [parArgs_bits hA hv hpas hvs hb] at hpre
        exact hpre
      · rw [if_neg hc, hv.1] at hP
        cases hP

/-- **The index a device call answers** is the one `devAnswer` reads from the
    facts. -/
theorem devAnswer_sound {S : TState} {Γ : Env} {w : World} {f : Ffi} {as : List AV} {bits : List UInt64}
    {n : Nat} {v : V} {w' : World} (hS : S.holds w) (hab : ArgBits Γ w as bits) (hd : devAnswer f as S = some n)
    (hc : callBits f bits w = some (some v, w')) : v = ofInt .i32 n := by
  have arg1 : ∀ {c : AV}, ArgBits Γ w [c] bits → c.bits = some cudaCtx → bits = [cudaCtx] := fun hab hcb => by
    obtain ⟨y, ys, rfl, hy, hys⟩ := ArgBits.cons_inv hab
    rw [ArgBits.nil_inv hys, known_bits hy hcb]
  unfold devAnswer at hd
  split at hd
  · split at hd
    · rename_i c
      split at hd
      · rename_i hif
        simp only [Bool.and_eq_true, beq_iff_eq] at hif
        obtain ⟨l, hl, rfl⟩ := Option.map_eq_some_iff.mp hd
        have hbits := arg1 hab hif.1
        subst hbits
        obtain ⟨hr, -⟩ := Device.streamCreate_dev (show ffiCudaStreamCreate [cudaCtx] w = _ from hc)
          (holds_of_mem hS hif.2)
        cases hr
        rw [← streamsOf_holds hS hl, Array.length_toList]
      · cases hd
    · cases hd
  · split at hd
    · rename_i c
      split at hd
      · rename_i hif
        simp only [Bool.and_eq_true, beq_iff_eq] at hif
        have hbits := arg1 hab hif.1
        subst hbits
        obtain ⟨hr, -⟩ := Device.eventCreate_dev (show ffiCudaEventCreate [cudaCtx] w = _ from hc)
          (holds_of_mem hS hif.2)
        cases hr
        rw [eventsOf_holds hS hd]
      · cases hd
    · cases hd
  · cases hd

/-- **A foreign call the checker accepts** answers, from every state the
    abstract one describes, and leaves a world the state after it describes. -/
theorem ffiCall_core {cfg : Cfg} {A B : AS} {f : Ffi} {args : List R} (h : A.ffiCall f args = .ok B)
    {Γ : Env} {w : World} {vs : List V} (hA : A.Holds vals Γ w) (hvs : args.mapM (fun r => Γ[r]?) = some vs) :
    ∃ r w' bits, callOf cfg.locals (.ffi f) vs (obsCall w (.ffi f) vs) = some (r, w') ∧ B.Holds vals Γ w' ∧
      ArgBits Γ (obsCall w (.ffi f) vs) (args.map A.get) bits ∧
      ∀ v, r = some v → ∃ x, v = .sc (f.result.getD .i64) x ∧ answerOk f bits x ∧
        ∀ n, devAnswer f (args.map A.get) A.facts = some n → v = ofInt .i32 n := by
  obtain ⟨hf, hz, hsc, P, hP, rfl⟩ := ffiCall_ok h
  have hA1 := hA.obsCall (.ffi f) vs
  have hd := descAll_of hA1 hvs
  obtain ⟨bits, hb⟩ := bits_exist hd hsc
  have hab := argBits_of hd hb
  have hz' : (obsCall w (.ffi f) vs).mem.frozen = false := holds_of_mem hA1.2.2 hz
  obtain ⟨hpre, rfl⟩ := ffiPre_pre hP hA1 hvs hb
  obtain ⟨⟨r, w'⟩, hc⟩ := Option.isSome_iff_exists.mp (Contracts.pre_safe f bits _ hpre)
  have hcall : callOf cfg.locals (.ffi f) vs (obsCall w (.ffi f) vs) = some (r, w') := by
    rw [Contracts.callOf_ffi hf cfg.locals vs hz', hb]; exact hc
  refine ⟨r, w', bits, hcall, (hA1.forgetHands.worldNH (AS.forgetHands_noHands A) (postA_sound hab hA1.2.2 hc)
    (callBits_rooms f hc)).withPar (hA1.1.2.2.keepWorld (fun _ _ => rfl) (fun _ _ => rfl)
      fun _ hk hx => keepA_sound hab hc hk hx), hab, fun v hv' => ?_⟩
  subst hv'
  obtain ⟨x, rfl⟩ := Contracts.callOf_ffi_scalar hf hcall
  exact ⟨x, rfl, answerOk_sound hc, fun n hn => devAnswer_sound hA1.2.2 hab hn hc⟩

theorem mapM_bits_eq {Γ : Env} {w : World} :
    ∀ {as : List AV} {bs bits : List UInt64}, as.mapM AV.bits = some bs → ArgBits Γ w as bits → bs = bits
  | [], bs, bits, h, hb => by
      simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h
      subst h; exact (ArgBits.nil_inv hb).symm
  | a :: as, bs, bits, h, hb => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h
      obtain ⟨x, hx, xs, hxs, rfl⟩ := h
      obtain ⟨y, ys, rfl, hy, hys⟩ := ArgBits.cons_inv hb
      rw [known_bits hy hx, mapM_bits_eq hxs hys]

theorem extKeepsA_sound {Γ : Env} {w : World} {e : Ext} {as : List AV} {bits : List UInt64}
    (hab : ArgBits Γ w as bits) {f : Fact} (h : extKeepsA e as f = true) : Contracts.extKeeps e bits f = true := by
  unfold extKeepsA at h
  split at h
  · split at h
    · rename_i bs hbs; rw [← mapM_bits_eq hbs hab]; exact h
    · cases h
  · split at h
    · rename_i bs hbs; rw [← mapM_bits_eq hbs hab]; exact h
    · cases h
  · rename_i hc hs
    cases f <;> first | exact absurd rfl (hc _ _) | exact absurd rfl (hs _ _) | simpa [Contracts.extKeeps] using h

/-- A library call keeps every region's size. -/
theorem ext_rooms {A : AS} {Γ : Env} {w : World} (hA : A.Holds vals Γ w) {e : Ext} {args : List R} {vs : List V}
    {lc : Locals} {res : Option V} {w2 : World} (ha : e.aside = true)
    (hsc : (args.map A.get).all (fun a => a.scTy.isSome) = true) (hvs : args.mapM (fun r => Γ[r]?) = some vs)
    (hc : callOf lc (.ext e) vs (obsCall w (.ext e) vs) = some (res, w2)) : Contracts.KeepsRooms w w2 := by
  have hA1 := hA.obsCall (.ext e) vs
  obtain ⟨bits, hb⟩ := bits_exist (descAll_of hA1 hvs) hsc
  have st := Contracts.lib_step ha hc hb
  exact fun r hr => st.mem.2.1 r hr

/-- **A library call the checker accepts** ends where its state says: the facts
    it keeps, the handles of the kind it may close forgotten, and its answer. -/
theorem extCall_core {cfg : Cfg} {A : AS} {e : Ext} {args : List R} (ha : e.aside = true)
    (hsc : (args.map A.get).all (fun a => a.scTy.isSome) = true) (hp : preExt A.facts e (args.map A.get) = true)
    {Γ : Env} {w : World} {vs : List V} (hA : A.Holds vals Γ w) (hvs : args.mapM (fun r => Γ[r]?) = some vs) :
    ∃ r w', callOf cfg.locals (.ext e) vs (obsCall w (.ext e) vs) = some (r, w') ∧
      ((A.forgetKind e.closes).withFacts (A.facts.filter (extKeepsA e (args.map A.get)))).Holds vals Γ w' ∧
      ∀ v, r = some v → (extAnswer e).Desc (Γ.push v) w' v := by
  have hA1 := hA.obsCall (.ext e) vs
  have hd := descAll_of hA1 hvs
  obtain ⟨bits, hb⟩ := bits_exist hd hsc
  have hab := argBits_of hd hb
  obtain ⟨hs, rs, hn, hlive, hroom⟩ := preExt_sound hA1.2.2 hab hp
  have hpre := Contracts.extNeeds_pre hn hlive hroom
  have hsome := ExtContracts.ext_pre_safe e bits _ hpre
  rw [Contracts.extCall_bits (cs := e.args bits) (cs' := vs) ha _
    (ExtContracts.mapM_asBits_zip _ _ (Contracts.extNeeds_length hn)) hb] at hsome
  obtain ⟨⟨r, w'⟩, hcall⟩ := Option.isSome_iff_exists.mp hsome
  have st := Contracts.lib_step ha hcall hb
  have hafter := Contracts.extAfter_sound hA1.2.2 hb hcall
  have hkept : TState.holds (A.facts.filter (extKeepsA e (args.map A.get))) w' := by
    intro f hf
    obtain ⟨hm, hk⟩ := List.mem_filter.mp hf
    refine hafter f ?_
    unfold Contracts.extAfter
    have : f ∈ A.facts.filter (Contracts.extKeeps e bits) :=
      List.mem_filter.mpr ⟨hm, extKeepsA_sound hab hk⟩
    split <;> first | exact List.mem_cons_of_mem _ this | exact this
  refine ⟨r, w', hcall, (hA1.forgetKind e.closes).worldP hkept (fun r hr => st.mem.2.1 r hr)
    (fun i k b hget x hl => st.held k x (AS.forgetKind_hand hget) hl), fun v hv => ?_⟩
  subst hv
  unfold extAnswer
  split
  · rename_i k hk
    obtain ⟨x, rfl⟩ := Contracts.lib_open_i64 hk hcall hb
    refine ⟨x, rfl, ?_⟩
    by_cases h0 : x = 0
    · exact .inl ⟨rfl, h0⟩
    · exact .inr (st.opens k x hk (by simp [asBits]) h0)
  · trivial

theorem callee_ffi {A B : AS} {f : Ffi} {args : List R} {av : AV} (h : A.callee (.ffi f) args = .ok (B, av)) :
    A.ffiCall f args = .ok B ∧ av = ffiAnswer A.facts f (args.map A.get) := by
  simp only [AS.callee] at h
  split at h
  · rename_i B' hB; cases h; exact ⟨hB, rfl⟩
  · cases h

theorem callee_ext {A B : AS} {e : Ext} {args : List R} {av : AV} (h : A.callee (.ext e) args = .ok (B, av)) :
    e.aside = true ∧ (args.map A.get).all (fun a => a.scTy.isSome) = true ∧
      preExt A.facts e (args.map A.get) = true ∧
      B = (A.forgetKind e.closes).withFacts (A.facts.filter (extKeepsA e (args.map A.get))) ∧
      av = extAnswer e := by
  simp only [AS.callee] at h
  split at h
  · rename_i hc
    cases h
    simp only [Bool.and_eq_true] at hc
    exact ⟨hc.1.1, hc.1.2, hc.2, rfl, rfl⟩
  · cases h

/-- **A call the checker accepts answers**, ends where the rule's state says,
    and its answer is the one the rule gives. -/
theorem callee_core {cfg : Cfg} {A B : AS} {c : Callee} {args : List R} {av : AV}
    (h : A.callee c args = .ok (B, av)) {Γ : Env} {w : World} {vs : List V} (hA : A.Holds vals Γ w)
    (hvs : args.mapM (fun r => Γ[r]?) = some vs) :
    ∃ r w', callOf cfg.locals c vs (obsCall w c vs) = some (r, w') ∧ B.Holds vals Γ w' ∧
      ∀ v, r = some v → av.Desc (Γ.push v) w' v := by
  cases c with
  | ffi f =>
      obtain ⟨hB, rfl⟩ := callee_ffi h
      obtain ⟨r, w', _bits, hcall, hA', -, hr⟩ := ffiCall_core hB hA hvs
      refine ⟨r, w', hcall, hA', fun v hv => ?_⟩
      obtain ⟨x, rfl, -, hdn⟩ := hr v hv
      unfold ffiAnswer
      split
      · rename_i n hn; exact hdn n hn
      · exact TyV.sc (ffi_result_lanes f)
  | ext e =>
      obtain ⟨ha, hsc, hp, rfl, rfl⟩ := callee_ext h
      exact extCall_core (cfg := cfg) ha hsc hp hA hvs
  | _ => simp [AS.callee] at h

theorem call_sound {cfg : Cfg} {A A' : AS} {c : Callee} {args : List R} {fo fa : Bool}
    (h : absStmt fa A (.call c args) = .ok A') : Stmt1 cfg (A.Holds vals) (.call c args) (A'.Holds vals) fo := by
  simp only [absStmt] at h
  split at h
  · rename_i B av hB
    split at h
    · cases h
      refine call_rule fun Γ w vs hA hvs => ?_
      obtain ⟨r, w', hcall, hA', hr⟩ := callee_core hB hA hvs
      exact ⟨r, w', hcall, fun v hv => hA'.push (hr v hv)⟩
    · cases h
  · cases h

theorem callVoid_sound {cfg : Cfg} {A A' : AS} {c : Callee} {args : List R} {fo fa : Bool}
    (h : absStmt fa A (.callVoid c args) = .ok A') : Stmt1 cfg (A.Holds vals) (.callVoid c args) (A'.Holds vals) fo := by
  simp only [absStmt] at h
  split at h
  · rename_i B av hB
    split at h
    · cases h
      refine callVoid_rule fun Γ w vs hA hvs => ?_
      obtain ⟨r, w', hcall, hA', -⟩ := callee_core hB hA hvs
      exact ⟨r, w', hcall, hA'⟩
    · cases h
  · cases h

/-- **A call's answer by cases**: where `answerCases` splits the state after a
    call, one of the two describes where it ends. -/
theorem answerCases_sound {cfg : Cfg} {A A' B1 B2 : AS} {s : Stmt} {fo fa : Bool}
    (h : absStmt fa A s = .ok A') (hc : answerCases A A' s = some (B1, B2)) :
    Stmt1 cfg (A.Holds vals) s (fun Γ w => B1.Holds vals Γ w ∨ B2.Holds vals Γ w) fo := by
  unfold answerCases at hc
  split at hc
  · rename_i b po dO fo' n
    split at hc
    · rename_i t0 sz hn
      split at hc
      · rename_i hs
        simp only [Option.some.injEq, Prod.mk.injEq] at hc
        obtain ⟨rfl, rfl⟩ := hc
        simp only [absStmt, answersAs, if_true] at h
        split at h
        · rename_i B av hB
          obtain ⟨hB', rfl⟩ := callee_ffi hB
          cases h
          refine call_rule fun Γ w vs hA hvs => ?_
          obtain ⟨r, w', bits, hcall, hA', hab, hr⟩ := ffiCall_core hB' hA hvs
          refine ⟨r, w', hcall, fun v hv => ?_⟩
          obtain ⟨x, rfl, hans, -⟩ := hr v hv
          have hP := hA'.push (av := .sc (Ffi.fileRead.result.getD .i64))
            (v := .sc (Ffi.fileRead.result.getD .i64) x) (TyV.sc (ffi_result_lanes Ffi.fileRead))
          have hlen : Γ.size = A.len := hA.1.1
          have hat : ∀ u, (Γ.push (.sc (Ffi.fileRead.result.getD .i64) x))[A.len]? = some u →
              u = .sc .i64 x := by
            intro u hu
            rw [← hlen, Array.getElem?_push_size] at hu
            cases hu; rfl
          -- the size asked for is the one the state knows
          simp only [List.map_cons, List.map_nil] at hab
          rcases bits with _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨_, _ | ⟨size, _ | ⟨_, _⟩⟩⟩⟩⟩⟩ <;>
            simp only [ArgBits, and_false] at hab
          obtain ⟨-, -, -, -, ⟨t, hd⟩, -⟩ := hab
          rw [hn] at hd
          simp only [AV.Desc, V.sc.injEq] at hd
          obtain ⟨-, rfl⟩ := hd
          simp only [answerOk] at hans
          rcases hans with h1 | h2
          · left
            refine hP.setAV fun u hu => ?_
            rw [hat u hu]
            show _ = _
            congr 1
            exact UInt64.toNat_inj.mp (by rw [h1]; rfl)
          · right
            refine hP.setAV fun u hu => ?_
            rw [hat u hu]
            exact ⟨x, rfl, Nat.zero_le _, h2 hs, Nat.one_dvd _, trivial⟩
        · cases h
      · cases hc
    · cases hc
  · cases hc

-- ---------------------------------------------------------------------------
-- Statements
-- ---------------------------------------------------------------------------

theorem relLoad_spec {A : AS} {o : Op} {av : AV} (h : relLoad A o = some av) :
    ∃ op a t, o = .load op a ∧ loadKindOk op = true ∧ (A.get a).scTy = some t ∧
      memRel A a (loadBytes op) = true ∧ av = loadAV op := by
  unfold relLoad at h
  split at h
  · rename_i op a
    split at h
    · rename_i hc
      cases h
      simp only [Bool.and_eq_true, Option.isSome_iff_exists] at hc
      obtain ⟨⟨hok, t, ht⟩, hm⟩ := hc
      exact ⟨op, a, t, rfl, hok, ht, hm, rfl⟩
    · cases h
  · cases h

theorem AS.setAV_par (A : AS) (r : R) (av : AV) : (A.setAV r av).par = A.par := by
  unfold AS.setAV; split <;> (try split) <;> rfl

theorem AS.tightenHi_par (A : AS) (r : R) (s : Nat) (k : Int) : (A.tightenHi r s k).par = A.par := by
  unfold AS.tightenHi; split
  · split <;> (try split) <;> simp only [AS.setAV_par]
  · rfl

theorem AS.addUb_par (A : AS) (a : R) (p : Nat × Lin) : (A.addUb a p).par = A.par := by
  unfold AS.addUb; split <;> rfl

theorem AS.foldl_par {α : Type} (g : AS → α → AS) (hg : ∀ C x, (g C x).par = C.par) :
    ∀ (l : List α) (B : AS), (l.foldl g B).par = B.par
  | [], _ => rfl
  | x :: l, B => by rw [List.foldl_cons, AS.foldl_par g hg l, hg]

theorem AS.pushOp_par (A : AS) (o : Op) (av : AV) : (A.pushOp o av).par = A.par := by
  unfold AS.pushOp
  rw [AS.foldl_par _ (fun D x => by obtain ⟨r, p⟩ := x; dsimp only; split <;> simp only [AS.tightenHi_par]),
    AS.foldl_par _ (fun C x => by obtain ⟨r, p⟩ := x; dsimp only; split <;> simp only [AS.addUb_par])]
  rfl

/-- **A number `parOp` computes from the parameters** is the operation's
    value. -/
theorem parOp_sound {A : AS} {Γ : Env} {w : World} {vals : List UInt64} (hA : A.Holds vals Γ w)
    (hP : A.par.At vals Γ w) {o : Op} {t : ClifTy} {e : PExpr} (he : parOp A o = some (t, e)) {v : V}
    (hv : evalOp w.mem Γ o = some v) : v = .sc t (e.eval vals) := by
  cases o with
  | load op a =>
      simp only [parOp] at he
      split at he
      · rename_i t0 x hk hty hkn
        obtain ⟨c, hc, he⟩ := Option.map_eq_some_iff.mp he
        cases he
        have hcm := List.mem_of_find?_eq_some hc
        have hcx : c.1 = x := by simpa using List.find?_some hc
        have hr := AS.get_lt_known (r := a) (by rw [hkn]; rfl)
        obtain ⟨u, hu, hd⟩ := hA.get hr
        rw [hkn] at hd; simp only [AV.Desc] at hd; subst hd
        have hload : w.mem.load x (tyBytes op.ty) = some (vals.getD c.2 0) := by
          rw [hty, ← hcx]; exact hP.2.2.1 _ (List.mem_append_left _ (List.mem_map_of_mem hcm))
        rw [evalOp_load hk (by rw [hty]; rfl) hu hload] at hv
        cases hv; rw [hty]; rfl
      · cases he
  | ireduce32 a =>
      simp only [parOp] at he
      split at he
      · rename_i e0 hs
        cases he
        have ha := hP.slotOf hs
        have hev : evalOp w.mem Γ (.ireduce32 a) = some (norm .i32 (e0.eval vals)) := by
          simp [evalOp, un, Sem.get, ha, ClifTy.isInt, ClifTy.width]
        rw [hev] at hv; cases hv; rfl
      · cases he
  | imul a b =>
      simp only [parOp] at he
      split at he
      · rename_i e0 c hs hb
        cases he
        have ha := hP.slotOf hs
        have hr := AS.get_lt_known (r := b) (by rw [hb]; rfl)
        obtain ⟨u, hu, hd⟩ := hA.get hr
        rw [hb] at hd; simp only [AV.Desc] at hd; subst hd
        have hev : evalOp w.mem Γ (.imul a b) = some (norm .i64 (e0.eval vals * c)) := by
          simp only [evalOp]; rw [bin_eval ha hu rfl]
        rw [hev, norm_i64] at hv; cases hv; rfl
      · cases he
  | _ => simp [parOp] at he

/-- **What the checker steps a statement to is true of it.** -/
theorem absStmt_sound {cfg : Cfg} {A A' : AS} {s : Stmt} {fo : Bool} (h : absStmt fo A s = .ok A') :
    Stmt1 cfg (A.Holds vals) s (A'.Holds vals) fo := by
  cases s with
  | op o =>
      simp only [absStmt] at h
      split at h
      · rename_i av hav
        cases h
        refine op_rule (fun Γ w v hA hv => ?_) (fun Γ w hA hn => ?_)
        · obtain ⟨v', hv', hd⟩ := absOp_sound hA hav
          rw [hv] at hv'; cases hv'
          have hB := hA.pushS (opAV_desc hA hv hd) (symOp_fits hA hv)
          have hC : (A.pushOp o av).Holds vals (Γ.push v) w :=
            foldl_tighten hB _ _ (foldl_addUb _ _ hB
              (fun q hq hok => ⟨opFacts_sound hA hv q hq, hok⟩) (fun q _ C hC => ubOk_len hC _))
              (fun q hq => opFacts_sound hA hv q hq)
          refine hC.parSlot fun t e he => ?_
          rw [← hA.1.1, Array.getElem?_push_size, parOp_sound hA hA.1.2.2 he hv]
        · obtain ⟨v', hv', -⟩ := absOp_sound hA hav
          rw [hn] at hv'; cases hv'
      · split at h
        · rename_i av hrl
          cases h
          obtain ⟨op, a, t, rfl, hok, ht, hm, rfl⟩ := relLoad_spec hrl
          refine op_rule (fun Γ w v hA hv => ?_) (fun Γ w hA hn => ?_)
          · obtain ⟨x, hx, -⟩ := hA.getSc ht
            obtain ⟨v', hv', hd⟩ := load_of_fits hok hx (memRel_fits hA hm hx)
            rw [hv] at hv'; cases hv'
            exact hA.push (hd _)
          · obtain ⟨x, hx, -⟩ := hA.getSc ht
            obtain ⟨v', hv', -⟩ := load_of_fits hok hx (memRel_fits hA hm hx)
            rw [hn] at hv'; cases hv'
        · split at h
          · rename_i hfo
            cases h
            subst hfo
            exact op_rule (fun Γ w v hA _ => hA.push (av := .any) trivial) (fun _ _ _ _ => rfl)
          · cases h
  | store ty v a => exact storeF_sound h
  | storeUnaligned v a => exact storeUF_sound h
  | istore8 v a =>
      simp only [absStmt] at h
      split at h
      · rename_i tv hsc; exact istore8F_sound hsc h
      · split at h
        · rename_i hfo; cases h; subst hfo; exact wild_istore8
        · cases h
  | call c args => exact call_sound h
  | callVoid c args => exact callVoid_sound h

theorem absStmtsH_sound {cfg : Cfg} {fo : Bool} :
    ∀ {hist : List (Nat × Stmt)} {A A' : AS} {ss : List Stmt}, absStmtsH fo hist A ss = .ok A' →
      Stmts cfg (A.Holds vals) ss (A'.Holds vals) fo
  | _, A, A', [], h => by
      simp only [absStmtsH, Except.ok.injEq] at h
      subst h
      exact fun _ _ hP => hP
  | _, A, A', s :: ss, h => by
      simp only [absStmtsH] at h
      split at h
      · rename_i A1 h1
        exact stmts_cons (absStmt_sound h1) (absStmtsH_sound h)
      · cases h

theorem absStmts_sound {cfg : Cfg} {A A' : AS} {ss : List Stmt} {fo : Bool} (h : absStmts fo A ss = .ok A') :
    Stmts cfg (A.Holds vals) ss (A'.Holds vals) fo :=
  absStmtsH_sound h

-- ---------------------------------------------------------------------------
-- Joins
-- ---------------------------------------------------------------------------

theorem ICmpCond.eq_of_beq {a b : ICmpCond} (h : (a == b) = true) : a = b := by
  revert h; cases a <;> cases b <;> first | (intro; rfl) | (intro h; exact absurd h (by decide))

theorem V.eq_of_beqv {a b : V} (h : V.beqv a b = true) : a = b := by
  cases a <;> cases b <;> simp_all [V.beqv]

theorem AV.eq_of_beq {a b : AV} (h : a.beq b = true) : a = b := by
  cases a <;> cases b <;> simp only [AV.beq, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq,
    Bool.false_eq_true, reduceCtorEq] at h ⊢
  · rw [V.eq_of_beqv h]
  · rw [h]
  · obtain ⟨⟨⟨rfl, rfl⟩, rfl⟩, rfl⟩ := h; rfl
  · obtain ⟨⟨h1, rfl⟩, rfl⟩ := h; rw [ICmpCond.eq_of_beq h1]
  · rw [h]
  · obtain ⟨rfl, rfl⟩ := h; rfl

theorem AV.tyJoin_left {Γ : Env} {w : World} {a b : AV} {v : V} (hd : a.Desc Γ w v) :
    (a.tyJoin b).Desc Γ w v := by
  unfold AV.tyJoin
  split
  · rename_i t t' ht ht'
    split
    · exact AV.ty_desc hd ht
    · trivial
  · trivial

theorem AV.tyJoin_right {Γ : Env} {w : World} {a b : AV} {v : V} (hd : b.Desc Γ w v) :
    (a.tyJoin b).Desc Γ w v := by
  unfold AV.tyJoin
  split
  · rename_i t t' ht ht'
    split
    · rename_i e; subst e; exact AV.ty_desc hd ht'
    · trivial
  · trivial

/-- The pointer two pointers into one region join to covers each. -/
theorem ptrJoin_desc {Γ : Env} {w : World} {r : Region} {n j : Num} {v : V}
    (hd : (AV.ptr r n).Desc Γ w v) (hj : ∀ z, n.Holds w z → j.Holds w z) (hs : j.hi ≤ span) :
    (AV.ptr r j).Desc Γ w v := by
  obtain ⟨off, rfl, hn, -⟩ := hd
  exact ⟨off, rfl, hj off hn, hs⟩

theorem AV.join_left {Γ : Env} {w : World} {a b : AV} {v : V} (hd : a.Desc Γ w v) :
    (a.join b).Desc Γ w v := by
  unfold AV.join
  split
  · exact hd
  · have hp := AV.asPtr_desc hd
    generalize a.asPtr = pa at hp
    generalize b.asPtr = pb
    split
    · rename_i r n r' n'
      split
      · rename_i hr
        subst hr
        split
        · rename_i j hjn
          split
          · rename_i hs
            exact ptrJoin_desc hp (fun z hz => (Num.join_left hz hjn).2) hs
          · obtain ⟨off, rfl, -⟩ := hp
            exact TyV.sc (t := .i64) rfl
        · obtain ⟨off, rfl, -⟩ := hp
          exact TyV.sc (t := .i64) rfl
      · obtain ⟨off, rfl, -⟩ := hp
        exact TyV.sc (t := .i64) rfl
    · split
      · rename_i x y hx hy
        split
        · rename_i n hn
          obtain ⟨z, rfl, hz⟩ := AV.num_desc hd hx
          obtain ⟨ht, hn'⟩ := Num.join_left hz hn
          exact ⟨z, by rw [ht], hn'⟩
        · exact AV.tyJoin_left hd
      · exact AV.tyJoin_left hd

theorem AV.join_right {Γ : Env} {w : World} {a b : AV} {v : V} (hd : b.Desc Γ w v) :
    (a.join b).Desc Γ w v := by
  unfold AV.join
  split
  · rename_i he; rw [AV.eq_of_beq he]; exact hd
  · have hp := AV.asPtr_desc hd
    generalize a.asPtr = pa
    generalize b.asPtr = pb at hp
    split
    · rename_i r n r' n'
      split
      · rename_i hr
        subst hr
        split
        · rename_i j hjn
          split
          · rename_i hs
            exact ptrJoin_desc hp (fun z hz => (Num.join_right hz hjn).2) hs
          · obtain ⟨off, rfl, -⟩ := hp
            exact TyV.sc (t := .i64) rfl
        · obtain ⟨off, rfl, -⟩ := hp
          exact TyV.sc (t := .i64) rfl
      · obtain ⟨off, rfl, -⟩ := hp
        exact TyV.sc (t := .i64) rfl
    · split
      · rename_i x y hx hy
        split
        · rename_i n hn
          obtain ⟨z, rfl, hz⟩ := AV.num_desc hd hy
          obtain ⟨ht, hn'⟩ := Num.join_right hz hn
          exact ⟨z, by rw [ht], hn'⟩
        · exact AV.tyJoin_right hd
      · exact AV.tyJoin_right hd

theorem meetFact_left {S' : TState} {x y : Fact} {w : World} (h : meetFact S' x = some y)
    (hx : x.holds w) : y.holds w := by
  unfold meetFact at h
  by_cases hc : S'.contains x = true
  · rw [if_pos hc] at h; cases h; exact hx
  · rw [if_neg hc] at h
    cases x with
    | htVals n =>
        obtain ⟨m, -, rfl⟩ := Option.map_eq_some_iff.mp h
        intro k v hv
        exact Nat.le_trans (hx k v hv) (Nat.le_max_left _ _)
    | readyAll p =>
        by_cases hr : readyIn S' p = true
        · simp only [hr, if_true, Option.some.injEq] at h; subst h; exact hx
        · simp only [hr, Bool.false_eq_true, if_false, reduceCtorEq] at h
    | _ => simp at h

theorem meetFact_right {S' : TState} {x y : Fact} {w : World} (h : meetFact S' x = some y)
    (hS : S'.holds w) : y.holds w := by
  unfold meetFact at h
  by_cases hc : S'.contains x = true
  · rw [if_pos hc] at h; cases h; exact hS _ (by simpa using hc)
  · rw [if_neg hc] at h
    cases x with
    | htVals n =>
        obtain ⟨m, hm, rfl⟩ := Option.map_eq_some_iff.mp h
        have hv := hS _ (htValsOf_mem hm)
        intro k v hg
        exact Nat.le_trans (hv k v hg) (Nat.le_max_right _ _)
    | readyAll p =>
        by_cases hr : readyIn S' p = true
        · simp only [hr, if_true, Option.some.injEq] at h; subst h; exact Contracts.readyIn_sound hS hr
        · simp only [hr, Bool.false_eq_true, if_false, reduceCtorEq] at h
    | _ => simp at h

theorem meetReady_left {S : TState} {x : Fact} {w : World} (h : meetReady S x = true) (hS : S.holds w) :
    x.holds w := by
  unfold meetReady at h
  split at h
  · simp only [Bool.and_eq_true] at h
    exact Contracts.readyIn_sound hS h.2
  · cases h

theorem factsMeet_left {S S' : TState} {w : World} (h : S.holds w) : (factsMeet S S').holds w := by
  intro y hy
  rcases List.mem_append.mp hy with hy | hy
  · obtain ⟨x, hx, hxy⟩ := List.mem_filterMap.mp hy
    exact meetFact_left hxy (h x hx)
  · exact meetReady_left (List.mem_filter.mp hy).2 h

theorem factsMeet_right {S S' : TState} {w : World} (h : S'.holds w) : (factsMeet S S').holds w := by
  intro y hy
  rcases List.mem_append.mp hy with hy | hy
  · obtain ⟨x, -, hxy⟩ := List.mem_filterMap.mp hy
    exact meetFact_right hxy h
  · exact h y (List.mem_filter.mp hy).1

theorem AS.join_holds {A B C : AS} (h : A.join B = .ok C) {Γ : Env} {w : World}
    (hAB : A.Holds vals Γ w ∨ B.Holds vals Γ w) : C.Holds vals Γ w := by
  unfold AS.join at h
  split at h
  · rename_i hl
    cases h
    have hs : Γ.size = A.len := by rcases hAB with hA | hB; exact hA.1.1; rw [hB.1.1, hl]
    refine ⟨⟨hs, fun i hi => ?_, ?_⟩, fun i hi => ?_, ?_⟩
    · show (Slots.zip Sym.join A.syms B.syms |>.get i).Fits Γ w i A.len
      rw [Slots.get_zip (by rfl)]
      rcases hAB with hA | hB
      · exact Sym.join_left (hA.1.2.1 i hi)
      · exact Sym.join_right (hl ▸ hB.1.2.1 i (hl ▸ hi))
    · rcases hAB with hA | hB
      · exact hA.1.2.2.join_left B.par
      · exact hB.1.2.2.join_right A.par
    · have hi' : i < B.len := hl ▸ hi
      have e : AS.get ⟨Slots.zip AV.join A.slots B.slots, A.len, factsMeet A.facts B.facts,
          Slots.zip Sym.join A.syms B.syms, A.par.join B.par⟩ i
          = (A.get i).join (B.get i) := by
        unfold AS.get
        rw [if_pos hi, if_pos hi, if_pos hi', Slots.get_zip (by rfl)]
      rw [e]
      rcases hAB with hA | hB
      · obtain ⟨v, hv, hd⟩ := hA.get hi
        exact ⟨v, hv, AV.join_left hd⟩
      · obtain ⟨v, hv, hd⟩ := hB.get hi'
        exact ⟨v, hv, AV.join_right hd⟩
    · rcases hAB with hA | hB
      · exact factsMeet_left hA.2.2
      · exact factsMeet_right hB.2.2
  · cases h

theorem joinOpt_holds {a b : Option AS} {r : Option AS} (h : joinOpt a b = .ok r) {Γ : Env} {w : World}
    (hab : (∃ A, a = some A ∧ A.Holds vals Γ w) ∨ (∃ B, b = some B ∧ B.Holds vals Γ w)) :
    ∃ C, r = some C ∧ C.Holds vals Γ w := by
  cases a with
  | none =>
      simp only [joinOpt, Except.ok.injEq] at h; subst h
      rcases hab with ⟨A, e, -⟩ | hb
      · cases e
      · exact hb
  | some A =>
      cases b with
      | none =>
          simp only [joinOpt, Except.ok.injEq] at h; subst h
          rcases hab with ha | ⟨B, e, -⟩
          · exact ha
          · cases e
      | some B =>
          simp only [joinOpt, bind, Except.bind] at h
          split at h
          · cases h
          · rename_i C hC
            simp only [pure, Except.pure, Except.ok.injEq] at h; subst h
            refine ⟨C, rfl, AS.join_holds hC ?_⟩
            rcases hab with ⟨A', e, hA⟩ | ⟨B', e, hB⟩
            · cases e; exact .inl hA
            · cases e; exact .inr hB

theorem joinAll_holds : ∀ {As : List AS} {r : Option AS}, joinAll As = .ok r → ∀ {A : AS}, A ∈ As →
    ∀ {Γ : Env} {w : World}, A.Holds vals Γ w → ∃ C, r = some C ∧ C.Holds vals Γ w
  | [], _, _, _, hA, _, _, _ => by cases hA
  | A0 :: As, r, h, A, hA, Γ, w, hH => by
      simp only [joinAll, bind, Except.bind] at h
      split at h
      · cases h
      · rename_i o ho
        cases o with
        | none =>
            simp only [pure, Except.pure, Except.ok.injEq] at h; subst h
            rcases List.mem_cons.mp hA with rfl | hA
            · exact ⟨_, rfl, hH⟩
            · obtain ⟨C, e, -⟩ := joinAll_holds ho hA hH; cases e
        | some B =>
            dsimp only at h
            split at h
            · cases h
            · rename_i C hC
              simp only [pure, Except.pure, Except.ok.injEq] at h; subst h
              refine ⟨C, rfl, AS.join_holds hC ?_⟩
              rcases List.mem_cons.mp hA with rfl | hA
              · exact .inl hH
              · obtain ⟨B', e, hB⟩ := joinAll_holds ho hA hH
                cases e; exact .inr hB

-- ---------------------------------------------------------------------------
-- Binding at a slot count
-- ---------------------------------------------------------------------------

theorem DescAll.length {Γ : Env} {w : World} :
    ∀ {as : List AV} {vs : List V}, DescAll Γ w as vs → as.length = vs.length
  | [], [], _ => rfl
  | _ :: _, _ :: _, ⟨_, h⟩ => by simp [DescAll.length h]
  | [], _ :: _, h => h.elim
  | _ :: _, [], h => h.elim

theorem DescAll.get {Γ : Env} {w : World} :
    ∀ {as : List AV} {vs : List V}, DescAll Γ w as vs → ∀ {j : Nat}, j < as.length →
      ∃ v, vs[j]? = some v ∧ (as.getD j .any).Desc Γ w v
  | _ :: _, v :: _, ⟨hd, _⟩, 0, _ => ⟨v, rfl, hd⟩
  | _ :: _, _ :: _, ⟨_, hds⟩, j + 1, hj => by
      simpa using DescAll.get hds (j := j) (by simp at hj; omega)
  | [], _, _, _, hj => by simp at hj
  | _ :: _, [], h, _, _ => h.elim

theorem DescAll.portable {Γ Γ' : Env} {w : World} :
    ∀ {as : List AV} {vs : List V}, DescAll Γ w as vs → DescAll Γ' w (as.map AV.portable) vs
  | [], [], _ => trivial
  | _ :: _, _ :: _, ⟨hd, hds⟩ => ⟨AV.portable_desc hd, DescAll.portable hds⟩
  | [], _ :: _, h => h.elim
  | _ :: _, [], h => h.elim

theorem AS.pushAll_cons (B : AS) (x : AV × Sym) (xs : List (AV × Sym)) :
    B.pushAll (x :: xs) = (B.pushS x.1 x.2).pushAll xs := rfl

theorem AS.pushAll_len : ∀ (xs : List (AV × Sym)) (B : AS), (B.pushAll xs).len = B.len + xs.length
  | [], _ => rfl
  | x :: xs, B => by
      rw [AS.pushAll_cons, AS.pushAll_len xs]
      simp only [AS.pushS, List.length_cons]
      omega

theorem AS.pushAll_facts : ∀ (xs : List (AV × Sym)) (B : AS), (B.pushAll xs).facts = B.facts
  | [], _ => rfl
  | x :: xs, B => by rw [AS.pushAll_cons, AS.pushAll_facts xs]; rfl

theorem AS.pushAll_par : ∀ (xs : List (AV × Sym)) (B : AS), (B.pushAll xs).par = B.par
  | [], _ => rfl
  | x :: xs, B => by rw [AS.pushAll_cons, AS.pushAll_par xs]; rfl

theorem AS.pushAll_syms : ∀ (xs : List (AV × Sym)) (B : AS) (i : Nat), i < B.len + xs.length →
    (B.pushAll xs).syms.get i = if i < B.len then B.syms.get i else (xs.getD (i - B.len) (.any, {})).2
  | [], B, i, hi => by
      simp only [List.length_nil, Nat.add_zero] at hi
      simp only [AS.pushAll, List.foldl_nil, if_pos hi]
  | x :: xs, B, i, hi => by
      have hi' : i < (B.pushS x.1 x.2).len + xs.length := by
        simp only [AS.pushS, List.length_cons] at hi ⊢; omega
      rw [AS.pushAll_cons, AS.pushAll_syms xs _ i hi']
      have hl : (B.pushS x.1 x.2).len = B.len + 1 := rfl
      rw [hl]
      show (if i < B.len + 1 then (B.syms.set B.len x.2).get i else _) = _
      rw [Slots.get_set]
      by_cases h1 : i < B.len
      · rw [if_pos (by omega), if_pos h1, if_neg (by omega)]
      · by_cases h2 : i = B.len
        · subst h2; rw [if_pos (by omega), if_pos rfl, if_neg h1, Nat.sub_self]; rfl
        · rw [if_neg (by omega), if_neg h1]
          obtain ⟨k, hk⟩ : ∃ k, i - B.len = k + 1 := ⟨i - B.len - 1, by omega⟩
          rw [hk, show i - (B.len + 1) = k by omega]
          rfl

theorem AS.pushAll_get : ∀ (xs : List (AV × Sym)) (B : AS) (i : Nat),
    (B.pushAll xs).get i = if i < B.len then B.get i else (xs.getD (i - B.len) (.any, {})).1
  | [], B, i => by
      simp only [AS.pushAll, List.foldl_nil, List.getD_nil]
      split
      · rfl
      · rename_i h; exact AS.get_out h
  | x :: xs, B, i => by
      rw [AS.pushAll_cons, AS.pushAll_get xs _ i]
      have hl : (B.pushS x.1 x.2).len = B.len + 1 := rfl
      rw [hl]
      by_cases h1 : i < B.len
      · rw [if_pos (by omega), if_pos h1, AS.get_pushS_lt h1]
      · by_cases h2 : i = B.len
        · subst h2
          rw [if_pos (by omega), if_neg h1, AS.get_pushS_eq, Nat.sub_self]
          rfl
        · rw [if_neg (by omega), if_neg h1]
          obtain ⟨k, hk⟩ : ∃ k, i - B.len = k + 1 := ⟨i - B.len - 1, by omega⟩
          rw [hk, show i - (B.len + 1) = k by omega]
          rfl

theorem Slots.get_clear {α : Type} [Blank α] : ∀ (k : Nat) (t : Slots α) (i j : Nat),
    (t.clear i k).get j = if i ≤ j ∧ j < i + k then Blank.blank else t.get j
  | 0, t, i, j => by
      simp only [Slots.clear]
      rw [if_neg (by omega)]
  | k + 1, t, i, j => by
      simp only [Slots.clear]
      rw [Slots.get_clear k (t.set i Blank.blank) (i + 1) j, Slots.get_set]
      by_cases h1 : i + 1 ≤ j ∧ j < i + 1 + k
      · rw [if_pos h1, if_pos (by omega)]
      · rw [if_neg h1]
        by_cases h2 : j = i
        · rw [if_pos h2, if_pos (by omega)]
        · rw [if_neg h2, if_neg (by omega)]

theorem AS.bindAt_len (A : AS) (n : Nat) (vs : List AV) (ss : List Sym) :
    (A.bindAt n vs ss).len = n + vs.length := by
  unfold AS.bindAt
  rw [AS.pushAll_len, List.length_mapIdx]
  split <;> rfl

theorem AS.bindAt_facts (A : AS) (n : Nat) (vs : List AV) (ss : List Sym) :
    (A.bindAt n vs ss).facts = A.facts := by
  unfold AS.bindAt
  rw [AS.pushAll_facts]
  split <;> rfl

theorem AS.bindAt_par {A : AS} {n : Nat} {vs : List AV} {ss : List Sym} :
    (A.bindAt n vs ss).par = if A.len ≤ n then A.par else A.par.cut n := by
  unfold AS.bindAt
  rw [AS.pushAll_par]
  split <;> rfl

theorem AS.bindAt_get_lt {A : AS} {n i : Nat} {vs : List AV} {ss : List Sym} (hi : i < n) :
    (A.bindAt n vs ss).get i = if A.len ≤ n then A.get i else AV.cut n (A.get i) := by
  unfold AS.bindAt
  rw [AS.pushAll_get]
  split
  · rename_i hle
    have e : AS.get ⟨A.slots.clear A.len (n - A.len), n, A.facts, A.syms.clear A.len (n - A.len), A.par⟩ i
        = A.get i := by
      unfold AS.get
      simp only
      rw [if_pos hi, Slots.get_clear]
      by_cases h : i < A.len
      · rw [if_neg (by omega), if_pos h]
      · rw [if_pos (by omega), if_neg h]; rfl
    first | exact e | (rw [if_pos (show i < n from hi)]; exact e)
  · rename_i hle
    have e : AS.get ⟨A.slots.map (AV.cut n), n, A.facts, A.syms.map (Sym.cut n), A.par.cut n⟩ i = AV.cut n (A.get i) := by
      unfold AS.get
      simp only
      rw [if_pos hi, Slots.get_map (by rfl), if_pos (show i < A.len by omega)]
    first | exact e | (rw [if_pos (show i < n from hi)]; exact e)

theorem getD_out {α : Type} {l : List α} {d : α} {j : Nat} (h : l.length ≤ j) : l.getD j d = d := by
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_none h]

theorem getD_mapIdx_pair {vs : List AV} {ss : List Sym} {j : Nat} :
    (vs.mapIdx fun i v => (v.portable, ss.getD i {})).getD j (.any, {}) =
      if j < vs.length then ((vs.getD j .any).portable, ss.getD j {}) else (.any, {}) := by
  split
  · rename_i hj
    simp [List.getD_eq_getElem?_getD, List.getElem?_mapIdx, List.getElem?_eq_getElem hj]
  · rename_i hj
    exact getD_out (by simp; omega)

theorem AS.bindAt_get_ge {A : AS} {n j : Nat} {vs : List AV} {ss : List Sym} :
    (A.bindAt n vs ss).get (n + j) = (vs.getD j .any).portable := by
  unfold AS.bindAt
  rw [AS.pushAll_get]
  have hb : ∀ B : AS, B.len = n → (if n + j < B.len then B.get (n + j)
      else ((vs.mapIdx fun i v => (v.portable, ss.getD i {})).getD (n + j - B.len) (.any, {})).1) =
        (vs.getD j .any).portable := by
    intro B hB
    rw [hB, if_neg (by omega), Nat.add_sub_cancel_left, getD_mapIdx_pair]
    split
    · rfl
    · rename_i hj; rw [getD_out (by omega)]; rfl
  split
  · exact hb _ rfl
  · exact hb _ rfl

/-- A description read in an environment that keeps every slot it had. -/
theorem AV.desc_extend {Γ Γ' : Env} {w : World} {av : AV} {v : V} (hd : av.Desc Γ w v)
    (hΓ : ∀ a, a < Γ.size → Γ'[a]? = Γ[a]?) : av.Desc Γ' w v := by
  refine AV.desc_transport hd (fun a ha => ?_) (fun _ _ => rfl) (fun _ _ _ _ h => h)
  cases av with
  | cmp c x y =>
      obtain ⟨_, _, _, hx, hy, -⟩ := hd
      simp only [AV.refs, List.mem_cons, List.not_mem_nil, or_false] at ha
      have hlt : ∀ r t z, Γ[r]? = some (V.sc t z) → r < Γ.size := fun r t z h => by
        rcases Nat.lt_or_ge r Γ.size with h' | h'
        · exact h'
        · rw [Array.getElem?_eq_none h'] at h; cases h
      rcases ha with rfl | rfl
      · exact hΓ _ (hlt _ _ _ hx)
      · exact hΓ _ (hlt _ _ _ hy)
  | _ => simp [AV.refs] at ha

theorem AV.cut_desc {Γ Γ' : Env} {w : World} {av : AV} {v : V} {n : Nat} (hd : av.Desc Γ w v)
    (hΓ : ∀ a, a < n → a < Γ.size → Γ'[a]? = Γ[a]?) : (av.cut n).Desc Γ' w v := by
  cases av with
  | cmp c a b =>
      simp only [AV.cut]
      split
      · rename_i hab
        obtain ⟨t, x, y, ha, hb, ht, rfl⟩ := hd
        have hla : a < Γ.size := by
          rcases Nat.lt_or_ge a Γ.size with h | h
          · exact h
          · rw [Array.getElem?_eq_none h] at ha; cases ha
        have hlb : b < Γ.size := by
          rcases Nat.lt_or_ge b Γ.size with h | h
          · exact h
          · rw [Array.getElem?_eq_none h] at hb; cases hb
        exact ⟨t, x, y, by rw [hΓ a hab.1 hla]; exact ha, by rw [hΓ b hab.2 hlb]; exact hb, ht, rfl⟩
      · obtain ⟨_, _, _, -, -, -, rfl⟩ := hd
        exact TyV.sc (t := .i8) rfl
  | _ =>
      have := AV.desc_transport (Γ' := Γ') hd (fun _ h => by simp [AV.refs] at h) (fun _ _ => rfl) (fun _ _ _ _ h => h)
      exact this

theorem bindAt_size' (Γ : Env) (n : Nat) (vs : List V) : (bindAt Γ n vs).size = n + vs.length := by
  simp only [bindAt, Array.size_append, Array.take_eq_extract, Array.size_extract, Array.size_replicate,
    List.size_toArray]
  omega

theorem AS.bindAt_syms {A : AS} {n i : Nat} {vs : List AV} {ss : List Sym} (hi : i < n + vs.length) :
    (A.bindAt n vs ss).syms.get i =
      if i < n then (if A.len ≤ n then (if i < A.len then A.syms.get i else {}) else Sym.cut n (A.syms.get i))
      else ss.getD (i - n) {} := by
  unfold AS.bindAt
  rw [AS.pushAll_syms _ _ _ (by split <;> simpa using hi)]
  have tail : ¬ i < n → ((vs.mapIdx fun i v => (v.portable, ss.getD i {})).getD (i - n) (.any, {})).2 =
      ss.getD (i - n) {} := fun hn => by
    rw [getD_mapIdx_pair, if_pos (by omega)]
  split
  · rename_i hle
    show (if i < n then (A.syms.clear A.len (n - A.len)).get i else _) = _
    split
    · rw [Slots.get_clear]
      by_cases h : i < A.len
      · rw [if_neg (by omega), if_pos h]
      · rw [if_pos (by omega), if_neg h]; rfl
    · rename_i hn; exact tail hn
  · rename_i hle
    show (if i < n then (A.syms.map (Sym.cut n)).get i else _) = _
    split
    · rw [Slots.get_map (by rfl)]
    · rename_i hn; exact tail hn

theorem slotVal_bindAt {Γ : Env} {n x : Nat} {vs : List V} (hx : x < n) (hs : x < Γ.size) :
    slotVal (Sem.bindAt Γ n vs) x = slotVal Γ x := by
  unfold slotVal; rw [Hoare.bindAt_lt hx hs]

/-- **Binding, abstractly and concretely**: `AS.bindAt` describes `bindAt`. -/
theorem AS.Holds.bindAtS {A : AS} {Γ : Env} {w : World} (hA : A.Holds vals Γ w) (n : Nat) {avs : List AV}
    {vs : List V} {ss : List Sym} (hd : DescAll Γ w avs vs)
    (hss : ∀ j, j < avs.length → (ss.getD j {}).Fits (Sem.bindAt Γ n vs) w (n + j) (n + avs.length)) :
    (A.bindAt n avs ss).Holds vals (Sem.bindAt Γ n vs) w := by
  have hlen := hd.length
  refine ⟨⟨by rw [bindAt_size', AS.bindAt_len, hlen], fun i hi => ?_, ?_⟩, fun i hi => ?_,
    by rw [AS.bindAt_facts]; exact hA.2.2⟩
  · rw [AS.bindAt_len] at hi
    rw [AS.bindAt_syms (by simpa using hi), AS.bindAt_len]
    have hs := hA.1.1
    split
    · rename_i hn
      split
      · rename_i hle
        split
        · rename_i hia
          exact Sym.fits_congr (hA.1.2.1 i hia) (fun x hx => slotVal_bindAt (by omega) (by omega))
            (slotVal_bindAt hn (by omega)) (fun _ _ => rfl) (by omega)
        · exact Sym.fits_blank _ _ _ _
      · rename_i hle
        exact Sym.cut_fits (hA.1.2.1 i (by omega)) (fun x hx => slotVal_bindAt hx (by omega))
          (slotVal_bindAt hn (by omega)) (by omega)
    · rename_i hn
      obtain ⟨j, rfl⟩ : ∃ j, i = n + j := ⟨i - n, by omega⟩
      rw [Nat.add_sub_cancel_left]
      exact hss j (by omega)
  · rw [AS.bindAt_par]
    have hs := hA.1.1
    split
    · exact hA.1.2.2.extend fun r hr => Hoare.bindAt_lt (by omega) hr
    · exact hA.1.2.2.cut fun r hr => Hoare.bindAt_lt hr (by omega)
  rw [AS.bindAt_len] at hi
  by_cases hn : i < n
  · rw [AS.bindAt_get_lt hn, bindAt_get?, if_pos hn]
    by_cases hs : i < Γ.size
    · rw [if_pos hs]
      obtain ⟨v, hv, hdv⟩ := hA.get (hA.1.1 ▸ hs)
      refine ⟨v, hv, ?_⟩
      split
      · rename_i hle
        exact AV.desc_extend hdv fun a has => Hoare.bindAt_lt (by have := hA.1.1; omega) has
      · exact AV.cut_desc hdv fun a ha has => Hoare.bindAt_lt ha has
    · rw [if_neg hs, AS.get_out (by rw [← hA.1.1]; exact hs)]
      exact ⟨_, rfl, by split <;> trivial⟩
  · obtain ⟨j, rfl⟩ : ∃ j, i = n + j := ⟨i - n, by omega⟩
    rw [AS.bindAt_get_ge, Hoare.bindAt_at]
    obtain ⟨v, hv, hdv⟩ := hd.get (j := j) (by omega)
    exact ⟨v, hv, AV.portable_desc hdv⟩

-- ---------------------------------------------------------------------------
-- What a run keeps: the slots it started with, and every region's size
-- ---------------------------------------------------------------------------

/-- `Γ` extends `Γ0`: every slot of `Γ0` is still there, unchanged. -/
def Ext (Γ0 Γ : Env) : Prop := Γ0.size ≤ Γ.size ∧ ∀ i, i < Γ0.size → Γ[i]? = Γ0[i]?


theorem AS.Holds.bindAt {A : AS} {Γ : Env} {w : World} (hA : A.Holds vals Γ w) (n : Nat) {avs : List AV}
    {vs : List V} (hd : DescAll Γ w avs vs) : (A.bindAt n avs).Holds vals (Sem.bindAt Γ n vs) w :=
  hA.bindAtS n hd fun _ _ => by simp only [List.getD_nil]; exact Sym.fits_blank _ _ _ _

theorem Ext.refl (Γ : Env) : Ext Γ Γ := ⟨Nat.le_refl _, fun _ _ => rfl⟩

theorem Ext.trans {Γ0 Γ1 Γ2 : Env} (h1 : Ext Γ0 Γ1) (h2 : Ext Γ1 Γ2) : Ext Γ0 Γ2 :=
  ⟨Nat.le_trans h1.1 h2.1, fun i hi => (h2.2 i (by have := h1.1; omega)).trans (h1.2 i hi)⟩

theorem Ext.push (Γ : Env) (v : V) : Ext Γ (Γ.push v) :=
  ⟨by simp, fun i hi => by rw [Array.getElem?_push, if_neg (by omega)]⟩

theorem Ext.bindAt {Γ0 Γ : Env} (h : Ext Γ0 Γ) {n : Nat} (hn : Γ0.size ≤ n) (vs : List V) :
    Ext Γ0 (bindAt Γ n vs) :=
  ⟨by rw [bindAt_size']; omega, fun i hi => by
    rw [Hoare.bindAt_lt (by omega) (by have := h.1; omega)]; exact h.2 i hi⟩

theorem Ext.get {Γ0 Γ : Env} (h : Ext Γ0 Γ) {i : Nat} {v : V} (hv : Γ0[i]? = some v) : Γ[i]? = some v := by
  have : i < Γ0.size := by
    rcases Nat.lt_or_ge i Γ0.size with h' | h'
    · exact h'
    · rw [Array.getElem?_eq_none h'] at hv; cases hv
  rw [h.2 i this, hv]

open AlgorithmLib.HProg.Contracts (KeepsRooms) in
theorem KeepsRooms.refl (w : World) : KeepsRooms w w := fun _ _ => rfl

open AlgorithmLib.HProg.Contracts (KeepsRooms) in
theorem KeepsRooms.trans {w0 w1 w2 : World} (h1 : KeepsRooms w0 w1) (h2 : KeepsRooms w1 w2) :
    KeepsRooms w0 w2 := fun r hr => (h2 r hr).trans (h1 r hr)

open AlgorithmLib.HProg.Contracts (KeepsRooms) in
/-- From `w0` to `w` every region keeps its size, and the calls made are ones
    `t` admits. -/
structure Kept (t : Tr) (w0 w : World) : Prop where
  rooms : KeepsRooms w0 w
  trace : Traced t w0 w

theorem Kept.refl (w : World) : Kept (.lit []) w w := ⟨KeepsRooms.refl w, Traced.refl w⟩

theorem Kept.starNil {a : Tr} (w : World) : Kept (.star a) w w := ⟨KeepsRooms.refl w, [], rfl, .starNil⟩

theorem Kept.mono {a b : Tr} {w0 w : World} (h : Kept a w0 w) (hab : ∀ s, a.Has s → b.Has s) : Kept b w0 w :=
  ⟨h.rooms, h.trace.mono hab⟩

theorem Kept.seq {a b : Tr} {w0 w1 w2 : World} (h1 : Kept a w0 w1) (h2 : Kept b w1 w2) :
    Kept (.seq a b) w0 w2 :=
  ⟨KeepsRooms.trans h1.rooms h2.rooms, h1.trace.seq h2.trace⟩

theorem Kept.starSnoc {a : Tr} {w0 w1 w2 : World} (h1 : Kept (.star a) w0 w1) (h2 : Kept a w1 w2) :
    Kept (.star a) w0 w2 :=
  ⟨KeepsRooms.trans h1.rooms h2.rooms, h1.trace.starSnoc h2.trace⟩

theorem Kept.altL {a b : Tr} {w0 w : World} (h : Kept a w0 w) : Kept (.alt a b) w0 w :=
  h.mono fun _ hs => .altL hs

theorem Kept.altR {a b : Tr} {w0 w : World} (h : Kept b w0 w) : Kept (.alt a b) w0 w :=
  h.mono fun _ hs => .altR hs

theorem Kept.any {a : Tr} {w0 w : World} (h : Kept a w0 w) : Kept .any w0 w := h.mono fun s _ => .any s

theorem Kept.repZero {a : Tr} (w : World) : Kept (.rep 0 a) w w := ⟨KeepsRooms.refl w, Traced.repZero w⟩

theorem Kept.repSucc {a : Tr} {n : Nat} {w0 w1 w2 : World} (h1 : Kept (.rep n a) w0 w1) (h2 : Kept a w1 w2) :
    Kept (.rep (n + 1) a) w0 w2 :=
  ⟨KeepsRooms.trans h1.rooms h2.rooms, h1.trace.repSucc h2.trace⟩

/-- A straight run's calls, as its piece records them or leaves them open. -/
theorem Kept.straight {cs : List CallRec} {w0 w : World} {b : Bool}
    (h : Kept (.seq (.lit []) (.lit cs)) w0 w) : Kept (if b then .lit cs else .any) w0 w :=
  h.mono fun s hs => by
    cases hs with
    | seq h1 h2 =>
        rw [h1.lit_nil, List.nil_append]
        cases b
        · exact .any _
        · exact h2

/-- A first piece's calls and then the rest's, in the pattern a region gives
    the two. -/
theorem Kept.then {a b : Tr} {w0 w1 w2 : World} {c : Bool} (h1 : Kept a w0 w1) (h2 : Kept b w1 w2) :
    Kept (if c then .seq a b else .seq a (.alt (.lit []) b)) w0 w2 :=
  (h1.seq h2).mono fun s hs => by
    cases c
    · cases hs with
      | seq x y => exact .seq x (.altR y)
    · exact hs

/-- A first piece's calls alone, where it may leave the region. -/
theorem Kept.alone {a b : Tr} {w0 w : World} {c : Bool} (hc : c = false) (h : Kept a w0 w) :
    Kept (if c then .seq a b else .seq a (.alt (.lit []) b)) w0 w :=
  h.mono fun s hs => by
    subst hc
    have := Tr.Has.seq hs (Tr.Has.altL (b := b) .nil)
    rw [List.append_nil] at this
    exact this

theorem notEmpty_left {α β : Type} {l : List α} {m : List β} {x : α} (h : x ∈ l) :
    (l.isEmpty && m.isEmpty) = false := by
  cases l with
  | nil => cases h
  | cons _ _ => rfl

theorem notEmpty_right {α β : Type} {l : List α} {m : List β} {x : β} (h : x ∈ m) :
    (l.isEmpty && m.isEmpty) = false := by
  cases m with
  | nil => cases h
  | cons _ _ => cases l <;> rfl

open AlgorithmLib.HProg.Contracts (KeepsRooms) in
/-- A step that makes no call. -/
theorem Kept.same {w w' : World} (hs : w'.mem.sizes = w.mem.sizes) (ho : callsOf w'.obs = callsOf w.obs) :
    Kept (.lit []) w w' :=
  ⟨fun r _ => congrFun hs r, [], ho, .nil⟩

open AlgorithmLib.HProg.Contracts (KeepsRooms) in
/-- A step that makes one call, observed with arguments a record describes. -/
theorem Kept.call {w w' : World} {c : Callee} {vs : List V} {os : List (Option UInt64)} (hk : KeepsRooms w w')
    (ho : w'.obs = .call c vs :: w.obs) (ha : argsOk os vs) : Kept (.lit [⟨c, os⟩]) w w' :=
  ⟨hk, [(c, vs)], by rw [ho]; rfl, .lit ⟨⟨rfl, ha⟩, trivial⟩⟩

theorem stmtsSlots_ge (n : Nat) (ss : List Stmt) : n ≤ stmtsSlots n ss := by
  unfold stmtsSlots
  induction ss generalizing n with
  | nil => exact Nat.le_refl _
  | cons s ss ih =>
      rw [List.foldl_cons]
      refine Nat.le_trans ?_ (ih _)
      split <;> omega

theorem slotsGo_ge : ∀ (f n : Nat) (c : Code), n ≤ slotsGo f n c
  | 0, n, _ => by simp [slotsGo]
  | _ + 1, n, [] => by simp [slotsGo]
  | f + 1, n, .straight ss :: ps => by
      simp only [slotsGo]
      exact Nat.le_trans (stmtsSlots_ge n ss) (slotsGo_ge f _ ps)
  | f + 1, n, .loop l pre body :: ps => by
      simp only [slotsGo]
      have := slotsGo_ge f (n + l.pTys.length) pre
      have := slotsGo_ge f (slotsGo f (n + l.pTys.length) pre + l.pTys.length) body
      have := slotsGo_ge f (slotsGo f (slotsGo f (n + l.pTys.length) pre + l.pTys.length) body +
        l.exitTys.length) ps
      omega
  | f + 1, n, .ite m thn els _ _ :: ps => by
      simp only [slotsGo]
      have := slotsGo_ge f n thn
      have := slotsGo_ge f (slotsGo f n thn) els
      have := slotsGo_ge f (slotsGo f (slotsGo f n thn) els +
        (if termsGo f thn && termsGo f els then 0 else m.jTys.length)) ps
      omega
  | f + 1, n, .dloop l body :: ps => by
      simp only [slotsGo]
      have := slotsGo_ge f (n + l.pTys.length) body
      have := slotsGo_ge f (slotsGo f (n + l.pTys.length) body + l.exitTys.length) ps
      omega
  | f + 1, n, .br _ _ :: ps => by simp only [slotsGo]; exact slotsGo_ge f n ps
  | f + 1, n, .cont _ _ :: ps => by simp only [slotsGo]; exact slotsGo_ge f n ps

theorem slotsOf_ge (n : Nat) (c : Code) : n ≤ slotsOf n c := slotsGo_ge _ n c

-- ---------------------------------------------------------------------------
-- What a branch learns
-- ---------------------------------------------------------------------------

theorem signed_small {t : ClifTy} (ht : t.isInt = true) {x : UInt64} (hx : x.toNat < 2 ^ (t.width - 1)) :
    signed t x = x.toNat := by
  have hm := mask_toNat ht x
  have hw : 1 ≤ t.width ∧ t.width ≤ 64 := by
    cases t <;> first | exact ⟨by decide, by decide⟩ | cases ht
  have hx' : x.toNat < 2 ^ t.width :=
    Nat.lt_of_lt_of_le hx (Nat.pow_le_pow_right (by decide) (by omega))
  rw [Nat.mod_eq_of_lt hx'] at hm
  have hh : ((1 : UInt64) <<< UInt64.ofNat (t.width - 1)).toNat = 2 ^ (t.width - 1) := by
    cases t <;> first | decide | cases ht
  have h63 : (9223372036854775808 : UInt64).toNat = 2 ^ 63 := by decide
  unfold signed
  dsimp only
  split
  · rename_i h64
    rw [if_neg, hm]
    simp only [ge_iff_le, UInt64.le_iff_toNat_le, hm, h63]
    have e : t.width - 1 = 63 := by omega
    rw [e] at hx
    omega
  · rename_i h64
    rw [if_neg, hm]
    simp only [ge_iff_le, UInt64.le_iff_toNat_le, hm, hh]
    omega

/-- Where both operands are not negative, the unsigned comparison
    `unsignedCond` names agrees with the signed one. -/
theorem unsignedCond_sound {A : AS} {Γ : Env} {w : World} (hA : A.Holds vals Γ w) {c : ICmpCond} {x y : R}
    {t : ClifTy} {a b : UInt64} (hx : Γ[x]? = some (.sc t a)) (hy : Γ[y]? = some (.sc t b))
    (ht : t.isInt = true) : cmpInt (unsignedCond A c x y) t a b = cmpInt c t a b := by
  unfold unsignedCond
  split
  all_goals first
    | rfl
    | (rename_i nx ny hnx hny
       split
       · rename_i hsm
         obtain ⟨a', ea, hxa⟩ := AV.num_desc (hA.desc hx) hnx
         obtain ⟨b', eb, hyb⟩ := AV.num_desc (hA.desc hy) hny
         simp only [V.sc.injEq] at ea eb
         obtain ⟨rfl, rfl⟩ := ea
         obtain ⟨e2, rfl⟩ := eb
         have sa := signed_small ht (x := a) (by have := hxa.2.1; have := hsm.1; omega)
         have sb := signed_small ht (x := b) (by rw [e2]; have := hyb.2.1; have := hsm.2; omega)
         simp only [cmpInt, sa, sb, Int.ofNat_lt, Int.ofNat_le, UInt64.lt_iff_toNat_lt,
           UInt64.le_iff_toNat_le, gt_iff_lt, ge_iff_le]
       · rfl)

/-- What a relation between two slots' numbers says. -/
def RelHolds : ICmpCond → Nat → Nat → Prop
  | .ult, a, b => a < b
  | .ule, a, b => a ≤ b
  | .ugt, a, b => b < a
  | .uge, a, b => b ≤ a
  | .eq, a, b => a = b
  | .ne, a, b => a ≠ b
  | _, _, _ => True

theorem holdsRel_sound {A : AS} {Γ : Env} {w : World} (hA : A.Holds vals Γ w) {c : ICmpCond} {x y : R}
    {t : ClifTy} {a b : UInt64} {bf : Bool} {rel : ICmpCond} (hx : Γ[x]? = some (.sc t a))
    (hy : Γ[y]? = some (.sc t b)) (ht : t.isInt = true) (hc : cmpInt c t a b = bf)
    (h : holdsRel A c x y bf = some rel) : RelHolds rel a.toNat b.toNat := by
  rw [← unsignedCond_sound hA hx hy ht] at hc
  unfold holdsRel at h
  generalize unsignedCond A c x y = c' at h hc
  subst hc
  split at h <;> (try cases h) <;> rename_i hc <;>
    simp only [cmpInt, decide_eq_true_eq, decide_eq_false_iff_not, UInt64.lt_iff_toNat_lt,
      UInt64.le_iff_toNat_le, gt_iff_lt, ge_iff_le, Nat.not_lt, Nat.not_le, beq_iff_eq,
      bne_eq_false_iff_eq] at hc <;> simp only [RelHolds] <;>
    first | omega | rw [hc] | (intro h; have := UInt64.toNat_inj.mp h; subst this; simp at hc)

/-- The number `offs` reads a value as: an address less its region's base, or
    the value itself. -/
def numOf : Option Region → UInt64 → Nat
  | none, x => x.toNat
  | some r, x => x.toNat - (regionBase r).toNat

/-- What `offs` reads of a value is true of it, and any number true of the same
    number, typed alike and no larger, describes the value too. -/
theorem AV.offs_desc {Γ : Env} {w : World} {av : AV} {t : ClifTy} {x : UInt64} {g : Option Region}
    {n : Num} (hd : av.Desc Γ w (.sc t x)) (h : av.offs = some (g, n)) :
    n.Holds w (numOf g x) ∧ (∀ r', g = some r' → x.toNat = (regionBase r').toNat + numOf g x) ∧
      ∀ n' : Num, n'.Holds w (numOf g x) → n'.t = n.t → n'.hi ≤ n.hi → (offAV g n').Desc Γ w (.sc t x) := by
  have hp := AV.asPtr_desc hd
  unfold AV.offs at h
  split at h
  · rename_i r m hpe
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    rw [hpe] at hp
    obtain ⟨off, e, hm, hs⟩ := hp
    simp only [V.sc.injEq] at e
    obtain ⟨rfl, rfl⟩ := e
    have hx := addr_toNat r (show off ≤ span by have := hm.2.1; omega)
    have hnum : numOf (some r) (regionBase r + UInt64.ofNat off) = off := by
      simp only [numOf]; omega
    rw [hnum]
    exact ⟨hm, fun r' e => by cases e; rw [hnum] at *; omega,
      fun n' hn' _ hhi => ⟨off, rfl, hn', by omega⟩⟩
  · rename_i b hb
    obtain ⟨m, hm, e⟩ := Option.map_eq_some_iff.mp h
    simp only [Prod.mk.injEq] at e
    obtain ⟨rfl, rfl⟩ := e
    obtain ⟨z, e, hz⟩ := AV.num_desc hp hm
    simp only [V.sc.injEq] at e
    obtain ⟨rfl, rfl⟩ := e
    exact ⟨hz, fun _ e => absurd e (by simp), fun n' hn' ht _ => ⟨x, by rw [ht], hn'⟩⟩

theorem lessRange_holds {A : AS} {Γ : Env} {w : World} (hA : A.Holds vals Γ w) {x y : R} {tx ty : ClifTy}
    {a b : UInt64} {e : Nat} (hx : Γ[x]? = some (.sc tx a)) (hy : Γ[y]? = some (.sc ty b)) (he : e ≤ 1)
    (hab : a.toNat + e ≤ b.toNat) : (lessRange A x y e).Holds vals Γ w := by
  unfold lessRange
  split
  · rename_i gx nx gy ny hox hoy
    split
    · rename_i hg
      subst hg
      obtain ⟨hxa, hbx, hnx⟩ := AV.offs_desc (hA.desc hx) hox
      obtain ⟨hyb, hby, hny⟩ := AV.offs_desc (hA.desc hy) hoy
      have hz : numOf gx a + e ≤ numOf gx b := by
        cases gx with
        | none => exact hab
        | some r => have := hbx r rfl; have := hby r rfl; omega
      generalize numOf gx a = za at hxa hz hnx
      generalize numOf gx b = zb at hyb hz hny
      dsimp only
      generalize hs : (if e = 1 then max 1 (Nat.gcd nx.al ny.al) else 0) = s
      have hstep : za + s ≤ zb := by
        subst hs
        split
        · rename_i h1
          subst h1
          have hga : Nat.gcd nx.al ny.al ∣ za := Nat.dvd_trans (Nat.gcd_dvd_left _ _) hxa.2.2.1
          have hgb : Nat.gcd nx.al ny.al ∣ zb := Nat.dvd_trans (Nat.gcd_dvd_right _ _) hyb.2.2.1
          rcases Nat.eq_zero_or_pos (Nat.gcd nx.al ny.al) with h0 | hpos
          · rw [h0]; omega
          · have := Nat.le_of_dvd (by omega) (Nat.dvd_sub hgb hga)
            have : max 1 (Nat.gcd nx.al ny.al) = Nat.gcd nx.al ny.al := by omega
            omega
        · omega
      have hfx : Fit w (match ny.fit with | some (r, s', d) => some (r, s', d + s * s') | none => none) za := by
        have hf := hyb.2.2.2
        split
        · rename_i r s' d hfit
          rw [hfit] at hf
          have hm := Nat.mul_le_mul_right s' hstep
          rw [Nat.add_mul] at hm
          exact ⟨hf.1, by have := hf.2; omega⟩
        · trivial
      have hA1 := hA.setAV (r := x)
        (av := offAV gx { nx with hi := min nx.hi (ny.hi - s), fit := (match ny.fit with | some (r, s', d) => some (r, s', d + s * s') | none => none).or nx.fit })
        (fun v hv => by
          rw [hx] at hv; cases hv
          exact hnx _ ⟨hxa.1, by have := hxa.2.1; have := hyb.2.1; dsimp only; omega, hxa.2.2.1,
            Fit.or hfx hxa.2.2.2⟩ rfl (Nat.min_le_left _ _))
      have hA2 := hA1.setAV (r := y) (av := offAV gx { ny with lo := max ny.lo (nx.lo + s) })
        (fun v hv => by
          rw [hy] at hv; cases hv
          exact hny _ ⟨by have := hxa.1; have := hyb.1; dsimp only; omega, hyb.2.1, hyb.2.2.1, hyb.2.2.2⟩ rfl
            (Nat.le_refl _))
      have hf := hyb.2.2.2
      revert hA2
      cases hfit : ny.fit with
      | none => dsimp only; exact id
      | some f =>
        obtain ⟨r, s', d⟩ := f
        dsimp only
        intro hA2
        by_cases hpos : 0 < nx.lo + s
        · rw [if_pos hpos]
          rw [hfit] at hf
          refine ⟨hA2.1, hA2.2.1, TState.holds_cons ?_ hA2.2.2⟩
          show (nx.lo + s) * s' + d ≤ w.mem.sizes r
          have hm := Nat.mul_le_mul_right s' (show nx.lo + s ≤ zb by have := hxa.1; omega)
          have := hf.2
          omega
        · rw [if_neg hpos]; exact hA2
    · exact hA
  · exact hA


theorem lessThan_holds {A : AS} {Γ : Env} {w : World} (hA : A.Holds vals Γ w) {x y : R} {tx ty : ClifTy}
    {a b : UInt64} {e : Nat} (hx : Γ[x]? = some (.sc tx a)) (hy : Γ[y]? = some (.sc ty b)) (he : e ≤ 1)
    (hab : a.toNat + e ≤ b.toNat) : (lessThan A x y e).Holds vals Γ w := by
  have hal : a.toNat + alignE A x y e ≤ b.toNat := by
    unfold alignE
    dsimp only
    split
    · rename_i hg
      obtain ⟨rfl, hg⟩ := hg
      have hga : Nat.gcd (A.get x).absAl (A.get y).absAl ∣ a.toNat :=
        Nat.dvd_trans (Nat.gcd_dvd_left _ _) (AV.absAl_dvd (hA.descAt hx))
      have hgb : Nat.gcd (A.get x).absAl (A.get y).absAl ∣ b.toNat :=
        Nat.dvd_trans (Nat.gcd_dvd_right _ _) (AV.absAl_dvd (hA.descAt hy))
      generalize Nat.gcd (A.get x).absAl (A.get y).absAl = g at hg hga hgb ⊢
      obtain ⟨i, hi⟩ := hga
      obtain ⟨j, hj⟩ := hgb
      rw [hi, hj] at hab ⊢
      have hij : i + 1 ≤ j := by
        rcases Nat.lt_or_ge i j with h | h
        · exact h
        · have := Nat.mul_le_mul_left g h; omega
      have := Nat.mul_le_mul_left g hij
      rw [Nat.mul_add, Nat.mul_one] at this
      exact this
    · exact hab
  exact relLo_holds (relLess_holds (lessRange_holds hA hx hy he hab) (by rw [slotVal_sc hx, slotVal_sc hy]; omega)) x

theorem nonZero_holds {A : AS} {Γ : Env} {w : World} (hA : A.Holds vals Γ w) {x y : R} {tx ty : ClifTy}
    {a b : UInt64} (hx : Γ[x]? = some (.sc tx a)) (hy : Γ[y]? = some (.sc ty b)) (hab : a.toNat ≠ b.toNat) :
    (nonZero A x y).Holds vals Γ w := by
  unfold nonZero
  split
  · rename_i t hb
    have hd := hA.descAt hy
    rw [hb] at hd
    simp only [AV.Desc, V.sc.injEq] at hd
    obtain ⟨-, rfl⟩ := hd
    simp only [UInt64.toNat_zero] at hab
    split
    · rename_i k hk
      refine hA.setAV fun v hv => ?_
      rw [hx] at hv; cases hv
      have hd := hA.descAt hx
      rw [hk] at hd
      obtain ⟨x', e, hl⟩ := hd
      refine ⟨x', e, .inr ?_⟩
      cases e
      rcases hl with ⟨-, h0⟩ | hl
      · subst h0; simp at hab
      · exact hl
    · refine hA.tightenLo ?_
      rw [slotVal_sc hx]
      push_cast
      omega
  · exact hA

theorem learn_holds {A : AS} {Γ : Env} {w : World} (hA : A.Holds vals Γ w) {rel : ICmpCond} {x y : R}
    {tx ty : ClifTy} {a b : UInt64} (hx : Γ[x]? = some (.sc tx a)) (hy : Γ[y]? = some (.sc ty b))
    (hr : RelHolds rel a.toNat b.toNat) : (learn A rel x y).Holds vals Γ w := by
  cases rel <;> simp only [learn, RelHolds] at hr ⊢
  · exact lessThan_holds (lessThan_holds hA hx hy (by decide) (by omega)) hy hx (by decide) (by omega)
  · exact nonZero_holds (nonZero_holds hA hx hy hr) hy hx (Ne.symm hr)
  · exact lessThan_holds hA hy hx (by decide) (by omega)
  · exact lessThan_holds hA hy hx (by decide) (by omega)
  · exact lessThan_holds hA hx hy (by decide) (by omega)
  · exact lessThan_holds hA hx hy (by decide) (by omega)
  all_goals exact hA

/-- **What a branch learns is true**: where the flag is `b`, `refine` gives a
    state describing the one the branch starts from. -/
theorem refine_sound {A : AS} {Γ : Env} {w : World} (hA : A.Holds vals Γ w) {flag : R} {t : ClifTy}
    {f : UInt64} {b : Bool} (hf : Γ[flag]? = some (.sc t f)) (hb : (f != 0) = b) :
    ∃ A', refine A flag b = some A' ∧ A'.Holds vals Γ w := by
  have hd := hA.desc hf
  unfold refine
  split
  · rename_i t' f' he
    rw [he] at hd
    simp only [AV.Desc, V.sc.injEq] at hd
    obtain ⟨-, rfl⟩ := hd
    rw [hb]
    exact ⟨A, by simp, hA⟩
  · rename_i c x y he
    rw [he] at hd
    obtain ⟨t', a, b', hx, hy, ht, hv⟩ := hd
    have hc : cmpInt c t' a b' = b := by
      rw [← hb]
      unfold boolV at hv
      simp only [V.sc.injEq] at hv
      obtain ⟨-, rfl⟩ := hv
      split <;> simp_all
    split
    · rename_i rel hrel
      exact ⟨_, rfl, learn_holds hA hx hy (holdsRel_sound hA hx hy ht hc hrel)⟩
    · exact ⟨A, rfl, hA⟩
  · exact ⟨A, rfl, hA⟩

-- ---------------------------------------------------------------------------
-- Carries
-- ---------------------------------------------------------------------------

/-- Each value's linear facts hold of it, read in `Γb`, the environment its
    loop was entered from. -/
def SymsAll (Γb : Env) (w : World) (ss : List Sym) (cs : List V) : Prop :=
  ∀ i, i < cs.length → (ss.getD i {}).FitsAt Γb w (vVal (cs.getD i default)) Γb.size

/-- What carries stepping together say of the carries `cs`, the loop entered
    from `Γb` with the slots `inits`. -/
def IvsAll (Γb : Env) (inits : List R) (ivs : List (Option (Nat × Int))) (cs : List V) : Prop :=
  ∀ i j r, i < cs.length → ivs.getD i none = some (j, r) → j < cs.length →
    inits.getD i 0 < Γb.size → inits.getD j 0 < Γb.size →
    vVal (cs.getD i default) % 2 ^ 64 =
      (slotVal Γb (inits.getD i 0) + r * (vVal (cs.getD j default) - slotVal Γb (inits.getD j 0))) % 2 ^ 64

/-- The carries and the world a loop's carry state describes, the loop entered
    from `Γb`. Its values read no slot, so they are read in the empty
    environment; its linear facts read slots from before the loop. -/
def Carry.Desc (c : Carry) (Γb : Env) (cs : List V) (w : World) : Prop :=
  DescAll #[] w c.vals cs ∧ c.facts.holds w ∧ SymsAll Γb w c.syms cs ∧ IvsAll Γb c.inits c.ivs cs

theorem mapM_getD {Γ : Env} : ∀ {xs : List R} {vs : List V}, xs.mapM (fun r => Γ[r]?) = some vs →
    ∀ i, i < vs.length → Γ[xs.getD i 0]? = some (vs.getD i default)
  | [], vs, h, i, hi => by
      simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h; subst h; simp at hi
  | x :: xs, vs, h, i, hi => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h
      obtain ⟨v, hv, vs', hvs', rfl⟩ := h
      cases i with
      | zero => simpa using hv
      | succ i => simpa using mapM_getD hvs' i (by simpa using hi)

/-- Carries entered with the values of `inits` step together however they are
    said to. -/
theorem ivsAll_entry {Γ : Env} {inits : List R} {cs : List V}
    (h : inits.mapM (fun r => Γ[r]?) = some cs) (ivs : List (Option (Nat × Int))) : IvsAll Γ inits ivs cs := by
  intro i j r hi _ hj _ _
  rw [← slotVal_of (mapM_getD h i hi), ← slotVal_of (mapM_getD h j hj)]
  simp

theorem getD_zipWith_eq {a b : List (Option (Nat × Int))} {i : Nat} {p : Nat × Int}
    (h : (List.zipWith (fun x y => if x = y then x else none) a b).getD i none = some p) :
    a.getD i none = some p ∧ b.getD i none = some p := by
  induction a generalizing b i with
  | nil => simp at h
  | cons x a ih =>
      cases b with
      | nil => simp at h
      | cons y b =>
          cases i with
          | zero =>
              simp only [List.zipWith_cons_cons, List.getD_cons_zero] at h ⊢
              split at h
              · rename_i e; subst e; exact ⟨h, h⟩
              · cases h
          | succ i =>
              simp only [List.zipWith_cons_cons, List.getD_cons_succ] at h ⊢
              exact ih h

theorem IvsAll.join_left {Γb : Env} {a b : Carry} {cs : List V} (h : IvsAll Γb a.inits a.ivs cs) :
    IvsAll Γb (a.join b).inits (a.join b).ivs cs := by
  intro i j r hi hiv
  simp only [Carry.join] at hiv ⊢
  split at hiv
  · exact h i j r hi (getD_zipWith_eq hiv).1
  · simp at hiv

theorem IvsAll.join_right {Γb : Env} {a b : Carry} {cs : List V} (h : IvsAll Γb b.inits b.ivs cs) :
    IvsAll Γb (a.join b).inits (a.join b).ivs cs := by
  intro i j r hi hiv
  simp only [Carry.join] at hiv ⊢
  split at hiv
  · rename_i e; rw [e]; exact h i j r hi (getD_zipWith_eq hiv).2
  · simp at hiv

theorem Sym.fitsAt_merge {σ τ : Sym} {Γ : Env} {w : World} {x : Int} {m : Nat} (h1 : σ.FitsAt Γ w x m)
    (h2 : τ.FitsAt Γ w x m) : (Sym.merge σ τ).FitsAt Γ w x m := by
  obtain ⟨⟨d1, u1⟩, s1, g1⟩ := h1
  obtain ⟨⟨d2, u2⟩, s2, g2⟩ := h2
  refine ⟨⟨fun D e => ?_, fun p hp => ?_⟩, fun y hy => ?_, fun g hg => ?_⟩
  · simp only [Sym.merge] at e
    cases e1 : σ.dfn with
    | some D1 => rw [e1] at e; simp at e; subst e; exact d1 _ e1
    | none => rw [e1] at e; simp at e; exact d2 _ e
  · simp only [Sym.merge, List.mem_append] at hp
    rcases hp with hp | hp
    · exact u1 p hp
    · exact u2 p hp
  · simp only [Sym.slots, Sym.merge, List.mem_append, List.mem_flatten, List.mem_map, List.map_append,
      List.flatten_append] at hy
    rcases hy with hy | hy
    · cases e1 : σ.dfn with
      | some D1 =>
          rw [e1] at hy; simp at hy
          exact s1 y (by simp [Sym.slots, e1, hy])
      | none =>
          rw [e1] at hy; simp at hy
          exact s2 y (by simp only [Sym.slots, List.mem_append]; left; simpa using hy)
    · rcases hy with ⟨_, ⟨p, hp, rfl⟩, hy⟩ | ⟨_, ⟨p, hp, rfl⟩, hy⟩
      · exact s1 y (by simp only [Sym.slots, List.mem_append, List.mem_flatten, List.mem_map]; exact .inr ⟨_, ⟨p, hp, rfl⟩, hy⟩)
      · exact s2 y (by simp only [Sym.slots, List.mem_append, List.mem_flatten, List.mem_map]; exact .inr ⟨_, ⟨p, hp, rfl⟩, hy⟩)
  · simp only [Sym.regions, Sym.merge, List.mem_append, List.mem_flatten, List.mem_map, List.map_append,
      List.flatten_append] at hg
    rcases hg with hg | hg
    · cases e1 : σ.dfn with
      | some D1 =>
          rw [e1] at hg; simp at hg
          exact g1 g (by simp [Sym.regions, e1, hg])
      | none =>
          rw [e1] at hg; simp at hg
          exact g2 g (by simp only [Sym.regions, List.mem_append]; left; simpa using hg)
    · rcases hg with ⟨_, ⟨p, hp, rfl⟩, hg⟩ | ⟨_, ⟨p, hp, rfl⟩, hg⟩
      · exact g1 g (by simp only [Sym.regions, List.mem_append, List.mem_flatten, List.mem_map]; exact .inr ⟨_, ⟨p, hp, rfl⟩, hg⟩)
      · exact g2 g (by simp only [Sym.regions, List.mem_append, List.mem_flatten, List.mem_map]; exact .inr ⟨_, ⟨p, hp, rfl⟩, hg⟩)

/-- What carry `i` steps with fits where the carries are bound at `Γb`'s end. -/
theorem ivSym_fits {Γb : Env} {w : World} {inits : List R} {ivs : List (Option (Nat × Int))} {cs : List V}
    (h : IvsAll Γb inits ivs cs) {i : Nat} (hi : i < cs.length) :
    (ivSym Γb.size cs.length inits i (ivs.getD i none)).Fits (Sem.bindAt Γb Γb.size cs) w
      (Γb.size + i) (Γb.size + cs.length) := by
  show (ivSym Γb.size cs.length inits i (ivs.getD i none)).FitsAt _ w
    (slotVal (Sem.bindAt Γb Γb.size cs) (Γb.size + i)) _
  rw [slotVal_of (v := cs[i]) (by rw [Hoare.bindAt_at, List.getElem?_eq_getElem hi])]
  cases hv : ivs.getD i none with
  | none => exact Sym.fitsAt_blank _ _ _ _
  | some p =>
      obtain ⟨j, r⟩ := p
      simp only [ivSym]
      split
      · rename_i hc
        obtain ⟨hj, hii, hij⟩ := hc
        have key := h i j r hi hv hj hii hij
        have hcj : cs.getD j default = cs[j] := by simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj]
        have hci : cs.getD i default = cs[i] := by simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi]
        rw [hci, hcj] at key
        refine ⟨⟨fun D e => ?_, fun p hp => by simp at hp⟩, fun y hy => ?_, fun g hg => ?_⟩
        · simp only [Option.some.injEq] at e; subst e
          simp only [Lin.eval, Lin.evalT, Atom.val]
          rw [slotVal_bindAt hii hii, slotVal_bindAt hij hij,
            slotVal_of (Γ := Sem.bindAt Γb Γb.size cs) (r := Γb.size + j) (v := cs[j])
              (by rw [Hoare.bindAt_at, List.getElem?_eq_getElem hj])]
          have e : 1 * slotVal Γb (inits.getD i 0) + (r * vVal cs[j] + (-r * slotVal Γb (inits.getD j 0) + 0)) + 0 =
              slotVal Γb (inits.getD i 0) + r * (vVal cs[j] - slotVal Γb (inits.getD j 0)) := by
            rw [Int.mul_sub, Int.neg_mul]; omega
          rw [e]; exact key
        · have hy' : y = inits.getD i 0 ∨ y = Γb.size + j ∨ y = inits.getD j 0 := by
            simpa [Sym.slots, Lin.slots] using hy
          rcases hy' with rfl | rfl | rfl
          · exact Nat.lt_of_lt_of_le hii (Nat.le_add_right _ _)
          · exact Nat.add_lt_add_left hj _
          · exact Nat.lt_of_lt_of_le hij (Nat.le_add_right _ _)
        · simp [Sym.regions, Lin.regions] at hg
      · exact Sym.fitsAt_blank _ _ _ _

/-- Every carry state `b` describes, `c` does. -/
def Carry.Le (b c : Carry) : Prop := ∀ Γb cs w, b.Desc Γb cs w → c.Desc Γb cs w

theorem Carry.Le.refl (c : Carry) : c.Le c := fun _ _ _ h => h

theorem Carry.Le.trans {a b c : Carry} (h1 : a.Le b) (h2 : b.Le c) : a.Le c :=
  fun Γb cs w h => h2 Γb cs w (h1 Γb cs w h)

theorem Sym.join_blank_left (τ : Sym) : Sym.join {} τ = {} := by
  simp only [Sym.join]; split <;> simp

theorem Sym.join_blank_right (σ : Sym) : Sym.join σ {} = {} := by
  simp only [Sym.join]; split <;> simp_all [Sym.like]

theorem getD_zipWith_join : ∀ (a b : List Sym) (i : Nat),
    (List.zipWith Sym.join a b).getD i {} = Sym.join (a.getD i {}) (b.getD i {})
  | [], b, i => by simp [Sym.join_blank_left]
  | _ :: _, [], i => by simp [Sym.join_blank_right]
  | x :: a, y :: b, 0 => rfl
  | x :: a, y :: b, i + 1 => by simpa using getD_zipWith_join a b i

theorem SymsAll.join_left {Γb : Env} {w : World} {a b : List Sym} {cs : List V} (h : SymsAll Γb w a cs) :
    SymsAll Γb w (List.zipWith Sym.join a b) cs := fun i hi => by
  rw [getD_zipWith_join]; exact Sym.join_left (h i hi)

theorem SymsAll.join_right {Γb : Env} {w : World} {a b : List Sym} {cs : List V} (h : SymsAll Γb w b cs) :
    SymsAll Γb w (List.zipWith Sym.join a b) cs := fun i hi => by
  rw [getD_zipWith_join]; exact Sym.join_right (h i hi)

theorem descAll_zip {Γ : Env} {w : World} {f : AV → AV → AV}
    (hf : ∀ a b v, a.Desc Γ w v → (f a b).Desc Γ w v) :
    ∀ {as bs : List AV} {vs : List V}, as.length = bs.length → DescAll Γ w as vs →
      DescAll Γ w (List.zipWith f as bs) vs
  | [], [], [], _, _ => trivial
  | [], [], _ :: _, _, h => h.elim
  | [], _ :: _, _, hl, _ => by simp at hl
  | _ :: _, [], _, hl, _ => by simp at hl
  | _ :: _, _ :: _, [], _, h => h.elim
  | a :: _, b :: _, v :: _, hl, ⟨hd, hds⟩ => ⟨hf a b v hd, descAll_zip hf (by simpa using hl) hds⟩

theorem descAll_zip_right {Γ : Env} {w : World} {f : AV → AV → AV}
    (hf : ∀ a b v, b.Desc Γ w v → (f a b).Desc Γ w v) :
    ∀ {as bs : List AV} {vs : List V}, as.length = bs.length → DescAll Γ w bs vs →
      DescAll Γ w (List.zipWith f as bs) vs
  | [], [], [], _, _ => trivial
  | [], [], _ :: _, _, h => h.elim
  | [], _ :: _, _, hl, _ => by simp at hl
  | _ :: _, [], _, hl, _ => by simp at hl
  | _ :: _, _ :: _, [], _, h => h.elim
  | a :: _, b :: _, v :: _, hl, ⟨hd, hds⟩ => ⟨hf a b v hd, descAll_zip_right hf (by simpa using hl) hds⟩

theorem Carry.join_length {a b : Carry} (h : b.vals.length = a.vals.length) :
    (a.join b).vals.length = a.vals.length := by
  simp [Carry.join, List.length_zipWith, h]

theorem Carry.join_left {a b : Carry} (h : a.vals.length = b.vals.length) : a.Le (a.join b) :=
  fun _ _ _ ⟨hd, hf, hs, hi⟩ => ⟨descAll_zip (fun _ _ _ h => AV.join_left h) h hd,
    factsMeet_left hf, hs.join_left, hi.join_left⟩

theorem Carry.join_right {a b : Carry} (h : a.vals.length = b.vals.length) : b.Le (a.join b) :=
  fun _ _ _ ⟨hd, hf, hs, hi⟩ => ⟨descAll_zip_right (fun _ _ _ h => AV.join_right h) h hd,
    factsMeet_right hf, hs.join_right, hi.join_right⟩

theorem foldl_join : ∀ (backs : List Carry) (c : Carry), (∀ b ∈ backs, b.vals.length = c.vals.length) →
    (backs.foldl Carry.join c).vals.length = c.vals.length ∧ c.Le (backs.foldl Carry.join c) ∧
      ∀ b ∈ backs, b.Le (backs.foldl Carry.join c)
  | [], c, _ => ⟨rfl, Carry.Le.refl c, fun _ h => by cases h⟩
  | b :: bs, c, hl => by
      have hb := hl b (List.mem_cons_self ..)
      have hl' : ∀ x ∈ bs, x.vals.length = (c.join b).vals.length := fun x hx => by
        rw [Carry.join_length hb]; exact hl x (List.mem_cons_of_mem _ hx)
      obtain ⟨h1, h2, h3⟩ := foldl_join bs (c.join b) hl'
      rw [List.foldl_cons]
      refine ⟨h1.trans (Carry.join_length hb), (Carry.join_left hb.symm).trans h2, fun x hx => ?_⟩
      rcases List.mem_cons.mp hx with rfl | hx
      · exact (Carry.join_right hb.symm).trans h2
      · exact h3 x hx

theorem Num.widen_holds {w : World} {o n : Num} {top z : Nat} (h : n.Holds w z) :
    (o.widen n top).Holds w z ∧ (o.widen n top).t = n.t ∧ (o.widen n top).hi ≤ max n.hi top := by
  obtain ⟨h1, h2, h3, h4⟩ := h
  refine ⟨⟨?_, ?_, h3, ?_⟩, rfl, ?_⟩
  · show (if o.lo ≤ n.lo then n.lo else 0) ≤ z
    split <;> omega
  · show z ≤ if n.hi ≤ o.hi then n.hi else max n.hi top
    split
    · exact h2
    · exact Nat.le_trans h2 (Nat.le_max_left _ _)
  · show Fit w (if o.fit = n.fit then n.fit else none) z
    split
    · exact h4
    · trivial
  · show (if n.hi ≤ o.hi then n.hi else max n.hi top) ≤ max n.hi top
    split
    · exact Nat.le_max_left _ _
    · exact Nat.le_refl _

theorem AV.widen_desc {Γ : Env} {w : World} {o n : AV} {v : V} (hd : n.Desc Γ w v) :
    (AV.widen o n).Desc Γ w v := by
  unfold AV.widen
  split
  · rename_i a b
    obtain ⟨x, rfl, hx⟩ := hd
    obtain ⟨hh, ht, -⟩ := Num.widen_holds (o := a) (top := if b.hi < 2 ^ (b.t.width - 1) then
      2 ^ (b.t.width - 1) - 1 else 2 ^ b.t.width - 1) hx
    exact ⟨x, by rw [ht], hh⟩
  · rename_i o' r n'
    obtain ⟨off, rfl, hn, hs⟩ := hd
    obtain ⟨hh, -, hhi⟩ := Num.widen_holds (o := o')
      (top := span - if 0 < n'.al ∧ n'.al ≤ span then n'.al else 1) hn
    refine ⟨off, rfl, hh, ?_⟩
    have : max n'.hi (span - if 0 < n'.al ∧ n'.al ≤ span then n'.al else 1) ≤ span := by
      unfold span at hs ⊢; split <;> omega
    omega
  · exact hd

theorem Carry.widen_le {a b : Carry} (h : a.vals.length = b.vals.length) : b.Le (Carry.widen a b) :=
  fun _ _ _ ⟨hd, hf, hs, hi⟩ => ⟨descAll_zip_right (fun _ _ _ h => AV.widen_desc h) h hd, hf, hs, hi⟩

theorem Carry.widen_length {a b : Carry} (h : a.vals.length = b.vals.length) :
    (Carry.widen a b).vals.length = a.vals.length := by
  simp [Carry.widen, List.length_zipWith, h]

theorem eq_of_zip_beq : ∀ {as bs : List AV}, as.length = bs.length →
    (List.zipWith AV.beq as bs).all id = true → as = bs
  | [], [], _, _ => rfl
  | a :: as, b :: bs, hl, h => by
      simp only [List.zipWith_cons_cons, List.all_cons, id, Bool.and_eq_true] at h
      rw [AV.eq_of_beq h.1, eq_of_zip_beq (by simpa using hl) h.2]
  | [], _ :: _, hl, _ => by simp at hl
  | _ :: _, [], hl, _ => by simp at hl

theorem Carry.beq_le {a b : Carry} (h : a.beq b = true) : a.Le b := by
  simp only [Carry.beq, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨hl, hv⟩, -⟩, hf⟩, hs⟩, hin⟩, hiv⟩ := h
  intro Γb cs w ⟨hd, hfa, hsa, hia⟩
  refine ⟨?_, fun x hx => hfa x (by simpa using List.all_eq_true.mp hf x hx), by rw [← hs]; exact hsa,
    by rw [← hin, ← hiv]; exact hia⟩
  rw [← eq_of_zip_beq hl hv]
  exact hd

/-- What checking a loop gives: an invariant covering the carries it is
    entered with, from which one trip goes round only with states it covers. -/
def LoopInv (path : List Nat) (trip : Carry → Except String (List Carry × List AS × AOut)) (c0 : Carry)
    (o : AOut) : Prop :=
  ∃ c backs exits out j, c0.Le c ∧ c.vals.length = c0.vals.length ∧
    trip c = .ok (backs, exits, out) ∧ (∀ b ∈ backs, b.Le c) ∧ joinAll exits = .ok j ∧
    o = ⟨j, out.brk, out.cont, { out.certs with loops := (path, c) :: out.certs.loops }, .star out.tr, out.cnt⟩

theorem AV.enrich_desc {Γ : Env} {w : World} {S : TState} {a : AV} {v : V} (hS : S.holds w)
    (hd : a.Desc Γ w v) : (a.enrich S).Desc Γ w v := by
  have hp := AV.asPtr_desc hd
  unfold AV.enrich
  split
  · rename_i r n hpe
    rw [hpe] at hp
    split
    · rename_i k hk hf
      split
      · rename_i hc
        obtain ⟨off, rfl, hn, hs⟩ := hp
        have hok := Num.exact_eq hn hk
        have := roomOf_le hS r
        exact ⟨off, rfl, ⟨hn.1, hn.2.1, hn.2.2.1, ⟨hc.2, by omega⟩⟩, hs⟩
      · exact hd
    · exact hd
  · exact hd

theorem DescAll.enrich {Γ : Env} {w : World} {S : TState} (hS : S.holds w) :
    ∀ {as : List AV} {vs : List V}, DescAll Γ w as vs → DescAll Γ w (as.map (AV.enrich S)) vs
  | [], [], _ => trivial
  | _ :: _, _ :: _, ⟨hd, hds⟩ => ⟨AV.enrich_desc hS hd, DescAll.enrich hS hds⟩
  | [], _ :: _, h => h.elim
  | _ :: _, [], h => h.elim

/-- **`settle` ends at an invariant**: a carry state covering the one it
    started from, from which one trip goes round only with states it covers. -/
theorem settle_sound {path : List Nat} {trip : Carry → Except String (List Carry × List AS × AOut)}
    (n : Nat) : ∀ {c0 : Carry} {o : AOut}, settle path trip n c0 = .ok o → LoopInv path trip c0 o := by
  induction n with
  | zero =>
      intro c o h
      unfold settle at h
      cases ht : trip c with
      | error e => simp [ht] at h
      | ok r =>
          obtain ⟨backs, exits, out⟩ := r
          simp only [ht] at h
          split at h
          · rename_i hlen
            have hl : ∀ b ∈ backs, b.vals.length = c.vals.length := fun b hb => by
              simpa using List.all_eq_true.mp hlen b hb
            obtain ⟨-, -, hb⟩ := foldl_join backs c hl
            split at h
            · rename_i hbeq
              split at h
              · cases h
              · rename_i j hj
                cases h
                exact ⟨c, backs, exits, out, j, Carry.Le.refl c, rfl, ht,
                  fun b hm => (hb b hm).trans (Carry.beq_le hbeq), hj, rfl⟩
            · cases h
          · cases h
  | succ m ih =>
      intro c o h
      unfold settle at h
      cases ht : trip c with
      | error e => simp [ht] at h
      | ok r =>
          obtain ⟨backs, exits, out⟩ := r
          simp only [ht] at h
          split at h
          · rename_i hlen
            have hl : ∀ b ∈ backs, b.vals.length = c.vals.length := fun b hb => by
              simpa using List.all_eq_true.mp hlen b hb
            obtain ⟨hl', hc, hb⟩ := foldl_join backs c hl
            split at h
            · rename_i hbeq
              split at h
              · cases h
              · rename_i j hj
                cases h
                exact ⟨c, backs, exits, out, j, Carry.Le.refl c, rfl, ht,
                  fun b hm => (hb b hm).trans (Carry.beq_le hbeq), hj, rfl⟩
            · obtain ⟨c2, backs2, exits2, out2, j2, hle, hlen2, ht2, hb2, hj2, ho⟩ := ih h
              have hnext : c.Le (if m < 9 then Carry.widen c (backs.foldl Carry.join c)
                    else backs.foldl Carry.join c) ∧
                  (if m < 9 then Carry.widen c (backs.foldl Carry.join c)
                    else backs.foldl Carry.join c).vals.length = c.vals.length := by
                split
                · exact ⟨hc.trans (Carry.widen_le hl'.symm), Carry.widen_length hl'.symm⟩
                · exact ⟨hc, hl'⟩
              exact ⟨c2, backs2, exits2, out2, j2, hnext.1.trans hle, hlen2.trans hnext.2, ht2, hb2, hj2, ho⟩
          · cases h

/-- **A loop checked against a given invariant**: the invariant it checks
    covers the carries the loop is entered with, and one trip from it goes
    round only with states it covers. -/
theorem checkLoop_sound {path : List Nat} {trip : Carry → Except String (List Carry × List AS × AOut)}
    {c0 c : Carry} {o : AOut} (h : checkLoop path trip c0 c = .ok o) : LoopInv path trip c0 o := by
  unfold checkLoop at h
  split at h
  · rename_i hl
    dsimp only at h
    have hci : c0.Le (if (c0.join c).beq c then c else c0.join c) ∧
        (if (c0.join c).beq c then c else c0.join c).vals.length = c0.vals.length := by
      split
      · rename_i hb; exact ⟨(Carry.join_left hl).trans (Carry.beq_le hb), hl.symm⟩
      · exact ⟨Carry.join_left hl, Carry.join_length hl.symm⟩
    generalize (if (c0.join c).beq c then c else c0.join c) = ci at h hci
    cases ht : trip ci with
    | error e => simp [ht] at h
    | ok r =>
        obtain ⟨backs, exits, out⟩ := r
        simp only [ht] at h
        split at h
        · rename_i hc
          simp only [Bool.and_eq_true] at hc
          obtain ⟨hlen, hbeq⟩ := hc
          have hl' : ∀ b ∈ backs, b.vals.length = ci.vals.length := fun b hb => by
            simpa using List.all_eq_true.mp hlen b hb
          obtain ⟨-, -, hb⟩ := foldl_join backs _ hl'
          split at h
          · cases h
          · rename_i j hj
            cases h
            exact ⟨ci, backs, exits, out, j, hci.1, hci.2, ht,
              fun b hm => (hb b hm).trans (Carry.beq_le hbeq), hj, rfl⟩
        · cases h
  · cases h

theorem runLoop_sound {cert : Cert} {path : List Nat}
    {trip : Carry → Except String (List Carry × List AS × AOut)} {c0 : Carry} {o : AOut}
    (h : runLoop cert path trip c0 = .ok o) : LoopInv path trip c0 o := by
  unfold runLoop at h
  split at h
  · exact checkLoop_sound h
  · split at h
    · cases h
    · rename_i o1 hs
      have h1 := settle_sound _ hs
      split at h
      · split at h
        · split at h
          · cases h; exact checkLoop_sound (by assumption)
          · cases h; exact h1
        · cases h; exact h1
      · cases h; exact h1

/-- What `runLoopIv` gives: what `runLoop` gives, the carries entered with
    some steps-together. -/
theorem withIvs_sound {cert : Cert} {path : List Nat}
    {trip tripFo : Carry → Except String (List Carry × List AS × AOut)} {c0 : Carry} {o : AOut}
    (h : withIvs cert path trip tripFo c0 = .ok o) : ∃ ivs, LoopInv path trip { c0 with ivs := ivs } o := by
  unfold withIvs at h
  split at h
  · cases h
  · dsimp only at h
    generalize ivGuess cert path tripFo c0 = ivs at h
    split at h
    · exact ⟨_, runLoop_sound h⟩
    · cases h

theorem runLoopIv_sound {cert : Cert} {path : List Nat}
    {trip tripFo : Carry → Except String (List Carry × List AS × AOut)} {c0 : Carry} {o : AOut}
    (h : runLoopIv cert path trip tripFo c0 = .ok o) : ∃ ivs, LoopInv path trip { c0 with ivs := ivs } o := by
  unfold runLoopIv at h
  split at h
  · exact ⟨c0.ivs, runLoop_sound h⟩
  · split at h
    · split at h
      · rename_i o' h'; cases h; exact withIvs_sound h'
      · exact ⟨c0.ivs, runLoop_sound h⟩
    · split at h
      · rename_i o' h'; cases h; exact ⟨c0.ivs, runLoop_sound h'⟩
      · split at h
        · rename_i o' h'; cases h; exact withIvs_sound h'
        · cases h

-- ---------------------------------------------------------------------------
-- One trip of a loop
-- ---------------------------------------------------------------------------

theorem mem_brks0 {xs : List (Nat × AS × List R)} {A : AS} {args : List R} {ab n0 : Nat} {Y : List R}
    (h : (0, A, args) ∈ xs) : A.bindAt ab (args.map A.get) (args.map (exitSym A Y n0)) ∈ brks0 xs ab Y n0 :=
  List.mem_filterMap.mpr ⟨_, h, by simp⟩

theorem mem_depthDown {xs : List (Nat × AS × List R)} {d : Nat} {A : AS} {args : List R}
    (h : (d + 1, A, args) ∈ xs) : (d, A, args) ∈ depthDown xs :=
  List.mem_filterMap.mpr ⟨_, h, rfl⟩

theorem mem_conts0 {xs : List (Nat × AS × List R)} {A : AS} {args : List R} {Y : List R} {c : Carry}
    {n0 : Nat} (h : (0, A, args) ∈ xs) : backCarry A Y c n0 args ∈ conts0 xs Y c n0 :=
  List.mem_filterMap.mpr ⟨_, h, by simp⟩

theorem mapM_len {f : R → Option V} : ∀ {xs : List R} {vs : List V}, xs.mapM f = some vs → vs.length = xs.length
  | [], vs, h => by simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h; subst h; rfl
  | x :: xs, vs, h => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h
      obtain ⟨v, -, vs', hvs', rfl⟩ := h
      simp [mapM_len hvs']

theorem loopCands_lt {n0 : Nat} {c : Code} : ∀ y ∈ loopCands n0 c, y < n0 := fun y hy => by
  simpa using (List.mem_filter.mp hy).2

/-- **What `closeSym` says of a value is true**, read where the loop was
    entered from: an environment the present one agrees with on the slots
    before the loop. -/
theorem closeSym_fitsAt {A : AS} {Γ Γb : Env} {w : World} {Y : List R} {a : R} (hA : A.Holds vals Γ w)
    (hY : ∀ y ∈ Y, y < Γb.size) (hΓ : ∀ y, y < Γb.size → slotVal Γ y = slotVal Γb y) :
    (closeSym A Y a).FitsAt Γb w (slotVal Γ a) Γb.size := by
  -- each bound `closeSym` keeps is `y` less some number, `y` one of `Y`
  have hform : ∀ p ∈ (closeSym A Y a).ubs, ∃ y ∈ Y, ∃ g : Int, p = (1, ⟨[(.slot y, 1)], -g⟩) ∧
      slotVal Γ a + g ≤ slotVal Γ y := by
    intro p hp
    simp only [closeSym, List.mem_filterMap] at hp
    obtain ⟨y, hy, hpy⟩ := hp
    split at hpy
    · rename_i hpr
      cases hpy
      have := Ctx.prove_sound (AS.ctx_sound hA) hpr
      simp only [Lin.eval_add, Lin.eval_sub, eval_lin1, Lin.eval_const] at this
      exact ⟨y, hy, _, rfl, by omega⟩
    · split at hpy
      · rename_i hpr
        cases hpy
        have := Ctx.prove_sound (AS.ctx_sound hA) hpr
        simp only [Lin.eval_sub, eval_lin1] at this
        exact ⟨y, hy, 0, by simp [lin1], by omega⟩
      · cases hpy
  refine ⟨⟨fun D hD => by simp [closeSym] at hD, fun p hp => ?_⟩, fun x hx => ?_, fun g hg => ?_⟩
  · obtain ⟨y, hy, g, rfl, hle⟩ := hform p hp
    simp only [Lin.eval, Lin.evalT, Atom.val]
    rw [← hΓ y (hY y hy)]
    omega
  · simp only [Sym.slots, closeSym, Option.map_none, Option.getD_none, List.nil_append, List.mem_flatten,
      List.mem_map] at hx
    obtain ⟨_, ⟨p, hp, rfl⟩, hx⟩ := hx
    obtain ⟨y, hy, g, rfl, -⟩ := hform p hp
    simp only [Lin.slots, List.filterMap_cons, List.filterMap_nil, List.mem_singleton] at hx
    subst hx; exact hY _ hy
  · simp only [Sym.regions, closeSym, Option.map_none, Option.getD_none, List.nil_append, List.mem_flatten,
      List.mem_map] at hg
    obtain ⟨_, ⟨p, hp, rfl⟩, hg⟩ := hg
    obtain ⟨y, hy, g', rfl, -⟩ := hform p hp
    simp [Lin.regions] at hg

/-- The values a list of slots passes on, each with what `closeSym` says of it. -/
theorem closeSym_all {A : AS} {Γ Γb : Env} {w : World} {Y : List R} (hA : A.Holds vals Γ w)
    (hY : ∀ y ∈ Y, y < Γb.size) (hΓ : ∀ y, y < Γb.size → slotVal Γ y = slotVal Γb y) :
    ∀ {args : List R} {vs : List V}, args.mapM (fun r => Γ[r]?) = some vs →
      SymsAll Γb w (args.map (closeSym A Y)) vs
  | [], vs, h => by
      simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h
      subst h; intro i hi; simp at hi
  | a :: as, vs, h => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h
      obtain ⟨v, hv, vs', hvs', rfl⟩ := h
      intro i hi
      cases i with
      | zero =>
          simp only [List.map_cons, List.getD_cons_zero]
          rw [← slotVal_of hv]
          exact closeSym_fitsAt hA hY hΓ
      | succ i =>
          simp only [List.map_cons, List.getD_cons_succ, List.length_cons] at hi ⊢
          exact closeSym_all hA hY hΓ hvs' i (by omega)

theorem Ext.slotVal {Γ0 Γ : Env} (h : Ext Γ0 Γ) {y : Nat} (hy : y < Γ0.size) :
    _root_.AlgorithmLib.Check.slotVal Γ y = _root_.AlgorithmLib.Check.slotVal Γ0 y := by
  unfold AlgorithmLib.Check.slotVal; rw [h.2 y hy]

/-- **What `closeDfn` says of a value is true**, read where the loop was
    entered from. -/
theorem closeDfn_fitsAt {A : AS} {Γ Γb : Env} {w : World} (hA : A.Holds vals Γ w)
    (hΓ : ∀ y, y < Γb.size → slotVal Γ y = slotVal Γb y) (a : R) :
    ({ dfn := closeDfn A Γb.size a } : Sym).FitsAt Γb w (slotVal Γ a) Γb.size := by
  unfold closeDfn
  dsimp only
  split
  · rename_i hc
    obtain ⟨hs, hg⟩ := hc
    simp only [List.all_eq_true, decide_eq_true_eq] at hs hg
    have hC := AS.ctx_sound hA
    refine ⟨⟨fun D e => ?_, fun p hp => by simp at hp⟩, fun y hy => ?_, fun g hgm => ?_⟩
    · simp only [Option.some.injEq] at e
      subst e
      rw [Lin.eval_congr (Γ := Γ) (w := w) (fun x hx => (hΓ x (hs x hx)).symm) (fun _ _ => rfl),
        Ctx.foldKnown_eval hC, Lin.eval_norm, Ctx.deepen_mod hC, eval_lin1, Int.one_mul]
    · simp only [Sym.slots, Option.map_some, Option.getD_some, List.map_nil, List.flatten_nil,
        List.append_nil] at hy
      exact hs y hy
    · simp only [Sym.regions, Option.map_some, Option.getD_some, List.map_nil, List.flatten_nil,
        List.append_nil] at hgm
      exact hg g hgm
  · exact Sym.fitsAt_blank _ _ _ _

theorem exitSym_all {A : AS} {Γ Γb : Env} {w : World} {Y : List R} (hA : A.Holds vals Γ w)
    (hY : ∀ y ∈ Y, y < Γb.size) (hΓ : ∀ y, y < Γb.size → slotVal Γ y = slotVal Γb y) :
    ∀ {args : List R} {vs : List V}, args.mapM (fun r => Γ[r]?) = some vs →
      SymsAll Γb w (args.map (exitSym A Y Γb.size)) vs
  | [], vs, h => by
      simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h
      subst h; intro i hi; simp at hi
  | a :: as, vs, h => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h
      obtain ⟨v, hv, vs', hvs', rfl⟩ := h
      intro i hi
      cases i with
      | zero =>
          simp only [List.map_cons, List.getD_cons_zero]
          rw [← slotVal_of hv]
          exact Sym.fitsAt_merge (closeDfn_fitsAt hA hΓ a) (closeSym_fitsAt hA hY hΓ)
      | succ i =>
          simp only [List.map_cons, List.getD_cons_succ, List.length_cons] at hi ⊢
          exact exitSym_all hA hY hΓ hvs' i (by omega)

/-- A form whose unfolding comes to a multiple of `2 ^ 64` and no atoms is
    zero modulo `2 ^ 64`. -/
theorem ivForm_mod {C : Ctx} {Γ : Env} {w : World} (hC : C.Sound Γ w) {D : Lin}
    (ht : (C.foldKnown (C.deepen ivDepth D)).norm.terms = [])
    (hk : (C.foldKnown (C.deepen ivDepth D)).norm.k % 2 ^ 64 = 0) : D.eval Γ w % 2 ^ 64 = 0 := by
  have e : (C.foldKnown (C.deepen ivDepth D)).norm.eval Γ w = (C.foldKnown (C.deepen ivDepth D)).norm.k := by
    simp [Lin.eval, ht, Lin.evalT]
  rw [Lin.eval_norm, Ctx.foldKnown_eval hC] at e
  have := Ctx.deepen_mod hC ivDepth D
  omega

/-- **The steps-together a trip keeps hold** of the values it goes round with. -/
theorem ivKeep_sound {A : AS} {Γ Γb : Env} {w : World} {c : Carry} {args : List R} {vs : List V}
    (hA : A.Holds vals Γ w) (hΓ : ∀ y, y < Γb.size → slotVal Γ y = slotVal Γb y)
    (hvs : args.mapM (fun r => Γ[r]?) = some vs) : IvsAll Γb c.inits (ivKeep A Γb.size c args) vs := by
  intro i j r hi hiv hj hii hij
  have hvl := mapM_len hvs
  unfold ivKeep at hiv
  rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range (by omega)] at hiv
  simp only [Option.map_some, Option.getD_some] at hiv
  split at hiv
  · rename_i j' r' hcv
    split at hiv
    · rename_i hc
      simp only [Option.some.injEq, Prod.mk.injEq] at hiv
      obtain ⟨rfl, rfl⟩ := hiv
      obtain ⟨hterms, hk, -, -, -⟩ := hc
      have h0 := ivForm_mod (AS.ctx_sound hA) hterms hk
      simp only [Lin.eval_sub, Lin.eval_scale, eval_lin1, Int.one_mul] at h0
      rw [slotVal_of (mapM_getD hvs i hi), slotVal_of (mapM_getD hvs _ hj), hΓ _ hii, hΓ _ hij] at h0
      generalize _ * (vVal (vs.getD _ default) - slotVal Γb (c.inits.getD _ 0)) = X at h0 ⊢
      omega
    · cases hiv
  · cases hiv

theorem backCarry_desc {A : AS} {Γ Γb : Env} {w : World} {Y : List R} {c : Carry} {n0 : Nat} {args : List R}
    {vs : List V} (hA : A.Holds vals Γ w) (hn0 : n0 = Γb.size) (hY : ∀ y ∈ Y, y < Γb.size)
    (hΓ : ∀ y, y < Γb.size → slotVal Γ y = slotVal Γb y) (hvs : args.mapM (fun r => Γ[r]?) = some vs) :
    (backCarry A Y c n0 args).Desc Γb vs w := by
  subst hn0
  exact ⟨(descAll_of hA hvs).portable, hA.2.2, closeSym_all hA hY hΓ hvs, ivKeep_sound hA hΓ hvs⟩

theorem exitSym_all' {A : AS} {Γ Γb : Env} {w : World} {Y : List R} {n0 : Nat} (hA : A.Holds vals Γ w)
    (hn0 : n0 = Γb.size) (hY : ∀ y ∈ Y, y < Γb.size) (hΓ : ∀ y, y < Γb.size → slotVal Γ y = slotVal Γb y)
    {args : List R} {vs : List V} (hvs : args.mapM (fun r => Γ[r]?) = some vs) :
    SymsAll Γb w (args.map (exitSym A Y n0)) vs := by
  subst hn0; exact exitSym_all hA hY hΓ hvs

/-- Values bound after a loop, or as its carries, with the linear facts
    `SymsAll` gives them: they fit where they are bound. -/
theorem SymsAll.fits_bindAt {Γb Γ : Env} {w : World} {ss : List Sym} {cs : List V} {n : Nat}
    (h : SymsAll Γb w ss cs) (hE : Ext Γb (Sem.bindAt Γ n cs)) (hn : Γb.size ≤ n) :
    ∀ j, j < cs.length → (ss.getD j {}).Fits (Sem.bindAt Γ n cs) w (n + j) (n + cs.length) := by
  intro j hj
  show (ss.getD j {}).FitsAt _ w (slotVal (Sem.bindAt Γ n cs) (n + j)) _
  rw [slotVal_of (v := cs[j]) (by rw [Hoare.bindAt_at, List.getElem?_eq_getElem hj])]
  have hj' := h j hj
  rw [show cs.getD j default = cs[j] by simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj]] at hj'
  exact Sym.fitsAt_congr hj' (fun y hy => hE.slotVal hy) (fun _ _ => rfl) (by omega)

/-- A loop's carries, bound at its head with what their state says of them,
    fit there. -/
theorem headSyms_fits {c : Carry} {Γ0 : Env} {w : World} {cs : List V} (hsc : SymsAll Γ0 w c.syms cs)
    (hiv : IvsAll Γ0 c.inits c.ivs cs) (hl : cs.length = c.vals.length) :
    ∀ i, i < cs.length → ((c.headSyms Γ0.size).getD i {}).Fits (Sem.bindAt Γ0 Γ0.size cs) w
      (Γ0.size + i) (Γ0.size + cs.length) := by
  intro i hi
  have e : (c.headSyms Γ0.size).getD i {} =
      Sym.merge (c.syms.getD i {}) (ivSym Γ0.size c.vals.length c.inits i (c.ivs.getD i none)) := by
    simp [Carry.headSyms, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range (hl ▸ hi)]
  rw [e, ← hl]
  exact Sym.fitsAt_merge (SymsAll.fits_bindAt hsc ((Ext.refl Γ0).bindAt (Nat.le_refl _) cs) (Nat.le_refl _) i hi)
    (ivSym_fits hiv hi)

/-- The values picked by index out of ones with linear facts, with those facts. -/
theorem SymsAll.pick {Γb : Env} {w : World} {ss : List Sym} {cs : List V} (h : SymsAll Γb w ss cs) :
    ∀ {idx : List Nat} {outs : List V}, idx.mapM (fun i => cs[i]?) = some outs →
      SymsAll Γb w (idx.map fun i => ss.getD i {}) outs
  | [], outs, hm => by
      simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at hm
      subst hm; intro i hi; simp at hi
  | i :: is, outs, hm => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at hm
      obtain ⟨o, ho, os, hos, rfl⟩ := hm
      intro k hk
      cases k with
      | zero =>
          simp only [List.map_cons, List.getD_cons_zero]
          have hi : i < cs.length := by
            rcases Nat.lt_or_ge i cs.length with h' | h'
            · exact h'
            · rw [List.getElem?_eq_none h'] at ho; cases ho
          have e : cs.getD i default = o := by
            simp [List.getD_eq_getElem?_getD, ho]
          have := h i hi
          rwa [e] at this
      | succ k =>
          simp only [List.map_cons, List.getD_cons_succ, List.length_cons] at hk ⊢
          exact SymsAll.pick h hos k (by omega)

/-- A loop body's parameters, bound from the carries `cs` the slots from `n0`
    hold, are each exactly the slot it is bound from. -/
theorem copySyms_fits {Γ : Env} {w : World} {n0 n k : Nat} {cs : List V}
    (hΓ : ∀ i, i < k → Γ[n0 + i]? = cs[i]?) (hcl : cs.length = k) (hn : n0 + k ≤ n) :
    ∀ j, j < k → ((copySyms n0 k).getD j {}).Fits (Sem.bindAt Γ n cs) w (n + j) (n + k) := by
  intro j hj
  have e : (copySyms n0 k).getD j {} = { dfn := some (lin1 (n0 + j) 1) } := by
    simp [copySyms, List.getD_eq_getElem?_getD, List.getElem?_range hj]
  rw [e]
  have hj' : j < cs.length := hcl ▸ hj
  have hv : Γ[n0 + j]? = some cs[j] := by rw [hΓ j hj, List.getElem?_eq_getElem hj']
  have hlt : n0 + j < Γ.size := by
    rcases Nat.lt_or_ge (n0 + j) Γ.size with h | h
    · exact h
    · rw [Array.getElem?_eq_none h] at hv; cases hv
  refine ⟨⟨fun D hD => ?_, fun p hp => by simp at hp⟩, fun y hy => ?_, fun g hg => ?_⟩
  · simp only [Option.some.injEq] at hD
    subst hD
    rw [eval_lin1, slotVal_bindAt (by omega) hlt, slotVal_of hv,
      slotVal_of (v := cs[j]) (by rw [Hoare.bindAt_at, List.getElem?_eq_getElem hj'])]
    simp
  · simp only [Sym.slots, Option.map_some, Option.getD_some, List.map_nil, List.flatten_nil,
      List.append_nil] at hy
    rw [lin1_slots _ 1 y hy]; show n0 + j < n + k; omega
  · simp only [Sym.regions, Option.map_some, Option.getD_some, List.map_nil, List.flatten_nil,
      List.append_nil] at hg
    exact lin1_regions _ 1 g hg

/-- What one trip collects: everything its prefix and body end with. -/
theorem loopTrip_spec {l : Loop} {n0 k preEnd ab : Nat} {Y : List R} {c : Carry} {op : AOut}
    {runBody : AS → Except String AOut} {backs : List Carry} {exits : List AS} {out : AOut}
    (h : loopTrip l n0 k preEnd ab Y c op runBody = .ok (backs, exits, out)) :
    (∀ x ∈ conts0 op.cont Y c n0, x ∈ backs) ∧ (∀ x ∈ brks0 op.brk ab Y n0, x ∈ exits) ∧
    (∀ x ∈ depthDown op.brk, x ∈ out.brk) ∧ (∀ x ∈ depthDown op.cont, x ∈ out.cont) ∧
    (∀ s, op.tr.Has s → out.tr.Has s) ∧
    (∀ q, out.cnt = some q → ∃ Ap Ag ob, op.ok = some Ap ∧ refine Ap l.flag (!l.exitOnTrue) = some Ag ∧
      runBody (Ag.bindAt preEnd ((List.range k).map fun i => Ag.get (n0 + i)) (copySyms n0 k)) = .ok ob ∧
      tripCnt l n0 k op Ap ob = some q) ∧
    ∀ Ap, op.ok = some Ap →
      (∀ Ax, refine Ap l.flag l.exitOnTrue = some Ax →
        Ax.bindAt ab (l.exitR.map Ax.get) (l.exitR.map (exitSym Ax Y n0)) ∈ exits) ∧
      ∀ Ag, refine Ap l.flag (!l.exitOnTrue) = some Ag →
        ∃ ob, runBody (Ag.bindAt preEnd ((List.range k).map fun i => Ag.get (n0 + i)) (copySyms n0 k)) = .ok ob ∧
          (∀ x ∈ conts0 ob.cont Y c n0, x ∈ backs) ∧ (∀ x ∈ brks0 ob.brk ab Y n0, x ∈ exits) ∧
          (∀ x ∈ depthDown ob.brk, x ∈ out.brk) ∧ (∀ x ∈ depthDown ob.cont, x ∈ out.cont) ∧
          (∀ s, (Tr.seq op.tr ob.tr).Has s → out.tr.Has s) ∧
          ∀ A2, ob.ok = some A2 → backCarry A2 Y c n0 l.cont ∈ backs := by
  unfold loopTrip at h
  dsimp only at h
  split at h
  · rename_i hn
    simp only [Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl, rfl⟩ := h
    exact ⟨fun _ h => h, fun _ h => h, fun _ h => h, fun _ h => h, fun _ h => h, fun _ h => (by simp at h),
      fun Ap e => by rw [hn] at e; cases e⟩
  · rename_i Ap hAp
    split at h
    · rename_i hg
      simp only [Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl⟩ := h
      refine ⟨fun _ h => h, fun _ h => List.mem_append_left _ h, fun _ h => h, fun _ h => h, fun _ h => h,
        fun _ h => (by simp at h), fun Ap' e => ?_⟩
      rw [hAp] at e; cases e
      refine ⟨fun Ax hx => List.mem_append_right _ (by simp [hx]), fun Ag hg' => ?_⟩
      rw [hg] at hg'; cases hg'
    · rename_i Ag hg
      split at h
      · cases h
      · rename_i ob hob
        simp only [Except.ok.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl, rfl⟩ := h
        refine ⟨fun _ h => List.mem_append_left _ (List.mem_append_left _ h),
          fun _ h => List.mem_append_left _ (List.mem_append_left _ h),
          fun _ h => List.mem_append_left _ h, fun _ h => List.mem_append_left _ h,
          fun s h => by
            have := Tr.Has.seq h (Tr.Has.altL (b := ob.tr) .nil)
            rw [List.append_nil] at this
            exact this,
          fun q h => ⟨Ap, Ag, ob, hAp, hg, hob, h⟩,
          fun Ap' e => ?_⟩
        rw [hAp] at e; cases e
        refine ⟨fun Ax hx => List.mem_append_left _ (List.mem_append_right _ (by simp [hx])),
          fun Ag' hg' => ?_⟩
        rw [hg] at hg'; cases hg'
        exact ⟨ob, hob, fun _ h => List.mem_append_left _ (List.mem_append_right _ h),
          fun _ h => List.mem_append_right _ h, fun _ h => List.mem_append_right _ h,
          fun _ h => List.mem_append_right _ h,
          fun s h => by
            cases h with
            | seq a b => exact .seq a (.altR b),
          fun A2 e => List.mem_append_right _ (by simp [e])⟩

-- ---------------------------------------------------------------------------
-- What a statement keeps
-- ---------------------------------------------------------------------------

open AlgorithmLib.HProg.Contracts (KeepsRooms foldl_stores_sizes)

theorem store_world {cfg : Cfg} {Γ Γ' : Env} {w w' : World} {ty : ClifTy} {v a : R}
    (hr : runStmt cfg Γ w (.store ty v a) = .ok Γ' w') :
    Γ' = Γ ∧ w'.mem.sizes = w.mem.sizes ∧ callsOf w'.obs = callsOf w.obs := by
  rw [runStmt_store] at hr
  split at hr
  · dsimp only at hr
    split at hr
    · split at hr
      · rename_i m hm; cases hr; exact ⟨rfl, store_sizes_eq hm, rfl⟩
      · cases hr
    · split at hr
      · rename_i m hm
        cases hr
        rw [← Array.foldlM_toList] at hm
        exact ⟨rfl, (foldl_stores_sizes _ (fun _ _ _ h => ⟨store_sizes_eq h, store_frozen h⟩) _ _ _ hm).1, rfl⟩
      · cases hr
  · cases hr

theorem storeU_world {cfg : Cfg} {Γ Γ' : Env} {w w' : World} {v a : R}
    (hr : runStmt cfg Γ w (.storeUnaligned v a) = .ok Γ' w') :
    Γ' = Γ ∧ w'.mem.sizes = w.mem.sizes ∧ callsOf w'.obs = callsOf w.obs := by
  rw [runStmt_storeUnaligned] at hr
  split at hr
  · split at hr
    · rename_i m hm; cases hr; exact ⟨rfl, store_sizes_eq hm, rfl⟩
    · cases hr
  · dsimp only at hr
    split at hr
    · rename_i m hm
      cases hr
      rw [← Array.foldlM_toList] at hm
      exact ⟨rfl, (foldl_stores_sizes _ (fun _ _ _ h => ⟨store_sizes_eq h, store_frozen h⟩) _ _ _ hm).1, rfl⟩
    · cases hr
  · cases hr

theorem istore8_world {cfg : Cfg} {Γ Γ' : Env} {w w' : World} {v a : R}
    (hr : runStmt cfg Γ w (.istore8 v a) = .ok Γ' w') :
    Γ' = Γ ∧ w'.mem.sizes = w.mem.sizes ∧ callsOf w'.obs = callsOf w.obs := by
  rw [runStmt_istore8] at hr
  split at hr
  · split at hr
    · rename_i m hm; cases hr; exact ⟨rfl, store_sizes_eq hm, rfl⟩
    · cases hr
  · cases hr

/-- An accepted foreign call keeps every region's size but the pinned one's. -/
theorem ffi_rooms {cfg : Cfg} {A : AS} {f : Ffi} {Γ : Env} {w : World} {vs : List V} {r : Option V}
    {w' : World} (hA : A.Holds vals Γ w) (hf : f ≠ .threadSpawn)
    (hz : A.facts.contains (.part .frozen false) = true)
    (hc : callOf cfg.locals (.ffi f) vs (obsCall w (.ffi f) vs) = some (r, w')) : KeepsRooms w w' := by
  have hz' : (obsCall w (.ffi f) vs).mem.frozen = false := holds_of_mem hA.2.2 hz
  rw [Contracts.callOf_ffi hf _ _ hz'] at hc
  obtain ⟨bits, -, hc⟩ := Option.bind_eq_some_iff.mp hc
  exact fun rg hrg => callBits_rooms f hc rg hrg

/-- An accepted call keeps every region's size but the pinned one's. -/
theorem callee_rooms {cfg : Cfg} {A B : AS} {c : Callee} {args : List R} {av : AV}
    (h : A.callee c args = .ok (B, av)) {Γ : Env} {w : World} {vs : List V} {res : Option V} {w2 : World}
    (hA : A.Holds vals Γ w) (hvs : args.mapM (fun r => Γ[r]?) = some vs)
    (hc : callOf cfg.locals c vs (obsCall w c vs) = some (res, w2)) : KeepsRooms w w2 := by
  cases c with
  | ffi f =>
      obtain ⟨hB, -⟩ := callee_ffi h
      obtain ⟨hf, hz, -⟩ := ffiCall_ok hB
      exact ffi_rooms hA hf hz hc
  | ext e =>
      obtain ⟨ha, hsc, -, -, -⟩ := callee_ext h
      exact ext_rooms hA ha hsc hvs hc
  | _ => simp [AS.callee] at h

/-- An accepted call leaves the observations as they were once it was made. -/
theorem callee_obs {cfg : Cfg} {A B : AS} {c : Callee} {args : List R} {av : AV}
    (h : A.callee c args = .ok (B, av)) {Γ : Env} {w : World} {vs : List V} {res : Option V} {w2 : World}
    (hA : A.Holds vals Γ w) (hvs : args.mapM (fun r => Γ[r]?) = some vs)
    (hc : callOf cfg.locals c vs (obsCall w c vs) = some (res, w2)) : w2.obs = .call c vs :: w.obs := by
  cases c with
  | ffi f =>
      obtain ⟨hB, -⟩ := callee_ffi h
      obtain ⟨hf, hz, -⟩ := ffiCall_ok hB
      have hz' : (obsCall w (.ffi f) vs).mem.frozen = false := holds_of_mem hA.2.2 hz
      rw [Contracts.callOf_ffi hf _ _ hz'] at hc
      obtain ⟨bits, -, hc⟩ := Option.bind_eq_some_iff.mp hc
      exact Contracts.callBits_obs f hc
  | ext e =>
      obtain ⟨ha, hsc, -, -, -⟩ := callee_ext h
      have hA1 := hA.obsCall (.ext e) vs
      obtain ⟨bits, hb⟩ := bits_exist (descAll_of hA1 hvs) hsc
      exact (Contracts.lib_step ha hc hb).obs
  | _ => simp [AS.callee] at h

/-- Arguments in scope carry the bits the state records of them. -/
theorem argsOk_of {A : AS} {Γ : Env} {w : World} (hA : A.Holds vals Γ w) :
    ∀ {args : List R} {vs : List V}, args.mapM (fun r => Γ[r]?) = some vs →
      argsOk (args.map fun r => (A.get r).bits) vs
  | [], vs, h => by
      simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h
      subst h; trivial
  | r :: args, vs, h => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h
      obtain ⟨v, hv, vs', hrest, rfl⟩ := h
      simp only [List.map_cons]
      refine ⟨?_, argsOk_of hA hrest⟩
      cases hb : (A.get r).bits with
      | none => trivial
      | some b =>
          obtain ⟨t, rfl⟩ := AV.bits_desc (hA.desc hv) hb
          rfl

/-- **What a statement the checker accepts keeps**: the environment extended,
    every region's size, and of calls only the one its record describes. -/
theorem absStmt_step {cfg : Cfg} {A A' : AS} {s : Stmt} {fo : Bool} (h : absStmt fo A s = .ok A') {Γ : Env} {w : World}
    (hA : A.Holds vals Γ w) {Γ' : Env} {w' : World} (hr : runStmt cfg Γ w s = .ok Γ' w') :
    Ext Γ Γ' ∧ Kept (.lit (recOf A s)) w w' := by
  cases s with
  | op o =>
      simp only [runStmt] at hr
      split at hr
      · cases hr; exact ⟨Ext.push _ _, Kept.refl _⟩
      · cases hr
  | store ty v a =>
      obtain ⟨rfl, hs, ho⟩ := store_world hr
      exact ⟨Ext.refl _, Kept.same hs ho⟩
  | storeUnaligned v a =>
      obtain ⟨rfl, hs, ho⟩ := storeU_world hr
      exact ⟨Ext.refl _, Kept.same hs ho⟩
  | istore8 v a =>
      obtain ⟨rfl, hs, ho⟩ := istore8_world hr
      exact ⟨Ext.refl _, Kept.same hs ho⟩
  | call c args =>
      simp only [absStmt] at h
      split at h
      · rename_i B av hB
        rw [runStmt_call] at hr
        split at hr
        · cases hr
        · rename_i vs hvs
          split at hr
          · exact absurd hr failOf_ne_ok
          · rename_i res w2 hc
            split at hr
            · cases hr
              exact ⟨Ext.push _ _, Kept.call (callee_rooms hB hA hvs hc) (callee_obs hB hA hvs hc)
                (argsOk_of hA hvs)⟩
            · cases hr
      · cases h
  | callVoid c args =>
      simp only [absStmt] at h
      split at h
      · rename_i B av hB
        rw [runStmt_callVoid] at hr
        split at hr
        · cases hr
        · rename_i vs hvs
          split at hr
          · exact absurd hr failOf_ne_ok
          · rename_i res w2 hc
            cases hr
            exact ⟨Ext.refl _, Kept.call (callee_rooms hB hA hvs hc) (callee_obs hB hA hvs hc)
              (argsOk_of hA hvs)⟩
      · cases h

theorem stmts_conseq {cfg : Cfg} {P P' Q Q' : Env → World → Prop} {ss : List Stmt} {fo : Bool}
    (h : Stmts cfg P ss Q fo) (hP : ∀ Γ w, P' Γ w → P Γ w) (hQ : ∀ Γ w, Q Γ w → Q' Γ w) :
    Stmts cfg P' ss Q' fo := by
  intro Γ w hp
  have := h Γ w (hP Γ w hp)
  cases hr : runStmts cfg Γ w ss with
  | ok Γ' w' => rw [hr] at this; exact hQ _ _ this
  | stuck _ => trivial
  | misuse _ => rw [hr] at this; exact this
  | fault _ => rw [hr] at this; exact this

/-- The state a run is in, as seen from where it started: the checker's state
    describes it, its environment extends the one it started in, every region
    keeps its size, and the calls made since are ones `t` admits. -/
def Within (vals : List UInt64) (A : AS) (t : Tr) (Γ0 : Env) (w0 : World) (Γ : Env) (w : World) : Prop :=
  A.Holds vals Γ w ∧ Ext Γ0 Γ ∧ Kept t w0 w

theorem absStmt_frame {cfg : Cfg} {A A' : AS} {s : Stmt} {fo : Bool} (h : absStmt fo A s = .ok A') {t : Tr}
    {Γ0 : Env} {w0 : World} :
    Stmt1 cfg (Within vals A t Γ0 w0) s (Within vals A' (.seq t (.lit (recOf A s))) Γ0 w0) fo := by
  intro Γ w ⟨hA, he, hk⟩
  have h1 := absStmt_sound (cfg := cfg) h Γ w hA
  cases hr : runStmt cfg Γ w s with
  | ok Γ' w' =>
      rw [hr] at h1
      obtain ⟨e1, k1⟩ := absStmt_step h hA hr
      exact ⟨h1, he.trans e1, hk.seq k1⟩
  | stuck _ => trivial
  | misuse _ => rw [hr] at h1; exact h1
  | fault _ => rw [hr] at h1; exact h1

/-- **A straight run the checker accepts makes the calls it records**, in
    order, after those made before it. -/
theorem absStmtsH_frame {cfg : Cfg} {Γ0 : Env} {w0 : World} {fo : Bool} :
    ∀ {hist : List (Nat × Stmt)} {A A' : AS} {ss : List Stmt} {t : Tr}, absStmtsH fo hist A ss = .ok A' →
      Stmts cfg (Within vals A t Γ0 w0) ss (Within vals A' (.seq t (.lit (callRecs fo A ss))) Γ0 w0) fo
  | _, A, A', [], t, h => by
      simp only [absStmtsH, Except.ok.injEq] at h
      subst h
      exact fun _ _ ⟨h1, h2, h3⟩ => ⟨h1, h2, h3.seq (Kept.refl _)⟩
  | _, A, A', s :: ss, t, h => by
      simp only [absStmtsH] at h
      split at h
      · rename_i A1 h1
        refine stmts_conseq (stmts_cons (absStmt_frame h1) (absStmtsH_frame h)) (fun _ _ hp => hp) ?_
        rintro Γ w ⟨hx, he, hk⟩
        refine ⟨hx, he, hk.mono fun x hs => ?_⟩
        have e : callRecs fo A (s :: ss) = recOf A s ++ callRecs fo A1 ss := by simp only [callRecs, h1]
        rw [e]
        exact hs.seq_lits
      · cases h

-- ---------------------------------------------------------------------------
-- Regions
-- ---------------------------------------------------------------------------

/-- What a region the checker accepted may end with, from `Γ0` and `w0`: in a
    state its result describes, or by a `br` or `cont` from a state it
    recorded, passing the values of the slots it recorded; with the
    environment extended, every region's size kept and the calls made ones its
    pattern admits; and no fault but where `fo` allows one. -/
def AOut.post (o : AOut) (vals : List UInt64) (fo : Bool) (Γ0 : Env) (w0 : World) : Post where
  ok Γ w := (∃ A, o.ok = some A ∧ A.Holds vals Γ w) ∧ Ext Γ0 Γ ∧ Kept o.tr w0 w
  brk d Γ vs w := (∃ A args, (d, A, args) ∈ o.brk ∧ A.Holds vals Γ w ∧ args.mapM (fun r => Γ[r]?) = some vs) ∧
    Ext Γ0 Γ ∧ Kept o.tr w0 w
  cont d vs w := (∃ A args Γ, (d, A, args) ∈ o.cont ∧ A.Holds vals Γ w ∧ args.mapM (fun r => Γ[r]?) = some vs ∧
    Ext Γ0 Γ) ∧ Kept o.tr w0 w
  faultOk := fo

theorem sat_mono {Q Q' : Post} {r : CodeRes} (h : CodeRes.Sat Q r)
    (hok : ∀ Γ w, Q.ok Γ w → Q'.ok Γ w) (hbrk : ∀ d Γ vs w, Q.brk d Γ vs w → Q'.brk d Γ vs w)
    (hcont : ∀ d vs w, Q.cont d vs w → Q'.cont d vs w) (hf : Q.faultOk = true → Q'.faultOk = true) :
    CodeRes.Sat Q' r := by
  cases r with
  | ok Γ w => exact hok _ _ h
  | brk d Γ vs w => exact hbrk _ _ _ _ h
  | cont d vs w => exact hcont _ _ _ h
  | stuck _ => trivial
  | misuse _ => exact h
  | fault _ => exact hf h

theorem pieceT_conseq {cfg : Cfg} {P : Env → World → Prop} {p : Piece} {Q Q' : Post}
    (h : PieceT cfg P p Q)
    (hok : ∀ Γ w, Q.ok Γ w → Q'.ok Γ w) (hbrk : ∀ d Γ vs w, Q.brk d Γ vs w → Q'.brk d Γ vs w)
    (hcont : ∀ d vs w, Q.cont d vs w → Q'.cont d vs w) (hf : Q.faultOk = true → Q'.faultOk = true) :
    PieceT cfg P p Q' :=
  fun fuel Γ w hP => sat_mono (h fuel Γ w hP) hok hbrk hcont hf

theorem triple_of_at {cfg : Cfg} {P : Env → World → Prop} {c : Code} {Q : Post}
    (h : ∀ Γ w, P Γ w → Triple cfg (At Γ w) c Q) : Triple cfg P c Q :=
  fun fuel Γ w hP => h Γ w hP fuel Γ w ⟨rfl, rfl⟩

theorem answerCases_frame {cfg : Cfg} {A A' B1 B2 : AS} {s : Stmt} {fo : Bool} (h : absStmt fo A s = .ok A')
    (hc : answerCases A A' s = some (B1, B2)) {t : Tr} {Γ0 : Env} {w0 : World} :
    Stmt1 cfg (Within vals A t Γ0 w0) s
      (fun Γ w => Within vals B1 .any Γ0 w0 Γ w ∨ Within vals B2 .any Γ0 w0 Γ w) fo := by
  intro Γ w ⟨hA, he, hk⟩
  have h1 := answerCases_sound (cfg := cfg) (fo := fo) h hc Γ w hA
  cases hr : runStmt cfg Γ w s with
  | ok Γ' w' =>
      rw [hr] at h1
      obtain ⟨e1, k1⟩ := absStmt_step h hA hr
      rcases h1 with h1 | h1
      · exact .inl ⟨h1, he.trans e1, (hk.seq k1).any⟩
      · exact .inr ⟨h1, he.trans e1, (hk.seq k1).any⟩
  | stuck _ => trivial
  | misuse _ => rw [hr] at h1; exact h1
  | fault _ => rw [hr] at h1; exact h1

theorem stmts_or {cfg : Cfg} {P1 P2 Q1 Q2 : Env → World → Prop} {ss : List Stmt} {fo : Bool}
    (h1 : Stmts cfg P1 ss Q1 fo) (h2 : Stmts cfg P2 ss Q2 fo) :
    Stmts cfg (fun Γ w => P1 Γ w ∨ P2 Γ w) ss (fun Γ w => Q1 Γ w ∨ Q2 Γ w) fo := by
  intro Γ w hp
  rcases hp with hp | hp
  · have := h1 Γ w hp
    cases hr : runStmts cfg Γ w ss with
    | ok Γ' w' => rw [hr] at this; exact .inl this
    | stuck _ => trivial
    | misuse _ => rw [hr] at this; exact this
    | fault _ => rw [hr] at this; exact this
  · have := h2 Γ w hp
    cases hr : runStmts cfg Γ w ss with
    | ok Γ' w' => rw [hr] at this; exact .inr this
    | stuck _ => trivial
    | misuse _ => rw [hr] at this; exact this
    | fault _ => rw [hr] at this; exact this

/-- A straight run `splitCases` splits ends in one of its two states. -/
theorem splitCases_frame {cfg : Cfg} {Γ0 : Env} {w0 : World} {fo : Bool} :
    ∀ {A C1 C2 : AS} {ss : List Stmt} {t : Tr}, splitCases fo A ss = some (C1, C2) →
      Stmts cfg (Within vals A t Γ0 w0) ss
        (fun Γ w => Within vals C1 .any Γ0 w0 Γ w ∨ Within vals C2 .any Γ0 w0 Γ w) fo
  | _, _, _, [], _, h => by simp [splitCases] at h
  | A, C1, C2, s :: ss, t, h => by
      simp only [splitCases] at h
      split at h
      · cases h
      · rename_i A' hA'
        split at h
        · rename_i B1 B2 hc
          split at h
          · rename_i D1 D2 h1 h2
            simp only [Option.some.injEq, Prod.mk.injEq] at h
            obtain ⟨rfl, rfl⟩ := h
            exact stmts_cons (answerCases_frame hA' hc)
              (stmts_or
                (stmts_conseq (absStmtsH_frame h1) (fun _ _ hp => hp) fun _ _ ⟨a, b, c⟩ => ⟨a, b, c.any⟩)
                (stmts_conseq (absStmtsH_frame h2) (fun _ _ hp => hp) fun _ _ ⟨a, b, c⟩ => ⟨a, b, c.any⟩))
          · cases h
        · exact stmts_cons (absStmt_frame hA') (splitCases_frame h)

theorem desc_empty {Γ : Env} {w : World} {av : AV} {v : V} (hd : av.Desc #[] w v) : av.Desc Γ w v := by
  cases av with
  | cmp c a b => obtain ⟨_, _, _, ha, -⟩ := hd; simp at ha
  | _ => exact AV.desc_transport hd (fun _ h => by simp [AV.refs] at h) (fun _ _ => rfl) (fun _ _ _ _ h => h)

theorem DescAll.of_empty {Γ : Env} {w : World} :
    ∀ {as : List AV} {vs : List V}, DescAll #[] w as vs → DescAll Γ w as vs
  | [], [], _ => trivial
  | _ :: _, _ :: _, ⟨hd, hds⟩ => ⟨desc_empty hd, DescAll.of_empty hds⟩
  | [], _ :: _, h => h.elim
  | _ :: _, [], h => h.elim

theorem descAll_of_get {Γ : Env} {w : World} :
    ∀ {as : List AV} {vs : List V}, as.length = vs.length →
      (∀ i, i < as.length → ∀ v, vs[i]? = some v → (as.getD i .any).Desc Γ w v) → DescAll Γ w as vs
  | [], [], _, _ => trivial
  | _ :: _, v :: _, hl, h =>
      ⟨h 0 (by simp) v rfl, descAll_of_get (by simpa using hl) fun i hi u hu => by
        simpa using h (i + 1) (by simp; omega) u (by simpa using hu)⟩
  | [], _ :: _, hl, _ => by simp at hl
  | _ :: _, [], hl, _ => by simp at hl

theorem descAll_pick {Γ : Env} {w : World} {as : List AV} {vs : List V} (hd : DescAll Γ w as vs) :
    ∀ {idx : List Nat} {outs : List V}, idx.mapM (fun i => vs[i]?) = some outs →
      DescAll Γ w (idx.map fun i => as.getD i .any) outs
  | [], outs, h => by
      simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h
      subst h; trivial
  | i :: idx, outs, h => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h
      obtain ⟨v, hv, outs', hrest, rfl⟩ := h
      refine ⟨?_, descAll_pick hd hrest⟩
      have hi : i < as.length := by rw [hd.length]; exact (List.getElem?_eq_some_iff.mp hv).1
      obtain ⟨v', hv', hd'⟩ := hd.get hi
      rw [hv] at hv'; cases hv'; exact hd'

/-- What one trip of a bottom-tested loop collects. -/
theorem dloopTrip_spec {l : DLoop} {n0 ab : Nat} {Y : List R} {c : Carry} {ob : AOut} {backs : List Carry}
    {exits : List AS} {out : AOut} (h : dloopTrip l n0 ab Y c ob = (backs, exits, out)) :
    (∀ x ∈ conts0 ob.cont Y c n0, x ∈ backs) ∧ (∀ x ∈ brks0 ob.brk ab Y n0, x ∈ exits) ∧
    (∀ x ∈ depthDown ob.brk, x ∈ out.brk) ∧ (∀ x ∈ depthDown ob.cont, x ∈ out.cont) ∧
    (∀ s, ob.tr.Has s → out.tr.Has s) ∧
    ∀ A2, ob.ok = some A2 →
      (∀ Ag, refine A2 l.flag l.contOnTrue = some Ag → backCarry Ag Y c n0 l.cont ∈ backs) ∧
      (∀ Ax, refine A2 l.flag (!l.contOnTrue) = some Ax →
        Ax.bindAt ab (l.exitIdx.map fun i => (l.cont.map Ax.get).getD i .any)
          (l.exitIdx.map fun i => (l.cont.map (exitSym Ax Y n0)).getD i {}) ∈ exits) := by
  unfold dloopTrip at h
  simp only [Prod.mk.injEq] at h
  obtain ⟨rfl, rfl, rfl⟩ := h
  refine ⟨fun _ h => List.mem_append_left _ h, fun _ h => List.mem_append_left _ h, fun _ h => h,
    fun _ h => h, fun _ h => h, fun A2 e => ⟨fun Ag hg => List.mem_append_right _ (by simp [e, hg]),
      fun Ax hx => List.mem_append_right _ (by simp [e, hx])⟩⟩

theorem Slots.eq_of_beqBy {α : Type} {f : α → α → Bool} (hf : ∀ a b, f a b = true → a = b) :
    ∀ {a b : Slots α}, Slots.beqBy f a b = true → a = b
  | .leaf, .leaf, _ => rfl
  | .node l x r, .node l' x' r', h => by
      simp only [Slots.beqBy, Bool.and_eq_true] at h
      rw [hf _ _ h.1.1, Slots.eq_of_beqBy hf h.1.2, Slots.eq_of_beqBy hf h.2]
  | .leaf, .node .., h => by simp [Slots.beqBy] at h
  | .node .., .leaf, h => by simp [Slots.beqBy] at h

theorem AS.eq_of_beq {A B : AS} (h : A.beq B = true) : A = B := by
  simp only [AS.beq, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
  obtain ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩ := h
  cases A; cases B
  simp only at h1 h2 h3 h4 h5
  rw [Slots.eq_of_beqBy (fun _ _ => AV.eq_of_beq) h2, h1, h3,
    Slots.eq_of_beqBy (fun _ _ e => of_decide_eq_true e) h4, h5]

/-- A state written out is the one reached. -/
theorem pin_eq {cert : Cert} {path : List Nat} {A A' : AS} (h : pin cert path A = .ok A') : A' = A := by
  unfold pin at h
  split at h
  · split at h
    · rename_i hb; cases h; exact (AS.eq_of_beq hb).symm
    · cases h
  · cases h; rfl

-- ---------------------------------------------------------------------------
-- Counted loops
-- ---------------------------------------------------------------------------

theorem stepsOne_sound {A : AS} {Γ : Env} {w : World} (hA : A.Holds vals Γ w) {a x : R}
    (h : stepsOne A a x = true) : (slotVal Γ a - slotVal Γ x - 1) % 2 ^ 64 = 0 := by
  simp only [stepsOne, Bool.and_eq_true, decide_eq_true_eq] at h
  have h0 := ivForm_mod (AS.ctx_sound hA) h.1 h.2
  simpa only [Lin.eval_sub, eval_lin1, Lin.eval_const, Int.one_mul] using h0

/-- What `tripCnt` checked of a trip it counts. -/
theorem tripCnt_spec {l : Loop} {n0 k : Nat} {op : AOut} {Ap : AS} {ob : AOut} {j : Nat} {n : UInt64} {P B : Tr}
    (h : tripCnt l n0 k op Ap ob = some (j, n, P, B)) :
    l.exitOnTrue = false ∧ op.brk = [] ∧ op.cont = [] ∧ ob.brk = [] ∧ ob.cont = [] ∧ P = op.tr ∧ B = ob.tr ∧
    j < k ∧ (∃ y, Ap.get l.flag = .cmp .ult (n0 + j) y ∧ Ap.get y = .known (.sc .i64 n)) ∧
    ∀ A2, ob.ok = some A2 → j < l.cont.length ∧ (A2.get (l.cont.getD j 0)).scTy = some .i64 ∧
      stepsOne A2 (l.cont.getD j 0) (n0 + j) = true := by
  unfold tripCnt at h
  split at h
  · rename_i x y hfl
    split at h
    · rename_i n' hy
      split at h
      · rename_i hc
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl, rfl, rfl⟩ := h
        simp only [cntOk, Bool.and_eq_true, Bool.not_eq_true', List.isEmpty_iff, decide_eq_true_eq] at hc
        obtain ⟨⟨⟨⟨⟨⟨hx, hb1⟩, hc1⟩, hb2⟩, hc2⟩, hr1, hr2⟩, hs⟩ := hc
        have hxe : n0 + (x - n0) = x := Nat.add_sub_cancel' hr1
        refine ⟨hx, hb1, hc1, hb2, hc2, rfl, rfl, Nat.sub_lt_left_of_lt_add hr1 hr2, ⟨y, by rw [hxe]; exact hfl, hy⟩, fun A2 e => ?_⟩
        rw [e] at hs
        simp only [Bool.and_eq_true, decide_eq_true_eq] at hs
        rw [hxe]
        exact ⟨hs.1.1, hs.1.2, hs.2⟩
      · cases h
    · cases h
  · cases h

/-- What `cntTr` gives: the pattern it was handed, or the counted one. -/
theorem cntTr_cases (A : AS) (l : Loop) (t : Tr) (q : Option (Nat × UInt64 × Tr × Tr)) :
    cntTr A l t q = t ∨ ∃ j n P B c0, q = some (j, n, P, B) ∧
      A.get (l.init.getD j 0) = .known (.sc .i64 c0) ∧ j < l.init.length ∧ c0.toNat ≤ n.toNat ∧
      cntTr A l t q = .seq (.rep (n.toNat - c0.toNat) (.seq P B)) P := by
  unfold cntTr
  split
  · rename_i j n P B
    split
    · rename_i c0 hc0
      split
      · rename_i hc
        exact .inr ⟨j, n, P, B, c0, rfl, hc0, hc.1, hc.2, rfl⟩
      · exact .inl rfl
    · exact .inl rfl
  · exact .inl rfl

/-- Calls a loop's pattern admits are ones its counted pattern does, where
    its trips are not counted. -/
theorem Kept.uncounted {A : AS} {l : Loop} {t : Tr} {q : Option (Nat × UInt64 × Tr × Tr)} {w0 w : World}
    (h : Kept t w0 w) (hq : ∀ x, q = some x → False) : Kept (cntTr A l t q) w0 w := by
  rcases cntTr_cases A l t q with e | ⟨j, n, P, B, c0, e, -⟩
  · rw [e]; exact h
  · exact (hq _ e).elim

/-- **Where a loop's trips are counted**, at its head: the counting carry is
    between the number it was entered with and its bound, and the calls made
    since the loop was entered are as many trips' as it is above that number. -/
def CntInv (A : AS) (l : Loop) (q : Option (Nat × UInt64 × Tr × Tr)) (w0 : World) (cs : List V) (w : World) :
    Prop :=
  ∀ j n P B, q = some (j, n, P, B) → ∀ c0 : UInt64,
    A.get (l.init.getD j 0) = .known (.sc .i64 c0) → j < l.init.length → c0.toNat ≤ n.toNat →
    ∃ x : UInt64, cs[j]? = some (.sc .i64 x) ∧ c0.toNat ≤ x.toNat ∧ x.toNat ≤ n.toNat ∧
      Kept (.rep (x.toNat - c0.toNat) (.seq P B)) w0 w

theorem getD_getElem? {vs : List V} {j : Nat} (hj : j < vs.length) : vs[j]? = some (vs.getD j default) := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj]; rfl

/-- A loop is entered with no trip made. -/
theorem cntInv_entry {A : AS} {l : Loop} {q : Option (Nat × UInt64 × Tr × Tr)} {Γ0 : Env} {w0 : World}
    (hA : A.Holds vals Γ0 w0) {cs : List V} (hcs : l.init.mapM (fun r => Γ0[r]?) = some cs) :
    CntInv A l q w0 cs w0 := by
  intro j n P B _ c0 hc0 hj hle
  have hl := mapM_len hcs
  have hg := mapM_getD hcs j (by omega)
  have hd := hA.desc hg
  rw [hc0] at hd
  have hd : cs.getD j default = .sc .i64 c0 := hd
  refine ⟨c0, ?_, Nat.le_refl _, hle, ?_⟩
  · rw [getD_getElem? (by omega), hd]
  · rw [Nat.sub_self]; exact Kept.repZero _

/-- What the test of a counted loop says: go on exactly while the counting
    carry is below the bound. -/
theorem counted_flag {Ap : AS} {Γ1 : Env} {w1 : World} (hAp : Ap.Holds vals Γ1 w1) {flag xr y : R}
    {n x f : UInt64} {t : ClifTy} (hfl : Ap.get flag = .cmp .ult xr y) (hy : Ap.get y = .known (.sc .i64 n))
    (hf : Γ1[flag]? = some (.sc t f)) (hx : Γ1[xr]? = some (.sc .i64 x)) :
    (f != 0) = decide (x.toNat < n.toNat) := by
  have hd := hAp.desc hf
  rw [hfl] at hd
  obtain ⟨t', x', y', hx', hy', -, e⟩ := hd
  rw [hx] at hx'
  simp only [Option.some.injEq, V.sc.injEq] at hx'
  obtain ⟨rfl, rfl⟩ := hx'
  have hdy := hAp.desc hy'
  rw [hy] at hdy
  simp only [AV.Desc, V.sc.injEq, true_and] at hdy
  subst hdy
  simp only [boolV, cmpInt, V.sc.injEq] at e
  obtain ⟨-, rfl⟩ := e
  by_cases hlt : x < y'
  · have : x.toNat < y'.toNat := UInt64.lt_iff_toNat_lt.mp hlt
    simp [hlt, this]
  · have : ¬ x.toNat < y'.toNat := fun h => hlt (UInt64.lt_iff_toNat_lt.mpr h)
    simp [hlt, this]

mutual

/-- **A piece the checker accepts** ends, from any state the abstract one
    describes, as its result says. -/
theorem execP_sound {cfg : Cfg} {cert : Cert} :
    ∀ {path : List Nat} {A : AS} {p : Piece} {o : AOut}, execP cert path A p = .ok o →
      ∀ {Γ0 : Env} {w0 : World}, A.Holds vals Γ0 w0 → PieceT cfg (At Γ0 w0) p (o.post vals cert.fo Γ0 w0)
  | _, A, .straight ss, o, h, Γ0, w0, hA => by
      simp only [execP] at h
      split at h
      · cases h
      · rename_i A' hss
        cases h
        exact straight_rule (stmts_conseq (absStmtsH_frame (t := .lit []) (Γ0 := Γ0) (w0 := w0) hss)
          (fun Γ w ⟨e1, e2⟩ => by subst Γ w; exact ⟨hA, Ext.refl _, Kept.refl _⟩)
          (fun Γ w ⟨h1, h2, h3⟩ => ⟨⟨A', rfl, h1⟩, h2, h3.straight⟩))
  | _, A, .br d args, o, h, Γ0, w0, hA => by
      simp only [execP, Except.ok.injEq] at h
      subst h
      exact br_rule fun Γ w vs ⟨e1, e2⟩ hvs => by
        subst Γ w
        exact ⟨⟨A, args, List.mem_singleton.mpr rfl, hA, hvs⟩, Ext.refl _, Kept.refl _⟩
  | _, A, .cont d args, o, h, Γ0, w0, hA => by
      simp only [execP, Except.ok.injEq] at h
      subst h
      exact cont_rule fun Γ w vs ⟨e1, e2⟩ hvs => by
        subst Γ w
        exact ⟨⟨A, args, Γ0, List.mem_singleton.mpr rfl, hA, hvs, Ext.refl _⟩, Kept.refl _⟩
  | _, A, .ite m thn els thnR elsR, o, h, Γ0, w0, hA => by
      have hn : Γ0.size = A.len := hA.1.1
      have hj0 : A.len ≤ slotsOf (slotsOf A.len thn) els :=
        Nat.le_trans (slotsOf_ge _ _) (slotsOf_ge _ _)
      simp only [execP] at h
      split at h
      · rename_i ot oe hT hE
        split at h
        · rename_i j hj
          cases h
          refine ite_rule (fun Γ w t f ⟨e1, e2⟩ hf hb => ?_) (fun Γ w t f ⟨e1, e2⟩ hf hb => ?_)
          · subst Γ w
            obtain ⟨At', hr, hAt⟩ := refine_sound hA hf hb
            rw [hr] at hT
            rw [hn]
            refine Hoare.conseq (execC_sound hT hAt) (fun _ _ h => h) ?_ ?_ ?_
            · intro Γ' w' ⟨⟨A', e, hA'⟩, he, hk⟩ vs hvs
              obtain ⟨C, hC, hCh⟩ := joinOpt_holds hj
                (.inl ⟨_, by rw [e]; rfl, hA'.bindAt _ (descAll_of hA' hvs)⟩)
              exact ⟨⟨C, hC, hCh⟩, he.bindAt (by omega) vs, hk.altL⟩
            · intro d Γ' vs w' ⟨⟨A', avs, hm, h1, h2⟩, he, hk⟩
              exact ⟨⟨A', avs, List.mem_append_left _ hm, h1, h2⟩, he, hk.altL⟩
            · intro d vs w' ⟨⟨A', args, Γ', hm, h1, h2, he⟩, hk⟩
              exact ⟨⟨A', args, Γ', List.mem_append_left _ hm, h1, h2, he⟩, hk.altL⟩
          · subst Γ w
            obtain ⟨Ae, hr, hAe⟩ := refine_sound hA hf hb
            rw [hr] at hE
            have hAe' := hAe.bindAt (slotsOf A.len thn) (avs := []) (vs := []) trivial
            have he0 : Ext Γ0 (bindAt Γ0 (slotsOf A.len thn) []) :=
              (Ext.refl Γ0).bindAt (by have := slotsOf_ge A.len thn; omega) []
            rw [hn]
            refine Hoare.conseq (execC_sound hE hAe') (fun _ _ h => h) ?_ ?_ ?_
            · intro Γ' w' ⟨⟨A', e, hA'⟩, he, hk⟩ vs hvs
              obtain ⟨C, hC, hCh⟩ := joinOpt_holds hj
                (.inr ⟨_, by rw [e]; rfl, hA'.bindAt _ (descAll_of hA' hvs)⟩)
              exact ⟨⟨C, hC, hCh⟩, (he0.trans he).bindAt (by omega) vs, hk.altR⟩
            · intro d Γ' vs w' ⟨⟨A', avs, hm, h1, h2⟩, he, hk⟩
              exact ⟨⟨A', avs, List.mem_append_right _ hm, h1, h2⟩, he0.trans he, hk.altR⟩
            · intro d vs w' ⟨⟨A', args, Γ', hm, h1, h2, he⟩, hk⟩
              exact ⟨⟨A', args, Γ', List.mem_append_right _ hm, h1, h2, he0.trans he⟩, hk.altR⟩
        · cases h
      · cases h
      · cases h
  | path, A, .loop l pre body, o, h, Γ0, w0, hA => by
      have hn : Γ0.size = A.len := hA.1.1
      simp only [execP] at h
      split at h
      · rename_i hk
        split at h
        · cases h
        rename_i o1 hrun
        cases h
        obtain ⟨ivs, c, backs, exits, out, j, hc0, hlen, ht, hback, hj, rfl⟩ := runLoopIv_sound hrun
        dsimp only at ht
        split at ht
        · cases ht
        · rename_i op hop
          obtain ⟨hcB, hbB, hdB, hdC, htr, hq, hsome⟩ := loopTrip_spec ht
          -- a counted trip leaves only by its test
          have hnp : ∀ q, out.cnt = some q → op.brk = [] ∧ op.cont = [] := fun q e => by
            obtain ⟨Ap0, Ag0, ob0, -, -, -, htc⟩ := hq q e
            obtain ⟨jc, n, P, B⟩ := q
            exact ⟨(tripCnt_spec htc).2.1, (tripCnt_spec htc).2.2.1⟩
          have hck : c.vals.length = l.pTys.length := by rw [hlen]; simp [hk]
          have hpre := slotsOf_ge (A.len + l.pTys.length) pre
          have hab := slotsOf_ge (slotsOf (A.len + l.pTys.length) pre + l.pTys.length) body
          have hY : ∀ y ∈ loopCands A.len (pre ++ body), y < Γ0.size := fun y hy => hn ▸ loopCands_lt y hy
          have hab0 : Γ0.size ≤ slotsOf (slotsOf (A.len + l.pTys.length) pre + l.pTys.length) body := by omega
          refine loop_rule
            (fun _ cs w => c.Desc Γ0 cs w ∧ Kept (.star out.tr) w0 w ∧ CntInv A l out.cnt w0 cs w) ?_ ?_
          · intro Γ w cs ⟨e1, e2⟩ hcs
            subst Γ w
            exact ⟨hc0 Γ0 cs w0 ⟨((descAll_of hA hcs).portable).enrich hA.2.2, hA.2.2,
              closeSym_all hA hY (fun _ _ => rfl) hcs, ivsAll_entry hcs _⟩, Kept.starNil _, cntInv_entry hA hcs⟩
          · intro Γ w0' cs w ⟨e1, e2⟩ ⟨⟨hdc, hfc, hsc, hiv⟩, hkw, hcnt⟩
            subst Γ w0'
            have hcl : cs.length = l.pTys.length := hdc.length.symm.trans hck
            have hΓb : Ext Γ0 (bindAt Γ0 A.len cs) := (Ext.refl Γ0).bindAt (by omega) cs
            have hA1 : AS.Holds ((A.forgetHands.withFacts c.facts).bindAt A.len c.vals (c.headSyms A.len)) vals
                (bindAt Γ0 A.len cs) w :=
              (hA.forgetHands.worldNH (AS.forgetHands_noHands A) hfc fun r hr => hkw.rooms r hr).bindAtS A.len hdc.of_empty fun i hi => by
                have := headSyms_fits hsc hiv hdc.length.symm i (by rw [← hdc.length]; exact hi)
                rw [hn] at this
                rwa [← hdc.length] at this
            rw [hn]
            refine Hoare.conseq (execC_sound hop hA1) (fun _ _ h => h) ?_ ?_ ?_
            · intro Γ1 w1 ⟨⟨Ap, hApE, hAp⟩, he1, hk1⟩ t f hf
              obtain ⟨hexit, hgo⟩ := hsome Ap hApE
              have hcs0 : ∀ i, i < l.pTys.length → Γ1[A.len + i]? = cs[i]? := fun i hi => by
                rw [he1.2 _ (by rw [bindAt_size']; omega), Hoare.bindAt_at]
              refine ⟨fun hx vs hvs => ?_, fun hx => ?_⟩
              · have hb : (f != 0) = l.exitOnTrue := by simpa using hx
                obtain ⟨Ax, hr, hAx⟩ := refine_sound hAp hf hb
                have hE1 : Ext Γ0 Γ1 := hΓb.trans he1
                have hvl := mapM_len hvs
                obtain ⟨C, hC, hCh⟩ := joinAll_holds hj (hexit Ax hr)
                  (hAx.bindAtS _ (descAll_of hAx hvs) fun i hi => by
                    have := (exitSym_all' hAx hn.symm hY (fun y hy => hE1.slotVal hy) hvs).fits_bindAt
                      (hE1.bindAt hab0 vs) hab0 i (by simp at hi; omega)
                    simpa [hvl] using this)
                refine ⟨⟨C, hC, hCh⟩, hE1.bindAt hab0 vs, ?_⟩
                show Kept (cntTr A l (.star out.tr) out.cnt) w0 w1
                rcases cntTr_cases A l (.star out.tr) out.cnt with e | ⟨jc, n, P, B, c0, eq, hc0', hjl, hle, e⟩
                · rw [e]; exact hkw.starSnoc (hk1.mono htr)
                · rw [e]
                  obtain ⟨x, hxc, hlo, hhi, hrep⟩ := hcnt jc n P B eq c0 hc0' hjl hle
                  obtain ⟨Ap0, Ag0, ob0, hAp0, -, -, htc⟩ := hq _ eq
                  rw [hApE] at hAp0; cases hAp0
                  obtain ⟨hxt, -, -, -, -, rfl, rfl, hjk, ⟨y, hfl, hy⟩, -⟩ := tripCnt_spec htc
                  have hfe := counted_flag hAp hfl hy hf (by rw [hcs0 jc hjk]; exact hxc)
                  rw [hxt] at hb
                  rw [hb] at hfe
                  have hxn : x.toNat = n.toNat := by
                    have : ¬ x.toNat < n.toNat := by simpa using hfe.symm
                    omega
                  rw [← hxn]
                  exact hrep.seq hk1
              · have hb : (f != 0) = !l.exitOnTrue := by
                  revert hx; cases (f != 0) <;> cases l.exitOnTrue <;> simp
                obtain ⟨Ag, hr, hAg⟩ := refine_sound hAp hf hb
                obtain ⟨ob, hob, hbc, hbe, hbB', hbC', hbody, hfall⟩ := hgo Ag hr
                have hnb : ∀ q, out.cnt = some q → ob.brk = [] ∧ ob.cont = [] := fun q e => by
                  obtain ⟨Ap0, Ag0, ob0, hAp0, hAg0, hob0, htc⟩ := hq q e
                  rw [hApE] at hAp0; cases hAp0
                  rw [hr] at hAg0; cases hAg0
                  rw [hob] at hob0; cases hob0
                  obtain ⟨jc, n, P, B⟩ := q
                  exact ⟨(tripCnt_spec htc).2.2.2.1, (tripCnt_spec htc).2.2.2.2.1⟩
                have hcs : DescAll Γ1 w1
                    ((List.range l.pTys.length).map fun i => Ag.get (A.len + i)) cs := by
                  refine descAll_of_get (by simp [hcl]) fun i hi v hv => ?_
                  simp only [List.length_map, List.length_range] at hi
                  have e : ((List.range l.pTys.length).map fun i => Ag.get (A.len + i)).getD i .any
                      = Ag.get (A.len + i) := by
                    simp [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hi]
                  rw [e]
                  have h1 : Γ1[A.len + i]? = some v := by
                    rw [he1.2 _ (by rw [bindAt_size']; omega), Hoare.bindAt_at, hv]
                  exact hAg.desc h1
                have hΓb2 : Ext Γ0 (bindAt Γ1 (slotsOf (A.len + l.pTys.length) pre) cs) :=
                  (hΓb.trans he1).bindAt (by omega) cs
                have hcs1 : ∀ i, i < l.pTys.length → Γ1[A.len + i]? = cs[i]? := fun i hi => by
                  rw [he1.2 _ (by rw [bindAt_size']; omega), Hoare.bindAt_at]
                refine Hoare.conseq (execC_sound hob (hAg.bindAtS _ hcs (by
                    simpa using copySyms_fits hcs1 hcl (by omega))))
                  (fun _ _ h => h) ?_ ?_ ?_
                · intro Γ2 w2 ⟨⟨A2, hA2E, hA2⟩, he2, hk2⟩ next hnext
                  refine ⟨hback _ (hfall A2 hA2E) Γ0 next w2
                      (backCarry_desc hA2 hn.symm hY (fun y hy => (hΓb2.trans he2).slotVal hy) hnext),
                    hkw.starSnoc ((hk1.seq hk2).mono hbody), ?_⟩
                  intro jc n P B eq c0 hc0' hjl hle
                  obtain ⟨x, hxc, hlo, hhi, hrep⟩ := hcnt jc n P B eq c0 hc0' hjl hle
                  obtain ⟨Ap0, Ag0, ob0, hAp0, hAg0, hob0, htc⟩ := hq _ eq
                  rw [hApE] at hAp0; cases hAp0
                  rw [hr] at hAg0; cases hAg0
                  rw [hob] at hob0; cases hob0
                  obtain ⟨hxt, -, -, -, -, rfl, rfl, hjk, ⟨y, hfl, hy⟩, hstep⟩ := tripCnt_spec htc
                  obtain ⟨hjc, hty, hst⟩ := hstep A2 hA2E
                  have hx1 : Γ1[A.len + jc]? = some (.sc .i64 x) := by rw [hcs0 jc hjk]; exact hxc
                  have hfe := counted_flag hAp hfl hy hf hx1
                  rw [hxt] at hb
                  have hlt : x.toNat < n.toNat := by
                    have : (f != 0) = true := by simpa using hb
                    rw [this] at hfe
                    simpa using hfe.symm
                  -- the carry it goes round with is one more
                  obtain ⟨x', hx', -⟩ := hA2.getSc hty
                  have hs := stepsOne_sound hA2 hst
                  have hsz : A.len + jc < Γ1.size := by
                    rcases Nat.lt_or_ge (A.len + jc) Γ1.size with h' | h'
                    · exact h'
                    · rw [Array.getElem?_eq_none h'] at hx1; cases hx1
                  have hsx : slotVal Γ2 (A.len + jc) = x.toNat :=
                    ((he2.slotVal (by rw [bindAt_size']; omega)).trans (slotVal_bindAt (by omega) hsz)).trans
                      (slotVal_of hx1)
                  rw [slotVal_of hx', hsx] at hs
                  have hx'n : x'.toNat = x.toNat + 1 := by
                    have h1 := x'.toNat_lt
                    have h2 := n.toNat_lt
                    simp only [vVal] at hs
                    omega
                  have hnl := mapM_len hnext
                  have hg := mapM_getD hnext jc (by omega)
                  rw [hx'] at hg
                  refine ⟨x', by rw [getD_getElem? (by omega)]; exact hg.symm, by omega, by omega, ?_⟩
                  have e : x'.toNat - c0.toNat = (x.toNat - c0.toNat) + 1 := by omega
                  rw [e]
                  exact hrep.repSucc (hk1.seq hk2)
                · intro d Γb vs w' ⟨⟨Ab, args, hm, hAb, hvs⟩, he', hk'⟩
                  cases d with
                  | zero =>
                      have hE : Ext Γ0 Γb := hΓb2.trans he'
                      have hvl := mapM_len hvs
                      obtain ⟨C, hC, hCh⟩ := joinAll_holds hj (hbe _ (mem_brks0 hm))
                        (hAb.bindAtS _ (descAll_of hAb hvs) fun i hi => by
                          have := (exitSym_all' hAb hn.symm hY (fun y hy => hE.slotVal hy) hvs).fits_bindAt
                            (hE.bindAt hab0 vs) hab0 i (by simp at hi; omega)
                          simpa [hvl] using this)
                      exact ⟨⟨C, hC, hCh⟩, hE.bindAt hab0 vs,
                        (hkw.starSnoc ((hk1.seq hk').mono hbody)).uncounted fun q e => by
                          rw [(hnb q e).1] at hm; cases hm⟩
                  | succ d =>
                      exact ⟨⟨Ab, args, hbB' _ (mem_depthDown hm), hAb, hvs⟩, hΓb2.trans he',
                        (hkw.starSnoc ((hk1.seq hk').mono hbody)).uncounted fun q e => by
                          rw [(hnb q e).1] at hm; cases hm⟩
                · intro d vs w' ⟨⟨Ac, args, Γc, hm, hAc, hvs, hec⟩, hk'⟩
                  cases d with
                  | zero =>
                      exact ⟨hback _ (hbc _ (mem_conts0 hm)) Γ0 vs w'
                          (backCarry_desc hAc hn.symm hY (fun y hy => (hΓb2.trans hec).slotVal hy) hvs),
                        hkw.starSnoc ((hk1.seq hk').mono hbody), fun jc n P B e => by
                          rw [(hnb _ e).2] at hm; cases hm⟩
                  | succ d =>
                      exact ⟨⟨Ac, args, Γc, hbC' _ (mem_depthDown hm), hAc, hvs, hΓb2.trans hec⟩,
                        (hkw.starSnoc ((hk1.seq hk').mono hbody)).uncounted fun q e => by
                          rw [(hnb q e).2] at hm; cases hm⟩
            · intro d Γb vs w' ⟨⟨Ab, args, hm, hAb, hvs⟩, he', hk'⟩
              cases d with
              | zero =>
                  have hE : Ext Γ0 Γb := hΓb.trans he'
                  have hvl := mapM_len hvs
                  obtain ⟨C, hC, hCh⟩ := joinAll_holds hj (hbB _ (mem_brks0 hm))
                    (hAb.bindAtS _ (descAll_of hAb hvs) fun i hi => by
                      have := (exitSym_all' hAb hn.symm hY (fun y hy => hE.slotVal hy) hvs).fits_bindAt
                        (hE.bindAt hab0 vs) hab0 i (by simp at hi; omega)
                      simpa [hvl] using this)
                  exact ⟨⟨C, hC, hCh⟩, hE.bindAt hab0 vs,
                    (hkw.starSnoc (hk'.mono htr)).uncounted fun q e => by rw [(hnp q e).1] at hm; cases hm⟩
              | succ d =>
                  exact ⟨⟨Ab, args, hdB _ (mem_depthDown hm), hAb, hvs⟩, hΓb.trans he',
                    (hkw.starSnoc (hk'.mono htr)).uncounted fun q e => by rw [(hnp q e).1] at hm; cases hm⟩
            · intro d vs w' ⟨⟨Ac, args, Γc, hm, hAc, hvs, hec⟩, hk'⟩
              cases d with
              | zero =>
                  exact ⟨hback _ (hcB _ (mem_conts0 hm)) Γ0 vs w'
                    (backCarry_desc hAc hn.symm hY (fun y hy => (hΓb.trans hec).slotVal hy) hvs),
                    hkw.starSnoc (hk'.mono htr), fun jc n P B e => by rw [(hnp _ e).2] at hm; cases hm⟩
              | succ d =>
                  exact ⟨⟨Ac, args, Γc, hdC _ (mem_depthDown hm), hAc, hvs, hΓb.trans hec⟩,
                    (hkw.starSnoc (hk'.mono htr)).uncounted fun q e => by rw [(hnp q e).2] at hm; cases hm⟩
      · cases h
  | path, A, .dloop l body, o, h, Γ0, w0, hA => by
      have hn : Γ0.size = A.len := hA.1.1
      simp only [execP] at h
      split at h
      · rename_i hk
        have hab := slotsOf_ge (A.len + l.pTys.length) body
        have hY : ∀ y ∈ loopCands A.len body, y < Γ0.size := fun y hy => hn ▸ loopCands_lt y hy
        have hab0 : Γ0.size ≤ slotsOf (A.len + l.pTys.length) body := by omega
        have hΓs : ∀ outs, Ext Γ0 (bindAt Γ0 (slotsOf (A.len + l.pTys.length) body) outs) :=
          fun outs => (Ext.refl Γ0).bindAt hab0 outs
        -- the loop skipped: its exits bound from the carries it would be entered with
        have hskip : ∀ g t f cs, l.guard = some g → Γ0[g]? = some (.sc t f) →
            ((f != 0) == l.contOnTrue) = false → l.init.mapM (fun r => Γ0[r]?) = some cs →
            ∀ outs, l.exitIdx.mapM (fun i => cs[i]?) = some outs →
            ∃ C, dSkip A l (slotsOf (A.len + l.pTys.length) body) (loopCands A.len body) = some C ∧
              C.Holds vals (bindAt Γ0 (slotsOf (A.len + l.pTys.length) body) outs) w0 := by
          intro g t f cs hg hf hx hcs outs houts
          have hb : (f != 0) = !l.contOnTrue := by
            revert hx; cases (f != 0) <;> cases l.contOnTrue <;> simp
          obtain ⟨Ax, hr, hAx⟩ := refine_sound hA hf hb
          have hpk := descAll_pick (descAll_of hAx hcs) houts
          refine ⟨_, by simp [dSkip, hg, hr], hAx.bindAtS (ss := l.exitIdx.map fun i =>
            (l.init.map (exitSym Ax (loopCands A.len body) A.len)).getD i {}) _ hpk fun i hi => ?_⟩
          have hsy := ((exitSym_all' hAx hn.symm hY (fun _ _ => rfl) hcs).pick houts).fits_bindAt (hΓs outs) hab0 i
            (by rw [← hpk.length]; exact hi)
          rwa [← hpk.length] at hsy
        cases he : dEnter A l with
        | none =>
            rw [he] at h
            simp only [Except.ok.injEq] at h
            subst h
            refine dloop_rule (fun _ _ _ => False) ?_ (fun _ _ _ _ _ h => h.elim)
            intro Γ w cs ⟨e1, e2⟩ hcs
            subst Γ w
            cases hg : l.guard with
            | none => simp [dEnter, hg] at he
            | some g =>
                intro t f hf
                refine ⟨fun hx => ?_, fun hx outs houts => ?_⟩
                · have hb : (f != 0) = l.contOnTrue := by simpa using hx
                  obtain ⟨Ae, hr, -⟩ := refine_sound hA hf hb
                  simp [dEnter, hg, hr] at he
                · obtain ⟨C, hC, hCh⟩ := hskip g t f cs hg hf hx hcs outs houts
                  rw [hn]
                  exact ⟨⟨C, hC, hCh⟩, hΓs outs, Kept.refl _⟩
        | some Ae =>
            rw [he] at h
            dsimp only at h
            split at h
            · cases h
            · rename_i o1 hrun
              split at h
              · rename_i j hj
                cases h
                obtain ⟨ivs, c, backs, exits, out, j', hc0, hlen, ht, hback, hj', rfl⟩ := runLoopIv_sound hrun
                dsimp only at ht
                split at ht
                · cases ht
                · rename_i ob hob
                  simp only [Except.ok.injEq] at ht
                  obtain ⟨hcB, hbB, hdB, hdC, htr, hsome⟩ := dloopTrip_spec ht
                  have hck : c.vals.length = l.pTys.length := by rw [hlen]; simp [hk]
                  -- where the loop is entered, the state entered with describes it
                  have hent : ∀ cs, l.init.mapM (fun r => Γ0[r]?) = some cs → Ae.Holds vals Γ0 w0 →
                      c.Desc Γ0 cs w0 := fun cs hcs hAe =>
                    hc0 Γ0 cs w0 ⟨((descAll_of hAe hcs).portable).enrich hAe.2.2, hAe.2.2,
                      closeSym_all hAe hY (fun _ _ => rfl) hcs, ivsAll_entry hcs _⟩
                  have hexit : ∀ {Γ' : Env} {w' : World} {C : AS}, C ∈ exits → C.Holds vals Γ' w' →
                      Ext Γ0 Γ' → Kept (.star out.tr) w0 w' →
                      (∃ D, j = some D ∧ D.Holds vals Γ' w') ∧ Ext Γ0 Γ' ∧ Kept (.star out.tr) w0 w' :=
                    fun hm hC he hk => by
                      obtain ⟨C', hC', hC'h⟩ := joinAll_holds hj' hm hC
                      obtain ⟨D, hD, hDh⟩ := joinOpt_holds hj (.inl ⟨C', hC', hC'h⟩)
                      exact ⟨⟨D, hD, hDh⟩, he, hk⟩
                  refine dloop_rule
                    (fun _ cs w => c.Desc Γ0 cs w ∧ Kept (.star out.tr) w0 w ∧ Ae.Holds vals Γ0 w0) ?_ ?_
                  · intro Γ w cs ⟨e1, e2⟩ hcs
                    subst Γ w
                    cases hg : l.guard with
                    | none =>
                        have : Ae = A := by simpa [dEnter, hg] using he.symm
                        subst this
                        exact ⟨hent cs hcs hA, Kept.starNil _, hA⟩
                    | some g =>
                        intro t f hf
                        refine ⟨fun hx => ?_, fun hx outs houts => ?_⟩
                        · have hb : (f != 0) = l.contOnTrue := by simpa using hx
                          obtain ⟨Ae', hr, hAe'⟩ := refine_sound hA hf hb
                          have : Ae' = Ae := by simp [dEnter, hg, hr] at he; exact he
                          subst this
                          exact ⟨hent cs hcs hAe', Kept.starNil _, hAe'⟩
                        · obtain ⟨C, hC, hCh⟩ := hskip g t f cs hg hf hx hcs outs houts
                          rw [hn]
                          obtain ⟨D, hD, hDh⟩ := joinOpt_holds hj (.inr ⟨C, hC, hCh⟩)
                          exact ⟨⟨D, hD, hDh⟩, hΓs outs, Kept.starNil _⟩
                  · intro Γ w0' cs w ⟨e1, e2⟩ ⟨⟨hdc, hfc, hsc, hiv⟩, hkw, hAe⟩
                    subst Γ w0'
                    have hΓb : Ext Γ0 (bindAt Γ0 A.len cs) := (Ext.refl Γ0).bindAt (by omega) cs
                    have hA1 : AS.Holds ((Ae.forgetHands.withFacts c.facts).bindAt A.len c.vals (c.headSyms A.len)) vals
                        (bindAt Γ0 A.len cs) w :=
                      (hAe.forgetHands.worldNH (AS.forgetHands_noHands Ae) hfc fun r hr => hkw.rooms r hr).bindAtS A.len hdc.of_empty fun i hi => by
                        have := headSyms_fits hsc hiv hdc.length.symm i (by rw [← hdc.length]; exact hi)
                        rw [hn] at this
                        rwa [← hdc.length] at this
                    rw [hn]
                    refine Hoare.conseq (execC_sound hob hA1) (fun _ _ h => h) ?_ ?_ ?_
                    · intro Γ2 w2 ⟨⟨A2, hA2E, hA2⟩, he2, hk2⟩
                      obtain ⟨hgo, hstop⟩ := hsome A2 hA2E
                      have hE2 : Ext Γ0 Γ2 := hΓb.trans he2
                      refine ⟨fun next hnext t f hf hx => ?_, fun next hnext t f hf hx outs houts => ?_⟩
                      · have hb : (f != 0) = l.contOnTrue := by simpa using hx
                        obtain ⟨Ag, hr, hAg⟩ := refine_sound hA2 hf hb
                        exact ⟨hback _ (hgo Ag hr) Γ0 next w2
                            (backCarry_desc hAg hn.symm hY (fun y hy => hE2.slotVal hy) hnext),
                          hkw.starSnoc (hk2.mono htr), hAe⟩
                      · have hb : (f != 0) = !l.contOnTrue := by
                          revert hx; cases (f != 0) <;> cases l.contOnTrue <;> simp
                        obtain ⟨Ax, hr, hAx⟩ := refine_sound hA2 hf hb
                        have hpk := descAll_pick (descAll_of hAx hnext) houts
                        exact hexit (hstop Ax hr) (hAx.bindAtS _ hpk fun i hi => by
                            have hsy := ((exitSym_all' hAx hn.symm hY (fun y hy => hE2.slotVal hy) hnext).pick
                              houts).fits_bindAt (hE2.bindAt hab0 outs) hab0 i (by rw [← hpk.length]; exact hi)
                            rw [← hpk.length] at hsy
                            simpa [List.getD_eq_getElem?_getD, List.getElem?_map] using hsy)
                          (hE2.bindAt hab0 outs) (hkw.starSnoc (hk2.mono htr))
                    · intro d Γb vs w' ⟨⟨Ab, args, hm, hAb, hvs⟩, he', hk'⟩
                      cases d with
                      | zero =>
                          have hE : Ext Γ0 Γb := hΓb.trans he'
                          have hvl := mapM_len hvs
                          exact hexit (hbB _ (mem_brks0 hm)) (hAb.bindAtS _ (descAll_of hAb hvs) fun i hi => by
                              have := (exitSym_all' hAb hn.symm hY (fun y hy => hE.slotVal hy) hvs).fits_bindAt
                                (hE.bindAt hab0 vs) hab0 i (by simp at hi; omega)
                              simpa [hvl] using this)
                            (hE.bindAt hab0 vs) (hkw.starSnoc (hk'.mono htr))
                      | succ d =>
                          exact ⟨⟨Ab, args, hdB _ (mem_depthDown hm), hAb, hvs⟩, hΓb.trans he',
                            hkw.starSnoc (hk'.mono htr)⟩
                    · intro d vs w' ⟨⟨Ac, args, Γc, hm, hAc, hvs, hec⟩, hk'⟩
                      cases d with
                      | zero =>
                          exact ⟨hback _ (hcB _ (mem_conts0 hm)) Γ0 vs w'
                              (backCarry_desc hAc hn.symm hY (fun y hy => (hΓb.trans hec).slotVal hy) hvs),
                            hkw.starSnoc (hk'.mono htr), hAe⟩
                      | succ d =>
                          exact ⟨⟨Ac, args, Γc, hdC _ (mem_depthDown hm), hAc, hvs, hΓb.trans hec⟩,
                            hkw.starSnoc (hk'.mono htr)⟩
              · cases h
      · cases h

/-- **A region the checker accepts** ends, from any state the abstract one
    describes, as its result says. -/
theorem execC_sound {cfg : Cfg} {cert : Cert} :
    ∀ {base : List Nat} {i : Nat} {A : AS} {c : Code} {o : AOut}, execC cert base i A c = .ok o →
      ∀ {Γ0 : Env} {w0 : World}, A.Holds vals Γ0 w0 → Triple cfg (At Γ0 w0) c (o.post vals cert.fo Γ0 w0)
  | _, _, A, [], o, h, Γ0, w0, hA => by
      simp only [execC, Except.ok.injEq] at h
      subst h
      exact nil_rule fun Γ w ⟨e1, e2⟩ => by
        subst Γ w; exact ⟨⟨A, rfl, hA⟩, Ext.refl _, Kept.refl _⟩
  | _, _, A0, p :: ps, o, h, Γ0, w0, hA => by
      simp only [execC] at h
      split at h
      · cases h
      · rename_i A hpin
        obtain rfl := pin_eq hpin
        split at h
        · rename_i C1 C2 hpc
          obtain ⟨ss, rfl, hsc⟩ : ∃ ss, p = .straight ss ∧ splitCases cert.fo A ss = some (C1, C2) := by
            cases p <;> simp only [pieceCases, reduceCtorEq] at hpc
            exact ⟨_, rfl, hpc⟩
          split at h
          · rename_i o1 o2 h1 h2
            split at h
            · rename_i j hj
              cases h
              refine cons_rule (R := fun Γ w => Within vals C1 .any Γ0 w0 Γ w ∨ Within vals C2 .any Γ0 w0 Γ w)
                (straight_rule (stmts_conseq (splitCases_frame (t := .lit []) hsc)
                  (fun Γ w ⟨e1, e2⟩ => by subst Γ w; exact ⟨hA, Ext.refl _, Kept.refl _⟩)
                  (fun _ _ h => h)))
                (triple_of_at fun Γ1 w1 hR => ?_)
              rcases hR with ⟨hC, he, hk⟩ | ⟨hC, he, hk⟩
              · refine Hoare.conseq (execC_sound h1 hC) (fun _ _ h => h) ?_ ?_ ?_
                · intro Γ w ⟨⟨A', e, hA'⟩, he', hk'⟩
                  exact ⟨joinOpt_holds hj (.inl ⟨A', e, hA'⟩), he.trans he', hk.seq hk'.altL⟩
                · intro d Γ vs w ⟨⟨Ab, avs, hm, h1, h2⟩, he', hk'⟩
                  exact ⟨⟨Ab, avs, List.mem_append_left _ hm, h1, h2⟩, he.trans he', hk.seq hk'.altL⟩
                · intro d vs w ⟨⟨Ac, args, Γc, hm, h1, h2, hec⟩, hk'⟩
                  exact ⟨⟨Ac, args, Γc, List.mem_append_left _ hm, h1, h2, he.trans hec⟩, hk.seq hk'.altL⟩
              · refine Hoare.conseq (execC_sound h2 hC) (fun _ _ h => h) ?_ ?_ ?_
                · intro Γ w ⟨⟨A', e, hA'⟩, he', hk'⟩
                  exact ⟨joinOpt_holds hj (.inr ⟨A', e, hA'⟩), he.trans he', hk.seq hk'.altR⟩
                · intro d Γ vs w ⟨⟨Ab, avs, hm, h1, h2⟩, he', hk'⟩
                  exact ⟨⟨Ab, avs, List.mem_append_right _ hm, h1, h2⟩, he.trans he', hk.seq hk'.altR⟩
                · intro d vs w ⟨⟨Ac, args, Γc, hm, h1, h2, hec⟩, hk'⟩
                  exact ⟨⟨Ac, args, Γc, List.mem_append_right _ hm, h1, h2, he.trans hec⟩, hk.seq hk'.altR⟩
            · cases h
          · cases h
          · cases h
        split at h
        · cases h
        · rename_i o1 h1
          have ih1 := execP_sound (cfg := cfg) h1 hA
          split at h
          · rename_i hn
            cases h
            exact cons_rule (R := fun _ _ => False)
              (pieceT_conseq ih1 (fun Γ w ⟨⟨A', e, _⟩, _⟩ => by rw [hn] at e; cases e)
                (fun _ _ _ _ h => h) (fun _ _ _ h => h) (fun h => h)) (fun _ _ _ h => h.elim)
          · rename_i A' hA'
            split at h
            · cases h
            · rename_i o2 h2
              cases h
              refine cons_rule (R := Within vals A' o1.tr Γ0 w0) (pieceT_conseq ih1 ?_ ?_ ?_ (fun h => h))
                (triple_of_at fun Γ1 w1 ⟨h1', he, hk⟩ => ?_)
              · intro Γ w ⟨⟨A1, e, hA1⟩, he, hk⟩
                rw [hA'] at e; cases e; exact ⟨hA1, he, hk⟩
              · intro d Γ vs w ⟨⟨Ab, avs, hm, h1, h2⟩, he, hk⟩
                exact ⟨⟨Ab, avs, List.mem_append_left _ hm, h1, h2⟩, he, hk.alone (notEmpty_left hm)⟩
              · intro d vs w ⟨⟨Ac, args, Γc, hm, h1, h2, hec⟩, hk⟩
                exact ⟨⟨Ac, args, Γc, List.mem_append_left _ hm, h1, h2, hec⟩, hk.alone (notEmpty_right hm)⟩
              · refine Hoare.conseq (execC_sound h2 h1') (fun _ _ h => h) ?_ ?_ ?_
                · intro Γ w ⟨hx, he', hk'⟩
                  exact ⟨hx, he.trans he', hk.then hk'⟩
                · intro d Γ vs w ⟨⟨Ab, avs, hm, h1, h2⟩, he', hk'⟩
                  exact ⟨⟨Ab, avs, List.mem_append_right _ hm, h1, h2⟩, he.trans he', hk.then hk'⟩
                · intro d vs w ⟨⟨Ac, args, Γc, hm, h1, h2, hec⟩, hk'⟩
                  exact ⟨⟨Ac, args, Γc, List.mem_append_right _ hm, h1, h2, he.trans hec⟩, hk.then hk'⟩

end

/-- **The checker is sound.** A body the checker accepts, run from arguments
    and a world its entry state describes, never misuses a call and, checked
    as one that may not fault, never reads or writes memory that is not there. -/
theorem execC_run {cfg : Cfg} {cert : Cert} {base : List Nat} {i : Nat} {A : AS} {c : Code} {o : AOut}
    (h : execC cert base i A c = .ok o) (hfo : cert.fo = false)
    {args : List V} {w : World} (hA : A.Holds vals args.toArray w) (m : String) :
    run cfg args w c ≠ .misuse m ∧ run cfg args w c ≠ .fault m :=
  run_sound (execC_sound h hA) hfo m

/-- A body the checker accepts, as one that may fault or not, never misuses a
    call. -/
theorem execC_safe {cfg : Cfg} {cert : Cert} {base : List Nat} {i : Nat} {A : AS} {c : Code} {o : AOut}
    (h : execC cert base i A c = .ok o)
    {args : List V} {w : World} (hA : A.Holds vals args.toArray w) (m : String) :
    run cfg args w c ≠ .misuse m :=
  run_safe (execC_sound h hA) m

/-- **The calls a checked body makes are ones its pattern admits.** Where a
    body the checker accepts ends, the calls it observed, in order, are the
    ones made before it and then a sequence the pattern the checker gives the
    body admits. -/
theorem execC_trace {cfg : Cfg} {cert : Cert} {base : List Nat} {i : Nat} {A : AS} {c : Code} {o : AOut}
    (h : execC cert base i A c = .ok o) {args : List V} {w : World} (hA : A.Holds vals args.toArray w)
    {obs : List Obs} {w' : World} (hr : run cfg args w c = .ok obs w') : o.tr.Covers w obs := by
  have hs := execC_sound (cfg := cfg) h hA cfg.steps args.toArray w ⟨rfl, rfl⟩
  simp only [Sem.run] at hr
  cases hc : runCode cfg.steps cfg args.toArray w c with
  | ok Γ' w1 =>
      rw [hc] at hr hs
      simp only [Outcome.ok.injEq] at hr
      obtain ⟨rfl, rfl⟩ := hr
      obtain ⟨cs, e, hh⟩ := hs.2.2.trace
      exact ⟨cs, by rw [callsOf_reverse, e, List.reverse_append, List.reverse_reverse], hh⟩
  | _ => rw [hc] at hr; cases hr

end AlgorithmLib.Check
