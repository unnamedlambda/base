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

open AlgorithmLib.HProg.Contracts (Fact TState holds_of_mem Part afterStore afterStore_sound storeKeeps
  storeKeeps_sound storeKeeps_cell_of storeKeeps_cstrIn_of cellOk cellOk_spec nonCell_kept store_sizes_eq
  store_frozen callBits_rooms after)

-- ---------------------------------------------------------------------------
-- Addresses as numbers
-- ---------------------------------------------------------------------------

/-- An address a region decodes, as a number: its base plus the offset. -/
theorem decode_toNat {a : UInt64} {r : Region} {off : Nat} (h : decodeAddr a = some (r, off)) :
    a.toNat = (regionBase r).toNat + off ∧ (regionBase r).toNat + off < 2 ^ 39 := by
  obtain ⟨rfl, ho⟩ := Static.decodeAddr_some h
  have hb := Static.regionBase_toNat r
  rw [Static.regionSpan_toNat] at ho hb
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

theorem AS.Holds.desc {A : AS} {Γ : Env} {w : World} (hA : A.Holds Γ w) {r : R} {u : V}
    (hr : Γ[r]? = some u) : (A.get r).Desc Γ w u := by
  have hl : r < A.len := by
    rw [← hA.1]
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
theorem AS.Holds.keep {A : AS} {Γ : Env} {w w' : World} (h : A.Holds Γ w) {S : TState}
    (hS : S.holds w') (hw : w'.mem.sizes = w.mem.sizes) : (A.withFacts S).Holds Γ w' :=
  h.world hS fun r _ => congrFun hw r

theorem AS.Holds.obsCall {A : AS} {Γ : Env} {w : World} (h : A.Holds Γ w) (c : Callee) (vs : List V) :
    A.Holds Γ (obsCall w c vs) :=
  h.world (AlgorithmLib.Prog.TState.holds_obsCall h.2.2) fun _ _ => rfl

theorem descAll_of {A : AS} {Γ : Env} {w : World} (hA : A.Holds Γ w) :
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
    {w : World} (hA : A.Holds Γ w) :
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
    {w : World} (hA : A.Holds Γ w) {t tv : ClifTy} {x b b' c : UInt64} {m : Mem} {k : Nat}
    (ha : Γ[a]? = some (.sc t x)) (hsc : (A.get v).scTy = some tv) (hv : Γ[v]? = some (.sc tv b))
    (hm : w.mem.store x n b' = some m) (hb : n = 8 → b' = b) :
    A'.Holds Γ { obsStore w x k c with mem := m } := by
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
  split at h
  next t0 x0 _ _ =>
    cases h
    simp only [AV.Desc, V.sc.injEq] at hda
    obtain ⟨-, rfl⟩ := hda
    refine hA.keep ?_ hsz
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
    refine hA.keep ?_ hsz
    have hk := storeKeeps_sound hA.2.2 hp (fun kk _ => by cases kk <;> rfl) rfl ⟨rfl, rfl, rfl⟩ hsz hm hn
    intro f hf
    obtain ⟨hf1, hf2⟩ := List.mem_filter.mp hf
    exact hk f (List.mem_filter.mpr ⟨hf1, rangeKeeps_store hf2 hlo hhi (by omega)⟩)
  all_goals contradiction

/-- **A vector store**: every fact but the cells survives. -/
theorem absStore_vec {A A' : AS} {v a : R} {n : Nat} (h : absStore A v a n = .ok A') {Γ : Env}
    {w : World} (hA : A.Holds Γ w) (hsc : (A.get v).scTy = none) {x c : UInt64} {m : Mem} {k : Nat}
    (hsz : m.sizes = w.mem.sizes) (hz : m.frozen = w.mem.frozen) :
    A'.Holds Γ { obsStore w x k c with mem := m } := by
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
           exact hA.keep (nonCell_kept hA.2.2 (store_parts hz) (fun kk _ => by cases kk <;> rfl) rfl
             ⟨rfl, rfl, rfl⟩ hsz) hsz)

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
    (h : absStore A v a (tyBytes ty) = .ok A') : Stmt1 cfg A.Holds (.store ty v a) A'.Holds fo := by
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
    Stmt1 cfg A.Holds (.storeUnaligned v a) A'.Holds fo := by
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
    (h : absStore A v a 1 = .ok A') : Stmt1 cfg A.Holds (.istore8 v a) A'.Holds fo := by
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
theorem wild_holds {A : AS} {Γ : Env} {w : World} (hA : A.Holds Γ w) {x c : UInt64} {k : Nat} {m : Mem}
    (hsz : m.sizes = w.mem.sizes) (hz : m.frozen = w.mem.frozen) :
    (A.withFacts (A.facts.filter (!·.isCell))).Holds Γ { obsStore w x k c with mem := m } :=
  hA.keep (nonCell_kept hA.2.2 (store_parts hz) (fun kk _ => by cases kk <;> rfl) rfl ⟨rfl, rfl, rfl⟩ hsz) hsz

theorem wild_store {cfg : Cfg} {A : AS} {ty : ClifTy} {v a : R} :
    Stmt1 cfg A.Holds (.store ty v a) (A.withFacts (A.facts.filter (!·.isCell))).Holds true := by
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
    Stmt1 cfg A.Holds (.storeUnaligned v a) (A.withFacts (A.facts.filter (!·.isCell))).Holds true := by
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
    Stmt1 cfg A.Holds (.istore8 v a) (A.withFacts (A.facts.filter (!·.isCell))).Holds true := by
  intro Γ w hA
  rw [runStmt_istore8]
  split
  · split
    · rename_i m hm; exact wild_holds hA (store_sizes_eq hm) (store_frozen hm)
    · rfl
  · trivial

theorem storeF_sound {cfg : Cfg} {A A' : AS} {ty : ClifTy} {v a : R} {fo : Bool}
    (h : absStoreF fo A v a (tyBytes ty) = .ok A') : Stmt1 cfg A.Holds (.store ty v a) A'.Holds fo := by
  unfold absStoreF at h
  split at h
  · rename_i A1 h1; cases h; exact store_sound h1
  · split at h
    · rename_i hfo; cases h; subst hfo; exact wild_store
    · cases h

theorem storeUF_sound {cfg : Cfg} {A A' : AS} {v a : R} {fo : Bool}
    (h : absStoreF fo A v a (tyBytes ((A.get v).scTy.getD .i64)) = .ok A') :
    Stmt1 cfg A.Holds (.storeUnaligned v a) A'.Holds fo := by
  unfold absStoreF at h
  split at h
  · rename_i A1 h1; cases h; exact storeU_sound h1
  · split at h
    · rename_i hfo; cases h; subst hfo; exact wild_storeU
    · cases h

theorem istore8F_sound {cfg : Cfg} {A A' : AS} {v a : R} {tv : ClifTy} {fo : Bool}
    (hsc : (A.get v).scTy = some tv) (h : absStoreF fo A v a 1 = .ok A') :
    Stmt1 cfg A.Holds (.istore8 v a) A'.Holds fo := by
  unfold absStoreF at h
  split at h
  · rename_i A1 h1; cases h; exact istore8_sound hsc h1
  · split at h
    · rename_i hfo; cases h; subst hfo; exact wild_istore8
    · cases h
-- ---------------------------------------------------------------------------
-- Calls
-- ---------------------------------------------------------------------------

theorem ffi_result_lanes (f : Ffi) : (f.result.getD .i64).lanes = none := by
  cases f <;> rfl

/-- **A foreign call the checker accepts** answers, from every state the
    abstract one describes, and leaves a world `postA` describes. -/
theorem call_core {cfg : Cfg} {A : AS} {f : Ffi} {args : List R} (hf : f ≠ .threadSpawn)
    (hz : A.facts.contains (.part .frozen false) = true)
    (hsc : (args.map A.get).all (fun a => a.scTy.isSome) = true)
    (hp : preB f (args.map A.get) A.facts = true) {Γ : Env} {w : World} {vs : List V}
    (hA : A.Holds Γ w) (hvs : args.mapM (fun r => Γ[r]?) = some vs) :
    ∃ r w', callOf cfg.locals (.ffi f) vs (obsCall w (.ffi f) vs) = some (r, w') ∧
      (A.withFacts (postA f (args.map A.get) A.facts)).Holds Γ w' ∧
      ∀ v, r = some v → ∃ x, v = .sc (f.result.getD .i64) x := by
  have hA1 := hA.obsCall (.ffi f) vs
  have hd := descAll_of hA1 hvs
  obtain ⟨bits, hb⟩ := bits_exist hd hsc
  have hab := argBits_of hd hb
  have hz' : (obsCall w (.ffi f) vs).mem.frozen = false := holds_of_mem hA1.2.2 hz
  have hpre := preB_sound hA1.2.2 hab hp
  obtain ⟨⟨r, w'⟩, hc⟩ := Option.isSome_iff_exists.mp (Contracts.pre_safe f bits _ hpre)
  have hcall : callOf cfg.locals (.ffi f) vs (obsCall w (.ffi f) vs) = some (r, w') := by
    rw [Contracts.callOf_ffi hf cfg.locals vs hz', hb]; exact hc
  refine ⟨r, w', hcall, hA1.world (postA_sound hab hA1.2.2 hc) (callBits_rooms f hc), fun v hv => ?_⟩
  subst hv
  exact Contracts.callOf_ffi_scalar hf hcall

theorem call_sound {cfg : Cfg} {A A' : AS} {c : Callee} {args : List R} {fo fa : Bool}
    (h : absStmt fa A (.call c args) = .ok A') : Stmt1 cfg A.Holds (.call c args) A'.Holds fo := by
  cases c with
  | ffi f =>
      simp only [absStmt] at h
      split at h
      · rename_i hc
        cases h
        simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at hc
        obtain ⟨⟨⟨hf, hz⟩, hsc⟩, hp⟩ := hc
        refine call_rule fun Γ w vs hA hvs => ?_
        obtain ⟨r, w', hcall, hA', hr⟩ := call_core hf hz hsc hp hA hvs
        refine ⟨r, w', hcall, fun v hv => ?_⟩
        obtain ⟨x, rfl⟩ := hr v hv
        exact hA'.push (av := .sc (f.result.getD .i64)) (TyV.sc (ffi_result_lanes f))
      · cases h
  | _ => simp [absStmt] at h

theorem callVoid_sound {cfg : Cfg} {A A' : AS} {c : Callee} {args : List R} {fo fa : Bool}
    (h : absStmt fa A (.callVoid c args) = .ok A') : Stmt1 cfg A.Holds (.callVoid c args) A'.Holds fo := by
  cases c with
  | ffi f =>
      simp only [absStmt] at h
      split at h
      · rename_i hc
        cases h
        simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at hc
        obtain ⟨⟨⟨hf, hz⟩, hsc⟩, hp⟩ := hc
        refine callVoid_rule fun Γ w vs hA hvs => ?_
        obtain ⟨r, w', hcall, hA', -⟩ := call_core hf hz hsc hp hA hvs
        exact ⟨r, w', hcall, hA'⟩
      · cases h
  | _ => simp [absStmt] at h

-- ---------------------------------------------------------------------------
-- Statements
-- ---------------------------------------------------------------------------

/-- **What the checker steps a statement to is true of it.** -/
theorem absStmt_sound {cfg : Cfg} {A A' : AS} {s : Stmt} {fo : Bool} (h : absStmt fo A s = .ok A') :
    Stmt1 cfg A.Holds s A'.Holds fo := by
  cases s with
  | op o =>
      simp only [absStmt] at h
      split at h
      · rename_i av hav
        cases h
        refine op_rule (fun Γ w v hA hv => ?_) (fun Γ w hA hn => ?_)
        · obtain ⟨v', hv', hd⟩ := absOp_sound hA hav
          rw [hv] at hv'; cases hv'
          exact hA.push hd
        · obtain ⟨v', hv', -⟩ := absOp_sound hA hav
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
      Stmts cfg A.Holds ss A'.Holds fo
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
    Stmts cfg A.Holds ss A'.Holds fo :=
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
  split at h
  · cases h; exact hx
  · cases x with
    | htVals n =>
        obtain ⟨m, -, rfl⟩ := Option.map_eq_some_iff.mp h
        intro k v hv
        exact Nat.le_trans (hx k v hv) (Nat.le_max_left _ _)
    | _ => simp at h

theorem meetFact_right {S' : TState} {x y : Fact} {w : World} (h : meetFact S' x = some y)
    (hS : S'.holds w) : y.holds w := by
  unfold meetFact at h
  split at h
  · rename_i hc
    cases h
    exact hS _ (by simpa using hc)
  · cases x with
    | htVals n =>
        obtain ⟨m, hm, rfl⟩ := Option.map_eq_some_iff.mp h
        have hv := hS _ (htValsOf_mem hm)
        intro k v hg
        exact Nat.le_trans (hv k v hg) (Nat.le_max_right _ _)
    | _ => simp at h

theorem factsMeet_left {S S' : TState} {w : World} (h : S.holds w) : (factsMeet S S').holds w := by
  intro y hy
  obtain ⟨x, hx, hxy⟩ := List.mem_filterMap.mp hy
  exact meetFact_left hxy (h x hx)

theorem factsMeet_right {S S' : TState} {w : World} (h : S'.holds w) : (factsMeet S S').holds w := by
  intro y hy
  obtain ⟨x, -, hxy⟩ := List.mem_filterMap.mp hy
  exact meetFact_right hxy h

theorem AS.join_holds {A B C : AS} (h : A.join B = .ok C) {Γ : Env} {w : World}
    (hAB : A.Holds Γ w ∨ B.Holds Γ w) : C.Holds Γ w := by
  unfold AS.join at h
  split at h
  · rename_i hl
    cases h
    have hs : Γ.size = A.len := by rcases hAB with hA | hB; exact hA.1; rw [hB.1, hl]
    refine ⟨hs, fun i hi => ?_, ?_⟩
    · have hi' : i < B.len := hl ▸ hi
      have e : AS.get ⟨Slots.zip AV.join A.slots B.slots, A.len, factsMeet A.facts B.facts⟩ i
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
    (hab : (∃ A, a = some A ∧ A.Holds Γ w) ∨ (∃ B, b = some B ∧ B.Holds Γ w)) :
    ∃ C, r = some C ∧ C.Holds Γ w := by
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
    ∀ {Γ : Env} {w : World}, A.Holds Γ w → ∃ C, r = some C ∧ C.Holds Γ w
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

theorem AS.foldl_push_len : ∀ (vs : List AV) (B : AS), (vs.foldl AS.push B).len = B.len + vs.length
  | [], _ => rfl
  | v :: vs, B => by
      rw [List.foldl_cons, AS.foldl_push_len vs (B.push v)]
      simp only [AS.push, List.length_cons]
      omega

theorem AS.foldl_push_facts : ∀ (vs : List AV) (B : AS), (vs.foldl AS.push B).facts = B.facts
  | [], _ => rfl
  | v :: vs, B => by rw [List.foldl_cons, AS.foldl_push_facts vs (B.push v)]; rfl

theorem AS.foldl_push_get : ∀ (vs : List AV) (B : AS) (i : Nat),
    (vs.foldl AS.push B).get i = if i < B.len then B.get i else vs.getD (i - B.len) .any
  | [], B, i => by
      simp only [List.foldl_nil, List.getD_nil]
      split
      · rfl
      · rename_i h; exact AS.get_out h
  | v :: vs, B, i => by
      rw [List.foldl_cons, AS.foldl_push_get vs (B.push v) i]
      have hl : (B.push v).len = B.len + 1 := rfl
      rw [hl]
      by_cases h1 : i < B.len
      · rw [if_pos (by omega), if_pos h1, AS.get_push_lt h1]
      · by_cases h2 : i = B.len
        · subst h2
          rw [if_pos (by omega), if_neg h1, AS.get_push_eq, Nat.sub_self]
          rfl
        · rw [if_neg (by omega), if_neg h1]
          obtain ⟨k, hk⟩ : ∃ k, i - B.len = k + 1 := ⟨i - B.len - 1, by omega⟩
          rw [hk, show i - (B.len + 1) = k by omega]
          rfl

theorem Slots.get_clear : ∀ (k : Nat) (t : Slots) (i j : Nat),
    (t.clear i k).get j = if i ≤ j ∧ j < i + k then .any else t.get j
  | 0, t, i, j => by
      simp only [Slots.clear]
      rw [if_neg (by omega)]
  | k + 1, t, i, j => by
      simp only [Slots.clear]
      rw [Slots.get_clear k (t.set i .any) (i + 1) j, Slots.get_set]
      by_cases h1 : i + 1 ≤ j ∧ j < i + 1 + k
      · rw [if_pos h1, if_pos (by omega)]
      · rw [if_neg h1]
        by_cases h2 : j = i
        · rw [if_pos h2, if_pos (by omega)]
        · rw [if_neg h2, if_neg (by omega)]

theorem AS.bindAt_len (A : AS) (n : Nat) (vs : List AV) : (A.bindAt n vs).len = n + vs.length := by
  unfold AS.bindAt
  rw [AS.foldl_push_len, List.length_map]
  split <;> rfl

theorem AS.bindAt_facts (A : AS) (n : Nat) (vs : List AV) : (A.bindAt n vs).facts = A.facts := by
  unfold AS.bindAt
  rw [AS.foldl_push_facts]
  split <;> rfl

theorem AS.bindAt_get_lt {A : AS} {n i : Nat} {vs : List AV} (hi : i < n) :
    (A.bindAt n vs).get i = if A.len ≤ n then A.get i else AV.cut n (A.get i) := by
  unfold AS.bindAt
  rw [AS.foldl_push_get]
  split
  · rename_i hle
    have e : AS.get ⟨A.slots.clear A.len (n - A.len), n, A.facts⟩ i = A.get i := by
      unfold AS.get
      simp only
      rw [if_pos hi, Slots.get_clear]
      by_cases h : i < A.len
      · rw [if_neg (by omega), if_pos h]
      · rw [if_pos (by omega), if_neg h]
    first | exact e | (rw [if_pos (show i < n from hi)]; exact e)
  · rename_i hle
    have e : AS.get ⟨A.slots.map (AV.cut n), n, A.facts⟩ i = AV.cut n (A.get i) := by
      unfold AS.get
      simp only
      rw [if_pos hi, Slots.get_map (by rfl), if_pos (show i < A.len by omega)]
    first | exact e | (rw [if_pos (show i < n from hi)]; exact e)

theorem AS.bindAt_get_ge {A : AS} {n j : Nat} {vs : List AV} :
    (A.bindAt n vs).get (n + j) = (vs.getD j .any).portable := by
  unfold AS.bindAt
  rw [AS.foldl_push_get]
  have hb : ∀ B : AS, B.len = n → (if n + j < B.len then B.get (n + j)
      else (vs.map AV.portable).getD (n + j - B.len) .any) = (vs.getD j .any).portable := by
    intro B hB
    rw [hB, if_neg (by omega), Nat.add_sub_cancel_left]
    simp only [List.getD_eq_getElem?_getD, List.getElem?_map]
    cases vs[j]? <;> rfl
  split
  · exact hb _ rfl
  · exact hb _ rfl

/-- A description read in an environment that keeps every slot it had. -/
theorem AV.desc_extend {Γ Γ' : Env} {w : World} {av : AV} {v : V} (hd : av.Desc Γ w v)
    (hΓ : ∀ a, a < Γ.size → Γ'[a]? = Γ[a]?) : av.Desc Γ' w v := by
  refine AV.desc_transport hd (fun a ha => ?_) (fun _ _ => rfl)
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
      have := AV.desc_transport (Γ' := Γ') hd (fun _ h => by simp [AV.refs] at h) (fun _ _ => rfl)
      exact this

theorem bindAt_size' (Γ : Env) (n : Nat) (vs : List V) : (bindAt Γ n vs).size = n + vs.length := by
  simp only [bindAt, Array.size_append, Array.take_eq_extract, Array.size_extract, Array.size_replicate,
    List.size_toArray]
  omega

/-- **Binding, abstractly and concretely**: `AS.bindAt` describes `bindAt`. -/
theorem AS.Holds.bindAt {A : AS} {Γ : Env} {w : World} (hA : A.Holds Γ w) (n : Nat) {avs : List AV}
    {vs : List V} (hd : DescAll Γ w avs vs) : (A.bindAt n avs).Holds (Sem.bindAt Γ n vs) w := by
  have hlen := hd.length
  refine ⟨by rw [bindAt_size', AS.bindAt_len, hlen], fun i hi => ?_, by rw [AS.bindAt_facts]; exact hA.2.2⟩
  rw [AS.bindAt_len] at hi
  by_cases hn : i < n
  · rw [AS.bindAt_get_lt hn, bindAt_get?, if_pos hn]
    by_cases hs : i < Γ.size
    · rw [if_pos hs]
      obtain ⟨v, hv, hdv⟩ := hA.get (hA.1 ▸ hs)
      refine ⟨v, hv, ?_⟩
      split
      · rename_i hle
        exact AV.desc_extend hdv fun a has => Hoare.bindAt_lt (by have := hA.1; omega) has
      · exact AV.cut_desc hdv fun a ha has => Hoare.bindAt_lt ha has
    · rw [if_neg hs, AS.get_out (by rw [← hA.1]; exact hs)]
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
theorem unsignedCond_sound {A : AS} {Γ : Env} {w : World} (hA : A.Holds Γ w) {c : ICmpCond} {x y : R}
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
  | _, _, _ => True

theorem holdsRel_sound {A : AS} {Γ : Env} {w : World} (hA : A.Holds Γ w) {c : ICmpCond} {x y : R}
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
      bne_eq_false_iff_eq] at hc <;> simp only [RelHolds] <;> first | omega | rw [hc]

theorem AS.Holds.setAV {A : AS} {Γ : Env} {w : World} (hA : A.Holds Γ w) {r : R} {av : AV}
    (hv : ∀ v, Γ[r]? = some v → av.Desc Γ w v) : (A.setAV r av).Holds Γ w := by
  unfold AS.setAV
  split
  · exact hA
  · split
    · rename_i hr
      refine ⟨hA.1, fun i hi => ?_, hA.2.2⟩
      obtain ⟨u, hu, hd⟩ := hA.get hi
      refine ⟨u, hu, ?_⟩
      have e : AS.get ⟨A.slots.set r av, A.len, A.facts⟩ i = if i = r then av else A.get i := by
        unfold AS.get
        rw [if_pos hi, Slots.get_set]
        split
        · rfl
        · first | rfl | rw [if_pos hi]
      rw [e]
      split
      · rename_i h; subst h
        exact hv u hu
      · exact hd
    · exact hA

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

theorem lessThan_holds {A : AS} {Γ : Env} {w : World} (hA : A.Holds Γ w) {x y : R} {tx ty : ClifTy}
    {a b : UInt64} {e : Nat} (hx : Γ[x]? = some (.sc tx a)) (hy : Γ[y]? = some (.sc ty b)) (he : e ≤ 1)
    (hab : a.toNat + e ≤ b.toNat) : (lessThan A x y e).Holds Γ w := by
  unfold lessThan
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
      split
      · rename_i k r s' d hk hfit
        have hak := Num.exact_eq hxa hk
        have hf := hyb.2.2.2
        rw [hfit] at hf
        refine ⟨hA2.1, hA2.2.1, TState.holds_cons ?_ hA2.2.2⟩
        show (k + s) * s' + d ≤ w.mem.sizes r
        have hm := Nat.mul_le_mul_right s' (show k + s ≤ zb by omega)
        have := hf.2
        omega
      · exact hA2
    · exact hA
  · exact hA

theorem learn_holds {A : AS} {Γ : Env} {w : World} (hA : A.Holds Γ w) {rel : ICmpCond} {x y : R}
    {tx ty : ClifTy} {a b : UInt64} (hx : Γ[x]? = some (.sc tx a)) (hy : Γ[y]? = some (.sc ty b))
    (hr : RelHolds rel a.toNat b.toNat) : (learn A rel x y).Holds Γ w := by
  cases rel <;> simp only [learn, RelHolds] at hr ⊢
  · exact lessThan_holds (lessThan_holds hA hx hy (by decide) (by omega)) hy hx (by decide) (by omega)
  · exact hA
  · exact lessThan_holds hA hy hx (by decide) (by omega)
  · exact lessThan_holds hA hy hx (by decide) (by omega)
  · exact lessThan_holds hA hx hy (by decide) (by omega)
  · exact lessThan_holds hA hx hy (by decide) (by omega)
  all_goals exact hA

/-- **What a branch learns is true**: where the flag is `b`, `refine` gives a
    state describing the one the branch starts from. -/
theorem refine_sound {A : AS} {Γ : Env} {w : World} (hA : A.Holds Γ w) {flag : R} {t : ClifTy}
    {f : UInt64} {b : Bool} (hf : Γ[flag]? = some (.sc t f)) (hb : (f != 0) = b) :
    ∃ A', refine A flag b = some A' ∧ A'.Holds Γ w := by
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

/-- The carries and the world a loop's carry state describes. Its values read
    no slot, so they are read in the empty environment. -/
def Carry.Desc (c : Carry) (cs : List V) (w : World) : Prop :=
  DescAll #[] w c.vals cs ∧ c.facts.holds w

/-- Every carry state `b` describes, `c` does. -/
def Carry.Le (b c : Carry) : Prop := ∀ cs w, b.Desc cs w → c.Desc cs w

theorem Carry.Le.refl (c : Carry) : c.Le c := fun _ _ h => h

theorem Carry.Le.trans {a b c : Carry} (h1 : a.Le b) (h2 : b.Le c) : a.Le c :=
  fun cs w h => h2 cs w (h1 cs w h)

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
  fun _ _ ⟨hd, hf⟩ => ⟨descAll_zip (fun _ _ _ h => AV.join_left h) h hd,
    factsMeet_left hf⟩

theorem Carry.join_right {a b : Carry} (h : a.vals.length = b.vals.length) : b.Le (a.join b) :=
  fun _ _ ⟨hd, hf⟩ => ⟨descAll_zip_right (fun _ _ _ h => AV.join_right h) h hd,
    factsMeet_right hf⟩

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
  fun _ _ ⟨hd, hf⟩ => ⟨descAll_zip_right (fun _ _ _ h => AV.widen_desc h) h hd, hf⟩

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
  simp only [Carry.beq, Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨hl, hv⟩, -⟩, hf⟩ := h
  intro cs w ⟨hd, hfa⟩
  refine ⟨?_, fun x hx => hfa x (by simpa using List.all_eq_true.mp hf x hx)⟩
  rw [← eq_of_zip_beq hl hv]
  exact hd

/-- What checking a loop gives: an invariant covering the carries it is
    entered with, from which one trip goes round only with states it covers. -/
def LoopInv (path : List Nat) (trip : Carry → Except String (List Carry × List AS × AOut)) (c0 : Carry)
    (o : AOut) : Prop :=
  ∃ c backs exits out j, c0.Le c ∧ c.vals.length = c0.vals.length ∧
    trip c = .ok (backs, exits, out) ∧ (∀ b ∈ backs, b.Le c) ∧ joinAll exits = .ok j ∧
    o = ⟨j, out.brk, out.cont, { out.certs with loops := (path, c) :: out.certs.loops }⟩

/-- **`settle` ends at an invariant**: a carry state covering the one it
    started from, from which one trip goes round only with states it covers. -/
theorem roomOf_le {S : TState} {w : World} (hS : S.holds w) (r : Region) : roomOf S r ≤ w.mem.sizes r := by
  unfold roomOf
  suffices h : ∀ (l : List Fact) (m : Nat), m ≤ w.mem.sizes r → (∀ f ∈ l, f.holds w) →
      l.foldl (fun m f => match f with | .room r' n => if r' = r then max m n else m | _ => m) m ≤ w.mem.sizes r from
    h S 0 (Nat.zero_le _) hS
  intro l
  induction l with
  | nil => intro m hm _; exact hm
  | cons f l ih =>
      intro m hm hl
      rw [List.foldl_cons]
      refine ih _ ?_ (fun g hg => hl g (List.mem_cons_of_mem _ hg))
      have hf := hl f (List.mem_cons_self ..)
      cases f with
      | room r' n =>
          dsimp only
          split
          · rename_i e; subst e; exact Nat.max_le.mpr ⟨hm, hf⟩
          · exact hm
      | _ => exact hm

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

-- ---------------------------------------------------------------------------
-- One trip of a loop
-- ---------------------------------------------------------------------------

theorem mem_brks0 {xs : List (Nat × AS × List AV)} {A : AS} {avs : List AV} {ab : Nat}
    (h : (0, A, avs) ∈ xs) : A.bindAt ab avs ∈ brks0 xs ab :=
  List.mem_filterMap.mpr ⟨_, h, by simp⟩

theorem mem_depthDown {xs : List (Nat × AS × List AV)} {d : Nat} {A : AS} {avs : List AV}
    (h : (d + 1, A, avs) ∈ xs) : (d, A, avs) ∈ depthDown xs :=
  List.mem_filterMap.mpr ⟨_, h, rfl⟩

theorem mem_conts0 {xs : List (Nat × List AV × TState)} {avs : List AV} {S : TState}
    (h : (0, avs, S) ∈ xs) : (⟨avs, S⟩ : Carry) ∈ conts0 xs :=
  List.mem_filterMap.mpr ⟨_, h, rfl⟩

theorem mem_depthDownC {xs : List (Nat × List AV × TState)} {d : Nat} {avs : List AV} {S : TState}
    (h : (d + 1, avs, S) ∈ xs) : (d, avs, S) ∈ depthDownC xs :=
  List.mem_filterMap.mpr ⟨_, h, rfl⟩

/-- What one trip collects: everything its prefix and body end with. -/
theorem loopTrip_spec {l : Loop} {n0 k preEnd ab : Nat} {op : AOut} {runBody : AS → Except String AOut}
    {backs : List Carry} {exits : List AS} {out : AOut}
    (h : loopTrip l n0 k preEnd ab op runBody = .ok (backs, exits, out)) :
    (∀ x ∈ conts0 op.cont, x ∈ backs) ∧ (∀ x ∈ brks0 op.brk ab, x ∈ exits) ∧
    (∀ x ∈ depthDown op.brk, x ∈ out.brk) ∧ (∀ x ∈ depthDownC op.cont, x ∈ out.cont) ∧
    ∀ Ap, op.ok = some Ap →
      (∀ Ax, refine Ap l.flag l.exitOnTrue = some Ax → Ax.bindAt ab (l.exitR.map Ax.get) ∈ exits) ∧
      ∀ Ag, refine Ap l.flag (!l.exitOnTrue) = some Ag →
        ∃ ob, runBody (Ag.bindAt preEnd ((List.range k).map fun i => Ag.get (n0 + i))) = .ok ob ∧
          (∀ x ∈ conts0 ob.cont, x ∈ backs) ∧ (∀ x ∈ brks0 ob.brk ab, x ∈ exits) ∧
          (∀ x ∈ depthDown ob.brk, x ∈ out.brk) ∧ (∀ x ∈ depthDownC ob.cont, x ∈ out.cont) ∧
          ∀ A2, ob.ok = some A2 → (⟨(l.cont.map A2.get).map AV.portable, A2.facts⟩ : Carry) ∈ backs := by
  unfold loopTrip at h
  dsimp only at h
  split at h
  · rename_i hn
    simp only [Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl, rfl⟩ := h
    exact ⟨fun _ h => h, fun _ h => h, fun _ h => h, fun _ h => h, fun Ap e => by rw [hn] at e; cases e⟩
  · rename_i Ap hAp
    split at h
    · rename_i hg
      simp only [Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl⟩ := h
      refine ⟨fun _ h => h, fun _ h => List.mem_append_left _ h, fun _ h => h, fun _ h => h,
        fun Ap' e => ?_⟩
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
          fun _ h => List.mem_append_left _ h, fun _ h => List.mem_append_left _ h, fun Ap' e => ?_⟩
        rw [hAp] at e; cases e
        refine ⟨fun Ax hx => List.mem_append_left _ (List.mem_append_right _ (by simp [hx])),
          fun Ag' hg' => ?_⟩
        rw [hg] at hg'; cases hg'
        exact ⟨ob, hob, fun _ h => List.mem_append_left _ (List.mem_append_right _ h),
          fun _ h => List.mem_append_right _ h, fun _ h => List.mem_append_right _ h,
          fun _ h => List.mem_append_right _ h,
          fun A2 e => List.mem_append_right _ (by simp [e])⟩

-- ---------------------------------------------------------------------------
-- What a statement keeps
-- ---------------------------------------------------------------------------

open AlgorithmLib.HProg.Contracts (KeepsRooms foldl_stores_sizes)

theorem store_world {cfg : Cfg} {Γ Γ' : Env} {w w' : World} {ty : ClifTy} {v a : R}
    (hr : runStmt cfg Γ w (.store ty v a) = .ok Γ' w') : Γ' = Γ ∧ w'.mem.sizes = w.mem.sizes := by
  rw [runStmt_store] at hr
  split at hr
  · dsimp only at hr
    split at hr
    · split at hr
      · rename_i m hm; cases hr; exact ⟨rfl, store_sizes_eq hm⟩
      · cases hr
    · split at hr
      · rename_i m hm
        cases hr
        rw [← Array.foldlM_toList] at hm
        exact ⟨rfl, (foldl_stores_sizes _ (fun _ _ _ h => ⟨store_sizes_eq h, store_frozen h⟩) _ _ _ hm).1⟩
      · cases hr
  · cases hr

theorem storeU_world {cfg : Cfg} {Γ Γ' : Env} {w w' : World} {v a : R}
    (hr : runStmt cfg Γ w (.storeUnaligned v a) = .ok Γ' w') : Γ' = Γ ∧ w'.mem.sizes = w.mem.sizes := by
  rw [runStmt_storeUnaligned] at hr
  split at hr
  · split at hr
    · rename_i m hm; cases hr; exact ⟨rfl, store_sizes_eq hm⟩
    · cases hr
  · dsimp only at hr
    split at hr
    · rename_i m hm
      cases hr
      rw [← Array.foldlM_toList] at hm
      exact ⟨rfl, (foldl_stores_sizes _ (fun _ _ _ h => ⟨store_sizes_eq h, store_frozen h⟩) _ _ _ hm).1⟩
    · cases hr
  · cases hr

theorem istore8_world {cfg : Cfg} {Γ Γ' : Env} {w w' : World} {v a : R}
    (hr : runStmt cfg Γ w (.istore8 v a) = .ok Γ' w') : Γ' = Γ ∧ w'.mem.sizes = w.mem.sizes := by
  rw [runStmt_istore8] at hr
  split at hr
  · split at hr
    · rename_i m hm; cases hr; exact ⟨rfl, store_sizes_eq hm⟩
    · cases hr
  · cases hr

/-- An accepted foreign call keeps every region's size but the pinned one's. -/
theorem ffi_rooms {cfg : Cfg} {A : AS} {f : Ffi} {Γ : Env} {w : World} {vs : List V} {r : Option V}
    {w' : World} (hA : A.Holds Γ w) (hf : f ≠ .threadSpawn)
    (hz : A.facts.contains (.part .frozen false) = true)
    (hc : callOf cfg.locals (.ffi f) vs (obsCall w (.ffi f) vs) = some (r, w')) : KeepsRooms w w' := by
  have hz' : (obsCall w (.ffi f) vs).mem.frozen = false := holds_of_mem hA.2.2 hz
  rw [Contracts.callOf_ffi hf _ _ hz'] at hc
  obtain ⟨bits, -, hc⟩ := Option.bind_eq_some_iff.mp hc
  exact fun rg hrg => callBits_rooms f hc rg hrg

theorem absStmt_step {cfg : Cfg} {A A' : AS} {s : Stmt} {fo : Bool} (h : absStmt fo A s = .ok A') {Γ : Env} {w : World}
    (hA : A.Holds Γ w) {Γ' : Env} {w' : World} (hr : runStmt cfg Γ w s = .ok Γ' w') :
    Ext Γ Γ' ∧ KeepsRooms w w' := by
  cases s with
  | op o =>
      simp only [runStmt] at hr
      split at hr
      · cases hr; exact ⟨Ext.push _ _, KeepsRooms.refl _⟩
      · cases hr
  | store ty v a =>
      obtain ⟨rfl, hs⟩ := store_world hr
      exact ⟨Ext.refl _, fun r _ => congrFun hs r⟩
  | storeUnaligned v a =>
      obtain ⟨rfl, hs⟩ := storeU_world hr
      exact ⟨Ext.refl _, fun r _ => congrFun hs r⟩
  | istore8 v a =>
      obtain ⟨rfl, hs⟩ := istore8_world hr
      exact ⟨Ext.refl _, fun r _ => congrFun hs r⟩
  | call c args =>
      cases c with
      | ffi f =>
          simp only [absStmt] at h
          split at h
          · rename_i hc
            simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at hc
            obtain ⟨⟨⟨hf, hz⟩, -⟩, -⟩ := hc
            rw [runStmt_call] at hr
            split at hr
            · cases hr
            · rename_i vs hvs
              split at hr
              · exact absurd hr failOf_ne_ok
              · rename_i res w2 hc
                split at hr
                · cases hr; exact ⟨Ext.push _ _, ffi_rooms hA hf hz hc⟩
                · cases hr
          · cases h
      | _ => simp [absStmt] at h
  | callVoid c args =>
      cases c with
      | ffi f =>
          simp only [absStmt] at h
          split at h
          · rename_i hc
            simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at hc
            obtain ⟨⟨⟨hf, hz⟩, -⟩, -⟩ := hc
            rw [runStmt_callVoid] at hr
            split at hr
            · cases hr
            · rename_i vs hvs
              split at hr
              · exact absurd hr failOf_ne_ok
              · rename_i res w2 hc
                cases hr; exact ⟨Ext.refl _, ffi_rooms hA hf hz hc⟩
          · cases h
      | _ => simp [absStmt] at h

/-- The state a run is in, as seen from where it started: the checker's state
    describes it, its environment extends the one it started in, and every
    region keeps its size. -/
def Within (A : AS) (Γ0 : Env) (w0 : World) (Γ : Env) (w : World) : Prop :=
  A.Holds Γ w ∧ Ext Γ0 Γ ∧ KeepsRooms w0 w

theorem absStmt_frame {cfg : Cfg} {A A' : AS} {s : Stmt} {fo : Bool} (h : absStmt fo A s = .ok A') {Γ0 : Env}
    {w0 : World} : Stmt1 cfg (Within A Γ0 w0) s (Within A' Γ0 w0) fo := by
  intro Γ w ⟨hA, he, hk⟩
  have h1 := absStmt_sound (cfg := cfg) h Γ w hA
  cases hr : runStmt cfg Γ w s with
  | ok Γ' w' =>
      rw [hr] at h1
      obtain ⟨e1, k1⟩ := absStmt_step h hA hr
      exact ⟨h1, he.trans e1, KeepsRooms.trans hk k1⟩
  | stuck _ => trivial
  | misuse _ => rw [hr] at h1; exact h1
  | fault _ => rw [hr] at h1; exact h1

theorem absStmtsH_frame {cfg : Cfg} {Γ0 : Env} {w0 : World} {fo : Bool} :
    ∀ {hist : List (Nat × Stmt)} {A A' : AS} {ss : List Stmt}, absStmtsH fo hist A ss = .ok A' →
      Stmts cfg (Within A Γ0 w0) ss (Within A' Γ0 w0) fo
  | _, A, A', [], h => by
      simp only [absStmtsH, Except.ok.injEq] at h
      subst h
      exact fun _ _ hP => hP
  | _, A, A', s :: ss, h => by
      simp only [absStmtsH] at h
      split at h
      · rename_i A1 h1
        exact stmts_cons (absStmt_frame h1) (absStmtsH_frame h)
      · cases h

-- ---------------------------------------------------------------------------
-- Regions
-- ---------------------------------------------------------------------------

/-- What a region the checker accepted may end with, from `Γ0` and `w0`: in a
    state its result describes, by a `br` from a state it recorded with the
    values it passes, or by a `cont` with carries and facts it recorded; with
    the environment extended and every region's size kept; and no fault but where `fo` allows one. -/
def AOut.post (o : AOut) (fo : Bool) (Γ0 : Env) (w0 : World) : Post where
  ok Γ w := (∃ A, o.ok = some A ∧ A.Holds Γ w) ∧ Ext Γ0 Γ ∧ KeepsRooms w0 w
  brk d Γ vs w := (∃ A avs, (d, A, avs) ∈ o.brk ∧ A.Holds Γ w ∧ DescAll Γ w avs vs) ∧
    Ext Γ0 Γ ∧ KeepsRooms w0 w
  cont d vs w := (∃ avs S, (d, avs, S) ∈ o.cont ∧ S.holds w ∧ DescAll #[] w avs vs) ∧ KeepsRooms w0 w
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

theorem desc_empty {Γ : Env} {w : World} {av : AV} {v : V} (hd : av.Desc #[] w v) : av.Desc Γ w v := by
  cases av with
  | cmp c a b => obtain ⟨_, _, _, ha, -⟩ := hd; simp at ha
  | _ => exact AV.desc_transport hd (fun _ h => by simp [AV.refs] at h) (fun _ _ => rfl)

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
theorem dloopTrip_spec {l : DLoop} {ab : Nat} {ob : AOut} {backs : List Carry} {exits : List AS}
    {out : AOut} (h : dloopTrip l ab ob = (backs, exits, out)) :
    (∀ x ∈ conts0 ob.cont, x ∈ backs) ∧ (∀ x ∈ brks0 ob.brk ab, x ∈ exits) ∧
    (∀ x ∈ depthDown ob.brk, x ∈ out.brk) ∧ (∀ x ∈ depthDownC ob.cont, x ∈ out.cont) ∧
    ∀ A2, ob.ok = some A2 →
      (∀ Ag, refine A2 l.flag l.contOnTrue = some Ag →
        (⟨(l.cont.map Ag.get).map AV.portable, Ag.facts⟩ : Carry) ∈ backs) ∧
      (∀ Ax, refine A2 l.flag (!l.contOnTrue) = some Ax →
        Ax.bindAt ab (l.exitIdx.map fun i => (l.cont.map Ax.get).getD i .any) ∈ exits) := by
  unfold dloopTrip at h
  simp only [Prod.mk.injEq] at h
  obtain ⟨rfl, rfl, rfl⟩ := h
  refine ⟨fun _ h => List.mem_append_left _ h, fun _ h => List.mem_append_left _ h, fun _ h => h,
    fun _ h => h, fun A2 e => ⟨fun Ag hg => List.mem_append_right _ (by simp [e, hg]),
      fun Ax hx => List.mem_append_right _ (by simp [e, hx])⟩⟩

theorem Slots.eq_of_beq : ∀ {a b : Slots}, a.beq b = true → a = b
  | .leaf, .leaf, _ => rfl
  | .node l x r, .node l' x' r', h => by
      simp only [Slots.beq, Bool.and_eq_true] at h
      rw [AV.eq_of_beq h.1.1, Slots.eq_of_beq h.1.2, Slots.eq_of_beq h.2]
  | .leaf, .node .., h => by simp [Slots.beq] at h
  | .node .., .leaf, h => by simp [Slots.beq] at h

theorem AS.eq_of_beq {A B : AS} (h : A.beq B = true) : A = B := by
  simp only [AS.beq, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
  obtain ⟨⟨h1, h2⟩, h3⟩ := h
  cases A; cases B
  simp only at h1 h2 h3
  rw [Slots.eq_of_beq h2, h1, h3]

/-- A state written out is the one reached. -/
theorem pin_eq {cert : Cert} {path : List Nat} {A A' : AS} (h : pin cert path A = .ok A') : A' = A := by
  unfold pin at h
  split at h
  · split at h
    · rename_i hb; cases h; exact (AS.eq_of_beq hb).symm
    · cases h
  · cases h; rfl

mutual

/-- **A piece the checker accepts** ends, from any state the abstract one
    describes, as its result says. -/
theorem execP_sound {cfg : Cfg} {cert : Cert} :
    ∀ {path : List Nat} {A : AS} {p : Piece} {o : AOut}, execP cert path A p = .ok o →
      ∀ {Γ0 : Env} {w0 : World}, A.Holds Γ0 w0 → PieceT cfg (At Γ0 w0) p (o.post cert.fo Γ0 w0)
  | _, A, .straight ss, o, h, Γ0, w0, hA => by
      simp only [execP] at h
      split at h
      · cases h
      · rename_i A' hss
        cases h
        exact straight_rule (stmts_conseq (absStmtsH_frame (Γ0 := Γ0) (w0 := w0) hss)
          (fun Γ w ⟨e1, e2⟩ => by subst Γ w; exact ⟨hA, Ext.refl _, KeepsRooms.refl _⟩)
          (fun Γ w ⟨h1, h2, h3⟩ => ⟨⟨A', rfl, h1⟩, h2, h3⟩))
  | _, A, .br d args, o, h, Γ0, w0, hA => by
      simp only [execP, Except.ok.injEq] at h
      subst h
      exact br_rule fun Γ w vs ⟨e1, e2⟩ hvs => by
        subst Γ w
        exact ⟨⟨A, args.map A.get, List.mem_singleton.mpr rfl, hA, descAll_of hA hvs⟩, Ext.refl _,
          KeepsRooms.refl _⟩
  | _, A, .cont d args, o, h, Γ0, w0, hA => by
      simp only [execP, Except.ok.injEq] at h
      subst h
      exact cont_rule fun Γ w vs ⟨e1, e2⟩ hvs => by
        subst Γ w
        exact ⟨⟨_, A.facts, List.mem_singleton.mpr rfl, hA.2.2, (descAll_of hA hvs).portable⟩,
          KeepsRooms.refl _⟩
  | _, A, .ite m thn els thnR elsR, o, h, Γ0, w0, hA => by
      have hn : Γ0.size = A.len := hA.1
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
              exact ⟨⟨C, hC, hCh⟩, he.bindAt (by omega) vs, hk⟩
            · intro d Γ' vs w' ⟨⟨A', avs, hm, h1, h2⟩, he, hk⟩
              exact ⟨⟨A', avs, List.mem_append_left _ hm, h1, h2⟩, he, hk⟩
            · intro d vs w' ⟨⟨avs, S, hm, h1, h2⟩, hk⟩
              exact ⟨⟨avs, S, List.mem_append_left _ hm, h1, h2⟩, hk⟩
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
              exact ⟨⟨C, hC, hCh⟩, (he0.trans he).bindAt (by omega) vs, hk⟩
            · intro d Γ' vs w' ⟨⟨A', avs, hm, h1, h2⟩, he, hk⟩
              exact ⟨⟨A', avs, List.mem_append_right _ hm, h1, h2⟩, he0.trans he, hk⟩
            · intro d vs w' ⟨⟨avs, S, hm, h1, h2⟩, hk⟩
              exact ⟨⟨avs, S, List.mem_append_right _ hm, h1, h2⟩, hk⟩
        · cases h
      · cases h
      · cases h
  | path, A, .loop l pre body, o, h, Γ0, w0, hA => by
      have hn : Γ0.size = A.len := hA.1
      simp only [execP] at h
      split at h
      · rename_i hk
        obtain ⟨c, backs, exits, out, j, hc0, hlen, ht, hback, hj, rfl⟩ := runLoop_sound h
        dsimp only at ht
        split at ht
        · cases ht
        · rename_i op hop
          obtain ⟨hcB, hbB, hdB, hdC, hsome⟩ := loopTrip_spec ht
          have hck : c.vals.length = l.pTys.length := by rw [hlen]; simp [hk]
          have hpre := slotsOf_ge (A.len + l.pTys.length) pre
          have hab := slotsOf_ge (slotsOf (A.len + l.pTys.length) pre + l.pTys.length) body
          refine loop_rule (fun _ cs w => c.Desc cs w ∧ KeepsRooms w0 w) ?_ ?_
          · intro Γ w cs ⟨e1, e2⟩ hcs
            subst Γ w
            exact ⟨hc0 cs w0 ⟨((descAll_of hA hcs).portable).enrich hA.2.2, hA.2.2⟩, KeepsRooms.refl _⟩
          · intro Γ w0' cs w ⟨e1, e2⟩ ⟨⟨hdc, hfc⟩, hkw⟩
            subst Γ w0'
            rw [hn]
            have hcl : cs.length = l.pTys.length := hdc.length.symm.trans hck
            have hA1 : AS.Holds ((A.withFacts c.facts).bindAt A.len c.vals)
                (bindAt Γ0 A.len cs) w :=
              (hA.world hfc fun r hr => hkw r hr).bindAt A.len hdc.of_empty
            have hΓb : Ext Γ0 (bindAt Γ0 A.len cs) := (Ext.refl Γ0).bindAt (by omega) cs
            refine Hoare.conseq (execC_sound hop hA1) (fun _ _ h => h) ?_ ?_ ?_
            · intro Γ1 w1 ⟨⟨Ap, hApE, hAp⟩, he1, hk1⟩ t f hf
              obtain ⟨hexit, hgo⟩ := hsome Ap hApE
              refine ⟨fun hx vs hvs => ?_, fun hx => ?_⟩
              · have hb : (f != 0) = l.exitOnTrue := by simpa using hx
                obtain ⟨Ax, hr, hAx⟩ := refine_sound hAp hf hb
                obtain ⟨C, hC, hCh⟩ := joinAll_holds hj (hexit Ax hr) (hAx.bindAt _ (descAll_of hAx hvs))
                exact ⟨⟨C, hC, hCh⟩, (hΓb.trans he1).bindAt (by omega) vs, KeepsRooms.trans hkw hk1⟩
              · have hb : (f != 0) = !l.exitOnTrue := by
                  revert hx; cases (f != 0) <;> cases l.exitOnTrue <;> simp
                obtain ⟨Ag, hr, hAg⟩ := refine_sound hAp hf hb
                obtain ⟨ob, hob, hbc, hbe, hbB', hbC', hfall⟩ := hgo Ag hr
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
                refine Hoare.conseq (execC_sound hob (hAg.bindAt _ hcs)) (fun _ _ h => h) ?_ ?_ ?_
                · intro Γ2 w2 ⟨⟨A2, hA2E, hA2⟩, he2, hk2⟩ next hnext
                  exact ⟨hback _ (hfall A2 hA2E) next w2 ⟨(descAll_of hA2 hnext).portable, hA2.2.2⟩,
                    KeepsRooms.trans hkw (KeepsRooms.trans hk1 hk2)⟩
                · intro d Γb vs w' ⟨⟨Ab, avs, hm, hAb, hdv⟩, he', hk'⟩
                  cases d with
                  | zero =>
                      obtain ⟨C, hC, hCh⟩ := joinAll_holds hj (hbe _ (mem_brks0 hm)) (hAb.bindAt _ hdv)
                      exact ⟨⟨C, hC, hCh⟩, (hΓb2.trans he').bindAt (by omega) vs,
                        KeepsRooms.trans hkw (KeepsRooms.trans hk1 hk')⟩
                  | succ d =>
                      exact ⟨⟨Ab, avs, hbB' _ (mem_depthDown hm), hAb, hdv⟩, hΓb2.trans he',
                        KeepsRooms.trans hkw (KeepsRooms.trans hk1 hk')⟩
                · intro d vs w' ⟨⟨avs, S, hm, hS, hdv⟩, hk'⟩
                  cases d with
                  | zero =>
                      exact ⟨hback _ (hbc _ (mem_conts0 hm)) vs w' ⟨hdv, hS⟩,
                        KeepsRooms.trans hkw (KeepsRooms.trans hk1 hk')⟩
                  | succ d =>
                      exact ⟨⟨avs, S, hbC' _ (mem_depthDownC hm), hS, hdv⟩,
                        KeepsRooms.trans hkw (KeepsRooms.trans hk1 hk')⟩
            · intro d Γb vs w' ⟨⟨Ab, avs, hm, hAb, hdv⟩, he', hk'⟩
              cases d with
              | zero =>
                  obtain ⟨C, hC, hCh⟩ := joinAll_holds hj (hbB _ (mem_brks0 hm)) (hAb.bindAt _ hdv)
                  exact ⟨⟨C, hC, hCh⟩, (hΓb.trans he').bindAt (by omega) vs, KeepsRooms.trans hkw hk'⟩
              | succ d =>
                  exact ⟨⟨Ab, avs, hdB _ (mem_depthDown hm), hAb, hdv⟩, hΓb.trans he', KeepsRooms.trans hkw hk'⟩
            · intro d vs w' ⟨⟨avs, S, hm, hS, hdv⟩, hk'⟩
              cases d with
              | zero => exact ⟨hback _ (hcB _ (mem_conts0 hm)) vs w' ⟨hdv, hS⟩, KeepsRooms.trans hkw hk'⟩
              | succ d =>
                  exact ⟨⟨avs, S, hdC _ (mem_depthDownC hm), hS, hdv⟩, KeepsRooms.trans hkw hk'⟩
      · cases h
  | path, A, .dloop l body, o, h, Γ0, w0, hA => by
      have hn : Γ0.size = A.len := hA.1
      simp only [execP] at h
      split at h
      · rename_i hk
        have hab := slotsOf_ge (A.len + l.pTys.length) body
        -- the loop skipped: its exits bound from the carries it would be entered with
        have hskip : ∀ g t f cs, l.guard = some g → Γ0[g]? = some (.sc t f) →
            ((f != 0) == l.contOnTrue) = false → l.init.mapM (fun r => Γ0[r]?) = some cs →
            ∀ outs, l.exitIdx.mapM (fun i => cs[i]?) = some outs →
            ∃ C, dSkip A l (slotsOf (A.len + l.pTys.length) body) = some C ∧
              C.Holds (bindAt Γ0 (slotsOf (A.len + l.pTys.length) body) outs) w0 := by
          intro g t f cs hg hf hx hcs outs houts
          have hb : (f != 0) = !l.contOnTrue := by
            revert hx; cases (f != 0) <;> cases l.contOnTrue <;> simp
          obtain ⟨Ax, hr, hAx⟩ := refine_sound hA hf hb
          exact ⟨_, by simp [dSkip, hg, hr], hAx.bindAt _ (descAll_pick (descAll_of hAx hcs) houts)⟩
        have hΓs : ∀ outs, Ext Γ0 (bindAt Γ0 (slotsOf (A.len + l.pTys.length) body) outs) :=
          fun outs => (Ext.refl Γ0).bindAt (by omega) outs
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
                  exact ⟨⟨C, hC, hCh⟩, hΓs outs, KeepsRooms.refl _⟩
        | some Ae =>
            rw [he] at h
            dsimp only at h
            split at h
            · cases h
            · rename_i o1 hrun
              split at h
              · rename_i j hj
                cases h
                obtain ⟨c, backs, exits, out, j', hc0, hlen, ht, hback, hj', rfl⟩ := runLoop_sound hrun
                dsimp only at ht
                split at ht
                · cases ht
                · rename_i ob hob
                  simp only [Except.ok.injEq] at ht
                  obtain ⟨hcB, hbB, hdB, hdC, hsome⟩ := dloopTrip_spec ht
                  have hck : c.vals.length = l.pTys.length := by rw [hlen]; simp [hk]
                  -- where the loop is entered, the state entered with describes it
                  have hent : ∀ cs, l.init.mapM (fun r => Γ0[r]?) = some cs → Ae.Holds Γ0 w0 →
                      c.Desc cs w0 := fun cs hcs hAe =>
                    hc0 cs w0 ⟨((descAll_of hAe hcs).portable).enrich hAe.2.2, hAe.2.2⟩
                  have hexit : ∀ {Γ' : Env} {w' : World} {C : AS}, C ∈ exits → C.Holds Γ' w' →
                      Ext Γ0 Γ' → KeepsRooms w0 w' →
                      (∃ D, j = some D ∧ D.Holds Γ' w') ∧ Ext Γ0 Γ' ∧ KeepsRooms w0 w' :=
                    fun hm hC he hk => by
                      obtain ⟨C', hC', hC'h⟩ := joinAll_holds hj' hm hC
                      obtain ⟨D, hD, hDh⟩ := joinOpt_holds hj (.inl ⟨C', hC', hC'h⟩)
                      exact ⟨⟨D, hD, hDh⟩, he, hk⟩
                  refine dloop_rule (fun _ cs w => c.Desc cs w ∧ KeepsRooms w0 w ∧ Ae.Holds Γ0 w0) ?_ ?_
                  · intro Γ w cs ⟨e1, e2⟩ hcs
                    subst Γ w
                    cases hg : l.guard with
                    | none =>
                        have : Ae = A := by simpa [dEnter, hg] using he.symm
                        subst this
                        exact ⟨hent cs hcs hA, KeepsRooms.refl _, hA⟩
                    | some g =>
                        intro t f hf
                        refine ⟨fun hx => ?_, fun hx outs houts => ?_⟩
                        · have hb : (f != 0) = l.contOnTrue := by simpa using hx
                          obtain ⟨Ae', hr, hAe'⟩ := refine_sound hA hf hb
                          have : Ae' = Ae := by simp [dEnter, hg, hr] at he; exact he
                          subst this
                          exact ⟨hent cs hcs hAe', KeepsRooms.refl _, hAe'⟩
                        · obtain ⟨C, hC, hCh⟩ := hskip g t f cs hg hf hx hcs outs houts
                          rw [hn]
                          obtain ⟨D, hD, hDh⟩ := joinOpt_holds hj (.inr ⟨C, hC, hCh⟩)
                          exact ⟨⟨D, hD, hDh⟩, hΓs outs, KeepsRooms.refl _⟩
                  · intro Γ w0' cs w ⟨e1, e2⟩ ⟨⟨hdc, hfc⟩, hkw, hAe⟩
                    subst Γ w0'
                    rw [hn]
                    have hA1 : AS.Holds ((Ae.withFacts c.facts).bindAt A.len c.vals)
                        (bindAt Γ0 A.len cs) w :=
                      (hAe.world hfc fun r hr => hkw r hr).bindAt A.len hdc.of_empty
                    have hΓb : Ext Γ0 (bindAt Γ0 A.len cs) := (Ext.refl Γ0).bindAt (by omega) cs
                    refine Hoare.conseq (execC_sound hob hA1) (fun _ _ h => h) ?_ ?_ ?_
                    · intro Γ2 w2 ⟨⟨A2, hA2E, hA2⟩, he2, hk2⟩
                      obtain ⟨hgo, hstop⟩ := hsome A2 hA2E
                      refine ⟨fun next hnext t f hf hx => ?_, fun next hnext t f hf hx outs houts => ?_⟩
                      · have hb : (f != 0) = l.contOnTrue := by simpa using hx
                        obtain ⟨Ag, hr, hAg⟩ := refine_sound hA2 hf hb
                        exact ⟨hback _ (hgo Ag hr) next w2 ⟨(descAll_of hAg hnext).portable, hAg.2.2⟩,
                          KeepsRooms.trans hkw hk2, hAe⟩
                      · have hb : (f != 0) = !l.contOnTrue := by
                          revert hx; cases (f != 0) <;> cases l.contOnTrue <;> simp
                        obtain ⟨Ax, hr, hAx⟩ := refine_sound hA2 hf hb
                        exact hexit (hstop Ax hr) (hAx.bindAt _ (descAll_pick (descAll_of hAx hnext) houts))
                          ((hΓb.trans he2).bindAt (by omega) outs) (KeepsRooms.trans hkw hk2)
                    · intro d Γb vs w' ⟨⟨Ab, avs, hm, hAb, hdv⟩, he', hk'⟩
                      cases d with
                      | zero =>
                          exact hexit (hbB _ (mem_brks0 hm)) (hAb.bindAt _ hdv)
                            ((hΓb.trans he').bindAt (by omega) vs) (KeepsRooms.trans hkw hk')
                      | succ d =>
                          exact ⟨⟨Ab, avs, hdB _ (mem_depthDown hm), hAb, hdv⟩, hΓb.trans he',
                            KeepsRooms.trans hkw hk'⟩
                    · intro d vs w' ⟨⟨avs, S, hm, hS, hdv⟩, hk'⟩
                      cases d with
                      | zero => exact ⟨hback _ (hcB _ (mem_conts0 hm)) vs w' ⟨hdv, hS⟩, KeepsRooms.trans hkw hk', hAe⟩
                      | succ d =>
                          exact ⟨⟨avs, S, hdC _ (mem_depthDownC hm), hS, hdv⟩, KeepsRooms.trans hkw hk'⟩
              · cases h
      · cases h

/-- **A region the checker accepts** ends, from any state the abstract one
    describes, as its result says. -/
theorem execC_sound {cfg : Cfg} {cert : Cert} :
    ∀ {base : List Nat} {i : Nat} {A : AS} {c : Code} {o : AOut}, execC cert base i A c = .ok o →
      ∀ {Γ0 : Env} {w0 : World}, A.Holds Γ0 w0 → Triple cfg (At Γ0 w0) c (o.post cert.fo Γ0 w0)
  | _, _, A, [], o, h, Γ0, w0, hA => by
      simp only [execC, Except.ok.injEq] at h
      subst h
      exact nil_rule fun Γ w ⟨e1, e2⟩ => by
        subst Γ w; exact ⟨⟨A, rfl, hA⟩, Ext.refl _, KeepsRooms.refl _⟩
  | _, _, A0, p :: ps, o, h, Γ0, w0, hA => by
      simp only [execC] at h
      split at h
      · cases h
      · rename_i A hpin
        obtain rfl := pin_eq hpin
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
              refine cons_rule (R := Within A' Γ0 w0) (pieceT_conseq ih1 ?_ ?_ ?_ (fun h => h))
                (triple_of_at fun Γ1 w1 ⟨h1', he, hk⟩ => ?_)
              · intro Γ w ⟨⟨A1, e, hA1⟩, he, hk⟩
                rw [hA'] at e; cases e; exact ⟨hA1, he, hk⟩
              · intro d Γ vs w ⟨⟨Ab, avs, hm, h1, h2⟩, he, hk⟩
                exact ⟨⟨Ab, avs, List.mem_append_left _ hm, h1, h2⟩, he, hk⟩
              · intro d vs w ⟨⟨avs, S, hm, h1, h2⟩, hk⟩
                exact ⟨⟨avs, S, List.mem_append_left _ hm, h1, h2⟩, hk⟩
              · refine Hoare.conseq (execC_sound h2 h1') (fun _ _ h => h) ?_ ?_ ?_
                · intro Γ w ⟨hx, he', hk'⟩
                  exact ⟨hx, he.trans he', KeepsRooms.trans hk hk'⟩
                · intro d Γ vs w ⟨⟨Ab, avs, hm, h1, h2⟩, he', hk'⟩
                  exact ⟨⟨Ab, avs, List.mem_append_right _ hm, h1, h2⟩, he.trans he', KeepsRooms.trans hk hk'⟩
                · intro d vs w ⟨⟨avs, S, hm, h1, h2⟩, hk'⟩
                  exact ⟨⟨avs, S, List.mem_append_right _ hm, h1, h2⟩, KeepsRooms.trans hk hk'⟩

end

/-- **The checker is sound.** A body the checker accepts, run from arguments
    and a world its entry state describes, never misuses a call and, checked
    as one that may not fault, never reads or writes memory that is not there. -/
theorem execC_run {cfg : Cfg} {cert : Cert} {base : List Nat} {i : Nat} {A : AS} {c : Code} {o : AOut}
    (h : execC cert base i A c = .ok o) (hfo : cert.fo = false)
    {args : List V} {w : World} (hA : A.Holds args.toArray w) (m : String) :
    run cfg args w c ≠ .misuse m ∧ run cfg args w c ≠ .fault m :=
  run_sound (execC_sound h hA) hfo m

/-- A body the checker accepts, as one that may fault or not, never misuses a
    call. -/
theorem execC_safe {cfg : Cfg} {cert : Cert} {base : List Nat} {i : Nat} {A : AS} {c : Code} {o : AOut}
    (h : execC cert base i A c = .ok o)
    {args : List V} {w : World} (hA : A.Holds args.toArray w) (m : String) :
    run cfg args w c ≠ .misuse m :=
  run_safe (execC_sound h hA) m

end AlgorithmLib.Check
