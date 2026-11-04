module
public import AlgorithmLib.Check.Exec
public import AlgorithmLib.Check.Calls
public import AlgorithmLib.Host.Threads
@[expose] public section

/-!
# `Check.ThreadSound` — starting and joining a thread, as the checker reads them

`spawn_core` and `join_core`: where the checker's rule for a thread's start
or join (`AS.spawnRule`, `AS.joinRule`) gives a state, the call, made from any
world the state before describes, answers, and leaves a world the state after
describes. A start lends its span: the arena is no longer wholly the
program's, and nothing the typestate said of the span survives. A join hands
the span back as the thread left it, which the typestate knows nothing of.
-/

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Prog

namespace AlgorithmLib.Check

open AlgorithmLib.HProg.Contracts (Fact TState Held LiveOk cellOk cellOk_spec)

variable {vals : List UInt64}

theorem AS.Holds.obsCall {A : AS} {Γ : Env} {w : World} (h : A.Holds vals Γ w) (c : Callee) (vs : List V) :
    A.Holds vals Γ (obsCall w c vs) := by
  obtain ⟨⟨hs, hy, hp⟩, hd, hf⟩ := h
  refine ⟨⟨hs, fun i hi => Sym.fits_congr (hy i hi) (fun _ _ => rfl) rfl (fun _ _ => rfl) (Nat.le_refl _),
    hp.1, hp.2.1, AlgorithmLib.Prog.TState.holds_obsCall hp.2.2.1, hp.2.2.2⟩,
    fun i hi => ?_, AlgorithmLib.Prog.TState.holds_obsCall hf⟩
  obtain ⟨v, hv, hd⟩ := hd i hi
  exact ⟨v, hv, AV.desc_transport hd (fun _ _ => rfl) (fun _ _ => rfl) (fun k _ _ _ hk => by cases k <;> exact hk)
    (fun _ _ _ _ hk => hk.same rfl (Nat.le_refl _) rfl)⟩

theorem descAll_of {A : AS} {Γ : Env} {w : World} (hA : A.Holds vals Γ w) :
    ∀ {args : List R} {vs : List V}, args.mapM (fun r => Γ[r]?) = some vs → DescAll Γ w (args.map A.get) vs
  | [], vs, h => by
      simp only [List.mapM_nil, Option.pure_def, Option.some.injEq] at h
      subst h; trivial
  | r :: rs, vs, h => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h
      obtain ⟨v, hv, vs', hvs, rfl⟩ := h
      exact ⟨hA.descAt hv, descAll_of hA hvs⟩

-- ---------------------------------------------------------------------------
-- What the calls do
-- ---------------------------------------------------------------------------

theorem ffiThreadSpawn_ok {w : World} {x y b : UInt64} {off len : Nat} (hx : fnBase ≤ x)
    (hl : w.lend (x - fnBase).toNat y = some (off, len)) (hsz : off + len ≤ w.mem.arena.size)
    (hc : w.mem.lentClear off len = true) (hn : w.threads.size < 2 ^ 48) (hg : w.grants "thread" = true) :
    ffiThreadSpawn [x, y, b] w = some (some (.sc .i64 (threadHandle w.threads.size)),
      { w with threads := w.threads.push (some ⟨(x - fnBase).toNat, y, off, len⟩),
               mem := { w.mem with lent := (off, len) :: w.mem.lent } }) := by
  have hx' : ¬ x < fnBase := fun h => absurd hx (UInt64.not_le.mpr h)
  simp only [ffiThreadSpawn, hx', decide_false, Bool.false_eq_true, if_false, hl, hc, hg]
  rw [if_neg (by simp; omega)]
  simp

/-- A load whose bytes are clear of a span is not stopped by lending it. -/
theorem load_lend {m : Mem} {a : UInt64} {n : Nat} {v : UInt64} {o l : Nat} (h : m.load a n = some v)
    (hd : ∀ off, decodeAddr a = some (.arena, off) → off + n ≤ o ∨ o + l ≤ off) :
    ({ m with lent := (o, l) :: m.lent } : Mem).load a n = some v := by
  cases hdec : decodeAddr a with
  | none => simp [Mem.load, hdec] at h
  | some p =>
    obtain ⟨r, off⟩ := p
    obtain ⟨hs, hr, rfl⟩ := Contracts.load_some hdec h
    rw [Contracts.load_of (m := ({ m with lent := (o, l) :: m.lent } : Mem)) hdec (by cases r <;> exact hs)]
    · cases r <;> rfl
    · unfold Mem.readable Mem.lentClear at hr ⊢
      simp only [Bool.and_eq_true, Bool.or_eq_true, List.all_cons] at hr ⊢
      refine ⟨by cases r <;> exact hr.1, ?_⟩
      by_cases ha : r = .arena
      · subst ha
        rcases hr.2 with h0 | h0
        · exact .inl h0
        · refine .inr ?_
          simp only [Bool.or_eq_true, decide_eq_true_eq]
          exact ⟨hd off hdec, h0⟩
      · exact .inl (by cases r <;> simp_all)

/-- An arena address as a number: the arena's base and its offset. -/
theorem arena_toNat {a : UInt64} {off : Nat} (h : decodeAddr a = some (.arena, off)) :
    a.toNat = (regionBase .arena).toNat + off := by
  obtain ⟨rfl, ho⟩ := Sem.decodeAddr_some h
  rw [Sem.regionSpan_toNat] at ho
  exact addr_toNat .arena (by unfold span; omega)

/-- **What a thread's start keeps**: every fact `spawnKeeps` keeps holds once
    the span is lent. -/
theorem spawn_keeps {w : World} {T : Array (Option Thread)} {off len : Nat} {x : Fact}
    (hk : spawnKeeps ((regionBase .arena).toNat + off) ((regionBase .arena).toNat + off + len) off len x = true)
    (hx : x.holds w) :
    x.holds { w with threads := T, mem := { w.mem with lent := (off, len) :: w.mem.lent } } := by
  cases x with
  | room r k =>
      simp only [spawnKeeps, bne_iff_ne, ne_eq] at hk
      exact ⟨by cases r <;> exact hx.1, fun h => absurd h hk⟩
  | owns o l =>
      simp only [spawnKeeps, Bool.or_eq_true, decide_eq_true_eq] at hk
      refine ⟨hx.1, hx.2.1, ?_⟩
      have h3 := hx.2.2
      unfold Mem.lentClear at h3 ⊢
      simp only [List.all_cons, Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq]
      exact ⟨by omega, by simpa using h3⟩
  | cell c v =>
      simp only [spawnKeeps, rangeKeeps, Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq] at hk
      refine load_lend hx fun co hd => ?_
      have := arena_toNat hd
      omega
  | threads _ _ _ _ _ | htAt _ _ _ | cstr _ | cstrIn _ _ | text _ _ _ _ | wgText _ _ _ =>
      simp [spawnKeeps] at hk
  | part p _ => cases p <;> exact hx
  | held k _ => cases k <;> exact hx
  | opened k _ => cases k <;> exact hx
  | roomArg r _ => exact ⟨by cases r <;> exact hx.1, hx.2⟩
  | wg1 _ o => cases o <;> exact hx
  | _ => exact hx

theorem AV.scTy_of_bits {a : AV} {x : UInt64} (h : a.bits = some x) : a.scTy.isSome = true := by
  unfold AV.bits at h
  split at h
  · simp [AV.scTy]
  · cases h

theorem threadsOf_mem {S : TState} {n sz : Nat} {live : List (Nat × Nat × UInt64 × Nat × Nat)}
    {lends : List (Nat × UInt64 × Nat × Nat)} {g : Bool} (h : threadsOf S = some (n, sz, live, lends, g)) :
    Fact.threads n sz live lends g ∈ S := by
  unfold threadsOf at h
  obtain ⟨x, hx, hk⟩ := List.exists_of_findSome?_eq_some h
  cases x <;> simp at hk
  obtain ⟨rfl, rfl, rfl, rfl, rfl⟩ := hk
  exact hx

/-- A span clear of every running thread's, by the list, is clear of every
    span lent. -/
theorem spanFree_clear {m : Mem} {live : List (Nat × Nat × UInt64 × Nat × Nat)} {off len : Nat}
    (hl : m.lent = live.map fun t => (t.2.2.2.1, t.2.2.2.2)) (h : spanFree live off len = true) :
    m.lentClear off len = true := by
  unfold Mem.lentClear spanFree at *
  rw [hl, List.all_map]
  simpa [Function.comp] using h

/-- **A thread's start the checker accepts** answers the handle its rule
    says, and leaves a world the state after describes, every region's size
    and the observations as they were once it was made. -/
theorem spawn_core {cfg : Cfg} {A B : AS} {args : List R} {av : AV} (h : A.spawnRule args = some (B, av))
    {Γ : Env} {w : World} {vs : List V} (hA : A.Holds vals Γ w) (hvs : args.mapM (fun r => Γ[r]?) = some vs) :
    ∃ r w', callOf cfg.locals (.ext (.base .threadSpawn)) vs (obsCall w (.ext (.base .threadSpawn)) vs) =
        some (r, w') ∧ B.Holds vals Γ w' ∧ (∀ v, r = some v → av.Desc (Γ.push v) w' v) ∧
      Contracts.KeepsRooms w w' ∧ w'.obs = (obsCall w (.ext (.base .threadSpawn)) vs).obs := by
  unfold AS.spawnRule at h
  split at h
  · rename_i f a st
    split at h
    · rename_i x y n sz live lends hfx hay hth
      split at h
      · rename_i hcond
        simp only [Bool.and_eq_true, decide_eq_true_eq] at hcond
        obtain ⟨hxb, hst⟩ := hcond
        split at h
        · rename_i i0 y0 off len hfind
          split at h
          · rename_i hc
            simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
            obtain ⟨⟨⟨hlen, hsz⟩, hfree⟩, hn⟩ := hc
            simp only [Option.some.injEq, Prod.mk.injEq] at h
            obtain ⟨rfl, rfl⟩ := h
            -- the world before, as the call sees it
            let c : Callee := .ext (.base .threadSpawn)
            have hA1 := hA.obsCall c vs
            have hS := hA1.2.2
            obtain ⟨t1, t2, t3, t4, t5, t6, t7, t8⟩ := hS _ (threadsOf_mem hth)
            have hl := List.find?_some hfind
            simp only [Bool.and_eq_true, beq_iff_eq] at hl
            obtain ⟨hli, hly⟩ := hl
            have hlend : (obsCall w c vs).lend (x - fnBase).toNat y = some (off, len) := by
              have := t5 _ (List.mem_of_find?_eq_some hfind)
              rw [hli, hly] at this; exact this
            -- the arguments' bits
            have hd := descAll_of hA1 hvs
            have hsc : ([f, a, st].map A.get).all (fun a => a.scTy.isSome) = true := by
              simp only [List.map_cons, List.map_nil, List.all_cons, List.all_nil, Bool.and_true, hst,
                Bool.and_eq_true]
              exact ⟨AV.scTy_of_bits hfx, AV.scTy_of_bits hay⟩
            obtain ⟨bits, hb⟩ := bits_exist hd hsc
            have hab := argBits_of hd hb
            simp only [List.map_cons, List.map_nil] at hab
            obtain ⟨x', bs, rfl, hx', hab⟩ := ArgBits.cons_inv hab
            obtain ⟨y', bs, rfl, hy', hab⟩ := ArgBits.cons_inv hab
            obtain ⟨z', bs, rfl, -, hab⟩ := ArgBits.cons_inv hab
            obtain rfl := ArgBits.nil_inv hab
            rw [known_bits hx' hfx, known_bits hy' hay] at hb
            have hcall := ffiThreadSpawn_ok (w := obsCall w c vs) (b := z') hxb hlend
              (by have : sz ≤ (obsCall w c vs).mem.arena.size := t2; omega) (spanFree_clear t7 hfree)
              (by rw [t1]; exact hn) (t4 rfl)
            let w' : World := { obsCall w c vs with
              threads := (obsCall w c vs).threads.push (some ⟨(x - fnBase).toNat, y, off, len⟩),
              mem := { (obsCall w c vs).mem with lent := (off, len) :: (obsCall w c vs).mem.lent } }
            refine ⟨some (.sc .i64 (threadHandle (obsCall w c vs).threads.size)), w', ?_, ?_, ?_,
              fun r _ => by cases r <;> rfl, rfl⟩
            · simp only [callOf, extCall, World.has, Ext.lib, Lib.builtin, Bool.true_or, Bool.not_true,
                Bool.false_eq_true, if_false, hb, Option.bind_eq_bind, Option.bind_some, baseCall]
              rw [hcall]
            · have hnew : (Fact.threads (n + 1) sz ((n, (x - fnBase).toNat, y, off, len) :: live) lends true).holds w' := by
                refine ⟨by simp only [w', Array.size_push]; rw [t1], t2, t3, fun _ => t4 rfl, t5, ?_, ?_, ?_⟩
                · intro t ht
                  rcases List.mem_cons.mp ht with rfl | ht
                  · simp only [w']; rw [← t1, Array.getElem?_push_size]
                  · have h6 := t6 t ht
                    have hlt : t.1 < (obsCall w c vs).threads.size := by
                      rcases Nat.lt_or_ge t.1 (obsCall w c vs).threads.size with h | h
                      · exact h
                      · rw [Array.getElem?_eq_none h] at h6; cases h6
                    simp only [w']; rw [Array.getElem?_push_lt hlt]
                    rw [Array.getElem?_eq_getElem hlt] at h6; exact h6
                · show (off, len) :: (obsCall w c vs).mem.lent = _
                  rw [t7]; rfl
                · refine ⟨fun t ht => ?_, List.Pairwise.cons (fun b hb => ⟨?_, ?_⟩) t8.2⟩
                  · rcases List.mem_cons.mp ht with rfl | ht
                    · exact hlen
                    · exact t8.1 t ht
                  · intro e
                    have h6 := t6 b hb
                    rw [← e, ← t1, Array.getElem?_eq_none (Nat.le_refl _)] at h6
                    cases h6
                  · have := List.all_eq_true.mp hfree b hb
                    simpa using this
              have hS' : TState.holds
                  (Fact.threads (n + 1) sz ((n, (x - fnBase).toNat, y, off, len) :: live) lends true ::
                    A.facts.filter (spawnKeeps ((regionBase .arena).toNat + off)
                      ((regionBase .arena).toNat + off + len) off len)) w' :=
                Contracts.TState.holds_cons hnew fun q hq => by
                  obtain ⟨hm, hk⟩ := List.mem_filter.mp hq
                  exact spawn_keeps hk (hS q hm)
              refine AS.Holds.withPar (hA1.world hS' (fun r _ => by cases r <;> rfl)
                (fun k _ hl => by cases k <;> exact hl) (fun _ _ hb => hb)) ?_
              exact hA1.1.2.2.keepWorld (fun _ _ => rfl) (fun _ _ => rfl) (fun q hk hq => spawn_keeps hk hq)
            · intro v hv
              cases hv
              rw [t1]; rfl
          · cases h
        · cases h
      · cases h
    · cases h
  · cases h

-- ---------------------------------------------------------------------------
-- Joining
-- ---------------------------------------------------------------------------

/-- A load clear of `[o, o + l)` reads the same once those bytes change and
    spans are handed back. -/
theorem load_join {m : Mem} {ar : ByteArray} {lt : List (Nat × Nat)} {a : UInt64} {n : Nat} {v : UInt64}
    {o l : Nat} (h : m.load a n = some v) (hsz : ar.size = m.arena.size)
    (hb : ∀ i, i < o ∨ o + l ≤ i → ar.get! i = m.arena.get! i) (hl : ∀ p ∈ lt, p ∈ m.lent)
    (hd : ∀ off, decodeAddr a = some (.arena, off) → off + n ≤ o ∨ o + l ≤ off) :
    ({ m with arena := ar, lent := lt } : Mem).load a n = some v := by
  cases hdec : decodeAddr a with
  | none => simp [Mem.load, hdec] at h
  | some p =>
    obtain ⟨r, off⟩ := p
    obtain ⟨hs, hr, rfl⟩ := Contracts.load_some hdec h
    have hrd : ({ m with arena := ar, lent := lt } : Mem).readable r off n = true := by
      unfold Mem.readable Mem.lentClear at hr ⊢
      simp only [Bool.and_eq_true, Bool.or_eq_true, List.all_eq_true] at hr ⊢
      refine ⟨by cases r <;> exact hr.1, ?_⟩
      rcases hr.2 with h0 | h0
      · exact .inl h0
      · exact .inr fun x hx => h0 x (hl x hx)
    cases r with
    | arena =>
        rw [Contracts.load_of (m := ({ m with arena := ar, lent := lt } : Mem)) hdec
          (by show off + n ≤ ar.size; rw [hsz]; exact hs) hrd]
        congr 1
        unfold Contracts.bytesAt
        exact Contracts.foldr_bytes_congr _ _ fun i hi => hb _ (by
          have := List.mem_range.mp hi; rcases hd off hdec with h1 | h1 <;> omega)
    | _ => rw [Contracts.load_of (m := ({ m with arena := ar, lent := lt } : Mem)) hdec hs hrd]; rfl

/-- **What a thread's join keeps**: every fact `joinKeeps` keeps holds once
    the span's bytes change and the span is handed back. -/
theorem join_keeps {w : World} {T : Array (Option Thread)} {ar : ByteArray} {lt : List (Nat × Nat)}
    {off len : Nat} {x : Fact}
    (hk : joinKeeps ((regionBase .arena).toNat + off) ((regionBase .arena).toNat + off + len) x = true)
    (hsz : ar.size = w.mem.arena.size) (hb : ∀ i, i < off ∨ off + len ≤ i → ar.get! i = w.mem.arena.get! i)
    (hl : ∀ p ∈ lt, p ∈ w.mem.lent) (hx : x.holds w) :
    x.holds { w with threads := T, mem := { w.mem with arena := ar, lent := lt } } := by
  cases x with
  | room r k =>
      refine ⟨by cases r <;> first | exact hx.1 | (show k ≤ ar.size; rw [hsz]; exact hx.1), fun h => ?_⟩
      have h0 := hx.2 h
      show lt = []
      exact List.eq_nil_iff_forall_not_mem.mpr fun p hp => by have := hl p hp; rw [h0] at this; cases this
  | owns o l =>
      refine ⟨by show _ ≤ ar.size; rw [hsz]; exact hx.1, hx.2.1, ?_⟩
      have h3 := hx.2.2
      unfold Mem.lentClear at h3 ⊢
      simp only [List.all_eq_true] at h3 ⊢
      exact fun p hp => h3 p (hl p hp)
  | cell c v =>
      simp only [joinKeeps, rangeKeeps, Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq] at hk
      refine load_join hx hsz hb hl fun co hd => ?_
      have := arena_toNat hd
      omega
  | roomArg r _ => exact ⟨by cases r <;> first | exact hx.1 | exact absurd rfl hx.2.2, hx.2.1, hx.2.2⟩
  | threads _ _ _ _ _ | htAt _ _ _ | cstr _ | cstrIn _ _ | text _ _ _ _ | wgText _ _ _ =>
      simp [joinKeeps] at hk
  | pinnedUsed _ => exact hx
  | part p _ => cases p <;> exact hx
  | held k _ => cases k <;> exact hx
  | opened k _ => cases k <;> exact hx
  | wg1 _ o => cases o <;> exact hx
  | _ => exact hx

/-- Handing back a running thread's span leaves the spans of the others. -/
theorem erase_span : ∀ {live : List (Nat × Nat × UInt64 × Nat × Nat)} {e : Nat × Nat × UInt64 × Nat × Nat}
    {k : Nat}, LiveOk live → e ∈ live → e.1 = k →
    (live.map fun t => (t.2.2.2.1, t.2.2.2.2)).erase (e.2.2.2.1, e.2.2.2.2) =
      (live.filter (·.1 != k)).map fun t => (t.2.2.2.1, t.2.2.2.2)
  | [], _, _, _, he, _ => by cases he
  | a :: rest, e, k, hok, he, hk => by
      obtain ⟨hpa, hpr⟩ := List.pairwise_cons.mp hok.2
      have hok' : LiveOk rest := ⟨fun t ht => hok.1 t (List.mem_cons_of_mem _ ht), hpr⟩
      by_cases hea : e = a
      · subst hea
        have hrest : rest.filter (·.1 != k) = rest := List.filter_eq_self.mpr fun t ht => by
          have := (hpa t ht).1
          simp only [bne_iff_ne, ne_eq]; omega
        simp [hk, hrest]
      · have he' : e ∈ rest := (List.mem_cons.mp he).resolve_left hea
        have hne : a.1 ≠ k := fun h => (hpa e he').1 (h.trans hk.symm)
        have hsp : ¬((a.2.2.2.1, a.2.2.2.2) == (e.2.2.2.1, e.2.2.2.2)) = true := by
          simp only [beq_iff_eq, Prod.mk.injEq]
          intro ⟨h1, h2⟩
          have := (hpa e he').2; have := hok.1 a List.mem_cons_self; omega
        rw [List.map_cons, List.erase_cons_tail hsp, erase_span hok' he' hk]
        simp [hne]

/-- **A thread's join the checker accepts** answers, and leaves a world the
    state after describes, every region's size and the observations as they
    were once it was made. -/
theorem join_core {cfg : Cfg} {A B : AS} {args : List R} {av : AV} (h : A.joinRule args = some (B, av))
    {Γ : Env} {w : World} {vs : List V} (hA : A.Holds vals Γ w) (hvs : args.mapM (fun r => Γ[r]?) = some vs) :
    ∃ r w', callOf cfg.locals (.ext (.base .threadJoin)) vs (obsCall w (.ext (.base .threadJoin)) vs) =
        some (r, w') ∧ B.Holds vals Γ w' ∧ (∀ v, r = some v → av.Desc (Γ.push v) w' v) ∧
      Contracts.KeepsRooms w w' ∧ w'.obs = (obsCall w (.ext (.base .threadJoin)) vs).obs := by
  unfold AS.joinRule at h
  split at h
  · rename_i hr
    split at h
    · rename_i x n sz live lends g hx hth
      split at h
      · rename_i k hk
        split at h
        · rename_i i0 f0 a0 off len hfind
          split at h
          · rename_i hsz
            simp only [Option.some.injEq, Prod.mk.injEq] at h
            obtain ⟨rfl, rfl⟩ := h
            let c : Callee := .ext (.base .threadJoin)
            let W := obsCall w c vs
            have hA1 := hA.obsCall c vs
            have hS := hA1.2.2
            obtain ⟨t1, t2, t3, t4, t5, t6, t7, t8⟩ := hS _ (threadsOf_mem hth)
            have he := List.mem_of_find?_eq_some hfind
            have hek : i0 = k := by simpa using List.find?_some hfind
            subst hek
            have h6 : W.threads[i0]? = some (some ⟨f0, a0, off, len⟩) := t6 _ he
            have hle : off + len ≤ W.mem.arena.size := by have : sz ≤ W.mem.arena.size := t2; omega
            -- the argument's bits
            have hd := descAll_of hA1 hvs
            have hsc : ([hr].map A.get).all (fun a => a.scTy.isSome) = true := by
              simp only [List.map_cons, List.map_nil, List.all_cons, List.all_nil, Bool.and_true]
              exact AV.scTy_of_bits hx
            obtain ⟨bits, hb⟩ := bits_exist hd hsc
            have hab := argBits_of hd hb
            simp only [List.map_cons, List.map_nil] at hab
            obtain ⟨x', bs, rfl, hx', hab⟩ := ArgBits.cons_inv hab
            obtain rfl := ArgBits.nil_inv hab
            rw [known_bits hx' hx] at hb
            obtain ⟨after, res, hout⟩ : ∃ a r, W.threadOut ⟨f0, a0, off, len⟩
                (W.mem.arena.extract off (off + len)) = (a, r) := ⟨_, _, rfl⟩
            let ar := (fitBytes len after).copySlice 0 W.mem.arena off len
            let w' : World := { W with
              threads := W.threads.set! i0 none,
              mem := { W.mem with arena := ar, lent := W.mem.lent.erase (off, len) } }
            have hcall : ffiThreadJoin [x] W = some (some (.sc .i64 res), w') := by
              simp only [ffiThreadJoin, hk, Option.bind_eq_bind, Option.bind_some, h6, Option.join_some, hout]
              rfl
            have har : ar.size = W.mem.arena.size := copySlice_size (fitBytes_size _ _) hle
            have hbytes : ∀ i, i < off ∨ off + len ≤ i → ar.get! i = W.mem.arena.get! i :=
              fun _ hi => copySlice_get_out (fitBytes_size _ _) hle hi
            have hlt : ∀ p ∈ W.mem.lent.erase (off, len), p ∈ W.mem.lent := fun _ hp => List.mem_of_mem_erase hp
            have hrooms : ∀ r, r ≠ .pinned → w'.mem.sizes r = W.mem.sizes r := by
              intro r _; cases r <;> first | rfl | exact har
            have hlive : W.mem.lent.erase (off, len) =
                (live.filter (·.1 != i0)).map fun t => (t.2.2.2.1, t.2.2.2.2) := by
              rw [t7]; exact erase_span t8 he rfl
            refine ⟨some (.sc .i64 res), w', ?_, ?_, ?_, hrooms, rfl⟩
            · simp only [callOf, extCall, World.has, Ext.lib, Lib.builtin, Bool.true_or, Bool.not_true,
                Bool.false_eq_true, if_false, hb, Option.bind_eq_bind, Option.bind_some, baseCall]
              exact hcall
            · have hnew : (Fact.threads n sz (live.filter (·.1 != i0)) lends g).holds w' := by
                refine ⟨by simp only [w', Array.set!, Array.size_setIfInBounds]; exact t1,
                  by show sz ≤ ar.size; rw [har]; exact t2, t3, t4, t5, ?_, hlive, ?_⟩
                · intro t ht
                  have hne : t.1 ≠ i0 := by simpa using (List.mem_filter.mp ht).2
                  simp only [w', Array.set!]
                  rw [Array.getElem?_setIfInBounds_ne (Ne.symm hne)]
                  exact t6 t (List.mem_filter.mp ht).1
                · exact ⟨fun t ht => t8.1 t (List.mem_filter.mp ht).1, t8.2.filter _⟩
              have hS' : TState.holds
                  (if (live.filter (·.1 != i0)).isEmpty then
                    .room .arena sz :: (Fact.threads n sz (live.filter (·.1 != i0)) lends g ::
                      A.facts.filter (joinKeeps ((regionBase .arena).toNat + off)
                        ((regionBase .arena).toNat + off + len)))
                  else Fact.threads n sz (live.filter (·.1 != i0)) lends g ::
                      A.facts.filter (joinKeeps ((regionBase .arena).toNat + off)
                        ((regionBase .arena).toNat + off + len))) w' := by
                have hrest := Contracts.TState.holds_cons hnew fun q hq => by
                  obtain ⟨hm, hk⟩ := List.mem_filter.mp hq
                  exact join_keeps hk har hbytes hlt (hS q hm)
                split
                · rename_i hem
                  refine Contracts.TState.holds_cons ⟨by show sz ≤ ar.size; rw [har]; exact t2, fun _ => ?_⟩ hrest
                  show W.mem.lent.erase (off, len) = []
                  rw [hlive, List.isEmpty_iff.mp hem]; rfl
                · exact hrest
              refine AS.Holds.withPar (hA1.world hS' hrooms
                (fun k _ hl => by cases k <;> exact hl) (fun _ _ hb => hb)) ?_
              exact hA1.1.2.2.keepWorld (fun _ _ => rfl) (fun _ _ => rfl)
                (fun q hk hq => join_keeps hk har hbytes hlt hq)
            · intro v hv
              cases hv
              exact TyV.sc (t := .i64) rfl
          · cases h
        · cases h
      · cases h
    · cases h
  · cases h

end AlgorithmLib.Check
