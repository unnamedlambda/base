module
public import AlgorithmLib.ML.Ptx.BodyPhase
meta import AlgorithmLib.ML.Ptx.BodyPhase
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `ML.Ptx.BodyEnds` — a warp statement's launch runs

A launch of a statement's kernel has a run (`LaunchRuns`, `launch_exists`):
each block's threads, run in order, answered as the laws give back, each end
(`block_exists`). Two runs of a lane whose answers agree below its `K`-th
shuffle make the same collectives up to it, and runs from memories that agree
on what the lane may touch compute alike (`runL_agree`); so, phase by phase, do
a block's threads run in order (`thread_agree`, `lcP_causal`), and the answers
the laws give back are a table built one shuffle at a time (`orc`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec AlgorithmLib.Kernel AlgorithmLib.ML AlgorithmLib.ML.BodyProof
open AlgorithmLib.ML.BodyWarp AlgorithmLib.ML.BodyPhase

namespace AlgorithmLib.ML.BodyEnds

/-! ## Two runs of a lane that agree -/

/-- Two lane states that agree up to the `K`-th shuffle: as many collectives,
    the same up to the `K`-th, memory of one shape; and, until the `K`-th
    shuffle is answered, the same registers and the same word at every
    element the lane may touch. -/
def Agr (nbuf : Nat) (A : Buf → Nat → Prop) (K : Nat) (st st' : LSt) : Prop :=
  st.calls.length = st'.calls.length ∧ MemSame st.mem st'.mem ∧
  (∀ j, j ≤ K → st.calls[j]? = st'.calls[j]?) ∧
  (st.calls.length ≤ K → st.reg = st'.reg ∧
    ∀ b j, A b j → st.mem.load (bufW nbuf b j) 4 = st'.mem.load (bufW nbuf b j) 4)

/-- A store of one value at one element keeps two memories agreeing on the
    elements `A` holds. -/
theorem agree_store {nbuf b j : Nat} {A : Buf → Nat → Prop}
    (hA : ∀ b j, A b j → b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) (hj : A b j)
    {m1 m2 m1' m2' : Mem} {v : UInt64} (hv : v.toNat < 2 ^ 32)
    (h1 : m1.store (bufW nbuf b j) 4 v = some m1') (h2 : m2.store (bufW nbuf b j) 4 v = some m2')
    (hag : ∀ b j, A b j → m1.load (bufW nbuf b j) 4 = m2.load (bufW nbuf b j) 4) :
    ∀ b' j', A b' j' → m1'.load (bufW nbuf b' j') 4 = m2'.load (bufW nbuf b' j') 4 := by
  intro b' j' hj'
  by_cases e : b' = b ∧ j' = j
  · obtain ⟨rfl, rfl⟩ := e
    rw [load_store4 hv h1, load_store4 hv h2]
  · rw [apart_elems (hA b j hj) (hA b' j' hj') e h1, apart_elems (hA b j hj) (hA b' j' hj') e h2]
    exact hag b' j' hj'

/-- **Two runs of a lane agree up to the shuffle their answers first differ
    at**: from states that agree, answered alike below `K`, each access one
    `A` allows, the runs end in states that agree. Before the `K`-th shuffle is
    answered they are one run; after it, they make their collectives at the
    same places, and memory keeps one shape. -/
theorem runL_agree (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (nbuf cta : Nat) (Ac : Acc)
    (rd : Buf → Nat → Option Nat) (hA : ∀ l b j, Ac l b j → b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32)
    (l : Lane) (o o' : Nat → UInt32) (K : Nat) (ho : ∀ j, j < K → o j = o' j) :
    ∀ (s : EWStmt) (i : Nat) (st st' r r' : LSt), AccOk cta Ac rd i s → Regs32 st.reg →
      Agr nbuf (Ac l) K st st' →
      runL hw nbuf cta l.val o rd i s st = some r → runL hw nbuf cta l.val o' rd i s st' = some r' →
      Agr nbuf (Ac l) K r r' := by
  have hAl := hA l
  have fold : ∀ (body : EWStmt) (L : List Nat),
      (∀ j, j ∈ L → ∀ (st st' r r' : LSt), AccOk cta Ac rd j body → Regs32 st.reg → Agr nbuf (Ac l) K st st' →
        runL hw nbuf cta l.val o rd j body st = some r → runL hw nbuf cta l.val o' rd j body st' = some r' →
        Agr nbuf (Ac l) K r r') →
      (∀ j, j ∈ L → AccOk cta Ac rd j body) →
      ∀ (sa sb ra rb : LSt), Regs32 sa.reg → Agr nbuf (Ac l) K sa sb →
        L.foldlM (fun st j => runL hw nbuf cta l.val o rd j body st) sa = some ra →
        L.foldlM (fun st j => runL hw nbuf cta l.val o' rd j body st) sb = some rb →
        Agr nbuf (Ac l) K ra rb := by
    intro body L
    induction L with
    | nil => intro _ _ sa sb ra rb _ hag ha hb; cases ha; cases hb; exact hag
    | cons j js ihj =>
      intro ih hacc sa sb ra rb h32 hag ha hb
      simp only [List.foldlM_cons, bind, Option.bind] at ha hb
      split at ha; · cases ha
      rename_i s1 hs1
      split at hb; · cases hb
      rename_i t1 ht1
      exact ihj (fun j' hj' => ih j' (List.mem_cons_of_mem _ hj')) (fun j' hj' => hacc j' (List.mem_cons_of_mem _ hj'))
        s1 t1 ra rb (runL_32 hw hw32 nbuf cta l.val _ _ body j sa s1 hs1 h32)
        (ih j List.mem_cons_self sa sb s1 t1 (hacc j List.mem_cons_self) h32 hag hs1 ht1) ha hb
  intro s
  induction s with
  | skip => intro i st st' r r' _ _ hag h h'; cases h; cases h'; exact hag
  | seq a b iha ihb =>
    intro i st st' r r' hacc h32 hag h h'
    simp only [runL, bind, Option.bind] at h h'
    split at h; · cases h
    rename_i sa hsa
    split at h'; · cases h'
    rename_i sb hsb
    exact ihb i sa sb r r' hacc.2 (runL_32 hw hw32 nbuf cta l.val _ _ a i st sa hsa h32)
      (iha i st st' sa sb hacc.1 h32 hag hsa hsb) h h'
  | setR n e =>
    intro i st st' r r' _ _ hag h h'
    cases h; cases h'
    obtain ⟨hlen, hms, hpre, hreg⟩ := hag
    refine ⟨hlen, hms, hpre, fun hk => ?_⟩
    obtain ⟨hr, hl⟩ := hreg hk
    exact ⟨by simp only [hr], hl⟩
  | shflXor d s m =>
    intro i st st' r r' _ _ hag h h'
    simp only [runL] at h h'
    split at h
    · rw [if_pos (by assumption)] at h'
      cases h; cases h'
      obtain ⟨hlen, hms, hpre, hreg⟩ := hag
      refine ⟨by simp [hlen], hms, fun j hj => ?_, fun hk => ?_⟩
      · by_cases hjn : j < st.calls.length
        · rw [List.getElem?_append_left hjn, List.getElem?_append_left (hlen ▸ hjn)]; exact hpre j hj
        · by_cases hje : j = st.calls.length
          · subst hje
            have hr := (hreg (by omega)).1
            rw [List.getElem?_append_right (Nat.le_refl _), hlen, List.getElem?_append_right (Nat.le_refl _),
              hr, ← hlen]
          · rw [List.getElem?_eq_none (by simp; omega), List.getElem?_eq_none (by simp; omega)]
      · simp only [List.length_append, List.length_cons, List.length_nil] at hk
        obtain ⟨hr, hl⟩ := hreg (by omega)
        refine ⟨?_, hl⟩
        simp only [hr, hlen, ho st'.calls.length (by omega)]
    · cases h
  | loadIdx dd b ix =>
    intro i st st' r r' hacc _ hag h h'
    simp only [runL, bind, Option.bind] at h h'
    split at h; · cases h
    rename_i j hj
    rw [hj] at h'
    dsimp only at h h'
    split at h; · cases h
    rename_i v hv
    split at h'; · cases h'
    rename_i v' hv'
    cases h; cases h'
    obtain ⟨hlen, hms, hpre, hreg⟩ := hag
    refine ⟨hlen, hms, hpre, fun hk => ?_⟩
    obtain ⟨hr, hl⟩ := hreg hk
    have hAj : Ac l b j := by rw [← idxL_eval rd cta i l ix j hj]; exact hacc l
    have : v = v' := by rw [hl b j hAj] at hv; rw [hv] at hv'; exact Option.some.inj hv'
    subst this
    exact ⟨by simp only [hr], hl⟩
  | loadV4 d0 d1 d2 d3 b ix =>
    intro i st st' r r' hacc _ hag h h'
    simp only [runL, bind, Option.bind] at h h'
    split at h; · cases h
    rename_i j hj
    rw [hj] at h'
    dsimp only at h h'
    obtain ⟨hlen, hms, hpre, hreg⟩ := hag
    have hj' : j = ix.eval cta i l (fun _ _ => 0) (imOf rd) := (idxL_eval rd cta i l ix j hj).symm
    have hq : ∀ (q : Nat) (k : Int), k = 4 * (q : Int) → q < 4 → st.calls.length ≤ K →
        st.mem.load (bufW nbuf b j + c64 k) 4 = st'.mem.load (bufW nbuf b j + c64 k) 4 := by
      intro q k hk hq4 hK
      have hA' := hacc l q hq4
      rw [← hj'] at hA'
      rw [bufW_add nbuf b j q k hk (hAl b _ hA').2.2]
      exact (hreg hK).2 b _ hA'
    split at h; · cases h
    rename_i v0 h0
    split at h'; · cases h'
    rename_i u0 hu0
    dsimp only at h h'
    split at h; · cases h
    rename_i v1 h1
    split at h'; · cases h'
    rename_i u1 hu1
    dsimp only at h h'
    split at h; · cases h
    rename_i v2 h2
    split at h'; · cases h'
    rename_i u2 hu2
    dsimp only at h h'
    split at h; · cases h
    rename_i v3 h3
    split at h'; · cases h'
    rename_i u3 hu3
    cases h; cases h'
    refine ⟨hlen, hms, hpre, fun hk => ?_⟩
    obtain ⟨hr, hl⟩ := hreg hk
    have e0 : v0 = u0 := by
      have hA0 := hacc l 0 (by decide); rw [← hj'] at hA0
      rw [show j + 0 = j by rfl] at hA0
      rw [hl b j hA0] at h0; rw [h0] at hu0; exact Option.some.inj hu0
    have e1 : v1 = u1 := by rw [hq 1 4 (by omega) (by decide) hk] at h1; rw [h1] at hu1; exact Option.some.inj hu1
    have e2 : v2 = u2 := by rw [hq 2 8 (by omega) (by decide) hk] at h2; rw [h2] at hu2; exact Option.some.inj hu2
    have e3 : v3 = u3 := by rw [hq 3 12 (by omega) (by decide) hk] at h3; rw [h3] at hu3; exact Option.some.inj hu3
    subst e0 e1 e2 e3
    exact ⟨by simp only [hr], hl⟩
  | storeLane b ix n =>
    intro i st st' r r' hacc h32 hag h h'
    obtain ⟨j, m1, hj, -, -, -, hm1, rfl⟩ := runL_store h
    obtain ⟨j', m2, hj2, -, -, -, hm2, rfl⟩ := runL_store h'
    rw [hj] at hj2; cases hj2
    obtain ⟨hlen, hms, hpre, hreg⟩ := hag
    refine ⟨hlen, (memSame_store hm1).symm.trans (hms.trans (memSame_store hm2)), hpre, fun hk => ?_⟩
    obtain ⟨hr, hl⟩ := hreg hk
    have hAj : Ac l b j := by rw [← idxL_eval rd cta i l ix j hj]; exact (hacc l).1
    rw [hr] at hm1
    exact ⟨hr, agree_store hAl hAj (by rw [← hr]; exact h32 n) hm1 hm2 hl⟩
  | storeLane0 b ix n =>
    intro i st st' r r' hacc h32 hag h h'
    rcases runL_store0 h with ⟨hl0, j, m1, hj, -, -, -, hm1, rfl⟩ | ⟨hl0, rfl⟩
    · rcases runL_store0 h' with ⟨-, j', m2, hj2, -, -, -, hm2, rfl⟩ | ⟨hl1, -⟩
      · rw [hj] at hj2; cases hj2
        obtain ⟨hlen, hms, hpre, hreg⟩ := hag
        refine ⟨hlen, (memSame_store hm1).symm.trans (hms.trans (memSame_store hm2)), hpre, fun hk => ?_⟩
        obtain ⟨hr, hl⟩ := hreg hk
        have hl00 : l = lane0 := Fin.ext hl0
        subst hl00
        have hAj : Ac lane0 b j := by rw [← idxL_eval rd cta i lane0 ix j hj]; exact hacc.1
        rw [hr] at hm1
        exact ⟨hr, agree_store hAl hAj (by rw [← hr]; exact h32 n) hm1 hm2 hl⟩
      · exact absurd hl0 hl1
    · rcases runL_store0 h' with ⟨hl1, -⟩ | ⟨-, rfl⟩
      · exact absurd hl1 hl0
      · exact hag
  | forN n body ih =>
    intro i st st' r r' hacc h32 hag h h'
    simp only [runL] at h h'
    split at h
    · rename_i hn
      rw [if_pos hn] at h'
      exact fold body _ (fun j _ => ih j) (fun j hj => hacc j (List.mem_range.mp hj)) st st' r r' h32 hag h h'
    · cases h
  | forM bb a body ih =>
    intro i st st' r r' hacc h32 hag h h'
    obtain ⟨ha, n, hn, hn31, hf⟩ := runL_forM h
    obtain ⟨-, n', hn', -, hf'⟩ := runL_forM h'
    rw [hn] at hn'; cases hn'
    exact fold body _ (fun j _ => ih j) (fun j hj => hacc j (by simp [imOf, hn]; exact List.mem_range.mp hj))
      st st' r r' h32 hag hf hf'
  | _ => intro i st st' r r' _ _ _ h; simp [runL] at h

/-! ## Answers the laws give back

For any family of runs `lc t O` (thread `t`'s collectives, every thread
answered by the oracles `O`) whose `k`-th collectives depend only on the
answers before the `k`-th, each a butterfly shuffle and as many for every
thread (`Lanes`), answers the laws give back exist: thread `t`'s `k`-th is its
partner's `k`-th passed word, in runs answered as the table has them up to `k`
(`tbl`, by recursion on `k`). -/

/-- The thread a butterfly by mask `m` pairs thread `t` with, in its warp. -/
def partner (t m : Nat) : Nat := 32 * (t / 32) + ((t % 32) ^^^ m) % 32

/-- The mask a shuffle passed, if the collective is one. -/
def maskOf : Option (Intr × List V) → Option Nat
  | some (_, [_, .sc _ m]) => some m.toNat
  | _ => none

/-- Thread `t`'s `k`-th answer, from runs answered by `O`: its partner's `k`-th
    passed word. -/
def word (lc : Nat → (Nat → Nat → UInt32) → List (Intr × List V)) (O : Nat → Nat → UInt32) (t k : Nat) : UInt32 :=
  match maskOf (lc t O)[k]? with
  | some m => match (lc (partner t m) O)[k]? with
      | some (_, args) => wordOf (lane32 args)
      | none => 0
  | none => 0

/-- Every thread's first `k` answers. -/
def tbl (lc : Nat → (Nat → Nat → UInt32) → List (Intr × List V)) : Nat → Nat → List UInt32
  | 0, _ => []
  | k + 1, t => tbl lc k t ++ [word lc (fun t' j => (tbl lc k t').getD j 0) t k]

/-- Thread `t`'s answers, the table's. -/
def orc (lc : Nat → (Nat → Nat → UInt32) → List (Intr × List V)) (t j : Nat) : UInt32 := (tbl lc (j + 1) t).getD j 0

theorem tbl_length (lc : Nat → (Nat → Nat → UInt32) → List (Intr × List V)) : ∀ k t, (tbl lc k t).length = k
  | 0, _ => rfl
  | k + 1, t => by simp [tbl, tbl_length lc k t]

theorem tbl_get (lc : Nat → (Nat → Nat → UInt32) → List (Intr × List V)) :
    ∀ k t j, j < k → (tbl lc k t).getD j 0 = orc lc t j
  | 0, _, _, h => absurd h (Nat.not_lt_zero _)
  | k + 1, t, j, h => by
    rcases Nat.lt_or_ge j k with hj | hj
    · rw [show tbl lc (k + 1) t = tbl lc k t ++ _ from rfl, List.getD_eq_getElem?_getD,
        List.getElem?_append_left (by rw [tbl_length]; exact hj), ← List.getD_eq_getElem?_getD]
      exact tbl_get lc k t j hj
    · have : j = k := by omega
      subst this
      rfl

/-- What the runs must be: the threads' `k`-th collectives depend only on the
    answers before the `k`-th; each is a butterfly shuffle, as many for every
    thread. -/
structure Lanes (N L : Nat) (lc : Nat → (Nat → Nat → UInt32) → List (Intr × List V)) : Prop where
  causal : ∀ (O O' : Nat → Nat → UInt32) (k : Nat), (∀ t j, j < k → O t j = O' t j) →
    ∀ t, t < N → ∀ j, j ≤ k → (lc t O)[j]? = (lc t O')[j]?
  shape : ∀ t, t < N → ∀ (O : Nat → Nat → UInt32) (e : Intr × List V), e ∈ lc t O →
    ∃ x m, e = (.shflBfly, [.sc .i32 x, .sc .i32 m])
  count : ∀ t, t < N → ∀ (O : Nat → Nat → UInt32), (lc t O).length = L
  closed : ∀ t m, t < N → partner t m < N

theorem orc_word {lc : Nat → (Nat → Nat → UInt32) → List (Intr × List V)} (t k : Nat) :
    orc lc t k = word lc (fun t' j => (tbl lc k t').getD j 0) t k := by
  simp only [orc, tbl, List.getD_eq_getElem?_getD]
  rw [List.getElem?_append_right (by simp [tbl_length])]
  simp [tbl_length]

/-- **Each thread is answered its partner's word**: where thread `t`'s `k`-th
    collective, run as the table answers it, is a shuffle by mask `m`, its
    partner's `k`-th is a shuffle too, and the table's `k`-th answer to `t` is the
    word it passed. -/
theorem orc_settled {N L : Nat} {lc : Nat → (Nat → Nat → UInt32) → List (Intr × List V)} (hL : Lanes N L lc)
    {t k : Nat} (ht : t < N) {x m : UInt64}
    (he : (lc t (orc lc))[k]? = some (.shflBfly, [.sc .i32 x, .sc .i32 m])) :
    ∃ args, (lc (partner t m.toNat) (orc lc))[k]? = some (.shflBfly, args) ∧
      orc lc t k = wordOf (lane32 args) := by
  let O : Nat → Nat → UInt32 := fun t' j => (tbl lc k t').getD j 0
  have hO : ∀ t' j, j < k → O t' j = orc lc t' j := fun t' j hj => tbl_get lc k t' j hj
  have hp := hL.closed t m.toNat ht
  have e1 : (lc t O)[k]? = (lc t (orc lc))[k]? := hL.causal O (orc lc) k hO t ht k (Nat.le_refl _)
  have e2 : (lc (partner t m.toNat) O)[k]? = (lc (partner t m.toNat) (orc lc))[k]? :=
    hL.causal O (orc lc) k hO _ hp k (Nat.le_refl _)
  have hk : k < L := by
    have := hL.count t ht (orc lc)
    have := (List.getElem?_eq_some_iff.mp he).1
    omega
  obtain ⟨c, args, hca⟩ : ∃ c args, (lc (partner t m.toNat) (orc lc))[k]? = some (c, args) := by
    have hlen := hL.count _ hp (orc lc)
    exact ⟨_, _, List.getElem?_eq_getElem (by omega)⟩
  obtain ⟨x', m', hsh⟩ := hL.shape _ hp _ _ (List.mem_of_getElem? hca)
  cases hsh
  refine ⟨_, hca, ?_⟩
  rw [orc_word]
  show (match maskOf (lc t O)[k]? with
    | some m => match (lc (partner t m) O)[k]? with
        | some (_, args) => wordOf (lane32 args)
        | none => 0
    | none => 0) = _
  rw [e1, he]
  simp only [maskOf]
  rw [e2, hca]

/-- Past its collectives, a thread's table answers are zero. -/
theorem orc_past {N L : Nat} {lc : Nat → (Nat → Nat → UInt32) → List (Intr × List V)} (hL : Lanes N L lc)
    {t k : Nat} (ht : t < N) (hk : L ≤ k) : orc lc t k = 0 := by
  rw [orc_word]
  simp only [word]
  have e1 := hL.causal (fun t' j => (tbl lc k t').getD j 0) (orc lc) k (fun t' j hj => tbl_get lc k t' j hj) t ht k
    (Nat.le_refl _)
  have hn : (lc t (orc lc))[k]? = none :=
    List.getElem?_eq_none (by rw [hL.count t ht]; exact hk)
  rw [e1, hn]
  rfl

/-- **The laws on butterfly shuffles**: where every collective of thread `t` is
    a shuffle and its partner made a shuffle at the same place, the laws answer
    `t` what `lawAt` says, one answer a collective. -/
theorem answersOf_shfl {calls : Nat → List (Intr × List V)} {snaps : Array ByteArray} {smem0 : ByteArray}
    {t : Nat} (lawAt : Nat → Answer)
    (h : ∀ k x m, (calls t)[k]? = some (.shflBfly, [.sc .i32 x, .sc .i32 m]) →
      ∃ args, (calls (partner t m.toNat))[k]? = some (.shflBfly, args) ∧ lawAt k = .val (lane32 args))
    (hsh : ∀ e ∈ calls t, ∃ x m, e = (.shflBfly, [.sc .i32 x, .sc .i32 m])) :
    answersOf calls snaps smem0 t = .ok ((List.range (calls t).length).map lawAt) := by
  unfold answersOf
  suffices hs : ∀ (l : List (Intr × List V)) (st : Nat) (out : Array Answer),
      (∀ k e, l[k]? = some e → (calls t)[st + k]? = some e) →
      (l.zipIdx st).foldlM (answerStep calls snaps smem0 (t / 32) (t % 32)) (0, out) =
        .ok (0, out ++ ((List.range l.length).map fun k => lawAt (st + k)).toArray) by
    have := hs (calls t) 0 #[] (fun k e he => by simpa using he)
    simp only [bind, Except.bind, this, pure, Except.pure]
    simp
  intro l
  induction l with
  | nil => intro st out _; simp [pure, Except.pure]
  | cons e es ih =>
    intro st out hl
    have he := hl 0 e rfl
    rw [Nat.add_zero] at he
    obtain ⟨x, m, rfl⟩ := hsh e (List.mem_of_getElem? he)
    obtain ⟨args, hp, hlaw⟩ := h st x m he
    have hstep : answerStep calls snaps smem0 (t / 32) (t % 32) (0, out)
        ((.shflBfly, [.sc .i32 x, .sc .i32 m]), st) = .ok (0, out.push (lawAt st)) := by
      simp only [answerStep, laneArg, bind, Except.bind, pure, Except.pure]
      have hp' : (calls (32 * (t / 32) + (t % 32 ^^^ m.toNat) % 32))[st]? = some (.shflBfly, args) := hp
      rw [hp', hlaw]
      rfl
    simp only [List.zipIdx_cons, List.foldlM_cons, bind, Except.bind]
    rw [hstep]
    dsimp only
    rw [ih (st + 1) (out.push (lawAt st)) (fun k e' he' => by
      have := hl (k + 1) e' he'; rwa [show st + (k + 1) = st + 1 + k by omega] at this)]
    congr 2
    simp only [List.length_cons, List.range_succ_eq_map, List.map_cons, List.map_map, Nat.add_zero]
    apply Array.ext'
    simp [Function.comp_def, Nat.add_assoc, Nat.add_comm 1]

/-! ## A statement's lanes -/

/-- What a shuffle's collective records: a butterfly of two 32-bit words. -/
def IsShfl (e : Intr × List V) : Prop := ∃ x m, e = (.shflBfly, [.sc .i32 x, .sc .i32 m])

/-- **A lane's collectives are butterfly shuffles.** -/
theorem runL_shape (hw : Hw) (nbuf cta l : Nat) (o : Nat → UInt32) (rd : Buf → Nat → Option Nat) :
    ∀ (s : EWStmt) (i : Nat) (st st' : LSt), runL hw nbuf cta l o rd i s st = some st' →
      (∀ e ∈ st.calls, IsShfl e) → ∀ e ∈ st'.calls, IsShfl e := by
  intro s
  induction s with
  | skip => intro i st st' h hs; cases h; exact hs
  | seq a b iha ihb =>
    intro i st st' h hs
    simp only [runL, bind, Option.bind] at h
    split at h; · cases h
    rename_i sa hsa
    exact ihb i sa st' h (iha i st sa hsa hs)
  | setR n e => intro i st st' h hs; cases h; exact hs
  | shflXor d s m =>
    intro i st st' h hs
    simp only [runL] at h
    split at h
    · cases h
      intro e he
      simp only [List.mem_append, List.mem_singleton] at he
      rcases he with he | rfl
      · exact hs e he
      · exact ⟨_, _, rfl⟩
    · cases h
  | loadIdx d b ix =>
    intro i st st' h hs
    simp only [runL, bind, Option.bind] at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    cases h; exact hs
  | loadV4 d0 d1 d2 d3 b ix =>
    intro i st st' h hs
    simp only [runL, bind, Option.bind] at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    cases h; exact hs
  | storeLane b ix n =>
    intro i st st' h hs
    obtain ⟨_, _, _, _, _, _, _, rfl⟩ := runL_store h
    exact hs
  | storeLane0 b ix n =>
    intro i st st' h hs
    rcases runL_store0 h with ⟨_, _, _, _, _, _, _, _, rfl⟩ | ⟨_, rfl⟩ <;> exact hs
  | forN n body ih =>
    intro i st st' h hs
    simp only [runL] at h
    split at h
    · exact foldlM_inv (fun st j => runL hw nbuf cta l o rd j body st) (fun st2 => ∀ e ∈ st2.calls, IsShfl e)
        (fun s1 j s2 h12 h1 => ih j s1 s2 h12 h1) _ _ _ h hs
    · cases h
  | forM b a body ih =>
    intro i st st' h hs
    obtain ⟨_, n, _, _, hf⟩ := runL_forM h
    exact foldlM_inv (fun st j => runL hw nbuf cta l o rd j body st) (fun st2 => ∀ e ∈ st2.calls, IsShfl e)
      (fun s1 j s2 h12 h1 => ih j s1 s2 h12 h1) _ _ _ hf hs
  | _ => intro i st st' h; simp [runL] at h

/-- Where a word is shuffled, it is the 32-bit word of what it passed. -/
theorem lane32_wordOf (a : List V) : lane32 a = .sc .i32 (s32 (wordOf (lane32 a))) := by
  unfold lane32
  split
  · simp only [wordOf, s32]
    congr 1
    apply UInt64.toNat_inj.mp
    simp [Nat.mod_mod]
  · rfl

/-- **A pass of threads that end**: where each thread, from memory its
    precondition allows, ends at some fuel leaving memory as `f` says, and no
    thread disturbs another's precondition, the pass finishes, the same way at
    every fuel past some. -/
theorem pass_exists {k : Kernel} {smem0 : ByteArray} {args : Nat → List V} {ans : Nat → List Answer} {hw : Hw}
    (P : Nat → Mem → Prop) (f : Nat → Mem → Mem)
    (hrun : ∀ t (sw : World), P t sw.mem → ∃ steps u,
      thread k.code (ans t) (args t) { sw with mem := { sw.mem with arena := smem0 } } hw steps = .ok u ∧
        u.mem = f t sw.mem)
    (hkeep : ∀ t u m, t ≠ u → P u m → P u (f t m)) :
    ∀ (ts : List Nat) (s : PassSt), ts.Nodup → (∀ t ∈ ts, P t s.1.mem) →
      ∃ S r, ∀ S', S ≤ S' → ts.foldlM (passStep (k.run hw S') smem0 args ans) s = .ok r
  | [], s, _, _ => ⟨0, s, fun _ _ => rfl⟩
  | t :: ts, (w1, c1, st1), hnd, hP => by
    obtain ⟨k0, u, hu, hm⟩ := hrun t w1 (hP t List.mem_cons_self)
    have hnt : t ∉ ts := (List.nodup_cons.mp hnd).1
    obtain ⟨S1, r, hr⟩ := pass_exists P f hrun hkeep ts
      (u, fun t' => if t' = t then collCalls u w1.obs.length else c1 t',
        fun t' => if t' = t then sharedStores u w1.obs.length else st1 t') (List.nodup_cons.mp hnd).2
      (fun t' ht' => by
        show P t' u.mem
        rw [hm]; exact hkeep t t' _ (fun e => hnt (e ▸ ht')) (hP t' (List.mem_cons_of_mem _ ht')))
    refine ⟨max k0 S1, r, fun S' hS' => ?_⟩
    simp only [List.foldlM_cons, passStep, Kernel.run, bind, Except.bind]
    rw [thread_mono hu (by omega)]
    exact hr S' (by omega)

/-! ## A block run in phases

A statement split into phases (`Split`) may have a lane read, in a later phase,
a word a lower lane of its warp stored in an earlier one (`hlate`). Run in
order, thread by thread, a thread's `k`-th collective still depends only on
the answers before the `k`-th, every thread's: each word it reads before then
was last written by a phase that ended before then (`Done`), and those words
two passes answered alike below `k` agree on (`Agrees`, `thread_agree`,
`lcP_causal`). -/

/-- The shuffles before phase `q`. -/
def before (rd : Buf → Nat → Option Nat) (ps : List EWStmt) (q : Nat) : Nat :=
  ((ps.take q).map (nsh (imOf rd))).sum

theorem before_succ (rd : Buf → Nat → Option Nat) (ps : List EWStmt) (q : Nat) (hq : q < ps.length) :
    before rd ps (q + 1) = before rd ps q + nsh (imOf rd) (ps.getD q .skip) := by
  simp only [before, List.take_succ, List.getElem?_eq_getElem hq, Option.toList_some, List.map_append,
    List.sum_append_nat, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero,
    List.getD_eq_getElem?_getD, Option.getD_some]

theorem before_mono (rd : Buf → Nat → Option Nat) (ps : List EWStmt) {p q : Nat} (h : p ≤ q) :
    before rd ps p ≤ before rd ps q := by
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le h
  simp only [before, List.take_add, List.map_append, List.sum_append_nat]
  omega

/-- Lane `t % 32` of thread `t`. -/
def lnOf (t : Nat) : Lane := ⟨t % 32, Nat.mod_lt _ (by decide)⟩

/-- Every writer of word `(b, j)` so far — a thread before `t`, or `t` in a
    phase before `q` — ended that phase before shuffle `k`. -/
def Done (cta nw N : Nat) (rd : Buf → Nat → Option Nat) (ps : List EWStmt) (k t q : Nat) (b : Buf) (j : Nat) :
    Prop :=
  ∀ o p, (o < t ∨ (o = t ∧ p < q)) → o < N → p < ps.length →
    StoresL (cta * nw + o / 32) (imOf rd) (lnOf o) 0 (ps.getD p .skip) b j → before rd ps (p + 1) ≤ k

/-- Two memories agree on every word whose writers so far are `Done`. -/
def Agrees (nbuf cta nw N : Nat) (rd : Buf → Nat → Option Nat) (ps : List EWStmt) (k t q : Nat) (m m' : Mem) :
    Prop :=
  ∀ b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) → Done cta nw N rd ps k t q b j →
    m.load (bufW nbuf b j) 4 = m'.load (bufW nbuf b j) 4

section Phases
variable (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (nbuf : Nat) {s : EWStmt}
  {ps : List EWStmt} (hsp : Split s ps) (nw cta N : Nat) (A : Nat → Nat → Acc) (rd : Buf → Nat → Option Nat)
  (hacc : ∀ t q, t < N → q < ps.length → AccOk (cta * nw + t / 32) (A q (cta * nw + t / 32)) rd 0 (ps.getD q .skip))
  (hA : ∀ q c l b j, A q c l b j → b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32)
  (hlate : ∀ t o q p b j, t < N → o < N → q < ps.length → p < ps.length →
    A q (cta * nw + t / 32) (lnOf t) b j →
    StoresL (cta * nw + o / 32) (imOf rd) (lnOf o) 0 (ps.getD p .skip) b j → t ≠ o → p < q ∧ o < t)
include hw32 hsp hacc hA hlate

/-- **One thread's two runs agree up to shuffle `k`**, answered alike below
    it, from memories that agree on every word whose writers are `Done`:
    the same collectives up to the `k`-th, memory of one shape, and memories
    that agree on every word whose writers, the thread now among them, are
    `Done`. -/
theorem thread_agree (k t : Nat) (ht : t < N) (o o' : Nat → UInt32) (ho : ∀ j, j < k → o j = o' j)
    (st0 st0' : LSt) (hc0 : st0.calls = []) (hc0' : st0'.calls = []) (hr0 : st0.reg = st0'.reg)
    (h32 : Regs32 st0.reg) (hms : MemSame st0.mem st0'.mem) (hrd : RdOk nbuf st0.mem rd)
    (hrd' : RdOk nbuf st0'.mem rd) (hag : Agrees nbuf cta nw N rd ps k t 0 st0.mem st0'.mem)
    (st st' : LSt) (hst : runL hw nbuf (cta * nw + t / 32) (t % 32) o rd 0 s st0 = some st)
    (hst' : runL hw nbuf (cta * nw + t / 32) (t % 32) o' rd 0 s st0' = some st') :
    st.calls.length = st'.calls.length ∧ (∀ j, j ≤ k → st.calls[j]? = st'.calls[j]?) ∧
      MemSame st.mem st'.mem ∧ Agrees nbuf cta nw N rd ps k (t + 1) 0 st.mem st'.mem := by
  let vc := cta * nw + t / 32
  let f (o : Nat → UInt32) : LSt → EWStmt → Option LSt := fun x p => runL hw nbuf vc (t % 32) o rd 0 p x
  rw [runL_split hw nbuf vc (t % 32) o rd 0 hsp] at hst
  rw [runL_split hw nbuf vc (t % 32) o' rd 0 hsp] at hst'
  let Inv (q : Nat) (x x' : LSt) : Prop :=
    x.calls.length = before rd ps q ∧ x'.calls.length = before rd ps q ∧
    (∀ j, j ≤ k → x.calls[j]? = x'.calls[j]?) ∧ MemSame x.mem x'.mem ∧
    (before rd ps q ≤ k → x.reg = x'.reg) ∧ Regs32 x.reg ∧ RdOk nbuf x.mem rd ∧ RdOk nbuf x'.mem rd ∧
    Agrees nbuf cta nw N rd ps k t q x.mem x'.mem
  have step : ∀ q (hq : q < ps.length) x x' x1 x1', Inv q x x' → f o x (ps[q]'hq) = some x1 →
      f o' x' (ps[q]'hq) = some x1' → Inv (q + 1) x1 x1' := by
    intro q hq x x' x1 x1' ⟨hl, hl', hpre, hm, hreg, h32x, hrdx, hrdx', hagx⟩ h1 h1'
    have hpq : ps.getD q .skip = ps[q] := by simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hq]
    have hAq := hA q vc (lnOf t)
    have hagr : Agr nbuf (A q vc (lnOf t)) k x x' := by
      refine ⟨hl.trans hl'.symm, hm, hpre, fun hk => ⟨hreg (hl ▸ hk), fun b j hAj => ?_⟩⟩
      refine hagx b j (hAq b j hAj) (fun o2 p ho2 hoN hp hS => ?_)
      rcases ho2 with ho2 | ⟨rfl, hpq'⟩
      · have := (hlate t o2 q p b j ht hoN hq hp hAj hS (by omega)).1
        exact Nat.le_trans (before_mono rd ps (by omega)) (hl ▸ hk)
      · exact Nat.le_trans (before_mono rd ps (by omega)) (hl ▸ hk)
    have hacc' := hacc t q ht hq
    rw [hpq] at hacc'
    have hag1 := runL_agree hw hw32 nbuf vc (A q vc) rd (hA q vc) (lnOf t) o o' k ho (ps[q]) 0 x x' x1 x1' hacc'
      h32x hagr h1 h1'
    obtain ⟨hlen1, hm1, hpre1, hreg1⟩ := hag1
    have hc1 := (runL_calls hw nbuf vc (t % 32) o rd (ps[q]) 0 x x1 h1).2
    have hc1' := (runL_calls hw nbuf vc (t % 32) o' rd (ps[q]) 0 x' x1' h1').2
    have hb := before_succ rd ps q hq
    rw [hpq] at hb
    refine ⟨by rw [hc1, hl, hb], by rw [hc1', hl', hb], hpre1, hm1, fun hk => (hreg1 (by omega)).1,
      runL_32 hw hw32 nbuf vc (t % 32) o rd (ps[q]) 0 x x1 h1 h32x, runL_rdOk (ps[q]) 0 x x1 h1 hrdx,
      runL_rdOk (ps[q]) 0 x' x1' h1' hrdx', fun b j hbd hD => ?_⟩
    by_cases hS : StoresL vc (imOf rd) (lnOf t) 0 (ps[q]) b j
    · have hk : before rd ps (q + 1) ≤ k := hD t q (.inr ⟨rfl, Nat.lt_succ_self q⟩) ht hq (by rw [hpq]; exact hS)
      exact (hreg1 (by omega)).2 b j (stores_acc vc (A q vc) rd (lnOf t) (ps[q]) 0 b j hacc' hS)
    · rw [runL_frame_st hw nbuf vc rd (lnOf t) o (ps[q]) 0 x x1 h1 hrdx b j hbd hS,
        runL_frame_st hw nbuf vc rd (lnOf t) o' (ps[q]) 0 x' x1' h1' hrdx' b j hbd hS]
      exact hagx b j hbd (fun o2 p ho2 => hD o2 p (by
        rcases ho2 with h | ⟨h1, h2⟩
        · exact .inl h
        · exact .inr ⟨h1, by omega⟩))
  have run : ∀ q, q ≤ ps.length → ∀ x x', (ps.take q).foldlM (f o) st0 = some x →
      (ps.take q).foldlM (f o') st0' = some x' → Inv q x x' := by
    intro q
    induction q with
    | zero =>
      intro _ x x' h h'
      simp only [List.take_zero, List.foldlM_nil, pure, Option.some.injEq] at h h'
      subst h h'
      refine ⟨by simp [hc0, before], by simp [hc0', before], fun j _ => by rw [hc0, hc0'], hms,
        fun _ => hr0, h32, hrd, hrd', hag⟩
    | succ q ih =>
      intro hq x x' h h'
      rw [foldlM_take_succ (f o) ps st0 q (by omega)] at h
      rw [foldlM_take_succ (f o') ps st0' q (by omega)] at h'
      cases hx : (ps.take q).foldlM (f o) st0 with
      | none => rw [hx] at h; cases h
      | some y =>
        cases hx' : (ps.take q).foldlM (f o') st0' with
        | none => rw [hx'] at h'; cases h'
        | some y' =>
          rw [hx] at h; rw [hx'] at h'
          exact step q (by omega) y y' x x' (ih (by omega) y y' hx hx') h h'
  have hfin := run ps.length (Nat.le_refl _) st st' (by rw [List.take_length]; exact hst)
    (by rw [List.take_length]; exact hst')
  obtain ⟨hl, hl', hpre, hm, -, -, -, -, hagP⟩ := hfin
  refine ⟨hl.trans hl'.symm, hpre, hm, fun b j hbd hD => hagP b j hbd (fun o2 p ho2 hoN hp hS => ?_)⟩
  exact hD o2 p (.inl (by rcases ho2 with h | ⟨h1, _⟩ <;> omega)) hoN hp hS

end Phases


/-- Thread `t`'s lane run from memory `m`, shared memory `A0`, answered by `o`. -/
def lrun (hw : Hw) (nbuf : Nat) (s : EWStmt) (nw cta : Nat) (rd : Buf → Nat → Option Nat) (A0 : ByteArray)
    (m : Mem) (t : Nat) (o : Nat → UInt32) : Option LSt :=
  runL hw nbuf (cta * nw + t / 32) (t % 32) o rd 0 s ⟨fun _ => ⟨0⟩, rm A0 m, []⟩

/-- The memory before thread `t` when a block's threads run in order from `m0`,
    thread `t'` answered by `O t'`. -/
def passMem (hw : Hw) (nbuf : Nat) (s : EWStmt) (nw cta : Nat) (rd : Buf → Nat → Option Nat) (A0 : ByteArray)
    (m0 : Mem) (O : Nat → Nat → UInt32) : Nat → Mem
  | 0 => m0
  | t + 1 => ((lrun hw nbuf s nw cta rd A0 (passMem hw nbuf s nw cta rd A0 m0 O t) t (O t)).map LSt.mem).getD
      (passMem hw nbuf s nw cta rd A0 m0 O t)

/-- Thread `t`'s collectives in that run. -/
def lcP (hw : Hw) (nbuf : Nat) (s : EWStmt) (nw cta : Nat) (rd : Buf → Nat → Option Nat) (A0 : ByteArray)
    (m0 : Mem) (t : Nat) (O : Nat → Nat → UInt32) : List (Intr × List V) :=
  ((lrun hw nbuf s nw cta rd A0 (passMem hw nbuf s nw cta rd A0 m0 O t) t (O t)).map LSt.calls).getD []

theorem agrees_rm {nbuf cta nw N : Nat} {rd : Buf → Nat → Option Nat} {ps : List EWStmt} {k t q : Nat}
    {m m' : Mem} (A0 : ByteArray) (h : Agrees nbuf cta nw N rd ps k t q m m') :
    Agrees nbuf cta nw N rd ps k t q (rm A0 m) (rm A0 m') := by
  intro b j hb hD
  rw [load_buf_arena m A0 (decode_bufW hb.1 hb.2.1 hb.2.2), load_buf_arena m' A0 (decode_bufW hb.1 hb.2.1 hb.2.2)]
  exact h b j hb hD

/-- **A block's threads, run in order, make their `k`-th collectives as the
    answers before the `k`-th say**: two runs answered alike below `k` make
    the same collectives up to it, every thread. -/
theorem lcP_causal (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (nbuf : Nat) {s : EWStmt}
    {ps : List EWStmt} (hsp : Split s ps) (nw cta N : Nat) (A : Nat → Nat → Acc) (rd : Buf → Nat → Option Nat)
    (hacc : ∀ t q, t < N → q < ps.length → AccOk (cta * nw + t / 32) (A q (cta * nw + t / 32)) rd 0 (ps.getD q .skip))
    (hA : ∀ q c l b j, A q c l b j → b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32)
    (hlate : ∀ t o q p b j, t < N → o < N → q < ps.length → p < ps.length →
      A q (cta * nw + t / 32) (lnOf t) b j →
      StoresL (cta * nw + o / 32) (imOf rd) (lnOf o) 0 (ps.getD p .skip) b j → t ≠ o → p < q ∧ o < t)
    (A0 : ByteArray) (m0 : Mem) (hrd0 : RdOk nbuf m0 rd)
    (hok : ∀ t, t < N → ∀ (o : Nat → UInt32) (m : Mem), MemSame (rm A0 m) (rm A0 m0) →
      ∃ st', lrun hw nbuf s nw cta rd A0 m t o = some st')
    (O O' : Nat → Nat → UInt32) (k : Nat) (hO : ∀ t j, j < k → O t j = O' t j) :
    ∀ t, t ≤ N → MemSame (rm A0 (passMem hw nbuf s nw cta rd A0 m0 O t)) (rm A0 m0) ∧
      MemSame (rm A0 (passMem hw nbuf s nw cta rd A0 m0 O' t)) (rm A0 m0) ∧
      RdOk nbuf (passMem hw nbuf s nw cta rd A0 m0 O t) rd ∧ RdOk nbuf (passMem hw nbuf s nw cta rd A0 m0 O' t) rd ∧
      Agrees nbuf cta nw N rd ps k t 0 (passMem hw nbuf s nw cta rd A0 m0 O t) (passMem hw nbuf s nw cta rd A0 m0 O' t) ∧
      ∀ t', t' < t → ∀ j, j ≤ k →
        (lcP hw nbuf s nw cta rd A0 m0 t' O)[j]? = (lcP hw nbuf s nw cta rd A0 m0 t' O')[j]? := by
  intro t
  induction t with
  | zero =>
    intro _
    exact ⟨MemSame.refl _, MemSame.refl _, hrd0, hrd0, fun _ _ _ _ => rfl, fun _ h => absurd h (Nat.not_lt_zero _)⟩
  | succ t ih =>
    intro ht
    obtain ⟨hm, hm', hr, hr', hag, hpre⟩ := ih (by omega)
    obtain ⟨st, hst⟩ := hok t (by omega) (O t) _ hm
    obtain ⟨st', hst'⟩ := hok t (by omega) (O' t) _ hm'
    have ht' := thread_agree hw hw32 nbuf hsp nw cta N A rd hacc hA hlate k t (by omega) (O t) (O' t)
      (fun j hj => hO t j hj) ⟨fun _ => ⟨0⟩, rm A0 (passMem hw nbuf s nw cta rd A0 m0 O t), []⟩
      ⟨fun _ => ⟨0⟩, rm A0 (passMem hw nbuf s nw cta rd A0 m0 O' t), []⟩ rfl rfl rfl
      (fun _ => by show (0 : UInt64).toNat < 2 ^ 32; decide) (hm.trans hm'.symm)
      ((rdOk_arena A0).mpr hr) ((rdOk_arena A0).mpr hr') (agrees_rm A0 hag) st st' hst hst'
    obtain ⟨-, hpk, hms, hag1⟩ := ht'
    have e1 : passMem hw nbuf s nw cta rd A0 m0 O (t + 1) = st.mem := by
      show ((lrun hw nbuf s nw cta rd A0 _ t (O t)).map LSt.mem).getD _ = _
      rw [hst]; rfl
    have e1' : passMem hw nbuf s nw cta rd A0 m0 O' (t + 1) = st'.mem := by
      show ((lrun hw nbuf s nw cta rd A0 _ t (O' t)).map LSt.mem).getD _ = _
      rw [hst']; rfl
    have hs1 := runL_same hw nbuf _ _ _ rd s 0 _ st hst
    have hs1' := runL_same hw nbuf _ _ _ rd s 0 _ st' hst'
    refine ⟨?_, ?_, ?_, ?_, ?_, fun t'' ht'' j hj => ?_⟩
    · rw [e1]; exact (memSame_rm hs1 A0).symm.trans (by simpa [rm] using hm)
    · rw [e1']; exact (memSame_rm hs1' A0).symm.trans (by simpa [rm] using hm')
    · rw [e1]; exact runL_rdOk s 0 _ _ hst ((rdOk_arena A0).mpr hr)
    · rw [e1']; exact runL_rdOk s 0 _ _ hst' ((rdOk_arena A0).mpr hr')
    · rw [e1, e1']; exact hag1
    · rcases Nat.lt_or_ge t'' t with h | h
      · exact hpre t'' h j hj
      · have : t'' = t := by omega
        subst this
        have e2 : lcP hw nbuf s nw cta rd A0 m0 t'' O = st.calls := by
          show ((lrun hw nbuf s nw cta rd A0 _ t'' (O t'')).map LSt.calls).getD [] = _
          rw [hst]; rfl
        have e2' : lcP hw nbuf s nw cta rd A0 m0 t'' O' = st'.calls := by
          show ((lrun hw nbuf s nw cta rd A0 _ t'' (O' t'')).map LSt.calls).getD [] = _
          rw [hst']; rfl
        rw [e2, e2']
        exact hpk j hj

/-- **A block's threads, each ending, run in order**: where thread `t`, from
    memory as `M t` (shared memory aside), ends at some fuel leaving `M (t+1)`
    and passing `C t`, the pass finishes, the same way at every fuel past some,
    its threads having passed `C`. -/
theorem pass_seq {k : Kernel} {smem0 : ByteArray} {args : Nat → List V} {ans : Nat → List Answer} {hw : Hw}
    (M : Nat → Mem) (C : Nat → List (Intr × List V)) :
    ∀ (n : Nat) (s0 : PassSt), rm smem0 s0.1.mem = rm smem0 (M 0) →
      (∀ t, t < n → ∀ sw : World, rm smem0 sw.mem = rm smem0 (M t) → ∃ steps u,
        thread k.code (ans t) (args t) { sw with mem := { sw.mem with arena := smem0 } } hw steps = .ok u ∧
          u.mem = M (t + 1) ∧ collCalls u sw.obs.length = C t) →
      ∃ S r, ∀ S', S ≤ S' → (List.range n).foldlM (passStep (k.run hw S') smem0 args ans) s0 = .ok r ∧
        rm smem0 r.1.mem = rm smem0 (M n) ∧ ∀ t, t < n → r.2.1 t = C t
  | 0, s0, h0, _ => ⟨0, s0, fun _ _ => ⟨rfl, h0, fun _ h => absurd h (Nat.not_lt_zero _)⟩⟩
  | n + 1, s0, h0, hrun => by
    obtain ⟨Sn, ⟨w1, c1, st1⟩, hn⟩ := pass_seq M C n s0 h0 (fun t ht => hrun t (by omega))
    obtain ⟨-, hm1, -⟩ := hn Sn (Nat.le_refl _)
    obtain ⟨kn, u, hu, hum, huc⟩ := hrun n (Nat.lt_succ_self n) w1 hm1
    refine ⟨max Sn kn, (u, fun t' => if t' = n then collCalls u w1.obs.length else c1 t',
      fun t' => if t' = n then sharedStores u w1.obs.length else st1 t'), fun S' hS' => ?_⟩
    obtain ⟨hf, -, hc⟩ := hn S' (by omega)
    refine ⟨?_, by rw [hum], fun t ht => ?_⟩
    · rw [List.range_succ, List.foldlM_append, hf]
      simp only [bind, Except.bind, List.foldlM_cons, List.foldlM_nil, passStep, Kernel.run]
      rw [thread_mono hu (by omega)]
      rfl
    · by_cases e : t = n
      · subst e; simp [huc]
      · simp only [e, if_false]; exact hc t (by omega)

/-- **A warp statement's block runs, in phases**: a block of `nw` warps of the
    statement's kernel, split into phases (`Split`), each phase's accesses
    ones its lanes may make there, a lane touching what another lane stores
    only in a later phase and from a higher lane (`hlate`), every element a
    lane may touch inside its buffer, has a run (`BlockRuns`): its threads,
    answered as the laws give back, each end. -/
theorem block_exists (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (name : String) (nbuf smem : Nat)
    (s : EWStmt) (ps : List EWStmt) (hsp : Split s ps)
    (nw : Nat) (hnw : 0 < nw) (hnw31 : nw < 2 ^ 31) (cta : Nat) (hcta : cta * nw + nw ≤ 2 ^ 32)
    (hN : 32 * nw ≤ 2 ^ 32) (A : Nat → Nat → Acc) (rd : Buf → Nat → Option Nat)
    (hacc : ∀ wi q, wi < nw → q < ps.length → AccOk (cta * nw + wi) (A q (cta * nw + wi)) rd 0 (ps.getD q .skip))
    (hA : ∀ q c l b j, A q c l b j → b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32)
    (hlate : ∀ t o q p b j, t < 32 * nw → o < 32 * nw → q < ps.length → p < ps.length →
      A q (cta * nw + t / 32) (lnOf t) b j →
      StoresL (cta * nw + o / 32) (imOf rd) (lnOf o) 0 (ps.getD p .skip) b j → t ≠ o → p < q ∧ o < t)
    (hfine : Fine (emitGo (fnBody (params nbuf) (Body.body nbuf s nw)) ⟨(params nbuf).length, 0, [], [], [], none⟩).2)
    (hRd : ∀ q wi l b j, wi < nw → A q (cta * nw + wi) l b j → rd b j = none)
    (hfits : ∀ wi, wi < nw → fits rd (cta * nw + wi) 0 s = true)
    (w : World) (hrd : RdOk nbuf w.mem rd) (hnan : w.mem.nan = hw.nan)
    (hin : ∀ q wi l b j, q < ps.length → wi < nw → A q (cta * nw + wi) l b j → 4 * j + 4 ≤ (w.mem.bufs b).size) :
    ∃ u, BlockRuns (Body.kernel name nbuf smem s nw) cta (32 * nw) (bufArgs nbuf) hw w u := by
  let N := 32 * nw
  let A0 := ByteArray.mk (Array.replicate smem 0)
  let lr := lrun hw nbuf s nw cta rd A0
  let lc := lcP hw nbuf s nw cta rd A0 w.mem
  let L := nsh (imOf rd) s
  have hvc : ∀ t, t < N → cta * nw + t / 32 < 2 ^ 32 := fun t ht => by
    have : t / 32 < nw := by omega
    omega
  have hacc' : ∀ t q, t < N → q < ps.length →
      AccOk (cta * nw + t / 32) (A q (cta * nw + t / 32)) rd 0 (ps.getD q .skip) :=
    fun t q ht hq => hacc (t / 32) q (by omega) hq
  have hok : ∀ t, t < N → ∀ (o : Nat → UInt32) (m : Mem), MemSame (rm A0 m) (rm A0 w.mem) →
      ∃ st', lr m t o = some st' := by
    intro t ht o m hs
    obtain ⟨st0, h0⟩ := runL_total_split hw nbuf (cta * nw + t / 32) (fun q => A q (cta * nw + t / 32)) rd (lnOf t) o
      hsp (fun q b j h => hA _ _ _ b j h) (fun q b j h => hRd q (t / 32) _ b j (by omega) h)
      (fun q hq => hacc' t q ht hq) (hfits (t / 32) (by omega)) ⟨fun _ => ⟨0⟩, rm A0 w.mem, []⟩
      (fun q b j hq h => hin q (t / 32) _ b j hq (by omega) h)
    exact runL_transfer hw nbuf (cta * nw + t / 32) (t % 32) o o rd s 0 _ st0 _ h0 hs.symm
  have hpass := lcP_causal hw hw32 nbuf hsp nw cta N A rd hacc' hA hlate A0 w.mem hrd hok
  have hpnan : ∀ O t, t ≤ N → (passMem hw nbuf s nw cta rd A0 w.mem O t).nan = hw.nan := by
    intro O t
    induction t with
    | zero => intro _; exact hnan
    | succ t ih =>
      intro ht
      show (((lr (passMem hw nbuf s nw cta rd A0 w.mem O t) t (O t)).map LSt.mem).getD
        (passMem hw nbuf s nw cta rd A0 w.mem O t)).nan = _
      cases hr : lr (passMem hw nbuf s nw cta rd A0 w.mem O t) t (O t) with
      | none => exact ih (by omega)
      | some r =>
        simp only [Option.map_some, Option.getD_some]
        exact (runL_nan s 0 _ _ hr ((rdOk_arena A0).mpr
          (hpass O O 0 (fun _ _ h => absurd h (Nat.not_lt_zero _)) t (by omega)).2.2.1)).trans (ih (by omega))
  have hrun_of : ∀ O t, t < N → ∃ r, lr (passMem hw nbuf s nw cta rd A0 w.mem O t) t (O t) = some r ∧
      lc t O = r.calls ∧ passMem hw nbuf s nw cta rd A0 w.mem O (t + 1) = r.mem := by
    intro O t ht
    have hm := (hpass O O 0 (fun _ _ h => absurd h (Nat.not_lt_zero _)) t (by omega)).1
    obtain ⟨r, hr⟩ := hok t ht (O t) _ hm
    refine ⟨r, hr, ?_, ?_⟩
    · show ((lr _ t (O t)).map LSt.calls).getD [] = _
      rw [hr]; rfl
    · show ((lr _ t (O t)).map LSt.mem).getD _ = _
      rw [hr]; rfl
  have hLanes : Lanes N L lc := by
    refine ⟨fun O O' k hk t ht j hj => (hpass O O' k hk (t + 1) (by omega)).2.2.2.2.2 t (Nat.lt_succ_self t) j hj,
      ?_, ?_, ?_⟩
    · intro t ht O e he
      obtain ⟨r, hr, hlr, -⟩ := hrun_of O t ht
      rw [hlr] at he
      exact runL_shape hw nbuf _ _ _ rd s 0 _ r hr (fun e h => by simp at h) e he
    · intro t ht O
      obtain ⟨r, hr, hlr, -⟩ := hrun_of O t ht
      rw [hlr, (runL_calls hw nbuf _ _ _ rd s 0 _ r hr).2]
      simp [L]
    · intro t m ht
      simp only [partner]
      have h1 : t / 32 < nw := by omega
      have h2 := Nat.mod_lt ((t % 32) ^^^ m) (show 0 < 32 by decide)
      show 32 * (t / 32) + ((t % 32) ^^^ m) % 32 < 32 * nw
      omega
  let O := orc lc
  let ans (t : Nat) : List Answer := (List.range L).map fun k => .val (.sc .i32 (s32 (O t k)))
  have hword : ∀ t, t < N → wordAt (ans t) = O t := by
    intro t ht
    funext k
    rcases Nat.lt_or_ge k L with hk | hk
    · simp [wordAt, ans, List.getElem?_map, List.getElem?_range hk, s32]
    · show wordAt (ans t) k = orc lc t k
      rw [orc_past hLanes ht hk]
      simp [wordAt, ans, List.getElem?_eq_none (show (List.map _ (List.range L)).length ≤ k by simp; omega)]
  have hWords : ∀ t, Words (ans t) := by
    intro t a ha v hv
    simp only [ans, List.mem_map, List.mem_range] at ha
    obtain ⟨k, -, rfl⟩ := ha
    cases hv
    exact ⟨_, rfl⟩
  -- the pass, as `passMem` runs it
  obtain ⟨S, r, hS⟩ := pass_seq (k := Body.kernel name nbuf smem s nw) (smem0 := A0)
    (args := threadArgs cta N (bufArgs nbuf)) (ans := ans) (hw := hw)
    (passMem hw nbuf s nw cta rd A0 w.mem O) (fun t => lc t O) N
    ({ w with mem := { w.mem with arena := A0 } }, fun _ => [], fun _ => []) rfl
    (fun t ht sw hsw => by
      obtain ⟨st', hst', hlc, hpm⟩ := hrun_of O t ht
      have hst'' : runL hw nbuf (cta * nw + t / 32) (t % 32) (wordAt (ans t)) rd 0 s
          ⟨fun _ => ⟨0⟩, ({ sw with mem := { sw.mem with arena := A0 } } : World).mem, []⟩ = some st' := by
        rw [hword t ht]
        show runL hw nbuf (cta * nw + t / 32) (t % 32) (O t) rd 0 s ⟨fun _ => ⟨0⟩, rm A0 sw.mem, []⟩ = some st'
        rw [hsw]; exact hst'
      have hrdsw : RdOk nbuf ({ sw with mem := { sw.mem with arena := A0 } } : World).mem rd := by
        show RdOk nbuf (rm A0 sw.mem) rd
        rw [hsw]
        exact (rdOk_arena A0).mpr (hpass O O 0 (fun _ _ h => absurd h (Nat.not_lt_zero _)) t (by omega)).2.2.1
      obtain ⟨k, u, hu, hm, hc⟩ := thread_ends hw hw32 name nbuf smem s nw hnw hnw31 cta t N ht (by omega)
        (hvc t ht) { sw with mem := { sw.mem with arena := A0 } } (ans t) (hWords t) st' hfine rd hrdsw
        (by show (rm A0 sw.mem).nan = _; rw [hsw]; exact hpnan O t (by omega)) hst''
      exact ⟨k, u, hu, by rw [hm, hpm], by show collCalls u sw.obs.length = lc t O; rw [hlc]; exact hc⟩)
  obtain ⟨hfold, -, hcalls0⟩ := hS S (Nat.le_refl _)
  obtain ⟨w', calls, stores⟩ := r
  have hcalls : ∀ t, t < N → calls t = lc t O := hcalls0
  -- the laws give the answers back
  have hans_t : ∀ t, t < N → answersOf calls (snapshots N A0 calls stores) A0 t = .ok (ans t) := by
    intro t ht
    have hct := hcalls t ht
    rw [answersOf_shfl (fun k => .val (.sc .i32 (s32 (O t k)))) ?_ ?_]
    · rw [hct, hLanes.count t ht O]
    · intro k x m hk
      rw [hct] at hk
      obtain ⟨args, hp, hwd⟩ := orc_settled hLanes ht hk
      refine ⟨args, ?_, ?_⟩
      · rw [hcalls _ (hLanes.closed t _ ht)]; exact hp
      · show Answer.val (.sc .i32 (s32 (orc lc t k))) = _
        rw [hwd, ← lane32_wordOf]
    · intro e he
      rw [hct] at he
      exact hLanes.shape t ht _ e he
  have hans : Kernel.answers N A0 calls stores = .ok fun t => ((List.range N).map ans).getD t [] := by
    unfold Kernel.answers
    rw [mapM_range_ok _ ans N hans_t]
    rfl
  refine ⟨{ w' with mem := { w'.mem with arena := w.mem.arena } }, S, ans, w', calls, stores, _, hfold, hans,
    fun t ht => ?_, rfl⟩
  simp only [List.getD_eq_getElem?_getD, List.getElem?_map]
  rw [List.getElem?_range (show t < N from ht)]
  rfl

/-- **A warp statement's launch runs, in phases** (`LaunchRuns`): `grid`
    blocks of `nw` warps of the statement's kernel, split into phases, as
    `BodyPhase.launch_phases` asks of them — each phase's accesses ones its
    lanes may make there, a stored word one lane's in one phase, a lane
    touching what another stores only later and from a higher lane, no block
    storing what another touches, every touched word inside its buffer. -/
theorem launch_exists (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (name : String) (nbuf smem : Nat)
    (s : EWStmt) (ps : List EWStmt) (hsp : Split s ps)
    (nw : Nat) (hnw : 0 < nw) (hnw31 : nw < 2 ^ 31) (grid : Nat) (hgrid : grid * nw ≤ 2 ^ 32)
    (hN : 32 * nw ≤ 2 ^ 32) (A : Nat → Nat → Acc) (rd : Buf → Nat → Option Nat)
    (hacc : ∀ c q, c < grid * nw → q < ps.length → AccOk c (A q c) rd 0 (ps.getD q .skip))
    (hA : ∀ q c l b j, A q c l b j → b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32)
    (hown : ∀ c (l l' : Lane) q q' b j, c < grid * nw → q < ps.length → q' < ps.length →
      StoresL c (imOf rd) l 0 (ps.getD q .skip) b j → StoresL c (imOf rd) l' 0 (ps.getD q' .skip) b j →
      l = l' ∧ q = q')
    (hrdr : ∀ c (t o : Lane) q p b j, c < grid * nw → q < ps.length → p < ps.length → A q c t b j →
      StoresL c (imOf rd) o 0 (ps.getD p .skip) b j → t ≠ o → p < q ∧ o.val < t.val)
    (hdisj : ∀ c c' l l' q b j, c < grid * nw → c' < grid * nw → c ≠ c' → q < ps.length → A q c l b j →
      ¬ StoresL c' (imOf rd) l' 0 s b j)
    (hfine : Fine (emitGo (fnBody (params nbuf) (Body.body nbuf s nw)) ⟨(params nbuf).length, 0, [], [], [], none⟩).2)
    (bufs : Nat → ByteArray) (hrd : RdOk nbuf (kmem bufs) rd)
    (hRd : ∀ q c l b j, c < grid * nw → A q c l b j → rd b j = none)
    (hfits : ∀ c, c < grid * nw → fits rd c 0 s = true)
    (hin : ∀ q c l b j, c < grid * nw → q < ps.length → A q c l b j → 4 * j + 4 ≤ (bufs b).size) :
    ∃ bufs', LaunchRuns (Body.kernel name nbuf smem s nw) grid (32 * nw) (bufArgs nbuf) bufs bufs' hw := by
  have hblk : ∀ bi, bi < grid → bi * nw + nw ≤ grid * nw := fun bi hb => by
    rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hb
  have hsto : ∀ c l b j q, q < ps.length → StoresL c (imOf rd) l 0 (ps.getD q .skip) b j →
      StoresL c (imOf rd) l 0 s b j := fun c l b j q hq h =>
    (storesL_split c (imOf rd) l 0 hsp b j).mpr ⟨ps[q], List.getElem_mem hq, by
      have hpq : ps.getD q .skip = ps[q] := by simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hq]
      rw [← hpq]; exact h⟩
  have inv : ∀ c, c ≤ grid → ∃ Wc, BlocksRun (Body.kernel name nbuf smem s nw) (32 * nw) (bufArgs nbuf) hw c
      ({ mem := { arena := .empty, data := .empty, out := .empty, bufs, nan := hw.nan } } : World) Wc ∧
      MemSame (kmem bufs) Wc.mem ∧ RdOk nbuf Wc.mem rd ∧ Wc.mem.nan = hw.nan := by
    intro c
    induction c with
    | zero =>
      intro _
      have hld : ∀ a n, ({ arena := .empty, data := .empty, out := .empty, bufs, nan := hw.nan } : Mem).load a n =
          (kmem bufs).load a n := Mem.load_congr (fun r => by cases r <;> rfl) rfl rfl
      refine ⟨_, rfl, ⟨fun r => by cases r <;> rfl, rfl, rfl⟩, fun b j n h => ?_, rfl⟩
      obtain ⟨h1, h2⟩ := hrd b j n h
      exact ⟨h1, by simp only [rdM] at h2 ⊢; rw [hld]; exact h2⟩
    | succ c ih =>
      intro hc
      obtain ⟨Wp, hp, hs, hr, hn⟩ := ih (by omega)
      have hcg : c < grid := by omega
      have hb := hblk c hcg
      have hvt : ∀ t, t < 32 * nw → c * nw + t / 32 < grid * nw := fun t ht => by
        have : t / 32 < nw := by omega
        omega
      have hsz : ∀ q wi l b j, q < ps.length → wi < nw → A q (c * nw + wi) l b j →
          4 * j + 4 ≤ (Wp.mem.bufs b).size := by
        intro q wi l b j hq hwi hAl
        have := hs.size (.buf ⟨b, (hA _ _ l b j hAl).2.1⟩)
        simp only [Mem.region] at this
        rw [← this]
        exact hin q _ l b j (by omega) hq hAl
      have hlate : ∀ t o' q p b j, t < 32 * nw → o' < 32 * nw → q < ps.length → p < ps.length →
          A q (c * nw + t / 32) (lnOf t) b j →
          StoresL (c * nw + o' / 32) (imOf rd) (lnOf o') 0 (ps.getD p .skip) b j → t ≠ o' →
          p < q ∧ o' < t ∧ o' / 32 = t / 32 := by
        intro t o' q p b j ht ho hq hp hAt hs' hto
        by_cases hw : o' / 32 = t / 32
        · rw [hw] at hs'
          have hne : lnOf t ≠ lnOf o' := fun e => hto (by
            have := congrArg Fin.val e; simp only [lnOf] at this; omega)
          obtain ⟨hpq, hlt⟩ := hrdr _ (lnOf t) (lnOf o') q p b j (hvt t ht) hq hp hAt hs' hne
          simp only [lnOf] at hlt
          exact ⟨hpq, by omega, hw⟩
        · exfalso
          exact hdisj (c * nw + t / 32) (c * nw + o' / 32) (lnOf t) (lnOf o') q b j (hvt t ht) (hvt o' ho)
            (by omega) hq hAt (hsto _ _ b j p hp hs')
      obtain ⟨Wc, hWc⟩ := block_exists hw hw32 name nbuf smem s ps hsp nw hnw hnw31 c (by omega) hN A rd
        (fun wi q hwi hq => hacc _ q (by omega) hq) hA
        (fun t o' q p b j ht ho hq hp hAt hs' hto => ⟨(hlate t o' q p b j ht ho hq hp hAt hs' hto).1,
          (hlate t o' q p b j ht ho hq hp hAt hs' hto).2.1⟩)
        hfine (fun q wi l b j hwi h => hRd q _ l b j (by omega) h) (fun wi hwi => hfits _ (by omega)) Wp hr hn
        (fun q wi l b j hq hwi h => hsz q wi l b j hq hwi h)
      obtain ⟨-, -, h3, h4, h5⟩ := block_phases hw hw32 name nbuf smem s ps hsp nw hnw hnw31 c (by omega) hN A rd
        (fun wi q hwi hq => hacc _ q (by omega) hq) hA
        (fun t t' q q' b j ht ht' hq hq' hs1 hs2 => by
          by_cases hw : t / 32 = t' / 32
          · rw [hw] at hs1
            obtain ⟨hl, hqq⟩ := hown _ _ _ q q' b j (hvt t' ht') hq hq' hs1 hs2
            have := congrArg Fin.val hl
            simp only at this
            exact ⟨by omega, hqq⟩
          · exfalso
            exact hdisj (c * nw + t / 32) (c * nw + t' / 32) (lnOf t) (lnOf t') q b j (hvt t ht) (hvt t' ht')
              (by omega) hq (stores_acc _ _ rd _ _ 0 b j (hacc _ q (hvt t ht) hq) hs1) (hsto _ _ b j q' hq' hs2))
        hlate hfine Wp Wc hr hn
        (fun t ht => runL_total_split hw nbuf (c * nw + t / 32) (fun q => A q (c * nw + t / 32)) rd (lnOf t) _ hsp
          (fun q b j h => hA q _ _ b j h) (fun q b j h => hRd q _ _ b j (hvt t ht) h)
          (fun q hq => hacc _ q (hvt t ht) hq) (hfits _ (hvt t ht)) _
          (fun q b j hq h => hsz q (t / 32) _ b j hq (by omega) h))
        Wp.mem
        (fun q wi l b j hq hwi hAl => ⟨rfl, (bufW_ok (hA _ _ l b j hAl) (hsz q wi l b j hq hwi hAl) 0).1⟩)
        hWc
      exact ⟨Wc, ⟨Wp, hp, hWc⟩, hs.trans h3, h4, h5⟩
  obtain ⟨W, hW, -, -, -⟩ := inv grid (Nat.le_refl _)
  exact ⟨W.mem.bufs, W, hW, rfl⟩

end AlgorithmLib.ML.BodyEnds
