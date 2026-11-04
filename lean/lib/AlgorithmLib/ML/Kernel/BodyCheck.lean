module
public import AlgorithmLib.ML.Ptx.BodyPhase
meta import AlgorithmLib.ML.Ptx.BodyPhase
public import Std.Data.HashMap
meta import Std.Data.HashMap
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `ML.Kernel.BodyCheck` — a static kernel's launch obligations, decided

A statement whose indexes read no memory touches, at each block, words that
running it lists (`touchL`, the stores among them `storeL`), phase by phase
(`phTouches`, `phStores`; a fused group's phases are its ops, `spine`).
Where each buffer's indexes grow by a stride from one block to the next
(`strideOk`), block `c`'s words are block 0's moved `c` strides along
(`touchL_shift`), so `staticOk` decides what `launch_phases` asks from block 0
alone: each word it stores has one owner lane and phase, its other lanes
touching it only in later phases and from higher lanes (`ownB`), the last
block's indexes fit (and so every earlier one's, `fits_mono`), and per buffer
the last block's words lie inside it and no two blocks' stored and touched
ranges meet (`bufOk`). `launch_static` is `launch_phases` with the touched
words as the access maps and `staticOk` in place of its premises.

`staticOk` reads no lane expression, so launches whose statements are alike
erased (`erase`, compared by `eqE`) are decided once: `keysOk` checks a list
of distinct keys and that every launch's key is among them (`keysOk_mem`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec AlgorithmLib.Kernel AlgorithmLib.ML AlgorithmLib.ML.BodyProof
open AlgorithmLib.ML.BodyWarp AlgorithmLib.ML.BodyPhase

namespace AlgorithmLib.ML.BodyCheck

section
variable (cta : Nat) (l : Lane)

/-- The words lane `l` of block `cta` touches, at trip `i`. -/
def touchL : Nat → EWStmt → List (Buf × Nat)
  | i, .loadIdx _ b ix => [(b, ix.eval cta i l)]
  | i, .loadV4 _ _ _ _ b ix => (List.range 4).map fun q => (b, ix.eval cta i l + q)
  | i, .storeLane b ix _ => [(b, ix.eval cta i l)]
  | i, .storeLane0 b ix _ => if l = lane0 then [(b, ix.eval cta i lane0)] else []
  | i, .seq x y => touchL i x ++ touchL i y
  | _, .forN n body => (List.range n).flatMap fun t => touchL t body
  | _, _ => []

/-- The words lane `l` of block `cta` stores, at trip `i`. -/
def storeL : Nat → EWStmt → List (Buf × Nat)
  | i, .storeLane b ix _ => [(b, ix.eval cta i l)]
  | i, .storeLane0 b ix _ => if l = lane0 then [(b, ix.eval cta i lane0)] else []
  | i, .seq x y => storeL i x ++ storeL i y
  | _, .forN n body => (List.range n).flatMap fun t => storeL t body
  | _, _ => []

end

theorem storeL_sub (cta : Nat) (l : Lane) : ∀ (s : EWStmt) (i : Nat), storeL cta l i s ⊆ touchL cta l i s := by
  intro s
  induction s with
  | storeLane b ix n => intro i; exact List.Subset.refl _
  | storeLane0 b ix n => intro i; exact List.Subset.refl _
  | seq x y ihx ihy =>
    intro i e he
    rcases List.mem_append.mp he with h | h
    · exact List.mem_append_left _ (ihx i h)
    · exact List.mem_append_right _ (ihy i h)
  | forN n body ih =>
    intro i e he
    simp only [storeL, List.mem_flatMap] at he ⊢
    obtain ⟨t, ht, he⟩ := he
    simp only [touchL, List.mem_flatMap]
    exact ⟨t, ht, ih t he⟩
  | _ => intro i e he; simp [storeL] at he

/-- **What the lists say, the access map allows**: where every word a lane's
    stores in `S` is touched by no other lane in `T`, a statement whose words
    are in `T` and stores in `S` makes only accesses its words allow. -/
theorem accOk_sub (cta : Nat) (T S : Lane → List (Buf × Nat))
    (hex : ∀ l e, e ∈ S l → ∀ l', e ∈ T l' → l' = l) :
    ∀ (s : EWStmt) (i : Nat), (∀ l, touchL cta l i s ⊆ T l) → (∀ l, storeL cta l i s ⊆ S l) →
      AccOk cta (fun l b j => (b, j) ∈ T l) (fun _ _ => none) i s := by
  intro s
  induction s with
  | loadIdx d b ix => intro i hT _ l; exact hT l (List.mem_singleton_self _)
  | loadV4 d0 d1 d2 d3 b ix =>
    intro i hT _ l q hq
    exact hT l (List.mem_map.mpr ⟨q, List.mem_range.mpr hq, rfl⟩)
  | storeLane b ix n =>
    intro i hT hS l
    have hm : (b, ix.eval cta i l) ∈ T l := hT l (List.mem_singleton_self _)
    refine ⟨hm, fun l' h => ?_⟩
    exact hex l _ (hS l (List.mem_singleton_self _)) l' h
  | storeLane0 b ix n =>
    intro i hT hS
    have ht : touchL cta lane0 i (.storeLane0 b ix n) = [(b, ix.eval cta i lane0)] := by simp [touchL]
    have hs : storeL cta lane0 i (.storeLane0 b ix n) = [(b, ix.eval cta i lane0)] := by simp [storeL]
    have hm : (b, ix.eval cta i lane0) ∈ T lane0 := hT lane0 (by rw [ht]; exact List.mem_singleton_self _)
    refine ⟨hm, fun l' h => ?_⟩
    exact hex lane0 _ (hS lane0 (by rw [hs]; exact List.mem_singleton_self _)) l' h
  | seq x y ihx ihy =>
    intro i hT hS
    exact ⟨ihx i (fun l e he => hT l (List.mem_append_left _ he)) (fun l e he => hS l (List.mem_append_left _ he)),
      ihy i (fun l e he => hT l (List.mem_append_right _ he)) (fun l e he => hS l (List.mem_append_right _ he))⟩
  | forN n body ih =>
    intro i hT hS j hj
    refine ih j (fun l e he => hT l ?_) (fun l e he => hS l ?_)
    · simp only [touchL, List.mem_flatMap, List.mem_range]; exact ⟨j, hj, he⟩
    · simp only [storeL, List.mem_flatMap, List.mem_range]; exact ⟨j, hj, he⟩
  | forM bb a body _ => intro i _ _ j hj; simp [imOf] at hj
  | _ => intro i _ _; trivial

/-- What a lane may store, its list holds. -/
theorem storesL_mem (cta : Nat) (l : Lane) :
    ∀ (s : EWStmt) (i : Nat) (b : Buf) (j : Nat), StoresL cta (imOf fun _ _ => none) l i s b j →
      (b, j) ∈ storeL cta l i s := by
  intro s
  induction s with
  | storeLane b0 ix n => intro i b j ⟨hb, hj⟩; subst hb hj; simp [storeL]; rfl
  | storeLane0 b0 ix n => intro i b j ⟨hl, hb, hj⟩; subst hl hb hj; simp [storeL]; rfl
  | seq x y ihx ihy =>
    intro i b j h
    rcases h with h | h
    · exact List.mem_append_left _ (ihx i b j h)
    · exact List.mem_append_right _ (ihy i b j h)
  | forN n body ih =>
    intro i b j ⟨t, ht, h⟩
    simp only [storeL, List.mem_flatMap, List.mem_range]
    exact ⟨t, ht, ih t b j h⟩
  | forM bb a body ih => intro i b j ⟨t, ht, _⟩; simp [imOf] at ht
  | _ => intro i b j h; exact absurd h id

/-- The range of a list of numbers, none if it is empty. -/
def rng (js : List Nat) : Option (Nat × Nat) :=
  js.foldl (fun acc j => match acc with | none => some (j, j) | some (a, b) => some (min a j, max b j)) none

theorem rng_mem : ∀ (js : List Nat) (acc : Option (Nat × Nat)) (j : Nat),
    (j ∈ js ∨ ∃ a b, acc = some (a, b) ∧ a ≤ j ∧ j ≤ b) →
    ∃ a b, js.foldl (fun acc j => match acc with | none => some (j, j) | some (a, b) => some (min a j, max b j)) acc =
      some (a, b) ∧ a ≤ j ∧ j ≤ b := by
  intro js
  induction js with
  | nil =>
    intro acc j h
    rcases h with h | h
    · cases h
    · exact h
  | cons x xs ih =>
    intro acc j h
    simp only [List.foldl_cons]
    apply ih
    rcases h with h | ⟨a, b, rfl, ha, hb⟩
    · rcases List.mem_cons.mp h with rfl | h
      · right
        cases acc with
        | none => exact ⟨j, j, rfl, Nat.le_refl _, Nat.le_refl _⟩
        | some p => exact ⟨min p.1 j, max p.2 j, rfl, Nat.min_le_right _ _, Nat.le_max_right _ _⟩
      · exact Or.inl h
    · right
      exact ⟨min a x, max b x, rfl, Nat.le_trans (Nat.min_le_left _ _) ha, Nat.le_trans hb (Nat.le_max_left _ _)⟩

theorem rng_of_mem {js : List Nat} {j : Nat} (h : j ∈ js) : ∃ a b, rng js = some (a, b) ∧ a ≤ j ∧ j ≤ b :=
  rng_mem js none j (Or.inl h)

/-- Two ranges that share no number. -/
def apart : Option (Nat × Nat) → Option (Nat × Nat) → Bool
  | some (a, b), some (c, d) => b < c || d < a
  | _, _ => true

/-! ## Phases -/

/-- A statement's parts along its left spine: a fused group's ops. -/
def spine : EWStmt → List EWStmt
  | .seq a b => spine a ++ [b]
  | s => [s]

theorem split_spine : ∀ s, Split s (spine s)
  | .seq a b => Split.seq (split_spine a) (Split.one b)
  | .skip => Split.one _
  | .setR _ _ => Split.one _
  | .shflXor _ _ _ => Split.one _
  | .loadIdx _ _ _ => Split.one _
  | .loadV4 _ _ _ _ _ _ => Split.one _
  | .storeLane0 _ _ _ => Split.one _
  | .storeLane _ _ _ => Split.one _
  | .stSm _ _ => Split.one _
  | .ldSm _ _ => Split.one _
  | .barrier => Split.one _
  | .forN _ _ => Split.one _
  | .forM _ _ _ => Split.one _
  | .cvtIF _ _ => Split.one _

theorem touchL_split (cta : Nat) (l : Lane) (i : Nat) :
    ∀ {s : EWStmt} {ps : List EWStmt}, Split s ps → ∀ p ∈ ps, touchL cta l i p ⊆ touchL cta l i s := by
  intro s ps h
  induction h with
  | one s => intro p hp; simp at hp; rw [hp]; exact List.Subset.refl _
  | seq ha hb iha ihb =>
    intro p hp e he
    rcases List.mem_append.mp hp with hp | hp
    · exact List.mem_append_left _ (iha p hp he)
    · exact List.mem_append_right _ (ihb p hp he)

theorem touchL_mem_split (cta : Nat) (l : Lane) (i : Nat) :
    ∀ {s : EWStmt} {ps : List EWStmt}, Split s ps → ∀ e ∈ touchL cta l i s, ∃ q, q < ps.length ∧
      e ∈ touchL cta l i (ps.getD q .skip) := by
  intro s ps h
  induction h with
  | one s => intro e he; exact ⟨0, by simp, by simpa using he⟩
  | seq ha hb iha ihb =>
    rename_i pa pb
    intro e he
    rcases List.mem_append.mp he with he | he
    · obtain ⟨q, hq, h⟩ := iha e he
      refine ⟨q, by simp; omega, ?_⟩
      rwa [List.getD_eq_getElem?_getD, List.getElem?_append_left hq, ← List.getD_eq_getElem?_getD]
    · obtain ⟨q, hq, h⟩ := ihb e he
      refine ⟨pa.length + q, by simp; omega, ?_⟩
      rwa [List.getD_eq_getElem?_getD, List.getElem?_append_right (by omega), Nat.add_sub_cancel_left,
        ← List.getD_eq_getElem?_getD]

theorem storeL_mem_split (cta : Nat) (l : Lane) (i : Nat) :
    ∀ {s : EWStmt} {ps : List EWStmt}, Split s ps → ∀ e ∈ storeL cta l i s, ∃ q, q < ps.length ∧
      e ∈ storeL cta l i (ps.getD q .skip) := by
  intro s ps h
  induction h with
  | one s => intro e he; exact ⟨0, by simp, by simpa using he⟩
  | seq ha hb iha ihb =>
    rename_i pa pb
    intro e he
    rcases List.mem_append.mp he with he | he
    · obtain ⟨q, hq, h⟩ := iha e he
      refine ⟨q, by simp; omega, ?_⟩
      rwa [List.getD_eq_getElem?_getD, List.getElem?_append_left hq, ← List.getD_eq_getElem?_getD]
    · obtain ⟨q, hq, h⟩ := ihb e he
      refine ⟨pa.length + q, by simp; omega, ?_⟩
      rwa [List.getD_eq_getElem?_getD, List.getElem?_append_right (by omega), Nat.add_sub_cancel_left,
        ← List.getD_eq_getElem?_getD]

/-- Block `cta`'s words, with lane and phase, its phases store, or touch. -/
def phStores (ps : List EWStmt) (cta : Nat) : List ((Nat × Nat) × Nat × Nat) :=
  (List.range ps.length).flatMap fun q => (List.finRange W).flatMap fun l =>
    (storeL cta l 0 (ps.getD q .skip)).map fun e => (e, l.val, q)
def phTouches (ps : List EWStmt) (cta : Nat) : List ((Nat × Nat) × Nat × Nat) :=
  (List.range ps.length).flatMap fun q => (List.finRange W).flatMap fun l =>
    (touchL cta l 0 (ps.getD q .skip)).map fun e => (e, l.val, q)

/-- **Each word block `cta` stores has one owner**, a lane and a phase, and its
    other lanes touch it only in later phases and from higher lanes: decided by
    a map from each stored word to the lane and phase its stores name. -/
def ownB (S T : List ((Nat × Nat) × Nat × Nat)) : Bool :=
  let m : Std.HashMap (Nat × Nat) (Nat × Nat) := S.foldl (fun m x => m.insert x.1 x.2) {}
  S.all (fun x => m[x.1]? == some x.2) &&
    T.all fun y => match m[y.1]? with
      | none => true
      | some (o, p) => y.2.1 == o || (decide (p < y.2.2) && decide (o < y.2.1))

theorem mem_phStores {ps : List EWStmt} {cta : Nat} {l : Lane} {q : Nat} {e : Nat × Nat} (hq : q < ps.length)
    (h : e ∈ storeL cta l 0 (ps.getD q .skip)) : (e, l.val, q) ∈ phStores ps cta :=
  List.mem_flatMap.mpr ⟨q, List.mem_range.mpr hq, List.mem_flatMap.mpr ⟨l, List.mem_finRange l, List.mem_map.mpr ⟨e, h, rfl⟩⟩⟩

theorem mem_phTouches {ps : List EWStmt} {cta : Nat} {l : Lane} {q : Nat} {e : Nat × Nat} (hq : q < ps.length)
    (h : e ∈ touchL cta l 0 (ps.getD q .skip)) : (e, l.val, q) ∈ phTouches ps cta :=
  List.mem_flatMap.mpr ⟨q, List.mem_range.mpr hq, List.mem_flatMap.mpr ⟨l, List.mem_finRange l, List.mem_map.mpr ⟨e, h, rfl⟩⟩⟩

theorem ownB_sound {ps : List EWStmt} {cta : Nat} (h : ownB (phStores ps cta) (phTouches ps cta) = true) :
    (∀ (l l' : Lane) q q' (e : Nat × Nat), q < ps.length → q' < ps.length →
      e ∈ storeL cta l 0 (ps.getD q .skip) → e ∈ storeL cta l' 0 (ps.getD q' .skip) → l = l' ∧ q = q') ∧
    (∀ (t o : Lane) q p (e : Nat × Nat), q < ps.length → p < ps.length →
      e ∈ touchL cta t 0 (ps.getD q .skip) → e ∈ storeL cta o 0 (ps.getD p .skip) → t ≠ o → p < q ∧ o.val < t.val) := by
  simp only [ownB, Bool.and_eq_true, List.all_eq_true] at h
  obtain ⟨hS, hT⟩ := h
  generalize hm : (phStores ps cta).foldl (fun m x => m.insert x.1 x.2) ({} : Std.HashMap (Nat × Nat) (Nat × Nat)) = m
    at hS hT
  have mS : ∀ (l : Lane) q e, q < ps.length → e ∈ storeL cta l 0 (ps.getD q .skip) → m[e]? = some (l.val, q) := by
    intro l q e hq he
    simpa using hS _ (mem_phStores hq he)
  refine ⟨fun l l' q q' e hq hq' h1 h2 => ?_, fun t o q p e hq hp h1 h2 hne => ?_⟩
  · have e1 := mS l q e hq h1
    rw [mS l' q' e hq' h2] at e1
    simp only [Option.some.injEq, Prod.mk.injEq] at e1
    exact ⟨Fin.ext e1.1.symm, e1.2.symm⟩
  · have hy := hT _ (mem_phTouches hq h1)
    simp only [mS o p e hp h2] at hy
    simp only [Bool.or_eq_true, beq_iff_eq, Bool.and_eq_true, decide_eq_true_eq] at hy
    rcases hy with hy | hy
    · exact absurd (Fin.ext hy) hne
    · exact hy

/-- The words of buffer `b` in a list of words with lanes and phases. -/
def ofBuf (L : List ((Nat × Nat) × Nat × Nat)) (b : Buf) : List Nat :=
  L.filterMap fun x => if x.1.1 = b then some x.1.2 else none

theorem mem_ofBuf {L : List ((Nat × Nat) × Nat × Nat)} {b j l q : Nat} (h : ((b, j), l, q) ∈ L) : j ∈ ofBuf L b :=
  List.mem_filterMap.mpr ⟨((b, j), l, q), h, by simp⟩

/-! ## Blocks a stride apart -/

/-- An index's value where it is literals alone. -/
def litV : IdxE → Option Nat
  | .lit n => some n
  | .add a b => do let x ← litV a; let y ← litV b; pure (x + y)
  | .mul a b => do let x ← litV a; let y ← litV b; pure (x * y)
  | _ => none

theorem litV_eval (cta i : Nat) (l : Lane) : ∀ (e : IdxE) (v : Nat), litV e = some v → e.eval cta i l = v
  | .lit n, v, h => by simp only [litV, Option.some.injEq] at h; simp [IdxE.eval, h]
  | .add a b, v, h => by
    cases ha : litV a with
    | none => simp [litV, ha] at h
    | some x =>
      cases hb : litV b with
      | none => simp [litV, ha, hb] at h
      | some y =>
        simp only [litV, ha, hb, bind, Option.bind, pure, Option.some.injEq] at h
        simp only [IdxE.eval]; rw [litV_eval cta i l a x ha, litV_eval cta i l b y hb]; exact h
  | .mul a b, v, h => by
    cases ha : litV a with
    | none => simp [litV, ha] at h
    | some x =>
      cases hb : litV b with
      | none => simp [litV, ha, hb] at h
      | some y =>
        simp only [litV, ha, hb, bind, Option.bind, pure, Option.some.injEq] at h
        simp only [IdxE.eval]; rw [litV_eval cta i l a x ha, litV_eval cta i l b y hb]; exact h
  | .laneId, v, h | .loopI, v, h | .ctaId, v, h | .ireg _, v, h | .ldIdx _ _, v, h => by simp [litV] at h

/-- How much an index grows from one block to the next, where it grows by as
    much at every lane and trip. -/
def coefC : IdxE → Option Nat
  | .ctaId => some 1
  | .laneId => some 0
  | .loopI => some 0
  | .lit _ => some 0
  | .add a b => do let x ← coefC a; let y ← coefC b; pure (x + y)
  | .mul a b => do
      let x ← coefC a; let y ← coefC b
      match litV a, litV b with
      | _, some v => pure (x * v)
      | some v, none => pure (v * y)
      | none, none => if x = 0 ∧ y = 0 then pure 0 else none
  | _ => none

theorem coefC_eval (cta i : Nat) (l : Lane) : ∀ (e : IdxE) (k : Nat), coefC e = some k →
    e.eval cta i l = e.eval 0 i l + cta * k
  | .ctaId, k, h => by simp only [coefC, Option.some.injEq] at h; subst h; simp [IdxE.eval]
  | .laneId, k, h | .loopI, k, h | .lit _, k, h => by
    simp only [coefC, Option.some.injEq] at h; subst h; simp [IdxE.eval]
  | .add a b, k, h => by
    cases ha : coefC a with
    | none => simp [coefC, ha] at h
    | some x =>
      cases hb : coefC b with
      | none => simp [coefC, ha, hb] at h
      | some y =>
        simp only [coefC, ha, hb, bind, Option.bind, pure, Option.some.injEq] at h
        subst h
        simp only [IdxE.eval]; rw [coefC_eval cta i l a x ha, coefC_eval cta i l b y hb, Nat.mul_add]; omega
  | .mul a b, k, h => by
    cases ha : coefC a with
    | none => simp [coefC, ha] at h
    | some x =>
      cases hb : coefC b with
      | none => simp [coefC, ha, hb] at h
      | some y =>
        simp only [coefC, ha, hb, bind, Option.bind] at h
        have ea := coefC_eval cta i l a x ha
        have eb := coefC_eval cta i l b y hb
        simp only [IdxE.eval]
        split at h
        · rename_i v hv
          simp only [pure, Option.some.injEq] at h; subst h
          rw [litV_eval cta i l b v hv, litV_eval 0 i l b v hv, ea, Nat.add_mul, Nat.mul_assoc]
        · rename_i v hv _
          simp only [pure, Option.some.injEq] at h; subst h
          rw [litV_eval cta i l a v hv, litV_eval 0 i l a v hv, eb, Nat.mul_add, ← Nat.mul_assoc v cta y,
            Nat.mul_comm v cta, Nat.mul_assoc]
        · split at h
          · rename_i hxy
            simp only [pure, Option.some.injEq] at h; subst h
            rw [ea, eb, hxy.1, hxy.2]; simp
          · cases h
  | .ireg _, k, h | .ldIdx _ _, k, h => by simp [coefC] at h

/-- Every index into buffer `b` grows by `S b` from one block to the next. -/
def strideOk (S : Buf → Nat) : EWStmt → Bool
  | .loadIdx _ b ix => coefC ix == some (S b)
  | .loadV4 _ _ _ _ b ix => coefC ix == some (S b)
  | .storeLane b ix _ => coefC ix == some (S b)
  | .storeLane0 b ix _ => coefC ix == some (S b)
  | .seq x y => strideOk S x && strideOk S y
  | .forN _ body => strideOk S body
  | _ => true

theorem strideOk_split (S : Buf → Nat) :
    ∀ {s : EWStmt} {ps : List EWStmt}, Split s ps → strideOk S s = true → ∀ p ∈ ps, strideOk S p = true := by
  intro s ps h
  induction h with
  | one s => intro hs p hp; simp at hp; rw [hp]; exact hs
  | seq ha hb iha ihb =>
    intro hs p hp
    simp only [strideOk, Bool.and_eq_true] at hs
    rcases List.mem_append.mp hp with hp | hp
    · exact iha hs.1 p hp
    · exact ihb hs.2 p hp

/-- Each buffer's strides: the growth of its first index. -/
def strides (s : EWStmt) : Buf → Nat :=
  let rec accs : EWStmt → List (Buf × Option Nat)
    | .loadIdx _ b ix => [(b, coefC ix)]
    | .loadV4 _ _ _ _ b ix => [(b, coefC ix)]
    | .storeLane b ix _ => [(b, coefC ix)]
    | .storeLane0 b ix _ => [(b, coefC ix)]
    | .seq x y => accs x ++ accs y
    | .forN _ body => accs body
    | _ => []
  fun b => match (accs s).find? (·.1 == b) with
    | some (_, some k) => k
    | _ => 0

/-- A word moved `cta` strides along its buffer. -/
def shiftE (S : Buf → Nat) (cta : Nat) (e : Buf × Nat) : Buf × Nat := (e.1, e.2 + cta * S e.1)

theorem touchL_shift (S : Buf → Nat) (cta : Nat) (l : Lane) :
    ∀ (s : EWStmt) (i : Nat), strideOk S s = true → touchL cta l i s = (touchL 0 l i s).map (shiftE S cta) := by
  intro s
  induction s with
  | loadIdx d b ix =>
    intro i h; simp only [strideOk, beq_iff_eq] at h
    simp [touchL, shiftE, coefC_eval cta i l ix _ h]
  | loadV4 d0 d1 d2 d3 b ix =>
    intro i h; simp only [strideOk, beq_iff_eq] at h
    simp only [touchL, List.map_map]
    apply List.map_congr_left; intro q _
    simp only [Function.comp, shiftE, coefC_eval cta i l ix _ h]; congr 1; omega
  | storeLane b ix n =>
    intro i h; simp only [strideOk, beq_iff_eq] at h
    simp [touchL, shiftE, coefC_eval cta i l ix _ h]
  | storeLane0 b ix n =>
    intro i h; simp only [strideOk, beq_iff_eq] at h
    by_cases hl : l = lane0 <;> simp [touchL, shiftE, hl, coefC_eval cta i lane0 ix _ h]
  | seq x y ihx ihy =>
    intro i h; simp only [strideOk, Bool.and_eq_true] at h
    simp only [touchL, List.map_append, ihx i h.1, ihy i h.2]
  | forN n body ih =>
    intro i h; simp only [strideOk] at h
    simp only [touchL, List.map_flatMap]
    congr 1; funext t; exact ih t h
  | _ => intro i _; rfl

theorem storeL_shift (S : Buf → Nat) (cta : Nat) (l : Lane) :
    ∀ (s : EWStmt) (i : Nat), strideOk S s = true → storeL cta l i s = (storeL 0 l i s).map (shiftE S cta) := by
  intro s
  induction s with
  | storeLane b ix n =>
    intro i h; simp only [strideOk, beq_iff_eq] at h
    simp [storeL, shiftE, coefC_eval cta i l ix _ h]
  | storeLane0 b ix n =>
    intro i h; simp only [strideOk, beq_iff_eq] at h
    by_cases hl : l = lane0 <;> simp [storeL, shiftE, hl, coefC_eval cta i lane0 ix _ h]
  | seq x y ihx ihy =>
    intro i h; simp only [strideOk, Bool.and_eq_true] at h
    simp only [storeL, List.map_append, ihx i h.1, ihy i h.2]
  | forN n body ih =>
    intro i h; simp only [strideOk] at h
    simp only [storeL, List.map_flatMap]
    congr 1; funext t; exact ih t h
  | _ => intro i _; rfl

theorem shiftE_inj {S : Buf → Nat} {cta : Nat} {e e' : Buf × Nat} (h : shiftE S cta e = shiftE S cta e') : e = e' := by
  obtain ⟨b, j⟩ := e; obtain ⟨b', j'⟩ := e'
  simp only [shiftE, Prod.mk.injEq] at h
  obtain ⟨rfl, h⟩ := h
  simp only [Prod.mk.injEq, true_and]; omega

/-- An index fitting at a block fits at every earlier one, and is no larger. -/
theorem idxL_mono {cta cta' i l : Nat} (hc : cta' ≤ cta) :
    ∀ (e : IdxE) (x : Nat), idxL (fun _ _ => none) cta i l e = some x →
      ∃ y, idxL (fun _ _ => none) cta' i l e = some y ∧ y ≤ x
  | .laneId, x, h | .loopI, x, h | .lit _, x, h => ⟨x, h, Nat.le_refl _⟩
  | .ctaId, x, h => by
    simp only [idxL] at h ⊢
    split at h
    · simp only [Option.some.injEq] at h; subst h; exact ⟨cta', by rw [if_pos (by omega)], hc⟩
    · cases h
  | .add a b, x, h => by
    simp only [idxL, bind, Option.bind] at h
    cases ha : idxL (fun _ _ => none) cta i l a with
    | none => simp [ha] at h
    | some u =>
      cases hb : idxL (fun _ _ => none) cta i l b with
      | none => simp [ha, hb] at h
      | some v =>
        simp only [ha, hb] at h
        split at h
        · simp only [Option.some.injEq] at h; subst h
          obtain ⟨u', hu', hu⟩ := idxL_mono hc a u ha
          obtain ⟨v', hv', hv⟩ := idxL_mono hc b v hb
          refine ⟨u' + v', ?_, Nat.add_le_add hu hv⟩
          simp only [idxL, bind, Option.bind, hu', hv']; rw [if_pos (by omega)]
        · cases h
  | .mul a b, x, h => by
    simp only [idxL, bind, Option.bind] at h
    cases ha : idxL (fun _ _ => none) cta i l a with
    | none => simp [ha] at h
    | some u =>
      cases hb : idxL (fun _ _ => none) cta i l b with
      | none => simp [ha, hb] at h
      | some v =>
        simp only [ha, hb] at h
        split at h
        · rename_i hlt
          simp only [Option.some.injEq] at h; subst h
          obtain ⟨u', hu', hu⟩ := idxL_mono hc a u ha
          obtain ⟨v', hv', hv⟩ := idxL_mono hc b v hb
          have hle := Nat.mul_le_mul hu hv
          refine ⟨u' * v', ?_, hle⟩
          simp only [idxL, bind, Option.bind, hu', hv']; rw [if_pos (by omega)]
        · cases h
  | .ldIdx _ _, x, h => by
    simp only [idxL, bind, Option.bind] at h
    split at h <;> cases h
  | .ireg _, x, h => by simp [idxL] at h

/-- A static statement fitting at a block fits at every earlier one. -/
theorem fits_mono {cta cta' : Nat} (hc : cta' ≤ cta) :
    ∀ (s : EWStmt) (i : Nat), fits (fun _ _ => none) cta i s = true → fits (fun _ _ => none) cta' i s = true := by
  have hall : ∀ (ix : IdxE) i, ((List.range 32).all fun l => (idxL (fun _ _ => none) cta i l ix).isSome) = true →
      ((List.range 32).all fun l => (idxL (fun _ _ => none) cta' i l ix).isSome) = true := by
    intro ix i h
    simp only [List.all_eq_true, List.mem_range, Option.isSome_iff_exists] at h ⊢
    intro l hl
    obtain ⟨x, hx⟩ := h l hl
    obtain ⟨y, hy, _⟩ := idxL_mono hc ix x hx
    exact ⟨y, hy⟩
  intro s
  induction s with
  | loadIdx _ _ ix => exact fun i h => hall ix i h
  | loadV4 _ _ _ _ _ ix => exact fun i h => hall ix i h
  | storeLane _ ix _ => exact fun i h => hall ix i h
  | storeLane0 _ ix _ =>
    intro i h
    simp only [fits, Option.isSome_iff_exists] at h ⊢
    obtain ⟨x, hx⟩ := h
    obtain ⟨y, hy, _⟩ := idxL_mono hc ix x hx
    exact ⟨y, hy⟩
  | seq a b iha ihb =>
    intro i h; simp only [fits, Bool.and_eq_true] at h ⊢; exact ⟨iha i h.1, ihb i h.2⟩
  | forN n body ih =>
    intro i h
    simp only [fits, Bool.and_eq_true, List.all_eq_true] at h ⊢
    exact ⟨h.1, fun j hj => ih j (h.2 j hj)⟩
  | forM b a body _ => intro i h; simp [fits] at h
  | _ => intro i h; exact h

/-- Ranges, a stride apart from one block to the next, that two different
    blocks' copies never share a number of. -/
def apartS (k : Nat) : Option (Nat × Nat) → Option (Nat × Nat) → Bool
  | some (s0, s1), some (t0, t1) => t1 < s0 + k && s1 < t0 + k
  | _, _ => true

/-- Words `js` of block 0, the least `a`, that lie, from `a`, within the first
    `k` of every `G·k`: block `c`'s copies then lie within the `c`-th `k` of
    every `G·k`, so no two blocks' copies meet (`win_apart`). A batch of rows a
    block reduces, one word per row, is such a set. -/
def winOk (G k : Nat) (rt : Option (Nat × Nat)) (js : List Nat) : Bool :=
  match rt with
  | some (a, _) => decide (0 < k) && js.all fun j => decide ((j - a) % (G * k) < k)
  | none => true

theorem win_apart {G k a u t c c' : Nat} (hk : 0 < k) (hc : c < G) (hc' : c' < G) (hau : a ≤ u)
    (hat : a ≤ t) (hu : (u - a) % (G * k) < k) (ht : (t - a) % (G * k) < k)
    (he : u + c' * k = t + c * k) : c = c' := by
  have hQ : 0 < G * k := Nat.mul_pos (by omega) hk
  have du := Nat.div_add_mod (u - a) (G * k)
  have dt := Nat.div_add_mod (t - a) (G * k)
  generalize (u - a) / (G * k) = x at du
  generalize (u - a) % (G * k) = y at du hu
  generalize (t - a) / (G * k) = x' at dt
  generalize (t - a) % (G * k) = y' at dt ht
  have h1 : (c' + 1) * k ≤ G * k := Nat.mul_le_mul_right _ (by omega)
  have h2 : (c + 1) * k ≤ G * k := Nat.mul_le_mul_right _ (by omega)
  rw [Nat.succ_mul] at h1 h2
  have hA : y + c' * k < G * k := by omega
  have hB : y' + c * k < G * k := by omega
  have heq : G * k * x + (y + c' * k) = G * k * x' + (y' + c * k) := by omega
  have hx : x = x' := by
    have := congrArg (· / (G * k)) heq
    simp only [Nat.mul_add_div hQ, Nat.div_eq_of_lt hA, Nat.div_eq_of_lt hB] at this
    omega
  subst hx
  have hy : y + c' * k = y' + c * k := by omega
  have := congrArg (· / k) hy
  simp only [Nat.add_mul_div_right _ _ hk, Nat.div_eq_of_lt hu, Nat.div_eq_of_lt ht] at this
  omega

/-- **A buffer's words at `G` blocks a stride `k` apart**, block 0 storing the
    range `rs` and touching `rt`, its touched words `ts`: the last block's
    words inside its `szb` bytes, and no block storing a word another touches —
    the ranges a stride apart (`apartS`) or the words in windows (`winOk`). -/
def bufOk (G k szb : Nat) (rs rt : Option (Nat × Nat)) (ts : List Nat) : Bool :=
  (match rt with
    | some (_, t1) => decide (t1 + (G - 1) * k < 2 ^ 32 ∧ 4 * (t1 + (G - 1) * k) + 4 ≤ szb)
    | none => true) &&
  (decide (G ≤ 1) || apartS k rs rt || winOk G k rt ts)

/-- **A static statement's launch of `G` blocks over buffers of `sz` bytes, in
    phases `ps`, is what `launch_phases` needs**, decided at block 0 where each
    buffer's indexes grow by a stride from one block to the next: block 0's
    words owned and in the buffers, the last block's indexes fitting (and so
    every earlier one's), and each buffer's words passing `bufOk`. -/
def staticOk (nbuf G : Nat) (sz : Nat → Nat) (ps : List EWStmt) (s : EWStmt) : Bool :=
  let S := strides s
  let St := phStores ps 0
  let T := phTouches ps 0
  strideOk S s && fits (fun _ _ => none) (G - 1) 0 s && ownB St T &&
    T.all (fun e => decide (e.1.1 < nbuf ∧ e.1.1 < 268435449)) &&
    (List.range nbuf).all fun b => bufOk G (S b) (sz b) (rng (ofBuf St b)) (rng (ofBuf T b)) (ofBuf T b)

/-! ## Statements alike in what they touch -/

/-- A statement with every lane expression it sets made `.reg 0`: what it
    touches, stores and fits is the statement's (`staticOk_erase`). -/
def erase : EWStmt → EWStmt
  | .setR r _ => .setR r (.reg 0)
  | .seq a b => .seq (erase a) (erase b)
  | .forN n b => .forN n (erase b)
  | .forM bb a b => .forM bb a (erase b)
  | s => s

deriving instance DecidableEq for IdxE

/-- Equality of lane expressions built from registers alone. -/
def regEq : WFExp → WFExp → Bool
  | .reg r, .reg r' => r == r'
  | _, _ => false

/-- Equality of erased statements. -/
def eqE : EWStmt → EWStmt → Bool
  | .skip, .skip => true
  | .seq a b, .seq a' b' => eqE a a' && eqE b b'
  | .setR r e, .setR r' e' => r == r' && regEq e e'
  | .shflXor a b c, .shflXor a' b' c' => a == a' && b == b' && c == c'
  | .loadIdx d b ix, .loadIdx d' b' ix' => d == d' && b == b' && decide (ix = ix')
  | .loadV4 d0 d1 d2 d3 b ix, .loadV4 d0' d1' d2' d3' b' ix' =>
      d0 == d0' && d1 == d1' && d2 == d2' && d3 == d3' && b == b' && decide (ix = ix')
  | .storeLane0 b ix r, .storeLane0 b' ix' r' => b == b' && decide (ix = ix') && r == r'
  | .storeLane b ix r, .storeLane b' ix' r' => b == b' && decide (ix = ix') && r == r'
  | .stSm ix r, .stSm ix' r' => decide (ix = ix') && r == r'
  | .ldSm d ix, .ldSm d' ix' => d == d' && decide (ix = ix')
  | .barrier, .barrier => true
  | .forN n b, .forN n' b' => n == n' && eqE b b'
  | .forM bb a b, .forM bb' a' b' => bb == bb' && a == a' && eqE b b'
  | .cvtIF d ix, .cvtIF d' ix' => d == d' && decide (ix = ix')
  | _, _ => false

theorem regEq_eq : ∀ {e e' : WFExp}, regEq e e' = true → e = e'
  | .reg r, .reg r', h => by simp only [regEq, beq_iff_eq] at h; rw [h]
  | .reg _, .lit _, h | .reg _, .add _ _, h | .reg _, .mul _ _, h | .reg _, .neg _, h | .reg _, .inv _, h
  | .reg _, .exp _, h | .reg _, .ex2 _, h | .reg _, .rsqrt _, h | .reg _, .maxW _ _, h
  | .reg _, .geF _ _, h => by simp [regEq] at h
  | .lit _, _, h | .add _ _, _, h | .mul _ _, _, h | .neg _, _, h | .inv _, _, h | .exp _, _, h
  | .ex2 _, _, h | .rsqrt _, _, h | .maxW _ _, _, h | .geF _ _, _, h => by simp [regEq] at h

theorem eqE_eq : ∀ {s s' : EWStmt}, eqE s s' = true → s = s' := by
  intro s
  induction s with
  | seq a b iha ihb =>
    intro s' h; cases s' <;> simp only [eqE, Bool.and_eq_true, reduceCtorEq] at h
    rw [iha h.1, ihb h.2]
  | forN n b ih =>
    intro s' h; cases s' <;> simp only [eqE, Bool.and_eq_true, beq_iff_eq, reduceCtorEq] at h
    rw [h.1, ih h.2]
  | forM bb a b ih =>
    intro s' h; cases s' <;> simp only [eqE, Bool.and_eq_true, beq_iff_eq, reduceCtorEq] at h
    rw [h.1.1, h.1.2, ih h.2]
  | setR r e =>
    intro s' h; cases s' <;> simp only [eqE, Bool.and_eq_true, beq_iff_eq, reduceCtorEq] at h
    rw [h.1, regEq_eq h.2]
  | _ =>
    intro s' h; cases s' <;> simp_all [eqE]

theorem touchL_erase (cta : Nat) (l : Lane) : ∀ (s : EWStmt) (i : Nat), touchL cta l i (erase s) = touchL cta l i s := by
  intro s; induction s with
  | seq a b iha ihb => intro i; simp only [erase, touchL, iha, ihb]
  | forN n b ih => intro i; simp only [erase, touchL, ih]
  | _ => intro i; rfl

theorem storeL_erase (cta : Nat) (l : Lane) : ∀ (s : EWStmt) (i : Nat), storeL cta l i (erase s) = storeL cta l i s := by
  intro s; induction s with
  | seq a b iha ihb => intro i; simp only [erase, storeL, iha, ihb]
  | forN n b ih => intro i; simp only [erase, storeL, ih]
  | _ => intro i; rfl

theorem strideOk_erase (S : Buf → Nat) : ∀ s : EWStmt, strideOk S (erase s) = strideOk S s := by
  intro s; induction s with
  | seq a b iha ihb => simp only [erase, strideOk, iha, ihb]
  | forN n b ih => simp only [erase, strideOk, ih]
  | _ => rfl

theorem accs_erase : ∀ s : EWStmt, strides.accs (erase s) = strides.accs s := by
  intro s; induction s with
  | seq a b iha ihb => simp only [erase, strides.accs, iha, ihb]
  | forN n b ih => simp only [erase, strides.accs, ih]
  | _ => rfl

theorem strides_erase (s : EWStmt) : strides (erase s) = strides s := by
  unfold strides; rw [accs_erase]

theorem fits_erase (rd : Buf → Nat → Option Nat) (cta : Nat) :
    ∀ (s : EWStmt) (i : Nat), fits rd cta i (erase s) = fits rd cta i s := by
  intro s; induction s with
  | seq a b iha ihb => intro i; simp only [erase, fits, iha, ihb]
  | forN n b ih => intro i; simp only [erase, fits, ih]
  | forM bb a b ih => intro i; simp only [erase, fits, ih]
  | _ => intro i; rfl

theorem spine_erase : ∀ s : EWStmt, spine (erase s) = (spine s).map erase := by
  intro s; induction s with
  | seq a b iha _ => simp only [erase, spine, iha, List.map_append, List.map_cons, List.map_nil]
  | _ => rfl

theorem getD_erase (ps : List EWStmt) (q : Nat) : (ps.map erase).getD q .skip = erase (ps.getD q .skip) := by
  simp only [List.getD_eq_getElem?_getD, List.getElem?_map]
  cases ps[q]? <;> rfl

theorem phStores_erase (ps : List EWStmt) (cta : Nat) : phStores (ps.map erase) cta = phStores ps cta := by
  simp only [phStores, List.length_map, getD_erase, storeL_erase]

theorem phTouches_erase (ps : List EWStmt) (cta : Nat) : phTouches (ps.map erase) cta = phTouches ps cta := by
  simp only [phTouches, List.length_map, getD_erase, touchL_erase]

/-- **What `staticOk` decides of a statement, it decides of the statement
    erased.** -/
theorem staticOk_erase (nbuf G : Nat) (sz : Nat → Nat) (s : EWStmt) :
    staticOk nbuf G sz (spine (erase s)) (erase s) = staticOk nbuf G sz (spine s) s := by
  simp only [staticOk, spine_erase, phStores_erase, phTouches_erase, strides_erase, strideOk_erase, fits_erase]

/-- A launch's key: its buffers, blocks, buffer sizes in bytes and statement. -/
abbrev Key := Nat × Nat × List Nat × EWStmt

/-- Two keys alike, statements compared erased. -/
def keyEq (k k' : Key) : Bool :=
  k.1 == k'.1 && k.2.1 == k'.2.1 && k.2.2.1 == k'.2.2.1 && eqE (erase k.2.2.2) (erase k'.2.2.2)

/-- `staticOk` at a key, sizes read from its list. -/
def keyOk (k : Key) : Bool := staticOk k.1 k.2.1 (fun b => k.2.2.1.getD b 0) (spine k.2.2.2) k.2.2.2

/-- **Keys passing `staticOk` once each, and every launch's key among them.** -/
def keysOk (ks : List Key) (xs : List Key) : Bool := ks.all keyOk && xs.all fun x => ks.any (keyEq x)

theorem keyOk_of_eq {k k' : Key} (h : keyEq k k' = true) (hk : keyOk k' = true) : keyOk k = true := by
  obtain ⟨n, G, sz, s⟩ := k
  obtain ⟨n', G', sz', s'⟩ := k'
  simp only [keyEq, Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨rfl, rfl⟩, rfl⟩, h⟩ := h
  simp only [keyOk] at hk ⊢
  rw [← staticOk_erase, eqE_eq h, staticOk_erase]
  exact hk

theorem keysOk_mem {ks xs : List Key} (h : keysOk ks xs = true) : ∀ x ∈ xs, keyOk x = true := by
  simp only [keysOk, Bool.and_eq_true, List.all_eq_true, List.any_eq_true] at h
  intro x hx
  obtain ⟨k, hk, he⟩ := h.2 x hx
  exact keyOk_of_eq he (h.1 k hk)

/-- **No block stores a word another touches**, in any phases, when `staticOk`
    holds. -/
theorem staticOk_apart (nbuf G : Nat) (sz : Nat → Nat) (s : EWStmt) (ps : List EWStmt) (hsp : Split s ps)
    (hok : staticOk nbuf G sz ps s = true) :
    ∀ c c' (l l' : Lane) q p b j, c < G → c' < G → c ≠ c' → q < ps.length → p < ps.length →
      (b, j) ∈ touchL c l 0 (ps.getD q .skip) → (b, j) ∈ storeL c' l' 0 (ps.getD p .skip) → False := by
  simp only [staticOk, bufOk, Bool.and_eq_true, Bool.or_eq_true, List.all_eq_true, List.mem_range] at hok
  obtain ⟨⟨⟨⟨hst, _⟩, _⟩, hT0⟩, hap⟩ := hok
  generalize strides s = S at hst hT0 hap
  have hph : ∀ q, q < ps.length → strideOk S (ps.getD q .skip) = true := fun q hq =>
    strideOk_split S hsp hst _ (by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hq]; exact List.getElem_mem hq)
  have htc : ∀ c (l : Lane) q e, q < ps.length → e ∈ touchL c l 0 (ps.getD q .skip) →
      ∃ e0 ∈ touchL 0 l 0 (ps.getD q .skip), e = shiftE S c e0 := fun c l q e hq h => by
    rw [touchL_shift S c l _ 0 (hph q hq)] at h
    obtain ⟨e0, h0, rfl⟩ := List.mem_map.mp h
    exact ⟨e0, h0, rfl⟩
  have hsc : ∀ c (l : Lane) q e, q < ps.length → e ∈ storeL c l 0 (ps.getD q .skip) →
      ∃ e0 ∈ storeL 0 l 0 (ps.getD q .skip), e = shiftE S c e0 := fun c l q e hq h => by
    rw [storeL_shift S c l _ 0 (hph q hq)] at h
    obtain ⟨e0, h0, rfl⟩ := List.mem_map.mp h
    exact ⟨e0, h0, rfl⟩
  intro c c' l l' q p b j hc hc' hne hq hp h hs
  obtain ⟨⟨b1, u⟩, hu, he1⟩ := hsc c' l' p _ hp hs
  obtain ⟨⟨b2, t⟩, ht, he2⟩ := htc c l q _ hq h
  simp only [shiftE, Prod.mk.injEq] at he1 he2
  obtain ⟨rfl, hj1⟩ := he1
  obtain ⟨hb2, hj2⟩ := he2
  subst hb2
  have hb := hT0 _ (mem_phTouches hq ht)
  simp only [decide_eq_true_eq] at hb
  obtain ⟨s0, s1, e1, h11, h12⟩ := rng_of_mem (mem_ofBuf (mem_phStores hp hu))
  obtain ⟨t0, t1, e2, h21, h22⟩ := rng_of_mem (mem_ofBuf (mem_phTouches hq ht))
  have hG : ¬ G ≤ 1 := by omega
  rcases (hap b hb.1).2 with (hG1 | this) | this
  · exact absurd (of_decide_eq_true hG1) hG
  · simp only [e1, e2, apartS, Bool.and_eq_true, decide_eq_true_eq] at this
    rcases Nat.lt_or_gt_of_ne hne with hlt | hlt
    · have := Nat.mul_le_mul_right (S b) (show c + 1 ≤ c' by omega)
      rw [Nat.succ_mul] at this; omega
    · have := Nat.mul_le_mul_right (S b) (show c' + 1 ≤ c by omega)
      rw [Nat.succ_mul] at this; omega
  · simp only [e2, winOk, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at this
    have hut := mem_ofBuf (mem_phTouches hp (storeL_sub 0 l' _ 0 hu))
    obtain ⟨a3, b3, e3, h31, _⟩ := rng_of_mem hut
    rw [e2] at e3
    simp only [Option.some.injEq, Prod.mk.injEq] at e3
    obtain ⟨rfl, rfl⟩ := e3
    exact hne (win_apart this.1 hc hc' h31 h21 (this.2 u hut) (this.2 t (mem_ofBuf (mem_phTouches hq ht)))
      (by omega))

/-- **…and in the whole statement.** -/
theorem staticOk_apart_all (nbuf G : Nat) (sz : Nat → Nat) (s : EWStmt) (ps : List EWStmt) (hsp : Split s ps)
    (hok : staticOk nbuf G sz ps s = true) :
    ∀ c c' (l l' : Lane) b j, c < G → c' < G → c ≠ c' →
      (b, j) ∈ touchL c l 0 s → (b, j) ∈ storeL c' l' 0 s → False := by
  intro c c' l l' b j hc hc' hne ht hs
  obtain ⟨q, hq, ht⟩ := touchL_mem_split c l 0 hsp _ ht
  obtain ⟨p, hp, hs⟩ := storeL_mem_split c' l' 0 hsp _ hs
  exact staticOk_apart nbuf G sz s ps hsp hok c c' l l' q p b j hc hc' hne hq hp ht hs

/-- **Every word a statement `staticOk` passes touches lies inside its
    buffer**, in every block of the grid: a buffer of the launch, at an index
    below `2^32`, its four bytes within the size `sz` gives it. -/
theorem staticOk_bounds (nbuf G : Nat) (sz : Nat → Nat) (s : EWStmt) (ps : List EWStmt) (hsp : Split s ps)
    (hok : staticOk nbuf G sz ps s = true) :
    ∀ c (l : Lane) b j, c < G → (b, j) ∈ touchL c l 0 s →
      b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32 ∧ 4 * j + 4 ≤ sz b := by
  simp only [staticOk, bufOk, Bool.and_eq_true, Bool.or_eq_true, List.all_eq_true, List.mem_range] at hok
  obtain ⟨⟨⟨⟨hst, -⟩, -⟩, hT0⟩, hap⟩ := hok
  generalize strides s = S at hst hT0 hap
  have hph : ∀ q, q < ps.length → strideOk S (ps.getD q .skip) = true := fun q hq =>
    strideOk_split S hsp hst _ (by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hq]; exact List.getElem_mem hq)
  intro c l b j hc h
  obtain ⟨q, hq, h⟩ := touchL_mem_split c l 0 hsp _ h
  have h' : (b, j) ∈ (touchL 0 l 0 (ps.getD q .skip)).map (shiftE S c) := by
    rw [← touchL_shift S c l _ 0 (hph q hq)]; exact h
  obtain ⟨⟨b0, j0⟩, h0, he⟩ := List.mem_map.mp h'
  simp only [shiftE, Prod.mk.injEq] at he
  obtain ⟨rfl, rfl⟩ := he
  have hb := hT0 _ (mem_phTouches hq h0)
  simp only [decide_eq_true_eq] at hb
  obtain ⟨t0, t1, e, _, ht⟩ := rng_of_mem (mem_ofBuf (mem_phTouches hq h0))
  have hB := (hap b0 hb.1).1
  simp only [e, decide_eq_true_eq] at hB
  have hm : c * S b0 ≤ (G - 1) * S b0 := Nat.mul_le_mul_right _ (by omega)
  exact ⟨hb.1, hb.2, by omega, by omega⟩

/-- The words lane `l` of block `c` touches in phase `q`, for blocks below `G`. -/
def staticAcc (G : Nat) (ps : List EWStmt) (q c : Nat) : Acc := fun l b j =>
  c < G ∧ (b, j) ∈ touchL c l 0 (ps.getD q .skip)

/-- **What `staticOk` decides**: with the phases' touches as the words lanes may
    touch, every obligation a launch of `G` blocks in phases `ps` must meet
    (`BodyPhase.launch_phases`, `BodyEnds.launch_exists`) — accesses each lane
    may make, inside the buffers and their sizes `sz`, a stored word one lane's
    in one phase, another lane touching it only later and from a higher lane,
    no block storing what another touches, and indexes that fit. -/
structure StaticObl (nbuf G : Nat) (sz : Nat → Nat) (s : EWStmt) (ps : List EWStmt) : Prop where
  acc : ∀ c q, c < G → q < ps.length → AccOk c (staticAcc G ps q c) (fun _ _ => none) 0 (ps.getD q .skip)
  bd : ∀ q c l b j, staticAcc G ps q c l b j → b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32
  sz : ∀ q c l b j, q < ps.length → staticAcc G ps q c l b j → 4 * j + 4 ≤ sz b
  own : ∀ c (l l' : Lane) q q' b j, c < G → q < ps.length → q' < ps.length →
    StoresL c (imOf fun _ _ => none) l 0 (ps.getD q .skip) b j →
    StoresL c (imOf fun _ _ => none) l' 0 (ps.getD q' .skip) b j → l = l' ∧ q = q'
  rdr : ∀ c (t o : Lane) q p b j, c < G → q < ps.length → p < ps.length → staticAcc G ps q c t b j →
    StoresL c (imOf fun _ _ => none) o 0 (ps.getD p .skip) b j → t ≠ o → p < q ∧ o.val < t.val
  disj : ∀ c c' l l' q b j, c < G → c' < G → c ≠ c' → q < ps.length → staticAcc G ps q c l b j →
    ¬ StoresL c' (imOf fun _ _ => none) l' 0 s b j
  fits : ∀ c, c < G → fits (fun _ _ => none) c 0 s = true

theorem staticOk_obl (nbuf G : Nat) (sz : Nat → Nat) (s : EWStmt) (ps : List EWStmt) (hsp : Split s ps)
    (hok : staticOk nbuf G sz ps s = true) : StaticObl nbuf G sz s ps := by
  have hok0 := hok
  simp only [staticOk, bufOk, Bool.and_eq_true, Bool.or_eq_true, List.all_eq_true, List.mem_range] at hok
  obtain ⟨⟨⟨⟨hst, hfit⟩, hown0⟩, hT0⟩, hap⟩ := hok
  generalize strides s = S at hst hT0 hap
  have hph : ∀ q, q < ps.length → strideOk S (ps.getD q .skip) = true := fun q hq =>
    strideOk_split S hsp hst _ (by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hq]; exact List.getElem_mem hq)
  have htc : ∀ c (l : Lane) q e, q < ps.length → e ∈ touchL c l 0 (ps.getD q .skip) →
      ∃ e0 ∈ touchL 0 l 0 (ps.getD q .skip), e = shiftE S c e0 := fun c l q e hq h => by
    rw [touchL_shift S c l _ 0 (hph q hq)] at h
    obtain ⟨e0, h0, rfl⟩ := List.mem_map.mp h
    exact ⟨e0, h0, rfl⟩
  have hsc : ∀ c (l : Lane) q e, q < ps.length → e ∈ storeL c l 0 (ps.getD q .skip) →
      ∃ e0 ∈ storeL 0 l 0 (ps.getD q .skip), e = shiftE S c e0 := fun c l q e hq h => by
    rw [storeL_shift S c l _ 0 (hph q hq)] at h
    obtain ⟨e0, h0, rfl⟩ := List.mem_map.mp h
    exact ⟨e0, h0, rfl⟩
  have hbd : ∀ c (l : Lane) q b j, c < G → q < ps.length → (b, j) ∈ touchL c l 0 (ps.getD q .skip) →
      b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32 ∧ 4 * j + 4 ≤ sz b := fun c l q b j hc hq h => by
    obtain ⟨⟨b0, j0⟩, h0, he⟩ := htc c l q _ hq h
    simp only [shiftE, Prod.mk.injEq] at he
    obtain ⟨rfl, rfl⟩ := he
    have hb := hT0 _ (mem_phTouches hq h0)
    simp only [decide_eq_true_eq] at hb
    obtain ⟨t0, t1, e, _, ht⟩ := rng_of_mem (mem_ofBuf (mem_phTouches hq h0))
    have hB := (hap b hb.1).1
    simp only [e, decide_eq_true_eq] at hB
    have hm : c * S b ≤ (G - 1) * S b := Nat.mul_le_mul_right _ (by omega)
    exact ⟨hb.1, hb.2, by omega, by omega⟩
  have hown : ∀ c, c < G →
      (∀ (l l' : Lane) q q' (e : Nat × Nat), q < ps.length → q' < ps.length →
        e ∈ storeL c l 0 (ps.getD q .skip) → e ∈ storeL c l' 0 (ps.getD q' .skip) → l = l' ∧ q = q') ∧
      (∀ (t o : Lane) q p (e : Nat × Nat), q < ps.length → p < ps.length →
        e ∈ touchL c t 0 (ps.getD q .skip) → e ∈ storeL c o 0 (ps.getD p .skip) → t ≠ o → p < q ∧ o.val < t.val) := by
    intro c _
    have h0 := ownB_sound hown0
    refine ⟨fun l l' q q' e hq hq' h1 h2 => ?_, fun t o q p e hq hp h1 h2 hne => ?_⟩
    · obtain ⟨e1, g1, rfl⟩ := hsc c l q e hq h1
      obtain ⟨e2, g2, he⟩ := hsc c l' q' _ hq' h2
      rw [shiftE_inj he] at g1
      exact h0.1 l l' q q' e2 hq hq' g1 g2
    · obtain ⟨e1, g1, rfl⟩ := htc c t q e hq h1
      obtain ⟨e2, g2, he⟩ := hsc c o p _ hp h2
      rw [shiftE_inj he] at g1
      exact h0.2 t o q p e2 hq hp g1 g2 hne
  refine ⟨fun c q hc hq => ?_, fun q c l b j h => ?_, fun q c l b j hq h => (hbd c l q b j h.1 hq h.2).2.2.2,
    fun c l l' q q' b j hc hq hq' h1 h2 => (hown c hc).1 l l' q q' (b, j) hq hq'
      (storesL_mem c l _ 0 b j h1) (storesL_mem c l' _ 0 b j h2),
    fun c t o q p b j hc hq hp hAt hst hne => (hown c hc).2 t o q p (b, j) hq hp hAt.2
      (storesL_mem c o _ 0 b j hst) hne,
    fun c c' l l' q b j hc hc' hne hq h hst => ?_,
    fun c hc => fits_mono (by omega : c ≤ G - 1) s 0 hfit⟩
  · have he : staticAcc G ps q c = fun l b j => (b, j) ∈ touchL c l 0 (ps.getD q .skip) := by
      funext l b j; simp [staticAcc, hc]
    rw [he]
    refine accOk_sub c (fun l => touchL c l 0 (ps.getD q .skip)) (fun l => storeL c l 0 (ps.getD q .skip))
      (fun l e he l' ht => ?_) _ 0 (fun _ => List.Subset.refl _) (fun _ => List.Subset.refl _)
    by_cases hll : l' = l
    · exact hll
    · have := (hown c hc).2 l' l q q e hq hq ht he hll; omega
  · obtain ⟨hc, h⟩ := h
    by_cases hq : q < ps.length
    · obtain ⟨h1, h2, h3, _⟩ := hbd c l q b j hc hq h; exact ⟨h1, h2, h3⟩
    · rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none (by omega)] at h
      simp [touchL] at h
  · obtain ⟨_, h⟩ := h
    obtain ⟨pp, hpp, hs⟩ := (storesL_split c' (imOf fun _ _ => none) l' 0 hsp b j).mp hst
    obtain ⟨p, hp, rfl⟩ := List.getElem_of_mem hpp
    have hpq : ps.getD p .skip = ps[p] := by simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hp]
    rw [← hpq] at hs
    exact staticOk_apart nbuf G sz s ps hsp hok0 c c' l l' q p b j hc hc' hne hq hp h
      (storesL_mem c' l' _ 0 b j hs)

/-- **A static statement's launch runs the warp machine, its obligations
    decided**: `grid` blocks of `nw` warps of the statement's kernel, in phases
    `ps` of it, its launch of `grid·nw` blocks over buffers of at least `sz`
    bytes passing `staticOk`, leave every word a lane touches as the warp
    machine's run of the statement for that lane's block does from the
    launch's memory. -/
theorem launch_static_both (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (name : String)
    (nbuf smem : Nat) (s : EWStmt) (ps : List EWStmt) (hsp : Split s ps) (nw : Nat) (hnw : 0 < nw)
    (hnw31 : nw < 2 ^ 31) (grid : Nat) (hgrid : grid * nw ≤ 2 ^ 32) (hN : 32 * nw ≤ 2 ^ 32) (sz : Nat → Nat)
    (hok : staticOk nbuf (grid * nw) sz ps s = true)
    (hfine : Fine (emitGo (fnBody (params nbuf) (Body.body nbuf s nw)) ⟨(params nbuf).length, 0, [], [], [], none⟩).2)
    (bufs bufs' : Nat → ByteArray) (hsz : ∀ b, b < nbuf → sz b ≤ (bufs b).size)
    (hr : LaunchRuns (Body.kernel name nbuf smem s nw) grid (32 * nw) (bufArgs nbuf) bufs bufs' hw) :
    (∀ cta (l : Lane) b j, cta < grid * nw → (b, j) ∈ touchL cta l 0 s →
      ({ kmem bufs with bufs := bufs' } : Mem).load (bufW nbuf b j) 4 =
        some ((@WStmt.run F32 (F32.ops hw) _ (s.elabAt cta 0) (ws0 nbuf (kmem bufs))).mem b j).bits) ∧
    (∀ b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) →
      (∀ cta, cta < grid * nw → ∀ (l : Lane), (b, j) ∉ storeL cta l 0 s) →
      ({ kmem bufs with bufs := bufs' } : Mem).load (bufW nbuf b j) 4 = (kmem bufs).load (bufW nbuf b j) 4):= by
  have ob := staticOk_obl nbuf (grid * nw) sz s ps hsp hok
  have h := launch_phases hw hw32 name nbuf smem s ps hsp nw hnw hnw31 grid hgrid hN (staticAcc (grid * nw) ps)
    (fun _ _ => none) ob.acc ob.bd ob.own ob.rdr ob.disj hfine bufs bufs' (fun _ _ _ h => by cases h)
    (fun _ _ _ _ _ _ _ => rfl) ob.fits
    (fun q c l b j hc hq h => Nat.le_trans (ob.sz q c l b j hq h) (hsz b (ob.bd q c l b j h).1)) hr
  refine ⟨fun cta l b j hc hm => ?_, fun b j hv hn => h.2 b j hv (fun c l hc hs => hn c hc l
    (storesL_mem c l s 0 b j (by rw [imOf_none] at hs ⊢; exact hs)))⟩
  obtain ⟨q, hq, hm⟩ := touchL_mem_split cta l 0 hsp _ hm
  have := h.1 q cta l b j hc hq ⟨hc, hm⟩
  rw [imOf_none] at this
  exact this


/-- **A static statement's launch runs the warp machine, its obligations
    decided**: `grid` blocks of `nw` warps of the statement's kernel, in phases
    `ps` of it, its launch of `grid·nw` blocks over buffers of at least `sz`
    bytes passing `staticOk`, leave every word a lane touches as the warp
    machine's run of the statement for that lane's block does from the
    launch's memory. -/
theorem launch_static (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (name : String)
    (nbuf smem : Nat) (s : EWStmt) (ps : List EWStmt) (hsp : Split s ps) (nw : Nat) (hnw : 0 < nw)
    (hnw31 : nw < 2 ^ 31) (grid : Nat) (hgrid : grid * nw ≤ 2 ^ 32) (hN : 32 * nw ≤ 2 ^ 32) (sz : Nat → Nat)
    (hok : staticOk nbuf (grid * nw) sz ps s = true)
    (hfine : Fine (emitGo (fnBody (params nbuf) (Body.body nbuf s nw)) ⟨(params nbuf).length, 0, [], [], [], none⟩).2)
    (bufs bufs' : Nat → ByteArray) (hsz : ∀ b, b < nbuf → sz b ≤ (bufs b).size)
    (hr : LaunchRuns (Body.kernel name nbuf smem s nw) grid (32 * nw) (bufArgs nbuf) bufs bufs' hw) :
    ∀ cta (l : Lane) b j, cta < grid * nw → (b, j) ∈ touchL cta l 0 s →
      ({ kmem bufs with bufs := bufs' } : Mem).load (bufW nbuf b j) 4 =
        some ((@WStmt.run F32 (F32.ops hw) _ (s.elabAt cta 0) (ws0 nbuf (kmem bufs))).mem b j).bits :=
  (launch_static_both hw hw32 name nbuf smem s ps hsp nw hnw hnw31 grid hgrid hN sz hok hfine bufs bufs' hsz hr).1

/-- **…and leave every word no lane stores as it was.** -/
theorem launch_static_keep (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (name : String)
    (nbuf smem : Nat) (s : EWStmt) (ps : List EWStmt) (hsp : Split s ps) (nw : Nat) (hnw : 0 < nw)
    (hnw31 : nw < 2 ^ 31) (grid : Nat) (hgrid : grid * nw ≤ 2 ^ 32) (hN : 32 * nw ≤ 2 ^ 32) (sz : Nat → Nat)
    (hok : staticOk nbuf (grid * nw) sz ps s = true)
    (hfine : Fine (emitGo (fnBody (params nbuf) (Body.body nbuf s nw)) ⟨(params nbuf).length, 0, [], [], [], none⟩).2)
    (bufs bufs' : Nat → ByteArray) (hsz : ∀ b, b < nbuf → sz b ≤ (bufs b).size)
    (hr : LaunchRuns (Body.kernel name nbuf smem s nw) grid (32 * nw) (bufArgs nbuf) bufs bufs' hw) :
    ∀ b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) →
      (∀ cta, cta < grid * nw → ∀ (l : Lane), (b, j) ∉ storeL cta l 0 s) →
      ({ kmem bufs with bufs := bufs' } : Mem).load (bufW nbuf b j) 4 = (kmem bufs).load (bufW nbuf b j) 4 :=
  (launch_static_both hw hw32 name nbuf smem s ps hsp nw hnw hnw31 grid hgrid hN sz hok hfine bufs bufs' hsz hr).2

/-- A statement's body kernel of `nw` warps a block is emitted whole. -/
def fineB (nbuf : Nat) (s : EWStmt) (nw : Nat) : Bool :=
  let t := (emitGo (fnBody (params nbuf) (Body.body nbuf s nw)) ⟨(params nbuf).length, 0, [], [], [], none⟩).2
  decide (t.err = none) && decide (needS t.out ≤ HProg.fuel)

/-- **What a launch of `grid` blocks of `nw` warps over buffers of `sz` bytes
    asks of its statement, decided**: `staticOk` with the statement's ops as
    phases, the kernel emitted whole, and a launch 32-bit indexes reach. -/
def launchOk (nbuf grid nw : Nat) (sz : Nat → Nat) (s : EWStmt) : Bool :=
  staticOk nbuf (grid * nw) sz (spine s) s && fineB nbuf s nw &&
    decide (0 < nw ∧ nw < 2 ^ 31 ∧ grid * nw ≤ 2 ^ 32 ∧ 32 * nw ≤ 2 ^ 32)

/-- **A launch passing `launchOk` runs its statement**: `grid` blocks of `nw`
    warps of the statement's kernel, over buffers of at least `sz` bytes,
    leave every word a lane touches as the warp machine's run of the statement
    for that lane's block does from the launch's memory. -/
theorem launch_body (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (name : String)
    (nbuf smem : Nat) (s : EWStmt) (nw grid : Nat) (sz : Nat → Nat) (hok : launchOk nbuf grid nw sz s = true)
    (bufs bufs' : Nat → ByteArray) (hsz : ∀ b, b < nbuf → sz b ≤ (bufs b).size)
    (hr : LaunchRuns (Body.kernel name nbuf smem s nw) grid (32 * nw) (bufArgs nbuf) bufs bufs' hw) :
    ∀ cta (l : Lane) b j, cta < grid * nw → (b, j) ∈ touchL cta l 0 s →
      ({ kmem bufs with bufs := bufs' } : Mem).load (bufW nbuf b j) 4 =
        some ((@WStmt.run F32 (F32.ops hw) _ (s.elabAt cta 0) (ws0 nbuf (kmem bufs))).mem b j).bits := by
  simp only [launchOk, fineB, Bool.and_eq_true, decide_eq_true_eq] at hok
  obtain ⟨⟨hst, herr, hneed⟩, h0, h31, hg, hN⟩ := hok
  exact launch_static hw hw32 name nbuf smem s (spine s) (split_spine s) nw h0 h31 grid hg hN sz hst
    ⟨herr, hneed⟩ bufs bufs' hsz hr

end AlgorithmLib.ML.BodyCheck
