module
public import AlgorithmLib.ML.Launch.GroupRun
meta import AlgorithmLib.ML.Launch.GroupRun
public import AlgorithmLib.ML.Launch.Bind
meta import AlgorithmLib.ML.Launch.Bind
public import AlgorithmLib.ML.Model.TenDenote
meta import AlgorithmLib.ML.Model.TenDenote
public import AlgorithmLib.ML.Model.BufsOf
meta import AlgorithmLib.ML.Model.BufsOf
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `ML.Model.GroupLink` — a group's kernel computes its members' denotations

A group of operations is one kernel: each member's statement, renamed onto the
group's buffer table (`TOp.groupStmt`). `group_den` is what a block of it
does: every word it touches is left as the members' denotations, folded in
order (`TOp.den`), leave the word of the global buffer it stands for — when
every member lowers to a stage over the group's grid and no block stores a
word another touches.

The renaming is `compactMap` on the table and, past it, a shift that keeps it
injective (`tabInj`); the members' stages renamed by it are the group's
(`StageSpec.rename`), and a renamed pipeline is the original read through the
renaming (`denote_rename`).
-/

open AlgorithmLib AlgorithmLib.ML AlgorithmLib.ML.BodyWarp AlgorithmLib.ML.BodyCheck AlgorithmLib.ML.BodyPhase
open AlgorithmLib.ML.BodyGroup AlgorithmLib.ML.GroupRun

namespace AlgorithmLib.ML.GroupLink

/-- Renaming agrees wherever the renamings do, on the buffers named. -/
theorem IdxE.renameBuf_congr (f g : Buf → Buf) : ∀ ix : IdxE, (∀ b ∈ ix.bufsOf, f b = g b) →
    ix.renameBuf f = ix.renameBuf g := by
  intro ix
  induction ix with
  | ldIdx b off ih =>
    intro h
    simp only [IdxE.renameBuf, h b (by simp [IdxE.bufsOf]), ih (fun c hc => h c (by simp [IdxE.bufsOf, hc]))]
  | add a b iha ihb =>
    intro h
    simp only [IdxE.renameBuf, iha (fun c hc => h c (by simp [IdxE.bufsOf, hc])),
      ihb (fun c hc => h c (by simp [IdxE.bufsOf, hc]))]
  | mul a b iha ihb =>
    intro h
    simp only [IdxE.renameBuf, iha (fun c hc => h c (by simp [IdxE.bufsOf, hc])),
      ihb (fun c hc => h c (by simp [IdxE.bufsOf, hc]))]
  | _ => intro _; rfl

theorem IExp.renameBuf_congr (f g : Buf → Buf) : ∀ e : IExp, (∀ b ∈ e.bufsOf, f b = g b) →
    e.renameBuf f = e.renameBuf g := by
  intro e
  induction e with
  | ptr b => intro h; simp only [IExp.renameBuf, h b (by simp [IExp.bufsOf])]
  | bin o a b iha ihb =>
    intro h
    simp only [IExp.renameBuf, iha (fun c hc => h c (by simp [IExp.bufsOf, hc])),
      ihb (fun c hc => h c (by simp [IExp.bufsOf, hc]))]
  | cmp o a b iha ihb =>
    intro h
    simp only [IExp.renameBuf, iha (fun c hc => h c (by simp [IExp.bufsOf, hc])),
      ihb (fun c hc => h c (by simp [IExp.bufsOf, hc]))]
  | ctz a ih => intro h; simp only [IExp.renameBuf, ih (fun c hc => h c (by simp [IExp.bufsOf, hc]))]
  | _ => intro _; rfl

theorem EWStmt.renameBuf_congr (f g : Buf → Buf) : ∀ s : EWStmt, (∀ b ∈ s.bufsOf, f b = g b) →
    s.renameBuf f = s.renameBuf g := by
  intro s
  induction s with
  | seq x y ihx ihy =>
    intro h
    simp only [EWStmt.renameBuf, ihx (fun c hc => h c (by simp [EWStmt.bufsOf, hc])),
      ihy (fun c hc => h c (by simp [EWStmt.bufsOf, hc]))]
  | loadIdx d b ix =>
    intro h
    simp only [EWStmt.renameBuf, h b (by simp [EWStmt.bufsOf]),
      IdxE.renameBuf_congr f g ix (fun c hc => h c (by simp [EWStmt.bufsOf, hc]))]
  | loadV4 d0 d1 d2 d3 b ix =>
    intro h
    simp only [EWStmt.renameBuf, h b (by simp [EWStmt.bufsOf]),
      IdxE.renameBuf_congr f g ix (fun c hc => h c (by simp [EWStmt.bufsOf, hc]))]
  | storeLane0 b ix r =>
    intro h
    simp only [EWStmt.renameBuf, h b (by simp [EWStmt.bufsOf]),
      IdxE.renameBuf_congr f g ix (fun c hc => h c (by simp [EWStmt.bufsOf, hc]))]
  | storeLane b ix r =>
    intro h
    simp only [EWStmt.renameBuf, h b (by simp [EWStmt.bufsOf]),
      IdxE.renameBuf_congr f g ix (fun c hc => h c (by simp [EWStmt.bufsOf, hc]))]
  | stSm ix r =>
    intro h
    simp only [EWStmt.renameBuf, IdxE.renameBuf_congr f g ix (fun c hc => h c (by simp [EWStmt.bufsOf, hc]))]
  | ldSm d ix =>
    intro h
    simp only [EWStmt.renameBuf, IdxE.renameBuf_congr f g ix (fun c hc => h c (by simp [EWStmt.bufsOf, hc]))]
  | cvtIF d ix =>
    intro h
    simp only [EWStmt.renameBuf, IdxE.renameBuf_congr f g ix (fun c hc => h c (by simp [EWStmt.bufsOf, hc]))]
  | forN n body ih =>
    intro h
    simp only [EWStmt.renameBuf, ih (fun c hc => h c (by simp [EWStmt.bufsOf, hc]))]
  | forM bu a body ih =>
    intro h
    simp only [EWStmt.renameBuf, h bu (by simp [EWStmt.bufsOf]),
      ih (fun c hc => h c (by simp [EWStmt.bufsOf, hc]))]
  | setI _ e | ldB _ e | ldH _ e | ballot _ e =>
    intro h; simp only [EWStmt.renameBuf, IExp.renameBuf_congr f g e (fun c hc => h c (by simp [EWStmt.bufsOf, hc]))]
  | stB a v g' | stH a v g' =>
    intro h
    simp only [EWStmt.renameBuf, IExp.renameBuf_congr f g a (fun c hc => h c (by simp [EWStmt.bufsOf, hc])),
      IExp.renameBuf_congr f g v (fun c hc => h c (by simp [EWStmt.bufsOf, hc])),
      IExp.renameBuf_congr f g g' (fun c hc => h c (by simp [EWStmt.bufsOf, hc]))]
  | shflIdx _ v s' =>
    intro h
    simp only [EWStmt.renameBuf, IExp.renameBuf_congr f g v (fun c hc => h c (by simp [EWStmt.bufsOf, hc])),
      IExp.renameBuf_congr f g s' (fun c hc => h c (by simp [EWStmt.bufsOf, hc]))]
  | iif c x y ihx ihy =>
    intro h
    simp only [EWStmt.renameBuf, IExp.renameBuf_congr f g c (fun c hc => h c (by simp [EWStmt.bufsOf, hc])),
      ihx (fun c hc => h c (by simp [EWStmt.bufsOf, hc])), ihy (fun c hc => h c (by simp [EWStmt.bufsOf, hc]))]
  | iwhile n c cs body ih =>
    intro h
    simp only [EWStmt.renameBuf, IExp.renameBuf_congr f g c (fun c hc => h c (by simp [EWStmt.bufsOf, hc])),
      ih (fun c hc => h c (by simp [EWStmt.bufsOf, hc]))]
  | locals rs body ih =>
    intro h
    simp only [EWStmt.renameBuf, ih (fun c hc => h c (by simp [EWStmt.bufsOf, hc]))]
  | _ => intro _; rfl

/-- A table's buffer sits at its index. -/
theorem compactMap_get (bs : List Buf) (b : Buf) (h : b ∈ bs) :
    ∃ hl : compactMap bs b < bs.length, bs[compactMap bs b] = b := by
  have hl : bs.findIdx (· == b) < bs.length := List.findIdx_lt_length_of_exists ⟨b, h, by simp⟩
  refine ⟨hl, ?_⟩
  have := List.findIdx_getElem (w := hl)
  simpa using this

/-- **The table's renaming, made injective**: `compactMap` on the table, and a
    buffer past it shifted past every index the table has. -/
def tabInj (bs : List Buf) (b : Buf) : Buf :=
  if b ∈ bs then compactMap bs b else bs.length + 1 + b

theorem tabInj_on (bs : List Buf) (b : Buf) (h : b ∈ bs) : tabInj bs b = compactMap bs b := by
  simp [tabInj, h]

theorem tabInj_inj (bs : List Buf) : BufInj (tabInj bs) := by
  intro b b' h
  unfold tabInj at h
  by_cases hb : b ∈ bs <;> by_cases hb' : b' ∈ bs
  · rw [if_pos hb, if_pos hb'] at h
    obtain ⟨_, e⟩ := compactMap_get bs b hb
    obtain ⟨_, e'⟩ := compactMap_get bs b' hb'
    rw [← e, ← e']; simp only [h]
  · rw [if_pos hb, if_neg hb'] at h
    obtain ⟨hl, _⟩ := compactMap_get bs b hb
    rw [h, Nat.add_assoc] at hl
    exact absurd hl (Nat.not_lt.mpr (Nat.le_add_right _ _))
  · rw [if_neg hb, if_pos hb'] at h
    obtain ⟨hl, _⟩ := compactMap_get bs b' hb'
    rw [← h, Nat.add_assoc] at hl
    exact absurd hl (Nat.not_lt.mpr (Nat.le_add_right _ _))
  · rw [if_neg hb, if_neg hb'] at h
    exact Nat.add_left_cancel h

section Carrier
variable {R : Type} [NumOps R] [NumLit R]

/-- A stage against no integer memory, renamed by an injective `f`. -/
def renZ (f : Buf → Buf) (hf : BufInj f) (S : StageSpec R) (h : S.imem = fun _ _ => 0) : StageSpec R :=
  S.rename f hf (fun _ _ => 0) (fun b _ => by rw [h])

theorem step_renZ (f : Buf → Buf) (hf : BufInj f) (S : StageSpec R) (h : S.imem = fun _ _ => 0)
    (m : Buf → Nat → R) (b : Buf) (a : Nat) :
    (renZ f hf S h).step m (f b) a = S.step (fun b' => m (f b')) b a := by
  by_cases hc : b = S.out ∧ ∃ cta, cta < S.grid ∧ S.dom cta a
  · have hc' : f b = (renZ f hf S h).out ∧ ∃ cta, cta < (renZ f hf S h).grid ∧ (renZ f hf S h).dom cta a :=
      ⟨congrArg f hc.1, hc.2⟩
    rw [StageSpec.step, dif_pos hc', StageSpec.step, dif_pos hc]
    rfl
  · have hc' : ¬ (f b = (renZ f hf S h).out ∧ ∃ cta, cta < (renZ f hf S h).grid ∧ (renZ f hf S h).dom cta a) :=
      fun ⟨h1, h2⟩ => hc ⟨hf _ _ h1, h2⟩
    rw [StageSpec.step, dif_neg hc', StageSpec.step, dif_neg hc]

/-- **A renamed pipeline is the original read through the renaming.** -/
theorem denote_renZ (f : Buf → Buf) (hf : BufInj f) : ∀ (ss : List (StageSpec R))
    (H : ∀ S ∈ ss, S.imem = fun _ _ => 0) (m : Buf → Nat → R) (b : Buf) (a : Nat),
    (Pipeline.mk (ss.pmap (renZ f hf) H)).denote m (f b) a = (Pipeline.mk ss).denote (fun b' => m (f b')) b a := by
  intro ss
  induction ss with
  | nil => intro _ m b a; rfl
  | cons S rest ih =>
    intro H m b a
    show (rest.pmap (renZ f hf) _).foldl (fun mm S => S.step mm) ((renZ f hf S (H S List.mem_cons_self)).step m) (f b) a
      = rest.foldl (fun mm S => S.step mm) (S.step (fun b' => m (f b'))) b a
    have e : (fun b' => (renZ f hf S (H S List.mem_cons_self)).step m (f b')) = S.step (fun b' => m (f b')) :=
      funext fun b' => funext fun a' => step_renZ f hf S _ m b' a'
    have := ih (fun S' h => H S' (List.mem_cons_of_mem _ h)) ((renZ f hf S (H S List.mem_cons_self)).step m) b a
    rw [e] at this
    exact this

theorem foldl_congr_mem {α β : Type} (f g : α → β → α) : ∀ (l : List β) (a : α),
    (∀ x ∈ l, ∀ a, f a x = g a x) → l.foldl f a = l.foldl g a := by
  intro l
  induction l with
  | nil => intro _ _; rfl
  | cons x xs ih =>
    intro a h
    simp only [List.foldl_cons, h x List.mem_cons_self a]
    exact ih _ (fun y hy => h y (List.mem_cons_of_mem _ hy))

/-- A sequence of statements, run, is its parts run one after another. -/
theorem run_foldl_seq (cta : Nat) : ∀ (E : List EWStmt) (acc : EWStmt) (w : WSt R),
    ((E.foldl (fun a e => EWStmt.seq a e) acc).run w cta 0)
      = E.foldl (fun st e => (e.run st cta 0)) ((acc.run w cta 0)) := by
  intro E
  induction E with
  | nil => intro _ _; rfl
  | cons e es ih => intro acc w; exact ih (.seq acc e) w

theorem touchL_foldl_seq (cta : Nat) (l : Lane) (i : Nat) : ∀ (E : List EWStmt) (acc : EWStmt) (x : Buf × Nat),
    ((∃ e ∈ E, x ∈ touchL cta l i e) ∨ x ∈ touchL cta l i acc) →
      x ∈ touchL cta l i (E.foldl (fun a e => EWStmt.seq a e) acc) := by
  intro E
  induction E with
  | nil => intro acc x h; exact h.elim (fun ⟨_, he, _⟩ => absurd he (by simp)) id
  | cons e es ih =>
    intro acc x h
    refine ih (.seq acc e) x ?_
    rcases h with ⟨e', he', hx⟩ | h
    · rcases List.mem_cons.mp he' with rfl | he'
      · exact Or.inr (List.mem_append_right _ hx)
      · exact Or.inl ⟨e', he', hx⟩
    · exact Or.inr (List.mem_append_left _ h)

theorem storeL_foldl_seq (cta : Nat) (l : Lane) (i : Nat) : ∀ (E : List EWStmt) (acc : EWStmt) (x : Buf × Nat),
    ((∃ e ∈ E, x ∈ storeL cta l i e) ∨ x ∈ storeL cta l i acc) →
      x ∈ storeL cta l i (E.foldl (fun a e => EWStmt.seq a e) acc) := by
  intro E
  induction E with
  | nil => intro acc x h; exact h.elim (fun ⟨_, he, _⟩ => absurd he (by simp)) id
  | cons e es ih =>
    intro acc x h
    refine ih (.seq acc e) x ?_
    rcases h with ⟨e', he', hx⟩ | h
    · rcases List.mem_cons.mp he' with rfl | he'
      · exact Or.inr (List.mem_append_right _ hx)
      · exact Or.inl ⟨e', he', hx⟩
    · exact Or.inr (List.mem_append_left _ h)

/-- **A group's members, as stages of the group's own table**: plain,
    exclusive stages over the grid `G` whose statements, in sequence, are the
    group statement, and whose pipeline, read through the table's renaming, is
    the operations' denotations folded in order. -/
theorem group_pipe [LitNat R] (n G : Nat) (ops : List TOp) (bs : List Buf) (xs : List (XStage R))
    (hlow : lowerNet n (forget (ops.map TOp.node)) = some xs)
    (hgrid : ∀ op ∈ ops, op.gridOf = G)
    (hcov : ∀ op ∈ ops, ∀ b ∈ (op.stmt n).bufsOf, b ∈ bs) :
    ∃ rss : List (StageSpec R), (∀ S ∈ rss, Plain S ∧ S.Exclusive ∧ S.grid = G) ∧
      TOp.groupStmt n bs ops = (rss.map fun S => S.ew).foldl (fun a e => EWStmt.seq a e) .skip ∧
      ∀ (m : Buf → Nat → R) (g : Buf) (j : Nat),
        (Pipeline.mk rss).denote m (tabInj bs g) j = ops.foldl (fun m op => op.den m) (fun g' a => m (tabInj bs g') a) g j := by
  obtain ⟨hew, hgr, hpl⟩ := lowerTOps_stages n ops xs hlow
  let f := tabInj bs
  have hf := tabInj_inj bs
  let ss : List (StageSpec R) := xs.map Subtype.val
  have H : ∀ S ∈ ss, S.imem = fun _ _ => 0 := fun S hS => by
    obtain ⟨X, hX, rfl⟩ := List.mem_map.mp hS; exact (hpl X hX).2
  refine ⟨ss.pmap (renZ f hf) H, ?_, ?_, ?_⟩
  · intro S hS
    obtain ⟨S0, hS0, rfl⟩ := List.mem_pmap.mp hS
    obtain ⟨X, hX, rfl⟩ := List.mem_map.mp hS0
    refine ⟨⟨(hpl X hX).1, rfl⟩, StageSpec.rename_exclusive _ _ _ _ _ X.property, ?_⟩
    have hm : X.val.grid ∈ ops.map TOp.gridOf := by rw [← hgr]; exact List.mem_map.mpr ⟨X, hX, rfl⟩
    obtain ⟨op, hop, he⟩ := List.mem_map.mp hm
    show X.val.grid = G
    rw [← he]; exact hgrid op hop
  · have hE : ops.map (fun op => (op.stmt n).renameBuf (compactMap bs)) = (ss.pmap (renZ f hf) H).map (fun S => S.ew) := by
      simp only [List.map_pmap]
      have hr : ∀ (S : StageSpec R) (h : S.imem = fun _ _ => 0), (renZ f hf S h).ew = S.ew.renameBuf f :=
        fun _ _ => rfl
      simp only [hr]
      rw [List.pmap_eq_map]
      simp only [ss, List.map_map]
      have : (xs.map fun X => X.val.ew.renameBuf f) = (xs.map fun X => X.val.ew).map (EWStmt.renameBuf f) := by
        simp [List.map_map, Function.comp]
      rw [show ((fun S : StageSpec R => S.ew.renameBuf f) ∘ Subtype.val) = fun X : XStage R => X.val.ew.renameBuf f
        from rfl, this, hew, List.map_map]
      refine List.map_congr_left (fun op hop => ?_)
      exact (EWStmt.renameBuf_congr _ _ _ (fun b hb => tabInj_on bs b (hcov op hop b hb))).symm
    rw [← hE, TOp.groupStmt, List.foldl_map]
  · intro m g j
    rw [denote_renZ f hf ss H m g j]
    show ss.foldl (fun mm S => S.step mm) _ g j = _
    rw [List.foldl_map, lowerTOps_den n ops xs hlow]

theorem run_stages (rss : List (StageSpec R)) (hp : ∀ S ∈ rss, Plain S) (w : WSt R) (c : Nat) :
    ((((rss.map fun S => S.ew).foldl (fun a e => EWStmt.seq a e) .skip)).run w c 0)
      = rss.foldl (fun st S => S.blk c st) w := by
  rw [run_foldl_seq, List.foldl_map]
  refine foldl_congr_mem _ _ rss _ (fun S hS st => ?_)
  rw [(hp S hS).blk c]

/-- **A group's kernel computes its members' denotations.**  Operations `ops`
    over the batch `n`, lowering to stages over the grid `G`, every buffer
    they name in the table `bs`, and no block of the group statement storing a
    word another touches: block `c`'s run, from any state, leaves each word it
    touches of the slot standing for global buffer `g` as the operations'
    denotations, folded in order from the memory the slots stand for, leave
    `g`'s word. -/
theorem group_den [LitNat R] (n G : Nat) (ops : List TOp) (bs : List Buf) (xs : List (XStage R))
    (hlow : lowerNet n (forget (ops.map TOp.node)) = some xs)
    (hgrid : ∀ op ∈ ops, op.gridOf = G)
    (hcov : ∀ op ∈ ops, ∀ b ∈ (op.stmt n).bufsOf, b ∈ bs)
    (hap : ∀ c c' (l l' : Lane) b j, c < G → c' < G → c ≠ c' →
      (b, j) ∈ touchL c l 0 (TOp.groupStmt n bs ops) → (b, j) ∈ storeL c' l' 0 (TOp.groupStmt n bs ops) → False)
    (w : WSt R) (c : Nat) (hc : c < G) (l : Lane) (g : Buf) (j : Nat)
    (ht : (tabInj bs g, j) ∈ touchL c l 0 (TOp.groupStmt n bs ops)) :
    (((TOp.groupStmt n bs ops).run w c 0)).mem (tabInj bs g) j
      = ops.foldl (fun m op => op.den m) (fun g' a => w.mem (tabInj bs g') a) g j := by
  obtain ⟨rss, hS, hgs, hden⟩ := group_pipe n G ops bs xs hlow hgrid hcov
  let T : Nat → Buf → Nat → Prop := fun c b j => ∃ l : Lane, (b, j) ∈ touchL c l 0 (TOp.groupStmt n bs ops)
  have hTS : ∀ S ∈ rss, ∀ c (l : Lane) b j, (b, j) ∈ touchL c l 0 S.ew → T c b j := by
    intro S hS c l b j h
    refine ⟨l, ?_⟩
    rw [hgs]
    exact touchL_foldl_seq c l 0 _ _ _ (Or.inl ⟨S.ew, List.mem_map.mpr ⟨S, hS, rfl⟩, h⟩)
  have hAp : ∀ S ∈ rss, ∀ c c', c < G → c' < G → c ≠ c' → ∀ (l : Lane) b j,
      (b, j) ∈ storeL c' l 0 S.ew → ¬ T c b j := by
    intro S hS c c' hc hc' hne l b j h ⟨l0, ht⟩
    refine hap c c' l0 l b j hc hc' hne ht ?_
    rw [hgs]
    exact storeL_foldl_seq c' l 0 _ _ _ (Or.inl ⟨S.ew, List.mem_map.mpr ⟨S, hS, rfl⟩, h⟩)
  have key := group_run G T rss hS hTS hAp (fun _ => w) w.mem (fun _ _ _ _ _ => rfl) c hc (tabInj bs g) j ⟨l, ht⟩
  rw [hgs, run_stages rss (fun S h => (hS S h).1), key, hden]

/-- **…and leaves every word no block of it stores**: there the operations'
    denotations leave the word the slot held. -/
theorem group_keep [LitNat R] (n G : Nat) (ops : List TOp) (bs : List Buf) (xs : List (XStage R))
    (hlow : lowerNet n (forget (ops.map TOp.node)) = some xs)
    (hgrid : ∀ op ∈ ops, op.gridOf = G)
    (hcov : ∀ op ∈ ops, ∀ b ∈ (op.stmt n).bufsOf, b ∈ bs)
    (m : Buf → Nat → R) (g : Buf) (j : Nat)
    (hn : ∀ c, c < G → ∀ l : Lane, (tabInj bs g, j) ∉ storeL c l 0 (TOp.groupStmt n bs ops)) :
    ops.foldl (fun m op => op.den m) (fun g' a => m (tabInj bs g') a) g j = m (tabInj bs g) j := by
  obtain ⟨rss, hS, hgs, hden⟩ := group_pipe n G ops bs xs hlow hgrid hcov
  rw [← hden]
  refine denote_keep rss (fun S h => (hS S h).1) _ j (fun S hS' c hc l h => hn c ((hS S hS').2.2 ▸ hc) l ?_) m
  rw [hgs]
  exact storeL_foldl_seq c l 0 _ _ _ (Or.inl ⟨S.ew, List.mem_map.mpr ⟨S, hS', rfl⟩, h⟩)

/-- **A group's operations at a word read only what its statement touches**:
    from two slot memories that agree on every word a block of the group
    touches, and on the word itself, the operations' denotations, folded in
    order, agree there. -/
theorem group_local [LitNat R] (n G : Nat) (ops : List TOp) (bs : List Buf) (xs : List (XStage R))
    (hlow : lowerNet n (forget (ops.map TOp.node)) = some xs)
    (hgrid : ∀ op ∈ ops, op.gridOf = G)
    (hcov : ∀ op ∈ ops, ∀ b ∈ (op.stmt n).bufsOf, b ∈ bs)
    (hap : ∀ c c' (l l' : Lane) b j, c < G → c' < G → c ≠ c' →
      (b, j) ∈ touchL c l 0 (TOp.groupStmt n bs ops) → (b, j) ∈ storeL c' l' 0 (TOp.groupStmt n bs ops) → False)
    (M M' : Buf → Nat → R) (g : Buf) (j : Nat)
    (hag : ∀ c, c < G → ∀ (l : Lane) b a, (b, a) ∈ touchL c l 0 (TOp.groupStmt n bs ops) → M b a = M' b a)
    (hat : M (tabInj bs g) j = M' (tabInj bs g) j) :
    ops.foldl (fun m op => op.den m) (fun g' a => M (tabInj bs g') a) g j
      = ops.foldl (fun m op => op.den m) (fun g' a => M' (tabInj bs g') a) g j := by
  by_cases ht : ∃ c, c < G ∧ ∃ l : Lane, (tabInj bs g, j) ∈ touchL c l 0 (TOp.groupStmt n bs ops)
  · obtain ⟨c, hc, l, ht⟩ := ht
    let w : WSt R := ⟨fun _ _ => NumOps.zero, M, fun _ => NumOps.zero, {}⟩
    let w' : WSt R := ⟨fun _ _ => NumOps.zero, M', fun _ => NumOps.zero, {}⟩
    rw [← group_den n G ops bs xs hlow hgrid hcov hap w c hc l g j ht,
      ← group_den n G ops bs xs hlow hgrid hcov hap w' c hc l g j ht]
    exact (run_agree c _ 0 w w' ⟨rfl, rfl, fun b a ⟨l', h⟩ => hag c hc l' b a h, rfl⟩).2.2.1 _ _ ⟨l, ht⟩
  · have hn : ∀ c, c < G → ∀ l : Lane, (tabInj bs g, j) ∉ storeL c l 0 (TOp.groupStmt n bs ops) :=
      fun c hc l hs => ht ⟨c, hc, l, storeL_sub c l _ 0 hs⟩
    rw [group_keep n G ops bs xs hlow hgrid hcov M g j hn, group_keep n G ops bs xs hlow hgrid hcov M' g j hn]
    exact hat

end Carrier

end AlgorithmLib.ML.GroupLink
