module
public import AlgorithmLib.ML.Ptx.BodyWarp
meta import AlgorithmLib.ML.Ptx.BodyWarp
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `ML.Ptx.BodyPhase` — a statement run in phases

A statement split into phases (`Split`: a sequence's phases are its parts'):
one lane's run of it is its runs of the phases one after another
(`runL_split`), and so is the warp machine's (`run_split`); what it stores, one
of its phases stores (`storesL_split`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec AlgorithmLib.Kernel AlgorithmLib.ML AlgorithmLib.ML.BodyProof
open AlgorithmLib.ML.BodyWarp

namespace AlgorithmLib.ML.BodyPhase

/-- **Phases of a statement**: a statement is its own one phase, and a
    sequence's phases are its first part's, then its second's. -/
inductive Split : EWStmt → List EWStmt → Prop
  | one (s : EWStmt) : Split s [s]
  | seq {a b : EWStmt} {pa pb : List EWStmt} : Split a pa → Split b pb → Split (.seq a b) (pa ++ pb)

theorem runL_split (hw : Hw) (nbuf cta l : Nat) (o : Nat → UInt32) (rd : Buf → Nat → Option Nat) (i : Nat) :
    ∀ {s : EWStmt} {ps : List EWStmt}, Split s ps → ∀ st,
      runL hw nbuf cta l o rd i s st = ps.foldlM (m := Option) (fun st p => runL hw nbuf cta l o rd i p st) st := by
  intro s ps h
  induction h with
  | one s => intro st; simp only [List.foldlM_cons, List.foldlM_nil, bind, Option.bind]; cases runL hw nbuf cta l o rd i s st <;> rfl
  | seq ha hb iha ihb =>
    intro st
    rw [List.foldlM_append]
    simp only [runL, bind]
    rw [iha st]
    congr 1
    funext x
    exact ihb x

theorem run_split {R : Type} [NumOps R] [NumLit R] (cta i : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) :
    ∀ {s : EWStmt} {ps : List EWStmt}, Split s ps → ∀ (ws : WSt R),
      (s.elabAt cta i ir im).run ws = ps.foldl (fun ws p => (p.elabAt cta i ir im).run ws) ws := by
  intro s ps h
  induction h with
  | one s => intro ws; rfl
  | seq ha hb iha ihb =>
    intro ws
    rw [List.foldl_append, ← iha ws, ← ihb]
    rfl

theorem storesL_split (cta : Nat) (im : Buf → Nat → Nat) (l : Lane) (i : Nat) :
    ∀ {s : EWStmt} {ps : List EWStmt}, Split s ps → ∀ b j,
      StoresL cta im l i s b j ↔ ∃ p ∈ ps, StoresL cta im l i p b j := by
  intro s ps h
  induction h with
  | one s => intro b j; simp
  | seq ha hb iha ihb =>
    intro b j
    simp only [StoresL, iha b j, ihb b j, List.mem_append]
    constructor
    · rintro (⟨p, hp, h⟩ | ⟨p, hp, h⟩)
      · exact ⟨p, Or.inl hp, h⟩
      · exact ⟨p, Or.inr hp, h⟩
    · rintro ⟨p, hp | hp, h⟩
      · exact Or.inl ⟨p, hp, h⟩
      · exact Or.inr ⟨p, hp, h⟩

/-- **The warp machine leaves alone every word no lane stores.** -/
theorem run_frame_st {R : Type} [NumOps R] [NumLit R] (cta : Nat) (im : Buf → Nat → Nat) :
    ∀ (s : EWStmt) (i : Nat) (ws : WSt R) (b : Buf) (j : Nat), (∀ l, ¬ StoresL cta im l i s b j) →
      ((s.elabAt cta i (fun _ _ => 0) im).run ws).mem b j = ws.mem b j := by
  have loop : ∀ (body : EWStmt) (L : List Nat) (ws : WSt R) (b : Buf) (j : Nat),
      (∀ t ∈ L, ∀ (ws : WSt R), ((body.elabAt cta t (fun _ _ => 0) im).run ws).mem b j = ws.mem b j) →
      (L.foldl (fun s' t => (body.elabAt cta t (fun _ _ => 0) im).run s') ws).mem b j = ws.mem b j := by
    intro body L
    induction L with
    | nil => intros; rfl
    | cons t ts ih =>
      intro ws b j h
      rw [List.foldl_cons, ih _ b j (fun t' ht' => h t' (List.mem_cons_of_mem _ ht')), h t List.mem_cons_self]
  intro s
  induction s with
  | storeLane b0 ix n =>
    intro i ws b j h
    exact storeFold_other b0 (fun l => ix.eval cta i l (fun _ _ => 0) im) _ b j _ ws
      (fun l _ hb => h l ⟨hb.1, hb.2⟩)
  | storeLane0 b0 ix n =>
    intro i ws b j h
    show (if b = b0 ∧ j = ix.eval cta i ⟨0, by decide⟩ (fun _ _ => 0) im then _ else _) = _
    rw [if_neg (fun hb => h lane0 ⟨rfl, hb.1, hb.2⟩)]
  | seq x y ihx ihy =>
    intro i ws b j h
    show ((y.elabAt cta i (fun _ _ => 0) im).run ((x.elabAt cta i (fun _ _ => 0) im).run ws)).mem b j = _
    rw [ihy i _ b j (fun l hl => h l (Or.inr hl)), ihx i ws b j (fun l hl => h l (Or.inl hl))]
  | forN n body ih =>
    intro i ws b j h
    exact loop body _ ws b j (fun t ht ws' => ih t ws' b j (fun l hl => h l ⟨t, List.mem_range.mp ht, hl⟩))
  | forM bb a body ih =>
    intro i ws b j h
    exact loop body _ ws b j (fun t ht ws' => ih t ws' b j (fun l hl => h l ⟨t, List.mem_range.mp ht, hl⟩))
  | _ => intros; rfl

theorem foldlM_take {α β : Type} (f : β → α → Option β) :
    ∀ (L : List α) (x y : β), L.foldlM f x = some y → ∀ q, q ≤ L.length → ∃ z, (L.take q).foldlM f x = some z := by
  intro L
  induction L with
  | nil => intro x y _ q _; exact ⟨x, by simp⟩
  | cons a as ih =>
    intro x y h q hq
    cases q with
    | zero => exact ⟨x, rfl⟩
    | succ q =>
      simp only [List.foldlM_cons, bind, Option.bind] at h
      split at h; · cases h
      rename_i x1 hx1
      obtain ⟨z, hz⟩ := ih x1 y h q (by simp at hq; omega)
      exact ⟨z, by simp only [List.take_succ_cons, List.foldlM_cons, bind, Option.bind, hx1, hz]⟩

theorem foldlM_take_succ {α β : Type} (f : β → α → Option β) (L : List α) (x : β) (q : Nat) (hq : q < L.length) :
    (L.take (q + 1)).foldlM f x = ((L.take q).foldlM f x).bind (fun z => f z L[q]) := by
  rw [List.take_succ, List.foldlM_append, List.getElem?_eq_getElem hq]
  simp only [Option.toList_some, List.foldlM_cons, List.foldlM_nil, bind]
  cases (L.take q).foldlM f x with
  | none => rfl
  | some z => simp only [Option.bind]; cases f z L[q] <;> rfl

theorem foldl_take_succ {α β : Type} (f : β → α → β) (L : List α) (x : β) (q : Nat) (hq : q < L.length) :
    (L.take (q + 1)).foldl f x = f ((L.take q).foldl f x) L[q] := by
  rw [List.take_succ, List.foldl_append, List.getElem?_eq_getElem hq]
  rfl

/-- **A block runs its statement in phases**: a block of `nw` warps of a
    statement's kernel, the statement split into phases (`Split`), each phase's
    accesses those its map allows, leaves every word a lane of warp `w` touches
    in any phase as the warp machine's run of the statement for block
    `cta·nw + w` does from `m0`, where each word is stored in one phase by one
    thread, and any other thread touches it only in a later phase, after that
    thread in the block's pass and in its warp. -/
theorem block_phases (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (name : String) (nbuf smem : Nat)
    (s : EWStmt) (ps : List EWStmt) (hsp : Split s ps)
    (nw : Nat) (hnw : 0 < nw) (hnw31 : nw < 2 ^ 31) (cta : Nat) (hcta : cta * nw + nw ≤ 2 ^ 32)
    (hN : 32 * nw ≤ 2 ^ 32) (A : Nat → Nat → Acc) (rd : Buf → Nat → Option Nat)
    (hacc : ∀ wi q, wi < nw → q < ps.length → AccOk (cta * nw + wi) (A q (cta * nw + wi)) rd 0 (ps.getD q .skip))
    (hA : ∀ q c l b j, A q c l b j → b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32)
    (hone : ∀ t t' q q' b j, t < 32 * nw → t' < 32 * nw → q < ps.length → q' < ps.length →
      StoresL (cta * nw + t / 32) (imOf rd) ⟨t % 32, Nat.mod_lt _ (by decide)⟩ 0 (ps.getD q .skip) b j →
      StoresL (cta * nw + t' / 32) (imOf rd) ⟨t' % 32, Nat.mod_lt _ (by decide)⟩ 0 (ps.getD q' .skip) b j →
      t = t' ∧ q = q')
    (hlate : ∀ t o q p b j, t < 32 * nw → o < 32 * nw → q < ps.length → p < ps.length →
      A q (cta * nw + t / 32) ⟨t % 32, Nat.mod_lt _ (by decide)⟩ b j →
      StoresL (cta * nw + o / 32) (imOf rd) ⟨o % 32, Nat.mod_lt _ (by decide)⟩ 0 (ps.getD p .skip) b j →
      t ≠ o → p < q ∧ o < t ∧ o / 32 = t / 32)
    (hfine : Fine (emitGo (fnBody (params nbuf) (Body.body nbuf s nw)) ⟨(params nbuf).length, 0, [], [], [], none⟩).2)
    (w w' : World) (hrd : RdOk nbuf w.mem rd) (hnan : w.mem.nan = hw.nan)
    (hok : ∀ t, t < 32 * nw → ∃ st', runL hw nbuf (cta * nw + t / 32) (t % 32) (fun _ => 0) rd 0 s
      ⟨fun _ => ⟨0⟩, rm (ByteArray.mk (Array.replicate smem 0)) w.mem, []⟩ = some st')
    (m0 : Mem) (hin : ∀ q wi (l : Lane) b j, q < ps.length → wi < nw → A q (cta * nw + wi) l b j →
      w.mem.load (bufW nbuf b j) 4 = m0.load (bufW nbuf b j) 4 ∧ (m0.load (bufW nbuf b j) 4).isSome)
    (hr : BlockRuns (Body.kernel name nbuf smem s nw) cta (32 * nw) (bufArgs nbuf) hw w w') :
    (∀ q wi (l : Lane) b j, q < ps.length → wi < nw → A q (cta * nw + wi) l b j →
      w'.mem.load (bufW nbuf b j) 4 =
        some ((@WStmt.run F32 (F32.ops hw) _ (s.elabAt (cta * nw + wi) 0 (fun _ _ => 0) (imOf rd))
          (ws0 nbuf m0)).mem b j).bits) ∧
    (∀ b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) →
      (∀ wi l, wi < nw → ¬ StoresL (cta * nw + wi) (imOf rd) l 0 s b j) →
      w'.mem.load (bufW nbuf b j) 4 = w.mem.load (bufW nbuf b j) 4) ∧
    MemSame w.mem w'.mem ∧ RdOk nbuf w'.mem rd ∧ w'.mem.nan = hw.nan := by
  obtain ⟨steps, ans, w'', calls, stores, ans', hfold, hans, hset, rfl⟩ := hr
  let N := 32 * nw
  let K := ps.length
  let ph (q : Nat) : EWStmt := ps.getD q .skip
  let A0 := ByteArray.mk (Array.replicate smem 0)
  let W0 := rm A0 w.mem
  let vc (t : Nat) : Nat := cta * nw + t / 32
  let ln (t : Nat) : Lane := ⟨t % 32, Nat.mod_lt _ (by decide)⟩
  let St (t q : Nat) (b : Buf) (j : Nat) : Prop := StoresL (vc t) (imOf rd) (ln t) 0 (ph q) b j
  have hvc : ∀ t, t < N → vc t < 2 ^ 32 := fun t ht => by
    have : t / 32 < nw := by omega
    show cta * nw + t / 32 < 2 ^ 32; omega
  have hWd : ∀ t, t < N → Words (ans t) := fun t ht => by
    rw [← hset t ht]; exact answersOf_words (answers_at hans ht)
  let o (t : Nat) : Nat → UInt32 := wordAt (ans t)
  let start (m : Mem) : LSt := ⟨fun _ => ⟨0⟩, rm A0 m, []⟩
  let f (t : Nat) (m : Mem) : Mem :=
    if t < N then ((runL hw nbuf (vc t) (t % 32) (o t) rd 0 s (start m)).map LSt.mem).getD m else m
  let P (t : Nat) (m : Mem) : Prop := t < N ∧ MemSame (rm A0 m) W0 ∧ RdOk nbuf m rd ∧ m.nan = hw.nan
  let C (t : Nat) (m : Mem) (cs : List (Intr × List V)) (_ : List (Nat × UInt64 × Nat × UInt64)) : Prop :=
    ∃ st', runL hw nbuf (vc t) (t % 32) (o t) rd 0 s (start m) = some st' ∧ cs = st'.calls
  have hsucc : ∀ t m, P t m → ∃ st', runL hw nbuf (vc t) (t % 32) (o t) rd 0 s (start m) = some st' := by
    intro t m hP
    obtain ⟨st0, h0⟩ := hok t hP.1
    exact runL_transfer hw nbuf (vc t) (t % 32) _ (o t) rd s 0 _ st0 (start m) h0 hP.2.1.symm
  have hfrd : ∀ t m, RdOk nbuf m rd → RdOk nbuf (f t m) rd := by
    intro t m hm
    simp only [f]
    split
    · cases hrun : runL hw nbuf (vc t) (t % 32) (o t) rd 0 s (start m) with
      | none => exact hm
      | some st' => exact runL_rdOk s 0 _ _ hrun ((rdOk_arena A0).mpr hm)
    · exact hm
  have hfS : ∀ t m, RdOk nbuf m rd → ∀ b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) →
      ¬ StoresL (vc t) (imOf rd) (ln t) 0 s b j → (f t m).load (bufW nbuf b j) 4 = m.load (bufW nbuf b j) 4 := by
    intro t m hm b j hv hn
    simp only [f]
    split
    · rename_i ht
      cases hrun : runL hw nbuf (vc t) (t % 32) (o t) rd 0 s (start m) with
      | none => rfl
      | some st' =>
        simp only [Option.map_some, Option.getD_some]
        rw [runL_frame_st hw nbuf (vc t) rd (ln t) (o t) s 0 (start m) st' hrun ((rdOk_arena A0).mpr hm) b j hv hn]
        exact load_buf_arena m A0 (decode_bufW hv.1 hv.2.1 hv.2.2) 4
    · rfl
  have hfnan : ∀ t m, RdOk nbuf m rd → m.nan = hw.nan → (f t m).nan = hw.nan := by
    intro t m hm hn
    simp only [f]
    split
    · cases hrun : runL hw nbuf (vc t) (t % 32) (o t) rd 0 s (start m) with
      | none => exact hn
      | some st' => exact (runL_nan s 0 _ _ hrun ((rdOk_arena A0).mpr hm)).trans hn
    · exact hn
  have hkeep : ∀ t u m, t ≠ u → P u m → P u (f t m) := by
    intro t u m _ ⟨hu, hs, hrm, hn⟩
    refine ⟨hu, ?_, hfrd t m hrm, hfnan t m hrm hn⟩
    simp only [f]
    split
    · cases hrun : runL hw nbuf (vc t) (t % 32) (o t) rd 0 s (start m) with
      | none => exact hs
      | some st' =>
        simp only [Option.map_some, Option.getD_some]
        have h1 := runL_same hw nbuf (vc t) (t % 32) (o t) rd s 0 (start m) st' hrun
        exact ((memSame_rm h1 A0).symm.trans (by simpa [rm] using hs))
    · exact hs
  have hP0 : ∀ t, t < N → P t W0 := fun t ht =>
    ⟨ht, by simpa [W0, rm] using MemSame.refl W0, (rdOk_arena A0).mpr hrd, hnan⟩
  obtain ⟨hmem, hC⟩ := pass_spec P f C (fun t sw u hP hu => by
      obtain ⟨st', hst'⟩ := hsucc t sw.mem hP
      have h := thread_body hw hw32 name nbuf smem s nw hnw hnw31 cta t (32 * nw) hP.1 (by omega) (hvc t hP.1)
        _ u (ans t) (hWd t hP.1) st' hfine rd (by exact (rdOk_arena A0).mpr hP.2.2.1) (by exact hP.2.2.2) hst' hu
      refine ⟨?_, st', hst', h.2⟩
      rw [h.1]
      simp only [f, if_pos hP.1, hst', Option.map_some, Option.getD_some])
    hkeep (List.range N) _ _ List.nodup_range (fun t ht => hP0 t (List.mem_range.mp ht)) hfold
  let M (t : Nat) : Mem := (List.range t).foldl (fun m t => f t m) W0
  have hMS : ∀ t, f t (M t) = M (t + 1) := fun t => by
    show f t (M t) = (List.range (t + 1)).foldl (fun m t => f t m) W0
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
  have hPM : ∀ t, t < N → ∀ t', t' ≤ t → P t (M t') := by
    intro t ht t' ht'
    induction t' with
    | zero => exact hP0 t ht
    | succ t' ih => rw [← hMS]; exact hkeep t' t _ (by omega) (ih (by omega))
  have hMrd : ∀ t, RdOk nbuf (M t) rd := by
    intro t
    induction t with
    | zero => exact (rdOk_arena A0).mpr hrd
    | succ t ih => rw [← hMS]; exact hfrd t _ ih
  have hMnan : ∀ t, (M t).nan = hw.nan := by
    intro t
    induction t with
    | zero => exact hnan
    | succ t ih => rw [← hMS]; exact hfnan t _ (hMrd t) ih
  have hCl : ∀ t, t < N → ∃ st', runL hw nbuf (vc t) (t % 32) (o t) rd 0 s (start (M t)) = some st' ∧
      calls t = st'.calls := by
    intro t ht
    have h := hC t (List.mem_range.mpr ht)
    rw [takeWhile_range N t ht] at h
    exact h
  let tr (t : Nat) : LSt := (runL hw nbuf (vc t) (t % 32) (o t) rd 0 s (start (M t))).getD (start (M t))
  have hruns : ∀ t, t < N → runL hw nbuf (vc t) (t % 32) (o t) rd 0 s (start (M t)) = some (tr t) := by
    intro t ht; obtain ⟨st', h, _⟩ := hCl t ht; simp only [tr, h, Option.getD_some]
  have hcalls : ∀ t, t < N → calls t = (tr t).calls := by
    intro t ht; obtain ⟨st', h, hc⟩ := hCl t ht; simp only [tr, h, Option.getD_some]; exact hc
  have hMtr : ∀ t, t < N → (M (t + 1)).load = (tr t).mem.load := fun t ht => by
    rw [← hMS]; simp only [f, if_pos ht, hruns t ht, Option.map_some, Option.getD_some]
  -- a thread's stores in its run are its stores in some phase
  have hsto : ∀ t b j, StoresL (vc t) (imOf rd) (ln t) 0 s b j → ∃ q, q < K ∧ St t q b j := by
    intro t b j h
    obtain ⟨p, hp, hs⟩ := (storesL_split (vc t) (imOf rd) (ln t) 0 hsp b j).mp h
    obtain ⟨q, hq, rfl⟩ := List.getElem_of_mem hp
    exact ⟨q, hq, by simp only [St, ph, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hq, Option.getD_some]; exact hs⟩
  -- F1: a word no earlier thread stores is as the block found it
  have hF1 : ∀ t, t ≤ N → ∀ b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) →
      (∀ t', t' < t → ¬ StoresL (vc t') (imOf rd) (ln t') 0 s b j) →
      (M t).load (bufW nbuf b j) 4 = W0.load (bufW nbuf b j) 4 := by
    intro t
    induction t with
    | zero => intros; rfl
    | succ t ih =>
      intro ht b j hv hn
      rw [← hMS, hfS t _ (hMrd t) b j hv (hn t (by omega)), ih (by omega) b j hv (fun t' ht' => hn t' (by omega))]
  -- F2: past its one storing thread, a word is as that thread left it
  have hF2 : ∀ o' t, o' < t → t ≤ N → ∀ b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) →
      (∀ t', o' < t' → t' < t → ¬ StoresL (vc t') (imOf rd) (ln t') 0 s b j) →
      (M t).load (bufW nbuf b j) 4 = (M (o' + 1)).load (bufW nbuf b j) 4 := by
    intro o' t hot
    induction t with
    | zero => omega
    | succ t ih =>
      intro ht b j hv hn
      by_cases he : t = o'
      · subst he; rfl
      · rw [← hMS, hfS t _ (hMrd t) b j hv (hn t (by omega) (by omega)),
          ih (by omega) (by omega) b j hv (fun t' h1 h2 => hn t' h1 (by omega))]
  -- the phases a thread runs
  let runP (t : Nat) (L : List EWStmt) (st : LSt) : Option LSt :=
    L.foldlM (m := Option) (fun st p => runL hw nbuf (vc t) (t % 32) (o t) rd 0 p st) st
  have hfull : ∀ t, t < N → runP t ps (start (M t)) = some (tr t) := fun t ht =>
    (runL_split hw nbuf (vc t) (t % 32) (o t) rd 0 hsp _).symm.trans (hruns t ht)
  let tps (t q : Nat) : LSt := (runP t (ps.take q) (start (M t))).getD (start (M t))
  have htpsE : ∀ t q, t < N → q ≤ K → runP t (ps.take q) (start (M t)) = some (tps t q) := fun t q ht hq => by
    obtain ⟨z, hz⟩ := foldlM_take _ ps _ _ (hfull t ht) q hq
    simp only [tps, runP] at hz ⊢; rw [hz]; rfl
  have htpsS : ∀ t q, t < N → q < K →
      runL hw nbuf (vc t) (t % 32) (o t) rd 0 (ph q) (tps t q) = some (tps t (q + 1)) := fun t q ht hq => by
    have h1 := htpsE t (q + 1) ht hq
    simp only [runP] at h1
    rw [foldlM_take_succ _ ps _ q hq] at h1
    have h0 := htpsE t q ht (by omega)
    simp only [runP] at h0
    rw [h0] at h1
    simp only [Option.bind] at h1
    have hpq : ph q = ps[q] := by simp [ph, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hq]
    rw [hpq]; exact h1
  have htps0 : ∀ t, tps t 0 = start (M t) := fun t => rfl
  have htpsK : ∀ t, t < N → tps t K = tr t := fun t ht => by
    have h := htpsE t K ht (Nat.le_refl _)
    rw [List.take_length, hfull t ht] at h
    exact (Option.some.inj h).symm
  have htrd : ∀ t q, t < N → q ≤ K → RdOk nbuf (tps t q).mem rd := by
    intro t q ht hq
    induction q with
    | zero => exact (rdOk_arena A0).mpr (hMrd t)
    | succ q ih => exact runL_rdOk (ph q) 0 _ _ (htpsS t q ht (by omega)) (ih (by omega))
  -- within a thread, a phase that does not store a word leaves it alone
  have htf : ∀ t q q', t < N → q ≤ q' → q' ≤ K → ∀ b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) →
      (∀ r, q ≤ r → r < q' → ¬ St t r b j) →
      (tps t q').mem.load (bufW nbuf b j) 4 = (tps t q).mem.load (bufW nbuf b j) 4 := by
    intro t q q' ht hqq hq'
    induction q' with
    | zero =>
      intro b j _ _
      have : q = 0 := by omega
      subst this; rfl
    | succ q' ih =>
      intro b j hv hn
      by_cases hq : q = q' + 1
      · subst hq; rfl
      · rw [runL_frame_st hw nbuf (vc t) rd (ln t) (o t) (ph q') 0 _ _ (htpsS t q' ht (by omega))
          (htrd t q' ht (by omega)) b j hv (hn q' (by omega) (by omega)),
          ih (by omega) (by omega) b j hv (fun r h1 h2 => hn r h1 (by omega))]
  -- the warp machine, phase by phase
  let wps (wi q : Nat) : WSt F32 := (ps.take q).foldl
    (fun ws p => @WStmt.run F32 (F32.ops hw) _ (p.elabAt (cta * nw + wi) 0 (fun _ _ => 0) (imOf rd)) ws) (ws0 nbuf m0)
  have hwpsS : ∀ wi q, q < K → wps wi (q + 1) =
      @WStmt.run F32 (F32.ops hw) _ ((ph q).elabAt (cta * nw + wi) 0 (fun _ _ => 0) (imOf rd)) (wps wi q) := by
    intro wi q hq
    have hpq : ph q = ps[q] := by simp [ph, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hq]
    rw [hpq]
    exact foldl_take_succ _ ps _ q hq
  have hwpsK : ∀ wi, wps wi K =
      @WStmt.run F32 (F32.ops hw) _ (s.elabAt (cta * nw + wi) 0 (fun _ _ => 0) (imOf rd)) (ws0 nbuf m0) := by
    intro wi
    show (ps.take K).foldl _ _ = _
    rw [List.take_length, @run_split F32 (F32.ops hw) _ (cta * nw + wi) 0 (fun _ _ => 0) (imOf rd) _ _ hsp]
  have hwf : ∀ wi q q', q ≤ q' → q' ≤ K → ∀ b j,
      (∀ r (l : Lane), q ≤ r → r < q' → ¬ StoresL (cta * nw + wi) (imOf rd) l 0 (ph r) b j) →
      (wps wi q').mem b j = (wps wi q).mem b j := by
    intro wi q q' hqq hq'
    induction q' with
    | zero =>
      intro b j _
      have : q = 0 := by omega
      subst this; rfl
    | succ q' ih =>
      intro b j hn
      by_cases hq : q = q' + 1
      · subst hq; rfl
      · rw [hwpsS wi q' (by omega), @run_frame_st F32 (F32.ops hw) _ (cta * nw + wi) (imOf rd) (ph q') 0 _ b j
          (fun l => hn q' l (by omega) (by omega)), ih (by omega) (by omega) b j (fun r l h1 h2 => hn r l h1 (by omega))]
  -- lanes of warp `wi`
  have hT : ∀ wi, wi < nw → ∀ l : Lane, 32 * wi + l.val < N := fun wi hwi l => by
    have : l.val < 32 := l.isLt; show 32 * wi + l.val < 32 * nw; omega
  have hvcl : ∀ wi (l : Lane), vc (32 * wi + l.val) = cta * nw + wi := fun wi l => by
    have : l.val < 32 := l.isLt; show cta * nw + (32 * wi + l.val) / 32 = cta * nw + wi; omega
  have hlnl : ∀ wi (l : Lane), ln (32 * wi + l.val) = l := fun wi l =>
    Fin.ext (by have : l.val < 32 := l.isLt; show (32 * wi + l.val) % 32 = l.val; omega)
  have hlnv : ∀ wi (l : Lane), (32 * wi + l.val) % 32 = l.val := fun wi l => by
    have : l.val < 32 := l.isLt; omega
  have hr32 : ∀ t q, t < N → q ≤ K → Regs32 (tps t q).reg := by
    intro t q ht hq
    induction q with
    | zero => exact fun _ => by show (0 : UInt64).toNat < 2 ^ 32; decide
    | succ q ih => exact runL_32 hw hw32 nbuf _ _ _ _ (ph q) 0 _ _ (htpsS t q ht (by omega)) (ih (by omega))
  have hcl : ∀ t t' q, t < N → t' < N → q ≤ K → (tps t q).calls.length = (tps t' q).calls.length := by
    intro t t' q ht ht' hq
    induction q with
    | zero => rfl
    | succ q ih =>
      rw [(runL_calls hw nbuf _ _ _ rd (ph q) 0 _ _ (htpsS t q ht (by omega))).2,
        (runL_calls hw nbuf _ _ _ rd (ph q) 0 _ _ (htpsS t' q ht' (by omega))).2, ih (by omega)]
  have hpre : ∀ t q, t < N → q ≤ K → (tps t q).calls <+: calls t := by
    intro t q ht hq
    have key : ∀ d, q + d ≤ K → (tps t q).calls <+: (tps t (q + d)).calls := by
      intro d
      induction d with
      | zero => intro _; exact List.prefix_refl _
      | succ d ihd =>
        intro hd
        exact (ihd (by omega)).trans (runL_calls hw nbuf _ _ _ rd (ph (q + d)) 0 _ _
          (htpsS t (q + d) ht (by omega))).1
    have := key (K - q) (by omega)
    rw [show q + (K - q) = K by omega, htpsK t ht, ← hcalls t ht] at this
    exact this
  have hS : ∀ wi, wi < nw → Settled (fun l => o (32 * wi + l.val)) (fun l => calls (32 * wi + l.val)) := by
    intro wi hwi l k a ty m hF
    have ha := answers_at hans (hT wi hwi l)
    rw [hset _ (hT wi hwi l)] at ha
    obtain ⟨args, hl, hk⟩ := answersOf_bfly ha hF
    have h32 : (32 * wi + l.val) / 32 = wi := by have : l.val < 32 := l.isLt; omega
    rw [h32, hlnv wi l] at hl
    unfold laneArg at hl
    refine ⟨args, ?_, ?_⟩
    · show (calls (32 * wi + (l.val ^^^ m.toNat) % 32))[k]? = _
      split at hl
      · rename_i c' a' hca
        split at hl
        · rename_i hc
          cases hl
          rw [hca, show c' = .shflBfly from by revert hc; cases c' <;> decide]
        · cases hl
      · cases hl
    · show wordAt (ans (32 * wi + l.val)) k = _
      simp only [wordAt, hk]
      cases args with
      | nil => rfl
      | cons x xs => cases x <;> rfl
  -- stores, by thread and phase
  have hstA : ∀ t q b j, t < N → q < K → St t q b j → A q (vc t) (ln t) b j := fun t q b j ht hq h =>
    stores_acc (vc t) (A q (vc t)) rd (ln t) (ph q) 0 b j (hacc (t / 32) q (by omega) hq) h
  have hsto' : ∀ t q b j, q < K → St t q b j → StoresL (vc t) (imOf rd) (ln t) 0 s b j := fun t q b j hq h =>
    (storesL_split (vc t) (imOf rd) (ln t) 0 hsp b j).mpr ⟨ps[q], List.getElem_mem hq, by
      have hpq : ph q = ps[q] := by simp [ph, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hq]
      rw [← hpq]; exact h⟩
  have hStw : ∀ wi (l : Lane) r b j, StoresL (cta * nw + wi) (imOf rd) l 0 (ph r) b j → St (32 * wi + l.val) r b j :=
    fun wi l r b j h => by show StoresL (vc _) _ (ln _) _ _ _ _; rw [hvcl, hlnl]; exact h
  have hinit : ∀ q wi (l : Lane) b j, q < K → wi < nw → A q (cta * nw + wi) l b j →
      W0.load (bufW nbuf b j) 4 = some ((ws0 nbuf m0).mem b j).bits := by
    intro q wi l b j hq hwi hAe
    have hv := hA q _ l b j hAe
    obtain ⟨h1, h2⟩ := hin q wi l b j hq hwi hAe
    show (rm A0 w.mem).load _ 4 = _
    rw [load_buf_arena _ _ (decode_bufW hv.1 hv.2.1 hv.2.2), h1]
    obtain ⟨x, hx⟩ := Option.isSome_iff_exists.mp h2
    simp [ws0, hx]
  have hstart : ∀ t b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) →
      (tps t 0).mem.load (bufW nbuf b j) 4 = (M t).load (bufW nbuf b j) 4 := fun t b j hv =>
    load_buf_arena _ _ (decode_bufW hv.1 hv.2.1 hv.2.2) 4
  -- before a thread's phase, a word it touches holds what the warp machine does
  have hrel : ∀ q, q < K → (∀ p, p < q → ∀ wi, wi < nw → ∀ l : Lane,
      Rel nbuf (A p (cta * nw + wi)) (wps wi (p + 1)) (tps (32 * wi + l.val) (p + 1)) l) →
      ∀ wi, wi < nw → ∀ (l : Lane) b j, A q (cta * nw + wi) l b j →
        (tps (32 * wi + l.val) q).mem.load (bufW nbuf b j) 4 = some ((wps wi q).mem b j).bits := by
    intro q hq ih wi hwi l b j hAe
    have ht := hT wi hwi l
    have hv := hA q _ l b j hAe
    have hAt : A q (vc (32 * wi + l.val)) (ln (32 * wi + l.val)) b j := by rw [hvcl, hlnl]; exact hAe
    by_cases hex : ∃ o' p', o' < N ∧ p' < K ∧ St o' p' b j
    · obtain ⟨o', p', ho, hp, hst⟩ := hex
      have huniq : ∀ t'' r, t'' < N → r < K → St t'' r b j → t'' = o' ∧ r = p' :=
        fun t'' r h1 h2 h3 => hone t'' o' r p' b j h1 ho h2 hp h3 hst
      have hwnot : ∀ r (l' : Lane), r < K → r ≠ p' → ¬ StoresL (cta * nw + wi) (imOf rd) l' 0 (ph r) b j :=
        fun r l' hr hne hs => hne (huniq _ r (hT wi hwi l') hr (hStw wi l' r b j hs)).2
      by_cases hto : 32 * wi + l.val = o'
      · subst hto
        by_cases hqp : q ≤ p'
        · rw [htf _ 0 q ht (Nat.zero_le _) (by omega) b j hv (fun r _ hr hs => by
              have := huniq _ r ht (by omega) hs; omega), hstart _ b j hv,
            hF1 _ (by omega) b j hv (fun t' ht' hs => by
              obtain ⟨r, hr, hs'⟩ := hsto t' b j hs
              have := huniq t' r (by omega) hr hs'; omega),
            hwf wi 0 q (Nat.zero_le _) (by omega) b j (fun r l' _ hr => hwnot r l' (by omega) (by omega))]
          exact hinit q wi l b j hq hwi hAe
        · rw [htf _ (p' + 1) q ht (by omega) (by omega) b j hv (fun r h1 h2 hs => by
              have := huniq _ r ht (by omega) hs; omega),
            (ih p' (by omega) wi hwi l).2 b j (by
              have := hstA _ p' b j ht hp hst; rw [hvcl, hlnl] at this; exact this),
            hwf wi (p' + 1) q (by omega) (by omega) b j (fun r l' h1 h2 => hwnot r l' (by omega) (by omega))]
      · obtain ⟨hpq, hot, hw⟩ := hlate _ o' q p' b j ht ho hq hp hAt hst hto
        have hwo : o' / 32 = wi := by rw [hw]; have : l.val < 32 := l.isLt; omega
        let lo : Lane := ln o'
        have hoeq : 32 * wi + lo.val = o' := by show 32 * wi + o' % 32 = o'; omega
        rw [htf _ 0 q ht (Nat.zero_le _) (by omega) b j hv (fun r _ hr hs => hto (huniq _ r ht (by omega) hs).1),
          hstart _ b j hv,
          hF2 o' _ hot (by omega) b j hv (fun t' h1 h2 hs => by
            obtain ⟨r, hr, hs'⟩ := hsto t' b j hs
            have := huniq t' r (by omega) hr hs'; omega),
          hMtr o' ho, ← htpsK o' ho,
          htf o' (p' + 1) K ho (by omega) (Nat.le_refl _) b j hv (fun r h1 h2 hs => by
            have := huniq o' r ho (by omega) hs; omega)]
        have hpost := (ih p' hpq wi hwi lo).2 b j (by
          have := hstA o' p' b j ho hp hst
          have hvo : vc o' = cta * nw + wi := by show cta * nw + o' / 32 = cta * nw + wi; rw [hwo]
          rw [hvo] at this; exact this)
        rw [hoeq] at hpost
        rw [hpost, hwf wi (p' + 1) q (by omega) (by omega) b j (fun r l' h1 h2 => hwnot r l' (by omega) (by omega))]
    · have hnone : ∀ t'' r, t'' < N → r < K → ¬ St t'' r b j := fun t'' r h1 h2 h3 => hex ⟨t'', r, h1, h2, h3⟩
      rw [htf _ 0 q ht (Nat.zero_le _) (by omega) b j hv (fun r _ hr => hnone _ r ht (by omega)),
        hstart _ b j hv,
        hF1 _ (by omega) b j hv (fun t' ht' hs => by
          obtain ⟨r, hr, hs'⟩ := hsto t' b j hs; exact hnone t' r (by omega) hr hs'),
        hwf wi 0 q (Nat.zero_le _) (by omega) b j (fun r l' _ hr hs =>
          hnone _ r (hT wi hwi l') (by omega) (hStw wi l' r b j hs))]
      exact hinit q wi l b j hq hwi hAe
  -- each phase, each warp runs with the warp machine
  have hpost : ∀ q, q < K → ∀ wi, wi < nw → ∀ l : Lane,
      Rel nbuf (A q (cta * nw + wi)) (wps wi (q + 1)) (tps (32 * wi + l.val) (q + 1)) l := by
    intro q
    induction q using Nat.strongRecOn with
    | ind q ih =>
    intro hq wi hwi
    rw [hwpsS wi q hq]
    refine lockstep hw hw32 nbuf (cta * nw + wi) (A q (cta * nw + wi)) rd (fun l => o (32 * wi + l.val))
      (fun l => calls (32 * wi + l.val)) (hS wi hwi) (hA q _) (ph q) 0 (wps wi q)
      (fun l => tps (32 * wi + l.val) q) (fun l => tps (32 * wi + l.val) (q + 1)) (hacc wi q hwi hq)
      (fun l => by have h := htpsS _ q (hT wi hwi l) hq; rw [hvcl, hlnv] at h; exact h)
      (fun l => ⟨fun n => ?_, fun b j hAe => hrel q hq (fun p hp => ih p hp (by omega)) wi hwi l b j hAe⟩)
      (fun l => hr32 _ q (hT wi hwi l) (by omega))
      (fun l l' => hcl _ _ q (hT wi hwi l) (hT wi hwi l') (by omega))
      (fun l => hpre _ (q + 1) (hT wi hwi l) hq)
    cases q with
    | zero => rfl
    | succ q' => exact (ih q' (by omega) (by omega) wi hwi l).1 n
  have hMSall : ∀ t, MemSame W0 (rm A0 (M t)) := by
    intro t
    induction t with
    | zero => exact MemSame.refl _
    | succ t ih =>
      rw [← hMS]
      refine ih.trans ?_
      show MemSame (rm A0 (M t)) (rm A0 (f t (M t)))
      simp only [f]
      split
      · cases hrun : runL hw nbuf (vc t) (t % 32) (o t) rd 0 s (start (M t)) with
        | none => exact MemSame.refl _
        | some st' =>
          simp only [Option.map_some, Option.getD_some]
          exact memSame_rm (runL_same hw nbuf (vc t) (t % 32) (o t) rd s 0 (start (M t)) st' hrun) A0
      · exact MemSame.refl _
  simp only at hmem
  refine ⟨fun q wi l b j hq hwi hAe => ?_, fun b j hv hn => ?_, ?_, ?_, ?_⟩
  · have ht := hT wi hwi l
    have hv := hA q _ l b j hAe
    have hAt : A q (vc (32 * wi + l.val)) (ln (32 * wi + l.val)) b j := by rw [hvcl, hlnl]; exact hAe
    show ({ w''.mem with arena := w.mem.arena } : Mem).load _ 4 = _
    rw [load_buf_arena _ _ (decode_bufW hv.1 hv.2.1 hv.2.2), hmem, ← hwpsK wi]
    show (M N).load (bufW nbuf b j) 4 = _
    by_cases hex : ∃ o' p', o' < N ∧ p' < K ∧ St o' p' b j
    · obtain ⟨o', p', ho, hp, hst⟩ := hex
      have huniq : ∀ t'' r, t'' < N → r < K → St t'' r b j → t'' = o' ∧ r = p' :=
        fun t'' r h1 h2 h3 => hone t'' o' r p' b j h1 ho h2 hp h3 hst
      have hwo : o' / 32 = wi := by
        by_cases hto : 32 * wi + l.val = o'
        · rw [← hto]; have : l.val < 32 := l.isLt; omega
        · obtain ⟨_, _, hw⟩ := hlate _ o' q p' b j ht ho hq hp hAt hst hto
          rw [hw]; have : l.val < 32 := l.isLt; omega
      let lo : Lane := ln o'
      have hoeq : 32 * wi + lo.val = o' := by show 32 * wi + o' % 32 = o'; omega
      rw [hF2 o' N (by omega) (Nat.le_refl _) b j hv (fun t' h1 h2 hs => by
            obtain ⟨r, hr, hs'⟩ := hsto t' b j hs
            have := huniq t' r (by omega) hr hs'; omega),
        hMtr o' ho, ← htpsK o' ho,
        htf o' (p' + 1) K ho (by omega) (Nat.le_refl _) b j hv (fun r h1 h2 hs => by
          have := huniq o' r ho (by omega) hs; omega)]
      have hp1 := (hpost p' hp wi hwi lo).2 b j (by
        have := hstA o' p' b j ho hp hst
        have hvo : vc o' = cta * nw + wi := by show cta * nw + o' / 32 = cta * nw + wi; rw [hwo]
        rw [hvo] at this; exact this)
      rw [hoeq] at hp1
      rw [hp1, hwf wi (p' + 1) K (by omega) (Nat.le_refl _) b j (fun r l' h1 h2 hs => by
        have := huniq _ r (hT wi hwi l') (by omega) (hStw wi l' r b j hs); omega)]
    · have hnone : ∀ t'' r, t'' < N → r < K → ¬ St t'' r b j := fun t'' r h1 h2 h3 => hex ⟨t'', r, h1, h2, h3⟩
      rw [hF1 N (Nat.le_refl _) b j hv (fun t' ht' hs => by
          obtain ⟨r, hr, hs'⟩ := hsto t' b j hs; exact hnone t' r ht' hr hs'),
        hwf wi 0 K (Nat.zero_le _) (Nat.le_refl _) b j (fun r l' _ hr hs =>
          hnone _ r (hT wi hwi l') hr (hStw wi l' r b j hs))]
      exact hinit q wi l b j hq hwi hAe
  · show ({ w''.mem with arena := w.mem.arena } : Mem).load _ 4 = _
    rw [load_buf_arena _ _ (decode_bufW hv.1 hv.2.1 hv.2.2), hmem]
    show (M N).load (bufW nbuf b j) 4 = _
    rw [hF1 N (Nat.le_refl _) b j hv (fun t' ht' hs => hn (t' / 32) (ln t') (by omega) hs)]
    exact load_buf_arena w.mem A0 (decode_bufW hv.1 hv.2.1 hv.2.2) 4
  · show MemSame w.mem ({ w''.mem with arena := w.mem.arena } : Mem)
    rw [hmem]
    exact memSame_rm (hMSall N) w.mem.arena
  · show RdOk nbuf ({ w''.mem with arena := w.mem.arena } : Mem) rd
    rw [hmem]
    exact (rdOk_arena w.mem.arena).mpr (hMrd N)
  · show ({ w''.mem with arena := w.mem.arena } : Mem).nan = _
    rw [hmem]
    exact hMnan N

theorem fits_split (rd : Buf → Nat → Option Nat) (cta i : Nat) :
    ∀ {s : EWStmt} {ps : List EWStmt}, Split s ps → fits rd cta i s = true → ∀ p ∈ ps, fits rd cta i p = true := by
  intro s ps h
  induction h with
  | one s => intro hf p hp; simp at hp; rw [hp]; exact hf
  | seq ha hb iha ihb =>
    intro hf p hp
    simp only [fits, Bool.and_eq_true] at hf
    rcases List.mem_append.mp hp with hp | hp
    · exact iha hf.1 p hp
    · exact ihb hf.2 p hp

/-- **A lane's run of a phased statement finishes** where its indexes fit and
    every word each phase may touch lies inside its buffer, none a word `rd`
    reads. -/
theorem runL_total_split (hw : Hw) (nbuf cta : Nat) (A : Nat → Acc) (rd : Buf → Nat → Option Nat) (l : Lane)
    (o : Nat → UInt32) {s : EWStmt} {ps : List EWStmt} (hsp : Split s ps)
    (hA : ∀ q b j, A q l b j → b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32)
    (hRd : ∀ q b j, A q l b j → rd b j = none)
    (hacc : ∀ q, q < ps.length → AccOk cta (A q) rd 0 (ps.getD q .skip))
    (hfits : fits rd cta 0 s = true) (st : LSt)
    (hsz : ∀ q b j, q < ps.length → A q l b j → 4 * j + 4 ≤ (st.mem.bufs b).size) :
    ∃ st', runL hw nbuf cta l.val o rd 0 s st = some st' := by
  rw [runL_split hw nbuf cta l.val o rd 0 hsp]
  have key : ∀ (L : List EWStmt) (k : Nat) (st : LSt), (∀ i, i < L.length → L[i]? = ps[k + i]?) → k + L.length ≤ ps.length →
      (∀ q b j, q < ps.length → A q l b j → 4 * j + 4 ≤ (st.mem.bufs b).size) →
      ∃ st', L.foldlM (m := Option) (fun st p => runL hw nbuf cta l.val o rd 0 p st) st = some st' := by
    intro L
    induction L with
    | nil => intro k st _ _ _; exact ⟨st, rfl⟩
    | cons p ps' ih =>
      intro k st hL hk hs
      have hp : ps[k]? = some p := by have := hL 0 (by simp); simpa using this.symm
      have hk' : k < ps.length := by simp at hk; omega
      have hpk : ps.getD k .skip = p := by simp [List.getD_eq_getElem?_getD, hp]
      obtain ⟨s1, hs1⟩ := runL_total hw nbuf cta (A k) rd l o (hA k) (hRd k) p 0 st
        (fits_split rd cta 0 hsp hfits p (List.mem_of_getElem? hp)) (hpk ▸ hacc k hk')
        (fun b j h => hs k b j hk' h)
      have hsame := runL_same hw nbuf cta l.val o rd p 0 st s1 hs1
      obtain ⟨s2, hs2⟩ := ih (k + 1) s1 (fun i hi => by rw [show k + 1 + i = k + (i + 1) by omega, ← hL (i + 1) (by simp; omega)]; rfl)
        (by simp at hk; omega) (fun q b j hq h => by
          have := hsame.size (.buf ⟨b, (hA q b j h).2.1⟩)
          simp only [Mem.region] at this
          rw [← this]; exact hs q b j hq h)
      exact ⟨s2, by simp only [List.foldlM_cons, bind, Option.bind, hs1, hs2]⟩
  exact key ps 0 st (fun i _ => by simp) (by simp) hsz

/-- **A phased statement's launch runs the warp machine**: `grid` blocks of
    `nw` warps of the statement's kernel, warp `w` of block `b` standing as
    block `b·nw + w`, each phase's accesses those its map allows, each word a
    block stores stored by one lane in one phase and touched by its other lanes
    only in later phases and from higher lanes, and no block touching what
    another stores, leave every word a lane touches in any phase as the warp
    machine's run of the statement for that block does from the launch's
    memory. -/
theorem launch_phases (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (name : String) (nbuf smem : Nat)
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
    (bufs bufs' : Nat → ByteArray) (hrd : RdOk nbuf (kmem bufs) rd)
    (hRd : ∀ q c l b j, c < grid * nw → A q c l b j → rd b j = none)
    (hfits : ∀ c, c < grid * nw → fits rd c 0 s = true)
    (hin : ∀ q c l b j, c < grid * nw → q < ps.length → A q c l b j → 4 * j + 4 ≤ (bufs b).size)
    (hr : LaunchRuns (Body.kernel name nbuf smem s nw) grid (32 * nw) (bufArgs nbuf) bufs bufs' hw) :
    (∀ q c l b j, c < grid * nw → q < ps.length → A q c l b j →
      ({ kmem bufs with bufs := bufs' } : Mem).load (bufW nbuf b j) 4 =
        some ((@WStmt.run F32 (F32.ops hw) _ (s.elabAt c 0 (fun _ _ => 0) (imOf rd)) (ws0 nbuf (kmem bufs))).mem b j).bits) ∧
    (∀ b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) →
      (∀ c (l : Lane), c < grid * nw → ¬ StoresL c (imOf rd) l 0 s b j) →
      ({ kmem bufs with bufs := bufs' } : Mem).load (bufW nbuf b j) 4 = (kmem bufs).load (bufW nbuf b j) 4) := by
  have hblk : ∀ bi, bi < grid → bi * nw + nw ≤ grid * nw := fun bi hb => by
    rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hb
  have hsto : ∀ c l b j q, q < ps.length → StoresL c (imOf rd) l 0 (ps.getD q .skip) b j →
      StoresL c (imOf rd) l 0 s b j := fun c l b j q hq h =>
    (storesL_split c (imOf rd) l 0 hsp b j).mpr ⟨ps[q], List.getElem_mem hq, by
      have hpq : ps.getD q .skip = ps[q] := by simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hq]
      rw [← hpq]; exact h⟩
  have hok : ∀ c, c < grid * nw → ∀ l : Lane, ∃ st', runL hw nbuf c l.val (fun _ => 0) rd 0 s
      ⟨fun _ => ⟨0⟩, rm (ByteArray.mk (Array.replicate smem 0)) (kmem bufs), []⟩ = some st' := fun c hc l =>
    runL_total_split hw nbuf c (fun q => A q c) rd l _ hsp (fun q b j h => hA q c l b j h)
      (fun q b j h => hRd q c l b j hc h) (fun q hq => hacc c q hc hq) (hfits c hc) _
      (fun q b j hq h => hin q c l b j hc hq h)
  obtain ⟨W, hfold, rfl⟩ := hr
  let run (c : Nat) : WSt F32 := @WStmt.run F32 (F32.ops hw) _ (s.elabAt c 0 (fun _ _ => 0) (imOf rd)) (ws0 nbuf (kmem bufs))
  have inv : ∀ c, c ≤ grid → ∀ Wc, BlocksRun (Body.kernel name nbuf smem s nw) (32 * nw) (bufArgs nbuf) hw c
      ({ mem := { arena := .empty, data := .empty, out := .empty, bufs, nan := hw.nan } } : World) Wc →
      MemSame (kmem bufs) Wc.mem ∧
      (∀ q' c' (l : Lane) b j, c' < c * nw → q' < ps.length → A q' c' l b j →
        Wc.mem.load (bufW nbuf b j) 4 = some ((run c').mem b j).bits) ∧
      (∀ b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) →
        (∀ c' (l : Lane), c' < c * nw → ¬ StoresL c' (imOf rd) l 0 s b j) →
        Wc.mem.load (bufW nbuf b j) 4 = (kmem bufs).load (bufW nbuf b j) 4) ∧
      RdOk nbuf Wc.mem rd ∧ Wc.mem.nan = hw.nan := by
    intro c
    induction c with
    | zero =>
      intro _ Wc h
      cases h
      have hld : ∀ a n, ({ arena := .empty, data := .empty, out := .empty, bufs, nan := hw.nan } : Mem).load a n =
          (kmem bufs).load a n := Mem.load_congr (fun r => by cases r <;> rfl) rfl rfl
      refine ⟨⟨fun r => by cases r <;> rfl, rfl, rfl⟩, fun _ _ _ _ _ h => absurd h (by omega), fun _ _ _ _ => hld _ _,
        fun b j n h => ?_, rfl⟩
      obtain ⟨h1, h2⟩ := hrd b j n h
      exact ⟨h1, by simp only [rdM] at h2 ⊢; rw [hld]; exact h2⟩
    | succ c ih =>
      intro hc Wc h
      obtain ⟨Wp, hp, hblock⟩ := h
      have hcg : c < grid := by omega
      have hb := hblk c hcg
      obtain ⟨hI1, hI2, hI3, hI4, hI5⟩ := ih (by omega) Wp hp
      let ln (t : Nat) : Lane := ⟨t % 32, Nat.mod_lt _ (by decide)⟩
      have hvt : ∀ t, t < 32 * nw → c * nw + t / 32 < grid * nw := fun t ht => by
        have : t / 32 < nw := by omega
        omega
      obtain ⟨h1, h2, h3, h4, h5⟩ := block_phases hw hw32 name nbuf smem s ps hsp nw hnw hnw31 c (by omega) hN A rd
        (fun wi q hwi hq => hacc _ q (by omega) hq) hA
        (fun t t' q q' b j ht ht' hq hq' hs hs' => by
          by_cases hw : t / 32 = t' / 32
          · rw [hw] at hs
            obtain ⟨hl, hqq⟩ := hown _ _ _ q q' b j (hvt t' ht') hq hq' hs hs'
            have := congrArg Fin.val hl
            simp only at this
            exact ⟨by omega, hqq⟩
          · exfalso
            exact hdisj (c * nw + t / 32) (c * nw + t' / 32) (ln t) (ln t') q b j (hvt t ht) (hvt t' ht')
              (by omega) hq (stores_acc _ _ rd _ _ 0 b j (hacc _ q (hvt t ht) hq) hs) (hsto _ _ b j q' hq' hs'))
        (fun t o' q p b j ht ho hq hp hAt hs hto => by
          by_cases hw : o' / 32 = t / 32
          · rw [hw] at hs
            have hne : ln t ≠ ln o' := fun e => hto (by
              have := congrArg Fin.val e; simp only [ln] at this; omega)
            obtain ⟨hpq, hlt⟩ := hrdr _ (ln t) (ln o') q p b j (hvt t ht) hq hp hAt hs hne
            simp only [ln] at hlt
            exact ⟨hpq, by omega, hw⟩
          · exfalso
            exact hdisj (c * nw + t / 32) (c * nw + o' / 32) (ln t) (ln o') q b j (hvt t ht) (hvt o' ho)
              (by omega) hq hAt (hsto _ _ b j p hp hs))
        hfine Wp Wc hI4 hI5
        (fun t ht => by
          obtain ⟨st0, h0⟩ := hok (c * nw + t / 32) (hvt t ht) (ln t)
          exact runL_transfer hw nbuf _ _ _ _ rd s 0 _ st0 _ h0 (memSame_rm hI1 _))
        (kmem bufs)
        (fun q wi l b j hq hwi hAl => ⟨hI3 b j (hA _ _ l b j hAl) (fun c' l' hc' hst =>
            hdisj (c * nw + wi) c' l l' q b j (by omega) (by omega) (by omega) hq hAl hst),
          (bufW_ok (hA _ _ l b j hAl) (hin _ _ l b j (by omega) hq hAl) 0).1⟩)
        hblock
      refine ⟨hI1.trans h3, fun q' c' l b j hc' hq' hA' => ?_, fun b j hv hn => ?_, h4, h5⟩
      · by_cases hcc : c * nw ≤ c'
        · have hwi : c' - c * nw < nw := by rw [Nat.succ_mul] at hc'; omega
          have he : c * nw + (c' - c * nw) = c' := by omega
          have := h1 q' (c' - c * nw) l b j hq' hwi (by rw [he]; exact hA')
          rw [he] at this; exact this
        · rw [h2 b j (hA _ c' l b j hA') (fun wi l'' hwi hst => hdisj c' _ l l'' q' b j (by omega) (by omega)
              (by omega) hq' hA' hst),
            hI2 q' c' l b j (by omega) hq' hA']
      · rw [h2 b j hv (fun wi l'' hwi => hn _ l'' (by rw [Nat.succ_mul]; omega)),
          hI3 b j hv (fun c' l' hc' => hn c' l' (by rw [Nat.succ_mul]; omega))]
  obtain ⟨_, hI2, hI3, _, _⟩ := inv grid (Nat.le_refl _) W hfold
  refine ⟨fun q cta l b j hc hq hAc => ?_, fun b j hv hn => ?_⟩
  · obtain ⟨hb1, hb2, hj1⟩ := hA q cta l b j hAc
    exact (load_bufs (m₁ := { kmem bufs with bufs := W.mem.bufs }) (m₂ := W.mem) (decode_bufW hb1 hb2 hj1) rfl).trans
      (hI2 q cta l b j hc hq hAc)
  · exact (load_bufs (m₁ := { kmem bufs with bufs := W.mem.bufs }) (m₂ := W.mem) (decode_bufW hv.1 hv.2.1 hv.2.2)
      rfl).trans (hI3 b j hv hn)

end AlgorithmLib.ML.BodyPhase
