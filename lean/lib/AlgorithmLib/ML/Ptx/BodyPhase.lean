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
(`laneI_split`), and so is the warp machine's (`run_split`); what it stores,
one of its phases stores (`storesL_split`).

`pass_phases`: a block's pass, each thread run as its lane phase by phase,
where the laws answer a collective as the warp does once every lane of the
warp has made the warp's (`hlaw`), leaves every word a lane touches as the
warp machine's run of the statement does. A thread's run ends and stores only
what it may whatever its answers (`laneI_shape`), so the threads before it
leave each word it reads in a phase as the warp machine has it then, and the
lanes of each warp settle on the warp's answers phase by phase
(`LaneSettle.lane_settle`). `block_phases` is it for a block that runs
(`BlockRuns`), its answers the laws' (`law_bfly`); `launch_phases` composes a
grid; `launch_lockstep` is a launch of one phase.
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

theorem run_split {R : Type} [NumOps R] [NumLit R] (cta i : Nat) (ir : Nat → Lane → Nat) (im : Buf → Nat → Nat) :
    ∀ {s : EWStmt} {ps : List EWStmt}, Split s ps → ∀ (ws : WSt R),
      (s.run ws cta i ir im) = ps.foldl (fun ws p => (p.run ws cta i ir im)) ws := by
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
      ((s.run ws cta i (fun _ _ => 0) im)).mem b j = ws.mem b j := by
  have loop : ∀ (body : EWStmt) (L : List Nat) (ws : WSt R) (b : Buf) (j : Nat),
      (∀ t ∈ L, ∀ (ws : WSt R), ((body.run ws cta t (fun _ _ => 0) im)).mem b j = ws.mem b j) →
      (L.foldl (fun s' t => (body.run s' cta t (fun _ _ => 0) im)) ws).mem b j = ws.mem b j := by
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
    show ((y.run ((x.run ws cta i (fun _ _ => 0) im)) cta i (fun _ _ => 0) im)).mem b j = _
    rw [ihy i _ b j (fun l hl => h l (Or.inr hl)), ihx i ws b j (fun l hl => h l (Or.inl hl))]
  | forN n body ih =>
    intro i ws b j h
    exact loop body _ ws b j (fun t ht ws' => ih t ws' b j (fun l hl => h l ⟨t, List.mem_range.mp ht, hl⟩))
  | forM bb a body ih =>
    intro i ws b j h
    exact loop body _ ws b j (fun t ht ws' => ih t ws' b j (fun l hl => h l ⟨t, List.mem_range.mp ht, hl⟩))
  | iif c x y ihx ihy =>
    intro i ws b j h
    simp only [EWStmt.run]
    split
    · exact ihx i ws b j (fun l hl => h l (Or.inl hl))
    · exact ihy i ws b j (fun l hl => h l (Or.inr hl))
  | iwhile n c cs body ih =>
    intro i ws b j h
    simp only [EWStmt.run]
    refine foldl_inv _ (fun s : WSt R => s.mem b j = ws.mem b j) (fun s _ hs => ?_) _ ws rfl
    split
    · exact (ih i s b j h).trans hs
    · exact hs
  | locals rs body ih => intro i ws b j h; exact ih i ws b j h
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

/-! ## Lanes of a block, phase by phase -/

section Lanes
open BodyLane LaneWarp LaneSettle

theorem laneI_split (x : LCtx) :
    ∀ {s : EWStmt} {ps : List EWStmt}, Split s ps → ∀ li, laneI x s li = ps.foldlM (m := Option) (fun li p => laneI x p li) li := by
  intro s ps h
  induction h with
  | one s => intro li; simp only [List.foldlM_cons, List.foldlM_nil, bind, Option.bind]; cases laneI x s li <;> rfl
  | seq ha hb iha ihb =>
    intro li
    rw [List.foldlM_append]
    simp only [laneI, bind]
    rw [iha li]
    congr 1
    funext y
    exact ihb y

/-- **A pass's worlds**: the world before each thread, each thread's run from
    it with shared memory reset ending in the next, its collectives what the
    pass records. -/
theorem pass_runs {run : Runner} {smem0 : ByteArray} {args : Nat → List V} {ans : Nat → List Answer} :
    ∀ (n : Nat) (s r : PassSt), (List.range n).foldlM (passStep run smem0 args ans) s = .ok r →
      ∃ Ws : Nat → World, Ws 0 = s.1 ∧ Ws n = r.1 ∧ ∀ t, t < n →
        run false (ans t) (args t) { Ws t with mem := { (Ws t).mem with arena := smem0 } } = .ok (Ws (t + 1)) ∧
        r.2.1 t = collCalls (Ws (t + 1)) (Ws t).obs.length := by
  intro n
  induction n with
  | zero => intro s r h; cases h; exact ⟨fun _ => s.1, rfl, rfl, fun t ht => absurd ht (by omega)⟩
  | succ n ih =>
    intro s r h
    rw [List.range_succ, List.foldlM_append] at h
    simp only [bind, Except.bind] at h
    split at h
    · cases h
    rename_i r0 h0
    simp only [List.foldlM_cons, List.foldlM_nil, bind, Except.bind] at h
    split at h
    · cases h
    rename_i r1 h1
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    obtain ⟨Ws0, hW0, hWn, hWs⟩ := ih s r0 h0
    obtain ⟨w0', c0, s0⟩ := r0
    simp only [passStep, bind, Except.bind] at h1
    split at h1
    · cases h1
    rename_i u hu
    simp only [pure, Except.pure, Except.ok.injEq] at h1
    subst h1
    simp only at hWn
    refine ⟨fun t => if t ≤ n then Ws0 t else u, by simp [hW0], by simp only [show ¬ n + 1 ≤ n by omega, if_false], fun t ht => ?_⟩
    by_cases htn : t < n
    · obtain ⟨e1, e2⟩ := hWs t htn
      simp only [if_pos (Nat.le_of_lt htn), if_pos (show t + 1 ≤ n by omega), if_neg (show t ≠ n by omega)]
      exact ⟨e1, e2⟩
    · have : t = n := by omega
      subst this
      simp only [Nat.le_refl, if_true, show ¬ t + 1 ≤ t by omega, if_false]
      rw [hWn]
      exact ⟨hu, rfl⟩

/-- The warp machine after the first `q` phases. -/
def wpsOf (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat) (ps : List EWStmt) (ws0 : WSt F32) (q : Nat) : WSt F32 :=
  (ps.take q).foldl (fun ws p => run hw cta im p ws 0) ws0

/-- The collectives a lane makes in the first `q` phases. -/
def cntOf (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat) (ps : List EWStmt) (ws0 : WSt F32) : Nat → Nat
  | 0 => 0
  | q + 1 => cntOf hw cta im ps ws0 q + (trace hw cta im jLeaf 0 (ps.getD q .skip) (wpsOf hw cta im ps ws0 q)).length

theorem wpsOf_succ (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat) (ps : List EWStmt) (ws0 : WSt F32) (q : Nat)
    (hq : q < ps.length) : wpsOf hw cta im ps ws0 (q + 1) = run hw cta im (ps.getD q .skip) (wpsOf hw cta im ps ws0 q) 0 := by
  have hpq : ps.getD q .skip = ps[q] := by simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hq]
  rw [hpq]
  exact foldl_take_succ _ ps _ q hq

/-- The collective leaves of warp `wi` of block `cta`, phase after phase. -/
def jOf (hw : Hw) (nbuf cta nw : Nat) (rd : Buf → Nat → Option Nat) (ps : List EWStmt) (m0 : Mem) (wi : Nat) :
    List (Nat × EWStmt × WSt F32) :=
  (List.range ps.length).flatMap fun q => trace hw (cta * nw + wi) (imOf rd) jLeaf 0 (ps.getD q .skip)
    (wpsOf hw (cta * nw + wi) (imOf rd) ps (ws0P nbuf m0) q)

theorem cntOf_eq (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat) (ps : List EWStmt) (ws0 : WSt F32) :
    ∀ q, cntOf hw cta im ps ws0 q =
      ((List.range q).flatMap fun q => trace hw cta im jLeaf 0 (ps.getD q .skip) (wpsOf hw cta im ps ws0 q)).length
  | 0 => rfl
  | q + 1 => by
      rw [cntOf, cntOf_eq hw cta im ps ws0 q, List.range_succ, List.flatMap_append, List.length_append]
      simp

end Lanes

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

/-- A statement's fragment's collectives are butterfly shuffles. -/
theorem allAt_of_fits (rd : Buf → Nat → Option Nat) (cta : Nat) :
    ∀ (s : EWStmt) (i : Nat), fits rd cta i s = true → LaneSettle.AllAt (imOf rd) LaneSettle.BflyOnly i s := by
  intro s
  induction s with
  | seq a b iha ihb =>
      intro i hf; simp only [fits, Bool.and_eq_true] at hf; exact ⟨iha i hf.1, ihb i hf.2⟩
  | forN n body ih =>
      intro i hf
      simp only [fits, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range] at hf
      exact fun j hj => ih j (hf.2 j hj)
  | forM bu a body ih =>
      intro i hf
      simp only [fits, Bool.and_eq_true, decide_eq_true_eq] at hf
      obtain ⟨_, hf⟩ := hf
      split at hf
      · rename_i n hn
        simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range] at hf
        have him : imOf rd bu a = n := by simp [imOf, hn]
        exact fun j hj => ih j (hf.2 j (by rw [← him]; exact hj))
      · cases hf
  | shflXor d sr m => intro i hf; simpa [fits] using hf
  | skip | setR | loadIdx | loadV4 | storeLane | storeLane0 => intro _ _; trivial
  | _ => intro _ hf; simp [fits] at hf

section Block
open BodyLane LaneWarp LaneSettle

/-- **A block runs its statement in phases**: a block of `nw` warps of a
    statement's kernel, the statement split into phases (`Split`), each phase's
    accesses those its map allows, leaves every word a lane of warp `w` touches
    in any phase as the warp machine's run of the statement for block
    `cta·nw + w` does from `m0`, where each word is stored in one phase by one
    thread, and any other thread touches it only in a later phase, after that
    thread in the block's pass and in its warp. Each thread runs as its lane,
    phase by phase; whatever its answers, its run ends and stores only what it
    may (`laneI_shape`), so the threads before it leave each word it reads as
    the warp machine has it, and the lanes of each warp settle on the warp's
    answers (`lane_settle`). -/
theorem pass_phases (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (name : String) (nbuf smem : Nat)
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
    (w : World) (hrd : RdOk nbuf w.mem rd) (hnan : w.mem.nan = hw.nan)
    (hfits : ∀ wi, wi < nw → fits rd (cta * nw + wi) 0 s = true)
    (hRd : ∀ q wi l b j, wi < nw → A q (cta * nw + wi) l b j → rd b j = none)
    (hsz : ∀ q wi l b j, q < ps.length → wi < nw → A q (cta * nw + wi) l b j → 4 * j + 4 ≤ (w.mem.bufs b).size)
    (m0 : Mem) (hin : ∀ q wi (l : Lane) b j, q < ps.length → wi < nw → A q (cta * nw + wi) l b j →
      w.mem.load (bufW nbuf b j) 4 = m0.load (bufW nbuf b j) 4 ∧ (m0.load (bufW nbuf b j) 4).isSome)
    (ans : Nat → List Answer) (hWd : ∀ t, t < 32 * nw → Words (ans t))
    (Ws : Nat → World) (hW0 : Ws 0 = { w with mem := { w.mem with arena := ByteArray.mk (Array.replicate smem 0) } })
    (hWs : ∀ t, t < 32 * nw →
      MemSame (rm (ByteArray.mk (Array.replicate smem 0)) w.mem) (rm (ByteArray.mk (Array.replicate smem 0)) (Ws t).mem) →
      RdOk nbuf (Ws t).mem rd → (Ws t).mem.nan = hw.nan →
      ∃ steps, thread (Body.kernel name nbuf smem s nw).code (ans t) (threadArgs cta (32 * nw) (bufArgs nbuf) t)
        { Ws t with mem := { (Ws t).mem with arena := ByteArray.mk (Array.replicate smem 0) } } hw steps = .ok (Ws (t + 1)))
    (calls : Nat → List (Intr × List V))
    (hcalls : ∀ t, t < 32 * nw → calls t = collCalls (Ws (t + 1)) (Ws t).obs.length)
    (hlen : ∀ t, t < 32 * nw → (calls t).length ≤ (ans t).length ∨
      (jOf hw nbuf cta nw rd ps m0 (t / 32)).length ≤ (ans t).length)
    (hlaw : ∀ q wi j, q < ps.length → wi < nw →
      j < (trace hw (cta * nw + wi) (imOf rd) jLeaf 0 (ps.getD q .skip)
        (wpsOf hw (cta * nw + wi) (imOf rd) ps (ws0P nbuf m0) q)).length →
      (∀ l : Lane, (calls (32 * wi + l.val))[cntOf hw (cta * nw + wi) (imOf rd) ps (ws0P nbuf m0) q + j]? =
        (trace hw (cta * nw + wi) (imOf rd) (callLeaf (cta * nw + wi) l) 0 (ps.getD q .skip)
          (wpsOf hw (cta * nw + wi) (imOf rd) ps (ws0P nbuf m0) q))[j]?) →
      ∀ l : Lane, (ans (32 * wi + l.val))[cntOf hw (cta * nw + wi) (imOf rd) ps (ws0P nbuf m0) q + j]? =
        (trace hw (cta * nw + wi) (imOf rd) (ansLeaf (cta * nw + wi) smem l) 0 (ps.getD q .skip)
          (wpsOf hw (cta * nw + wi) (imOf rd) ps (ws0P nbuf m0) q))[j]?) :
    (∀ t, t ≤ 32 * nw → MemSame (rm (ByteArray.mk (Array.replicate smem 0)) w.mem)
      (rm (ByteArray.mk (Array.replicate smem 0)) (Ws t).mem) ∧ RdOk nbuf (Ws t).mem rd ∧ (Ws t).mem.nan = hw.nan) ∧
    (∀ wi, wi < nw → ∀ l : Lane, calls (32 * wi + l.val) = (jOf hw nbuf cta nw rd ps m0 wi).map (callOne (cta * nw + wi) l)) ∧
    (∀ q wi (l : Lane) b j, q < ps.length → wi < nw → A q (cta * nw + wi) l b j →
      (Ws (32 * nw)).mem.load (bufW nbuf b j) 4 =
        some ((@EWStmt.run F32 (F32.ops hw) _ s (ws0 nbuf m0) (cta * nw + wi) 0 (fun _ _ => 0) (imOf rd)).mem b j).bits) ∧
    (∀ b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) →
      (∀ wi l, wi < nw → ¬ StoresL (cta * nw + wi) (imOf rd) l 0 s b j) →
      (Ws (32 * nw)).mem.load (bufW nbuf b j) 4 = w.mem.load (bufW nbuf b j) 4) := by
  let N := 32 * nw
  let K := ps.length
  let ph (q : Nat) : EWStmt := ps.getD q .skip
  let A0 := ByteArray.mk (Array.replicate smem 0)
  let vc (t : Nat) : Nat := cta * nw + t / 32
  let ln (t : Nat) : Lane := ⟨t % 32, Nat.mod_lt _ (by decide)⟩
  let St (t q : Nat) (b : Buf) (j : Nat) : Prop := StoresL (vc t) (imOf rd) (ln t) 0 (ph q) b j
  have hvc : ∀ t, t < N → vc t < 2 ^ 32 := fun t ht => by
    have : t / 32 < nw := by omega
    show cta * nw + t / 32 < 2 ^ 32; omega
  let V (t : Nat) : World := { Ws t with mem := { (Ws t).mem with arena := A0 } }
  let x (t : Nat) : LCtx := laneCtx nbuf cta nw t (ans t) (Ws t).obs.length hw
  let runP (t q : Nat) : Option LI := (ps.take q).foldlM (fun li p => laneI (x t) p li) (LI.init (V t))
  let tps (t q : Nat) : LI := (runP t q).getD (LI.init (V t))
  let Inv (m : Mem) : Prop := MemSame (rm A0 w.mem) (rm A0 m) ∧ RdOk nbuf m rd ∧ m.nan = hw.nan
  have hpq : ∀ q (hq : q < ps.length), ph q = ps[q]'hq := fun q hq => by
    simp [ph, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hq]
  have hph : ∀ t, t < N → ∀ q, q < K → fits rd (vc t) 0 (ph q) = true ∧ AccOk (vc t) (A q (vc t)) rd 0 (ph q) := by
    intro t ht q hq
    have hwi : t / 32 < nw := by omega
    refine ⟨?_, hacc (t / 32) q hwi hq⟩
    rw [hpq q hq]; exact fits_split rd (vc t) 0 hsp (hfits (t / 32) hwi) _ (List.getElem_mem hq)
  have hxok : ∀ t, t < N → CtxOk nbuf (vc t) (ln t) (x t) := fun t ht => laneCtx_ok nbuf cta nw t _ _ hw (hvc t ht)
  have hsz' : ∀ (m : Mem), MemSame (rm A0 w.mem) m → ∀ t q b j, t < N → q < K → A q (vc t) (ln t) b j →
      4 * j + 4 ≤ (m.bufs b).size := fun m hm t q b j ht hq hAe => by
    have := hm.size (.buf ⟨b, (hA q _ _ b j hAe).2.1⟩)
    simp only [Mem.region] at this
    rw [← this]; exact hsz q (t / 32) (ln t) b j hq (by omega) hAe
  -- each thread's phases run, whatever its answers, from memory the invariant holds of
  have hrunq : ∀ t, t < N → Inv (Ws t).mem → ∀ q, q ≤ K →
      runP t q = some (tps t q) ∧ MemSame (V t).mem (tps t q).w.mem ∧ RdOk nbuf (tps t q).w.mem rd ∧
        (tps t q).w.mem.nan = hw.nan ∧ (Ws t).obs.length ≤ (tps t q).w.obs.length := by
    intro t ht hI q
    induction q with
    | zero => intro _; exact ⟨rfl, MemSame.refl _, (rdOk_arena A0).mpr hI.2.1, hI.2.2, Nat.le_refl _⟩
    | succ q ih =>
      intro hq
      obtain ⟨e, h1, h2, h3, h4⟩ := ih (by omega)
      obtain ⟨hf, hac⟩ := hph t ht q (by omega)
      obtain ⟨li', e', sh⟩ := laneI_shape nbuf (vc t) (A q (vc t)) rd (ln t) (fun b j h => hA _ _ _ b j h)
        (fun b j h => hRd q (t / 32) (ln t) b j (by omega) h) (ph q) (x t) (tps t q) (hxok t ht) hf hac h4 h2
        (fun b j h => hsz' _ (hI.1.trans h1) t q b j ht (by omega) h)
      have hr : runP t (q + 1) = some li' := by
        show (ps.take (q + 1)).foldlM _ _ = _
        rw [foldlM_take_succ _ ps _ q (by omega)]
        show (runP t q).bind _ = _
        rw [e, Option.bind_some, ← hpq q (by omega)]; exact e'
      have htp : tps t (q + 1) = li' := by simp only [tps, hr, Option.getD_some]
      rw [htp]
      refine ⟨hr, h1.trans sh.same, sh.rdok, sh.nan.trans h3, ?_⟩
      obtain ⟨L, hL⟩ := laneI_grows _ _ _ _ e'
      rw [hL, List.length_append]; omega
  have hstep : ∀ t, t < N → Inv (Ws t).mem → ∀ q, q < K →
      laneI (x t) (ph q) (tps t q) = some (tps t (q + 1)) ∧
        Shape nbuf (vc t) rd (ln t) 0 (ph q) (tps t q) (tps t (q + 1)) := by
    intro t ht hI q hq
    obtain ⟨e, h1, h2, h3, h4⟩ := hrunq t ht hI q (by omega)
    obtain ⟨e1, _⟩ := hrunq t ht hI (q + 1) hq
    obtain ⟨hf, hac⟩ := hph t ht q hq
    obtain ⟨li', e', sh⟩ := laneI_shape nbuf (vc t) (A q (vc t)) rd (ln t) (fun b j h => hA _ _ _ b j h)
      (fun b j h => hRd q (t / 32) (ln t) b j (by omega) h) (ph q) (x t) (tps t q) (hxok t ht) hf hac h4 h2
      (fun b j h => hsz' _ (hI.1.trans h1) t q b j ht hq h)
    have hr : runP t (q + 1) = some li' := by
      show (ps.take (q + 1)).foldlM _ _ = _
      rw [foldlM_take_succ _ ps _ q hq]
      show (runP t q).bind _ = _
      rw [e, Option.bind_some, ← hpq q hq]; exact e'
    rw [e1] at hr; cases hr
    exact ⟨e', sh⟩
  have hthr : ∀ t, t < N → Inv (Ws t).mem → Ws (t + 1) = (tps t K).w ∧ calls t = collCalls (tps t K).w (Ws t).obs.length := by
    intro t ht hI
    obtain ⟨e, _⟩ := hrunq t ht hI K (Nat.le_refl _)
    have hl : laneI (x t) s (LI.init (V t)) = some (tps t K) := by
      rw [laneI_split (x t) hsp]
      have : runP t K = ps.foldlM (fun li p => laneI (x t) p li) (LI.init (V t)) := by
        show (ps.take K).foldlM _ _ = _; rw [List.take_length]
      rw [← this]; exact e
    obtain ⟨steps, hW1⟩ := hWs t ht hI.1 hI.2.1 hI.2.2
    have hu := thread_body hw hw32 name nbuf smem s nw hnw hnw31 cta t (32 * nw) ht (by omega) (hvc t ht) (V t)
      (Ws (t + 1)) (ans t) (hWd t ht) (tps t K) hfine hl hW1
    exact ⟨hu, by rw [hcalls t ht, hu]⟩
  have hInv : ∀ t, t ≤ N → Inv (Ws t).mem := by
    intro t
    induction t with
    | zero =>
      intro _
      rw [hW0]
      exact ⟨MemSame.refl _, (rdOk_arena A0).mpr hrd, hnan⟩
    | succ t ih =>
      intro ht
      have hI := ih (by omega)
      obtain ⟨_, h1, h2, h3, _⟩ := hrunq t (by omega) hI K (Nat.le_refl _)
      rw [(hthr t (by omega) hI).1]
      exact ⟨hI.1.trans ((memSame_rm h1 A0)), h2, h3⟩
  have hS : ∀ t q, t < N → q < K → laneI (x t) (ph q) (tps t q) = some (tps t (q + 1)) ∧
      Shape nbuf (vc t) rd (ln t) 0 (ph q) (tps t q) (tps t (q + 1)) := fun t q ht hq =>
    hstep t ht (hInv t (by omega)) q hq
  have hR : ∀ t q, t < N → q ≤ K → MemSame (V t).mem (tps t q).w.mem ∧ RdOk nbuf (tps t q).w.mem rd ∧
      (tps t q).w.mem.nan = hw.nan ∧ (Ws t).obs.length ≤ (tps t q).w.obs.length := fun t q ht hq =>
    (hrunq t ht (hInv t (by omega)) q hq).2
  have hT : ∀ t, t < N → Ws (t + 1) = (tps t K).w ∧ calls t = collCalls (tps t K).w (Ws t).obs.length :=
    fun t ht => hthr t ht (hInv t (by omega))
  -- within a thread, a phase that does not store a word leaves it alone
  have htf : ∀ t q q', t < N → q ≤ q' → q' ≤ K → ∀ b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) →
      (∀ r, q ≤ r → r < q' → ¬ St t r b j) →
      (tps t q').w.mem.load (bufW nbuf b j) 4 = (tps t q).w.mem.load (bufW nbuf b j) 4 := by
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
      · rw [(hS t q' ht (by omega)).2.keep b j hv (hn q' (by omega) (by omega)),
          ih (by omega) (by omega) b j hv (fun r h1 h2 => hn r h1 (by omega))]
  -- a thread leaves alone what it does not store
  have hfS : ∀ t, t < N → ∀ b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) → (∀ q, q < K → ¬ St t q b j) →
      (Ws (t + 1)).mem.load (bufW nbuf b j) 4 = (Ws t).mem.load (bufW nbuf b j) 4 := by
    intro t ht b j hv hn
    rw [(hT t ht).1, htf t 0 K ht (Nat.zero_le _) (Nat.le_refl _) b j hv (fun r _ hr => hn r hr)]
    exact load_buf_arena _ A0 (decode_bufW hv.1 hv.2.1 hv.2.2) 4
  have hsto : ∀ t b j, StoresL (vc t) (imOf rd) (ln t) 0 s b j → ∃ q, q < K ∧ St t q b j := by
    intro t b j h
    obtain ⟨p, hp, hs⟩ := (storesL_split (vc t) (imOf rd) (ln t) 0 hsp b j).mp h
    obtain ⟨q, hq, rfl⟩ := List.getElem_of_mem hp
    exact ⟨q, hq, by rw [show St t q b j = StoresL (vc t) (imOf rd) (ln t) 0 (ph q) b j from rfl, hpq q hq]; exact hs⟩
  -- F1: a word no earlier thread stores is as the block found it
  have hF1 : ∀ t, t ≤ N → ∀ b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) →
      (∀ t', t' < t → ∀ q, q < K → ¬ St t' q b j) →
      (Ws t).mem.load (bufW nbuf b j) 4 = w.mem.load (bufW nbuf b j) 4 := by
    intro t
    induction t with
    | zero => intro _ b j hv _; rw [hW0]; exact load_buf_arena _ A0 (decode_bufW hv.1 hv.2.1 hv.2.2) 4
    | succ t ih =>
      intro ht b j hv hn
      rw [hfS t (by omega) b j hv (hn t (by omega)), ih (by omega) b j hv (fun t' ht' => hn t' (by omega))]
  -- F2: past its one storing thread, a word is as that thread left it
  have hF2 : ∀ o' t, o' < t → t ≤ N → ∀ b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) →
      (∀ t', o' < t' → t' < t → ∀ q, q < K → ¬ St t' q b j) →
      (Ws t).mem.load (bufW nbuf b j) 4 = (Ws (o' + 1)).mem.load (bufW nbuf b j) 4 := by
    intro o' t hot
    induction t with
    | zero => omega
    | succ t ih =>
      intro ht b j hv hn
      by_cases he : t = o'
      · subst he; rfl
      · rw [hfS t (by omega) b j hv (hn t (by omega) (by omega)),
          ih (by omega) (by omega) b j hv (fun t' h1 h2 => hn t' h1 (by omega))]
  -- the warp machine, phase by phase
  let I0 : ISt := { ptr := bufB nbuf }
  let wps (wi q : Nat) : WSt F32 := wpsOf hw (cta * nw + wi) (imOf rd) ps (ws0 nbuf m0) q
  let wpsP (wi q : Nat) : WSt F32 := wpsOf hw (cta * nw + wi) (imOf rd) ps (ws0P nbuf m0) q
  have hwP : ∀ wi, wi < nw → ∀ q, q ≤ K → wpsP wi q = (wps wi q).withInt I0 := by
    intro wi hwi q
    induction q with
    | zero => intro _; rfl
    | succ q ih =>
      intro hq
      show wpsOf _ _ _ ps _ (q + 1) = (wpsOf _ _ _ ps _ (q + 1)).withInt I0
      rw [wpsOf_succ _ _ _ ps _ q hq, wpsOf_succ _ _ _ ps _ q hq]
      show run hw (cta * nw + wi) (imOf rd) (ph q) (wpsP wi q) 0 = _
      rw [ih (by omega)]
      have h := (hph (32 * wi) (by omega) q hq).1
      have e : vc (32 * wi) = cta * nw + wi := by
        show cta * nw + 32 * wi / 32 = _; rw [Nat.mul_div_cancel_left _ (by decide)]
      rw [e] at h
      exact run_withInt hw (cta * nw + wi) rd I0 (ph q) 0 (wps wi q) h
  have hwpsK : ∀ wi, wps wi K =
      @EWStmt.run F32 (F32.ops hw) _ s (ws0 nbuf m0) (cta * nw + wi) 0 (fun _ _ => 0) (imOf rd) := by
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
      · show (wpsOf _ _ _ ps _ (q' + 1)).mem b j = _
        rw [wpsOf_succ _ _ _ ps _ q' (by omega), LaneWarp.run, @run_frame_st F32 (F32.ops hw) _ (cta * nw + wi) (imOf rd) (ph q') 0 _ b j
          (fun l => hn q' l (by omega) (by omega)), ih (by omega) (by omega) b j (fun r l h1 h2 => hn r l h1 (by omega))]
  -- lanes of warp `wi`
  have hTl : ∀ wi, wi < nw → ∀ l : Lane, 32 * wi + l.val < N := fun wi hwi l => by
    have : l.val < 32 := l.isLt; show 32 * wi + l.val < 32 * nw; omega
  have hvcl : ∀ wi (l : Lane), vc (32 * wi + l.val) = cta * nw + wi := fun wi l => by
    have : l.val < 32 := l.isLt; show cta * nw + (32 * wi + l.val) / 32 = cta * nw + wi; omega
  have hlnl : ∀ wi (l : Lane), ln (32 * wi + l.val) = l := fun wi l =>
    Fin.ext (by have : l.val < 32 := l.isLt; show (32 * wi + l.val) % 32 = l.val; omega)
  have hStw : ∀ wi (l : Lane) r b j, StoresL (cta * nw + wi) (imOf rd) l 0 (ph r) b j → St (32 * wi + l.val) r b j :=
    fun wi l r b j h => by show StoresL (vc _) _ (ln _) _ _ _ _; rw [hvcl, hlnl]; exact h
  have hstA : ∀ t q b j, t < N → q < K → St t q b j → A q (vc t) (ln t) b j := fun t q b j ht hq h =>
    stores_acc (vc t) (A q (vc t)) rd (ln t) (ph q) 0 b j (hph t ht q hq).2 h
  have hinit : ∀ q wi (l : Lane) b j, q < K → wi < nw → A q (cta * nw + wi) l b j →
      w.mem.load (bufW nbuf b j) 4 = some ((ws0 nbuf m0).mem b j).bits := by
    intro q wi l b j hq hwi hAe
    obtain ⟨h1, h2⟩ := hin q wi l b j hq hwi hAe
    rw [h1]
    obtain ⟨y, hy⟩ := Option.isSome_iff_exists.mp h2
    simp [ws0, hy]
  have hstart : ∀ t b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) →
      (tps t 0).w.mem.load (bufW nbuf b j) 4 = (Ws t).mem.load (bufW nbuf b j) 4 := fun t b j hv =>
    load_buf_arena _ _ (decode_bufW hv.1 hv.2.1 hv.2.2) 4
  have hptr : ∀ t, (x t).ptr = bufB nbuf := fun t => funext (bufPs_get nbuf)
  have hgrow : ∀ t, t < N → ∀ q d, q + d ≤ K → LaneSettle.Grows (tps t q).w (tps t (q + d)).w := by
    intro t ht q d
    induction d with
    | zero => intro _; exact LaneSettle.Grows.refl _
    | succ d ih =>
      intro hd
      exact (ih (by omega)).trans (laneI_grows _ _ _ _ (hS t (q + d) ht (by omega)).1)
  have hpref : ∀ t, t < N → ∀ q, q ≤ K → collCalls (tps t q).w (Ws t).obs.length <+: calls t := by
    intro t ht q hq
    rw [(hT t ht).2]
    have := collCalls_prefix (hgrow t ht q (K - q) (by omega)) (Ws t).obs.length
    rwa [show q + (K - q) = K by omega] at this
  let WSq (q wi : Nat) : Lane → UInt64 → Nat → Prop := fun l a n =>
    n = 4 ∧ ∃ b j, A q (cta * nw + wi) l b j ∧ a = bufB nbuf b + off (UInt32.ofNat j)
  let cnt (wi q : Nat) : Nat := cntOf hw (cta * nw + wi) (imOf rd) ps (ws0P nbuf m0) q
  let Post (q wi : Nat) (l : Lane) : Prop :=
    (∃ cl, RelW hw smem (A q (cta * nw + wi)) (fun _ _ => False) (rdI rd) (WSq q wi) (imOf rd) l (x (32 * wi + l.val))
      (wpsP wi (q + 1)) cl (tps (32 * wi + l.val) (q + 1))) ∧
    (collCalls (tps (32 * wi + l.val) (q + 1)).w (Ws (32 * wi + l.val)).obs.length).length = cnt wi (q + 1) ∧
    collCalls (tps (32 * wi + l.val) (q + 1)).w (Ws (32 * wi + l.val)).obs.length =
      collCalls (tps (32 * wi + l.val) q).w (Ws (32 * wi + l.val)).obs.length ++
        trace hw (cta * nw + wi) (imOf rd) (callLeaf (cta * nw + wi) l) 0 (ph q) (wpsP wi q)
  have hfm : ∀ p wi (l : Lane), p < K → wi < nw → Post p wi l → ∀ b j, A p (cta * nw + wi) l b j →
      (tps (32 * wi + l.val) (p + 1)).w.mem.load (bufW nbuf b j) 4 = some ((wps wi (p + 1)).mem b j).bits := by
    intro p wi l hp hwi hP b j hAe
    obtain ⟨cl, hr⟩ := hP.1
    have := hr.fmem b j hAe
    rw [hptr, hwP wi hwi (p + 1) hp] at this
    exact this
  -- before a thread's phase, a word it touches holds what the warp machine does
  have hrel : ∀ q, q < K → (∀ p, p < q → ∀ wi, wi < nw → ∀ l : Lane, Post p wi l) →
      ∀ wi, wi < nw → ∀ (l : Lane) b j, A q (cta * nw + wi) l b j →
        (tps (32 * wi + l.val) q).w.mem.load (bufW nbuf b j) 4 = some ((wps wi q).mem b j).bits := by
    intro q hq ih wi hwi l b j hAe
    have ht := hTl wi hwi l
    have hv := hA q _ l b j hAe
    have hAt : A q (vc (32 * wi + l.val)) (ln (32 * wi + l.val)) b j := by rw [hvcl, hlnl]; exact hAe
    by_cases hex : ∃ o' p', o' < N ∧ p' < K ∧ St o' p' b j
    · obtain ⟨o', p', ho, hp, hst⟩ := hex
      have huniq : ∀ t'' r, t'' < N → r < K → St t'' r b j → t'' = o' ∧ r = p' :=
        fun t'' r h1 h2 h3 => hone t'' o' r p' b j h1 ho h2 hp h3 hst
      have hwnot : ∀ r (l' : Lane), r < K → r ≠ p' → ¬ StoresL (cta * nw + wi) (imOf rd) l' 0 (ph r) b j :=
        fun r l' hr hne hs => hne (huniq _ r (hTl wi hwi l') hr (hStw wi l' r b j hs)).2
      by_cases hto : 32 * wi + l.val = o'
      · subst hto
        by_cases hqp : q ≤ p'
        · rw [htf _ 0 q ht (Nat.zero_le _) (by omega) b j hv (fun r _ hr hs => by
              have := huniq _ r ht (by omega) hs; omega), hstart _ b j hv,
            hF1 _ (by omega) b j hv (fun t' ht' q' hq' hs => by
              have := huniq t' q' (by omega) hq' hs; omega),
            hwf wi 0 q (Nat.zero_le _) (by omega) b j (fun r l' _ hr => hwnot r l' (by omega) (by omega))]
          exact hinit q wi l b j hq hwi hAe
        · rw [htf _ (p' + 1) q ht (by omega) (by omega) b j hv (fun r h1 h2 hs => by
              have := huniq _ r ht (by omega) hs; omega),
            hfm p' wi l hp hwi (ih p' (by omega) wi hwi l) b j (by
              have := hstA _ p' b j ht hp hst; rw [hvcl, hlnl] at this; exact this),
            hwf wi (p' + 1) q (by omega) (by omega) b j (fun r l' h1 h2 => hwnot r l' (by omega) (by omega))]
      · obtain ⟨hpq', hot, hw'⟩ := hlate _ o' q p' b j ht ho hq hp hAt hst hto
        have hwo : o' / 32 = wi := by rw [hw']; have : l.val < 32 := l.isLt; omega
        let lo : Lane := ln o'
        have hoeq : 32 * wi + lo.val = o' := by show 32 * wi + o' % 32 = o'; omega
        rw [htf _ 0 q ht (Nat.zero_le _) (by omega) b j hv (fun r _ hr hs => hto (huniq _ r ht (by omega) hs).1),
          hstart _ b j hv,
          hF2 o' _ hot (by omega) b j hv (fun t' h1 h2 q' hq' hs => by
            have := huniq t' q' (by omega) hq' hs; omega),
          (hT o' ho).1,
          htf o' (p' + 1) K ho (by omega) (Nat.le_refl _) b j hv (fun r h1 h2 hs => by
            have := huniq o' r ho (by omega) hs; omega)]
        have hpost := hfm p' wi lo hp hwi (ih p' hpq' wi hwi lo) b j (by
          have := hstA o' p' b j ho hp hst
          have hvo : vc o' = cta * nw + wi := by show cta * nw + o' / 32 = cta * nw + wi; rw [hwo]
          rw [hvo] at this; exact this)
        rw [hoeq] at hpost
        rw [hpost, hwf wi (p' + 1) q (by omega) (by omega) b j (fun r l' h1 h2 => hwnot r l' (by omega) (by omega))]
    · have hnone : ∀ t'' r, t'' < N → r < K → ¬ St t'' r b j := fun t'' r h1 h2 h3 => hex ⟨t'', r, h1, h2, h3⟩
      rw [htf _ 0 q ht (Nat.zero_le _) (by omega) b j hv (fun r _ hr => hnone _ r ht (by omega)),
        hstart _ b j hv,
        hF1 _ (by omega) b j hv (fun t' ht' q' hq' => hnone t' q' (by omega) hq'),
        hwf wi 0 q (Nat.zero_le _) (by omega) b j (fun r l' _ hr hs =>
          hnone _ r (hTl wi hwi l') (by omega) (hStw wi l' r b j hs))]
      exact hinit q wi l b j hq hwi hAe
  -- each phase, each warp's lanes settle on the warp machine's answers
  have hpost : ∀ q, q < K → ∀ wi, wi < nw → ∀ l : Lane, Post q wi l := by
    intro q
    induction q using Nat.strongRecOn with
    | ind q ih =>
    intro hq wi hwi
    have hc0 : ∀ l : Lane, (collCalls (tps (32 * wi + l.val) q).w (Ws (32 * wi + l.val)).obs.length).length =
        cnt wi q := by
      intro l
      cases q with
      | zero => exact congrArg List.length (collCalls_self (V (32 * wi + l.val)))
      | succ q' => exact (ih q' (by omega) (by omega) wi hwi l).2.1
    have hfq := (hph (32 * wi) (by omega) q hq)
    have e32 : vc (32 * wi) = cta * nw + wi := by
      show cta * nw + 32 * wi / 32 = _; rw [Nat.mul_div_cancel_left _ (by decide)]
    rw [e32] at hfq
    have hrel0 : ∀ l : Lane, RelW hw smem (A q (cta * nw + wi)) (fun _ _ => False) (rdI rd) (WSq q wi) (imOf rd) l
        (x (32 * wi + l.val)) (wpsP wi q) false (tps (32 * wi + l.val) q) := by
      intro l
      have ht := hTl wi hwi l
      have hR' := hR _ q ht (by omega)
      have fm : ∀ b j, A q (cta * nw + wi) l b j → (tps (32 * wi + l.val) q).w.mem.load
          ((x (32 * wi + l.val)).ptr b + off (UInt32.ofNat j)) 4 = some ((wpsP wi q).mem b j).bits := by
        intro b j hAe
        rw [hptr, hwP wi hwi q (by omega)]
        exact hrel q hq (fun p hp => ih p hp (by omega)) wi hwi l b j hAe
      have im : ∀ b j, rdI rd b j → (tps (32 * wi + l.val) q).w.mem.load
          ((x (32 * wi + l.val)).ptr b + off (UInt32.ofNat j)) 4 = some (UInt64.ofNat (imOf rd b j)) := by
        intro b j h; rw [hptr]; exact rdOk_load hR'.2.1 h
      have wr : ∀ a n, WSq q wi l a n → Writable (tps (32 * wi + l.val) q).w.mem a n := by
        rintro a n ⟨rfl, b, j, hAe, rfl⟩
        have hv := hA q _ l b j hAe
        refine ⟨.buf ⟨b, hv.2.1⟩, 4 * j, bufW_decode hv, ?_, by intro h; cases h⟩
        have := hsz' _ ((hInv _ (by omega)).1.trans hR'.1) _ q b j ht hq (by rw [hvcl, hlnl]; exact hAe)
        simpa [Mem.region] using this
      cases q with
      | zero =>
        have hasz : A0.size = smem := by simp [A0, ByteArray.size]
        exact ⟨fun _ => rfl, fun _ => rfl, fm, fun _ h => h.elim, im, wr, hasz, (fun h => by cases h),
          (hInv _ (by omega)).2.2, Nat.le_refl _, fun b => bufPs_get nbuf b, w32_ws0 nbuf m0⟩
      | succ q' =>
        obtain ⟨cl, hr⟩ := (ih q' (by omega) (by omega) wi hwi l).1
        refine ⟨hr.reg, hr.f, ?_, fun _ h => h.elim, ?_, ?_, hr.asz, (fun h => by cases h), hr.nan, hr.start, hr.ptr,
          hr.w32⟩
        · exact fm
        · exact im
        · exact wr
    have hlenL : ∀ l : Lane, cnt wi q ≤ (ans (32 * wi + l.val)).length := by
      intro l
      have ht := hTl wi hwi l
      rcases hlen _ ht with h | h
      · rw [← hc0 l]; exact Nat.le_trans (hpref _ ht q (by omega)).length_le h
      · refine Nat.le_trans ?_ h
        have e : (32 * wi + l.val) / 32 = wi := by have : l.val < 32 := l.isLt; omega
        rw [e]
        show cntOf _ _ _ ps _ q ≤ (jOf hw nbuf cta nw rd ps m0 wi).length
        rw [cntOf_eq]
        simp only [jOf]
        rw [show ps.length = q + (ps.length - q) by omega, List.range_add, List.flatMap_append, List.length_append]
        exact Nat.le_add_right _ _
    have hset' := lane_settle hw (cta * nw + wi) (imOf rd) smem (A q (cta * nw + wi)) (fun _ _ => False) (rdI rd)
      (WSq q wi) hw32 (ph q) 0 (wpsP wi q) false (fun l => x (32 * wi + l.val)) (fun l => tps (32 * wi + l.val) q)
      (fun l => tps (32 * wi + l.val) (q + 1)) (fun l => calls (32 * wi + l.val)) (cnt wi q)
      (fun l => (hS _ q (hTl wi hwi l) hq).1) hc0
      hlenL
      (fun l => by
        have ht := hTl wi hwi l
        refine ⟨rfl, ?_, ?_, by omega, rfl, rfl, ?_, ?_, ?_, ?_, fun _ _ h => h.elim, ?_⟩
        · have := (hxok _ ht).lv; rw [hlnl] at this; exact this
        · have := (hxok _ ht).cv; rw [hvcl] at this; exact this
        · show OkW hw _ _ _ (x (32 * wi + l.val)).ptr _ _ _ _ _ _ _ _
          rw [hptr]
          exact okW_of hw nbuf (cta * nw + wi) smem (A q (cta * nw + wi)) rd (WSq q wi) (fun l b j h => hA _ _ l b j h)
            (fun l b j h => hRd q wi l b j hwi h) (fun l b j h => ⟨rfl, b, j, h, rfl⟩) hrd (ph q) 0 hfq.1 hfq.2 _ _
        · have hr := hrel0 l
          exact ⟨hr.reg, hr.f, hr.fmem, hr.bmem, hr.imem, hr.wr, hr.asz, hr.sm, hr.nan, hr.start, hr.ptr, hr.w32⟩
        · show List.drop (collCalls (tps (32 * wi + l.val) q).w (Ws (32 * wi + l.val)).obs.length).length
            (List.take (cnt wi q) (ans (32 * wi + l.val)) ++ _) = _
          rw [hc0 l, List.drop_left' (by simp; exact Nat.min_eq_left (hlenL l)), List.append_nil]
        · intro l' b j _; rw [hptr]; exact notArena_bufW (hA _ _ _ b j (by assumption))
        · intro b j h; rw [hptr]
          obtain ⟨n, hn⟩ := Option.ne_none_iff_exists'.mp h
          exact notArena_bufW (hrd b j n hn).1)
      (fun l => hpref _ (hTl wi hwi l) (q + 1) hq)
      (fun j hj hall => hlaw q wi j hq hwi hj hall)
    intro l
    obtain ⟨li', e, rel, cc, _, _⟩ := hset' l
    have hli : li' = tps (32 * wi + l.val) (q + 1) := by
      rw [(hS _ q (hTl wi hwi l) hq).1] at e; exact (Option.some.inj e).symm
    subst hli
    have cc' : collCalls (tps (32 * wi + l.val) (q + 1)).w (Ws (32 * wi + l.val)).obs.length =
        collCalls (tps (32 * wi + l.val) q).w (Ws (32 * wi + l.val)).obs.length ++
          trace hw (cta * nw + wi) (imOf rd) (callLeaf (cta * nw + wi) l) 0 (ph q) (wpsP wi q) := cc
    refine ⟨⟨_, by rw [show wpsP wi (q + 1) = run hw (cta * nw + wi) (imOf rd) (ph q) (wpsP wi q) 0 from
      wpsOf_succ _ _ _ ps _ q hq]; exact rel⟩, ?_, cc'⟩
    rw [cc', List.length_append, hc0 l, trace_call, List.length_map]
    rfl
  -- each lane's collectives, phase after phase
  have hcallsq : ∀ wi, wi < nw → ∀ (l : Lane) q, q ≤ K →
      collCalls (tps (32 * wi + l.val) q).w (Ws (32 * wi + l.val)).obs.length =
        ((List.range q).flatMap fun q => trace hw (cta * nw + wi) (imOf rd) jLeaf 0 (ph q) (wpsP wi q)).map
          (callOne (cta * nw + wi) l) := by
    intro wi hwi l q
    induction q with
    | zero => intro _; exact collCalls_self (V (32 * wi + l.val))
    | succ q ih =>
      intro hq
      rw [(hpost q hq wi hwi l).2.2, ih (by omega), List.range_succ, List.flatMap_append, List.map_append, trace_call]
      simp
  refine ⟨fun t ht => hInv t ht, fun wi hwi l => ?_, fun q wi l b j hq hwi hAe => ?_, fun b j hv hn => ?_⟩
  · rw [(hT _ (hTl wi hwi l)).2, hcallsq wi hwi l K (Nat.le_refl _)]; rfl
  · have ht := hTl wi hwi l
    have hv := hA q _ l b j hAe
    have hAt : A q (vc (32 * wi + l.val)) (ln (32 * wi + l.val)) b j := by rw [hvcl, hlnl]; exact hAe
    rw [← hwpsK wi]
    by_cases hex : ∃ o' p', o' < N ∧ p' < K ∧ St o' p' b j
    · obtain ⟨o', p', ho, hp, hst⟩ := hex
      have huniq : ∀ t'' r, t'' < N → r < K → St t'' r b j → t'' = o' ∧ r = p' :=
        fun t'' r h1 h2 h3 => hone t'' o' r p' b j h1 ho h2 hp h3 hst
      have hwo : o' / 32 = wi := by
        by_cases hto : 32 * wi + l.val = o'
        · rw [← hto]; have : l.val < 32 := l.isLt; omega
        · obtain ⟨_, _, hw'⟩ := hlate _ o' q p' b j ht ho hq hp hAt hst hto
          rw [hw']; have : l.val < 32 := l.isLt; omega
      let lo : Lane := ln o'
      have hoeq : 32 * wi + lo.val = o' := by show 32 * wi + o' % 32 = o'; omega
      rw [hF2 o' N (by omega) (Nat.le_refl _) b j hv (fun t' h1 h2 q' hq' hs => by
            have := huniq t' q' (by omega) hq' hs; omega),
        (hT o' ho).1,
        htf o' (p' + 1) K ho (by omega) (Nat.le_refl _) b j hv (fun r h1 h2 hs => by
          have := huniq o' r ho (by omega) hs; omega)]
      have hp1 := hfm p' wi lo hp hwi (hpost p' hp wi hwi lo) b j (by
        have := hstA o' p' b j ho hp hst
        have hvo : vc o' = cta * nw + wi := by show cta * nw + o' / 32 = cta * nw + wi; rw [hwo]
        rw [hvo] at this; exact this)
      rw [hoeq] at hp1
      rw [hp1, hwf wi (p' + 1) K (by omega) (Nat.le_refl _) b j (fun r l' h1 h2 hs => by
        have := huniq _ r (hTl wi hwi l') (by omega) (hStw wi l' r b j hs); omega)]
    · have hnone : ∀ t'' r, t'' < N → r < K → ¬ St t'' r b j := fun t'' r h1 h2 h3 => hex ⟨t'', r, h1, h2, h3⟩
      rw [hF1 N (Nat.le_refl _) b j hv (fun t' ht' q' hq' => hnone t' q' ht' hq'),
        hwf wi 0 K (Nat.zero_le _) (Nat.le_refl _) b j (fun r l' _ hr hs =>
          hnone _ r (hTl wi hwi l') hr (hStw wi l' r b j hs))]
      exact hinit q wi l b j hq hwi hAe
  · exact hF1 N (Nat.le_refl _) b j hv (fun t' ht' q hq hs => hn (t' / 32) (ln t') (by omega)
      ((storesL_split (vc t') (imOf rd) (ln t') 0 hsp b j).mpr ⟨ps[q], List.getElem_mem hq, by
        rw [← hpq q hq]; exact hs⟩))

/-- The laws answer each collective of a pass with at least as many answers as
    it made collectives. -/
theorem calls_le_ans {calls : Nat → List (Intr × List V)} {snaps : Array ByteArray} {smem0 : ByteArray} {t : Nat}
    {as : List Answer} (h : answersOf calls snaps smem0 t = .ok as) : (calls t).length ≤ as.length := by
  rcases Nat.eq_zero_or_pos (calls t).length with h0 | hpos
  · rw [h0]; exact Nat.zero_le _
  · obtain ⟨z, _, hz, _⟩ := answersOf_get h (List.getElem?_eq_getElem (show (calls t).length - 1 < (calls t).length by omega))
    obtain ⟨hlt, _⟩ := List.getElem?_eq_some_iff.mp hz
    omega

/-- **A block runs its statement in phases**: a block of `nw` warps of a
    statement's kernel, the statement split into phases (`Split`), each phase's
    accesses those its map allows, leaves every word a lane of warp `w` touches
    in any phase as the warp machine's run of the statement for block
    `cta·nw + w` does from `m0`, where each word is stored in one phase by one
    thread, and any other thread touches it only in a later phase, after that
    thread in the block's pass and in its warp. The block's answers are what
    the laws give back from its threads' collectives, so they are the
    warp's (`law_bfly`, `pass_phases`). -/
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
    (hfits : ∀ wi, wi < nw → fits rd (cta * nw + wi) 0 s = true)
    (hRd : ∀ q wi l b j, wi < nw → A q (cta * nw + wi) l b j → rd b j = none)
    (hsz : ∀ q wi l b j, q < ps.length → wi < nw → A q (cta * nw + wi) l b j → 4 * j + 4 ≤ (w.mem.bufs b).size)
    (m0 : Mem) (hin : ∀ q wi (l : Lane) b j, q < ps.length → wi < nw → A q (cta * nw + wi) l b j →
      w.mem.load (bufW nbuf b j) 4 = m0.load (bufW nbuf b j) 4 ∧ (m0.load (bufW nbuf b j) 4).isSome)
    (hr : BlockRuns (Body.kernel name nbuf smem s nw) cta (32 * nw) (bufArgs nbuf) hw w w') :
    (∀ q wi (l : Lane) b j, q < ps.length → wi < nw → A q (cta * nw + wi) l b j →
      w'.mem.load (bufW nbuf b j) 4 =
        some ((@EWStmt.run F32 (F32.ops hw) _ s (ws0 nbuf m0) (cta * nw + wi) 0 (fun _ _ => 0) (imOf rd)).mem b j).bits) ∧
    (∀ b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) →
      (∀ wi l, wi < nw → ¬ StoresL (cta * nw + wi) (imOf rd) l 0 s b j) →
      w'.mem.load (bufW nbuf b j) 4 = w.mem.load (bufW nbuf b j) 4) ∧
    MemSame w.mem w'.mem ∧ RdOk nbuf w'.mem rd ∧ w'.mem.nan = hw.nan := by
  obtain ⟨steps, ans, w'', calls, stores, ans', hfold, hans, hset, rfl⟩ := hr
  obtain ⟨Ws, hW0, hWN, hWs⟩ := pass_runs (32 * nw) _ _ hfold
  have hWN' : Ws (32 * nw) = w'' := hWN
  have hA' : ∀ t, t < 32 * nw → answersOf calls (snapshots (32 * nw) (ByteArray.mk (Array.replicate smem 0)) calls stores)
      (ByteArray.mk (Array.replicate smem 0)) t = .ok (ans t) := fun t ht => by
    have := answers_at hans ht; rwa [hset t ht] at this
  obtain ⟨hInv, _, h1, h2⟩ := pass_phases hw hw32 name nbuf smem s ps hsp nw hnw hnw31 cta hcta hN A rd hacc hA hone
    hlate hfine w hrd hnan hfits hRd hsz m0 hin ans (fun t ht => answersOf_words (hA' t ht)) Ws hW0
    (fun t ht _ _ _ => ⟨steps, (hWs t ht).1⟩) calls (fun t ht => (hWs t ht).2)
    (fun t ht => Or.inl (calls_le_ans (hA' t ht)))
    (fun q wi j hq hwi hj hall => law_bfly hw (cta * nw + wi) (imOf rd) smem (fun l => ans (32 * wi + l.val)) wi
      (fun l => hA' _ (by have : l.val < 32 := l.isLt; omega)) (ps.getD q .skip) 0 _
      (allAt_of_fits rd (cta * nw + wi) _ 0 (fits_split rd _ 0 hsp (hfits wi hwi) _ (by
        rw [show ps.getD q .skip = ps[q] by simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hq]]
        exact List.getElem_mem hq))) _ j hj hall)
  rw [hWN'] at h1 h2
  have hIN := hInv (32 * nw) (Nat.le_refl _)
  rw [hWN'] at hIN
  refine ⟨fun q wi l b j hq hwi hAe => ?_, fun b j hv hn => ?_, ?_, ?_, ?_⟩
  · have hv := hA q _ l b j hAe
    show ({ w''.mem with arena := w.mem.arena } : Mem).load _ 4 = _
    rw [load_buf_arena _ _ (decode_bufW hv.1 hv.2.1 hv.2.2)]
    exact h1 q wi l b j hq hwi hAe
  · show ({ w''.mem with arena := w.mem.arena } : Mem).load _ 4 = _
    rw [load_buf_arena _ _ (decode_bufW hv.1 hv.2.1 hv.2.2)]
    exact h2 b j hv hn
  · show MemSame w.mem ({ w''.mem with arena := w.mem.arena } : Mem)
    exact memSame_rm hIN.1 w.mem.arena
  · show RdOk nbuf ({ w''.mem with arena := w.mem.arena } : Mem) rd
    exact (rdOk_arena w.mem.arena).mpr hIN.2.1
  · show ({ w''.mem with arena := w.mem.arena } : Mem).nan = _
    exact hIN.2.2

end Block

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
        some ((@EWStmt.run F32 (F32.ops hw) _ s (ws0 nbuf (kmem bufs)) c 0 (fun _ _ => 0) (imOf rd)).mem b j).bits) ∧
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
  obtain ⟨W, hfold, rfl⟩ := hr
  let run (c : Nat) : WSt F32 := @EWStmt.run F32 (F32.ops hw) _ s (ws0 nbuf (kmem bufs)) c 0 (fun _ _ => 0) (imOf rd)
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
        hfine Wp Wc hI4 hI5 (fun wi hwi => hfits _ (by omega)) (fun q wi l b j hwi h => hRd q _ l b j (by omega) h)
        (fun q wi l b j hq hwi h => by
          have := hI1.size (.buf ⟨b, (hA q _ l b j h).2.1⟩)
          simp only [Mem.region] at this
          rw [← this]; exact hin q _ l b j (by omega) hq h)
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

/-- A word a lane stores no other lane may touch. -/
theorem stores_own (cta : Nat) (A : Acc) (rd : Buf → Nat → Option Nat) (l l' : Lane) :
    ∀ (s : EWStmt) (i : Nat) (b : Buf) (j : Nat), AccOk cta A rd i s → StoresL cta (imOf rd) l i s b j →
      A l' b j → l' = l := by
  intro s
  induction s with
  | storeLane b0 ix n => intro i b j hacc ⟨hb, hj⟩ h; subst hb hj; exact (hacc l).2 l' h
  | storeLane0 b0 ix n => intro i b j hacc ⟨hl, hb, hj⟩ h; subst hl hb hj; exact hacc.2 l' h
  | seq x y ihx ihy =>
    intro i b j hacc h hA
    rcases h with h | h
    · exact ihx i b j hacc.1 h hA
    · exact ihy i b j hacc.2 h hA
  | forN n body ih => intro i b j hacc ⟨t, ht, h⟩ hA; exact ih t b j (hacc t ht) h hA
  | forM bb a body ih => intro i b j hacc ⟨t, ht, h⟩ hA; exact ih t b j (hacc t ht) h hA
  | iif _ x y ihx ihy =>
    intro i b j hacc h hA
    rcases h with h | h
    · exact ihx i b j hacc.1 h hA
    · exact ihy i b j hacc.2 h hA
  | iwhile _ _ _ body ih => intro i b j hacc h hA; exact ih i b j hacc h hA
  | locals _ body ih => intro i b j hacc h hA; exact ih i b j hacc h hA
  | _ => intro i b j _ h; exact absurd h id

/-- **A warp statement's launch runs the warp machine, block by block**:
    `grid` blocks of `nw` warps of the statement's kernel, warp `w` of block `b`
    standing as block `b·nw + w`, no block storing an element another touches,
    leave every element a warp's lanes may touch as the warp machine's run of
    the statement for that block does from the launch's memory:
    `launch_phases` of the statement as its one phase. -/
theorem launch_lockstep (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (name : String) (nbuf smem : Nat)
    (s : EWStmt) (nw : Nat) (hnw : 0 < nw) (hnw31 : nw < 2 ^ 31) (grid : Nat) (hgrid : grid * nw ≤ 2 ^ 32)
    (hN : 32 * nw ≤ 2 ^ 32) (A : Nat → Acc) (rd : Buf → Nat → Option Nat)
    (hacc : ∀ cta, cta < grid * nw → AccOk cta (A cta) rd 0 s)
    (hA : ∀ cta l b j, A cta l b j → b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32)
    (hdisj : ∀ c c' l l' b j, c < grid * nw → c' < grid * nw → c ≠ c' → A c l b j →
      ¬ StoresL c' (imOf rd) l' 0 s b j)
    (hfine : Fine (emitGo (fnBody (params nbuf) (Body.body nbuf s nw)) ⟨(params nbuf).length, 0, [], [], [], none⟩).2)
    (bufs bufs' : Nat → ByteArray) (hrd : RdOk nbuf (kmem bufs) rd)
    (hRd : ∀ cta l b j, cta < grid * nw → A cta l b j → rd b j = none)
    (hfits : ∀ cta, cta < grid * nw → fits rd cta 0 s = true)
    (hin : ∀ cta l b j, cta < grid * nw → A cta l b j → 4 * j + 4 ≤ (bufs b).size)
    (hr : LaunchRuns (Body.kernel name nbuf smem s nw) grid (32 * nw) (bufArgs nbuf) bufs bufs' hw) :
    ∀ cta l b j, cta < grid * nw → A cta l b j →
      ({ kmem bufs with bufs := bufs' } : Mem).load (bufW nbuf b j) 4 =
        some ((@EWStmt.run F32 (F32.ops hw) _ s (ws0 nbuf (kmem bufs)) cta 0 (fun _ _ => 0) (imOf rd)).mem b j).bits := by
  have h0 : ∀ q, q < [s].length → q = 0 := fun q hq => by simp at hq; omega
  have h := launch_phases hw hw32 name nbuf smem s [s] (Split.one s) nw hnw hnw31 grid hgrid hN (fun _ c => A c) rd
    (fun c q hc hq => by rw [h0 q hq]; exact hacc c hc) (fun q c l b j h => hA c l b j h)
    (fun c l l' q q' b j hc hq hq' h1 h2 => by
      rw [h0 q hq] at h1 ⊢; rw [h0 q' hq'] at h2 ⊢
      exact ⟨(stores_own c (A c) rd l' l s 0 b j (hacc c hc) h2
        (stores_acc c (A c) rd l s 0 b j (hacc c hc) h1)), rfl⟩)
    (fun c t o q p b j hc hq hp hAt hs hne => by
      rw [h0 p hp] at hs
      exact absurd (stores_own c (A c) rd o t s 0 b j (hacc c hc) hs hAt) hne)
    (fun c c' l l' q b j hc hc' hne _ hAq => hdisj c c' l l' b j hc hc' hne hAq)
    hfine bufs bufs' hrd (fun q c l b j hc h => hRd c l b j hc h) hfits (fun q c l b j hc _ h => hin c l b j hc h) hr
  exact fun cta l b j hc hAc => h.1 0 cta l b j hc (by simp) hAc

/-- `launch_lockstep` for one-warp blocks. -/
theorem launch_lockstep_one (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (name : String)
    (nbuf smem : Nat) (s : EWStmt) (grid : Nat) (hgrid : grid ≤ 2 ^ 32) (A : Nat → Acc) (rd : Buf → Nat → Option Nat)
    (hacc : ∀ cta, cta < grid → AccOk cta (A cta) rd 0 s)
    (hA : ∀ cta l b j, A cta l b j → b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32)
    (hdisj : ∀ c c' l l' b j, c < grid → c' < grid → c ≠ c' → A c l b j → ¬ StoresL c' (imOf rd) l' 0 s b j)
    (hfine : Fine (emitGo (fnBody (params nbuf) (Body.body nbuf s)) ⟨(params nbuf).length, 0, [], [], [], none⟩).2)
    (bufs bufs' : Nat → ByteArray) (hrd : RdOk nbuf (kmem bufs) rd)
    (hRd : ∀ cta l b j, cta < grid → A cta l b j → rd b j = none)
    (hfits : ∀ cta, cta < grid → fits rd cta 0 s = true)
    (hin : ∀ cta l b j, cta < grid → A cta l b j → 4 * j + 4 ≤ (bufs b).size)
    (hr : LaunchRuns (Body.kernel name nbuf smem s) grid 32 (bufArgs nbuf) bufs bufs' hw) :
    ∀ cta l b j, cta < grid → A cta l b j →
      ({ kmem bufs with bufs := bufs' } : Mem).load (bufW nbuf b j) 4 =
        some ((@EWStmt.run F32 (F32.ops hw) _ s (ws0 nbuf (kmem bufs)) cta 0 (fun _ _ => 0) (imOf rd)).mem b j).bits := by
  have e : ∀ c, c < grid * 1 ↔ c < grid := fun c => by rw [Nat.mul_one]
  intro cta l b j hc hAc
  exact launch_lockstep hw hw32 name nbuf smem s 1 (by decide) (by decide) grid (by omega) (by decide) A rd
    (fun c hc => hacc c ((e c).mp hc)) hA (fun c c' l l' b j hc hc' => hdisj c c' l l' b j ((e c).mp hc) ((e c').mp hc'))
    hfine bufs bufs' hrd (fun c l b j hc => hRd c l b j ((e c).mp hc)) (fun c hc => hfits c ((e c).mp hc))
    (fun c l b j hc => hin c l b j ((e c).mp hc)) hr cta l b j ((e cta).mpr hc) hAc

end AlgorithmLib.ML.BodyPhase
