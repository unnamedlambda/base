module
public import AlgorithmLib.Host.DevProg
meta import AlgorithmLib.Host.DevProg
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The race tracker is complete

The device model runs every operation when it is issued, and refuses an
operation that conflicts with an earlier one it is not ordered after. That
issue order is one the hardware could have chosen; what makes it the *only*
answer is that every two operations the hardware might reorder commute.

Here is why the tracker ensures it. An abstract tracker --- vector clocks per
party, snapshots events hold, each buffer's last write and reads since ---
accepts an operation only if it follows that last write, and, if it writes,
those reads. **`accepted_ordered`**: in any history it accepts, every two
conflicting operations are ordered by happens-before. Happens-before is read
off the clocks, and is transitive because knowledge only ever travels whole
(`KnowClosed`): a clock that knows an operation's own tick knows everything
the operation knew.

So the issued operations form a race-free device program (`DevProg.Disjoint`,
with the happens-before pairs as edges), and `DevProg.denote_linearisation`
says every order the hardware may pick computes what the issue order does.
-/

namespace AlgorithmLib.Device

/-- A vector clock: for each party, how many of its operations are known. -/
abbrev VC := Nat → Nat

def VC.join (a b : VC) : VC := fun x => max (a x) (b x)

def VC.bump (c : VC) (p : Nat) : VC := fun x => if x = p then c p + 1 else c x

/-- An issued operation: its party, its clock, what it reads and writes. -/
structure Op where
  party : Nat
  clock : VC
  rs : List Nat
  ws : List Nat

/-- The tracker's state: party clocks, clocks events hold, the operations
    issued so far, and each buffer's last write and reads since. -/
structure TSt where
  clk : Nat → VC
  snaps : List VC
  n : Nat
  op : Nat → Op
  lastW : Nat → Option Nat
  reads : Nat → List Nat

/-- The host: every operation is issued knowing what the host knows. -/
def host : Nat := 0

/-- Operation `i` happens before operation `j`: `j`'s clock knows `i`'s tick. -/
def TSt.hb (σ : TSt) (i j : Nat) : Prop :=
  (σ.op i).clock (σ.op i).party ≤ (σ.op j).clock (σ.op i).party

/-- A clock the state holds. -/
def TSt.Holds (σ : TSt) (C : VC) : Prop :=
  (∃ q, C = σ.clk q) ∨ C ∈ σ.snaps ∨ (∃ i < σ.n, C = (σ.op i).clock)

/-- What an operation with clock `c` on party `p` may do: follow the last write
    of everything it touches and, if it writes a buffer, the reads since. -/
def TSt.allows (σ : TSt) (p : Nat) (c : VC) (rs ws : List Nat) : Prop :=
  (∀ b ∈ rs ++ ws, ∀ w, σ.lastW b = some w → (σ.op w).clock (σ.op w).party ≤ c (σ.op w).party) ∧
  (∀ b ∈ ws, ∀ r ∈ σ.reads b, (σ.op r).clock (σ.op r).party ≤ c (σ.op r).party)

/-- Record an accepted operation. -/
def TSt.record (σ : TSt) (p : Nat) (c : VC) (rs ws : List Nat) : TSt where
  clk q := if q = p then c else σ.clk q
  snaps := σ.snaps
  n := σ.n + 1
  op k := if k = σ.n then { party := p, clock := c, rs, ws } else σ.op k
  lastW b := if b ∈ ws then some σ.n else σ.lastW b
  reads b := if b ∈ ws then [] else if b ∈ rs then σ.n :: σ.reads b else σ.reads b

/-- The tracker's moves: issue an operation on a party (checked), let a party
    learn a clock the state holds (a sync, an event wait, a stream fence), or
    let an event take a snapshot of one. -/
inductive Step : TSt → TSt → Prop
  | issue (σ : TSt) (p : Nat) (rs ws : List Nat) :
      σ.allows p ((VC.join (σ.clk p) (σ.clk host)).bump p) rs ws →
      Step σ (σ.record p ((VC.join (σ.clk p) (σ.clk host)).bump p) rs ws)
  | learn (σ : TSt) (p : Nat) (C : VC) : σ.Holds C →
      Step σ { σ with clk := fun q => if q = p then VC.join (σ.clk p) C else σ.clk q }
  | snap (σ : TSt) (C : VC) : σ.Holds C → Step σ { σ with snaps := C :: σ.snaps }

/-- Nothing issued, every clock zero. -/
def TSt.init : TSt where
  clk _ := fun _ => 0
  snaps := []
  n := 0
  op _ := { party := 0, clock := fun _ => 0, rs := [], ws := [] }
  lastW _ := none
  reads _ := []

/-- The states the tracker reaches. -/
inductive Reached : TSt → Prop
  | init : Reached TSt.init
  | step {σ σ'} : Reached σ → Step σ σ' → Reached σ'

-- ---------------------------------------------------------------------------
-- The clock invariants
-- ---------------------------------------------------------------------------

/-- **Knowledge travels whole**: a held clock that knows an operation's own
    tick knows everything that operation knew. -/
def KnowClosed (σ : TSt) : Prop :=
  ∀ C, σ.Holds C → ∀ i < σ.n,
    (σ.op i).clock (σ.op i).party ≤ C (σ.op i).party → ∀ x, (σ.op i).clock x ≤ C x

/-- **A party leads on its own ticks**: no held clock knows more of party `q`
    than `q`'s own clock. -/
def OwnLeads (σ : TSt) : Prop := ∀ C, σ.Holds C → ∀ q, C q ≤ σ.clk q q

/-- **The tracker accounts for every access**: a write to `b` is the last write
    or happens before it; a read of `b` is among the reads since, or happens
    before (or is) the last write. -/
def Accounts (σ : TSt) : Prop :=
  (∀ b w, σ.lastW b = some w → w < σ.n) ∧ (∀ b r, r ∈ σ.reads b → r < σ.n) ∧
  ∀ i < σ.n, ∀ b,
    (b ∈ (σ.op i).ws → ∃ w, σ.lastW b = some w ∧ (w = i ∨ σ.hb i w)) ∧
    (b ∈ (σ.op i).rs → i ∈ σ.reads b ∨ ∃ w, σ.lastW b = some w ∧ (w = i ∨ σ.hb i w))

/-- Two operations conflict when one writes what the other reads or writes. -/
def Conflict (a b : Op) : Prop :=
  (∃ x ∈ a.ws, x ∈ b.rs ∨ x ∈ b.ws) ∨ (∃ x ∈ b.ws, x ∈ a.rs)

/-- **Every two conflicting operations are ordered.** -/
def Ordered (σ : TSt) : Prop :=
  ∀ i j, i < j → j < σ.n → Conflict (σ.op i) (σ.op j) → σ.hb i j

def Good (σ : TSt) : Prop := KnowClosed σ ∧ OwnLeads σ ∧ Accounts σ ∧ Ordered σ

theorem hb_trans {σ : TSt} (hk : KnowClosed σ) {i k j : Nat} (hi : i < σ.n) (hkn : k < σ.n)
    (hj : j < σ.n) (h1 : σ.hb i k) (h2 : σ.hb k j) : σ.hb i j := by
  have hck : ∀ x, (σ.op i).clock x ≤ (σ.op k).clock x :=
    hk _ (Or.inr (Or.inr ⟨k, hkn, rfl⟩)) i hi h1
  have hcj : ∀ x, (σ.op k).clock x ≤ (σ.op j).clock x :=
    hk _ (Or.inr (Or.inr ⟨j, hj, rfl⟩)) k hkn h2
  exact Nat.le_trans (hck _) (hcj _)

theorem good_init : Good TSt.init := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro C _ i hi; simp [TSt.init] at hi
  · intro C hC q
    rcases hC with ⟨q', rfl⟩ | hC | ⟨i, hi, _⟩
    · simp [TSt.init]
    · simp [TSt.init] at hC
    · simp [TSt.init] at hi
  · refine ⟨fun b w h => by simp [TSt.init] at h, fun b r h => by simp [TSt.init] at h, ?_⟩
    intro i hi; simp [TSt.init] at hi
  · intro i j _ hj; simp [TSt.init] at hj

-- ---------------------------------------------------------------------------
-- Snapshots and learning keep the invariants
-- ---------------------------------------------------------------------------

theorem good_snap {σ : TSt} {C : VC} (hC : σ.Holds C) (h : Good σ) :
    Good { σ with snaps := C :: σ.snaps } := by
  obtain ⟨hk, ho, ha, hord⟩ := h
  have back : ∀ D, TSt.Holds { σ with snaps := C :: σ.snaps } D → σ.Holds D := by
    rintro D (⟨q, rfl⟩ | hD | ⟨i, hi, rfl⟩)
    · exact Or.inl ⟨q, rfl⟩
    · rcases List.mem_cons.mp hD with rfl | hD
      · exact hC
      · exact Or.inr (Or.inl hD)
    · exact Or.inr (Or.inr ⟨i, hi, rfl⟩)
  exact ⟨fun D hD => hk D (back D hD), fun D hD => ho D (back D hD), ha, hord⟩

theorem good_learn {σ : TSt} {p : Nat} {C : VC} (hC : σ.Holds C) (h : Good σ) :
    Good { σ with clk := fun q => if q = p then VC.join (σ.clk p) C else σ.clk q } := by
  obtain ⟨hk, ho, ha, hord⟩ := h
  -- what the new state holds: an old clock, or the joined one
  have back : ∀ D, TSt.Holds { σ with clk := fun q => if q = p then VC.join (σ.clk p) C else σ.clk q } D →
      σ.Holds D ∨ D = VC.join (σ.clk p) C := by
    rintro D (⟨q, rfl⟩ | hD | ⟨i, hi, rfl⟩)
    · by_cases hq : q = p
      · subst hq; exact Or.inr (by simp)
      · exact Or.inl (Or.inl ⟨q, by simp [hq]⟩)
    · exact Or.inl (Or.inr (Or.inl hD))
    · exact Or.inl (Or.inr (Or.inr ⟨i, hi, rfl⟩))
  have hcp : C p ≤ σ.clk p p := ho C hC p
  refine ⟨?_, ?_, ha, hord⟩
  · intro D hD i hi hle x
    rcases back D hD with hD | rfl
    · exact hk D hD i hi hle x
    · simp only [VC.join] at hle ⊢
      rcases (show _ ∨ _ from by omega : (σ.op i).clock (σ.op i).party ≤ σ.clk p (σ.op i).party ∨
          (σ.op i).clock (σ.op i).party ≤ C (σ.op i).party) with h1 | h1
      · exact Nat.le_trans (hk _ (Or.inl ⟨p, rfl⟩) i hi h1 x) (Nat.le_max_left _ _)
      · exact Nat.le_trans (hk _ hC i hi h1 x) (Nat.le_max_right _ _)
  · intro D hD q
    have hq' : σ.clk q q ≤ (if q = p then VC.join (σ.clk p) C else σ.clk q) q := by
      by_cases hq : q = p
      · subst hq; simp only [if_pos rfl, VC.join]; exact Nat.le_max_left _ _
      · simp [hq]
    rcases back D hD with hD | rfl
    · exact Nat.le_trans (ho D hD q) hq'
    · by_cases hq : q = p
      · subst hq; simp only [if_pos rfl, VC.join]; exact Nat.le_refl _
      · simp only [VC.join, hq, if_false]
        exact Nat.max_le.mpr ⟨ho _ (Or.inl ⟨p, rfl⟩) q, ho C hC q⟩

-- ---------------------------------------------------------------------------
-- Issuing an accepted operation keeps them
-- ---------------------------------------------------------------------------

section Issue
variable {σ : TSt} {p : Nat} {rs ws : List Nat} {c : VC}

theorem record_op_old {k : Nat} (hk : k < σ.n) : (σ.record p c rs ws).op k = σ.op k := by
  simp [TSt.record, Nat.ne_of_lt hk]

theorem record_op_new : (σ.record p c rs ws).op σ.n = { party := p, clock := c, rs, ws } := by
  simp [TSt.record]

theorem record_holds {D : VC} (hD : (σ.record p c rs ws).Holds D) : σ.Holds D ∨ D = c := by
  rcases hD with ⟨q, rfl⟩ | hD | ⟨i, hi, rfl⟩
  · by_cases hq : q = p
    · subst hq; exact Or.inr (by simp [TSt.record])
    · exact Or.inl (Or.inl ⟨q, by simp [TSt.record, hq]⟩)
  · exact Or.inl (Or.inr (Or.inl hD))
  · by_cases hin : i = σ.n
    · subst hin; exact Or.inr (by rw [record_op_new])
    · have hi' : i < σ.n := by simp [TSt.record] at hi; omega
      exact Or.inl (Or.inr (Or.inr ⟨i, hi', by rw [record_op_old hi']⟩))

theorem record_hb_old {i j : Nat} (hi : i < σ.n) (hj : j < σ.n) :
    (σ.record p c rs ws).hb i j ↔ σ.hb i j := by
  simp only [TSt.hb, record_op_old hi, record_op_old hj]

theorem record_hb_new {i : Nat} (hi : i < σ.n)
    (h : (σ.op i).clock (σ.op i).party ≤ c (σ.op i).party) : (σ.record p c rs ws).hb i σ.n := by
  simp only [TSt.hb, record_op_old hi, record_op_new]; exact h

theorem good_issue (hc : c = (VC.join (σ.clk p) (σ.clk host)).bump p)
    (hal : σ.allows p c rs ws) (h : Good σ) : Good (σ.record p c rs ws) := by
  obtain ⟨hk, ho, ha, hord⟩ := h
  have hF1 : ∀ x, σ.clk p x ≤ c x ∧ σ.clk host x ≤ c x := by
    intro x; subst hc
    by_cases hx : x = p
    · subst hx; simp [VC.bump, VC.join]; omega
    · simp only [VC.bump, VC.join, if_neg hx]; omega
  have hhost : σ.clk host p ≤ σ.clk p p := ho _ (Or.inl ⟨host, rfl⟩) p
  have hF2 : c p = σ.clk p p + 1 := by subst hc; simp [VC.bump, VC.join]; omega
  have hlead : ∀ D, σ.Holds D → D p < c p := fun D hD => by have := ho D hD p; omega
  -- knowledge stays closed
  have hk' : KnowClosed (σ.record p c rs ws) := by
    intro D hD i hi hle x
    by_cases hin : i = σ.n
    · subst hin
      rw [record_op_new] at hle ⊢
      rcases record_holds hD with hD | rfl
      · exact absurd hle (Nat.not_le.mpr (hlead D hD))
      · exact Nat.le_refl _
    · have hi' : i < σ.n := by simp [TSt.record] at hi; omega
      rw [record_op_old hi'] at hle ⊢
      rcases record_holds hD with hD | rfl
      · exact hk D hD i hi' hle x
      · by_cases hp : (σ.op i).party = p
        · have h1 : (σ.op i).clock p ≤ σ.clk p p := ho _ (Or.inr (Or.inr ⟨i, hi', rfl⟩)) p
          exact Nat.le_trans (hk _ (Or.inl ⟨p, rfl⟩) i hi' (by rw [hp]; exact h1) x) (hF1 x).1
        · have hce : D (σ.op i).party
              = max (σ.clk p (σ.op i).party) (σ.clk host (σ.op i).party) := by
            subst hc; simp [VC.bump, VC.join, hp]
          rw [hce] at hle
          rcases (show _ ∨ _ from by omega : (σ.op i).clock (σ.op i).party ≤ σ.clk p (σ.op i).party ∨
              (σ.op i).clock (σ.op i).party ≤ σ.clk host (σ.op i).party) with h1 | h1
          · exact Nat.le_trans (hk _ (Or.inl ⟨p, rfl⟩) i hi' h1 x) (hF1 x).1
          · exact Nat.le_trans (hk _ (Or.inl ⟨host, rfl⟩) i hi' h1 x) (hF1 x).2
  -- a party still leads on its own ticks
  have ho' : OwnLeads (σ.record p c rs ws) := by
    intro D hD q
    have hclk : (σ.record p c rs ws).clk q = if q = p then c else σ.clk q := rfl
    rw [hclk]
    rcases record_holds hD with hD' | rfl
    · by_cases hq : q = p
      · subst hq; simp; exact Nat.le_of_lt (hlead D hD')
      · simp only [hq, if_false]; exact ho D hD' q
    · by_cases hq : q = p
      · subst hq; simp
      · simp only [hq, if_false]
        have e : D q = max (σ.clk p q) (σ.clk host q) := by subst hc; simp [VC.bump, VC.join, hq]
        rw [e]
        exact Nat.max_le.mpr ⟨ho _ (Or.inl ⟨p, rfl⟩) q, ho _ (Or.inl ⟨host, rfl⟩) q⟩
  have trans' : ∀ i w, i < σ.n → w < σ.n → σ.hb i w →
      (σ.op w).clock (σ.op w).party ≤ c (σ.op w).party → (σ.record p c rs ws).hb i σ.n :=
    fun i w hi hw h1 h2 => hb_trans hk' (by simp [TSt.record]; omega) (by simp [TSt.record]; omega)
      (by simp [TSt.record]) ((record_hb_old hi hw).mpr h1) (record_hb_new hw h2)
  obtain ⟨hlw, hrd, hacc⟩ := ha
  refine ⟨hk', ho', ⟨?_, ?_, ?_⟩, ?_⟩
  · intro b w h
    simp only [TSt.record] at h ⊢
    split at h
    · cases h; omega
    · have := hlw b w h; omega
  · intro b r h
    simp only [TSt.record] at h ⊢
    split at h
    · cases h
    · split at h
      · rcases List.mem_cons.mp h with rfl | h
        · omega
        · have := hrd b r h; omega
      · have := hrd b r h; omega
  · intro i hi b
    by_cases hin : i = σ.n
    · subst hin
      rw [record_op_new]
      refine ⟨fun hb => ⟨σ.n, by simp [TSt.record, hb], Or.inl rfl⟩, fun hb => ?_⟩
      by_cases hw : b ∈ ws
      · exact Or.inr ⟨σ.n, by simp [TSt.record, hw], Or.inl rfl⟩
      · exact Or.inl (by simp [TSt.record, hw, hb])
    · have hi' : i < σ.n := by simp [TSt.record] at hi; omega
      rw [record_op_old hi']
      obtain ⟨hwr, hrr⟩ := hacc i hi' b
      refine ⟨fun hb => ?_, fun hb => ?_⟩
      · obtain ⟨w, hwb, hwi⟩ := hwr hb
        by_cases hbw : b ∈ ws
        · refine ⟨σ.n, by simp [TSt.record, hbw], Or.inr ?_⟩
          have hnew := hal.1 b (List.mem_append_right _ hbw) w hwb
          rcases hwi with rfl | hwi
          · exact record_hb_new hi' hnew
          · exact trans' i w hi' (hlw b w hwb) hwi hnew
        · refine ⟨w, by simp [TSt.record, hbw, hwb], ?_⟩
          rcases hwi with rfl | hwi
          · exact Or.inl rfl
          · exact Or.inr ((record_hb_old hi' (hlw b w hwb)).mpr hwi)
      · by_cases hbw : b ∈ ws
        · refine Or.inr ⟨σ.n, by simp [TSt.record, hbw], Or.inr ?_⟩
          rcases hrr hb with hr | ⟨w, hwb, hwi⟩
          · exact record_hb_new hi' (hal.2 b hbw i hr)
          · have hnew := hal.1 b (List.mem_append_right _ hbw) w hwb
            rcases hwi with rfl | hwi
            · exact record_hb_new hi' hnew
            · exact trans' i w hi' (hlw b w hwb) hwi hnew
        · rcases hrr hb with hr | ⟨w, hwb, hwi⟩
          · refine Or.inl ?_
            simp only [TSt.record, hbw, if_false]
            split
            · exact List.mem_cons_of_mem _ hr
            · exact hr
          · refine Or.inr ⟨w, by simp [TSt.record, hbw, hwb], ?_⟩
            rcases hwi with rfl | hwi
            · exact Or.inl rfl
            · exact Or.inr ((record_hb_old hi' (hlw b w hwb)).mpr hwi)
  · intro i j hij hj hcon
    by_cases hjn : j = σ.n
    · subst hjn
      rw [record_op_old hij, record_op_new] at hcon
      rcases hcon with ⟨x, hx, hxo⟩ | ⟨x, hx, hxi⟩
      · obtain ⟨w, hwb, hwi⟩ := (hacc i hij x).1 hx
        have hnew := hal.1 x (by rcases hxo with h | h <;> simp [h]) w hwb
        rcases hwi with rfl | hwi
        · exact record_hb_new hij hnew
        · exact trans' i w hij (hlw x w hwb) hwi hnew
      · rcases (hacc i hij x).2 hxi with hr | ⟨w, hwb, hwi⟩
        · exact record_hb_new hij (hal.2 x hx i hr)
        · have hnew := hal.1 x (List.mem_append_right _ hx) w hwb
          rcases hwi with rfl | hwi
          · exact record_hb_new hij hnew
          · exact trans' i w hij (hlw x w hwb) hwi hnew
    · have hj' : j < σ.n := by simp [TSt.record] at hj; omega
      rw [record_op_old (Nat.lt_trans hij hj'), record_op_old hj'] at hcon
      exact (record_hb_old (Nat.lt_trans hij hj') hj').mpr (hord i j hij hj' hcon)

end Issue

/-- **The tracker's invariants hold in every state it reaches.** -/
theorem reached_good {σ : TSt} (h : Reached σ) : Good σ := by
  induction h with
  | init => exact good_init
  | step _ hs ih =>
    cases hs with
    | issue p rs ws hal => exact good_issue rfl hal ih
    | learn p C hC => exact good_learn hC ih
    | snap C hC => exact good_snap hC ih

/-- **Every two conflicting operations the tracker accepted are ordered.** -/
theorem accepted_ordered {σ : TSt} (h : Reached σ) : Ordered σ := (reached_good h).2.2.2

-- ---------------------------------------------------------------------------
-- So the issued operations are a race-free device program
-- ---------------------------------------------------------------------------

/-- The happens-before pairs among the issued operations, as edges. -/
def TSt.edges (σ : TSt) : List (Nat × Nat) :=
  (List.range σ.n).flatMap fun j => (List.range j).filterMap fun i =>
    if (σ.op i).clock (σ.op i).party ≤ (σ.op j).clock (σ.op i).party then some (i, j) else none

theorem mem_edges {σ : TSt} {i j : Nat} (hij : i < j) (hj : j < σ.n) (h : σ.hb i j) :
    (i, j) ∈ σ.edges := by
  simp only [TSt.edges, List.mem_flatMap, List.mem_range, List.mem_filterMap]
  exact ⟨j, hj, i, hij, by simp only [TSt.hb] at h; simp [h]⟩

theorem edge_forward {σ : TSt} {i j : Nat} (h : (i, j) ∈ σ.edges) : i < j := by
  simp only [TSt.edges, List.mem_flatMap, List.mem_range, List.mem_filterMap] at h
  obtain ⟨j', _, i', hi', he⟩ := h
  split at he
  · simp only [Option.some.injEq, Prod.mk.injEq] at he
    obtain ⟨rfl, rfl⟩ := he; exact hi'
  · cases he

theorem reach_forward {σ : TSt} {u v : Nat} (h : Reach σ.edges u v) : u < v := by
  induction h with
  | edge he => exact edge_forward he
  | trans _ _ h1 h2 => omega

/-- **The issue order keeps the edges.** -/
theorem resp_range (σ : TSt) : Resp σ.edges (List.range σ.n) := by
  intro u v hr hs
  have huv := reach_forward hr
  -- in `range`, what comes first is smaller
  have key : ∀ (l : List Nat), l.Pairwise (· < ·) → [v, u].Sublist l → v < u := by
    intro l hl hs
    have := hl.sublist hs
    simpa using this
  exact absurd (key _ (List.pairwise_lt_range) hs) (by omega)

/-- A device program whose steps are the issued operations: the same number,
    each with the operation's footprint, and the happens-before pairs as its
    edges. -/
def Matches (σ : TSt) (P : DevProg) : Prop :=
  P.steps.length = σ.n ∧ P.edges = σ.edges ∧
    ∀ i s, P.steps[i]? = some s → s.reads = (σ.op i).rs ∧ s.writes = (σ.op i).ws

/-- **What the tracker accepted is race-free**: two steps no path orders touch
    no buffer one of them writes. -/
theorem accepted_disjoint {σ : TSt} (h : Reached σ) {P : DevProg} (hm : Matches σ P) :
    P.Disjoint := by
  obtain ⟨hlen, hedges, hfp⟩ := hm
  have hord := accepted_ordered h
  intro u v s t huv hnu hnv hs ht
  have hu : u < σ.n := hlen ▸ (List.getElem?_eq_some_iff.mp hs).1
  have hv : v < σ.n := hlen ▸ (List.getElem?_eq_some_iff.mp ht).1
  obtain ⟨hsr, hsw⟩ := hfp u s hs
  obtain ⟨htr, htw⟩ := hfp v t ht
  rw [hedges] at hnu hnv
  -- a conflict would put them in order, and so on a path
  have noConflict : ¬ Conflict (σ.op u) (σ.op v) := by
    intro hc
    rcases Nat.lt_or_gt_of_ne huv with huv' | huv'
    · exact hnu (.edge (mem_edges huv' hv (hord u v huv' hv hc)))
    · have hc' : Conflict (σ.op v) (σ.op u) := by
        rcases hc with ⟨x, hx, hxo⟩ | ⟨x, hx, hxi⟩
        · rcases hxo with h1 | h1
          · exact Or.inr ⟨x, hx, h1⟩
          · exact Or.inl ⟨x, h1, Or.inr hx⟩
        · exact Or.inl ⟨x, hx, Or.inl hxi⟩
      exact hnv (.edge (mem_edges huv' hu (hord v u huv' hu hc')))
  refine ⟨fun b hb => ⟨fun hbt => ?_, fun hbt => ?_⟩, fun b hb hbs => ?_⟩
  · rw [hsw] at hb; rw [htr] at hbt; exact noConflict (Or.inl ⟨b, hb, Or.inl hbt⟩)
  · rw [hsw] at hb; rw [htw] at hbt; exact noConflict (Or.inl ⟨b, hb, Or.inr hbt⟩)
  · rw [htw] at hb; rw [hsr] at hbs; exact noConflict (Or.inr ⟨b, hb, hbs⟩)

/-- **Every order the hardware may run the accepted operations in computes what
    running them as issued computes** --- any order that runs each once and
    keeps the happens-before edges the streams, events and syncs impose. -/
theorem accepted_any_order {σ : TSt} (h : Reached σ) {P : DevProg} (hm : Matches σ P)
    (o : List Nat) (hp : o.Perm (List.range σ.n)) (ho : Resp P.edges o) (m : DevMem) :
    denote P.stepAt o m = denote P.stepAt (List.range σ.n) m := by
  have hdis := accepted_disjoint h hm
  have hr : Resp P.edges (List.range σ.n) := hm.2.1 ▸ resp_range σ
  exact P.denote_linearisation hdis o _ hp (hp.nodup_iff.mpr List.nodup_range) ho hr m

end AlgorithmLib.Device
