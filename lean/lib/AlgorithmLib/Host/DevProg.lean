module
public import AlgorithmLib.Host.World
meta import AlgorithmLib.Host.World
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Host.DevProg` — a device program, and why its order is not a choice

A device program is steps --- launches, vendor calls, copies --- with the order
some of them must keep: an edge `(u, v)` says `u` runs before `v`. What a
program computes should not depend on which admissible order the host, the
driver or a captured graph picks, and **`denote_linearisation`** says when it
does not: if every two steps no path orders are independent (they commute),
every order that keeps the edges computes the same thing.

Independence is stated as commutation, so the theorem holds for any memory
model. For device memory as bytes per buffer, **`BufStep.commute`** gives it
from footprints: a step that writes only its write set and reads only its read
set commutes with one whose sets it does not touch --- the race tracker's
condition.

A `KernelDef` is a kernel with, for each binding of its parameters, the step
it performs. Each development supplies its own (`.ew` kernels through the warp
machine, LZ4 through `sstep`); the device program is where they meet.
-/

namespace AlgorithmLib.Device

/-- `u` reaches `v` along edges. -/
inductive Reach (E : List (Nat × Nat)) : Nat → Nat → Prop
  | edge {u v} : (u, v) ∈ E → Reach E u v
  | trans {u v w} : Reach E u v → Reach E v w → Reach E u w

/-- An order keeps `E` when nothing runs before a step that reaches it. -/
def Resp (E : List (Nat × Nat)) (o : List Nat) : Prop :=
  ∀ u v, Reach E u v → ¬ [v, u].Sublist o

/-- Steps no path orders commute. -/
def RaceFree {M : Type} (E : List (Nat × Nat)) (f : Nat → M → M) : Prop :=
  ∀ u v, u ≠ v → ¬ Reach E u v → ¬ Reach E v u → ∀ m, f u (f v m) = f v (f u m)

/-- Run the steps in the order `o`. -/
def denote {M : Type} (f : Nat → M → M) (o : List Nat) (m : M) : M :=
  o.foldl (fun m i => f i m) m

theorem Resp.tail {E : List (Nat × Nat)} {a : Nat} {o : List Nat} (h : Resp E (a :: o)) :
    Resp E o := fun u v hr hs => h u v hr (hs.cons a)

theorem Resp.erase {E : List (Nat × Nat)} {a : Nat} {pre post : List Nat}
    (h : Resp E (pre ++ a :: post)) : Resp E (pre ++ post) := fun u v hr hs =>
  h u v hr (hs.trans ((List.sublist_cons_self a post).append_left pre))

/-- A step every step before it commutes with can run first. -/
theorem denote_bubble {M : Type} (f : Nat → M → M) (a : Nat) :
    ∀ (pre post : List Nat) (m : M), (∀ x ∈ pre, ∀ m, f a (f x m) = f x (f a m)) →
      denote f (pre ++ a :: post) m = denote f (a :: (pre ++ post)) m
  | [], post, m, _ => rfl
  | x :: pre, post, m, h => by
      have ih := denote_bubble f a pre post (f x m) (fun y hy => h y (by simp [hy]))
      simp only [denote, List.cons_append, List.foldl_cons] at ih ⊢
      rw [ih, h x (by simp)]

/-- **Every order that keeps the edges computes the same thing**, when steps
    no path orders commute. The first step of one order has nothing before it;
    in the other, everything before it is unordered with it, so it moves to the
    front one commutation at a time; the rest is the same claim, shorter. -/
theorem denote_linearisation {M : Type} (E : List (Nat × Nat)) (f : Nat → M → M)
    (hrf : RaceFree E f) :
    ∀ (o₁ o₂ : List Nat), o₁.Perm o₂ → o₁.Nodup → Resp E o₁ → Resp E o₂ →
      ∀ m, denote f o₁ m = denote f o₂ m := by
  intro o₁
  induction o₁ with
  | nil => intro o₂ hp _ _ _ m; rw [List.nil_perm.mp hp]
  | cons a o₁ ih =>
    intro o₂ hp hnd hr₁ hr₂ m
    have ha : a ∈ o₂ := hp.subset (List.mem_cons_self ..)
    obtain ⟨pre, post, rfl⟩ := List.append_of_mem ha
    have hnd₂ : (pre ++ a :: post).Nodup := hp.nodup_iff.mp hnd
    -- everything before `a` in the second order is unordered with it
    have hind : ∀ x ∈ pre, ∀ m, f a (f x m) = f x (f a m) := by
      intro x hx
      have hxa : x ≠ a := by
        intro e; subst e
        exact (List.nodup_append.mp hnd₂).2.2 x hx x (List.mem_cons_self ..) rfl
      refine hrf a x (Ne.symm hxa) ?_ ?_
      · intro hax
        exact hr₂ a x hax ((List.singleton_sublist.mpr hx).append (List.sublist_append_left [a] post)
          |>.trans (by simp))
      · intro hxa'
        have hx₁ : x ∈ o₁ := by
          have : x ∈ a :: o₁ := hp.symm.subset (List.mem_append_left _ hx)
          rcases List.mem_cons.mp this with h | h
          · exact absurd h hxa
          · exact h
        exact hr₁ x a hxa' ((List.singleton_sublist.mpr hx₁).cons₂ a)
    rw [denote_bubble f a pre post m hind]
    simp only [denote, List.foldl_cons]
    exact ih (pre ++ post) (hp.trans List.perm_middle).cons_inv
      (List.nodup_cons.mp hnd).2 hr₁.tail hr₂.erase (f a m)

-- ---------------------------------------------------------------------------
-- Device memory as bytes per buffer
-- ---------------------------------------------------------------------------

/-- Device memory: the bytes of each buffer, by id. -/
abbrev DevMem := Nat → ByteArray

/-- A step on device memory with its footprint: it changes nothing outside
    `writes`, and what it writes depends on nothing outside `reads` and
    `writes`. -/
structure BufStep where
  run : DevMem → DevMem
  reads : List Nat
  writes : List Nat
  frame : ∀ m b, b ∉ writes → run m b = m b
  local_ : ∀ m m', (∀ b ∈ reads, m b = m' b) → (∀ b ∈ writes, m b = m' b) →
    ∀ b ∈ writes, run m b = run m' b

/-- **Steps whose footprints do not conflict commute**: neither writes what the
    other reads or writes. -/
theorem BufStep.commute (s t : BufStep)
    (hst : ∀ b ∈ s.writes, b ∉ t.reads ∧ b ∉ t.writes) (hts : ∀ b ∈ t.writes, b ∉ s.reads)
    (m : DevMem) : s.run (t.run m) = t.run (s.run m) := by
  funext b
  by_cases hs : b ∈ s.writes
  · have hbt := (hst b hs).2
    rw [t.frame (s.run m) b hbt]
    exact s.local_ (t.run m) m
      (fun c hc => t.frame m c (fun hct => hts c hct hc))
      (fun c hc => t.frame m c (hst c hc).2) b hs
  · rw [s.frame (t.run m) b hs]
    by_cases ht : b ∈ t.writes
    · exact t.local_ m (s.run m)
        (fun c hc => (s.frame m c (fun hcs => (hst c hcs).1 hc)).symm)
        (fun c hc => (s.frame m c (fun hcs => (hst c hcs).2 hc)).symm) b ht
    · rw [t.frame m b ht, t.frame (s.run m) b ht, s.frame m b hs]

/-- A kernel: its code, in whatever language its development proves things
    about, and for each binding of its parameters to buffers, the step it
    performs there. -/
structure KernelDef (C : Type) where
  name : String
  code : C
  step : List Nat → BufStep

/-- A device program over byte buffers: its steps by id and its edges. -/
structure DevProg where
  steps : List BufStep
  edges : List (Nat × Nat)

def DevProg.stepAt (P : DevProg) (i : Nat) : DevMem → DevMem :=
  match P.steps[i]? with
  | some s => s.run
  | none => id

/-- No two steps that no path orders touch a buffer one of them writes. -/
def DevProg.Disjoint (P : DevProg) : Prop :=
  ∀ u v s t, u ≠ v → ¬ Reach P.edges u v → ¬ Reach P.edges v u →
    P.steps[u]? = some s → P.steps[v]? = some t →
      (∀ b ∈ s.writes, b ∉ t.reads ∧ b ∉ t.writes) ∧ (∀ b ∈ t.writes, b ∉ s.reads)

theorem DevProg.raceFree (P : DevProg) (h : P.Disjoint) : RaceFree P.edges P.stepAt := by
  intro u v huv hu hv m
  unfold DevProg.stepAt
  cases hs : P.steps[u]? with
  | none => rfl
  | some s =>
    cases ht : P.steps[v]? with
    | none => rfl
    | some t =>
      obtain ⟨h1, h2⟩ := h u v s t huv hu hv hs ht
      exact BufStep.commute s t h1 h2 m

/-- **A race-free device program means one thing**: any two orders that run
    each step once and keep the edges leave the same memory. -/
theorem DevProg.denote_linearisation (P : DevProg) (h : P.Disjoint) (o₁ o₂ : List Nat)
    (hp : o₁.Perm o₂) (hnd : o₁.Nodup) (h₁ : Resp P.edges o₁) (h₂ : Resp P.edges o₂) (m : DevMem) :
    denote P.stepAt o₁ m = denote P.stepAt o₂ m :=
  AlgorithmLib.Device.denote_linearisation P.edges P.stepAt (P.raceFree h) o₁ o₂ hp hnd h₁ h₂ m

end AlgorithmLib.Device
