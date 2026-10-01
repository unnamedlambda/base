module
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic

@[expose] public section

/-!
# A registry: every value of a type, listed once

The runtime's entry points, the laws a floating-point claim rests on, the
assumptions a scan reports — each is a finite set whose completeness matters:
a report that enumerates one cannot describe a stack as resting on less than it
does only if the enumeration misses nothing. `Enum` is that property as a
class, so each registry states it the same way and a proof about "every
registered thing" is a proof over `Enum.all`.

It is Mathlib's `Fintype` with the list kept, which is what the generators and
reports iterate, and without Mathlib, which the executable side does not import.
-/

namespace AlgorithmLib

/-- Every value of `α`, each once. -/
class Enum (α : Type) [DecidableEq α] where
  all : List α
  complete : ∀ a, a ∈ all
  nodup : all.Nodup

namespace Enum

variable {α : Type} [DecidableEq α] [Enum α]

/-- How many values there are. -/
def card (α : Type) [DecidableEq α] [Enum α] : Nat := (Enum.all : List α).length

/-- A value's position in the registry: the id a registry-numbered thing gets. -/
def index (a : α) : Nat := (Enum.all : List α).idxOf a

theorem index_lt (a : α) : index a < card α :=
  List.idxOf_lt_length_of_mem (Enum.complete a)

/-- The value at a member's position is that member. -/
theorem _root_.List.getElem_idxOf_of_mem {β} [DecidableEq β] : ∀ {l : List β} {a : β} (h : a ∈ l),
    l[l.idxOf a]'(List.idxOf_lt_length_of_mem h) = a
  | x :: xs, a, h => by
    by_cases hx : x = a
    · subst hx; simp [List.idxOf_cons_self]
    · have hm : a ∈ xs := by simp at h; rcases h with h | h; exact absurd h.symm hx; exact h
      have hb : (x == a) = false := by simpa using hx
      have e : (x :: xs).idxOf a = xs.idxOf a + 1 := by simp [List.idxOf_cons, hb]
      simp only [e, List.getElem_cons_succ]
      exact List.getElem_idxOf_of_mem hm

/-- Positions name values: two values at one position are one value. -/
theorem index_injective {a b : α} (h : index a = index b) : a = b := by
  rw [← List.getElem_idxOf_of_mem (Enum.complete (α := α) a),
      ← List.getElem_idxOf_of_mem (Enum.complete (α := α) b)]
  simp only [index] at h
  simp [h]

/-- A property of every registered value is a property of every value. -/
theorem forall_iff (p : α → Prop) : (∀ a, p a) ↔ (∀ a ∈ (Enum.all : List α), p a) :=
  ⟨fun h a _ => h a, fun h a => h a (Enum.complete a)⟩

end Enum

end AlgorithmLib
