module
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# Facts about words and lists every layer uses

Proven once here, below every layer that needs them: a fold over a list is
fixed by what its step does on the list's elements, and masking a word to its
low byte changes nothing its low byte shows.
-/

namespace AlgorithmLib

/-- Two steps that agree on every element of a list fold it alike. -/
theorem foldl_congr_mem {α β : Type} (f g : α → β → α) : ∀ (l : List β) (a : α),
    (∀ x ∈ l, ∀ a, f a x = g a x) → l.foldl f a = l.foldl g a
  | [], _, _ => rfl
  | x :: xs, a, h => by
    simp only [List.foldl_cons]
    rw [h x List.mem_cons_self, foldl_congr_mem f g xs _ (fun y hy => h y (List.mem_cons_of_mem _ hy))]

/-- Masking with `255` is a no-op under an 8-bit truncation. -/
theorem and255_toUInt8 (v : UInt64) : (v &&& 255).toUInt8 = v.toUInt8 := by
  apply UInt8.toNat_inj.mp
  simp only [UInt64.toNat_toUInt8, UInt64.toNat_and, show (255 : UInt64).toNat = 2 ^ 8 - 1 from rfl,
    Nat.and_two_pow_sub_one_eq_mod, Nat.mod_mod]

end AlgorithmLib
