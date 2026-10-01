module
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic

@[expose] public section

/-!
# A compiler is its correctness theorem

Every translation in the tree — the host body to Cranelift blocks, an `Expr`
to warp statements, a kernel statement to structured or flat PTX — comes with a
theorem saying what the output does relative to the input. Each states it in
its own words: which inputs it accepts, and what "does the same" means for that
pair of languages. `Compiler` holds the three together, so a compiler cannot be
named without its theorem and the two cannot drift apart.

* `compile` is the function that ships.
* `Pre` is the door: what an input must satisfy for the theorem to speak.
* `Refines s t` is what the theorem concludes about `t = compile s` — forward
  simulation, a denotation equation, a frame; whatever that pair of languages
  needs.
* `sound` is the theorem.

`comp` chains two compilers and is proven here once; the composite's relation
is the relational composition of the two, which a chain then turns into an
end-to-end statement with one lemma about the middle language.
-/

namespace AlgorithmLib

/-- A translation from `S` to `T` together with the theorem that justifies it. -/
structure Compiler (S T : Type) where
  compile : S → T
  Pre : S → Prop
  Refines : S → T → Prop
  sound : ∀ s, Pre s → Refines s (compile s)

namespace Compiler

variable {A B C S T : Type}

/-- Compile with `c₁`, then with `c₂`. `pre` says `c₁`'s output passes `c₂`'s
    door, which is the one obligation composing leaves. -/
def comp (c₁ : Compiler A B) (c₂ : Compiler B C)
    (pre : ∀ s, c₁.Pre s → c₂.Pre (c₁.compile s)) : Compiler A C where
  compile := c₂.compile ∘ c₁.compile
  Pre := c₁.Pre
  Refines a c := ∃ b, c₁.Refines a b ∧ c₂.Refines b c
  sound s hs := ⟨c₁.compile s, c₁.sound s hs, c₂.sound _ (pre s hs)⟩

@[simp] theorem comp_compile (c₁ : Compiler A B) (c₂ : Compiler B C)
    (pre : ∀ s, c₁.Pre s → c₂.Pre (c₁.compile s)) (s : A) :
    (c₁.comp c₂ pre).compile s = c₂.compile (c₁.compile s) := rfl

/-- The composite's witness is the intermediate program itself. -/
theorem comp_sound_at (c₁ : Compiler A B) (c₂ : Compiler B C)
    (pre : ∀ s, c₁.Pre s → c₂.Pre (c₁.compile s)) (s : A) (hs : c₁.Pre s) :
    c₁.Refines s (c₁.compile s) ∧ c₂.Refines (c₁.compile s) (c₂.compile (c₁.compile s)) :=
  ⟨c₁.sound s hs, c₂.sound _ (pre s hs)⟩

/-- Compile with `c₁`, then with `c₂`, accepting exactly the inputs whose
    intermediate program passes `c₂`'s door. `comp` without its obligation:
    the door is checked where the chain is used rather than proven for every
    input. -/
def seq (c₁ : Compiler A B) (c₂ : Compiler B C) : Compiler A C where
  compile := c₂.compile ∘ c₁.compile
  Pre s := c₁.Pre s ∧ c₂.Pre (c₁.compile s)
  Refines a c := ∃ b, c₁.Refines a b ∧ c₂.Refines b c
  sound s hs := ⟨c₁.compile s, c₁.sound s hs.1, c₂.sound _ hs.2⟩

/-- Act on the first component of a pair and carry the second through: for a
    compiler whose output is a program together with a value. -/
def onFst {X : Type} (c : Compiler B C) : Compiler (B × X) (C × X) where
  compile p := (c.compile p.1, p.2)
  Pre p := c.Pre p.1
  Refines p q := c.Refines p.1 q.1 ∧ p.2 = q.2
  sound p hp := ⟨c.sound p.1 hp, rfl⟩

/-- Keep the function and the door; state less. A compiler whose relation
    implies `R` is a compiler for `R`. -/
def weaken (c : Compiler S T) (R : S → T → Prop)
    (h : ∀ s t, c.Pre s → c.Refines s t → R s t) : Compiler S T where
  compile := c.compile
  Pre := c.Pre
  Refines := R
  sound s hs := h s _ hs (c.sound s hs)

/-- Narrow the door: accept only inputs that also satisfy `P`. -/
def restrict (c : Compiler S T) (P : S → Prop) : Compiler S T where
  compile := c.compile
  Pre s := c.Pre s ∧ P s
  Refines := c.Refines
  sound s hs := c.sound s hs.1

end Compiler

end AlgorithmLib
