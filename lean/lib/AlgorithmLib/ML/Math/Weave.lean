import AlgorithmLib.ML.Math.Reindex

/-!
  # The broadcasting algebra

  An architecture is a composition of operations that each act on *some* axes
  of an array and are tiled across the rest.  This module is that structure as
  algebra: a *weave* tags which axes an operation consumes and which it is
  tiled over, `lift` tiles an operation across a block of them, `pull` reads an
  array through a reindexing, and `par` places two operations side by side.

  Arrays are flat address-indexed functions, row-major, exactly as the kernel
  denotations in `TenDenote` read memory — so an equation proven here is an
  equation about the addresses a kernel actually touches, not about an
  idealised tensor that would have to be related to one later.

  The four laws at the bottom are the algebra's content:

  * `lift_comp`      — tiling is functorial: tile-then-tile = tile the composite
  * `pull_comp`      — reindexing is functorial, contravariantly
  * `slide`          — a reindexing of the tiling axes commutes with the tiled
                       operation
  * `interchange`    — sequential and parallel composition commute

  Each carries its side condition explicitly.  `ReadsBelow` is the honest one:
  an operation on a block of `n` addresses must not read past it, or tiling it
  is not well defined.  Nothing here is stochastic — every operation is a
  function — which is exactly why `slide` holds; the corresponding law for a
  sampling operation would need a naturality hypothesis this instantiation
  does not have to make.
-/

namespace AlgorithmLib.ML.Broadcast

/-- A flat array: a value at every address.  Polymorphic in the element, so
    the laws are about the addressing and not about arithmetic. -/
abbrev Arr (α : Type) : Type := Nat → α

variable {α β γ : Type}

/-- **An operation confined to a block.**  `f`'s result depends only on the
    first `n` addresses of its input.  Tiling an operation that reads past its
    block would read its neighbour's data, so this is the precondition every
    law about `lift` carries. -/
def ReadsBelow (n : Nat) (f : Arr α → Arr β) : Prop :=
  ∀ m m' : Arr α, (∀ a, a < n → m a = m' a) → f m = f m'

/-- **Tiling**, the paper's batch lifting `[f;P]`.

    `f` maps a block of `dA` addresses to a block of `dB`.  The tiled operation
    applies it independently to each consecutive block: the output address
    `addr` sits in block `addr / dB` at offset `addr % dB`, and reads the input
    block starting at `(addr / dB) * dA`. -/
def lift (dA dB : Nat) (f : Arr α → Arr β) : Arr α → Arr β :=
  fun m addr => f (fun a => m ((addr / dB) * dA + a)) (addr % dB)

/-- **Reading an array through a reindexing** — the paper's `[X;η]`.

    An array over `Q` becomes an array over `P` by sending each point of `P`
    through `η` and reading there.  Contravariant, which is why the
    composition law below reverses. -/
def pull (P Q : Ix) (η : Reindex P Q) (m : Arr α) : Arr α :=
  fun addr => m (flatten Q (η.apply (unflatten P addr)))

/-- Reindexing of the *tiling* axes only: block `p` of the result is block
    `r p` of the argument.  This is the form a broadcast takes once the axes
    it tiles over have been fixed. -/
def pullFront (d : Nat) (r : Nat → Nat) (m : Arr α) : Arr α :=
  fun addr => m (r (addr / d) * d + addr % d)

/-- **Two operations side by side**, the monoidal product.  Inputs are
    concatenated at `dA`, outputs at `dB`. -/
def par (dA dB : Nat) (f g : Arr α → Arr β) : Arr α → Arr β :=
  fun m addr =>
    if addr < dB then f m addr else g (fun a => m (dA + a)) (addr - dB)

/-- Sequential composition, in diagrammatic order. -/
def seq (f : Arr α → Arr β) (g : Arr β → Arr γ) : Arr α → Arr γ :=
  fun m => g (f m)

-- ---------------------------------------------------------------------------
-- The laws
-- ---------------------------------------------------------------------------

/-- Tiling the identity is the identity. -/
theorem lift_id (d : Nat) (m : Arr α) : lift d d (fun mm => mm) m = m := by
  funext addr
  show m ((addr / d) * d + addr % d) = m addr
  rw [Nat.mul_comm, Nat.div_add_mod]

/-- **Tiling is functorial**: `[g;P] ∘ [f;P] = [g∘f;P]`.

    The hypothesis is what makes tiling well defined — `g` must not read past
    the block it was tiled over. -/
theorem lift_comp {dA dB dC : Nat} (f : Arr α → Arr β) (g : Arr β → Arr γ)
    (hg : ReadsBelow dB g) (hB : 0 < dB) (m : Arr α) :
    lift dB dC g (lift dA dB f m) = lift dA dC (seq f g) m := by
  funext addr
  have hblk : g (fun a => lift dA dB f m ((addr / dC) * dB + a))
      = g (f (fun b => m ((addr / dC) * dA + b))) := by
    refine hg _ _ (fun a ha => ?_)
    have hd : ((addr / dC) * dB + a) / dB = addr / dC := by
      rw [Nat.mul_comm, Nat.mul_add_div hB, Nat.div_eq_of_lt ha, Nat.add_zero]
    have hm : ((addr / dC) * dB + a) % dB = a := by
      rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt ha]
    show f (fun b => m ((((addr / dC) * dB + a) / dB) * dA + b))
        ((((addr / dC) * dB + a)) % dB) = _
    rw [hd, hm]
  exact congrFun hblk (addr % dC)

/-- **Reindexing is functorial, contravariantly**: `[X; ρ∘η] = [X;η] ∘ [X;ρ]`.

    Stated at a single address, with the condition that the address really is
    reindexed into `Q` — off that condition the round trip through `Q`'s
    addressing is not an identity and the law genuinely fails. -/
theorem pull_comp {P Q R : Ix} (η : Reindex P Q) (ρ : Reindex Q R)
    (m : Arr α) (addr : Nat) (h : InBounds Q (η.apply (unflatten P addr))) :
    pull P R (η.comp ρ) m addr = pull P Q η (pull Q R ρ m) addr := by
  show m (flatten R ((η.comp ρ).apply (unflatten P addr)))
      = m (flatten R (ρ.apply (unflatten Q (flatten Q (η.apply (unflatten P addr))))))
  rw [unflatten_flatten h, Reindex.apply_comp]

/-- **Sliding**: a reindexing of the tiling axes commutes with the tiled
    operation — the paper's `[f;Q] ∘ [Y;η] = [X;η] ∘ [f;P]`.

    Both sides read block `r p` and apply `f` to it; they differ only in the
    order the two are written down.  For a stochastic `f` this would need
    naturality as an extra assumption; here `f` is a function, so it does
    not. -/
theorem slide {dA dB : Nat} (f : Arr α → Arr β) (r : Nat → Nat)
    (hf : ReadsBelow dA f) (hB : 0 < dB) (m : Arr α) :
    lift dA dB f (pullFront dA r m) = pullFront dB r (lift dA dB f m) := by
  funext addr
  have hq : addr % dB < dB := Nat.mod_lt _ hB
  have hd : (r (addr / dB) * dB + addr % dB) / dB = r (addr / dB) := by
    rw [Nat.mul_comm, Nat.mul_add_div hB, Nat.div_eq_of_lt hq, Nat.add_zero]
  have hm : (r (addr / dB) * dB + addr % dB) % dB = addr % dB := by
    rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hq]
  have hblk : f (fun a => pullFront dA r m ((addr / dB) * dA + a))
      = f (fun a => m (r (addr / dB) * dA + a)) := by
    refine hf _ _ (fun a ha => ?_)
    have hpos : 0 < dA := Nat.lt_of_le_of_lt (Nat.zero_le _) ha
    have hd' : ((addr / dB) * dA + a) / dA = addr / dB := by
      rw [Nat.mul_comm, Nat.mul_add_div hpos, Nat.div_eq_of_lt ha, Nat.add_zero]
    have hm' : ((addr / dB) * dA + a) % dA = a := by
      rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt ha]
    show m (r (((addr / dB) * dA + a) / dA) * dA + ((addr / dB) * dA + a) % dA) = _
    rw [hd', hm']
  show f (fun a => pullFront dA r m ((addr / dB) * dA + a)) (addr % dB)
      = lift dA dB f m (r (addr / dB) * dB + addr % dB)
  rw [hblk]
  show _ = f (fun a => m (((r (addr / dB) * dB + addr % dB) / dB) * dA + a))
      ((r (addr / dB) * dB + addr % dB) % dB)
  rw [hd, hm]

/-- **Interchange**: `(f₂ ⊗ g₂) ∘ (f₁ ⊗ g₁) = (f₂ ∘ f₁) ⊗ (g₂ ∘ g₁)`.

    The second component needs no hypothesis — it reads its own half by
    construction.  The first does, for the same reason `lift_comp` does. -/
theorem interchange {dA dB dC : Nat} (f₁ g₁ : Arr α → Arr β) (f₂ g₂ : Arr β → Arr γ)
    (hf₂ : ReadsBelow dB f₂) (m : Arr α) :
    par dB dC f₂ g₂ (par dA dB f₁ g₁ m) = par dA dC (seq f₁ f₂) (seq g₁ g₂) m := by
  funext addr
  have hL : par dB dC f₂ g₂ (par dA dB f₁ g₁ m) addr
      = if addr < dC then f₂ (par dA dB f₁ g₁ m) addr
        else g₂ (fun a => par dA dB f₁ g₁ m (dB + a)) (addr - dC) := rfl
  have hR : par dA dC (seq f₁ f₂) (seq g₁ g₂) m addr
      = if addr < dC then f₂ (f₁ m) addr
        else g₂ (g₁ (fun a => m (dA + a))) (addr - dC) := rfl
  rw [hL, hR]
  by_cases hc : addr < dC
  · have hhalf : f₂ (par dA dB f₁ g₁ m) = f₂ (f₁ m) :=
      hf₂ _ _ (fun a ha => if_pos ha)
    rw [if_pos hc, if_pos hc, hhalf]
  · have hsh : (fun a => par dA dB f₁ g₁ m (dB + a)) = g₁ (fun a => m (dA + a)) := by
      funext a
      show (if dB + a < dB then f₁ m (dB + a)
            else g₁ (fun b => m (dA + b)) (dB + a - dB)) = _
      rw [if_neg (by omega), Nat.add_sub_cancel_left]
    rw [if_neg hc, if_neg hc, hsh]

/-- Two reindexings of the tiling axes compose into one. -/
theorem pullFront_comp {d : Nat} (r₁ r₂ : Nat → Nat) (hd : 0 < d) (m : Arr α) :
    pullFront d r₂ (pullFront d r₁ m) = pullFront d (fun p => r₁ (r₂ p)) m := by
  funext addr
  have hq : addr % d < d := Nat.mod_lt _ hd
  have hdiv : (r₂ (addr / d) * d + addr % d) / d = r₂ (addr / d) := by
    rw [Nat.mul_comm, Nat.mul_add_div hd, Nat.div_eq_of_lt hq, Nat.add_zero]
  have hmod : (r₂ (addr / d) * d + addr % d) % d = addr % d := by
    rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hq]
  show m (r₁ ((r₂ (addr / d) * d + addr % d) / d) * d
      + (r₂ (addr / d) * d + addr % d) % d) = _
  rw [hdiv, hmod]
  rfl

-- ---------------------------------------------------------------------------
-- Broadcasted operations
-- ---------------------------------------------------------------------------

/-- **A broadcasted operation**, the paper's Definition 11.

    `target` is the operation proper, acting on a block of `dIn` addresses and
    producing `dOut`; `front` says which block of the argument each block of
    the result reads, which is where a broadcast lives once the tiled axes are
    fixed.  The block sizes are type indices, so composing two of them can
    only typecheck when the middle sizes agree. -/
structure BOp (α β : Type) (dIn dOut : Nat) where
  front  : Nat → Nat
  target : Arr α → Arr β

/-- What a broadcasted operation computes: read through the reindexing, then
    apply the target to every block. -/
def BOp.den {dIn dOut : Nat} (B : BOp α β dIn dOut) (m : Arr α) : Arr β :=
  lift dIn dOut B.target (pullFront dIn B.front m)

/-- The composite, built without reference to what it computes. -/
def BOp.comp {d₀ d₁ d₂ : Nat} (B₁ : BOp α β d₀ d₁) (B₂ : BOp β γ d₁ d₂) :
    BOp α γ d₀ d₂ :=
  { front := fun p => B₁.front (B₂.front p)
    target := seq B₁.target B₂.target }

/-- **The algebra is closed under composition**: running one broadcasted
    operation after another is a single broadcasted operation, whose target is
    the composite of the targets and whose reindexing is the composite of the
    reindexings.

    This is what makes a whole architecture one term rather than a list of
    stages that can only be reasoned about pairwise. -/
theorem BOp.comp_den {d₀ d₁ d₂ : Nat} (B₁ : BOp α β d₀ d₁) (B₂ : BOp β γ d₁ d₂)
    (h₁ : ReadsBelow d₀ B₁.target) (h₂ : ReadsBelow d₁ B₂.target)
    (hd₀ : 0 < d₀) (hd₁ : 0 < d₁) (m : Arr α) :
    (B₁.comp B₂).den m = B₂.den (B₁.den m) := by
  show lift d₀ d₂ (seq B₁.target B₂.target)
        (pullFront d₀ (fun p => B₁.front (B₂.front p)) m)
      = lift d₁ d₂ B₂.target
        (pullFront d₁ B₂.front (lift d₀ d₁ B₁.target (pullFront d₀ B₁.front m)))
  rw [← slide B₁.target B₂.front h₁ hd₁, lift_comp B₁.target B₂.target h₂ hd₁,
    pullFront_comp B₁.front B₂.front hd₀]

-- ---------------------------------------------------------------------------
-- Weaves
-- ---------------------------------------------------------------------------

/-- **A weave**: one tag per axis.  `true` sends the axis to the front — the
    axes an operation is tiled *over* — and `false` to the back, the axes it
    consumes.  This is the whole of the paper's front/back remapping; the
    permutation it induces on points is `splitPt`/`mergePt` below. -/
abbrev Weave : Type := List Bool

/-- The extents of the tiled axes. -/
def frontIx : Weave → Ix → Ix
  | [],          _      => []
  | _ :: _,      []     => []
  | true :: w,   d :: s => d :: frontIx w s
  | false :: w,  _ :: s => frontIx w s

/-- The extents of the consumed axes. -/
def backIx : Weave → Ix → Ix
  | [],          _      => []
  | _ :: _,      []     => []
  | true :: w,   _ :: s => backIx w s
  | false :: w,  d :: s => d :: backIx w s

/-- Split a point into its tiled and consumed coordinates. -/
def splitPt : Weave → List Nat → List Nat × List Nat
  | [],          _      => ([], [])
  | _ :: _,      []     => ([], [])
  | true :: w,   i :: p => (i :: (splitPt w p).1, (splitPt w p).2)
  | false :: w,  i :: p => ((splitPt w p).1, i :: (splitPt w p).2)

/-- Rebuild a point from its tiled and consumed coordinates. -/
def mergePt : Weave → List Nat → List Nat → List Nat
  | [],          _,      _      => []
  | true :: w,   i :: f, b      => i :: mergePt w f b
  | false :: w,  f,      i :: b => i :: mergePt w f b
  | _ :: _,      _,      _      => []

/-- **A weave loses nothing**: splitting a point and merging it back is the
    identity.  This is the paper's `σ` being an isomorphism. -/
theorem mergePt_splitPt (w : Weave) (p : List Nat) (h : w.length = p.length) :
    mergePt w (splitPt w p).1 (splitPt w p).2 = p := by
  induction w generalizing p with
  | nil => cases p with
    | nil => rfl
    | cons _ _ => simp at h
  | cons t w ih =>
      cases p with
      | nil => simp at h
      | cons i p =>
          have h' : w.length = p.length := by simpa using h
          cases t with
          | true => show i :: mergePt w (splitPt w p).1 (splitPt w p).2 = _
                    rw [ih p h']
          | false => show i :: mergePt w (splitPt w p).1 (splitPt w p).2 = _
                     rw [ih p h']

/-- Splitting keeps every coordinate inside the extent the weave sent it to. -/
theorem splitPt_inBounds (w : Weave) (s : Ix) (p : List Nat)
    (hw : w.length = s.length) (h : InBounds s p) :
    InBounds (frontIx w s) (splitPt w p).1 ∧ InBounds (backIx w s) (splitPt w p).2 := by
  induction w generalizing s p with
  | nil => cases s with
    | nil => cases p with
      | nil => exact ⟨trivial, trivial⟩
      | cons _ _ => exact absurd h (by simp [InBounds])
    | cons _ _ => simp at hw
  | cons t w ih =>
      cases s with
      | nil => simp at hw
      | cons d s =>
          cases p with
          | nil => exact absurd h (by simp [InBounds])
          | cons i p =>
              have hw' : w.length = s.length := by simpa using hw
              have hrec := ih s p hw' h.2
              cases t with
              | true => exact ⟨⟨h.1, hrec.1⟩, hrec.2⟩
              | false => exact ⟨hrec.1, ⟨h.1, hrec.2⟩⟩

end AlgorithmLib.ML.Broadcast
