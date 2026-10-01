module
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
  # The indexing category

  A *shape* is a list of extents and a *point* is a list of coordinates, one
  per axis.  A `Reindex` is an affine map between points — a matrix `lin` and
  an offset `off`, acting as `x ↦ lin · x + off`.  These are the morphisms of
  the category the broadcasting algebra indexes over: composing two of them is
  again affine, which is what lets a chain of axis manipulations collapse into
  one address expression rather than a chain of passes.

  Everything here is `Nat` arithmetic on `List`s, proven by structural
  induction.  Lengths are deliberately *not* carried in the types: `vadd` keeps
  the longer argument and `dotN` stops at the shorter one, so every operation
  is total and the algebraic laws hold with no side conditions.  The category
  laws are stated on the *action* — `apply_comp` and friends — because that is
  the semantics the lowering rests on; the underlying matrices are data whose
  shape is checked by `WF` where a caller needs it.

  `InBounds` is the separate predicate the row-major addressing needs, and it
  is demanded only where a coordinate must really lie inside its extent.
-/

namespace AlgorithmLib.ML.Broadcast

/-- A shape: one extent per axis.  The paper's indexing object. -/
abbrev Ix : Type := List Nat

/-- The number of elements a shape holds. -/
def Ix.size : Ix → Nat
  | []      => 1
  | d :: ss => d * Ix.size ss

/-- Inner product, stopping at the shorter list. -/
def dotN : List Nat → List Nat → Nat
  | x :: xs, y :: ys => x * y + dotN xs ys
  | _, _             => 0

/-- Pointwise sum, keeping the longer list.  Total, and associative. -/
def vadd : List Nat → List Nat → List Nat
  | [],      ys      => ys
  | xs,      []      => xs
  | x :: xs, y :: ys => (x + y) :: vadd xs ys

/-- Scale every entry. -/
def smul (a : Nat) (v : List Nat) : List Nat := v.map (a * ·)

/-- The linear combination `Σ cᵢ • rowᵢ` — one row of a matrix product. -/
def lcomb : List Nat → List (List Nat) → List Nat
  | c :: cs, r :: rs => vadd (smul c r) (lcomb cs rs)
  | _, _             => []

/-- Matrix times vector: one `dotN` per row. -/
def matVec (m : List (List Nat)) (x : List Nat) : List Nat :=
  m.map (fun r => dotN r x)

/-- Matrix product, row by row. -/
def matMul (a b : List (List Nat)) : List (List Nat) :=
  a.map (fun r => lcomb r b)

/-- The `n × n` identity. -/
def idMat : Nat → List (List Nat)
  | 0     => []
  | n + 1 => (1 :: List.replicate n 0) :: (idMat n).map (0 :: ·)

-- ---------------------------------------------------------------------------
-- Arithmetic laws
-- ---------------------------------------------------------------------------

theorem dotN_nil_right (xs : List Nat) : dotN xs [] = 0 := by
  cases xs <;> rfl

theorem dotN_vadd_left (a b x : List Nat) :
    dotN (vadd a b) x = dotN a x + dotN b x := by
  induction a generalizing b x with
  | nil => simp [vadd, dotN]
  | cons a as ih =>
      cases b with
      | nil => simp [vadd, dotN]
      | cons b bs =>
          cases x with
          | nil => simp [vadd, dotN]
          | cons x xs =>
              simp only [vadd, dotN, ih bs xs, Nat.add_mul]
              omega

theorem dotN_vadd_right (r a b : List Nat) :
    dotN r (vadd a b) = dotN r a + dotN r b := by
  induction r generalizing a b with
  | nil => simp [dotN]
  | cons c cs ih =>
      cases a with
      | nil => simp [vadd, dotN]
      | cons a as =>
          cases b with
          | nil => simp [vadd, dotN]
          | cons b bs =>
              simp only [vadd, dotN, ih as bs, Nat.mul_add]
              omega

theorem dotN_smul (c : Nat) (r x : List Nat) :
    dotN (smul c r) x = c * dotN r x := by
  induction r generalizing x with
  | nil => simp [smul, dotN]
  | cons a as ih =>
      cases x with
      | nil => simp [smul, dotN, dotN_nil_right]
      | cons b bs =>
          have h : dotN (smul c as) bs = c * dotN as bs := ih bs
          simp only [smul, List.map_cons, dotN] at h ⊢
          rw [h, Nat.mul_add, Nat.mul_assoc]

theorem vadd_assoc (a b c : List Nat) : vadd (vadd a b) c = vadd a (vadd b c) := by
  induction a generalizing b c with
  | nil => cases b <;> cases c <;> rfl
  | cons a as ih =>
      cases b with
      | nil => cases c <;> rfl
      | cons b bs =>
          cases c with
          | nil => rfl
          | cons c cs => simp only [vadd, ih bs cs, Nat.add_assoc]

theorem vadd_replicate_zero (x : List Nat) :
    vadd x (List.replicate x.length 0) = x := by
  induction x with
  | nil => rfl
  | cons a as ih => simp only [List.length_cons, List.replicate, vadd, ih, Nat.add_zero]

/-- The defining property of a linear combination. -/
theorem dotN_lcomb (c : List Nat) (b : List (List Nat)) (x : List Nat) :
    dotN (lcomb c b) x = dotN c (matVec b x) := by
  induction c generalizing b with
  | nil => simp [lcomb, dotN]
  | cons c cs ih =>
      cases b with
      | nil => simp [lcomb, dotN, matVec, dotN_nil_right]
      | cons r rs =>
          simp only [lcomb, matVec, List.map_cons, dotN, dotN_vadd_left,
            dotN_smul, ih rs]

/-- Matrix multiplication computes composition of the maps. -/
theorem matVec_matMul (a b : List (List Nat)) (x : List Nat) :
    matVec (matMul a b) x = matVec a (matVec b x) := by
  simp only [matMul, matVec, List.map_map]
  exact List.map_congr_left (fun r _ => dotN_lcomb r b x)

/-- A matrix acts linearly on sums. -/
theorem matVec_vadd (m : List (List Nat)) (a b : List Nat) :
    matVec m (vadd a b) = vadd (matVec m a) (matVec m b) := by
  induction m with
  | nil => rfl
  | cons r rs ih =>
      show dotN r (vadd a b) :: matVec rs (vadd a b)
          = (dotN r a + dotN r b) :: vadd (matVec rs a) (matVec rs b)
      rw [dotN_vadd_right, ih]

theorem dotN_replicate_zero (n : Nat) (x : List Nat) :
    dotN (List.replicate n 0) x = 0 := by
  induction n generalizing x with
  | zero => rfl
  | succ n ih =>
      cases x with
      | nil => rfl
      | cons a as => simp [List.replicate, dotN, ih as]

/-- The identity matrix acts as the identity on points of matching length. -/
theorem matVec_idMat (x : List Nat) : matVec (idMat x.length) x = x := by
  induction x with
  | nil => rfl
  | cons a as ih =>
      have h1 : dotN (1 :: List.replicate as.length 0) (a :: as) = a := by
        simp [dotN, dotN_replicate_zero]
      have h2 : ((idMat as.length).map (0 :: ·)).map (fun r => dotN r (a :: as))
          = (idMat as.length).map (fun r => dotN r as) := by
        simp [List.map_map, dotN]
      simp only [List.length_cons, idMat, matVec, List.map_cons, h1, h2]
      exact congrArg (a :: ·) ih

-- ---------------------------------------------------------------------------
-- Reindexings
-- ---------------------------------------------------------------------------

/-- **An affine map between points**, the paper's reindexing morphism `η : P → Q`.

    `lin` has one row per axis of `Q`, each row as wide as `P`; `off` is one
    entry per axis of `Q`.  The shapes are phantom parameters — they say what
    the map is *for*, while `WF` is what checks the data against them. -/
structure Reindex (P Q : Ix) where
  lin : List (List Nat)
  off : List Nat
  deriving Repr, DecidableEq

namespace Reindex

variable {P Q R S : Ix}

/-- `x ↦ lin · x + off`. -/
def apply (η : Reindex P Q) (x : List Nat) : List Nat :=
  vadd (matVec η.lin x) η.off

/-- The rows and the offset match the shapes the morphism claims. -/
def WF (η : Reindex P Q) : Prop :=
  η.lin.length = Q.length ∧ η.off.length = Q.length ∧
    ∀ r ∈ η.lin, r.length = P.length

def id (P : Ix) : Reindex P P :=
  { lin := idMat P.length, off := List.replicate P.length 0 }

/-- Composition in diagrammatic order: `comp η ρ` is `η` first, then `ρ`. -/
def comp (η : Reindex P Q) (ρ : Reindex Q R) : Reindex P R :=
  { lin := matMul ρ.lin η.lin
    off := vadd (matVec ρ.lin η.off) ρ.off }

/-- **Composition computes composition** — the reason a chain of axis
    manipulations collapses to one affine address expression. -/
theorem apply_comp (η : Reindex P Q) (ρ : Reindex Q R) (x : List Nat) :
    (η.comp ρ).apply x = ρ.apply (η.apply x) := by
  simp only [apply, comp, matVec_matMul, matVec_vadd, vadd_assoc]

/-- Composition is associative on the action. -/
theorem apply_comp_assoc (η : Reindex P Q) (ρ : Reindex Q R) (θ : Reindex R S)
    (x : List Nat) :
    ((η.comp ρ).comp θ).apply x = (η.comp (ρ.comp θ)).apply x := by
  rw [apply_comp, apply_comp, apply_comp, apply_comp]

/-- The identity acts as the identity, on points of the right rank. -/
theorem apply_id {x : List Nat} (hx : x.length = P.length) :
    (Reindex.id P).apply x = x := by
  simp only [apply, Reindex.id, ← hx, matVec_idMat, vadd_replicate_zero]

theorem id_comp (η : Reindex P Q) {x : List Nat} (hx : x.length = P.length) :
    ((Reindex.id P).comp η).apply x = η.apply x := by
  rw [apply_comp, apply_id hx]

theorem comp_id (η : Reindex P Q) {x : List Nat}
    (hx : (η.apply x).length = Q.length) :
    (η.comp (Reindex.id Q)).apply x = η.apply x := by
  rw [apply_comp, apply_id hx]

end Reindex

-- ---------------------------------------------------------------------------
-- Row-major addressing
-- ---------------------------------------------------------------------------

/-- Every coordinate lies inside its extent, and the ranks agree. -/
def InBounds : Ix → List Nat → Prop
  | [],      []      => True
  | d :: ss, i :: is => i < d ∧ InBounds ss is
  | _, _             => False

/-- Row-major address of a point. -/
def flatten : Ix → List Nat → Nat
  | _ :: ss, i :: is => i * Ix.size ss + flatten ss is
  | _, _             => 0

/-- The point a row-major address names. -/
def unflatten : Ix → Nat → List Nat
  | [],      _ => []
  | d :: ss, a => (a / Ix.size ss) % d :: unflatten ss (a % Ix.size ss)

theorem flatten_lt {s : Ix} {p : List Nat} (h : InBounds s p) :
    flatten s p < Ix.size s := by
  induction s generalizing p with
  | nil =>
      cases p with
      | nil => exact Nat.zero_lt_one
      | cons _ _ => exact absurd h (by simp [InBounds])
  | cons d ss ih =>
      cases p with
      | nil => exact absurd h (by simp [InBounds])
      | cons i is =>
          have hi : i < d := h.1
          have hf : flatten ss is < Ix.size ss := ih h.2
          have hstep : i * Ix.size ss + flatten ss is < (i + 1) * Ix.size ss := by
            rw [Nat.succ_mul]; omega
          exact Nat.lt_of_lt_of_le hstep (Nat.mul_le_mul hi (Nat.le_refl _))

/-- Addressing round-trips on points that are in bounds. -/
theorem unflatten_flatten {s : Ix} {p : List Nat} (h : InBounds s p) :
    unflatten s (flatten s p) = p := by
  induction s generalizing p with
  | nil =>
      cases p with
      | nil => rfl
      | cons _ _ => exact absurd h (by simp [InBounds])
  | cons d ss ih =>
      cases p with
      | nil => exact absurd h (by simp [InBounds])
      | cons i is =>
          have hi : i < d := h.1
          have hf : flatten ss is < Ix.size ss := flatten_lt h.2
          have hpos : 0 < Ix.size ss := Nat.lt_of_le_of_lt (Nat.zero_le _) hf
          have hdiv : (i * Ix.size ss + flatten ss is) / Ix.size ss = i := by
            rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hf,
              Nat.zero_add]
          have hmod : (i * Ix.size ss + flatten ss is) % Ix.size ss
              = flatten ss is := by
            rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hf]
          simp only [flatten, unflatten, hdiv, hmod, Nat.mod_eq_of_lt hi, ih h.2]

end AlgorithmLib.ML.Broadcast
