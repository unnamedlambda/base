import AlgorithmLib.ML.WarpCompile
import AlgorithmLib.ML.Quant
import AlgorithmLib.ML.Schema
import AlgorithmLib.ML.Butterfly

/-!
  # A rewrite calculus: transformations that carry their own justification

  A user tunes by transforming their model.  The question this file answers is
  *which transformations are equalities, and what happens to the ones that are
  not*.

  Three tiers, and the type system keeps them apart:

  * `Rewrite`  — `denote` preserved **exactly**, in every carrier.  Free to
    apply; correctness of the compiled kernel follows with no new obligation.
  * `FloatRewrite` — preserved exactly at `Float` only (relies on a specific
    reduction order or a `NumLaws` fact `Float` lacks).
  * `Approx` — deliberately changes the value, carrying a *name*.  These are
    the only places a compiled kernel can differ from its spec, and they cannot
    be applied silently.

  A butterfly reduction is **not** a `Rewrite` of `Expr.sum` — measured, the two
  differ (`1.0` vs `0.0` on 32 lanes).  The calculus makes that a type error
  rather than a subtle bug.
-/

namespace AlgorithmLib.ML

/-- A meaning-preserving transformation, in every carrier at once. -/
structure Rewrite (Γ : Nat) where
  run   : Expr Γ → Expr Γ
  sound : ∀ {R : Type} [NumOps R] (env : Fin Γ → R) (e : Expr Γ),
            denote env (run e) = denote env e

namespace Rewrite

variable {Γ : Nat}

def id : Rewrite Γ := ⟨fun e => e, by intro R _ env e; rfl⟩

end Rewrite

-- ---------------------------------------------------------------------------
-- Tier 1: exact rewrites
-- ---------------------------------------------------------------------------


-- ---------------------------------------------------------------------------
-- Tier 3: declared approximations
-- ---------------------------------------------------------------------------

/-- A transformation that deliberately changes the value.  The `name` is what
    appears in `Assumptions.lean`; `why` is the justification a reader needs.
    There is no `sound` field — that is the point. -/
structure Approx (Γ : Nat) where
  run  : Expr Γ → Expr Γ
  name : String
  why  : String

/-- The softmax max-shift: mathematically the identity, numerically the reason
    softmax does not overflow.  It is **not** a `Rewrite`, because in `Float`
    `e^(x-m)/Σe^(x-m) ≠ e^x/Σe^x` bit-for-bit. -/
def maxShift {Γ : Nat} (shift : Expr Γ) : Approx Γ where
  run := fun e => .mul (.exp (.neg shift)) (.mul (.exp shift) e)
  name := "softmax-max-shift"
  why := "identity in ℝ; in Float it trades bit-exactness for overflow safety"

/-- Substituting the hardware's `ex2.approx` for `exp`.

    This is the *spec-level* name for what `AlgorithmLib.ML.expandExp` does
    concretely at the machine-expression level, where the identity it needs is
    the single named proposition `ExpIsEx2` and the only theorem mentioning it
    is `expandExp_approx`.  Measured: max relative error 5.06e-7 on silu,
    3.27e-7 end to end on the three-layer model. -/
def ex2Approx {Γ : Nat} : Approx Γ where
  run := fun e => e
  name := "ex2.approx-for-exp"
  why := "PTX has no exact e^x; e^x = 2^(x·log₂e) via ex2.approx, ~2 ULP"

-- ---------------------------------------------------------------------------
-- The declared-law registry
-- ---------------------------------------------------------------------------

/-!
  ## Tier 2, made concrete

  The three tiers above were designed before there was anything to apply them
  to.  There is now: every reduction in this stack is proven against the
  **order the hardware actually walks** — sequential within a lane, butterfly
  across lanes — which is what makes those theorems exact `Float32` equalities
  rather than bounds.

  A user who writes `Expr.sum` and expects the kernel to compute *that* is
  relying on reassociation, and reassociation is not a `Float32` identity.
  Rather than leave that unstated, it is a **named law** here, alongside the one
  the `exp` lowering already carries.

  This is the gap every framework has and none states: a butterfly reduction and
  a sequential fold are different functions on floats, and which one your model
  was trained with is not recorded anywhere.
-/

/-- The value a vendor GEMM lands. Opaque on purpose: the point of
    `CuBlasIsMatvec` is that nothing in this development knows how it was
    computed, only what it is claimed to equal. -/
opaque cublasSgemvResult (rows cols : Nat) (a : Fin rows → Fin cols → Float32)
    (x : Fin cols → Float32) (i : Fin rows) : Float32

/-- The idealised order: fold the lanes left to right. -/
def laneSum (v : Lane → Float32) : Float32 :=
  (List.finRange W).foldl (fun a l => NumOps.add a (v l)) (NumOps.ofNat 0)

/-- **What relating the butterfly to a sequential sum would cost.**

    Not a theorem — a *proposition*, false for `Float32`, and the exact price of
    writing `Expr.sum` and expecting the shuffle network.

    Measured, on 32 lanes: with lane 0 holding `1e8` and the rest `1.0`, the
    butterfly gives `100000024` and the sequential fold `100000000` — the tree
    accumulates the 31 ones into `31` before adding, where the left fold loses
    each one individually to `1e8`'s 8-ULP spacing.  With lane 0 at `1e6` and
    the rest at `0.03125`, `1000000.9375` against `1000000`.

    Both are *correct* summations; they are not the *same* summation.  Which one
    a model was trained with is not recorded anywhere, in any framework. -/
def SumAssoc : Prop :=
  ∀ v : Lane → Float32,
    bflyFoldOp (fun a b => NumOps.add a b) v ⟨0, by decide⟩ = laneSum v

/-- **What relying on a vendor GEMM costs.**

    `cl_cublas_sgemv` computes `y = A·x`, and the shipped Qwen2 model routes
    every matmul through it — around 99.9% of the model's arithmetic.

    It differs from `SumAssoc` in a way that matters. `SumAssoc` names a
    *specific* identity you can write down: butterfly equals sequential fold.
    cuBLAS's fold order is not merely different, it is **unspecified** — it
    varies with library version, architecture, and shape, and NVIDIA documents
    no guarantee about it. So there is no exact-`Float32` statement to make.

    **What is written below is a `Float32` equation to one specific left-to-right
    fold**, because that is the only thing this development can express: there
    is no `Float32 → ℝ` embedding here and no rounding model, so "computes the
    product up to rounding" is not a statable proposition. The honest reading is
    therefore that this law is *stronger than what NVIDIA guarantees* — if the
    vendor folds in another order, or contracts a multiply-add, the hypothesis
    is false rather than approximately true.

    That is a real cost and it is why the law is named, printed at build time,
    and cross-checked against a proven kernel by a differential test rather than
    trusted. Stating it at the reals instead would need an ℝ semantics for
    `Float32` plus error propagation through every composition theorem in the
    stack, and the resulting bound would be vacuous at depth.

    This is the gap every framework has. PyTorch defaulted `allow_tf32` on for
    matmul across several releases — silently truncating the mantissa to 10
    bits on Ampere — and it was discoverable only from release notes. Here it
    appears in the type of every theorem downstream of a matmul. -/
def CuBlasIsMatvec : Prop :=
  ∀ (rows cols : Nat) (a : Fin rows → Fin cols → Float32) (x : Fin cols → Float32)
    (i : Fin rows),
    cublasSgemvResult rows cols a x i
      = (List.finRange cols).foldl (fun acc j => NumOps.add acc (NumOps.mul (a i j) (x j)))
          (NumOps.ofNat 0)

/-! ### The same call, assumed honestly

    `CuBlasIsMatvec` names one order.  What the vendor actually promises is a
    sum of the right products in *some* association, and that is statable here
    — a summation tree over the index set, with a permutation side condition,
    exactly the shape `LaneRegroup` takes for a schedule this stack owns.

    Both laws are kept because they buy different things.  The weak one is
    true of each call and says nothing about a sequence of them; the strong one
    is what lets an equality chain through twenty-four layers.  Which a theorem
    needs is visible in its type, and `cublasIsMatvec_strengthens` is the proof
    that choosing the strong one is a *refinement* rather than a second,
    unrelated assumption. -/

/-- An association of a sum: `nil` is the accumulator a fold starts from, a leaf
    is one product, a node is one addition.

    `nil` is a constructor rather than an omission because a real
    implementation starts from a zeroed accumulator and a split contraction
    starts from several — so a tree that adds zeros is one this admits. -/
inductive SumTree where
  | nil  : SumTree
  | leaf : Nat → SumTree
  | node : SumTree → SumTree → SumTree
  deriving Repr

/-- Which products the tree sums, in the order it reaches them. -/
def SumTree.indices : SumTree → List Nat
  | .nil      => []
  | .leaf i   => [i]
  | .node a b => a.indices ++ b.indices

def SumTree.eval (f : Nat → Float32) : SumTree → Float32
  | .nil      => NumOps.ofNat 0
  | .leaf i   => f i
  | .node a b => NumOps.add (SumTree.eval f a) (SumTree.eval f b)

/-- The left-leaning association: what a sequential fold from zero performs. -/
def foldTree : Nat → SumTree
  | 0     => .nil
  | n + 1 => .node (foldTree n) (.leaf n)

theorem foldTree_indices : ∀ n, (foldTree n).indices = List.range n := by
  intro n
  induction n with
  | zero => rfl
  | succ k ih => simp [foldTree, SumTree.indices, ih, List.range_succ]

theorem foldTree_eval (f : Nat → Float32) : ∀ n,
    (foldTree n).eval f
      = (List.range n).foldl (fun acc j => NumOps.add acc (f j)) (NumOps.ofNat 0) := by
  intro n
  induction n with
  | zero => rfl
  | succ k ih => simp [foldTree, SumTree.eval, ih, List.range_succ, List.foldl_append]

/-- The product at column `j` of row `i`, as a total function of a `Nat` — so a
    tree's leaves can be plain indices and the permutation condition is what
    keeps them in range. -/
def gemvTerm {rows cols : Nat} (a : Fin rows → Fin cols → Float32)
    (x : Fin cols → Float32) (i : Fin rows) (j : Nat) : Float32 :=
  if h : j < cols then NumOps.mul (a i ⟨j, h⟩) (x ⟨j, h⟩) else NumOps.ofNat 0

/-- **What a vendor GEMV actually promises.**

    Every product once, summed in some association.  This is the leeway the
    vendor takes and it is all of it that is expressible over `Float32`: what
    remains outside is fusing a multiply-add into one rounding and accumulating
    at another width, neither of which is a reassociation.  `VendorKernel.withholds`
    is where those are named. -/
def CuBlasIsSomeReassoc : Prop :=
  ∀ (rows cols : Nat) (a : Fin rows → Fin cols → Float32) (x : Fin cols → Float32)
    (i : Fin rows),
    ∃ t : SumTree, t.indices.Perm (List.range cols) ∧
      cublasSgemvResult rows cols a x i = t.eval (gemvTerm a x i)

/-- **The strict law refines the honest one.**

    Assuming the vendor folds left-to-right is assuming one particular
    association, so anything proven from the weak law is available from the
    strong one — the strong one is a *choice among* the associations the weak
    one permits, not an extra assumption pointing elsewhere. -/
theorem gemvTerm_val {rows cols : Nat} (a : Fin rows → Fin cols → Float32)
    (x : Fin cols → Float32) (i : Fin rows) (j : Fin cols) :
    gemvTerm a x i j.val = NumOps.mul (a i j) (x j) := by
  simp only [gemvTerm, dif_pos j.isLt, Fin.eta]

theorem cublasIsMatvec_strengthens (h : CuBlasIsMatvec) : CuBlasIsSomeReassoc := by
  intro rows cols a x i
  refine ⟨foldTree cols, by rw [foldTree_indices], ?_⟩
  rw [foldTree_eval, h rows cols a x i]
  have hmap : (List.finRange cols).map Fin.val = List.range cols := by
    apply List.ext_getElem
    · simp
    · intro k h1 h2; simp
  rw [← hmap, List.foldl_map]
  have hfun : (fun (acc : Float32) (y : Fin cols) => NumOps.add acc (gemvTerm a x i y.val))
      = (fun (acc : Float32) (j : Fin cols) => NumOps.add acc (NumOps.mul (a i j) (x j))) := by
    funext acc j; rw [gemvTerm_val]
  rw [hfun]

/-- **What it costs to read a lane-partitioned fold as a flat one.**

    `SumAssoc` bridges the *butterfly* to a sequential lane fold.  A kernel does
    something more: each lane also folds its own strided slice sequentially, so
    the whole reduction is a two-level regrouping of the flat sum over all `n`
    elements.  Reading the kernel's answer as `Σᵢ aᵢ·bᵢ` needs both levels.

    This is the *second* half of the bridge between the pipeline theorems
    (`Pipeline.lean`, which end at the committed fold) and the calculus
    (`Backprop.lean`, which ends at `Expr.sum`).  Naming it is what makes the
    composite sentence a theorem with a stated hypothesis, rather than something
    a reader has to assemble from two halves.

    False at `Float32` for the same reason `SumAssoc` is — measured there:
    `.vec4` and `.strided` schedules of one reduction differ by 14% on a
    cancelling input (`Sched.lean`).

    **Stated for an arbitrary per-element summand `f`, not for `aᵢ·bᵢ`.**  The
    regrouping is a fact about the *shape of the fold*, not about what is being
    summed: the same two-level walk appears in RMSNorm at `f i = xᵢ·xᵢ` and in
    softmax's denominator at `f i = exp(zᵢ − max)`.  Specialising the law to a
    dot product would have meant a second, separately-named assumption for
    softmax describing the identical rearrangement, which is worse — one law
    covering both is the smaller surface, not the larger.  `base` is there for
    the same reason: softmax reduces one attention row, which starts at
    `cta·seqLen`, not at zero.  `strided_eq_flatSum` below recovers the
    dot-product form. -/
def StridedRegroup : Prop :=
  ∀ (f : Nat → Float32) (base K : Nat),
    bflyFoldOp (fun a b => NumOps.add a b)
        (fun l => (List.range K).foldl
          (fun acc j => NumOps.add acc (f (base + (j * 32 + l.val)))) NumOps.zero)
        ⟨0, by decide⟩
      = (List.range (K * 32)).foldl
          (fun a i => NumOps.add a (f (base + i))) NumOps.zero

/-- **Regrouping a warp-wide sum.**

    Each lane accumulates `K` values, the butterfly combines the 32 lanes, and
    the claim is that the result is the flat sequential sum over the same
    indices.  `idx j l` is the index lane `l` reads on trip `j`; the
    permutation hypothesis says those indices are exactly `0 … K*32-1`, each
    once — so the law grants *reassociation*, not a free pass on coverage,
    and every schedule still has to show it reads each element.

    This is what current frameworks already assume: `torch.sum` promises a sum
    over `n` elements, not an order, and cuBLAS varies its fold order with
    shape, architecture and version.  Stating it once over an arbitrary `idx`
    means a new schedule needs a permutation proof rather than a new law.

    `StridedRegroup` is the instance at `idx j l = j*32 + l`. -/
def LaneRegroup : Prop :=
  ∀ (f : Nat → Float32) (K : Nat) (idx : Nat → Lane → Nat),
    (((List.range K).flatMap
        (fun j => (List.finRange 32).map (fun l => idx j l))).Perm
      (List.range (K * 32))) →
    bflyFoldOp (fun a b => NumOps.add a b)
        (fun l => (List.range K).foldl
          (fun acc j => NumOps.add acc (f (idx j l))) NumOps.zero)
        ⟨0, by decide⟩
      = (List.range (K * 32)).foldl
          (fun a i => NumOps.add a (f i)) NumOps.zero

/-- The strided walk covers `0 … K*32-1` exactly once, in order.  Equality
    rather than a permutation, since lane `l` on trip `j` reads `j*32 + l`. -/
theorem strided_idx_range : ∀ K : Nat,
    (List.range K).flatMap (fun j => (List.finRange 32).map (fun l => j * 32 + l.val))
      = List.range (K * 32) := by
  intro K
  induction K with
  | zero => rfl
  | succ k ih =>
      rw [List.range_succ, List.flatMap_append, ih]
      simp only [List.range_add, Nat.succ_mul, List.flatMap_cons, List.flatMap_nil,
                 List.append_nil]
      rfl

/-- `A` blocks of `w` consecutive indices are `0 … A*w-1`. -/
theorem range_flat_blocks (w : Nat) : ∀ K : Nat,
    (List.range K).flatMap (fun j => (List.range w).map (fun x => j * w + x))
      = List.range (K * w) := by
  intro K
  induction K with
  | zero => rw [Nat.zero_mul]; rfl
  | succ k ih =>
      rw [List.range_succ, List.flatMap_append, ih]
      simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, Nat.succ_mul]
      rw [List.range_add]

/-- Peeling the last element out of each block of a `flatMap`. -/
theorem flatMap_append_singleton {α β : Type} (g : α → List β) (h : α → β) :
    ∀ L : List α, (L.flatMap (fun b => g b ++ [h b])).Perm (L.flatMap g ++ L.map h) := by
  intro L
  induction L with
  | nil => exact List.Perm.refl _
  | cons a L ih =>
      simp only [List.flatMap_cons, List.map_cons, List.append_assoc]
      refine List.Perm.append_left (g a) ?_
      refine (List.Perm.append_left [h a] ih).trans ?_
      simp only [List.singleton_append]
      exact (List.perm_middle).symm

/-- **Swapping two nested loops is a permutation of the indices they visit.**

    The reusable half of every "walk the same data in a different order"
    schedule: whichever loop a schedule puts outside, it reads the same
    multiset, so `LaneRegroup`'s coverage obligation reduces to this. -/
theorem flatMap_map_comm {β : Type} (f : Nat → Nat → β) (B : Nat) : ∀ A : Nat,
    (((List.range A).flatMap (fun a => (List.range B).map (f a))).Perm
      ((List.range B).flatMap (fun b => (List.range A).map (fun a => f a b)))) := by
  intro A
  induction A with
  | zero =>
      simp only [List.range_zero, List.flatMap_nil, List.map_nil]
      have hn : ∀ L : List Nat, L.flatMap (fun _ : Nat => ([] : List β)) = [] := by
        intro L; induction L with
        | nil => rfl
        | cons _ _ ih => simpa using ih
      rw [hn]
  | succ a ih =>
      have hr : ∀ b, (List.range (a+1)).map (fun x => f x b)
          = (List.range a).map (fun x => f x b) ++ [f a b] := by
        intro b; rw [List.range_succ, List.map_append]; rfl
      have hL : (List.range (a+1)).flatMap (fun x => (List.range B).map (f x))
          = (List.range a).flatMap (fun x => (List.range B).map (f x))
            ++ (List.range B).map (f a) := by
        rw [List.range_succ, List.flatMap_append]
        simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil]
      rw [hL]
      simp only [hr]
      exact (ih.append (List.Perm.refl _)).trans
        (flatMap_append_singleton (fun b => (List.range a).map (fun x => f x b))
          (fun b => f a b) (List.range B)).symm

/-- **`StridedRegroup` is the strided instance of `LaneRegroup`**, so the
    registry carries one reassociation law rather than one per schedule. -/
theorem stridedRegroup_of_laneRegroup (h : LaneRegroup) : StridedRegroup := by
  intro f base K
  exact h (fun i => f (base + i)) K (fun j l => j * 32 + l.val)
    (by rw [strided_idx_range K])

/-- One quad step, with the four products spelled out.  `dotLane` adds them in
    load order, left-associated, which is what the four `ld.global.v4.f32`
    lanes of a step commit to. -/
theorem dotLane_succ (memA memB : Nat → Float32) (fA fB : Nat → Lane → Nat)
    (l : Lane) (k : Nat) :
    dotLane memA memB fA fB (k+1) l
      = NumOps.add (NumOps.add (NumOps.add (NumOps.add (dotLane memA memB fA fB k l)
          (NumOps.mul (memA (fA k l)) (memB (fB k l))))
          (NumOps.mul (memA (fA k l + 1)) (memB (fB k l + 1))))
          (NumOps.mul (memA (fA k l + 2)) (memB (fB k l + 2))))
          (NumOps.mul (memA (fA k l + 3)) (memB (fB k l + 3))) := by
  simp only [dotLane, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]

/-- **A quad fold is a scalar fold at four times the length.**

    `LaneRegroup` is stated for one addition per step, and `dotLane` performs
    four.  Because those four are left-associated, `K` quad steps are `4K`
    scalar steps over the reindexed walk `m ↦ fA (m/4) l + m%4`, so the law
    applies to the quad schedule without a second law describing it. -/
theorem dotLane_flat (memA memB : Nat → Float32) (fA fB : Nat → Lane → Nat)
    (l : Lane) : ∀ K : Nat,
    dotLane memA memB fA fB K l
      = (List.range (K * 4)).foldl
          (fun acc m => NumOps.add acc
            (NumOps.mul (memA (fA (m/4) l + m%4)) (memB (fB (m/4) l + m%4))))
          (NumOps.ofNat 0) := by
  intro K
  induction K with
  | zero => rfl
  | succ k ih =>
      rw [dotLane_succ, ih]
      have e : (k+1) * 4 = k*4 + 1 + 1 + 1 + 1 := by omega
      rw [e, List.range_succ, List.range_succ, List.range_succ, List.range_succ,
          List.foldl_append, List.foldl_append, List.foldl_append, List.foldl_append]
      have d0 : (k*4)/4 = k := by omega
      have m0 : (k*4)%4 = 0 := by omega
      have d1 : (k*4+1)/4 = k := by omega
      have m1 : (k*4+1)%4 = 1 := by omega
      have d2 : (k*4+1+1)/4 = k := by omega
      have m2 : (k*4+1+1)%4 = 2 := by omega
      have d3 : (k*4+1+1+1)/4 = k := by omega
      have m3 : (k*4+1+1+1)%4 = 3 := by omega
      simp only [List.foldl_cons, List.foldl_nil, d0, m0, d1, m1, d2, m2, d3, m3,
                 Nat.add_zero]

/-- One quad step across the warp covers 128 consecutive elements: the 32 lanes
    of offset `r ∈ 0…3` interleave four-apart.  A permutation, not an equality
    — which is the difference between this schedule and the strided one. -/
theorem quad_block (c : Nat) :
    ((List.finRange 32).map (fun l => c + l.val*4) ++
     ((List.finRange 32).map (fun l => c + l.val*4 + 1) ++
      ((List.finRange 32).map (fun l => c + l.val*4 + 2) ++
       (List.finRange 32).map (fun l => c + l.val*4 + 3)))).Perm
    ((List.range 128).map (fun x => c + x)) := by
  have hb : ((List.finRange 32).map (fun (l : Fin 32) => l.val*4) ++
     ((List.finRange 32).map (fun (l : Fin 32) => l.val*4 + 1) ++
      ((List.finRange 32).map (fun (l : Fin 32) => l.val*4 + 2) ++
       (List.finRange 32).map (fun (l : Fin 32) => l.val*4 + 3)))).Perm (List.range 128) := by
    rw [← List.isPerm_iff]; decide
  have h2 := hb.map (fun x => c + x)
  simpa [List.map_append, List.map_map, Function.comp, Nat.add_assoc] using h2

/-- The quad walk reads `0 … K*128-1` exactly once — the coverage obligation
    `LaneRegroup` demands of any schedule that wants the reassociation. -/
theorem quad_idx_perm : ∀ K : Nat,
    ((List.range (K*4)).flatMap
        (fun m => (List.finRange 32).map (fun l : Lane => (m/4)*128 + l.val*4 + m%4))).Perm
      (List.range (K*128)) := by
  intro K
  induction K with
  | zero => exact List.Perm.refl _
  | succ k ih =>
      have e4 : (k+1)*4 = k*4+4 := by omega
      have e128 : (k+1)*128 = k*128 + 128 := by omega
      rw [e4, e128, List.range_add, List.range_add]
      have hr4 : List.range 4 = [0,1,2,3] := rfl
      have d0 : (k*4)/4 = k := by omega
      have m0 : (k*4)%4 = 0 := by omega
      have d1 : (k*4+1)/4 = k := by omega
      have m1 : (k*4+1)%4 = 1 := by omega
      have d2 : (k*4+2)/4 = k := by omega
      have m2 : (k*4+2)%4 = 2 := by omega
      have d3 : (k*4+3)/4 = k := by omega
      have m3 : (k*4+3)%4 = 3 := by omega
      simp only [hr4, List.map_cons, List.map_nil, List.flatMap_append,
                 List.flatMap_cons, List.flatMap_nil, List.append_nil,
                 Nat.add_zero, d0, m0, d1, m1, d2, m2, d3, m3]
      exact ih.append (quad_block (k*128))

/-- **The two warp combiners commute.**

    Needed for exactly one thing: a butterfly reduction leaves the *same* value
    in every lane.  Softmax's remainder pass has all 32 lanes write one address,
    so the kernel is correct only if they agree on the row maximum and the
    reciprocal sum; `bflyFoldOp_const` derives that from this and nothing else.

    Deliberately **not** `sumAssoc`.  Every lane of a butterfly walks a tree of
    the *same shape* over the *same* leaves — only the argument order within
    each node differs — so commutativity suffices, and unlike associativity it
    is *true* at IEEE-754 for all non-NaN inputs.  `Float32.add` is opaque in
    Lean, which is the only reason this is a law rather than a theorem.

    The NaN caveat is real and small: IEEE-754 leaves the payload of a NaN
    result from `a + b` implementation-defined, so bit-exact commutativity can
    fail when an operand is NaN.  A softmax row that is entirely NaN is already
    outside what any of these theorems say something useful about. -/
def CombinerComm : Prop :=
  (∀ a b : Float32, NumOps.add a b = NumOps.add b a)
    ∧ (∀ a b : Float32, NumOps.max a b = NumOps.max b a)

/-- A named numerical law.  Transformations and specs that need one say so in
    their type; `Assumptions.lean` names it; nothing applies one silently. -/
inductive Law where
  /-- `e^x = 2^(x·log₂e)` — PTX has no exact `exp`. -/
  | expIsEx2
  /-- A butterfly reduction equals a sequential fold. -/
  | sumAssoc
  /-- A vendor GEMV equals a left-to-right fold of the matrix-vector product. -/
  | cublasIsMatvec
  /-- A vendor GEMV sums the right products in *some* association. -/
  | cublasIsSomeReassoc
  /-- The warp combiners commute, so a butterfly is lane-uniform. -/
  | combinerComm
  /-- Any lane-partitioned fold equals the flat fold, given coverage. -/
  | laneRegroup
  deriving DecidableEq, Repr

/-- **The registry, as a list.**

    `Law.all_covers` is what makes it the registry rather than a list beside
    one: a constructor missing from it fails to build.  A report that
    enumerates this cannot describe a stack as resting on fewer assumptions
    than it has. -/
def Law.all : List Law :=
  [.expIsEx2, .sumAssoc, .cublasIsMatvec, .cublasIsSomeReassoc,
   .combinerComm, .laneRegroup]

theorem Law.all_covers : ∀ l : Law, l ∈ Law.all := by
  intro l; cases l <;> decide

def Law.title : Law → String
  | .expIsEx2 => "ex2.approx-for-exp"
  | .sumAssoc => "sum-reassociation"
  | .laneRegroup => "lane-fold-regrouping"
  | .cublasIsMatvec => "cublas-is-matvec"
  | .cublasIsSomeReassoc => "cublas-sums-in-some-order"
  | .combinerComm => "warp-combiner-commutes"

def Law.why : Law → String
  | .laneRegroup =>
      "A lane-partitioned fold and the flat fold are different functions on " ++
      "Float32. This is the reassociation every framework already assumes: " ++
      "torch.sum promises a sum over n elements, not an order, and cuBLAS " ++
      "varies its fold order with shape, architecture and version. Coverage " ++
      "is not granted — each schedule proves its indices are a permutation."
  | .expIsEx2 =>
      "PTX has no exact e^x; the emitter uses 2^(x·log₂e) via ex2.approx. " ++
      "Measured 5.06e-7 on silu, 3.27e-7 end to end on a three-layer model."
  | .sumAssoc =>
      "A warp butterfly and a left fold are different functions on Float32. " ++
      "Every kernel here is proven against the butterfly, which is what the " ++
      "hardware does; this law is what it would cost to claim Expr.sum instead."
  | .cublasIsSomeReassoc =>
      "cl_cublas_sgemv sums the right products, each once, in some association. " ++
      "This is what the vendor actually promises and it is all of the leeway " ++
      "that is expressible over Float32; it is true of one call and says " ++
      "nothing about a sequence of them, which is why a chained theorem needs " ++
      "the stronger cublas-is-matvec instead. Still outside it: fusing a " ++
      "multiply-add into one rounding, and accumulating at another width."
  | .cublasIsMatvec =>
      "cl_cublas_sgemv computes y = A·x, but NVIDIA specifies no fold order, " ++
      "so this states the one order that is expressible here: a left-to-right " ++
      "Float32 fold. That is stronger than the vendor guarantees, which is the " ++
      "cost of having no R semantics for Float32 to state it at instead. " ++
      "~99.9% of the shipped Qwen2 model's arithmetic depends on it. Compare " ++
      "TF32, which other frameworks enable by default without stating it at all."
  | .combinerComm =>
      "Float32 add and max commute. Strictly weaker than sumAssoc and, unlike " ++
      "it, true at IEEE-754 for non-NaN inputs: a butterfly's lanes walk the " ++
      "same tree shape and differ only in argument order within each node. " ++
      "Needed only so softmax's remainder pass, where all 32 lanes write one " ++
      "address, has all 32 lanes agreeing on what to write."

/-- **The weaker law a law refines**, where one exists.

    A strict assumption is a choice *among* the behaviours a weak one permits,
    so this records which laws are strictifications rather than separate
    commitments.  `Law.weakens_holds` is the proof that the relation means what
    it says, so the registry cannot claim a refinement that is not one. -/
def Law.weakens : Law → Option Law
  | .cublasIsMatvec => some .cublasIsSomeReassoc
  | _               => none

def Law.holds : Law → Prop
  | .expIsEx2 => ExpIsEx2
  | .sumAssoc => SumAssoc
  | .cublasIsMatvec => CuBlasIsMatvec
  | .cublasIsSomeReassoc => CuBlasIsSomeReassoc
  | .combinerComm => CombinerComm
  | .laneRegroup => LaneRegroup

/-- **Assuming the strict law gives you the weak one.**  General over the
    registry, so a new refinement entry without a proof behind it fails here. -/
theorem Law.weakens_holds : ∀ (l w : Law), l.weakens = some w → l.holds → w.holds := by
  intro l w h hl
  cases l with
  | cublasIsMatvec =>
      injection h with h'; subst h'; exact cublasIsMatvec_strengthens hl
  | _ => exact absurd h (by simp [Law.weakens])

/-- Laws a claim depends on.  `[]` means exact — which is what every shipped
    lowering theorem in this stack actually is. -/
def AllHold (ls : List Law) : Prop := ∀ l ∈ ls, l.holds

/-- **The bridge, under its law.**  A user's `Expr.sum`-shaped spec meets the
    kernel's committed butterfly order here, and nowhere else. -/
theorem bfly_eq_laneSum (h : AllHold [Law.sumAssoc]) (v : Lane → Float32) :
    bflyFoldOp (fun a b => NumOps.add a b) v ⟨0, by decide⟩ = laneSum v :=
  h Law.sumAssoc (by simp) v

/-- **The pipeline's answer, read as a flat sum — under its named law.**

    `Pipeline.lean`'s theorems end at the committed two-level fold, because that
    is what the hardware computes and stating it that way keeps them exact.
    `Backprop.lean`'s theorems end at `Expr.sum`, because that is what `sderiv`
    produces.  This law is what joins the two ends, so that "the launched
    pipeline computes the layer's gradient" is a theorem rather than an informal
    assembly — with `Law.laneRegroup` visible in its type,
    which is the entire point.  Nothing here is claimed to hold at `Float32`;
    what is claimed is that the only thing separating the two halves is one
    named, measured, registry-listed law. -/
theorem strided_eq_flatSum (h : AllHold [Law.laneRegroup])
    (memA memB : Nat → Float32) (K : Nat) :
    bflyFold (dotStridedLane memA memB
        (fun i l => i * 32 + l.val) (fun i l => i * 32 + l.val) K) ⟨0, by decide⟩
      = (List.range (K * 32)).foldl
          (fun a i => NumOps.add a (NumOps.mul (memA i) (memB i))) (NumOps.ofNat 0) := by
  have h0 := stridedRegroup_of_laneRegroup (h Law.laneRegroup (by simp))
    (fun i => NumOps.mul (memA i) (memB i)) 0 K
  simp only [Nat.zero_add] at h0
  exact h0

/-- **The quad schedule's answer, read as the same flat sum, under the same law.**

    The strided and quad walks are different functions at `Float32` — different
    fold shapes over different orders — and this is the sense in which they
    nonetheless compute one thing.  It costs `Law.laneRegroup` and a coverage
    proof (`quad_idx_perm`), and no second law: a schedule earns the
    reassociation by showing it reads each element once. -/
theorem quad_eq_flatSum_at (h : AllHold [Law.laneRegroup])
    (memA memB : Nat → Float32) (bA bB K : Nat) :
    bflyFold (dotLane memA memB
        (fun i l => i * 128 + l.val * 4 + bA) (fun i l => i * 128 + l.val * 4 + bB) K)
        ⟨0, by decide⟩
      = (List.range (K * 128)).foldl
          (fun a i => NumOps.add a (NumOps.mul (memA (i + bA)) (memB (i + bB))))
          (NumOps.ofNat 0) := by
  have e : K*4*32 = K*128 := by omega
  have hl := h Law.laneRegroup (by simp)
    (fun i => NumOps.mul (memA (i + bA)) (memB (i + bB))) (K*4)
    (fun m l => (m/4)*128 + l.val*4 + m%4)
    (by rw [e]; exact quad_idx_perm K)
  rw [e] at hl
  have hd : (fun l : Lane => dotLane memA memB
        (fun i l => i * 128 + l.val * 4 + bA) (fun i l => i * 128 + l.val * 4 + bB) K l)
      = (fun l : Lane => (List.range (K*4)).foldl
          (fun acc m => NumOps.add acc
            (NumOps.mul (memA ((m/4)*128 + l.val*4 + m%4 + bA))
                        (memB ((m/4)*128 + l.val*4 + m%4 + bB)))) (NumOps.ofNat 0)) := by
    funext l
    rw [dotLane_flat memA memB _ _ l K]
    simp only [Nat.add_right_comm]
  exact (congrArg (fun v => bflyFold v (⟨0, by decide⟩ : Lane)) hd).trans hl

/-- The blocked walk gives lane `l` the contiguous run `l*K … l*K+K-1`, so the
    warp covers `0 … K*32-1` — the same elements the interleaved walk reads,
    visited with the two loops swapped. -/
theorem blocked_idx_perm (K : Nat) :
    (((List.range K).flatMap
        (fun i => (List.finRange 32).map (fun l : Lane => l.val * K + i))).Perm
      (List.range (K * 32))) := by
  have hfr : ∀ F : Nat → Nat,
      (List.finRange 32).map (fun l : Lane => F l.val) = (List.range 32).map F := by
    intro F; rfl
  rw [show (fun i => (List.finRange 32).map (fun l : Lane => l.val * K + i))
        = (fun i => (List.range 32).map (fun l => l * K + i)) from
      funext (fun i => hfr (fun l => l * K + i))]
  refine (flatMap_map_comm (fun i l => l * K + i) 32 K).trans ?_
  rw [range_flat_blocks K 32]
  have hc : 32 * K = K * 32 := by omega
  rw [hc]

/-- **The blocked schedule's answer, as the same flat sum.**

    It reuses the strided kernel and the strided fold verbatim — only the walk
    differs.  So this costs exactly one thing beyond what `strided_eq_flatSum_at`
    already cost: `blocked_idx_perm`.  That is the claim `Sched` is designed to
    make cheap. -/
theorem blocked_eq_flatSum_at (h : AllHold [Law.laneRegroup])
    (memA memB : Nat → Float32) (bA bB K : Nat) :
    bflyFold (dotStridedLane memA memB
        (fun i l => l.val * K + i + bA) (fun i l => l.val * K + i + bB) K)
        ⟨0, by decide⟩
      = (List.range (K * 32)).foldl
          (fun a i => NumOps.add a (NumOps.mul (memA (i + bA)) (memB (i + bB))))
          (NumOps.ofNat 0) :=
  h Law.laneRegroup (by simp)
    (fun i => NumOps.mul (memA (i + bA)) (memB (i + bB))) K
    (fun i l => l.val * K + i)
    (blocked_idx_perm K)

/-- The strided counterpart at a row base — the shape a matrix row walk emits,
    where the base is added *after* the lane offset. -/
theorem strided_eq_flatSum_at (h : AllHold [Law.laneRegroup])
    (memA memB : Nat → Float32) (bA bB K : Nat) :
    bflyFold (dotStridedLane memA memB
        (fun i l => i * 32 + l.val + bA) (fun i l => i * 32 + l.val + bB) K)
        ⟨0, by decide⟩
      = (List.range (K * 32)).foldl
          (fun a i => NumOps.add a (NumOps.mul (memA (i + bA)) (memB (i + bB))))
          (NumOps.ofNat 0) :=
  h Law.laneRegroup (by simp)
    (fun i => NumOps.mul (memA (i + bA)) (memB (i + bB))) K
    (fun j l => j * 32 + l.val)
    (by rw [strided_idx_range K])

/-- **The regrouping at an arbitrary row base and summand** — the law itself,
    named so call sites do not have to spell out the registry lookup.  Softmax's
    denominator uses it at `f i = exp(zᵢ − max)`, `base = cta·seqLen`. -/
theorem strided_regroup_at (h : AllHold [Law.laneRegroup])
    (f : Nat → Float32) (base K : Nat) :
    bflyFoldOp (fun a b => NumOps.add a b)
        (fun l => (List.range K).foldl
          (fun acc j => NumOps.add acc (f (base + (j * 32 + l.val)))) NumOps.zero)
        ⟨0, by decide⟩
      = (List.range (K * 32)).foldl
          (fun a i => NumOps.add a (f (base + i))) NumOps.zero :=
  stridedRegroup_of_laneRegroup (h Law.laneRegroup (by simp)) f base K

/-- **The butterfly is lane-uniform, under its named law.**

    The bridge `softmax_stores_tail` needs: all 32 lanes hold the same reduction
    result, so the remainder pass — where they all write one address — writes
    one value.  `Law.combinerComm` is visible in the type, which is the point. -/
theorem bfly_lane_uniform_add (h : AllHold [Law.combinerComm]) (v : Lane → Float32)
    (l l' : Lane) :
    bflyFoldOp (fun a b => NumOps.add a b) v l = bflyFoldOp (fun a b => NumOps.add a b) v l' :=
  bflyFoldOp_const _ (h Law.combinerComm (by simp)).1 v l l'

theorem bfly_lane_uniform_max (h : AllHold [Law.combinerComm]) (v : Lane → Float32)
    (l l' : Lane) :
    bflyFoldOp (fun a b => NumOps.max a b) v l = bflyFoldOp (fun a b => NumOps.max a b) v l' :=
  bflyFoldOp_const _ (h Law.combinerComm (by simp)).2 v l l'

/-- The laws a "launched pipeline = `sderiv`" claim rests on, as a list.

    Anyone quoting the composite must quote this too. -/
def gradientPipelineLaws : List Law := [Law.expIsEx2, Law.laneRegroup]

end AlgorithmLib.ML
