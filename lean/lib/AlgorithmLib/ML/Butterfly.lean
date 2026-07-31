import AlgorithmLib.ML.Schema

/-!
  # The butterfly leaves the same value in every lane

  Softmax's remainder pass has all 32 lanes write the *same* address, so the
  kernel is correct only if all 32 lanes hold the *same* row maximum and the
  same reciprocal sum.  `softmax_stores_tail` takes that as a hypothesis
  (`huni`), and until it is discharged softmax cannot be a `StageSpec`.

  The fact needed is that `bflyFoldOp g v` is constant in the lane.  It is worth
  being precise about what that costs, because the obvious guess is wrong:

  * It does **not** need associativity.  Every lane walks a tree of the *same
    shape* over the *same* 32 leaves; only the argument order *within each node*
    differs.  So commutativity of the combiner alone suffices — which is why
    this is a much weaker requirement than `Law.sumAssoc`, and why it is *true*
    at `Float32` where `sumAssoc` is measurably false.
  * It does need commutativity, and `Float32.add` is opaque in Lean, so that one
    fact is a named law (`Law.combinerComm`).  Everything else here is proven.

  The structure is: `Inv m u` — `u` is unchanged by flipping the bits of `m`;
  each butterfly round establishes `Inv` for its own mask and preserves it for
  every earlier one; five masks covering five bits then force constancy.
-/

namespace AlgorithmLib.ML

/-- `u` is unchanged by flipping the lane bits in `m`. -/
def LaneInv (m : Nat) (u : Lane → Float32) : Prop := ∀ l : Lane, u (xorLane l m) = u l

theorem xorLane_xorLane (l : Lane) (m m' : Nat) (hm : m < 32) (hm' : m' < 32) :
    xorLane (xorLane l m) m' = xorLane l (m ^^^ m') := by
  have hl : l.val < 32 := l.isLt
  refine Fin.ext ?_
  show (((l.val ^^^ m) % 32) ^^^ m') % 32 = (l.val ^^^ (m ^^^ m')) % 32
  have h1 : l.val ^^^ m < 32 := Nat.xor_lt_two_pow (n := 5) hl hm
  rw [Nat.mod_eq_of_lt h1, Nat.xor_assoc]

theorem xorLane_zero (l : Lane) : xorLane l 0 = l := by
  refine Fin.ext ?_
  show (l.val ^^^ 0) % 32 = l.val
  rw [Nat.xor_zero]
  exact Nat.mod_eq_of_lt l.isLt

theorem xorLane_self (l : Lane) (m : Nat) (hm : m < 32) :
    xorLane (xorLane l m) m = l := by
  rw [xorLane_xorLane l m m hm hm, Nat.xor_self]
  exact xorLane_zero l

/-- **A round makes its own mask invisible** — by commutativity, and nothing
    else.  `g (u l) (u (l^m))` and `g (u (l^m)) (u l)` are the same node with
    its two arguments swapped. -/
theorem bflyStepOp_inv_self (g : Float32 → Float32 → Float32)
    (hc : ∀ a b, g a b = g b a) (m : Nat) (hm : m < 32) (u : Lane → Float32) :
    LaneInv m (bflyStepOp g m u) := by
  intro l
  show g (u (xorLane l m)) (u (xorLane (xorLane l m) m)) = g (u l) (u (xorLane l m))
  rw [xorLane_self l m hm, hc]

/-- **…and preserves every earlier one.**  Flipping `m'` in the lane permutes
    the round's two operands consistently, so the node is unchanged. -/
theorem bflyStepOp_inv_of (g : Float32 → Float32 → Float32) (m m' : Nat)
    (hm : m < 32) (hm' : m' < 32) (u : Lane → Float32) (h : LaneInv m' u) :
    LaneInv m' (bflyStepOp g m u) := by
  intro l
  show g (u (xorLane l m')) (u (xorLane (xorLane l m') m)) = g (u l) (u (xorLane l m))
  rw [h l, xorLane_xorLane l m' m hm' hm, Nat.xor_comm m' m,
      ← xorLane_xorLane l m m' hm hm', h (xorLane l m)]

/-- Bit `i` of `b`, as the value `0` or `2^i`. -/
def bitAt (b i : Nat) : Nat := (b >>> i % 2) <<< i

theorem bitAt_xor_decomp (b : Fin 32) :
    b.val = bitAt b.val 0 ^^^ bitAt b.val 1 ^^^ bitAt b.val 2
              ^^^ bitAt b.val 3 ^^^ bitAt b.val 4 := by
  revert b
  decide

theorem bitAt_cases0 (b : Fin 32) : bitAt b.val 0 = 0 ∨ bitAt b.val 0 = 1 := by revert b; decide
theorem bitAt_cases1 (b : Fin 32) : bitAt b.val 1 = 0 ∨ bitAt b.val 1 = 2 := by revert b; decide
theorem bitAt_cases2 (b : Fin 32) : bitAt b.val 2 = 0 ∨ bitAt b.val 2 = 4 := by revert b; decide
theorem bitAt_cases3 (b : Fin 32) : bitAt b.val 3 = 0 ∨ bitAt b.val 3 = 8 := by revert b; decide
theorem bitAt_cases4 (b : Fin 32) : bitAt b.val 4 = 0 ∨ bitAt b.val 4 = 16 := by revert b; decide

theorem laneInv_zero (u : Lane → Float32) : LaneInv 0 u := by
  intro l
  rw [xorLane_zero]

/-- Invariance under a mask that is either `0` or one of the five. -/
theorem laneInv_of_cases {u : Lane → Float32} {m p : Nat} (h : m = 0 ∨ m = p)
    (hp : LaneInv p u) : LaneInv m u := by
  rcases h with h | h
  · rw [h]; exact laneInv_zero u
  · rw [h]; exact hp

/-- **Five invariances over five bits force constancy.**

    Any lane differs from lane `0` in some subset of the five bits, and each is
    individually invisible, so composing the five gives the whole warp. -/
theorem laneInv_const {u : Lane → Float32}
    (h1 : LaneInv 1 u) (h2 : LaneInv 2 u) (h4 : LaneInv 4 u)
    (h8 : LaneInv 8 u) (h16 : LaneInv 16 u) (l l' : Lane) : u l = u l' := by
  have hb : (l.val ^^^ l'.val) < 32 :=
    Nat.xor_lt_two_pow (n := 5) l.isLt l'.isLt
  obtain ⟨b, hbd⟩ : ∃ b : Fin 32, b.val = l.val ^^^ l'.val := ⟨⟨_, hb⟩, rfl⟩
  -- peel the five bits off `b`, each step invisible to `u`
  have e0 := laneInv_of_cases (bitAt_cases0 b) h1
  have e1 := laneInv_of_cases (bitAt_cases1 b) h2
  have e2 := laneInv_of_cases (bitAt_cases2 b) h4
  have e3 := laneInv_of_cases (bitAt_cases3 b) h8
  have e4 := laneInv_of_cases (bitAt_cases4 b) h16
  have b0 : bitAt b.val 0 < 32 := by rcases bitAt_cases0 b with h | h <;> omega
  have b1 : bitAt b.val 1 < 32 := by rcases bitAt_cases1 b with h | h <;> omega
  have b2 : bitAt b.val 2 < 32 := by rcases bitAt_cases2 b with h | h <;> omega
  have b3 : bitAt b.val 3 < 32 := by rcases bitAt_cases3 b with h | h <;> omega
  have b4 : bitAt b.val 4 < 32 := by rcases bitAt_cases4 b with h | h <;> omega
  -- `l` xor'd with all five bits of `b` is `l'`
  have hcompose : xorLane l b.val = l' := by
    refine Fin.ext ?_
    show (l.val ^^^ b.val) % 32 = l'.val
    rw [hbd, ← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor,
        Nat.mod_eq_of_lt l'.isLt]
  have hstep : xorLane l b.val
      = xorLane (xorLane (xorLane (xorLane (xorLane l (bitAt b.val 0))
          (bitAt b.val 1)) (bitAt b.val 2)) (bitAt b.val 3)) (bitAt b.val 4) := by
    rw [xorLane_xorLane l _ _ b0 b1, xorLane_xorLane l _ _
          (Nat.xor_lt_two_pow (n := 5) b0 b1) b2,
        xorLane_xorLane l _ _ (Nat.xor_lt_two_pow (n := 5)
          (Nat.xor_lt_two_pow (n := 5) b0 b1) b2) b3,
        xorLane_xorLane l _ _ (Nat.xor_lt_two_pow (n := 5)
          (Nat.xor_lt_two_pow (n := 5) (Nat.xor_lt_two_pow (n := 5) b0 b1) b2) b3) b4]
    exact congrArg (xorLane l) (bitAt_xor_decomp b)
  rw [← hcompose, hstep, e4 _, e3 _, e2 _, e1 _, e0 _]

/-- **The butterfly is lane-uniform, given only a commutative combiner.** -/
theorem bflyFoldOp_const (g : Float32 → Float32 → Float32) (hc : ∀ a b, g a b = g b a)
    (v : Lane → Float32) (l l' : Lane) : bflyFoldOp g v l = bflyFoldOp g v l' := by
  refine laneInv_const (u := bflyFoldOp g v) ?_ ?_ ?_ ?_ ?_ l l'
  · exact bflyStepOp_inv_self g hc 1 (by decide) _
  · exact bflyStepOp_inv_of g 1 2 (by decide) (by decide) _
      (bflyStepOp_inv_self g hc 2 (by decide) _)
  · exact bflyStepOp_inv_of g 1 4 (by decide) (by decide) _
      (bflyStepOp_inv_of g 2 4 (by decide) (by decide) _
        (bflyStepOp_inv_self g hc 4 (by decide) _))
  · exact bflyStepOp_inv_of g 1 8 (by decide) (by decide) _
      (bflyStepOp_inv_of g 2 8 (by decide) (by decide) _
        (bflyStepOp_inv_of g 4 8 (by decide) (by decide) _
          (bflyStepOp_inv_self g hc 8 (by decide) _)))
  · exact bflyStepOp_inv_of g 1 16 (by decide) (by decide) _
      (bflyStepOp_inv_of g 2 16 (by decide) (by decide) _
        (bflyStepOp_inv_of g 4 16 (by decide) (by decide) _
          (bflyStepOp_inv_of g 8 16 (by decide) (by decide) _
            (bflyStepOp_inv_self g hc 16 (by decide) _))))

end AlgorithmLib.ML
