import AlgorithmLib.ML.Schema

/-!
  # MXFP4 — block-scaled 4-bit weights

  `Quant.lean` covers int8, where the win is that integer arithmetic is *lawful*
  and the accumulator bound is a real theorem.  MXFP4 is the other format a
  frontier model actually ships in: each group of 32 weights shares one
  power-of-two scale, and every element is four bits.

  ## The format, exactly

  An element is **E2M1**: one sign bit and three magnitude bits selecting from
  eight values — `0, 0.5, 1, 1.5, 2, 3, 4, 6`.  That is the whole table, and it
  is finite, so `fp4Mag` is written out rather than computed.  A block carries an
  **E8M0** scale: a power of two, stored as its exponent.

  ## What is proven here

  * `fp4_roundtrip` — encoding a sign/magnitude pair and decoding recovers it.
    The code layout (`sign * 8 + magnitude`) is where an off-by-one in the sign
    bit would live, and it is checked rather than assumed.
  * `fp4Mag_cases` — the decode table is exactly the eight E2M1 magnitudes,
    which is what a scale gets chosen against.  Stated as an enumeration, not as
    `≤ 6`: `Float32` comparison is a native operation with no kernel reduction,
    so a numeric bound is not provable here at all without an IEEE model.
  * `mxDot_spec` — **quantization changes the weights, not the algorithm.**  A
    dequantized dot product is exactly the spec's fold over the dequantized
    values, in the same committed order.  So a quantized kernel is an instance
    of the *same* proven schema (`warpDotV4`), with `mxDeq` supplying the buffer
    — no new kernel proof, which is the whole point of schemas.

  ## What is not proven, and why

  **Nibble unpacking is not modelled.**  Two elements share a byte, and the
  machine here is element-addressed: `mem : Buf → Nat → Float32` has no
  byte layout and no bit operations.  So "load a packed byte, extract the high
  or low nibble" is outside this model, exactly as it is for int8.  The
  dequantized values are taken as given in a buffer; what is proven is
  everything downstream of that.

  Also not proven, deliberately: that rounding a `Float32` to the nearest code
  is optimal.  That is a claim about real-number distance and needs an IEEE
  model.
-/

namespace AlgorithmLib.ML

open NumOps

-- ---------------------------------------------------------------------------
-- The element format
-- ---------------------------------------------------------------------------

/-- The eight E2M1 magnitudes.  Finite, so written out. -/
def fp4Mag : Fin 8 → Float32
  | ⟨0, _⟩ => 0.0
  | ⟨1, _⟩ => 0.5
  | ⟨2, _⟩ => 1.0
  | ⟨3, _⟩ => 1.5
  | ⟨4, _⟩ => 2.0
  | ⟨5, _⟩ => 3.0
  | ⟨6, _⟩ => 4.0
  | ⟨7, _⟩ => 6.0

/-- A 4-bit code: sign in the high bit, magnitude in the low three. -/
def fp4Code (s : Bool) (m : Fin 8) : Nat := (if s then 8 else 0) + m.val

/-- Decode a 4-bit code.  Codes `≥ 16` cannot occur; they decode to zero so the
    function is total. -/
def fp4Val (c : Nat) : Float32 :=
  if h : c < 8 then fp4Mag ⟨c, h⟩
  else if h' : c < 16 then NumOps.neg (fp4Mag ⟨c - 8, by omega⟩)
  else 0.0

theorem fp4Code_false (m : Fin 8) : fp4Code false m = m.val := by simp [fp4Code]
theorem fp4Code_true (m : Fin 8) : fp4Code true m = 8 + m.val := by simp [fp4Code]

/-- **The code layout round-trips.**  Decoding what the encoder produced gives
    back the sign and magnitude it started from — where an off-by-one in the
    sign bit would show up. -/
theorem fp4_roundtrip (s : Bool) (m : Fin 8) :
    fp4Val (fp4Code s m) = if s then NumOps.neg (fp4Mag m) else fp4Mag m := by
  have hm := m.isLt
  cases s with
  | false =>
      show (if h : fp4Code false m < 8 then _ else _) = _
      rw [dif_pos (show fp4Code false m < 8 by rw [fp4Code_false]; omega)]
      show fp4Mag ⟨(fp4Code false m), _⟩ = _
      congr 1
      exact Fin.ext (fp4Code_false m)
  | true =>
      show (if h : fp4Code true m < 8 then _ else _) = _
      rw [dif_neg (show ¬ fp4Code true m < 8 by rw [fp4Code_true]; omega),
          dif_pos (show fp4Code true m < 16 by rw [fp4Code_true]; omega)]
      show NumOps.neg (fp4Mag ⟨fp4Code true m - 8, _⟩) = _
      congr 2
      refine Fin.ext ?_
      show fp4Code true m - 8 = m.val
      have h := fp4Code_true m
      omega

/-- **The table is exactly the eight E2M1 magnitudes.**

    Stated as an enumeration rather than as `≤ 6`, because a numeric bound on a
    `Float32` is not something this development can prove: `Float32` comparison
    is a native operation with no kernel reduction, so neither `decide` nor
    `rfl` can settle `0.5 ≤ 6.0`.  Proving it would need an IEEE model, which
    this development deliberately does not carry.

    The enumeration is what a scale is actually chosen against, and it *is*
    provable — by unfolding the table, with no float arithmetic involved. -/
theorem fp4Mag_cases (m : Fin 8) :
    fp4Mag m = 0.0 ∨ fp4Mag m = 0.5 ∨ fp4Mag m = 1.0 ∨ fp4Mag m = 1.5
    ∨ fp4Mag m = 2.0 ∨ fp4Mag m = 3.0 ∨ fp4Mag m = 4.0 ∨ fp4Mag m = 6.0 := by
  rcases m with ⟨v, hv⟩
  match v, hv with
  | 0, _ => exact Or.inl rfl
  | 1, _ => exact Or.inr (Or.inl rfl)
  | 2, _ => exact Or.inr (Or.inr (Or.inl rfl))
  | 3, _ => exact Or.inr (Or.inr (Or.inr (Or.inl rfl)))
  | 4, _ => exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
  | 5, _ => exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
  | 6, _ => exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))))
  | 7, _ => exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr rfl))))))

-- ---------------------------------------------------------------------------
-- Blocks
-- ---------------------------------------------------------------------------

/-- Elements per scale.  32 is the MX group size, and it is also the warp
    width — one block per warp is the natural mapping. -/
abbrev MXGroup : Nat := 32

/-- A microscaled block: one shared power-of-two scale, 32 four-bit codes. -/
structure MXBlock where
  /-- The E8M0 scale, already as a `Float32` power of two. -/
  scale : Float32
  codes : Fin MXGroup → Nat

/-- Dequantize one element. -/
def mxDeq (b : MXBlock) (i : Fin MXGroup) : Float32 :=
  NumOps.mul b.scale (fp4Val (b.codes i))

/-- A whole block, as the buffer contents a kernel would read. -/
def mxDeqBuf (b : MXBlock) (base : Nat) (j : Nat) : Float32 :=
  if h : j - base < MXGroup ∧ base ≤ j then mxDeq b ⟨j - base, h.1⟩ else 0.0

-- ---------------------------------------------------------------------------
-- Quantization changes the weights, not the algorithm
-- ---------------------------------------------------------------------------

/-- The dot product of a dequantized block with an activation vector, in the
    fold order the spec commits to. -/
def mxDot (b : MXBlock) (x : Fin MXGroup → Float32) : Float32 :=
  (List.finRange MXGroup).foldl
    (fun acc i => NumOps.add acc (NumOps.mul (mxDeq b i) (x i))) (NumOps.ofNat 0)

/-- **A quantized dot is the spec's dot over dequantized weights.**

    This is the statement that matters: the quantized path is not a different
    algorithm to be re-proven, it is the *same* fold with different numbers in
    it.  Anything already proven about that fold — `warpDotV4`, and every kernel
    instantiating it — therefore applies to the quantized kernel unchanged.

    Exact `Float32` equality; no epsilon, because there is no approximation
    here.  The approximation lives in the *choice* of codes, which is upstream
    of the kernel and is where it belongs. -/
theorem mxDot_spec (b : MXBlock) (x : Fin MXGroup → Float32) :
    mxDot b x
      = (List.finRange MXGroup).foldl
          (fun acc i => NumOps.add acc (NumOps.mul (mxDeqBuf b 0 i.val) (x i)))
          (NumOps.ofNat 0) := by
  have hpt : ∀ i : Fin MXGroup, mxDeqBuf b 0 i.val = mxDeq b i := by
    intro i
    show (if h : i.val - 0 < MXGroup ∧ 0 ≤ i.val then _ else _) = _
    rw [dif_pos ⟨by have := i.isLt; omega, Nat.zero_le _⟩]
    congr 1
  have key : ∀ (L : List (Fin MXGroup)) (init : Float32),
      L.foldl (fun acc i => NumOps.add acc (NumOps.mul (mxDeq b i) (x i))) init
        = L.foldl (fun acc i => NumOps.add acc (NumOps.mul (mxDeqBuf b 0 i.val) (x i))) init := by
    intro L
    induction L with
    | nil => intro _; rfl
    | cons a L ih => intro init; rw [List.foldl_cons, List.foldl_cons, hpt a, ih]
  exact key (List.finRange MXGroup) _

/-! The scale-choice bound — `|scale · value| ≤ 6 · scale` — needs an *ordered*
    theory of `Float32` multiplication, which this development deliberately does
    not have.  `fp4Mag_cases` is the part that can be stated without one:
    the table is a known finite set, which is the fact a scale is chosen
    against. -/

end AlgorithmLib.ML
