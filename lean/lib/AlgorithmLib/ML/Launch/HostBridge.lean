module
public import AlgorithmLib.Host.DrvCodes
meta import AlgorithmLib.Host.DrvCodes
public import AlgorithmLib.Host.World
meta import AlgorithmLib.Host.World
public import AlgorithmLib.ML.Launch.Declared
meta import AlgorithmLib.ML.Launch.Declared
public import AlgorithmLib.Surface.Drv
meta import AlgorithmLib.Surface.Drv
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
  # Products and vendor laws at the host

  A cuBLAS product of a program in the driver's standard layout (`Gemm`), the
  words a strided-batched call is handed for it (`Drv.App.stridedWords`), and
  the vendor routines as declared steps, each with the law that states what it
  computes (`cublasStep`, `cublasGemmStep`).
-/

namespace AlgorithmLib.ML

open AlgorithmLib.IR AlgorithmLib.HProg.Sem

/-- **A cuBLAS product** of a program in the standard layout, as
    `Drv.App.sgemm` takes it: `(ta, tb, m, n, k, A, B, C)`. -/
abbrev Gemm := Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat

/-- The words `cublasSgemmStridedBatched` is handed for product `g` on handle
    `h` at a batch of one: zero strides, the leading dimensions the shape
    implies. -/
def _root_.AlgorithmLib.Drv.App.stridedWords (a : Drv.App) (h : UInt64) : Gemm → List UInt64
  | (ta, tb, m, n, k, A, B, C) =>
      [h, UInt64.ofNat ta, UInt64.ofNat tb, UInt64.ofNat m, UInt64.ofNat n, UInt64.ofNat k,
       (regionBase .arena + UInt64.ofNat a.ONE), devPtr A, UInt64.ofNat (if ta = 1 then k else m), 0, devPtr B,
       UInt64.ofNat (if tb = 1 then n else k), 0, (regionBase .arena + UInt64.ofNat a.ZERO), devPtr C,
       UInt64.ofNat m, 0, 1]

-- ---------------------------------------------------------------------------
-- The vendor GEMV, as a declared step
-- ---------------------------------------------------------------------------

/-- **The matrix a GEMV's `A` handle denotes**, row-major with `cols` columns.
    A buffer is a flat address space; this is the one place the shape enters. -/
def matAt (mem : Buf → Nat → Float32) (aB : Buf) (rows cols : Nat) :
    Fin rows → Fin cols → Float32 := fun r c => mem aB (r.val * cols + c.val)

def vecAt (mem : Buf → Nat → Float32) (xB : Buf) (cols : Nat) :
    Fin cols → Float32 := fun c => mem xB c.val

/-- **`y = A·x` as a plan step.**

    Assumed in exactly one respect — *what* lands in `y` — and that respect is
    the *named law's* own opaque, `cublasSgemvResult`, not a second symbol
    beside it.  A step written against any other opaque would leave
    `Law.cublasIsMatvec` constraining something no plan mentions: stated,
    listed in the assumption ledger, and load-bearing for nothing.  Writing it
    this way is what makes `cublasStep_isMatvec` below provable, and with it
    the law reaches the seven `sgemv`s a layer performs.

    *Where* it can land is proven: `frame` is discharged here, so the step
    cannot touch a buffer the rest of the plan reasons about. -/
noncomputable def cublasStep (aB xB yB : Buf) (rows cols : Nat) : DeclaredStep where
  kernel := .cublasSgemv
  outs  := [yB]
  step  := fun mem b a =>
             if h : b = yB ∧ a < rows then
               cublasSgemvResult rows cols (matAt mem aB rows cols)
                 (vecAt mem xB cols) ⟨a, h.2⟩
             else mem b a
  frame := by
    intro mem b hb
    funext a
    rw [dif_neg (fun h => hb (by simp [h.1]))]

/-- **What the vendor GEMV computes, under the law that names it.**

    The statement `Law.cublasIsMatvec` exists to license: row `i` of the output
    is `Σⱼ A[i,j]·x[j]` over the buffer's own contents.  Everything a plan
    containing `cublasStep` denotes is this, up to the fold order the law
    abstracts away — which is the whole of the gap, and it is a rewrite a
    downstream spec applies rather than a sentence in a docstring. -/
theorem cublasStep_isMatvec (hl : CuBlasIsMatvec) (aB xB yB : Buf) (rows cols : Nat)
    (mem : Buf → Nat → Float32) (i : Nat) (hi : i < rows) :
    (cublasStep aB xB yB rows cols).step mem yB i
      = (List.finRange cols).foldl
          (fun acc j => NumOps.add acc
            (NumOps.mul (mem aB (i * cols + j.val)) (mem xB j.val)))
          (NumOps.ofNat 0) := by
  show (if h : yB = yB ∧ i < rows then
          cublasSgemvResult rows cols (matAt mem aB rows cols)
            (vecAt mem xB cols) ⟨i, h.2⟩
        else mem yB i) = _
  rw [dif_pos (And.intro rfl hi), hl]
  rfl

/-- **`C = A·B` as a plan step, at a batch of one.**

    The GEMM counterpart of `cublasStep`, and written the same way and for the
    same reason: against the *named law's* own opaque, `cublasSgemmResult`, so
    that `Law.cublasGemmIsSomeReassoc` constrains something a plan mentions
    rather than sitting in the registry.  Output address `a` is read row-major
    as element `(a / n, a % n)`, which is the shape entering — the one place it
    does.

    What the law does not say, and this step does not either: which operand of
    the column-major call plays `A`.  That is the lowering's business
    (`TOp.vendorGemm` picks it per operation) and a transposition is not a
    reassociation, so it could not be smuggled in here.

    *Where* it can land is proven: `frame` is discharged, so the step cannot
    touch a buffer the rest of the plan reasons about. -/
noncomputable def cublasGemmStep (aB bB cB : Buf) (m n k : Nat) : DeclaredStep where
  kernel := .cublasSgemmAt1OnStream
  outs  := [cB]
  step  := fun mem b a =>
             if h : b = cB ∧ a < m * n then
               have hn : 0 < n := Nat.pos_of_ne_zero (fun h0 => by
                 rw [h0, Nat.mul_zero] at h; exact Nat.not_lt_zero a h.2)
               cublasSgemmResult m n k (matAt mem aB m k) (matAt mem bB k n)
                 ⟨a / n, Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact h.2)⟩
                 ⟨a % n, Nat.mod_lt _ hn⟩
             else mem b a
  frame := by
    intro mem b hb
    funext a
    rw [dif_neg (fun h => hb (by simp [h.1]))]

/-- **What the vendor GEMM computes, under the law that names it.**

    Output element `(i, j)` sums that element's own `k` products, each once, in
    some association.  Weaker than `cublasStep_isMatvec` in exactly the way the
    two laws differ — there is a tree here rather than a fold — and that is the
    whole of what a batch-of-one contraction may be assumed to be. -/
theorem cublasGemmStep_isSomeReassoc (hg : CuBlasGemmIsSomeReassoc)
    (aB bB cB : Buf) (m n k : Nat) (mem : Buf → Nat → Float32)
    (i j : Nat) (hi : i < m) (hj : j < n) :
    ∃ t : SumTree, t.indices.Perm (List.range k) ∧
      (cublasGemmStep aB bB cB m n k).step mem cB (i * n + j)
        = t.eval (gemmTerm (matAt mem aB m k) (matAt mem bB k n) ⟨i, hi⟩ ⟨j, hj⟩) := by
  have ha : i * n + j < m * n := by
    calc i * n + j < i * n + n := Nat.add_lt_add_left hj _
      _ = (i + 1) * n := by rw [Nat.succ_mul]
      _ ≤ m * n := Nat.mul_le_mul_right n hi
  obtain ⟨t, hperm, ht⟩ :=
    hg m n k (matAt mem aB m k) (matAt mem bB k n) ⟨i, hi⟩ ⟨j, hj⟩
  refine ⟨t, hperm, ?_⟩
  show (if h : cB = cB ∧ i * n + j < m * n then
          cublasSgemmResult m n k (matAt mem aB m k) (matAt mem bB k n)
            ⟨(i * n + j) / n, _⟩ ⟨(i * n + j) % n, _⟩
        else mem cB (i * n + j)) = _
  rw [dif_pos (And.intro rfl ha)]
  have hd : (i * n + j) / n = i := by
    rw [Nat.mul_comm, Nat.mul_add_div (by omega), Nat.div_eq_of_lt hj, Nat.add_zero]
  have hm : (i * n + j) % n = j := by
    rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hj]
  rw [show (⟨(i * n + j) / n, by omega⟩ : Fin m) = ⟨i, hi⟩ from Fin.ext hd,
      show (⟨(i * n + j) % n, by omega⟩ : Fin n) = ⟨j, hj⟩ from Fin.ext hm]
  exact ht

end AlgorithmLib.ML
