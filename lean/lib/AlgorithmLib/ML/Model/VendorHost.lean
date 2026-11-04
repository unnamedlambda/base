module
public import AlgorithmLib.ML.Model.Vendor
meta import AlgorithmLib.ML.Model.Vendor
public import AlgorithmLib.ML.Launch.AppRun
meta import AlgorithmLib.ML.Launch.AppRun
public import AlgorithmLib.ML.Launch.BodyHost
meta import AlgorithmLib.ML.Launch.BodyHost
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The device's products, as the vendor's GEMMs

`VendorGemmHonest` is what the device must be honest about for a
strided-batched product at a batch of one with `α = 1` and `β = 0`: the
output's word `r + c · m` is the vendor's sum of row `r` of `op(A)` and column
`c` of `op(B)` (`cublasSgemmResult`), read from the operands' words as cuBLAS
lays them out column-major, and its words past `m · n` are as they were. It is
the vendor's counterpart of `Honest` for a kernel.
-/

namespace AlgorithmLib.ML

open AlgorithmLib AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Device

/-- Where cuBLAS reads element `(r, t)` of `op(A)`, `A` of leading dimension
    `ld`, transposed where `ta` is `1`. -/
def cublasAt (ta ld r t : Nat) : Nat := if ta = 1 then r * ld + t else r + t * ld

/-- Where cuBLAS reads element `(t, c)` of `op(B)`, `B` of leading dimension
    `ld`, transposed where `tb` is `1`. -/
def cublasBt (tb ld t c : Nat) : Nat := if tb = 1 then c + t * ld else t + c * ld

/-- The vendor's answer for output element `j` of product `(ta, tb, m, n, k)`
    over operands `a` and `b`, as a word. -/
noncomputable def gemmWord (ta tb m n k : Nat) (a b : ByteArray) (j : Nat) (hj : j < m * n) : UInt64 :=
  have hm : 0 < m := Nat.pos_of_ne_zero fun e => by subst e; simp at hj
  (F32.ofFloat (cublasSgemmResult m n k
    (fun r t => (F32.mk (loadWord a (cublasAt ta (if ta = 1 then k else m) r t))).toFloat)
    (fun t c => (F32.mk (loadWord b (cublasBt tb (if tb = 1 then n else k) t c))).toFloat)
    ⟨j % m, Nat.mod_lt _ hm⟩ ⟨j / m, (Nat.div_lt_iff_lt_mul hm).mpr (by rw [Nat.mul_comm]; exact hj)⟩)).bits

/-- **What the device must be honest about for its products**: the vendor's
    answer to a strided-batched product at a batch of one with `α = 1` and
    `β = 0` holds at each output element the vendor's GEMM of its operands,
    and past them what the output held. -/
def VendorGemmHonest (v : VendorCall → List ByteArray → ByteArray) : Prop :=
  ∀ (ta tb m n k A B C : Nat) (a b c : ByteArray),
    (∀ j (hj : j < m * n), loadWord (v (stridedCall 0x3f800000 0 0 0 0 (ta, tb, m, n, k, A, B, C)) [a, b, c]) j
      = gemmWord ta tb m n k a b j hj) ∧
    (∀ j, m * n ≤ j → loadWord (v (stridedCall 0x3f800000 0 0 0 0 (ta, tb, m, n, k, A, B, C)) [a, b, c]) j
      = loadWord c j)

/-- **A contraction's shape is cuBLAS's column-major reading of the product
    the host issues for it**: the tape lays `A` and `B` where cuBLAS reads
    `op(A)` and `op(B)` for `TOp.vendorGemm`'s transposes and leading
    dimensions. -/
theorem gemmShape_cublas {op : TOp} {ta tb m n k A B C : Nat}
    (hg : op.vendorGemm = some (ta, tb, m, n, k, A, B, C)) :
    op.gemmShape = some (m, n, k, A, cublasAt ta (if ta = 1 then k else m), B,
      cublasBt tb (if tb = 1 then n else k), C) := by
  cases op with
  | mv bk w x o b i ow bA =>
      cases bk with
      | proven => simp [TOp.vendorGemm] at hg
      | vendor v =>
          simp only [TOp.vendorGemm, Option.some.injEq, Prod.mk.injEq] at hg
          obtain ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ := hg
          simp only [TOp.gemmShape, Option.some.injEq, Prod.mk.injEq, true_and, and_true]
          constructor <;> funext p q <;> simp [cublasAt, cublasBt] <;> omega
  | mvT bk w d o b i ow =>
      cases bk with
      | proven => simp [TOp.vendorGemm] at hg
      | vendor v =>
          simp only [TOp.vendorGemm, Option.some.injEq, Prod.mk.injEq] at hg
          obtain ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ := hg
          simp only [TOp.gemmShape, Option.some.injEq, Prod.mk.injEq, true_and, and_true]
          constructor <;> funext p q <;> simp [cublasAt, cublasBt] <;> omega
  | outer bk d x o b i ow =>
      cases bk with
      | proven => simp [TOp.vendorGemm] at hg
      | vendor v =>
          simp only [TOp.vendorGemm, Option.some.injEq, Prod.mk.injEq] at hg
          obtain ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ := hg
          simp only [TOp.gemmShape, Option.some.injEq, Prod.mk.injEq, true_and, and_true]
          constructor <;> funext p q <;> simp [cublasAt, cublasBt] <;> omega
  | _ => simp [TOp.vendorGemm] at hg

/-- An operation the vendor does not perform has no product's shape. -/
theorem gemmShape_none {op : TOp} (h : op.vendorGemm = none) : op.gemmShape = none := by
  unfold TOp.gemmShape
  split <;> simp_all [TOp.vendorGemm]

theorem idx_lt {r m t k : Nat} (hr : r < m) (ht : t < k) : r * k + t < m * k := by
  have : (r + 1) * k ≤ m * k := Nat.mul_le_mul_right _ hr
  rw [Nat.succ_mul] at this; omega

theorem cublasAt_lt {ta m k r t : Nat} (hr : r < m) (ht : t < k) :
    cublasAt ta (if ta = 1 then k else m) r t < m * k := by
  by_cases h : ta = 1
  · simp only [cublasAt, h, if_true]; exact idx_lt hr ht
  · simp only [cublasAt, h, if_false]
    have := idx_lt ht hr
    rw [Nat.mul_comm k m] at this; omega

theorem cublasBt_lt {tb n k t c : Nat} (ht : t < k) (hc : c < n) :
    cublasBt tb (if tb = 1 then n else k) t c < k * n := by
  by_cases h : tb = 1
  · simp only [cublasBt, h, if_true]
    have := idx_lt ht hc
    omega
  · simp only [cublasBt, h, if_false]
    have := idx_lt hc ht
    rw [Nat.mul_comm n k] at this; omega

theorem vendorStep_run {v : VendorCall → List ByteArray → ByteArray} {c : VendorCall} {ins : List Nat} {out : Nat}
    {mem : DevMem} (h : (v c (ins.map mem)).size = (mem out).size) (b : Nat) :
    (vendorStep v c ins out).run mem b = if b = out then v c (ins.map mem) else mem b := by
  simp only [vendorStep, h, bne_self_eq_false, Bool.false_eq_true, if_false, setBuf]

theorem denV_gemm {hw : Kernel.Hw} {op : TOp} {m n k : Nat} {A B C : Buf} {ai bi : Nat → Nat → Nat}
    (hg : op.gemmShape = some (m, n, k, A, ai, B, bi, C)) (mem : Buf → Nat → F32) (b a : Nat) :
    TOp.denV hw op mem b a =
      if h : b = C ∧ a < m * n then
        F32.ofFloat (cublasSgemmResult m n k (fun r t => (mem A (ai r t)).toFloat)
          (fun t c => (mem B (bi t c)).toFloat)
          ⟨a % m, Nat.mod_lt _ (Nat.pos_of_ne_zero fun e => by subst e; simp at h)⟩
          ⟨a / m, (Nat.div_lt_iff_lt_mul (Nat.pos_of_ne_zero fun e => by subst e; simp at h)).mpr
            (by rw [Nat.mul_comm]; exact h.2)⟩)
      else mem b a := by
  simp only [TOp.denV, hg]

/-- **A contraction's product, on a device honest about its products, leaves
    every word as the tape read with its contractions the vendor's does**:
    the output's words the vendor's GEMM of its operands' words, every other
    word as it was. -/
theorem gemm_words (hw : Kernel.Hw) {v : VendorCall → List ByteArray → ByteArray} (hv : VendorGemmHonest v)
    (hvk : Contracts.VendorKeeps v) {op : TOp} {ta tb m n k A B C : Nat}
    (hg : op.vendorGemm = some (ta, tb, m, n, k, A, B, C)) (hmk : m * k ≤ 2 ^ 32) (hkn : k * n ≤ 2 ^ 32)
    (mem : DevMem) :
    ∀ b j, j < 2 ^ 32 →
      loadWord ((vendorStep v (stridedCall 0x3f800000 0 0 0 0 (ta, tb, m, n, k, A, B, C)) [A, B, C] C).run mem b) j
        = (TOp.denV hw op (wordView mem) b j).bits := by
  intro b j hj
  have hsize : (v (stridedCall 0x3f800000 0 0 0 0 (ta, tb, m, n, k, A, B, C)) ([A, B, C].map mem)).size
      = (mem C).size :=
    hvk _ _ _ (by simp)
  rw [vendorStep_run hsize, denV_gemm (gemmShape_cublas hg)]
  have hvC := hv ta tb m n k A B C (mem A) (mem B) (mem C)
  by_cases hb : b = C
  · rw [if_pos hb]
    by_cases hjr : j < m * n
    · rw [dif_pos ⟨hb, hjr⟩]
      simp only [List.map_cons, List.map_nil]
      rw [hvC.1 j hjr]
      have hfa : (fun (r : Fin m) (t : Fin k) => (wordView mem A (cublasAt ta (if ta = 1 then k else m) r t)).toFloat)
          = (fun (r : Fin m) (t : Fin k) =>
              (F32.mk (loadWord (mem A) (cublasAt ta (if ta = 1 then k else m) r t))).toFloat) :=
        funext fun r => funext fun t => by
          simp only [wordView, Nat.mod_eq_of_lt (Nat.lt_of_lt_of_le (cublasAt_lt r.2 t.2) hmk)]
      have hfb : (fun (t : Fin k) (c : Fin n) => (wordView mem B (cublasBt tb (if tb = 1 then n else k) t c)).toFloat)
          = (fun (t : Fin k) (c : Fin n) =>
              (F32.mk (loadWord (mem B) (cublasBt tb (if tb = 1 then n else k) t c))).toFloat) :=
        funext fun t => funext fun c => by
          simp only [wordView, Nat.mod_eq_of_lt (Nat.lt_of_lt_of_le (cublasBt_lt t.2 c.2) hkn)]
      rw [hfa, hfb]
      rfl
    · rw [dif_neg (fun h => hjr h.2)]
      simp only [List.map_cons, List.map_nil]
      rw [hvC.2 j (Nat.le_of_not_lt hjr), hb]
      simp only [wordView, Nat.mod_eq_of_lt hj]
  · rw [if_neg hb, dif_neg (fun h => hb h.1)]
    simp only [wordView, Nat.mod_eq_of_lt hj]

end AlgorithmLib.ML
