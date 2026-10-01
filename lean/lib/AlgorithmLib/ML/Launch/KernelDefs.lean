module
public import AlgorithmLib.ML.Launch.Dag
meta import AlgorithmLib.ML.Launch.Dag
public import AlgorithmLib.Host.DevWorld
meta import AlgorithmLib.Host.DevWorld
public import Std.Tactic.BVDecide
meta import Std.Tactic.BVDecide
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# Warp-machine stages as device kernels

A stage is proven over the warp machine's memory, `Buf → Nat → Float32`; a
device program orders steps over bytes. `StageSpec.bufStep` is the stage on
bytes: the output buffer gets, at every word the stage owns, the bits of the
stage's value there (`bufStep_owned`), and keeps its bytes everywhere else.
Given the buffers the stage's values read, it is a `BufStep`, so stages meet
the World's launches in one `DevProg` and `Device.denote_linearisation`
applies to them. `mapStage_kernel` is the elementwise schema's instance, its
read set its inputs.

A value read back from the bytes is `ofBits (toBits v)`; that this is `v` is
IEEE's and is not assumed here.
-/

namespace AlgorithmLib.ML

open AlgorithmLib.Device

/-- Byte `i` of a word, little-endian. -/
def byteOf (x : UInt32) (i : Nat) : UInt8 := (x >>> (8 * i.toUInt32)).toUInt8

/-- The word four bytes make, little-endian. -/
def u32Of (b0 b1 b2 b3 : UInt8) : UInt32 :=
  b0.toUInt32 ||| (b1.toUInt32 <<< 8) ||| (b2.toUInt32 <<< 16) ||| (b3.toUInt32 <<< 24)

theorem u32Of_bytes (x : UInt32) : u32Of (byteOf x 0) (byteOf x 1) (byteOf x 2) (byteOf x 3) = x := by
  simp only [u32Of, byteOf, show (8 * (0 : Nat).toUInt32) = 0 from rfl,
    show (8 * (1 : Nat).toUInt32) = 8 from rfl, show (8 * (2 : Nat).toUInt32) = 16 from rfl,
    show (8 * (3 : Nat).toUInt32) = 24 from rfl]
  bv_decide

/-- Word `a` of a buffer. -/
def wordAt (bs : ByteArray) (a : Nat) : UInt32 :=
  u32Of (bs.get! (4 * a)) (bs.get! (4 * a + 1)) (bs.get! (4 * a + 2)) (bs.get! (4 * a + 3))

/-- Device memory read as the warp machine's: word `a` of buffer `b` as a float. -/
def viewMem (m : DevMem) : Buf → Nat → Float32 := fun b a => Float32.ofBits (wordAt (m b) a)

theorem get_ofFn (n : Nat) (f : Fin n → UInt8) (k : Nat) (h : k < n) :
    (ByteArray.mk (Array.ofFn f)).get! k = f ⟨k, h⟩ := by
  simp [ByteArray.get!, h]

open Classical in
/-- The stage's output buffer after it runs: at a word it owns, the bits of its
    value; elsewhere, the bytes that were there. -/
noncomputable def StageSpec.bytesOut (S : StageSpec) (m : DevMem) : ByteArray :=
  let bs := m S.out
  ByteArray.mk (Array.ofFn (n := bs.size) fun k =>
    if ∃ cta, cta < S.grid ∧ S.dom cta (k.val / 4) then
      byteOf (S.step (viewMem m) S.out (k.val / 4)).toBits (k.val % 4)
    else bs.get! k.val)

/-- **What the stage leaves at a word it owns is its value's bits.** -/
theorem bufStep_owned (S : StageSpec) (m : DevMem) (a : Nat) (ha : 4 * a + 3 < (m S.out).size)
    (hown : ∃ cta, cta < S.grid ∧ S.dom cta a) :
    wordAt (S.bytesOut m) a = (S.step (viewMem m) S.out a).toBits := by
  have e : ∀ i, i < 4 → (S.bytesOut m).get! (4 * a + i)
      = byteOf (S.step (viewMem m) S.out a).toBits i := by
    intro i hi
    simp only [StageSpec.bytesOut]
    rw [get_ofFn _ _ _ (by omega)]
    have hd : (4 * a + i) / 4 = a := by omega
    have hm : (4 * a + i) % 4 = i := by omega
    rw [hd, if_pos hown, hm]
  simp only [wordAt]
  rw [show 4 * a = 4 * a + 0 from rfl, e 0 (by omega), e 1 (by omega), e 2 (by omega),
    e 3 (by omega)]
  exact u32Of_bytes _

/-- **A stage as a step on bytes**, given the buffers its values read. -/
noncomputable def StageSpec.bufStep (S : StageSpec) (R : List Nat)
    (hread : ∀ m m' : Buf → Nat → Float32, (∀ b ∈ R, m b = m' b) → m S.out = m' S.out →
      ∀ a, (∃ cta, cta < S.grid ∧ S.dom cta a) → S.step m S.out a = S.step m' S.out a) :
    BufStep where
  run m := setBuf m S.out (S.bytesOut m)
  reads := R
  writes := [S.out]
  frame m b hb := by
    simp only [List.mem_singleton] at hb
    simp [setBuf, hb]
  local_ m m' hr hw b hb := by
    simp only [List.mem_singleton] at hb
    subst hb
    have hout : m S.out = m' S.out := hw _ (List.mem_singleton_self _)
    have hv : ∀ c ∈ R, viewMem m c = viewMem m' c := fun c hc => by
      funext a; simp [viewMem, hr c hc]
    have hvo : viewMem m S.out = viewMem m' S.out := by funext a; simp [viewMem, hout]
    simp only [setBuf, StageSpec.bytesOut]
    congr 1
    rw [hout]
    refine congrArg ByteArray.mk (congrArg Array.ofFn (funext fun k => ?_))
    by_cases hown : ∃ cta, cta < S.grid ∧ S.dom cta (k.val / 4)
    · simp only [if_pos hown, hread _ _ hv hvo _ hown]
    · simp only [if_neg hown]

/-- A stage as a device kernel: its warp code, and at any binding its step. -/
noncomputable def StageSpec.kernel (S : StageSpec) (R : List Nat)
    (hread : ∀ m m' : Buf → Nat → Float32, (∀ b ∈ R, m b = m' b) → m S.out = m' S.out →
      ∀ a, (∃ cta, cta < S.grid ∧ S.dom cta a) → S.step m S.out a = S.step m' S.out a) :
    KernelDef EWStmt where
  name := "ew"
  code := S.ew
  step _ := S.bufStep R hread

/-- **The elementwise schema is a device kernel**, reading exactly its inputs. -/
noncomputable def mapStage_kernel {Γ : Nat} (spec : Expr Γ) (inB : Fin Γ → Buf) (out : Buf)
    (grid : Nat) (hio : ∀ i, inB i ≠ out) : KernelDef EWStmt :=
  (mapStage spec inB out grid hio).kernel ((List.finRange Γ).map inB) (by
    intro m m' hr _ a hown
    obtain ⟨cta, hc, hd⟩ := hown
    have hex := mapStage_exclusive spec inB out grid hio
    rw [StageSpec.step_val _ hex m cta a hc hd, StageSpec.step_val _ hex m' cta a hc hd]
    show denote (fun i => m (inB i) a) spec = denote (fun i => m' (inB i) a) spec
    congr 1
    funext i
    rw [hr (inB i) (List.mem_map.mpr ⟨i, List.mem_finRange i, rfl⟩)])

end AlgorithmLib.ML
