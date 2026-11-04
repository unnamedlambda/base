module
public import AlgorithmLib.ML.Kernel.BodyCheck
meta import AlgorithmLib.ML.Kernel.BodyCheck
public import AlgorithmLib.ML.Launch.HostLaunch
meta import AlgorithmLib.ML.Launch.HostLaunch
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# A statement kernel's words, as the device's bytes

A statement kernel reads word `j` of its buffer `b` with a four-byte load at
`bufW`: `loadWord`, the four bytes little-endian (`load_bufW`), its index in
32 bits. `wordView`
reads device memory the same way, so the warp machine's starting memory
(`ws0`) over a launch's buffers is `wordView` of those buffers wherever each
holds whole words (`ws0_wordView`).
-/

namespace AlgorithmLib.ML

open AlgorithmLib AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Device
  AlgorithmLib.ML.BodyProof AlgorithmLib.ML.BodyWarp AlgorithmLib.Kernel

/-- Word `j` of `bs`, its four bytes little-endian, as a load reads it. -/
def loadWord (bs : ByteArray) (j : Nat) : UInt64 :=
  (List.range 4).foldr (fun i acc => (acc <<< 8) ||| (bs.get! (4 * j + i)).toUInt64) 0

/-- Device memory as a statement kernel reads it: word `a` of buffer `b`,
    its index taken in 32 bits, as an f32 word. -/
def wordView (m : DevMem) : Buf → Nat → F32 := fun b a => ⟨loadWord (m b) (a % 2 ^ 32)⟩

/-- **A word load reads the word's four bytes**, little-endian. -/
theorem load_bufW {m : Mem} {nbuf b j : Nat} (hb : b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32)
    (hs : 4 * j + 4 ≤ (m.bufs b).size) :
    m.load (bufW nbuf b j) 4 = some (loadWord (m.bufs b) j) := by
  have hd := decode_bufW hb.1 hb.2.1 hb.2.2
  simp only [Mem.load, hd, Mem.region, Mem.readable, bind, Option.bind]
  rw [if_neg (by simp; omega)]
  rfl

/-- A word load that answers reads the word. -/
theorem load_bufW_some {m : Mem} {nbuf b j : Nat} (hb : b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) {v : UInt64}
    (h : m.load (bufW nbuf b j) 4 = some v) : v = loadWord (m.bufs b) j := by
  by_cases hs : 4 * j + 4 ≤ (m.bufs b).size
  · rw [load_bufW hb hs] at h; exact (Option.some.inj h).symm
  · have hd := decode_bufW hb.1 hb.2.1 hb.2.2
    simp only [Mem.load, hd, Mem.region, bind, Option.bind] at h
    rw [if_pos (by simp; omega)] at h
    cases h

theorem bufW_mod (nbuf b j : Nat) : bufW nbuf b j = bufW nbuf b (j % 2 ^ 32) := by
  simp only [bufW]
  congr 2
  apply UInt32.toNat_inj.mp
  simp [UInt32.toNat_ofNat]

theorem loadWord_past {bs : ByteArray} {j : Nat} (h : bs.size ≤ 4 * j) : loadWord bs j = 0 := by
  have g : ∀ k, bs.size ≤ k → bs.get! k = 0 := fun k hk => by
    have hn : ¬ k < bs.data.size := by simp only [ByteArray.size] at hk; omega
    simp only [ByteArray.get!]
    rw [getElem!_neg bs.data k hn]; rfl
  simp only [loadWord, List.range, List.range.loop, List.foldr, g _ (by omega : bs.size ≤ 4 * j + 0),
    g _ (by omega : bs.size ≤ 4 * j + 1), g _ (by omega : bs.size ≤ 4 * j + 2), g _ (by omega : bs.size ≤ 4 * j + 3)]
  rfl

/-- **The warp machine's starting memory is the buffers' word view**, at a
    buffer that holds whole words. -/
theorem ws0_wordView {nbuf b : Nat} (hb : b < nbuf ∧ b < 268435449) (bufs : Nat → ByteArray)
    (h4 : (bufs b).size % 4 = 0) (a : Nat) :
    (ws0 nbuf (kmem bufs)).mem b a = wordView bufs b a := by
  have hj : a % 2 ^ 32 < 2 ^ 32 := Nat.mod_lt _ (by decide)
  simp only [ws0, wordView]
  rw [bufW_mod]
  by_cases hs : 4 * (a % 2 ^ 32) + 4 ≤ (bufs b).size
  · rw [load_bufW ⟨hb.1, hb.2, hj⟩ (by simpa [kmem] using hs)]; rfl
  · have hd := decode_bufW hb.1 hb.2 hj
    have hn : (kmem bufs).load (bufW nbuf b (a % 2 ^ 32)) 4 = none := by
      simp only [Mem.load, hd, Mem.region, bind, Option.bind]
      rw [if_pos (by simp [kmem]; omega)]
    rw [hn, loadWord_past (by omega)]
    rfl

end AlgorithmLib.ML
