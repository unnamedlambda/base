import Std.Tactic.BVDecide
import AlgorithmLib.LZ4Ptx

namespace AlgorithmLib

/-- Read a little-endian `u32` out of a byte array — exactly what the host does to
    recover a block's compressed length. -/
def readU32LE (g : Array UInt8) (a : Nat) : Nat :=
  (g.getD a 0).toNat + 256 * (g.getD (a + 1) 0).toNat
    + 65536 * (g.getD (a + 2) 0).toNat + 16777216 * (g.getD (a + 3) 0).toNat

/-- Masking with `255` is a no-op under an 8-bit truncation. -/
theorem and255_toUInt8 (x : UInt64) : (x &&& 255).toUInt8 = x.toUInt8 := by bv_decide

theorem toUInt8_toNat (x : UInt64) : x.toUInt8.toNat = x.toNat % 256 := by
  simp [UInt64.toUInt8, UInt8.toNat, UInt8.ofNat, Nat.toUInt8]

theorem shiftRight_toNat (x : UInt64) (k : Nat) (hk : k < 64) :
    (x >>> (UInt64.ofNat k)).toNat = x.toNat / 2 ^ k := by
  rw [UInt64.toNat_shiftRight, LZ4Ptx.toNat_ofNat_lt k (by omega), Nat.mod_eq_of_lt hk,
    Nat.shiftRight_eq_div_pow]

/-- Little-endian byte reassembly of a `u32`. -/
theorem le_reassemble (L : Nat) (h : L < 4294967296) :
    L % 256 + 256 * (L / 256 % 256) + 65536 * (L / 65536 % 256)
      + 16777216 * (L / 16777216 % 256) = L := by omega

end AlgorithmLib
