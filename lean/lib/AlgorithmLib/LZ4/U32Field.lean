module
public import Std.Tactic.BVDecide
public import AlgorithmLib.LZ4.Ptx
meta import AlgorithmLib.LZ4.Ptx
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

namespace AlgorithmLib

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
