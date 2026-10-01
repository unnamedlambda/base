
namespace AlgorithmLib.LZ4Ptx

theorem toNat_ofNat_lt (a : Nat) (h : a < 2 ^ 64) : (UInt64.ofNat a).toNat = a := by
  show a % 2 ^ 64 = a
  exact Nat.mod_eq_of_lt h

theorem u64_add_ofNat (a b : Nat) (h : a + b < 2 ^ 64) :
    (UInt64.ofNat a + UInt64.ofNat b).toNat = a + b := by
  rw [UInt64.toNat_add, toNat_ofNat_lt a (by omega), toNat_ofNat_lt b (by omega),
      Nat.mod_eq_of_lt h]

theorem u64_sub_ofNat (a b : Nat) (hb : b ≤ a) (ha : a < 2 ^ 64) :
    (UInt64.ofNat a - UInt64.ofNat b).toNat = a - b := by
  rw [UInt64.toNat_sub, toNat_ofNat_lt a ha, toNat_ofNat_lt b (by omega),
      show 2 ^ 64 - b + a = 2 ^ 64 + (a - b) from by omega, Nat.add_mod_left,
      Nat.mod_eq_of_lt (by omega)]

theorem u64_sub_ofNat' (a b : Nat) (hb : b ≤ a) (ha : a < 2 ^ 64) :
    UInt64.ofNat a - UInt64.ofNat b = UInt64.ofNat (a - b) := by
  apply UInt64.toNat_inj.mp
  rw [u64_sub_ofNat a b hb ha, toNat_ofNat_lt _ (by omega)]

end AlgorithmLib.LZ4Ptx
