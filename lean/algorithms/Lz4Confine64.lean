import Lz4Extend
import Lz4Shape64
import Lz4Stores64

set_option maxRecDepth 8192

/-!
  # `RegConfined` at the 64 KiB geometry

  The body of the confinement proof is generic over `(p, S)` — see `Lz4Extend`'s
  `Generic` section.  This file instantiates it at `K16`, stride
  65536, using the `Shape`/`Geo`/`Loads` instances proven in `Lz4Shape64` and
  `Lz4Sites`.  Nothing here is a new assumption: every premise is the same
  machine-checked bundle, decided against the 64 KiB array.
-/

namespace Lz4Sites

open Algorithm
open AlgorithmLib.LZ4Simt
open AlgorithmLib.LZ4SimtBits

-- ── The window clamps, at 64 KiB ──────────────────────────────────────────





-- ── `RegConfined`'s load half ───────────────────────────────────────────────

theorem loads_confined64 (inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hplace : outPtr = inPtr + 209715232) (htop : inPtr + 209715232 < 2 ^ 64)
    (w : Fin (WP.mk 16).numBlk) (k : Nat) (r : String) (off : Nat)
    (hmem : ((siter K16 k (initSt w.val inPtr outPtr gm smemB)).pc, r, off) ∈ loadSites K16)
    (l : Lane) :
    ((siter K16 k (initSt w.val inPtr outPtr gm smemB)).regs r l).toNat + off < outPtr := by
  have hwlt : w.val < 3200 := Nat.lt_of_lt_of_le w.isLt (by decide)
  have hw32 : w.val * 32 + 32 < 2 ^ 64 := by omega
  -- the pc, register and offset are one of twelve concrete triples
  rw [Loads.loadSitesEq (p := K16)] at hmem
  -- the input base, as a number
  have hibn : ∀ j : Nat, (siter K16 j (initSt w.val inPtr outPtr gm smemB)).pc ∈ ibS →
      ((siter K16 j (initSt w.val inPtr outPtr gm smemB)).regs "inBase" l).toNat
        = inPtr + w.val * 65536 := by
    intro j hj
    rw [inBase_eq (S := 65536) w.val inPtr outPtr gm smemB hw32 j hj l, UInt64.toNat_add, UInt64.toNat_mul,
      AlgorithmLib.LZ4Ptx.toNat_ofNat_lt inPtr (by omega),
      AlgorithmLib.LZ4Ptx.toNat_ofNat_lt w.val (by omega),
      AlgorithmLib.LZ4Ptx.toNat_ofNat_lt 65536 (by omega),
      Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)]
  -- the uniform step: `regs r l = inBase + X` with `X` bounded
  have main : ∀ (q : Nat), (siter K16 k (initSt w.val inPtr outPtr gm smemB)).pc = q →
      q ∈ ibS → ∀ X : UInt64,
      (siter K16 k (initSt w.val inPtr outPtr gm smemB)).regs r l
        = (siter K16 k (initSt w.val inPtr outPtr gm smemB)).regs "inBase" l + X →
      X.toNat + off ≤ (65536 + 31) →
      ((siter K16 k (initSt w.val inPtr outPtr gm smemB)).regs r l).toNat + off < outPtr := by
    intro q hq hqm X hX hb
    have hb2 := hibn k (by rw [hq]; exact hqm)
    rw [hX, UInt64.toNat_add, hb2, Nat.mod_eq_of_lt (by omega), hplace]
    omega
  have mem19 : ∀ q : Nat, 19 ≤ q → q ≤ 271 → q ∈ ibS := by
    intro q h1 h2
    simp only [ibS, List.mem_map, List.mem_range]
    exact ⟨q - 19, by omega, by omega⟩
  obtain ⟨lat1, lat2, lat3, lat4, lat5, lat6⟩ := Loads.loadAt (p := K16) w.val inPtr outPtr gm smemB k l
  simp only [List.mem_cons, List.not_mem_nil, or_false, Prod.mk.injEq] at hmem
  rcases hmem with ⟨hp, hr, ho⟩ | ⟨hp, hr, ho⟩ | ⟨hp, hr, ho⟩ | ⟨hp, hr, ho⟩
    | ⟨hp, hr, ho⟩ | ⟨hp, hr, ho⟩ | ⟨hp, hr, ho⟩ | ⟨hp, hr, ho⟩
    | ⟨hp, hr, ho⟩ | ⟨hp, hr, ho⟩ | ⟨hp, hr, ho⟩ | ⟨hp, hr, ho⟩ <;> subst hr <;> subst ho
  -- `rpA`, offsets 0–3
  · refine main 47 hp (mem19 47 (by omega) (by omega)) _ (lat1 47 (by decide) hp) ?_
    exact Nat.le_trans (Nat.add_le_add_right
      (rp_clamped64 w.val inPtr outPtr gm smemB k (by rw [hp]; omega) (by rw [hp]; omega) l) _)
      (by decide)
  · refine main 48 hp (mem19 48 (by omega) (by omega)) _ (lat1 48 (by decide) hp) ?_
    exact Nat.le_trans (Nat.add_le_add_right
      (rp_clamped64 w.val inPtr outPtr gm smemB k (by rw [hp]; omega) (by rw [hp]; omega) l) _)
      (by decide)
  · refine main 49 hp (mem19 49 (by omega) (by omega)) _ (lat1 49 (by decide) hp) ?_
    exact Nat.le_trans (Nat.add_le_add_right
      (rp_clamped64 w.val inPtr outPtr gm smemB k (by rw [hp]; omega) (by rw [hp]; omega) l) _)
      (by decide)
  · refine main 50 hp (mem19 50 (by omega) (by omega)) _ (lat1 50 (by decide) hp) ?_
    exact Nat.le_trans (Nat.add_le_add_right
      (rp_clamped64 w.val inPtr outPtr gm smemB k (by rw [hp]; omega) (by rw [hp]; omega) l) _)
      (by decide)
  -- `rcA`, offsets 0–3
  · refine main 66 hp (mem19 66 (by omega) (by omega)) _ (lat2 66 (by decide) hp) ?_
    exact Nat.le_trans (Nat.add_le_add_right
      (rc_clamped64 w.val inPtr outPtr gm smemB k (by rw [hp]; omega) (by rw [hp]; omega) l) _)
      (by decide)
  · refine main 67 hp (mem19 67 (by omega) (by omega)) _ (lat2 67 (by decide) hp) ?_
    exact Nat.le_trans (Nat.add_le_add_right
      (rc_clamped64 w.val inPtr outPtr gm smemB k (by rw [hp]; omega) (by rw [hp]; omega) l) _)
      (by decide)
  · refine main 68 hp (mem19 68 (by omega) (by omega)) _ (lat2 68 (by decide) hp) ?_
    exact Nat.le_trans (Nat.add_le_add_right
      (rc_clamped64 w.val inPtr outPtr gm smemB k (by rw [hp]; omega) (by rw [hp]; omega) l) _)
      (by decide)
  · refine main 69 hp (mem19 69 (by omega) (by omega)) _ (lat2 69 (by decide) hp) ?_
    exact Nat.le_trans (Nat.add_le_add_right
      (rc_clamped64 w.val inPtr outPtr gm smemB k (by rw [hp]; omega) (by rw [hp]; omega) l) _)
      (by decide)
  -- the extend's two byte reads
  · refine main 110 hp (mem19 110 (by omega) (by omega)) _ (lat3 hp) ?_
    exact Nat.le_trans (Nat.add_le_add_right (peD_le (S := 65536) w.val inPtr outPtr gm smemB k hp l) _)
      (by decide)
  · refine main 111 hp (mem19 111 (by omega) (by omega)) _ (lat4 hp) ?_
    exact Nat.le_trans (Nat.add_le_add_right (caD_le (S := 65536) w.val inPtr outPtr gm smemB k hp l) _)
      (by decide)
  -- the two cooperative copies
  · obtain ⟨o1, ho1, he1⟩ := cpSo_off (S := 65536) w.val inPtr outPtr gm smemB k hp l
    refine main 164 hp (mem19 164 (by omega) (by omega)) (UInt64.ofNat o1) ?_ ?_
    · exact he1
    · rw [AlgorithmLib.LZ4Ptx.toNat_ofNat_lt o1 (by omega)]; omega
  · obtain ⟨o2, ho2, he2⟩ := cpSo2_off (S := 65536) w.val inPtr outPtr gm smemB k hp l
    refine main 250 hp (mem19 250 (by omega) (by omega)) (UInt64.ofNat o2) ?_ ?_
    · exact he2
    · rw [AlgorithmLib.LZ4Ptx.toNat_ofNat_lt o2 (by omega)]; omega

-- ── `RegConfined`, assembled ─────────────────────────────────────────────────

theorem regConfined_shipped64 (inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hderive : outPtr = inPtr + ((WP.mk 16).numBlk * (WP.mk 16).inStride + AlgorithmLib.LZ4Simt.copySlack))
    (hib40 : ∀ w, w < (WP.mk 16).numBlk → inPtr + w * (WP.mk 16).inStride < 2 ^ 40)
    (htop32 : ∀ w, w < (WP.mk 16).numBlk →
      outPtr + w * (WP.mk 16).outStride + 9 * (WP.mk 16).inStride < 2 ^ 32)
    (hbuf : ∀ w, w < (WP.mk 16).numBlk →
      outPtr + w * (WP.mk 16).outStride + 9 * (WP.mk 16).inStride ≤ gm.size)
    (hdisj : ∀ w, w < (WP.mk 16).numBlk →
      inPtr + w * (WP.mk 16).inStride + (WP.mk 16).inStride
        ≤ outPtr + w * (WP.mk 16).outStride) :
    RegConfined 16 inPtr outPtr gm smemB where
  loads := by
    intro w k r off hmem l
    have hplace : outPtr = inPtr + 209715232 := by
      rw [hderive, show (WP.mk 16).numBlk * (WP.mk 16).inStride + AlgorithmLib.LZ4Simt.copySlack
        = 209715232 from by decide]
    have h32 := htop32 0 (by decide)
    exact loads_confined64 inPtr outPtr gm smemB hplace (by omega) w k r off hmem l
  stores := by
    intro w k r hsite l hact
    by_cases hc : r = "cpDo"
    · subst hc
      have hsites : ∀ x ∈ storeSites K16, x.2 = "cpDo" → x.1 = 165 ∨ x.1 = 251 :=
        Loads.storeCpDo (p := K16)
      have hcp : ∀ q, (siter K16 k (initSt w.val inPtr outPtr gm smemB)).pc = q →
          K16[q]? = some (SInstr.stgp "cpP" "cpDo" "cpB") →
          (siter K16 k (initSt w.val inPtr outPtr gm smemB)).regs "cpP" l = 1 := by
        intro q hq hK
        rw [ActiveAt, show K16[(siter K16 k (initSt w.val inPtr outPtr gm smemB)).pc]?
          = some (SInstr.stgp "cpP" "cpDo" "cpB") from by rw [hq]; exact hK] at hact
        exact hact
      rcases hsites _ hsite rfl with e | e
      · exact stores_cpDo16564 inPtr outPtr gm smemB hderive hib40 htop32 hbuf hdisj w k l e
          (hcp 165 e (by decide))
      · exact stores_cpDo25164 inPtr outPtr gm smemB hderive hib40 htop32 hbuf hdisj w k l e
          (hcp 251 e (by decide))
    · exact stores_except_copy64 inPtr outPtr gm smemB hderive hib40 htop32 hbuf hdisj w k l r
        hsite hc

theorem kernelConfined_shipped64 (inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hderive : outPtr = inPtr + ((WP.mk 16).numBlk * (WP.mk 16).inStride + AlgorithmLib.LZ4Simt.copySlack))
    (hib40 : ∀ w, w < (WP.mk 16).numBlk → inPtr + w * (WP.mk 16).inStride < 2 ^ 40)
    (htop32 : ∀ w, w < (WP.mk 16).numBlk →
      outPtr + w * (WP.mk 16).outStride + 9 * (WP.mk 16).inStride < 2 ^ 32)
    (hbuf : ∀ w, w < (WP.mk 16).numBlk →
      outPtr + w * (WP.mk 16).outStride + 9 * (WP.mk 16).inStride ≤ gm.size)
    (hdisj : ∀ w, w < (WP.mk 16).numBlk →
      inPtr + w * (WP.mk 16).inStride + (WP.mk 16).inStride
        ≤ outPtr + w * (WP.mk 16).outStride) :
    Lz4Interleave.KernelConfined 16 inPtr outPtr gm smemB :=
  kernelConfined_of_regConfined3264 inPtr outPtr gm smemB
    (regConfined_shipped64 inPtr outPtr gm smemB hderive hib40 htop32 hbuf hdisj)

end Lz4Sites
