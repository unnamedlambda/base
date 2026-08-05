import Lz4Ckpt64

set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

namespace Lz4Sites

open Algorithm
open AlgorithmLib.LZ4Simt
open AlgorithmLib.LZ4SimtBits

theorem cursorAtSites_shipped64 (inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8)
    (hderive : outPtr = inPtr + ((WP.mk 16).numBlk * (WP.mk 16).inStride + AlgorithmLib.LZ4Simt.copySlack))
    (hib40 : ∀ w, w < (WP.mk 16).numBlk → inPtr + w * (WP.mk 16).inStride < 2 ^ 40)
    (htop : ∀ w, w < (WP.mk 16).numBlk →
      outPtr + w * (WP.mk 16).outStride + 9 * (WP.mk 16).inStride < 2 ^ 32)
    (hbuf : ∀ w, w < (WP.mk 16).numBlk →
      outPtr + w * (WP.mk 16).outStride + 9 * (WP.mk 16).inStride ≤ gm.size)
    (hdisj : ∀ w, w < (WP.mk 16).numBlk →
      inPtr + w * (WP.mk 16).inStride + (WP.mk 16).inStride
        ≤ outPtr + w * (WP.mk 16).outStride) :
    CursorAtSites 16 inPtr outPtr gm smemB := by
  refine ⟨fun w k l hsite => ?_⟩
  obtain ⟨n, hpc39, hpc40, hck, hpc209, wE, hcE, hQE, hTE, hlaE⟩ :=
    shipped_loop_ckpt64 w.val inPtr outPtr gm smemB w.isLt
      (by have := w.isLt; have := numBlk3264; omega)
      (hib40 _ w.isLt) (htop _ w.isLt) (hbuf _ w.isLt) hderive (hdisj _ w.isLt)
  have hsites : (siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc = 130
      ∨ (siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc = 140
      ∨ (siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc = 147
      ∨ (siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc = 173
      ∨ (siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc = 178
      ∨ (siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc = 188
      ∨ (siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc = 195
      ∨ (siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc = 216
      ∨ (siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc = 226
      ∨ (siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc = 233 := by
    simpa only [sbAddrSites, List.mem_cons, List.not_mem_nil, or_false] using hsite
  -- the seven token stores
  have HTOK : ((siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc = 130
      ∨ (siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc = 140
      ∨ (siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc = 147
      ∨ (siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc = 173
      ∨ (siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc = 178
      ∨ (siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc = 188
      ∨ (siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc = 195) →
      ((siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB)).regs "op" l).toNat
        ≤ (WP.mk 16).lenOff := by
    intro hq
    have hmemB : (siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc ∈ bodyRegion := by
      rcases hq with e | e | e | e | e | e | e <;> rw [e] <;> decide
    obtain ⟨hge, -, -⟩ :=
      body_entry_at64 (initSt w.val inPtr outPtr gm smemB) preSteps hpc39 (k + 1) hmemB
    have hsplit : siter K16 (k + 1 - (preSteps + 1))
        (siter K16 (preSteps + 1) (initSt w.val inPtr outPtr gm smemB))
        = siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB) := by
      rw [← siter_add K16 (preSteps + 1) (k + 1 - (preSteps + 1)),
        show preSteps + 1 + (k + 1 - (preSteps + 1)) = k + 1 from by omega]
    have hfin := tok_sites_bounded64 (WP.mk 16).lenOff (by decide)
      (siter K16 (preSteps + 1) (initSt w.val inPtr outPtr gm smemB)) (by rw [hpc40]; decide)
      l (k + 1 - (preSteps + 1))
      (fun j _ hj => by
        rw [← siter_add K16 (preSteps + 1) j] at hj ⊢
        exact hck j hj)
      (by rw [hsplit]; exact hq)
    rw [hsplit] at hfin
    exact hfin
  -- the three stores of the final literal run
  have HTAIL : ((siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc = 216
      ∨ (siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc = 226
      ∨ (siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc = 233) →
      ((siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB)).regs "op" l).toNat
        ≤ (WP.mk 16).lenOff := by
    intro hq
    have hkn : preSteps + 1 + n ≤ k + 1 := by
      rcases Nat.lt_or_ge (k + 1) (preSteps + 1 + n) with hlt | hge
      · exfalso
        have h210 : 210 ≤ (siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB)).pc := by
          rcases hq with e | e | e <;> rw [e] <;> omega
        have hst := stays_from_21064 (initSt w.val inPtr outPtr gm smemB) (k + 1) h210
          (preSteps + 1 + n) (by omega)
        rw [hpc209] at hst; omega
      · exact hge
    have hsplit : siter K16 (k + 1 - (preSteps + 1 + n))
        (siter K16 (preSteps + 1 + n) (initSt w.val inPtr outPtr gm smemB))
        = siter K16 (k + 1) (initSt w.val inPtr outPtr gm smemB) := by
      rw [← siter_add K16 (preSteps + 1 + n) (k + 1 - (preSteps + 1 + n)),
        show preSteps + 1 + n + (k + 1 - (preSteps + 1 + n)) = k + 1 from by omega]
    have hlaC : ∀ j : Lane,
        (siter K16 (preSteps + 1 + n) (initSt w.val inPtr outPtr gm smemB)).regs "litAnchor" j
          = wE.regs "litAnchor" := fun j => hcE.reg (by decide) j
    have hopC : ∀ j : Lane,
        (siter K16 (preSteps + 1 + n) (initSt w.val inPtr outPtr gm smemB)).regs "op" j
          = wE.regs "op" := fun j => hcE.reg (by decide) j
    simp only [AlgorithmLib.LZ4WarpDSL.TightQ, AlgorithmLib.LZ4WarpDSL.tightRem] at hTE
    have hfin := tail_sites_bounded64 (WP.mk 16).lenOff (by decide)
      (siter K16 (preSteps + 1 + n) (initSt w.val inPtr outPtr gm smemB)) hpc209
      (by rw [hlaC 0]; exact Nat.le_trans hlaE (by decide))
      (fun j => by rw [hlaC j, hlaC 0])
      (fun j => by rw [hopC j, hopC 0])
      (by rw [hlaC 0, hopC 0]
          rw [show (65536 : Nat) = (WP.mk 16).inStride from rfl]
          omega)
      l (k + 1 - (preSteps + 1 + n))
      (by rw [hsplit]; exact hq)
    rw [hsplit] at hfin
    exact hfin
  rcases hsites with h | h | h | h | h | h | h | h | h | h
  · exact HTOK (Or.inl h)
  · exact HTOK (Or.inr (Or.inl h))
  · exact HTOK (Or.inr (Or.inr (Or.inl h)))
  · exact HTOK (Or.inr (Or.inr (Or.inr (Or.inl h))))
  · exact HTOK (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl h)))))
  · exact HTOK (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl h))))))
  · exact HTOK (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr h))))))
  · exact HTAIL (Or.inl h)
  · exact HTAIL (Or.inr (Or.inl h))
  · exact HTAIL (Or.inr (Or.inr h))








end Lz4Sites
