import Lz4CompAlgorithm

/-!
  # The compressor's claims, at the two geometries that ship

  `launch_correct` is stated for an arbitrary block-size log `b` and takes
  `ShippedCorrect b`, `LayoutOK` and `LaunchAgreesPerWarp` as hypotheses.  Until
  something feeds `shipped32_correct` into it, nothing connects the top-level
  theorem to the artifacts the build emits, and nothing shows `LayoutOK` is
  satisfiable at all.

  Both matter for the same reason: `theorem foo (h : P) : Q` is trivially true
  when `P` is unsatisfiable, and no axiom count can see it.  This file closes
  both — `layoutOK_witness` exhibits concrete buffer addresses and a concrete
  memory satisfying the contract, and `shipped32_launch_correct` /
  `shipped64_launch_correct` are the whole-launch claim at `blkLog` 15 and 16,
  carrying only the two obligations named in `Lz4Assumptions`.
-/

namespace Lz4NonVacuity

open Algorithm
open AlgorithmLib.LZ4WarpDSL

/-- A concrete zero-filled memory.  `Array.size` of this reduces by
    `List.length_replicate`, so nothing evaluates 434 MB of data. -/
def zeroMem (n : Nat) : Array UInt8 := (List.replicate n (0 : UInt8)).toArray

@[simp] theorem zeroMem_size (n : Nat) : (zeroMem n).size = n := by
  simp [zeroMem]

/-- The geometry at `blkLog = 15`, as literals, so `omega` can use it. -/
theorem geom32 :
    (WP.mk 15).numBlk = 6400 ∧ (WP.mk 15).inStride = 32768 ∧
    (WP.mk 15).outStride = 35080 ∧ (WP.mk 15).lenOff = 35072 := by
  refine ⟨by decide, by decide, by decide, by decide⟩

theorem geom64 :
    (WP.mk 16).numBlk = 3200 ∧ (WP.mk 16).inStride = 65536 ∧
    (WP.mk 16).outStride = 69896 ∧ (WP.mk 16).lenOff = 69888 := by
  refine ⟨by decide, by decide, by decide, by decide⟩

/-- **`LayoutOK` is satisfiable at the shipped 32 KiB geometry.**  The input
    buffer sits at 0, the output buffer immediately after the whole input, and
    the memory is large enough for the output stride plus the body simulation's
    `9 * inStride` of headroom.  Without this, everything below is vacuously
    true. -/
theorem layoutOK_witness32 :
    LayoutOK 15 0 209715232 (zeroMem 434522144) := by
  obtain ⟨hnb, hin, hout, hlen⟩ := geom32
  refine ⟨by rw [hnb, hin]; rfl, ?_⟩
  · intro w hw
    rw [hnb] at hw
    rw [hin, hout, hlen, zeroMem_size]
    refine ⟨by omega, by omega, by omega, by omega, by omega⟩

theorem layoutOK_witness64 :
    LayoutOK 16 0 209715232 (zeroMem 434522144) := by
  obtain ⟨hnb, hin, hout, hlen⟩ := geom64
  refine ⟨by rw [hnb, hin]; rfl, ?_⟩
  · intro w hw
    rw [hnb] at hw
    rw [hin, hout, hlen, zeroMem_size]
    refine ⟨by omega, by omega, by omega, by omega, by omega⟩

/-- Existential form, for the record: the contract is inhabited. -/
theorem layoutOK_witness :
    ∃ (inPtr outPtr : Nat) (gm : Array UInt8), LayoutOK 15 inPtr outPtr gm :=
  ⟨0, 209715232, zeroMem 434522144, layoutOK_witness32⟩

/-- **The whole-launch claim for the artifact `lz4_comp_warpdsl` ships.**
    `ShippedCorrect 15` is discharged by `shipped32_correct`; what remains are
    the two obligations the ledger names. -/
theorem shipped32_launch_correct
    (inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) (gfinal : Array UInt8)
    (hlayout : LayoutOK 15 inPtr outPtr gm)
    (hSC : LaunchAgreesPerWarp 15 inPtr outPtr gm smemB gfinal) :
    ∀ w, w < (WP.mk 15).numBlk →
      ∃ k, 0 < k ∧ k ≤ (WP.mk 15).lenOff ∧
        AlgorithmLib.readU32LE gfinal
          (outPtr + w * (WP.mk 15).outStride + (WP.mk 15).lenOff) = k ∧
        AlgorithmLib.LZ4Imp.decompress
          ((List.range k).map (fun i => gfinal.getD
            (outPtr + w * (WP.mk 15).outStride + i) 0))
          (WP.mk 15).inStride
          = some (gmemInpAt gm (inPtr + w * (WP.mk 15).inStride) (WP.mk 15).inStride) :=
  launch_correct 15 inPtr outPtr gm smemB gfinal shipped32_correct hlayout hSC

/-- The same for `lz4_comp_warpdsl64`. -/
theorem shipped64_launch_correct
    (inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8) (gfinal : Array UInt8)
    (hlayout : LayoutOK 16 inPtr outPtr gm)
    (hSC : LaunchAgreesPerWarp 16 inPtr outPtr gm smemB gfinal) :
    ∀ w, w < (WP.mk 16).numBlk →
      ∃ k, 0 < k ∧ k ≤ (WP.mk 16).lenOff ∧
        AlgorithmLib.readU32LE gfinal
          (outPtr + w * (WP.mk 16).outStride + (WP.mk 16).lenOff) = k ∧
        AlgorithmLib.LZ4Imp.decompress
          ((List.range k).map (fun i => gfinal.getD
            (outPtr + w * (WP.mk 16).outStride + i) 0))
          (WP.mk 16).inStride
          = some (gmemInpAt gm (inPtr + w * (WP.mk 16).inStride) (WP.mk 16).inStride) :=
  launch_correct 16 inPtr outPtr gm smemB gfinal shipped64_correct hlayout hSC

/-- info: 'Lz4NonVacuity.shipped32_launch_correct' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms shipped32_launch_correct

/-- info: 'Lz4NonVacuity.layoutOK_witness32' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms layoutOK_witness32

-- ── What the allocator owes, and what it does not ─────────────────────────────

/-- **`LayoutOK` reduces to one comparison of two `cudaMalloc` results.**

    The concern about the flat-memory model has always been that the runtime
    makes two independent `cudaCreateBuffer` calls and the model orders them.
    This says how much of the contract that concern actually covers: given the
    sizes the host allocates (`totIn` for the input, `outAlloc` for the output),
    that the output allocation fits in memory, and that the address space is
    32-bit-addressable, every per-warp clause of `LayoutOK` follows — for all
    6400 warps at once — from the single hypothesis `hord`.

    `hord` is an EQUATION, which the host establishes by allocating once;
    `Lz4Host.host_single_allocation` reads that single `cudaCreateBuffer` back out
    of the emitted CLIF, and the kernel's own prologue supplies the other half by
    computing `outBase` from `in_ptr`.  An inequality between two independent
    allocations would leave an informal step in the chain. -/
theorem layoutOK_of_alloc (b inPtr outPtr : Nat) (gm : Array UInt8)
    (hord : outPtr = inPtr + ((WP.mk b).totIn + AlgorithmLib.LZ4Simt.copySlack))
    (h40 : inPtr + (WP.mk b).totIn < 2 ^ 40)
    (h32 : outPtr + (WP.mk b).outAlloc < 2 ^ 32)
    (hsize : outPtr + (WP.mk b).outAlloc ≤ gm.size) :
    LayoutOK b inPtr outPtr gm := by
  have htotIn : (WP.mk b).totIn = (WP.mk b).numBlk * (WP.mk b).inStride := rfl
  have hAlloc : (WP.mk b).outAlloc
      = (WP.mk b).numBlk * (WP.mk b).outStride + 9 * (WP.mk b).inStride := rfl
  refine ⟨by omega, ?_⟩
  intro w hw
  -- warp `w`'s slices sit inside the two allocations
  have hin : w * (WP.mk b).inStride + (WP.mk b).inStride ≤ (WP.mk b).totIn := by
    rw [htotIn]
    have : (w + 1) * (WP.mk b).inStride ≤ (WP.mk b).numBlk * (WP.mk b).inStride :=
      Nat.mul_le_mul_right _ hw
    have he : (w + 1) * (WP.mk b).inStride
        = w * (WP.mk b).inStride + (WP.mk b).inStride := Nat.succ_mul w _
    omega
  have hout : w * (WP.mk b).outStride + (WP.mk b).outStride
      ≤ (WP.mk b).numBlk * (WP.mk b).outStride := by
    have : (w + 1) * (WP.mk b).outStride ≤ (WP.mk b).numBlk * (WP.mk b).outStride :=
      Nat.mul_le_mul_right _ hw
    have he : (w + 1) * (WP.mk b).outStride
        = w * (WP.mk b).outStride + (WP.mk b).outStride := Nat.succ_mul w _
    omega
  have hstride : (WP.mk b).outStride = (WP.mk b).lenOff + 8 := rfl
  have h64 : (2 : Nat) ^ 32 < 2 ^ 64 := by decide
  exact ⟨by omega, by omega, by omega, by omega, by omega⟩

/-- The witness satisfies the allocator-level form too, so the reduction above is
    not vacuous either. -/
theorem layoutOK_of_alloc_witness :
    LayoutOK 15 0 209715232 (zeroMem 434522144) :=
  layoutOK_of_alloc 15 0 209715232 _ (by decide) (by decide) (by decide)
    (by rw [zeroMem_size]; decide)

end Lz4NonVacuity
