module
public import AlgorithmLib.LZ4.CompTop
meta import AlgorithmLib.LZ4.CompTop
public import AlgorithmLib.LZ4.SimtSerialize
meta import AlgorithmLib.LZ4.SimtSerialize
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The warp compressor's kernel, its memory layout, and its launch correctness

Everything here is about the SIMT kernel and the memory it touches, and none of
it is about the function the artifact ships — so none of it needs the generator
toolkit. The confinement and geometry proofs rest on this rather than on the
generator, which is what keeps them clear of a change to the CLIF surface.
-/

open AlgorithmLib
open AlgorithmLib.PTX

namespace Lz4Ship

def corpusBytes : Nat := 209715200   -- 3200 * 65536 (fixed corpus prefix)
def rLaunches  : Nat := 20

def rPTX_OFF   : Nat := 0x100


-- ── Warp-cooperative compressor: 1 warp/block, 4096-entry shared-mem hash table.
-- Parameterized by block-size log2 (pos+1 must fit u16: blkLog ≤ 16). --
structure WP where
  blkLog : Nat
def wHashLog     : Nat := 12                  -- 4096-entry shared table (8 KiB/warp)

def WP.inStride  (w : WP) : Nat := 2 ^ w.blkLog
def WP.numBlk    (w : WP) : Nat := corpusBytes / w.inStride
def WP.lenOff    (w : WP) : Nat := w.inStride + w.inStride / 16 + 256
def WP.outStride (w : WP) : Nat := w.lenOff + 8  -- compressed | u32 length (no global table)
def wTableBytes  : Nat := (2 ^ wHashLog) * 2
def wSmem        : Nat := wTableBytes
def wBlockDim    : Nat := AlgorithmLib.LZ4Simt.modelBlockDim
def WP.gridX     (w : WP) : Nat := w.numBlk
def WP.totIn     (w : WP) : Nat := w.numBlk * w.inStride
def WP.totOut    (w : WP) : Nat := w.numBlk * w.outStride
-- `9*inStride` of headroom past the last warp's stride: the body simulation's
-- budget is 9 bytes per remaining input byte, which `hbuf` needs for every warp.
def WP.outAlloc  (w : WP) : Nat := w.totOut + 9 * w.inStride
/-- Where warp 0's output begins.  `copySlack` is the KERNEL's constant
    (`LZ4SimtEmit.copySlack`, baked into the prologue's `outP` immediate); using it
    here is what keeps the host allocation and the kernel's own base in step. -/
def WP.outOff    (w : WP) : Nat := w.totIn + AlgorithmLib.LZ4Simt.copySlack

/-- The kernel this artifact ships.  `warpKernelDSLStr` serializes THIS, and
    `ShippedCorrect` is stated about THIS — one definition, so the proven object and
    the executed object cannot be different programs. -/
def WP.kernel (w : WP) : Array AlgorithmLib.LZ4Simt.SInstr :=
  AlgorithmLib.LZ4WarpDSL.warpKernelDSL w.numBlk w.inStride w.outStride w.lenOff wHashLog

def warpKernelDSLStr (w : WP) : String :=
  AlgorithmLib.LZ4Simt.serializeKernel w.kernel wSmem

-- ── Payload layout, derived from the serialized kernel ────────────────────────
-- Offsets are relative to the PTX's actual length, so there is no fixed slot to
-- overflow (Nat subtraction would truncate silently rather than fail).
/-- The bytes at `rPTX_OFF`.  `ptxLen` is this list's length and `warpPayloadDSL`
    emits this same list, so offsets and image cannot disagree. -/
def WP.ptxBytes  (w : WP) : List UInt8 := (warpKernelDSLStr w).toUTF8.toList ++ [0]
def WP.ptxLen    (w : WP) : Nat := w.ptxBytes.length
/-- The binding table, 16-byte aligned. -/
def WP.bindOff   (w : WP) : Nat := (rPTX_OFF + w.ptxLen + 15) / 16 * 16
def WP.memSize   (w : WP) : Nat := w.bindOff + 0x80

-- ── The shipped claim ─────────────────────────────────────────────────────────
-- `ShippedCorrect b` is the correctness statement for the artifact
-- `warpArtifactDSL _ b` emits, in that artifact's own geometry.  Every numeric
-- side condition is discharged inside the proof; the remaining hypotheses are the
-- artifact's contract.  `inpAll.length = totIn` is NOT enforced at run time.
section ShippedClaim
open AlgorithmLib.LZ4WarpDSL

/-- Correctness of the artifact `warpArtifactDSL _ b` emits, for warp `w`.

    The output base is not a free parameter: the kernel computes it as
    `in_ptr + totIn` (`prologueInstrs` index 1), so the only placements the
    theorem admits are the ones the program actually produces.  That is the
    equation `hderive`, which the host establishes by making one allocation. -/
def ShippedCorrect (b : Nat) : Prop :=
  ∀ (w inPtr outPtr : Nat) (gm : Array UInt8) (smemB : List UInt8),
    w < (WP.mk b).numBlk →
    -- the output buffer is the tail of the input allocation — by construction
    outPtr = inPtr + ((WP.mk b).numBlk * (WP.mk b).inStride
      + AlgorithmLib.LZ4Simt.copySlack) →
    -- remaining layout contract: sizes and address-space bounds
    inPtr + w * (WP.mk b).inStride < 2 ^ 40 →
    outPtr + w * (WP.mk b).outStride + 9 * (WP.mk b).inStride < 2 ^ 32 →
    outPtr + w * (WP.mk b).outStride + 9 * (WP.mk b).inStride ≤ gm.size →
    outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff + 3 < 2 ^ 64 →
    outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff + 4 ≤ gm.size →
    ∃ (n : Nat) (ss' : AlgorithmLib.LZ4Simt.SState) (k : Nat),
      AlgorithmLib.LZ4Simt.SReaches (WP.mk b).kernel n
        (AlgorithmLib.LZ4Simt.initSt w inPtr outPtr gm smemB) ss' ∧
      ss'.pc = 272 ∧ 0 < k ∧ k ≤ (WP.mk b).lenOff ∧
      AlgorithmLib.readU32LE ss'.gmem
        (outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff) = k ∧
      AlgorithmLib.LZ4Imp.decompress
        ((List.range k).map (fun i => ss'.gmem.getD
          (outPtr + w * (WP.mk b).outStride + i) 0))
        (WP.mk b).inStride
        = some (gmemInpAt gm (inPtr + w * (WP.mk b).inStride) (WP.mk b).inStride) ∧
      -- warp `w` writes ONLY its own stride, so the per-warp results compose
      (∀ j, j < outPtr + w * (WP.mk b).outStride ∨
            outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff + 4 ≤ j →
        ss'.gmem.getD j 0 = gm.getD j 0)

theorem shipped32_correct : ShippedCorrect 15 := by
  intro w inPtr outPtr gm smemB hw hderive hib40 htop hbuf hlOtop hlOfit
  have hw64 : w * 32 + 32 < 2 ^ 64 := by
    have h1 : w * 32 ≤ (WP.mk 15).numBlk * 32 := Nat.mul_le_mul_right 32 (Nat.le_of_lt hw)
    have h2 : (WP.mk 15).numBlk * 32 + 32 < 2 ^ 64 := by decide
    omega
  -- the input/output separation the body simulation wants is now a CONSEQUENCE
  have hdisj : inPtr + w * (WP.mk 15).inStride + (WP.mk 15).inStride
      ≤ outPtr + w * (WP.mk 15).outStride := by
    have h1 : (w + 1) * (WP.mk 15).inStride ≤ (WP.mk 15).numBlk * (WP.mk 15).inStride :=
      Nat.mul_le_mul_right _ hw
    have h2 : (w + 1) * (WP.mk 15).inStride
        = w * (WP.mk 15).inStride + (WP.mk 15).inStride := Nat.succ_mul w _
    have h3 : w * (WP.mk 15).inStride ≤ w * (WP.mk 15).outStride :=
      Nat.mul_le_mul_left _ (by decide)
    omega
  exact warpKernelDSL_tail_roundtrips
    (WP.mk 15).numBlk (WP.mk 15).inStride (WP.mk 15).outStride (WP.mk 15).lenOff wHashLog
    w inPtr outPtr gm smemB
    (by decide) (by decide) hw (by decide) (by decide) (by decide)
    hw64 hib40 htop hbuf hderive hdisj (Nat.le_refl _) hlOtop hlOfit

theorem shipped64_correct : ShippedCorrect 16 := by
  intro w inPtr outPtr gm smemB hw hderive hib40 htop hbuf hlOtop hlOfit
  have hw64 : w * 32 + 32 < 2 ^ 64 := by
    have h1 : w * 32 ≤ (WP.mk 16).numBlk * 32 := Nat.mul_le_mul_right 32 (Nat.le_of_lt hw)
    have h2 : (WP.mk 16).numBlk * 32 + 32 < 2 ^ 64 := by decide
    omega
  -- the input/output separation the body simulation wants is now a CONSEQUENCE
  have hdisj : inPtr + w * (WP.mk 16).inStride + (WP.mk 16).inStride
      ≤ outPtr + w * (WP.mk 16).outStride := by
    have h1 : (w + 1) * (WP.mk 16).inStride ≤ (WP.mk 16).numBlk * (WP.mk 16).inStride :=
      Nat.mul_le_mul_right _ hw
    have h2 : (w + 1) * (WP.mk 16).inStride
        = w * (WP.mk 16).inStride + (WP.mk 16).inStride := Nat.succ_mul w _
    have h3 : w * (WP.mk 16).inStride ≤ w * (WP.mk 16).outStride :=
      Nat.mul_le_mul_left _ (by decide)
    omega
  exact warpKernelDSL_tail_roundtrips
    (WP.mk 16).numBlk (WP.mk 16).inStride (WP.mk 16).outStride (WP.mk 16).lenOff wHashLog
    w inPtr outPtr gm smemB
    (by decide) (by decide) hw (by decide) (by decide) (by decide)
    hw64 hib40 htop hbuf hderive hdisj (Nat.le_refl _) hlOtop hlOfit


/-- The buffer-placement contract the runtime must satisfy.

    The first clause is an EQUATION, not an inequality between two independent
    allocations, and the host makes it true by construction: one
    `cudaCreateBuffer` of `totIn + outAlloc` bytes, and a kernel that computes its
    output base as `in_ptr + totIn`.  `Lz4Host.host_single_allocation` reads that
    back out of the emitted program.  An inequality here would be the one clause
    no theorem could reach. -/
def LayoutOK (b : Nat) (inPtr outPtr : Nat) (gm : Array UInt8) : Prop :=
  -- The output region is the tail of the single allocation.  Per-warp
  -- disjointness would not be enough: warp `w'`'s output must also miss warp
  -- `w`'s INPUT slice, or warps could clobber each other's source data.
  outPtr = inPtr + ((WP.mk b).numBlk * (WP.mk b).inStride
    + AlgorithmLib.LZ4Simt.copySlack) ∧
  ∀ w, w < (WP.mk b).numBlk →
    inPtr + w * (WP.mk b).inStride < 2 ^ 40 ∧
    outPtr + w * (WP.mk b).outStride + 9 * (WP.mk b).inStride < 2 ^ 32 ∧
    outPtr + w * (WP.mk b).outStride + 9 * (WP.mk b).inStride ≤ gm.size ∧
    outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff + 3 < 2 ^ 64 ∧
    outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff + 4 ≤ gm.size

/-- **Disjointness / data-race-freedom.**  Distinct warps write disjoint ranges:
    warp `w` touches only `[outPtr + w*outStride, … + lenOff + 4)`, and consecutive
    strides are `outStride = lenOff + 8` apart, leaving 4 bytes of clearance. -/
theorem warp_regions_disjoint (b outPtr w w' : Nat) (hne : w ≠ w') (j : Nat)
    (hj : outPtr + w * (WP.mk b).outStride ≤ j)
    (hj2 : j < outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff + 4) :
    j < outPtr + w' * (WP.mk b).outStride ∨
      outPtr + w' * (WP.mk b).outStride + (WP.mk b).lenOff + 4 ≤ j := by
  have hstride : (WP.mk b).outStride = (WP.mk b).lenOff + 8 := rfl
  rcases Nat.lt_or_ge w w' with h | h
  · have h1 : (w + 1) * (WP.mk b).outStride ≤ w' * (WP.mk b).outStride :=
      Nat.mul_le_mul_right _ h
    have he : (w + 1) * (WP.mk b).outStride = w * (WP.mk b).outStride + (WP.mk b).outStride :=
      Nat.succ_mul w _
    left; omega
  · have hlt : w' < w := by omega
    have h1 : (w' + 1) * (WP.mk b).outStride ≤ w * (WP.mk b).outStride :=
      Nat.mul_le_mul_right _ hlt
    have he : (w' + 1) * (WP.mk b).outStride = w' * (WP.mk b).outStride + (WP.mk b).outStride :=
      Nat.succ_mul w' _
    right; omega

def LaunchAgreesPerWarp (b : Nat) (inPtr outPtr : Nat) (gm : Array UInt8)
    (smemB : List UInt8) (gfinal : Array UInt8) : Prop :=
  ∀ w, w < (WP.mk b).numBlk → ∀ (ss' : AlgorithmLib.LZ4Simt.SState),
    (∃ n, AlgorithmLib.LZ4Simt.SReaches (WP.mk b).kernel n
            (AlgorithmLib.LZ4Simt.initSt w inPtr outPtr gm smemB) ss') →
    ss'.pc = 272 →
    ∀ j, outPtr + w * (WP.mk b).outStride ≤ j →
         j < outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff + 4 →
      gfinal.getD j 0 = ss'.gmem.getD j 0

/-- **Whole-launch correctness.**  Given the memory-model assumption, EVERY block
    decodes correctly out of the FINAL memory — not merely each warp in isolation.
    This is what the host's download relies on. -/
theorem launch_correct (b : Nat) (inPtr outPtr : Nat) (gm : Array UInt8)
    (smemB : List UInt8) (gfinal : Array UInt8)
    (hcorrect : ShippedCorrect b)
    (hlayout : LayoutOK b inPtr outPtr gm)
    (hSC : LaunchAgreesPerWarp b inPtr outPtr gm smemB gfinal) :
    ∀ w, w < (WP.mk b).numBlk →
      ∃ k, 0 < k ∧ k ≤ (WP.mk b).lenOff ∧
        AlgorithmLib.readU32LE gfinal
          (outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff) = k ∧
        AlgorithmLib.LZ4Imp.decompress
          ((List.range k).map (fun i => gfinal.getD
            (outPtr + w * (WP.mk b).outStride + i) 0))
          (WP.mk b).inStride
          = some (gmemInpAt gm (inPtr + w * (WP.mk b).inStride) (WP.mk b).inStride) := by
  intro w hw
  obtain ⟨hglob, hper⟩ := hlayout
  obtain ⟨l1, l2, l3, l5, l6⟩ := hper w hw
  obtain ⟨n, ss', k, hreach, hpc272, hk0, hkle, hlenf, hdec, _hconf⟩ :=
    hcorrect w inPtr outPtr gm smemB hw hglob l1 l2 l3 l5 l6
  have hagree := hSC w hw ss' ⟨n, hreach⟩ hpc272
  refine ⟨k, hk0, hkle, ?_, ?_⟩
  · rw [← hlenf]
    unfold AlgorithmLib.readU32LE
    rw [hagree _ (by omega) (by omega), hagree _ (by omega) (by omega),
      hagree _ (by omega) (by omega), hagree _ (by omega) (by omega)]
  · rw [← hdec]
    congr 1
    apply List.map_congr_left
    intro i hi
    rw [List.mem_range] at hi
    exact hagree _ (by omega) (by omega)

end ShippedClaim

-- ── Seam guards: launch facts the roundtrip theorem cannot see ────────────────
section SeamGuards

/-- The corpus divides into whole blocks, so `inpAll.length = totIn` is coherent. -/
example : (WP.mk 15).totIn = corpusBytes := by decide
example : (WP.mk 16).totIn = corpusBytes := by decide

end SeamGuards

end Lz4Ship
