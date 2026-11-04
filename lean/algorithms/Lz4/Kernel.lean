module
public import Lz4.Compressor
meta import Lz4.Compressor
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The warp compressor's kernel and its memory layout

The geometry the artifact ships (`WP`: block size, output stride, length
offset), the kernel it ships (`WP.code`) and the payload's layout derived from
the kernel's PTX text.
-/

open AlgorithmLib

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
def wBlockDim    : Nat := 32                  -- one warp a block (`WP.code` is a warp kernel)
def WP.gridX     (w : WP) : Nat := w.numBlk
def WP.totIn     (w : WP) : Nat := w.numBlk * w.inStride
def WP.totOut    (w : WP) : Nat := w.numBlk * w.outStride
/-- The allocation past the inputs and the copy slack: the blocks' outputs,
    which is all `Lz4Code.shipped_roundtrips` asks for (`WP.bytes`). -/
def WP.outAlloc  (w : WP) : Nat := w.totOut
/-- Where warp 0's output begins.  `copySlack` is the KERNEL's constant
    (`LZ4WarpDSL.copySlack`, the offset `Lz4Code.compressorS` puts the outputs at); using it
    here is what keeps the host allocation and the kernel's own base in step. -/
def WP.outOff    (w : WP) : Nat := w.totIn + AlgorithmLib.LZ4WarpDSL.copySlack

/-- The kernel this artifact ships: the warp language's compressor as Code,
    its entry point `main` (`Lz4Code.shipped_roundtrips` is stated about it). -/
def WP.code (w : WP) : AlgorithmLib.Kernel.Kernel :=
  Lz4Code.compressorKernel "main" w.numBlk w.inStride w.outStride w.lenOff wHashLog

/-- The PTX text the artifact carries: `WP.code`'s module, printed. Empty if it
    does not compile, which `Lz4.Comp.warpArtifactDSL` refuses. -/
def warpKernelDSLStr (w : WP) : String :=
  match w.code.ptx with
  | .ok s => s
  | .error _ => ""

-- ── Payload layout, derived from the serialized kernel ────────────────────────
-- Offsets are relative to the PTX's actual length, so there is no fixed slot to
-- overflow (Nat subtraction would truncate silently rather than fail).
/-- The bytes at `rPTX_OFF`.  `ptxLen` is this list's length and `warpPayloadDSL`
    emits this same list, so offsets and image cannot disagree. -/
def WP.ptxBytes  (w : WP) : List UInt8 := (warpKernelDSLStr w).toUTF8.toList ++ [0]
def WP.ptxLen    (w : WP) : Nat := w.ptxBytes.length
/-- The binding table, 16-byte aligned. -/
def WP.bindOff   (w : WP) : Nat := (rPTX_OFF + w.ptxLen + 15) / 16 * 16
def WP.memSize   (w : WP) : Nat := w.bindOff + 0x140

-- ── Seam guards: launch facts the roundtrip theorem cannot see ────────────────
section SeamGuards

/-- The corpus divides into whole blocks, so `inpAll.length = totIn` is coherent. -/
example : (WP.mk 15).totIn = corpusBytes := by decide
example : (WP.mk 16).totIn = corpusBytes := by decide

end SeamGuards

end Lz4Ship
