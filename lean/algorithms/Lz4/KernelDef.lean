module
public import Lz4.Kernel
meta import Lz4.Kernel
public import AlgorithmLib.Host.DevWorld
meta import AlgorithmLib.Host.DevWorld
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The LZ4 decompressor as a device kernel

The kernel is its SIMT program; its step, on the one buffer it is bound to, is
the launch as the World runs it. `launch_correct` says what the launch leaves
there, given the memory-model assumption it names (`LaunchAgreesPerWarp`):
every block decodes out of the final bytes. `lz4Kernel_decodes` is that
statement about the bytes this step leaves, so the decompressor stands in a
`DevProg` beside the other kernels with its correctness attached.
-/

namespace Lz4Ship

open AlgorithmLib.Device
open AlgorithmLib.HProg.Sem
open AlgorithmLib.LZ4WarpDSL

/-- The launch of the decompressor for block size `2^b`, bound to one buffer. -/
def lz4Launch (b buf : Nat) : Launch :=
  { kernel := "lz4", entry := "main", binds := [buf], dims := [] }

/-- **The decompressor as a kernel**: its SIMT code, and at a binding the launch
    step whose answer is the kernel oracle's. -/
def lz4Kernel (b : Nat) (kernel : Launch → List ByteArray → List ByteArray) :
    KernelDef (Array AlgorithmLib.LZ4Simt.SInstr) where
  name := "lz4"
  code := (WP.mk b).kernel
  step binds := launchStep kernel (lz4Launch b (binds.headD 0)) binds

/-- **Every block decodes out of what the kernel's step leaves**, given the
    shipped-kernel correctness, the layout, and the memory-model assumption
    `launch_correct` rests on --- stated of the step's own output. -/
theorem lz4Kernel_decodes (b buf inPtr outPtr : Nat) (kernel : Launch → List ByteArray → List ByteArray)
    (m : DevMem) (smemB : List UInt8) (hcorrect : ShippedCorrect b)
    (hlayout : LayoutOK b inPtr outPtr (m buf).data)
    (hSC : LaunchAgreesPerWarp b inPtr outPtr (m buf).data smemB
      (((lz4Kernel b kernel).step [buf]).run m buf).data) :
    ∀ w, w < (WP.mk b).numBlk →
      ∃ k, 0 < k ∧ k ≤ (WP.mk b).lenOff ∧
        AlgorithmLib.readU32LE (((lz4Kernel b kernel).step [buf]).run m buf).data
          (outPtr + w * (WP.mk b).outStride + (WP.mk b).lenOff) = k ∧
        AlgorithmLib.LZ4Imp.decompress
          ((List.range k).map (fun i => (((lz4Kernel b kernel).step [buf]).run m buf).data.getD
            (outPtr + w * (WP.mk b).outStride + i) 0))
          (WP.mk b).inStride
          = some (gmemInpAt (m buf).data (inPtr + w * (WP.mk b).inStride) (WP.mk b).inStride) :=
  launch_correct b inPtr outPtr (m buf).data smemB _ hcorrect hlayout hSC

end Lz4Ship
