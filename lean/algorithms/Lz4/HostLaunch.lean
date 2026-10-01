module
public import Lz4.Kernel
meta import Lz4.Kernel
public import AlgorithmLib.Host.LaunchSpec
meta import AlgorithmLib.Host.LaunchSpec
public import AlgorithmLib.Host.DrvLaunch
meta import AlgorithmLib.Host.DrvLaunch
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The shipped LZ4 kernel, launched by a host program

`launch_correct` is about the SIMT machine: every block decodes out of any
final memory that agrees with each warp's own final memory on that warp's
output (`LaunchAgreesPerWarp`). `lz4Machine` is that agreement as the relation
the device must be honest about, on the one buffer the kernel is bound to, and
`lz4_launch` is the host triple: a `cudaLaunch` of the kernel from a world
where the call answers, on a device honest about this launch, leaves a buffer
out of which every block decodes to the input it was given. `lz4_drv_launch`
is the same through the driver's `cuLaunchKernel`, the kernel's one parameter
pointing into the allocation it decodes in.
-/

namespace Lz4Ship

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.DevSpec
  AlgorithmLib.HProg.Static AlgorithmLib.HProg.Contracts AlgorithmLib.Device AlgorithmLib.LZ4WarpDSL

/-- **What the kernel's machine leaves**, as a relation between the one buffer
    it is bound to before and after the launch. -/
def lz4Machine (b inPtr outPtr : Nat) (smemB : List UInt8) : List ByteArray → List ByteArray → Prop
  | [i], [o] => LaunchAgreesPerWarp b inPtr outPtr i.data smemB o.data
  | _, _ => True

/-- **The LZ4 launch triple.** From a world where `cudaLaunch` answers, bound to
    one live buffer whose layout the kernel expects, on a device honest about
    this launch, the call answers, and every block decodes out of the buffer
    it leaves to the input the buffer held before. -/
theorem lz4_launch {w : World} {ctx kptr nBufs bindPtr gx gy gz bx by_ bz : UInt64} {text : String}
    {b buf inPtr outPtr : Nat} {smemB : List UInt8}
    (hc : CtxOk w ctx) (hctx : ctx ≠ 0) (hk : KeepsSize w.kernel) (hcap : w.dev.capture = none)
    (hb : BindsReady w defaultParty nBufs bindPtr)
    (hdims : launchDimsBad nBufs [gx, gy, gz, bx, by_, bz] = false)
    (htext : readCStrAt w.mem kptr = some text)
    (hids : readIds w.mem bindPtr (asI32 nBufs).toNat = some [(buf : Int)])
    (hlive : (w.dev.get? (buf : Int)).isSome = true)
    (hcorrect : ShippedCorrect b) (hlayout : LayoutOK b inPtr outPtr (devMem w.dev buf).data)
    (hM : Honest w.kernel ⟨text, "main", [buf], [gx, gy, gz, bx, by_, bz], []⟩ (lz4Machine b inPtr outPtr smemB)) :
    ∃ r w', callBits .cudaLaunch [ctx, kptr, nBufs, bindPtr, gx, gy, gz, bx, by_, bz] w = some (r, w') ∧
      ∀ blk, blk < (WP.mk b).numBlk →
        ∃ k, 0 < k ∧ k ≤ (WP.mk b).lenOff ∧
          AlgorithmLib.readU32LE (devMem w'.dev buf).data
            (outPtr + blk * (WP.mk b).outStride + (WP.mk b).lenOff) = k ∧
          AlgorithmLib.LZ4Imp.decompress
            ((List.range k).map (fun i => (devMem w'.dev buf).data.getD
              (outPtr + blk * (WP.mk b).outStride + i) 0))
            (WP.mk b).inStride
            = some (gmemInpAt (devMem w.dev buf).data (inPtr + blk * (WP.mk b).inStride) (WP.mk b).inStride) := by
  obtain ⟨r, w', h, hMr⟩ := launch_triple (ids := [(buf : Int)]) (lz4Machine b inPtr outPtr smemB)
    hc hctx hk hcap hb hdims htext hids (by simp [hlive]) (by simp) (by simpa using hM)
  refine ⟨r, w', h, ?_⟩
  simp only [List.map, Int.toNat_natCast] at hMr
  exact launch_correct b inPtr outPtr _ smemB _ hcorrect hlayout hMr

/-- **The LZ4 launch triple, through the driver.** From a world where
    `cuLaunchKernel` answers with its one parameter pointing into allocation
    `buf`, laid out as the kernel expects, on a device honest about this
    launch, the call answers, and every block decodes out of the allocation it
    leaves to the input the allocation held before. -/
theorem lz4_drv_launch {w : World} {hf gx gy gz bx by_ bz shmem hstream params a : UInt64}
    {fi mi p buf off : Nat} {entry ptx : String} {sizes : List Nat} {B : ByteArray}
    {b inPtr outPtr : Nat} {smemB : List UInt8}
    (hl : w.dev.live = true) (hcap : w.dev.capture = none) (hk : KeepsSize w.kernel)
    (h1 : handleOf kFunc hf = some fi) (h2 : w.dev.funcs[fi]? = some (mi, entry))
    (h3 : w.dev.modules[mi]? = some (some ptx)) (h4 : w.dev.streamParty? hstream = some p)
    (h5 : ptxParamSizes ptx entry = some sizes) (h6 : kernelArgs w.mem params sizes = some [a])
    (ha : w.dev.range? a 0 = some (buf, off, B))
    (hdims : (DrvLaunch.dimsOf gx gy gz bx by_ bz).any (· == 0) = false)
    (hready : Ready w.dev.race p buf true)
    (hcorrect : ShippedCorrect b) (hlayout : LayoutOK b inPtr outPtr (devMem w.dev buf).data)
    (hM : Honest w.kernel ⟨ptx, entry, [buf], DrvLaunch.dimsOf gx gy gz bx by_ bz, [a]⟩
      (lz4Machine b inPtr outPtr smemB)) :
    ∃ r w', cudaDrv .launchKernel [hf, gx, gy, gz, bx, by_, bz, shmem, hstream, params, 0] w
        = some (r, w') ∧
      ∀ blk, blk < (WP.mk b).numBlk →
        ∃ k, 0 < k ∧ k ≤ (WP.mk b).lenOff ∧
          AlgorithmLib.readU32LE (devMem w'.dev buf).data
            (outPtr + blk * (WP.mk b).outStride + (WP.mk b).lenOff) = k ∧
          AlgorithmLib.LZ4Imp.decompress
            ((List.range k).map (fun i => (devMem w'.dev buf).data.getD
              (outPtr + blk * (WP.mk b).outStride + i) 0))
            (WP.mk b).inStride
            = some (gmemInpAt (devMem w.dev buf).data (inPtr + blk * (WP.mk b).inStride) (WP.mk b).inStride) := by
  have hb : DrvLaunch.bindsOf w.dev [a] = [buf] := by simp [DrvLaunch.bindsOf, ha, List.eraseDups_cons]
  obtain ⟨r, w', h, hMr⟩ := DrvLaunch.drvLaunch_triple (lz4Machine b inPtr outPtr smemB)
    hl hcap hk h1 h2 h3 h4 h5 h6 hdims
    (fun x hx id off' B' hr => by
      simp only [List.mem_singleton] at hx; subst hx
      rw [ha] at hr; cases hr; exact hready)
    (by rw [hb]; exact hM)
  refine ⟨r, w', h, ?_⟩
  rw [hb] at hMr
  exact launch_correct b inPtr outPtr _ smemB _ hcorrect hlayout hMr

end Lz4Ship
