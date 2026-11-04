module
public import Lz4.CodeShip
meta import Lz4.CodeShip
public import AlgorithmLib.Host.LaunchSpec
meta import AlgorithmLib.Host.LaunchSpec
public import AlgorithmLib.Host.DrvLaunch
meta import AlgorithmLib.Host.DrvLaunch
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The shipped LZ4 kernel, launched by a host program

A launch of the kernel leaves, in the one allocation it is bound to, what the
warp's runs of the blocks' bodies denote (`Lz4Code.Denotes`), on the Code and
on the PTX it compiles to (`launch_denotes`). `lz4Machine` is that denotation
as the relation the device must be honest about, and the PTX model's own
launch realizes it (`lz4Machine_realized`). `lz4_drv_launch` is the host
triple: a `cuLaunchKernel` of the kernel, its two parameters pointing at that
allocation, on a device honest about the launch, leaves an allocation out of
which every block decodes to the input it held.
-/

namespace Lz4Code

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.DevSpec
  AlgorithmLib.HProg.Contracts AlgorithmLib.Device AlgorithmLib.Kernel AlgorithmLib.LZ4WarpDSL Lz4Ship

/-- **What the kernel leaves**, as a relation between the one allocation it is
    bound to before and after the launch: what the warp's runs denote. -/
def lz4Machine (b : Nat) : List ByteArray → List ByteArray → Prop
  | [i], [o] => Denotes (WP.mk b).numBlk (WP.mk b).inStride (WP.mk b).outStride (WP.mk b).lenOff wHashLog i o
  | _, _ => True

/-- The geometry facts a shipped launch rests on. -/
theorem shipped_geometry (b : Nat) (hb : b = 15 ∨ b = 16) :
    16 ≤ (WP.mk b).inStride ∧ (WP.mk b).inStride ≤ 65536 ∧ wHashLog ≤ 32 ∧ 2 * 2 ^ wHashLog + 64 < 2 ^ 36 ∧
      (WP.mk b).inStride + (WP.mk b).inStride / 16 + 256 ≤ (WP.mk b).lenOff ∧
      (WP.mk b).lenOff + 4 ≤ (WP.mk b).outStride ∧ (WP.mk b).bytes ≤ 2 ^ 36 := by
  rcases hb with rfl | rfl <;> decide

/-- **The PTX model's launch realizes the machine**: a launch of the shipped
    kernel's entry, on device memory whose buffer 0 holds the layout, leaves
    buffer 0 as `lz4Machine` says. -/
theorem lz4Machine_realized (b : Nat) (hb : b = 15 ∨ b = 16) (hw : Hw) (hhw : PtxSound.HwOk hw) (outA : UInt64)
    (bufs : Nat → ByteArray) (hsize : (WP.mk b).bytes ≤ (bufs 0).size) :
    ∃ e, (WP.mk b).code.shipped = .ok e ∧ ∃ bufs', RunsLaunch (PtxRun.runner e hw) (WP.mk b).code.smem
      (WP.mk b).numBlk 32 [.sc .i64 base0, .sc .i64 outA] bufs bufs' hw.nan ∧
      lz4Machine b [bufs 0] [bufs' 0] := by
  have hc : ∀ k : Kernel, k.entry.isOk = true → ∃ e, k.entry = .ok e := fun k h => by
    cases he : k.entry with
    | error _ => rw [he] at h; cases h
    | ok e => exact ⟨e, rfl⟩
  obtain ⟨g1, g2, g3, g4, g5, g6, g7⟩ := shipped_geometry b hb
  rcases hb with rfl | rfl
  · obtain ⟨e, he⟩ := hc _ compiles15
    have he' := he
    rw [code15] at he'
    obtain ⟨u, -, hP, hd⟩ := launch_denotes hw hhw "main" outA he' bufs (by decide) (by decide) (by decide) (by decide)
      fine15 flags15 (by decide) (by decide) (by decide) (by decide) hsize (by decide)
    exact ⟨Coalesce.run e, PtxSound.shipped_ok _ e he, u.mem.bufs,
      by rw [PtxSound.runner_shipped e hw hhw, code15]; exact hP, hd⟩
  · obtain ⟨e, he⟩ := hc _ compiles16
    have he' := he
    rw [code16] at he'
    obtain ⟨u, -, hP, hd⟩ := launch_denotes hw hhw "main" outA he' bufs (by decide) (by decide) (by decide) (by decide)
      fine16 flags16 (by decide) (by decide) (by decide) (by decide) hsize (by decide)
    exact ⟨Coalesce.run e, PtxSound.shipped_ok _ e he, u.mem.bufs,
      by rw [PtxSound.runner_shipped e hw hhw, code16]; exact hP, hd⟩

/-- **The LZ4 launch triple, through the driver.** From a world where
    `cuLaunchKernel` answers with both its parameters pointing into allocation
    `buf`, which holds the layout, on a device honest about this launch, the
    call answers, and every block decodes out of the allocation it leaves to
    the input the allocation held before. -/
theorem lz4_drv_launch {w : World} {hf gx gy gz bx by_ bz shmem hstream params a : UInt64}
    {fi mi p buf off : Nat} {entry ptx : String} {sizes : List Nat} {B : ByteArray}
    (b : Nat) (hb : b = 15 ∨ b = 16)
    (hl : w.dev.live = true) (hcap : w.dev.capture = none) (hk : KeepsSize w.kernel)
    (h1 : handleOf kFunc hf = some fi) (h2 : w.dev.funcs[fi]? = some (mi, entry))
    (h3 : w.dev.modules[mi]? = some (some ptx)) (h4 : w.dev.streamParty? hstream = some p)
    (h5 : ptxParamSizes ptx entry = some sizes) (h6 : kernelArgs w.mem params sizes = some [a, a])
    (ha : w.dev.range? a 0 = some (buf, off, B))
    (hdims : (DrvLaunch.dimsOf gx gy gz bx by_ bz).any (· == 0) = false)
    (hready : Ready w.dev.race p buf true)
    (hsize : (WP.mk b).bytes ≤ (devMem w.dev buf).size)
    (hM : Honest w.kernel ⟨ptx, entry, [buf], DrvLaunch.dimsOf gx gy gz bx by_ bz, [a, a],
      w.dev.writesOf (w.dev.kernelWrites ptx entry) [a, a]⟩ (lz4Machine b)) :
    ∃ r w', cudaDrv .launchKernel [hf, gx, gy, gz, bx, by_, bz, shmem, hstream, params, 0] w
        = some (r, w') ∧
      ∀ c, c < (WP.mk b).numBlk → ∃ n, n ≤ (WP.mk b).lenOff ∧
        u32At (devMem w'.dev buf) ((WP.mk b).numBlk * (WP.mk b).inStride + copySlack + c * (WP.mk b).outStride +
          (WP.mk b).lenOff) = n ∧
        AlgorithmLib.LZ4Imp.decompress ((List.range n).map fun i =>
            (devMem w'.dev buf).get! ((WP.mk b).numBlk * (WP.mk b).inStride + copySlack + c * (WP.mk b).outStride + i))
            (WP.mk b).inStride
          = some ((List.range (WP.mk b).inStride).map fun j => (devMem w.dev buf).get! (c * (WP.mk b).inStride + j)) := by
  have hbs : DrvLaunch.bindsOf w.dev [a, a] = [buf] := by simp [DrvLaunch.bindsOf, ha, List.eraseDups_cons]
  obtain ⟨r, w', h, hMr, -⟩ := DrvLaunch.drvLaunch_triple (lz4Machine b)
    hl hcap hk h1 h2 h3 h4 h5 h6 hdims
    (fun x hx id off' B' hr => by
      simp only [List.mem_cons, List.not_mem_nil, or_false, or_self] at hx; subst hx
      rw [ha] at hr; cases hr; exact hready)
    (by rw [hbs]; exact hM)
  refine ⟨r, w', h, ?_⟩
  rw [hbs] at hMr
  obtain ⟨g1, g2, g3, g4, g5, g6, g7⟩ := shipped_geometry b hb
  exact decode_of_denotes g1 g2 g3 g4 g5 g6 g7 hsize hMr

end Lz4Code
