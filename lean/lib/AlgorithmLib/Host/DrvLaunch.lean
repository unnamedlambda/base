module
public import AlgorithmLib.Host.LaunchSpec
meta import AlgorithmLib.Host.LaunchSpec
public import AlgorithmLib.Host.ExtContracts
meta import AlgorithmLib.Host.ExtContracts
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Host.DrvLaunch` — `cuLaunchKernel`, as a triple over what the kernel computes

The direct launch's counterpart of `launch_triple`. The allocations a launch
binds are those its parameters point into, each once; under the device's
honesty about the launch it makes (`Honest`), the call answers and those
allocations after it stand in `M` to what they held before. A kernel theorem
about `M` then says what they hold (`Lz4.HostLaunch` for the shipped LZ4
kernel).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.DevSpec
  AlgorithmLib.HProg.Static AlgorithmLib.HProg.Contracts AlgorithmLib.HProg.ExtContracts AlgorithmLib.Device

namespace AlgorithmLib.HProg.DrvLaunch

theorem nodup_eraseDups {α} [BEq α] [LawfulBEq α] : ∀ (l : List α), l.eraseDups.Nodup
  | [] => by simp
  | a :: as => by
      rw [List.eraseDups_cons]
      refine List.nodup_cons.mpr ⟨fun h => ?_, nodup_eraseDups _⟩
      have := (List.mem_filter.mp (mem_of_mem_eraseDups h)).2
      simp at this
termination_by l => l.length
decreasing_by exact Nat.lt_succ_of_le (List.length_filter_le _ _)

/-- The allocations a launch with parameter values `args` binds. -/
def bindsOf (d : Dev) (args : List UInt64) : List Nat :=
  (args.filterMap fun a => (d.range? a 0).map (·.1)).eraseDups

/-- The launch grid as the driver reads it: six 32-bit counts. -/
def dimsOf (gx gy gz bx by_ bz : UInt64) : List UInt64 :=
  [gx, gy, gz, bx, by_, bz].map (· &&& 0xffffffff)

/-- **The direct launch triple.** From a world where `cuLaunchKernel` answers
    — a live context, no capture in progress, a function from a loaded module,
    a live stream, parameters it can read, a grid with no zero dimension, and
    each allocation a parameter points into ready for the stream — on a device
    honest about the launch it makes, the call answers, and the allocations it
    binds stand in `M` to what they held. -/
theorem drvLaunch_triple {w : World} {hf gx gy gz bx by_ bz shmem hstream params : UInt64}
    {fi mi p : Nat} {entry ptx : String} {sizes : List Nat} {args : List UInt64}
    (M : List ByteArray → List ByteArray → Prop)
    (hl : w.dev.live = true) (hcap : w.dev.capture = none) (hk : KeepsSize w.kernel)
    (h1 : handleOf kFunc hf = some fi) (h2 : w.dev.funcs[fi]? = some (mi, entry))
    (h3 : w.dev.modules[mi]? = some (some ptx)) (h4 : w.dev.streamParty? hstream = some p)
    (h5 : ptxParamSizes ptx entry = some sizes) (h6 : kernelArgs w.mem params sizes = some args)
    (hdims : (dimsOf gx gy gz bx by_ bz).any (· == 0) = false)
    (hr : ∀ a ∈ args, ∀ id off b, w.dev.range? a 0 = some (id, off, b) → Ready w.dev.race p id true)
    (hM : Honest w.kernel ⟨ptx, entry, bindsOf w.dev args, dimsOf gx gy gz bx by_ bz, args⟩ M) :
    ∃ r w', cudaDrv .launchKernel [hf, gx, gy, gz, bx, by_, bz, shmem, hstream, params, 0] w
        = some (r, w') ∧
      M ((bindsOf w.dev args).map (devMem w.dev)) ((bindsOf w.dev args).map (devMem w'.dev)) := by
  have hpre : cudaPre .launchKernel [hf, gx, gy, gz, bx, by_, bz, shmem, hstream, params, 0] w :=
    ⟨Or.inl hcap, hl, rfl, hk, fi, mi, entry, ptx, p, sizes, args, h1, h2, h3, h4, h5, h6,
      fun a ha id off b hra => by rw [raceFor_of_not_capturing (by simp [Dev.capturing, hcap])]
                                  exact hr a ha id off b hra⟩
  obtain ⟨⟨r, w'⟩, h⟩ := Option.isSome_iff_exists.mp (cudaPre_safe _ _ w hpre)
  refine ⟨r, w', h, ?_⟩
  unfold cudaDrv at h
  simp only [hcap, Option.isSome_none, Bool.false_and, Bool.false_eq_true, if_false] at h
  simp only [cudaStep, hl, Bool.not_true, bne_self_eq_false, Bool.or_false, Bool.false_eq_true,
    if_false, h1, h2, h3, h4, h5, h6, Option.bind_eq_bind, Option.bind_some, Option.join_some] at h
  simp only [dimsOf] at hdims hM
  rw [if_neg (by rw [hdims]; exact Bool.false_ne_true)] at h
  obtain ⟨d, hkd, rfl⟩ := devOnly_eq h
  obtain ⟨d0, hd0, hkd⟩ := Option.bind_eq_some_iff.mp hkd
  simp only [cuRes, Option.some.injEq, Prod.mk.injEq] at hkd
  obtain ⟨-, rfl⟩ := hkd
  simp only [bindsOf] at hM ⊢
  rw [devOp_launch_mem hcap hd0]
  exact launchStep_honest hk hM (nodup_eraseDups _) _

end AlgorithmLib.HProg.DrvLaunch
