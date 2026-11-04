module
public import AlgorithmLib.Kernel.Law
meta import AlgorithmLib.Kernel.Law
public import AlgorithmLib.Host.DrvLaunch
meta import AlgorithmLib.Host.DrvLaunch
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Kernel.Host` — a host program's launch of a printed kernel

`drv_launch` is the host triple for a kernel printed by `Kernel.ptx`: a
`cuLaunchKernel` of its text, one-dimensional, from a world where the call
answers, on a device that runs the text as `Kernel.Runs` says, answers; and
where the body's threads run, each buffer the launch binds holds afterwards
what one run of them leaves in it where the kernel may write it, and what it
held where it only reads it (`drv_launch_flags`). A kernel's own theorem
about a run (saxpy's `launch_spec`) then says what that is.
-/

namespace AlgorithmLib.Kernel

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.DevSpec
  AlgorithmLib.HProg.Contracts AlgorithmLib.Device

/-- A word below `2^32` points into no device buffer. -/
theorem devPtrOf_small {x : UInt64} (h : x.toNat < 2 ^ 32) : devPtrOf x = none := by
  unfold devPtrOf; simp only; rw [if_neg (by simp; omega)]

/-- The launch a driver call of `k` makes. -/
def drvLaunchOf (k : Kernel) (d : Dev) (gx bx : UInt64) (args : List UInt64) : Launch :=
  ⟨k.text, k.name, DrvLaunch.bindsOf d args, DrvLaunch.dimsOf gx 1 1 bx 1 1, args,
    d.writesOf (d.kernelWrites k.text k.name) args⟩

/-- **A printed kernel's launch through the driver.** Each buffer it binds
    holds afterwards what the body's threads leave in it where the kernel may
    write it (`Dev.kernelWrites`), and what it held where the kernel only
    reads it. -/
theorem drv_launch_flags {w : World} {k : Kernel} {hf gx bx shmem hstream params : UInt64}
    {fi mi p : Nat} {sizes : List Nat} {args : List UInt64}
    (hruns : Kernel.Runs w.kernel k)
    (hl : w.dev.live = true) (hcap : w.dev.capture = none) (hk : KeepsSize w.kernel)
    (h1 : handleOf kFunc hf = some fi) (h2 : w.dev.funcs[fi]? = some (mi, k.name))
    (h3 : w.dev.modules[mi]? = some (some k.text)) (h4 : w.dev.streamParty? hstream = some p)
    (h5 : ptxParamSizes k.text k.name = some sizes) (h6 : kernelArgs w.mem params sizes = some args)
    (hdims : (DrvLaunch.dimsOf gx 1 1 bx 1 1).any (· == 0) = false)
    (hr : ∀ a ∈ args, ∀ id off b, w.dev.range? a 0 = some (id, off, b) → Ready w.dev.race p id true) :
    ∃ r w', cudaDrv .launchKernel [hf, gx, 1, 1, bx, 1, 1, shmem, hstream, params, 0] w = some (r, w') ∧
      let l := drvLaunchOf k w.dev gx bx args
      (∃ bufs, LaunchRuns k (gx &&& 0xffffffff).toNat (bx &&& 0xffffffff).toNat (k.argsOf l)
        (bindsMem l (l.binds.map (devMem w.dev))) bufs) →
      ∃ bufs, LaunchRuns k (gx &&& 0xffffffff).toNat (bx &&& 0xffffffff).toNat (k.argsOf l)
          (bindsMem l (l.binds.map (devMem w.dev))) bufs ∧
        l.binds.map (devMem w'.dev) = l.after l.binds (l.binds.map (devMem w.dev)) (l.binds.map bufs) := by
  let l := drvLaunchOf k w.dev gx bx args
  obtain ⟨r, w', h, hM, -⟩ := DrvLaunch.drvLaunch_triple (w := w)
    (fun ins outs => (∃ bufs, LaunchRuns k (gx &&& 0xffffffff).toNat (bx &&& 0xffffffff).toNat (k.argsOf l)
        (bindsMem l ins) bufs) →
      ∃ bufs, LaunchRuns k (gx &&& 0xffffffff).toNat (bx &&& 0xffffffff).toNat (k.argsOf l) (bindsMem l ins) bufs ∧
        outs = l.after l.binds ins (l.binds.map bufs))
    hl hcap hk h1 h2 h3 h4 h5 h6 hdims hr
    (fun ins hex => by
      obtain ⟨bufs, hb, e⟩ := hruns l ins _ _ rfl rfl
        (by simp only [l, drvLaunchOf, DrvLaunch.dimsOf, List.map]; rfl) hex
      exact ⟨bufs, hb, congrArg (l.after l.binds ins) e⟩)
  exact ⟨r, w', h, hM⟩

/-- **A printed kernel's launch through the driver**, of a kernel the device
    says may write every buffer it binds: they hold afterwards what the body's
    threads leave in them. -/
theorem drv_launch {w : World} {k : Kernel} {hf gx bx shmem hstream params : UInt64}
    {fi mi p : Nat} {sizes : List Nat} {args : List UInt64}
    (hruns : Kernel.Runs w.kernel k)
    (hl : w.dev.live = true) (hcap : w.dev.capture = none) (hk : KeepsSize w.kernel)
    (h1 : handleOf kFunc hf = some fi) (h2 : w.dev.funcs[fi]? = some (mi, k.name))
    (h3 : w.dev.modules[mi]? = some (some k.text)) (h4 : w.dev.streamParty? hstream = some p)
    (h5 : ptxParamSizes k.text k.name = some sizes) (h6 : kernelArgs w.mem params sizes = some args)
    (hdims : (DrvLaunch.dimsOf gx 1 1 bx 1 1).any (· == 0) = false)
    (hr : ∀ a ∈ args, ∀ id off b, w.dev.range? a 0 = some (id, off, b) → Ready w.dev.race p id true)
    (hall : ∀ i : Nat, ((w.dev.kernelWrites k.text k.name).bind (·[i]?)).getD true = true) :
    ∃ r w', cudaDrv .launchKernel [hf, gx, 1, 1, bx, 1, 1, shmem, hstream, params, 0] w = some (r, w') ∧
      let l := drvLaunchOf k w.dev gx bx args
      (∃ bufs, LaunchRuns k (gx &&& 0xffffffff).toNat (bx &&& 0xffffffff).toNat (k.argsOf l)
        (bindsMem l (l.binds.map (devMem w.dev))) bufs) →
      ∃ bufs, LaunchRuns k (gx &&& 0xffffffff).toNat (bx &&& 0xffffffff).toNat (k.argsOf l)
          (bindsMem l (l.binds.map (devMem w.dev))) bufs ∧
        l.binds.map (devMem w'.dev) = l.binds.map bufs := by
  obtain ⟨r, w', h, hM⟩ := drv_launch_flags (shmem := shmem) hruns hl hcap hk h1 h2 h3 h4 h5 h6 hdims hr
  refine ⟨r, w', h, fun hex => ?_⟩
  obtain ⟨bufs, hb, e⟩ := hM hex
  refine ⟨bufs, hb, ?_⟩
  rw [e, Launch.after_all _ _ _ _ (fun b hb' => by
    show b ∈ (drvLaunchOf k w.dev gx bx args).writes
    simp only [drvLaunchOf, DrvLaunch.writesOf_all args hall]; exact hb')]

end AlgorithmLib.Kernel
