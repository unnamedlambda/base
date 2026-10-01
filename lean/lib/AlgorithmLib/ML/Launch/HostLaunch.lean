module
public import AlgorithmLib.ML.Launch.KernelDefs
meta import AlgorithmLib.ML.Launch.KernelDefs
public import AlgorithmLib.ML.Launch.Sequence
meta import AlgorithmLib.ML.Launch.Sequence
public import AlgorithmLib.Host.LaunchSpec
meta import AlgorithmLib.Host.LaunchSpec
public import AlgorithmLib.Host.DrvLaunch
meta import AlgorithmLib.Host.DrvLaunch
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# A stage, launched by a host program

A stage is proven over the warp machine: `runGrid_step` says its run leaves
`S.step` of the memory it starts from. `stageMachine` is what the device must be
honest about for a stage's launch --- at every word the stage owns, the bits the
warp machine's run leaves there, from the bound buffers read as floats --- and
`stage_launch` is the host triple: a `cudaLaunch` of the stage, on a device
honest about that launch, leaves at every owned word of its output the bits of
the stage's value there. `stage_drv_launch` is the same through
`cuLaunchKernel` called directly, where the buffers bound are those the
parameters point into.
-/

namespace AlgorithmLib.ML

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.DevSpec
  AlgorithmLib.HProg.Static AlgorithmLib.HProg.Contracts AlgorithmLib.Device

/-- The device memory a launch's bound buffers make, by id; an unbound id reads
    empty. -/
def boundMem (ns : List Nat) (bs : List ByteArray) : DevMem :=
  writeAll (fun _ => ByteArray.empty) (ns.zip bs)

theorem boundMem_map {ns : List Nat} (hnd : ns.Nodup) (m : DevMem) {b : Nat} (hb : b ∈ ns) :
    boundMem ns (ns.map m) b = m b := by
  have h := writeAll_zip_map ns (ns.map m) (fun _ => ByteArray.empty) hnd (by simp)
  exact (List.map_inj_left.mp h) b hb

/-- **What a stage's machine leaves**, on the buffers bound to its launch: at
    every word the stage owns in its output, the bits the warp machine's run
    leaves there, from any state whose memory is the bound buffers' view. -/
def stageMachine (S : StageSpec) (ns : List Nat) (ins outs : List ByteArray) : Prop :=
  ∀ st : WSt, st.mem = viewMem (boundMem ns ins) → ∀ a, (∃ cta, cta < S.grid ∧ S.dom cta a) →
    wordAt (boundMem ns outs S.out) a = ((runGrid S.blk S.grid st).mem S.out a).toBits

/-- **The stage launch triple.** From a world where `cudaLaunch` answers, bound
    to live buffers once each, among them the stage's output, on a device honest
    about this launch, the call answers, and at every word the stage owns its
    output holds the bits of the stage's value there, computed from the bound
    buffers. -/
theorem stage_launch {w : World} {ctx kptr nBufs bindPtr gx gy gz bx by_ bz : UInt64} {text : String}
    (S : StageSpec) (hex : S.Exclusive) {ns : List Nat}
    (hc : CtxOk w ctx) (hctx : ctx ≠ 0) (hk : KeepsSize w.kernel) (hcap : w.dev.capture = none)
    (hb : BindsReady w defaultParty nBufs bindPtr)
    (hdims : launchDimsBad nBufs [gx, gy, gz, bx, by_, bz] = false)
    (htext : readCStrAt w.mem kptr = some text)
    (hids : readIds w.mem bindPtr (asI32 nBufs).toNat = some (ns.map Int.ofNat))
    (hlive : (ns.map Int.ofNat).any (fun id => (w.dev.get? id).isNone) = false)
    (hnd : ns.Nodup) (hout : S.out ∈ ns)
    (hM : Honest w.kernel ⟨text, "main", ns, [gx, gy, gz, bx, by_, bz], []⟩ (stageMachine S ns)) :
    ∃ r w', callBits .cudaLaunch [ctx, kptr, nBufs, bindPtr, gx, gy, gz, bx, by_, bz] w = some (r, w') ∧
      ∀ a, (∃ cta, cta < S.grid ∧ S.dom cta a) →
        wordAt (devMem w'.dev S.out) a =
          (S.step (viewMem (boundMem ns (ns.map (devMem w.dev)))) S.out a).toBits := by
  have hmap : (ns.map Int.ofNat).map Int.toNat = ns := by simp [List.map_map, Function.comp_def]
  obtain ⟨r, w', h, hMr⟩ := launch_triple (ids := ns.map Int.ofNat) (stageMachine S ns)
    hc hctx hk hcap hb hdims htext hids hlive (by rw [hmap]; exact hnd) (by rw [hmap]; exact hM)
  rw [hmap] at hMr
  refine ⟨r, w', h, fun a hown => ?_⟩
  have e := hMr ⟨fun _ _ => 0, viewMem (boundMem ns (ns.map (devMem w.dev))), fun _ => 0⟩ rfl a hown
  rw [boundMem_map hnd _ hout, runGrid_step S hex] at e
  exact e

/-- **The stage launch triple, through the driver.** From a world where
    `cuLaunchKernel` answers, whose parameters point into allocations `ns`,
    among them the stage's output, on a device honest about this launch, the
    call answers, and at every word the stage owns its output holds the bits of
    the stage's value there, computed from those allocations. -/
theorem stage_drv_launch {w : World} {hf gx gy gz bx by_ bz shmem hstream params : UInt64}
    {fi mi p : Nat} {entry ptx : String} {sizes : List Nat} {args : List UInt64}
    (S : StageSpec) (hex : S.Exclusive) {ns : List Nat}
    (hl : w.dev.live = true) (hcap : w.dev.capture = none) (hk : KeepsSize w.kernel)
    (h1 : handleOf kFunc hf = some fi) (h2 : w.dev.funcs[fi]? = some (mi, entry))
    (h3 : w.dev.modules[mi]? = some (some ptx)) (h4 : w.dev.streamParty? hstream = some p)
    (h5 : ptxParamSizes ptx entry = some sizes) (h6 : kernelArgs w.mem params sizes = some args)
    (hdims : (DrvLaunch.dimsOf gx gy gz bx by_ bz).any (· == 0) = false)
    (hr : ∀ a ∈ args, ∀ id off b, w.dev.range? a 0 = some (id, off, b) → Ready w.dev.race p id true)
    (hns : DrvLaunch.bindsOf w.dev args = ns) (hout : S.out ∈ ns)
    (hM : Honest w.kernel ⟨ptx, entry, ns, DrvLaunch.dimsOf gx gy gz bx by_ bz, args⟩
      (stageMachine S ns)) :
    ∃ r w', cudaDrv .launchKernel [hf, gx, gy, gz, bx, by_, bz, shmem, hstream, params, 0] w
        = some (r, w') ∧
      ∀ a, (∃ cta, cta < S.grid ∧ S.dom cta a) →
        wordAt (devMem w'.dev S.out) a =
          (S.step (viewMem (boundMem ns (ns.map (devMem w.dev)))) S.out a).toBits := by
  have hnd : ns.Nodup := hns ▸ DrvLaunch.nodup_eraseDups _
  subst hns
  obtain ⟨r, w', h, hMr⟩ := DrvLaunch.drvLaunch_triple (shmem := shmem) (stageMachine S _)
    hl hcap hk h1 h2 h3 h4 h5 h6 hdims hr hM
  refine ⟨r, w', h, fun a hown => ?_⟩
  have e := hMr ⟨fun _ _ => 0, viewMem (boundMem _ ((DrvLaunch.bindsOf w.dev args).map (devMem w.dev))),
    fun _ => 0⟩ rfl a hown
  rw [boundMem_map hnd _ hout, runGrid_step S hex] at e
  exact e

end AlgorithmLib.ML
