module
public import AlgorithmLib.Host.Lifecycle
meta import AlgorithmLib.Host.Lifecycle
public import AlgorithmLib.Host.DevWorld
meta import AlgorithmLib.Host.DevWorld
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Host.LaunchSpec` — a launch, as a triple over what the kernel computes

The model does not run PTX: what a launch leaves in its bound buffers is the
world's kernel oracle, `World.kernel`. A kernel's own proof is about a modeled
machine, so what joins the two is one statement about the device, named here
once: `Honest w.kernel l M`, every output the device gives for launch `l` is one
the modeled machine relates to its input by `M`. That is the GPU-side trusted
reading, as `callBits` is the host-side one.

The host rule for a launch made directly is `DrvLaunch.drvLaunch_triple`,
through `launchStep_honest`: from a world where `cuLaunchKernel` answers, every
bound allocation live and ready, and the device honest about the launch it
makes, the call answers and the bound allocations after it stand in `M` to
what they held before. A kernel theorem about `M` then says what they hold
(`Lz4.HostLaunch` for the shipped LZ4 kernel).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.DevSpec
  AlgorithmLib.Device

namespace AlgorithmLib.HProg.Contracts

/-- **The GPU-side reading, named once.** Every output the device gives for
    launch `l` is one the modeled machine relates to the launch's input by `M`. -/
def Honest (k : Launch → List ByteArray → List ByteArray) (l : Launch)
    (M : List ByteArray → List ByteArray → Prop) : Prop :=
  ∀ ins, M ins (k l ins)

theorem devOp_launch_mem {d d' : Dev} {w : World} {p : Nat} {l : Launch} {ns : List Nat}
    (hc : d.capture = none) (h : d.devOp w p (.launch l ns) [] ns = some d') :
    devMem d' = (launchStep w.kernel l ns).run (devMem d) := by
  unfold Dev.devOp at h
  rw [hc] at h
  unfold Dev.devOp.run at h
  obtain ⟨r, -, h⟩ := Option.bind_eq_some_iff.mp h
  dsimp only at h
  have := runLaunch_mem _ _ _ _ _ h
  exact this

theorem zip_map_eq {α β : Type} (f : α → β) : ∀ (as bs : List α), as.map f = bs.map f →
    ∀ a b, (a, b) ∈ as.zip bs → f a = f b
  | [], _, _, _, _, h => by simp at h
  | _ :: _, [], _, _, _, h => by simp at h
  | x :: as, y :: bs, he, a, b, h => by
      simp only [List.map_cons, List.cons.injEq] at he
      simp only [List.zip_cons_cons, List.mem_cons, Prod.mk.injEq] at h
      rcases h with ⟨rfl, rfl⟩ | h
      · exact he.1
      · exact zip_map_eq f as bs he.2 a b h

theorem writeAll_zip_map : ∀ (ns : List Nat) (outs : List ByteArray) (m : DevMem),
    ns.Nodup → outs.length = ns.length → ns.map (writeAll m (ns.zip outs)) = outs
  | [], [], _, _, _ => rfl
  | n :: ns, o :: outs, m, hnd, hl => by
      replace hnd := List.nodup_cons.mp hnd
      show writeAll (setBuf m n o) (ns.zip outs) n :: ns.map (writeAll (setBuf m n o) (ns.zip outs)) = o :: outs
      rw [writeAll_other n _ _ (fun h => hnd.1 (map_fst_zip_sub h)), setBuf, if_pos rfl]
      congr 1
      exact writeAll_zip_map ns outs (setBuf m n o) hnd.2 (by simpa using hl)
  | [], _ :: _, _, _, hl => by simp at hl
  | _ :: _, [], _, _, hl => by simp at hl

/-- **A launch performs what the machine says**: under the device's honesty
    for this launch, the bound buffers after it stand in `M` to what they held
    before. -/
theorem launchStep_honest {k : Launch → List ByteArray → List ByteArray} {l : Launch} {ns : List Nat}
    {M : List ByteArray → List ByteArray → Prop} (hk : KeepsSize k) (hM : Honest k l M) (hnd : ns.Nodup)
    (m : DevMem) : M (ns.map m) (ns.map ((launchStep k l ns).run m)) := by
  have hs := hk l (ns.map m)
  have hlen : (k l (ns.map m)).length = ns.length := by
    have := congrArg List.length hs; simpa using this
  have hkept : sizesKept (k l (ns.map m)) (ns.map m) = false := by
    simp only [sizesKept, Bool.or_eq_false_iff, bne_eq_false_iff_eq, List.any_eq_false]
    refine ⟨by simp [hlen], fun ⟨o, i⟩ hoi => ?_⟩
    simp only [bne_iff_ne, ne_eq, Decidable.not_not]
    exact zip_map_eq ByteArray.size _ _ hs o i hoi
  simp only [launchStep, hkept, Bool.false_eq_true, if_false]
  rw [writeAll_zip_map ns _ m hnd hlen]
  exact hM _

end AlgorithmLib.HProg.Contracts
