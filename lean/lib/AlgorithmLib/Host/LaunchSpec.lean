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
reading, as the models of the C libraries (`Ext`) are the host-side one.

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

/-- What launch `l`'s bound buffers `ns` hold after it, from what they held
    (`ins`) and the device's answers (`outs`): the answer where the launch
    writes the buffer, what it held where the launch only reads it. -/
def _root_.AlgorithmLib.HProg.Sem.Launch.after (l : Launch) : List Nat → List ByteArray → List ByteArray → List ByteArray
  | b :: ns, i :: ins, o :: outs => (if l.writes.contains b then o else i) :: l.after ns ins outs
  | _, _, outs => outs

/-- A launch that writes every buffer it binds leaves the device's answers. -/
theorem _root_.AlgorithmLib.HProg.Sem.Launch.after_all (l : Launch) (ns : List Nat) :
    ∀ (ins outs : List ByteArray), (∀ b ∈ ns, b ∈ l.writes) → l.after ns ins outs = outs := by
  induction ns with
  | nil => intro ins outs _; cases ins <;> cases outs <;> rfl
  | cons b ns ih =>
    intro ins outs hw
    cases ins with
    | nil => cases outs <;> rfl
    | cons i ins =>
      cases outs with
      | nil => rfl
      | cons o outs =>
        have hb : l.writes.contains b = true := by simpa using hw b List.mem_cons_self
        simp only [Launch.after, hb, if_true]
        rw [ih ins outs (fun c hc => hw c (List.mem_cons_of_mem _ hc))]

/-- **The GPU-side reading, named once.** What the device leaves in launch
    `l`'s bound buffers — its answer in each one the launch writes, the rest
    as they were — the modeled machine relates to the launch's input by `M`. -/
def Honest (k : Launch → List ByteArray → List ByteArray) (l : Launch)
    (M : List ByteArray → List ByteArray → Prop) : Prop :=
  ∀ ins, M ins (l.after l.binds ins (k l ins))

theorem devOp_launch_mem {d d' : Dev} {w : World} {p : Nat} {l : Launch} {ns : List Nat}
    (hc : d.capture = none) (h : d.devOp w p (.launch l ns) ns l.writes = some d') :
    devMem d' = (launchStep w.kernel l ns).run (devMem d) := by
  unfold Dev.devOp at h
  rw [hc] at h
  unfold Dev.devOp.run at h
  obtain ⟨r, -, h⟩ := Option.bind_eq_some_iff.mp h
  dsimp only at h
  have := runLaunch_mem _ _ _ _ _ h
  exact this

theorem devOp_vendor_mem {d d' : Dev} {w : World} {p : Nat} {c : VendorCall} {ins rs ws : List Nat} {out : Nat}
    (hc : d.capture = none) (h : d.devOp w p (.vendor c ins out) rs ws = some d') :
    devMem d' = (vendorStep w.vendor c ins out).run (devMem d) := by
  unfold Dev.devOp at h
  rw [hc] at h
  unfold Dev.devOp.run at h
  obtain ⟨r, -, h⟩ := Option.bind_eq_some_iff.mp h
  dsimp only at h
  have := runVendor_mem _ _ _ _ _ _ h
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

theorem writeAll_stores (l : Launch) : ∀ (ns : List Nat) (outs : List ByteArray) (m : DevMem),
    ns.Nodup → outs.length = ns.length → ns.map (writeAll m (l.stores ns outs)) = l.after ns (ns.map m) outs
  | [], [], _, _, _ => rfl
  | n :: ns, o :: outs, m, hnd, hl => by
      replace hnd := List.nodup_cons.mp hnd
      have hl' : outs.length = ns.length := by simpa using hl
      have hrest : ∀ m' : DevMem, n ∉ (l.stores ns outs).map Prod.fst := fun _ h =>
        hnd.1 (List.mem_filter.mp (Launch.stores_fst h)).1
      have hmap : ∀ x, ns.map (setBuf m n x) = ns.map m := fun x =>
        List.map_congr_left fun b hb => by
          have hne : b ≠ n := fun e => hnd.1 (by rw [← e]; exact hb)
          simp [setBuf, hne, Ne.symm hne]
      by_cases hw : l.writes.contains n
      · have hw' : n ∈ l.writes := by simpa using hw
        have e : l.stores (n :: ns) (o :: outs) = (n, o) :: l.stores ns outs := by
          simp [Launch.stores, List.filter_cons, hw']
        rw [e]
        show writeAll (setBuf m n o) (l.stores ns outs) n :: ns.map (writeAll (setBuf m n o) (l.stores ns outs)) =
          l.after (n :: ns) (m n :: ns.map m) (o :: outs)
        rw [writeAll_other n _ _ (hrest m), writeAll_stores l ns outs (setBuf m n o) hnd.2 hl', hmap]
        simp [Launch.after, hw', setBuf]
      · have hw' : n ∉ l.writes := by simpa using hw
        have e : l.stores (n :: ns) (o :: outs) = l.stores ns outs := by
          simp [Launch.stores, List.filter_cons, hw']
        rw [e]
        show writeAll m (l.stores ns outs) n :: ns.map (writeAll m (l.stores ns outs)) =
          l.after (n :: ns) (m n :: ns.map m) (o :: outs)
        rw [writeAll_other n _ _ (hrest m), writeAll_stores l ns outs m hnd.2 hl']
        simp [Launch.after, hw']
  | [], _ :: _, _, _, hl => by simp at hl
  | _ :: _, [], _, _, hl => by simp at hl

/-- **A launch performs what the machine says**: under the device's honesty
    for this launch, the bound buffers after it stand in `M` to what they held
    before. -/
theorem launchStep_honest {k : Launch → List ByteArray → List ByteArray} {l : Launch} {ns : List Nat}
    {M : List ByteArray → List ByteArray → Prop} (hk : KeepsSize k) (hM : Honest k l M) (hnd : ns.Nodup)
    (hb : l.binds = ns) (m : DevMem) : M (ns.map m) (ns.map ((launchStep k l ns).run m)) := by
  have hs := hk l (ns.map m)
  have hlen : (k l (ns.map m)).length = ns.length := by
    have := congrArg List.length hs; simpa using this
  have hkept : sizesKept (k l (ns.map m)) (ns.map m) = false := by
    simp only [sizesKept, Bool.or_eq_false_iff, bne_eq_false_iff_eq, List.any_eq_false]
    refine ⟨by simp [hlen], fun ⟨o, i⟩ hoi => ?_⟩
    simp only [bne_iff_ne, ne_eq, Decidable.not_not]
    exact zip_map_eq ByteArray.size _ _ hs o i hoi
  simp only [launchStep, hkept, Bool.false_eq_true, if_false]
  rw [writeAll_stores l ns _ m hnd hlen, ← hb]
  exact hM _

end AlgorithmLib.HProg.Contracts
