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

`launch_triple` is the host rule, and `launchNamed_triple`, `launchOnStream_triple`
and `launchNamedOnStream_triple` the same for the other launch entry points, all
through `launchOn_honest`. From a world where `cudaLaunch` answers, with
its kernel text and bindings known, every bound buffer live and bound once, and
the device honest about the launch it makes, the call answers and the bound
buffers after it stand in `M` to what they held before. A kernel theorem about
`M` then says what they hold (`Lz4.HostLaunch` for the shipped LZ4 kernel).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.DevSpec
  AlgorithmLib.HProg.Static AlgorithmLib.Device

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

/-- **A launch on any stream performs what the machine says**: where the launch
    of kernel text `kernel` at `entry` answers `0`, the buffers it binds stand in
    `M` to what they held, for a device honest about it. -/
theorem launchOn_honest {w : World} {p : Nat} {kernel entry : String} {nBufs bindPtr : UInt64}
    {dims : List UInt64} {ids : List Int} {r : Option V} {d : Dev} (M : List ByteArray → List ByteArray → Prop)
    (hk : KeepsSize w.kernel) (hcap : w.dev.capture = none)
    (hids : readIds w.mem bindPtr (asI32 nBufs).toNat = some ids)
    (hlive : ids.any (fun id => (w.dev.get? id).isNone) = false) (hnd : (ids.map Int.toNat).Nodup)
    (hM : Honest w.kernel ⟨kernel, entry, ids.map Int.toNat, dims, []⟩ M)
    (h : cudaLaunchOn w p kernel entry nBufs bindPtr dims = some (r, d)) :
    M ((ids.map Int.toNat).map (devMem w.dev)) ((ids.map Int.toNat).map (devMem d)) := by
  simp only [cudaLaunchOn, bind, Option.bind, hids, hlive, Bool.false_eq_true, if_false] at h
  split at h
  · cases h
  · rename_i d0 hd0
    change some (some (ofInt .i32 0), d0) = some (r, d) at h
    injection h with h1
    injection h1 with h2 h3
    subst h3
    rw [devOp_launch_mem hcap hd0]
    exact launchStep_honest hk hM hnd _

/-- **The launch triple.** From a world where `cudaLaunch` answers, with the
    kernel text and bindings it reads known, every bound buffer live and bound
    once, and the device honest about the launch it makes, the call answers,
    and the bound buffers after it stand in `M` to what they held before. -/
theorem launch_triple {w : World} {ctx kptr nBufs bindPtr gx gy gz bx by_ bz : UInt64} {text : String}
    {ids : List Int} (M : List ByteArray → List ByteArray → Prop)
    (hc : CtxOk w ctx) (hctx : ctx ≠ 0) (hk : KeepsSize w.kernel) (hcap : w.dev.capture = none)
    (hb : BindsReady w defaultParty nBufs bindPtr)
    (hdims : launchDimsBad nBufs [gx, gy, gz, bx, by_, bz] = false)
    (htext : readCStrAt w.mem kptr = some text) (hids : readIds w.mem bindPtr (asI32 nBufs).toNat = some ids)
    (hlive : ids.any (fun id => (w.dev.get? id).isNone) = false) (hnd : (ids.map Int.toNat).Nodup)
    (hM : Honest w.kernel ⟨text, "main", ids.map Int.toNat, [gx, gy, gz, bx, by_, bz], []⟩ M) :
    ∃ r w', callBits .cudaLaunch [ctx, kptr, nBufs, bindPtr, gx, gy, gz, bx, by_, bz] w = some (r, w') ∧
      M ((ids.map Int.toNat).map (devMem w.dev)) ((ids.map Int.toNat).map (devMem w'.dev)) := by
  obtain ⟨⟨r, w'⟩, h⟩ := Option.isSome_iff_exists.mp
    (launch_safe (gx := gx) (gy := gy) (gz := gz) (bx := bx) (by_ := by_) (bz := bz) hc hk hcap
      (by rw [htext]; rfl) hb)
  refine ⟨r, w', h, ?_⟩
  unfold ffiCudaLaunch at h
  have hne : (ctx != 0) = true := by simpa using hctx
  have hco := cudaCtxOk_of hc
  obtain ⟨d, hkd, rfl⟩ := devOnly_eq h
  clear h
  simp only [*, Bool.false_eq_true, if_false, Bool.not_true, bind, Option.bind] at hkd
  exact launchOn_honest M hk hcap hids hlive hnd hM hkd

/-- The same for a launch at an entry point named in memory. -/
theorem launchNamed_triple {w : World} {ctx kptr namePtr nBufs bindPtr gx gy gz bx by_ bz : UInt64}
    {text entry : String} {ids : List Int} (M : List ByteArray → List ByteArray → Prop)
    (hc : CtxOk w ctx) (hctx : ctx ≠ 0) (hk : KeepsSize w.kernel) (hcap : w.dev.capture = none)
    (hb : BindsReady w defaultParty nBufs bindPtr)
    (hdims : launchDimsBad nBufs [gx, gy, gz, bx, by_, bz] = false)
    (htext : readCStrAt w.mem kptr = some text) (hentry : readCStrAt w.mem namePtr = some entry)
    (hids : readIds w.mem bindPtr (asI32 nBufs).toNat = some ids)
    (hlive : ids.any (fun id => (w.dev.get? id).isNone) = false) (hnd : (ids.map Int.toNat).Nodup)
    (hM : Honest w.kernel ⟨text, entry, ids.map Int.toNat, [gx, gy, gz, bx, by_, bz], []⟩ M) :
    ∃ r w', callBits .cudaLaunchNamed [ctx, kptr, namePtr, nBufs, bindPtr, gx, gy, gz, bx, by_, bz] w
        = some (r, w') ∧
      M ((ids.map Int.toNat).map (devMem w.dev)) ((ids.map Int.toNat).map (devMem w'.dev)) := by
  obtain ⟨⟨r, w'⟩, h⟩ := Option.isSome_iff_exists.mp
    (launchNamed_safe (gx := gx) (gy := gy) (gz := gz) (bx := bx) (by_ := by_) (bz := bz) hc hk hcap
      (by rw [htext]; rfl) (by rw [hentry]; rfl) hb)
  refine ⟨r, w', h, ?_⟩
  unfold ffiCudaLaunchNamed at h
  have hne : (ctx != 0) = true := by simpa using hctx
  have hco := cudaCtxOk_of hc
  obtain ⟨d, hkd, rfl⟩ := devOnly_eq h
  clear h
  simp only [*, Bool.false_eq_true, if_false, Bool.not_true, bind, Option.bind] at hkd
  exact launchOn_honest M hk hcap hids hlive hnd hM hkd

/-- The same for a launch on a stream `sid` names. -/
theorem launchOnStream_triple {w : World} {ctx kptr nBufs bindPtr gx gy gz bx by_ bz sid : UInt64}
    {text : String} {ids : List Int} {p : Nat} (M : List ByteArray → List ByteArray → Prop)
    (hc : CtxOk w ctx) (hctx : ctx ≠ 0) (hk : KeepsSize w.kernel) (hcap : w.dev.capture = none)
    (hp : w.dev.party? (asI32 sid) = some p) (hb : BindsReady w p nBufs bindPtr)
    (hdims : launchDimsBad nBufs [gx, gy, gz, bx, by_, bz] = false)
    (htext : readCStrAt w.mem kptr = some text) (hids : readIds w.mem bindPtr (asI32 nBufs).toNat = some ids)
    (hlive : ids.any (fun id => (w.dev.get? id).isNone) = false) (hnd : (ids.map Int.toNat).Nodup)
    (hM : Honest w.kernel ⟨text, "main", ids.map Int.toNat, [gx, gy, gz, bx, by_, bz], []⟩ M) :
    ∃ r w', callBits .cudaLaunchOnStream [ctx, kptr, nBufs, bindPtr, gx, gy, gz, bx, by_, bz, sid] w
        = some (r, w') ∧
      M ((ids.map Int.toNat).map (devMem w.dev)) ((ids.map Int.toNat).map (devMem w'.dev)) := by
  obtain ⟨⟨r, w'⟩, h⟩ := Option.isSome_iff_exists.mp
    (launchOnStream_safe (gx := gx) (gy := gy) (gz := gz) (bx := bx) (by_ := by_) (bz := bz) hc hk hcap
      (by rw [htext]; rfl) (fun q hq => by rw [hp] at hq; cases hq; exact hb))
  refine ⟨r, w', h, ?_⟩
  unfold ffiCudaLaunchOnStream at h
  have hne : (ctx != 0) = true := by simpa using hctx
  have hco := cudaCtxOk_of hc
  obtain ⟨d, hkd, rfl⟩ := devOnly_eq h
  clear h
  simp only [*, Bool.false_eq_true, if_false, Bool.not_true, bind, Option.bind] at hkd
  exact launchOn_honest M hk hcap hids hlive hnd hM hkd

/-- The same for a named entry point on a stream. -/
theorem launchNamedOnStream_triple {w : World}
    {ctx kptr namePtr nBufs bindPtr gx gy gz bx by_ bz sid : UInt64}
    {text entry : String} {ids : List Int} {p : Nat} (M : List ByteArray → List ByteArray → Prop)
    (hc : CtxOk w ctx) (hctx : ctx ≠ 0) (hk : KeepsSize w.kernel) (hcap : w.dev.capture = none)
    (hp : w.dev.party? (asI32 sid) = some p) (hb : BindsReady w p nBufs bindPtr)
    (hdims : launchDimsBad nBufs [gx, gy, gz, bx, by_, bz] = false)
    (htext : readCStrAt w.mem kptr = some text) (hentry : readCStrAt w.mem namePtr = some entry)
    (hids : readIds w.mem bindPtr (asI32 nBufs).toNat = some ids)
    (hlive : ids.any (fun id => (w.dev.get? id).isNone) = false) (hnd : (ids.map Int.toNat).Nodup)
    (hM : Honest w.kernel ⟨text, entry, ids.map Int.toNat, [gx, gy, gz, bx, by_, bz], []⟩ M) :
    ∃ r w', callBits .cudaLaunchNamedOnStream
        [ctx, kptr, namePtr, nBufs, bindPtr, gx, gy, gz, bx, by_, bz, sid] w = some (r, w') ∧
      M ((ids.map Int.toNat).map (devMem w.dev)) ((ids.map Int.toNat).map (devMem w'.dev)) := by
  obtain ⟨⟨r, w'⟩, h⟩ := Option.isSome_iff_exists.mp
    (launchNamedOnStream_safe (gx := gx) (gy := gy) (gz := gz) (bx := bx) (by_ := by_) (bz := bz) hc hk hcap
      (by rw [htext]; rfl) (by rw [hentry]; rfl) (fun q hq => by rw [hp] at hq; cases hq; exact hb))
  refine ⟨r, w', h, ?_⟩
  unfold ffiCudaLaunchNamedOnStream at h
  have hne : (ctx != 0) = true := by simpa using hctx
  have hco := cudaCtxOk_of hc
  obtain ⟨d, hkd, rfl⟩ := devOnly_eq h
  clear h
  simp only [*, Bool.false_eq_true, if_false, Bool.not_true, bind, Option.bind] at hkd
  exact launchOn_honest M hk hcap hids hlive hnd hM hkd

end AlgorithmLib.HProg.Contracts
