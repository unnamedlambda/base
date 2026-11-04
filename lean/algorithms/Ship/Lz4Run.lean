import AlgorithmLib.MLHost.TapeRun
import AlgorithmLib.MLHost.AppGraph
import Ship.Lz4Comp
import Lz4.HostLaunch
import AlgorithmLib.Proof.LibSpec

/-!
# What one compression does to the caller's bytes

Every run of the emitted compression that ends (`warpCode`, both shipped
geometries) passes its calls into the driver in the order `warpCalls` admits
(`warp15_calls`), so device memory moves only by them. The upload puts the
bytes it reads from the caller's data at the start of the allocation; each of
the `rLaunches` launches, on a device honest about the kernel it loaded
(`LaunchHonest`), leaves what the warp's runs denote (`Lz4Code.lz4Machine`);
the download copies the blocks' outputs into the caller's output. The checker
shows no host store touches the caller's data or its output's blocks
(`keptRanges`), and between calls host memory changes only where stores were
observed (`MemLogged`), so the upload reads the caller's data as handed over
and the output holds, at the end, what the download wrote: every block the
caller gets back decodes to its input (`warp_run`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Contracts
  AlgorithmLib.LibSpec AlgorithmLib.Device AlgorithmLib.Kernel AlgorithmLib.Check AlgorithmLib.ML

namespace Lz4CompSafe

/-- **Every block of `o` decodes to the block of `i`** at geometry `b`: block
    `c`'s output holds its length `n ≤ lenOff` at `lenOff`, and its first `n`
    bytes decode to block `c` of `i`'s inputs. -/
def Decodes (b : Nat) (i o : ByteArray) : Prop :=
  ∀ c, c < (Lz4Ship.WP.mk b).numBlk → ∃ n, n ≤ (Lz4Ship.WP.mk b).lenOff ∧
    Lz4Code.u32At o ((Lz4Ship.WP.mk b).numBlk * (Lz4Ship.WP.mk b).inStride + LZ4WarpDSL.copySlack +
      c * (Lz4Ship.WP.mk b).outStride + (Lz4Ship.WP.mk b).lenOff) = n ∧
    LZ4Imp.decompress ((List.range n).map fun j =>
        o.get! ((Lz4Ship.WP.mk b).numBlk * (Lz4Ship.WP.mk b).inStride + LZ4WarpDSL.copySlack +
          c * (Lz4Ship.WP.mk b).outStride + j)) (Lz4Ship.WP.mk b).inStride
      = some ((List.range (Lz4Ship.WP.mk b).inStride).map fun j => i.get! (c * (Lz4Ship.WP.mk b).inStride + j))

/-- The application the compression is laid out as. -/
abbrev app (b : Nat) : Drv.App := Lz4Ship.lzAppAt (Lz4Ship.WP.mk b) 17680

/-- **The device is honest about the compressor's launch**: launching
    function 0 of the layout, from whatever module text and entry it was
    loaded, leaves the allocation it binds as `Lz4Code.lz4Machine` says. -/
def LaunchHonest (b : Nat) (w : World) : Prop :=
  ∀ k mi entry ptx, (app b).kernels[0]? = some k → w.dev.funcs[0]? = some (mi, entry) →
    (w.dev.modules[mi]?).join = some ptx →
    Honest w.kernel ⟨ptx, entry, DrvLaunch.bindsOf w.dev (recLaunchArgs k), recDims k, recLaunchArgs k,
      w.dev.writesOf (w.dev.kernelWrites ptx entry) (recLaunchArgs k)⟩ (Lz4Code.lz4Machine b)

theorem app_kernel (b : Nat) : ∃ k, (app b).kernels[0]? = some k ∧ recLaunchArgs k = [devPtr 0, devPtr 0] :=
  ⟨_, rfl, rfl⟩

/-- **Launches of the compressor, each on an honest device, keep the inputs,
    the size and, after any, every block decoding to its input.** -/
theorem launches_decode (b : Nat) (hb : b = 15 ∨ b = 16) {w0 : World} (hks : KeepsSize w0.kernel)
    (hH : LaunchHonest b w0) (hbind : DrvLaunch.bindsOf w0.dev [devPtr 0, devPtr 0] = [0]) :
    ∀ {ss : List DevCall} {m m' : DevMem}, DevCall.Steps (app b) w0 0 0 ss m m' →
      (∀ s ∈ ss, s = .kernel 0) → (Lz4Ship.WP.mk b).bytes ≤ (m 0).size →
      (∀ j, j < (Lz4Ship.WP.mk b).numBlk * (Lz4Ship.WP.mk b).inStride + LZ4WarpDSL.copySlack →
        (m' 0).get! j = (m 0).get! j) ∧ (Lz4Ship.WP.mk b).bytes ≤ (m' 0).size ∧
      (ss ≠ [] → Decodes b (m 0) (m' 0)) := by
  intro ss m m' hS
  induction hS with
  | nil m => exact fun _ h => ⟨fun _ _ => rfl, h, fun h => absurd rfl h⟩
  | @snoc ss s m m1 m2 _ hst ih =>
      intro hall hsz
      obtain ⟨hin, hsz1, -⟩ := ih (fun s hs => hall s (List.mem_append_left _ hs)) hsz
      have hs0 : s = .kernel 0 := hall s (List.mem_append_right _ (List.mem_singleton_self _))
      subst hs0
      obtain ⟨k, mi, entry, ptx, hk, hf, hmod, hm2⟩ := hst
      obtain ⟨k', hk', hargs⟩ := app_kernel b
      rw [hk] at hk'; cases hk'
      have hM := hH k mi entry ptx hk hf hmod
      rw [hargs, hbind] at hM
      have hD := launchStep_honest hks hM (by simp) rfl m1
      rw [hargs, hbind] at hm2
      rw [← hm2] at hD
      simp only [List.map_cons, List.map_nil, Lz4Code.lz4Machine] at hD
      obtain ⟨g1, g2, g3, g4, g5, g6, g7⟩ := Lz4Code.shipped_geometry b hb
      have hdec := Lz4Code.decode_of_denotes g1 g2 g3 g4 g5 g6 g7 hsz1 hD
      refine ⟨fun j hj => (hD.1 j hj).trans (hin j hj), hD.2.1, fun _ c hc => ?_⟩
      obtain ⟨n, hn, hlen, hd⟩ := hdec c hc
      have hc' : c * (Lz4Ship.WP.mk b).inStride + (Lz4Ship.WP.mk b).inStride ≤
          (Lz4Ship.WP.mk b).numBlk * (Lz4Ship.WP.mk b).inStride := by
        have := Nat.mul_le_mul_right (Lz4Ship.WP.mk b).inStride (show c + 1 ≤ (Lz4Ship.WP.mk b).numBlk by omega)
        rw [Nat.succ_mul] at this; exact this
      have hmap : ((List.range (Lz4Ship.WP.mk b).inStride).map fun j =>
            (m1 0).get! (c * (Lz4Ship.WP.mk b).inStride + j)) =
          (List.range (Lz4Ship.WP.mk b).inStride).map fun j => (m 0).get! (c * (Lz4Ship.WP.mk b).inStride + j) :=
        List.map_congr_left fun j hj => hin _ (by have := List.mem_range.mp hj; omega)
      exact ⟨n, hn, hlen, hd.trans (congrArg some hmap)⟩

/-- The upload `warpCalls` admits. -/
def htodRec (b : Nat) : CallRec :=
  ⟨.ext (.cuda .memcpyHtoD), [.is (devPtr 0), .is (regionBase .data), .within 0 (Lz4Ship.WP.mk b).outOff]⟩

/-- The download `warpCalls` admits. -/
def dtohRec (b : Nat) : CallRec :=
  ⟨.ext (.cuda .memcpyDtoH), [.is (regionBase .out), .is (devPtr 0 + UInt64.ofNat (Lz4Ship.WP.mk b).outOff),
    .is (UInt64.ofNat (Lz4Ship.WP.mk b).totOut)]⟩

/-- **The calls one compression makes, as the layout's records**: an upload
    or none, `rLaunches` launches of kernel 0, a download or none. -/
theorem warpCalls_eq (b : Nat) (hb : b = 15 ∨ b = 16) : warpCalls (Lz4Ship.WP.mk b) 17680 =
    .seq (.alt (.lit [htodRec b]) (.lit []))
      (.seq (.rep Lz4Ship.rLaunches (.lit [(app b).launchRec 0])) (.alt (.lit [dtohRec b]) (.lit []))) := by
  rcases hb with rfl | rfl <;> rfl

/-- The numbers of the shipped geometries the run's calls rest on. -/
theorem run_geometry (b : Nat) (hb : b = 15 ∨ b = 16) :
    devPtrOf (devPtr 0) = some (0, 0) ∧
    devPtrOf (devPtr 0 + UInt64.ofNat (Lz4Ship.WP.mk b).outOff) = some (0, (Lz4Ship.WP.mk b).outOff) ∧
    (UInt64.ofNat (Lz4Ship.WP.mk b).totOut).toNat = (Lz4Ship.WP.mk b).totOut ∧
    (app b).bufs.getD 0 0 = (Lz4Ship.WP.mk b).bytes ∧ 0 < (app b).bufs.length ∧
    DevCall.okB (app b) (.kernel 0) = true ∧ 0 < Lz4Ship.rLaunches := by
  rcases hb with rfl | rfl <;> decide

/-- A run's calls `warpCalls` admits, cut where the launches begin and end. -/
theorem warp_has {b : Nat} (hb : b = 15 ∨ b = 16) {cs : List (Callee × List V)}
    (h : (warpCalls (Lz4Ship.WP.mk b) 17680).Has cs) :
    ∃ su sl sd, cs = su ++ (sl ++ sd) ∧ (su = [] ∨ ∃ x, su = [x] ∧ (htodRec b).Matches x) ∧
      recsMatch (List.replicate Lz4Ship.rLaunches ((app b).launchRec 0)) sl ∧
      (sd = [] ∨ ∃ x, sd = [x] ∧ (dtohRec b).Matches x) := by
  rw [warpCalls_eq b hb] at h
  cases h with
  | seq hu hld =>
      cases hld with
      | seq hl hd => exact ⟨_, _, _, rfl, hu.lit_or_nil, hl.rep_lit, hd.lit_or_nil⟩

theorem warp_has_ext {b : Nat} (hb : b = 15 ∨ b = 16) (cs : List (Callee × List V))
    (h : (warpCalls (Lz4Ship.WP.mk b) 17680).Has cs) : ∀ x ∈ cs, ∃ e, x.1 = .ext e := by
  obtain ⟨su, sl, sd, rfl, hu, hl, hd⟩ := warp_has hb h
  intro x hx
  rcases List.mem_append.mp hx with hx | hx
  · rcases hu with rfl | ⟨y, rfl, hy⟩
    · cases hx
    · rw [List.mem_singleton.mp hx]; exact ⟨_, hy.1.symm⟩
  rcases List.mem_append.mp hx with hx | hx
  · obtain ⟨r, hr, e⟩ := recsMatch_callees hl x hx
    rw [List.eq_of_mem_replicate hr] at e
    exact ⟨_, e.symm⟩
  · rcases hd with rfl | ⟨y, rfl, hy⟩
    · cases hx
    · rw [List.mem_singleton.mp hx]; exact ⟨_, hy.1.symm⟩

/-- **What one compression does to the caller's bytes.** Every run of the
    emitted compression that ends, from what `load` leaves, on a device honest
    about the kernel it loaded (`LaunchHonest`), makes calls `cs` after those
    observed before it, such that: allocation 0, as the launches find it
    (`B`), is what it held before the run, or that with bytes of the caller's
    data, as the caller handed it over, written over its start; every block of
    allocation 0 at the end decodes to the block of `B`; and where `cs`
    downloads, the caller's output holds, at the end, allocation 0's outputs
    byte for byte. -/
theorem warp_run (b : Nat) (hb : b = 15 ∨ b = 16) (cfg : Cfg) (dl ol : UInt64) (w : World)
    (h : Run b dl ol w) {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dl ol) w (emit (Lz4Ship.warpCode (Lz4Ship.WP.mk b) : Prog Slot Lvl Unit))
      = .ok obs w')
    (hH : LaunchHonest b w) :
    ∃ cs B, callsOf obs = (callsOf w.obs).reverse ++ cs ∧
      (B = devMem w.dev 0 ∨ ∃ bytes : ByteArray, bytes.size ≤ (Lz4Ship.WP.mk b).outOff ∧
        readBytes w.mem (regionBase .data) bytes.size = some bytes ∧ B = overwrite (devMem w.dev 0) 0 bytes) ∧
      Decodes b B (devMem w'.dev 0) ∧
      ∀ x ∈ cs, x.1 = .ext (.cuda .memcpyDtoH) → ∀ j < (Lz4Ship.WP.mk b).totOut,
        w'.mem.load (regionBase .out + UInt64.ofNat j) 1 =
          some ((devMem w'.dev 0).get! ((Lz4Ship.WP.mk b).outOff + j)).toUInt64 := by
  have hcalls : (warpCalls (Lz4Ship.WP.mk b) 17680).Covers w obs ∧ DevPath w w' ∧
      StoresOut (keptRanges (Lz4Ship.WP.mk b)) w w' := by
    rcases hb with rfl | rfl
    · exact warp15_calls cfg dl ol w h hr
    · exact warp16_calls cfg dl ol w h hr
  obtain ⟨dT, eT, kT⟩ : ∃ dT, w'.obs = dT ++ w.obs ∧ ∀ b' n v, Obs.store b' n v ∈ dT →
      ∀ i < n, Outside (keptRanges (Lz4Ship.WP.mk b)) (b' + UInt64.ofNat i).toNat := by
    rcases hcalls.2.2 with h0 | h0
    · simp [keptRanges] at h0
    · exact h0
  obtain ⟨cs, hcs, hh, hx⟩ := extRun_of hr ⟨hcalls.1, hcalls.2.1⟩ (warp_has_ext hb)
  obtain ⟨su, sl, sd, rfl, hu, hl, hd⟩ := warp_has hb hh
  obtain ⟨gp0, gpo, gtot, gsz, glen, gok, gn⟩ := run_geometry b hb
  obtain ⟨gout, gdat, gtot2⟩ : (Lz4Ship.WP.mk b).outOff + (Lz4Ship.WP.mk b).totOut ≤ (Lz4Ship.WP.mk b).bytes ∧
      (Lz4Ship.WP.mk b).outOff < 2 ^ 36 ∧ (Lz4Ship.WP.mk b).totOut < 2 ^ 36 := by
    rcases hb with rfl | rfl <;> decide
  obtain ⟨hcu, hcap, hks, -, halloc⟩ := runTS_ready h
  obtain ⟨B0, hB0, hsz0⟩ := halloc 0 glen
  have hm0 : devMem w.dev 0 = B0 := by simp [devMem, hB0]
  have hran : ∀ d : Dev, (d.bufs[0]?).join = some (devMem d 0) → d.range? (devPtr 0) 0 = some (0, 0, devMem d 0) :=
    fun d hd => by simp [Dev.range?, gp0, hd]
  have hbinds : ∀ d : Dev, (d.bufs[0]?).join = some (devMem d 0) →
      DrvLaunch.bindsOf d [devPtr 0, devPtr 0] = [0] := fun d hd => by
    simp [DrvLaunch.bindsOf, hran d hd, List.eraseDups_cons]
  have hhas : ∀ (w1 : World) (f : CudaFn), w1.present = w.present → w1.has (Ext.cuda f).lib = true :=
    fun w1 f hp => by unfold World.has at hcu ⊢; rw [hp]; exact hcu
  obtain ⟨w1, hx1, hx23⟩ := hx.split su (sl ++ sd) rfl
  obtain ⟨w2, hx2, hx3⟩ := hx23.split sl sd rfl
  have hp0 : ((regionBase .data).toNat, (regionBase .data).toNat + 2 ^ 36) ∈ keptRanges (Lz4Ship.WP.mk b) := by
    simp [keptRanges]
  have hp1 : ((regionBase .out).toNat, (regionBase .out).toNat + (Lz4Ship.WP.mk b).totOut) ∈
      keptRanges (Lz4Ship.WP.mk b) := by simp [keptRanges]
  -- the upload
  obtain ⟨B, hB, hm1, hsame1, hfix1⟩ : ∃ B, (B = devMem w.dev 0 ∨ ∃ bytes : ByteArray,
      bytes.size ≤ (Lz4Ship.WP.mk b).outOff ∧ readBytes w.mem (regionBase .data) bytes.size = some bytes ∧
      B = overwrite (devMem w.dev 0) 0 bytes) ∧ devMem w1.dev 0 = B ∧ DevSame w.dev w1.dev ∧ Fixed w w1 := by
    rcases hu with rfl | ⟨x, rfl, hmx⟩
    · obtain ⟨hd1, hf1, -⟩ := hx1.nil_inv
      exact ⟨_, .inl rfl, by rw [hd1], by rw [hd1]; exact DevSame.refl _, hf1⟩
    · obtain ⟨wa, wa', e, vs, r, hda, hfa, rfl, hext, hd1, hf1, hma, hma', hoa⟩ := hx1.one_inv
      have he : Callee.ext e = .ext (.cuda .memcpyHtoD) := hmx.1.symm
      injection he with he
      subst he
      obtain ⟨id, off, B', bytes, hrg, -, hle, hrd, hmem, hkeep⟩ :=
        htod_done (hhas wa _ hfa.2.2) (by rw [hda]; exact hcap) hmx hext
      obtain ⟨hp, hB'⟩ := range_eq hrg
      rw [gp0] at hp
      cases hp
      rw [hda] at hB'
      subst hB'
      -- the caller's data is as it was when the upload reads it
      have hsub := stretch_sub (wA := w) (wB := wa) eT
        (by
          obtain ⟨d1, e1, -⟩ := hma'
          obtain ⟨d2, e2⟩ := hx23.obs
          refine ⟨d2 ++ d1 ++ [Obs.call (.ext (.cuda .memcpyHtoD)) (vs ++ callWords wa (.ext (.cuda .memcpyHtoD)) vs)], ?_⟩
          rw [e2, e1, hoa]; simp)
        ⟨[], rfl⟩
      have hrd' : readBytes w.mem (regionBase .data) bytes.size = some bytes := by
        rw [← hrd]
        refine (copyOut_congr fun i hi => ?_).symm
        refine kept_load kT hp0 ?_ hma hsub
        rw [addr_toNat .data (show i ≤ 2 ^ 36 by omega)]
        omega
      refine ⟨_, .inr ⟨bytes, hle, hrd', rfl⟩, ?_, ?_, hfa.trans (hkeep.2.trans hf1)⟩
      · rw [hd1, hmem, hda]; simp [setBuf]
      · rw [hd1, ← hda]; exact hkeep.1
  -- the launches
  have hks1 : KeepsSize w1.kernel := by rw [hfix1.1]; exact hks
  have hl1 : (w1.dev.bufs[0]?).join = some (devMem w1.dev 0) := by
    have := hsame1.sizes 0
    simp only [Dev.get?] at this
    cases e1 : (w1.dev.bufs[0]?).join with
    | none => simp [e1, hB0] at this
    | some c => simp [devMem, e1]
  have hH1 : LaunchHonest b w1 := fun k mi entry ptx hk hf hmod => by
    rw [hsame1.funcs] at hf
    rw [hsame1.modules] at hmod
    rw [hfix1.1, bindsOf_same hsame1, writesOf_same hsame1, hsame1.kernelWrites]
    exact hH k mi entry ptx hk hf hmod
  have hrecs : recsMatch ((List.replicate Lz4Ship.rLaunches (DevCall.kernel 0)).map
      (DevCall.record (app b) 0 0 0)) sl := by
    rw [List.map_replicate]; exact hl
  obtain ⟨hst, hsame2, hfix2⟩ := extRun_steps (hhas w1 _ hfix1.2.2) (by rw [hsame1.capture]; exact hcap) hks1 hx2
    _ hrecs (fun s hs => by rw [List.eq_of_mem_replicate hs]; exact DevCall.okB_ok gok)
    (fun g hg => by cases List.eq_of_mem_replicate hg)
  have hsz1 : (Lz4Ship.WP.mk b).bytes ≤ (devMem w1.dev 0).size := by
    have := hsame1.sizes 0
    have e1 : w1.dev.get? 0 = some (devMem w1.dev 0) := by simp [Dev.get?, hl1]
    have e0 : w.dev.get? 0 = some B0 := by simp [Dev.get?, hB0]
    rw [e1, e0] at this
    simp only [Option.map_some, Option.some.injEq] at this
    have : (app b).bufs.getD 0 0 = B0.size := hsz0.symm
    omega
  obtain ⟨-, hsz2, hdec⟩ := launches_decode b hb hks1 hH1 (hbinds _ hl1) hst
    (fun s hs => List.eq_of_mem_replicate hs) hsz1
  have hdec' := hdec (by simp; omega)
  rw [hm1] at hdec'
  -- the download
  have hfin : devMem w'.dev 0 = devMem w2.dev 0 ∧ (sd = [] ∨ ∀ j < (Lz4Ship.WP.mk b).totOut,
      w'.mem.load (regionBase .out + UInt64.ofNat j) 1 =
        some ((devMem w2.dev 0).get! ((Lz4Ship.WP.mk b).outOff + j)).toUInt64) := by
    rcases hd with rfl | ⟨x, rfl, hmx⟩
    · exact ⟨by rw [hx3.nil_inv.1], .inl rfl⟩
    · obtain ⟨wd, wd', e, vs, r, hdd, hfd, rfl, hext, hd', -, hmd, hmd', hod⟩ := hx3.one_inv
      have he : Callee.ext e = .ext (.cuda .memcpyDtoH) := hmx.1.symm
      injection he with he
      subst he
      have hcapd : wd.dev.capture = none := by rw [hdd, hsame2.capture, hsame1.capture]; exact hcap
      have hfd' : wd.present = w.present := hfd.2.2.trans (hfix2.2.2.trans hfix1.2.2)
      obtain ⟨id, off, B', hrg, hci, hmem⟩ := dtoh_done (hhas wd _ hfd') hcapd hmx hext
      obtain ⟨hp, hB'⟩ := range_eq hrg
      rw [gpo] at hp
      cases hp
      rw [hdd] at hB'
      subst hB'
      rw [gtot] at hci
      refine ⟨by rw [hd', hmem, hdd], .inr fun j hj => ?_⟩
      -- the caller's output is as the download left it
      have hsub := stretch_sub (wA := wd') (wB := w') eT ⟨[], rfl⟩
        (by
          obtain ⟨d1, e1⟩ := hx1.obs
          obtain ⟨d2, e2⟩ := hx2.obs
          obtain ⟨d3, e3, -⟩ := hmd
          refine ⟨[Obs.call (.ext (.cuda .memcpyDtoH)) (vs ++ callWords wd (.ext (.cuda .memcpyDtoH)) vs)] ++ d3 ++ d2 ++
            d1, ?_⟩
          rw [hod, e3, e2, e1]; simp)
      have hsrc : ((devMem w2.dev 0).extract (Lz4Ship.WP.mk b).outOff
          ((Lz4Ship.WP.mk b).outOff + (Lz4Ship.WP.mk b).totOut)).size = (Lz4Ship.WP.mk b).totOut := by
        simp [ByteArray.size_extract]; omega
      rw [kept_load kT hp1 (by rw [addr_toNat .out (show j ≤ 2 ^ 36 by omega)]; omega) hmd' hsub,
        copyIn_load hci (by omega) j (by omega), extract_get! (by omega) (by omega)]
  obtain ⟨hfin1, hfin2⟩ := hfin
  refine ⟨su ++ (sl ++ sd), B, hcs, hB, by rw [hfin1]; exact hdec', fun x hx he => ?_⟩
  rw [hfin1]
  rcases List.mem_append.mp hx with hx | hx
  · rcases hu with rfl | ⟨y, rfl, hy⟩
    · cases hx
    · rw [List.mem_singleton.mp hx] at he; rw [← hy.1] at he; cases he
  rcases List.mem_append.mp hx with hx | hx
  · obtain ⟨r, hr', e⟩ := recsMatch_callees hl x hx
    rw [List.eq_of_mem_replicate hr', he] at e
    have hc : ((app b).launchRec 0).callee = .ext (.cuda .launchKernel) := rfl
    rw [hc] at e
    cases e
  · rcases hfin2 with rfl | hw
    · cases hx
    · exact hw

end Lz4CompSafe
