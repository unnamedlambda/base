import Ship.VitSafe
import AlgorithmLib.ML.Launch.AppGraph
import AlgorithmLib.Proof.LibSpec

/-!
# What the ViT's `fetch` and `seed` do to the caller's bytes

`fetch` copies the logits buffer into the caller's output: every run of it
that ends leaves the output's first 102400 bytes as the buffer held them
(`fetch_out`). `seed` copies the caller's data into the buffer the backward
starts from: every run of it that ends leaves that buffer's first 102400
bytes as the caller's data held them when it was handed over (`seed_in`).
Each is the checker's call trace (`check_calls`, the one copy and nothing
else), its proof that no store of the entry touches the caller's bytes it
copies, and the driver's copy.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
  AlgorithmLib.HProg.Contracts AlgorithmLib.LibSpec AlgorithmLib.Device AlgorithmLib.Check AlgorithmLib.ML

namespace VitIo

/-- The one call `fetch` makes. -/
def fetchRec : CallRec :=
  ⟨.ext (.cuda .memcpyDtoH), [.is (regionBase .out), .is (devPtr 976), .is 102400]⟩

/-- The caller's output bytes `fetch` fills. -/
def outRange : List (Nat × Nat) := [((regionBase .out).toNat, (regionBase .out).toNat + 102400)]

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem fetch_calls (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : (VitSafe.Run 976 (Vit.vApp.bufSlot Vit.VOUT) 0 102400) w) {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit VitSafe.fetchCode) = .ok obs w') :
    (Tr.lit [fetchRec]).Covers w obs ∧ DevPath w w' ∧ StoresOut outRange w w' := by
  rw [VitSafe.fetch_eq] at hr
  check_calls

/-- What a world `VitSafe.Run` describes gives a copy: the driver loaded, no
    capture open, allocation `i` live at 102400 bytes. -/
theorem run_facts {i s d o : Nat} {w : World} (h : VitSafe.Run i s d o w) :
    (∀ f, w.has (Ext.cuda f).lib = true) ∧ w.dev.capture = none ∧
      ∃ B, (w.dev.bufs[i]?).join = some B ∧ B.size = 102400 := by
  refine ⟨fun f => h (.part (.has .cuda) true) (by simp), ?_, h (.alloc i 102400) (by simp)⟩
  have : w.dev.capture.isSome = false := h (.part .capturing false) (by simp)
  cases hc : w.dev.capture with
  | none => rfl
  | some _ => rw [hc] at this; cases this

/-- **`fetch` leaves the logits in the caller's output**: every run of it that
    ends, from what `main` leaves, holds in the output's first 102400 bytes,
    at the end, the logits buffer's bytes as the run found them. -/
theorem fetch_out (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : (VitSafe.Run 976 (Vit.vApp.bufSlot Vit.VOUT) 0 102400) w) {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit VitSafe.fetchCode) = .ok obs w') :
    ∀ j < 102400, w'.mem.load (regionBase .out + UInt64.ofNat j) 1 = some ((devMem w.dev 976).get! j).toUInt64 := by
  obtain ⟨hcov, hdp, hst⟩ := fetch_calls cfg dataLen outLen w h hr
  obtain ⟨dT, eT, kT⟩ : ∃ dT, w'.obs = dT ++ w.obs ∧ ∀ b n v, Obs.store b n v ∈ dT →
      ∀ i < n, Outside outRange (b + UInt64.ofNat i).toNat := by
    rcases hst with h0 | h0
    · simp [outRange] at h0
    · exact h0
  obtain ⟨hcu, hcap, -⟩ := run_facts h
  obtain ⟨cs, -, hh, hx⟩ := extRun_of hr ⟨hcov, hdp⟩ (fun _ hh => hh.lit_ext rfl)
  cases hh with
  | lit hm =>
  rcases cs with _ | ⟨x, _ | ⟨y, cs⟩⟩
  · exact hm.elim
  · obtain ⟨hmx, -⟩ := hm
    obtain ⟨wd, wd', e, vs, r, hdd, hfd, rfl, hext, hd', -, hm1, hm2, hod⟩ := hx.one_inv
    have he : Callee.ext e = .ext (.cuda .memcpyDtoH) := hmx.1.symm
    injection he with he
    subst he
    have hhas : wd.has (Ext.cuda .memcpyDtoH).lib = true := by
      have := hcu .memcpyDtoH; unfold World.has at this ⊢; rw [hfd.2.2]; exact this
    obtain ⟨id, off, B, hrg, hci, -⟩ := dtoh_done hhas (by rw [hdd]; exact hcap) hmx hext
    obtain ⟨hp, hB⟩ := range_eq hrg
    have h976 : devPtrOf (devPtr 976) = some (976, 0) := by decide
    rw [h976] at hp
    cases hp
    rw [hdd] at hB
    subst hB
    have hfit := (ExtContracts.range?_live hrg).2
    have h102 : (102400 : UInt64).toNat = 102400 := by decide
    rw [h102] at hci hfit
    have hsub := stretch_sub (wA := wd') (wB := w') eT ⟨[], rfl⟩
      (by
        obtain ⟨d1, e1, -⟩ := hm1
        refine ⟨[Obs.call (.ext (.cuda .memcpyDtoH)) (vs ++ callWords wd (.ext (.cuda .memcpyDtoH)) vs)] ++ d1, ?_⟩
        rw [hod, e1]; simp)
    intro j hj
    have hsrc : ((devMem w.dev 976).extract 0 (0 + 102400)).size = 102400 := by
      simp [ByteArray.size_extract]; omega
    rw [kept_load kT (List.mem_singleton_self _)
        (by rw [addr_toNat .out (show j ≤ 2 ^ 36 by omega)]; omega) hm2 hsub,
      copyIn_load hci (by omega) j (by omega), extract_get! (by omega) (by omega), Nat.zero_add]
  · exact hm.2.elim

/-- The calls `seed` makes: the upload, then the wait. -/
def seedRecs : List CallRec :=
  [⟨.ext (.cuda .memcpyHtoD), [.is (devPtr 977), .is (regionBase .data), .is 102400]⟩,
   ⟨.ext (.cuda .ctxSynchronize), []⟩]

/-- The caller's data bytes `seed` reads. -/
def dataRange : List (Nat × Nat) := [((regionBase .data).toNat, (regionBase .data).toNat + 102400)]

set_option maxRecDepth 200000 in
set_option maxHeartbeats 4000000 in
theorem seed_calls (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : (VitSafe.Run 977 (Vit.vApp.bufSlot Vit.VSEED) 102400 0) w) {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit VitSafe.seedCode) = .ok obs w') :
    (Tr.lit seedRecs).Covers w obs ∧ DevPath w w' ∧ StoresOut dataRange w w' := by
  rw [VitSafe.seed_eq] at hr
  check_calls

/-- An allocation overwritten from its start by as many bytes is those bytes. -/
theorem overwrite_all {b src : ByteArray} (h : src.size = b.size) : overwrite b 0 src = src := by
  apply ByteArray.ext
  simp [overwrite, h]

/-- **`seed` puts the caller's data in the backward's start buffer**: every
    run of it that ends, from what `main` leaves, leaves that buffer holding
    the caller's first 102400 data bytes as the caller handed them over. -/
theorem seed_in (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : (VitSafe.Run 977 (Vit.vApp.bufSlot Vit.VSEED) 102400 0) w) {obs : List Obs} {w' : World}
    (hr : Sem.run cfg (runArgs dataLen outLen) w (emit VitSafe.seedCode) = .ok obs w') :
    readBytes w.mem (regionBase .data) 102400 = some (devMem w'.dev 977) := by
  obtain ⟨hcov, hdp, hst⟩ := seed_calls cfg dataLen outLen w h hr
  obtain ⟨dT, eT, kT⟩ : ∃ dT, w'.obs = dT ++ w.obs ∧ ∀ b n v, Obs.store b n v ∈ dT →
      ∀ i < n, Outside dataRange (b + UInt64.ofNat i).toNat := by
    rcases hst with h0 | h0
    · simp [dataRange] at h0
    · exact h0
  obtain ⟨hcu, hcap, B0, hB0, hsz0⟩ := run_facts h
  obtain ⟨cs, -, hh, hx⟩ := extRun_of hr ⟨hcov, hdp⟩ (fun _ hh => hh.lit_ext rfl)
  cases hh with
  | lit hm =>
  rcases cs with _ | ⟨x, _ | ⟨y, _ | ⟨z, cs⟩⟩⟩
  · exact hm.elim
  · exact hm.2.elim
  · simp only [seedRecs] at hm
    obtain ⟨hmx, hmy, -⟩ := hm
    obtain ⟨w1, hx1, hx2⟩ := hx.split [x] [y] rfl
    obtain ⟨wa, wa', e, vs, r, hda, hfa, rfl, hext, hd1, hf1, hma, hma', hoa⟩ := hx1.one_inv
    have he : e = .cuda .memcpyHtoD := Callee.ext.inj hmx.1.symm
    subst he
    -- the upload's record, its length as a range
    have hmx' : CallRec.Matches ⟨.ext (.cuda .memcpyHtoD), [.is (devPtr 977), .is (regionBase .data),
        .within 102400 102400]⟩ (.ext (.cuda .memcpyHtoD), vs ++ callWords wa (.ext (.cuda .memcpyHtoD)) vs) := by
      refine ⟨rfl, ?_⟩
      have ha := hmx.2
      rw [callWords_htod, List.append_nil] at ha ⊢
      obtain ⟨v0, v1, v2, rfl, h0, h1, h2⟩ := args_three ha
      exact ⟨h0, h1, ⟨102400, h2, by decide, by decide⟩, trivial⟩
    have hhas : wa.has (Ext.cuda .memcpyHtoD).lib = true := by
      have := hcu .memcpyHtoD; unfold World.has at this ⊢; rw [hfa.2.2]; exact this
    obtain ⟨id, off, B, bytes, hrg, hlo, hhi, hrd, hmem, hkeep⟩ :=
      htod_done hhas (by rw [hda]; exact hcap) hmx' hext
    obtain ⟨hp, hB⟩ := range_eq hrg
    have h977 : devPtrOf (devPtr 977) = some (977, 0) := by decide
    rw [h977] at hp
    cases hp
    rw [hda] at hB
    have hsz : bytes.size = 102400 := by omega
    -- the wait leaves device memory
    obtain ⟨wb, wb', e2, vs2, r2, hdb, hfb, rfl, hext2, hd2, -, -, -, -⟩ := hx2.one_inv
    have he2 : e2 = .cuda .ctxSynchronize := Callee.ext.inj hmy.1.symm
    subst he2
    have hcapb : wb.dev.capture = none := by
      rw [hdb, hd1, hkeep.1.capture, hda]; exact hcap
    have hdev : devMem w'.dev 977 = bytes := by
      rw [hd2, syncRec_done hcapb hext2, hdb, hd1, hmem]
      simp only [setBuf, if_true]
      rw [overwrite_all (by rw [hsz, ← hB]; simp [devMem, hB0, hsz0])]
    -- the caller's data is as it was when the upload reads it
    have hsub := stretch_sub (wA := w) (wB := wa) eT
      (by
        obtain ⟨d1, e1, -⟩ := hma'
        obtain ⟨d2, e2⟩ := hx2.obs
        refine ⟨d2 ++ d1 ++ [Obs.call (.ext (.cuda .memcpyHtoD)) (vs ++ callWords wa (.ext (.cuda .memcpyHtoD)) vs)], ?_⟩
        rw [e2, e1, hoa]; simp)
      ⟨[], rfl⟩
    rw [hdev, ← hsz, ← hrd]
    refine (copyOut_congr fun i hi => ?_).symm
    refine kept_load kT (List.mem_singleton_self _) ?_ hma hsub
    rw [addr_toNat .data (show i ≤ 2 ^ 36 by omega)]
    omega
  · exact hm.2.2.elim

end VitIo
