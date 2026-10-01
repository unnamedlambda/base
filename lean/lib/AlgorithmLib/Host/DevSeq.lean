module
public import AlgorithmLib.Host.DevRace
meta import AlgorithmLib.Host.DevRace
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# Calls on the default stream keep every access the default stream's

`DefaultOnly` (every access the tracker records is the default stream's) is
kept by each call that issues work only on the default stream: buffers made,
filled, read back and freed, synchronization, launches, and cuBLAS without a
stream. With `CReached.ownBelow` it makes every buffer ready for the default
stream (`ready_default`).
-/

namespace AlgorithmLib.Device

open AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.DevSpec (wAcc rAcc know Ready Below)

theorem devOnly_seq {w : World} {k : Option (Option V × Dev)} {r : Option V} {w' : World}
    (h : devOnly w k = some (r, w')) (hk : ∀ r d, k = some (r, d) → DefaultOnly d.race) :
    DefaultOnly w'.dev.race := by
  unfold devOnly at h
  obtain ⟨⟨r0, d⟩, hkd, he⟩ := Option.map_eq_some_iff.mp h
  simp only [Prod.mk.injEq] at he
  obtain ⟨-, rfl⟩ := he
  exact hk _ _ hkd

theorem syncWrite_seq {d d' : Dev} {id : Nat} {b : ByteArray} (h : d.syncWrite id b = some d')
    (ht : DefaultOnly d.race) : DefaultOnly d'.race := by
  unfold Dev.syncWrite at h
  obtain ⟨r, hr, h⟩ := Option.bind_eq_some_iff.mp h
  simp only [Option.some.injEq] at h
  subst h
  exact (ht.op hr).sync _

theorem syncRead_seq {d d' : Dev} {id : Nat} (h : d.syncRead id = some d') (ht : DefaultOnly d.race) :
    DefaultOnly d'.race := by
  unfold Dev.syncRead at h
  obtain ⟨r, hr, h⟩ := Option.bind_eq_some_iff.mp h
  simp only [Option.some.injEq] at h
  subst h
  exact (ht.op hr).sync _

/-- **A device operation of the default stream keeps every access its own**:
    captured, it moves only the capture's tracker. -/
theorem devOp_seq {d d' : Dev} {w : World} {op : DevOp} {rs ws : List Nat}
    (h : d.devOp w defaultParty op rs ws = some d') (ht : DefaultOnly d.race) : DefaultOnly d'.race := by
  unfold Dev.devOp at h
  have run_ok : ∀ {d'}, Dev.devOp.run d w defaultParty op rs ws = some d' → DefaultOnly d'.race := by
    intro d' h
    unfold Dev.devOp.run at h
    obtain ⟨r, hr, h⟩ := Option.bind_eq_some_iff.mp h
    split at h
    · obtain ⟨e1, -, -⟩ := runLaunch_meta h
      rw [e1]; exact ht.op hr
    · obtain ⟨e1, -, -⟩ := runVendor_meta h
      rw [e1]; exact ht.op hr
  split at h
  · split at h
    · obtain ⟨r, -, h⟩ := Option.bind_eq_some_iff.mp h
      simp only [Option.some.injEq] at h
      subst h
      exact ht
    · exact run_ok h
  · exact run_ok h

theorem cudaLaunchOn_seq {w : World} {kernel entry : String} {nBufs bindPtr : UInt64}
    {dims : List UInt64} {r : Option V} {d : Dev}
    (hk : cudaLaunchOn w defaultParty kernel entry nBufs bindPtr dims = some (r, d)) (ht : DefaultOnly w.dev.race) :
    DefaultOnly d.race := by
  unfold cudaLaunchOn at hk
  obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
  split at hk
  · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
  · obtain ⟨d', hop, hk⟩ := Option.bind_eq_some_iff.mp hk
    simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact devOp_seq hop ht

theorem sgemvOn_seq {w : World} {trans m n alpha a x beta y : UInt64} {r : Option V}
    {d : Dev} (hk : sgemvOn w defaultParty trans m n alpha a x beta y = some (r, d)) (ht : DefaultOnly w.dev.race) :
    DefaultOnly d.race := by
  unfold sgemvOn at hk
  split at hk
  · split at hk
    · cases hk
    · dsimp only at hk
      obtain ⟨d', hop, hk⟩ := Option.bind_eq_some_iff.mp hk
      simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact devOp_seq hop ht
  · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht

theorem sgemmOn_seq {w : World}
    {ta tb m n k alpha a sa b sb beta c sc batch oa ob oc la lb lc : UInt64} {r : Option V} {d : Dev}
    (hk : sgemmOn w defaultParty ta tb m n k alpha a sa b sb beta c sc batch oa ob oc la lb lc = some (r, d))
    (ht : DefaultOnly w.dev.race) : DefaultOnly d.race := by
  unfold sgemmOn at hk
  split at hk
  · split at hk
    · cases hk
    · dsimp only at hk
      obtain ⟨d', hop, hk⟩ := Option.bind_eq_some_iff.mp hk
      simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact devOp_seq hop ht
  · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht

theorem seq_cudaCreateBuffer {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaCreateBuffer bits w = some (r, w')) (ht : DefaultOnly w.dev.race) : DefaultOnly w'.dev.race := by
  unfold ffiCudaCreateBuffer at h
  refine devOnly_seq h ?_
  intro r d hk
  split at hk
  · split at hk
    · simp only [devFail, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
      split at hk
      · simp only [devFail, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
      · simp only [Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk
        exact ht
  · cases hk

theorem seq_cudaUpload {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaUpload bits w = some (r, w')) (ht : DefaultOnly w.dev.race) : DefaultOnly w'.dev.race := by
  unfold ffiCudaUpload at h
  refine devOnly_seq h ?_
  intro r d hk
  split at hk
  · split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
      split at hk
      · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
      · split at hk
        · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
        · split at hk
          · simp only [devFail, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
          · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
            obtain ⟨d', hsw, hk⟩ := Option.bind_eq_some_iff.mp hk
            simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact syncWrite_seq hsw ht
  · cases hk

theorem seq_cudaUploadOffset {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaUploadOffset bits w = some (r, w')) (ht : DefaultOnly w.dev.race) : DefaultOnly w'.dev.race := by
  unfold ffiCudaUploadOffset at h
  refine devOnly_seq h ?_
  intro r d hk
  split at hk
  · split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
      split at hk
      · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
      · split at hk
        · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
        · split at hk
          · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
          · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
            obtain ⟨d', hsw, hk⟩ := Option.bind_eq_some_iff.mp hk
            simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact syncWrite_seq hsw ht
  · cases hk

theorem seq_cudaDownload {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaDownload bits w = some (r, w')) (ht : DefaultOnly w.dev.race) : DefaultOnly w'.dev.race := by
  unfold ffiCudaDownload at h
  split at h
  · split at h
    · simp only [cudaFail, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; exact ht
    · obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      split at h
      · simp only [cudaFail, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; exact ht
      · split at h
        · simp only [cudaFail, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; exact ht
        · split at h
          · simp only [cudaFail, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; exact ht
          · obtain ⟨d', hsr, h⟩ := Option.bind_eq_some_iff.mp h
            obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
            simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
            exact syncRead_seq hsr ht
  · cases h

theorem seq_cudaDownloadOffset {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaDownloadOffset bits w = some (r, w')) (ht : DefaultOnly w.dev.race) : DefaultOnly w'.dev.race := by
  unfold ffiCudaDownloadOffset at h
  split at h
  · split at h
    · simp only [cudaFail, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; exact ht
    · obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      split at h
      · simp only [cudaFail, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; exact ht
      · split at h
        · simp only [cudaFail, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; exact ht
        · split at h
          · simp only [cudaFail, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; exact ht
          · obtain ⟨d', hsr, h⟩ := Option.bind_eq_some_iff.mp h
            obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
            simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
            exact syncRead_seq hsr ht
  · cases h

theorem seq_cudaFreeBuffer {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaFreeBuffer bits w = some (r, w')) (ht : DefaultOnly w.dev.race) : DefaultOnly w'.dev.race := by
  unfold ffiCudaFreeBuffer at h
  refine devOnly_seq h ?_
  intro r d hk
  split at hk
  · split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
      split at hk
      · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
      · split at hk
        · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
        · obtain ⟨r', hop, hk⟩ := Option.bind_eq_some_iff.mp hk
          simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk
          exact ht.op hop
  · cases hk

theorem seq_cudaSync {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaSync bits w = some (r, w')) (ht : DefaultOnly w.dev.race) : DefaultOnly w'.dev.race := by
  unfold ffiCudaSync at h
  refine devOnly_seq h ?_
  intro r d hk
  split at hk
  · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
    split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk
      exact ht.sync _
  · cases hk

theorem seq_cudaLaunch {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaLaunch bits w = some (r, w')) (ht : DefaultOnly w.dev.race) : DefaultOnly w'.dev.race := by
  unfold ffiCudaLaunch at h
  refine devOnly_seq h ?_
  intro r d hk
  split at hk
  · split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
      split at hk
      · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
      · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
        exact cudaLaunchOn_seq hk ht
  · cases hk

theorem seq_cudaLaunchNamed {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaLaunchNamed bits w = some (r, w')) (ht : DefaultOnly w.dev.race) : DefaultOnly w'.dev.race := by
  unfold ffiCudaLaunchNamed at h
  refine devOnly_seq h ?_
  intro r d hk
  split at hk
  · split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
      split at hk
      · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
      · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
        obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
        exact cudaLaunchOn_seq hk ht
  · cases hk

theorem seq_cublasSgemv {bits : List UInt64} {w : World} {r w'}
    (h : ffiCublasSgemv bits w = some (r, w')) (ht : DefaultOnly w.dev.race) : DefaultOnly w'.dev.race := by
  unfold ffiCublasSgemv at h
  refine devOnly_seq h ?_
  intro r d hk
  split at hk
  · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
    split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · exact sgemvOn_seq hk ht
  · cases hk

theorem seq_cublasSgemm {bits : List UInt64} {w : World} {r w'}
    (h : ffiCublasSgemm bits w = some (r, w')) (ht : DefaultOnly w.dev.race) : DefaultOnly w'.dev.race := by
  unfold ffiCublasSgemm at h
  refine devOnly_seq h ?_
  intro r d hk
  split at hk
  · split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
      split at hk
      · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
      · exact sgemmOn_seq hk ht
  · cases hk

theorem seq_cublasGemmExBf16 {bits : List UInt64} {w : World} {r w'}
    (h : ffiCublasGemmExBf16 bits w = some (r, w')) (ht : DefaultOnly w.dev.race) : DefaultOnly w'.dev.race := by
  unfold ffiCublasGemmExBf16 at h
  refine devOnly_seq h ?_
  intro r d hk
  split at hk
  · split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
      split at hk
      · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
      · split at hk
        · split at hk
          · cases hk
          · dsimp only at hk
            obtain ⟨d', hop, hk⟩ := Option.bind_eq_some_iff.mp hk
            simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact devOp_seq hop ht
        · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
  · cases hk

theorem seq_cublasGemmStridedBatchedExBf16 {bits : List UInt64} {w : World} {r w'}
    (h : ffiCublasGemmStridedBatchedExBf16 bits w = some (r, w')) (ht : DefaultOnly w.dev.race) :
    DefaultOnly w'.dev.race := by
  unfold ffiCublasGemmStridedBatchedExBf16 at h
  refine devOnly_seq h ?_
  intro r d hk
  split at hk
  · split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
      split at hk
      · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
      · split at hk
        · split at hk
          · cases hk
          · dsimp only at hk
            obtain ⟨d', hop, hk⟩ := Option.bind_eq_some_iff.mp hk
            simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact devOp_seq hop ht
        · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
  · cases hk

theorem memInfoOf_seq {w : World} {ctx : UInt64} {pick : UInt64 × UInt64 → UInt64}
    {r : Option V} {d : Dev} (hk : memInfoOf w ctx pick = some (r, d)) (ht : DefaultOnly w.dev.race) :
    DefaultOnly d.race := by
  unfold memInfoOf at hk
  obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
  split at hk <;> (simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht)

theorem seq_cudaPinnedAlloc {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaPinnedAlloc bits w = some (r, w')) (ht : DefaultOnly w.dev.race) : DefaultOnly w'.dev.race := by
  unfold ffiCudaPinnedAlloc at h
  split at h
  · split at h
    · simp only [cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; exact ht
    · obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      split at h
      · simp only [cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; exact ht
      · dsimp only at h
        split at h
        · cases h
        · simp only [cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; exact ht
  · cases h

theorem seq_cudaPinnedFree {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaPinnedFree bits w = some (r, w')) (ht : DefaultOnly w.dev.race) : DefaultOnly w'.dev.race := by
  unfold ffiCudaPinnedFree at h
  split at h
  · split at h
    · simp only [cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; exact ht
    · obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      split at h
      · simp only [cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; exact ht
      · split at h
        · simp only [cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; exact ht
        · simp only [cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h; exact ht
  · cases h

theorem seq_cudaPinnedPtr {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaPinnedPtr bits w = some (r, w')) (ht : DefaultOnly w.dev.race) : DefaultOnly w'.dev.race := by
  unfold ffiCudaPinnedPtr at h
  refine devOnly_seq h ?_
  intro r d hk
  split at hk
  · split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
      split at hk
      · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
      · split at hk <;> (simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht)
  · cases hk

theorem seq_cudaPinnedPtrAt {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaPinnedPtrAt bits w = some (r, w')) (ht : DefaultOnly w.dev.race) : DefaultOnly w'.dev.race := by
  unfold ffiCudaPinnedPtrAt at h
  refine devOnly_seq h ?_
  intro r d hk
  split at hk
  · split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
      split at hk
      · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
      · split at hk
        · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
        · split at hk <;> (simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht)
  · cases hk

theorem seq_cudaMemInfoFree {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaMemInfoFree bits w = some (r, w')) (ht : DefaultOnly w.dev.race) : DefaultOnly w'.dev.race := by
  unfold ffiCudaMemInfoFree at h
  refine devOnly_seq h ?_
  intro r d hk
  split at hk
  · exact memInfoOf_seq hk ht
  · cases hk

theorem seq_cudaMemInfoTotal {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaMemInfoTotal bits w = some (r, w')) (ht : DefaultOnly w.dev.race) : DefaultOnly w'.dev.race := by
  unfold ffiCudaMemInfoTotal at h
  refine devOnly_seq h ?_
  intro r d hk
  split at hk
  · exact memInfoOf_seq hk ht
  · cases hk

theorem seq_cudaStreamCreate {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaStreamCreate bits w = some (r, w')) (ht : DefaultOnly w.dev.race) : DefaultOnly w'.dev.race := by
  unfold ffiCudaStreamCreate at h
  refine devOnly_seq h ?_
  intro r d hk
  split at hk
  · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
    split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk
      first | exact ht | exact ht.joinInto _ _ | exact (ht.joinInto _ _).joinInto _ _
  · cases hk

theorem seq_cudaStreamSync {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaStreamSync bits w = some (r, w')) (ht : DefaultOnly w.dev.race) : DefaultOnly w'.dev.race := by
  unfold ffiCudaStreamSync at h
  refine devOnly_seq h ?_
  intro r d hk
  split at hk
  · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
    split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · split at hk
      · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
      · split at hk
        · cases hk
        · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk
          first | exact ht | exact ht.joinInto _ _ | exact (ht.joinInto _ _).joinInto _ _
  · cases hk

theorem seq_cudaStreamDestroy {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaStreamDestroy bits w = some (r, w')) (ht : DefaultOnly w.dev.race) : DefaultOnly w'.dev.race := by
  unfold ffiCudaStreamDestroy at h
  refine devOnly_seq h ?_
  intro r d hk
  split at hk
  · split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
      split at hk
      · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
      · split at hk
        · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
        · dsimp only at hk
          split at hk
          · cases hk
          · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk
            first | exact ht | exact ht.joinInto _ _ | exact (ht.joinInto _ _).joinInto _ _
  · cases hk

theorem seq_cudaEventElapsedMsBits {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaEventElapsedMsBits bits w = some (r, w')) (ht : DefaultOnly w.dev.race) : DefaultOnly w'.dev.race := by
  unfold ffiCudaEventElapsedMsBits at h
  refine devOnly_seq h ?_
  intro r d hk
  rcases bits with _ | ⟨c, _ | ⟨s, _ | ⟨e, _ | ⟨_, _⟩⟩⟩⟩ <;> simp only at hk <;> try cases hk
  obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
  split at hk
  · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
  · split at hk <;> first
      | (simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht)
      | cases hk
      | (split at hk <;> first | (simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht) | cases hk)

theorem seq_cublasPtrArray {bits : List UInt64} {w : World} {r w'}
    (h : ffiCublasPtrArray bits w = some (r, w')) (ht : DefaultOnly w.dev.race) : DefaultOnly w'.dev.race := by
  unfold ffiCublasPtrArray at h
  refine devOnly_seq h ?_
  intro r d hk
  split at hk
  · split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
      split at hk
      · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
      · split at hk
        · dsimp only at hk
          split at hk
          · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
          · obtain ⟨d', hsw, hk⟩ := Option.bind_eq_some_iff.mp hk
            simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact syncWrite_seq hsw ht
        · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
  · cases hk

theorem seq_cudaGraphDestroy {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaGraphDestroy bits w = some (r, w')) (ht : DefaultOnly w.dev.race) : DefaultOnly w'.dev.race := by
  unfold ffiCudaGraphDestroy at h
  refine devOnly_seq h ?_
  intro r d hk
  split at hk
  · split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
      split at hk
      · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
      · split at hk
        · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
        · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
  · cases hk

theorem seq_cudaGraphUpload {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaGraphUpload bits w = some (r, w')) (ht : DefaultOnly w.dev.race) : DefaultOnly w'.dev.race := by
  unfold ffiCudaGraphUpload at h
  refine devOnly_seq h ?_
  intro r d hk
  split at hk
  · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
    split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · split at hk <;> (simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht)
  · cases hk

/-- A fresh device records no access. -/
theorem seq_cudaInit {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaInit bits w = some (r, w')) : DefaultOnly w'.dev.race := by
  unfold ffiCudaInit at h
  split at h
  · split at h <;>
    · obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
      simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
      exact defaultOnly_empty
  · cases h

theorem seq_cudaCleanup {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaCleanup bits w = some (r, w')) : DefaultOnly w'.dev.race := by
  unfold ffiCudaCleanup at h
  split at h
  · obtain ⟨_, _, h⟩ := Option.bind_eq_some_iff.mp h
    simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨-, rfl⟩ := h
    exact defaultOnly_empty
  · cases h

/-- The calls that issue work only on the default stream, and those that make,
    wait on and drop streams and page-locked memory without issuing any. -/
def seqKeepers : List IR.Ffi :=
  [.cudaCreateBuffer, .cudaUpload, .cudaUploadOffset, .cudaDownload, .cudaDownloadOffset, .cudaFreeBuffer, .cudaSync, .cudaLaunch, .cudaLaunchNamed, .cublasSgemv, .cublasSgemm, .cublasGemmExBf16, .cublasGemmStridedBatchedExBf16, .cudaPinnedAlloc, .cudaPinnedFree, .cudaPinnedPtr, .cudaPinnedPtrAt, .cudaMemInfoFree, .cudaMemInfoTotal, .cudaStreamCreate, .cudaStreamSync, .cudaStreamDestroy, .cudaEventElapsedMsBits, .cublasPtrArray, .cudaGraphDestroy, .cudaGraphUpload]

/-- **A call that issues work only on the default stream keeps every access
    the default stream's.** -/
theorem callBits_seq (f : IR.Ffi) (hf : f ∈ seqKeepers) {bits : List UInt64} {w : World} {r : Option V}
    {w' : World} (h : callBits f bits w = some (r, w')) (hd : DefaultOnly w.dev.race) :
    DefaultOnly w'.dev.race := by
  cases f
  case cudaCreateBuffer => exact seq_cudaCreateBuffer h hd
  case cudaUpload => exact seq_cudaUpload h hd
  case cudaUploadOffset => exact seq_cudaUploadOffset h hd
  case cudaDownload => exact seq_cudaDownload h hd
  case cudaDownloadOffset => exact seq_cudaDownloadOffset h hd
  case cudaFreeBuffer => exact seq_cudaFreeBuffer h hd
  case cudaSync => exact seq_cudaSync h hd
  case cudaLaunch => exact seq_cudaLaunch h hd
  case cudaLaunchNamed => exact seq_cudaLaunchNamed h hd
  case cublasSgemv => exact seq_cublasSgemv h hd
  case cublasSgemm => exact seq_cublasSgemm h hd
  case cublasGemmExBf16 => exact seq_cublasGemmExBf16 h hd
  case cublasGemmStridedBatchedExBf16 => exact seq_cublasGemmStridedBatchedExBf16 h hd
  case cudaPinnedAlloc => exact seq_cudaPinnedAlloc h hd
  case cudaPinnedFree => exact seq_cudaPinnedFree h hd
  case cudaPinnedPtr => exact seq_cudaPinnedPtr h hd
  case cudaPinnedPtrAt => exact seq_cudaPinnedPtrAt h hd
  case cudaMemInfoFree => exact seq_cudaMemInfoFree h hd
  case cudaMemInfoTotal => exact seq_cudaMemInfoTotal h hd
  case cudaStreamCreate => exact seq_cudaStreamCreate h hd
  case cudaStreamSync => exact seq_cudaStreamSync h hd
  case cudaStreamDestroy => exact seq_cudaStreamDestroy h hd
  case cudaEventElapsedMsBits => exact seq_cudaEventElapsedMsBits h hd
  case cublasPtrArray => exact seq_cublasPtrArray h hd
  case cudaGraphDestroy => exact seq_cudaGraphDestroy h hd
  case cudaGraphUpload => exact seq_cudaGraphUpload h hd
  all_goals simp [seqKeepers] at hf

end AlgorithmLib.Device
