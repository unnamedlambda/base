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

/-- Close `DefaultOnly` at one answer of a contract, through the steps it took. -/
macro "seq_close" : tactic => `(tactic| (
  repeat' (first
    | assumption
    | exact defaultOnly_empty
    | refine devOp_seq (by assumption) ?_
    | refine syncWrite_seq (by assumption) ?_
    | refine syncRead_seq (by assumption) ?_
    | refine memInfoOf_seq (by assumption) ?_
    | refine sgemvOn_seq (by assumption) ?_
    | refine sgemmOn_seq (by assumption) ?_
    | refine DefaultOnly.sync ?_ _
    | refine DefaultOnly.joinInto ?_ _ _
    | refine DefaultOnly.op ?_ (by assumption))))

end AlgorithmLib.Device
