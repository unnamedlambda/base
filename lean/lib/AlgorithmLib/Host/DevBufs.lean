module
public import AlgorithmLib.Host.DevSeq
meta import AlgorithmLib.Host.DevSeq
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# A buffer made keeps its size

Buffer ids are handed out in order and never handed out again, and nothing
the device does changes a buffer's length: uploads write a buffer's own
length, kernels and cuBLAS are refused an output of another length, and a
free empties the id for good. So a buffer made with `n` bytes has `n` bytes
for as long as it lives (`BufIs`), through every call that does not start the
device afresh (`callBits_bufs`).
-/

namespace AlgorithmLib.Device

open AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.DevSpec (get?_put_size mapM_get_sizes)

theorem put_bufs_size (d : Dev) (id : Nat) (o : Option ByteArray) : (d.put id o).bufs.size = d.bufs.size := by
  simp [Dev.put]

/-- Buffer `i` has been made, and has `n` bytes while it lives. -/
def BufIs (i : Int) (n : Nat) (d : Dev) : Prop :=
  0 ≤ i ∧ i.toNat < d.bufs.size ∧ ∀ b, d.get? i = some b → b.size = n

theorem BufIs.of_sizes {i : Int} {n : Nat} {d d' : Dev} (h : BufIs i n d) (hs : d.bufs.size ≤ d'.bufs.size)
    (hm : ∀ j, (d'.get? j).map ByteArray.size = (d.get? j).map ByteArray.size) : BufIs i n d' := by
  obtain ⟨h0, hlt, hb⟩ := h
  refine ⟨h0, Nat.lt_of_lt_of_le hlt hs, fun b' hb' => ?_⟩
  have := hm i
  rw [hb'] at this
  cases e : d.get? i with
  | none => rw [e] at this; cases this
  | some b => rw [e] at this; simp only [Option.map_some, Option.some.injEq] at this; rw [this]; exact hb b e

theorem BufIs.push {i : Int} {n : Nat} {d : Dev} (h : BufIs i n d) (x : Option ByteArray) :
    BufIs i n { d with bufs := d.bufs.push x } := by
  obtain ⟨h0, hlt, hb⟩ := h
  refine ⟨h0, by simp only [Array.size_push]; omega, fun b hb' => hb b ?_⟩
  have hn : ¬ i < 0 := by omega
  simp only [Dev.get?, hn, if_false, Array.getElem?_push] at hb' ⊢
  rw [if_neg (by omega)] at hb'
  exact hb'

theorem BufIs.put_none {i : Int} {n : Nat} {d : Dev} (h : BufIs i n d) (id : Nat) : BufIs i n (d.put id none) := by
  obtain ⟨h0, hlt, hb⟩ := h
  refine ⟨h0, by rw [put_bufs_size]; exact hlt, fun b hb' => hb b ?_⟩
  have hn : ¬ i < 0 := by omega
  simp only [Dev.get?, Dev.put, hn, if_false, Array.getElem?_setIfInBounds] at hb' ⊢
  by_cases e : id = i.toNat
  · subst e; simp [hlt] at hb'
  · rw [if_neg e] at hb'; exact hb'

theorem BufIs.put_same {i : Int} {n : Nat} {d : Dev} (h : BufIs i n d) {id : Nat} {b : ByteArray}
    (hb : ∀ x, d.get? id = some x → x.size = b.size) (hlive : (d.get? id).isSome = true) :
    BufIs i n (d.put id (some b)) :=
  h.of_sizes (Nat.le_of_eq (put_bufs_size _ _ _).symm) (get?_put_size d id b · hb hlive)

theorem map_size_of_check : ∀ {os is : List ByteArray},
    (os.length != is.length || (os.zip is).any (fun (o, i) => o.size != i.size)) = false →
      os.map ByteArray.size = is.map ByteArray.size
  | [], [], _ => rfl
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | o :: os, i :: is, h => by
      simp only [List.length_cons, List.zip_cons_cons, List.any_cons, Bool.or_eq_false_iff,
        bne_eq_false_iff_eq, Nat.add_right_cancel_iff] at h
      simp only [List.map_cons, List.cons.injEq]
      exact ⟨h.2.1, map_size_of_check (by simp [h.1, h.2.2])⟩

/-- A kernel run keeps every buffer's length: it is refused outputs of other
    lengths. -/
theorem runLaunch_bufs {i : Int} {n : Nat} {d d' : Dev} {k : Launch → List ByteArray → List ByteArray}
    {l : Launch} {ids : List Nat} (h : d.runLaunch k l ids = some d') (hb : BufIs i n d) : BufIs i n d' := by
  unfold Dev.runLaunch at h
  obtain ⟨ins, hins, h⟩ := Option.bind_eq_some_iff.mp h
  dsimp only at h
  split at h
  · cases h
  rename_i hc
  simp only [Option.some.injEq] at h
  subst h
  have hsz := map_size_of_check (Bool.eq_false_iff.mpr hc)
  suffices ∀ (ids : List Nat) (outs ins : List ByteArray) (d : Dev),
      ids.mapM (fun i => d.get? (Int.ofNat i)) = some ins → outs.map ByteArray.size = ins.map ByteArray.size →
      BufIs i n d → BufIs i n ((ids.zip outs).foldl (fun d (id, o) => d.put id (some o)) d) from
    this ids _ ins d hins hsz hb
  intro ids
  induction ids with
  | nil => intro _ _ _ _ _ hb; exact hb
  | cons id ids ih =>
      intro outs ins d hm hs hb
      cases outs with
      | nil => exact hb
      | cons o os =>
          simp only [List.mapM_cons] at hm
          obtain ⟨x, hx, hm⟩ := Option.bind_eq_some_iff.mp hm
          obtain ⟨xs, hxs, hm⟩ := Option.bind_eq_some_iff.mp hm
          simp only [Option.pure_def, Option.some.injEq] at hm
          subst hm
          simp only [List.map_cons, List.cons.injEq] at hs
          simp only [List.zip_cons_cons, List.foldl_cons]
          have hget : d.get? id = d.get? (Int.ofNat id) := rfl
          have hput : ∀ j, ((d.put id (some o)).get? j).map ByteArray.size = (d.get? j).map ByteArray.size :=
            fun j => get?_put_size d id o j (fun y hy => by rw [hget, hx] at hy; cases hy; exact hs.1.symm)
              (by rw [hget, hx]; rfl)
          obtain ⟨ys, hys, hyssz⟩ := mapM_get_sizes hput ids xs hxs
          exact ih os ys (d.put id (some o)) hys (by rw [hs.2, hyssz])
            (hb.of_sizes (Nat.le_of_eq (put_bufs_size _ _ _).symm) hput)

/-- A vendor routine keeps every buffer's length: it is refused an output of
    another length. -/
theorem runVendor_bufs {i : Int} {n : Nat} {d d' : Dev} {v : VendorCall → List ByteArray → ByteArray}
    {c : VendorCall} {ins : List Nat} {out : Nat} (h : d.runVendor v c ins out = some d') (hb : BufIs i n d) :
    BufIs i n d' := by
  unfold Dev.runVendor at h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨old, hold, h⟩ := Option.bind_eq_some_iff.mp h
  dsimp only at h
  split at h
  · cases h
  rename_i hc
  simp only [Option.some.injEq] at h
  subst h
  have hget : d.get? out = d.get? (Int.ofNat out) := rfl
  refine hb.put_same (fun x hx => ?_) (by rw [hget, hold]; rfl)
  rw [hget, hold] at hx
  cases hx
  simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hc
  exact hc.symm

theorem devOp_bufs {i : Int} {n : Nat} {d d' : Dev} {w : World} {p : Nat} {op : DevOp} {rs ws : List Nat}
    (h : d.devOp w p op rs ws = some d') (hb : BufIs i n d) : BufIs i n d' := by
  unfold Dev.devOp at h
  have run_ok : ∀ {d'}, Dev.devOp.run d w p op rs ws = some d' → BufIs i n d' := by
    intro d' h
    unfold Dev.devOp.run at h
    obtain ⟨r, -, h⟩ := Option.bind_eq_some_iff.mp h
    split at h
    · exact runLaunch_bufs h hb
    · exact runVendor_bufs h hb
  split at h
  · split at h
    · obtain ⟨r, -, h⟩ := Option.bind_eq_some_iff.mp h
      simp only [Option.some.injEq] at h
      subst h
      exact hb
    · exact run_ok h
  · exact run_ok h

theorem devOnly_bufs {i : Int} {n : Nat} {w : World} {k : Option (Option V × Dev)} {r : Option V} {w' : World}
    (h : devOnly w k = some (r, w')) (hk : ∀ r d, k = some (r, d) → BufIs i n d) : BufIs i n w'.dev := by
  unfold devOnly at h
  obtain ⟨⟨r0, d⟩, hkd, he⟩ := Option.map_eq_some_iff.mp h
  simp only [Prod.mk.injEq] at he
  obtain ⟨-, rfl⟩ := he
  exact hk _ _ hkd

theorem syncRead_bufs {i : Int} {n : Nat} {d d' : Dev} {id : Nat} (h : d.syncRead id = some d')
    (hb : BufIs i n d) : BufIs i n d' := by
  unfold Dev.syncRead at h
  obtain ⟨r, -, h⟩ := Option.bind_eq_some_iff.mp h
  simp only [Option.some.injEq] at h
  subst h
  exact hb

/-- A write of a live buffer's own length keeps every length. -/
theorem syncWrite_bufs {i : Int} {n : Nat} {d d' : Dev} {id : Nat} {b : ByteArray}
    (h : d.syncWrite id b = some d') (hsz : ∀ x, d.get? id = some x → x.size = b.size)
    (hlive : (d.get? id).isSome = true) (hb : BufIs i n d) : BufIs i n d' := by
  unfold Dev.syncWrite at h
  obtain ⟨r, -, h⟩ := Option.bind_eq_some_iff.mp h
  simp only [Option.some.injEq] at h
  subst h
  exact hb.put_same hsz hlive

theorem overwrite_size {b src : ByteArray} {off : Nat} (h : off + src.size ≤ b.size) :
    (overwrite b off src).size = b.size := by
  simp only [overwrite, ByteArray.size_append, ByteArray.size_extract]
  omega

theorem cudaLaunchOn_bufs {i₀ : Int} {n₀ : Nat} {w : World} {kernel entry : String} {nBufs bindPtr : UInt64}
    {dims : List UInt64} {r : Option V} {d : Dev}
    (hk : cudaLaunchOn w defaultParty kernel entry nBufs bindPtr dims = some (r, d)) (ht : BufIs i₀ n₀ w.dev) :
    BufIs i₀ n₀ d := by
  unfold cudaLaunchOn at hk
  obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
  split at hk
  · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
  · obtain ⟨d', hop, hk⟩ := Option.bind_eq_some_iff.mp hk
    simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact devOp_bufs hop ht

theorem sgemvOn_bufs {i₀ : Int} {n₀ : Nat} {w : World} {trans m n alpha a x beta y : UInt64} {r : Option V}
    {d : Dev} (hk : sgemvOn w defaultParty trans m n alpha a x beta y = some (r, d)) (ht : BufIs i₀ n₀ w.dev) :
    BufIs i₀ n₀ d := by
  unfold sgemvOn at hk
  split at hk
  · split at hk
    · cases hk
    · dsimp only at hk
      obtain ⟨d', hop, hk⟩ := Option.bind_eq_some_iff.mp hk
      simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact devOp_bufs hop ht
  · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht

theorem sgemmOn_bufs {i₀ : Int} {n₀ : Nat} {w : World}
    {ta tb m n k alpha a sa b sb beta c sc batch oa ob oc la lb lc : UInt64} {r : Option V} {d : Dev}
    (hk : sgemmOn w defaultParty ta tb m n k alpha a sa b sb beta c sc batch oa ob oc la lb lc = some (r, d))
    (ht : BufIs i₀ n₀ w.dev) : BufIs i₀ n₀ d := by
  unfold sgemmOn at hk
  split at hk
  · split at hk
    · cases hk
    · dsimp only at hk
      obtain ⟨d', hop, hk⟩ := Option.bind_eq_some_iff.mp hk
      simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact devOp_bufs hop ht
  · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht

theorem bufs_cudaCreateBuffer {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaCreateBuffer bits w = some (r, w')) (ht : BufIs i₀ n₀ w.dev) : BufIs i₀ n₀ w'.dev := by
  unfold ffiCudaCreateBuffer at h
  refine devOnly_bufs h ?_
  intro r d hk
  split at hk
  · split at hk
    · simp only [devFail, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
      split at hk
      · simp only [devFail, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
      · simp only [Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk
        exact ht.push _
  · cases hk

theorem bufs_cudaUpload {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaUpload bits w = some (r, w')) (ht : BufIs i₀ n₀ w.dev) : BufIs i₀ n₀ w'.dev := by
  unfold ffiCudaUpload at h
  refine devOnly_bufs h ?_
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
            simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk
            rename_i hc _ _ _ _ b hget hfit bytes hbytes _ _
            simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at hc
            have hid := Int.toNat_of_nonneg (Int.not_lt.mp hc.1.1)
            have hbs := AlgorithmLib.HProg.Static.copyOut_size hbytes
            have hb := hfit
            simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hb
            exact syncWrite_bufs hsw (fun x hx => by rw [hid, hget] at hx; cases hx; omega)
              (by rw [hid, hget]; rfl) ht
  · cases hk

theorem bufs_cudaUploadOffset {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaUploadOffset bits w = some (r, w')) (ht : BufIs i₀ n₀ w.dev) : BufIs i₀ n₀ w'.dev := by
  unfold ffiCudaUploadOffset at h
  refine devOnly_bufs h ?_
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
            simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk
            rename_i hc _ _ _ _ b hget hfit bytes hbytes _ _
            simp only [Bool.or_eq_true, decide_eq_true_eq, not_or] at hc
            have hid := Int.toNat_of_nonneg (Int.not_lt.mp hc.1.1.1)
            have hbs := AlgorithmLib.HProg.Static.copyOut_size hbytes
            exact syncWrite_bufs hsw (fun x hx => by
                rw [hid, hget] at hx; cases hx; rw [overwrite_size (by omega)])
              (by rw [hid, hget]; rfl) ht
  · cases hk

theorem bufs_cudaDownload {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaDownload bits w = some (r, w')) (ht : BufIs i₀ n₀ w.dev) : BufIs i₀ n₀ w'.dev := by
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
            exact syncRead_bufs hsr ht
  · cases h

theorem bufs_cudaDownloadOffset {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaDownloadOffset bits w = some (r, w')) (ht : BufIs i₀ n₀ w.dev) : BufIs i₀ n₀ w'.dev := by
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
            exact syncRead_bufs hsr ht
  · cases h

theorem bufs_cudaFreeBuffer {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaFreeBuffer bits w = some (r, w')) (ht : BufIs i₀ n₀ w.dev) : BufIs i₀ n₀ w'.dev := by
  unfold ffiCudaFreeBuffer at h
  refine devOnly_bufs h ?_
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
          exact ht.put_none _
  · cases hk

theorem bufs_cudaSync {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaSync bits w = some (r, w')) (ht : BufIs i₀ n₀ w.dev) : BufIs i₀ n₀ w'.dev := by
  unfold ffiCudaSync at h
  refine devOnly_bufs h ?_
  intro r d hk
  split at hk
  · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
    split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk
      exact ht
  · cases hk

theorem bufs_cudaLaunch {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaLaunch bits w = some (r, w')) (ht : BufIs i₀ n₀ w.dev) : BufIs i₀ n₀ w'.dev := by
  unfold ffiCudaLaunch at h
  refine devOnly_bufs h ?_
  intro r d hk
  split at hk
  · split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
      split at hk
      · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
      · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
        exact cudaLaunchOn_bufs hk ht
  · cases hk

theorem bufs_cudaLaunchNamed {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaLaunchNamed bits w = some (r, w')) (ht : BufIs i₀ n₀ w.dev) : BufIs i₀ n₀ w'.dev := by
  unfold ffiCudaLaunchNamed at h
  refine devOnly_bufs h ?_
  intro r d hk
  split at hk
  · split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
      split at hk
      · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
      · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
        obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
        exact cudaLaunchOn_bufs hk ht
  · cases hk

theorem bufs_cublasSgemv {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World} {r w'}
    (h : ffiCublasSgemv bits w = some (r, w')) (ht : BufIs i₀ n₀ w.dev) : BufIs i₀ n₀ w'.dev := by
  unfold ffiCublasSgemv at h
  refine devOnly_bufs h ?_
  intro r d hk
  split at hk
  · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
    split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · exact sgemvOn_bufs hk ht
  · cases hk

theorem bufs_cublasSgemm {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World} {r w'}
    (h : ffiCublasSgemm bits w = some (r, w')) (ht : BufIs i₀ n₀ w.dev) : BufIs i₀ n₀ w'.dev := by
  unfold ffiCublasSgemm at h
  refine devOnly_bufs h ?_
  intro r d hk
  split at hk
  · split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
      split at hk
      · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
      · exact sgemmOn_bufs hk ht
  · cases hk

theorem bufs_cublasGemmExBf16 {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World} {r w'}
    (h : ffiCublasGemmExBf16 bits w = some (r, w')) (ht : BufIs i₀ n₀ w.dev) : BufIs i₀ n₀ w'.dev := by
  unfold ffiCublasGemmExBf16 at h
  refine devOnly_bufs h ?_
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
            simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact devOp_bufs hop ht
        · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
  · cases hk

theorem bufs_cublasGemmStridedBatchedExBf16 {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World} {r w'}
    (h : ffiCublasGemmStridedBatchedExBf16 bits w = some (r, w')) (ht : BufIs i₀ n₀ w.dev) :
    BufIs i₀ n₀ w'.dev := by
  unfold ffiCublasGemmStridedBatchedExBf16 at h
  refine devOnly_bufs h ?_
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
            simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact devOp_bufs hop ht
        · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
  · cases hk

theorem memInfoOf_bufs {i₀ : Int} {n₀ : Nat} {w : World} {ctx : UInt64} {pick : UInt64 × UInt64 → UInt64}
    {r : Option V} {d : Dev} (hk : memInfoOf w ctx pick = some (r, d)) (ht : BufIs i₀ n₀ w.dev) :
    BufIs i₀ n₀ d := by
  unfold memInfoOf at hk
  obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
  split at hk <;> (simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht)

theorem bufs_cudaPinnedAlloc {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaPinnedAlloc bits w = some (r, w')) (ht : BufIs i₀ n₀ w.dev) : BufIs i₀ n₀ w'.dev := by
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

theorem bufs_cudaPinnedFree {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaPinnedFree bits w = some (r, w')) (ht : BufIs i₀ n₀ w.dev) : BufIs i₀ n₀ w'.dev := by
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

theorem bufs_cudaPinnedPtr {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaPinnedPtr bits w = some (r, w')) (ht : BufIs i₀ n₀ w.dev) : BufIs i₀ n₀ w'.dev := by
  unfold ffiCudaPinnedPtr at h
  refine devOnly_bufs h ?_
  intro r d hk
  split at hk
  · split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
      split at hk
      · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
      · split at hk <;> (simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht)
  · cases hk

theorem bufs_cudaPinnedPtrAt {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaPinnedPtrAt bits w = some (r, w')) (ht : BufIs i₀ n₀ w.dev) : BufIs i₀ n₀ w'.dev := by
  unfold ffiCudaPinnedPtrAt at h
  refine devOnly_bufs h ?_
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

theorem bufs_cudaMemInfoFree {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaMemInfoFree bits w = some (r, w')) (ht : BufIs i₀ n₀ w.dev) : BufIs i₀ n₀ w'.dev := by
  unfold ffiCudaMemInfoFree at h
  refine devOnly_bufs h ?_
  intro r d hk
  split at hk
  · exact memInfoOf_bufs hk ht
  · cases hk

theorem bufs_cudaMemInfoTotal {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaMemInfoTotal bits w = some (r, w')) (ht : BufIs i₀ n₀ w.dev) : BufIs i₀ n₀ w'.dev := by
  unfold ffiCudaMemInfoTotal at h
  refine devOnly_bufs h ?_
  intro r d hk
  split at hk
  · exact memInfoOf_bufs hk ht
  · cases hk

theorem bufs_cudaStreamCreate {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaStreamCreate bits w = some (r, w')) (ht : BufIs i₀ n₀ w.dev) : BufIs i₀ n₀ w'.dev := by
  unfold ffiCudaStreamCreate at h
  refine devOnly_bufs h ?_
  intro r d hk
  split at hk
  · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
    split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk
      first | exact ht | exact ht.joinInto _ _ | exact (ht.joinInto _ _).joinInto _ _
  · cases hk

theorem bufs_cudaStreamSync {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaStreamSync bits w = some (r, w')) (ht : BufIs i₀ n₀ w.dev) : BufIs i₀ n₀ w'.dev := by
  unfold ffiCudaStreamSync at h
  refine devOnly_bufs h ?_
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

theorem bufs_cudaStreamDestroy {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaStreamDestroy bits w = some (r, w')) (ht : BufIs i₀ n₀ w.dev) : BufIs i₀ n₀ w'.dev := by
  unfold ffiCudaStreamDestroy at h
  refine devOnly_bufs h ?_
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

theorem bufs_cudaEventElapsedMsBits {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaEventElapsedMsBits bits w = some (r, w')) (ht : BufIs i₀ n₀ w.dev) : BufIs i₀ n₀ w'.dev := by
  unfold ffiCudaEventElapsedMsBits at h
  refine devOnly_bufs h ?_
  intro r d hk
  rcases bits with _ | ⟨c, _ | ⟨s, _ | ⟨e, _ | ⟨_, _⟩⟩⟩⟩ <;> simp only at hk <;> try cases hk
  obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
  split at hk
  · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
  · split at hk <;> first
      | (simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht)
      | cases hk
      | (split at hk <;> first | (simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht) | cases hk)

theorem bufs_cudaGraphDestroy {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaGraphDestroy bits w = some (r, w')) (ht : BufIs i₀ n₀ w.dev) : BufIs i₀ n₀ w'.dev := by
  unfold ffiCudaGraphDestroy at h
  refine devOnly_bufs h ?_
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

theorem bufs_cudaGraphUpload {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World} {r w'}
    (h : ffiCudaGraphUpload bits w = some (r, w')) (ht : BufIs i₀ n₀ w.dev) : BufIs i₀ n₀ w'.dev := by
  unfold ffiCudaGraphUpload at h
  refine devOnly_bufs h ?_
  intro r d hk
  split at hk
  · obtain ⟨_, _, hk⟩ := Option.bind_eq_some_iff.mp hk
    split at hk
    · simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht
    · split at hk <;> (simp only [devFail, devOk, devI64, cudaFail, cudaI64, Option.some.injEq, Prod.mk.injEq] at hk; obtain ⟨-, rfl⟩ := hk; exact ht)
  · cases hk

/-- The calls that keep every buffer made at its size: all that issue work on
    the default stream, and those that make, wait on and drop streams,
    page-locked memory and graphs. -/
def bufKeepers : List IR.Ffi :=
  [.cudaCreateBuffer, .cudaUpload, .cudaUploadOffset, .cudaDownload, .cudaDownloadOffset, .cudaFreeBuffer, .cudaSync, .cudaLaunch, .cudaLaunchNamed, .cublasSgemv, .cublasSgemm, .cublasGemmExBf16, .cublasGemmStridedBatchedExBf16, .cudaPinnedAlloc, .cudaPinnedFree, .cudaPinnedPtr, .cudaPinnedPtrAt, .cudaMemInfoFree, .cudaMemInfoTotal, .cudaStreamCreate, .cudaStreamSync, .cudaStreamDestroy, .cudaEventElapsedMsBits, .cudaGraphDestroy, .cudaGraphUpload]

/-- **A call that keeps buffers keeps each one's size.** -/
theorem callBits_bufs (f : IR.Ffi) (hf : f ∈ bufKeepers) {i₀ : Int} {n₀ : Nat} {bits : List UInt64} {w : World}
    {r : Option V} {w' : World} (h : callBits f bits w = some (r, w')) (hd : BufIs i₀ n₀ w.dev) :
    BufIs i₀ n₀ w'.dev := by
  cases f
  case cudaCreateBuffer => exact bufs_cudaCreateBuffer h hd
  case cudaUpload => exact bufs_cudaUpload h hd
  case cudaUploadOffset => exact bufs_cudaUploadOffset h hd
  case cudaDownload => exact bufs_cudaDownload h hd
  case cudaDownloadOffset => exact bufs_cudaDownloadOffset h hd
  case cudaFreeBuffer => exact bufs_cudaFreeBuffer h hd
  case cudaSync => exact bufs_cudaSync h hd
  case cudaLaunch => exact bufs_cudaLaunch h hd
  case cudaLaunchNamed => exact bufs_cudaLaunchNamed h hd
  case cublasSgemv => exact bufs_cublasSgemv h hd
  case cublasSgemm => exact bufs_cublasSgemm h hd
  case cublasGemmExBf16 => exact bufs_cublasGemmExBf16 h hd
  case cublasGemmStridedBatchedExBf16 => exact bufs_cublasGemmStridedBatchedExBf16 h hd
  case cudaPinnedAlloc => exact bufs_cudaPinnedAlloc h hd
  case cudaPinnedFree => exact bufs_cudaPinnedFree h hd
  case cudaPinnedPtr => exact bufs_cudaPinnedPtr h hd
  case cudaPinnedPtrAt => exact bufs_cudaPinnedPtrAt h hd
  case cudaMemInfoFree => exact bufs_cudaMemInfoFree h hd
  case cudaMemInfoTotal => exact bufs_cudaMemInfoTotal h hd
  case cudaStreamCreate => exact bufs_cudaStreamCreate h hd
  case cudaStreamSync => exact bufs_cudaStreamSync h hd
  case cudaStreamDestroy => exact bufs_cudaStreamDestroy h hd
  case cudaEventElapsedMsBits => exact bufs_cudaEventElapsedMsBits h hd
  case cudaGraphDestroy => exact bufs_cudaGraphDestroy h hd
  case cudaGraphUpload => exact bufs_cudaGraphUpload h hd
  all_goals simp [bufKeepers] at hf

end AlgorithmLib.Device
