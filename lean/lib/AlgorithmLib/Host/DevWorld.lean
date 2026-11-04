module
public import AlgorithmLib.Host.Ffi
meta import AlgorithmLib.Host.Ffi
public import AlgorithmLib.Host.DevProg
meta import AlgorithmLib.Host.DevProg
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The World's device work, as device-program steps

A launch the model runs changes at most the buffers bound to it that its
kernel may write, to what the kernel oracle says of their contents; a vendor
call changes its one output. Read as `BufStep`s --- reading every buffer they
bind, writing those --- they are steps a `DevProg` can order, and `runLaunch_mem` says the World's
own `runLaunch` performs its step. So `DevProg.denote_linearisation` is a
statement about the launches the host contracts issue, not only about an
abstract program.
-/

namespace AlgorithmLib.Device

open AlgorithmLib.HProg.Sem

/-- The World's device memory as bytes per buffer; a buffer that was never made
    or was freed reads empty. -/
def devMem (d : Dev) : DevMem := fun i => ((d.bufs[i]?).join).getD ByteArray.empty

/-- Buffer `b` replaced. -/
def setBuf (m : DevMem) (b : Nat) (x : ByteArray) : DevMem := fun i => if i = b then x else m i

/-- Several buffers replaced, in order. -/
def writeAll (m : DevMem) : List (Nat × ByteArray) → DevMem
  | [] => m
  | (b, x) :: ws => writeAll (setBuf m b x) ws

theorem writeAll_other (b : Nat) : ∀ (ws : List (Nat × ByteArray)) (m : DevMem),
    b ∉ ws.map Prod.fst → writeAll m ws b = m b
  | [], _, _ => rfl
  | (c, x) :: ws, m, h => by
      simp only [List.map_cons, List.mem_cons, not_or] at h
      rw [writeAll, writeAll_other b ws _ h.2, setBuf, if_neg h.1]

/-- A buffer that is written ends up with what was written last, whatever it
    held before. -/
theorem writeAll_written (b : Nat) : ∀ (ws : List (Nat × ByteArray)) (m m' : DevMem),
    b ∈ ws.map Prod.fst → writeAll m ws b = writeAll m' ws b
  | [], _, _, h => by simp at h
  | (c, x) :: ws, m, m', h => by
      simp only [writeAll]
      by_cases hb : b ∈ ws.map Prod.fst
      · exact writeAll_written b ws _ _ hb
      · rw [writeAll_other b ws _ hb, writeAll_other b ws _ hb]
        simp only [List.map_cons, List.mem_cons] at h
        rcases h with rfl | h
        · simp [setBuf]
        · exact absurd h hb

theorem map_fst_zip_sub {ids : List Nat} {outs : List ByteArray} {b : Nat}
    (h : b ∈ (ids.zip outs).map Prod.fst) : b ∈ ids := by
  induction ids generalizing outs with
  | nil => simp at h
  | cons i ids ih =>
      cases outs with
      | nil => simp at h
      | cons o outs =>
          simp only [List.zip_cons_cons, List.map_cons, List.mem_cons] at h ⊢
          rcases h with h | h
          · exact Or.inl h
          · exact Or.inr (ih h)

/-- Whether an oracle's answer keeps every bound buffer's size. -/
def sizesKept (outs ins : List ByteArray) : Bool :=
  outs.length != ins.length || (outs.zip ins).any (fun (o, i) => o.size != i.size)

theorem Launch.stores_fst {l : Launch} {ids : List Nat} {outs : List ByteArray} {b : Nat}
    (h : b ∈ (l.stores ids outs).map Prod.fst) : b ∈ ids.filter l.writes.contains := by
  obtain ⟨p, hp, rfl⟩ := List.mem_map.mp h
  obtain ⟨hz, hw⟩ := List.mem_filter.mp hp
  exact List.mem_filter.mpr ⟨map_fst_zip_sub (List.mem_map_of_mem hz), hw⟩

/-- **A launch, as a step**: the kernel oracle's answer for the buffers bound
    to it that it writes; it reads every one. An answer that would change a
    size is one the World refuses; the step leaves memory alone there. -/
def launchStep (kernel : Launch → List ByteArray → List ByteArray) (l : Launch)
    (ids : List Nat) : BufStep where
  run m :=
    let ins := ids.map m
    let outs := kernel l ins
    if sizesKept outs ins then m else writeAll m (l.stores ids outs)
  reads := ids
  writes := ids.filter l.writes.contains
  frame m b hb := by
    dsimp only
    split
    · rfl
    · exact writeAll_other b _ m (fun h => hb (Launch.stores_fst h))
  local_ m m' hr _ b hb := by
    dsimp only
    have hins : ids.map m = ids.map m' := List.map_congr_left hr
    rw [hins]
    have hbi : b ∈ ids := (List.mem_filter.mp hb).1
    split
    · exact hr b hbi
    · by_cases hw : b ∈ (l.stores ids (kernel l (ids.map m'))).map Prod.fst
      · exact writeAll_written b _ m m' hw
      · rw [writeAll_other b _ m hw, writeAll_other b _ m' hw]; exact hr b hbi

/-- **A vendor call, as a step**: its output gets the oracle's answer for its
    inputs, where that keeps the output's size. It reads its inputs and its
    output (a `beta` term reads the output) and writes the output. -/
def vendorStep (vendor : VendorCall → List ByteArray → ByteArray) (c : VendorCall)
    (ins : List Nat) (out : Nat) : BufStep where
  run m :=
    let new := vendor c (ins.map m)
    if new.size != (m out).size then m else setBuf m out new
  reads := out :: ins
  writes := [out]
  frame m b hb := by
    dsimp only
    split
    · rfl
    · simp only [List.mem_singleton] at hb
      simp [setBuf, hb]
  local_ m m' hr hw b hb := by
    dsimp only
    simp only [List.mem_singleton] at hb
    subst hb
    have hins : ins.map m = ins.map m' := List.map_congr_left (fun i hi => hr i (List.mem_cons_of_mem _ hi))
    have hout : m b = m' b := hr b (List.mem_cons_self ..)
    rw [hins, hout]
    split
    · exact hout
    · simp [setBuf]

theorem devMem_put (d : Dev) (id : Nat) (o : ByteArray) (h : id < d.bufs.size) :
    devMem (d.put id (some o)) = setBuf (devMem d) id o := by
  funext i
  simp only [devMem, Dev.put, setBuf]
  by_cases hi : i = id
  · subst hi; simp [Array.getElem?_setIfInBounds, h]
  · simp [Array.getElem?_setIfInBounds, Ne.symm hi, hi]

theorem devMem_putAll : ∀ (ws : List (Nat × ByteArray)) (d : Dev),
    (∀ p ∈ ws, p.1 < d.bufs.size) →
    devMem (ws.foldl (fun d (id, o) => d.put id (some o)) d) = writeAll (devMem d) ws
  | [], _, _ => rfl
  | (id, o) :: ws, d, h => by
      simp only [List.foldl_cons, writeAll]
      rw [devMem_putAll ws (d.put id (some o))
        (fun p hp => by rw [Dev.put_bufs_size]; exact h p (List.mem_cons_of_mem _ hp)),
        devMem_put d id o (h (id, o) (List.mem_cons_self ..))]

theorem mapM_get (d : Dev) : ∀ (ids : List Nat) (ins : List ByteArray),
    ids.mapM (fun i => d.get? (Int.ofNat i)) = some ins →
    ins = ids.map (devMem d) ∧ ∀ i ∈ ids, i < d.bufs.size
  | [], ins, h => by simp at h; subst h; simp
  | i :: ids, ins, h => by
      simp only [List.mapM_cons] at h
      obtain ⟨b, hg, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨bs, hr, h⟩ := Option.bind_eq_some_iff.mp h
      simp only [Option.pure_def, Option.some.injEq] at h
      subst h
      obtain ⟨h1, h2⟩ := mapM_get d ids bs hr
      have hb : (d.bufs[i]?).join = some b := by
        simpa [Dev.get?] using hg
      have hlt : i < d.bufs.size := by
        cases e : d.bufs[i]? with
        | none => rw [e] at hb; cases hb
        | some _ => exact (Array.getElem?_eq_some_iff.mp e).1
      refine ⟨?_, ?_⟩
      · simp [h1, devMem, hb]
      · intro j hj
        rcases List.mem_cons.mp hj with rfl | hj
        · exact hlt
        · exact h2 j hj

/-- **The World's launch performs its step**: where `runLaunch` succeeds, the
    device memory it leaves is what `launchStep` makes of the memory before. -/
theorem runLaunch_mem (d d' : Dev) (kernel : Launch → List ByteArray → List ByteArray)
    (l : Launch) (ids : List Nat) (h : d.runLaunch kernel l ids = some d') :
    devMem d' = (launchStep kernel l ids).run (devMem d) := by
  unfold Dev.runLaunch at h
  obtain ⟨ins, hins, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨hmap, hlt⟩ := mapM_get d ids ins hins
  subst hmap
  simp only [launchStep]
  dsimp only at h
  split at h
  · cases h
  · rename_i hk
    simp only [Option.some.injEq] at h
    subst h
    simp only [sizesKept]
    rw [if_neg hk]
    exact devMem_putAll _ d (fun p hp => hlt p.1 (List.mem_filter.mp (Launch.stores_fst (List.mem_map_of_mem hp))).1)

/-- **The World's vendor call performs its step**: where `runVendor`
    succeeds, the device memory it leaves is what `vendorStep` makes of the
    memory before. -/
theorem runVendor_mem (d d' : Dev) (vendor : VendorCall → List ByteArray → ByteArray) (c : VendorCall)
    (ins : List Nat) (out : Nat) (h : d.runVendor vendor c ins out = some d') :
    devMem d' = (vendorStep vendor c ins out).run (devMem d) := by
  unfold Dev.runVendor at h
  obtain ⟨bs, hbs, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨hmap, -⟩ := mapM_get d ins bs hbs
  subst hmap
  obtain ⟨old, hold, h⟩ := Option.bind_eq_some_iff.mp h
  have hb : (d.bufs[out]?).join = some old := by simpa [Dev.get?] using hold
  have hlt : out < d.bufs.size := by
    cases e : d.bufs[out]? with
    | none => rw [e] at hb; cases hb
    | some _ => exact (Array.getElem?_eq_some_iff.mp e).1
  have hm : devMem d out = old := by simp [devMem, hb]
  simp only [vendorStep]
  dsimp only at h
  rw [hm]
  split at h
  · cases h
  · rename_i hsz
    simp only [Option.some.injEq] at h
    subst h
    rw [if_neg hsz]
    exact devMem_put d out _ hlt

end AlgorithmLib.Device
