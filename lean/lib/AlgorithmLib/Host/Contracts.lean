module
public import AlgorithmLib.Host.DevSpec
meta import AlgorithmLib.Host.DevSpec
public import AlgorithmLib.Host.Spec
meta import AlgorithmLib.Host.Spec
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Host.Contracts` — every entry point's contract, in one vocabulary

`callBits` is the one reading of the Rust entry points: for each `Ffi` callee it
says, over the argument bits and the world, what the call answers or that it
refuses. This module states, for each of them, a precondition under which it
answers, and proves it. `Pre f bits w` is that precondition and `pre_safe` the
proof; a frontend that establishes `Pre` before every call has shown that no
call it makes is misused, whichever subsystem it touches.

The vocabulary is shared across subsystems:

* memory: `CStr` (a readable C string), `Slot8` (an eight-byte slot the
  program may write), `Load8`, `Readable`, `Writable`;
* context arguments: `CtxOk`, `LmdbOk`, `GpuOk`, `WinOk`, `ThreadOk` --- the
  handle is either zero (the call answers with an error code) or the live one;
* the device: `Ready` from `DevSpec` (the race vocabulary), `VendorKeeps`,
  `KeepsSize`, `SgemvOk`, `GemmOk`, `GraphRuns`, `ElapsedOk`, `PinnedRoom`;
* the counts a call copies: `readCount`, `readCountPtr`, `scanRows`, and
  `batchedRun`, the batched GEMM's own walk over its pointer arrays.

A precondition is weak where the entry point is forgiving: most calls given a
zero handle answer with an error code, and their preconditions ask nothing
more of them then.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.DevSpec AlgorithmLib.HProg.Static

namespace AlgorithmLib.HProg.Contracts

/-- The bytes at `a` up to the first zero can be read as a path. -/
def CStr (m : Mem) (a : UInt64) : Prop := (readCStr m a).isSome = true

/-- The bytes at `a` can be scanned for a path: a zero comes before the region
    ends, or `pathMax` bytes are there to scan. -/
def PathOk (m : Mem) (a : UInt64) : Prop := (readPath m a).isSome = true

/-- A C string is a path the file shims can scan. -/
theorem pathOk_of_cstr {m : Mem} {a : UInt64} (h : CStr m a) : PathOk m a := by
  unfold CStr readCStr at h
  unfold PathOk readPath
  cases hd : decodeAddr a with
  | none => rw [hd] at h; cases h
  | some p =>
    obtain ⟨r, off⟩ := p
    rw [hd] at h
    simp only [Option.bind_eq_bind, Option.bind_some] at h ⊢
    split
    · rfl
    · next hn =>
      split
      · next hlt =>
        rw [Nat.min_eq_left (Nat.le_of_lt hlt)] at hn
        rw [hn] at h; cases h
      · rfl

/-- Room for `pathMax` bytes is room to scan a path, whatever they hold. -/
theorem pathOk_of_room {m : Mem} {a : UInt64} {r : Region} {off : Nat} (hd : decodeAddr a = some (r, off))
    (hs : off + pathMax ≤ (m.region r).size) : PathOk m a := by
  unfold PathOk readPath
  rw [hd]
  simp only [Option.bind_eq_bind, Option.bind_some]
  split
  · rfl
  · rw [if_neg (by omega)]; rfl

/-- An eight-byte slot the program may write. -/
def Slot8 (m : Mem) (a : UInt64) : Prop := ∀ v, (m.store a 8 v).isSome = true

/-- An eight-byte slot the program may read. -/
def Load8 (m : Mem) (a : UInt64) : Prop := (m.load a 8).isSome = true

theorem writable_ne_none {m : Mem} {a : UInt64} {n : Nat} {src : ByteArray} (h : Writable m a n)
    (hs : src.size = n) (e : copyIn m a src = none) : False := by
  have := h src hs; rw [e] at this; cases this
theorem readable_ne_none {m : Mem} {a : UInt64} {n : Nat} (h : Readable m a n)
    (e : readBytes m a n = none) : False := by
  unfold Readable at h; rw [e] at h; cases h
theorem cstr_ne_none {m : Mem} {a : UInt64} (h : CStr m a) (e : readCStr m a = none) : False := by
  unfold CStr at h; rw [e] at h; cases h
theorem slot8_ne_none {m : Mem} {a v : UInt64} (h : Slot8 m a) (e : m.store a 8 v = none) : False := by
  have := h v; rw [e] at this; cases this
theorem load8_ne_none {m : Mem} {a : UInt64} (h : Load8 m a) (e : m.load a 8 = none) : False := by
  unfold Load8 at h; rw [e] at h; cases h

/-- A bind answers when its first part does and the rest does on what it answers. -/
theorem bind_some {α β : Type} {x : Option α} {f : α → Option β} (h : x.isSome = true)
    (hf : ∀ a, x = some a → (f a).isSome = true) : (x >>= f).isSome = true := by
  cases x with
  | none => cases h
  | some a => exact hf a rfl

/-- The same, for a bind already unfolded to `Option.bind`. -/
theorem obind_some {α β : Type} {x : Option α} {f : α → Option β} (h : x.isSome = true)
    (hf : ∀ a, x = some a → (f a).isSome = true) : (x.bind f).isSome = true := bind_some h hf

theorem sinf_safe (x : UInt64) (w : World) : (ffiSinf [x] w).isSome = true := rfl
theorem cosf_safe (x : UInt64) (w : World) : (ffiCosf [x] w).isSome = true := rfl
theorem powf_safe (b e : UInt64) (w : World) : (ffiPowf [b, e] w).isSome = true := rfl

/-- How many bytes a file read copies out of `c`. -/
def readCount (c : ByteArray) (fileOff size : UInt64) : Nat :=
  min (if size == 0 then c.size - min fileOff.toNat c.size else size.toNat) (c.size - min fileOff.toNat c.size)

theorem fileRead_safe {w : World} {base pathOff dstOff fileOff size : UInt64}
    (hp : PathOk w.mem (base + pathOff))
    (hw : ∀ path c, readPath w.mem (base + pathOff) = some path → w.fs.get path = some c →
      Writable w.mem (base + dstOff) (readCount c fileOff size)) :
    (ffiFileRead [base, pathOff, dstOff, fileOff, size] w).isSome = true := by
  unfold ffiFileRead; dsimp only
  refine bind_some hp fun path e => ?_
  dsimp only
  split
  · rfl
  · next c ec =>
      have ec' : w.fs.get path = some c := by split at ec <;> simp_all
      exact bind_some (hw path c e ec' _ (by simp [readCount])) fun _ _ => rfl

theorem fileWrite_safe {w : World} {base pathOff srcOff fileOff size : UInt64}
    (hp : PathOk w.mem (base + pathOff))
    (hs : if size = 0 then CStr w.mem (base + srcOff) else Readable w.mem (base + srcOff) size.toNat) :
    (ffiFileWrite [base, pathOff, srcOff, fileOff, size] w).isSome = true := by
  unfold ffiFileWrite readCStrAt; dsimp only
  refine bind_some hp fun path _ => ?_
  dsimp only
  split
  · rfl
  split
  · next h0 =>
    rw [if_pos (by simpa using h0)] at hs
    exact bind_some (by simpa using hs) fun _ _ => rfl
  · next h0 =>
    rw [if_neg (by simpa using h0)] at hs
    exact bind_some hs fun _ _ => rfl

/-- How many bytes a read through raw pointers copies out of `c`. -/
def readCountPtr (c : ByteArray) (fileOff size : UInt64) : Nat :=
  min size.toNat (c.size - min fileOff.toNat c.size)

theorem fileReadToPtr_safe {w : World} {pathPtr dstPtr fileOff size : UInt64}
    (hp : ¬ asI64 size ≤ 0 → PathOk w.mem pathPtr)
    (hw : ∀ path c, readPath w.mem pathPtr = some path → w.fs.get path = some c →
      Writable w.mem dstPtr (readCountPtr c fileOff size)) :
    (ffiFileReadToPtr [pathPtr, dstPtr, fileOff, size] w).isSome = true := by
  unfold ffiFileReadToPtr; dsimp only
  split
  · rfl
  · next hsz =>
    refine bind_some (hp hsz) fun path e => ?_
    dsimp only
    split
    · rfl
    · next c ec =>
        have ec' : w.fs.get path = some c := by split at ec <;> simp_all
        exact bind_some (hw path c e ec' _ (by simp [readCountPtr])) fun _ _ => rfl

theorem fileWriteFromPtr_safe {w : World} {pathPtr srcPtr fileOff size : UInt64}
    (hp : ¬ (asI64 size ≤ 0 ∨ asI64 fileOff < 0) → PathOk w.mem pathPtr ∧ Readable w.mem srcPtr size.toNat) :
    (ffiFileWriteFromPtr [pathPtr, srcPtr, fileOff, size] w).isSome = true := by
  unfold ffiFileWriteFromPtr; dsimp only
  split
  · rfl
  · next hsz =>
    have h := hp (by simpa using hsz)
    exact bind_some h.1 fun _ _ => by
      try dsimp only
      split
      · rfl
      · exact bind_some h.2 fun _ _ => rfl

theorem stdinReadline_safe {w : World} {base dstOff maxLen : UInt64}
    (hw : ∀ line rest, w.stdin = line :: rest → ¬ asI64 maxLen ≤ 0 →
      ∀ src : ByteArray, src.size = min line.size (maxLen.toNat - 1) →
        ∃ m, copyIn w.mem (base + dstOff) src = some m ∧
          (m.store (base + dstOff + UInt64.ofNat src.size) 1 0).isSome = true) :
    (ffiStdinReadline [base, dstOff, maxLen] w).isSome = true := by
  unfold ffiStdinReadline; dsimp only
  split
  · rfl
  · next hm =>
    split
    · rfl
    · next line rest hs =>
      obtain ⟨m, hc, hst⟩ := hw line rest hs hm _ (size_toByteArray_map_range _ _)
      rw [size_toByteArray_map_range] at hst
      exact bind_some (by rw [hc]; rfl) fun m' e => by
        rw [hc] at e; cases e
        exact bind_some hst fun _ _ => rfl

theorem stdoutWrite_safe {w : World} {base srcOff size : UInt64}
    (hr : ¬ asI64 size < 0 → Readable w.mem (base + srcOff) size.toNat) :
    (ffiStdoutWrite [base, srcOff, size] w).isSome = true := by
  unfold ffiStdoutWrite; dsimp only
  split
  · rfl
  · next h => exact bind_some (hr h) fun _ _ => rfl

theorem htInit_safe {w : World} {slot : UInt64} (h : Slot8 w.mem slot) :
    (ffiHtInit [slot] w).isSome = true := by
  unfold ffiHtInit; dsimp only; exact bind_some (h _) fun _ _ => rfl

theorem htCleanup_safe {w : World} {slot : UInt64} (h : Slot8 w.mem slot) :
    (ffiHtCleanup [slot] w).isSome = true := by
  unfold ffiHtCleanup; dsimp only; exact bind_some (h _) fun _ _ => rfl

theorem htCreate_safe (ctx : UInt64) (w : World) : (ffiHtCreate [ctx] w).isSome = true := by
  unfold ffiHtCreate; dsimp only; split <;> rfl

theorem htCount_safe (ctx : UInt64) (w : World) : (ffiHtCount [ctx] w).isSome = true := by
  unfold ffiHtCount; dsimp only; split <;> rfl

theorem htLookup_safe {w : World} {ctx keyPtr keyLen resultPtr : UInt64}
    (hk : Readable w.mem keyPtr keyLen.toNat)
    (hw : ∀ key val, readBytes w.mem keyPtr keyLen.toNat = some key → w.ht.get key = some val →
      Writable w.mem resultPtr val.size) :
    (ffiHtLookup [ctx, keyPtr, keyLen, resultPtr] w).isSome = true := by
  unfold ffiHtLookup; dsimp only
  refine bind_some hk fun key ek => ?_
  dsimp only
  split
  · rfl
  · cases ev : w.ht.get key with
    | none => rfl
    | some val => exact bind_some (hw key val ek ev _ rfl) fun _ _ => rfl

theorem htInsert_safe {w : World} {ctx keyPtr keyLen valPtr valLen : UInt64}
    (hk : Readable w.mem keyPtr keyLen.toNat) (hv : Readable w.mem valPtr valLen.toNat) :
    (ffiHtInsert [ctx, keyPtr, keyLen, valPtr, valLen] w).isSome = true := by
  unfold ffiHtInsert; dsimp only
  exact bind_some hk fun _ _ => bind_some hv fun _ _ => by split <;> rfl

/-- A counter the table holds is at least eight bytes, as the runtime indexes it. -/
def HtCounters (h : Ht) : Prop := ∀ k v, h.get k = some v → 8 ≤ v.size

theorem htIncrement_safe {w : World} {ctx keyPtr keyLen addend : UInt64}
    (hk : Readable w.mem keyPtr keyLen.toNat) (hc : HtCounters w.ht) :
    (ffiHtIncrement [ctx, keyPtr, keyLen, addend] w).isSome = true := by
  unfold ffiHtIncrement; dsimp only
  refine bind_some hk fun key _ => ?_
  dsimp only
  split
  · rfl
  · cases ev : w.ht.get key with
    | none => rfl
    | some v =>
        have := hc key v ev
        dsimp only
        rw [if_neg (by omega)]; rfl

theorem htGetEntry_safe {w : World} {ctx index keyOut valOut : UInt64}
    (hw : ∀ k v, w.ht.entries[index.toNat]? = some (k, v) →
      ∃ m, copyIn w.mem keyOut k = some m ∧ (copyIn m valOut v).isSome = true) :
    (ffiHtGetEntry [ctx, index, keyOut, valOut] w).isSome = true := by
  unfold ffiHtGetEntry; dsimp only
  split
  · rfl
  · split
    · rfl
    · next k v he =>
      obtain ⟨m, hm, hm2⟩ := hw k v he
      exact bind_some (by rw [hm]; rfl) fun m' e => by
        rw [hm] at e; cases e
        exact bind_some hm2 fun _ _ => rfl

-- LMDB

/-- The LMDB context argument is null or the live context. -/
def LmdbOk (w : World) (ctx : UInt64) : Prop := ctx = 0 ∨ (ctx = lmdbCtx ∧ w.lmdb.live = true)

theorem lmdbCtxOk_of {w : World} {ctx : UInt64} (h : LmdbOk w ctx) : ∃ b, lmdbCtxOk w ctx = some b := by
  unfold lmdbCtxOk
  rcases h with rfl | ⟨rfl, hl⟩
  · exact ⟨false, rfl⟩
  · exact ⟨true, by simp [hl, show lmdbCtx ≠ 0 by decide]⟩

theorem lmdbInit_safe {w : World} {slot : UInt64} (hl : w.lmdb.live = false) (h : Slot8 w.mem slot) :
    (ffiLmdbInit [slot] w).isSome = true := by
  unfold ffiLmdbInit; dsimp only
  rw [if_neg (by simp [hl])]
  exact bind_some (h _) fun _ _ => rfl

/-- The slot a cleanup reads holds null or the live context. -/
def LmdbSlotOk (w : World) (slot : UInt64) : Prop :=
  ∀ c, w.mem.load slot 8 = some c → c = 0 ∨ (c = lmdbCtx ∧ w.lmdb.live = true)

theorem lmdbCleanup_safe {w : World} {slot : UInt64} (hr : Load8 w.mem slot) (hc : LmdbSlotOk w slot)
    (hs : Slot8 w.mem slot) :
    (ffiLmdbCleanup [slot] w).isSome = true := by
  unfold ffiLmdbCleanup; dsimp only
  refine bind_some hr fun c e => ?_
  rcases hc c e with rfl | ⟨rfl, hl⟩
  · rfl
  · simp only [show (lmdbCtx == 0) = false by decide, hl, beq_self_eq_true, Bool.and_self, if_true,
      Bool.false_eq_true, if_false]
    exact bind_some (hs _) fun _ _ => rfl

theorem lmdbOpen_safe {w : World} {ctx pathPtr x : UInt64} (hc : LmdbOk w ctx)
    (hp : ctx ≠ 0 → CStr w.mem pathPtr)
    (hnew : ∀ p, readCStr w.mem pathPtr = some p → w.lmdb.envs.any (·.path == p) = false) :
    (ffiLmdbOpen [ctx, pathPtr, x] w).isSome = true := by
  unfold ffiLmdbOpen; dsimp only
  obtain ⟨b, hb⟩ := lmdbCtxOk_of hc
  rw [hb]
  cases b with
  | false => rfl
  | true =>
      dsimp only
      have h0 : ctx ≠ 0 := by
        intro h0; subst h0; simp [lmdbCtxOk] at hb
      refine bind_some (hp h0) fun p e => ?_
      dsimp only
      split
      · rfl
      · rw [if_neg (by simp [hnew p e])]; rfl

theorem lmdbBeginWriteTxn_safe {w : World} {ctx handle : UInt64} (hc : LmdbOk w ctx) :
    (ffiLmdbBeginWriteTxn [ctx, handle] w).isSome = true := by
  unfold ffiLmdbBeginWriteTxn; dsimp only
  obtain ⟨b, hb⟩ := lmdbCtxOk_of hc
  rw [hb]
  cases b with
  | false => rfl
  | true => dsimp only; split <;> rfl

theorem lmdbPut_safe {w : World} {ctx handle keyPtr keyLen valPtr valLen : UInt64} (hc : LmdbOk w ctx)
    (hr : ctx ≠ 0 → Readable w.mem keyPtr (asI32 keyLen).toNat ∧ Readable w.mem valPtr (asI32 valLen).toNat) :
    (ffiLmdbPut [ctx, handle, keyPtr, keyLen, valPtr, valLen] w).isSome = true := by
  unfold ffiLmdbPut; dsimp only
  split
  · rfl
  · obtain ⟨b, hb⟩ := lmdbCtxOk_of hc
    rw [hb]
    cases b with
    | false => rfl
    | true =>
        dsimp only
        have h0 : ctx ≠ 0 := by
          intro h0; subst h0; simp [lmdbCtxOk] at hb
        split
        · rfl
        · exact bind_some (hr h0).1 fun _ _ => bind_some (hr h0).2 fun _ _ => by
            split
            · rfl
            · split <;> rfl

theorem lmdbCommitWriteTxn_safe {w : World} {ctx handle : UInt64} (hc : LmdbOk w ctx) :
    (ffiLmdbCommitWriteTxn [ctx, handle] w).isSome = true := by
  unfold ffiLmdbCommitWriteTxn; dsimp only
  obtain ⟨b, hb⟩ := lmdbCtxOk_of hc
  rw [hb]
  cases b with
  | false => rfl
  | true => dsimp only; split <;> (try split) <;> rfl

/-- A scan: the start key, when there is one, can be read and is one LMDB
    accepts, and there is room for as many bytes as the caller says. -/
theorem lmdbCursorScan_safe {w : World} {ctx handle keyPtr keyLen maxEntries resultPtr cap : UInt64}
    (hc : LmdbOk w ctx)
    (hw : ctx ≠ 0 → ∀ n, n ≤ cap.toNat → Writable w.mem resultPtr n)
    (hkey : ctx ≠ 0 → 0 < asI32 keyLen → ∃ k, readBytes w.mem keyPtr (asI32 keyLen).toNat = some k ∧
      lmdbKeyOk k = true) :
    (ffiLmdbCursorScan [ctx, handle, keyPtr, keyLen, maxEntries, resultPtr, cap] w).isSome = true := by
  unfold ffiLmdbCursorScan; dsimp only
  obtain ⟨b, hb⟩ := lmdbCtxOk_of hc
  rw [hb]
  cases b with
  | false => rfl
  | true =>
      dsimp only
      have h0 : ctx ≠ 0 := by
        intro h0; subst h0; simp [lmdbCtxOk] at hb
      split
      · rfl
      · rename_i hcap
        have hfit : ∀ rows, (lmdbScanBytes (lmdbFit (cap.toNat - 4) rows)).size ≤ cap.toNat := fun rows => by
          have := lmdbScanBytes_fit (cap.toNat - 4) rows; omega
        split
        · exact bind_some (hw h0 4 (by omega) _ (by decide)) fun _ _ => rfl
        · split
          · next hk =>
            obtain ⟨k, hk', hok⟩ := hkey h0 hk
            refine bind_some (by simp [hk']) fun y ey => ?_
            simp [hk'] at ey; subst ey
            rw [if_neg (by simp [hok])]
            exact bind_some (hw h0 _ (hfit _) _ rfl) fun _ _ => rfl
          · refine bind_some rfl fun y ey => ?_
            cases ey
            exact bind_some (hw h0 _ (hfit _) _ rfl) fun _ _ => rfl

-- CUDA: pinned memory, memory info, events, graphs

/-- Pinned memory has room for `size` more bytes after aligning to 64. -/
def PinnedRoom (w : World) (size : UInt64) : Prop :=
  w.mem.pinned.size + (64 - w.mem.pinned.size % 64) % 64 + size.toNat ≤ regionSpan.toNat

theorem pinnedAlloc_safe {w : World} {ctx size : UInt64} (hc : CtxOk w ctx) (hr : PinnedRoom w size) :
    (ffiCudaPinnedAlloc [ctx, size] w).isSome = true := by
  unfold ffiCudaPinnedAlloc; dsimp only
  split
  · rfl
  · simp only [cudaCtxOk_of hc, bind, Option.bind]
    split
    · rfl
    · rw [if_neg (by unfold PinnedRoom at hr; omega)]; rfl

theorem pinnedPtr_safe {w : World} {ctx id : UInt64} (hc : CtxOk w ctx) :
    (ffiCudaPinnedPtr [ctx, id] w).isSome = true := by
  unfold ffiCudaPinnedPtr; rw [devOnly_isSome]; dsimp only
  split
  · rfl
  · simp only [cudaCtxOk_of hc, bind, Option.bind]; split
    · rfl
    · split <;> rfl

theorem pinnedPtrAt_safe {w : World} {ctx id off len : UInt64} (hc : CtxOk w ctx) :
    (ffiCudaPinnedPtrAt [ctx, id, off, len] w).isSome = true := by
  unfold ffiCudaPinnedPtrAt; rw [devOnly_isSome]; dsimp only
  split
  · rfl
  · simp only [cudaCtxOk_of hc, bind, Option.bind]; split
    · rfl
    · split
      · rfl
      · split <;> rfl

theorem pinnedFree_safe {w : World} {ctx id : UInt64} (hc : CtxOk w ctx) :
    (ffiCudaPinnedFree [ctx, id] w).isSome = true := by
  unfold ffiCudaPinnedFree; dsimp only
  split
  · rfl
  · simp only [cudaCtxOk_of hc, bind, Option.bind]; split
    · rfl
    · split <;> rfl

theorem memInfoOf_isSome {w : World} {ctx : UInt64} {pick : UInt64 × UInt64 → UInt64} (hc : CtxOk w ctx) :
    (memInfoOf w ctx pick).isSome = true := by
  unfold memInfoOf; simp only [cudaCtxOk_of hc, bind, Option.bind]; split <;> rfl

theorem memInfoFree_safe {w : World} {ctx : UInt64} (hc : CtxOk w ctx) :
    (ffiCudaMemInfoFree [ctx] w).isSome = true := by
  unfold ffiCudaMemInfoFree; rw [devOnly_isSome]; exact memInfoOf_isSome hc

theorem memInfoTotal_safe {w : World} {ctx : UInt64} (hc : CtxOk w ctx) :
    (ffiCudaMemInfoTotal [ctx] w).isSome = true := by
  unfold ffiCudaMemInfoTotal; rw [devOnly_isSome]; exact memInfoOf_isSome hc

/-- Two events a timing can be taken between: either is unrecorded, or both
    were recorded outside a capture and the host has waited for both. -/
def ElapsedOk (w : World) (x y : Option (Clock × Bool)) : Prop :=
  x = none ∨ y = none ∨ ∃ cs ce, x = some (cs, false) ∧ y = some (ce, false) ∧
    w.dev.race.hostSaw cs = true ∧ w.dev.race.hostSaw ce = true

theorem eventElapsedMsBits_safe {w : World} {ctx s e : UInt64} (hc : CtxOk w ctx)
    (he : ∀ x y, w.dev.event? (asI32 s) = some x → w.dev.event? (asI32 e) = some y → ElapsedOk w x y) :
    (ffiCudaEventElapsedMsBits [ctx, s, e] w).isSome = true := by
  unfold ffiCudaEventElapsedMsBits; rw [devOnly_isSome]; dsimp only
  simp only [cudaCtxOk_of hc, bind, Option.bind]
  split
  · rfl
  · split
    · next cs ce h1 h2 =>
      rcases he _ _ h1 h2 with h | h | ⟨_, _, h, h', hs, hs'⟩
      · cases h
      · cases h
      · cases h; cases h'; simp [hs, hs']
    · rfl
    · rfl
    · next a b hov hna hnb e1 e2 =>
      rcases he _ _ e1 e2 with rfl | rfl | ⟨cs, ce, rfl, rfl, hs, hs'⟩
      · exact (hna rfl).elim
      · exact (hnb rfl).elim
      · exact (hov cs ce rfl rfl).elim
    · rfl

theorem graphBeginCapture_safe {w : World} {ctx sid : UInt64} (hc : CtxOk w ctx)
    (hcap : w.dev.capture = none) :
    (ffiCudaGraphBeginCapture [ctx, sid] w).isSome = true := by
  unfold ffiCudaGraphBeginCapture; rw [devOnly_isSome]
  simp only [cudaCtxOk_of hc, bind, Option.bind]
  split
  · rfl
  · split
    · rfl
    · split
      · rfl
      · simp [hcap, devOk]

/-- Ending a capture on the stream it began on. -/
theorem graphEndCapture_safe {w : World} {ctx sid : UInt64} (hc : CtxOk w ctx)
    (ho : ∀ p c, w.dev.party? (asI32 sid) = some p → w.dev.capture = some c → c.origin = p) :
    (ffiCudaGraphEndCapture [ctx, sid] w).isSome = true := by
  unfold ffiCudaGraphEndCapture; rw [devOnly_isSome]
  simp only [cudaCtxOk_of hc, bind, Option.bind]
  split
  · rfl
  · split
    · next p c hp hcap =>
      rw [if_neg (by simp [ho p c hp hcap])]
      split <;> rfl
    · rfl

theorem graphUpload_safe {w : World} {ctx gid sid : UInt64} (hc : CtxOk w ctx) :
    (ffiCudaGraphUpload [ctx, gid, sid] w).isSome = true := by
  unfold ffiCudaGraphUpload; rw [devOnly_isSome]
  simp only [cudaCtxOk_of hc, bind, Option.bind]
  split
  · rfl
  · split <;> rfl

theorem graphDestroy_safe {w : World} {ctx gid : UInt64} (hc : CtxOk w ctx) :
    (ffiCudaGraphDestroy [ctx, gid] w).isSome = true := by
  unfold ffiCudaGraphDestroy; rw [devOnly_isSome]; dsimp only
  split
  · rfl
  · simp only [cudaCtxOk_of hc, bind, Option.bind]; split
    · rfl
    · split <;> rfl

/-- A captured graph replays on `p`: its accesses are not a race, and every node
    runs on the buffers it names. -/
def GraphRuns (w : World) (ops : Array DevOp) (p : Nat) : Prop :=
  ∃ r, w.dev.race.op p (graphAccess ops).1 (graphAccess ops).2 = some r ∧
    (ops.foldlM (fun (d : Dev) (op : DevOp) => match op with
      | .launch l ids => d.runLaunch w.kernel l ids
      | .vendor c ins out => d.runVendor w.vendor c ins out) { w.dev with race := r }).isSome = true

theorem graphLaunch_safe {w : World} {ctx gid sid : UInt64} (hc : CtxOk w ctx)
    (hcap : w.dev.capture = none)
    (hg : ∀ ops p, w.dev.graph? (asI32 gid) = some ops → w.dev.party? (asI32 sid) = some p → GraphRuns w ops p) :
    (ffiCudaGraphLaunch [ctx, gid, sid] w).isSome = true := by
  unfold ffiCudaGraphLaunch; rw [devOnly_isSome]
  simp only [cudaCtxOk_of hc, bind, Option.bind]
  split
  · rfl
  · split
    · next ops p hg1 hp =>
      rw [if_neg (by simp [capturing_none hcap])]
      obtain ⟨r, hr, hf⟩ := hg ops p hg1 hp
      simp only [hr]
      exact obind_some hf fun _ _ => rfl
    · rfl

-- CUDA: asynchronous copies and cuBLAS

/-- An operation by `p` reading `rs` and writing `ws` is not a race. -/
def OpReady (r : Race) (p : Nat) (rs ws : List Nat) : Prop :=
  (∀ b ∈ rs ++ ws, Below (wAcc r b) (know r p)) ∧ (∀ b ∈ ws, Below (rAcc r b) (know r p))

theorem OpReady.op {r : Race} {p : Nat} {rs ws : List Nat} (h : OpReady r p rs ws) :
    (r.op p rs ws).isSome = true := by rw [op_some h.1 h.2]; rfl

/-- The vendor oracle answers each routine at the length of its output, which
    is the last buffer it is handed. -/
def VendorKeeps (v : VendorCall → List ByteArray → ByteArray) : Prop :=
  ∀ c bs b, bs.getLast? = some b → (v c bs).size = b.size

theorem get?_toNat {d : Dev} {i : Int} {b : ByteArray} (h : d.get? i = some b) :
    d.get? (Int.ofNat i.toNat) = some b := by
  have hi : 0 ≤ i := by
    rcases Int.lt_or_le i 0 with hn | hn
    · simp [Dev.get?, hn] at h
    · exact hn
  rw [Int.ofNat_eq_natCast, Int.toNat_of_nonneg hi]; exact h

theorem runVendor_isSome {d : Dev} {v : VendorCall → List ByteArray → ByteArray} (hv : VendorKeeps v)
    {c : VendorCall} {ins : List Nat} {out : Nat} {bs : List ByteArray} {o : ByteArray}
    (hins : ins.mapM (fun i => d.get? (Int.ofNat i)) = some bs) (hout : d.get? (Int.ofNat out) = some o)
    (hl : bs.getLast? = some o) : (d.runVendor v c ins out).isSome = true := by
  unfold Dev.runVendor
  simp only [hins, hout, bind, Option.bind, hv c bs o hl, bne_self_eq_false, Bool.false_eq_true, if_false]
  rfl

/-- A vendor routine on `p`, run at once with nothing captured. -/
theorem devOp_vendor_isSome {d : Dev} {w : World} {p : Nat} {c : VendorCall} {ins : List Nat} {out : Nat}
    {rs ws : List Nat} (hcap : d.capture = none) (hr : OpReady d.race p rs ws) (hv : VendorKeeps w.vendor)
    {bs : List ByteArray} {o : ByteArray}
    (hins : ins.mapM (fun i => d.get? (Int.ofNat i)) = some bs) (hout : d.get? (Int.ofNat out) = some o)
    (hl : bs.getLast? = some o) : (d.devOp w p (.vendor c ins out) rs ws).isSome = true := by
  unfold Dev.devOp Dev.devOp.run
  simp only [hcap]
  obtain ⟨r, e⟩ := Option.isSome_iff_exists.mp hr.op
  simp only [e, bind, Option.bind]
  exact runVendor_isSome (d := { d with race := r, capture := none }) (c := c) hv hins hout hl

theorem ffiCudaUploadAsyncAt_safe {w : World} {ctx buf off src size sid : UInt64} (hc : CtxOk w ctx)
    (hcap : w.dev.capture = none)
    (h : ∀ p b, w.dev.party? (asI32 sid) = some p → w.dev.get? (asI32 buf) = some b →
      off.toNat + size.toNat ≤ b.size →
      Readable (w.mem.forParty p) src size.toNat ∧ Ready w.dev.race p (asI32 buf).toNat true) :
    (ffiCudaUploadAsyncAt w ctx buf off src size sid).isSome = true := by
  unfold ffiCudaUploadAsyncAt
  split
  · rfl
  · simp only [cudaCtxOk_of hc, bind, Option.bind]
    split
    · rfl
    · split
      · rfl
      · next p hp =>
        split
        · rfl
        · next b hb =>
          split
          · rfl
          · next hfit =>
            obtain ⟨hrd, hready⟩ := h p b hp hb (by omega)
            obtain ⟨bytes, hbytes⟩ := Option.isSome_iff_exists.mp hrd
            simp only [hbytes]
            unfold asyncCopy
            simp only [hcap, Option.isSome_none, Bool.false_and, Bool.false_eq_true, if_false]
            have := op_isSome hready
            simp only [if_true] at this
            obtain ⟨r, hr⟩ := Option.isSome_iff_exists.mp this
            simp [hr]

theorem uploadAsync_safe {w : World} {ctx buf src size sid : UInt64} (hc : CtxOk w ctx)
    (hcap : w.dev.capture = none)
    (h : ∀ p b, w.dev.party? (asI32 sid) = some p → w.dev.get? (asI32 buf) = some b →
      size.toNat ≤ b.size →
      Readable (w.mem.forParty p) src size.toNat ∧ Ready w.dev.race p (asI32 buf).toNat true) :
    (ffiCudaUploadAsync [ctx, buf, src, size, sid] w).isSome = true :=
  ffiCudaUploadAsyncAt_safe hc hcap fun p b hp hb hf => h p b hp hb (by simpa using hf)

theorem uploadOffsetAsync_safe {w : World} {ctx buf off src size sid : UInt64} (hc : CtxOk w ctx)
    (hcap : w.dev.capture = none)
    (h : ∀ p b, w.dev.party? (asI32 sid) = some p → w.dev.get? (asI32 buf) = some b →
      off.toNat + size.toNat ≤ b.size →
      Readable (w.mem.forParty p) src size.toNat ∧ Ready w.dev.race p (asI32 buf).toNat true) :
    (ffiCudaUploadOffsetAsync [ctx, buf, off, src, size, sid] w).isSome = true :=
  ffiCudaUploadAsyncAt_safe hc hcap h

theorem downloadAsync_safe {w : World} {ctx buf dst size sid : UInt64} (hc : CtxOk w ctx)
    (hcap : w.dev.capture = none)
    (h : ∀ p b, w.dev.party? (asI32 sid) = some p → w.dev.get? (asI32 buf) = some b →
      size.toNat ≤ b.size →
      Writable (w.mem.forParty p) dst size.toNat ∧ Ready w.dev.race p (asI32 buf).toNat false) :
    (ffiCudaDownloadAsync [ctx, buf, dst, size, sid] w).isSome = true := by
  unfold ffiCudaDownloadAsync ffiCudaDownloadAsyncAt; dsimp only
  split
  · rfl
  · simp only [cudaCtxOk_of hc, bind, Option.bind]
    split
    · rfl
    · split
      · rfl
      · next p hp =>
        split
        · rfl
        · next b hb =>
          split
          · rfl
          · next hfit =>
            obtain ⟨hwr, hready⟩ := h p b hp hb (by omega)
            unfold asyncCopy
            simp only [hcap, Option.isSome_none, Bool.false_and, Bool.false_eq_true, if_false]
            have := op_isSome hready
            simp only [Bool.false_eq_true, if_false] at this
            obtain ⟨r, hr⟩ := Option.isSome_iff_exists.mp this
            obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (hwr (b.extract 0 size.toNat) (by simp; omega))
            simp [hr, hm]

/-- Three live buffers read in order: the `mapM` a vendor routine does. -/
theorem mapM_three {d : Dev} {a x y : Int} {A X Y : ByteArray} (ha : d.get? a = some A)
    (hx : d.get? x = some X) (hy : d.get? y = some Y) :
    [a.toNat, x.toNat, y.toNat].mapM (fun i => d.get? (Int.ofNat i)) = some [A, X, Y] := by
  simp only [List.mapM_cons, List.mapM_nil, get?_toNat ha, get?_toNat hx, get?_toNat hy]; rfl

/-- An `sgemv` answers when its three buffers hold its extents and its accesses
    are not a race. -/
def SgemvOk (w : World) (p : Nat) (trans m n a x y : UInt64) : Prop :=
  ∀ A X Y, w.dev.get? (asI32 a) = some A → w.dev.get? (asI32 x) = some X → w.dev.get? (asI32 y) = some Y →
    0 < asI32 m ∧ 0 < asI32 n ∧ sgemvFits trans (asI32 m).toNat (asI32 n).toNat A X Y = true ∧
    OpReady w.dev.race p [(asI32 a).toNat, (asI32 x).toNat, (asI32 y).toNat] [(asI32 y).toNat]

theorem sgemvOn_isSome {w : World} {p : Nat} {trans m n alpha a x beta y : UInt64}
    (hcap : w.dev.capture = none) (hv : VendorKeeps w.vendor) (h : SgemvOk w p trans m n a x y) :
    (sgemvOn w p trans m n alpha a x beta y).isSome = true := by
  unfold sgemvOn
  split
  · next A X Y ha hx hy =>
    obtain ⟨hm, hn, hf, hr⟩ := h A X Y ha hx hy
    rw [if_neg (by simp [hf]; omega)]
    dsimp only
    refine bind_some (devOp_vendor_isSome hcap hr hv (mapM_three ha hx hy) (get?_toNat hy) rfl) fun _ _ => rfl
  · rfl

theorem cublasSgemv_safe {w : World} {ctx trans m n alpha a x beta y : UInt64} (hc : CtxOk w ctx)
    (hcap : w.dev.capture = none) (hv : VendorKeeps w.vendor) (h : SgemvOk w defaultParty trans m n a x y) :
    (ffiCublasSgemv [ctx, trans, m, n, alpha, a, x, beta, y] w).isSome = true := by
  unfold ffiCublasSgemv; rw [devOnly_isSome]
  simp only [cudaCtxOk_of hc, bind, Option.bind]
  split
  · rfl
  · exact sgemvOn_isSome hcap hv h

theorem cublasSgemvOnStream_safe {w : World} {ctx trans m n alpha a x beta y sid : UInt64} (hc : CtxOk w ctx)
    (hcap : w.dev.capture = none) (hv : VendorKeeps w.vendor)
    (h : ∀ p, w.dev.party? (asI32 sid) = some p → SgemvOk w p trans m n a x y) :
    (ffiCublasSgemvOnStream [ctx, trans, m, n, alpha, a, x, beta, y, sid] w).isSome = true := by
  unfold ffiCublasSgemvOnStream; rw [devOnly_isSome]
  simp only [cudaCtxOk_of hc, bind, Option.bind]
  split
  · rfl
  · split
    · rfl
    · next p hp => exact sgemvOn_isSome hcap hv (h p hp)

/-- A strided-batched GEMM answers when its three buffers hold every batch's
    extents and its accesses are not a race; `fits` is the element sizes'
    own check. -/
def GemmOk (fits : ByteArray → ByteArray → ByteArray → Bool) (w : World) (p : Nat) (a b c : UInt64) : Prop :=
  ∀ A B C, w.dev.get? (asI32 a) = some A → w.dev.get? (asI32 b) = some B → w.dev.get? (asI32 c) = some C →
    fits A B C = true ∧
    OpReady w.dev.race p [(asI32 a).toNat, (asI32 b).toNat, (asI32 c).toNat] [(asI32 c).toNat]

theorem sgemmOn_isSome {w : World} {p : Nat}
    {ta tb m n k alpha a sa b sb beta c sc batch oa ob oc la lb lc : UInt64}
    (hcap : w.dev.capture = none) (hv : VendorKeeps w.vendor)
    (h : GemmOk (sgemmFits ta tb m n k sa sb sc batch oa ob oc la lb lc) w p a b c) :
    (sgemmOn w p ta tb m n k alpha a sa b sb beta c sc batch oa ob oc la lb lc).isSome = true := by
  unfold sgemmOn
  split
  · next A B C ha hb hc =>
    obtain ⟨hf, hr⟩ := h A B C ha hb hc
    rw [if_neg (by simp [hf])]
    dsimp only
    exact bind_some (devOp_vendor_isSome hcap hr hv (mapM_three ha hb hc) (get?_toNat hc) rfl) fun _ _ => rfl
  · rfl

theorem cublasSgemm_safe {w : World}
    {ctx ta tb m n k alpha a sa b sb beta c sc batch oa ob oc la lb lc : UInt64}
    (hc : CtxOk w ctx) (hcap : w.dev.capture = none) (hv : VendorKeeps w.vendor)
    (h : GemmOk (sgemmFits ta tb m n k sa sb sc batch oa ob oc la lb lc) w defaultParty a b c) :
    (ffiCublasSgemm [ctx, ta, tb, m, n, k, alpha, a, sa, b, sb, beta, c, sc, batch, oa, ob, oc, la, lb, lc]
      w).isSome = true := by
  unfold ffiCublasSgemm; rw [devOnly_isSome]; dsimp only
  split
  · rfl
  · simp only [cudaCtxOk_of hc, bind, Option.bind]
    split
    · rfl
    · exact sgemmOn_isSome hcap hv h

theorem cublasSgemmOnStream_safe {w : World}
    {ctx ta tb m n k alpha a sa b sb beta c sc batch sid oa ob oc la lb lc : UInt64}
    (hc : CtxOk w ctx) (hcap : w.dev.capture = none) (hv : VendorKeeps w.vendor)
    (h : ∀ p, w.dev.party? (asI32 sid) = some p →
      GemmOk (sgemmFits ta tb m n k sa sb sc batch oa ob oc la lb lc) w p a b c) :
    (ffiCublasSgemmOnStream [ctx, ta, tb, m, n, k, alpha, a, sa, b, sb, beta, c, sc, batch, sid,
      oa, ob, oc, la, lb, lc] w).isSome = true := by
  unfold ffiCublasSgemmOnStream; rw [devOnly_isSome]; dsimp only
  split
  · rfl
  · simp only [cudaCtxOk_of hc, bind, Option.bind]
    split
    · rfl
    · split
      · rfl
      · next p hp => exact sgemmOn_isSome hcap hv (h p hp)

theorem cublasGemmExBf16_safe {w : World} {ctx ta tb m n k alpha a b beta c oa ob oc la lb lc : UInt64}
    (hc : CtxOk w ctx) (hcap : w.dev.capture = none) (hv : VendorKeeps w.vendor)
    (h : GemmOk (gemmBf16Fits ta tb m n k oa ob oc la lb lc) w defaultParty a b c) :
    (ffiCublasGemmExBf16 [ctx, ta, tb, m, n, k, alpha, a, b, beta, c, oa, ob, oc, la, lb, lc] w).isSome = true := by
  unfold ffiCublasGemmExBf16; rw [devOnly_isSome]; dsimp only
  split
  · rfl
  · simp only [cudaCtxOk_of hc, bind, Option.bind]
    split
    · rfl
    · split
      · next A B C ha hb hc' =>
        obtain ⟨hf, hr⟩ := h A B C ha hb hc'
        rw [if_neg (by simp [hf])]
        exact bind_some (devOp_vendor_isSome hcap hr hv (mapM_three ha hb hc') (get?_toNat hc') rfl)
          fun _ _ => rfl
      · rfl

theorem cublasGemmStridedBatchedExBf16_safe {w : World}
    {ctx ta tb m n k alpha a sa b sb beta c sc batch oa ob oc la lb lc : UInt64}
    (hc : CtxOk w ctx) (hcap : w.dev.capture = none) (hv : VendorKeeps w.vendor)
    (h : GemmOk (gemmStridedFits 2 2 4 ta tb m n k sa sb sc batch oa ob oc la lb lc) w defaultParty a b c) :
    (ffiCublasGemmStridedBatchedExBf16
      [ctx, ta, tb, m, n, k, alpha, a, sa, b, sb, beta, c, sc, batch, oa, ob, oc, la, lb, lc] w).isSome = true := by
  unfold ffiCublasGemmStridedBatchedExBf16; rw [devOnly_isSome]; dsimp only
  split
  · rfl
  · simp only [cudaCtxOk_of hc, bind, Option.bind]
    split
    · rfl
    · split
      · next A B C ha hb hc' =>
        obtain ⟨hf, hr⟩ := h A B C ha hb hc'
        rw [if_neg (by simp [hf])]
        exact bind_some (devOp_vendor_isSome hcap hr hv (mapM_three ha hb hc') (get?_toNat hc') rfl)
          fun _ _ => rfl
      · rfl

theorem cublasPtrArray_safe {w : World} {ctx arr slot src off : UInt64} (hc : CtxOk w ctx)
    (h : ∀ A, w.dev.get? (asI32 arr) = some A → Ready w.dev.race defaultParty (asI32 arr).toNat true) :
    (ffiCublasPtrArray [ctx, arr, slot, src, off] w).isSome = true := by
  unfold ffiCublasPtrArray; rw [devOnly_isSome]; dsimp only
  split
  · rfl
  · simp only [cudaCtxOk_of hc, bind, Option.bind]
    split
    · rfl
    · split
      · next S A hs ha =>
        split
        · rfl
        · obtain ⟨d, hd⟩ := Option.isSome_iff_exists.mp (syncWrite_isSome (d := w.dev)
            (overwrite A (8 * (asI32 slot).toNat)
              (le64 (w.devAddr (asI32 src).toNat + 4 * UInt64.ofNat (asI64 off).toNat))) (h A ha))
          simp [hd, devOk]
      · rfl

/-- What a pointer-array batch does once its three arrays are read, exactly as
    the contract states it: every member resolves to live buffers it stays
    inside, no member writes what another touches, and each member's routine
    runs, in order. Its checks are the contract's own, so this names them. -/
def batchedRun (w : World) (p : Nat) (ta tb m n k alpha aArr bArr cArr batch : UInt64) (beta : UInt64)
    (PA PB PC : ByteArray) : Option (Option V × Dev) := do
  let (M, N, K) := ((asI32 m).toNat, (asI32 n).toNat, (asI32 k).toNat)
  let (ra, ca) := if asI32 ta != 0 then (K, M) else (M, K)
  let (rb, cb) := if asI32 tb != 0 then (N, K) else (K, N)
  let ms ← (List.range (asI32 batch).toNat).mapM (batchMember w PA PB PC)
  let inside (x : Nat × Nat) (span : Nat) : Bool :=
    match w.dev.get? x.1 with
    | some buf => decide (x.2 % 4 = 0) && decide (x.2 + 4 * span ≤ buf.size)
    | none => false
  if !ms.all (fun (a, b, c) => inside a (colMajorSpan ra ca ra)
      && inside b (colMajorSpan rb cb rb) && inside c (colMajorSpan M N M)) then none
  else
    let outs := ms.map (·.2.2.1)
    let ins := ms.flatMap (fun (a, b, _) => [a.1, b.1])
    if !outs.Nodup || outs.any (ins.contains ·) then none
    else
      let arrs := [(asI32 aArr).toNat, (asI32 bArr).toNat, (asI32 cArr).toNat]
      let d ← ms.foldlM (fun (d : Dev) (a, b, c) =>
        d.devOp w p (.vendor ⟨"sgemmBatchedMember",
            [ta, tb, m, n, k, alpha, beta, UInt64.ofNat a.2, UInt64.ofNat b.2,
             UInt64.ofNat c.2]⟩ [a.1, b.1, c.1] c.1)
          (arrs ++ [a.1, b.1, c.1]) [c.1]) w.dev
      devOk d

theorem cublasSgemmBatchedOnStream_safe {w : World}
    {ctx ta tb m n k alpha aArr bArr beta cArr batch sid : UInt64} (hc : CtxOk w ctx)
    (h : ∀ p PA PB PC, w.dev.party? (asI32 sid) = some p → w.dev.get? (asI32 aArr) = some PA →
      w.dev.get? (asI32 bArr) = some PB → w.dev.get? (asI32 cArr) = some PC →
      (batchedRun w p ta tb m n k alpha aArr bArr cArr batch beta PA PB PC).isSome = true) :
    (ffiCublasSgemmBatchedOnStream [ctx, ta, tb, m, n, k, alpha, aArr, bArr, beta, cArr, batch, sid]
      w).isSome = true := by
  unfold ffiCublasSgemmBatchedOnStream; rw [devOnly_isSome]; dsimp only
  split
  · rfl
  · simp only [cudaCtxOk_of hc, bind, Option.bind]
    split
    · rfl
    · split
      · rfl
      · next p hp =>
        split
        · next PA PB PC ha hb hcc => exact h p PA PB PC hp ha hb hcc
        · rfl

-- wgpu

/-- The wgpu context argument is null or the live context. -/
def GpuOk (w : World) (ctx : UInt64) : Prop := ctx = 0 ∨ (ctx = gpuCtx ∧ w.gpu.live = true)

theorem gpuCtxOk_of {w : World} {ctx : UInt64} (h : GpuOk w ctx) : gpuCtxOk w ctx = some (ctx != 0) := by
  unfold gpuCtxOk
  rcases h with rfl | ⟨rfl, hl⟩
  · rfl
  · simp [hl, show gpuCtx ≠ 0 by decide]

theorem runOne_isSome (sh : Dispatch → List ByteArray → List ByteArray) (g : Wgpu)
    (d : Nat × List UInt64) : ∃ g', Wgpu.runOne sh g d = some g' := by
  unfold Wgpu.runOne
  split
  · exact ⟨_, rfl⟩
  · split <;> exact ⟨_, rfl⟩

/-- What is pending on the queue always runs. -/
theorem flush_isSome (sh : Dispatch → List ByteArray → List ByteArray) (g : Wgpu) :
    (g.flush sh).isSome = true := by
  unfold Wgpu.flush
  suffices ∀ (ds : List (Nat × List UInt64)) g, ∃ g', ds.foldlM (Wgpu.runOne sh) g = some g' by
    obtain ⟨g', h⟩ := this g.pending g; rw [h]; rfl
  intro ds
  induction ds with
  | nil => intro g; exact ⟨g, rfl⟩
  | cons d ds ih =>
      intro g
      obtain ⟨g1, h1⟩ := runOne_isSome sh g d
      obtain ⟨g2, h2⟩ := ih g1
      exact ⟨g2, by simp only [List.foldlM_cons, h1, Option.bind_eq_bind, Option.bind_some, h2]⟩

theorem gpuInit_safe {w : World} {slot : UInt64} (h : Slot8 w.mem slot) :
    (ffiGpuInit [slot] w).isSome = true := by
  unfold ffiGpuInit; dsimp only; split <;> exact bind_some (h _) fun _ _ => rfl

theorem gpuCleanup_safe {w : World} {slot : UInt64} (h : Slot8 w.mem slot) :
    (ffiGpuCleanup [slot] w).isSome = true := by
  unfold ffiGpuCleanup; dsimp only; exact bind_some (h _) fun _ _ => rfl

theorem gpuCreateBuffer_safe {w : World} {ctx size : UInt64} (hc : GpuOk w ctx) :
    (ffiGpuCreateBuffer [ctx, size] w).isSome = true := by
  unfold ffiGpuCreateBuffer; dsimp only
  split
  · rfl
  · simp only [gpuCtxOk_of hc, bind, Option.bind]; split <;> rfl

theorem gpuCreatePipeline_safe {w : World} {ctx shaderPtr bindPtr n : UInt64} (hc : GpuOk w ctx)
    (hs : ctx ≠ 0 → ¬ asI32 n < 0 → CStr w.mem shaderPtr ∧ (readBinds w.mem bindPtr (asI32 n).toNat).isSome = true) :
    (ffiGpuCreatePipeline [ctx, shaderPtr, bindPtr, n] w).isSome = true := by
  unfold ffiGpuCreatePipeline readCStrAt; dsimp only
  split
  · rfl
  · next hn =>
    simp only [gpuCtxOk_of hc, bind, Option.bind]
    split
    · rfl
    · next hnz =>
      have h0 : ctx ≠ 0 := by simpa using hnz
      obtain ⟨h1, h2⟩ := hs h0 (by simpa using hn)
      exact obind_some h1 fun _ _ => obind_some h2 fun _ _ => by split <;> rfl

/-- An upload: the source can be read where it is a whole number of words
    that fits the buffer; otherwise the call answers `-1`. -/
theorem gpuUploadAt_safe {w : World} {ctx buf src size : UInt64} (hc : GpuOk w ctx)
    (h : ∀ b, ctx ≠ 0 → w.gpu.bufs[(asI32 buf).toNat]? = some b →
      size.toNat ≤ b.size → size.toNat % 4 = 0 → Readable w.mem src size.toNat) :
    (gpuUploadAt [ctx, buf, src, size] w).isSome = true := by
  unfold gpuUploadAt; dsimp only
  split
  · rfl
  · simp only [gpuCtxOk_of hc, bind, Option.bind]
    split
    · rfl
    · next hnz =>
      have h0 : ctx ≠ 0 := by simpa using hnz
      split
      · rfl
      · next b hb =>
        split
        · rfl
        · next hn =>
          simp only [Bool.or_eq_true, decide_eq_true_eq, bne_iff_ne, ne_eq, not_or,
            Decidable.not_not] at hn
          exact obind_some (h b h0 hb (by omega) hn.2) fun _ _ => rfl

theorem gpuUpload_safe {w : World} {ctx buf src size : UInt64} (hc : GpuOk w ctx)
    (h : ∀ b, ctx ≠ 0 → w.gpu.bufs[(asI32 buf).toNat]? = some b →
      size.toNat ≤ b.size → size.toNat % 4 = 0 → Readable w.mem src size.toNat) :
    (ffiGpuUpload [ctx, buf, src, size] w).isSome = true := gpuUploadAt_safe hc h

theorem gpuUploadPtr_safe {w : World} {ctx buf src size : UInt64} (hc : GpuOk w ctx)
    (h : ∀ b, ctx ≠ 0 → w.gpu.bufs[(asI32 buf).toNat]? = some b →
      size.toNat ≤ b.size → size.toNat % 4 = 0 → Readable w.mem src size.toNat) :
    (ffiGpuUploadPtr [ctx, buf, src, size] w).isSome = true := gpuUploadAt_safe hc h

theorem gpuDispatch_safe {w : World} {ctx pipe x y z : UInt64} (hc : GpuOk w ctx) :
    (ffiGpuDispatch [ctx, pipe, x, y, z] w).isSome = true := by
  unfold ffiGpuDispatch; dsimp only
  split
  · rfl
  · simp only [gpuCtxOk_of hc, bind, Option.bind]
    split
    · rfl
    · split
      · rfl
      · exact obind_some (flush_isSome _ _) fun _ _ => rfl

theorem gpuDownload_safe {w : World} {ctx buf dst size : UInt64} (hc : GpuOk w ctx)
    (h : ∀ g b, ctx ≠ 0 → w.gpu.flush w.shader = some g → g.bufs[(asI32 buf).toNat]? = some b →
      size.toNat = b.size → size.toNat % 4 = 0 → Writable w.mem dst size.toNat) :
    (ffiGpuDownload [ctx, buf, dst, size] w).isSome = true := by
  unfold ffiGpuDownload; dsimp only
  split
  · rfl
  · simp only [gpuCtxOk_of hc, bind, Option.bind]
    split
    · rfl
    · next hnz =>
      have h0 : ctx ≠ 0 := by simpa using hnz
      split
      · rfl
      · refine obind_some (flush_isSome _ _) fun g hg => ?_
        split
        · rfl
        · next b hb =>
          split
          · rfl
          · next hn =>
            simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hn
            have hw := h g b h0 hg hb hn.1 hn.2
            rw [hn.1] at hw
            exact obind_some (hw b rfl) fun _ _ => rfl

theorem gpuDownloadPtr_safe {w : World} {ctx buf off dst size : UInt64} (hc : GpuOk w ctx)
    (h : ∀ g b, ctx ≠ 0 → w.gpu.flush w.shader = some g → g.bufs[(asI32 buf).toNat]? = some b →
      off.toNat + size.toNat ≤ b.size → off.toNat % 4 = 0 → size.toNat % 4 = 0 →
      Writable w.mem dst size.toNat) :
    (ffiGpuDownloadPtr [ctx, buf, off, dst, size] w).isSome = true := by
  unfold ffiGpuDownloadPtr; dsimp only
  split
  · rfl
  · simp only [gpuCtxOk_of hc, bind, Option.bind]
    split
    · rfl
    · next hnz =>
      have h0 : ctx ≠ 0 := by simpa using hnz
      split
      · rfl
      · refine obind_some (flush_isSome _ _) fun g hg => ?_
        split
        · rfl
        · next b hb =>
          split
          · rfl
          · next hn =>
            simp only [Bool.or_eq_true, decide_eq_true_eq, bne_iff_ne, ne_eq, not_or,
              Decidable.not_not] at hn
            exact obind_some (h g b h0 hg hb (by omega) hn.1.2 hn.2 _ (by simp; omega)) fun _ _ => rfl

-- window

/-- The window context argument is null or the live context. -/
def WinOk (w : World) (ctx : UInt64) : Prop := ctx = 0 ∨ (ctx = winCtx ∧ w.win.live = true)

theorem winCtxOk_of {w : World} {ctx : UInt64} (h : WinOk w ctx) : ∃ b, winCtxOk w ctx = some b := by
  unfold winCtxOk
  rcases h with rfl | ⟨rfl, hl⟩
  · exact ⟨false, rfl⟩
  · exact ⟨true, by simp [hl, show winCtx ≠ 0 by decide]⟩

theorem windowInit_safe {w : World} {slot : UInt64} (hl : w.win.live = false) (h : Slot8 w.mem slot) :
    (ffiWindowInit [slot] w).isSome = true := by
  unfold ffiWindowInit; dsimp only
  rw [if_neg (by simp [hl])]
  split
  · exact bind_some (h _) fun _ _ => rfl
  · exact bind_some (h _) fun _ _ => rfl

/-- The slot a window cleanup reads holds null or the live context. -/
def WinSlotOk (w : World) (slot : UInt64) : Prop :=
  ∀ c, w.mem.load slot 8 = some c → c = 0 ∨ (c = winCtx ∧ w.win.live = true)

theorem windowCleanup_safe {w : World} {slot : UInt64} (hr : Load8 w.mem slot) (hc : WinSlotOk w slot)
    (hs : Slot8 w.mem slot) :
    (ffiWindowCleanup [slot] w).isSome = true := by
  unfold ffiWindowCleanup; dsimp only
  refine bind_some hr fun c e => ?_
  rcases hc c e with rfl | ⟨rfl, hl⟩
  · rfl
  · simp only [show (winCtx == 0) = false by decide, hl, beq_self_eq_true, Bool.and_self, if_true,
      Bool.false_eq_true, if_false]
    exact bind_some (hs _) fun _ _ => rfl

theorem windowOpen_safe {w : World} {ctx width height titlePtr titleLen blitPtr blitLen : UInt64}
    (hc : WinOk w ctx)
    (hr : ctx ≠ 0 → Readable w.mem titlePtr (asI64 titleLen).toNat ∧ Readable w.mem blitPtr (asI64 blitLen).toNat) :
    (ffiWindowOpen [ctx, width, height, titlePtr, titleLen, blitPtr, blitLen] w).isSome = true := by
  unfold ffiWindowOpen; dsimp only
  split
  · rfl
  · obtain ⟨b, hb⟩ := winCtxOk_of hc
    rw [hb]
    cases b with
    | false => rfl
    | true =>
        have h0 : ctx ≠ 0 := by
          intro h0; subst h0; simp [winCtxOk] at hb
        dsimp only
        split
        · rfl
        · exact bind_some (hr h0).1 fun _ _ => bind_some (hr h0).2 fun _ _ => by split <;> rfl

theorem windowPoll_safe {w : World} {ctx eventsPtr maxEvents : UInt64} (hc : WinOk w ctx)
    (hw : ctx ≠ 0 → Writable w.pump.mem eventsPtr
      (winEventBytes (w.pump.win.pending.take (min w.pump.win.pending.length (asI32 maxEvents).toNat))).size) :
    (ffiWindowPoll [ctx, eventsPtr, maxEvents] w).isSome = true := by
  unfold ffiWindowPoll; dsimp only
  split
  · rfl
  · obtain ⟨b, hb⟩ := winCtxOk_of hc
    rw [hb]
    cases b with
    | false => rfl
    | true =>
        have h0 : ctx ≠ 0 := by
          intro h0; subst h0; simp [winCtxOk] at hb
        dsimp only
        exact bind_some (hw h0 _ rfl) fun _ _ => rfl

theorem windowPresentGpuBuffer_safe {w : World} {ctx gctx buf : UInt64} (hc : WinOk w ctx)
    (hg : ctx ≠ 0 → GpuOk w.pump gctx) :
    (ffiWindowPresentGpuBuffer [ctx, gctx, buf] w).isSome = true := by
  unfold ffiWindowPresentGpuBuffer; dsimp only
  split
  · rfl
  · obtain ⟨b, hb⟩ := winCtxOk_of hc
    rw [hb]
    cases b with
    | false => rfl
    | true =>
        have h0 : ctx ≠ 0 := by
          intro h0; subst h0; simp [winCtxOk] at hb
        dsimp only
        simp only [gpuCtxOk_of (hg h0), bind, Option.bind]
        split
        · rfl
        · next hnz =>
          have g0 : gctx ≠ 0 := by simpa using hnz
          refine obind_some (flush_isSome _ _) fun g _ => ?_
          try dsimp only
          split
          · rfl
          · split <;> rfl

-- native code and the CPU

theorem nativeLoad_safe {w : World} {src len : UInt64}
    (hr : ¬ (src = 0 ∨ asI64 len ≤ 0) → Readable w.mem src (asI64 len).toNat) :
    (ffiNativeLoad [src, len] w).isSome = true := by
  unfold ffiNativeLoad; dsimp only
  split
  · rfl
  · next h => exact bind_some (hr (by simpa using h)) fun _ _ => rfl

theorem nativeFree_safe (addr : UInt64) (w : World) : (ffiNativeFree [addr] w).isSome = true := by
  unfold ffiNativeFree; dsimp only; split <;> rfl

theorem nativeArch_safe (w : World) : (ffiNativeArch [] w).isSome = true := rfl

theorem cpuHas_safe {w : World} {name : UInt64} (hn : name ≠ 0 → CStr w.mem name) :
    (ffiCpuHas [name] w).isSome = true := by
  unfold ffiCpuHas readName; dsimp only
  refine bind_some ?_ fun _ _ => rfl
  split
  · rfl
  · next h =>
    rw [Option.isSome_map]; exact hn (by simpa using h)

theorem fileCreateDirAll_safe {w : World} {path : UInt64} (hp : PathOk w.mem path) :
    (ffiFileCreateDirAll [path] w).isSome = true := by
  unfold ffiFileCreateDirAll; dsimp only
  exact bind_some hp fun _ _ => by split <;> (try split) <;> rfl

-- threads

/-- The thread context argument is null or the live context. -/
def ThreadOk (w : World) (ctx : UInt64) : Prop := ctx = 0 ∨ (ctx = threadCtx ∧ w.thread.live = true)

theorem threadInit_safe {w : World} {slot : UInt64} (hl : w.thread.live = false) (h : Slot8 w.mem slot) :
    (ffiThreadInit [slot] w).isSome = true := by
  unfold ffiThreadInit; dsimp only
  rw [if_neg (by simp [hl])]
  exact bind_some (h _) fun _ _ => rfl

/-- The slot a thread cleanup reads holds null or the live context, read and
    written with memory unfrozen. -/
def ThreadSlotOk (w : World) (slot : UInt64) : Prop :=
  (∀ c, ({ w.mem with frozen := false } : Mem).load slot 8 = some c →
    c = 0 ∨ (c = threadCtx ∧ w.thread.live = true)) ∧
  Load8 { w.mem with frozen := false } slot ∧ Slot8 { w.mem with frozen := false } slot

theorem threadCleanup_safe {w : World} {slot : UInt64} (h : ThreadSlotOk w slot) :
    (ffiThreadCleanup [slot] w).isSome = true := by
  unfold ffiThreadCleanup; dsimp only
  refine bind_some h.2.1 fun c e => ?_
  rcases h.1 c e with rfl | ⟨rfl, hl⟩
  · rfl
  · simp only [show (threadCtx == 0) = false by decide, hl, beq_self_eq_true, Bool.and_self, if_true,
      Bool.false_eq_true, if_false]
    exact bind_some (h.2.2 _) fun _ _ => rfl

theorem threadJoin_safe {w : World} {ctx h : UInt64} (hc : ThreadOk w ctx) :
    (ffiThreadJoin [ctx, h] w).isSome = true := by
  unfold ffiThreadJoin; dsimp only
  unfold threadCtxOk
  rcases hc with rfl | ⟨rfl, hl⟩
  · rfl
  · simp only [show (threadCtx == 0) = false by decide, hl, beq_self_eq_true, Bool.and_self, if_true,
      Bool.false_eq_true, if_false]
    split <;> rfl

/-- **When each entry point answers.** Stated over the argument bits and the
    world, in the vocabulary above. A call with the wrong number of arguments has
    none, and neither has `threadSpawn`, whose callee is a local function the
    program logic answers for. -/
def Pre : IR.Ffi → List UInt64 → World → Prop
  | .fileRead => fun bits w => match bits with
    | [base, pathOff, dstOff, fileOff, size] => (PathOk w.mem (base + pathOff)) ∧ (∀ path c, readPath w.mem (base + pathOff) = some path → w.fs.get path = some c → Writable w.mem (base + dstOff) (readCount c fileOff size))
    | _ => False
  | .fileWrite => fun bits w => match bits with
    | [base, pathOff, srcOff, _fileOff, size] => (PathOk w.mem (base + pathOff)) ∧ (if size = 0 then CStr w.mem (base + srcOff) else Readable w.mem (base + srcOff) size.toNat)
    | _ => False
  | .fileReadToPtr => fun bits w => match bits with
    | [pathPtr, dstPtr, fileOff, size] => (¬ asI64 size ≤ 0 → PathOk w.mem pathPtr) ∧ (∀ path c, readPath w.mem pathPtr = some path → w.fs.get path = some c → Writable w.mem dstPtr (readCountPtr c fileOff size))
    | _ => False
  | .fileWriteFromPtr => fun bits w => match bits with
    | [pathPtr, srcPtr, fileOff, size] => (¬ (asI64 size ≤ 0 ∨ asI64 fileOff < 0) → PathOk w.mem pathPtr ∧ Readable w.mem srcPtr size.toNat)
    | _ => False
  | .stdinReadline => fun bits w => match bits with
    | [base, dstOff, maxLen] => (∀ line rest, w.stdin = line :: rest → ¬ asI64 maxLen ≤ 0 → ∀ src : ByteArray, src.size = min line.size (maxLen.toNat - 1) → ∃ m, copyIn w.mem (base + dstOff) src = some m ∧ (m.store (base + dstOff + UInt64.ofNat src.size) 1 0).isSome = true)
    | _ => False
  | .stdoutWrite => fun bits w => match bits with
    | [base, srcOff, size] => (¬ asI64 size < 0 → Readable w.mem (base + srcOff) size.toNat)
    | _ => False
  | .sinf => fun bits w => match bits with
    | [_x] => True
    | _ => False
  | .cosf => fun bits w => match bits with
    | [_x] => True
    | _ => False
  | .powf => fun bits w => match bits with
    | [_b, _e] => True
    | _ => False
  | .htInit => fun bits w => match bits with
    | [slot] => (Slot8 w.mem slot)
    | _ => False
  | .htCleanup => fun bits w => match bits with
    | [slot] => (Slot8 w.mem slot)
    | _ => False
  | .htCreate => fun bits w => match bits with
    | [_ctx] => True
    | _ => False
  | .htCount => fun bits w => match bits with
    | [_ctx] => True
    | _ => False
  | .htLookup => fun bits w => match bits with
    | [_ctx, keyPtr, keyLen, resultPtr] => (Readable w.mem keyPtr keyLen.toNat) ∧ (∀ key val, readBytes w.mem keyPtr keyLen.toNat = some key → w.ht.get key = some val → Writable w.mem resultPtr val.size)
    | _ => False
  | .htInsert => fun bits w => match bits with
    | [_ctx, keyPtr, keyLen, valPtr, valLen] => (Readable w.mem keyPtr keyLen.toNat) ∧ (Readable w.mem valPtr valLen.toNat)
    | _ => False
  | .htIncrement => fun bits w => match bits with
    | [_ctx, keyPtr, keyLen, _addend] => (Readable w.mem keyPtr keyLen.toNat) ∧ (HtCounters w.ht)
    | _ => False
  | .htGetEntry => fun bits w => match bits with
    | [_ctx, index, keyOut, valOut] => (∀ k v, w.ht.entries[index.toNat]? = some (k, v) → ∃ m, copyIn w.mem keyOut k = some m ∧ (copyIn m valOut v).isSome = true)
    | _ => False
  | .cudaInit => fun bits w => match bits with
    | [slot] => ((w.mem.store slot 8 cudaCtx).isSome = true)
    | _ => False
  | .cudaCleanup => fun bits w => match bits with
    | [slot] => ((w.mem.store slot 8 0).isSome = true)
    | _ => False
  | .cudaCreateBuffer => fun bits w => match bits with
    | [ctx, _size] => (CtxOk w ctx)
    | _ => False
  | .cudaUpload => fun bits w => match bits with
    | [ctx, buf, src, size] => (CtxOk w ctx) ∧ (∀ b, w.dev.get? (asI32 buf) = some b → b.size = size.toNat → Readable w.mem src size.toNat ∧ Ready w.dev.race defaultParty (asI32 buf).toNat true)
    | _ => False
  | .cudaUploadOffset => fun bits w => match bits with
    | [ctx, buf, _off, src, size] => (CtxOk w ctx) ∧ (∀ b, w.dev.get? (asI32 buf) = some b → Readable w.mem src size.toNat ∧ Ready w.dev.race defaultParty (asI32 buf).toNat true)
    | _ => False
  | .cudaDownload => fun bits w => match bits with
    | [ctx, buf, dst, size] => (CtxOk w ctx) ∧ (∀ b, w.dev.get? (asI32 buf) = some b → b.size = size.toNat → Writable w.mem dst size.toNat ∧ Ready w.dev.race defaultParty (asI32 buf).toNat false)
    | _ => False
  | .cudaDownloadOffset => fun bits w => match bits with
    | [ctx, buf, _off, dst, size] => (CtxOk w ctx) ∧ (∀ b, w.dev.get? (asI32 buf) = some b → Writable w.mem dst size.toNat ∧ Ready w.dev.race defaultParty (asI32 buf).toNat false)
    | _ => False
  | .cudaFreeBuffer => fun bits w => match bits with
    | [ctx, buf] => (CtxOk w ctx) ∧ ((w.dev.get? (asI32 buf)).isSome = true → Ready w.dev.race defaultParty (asI32 buf).toNat true)
    | _ => False
  | .cudaSync => fun bits w => match bits with
    | [ctx] => (CtxOk w ctx)
    | _ => False
  | .cudaLaunch => fun bits w => match bits with
    | [ctx, kptr, nBufs, bindPtr, _gx, _gy, _gz, _bx, _by_, _bz] => (CtxOk w ctx) ∧ (KeepsSize w.kernel) ∧ (w.dev.capture = none) ∧ ((readCStrAt w.mem kptr).isSome = true) ∧ (BindsReady w defaultParty nBufs bindPtr)
    | _ => False
  | .cublasSgemv => fun bits w => match bits with
    | [ctx, trans, m, n, _alpha, a, x, _beta, y] => (CtxOk w ctx) ∧ (w.dev.capture = none) ∧ (VendorKeeps w.vendor) ∧ (SgemvOk w defaultParty trans m n a x y)
    | _ => False
  | .cublasSgemm => fun bits w => match bits with
    | [ctx, ta, tb, m, n, k, _alpha, a, sa, b, sb, _beta, c, sc, batch, oa, ob, oc, la, lb, lc] => (CtxOk w ctx) ∧ (w.dev.capture = none) ∧ (VendorKeeps w.vendor) ∧ (GemmOk (sgemmFits ta tb m n k sa sb sc batch oa ob oc la lb lc) w defaultParty a b c)
    | _ => False
  | .cublasGemmExBf16 => fun bits w => match bits with
    | [ctx, ta, tb, m, n, k, _alpha, a, b, _beta, c, oa, ob, oc, la, lb, lc] => (CtxOk w ctx) ∧ (w.dev.capture = none) ∧ (VendorKeeps w.vendor) ∧ (GemmOk (gemmBf16Fits ta tb m n k oa ob oc la lb lc) w defaultParty a b c)
    | _ => False
  | .cublasSgemvOnStream => fun bits w => match bits with
    | [ctx, trans, m, n, _alpha, a, x, _beta, y, sid] => (CtxOk w ctx) ∧ (w.dev.capture = none) ∧ (VendorKeeps w.vendor) ∧ (∀ p, w.dev.party? (asI32 sid) = some p → SgemvOk w p trans m n a x y)
    | _ => False
  | .cublasGemmStridedBatchedExBf16 => fun bits w => match bits with
    | [ctx, ta, tb, m, n, k, _alpha, a, sa, b, sb, _beta, c, sc, batch, oa, ob, oc, la, lb, lc] => (CtxOk w ctx) ∧ (w.dev.capture = none) ∧ (VendorKeeps w.vendor) ∧ (GemmOk (gemmStridedFits 2 2 4 ta tb m n k sa sb sc batch oa ob oc la lb lc) w defaultParty a b c)
    | _ => False
  | .cublasPtrArray => fun bits w => match bits with
    | [ctx, arr, _slot, _src, _off] => (CtxOk w ctx) ∧ (∀ A, w.dev.get? (asI32 arr) = some A → Ready w.dev.race defaultParty (asI32 arr).toNat true)
    | _ => False
  | .cublasSgemmBatchedOnStream => fun bits w => match bits with
    | [ctx, ta, tb, m, n, k, alpha, aArr, bArr, beta, cArr, batch, sid] => (CtxOk w ctx) ∧ (∀ p PA PB PC, w.dev.party? (asI32 sid) = some p → w.dev.get? (asI32 aArr) = some PA → w.dev.get? (asI32 bArr) = some PB → w.dev.get? (asI32 cArr) = some PC → (batchedRun w p ta tb m n k alpha aArr bArr cArr batch beta PA PB PC).isSome = true)
    | _ => False
  | .cudaLaunchNamedOnStream => fun bits w => match bits with
    | [ctx, kptr, namePtr, nBufs, bindPtr, _gx, _gy, _gz, _bx, _by_, _bz, sid] => (CtxOk w ctx) ∧ (KeepsSize w.kernel) ∧ (w.dev.capture = none) ∧ ((readCStrAt w.mem kptr).isSome = true) ∧ ((readCStrAt w.mem namePtr).isSome = true) ∧ (∀ p, w.dev.party? (asI32 sid) = some p → BindsReady w p nBufs bindPtr)
    | _ => False
  | .cudaEventElapsedMsBits => fun bits w => match bits with
    | [ctx, s, e] => (CtxOk w ctx) ∧ (∀ x y, w.dev.event? (asI32 s) = some x → w.dev.event? (asI32 e) = some y → ElapsedOk w x y)
    | _ => False
  | .cudaUploadAsync => fun bits w => match bits with
    | [ctx, buf, src, size, sid] => (CtxOk w ctx) ∧ (w.dev.capture = none) ∧ (∀ p b, w.dev.party? (asI32 sid) = some p → w.dev.get? (asI32 buf) = some b → size.toNat ≤ b.size → Readable (w.mem.forParty p) src size.toNat ∧ Ready w.dev.race p (asI32 buf).toNat true)
    | _ => False
  | .cudaUploadOffsetAsync => fun bits w => match bits with
    | [ctx, buf, off, src, size, sid] => (CtxOk w ctx) ∧ (w.dev.capture = none) ∧ (∀ p b, w.dev.party? (asI32 sid) = some p → w.dev.get? (asI32 buf) = some b → off.toNat + size.toNat ≤ b.size → Readable (w.mem.forParty p) src size.toNat ∧ Ready w.dev.race p (asI32 buf).toNat true)
    | _ => False
  | .cudaDownloadAsync => fun bits w => match bits with
    | [ctx, buf, dst, size, sid] => (CtxOk w ctx) ∧ (w.dev.capture = none) ∧ (∀ p b, w.dev.party? (asI32 sid) = some p → w.dev.get? (asI32 buf) = some b → size.toNat ≤ b.size → Writable (w.mem.forParty p) dst size.toNat ∧ Ready w.dev.race p (asI32 buf).toNat false)
    | _ => False
  | .cublasSgemmOnStream => fun bits w => match bits with
    | [ctx, ta, tb, m, n, k, _alpha, a, sa, b, sb, _beta, c, sc, batch, sid, oa, ob, oc, la, lb, lc] => (CtxOk w ctx) ∧ (w.dev.capture = none) ∧ (VendorKeeps w.vendor) ∧ (∀ p, w.dev.party? (asI32 sid) = some p → GemmOk (sgemmFits ta tb m n k sa sb sc batch oa ob oc la lb lc) w p a b c)
    | _ => False
  | .cudaLaunchOnStream => fun bits w => match bits with
    | [ctx, kptr, nBufs, bindPtr, _gx, _gy, _gz, _bx, _by_, _bz, sid] => (CtxOk w ctx) ∧ (KeepsSize w.kernel) ∧ (w.dev.capture = none) ∧ ((readCStrAt w.mem kptr).isSome = true) ∧ (∀ p, w.dev.party? (asI32 sid) = some p → BindsReady w p nBufs bindPtr)
    | _ => False
  | .cudaStreamCreate => fun bits w => match bits with
    | [ctx] => (CtxOk w ctx)
    | _ => False
  | .cudaStreamSync => fun bits w => match bits with
    | [ctx, _sid] => (CtxOk w ctx) ∧ (w.dev.capture = none)
    | _ => False
  | .cudaStreamDestroy => fun bits w => match bits with
    | [ctx, _sid] => (CtxOk w ctx) ∧ (w.dev.capture = none)
    | _ => False
  | .cudaEventCreate => fun bits w => match bits with
    | [ctx] => (CtxOk w ctx)
    | _ => False
  | .cudaEventRecord => fun bits w => match bits with
    | [ctx, _eid, _sid] => (CtxOk w ctx) ∧ (w.dev.capture = none)
    | _ => False
  | .cudaStreamWaitEvent => fun bits w => match bits with
    | [ctx, _sid, eid] => (CtxOk w ctx) ∧ (w.dev.capture = none) ∧ (∀ c, w.dev.event? (asI32 eid) ≠ some (some (c, true)))
    | _ => False
  | .cudaEventDestroy => fun bits w => match bits with
    | [ctx, _eid] => (CtxOk w ctx)
    | _ => False
  | .cudaGraphBeginCapture => fun bits w => match bits with
    | [ctx, _sid] => (CtxOk w ctx) ∧ (w.dev.capture = none)
    | _ => False
  | .cudaGraphEndCapture => fun bits w => match bits with
    | [ctx, sid] => (CtxOk w ctx) ∧ (∀ p c, w.dev.party? (asI32 sid) = some p → w.dev.capture = some c → c.origin = p)
    | _ => False
  | .cudaGraphUpload => fun bits w => match bits with
    | [ctx, _gid, _sid] => (CtxOk w ctx)
    | _ => False
  | .cudaGraphLaunch => fun bits w => match bits with
    | [ctx, gid, sid] => (CtxOk w ctx) ∧ (w.dev.capture = none) ∧ (∀ ops p, w.dev.graph? (asI32 gid) = some ops → w.dev.party? (asI32 sid) = some p → GraphRuns w ops p)
    | _ => False
  | .cudaGraphDestroy => fun bits w => match bits with
    | [ctx, _gid] => (CtxOk w ctx)
    | _ => False
  | .cudaLaunchNamed => fun bits w => match bits with
    | [ctx, kptr, namePtr, nBufs, bindPtr, _gx, _gy, _gz, _bx, _by_, _bz] => (CtxOk w ctx) ∧ (KeepsSize w.kernel) ∧ (w.dev.capture = none) ∧ ((readCStrAt w.mem kptr).isSome = true) ∧ ((readCStrAt w.mem namePtr).isSome = true) ∧ (BindsReady w defaultParty nBufs bindPtr)
    | _ => False
  | .cudaPinnedAlloc => fun bits w => match bits with
    | [ctx, size] => (CtxOk w ctx) ∧ (PinnedRoom w size)
    | _ => False
  | .cudaPinnedPtr => fun bits w => match bits with
    | [ctx, _id] => (CtxOk w ctx)
    | _ => False
  | .cudaPinnedPtrAt => fun bits w => match bits with
    | [ctx, _id, _off, _len] => (CtxOk w ctx)
    | _ => False
  | .cudaPinnedFree => fun bits w => match bits with
    | [ctx, _id] => (CtxOk w ctx)
    | _ => False
  | .cudaMemInfoFree => fun bits w => match bits with
    | [ctx] => (CtxOk w ctx)
    | _ => False
  | .cudaMemInfoTotal => fun bits w => match bits with
    | [ctx] => (CtxOk w ctx)
    | _ => False
  | .gpuInit => fun bits w => match bits with
    | [slot] => (Slot8 w.mem slot)
    | _ => False
  | .gpuCleanup => fun bits w => match bits with
    | [slot] => (Slot8 w.mem slot)
    | _ => False
  | .gpuCreateBuffer => fun bits w => match bits with
    | [ctx, _size] => (GpuOk w ctx)
    | _ => False
  | .gpuCreatePipeline => fun bits w => match bits with
    | [ctx, shaderPtr, bindPtr, n] => (GpuOk w ctx) ∧ (ctx ≠ 0 → ¬ asI32 n < 0 → CStr w.mem shaderPtr ∧ (readBinds w.mem bindPtr (asI32 n).toNat).isSome = true)
    | _ => False
  | .gpuUpload => fun bits w => match bits with
    | [ctx, buf, src, size] => (GpuOk w ctx) ∧ (∀ b, ctx ≠ 0 → w.gpu.bufs[(asI32 buf).toNat]? = some b → size.toNat ≤ b.size → size.toNat % 4 = 0 → Readable w.mem src size.toNat)
    | _ => False
  | .gpuUploadPtr => fun bits w => match bits with
    | [ctx, buf, src, size] => (GpuOk w ctx) ∧ (∀ b, ctx ≠ 0 → w.gpu.bufs[(asI32 buf).toNat]? = some b → size.toNat ≤ b.size → size.toNat % 4 = 0 → Readable w.mem src size.toNat)
    | _ => False
  | .gpuDispatch => fun bits w => match bits with
    | [ctx, _pipe, _x, _y, _z] => (GpuOk w ctx)
    | _ => False
  | .gpuDownload => fun bits w => match bits with
    | [ctx, buf, dst, size] => (GpuOk w ctx) ∧ (∀ g b, ctx ≠ 0 → w.gpu.flush w.shader = some g → g.bufs[(asI32 buf).toNat]? = some b → size.toNat = b.size → size.toNat % 4 = 0 → Writable w.mem dst size.toNat)
    | _ => False
  | .gpuDownloadPtr => fun bits w => match bits with
    | [ctx, buf, off, dst, size] => (GpuOk w ctx) ∧ (∀ g b, ctx ≠ 0 → w.gpu.flush w.shader = some g → g.bufs[(asI32 buf).toNat]? = some b → off.toNat + size.toNat ≤ b.size → off.toNat % 4 = 0 → size.toNat % 4 = 0 → Writable w.mem dst size.toNat)
    | _ => False
  | .lmdbInit => fun bits w => match bits with
    | [slot] => (w.lmdb.live = false) ∧ (Slot8 w.mem slot)
    | _ => False
  | .lmdbCleanup => fun bits w => match bits with
    | [slot] => (Load8 w.mem slot) ∧ (LmdbSlotOk w slot) ∧ (Slot8 w.mem slot)
    | _ => False
  | .lmdbOpen => fun bits w => match bits with
    | [ctx, pathPtr, _x] => (LmdbOk w ctx) ∧ (ctx ≠ 0 → CStr w.mem pathPtr) ∧ (∀ p, readCStr w.mem pathPtr = some p → w.lmdb.envs.any (·.path == p) = false)
    | _ => False
  | .lmdbBeginWriteTxn => fun bits w => match bits with
    | [ctx, _handle] => (LmdbOk w ctx)
    | _ => False
  | .lmdbPut => fun bits w => match bits with
    | [ctx, _handle, keyPtr, keyLen, valPtr, valLen] => (LmdbOk w ctx) ∧ (ctx ≠ 0 → Readable w.mem keyPtr (asI32 keyLen).toNat ∧ Readable w.mem valPtr (asI32 valLen).toNat)
    | _ => False
  | .lmdbCommitWriteTxn => fun bits w => match bits with
    | [ctx, _handle] => (LmdbOk w ctx)
    | _ => False
  | .lmdbCursorScan => fun bits w => match bits with
    | [ctx, _handle, keyPtr, keyLen, _maxEntries, resultPtr, cap] => (LmdbOk w ctx) ∧ (ctx ≠ 0 → ∀ n, n ≤ cap.toNat → Writable w.mem resultPtr n) ∧ (ctx ≠ 0 → 0 < asI32 keyLen → ∃ k, readBytes w.mem keyPtr (asI32 keyLen).toNat = some k ∧ lmdbKeyOk k = true)
    | _ => False
  | .windowInit => fun bits w => match bits with
    | [slot] => (w.win.live = false) ∧ (Slot8 w.mem slot)
    | _ => False
  | .windowCleanup => fun bits w => match bits with
    | [slot] => (Load8 w.mem slot) ∧ (WinSlotOk w slot) ∧ (Slot8 w.mem slot)
    | _ => False
  | .windowOpen => fun bits w => match bits with
    | [ctx, _width, _height, titlePtr, titleLen, blitPtr, blitLen] => (WinOk w ctx) ∧ (ctx ≠ 0 → Readable w.mem titlePtr (asI64 titleLen).toNat ∧ Readable w.mem blitPtr (asI64 blitLen).toNat)
    | _ => False
  | .windowPoll => fun bits w => match bits with
    | [ctx, eventsPtr, maxEvents] => (WinOk w ctx) ∧ (ctx ≠ 0 → Writable w.pump.mem eventsPtr (winEventBytes (w.pump.win.pending.take (min w.pump.win.pending.length (asI32 maxEvents).toNat))).size)
    | _ => False
  | .windowPresentGpuBuffer => fun bits w => match bits with
    | [ctx, gctx, _buf] => (WinOk w ctx) ∧ (ctx ≠ 0 → GpuOk w.pump gctx)
    | _ => False
  | .nativeLoad => fun bits w => match bits with
    | [src, len] => (¬ (src = 0 ∨ asI64 len ≤ 0) → Readable w.mem src (asI64 len).toNat)
    | _ => False
  | .threadInit => fun bits w => match bits with
    | [slot] => (w.thread.live = false) ∧ (Slot8 w.mem slot)
    | _ => False
  | .threadCleanup => fun bits w => match bits with
    | [slot] => (ThreadSlotOk w slot)
    | _ => False
  | .threadJoin => fun bits w => match bits with
    | [ctx, _hd] => (ThreadOk w ctx)
    | _ => False
  | .nativeFree => fun bits w => match bits with
    | [_addr] => True
    | _ => False
  | .nativeArch => fun bits w => match bits with
    | [] => True
    | _ => False
  | .cpuHas => fun bits w => match bits with
    | [name] => (name ≠ 0 → CStr w.mem name)
    | _ => False
  | .threadSpawn => fun _ _ => False
  | .fileCreateDirAll => fun bits w => match bits with
    | [path] => PathOk w.mem path
    | _ => False
  -- The thread shims are stated where programs meet them, as `threadSpawn`
  -- and `threadJoin`, whose functions call them.
  | .threadStart | .threadFinish => fun _ _ => False
  | .memLock => fun bits _ => bits.length = 2
  | .memUnlock => fun bits _ => bits.length = 2
  | .memAdviseHuge => fun bits _ => bits.length = 2
  | .threadPriority => fun bits _ => bits.length = 1

/-- **Every entry point answers under its precondition**: whatever the call,
    `Pre` is enough for the model not to refuse it. -/
theorem pre_safe (f : IR.Ffi) (bits : List UInt64) (w : World) (h : Pre f bits w) :
    (callBits f bits w).isSome = true := by
  cases f
  case fileRead =>
    simp only [Pre] at h; split at h
    · apply fileRead_safe h.1 h.2
    · exact h.elim
  case fileWrite =>
    simp only [Pre] at h; split at h
    · apply fileWrite_safe h.1 h.2
    · exact h.elim
  case fileReadToPtr =>
    simp only [Pre] at h; split at h
    · apply fileReadToPtr_safe h.1 h.2
    · exact h.elim
  case fileWriteFromPtr =>
    simp only [Pre] at h; split at h
    · apply fileWriteFromPtr_safe h
    · exact h.elim
  case stdinReadline =>
    simp only [Pre] at h; split at h
    · apply stdinReadline_safe h
    · exact h.elim
  case stdoutWrite =>
    simp only [Pre] at h; split at h
    · apply stdoutWrite_safe h
    · exact h.elim
  case sinf =>
    simp only [Pre] at h; split at h
    · apply sinf_safe _ _
    · exact h.elim
  case cosf =>
    simp only [Pre] at h; split at h
    · apply cosf_safe _ _
    · exact h.elim
  case powf =>
    simp only [Pre] at h; split at h
    · apply powf_safe _ _ _
    · exact h.elim
  case htInit =>
    simp only [Pre] at h; split at h
    · apply htInit_safe h
    · exact h.elim
  case htCleanup =>
    simp only [Pre] at h; split at h
    · apply htCleanup_safe h
    · exact h.elim
  case htCreate =>
    simp only [Pre] at h; split at h
    · apply htCreate_safe _ _
    · exact h.elim
  case htCount =>
    simp only [Pre] at h; split at h
    · apply htCount_safe _ _
    · exact h.elim
  case htLookup =>
    simp only [Pre] at h; split at h
    · apply htLookup_safe h.1 h.2
    · exact h.elim
  case htInsert =>
    simp only [Pre] at h; split at h
    · apply htInsert_safe h.1 h.2
    · exact h.elim
  case htIncrement =>
    simp only [Pre] at h; split at h
    · apply htIncrement_safe h.1 h.2
    · exact h.elim
  case htGetEntry =>
    simp only [Pre] at h; split at h
    · apply htGetEntry_safe h
    · exact h.elim
  case cudaInit =>
    simp only [Pre] at h; split at h
    · apply init_safe h
    · exact h.elim
  case cudaCleanup =>
    simp only [Pre] at h; split at h
    · apply cleanup_safe h
    · exact h.elim
  case cudaCreateBuffer =>
    simp only [Pre] at h; split at h
    · apply createBuffer_safe h
    · exact h.elim
  case cudaUpload =>
    simp only [Pre] at h; split at h
    · apply upload_safe h.1 h.2
    · exact h.elim
  case cudaUploadOffset =>
    simp only [Pre] at h; split at h
    · apply uploadOffset_safe h.1 h.2
    · exact h.elim
  case cudaDownload =>
    simp only [Pre] at h; split at h
    · apply download_safe h.1 h.2
    · exact h.elim
  case cudaDownloadOffset =>
    simp only [Pre] at h; split at h
    · apply downloadOffset_safe h.1 h.2
    · exact h.elim
  case cudaFreeBuffer =>
    simp only [Pre] at h; split at h
    · apply freeBuffer_safe h.1 h.2
    · exact h.elim
  case cudaSync =>
    simp only [Pre] at h; split at h
    · apply sync_safe h
    · exact h.elim
  case cudaLaunch =>
    simp only [Pre] at h; split at h
    · apply launch_safe h.1 h.2.1 h.2.2.1 h.2.2.2.1 h.2.2.2.2
    · exact h.elim
  case cublasSgemv =>
    simp only [Pre] at h; split at h
    · apply cublasSgemv_safe h.1 h.2.1 h.2.2.1 h.2.2.2
    · exact h.elim
  case cublasSgemm =>
    simp only [Pre] at h; split at h
    · apply cublasSgemm_safe h.1 h.2.1 h.2.2.1 h.2.2.2
    · exact h.elim
  case cublasGemmExBf16 =>
    simp only [Pre] at h; split at h
    · apply cublasGemmExBf16_safe h.1 h.2.1 h.2.2.1 h.2.2.2
    · exact h.elim
  case cublasSgemvOnStream =>
    simp only [Pre] at h; split at h
    · apply cublasSgemvOnStream_safe h.1 h.2.1 h.2.2.1 h.2.2.2
    · exact h.elim
  case cublasGemmStridedBatchedExBf16 =>
    simp only [Pre] at h; split at h
    · apply cublasGemmStridedBatchedExBf16_safe h.1 h.2.1 h.2.2.1 h.2.2.2
    · exact h.elim
  case cublasPtrArray =>
    simp only [Pre] at h; split at h
    · apply cublasPtrArray_safe h.1 h.2
    · exact h.elim
  case cublasSgemmBatchedOnStream =>
    simp only [Pre] at h; split at h
    · apply cublasSgemmBatchedOnStream_safe h.1 h.2
    · exact h.elim
  case cudaLaunchNamedOnStream =>
    simp only [Pre] at h; split at h
    · apply launchNamedOnStream_safe h.1 h.2.1 h.2.2.1 h.2.2.2.1 h.2.2.2.2.1 h.2.2.2.2.2
    · exact h.elim
  case cudaEventElapsedMsBits =>
    simp only [Pre] at h; split at h
    · apply eventElapsedMsBits_safe h.1 h.2
    · exact h.elim
  case cudaUploadAsync =>
    simp only [Pre] at h; split at h
    · apply uploadAsync_safe h.1 h.2.1 h.2.2
    · exact h.elim
  case cudaUploadOffsetAsync =>
    simp only [Pre] at h; split at h
    · apply uploadOffsetAsync_safe h.1 h.2.1 h.2.2
    · exact h.elim
  case cudaDownloadAsync =>
    simp only [Pre] at h; split at h
    · apply downloadAsync_safe h.1 h.2.1 h.2.2
    · exact h.elim
  case cublasSgemmOnStream =>
    simp only [Pre] at h; split at h
    · apply cublasSgemmOnStream_safe h.1 h.2.1 h.2.2.1 h.2.2.2
    · exact h.elim
  case cudaLaunchOnStream =>
    simp only [Pre] at h; split at h
    · apply launchOnStream_safe h.1 h.2.1 h.2.2.1 h.2.2.2.1 h.2.2.2.2
    · exact h.elim
  case cudaStreamCreate =>
    simp only [Pre] at h; split at h
    · apply streamCreate_safe h
    · exact h.elim
  case cudaStreamSync =>
    simp only [Pre] at h; split at h
    · apply streamSync_safe h.1 h.2
    · exact h.elim
  case cudaStreamDestroy =>
    simp only [Pre] at h; split at h
    · apply streamDestroy_safe h.1 h.2
    · exact h.elim
  case cudaEventCreate =>
    simp only [Pre] at h; split at h
    · apply eventCreate_safe h
    · exact h.elim
  case cudaEventRecord =>
    simp only [Pre] at h; split at h
    · apply eventRecord_safe h.1 h.2
    · exact h.elim
  case cudaStreamWaitEvent =>
    simp only [Pre] at h; split at h
    · apply streamWaitEvent_safe h.1 h.2.1 h.2.2
    · exact h.elim
  case cudaEventDestroy =>
    simp only [Pre] at h; split at h
    · apply eventDestroy_safe h
    · exact h.elim
  case cudaGraphBeginCapture =>
    simp only [Pre] at h; split at h
    · apply graphBeginCapture_safe h.1 h.2
    · exact h.elim
  case cudaGraphEndCapture =>
    simp only [Pre] at h; split at h
    · apply graphEndCapture_safe h.1 h.2
    · exact h.elim
  case cudaGraphUpload =>
    simp only [Pre] at h; split at h
    · apply graphUpload_safe h
    · exact h.elim
  case cudaGraphLaunch =>
    simp only [Pre] at h; split at h
    · apply graphLaunch_safe h.1 h.2.1 h.2.2
    · exact h.elim
  case cudaGraphDestroy =>
    simp only [Pre] at h; split at h
    · apply graphDestroy_safe h
    · exact h.elim
  case cudaLaunchNamed =>
    simp only [Pre] at h; split at h
    · apply launchNamed_safe h.1 h.2.1 h.2.2.1 h.2.2.2.1 h.2.2.2.2.1 h.2.2.2.2.2
    · exact h.elim
  case cudaPinnedAlloc =>
    simp only [Pre] at h; split at h
    · apply pinnedAlloc_safe h.1 h.2
    · exact h.elim
  case cudaPinnedPtr =>
    simp only [Pre] at h; split at h
    · apply pinnedPtr_safe h
    · exact h.elim
  case cudaPinnedPtrAt =>
    simp only [Pre] at h; split at h
    · apply pinnedPtrAt_safe h
    · exact h.elim
  case cudaPinnedFree =>
    simp only [Pre] at h; split at h
    · apply pinnedFree_safe h
    · exact h.elim
  case cudaMemInfoFree =>
    simp only [Pre] at h; split at h
    · apply memInfoFree_safe h
    · exact h.elim
  case cudaMemInfoTotal =>
    simp only [Pre] at h; split at h
    · apply memInfoTotal_safe h
    · exact h.elim
  case gpuInit =>
    simp only [Pre] at h; split at h
    · apply gpuInit_safe h
    · exact h.elim
  case gpuCleanup =>
    simp only [Pre] at h; split at h
    · apply gpuCleanup_safe h
    · exact h.elim
  case gpuCreateBuffer =>
    simp only [Pre] at h; split at h
    · apply gpuCreateBuffer_safe h
    · exact h.elim
  case gpuCreatePipeline =>
    simp only [Pre] at h; split at h
    · apply gpuCreatePipeline_safe h.1 h.2
    · exact h.elim
  case gpuUpload =>
    simp only [Pre] at h; split at h
    · apply gpuUpload_safe h.1 h.2
    · exact h.elim
  case gpuUploadPtr =>
    simp only [Pre] at h; split at h
    · apply gpuUploadPtr_safe h.1 h.2
    · exact h.elim
  case gpuDispatch =>
    simp only [Pre] at h; split at h
    · apply gpuDispatch_safe h
    · exact h.elim
  case gpuDownload =>
    simp only [Pre] at h; split at h
    · apply gpuDownload_safe h.1 h.2
    · exact h.elim
  case gpuDownloadPtr =>
    simp only [Pre] at h; split at h
    · apply gpuDownloadPtr_safe h.1 h.2
    · exact h.elim
  case lmdbInit =>
    simp only [Pre] at h; split at h
    · apply lmdbInit_safe h.1 h.2
    · exact h.elim
  case lmdbCleanup =>
    simp only [Pre] at h; split at h
    · apply lmdbCleanup_safe h.1 h.2.1 h.2.2
    · exact h.elim
  case lmdbOpen =>
    simp only [Pre] at h; split at h
    · apply lmdbOpen_safe (x := 0) h.1 h.2.1 h.2.2
    · exact h.elim
  case lmdbBeginWriteTxn =>
    simp only [Pre] at h; split at h
    · apply lmdbBeginWriteTxn_safe h
    · exact h.elim
  case lmdbPut =>
    simp only [Pre] at h; split at h
    · apply lmdbPut_safe h.1 h.2
    · exact h.elim
  case lmdbCommitWriteTxn =>
    simp only [Pre] at h; split at h
    · apply lmdbCommitWriteTxn_safe h
    · exact h.elim
  case lmdbCursorScan =>
    simp only [Pre] at h; split at h
    · apply lmdbCursorScan_safe h.1 h.2.1 h.2.2
    · exact h.elim
  case windowInit =>
    simp only [Pre] at h; split at h
    · apply windowInit_safe h.1 h.2
    · exact h.elim
  case windowCleanup =>
    simp only [Pre] at h; split at h
    · apply windowCleanup_safe h.1 h.2.1 h.2.2
    · exact h.elim
  case windowOpen =>
    simp only [Pre] at h; split at h
    · apply windowOpen_safe h.1 h.2
    · exact h.elim
  case windowPoll =>
    simp only [Pre] at h; split at h
    · apply windowPoll_safe h.1 h.2
    · exact h.elim
  case windowPresentGpuBuffer =>
    simp only [Pre] at h; split at h
    · apply windowPresentGpuBuffer_safe h.1 h.2
    · exact h.elim
  case nativeLoad =>
    simp only [Pre] at h; split at h
    · apply nativeLoad_safe h
    · exact h.elim
  case threadInit =>
    simp only [Pre] at h; split at h
    · apply threadInit_safe h.1 h.2
    · exact h.elim
  case threadCleanup =>
    simp only [Pre] at h; split at h
    · apply threadCleanup_safe h
    · exact h.elim
  case threadJoin =>
    simp only [Pre] at h; split at h
    · apply threadJoin_safe h
    · exact h.elim
  case nativeFree =>
    simp only [Pre] at h; split at h
    · apply nativeFree_safe _ _
    · exact h.elim
  case nativeArch =>
    simp only [Pre] at h; split at h
    · apply nativeArch_safe _
    · exact h.elim
  case cpuHas =>
    simp only [Pre] at h; split at h
    · apply cpuHas_safe h
    · exact h.elim
  case threadSpawn => exact h.elim
  case fileCreateDirAll =>
    simp only [Pre] at h; split at h
    · apply fileCreateDirAll_safe h
    · exact h.elim
  case threadStart => exact h.elim
  case threadFinish => exact h.elim
  case memLock =>
    simp only [Pre] at h
    show (ffiGrant "lock" 2 bits w).isSome = true
    unfold ffiGrant; simp [h]
  case memUnlock =>
    simp only [Pre] at h
    show (ffiGrant "unlock" 2 bits w).isSome = true
    unfold ffiGrant; simp [h]
  case memAdviseHuge =>
    simp only [Pre] at h
    show (ffiGrant "hugepages" 2 bits w).isSome = true
    unfold ffiGrant; simp [h]
  case threadPriority =>
    simp only [Pre] at h
    show (ffiGrant "priority" 1 bits w).isSome = true
    unfold ffiGrant; simp [h]

end AlgorithmLib.HProg.Contracts
