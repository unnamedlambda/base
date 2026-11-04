module
public import AlgorithmLib.Host.DevSpec
meta import AlgorithmLib.Host.DevSpec
public import AlgorithmLib.Host.Spec
meta import AlgorithmLib.Host.Spec
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Host.Contracts` — when the engine's base functions and the device answer

The facts the base functions' and the device driver's preconditions are stated
in, and the proofs that each model answers under them (`ExtContracts` reads
them into its rows):

* memory: `CStr` (a readable C string), `Slot8` (an eight-byte slot the
  program may write), `Load8`, `Readable`, `Writable`, `Field`;
* the device: `Ready` from `DevSpec` (the race vocabulary), `VendorKeeps`,
  `SgemvOk`, `GemmOk`, `PinnedRoom`;
* the counts a call copies: `readCountPtr`.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.DevSpec

namespace AlgorithmLib.HProg.Contracts

/-- The bytes at `a` are a C string: a zero ends them, and they are UTF-8. -/
def CStr (m : Mem) (a : UInt64) : Prop := (readCStr m a).isSome = true

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

/-- How many bytes a read through raw pointers copies out of `c`. -/
def readCountPtr (c : ByteArray) (fileOff size : UInt64) : Nat :=
  min size.toNat (c.size - min fileOff.toNat c.size)

/-- The `cap`-byte field at `a` is there to read: all a path or a name that
    base's own calls take needs ([`readField`]). -/
def Field (m : Mem) (a cap : UInt64) : Prop := Readable m a cap.toNat

theorem readField_isSome {m : Mem} {a cap : UInt64} (h : Field m a cap) :
    (readField m a cap).isSome = true := by
  unfold readField; exact bind_some h fun _ _ => rfl

theorem fileReadToPtr_safe {w : World} {pathPtr pathCap dstPtr fileOff size : UInt64}
    (hp : ¬ asI64 size ≤ 0 → Field w.mem pathPtr pathCap)
    (hw : ∀ path c, readField w.mem pathPtr pathCap = some (some path) → w.fs.get path = some c →
      Writable w.mem dstPtr (readCountPtr c fileOff size)) :
    (ffiFileReadToPtr [pathPtr, pathCap, dstPtr, fileOff, size] w).isSome = true := by
  unfold ffiFileReadToPtr; dsimp only
  split
  · rfl
  · next hsz =>
    refine bind_some (readField_isSome (hp hsz)) fun path e => ?_
    dsimp only
    split
    · rfl
    · next c ec =>
        obtain ⟨p, rfl, ec'⟩ : ∃ p, path = some p ∧ w.fs.get p = some c := by
          rcases path with _ | p
          · simp at ec
          · exact ⟨p, rfl, by split at ec <;> simp_all⟩
        exact bind_some (hw p c e ec' _ (by simp [readCountPtr])) fun _ _ => rfl

theorem fileWriteFromPtr_safe {w : World} {pathPtr pathCap srcPtr fileOff size : UInt64}
    (hp : ¬ (asI64 size ≤ 0 ∨ asI64 fileOff < 0) → Field w.mem pathPtr pathCap ∧ Readable w.mem srcPtr size.toNat) :
    (ffiFileWriteFromPtr [pathPtr, pathCap, srcPtr, fileOff, size] w).isSome = true := by
  unfold ffiFileWriteFromPtr; dsimp only
  split
  · rfl
  · next hsz =>
    have h := hp (by simpa using hsz)
    exact bind_some (readField_isSome h.1) fun _ _ => by
      try dsimp only
      split
      · split
        · rfl
        · exact bind_some h.2 fun _ _ => rfl
      · rfl

theorem stdinReadline_safe {w : World} {dst cap : UInt64}
    (hw : ∀ line rest, w.stdin = line :: rest → ¬ asI64 cap ≤ 0 →
      Writable w.mem dst (min line.size cap.toNat)) :
    (ffiStdinReadline [dst, cap] w).isSome = true := by
  unfold ffiStdinReadline; dsimp only
  split
  · rfl
  · next hc =>
    split
    · rfl
    · next line rest hs =>
      exact bind_some (hw line rest hs hc _ (size_toByteArray_map_range _ _)) fun _ _ => rfl

theorem stdoutWrite_safe {w : World} {src size : UInt64}
    (hr : ¬ asI64 size < 0 → Readable w.mem src size.toNat) :
    (ffiStdoutWrite [src, size] w).isSome = true := by
  unfold ffiStdoutWrite; dsimp only
  split
  · rfl
  · next h => exact bind_some (hr h) fun _ _ => rfl

-- CUDA: pinned memory, memory info, events, graphs

/-- Pinned memory has room for `size` more bytes after aligning to 64. -/
def PinnedRoom (w : World) (size : UInt64) : Prop :=
  w.mem.pinned.size + (64 - w.mem.pinned.size % 64) % 64 + size.toNat ≤ regionSpan.toNat

theorem memInfoOf_isSome {w : World} {ctx : UInt64} {pick : UInt64 × UInt64 → UInt64} (hc : CtxOk w ctx) :
    (memInfoOf w ctx pick).isSome = true := by
  unfold memInfoOf; simp only [cudaCtxOk_of hc, bind, Option.bind]; split <;> rfl

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

-- native code and the CPU

theorem nativeLoad_safe {w : World} {src len : UInt64}
    (hr : ¬ (src = 0 ∨ asI64 len ≤ 0) → Readable w.mem src (asI64 len).toNat) :
    (ffiNativeLoad [src, len] w).isSome = true := by
  unfold ffiNativeLoad; dsimp only
  split
  · rfl
  · next h => exact bind_some (hr (by simpa using h)) fun _ _ => rfl

theorem nativeFree_safe {addr : UInt64} {w : World} (h : (w.nativeAt? addr).isSome) :
    (ffiNativeFree [addr] w).isSome = true := by
  unfold ffiNativeFree; dsimp only
  split
  · next e => rw [e] at h; cases h
  · rfl

theorem nativeArch_safe (w : World) : (ffiNativeArch [] w).isSome = true := rfl

theorem cpuHas_safe {w : World} {name cap : UInt64} (hn : Field w.mem name cap) :
    (ffiCpuHas [name, cap] w).isSome = true := by
  unfold ffiCpuHas; dsimp only
  exact bind_some (readField_isSome hn) fun _ _ => by split <;> rfl

theorem fileCreateDirAll_safe {w : World} {path cap : UInt64} (hp : Field w.mem path cap) :
    (ffiFileCreateDirAll [path, cap] w).isSome = true := by
  unfold ffiFileCreateDirAll; dsimp only
  exact bind_some (readField_isSome hp) fun _ _ => by
    split <;> (try split) <;> (try split) <;> rfl

theorem fileSize_safe {w : World} {path cap : UInt64} (hp : Field w.mem path cap) :
    (ffiFileSize [path, cap] w).isSome = true := by
  unfold ffiFileSize; dsimp only
  exact bind_some (readField_isSome hp) fun _ _ => by split <;> rfl

theorem fileSetLen_safe {w : World} {path cap len : UInt64} (hp : Field w.mem path cap) :
    (ffiFileSetLen [path, cap, len] w).isSome = true := by
  unfold ffiFileSetLen; dsimp only
  exact bind_some (readField_isSome hp) fun _ _ => by split <;> (try split) <;> rfl

-- threads

end AlgorithmLib.HProg.Contracts
