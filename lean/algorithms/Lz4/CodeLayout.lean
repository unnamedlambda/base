module
public import Lz4.CodeBody
public import AlgorithmLib.LZ4.U32Field
@[expose] public section

/-!
# `Lz4.CodeLayout` — where a launch's blocks read and write

The launch's inputs and outputs live in device buffer 0, at `base0`: block
`c`'s input at offset `c * inStride`, its output past every input and the copy
slack, at `numBlocks * inStride + copySlack + c * outStride`. The warp's model
sees device memory as buffer 0's image (`image`): byte `base0 + o` is byte `o`
of the buffer. So each block's input and output are in buffer 0, outside
shared memory, writable, and the image there is what the device holds
(`layout_obligations`); the outputs lie past every input and apart.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Prog
  AlgorithmLib.LZ4WarpDSL AlgorithmLib.LibSpec AlgorithmLib.Kernel LZ4WarpEvalBytes
  AlgorithmLib.LZ4Plan

namespace Lz4Code

/-- Device buffer 0's address. -/
abbrev base0 : UInt64 := addrOf (.buf 0) 0

theorem base0_toNat : base0.toNat = 6 * 2 ^ 36 := by decide

/-- Device buffer 0 as the warp's model sees memory: byte `base0 + o` is byte
    `o` of the buffer, every other byte 0. -/
def image (b : ByteArray) : Array UInt8 :=
  Array.ofFn (n := base0.toNat + 2 ^ 37) fun i => if base0.toNat ≤ i.val then b.get! (i.val - base0.toNat) else 0

theorem image_size (b : ByteArray) : (image b).size = base0.toNat + 2 ^ 37 := by simp [image]

theorem image_getD (b : ByteArray) {o : Nat} (ho : o < 2 ^ 37) :
    (image b).getD (base0.toNat + o) 0 = b.get! o := by
  simp only [image, Array.getD_eq_getD_getElem?, Array.getElem?_ofFn]
  rw [dif_pos (by omega)]
  simp

/-- Byte `o` of buffer 0, as an address. -/
theorem buf0_addr {o : Nat} (ho : o < 2 ^ 36) : (base0 + UInt64.ofNat o).toNat = base0.toNat + o := by
  rw [UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show o < 2 ^ 64 by omega), base0_toNat]
  exact Nat.mod_eq_of_lt (by omega)

theorem buf0_decode {o : Nat} (ho : o < 2 ^ 36) : decodeAddr (base0 + UInt64.ofNat o) = some (.buf 0, o) :=
  decode_buf 0 o ho

theorem buf0_notArena {o : Nat} (ho : o < 2 ^ 36) : NotArena (base0 + UInt64.ofNat o) := by
  intro off h; rw [buf0_decode ho] at h; cases h

/-- A byte of buffer 0 loads as the buffer holds it. -/
theorem buf0_load {bufs : Nat → ByteArray} {nan : ClifTy → UInt64 → UInt64} {o : Nat} (ho : o < 2 ^ 36)
    (hs : o < (bufs 0).size) :
    (launchWorld bufs nan).mem.load (base0 + UInt64.ofNat o) 1 = some ((bufs 0).get! o).toUInt64 := by
  simp only [Mem.load, buf0_decode ho, Option.bind_eq_bind, Option.bind_some, Mem.region, Mem.readable]
  rw [if_neg (by simp; omega)]
  simp

theorem buf0_writable {bufs : Nat → ByteArray} {nan : ClifTy → UInt64 → UInt64} {o : Nat} (ho : o < 2 ^ 36)
    (hs : o < (bufs 0).size) : Writable (launchWorld bufs nan).mem (base0 + UInt64.ofNat o) :=
  ⟨.buf 0, o, buf0_decode ho, by simp [Mem.region]; omega, by simp⟩

/-- A byte of buffer 0 that loads is the byte the buffer holds. -/
theorem buf0_load_eq {m : Mem} {o : Nat} {v : UInt64} (ho : o < 2 ^ 36)
    (h : m.load (base0 + UInt64.ofNat o) 1 = some v) : v = ((m.bufs 0).get! o).toUInt64 := by
  simp only [Mem.load, buf0_decode ho, Option.bind_eq_bind, Option.bind_some, Mem.region] at h
  split at h
  · cases h
  · simp at h; exact h.symm

/-- A byte of buffer 0 that loads lies inside it. -/
theorem buf0_load_lt {m : Mem} {o : Nat} {v : UInt64} (ho : o < 2 ^ 36)
    (h : m.load (base0 + UInt64.ofNat o) 1 = some v) : o < (m.bufs 0).size := by
  simp only [Mem.load, buf0_decode ho, Option.bind_eq_bind, Option.bind_some, Mem.region] at h
  split at h
  · cases h
  · rename_i hc; simp at hc; omega

/-- The header's byte `j`'s address. -/
theorem hdr_idx (ob : UInt64) (lenOff j : Nat) (hov : ob.toNat + lenOff + 4 < 2 ^ 64) (hj : j < 4) :
    (ob + UInt64.ofNat lenOff + UInt64.ofNat j).toNat = ob.toNat + lenOff + j := by
  rw [UInt64.toNat_add, UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show lenOff < 2 ^ 64 by omega),
    UInt64.toNat_ofNat_of_lt' (show j < 2 ^ 64 by omega), Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)]

/-- The length header stores the cursor's four low bytes at `outBase + lenOff` on. -/
theorem lenHeader_eval (lenOff f : Nat) (S : WState) (hov : (S.regs "outBase").toNat + lenOff + 4 < 2 ^ 64) :
    ((wseq (lenHeader lenOff)).eval f S).gmem =
      ((((S.gmem.setIfInBounds ((S.regs "outBase").toNat + lenOff) (S.regs "op").toUInt8).setIfInBounds
        ((S.regs "outBase").toNat + lenOff + 1) (S.regs "op" >>> 8).toUInt8).setIfInBounds
        ((S.regs "outBase").toNat + lenOff + 2) (S.regs "op" >>> 16).toUInt8).setIfInBounds
        ((S.regs "outBase").toNat + lenOff + 3) (S.regs "op" >>> 24).toUInt8) := by
  have h0 := hdr_idx (S.regs "outBase") lenOff 0 hov (by omega)
  simp only [Nat.add_zero, show UInt64.ofNat 0 = 0 from rfl, UInt64.add_zero] at h0
  simp only [lenHeader, wseq, WStmt.eval, WState.setReg, WState.stgByte, WArg.eval, WOp.run, if_true,
    String.reduceEq, if_false, Array.set!_eq_setIfInBounds]
  rw [h0, hdr_idx _ _ 1 hov (by omega), hdr_idx _ _ 2 hov (by omega), hdr_idx _ _ 3 hov (by omega)]
  simp only [show UInt64.ofNat 255 = 255 from rfl, show UInt64.ofNat 8 = 8 from rfl, show UInt64.ofNat 16 = 16 from rfl,
    show UInt64.ofNat 24 = 24 from rfl, and255_toUInt8]

/-- The length header leaves every byte below it. -/
theorem lenHeader_keeps (lenOff f : Nat) (S : WState) {a : Nat}
    (hov : (S.regs "outBase").toNat + lenOff + 4 < 2 ^ 64) (ha : a < (S.regs "outBase").toNat + lenOff) :
    ((wseq (lenHeader lenOff)).eval f S).gmem.getD a 0 = S.gmem.getD a 0 := by
  rw [lenHeader_eval lenOff f S hov]
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_setIfInBounds]
  rw [if_neg (by omega), if_neg (by omega), if_neg (by omega), if_neg (by omega)]

/-- The length header reads back as the cursor, below `2 ^ 32`. -/
theorem lenHeader_reads (lenOff f : Nat) (S : WState) (hov : (S.regs "outBase").toNat + lenOff + 4 < 2 ^ 64)
    (hsz : (S.regs "outBase").toNat + lenOff + 4 ≤ S.gmem.size) :
    readU32LE ((wseq (lenHeader lenOff)).eval f S).gmem ((S.regs "outBase").toNat + lenOff) = (S.regs "op").toNat % 2 ^ 32 := by
  rw [lenHeader_eval lenOff f S hov]
  simp only [readU32LE, Array.getD_eq_getD_getElem?, Array.getElem?_setIfInBounds, Array.size_setIfInBounds]
  simp only [Nat.add_left_cancel_iff, show ((S.regs "outBase").toNat + lenOff + 3 < S.gmem.size) = True by simp; omega,
    show ((S.regs "outBase").toNat + lenOff + 2 < S.gmem.size) = True by simp; omega,
    show ((S.regs "outBase").toNat + lenOff + 1 < S.gmem.size) = True by simp; omega,
    show ((S.regs "outBase").toNat + lenOff < S.gmem.size) = True by simp; omega]
  simp only [toUInt8_toNat, Option.getD_some, if_true]
  rw [if_neg (by omega), if_neg (by omega), if_neg (by omega)]
  simp only [show ¬ (3 : Nat) = 1 by decide, show ¬ (2 : Nat) = 1 by decide, show ¬ (3 : Nat) = 2 by decide, if_false,
    Option.getD_some, toUInt8_toNat, UInt64.toNat_shiftRight, Nat.shiftRight_eq_div_pow]
  simp only [show (24 : UInt64).toNat % 64 = 24 from rfl, show (16 : UInt64).toNat % 64 = 16 from rfl,
    show (8 : UInt64).toNat % 64 = 8 from rfl]
  omega

/-- The little-endian 32-bit number at byte `o` of `b`. -/
def u32At (b : ByteArray) (o : Nat) : Nat :=
  (b.get! o).toNat + 256 * (b.get! (o + 1)).toNat + 65536 * (b.get! (o + 2)).toNat + 16777216 * (b.get! (o + 3)).toNat

/-- Block `c`'s input: `inStride` bytes at offset `c * inStride` of buffer 0. -/
def inAt (inStride c : Nat) : UInt64 := base0 + UInt64.ofNat (c * inStride)

/-- Block `c`'s output: past every input and the copy slack, `outStride` bytes a block. -/
def outAt (numBlocks inStride outStride c : Nat) : UInt64 :=
  base0 + UInt64.ofNat (numBlocks * inStride + copySlack + c * outStride)

theorem startWS_in (numBlocks inStride outStride hashLog c : Nat) (G : Array UInt8) :
    (startWS numBlocks inStride outStride hashLog c base0 G).regs "inBase" = inAt inStride c := by
  simp only [startWS, regs0, if_true, inAt, c64_ofNat, ← UInt64.ofNat_mul, Nat.mul_comm c]

theorem startWS_out (numBlocks inStride outStride hashLog c : Nat) (G : Array UInt8) :
    (startWS numBlocks inStride outStride hashLog c base0 G).regs "outBase" = outAt numBlocks inStride outStride c := by
  simp only [startWS, regs0, String.reduceEq, if_false, if_true, outAt, c64_ofNat, ← UInt64.ofNat_mul,
    UInt64.add_assoc, Nat.mul_comm c]
  rw [show ((numBlocks : Int) * inStride + copySlack) = ((numBlocks * inStride + copySlack : Nat) : Int) by push_cast; rfl,
    c64_ofNat, ← UInt64.ofNat_add]

theorem inAt_add (inStride c j : Nat) : inAt inStride c + UInt64.ofNat j = base0 + UInt64.ofNat (c * inStride + j) := by
  simp only [inAt, UInt64.add_assoc, ← UInt64.ofNat_add]

theorem outAt_add (numBlocks inStride outStride c j : Nat) :
    outAt numBlocks inStride outStride c + UInt64.ofNat j
      = base0 + UInt64.ofNat (numBlocks * inStride + copySlack + c * outStride + j) := by
  simp only [outAt, UInt64.add_assoc, ← UInt64.ofNat_add]

/-- What each block reads: its input and the 32 bytes after it (`blockIn`), which lie inside the
    next block's input or, for the last, the copy slack. -/
def inputs (numBlocks inStride c : Nat) (a : UInt64) : Prop := c < numBlocks ∧ blockIn (inAt inStride c) inStride a

/-- The blocks' outputs: each block's encoded block and length header. -/
def outputs (numBlocks inStride outStride lenOff c : Nat) (a : UInt64) : Prop :=
  c < numBlocks ∧ outputOf (outAt numBlocks inStride outStride c) (lenOff + 4) a

/-- **What a launch of the compressor leaves in buffer 0**, denoted by the
    warp's runs: from `i`, an `o` that keeps the inputs and the slack after
    them, still holds the whole layout, and holds in each block's output the
    bytes the warp's run of that block's body leaves. -/
def Denotes (numBlocks inStride outStride lenOff hashLog : Nat) (i o : ByteArray) : Prop :=
  (∀ j, j < numBlocks * inStride + copySlack → o.get! j = i.get! j) ∧
  numBlocks * inStride + copySlack + numBlocks * outStride ≤ o.size ∧
  ∀ c, c < numBlocks → ∀ q, q < lenOff + 4 →
    o.get! (numBlocks * inStride + copySlack + c * outStride + q) =
      ((compressorBodyWStmt inStride lenOff hashLog).eval (inStride + 34 * inStride)
        (startWS numBlocks inStride outStride hashLog c base0 (image i))).gmem.getD
        (base0.toNat + (numBlocks * inStride + copySlack + c * outStride + q)) 0

set_option maxHeartbeats 4000000 in
/-- **A launch of the compressor leaves what the warp's runs denote**, on the
    Code and on the PTX it compiles to: with the blocks' inputs at the start
    of device buffer 0 and their outputs after them, the launch runs, and
    buffer 0 after it is as `Denotes` says. -/
theorem launch_denotes (hw : Hw) (hhw : PtxSound.HwOk hw) (name : String) {e : PtxIr.Entry}
    {numBlocks inStride outStride lenOff hashLog : Nat} (outA : UInt64)
    (he : (codeKernel name numBlocks inStride outStride lenOff hashLog).entry = .ok e) (bufs : Nat → ByteArray)
    (hnb : numBlocks * 32 + 32 < 2 ^ 32) (hh : 2 * 2 ^ hashLog + 64 < 2 ^ 36) (hh32 : hashLog ≤ 32)
    (hfine : Fine (emitGo (fnBody (idParams ++ [.i64, .i64]) (compressor numBlocks inStride outStride lenOff hashLog))
      ⟨(idParams ++ [ClifTy.i64, .i64]).length, 0, [], [], [], none⟩).2)
    (hflags : ∀ r, r ∈ ["inBase","outBase","tbl","op","searchPos","litAnchor"] →
      (flagRegs (compressorBodyWStmt inStride lenOff hashLog)).contains r = false)
    (hst : 16 ≤ inStride) (hst2 : inStride ≤ 65536) (hlen : inStride + inStride / 16 + 256 ≤ lenOff)
    (hlo : lenOff + 4 ≤ outStride)
    (hend : numBlocks * inStride + copySlack + numBlocks * outStride ≤ (bufs 0).size)
    (hend36 : numBlocks * inStride + copySlack + numBlocks * outStride ≤ 2 ^ 36) :
    ∃ u : World, LaunchRuns (codeKernel name numBlocks inStride outStride lenOff hashLog) numBlocks 32
        [.sc .i64 base0, .sc .i64 outA] bufs u.mem.bufs hw ∧
      RunsLaunch (PtxRun.runner e hw) (codeKernel name numBlocks inStride outStride lenOff hashLog).smem numBlocks 32
        [.sc .i64 base0, .sc .i64 outA] bufs u.mem.bufs hw.nan ∧
      Denotes numBlocks inStride outStride lenOff hashLog (bufs 0) (u.mem.bufs 0) := by
  have hcs : copySlack = 32 := rfl
  -- where block `c`'s bytes are
  have hinc : ∀ c, c < numBlocks → c * inStride + inStride ≤ numBlocks * inStride := fun c hc => by
    have := Nat.mul_le_mul_right inStride (show c + 1 ≤ numBlocks by omega); rw [Nat.succ_mul] at this; exact this
  have houtc : ∀ c, c < numBlocks → c * outStride + outStride ≤ numBlocks * outStride := fun c hc => by
    have := Nat.mul_le_mul_right outStride (show c + 1 ≤ numBlocks by omega); rw [Nat.succ_mul] at this; exact this
  generalize hF : inStride + 34 * inStride = F
  let ws := fun c => startWS numBlocks inStride outStride hashLog c base0 (image (bufs 0))
  have hws : ∀ c, startWS numBlocks inStride outStride hashLog c base0 (image (bufs 0)) = ws c := fun _ => rfl
  have r0 : ∀ c r, r ≠ "inBase" → r ≠ "outBase" → (ws c).regs r = 0 := by
    intro c r h1 h2; simp [ws, startWS, regs0, h1, h2]
  have hinN : ∀ c, c < numBlocks → (inAt inStride c).toNat = base0.toNat + c * inStride := fun c hc => by
    rw [inAt]; exact buf0_addr (by have := hinc c hc; omega)
  have houtN : ∀ c, c < numBlocks →
      (outAt numBlocks inStride outStride c).toNat = base0.toNat + (numBlocks * inStride + copySlack + c * outStride) :=
    fun c hc => by rw [outAt]; exact buf0_addr (by have := houtc c hc; omega)
  have hsmem : ∀ c, (ws c).smem.size = 2 ^ hashLog * 2 := fun c => clearMem_size hashLog
  have hgsz : ∀ c, (ws c).gmem.size = base0.toNat + 2 ^ 37 := fun c => image_size _
  have hbo : ∀ c, c < numBlocks →
      okOf (blockIn (inAt inStride c) inStride) (outputOf (outAt numBlocks inStride outStride c) (lenOff + 4))
        (okLZ4 (blockIn (inAt inStride c) inStride)) F (compressorBodyWStmt inStride lenOff hashLog) (ws c) ∧
      (gstOf F (compressorBodyWStmt inStride lenOff hashLog) (ws c)).Pairwise (fun x y => x.2.1 ≠ y.2.1) := by
    intro c hc
    have := hinc c hc
    have := houtc c hc
    rw [← hF]
    refine body_obligations (ws c) hst hst2 hh32 (by omega) (r0 c _ (by decide) (by decide))
      (r0 c _ (by decide) (by decide)) (r0 c _ (by decide) (by decide)) (startWS_in _ _ _ _ _ _)
      (startWS_out _ _ _ _ _ _) (by rw [hinN c hc, base0_toNat]; omega) (by rw [houtN c hc, base0_toNat]; omega)
      (by rw [houtN c hc, hgsz]; omega) (by rw [hinN c hc, houtN c hc]; omega) (r0 c _ (by decide) (by decide))
      (by rw [hsmem]; omega) hlen (by rw [houtN c hc, base0_toNat]; omega) (fun j hj => ?_)
    rw [outAt_add]; exact buf0_notArena (by omega)
  have hI : ∀ c, c < numBlocks → inputs numBlocks inStride c = blockIn (inAt inStride c) inStride :=
    fun c hc => funext fun a => propext ⟨fun h => h.2, fun h => ⟨hc, h⟩⟩
  have hO : ∀ c, c < numBlocks → outputs numBlocks inStride outStride lenOff c
      = outputOf (outAt numBlocks inStride outStride c) (lenOff + 4) :=
    fun c hc => funext fun a => propext ⟨fun h => h.2, fun h => ⟨hc, h⟩⟩
  -- each address of a block's input or output is a byte of buffer 0
  have hInB : ∀ c a, inputs numBlocks inStride c a →
      ∃ o, o < numBlocks * inStride + copySlack ∧ a = base0 + UInt64.ofNat o := by
    rintro c a ⟨hc, j, hj, rfl⟩
    exact ⟨c * inStride + j, by have := hinc c hc; omega, inAt_add _ _ _⟩
  have hOutB : ∀ c a, outputs numBlocks inStride outStride lenOff c a →
      ∃ j, j < lenOff + 4 ∧ a = base0 + UInt64.ofNat (numBlocks * inStride + copySlack + c * outStride + j) := by
    rintro c a ⟨hc, j, hj, rfl⟩
    exact ⟨j, hj, outAt_add _ _ _ _ _⟩
  have hbyte : ∀ c a, inputs numBlocks inStride c a ∨ outputs numBlocks inStride outStride lenOff c a →
      ∃ o, o < numBlocks * inStride + copySlack + numBlocks * outStride ∧ a = base0 + UInt64.ofNat o := by
    rintro c a (h | h)
    · obtain ⟨o, ho, rfl⟩ := hInB c a h; exact ⟨o, by omega, rfl⟩
    · obtain ⟨j, hj, rfl⟩ := hOutB c a h
      exact ⟨_, by have := houtc c h.1; omega, rfl⟩
  obtain ⟨u, hL, hP, hout, hframe⟩ := launch_output (outA := outA) (f := F) hw hhw name he
    (inputs numBlocks inStride) (outputs numBlocks inStride outStride lenOff) (fun _ => image (bufs 0)) bufs
    hnb hh hfine hflags
    (fun c a h => by obtain ⟨o, ho, rfl⟩ := hbyte c a (.inl h); exact buf0_notArena (by omega))
    (fun c a h => by obtain ⟨o, ho, rfl⟩ := hbyte c a (.inr h); exact buf0_notArena (by omega))
    (fun c hc => by rw [hI c hc, hO c hc]; exact (hbo c hc).1)
    (fun c hc => (hbo c hc).2)
    (fun c hc a h => by
      obtain ⟨o, ho, rfl⟩ := hbyte c a h
      rw [buf0_load (by omega) (by omega), buf0_addr (by omega), image_getD _ (by omega)])
    (fun c hc a h => by
      obtain ⟨o, ho, rfl⟩ := hbyte c a (.inr h)
      show _ < (image (bufs 0)).size
      rw [buf0_addr (by omega), image_size]; omega)
    (fun c hc a h => by obtain ⟨o, ho, rfl⟩ := hbyte c a (.inr h); exact buf0_writable (by omega) (by omega))
    (fun c c' hlt hc a h h' => by
      obtain ⟨j, hj, rfl⟩ := hOutB c' a h
      obtain ⟨o, ho, e⟩ := hInB c _ h'
      have he := congrArg UInt64.toNat e
      have := houtc c' (by omega)
      rw [buf0_addr (by omega), buf0_addr (by omega)] at he
      omega)
    (fun c c' hc hc' hne a h h' => by
      obtain ⟨j, hj, rfl⟩ := hOutB c' a h
      obtain ⟨j', hj', e⟩ := hOutB c _ h'
      have he := congrArg UInt64.toNat e
      have := houtc c hc
      have := houtc c' hc'
      rw [buf0_addr (by omega), buf0_addr (by omega)] at he
      rcases Nat.lt_or_gt_of_ne hne with hlt | hlt
      · have := Nat.mul_le_mul_right outStride (show c' + 1 ≤ c by omega); rw [Nat.succ_mul] at this; omega
      · have := Nat.mul_le_mul_right outStride (show c + 1 ≤ c' by omega); rw [Nat.succ_mul] at this; omega)
  -- a byte outside every output is as it was
  have hkeep : ∀ o, o < numBlocks * inStride + copySlack →
      u.mem.load (base0 + UInt64.ofNat o) 1 = some ((bufs 0).get! o).toUInt64 := by
    intro o ho
    rw [hframe _ (buf0_notArena (by omega)) (fun c hc h => ?_), buf0_load (by omega) (by omega)]
    obtain ⟨j, hj, e⟩ := hOutB c _ h
    have := congrArg UInt64.toNat e
    rw [buf0_addr (by omega), buf0_addr (by have := houtc c hc; omega)] at this
    omega
  have hsize' : numBlocks * inStride + copySlack + numBlocks * outStride ≤ (u.mem.bufs 0).size := by
    rcases Nat.eq_zero_or_pos numBlocks with h0 | hpos
    · subst h0
      have := buf0_load_lt (o := 0 * inStride + copySlack - 1) (by omega) (hkeep _ (by omega))
      omega
    · -- the last byte of the last block's stride: an output byte, or kept
      have hlast : (numBlocks - 1) * outStride + outStride = numBlocks * outStride := by
        rw [← Nat.succ_mul, Nat.succ_eq_add_one, Nat.sub_add_cancel hpos]
      have hsz : numBlocks * inStride + copySlack + (numBlocks - 1) * outStride + lenOff + 3
          < (u.mem.bufs 0).size := by
        have ha : outputs numBlocks inStride outStride lenOff (numBlocks - 1)
            (outAt numBlocks inStride outStride (numBlocks - 1) + UInt64.ofNat (lenOff + 3)) :=
          ⟨by omega, lenOff + 3, by omega, rfl⟩
        have hl := hout (numBlocks - 1) (by omega) _ ha
        rw [outAt_add] at hl
        exact buf0_load_lt (by omega) hl
      by_cases hgap : (numBlocks - 1) * outStride + outStride ≤ (numBlocks - 1) * outStride + lenOff + 4
      · omega
      · -- bytes past the last header: kept by the launch, so they still load
        have hlt : numBlocks * inStride + copySlack + numBlocks * outStride - 1 < (u.mem.bufs 0).size := by
          have hna : NotArena (base0 + UInt64.ofNat (numBlocks * inStride + copySlack + numBlocks * outStride - 1)) :=
            buf0_notArena (by omega)
          have h := hframe _ hna (fun c hc h => by
            obtain ⟨j, hj, e⟩ := hOutB c _ h
            have he := congrArg UInt64.toNat e
            have := houtc c hc
            rw [buf0_addr (by omega), buf0_addr (by omega)] at he
            rcases Nat.lt_or_ge c (numBlocks - 1) with hc' | hc'
            · have := Nat.mul_le_mul_right outStride (show c + 1 ≤ numBlocks - 1 by omega)
              rw [Nat.succ_mul] at this; omega
            · have : c = numBlocks - 1 := by omega
              subst this; omega)
          rw [buf0_load (by omega) (by omega)] at h
          exact buf0_load_lt (by omega) h
        omega
  refine ⟨u, hL, hP, fun o ho => ?_, hsize', fun c hc q hq => ?_⟩
  · have := buf0_load_eq (by omega) (hkeep o ho)
    exact (UInt8.toUInt64_inj.mp this).symm
  have := houtc c hc
  have ha : outputs numBlocks inStride outStride lenOff c (outAt numBlocks inStride outStride c + UInt64.ofNat q) :=
    ⟨hc, q, hq, rfl⟩
  have hl := hout c hc _ ha
  rw [outAt_add] at hl
  have hb := buf0_load_eq (by omega) hl
  rw [buf0_addr (by omega), ← hF] at hb
  exact (UInt8.toUInt64_inj.mp hb).symm

set_option maxHeartbeats 4000000 in
/-- **What the warp's runs denote round-trips every block**: where buffer 0
    went from `i` to `o` as `Denotes` says, each block's output holds its
    encoded length `n ≤ lenOff` at `lenOff`, and its first `n` bytes decode to
    the block's input in `i`. -/
theorem decode_of_denotes {numBlocks inStride outStride lenOff hashLog : Nat} (hst : 16 ≤ inStride) (hst2 : inStride ≤ 65536)
    (hh32 : hashLog ≤ 32) (hh : 2 * 2 ^ hashLog + 64 < 2 ^ 36) (hlen : inStride + inStride / 16 + 256 ≤ lenOff)
    (hlo : lenOff + 4 ≤ outStride)
    (hend36 : numBlocks * inStride + copySlack + numBlocks * outStride ≤ 2 ^ 36) {i o : ByteArray}
    (hend : numBlocks * inStride + copySlack + numBlocks * outStride ≤ i.size)
    (hd : Denotes numBlocks inStride outStride lenOff hashLog i o) :
    ∀ c, c < numBlocks → ∃ n, n ≤ lenOff ∧
      u32At o (numBlocks * inStride + copySlack + c * outStride + lenOff) = n ∧
      AlgorithmLib.LZ4Imp.decompress ((List.range n).map fun i => o.get! (numBlocks * inStride + copySlack + c * outStride + i))
        inStride = some ((List.range inStride).map fun j => i.get! (c * inStride + j)) := by
  have hcs : copySlack = 32 := rfl
  -- where block `c`'s bytes are
  have hinc : ∀ c, c < numBlocks → c * inStride + inStride ≤ numBlocks * inStride := fun c hc => by
    have := Nat.mul_le_mul_right inStride (show c + 1 ≤ numBlocks by omega); rw [Nat.succ_mul] at this; exact this
  have houtc : ∀ c, c < numBlocks → c * outStride + outStride ≤ numBlocks * outStride := fun c hc => by
    have := Nat.mul_le_mul_right outStride (show c + 1 ≤ numBlocks by omega); rw [Nat.succ_mul] at this; exact this
  generalize hF : inStride + 34 * inStride = F
  let ws := fun c => startWS numBlocks inStride outStride hashLog c base0 (image i)
  have hws : ∀ c, startWS numBlocks inStride outStride hashLog c base0 (image i) = ws c := fun _ => rfl
  have r0 : ∀ c r, r ≠ "inBase" → r ≠ "outBase" → (ws c).regs r = 0 := by
    intro c r h1 h2; simp [ws, startWS, regs0, h1, h2]
  have hinN : ∀ c, c < numBlocks → (inAt inStride c).toNat = base0.toNat + c * inStride := fun c hc => by
    rw [inAt]; exact buf0_addr (by have := hinc c hc; omega)
  have houtN : ∀ c, c < numBlocks →
      (outAt numBlocks inStride outStride c).toNat = base0.toNat + (numBlocks * inStride + copySlack + c * outStride) :=
    fun c hc => by rw [outAt]; exact buf0_addr (by have := houtc c hc; omega)
  have hsmem : ∀ c, (ws c).smem.size = 2 ^ hashLog * 2 := fun c => clearMem_size hashLog
  have hgsz : ∀ c, (ws c).gmem.size = base0.toNat + 2 ^ 37 := fun c => image_size _
  intro c hc
  have := hinc c hc
  have := houtc c hc
  -- the model's run of block `c`: the encoding prefix, then the length header
  have hib := startWS_in numBlocks inStride outStride hashLog c (image i)
  have hob := startWS_out numBlocks inStride outStride hashLog c (image i)
  rw [hws] at hib hob
  have hilen : (gmemInpAt (ws c).gmem ((ws c).regs "inBase").toNat inStride).length = inStride := by simp [gmemInpAt]
  have hv := evalPlan_valid (ws c).gmem (ws c).smem ((ws c).regs "inBase").toNat inStride hashLog hst2
  have h9 := planBlock_encode_le9 _ _ hv (by rw [hilen]; omega)
  have hLo := planBlock_encode_le_lenOff _ _ hv inStride hilen
  rw [hilen] at h9
  have hpre := compressorBody_output_eq inStride hashLog (ws c) hst2 (by omega) (by omega)
    (r0 c _ (by decide) (by decide)) (r0 c _ (by decide) (by decide)) (r0 c _ (by decide) (by decide))
    (by rw [hib, hinN c hc, base0_toNat]; omega) (by rw [hob, houtN c hc, base0_toNat]; omega)
    (by rw [hob, houtN c hc, hgsz]; omega) (by rw [hib, hob, hinN c hc, houtN c hc]; omega)
    (by rw [hib, hinN c hc, base0_toNat]; omega) h9
  have hrt := warpKernelDSL_body_roundtrips inStride hashLog (ws c) hst2 (by omega) (by omega)
    (r0 c _ (by decide) (by decide)) (r0 c _ (by decide) (by decide)) (r0 c _ (by decide) (by decide))
    (by rw [hib, hinN c hc, base0_toNat]; omega) (by rw [hob, houtN c hc, base0_toNat]; omega)
    (by rw [hob, houtN c hc, hgsz]; omega) (by rw [hib, hob, hinN c hc, houtN c hc]; omega)
    (by rw [hib, hinN c hc, base0_toNat]; omega) h9
  rw [hF] at hpre hrt
  have hk := Nat.le_trans hLo hlen
  -- the model's full run is the length header's run after the prefix's
  have hsplit := (wseq_append (I := blockIn (inAt inStride c) inStride) (okC := okLZ4 (blockIn (inAt inStride c) inStride))
    (prefixList inStride hashLog) (lenHeader lenOff) (by simp [prefixList]) (by simp [lenHeader]) F (ws c)).1
  rw [← body_split] at hsplit
  simp only [WStmt.eval] at hsplit
  generalize hP : (wseq (prefixList inStride hashLog)).eval F (ws c) = P at hsplit
  have hob2 : P.regs "outBase" = outAt numBlocks inStride outStride c := by
    rw [← hP, ← prefix_eq, hpre.2.2, hob]
  have hop2 : P.regs "op" = UInt64.ofNat (planToBlock (gmemInpAt (ws c).gmem ((ws c).regs "inBase").toNat inStride)
      (evalPlan (ws c).gmem (ws c).smem ((ws c).regs "inBase").toNat inStride hashLog)).encode.length := by
    rw [← hP, ← prefix_eq, hpre.2.1]
  have hPsz : P.gmem.size = base0.toNat + 2 ^ 37 := by rw [← hP, eval_size, hgsz]
  -- each byte of the output on the device is the model's
  have hdev : ∀ j, j < lenOff + 4 → o.get! (numBlocks * inStride + copySlack + c * outStride + j) =
      ((wseq (lenHeader lenOff)).eval F P).gmem.getD ((P.regs "outBase").toNat + j) 0 := by
    intro j hj
    have hb := hd.2.2 c hc j hj
    rw [hF, hws, hsplit] at hb
    rw [hb, hob2, houtN c hc, Nat.add_assoc base0.toNat]
  refine ⟨_, hk, ?_, ?_⟩
  · -- the length header holds the encoded length
    have e1 := hdev (lenOff + 1) (by omega)
    have e2 := hdev (lenOff + 2) (by omega)
    have e3 := hdev (lenOff + 3) (by omega)
    simp only [← Nat.add_assoc] at e1 e2 e3
    simp only [u32At]
    rw [hdev lenOff (by omega), e1, e2, e3]
    have := lenHeader_reads lenOff F P (by rw [hob2, houtN c hc, base0_toNat]; omega)
      (by rw [hob2, houtN c hc, hPsz]; omega)
    simp only [readU32LE] at this
    rw [this, hop2, UInt64.toNat_ofNat_of_lt' (show _ < 2 ^ 64 by omega)]
    omega
  -- the device's input bytes are the model's
  have hin : gmemInpAt (ws c).gmem ((ws c).regs "inBase").toNat inStride
      = (List.range inStride).map fun j => i.get! (c * inStride + j) := by
    simp only [gmemInpAt]
    apply List.map_congr_left
    intro j hj
    rw [List.mem_range] at hj
    rw [hib, hinN c hc, Nat.add_assoc]
    exact image_getD _ (by omega)
  rw [hilen] at hrt
  refine Eq.trans (congrArg (fun L => AlgorithmLib.LZ4Imp.decompress L inStride) ?_) (hrt.trans (congrArg some hin))
  apply List.map_congr_left
  intro i hi
  rw [List.mem_range] at hi
  -- below the header, the full run's bytes are the prefix's
  rw [hdev i (by omega), lenHeader_keeps lenOff F P (by rw [hob2, houtN c hc, base0_toNat]; omega)
    (by rw [hob2]; omega), hob2, ← hob, ← hP, prefix_eq]
set_option maxHeartbeats 4000000 in
/-- **A launch of the compressor round-trips every block, on the Code and on
    the PTX it compiles to**: with the blocks' inputs at the start of device
    buffer 0 and their outputs after them, the launch runs, and each block's
    output begins with an encoded block, at most `lenOff` bytes long, that the
    LZ4 decompressor turns back into the block's input, and holds its length
    at `lenOff`; the inputs and the slack after them are as they were, and
    buffer 0 still holds the whole layout, so the launch can run again. -/
theorem launch_roundtrips (hw : Hw) (hhw : PtxSound.HwOk hw) (name : String) {e : PtxIr.Entry}
    {numBlocks inStride outStride lenOff hashLog : Nat} (outA : UInt64)
    (he : (codeKernel name numBlocks inStride outStride lenOff hashLog).entry = .ok e) (bufs : Nat → ByteArray)
    (hnb : numBlocks * 32 + 32 < 2 ^ 32) (hh : 2 * 2 ^ hashLog + 64 < 2 ^ 36) (hh32 : hashLog ≤ 32)
    (hfine : Fine (emitGo (fnBody (idParams ++ [.i64, .i64]) (compressor numBlocks inStride outStride lenOff hashLog))
      ⟨(idParams ++ [ClifTy.i64, .i64]).length, 0, [], [], [], none⟩).2)
    (hflags : ∀ r, r ∈ ["inBase","outBase","tbl","op","searchPos","litAnchor"] →
      (flagRegs (compressorBodyWStmt inStride lenOff hashLog)).contains r = false)
    (hst : 16 ≤ inStride) (hst2 : inStride ≤ 65536) (hlen : inStride + inStride / 16 + 256 ≤ lenOff)
    (hlo : lenOff + 4 ≤ outStride)
    (hend : numBlocks * inStride + copySlack + numBlocks * outStride ≤ (bufs 0).size)
    (hend36 : numBlocks * inStride + copySlack + numBlocks * outStride ≤ 2 ^ 36) :
    ∃ u : World, LaunchRuns (codeKernel name numBlocks inStride outStride lenOff hashLog) numBlocks 32
        [.sc .i64 base0, .sc .i64 outA] bufs u.mem.bufs hw ∧
      RunsLaunch (PtxRun.runner e hw) (codeKernel name numBlocks inStride outStride lenOff hashLog).smem numBlocks 32
        [.sc .i64 base0, .sc .i64 outA] bufs u.mem.bufs hw.nan ∧
      (∀ o, o < numBlocks * inStride + copySlack → (u.mem.bufs 0).get! o = (bufs 0).get! o) ∧
      numBlocks * inStride + copySlack + numBlocks * outStride ≤ (u.mem.bufs 0).size ∧
      ∀ c, c < numBlocks → ∃ n, n ≤ lenOff ∧
        u32At (u.mem.bufs 0) (numBlocks * inStride + copySlack + c * outStride + lenOff) = n ∧
        AlgorithmLib.LZ4Imp.decompress ((List.range n).map fun i =>
            (u.mem.bufs 0).get! (numBlocks * inStride + copySlack + c * outStride + i)) inStride
          = some ((List.range inStride).map fun j => (bufs 0).get! (c * inStride + j)) := by
  obtain ⟨u, hL, hP, hd⟩ := launch_denotes hw hhw name outA he bufs hnb hh hh32 hfine hflags hst hst2 hlen hlo hend hend36
  exact ⟨u, hL, hP, hd.1, hd.2.1,
    decode_of_denotes hst hst2 hh32 hh hlen hlo hend36 hend hd⟩

end Lz4Code
