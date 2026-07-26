import AlgorithmLib.LZ4SimtEmit
import AlgorithmLib.LZ4WarpColl
set_option maxRecDepth 8192

namespace AlgorithmLib.LZ4Simt
open AlgorithmLib

-- ── Initial state for one warp compressing block `inp` (length = inStride) ──────

/-- Registers pre-set at launch for the warp in CTA `w`: `%tid.x = lane`,
    `%ctaid.x = w`, `%ntid.x = modelBlockDim`; `in_ptr = 0`, `out_ptr = iTot`. -/
def initRegs (w iTot : Nat) : String → Fin 32 → UInt64 := fun name lane =>
  if name = "in_ptr" then 0
  else if name = "out_ptr" then UInt64.ofNat iTot
  else if name = "%tid.x" then UInt64.ofNat lane.val
  else if name = "%ctaid.x" then UInt64.ofNat w
  else if name = "%ntid.x" then 32
  else 0

/-- The prologue's `gwarp = (ctaid*ntid + tid) >>> 5` is the CTA index `w`, since
    one CTA holds exactly one warp (`ntid = 32`, `tid = lane < 32`). -/
theorem u64_gwarp (w : Nat) (hw : w * 32 + 32 < 2 ^ 64) (l : Fin 32) :
    (UInt64.ofNat w * (32 : UInt64) + UInt64.ofNat l.val) >>> UInt64.ofNat 5
      = UInt64.ofNat w := by
  have hl := l.isLt
  have hwlt : w < 2 ^ 64 := by omega
  have h32 : ((32 : UInt64)).toNat = 32 := by decide
  have h5 : (UInt64.ofNat 5).toNat = 5 := by decide
  have e1 : (UInt64.ofNat w * (32 : UInt64) + UInt64.ofNat l.val).toNat = w * 32 + l.val := by
    rw [UInt64.toNat_add, UInt64.toNat_mul, h32,
      AlgorithmLib.LZ4Ptx.toNat_ofNat_lt w hwlt,
      AlgorithmLib.LZ4Ptx.toNat_ofNat_lt l.val (by omega),
      Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)]
  rw [← UInt64.toNat_inj, AlgorithmLib.LZ4Ptx.toNat_ofNat_lt w hwlt,
    UInt64.toNat_shiftRight, e1, h5, Nat.mod_eq_of_lt (by omega),
    Nat.shiftRight_eq_div_pow]
  omega

/-- Launch memory: the input block followed by an output region of ARBITRARY
    content (a relaunch sees the previous run's bytes), and shared memory of
    ARBITRARY content (real `.shared` is uninitialized at launch — the kernel's
    clear loop is what makes it usable). -/
def initGmem (inp outB : List UInt8) : Array UInt8 := inp.toArray ++ outB.toArray

def initSmem (smemB : List UInt8) : Array UInt8 := smemB.toArray

def initSt (w : Nat) (inp outB smemB : List UInt8) : SState :=
  { regs := initRegs w inp.length
    gmem := initGmem inp outB
    smem := initSmem smemB
    pc := 0 }

/-- The block dimension this launch model assumes (one warp per CTA).  The shipped
    `wBlockDim` is DEFINED to be this, and `initRegs_ntid` proves `initRegs` really
    returns it — so the model and the launch cannot drift apart. -/
def modelBlockDim : Nat := 32

theorem initRegs_ntid (w iTot : Nat) (l : Fin 32) :
    initRegs w iTot "%ntid.x" l = UInt64.ofNat modelBlockDim := rfl

def clearIters (hashLog : Nat) : Nat :=
  if hashLog < 5 then 1 else 2 ^ (hashLog - 5)

theorem clearIters_exit_bound (hashLog : Nat) (hHash : hashLog ≤ 32) :
    2 ^ hashLog ≤ 32 * clearIters hashLog := by
  by_cases hlt : hashLog < 5
  · simp [clearIters, hlt]
    exact Nat.le_trans
      (Nat.pow_le_pow_right (show 0 < 2 by decide) (by omega : hashLog ≤ 5))
      (by decide : 2 ^ 5 ≤ 32)
  · have hge : 5 ≤ hashLog := by omega
    simp [clearIters, hlt]
    have hsplit : hashLog = hashLog - 5 + 5 := by omega
    rw [hsplit, Nat.pow_add]
    change 2 ^ (hashLog - 5) * 32 ≤ 32 * 2 ^ (hashLog - 5)
    rw [Nat.mul_comm (2 ^ (hashLog - 5)) 32]
    exact Nat.le_refl _

theorem clearIters_body_bound (hashLog k : Nat) (hHash : hashLog ≤ 32)
    (hk : k < clearIters hashLog) :
    32 * k < 2 ^ hashLog := by
  by_cases hlt : hashLog < 5
  · simp [clearIters, hlt] at hk
    have hk0 : k = 0 := by omega
    subst hk0
    exact Nat.pow_pos (show 0 < 2 by decide)
  · have hge : 5 ≤ hashLog := by omega
    simp [clearIters, hlt] at hk
    have hsplit : hashLog = hashLog - 5 + 5 := by omega
    rw [hsplit, Nat.pow_add]
    change 32 * k < 2 ^ (hashLog - 5) * 32
    rw [Nat.mul_comm (2 ^ (hashLog - 5)) 32]
    exact Nat.mul_lt_mul_of_pos_left hk (by omega)

theorem clearIters_lt_u64 (hashLog : Nat) (hHash : hashLog ≤ 32) :
    clearIters hashLog < 2 ^ 64 := by
  by_cases hlt : hashLog < 5
  · simp [clearIters, hlt]
  · have hminus : hashLog - 5 ≤ 27 := by omega
    have hpow : 2 ^ (hashLog - 5) ≤ 2 ^ 27 :=
      Nat.pow_le_pow_right (show 0 < 2 by decide) hminus
    simp [clearIters, hlt]
    exact Nat.lt_of_le_of_lt hpow (by decide : 2 ^ 27 < 2 ^ 64)

theorem clearIters_bytes_lt_u64 (hashLog : Nat) (hHash : hashLog ≤ 32) :
    32 * clearIters hashLog < 2 ^ 64 := by
  by_cases hlt : hashLog < 5
  · simp [clearIters, hlt]
  · have hminus : hashLog - 5 ≤ 27 := by omega
    have hpow : 2 ^ (hashLog - 5) ≤ 2 ^ 27 :=
      Nat.pow_le_pow_right (show 0 < 2 by decide) hminus
    have hmul : 32 * 2 ^ (hashLog - 5) ≤ 32 * 2 ^ 27 :=
      Nat.mul_le_mul_left 32 hpow
    simp [clearIters, hlt]
    exact Nat.lt_of_le_of_lt hmul (by decide : 32 * 2 ^ 27 < 2 ^ 64)

theorem hashEntries_lt_u64 (hashLog : Nat) (hHash : hashLog ≤ 32) :
    2 ^ hashLog < 2 ^ 64 := by
  have hlt : hashLog < 64 := by omega
  exact Nat.pow_lt_pow_right (show 1 < 2 by decide) hlt

/-- `inBase = in_ptr + gwarp*inStride` with `in_ptr = 0`. -/
theorem u64_inBase (w iS : Nat) :
    UInt64.ofNat w * UInt64.ofNat iS = UInt64.ofNat (w * iS) :=
  (UInt64.ofNat_mul w iS).symm

/-- `outBase = out_ptr + gwarp*outStride` with `out_ptr = iTot`. -/
theorem u64_outBase (iT k : Nat) :
    UInt64.ofNat iT + UInt64.ofNat k = UInt64.ofNat (iT + k) :=
  (UInt64.ofNat_add iT k).symm

theorem u64_and31 : ∀ l : Fin 32, UInt64.ofNat l.val &&& 31 = UInt64.ofNat l.val := by decide
theorem u64_shr5  : ∀ l : Fin 32, UInt64.ofNat l.val >>> 5 = 0 := by decide

theorem clearLoop_body_branch_false (hashLog k : Nat) (hHash : hashLog ≤ 32)
    (hk : k < clearIters hashLog) :
    ((if SCmp.run .ge (UInt64.ofNat (32 * k)) (UInt64.ofNat (2 ^ hashLog)) then
        (1 : UInt64)
      else
        0) == 1) = false := by
  have hb : 32 * k < 2 ^ hashLog := clearIters_body_bound hashLog k hHash hk
  have hprod : 32 * k < 2 ^ 64 := by
    have hpow : 2 ^ hashLog ≤ 2 ^ 32 :=
      Nat.pow_le_pow_right (show 0 < 2 by decide) hHash
    omega
  have hk64 : k < 2 ^ 64 := by
    have hci := clearIters_lt_u64 hashLog hHash
    omega
  have hright := hashEntries_lt_u64 hashLog hHash
  have hlt64 : 32 * UInt64.ofNat k < UInt64.ofNat (2 ^ hashLog) := by
    rw [UInt64.lt_iff_toNat_lt, UInt64.toNat_mul]
    rw [show UInt64.toNat (32 : UInt64) = 32 by decide]
    rw [AlgorithmLib.LZ4Ptx.toNat_ofNat_lt k hk64,
      AlgorithmLib.LZ4Ptx.toNat_ofNat_lt (2 ^ hashLog) hright]
    rw [Nat.mod_eq_of_lt hprod]
    exact hb
  have hcmp : SCmp.run .ge (UInt64.ofNat (32 * k)) (UInt64.ofNat (2 ^ hashLog)) = false := by
    simp [SCmp.run]
    exact hlt64
  rw [hcmp]
  decide

theorem clearLoop_exit_branch_true (hashLog : Nat) (hHash : hashLog ≤ 32) :
    ((if SCmp.run .ge (UInt64.ofNat (32 * clearIters hashLog)) (UInt64.ofNat (2 ^ hashLog)) then
        (1 : UInt64)
      else
        0) == 1) = true := by
  have hb : 2 ^ hashLog ≤ 32 * clearIters hashLog := clearIters_exit_bound hashLog hHash
  have hprod := clearIters_bytes_lt_u64 hashLog hHash
  have hk64 := clearIters_lt_u64 hashLog hHash
  have hright := hashEntries_lt_u64 hashLog hHash
  have hle64 : UInt64.ofNat (2 ^ hashLog) ≤ 32 * UInt64.ofNat (clearIters hashLog) := by
    rw [UInt64.le_iff_toNat_le, UInt64.toNat_mul]
    rw [show UInt64.toNat (32 : UInt64) = 32 by decide]
    rw [AlgorithmLib.LZ4Ptx.toNat_ofNat_lt (clearIters hashLog) hk64,
      AlgorithmLib.LZ4Ptx.toNat_ofNat_lt (2 ^ hashLog) hright]
    rw [Nat.mod_eq_of_lt hprod]
    exact hb
  have hcmp :
      SCmp.run .ge (UInt64.ofNat (32 * clearIters hashLog)) (UInt64.ofNat (2 ^ hashLog)) =
        true := by
    simp [SCmp.run]
    exact hle64
  rw [hcmp]
  decide

theorem prologue_clrDone_to_loop_slice (prog : Array SInstr) (st : SState)
    (h33 : prog[st.pc]? = some (.lbl "clrDone"))
    (h34 : prog[st.pc + 1]? = some .barwarp)
    (h35 : prog[st.pc + 2]? = some (.mov rLitAnchor (.imm 0)))
    (h36 : prog[st.pc + 3]? = some (.mov rSearchPos (.imm 0)))
    (h37 : prog[st.pc + 4]? = some (.mov rOp (.imm 0)))
    (hpc : st.pc = 33)
    (hlane : ∀ l : Fin 32, st.regs rLane l = UInt64.ofNat l.val) :
    (snsteps prog 5 st).pc = 38 ∧
      (∀ l : Fin 32, (snsteps prog 5 st).regs rLane l = UInt64.ofNat l.val) ∧
      (snsteps prog 5 st).regs rSearchPos 0 = 0 ∧
      (snsteps prog 5 st).regs rLitAnchor 0 = 0 ∧
      (snsteps prog 5 st).regs rOp 0 = 0 ∧
      (snsteps prog 5 st).gmem = st.gmem := by
  simp only [snsteps]
  rw [AlgorithmLib.LZ4WarpDSL.lbl_step prog st "clrDone" h33]
  let st34 := st.setPc (st.pc + 1)
  change (snsteps prog 4 st34).pc = 38 ∧
      (∀ l : Fin 32, (snsteps prog 4 st34).regs rLane l = UInt64.ofNat l.val) ∧
      (snsteps prog 4 st34).regs rSearchPos 0 = 0 ∧
      (snsteps prog 4 st34).regs rLitAnchor 0 = 0 ∧
      (snsteps prog 4 st34).regs rOp 0 = 0 ∧
      (snsteps prog 4 st34).gmem = st.gmem
  simp only [snsteps]
  rw [AlgorithmLib.LZ4WarpDSL.barwarp_step prog st34
    (by simpa [st34, SState.setPc, hpc] using h34)]
  let st35 := st34.setPc (st34.pc + 1)
  change (snsteps prog 3 st35).pc = 38 ∧
      (∀ l : Fin 32, (snsteps prog 3 st35).regs rLane l = UInt64.ofNat l.val) ∧
      (snsteps prog 3 st35).regs rSearchPos 0 = 0 ∧
      (snsteps prog 3 st35).regs rLitAnchor 0 = 0 ∧
      (snsteps prog 3 st35).regs rOp 0 = 0 ∧
      (snsteps prog 3 st35).gmem = st.gmem
  simp only [snsteps]
  rw [AlgorithmLib.LZ4WarpDSL.mov_step prog st35 rLitAnchor (.imm 0)
    (by simpa [st35, st34, SState.setPc, hpc] using h35)]
  let st36 := (st35.setReg rLitAnchor (fun l => st35.get l (.imm 0))).setPc (st35.pc + 1)
  change (snsteps prog 2 st36).pc = 38 ∧
      (∀ l : Fin 32, (snsteps prog 2 st36).regs rLane l = UInt64.ofNat l.val) ∧
      (snsteps prog 2 st36).regs rSearchPos 0 = 0 ∧
      (snsteps prog 2 st36).regs rLitAnchor 0 = 0 ∧
      (snsteps prog 2 st36).regs rOp 0 = 0 ∧
      (snsteps prog 2 st36).gmem = st.gmem
  simp only [snsteps]
  rw [AlgorithmLib.LZ4WarpDSL.mov_step prog st36 rSearchPos (.imm 0)
    (by simpa [st36, st35, st34, SState.setPc, hpc] using h36)]
  let st37 := (st36.setReg rSearchPos (fun l => st36.get l (.imm 0))).setPc (st36.pc + 1)
  change (snsteps prog 1 st37).pc = 38 ∧
      (∀ l : Fin 32, (snsteps prog 1 st37).regs rLane l = UInt64.ofNat l.val) ∧
      (snsteps prog 1 st37).regs rSearchPos 0 = 0 ∧
      (snsteps prog 1 st37).regs rLitAnchor 0 = 0 ∧
      (snsteps prog 1 st37).regs rOp 0 = 0 ∧
      (snsteps prog 1 st37).gmem = st.gmem
  simp only [snsteps]
  rw [AlgorithmLib.LZ4WarpDSL.mov_step prog st37 rOp (.imm 0)
    (by simpa [st37, st36, st35, st34, SState.setPc, hpc] using h37)]
  constructor
  · simp [st37, st36, st35, st34, SState.setPc, hpc]
  constructor
  · intro l
    simpa [st37, st36, st35, st34, SState.setReg, SState.setPc, SState.get, rLane,
      rLitAnchor, rSearchPos, rOp] using hlane l
  constructor
  · simp [st37, st36, st35, st34, SState.setReg, SState.setPc, SState.get, rSearchPos,
      rLitAnchor, rOp]
  constructor
  · simp [st37, st36, st35, st34, SState.setReg, SState.setPc, SState.get, rSearchPos,
      rLitAnchor, rOp]
  constructor
  · simp [st37, st36, st35, st34, SState.setReg, SState.setPc, SState.get, rSearchPos,
      rLitAnchor, rOp]
  · simp [st37, st36, st35, st34, SState.setReg, SState.setPc]

theorem prologue_clr_loop_body_slice (prog : Array SInstr) (st : SState) (entries : Nat)
    (h25 : prog[st.pc]? = some (.lbl "clr"))
    (h26 : prog[st.pc + 1]? = some (.setp .ge "pcnd" "ci" (.imm entries)))
    (h27 : prog[st.pc + 2]? = some (.braif "pcnd" "clrDone"))
    (h28 : prog[st.pc + 3]? = some (.bin .shl "ca" "ci" (.imm 1)))
    (h29 : prog[st.pc + 4]? = some (.binr .add "ca" "ca" rTbl))
    (h30 : prog[st.pc + 5]? = some (.stsh "ca" "z"))
    (h31 : prog[st.pc + 6]? = some (.bin .add "ci" "ci" (.imm 32)))
    (h32 : prog[st.pc + 7]? = some (.bra "clr"))
    (hfind : sfindLabel prog "clr" = 25)
    (hpc : st.pc = 25)
    (hbrHead :
      ((if SCmp.run .ge (st.regs "ci" 0) (st.get 0 (.imm entries)) then (1 : UInt64) else 0) ==
          1) = false)
    (hlane : ∀ l : Fin 32, st.regs rLane l = UInt64.ofNat l.val) :
    (snsteps prog 8 st).pc = 25 ∧
      (snsteps prog 8 st).regs "ci" 0 = st.regs "ci" 0 + 32 ∧
      (∀ l : Fin 32, (snsteps prog 8 st).regs rLane l = UInt64.ofNat l.val) ∧
      (snsteps prog 8 st).gmem = st.gmem := by
  simp only [snsteps]
  rw [AlgorithmLib.LZ4WarpDSL.lbl_step prog st "clr" h25]
  let st26 := st.setPc (st.pc + 1)
  change (snsteps prog 7 st26).pc = 25 ∧
    (snsteps prog 7 st26).regs "ci" 0 = st.regs "ci" 0 + 32 ∧
    (∀ l : Fin 32, (snsteps prog 7 st26).regs rLane l = UInt64.ofNat l.val) ∧
    (snsteps prog 7 st26).gmem = st.gmem
  simp only [snsteps]
  rw [AlgorithmLib.LZ4WarpDSL.setp_step prog st26 .ge "pcnd" "ci" (.imm entries)
    (by simpa [st26, SState.setPc, hpc] using h26)]
  let st27 := (st26.setReg "pcnd"
    (fun l => if SCmp.run .ge (st26.regs "ci" l) (st26.get l (.imm entries)) then 1 else 0)).setPc
      (st26.pc + 1)
  change (snsteps prog 6 st27).pc = 25 ∧
    (snsteps prog 6 st27).regs "ci" 0 = st.regs "ci" 0 + 32 ∧
    (∀ l : Fin 32, (snsteps prog 6 st27).regs rLane l = UInt64.ofNat l.val) ∧
    (snsteps prog 6 st27).gmem = st.gmem
  simp only [snsteps]
  rw [AlgorithmLib.LZ4WarpDSL.braif_step prog st27 "pcnd" "clrDone"
    (by simpa [st27, st26, SState.setPc, hpc] using h27)]
  have hbr : (st27.regs "pcnd" 0 == 1) = false := by
    simpa [st27, st26, SState.setReg, SState.setPc, SState.get] using hbrHead
  change (snsteps prog 5
      (st27.setPc (if st27.regs "pcnd" 0 == 1 then sfindLabel prog "clrDone" else st27.pc + 1))).pc =
      25 ∧
    (snsteps prog 5
      (st27.setPc (if st27.regs "pcnd" 0 == 1 then sfindLabel prog "clrDone" else st27.pc + 1))).regs "ci" 0 =
      st.regs "ci" 0 + 32 ∧
    (∀ l : Fin 32, (snsteps prog 5
      (st27.setPc (if st27.regs "pcnd" 0 == 1 then sfindLabel prog "clrDone" else st27.pc + 1))).regs rLane l =
        UInt64.ofNat l.val) ∧
    (snsteps prog 5
      (st27.setPc (if st27.regs "pcnd" 0 == 1 then sfindLabel prog "clrDone" else st27.pc + 1))).gmem =
      st.gmem
  rw [hbr]
  let st28 := st27.setPc (st27.pc + 1)
  change (snsteps prog 5 st28).pc = 25 ∧
    (snsteps prog 5 st28).regs "ci" 0 = st.regs "ci" 0 + 32 ∧
    (∀ l : Fin 32, (snsteps prog 5 st28).regs rLane l = UInt64.ofNat l.val) ∧
    (snsteps prog 5 st28).gmem = st.gmem
  simp only [snsteps]
  rw [AlgorithmLib.LZ4WarpDSL.bin_step prog st28 .shl "ca" "ci" (.imm 1)
    (by simpa [st28, st27, st26, SState.setPc, hpc, hbr] using h28)]
  let st29 := (st28.setReg "ca"
    (fun l => SOp.run .shl (st28.regs "ci" l) (st28.get l (.imm 1)))).setPc
      (st28.pc + 1)
  change (snsteps prog 4 st29).pc = 25 ∧
    (snsteps prog 4 st29).regs "ci" 0 = st.regs "ci" 0 + 32 ∧
    (∀ l : Fin 32, (snsteps prog 4 st29).regs rLane l = UInt64.ofNat l.val) ∧
    (snsteps prog 4 st29).gmem = st.gmem
  simp only [snsteps]
  rw [AlgorithmLib.LZ4WarpDSL.binr_step prog st29 .add "ca" "ca" rTbl
    (by simpa [st29, st28, st27, st26, SState.setPc, hpc, hbr] using h29)]
  let st30 := (st29.setReg "ca"
    (fun l => SOp.run .add (st29.regs "ca" l) (st29.regs rTbl l))).setPc
      (st29.pc + 1)
  change (snsteps prog 3 st30).pc = 25 ∧
    (snsteps prog 3 st30).regs "ci" 0 = st.regs "ci" 0 + 32 ∧
    (∀ l : Fin 32, (snsteps prog 3 st30).regs rLane l = UInt64.ofNat l.val) ∧
    (snsteps prog 3 st30).gmem = st.gmem
  simp only [snsteps]
  rw [AlgorithmLib.LZ4WarpDSL.stsh_step prog st30 "ca" "z"
    (by simpa [st30, st29, st28, st27, st26, SState.setPc, hpc, hbr] using h30)]
  let st31 : SState :=
    { st30 with
      smem :=
        (let s0 := storeBytes st30.smem (fun _ => true) (st30.regs "ca") (st30.regs "z")
         storeBytes s0 (fun _ => true) (fun l => st30.regs "ca" l + 1) (fun l => st30.regs "z" l >>> 8))
      pc := st30.pc + 1 }
  change (snsteps prog 2 st31).pc = 25 ∧
    (snsteps prog 2 st31).regs "ci" 0 = st.regs "ci" 0 + 32 ∧
    (∀ l : Fin 32, (snsteps prog 2 st31).regs rLane l = UInt64.ofNat l.val) ∧
    (snsteps prog 2 st31).gmem = st.gmem
  simp only [snsteps]
  rw [AlgorithmLib.LZ4WarpDSL.bin_step prog st31 .add "ci" "ci" (.imm 32)
    (by simpa [st31, st30, st29, st28, st27, st26, SState.setPc, hpc, hbr] using h31)]
  let st32 := (st31.setReg "ci"
    (fun l => SOp.run .add (st31.regs "ci" l) (st31.get l (.imm 32)))).setPc
      (st31.pc + 1)
  change (snsteps prog 1 st32).pc = 25 ∧
    (snsteps prog 1 st32).regs "ci" 0 = st.regs "ci" 0 + 32 ∧
    (∀ l : Fin 32, (snsteps prog 1 st32).regs rLane l = UInt64.ofNat l.val) ∧
    (snsteps prog 1 st32).gmem = st.gmem
  simp only [snsteps]
  rw [AlgorithmLib.LZ4WarpDSL.bra_step prog st32 "clr"
    (by simpa [st32, st31, st30, st29, st28, st27, st26, SState.setPc, hpc, hbr] using h32)]
  constructor
  · simp [SState.setPc, hfind]
  constructor
  · simp [st32, st31, st30, st29, st28, st27, st26, SState.setReg, SState.setPc,
      SState.get, SOp.run]
  constructor
  · intro l
    simpa [st32, st31, st30, st29, st28, st27, st26, SState.setReg, SState.setPc,
      SState.get, rLane] using hlane l
  · simp [st32, st31, st30, st29, st28, st27, st26, SState.setReg, SState.setPc]

theorem prologue_clr_loop_exit_slice (prog : Array SInstr) (st : SState) (entries : Nat)
    (h25 : prog[st.pc]? = some (.lbl "clr"))
    (h26 : prog[st.pc + 1]? = some (.setp .ge "pcnd" "ci" (.imm entries)))
    (h27 : prog[st.pc + 2]? = some (.braif "pcnd" "clrDone"))
    (hfind : sfindLabel prog "clrDone" = 33)
    (hpc : st.pc = 25)
    (hbrHead :
      ((if SCmp.run .ge (st.regs "ci" 0) (st.get 0 (.imm entries)) then (1 : UInt64) else 0) ==
          1) = true)
    (hlane : ∀ l : Fin 32, st.regs rLane l = UInt64.ofNat l.val) :
    (snsteps prog 3 st).pc = 33 ∧
      (∀ l : Fin 32, (snsteps prog 3 st).regs rLane l = UInt64.ofNat l.val) ∧
      (snsteps prog 3 st).gmem = st.gmem := by
  simp only [snsteps]
  rw [AlgorithmLib.LZ4WarpDSL.lbl_step prog st "clr" h25]
  let st26 := st.setPc (st.pc + 1)
  change (snsteps prog 2 st26).pc = 33 ∧
    (∀ l : Fin 32, (snsteps prog 2 st26).regs rLane l = UInt64.ofNat l.val) ∧
    (snsteps prog 2 st26).gmem = st.gmem
  simp only [snsteps]
  rw [AlgorithmLib.LZ4WarpDSL.setp_step prog st26 .ge "pcnd" "ci" (.imm entries)
    (by simpa [st26, SState.setPc, hpc] using h26)]
  let st27 := (st26.setReg "pcnd"
    (fun l => if SCmp.run .ge (st26.regs "ci" l) (st26.get l (.imm entries)) then 1 else 0)).setPc
      (st26.pc + 1)
  change (snsteps prog 1 st27).pc = 33 ∧
    (∀ l : Fin 32, (snsteps prog 1 st27).regs rLane l = UInt64.ofNat l.val) ∧
    (snsteps prog 1 st27).gmem = st.gmem
  simp only [snsteps]
  rw [AlgorithmLib.LZ4WarpDSL.braif_step prog st27 "pcnd" "clrDone"
    (by simpa [st27, st26, SState.setPc, hpc] using h27)]
  have hbr : (st27.regs "pcnd" 0 == 1) = true := by
    simpa [st27, st26, SState.setReg, SState.setPc, SState.get] using hbrHead
  constructor
  · simp [SState.setPc, hbr, hfind]
  constructor
  · intro l
    simpa [st27, st26, SState.setReg, SState.setPc, SState.get, rLane] using hlane l
  · simp [st27, st26, SState.setReg, SState.setPc]

theorem prologue_clr_exit_to_loop_slice (prog : Array SInstr) (st : SState) (entries : Nat)
    (h25 : prog[st.pc]? = some (.lbl "clr"))
    (h26 : prog[st.pc + 1]? = some (.setp .ge "pcnd" "ci" (.imm entries)))
    (h27 : prog[st.pc + 2]? = some (.braif "pcnd" "clrDone"))
    (h33 : prog[33]? = some (.lbl "clrDone"))
    (h34 : prog[34]? = some .barwarp)
    (h35 : prog[35]? = some (.mov rLitAnchor (.imm 0)))
    (h36 : prog[36]? = some (.mov rSearchPos (.imm 0)))
    (h37 : prog[37]? = some (.mov rOp (.imm 0)))
    (hfind : sfindLabel prog "clrDone" = 33)
    (hpc : st.pc = 25)
    (hbrHead :
      ((if SCmp.run .ge (st.regs "ci" 0) (st.get 0 (.imm entries)) then (1 : UInt64) else 0) ==
          1) = true)
    (hlane : ∀ l : Fin 32, st.regs rLane l = UInt64.ofNat l.val) :
    (snsteps prog 8 st).pc = 38 ∧
      (∀ l : Fin 32, (snsteps prog 8 st).regs rLane l = UInt64.ofNat l.val) ∧
      (snsteps prog 8 st).regs rSearchPos 0 = 0 ∧
      (snsteps prog 8 st).regs rLitAnchor 0 = 0 ∧
      (snsteps prog 8 st).regs rOp 0 = 0 ∧
      (snsteps prog 8 st).gmem = st.gmem := by
  rw [show 8 = 3 + 5 by omega, snsteps_add]
  have hexit := prologue_clr_loop_exit_slice prog st entries h25 h26 h27 hfind hpc hbrHead hlane
  have hdone := prologue_clrDone_to_loop_slice prog (snsteps prog 3 st)
    (by rw [hexit.1]; exact h33)
    (by rw [hexit.1]; exact h34)
    (by rw [hexit.1]; exact h35)
    (by rw [hexit.1]; exact h36)
    (by rw [hexit.1]; exact h37)
    hexit.1 hexit.2.1
  constructor
  · exact hdone.1
  constructor
  · exact hdone.2.1
  constructor
  · exact hdone.2.2.1
  constructor
  · exact hdone.2.2.2.1
  constructor
  · exact hdone.2.2.2.2.1
  · rw [hdone.2.2.2.2.2, hexit.2.2]

theorem prologue_clear_body_iter (prog : Array SInstr) (st : SState) (hashLog entries : Nat)
    (h25 : prog[25]? = some (.lbl "clr"))
    (h26 : prog[26]? = some (.setp .ge "pcnd" "ci" (.imm entries)))
    (h27 : prog[27]? = some (.braif "pcnd" "clrDone"))
    (h28 : prog[28]? = some (.bin .shl "ca" "ci" (.imm 1)))
    (h29 : prog[29]? = some (.binr .add "ca" "ca" rTbl))
    (h30 : prog[30]? = some (.stsh "ca" "z"))
    (h31 : prog[31]? = some (.bin .add "ci" "ci" (.imm 32)))
    (h32 : prog[32]? = some (.bra "clr"))
    (hclr : sfindLabel prog "clr" = 25)
    (hEntries : entries = 2 ^ hashLog)
    (hHash : hashLog ≤ 32)
    (hstpc : st.pc = 25)
    (hstci : st.regs "ci" 0 = 0)
    (hlane : ∀ l : Fin 32, st.regs rLane l = UInt64.ofNat l.val)
    (k : Nat) (hk : k ≤ clearIters hashLog) :
    let st' := snsteps prog (8 * k) st
    st'.pc = 25 ∧
      st'.regs "ci" 0 = UInt64.ofNat (32 * k) ∧
      (∀ l : Fin 32, st'.regs rLane l = UInt64.ofNat l.val) ∧
      st'.gmem = st.gmem := by
  induction k with
  | zero =>
      change st.pc = 25 ∧ st.regs "ci" 0 = UInt64.ofNat (32 * 0) ∧
        (∀ l : Fin 32, st.regs rLane l = UInt64.ofNat l.val) ∧ st.gmem = st.gmem
      exact ⟨hstpc, by simpa using hstci, hlane, rfl⟩
  | succ k ih =>
      have hkprev : k ≤ clearIters hashLog := by omega
      have ihv := ih hkprev
      dsimp only at ihv ⊢
      rw [show 8 * (k + 1) = 8 * k + 8 by omega, snsteps_add]
      let sk := snsteps prog (8 * k) st
      have hklt : k < clearIters hashLog := by omega
      have hbr :
          ((if SCmp.run .ge (sk.regs "ci" 0) (sk.get 0 (.imm (2 ^ hashLog))) then
              (1 : UInt64)
            else
              0) == 1) = false := by
        have hci : sk.regs "ci" 0 = UInt64.ofNat (32 * k) := ihv.2.1
        rw [hci]
        change ((if SCmp.run .ge (UInt64.ofNat (32 * k)) (UInt64.ofNat (2 ^ hashLog)) then
            (1 : UInt64) else 0) == 1) = false
        exact clearLoop_body_branch_false hashLog k hHash hklt
      have hbody := prologue_clr_loop_body_slice prog sk entries
        (by rw [ihv.1]; exact h25)
        (by rw [ihv.1]; exact h26)
        (by rw [ihv.1]; exact h27)
        (by rw [ihv.1]; exact h28)
        (by rw [ihv.1]; exact h29)
        (by rw [ihv.1]; exact h30)
        (by rw [ihv.1]; exact h31)
        (by rw [ihv.1]; exact h32)
        hclr ihv.1 (by simpa [hEntries] using hbr) ihv.2.2.1
      constructor
      · exact hbody.1
      constructor
      · rw [hbody.2.1, ihv.2.1]
        have hsum : 32 * k + 32 < 2 ^ 64 := by
          have hle : k + 1 ≤ clearIters hashLog := by omega
          have hb := clearIters_bytes_lt_u64 hashLog hHash
          have hmul : 32 * (k + 1) ≤ 32 * clearIters hashLog :=
            Nat.mul_le_mul_left 32 hle
          have : 32 * (k + 1) < 2 ^ 64 := Nat.lt_of_le_of_lt hmul hb
          rwa [Nat.mul_add, Nat.mul_one] at this
        apply UInt64.toNat_inj.mp
        rw [UInt64.toNat_add,
          AlgorithmLib.LZ4Ptx.toNat_ofNat_lt (32 * k) (by omega)]
        rw [show UInt64.toNat (32 : UInt64) = 32 by decide]
        rw [AlgorithmLib.LZ4Ptx.toNat_ofNat_lt (32 * (k + 1))
          (by rwa [Nat.mul_add, Nat.mul_one])]
        rw [Nat.mod_eq_of_lt hsum]
        rw [Nat.mul_add, Nat.mul_one]
      constructor
      · exact hbody.2.2.1
      · rw [hbody.2.2.2, ihv.2.2.2]

end AlgorithmLib.LZ4Simt
