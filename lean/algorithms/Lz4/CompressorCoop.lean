module
public import Lz4.CompressorWarp
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lz4.CompressorCoop` — the warp's run of the cooperative statements

The warp's run of each cooperative statement's translation agrees with the
model's (`sim_extend`, `sim_copy`, `sim_window`), so the translation of every
statement does (`sim`).
-/

namespace Lz4Code

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Prog
  AlgorithmLib.LZ4WarpDSL AlgorithmLib.LibSpec AlgorithmLib.Kernel AlgorithmLib.ML
  AlgorithmLib.ML.BodyLane AlgorithmLib.ML.LaneWarp AlgorithmLib.ML.LaneSettle AlgorithmLib.ML.LaneShared
  AlgorithmLib.ML.LaneLaws

/-- The warp's ballot is the mask of its lanes' flags. -/
theorem ballot_maskN (p : Lane → UInt64) : ISt.ballot p = (UInt32.ofNat (maskN (flagN p))).toUInt64 := by
  apply UInt64.toNat_inj.mp
  rw [ISt.ballot, fold_u64, finRange_val, UInt32.toNat_toUInt64, UInt32.toNat_ofNat_of_lt' (maskN_lt _)]
  rfl

/-- The model's byte at `a`, as a word. -/
abbrev byteW (W : WState) (a : UInt64) : UInt64 := (W.gmem.getD a.toNat 0).toUInt64

/-- **Lane `l`'s extend flag**: its byte of the match, clamped below the end
    cap, and the candidate's are input bytes, and the flag (below the end cap
    and the bytes equal) is the model's `good` at `ml + l`. -/
theorem ext_flag {I : UInt64 → Prop} {W : WState} {p0 cand0 ml : String} {inStride endCap l : Nat} (hl : l < 32)
    (hecR : W.regs "ecR" = UInt64.ofNat endCap) (hec1 : W.regs "ec1" = UInt64.ofNat (endCap - 1))
    (hend : 1 ≤ endCap) (hendI : endCap ≤ inStride)
    (hcp : (W.regs cand0).toNat < (W.regs p0).toNat)
    (hov : (W.regs p0).toNat + (W.regs ml).toNat + 32 < 2 ^ 64)
    (hin : ∀ j, j < inStride → I (W.regs "inBase" + UInt64.ofNat j))
    (hib : (W.regs "inBase").toNat + inStride < 2 ^ 64) (hpe : (W.regs p0).toNat < endCap) :
    let pe := W.regs p0 + (W.regs ml + UInt64.ofNat l)
    let peC := if pe ≤ W.regs "ec1" then pe else W.regs "ec1"
    let caC := W.regs cand0 + (peC - W.regs p0)
    I (W.regs "inBase" + peC) ∧ I (W.regs "inBase" + caC) ∧
      (decide (pe < W.regs "ecR") && (byteW W (W.regs "inBase" + peC) == byteW W (W.regs "inBase" + caC))) =
        extGood W p0 cand0 ml inStride endCap l := by
  intro pe peC caC
  have hpeN : pe.toNat = (W.regs p0).toNat + (W.regs ml).toNat + l := by
    simp only [pe]
    rw [UInt64.toNat_add, UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show l < 2 ^ 64 by omega),
      Nat.mod_eq_of_lt (show (W.regs ml).toNat + l < 2 ^ 64 by omega), Nat.mod_eq_of_lt (by omega)]
    omega
  have he1 : (W.regs "ec1").toNat = endCap - 1 := by
    rw [hec1, UInt64.toNat_ofNat_of_lt' (show endCap - 1 < 2 ^ 64 by omega)]
  have hpc1 : peC.toNat < endCap ∧ (W.regs p0).toNat ≤ peC.toNat := by
    simp only [peC]
    split
    · rename_i hle; rw [UInt64.le_iff_toNat_le, hpeN, he1] at hle; rw [hpeN]; omega
    · rw [he1]; omega
  have hcaN : caC.toNat = (W.regs cand0).toNat + (peC.toNat - (W.regs p0).toNat) := by
    simp only [caC]
    rw [UInt64.toNat_add, UInt64.toNat_sub_of_le _ _ (by rw [UInt64.le_iff_toNat_le]; omega),
      Nat.mod_eq_of_lt (by omega)]
  have hta : ∀ x : UInt64, x.toNat < inStride → W.regs "inBase" + x = W.regs "inBase" + UInt64.ofNat x.toNat := by
    intro x _; rw [UInt64.ofNat_toNat]
  have htn : ∀ x, x < inStride → (W.regs "inBase" + UInt64.ofNat x).toNat = (W.regs "inBase").toNat + x := by
    intro x hx
    rw [UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show x < 2 ^ 64 by omega), Nat.mod_eq_of_lt (by omega)]
  refine ⟨by rw [hta _ (by omega)]; exact hin _ (by omega), by rw [hta _ (by omega)]; exact hin _ (by omega), ?_⟩
  have hE64 : endCap < 2 ^ 64 := by omega
  rw [hecR]
  simp only [extGood, AlgorithmLib.LZ4WarpSched.good]
  by_cases hlt : (W.regs p0).toNat + (W.regs ml).toNat + l < endCap
  · have hd1 : decide (pe < UInt64.ofNat endCap) = true := by
      simp [UInt64.lt_iff_toNat_lt, hpeN, UInt64.toNat_ofNat_of_lt' hE64, hlt]
    have hd2 : decide ((W.regs p0).toNat + ((W.regs ml).toNat + l) < endCap) = true := by
      simp; omega
    rw [hd1, hd2, Bool.true_and, Bool.true_and]
    have hle : pe ≤ W.regs "ec1" := by
      rw [UInt64.le_iff_toNat_le, hpeN, he1]; omega
    have hpc : peC = pe := by simp only [peC, if_pos hle]
    have hcaC : caC.toNat = (W.regs cand0).toNat + (W.regs ml).toNat + l := by
      rw [hcaN, hpc, hpeN]; omega
    rw [hta peC (by omega), hta caC (by omega)]
    simp only [byteW]
    rw [htn _ (by omega), htn _ (by omega), byte_beq, hcaC, hpc, hpeN]
    simp only [AlgorithmLib.LZ4WarpFind.byte]
    rw [getD_inp (by omega), getD_inp (by omega)]
    simp only [Nat.add_assoc]
  · have hd1 : decide (pe < UInt64.ofNat endCap) = false := by
      simp [UInt64.lt_iff_toNat_lt, hpeN, UInt64.toNat_ofNat_of_lt' hE64]; omega
    have hd2 : decide ((W.regs p0).toNat + ((W.regs ml).toNat + l) < endCap) = false := by
      simp; omega
    rw [hd1, hd2]; rfl

theorem maskN_congr {h h' : Nat → Bool} (e : ∀ k, k < 32 → h k = h' k) : maskN h = maskN h' := by
  have : ∀ (L : List Nat) (acc : Nat), (∀ k ∈ L, k < 32) →
      L.foldl (fun acc l => if h l then acc ||| 2 ^ l else acc) acc =
        L.foldl (fun acc l => if h' l then acc ||| 2 ^ l else acc) acc := by
    intro L
    induction L with
    | nil => intro _ _; rfl
    | cons k L ih =>
      intro acc hL
      rw [List.foldl_cons, List.foldl_cons, e k (hL k (by simp))]
      exact ih _ (fun x hx => hL x (by simp [hx]))
  exact this _ 0 (fun k hk => List.mem_range.mp hk)

theorem set_regs_ne (st : ISt) {d r : Nat} (F : Lane → UInt64) (h : r ≠ d) : (st.set d F).regs r = st.regs r := by
  simp [ISt.set, h]

theorem set_gmem (st : ISt) (d : Nat) (F : Lane → UInt64) : (st.set d F).gmem = st.gmem := rfl
theorem set_smem (st : ISt) (d : Nat) (F : Lane → UInt64) : (st.set d F).smem = st.smem := rfl

theorem flag_and (x : Prop) [Decidable x] (y : Bool) :
    ((if x then (1 : UInt64) else 0) &&& (if y = true then 1 else 0) != 0) = (decide x && y) := by
  by_cases hx : x <;> cases y <;> simp [hx]

theorem set_regs_eq (st : ISt) (d : Nat) (F : Lane → UInt64) : (st.set d F).regs d = F := by
  simp [ISt.set]

@[simp] theorem withInt_int (ws : WSt F32) (X : ISt) : (ws.withInt X).int = X := rfl
@[simp] theorem withInt_withInt (ws : WSt F32) (X Y : ISt) : (ws.withInt X).withInt Y = ws.withInt Y := rfl

/-- A statement that neither stores nor synchronises. -/
def quiet : EWStmt → Bool
  | .setI .. | .ldB .. | .ldH .. | .ballot .. | .shflIdx .. | .skip => true
  | .seq a b => quiet a && quiet b
  | .locals _ b => quiet b
  | _ => false

theorem quiet_seqs : ∀ (L : List EWStmt), (∀ s ∈ L, quiet s = true) → quiet (seqs L) = true
  | [], _ => rfl
  | [s], h => h s (by simp)
  | s :: t :: L, h => by
      show (quiet s && quiet (seqs (t :: L))) = true
      rw [h s (by simp), quiet_seqs (t :: L) (fun x hx => h x (by simp [hx]))]; rfl

/-- **A quiet statement's run stores nothing and leaves shared memory alone.** -/
theorem quiet_trace (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat) (l : Lane) (ptr : Buf → UInt64) (sm : UInt64) :
    ∀ (s : EWStmt), quiet s = true → ∀ i ws,
      LaneWarp.trace hw cta im (storeLeaf cta im l ptr sm) i s ws = [] ∧ LaneWarp.trace hw cta im shLeaf i s ws = [] ∧
      LaneWarp.trace hw cta im shEvLeaf i s ws = [] := by
  intro s
  induction s with
  | seq a b iha ihb =>
    intro h i ws
    simp only [quiet, Bool.and_eq_true] at h
    obtain ⟨a1, a2, a3⟩ := iha h.1 i ws
    obtain ⟨b1, b2, b3⟩ := ihb h.2 i (LaneWarp.run hw cta im a ws i)
    refine ⟨?_, ?_, ?_⟩ <;> simp only [LaneWarp.trace, a1, a2, a3, b1, b2, b3, List.append_nil]
  | locals rs b ih => intro h i ws; exact ih h i ws
  | setI | ldB | ldH | ballot | shflIdx | skip => intro _ i ws; exact ⟨rfl, rfl, rfl⟩
  | _ => intro h; simp [quiet] at h

/-- A statement with no shared store and no barrier. -/
def noSh : EWStmt → Bool
  | .setI .. | .ldB .. | .ldH .. | .ballot .. | .shflIdx .. | .skip | .stB .. => true
  | .seq a b => noSh a && noSh b
  | .locals _ b => noSh b
  | .iwhile _ _ _ b => noSh b
  | _ => false

/-- **A statement with no shared store and no barrier leaves shared memory's
    traces empty.** -/
theorem noSh_trace (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat) :
    ∀ (s : EWStmt), noSh s = true → ∀ i ws,
      LaneWarp.trace hw cta im shLeaf i s ws = [] ∧ LaneWarp.trace hw cta im shEvLeaf i s ws = [] := by
  intro s
  induction s with
  | seq a b iha ihb =>
    intro h i ws
    simp only [noSh, Bool.and_eq_true] at h
    obtain ⟨a1, a2⟩ := iha h.1 i ws
    obtain ⟨b1, b2⟩ := ihb h.2 i (LaneWarp.run hw cta im a ws i)
    refine ⟨?_, ?_⟩ <;> simp only [LaneWarp.trace, a1, a2, b1, b2, List.append_nil]
  | locals rs b ih => intro h i ws; exact ih h i ws
  | iwhile n c cs b ih =>
    intro h i
    have h' : noSh b = true := h
    clear h
    induction n with
    | zero => intro ws; exact ⟨rfl, rfl⟩
    | succ n ihn =>
      intro ws
      by_cases hc : c.eval ws.int cta lane0 ≠ 0
      · obtain ⟨b1, b2⟩ := ih h' i ws
        obtain ⟨r1, r2⟩ := ihn (wtrip hw cta im c cs b ws i ws)
        rw [trace_iwhile_peel hw cta im shLeaf n c cs b ws i hc, trace_iwhile_peel hw cta im shEvLeaf n c cs b ws i hc,
          b1, b2, r1, r2]
        exact ⟨rfl, rfl⟩
      · exact ⟨trace_iwhile_stop hw cta im shLeaf _ c cs b ws i hc, trace_iwhile_stop hw cta im shEvLeaf _ c cs b ws i hc⟩
  | setI | ldB | ldH | ballot | shflIdx | skip | stB => intro _ i ws; exact ⟨rfl, rfl⟩
  | _ => intro h; simp [noSh] at h

theorem ofNat_add_ofNat (a b : Nat) : UInt64.ofNat a + UInt64.ofNat b = UInt64.ofNat (a + b) := by
  rw [← UInt64.ofNat_add]

/-- The warp agrees with the model after a statement with local registers
    `rs`, from what the statement's run leaves. -/
theorem simW_restore {ns : List String} {I : UInt64 → Prop} {ws Y : WSt F32} {W : WState} {rs : List Nat}
    (hreg : ∀ n ∈ ns, ∀ l, (ISt.restore rs ws.int Y.int).regs (ns.idxOf n) l = W.regs n)
    (hlane : ∀ l : Lane, (ISt.restore rs ws.int Y.int).regs (laneR ns.length) l = UInt64.ofNat l.val)
    (hgin : ∀ a, I a → Y.int.gmem a.toNat = W.gmem.getD a.toNat 0)
    (hsm : ∀ k, k < W.smem.size → Y.int.smem k = W.smem.getD k 0) :
    SimW ns I (Y.withInt (ISt.restore rs ws.int Y.int)) W := ⟨hreg, hlane, hgin, hsm⟩

theorem run_locals (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat) (rs : List Nat) (s : EWStmt) (ws : WSt F32) (i : Nat) :
    LaneWarp.run hw cta im (.locals rs s) ws i = (LaneWarp.run hw cta im s ws i).withInt
      (ISt.restore rs ws.int (LaneWarp.run hw cta im s ws i).int) := rfl

theorem run_seq (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat) (a b : EWStmt) (ws : WSt F32) (i : Nat) :
    LaneWarp.run hw cta im (.seq a b) ws i = LaneWarp.run hw cta im b (LaneWarp.run hw cta im a ws i) i := rfl

theorem trace_locals {α : Type} (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat) (leaf : Nat → EWStmt → WSt F32 → List α)
    (rs : List Nat) (s : EWStmt) (ws : WSt F32) (i : Nat) :
    LaneWarp.trace hw cta im leaf i (.locals rs s) ws = LaneWarp.trace hw cta im leaf i s ws := rfl

theorem trace_seq {α : Type} (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat) (leaf : Nat → EWStmt → WSt F32 → List α)
    (a b : EWStmt) (ws : WSt F32) (i : Nat) :
    LaneWarp.trace hw cta im leaf i (.seq a b) ws =
      LaneWarp.trace hw cta im leaf i a ws ++ LaneWarp.trace hw cta im leaf i b (LaneWarp.run hw cta im a ws i) := rfl

/-- A statement that sets only registers. -/
def regOnly : EWStmt → Bool
  | .setI .. | .ldB .. | .ldH .. | .ballot .. | .shflIdx .. | .skip | .barrier => true
  | .seq a b => regOnly a && regOnly b
  | _ => false

/-- **A statement that sets only registers leaves memory and shared memory.** -/
theorem regOnly_mem (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat) :
    ∀ (s : EWStmt), regOnly s = true → ∀ (X : WSt F32) i,
      (LaneWarp.run hw cta im s X i).int.gmem = X.int.gmem ∧ (LaneWarp.run hw cta im s X i).int.smem = X.int.smem := by
  intro s
  induction s with
  | seq a b iha ihb =>
    intro h X i
    simp only [regOnly, Bool.and_eq_true] at h
    obtain ⟨a1, a2⟩ := iha h.1 X i
    obtain ⟨b1, b2⟩ := ihb h.2 (LaneWarp.run hw cta im a X i) i
    rw [run_seq]
    exact ⟨b1.trans a1, b2.trans a2⟩
  | setI | ldB | ldH | ballot | shflIdx | skip | barrier => intro _ X i; exact ⟨rfl, rfl⟩
  | _ => intro h; simp [regOnly] at h

theorem half_toNat (b0 b1 : UInt8) : (b0.toUInt64 ||| b1.toUInt64 <<< 8).toNat = b0.toNat + 256 * b1.toNat := by
  have l0 : b0.toUInt64.toNat < 256 := by simpa using b0.toNat_lt
  have l1 : b1.toUInt64.toNat < 256 := by simpa using b1.toNat_lt
  rw [UInt64.toNat_or, show (8 : UInt64) = UInt64.ofNat 8 from rfl, shl_toNat (by decide) (by omega),
    Nat.shiftLeft_eq, show b1.toUInt64.toNat * 2 ^ 8 = b1.toUInt64.toNat <<< 8 from (Nat.shiftLeft_eq _ _).symm,
    or_shl_add (by simpa using l0)]
  simp; omega

theorem clamp_le {x : UInt64} {n : Nat} (h4 : 4 ≤ n) (hn : n < 2 ^ 64) :
    (if x ≤ c64 (n - 4) then x else c64 (n - 4)).toNat ≤ n - 4 := by
  rw [c64_sub4 h4]
  split
  · rename_i hle; rwa [toNat_le_ofNat (Nat.lt_of_le_of_lt (Nat.sub_le n 4) hn)] at hle
  · rw [UInt64.toNat_ofNat_of_lt' (Nat.lt_of_le_of_lt (Nat.sub_le n 4) hn)]; exact Nat.le_refl _

theorem flag4P (p r : Prop) [Decidable p] [Decidable r] (b d : Bool) :
    (((if p then (1 : UInt64) else 0) &&& (if b = true then 1 else 0)) &&&
      ((if r then 1 else 0) &&& (if d = true then 1 else 0)) != 0) = ((decide p && b) && (decide r && d)) := by
  by_cases hp : p <;> by_cases hr : r <;> cases b <;> cases d <;> simp [hp, hr]

theorem mask32 (x : UInt32) : x.toUInt64 &&& 0xffffffff = x.toUInt64 := by
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_and, show (0xffffffff : UInt64).toNat = 2 ^ 32 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod,
    UInt32.toNat_toUInt64, Nat.mod_eq_of_lt x.toNat_lt]

theorem lo8 {n : Nat} (hn : n < 2 ^ 64) : (UInt64.ofNat n).toUInt8 = UInt8.ofNat (n % 256) := by
  apply UInt8.toNat_inj.mp
  simp only [UInt64.toNat_toUInt8, UInt64.toNat_ofNat_of_lt' hn, UInt8.toNat_ofNat']
  omega

theorem hi8 {n : Nat} (hn : n < 2 ^ 64) : ((UInt64.ofNat n) >>> 8).toUInt8 = UInt8.ofNat (n / 256 % 256) := by
  apply UInt8.toNat_inj.mp
  simp only [UInt64.toNat_toUInt8, UInt64.toNat_shiftRight, UInt64.toNat_ofNat_of_lt' hn, UInt8.toNat_ofNat']
  rw [show (8 : UInt64).toNat % 64 = 8 from rfl, Nat.shiftRight_eq_div_pow]
  have : n / 2 ^ 8 < 2 ^ 64 := Nat.lt_of_le_of_lt (Nat.div_le_self _ _) hn
  omega

/-- A statement with no byte store. -/
def noStB : EWStmt → Bool
  | .stB .. => false
  | .seq a b => noStB a && noStB b
  | .locals _ b => noStB b
  | .iif .. | .forN .. | .forM .. | .iwhile .. => false
  | _ => true

/-- **A statement with no byte store makes no one-byte store.** -/
theorem noStB_trace (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat) (l : Lane) (ptr : Buf → UInt64) (sm : UInt64) :
    ∀ (s : EWStmt), noStB s = true → ∀ i ws,
      byteSt (LaneWarp.trace hw cta im (storeLeaf cta im l ptr sm) i s ws) = [] := by
  intro s
  induction s with
  | seq a b iha ihb =>
    intro h i ws
    simp only [noStB, Bool.and_eq_true] at h
    rw [trace_seq, byteSt_append, iha h.1, ihb h.2]; rfl
  | locals rs b ih => intro h i ws; exact ih h i ws
  | stH a v g =>
    intro _ i ws
    show byteSt (if g.eval ws.int cta l ≠ 0 then [some (sm + a.eval ws.int cta l, 2, v.eval ws.int cta l &&& 0xffff)]
      else []) = []
    split <;> rfl
  | storeLane0 bu ix r =>
    intro _ i ws
    show byteSt (if l = lane0 then [some (ptr bu + off (UInt32.ofNat (ix.eval cta i l (fun _ _ => 0) im)), 4,
      (ws.regs r l).bits)] else []) = []
    split <;> rfl
  | stB => intro h; simp [noStB] at h
  | iif | forN | forM | iwhile => intro h; simp [noStB] at h
  | _ => intro _ i ws; rfl

/-- Four bytes at `A`, little-endian, as a warp's lane loads them. -/
def w4 (st : ISt) (A : UInt64) : UInt64 :=
  (st.byte A ||| st.byte (A + UInt64.ofNat 1) <<< UInt64.ofNat 8) |||
    (st.byte (A + UInt64.ofNat 2) <<< UInt64.ofNat 16 ||| st.byte (A + UInt64.ofNat 3) <<< UInt64.ofNat 24)

theorem w4_congr {st st' : ISt} (h : st.gmem = st'.gmem) (A : UInt64) : w4 st A = w4 st' A := by
  simp only [w4, ISt.byte, h]

/-- **Four input bytes are the model's word.** -/
theorem w4_word {ns : List String} {I : UInt64 → Prop} {ws : WSt F32} {W : WState} (hS : SimW ns I ws W) {A : UInt64}
    (hI : ∀ k, k < 4 → I (A + UInt64.ofNat k)) (hA : A.toNat + 4 ≤ 2 ^ 64) :
    (w4 ws.int A).toNat = wLoad4 W.gmem A.toNat := by
  have hb : ∀ k, k < 4 → ws.int.byte (A + UInt64.ofNat k) = (W.gmem.getD (A.toNat + k) 0).toUInt64 := by
    intro k hk
    have hn : (A + UInt64.ofNat k).toNat = A.toNat + k := by
      rw [UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show k < 2 ^ 64 by omega)]
      exact Nat.mod_eq_of_lt (by omega)
    simp only [ISt.byte]
    rw [hS.gin _ (hI k hk), hn]
  have h0 : ws.int.byte A = (W.gmem.getD (A.toNat + 0) 0).toUInt64 := by simpa using hb 0 (by decide)
  simp only [w4]
  rw [h0, hb 1 (by decide), hb 2 (by decide), hb 3 (by decide)]
  have hb8 : ∀ b : UInt8, b.toUInt64.toNat < 256 := fun b => by simpa using b.toNat_lt
  have l0 := hb8 (W.gmem.getD (A.toNat + 0) 0); have l1 := hb8 (W.gmem.getD (A.toNat + 1) 0)
  have l2 := hb8 (W.gmem.getD (A.toNat + 2) 0); have l3 := hb8 (W.gmem.getD (A.toNat + 3) 0)
  rw [UInt64.toNat_or, UInt64.toNat_or, UInt64.toNat_or, shl_toNat (by decide) (by omega),
    shl_toNat (by decide) (by omega), shl_toNat (by decide) (by omega), bytes4 l0 l1 l2]
  simp [wLoad4]

theorem litI_evalW (k : Int) (st : ISt) (cta : Nat) (l : Lane) : (litI k).eval st cta l = c64 k := rfl

/-- **The warp's four-byte load**: `d` the four bytes at `a`, the temporaries
    `t … t + 4` changed, every other register, memory and shared memory kept. -/
theorem load4_run (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat) (d t : Nat) (a : IExp) (X : WSt F32) (i : Nat) :
    (∀ l, (LaneWarp.run hw cta im (seqs (load4E d t a)) X i).int.regs d l = w4 X.int (a.eval X.int cta l)) ∧
    (∀ r, r ≠ d → r < t ∨ t + 4 < r → (LaneWarp.run hw cta im (seqs (load4E d t a)) X i).int.regs r = X.int.regs r) ∧
    (LaneWarp.run hw cta im (seqs (load4E d t a)) X i).int.gmem = X.int.gmem ∧
    (LaneWarp.run hw cta im (seqs (load4E d t a)) X i).int.smem = X.int.smem := by
  simp only [load4E, seqs, LaneWarp.run, EWStmt.run, withInt_int, withInt_withInt]
  refine ⟨fun l => ?_, fun r h1 h2 => ?_, rfl, rfl⟩
  · simp (disch := omega) only [set_regs_eq, set_regs_ne, IExp.eval, IOp.run, ISt.byte, set_gmem, w4]
  · simp (disch := omega) only [set_regs_ne]

theorem load4_ok (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat) (S : Nat) (ptr : Buf → UInt64)
    (MF : Lane → Buf → Nat → Prop) (MB : Lane → UInt64 → Prop) (MI : Buf → Nat → Prop)
    (WS : Lane → UInt64 → Nat → Prop) (d t : Nat) (a : IExp) (X : WSt F32) (i : Nat) (cl : Bool)
    (h : ∀ l k, k < 4 → MB l (a.eval X.int cta l + UInt64.ofNat k)) :
    OkW hw cta im S ptr MF MB MI WS i (seqs (load4E d t a)) X cl := by
  have h0 : ∀ l, MB l (a.eval X.int cta l) := fun l => by simpa using h l 0 (by decide)
  refine ⟨trivial, fun l => ?_, fun l => ?_, trivial, fun l => ?_, trivial, fun l => ?_, trivial, trivial⟩
  · simp only [LaneWarp.run, EWStmt.run, withInt_int, set_regs_eq, IExp.eval]; exact h0 l
  · simp (disch := omega) only [LaneWarp.run, EWStmt.run, withInt_int, set_regs_eq, set_regs_ne, IExp.eval, IOp.run]
    exact h l 1 (by decide)
  · simp (disch := omega) only [LaneWarp.run, EWStmt.run, withInt_int, set_regs_eq, set_regs_ne, IExp.eval, IOp.run]
    exact h l 2 (by decide)
  · simp (disch := omega) only [LaneWarp.run, EWStmt.run, withInt_int, set_regs_eq, set_regs_ne, IExp.eval, IOp.run]
    exact h l 3 (by decide)

theorem seqs_cons (a : EWStmt) {L : List EWStmt} (h : L ≠ []) : seqs (a :: L) = .seq a (seqs L) := by
  cases L with
  | nil => exact absurd rfl h
  | cons b L => rfl

section SeqsAppend
variable (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat)

theorem run_seqs_append (B : List EWStmt) (hB : B ≠ []) : ∀ (A : List EWStmt), A ≠ [] → ∀ ws i,
    LaneWarp.run hw cta im (seqs (A ++ B)) ws i = LaneWarp.run hw cta im (seqs B) (LaneWarp.run hw cta im (seqs A) ws i) i
  | [], h, _, _ => absurd rfl h
  | [a], _, ws, i => by rw [List.singleton_append, seqs_cons a hB]; rfl
  | a :: a' :: A, _, ws, i => by
      rw [List.cons_append, seqs_cons a (by simp), seqs_cons a (by simp)]
      exact run_seqs_append B hB (a' :: A) (by simp) _ i

theorem trace_seqs_append {α : Type} (leaf : Nat → EWStmt → WSt F32 → List α) (B : List EWStmt) (hB : B ≠ []) :
    ∀ (A : List EWStmt), A ≠ [] → ∀ ws i,
    LaneWarp.trace hw cta im leaf i (seqs (A ++ B)) ws =
      LaneWarp.trace hw cta im leaf i (seqs A) ws ++
        LaneWarp.trace hw cta im leaf i (seqs B) (LaneWarp.run hw cta im (seqs A) ws i)
  | [], h, _, _ => absurd rfl h
  | [a], _, ws, i => by rw [List.singleton_append, seqs_cons a hB]; rfl
  | a :: a' :: A, _, ws, i => by
      rw [List.cons_append, seqs_cons a (by simp), seqs_cons a (by simp)]
      show LaneWarp.trace hw cta im leaf i a ws ++ LaneWarp.trace hw cta im leaf i (seqs ((a' :: A) ++ B))
        (LaneWarp.run hw cta im a ws i) = (LaneWarp.trace hw cta im leaf i a ws ++
          LaneWarp.trace hw cta im leaf i (seqs (a' :: A)) (LaneWarp.run hw cta im a ws i)) ++ _
      rw [trace_seqs_append leaf B hB (a' :: A) (by simp), List.append_assoc]
      rfl

theorem okW_seqs_append (S : Nat) (ptr : Buf → UInt64) (MF : Lane → Buf → Nat → Prop) (MB : Lane → UInt64 → Prop)
    (MI : Buf → Nat → Prop) (WS : Lane → UInt64 → Nat → Prop) (B : List EWStmt) (hB : B ≠ []) :
    ∀ (A : List EWStmt), A ≠ [] → ∀ ws i cl, OkW hw cta im S ptr MF MB MI WS i (seqs A) ws cl →
      OkW hw cta im S ptr MF MB MI WS i (seqs B) (LaneWarp.run hw cta im (seqs A) ws i)
        (lastOr cl (LaneWarp.trace hw cta im shLeaf i (seqs A) ws)) →
      OkW hw cta im S ptr MF MB MI WS i (seqs (A ++ B)) ws cl
  | [], h, _, _, _, _, _ => absurd rfl h
  | [a], _, ws, i, cl, h1, h2 => by rw [List.singleton_append, seqs_cons a hB]; exact ⟨h1, h2⟩
  | a :: a' :: A, _, ws, i, cl, h1, h2 => by
      rw [List.cons_append, seqs_cons a (by simp)]
      rw [seqs_cons a (by simp)] at h1 h2
      obtain ⟨o1, o2⟩ := h1
      refine ⟨o1, okW_seqs_append S ptr MF MB MI WS B hB (a' :: A) (by simp) _ i _ o2 ?_⟩
      have e : LaneWarp.trace hw cta im shLeaf i (.seq a (seqs (a' :: A))) ws =
          LaneWarp.trace hw cta im shLeaf i a ws ++
            LaneWarp.trace hw cta im shLeaf i (seqs (a' :: A)) (LaneWarp.run hw cta im a ws i) := rfl
      rw [e, lastOr_append] at h2
      exact h2

end SeqsAppend

/-- **A lane's probe is the model's**, from the values it loads. -/
theorem probe_vals {ws : WState}
    {inStride searchLim hashLog tbl l : Nat} {sp : String} (hib : (ws.regs "inBase").toNat + inStride < 2 ^ 64)
    (hsl : searchLim + 4 ≤ inStride) (hh : hashLog ≤ 32)
    (hs : (ws.regs sp).toNat + l < 2 ^ 64)
    (htbl : ws.regs "tbl" = UInt64.ofNat tbl) (htb : tbl + 2 * 2 ^ hashLog + 2 < 2 ^ 36)
    {posP rp v32 addr candRaw cand rc vc : UInt64}
    (hpos : posP = ws.regs sp + UInt64.ofNat l)
    (hrp : rp = if posP ≤ c64 (inStride - 4) then posP else c64 (inStride - 4))
    (hv32 : v32.toNat = wLoad4 ws.gmem ((ws.regs "inBase").toNat + rp.toNat))
    (haddr : addr = ((((v32 * c64 wHashK) >>> c64 (32 - hashLog)) &&& c64 (2 ^ hashLog - 1)) <<< c64 1) +
      ws.regs "tbl")
    (hcr : addr.toNat + 2 ≤ tbl + 2 * 2 ^ hashLog →
      candRaw.toNat = (ws.smem.getD addr.toNat 0).toNat + 256 * (ws.smem.getD (addr.toNat + 1) 0).toNat)
    (hcand : cand = candRaw + c64 (-1))
    (hrc : rc = if cand ≤ c64 (inStride - 4) then cand else c64 (inStride - 4))
    (hvc : vc.toNat = wLoad4 ws.gmem ((ws.regs "inBase").toNat + rc.toNat)) :
    ((decide (posP < c64 searchLim) && candRaw != c64 0) && (decide (cand < posP) && vc == v32)) =
        (AlgorithmLib.LZ4WarpFind.probe (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
          (tableOracle ws.gmem ws.smem hashLog tbl (ws.regs "inBase").toNat) searchLim
          ((ws.regs sp).toNat + l)).isSome ∧
      (∀ c, AlgorithmLib.LZ4WarpFind.probe (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
          (tableOracle ws.gmem ws.smem hashLog tbl (ws.regs "inBase").toNat) searchLim
          ((ws.regs sp).toNat + l) = some c → cand = UInt64.ofNat c) ∧
      addr.toNat = tbl + 2 * wHash ws.gmem hashLog ((ws.regs "inBase").toNat +
        min ((ws.regs sp).toNat + l) (inStride - 4)) ∧
      cand &&& 0xffffffff = (tableOracle ws.gmem ws.smem hashLog tbl (ws.regs "inBase").toNat
        (min ((ws.regs sp).toNat + l) (inStride - 4))).elim 0xffffffff UInt64.ofNat := by
  have hc4 := c64_sub4 (show 4 ≤ inStride by omega)
  have h64 : inStride - 4 < 2 ^ 64 := by omega
  have hposN : posP.toNat = (ws.regs sp).toNat + l := by
    rw [hpos, UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show l < 2 ^ 64 by omega), Nat.mod_eq_of_lt hs]
  have hrpN : rp.toNat = min ((ws.regs sp).toNat + l) (inStride - 4) := by
    rw [hrp, hc4]
    split
    · rename_i hle; rw [toNat_le_ofNat h64, hposN] at hle; rw [hposN]; omega
    · rename_i hle; rw [toNat_le_ofNat h64, hposN] at hle; rw [UInt64.toNat_ofNat_of_lt' h64]; omega
  have hv : v32.toNat = wLoad4 ws.gmem ((ws.regs "inBase").toNat + min ((ws.regs sp).toNat + l) (inStride - 4)) := by
    rw [← hrpN]; exact hv32
  have hH := hash_eq hv hh
  generalize hHd : wHash ws.gmem hashLog ((ws.regs "inBase").toNat + min ((ws.regs sp).toNat + l) (inStride - 4)) = H
    at hH
  have hHlt : H < 2 ^ hashLog := by rw [← hHd]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have hpow : 2 ^ hashLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by decide) hh
  have haddrN : addr.toNat = tbl + 2 * H := by
    rw [haddr, htbl, show c64 1 = UInt64.ofNat 1 by decide, UInt64.toNat_add, shl_toNat (by decide) (by omega),
      hH, UInt64.toNat_ofNat_of_lt' (show tbl < 2 ^ 64 by omega), Nat.shiftLeft_eq, Nat.mod_eq_of_lt (by omega)]
    omega
  have hHlt' := hHlt
  have hcrN := hcr (by rw [haddrN]; omega)
  rw [haddrN] at hcrN
  have horacle : ∀ q, q = min ((ws.regs sp).toNat + l) (inStride - 4) →
      tableOracle ws.gmem ws.smem hashLog tbl (ws.regs "inBase").toNat q =
        if candRaw.toNat = 0 then none else some (candRaw.toNat - 1) := by
    rintro q rfl
    simp only [tableOracle, hHd, hcrN]
  have hm1 : c64 (-1) = UInt64.ofNat (2 ^ 64 - 1) := by decide
  have hcN : candRaw.toNat ≠ 0 → cand.toNat = candRaw.toNat - 1 := by
    intro h0
    rw [hcand, hm1, UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (by decide)]
    have := UInt64.toNat_lt candRaw
    omega
  have hc0 : candRaw.toNat = 0 → cand = UInt64.ofNat (2 ^ 64 - 1) := by
    intro h0
    rw [hcand, hm1, show candRaw = 0 from UInt64.toNat_inj.mp (by simpa using h0)]
    decide
  have hraw : candRaw.toNat < 65536 := by
    rw [hcrN]
    have := (ws.smem.getD (tbl + 2 * H) 0).toNat_lt
    have := (ws.smem.getD (tbl + 2 * H + 1) 0).toNat_lt
    omega
  refine ⟨?_, ?_, haddrN, ?_⟩
  rotate_left 2
  · rw [horacle _ rfl]
    by_cases h0 : candRaw.toNat = 0
    · rw [if_pos h0, hc0 h0]; decide
    · rw [if_neg h0]
      apply UInt64.toNat_inj.mp
      rw [UInt64.toNat_and, hcN h0, show (0xffffffff : UInt64).toNat = 2 ^ 32 - 1 from rfl,
        Nat.and_two_pow_sub_one_eq_mod]
      simp only [Option.elim]
      rw [UInt64.toNat_ofNat_of_lt' (show candRaw.toNat - 1 < 2 ^ 64 by omega)]
      omega
  all_goals
    have hsl64 : searchLim < 2 ^ 64 := by omega
    try simp only [c64_ofNat]
    by_cases hpl : (ws.regs sp).toNat + l < searchLim
    case neg =>
      have hnone : AlgorithmLib.LZ4WarpFind.probe (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
          (tableOracle ws.gmem ws.smem hashLog tbl (ws.regs "inBase").toNat) searchLim
          ((ws.regs sp).toNat + l) = none := probe_past hpl
      first
        | (rw [hnone]; intro c hc; cases hc)
        | (rw [hnone, show decide (posP < UInt64.ofNat searchLim) = false by
              simp [toNat_lt_ofNat hsl64, hposN, hpl]]; simp)
    have hpc : min ((ws.regs sp).toNat + l) (inStride - 4) = (ws.regs sp).toNat + l := by omega
    have hor := horacle _ hpc.symm
    rw [hpc] at hv
    have hdec : decide (posP < UInt64.ofNat searchLim) = true := by simp [toNat_lt_ofNat hsl64, hposN, hpl]
    have hz0 : c64 0 = 0 := by decide
    by_cases h0 : candRaw.toNat = 0
    · have hnone : AlgorithmLib.LZ4WarpFind.probe (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
          (tableOracle ws.gmem ws.smem hashLog tbl (ws.regs "inBase").toNat) searchLim
          ((ws.regs sp).toNat + l) = none := by
        apply probe_none; rw [hor, if_pos h0]
      have hz : candRaw = 0 := UInt64.toNat_inj.mp (by simpa using h0)
      first
        | (rw [hnone]; intro c hc; cases hc)
        | (rw [hnone, hdec, hz, hz0]; simp)
    have hcn := hcN h0
    by_cases hcp : candRaw.toNat - 1 < (ws.regs sp).toNat + l
    · have hrcN : rc = cand := by
        rw [hrc, hc4, if_pos ((toNat_le_ofNat h64).mpr (by omega))]
      have hvcN : vc.toNat = wLoad4 ws.gmem ((ws.regs "inBase").toNat + (candRaw.toNat - 1)) := by
        rw [hrcN] at hvc; rw [← hcn]; exact hvc
      have hsome : AlgorithmLib.LZ4WarpFind.probe (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
          (tableOracle ws.gmem ws.smem hashLog tbl (ws.regs "inBase").toNat) searchLim
          ((ws.regs sp).toNat + l) = if wLoad4 ws.gmem ((ws.regs "inBase").toNat + (candRaw.toNat - 1)) =
            wLoad4 ws.gmem ((ws.regs "inBase").toNat + ((ws.regs sp).toNat + l))
            then some (candRaw.toNat - 1) else none := by
        rw [probe_some (by rw [hor, if_neg h0]), verify4_eq (by omega) (by omega)]
        by_cases hw : wLoad4 ws.gmem ((ws.regs "inBase").toNat + (candRaw.toNat - 1)) =
            wLoad4 ws.gmem ((ws.regs "inBase").toNat + ((ws.regs sp).toNat + l))
        · rw [if_pos ⟨hpl, hcp, by simpa using hw⟩, if_pos hw]
        · rw [if_neg (fun h => hw (by simpa using h.2.2)), if_neg hw]
      first
        | (rw [hsome]; intro c hc; split at hc
           · obtain rfl := Option.some.inj hc
             apply UInt64.toNat_inj.mp; rw [hcn, UInt64.toNat_ofNat_of_lt' (show candRaw.toNat - 1 < 2 ^ 64 by omega)]
           · cases hc)
        | (rw [hsome, hdec, show (candRaw != c64 0) = true from bne_iff_ne.mpr (fun e => h0 (by rw [e]; rfl))]
           have hlt : decide (cand < posP) = true := by
             simp [UInt64.lt_iff_toNat_lt, hcn, hposN, hcp]
           have heq : (vc == v32) = decide (wLoad4 ws.gmem ((ws.regs "inBase").toNat + (candRaw.toNat - 1)) =
               wLoad4 ws.gmem ((ws.regs "inBase").toNat + ((ws.regs sp).toNat + l))) := by
             rw [Bool.eq_iff_iff, beq_iff_eq, decide_eq_true_iff, ← UInt64.toNat_inj, hvcN, hv]
           rw [hlt, heq]
           by_cases hw : wLoad4 ws.gmem ((ws.regs "inBase").toNat + (candRaw.toNat - 1)) =
               wLoad4 ws.gmem ((ws.regs "inBase").toNat + ((ws.regs sp).toNat + l))
           · rw [if_pos hw, decide_eq_true hw]; rfl
           · rw [if_neg hw, decide_eq_false hw]; rfl)
    · have hnone : AlgorithmLib.LZ4WarpFind.probe (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
          (tableOracle ws.gmem ws.smem hashLog tbl (ws.regs "inBase").toNat) searchLim
          ((ws.regs sp).toNat + l) = none := by
        rw [probe_some (by rw [hor, if_neg h0]), if_neg (fun h => hcp h.2.1)]
      first
        | (rw [hnone]; intro c hc; cases hc)
        | (rw [hnone, hdec, show (candRaw != c64 0) = true from bne_iff_ne.mpr (fun e => h0 (by rw [e]; rfl))]
           have hlt : decide (cand < posP) = false := by
             simp [UInt64.lt_iff_toNat_lt, hcn, hposN]; omega
           rw [hlt]; simp)

theorem getD_setIfInBounds (arr : Array UInt8) (i j : Nat) (v : UInt8) (hj : j < arr.size) :
    (arr.setIfInBounds i v).getD j 0 = if j = i then v else arr.getD j 0 := by
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_setIfInBounds]
  by_cases e : j = i
  · subst e; simp [hj]
  · simp [Ne.symm e, e]

/-- **The warp's shared 16-bit stores are an array's two-byte writes**, lane
    by lane, wherever the array has a byte. -/
theorem storeH_arr (a v g : Lane → UInt64) :
    ∀ (L : List Lane) (st : ISt) (arr : Array UInt8), (∀ k, k < arr.size → st.smem k = arr.getD k 0) →
      ∀ k, k < arr.size →
        (L.foldl (fun s l => if g l ≠ 0 then
          { s with smem := fun x => if x = (a l).toNat then (v l).toUInt8
            else if x = (a l).toNat + 1 then ((v l) >>> 8).toUInt8 else s.smem x } else s) st).smem k =
        (L.foldl (fun acc l => if g l ≠ 0 then
          (acc.setIfInBounds (a l).toNat (v l).toUInt8).setIfInBounds ((a l).toNat + 1) ((v l) >>> 8).toUInt8
          else acc) arr).getD k 0
  | [], st, arr, h, k, hk => h k hk
  | l :: L, st, arr, h, k, hk => by
      rw [List.foldl_cons, List.foldl_cons]
      by_cases hg : g l ≠ 0
      · rw [if_pos hg, if_pos hg]
        refine storeH_arr a v g L _ _ (fun k' hk' => ?_) k (by simpa using hk)
        simp only [Array.size_setIfInBounds] at hk'
        dsimp only
        rw [getD_setIfInBounds _ _ _ _ (by simpa using hk'), getD_setIfInBounds _ _ _ _ hk']
        by_cases e1 : k' = (a l).toNat
        · rw [if_pos e1, if_neg (by omega), if_pos e1]
        · rw [if_neg e1, if_neg e1]
          by_cases e2 : k' = (a l).toNat + 1
          · rw [if_pos e2, if_pos e2]
          · rw [if_neg e2, if_neg e2]; exact h k' hk'
      · rw [if_neg hg, if_neg hg]
        exact storeH_arr a v g L st arr h k hk

set_option maxHeartbeats 4000000 in
/-- **The model's window**, by its first hitting lane: `p0` at it, `cand0` its
    candidate (lane 0's where none hits), `found` whether any lane hits, the
    lanes up to it inserting their positions into the table. -/
theorem window_model {W : WState} {found p0 cand0 sp : String} {inStride searchLim hashLog tbl f : Nat}
    (hfp : found ≠ p0) (hfc : found ≠ cand0) (hpc : p0 ≠ cand0) (hsl : searchLim + 4 ≤ inStride) :
    ((WStmt.coopWindow found p0 cand0 sp inStride searchLim hashLog tbl).eval f W).regs =
      (((W.setReg p0 (W.regs sp + UInt64.ofNat (firstSetNat (winHit W sp inStride searchLim hashLog tbl) 32 0))).setReg
        cand0 (winCand W sp inStride hashLog tbl (firstSetNat (winHit W sp inStride searchLim hashLog tbl) 32 0 % 32)
          &&& 0xffffffff)).setReg found
        (if (UInt32.ofNat (maskN (winHit W sp inStride searchLim hashLog tbl))).toUInt64 != 0 then 1 else 0)).regs ∧
    ((WStmt.coopWindow found p0 cand0 sp inStride searchLim hashLog tbl).eval f W).smem =
      tableInsert W.smem W.gmem hashLog tbl searchLim (W.regs "inBase").toNat (W.regs sp).toNat
        (firstSetNat (winHit W sp inStride searchLim hashLog tbl) 32 0) ∧
    ((WStmt.coopWindow found p0 cand0 sp inStride searchLim hashLog tbl).eval f W).gmem = W.gmem := by
  have hfs := firstSetNat_le (winHit W sp inStride searchLim hashLog tbl)
  have hfsN : firstSetNat (winHit W sp inStride searchLim hashLog tbl) 32 0 =
      (AlgorithmLib.LZ4WarpSched.firstHit (winHit W sp inStride searchLim hashLog tbl) 32 0).getD 32 :=
    firstSetNat_warp _
  have hwin := window_of_firstHit (inp := gmemInpAt W.gmem (W.regs "inBase").toNat inStride)
    (oracle := tableOracle W.gmem W.smem hashLog tbl (W.regs "inBase").toNat) (searchLim := searchLim)
    (s := (W.regs sp).toNat)
  refine ⟨?_, ?_, by rw [WStmt.eval.eq_11, evalCoopWindow_eq_go, evalCoopWindowGo_gmem]⟩
  · symm
    funext x
    rw [WStmt.eval.eq_11, evalCoopWindow_eq_go, evalCoopWindowGo_regs]
    have hs32 : ∀ L, L < 32 → (W.regs sp) + UInt64.ofNat L = UInt64.ofNat ((W.regs sp).toNat + L) := by
      intro L _; rw [UInt64.ofNat_add, UInt64.ofNat_toNat]
    cases e : AlgorithmLib.LZ4WarpSched.firstHit (winHit W sp inStride searchLim hashLog tbl) 32 0 with
    | none =>
        rw [hwin.2 e]
        have hall : ∀ j, j < 32 → winHit W sp inStride searchLim hashLog tbl j = false :=
          fun j hj => firstHit_none e j (Nat.zero_le _) (by omega)
        have hm0 : maskN (winHit W sp inStride searchLim hashLog tbl) = 0 := maskN_eq_zero hall
        have hfs32 : firstSetNat (winHit W sp inStride searchLim hashLog tbl) 32 0 = 32 := by rw [hfsN, e]; rfl
        have hc0 : winCand W sp inStride hashLog tbl 0 &&& 0xffffffff =
            (tableOracle W.gmem W.smem hashLog tbl (W.regs "inBase").toNat
              (min (W.regs sp).toNat (inStride - 4))).elim 0xffffffff UInt64.ofNat := by
          rw [winCand_mask]; simp [winCand, winPos]
        rw [hm0, hfs32, show (32 : Nat) % 32 = 0 from rfl, hc0]
        simp only [WState.setReg]
        by_cases hxf : x = found
        · subst hxf; simp [hfp, hfc]
        · by_cases hxc : x = cand0
          · subst hxc; simp [hxf]
          · by_cases hxp : x = p0
            · subst hxp; simp [hxf, hxc]
            · simp [hxf, hxc, hxp]
    | some L =>
        obtain ⟨c, hpc', hw⟩ := hwin.1 L e
        rw [hw]
        obtain ⟨hL, -, hL32⟩ := firstHit_some e
        have hfsL : firstSetNat (winHit W sp inStride searchLim hashLog tbl) 32 0 = L := by rw [hfsN, e]; rfl
        have hmne : (UInt32.ofNat (maskN (winHit W sp inStride searchLim hashLog tbl))).toUInt64 != 0 := by
          have hne := maskN_ne_zero (h := winHit W sp inStride searchLim hashLog tbl) (by omega) hL
          have hlt := maskN_lt (winHit W sp inStride searchLim hashLog tbl)
          simp only [bne_iff_ne, ne_eq]
          intro h0
          apply hne
          have := congrArg UInt64.toNat h0
          simpa [UInt32.toNat_ofNat_of_lt' hlt] using this
        obtain ⟨hor, hsl'⟩ := probe_some_inv hpc'
        have hcl' := tableOracle_lt hor
        have hcand : winCand W sp inStride hashLog tbl (L % 32) &&& 0xffffffff = UInt64.ofNat c := by
          rw [winCand_mask, Nat.mod_eq_of_lt (by omega)]
          simp only [winCand, winPos, winOracle]
          rw [show min ((W.regs sp).toNat + L) (inStride - 4) = (W.regs sp).toNat + L by omega, hor]
          rfl
        rw [hfsL, hcand, if_pos hmne, hs32 L (by omega)]
        simp only [WState.setReg]
        by_cases hxf : x = found
        · subst hxf; simp [hfp, hfc]
        · by_cases hxc : x = cand0
          · subst hxc; simp [hxf]
          · by_cases hxp : x = p0
            · subst hxp; simp [hxf, hxc]
            · simp [hxf, hxc, hxp]

  · rw [WStmt.eval.eq_11, evalCoopWindow_eq_go, evalCoopWindowGo_smem, hfsN]
    cases e : AlgorithmLib.LZ4WarpSched.firstHit (winHit W sp inStride searchLim hashLog tbl) 32 0 with
    | none => rw [hwin.2 e]; rfl
    | some L =>
        obtain ⟨c, -, hw⟩ := hwin.1 L e
        rw [hw]; simp

section Coop
variable {hw : Hw} {cta S : Nat} {ptr : Buf → UInt64} {I O : UInt64 → Prop} {ns fl : List String}

/-- **The extend step's translation agrees with the model's.** -/
theorem sim_extend {ws : WSt F32} {W : WState} (hS : SimW ns I ws W) {adv p0 cand0 ml : String} {inStride endCap : Nat}
    (hn : namesOf (.coopExtendStep adv p0 cand0 ml inStride endCap) ⊆ ns)
    (hok : okLZ4 I (.coopExtendStep adv p0 cand0 ml inStride endCap) W) (f i : Nat) :
    Out hw cta S ptr I O ns (ofWn ns fl (.coopExtendStep adv p0 cand0 ml inStride endCap))
      (.coopExtendStep adv p0 cand0 ml inStride endCap) f W ws i := by
  obtain ⟨hecR, hec1, hend, hendI, hcp, hov, hin, hib, hpe⟩ := hok
  have hR : ∀ x, x ∈ namesOf (.coopExtendStep adv p0 cand0 ml inStride endCap) → ns.idxOf x < ns.length :=
    fun x h => idxOf_lt (hn h)
  have ha := hR adv (by simp [namesOf])
  have hp := hR p0 (by simp [namesOf])
  have hc := hR cand0 (by simp [namesOf])
  have hm := hR ml (by simp [namesOf])
  have hib' := hR "inBase" (by simp [namesOf])
  have he1 := hR "ecR" (by simp [namesOf])
  have he2 := hR "ec1" (by simp [namesOf])
  have hlR : ∀ l : Lane, ws.int.regs (laneR ns.length) l = UInt64.ofNat l.val := hS.lane
  let b := ns.length + 23
  let e1 : IExp := .bin .add (.reg (ns.idxOf p0)) (.bin .add (.reg (ns.idxOf ml)) (.reg (laneR ns.length)))
  let e2 : IExp := .cmp .ult (.reg b) (.reg (ns.idxOf "ecR"))
  let e3 : IExp := .bin .umin (.reg b) (.reg (ns.idxOf "ec1"))
  let e4 : IExp := .bin .add (.reg (ns.idxOf cand0)) (.bin .sub (.reg (b + 2)) (.reg (ns.idxOf p0)))
  let e5 : IExp := .bin .add (.reg (ns.idxOf "inBase")) (.reg (b + 2))
  let e6 : IExp := .bin .add (.reg (ns.idxOf "inBase")) (.reg (b + 3))
  let e7 : IExp := .bin .band (.reg (b + 1)) (.cmp .eq (.reg (b + 4)) (.reg (b + 5)))
  let e9 : IExp := .ctz (.bin .bxor (.reg (b + 7)) (.lit 0xffffffff))
  let st1 := ws.int.set b (fun l => e1.eval ws.int cta l)
  let st2 := st1.set (b + 1) (fun l => e2.eval st1 cta l)
  let st3 := st2.set (b + 2) (fun l => e3.eval st2 cta l)
  let st4 := st3.set (b + 3) (fun l => e4.eval st3 cta l)
  let st5 := st4.set (b + 4) (fun l => st4.byte (e5.eval st4 cta l))
  let st6 := st5.set (b + 5) (fun l => st5.byte (e6.eval st5 cta l))
  let st7 := st6.set (b + 6) (fun l => e7.eval st6 cta l)
  let st8 := st7.set (b + 7) (fun _ => ISt.ballot (fun l => (IExp.reg (b + 6)).eval st7 cta l))
  let st9 := st8.set (ns.idxOf adv) (fun l => e9.eval st8 cta l)
  have hbody : ofWn ns fl (.coopExtendStep adv p0 cand0 ml inStride endCap) = .locals (List.range' b 8) (seqs [
      .setI b e1, .setI (b + 1) e2, .setI (b + 2) e3, .setI (b + 3) e4, .ldB (b + 4) e5, .ldB (b + 5) e6,
      .setI (b + 6) e7, .ballot (b + 7) (.reg (b + 6)), .setI (ns.idxOf adv) e9]) := rfl
  -- each lane's values
  let pe : Lane → UInt64 := fun l => W.regs p0 + (W.regs ml + UInt64.ofNat l.val)
  let peC : Lane → UInt64 := fun l => if pe l ≤ W.regs "ec1" then pe l else W.regs "ec1"
  let caC : Lane → UInt64 := fun l => W.regs cand0 + (peC l - W.regs p0)
  have v1 : ∀ l, st1.regs b l = pe l := fun l => by
    simp only [st1, set_regs_eq, e1, IExp.eval, IOp.run, hlR, hS.reg p0 (hn (by simp [namesOf])),
      hS.reg ml (hn (by simp [namesOf])), pe]
  have hb : b = ns.length + 23 := rfl
  have rp0 := hS.reg p0 (hn (by simp [namesOf]))
  have rc0 := hS.reg cand0 (hn (by simp [namesOf]))
  have rib := hS.reg "inBase" (hn (by simp [namesOf]))
  have rer := hS.reg "ecR" (hn (by simp [namesOf]))
  have re1 := hS.reg "ec1" (hn (by simp [namesOf]))
  have v2 : ∀ l, st2.regs (b + 1) l = if pe l < W.regs "ecR" then 1 else 0 := fun l => by
    simp (disch := omega) only [st2, set_regs_eq, set_regs_ne, e2, IExp.eval, ICmp.run, v1, st1, rer,
      decide_eq_true_eq]
  have v3 : ∀ l, st3.regs (b + 2) l = peC l := fun l => by
    simp (disch := omega) only [st3, set_regs_eq, set_regs_ne, e3, IExp.eval, IOp.run, st2, v1, st1, re1, peC]
  have v4 : ∀ l, st4.regs (b + 3) l = caC l := fun l => by
    simp (disch := omega) only [st4, set_regs_eq, set_regs_ne, e4, IExp.eval, IOp.run, st3, v3, st2, st1, rc0, rp0,
      caC]
  have hfl := fun (l : Lane) => ext_flag (I := I) (W := W) (p0 := p0) (cand0 := cand0) (ml := ml) (inStride := inStride)
    (endCap := endCap) l.isLt hecR hec1 hend hendI hcp hov hin hib hpe
  have v5 : ∀ l, st5.regs (b + 4) l = byteW W (W.regs "inBase" + peC l) := fun l => by
    simp (disch := omega) only [st5, set_regs_eq, set_regs_ne, e5, IExp.eval, IOp.run, ISt.byte, st4, set_gmem, v3,
      st3, st2, st1, rib]
    rw [hS.gin _ (hfl l).1]
  have v6 : ∀ l, st6.regs (b + 5) l = byteW W (W.regs "inBase" + caC l) := fun l => by
    simp (disch := omega) only [st6, set_regs_eq, set_regs_ne, e6, IExp.eval, IOp.run, ISt.byte, st5, set_gmem, v4,
      st4, st3, st2, st1, rib]
    rw [hS.gin _ (hfl l).2.1]
  have v7 : ∀ k, k < 32 → flagN (fun l => (IExp.reg (b + 6)).eval st7 cta l) k = extGood W p0 cand0 ml inStride endCap k := by
    intro k hk
    simp (disch := omega) only [flagN, hk, dite_true, IExp.eval, st7, set_regs_eq, e7, IOp.run, ICmp.run, set_regs_ne,
      v5, v6, st6, st5, st4, st3, v2]
    rw [flag_and]
    exact (hfl ⟨k, hk⟩).2.2
  have v8 : ∀ l, st8.regs (b + 7) l = (UInt32.ofNat (maskN (extGood W p0 cand0 ml inStride endCap))).toUInt64 :=
    fun l => by rw [set_regs_eq, ballot_maskN, maskN_congr v7]
  have hadv : ∀ l, e9.eval st8 cta l = UInt64.ofNat (AlgorithmLib.LZ4WarpSched.leadingGood
      (gmemInpAt W.gmem (W.regs "inBase").toNat inStride) (W.regs p0).toNat (W.regs cand0).toNat endCap
      (W.regs ml).toNat 32) := fun l => by
    simp only [e9, IExp.eval, IOp.run, v8, show UInt64.ofNat 4294967295 = 0xffffffff from rfl]
    rw [maskN_compl, ctz32_mask, firstSetNat_warp, ← AlgorithmLib.LZ4WarpSched.firstHit_leadingGood]
  have hev : (WStmt.coopExtendStep adv p0 cand0 ml inStride endCap).eval f W =
      W.setReg adv (UInt64.ofNat (AlgorithmLib.LZ4WarpSched.leadingGood
        (gmemInpAt W.gmem (W.regs "inBase").toNat inStride) (W.regs p0).toNat (W.regs cand0).toNat endCap
        (W.regs ml).toNat 32)) := by
    rw [WStmt.eval.eq_12]; rfl
  have hq : quiet (ofWn ns fl (.coopExtendStep adv p0 cand0 ml inStride endCap)) = true := by
    rw [hbody]; exact quiet_seqs _ (by simp [quiet])
  have hrun : wrun hw cta (ofWn ns fl (.coopExtendStep adv p0 cand0 ml inStride endCap)) ws i =
      ws.withInt (ISt.restore (List.range' b 8) ws.int st9) := by
    simp only [hbody, wrun, LaneWarp.run, seqs, EWStmt.run, withInt_int, withInt_withInt, st9, st8, st7, st6, st5,
      st4, st3, st2, st1]
  have hout : ∀ r, r < ns.length → (ISt.restore (List.range' b 8) ws.int st9).regs r = st9.regs r := fun r hr => by
    simp only [ISt.restore, List.mem_range']
    rw [if_neg (by omega)]
  refine ⟨?_, ?_, ?_, fun l => ?_, ?_⟩
  · rw [hbody]
    refine ⟨trivial, trivial, trivial, trivial, fun l => ?_, fun l => ?_, trivial, trivial, trivial⟩
    · show I (e5.eval st4 cta l)
      simp (disch := omega) only [e5, IExp.eval, IOp.run, v3, st4, set_regs_ne, st3, st2, st1, rib]
      exact (hfl l).1
    · show I (e6.eval st5 cta l)
      simp (disch := omega) only [e6, IExp.eval, IOp.run, v4, st5, set_regs_ne, st4, st3, st2, st1, rib]
      exact (hfl l).2.1
  · rw [hrun, hev]
    refine ⟨fun n hn' l => ?_, fun l => ?_, fun a ha' => ?_, fun k hk => ?_⟩
    · have hr := idxOf_lt hn'
      simp only [WSt.withInt]
      rw [hout _ hr]
      by_cases e : n = adv
      · subst e; simp only [st9, set_regs_eq, hadv, WState.setReg, if_true]
      · have hne : ns.idxOf n ≠ ns.idxOf adv := fun h => e (idxOf_inj hn' (hn (by simp [namesOf])) h)
        simp (disch := omega) only [st9, set_regs_ne _ _ hne, st8, set_regs_ne, st7, st6, st5, st4, st3, st2, st1]
        simp only [WState.setReg, if_neg e]; exact hS.reg n hn' l
    · simp only [WSt.withInt, ISt.restore, List.mem_range']
      rw [if_neg (by simp only [laneR]; omega)]
      simp (disch := (simp only [laneR]; omega)) only [st9, set_regs_ne, st8, st7, st6, st5, st4, st3, st2, st1]
      exact hS.lane l
    · exact hS.gin a ha'
    · exact hS.smem k hk
  · rw [wtr, (quiet_trace hw cta (fun _ _ => 0) lane0 ptr 0 _ hq i ws).2.1]; rfl
  · rw [wtr, (quiet_trace hw cta (fun _ _ => 0) l ptr _ _ hq i ws).1]; simp [gstOf, byteSt, lanePart]
  · rw [wtr, (quiet_trace hw cta (fun _ _ => 0) lane0 ptr 0 _ hq i ws).2.2]; exact Paired.nil

/-- **The copy's translation agrees with the model's.** -/
theorem sim_copy (hIdec : ∀ a, I a → ∃ r o, decodeAddr a = some (r, o))
    (hOdec : ∀ a, O a → ∃ r o, decodeAddr a = some (r, o))
    {ws : WSt F32} {W : WState} (hS : SimW ns I ws W) {d s n : String}
    (hn : namesOf (.coopCopy d s n) ⊆ ns) {f : Nat} (hok : okOf I O (okLZ4 I) f (.coopCopy d s n) W) (i : Nat) :
    Out hw cta S ptr I O ns (ofWn ns fl (.coopCopy d s n)) (.coopCopy d s n) f W ws i := by
  simp only [okOf] at hok
  obtain ⟨hn64, hsn, hdn, hsI, hdI⟩ := hok
  have hD := idxOf_lt (hn (by simp [namesOf] : d ∈ namesOf (.coopCopy d s n)))
  have hSr := idxOf_lt (hn (by simp [namesOf] : s ∈ namesOf (.coopCopy d s n)))
  have hN := idxOf_lt (hn (by simp [namesOf] : n ∈ namesOf (.coopCopy d s n)))
  have rd := hS.reg d (hn (by simp [namesOf]))
  have rs := hS.reg s (hn (by simp [namesOf]))
  have rn := hS.reg n (hn (by simp [namesOf]))
  let T := ns.length
  have hT : T = ns.length := rfl
  let c : IExp := .cmp .ult (.reg T) (.reg (ns.idxOf n))
  let a1 : EWStmt := .setI (T + 1) (.bin .add (.reg T) (.reg (laneR T)))
  let a2 : EWStmt := .ldB (T + 2) (.bin .add (.reg (ns.idxOf s)) (.reg (T + 1)))
  let a3 : EWStmt := .stB (.bin .add (.reg (ns.idxOf d)) (.reg (T + 1))) (.reg (T + 2))
    (.cmp .ult (.reg (T + 1)) (.reg (ns.idxOf n)))
  let a4 : EWStmt := .setI T (.bin .add (.reg T) (.lit 32))
  let body : EWStmt := .seq a1 (.seq a2 (.seq a3 a4))
  have hbody : ofWn ns fl (.coopCopy d s n) =
      .locals [T, T + 1, T + 2] (.seq (.setI T (.lit 0)) (.iwhile (2 ^ 64) c [T] body)) := rfl
  let nN := (W.regs n).toNat
  have hnN : nN = (W.regs n).toNat := rfl
  let e : Nat → UInt64 × UInt64 := fun j => (W.regs d + UInt64.ofNat j, (W.gmem.getD ((W.regs s).toNat + j) 0).toUInt64)
  have loop : ∀ F m (X : WSt F32), nN ≤ 32 * (m + F) → 32 * m ≤ nN + 32 →
      (∀ l, X.int.regs T l = UInt64.ofNat (32 * m)) → (∀ r, r ≠ T → X.int.regs r = ws.int.regs r) →
      (∀ a, I a → X.int.gmem a.toNat = ws.int.gmem a.toNat) → X.int.smem = ws.int.smem →
      OkL hw cta S ptr I O i (.iwhile F c [T] body) X true ∧
      (∀ r, r ≠ T → (LaneWarp.run hw cta (fun _ _ => 0) (.iwhile F c [T] body) X i).int.regs r = ws.int.regs r) ∧
      (∀ a, I a → (LaneWarp.run hw cta (fun _ _ => 0) (.iwhile F c [T] body) X i).int.gmem a.toNat =
        ws.int.gmem a.toNat) ∧
      (LaneWarp.run hw cta (fun _ _ => 0) (.iwhile F c [T] body) X i).int.smem = ws.int.smem ∧
      ∀ l : Lane, byteSt (LaneWarp.trace hw cta (fun _ _ => 0) (storeLeaf cta (fun _ _ => 0) l ptr (addrOf .arena 0)) i
        (.iwhile F c [T] body) X) =
        (List.range' (32 * m) (nN - 32 * m)).filterMap (fun j => if j % 32 = l.val then some (e j) else none) := by
    have h64 : nN + 64 < 2 ^ 64 := hn64
    have test : ∀ m (X : WSt F32), 32 * m ≤ nN + 32 → (∀ l, X.int.regs T l = UInt64.ofNat (32 * m)) →
        (∀ r, r ≠ T → X.int.regs r = ws.int.regs r) → ∀ l, (c.eval X.int cta l ≠ 0 ↔ 32 * m < nN) := by
      intro m X hm hXT hXr l
      have hlt : UInt64.ofNat (32 * m) < W.regs n ↔ 32 * m < nN := by
        rw [UInt64.lt_iff_toNat_lt, UInt64.toNat_ofNat_of_lt' (show 32 * m < 2 ^ 64 by omega)]
      simp only [c, IExp.eval, ICmp.run, hXT, hXr _ (show ns.idxOf n ≠ T by omega), rn]
      by_cases h : 32 * m < nN
      · rw [if_pos (decide_eq_true (hlt.mpr h))]; exact ⟨fun _ => h, fun _ => by decide⟩
      · rw [if_neg (fun e => h (hlt.mp (of_decide_eq_true e)))]; exact ⟨fun e => absurd rfl e, fun e => absurd e h⟩
    have stop : ∀ F m (X : WSt F32), ¬ 32 * m < nN → 32 * m ≤ nN + 32 →
        (∀ l, X.int.regs T l = UInt64.ofNat (32 * m)) → (∀ r, r ≠ T → X.int.regs r = ws.int.regs r) →
        (∀ a, I a → X.int.gmem a.toNat = ws.int.gmem a.toNat) → X.int.smem = ws.int.smem →
        OkL hw cta S ptr I O i (.iwhile F c [T] body) X true ∧
        (∀ r, r ≠ T → (LaneWarp.run hw cta (fun _ _ => 0) (.iwhile F c [T] body) X i).int.regs r = ws.int.regs r) ∧
        (∀ a, I a → (LaneWarp.run hw cta (fun _ _ => 0) (.iwhile F c [T] body) X i).int.gmem a.toNat =
          ws.int.gmem a.toNat) ∧
        (LaneWarp.run hw cta (fun _ _ => 0) (.iwhile F c [T] body) X i).int.smem = ws.int.smem ∧
        ∀ l : Lane, byteSt (LaneWarp.trace hw cta (fun _ _ => 0) (storeLeaf cta (fun _ _ => 0) l ptr (addrOf .arena 0)) i
          (.iwhile F c [T] body) X) =
          (List.range' (32 * m) (nN - 32 * m)).filterMap (fun j => if j % 32 = l.val then some (e j) else none) := by
      intro F m X h hm hXT hXr hXg hXs
      have ht : ¬ c.eval X.int cta lane0 ≠ 0 := fun h' => h ((test m X hm hXT hXr lane0).mp h')
      have hu : ∀ l, (c.eval X.int cta l ≠ 0 ↔ c.eval X.int cta lane0 ≠ 0) := fun l => by
        rw [test m X hm hXT hXr l, test m X hm hXT hXr lane0]
      have hr := run_iwhile_stop hw cta (fun _ _ => 0) F c [T] body X i ht
      refine ⟨okW_iwhile_stop hw cta (fun _ _ => 0) S ptr _ _ _ _ F c [T] body X i true hu ht, ?_, ?_, ?_, fun l => ?_⟩
      · rw [hr]; exact hXr
      · rw [hr]; exact hXg
      · rw [hr]; exact hXs
      · rw [trace_iwhile_stop hw cta (fun _ _ => 0) _ F c [T] body X i ht]
        simp [byteSt, show nN - 32 * m = 0 by omega]
    intro F
    induction F with
    | zero =>
      intro m X hF hm hXT hXr hXg hXs
      exact stop 0 m X (by omega) hm hXT hXr hXg hXs
    | succ F ih =>
      intro m X hF hm hXT hXr hXg hXs
      by_cases h : 32 * m < nN
      case neg => exact stop _ m X h hm hXT hXr hXg hXs
      have ht : c.eval X.int cta lane0 ≠ 0 := (test m X hm hXT hXr lane0).mpr h
      have hu : ∀ l, (c.eval X.int cta l ≠ 0 ↔ c.eval X.int cta lane0 ≠ 0) := fun l => by
        rw [test m X hm hXT hXr l, test m X hm hXT hXr lane0]
      obtain ⟨F', rfl⟩ : ∃ F', F = F' := ⟨F, rfl⟩
      let X1 := X.int.set (T + 1) (fun l => (IExp.bin .add (.reg T) (.reg (laneR T))).eval X.int cta l)
      let X2 := X1.set (T + 2) (fun l => X1.byte ((IExp.bin .add (.reg (ns.idxOf s)) (.reg (T + 1))).eval X1 cta l))
      let X3 := X2.storeB (fun l => (IExp.bin .add (.reg (ns.idxOf d)) (.reg (T + 1))).eval X2 cta l)
        (fun l => (IExp.reg (T + 2)).eval X2 cta l) (fun l => (IExp.cmp .ult (.reg (T + 1)) (.reg (ns.idxOf n))).eval X2 cta l)
      let X4 := X3.set T (fun l => (IExp.bin .add (.reg T) (.lit 32)).eval X3 cta l)
      have hr1 : LaneWarp.run hw cta (fun _ _ => 0) a1 X i = X.withInt X1 := rfl
      have hr2 : LaneWarp.run hw cta (fun _ _ => 0) a2 (X.withInt X1) i = X.withInt X2 := rfl
      have hr3 : LaneWarp.run hw cta (fun _ _ => 0) a3 (X.withInt X2) i = X.withInt X3 := rfl
      have hr4 : LaneWarp.run hw cta (fun _ _ => 0) a4 (X.withInt X3) i = X.withInt X4 := rfl
      have hrb : LaneWarp.run hw cta (fun _ _ => 0) body X i = X.withInt X4 := by
        show LaneWarp.run hw cta (fun _ _ => 0) a4 (LaneWarp.run hw cta (fun _ _ => 0) a3
          (LaneWarp.run hw cta (fun _ _ => 0) a2 (LaneWarp.run hw cta (fun _ _ => 0) a1 X i) i) i) i = _
        rw [hr1, hr2, hr3, hr4]
      have hwt : wtrip hw cta (fun _ _ => 0) c [T] body X i X = X.withInt (ISt.keep [T] X.int X4) := by
        simp only [wtrip, if_pos ht, hrb, withInt_int, withInt_withInt]
      have hlane : ∀ l : Lane, X.int.regs (laneR T) l = UInt64.ofNat l.val := fun l => by
        rw [hXr _ (by simp only [laneR]; omega)]; exact hS.lane l
      have j1 : ∀ l : Lane, X1.regs (T + 1) l = UInt64.ofNat (32 * m + l.val) := fun l => by
        simp only [X1, set_regs_eq, IExp.eval, IOp.run, hXT, hlane, ofNat_add_ofNat]
      have j2 : ∀ l : Lane, X2.regs (T + 1) l = UInt64.ofNat (32 * m + l.val) := fun l => by
        simp only [X2, set_regs_ne _ _ (show T + 1 ≠ T + 2 by omega), j1]
      have g2 : ∀ r, r < T → X2.regs r = ws.int.regs r := fun r hr => by
        simp only [X2, X1, set_regs_ne _ _ (show r ≠ T + 2 by omega), set_regs_ne _ _ (show r ≠ T + 1 by omega)]
        exact hXr r (by omega)
      have hsT : ∀ l : Lane, (IExp.bin .add (.reg (ns.idxOf s)) (.reg (T + 1))).eval X1 cta l =
          W.regs s + UInt64.ofNat (32 * m + l.val) := fun l => by
        simp only [IExp.eval, IOp.run, j1]
        simp only [X1, set_regs_ne _ _ (show ns.idxOf s ≠ T + 1 by omega), hXr _ (show ns.idxOf s ≠ T by omega), rs]
      have hdT : ∀ l : Lane, (IExp.bin .add (.reg (ns.idxOf d)) (.reg (T + 1))).eval X2 cta l =
          W.regs d + UInt64.ofNat (32 * m + l.val) := fun l => by
        simp only [IExp.eval, IOp.run, j2, g2 _ hD, rd]
      have hgT : ∀ l : Lane, (IExp.cmp .ult (.reg (T + 1)) (.reg (ns.idxOf n))).eval X2 cta l ≠ 0 ↔
          32 * m + l.val < nN := fun l => by
        have hl32 : l.val < 32 := l.isLt
        have hlt : UInt64.ofNat (32 * m + l.val) < W.regs n ↔ 32 * m + l.val < nN := by
          rw [UInt64.lt_iff_toNat_lt, UInt64.toNat_ofNat_of_lt' (show 32 * m + l.val < 2 ^ 64 by omega)]
        simp only [IExp.eval, ICmp.run, j2, g2 _ hN, rn]
        by_cases h : 32 * m + l.val < nN
        · rw [if_pos (decide_eq_true (hlt.mpr h))]; exact ⟨fun _ => h, fun _ => by decide⟩
        · rw [if_neg (fun e => h (hlt.mp (of_decide_eq_true e)))]; exact ⟨fun e => absurd rfl e, fun e => absurd e h⟩
      have hvT : ∀ l : Lane, 32 * m + l.val < nN → (IExp.reg (T + 2)).eval X2 cta l =
          (W.gmem.getD ((W.regs s).toNat + (32 * m + l.val)) 0).toUInt64 := fun l hl => by
        have hidx : (W.regs s + UInt64.ofNat (32 * m + l.val)).toNat = (W.regs s).toNat + (32 * m + l.val) := by
          rw [UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show 32 * m + l.val < 2 ^ 64 by omega)]
          exact Nat.mod_eq_of_lt (by omega)
        have hI := hsI (32 * m + l.val) (by omega)
        show X2.regs (T + 2) l = _
        simp only [X2]
        rw [set_regs_eq, hsT]
        show (X.int.gmem _).toUInt64 = _
        rw [hXg _ hI, hS.gin _ hI, hidx]
      -- the trip asks what a lane needs
      have hok1 : OkL hw cta S ptr I O i body X true := by
        refine ⟨trivial, ?_, ?_, trivial⟩
        · rw [hr1]; intro l
          show I ((IExp.bin .add (.reg (ns.idxOf s)) (.reg (T + 1))).eval X1 cta l)
          have hl32 : l.val < 32 := l.isLt
          rw [hsT]; exact hsI _ (by omega)
        · rw [hr1, hr2]
          intro l hg
          change (IExp.cmp .ult (.reg (T + 1)) (.reg (ns.idxOf n))).eval X2 cta l ≠ 0 at hg
          have hl := (hgT l).mp hg
          obtain ⟨hnI, hna, hO⟩ := hdI _ hl
          obtain ⟨r, o, hd⟩ := hOdec _ hO
          have har : r ≠ .arena := fun e => hna o (by rw [hd, e])
          show StoreOk ptr _ _ _ _ l ((IExp.bin .add (.reg (ns.idxOf d)) (.reg (T + 1))).eval X2 cta l) 1 _ _ ∧ _
          rw [hdT]
          refine ⟨⟨⟨rfl, hO⟩, r, o, hd, har, fun _ _ _ h => h.elim, fun l' a' ha' => Or.inr ?_, fun _ _ h => h.elim⟩,
            fun l' _ _ h => absurd (by simp only [withInt_int, hdT] at h; exact h) hnI⟩
          obtain ⟨r', o', hd'⟩ := hIdec a' ha'
          exact ⟨r', o', hd', apart_of_ne hd hd' (fun e => hnI (by rw [e]; exact ha'))⟩
      -- what a trip leaves
      have hX3 := storeB_regs (a := fun l => (IExp.bin .add (.reg (ns.idxOf d)) (.reg (T + 1))).eval X2 cta l)
        (v := fun l => (IExp.reg (T + 2)).eval X2 cta l)
        (g := fun l => (IExp.cmp .ult (.reg (T + 1)) (.reg (ns.idxOf n))).eval X2 cta l) X2
      have kT : ∀ l, (ISt.keep [T] X.int X4).regs T l = UInt64.ofNat (32 * (m + 1)) := fun l => by
        simp only [ISt.keep, List.mem_singleton, if_true, X4, set_regs_eq, IExp.eval, IOp.run]
        rw [show X3.regs = X2.regs from hX3.1]
        simp only [X2, X1, set_regs_ne _ _ (show T ≠ T + 2 by omega), set_regs_ne _ _ (show T ≠ T + 1 by omega), hXT,
          ofNat_add_ofNat, show 32 * m + 32 = 32 * (m + 1) by omega]
      have kr : ∀ r, r ≠ T → (ISt.keep [T] X.int X4).regs r = ws.int.regs r := fun r hr => by
        simp only [ISt.keep, List.mem_singleton, if_neg hr]; exact hXr r hr
      have kg : ∀ a, I a → (ISt.keep [T] X.int X4).gmem a.toNat = ws.int.gmem a.toNat := fun a ha => by
        show X3.gmem a.toNat = _
        rw [storeB_other _ _ _ X2 a.toNat (fun l hg => by
          have hl := (hgT l).mp hg
          simp only [hdT]
          intro e
          exact (hdI _ hl).1 (by rw [UInt64.toNat_inj.mp e]; exact ha))]
        exact hXg a ha
      have ks : (ISt.keep [T] X.int X4).smem = ws.int.smem := by
        show X3.smem = _
        rw [hX3.2.1]; exact hXs
      have hst : ∀ l : Lane, byteSt (LaneWarp.trace hw cta (fun _ _ => 0)
          (storeLeaf cta (fun _ _ => 0) l ptr (addrOf .arena 0)) i body X) =
          if 32 * m + l.val < nN then [e (32 * m + l.val)] else [] := fun l => by
        have htr : LaneWarp.trace hw cta (fun _ _ => 0) (storeLeaf cta (fun _ _ => 0) l ptr (addrOf .arena 0)) i body X =
            storeLeaf cta (fun _ _ => 0) l ptr (addrOf .arena 0) i a3 (X.withInt X2) := by
          show [] ++ ([] ++ (storeLeaf cta (fun _ _ => 0) l ptr (addrOf .arena 0) i a3
            (LaneWarp.run hw cta (fun _ _ => 0) a2 (LaneWarp.run hw cta (fun _ _ => 0) a1 X i) i) ++ [])) = _
          rw [hr1, hr2]; simp
        rw [htr]
        by_cases hl : 32 * m + l.val < nN
        · rw [if_pos hl]
          have hg : (IExp.cmp .ult (.reg (T + 1)) (.reg (ns.idxOf n))).eval X2 cta l ≠ 0 := (hgT l).mpr hl
          simp only [storeLeaf, a3, withInt_int, if_pos hg, hdT, hvT l hl, byte_mask, byteSt, List.filterMap_cons,
            List.filterMap_nil]
          rfl
        · rw [if_neg hl]
          have hg : ¬ (IExp.cmp .ult (.reg (T + 1)) (.reg (ns.idxOf n))).eval X2 cta l ≠ 0 := fun h' => hl ((hgT l).mp h')
          simp only [storeLeaf, a3, withInt_int, if_neg hg]; rfl
      obtain ⟨o', r', g', s', b'⟩ := ih (m + 1) (X.withInt (ISt.keep [T] X.int X4)) (by omega) (by omega) kT kr kg ks
      have hshb := noSh_trace hw cta (fun _ _ => 0) body rfl i X
      refine ⟨okW_iwhile_peel hw cta _ S ptr _ _ _ _ F c [T] body X i true hu ht hok1 (by rw [hshb.1, hwt]; exact o'),
        ?_, ?_, ?_, fun l => ?_⟩
      · rw [run_iwhile_peel, hwt]; exact r'
      · rw [run_iwhile_peel, hwt]; exact g'
      · rw [run_iwhile_peel, hwt]; exact s'
      · rw [trace_iwhile_peel hw cta (fun _ _ => 0) _ F c [T] body X i ht, byteSt_append, hst l, hwt, b' l,
          filterMap_blocks nN l.val l.isLt e m]
  let X0 := ws.withInt (ws.int.set T (fun l => (IExp.lit 0).eval ws.int cta l))
  obtain ⟨o, r, g, sm, b⟩ := loop (2 ^ 64) 0 X0 (by omega) (by omega) (fun l => by simp only [X0, withInt_int,
    set_regs_eq, IExp.eval]) (fun r hr => by simp only [X0, withInt_int, set_regs_ne _ _ hr])
    (fun a _ => rfl) rfl
  have hq : noSh (ofWn ns fl (.coopCopy d s n)) = true := by rw [hbody]; rfl
  have hrun : wrun hw cta (ofWn ns fl (.coopCopy d s n)) ws i =
      (LaneWarp.run hw cta (fun _ _ => 0) (.iwhile (2 ^ 64) c [T] body) X0 i).withInt
        (ISt.restore [T, T + 1, T + 2] ws.int (LaneWarp.run hw cta (fun _ _ => 0) (.iwhile (2 ^ 64) c [T] body) X0 i).int) := by
    have h0 : LaneWarp.run hw cta (fun _ _ => 0) (.setI T (.lit 0)) ws i = X0 := rfl
    rw [hbody, wrun, run_locals, run_seq, h0]
  have hev : (WStmt.coopCopy d s n).eval f W =
      { W with gmem := copyGmem W.gmem (W.regs d).toNat (W.regs s).toNat (W.regs n).toNat } := by
    simp only [WStmt.eval]
  refine ⟨by rw [hbody]; exact ⟨trivial, o⟩, ?_, ?_, fun l => ?_, ?_⟩
  · rw [hrun]
    generalize LaneWarp.run hw cta (fun _ _ => 0) (.iwhile (2 ^ 64) c [T] body) X0 i = Y at r g sm ⊢
    have hev' := congrArg WState.regs hev
    have hevg := congrArg WState.gmem hev
    have hevs := congrArg WState.smem hev
    refine simW_restore (fun n' hn' l => ?_) (fun l => ?_) (fun a ha => ?_) (fun k hk => ?_)
    · have hr := idxOf_lt hn'
      simp only [ISt.restore, List.mem_cons, List.mem_nil_iff, or_false]
      rw [if_neg (by omega), r _ (by omega), hev']
      exact hS.reg n' hn' l
    · simp only [ISt.restore, List.mem_cons, List.mem_nil_iff, or_false]
      rw [if_neg (by simp only [laneR]; omega), r _ (by simp only [laneR]; omega)]
      exact hS.lane l
    · rw [g a ha, hS.gin a ha, hevg]
      refine (copyGmem_getD_out _ _ _ _ _ (fun j hj e => (hdI j hj).1 ?_)).symm
      have hdj : (W.regs d + UInt64.ofNat j).toNat = (W.regs d).toNat + j := by
        rw [UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show j < 2 ^ 64 by omega)]
        exact Nat.mod_eq_of_lt (by omega)
      rw [show W.regs d + UInt64.ofNat j = a from UInt64.toNat_inj.mp (by rw [hdj, e])]; exact ha
    · rw [hevs] at hk ⊢
      rw [sm]; exact hS.smem k hk
  · rw [wtr, (noSh_trace hw cta (fun _ _ => 0) _ hq i ws).1]; rfl
  · have h0 : LaneWarp.run hw cta (fun _ _ => 0) (.setI T (.lit 0)) ws i = X0 := rfl
    rw [wtr, hbody, trace_locals, trace_seq, h0,
      show LaneWarp.trace hw cta (fun _ _ => 0) (storeLeaf cta (fun _ _ => 0) l ptr (addrOf .arena 0)) i
        (.setI T (.lit 0)) ws = [] from rfl, List.nil_append, b l]
    simp only [gstOf]
    rw [lanePart_copyTr]
  · rw [wtr, (noSh_trace hw cta (fun _ _ => 0) _ hq i ws).2]; exact Paired.nil

set_option maxHeartbeats 4000000 in
/-- **The window's translation agrees with the model's**, its table within
    the kernel's shared memory. -/
theorem sim_window {ws : WSt F32} {W : WState} (hS : SimW ns I ws W)
    {found p0 cand0 sp : String} {inStride searchLim hashLog tbl : Nat}
    (hn : namesOf (.coopWindow found p0 cand0 sp inStride searchLim hashLog tbl) ⊆ ns)
    (hok : okLZ4 I (.coopWindow found p0 cand0 sp inStride searchLim hashLog tbl) W)
    (hSz : W.smem.size ≤ S) (hS36 : S ≤ 2 ^ 36) (f i : Nat) :
    Out hw cta S ptr I O ns (ofWn ns fl (.coopWindow found p0 cand0 sp inStride searchLim hashLog tbl))
      (.coopWindow found p0 cand0 sp inStride searchLim hashLog tbl) f W ws i := by
  obtain ⟨hsl, hh, hin, hib, hsp, htbl, htb, hfp, hfc, hpc, hsz⟩ := hok
  have hmem : ∀ x, x ∈ namesOf (.coopWindow found p0 cand0 sp inStride searchLim hashLog tbl) → x ∈ ns := fun x h => hn h
  have hF := idxOf_lt (hmem found (by simp [namesOf]))
  have hP0 := idxOf_lt (hmem p0 (by simp [namesOf]))
  have hC0 := idxOf_lt (hmem cand0 (by simp [namesOf]))
  have hSP := idxOf_lt (hmem sp (by simp [namesOf]))
  have hIB := idxOf_lt (hmem "inBase" (by simp [namesOf]))
  have hTB := idxOf_lt (hmem "tbl" (by simp [namesOf]))
  have rsp := hS.reg sp (hmem sp (by simp [namesOf]))
  have rib := hS.reg "inBase" (hmem "inBase" (by simp [namesOf]))
  have rtb := hS.reg "tbl" (hmem "tbl" (by simp [namesOf]))
  let T := ns.length
  let b := T + 3
  have hb : b = ns.length + 3 := rfl
  let P1 : List EWStmt := [.setI b (.bin .add (.reg (ns.idxOf sp)) (.reg (laneR T))),
    .setI (b + 1) (.cmp .ult (.reg b) (litI searchLim)),
    .setI (b + 2) (.bin .umin (.reg b) (litI (inStride - 4)))]
  let L1 := load4E (b + 3) (b + 14) (.bin .add (.reg (ns.idxOf "inBase")) (.reg (b + 2)))
  let P2 : List EWStmt := [
    .setI (b + 4) (.bin .band (.bin .shr (.bin .mul (.reg (b + 3)) (litI wHashK)) (litI (32 - hashLog)))
      (litI (2 ^ hashLog - 1))),
    .setI (b + 5) (.bin .add (.bin .shl (.reg (b + 4)) (.lit 1)) (.reg (ns.idxOf "tbl"))),
    .ldH (b + 6) (.reg (b + 5)),
    .setI (b + 7) (.bin .add (.reg (b + 6)) (litI (-1))),
    .setI (b + 8) (.bin .umin (.reg (b + 7)) (litI (inStride - 4))) ]
  let L2 := load4E (b + 9) (b + 14) (.bin .add (.reg (ns.idxOf "inBase")) (.reg (b + 8)))
  let Suf := winSuf T (ns.idxOf found) (ns.idxOf p0) (ns.idxOf cand0) (ns.idxOf sp)
  have hwin : ofWn ns fl (.coopWindow found p0 cand0 sp inStride searchLim hashLog tbl) =
      .locals (List.range' b 19) (seqs ((P1 ++ (L1 ++ (P2 ++ L2))) ++ Suf)) := rfl
  have hT : T = ns.length := rfl
  have hlane : ∀ l : Lane, ws.int.regs (laneR T) l = UInt64.ofNat l.val := hS.lane
  have hLb : laneR T = b + 38 := by simp only [laneR]; omega
  -- each lane's probe
  let posP : Lane → UInt64 := fun l => W.regs sp + UInt64.ofNat l.val
  let rpv : Lane → UInt64 := fun l => if posP l ≤ c64 (inStride - 4) then posP l else c64 (inStride - 4)
  let Y1 := LaneWarp.run hw cta (fun _ _ => 0) (seqs P1) ws i
  have y1p : ∀ l, Y1.int.regs b l = posP l := fun l => by
    simp (disch := omega) only [Y1, P1, seqs, LaneWarp.run, EWStmt.run, withInt_int, set_regs_eq, set_regs_ne,
      IExp.eval, IOp.run, rsp, hlane, posP]
  have y1v : ∀ l, Y1.int.regs (b + 1) l = if posP l < c64 searchLim then 1 else 0 := fun l => by
    simp (disch := omega) only [Y1, P1, seqs, LaneWarp.run, EWStmt.run, withInt_int, set_regs_eq, set_regs_ne,
      IExp.eval, IOp.run, ICmp.run, litI_evalW, decide_eq_true_eq, rsp, hlane, posP]
  have y1r : ∀ l, Y1.int.regs (b + 2) l = rpv l := fun l => by
    simp (disch := omega) only [Y1, P1, seqs, LaneWarp.run, EWStmt.run, withInt_int, set_regs_eq, set_regs_ne,
      IExp.eval, IOp.run, litI_evalW, rsp, hlane, rpv, posP]
  have y1f : ∀ r, r < b ∨ b + 18 < r → Y1.int.regs r = ws.int.regs r := fun r hr => by
    simp (disch := omega) only [Y1, P1, seqs, LaneWarp.run, EWStmt.run, withInt_int, set_regs_ne]
  have y1g : Y1.int.gmem = ws.int.gmem := rfl
  have y1s : Y1.int.smem = ws.int.smem := rfl
  let v32 : Lane → UInt64 := fun l => w4 ws.int (W.regs "inBase" + rpv l)
  let Y2 := LaneWarp.run hw cta (fun _ _ => 0) (seqs L1) Y1 i
  obtain ⟨y2v, y2f, y2g, y2s⟩ := load4_run hw cta (fun _ _ => 0) (b + 3) (b + 14)
    (.bin .add (.reg (ns.idxOf "inBase")) (.reg (b + 2))) Y1 i
  have y2v' : ∀ l, Y2.int.regs (b + 3) l = v32 l := fun l => by
    rw [y2v l, w4_congr y1g]
    simp only [IExp.eval, IOp.run, y1r, y1f _ (show ns.idxOf "inBase" < b ∨ b + 18 < ns.idxOf "inBase" by omega), rib, v32]
  have y2f' : ∀ r, r < b + 3 ∨ b + 18 < r → Y2.int.regs r = Y1.int.regs r := fun r hr =>
    y2f r (by omega) (by omega)
  have y2s' : Y2.int.smem = ws.int.smem := y2s.trans y1s
  have y2g' : Y2.int.gmem = ws.int.gmem := y2g.trans y1g
  let addrv : Lane → UInt64 := fun l =>
    ((((v32 l * c64 wHashK) >>> c64 (32 - hashLog)) &&& c64 (2 ^ hashLog - 1)) <<< UInt64.ofNat 1) + W.regs "tbl"
  let crv : Lane → UInt64 := fun l => ws.int.half (addrv l)
  let candv : Lane → UInt64 := fun l => crv l + c64 (-1)
  let rcv : Lane → UInt64 := fun l => if candv l ≤ c64 (inStride - 4) then candv l else c64 (inStride - 4)
  let Y3 := LaneWarp.run hw cta (fun _ _ => 0) (seqs P2) Y2 i
  have y2tb : ∀ l, Y2.int.regs (ns.idxOf "tbl") l = W.regs "tbl" := fun l => by
    rw [y2f' _ (by omega), y1f _ (by omega), rtb]
  have y3a : ∀ l, Y3.int.regs (b + 5) l = addrv l := fun l => by
    simp (disch := omega) only [Y3, P2, seqs, LaneWarp.run, EWStmt.run, withInt_int, set_regs_eq, set_regs_ne,
      IExp.eval, IOp.run, litI_evalW, y2v', y2tb, addrv]
  have y3c : ∀ l, Y3.int.regs (b + 6) l = crv l := fun l => by
    simp (disch := omega) only [Y3, P2, seqs, LaneWarp.run, EWStmt.run, withInt_int, set_regs_eq, set_regs_ne,
      IExp.eval, IOp.run, litI_evalW, y2v', y2tb, ISt.half, set_smem, y2s', crv, addrv]
  have y3cand : ∀ l, Y3.int.regs (b + 7) l = candv l := fun l => by
    simp (disch := omega) only [Y3, P2, seqs, LaneWarp.run, EWStmt.run, withInt_int, set_regs_eq, set_regs_ne,
      IExp.eval, IOp.run, litI_evalW, y2v', y2tb, ISt.half, set_smem, y2s', candv, crv, addrv]
  have y3rc : ∀ l, Y3.int.regs (b + 8) l = rcv l := fun l => by
    simp (disch := omega) only [Y3, P2, seqs, LaneWarp.run, EWStmt.run, withInt_int, set_regs_eq, set_regs_ne,
      IExp.eval, IOp.run, litI_evalW, y2v', y2tb, ISt.half, set_smem, y2s', rcv, candv, crv, addrv]
  have y3f : ∀ r, r < b + 4 ∨ b + 8 < r → Y3.int.regs r = Y2.int.regs r := fun r hr => by
    simp (disch := omega) only [Y3, P2, seqs, LaneWarp.run, EWStmt.run, withInt_int, set_regs_ne]
  have y3m := regOnly_mem hw cta (fun _ _ => 0) (seqs P2) rfl Y2 i
  have y3g : Y3.int.gmem = ws.int.gmem := y3m.1.trans y2g'
  have y3s : Y3.int.smem = ws.int.smem := y3m.2.trans y2s'
  let vcv : Lane → UInt64 := fun l => w4 ws.int (W.regs "inBase" + rcv l)
  let Y4 := LaneWarp.run hw cta (fun _ _ => 0) (seqs L2) Y3 i
  obtain ⟨y4v, y4f, y4g, y4s⟩ := load4_run hw cta (fun _ _ => 0) (b + 9) (b + 14)
    (.bin .add (.reg (ns.idxOf "inBase")) (.reg (b + 8))) Y3 i
  have y4v' : ∀ l, Y4.int.regs (b + 9) l = vcv l := fun l => by
    rw [y4v l, w4_congr y3g]
    simp only [IExp.eval, IOp.run, y3rc, y3f _ (show ns.idxOf "inBase" < b + 4 ∨ b + 8 < ns.idxOf "inBase" by omega),
      y2f' _ (show ns.idxOf "inBase" < b + 3 ∨ b + 18 < ns.idxOf "inBase" by omega),
      y1f _ (show ns.idxOf "inBase" < b ∨ b + 18 < ns.idxOf "inBase" by omega), rib, vcv]
  have y4f' : ∀ r, r < b + 9 ∨ b + 18 < r → Y4.int.regs r = Y3.int.regs r := fun r hr =>
    y4f r (by omega) (by omega)
  have h4 : 4 ≤ inStride := by omega
  have hI64 : inStride < 2 ^ 64 := by omega
  have word : ∀ x : UInt64, x.toNat ≤ inStride - 4 →
      (w4 ws.int (W.regs "inBase" + x)).toNat = wLoad4 W.gmem ((W.regs "inBase").toNat + x.toNat) := fun x hx => by
    have hA : (W.regs "inBase" + x).toNat = (W.regs "inBase").toNat + x.toNat := by
      rw [UInt64.toNat_add]; exact Nat.mod_eq_of_lt (by omega)
    have hI : ∀ k, k < 4 → I (W.regs "inBase" + x + UInt64.ofNat k) := fun k hk => by
      rw [UInt64.add_assoc, show x + UInt64.ofNat k = UInt64.ofNat (x.toNat + k) by
        rw [UInt64.ofNat_add, UInt64.ofNat_toNat]]
      exact hin _ (by omega)
    rw [w4_word hS hI (by omega), hA]
  have hpv := fun (l : Lane) => probe_vals (ws := W) (l := l.val) (sp := sp) hib hsl hh
    (by have : l.val < 32 := l.isLt; omega) htbl htb (posP := posP l) (rp := rpv l) (v32 := v32 l) (addr := addrv l)
    (candRaw := crv l) (cand := candv l) (rc := rcv l) (vc := vcv l) rfl rfl
    (word _ (clamp_le h4 hI64))
    (by simp only [addrv, show c64 1 = UInt64.ofNat 1 by decide])
    (fun hb2 => by
      simp only [crv, ISt.half]
      rw [hS.smem _ (by omega), hS.smem _ (by omega), half_toNat])
    rfl rfl (word _ (clamp_le h4 hI64))
  -- Y4's registers the lanes' values
  have z_pos : ∀ l, Y4.int.regs b l = posP l := fun l => by
    rw [y4f' _ (by omega), y3f _ (by omega), y2f' _ (by omega), y1p]
  have z_pv : ∀ l, Y4.int.regs (b + 1) l = if posP l < c64 searchLim then 1 else 0 := fun l => by
    rw [y4f' _ (by omega), y3f _ (by omega), y2f' _ (by omega), y1v]
  have z_v32 : ∀ l, Y4.int.regs (b + 3) l = v32 l := fun l => by
    rw [y4f' _ (by omega), y3f _ (by omega), y2v']
  have z_addr : ∀ l, Y4.int.regs (b + 5) l = addrv l := fun l => by rw [y4f' _ (by omega), y3a]
  have z_cr : ∀ l, Y4.int.regs (b + 6) l = crv l := fun l => by rw [y4f' _ (by omega), y3c]
  have z_cand : ∀ l, Y4.int.regs (b + 7) l = candv l := fun l => by rw [y4f' _ (by omega), y3cand]
  have z_lane : ∀ l : Lane, Y4.int.regs (laneR T) l = UInt64.ofNat l.val := fun l => by
    rw [y4f' _ (by omega), y3f _ (by omega), y2f' _ (by omega), y1f _ (by omega), hlane]
  have z_f : ∀ r, r < b → Y4.int.regs r = ws.int.regs r := fun r hr => by
    rw [y4f' _ (by omega), y3f _ (by omega), y2f' _ (by omega), y1f _ (by omega)]
  have y4m : Y4.int.gmem = ws.int.gmem ∧ Y4.int.smem = ws.int.smem := ⟨y4g.trans y3g, y4s.trans y3s⟩
  -- the warp's choice
  let FS := firstSetNat (winHit W sp inStride searchLim hashLog tbl) 32 0
  have hFS : FS ≤ 32 := firstSetNat_le _
  let M : UInt64 := (UInt32.ofNat (maskN (winHit W sp inStride searchLim hashLog tbl))).toUInt64
  let sHit : EWStmt := .setI (b + 10) (.bin .band (.bin .band (.reg (b + 1)) (.cmp .ne (.reg (b + 6)) (.lit 0)))
    (.bin .band (.cmp .ult (.reg (b + 7)) (.reg b)) (.cmp .eq (.reg (b + 9)) (.reg (b + 3)))))
  let Z1 := LaneWarp.run hw cta (fun _ _ => 0) sHit Y4 i
  have z1 : ∀ k, k < 32 → flagN (fun l => (IExp.reg (b + 10)).eval Z1.int cta l) k =
      winHit W sp inStride searchLim hashLog tbl k := fun k hk => by
    simp only [flagN, hk, dite_true, IExp.eval, Z1, sHit, EWStmt.run, withInt_int, set_regs_eq, IOp.run,
      ICmp.run, z_pv, z_cr, z_cand, z_pos, z_v32, decide_eq_true_eq]
    rw [show (b + 9) = b + 9 from rfl, y4v', flag4P, show (UInt64.ofNat 0) = c64 0 from rfl]
    exact (hpv ⟨k, hk⟩).1
  let sBal : EWStmt := .ballot (b + 11) (.reg (b + 10))
  let Z2 := LaneWarp.run hw cta (fun _ _ => 0) sBal Z1 i
  have z2 : ∀ l, Z2.int.regs (b + 11) l = M := fun l => by
    show (Z1.int.set (b + 11) (fun _ => ISt.ballot (fun l => (IExp.reg (b + 10)).eval Z1.int cta l))).regs (b + 11) l = M
    rw [set_regs_eq, ballot_maskN, maskN_congr z1]
  let sFirst : EWStmt := .setI (b + 12) (.ctz (.reg (b + 11)))
  let Z3 := LaneWarp.run hw cta (fun _ _ => 0) sFirst Z2 i
  have z3 : ∀ l, Z3.int.regs (b + 12) l = UInt64.ofNat FS := fun l => by
    show (Z2.int.set (b + 12) (fun l => (IExp.ctz (.reg (b + 11))).eval Z2.int cta l)).regs (b + 12) l = _
    rw [set_regs_eq]
    show AlgorithmLib.ML.ctz32 ((Z2.int.regs (b + 11) l) &&& 0xffffffff) = _
    rw [z2, mask32, ctz32_mask]
  let sIns : EWStmt := .stH (.reg (b + 5)) (.bin .add (.reg b) (.lit 1))
    (.bin .band (.cmp .ule (.reg (laneR T)) (.reg (b + 12))) (.reg (b + 1)))
  let Z4 := LaneWarp.run hw cta (fun _ _ => 0) sIns Z3 i
  have hZ4 := storeH_regs (a := fun l => (IExp.reg (b + 5)).eval Z3.int cta l)
    (v := fun l => (IExp.bin .add (.reg b) (.lit 1)).eval Z3.int cta l)
    (g := fun l => (IExp.bin .band (.cmp .ule (.reg (laneR T)) (.reg (b + 12))) (.reg (b + 1))).eval Z3.int cta l) Z3.int
  have z4r : Z4.int.regs = Z3.int.regs := hZ4.1
  let Z5 := LaneWarp.run hw cta (fun _ _ => 0) .barrier Z4 i
  let sShf : EWStmt := .shflIdx (b + 13) (.reg (b + 7)) (.reg (b + 12))
  let Z6 := LaneWarp.run hw cta (fun _ _ => 0) sShf Z5 i
  -- registers of Z3 below the window's own
  have z3f : ∀ r, r < b + 10 ∨ b + 12 < r → Z3.int.regs r = Y4.int.regs r := fun r hr => by
    show ((Z1.int.set (b + 11) _).set (b + 12) _).regs r = _
    rw [set_regs_ne _ _ (by omega), set_regs_ne _ _ (by omega)]
    show (Y4.int.set (b + 10) _).regs r = _
    rw [set_regs_ne _ _ (by omega)]
  have hsrc : ISt.srcLane (UInt64.ofNat FS) = ⟨FS % 32, Nat.mod_lt _ (by decide)⟩ := by
    apply Fin.ext
    show (UInt64.ofNat FS).toNat % 32 = FS % 32
    rw [UInt64.toNat_ofNat_of_lt' (show FS < 2 ^ 64 by omega)]
  have z6 : ∀ l, Z6.int.regs (b + 13) l = winCand W sp inStride hashLog tbl (FS % 32) := fun l => by
    show (Z5.int.set (b + 13) (fun l => (IExp.reg (b + 7)).eval Z5.int cta
      (ISt.srcLane ((IExp.reg (b + 12)).eval Z5.int cta l)) &&& 0xffffffff)).regs (b + 13) l = _
    rw [set_regs_eq]
    show Z4.int.regs (b + 7) _ &&& 0xffffffff = _
    rw [z4r, show IExp.eval Z5.int cta l (IExp.reg (b + 12)) = Z3.int.regs (b + 12) l from congrFun (congrFun z4r _) l,
      z3, hsrc, z3f _ (by omega), z_cand]
    exact (hpv _).2.2.2
  let sP0 : EWStmt := .setI (ns.idxOf p0) (.bin .add (.reg (ns.idxOf sp)) (.reg (b + 12)))
  let Z7 := LaneWarp.run hw cta (fun _ _ => 0) sP0 Z6 i
  let sC0 : EWStmt := .setI (ns.idxOf cand0) (.reg (b + 13))
  let Z8 := LaneWarp.run hw cta (fun _ _ => 0) sC0 Z7 i
  let sF : EWStmt := .setI (ns.idxOf found) (.cmp .ne (.reg (b + 11)) (.lit 0))
  let Z9 := LaneWarp.run hw cta (fun _ _ => 0) sF Z8 i
  have hSuf : Suf = [sHit, sBal, sFirst, sIns, .barrier, sShf, sP0, sC0, sF] := rfl
  have hsuf : LaneWarp.run hw cta (fun _ _ => 0) (seqs Suf) Y4 i = Z9 := by
    rw [hSuf]; simp only [seqs, run_seq]; rfl
  have hpre : LaneWarp.run hw cta (fun _ _ => 0) (seqs (P1 ++ (L1 ++ (P2 ++ L2)))) ws i = Y4 := by
    rw [run_seqs_append hw cta _ (L1 ++ (P2 ++ L2)) (by simp [L1, load4E]) P1 (by simp [P1]),
      run_seqs_append hw cta _ (P2 ++ L2) (by simp [P2]) L1 (by simp [L1, load4E]),
      run_seqs_append hw cta _ L2 (by simp [L2, load4E]) P2 (by simp [P2])]
  have hrun : wrun hw cta (ofWn ns fl (.coopWindow found p0 cand0 sp inStride searchLim hashLog tbl)) ws i =
      Z9.withInt (ISt.restore (List.range' b 19) ws.int Z9.int) := by
    rw [hwin, wrun, run_locals, run_seqs_append hw cta _ Suf (by simp [hSuf]) _ (by simp [P1]), hpre, hsuf]
  -- Z9's registers
  have z6r : ∀ r, r < b + 10 ∨ b + 13 < r → Z6.int.regs r = Y4.int.regs r := fun r hr => by
    show (Z5.int.set (b + 13) _).regs r = _
    rw [set_regs_ne _ _ (by omega)]
    show Z4.int.regs r = _
    rw [z4r, z3f _ (by omega)]
  have z6b : Z6.int.regs (b + 11) = Z2.int.regs (b + 11) := by
    show (Z5.int.set (b + 13) _).regs (b + 11) = _
    rw [set_regs_ne _ _ (by omega)]
    show Z4.int.regs (b + 11) = _
    rw [z4r]
    show ((Z2.int.set (b + 12) _)).regs (b + 11) = _
    rw [set_regs_ne _ _ (by omega)]
  have z6f : Z6.int.regs (b + 12) = fun _ => UInt64.ofNat FS := by
    show (Z5.int.set (b + 13) _).regs (b + 12) = _
    rw [set_regs_ne _ _ (by omega)]
    show Z4.int.regs (b + 12) = _
    rw [z4r]; funext l; exact z3 l
  have z8b : Z8.int.regs (b + 11) = Z6.int.regs (b + 11) := by
    show ((Z6.int.set (ns.idxOf p0) _).set (ns.idxOf cand0) _).regs (b + 11) = _
    rw [set_regs_ne _ _ (by omega), set_regs_ne _ _ (by omega)]
  have z7c : Z7.int.regs (b + 13) = Z6.int.regs (b + 13) := by
    show (Z6.int.set (ns.idxOf p0) _).regs (b + 13) = _
    rw [set_regs_ne _ _ (by omega)]
  have z9 : ∀ n ∈ ns, ∀ l, Z9.int.regs (ns.idxOf n) l =
      (((W.setReg p0 (W.regs sp + UInt64.ofNat FS)).setReg cand0
        (winCand W sp inStride hashLog tbl (FS % 32) &&& 0xffffffff)).setReg found (if M != 0 then 1 else 0)).regs n := by
    intro n hn' l
    have hr := idxOf_lt hn'
    show ((((Z6.int.set (ns.idxOf p0) _).set (ns.idxOf cand0) _).set (ns.idxOf found) _)).regs (ns.idxOf n) l = _
    by_cases ef : n = found
    · subst ef
      rw [set_regs_eq]
      simp only [WState.setReg, if_true, IExp.eval, ICmp.run]
      simp only [z8b, z6b, z2]; rfl
    · rw [set_regs_ne _ _ (fun h => ef (idxOf_inj hn' (hmem found (by simp [namesOf])) h))]
      by_cases ec : n = cand0
      · subst ec
        rw [set_regs_eq]
        simp only [WState.setReg, if_neg ef, if_true, IExp.eval]
        rw [z7c, z6, winCand_mask]
      · rw [set_regs_ne _ _ (fun h => ec (idxOf_inj hn' (hmem cand0 (by simp [namesOf])) h))]
        by_cases ep : n = p0
        · subst ep
          rw [set_regs_eq]
          simp only [WState.setReg, if_neg ef, if_neg ec, if_true, IExp.eval, IOp.run]
          rw [z6f, z6r _ (Or.inl (by omega)), z_f _ (by omega), rsp]
        · rw [set_regs_ne _ _ (fun h => ep (idxOf_inj hn' (hmem p0 (by simp [namesOf])) h)), z6r _ (by omega),
            z_f _ (by omega)]
          simp only [WState.setReg, if_neg ef, if_neg ec, if_neg ep]
          exact hS.reg n hn' l
  have hWm := window_model (W := W) (found := found) (p0 := p0) (cand0 := cand0) (sp := sp) (inStride := inStride)
    (searchLim := searchLim) (hashLog := hashLog) (tbl := tbl) (f := f) hfp hfc hpc hsl
  have z3s : Z3.int.smem = ws.int.smem :=
    ((regOnly_mem hw cta (fun _ _ => 0) sFirst rfl Z2 i).2.trans ((regOnly_mem hw cta (fun _ _ => 0) sBal rfl Z1 i).2.trans
      ((regOnly_mem hw cta (fun _ _ => 0) sHit rfl Y4 i).2))).trans y4m.2
  have hposN : ∀ l : Lane, (posP l).toNat = (W.regs sp).toNat + l.val := fun l => by
    have : l.val < 32 := l.isLt
    simp only [posP]
    rw [UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show l.val < 2 ^ 64 by omega)]
    exact Nat.mod_eq_of_lt (by omega)
  have z4i : Z4.int = Z3.int.storeH (fun l => (IExp.reg (b + 5)).eval Z3.int cta l)
      (fun l => (IExp.bin .add (.reg b) (.lit 1)).eval Z3.int cta l)
      (fun l => (IExp.bin .band (.cmp .ule (.reg (laneR T)) (.reg (b + 12))) (.reg (b + 1))).eval Z3.int cta l) := rfl
  have z9s : Z9.int.smem = Z4.int.smem :=
    (regOnly_mem hw cta (fun _ _ => 0) sF rfl Z8 i).2.trans ((regOnly_mem hw cta (fun _ _ => 0) sC0 rfl Z7 i).2.trans
      ((regOnly_mem hw cta (fun _ _ => 0) sP0 rfl Z6 i).2.trans ((regOnly_mem hw cta (fun _ _ => 0) sShf rfl Z5 i).2.trans
        (regOnly_mem hw cta (fun _ _ => 0) .barrier rfl Z4 i).2)))
  have hsm : ∀ k, k < ((WStmt.coopWindow found p0 cand0 sp inStride searchLim hashLog tbl).eval f W).smem.size →
      Z9.int.smem k = ((WStmt.coopWindow found p0 cand0 sp inStride searchLim hashLog tbl).eval f W).smem.getD k 0 := by
    intro k hk
    rw [hWm.2.1] at hk ⊢
    rw [tableInsert_size] at hk
    rw [z9s, z4i]
    unfold ISt.storeH tableInsert
    rw [storeH_arr _ _ _ _ Z3.int W.smem (fun k' hk' => by rw [z3s]; exact hS.smem k' hk') k hk]
    congr 1
    rw [← finRange_val, List.foldl_map]
    congr 1
    funext acc l
    have hl32 : l.val < 32 := l.isLt
    simp only [IExp.eval, IOp.run, ICmp.run, z3, z3f _ (show b + 5 < b + 10 ∨ b + 12 < b + 5 by omega),
      z3f _ (show b < b + 10 ∨ b + 12 < b by omega), z3f _ (show b + 1 < b + 10 ∨ b + 12 < b + 1 by omega),
      z3f _ (show laneR T < b + 10 ∨ b + 12 < laneR T by omega), z_addr, z_pos, z_pv, z_lane, decide_eq_true_eq]
    have hle : UInt64.ofNat l.val ≤ UInt64.ofNat FS ↔ l.val ≤ FS := by
      rw [UInt64.le_iff_toNat_le, UInt64.toNat_ofNat_of_lt' (show l.val < 2 ^ 64 by omega),
        UInt64.toNat_ofNat_of_lt' (show FS < 2 ^ 64 by omega)]
    have hlt : posP l < c64 searchLim ↔ (W.regs sp).toNat + l.val < searchLim := by
      rw [show c64 (searchLim : Int) = UInt64.ofNat searchLim from c64_ofNat _, UInt64.lt_iff_toNat_lt, hposN,
        UInt64.toNat_ofNat_of_lt' (show searchLim < 2 ^ 64 by omega)]
    by_cases hc : l.val ≤ FS ∧ (W.regs sp).toNat + l.val < searchLim
    · rw [if_pos hc, if_pos (by rw [if_pos (hle.mpr hc.1), if_pos (hlt.mpr hc.2)]; decide)]
      have ha : (addrv l).toNat = tbl + 2 * wHash W.gmem hashLog ((W.regs "inBase").toNat + ((W.regs sp).toNat + l.val)) := by
        rw [(hpv l).2.2.1, show min ((W.regs sp).toNat + l.val) (inStride - 4) = (W.regs sp).toNat + l.val by omega]
      have hv : posP l + UInt64.ofNat 1 = UInt64.ofNat ((W.regs sp).toNat + l.val + 1) := by
        apply UInt64.toNat_inj.mp
        rw [UInt64.toNat_add, hposN, UInt64.toNat_ofNat_of_lt' (by decide),
          UInt64.toNat_ofNat_of_lt' (show (W.regs sp).toNat + l.val + 1 < 2 ^ 64 by omega)]
        exact Nat.mod_eq_of_lt (by omega)
      rw [ha, hv, lo8 (by omega), hi8 (by omega)]
    · rw [if_neg hc]
      apply if_neg
      intro h; apply h
      by_cases h1 : l.val ≤ FS
      · rw [if_pos (hle.mpr h1), if_neg (fun h => hc ⟨h1, hlt.mp h⟩)]; decide
      · rw [if_neg (fun h => h1 (hle.mp h))]; exact UInt64.zero_and
  have z9g : Z9.int.gmem = ws.int.gmem :=
    ((regOnly_mem hw cta (fun _ _ => 0) sF rfl Z8 i).1.trans ((regOnly_mem hw cta (fun _ _ => 0) sC0 rfl Z7 i).1.trans
      ((regOnly_mem hw cta (fun _ _ => 0) sP0 rfl Z6 i).1.trans ((regOnly_mem hw cta (fun _ _ => 0) sShf rfl Z5 i).1.trans
        (regOnly_mem hw cta (fun _ _ => 0) .barrier rfl Z4 i).1)))).trans
    ((show Z4.int.gmem = Z3.int.gmem by rw [z4i]; exact hZ4.2.1).trans
      (((regOnly_mem hw cta (fun _ _ => 0) sFirst rfl Z2 i).1.trans ((regOnly_mem hw cta (fun _ _ => 0) sBal rfl Z1 i).1.trans
        ((regOnly_mem hw cta (fun _ _ => 0) sHit rfl Y4 i).1))).trans y4m.1))
  have z9l : ∀ l : Lane, Z9.int.regs (laneR T) l = UInt64.ofNat l.val := fun l => by
    show ((((Z6.int.set (ns.idxOf p0) _).set (ns.idxOf cand0) _).set (ns.idxOf found) _)).regs (laneR T) l = _
    rw [set_regs_ne _ _ (by omega), set_regs_ne _ _ (by omega), set_regs_ne _ _ (by omega), z6r _ (by omega), z_lane]
  have hq : quiet (seqs (P1 ++ (L1 ++ (P2 ++ L2)))) = true := rfl
  have hIk : ∀ x : UInt64, x.toNat ≤ inStride - 4 → ∀ k, k < 4 → I (W.regs "inBase" + x + UInt64.ofNat k) :=
    fun x hx k hk => by
      rw [UInt64.add_assoc, show x + UInt64.ofNat k = UInt64.ofNat (x.toNat + k) by
        rw [UInt64.ofNat_add, UInt64.ofNat_toNat]]
      exact hin _ (by omega)
  have haddrS : ∀ l, (addrv l).toNat + 2 ≤ S := fun l => by
    have := wHash_lt W.gmem hashLog ((W.regs "inBase").toNat + min ((W.regs sp).toNat + l.val) (inStride - 4))
    rw [(hpv l).2.2.1]; omega
  refine ⟨?_, ?_, ?_, fun l => ?_, ?_⟩
  · -- what the lanes need
    rw [hwin]
    show OkW hw cta (fun _ _ => 0) S ptr _ _ _ _ i (seqs ((P1 ++ (L1 ++ (P2 ++ L2))) ++ Suf)) ws true
    refine okW_seqs_append hw cta _ S ptr _ _ _ _ Suf (by simp [hSuf]) _ (by simp [P1]) ws i true ?_ ?_
    · refine okW_seqs_append hw cta _ S ptr _ _ _ _ (L1 ++ (P2 ++ L2)) (by simp [L1, load4E]) P1 (by simp [P1])
        ws i true ⟨trivial, trivial, trivial⟩ ?_
      rw [(quiet_trace hw cta _ lane0 ptr 0 (seqs P1) rfl i ws).2.1]
      refine okW_seqs_append hw cta _ S ptr _ _ _ _ (P2 ++ L2) (by simp [P2]) L1 (by simp [L1, load4E]) Y1 i true
        (load4_ok hw cta _ S ptr _ _ _ _ _ _ _ Y1 i _ (fun l k hk => ?_)) ?_
      · show I ((IExp.bin .add (.reg (ns.idxOf "inBase")) (.reg (b + 2))).eval Y1.int cta l + UInt64.ofNat k)
        simp only [IExp.eval, IOp.run, y1r, y1f _ (show ns.idxOf "inBase" < b ∨ b + 18 < ns.idxOf "inBase" by omega), rib]
        exact hIk _ (clamp_le h4 hI64) k hk
      rw [(quiet_trace hw cta _ lane0 ptr 0 (seqs L1) rfl i Y1).2.1]
      refine okW_seqs_append hw cta _ S ptr _ _ _ _ L2 (by simp [L2, load4E]) P2 (by simp [P2]) Y2 i true
        ⟨trivial, trivial, ⟨rfl, hS36, fun l => ?_⟩, trivial, trivial⟩ ?_
      · simp (disch := omega) only [LaneWarp.run, EWStmt.run, withInt_int, set_regs_eq, set_regs_ne, IExp.eval, IOp.run,
          litI_evalW, y2v', y2tb]
        exact haddrS l
      rw [(quiet_trace hw cta _ lane0 ptr 0 (seqs P2) rfl i Y2).2.1]
      refine load4_ok hw cta _ S ptr _ _ _ _ _ _ _ Y3 i _ (fun l k hk => ?_)
      show I ((IExp.bin .add (.reg (ns.idxOf "inBase")) (.reg (b + 8))).eval Y3.int cta l + UInt64.ofNat k)
      simp only [IExp.eval, IOp.run, y3rc, y3f _ (show ns.idxOf "inBase" < b + 4 ∨ b + 8 < ns.idxOf "inBase" by omega),
        y2f' _ (show ns.idxOf "inBase" < b + 3 ∨ b + 18 < ns.idxOf "inBase" by omega),
        y1f _ (show ns.idxOf "inBase" < b ∨ b + 18 < ns.idxOf "inBase" by omega), rib]
      exact hIk _ (clamp_le h4 hI64) k hk
    · rw [(quiet_trace hw cta _ lane0 ptr 0 _ hq i ws).2.1, hpre, hSuf]
      refine ⟨trivial, trivial, trivial, ⟨hS36, fun l _ => ?_⟩, trivial, trivial, trivial, trivial, trivial⟩
      show (Z3.int.regs (b + 5) l).toNat + 2 ≤ S
      rw [z3f _ (by omega), z_addr]
      exact haddrS l
  · -- the model's window
    rw [hrun]
    refine simW_restore (fun n hn' l => ?_) (fun l => ?_) (fun a ha => ?_) hsm
    · have hr := idxOf_lt hn'
      simp only [ISt.restore, List.mem_range']
      rw [if_neg (by omega), z9 n hn' l, hWm.1]
    · simp only [ISt.restore, List.mem_range']
      rw [if_neg (by simp only [laneR]; omega)]
      exact z9l l
    · rw [z9g, hS.gin a ha, hWm.2.2]
  · -- shared memory clean at the end: the barrier is the last of the window's shared steps
    rw [wtr, hwin, trace_locals, trace_seqs_append hw cta _ shLeaf Suf (by simp [hSuf]) _ (by simp [P1]),
      (quiet_trace hw cta _ lane0 ptr 0 _ hq i ws).2.1, hpre, hSuf]
    rfl
  · rw [wtr, noStB_trace hw cta _ l ptr _ _ (by rw [hwin]; rfl) i ws]
    simp [gstOf, lanePart]
  · rw [wtr, hwin, trace_locals, trace_seqs_append hw cta _ shEvLeaf Suf (by simp [hSuf]) _ (by simp [P1]),
      (quiet_trace hw cta _ lane0 ptr 0 _ hq i ws).2.2, hpre, hSuf, List.nil_append]
    intro k ws' a v g h
    match k, h with
    | 0, _ => exact ⟨_, rfl⟩
    | 1, h => cases h
    | _ + 2, h => cases h


/-- **The translation of every statement agrees with the model's**: the
    plain statements by `sim_stmt`, the cooperative ones by `sim_copy`,
    `sim_window` and `sim_extend`. -/
theorem sim (hIdec : ∀ a, I a → ∃ r o, decodeAddr a = some (r, o))
    (hOdec : ∀ a, O a → ∃ r o, decodeAddr a = some (r, o)) (hS36 : S ≤ 2 ^ 36) :
    ∀ (s : WStmt), namesOf s ⊆ ns → Wf fl s → ∀ f W ws i, f ≤ 2 ^ 64 → BitsW fl W → W.smem.size ≤ S →
      okOf I O (okLZ4 I) f s W → SimW ns I ws W → Out hw cta S ptr I O ns (ofWn ns fl s) s f W ws i :=
  sim_stmt hIdec hOdec fun s hs f W ws i hn _ _ _ hsz hok hS => by
    match s, hs with
    | .coopCopy .., _ => exact sim_copy hIdec hOdec hS hn hok i
    | .coopWindow .., _ => simp only [okOf] at hok; exact sim_window hS hn hok hsz hS36 f i
    | .coopExtendStep .., _ => simp only [okOf] at hok; exact sim_extend hS hn hok f i

end Coop

end Lz4Code
