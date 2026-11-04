module
public import Lz4.CodeLoop
public import AlgorithmLib.LZ4.EvalValid
public import AlgorithmLib.LZ4.EncodeLen
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lz4.CodeBody` — what the warp's run of a block's body loads and stores

The body is its encoding prefix, then the length header (`body_split`,
`wseq_append`). The prefix emits the encoded block at the output's start
(`Emits.prefix`), its length the encoded length (`compressorBody_output_eq`),
at most `lenOff` (`planBlock_encode_le_lenOff`); the header stores four bytes
at `lenOff` (`lenHeader_addrs`). So, from a block's start, the warp's run makes
only the accesses the lanes need and stores each byte of its output once
(`body_obligations`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Prog
  AlgorithmLib.LZ4WarpDSL AlgorithmLib.LibSpec AlgorithmLib.Kernel LZ4WarpEvalBytes
  AlgorithmLib.LZ4Plan

open AlgorithmLib.ML.LaneWarp (NotArena Writable)

namespace Lz4Code

/-- Two statements the warp runs alike: the same run, the same byte stores,
    the same accesses. -/
def RunsAs (I : UInt64 → Prop) (okC : WStmt → WState → Prop) (a b : WStmt) : Prop :=
  ∀ f ws, a.eval f ws = b.eval f ws ∧ gstOf f a ws = gstOf f b ws ∧ (okBase I okC f a ws ↔ okBase I okC f b ws)

theorem wseq_append {I : UInt64 → Prop} {okC : WStmt → WState → Prop} :
    ∀ (L M : List WStmt), L ≠ [] → M ≠ [] → RunsAs I okC (wseq (L ++ M)) (.seq (wseq L) (wseq M))
  | [], _, h, _ => absurd rfl h
  | [x], M, _, hM => by
      intro f ws
      cases M with
      | nil => exact absurd rfl hM
      | cons y M => exact ⟨rfl, rfl, Iff.rfl⟩
  | x :: y :: L, M, _, hM => by
      intro f ws
      have ih := wseq_append (I := I) (okC := okC) (y :: L) M (List.cons_ne_nil _ _) hM
      have hw : wseq (x :: y :: L ++ M) = .seq x (wseq (y :: L ++ M)) := by
        cases hL : y :: L ++ M with
        | nil => simp at hL
        | cons z R => simp [wseq, hL]
      rw [hw]
      obtain ⟨h1, h2, h3⟩ := ih f (x.eval f ws)
      refine ⟨?_, ?_, ?_⟩
      · simp only [WStmt.eval, wseq]; rw [h1]; simp [WStmt.eval]
      · simp only [gstOf, wseq]; rw [h2]; simp [gstOf, WStmt.eval, List.append_assoc]
      · simp only [okBase, wseq]; rw [h3]; simp [okBase, WStmt.eval, and_assoc]

/-- The encoding prefix's statements. -/
def prefixList (inStride hashLog : Nat) : List WStmt :=
  [ .setp .lt "loopC" "searchPos" (.imm (inStride - 12)),
    .uwhile "loopC" (loopCBodyStmt inStride hashLog),
    .mov "fLen" (.imm inStride),
    .bin .sub "fLen" "fLen" (.reg "litAnchor"),
    wEmitFinalSeq "litAnchor" "fLen" ]

/-- The length header's statements: the cursor's four bytes at `lenOff`. -/
def lenHeader (lenOff : Nat) : List WStmt :=
  [ .bin .add "la0" "outBase" (.imm lenOff),
    .bin .band "lb" "op" (.imm 255), .stgB "la0" "lb",
    .bin .add "la1" "la0" (.imm 1), .bin .shr "ls" "op" (.imm 8), .bin .band "lb" "ls" (.imm 255), .stgB "la1" "lb",
    .bin .add "la2" "la0" (.imm 2), .bin .shr "ls" "op" (.imm 16), .bin .band "lb" "ls" (.imm 255), .stgB "la2" "lb",
    .bin .add "la3" "la0" (.imm 3), .bin .shr "ls" "op" (.imm 24), .bin .band "lb" "ls" (.imm 255), .stgB "la3" "lb" ]

theorem body_split (inStride lenOff hashLog : Nat) :
    compressorBodyWStmt inStride lenOff hashLog = wseq (prefixList inStride hashLog ++ lenHeader lenOff) := rfl

theorem prefix_eq (inStride hashLog : Nat) : bodyEncodePrefix inStride hashLog = wseq (prefixList inStride hashLog) := rfl

/-- The length header stores its four bytes at `outBase + lenOff` on. -/
theorem lenHeader_addrs (lenOff f : Nat) (ws : WState) :
    addrsOf f (wseq (lenHeader lenOff)) ws = [ws.regs "outBase" + UInt64.ofNat lenOff,
      ws.regs "outBase" + UInt64.ofNat lenOff + 1, ws.regs "outBase" + UInt64.ofNat lenOff + 2,
      ws.regs "outBase" + UInt64.ofNat lenOff + 3] := by
  simp [addrsOf, lenHeader, wseq, gstOf, WStmt.eval, WState.setReg, WState.stgByte, WArg.eval, WOp.run]

theorem lenHeader_base {I : UInt64 → Prop} {okC : WStmt → WState → Prop} (lenOff f : Nat) (ws : WState) :
    okBase I okC f (wseq (lenHeader lenOff)) ws := by
  simp [okBase, lenHeader, wseq]

/-- The output of a block: `n` bytes from `ob`. -/
def outputOf (ob : UInt64) (n : Nat) (a : UInt64) : Prop := ∃ j, j < n ∧ a = ob + UInt64.ofNat j

theorem clearMem_size (hashLog : Nat) : (clearMem hashLog).data.size = 2 ^ hashLog * 2 := by
  simp [clearMem]

set_option maxHeartbeats 4000000 in
/-- **The warp's run of a block's body makes only the accesses the lanes
    need, and writes each byte once**: from the state a block's body starts
    in, its loads lie in the block's input, its stores in its output (the
    encoded block, then the length header at `lenOff`), outside the input and
    shared memory, each byte stored once. -/
theorem body_obligations {inStride lenOff hashLog : Nat} {ib ob : UInt64} (ws : WState)
    (hst : 16 ≤ inStride) (hst2 : inStride ≤ 65536) (hh : hashLog ≤ 32) (htb : 2 * 2 ^ hashLog + 2 < 2 ^ 36)
    (hop0 : ws.regs "op" = 0) (hla0 : ws.regs "litAnchor" = 0) (hsp0 : ws.regs "searchPos" = 0)
    (hib : ws.regs "inBase" = ib) (hob : ws.regs "outBase" = ob) (hib40 : ib.toNat < 2 ^ 40)
    (hobN : ob.toNat + 9 * inStride < 2 ^ 64) (hsize : ob.toNat + 9 * inStride ≤ ws.gmem.size)
    (hobLB : ib.toNat + inStride + 32 ≤ ob.toNat) (htbl : ws.regs "tbl" = 0) (hsz : 2 * 2 ^ hashLog ≤ ws.smem.size)
    (hlen : inStride + inStride / 16 + 256 ≤ lenOff) (hOv : ob.toNat + lenOff + 4 < 2 ^ 64)
    (hna : ∀ j, j < lenOff + 4 → NotArena (ob + UInt64.ofNat j)) :
    okOf (blockIn ib inStride) (outputOf ob (lenOff + 4)) (okLZ4 (blockIn ib inStride)) (inStride + 34 * inStride)
        (compressorBodyWStmt inStride lenOff hashLog) ws ∧
      (gstOf (inStride + 34 * inStride) (compressorBodyWStmt inStride lenOff hashLog) ws).Pairwise
        (fun x y => x.2.1 ≠ y.2.1) := by
  generalize hF : inStride + 34 * inStride = F
  obtain ⟨k, e⟩ := Emits.prefix (inStride := inStride) (hashLog := hashLog) ws hst hst2 hh htb hop0 hla0 hsp0 hib
    hib40 (by rw [hob]; exact hobN) (by rw [hob]; exact hsize) (by rw [hib, hob]; omega)
    htbl hsz
  rw [hF] at e
  -- the prefix emits the encoded block
  have hv := evalPlan_valid ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog hst2
  have hilen : (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride).length = inStride := by simp [gmemInpAt]
  have h9 := planBlock_encode_le9 _ _ hv (by rw [hilen]; omega)
  rw [hilen] at h9
  have hL := planBlock_encode_le_lenOff _ _ hv inStride hilen
  obtain ⟨-, hopE, -⟩ := compressorBody_output_eq inStride hashLog ws hst2 (by omega) (by omega) hop0 hla0 hsp0
    (by rw [hib]; exact hib40) (by rw [hob]; omega) (by rw [hob]; exact hsize) (by rw [hib, hob]; omega)
    (by rw [hib]; omega) h9
  rw [hF] at hopE
  have hk : k = (planToBlock (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
      (evalPlan ws.gmem ws.smem (ws.regs "inBase").toNat inStride hashLog)).encode.length := by
    have h1 := e.op
    have h2 := e.lt
    rw [prefix_eq] at h1 hopE
    rw [hopE, hop0, UInt64.zero_add] at h1
    have := congrArg UInt64.toNat h1
    rw [UInt64.toNat_ofNat_of_lt' (show _ < 2 ^ 64 by omega), UInt64.toNat_ofNat_of_lt' (show k < 2 ^ 64 by
      rw [hop0] at h2; simpa using h2)] at this
    omega
  have hkL : k ≤ lenOff := by omega
  -- the body's byte stores: the encoded block, then the length header
  have hsplit := wseq_append (I := blockIn ib inStride) (okC := okLZ4 (blockIn ib inStride))
    (prefixList inStride hashLog) (lenHeader lenOff) (by simp [prefixList]) (by simp [lenHeader]) F ws
  rw [← body_split] at hsplit
  obtain ⟨-, hg, hb⟩ := hsplit
  have hpre := e.addrs
  rw [prefix_eq] at hpre
  have hob' : ((wseq (prefixList inStride hashLog)).eval F ws).regs "outBase" = ob := by
    have := e.ob; rw [prefix_eq] at this; rw [this, hob]
  have haddrs : (gstOf F (compressorBodyWStmt inStride lenOff hashLog) ws).map (·.2.1) =
      ((List.range k ++ [lenOff, lenOff + 1, lenOff + 2, lenOff + 3]).map fun j => ob + UInt64.ofNat j) := by
    rw [hg]
    simp only [gstOf, List.map_append]
    have h1 : (gstOf F (wseq (prefixList inStride hashLog)) ws).map (·.2.1) = outRun ws k := hpre
    have h2 := lenHeader_addrs lenOff F ((wseq (prefixList inStride hashLog)).eval F ws)
    simp only [addrsOf] at h2
    rw [h1, h2, hob', outRun, hob, hop0]
    simp only [List.map_append, List.map_map, List.map_cons, List.map_nil]
    congr 1
    · apply List.map_congr_left; intro i _; simp
    · simp only [UInt64.ofNat_add, UInt64.add_assoc]; rfl
  -- each stored byte: in the output, outside the input and shared memory, once
  have hmem : ∀ x ∈ gstOf F (compressorBodyWStmt inStride lenOff hashLog) ws,
      ∃ j, j < lenOff + 4 ∧ x.2.1 = ob + UInt64.ofNat j := by
    intro x hx
    have : x.2.1 ∈ (gstOf F (compressorBodyWStmt inStride lenOff hashLog) ws).map (·.2.1) := List.mem_map_of_mem hx
    rw [haddrs] at this
    simp only [List.mem_map, List.mem_append, List.mem_range, List.mem_cons, List.mem_nil_iff, or_false] at this
    obtain ⟨j, hj, he⟩ := this
    exact ⟨j, by omega, he.symm⟩
  refine ⟨okOf_of _ _ _ (by rw [hb]; simp only [okBase]; exact ⟨by rw [← prefix_eq]; exact e.base,
    lenHeader_base lenOff F _⟩) fun x hx => ?_, ?_⟩
  · obtain ⟨j, hj, he⟩ := hmem x hx
    refine ⟨fun ⟨j', hj', he'⟩ => ?_, he ▸ hna j hj, ⟨j, hj, he⟩⟩
    have := congrArg UInt64.toNat (he.symm.trans he')
    rw [UInt64.toNat_add, UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show j < 2 ^ 64 by omega),
      UInt64.toNat_ofNat_of_lt' (show j' < 2 ^ 64 by omega), Nat.mod_eq_of_lt (by omega),
      Nat.mod_eq_of_lt (by omega)] at this
    omega
  · rw [← List.pairwise_map (R := fun a b => a ≠ b), haddrs, List.pairwise_map]
    have hnd : (List.range k ++ [lenOff, lenOff + 1, lenOff + 2, lenOff + 3]).Nodup := by
      rw [List.nodup_append]
      refine ⟨List.nodup_range, by simp, ?_⟩
      intro a ha b hb e
      simp at ha hb
      omega
    have hlt : ∀ j ∈ List.range k ++ [lenOff, lenOff + 1, lenOff + 2, lenOff + 3], j < lenOff + 4 := by
      intro j hj; simp at hj; omega
    refine (List.Pairwise.imp_of_mem (R := fun a b => a ≠ b) ?_ hnd)
    intro a b ha hb hab e
    apply hab
    have := congrArg UInt64.toNat e
    have ha' := hlt a ha
    have hb' := hlt b hb
    rw [UInt64.toNat_add, UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show a < 2 ^ 64 by omega),
      UInt64.toNat_ofNat_of_lt' (show b < 2 ^ 64 by omega), Nat.mod_eq_of_lt (by omega),
      Nat.mod_eq_of_lt (by omega)] at this
    omega

end Lz4Code
