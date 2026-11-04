module
public import Lz4.CodeCoop
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lz4.CodeWarp` — a warp's lanes run with the warp

Where each of a warp's 32 lanes runs a warp statement on its own
(`laneRun` with `coopL`), every lane ends agreeing with the warp's run of it
(`WStmt.eval`): its registers the warp's, its shared memory the warp's
table, every byte of the input region `I` the warp's (`Rel`), and the bytes
it stores its share of the warp's stores (`gstOf`, `lanePart`): every
store the warp makes uniformly, and of a copy the bytes `i` with
`i % 32` its lane. That needs of the warp's run only that no store reaches
`I` or shared memory and every load lies in `I` (`okOf`). The
cooperative statements are taken as given (`hcoop`), with what they need of
the lanes together: each lane's run a prefix of where it ends (`Ext`).
-/

namespace Lz4Code

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Prog
  AlgorithmLib.LZ4WarpDSL AlgorithmLib.LibSpec AlgorithmLib.Kernel

/-- A thread's byte stores since its `start`-th observation, in order. -/
def gStores (w : World) (start : Nat) : List (UInt64 × UInt64) :=
  (w.obs.reverse.drop start).filterMap fun
    | .store a 1 v => some (a, v)
    | _ => none

/-- A lane's share of the warp's stores: those every lane makes, and its own. -/
def lanePart (l : Nat) (tr : List (Option Nat × UInt64 × UInt64)) : List (UInt64 × UInt64) :=
  tr.filterMap fun (who, a, v) => if who = none ∨ who = some l then some (a, v) else none

/-- A copy's stores: byte `i` by lane `i % 32`. -/
def copyTr (g : Array UInt8) (d s : UInt64) (n : Nat) : List (Option Nat × UInt64 × UInt64) :=
  (List.range n).map fun i => (some (i % 32), d + UInt64.ofNat i, (g.getD (s.toNat + i) 0).toUInt64)

/-- **The warp's byte stores**, each with the lane that makes it, none where
    every lane does. -/
def gstOf : Nat → WStmt → WState → List (Option Nat × UInt64 × UInt64)
  | f, .seq a b, st => gstOf f a st ++ gstOf f b (a.eval f st)
  | _, .stgB addr val, st => [(none, st.regs addr, st.regs val &&& 0xff)]
  | f, .uif c t e, st => if st.regs c == 1 then gstOf f t st else gstOf f e st
  | f + 1, .uwhile c b, st =>
      if st.regs c == 1 then gstOf f b st ++ gstOf f (WStmt.uwhile c b) (b.eval f st) else []
  | _, .coopCopy d s n, st => copyTr st.gmem (st.regs d) (st.regs s) (st.regs n).toNat
  | _, _, _ => []

/-- An address outside shared memory. -/
def NotArena (a : UInt64) : Prop := ∀ off, decodeAddr a ≠ some (.arena, off)

/-- **What the lanes need of the warp's run**: a store neither into `I` nor
    shared memory and into `O`; a copy's bytes, and the 31 past them its lanes
    read too, from `I`, into neither and into `O`, both runs below `2^64` with
    room for the lanes' strides; a loop out of fuel stopped; a cooperative
    statement's own (`okC`). -/
def okOf (I O : UInt64 → Prop) (okC : WStmt → WState → Prop) : Nat → WStmt → WState → Prop
  | f, .seq a b, st => okOf I O okC f a st ∧ okOf I O okC f b (a.eval f st)
  | _, .stgB addr _, st => ¬ I (st.regs addr) ∧ NotArena (st.regs addr) ∧ O (st.regs addr)
  | f, .uif c t e, st => if st.regs c == 1 then okOf I O okC f t st else okOf I O okC f e st
  | 0, .uwhile c _, st => (st.regs c == 1) = false
  | f + 1, .uwhile c b, st =>
      st.regs c == 1 → okOf I O okC f b st ∧ okOf I O okC f (WStmt.uwhile c b) (b.eval f st)
  | _, .coopCopy d s n, st =>
      (st.regs n).toNat + 64 < 2 ^ 64 ∧ (st.regs s).toNat + (st.regs n).toNat < 2 ^ 64 ∧
      (st.regs d).toNat + (st.regs n).toNat < 2 ^ 64 ∧
      (∀ i, i < (st.regs n).toNat + 31 → I (st.regs s + UInt64.ofNat i)) ∧
      ∀ i, i < (st.regs n).toNat → ¬ I (st.regs d + UInt64.ofNat i) ∧
        NotArena (st.regs d + UInt64.ofNat i) ∧ O (st.regs d + UInt64.ofNat i)
  | _, s@(.coopWindow ..), st | _, s@(.coopExtendStep ..), st => okC s st
  | _, _, _ => True

/-- **A lane agrees with the warp**: its registers, its shared memory the
    warp's table, and every byte of `I` the warp's. -/
structure Rel (I : UInt64 → Prop) (ws : WState) (st : LSt) : Prop where
  regs : st.regs = ws.regs
  smem : st.w.mem.arena.data = ws.smem
  gin : ∀ a, I a → st.w.mem.load a 1 = some (ws.gmem.getD a.toNat 0).toUInt64

/-- A lane's run goes on from where `st` is: its observations extend
    `st`'s. -/
def Ext (st st' : LSt) : Prop := ∃ L, st'.w.obs = L ++ st.w.obs

theorem Ext.refl (st : LSt) : Ext st st := ⟨[], rfl⟩

theorem Ext.trans {a b c : LSt} (h1 : Ext a b) (h2 : Ext b c) : Ext a c := by
  obtain ⟨L1, h1⟩ := h1
  obtain ⟨L2, h2⟩ := h2
  exact ⟨L2 ++ L1, by rw [h2, h1, List.append_assoc]⟩

theorem gStores_ext {w w' : World} {start : Nat} (hs : start ≤ w.obs.length) {L : List Obs}
    (h : w'.obs = L ++ w.obs) : gStores w' start = gStores w start ++ L.reverse.filterMap (fun
      | .store a 1 v => some (a, v)
      | _ => none) := by
  unfold gStores
  rw [h, List.reverse_append, List.drop_append_of_le_length (by simpa using hs), List.filterMap_append]

/-! ## Memory a lane's stores leave -/

/-- Byte stores among observations, in order. -/
def stores1 (L : List Obs) : List (UInt64 × UInt64) :=
  L.filterMap fun
    | .store a 1 v => some (a, v)
    | _ => none

theorem decodeAddr_inj {a b : UInt64} {p : Region × Nat} (ha : decodeAddr a = some p)
    (hb : decodeAddr b = some p) : a = b := by
  unfold decodeAddr at ha hb
  obtain ⟨ra, hra, ea⟩ := Option.map_eq_some_iff.mp ha
  obtain ⟨rb, hrb, eb⟩ := Option.map_eq_some_iff.mp hb
  subst ea
  simp only [Prod.mk.injEq] at eb
  obtain ⟨rfl, hm⟩ := eb
  have h1 := Region.idx_of_ofIdx hra
  have h2 := Region.idx_of_ofIdx hrb
  have hdiv : (a / regionSpan).toNat = (b / regionSpan).toNat := h1.symm.trans h2
  have hsp : regionSpan.toNat = 2 ^ 36 := rfl
  apply UInt64.toNat_inj.mp
  have ha' := Nat.div_add_mod a.toNat (2 ^ 36)
  have hb' := Nat.div_add_mod b.toNat (2 ^ 36)
  simp only [UInt64.toNat_div, UInt64.toNat_mod, hsp] at hdiv hm
  omega

/-- A byte stored elsewhere leaves a byte's load. -/
theorem load_store_ne {m m' : Mem} {a b v : UInt64} {x : UInt64} (h : m.store a 1 v = some m') (hne : a ≠ b)
    (hl : m.load b 1 = some x) : m'.load b 1 = some x := by
  obtain ⟨r, off, hd, -⟩ := store_frame h
  have hdb : ∃ r' off', decodeAddr b = some (r', off') := by
    unfold Mem.load at hl
    cases e : decodeAddr b with
    | none => simp [e] at hl
    | some p => exact ⟨p.1, p.2, rfl⟩
  obtain ⟨r', off', hdb⟩ := hdb
  rw [load_store_clear h hd hdb ?_, hl]
  by_cases hr : r' = r
  · subst hr
    by_cases ho : off' = off
    · subst ho; exact absurd (decodeAddr_inj hd hdb) hne
    · right; omega
  · left; exact hr

/-- A store outside shared memory leaves it. -/
theorem store_arena {m m' : Mem} {a v : UInt64} {n : Nat} (h : m.store a n v = some m') (hn : NotArena a) :
    m'.arena = m.arena := by
  obtain ⟨r, off, hd, hF⟩ := store_frame h
  have hr : Region.arena ≠ r := fun e => hn off (e ▸ hd)
  exact hF.other .arena hr

theorem stw_some {w w' : World} {a : UInt64} {n : Nat} {v : UInt64} (h : stw w a n v = some w') :
    ∃ m, w.mem.store a n v = some m ∧ w' = { obsStore w a n v with mem := m } := by
  simp only [stw] at h
  obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
  exact ⟨m, hm, (Option.some.inj h).symm⟩

theorem byte_mask (b : UInt8) : b.toUInt64 &&& 0xff = b.toUInt64 := by
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_and, show (0xff : UInt64).toNat = 2 ^ 8 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod]
  simp

/-! ## A lane's share of a copy -/

/-- The bytes of a block of 32 one lane copies: none, or its own. -/
theorem filterMap_block {α : Type} (l m : Nat) (hl : l < 32) (e : Nat → α) :
    ∀ c, c ≤ 32 → (List.range' (32 * m) c).filterMap (fun j => if j % 32 = l then some (e j) else none) =
      if l < c then [e (32 * m + l)] else []
  | 0, _ => by simp
  | c + 1, hc => by
      rw [List.range'_concat, List.filterMap_append, filterMap_block l m hl e c (by omega)]
      have hmod : (32 * m + c) % 32 = c := by omega
      simp only [List.filterMap_cons, List.filterMap_nil, Nat.one_mul]
      rw [hmod]
      by_cases h2 : c = l
      · rw [if_pos h2]; subst h2; simp
      · rw [if_neg h2]
        by_cases h1 : l < c
        · simp [h1, show l < c + 1 by omega]
        · simp [h1, show ¬ l < c + 1 by omega]

/-- A lane's bytes of a copy of `n`, block by block. -/
theorem filterMap_blocks {α : Type} (n l : Nat) (hl : l < 32) (e : Nat → α) (m : Nat) :
    (List.range' (32 * m) (n - 32 * m)).filterMap (fun j => if j % 32 = l then some (e j) else none) =
      (if 32 * m + l < n then [e (32 * m + l)] else []) ++
        (List.range' (32 * (m + 1)) (n - 32 * (m + 1))).filterMap (fun j => if j % 32 = l then some (e j) else none) := by
  by_cases hn : n ≤ 32 * m
  · simp [show n - 32 * m = 0 by omega, show n - 32 * (m + 1) = 0 by omega, show ¬ 32 * m + l < n by omega]
  · by_cases h2 : 32 * (m + 1) ≤ n
    · have hs : n - 32 * m = 32 + (n - 32 * (m + 1)) := by omega
      rw [hs, ← List.range'_append, List.filterMap_append, filterMap_block l m hl e 32 (Nat.le_refl _)]
      simp [show 32 * m + 32 = 32 * (m + 1) by omega, hl, show 32 * m + l < n by omega]
    · rw [show n - 32 * (m + 1) = 0 by omega, filterMap_block l m hl e _ (by omega)]
      by_cases h3 : 32 * m + l < n
      · simp [h3, show l < n - 32 * m by omega]
      · simp [h3, show ¬ l < n - 32 * m by omega]

theorem ofNat_lt_iff {a b : Nat} (ha : a < 2 ^ 64) (hb : b < 2 ^ 64) : UInt64.ofNat a < UInt64.ofNat b ↔ a < b := by
  rw [UInt64.lt_iff_toNat_lt, UInt64.toNat_ofNat_of_lt' ha, UInt64.toNat_ofNat_of_lt' hb]

/-- **A lane's share of a copy**: from block `m` on, the bytes `j` of the
    copy with `j % 32` its lane, in order, each the byte of `I` it copies;
    shared memory and every byte of `I` as they were. -/
theorem copyL_lane {I : UInt64 → Prop} {g : Array UInt8} {d s n : UInt64} {l : Nat} (hl : l < 32)
    (hn : n.toNat + 64 < 2 ^ 64) (hs : s.toNat + n.toNat < 2 ^ 64)
    (hI : ∀ i, i < n.toNat → I (s + UInt64.ofNat i) ∧ ¬ I (d + UInt64.ofNat i) ∧ NotArena (d + UInt64.ofNat i)) :
    ∀ k m (w w' : World), copyL d s n (UInt64.ofNat l) k (UInt64.ofNat (32 * m)) w = some w' →
      32 * m ≤ n.toNat + 32 → (∀ a, I a → w.mem.load a 1 = some (g.getD a.toNat 0).toUInt64) →
      (∃ L, w'.obs = L ++ w.obs ∧ stores1 L.reverse = (List.range' (32 * m) (n.toNat - 32 * m)).filterMap
          (fun j => if j % 32 = l then some (d + UInt64.ofNat j, (g.getD (s.toNat + j) 0).toUInt64) else none)) ∧
      w'.mem.arena = w.mem.arena ∧ ∀ a, I a → w'.mem.load a 1 = w.mem.load a 1 := by
  intro k
  induction k with
  | zero =>
      intro m w w' h hm hg
      simp only [copyL] at h
      split at h
      · cases h
      rename_i hlt
      obtain rfl := (Option.some.inj h).symm
      have : n.toNat ≤ 32 * m := by
        rw [← UInt64.toNat_ofNat_of_lt' (show 32 * m < 2 ^ 64 by omega)]
        exact Nat.not_lt.mp fun h => hlt (UInt64.lt_iff_toNat_lt.mpr h)
      exact ⟨⟨[], rfl, by simp [stores1, show n.toNat - 32 * m = 0 by omega]⟩, rfl, fun _ _ => rfl⟩
  | succ k ih =>
      intro m w w' h hm hg
      simp only [copyL] at h
      by_cases hlt : UInt64.ofNat (32 * m) < n
      case neg =>
        rw [if_neg hlt] at h
        obtain rfl := (Option.some.inj h).symm
        have : n.toNat ≤ 32 * m := by
          rw [← UInt64.toNat_ofNat_of_lt' (show 32 * m < 2 ^ 64 by omega)]
          exact Nat.not_lt.mp fun h => hlt (UInt64.lt_iff_toNat_lt.mpr h)
        exact ⟨⟨[], rfl, by simp [stores1, show n.toNat - 32 * m = 0 by omega]⟩, rfl, fun _ _ => rfl⟩
      rw [if_pos hlt] at h
      have hmn : 32 * m < n.toNat := by
        have := UInt64.lt_iff_toNat_lt.mp hlt
        rwa [UInt64.toNat_ofNat_of_lt' (show 32 * m < 2 ^ 64 by omega)] at this
      obtain ⟨w1, hw1, hrest⟩ := Option.bind_eq_some_iff.mp h
      have hnext : UInt64.ofNat (32 * m) + c64 32 = UInt64.ofNat (32 * (m + 1)) := by
        rw [show c64 32 = UInt64.ofNat 32 by decide, ← UInt64.ofNat_add]; congr 1
      rw [hnext] at hrest
      have hj : UInt64.ofNat (32 * m) + UInt64.ofNat l = UInt64.ofNat (32 * m + l) := by
        rw [← UInt64.ofNat_add]
      -- the trip's own byte
      have step : ∃ L1, w1.obs = L1 ++ w.obs ∧
          stores1 L1.reverse = (if 32 * m + l < n.toNat then
            [(d + UInt64.ofNat (32 * m + l), (g.getD (s.toNat + (32 * m + l)) 0).toUInt64)] else []) ∧
          w1.mem.arena = w.mem.arena ∧ ∀ a, I a → w1.mem.load a 1 = w.mem.load a 1 := by
        simp only [copyStep, hj] at hw1
        split at hw1
        · rename_i hjn
          have hjn' : 32 * m + l < n.toNat := by
            have := UInt64.lt_iff_toNat_lt.mp hjn
            rwa [UInt64.toNat_ofNat_of_lt' (show 32 * m + l < 2 ^ 64 by omega)] at this
          obtain ⟨hIs, hId, hna⟩ := hI _ hjn'
          obtain ⟨v, hv, hw1⟩ := Option.bind_eq_some_iff.mp hw1
          rw [hg _ hIs] at hv
          obtain rfl := (Option.some.inj hv).symm
          have hidx : (s + UInt64.ofNat (32 * m + l)).toNat = s.toNat + (32 * m + l) := by
            rw [UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show 32 * m + l < 2 ^ 64 by omega)]
            exact Nat.mod_eq_of_lt (by omega)
          obtain ⟨mm, hmm, rfl⟩ := stw_some hw1
          refine ⟨[.store (d + UInt64.ofNat (32 * m + l)) 1 _], rfl, ?_, store_arena hmm hna, fun a ha => ?_⟩
          · simp [stores1, hjn', byte_mask, Nat.mod_eq_of_lt (show s.toNat + (32 * m + l) < 18446744073709551616 by omega)]
          · cases hla : w.mem.load a 1 with
            | none => rw [hg a ha] at hla; cases hla
            | some x =>
                exact load_store_ne hmm (fun e => hId (e ▸ ha)) hla
        · rename_i hjn
          have hjn' : ¬ 32 * m + l < n.toNat := fun h => hjn (by
            rw [UInt64.lt_iff_toNat_lt, UInt64.toNat_ofNat_of_lt' (show 32 * m + l < 2 ^ 64 by omega)]; exact h)
          obtain rfl := bind_const_some hw1
          exact ⟨[], rfl, by simp [stores1, hjn'], rfl, fun _ _ => rfl⟩
      obtain ⟨L1, hL1, hs1, ha1, hI1⟩ := step
      obtain ⟨⟨L2, hL2, hs2⟩, ha2, hI2⟩ := ih (m + 1) w1 w' hrest (by omega)
        (fun a ha => by rw [hI1 a ha]; exact hg a ha)
      refine ⟨⟨L2 ++ L1, by rw [hL2, hL1, List.append_assoc], ?_⟩, ha2.trans ha1,
        fun a ha => (hI2 a ha).trans (hI1 a ha)⟩
      rw [filterMap_blocks _ l hl _ m, ← hs1, ← hs2]
      simp [stores1, List.filterMap_append]

theorem load_set_arena {m : Mem} {b : ByteArray} {a : UInt64} {k : Nat} (hn : NotArena a) :
    ({ m with arena := b } : Mem).load a k = m.load a k := by
  unfold Mem.load
  cases hd : decodeAddr a with
  | none => rfl
  | some p =>
      obtain ⟨r, off⟩ := p
      have hr : r ≠ .arena := fun e => hn off (e ▸ hd)
      cases r <;> first | exact absurd rfl hr | rfl

theorem store_setArena {m m' : Mem} {a v : UInt64} {n off : Nat} (hd : decodeAddr a = some (.arena, off))
    (h : m.store a n v = some m') : m' = { m with arena := m'.arena } := by
  obtain ⟨r, off', hd', -, rfl⟩ := store_some h
  rw [hd] at hd'; obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hd')
  rfl

/-! ## Runs only add observations -/

/-- A world goes on from `w`: its observations extend `w`'s. -/
def WExt (w w' : World) : Prop := ∃ L, w'.obs = L ++ w.obs

theorem WExt.refl (w : World) : WExt w w := ⟨[], rfl⟩

theorem WExt.trans {a b c : World} (h1 : WExt a b) (h2 : WExt b c) : WExt a c := by
  obtain ⟨L1, h1⟩ := h1
  obtain ⟨L2, h2⟩ := h2
  exact ⟨L2 ++ L1, by rw [h2, h1, List.append_assoc]⟩

theorem stw_ext {w w' : World} {a : UInt64} {n : Nat} {v : UInt64} (h : stw w a n v = some w') : WExt w w' := by
  obtain ⟨m, -, rfl⟩ := stw_some h
  exact ⟨[.store a n v], rfl⟩

theorem call_ext (w : World) (c : Intr) (args : List V) : WExt w (callW w c args) :=
  ⟨[_], rfl⟩

theorem copyL_ext {d s n lv : UInt64} : ∀ k i (w w' : World), copyL d s n lv k i w = some w' → WExt w w'
  | 0, i, w, w', h => by
      simp only [copyL] at h
      split at h
      · cases h
      · obtain rfl := (Option.some.inj h).symm; exact WExt.refl _
  | k + 1, i, w, w', h => by
      simp only [copyL] at h
      split at h
      · obtain ⟨w1, h1, h2⟩ := Option.bind_eq_some_iff.mp h
        refine WExt.trans ?_ (copyL_ext k _ w1 w' h2)
        simp only [copyStep] at h1
        split at h1
        · obtain ⟨v, -, h1⟩ := Option.bind_eq_some_iff.mp h1
          exact stw_ext h1
        · obtain rfl := bind_const_some h1; exact WExt.refl _
      · obtain rfl := (Option.some.inj h).symm; exact WExt.refl _

/-- A world goes on from `w` with no byte store. -/
def NoByte (w w' : World) : Prop := ∃ L, w'.obs = L ++ w.obs ∧ stores1 L = []

theorem NoByte.refl (w : World) : NoByte w w := ⟨[], rfl, rfl⟩

theorem NoByte.trans {a b c : World} (h1 : NoByte a b) (h2 : NoByte b c) : NoByte a c := by
  obtain ⟨L1, h1, s1⟩ := h1
  obtain ⟨L2, h2, s2⟩ := h2
  exact ⟨L2 ++ L1, by rw [h2, h1, List.append_assoc], by simp [stores1] at s1 s2 ⊢; exact ⟨s2, s1⟩⟩

theorem NoByte.ext {a b : World} (h : NoByte a b) : WExt a b := by
  obtain ⟨L, h, -⟩ := h; exact ⟨L, h⟩

theorem NoByte.call (w : World) (c : Intr) (args : List V) : NoByte w (callW w c args) := ⟨[_], rfl, rfl⟩

theorem NoByte.arena (w : World) (b : ByteArray) : NoByte w { w with mem := { w.mem with arena := b } } :=
  ⟨[], rfl, rfl⟩

theorem stwIf_nobyte {c : Bool} {w w' : World} {a v : UInt64} (h : stwIf c w a 2 v = some w') : NoByte w w' := by
  unfold stwIf at h
  split at h
  · obtain ⟨m, -, rfl⟩ := stw_some h
    exact ⟨[.store a 2 v], rfl, rfl⟩
  · obtain rfl := (Option.some.inj h).symm
    exact NoByte.refl _

/-- **A cooperative statement adds observations**, and of byte stores only
    a copy's. -/
theorem coopL_ext {ans : List Answer} {start : Nat} {smv lv : UInt64} :
    ∀ (s : WStmt) (st st' : LSt), coopL ans start smv lv s st = some st' →
      WExt st.w st'.w ∧ ((s matches .coopWindow .. | .coopExtendStep ..) → NoByte st.w st'.w)
  | .coopCopy dst src len, st, st', h => by
      simp only [coopL] at h
      obtain ⟨w, hw, h⟩ := Option.bind_eq_some_iff.mp h
      obtain rfl := (Option.some.inj h).symm
      exact ⟨copyL_ext _ _ _ _ hw, fun h => by cases h⟩
  | .coopWindow found p0 cand0 sp inStride searchLim hashLog tbl, st, st', h => by
      simp only [coopL] at h
      have hs : start ≤ st.w.obs.length := Classical.byContradiction fun hn => by simp [hn] at h
      simp only [hs, if_true] at h
      obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨w2, hw2, h⟩ := Option.bind_eq_some_iff.mp h
      split at h
      · obtain rfl := (Option.some.inj h).symm
        refine (fun hk => ⟨NoByte.ext hk, fun _ => hk⟩) ?_
        refine ((NoByte.call _ _ _).trans (stwIf_nobyte hw2)).trans ?_
        exact ((NoByte.call _ _ _).trans (NoByte.arena _ _)).trans (NoByte.call _ _ _)
      · cases h
  | .coopExtendStep adv p0 cand0 ml inStride endCap, st, st', h => by
      simp only [coopL] at h
      have hs : start ≤ st.w.obs.length := Classical.byContradiction fun hn => by simp [hn] at h
      simp only [hs, if_true] at h
      obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
      obtain rfl := (Option.some.inj h).symm
      exact ⟨call_ext _ _ _, fun _ => NoByte.call _ _ _⟩
  | .skip, _, _, h | .seq .., _, _, h | .bin .., _, _, h | .setp .., _, _, h | .mov .., _, _, h
  | .stgB .., _, _, h | .uif .., _, _, h | .uwhile .., _, _, h => by simp [coopL] at h

theorem laneRun_ext {ans : List Answer} {start : Nat} {smv lv : UInt64} :
    ∀ (s : WStmt) (f : Nat) (st st' : LSt), laneRun (coopL ans start smv lv) smv f s st = some st' →
      WExt st.w st'.w := by
  intro s
  induction s with
  | skip => intro f st st' h; simp [laneRun] at h; subst h; exact WExt.refl _
  | seq a b iha ihb =>
      intro f st st' h
      simp only [laneRun] at h
      obtain ⟨m, h1, h2⟩ := Option.bind_eq_some_iff.mp h
      exact (iha f st m h1).trans (ihb f m st' h2)
  | bin o d a b | setp c d a b | mov d a => intro f st st' h; simp [laneRun] at h; subst h; exact WExt.refl _
  | stgB addr val =>
      intro f st st' h
      simp only [laneRun] at h
      obtain ⟨w, hw, h⟩ := Option.bind_eq_some_iff.mp h
      obtain rfl := (Option.some.inj h).symm
      exact stw_ext hw
  | uif c t e iht ihe =>
      intro f st st' h
      simp only [laneRun] at h
      split at h
      · exact iht f st st' h
      · exact ihe f st st' h
  | uwhile c body ih =>
      intro f
      induction f with
      | zero =>
          intro st st' h
          simp only [laneRun] at h
          split at h
          · cases h
          · obtain rfl := (Option.some.inj h).symm; exact WExt.refl _
      | succ f ihf =>
          intro st st' h
          simp only [laneRun] at h
          split at h
          · obtain ⟨m, h1, h2⟩ := Option.bind_eq_some_iff.mp h
            exact (ih f st m h1).trans (ihf m st' h2)
          · obtain rfl := (Option.some.inj h).symm; exact WExt.refl _
  | coopCopy dst src len =>
      intro f st st' h
      exact (coopL_ext _ st st' (by simpa [laneRun] using h)).1
  | coopWindow found p0 cand0 sp inStride searchLim hashLog tbl =>
      intro f st st' h
      exact (coopL_ext _ st st' (by simpa [laneRun] using h)).1
  | coopExtendStep adv p0 cand0 ml inStride endCap =>
      intro f st st' h
      exact (coopL_ext _ st st' (by simpa [laneRun] using h)).1

theorem gStores_nobyte {w w' : World} {start : Nat} (hs : start ≤ w.obs.length) (h : NoByte w w') :
    gStores w' start = gStores w start := by
  obtain ⟨L, hL, h0⟩ := h
  rw [gStores_ext hs hL]
  have : stores1 L.reverse = [] := by
    simp only [stores1, List.filterMap_eq_nil_iff] at h0 ⊢
    intro x hx; exact h0 x (List.mem_reverse.mp hx)
  simpa [stores1] using this

/-- A load of bytes inside a region that is not pinned memory reads them. -/
theorem load_some {m : Mem} {a : UInt64} {n : Nat} {r : Region} {off : Nat} (hd : decodeAddr a = some (r, off))
    (hsz : off + n ≤ (m.region r).size) (hr : r ≠ .pinned) : ∃ v, m.load a n = some v := by
  simp only [Mem.load, hd, Option.bind_eq_bind, Option.bind_some]
  have hrd : m.readable r off n = true := by simp [Mem.readable, hr]
  rw [if_neg (by simp [hrd]; omega)]
  exact ⟨_, rfl⟩

/-- A store of bytes inside a region that is not pinned memory is made. -/
theorem store_some_of {m : Mem} {a : UInt64} {n : Nat} {v : UInt64} {r : Region} {off : Nat}
    (hd : decodeAddr a = some (r, off)) (hsz : off + n ≤ (m.region r).size) (hr : r ≠ .pinned) :
    ∃ m', m.store a n v = some m' := by
  simp only [Mem.store, hd, Option.bind_eq_bind, Option.bind_some]
  have hrc : m.reachable r off n = true := by simp [Mem.reachable, hr]
  rw [if_neg (by simp [hrc]; omega)]
  exact ⟨_, rfl⟩

/-- An offset below a region's span in shared memory is shared memory there. -/
theorem arena_decode {a : UInt64} (ha : a.toNat < 2 ^ 36) :
    decodeAddr (addrOf .arena 0 + a) = some (.arena, a.toNat) := by
  rw [show addrOf .arena 0 + a = regionBase .arena + UInt64.ofNat a.toNat by simp [addrOf]]
  exact decodeAddr_base _ _ (by rw [regionSpan_toNat]; exact ha)

/-- A run that goes on from a step that is made is made where the rest is. -/
theorem bind_some_of {α β : Type} {o : Option α} {a : α} {f : α → Option β} (h : o = some a)
    (hf : ∃ r, f a = some r) : ∃ r, o.bind f = some r := by
  rw [h]; exact hf

/-- A store into shared memory made where a condition holds is made, and
    leaves the collectives as they were. -/
theorem stwIf_bind_some {β : Type} {c : Bool} {w : World} {a v : UInt64} {n : Nat} {k : World → Option β}
    (ha : a.toNat < 2 ^ 36) (hsz : a.toNat + n ≤ w.mem.arena.size)
    (hk : ∀ w', (∀ s, s ≤ w.obs.length → collCalls w' s = collCalls w s) → w.obs.length ≤ w'.obs.length →
      ∃ r, k w' = some r) :
    ∃ r, (stwIf c w (addrOf .arena 0 + a) n v).bind k = some r := by
  cases c with
  | false => exact hk w (fun _ _ => rfl) (Nat.le_refl _)
  | true =>
      obtain ⟨m', hm'⟩ := store_some_of (v := v) (arena_decode ha) (by simpa [Mem.region] using hsz) (by decide)
      simp only [stwIf, if_true, stw, hm', Option.bind_eq_bind, Option.bind_some]
      exact hk _ (fun s hs => collCalls_obsStore hs _ _ _) (by simp [obsStore])

theorem WExt.len {w w' : World} (h : WExt w w') : w.obs.length ≤ w'.obs.length := by
  obtain ⟨L, h⟩ := h; rw [h]; simp

/-- A byte at `a` may be stored: it lies inside a region other than pinned
    memory. -/
def Writable (m : Mem) (a : UInt64) : Prop :=
  ∃ r off, decodeAddr a = some (r, off) ∧ off + 1 ≤ (m.region r).size ∧ r ≠ .pinned

theorem Writable.store_some {m : Mem} {a : UInt64} (h : Writable m a) (v : UInt64) :
    ∃ m', m.store a 1 v = some m' := by
  obtain ⟨r, off, hd, hsz, hr⟩ := h
  exact store_some_of hd hsz hr

/-- A store leaves every byte that could be stored storable. -/
theorem Writable.after_store {m m' : Mem} {a b : UInt64} {n : Nat} {v : UInt64} (h : m.store b n v = some m')
    (hw : Writable m a) : Writable m' a := by
  obtain ⟨r, off, hd, hsz, hr⟩ := hw
  obtain ⟨r', off', -, hf⟩ := store_frame h
  exact ⟨r, off, hd, by rw [← hf.same.size r]; exact hsz, hr⟩

/-- Replacing shared memory leaves every other byte that could be stored
    storable. -/
theorem Writable.arena {m : Mem} {a : UInt64} (hw : Writable m a) (ha : NotArena a) (b : ByteArray) :
    Writable { m with arena := b } a := by
  obtain ⟨r, off, hd, hsz, hr⟩ := hw
  refine ⟨r, off, hd, ?_, hr⟩
  cases r with
  | arena => exact absurd hd (ha off)
  | _ => exact hsz

theorem Writable.of_stw {w w' : World} {a b : UInt64} {n : Nat} {v : UInt64} (h : stw w b n v = some w')
    (hw : Writable w.mem a) : Writable w'.mem a := by
  obtain ⟨m, hm, rfl⟩ := stw_some h
  exact hw.after_store hm

theorem Writable.of_stwIf {c : Bool} {w w' : World} {a b : UInt64} {n : Nat} {v : UInt64}
    (h : stwIf c w b n v = some w') (hw : Writable w.mem a) : Writable w'.mem a := by
  unfold stwIf at h
  split at h
  · exact hw.of_stw h
  · cases h; exact hw

/-- A store leaves every byte that could be loaded loadable. -/
theorem load_after_store {m m' : Mem} {a b : UInt64} {n : Nat} {v : UInt64} (h : m.store b n v = some m')
    (hl : ∃ x, m.load a 1 = some x) : ∃ x, m'.load a 1 = some x := by
  obtain ⟨r', off', -, hf⟩ := store_frame h
  have := hf.same.load_isSome a 1
  obtain ⟨x, hx⟩ := hl
  rw [hx] at this
  exact Option.isSome_iff_exists.mp this.symm

/-- **A lane's share of a copy finishes** where its bytes load and may be
    stored, with fuel for every trip. -/
theorem copyL_some {I : UInt64 → Prop} {d s n : UInt64} {l : Nat} (hl : l < 32) (hn : n.toNat + 64 < 2 ^ 64)
    (hI : ∀ i, i < n.toNat + 31 → I (s + UInt64.ofNat i)) :
    ∀ k m (w : World), n.toNat ≤ 32 * m + 32 * k → 32 * m ≤ n.toNat + 32 →
      (∀ a, I a → ∃ v, w.mem.load a 1 = some v) → (∀ i, i < n.toNat → Writable w.mem (d + UInt64.ofNat i)) →
      ∃ w', copyL d s n (UInt64.ofNat l) k (UInt64.ofNat (32 * m)) w = some w' := by
  intro k
  induction k with
  | zero =>
      intro m w hk _ _ _
      have : ¬ UInt64.ofNat (32 * m) < n := by
        rw [UInt64.lt_iff_toNat_lt, UInt64.toNat_ofNat_of_lt' (show 32 * m < 2 ^ 64 by omega)]; omega
      exact ⟨w, by simp only [copyL, if_neg this]⟩
  | succ k ih =>
      intro m w hk hm hg hwr
      by_cases hlt : UInt64.ofNat (32 * m) < n
      case neg => exact ⟨w, by simp only [copyL, if_neg hlt]⟩
      have hmn : 32 * m < n.toNat := by
        have := UInt64.lt_iff_toNat_lt.mp hlt
        rwa [UInt64.toNat_ofNat_of_lt' (show 32 * m < 2 ^ 64 by omega)] at this
      have hnext : UInt64.ofNat (32 * m) + c64 32 = UInt64.ofNat (32 * (m + 1)) := by
        rw [show c64 32 = UInt64.ofNat 32 by decide, ← UInt64.ofNat_add]; congr 1
      have hj : UInt64.ofNat (32 * m) + UInt64.ofNat l = UInt64.ofNat (32 * m + l) := by
        rw [← UInt64.ofNat_add]
      simp only [copyL, hlt, if_true, hnext, Option.bind_eq_bind]
      have hstep : ∃ w1, copyStep d s n (UInt64.ofNat l) (UInt64.ofNat (32 * m)) w = some w1 ∧
          (∀ a, I a → ∃ v, w1.mem.load a 1 = some v) ∧
          (∀ i, i < n.toNat → Writable w1.mem (d + UInt64.ofNat i)) := by
        simp only [copyStep, hj]
        by_cases hjn : UInt64.ofNat (32 * m + l) < n
        · have hjn' : 32 * m + l < n.toNat := by
            have := UInt64.lt_iff_toNat_lt.mp hjn
            rwa [UInt64.toNat_ofNat_of_lt' (show 32 * m + l < 2 ^ 64 by omega)] at this
          obtain ⟨v, hv⟩ := hg _ (hI (32 * m + l) (by omega))
          obtain ⟨m', hm'⟩ := (hwr _ hjn').store_some (v &&& 0xff)
          refine ⟨{ obsStore w (d + UInt64.ofNat (32 * m + l)) 1 (v &&& 0xff) with mem := m' }, ?_,
            fun a ha => load_after_store hm' (hg a ha), fun i hi => (hwr i hi).after_store hm'⟩
          simp only [if_pos hjn, hv, stw, hm', Option.bind_eq_bind, Option.bind_some]
        · obtain ⟨v, hv⟩ := hg _ (hI (32 * m + l) (by omega))
          exact ⟨w, by simp only [if_neg hjn, hv, Option.bind_eq_bind, Option.bind_some], hg, hwr⟩
      obtain ⟨w1, hw1, hg1, hwr1⟩ := hstep
      exact bind_some_of hw1 (ih (m + 1) w1 (by omega) (by omega) hg1 hwr1)

theorem copyL_writable {d s n lv : UInt64} {a : UInt64} :
    ∀ k i (w w' : World), copyL d s n lv k i w = some w' → Writable w.mem a → Writable w'.mem a
  | 0, i, w, w', h, hw => by
      simp only [copyL] at h
      split at h
      · cases h
      · cases h; exact hw
  | k + 1, i, w, w', h, hw => by
      simp only [copyL] at h
      split at h
      · obtain ⟨w1, hw1, h⟩ := Option.bind_eq_some_iff.mp h
        refine copyL_writable k _ w1 w' h ?_
        simp only [copyStep] at hw1
        split at hw1
        · obtain ⟨_, -, hw1⟩ := Option.bind_eq_some_iff.mp hw1
          exact hw.of_stw hw1
        · obtain rfl := bind_const_some hw1; exact hw
      · cases h; exact hw

/-- **A lane's share keeps every byte outside shared memory that could be
    stored storable.** -/
theorem coopL_writable {ans : List Answer} {start : Nat} {smv lv : UInt64} {a : UInt64} (ha : NotArena a) :
    ∀ (s : WStmt) (st st' : LSt), coopL ans start smv lv s st = some st' → Writable st.w.mem a →
      Writable st'.w.mem a
  | .coopCopy dst src len, st, st', h, hw => by
      simp only [coopL] at h
      obtain ⟨w, hw', h⟩ := Option.bind_eq_some_iff.mp h
      obtain rfl := (Option.some.inj h).symm
      exact copyL_writable _ _ _ _ hw' hw
  | .coopWindow found p0 cand0 sp inStride searchLim hashLog tbl, st, st', h, hw => by
      simp only [coopL] at h
      have hs : start ≤ st.w.obs.length := Classical.byContradiction fun hn => by simp [hn] at h
      simp only [hs, if_true] at h
      obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨w2, hw2, h⟩ := Option.bind_eq_some_iff.mp h
      split at h
      · obtain rfl := (Option.some.inj h).symm
        exact (hw.of_stwIf (w := callW st.w _ _) hw2).arena ha _
      · cases h
  | .coopExtendStep adv p0 cand0 ml inStride endCap, st, st', h, hw => by
      simp only [coopL] at h
      have hs : start ≤ st.w.obs.length := Classical.byContradiction fun hn => by simp [hn] at h
      simp only [hs, if_true] at h
      obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.mp h
      obtain rfl := (Option.some.inj h).symm
      exact hw
  | .skip, _, _, h, _ | .seq .., _, _, h, _ | .bin .., _, _, h, _ | .setp .., _, _, h, _ | .mov .., _, _, h, _
  | .stgB .., _, _, h, _ | .uif .., _, _, h, _ | .uwhile .., _, _, h, _ => by simp [coopL] at h

/-- **A lane's run keeps every byte outside shared memory that could be
    stored storable.** -/
theorem laneRun_writable {ans : List Answer} {start : Nat} {smv lv : UInt64} {a : UInt64} (ha : NotArena a) :
    ∀ (s : WStmt) (f : Nat) (st st' : LSt), laneRun (coopL ans start smv lv) smv f s st = some st' →
      Writable st.w.mem a → Writable st'.w.mem a := by
  intro s
  induction s with
  | skip => intro f st st' h hw; simp [laneRun] at h; subst h; exact hw
  | seq a b iha ihb =>
      intro f st st' h hw
      simp only [laneRun] at h
      obtain ⟨m, h1, h2⟩ := Option.bind_eq_some_iff.mp h
      exact ihb f m st' h2 (iha f st m h1 hw)
  | bin o d a b | setp c d a b | mov d a => intro f st st' h hw; simp [laneRun] at h; subst h; exact hw
  | stgB addr val =>
      intro f st st' h hw
      simp only [laneRun] at h
      obtain ⟨w, hw', h⟩ := Option.bind_eq_some_iff.mp h
      obtain rfl := (Option.some.inj h).symm
      exact hw.of_stw hw'
  | uif c t e iht ihe =>
      intro f st st' h hw
      simp only [laneRun] at h
      split at h
      · exact iht f st st' h hw
      · exact ihe f st st' h hw
  | uwhile c body ih =>
      intro f
      induction f with
      | zero =>
          intro st st' h hw
          simp only [laneRun] at h
          split at h
          · cases h
          · obtain rfl := (Option.some.inj h).symm; exact hw
      | succ f ihf =>
          intro st st' h hw
          simp only [laneRun] at h
          split at h
          · obtain ⟨m, h1, h2⟩ := Option.bind_eq_some_iff.mp h
            exact ihf m st' h2 (ih f st m h1 hw)
          · obtain rfl := (Option.some.inj h).symm; exact hw
  | coopCopy dst src len =>
      intro f st st' h hw
      exact coopL_writable ha _ st st' (by simpa [laneRun] using h) hw
  | coopWindow found p0 cand0 sp inStride searchLim hashLog tbl =>
      intro f st st' h hw
      exact coopL_writable ha _ st st' (by simpa [laneRun] using h) hw
  | coopExtendStep adv p0 cand0 ml inStride endCap =>
      intro f st st' h hw
      exact coopL_writable ha _ st st' (by simpa [laneRun] using h) hw

theorem readable_of_reachable {m : Mem} {r : Region} {off n : Nat} (h : m.reachable r off n = true) :
    m.readable r off n = true := by
  simp only [Mem.reachable, Mem.readable, Bool.or_eq_true, Bool.and_eq_true, List.all_eq_true] at h ⊢
  rcases h with h | ⟨h1, h2⟩
  · exact .inl h
  · refine .inr ⟨h1, fun b hb => ?_⟩
    have := h2 b hb
    simp [this]

/-- **A byte stored loads back.** -/
theorem load_store_same1 {m m' : Mem} {a v : UInt64} (h : m.store a 1 v = some m') :
    m'.load a 1 = some (v &&& 0xff) := by
  unfold Mem.store at h
  cases hd : decodeAddr a with
  | none => simp [hd] at h
  | some p =>
    obtain ⟨r, off⟩ := p
    simp only [hd, Option.bind_eq_bind, Option.bind_some] at h
    split at h
    · cases h
    · rename_i hc
      simp only [Bool.or_eq_true, decide_eq_true_eq, Bool.not_eq_true', not_or, Bool.not_eq_false] at hc
      obtain ⟨hsz, hre⟩ := hc
      injection h with h; subst h
      unfold Mem.load
      simp only [hd, Option.bind_eq_bind, Option.bind_some, Sem.region_setRegion_self, Mem.readable_setRegion]
      simp only [show List.range 1 = [0] from rfl, List.foldl_cons, List.foldl_nil, List.foldr_cons, List.foldr_nil,
        Nat.add_zero, Sem.set!_size]
      rw [if_neg (by simp [readable_of_reachable hre]; omega), Sem.get!_set!_same _ _ _ (by omega)]
      congr 1
      apply UInt64.toNat_inj.mp
      have h255 : (v &&& 255).toNat < 256 := by
        rw [UInt64.toNat_and]; exact Nat.lt_of_le_of_lt Nat.and_le_right (by decide)
      simp only [show (8 : UInt64) * UInt64.ofNat 0 = 0 by decide, UInt64.shiftRight_zero, UInt64.zero_shiftLeft,
        UInt64.zero_or, UInt8.toNat_toUInt64, UInt64.toNat_toUInt8, UInt64.toNat_toUInt8]
      omega

/-- A byte `a` after byte stores `L` from `d`: the last one's byte. -/
def replay (L : List (UInt64 × UInt64)) (a : UInt64) (d : Option UInt64) : Option UInt64 :=
  L.foldl (fun acc x => if x.1 = a then some (x.2 &&& 0xff) else acc) d

theorem replay_append (L M : List (UInt64 × UInt64)) (a : UInt64) (d : Option UInt64) :
    replay (L ++ M) a d = replay M a (replay L a d) := by
  simp [replay, List.foldl_append]

/-- **`w'` goes on from `w` by its byte stores**: its observations extend
    `w`'s, and every byte outside shared memory is `w`'s with the new byte
    stores made. -/
def Tracks (w w' : World) : Prop :=
  ∃ L, w'.obs = L ++ w.obs ∧ ∀ a, NotArena a → w'.mem.load a 1 = replay (stores1 L.reverse) a (w.mem.load a 1)

theorem Tracks.refl (w : World) : Tracks w w := ⟨[], rfl, fun _ _ => rfl⟩

theorem Tracks.trans {a b c : World} (h1 : Tracks a b) (h2 : Tracks b c) : Tracks a c := by
  obtain ⟨L1, h1o, h1m⟩ := h1
  obtain ⟨L2, h2o, h2m⟩ := h2
  refine ⟨L2 ++ L1, by rw [h2o, h1o, List.append_assoc], fun x hx => ?_⟩
  rw [h2m x hx, h1m x hx, List.reverse_append]
  simp only [stores1, List.filterMap_append, replay_append]

theorem Tracks.call (w : World) (c : Callee) (args : List V) : Tracks w (obsCall w c args) :=
  ⟨[_], rfl, fun _ _ => by simp [stores1, replay, obsCall]⟩

theorem Tracks.arena (w : World) (b : ByteArray) : Tracks w { w with mem := { w.mem with arena := b } } :=
  ⟨[], rfl, fun a ha => by rw [load_set_arena ha]; rfl⟩

/-- A byte store goes on by itself. -/
theorem Tracks.stw1 {w w' : World} {a v : UInt64} (h : stw w a 1 v = some w') : Tracks w w' := by
  obtain ⟨m, hm, rfl⟩ := stw_some h
  refine ⟨[.store a 1 v], rfl, fun b _ => ?_⟩
  show m.load b 1 = _
  by_cases e : b = a
  · subst e; rw [load_store_same1 hm]; simp [stores1, replay]
  · rw [Mem.store_other hm b (fun i hi => by
      have : i = 0 := by omega
      subst this; simpa using e)]
    simp [stores1, replay, Ne.symm e]

/-- A store into shared memory goes on with no byte store. -/
theorem Tracks.stwArena {w w' : World} {a v : UInt64} {n off : Nat} (hd : decodeAddr a = some (.arena, off))
    (hn : n ≠ 1) (h : stw w a n v = some w') : Tracks w w' := by
  obtain ⟨m, hm, rfl⟩ := stw_some h
  refine ⟨[.store a n v], rfl, fun b hb => ?_⟩
  show m.load b 1 = _
  rw [store_setArena hd hm, load_set_arena hb]
  simp [stores1, replay, hn]

/-- A step with no byte store that leaves every byte outside shared memory
    goes on by none. -/
theorem Tracks.of_nobyte {w w' : World} (h : NoByte w w') (hm : ∀ a, NotArena a → w'.mem.load a 1 = w.mem.load a 1) :
    Tracks w w' := by
  obtain ⟨L, hL, h0⟩ := h
  have : stores1 L.reverse = [] := by
    simp only [stores1, List.filterMap_reverse] at h0 ⊢; rw [h0]; rfl
  exact ⟨L, hL, fun a ha => by rw [hm a ha, this]; rfl⟩

/-- **A lane's share of a copy goes on by its byte stores.** -/
theorem copyL_tracks {d s n lv : UInt64} : ∀ k i (w w' : World), copyL d s n lv k i w = some w' → Tracks w w'
  | 0, i, w, w', h => by
      simp only [copyL] at h
      split at h
      · cases h
      · cases h; exact Tracks.refl _
  | k + 1, i, w, w', h => by
      simp only [copyL] at h
      split at h
      · obtain ⟨w1, hw1, h⟩ := Option.bind_eq_some_iff.mp h
        refine Tracks.trans ?_ (copyL_tracks k _ w1 w' h)
        simp only [copyStep] at hw1
        split at hw1
        · obtain ⟨_, -, hw1⟩ := Option.bind_eq_some_iff.mp hw1
          exact Tracks.stw1 hw1
        · obtain rfl := bind_const_some hw1; exact Tracks.refl _
      · cases h; exact Tracks.refl _

theorem copyGmem_getD_out (j : Nat) : ∀ (n : Nat) (g : Array UInt8) (dst src : Nat),
    (∀ i, i < n → j ≠ dst + i) → (copyGmem g dst src n).getD j 0 = g.getD j 0
  | 0, _, _, _, _ => rfl
  | n + 1, g, dst, src, h => by
      show (copyGmem (g.setIfInBounds dst (g.getD src 0)) (dst + 1) (src + 1) n).getD j 0 = g.getD j 0
      rw [copyGmem_getD_out j n _ (dst + 1) (src + 1) (fun i hi => by
        have := h (i + 1) (by omega); omega)]
      have : dst ≠ j := fun e => h 0 (by omega) (by omega)
      simp [Array.getElem?_setIfInBounds_ne this]

theorem lanePart_copyTr (l : Nat) (g : Array UInt8) (d s : UInt64) (n : Nat) :
    lanePart l (copyTr g d s n) = (List.range' (32 * 0) (n - 32 * 0)).filterMap
      (fun j => if j % 32 = l then some (d + UInt64.ofNat j, (g.getD (s.toNat + j) 0).toUInt64) else none) := by
  simp only [lanePart, copyTr, Nat.mul_zero, Nat.sub_zero, ← List.range_eq_range', List.filterMap_map]
  congr 1
  funext i
  simp [Function.comp]

/-- One lane's run of a statement. -/
abbrev runL (ans : Nat → List Answer) (start : Nat → Nat) (smv : UInt64) (l f : Nat) (s : WStmt) (st : LSt) :
    Option LSt :=
  laneRun (coopL (ans l) (start l) smv (UInt64.ofNat l)) smv f s st

theorem wargL_eval {st : LSt} {ws : WState} (h : st.regs = ws.regs) (b : WArg) : wargL st b = b.eval ws := by
  cases b <;> simp [wargL, WArg.eval, h]

theorem getD_set!_ne {g : Array UInt8} {i j : Nat} {v : UInt8} (h : i ≠ j) : (g.set! i v).getD j 0 = g.getD j 0 := by
  simp [Array.set!_eq_setIfInBounds, Array.getD_eq_getD_getElem?, Array.getElem?_setIfInBounds_ne h]

/-- What a warp's run of a statement passes, in order: `leaf`'s for each
    cooperative statement it runs. -/
def trace {α : Type} (leaf : WStmt → WState → List α) : Nat → WStmt → WState → List α
  | f, .seq a b, ws => trace leaf f a ws ++ trace leaf f b (a.eval f ws)
  | f, .uif c t e, ws => if ws.regs c == 1 then trace leaf f t ws else trace leaf f e ws
  | 0, .uwhile _ _, _ => []
  | f + 1, .uwhile c b, ws =>
      if ws.regs c == 1 then trace leaf f b ws ++ trace leaf f (.uwhile c b) (b.eval f ws) else []
  | _, s@(.coopWindow ..), ws => leaf s ws
  | _, s@(.coopExtendStep ..), ws => leaf s ws
  | _, _, _ => []
termination_by f s => (f, s)

/-- The barriers among collectives. -/
abbrev bars (cs : List (Intr × List V)) : Nat := cs.countP (·.1 == .barrier)

/-- The shared stores a lane's run of a statement makes, in order, each tagged
    by the barriers before it: `leaf nb`'s for each cooperative statement, `nb`
    the barriers passed before it, counted by the collectives `cc` gives. -/
def strace {β : Type} (cc : WStmt → WState → List (Intr × List V)) (leaf : Nat → WStmt → WState → List β) :
    Nat → Nat → WStmt → WState → List β
  | f, nb, .seq a b, ws => strace cc leaf f nb a ws ++ strace cc leaf f (nb + bars (trace cc f a ws)) b (a.eval f ws)
  | f, nb, .uif c t e, ws => if ws.regs c == 1 then strace cc leaf f nb t ws else strace cc leaf f nb e ws
  | 0, _, .uwhile _ _, _ => []
  | f + 1, nb, .uwhile c b, ws =>
      if ws.regs c == 1 then
        strace cc leaf f nb b ws ++ strace cc leaf f (nb + bars (trace cc f b ws)) (.uwhile c b) (b.eval f ws)
      else []
  | _, nb, s@(.coopWindow ..), ws => leaf nb s ws
  | _, nb, s@(.coopExtendStep ..), ws => leaf nb s ws
  | _, _, _, _ => []
termination_by f _ s => (f, s)


/-- Every lane is answered `T` from its next collective on. -/
def TapeAt (ans : Nat → List Answer) (start : Nat → Nat) (T : List Answer) (sts : Nat → LSt) : Prop :=
  ∀ l, l < 32 → (ans l).drop (collCalls (sts l).w (start l)).length = T

theorem TapeAt.congr {ans : Nat → List Answer} {start : Nat → Nat} {T : List Answer} {sts sts' : Nat → LSt}
    (h : ∀ l, l < 32 → (collCalls (sts' l).w (start l)).length = (collCalls (sts l).w (start l)).length)
    (hT : TapeAt ans start T sts) : TapeAt ans start T sts' := fun l hl => by rw [h l hl]; exact hT l hl

end Lz4Code
