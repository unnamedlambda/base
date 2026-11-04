module
public import Lz4.Compressor
public import AlgorithmLib.Host.Heap
public import AlgorithmLib.Kernel.Warp
public import AlgorithmLib.ML.Ptx.LaneWarp
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lz4.CodeFacts` — what the model's warp run does, and the facts its proofs use

The accesses the model's run of a warp statement may make (`okOf`, `okLZ4`)
and its byte stores, each with the lane that makes it (`gstOf`, `lanePart`);
a world's bytes outside shared memory as its byte stores leave them
(`Tracks`); the ballot's mask and its trailing zeros (`maskN`, `ctz32_mask`);
each lane's probe of the window and the extend step's flags (`winHit`,
`winCand`, `extGood`), and what they come to (`window_of_firstHit`).
-/

namespace Lz4Code

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Prog
  AlgorithmLib.LZ4WarpDSL AlgorithmLib.LibSpec AlgorithmLib.Kernel AlgorithmLib.ML
  AlgorithmLib.ML.BodyLane AlgorithmLib.ML.LaneWarp

theorem mem_dedup_go (r : String) : ∀ (l acc : List String),
    r ∈ l.foldl (fun acc r => if acc.contains r then acc else acc ++ [r]) acc ↔ r ∈ acc ∨ r ∈ l
  | [], acc => by simp
  | x :: l, acc => by
      simp only [List.foldl_cons]
      rw [mem_dedup_go r l]
      split
      · rename_i hc
        have : x ∈ acc := by simpa using hc
        constructor
        · rintro (h | h)
          · exact .inl h
          · exact .inr (List.mem_cons_of_mem _ h)
        · rintro (h | h)
          · exact .inl h
          · rcases List.mem_cons.mp h with rfl | h
            · exact .inl this
            · exact .inr h
      · simp [or_assoc]

theorem mem_dedup {r : String} {l : List String} : r ∈ dedup l ↔ r ∈ l := by
  unfold dedup; rw [mem_dedup_go]; simp

/-- Each register a statement sets other than by a comparison is a value
    register. -/
def Wf (fl : List String) (s : WStmt) : Prop := ∀ r, (r, false) ∈ assigns s → fl.contains r = false

theorem idxOf_lt {ns : List String} {n : String} (h : n ∈ ns) : ns.idxOf n < ns.length :=
  List.idxOf_lt_length_of_mem h

theorem idxOf_inj : ∀ {ns : List String} {a b : String}, a ∈ ns → b ∈ ns →
    ns.idxOf a = ns.idxOf b → a = b
  | [], _, _, ha, _, _ => absurd ha (by simp)
  | y :: ys, a, b, ha, hb, h => by
      simp only [List.idxOf_cons] at h
      by_cases hya : y = a <;> by_cases hyb : y = b
      · exact hya.symm.trans hyb
      · subst hya; rw [show (y == b) = false by simpa using hyb] at h; simp at h
      · subst hyb; rw [show (y == a) = false by simpa using hya] at h; simp at h
      · rw [show (y == a) = false by simpa using hya, show (y == b) = false by simpa using hyb] at h
        simp only [cond_false, Nat.add_right_cancel_iff] at h
        exact idxOf_inj (List.mem_of_ne_of_mem (Ne.symm hya) ha) (List.mem_of_ne_of_mem (Ne.symm hyb) hb) h

theorem opW_run (o : WOp) (a b : UInt64) : (opW o).run a b = o.run a b := by
  cases o <;> rfl

theorem cmpW_run (c : SCmp) (a b : UInt64) : (cmpW c).run a b = c.run a b := by
  cases c <;> rfl

theorem setsOf_names : ∀ (s : WStmt) {n : String}, n ∈ setsOf s → n ∈ namesOf s
  | .seq a b, n, h => by
      simp only [setsOf, List.mem_append] at h; simp only [namesOf, List.mem_append]
      exact h.elim (fun h => .inl (setsOf_names a h)) (fun h => .inr (setsOf_names b h))
  | .uif c t e, n, h => by
      simp only [setsOf, List.mem_append] at h
      rcases h with h | h
      · simp [namesOf, setsOf_names t h]
      · simp [namesOf, setsOf_names e h]
  | .uwhile c b, n, h => by
      simp only [setsOf, List.mem_cons] at h; simp only [namesOf, List.mem_cons]
      exact h.elim .inl (fun h => .inr (setsOf_names b h))
  | .bin .., n, h | .setp .., n, h | .mov .., n, h => by
      simp only [setsOf, List.mem_singleton] at h; subst h; simp [namesOf]
  | .coopWindow .., n, h => by
      simp only [setsOf, List.mem_cons, List.not_mem_nil, or_false] at h
      rcases h with rfl | rfl | rfl <;> simp [namesOf]
  | .coopExtendStep .., n, h => by simp only [setsOf, List.mem_singleton] at h; subst h; simp [namesOf]
  | .skip, n, h | .stgB .., n, h | .coopCopy .., n, h => by simp [setsOf] at h

/-- **A statement's flags are set only by comparisons.** -/
theorem wf_flagRegs (s : WStmt) : Wf (flagRegs s) s := by
  intro r hr
  cases hc : (flagRegs s).contains r with
  | false => rfl
  | true =>
      exfalso
      have hm : r ∈ flagRegs s := by simpa using hc
      unfold flagRegs at hm
      rw [mem_dedup, List.mem_filterMap] at hm
      obtain ⟨⟨r', b⟩, hmem, hsome⟩ := hm
      dsimp only at hsome
      split at hsome
      · rename_i hcond
        obtain rfl := Option.some.inj hsome
        simp only [Bool.and_eq_true, Bool.not_eq_true'] at hcond
        have : (assigns s).contains (r', false) = true := by simpa using hr
        rw [hcond.2] at this; cases this
      · cases hsome

/-- The registers a thread's body starts from: its block's input and output
    bases, every other zero: the same in every lane. -/
def regs0 (inBase outBase : UInt64) : String → UInt64 := fun r =>
  if r = "inBase" then inBase else if r = "outBase" then outBase else 0

theorem toUInt64_ofNat {n : Nat} (h : n < 2 ^ 32) : (UInt32.ofNat n).toUInt64 = UInt64.ofNat n := by
  apply UInt64.toNat_inj.mp
  rw [UInt32.toNat_toUInt64, UInt32.toNat_ofNat', UInt64.toNat_ofNat']
  omega

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

theorem stw_some {w w' : World} {a : UInt64} {n : Nat} {v : UInt64} (h : stw w a n v = some w') :
    ∃ m, w.mem.store a n v = some m ∧ w' = { obsStore w a n v with mem := m } := by
  simp only [stw] at h
  obtain ⟨m, hm, h⟩ := Option.bind_eq_some_iff.mp h
  exact ⟨m, hm, (Option.some.inj h).symm⟩

theorem byte_mask (b : UInt8) : b.toUInt64 &&& 0xff = b.toUInt64 := by
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_and, show (0xff : UInt64).toNat = 2 ^ 8 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod]
  simp

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

/-- A store leaves every byte that could be stored storable. -/
theorem writable_after_store {m m' : Mem} {a b : UInt64} {n : Nat} {v : UInt64} (h : m.store b n v = some m')
    (hw : Writable m a 1) : Writable m' a 1 := by
  obtain ⟨r, off, hd, hsz, hr⟩ := hw
  obtain ⟨r', off', -, hf⟩ := store_frame h
  exact ⟨r, off, hd, by rw [← hf.same.size r]; exact hsz, hr⟩

/-- Replacing shared memory leaves every other byte that could be stored
    storable. -/
theorem writable_arena {m : Mem} {a : UInt64} (hw : Writable m a 1) (ha : NotArena a) (b : ByteArray) :
    Writable { m with arena := b } a 1 := by
  obtain ⟨r, off, hd, hsz, hr⟩ := hw
  refine ⟨r, off, hd, ?_, hr⟩
  cases r with
  | arena => exact absurd hd (ha off)
  | _ => exact hsz

theorem writable_of_stw {w w' : World} {a b : UInt64} {n : Nat} {v : UInt64} (h : stw w b n v = some w')
    (hw : Writable w.mem a 1) : Writable w'.mem a 1 := by
  obtain ⟨m, hm, rfl⟩ := stw_some h
  exact writable_after_store hm hw


/-- The mask of 32 lanes' flags. -/
def maskN (h : Nat → Bool) : Nat := (List.range 32).foldl (fun acc l => if h l then acc ||| 2 ^ l else acc) 0

theorem any_range_eq (Q : Nat → Bool) {j : Nat} (hj : j < 32) :
    (List.range 32).any (fun i => Q i && decide (id i = j)) = Q j := by
  cases hQ : Q j with
  | true =>
      rw [List.any_eq_true]
      exact ⟨j, List.mem_range.mpr hj, by simp [hQ]⟩
  | false =>
      rw [List.any_eq_false]
      intro i _ hi
      simp only [Bool.and_eq_true, decide_eq_true_eq, id] at hi
      obtain ⟨hq, rfl⟩ := hi
      rw [hQ] at hq; cases hq

/-- Bit `i` of the mask is lane `i`'s flag. -/
theorem maskN_testBit (h : Nat → Bool) {i : Nat} (hi : i < 32) : (maskN h).testBit i = h i := by
  unfold maskN
  have := testBit_foldl_or h id (List.range 32) 0 i
  simp only [id] at this
  rw [this, Nat.zero_testBit, Bool.false_or]
  simpa using any_range_eq h hi

theorem fold_or_lt (h : Nat → Bool) : ∀ (L : List Nat) (acc : Nat), acc < 2 ^ 32 → (∀ l ∈ L, l < 32) →
    L.foldl (fun acc l => if h l then acc ||| 2 ^ l else acc) acc < 2 ^ 32
  | [], acc, ha, _ => ha
  | l :: L, acc, ha, hL => by
      rw [List.foldl_cons]
      refine fold_or_lt h L _ ?_ (fun x hx => hL x (by simp [hx]))
      split
      · exact Nat.or_lt_two_pow ha (Nat.pow_lt_pow_right (by decide) (hL l (by simp)))
      · exact ha

theorem maskN_lt (h : Nat → Bool) : maskN h < 2 ^ 32 :=
  fold_or_lt h _ 0 (by decide) (fun l hl => List.mem_range.mp hl)

theorem shr_and_one (x : UInt64) {i : Nat} (hi : i < 64) :
    ((x >>> UInt64.ofNat i) &&& 1 == 1) = x.toNat.testBit i := by
  have hs : (x >>> UInt64.ofNat i).toNat = x.toNat >>> i := by
    rw [UInt64.toNat_shiftRight, UInt64.toNat_ofNat_of_lt' (by omega : i < 2 ^ 64), Nat.mod_eq_of_lt hi]
  have : ((x >>> UInt64.ofNat i) &&& 1 == 1) = decide ((x.toNat >>> i) % 2 = 1) := by
    rw [Bool.eq_iff_iff, beq_iff_eq, decide_eq_true_iff, ← UInt64.toNat_inj, UInt64.toNat_and, hs]
    simp [Nat.and_one_is_mod]
  rw [this, Nat.testBit_eq_decide_div_mod_eq, Nat.shiftRight_eq_div_pow]

/-- **The mask's trailing zeros are the first lane whose flag is set**, 32
    where none is. -/
theorem ctz32_mask (h : Nat → Bool) : ctz32 (UInt32.ofNat (maskN h)).toUInt64 = UInt64.ofNat (firstSetNat h 32 0) := by
  have hx : (UInt32.ofNat (maskN h)).toUInt64.toNat = maskN h := by
    simp [UInt32.toNat_ofNat_of_lt' (maskN_lt h)]
  have hf : (List.range 32).find? (fun i => ((UInt32.ofNat (maskN h)).toUInt64 >>> UInt64.ofNat i) &&& 1 == 1) =
      (List.range 32).find? h :=
    find?_congr _ fun i hi => by
      have hi := List.mem_range.mp hi
      rw [shr_and_one _ (by omega), hx, maskN_testBit h hi]
  unfold ctz32
  rw [hf, firstSetNat_eq_find, ← List.range_eq_range', Nat.zero_add]

theorem or_shl_add {a b k : Nat} (hb : b < 2 ^ k) : b ||| a <<< k = b + a * 2 ^ k := by
  rw [Nat.or_comm, ← Nat.shiftLeft_add_eq_or_of_lt hb, Nat.shiftLeft_eq]; omega

theorem shl_toNat {x : UInt64} {k : Nat} (hk : k < 64) (hx : x.toNat * 2 ^ k < 2 ^ 64) :
    (x <<< UInt64.ofNat k).toNat = x.toNat <<< k := by
  rw [UInt64.toNat_shiftLeft, UInt64.toNat_ofNat_of_lt' (Nat.lt_of_lt_of_le hk (by decide)), Nat.mod_eq_of_lt hk, Nat.shiftLeft_eq,
    Nat.mod_eq_of_lt hx]

theorem bytes4 {x0 x1 x2 x3 : Nat} (h0 : x0 < 256) (h1 : x1 < 256) (h2 : x2 < 256) :
    (x0 ||| x1 <<< 8) ||| (x2 <<< 16 ||| x3 <<< 24) = x0 + 256 * x1 + 65536 * x2 + 16777216 * x3 := by
  have hA : x0 ||| x1 <<< 8 = x0 + x1 * 2 ^ 8 := or_shl_add (by omega)
  have hB : x2 <<< 16 ||| x3 <<< 24 = (x2 ||| x3 <<< 8) <<< 16 := by
    rw [Nat.shiftLeft_or_distrib, ← Nat.shiftLeft_add]
  have hC : x2 ||| x3 <<< 8 = x2 + x3 * 2 ^ 8 := or_shl_add (by omega)
  rw [hA, hB, hC, or_shl_add (by omega)]
  omega

theorem c64_pow_sub_one (h : Nat) : c64 ((2 : Int) ^ h - 1) = UInt64.ofNat (2 ^ h - 1) := by
  have : (1 : Nat) ≤ 2 ^ h := Nat.one_le_two_pow
  rw [show ((2 : Int) ^ h - 1) = ((2 ^ h - 1 : Nat) : Int) by rw [Int.ofNat_sub this]; simp, c64_ofNat]

/-- **A lane's hash is the model's**: the input word times the multiplier,
    shifted to `hashLog` bits. -/
theorem hash_eq {v : UInt64} {g : Array UInt8} {pos hashLog : Nat} (hv : v.toNat = wLoad4 g pos)
    (hh : hashLog ≤ 32) :
    (((v * c64 wHashK) >>> c64 (32 - hashLog)) &&& c64 (2 ^ hashLog - 1)).toNat = wHash g hashLog pos := by
  have hv' : v = UInt64.ofNat (wLoad4 g pos) := by
    apply UInt64.toNat_inj.mp
    rw [hv, UInt64.toNat_ofNat_of_lt']
    have := UInt64.toNat_lt v; rw [hv] at this; exact this
  rw [hv', c64_ofNat, show ((32 : Int) - (hashLog : Int)) = ((32 - hashLog : Nat) : Int) by omega, c64_ofNat,
    c64_pow_sub_one, UInt64.toNat_and, UInt64.toNat_ofNat_of_lt' (show 2 ^ hashLog - 1 < UInt64.size by
      have : 2 ^ hashLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by decide) hh
      exact Nat.lt_of_lt_of_le (by omega) (show 2 ^ 32 ≤ UInt64.size by decide)),
    Nat.and_two_pow_sub_one_eq_mod]
  rfl

theorem getD_inp {g : Array UInt8} {ib len j : Nat} (hj : j < len) :
    (gmemInpAt g ib len).getD j 0 = g.getD (ib + j) 0 := by
  simp [gmemInpAt, List.getD_eq_getElem?_getD, hj]

/-- **The 4-byte check is equality of the two words.** -/
theorem verify4_eq {g : Array UInt8} {ib len p c : Nat} (hp : p + 4 ≤ len) (hc : c + 4 ≤ len) :
    AlgorithmLib.LZ4WarpFind.verify4 (gmemInpAt g ib len) p c = decide (wLoad4 g (ib + c) = wLoad4 g (ib + p)) := by
  have hb : ∀ i, i < 4 → AlgorithmLib.LZ4WarpFind.byte (gmemInpAt g ib len) (p + i) = g.getD (ib + p + i) 0 ∧
      AlgorithmLib.LZ4WarpFind.byte (gmemInpAt g ib len) (c + i) = g.getD (ib + c + i) 0 := by
    intro i hi
    simp only [AlgorithmLib.LZ4WarpFind.byte]
    rw [getD_inp (by omega), getD_inp (by omega), Nat.add_assoc, Nat.add_assoc]
    exact ⟨rfl, rfl⟩
  rw [Bool.eq_iff_iff, decide_eq_true_iff]
  simp only [AlgorithmLib.LZ4WarpFind.verify4, List.all_eq_true, List.mem_range, beq_iff_eq]
  have t := fun (b : UInt8) => b.toNat_lt
  constructor
  · intro h
    have e : ∀ i, i < 4 → (g.getD (ib + p + i) 0).toNat = (g.getD (ib + c + i) 0).toNat := fun i hi => by
      rw [← (hb i hi).1, ← (hb i hi).2, h i hi]
    have e0 := e 0 (by decide); have e1 := e 1 (by decide); have e2 := e 2 (by decide); have e3 := e 3 (by decide)
    simp only [Nat.add_zero] at e0
    simp only [wLoad4, Nat.add_assoc] at e0 e1 e2 e3 ⊢
    omega
  · intro h i hi
    rw [(hb i hi).1, (hb i hi).2]
    simp only [wLoad4] at h
    have := t (g.getD (ib + c) 0); have := t (g.getD (ib + c + 1) 0); have := t (g.getD (ib + c + 2) 0)
    have := t (g.getD (ib + c + 3) 0); have := t (g.getD (ib + p) 0); have := t (g.getD (ib + p + 1) 0)
    have := t (g.getD (ib + p + 2) 0); have := t (g.getD (ib + p + 3) 0)
    apply UInt8.toNat_inj.mp
    match i, hi with
    | 0, _ => simp only [Nat.add_zero]; omega
    | 1, _ => omega
    | 2, _ => omega
    | 3, _ => omega

theorem probe_past {inp : List UInt8} {oracle : Nat → Option Nat} {searchLim p : Nat} (h : ¬ p < searchLim) :
    AlgorithmLib.LZ4WarpFind.probe inp oracle searchLim p = none := by
  unfold AlgorithmLib.LZ4WarpFind.probe
  split
  · rw [if_neg (fun h' => h h'.1)]
  · rfl

theorem probe_none {inp : List UInt8} {oracle : Nat → Option Nat} {searchLim p : Nat} (h : oracle p = none) :
    AlgorithmLib.LZ4WarpFind.probe inp oracle searchLim p = none := by
  unfold AlgorithmLib.LZ4WarpFind.probe; rw [h]

theorem probe_some {inp : List UInt8} {oracle : Nat → Option Nat} {searchLim p c : Nat} (h : oracle p = some c) :
    AlgorithmLib.LZ4WarpFind.probe inp oracle searchLim p =
      if p < searchLim ∧ c < p ∧ AlgorithmLib.LZ4WarpFind.verify4 inp p c then some c else none := by
  unfold AlgorithmLib.LZ4WarpFind.probe; rw [h]

theorem c64_sub4 {n : Nat} (h : 4 ≤ n) : c64 ((n : Int) - 4) = UInt64.ofNat (n - 4) := by
  rw [show ((n : Int) - 4) = ((n - 4 : Nat) : Int) by omega, c64_ofNat]

theorem toNat_le_ofNat {x : UInt64} {n : Nat} (hn : n < 2 ^ 64) : x ≤ UInt64.ofNat n ↔ x.toNat ≤ n := by
  rw [UInt64.le_iff_toNat_le, UInt64.toNat_ofNat_of_lt' hn]

theorem toNat_lt_ofNat {x : UInt64} {n : Nat} (hn : n < 2 ^ 64) : x < UInt64.ofNat n ↔ x.toNat < n := by
  rw [UInt64.lt_iff_toNat_lt, UInt64.toNat_ofNat_of_lt' hn]

/-- Lane `l`'s position in the window. -/
abbrev winPos (ws : WState) (sp : String) (l : Nat) : Nat := (ws.regs sp).toNat + l

/-- The model's table, as the lanes read it. -/
abbrev winOracle (ws : WState) (hashLog tbl : Nat) : Nat → Option Nat :=
  tableOracle ws.gmem ws.smem hashLog tbl (ws.regs "inBase").toNat

/-- Whether lane `l`'s probe hits. -/
abbrev winHit (ws : WState) (sp : String) (inStride searchLim hashLog tbl l : Nat) : Bool :=
  (AlgorithmLib.LZ4WarpFind.probe (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (winOracle ws hashLog tbl)
    searchLim (winPos ws sp l)).isSome

/-- Lane `l`'s candidate, cut to a word. -/
abbrev winCand (ws : WState) (sp : String) (inStride hashLog tbl l : Nat) : UInt64 :=
  (winOracle ws hashLog tbl (min (winPos ws sp l) (inStride - 4))).elim 0xffffffff UInt64.ofNat

theorem wHash_lt (g : Array UInt8) (hashLog pos : Nat) : wHash g hashLog pos < 2 ^ hashLog :=
  Nat.mod_lt _ (Nat.two_pow_pos _)

theorem maskN_eq_zero {h : Nat → Bool} (hf : ∀ j, j < 32 → h j = false) : maskN h = 0 := by
  apply Nat.eq_of_testBit_eq
  intro i
  rw [Nat.zero_testBit]
  by_cases hi : i < 32
  · rw [maskN_testBit h hi, hf i hi]
  · have := maskN_lt h
    exact Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le this (Nat.pow_le_pow_right (by decide) (by omega)))

theorem maskN_ne_zero {h : Nat → Bool} {L : Nat} (hL : L < 32) (hh : h L = true) : maskN h ≠ 0 := by
  intro e
  have := maskN_testBit h hL
  rw [e, Nat.zero_testBit, hh] at this
  cases this

theorem firstSetNat_le (g : Nat → Bool) : firstSetNat g 32 0 ≤ 32 := by
  rw [firstSetNat_eq_find]
  cases e : (List.range' 0 32).find? g with
  | none => simp
  | some v => have := List.mem_range'.mp (List.mem_of_find?_eq_some e); simp; omega

theorem tableOracle_lt {g sm : Array UInt8} {hashLog tbl ib pos c : Nat}
    (h : tableOracle g sm hashLog tbl ib pos = some c) : c < 65535 := by
  simp only [tableOracle] at h
  split at h
  · cases h
  obtain rfl := (Option.some.inj h).symm
  have := (sm.getD (tbl + 2 * wHash g hashLog (ib + pos)) 0).toNat_lt
  have := (sm.getD (tbl + 2 * wHash g hashLog (ib + pos) + 1) 0).toNat_lt
  omega

theorem winCand_lt {ws : WState} {sp : String} {inStride hashLog tbl l : Nat} :
    (winCand ws sp inStride hashLog tbl l).toNat < 2 ^ 32 := by
  simp only [winCand]
  cases e : winOracle ws hashLog tbl (min (winPos ws sp l) (inStride - 4)) with
  | none => decide
  | some c =>
      have := tableOracle_lt e
      simp only [Option.elim]; rw [UInt64.toNat_ofNat_of_lt' (show c < 2 ^ 64 by omega)]; omega

theorem winCand_mask {ws : WState} {sp : String} {inStride hashLog tbl l : Nat} :
    winCand ws sp inStride hashLog tbl l &&& 0xffffffff = winCand ws sp inStride hashLog tbl l := by
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_and, show (0xffffffff : UInt64).toNat = 2 ^ 32 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod,
    Nat.mod_eq_of_lt winCand_lt]

theorem flatMap_congr_mem {α β : Type} {f g : α → List β} : ∀ {l : List α}, (∀ a ∈ l, f a = g a) →
    l.flatMap f = l.flatMap g
  | [], _ => rfl
  | a :: l, h => by
      rw [List.flatMap_cons, List.flatMap_cons, h a (by simp), flatMap_congr_mem (fun x hx => h x (by simp [hx]))]

theorem firstHit_some {f : Nat → Bool} : ∀ {b start L : Nat}, AlgorithmLib.LZ4WarpSched.firstHit f b start = some L →
    f L = true ∧ start ≤ L ∧ L < start + b
  | 0, _, _, h => by simp [AlgorithmLib.LZ4WarpSched.firstHit] at h
  | b + 1, start, L, h => by
      unfold AlgorithmLib.LZ4WarpSched.firstHit at h
      split at h
      · obtain rfl := (Option.some.inj h).symm; exact ⟨by assumption, Nat.le_refl _, by omega⟩
      · obtain ⟨h1, h2, h3⟩ := firstHit_some h; exact ⟨h1, by omega, by omega⟩

theorem firstHit_none {f : Nat → Bool} : ∀ {b start : Nat}, AlgorithmLib.LZ4WarpSched.firstHit f b start = none →
    ∀ j, start ≤ j → j < start + b → f j = false
  | 0, _, _, j, h1, h2 => by omega
  | b + 1, start, h, j, h1, h2 => by
      unfold AlgorithmLib.LZ4WarpSched.firstHit at h
      split at h
      · cases h
      · rename_i hf
        rcases Nat.eq_or_lt_of_le h1 with rfl | hlt
        · simpa using hf
        · exact firstHit_none h j (by omega) (by omega)

theorem probe_some_inv {inp : List UInt8} {oracle : Nat → Option Nat} {searchLim p c : Nat}
    (h : AlgorithmLib.LZ4WarpFind.probe inp oracle searchLim p = some c) : oracle p = some c ∧ p < searchLim := by
  unfold AlgorithmLib.LZ4WarpFind.probe at h
  split at h
  · rename_i c' hc
    split at h
    · rename_i hcond; obtain rfl := (Option.some.inj h).symm; exact ⟨hc, hcond.1⟩
    · cases h
  · cases h

/-- **The window is the first hitting lane's probe**, none where no lane hits. -/
theorem window_of_firstHit {inp : List UInt8} {oracle : Nat → Option Nat} {searchLim s : Nat} :
    (∀ L, AlgorithmLib.LZ4WarpSched.firstHit
        (fun L => (AlgorithmLib.LZ4WarpFind.probe inp oracle searchLim (s + L)).isSome) 32 0 = some L →
      ∃ c, AlgorithmLib.LZ4WarpFind.probe inp oracle searchLim (s + L) = some c ∧
        AlgorithmLib.LZ4WarpFind.window inp oracle searchLim s = some (s + L, c)) ∧
    (AlgorithmLib.LZ4WarpSched.firstHit
        (fun L => (AlgorithmLib.LZ4WarpFind.probe inp oracle searchLim (s + L)).isSome) 32 0 = none →
      AlgorithmLib.LZ4WarpFind.window inp oracle searchLim s = none) := by
  rw [← AlgorithmLib.LZ4WarpSched.coopWindow_eq_window]
  unfold AlgorithmLib.LZ4WarpSched.coopWindow
  refine ⟨fun L e => ?_, fun e => by rw [e]⟩
  have := (firstHit_some e).1
  obtain ⟨c, hc⟩ := Option.isSome_iff_exists.mp this
  exact ⟨c, hc, by rw [e]; simp only [hc]; rfl⟩

/-- Whether lane `l`'s byte of the match is good. -/
abbrev extGood (ws : WState) (p0 cand0 ml : String) (inStride endCap l : Nat) : Bool :=
  AlgorithmLib.LZ4WarpSched.good (gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride) (ws.regs p0).toNat
    (ws.regs cand0).toNat endCap ((ws.regs ml).toNat + l)

theorem byte_beq (a b : UInt8) : (a.toUInt64 == b.toUInt64) = (a == b) := by
  rw [Bool.eq_iff_iff, beq_iff_eq, beq_iff_eq]
  constructor
  · intro h; exact UInt8.toNat_inj.mp (by simpa using congrArg UInt64.toNat h)
  · intro h; rw [h]

/-- The ballot's complement, within 32 bits, is the mask of the negated flags. -/
theorem maskN_compl (h : Nat → Bool) :
    ((UInt32.ofNat (maskN h)).toUInt64 ^^^ 0xffffffff) &&& 0xffffffff =
      (UInt32.ofNat (maskN fun k => !h k)).toUInt64 := by
  apply UInt64.toNat_inj.mp
  have h1 := maskN_lt h
  have h2 := maskN_lt (fun k => !h k)
  simp only [UInt64.toNat_and, UInt64.toNat_xor, UInt32.toNat_toUInt64, UInt32.toNat_ofNat_of_lt' h1,
    UInt32.toNat_ofNat_of_lt' h2, show (0xffffffff : UInt64).toNat = 2 ^ 32 - 1 from rfl]
  apply Nat.eq_of_testBit_eq
  intro i
  rw [Nat.testBit_and, Nat.testBit_xor, Nat.testBit_two_pow_sub_one]
  by_cases hi : i < 32
  · rw [maskN_testBit _ hi, maskN_testBit _ hi]; simp [hi]
  · rw [Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le h2 (Nat.pow_le_pow_right (by decide) (by omega)))]
    simp [hi]

/-- What a cooperative statement needs of the warp's state. -/
def okLZ4 (I : UInt64 → Prop) : WStmt → WState → Prop
  | .coopWindow found p0 cand0 sp inStride searchLim hashLog tbl, ws =>
      searchLim + 4 ≤ inStride ∧ hashLog ≤ 32 ∧ (∀ j, j < inStride → I (ws.regs "inBase" + UInt64.ofNat j)) ∧
      (ws.regs "inBase").toNat + inStride < 2 ^ 64 ∧ (ws.regs sp).toNat + 32 < 2 ^ 64 ∧
      ws.regs "tbl" = UInt64.ofNat tbl ∧ tbl + 2 * 2 ^ hashLog + 2 < 2 ^ 36 ∧
      found ≠ p0 ∧ found ≠ cand0 ∧ p0 ≠ cand0 ∧ tbl + 2 * 2 ^ hashLog ≤ ws.smem.size
  | .coopExtendStep _ p0 cand0 ml inStride endCap, ws =>
      ws.regs "ecR" = UInt64.ofNat endCap ∧ ws.regs "ec1" = UInt64.ofNat (endCap - 1) ∧ 1 ≤ endCap ∧
      endCap ≤ inStride ∧ (ws.regs cand0).toNat < (ws.regs p0).toNat ∧
      (ws.regs p0).toNat + (ws.regs ml).toNat + 32 < 2 ^ 64 ∧
      (∀ j, j < inStride → I (ws.regs "inBase" + UInt64.ofNat j)) ∧ (ws.regs "inBase").toNat + inStride < 2 ^ 64 ∧
      (ws.regs p0).toNat < endCap
  | _, _ => True

end Lz4Code
