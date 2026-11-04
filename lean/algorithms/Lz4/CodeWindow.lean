module
public import Lz4.CodeBallot
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lz4.CodeWindow` — a warp's lanes run the window with the warp

What a lane's share of the window reads: its four input bytes (`load4L_eq`)
and its table entry (`load2_eq`), as the warp's model reads them.
-/

namespace Lz4Code

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Prog
  AlgorithmLib.LZ4WarpDSL AlgorithmLib.LibSpec AlgorithmLib.Kernel

theorem or_shl_add {a b k : Nat} (hb : b < 2 ^ k) : b ||| a <<< k = b + a * 2 ^ k := by
  rw [Nat.or_comm, ← Nat.shiftLeft_add_eq_or_of_lt hb, Nat.shiftLeft_eq]; omega

theorem byte_toNat_lt (b : UInt8) : b.toUInt64.toNat < 256 := by simpa using b.toNat_lt

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

/-- **Two bytes a load reads**, little-endian. -/
theorem load2_eq {m : Mem} {a v : UInt64} {r : Region} {off : Nat} (hd : decodeAddr a = some (r, off))
    (h : m.load a 2 = some v) :
    v.toNat = ((m.region r).get! off).toNat + 256 * ((m.region r).get! (off + 1)).toNat := by
  simp only [Mem.load, hd, Option.bind_eq_bind, Option.bind_some] at h
  split at h
  · cases h
  obtain rfl := (Option.some.inj h).symm
  simp only [List.range_succ, List.range_zero, List.nil_append, List.foldr_append, List.foldr_cons, List.foldr_nil,
    Nat.add_zero]
  have h0 := byte_toNat_lt ((m.region r).get! off)
  have h1 := byte_toNat_lt ((m.region r).get! (off + 1))
  rw [UInt64.toNat_or, UInt64.toNat_shiftLeft, UInt64.toNat_or, UInt64.toNat_shiftLeft]
  simp only [show (8 : UInt64).toNat % 64 = 8 from rfl, show (0 : UInt64).toNat = 0 from rfl, Nat.zero_shiftLeft,
    Nat.zero_mod, Nat.zero_or]
  have q0 : ((m.region r).get! off).toUInt64.toNat = ((m.region r).get! off).toNat := by simp
  have q1 : ((m.region r).get! (off + 1)).toUInt64.toNat = ((m.region r).get! (off + 1)).toNat := by simp
  have l0 := ((m.region r).get! off).toNat_lt
  have l1 := ((m.region r).get! (off + 1)).toNat_lt
  rw [q0, q1, Nat.mod_eq_of_lt (by rw [Nat.shiftLeft_eq]; omega), Nat.or_comm, or_shl_add (k := 8) (by omega)]
  omega

/-- **Four bytes a lane loads**, each as `I` holds it, little-endian. -/
theorem load4L_eq {m : Mem} {a v : UInt64} {b : Nat → UInt8}
    (hb0 : m.load a 1 = some (b 0).toUInt64) (hb1 : m.load (a + c64 1) 1 = some (b 1).toUInt64)
    (hb2 : m.load (a + c64 2) 1 = some (b 2).toUInt64) (hb3 : m.load (a + c64 3) 1 = some (b 3).toUInt64)
    (h : load4L m a = some v) :
    v.toNat = (b 0).toNat + 256 * (b 1).toNat + 65536 * (b 2).toNat + 16777216 * (b 3).toNat := by
  simp only [load4L] at h
  obtain ⟨b0, h0, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨b1, h1, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨b2, h2, h⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨b3, h3, h⟩ := Option.bind_eq_some_iff.mp h
  obtain rfl := (Option.some.inj h).symm
  have e0 := hb0; have e1 := hb1; have e2 := hb2; have e3 := hb3
  rw [h0] at e0; rw [h1] at e1; rw [h2] at e2; rw [h3] at e3
  obtain rfl := Option.some.inj e0; obtain rfl := Option.some.inj e1
  obtain rfl := Option.some.inj e2; obtain rfl := Option.some.inj e3
  have l0 := byte_toNat_lt (b 0); have l1 := byte_toNat_lt (b 1)
  have l2 := byte_toNat_lt (b 2); have l3 := byte_toNat_lt (b 3)
  rw [show c64 8 = UInt64.ofNat 8 by decide, show c64 16 = UInt64.ofNat 16 by decide,
    show c64 24 = UInt64.ofNat 24 by decide]
  rw [UInt64.toNat_or, UInt64.toNat_or, UInt64.toNat_or, shl_toNat (by decide) (by omega),
    shl_toNat (by decide) (by omega), shl_toNat (by decide) (by omega), bytes4 l0 l1 l2]
  simp

theorem c64_nat_sub {a b : Nat} (h : b ≤ a) : c64 ((a : Int) - (b : Int)) = UInt64.ofNat (a - b) := by
  rw [← Int.ofNat_sub h, c64_ofNat]

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

/-- **Four input bytes a lane loads**, from `I`, are the model's word. -/
theorem load4_input {I : UInt64 → Prop} {ws : WState} {st : LSt} (hR : Rel I ws st) {base x v : UInt64}
    {len : Nat} (hin : ∀ j, j < len → I (base + UInt64.ofNat j)) (hb : base.toNat + len < 2 ^ 64)
    (hx : x.toNat + 4 ≤ len) (h : load4L st.w.mem (base + x) = some v) :
    v.toNat = wLoad4 ws.gmem (base.toNat + x.toNat) := by
  have key : ∀ i : Nat, i < 4 → base + x + c64 (i : Int) = base + UInt64.ofNat (x.toNat + i) ∧
      (base + UInt64.ofNat (x.toNat + i)).toNat = base.toNat + x.toNat + i := by
    intro i hi
    constructor
    · rw [c64_ofNat, UInt64.add_assoc, UInt64.ofNat_add, UInt64.ofNat_toNat]
    · rw [UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show x.toNat + i < 2 ^ 64 by omega),
        Nat.mod_eq_of_lt (by omega)]; omega
  have hl : ∀ i : Nat, i < 4 → st.w.mem.load (base + x + c64 (i : Int)) 1 =
      some (ws.gmem.getD (base.toNat + x.toNat + i) 0).toUInt64 := by
    intro i hi
    rw [(key i hi).1, hR.gin _ (hin _ (by omega)), (key i hi).2]
  have e := load4L_eq (b := fun i => ws.gmem.getD (base.toNat + x.toNat + i) 0)
    (by simpa [show c64 0 = 0 by decide] using hl 0 (by decide)) (by simpa using hl 1 (by decide)) (by simpa using hl 2 (by decide))
    (by simpa using hl 3 (by decide)) h
  rw [e]; simp [wLoad4]

/-- **Two table bytes a lane loads**, from shared memory, are the model's. -/
theorem load_table {I : UInt64 → Prop} {ws : WState} {st : LSt} (hR : Rel I ws st) {a v : UInt64}
    (ha : a.toNat < 2 ^ 36) (h : st.w.mem.load (addrOf .arena 0 + a) 2 = some v) :
    v.toNat = (ws.smem.getD a.toNat 0).toNat + 256 * (ws.smem.getD (a.toNat + 1) 0).toNat := by
  have hd : decodeAddr (addrOf .arena 0 + a) = some (.arena, a.toNat) := by
    rw [show addrOf .arena 0 + a = regionBase .arena + UInt64.ofNat a.toNat by simp [addrOf]]
    exact decodeAddr_base _ _ (by rw [regionSpan_toNat]; exact ha)
  rw [load2_eq hd h]
  simp only [Mem.region, ByteArray.get!, ← hR.smem]
  simp only [Array.getD_eq_getD_getElem?, getElem!_def]
  cases st.w.mem.arena.data[a.toNat]? <;> cases st.w.mem.arena.data[a.toNat + 1]? <;> rfl

/-- Four bytes load where each does. -/
theorem load4L_some {m : Mem} {a : UInt64} (h : ∀ i : Nat, i < 4 → ∃ v, m.load (a + c64 (i : Int)) 1 = some v) :
    ∃ v, load4L m a = some v := by
  obtain ⟨b0, h0⟩ := h 0 (by decide)
  rw [show a + c64 ((0 : Nat) : Int) = a by rw [show c64 ((0 : Nat) : Int) = 0 by decide, UInt64.add_zero]] at h0
  obtain ⟨b1, h1⟩ : ∃ v, m.load (a + c64 1) 1 = some v := h 1 (by decide)
  obtain ⟨b2, h2⟩ : ∃ v, m.load (a + c64 2) 1 = some v := h 2 (by decide)
  obtain ⟨b3, h3⟩ : ∃ v, m.load (a + c64 3) 1 = some v := h 3 (by decide)
  simp only [load4L, h0, h1, h2, h3, Option.bind_eq_bind, Option.bind_some]
  exact ⟨_, rfl⟩

/-- **Four input bytes a lane loads** load, where they lie in `I`. -/
theorem load4_input_some {I : UInt64 → Prop} {ws : WState} {st : LSt} (hR : Rel I ws st) {base x : UInt64}
    {len : Nat} (hin : ∀ j, j < len → I (base + UInt64.ofNat j))
    (hx : x.toNat + 4 ≤ len) : ∃ v, load4L st.w.mem (base + x) = some v := by
  refine load4L_some fun i hi => ⟨(ws.gmem.getD (base + UInt64.ofNat (x.toNat + i)).toNat 0).toUInt64, ?_⟩
  have key : base + x + c64 (i : Int) = base + UInt64.ofNat (x.toNat + i) := by
    rw [c64_ofNat, UInt64.add_assoc, UInt64.ofNat_add, UInt64.ofNat_toNat]
  rw [key]
  exact hR.gin _ (hin _ (by omega))

/-- **Two table bytes a lane loads** load, where they lie in shared memory. -/
theorem load_table_some {m : Mem} {a : UInt64} (ha : a.toNat < 2 ^ 36) (hsz : a.toNat + 2 ≤ m.arena.size) :
    ∃ v, m.load (addrOf .arena 0 + a) 2 = some v :=
  load_some (arena_decode ha) (by simpa [Mem.region] using hsz) (by decide)

/-- **An input byte a lane loads** loads, where it lies in `I`. -/
theorem load_input_some {I : UInt64 → Prop} {ws : WState} {st : LSt} (hR : Rel I ws st) {base x : UInt64}
    {len : Nat} (hin : ∀ j, j < len → I (base + UInt64.ofNat j)) (hx : x.toNat < len) :
    ∃ v, st.w.mem.load (base + x) 1 = some v := by
  have := hR.gin _ (hin _ hx)
  rw [UInt64.ofNat_toNat] at this
  exact ⟨_, this⟩

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

/-- **A lane's probe is the model's**: its hit flag is whether the model's
    probe at `sp + lane` finds a candidate, which is then its candidate; its
    table slot and its candidate, cut to a word, are the model's at the
    position it reads. -/
theorem lane_probe {I : UInt64 → Prop} {ws : WState} {st : LSt} (hR : Rel I ws st)
    {inStride searchLim hashLog tbl l : Nat} {sp : String}
    (hsl : searchLim + 4 ≤ inStride) (hh : hashLog ≤ 32)
    (hin : ∀ j, j < inStride → I (ws.regs "inBase" + UInt64.ofNat j))
    (hib : (ws.regs "inBase").toNat + inStride < 2 ^ 64) (hs : (ws.regs sp).toNat + l < 2 ^ 64)
    (htbl : ws.regs "tbl" = UInt64.ofNat tbl) (htb : tbl + 2 * 2 ^ hashLog + 2 < 2 ^ 36)
    {posP rp v32 addr candRaw cand rc vc : UInt64}
    (hpos : posP = ws.regs sp + UInt64.ofNat l)
    (hrp : rp = if posP ≤ c64 (inStride - 4) then posP else c64 (inStride - 4))
    (hv32 : load4L st.w.mem (ws.regs "inBase" + rp) = some v32)
    (haddr : addr = ((((v32 * c64 wHashK) >>> c64 (32 - hashLog)) &&& c64 (2 ^ hashLog - 1)) <<< c64 1) +
      ws.regs "tbl")
    (hcr : st.w.mem.load (addrOf .arena 0 + addr) 2 = some candRaw)
    (hcand : cand = candRaw + c64 (-1))
    (hrc : rc = if cand ≤ c64 (inStride - 4) then cand else c64 (inStride - 4))
    (hvc : load4L st.w.mem (ws.regs "inBase" + rc) = some vc) :
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
    rw [← hrpN]; exact load4_input hR hin hib (by rw [hrpN]; omega) hv32
  have hH := hash_eq hv hh
  generalize hHd : wHash ws.gmem hashLog ((ws.regs "inBase").toNat + min ((ws.regs sp).toNat + l) (inStride - 4)) = H
    at hH
  have hHlt : H < 2 ^ hashLog := by rw [← hHd]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
  have hpow : 2 ^ hashLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by decide) hh
  have haddrN : addr.toNat = tbl + 2 * H := by
    rw [haddr, htbl, show c64 1 = UInt64.ofNat 1 by decide, UInt64.toNat_add, shl_toNat (by decide) (by omega),
      hH, UInt64.toNat_ofNat_of_lt' (show tbl < 2 ^ 64 by omega), Nat.shiftLeft_eq, Nat.mod_eq_of_lt (by omega)]
    omega
  have hcrN := load_table hR (by rw [haddrN]; omega) hcr
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
        rw [hrcN] at hvc; rw [← hcn]; exact load4_input hR hin hib (by omega) hvc
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

/-! ## Shared stores, tagged by the barriers before them -/

theorem shStep_fold_count : ∀ (L : List Obs) (n : Nat) (acc : List (Nat × UInt64 × Nat × UInt64)),
    (L.foldl shStep (n, acc)).1 = n + (L.filterMap collOf).countP (·.1 == .barrier)
  | [], n, acc => by simp
  | o :: L, n, acc => by
      rw [List.foldl_cons]
      cases o with
      | call c a =>
          cases c with
          | intr i =>
              by_cases hb : i = .barrier
              · subst hb
                simp only [shStep]
                rw [shStep_fold_count L (n + 1) acc]
                simp [collOf, Intr.isColl, List.countP_cons, show (Intr.barrier == Intr.barrier) = true by decide]
                omega
              · have hs : shStep (n, acc) (.call (.intr i) a) = (n, acc) := by
                  cases i <;> first | exact absurd rfl hb | rfl
                have hib : (i == Intr.barrier) = false := by cases i <;> first | exact absurd rfl hb | decide
                rw [hs, shStep_fold_count L n acc]
                simp only [List.filterMap_cons, collOf]
                split
                · rfl
                · rename_i b heq
                  split at heq
                  · cases heq; simp [List.countP_cons, hib]
                  · cases heq
          | _ => simp only [shStep]; rw [shStep_fold_count L n acc]; simp [collOf]
      | store a w v =>
          simp only [shStep]
          split
          · rw [shStep_fold_count]; simp [collOf]
          · rw [shStep_fold_count]; simp [collOf]

/-- What the fold adds: stores tagged at least where it starts. -/
theorem shStep_fold_new : ∀ (L : List Obs) (n : Nat) (acc : List (Nat × UInt64 × Nat × UInt64)),
    ∃ nw, (L.foldl shStep (n, acc)).2 = nw ++ acc ∧ ∀ x ∈ nw, n ≤ x.1
  | [], n, acc => ⟨[], by simp, by simp⟩
  | o :: L, n, acc => by
      rw [List.foldl_cons]
      have step : ∃ n' nw0, shStep (n, acc) o = (n', nw0 ++ acc) ∧ n ≤ n' ∧ ∀ x ∈ nw0, n ≤ x.1 := by
        cases o with
        | call c a =>
            cases c with
            | intr i =>
                by_cases hb : i = .barrier
                · subst hb; exact ⟨n + 1, [], rfl, by omega, by simp⟩
                · refine ⟨n, [], ?_, Nat.le_refl _, by simp⟩
                  cases i <;> first | exact absurd rfl hb | rfl
            | _ => exact ⟨n, [], rfl, Nat.le_refl _, by simp⟩
        | store a w v =>
            simp only [shStep]
            split
            · exact ⟨n, [(n, a, w, v)], rfl, Nat.le_refl _, by simp⟩
            · exact ⟨n, [], rfl, Nat.le_refl _, by simp⟩
      obtain ⟨n', nw0, hs, hn, hnw0⟩ := step
      rw [hs]
      obtain ⟨nw, hnw, hge⟩ := shStep_fold_new L n' (nw0 ++ acc)
      exact ⟨nw ++ nw0, by rw [hnw, List.append_assoc], fun x hx => by
        rcases List.mem_append.mp hx with h | h
        · exact Nat.le_trans hn (hge x h)
        · exact hnw0 x h⟩

/-! ## The barrier law's snapshots -/

/-- Shared memory after the first `b` phases of a block's stores. -/
def snapMem (all : List (Nat × UInt64 × Nat × UInt64)) (smem0 : ByteArray) : Nat → Mem
  | 0 => { arena := smem0, data := .empty, out := .empty }
  | b + 1 => phase all b (snapMem all smem0 b)

theorem snapshots_fold (all : List (Nat × UInt64 × Nat × UInt64)) (smem0 : ByteArray) : ∀ m,
    ((List.range m).foldl (fun (acc : Mem × Array ByteArray) b =>
        (phase all b acc.1, acc.2.push (phase all b acc.1).arena))
      (({ arena := smem0, data := .empty, out := .empty } : Mem), #[])) =
      (snapMem all smem0 m, ((List.range m).map fun b => (snapMem all smem0 (b + 1)).arena).toArray)
  | 0 => rfl
  | m + 1 => by
      rw [List.range_succ, List.foldl_append, snapshots_fold all smem0 m]
      simp [snapMem, List.range_succ]

/-- **The `b`-th snapshot is shared memory after phases `0 … b`**, where some
    thread passed that many barriers. -/
theorem snapshots_get {n : Nat} {smem0 : ByteArray} {calls : Nat → List (Intr × List V)}
    {stores : Nat → List (Nat × UInt64 × Nat × UInt64)} {b t : Nat} (ht : t < n)
    (hb : b < ((calls t).filter (·.1 == .barrier)).length) :
    (snapshots n smem0 calls stores)[b]? = some (snapMem ((List.range n).flatMap stores) smem0 (b + 1)).arena := by
  unfold snapshots
  simp only
  have hnb : ((calls t).filter (·.1 == .barrier)).length ≤
      (List.range n).foldl (fun m t => max m ((calls t).filter (·.1 == .barrier)).length) 0 := by
    have key : ∀ (L : List Nat) (m0 : Nat), t ∈ L ∨ ((calls t).filter (·.1 == .barrier)).length ≤ m0 →
        ((calls t).filter (·.1 == .barrier)).length ≤
          L.foldl (fun m t => max m ((calls t).filter (·.1 == .barrier)).length) m0 := by
      intro L
      induction L with
      | nil => intro m0 h; simpa using h
      | cons x L ih =>
          intro m0 h
          rw [List.foldl_cons]
          apply ih
          rcases h with h | h
          · rcases List.mem_cons.mp h with rfl | h
            · right; exact Nat.le_max_right _ _
            · left; exact h
          · right; exact Nat.le_trans h (Nat.le_max_left _ _)
    exact key _ 0 (.inl (List.mem_range.mpr ht))
  rw [snapshots_fold]
  simp only [List.getElem?_toArray, List.getElem?_map]
  rw [List.getElem?_range (by omega)]
  rfl

/-- A phase applies the stores tagged with it, in order. -/
theorem phase_filter (all : List (Nat × UInt64 × Nat × UInt64)) (b : Nat) (cur : Mem) :
    phase all b cur = (all.filter (·.1 = b)).foldl
      (fun c x => (c.store x.2.1 x.2.2.1 x.2.2.2).getD c) cur := by
  unfold phase
  induction all generalizing cur with
  | nil => rfl
  | cons x xs ih =>
      rw [List.foldl_cons, List.filter_cons]
      by_cases hx : x.1 = b
      · rw [if_pos hx, ih]; simp [hx]
      · rw [if_neg hx, ih]; simp [hx]

/-- The shared stores a run goes on to make are tagged from the barriers it
    had passed. -/
theorem sharedStores_ext {w w' : World} {start : Nat} (hs : start ≤ w.obs.length) {L : List Obs}
    (h : w'.obs = L ++ w.obs) :
    sharedStores w' start = sharedStores w start ++
      (L.reverse.foldl shStep ((collCalls w start).countP (·.1 == .barrier), [])).2.reverse := by
  unfold sharedStores
  rw [h, List.reverse_append, List.drop_append_of_le_length (by simpa using hs), List.foldl_append]
  generalize hS : (w.obs.reverse.drop start).foldl shStep (0, []) = S0
  obtain ⟨n0, acc0⟩ := S0
  have hn0 : n0 = (collCalls w start).countP (·.1 == .barrier) := by
    have := shStep_fold_count (w.obs.reverse.drop start) 0 []
    rw [hS] at this; simpa [collCalls] using this
  subst hn0
  have key : ∀ (M : List Obs) (n : Nat) (acc : List (Nat × UInt64 × Nat × UInt64)),
      (M.foldl shStep (n, acc)).2 = (M.foldl shStep (n, [])).2 ++ acc ∧
        (M.foldl shStep (n, acc)).1 = (M.foldl shStep (n, [])).1 := by
    intro M
    induction M with
    | nil => intro n acc; simp
    | cons o M ih =>
        intro n acc
        rw [List.foldl_cons, List.foldl_cons]
        have hstep : ∃ n' nw, shStep (n, acc) o = (n', nw ++ acc) ∧ shStep (n, []) o = (n', nw) := by
          cases o with
          | call c a =>
              cases c with
              | intr i =>
                  by_cases hb : i = .barrier
                  · subst hb; exact ⟨n + 1, [], rfl, rfl⟩
                  · refine ⟨n, [], ?_, ?_⟩ <;> (cases i <;> first | exact absurd rfl hb | rfl)
              | _ => exact ⟨n, [], rfl, rfl⟩
          | store a w v =>
              simp only [shStep]
              split
              · exact ⟨n, [(n, a, w, v)], rfl, rfl⟩
              · exact ⟨n, [], rfl, rfl⟩
        obtain ⟨n', nw, h1, h2⟩ := hstep
        rw [h1, h2]
        obtain ⟨e1, e2⟩ := ih n' (nw ++ acc)
        obtain ⟨f1, f2⟩ := ih n' nw
        exact ⟨by rw [e1, f1, List.append_assoc], by rw [e2, f2]⟩
  simp only
  rw [(key _ _ _).1, List.reverse_append]

theorem sharedStores_call {w : World} {start : Nat} (hs : start ≤ w.obs.length) {c : Intr} (hc : c ≠ .barrier)
    (args : List V) : sharedStores (callW w c args) start = sharedStores w start := by
  rw [sharedStores_ext hs (L := [_]) rfl]
  simp only [List.reverse_cons, List.reverse_nil, List.nil_append, List.foldl_cons, List.foldl_nil]
  have : ∀ s : Nat × List (Nat × UInt64 × Nat × UInt64), shStep s (.call (.intr c) (args ++ callWords w (.intr c) args)) = s := by
    intro s; obtain ⟨n, acc⟩ := s; cases c <;> first | exact absurd rfl hc | rfl
  rw [this]; simp

theorem sharedStores_bar {w : World} {start : Nat} (hs : start ≤ w.obs.length) :
    sharedStores (callW w .barrier []) start = sharedStores w start := by
  rw [sharedStores_ext hs (L := [_]) rfl]
  simp [shStep]

theorem sharedStores_store {w : World} {start : Nat} (hs : start ≤ w.obs.length) {a v : UInt64} {n off : Nat}
    (hd : decodeAddr a = some (.arena, off)) (m : Mem) :
    sharedStores { obsStore w a n v with mem := m } start =
      sharedStores w start ++ [((collCalls w start).countP (·.1 == .barrier), a, n, v)] := by
  rw [sharedStores_ext hs (L := [_]) rfl]
  simp [shStep, hd]

/-! ## One lane's window -/

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

/-- Lane `l`'s table slot. -/
abbrev winSlot (ws : WState) (sp : String) (inStride hashLog tbl l : Nat) : Nat :=
  tbl + 2 * wHash ws.gmem hashLog ((ws.regs "inBase").toNat + min (winPos ws sp l) (inStride - 4))

theorem wHash_lt (g : Array UInt8) (hashLog pos : Nat) : wHash g hashLog pos < 2 ^ hashLog :=
  Nat.mod_lt _ (Nat.two_pow_pos _)

theorem ansW_of {ans : List Answer} {k : Nat} {M : UInt64} (h : ans[k]? = some (.val (.sc .i32 M)))
    (hM : M.toNat < 2 ^ 32) : ansW ans k = M := by
  simp only [ansW, wordAt, h, s32]
  apply UInt64.toNat_inj.mp
  simp [Nat.mod_eq_of_lt hM]

set_option maxHeartbeats 1000000 in
/-- **One lane's window**, its ballot answered `M` and its shuffle `C`
    (whatever its answers hold): the collectives it passes, the store it inserts, shared memory as the barrier
    answers it, every byte of `I` kept, and its registers. -/
theorem lane_window_step {I : UInt64 → Prop} {ws : WState} {st st' : LSt} (hR : Rel I ws st)
    {ans : List Answer} {start : Nat} (hs : start ≤ st.w.obs.length)
    {found p0 cand0 sp : String} {inStride searchLim hashLog tbl tbl' l : Nat}
    (hl : l < 32) (hsl : searchLim + 4 ≤ inStride) (hh : hashLog ≤ 32)
    (hin : ∀ j, j < inStride → I (ws.regs "inBase" + UInt64.ofNat j))
    (hib : (ws.regs "inBase").toNat + inStride < 2 ^ 64) (hsp : (ws.regs sp).toNat + 32 < 2 ^ 64)
    (htbl : ws.regs "tbl" = UInt64.ofNat tbl) (htb : tbl + 2 * 2 ^ hashLog + 2 < 2 ^ 36)
    (hIna : ∀ a, I a → NotArena a)
    (hr : coopL ans start (addrOf .arena 0) (UInt64.ofNat l)
      (.coopWindow found p0 cand0 sp inStride searchLim hashLog tbl') st = some st')
    :
    let M := ansW ans (collCalls st.w start).length
    let C := ansW ans ((collCalls st.w start).length + 2)
    ∃ b, ans[(collCalls st.w start).length + 1]? = some (.smem b) ∧
      collCalls st'.w start = collCalls st.w start ++
        [(.ballot, [boolV (winHit ws sp inStride searchLim hashLog tbl l)]), (.barrier, []),
         (.shflIdx, [.sc .i32 (winCand ws sp inStride hashLog tbl l),
           .sc .i32 ((ctz32 M &&& 0xffffffff) &&& 0xffffffff)])] ∧
      sharedStores st'.w start = sharedStores st.w start ++
        (if UInt64.ofNat l ≤ ctz32 M &&& 0xffffffff ∧ winPos ws sp l < searchLim then
          [((collCalls st.w start).countP (·.1 == .barrier),
            addrOf .arena 0 + UInt64.ofNat (winSlot ws sp inStride hashLog tbl l), 2,
            UInt64.ofNat (winPos ws sp l + 1) &&& 0xffff)] else []) ∧
      (UInt64.ofNat l ≤ ctz32 M &&& 0xffffffff ∧ winPos ws sp l < searchLim →
        winSlot ws sp inStride hashLog tbl l + 2 ≤ st.w.mem.arena.size) ∧
      st'.w.mem.arena = b ∧ (∀ a, NotArena a → st'.w.mem.load a 1 = st.w.mem.load a 1) ∧
      st'.regs = (((ws.setReg p0 (ws.regs sp + (ctz32 M &&& 0xffffffff))).setReg cand0
        (C &&& 0xffffffff)).setReg found (if M != 0 then 1 else 0)).regs := by
  simp only [coopL, hs, if_true] at hr
  obtain ⟨v32, hv32, hr⟩ := Option.bind_eq_some_iff.mp hr
  obtain ⟨candRaw, hcr, hr⟩ := Option.bind_eq_some_iff.mp hr
  obtain ⟨vc, hvc, hr⟩ := Option.bind_eq_some_iff.mp hr
  obtain ⟨w2, hw2, hr⟩ := Option.bind_eq_some_iff.mp hr
  rw [hR.regs] at hv32 hcr hvc hw2 hr
  obtain ⟨hhit, -, hslot, hcandw⟩ := lane_probe hR (l := l) (sp := sp) hsl hh hin hib (by omega) htbl htb
    rfl rfl hv32 rfl hcr rfl rfl hvc
  intro M C
  rw [hhit] at hw2
  -- the ballot's call, and the insert, leave the collectives as the ballot made them
  have hc1 : collCalls (callW st.w .ballot [boolV (winHit ws sp inStride searchLim hashLog tbl l)]) start =
      collCalls st.w start ++ [(.ballot, [boolV (winHit ws sp inStride searchLim hashLog tbl l)])] := by
    rw [callW, collCalls_call hs]; rfl
  have hs1 : start ≤ (callW st.w .ballot [boolV (winHit ws sp inStride searchLim hashLog tbl l)]).obs.length := by
    simp [callW, obsCall_length]; omega
  have hc2 : collCalls w2 start = collCalls st.w start ++
      [(.ballot, [boolV (winHit ws sp inStride searchLim hashLog tbl l)])] := by
    rw [← hc1]
    unfold stwIf at hw2
    split at hw2
    · obtain ⟨m, -, rfl⟩ := stw_some hw2
      exact collCalls_obsStore hs1 _ _ _
    · obtain rfl := (Option.some.inj hw2).symm; rfl
  rw [hc2, List.length_append, List.length_singleton] at hr
  split at hr
  case h_2 => cases hr
  rename_i b hb
  obtain rfl := (Option.some.inj hr).symm
  have hs2 : start ≤ w2.obs.length := Nat.le_trans hs1 (stwIf_obs hw2)
  have hs3 : start ≤ (callW w2 .barrier []).obs.length := by simp [callW, obsCall_length]; omega
  have hc3 : collCalls (callW w2 .barrier []) start = collCalls st.w start ++
      [(.ballot, [boolV (winHit ws sp inStride searchLim hashLog tbl l)]), (.barrier, [])] := by
    rw [callW, collCalls_call hs2, hc2]; simp [Intr.isColl]
  have hposN : (ws.regs sp + UInt64.ofNat l).toNat = winPos ws sp l := by
    rw [UInt64.toNat_add, UInt64.toNat_ofNat_of_lt' (show l < 2 ^ 64 by omega), Nat.mod_eq_of_lt (by omega)]
  have hslotlt : winSlot ws sp inStride hashLog tbl l < 2 ^ 36 := by
    have := wHash_lt ws.gmem hashLog ((ws.regs "inBase").toNat + min (winPos ws sp l) (inStride - 4))
    simp only [winSlot]; omega
  generalize hAddr : ((((v32 * c64 ↑wHashK) >>> c64 (32 - ↑hashLog)) &&& c64 (2 ^ hashLog - 1)) <<< c64 1 +
    ws.regs "tbl") = addr at hslot hw2 hcr
  have haddr : addr = UInt64.ofNat (winSlot ws sp inStride hashLog tbl l) := by
    apply UInt64.toNat_inj.mp; rw [hslot, UInt64.toNat_ofNat_of_lt' (Nat.lt_of_lt_of_le hslotlt (by decide))]
  subst haddr
  have hd : decodeAddr (addrOf .arena 0 + UInt64.ofNat (winSlot ws sp inStride hashLog tbl l)) =
      some (.arena, winSlot ws sp inStride hashLog tbl l) := by
    rw [show addrOf .arena 0 + UInt64.ofNat (winSlot ws sp inStride hashLog tbl l) =
      regionBase .arena + UInt64.ofNat (winSlot ws sp inStride hashLog tbl l) by simp [addrOf]]
    exact decodeAddr_base _ _ (by rw [regionSpan_toNat]; exact hslotlt)
  have hval : (ws.regs sp + UInt64.ofNat l + c64 1) &&& 0xffff = UInt64.ofNat (winPos ws sp l + 1) &&& 0xffff := by
    congr 1
    apply UInt64.toNat_inj.mp
    rw [show c64 1 = UInt64.ofNat 1 by decide, UInt64.toNat_add, hposN, UInt64.toNat_ofNat_of_lt' (by decide),
      UInt64.toNat_ofNat_of_lt' (show winPos ws sp l + 1 < 2 ^ 64 by simp only [winPos]; omega),
      Nat.mod_eq_of_lt (by simp only [winPos]; omega)]
  have hins : (UInt64.ofNat l ≤ ctz32 M &&& 0xffffffff && decide (ws.regs sp + UInt64.ofNat l < c64 ↑searchLim)) =
      decide (UInt64.ofNat l ≤ ctz32 M &&& 0xffffffff ∧ winPos ws sp l < searchLim) := by
    have hlt : (ws.regs sp + UInt64.ofNat l < c64 ↑searchLim) ↔ winPos ws sp l < searchLim := by
      rw [c64_ofNat, UInt64.lt_iff_toNat_lt, hposN, UInt64.toNat_ofNat_of_lt' (show searchLim < 2 ^ 64 by omega)]
    exact Bool.eq_iff_iff.mpr (by simp only [Bool.and_eq_true, decide_eq_true_iff]; exact and_congr Iff.rfl hlt)
  rw [hins, hval] at hw2
  refine ⟨b, hb, ?_, ?_, ?_, rfl, ?_, ?_⟩
  · show collCalls (callW _ .shflIdx _) start = _
    rw [callW, collCalls_call (by simpa using hs3)]
    show collCalls (callW w2 .barrier []) start ++ _ = _
    rw [hc3, hcandw]; simp [Intr.isColl, M]
  · by_cases hc : UInt64.ofNat l ≤ ctz32 M &&& 0xffffffff ∧ winPos ws sp l < searchLim
    · unfold stwIf at hw2
      rw [if_pos (decide_eq_true hc)] at hw2
      obtain ⟨m, -, rfl⟩ := stw_some hw2
      rw [if_pos hc]
      show sharedStores (callW _ .shflIdx _) start = _
      rw [sharedStores_call (by simpa using hs3) (by decide)]
      show sharedStores (callW _ .barrier []) start = _
      rw [sharedStores_bar (by simpa using Nat.le_trans hs1 (Nat.le_succ _)),
        sharedStores_store hs1 hd, sharedStores_call hs (by decide)]
      congr 3
      rw [hc1]; simp [show (Intr.ballot == Intr.barrier) = false by decide]
    · unfold stwIf at hw2
      rw [if_neg (fun h => hc (of_decide_eq_true h))] at hw2
      obtain rfl := (Option.some.inj hw2).symm
      rw [sharedStores_ext hs (L := [_, _, _]) rfl, if_neg hc]
      simp [shStep]
  · intro hc
    unfold stwIf at hw2
    rw [if_pos (decide_eq_true hc)] at hw2
    obtain ⟨m, hm, -⟩ := stw_some hw2
    obtain ⟨r, off, hd', hsz, -⟩ := store_some hm
    rw [hd] at hd'
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hd')
    simpa [Mem.region, callW, obsCall] using hsz
  · intro a ha
    show ({ (w2.mem) with arena := b } : Mem).load a 1 = _
    rw [load_set_arena ha]
    unfold stwIf at hw2
    split at hw2
    · obtain ⟨m, hm, rfl⟩ := stw_some hw2
      show m.load a 1 = _
      rw [store_setArena hd hm, load_set_arena ha]; rfl
    · obtain rfl := (Option.some.inj hw2).symm; rfl
  · have hk : (collCalls { callW w2 .barrier [] with mem := { (callW w2 .barrier []).mem with arena := b } }
        start).length = (collCalls st.w start).length + 2 := by
      show (collCalls (callW w2 .barrier []) start).length = _
      rw [hc3]; simp
    funext x
    simp only [LSt.set, WState.setReg, hk, C, M]

/-! ## The barrier's phase is the model's insert -/

/-- **A lane's share of the window finishes** where the table lies in its
    shared memory and the barrier is answered shared memory. -/
theorem lane_window_some {I : UInt64 → Prop} {ws : WState} {st : LSt} (hR : Rel I ws st)
    {ans : List Answer} {start : Nat} (hs : start ≤ st.w.obs.length)
    {found p0 cand0 sp : String} {inStride searchLim hashLog tbl tbl' l : Nat}
    (hl : l < 32) (hsl : searchLim + 4 ≤ inStride) (hh : hashLog ≤ 32)
    (hin : ∀ j, j < inStride → I (ws.regs "inBase" + UInt64.ofNat j))
    (hib : (ws.regs "inBase").toNat + inStride < 2 ^ 64) (hsp : (ws.regs sp).toNat + 32 < 2 ^ 64)
    (htbl : ws.regs "tbl" = UInt64.ofNat tbl) (htb : tbl + 2 * 2 ^ hashLog + 2 < 2 ^ 36)
    (hsz : tbl + 2 * 2 ^ hashLog ≤ st.w.mem.arena.size)
    (hbar : ∃ b, ans[(collCalls st.w start).length + 1]? = some (.smem b)) :
    ∃ st', coopL ans start (addrOf .arena 0) (UInt64.ofNat l)
      (.coopWindow found p0 cand0 sp inStride searchLim hashLog tbl') st = some st' := by
  have hc4 := c64_sub4 (show 4 ≤ inStride by omega)
  have h64 : inStride - 4 < 2 ^ 64 := by omega
  have hcap : ∀ y : UInt64, (if y ≤ c64 (inStride - 4) then y else c64 (inStride - 4)).toNat + 4 ≤ inStride := by
    intro y
    rw [hc4]
    split
    · rename_i hle; rw [toNat_le_ofNat h64] at hle; omega
    · rw [UInt64.toNat_ofNat_of_lt' h64]; omega
  obtain ⟨v32, hv32⟩ := load4_input_some hR hin (hcap (ws.regs sp + UInt64.ofNat l))
  have hpow : 2 ^ hashLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by decide) hh
  have haddr : ((((v32 * c64 wHashK) >>> c64 (32 - hashLog)) &&& c64 (2 ^ hashLog - 1)) <<< c64 1 +
      ws.regs "tbl").toNat + 2 ≤ tbl + 2 * 2 ^ hashLog := by
    have hm : (((v32 * c64 wHashK) >>> c64 (32 - hashLog)) &&& c64 (2 ^ hashLog - 1)).toNat < 2 ^ hashLog := by
      rw [hash_eq (load4_input hR hin hib (hcap _) hv32) hh]; exact wHash_lt _ _ _
    rw [htbl, show c64 1 = UInt64.ofNat 1 by decide, UInt64.toNat_add, shl_toNat (by decide) (by omega),
      UInt64.toNat_ofNat_of_lt' (show tbl < 2 ^ 64 by omega), Nat.shiftLeft_eq, Nat.mod_eq_of_lt (by omega)]
    omega
  have hsz' : ((((v32 * c64 wHashK) >>> c64 (32 - hashLog)) &&& c64 (2 ^ hashLog - 1)) <<< c64 1 +
      ws.regs "tbl").toNat + 2 ≤ (st.w.mem.region .arena).size := by
    simp only [Mem.region]; omega
  obtain ⟨candRaw, hcr⟩ := load_some (arena_decode (by omega)) hsz' (by decide)
  obtain ⟨vc, hvc⟩ := load4_input_some hR hin (hcap (candRaw + c64 (-1)))
  obtain ⟨b, hb⟩ := hbar
  simp only [coopL, hs, if_true, hR.regs, Option.bind_eq_bind]
  refine bind_some_of hv32 (bind_some_of hcr (bind_some_of hvc ?_))
  refine stwIf_bind_some (by omega) (by simp only [callW, obsCall]; omega) fun w2 hc2 hl2 => ?_
  have hs1 : start ≤ (callW st.w .ballot [boolV
      (decide (ws.regs sp + UInt64.ofNat l < c64 ↑searchLim) && candRaw != c64 0 &&
        (decide (candRaw + c64 (-1) < ws.regs sp + UInt64.ofNat l) && vc == v32))]).obs.length := by
    simp [callW, obsCall_length]; omega
  rw [hc2 start hs1, callW, collCalls_call hs]
  simp only [List.length_append, List.length_singleton, Intr.isColl, if_true, ite_true, List.length_cons,
    List.length_nil, hb]
  exact ⟨_, rfl⟩

/-- **A 2-byte store into shared memory** sets its two bytes. -/
theorem store_arena2 (m : Mem) {s : Nat} (hs : s < 2 ^ 36) (hsz : s + 2 ≤ m.arena.size) (v : UInt64) :
    ∃ m', m.store (addrOf .arena 0 + UInt64.ofNat s) 2 v = some m' ∧
      m'.arena.data = (m.arena.data.setIfInBounds s ((v &&& 0xff).toUInt8)).setIfInBounds (s + 1)
        (((v >>> 8) &&& 0xff).toUInt8) ∧ m'.arena.size = m.arena.size := by
  have hd : decodeAddr (addrOf .arena 0 + UInt64.ofNat s) = some (.arena, s) := by
    rw [show addrOf .arena 0 + UInt64.ofNat s = regionBase .arena + UInt64.ofNat s by simp [addrOf]]
    exact decodeAddr_base _ _ (by rw [regionSpan_toNat]; exact hs)
  refine ⟨m.setRegion .arena ((List.range 2).foldl (fun (b : ByteArray) i =>
    b.set! (s + i) (((v >>> (8 * UInt64.ofNat i)) &&& 0xff).toUInt8)) m.arena), ?_, ?_, ?_⟩
  · unfold Mem.store
    rw [hd]
    simp only [bind, Option.bind, Mem.region, Mem.reachable]
    rw [if_neg (by simp; omega)]
  · simp only [Mem.setRegion]
    rw [show List.range 2 = [0, 1] from rfl]
    simp only [List.foldl_cons, List.foldl_nil]
    rw [show (8 : UInt64) * UInt64.ofNat 0 = 0 by decide, show (8 : UInt64) * UInt64.ofNat 1 = 8 by decide,
      UInt64.shiftRight_zero, Nat.add_zero]
    generalize m.arena = A
    cases A
    simp only [ByteArray.set!, Array.set!_eq_setIfInBounds]
  · simp only [Mem.setRegion]
    rw [show List.range 2 = [0, 1] from rfl]
    simp only [List.foldl_cons, List.foldl_nil]
    generalize m.arena = A
    cases A
    simp [ByteArray.set!, ByteArray.size, Array.set!_eq_setIfInBounds]

theorem byte_lo {val : Nat} (hv : val < 2 ^ 64) : ((UInt64.ofNat val &&& 0xffff) &&& 0xff).toUInt8 = UInt8.ofNat (val % 256) := by
  apply UInt8.toNat_inj.mp
  simp only [UInt64.toNat_toUInt8, UInt64.toNat_and, UInt64.toNat_ofNat_of_lt' hv, UInt8.toNat_ofNat']
  rw [show (0xffff : UInt64).toNat = 2 ^ 16 - 1 from rfl, show (0xff : UInt64).toNat = 2 ^ 8 - 1 from rfl,
    Nat.and_two_pow_sub_one_eq_mod, Nat.and_two_pow_sub_one_eq_mod]
  omega

theorem byte_hi {val : Nat} (hv : val < 2 ^ 64) :
    (((UInt64.ofNat val &&& 0xffff) >>> 8) &&& 0xff).toUInt8 = UInt8.ofNat (val / 256 % 256) := by
  apply UInt8.toNat_inj.mp
  simp only [UInt64.toNat_toUInt8, UInt64.toNat_and, UInt64.toNat_shiftRight, UInt64.toNat_ofNat_of_lt' hv,
    UInt8.toNat_ofNat']
  rw [show (0xffff : UInt64).toNat = 2 ^ 16 - 1 from rfl, show (0xff : UInt64).toNat = 2 ^ 8 - 1 from rfl,
    Nat.and_two_pow_sub_one_eq_mod, Nat.and_two_pow_sub_one_eq_mod, show (8 : UInt64).toNat % 64 = 8 from rfl,
    Nat.shiftRight_eq_div_pow]
  omega

/-- **The barrier's phase writes the lanes' inserts in lane order**, each its
    two bytes. -/
theorem phase_inserts {all : List (Nat × UInt64 × Nat × UInt64)} {NB : Nat} {c : Mem} {ins : Nat → Bool}
    {slot val : Nat → Nat}
    (hall : all.filter (·.1 = NB) = (List.range 32).flatMap (fun l => if ins l then
      [(NB, addrOf .arena 0 + UInt64.ofNat (slot l), 2, UInt64.ofNat (val l) &&& 0xffff)] else []))
    (hsz : ∀ l, l < 32 → ins l = true → slot l + 2 ≤ c.arena.size) (hslt : ∀ l, l < 32 → slot l < 2 ^ 36)
    (hv : ∀ l, l < 32 → val l < 2 ^ 64) :
    (phase all NB c).arena.data = (List.range 32).foldl (fun acc l => if ins l then
      (acc.setIfInBounds (slot l) (UInt8.ofNat (val l % 256))).setIfInBounds (slot l + 1)
        (UInt8.ofNat (val l / 256 % 256)) else acc) c.arena.data := by
  rw [phase_filter, hall]
  have key : ∀ (L : List Nat) (c : Mem), (∀ l ∈ L, l < 32) → c.arena.size = c.arena.size →
      (∀ l, l < 32 → ins l = true → slot l + 2 ≤ c.arena.size) →
      ((L.flatMap fun l => if ins l then
          [(NB, addrOf .arena 0 + UInt64.ofNat (slot l), 2, UInt64.ofNat (val l) &&& 0xffff)] else []).foldl
        (fun c x => (c.store x.2.1 x.2.2.1 x.2.2.2).getD c) c).arena.data =
      L.foldl (fun acc l => if ins l then
        (acc.setIfInBounds (slot l) (UInt8.ofNat (val l % 256))).setIfInBounds (slot l + 1)
          (UInt8.ofNat (val l / 256 % 256)) else acc) c.arena.data := by
    intro L
    induction L with
    | nil => intro c _ _ _; rfl
    | cons l L ih =>
        intro c hL _ hsz
        rw [List.flatMap_cons, List.foldl_append, List.foldl_cons]
        by_cases hi : ins l = true
        · rw [if_pos hi, if_pos hi]
          obtain ⟨m', hm', hd', hs'⟩ := store_arena2 c (hslt l (hL l (by simp))) (hsz l (hL l (by simp)) hi)
            (UInt64.ofNat (val l) &&& 0xffff)
          simp only [List.foldl_cons, List.foldl_nil, hm', Option.getD_some]
          rw [ih m' (fun x hx => hL x (by simp [hx])) rfl (fun x hx hix => by rw [hs']; exact hsz x hx hix), hd',
            byte_lo (hv l (hL l (by simp))), byte_hi (hv l (hL l (by simp)))]
        · rw [if_neg hi, if_neg hi]
          exact ih c (fun x hx => hL x (by simp [hx])) rfl hsz
  exact key _ c (fun l hl => List.mem_range.mp hl) rfl hsz

/-- **A lane's stores in a phase are those it made in that phase**: what came
    before is tagged lower, what comes after higher. -/
theorem lane_tagNB {w0 w1 wf : World} {start NB : Nat} {X : List (Nat × UInt64 × Nat × UInt64)}
    (h01 : sharedStores w1 start = sharedStores w0 start ++ X) (hX : ∀ x ∈ X, x.1 = NB)
    (htag : ∀ x ∈ sharedStores w0 start, x.1 < NB)
    (hcount : (collCalls w1 start).countP (·.1 == .barrier) = NB + 1)
    (hs1 : start ≤ w1.obs.length) (hext : WExt w1 wf) :
    (sharedStores wf start).filter (·.1 = NB) = X := by
  obtain ⟨L, hL⟩ := hext
  rw [sharedStores_ext hs1 hL, h01, hcount]
  obtain ⟨nw, hnw, hge⟩ := shStep_fold_new L.reverse (NB + 1) []
  rw [hnw, List.append_nil, List.filter_append, List.filter_append]
  have e1 : (sharedStores w0 start).filter (·.1 = NB) = [] :=
    List.filter_eq_nil_iff.mpr fun x hx => by have := htag x hx; simp; omega
  have e2 : X.filter (·.1 = NB) = X := List.filter_eq_self.mpr fun x hx => by simp [hX x hx]
  have e3 : nw.reverse.filter (·.1 = NB) = [] :=
    List.filter_eq_nil_iff.mpr fun x hx => by have := hge x (List.mem_reverse.mp hx); simp; omega
  rw [e1, e2, e3]; simp

/-! ## The warp's window -/

theorem collCalls_prefix {w wf : World} {start : Nat} (hs : start ≤ w.obs.length) (h : WExt w wf) :
    ∃ R, collCalls wf start = collCalls w start ++ R := by
  obtain ⟨L, hL⟩ := h
  refine ⟨L.reverse.filterMap collOf, ?_⟩
  unfold collCalls
  rw [hL, List.reverse_append, List.drop_append_of_le_length (by simpa using hs), List.filterMap_append]

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

theorem lane32_cand (x y : UInt64) (hx : x.toNat < 2 ^ 32) : lane32 [.sc .i32 x, .sc .i32 y] = .sc .i32 x := by
  simp only [lane32]
  congr 1
  apply UInt64.toNat_inj.mp
  simp [Nat.mod_eq_of_lt hx]

theorem foldl_congr_range {α : Type} {f g : α → Nat → α} {n : Nat} (h : ∀ l, l < n → ∀ a, f a l = g a l) (a : α) :
    (List.range n).foldl f a = (List.range n).foldl g a := by
  have key : ∀ (L : List Nat) (a : α), (∀ l ∈ L, l < n) → L.foldl f a = L.foldl g a := by
    intro L
    induction L with
    | nil => intro a _; rfl
    | cons x L ih =>
        intro a hL
        rw [List.foldl_cons, List.foldl_cons, h x (hL x (by simp)), ih _ (fun y hy => hL y (by simp [hy]))]
  exact key _ a (fun l hl => List.mem_range.mp hl)

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

/-- What the warp's window answers its three collectives: the ballot the mask
    of the lanes' hits, the barrier the table after the window's inserts, the
    shuffle the first hitting lane's candidate (lane 0's where none hits). -/
def winAns (ws : WState) (sp : String) (inStride searchLim hashLog tbl : Nat) : List Answer :=
  [.val (.sc .i32 (UInt32.ofNat (maskN (winHit ws sp inStride searchLim hashLog tbl))).toUInt64),
   .smem ⟨tableInsert ws.smem ws.gmem hashLog tbl searchLim (ws.regs "inBase").toNat (ws.regs sp).toNat
     (firstSetNat (winHit ws sp inStride searchLim hashLog tbl) 32 0)⟩,
   .val (.sc .i32 (winCand ws sp inStride hashLog tbl
     (firstSetNat (winHit ws sp inStride searchLim hashLog tbl) 32 0 % 32)))]

theorem getElem?_of_drop {α : Type} {l T R : List α} {K i : Nat} {a : α} (h : l.drop K = T ++ R)
    (hi : T[i]? = some a) : l[K + i]? = some a := by
  obtain ⟨hlt, -⟩ := List.getElem?_eq_some_iff.mp hi
  rw [← List.getElem?_drop, h, List.getElem?_append_left hlt, hi]

set_option maxHeartbeats 4000000 in
/-- What lane `l` passes in the warp's window: the ballot of its hit, the
    barrier, the shuffle of its candidate from the first hitting lane. -/
def winCalls (ws : WState) (sp : String) (inStride searchLim hashLog tbl l : Nat) : List (Intr × List V) :=
  [(.ballot, [boolV (winHit ws sp inStride searchLim hashLog tbl l)]), (.barrier, []),
   (.shflIdx, [.sc .i32 (winCand ws sp inStride hashLog tbl l),
     .sc .i32 (UInt64.ofNat (firstSetNat (winHit ws sp inStride searchLim hashLog tbl) 32 0) &&& 0xffffffff)])]

/-- The shared store lane `l` makes in the warp's window, tagged `nb`: its
    insert, where it is at or below the first hitting lane and in the window. -/
def winStores (ws : WState) (sp : String) (inStride searchLim hashLog tbl l nb : Nat) :
    List (Nat × UInt64 × Nat × UInt64) :=
  if UInt64.ofNat l ≤ UInt64.ofNat (firstSetNat (winHit ws sp inStride searchLim hashLog tbl) 32 0) ∧
      winPos ws sp l < searchLim then
    [(nb, addrOf .arena 0 + UInt64.ofNat (winSlot ws sp inStride hashLog tbl l), 2,
      UInt64.ofNat (winPos ws sp l + 1) &&& 0xffff)]
  else []

set_option maxHeartbeats 4000000 in
/-- **A lane's share of the window runs with the warp**: answered the warp's
    answers (`winAns`), the lane finishes agreeing with the warp's window, its
    collectives and shared store the warp's for it (`winCalls`, `winStores`). -/
theorem lane_window {I : UInt64 → Prop} {ws : WState} {st : LSt} (hR : Rel I ws st)
    {ans : List Answer} {start : Nat} (hs : start ≤ st.w.obs.length)
    {found p0 cand0 sp : String} {inStride searchLim hashLog tbl l f : Nat} (hl : l < 32)
    (hsl : searchLim + 4 ≤ inStride) (hh : hashLog ≤ 32)
    (hin : ∀ j, j < inStride → I (ws.regs "inBase" + UInt64.ofNat j))
    (hib : (ws.regs "inBase").toNat + inStride < 2 ^ 64) (hsp : (ws.regs sp).toNat + 32 < 2 ^ 64)
    (htbl : ws.regs "tbl" = UInt64.ofNat tbl) (htb : tbl + 2 * 2 ^ hashLog + 2 < 2 ^ 36)
    (hIna : ∀ a, I a → NotArena a) (hfp : found ≠ p0) (hfc : found ≠ cand0) (hpc : p0 ≠ cand0)
    (hsz : tbl + 2 * 2 ^ hashLog ≤ ws.smem.size) {R : List Answer}
    (htape : ans.drop (collCalls st.w start).length = winAns ws sp inStride searchLim hashLog tbl ++ R) :
    ∃ st', coopL ans start (addrOf .arena 0) (UInt64.ofNat l)
        (.coopWindow found p0 cand0 sp inStride searchLim hashLog tbl) st = some st' ∧
      Rel I ((WStmt.coopWindow found p0 cand0 sp inStride searchLim hashLog tbl).eval f ws) st' ∧
      collCalls st'.w start = collCalls st.w start ++ winCalls ws sp inStride searchLim hashLog tbl l ∧
      sharedStores st'.w start = sharedStores st.w start ++
        winStores ws sp inStride searchLim hashLog tbl l (bars (collCalls st.w start)) ∧
      ans.drop (collCalls st'.w start).length = R ∧ Tracks st.w st'.w := by
  have hbar : ∃ b, ans[(collCalls st.w start).length + 1]? = some (.smem b) :=
    ⟨_, getElem?_of_drop (i := 1) htape rfl⟩
  obtain ⟨st', hc⟩ := lane_window_some hR hs hl hsl hh hin hib hsp htbl htb
    (by show _ ≤ st.w.mem.arena.data.size; rw [hR.smem]; exact hsz) hbar
  obtain ⟨b, hb, hcl, hsh, -, harena, hgin, hregs⟩ := lane_window_step hR hs hl hsl hh hin hib hsp htbl htb hIna hc
  obtain ⟨K, hK⟩ : ∃ K, (collCalls st.w start).length = K := ⟨_, rfl⟩
  rw [hK] at htape hb hregs
  simp only [hK] at hcl hsh
  have hballot : ans[K]? = some (.val (.sc .i32
      (UInt32.ofNat (maskN (winHit ws sp inStride searchLim hashLog tbl))).toUInt64)) :=
    by simpa using getElem?_of_drop (i := 0) htape rfl
  have hMk : ansW ans K = (UInt32.ofNat (maskN (winHit ws sp inStride searchLim hashLog tbl))).toUInt64 :=
    ansW_of hballot (by simpa using (UInt32.ofNat (maskN (winHit ws sp inStride searchLim hashLog tbl))).toNat_lt)
  have hfs := firstSetNat_le (winHit ws sp inStride searchLim hashLog tbl)
  have hfirst : ctz32 (UInt32.ofNat (maskN (winHit ws sp inStride searchLim hashLog tbl))).toUInt64 &&& 0xffffffff =
      UInt64.ofNat (firstSetNat (winHit ws sp inStride searchLim hashLog tbl) 32 0) := by
    rw [ctz32_mask]
    apply UInt64.toNat_inj.mp
    rw [UInt64.toNat_and, UInt64.toNat_ofNat_of_lt'
        (show firstSetNat (winHit ws sp inStride searchLim hashLog tbl) 32 0 < 2 ^ 64 by omega),
      show (0xffffffff : UInt64).toNat = 2 ^ 32 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt (by omega)]
  have hbar' : ans[K + 1]? = some (.smem ⟨tableInsert ws.smem ws.gmem hashLog tbl searchLim
      (ws.regs "inBase").toNat (ws.regs sp).toNat (firstSetNat (winHit ws sp inStride searchLim hashLog tbl) 32 0)⟩) :=
    getElem?_of_drop (i := 1) htape rfl
  have hshfl : ans[K + 2]? = some (.val (.sc .i32 (winCand ws sp inStride hashLog tbl
      (firstSetNat (winHit ws sp inStride searchLim hashLog tbl) 32 0 % 32)))) :=
    getElem?_of_drop (i := 2) htape rfl
  rw [hbar'] at hb
  have hbb : b = ⟨tableInsert ws.smem ws.gmem hashLog tbl searchLim (ws.regs "inBase").toNat (ws.regs sp).toNat
      (firstSetNat (winHit ws sp inStride searchLim hashLog tbl) 32 0)⟩ := by
    have := Option.some.inj hb; cases this; rfl
  subst hbb
  rw [hMk, hfirst, ansW_of hshfl winCand_lt] at hregs
  rw [hMk, hfirst] at hcl hsh
  -- the warp's window, as the lanes find it
  have hfsN : firstSetNat (winHit ws sp inStride searchLim hashLog tbl) 32 0 =
      (AlgorithmLib.LZ4WarpSched.firstHit (winHit ws sp inStride searchLim hashLog tbl) 32 0).getD 32 :=
    firstSetNat_warp _
  have hwin := window_of_firstHit (inp := gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
    (oracle := tableOracle ws.gmem ws.smem hashLog tbl (ws.regs "inBase").toNat) (searchLim := searchLim)
    (s := (ws.regs sp).toNat)
  have hevsmem : ((WStmt.coopWindow found p0 cand0 sp inStride searchLim hashLog tbl).eval f ws).smem =
      tableInsert ws.smem ws.gmem hashLog tbl searchLim (ws.regs "inBase").toNat (ws.regs sp).toNat
        (firstSetNat (winHit ws sp inStride searchLim hashLog tbl) 32 0) := by
    rw [WStmt.eval.eq_11, evalCoopWindow_eq_go, evalCoopWindowGo_smem, hfsN]
    cases e : AlgorithmLib.LZ4WarpSched.firstHit (winHit ws sp inStride searchLim hashLog tbl) 32 0 with
    | none => rw [hwin.2 e]; rfl
    | some L =>
        obtain ⟨c, -, hw⟩ := hwin.1 L e
        rw [hw]; simp
  have hevgmem : ((WStmt.coopWindow found p0 cand0 sp inStride searchLim hashLog tbl).eval f ws).gmem = ws.gmem := by
    rw [WStmt.eval.eq_11, evalCoopWindow_eq_go, evalCoopWindowGo_gmem]
  refine ⟨st', hc, ⟨?_, ?_, ?_⟩, by rw [hcl]; rfl, by rw [hsh]; rfl, ?_,
    Tracks.of_nobyte ((coopL_ext _ _ _ hc).2 rfl) hgin⟩
  · rw [hregs]
    funext x
    rw [WStmt.eval.eq_11, evalCoopWindow_eq_go, evalCoopWindowGo_regs]
    have hs32 : ∀ L, L < 32 → (ws.regs sp) + UInt64.ofNat L = UInt64.ofNat ((ws.regs sp).toNat + L) := by
      intro L _; rw [UInt64.ofNat_add, UInt64.ofNat_toNat]
    cases e : AlgorithmLib.LZ4WarpSched.firstHit (winHit ws sp inStride searchLim hashLog tbl) 32 0 with
    | none =>
        rw [hwin.2 e]
        have hall : ∀ j, j < 32 → winHit ws sp inStride searchLim hashLog tbl j = false :=
          fun j hj => firstHit_none e j (Nat.zero_le _) (by omega)
        have hm0 : maskN (winHit ws sp inStride searchLim hashLog tbl) = 0 := maskN_eq_zero hall
        have hfs32 : firstSetNat (winHit ws sp inStride searchLim hashLog tbl) 32 0 = 32 := by rw [hfsN, e]; rfl
        have hc0 : winCand ws sp inStride hashLog tbl 0 &&& 0xffffffff =
            (tableOracle ws.gmem ws.smem hashLog tbl (ws.regs "inBase").toNat
              (min (ws.regs sp).toNat (inStride - 4))).elim 0xffffffff UInt64.ofNat := by
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
        have hfsL : firstSetNat (winHit ws sp inStride searchLim hashLog tbl) 32 0 = L := by rw [hfsN, e]; rfl
        have hmne : (UInt32.ofNat (maskN (winHit ws sp inStride searchLim hashLog tbl))).toUInt64 != 0 := by
          have hne := maskN_ne_zero (h := winHit ws sp inStride searchLim hashLog tbl) (by omega) hL
          have hlt := maskN_lt (winHit ws sp inStride searchLim hashLog tbl)
          simp only [bne_iff_ne, ne_eq]
          intro h0
          apply hne
          have := congrArg UInt64.toNat h0
          simpa [UInt32.toNat_ofNat_of_lt' hlt] using this
        obtain ⟨hor, hsl'⟩ := probe_some_inv hpc'
        have hcl' := tableOracle_lt hor
        have hcand : winCand ws sp inStride hashLog tbl (L % 32) &&& 0xffffffff = UInt64.ofNat c := by
          rw [winCand_mask, Nat.mod_eq_of_lt (by omega)]
          simp only [winCand, winPos, winOracle]
          rw [show min ((ws.regs sp).toNat + L) (inStride - 4) = (ws.regs sp).toNat + L by omega, hor]
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
  · rw [harena, hevsmem]
  · intro a ha
    rw [hgin a (hIna a ha), hR.gin a ha, hevgmem]
  · rw [hcl, List.length_append, ← List.drop_drop, hK]
    show (ans.drop K).drop 3 = R
    rw [htape]; rfl

set_option maxHeartbeats 4000000 in
/-- **The laws give the window's answers back**: where every lane has passed
    the warp's window as `winCalls` says after `K` collectives and `NB`
    barriers, its stores tagged `NB` the window's (`winStores`), and the warp's
    table is shared memory after the first `NB` phases, the laws answer each
    lane's three collectives with the warp's answers (`winAns`), and the
    table after the window is shared memory after one more phase. -/
theorem window_laws {ws : WState} {sp found p0 cand0 : String} {inStride searchLim hashLog tbl K NB f : Nat}
    {smem0 : ByteArray} {F : Nat → List (Intr × List V)} {S : Nat → List (Nat × UInt64 × Nat × UInt64)}
    (hF : ∀ l, l < 32 → ∃ P Q, F l = P ++ winCalls ws sp inStride searchLim hashLog tbl l ++ Q ∧
      P.length = K ∧ bars P = NB)
    (hS : ∀ l, l < 32 → (S l).filter (·.1 = NB) = winStores ws sp inStride searchLim hashLog tbl l NB)
    (hsnap : ws.smem = (snapMem ((List.range 32).flatMap S) smem0 NB).arena.data)
    (hsl : searchLim + 4 ≤ inStride) (hsp : (ws.regs sp).toNat + 32 < 2 ^ 64)
    (htb : tbl + 2 * 2 ^ hashLog + 2 < 2 ^ 36) (hsz : tbl + 2 * 2 ^ hashLog ≤ ws.smem.size) :
    (∀ l, l < 32 → ∀ i a, (winAns ws sp inStride searchLim hashLog tbl)[i]? = some a →
      LawAt F (snapshots 32 smem0 F S) smem0 l (K + i) a) ∧
    ((WStmt.coopWindow found p0 cand0 sp inStride searchLim hashLog tbl).eval f ws).smem =
      (snapMem ((List.range 32).flatMap S) smem0 (NB + 1)).arena.data := by
  simp only [winCalls] at hF
  simp only [winStores] at hS
  generalize hhit : winHit ws sp inStride searchLim hashLog tbl = hit at hF hS
  have hfs := firstSetNat_le hit
  have hFK : ∀ l, l < 32 → (F l)[K]? = some (.ballot, [boolV (hit l)]) := by
    intro l hl
    obtain ⟨P, Q, hP, hK, -⟩ := hF l hl
    rw [hP, List.append_assoc, List.getElem?_append_right (by omega), hK, Nat.sub_self]; rfl
  have hFK1 : ∀ l, l < 32 → (F l)[K + 1]? = some (.barrier, []) := by
    intro l hl
    obtain ⟨P, Q, hP, hK, -⟩ := hF l hl
    rw [hP, List.append_assoc, List.getElem?_append_right (by omega), hK]; simp
  have hFK2 : ∀ l, l < 32 → (F l)[K + 2]? = some (.shflIdx, [.sc .i32 (winCand ws sp inStride hashLog tbl l),
      .sc .i32 (UInt64.ofNat (firstSetNat hit 32 0) &&& 0xffffffff)]) := by
    intro l hl
    obtain ⟨P, Q, hP, hK, -⟩ := hF l hl
    rw [hP, List.append_assoc, List.getElem?_append_right (by omega), hK]; simp
  -- the ballot
  have hlaw0 : ∀ l, l < 32 → LawAt F (snapshots 32 smem0 F S) smem0 l K
      (.val (.sc .i32 (UInt32.ofNat (maskN hit)).toUInt64)) :=
    fun l hl => ⟨_, _, hFK l hl, answerStep_ballot hl hFK _⟩
  -- the shuffle
  have hm : (UInt64.ofNat (firstSetNat hit 32 0) &&& 0xffffffff).toNat % 32 = firstSetNat hit 32 0 % 32 := by
    rw [UInt64.toNat_and, UInt64.toNat_ofNat_of_lt' (show firstSetNat hit 32 0 < 2 ^ 64 by omega),
      show (0xffffffff : UInt64).toNat = 2 ^ 32 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod,
      Nat.mod_eq_of_lt (show firstSetNat hit 32 0 < 2 ^ 32 by omega)]
  have hlaw2 : ∀ l, l < 32 → LawAt F (snapshots 32 smem0 F S) smem0 l (K + 2)
      (.val (.sc .i32 (winCand ws sp inStride hashLog tbl (firstSetNat hit 32 0 % 32)))) := by
    intro l hl
    have harg : laneArg F (l / 32) ((UInt64.ofNat (firstSetNat hit 32 0) &&& 0xffffffff).toNat % 32) (K + 2) .shflIdx =
        .ok [.sc .i32 (winCand ws sp inStride hashLog tbl (firstSetNat hit 32 0 % 32)),
          .sc .i32 (UInt64.ofNat (firstSetNat hit 32 0) &&& 0xffffffff)] := by
      rw [hm, show l / 32 = 0 by omega]
      unfold laneArg
      rw [show 32 * 0 + firstSetNat hit 32 0 % 32 = firstSetNat hit 32 0 % 32 by omega,
        hFK2 _ (Nat.mod_lt _ (by decide))]
      simp [show (Intr.shflIdx == Intr.shflIdx) = true by decide]
    refine ⟨_, ((F l).take (K + 2)).countP (·.1 == .barrier), hFK2 l hl, ?_⟩
    simp only [answerStep, bind, Except.bind, harg, pure, Except.pure]
    rw [lane32_cand _ _ winCand_lt]; rfl
  -- the next phase is the warp's insert
  have hsize : (snapMem ((List.range 32).flatMap S) smem0 NB).arena.size = ws.smem.size := by
    show (snapMem ((List.range 32).flatMap S) smem0 NB).arena.data.size = _
    rw [hsnap]
  have hphase : (snapMem ((List.range 32).flatMap S) smem0 (NB + 1)).arena.data =
      tableInsert ws.smem ws.gmem hashLog tbl searchLim (ws.regs "inBase").toNat (ws.regs sp).toNat
        (firstSetNat hit 32 0) := by
    show (phase _ NB _).arena.data = _
    rw [phase_inserts (ins := fun l => decide (UInt64.ofNat l ≤ UInt64.ofNat (firstSetNat hit 32 0) ∧
        winPos ws sp l < searchLim)) (slot := winSlot ws sp inStride hashLog tbl)
        (val := fun l => winPos ws sp l + 1)]
    · rw [← hsnap]
      unfold tableInsert
      apply foldl_congr_range
      intro l hl acc
      have hle : (UInt64.ofNat l ≤ UInt64.ofNat (firstSetNat hit 32 0)) ↔ l ≤ firstSetNat hit 32 0 := by
        rw [UInt64.le_iff_toNat_le, UInt64.toNat_ofNat_of_lt' (show l < 2 ^ 64 by omega),
          UInt64.toNat_ofNat_of_lt' (show firstSetNat hit 32 0 < 2 ^ 64 by omega)]
      by_cases hc : l ≤ firstSetNat hit 32 0 ∧ (ws.regs sp).toNat + l < searchLim
      · have hmin : min ((ws.regs sp).toNat + l) (inStride - 4) = (ws.regs sp).toNat + l := by omega
        rw [if_pos (by simpa [hle] using hc), if_pos hc]
        simp only [winSlot, winPos, hmin]
      · rw [if_neg (by simpa [hle] using hc), if_neg hc]
    · rw [List.filter_flatMap]
      exact flatMap_congr_mem fun l hl => by
        rw [hS l (List.mem_range.mp hl)]; simp
    · intro l hl _
      rw [hsize]
      have := wHash_lt ws.gmem hashLog ((ws.regs "inBase").toNat + min (winPos ws sp l) (inStride - 4))
      simp only [winSlot]; omega
    · intro l _
      have := wHash_lt ws.gmem hashLog ((ws.regs "inBase").toNat + min (winPos ws sp l) (inStride - 4))
      simp only [winSlot]; omega
    · intro l hl; simp only [winPos]; omega
  -- the barrier
  have hlaw1 : ∀ l, l < 32 → LawAt F (snapshots 32 smem0 F S) smem0 l (K + 1)
      (.smem ⟨tableInsert ws.smem ws.gmem hashLog tbl searchLim (ws.regs "inBase").toNat (ws.regs sp).toNat
        (firstSetNat hit 32 0)⟩) := by
    intro l hl
    obtain ⟨P, Q, hP, hK, hNB⟩ := hF l hl
    have hnb : ((F l).take (K + 1)).countP (·.1 == .barrier) = NB := by
      rw [hP, List.append_assoc, List.take_append, List.take_of_length_le (by omega),
        hK, show K + 1 - K = 1 by omega]
      simp [List.countP_append, ← hNB, show (Intr.ballot == Intr.barrier) = false by decide]
    have hlen : NB < ((F 0).filter (·.1 == .barrier)).length := by
      obtain ⟨P0, Q0, hP0, -, hNB0⟩ := hF 0 (by decide)
      have hNB0' : P0.countP (·.1 == .barrier) = NB := hNB0
      rw [hP0, List.filter_append, List.filter_append, List.length_append, List.length_append,
        ← List.countP_eq_length_filter, hNB0']
      simp only [List.filter_cons, List.filter_nil, show (Intr.ballot == Intr.barrier) = false by decide,
        show (Intr.barrier == Intr.barrier) = true by decide, show (Intr.shflIdx == Intr.barrier) = false by decide,
        if_true, Bool.false_eq_true, if_false, List.length_cons, List.length_nil]
      omega
    have hsn := snapshots_get (n := 32) (smem0 := smem0) (calls := F) (stores := S) (by decide : 0 < 32) hlen
    have harena : (snapMem ((List.range 32).flatMap S) smem0 (NB + 1)).arena =
        ⟨tableInsert ws.smem ws.gmem hashLog tbl searchLim (ws.regs "inBase").toNat (ws.regs sp).toNat
          (firstSetNat hit 32 0)⟩ := by
      rw [← hphase]
    refine ⟨_, NB + 1, hFK1 l hl, ?_⟩
    rw [hnb]
    simp only [answerStep, pure, Except.pure, hsn, Option.getD_some, harena]; rfl
  -- the warp's table after the window
  have hfsN : firstSetNat (winHit ws sp inStride searchLim hashLog tbl) 32 0 =
      (AlgorithmLib.LZ4WarpSched.firstHit (winHit ws sp inStride searchLim hashLog tbl) 32 0).getD 32 :=
    firstSetNat_warp _
  have hwin := window_of_firstHit (inp := gmemInpAt ws.gmem (ws.regs "inBase").toNat inStride)
    (oracle := tableOracle ws.gmem ws.smem hashLog tbl (ws.regs "inBase").toNat) (searchLim := searchLim)
    (s := (ws.regs sp).toNat)
  have hevsmem : ((WStmt.coopWindow found p0 cand0 sp inStride searchLim hashLog tbl).eval f ws).smem =
      tableInsert ws.smem ws.gmem hashLog tbl searchLim (ws.regs "inBase").toNat (ws.regs sp).toNat
        (firstSetNat hit 32 0) := by
    rw [WStmt.eval.eq_11, evalCoopWindow_eq_go, evalCoopWindowGo_smem, ← hhit, hfsN]
    cases e : AlgorithmLib.LZ4WarpSched.firstHit (winHit ws sp inStride searchLim hashLog tbl) 32 0 with
    | none => rw [hwin.2 e]; rfl
    | some L =>
        obtain ⟨c, -, hw⟩ := hwin.1 L e
        rw [hw]; simp
  refine ⟨fun l hl i a hi => ?_, by rw [hevsmem, ← hphase]⟩
  rw [winAns, hhit] at hi
  match i, hi with
  | 0, hi => cases hi; exact hlaw0 l hl
  | 1, hi => cases hi; exact hlaw1 l hl
  | 2, hi => cases hi; exact hlaw2 l hl
  | _ + 3, hi => simp at hi


end Lz4Code
