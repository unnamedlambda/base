module
public import Lz4.CodeFacts
public import Lz4.CompressorFrame
public import AlgorithmLib.ML.Ptx.LaneShared
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lz4.CompressorWarp` — the warp's run of the compressor's translation

The warp's run (`EWStmt.run` at `F32`) of LZ4's warp statement's
translation (`ofW`) agrees with the model's (`WStmt.eval`): each register the
statement names holds the model's in every lane, the lane register the lane,
the input bytes and the table the model's (`SimW`). Along the way it makes
only what `lane_warp` asks of a lane (`OkW`, with the input bytes as what
lanes read and the output bytes as what they store), shared memory is clean
at every statement's end, its one-byte stores are each lane's share of the
model's (`lanePart`, `gstOf`), and every shared store is followed by a barrier
(`Paired`): `sim`.
-/

namespace Lz4Code

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Prog
  AlgorithmLib.LZ4WarpDSL AlgorithmLib.LibSpec AlgorithmLib.Kernel AlgorithmLib.ML
  AlgorithmLib.ML.BodyLane AlgorithmLib.ML.LaneWarp AlgorithmLib.ML.LaneSettle AlgorithmLib.ML.LaneShared

section
variable (ns : List String) (I : UInt64 → Prop)

/-- **The warp agrees with the model**: every register of `ns` the model's in
    every lane, the lane register the lane, the input bytes and the table the
    model's. -/
structure SimW (ws : WSt F32) (W : WState) : Prop where
  reg : ∀ n ∈ ns, ∀ l : Lane, ws.int.regs (ns.idxOf n) l = W.regs n
  lane : ∀ l : Lane, ws.int.regs (laneR ns.length) l = UInt64.ofNat l.val
  gin : ∀ a, I a → ws.int.gmem a.toNat = W.gmem.getD a.toNat 0
  smem : ∀ k, k < W.smem.size → ws.int.smem k = W.smem.getD k 0

end

/-- The model's predicate registers hold bits. -/
def BitsW (fl : List String) (W : WState) : Prop := ∀ r, fl.contains r = true → W.regs r ≤ 1

/-- The one-byte stores among stores. -/
def byteSt (L : List (Option (UInt64 × Nat × UInt64))) : List (UInt64 × UInt64) :=
  L.filterMap fun
    | some (a, 1, v) => some (a, v)
    | _ => none

theorem byteSt_append (A B : List (Option (UInt64 × Nat × UInt64))) : byteSt (A ++ B) = byteSt A ++ byteSt B :=
  List.filterMap_append

theorem lanePart_append (l : Nat) (A B : List (Option Nat × UInt64 × UInt64)) :
    lanePart l (A ++ B) = lanePart l A ++ lanePart l B := List.filterMap_append

/-- Every shared store is followed by a barrier. -/
def Paired (E : List ShEv) : Prop :=
  ∀ k ws a v g, E[k]? = some (.st ws a v g) → ∃ ws', E[k + 1]? = some (.bar ws')

theorem Paired.nil : Paired [] := fun k _ _ _ _ h => by simp at h

theorem Paired.append {A B : List ShEv} (hA : Paired A) (hB : Paired B) : Paired (A ++ B) := by
  intro k ws a v g h
  by_cases hk : k < A.length
  · rw [List.getElem?_append_left hk] at h
    obtain ⟨ws', h'⟩ := hA k ws a v g h
    exact ⟨ws', by rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp h').1]; exact h'⟩
  · rw [List.getElem?_append_right (by omega)] at h
    obtain ⟨ws', h'⟩ := hB _ ws a v g h
    exact ⟨ws', by rw [List.getElem?_append_right (by omega), show k + 1 - A.length = k - A.length + 1 by omega]; exact h'⟩

theorem eval_bits {fl : List String} : ∀ (s : WStmt), Wf fl s → ∀ (f : Nat) (W : WState), BitsW fl W →
    BitsW fl (s.eval f W) := by
  intro s
  induction s with
  | seq a b iha ihb =>
      intro hw f W hb
      simp only [WStmt.eval]
      exact ihb (fun r h => hw r (by simp [assigns, h])) f _ (iha (fun r h => hw r (by simp [assigns, h])) f W hb)
  | bin o d a b | mov d a =>
      intro hw f W hb r hr
      by_cases e : r = d
      · subst e; rw [hw r (by simp [assigns])] at hr; cases hr
      · simpa [WStmt.eval, WState.setReg, e] using hb r hr
  | setp c d a b =>
      intro _ f W hb r hr
      by_cases e : r = d
      · subst e; simp only [WStmt.eval, WState.setReg, if_true]; split <;> decide
      · simpa [WStmt.eval, WState.setReg, e] using hb r hr
  | stgB a v => intro _ f W hb r hr; simpa [WStmt.eval, WState.stgByte] using hb r hr
  | coopCopy d s n => intro _ f W hb r hr; simpa [WStmt.eval] using hb r hr
  | uif c t e iht ihe =>
      intro hw f W hb
      simp only [WStmt.eval]
      split
      · exact iht (fun r h => hw r (by simp [assigns, h])) f W hb
      · exact ihe (fun r h => hw r (by simp [assigns, h])) f W hb
  | uwhile c b ih =>
      intro hw f
      induction f with
      | zero => intro W hb; simpa [WStmt.eval] using hb
      | succ f ihf =>
          intro W hb
          simp only [WStmt.eval]
          split
          · exact ihf _ (ih (fun r h => hw r (by simp [assigns, h])) f W hb)
          · exact hb
  | coopWindow found p0 cand0 sp inStride searchLim hashLog tbl =>
      intro hw f W hb r hr
      simp only [WStmt.eval]
      rw [evalCoopWindow_eq_go, evalCoopWindowGo_regs]
      have hp := hw p0 (by simp [assigns])
      have hc := hw cand0 (by simp [assigns])
      by_cases ef : r = found
      · subst ef
        have h1 : r ≠ cand0 := fun e => by rw [e, hc] at hr; cases hr
        have h2 : r ≠ p0 := fun e => by rw [e, hp] at hr; cases hr
        split <;> simp [WState.setReg, h1, h2]
      · by_cases ep : r = p0
        · subst ep; rw [hp] at hr; cases hr
        · by_cases ec : r = cand0
          · subst ec; rw [hc] at hr; cases hr
          · split <;> simp [WState.setReg, ef, ep, ec] <;> exact hb r hr
  | coopExtendStep adv p0 cand0 ml inStride endCap =>
      intro hw f W hb r hr
      by_cases e : r = adv
      · subst e; rw [hw r (by simp [assigns])] at hr; cases hr
      · simpa [WStmt.eval, evalCoopExtendStep, WState.setReg, e] using hb r hr
  | skip => intro _ f W hb; simpa [WStmt.eval] using hb

section Sim
variable {ns : List String} {I : UInt64 → Prop} {cta : Nat}

theorem SimW.set {ws : WSt F32} {W : WState} (h : SimW ns I ws W) {d : String} (hd : d ∈ ns) (F : Lane → UInt64)
    (v : UInt64) (hF : ∀ l, F l = v) : SimW ns I (ws.withInt (ws.int.set (ns.idxOf d) F)) (W.setReg d v) := by
  refine ⟨fun n hn l => ?_, fun l => ?_, h.gin, h.smem⟩
  · simp only [WSt.withInt, ISt.set, WState.setReg]
    by_cases e : n = d
    · subst e; simp [hF]
    · rw [if_neg (fun h' => e (idxOf_inj hn hd h')), if_neg e]; exact h.reg n hn l
  · simp only [WSt.withInt, ISt.set, laneR]
    rw [if_neg (by have := idxOf_lt hd; omega)]; exact h.lane l

theorem argW_evalW {ws : WSt F32} {W : WState} (h : SimW ns I ws W) (a : WArg) (ha : ∀ n, a = .reg n → n ∈ ns)
    (l : Lane) : (argW (ns.idxOf ·) a).eval ws.int cta l = a.eval W := by
  cases a with
  | reg n => exact h.reg n (ha n rfl) l
  | imm v => rfl

theorem testW_evalW {fl : List String} {ws : WSt F32} {W : WState} (h : SimW ns I ws W) (hb : BitsW fl W) {c : String}
    (hc : c ∈ ns) (l : Lane) : ((testW (ns.idxOf ·) fl c).eval ws.int cta l ≠ 0) ↔ (W.regs c == 1) = true := by
  unfold testW
  split
  · rename_i hf
    simp only [IExp.eval, h.reg c hc l]
    rcases flag_cases _ (hb c hf) with h0 | h0 <;> rw [h0] <;> decide
  · simp only [IExp.eval, ICmp.run, h.reg c hc l]
    by_cases e : W.regs c = 1 <;> simp [e]

end Sim

section Out
variable (hw : Hw) (cta S : Nat) (ptr : Buf → UInt64) (I O : UInt64 → Prop) (ns fl : List String)

/-- What `lane_warp` asks of the translation's run: lanes read the input
    bytes and store output bytes. -/
abbrev OkL (i : Nat) (s : EWStmt) (ws : WSt F32) (cl : Bool) : Prop :=
  OkW hw cta (fun _ _ => 0) S ptr (fun _ _ _ => False) (fun _ a => I a) (fun _ _ => False)
    (fun _ a n => n = 1 ∧ O a) i s ws cl

/-- The translation, its registers numbered by `ns`. -/
abbrev ofWn (s : WStmt) : EWStmt := ofW (ns.idxOf ·) ns.length (2 ^ 64) fl s

abbrev wrun (s : EWStmt) (ws : WSt F32) (i : Nat) : WSt F32 := LaneWarp.run hw cta (fun _ _ => 0) s ws i

abbrev wtr {α : Type} (leaf : Nat → EWStmt → WSt F32 → List α) (i : Nat) (s : EWStmt) (ws : WSt F32) : List α :=
  LaneWarp.trace hw cta (fun _ _ => 0) leaf i s ws

/-- **What the warp's run of a statement's translation does**: what `lane_warp`
    asks, the model's run, shared memory clean at its end, each lane's byte
    stores its share of the model's, each shared store followed by a barrier. -/
def Out (e : EWStmt) (s : WStmt) (f : Nat) (W : WState) (ws : WSt F32) (i : Nat) : Prop :=
  OkL hw cta S ptr I O i e ws true ∧
  SimW ns I (wrun hw cta e ws i) (s.eval f W) ∧
  lastOr true (wtr hw cta shLeaf i e ws) = true ∧
  (∀ l : Lane, byteSt (wtr hw cta (storeLeaf cta (fun _ _ => 0) l ptr (addrOf .arena 0)) i e ws) =
    lanePart l.val (gstOf f s W)) ∧
  Paired (wtr hw cta shEvLeaf i e ws)

end Out

section Plain
variable {hw : Hw} {cta S : Nat} {ptr : Buf → UInt64} {I O : UInt64 → Prop} {ns fl : List String}

theorem keep_sim {ws : WSt F32} {W : WState} {cs : List Nat} {carry : List String} (b : WStmt)
    (hS : SimW ns I ws W) (hsub : ∀ n ∈ carry, n ∈ ns) (hcs : cs = carry.map (ns.idxOf ·))
    (hset : ∀ n, n ∈ setsOf b → n ∈ carry) {ws' : WSt F32} {f : Nat} (hS' : SimW ns I ws' (b.eval f W)) :
    SimW ns I (ws'.withInt (ISt.keep cs ws.int ws'.int)) (b.eval f W) := by
  subst hcs
  refine ⟨fun n hn l => ?_, fun l => ?_, hS'.gin, hS'.smem⟩
  · simp only [WSt.withInt, ISt.keep]
    by_cases hm : n ∈ carry
    · rw [if_pos (List.mem_map.mpr ⟨n, hm, rfl⟩)]; exact hS'.reg n hn l
    · rw [if_neg (fun h => by
        obtain ⟨m, hm', he⟩ := List.mem_map.mp h
        exact hm (by rw [← idxOf_inj (hsub m hm') hn he]; exact hm')), hS.reg n hn l,
        eval_frame n b f W (fun h => hm (hset n h))]
  · simp only [WSt.withInt, ISt.keep]
    rw [if_neg (fun h => by
      obtain ⟨m, hm', he⟩ := List.mem_map.mp h
      have := idxOf_lt (hsub m hm'); simp only [laneR] at he; omega)]
    exact hS.lane l

/-- **A loop's run agrees with the model's**, its body's does at every fuel:
    the loop runs a trip where the model's test holds, then the loop. -/
theorem sim_loop (c : String) (b : WStmt) (hcn : c ∈ ns) (hnb : namesOf b ⊆ ns) (hwb : Wf fl b)
    (hbody : ∀ f W ws i, f ≤ 2 ^ 64 → BitsW fl W → W.smem.size ≤ S → okOf I O (okLZ4 I) f b W → SimW ns I ws W →
      Out hw cta S ptr I O ns (ofWn ns fl b) b f W ws i) :
    ∀ f n, f ≤ n → n ≤ 2 ^ 64 → ∀ W ws i, BitsW fl W → W.smem.size ≤ S → okOf I O (okLZ4 I) f (.uwhile c b) W →
      SimW ns I ws W →
      Out hw cta S ptr I O ns (.iwhile n (testW (ns.idxOf ·) fl c) ((dedup (c :: setsOf b)).map (ns.idxOf ·)) (ofWn ns fl b))
        (.uwhile c b) f W ws i := by
  have hsub : ∀ n ∈ dedup (c :: setsOf b), n ∈ ns := fun n hm => by
    rw [mem_dedup] at hm
    rcases List.mem_cons.mp hm with rfl | hm
    · exact hcn
    · exact hnb (setsOf_names b hm)
  have hset : ∀ n, n ∈ setsOf b → n ∈ dedup (c :: setsOf b) := fun n h => mem_dedup.mpr (List.mem_cons_of_mem _ h)
  generalize hcs : (dedup (c :: setsOf b)).map (ns.idxOf ·) = cs
  have stop : ∀ n W ws i, SimW ns I ws W → BitsW fl W → (W.regs c == 1) = false →
      Out hw cta S ptr I O ns (.iwhile n (testW (ns.idxOf ·) fl c) cs (ofWn ns fl b)) (.uwhile c b) 0 W ws i ∧
      ∀ f, Out hw cta S ptr I O ns (.iwhile n (testW (ns.idxOf ·) fl c) cs (ofWn ns fl b)) (.uwhile c b) (f + 1) W ws i := by
    intro n W ws i hS hb hc
    have ht : ¬ (testW (ns.idxOf ·) fl c).eval ws.int cta lane0 ≠ 0 := by
      rw [testW_evalW hS hb hcn]; simp [hc]
    have hu : ∀ l, ((testW (ns.idxOf ·) fl c).eval ws.int cta l ≠ 0 ↔
        (testW (ns.idxOf ·) fl c).eval ws.int cta lane0 ≠ 0) := fun l => by
      rw [testW_evalW hS hb hcn, testW_evalW hS hb hcn]
    have hO := okW_iwhile_stop hw cta (fun _ _ => 0) S ptr (fun _ _ _ => False) (fun _ a => I a) (fun _ _ => False)
      (fun _ a n => n = 1 ∧ O a) n _ cs (ofWn ns fl b) ws i true hu ht
    have hrun := run_iwhile_stop hw cta (fun _ _ => 0) n _ cs (ofWn ns fl b) ws i ht
    have htr : ∀ {α : Type} (leaf : Nat → EWStmt → WSt F32 → List α),
        LaneWarp.trace hw cta (fun _ _ => 0) leaf i (.iwhile n (testW (ns.idxOf ·) fl c) cs (ofWn ns fl b)) ws = [] :=
      fun leaf => trace_iwhile_stop hw cta (fun _ _ => 0) leaf n _ cs (ofWn ns fl b) ws i ht
    refine ⟨⟨hO, by rw [wrun, hrun]; simpa [WStmt.eval] using hS, by rw [wtr, htr]; rfl,
      fun l => by rw [wtr, htr]; simp [byteSt, gstOf, lanePart], by rw [wtr, htr]; exact Paired.nil⟩, fun f => ?_⟩
    refine ⟨hO, by rw [wrun, hrun]; simpa [WStmt.eval, hc] using hS, by rw [wtr, htr]; rfl,
      fun l => by rw [wtr, htr]; simp [byteSt, gstOf, hc, lanePart], by rw [wtr, htr]; exact Paired.nil⟩
  intro f
  induction f with
  | zero =>
    intro n _ _ W ws i hb _ hok hS
    exact (stop n W ws i hS hb (by simpa [okOf] using hok)).1
  | succ f ih =>
    intro n hfn hn W ws i hb hsz hok hS
    by_cases hc : (W.regs c == 1) = true
    · obtain ⟨n', rfl⟩ : ∃ n', n = n' + 1 := ⟨n - 1, by omega⟩
      have hok' : okOf I O (okLZ4 I) f b W ∧ okOf I O (okLZ4 I) f (.uwhile c b) (b.eval f W) := by
        simp only [okOf] at hok; exact hok hc
      obtain ⟨ob, sb, lb, bb, pb⟩ := hbody f W ws i (by omega) hb hsz hok'.1 hS
      have ht : (testW (ns.idxOf ·) fl c).eval ws.int cta lane0 ≠ 0 := (testW_evalW hS hb hcn lane0).mpr hc
      have hu : ∀ l, ((testW (ns.idxOf ·) fl c).eval ws.int cta l ≠ 0 ↔
          (testW (ns.idxOf ·) fl c).eval ws.int cta lane0 ≠ 0) := fun l => by
        rw [testW_evalW hS hb hcn, testW_evalW hS hb hcn]
      have hwt : wtrip hw cta (fun _ _ => 0) (testW (ns.idxOf ·) fl c) cs (ofWn ns fl b) ws i ws =
          (wrun hw cta (ofWn ns fl b) ws i).withInt (ISt.keep cs ws.int (wrun hw cta (ofWn ns fl b) ws i).int) := by
        simp only [wtrip, if_pos ht]
      have hS1 : SimW ns I (wtrip hw cta (fun _ _ => 0) (testW (ns.idxOf ·) fl c) cs (ofWn ns fl b) ws i ws)
          (b.eval f W) := by
        rw [hwt]; exact keep_sim b hS hsub hcs.symm hset sb
      obtain ⟨ol, sl, ll, bl, pl⟩ := ih n' (by omega) (by omega) (b.eval f W) _ i (eval_bits b hwb f W hb)
        (by rw [eval_smem_size]; exact hsz) hok'.2 hS1
      have he : (WStmt.uwhile c b).eval (f + 1) W = (WStmt.uwhile c b).eval f (b.eval f W) := by
        simp [WStmt.eval, hc]
      refine ⟨okW_iwhile_peel hw cta (fun _ _ => 0) S ptr (fun _ _ _ => False) (fun _ a => I a) (fun _ _ => False)
        (fun _ a n => n = 1 ∧ O a) n' _ cs (ofWn ns fl b) ws i true hu ht ob (by rw [lb]; exact ol), ?_, ?_, ?_, ?_⟩
      · rw [wrun, run_iwhile_peel, he]; exact sl
      · rw [wtr, trace_iwhile_peel hw cta (fun _ _ => 0) shLeaf n' _ cs (ofWn ns fl b) ws i ht, lastOr_append]
        rw [show lastOr true (LaneWarp.trace hw cta (fun _ _ => 0) shLeaf i (ofWn ns fl b) ws) = true from lb]; exact ll
      · intro l
        rw [wtr, trace_iwhile_peel hw cta (fun _ _ => 0) _ n' _ cs (ofWn ns fl b) ws i ht, byteSt_append]
        rw [show byteSt (LaneWarp.trace hw cta (fun _ _ => 0) (storeLeaf cta (fun _ _ => 0) l ptr (addrOf .arena 0)) i
          (ofWn ns fl b) ws) = _ from bb l, bl l]
        simp only [gstOf, hc, if_true, lanePart_append]
      · rw [wtr, trace_iwhile_peel hw cta (fun _ _ => 0) shEvLeaf n' _ cs (ofWn ns fl b) ws i ht]
        exact pb.append pl
    · exact (stop n W ws i hS hb (by simpa using hc)).2 f

end Plain

section Stmt
variable {hw : Hw} {cta S : Nat} {ptr : Buf → UInt64} {I O : UInt64 → Prop} {ns fl : List String}

theorem apart_of_ne {a a' : UInt64} {r r' : Region} {o o' : Nat} (hd : decodeAddr a = some (r, o))
    (hd' : decodeAddr a' = some (r', o')) (hne : a ≠ a') : Apart r o 1 r' o' 1 := by
  by_cases er : r = r'
  · subst er; right
    have : o ≠ o' := fun e => hne (decodeAddr_inj hd (by rw [hd', e]))
    omega
  · exact Or.inl er

theorem getD_set_ne (g : Array UInt8) {i j : Nat} (h : i ≠ j) (v : UInt8) : (g.set! i v).getD j 0 = g.getD j 0 := by
  simp only [Array.set!_eq_setIfInBounds, Array.getD_eq_getD_getElem?, Array.getElem?_setIfInBounds, if_neg h]

/-- **A setting statement agrees.** -/
theorem sim_set {ws : WSt F32} {W : WState} (hS : SimW ns I ws W) {d : String} (hd : d ∈ ns) (e : IExp) (v : UInt64)
    (he : ∀ l, e.eval ws.int cta l = v) (i f : Nat) (s : WStmt) (hs : s.eval f W = W.setReg d v)
    (hg : gstOf f s W = []) :
    Out hw cta S ptr I O ns (.setI (ns.idxOf d) e) s f W ws i := by
  refine ⟨trivial, ?_, rfl, fun l => by simp [wtr, LaneWarp.trace, storeLeaf, byteSt, hg, lanePart], by
    simp only [wtr, LaneWarp.trace, shEvLeaf]; exact Paired.nil⟩
  rw [hs]; exact hS.set hd _ v he

set_option maxHeartbeats 1000000 in
/-- **The translation's run agrees with the model's**, given that the
    cooperative statements' do. -/
theorem sim_stmt (hIdec : ∀ a, I a → ∃ r o, decodeAddr a = some (r, o))
    (hOdec : ∀ a, O a → ∃ r o, decodeAddr a = some (r, o))
    (hcoop : ∀ s : WStmt, (s matches .coopCopy .. | .coopWindow .. | .coopExtendStep ..) → ∀ f W ws i,
      namesOf s ⊆ ns → Wf fl s → f ≤ 2 ^ 64 → BitsW fl W → W.smem.size ≤ S → okOf I O (okLZ4 I) f s W →
      SimW ns I ws W → Out hw cta S ptr I O ns (ofWn ns fl s) s f W ws i) :
    ∀ (s : WStmt), namesOf s ⊆ ns → Wf fl s → ∀ f W ws i, f ≤ 2 ^ 64 → BitsW fl W → W.smem.size ≤ S →
      okOf I O (okLZ4 I) f s W → SimW ns I ws W → Out hw cta S ptr I O ns (ofWn ns fl s) s f W ws i := by
  intro s
  induction s with
  | skip =>
    intro _ _ f W ws i _ _ _ _ hS
    exact ⟨trivial, by simpa [WStmt.eval] using hS, rfl, fun l => by
      show byteSt [] = lanePart l.val (gstOf f .skip W); simp [gstOf, byteSt, lanePart], Paired.nil⟩
  | seq a b iha ihb =>
    intro hn hwf f W ws i hf hb hsz hok hS
    rw [show ofWn ns fl (a.seq b) = EWStmt.seq (ofWn ns fl a) (ofWn ns fl b) from rfl]
    have hna : namesOf a ⊆ ns := fun n h => hn (by simp [namesOf, h])
    have hnb : namesOf b ⊆ ns := fun n h => hn (by simp [namesOf, h])
    have hwa : Wf fl a := fun r h => hwf r (by simp [assigns, h])
    have hwb : Wf fl b := fun r h => hwf r (by simp [assigns, h])
    simp only [okOf] at hok
    obtain ⟨oa, sa, la, ba, pa⟩ := iha hna hwa f W ws i hf hb hsz hok.1 hS
    obtain ⟨ob, sb, lb, bb, pb⟩ := ihb hnb hwb f _ _ i hf (eval_bits a hwa f W hb) (by rw [eval_smem_size]; exact hsz)
      hok.2 sa
    have la' : lastOr true (LaneWarp.trace hw cta (fun _ _ => 0) shLeaf i (ofWn ns fl a) ws) = true := la
    refine ⟨⟨oa, by rw [la']; exact ob⟩, by simpa [WStmt.eval] using sb, ?_, fun l => ?_, ?_⟩
    · simp only [wtr, LaneWarp.trace, lastOr_append]
      rw [la']; exact lb
    · simp only [wtr, LaneWarp.trace, byteSt_append, gstOf, lanePart_append]
      rw [show byteSt _ = _ from ba l, show byteSt _ = _ from bb l]
    · simp only [wtr, LaneWarp.trace]; exact pa.append pb
  | bin o d a b =>
    intro hn _ f W ws i _ _ _ _ hS
    exact sim_set hS (hn (by simp [namesOf])) _ (o.run (W.regs a) (b.eval W)) (fun l => by
      simp only [IExp.eval, opW_run, hS.reg a (hn (by simp [namesOf])) l,
        argW_evalW hS b (fun n hb' => hn (by subst hb'; simp [namesOf])) l]) i f _ (by simp [WStmt.eval])
        (by simp [gstOf])
  | setp c d a b =>
    intro hn _ f W ws i _ _ _ _ hS
    exact sim_set hS (hn (by simp [namesOf])) _ (if c.run (W.regs a) (b.eval W) then 1 else 0) (fun l => by
      simp only [IExp.eval, cmpW_run, hS.reg a (hn (by simp [namesOf])) l,
        argW_evalW hS b (fun n hb' => hn (by subst hb'; simp [namesOf])) l]) i f _ (by simp [WStmt.eval])
        (by simp [gstOf])
  | mov d a =>
    intro hn _ f W ws i _ _ _ _ hS
    exact sim_set hS (hn (by simp [namesOf])) _ (a.eval W) (fun l =>
      argW_evalW hS a (fun n hb' => hn (by subst hb'; simp [namesOf])) l) i f _ (by simp [WStmt.eval])
        (by simp [gstOf])
  | stgB addr val =>
    intro hn _ f W ws i _ _ _ hok hS
    rw [show ofWn ns fl (.stgB addr val) = EWStmt.stB (.reg (ns.idxOf addr)) (.reg (ns.idxOf val)) (.lit 1) from rfl]
    simp only [okOf] at hok
    obtain ⟨hnI, hna, hO⟩ := hok
    have ha : ∀ l, (IExp.reg (ns.idxOf addr)).eval ws.int cta l = W.regs addr := fun l =>
      hS.reg addr (hn (by simp [namesOf])) l
    have hv : ∀ l, (IExp.reg (ns.idxOf val)).eval ws.int cta l = W.regs val := fun l =>
      hS.reg val (hn (by simp [namesOf])) l
    have h1 : ∀ l, (IExp.lit 1).eval ws.int cta l = 1 := fun _ => rfl
    obtain ⟨r, o, hd⟩ := hOdec _ hO
    have har : r ≠ .arena := fun e => hna o (by rw [hd, e])
    have hst := storeB_regs (a := fun l => (IExp.reg (ns.idxOf addr)).eval ws.int cta l)
      (v := fun l => (IExp.reg (ns.idxOf val)).eval ws.int cta l)
      (g := fun l => (IExp.lit 1).eval ws.int cta l) ws.int
    have hne : ∀ a', I a' → (W.regs addr).toNat ≠ a'.toNat := fun a' ha' e => hnI (by
      rw [show W.regs addr = a' from UInt64.toNat_inj.mp e]; exact ha')
    refine ⟨fun l _ => ⟨⟨⟨rfl, by rw [ha]; exact hO⟩, r, o, by rw [ha]; exact hd, har, fun _ _ _ h => h.elim,
      fun l' a' ha' => Or.inr ?_, fun _ _ h => h.elim⟩, fun l' _ _ h => absurd (by rw [ha] at h; exact h) hnI⟩,
      ⟨fun n hn' l => ?_, fun l => ?_, fun a' ha' => ?_, fun k hk => ?_⟩, ?_, fun l => ?_, ?_⟩
    · obtain ⟨r', o', hd'⟩ := hIdec a' ha'
      exact ⟨r', o', hd', apart_of_ne hd hd' (fun e => hnI (by rw [e]; exact ha'))⟩
    · have e1 : (wrun hw cta (.stB (.reg (ns.idxOf addr)) (.reg (ns.idxOf val)) (.lit 1)) ws i).int.regs =
          ws.int.regs := hst.1
      rw [e1, hS.reg n hn' l]; simp [WStmt.eval, WState.stgByte]
    · have e1 : (wrun hw cta (.stB (.reg (ns.idxOf addr)) (.reg (ns.idxOf val)) (.lit 1)) ws i).int.regs =
          ws.int.regs := hst.1
      rw [e1]; exact hS.lane l
    · have e1 : (wrun hw cta (.stB (.reg (ns.idxOf addr)) (.reg (ns.idxOf val)) (.lit 1)) ws i).int.gmem a'.toNat =
          ws.int.gmem a'.toNat :=
        storeB_other (fun l => (IExp.reg (ns.idxOf addr)).eval ws.int cta l)
          (fun l => (IExp.reg (ns.idxOf val)).eval ws.int cta l) (fun l => (IExp.lit 1).eval ws.int cta l) ws.int
          a'.toNat (fun l _ => by simp only [ha]; exact hne a' ha')
      rw [e1, hS.gin a' ha']
      simp only [WStmt.eval, WState.stgByte]
      rw [getD_set_ne _ (hne a' ha')]
    · have e1 : (wrun hw cta (.stB (.reg (ns.idxOf addr)) (.reg (ns.idxOf val)) (.lit 1)) ws i).int.smem =
          ws.int.smem := hst.2.1
      have hk' : k < W.smem.size := by simpa [WStmt.eval, WState.stgByte] using hk
      rw [e1, hS.smem k hk']; simp [WStmt.eval, WState.stgByte]
    · show lastOr true [] = true; rfl
    · simp only [wtr, LaneWarp.trace, storeLeaf, ha, hv, h1, gstOf, lanePart]
      simp [byteSt]
    · simp only [wtr, LaneWarp.trace, shEvLeaf]; exact Paired.nil
  | uif c t e iht ihe =>
    intro hn hwf f W ws i hf hb hsz hok hS
    have hcn : c ∈ ns := hn (by simp [namesOf])
    have hnt : namesOf t ⊆ ns := fun n h => hn (by simp [namesOf, h])
    have hne : namesOf e ⊆ ns := fun n h => hn (by simp [namesOf, h])
    have hu : ∀ l, ((testW (ns.idxOf ·) fl c).eval ws.int cta l ≠ 0 ↔
        (testW (ns.idxOf ·) fl c).eval ws.int cta lane0 ≠ 0) := fun l => by
      rw [testW_evalW hS hb hcn, testW_evalW hS hb hcn]
    by_cases hc : (W.regs c == 1) = true
    · have ht : (testW (ns.idxOf ·) fl c).eval ws.int cta lane0 ≠ 0 := (testW_evalW hS hb hcn lane0).mpr hc
      have ht' : (testW (ns.idxOf ·) fl c).eval ws.int cta ⟨0, by decide⟩ ≠ 0 := ht
      simp only [okOf, hc, if_true] at hok
      obtain ⟨o1, s1, l1, b1, p1⟩ := iht hnt (fun r h => hwf r (by simp [assigns, h])) f W ws i hf hb hsz hok hS
      refine ⟨⟨hu, by rw [if_pos ht]; exact o1⟩, ?_, ?_, fun l => ?_, ?_⟩
      · have ht0 : ¬ IExp.eval ws.int cta 0 (testW (ns.idxOf ·) fl c) = 0 := ht
        simpa [wrun, LaneWarp.run, EWStmt.run, ofW, ht0, WStmt.eval, hc] using s1
      · simpa [wtr, LaneWarp.trace, ofW, if_pos ht] using l1
      · simpa [wtr, LaneWarp.trace, ofW, if_pos ht, gstOf, hc] using b1 l
      · simpa [wtr, LaneWarp.trace, ofW, if_pos ht] using p1
    · have ht : ¬ (testW (ns.idxOf ·) fl c).eval ws.int cta lane0 ≠ 0 := fun h => hc ((testW_evalW hS hb hcn lane0).mp h)
      have ht' : ¬ (testW (ns.idxOf ·) fl c).eval ws.int cta ⟨0, by decide⟩ ≠ 0 := ht
      simp only [okOf, hc, Bool.false_eq_true, if_false] at hok
      obtain ⟨o1, s1, l1, b1, p1⟩ := ihe hne (fun r h => hwf r (by simp [assigns, h])) f W ws i hf hb hsz hok hS
      refine ⟨⟨hu, by rw [if_neg ht]; exact o1⟩, ?_, ?_, fun l => ?_, ?_⟩
      · have ht0 : IExp.eval ws.int cta 0 (testW (ns.idxOf ·) fl c) = 0 := by simpa using ht
        simpa [wrun, LaneWarp.run, EWStmt.run, ofW, ht0, WStmt.eval, hc] using s1
      · simpa [wtr, LaneWarp.trace, ofW, if_neg ht] using l1
      · simpa [wtr, LaneWarp.trace, ofW, if_neg ht, gstOf, hc] using b1 l
      · simpa [wtr, LaneWarp.trace, ofW, if_neg ht] using p1
  | uwhile c b ih =>
    intro hn hwf f W ws i hf hb hsz hok hS
    have hnb : namesOf b ⊆ ns := fun n h => hn (by simp [namesOf, h])
    have hwb : Wf fl b := fun r h => hwf r (by simp [assigns, h])
    exact sim_loop c b (hn (by simp [namesOf])) hnb hwb
      (fun f W ws i hf hb hsz hok hS => ih hnb hwb f W ws i hf hb hsz hok hS) f (2 ^ 64) hf (Nat.le_refl _) W ws i hb
      hsz hok hS
  | coopCopy d s n => intro hn hwf f W ws i hf hb hsz hok hS; exact hcoop _ rfl f W ws i hn hwf hf hb hsz hok hS
  | coopWindow => intro hn hwf f W ws i hf hb hsz hok hS; exact hcoop _ rfl f W ws i hn hwf hf hb hsz hok hS
  | coopExtendStep => intro hn hwf f W ws i hf hb hsz hok hS; exact hcoop _ rfl f W ws i hn hwf hf hb hsz hok hS

end Stmt

end Lz4Code
