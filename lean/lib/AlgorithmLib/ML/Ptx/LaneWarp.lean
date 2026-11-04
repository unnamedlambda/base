module
public import AlgorithmLib.ML.Ptx.BodyLane
meta import AlgorithmLib.ML.Ptx.BodyLane
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `ML.Ptx.LaneWarp` — a lane runs a warp statement with the warp

The warp's run of a statement (`EWStmt.run` at `F32`, `run`) gives each lane
`l` what its collectives are answered (`trace` of `ansLeaf`: a ballot's mask,
a shuffle's word from its source lane, a barrier's shared memory), the
collectives it makes (`callLeaf`) and the stores it makes (`storeLeaf`), in
order.

`lane_warp`: a lane (`BodyLane.laneI`) started agreeing with the warp
(`RelW`: its registers the warp's column, its memory the warp's at the
elements, bytes and index words it reads, its shared memory the warp's while
shared memory is clean), answered from the warp's tape for it, runs a
statement whose warp run meets what a lane needs (`OkW`) to an end agreeing
with the warp's, its collectives and stores the warp's traces for it. `OkW`
asks that indexes fit 32 bits, that every element or byte a lane reads is
stored by that lane alone and apart from every other store, that a shared
load come while shared memory is clean (after a barrier, before any shared
store), that a branch's or loop's test be the same in every lane, and that a
`while` loop stop within its bound.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec AlgorithmLib.Kernel AlgorithmLib.ML AlgorithmLib.ML.BodyLane

namespace AlgorithmLib.ML.LaneWarp

def lane0 : Lane := ⟨0, by decide⟩

section
variable (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat)

/-- The warp's run at `F32` on the hardware `hw`, integer registers read
    nowhere by an index, integer memory `im`. -/
abbrev run (s : EWStmt) (ws : WSt F32) (i : Nat) : WSt F32 :=
  @EWStmt.run F32 (F32.ops hw) _ s ws cta i (fun _ _ => 0) im

/-- **What the warp's run of a statement does at its leaves, in order**:
    `leaf` of each statement holding no other, at the trip and state the run
    reaches it. -/
def trace {α : Type} (leaf : Nat → EWStmt → WSt F32 → List α) : Nat → EWStmt → WSt F32 → List α
  | i, .seq a b, ws => trace leaf i a ws ++ trace leaf i b (run hw cta im a ws i)
  | i, .iif c t e, ws => if c.eval ws.int cta lane0 ≠ 0 then trace leaf i t ws else trace leaf i e ws
  | _, .forN n b, ws =>
      ((List.range n).foldl (fun (p : List α × WSt F32) j => (p.1 ++ trace leaf j b p.2, run hw cta im b p.2 j))
        ([], ws)).1
  | _, .forM bu a b, ws =>
      ((List.range (im bu a)).foldl (fun (p : List α × WSt F32) j =>
        (p.1 ++ trace leaf j b p.2, run hw cta im b p.2 j)) ([], ws)).1
  | i, .iwhile n c cs b, ws =>
      ((List.range n).foldl (fun (p : List α × WSt F32) _ =>
        if c.eval p.2.int cta lane0 ≠ 0 then
          let s' := run hw cta im b p.2 i
          (p.1 ++ trace leaf i b p.2, s'.withInt (ISt.keep cs ws.int s'.int))
        else p) ([], ws)).1
  | i, .locals _ b, ws => trace leaf i b ws
  | i, s, ws => leaf i s ws

/-- The trace's state component is the run's. -/
theorem forN_state {α : Type} (leaf : Nat → EWStmt → WSt F32 → List α) (b : EWStmt) :
    ∀ (L : List Nat) (p : List α × WSt F32),
      (L.foldl (fun (p : List α × WSt F32) j => (p.1 ++ trace hw cta im leaf j b p.2, run hw cta im b p.2 j)) p).2 =
        L.foldl (fun s' j => run hw cta im b s' j) p.2
  | [], _ => rfl
  | j :: L, p => by simp only [List.foldl_cons]; exact forN_state leaf b L _

end

/-- Shared memory's first `n` bytes, as a barrier answers them. -/
def bytesOf (n : Nat) (f : Nat → UInt8) : ByteArray := ⟨(Array.range n).map f⟩

section
variable (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat) (S : Nat) (l : Lane) (ptr : Buf → UInt64) (sm : UInt64)

/-- The answer lane `l`'s collective at a leaf gets from the warp. -/
def ansLeaf : Nat → EWStmt → WSt F32 → List Answer
  | _, .ballot _ p, ws => [.val (.sc .i32 (s32 (ISt.ballot (fun l => p.eval ws.int cta l)).toUInt32))]
  | _, .shflIdx _ v sr, ws =>
      [.val (.sc .i32 (s32 (v.eval ws.int cta (ISt.srcLane (sr.eval ws.int cta l))).toUInt32))]
  | _, .shflXor _ sr m, ws => [.val (.sc .i32 (s32 (ws.regs sr (xorLane l m)).bits.toUInt32))]
  | _, .barrier, ws => [.smem (bytesOf S ws.int.smem)]
  | _, _, _ => []

/-- The collective lane `l` makes at a leaf. -/
def callLeaf : Nat → EWStmt → WSt F32 → List (Intr × List V)
  | _, .ballot _ p, ws => [(.ballot, [boolV (p.eval ws.int cta l != 0)])]
  | _, .shflIdx _ v sr, ws =>
      [(.shflIdx, [.sc .i32 (v.eval ws.int cta l &&& 0xffffffff), .sc .i32 (sr.eval ws.int cta l &&& 0xffffffff)])]
  | _, .shflXor _ sr m, ws => [(.shflBfly, [.sc .i32 (ws.regs sr l).bits, .sc .i32 (s32 (UInt32.ofNat m))])]
  | _, .barrier, _ => [(.barrier, [])]
  | _, _, _ => []

/-- The stores lane `l` makes at a leaf, address, width and value, and its
    barriers (`none`). -/
def storeLeaf : Nat → EWStmt → WSt F32 → List (Option (UInt64 × Nat × UInt64))
  | _, .stB a v g, ws => if g.eval ws.int cta l ≠ 0 then [some (a.eval ws.int cta l, 1, v.eval ws.int cta l &&& 0xff)] else []
  | _, .stH a v g, ws =>
      if g.eval ws.int cta l ≠ 0 then [some (sm + a.eval ws.int cta l, 2, v.eval ws.int cta l &&& 0xffff)] else []
  | i, .storeLane b ix r, ws =>
      [some (ptr b + off (UInt32.ofNat (ix.eval cta i l (fun _ _ => 0) im)), 4, (ws.regs r l).bits)]
  | i, .storeLane0 b ix r, ws =>
      if l = lane0 then [some (ptr b + off (UInt32.ofNat (ix.eval cta i l (fun _ _ => 0) im)), 4, (ws.regs r l).bits)]
      else []
  | _, .barrier, _ => [none]
  | _, _, _ => []

end

/-- A world's stores since its `start`-th observation, address, width and
    value, and its barriers (`none`). -/
def storesOf (w : World) (start : Nat) : List (Option (UInt64 × Nat × UInt64)) :=
  (w.obs.reverse.drop start).filterMap fun
    | .store a n v => some (some (a, n, v))
    | .call (.intr .barrier) _ => some none
    | _ => none

/-- Shared memory's state after a leaf: clean after a barrier, not after a
    shared store. -/
def shLeaf : Nat → EWStmt → WSt F32 → List Bool
  | _, .barrier, _ => [true]
  | _, .stH .., _ => [false]
  | _, _, _ => []

/-- The last of a list of states, or `d`. -/
def lastOr (d : Bool) (L : List Bool) : Bool := L.getLast?.getD d

section
variable (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat) (S : Nat) (ptr : Buf → UInt64) (sm : UInt64)
  (MF : Lane → Buf → Nat → Prop) (MB : Lane → UInt64 → Prop) (MI : Buf → Nat → Prop)
  (WS : Lane → UInt64 → Nat → Prop)

/-- **An index fits 32 bits in lane `l`**: every value it computes, an index
    read a word `MI` allows. -/
def IdxOk (i : Nat) (l : Lane) : IdxE → Prop
  | .laneId | .ireg _ => True
  | .loopI => i < 2 ^ 32
  | .ctaId => cta < 2 ^ 32
  | .lit n => n < 2 ^ 32
  | .ldIdx b o => IdxOk i l o ∧ MI b (o.eval cta i l (fun _ _ => 0) im) ∧
      im b (o.eval cta i l (fun _ _ => 0) im) < 2 ^ 32
  | .add a b => IdxOk i l a ∧ IdxOk i l b ∧
      a.eval cta i l (fun _ _ => 0) im + b.eval cta i l (fun _ _ => 0) im < 2 ^ 32
  | .mul a b => IdxOk i l a ∧ IdxOk i l b ∧
      a.eval cta i l (fun _ _ => 0) im * b.eval cta i l (fun _ _ => 0) im < 2 ^ 32

/-- Bytes `[off, off + n)` and `[off', off' + n')` of regions `r`, `r'` do not meet. -/
def Apart (r : Region) (o n : Nat) (r' : Region) (o' n' : Nat) : Prop := r ≠ r' ∨ o + n ≤ o' ∨ o' + n' ≤ o

/-- **A store of `n` bytes at `a` by lane `l` may be made**: lane `l` may
    store there, outside shared memory, and every element, byte or index word
    a lane reads is apart from it, or is what lane `l` itself stores (`own`). -/
def StoreOk (l : Lane) (a : UInt64) (n : Nat) (ownF : Buf → Nat → Prop) (ownB : UInt64 → Prop) : Prop :=
  WS l a n ∧ ∃ r o, decodeAddr a = some (r, o) ∧ r ≠ .arena ∧
    (∀ l' b j, MF l' b j → (l' = l ∧ ownF b j) ∨ ∃ r' o', decodeAddr (ptr b + off (UInt32.ofNat j)) = some (r', o') ∧
      Apart r o n r' o' 4) ∧
    (∀ l' a', MB l' a' → (l' = l ∧ ownB a') ∨ ∃ r' o', decodeAddr a' = some (r', o') ∧ Apart r o n r' o' 1) ∧
    (∀ b j, MI b j → ∃ r' o', decodeAddr (ptr b + off (UInt32.ofNat j)) = some (r', o') ∧ Apart r o n r' o' 4)

/-- **What a lane's run needs of the warp's**, from shared memory clean or
    not: indexes fit, every access lies where a lane reads or may store, an
    element or byte some lane reads is stored by that lane alone, a shared
    load comes while shared memory is clean, a branch's or loop's test is the
    same in every lane, a `while` loop stops within its bound, a counted
    loop runs fewer than `2^31` trips, and nothing stages floats through
    shared memory or converts an index. -/
def OkW : Nat → EWStmt → WSt F32 → Bool → Prop
  | _, .skip, _, _ | _, .setR .., _, _ | _, .setI .., _, _ | _, .ballot .., _, _ | _, .shflIdx .., _, _
  | _, .barrier, _, _ => True
  | _, .shflXor _ _ m, _, _ => m < 2 ^ 31
  | i, .loadIdx _ b ix, _, _ => ∀ l, IdxOk cta im MI i l ix ∧ MF l b (ix.eval cta i l (fun _ _ => 0) im)
  | i, .loadV4 _ _ _ _ b ix, _, _ => ∀ l, IdxOk cta im MI i l ix ∧ ix.eval cta i l (fun _ _ => 0) im + 3 < 2 ^ 32 ∧
      ∀ q, q < 4 → MF l b (ix.eval cta i l (fun _ _ => 0) im + q)
  | i, .storeLane b ix _, _, _ => ∀ l, IdxOk cta im MI i l ix ∧
      StoreOk ptr MF MB MI WS l (ptr b + off (UInt32.ofNat (ix.eval cta i l (fun _ _ => 0) im))) 4
        (fun b' j => b' = b ∧ j = ix.eval cta i l (fun _ _ => 0) im) (fun _ => False) ∧
      ∀ l', ix.eval cta i l' (fun _ _ => 0) im = ix.eval cta i l (fun _ _ => 0) im →
        MF l b (ix.eval cta i l (fun _ _ => 0) im) → l' = l
  | i, .storeLane0 b ix _, _, _ => IdxOk cta im MI i lane0 ix ∧
      StoreOk ptr MF MB MI WS lane0 (ptr b + off (UInt32.ofNat (ix.eval cta i lane0 (fun _ _ => 0) im))) 4
        (fun b' j => b' = b ∧ j = ix.eval cta i lane0 (fun _ _ => 0) im) (fun _ => False)
  | _, .ldB _ a, ws, _ => ∀ l, MB l (a.eval ws.int cta l)
  | _, .stB a _ g, ws, _ => ∀ l, g.eval ws.int cta l ≠ 0 →
      StoreOk ptr MF MB MI WS l (a.eval ws.int cta l) 1 (fun _ _ => False) (fun a' => a' = a.eval ws.int cta l) ∧
      ∀ l', g.eval ws.int cta l' ≠ 0 → a.eval ws.int cta l' = a.eval ws.int cta l → MB l (a.eval ws.int cta l) → l' = l
  | _, .ldH _ a, ws, cl => cl = true ∧ S ≤ 2 ^ 36 ∧ ∀ l, (a.eval ws.int cta l).toNat + 2 ≤ S
  | _, .stH a _ g, ws, _ => S ≤ 2 ^ 36 ∧ ∀ l, g.eval ws.int cta l ≠ 0 → (a.eval ws.int cta l).toNat + 2 ≤ S
  | i, .seq a b, ws, cl => OkW i a ws cl ∧
      OkW i b (run hw cta im a ws i) (lastOr cl (trace hw cta im shLeaf i a ws))
  | i, .iif c t e, ws, cl => (∀ l, (c.eval ws.int cta l ≠ 0 ↔ c.eval ws.int cta lane0 ≠ 0)) ∧
      if c.eval ws.int cta lane0 ≠ 0 then OkW i t ws cl else OkW i e ws cl
  | _, .forN n b, ws, cl => n < 2 ^ 31 ∧ ∀ j, j < n →
      OkW j b (run hw cta im (.forN j b) ws 0) (lastOr cl (trace hw cta im shLeaf 0 (.forN j b) ws))
  | _, .forM bu a b, ws, cl => a < 2 ^ 31 ∧ MI bu a ∧ im bu a < 2 ^ 31 ∧ ∀ j, j < im bu a →
      OkW j b (run hw cta im (.forN j b) ws 0) (lastOr cl (trace hw cta im shLeaf 0 (.forN j b) ws))
  | i, .iwhile n c cs b, ws, cl =>
      (∃ k, k ≤ n ∧ c.eval (run hw cta im (.iwhile k c cs b) ws i).int cta lane0 = 0) ∧
      ∀ j, j ≤ n →
        (∀ l, (c.eval (run hw cta im (.iwhile j c cs b) ws i).int cta l ≠ 0 ↔
          c.eval (run hw cta im (.iwhile j c cs b) ws i).int cta lane0 ≠ 0)) ∧
        (c.eval (run hw cta im (.iwhile j c cs b) ws i).int cta lane0 ≠ 0 →
          OkW i b (run hw cta im (.iwhile j c cs b) ws i)
            (lastOr cl (trace hw cta im shLeaf i (.iwhile j c cs b) ws)))
  | i, .locals _ b, ws, cl => OkW i b ws cl
  | _, .stSm .., _, _ | _, .ldSm .., _, _ | _, .cvtIF .., _, _ => False

end

section
variable (hw : Hw) (S : Nat) (MF : Lane → Buf → Nat → Prop) (MB : Lane → UInt64 → Prop) (MI : Buf → Nat → Prop)
  (WS : Lane → UInt64 → Nat → Prop) (im : Buf → Nat → Nat)

/-- Every float the warp holds is a 32-bit word. -/
def W32 (ws : WSt F32) : Prop :=
  (∀ r l, (ws.regs r l).bits.toNat < 2 ^ 32) ∧ (∀ b j, (ws.mem b j).bits.toNat < 2 ^ 32) ∧
    ∀ k, (ws.smem k).bits.toNat < 2 ^ 32

/-- A store of `n` bytes at `a` succeeds in `m`. -/
def Writable (m : Mem) (a : UInt64) (n : Nat) : Prop :=
  ∃ r off, decodeAddr a = some (r, off) ∧ off + n ≤ (m.region r).size ∧ r ≠ .pinned

/-- **Lane `l` agrees with the warp**: its registers the warp's column, its
    memory the warp's at every element and byte it reads and every index
    word, its stores writable, its shared memory the warp's while shared
    memory is clean, the hardware's NaN rule, its collectives made from its
    first observation on. -/
structure RelW (l : Lane) (x : LCtx) (ws : WSt F32) (cl : Bool) (li : LI) : Prop where
  reg : ∀ r, li.reg r = ws.int.regs r l
  f : ∀ r, li.f r = ws.regs r l
  fmem : ∀ b j, MF l b j → li.w.mem.load (x.ptr b + off (UInt32.ofNat j)) 4 = some (ws.mem b j).bits
  bmem : ∀ a, MB l a → li.w.mem.load a 1 = some (ws.int.gmem a.toNat).toUInt64
  imem : ∀ b j, MI b j → li.w.mem.load (x.ptr b + off (UInt32.ofNat j)) 4 = some (UInt64.ofNat (im b j))
  wr : ∀ a n, WS l a n → Writable li.w.mem a n
  asz : li.w.mem.arena.size = S
  sm : cl = true → li.w.mem.arena = bytesOf S ws.int.smem
  nan : li.w.mem.nan = hw.nan
  start : x.start ≤ li.w.obs.length
  ptr : ∀ b, x.ptr b = ws.int.ptr b
  w32 : W32 ws

end

/-! ## A lane's values are the warp's column -/

section Agree
variable {cta : Nat} {im : Buf → Nat → Nat} {l : Lane} {x : LCtx} {ws : WSt F32}

/-- **An integer expression's lane value is the warp's**. -/
theorem iexp_agree {reg : Nat → UInt64} (hreg : ∀ r, reg r = ws.int.regs r l) (hlv : x.lv = UInt64.ofNat l.val)
    (hcv : x.cv = UInt64.ofNat cta) (hptr : ∀ b, x.ptr b = ws.int.ptr b) :
    ∀ e : IExp, e.evalL x reg = e.eval ws.int cta l := by
  intro e
  induction e with
  | reg r => exact hreg r
  | lit n => rfl
  | lane => exact hlv
  | cta => exact hcv
  | ptr b => exact hptr b
  | bin o a b iha ihb => simp only [IExp.evalL, IExp.eval, iha, ihb]
  | cmp c a b iha ihb => simp only [IExp.evalL, IExp.eval, iha, ihb]
  | ctz a ih => simp only [IExp.evalL, IExp.eval, ih]

theorem hw_nan (hw : Hw) : ({ hw with nan := hw.nan } : Hw) = hw := rfl

/-- **A float expression's lane value is the warp's**, on the hardware with
    its NaN rule. -/
theorem fexp_agree (hw : Hw) {f : Nat → F32} (hf : ∀ r, f r = ws.regs r l) :
    ∀ e : WFExp, evalF hw f e = @WFExp.eval F32 (F32.ops hw) _ ws l e := by
  intro e
  induction e <;> simp only [evalF, WFExp.evalL, WFExp.eval, *] at *

theorem toUInt32_ofNat64 (n : Nat) : (UInt64.ofNat n).toUInt32 = UInt32.ofNat n := by
  apply UInt32.toNat_inj.mp
  simp only [UInt64.toNat_toUInt32]
  exact Nat.mod_mod_of_dvd n (by decide)

/-- **An index's lane value is the warp's** where it fits 32 bits. -/
theorem idx_agree {i : Nat} {MI : Buf → Nat → Prop} {m : Mem} (hlv : x.lv = UInt64.ofNat l.val)
    (hcv : x.cv = UInt64.ofNat cta) (hi : x.i = i)
    (himem : ∀ b j, MI b j → m.load (x.ptr b + off (UInt32.ofNat j)) 4 = some (UInt64.ofNat (im b j))) :
    ∀ ix : IdxE, IdxOk cta im MI i l ix →
      idxV x m ix = some (UInt32.ofNat (ix.eval cta i l (fun _ _ => 0) im)) ∧
        ix.eval cta i l (fun _ _ => 0) im < 2 ^ 32 := by
  intro ix
  induction ix with
  | laneId =>
      intro _
      have hl : l.val < 32 := l.isLt
      refine ⟨?_, by simp only [IdxE.eval]; omega⟩
      simp only [idxV, IdxE.eval, hlv, toUInt32_ofNat64]
  | loopI => intro h; exact ⟨by simp [idxV, IdxE.eval, hi], h⟩
  | ctaId =>
      intro h
      refine ⟨?_, h⟩
      simp only [idxV, IdxE.eval, hcv, toUInt32_ofNat64]
  | lit n =>
      intro h
      have h' : n < 2 ^ 32 := h
      exact ⟨by simp [idxV, IdxE.eval, h'], h'⟩
  | ireg r => intro _; exact ⟨by simp [idxV, IdxE.eval], by simp [IdxE.eval]⟩
  | add a b iha ihb =>
      intro ⟨ha, hb, hlt⟩
      obtain ⟨ea, la⟩ := iha ha
      obtain ⟨eb, lb⟩ := ihb hb
      refine ⟨?_, hlt⟩
      simp only [idxV, ea, eb, IdxE.eval, UInt32.ofNat_add]
      rfl
  | mul a b iha ihb =>
      intro ⟨ha, hb, hlt⟩
      obtain ⟨ea, la⟩ := iha ha
      obtain ⟨eb, lb⟩ := ihb hb
      refine ⟨?_, hlt⟩
      simp only [idxV, ea, eb, IdxE.eval, UInt32.ofNat_mul]
      rfl
  | ldIdx b o ih =>
      intro ⟨ho, hmi, hlt⟩
      obtain ⟨eo, lo⟩ := ih ho
      refine ⟨?_, by simpa [IdxE.eval] using hlt⟩
      have hm := himem b _ hmi
      simp only [off] at hm
      simp only [idxV, eo, IdxE.eval, Option.bind_eq_bind, Option.bind_some]
      simp only [off, hm, Option.bind_some, toUInt32_ofNat64]

end Agree

/-! ## The warp's integer stores -/

section IStores
variable (a v g : Lane → UInt64)

theorem storeB_fold_other (A : Nat) : ∀ (L : List Lane) (st : ISt), (∀ l ∈ L, g l ≠ 0 → (a l).toNat ≠ A) →
    (L.foldl (fun s l => if g l ≠ 0 then
      { s with gmem := fun x => if x = (a l).toNat then (v l).toUInt8 else s.gmem x } else s) st).gmem A = st.gmem A
  | [], _, _ => rfl
  | l :: L, st, h => by
      simp only [List.foldl_cons]
      rw [storeB_fold_other A L _ (fun l' hl' => h l' (List.mem_cons_of_mem _ hl'))]
      split
      · rename_i hg; simp [Ne.symm (h l (List.mem_cons_self ..) hg)]
      · rfl

/-- **A byte no lane stores is unchanged.** -/
theorem storeB_other (st : ISt) (A : Nat) (h : ∀ l, g l ≠ 0 → (a l).toNat ≠ A) :
    (st.storeB a v g).gmem A = st.gmem A :=
  storeB_fold_other a v g A _ st (fun l _ => h l)

theorem storeB_fold_at (l0 : Lane) (A : Nat) :
    ∀ (L : List Lane) (st : ISt), (∀ l ∈ L, g l ≠ 0 → (a l).toNat = A → l = l0) →
    (L.foldl (fun s l => if g l ≠ 0 then
      { s with gmem := fun x => if x = (a l).toNat then (v l).toUInt8 else s.gmem x } else s) st).gmem A =
      if l0 ∈ L ∧ g l0 ≠ 0 ∧ (a l0).toNat = A then (v l0).toUInt8 else st.gmem A
  | [], _, _ => by simp
  | l :: L, st, h => by
      simp only [List.foldl_cons]
      rw [storeB_fold_at l0 A L _ (fun l' hl' => h l' (List.mem_cons_of_mem _ hl'))]
      by_cases hl : l = l0
      · subst hl
        by_cases hg : g l ≠ 0
        · by_cases ha : (a l).toNat = A
          · simp [hg, ha]
          · simp [hg, ha, Ne.symm ha]
        · simp [hg]
      · have hnot : ¬ (g l ≠ 0 ∧ (a l).toNat = A) := fun ⟨hg, ha⟩ => hl (h l (List.mem_cons_self ..) hg ha)
        by_cases hg : g l ≠ 0
        · have ha : (a l).toNat ≠ A := fun e => hnot ⟨hg, e⟩
          simp [hg, Ne.symm ha, List.mem_cons, Ne.symm hl]
        · simp [hg, List.mem_cons, Ne.symm hl]

/-- **The byte one lane alone stores is its.** -/
theorem storeB_at (st : ISt) (l0 : Lane) (hg : g l0 ≠ 0)
    (h : ∀ l, g l ≠ 0 → (a l).toNat = (a l0).toNat → l = l0) :
    (st.storeB a v g).gmem (a l0).toNat = (v l0).toUInt8 := by
  unfold ISt.storeB
  rw [storeB_fold_at a v g l0 _ _ st (fun l _ => h l)]
  simp [hg, List.mem_finRange]

theorem storeB_regs (st : ISt) : (st.storeB a v g).regs = st.regs ∧ (st.storeB a v g).smem = st.smem ∧
    (st.storeB a v g).ptr = st.ptr := by
  unfold ISt.storeB
  refine foldl_inv _ (fun s => s.regs = st.regs ∧ s.smem = st.smem ∧ s.ptr = st.ptr) (fun s l h => ?_) _ st ⟨rfl, rfl, rfl⟩
  split <;> exact h

theorem storeH_regs (st : ISt) : (st.storeH a v g).regs = st.regs ∧ (st.storeH a v g).gmem = st.gmem ∧
    (st.storeH a v g).ptr = st.ptr := by
  unfold ISt.storeH
  refine foldl_inv _ (fun s => s.regs = st.regs ∧ s.gmem = st.gmem ∧ s.ptr = st.ptr) (fun s l h => ?_) _ st ⟨rfl, rfl, rfl⟩
  split <;> exact h

end IStores

/-! ## What the warp's run keeps -/

section Keeps
variable (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat)

/-- **The warp's run keeps the buffers' addresses.** -/
theorem run_ptr : ∀ (s : EWStmt) (ws : WSt F32) (i : Nat), (run hw cta im s ws i).int.ptr = ws.int.ptr := by
  intro s
  induction s with
  | seq a b iha ihb => intro ws i; simp only [run, EWStmt.run] at *; rw [ihb, iha]
  | iif c t e iht ihe => intro ws i; simp only [run, EWStmt.run] at *; split <;> simp only [iht, ihe]
  | forN n b ih =>
      intro ws i
      simp only [run, EWStmt.run] at *
      exact foldl_inv _ (fun s' : WSt F32 => s'.int.ptr = ws.int.ptr) (fun s' j h => (ih s' j).trans h) _ _ rfl
  | forM bu a b ih =>
      intro ws i
      simp only [run, EWStmt.run] at *
      exact foldl_inv _ (fun s' : WSt F32 => s'.int.ptr = ws.int.ptr) (fun s' j h => (ih s' j).trans h) _ _ rfl
  | iwhile n c cs b ih =>
      intro ws i
      simp only [run, EWStmt.run] at *
      refine foldl_inv _ (fun s' : WSt F32 => s'.int.ptr = ws.int.ptr) (fun s' j h => ?_) _ _ rfl
      split
      · simp only [WSt.withInt, ISt.keep]; exact (ih s' i).trans h
      · exact h
  | locals rs b ih => intro ws i; simp only [run, EWStmt.run, WSt.withInt, ISt.restore] at *; exact ih ws i
  | stB a v g => intro ws i; simp only [run, EWStmt.run, WSt.withInt]; exact (storeB_regs _ _ _ _).2.2
  | stH a v g => intro ws i; simp only [run, EWStmt.run, WSt.withInt]; exact (storeH_regs _ _ _ _).2.2
  | _ => intro ws i; rfl


end Keeps

/-! ## Observations -/

theorem storesOf_call {w : World} {start : Nat} (hs : start ≤ w.obs.length) (c : Callee) (vs : List V) :
    storesOf (obsCall w c vs) start = storesOf w start ++ (if c = .intr .barrier then [none] else []) := by
  unfold storesOf obsCall
  simp only [List.reverse_cons]
  rw [List.drop_append_of_le_length (by simpa using hs), List.filterMap_append]
  congr 1
  cases c with
  | intr i => cases i <;> simp
  | _ => simp

theorem storesOf_store {w : World} {start : Nat} (hs : start ≤ w.obs.length) (a : UInt64) (n : Nat) (v : UInt64) :
    storesOf (obsStore w a n v) start = storesOf w start ++ [some (a, n, v)] := by
  unfold storesOf obsStore
  simp only [List.reverse_cons]
  rw [List.drop_append_of_le_length (by simpa using hs), List.filterMap_append]
  simp

theorem storesOf_arena {w : World} (b : ByteArray) (start : Nat) :
    storesOf { w with mem := { w.mem with arena := b } } start = storesOf w start := rfl

/-- The `2^x` calls an expression makes leave the collectives, the stores and
    memory as they were, and only add observations. -/
theorem exLog_obs (hw : Hw) (f : Nat → F32) : ∀ (e : WFExp) (w : World) (start : Nat), start ≤ w.obs.length →
    collCalls (exLog hw f e w) start = collCalls w start ∧ storesOf (exLog hw f e w) start = storesOf w start ∧
      w.obs.length ≤ (exLog hw f e w).obs.length ∧ (exLog hw f e w).mem = w.mem := by
  intro e
  induction e with
  | reg _ | lit _ => intro w start _; exact ⟨rfl, rfl, Nat.le_refl _, rfl⟩
  | add a b iha ihb | mul a b iha ihb | maxW a b iha ihb | geF a b iha ihb =>
      intro w start hs
      obtain ⟨a1, a2, a3, a4⟩ := iha w start hs
      obtain ⟨b1, b2, b3, b4⟩ := ihb (exLog hw f a w) start (Nat.le_trans hs a3)
      exact ⟨b1.trans a1, b2.trans a2, Nat.le_trans a3 b3, b4.trans a4⟩
  | neg a ih | inv a ih | rsqrt a ih => intro w start hs; exact ih w start hs
  | ex2 a ih | exp a ih =>
      intro w start hs
      obtain ⟨a1, a2, a3, a4⟩ := ih w start hs
      have hs' := Nat.le_trans hs a3
      refine ⟨?_, ?_, by simp only [exLog, obsCall, List.length_cons]; omega, a4⟩
      · rw [exLog, collCalls_call hs']; simpa [Intr.isColl] using a1
      · rw [exLog, storesOf_call hs']; simpa using a2

/-! ## Answers -/

theorem get_of_drop {α : Type} {L R : List α} {k : Nat} {a : α} (h : L.drop k = a :: R) : L[k]? = some a := by
  have := congrArg (·[0]?) h
  simpa [List.getElem?_drop] using this

theorem drop_succ_of {α : Type} {L R : List α} {k : Nat} {a : α} (h : L.drop k = a :: R) : L.drop (k + 1) = R := by
  rw [← List.drop_drop, h]; rfl

theorem ansW_val {ans : List Answer} {k : Nat} {u : UInt32} (h : ans[k]? = some (.val (.sc .i32 (s32 u)))) :
    ansW ans k = s32 u := by
  simp only [ansW, wordAt, h]
  congr 1
  apply UInt32.toNat_inj.mp
  simp [s32]

theorem s32_m32 (v : UInt64) : s32 v.toUInt32 &&& 0xffffffff = v &&& 0xffffffff := by
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_and, UInt64.toNat_and]
  simp only [s32, UInt32.toNat_toUInt64, UInt64.toNat_toUInt32]
  rw [show (0xffffffff : UInt64).toNat = 2 ^ 32 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod,
    Nat.and_two_pow_sub_one_eq_mod, Nat.mod_mod]

theorem shl_lane_lt : ∀ k : Fin 32, ((1 : UInt64) <<< UInt64.ofNat k.val).toNat < 2 ^ 32 := by decide

theorem ballot_fold_lt (p : Lane → UInt64) : ∀ (L : List Lane) (m : UInt64), m.toNat < 2 ^ 32 →
    (L.foldl (fun m l => if p l ≠ 0 then m ||| ((1 : UInt64) <<< UInt64.ofNat l.val) else m) m).toNat < 2 ^ 32
  | [], _, h => h
  | l :: L, m, h => by
      simp only [List.foldl_cons]
      refine ballot_fold_lt p L _ ?_
      split
      · rw [UInt64.toNat_or]
        exact Nat.or_lt_two_pow h (shl_lane_lt ⟨l.val, l.isLt⟩)
      · exact h

theorem ballot_lt (p : Lane → UInt64) : (ISt.ballot p).toNat < 2 ^ 32 := ballot_fold_lt p _ 0 (by decide)

/-! ## Shared memory -/

/-- An address outside shared memory. -/
def NotArena (a : UInt64) : Prop := ∀ o, decodeAddr a ≠ some (.arena, o)

theorem load_setArena {m : Mem} {b : ByteArray} {a : UInt64} {k : Nat} (hn : NotArena a) :
    ({ m with arena := b } : Mem).load a k = m.load a k := by
  unfold Mem.load
  cases hd : decodeAddr a with
  | none => rfl
  | some p =>
      obtain ⟨r, off⟩ := p
      have hr : r ≠ .arena := fun e => hn off (e ▸ hd)
      cases r <;> first | exact absurd rfl hr | rfl

theorem Writable.setArena {m : Mem} {a : UInt64} {n : Nat} (hw : Writable m a n) (b : ByteArray)
    (hb : m.arena.size ≤ b.size) : Writable { m with arena := b } a n := by
  obtain ⟨r, off, hd, hsz, hr⟩ := hw
  refine ⟨r, off, hd, ?_, hr⟩
  cases r <;> first | exact Nat.le_trans hsz hb | exact hsz

theorem bytesOf_size (n : Nat) (f : Nat → UInt8) : (bytesOf n f).size = n := by
  simp [bytesOf, ByteArray.size]

theorem arena_decodeL {a : UInt64} (ha : a.toNat < 2 ^ 36) :
    decodeAddr (addrOf .arena 0 + a) = some (.arena, a.toNat) := by
  rw [show addrOf .arena 0 + a = regionBase .arena + UInt64.ofNat a.toNat by simp [addrOf]]
  exact decodeAddr_base _ _ (by rw [regionSpan_toNat]; exact ha)

theorem bytesOf_get (n : Nat) (f : Nat → UInt8) {k : Nat} (hk : k < n) : (bytesOf n f).get! k = f k := by
  simp [bytesOf, ByteArray.get!, hk]

/-- **A half-word of shared memory** the warp's bytes hold. -/
theorem load_half {m : Mem} {S : Nat} {f : Nat → UInt8} {a : UInt64} (hS : a.toNat + 2 ≤ S) (h36 : S ≤ 2 ^ 36)
    (har : m.arena = bytesOf S f) :
    m.load (addrOf .arena 0 + a) 2 = some ((f a.toNat).toUInt64 ||| ((f (a.toNat + 1)).toUInt64 <<< 8)) := by
  rw [Mem.load_eq (arena_decodeL (by omega))]
  have hsz : (m.region .arena).size = S := by show m.arena.size = S; rw [har, bytesOf_size]
  rw [if_neg (by simp [Mem.readable, hsz]; omega)]
  congr 1
  show (List.range 2).foldr _ 0 = _
  simp only [show m.region .arena = bytesOf S f from har, List.range, List.range.loop, List.foldr_cons,
    List.foldr_nil, bytesOf_get S f (show a.toNat < S by omega), bytesOf_get S f (show a.toNat + 1 < S by omega),
    Nat.add_zero]
  rw [show ∀ y : UInt64, ((0 : UInt64) <<< 8 ||| y) = y from fun y => by simp]
  exact UInt64.or_comm _ _

theorem Writable.store {m m' : Mem} {a b v : UInt64} {n k : Nat} (h : m.store b k v = some m')
    (hw : Writable m a n) : Writable m' a n := by
  obtain ⟨r, o, hd, hsz, hr⟩ := hw
  exact ⟨r, o, hd, by rw [HProg.Contracts.store_sizes h r]; exact hsz, hr⟩

theorem Writable.stores {m : Mem} {a : UInt64} {n : Nat} (hw : Writable m a n) (v : UInt64) :
    ∃ m', m.store a n v = some m' := by
  obtain ⟨r, o, hd, hsz, hr⟩ := hw
  simp only [Mem.store, hd, Option.bind_eq_bind, Option.bind_some]
  have hrc : m.reachable r o n = true := by simp [Mem.reachable, hr]
  rw [if_neg (by simp [hrc]; omega)]
  exact ⟨_, rfl⟩

/-- **A store leaves a load apart from it.** -/
theorem load_apart {m m' : Mem} {a A v : UInt64} {n k : Nat} {r r' : Region} {o o' : Nat}
    (h : m.store a n v = some m') (hd : decodeAddr a = some (r, o)) (hd' : decodeAddr A = some (r', o'))
    (hap : Apart r o n r' o' k) : m'.load A k = m.load A k :=
  load_store_clear h hd hd' (by unfold Apart at hap; rcases hap with h1 | h1 | h1
                                · exact Or.inl (Ne.symm h1)
                                · exact Or.inr (Or.inr h1)
                                · exact Or.inr (Or.inl h1))

/-- A load that is made decodes. -/
theorem decode_of_load {m : Mem} {A v : UInt64} {k : Nat} (h : m.load A k = some v) :
    ∃ r o, decodeAddr A = some (r, o) := by
  unfold Mem.load at h
  cases hd : decodeAddr A with
  | none => simp [hd] at h
  | some p => exact ⟨p.1, p.2, rfl⟩

/-- **A shared store leaves every load outside shared memory.** -/
theorem load_arenaStore {m m' : Mem} {a A v w : UInt64} {n k : Nat} {o : Nat}
    (h : m.store a n v = some m') (hd : decodeAddr a = some (.arena, o)) (hn : NotArena A)
    (hl : m.load A k = some w) : m'.load A k = some w := by
  obtain ⟨r', o', hd'⟩ := decode_of_load hl
  rw [load_apart h hd hd' (Or.inl fun e => hn o' (e ▸ hd')), hl]

theorem byte_low (V : UInt64) : (HProg.Contracts.byteOfV (V &&& 0xff) 0).toUInt64 = V.toUInt8.toUInt64 := by
  apply UInt64.toNat_inj.mp
  simp only [HProg.Contracts.byteOfV, UInt8.toNat_toUInt64, UInt64.toNat_toUInt8]
  simp only [UInt64.toNat_and, UInt64.toNat_shiftRight]
  simp only [show (0xff : UInt64).toNat = 2 ^ 8 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod]
  simp

/-- Addresses apart decode apart: they differ. -/
theorem apart_ne {a a' : UInt64} {r r' : Region} {o o' n k : Nat} (hn : 0 < n) (hk : 0 < k)
    (hd : decodeAddr a = some (r, o)) (hd' : decodeAddr a' = some (r', o')) (hap : Apart r o n r' o' k) : a ≠ a' := by
  rintro rfl
  rw [hd] at hd'
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hd')
  unfold Apart at hap
  rcases hap with h | h | h
  · exact h rfl
  · omega
  · omega

/-! ## Loops of the warp -/

section Loops
variable (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat)

theorem run_forN_succ (k : Nat) (b : EWStmt) (ws : WSt F32) (i : Nat) :
    run hw cta im (.forN (k + 1) b) ws i = run hw cta im b (run hw cta im (.forN k b) ws i) k := by
  simp only [run, EWStmt.run, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]

theorem run_forN_i (k : Nat) (b : EWStmt) (ws : WSt F32) (i : Nat) :
    run hw cta im (.forN k b) ws i = run hw cta im (.forN k b) ws 0 := rfl

theorem trace_forN_succ {α : Type} (leaf : Nat → EWStmt → WSt F32 → List α) (k : Nat) (b : EWStmt) (ws : WSt F32)
    (i : Nat) : trace hw cta im leaf i (.forN (k + 1) b) ws =
      trace hw cta im leaf i (.forN k b) ws ++ trace hw cta im leaf k b (run hw cta im (.forN k b) ws i) := by
  simp only [trace, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
  rw [forN_state]
  rfl

theorem trace_forN_i {α : Type} (leaf : Nat → EWStmt → WSt F32 → List α) (k : Nat) (b : EWStmt) (ws : WSt F32)
    (i : Nat) : trace hw cta im leaf i (.forN k b) ws = trace hw cta im leaf 0 (.forN k b) ws := rfl

/-- A `while` loop's trip of the warp: the body, then the registers it does
    not carry back as before the loop. -/
def wtrip (c : IExp) (cs : List Nat) (b : EWStmt) (ws0 : WSt F32) (i : Nat) (s' : WSt F32) : WSt F32 :=
  if c.eval s'.int cta lane0 ≠ 0 then
    let s'' := run hw cta im b s' i
    s''.withInt (ISt.keep cs ws0.int s''.int)
  else s'

theorem iwhile_state {α : Type} (leaf : Nat → EWStmt → WSt F32 → List α) (c : IExp) (cs : List Nat) (b : EWStmt)
    (ws0 : WSt F32) (i : Nat) :
    ∀ (L : List Nat) (p : List α × WSt F32),
      (L.foldl (fun (p : List α × WSt F32) _ =>
        if c.eval p.2.int cta lane0 ≠ 0 then
          let s' := run hw cta im b p.2 i
          (p.1 ++ trace hw cta im leaf i b p.2, s'.withInt (ISt.keep cs ws0.int s'.int))
        else p) p).2 = L.foldl (fun s' _ => wtrip hw cta im c cs b ws0 i s') p.2
  | [], _ => rfl
  | j :: L, p => by
      simp only [List.foldl_cons]
      rw [iwhile_state leaf c cs b ws0 i L]
      unfold wtrip
      split <;> rfl

theorem run_iwhile (k : Nat) (c : IExp) (cs : List Nat) (b : EWStmt) (ws : WSt F32) (i : Nat) :
    run hw cta im (.iwhile k c cs b) ws i = (List.range k).foldl (fun s' _ => wtrip hw cta im c cs b ws i s') ws := by
  simp only [run, EWStmt.run, wtrip]; rfl

theorem run_iwhile_succ (k : Nat) (c : IExp) (cs : List Nat) (b : EWStmt) (ws : WSt F32) (i : Nat) :
    run hw cta im (.iwhile (k + 1) c cs b) ws i = wtrip hw cta im c cs b ws i (run hw cta im (.iwhile k c cs b) ws i) := by
  rw [run_iwhile, run_iwhile, List.range_succ, List.foldl_append]; rfl

theorem trace_iwhile_succ {α : Type} (leaf : Nat → EWStmt → WSt F32 → List α) (k : Nat) (c : IExp) (cs : List Nat)
    (b : EWStmt) (ws : WSt F32) (i : Nat) :
    trace hw cta im leaf i (.iwhile (k + 1) c cs b) ws = trace hw cta im leaf i (.iwhile k c cs b) ws ++
      (if c.eval (run hw cta im (.iwhile k c cs b) ws i).int cta lane0 ≠ 0 then
        trace hw cta im leaf i b (run hw cta im (.iwhile k c cs b) ws i) else []) := by
  simp only [trace, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
  rw [iwhile_state, ← run_iwhile]
  split <;> simp

end Loops

/-! ## A lane runs with the warp -/

section Lane
variable (hw : Hw) (cta : Nat) (im : Buf → Nat → Nat) (S : Nat) (MF : Lane → Buf → Nat → Prop)
  (MB : Lane → UInt64 → Prop) (MI : Buf → Nat → Prop) (WS : Lane → UInt64 → Nat → Prop) (l : Lane)

/-- **Lane `l` ends with the warp**: its run of `s` ends agreeing with the
    warp's, its collectives and stores the warp's traces for it, its answers
    used up to `R`. -/
def LaneEnds (x : LCtx) (i : Nat) (s : EWStmt) (ws : WSt F32) (cl : Bool) (li : LI) (R : List Answer) : Prop :=
  ∃ li', laneI x s li = some li' ∧
    RelW hw S MF MB MI WS im l x (run hw cta im s ws i) (lastOr cl (trace hw cta im shLeaf i s ws)) li' ∧
    collCalls li'.w x.start = collCalls li.w x.start ++ trace hw cta im (callLeaf cta l) i s ws ∧
    storesOf li'.w x.start = storesOf li.w x.start ++ trace hw cta im (storeLeaf cta im l x.ptr x.sm) i s ws ∧
    x.ans.drop (collCalls li'.w x.start).length = R

/-- What lane `l` runs a statement from: its context the warp's lane, block,
    trip and hardware, the warp's conditions, agreeing with the warp, its
    answers the warp's tape for it. -/
structure LanePre (x : LCtx) (i : Nat) (s : EWStmt) (ws : WSt F32) (cl : Bool) (li : LI) (R : List Answer) : Prop where
  hwq : x.hw = hw
  lv : x.lv = UInt64.ofNat l.val
  cv : x.cv = UInt64.ofNat cta
  cta32 : cta < 2 ^ 32
  ieq : x.i = i
  smq : x.sm = addrOf .arena 0
  ok : OkW hw cta im S x.ptr MF MB MI WS i s ws cl
  rel : RelW hw S MF MB MI WS im l x ws cl li
  tape : x.ans.drop (collCalls li.w x.start).length = trace hw cta im (ansLeaf cta S l) i s ws ++ R
  naF : ∀ l b j, MF l b j → NotArena (x.ptr b + off (UInt32.ofNat j))
  naB : ∀ l a, MB l a → NotArena a
  naI : ∀ b j, MI b j → NotArena (x.ptr b + off (UInt32.ofNat j))

theorem lane_skip (x : LCtx) (i : Nat) (ws : WSt F32) (cl : Bool) (li : LI) (R : List Answer)
    (h : LanePre hw cta im S MF MB MI WS l x i .skip ws cl li R) :
    LaneEnds hw cta im S MF MB MI WS l x i .skip ws cl li R :=
  ⟨li, rfl, h.rel, by simp [trace, callLeaf], by simp [trace, storeLeaf], by simpa [trace, ansLeaf] using h.tape⟩

theorem lane_setI (x : LCtx) (i : Nat) (d : Nat) (e : IExp) (ws : WSt F32) (cl : Bool) (li : LI) (R : List Answer)
    (h : LanePre hw cta im S MF MB MI WS l x i (.setI d e) ws cl li R) :
    LaneEnds hw cta im S MF MB MI WS l x i (.setI d e) ws cl li R := by
  have hr := h.rel
  have he := iexp_agree hr.reg h.lv h.cv hr.ptr e
  refine ⟨li.set d (e.evalL x li.reg), rfl, ?_, by simp [trace, callLeaf, LI.set], by simp [trace, storeLeaf, LI.set],
    by simpa [trace, ansLeaf, LI.set] using h.tape⟩
  refine { hr with reg := fun r => ?_ }
  simp only [run, EWStmt.run, WSt.withInt, ISt.set, LI.set, he]
  by_cases hrd : r = d <;> simp [hrd, hr.reg r]

theorem lane_setR (hw32 : ∀ k v, (hw.approx k v).toNat < 2 ^ 32) (x : LCtx) (i : Nat) (d : Nat) (e : WFExp)
    (ws : WSt F32) (cl : Bool) (li : LI) (R : List Answer)
    (h : LanePre hw cta im S MF MB MI WS l x i (.setR d e) ws cl li R) :
    LaneEnds hw cta im S MF MB MI WS l x i (.setR d e) ws cl li R := by
  have hr := h.rel
  have hhw : ({ x.hw with nan := li.w.mem.nan } : Hw) = hw := by rw [h.hwq, hr.nan]
  obtain ⟨o1, o2, o3, o4⟩ := exLog_obs hw li.f e li.w x.start hr.start
  refine ⟨_, rfl, ?_, ?_, ?_, ?_⟩
  · simp only [hhw]
    refine ⟨hr.reg, fun r => ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, hr.ptr, ?_⟩
    · simp only [LI.setF, run, EWStmt.run, WSt.setReg]
      by_cases hrd : r = d
      · subst hrd; simp [fexp_agree hw hr.f e]
      · simp [hrd, hr.f r]
    · intro b j hb; rw [o4]; exact hr.fmem b j hb
    · intro a ha; rw [o4]; exact hr.bmem a ha
    · intro b j hb; rw [o4]; exact hr.imem b j hb
    · intro a n ha; rw [o4]; exact hr.wr a n ha
    · rw [o4]; exact hr.asz
    · intro hc; rw [o4]; exact hr.sm hc
    · rw [o4]; exact hr.nan
    · exact Nat.le_trans hr.start o3
    · refine ⟨fun r l' => ?_, hr.w32.2.1, hr.w32.2.2⟩
      simp only [run, EWStmt.run, WSt.setReg]
      split
      · show (@WFExp.eval F32 (F32.ops hw) _ ws l' e).bits.toNat < 2 ^ 32
        rw [← fexp_agree hw (fun _ => rfl) e]; exact evalF_32 hw hw32 (fun r => hr.w32.1 r l') e
      · exact hr.w32.1 r l'
  · simp only [hhw, o1, trace, callLeaf, List.append_nil]
  · simp only [hhw, o2, trace, storeLeaf, List.append_nil]
  · simp only [hhw, o1]; simpa [trace, ansLeaf] using h.tape

theorem lane_ballot (x : LCtx) (i : Nat) (d : Nat) (p : IExp) (ws : WSt F32) (cl : Bool) (li : LI) (R : List Answer)
    (h : LanePre hw cta im S MF MB MI WS l x i (.ballot d p) ws cl li R) :
    LaneEnds hw cta im S MF MB MI WS l x i (.ballot d p) ws cl li R := by
  have hr := h.rel
  have hp := iexp_agree hr.reg h.lv h.cv hr.ptr p
  have ht := h.tape
  simp only [trace, ansLeaf, List.singleton_append] at ht
  have hk := get_of_drop ht
  have hv : ansW x.ans (collCalls li.w x.start).length &&& 0xffffffff =
      ISt.ballot (fun l => p.eval ws.int cta l) := by
    rw [ansW_val hk, s32_m32, and_mask32 (ballot_lt _)]
  refine ⟨_, by simp only [laneI, if_pos hr.start]; rfl, ?_, ?_, ?_, ?_⟩
  · refine ⟨fun r => ?_, hr.f, hr.fmem, hr.bmem, hr.imem, hr.wr, hr.asz, hr.sm, hr.nan,
      by simp [obsCall]; have := hr.start; omega, hr.ptr, hr.w32⟩
    simp only [LI.set, run, EWStmt.run, WSt.withInt, ISt.set, hv]
    by_cases hrd : r = d <;> simp [hrd, hr.reg r]
  · simp [trace, callLeaf, collCalls_call hr.start, Intr.isColl, hp]
  · simp [trace, storeLeaf, storesOf_call hr.start]
  · simp only [collCalls_call hr.start, if_pos (show Intr.ballot.isColl = true from rfl), List.length_append,
      List.length_singleton]
    exact drop_succ_of ht

theorem lane_shflIdx (x : LCtx) (i : Nat) (d : Nat) (v sr : IExp) (ws : WSt F32) (cl : Bool) (li : LI)
    (R : List Answer) (h : LanePre hw cta im S MF MB MI WS l x i (.shflIdx d v sr) ws cl li R) :
    LaneEnds hw cta im S MF MB MI WS l x i (.shflIdx d v sr) ws cl li R := by
  have hr := h.rel
  have ht := h.tape
  simp only [trace, ansLeaf, List.singleton_append] at ht
  have hk := get_of_drop ht
  have hv : ansW x.ans (collCalls li.w x.start).length &&& 0xffffffff =
      v.eval ws.int cta (ISt.srcLane (sr.eval ws.int cta l)) &&& 0xffffffff := by
    rw [ansW_val hk, s32_m32]
  refine ⟨_, by simp only [laneI, if_pos hr.start]; rfl, ?_, ?_, ?_, ?_⟩
  · refine ⟨fun r => ?_, hr.f, hr.fmem, hr.bmem, hr.imem, hr.wr, hr.asz, hr.sm, hr.nan,
      by simp [obsCall]; have := hr.start; omega, hr.ptr, hr.w32⟩
    simp only [LI.set, run, EWStmt.run, WSt.withInt, ISt.set, hv]
    by_cases hrd : r = d <;> simp [hrd, hr.reg r]
  · simp [trace, callLeaf, collCalls_call hr.start, Intr.isColl, iexp_agree hr.reg h.lv h.cv hr.ptr]
  · simp [trace, storeLeaf, storesOf_call hr.start]
  · simp only [collCalls_call hr.start, if_pos (show Intr.shflIdx.isColl = true from rfl), List.length_append,
      List.length_singleton]
    exact drop_succ_of ht

theorem F32.eta (y : F32) : (⟨y.bits⟩ : F32) = y := rfl

theorem lane_shflXor (x : LCtx) (i : Nat) (d sr m : Nat) (ws : WSt F32) (cl : Bool) (li : LI)
    (R : List Answer) (h : LanePre hw cta im S MF MB MI WS l x i (.shflXor d sr m) ws cl li R) :
    LaneEnds hw cta im S MF MB MI WS l x i (.shflXor d sr m) ws cl li R := by
  have hr := h.rel
  have hm : m < 2 ^ 31 := h.ok
  have ht := h.tape
  simp only [trace, ansLeaf, List.singleton_append] at ht
  have hk := get_of_drop ht
  have hv : ansW x.ans (collCalls li.w x.start).length = (ws.regs sr (xorLane l m)).bits := by
    rw [ansW_val hk]
    apply UInt64.toNat_inj.mp
    have := hr.w32.1 sr (xorLane l m)
    simp only [s32, UInt32.toNat_toUInt64, UInt64.toNat_toUInt32]
    omega
  refine ⟨_, by simp only [laneI, if_pos (And.intro hm hr.start)]; rfl, ?_, ?_, ?_, ?_⟩
  · refine ⟨hr.reg, fun r => ?_, hr.fmem, hr.bmem, hr.imem, hr.wr, hr.asz, hr.sm, hr.nan,
      by simp [obsCall]; have := hr.start; omega, hr.ptr, ⟨fun r l' => ?_, hr.w32.2.1, hr.w32.2.2⟩⟩
    · simp only [LI.setF, run, EWStmt.run, WSt.setReg, hv]
      by_cases hrd : r = d <;> simp [hrd, hr.f r]
    · simp only [run, EWStmt.run, WSt.setReg]
      split
      · exact hr.w32.1 sr _
      · exact hr.w32.1 r l'
  · simp [trace, callLeaf, collCalls_call hr.start, Intr.isColl, hr.f]
  · simp [trace, storeLeaf, storesOf_call hr.start]
  · simp only [collCalls_call hr.start, if_pos (show Intr.shflBfly.isColl = true from rfl), List.length_append,
      List.length_singleton]
    exact drop_succ_of ht

theorem lane_barrier (x : LCtx) (i : Nat) (ws : WSt F32) (cl : Bool) (li : LI)
    (R : List Answer) (h : LanePre hw cta im S MF MB MI WS l x i .barrier ws cl li R) :
    LaneEnds hw cta im S MF MB MI WS l x i .barrier ws cl li R := by
  have hr := h.rel
  have ht := h.tape
  simp only [trace, ansLeaf, List.singleton_append] at ht
  have hk := get_of_drop ht
  refine ⟨_, by simp only [laneI, if_pos hr.start, hk]; rfl, ?_, ?_, ?_, ?_⟩
  · refine ⟨hr.reg, hr.f, fun b j hb => ?_, fun a ha => ?_, fun b j hb => ?_, fun a n ha => ?_, ?_,
      fun _ => rfl, hr.nan, by simp [obsCall]; have := hr.start; omega, hr.ptr, hr.w32⟩
    · rw [load_setArena (h.naF l b j hb)]; exact hr.fmem b j hb
    · rw [load_setArena (h.naB l a ha)]; exact hr.bmem a ha
    · rw [load_setArena (h.naI b j hb)]; exact hr.imem b j hb
    · exact (hr.wr a n ha).setArena _ (by rw [bytesOf_size, ← hr.asz]; exact Nat.le_refl _)
    · exact bytesOf_size _ _
  · show collCalls (callW li.w .barrier []) x.start = _
    simp [trace, callLeaf, collCalls_call hr.start, Intr.isColl]
  · show storesOf (callW li.w .barrier []) x.start = _
    simp [trace, storeLeaf, storesOf_call hr.start]
  · show x.ans.drop (collCalls (callW li.w .barrier []) x.start).length = R
    simp only [collCalls_call hr.start, if_pos (show Intr.barrier.isColl = true from rfl), List.length_append,
      List.length_singleton]
    exact drop_succ_of ht

theorem lane_ldB (x : LCtx) (i : Nat) (d : Nat) (a : IExp) (ws : WSt F32) (cl : Bool) (li : LI)
    (R : List Answer) (h : LanePre hw cta im S MF MB MI WS l x i (.ldB d a) ws cl li R) :
    LaneEnds hw cta im S MF MB MI WS l x i (.ldB d a) ws cl li R := by
  have hr := h.rel
  have ha := iexp_agree hr.reg h.lv h.cv hr.ptr a
  have hb : MB l (a.eval ws.int cta l) := h.ok l
  refine ⟨_, by simp only [laneI, ha, hr.bmem _ hb]; rfl, ?_, by simp [trace, callLeaf, LI.set],
    by simp [trace, storeLeaf, LI.set], by simpa [trace, ansLeaf, LI.set] using h.tape⟩
  refine ⟨fun r => ?_, hr.f, hr.fmem, hr.bmem, hr.imem, hr.wr, hr.asz, hr.sm, hr.nan, hr.start, hr.ptr, hr.w32⟩
  simp only [LI.set, run, EWStmt.run, WSt.withInt, ISt.set, ISt.byte]
  by_cases hrd : r = d <;> simp [hrd, hr.reg r]

theorem lane_loadIdx (x : LCtx) (i : Nat) (d : Nat) (b : Buf) (ix : IdxE) (ws : WSt F32) (cl : Bool) (li : LI)
    (R : List Answer) (h : LanePre hw cta im S MF MB MI WS l x i (.loadIdx d b ix) ws cl li R) :
    LaneEnds hw cta im S MF MB MI WS l x i (.loadIdx d b ix) ws cl li R := by
  have hr := h.rel
  obtain ⟨hix, hmf⟩ := h.ok l
  obtain ⟨hj, -⟩ := idx_agree h.lv h.cv h.ieq hr.imem ix hix
  refine ⟨_, by simp only [laneI, hj, Option.bind_eq_bind, Option.bind_some, hr.fmem _ _ hmf]; rfl, ?_, by simp [trace, callLeaf, LI.setF],
    by simp [trace, storeLeaf, LI.setF], by simpa [trace, ansLeaf, LI.setF] using h.tape⟩
  refine ⟨hr.reg, fun r => ?_, hr.fmem, hr.bmem, hr.imem, hr.wr, hr.asz, hr.sm, hr.nan, hr.start, hr.ptr,
    ⟨fun r l' => ?_, hr.w32.2.1, hr.w32.2.2⟩⟩
  · simp only [LI.setF, run, EWStmt.run, WSt.setReg]
    by_cases hrd : r = d <;> simp [hrd, hr.f r]
  · simp only [run, EWStmt.run, WSt.setReg]
    split
    · exact hr.w32.2.1 _ _
    · exact hr.w32.1 r l'

theorem off_add (j q : Nat) (h : j + q < 2 ^ 32) :
    off (UInt32.ofNat j) + c64 (4 * q : Nat) = off (UInt32.ofNat (j + q)) := by
  apply UInt64.toNat_inj.mp
  have h1 : (UInt32.ofNat j).toNat = j := toNat_ofNat32 (by omega)
  have h2 : (UInt32.ofNat (j + q)).toNat = j + q := toNat_ofNat32 h
  have h3 : (UInt64.ofNat (4 * q)).toNat = 4 * q := by simp; omega
  simp only [off, s32, c64_ofNat]
  rw [UInt64.toNat_add, UInt64.toNat_shiftLeft, UInt64.toNat_shiftLeft]
  simp only [UInt32.toNat_toUInt64, h1, h2, h3]
  have hc : (c64 2 % 64).toNat = 2 := by decide
  rw [hc, Nat.shiftLeft_eq, Nat.shiftLeft_eq]
  simp only [show (2 : Nat) % 64 = 2 from rfl, show (2 : Nat) ^ 2 = 4 from rfl]
  omega

theorem lane_loadV4 (x : LCtx) (i : Nat) (d0 d1 d2 d3 : Nat) (b : Buf) (ix : IdxE) (ws : WSt F32) (cl : Bool)
    (li : LI) (R : List Answer) (h : LanePre hw cta im S MF MB MI WS l x i (.loadV4 d0 d1 d2 d3 b ix) ws cl li R) :
    LaneEnds hw cta im S MF MB MI WS l x i (.loadV4 d0 d1 d2 d3 b ix) ws cl li R := by
  have hr := h.rel
  obtain ⟨hix, h4, hmf⟩ := h.ok l
  obtain ⟨hj, hj32⟩ := idx_agree h.lv h.cv h.ieq hr.imem ix hix
  generalize hJ : ix.eval cta i l (fun _ _ => 0) im = J at hj hj32 h4 hmf
  have ha : ∀ q, q < 4 → x.ptr b + off (UInt32.ofNat J) + c64 (4 * q : Nat) = x.ptr b + off (UInt32.ofNat (J + q)) :=
    fun q hq => by rw [UInt64.add_assoc, off_add J q (by omega)]
  have l0 := hr.fmem b J (by simpa using hmf 0 (by decide))
  have l1 := hr.fmem b (J + 1) (hmf 1 (by decide)); rw [← ha 1 (by decide)] at l1
  have l2 := hr.fmem b (J + 2) (hmf 2 (by decide)); rw [← ha 2 (by decide)] at l2
  have l3 := hr.fmem b (J + 3) (hmf 3 (by decide)); rw [← ha 3 (by decide)] at l3
  have e4 : c64 4 = c64 ((4 * 1 : Nat) : Int) := rfl
  have e8 : c64 8 = c64 ((4 * 2 : Nat) : Int) := rfl
  have e12 : c64 12 = c64 ((4 * 3 : Nat) : Int) := rfl
  rw [← e4] at l1; rw [← e8] at l2; rw [← e12] at l3
  refine ⟨_, by simp only [laneI, hj, Option.bind_eq_bind, Option.bind_some, l0, l1, l2, l3]; rfl, ?_,
    by simp [trace, callLeaf, LI.setF], by simp [trace, storeLeaf, LI.setF],
    by simpa [trace, ansLeaf, LI.setF] using h.tape⟩
  refine ⟨hr.reg, fun r => ?_, hr.fmem, hr.bmem, hr.imem, hr.wr, hr.asz, hr.sm, hr.nan, hr.start, hr.ptr,
    ⟨fun r l' => ?_, hr.w32.2.1, hr.w32.2.2⟩⟩
  · simp only [LI.setF, run, EWStmt.run, WSt.setReg]
    repeat' split
    all_goals simp [hJ, hr.f r]
  · simp only [run, EWStmt.run, WSt.setReg]
    repeat' split
    all_goals first | exact hr.w32.2.1 _ _ | exact hr.w32.1 _ _

theorem lane_ldH (x : LCtx) (i : Nat) (d : Nat) (a : IExp) (ws : WSt F32) (cl : Bool) (li : LI)
    (R : List Answer) (h : LanePre hw cta im S MF MB MI WS l x i (.ldH d a) ws cl li R) :
    LaneEnds hw cta im S MF MB MI WS l x i (.ldH d a) ws cl li R := by
  have hr := h.rel
  have ha := iexp_agree hr.reg h.lv h.cv hr.ptr a
  obtain ⟨hcl, hs36, hA⟩ := h.ok
  have hv := load_half (hA l) hs36 (hr.sm hcl)
  rw [← h.smq] at hv
  refine ⟨_, by simp only [laneI, ha, hv]; rfl, ?_, by simp [trace, callLeaf, LI.set],
    by simp [trace, storeLeaf, LI.set], by simpa [trace, ansLeaf, LI.set] using h.tape⟩
  refine ⟨fun r => ?_, hr.f, hr.fmem, hr.bmem, hr.imem, hr.wr, hr.asz, hr.sm, hr.nan, hr.start, hr.ptr, hr.w32⟩
  simp only [LI.set, run, EWStmt.run, WSt.withInt, ISt.set, ISt.half]
  by_cases hrd : r = d <;> simp [hrd, hr.reg r]

theorem lane_stH (x : LCtx) (i : Nat) (a v g : IExp) (ws : WSt F32) (cl : Bool) (li : LI)
    (R : List Answer) (h : LanePre hw cta im S MF MB MI WS l x i (.stH a v g) ws cl li R) :
    LaneEnds hw cta im S MF MB MI WS l x i (.stH a v g) ws cl li R := by
  have hr := h.rel
  have ea := iexp_agree hr.reg h.lv h.cv hr.ptr a
  have ev := iexp_agree hr.reg h.lv h.cv hr.ptr v
  have eg := iexp_agree hr.reg h.lv h.cv hr.ptr g
  have hrun : (run hw cta im (.stH a v g) ws i).int.regs = ws.int.regs ∧
      (run hw cta im (.stH a v g) ws i).int.ptr = ws.int.ptr ∧
      (run hw cta im (.stH a v g) ws i).int.gmem = ws.int.gmem :=
    ⟨(storeH_regs _ _ _ _).1, (storeH_regs _ _ _ _).2.2, (storeH_regs _ _ _ _).2.1⟩
  have hcl : lastOr cl (trace hw cta im shLeaf i (.stH a v g) ws) = false := rfl
  by_cases hg : g.eval ws.int cta l ≠ 0
  · have hA := h.ok.2 l hg
    have hd := arena_decodeL (a := a.eval ws.int cta l) (by have := h.ok.1; omega)
    rw [← h.smq] at hd
    obtain ⟨m', hm'⟩ : ∃ m', li.w.mem.store (x.sm + a.eval ws.int cta l) 2 (v.eval ws.int cta l &&& 0xffff) = some m' :=
      Writable.stores ⟨.arena, _, hd, by show _ ≤ li.w.mem.arena.size; rw [hr.asz]; exact hA, by decide⟩ _
    refine ⟨{ li with w := { obsStore li.w (x.sm + a.eval ws.int cta l) 2 (v.eval ws.int cta l &&& 0xffff) with mem := m' } },
      by simp only [laneI, eg, if_pos hg, ea, ev, stw, hm']; rfl, ?_, ?_, ?_, ?_⟩
    · refine ⟨fun r => by rw [hr.reg r]; exact (congrFun hrun.1 r ▸ rfl), hr.f, fun b j hb => ?_, fun a' ha' => ?_,
        fun b j hb => ?_, fun a' n ha' => (hr.wr a' n ha').store hm', ?_, fun hc => by simp [hcl] at hc,
        (Mem.store_nan hm').trans hr.nan, by simp [obsStore]; have := hr.start; omega,
        fun b => (hr.ptr b).trans (congrFun hrun.2.1 b).symm, hr.w32⟩
      · exact load_arenaStore hm' hd (h.naF l b j hb) (hr.fmem b j hb)
      · rw [hrun.2.2]; exact load_arenaStore hm' hd (h.naB l a' ha') (hr.bmem a' ha')
      · exact load_arenaStore hm' hd (h.naI b j hb) (hr.imem b j hb)
      · show m'.arena.size = S
        rw [show m'.arena.size = (m'.region .arena).size from rfl, HProg.Contracts.store_sizes hm']; exact hr.asz
    · show collCalls (obsStore li.w _ _ _) x.start = _
      simp [collCalls_obsStore hr.start, trace, callLeaf]
    · show storesOf (obsStore li.w _ _ _) x.start = _
      simp [storesOf_store hr.start, trace, storeLeaf, hg]
    · show x.ans.drop (collCalls (obsStore li.w _ _ _) x.start).length = R
      rw [collCalls_obsStore hr.start]; simpa [trace, ansLeaf] using h.tape
  · have hg0 : g.eval ws.int cta l = 0 := by simpa using hg
    refine ⟨li, by simp only [laneI, eg, hg0, ne_eq, not_true_eq_false, if_false], ?_, by simp [trace, callLeaf],
      by simp [trace, storeLeaf, hg0], by simpa [trace, ansLeaf] using h.tape⟩
    refine ⟨fun r => by rw [hr.reg r]; exact (congrFun hrun.1 r ▸ rfl), hr.f, hr.fmem, by rw [hrun.2.2]; exact hr.bmem,
      hr.imem, hr.wr,
      hr.asz, fun hc => by simp [hcl] at hc, hr.nan, hr.start, fun b => (hr.ptr b).trans (congrFun hrun.2.1 b).symm,
      hr.w32⟩

theorem lane_stB (x : LCtx) (i : Nat) (a v g : IExp) (ws : WSt F32) (cl : Bool) (li : LI)
    (R : List Answer) (h : LanePre hw cta im S MF MB MI WS l x i (.stB a v g) ws cl li R) :
    LaneEnds hw cta im S MF MB MI WS l x i (.stB a v g) ws cl li R := by
  have hr := h.rel
  have ea := iexp_agree hr.reg h.lv h.cv hr.ptr a
  have ev := iexp_agree hr.reg h.lv h.cv hr.ptr v
  have eg := iexp_agree hr.reg h.lv h.cv hr.ptr g
  have hok : ∀ l', g.eval ws.int cta l' ≠ 0 → _ := h.ok
  have hst := storeB_regs (fun l => a.eval ws.int cta l) (fun l => v.eval ws.int cta l) (fun l => g.eval ws.int cta l)
    ws.int
  have hrun : (run hw cta im (.stB a v g) ws i).int = ws.int.storeB (fun l => a.eval ws.int cta l)
      (fun l => v.eval ws.int cta l) (fun l => g.eval ws.int cta l) := rfl
  have hcl : lastOr cl (trace hw cta im shLeaf i (.stB a v g) ws) = cl := rfl
  -- no lane but `l` stores a byte `l` reads
  have hno : ∀ a', MB l a' → (g.eval ws.int cta l ≠ 0 → a' ≠ a.eval ws.int cta l) →
      ∀ l', g.eval ws.int cta l' ≠ 0 → (a.eval ws.int cta l').toNat ≠ a'.toNat := by
    intro a' ha' hnot l' hg' e
    have e' : a.eval ws.int cta l' = a' := UInt64.toNat_inj.mp e
    obtain ⟨⟨-, r, o, hd, -, -, hMB, -⟩, -⟩ := hok l' hg'
    rcases hMB l a' ha' with ⟨rfl, hown⟩ | ⟨r', o', hd', hap⟩
    · exact hnot hg' hown
    · exact apart_ne (by decide) (by decide) hd hd' hap e'
  by_cases hg : g.eval ws.int cta l ≠ 0
  · obtain ⟨⟨hws, r, o, hd, hra, hMF, hMB, hMI⟩, hcol⟩ := hok l hg
    obtain ⟨m', hm'⟩ := (hr.wr _ _ hws).stores (v.eval ws.int cta l &&& 0xff)
    obtain ⟨r0, o0, hd0, hF⟩ := store_frame hm'
    rw [hd] at hd0; obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hd0)
    have harena : m'.arena = li.w.mem.arena := hF.other .arena (Ne.symm hra)
    refine ⟨{ li with w := { obsStore li.w (a.eval ws.int cta l) 1 (v.eval ws.int cta l &&& 0xff) with mem := m' } },
      by simp only [laneI, eg, if_pos hg, ea, ev, stw, hm']; rfl, ?_, ?_, ?_, ?_⟩
    · refine ⟨fun r' => by rw [hr.reg r', hrun, hst.1], hr.f, fun b j hb => ?_, fun a' ha' => ?_,
        fun b j hb => ?_, fun a' n ha' => (hr.wr a' n ha').store hm', ?_, fun hc => ?_,
        (Mem.store_nan hm').trans hr.nan, by simp [obsStore]; have := hr.start; omega,
        fun b => by rw [hr.ptr b, hrun, hst.2.2], hr.w32⟩
      · rcases hMF l b j hb with ⟨-, hf⟩ | ⟨r', o', hd', hap⟩
        · exact hf.elim
        · rw [load_apart hm' hd hd' hap]; exact hr.fmem b j hb
      · by_cases hown : a' = a.eval ws.int cta l
        · subst hown
          rw [load_store1 hm', byte_low, hrun]
          rw [storeB_at _ _ _ _ l hg (fun l' hg' e => hcol l' hg' (UInt64.toNat_inj.mp e) ha')]
        · rcases hMB l a' ha' with ⟨-, hown'⟩ | ⟨r', o', hd', hap⟩
          · exact absurd hown' hown
          · rw [load_apart hm' hd hd' hap, hrun, storeB_other _ _ _ _ _ (hno a' ha' (fun _ => hown))]
            exact hr.bmem a' ha'
      · obtain ⟨r', o', hd', hap⟩ := hMI b j hb
        rw [load_apart hm' hd hd' hap]; exact hr.imem b j hb
      · show m'.arena.size = S; rw [harena]; exact hr.asz
      · show m'.arena = _; rw [harena, hr.sm hc, hrun, hst.2.1]
    · show collCalls (obsStore li.w _ _ _) x.start = _
      simp [collCalls_obsStore hr.start, trace, callLeaf]
    · show storesOf (obsStore li.w _ _ _) x.start = _
      simp [storesOf_store hr.start, trace, storeLeaf, hg]
    · show x.ans.drop (collCalls (obsStore li.w _ _ _) x.start).length = R
      rw [collCalls_obsStore hr.start]; simpa [trace, ansLeaf] using h.tape
  · have hg0 : g.eval ws.int cta l = 0 := by simpa using hg
    refine ⟨li, by simp only [laneI, eg, hg0, ne_eq, not_true_eq_false, if_false], ?_, by simp [trace, callLeaf],
      by simp [trace, storeLeaf, hg0], by simpa [trace, ansLeaf] using h.tape⟩
    refine ⟨fun r' => by rw [hr.reg r', hrun, hst.1], hr.f, hr.fmem, fun a' ha' => ?_, hr.imem, hr.wr,
      hr.asz, fun hc => by rw [hr.sm hc, hrun, hst.2.1], hr.nan, hr.start, fun b => by rw [hr.ptr b, hrun, hst.2.2],
      hr.w32⟩
    rw [hrun, storeB_other _ _ _ _ _ (hno a' ha' (fun h' => absurd hg0 h'))]
    exact hr.bmem a' ha'

theorem storeLanes_int (b : Buf) (ix : Lane → Nat) (r : Nat) (st : WSt F32) :
    (st.storeLanes b ix r).int = st.int ∧ (st.storeLanes b ix r).regs = st.regs ∧ (st.storeLanes b ix r).smem = st.smem := by
  unfold WSt.storeLanes
  exact foldl_inv _ (fun s : WSt F32 => s.int = st.int ∧ s.regs = st.regs ∧ s.smem = st.smem)
    (fun s l h => by simp only [WSt.store1]; exact h) _ st ⟨rfl, rfl, rfl⟩

theorem storeLanes_w32 (b : Buf) (ix : Lane → Nat) (r : Nat) (st : WSt F32) (h : W32 st) : W32 (st.storeLanes b ix r) := by
  obtain ⟨h1, h2, h3⟩ := storeLanes_int b ix r st
  refine ⟨fun r' l => by rw [h2]; exact h.1 r' l, ?_, fun k => by rw [h3]; exact h.2.2 k⟩
  unfold WSt.storeLanes
  exact foldl_inv _ (fun s : WSt F32 => ∀ b j, (s.mem b j).bits.toNat < 2 ^ 32)
    (fun s l hs b' j' => by
      simp only [WSt.store1]
      by_cases hc : b' = b ∧ j' = ix l
      · rw [if_pos hc]; exact h.1 r l
      · rw [if_neg hc]; exact hs b' j') _ st h.2.1

theorem store1_int (st : WSt F32) (b : Buf) (j : Nat) (v : F32) : (st.store1 b j v).int = st.int ∧
    (st.store1 b j v).regs = st.regs ∧ (st.store1 b j v).smem = st.smem := ⟨rfl, rfl, rfl⟩

theorem lane_storeLane (x : LCtx) (i : Nat) (b : Buf) (ix : IdxE) (n : Nat) (ws : WSt F32) (cl : Bool) (li : LI)
    (R : List Answer) (h : LanePre hw cta im S MF MB MI WS l x i (.storeLane b ix n) ws cl li R) :
    LaneEnds hw cta im S MF MB MI WS l x i (.storeLane b ix n) ws cl li R := by
  have hr := h.rel
  have hok := h.ok
  obtain ⟨hix, ⟨hws, r, o, hd, hra, hMF, hMB, hMI⟩, hcol⟩ := hok l
  obtain ⟨hj, hj32⟩ := idx_agree h.lv h.cv h.ieq hr.imem ix hix
  generalize hJ : ix.eval cta i l (fun _ _ => 0) im = J at hj hj32 hws hd hMF hMB hMI hcol
  obtain ⟨m', hm'⟩ := (hr.wr _ _ hws).stores (li.f n).bits
  obtain ⟨r0, o0, hd0, hF⟩ := store_frame hm'
  rw [hd] at hd0; obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hd0)
  have harena : m'.arena = li.w.mem.arena := hF.other .arena (Ne.symm hra)
  have hrun : run hw cta im (.storeLane b ix n) ws i =
      ws.storeLanes b (fun l => ix.eval cta i l (fun _ _ => 0) im) n := rfl
  obtain ⟨hi1, hi2, hi3⟩ := storeLanes_int b (fun l => ix.eval cta i l (fun _ _ => 0) im) n ws
  have hcl : lastOr cl (trace hw cta im shLeaf i (.storeLane b ix n) ws) = cl := rfl
  refine ⟨{ li with w := { obsStore li.w (x.ptr b + off (UInt32.ofNat J)) 4 (li.f n).bits with mem := m' } },
    by simp only [laneI, hj, Option.bind_eq_bind, Option.bind_some, stw, hm'], ?_, ?_, ?_, ?_⟩
  · rw [hrun]
    refine ⟨fun r' => by rw [hr.reg r', hi1], fun r' => by rw [hr.f r', hi2], fun b' j' hb' => ?_,
      fun a' ha' => ?_, fun b' j' hb' => ?_, fun a' k ha' => (hr.wr a' k ha').store hm',
      by show m'.arena.size = S; rw [harena]; exact hr.asz, fun hc => by show m'.arena = _; rw [harena, hr.sm hc, hi1],
      (Mem.store_nan hm').trans hr.nan, by simp [obsStore]; have := hr.start; omega,
      fun b'' => by rw [hr.ptr b'', hi1], storeLanes_w32 _ _ _ _ hr.w32⟩
    · by_cases hown : b' = b ∧ j' = J
      · obtain ⟨rfl, rfl⟩ := hown
        rw [load_store4 (hr.w32.1 n l |> fun h32 => by rw [hr.f n]; exact h32) hm', hr.f n]
        rw [← hJ, @storeLane_at F32 (F32.ops hw) _ _ _ _ _ l (fun l' e => by rw [hcol l' (e.trans hJ) hb'])]
      · rcases hMF l b' j' hb' with ⟨-, hown'⟩ | ⟨r', o', hd', hap⟩
        · exact absurd hown' hown
        · rw [load_apart hm' hd hd' hap, hr.fmem b' j' hb']
          by_cases hbb : b' = b
          · subst hbb
            rw [@storeLane_otherAddr F32 (F32.ops hw) _ _ _ _ _ _ (fun l'' e => ?_)]
            have hj' : j' ≠ J := fun e => hown ⟨rfl, e⟩
            by_cases hl : l'' = l
            · subst hl; exact hj' (e.symm.trans hJ)
            · obtain ⟨hix'', ⟨-, r2, o2, hd2, -, hMF2, -, -⟩, -⟩ := hok l''
              rcases hMF2 l b' j' hb' with ⟨hll, -⟩ | ⟨r3, o3, hd3, hap3⟩
              · exact hl hll.symm
              · exact apart_ne (by decide) (by decide) hd2 hd3 hap3 (by rw [e])
          · rw [@storeLane_otherBuf F32 (F32.ops hw) _ _ _ _ _ _ hbb]
    · rcases hMB l a' ha' with ⟨-, hf⟩ | ⟨r', o', hd', hap⟩
      · exact hf.elim
      · rw [load_apart hm' hd hd' hap, hi1]; exact hr.bmem a' ha'
    · obtain ⟨r', o', hd', hap⟩ := hMI b' j' hb'
      rw [load_apart hm' hd hd' hap]; exact hr.imem b' j' hb'
  · show collCalls (obsStore li.w _ _ _) x.start = _
    simp [collCalls_obsStore hr.start, trace, callLeaf]
  · show storesOf (obsStore li.w _ _ _) x.start = _
    simp [storesOf_store hr.start, trace, storeLeaf, hJ, hr.f n]
  · show x.ans.drop (collCalls (obsStore li.w _ _ _) x.start).length = R
    rw [collCalls_obsStore hr.start]; simpa [trace, ansLeaf] using h.tape

theorem lv_zero {x : LCtx} (hlv : x.lv = UInt64.ofNat l.val) : x.lv = 0 ↔ l = lane0 := by
  have hl : l.val < 32 := l.isLt
  constructor
  · intro h0; rw [hlv] at h0
    apply Fin.ext
    show l.val = 0
    have := congrArg UInt64.toNat h0
    simp at this; omega
  · rintro rfl; rw [hlv]; rfl

theorem lane_storeLane0 (x : LCtx) (i : Nat) (b : Buf) (ix : IdxE) (n : Nat) (ws : WSt F32) (cl : Bool) (li : LI)
    (R : List Answer) (h : LanePre hw cta im S MF MB MI WS l x i (.storeLane0 b ix n) ws cl li R) :
    LaneEnds hw cta im S MF MB MI WS l x i (.storeLane0 b ix n) ws cl li R := by
  have hr := h.rel
  obtain ⟨hix, hws, r, o, hd, hra, hMF, hMB, hMI⟩ := h.ok
  generalize hJ : ix.eval cta i lane0 (fun _ _ => 0) im = J at hws hd hMF hMB hMI
  have hrun : run hw cta im (.storeLane0 b ix n) ws i = ws.store1 b J (ws.regs n lane0) := by
    rw [← hJ]; rfl
  have hcl : lastOr cl (trace hw cta im shLeaf i (.storeLane0 b ix n) ws) = cl := rfl
  by_cases hl0 : l = lane0
  · subst hl0
    obtain ⟨hj, hj32⟩ := idx_agree h.lv h.cv h.ieq hr.imem ix hix
    rw [hJ] at hj
    obtain ⟨m', hm'⟩ := (hr.wr _ _ hws).stores (li.f n).bits
    obtain ⟨r0, o0, hd0, hF⟩ := store_frame hm'
    rw [hd] at hd0; obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hd0)
    have harena : m'.arena = li.w.mem.arena := hF.other .arena (Ne.symm hra)
    have h0 : x.lv = 0 := (lv_zero _ h.lv).mpr rfl
    refine ⟨{ li with w := { obsStore li.w (x.ptr b + off (UInt32.ofNat J)) 4 (li.f n).bits with mem := m' } },
      by simp only [laneI, h0, if_true, hj, Option.bind_eq_bind, Option.bind_some, stw, hm'], ?_, ?_, ?_, ?_⟩
    · rw [hrun]
      refine ⟨fun r' => hr.reg r', fun r' => hr.f r', fun b' j' hb' => ?_,
        fun a' ha' => ?_, fun b' j' hb' => ?_, fun a' k ha' => (hr.wr a' k ha').store hm',
        by show m'.arena.size = S; rw [harena]; exact hr.asz, fun hc => by show m'.arena = _; rw [harena, hr.sm hc]; rfl,
        (Mem.store_nan hm').trans hr.nan, by simp [obsStore]; have := hr.start; omega,
        fun b'' => hr.ptr b'', ⟨hr.w32.1, fun b1 j1 => ?_, hr.w32.2.2⟩⟩
      · by_cases hown : b' = b ∧ j' = J
        · obtain ⟨rfl, rfl⟩ := hown
          rw [load_store4 (by rw [hr.f n]; exact hr.w32.1 n lane0) hm', hr.f n]
          simp [WSt.store1]
        · rcases hMF lane0 b' j' hb' with ⟨-, hown'⟩ | ⟨r', o', hd', hap⟩
          · exact absurd hown' hown
          · rw [load_apart hm' hd hd' hap, hr.fmem b' j' hb']
            simp only [WSt.store1, if_neg hown]
      · rcases hMB lane0 a' ha' with ⟨-, hf⟩ | ⟨r', o', hd', hap⟩
        · exact hf.elim
        · rw [load_apart hm' hd hd' hap]; exact hr.bmem a' ha'
      · obtain ⟨r', o', hd', hap⟩ := hMI b' j' hb'
        rw [load_apart hm' hd hd' hap]; exact hr.imem b' j' hb'
      · simp only [WSt.store1]; split
        · exact hr.w32.1 n lane0
        · exact hr.w32.2.1 b1 j1
    · show collCalls (obsStore li.w _ _ _) x.start = _
      simp [collCalls_obsStore hr.start, trace, callLeaf]
    · show storesOf (obsStore li.w _ _ _) x.start = _
      simp [storesOf_store hr.start, trace, storeLeaf, hJ, hr.f n]
    · show x.ans.drop (collCalls (obsStore li.w _ _ _) x.start).length = R
      rw [collCalls_obsStore hr.start]; simpa [trace, ansLeaf] using h.tape
  · have hne : ¬ x.lv = 0 := fun e => hl0 ((lv_zero _ h.lv).mp e)
    refine ⟨li, by simp only [laneI, hne, if_false], ?_, by simp [trace, callLeaf], by simp [trace, storeLeaf, hl0],
      by simpa [trace, ansLeaf] using h.tape⟩
    rw [hrun]
    refine ⟨hr.reg, hr.f, fun b' j' hb' => ?_, hr.bmem, hr.imem, hr.wr, hr.asz, hr.sm, hr.nan, hr.start, hr.ptr,
      ⟨hr.w32.1, fun b1 j1 => ?_, hr.w32.2.2⟩⟩
    · rcases hMF l b' j' hb' with ⟨hll, -⟩ | ⟨r', o', hd', hap⟩
      · exact absurd hll hl0
      · have hne' : ¬ (b' = b ∧ j' = J) := by
          rintro ⟨rfl, rfl⟩
          exact apart_ne (by decide) (by decide) hd hd' hap rfl
        simp only [WSt.store1, if_neg hne']; exact hr.fmem b' j' hb'
    · simp only [WSt.store1]; split
      · exact hr.w32.1 n lane0
      · exact hr.w32.2.1 b1 j1

theorem lastOr_append (d : Bool) (A B : List Bool) : lastOr d (A ++ B) = lastOr (lastOr d A) B := by
  unfold lastOr
  rw [List.getLast?_append]
  cases B.getLast? <;> simp

/-- The relation sees of the context its buffers and its first observation alone. -/
theorem RelW.ctx {x x' : LCtx} {ws : WSt F32} {cl : Bool} {li : LI} (h : RelW hw S MF MB MI WS im l x ws cl li)
    (hp : x'.ptr = x.ptr) (hs : x'.start = x.start) : RelW hw S MF MB MI WS im l x' ws cl li := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12⟩ := h
  exact ⟨h1, h2, by rw [hp]; exact h3, h4, by rw [hp]; exact h5, h6, h7, h8, h9, by rw [hs]; exact h10,
    by rw [hp]; exact h11, h12⟩

theorem lane_seq (a b : EWStmt)
    (iha : ∀ x i ws cl li R, LanePre hw cta im S MF MB MI WS l x i a ws cl li R →
      LaneEnds hw cta im S MF MB MI WS l x i a ws cl li R)
    (ihb : ∀ x i ws cl li R, LanePre hw cta im S MF MB MI WS l x i b ws cl li R →
      LaneEnds hw cta im S MF MB MI WS l x i b ws cl li R)
    (x : LCtx) (i : Nat) (ws : WSt F32) (cl : Bool) (li : LI) (R : List Answer)
    (h : LanePre hw cta im S MF MB MI WS l x i (.seq a b) ws cl li R) :
    LaneEnds hw cta im S MF MB MI WS l x i (.seq a b) ws cl li R := by
  obtain ⟨hok1, hok2⟩ := h.ok
  have ht := h.tape
  simp only [trace, List.append_assoc] at ht
  obtain ⟨l1, r1, rel1, c1, s1, t1⟩ := iha x i ws cl li _ { h with ok := hok1, tape := ht }
  obtain ⟨l2, r2, rel2, c2, s2, t2⟩ := ihb x i _ _ l1 R { h with ok := hok2, rel := rel1, tape := t1 }
  refine ⟨l2, by simp only [laneI, bind, Option.bind, r1, r2], ?_, ?_, ?_, t2⟩
  · simp only [run, EWStmt.run, trace, lastOr_append] at rel2 ⊢; exact rel2
  · rw [c2, c1]; simp [trace]
  · rw [s2, s1]; simp [trace]

theorem lane_iif (c : IExp) (a b : EWStmt)
    (iha : ∀ x i ws cl li R, LanePre hw cta im S MF MB MI WS l x i a ws cl li R →
      LaneEnds hw cta im S MF MB MI WS l x i a ws cl li R)
    (ihb : ∀ x i ws cl li R, LanePre hw cta im S MF MB MI WS l x i b ws cl li R →
      LaneEnds hw cta im S MF MB MI WS l x i b ws cl li R)
    (x : LCtx) (i : Nat) (ws : WSt F32) (cl : Bool) (li : LI) (R : List Answer)
    (h : LanePre hw cta im S MF MB MI WS l x i (.iif c a b) ws cl li R) :
    LaneEnds hw cta im S MF MB MI WS l x i (.iif c a b) ws cl li R := by
  obtain ⟨hu, hok⟩ := h.ok
  have hr := h.rel
  have ec := iexp_agree hr.reg h.lv h.cv hr.ptr c
  by_cases hc : c.eval ws.int cta lane0 ≠ 0
  · have hcl : c.evalL x li.reg ≠ 0 := by rw [ec]; exact (hu l).mpr hc
    rw [if_pos hc] at hok
    have ht := h.tape
    simp only [trace, if_pos hc] at ht
    obtain ⟨l1, r1, rel1, c1, s1, t1⟩ := iha x i ws cl li R { h with ok := hok, tape := ht }
    refine ⟨l1, by simp only [laneI, if_pos hcl, r1], ?_, ?_, ?_, t1⟩
    · have e : run hw cta im (.iif c a b) ws i = run hw cta im a ws i := if_pos hc
      rw [e]; simpa only [trace, if_pos hc] using rel1
    · simpa only [trace, if_pos hc] using c1
    · simpa only [trace, if_pos hc] using s1
  · have hcl : ¬ c.evalL x li.reg ≠ 0 := by rw [ec]; exact fun e => hc ((hu l).mp e)
    rw [if_neg hc] at hok
    have ht := h.tape
    simp only [trace, if_neg hc] at ht
    obtain ⟨l1, r1, rel1, c1, s1, t1⟩ := ihb x i ws cl li R { h with ok := hok, tape := ht }
    refine ⟨l1, by simp only [laneI, if_neg hcl, r1], ?_, ?_, ?_, t1⟩
    · have e : run hw cta im (.iif c a b) ws i = run hw cta im b ws i := if_neg hc
      rw [e]; simpa only [trace, if_neg hc] using rel1
    · simpa only [trace, if_neg hc] using c1
    · simpa only [trace, if_neg hc] using s1

theorem lane_locals (rs : List Nat) (b : EWStmt)
    (ihb : ∀ x i ws cl li R, LanePre hw cta im S MF MB MI WS l x i b ws cl li R →
      LaneEnds hw cta im S MF MB MI WS l x i b ws cl li R)
    (x : LCtx) (i : Nat) (ws : WSt F32) (cl : Bool) (li : LI) (R : List Answer)
    (h : LanePre hw cta im S MF MB MI WS l x i (.locals rs b) ws cl li R) :
    LaneEnds hw cta im S MF MB MI WS l x i (.locals rs b) ws cl li R := by
  have hr := h.rel
  obtain ⟨l1, r1, rel1, c1, s1, t1⟩ := ihb x i ws cl li R { h with ok := h.ok, tape := h.tape }
  refine ⟨restoreL rs li l1, by simp only [laneI, r1, Option.map_some], ?_, c1, s1, t1⟩
  refine ⟨fun r => ?_, rel1.f, rel1.fmem, rel1.bmem, rel1.imem, rel1.wr, rel1.asz, rel1.sm, rel1.nan, rel1.start,
    rel1.ptr, rel1.w32⟩
  simp only [restoreL, run, EWStmt.run, WSt.withInt, ISt.restore]
  split
  · exact hr.reg r
  · exact rel1.reg r

/-- What every trip of a loop's lane starts from but its own conditions. -/
structure LaneCtx (x : LCtx) : Prop where
  hwq : x.hw = hw
  lv : x.lv = UInt64.ofNat l.val
  cv : x.cv = UInt64.ofNat cta
  cta32 : cta < 2 ^ 32
  smq : x.sm = addrOf .arena 0
  naF : ∀ l b j, MF l b j → NotArena (x.ptr b + off (UInt32.ofNat j))
  naB : ∀ l a, MB l a → NotArena a
  naI : ∀ b j, MI b j → NotArena (x.ptr b + off (UInt32.ofNat j))

/-- **A counted loop's lane runs its trips with the warp**. -/
theorem lane_trips (b : EWStmt)
    (ihb : ∀ x i ws cl li R, LanePre hw cta im S MF MB MI WS l x i b ws cl li R →
      LaneEnds hw cta im S MF MB MI WS l x i b ws cl li R)
    (x : LCtx) (hx : LaneCtx hw cta MF MB MI l x) (ws : WSt F32) (cl : Bool) (li : LI) (n : Nat)
    (hok : ∀ j, j < n → OkW hw cta im S x.ptr MF MB MI WS j b (run hw cta im (.forN j b) ws 0)
      (lastOr cl (trace hw cta im shLeaf 0 (.forN j b) ws)))
    (hrel : RelW hw S MF MB MI WS im l x ws cl li) :
    ∀ k, k ≤ n → ∀ R', x.ans.drop (collCalls li.w x.start).length = trace hw cta im (ansLeaf cta S l) 0 (.forN k b) ws ++ R' →
      ∃ lk, (List.range k).foldlM (fun s j => laneI { x with i := j } b s) li = some lk ∧
        RelW hw S MF MB MI WS im l x (run hw cta im (.forN k b) ws 0) (lastOr cl (trace hw cta im shLeaf 0 (.forN k b) ws)) lk ∧
        collCalls lk.w x.start = collCalls li.w x.start ++ trace hw cta im (callLeaf cta l) 0 (.forN k b) ws ∧
        storesOf lk.w x.start = storesOf li.w x.start ++ trace hw cta im (storeLeaf cta im l x.ptr x.sm) 0 (.forN k b) ws ∧
        x.ans.drop (collCalls lk.w x.start).length = R' := by
  intro k
  induction k with
  | zero => intro _ R' ht; exact ⟨li, rfl, hrel, by simp [trace], by simp [trace], by simpa [trace] using ht⟩
  | succ k ih =>
      intro hk R' ht
      rw [trace_forN_succ, List.append_assoc] at ht
      obtain ⟨lk, rk, relk, ck, sk, tk⟩ := ih (by omega) _ ht
      have hpre : LanePre hw cta im S MF MB MI WS l { x with i := k } k b (run hw cta im (.forN k b) ws 0)
          (lastOr cl (trace hw cta im shLeaf 0 (.forN k b) ws)) lk R' :=
        ⟨hx.hwq, hx.lv, hx.cv, hx.cta32, rfl, hx.smq, hok k (by omega), RelW.ctx _ _ _ _ _ _ _ _ relk rfl rfl, tk,
          hx.naF, hx.naB, hx.naI⟩
      obtain ⟨l1, r1, rel1, c1, s1, t1⟩ := ihb _ _ _ _ _ _ hpre
      refine ⟨l1, ?_, ?_, ?_, ?_, t1⟩
      · rw [List.range_succ, List.foldlM_append, rk]; simpa using r1
      · rw [run_forN_succ, trace_forN_succ, lastOr_append]; exact RelW.ctx _ _ _ _ _ _ _ _ rel1 rfl rfl
      · rw [c1, ck, trace_forN_succ, List.append_assoc]
      · rw [s1, sk, trace_forN_succ, List.append_assoc]

theorem LanePre.lctx {x : LCtx} {i : Nat} {s : EWStmt} {ws : WSt F32} {cl : Bool} {li : LI} {R : List Answer}
    (h : LanePre hw cta im S MF MB MI WS l x i s ws cl li R) : LaneCtx hw cta MF MB MI l x :=
  ⟨h.hwq, h.lv, h.cv, h.cta32, h.smq, h.naF, h.naB, h.naI⟩

theorem lane_forN (n : Nat) (b : EWStmt)
    (ihb : ∀ x i ws cl li R, LanePre hw cta im S MF MB MI WS l x i b ws cl li R →
      LaneEnds hw cta im S MF MB MI WS l x i b ws cl li R)
    (x : LCtx) (i : Nat) (ws : WSt F32) (cl : Bool) (li : LI) (R : List Answer)
    (h : LanePre hw cta im S MF MB MI WS l x i (.forN n b) ws cl li R) :
    LaneEnds hw cta im S MF MB MI WS l x i (.forN n b) ws cl li R := by
  obtain ⟨hn, hok⟩ := h.ok
  obtain ⟨lk, rk, relk, ck, sk, tk⟩ := lane_trips hw cta im S MF MB MI WS l b ihb x h.lctx ws cl li n hok h.rel n
    (Nat.le_refl _) R (by rw [← trace_forN_i]; exact h.tape)
  exact ⟨lk, by simp only [laneI, if_pos hn, rk], by rw [run_forN_i, trace_forN_i]; exact relk,
    by rw [trace_forN_i]; exact ck, by rw [trace_forN_i]; exact sk, tk⟩

theorem lane_forM (bu : Buf) (a : Nat) (b : EWStmt)
    (ihb : ∀ x i ws cl li R, LanePre hw cta im S MF MB MI WS l x i b ws cl li R →
      LaneEnds hw cta im S MF MB MI WS l x i b ws cl li R)
    (x : LCtx) (i : Nat) (ws : WSt F32) (cl : Bool) (li : LI) (R : List Answer)
    (h : LanePre hw cta im S MF MB MI WS l x i (.forM bu a b) ws cl li R) :
    LaneEnds hw cta im S MF MB MI WS l x i (.forM bu a b) ws cl li R := by
  obtain ⟨ha, hmi, hn, hok⟩ := h.ok
  have hv := h.rel.imem bu a hmi
  have hvn : (UInt64.ofNat (im bu a)).toNat = im bu a := by simp; omega
  obtain ⟨lk, rk, relk, ck, sk, tk⟩ := lane_trips hw cta im S MF MB MI WS l b ihb x h.lctx ws cl li (im bu a) hok
    h.rel (im bu a) (Nat.le_refl _) R h.tape
  exact ⟨lk, by simp only [laneI, if_pos ha, hv, Option.bind_eq_bind, Option.bind_some, hvn, if_pos hn, rk],
    relk, ck, sk, tk⟩

theorem lane_iwhile (n : Nat) (c : IExp) (cs : List Nat) (b : EWStmt)
    (ihb : ∀ x i ws cl li R, LanePre hw cta im S MF MB MI WS l x i b ws cl li R →
      LaneEnds hw cta im S MF MB MI WS l x i b ws cl li R)
    (x : LCtx) (i : Nat) (ws : WSt F32) (cl : Bool) (li : LI) (R : List Answer)
    (h : LanePre hw cta im S MF MB MI WS l x i (.iwhile n c cs b) ws cl li R) :
    LaneEnds hw cta im S MF MB MI WS l x i (.iwhile n c cs b) ws cl li R := by
  classical
  obtain ⟨hex, hall⟩ := h.ok
  have hr := h.rel
  -- the warp's states and traces after `j` trips
  let st : Nat → WSt F32 := fun j => run hw cta im (.iwhile j c cs b) ws i
  let T : ∀ {α : Type}, (Nat → EWStmt → WSt F32 → List α) → Nat → List α :=
    fun leaf j => trace hw cta im leaf i (.iwhile j c cs b) ws
  let go : Nat → Prop := fun j => c.eval (st j).int cta lane0 ≠ 0
  have hsucc : ∀ j, st (j + 1) = wtrip hw cta im c cs b ws i (st j) := fun j => run_iwhile_succ _ _ _ _ _ _ _ _ _
  have htsucc : ∀ {α : Type} (leaf : Nat → EWStmt → WSt F32 → List α) j,
      T leaf (j + 1) = T leaf j ++ (if go j then trace hw cta im leaf i b (st j) else []) :=
    fun leaf j => trace_iwhile_succ _ _ _ _ _ _ _ _ _ _
  -- the first trip where the test fails
  let P : Nat → Prop := fun k => ¬ go k
  obtain ⟨kw, hkw, hkw0⟩ := hex
  obtain ⟨hP0, hPmin⟩ := least_spec (P := P) (k := kw) (n := n) (fun e => e hkw0) hkw
  let k0 := least P n
  have hk0 : k0 ≤ n ∧ ¬ go k0 := ⟨Nat.le_trans (hPmin kw fun e => e hkw0) hkw, hP0⟩
  have hmin : ∀ j, j < k0 → go j := fun j hj =>
    Classical.byContradiction fun hn => absurd (hPmin j hn) (by omega)
  -- after it, nothing changes
  have hstop : ∀ m, st (k0 + m) = st k0 ∧ ∀ {α : Type} (leaf : Nat → EWStmt → WSt F32 → List α),
      T leaf (k0 + m) = T leaf k0 := by
    intro m
    induction m with
    | zero => exact ⟨rfl, fun _ => rfl⟩
    | succ m ihm =>
        have hg : ¬ go (k0 + m) := by
          show ¬ (c.eval (st (k0 + m)).int cta lane0 ≠ 0); rw [ihm.1]; exact hk0.2
        refine ⟨?_, fun leaf => ?_⟩
        · rw [← Nat.add_assoc, hsucc, ihm.1]; unfold wtrip; rw [if_neg hk0.2]
        · rw [← Nat.add_assoc, htsucc, if_neg hg, List.append_nil, ihm.2]
  -- a prefix's traces are prefixes
  have hpre : ∀ {α : Type} (leaf : Nat → EWStmt → WSt F32 → List α) d j, j + d = k0 →
      ∃ E, T leaf k0 = T leaf j ++ E := by
    intro α leaf d
    induction d with
    | zero => intro j hj; exact ⟨[], by rw [← hj]; simp⟩
    | succ d ihd =>
        intro j hj
        obtain ⟨E, hE⟩ := ihd (j + 1) (by omega)
        rw [htsucc] at hE
        exact ⟨_, by rw [hE, List.append_assoc]⟩
  let step : LI → Option LI := fun s' => (laneI x b s').map (keepL cs li)
  let gl : LI → Bool := fun s' => c.evalL x s'.reg != 0
  -- the trips from `j` on
  have claim : ∀ t j, j + t = k0 → ∀ lj : LI,
      RelW hw S MF MB MI WS im l x (st j) (lastOr cl (T shLeaf j)) lj →
      x.ans.drop (collCalls lj.w x.start).length = (T (ansLeaf cta S l) k0).drop (T (ansLeaf cta S l) j).length ++ R →
      collCalls lj.w x.start = collCalls li.w x.start ++ T (callLeaf cta l) j →
      storesOf lj.w x.start = storesOf li.w x.start ++ T (storeLeaf cta im l x.ptr x.sm) j →
      ∀ m, t ≤ m → ∃ le, whileL step gl m lj = some le ∧
        RelW hw S MF MB MI WS im l x (st k0) (lastOr cl (T shLeaf k0)) le ∧
        collCalls le.w x.start = collCalls li.w x.start ++ T (callLeaf cta l) k0 ∧
        storesOf le.w x.start = storesOf li.w x.start ++ T (storeLeaf cta im l x.ptr x.sm) k0 ∧
        x.ans.drop (collCalls le.w x.start).length = R := by
    intro t
    induction t with
    | zero =>
        intro j hj lj hrel ht hc hs m _
        have hj0 : j = k0 := by omega
        subst hj0
        have hgl : gl lj = false := by
          have e := iexp_agree hrel.reg h.lv h.cv hrel.ptr c
          have hu := ((hall k0 hk0.1).1 l)
          have hv0 : c.eval (st k0).int cta l = 0 := Classical.byContradiction fun hne => hk0.2 (hu.mp hne)
          show (c.evalL x lj.reg != 0) = false
          rw [e, hv0]; rfl
        exact ⟨lj, whileL_stop hgl m, hrel, hc, hs, by simpa using ht⟩
    | succ t iht =>
        intro j hj lj hrel ht hc hs m hm
        have hjk : j < k0 := by omega
        have hg := hmin j hjk
        have hgl : gl lj = true := by
          have e := iexp_agree hrel.reg h.lv h.cv hrel.ptr c
          have hu := ((hall j (by have := hk0.1; omega)).1 l)
          show (c.evalL x lj.reg != 0) = true
          rw [e]; exact bne_iff_ne.mpr (hu.mpr hg)
        obtain ⟨E, hE⟩ := hpre (ansLeaf cta S l) t (j + 1) (by omega)
        have htj := htsucc (ansLeaf cta S l) j
        rw [if_pos hg] at htj
        have htape : x.ans.drop (collCalls lj.w x.start).length =
            trace hw cta im (ansLeaf cta S l) i b (st j) ++
              ((T (ansLeaf cta S l) k0).drop (T (ansLeaf cta S l) (j + 1)).length ++ R) := by
          rw [ht, hE, htj]; simp
        have hpre' : LanePre hw cta im S MF MB MI WS l x i b (st j) (lastOr cl (T shLeaf j)) lj _ :=
          ⟨h.hwq, h.lv, h.cv, h.cta32, h.ieq, h.smq, (hall j (by have := hk0.1; omega)).2 hg, hrel, htape,
            h.naF, h.naB, h.naI⟩
        obtain ⟨l1, r1, rel1, c1, s1, t1⟩ := ihb _ _ _ _ _ _ hpre'
        obtain ⟨m', rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
        have hw1 : whileL step gl (m' + 1) lj = whileL step gl m' (keepL cs li l1) := by
          simp only [whileL, hgl, if_true, step, r1, Option.map_some, Option.bind_some]
        rw [hw1]
        refine iht (j + 1) (by omega) _ ?_ ?_ ?_ ?_ m' (by omega)
        · have hst : st (j + 1) = (run hw cta im b (st j) i).withInt
              (ISt.keep cs ws.int (run hw cta im b (st j) i).int) := by
            rw [hsucc]; unfold wtrip; rw [if_pos hg]
          rw [hst, htsucc, if_pos hg, lastOr_append]
          refine ⟨fun r => ?_, rel1.f, rel1.fmem, rel1.bmem, rel1.imem, rel1.wr, rel1.asz, rel1.sm, rel1.nan,
            rel1.start, rel1.ptr, rel1.w32⟩
          simp only [keepL, WSt.withInt, ISt.keep]
          split
          · exact rel1.reg r
          · exact hr.reg r
        · show x.ans.drop (collCalls l1.w x.start).length = _
          rw [t1]
        · show collCalls l1.w x.start = _
          rw [c1, hc, htsucc, if_pos hg, List.append_assoc]
        · show storesOf l1.w x.start = _
          rw [s1, hs, htsucc, if_pos hg, List.append_assoc]
  have hk0n : k0 ≤ n := hk0.1
  obtain ⟨hst, htr⟩ := hstop (n - k0)
  rw [show k0 + (n - k0) = n by omega] at hst htr
  obtain ⟨le, rle, rel, ce, se, te⟩ := claim k0 0 (by omega) li (by simpa [T, st, trace, lastOr] using hr)
    (by rw [show T (ansLeaf cta S l) 0 = [] from rfl, List.length_nil, List.drop_zero, ← htr]; exact h.tape)
    (by simp [T, trace]) (by simp [T, trace]) n hk0n
  refine ⟨le, by simpa only [laneI] using rle, ?_, ?_, ?_, te⟩
  · show RelW hw S MF MB MI WS im l x (st n) (lastOr cl (T shLeaf n)) le
    rw [hst, htr]; exact rel
  · show collCalls le.w x.start = collCalls li.w x.start ++ T (callLeaf cta l) n
    rw [htr]; exact ce
  · show storesOf le.w x.start = storesOf li.w x.start ++ T (storeLeaf cta im l x.ptr x.sm) n
    rw [htr]; exact se

/-- **A lane runs a warp statement with the warp**: from a state agreeing
    with the warp's, its answers the warp's tape for it, where the warp's run
    meets the conditions a lane needs (`OkW`), the lane's run ends agreeing
    with the warp's, its collectives and stores the warp's traces for it. -/
theorem lane_warp (hw32 : ∀ k v, (hw.approx k v).toNat < 2 ^ 32) :
    ∀ (s : EWStmt) (x : LCtx) (i : Nat) (ws : WSt F32) (cl : Bool) (li : LI) (R : List Answer),
      LanePre hw cta im S MF MB MI WS l x i s ws cl li R → LaneEnds hw cta im S MF MB MI WS l x i s ws cl li R := by
  intro s
  induction s with
  | skip => exact lane_skip _ _ _ _ _ _ _ _ _
  | setI d e => intro x i ws cl li R h; exact lane_setI _ _ _ _ _ _ _ _ _ x i d e ws cl li R h
  | setR d e => intro x i ws cl li R h; exact lane_setR _ _ _ _ _ _ _ _ _ hw32 x i d e ws cl li R h
  | ldB d a => intro x i ws cl li R h; exact lane_ldB _ _ _ _ _ _ _ _ _ x i d a ws cl li R h
  | stB a v g => intro x i ws cl li R h; exact lane_stB _ _ _ _ _ _ _ _ _ x i a v g ws cl li R h
  | ldH d a => intro x i ws cl li R h; exact lane_ldH _ _ _ _ _ _ _ _ _ x i d a ws cl li R h
  | stH a v g => intro x i ws cl li R h; exact lane_stH _ _ _ _ _ _ _ _ _ x i a v g ws cl li R h
  | ballot d p => intro x i ws cl li R h; exact lane_ballot _ _ _ _ _ _ _ _ _ x i d p ws cl li R h
  | shflIdx d v sr => intro x i ws cl li R h; exact lane_shflIdx _ _ _ _ _ _ _ _ _ x i d v sr ws cl li R h
  | shflXor d sr m => intro x i ws cl li R h; exact lane_shflXor _ _ _ _ _ _ _ _ _ x i d sr m ws cl li R h
  | barrier => exact lane_barrier _ _ _ _ _ _ _ _ _
  | loadIdx d b ix => intro x i ws cl li R h; exact lane_loadIdx _ _ _ _ _ _ _ _ _ x i d b ix ws cl li R h
  | loadV4 d0 d1 d2 d3 b ix => intro x i ws cl li R h; exact lane_loadV4 _ _ _ _ _ _ _ _ _ x i d0 d1 d2 d3 b ix ws cl li R h
  | storeLane b ix n => intro x i ws cl li R h; exact lane_storeLane _ _ _ _ _ _ _ _ _ x i b ix n ws cl li R h
  | storeLane0 b ix n => intro x i ws cl li R h; exact lane_storeLane0 _ _ _ _ _ _ _ _ _ x i b ix n ws cl li R h
  | seq a b iha ihb => exact lane_seq _ _ _ _ _ _ _ _ _ a b iha ihb
  | iif c a b iha ihb => exact lane_iif _ _ _ _ _ _ _ _ _ c a b iha ihb
  | locals rs b ih => exact lane_locals _ _ _ _ _ _ _ _ _ rs b ih
  | forN n b ih => exact lane_forN _ _ _ _ _ _ _ _ _ n b ih
  | forM bu a b ih => exact lane_forM _ _ _ _ _ _ _ _ _ bu a b ih
  | iwhile n c cs b ih => exact lane_iwhile _ _ _ _ _ _ _ _ _ n c cs b ih
  | stSm | ldSm | cvtIF => intro x i ws cl li R h; exact h.ok.elim

end Lane

end AlgorithmLib.ML.LaneWarp
