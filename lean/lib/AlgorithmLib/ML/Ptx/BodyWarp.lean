module
public import AlgorithmLib.ML.Ptx.BodyProof
meta import AlgorithmLib.ML.Ptx.BodyProof
public import AlgorithmLib.ML.Ptx.LaneSettle
meta import AlgorithmLib.ML.Ptx.LaneSettle
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `ML.Ptx.BodyWarp` — a warp statement's accesses, and a lane's run of it

A statement's accesses: which elements each lane may touch (`Acc`), that
every access is one they allow and a stored element its lane's alone
(`AccOk`), the elements a lane stores (`StoresL`), and that its indexes fit
32 bits reading the words `rd` gives (`fits`). The warp machine starts from
zero registers and the buffers' words (`ws0`), its integer memory the words
`rd` reads (`imOf`).

Whatever answers its collectives get, a lane's run (`BodyLane.laneI`) of a
statement those allow ends, keeps memory's shape and what `rd` reads, and
leaves every element it does not store (`laneI_shape`); the conditions
`LaneWarp.lane_warp` asks of the warp's run, `OkW`, they give (`okW_of`), the
integer state aside (`run_withInt`). A thread of the statement's kernel ends
as its lane's run (`thread_ends`, from `BodyLane.wpE_bodyW`), and one that
finishes ends so (`thread_body`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec AlgorithmLib.Kernel AlgorithmLib.ML AlgorithmLib.ML.BodyProof

namespace AlgorithmLib.ML.BodyWarp

/-- Which elements of which buffers each lane may touch. -/
abbrev Acc := Lane → Buf → Nat → Prop

def lane0 : Lane := ⟨0, by decide⟩

/-- The words `rd` reads, as the warp machine's integer memory: zero where it
    reads none. -/
def imOf (rd : Buf → Nat → Option Nat) (b : Buf) (j : Nat) : Nat := (rd b j).getD 0

theorem imOf_none : imOf (fun _ _ => none) = fun _ _ => 0 := rfl

section
variable (cta : Nat) (A : Acc) (rd : Buf → Nat → Option Nat)

/-- **Every access a statement makes is one `A` allows**, at trip `i` of its
    loop, indexes reading the words `rd` reads, and an element a lane stores
    is touched by that lane alone. -/
def AccOk : Nat → EWStmt → Prop
  | i, .loadIdx _ b ix => ∀ l : Lane, A l b (ix.eval cta i l (fun _ _ => 0) (imOf rd))
  | i, .loadV4 _ _ _ _ b ix => ∀ l : Lane, ∀ q, q < 4 → A l b (ix.eval cta i l (fun _ _ => 0) (imOf rd) + q)
  | i, .storeLane b ix _ => ∀ l : Lane, A l b (ix.eval cta i l (fun _ _ => 0) (imOf rd)) ∧
      ∀ l', A l' b (ix.eval cta i l (fun _ _ => 0) (imOf rd)) → l' = l
  | i, .storeLane0 b ix _ => A lane0 b (ix.eval cta i lane0 (fun _ _ => 0) (imOf rd)) ∧
      ∀ l', A l' b (ix.eval cta i lane0 (fun _ _ => 0) (imOf rd)) → l' = lane0
  | i, .seq a b => AccOk i a ∧ AccOk i b
  | _, .forN n body => ∀ j, j < n → AccOk j body
  | _, .forM b a body => ∀ j, j < imOf rd b a → AccOk j body
  | i, .iif _ x y => AccOk i x ∧ AccOk i y
  | i, .iwhile _ _ _ body => AccOk i body
  | i, .locals _ body => AccOk i body
  | _, _ => True

end

/-- The elements lane `l` of block `cta` may store, at trip `i` of its loop,
    its indexes reading `im`. -/
def StoresL (cta : Nat) (im : Buf → Nat → Nat) (l : Lane) : Nat → EWStmt → Buf → Nat → Prop
  | i, .storeLane b ix _, b', j => b' = b ∧ j = ix.eval cta i l (fun _ _ => 0) im
  | i, .storeLane0 b ix _, b', j => l = lane0 ∧ b' = b ∧ j = ix.eval cta i lane0 (fun _ _ => 0) im
  | i, .seq x y, b', j => StoresL cta im l i x b' j ∨ StoresL cta im l i y b' j
  | _, .forN n body, b', j => ∃ t, t < n ∧ StoresL cta im l t body b' j
  | _, .forM bb a body, b', j => ∃ t, t < im bb a ∧ StoresL cta im l t body b' j
  | i, .iif _ x y, b', j => StoresL cta im l i x b' j ∨ StoresL cta im l i y b' j
  | i, .iwhile _ _ _ body, b', j => StoresL cta im l i body b' j
  | i, .locals _ body, b', j => StoresL cta im l i body b' j
  | _, _, _, _ => False

/-- What a lane may store it may touch. -/
theorem stores_acc (cta : Nat) (A : Acc) (rd : Buf → Nat → Option Nat) (l : Lane) :
    ∀ (s : EWStmt) (i : Nat) (b : Buf) (j : Nat), AccOk cta A rd i s → StoresL cta (imOf rd) l i s b j → A l b j := by
  intro s
  induction s with
  | storeLane b0 ix n => intro i b j hacc ⟨hb, hj⟩; subst hb hj; exact (hacc l).1
  | storeLane0 b0 ix n => intro i b j hacc ⟨hl, hb, hj⟩; subst hl hb hj; exact hacc.1
  | seq x y ihx ihy =>
    intro i b j hacc h
    rcases h with h | h
    · exact ihx i b j hacc.1 h
    · exact ihy i b j hacc.2 h
  | forN n body ih => intro i b j hacc ⟨t, ht, h⟩; exact ih t b j (hacc t ht) h
  | forM bb a body ih => intro i b j hacc ⟨t, ht, h⟩; exact ih t b j (hacc t ht) h
  | iif _ x y ihx ihy =>
    intro i b j hacc h
    rcases h with h | h
    · exact ihx i b j hacc.1 h
    · exact ihy i b j hacc.2 h
  | iwhile _ _ _ body ih => intro i b j hacc h; exact ih i b j hacc h
  | locals _ body ih => intro i b j hacc h; exact ih i b j hacc h
  | _ => intro i b j _ h; exact absurd h id

theorem idxL_eval (rd : Buf → Nat → Option Nat) (cta i : Nat) (l : Lane) :
    ∀ (ix : IdxE) (j : Nat), idxL rd cta i l.val ix = some j → ix.eval cta i l (fun _ _ => 0) (imOf rd) = j := by
  intro ix
  induction ix with
  | laneId => intro j h; simp only [idxL] at h; split at h <;> cases h; rfl
  | loopI => intro j h; simp only [idxL] at h; split at h <;> cases h; rfl
  | ctaId => intro j h; simp only [idxL] at h; split at h <;> cases h; rfl
  | lit n => intro j h; simp only [idxL] at h; split at h <;> cases h; rfl
  | add a b iha ihb =>
    intro j h
    simp only [idxL, bind, Option.bind] at h
    split at h; · cases h
    rename_i x hx
    dsimp only at h
    split at h; · cases h
    rename_i y hy
    dsimp only at h
    split at h <;> cases h
    show a.eval cta i l (fun _ _ => 0) (imOf rd) + b.eval cta i l (fun _ _ => 0) (imOf rd) = x + y
    rw [iha x hx, ihb y hy]
  | mul a b iha ihb =>
    intro j h
    simp only [idxL, bind, Option.bind] at h
    split at h; · cases h
    rename_i x hx
    dsimp only at h
    split at h; · cases h
    rename_i y hy
    dsimp only at h
    split at h <;> cases h
    show a.eval cta i l (fun _ _ => 0) (imOf rd) * b.eval cta i l (fun _ _ => 0) (imOf rd) = x * y
    rw [iha x hx, ihb y hy]
  | ireg _ => intro j h; simp [idxL] at h
  | ldIdx b o ih =>
    intro j h
    simp only [idxL, bind, Option.bind] at h
    split at h; · cases h
    rename_i x hx
    show imOf rd b (o.eval cta i l (fun _ _ => 0) (imOf rd)) = j
    rw [ih x hx]; simp [imOf, h]

theorem bufW_add (nbuf b j q : Nat) (k : Int) (hk : k = 4 * (q : Int)) (h : j + q < 2 ^ 32) :
    bufW nbuf b j + c64 k = bufW nbuf b (j + q) := by
  subst hk
  have h64 : (1 <<< 64 : Nat) = 18446744073709551616 := rfl
  have he : (4 * (q : Int)).emod ((1 <<< 64 : Nat) : Int) = 4 * (q : Int) := by
    rw [h64]; exact Int.emod_eq_of_lt (by omega) (by omega)
  unfold bufW
  rw [off_eq, off_eq, toNat_ofNat32 (by omega), toNat_ofNat32 h, UInt64.add_assoc]
  congr 1
  simp only [c64]
  rw [he]
  rw [← UInt64.ofNat_add]
  congr 1
  omega

/-! ## A block of a statement's kernel -/

section
attribute [local irreducible] wp

/-- The launch arguments of a statement's `nbuf` buffers: buffer `k` at its
    base. -/
def bufArgs (nbuf : Nat) : List V := (List.range nbuf).map fun k => .sc .i64 (bufB nbuf k)

def params (nbuf : Nat) : List ClifTy := idParams ++ List.replicate nbuf .i64

theorem args_get (cta bd nbuf t : Nat) :
    ∀ k, k < nbuf → (List.toArray (threadArgs cta bd (bufArgs nbuf) t))[4 + k]? = some (.sc .i64 (bufB nbuf k)) := by
  intro k hk
  simp only [threadArgs, bufArgs, List.cons_append, List.nil_append, List.getElem?_toArray]
  rw [show 4 + k = k + 1 + 1 + 1 + 1 by omega]
  simp [List.getElem?_map, hk]

theorem memSame_store {m m' : Mem} {a : UInt64} {n : Nat} {v : UInt64} (h : m.store a n v = some m') :
    MemSame m m' := by
  obtain ⟨_, _, _, hF⟩ := store_frame h
  exact hF.same

/-- The warp machine's state a launch starts from: registers zero, memory
    the words of the buffers. -/
def ws0 (nbuf : Nat) (m : Mem) : WSt F32 :=
  { regs := fun _ _ => ⟨0⟩, mem := fun b j => ⟨(m.load (bufW nbuf b j) 4).getD 0⟩, smem := fun _ => ⟨0⟩ }

/-- Memory with shared memory zeroed, as a block's threads start. -/
abbrev rm (smem0 : ByteArray) (m : Mem) : Mem := { m with arena := smem0 }

theorem rdOk_arena {nbuf : Nat} {rd : Buf → Nat → Option Nat} {m : Mem} (A0 : ByteArray) :
    RdOk nbuf ({ m with arena := A0 } : Mem) rd ↔ RdOk nbuf m rd := by
  constructor
  · intro h b j n hr
    obtain ⟨hv, hm⟩ := h b j n hr
    refine ⟨hv, ?_⟩
    simp only [rdM] at hm ⊢
    rw [load_buf_arena m A0 (decode_bufW hv.1 hv.2.1 hv.2.2)] at hm; exact hm
  · intro h b j n hr
    obtain ⟨hv, hm⟩ := h b j n hr
    refine ⟨hv, ?_⟩
    simp only [rdM] at hm ⊢
    rw [load_buf_arena m A0 (decode_bufW hv.1 hv.2.1 hv.2.2)]; exact hm

theorem memSame_rm {a b : Mem} (h : MemSame a b) (A0 : ByteArray) : MemSame (rm A0 a) (rm A0 b) := by
  refine ⟨fun r => ?_, h.live, h.busy⟩
  cases r with
  | arena => rfl
  | data => exact h.size .data
  | out => exact h.size .out
  | pinned => exact h.size .pinned
  | buf i => exact h.size (.buf i)

/-- **A statement's indexes fit 32 bits** at block `cta`, trip `i`, for every
    lane that computes them, the words they and its loop counts read `rd`'s;
    its loops and shuffle masks 31. -/
def fits (rd : Buf → Nat → Option Nat) (cta : Nat) : Nat → EWStmt → Bool
  | _, .skip => true
  | _, .setR _ _ => true
  | _, .shflXor _ _ m => decide (m < 2 ^ 31)
  | i, .loadIdx _ _ ix => (List.range 32).all fun l => (idxL rd cta i l ix).isSome
  | i, .loadV4 _ _ _ _ _ ix => (List.range 32).all fun l => (idxL rd cta i l ix).isSome
  | i, .storeLane _ ix _ => (List.range 32).all fun l => (idxL rd cta i l ix).isSome
  | i, .storeLane0 _ ix _ => (idxL rd cta i 0 ix).isSome
  | i, .seq a b => fits rd cta i a && fits rd cta i b
  | _, .forN n body => decide (n < 2 ^ 31) && (List.range n).all fun j => fits rd cta j body
  | _, .forM b a body => decide (a < 2 ^ 31) && decide (Body.setsI body = []) && match rd b a with
      | some n => decide (n < 2 ^ 31) && (List.range n).all fun j => fits rd cta j body
      | none => false
  | _, _ => false

/-- A buffer's word inside the buffer may be loaded and stored. -/
theorem bufW_ok {m : Mem} {nbuf b j : Nat} (hb : b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32)
    (hs : 4 * j + 4 ≤ (m.bufs b).size) (v : UInt64) :
    (m.load (bufW nbuf b j) 4).isSome ∧ (m.store (bufW nbuf b j) 4 v).isSome := by
  have hd := decode_bufW hb.1 hb.2.1 hb.2.2
  constructor
  · simp only [Mem.load, hd, Mem.region, Mem.readable, bind, Option.bind]
    rw [if_neg (by simp; omega)]; rfl
  · simp only [Mem.store, hd, Mem.region, Mem.reachable, bind, Option.bind]
    rw [if_neg (by simp; omega)]; rfl

end

/-! ## A lane's run, whatever its answers -/

section Shape
open BodyLane LaneWarp LaneSettle

/-- The words `rd` reads. -/
def rdI (rd : Buf → Nat → Option Nat) (b : Buf) (j : Nat) : Prop := rd b j ≠ none

/-- Lane `l` of block `cta` over `nbuf` buffers. -/
structure CtxOk (nbuf cta : Nat) (l : Lane) (x : LCtx) : Prop where
  lv : x.lv = UInt64.ofNat l.val
  cv : x.cv = UInt64.ofNat cta
  ptr : ∀ b, x.ptr b = bufB nbuf b

theorem CtxOk.with_i {nbuf cta : Nat} {l : Lane} {x : LCtx} (h : CtxOk nbuf cta l x) (j : Nat) :
    CtxOk nbuf cta l { x with i := j } := ⟨h.lv, h.cv, h.ptr⟩

/-- Where `rd` reads, memory holds the word. -/
theorem rdOk_load {nbuf : Nat} {m : Mem} {rd : Buf → Nat → Option Nat} (hrd : RdOk nbuf m rd) {b : Buf} {j : Nat}
    (h : rdI rd b j) : m.load (bufW nbuf b j) 4 = some (UInt64.ofNat (imOf rd b j)) := by
  obtain ⟨n, hn⟩ := Option.ne_none_iff_exists'.mp h
  obtain ⟨_, hm⟩ := hrd b j n hn
  simp only [rdM, Option.map_eq_some_iff] at hm
  obtain ⟨v, hv, rfl⟩ := hm
  rw [hv]; simp [imOf, hn]

/-- An index `idxL` evaluates fits 32 bits as `IdxOk` asks. -/
theorem idxOk_of {nbuf cta i : Nat} {l : Lane} {m : Mem} {rd : Buf → Nat → Option Nat} (hrd : RdOk nbuf m rd) :
    ∀ (ix : IdxE) (j : Nat), idxL rd cta i l.val ix = some j → IdxOk cta (imOf rd) (rdI rd) i l ix := by
  intro ix
  induction ix with
  | laneId => intro j h; trivial
  | loopI => intro j h; simp only [idxL] at h; split at h <;> cases h; assumption
  | ctaId => intro j h; simp only [idxL] at h; split at h <;> cases h; assumption
  | lit n => intro j h; simp only [idxL] at h; split at h <;> cases h; show n < 2 ^ 32; omega
  | ireg _ => intro j h; simp [idxL] at h
  | ldIdx b o ih =>
      intro j h
      simp only [idxL, bind, Option.bind] at h
      split at h; · cases h
      rename_i x hx
      have he := idxL_eval rd cta i l o x hx
      refine ⟨ih x hx, ?_, ?_⟩
      · rw [he]; simp [rdI, h]
      · have h' : rd b x = some j := h
        rw [he]; simp only [imOf, h', Option.getD_some]
        obtain ⟨_, hm⟩ := hrd b x j h'
        simp only [rdM, Option.map_eq_some_iff] at hm
        obtain ⟨v, hv, rfl⟩ := hm
        exact load4_lt hv
  | add a b iha ihb =>
      intro j h
      have he := idxL_eval rd cta i l (.add a b) j h
      simp only [idxL, bind, Option.bind] at h
      split at h; · cases h
      rename_i x hx
      dsimp only at h
      split at h; · cases h
      rename_i y hy
      dsimp only at h
      split at h <;> cases h
      refine ⟨iha x hx, ihb y hy, ?_⟩
      rw [idxL_eval rd cta i l a x hx, idxL_eval rd cta i l b y hy]; assumption
  | mul a b iha ihb =>
      intro j h
      simp only [idxL, bind, Option.bind] at h
      split at h; · cases h
      rename_i x hx
      dsimp only at h
      split at h; · cases h
      rename_i y hy
      dsimp only at h
      split at h <;> cases h
      refine ⟨iha x hx, ihb y hy, ?_⟩
      rw [idxL_eval rd cta i l a x hx, idxL_eval rd cta i l b y hy]; assumption

/-- **A lane's index is the one `idxL` evaluates.** -/
theorem idxV_of {nbuf cta : Nat} {l : Lane} {x : LCtx} (hx : CtxOk nbuf cta l x) {m : Mem}
    {rd : Buf → Nat → Option Nat} (hrd : RdOk nbuf m rd) (ix : IdxE) (j : Nat)
    (h : idxL rd cta x.i l.val ix = some j) : idxV x m ix = some (UInt32.ofNat j) := by
  have := (idx_agree (MI := rdI rd) hx.lv hx.cv rfl (fun b j hb => by rw [hx.ptr]; exact rdOk_load hrd hb) ix
    (idxOk_of hrd ix j h)).1
  rwa [idxL_eval rd cta x.i l ix j h] at this

theorem bufB_off (nbuf : Nat) (b : Buf) (j : Nat) : bufB nbuf b + off (UInt32.ofNat j) = bufW nbuf b j := rfl

/-- What a lane's run leaves: memory of the same shape, holding what `rd`
    reads, the NaN rule, and every word the lane does not store. -/
structure Shape (nbuf cta : Nat) (rd : Buf → Nat → Option Nat) (l : Lane) (i : Nat) (s : EWStmt) (li li' : LI) :
    Prop where
  same : MemSame li.w.mem li'.w.mem
  rdok : RdOk nbuf li'.w.mem rd
  nan : li'.w.mem.nan = li.w.mem.nan
  keep : ∀ b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) → ¬ StoresL cta (imOf rd) l i s b j →
    li'.w.mem.load (bufW nbuf b j) 4 = li.w.mem.load (bufW nbuf b j) 4

theorem Shape.of_mem {nbuf cta : Nat} {rd : Buf → Nat → Option Nat} {l : Lane} {i : Nat} {s : EWStmt} {li li' : LI}
    (h : li'.w.mem = li.w.mem) (hr : RdOk nbuf li.w.mem rd) : Shape nbuf cta rd l i s li li' :=
  ⟨by rw [h]; exact MemSame.refl _, by rw [h]; exact hr, by rw [h], fun _ _ _ _ => by rw [h]⟩

theorem rdOk_store {nbuf : Nat} {rd : Buf → Nat → Option Nat} {m m' : Mem} {b : Buf} {j : Nat} {v : UInt64}
    (hn : rd b j = none) (hv : b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) (hs : m.store (bufW nbuf b j) 4 v = some m')
    (hrd : RdOk nbuf m rd) : RdOk nbuf m' rd := by
  intro b' j' n hr
  obtain ⟨hv', hm⟩ := hrd b' j' n hr
  refine ⟨hv', ?_⟩
  have hne : ¬(b' = b ∧ j' = j) := by
    rintro ⟨rfl, rfl⟩; rw [hn] at hr; cases hr
  simp only [rdM] at hm ⊢
  rw [apart_elems hv hv' hne hs]
  exact hm

/-- A store of one word the lane may store. -/
theorem shape_store {nbuf cta : Nat} {rd : Buf → Nat → Option Nat} {l : Lane} {i : Nat} {s : EWStmt}
    {li : LI} {b : Buf} {j : Nat} {v : UInt64} {m' : Mem}
    (hn : rd b j = none) (hv : b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32)
    (hst : ∀ b' j', StoresL cta (imOf rd) l i s b' j' ↔ b' = b ∧ j' = j)
    (hs : li.w.mem.store (bufW nbuf b j) 4 v = some m') (hrd : RdOk nbuf li.w.mem rd) (li' : LI)
    (hm : li'.w.mem = m') : Shape nbuf cta rd l i s li li' := by
  subst hm
  exact ⟨memSame_store hs, rdOk_store hn hv hs hrd, Mem.store_nan hs,
    fun b' j' hv' hns => apart_elems hv hv' (fun e => hns ((hst b' j').mpr e)) hs⟩

/-- **A lane's run of a statement's fragment ends, whatever its answers**,
    where its indexes fit and every word it touches is one it may and lies
    inside its buffer, and leaves memory as `Shape` says. -/
theorem laneI_shape (nbuf cta : Nat) (A : Acc) (rd : Buf → Nat → Option Nat) (l : Lane)
    (hA : ∀ b j, A l b j → b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) (hRd : ∀ b j, A l b j → rd b j = none) :
    ∀ (s : EWStmt) (x : LCtx) (li : LI), CtxOk nbuf cta l x → fits rd cta x.i s = true → AccOk cta A rd x.i s →
      x.start ≤ li.w.obs.length → RdOk nbuf li.w.mem rd → (∀ b j, A l b j → 4 * j + 4 ≤ (li.w.mem.bufs b).size) →
      ∃ li', laneI x s li = some li' ∧ Shape nbuf cta rd l x.i s li li' := by
  have keepIn : ∀ (m m' : Mem), MemSame m m' → (∀ b j, A l b j → 4 * j + 4 ≤ (m.bufs b).size) →
      ∀ b j, A l b j → 4 * j + 4 ≤ (m'.bufs b).size := fun m m' hs h b j hb => by
    have := hs.size (.buf ⟨b, (hA b j hb).2.1⟩)
    simp only [Mem.region] at this
    rw [← this]; exact h b j hb
  have ld : ∀ (m : Mem) b j, A l b j → (∀ b j, A l b j → 4 * j + 4 ≤ (m.bufs b).size) →
      ∃ v, m.load (bufW nbuf b j) 4 = some v := fun m b j hb h =>
    Option.isSome_iff_exists.mp (bufW_ok (hA b j hb) (h b j hb) 0).1
  have sto : ∀ (m : Mem) b j v, A l b j → (∀ b j, A l b j → 4 * j + 4 ≤ (m.bufs b).size) →
      ∃ m', m.store (bufW nbuf b j) 4 v = some m' := fun m b j v hb h =>
    Option.isSome_iff_exists.mp (bufW_ok (hA b j hb) (h b j hb) v).2
  have ix : ∀ (x : LCtx) (m : Mem) (e : IdxE), CtxOk nbuf cta l x → RdOk nbuf m rd →
      ((List.range 32).all fun l => (idxL rd cta x.i l e).isSome) = true →
      idxV x m e = some (UInt32.ofNat (e.eval cta x.i l (fun _ _ => 0) (imOf rd))) := by
    intro x m e hx hrd h
    obtain ⟨j, hj⟩ := Option.isSome_iff_exists.mp (List.all_eq_true.mp h l.val (List.mem_range.mpr l.isLt))
    rw [idxV_of hx hrd e j hj, idxL_eval rd cta x.i l e j hj]
  have loop : ∀ (body : EWStmt) (x : LCtx) (n : Nat), CtxOk nbuf cta l x →
      (∀ (x : LCtx) (li : LI), CtxOk nbuf cta l x → fits rd cta x.i body = true → AccOk cta A rd x.i body →
        x.start ≤ li.w.obs.length → RdOk nbuf li.w.mem rd → (∀ b j, A l b j → 4 * j + 4 ≤ (li.w.mem.bufs b).size) →
        ∃ li', laneI x body li = some li' ∧ Shape nbuf cta rd l x.i body li li') →
      (∀ j, j < n → fits rd cta j body = true) → (∀ j, j < n → AccOk cta A rd j body) →
      ∀ (L : List Nat) (li : LI), (∀ j ∈ L, j < n) → x.start ≤ li.w.obs.length → RdOk nbuf li.w.mem rd →
        (∀ b j, A l b j → 4 * j + 4 ≤ (li.w.mem.bufs b).size) →
        ∃ li', L.foldlM (fun s j => laneI { x with i := j } body s) li = some li' ∧
          MemSame li.w.mem li'.w.mem ∧ RdOk nbuf li'.w.mem rd ∧ li'.w.mem.nan = li.w.mem.nan ∧
          ∀ b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) → (∀ t ∈ L, ¬ StoresL cta (imOf rd) l t body b j) →
            li'.w.mem.load (bufW nbuf b j) 4 = li.w.mem.load (bufW nbuf b j) 4 := by
    intro body x n hx ih hf hacc L
    induction L with
    | nil => intro li _ _ hr _; exact ⟨li, rfl, MemSame.refl _, hr, rfl, fun _ _ _ _ => rfl⟩
    | cons j js ihj =>
        intro li hL hs hr hin
        have hj := hL j List.mem_cons_self
        obtain ⟨l1, e1, sh1⟩ := ih { x with i := j } li (hx.with_i j) (hf j hj) (hacc j hj) hs hr hin
        obtain ⟨l2, e2, s2, r2, n2, k2⟩ := ihj l1 (fun t ht => hL t (List.mem_cons_of_mem _ ht))
          (Nat.le_trans hs (by obtain ⟨L', hL'⟩ := laneI_grows body _ li l1 e1; rw [hL']; simp))
          sh1.rdok (keepIn _ _ sh1.same hin)
        refine ⟨l2, by simp only [List.foldlM_cons, e1, Option.bind_some]; exact e2, sh1.same.trans s2, r2,
          n2.trans sh1.nan, fun b' j' hv hns => ?_⟩
        rw [k2 b' j' hv (fun t ht => hns t (List.mem_cons_of_mem _ ht)),
          sh1.keep b' j' hv (hns j List.mem_cons_self)]
  intro s
  induction s with
  | skip => intro x li _ _ _ _ hr _; exact ⟨li, rfl, Shape.of_mem rfl hr⟩
  | seq a b iha ihb =>
      intro x li hx hf hacc hs hr hin
      simp only [fits, Bool.and_eq_true] at hf
      obtain ⟨l1, e1, sh1⟩ := iha x li hx hf.1 hacc.1 hs hr hin
      obtain ⟨l2, e2, sh2⟩ := ihb x l1 hx hf.2 hacc.2
        (Nat.le_trans hs (by obtain ⟨L', hL'⟩ := laneI_grows a _ li l1 e1; rw [hL']; simp)) sh1.rdok
        (keepIn _ _ sh1.same hin)
      refine ⟨l2, by simp only [laneI, e1, Option.bind_some]; exact e2,
        ⟨sh1.same.trans sh2.same, sh2.rdok, sh2.nan.trans sh1.nan, fun b' j' hv hns => ?_⟩⟩
      simp only [StoresL, not_or] at hns
      rw [sh2.keep b' j' hv hns.2, sh1.keep b' j' hv hns.1]
  | setR d e =>
      intro x li _ _ _ hs hr _
      exact ⟨_, rfl, Shape.of_mem (exLog_obs _ li.f e li.w x.start hs).2.2.2 hr⟩
  | shflXor d sr m =>
      intro x li _ hf _ hs hr _
      simp only [fits, decide_eq_true_eq] at hf
      have hc : m < 2 ^ 31 ∧ x.start ≤ li.w.obs.length := ⟨hf, hs⟩
      exact ⟨LI.mk li.reg (callW li.w .shflBfly [.sc .i32 (li.f sr).bits, .sc .i32 (s32 (UInt32.ofNat m))])
          (li.setF d (F32.mk (ansW x.ans (collCalls li.w x.start).length))).f,
        by simp only [laneI, if_pos hc], Shape.of_mem rfl hr⟩
  | loadIdx d b e =>
      intro x li hx hf hacc hs hr hin
      obtain ⟨v, hv⟩ := ld li.w.mem b _ (hacc l) hin
      refine ⟨li.setF d ⟨v⟩, ?_, Shape.of_mem rfl hr⟩
      simp only [laneI, ix x li.w.mem e hx hr hf, Option.bind_eq_bind, Option.bind_some, hx.ptr b, bufB_off, hv]
  | loadV4 d0 d1 d2 d3 b e =>
      intro x li hx hf hacc hs hr hin
      have hq : ∀ (q : Nat) (k : Int), k = 4 * (q : Int) → q < 4 →
          ∃ v, li.w.mem.load (bufW nbuf b (e.eval cta x.i l (fun _ _ => 0) (imOf rd)) + c64 k) 4 = some v := by
        intro q k hk hq
        have hA' := hacc l q hq
        rw [bufW_add nbuf b _ q k hk (hA b _ hA').2.2]
        exact ld li.w.mem b _ hA' hin
      obtain ⟨v0, h0⟩ := ld li.w.mem b _ (by simpa using hacc l 0 (by decide)) hin
      obtain ⟨v1, h1⟩ := hq 1 4 (by omega) (by decide)
      obtain ⟨v2, h2⟩ := hq 2 8 (by omega) (by decide)
      obtain ⟨v3, h3⟩ := hq 3 12 (by omega) (by decide)
      refine ⟨(((li.setF d0 ⟨v0⟩).setF d1 ⟨v1⟩).setF d2 ⟨v2⟩).setF d3 ⟨v3⟩, ?_, Shape.of_mem rfl hr⟩
      simp only [laneI, ix x li.w.mem e hx hr hf, Option.bind_eq_bind, Option.bind_some, hx.ptr b, bufB_off,
        h0, h1, h2, h3]
  | storeLane b e n =>
      intro x li hx hf hacc hs hr hin
      have hAe := (hacc l).1
      obtain ⟨m', hm'⟩ := sto li.w.mem b _ (li.f n).bits hAe hin
      let li' : LI := { li with w := { obsStore li.w (bufW nbuf b (e.eval cta x.i l (fun _ _ => 0) (imOf rd))) 4
        (li.f n).bits with mem := m' } }
      refine ⟨li', ?_, shape_store (hRd b _ hAe) (hA b _ hAe) (fun b' j' => by simp [StoresL]) hm' hr li' rfl⟩
      simp only [li', laneI, ix x li.w.mem e hx hr hf, Option.bind_eq_bind, Option.bind_some, hx.ptr b, bufB_off, stw,
        hm']
  | storeLane0 b e n =>
      intro x li hx hf hacc hs hr hin
      by_cases hl : l = lane0
      · subst hl
        have hAe := hacc.1
        obtain ⟨m', hm'⟩ := sto li.w.mem b _ (li.f n).bits hAe hin
        obtain ⟨j0, hj0⟩ := Option.isSome_iff_exists.mp (show (idxL rd cta x.i lane0.val e).isSome = true from hf)
        have hie := idxL_eval rd cta x.i lane0 e j0 hj0
        let li' : LI := { li with w := { obsStore li.w (bufW nbuf b (e.eval cta x.i lane0 (fun _ _ => 0) (imOf rd))) 4
          (li.f n).bits with mem := m' } }
        refine ⟨li', ?_, shape_store (hRd b _ hAe) (hA b _ hAe) (fun b' j' => by simp [StoresL]) hm' hr li' rfl⟩
        have hlv : x.lv = 0 := by rw [hx.lv]; rfl
        rw [← hie] at hj0
        simp only [li', laneI, hlv, if_true, idxV_of hx hr e _ hj0, Option.bind_eq_bind, Option.bind_some, hx.ptr b,
          bufB_off, stw, hm']
      · have hlv : x.lv ≠ 0 := by
          rw [hx.lv]; intro h; apply hl; apply Fin.ext
          have := congrArg UInt64.toNat h
          simp only [UInt64.toNat_ofNat', UInt64.toNat_zero] at this
          have hl32 : l.val < 32 := l.isLt
          rw [Nat.mod_eq_of_lt (by omega)] at this; exact this
        refine ⟨li, by simp only [laneI, if_neg hlv], Shape.of_mem rfl hr⟩
  | forN n body ih =>
      intro x li hx hf hacc hs hr hin
      simp only [fits, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range] at hf
      obtain ⟨li', e, s1, r1, n1, k1⟩ := loop body x n hx (fun x li hx => ih x li hx) (fun j hj => hf.2 j hj) hacc
        _ li (fun j hj => List.mem_range.mp hj) hs hr hin
      refine ⟨li', by simp only [laneI, if_pos (show n < 2 ^ 31 from hf.1)]; exact e, ⟨s1, r1, n1, fun b j hv hns => ?_⟩⟩
      exact k1 b j hv (fun t ht hst => hns ⟨t, List.mem_range.mp ht, hst⟩)
  | forM b a body ih =>
      intro x li hx hf hacc hs hr hin
      simp only [fits, Bool.and_eq_true, decide_eq_true_eq] at hf
      obtain ⟨⟨ha, _⟩, hf⟩ := hf
      split at hf
      · rename_i n hn
        simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range] at hf
        have him : imOf rd b a = n := by simp [imOf, hn]
        obtain ⟨li', e, s1, r1, n1, k1⟩ := loop body x n hx (fun x li hx => ih x li hx) (fun j hj => hf.2 j hj)
          (fun j hj => hacc j (by rw [him]; exact hj)) _ li (fun j hj => List.mem_range.mp hj) hs hr hin
        have hld := rdOk_load hr (show rdI rd b a by simp [rdI, hn])
        rw [him] at hld
        refine ⟨li', ?_, ⟨s1, r1, n1, fun b' j hv hns => ?_⟩⟩
        · simp only [laneI, if_pos ha, hx.ptr b, Option.bind_eq_bind]
          rw [show bufB nbuf b + off (UInt32.ofNat a) = bufW nbuf b a from rfl, hld]
          simp only [Option.bind_some, UInt64.toNat_ofNat']
          rw [Nat.mod_eq_of_lt (by omega), if_pos hf.1]; exact e
        · exact k1 b' j hv (fun t ht hst => hns ⟨t, by rw [him]; exact List.mem_range.mp ht, hst⟩)
      · cases hf
  | _ => intro x li _ hf; simp [fits] at hf

end Shape

/-! ## The warp machine without integer state, and what a lane needs of it -/

section Bridge
open BodyLane LaneWarp LaneSettle

theorem eval_withInt (ws : WSt F32) (I : ISt) (l : Lane) (e : WFExp) :
    @WFExp.eval F32 (F32.ops hw) _ (ws.withInt I) l e = @WFExp.eval F32 (F32.ops hw) _ ws l e := by
  induction e <;> simp only [WFExp.eval, *] <;> rfl

theorem storeLanes_withInt (ws : WSt F32) (I : ISt) (b : Buf) (ix : Lane → Nat) (r : Nat) :
    (ws.withInt I).storeLanes b ix r = (ws.storeLanes b ix r).withInt I := by
  unfold WSt.storeLanes
  have key : ∀ (L : List Lane) (s : WSt F32), L.foldl (fun s l => s.store1 b (ix l) (ws.regs r l)) (s.withInt I) =
      (L.foldl (fun s l => s.store1 b (ix l) (ws.regs r l)) s).withInt I := by
    intro L
    induction L with
    | nil => intro s; rfl
    | cons l L ih => intro s; simp only [List.foldl_cons]; exact ih (s.store1 b (ix l) (ws.regs r l))
  exact key (List.finRange W) ws

/-- **A statement's fragment runs alike whatever the integer state.** -/
theorem run_withInt (hw : Hw) (cta : Nat) (rd : Buf → Nat → Option Nat) (I : ISt) :
    ∀ (s : EWStmt) (i : Nat) (ws : WSt F32), fits rd cta i s = true →
      run hw cta (imOf rd) s (ws.withInt I) i = (run hw cta (imOf rd) s ws i).withInt I := by
  have loop : ∀ (body : EWStmt) (L : List Nat), (∀ j ∈ L, ∀ ws, run hw cta (imOf rd) body (ws.withInt I) j =
      (run hw cta (imOf rd) body ws j).withInt I) → ∀ ws,
      L.foldl (fun s' j => @EWStmt.run F32 (F32.ops hw) _ body s' cta j (fun _ _ => 0) (imOf rd)) (ws.withInt I) =
        (L.foldl (fun s' j => @EWStmt.run F32 (F32.ops hw) _ body s' cta j (fun _ _ => 0) (imOf rd)) ws).withInt I := by
    intro body L
    induction L with
    | nil => intro _ _; rfl
    | cons j L ih =>
        intro h ws
        simp only [List.foldl_cons]
        rw [show @EWStmt.run F32 (F32.ops hw) _ body (ws.withInt I) cta j (fun _ _ => 0) (imOf rd) =
          (run hw cta (imOf rd) body ws j).withInt I from h j List.mem_cons_self ws]
        exact ih (fun j' hj' => h j' (List.mem_cons_of_mem _ hj')) _
  intro s
  induction s with
  | skip => intro _ _ _; rfl
  | seq a b iha ihb =>
      intro i ws hf
      simp only [fits, Bool.and_eq_true] at hf
      show run hw cta (imOf rd) b (run hw cta (imOf rd) a (ws.withInt I) i) i = _
      rw [iha i ws hf.1, ihb i _ hf.2]; rfl
  | setR d e =>
      intro i ws _
      show (ws.withInt I).setReg d _ = _
      simp only [eval_withInt]; rfl
  | shflXor | loadIdx | loadV4 | storeLane0 => intro _ _ _; rfl
  | storeLane b ix r => intro i ws _; exact storeLanes_withInt ws I b _ r
  | forN n body ih =>
      intro i ws hf
      simp only [fits, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range] at hf
      exact loop body _ (fun j hj ws => ih j ws (hf.2 j (List.mem_range.mp hj))) ws
  | forM bu a body ih =>
      intro i ws hf
      simp only [fits, Bool.and_eq_true, decide_eq_true_eq] at hf
      obtain ⟨_, hf⟩ := hf
      split at hf
      · rename_i n hn
        simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range] at hf
        have him : imOf rd bu a = n := by simp [imOf, hn]
        exact loop body _ (fun j hj ws => ih j ws (hf.2 j (by rw [← him]; exact List.mem_range.mp hj))) ws
      · cases hf
  | _ => intro _ _ hf; simp [fits] at hf

/-- The warp machine's start, its buffers at their bases. -/
abbrev ws0P (nbuf : Nat) (m : Mem) : WSt F32 := (ws0 nbuf m).withInt { ptr := bufB nbuf }

theorem w32_ws0 (nbuf : Nat) (m : Mem) : W32 (ws0P nbuf m) := by
  refine ⟨fun _ _ => by show (0 : UInt64).toNat < 2 ^ 32; decide, fun b j => ?_, fun _ => by
    show (0 : UInt64).toNat < 2 ^ 32; decide⟩
  show ((m.load (bufW nbuf b j) 4).getD 0).toNat < 2 ^ 32
  cases h : m.load (bufW nbuf b j) 4 with
  | none => decide
  | some v => exact load4_lt h

theorem bufW_decode {nbuf b j : Nat} (hv : b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) :
    decodeAddr (bufB nbuf b + off (UInt32.ofNat j)) = some (.buf ⟨b, hv.2.1⟩, 4 * j) :=
  decode_bufW hv.1 hv.2.1 hv.2.2

theorem apart_bufW {b j b' j' : Nat} (hb : b < 268435449) (hb' : b' < 268435449) (hne : ¬(b' = b ∧ j' = j)) :
    Apart (.buf ⟨b, hb⟩) (4 * j) 4 (.buf ⟨b', hb'⟩) (4 * j') 4 := by
  by_cases e : b' = b
  · subst e; right
    have : j' ≠ j := fun h => hne ⟨rfl, h⟩
    omega
  · left; intro h; cases h; exact e rfl

theorem notArena_bufW {nbuf b j : Nat} (hv : b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) :
    NotArena (bufB nbuf b + off (UInt32.ofNat j)) := by
  intro o h; rw [bufW_decode hv] at h; cases h

/-- **A store of an element only its lane touches may be made.** -/
theorem storeOk_elem (nbuf : Nat) (A : Acc) (rd : Buf → Nat → Option Nat) (WS : Lane → UInt64 → Nat → Prop)
    (hA : ∀ l b j, A l b j → b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) {m : Mem} (hrd : RdOk nbuf m rd)
    (l : Lane) (b : Buf) (J : Nat) (hAJ : A l b J) (hown : ∀ l', A l' b J → l' = l) (hn : rd b J = none)
    (hWS : WS l (bufB nbuf b + off (UInt32.ofNat J)) 4) :
    StoreOk (bufB nbuf) A (fun _ _ => False) (rdI rd) WS l (bufB nbuf b + off (UInt32.ofNat J)) 4
      (fun b' j => b' = b ∧ j = J) (fun _ => False) := by
  have hv := hA l b J hAJ
  refine ⟨hWS, .buf ⟨b, hv.2.1⟩, 4 * J, bufW_decode hv, ?_⟩
  constructor
  · intro h; exact Region.noConfusion h
  constructor
  · intro l' b' j' hA'
    by_cases e : b' = b ∧ j' = J
    · obtain ⟨rfl, rfl⟩ := e; exact Or.inl ⟨hown l' hA', rfl, rfl⟩
    · have hv' := hA l' b' j' hA'
      exact Or.inr ⟨_, _, bufW_decode hv', apart_bufW hv.2.1 hv'.2.1 e⟩
  constructor
  · intro _ _ h; exact False.elim h
  · intro b' j' hmi
    obtain ⟨n, hn'⟩ := Option.ne_none_iff_exists'.mp hmi
    have hv' := (hrd b' j' n hn').1
    refine ⟨_, _, bufW_decode hv', apart_bufW hv.2.1 hv'.2.1 ?_⟩
    rintro ⟨rfl, rfl⟩; rw [hn] at hn'; cases hn'

/-- **What a statement's fragment asks of a lane's run, `OkW`, its accesses
    give**: indexes fit, every access one `A` allows, a stored element only
    its lane touches and none `rd` reads. -/
theorem okW_of (hw : Hw) (nbuf cta S : Nat) (A : Acc) (rd : Buf → Nat → Option Nat) (WS : Lane → UInt64 → Nat → Prop)
    (hA : ∀ l b j, A l b j → b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) (hRd : ∀ l b j, A l b j → rd b j = none)
    (hWS : ∀ l b j, A l b j → WS l (bufB nbuf b + off (UInt32.ofNat j)) 4) {m : Mem} (hrd : RdOk nbuf m rd) :
    ∀ (s : EWStmt) (i : Nat), fits rd cta i s = true → AccOk cta A rd i s →
      ∀ ws cl, OkW hw cta (imOf rd) S (bufB nbuf) A (fun _ _ => False) (rdI rd) WS i s ws cl := by
  have ixOk : ∀ i (e : IdxE) (l : Lane), ((List.range 32).all fun l => (idxL rd cta i l e).isSome) = true →
      IdxOk cta (imOf rd) (rdI rd) i l e := fun i e l h => by
    obtain ⟨j, hj⟩ := Option.isSome_iff_exists.mp (List.all_eq_true.mp h l.val (List.mem_range.mpr l.isLt))
    exact idxOk_of hrd e j hj
  intro s
  induction s with
  | skip | setR => intro _ _ _ _ _; trivial
  | shflXor d sr m => intro i hf _ _ _; simpa [fits] using hf
  | loadIdx d b e => intro i hf hacc _ _ l; exact ⟨ixOk i e l hf, hacc l⟩
  | loadV4 d0 d1 d2 d3 b e =>
      intro i hf hacc _ _ l
      exact ⟨ixOk i e l hf, (hA l b _ (hacc l 3 (by decide))).2.2, hacc l⟩
  | storeLane b e n =>
      intro i hf hacc _ _ l
      refine ⟨ixOk i e l hf, storeOk_elem nbuf A rd WS hA hrd l b _ (hacc l).1 (fun l' h => (hacc l).2 l' h)
        (hRd l b _ (hacc l).1) (hWS l b _ (hacc l).1), fun l' he _ => ?_⟩
      have := (hacc l').1
      rw [he] at this
      exact (hacc l).2 l' this
  | storeLane0 b e n =>
      intro i hf hacc _ _
      obtain ⟨j, hj⟩ := Option.isSome_iff_exists.mp (show (idxL rd cta i lane0.val e).isSome = true from hf)
      exact ⟨idxOk_of hrd e j hj, storeOk_elem nbuf A rd WS hA hrd lane0 b _ hacc.1 hacc.2 (hRd _ b _ hacc.1)
        (hWS _ b _ hacc.1)⟩
  | seq a b iha ihb =>
      intro i hf hacc ws cl
      simp only [fits, Bool.and_eq_true] at hf
      exact ⟨iha i hf.1 hacc.1 ws cl, ihb i hf.2 hacc.2 _ _⟩
  | forN n body ih =>
      intro i hf hacc ws cl
      simp only [fits, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range] at hf
      exact ⟨hf.1, fun j hj => ih j (hf.2 j hj) (hacc j hj) _ _⟩
  | forM bu a body ih =>
      intro i hf hacc ws cl
      simp only [fits, Bool.and_eq_true, decide_eq_true_eq] at hf
      obtain ⟨⟨ha, _⟩, hf⟩ := hf
      split at hf
      · rename_i n hn
        simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range] at hf
        have him : imOf rd bu a = n := by simp [imOf, hn]
        exact ⟨ha, by simp [rdI, hn], by rw [him]; exact hf.1,
          fun j hj => ih j (hf.2 j (by rw [← him]; exact hj)) (hacc j hj) _ _⟩
      · cases hf
  | _ => intro _ hf; simp [fits] at hf

end Bridge

/-! ## A thread runs as its lane -/

section Thread
open BodyLane

/-- The buffers' bases. -/
def bufPs (nbuf : Nat) : List UInt64 := (List.range nbuf).map (bufB nbuf)

theorem bufArgs_ps (nbuf : Nat) : bufArgs nbuf = (bufPs nbuf).map (.sc .i64) := by
  simp only [bufArgs, bufPs, List.map_map]; rfl

theorem bufPs_get (nbuf b : Nat) : (bufPs nbuf).getD b 0 = bufB nbuf b := by
  by_cases h : b < nbuf
  · simp [bufPs, List.getD_eq_getElem?_getD, h]
  · rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none (by simp [bufPs]; omega), Option.getD_none]
    simp [bufB, h]

/-- What thread `t` of a block of `W` warps runs as: lane `t mod 32` of block
    `cta·W + t/32`. -/
abbrev laneCtx (nbuf cta W t : Nat) (ans : List Answer) (start : Nat) (hw : Hw) : LCtx :=
  threadCtx (cta * W + t / 32) (t % 32) (bufPs nbuf) ans start hw

theorem ofNat32_64 {n : Nat} (h : n < 2 ^ 32) : (UInt32.ofNat n).toUInt64 = UInt64.ofNat n := by
  apply UInt64.toNat_inj.mp
  rw [UInt32.toNat_toUInt64, toNat_ofNat32 h, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega)]

theorem laneCtx_ok (nbuf cta W t : Nat) (ans : List Answer) (start : Nat) (hw : Hw) (hv : cta * W + t / 32 < 2 ^ 32) :
    CtxOk nbuf (cta * W + t / 32) ⟨t % 32, Nat.mod_lt _ (by decide)⟩ (laneCtx nbuf cta W t ans start hw) :=
  ⟨ofNat32_64 (by omega), ofNat32_64 hv, fun b => bufPs_get nbuf b⟩

/-- **A statement's thread ends as its lane's run**: thread `t` of block `cta`
    of the statement's kernel with `W` warps a block, its collectives answered
    from `ans` (words), ends with the world its lane's run ends with. -/
theorem thread_ends (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (name : String) (nbuf smem : Nat)
    (s : EWStmt) (W : Nat) (hW : 0 < W) (hW31 : W < 2 ^ 31) (cta t bd : Nat) (htW : t < 32 * W) (ht : t < 2 ^ 32)
    (hv : cta * W + t / 32 < 2 ^ 32) (w : World) (ans : List Answer) (hWo : Words ans) (li' : LI)
    (hfine : Fine (emitGo (fnBody (params nbuf) (Body.body nbuf s W)) ⟨(params nbuf).length, 0, [], [], [], none⟩).2)
    (hrun : laneI (laneCtx nbuf cta W t ans w.obs.length hw) s (LI.init w) = some li') :
    ∃ steps u, thread (Body.kernel name nbuf smem s W).code ans (threadArgs cta bd (bufArgs nbuf) t) w hw steps
        = .ok u ∧ u = li'.w := by
  have hb := wpE_bodyW (cfg := { env := [], locals := threadLocals ans w.obs.length hw }) nbuf s W hW hW31 cta t bd
    htW ht hv (bufPs nbuf) (by simp [bufPs]) w ans hw (laneOk_thread _ _ _ ans _ hw hWo hw32) li' hrun
  rw [← bufArgs_ps] at hb
  have hE := emit_ends (cfg := { env := [], locals := threadLocals ans w.obs.length hw })
    (P0 := At (List.toArray (threadArgs cta bd (bufArgs nbuf) t)) w) hb hfine
    (fun Γ' w' h => ⟨by rw [h.1]; simp [threadArgs, bufArgs, idParams], h⟩)
  obtain ⟨k, Γ', u, hk, hu⟩ := hE _ _ ⟨rfl, rfl⟩
  refine ⟨k, u, ?_, hu⟩
  unfold thread
  rw [show (Body.kernel name nbuf smem s W).code = emit (Body.body nbuf s W) (idParams ++ List.replicate nbuf .i64)
    from rfl, hk]

/-- **A statement's thread that finishes ends as its lane's run.** -/
theorem thread_body (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (name : String) (nbuf smem : Nat)
    (s : EWStmt) (W : Nat) (hW : 0 < W) (hW31 : W < 2 ^ 31) (cta t bd : Nat) (htW : t < 32 * W) (ht : t < 2 ^ 32)
    (hv : cta * W + t / 32 < 2 ^ 32) (w u : World) (ans : List Answer) (hWo : Words ans) (li' : LI)
    (hfine : Fine (emitGo (fnBody (params nbuf) (Body.body nbuf s W)) ⟨(params nbuf).length, 0, [], [], [], none⟩).2)
    (hrun : laneI (laneCtx nbuf cta W t ans w.obs.length hw) s (LI.init w) = some li')
    {steps : Nat} (hr : thread (Body.kernel name nbuf smem s W).code ans (threadArgs cta bd (bufArgs nbuf) t) w hw steps
      = .ok u) : u = li'.w := by
  obtain ⟨k, u', hk, hu⟩ := thread_ends hw hw32 name nbuf smem s W hW hW31 cta t bd htW ht hv w ans hWo li' hfine hrun
  rw [thread_det hr hk]; exact hu

end Thread

end AlgorithmLib.ML.BodyWarp
