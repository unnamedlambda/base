module
public import AlgorithmLib.ML.Ptx.BodyProof
meta import AlgorithmLib.ML.Ptx.BodyProof
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `ML.Ptx.BodyWarp` — a warp's lanes run with the warp

`lockstep`: where each of a warp's 32 lanes runs a warp statement on its own
(`runL`), every shuffle answered the word its partner passed at the same
shuffle (`Settled`), and every access is one `AccOk` allows — an element a
lane stores is touched by that lane alone — each lane ends as the warp
machine's run of the statement at `F32` says: its registers the warp's
column, its memory the warp's at every element it may touch (`Rel`). All
lanes make the same number of shuffles (`nsh`, `runL_calls`), which is what
lets one lane's `k`-th shuffle meet its partner's.

`block_lockstep` is the kernel's claim: a block of `nw` warps of a statement's
kernel (`ML.Body.kernel`) that finishes leaves every element a lane of warp
`w` may touch as the warp machine's run of the statement for block
`cta·nw + w` does, from zero registers and the buffers' words (`ws0`), its
integer memory the words `rd` reads (`imOf`), its shuffles answered by the
laws and its approximations the hardware's (`hw`), and leaves alone what no
lane stores. `launch_lockstep` composes a grid of such blocks, no block
storing what another touches (`StoresL`); `launch_lockstep_one` is it for
one-warp blocks. A thread ends as its lane's run (`thread_ends`, from
`wpE_stmt`), at some fuel and so at every larger one (`thread_det`), so a
thread that finishes runs its lane (`thread_body`);
whether a lane's run finishes depends on memory's shape alone
(`runL_transfer`), and a lane leaves alone what other lanes may touch
(`runL_frame_mem`) and what it does not store (`runL_frame_st`), so the
block's pass composes.
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
  | _ => intro i b j _ h; exact absurd h id

section
variable (hw : Hw) (nbuf : Nat) (A : Acc)

/-- **A lane agrees with the warp**: its registers are the warp's column, its
    memory the warp's at every element it may touch. -/
def Rel (ws : WSt F32) (st : LSt) (l : Lane) : Prop :=
  (∀ n, st.reg n = ws.regs n l) ∧ ∀ b j, A l b j → st.mem.load (bufW nbuf b j) 4 = some (ws.mem b j).bits

end

theorem evalL_eval {R : Type} [NumOps R] [NumLit R] (ws : WSt R) (l : Lane) (reg : Nat → R)
    (h : ∀ n, reg n = ws.regs n l) (e : WFExp) : evalL reg e = e.eval ws l := by
  induction e <;> simp [evalL, WFExp.eval, *]

/-- How many shuffles a statement makes, whatever lane runs it, its loop
    counts read from `im`. -/
def nsh (im : Buf → Nat → Nat) : EWStmt → Nat
  | .seq a b => nsh im a + nsh im b
  | .shflXor _ _ _ => 1
  | .forN n body => n * nsh im body
  | .forM b a body => im b a * nsh im body
  | _ => 0

theorem foldlM_calls (f : LSt → Nat → Option LSt) (c : Nat)
    (hf : ∀ st j st', f st j = some st' → st.calls <+: st'.calls ∧ st'.calls.length = st.calls.length + c) :
    ∀ (l : List Nat) (st st' : LSt), l.foldlM f st = some st' →
      st.calls <+: st'.calls ∧ st'.calls.length = st.calls.length + l.length * c
  | [], st, st', h => by simp only [List.foldlM_nil] at h; cases h; exact ⟨List.prefix_refl _, by simp⟩
  | j :: js, st, st', h => by
    simp only [List.foldlM_cons, bind, Option.bind] at h
    split at h
    · cases h
    rename_i s1 h1
    obtain ⟨p1, l1⟩ := hf st j s1 h1
    obtain ⟨p2, l2⟩ := foldlM_calls f c hf js s1 st' h
    exact ⟨p1.trans p2, by rw [l2, l1, List.length_cons, Nat.succ_mul]; omega⟩

/-- **A lane's run only adds collectives, as many as the statement makes.** -/
theorem runL_calls (hw : Hw) (nbuf cta l : Nat) (o : Nat → UInt32) (rd : Buf → Nat → Option Nat) :
    ∀ (s : EWStmt) (i : Nat) (st st' : LSt), runL hw nbuf cta l o rd i s st = some st' →
      st.calls <+: st'.calls ∧ st'.calls.length = st.calls.length + nsh (imOf rd) s := by
  intro s
  induction s with
  | seq a b iha ihb =>
    intro i st st' h
    simp only [runL, bind, Option.bind] at h
    split at h
    · cases h
    rename_i sa hsa
    obtain ⟨p1, l1⟩ := iha i st sa hsa
    obtain ⟨p2, l2⟩ := ihb i sa st' h
    exact ⟨p1.trans p2, by rw [l2, l1, nsh]; omega⟩
  | shflXor d s m =>
    intro i st st' h
    simp only [runL] at h; split at h
    · cases h; exact ⟨List.prefix_append _ _, by simp [nsh]⟩
    · cases h
  | forN n body ih =>
    intro i st st' h
    simp only [runL] at h; split at h
    · obtain ⟨p, hl⟩ := foldlM_calls _ (nsh (imOf rd) body) (fun st j st' h => ih j st st' h) _ _ _ h
      exact ⟨p, by rw [hl, nsh, List.length_range]⟩
    · cases h
  | forM b a body ih =>
    intro i st st' h
    obtain ⟨_, n, hn, _, hf⟩ := runL_forM h
    obtain ⟨p, hl⟩ := foldlM_calls _ (nsh (imOf rd) body) (fun st j st' h => ih j st st' h) _ _ _ hf
    exact ⟨p, by rw [hl, nsh, List.length_range, imOf, hn]; rfl⟩
  | skip => intro i st st' h; cases h; exact ⟨List.prefix_refl _, rfl⟩
  | setR n e => intro i st st' h; cases h; exact ⟨List.prefix_refl _, rfl⟩
  | loadIdx d b ix =>
    intro i st st' h
    simp only [runL, bind, Option.bind] at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    cases h; exact ⟨List.prefix_refl _, rfl⟩
  | storeLane b ix n =>
    intro i st st' h
    obtain ⟨_, _, _, _, _, _, _, rfl⟩ := runL_store h
    exact ⟨List.prefix_refl _, rfl⟩
  | storeLane0 b ix n =>
    intro i st st' h
    rcases runL_store0 h with ⟨_, _, _, _, _, _, _, _, rfl⟩ | ⟨_, rfl⟩ <;> exact ⟨List.prefix_refl _, rfl⟩
  | loadV4 d0 d1 d2 d3 b ix =>
    intro i st st' h
    simp only [runL, bind, Option.bind] at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    cases h; exact ⟨List.prefix_refl _, rfl⟩
  | _ => intro i st st' h; simp [runL] at h


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

theorem bits32_eq {x : UInt64} (h : x.toNat < 2 ^ 32) : s32 x.toUInt32.toUInt64.toUInt32 = x := by
  apply UInt64.toNat_inj.mp
  simp [s32, Nat.mod_eq_of_lt h]

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

theorem prefix_get {α : Type} {l1 l2 : List α} (h : l1 <+: l2) {k : Nat} {x : α} (hk : l1[k]? = some x) :
    l2[k]? = some x := by
  obtain ⟨t, rfl⟩ := h
  have hlt : k < l1.length := by
    rcases Nat.lt_or_ge k l1.length with h | h
    · exact h
    · rw [List.getElem?_eq_none h] at hk; cases hk
  rw [List.getElem?_append_left hlt]; exact hk

/-- The word a shuffle moves. -/
def wordOf : V → UInt32
  | .sc _ x => x.toUInt32
  | _ => 0

/-- **Each lane's shuffles answered its partner's word**: lane `l`'s `k`-th
    shuffle, by mask `m`, is answered the word lane `l xor m` passed at its
    `k`-th, as the lanes' final collectives `F` record them. -/
def Settled (o : Lane → Nat → UInt32) (F : Lane → List (Intr × List V)) : Prop :=
  ∀ (l : Lane) (k : Nat) (a : V) (ty : ClifTy) (m : UInt64), (F l)[k]? = some (.shflBfly, [a, .sc ty m]) →
    ∃ args, (F (xorLane l m.toNat))[k]? = some (.shflBfly, args) ∧ o l k = wordOf (lane32 args)

/-- The step a loop of `lockstep` makes: trips `0 … n-1` of `body`, each run
    with the warp, end with it. -/
theorem lockstep_loop (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (nbuf cta : Nat) (A : Acc)
    (rd : Buf → Nat → Option Nat) (o : Lane → Nat → UInt32) (F : Lane → List (Intr × List V)) (body : EWStmt)
    (ih : ∀ (i : Nat) (ws : WSt F32) (sts sts' : Lane → LSt),
      AccOk cta A rd i body →
      (∀ l : Lane, runL hw nbuf cta l.val (o l) rd i body (sts l) = some (sts' l)) →
      (∀ l, Rel nbuf A ws (sts l) l) → (∀ l, Regs32 (sts l).reg) →
      (∀ l l', (sts l).calls.length = (sts l').calls.length) →
      (∀ l, (sts' l).calls <+: F l) →
      ∀ l, Rel nbuf A (@WStmt.run F32 (F32.ops hw) _ (body.elabAt cta i (fun _ _ => 0) (imOf rd)) ws) (sts' l) l)
    (n : Nat) (ws : WSt F32) (sts sts' : Lane → LSt) (hacc : ∀ j, j < n → AccOk cta A rd j body)
    (hfold : ∀ l : Lane, (List.range n).foldlM (fun st j => runL hw nbuf cta l.val (o l) rd j body st) (sts l) =
      some (sts' l))
    (hrel : ∀ l, Rel nbuf A ws (sts l) l) (h32 : ∀ l, Regs32 (sts l).reg)
    (hlen : ∀ l l', (sts l).calls.length = (sts l').calls.length) (hpre : ∀ l, (sts' l).calls <+: F l) :
    ∀ l, Rel nbuf A ((List.range n).foldl
      (fun s' j => @WStmt.run F32 (F32.ops hw) _ (body.elabAt cta j (fun _ _ => 0) (imOf rd)) s') ws) (sts' l) l := by
    let tr (l : Lane) (j : Nat) : LSt := trips (fun st j => runL hw nbuf cta l.val (o l) rd j body st) (sts l) j
    let wsj (j : Nat) : WSt F32 := (List.range j).foldl (fun s' j => @WStmt.run F32 (F32.ops hw) _ (body.elabAt cta j (fun _ _ => 0) (imOf rd)) s') ws
    have hsucc : ∀ l : Lane, ∀ j, j < n → runL hw nbuf cta l.val (o l) rd j body (tr l j) = some (tr l (j + 1)) :=
      fun l j hj => trips_succ (hfold l) j hj
    have hpref : ∀ l : Lane, ∀ j, j ≤ n → (tr l j).calls <+: (sts' l).calls := by
      intro l j hj
      have key : ∀ d, j + d ≤ n → (tr l j).calls <+: (tr l (j + d)).calls := by
        intro d
        induction d with
        | zero => intro _; exact List.prefix_refl _
        | succ d ihd =>
          intro hd
          exact (ihd (by omega)).trans (runL_calls hw nbuf cta _ _ _ body _ _ _ (hsucc l (j + d) (by omega))).1
      have := key (n - j) (by omega)
      have hend : tr l n = sts' l := trips_end (hfold l)
      rw [show j + (n - j) = n by omega, hend] at this
      exact this
    have inv : ∀ j, j ≤ n → (∀ l, Rel nbuf A (wsj j) (tr l j) l) ∧ (∀ l, Regs32 (tr l j).reg) ∧
        (∀ l, (tr l j).calls.length = (sts l).calls.length + j * nsh (imOf rd) body) := by
      intro j hj
      induction j with
      | zero => exact ⟨hrel, h32, fun l => by simp [tr, trips]⟩
      | succ j ihj =>
        obtain ⟨hr1, h321, hl1⟩ := ihj (by omega)
        refine ⟨?_, fun l => runL_32 hw hw32 nbuf cta _ _ _ body j _ _ (hsucc l j (by omega)) (h321 l), fun l => ?_⟩
        · have e : wsj (j + 1) = @WStmt.run F32 (F32.ops hw) _ (body.elabAt cta j (fun _ _ => 0) (imOf rd)) (wsj j) := by
            simp only [wsj, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
          rw [e]
          exact ih j (wsj j) (fun l => tr l j) (fun l => tr l (j + 1)) (hacc j (by omega))
            (fun l => hsucc l j (by omega)) hr1 h321
            (fun l l' => by rw [hl1 l, hl1 l', hlen l l'])
            (fun l => (hpref l (j + 1) (by omega)).trans (hpre l))
        · rw [(runL_calls hw nbuf cta _ _ _ body j _ _ (hsucc l j (by omega))).2, hl1 l, Nat.succ_mul]; omega
    intro l
    have := (inv n (Nat.le_refl n)).1 l
    have hend : tr l n = sts' l := trips_end (hfold l)
    rw [hend] at this
    exact this

/-- **Lanes that agree with the warp run with it**: where every lane's run of
    a statement finishes, its shuffles answered as `Settled` says and every
    access one `A` allows, each lane ends agreeing with the warp's run. -/
theorem lockstep (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (nbuf cta : Nat) (A : Acc)
    (rd : Buf → Nat → Option Nat) (o : Lane → Nat → UInt32) (F : Lane → List (Intr × List V)) (hS : Settled o F)
    (hA : ∀ l b j, A l b j → b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) :
    ∀ (s : EWStmt) (i : Nat) (ws : WSt F32) (sts sts' : Lane → LSt),
      AccOk cta A rd i s →
      (∀ l : Lane, runL hw nbuf cta l.val (o l) rd i s (sts l) = some (sts' l)) →
      (∀ l, Rel nbuf A ws (sts l) l) → (∀ l, Regs32 (sts l).reg) →
      (∀ l l', (sts l).calls.length = (sts l').calls.length) →
      (∀ l, (sts' l).calls <+: F l) →
      ∀ l, Rel nbuf A (@WStmt.run F32 (F32.ops hw) _ (s.elabAt cta i (fun _ _ => 0) (imOf rd)) ws) (sts' l) l := by
  intro s
  induction s with
  | skip =>
    intro i ws sts sts' _ hr hrel _ _ _ l
    have e : sts' l = sts l := (Option.some.inj (hr l)).symm
    rw [e]; exact hrel l
  | seq a b iha ihb =>
    intro i ws sts sts' hacc hr hrel h32 hlen hpre
    have hmid : ∀ l : Lane, ∃ m, runL hw nbuf cta l.val (o l) rd i a (sts l) = some m ∧
        runL hw nbuf cta l.val (o l) rd i b m = some (sts' l) := by
      intro l
      have h := hr l
      simp only [runL, bind, Option.bind] at h
      split at h
      · cases h
      · rename_i m hm; exact ⟨m, hm, h⟩
    let mid : Lane → LSt := fun l => (runL hw nbuf cta l.val (o l) rd i a (sts l)).getD (sts l)
    have hma : ∀ l : Lane, runL hw nbuf cta l.val (o l) rd i a (sts l) = some (mid l) := by
      intro l; obtain ⟨m, h1, _⟩ := hmid l; simp only [mid, h1, Option.getD_some]
    have hmb : ∀ l : Lane, runL hw nbuf cta l.val (o l) rd i b (mid l) = some (sts' l) := by
      intro l; obtain ⟨m, h1, h2⟩ := hmid l; simp only [mid, h1, Option.getD_some]; exact h2
    have hrel1 := iha i ws sts mid hacc.1 hma hrel h32 hlen
      (fun l => (runL_calls hw nbuf cta _ _ _ b i _ _ (hmb l)).1.trans (hpre l))
    exact ihb i _ mid sts' hacc.2 hmb hrel1
      (fun l => runL_32 hw hw32 nbuf cta _ _ _ a i _ _ (hma l) (h32 l))
      (fun l l' => by
        rw [(runL_calls hw nbuf cta _ _ _ a i _ _ (hma l)).2, (runL_calls hw nbuf cta _ _ _ a i _ _ (hma l')).2,
          hlen l l'])
      hpre
  | setR n e =>
    intro i ws sts sts' _ hr hrel _ _ _ l
    have he := (Option.some.inj (hr l)).symm
    rw [he]
    refine ⟨fun k => ?_, (hrel l).2⟩
    show upd (sts l).reg n _ k = (WSt.setReg ws n _).regs k l
    simp only [upd, WSt.setReg]
    split
    · rw [@evalL_eval F32 (F32.ops hw) _ ws l _ (hrel l).1 e]
    · exact (hrel l).1 k
  | shflXor d sr m =>
    intro i ws sts sts' _ hr hrel h32 hlen hpre
    have hform : ∀ l : Lane, m < 2 ^ 31 ∧ sts' l = { sts l with
        reg := upd (sts l).reg d ⟨s32 (o l (sts l).calls.length)⟩,
        calls := (sts l).calls ++ [(.shflBfly, [.sc .i32 ((sts l).reg sr).bits, .sc .i32 (s32 (UInt32.ofNat m))])] } := by
      intro l
      have h := hr l
      simp only [runL] at h
      split at h
      · exact ⟨by assumption, (Option.some.inj h).symm⟩
      · cases h
    intro l
    obtain ⟨hm, he⟩ := hform l
    have hmt : (s32 (UInt32.ofNat m)).toNat = m := by
      simp only [s32, UInt32.toNat_toUInt64]; exact toNat_ofNat32 (by omega)
    have hFl : (F l)[(sts l).calls.length]? =
        some (.shflBfly, [.sc .i32 ((sts l).reg sr).bits, .sc .i32 (s32 (UInt32.ofNat m))]) :=
      prefix_get (hpre l) (by rw [he]; simp)
    obtain ⟨args, hFp, ho⟩ := hS l _ _ _ _ hFl
    rw [hmt] at hFp
    obtain ⟨_, hep⟩ := hform (xorLane l m)
    have hFp' : (F (xorLane l m))[(sts l).calls.length]? =
        some (.shflBfly, [.sc .i32 ((sts (xorLane l m)).reg sr).bits, .sc .i32 (s32 (UInt32.ofNat m))]) :=
      prefix_get (hpre (xorLane l m)) (by rw [hep, hlen l (xorLane l m)]; simp)
    rw [hFp'] at hFp
    cases hFp
    rw [he]
    refine ⟨fun k => ?_, fun b j hA => (hrel l).2 b j hA⟩
    show upd (sts l).reg d _ k = (WSt.setReg ws d _).regs k l
    simp only [upd, WSt.setReg]
    split
    · show _ = ws.regs sr (xorLane l m)
      rw [ho, ← (hrel (xorLane l m)).1 sr]
      simp only [wordOf, lane32]
      congr 1
      exact bits32_eq (h32 (xorLane l m) sr)
    · exact (hrel l).1 k
  | loadIdx d b ix =>
    intro i ws sts sts' hacc hr hrel _ _ _ l
    have h := hr l
    simp only [runL, bind, Option.bind] at h
    split at h; · cases h
    rename_i j hj
    dsimp only at h
    split at h; · cases h
    rename_i v hv
    have he := (Option.some.inj h).symm
    rw [he]
    have hjx := idxL_eval rd cta i l ix j hj
    have hl := (hrel l).2 b j (hjx ▸ hacc l)
    rw [hv] at hl
    cases hl
    refine ⟨fun k => ?_, (hrel l).2⟩
    show upd (sts l).reg d _ k = (WSt.setReg ws d _).regs k l
    simp only [upd, WSt.setReg]
    split
    · show _ = ws.mem b (ix.eval cta i l (fun _ _ => 0) (imOf rd)); rw [hjx]
    · exact (hrel l).1 k
  | storeLane b ix n =>
    intro i ws sts sts' hacc hr hrel h32 _ _ l
    obtain ⟨j, m', hj, -, -, -, hm', he⟩ := runL_store (hr l)
    rw [he]
    have hjx := idxL_eval rd cta i l ix j hj
    refine ⟨fun k => ?_, fun b' j' hA' => ?_⟩
    · show (sts l).reg k = ((List.finRange W).foldl _ ws).regs k l
      rw [storeFold_regs]; exact (hrel l).1 k
    · show m'.load _ 4 = some (((List.finRange W).foldl (fun s l' => s.store1 b (ix.eval cta i l' (fun _ _ => 0) (imOf rd)) (ws.regs n l')) ws).mem b' j').bits
      obtain ⟨hb1, hb2, hj1⟩ := hA l b j (hjx ▸ (hacc l).1)
      obtain ⟨hb1', hb2', hj1'⟩ := hA l b' j' hA'
      by_cases hsame : b' = b ∧ j' = j
      · obtain ⟨hbe, hje⟩ := hsame
        rw [hbe, hje]
        rw [load_store4 (h32 l n) hm', storeFold_at b (fun l' => ix.eval cta i l' (fun _ _ => 0) (imOf rd)) _ j (ws.regs n l) _ ws
          (fun l' _ hl' => by
            have h1 := (hacc l).1
            rw [hjx, ← hl'] at h1
            rw [(hacc l').2 l h1]) ⟨l, List.mem_finRange l, hjx⟩]
        rw [(hrel l).1 n]
      · rw [load_store_clear hm' (decode_bufW hb1 hb2 hj1) (decode_bufW hb1' hb2' hj1') ?_, (hrel l).2 b' j' hA',
          storeFold_other b _ _ b' j' _ ws ?_]
        · intro l' _ ⟨hbb, hjj⟩
          subst hbb
          have := (hacc l').2 l (hjj ▸ hA')
          subst this
          exact hsame ⟨rfl, hjj.trans hjx⟩
        · by_cases hbb : b' = b
          · subst hbb
            right
            have hne : j' ≠ j := fun e => hsame ⟨rfl, e⟩
            omega
          · left; intro e; apply hbb; cases e; rfl
  | storeLane0 b ix n =>
    intro i ws sts sts' hacc hr hrel h32 _ _ l
    by_cases hl0 : l = lane0
    · subst hl0
      obtain ⟨_, j, m', hj, -, -, -, hm', he⟩ := (runL_store0 (hr lane0)).resolve_right (fun h => h.1 rfl)
      rw [he]
      have hjx := idxL_eval rd cta i lane0 ix j hj
      refine ⟨fun k => (hrel lane0).1 k, fun b' j' hA' => ?_⟩
      show m'.load _ 4 = some ((ws.store1 b (ix.eval cta i lane0 (fun _ _ => 0) (imOf rd)) (ws.regs n lane0)).mem b' j').bits
      obtain ⟨hb1, hb2, hj1⟩ := hA lane0 b j (hjx ▸ hacc.1)
      obtain ⟨hb1', hb2', hj1'⟩ := hA lane0 b' j' hA'
      by_cases hsame : b' = b ∧ j' = j
      · obtain ⟨hbe, hje⟩ := hsame
        rw [hbe, hje, load_store4 (h32 lane0 n) hm', hjx]
        simp [WSt.store1, (hrel lane0).1 n]
      · rw [load_store_clear hm' (decode_bufW hb1 hb2 hj1) (decode_bufW hb1' hb2' hj1') ?_, (hrel lane0).2 b' j' hA']
        · simp only [WSt.store1, hjx]
          rw [if_neg hsame]
        · by_cases hbb : b' = b
          · subst hbb
            right
            have hne : j' ≠ j := fun e => hsame ⟨rfl, e⟩
            omega
          · left; intro e; apply hbb; cases e; rfl
    · obtain ⟨_, he⟩ := (runL_store0 (hr l)).resolve_left (fun h => hl0 (Fin.ext h.1))
      rw [he]
      refine ⟨fun k => (hrel l).1 k, fun b' j' hA' => ?_⟩
      show _ = some ((ws.store1 b (ix.eval cta i lane0 (fun _ _ => 0) (imOf rd)) (ws.regs n lane0)).mem b' j').bits
      simp only [WSt.store1]
      rw [if_neg (fun ⟨hbb, hjj⟩ => hl0 (hacc.2 l (hbb ▸ hjj ▸ hA')))]
      exact (hrel l).2 b' j' hA'
  | loadV4 d0 d1 d2 d3 b ix =>
    intro i ws sts sts' hacc hr hrel _ _ _ l
    have h := hr l
    simp only [runL, bind, Option.bind] at h
    split at h; · cases h
    rename_i j hj
    dsimp only at h
    split at h; · cases h
    rename_i v0 h0
    dsimp only at h
    split at h; · cases h
    rename_i v1 h1
    dsimp only at h
    split at h; · cases h
    rename_i v2 h2
    dsimp only at h
    split at h; · cases h
    rename_i v3 h3
    have he := (Option.some.inj h).symm
    rw [he]
    have hjx := idxL_eval rd cta i l ix j hj
    have hq : ∀ q, q < 4 → (sts l).mem.load (bufW nbuf b (j + q)) 4 = some (ws.mem b (j + q)).bits :=
      fun q hq => (hrel l).2 b (j + q) (hjx ▸ hacc l q hq)
    have hj3 : j + 3 < 2 ^ 32 := (hA l b (j + 3) (hjx ▸ hacc l 3 (by decide))).2.2
    rw [bufW_add nbuf b j 1 4 rfl (by omega), hq 1 (by decide)] at h1
    rw [bufW_add nbuf b j 2 8 rfl (by omega), hq 2 (by decide)] at h2
    rw [bufW_add nbuf b j 3 12 rfl hj3, hq 3 (by decide)] at h3
    have h0' := hq 0 (by decide)
    rw [Nat.add_zero, h0] at h0'
    cases h0'; cases h1; cases h2; cases h3
    refine ⟨fun k => ?_, (hrel l).2⟩
    simp only [EWStmt.elabAt, wrun_loadV4, WSt.setReg, upd]
    split
    · show F32.mk (ws.mem b (j + 3)).bits = ws.mem b (ix.eval cta i l (fun _ _ => 0) (imOf rd) + 3); rw [hjx]
    · split
      · show F32.mk (ws.mem b (j + 2)).bits = ws.mem b (ix.eval cta i l (fun _ _ => 0) (imOf rd) + 2); rw [hjx]
      · split
        · show F32.mk (ws.mem b (j + 1)).bits = ws.mem b (ix.eval cta i l (fun _ _ => 0) (imOf rd) + 1); rw [hjx]
        · split
          · show F32.mk (ws.mem b j).bits = ws.mem b (ix.eval cta i l (fun _ _ => 0) (imOf rd)); rw [hjx]
          · exact (hrel l).1 k
  | forN n body ih =>
    intro i ws sts sts' hacc hr hrel h32 hlen hpre
    have hn : n < 2 ^ 31 := by
      have h := hr lane0
      simp only [runL] at h
      split at h
      · assumption
      · cases h
    exact lockstep_loop hw hw32 nbuf cta A rd o F body ih n ws sts sts' hacc
      (fun l => by have h := hr l; simp only [runL, if_pos hn] at h; exact h) hrel h32 hlen hpre
  | forM b a body ih =>
    intro i ws sts sts' hacc hr hrel h32 hlen hpre
    obtain ⟨_, n, hn, _, _⟩ := runL_forM (hr lane0)
    have he : imOf rd b a = n := by simp [imOf, hn]
    have hl := lockstep_loop hw hw32 nbuf cta A rd o F body ih n ws sts sts' (fun j hj => hacc j (he ▸ hj))
      (fun l => by
        obtain ⟨_, n', hn', _, hf⟩ := runL_forM (hr l)
        rw [hn] at hn'; cases hn'; exact hf) hrel h32 hlen hpre
    intro l
    show Rel nbuf A ((List.range (imOf rd b a)).foldl _ ws) _ l
    rw [he]; exact hl l
  | _ => intro i ws sts sts' _ hr; have h := hr lane0; simp [runL] at h

/-! ## A block of a statement's kernel -/

section
attribute [local irreducible] wp

/-- The launch arguments of a statement's `nbuf` buffers: buffer `k` at its
    base. -/
def bufArgs (nbuf : Nat) : List V := (List.range nbuf).map fun k => .sc .i64 (bufB nbuf k)

def params (nbuf : Nat) : List ClifTy := idParams ++ List.replicate nbuf .i64


theorem drop4_slots (nbuf : Nat) :
    (Vals.drop 4 (carriesFrom 0 (idParams ++ List.replicate nbuf ClifTy.i64))).uniformToList = List.range' 4 nbuf := by
  simp only [idParams, List.cons_append, List.nil_append, carriesFrom, Vals.drop, Vals.tail]
  exact uniformToList_carries nbuf 4

theorem args_get (cta bd nbuf t : Nat) :
    ∀ k, k < nbuf → (List.toArray (threadArgs cta bd (bufArgs nbuf) t))[4 + k]? = some (.sc .i64 (bufB nbuf k)) := by
  intro k hk
  simp only [threadArgs, bufArgs, List.cons_append, List.nil_append, List.getElem?_toArray]
  rw [show 4 + k = k + 1 + 1 + 1 + 1 by omega]
  simp [List.getElem?_map, hk]

/-- **A statement's code ends as its lane's run**: run as lane `l` of block
    `cta` from slots holding them, the buffers' bases at `4 + k` and shared
    memory's at `3`, its collectives answered from `ans` (words), it ends as the
    lane's run from zero registers and the world's memory says. -/
theorem wpE_run (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (cfg : Cfg) (nbuf : Nat) (s : EWStmt)
    (cta l : Nat) (hl : l < 2 ^ 32) (w : World) (ans : List Answer)
    (hA : AnswersFrom ans w.obs.length cfg) (hX : ApproxFrom hw cfg) (hW : Words ans) (st' : LSt)
    (rd : Buf → Nat → Option Nat) (hrd : RdOk nbuf w.mem rd) (hnan : w.mem.nan = hw.nan)
    (hrun : runL hw nbuf cta l (wordAt ans) rd 0 s ⟨fun _ => ⟨0⟩, w.mem, []⟩ = some st')
    (Γ : Env) (lane ctaS : Slot .i32) (hlane : Γ[lane]? = some (.sc .i32 (s32 (UInt32.ofNat l))))
    (hctaS : Γ[ctaS]? = some (.sc .i32 (s32 (UInt32.ofNat cta))))
    (hb : ∀ k, k < nbuf → Γ[4 + k]? = some (.sc .i64 (bufB nbuf k))) :
    wpE cfg (Body.run s lane ctaS (List.range' 4 nbuf) 3)
      (fun _ _ u => u.mem = st'.mem ∧ collCalls u w.obs.length = st'.calls) Γ w := by
  simp only [Body.run, Body.c32]
  refine wpE_bind (wpE_iconst32u (by decide) ?_)
  refine wpE_bind (wpE_fconst32 ?_)
  refine wpE_bind (wpE_iconst32u (by decide) ?_)
  refine wpE_bind (wpE_iconst64 ?_)
  have hE : Extends Γ ((((Γ.push (V.sc .i32 (s32 (UInt32.ofNat (Int.ofNat 0).toNat)))).push
      (V.sc .f32 (0 &&& widthMask .f32))).push (V.sc .i32 (s32 (UInt32.ofNat (Int.ofNat 0).toNat)))).push
      (V.sc .i64 (c64 0))) := (((Extends.push _ _).push_of _).push_of _).push_of _
  refine wpE_bind (wpE_stmt hw hA hW (fun x w => hX .ex2Approx .f32 x w rfl) hw32 nbuf cta l hl rd s 0 _ [] _ w
    ⟨fun _ => ⟨0⟩, w.mem, []⟩ st' _ hrun ?_ ?_ (fun _ => by simp) hrd hnan rfl (collCalls_self w) (Nat.le_refl _) ?_)
  · refine ⟨hE.get hlane, hE.get hctaS, ?_, fun k => ?_, ?_⟩
    · exact get_push (get_push (get_push get_push_size))
    · simp only [Body.buf]
      by_cases hk : k < nbuf
      · rw [List.getD_eq_getElem?_getD, List.getElem?_range' hk]
        simpa [Nat.one_mul] using hE.get (hb k hk)
      · rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none (by simp; omega)]
        simp only [Option.getD_none]
        rw [show bufB nbuf k = c64 0 by rw [bufB, dif_neg (fun h => hk h.1)]; rfl]
        exact get_push_size
    · have h := get_push (v := V.sc .i64 (c64 0)) (get_push (v := V.sc .i32 (s32 (UInt32.ofNat (Int.ofNat 0).toNat)))
        (get_push_size (Γ := Γ.push (V.sc .i32 (s32 (UInt32.ofNat (Int.ofNat 0).toNat))))
          (v := V.sc .f32 (0 &&& widthMask .f32))))
      rw [show (0 : UInt64) &&& widthMask .f32 = 0 by decide] at h
      exact h
  · intro n
    have h := get_push (v := V.sc .i64 (c64 0)) (get_push (v := V.sc .i32 (s32 (UInt32.ofNat (Int.ofNat 0).toNat)))
      (get_push_size (Γ := Γ.push (V.sc .i32 (s32 (UInt32.ofNat (Int.ofNat 0).toNat))))
        (v := V.sc .f32 (0 &&& widthMask .f32))))
    rw [show (0 : UInt64) &&& widthMask .f32 = 0 by decide] at h
    exact h
  · intro r' Γ' w' _ _ hm hc _
    exact wpE_pure ⟨hm, hc⟩

theorem and31 (t : Nat) (ht : t < 2 ^ 32) :
    UInt32.ofNat t &&& UInt32.ofNat (Int.ofNat 31).toNat = UInt32.ofNat (t % 32) := by
  apply UInt32.toNat_inj.mp
  rw [UInt32.toNat_and, toNat_ofNat32 ht, toNat_ofNat32 (by decide : (Int.ofNat 31).toNat < 2 ^ 32),
    toNat_ofNat32 (by omega : t % 32 < 2 ^ 32), show (Int.ofNat 31).toNat = 2 ^ 5 - 1 from rfl,
    Nat.and_two_pow_sub_one_eq_mod]

/-- **A statement's thread ends as its lane's run**: thread `t` of block `cta`
    of the statement's kernel with `W` warps a block runs as lane `t mod 32` of
    block `cta·W + t/32`, its collectives answered from `ans` (words), and ends
    as that lane's run from zero registers and the thread's memory says. -/
theorem wpE_body (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (cfg : Cfg) (nbuf : Nat) (s : EWStmt)
    (W : Nat) (hW : 0 < W) (hW31 : W < 2 ^ 31) (cta t bd : Nat) (htW : t < 32 * W) (ht : t < 2 ^ 32)
    (hv : cta * W + t / 32 < 2 ^ 32) (w : World) (ans : List Answer)
    (hA : AnswersFrom ans w.obs.length cfg) (hX : ApproxFrom hw cfg) (hWo : Words ans) (st' : LSt)
    (rd : Buf → Nat → Option Nat) (hrd : RdOk nbuf w.mem rd) (hnan : w.mem.nan = hw.nan)
    (hrun : runL hw nbuf (cta * W + t / 32) (t % 32) (wordAt ans) rd 0 s ⟨fun _ => ⟨0⟩, w.mem, []⟩ = some st') :
    wpE cfg (fnBody (params nbuf) (Body.body nbuf s W))
      (fun _ _ u => u.mem = st'.mem ∧ collCalls u w.obs.length = st'.calls)
      (List.toArray (threadArgs cta bd (bufArgs nbuf) t)) w := by
  rw [fnBody_of (by simp [params, idParams, HProg.ptrParams])]
  simp only [Body.body]
  refine wpE_entryParams ?_
  rw [drop4_slots]
  simp only [idParams, List.cons_append, List.nil_append, carriesFrom, Vals.head, Vals.snd, Vals.drop, Vals.tail]
  generalize hΓa : List.toArray (threadArgs cta bd (bufArgs nbuf) t) = Γa
  have h0 : Γa[0]? = some (.sc .i32 (s32 (UInt32.ofNat cta))) := by rw [← hΓa]; simp [threadArgs]
  have h1 : Γa[1]? = some (.sc .i32 (s32 (UInt32.ofNat t))) := by rw [← hΓa]; simp [threadArgs]
  have hb : ∀ k, k < nbuf → Γa[4 + k]? = some (.sc .i64 (bufB nbuf k)) := by
    intro k hk; rw [← hΓa]; exact args_get cta bd nbuf t k hk
  have hl : t % 32 < 2 ^ 32 := by omega
  by_cases hW1 : W ≤ 1
  · rw [if_pos hW1]
    have hW1' : W = 1 := by omega
    subst hW1'
    have ht32 : t < 32 := by omega
    refine wpE_bind (wpE_pure ?_)
    refine wpE_run hw hw32 cfg nbuf s _ _ hl w ans hA hX hWo st' rd hrd hnan hrun Γa 1 0 ?_ ?_ hb
    · rw [Nat.mod_eq_of_lt ht32]; exact h1
    · rw [Nat.div_eq_of_lt ht32, Nat.mul_one, Nat.add_zero]; exact h0
  · rw [if_neg hW1]
    simp only [Body.c32]
    refine wpE_bind (wpE_iconst32u (k := Int.ofNat 31) (by decide) ?_)
    refine wpE_bind (wpE_band32 (x := UInt32.ofNat t) (get_push h1) get_push_size ?_)
    refine wpE_bind (wpE_ushrImm32 (by decide) (get_push (get_push h1)) ?_)
    refine wpE_bind (wpE_iconst32u (k := Int.ofNat W) (by simp only [Int.ofNat_eq_natCast]; constructor <;> omega) ?_)
    refine wpE_bind (wpE_imul32 (x := UInt32.ofNat cta) (get_push (get_push (get_push (get_push (get_push h0)))))
      get_push_size ?_)
    refine wpE_bind (wpE_iadd32 get_push_size (get_push (get_push (get_push_size))) ?_)
    refine wpE_bind (wpE_pure ?_)
    refine wpE_run hw hw32 cfg nbuf s _ _ hl w ans hA hX hWo st' rd hrd hnan hrun _ _ _ ?_ ?_
      (fun k hk => get_push (get_push (get_push (get_push (get_push (get_push (get_push (hb k hk))))))))
    · rw [← and31 t ht]; exact get_push (get_push (get_push (get_push (get_push get_push_size))))
    · have hc : UInt32.ofNat cta * UInt32.ofNat (Int.ofNat W).toNat + UInt32.ofNat t >>> UInt32.ofNat (Int.ofNat 5).toNat
          = UInt32.ofNat (cta * W + t / 32) := by
        have hcW : cta * W < 2 ^ 32 := by omega
        have hc32 : cta < 2 ^ 32 := by
          rcases Nat.eq_zero_or_pos cta with h | h
          · omega
          · exact Nat.lt_of_le_of_lt (Nat.le_mul_of_pos_right cta hW) hcW
        have hWn : (Int.ofNat W).toNat = W := rfl
        have h5 : (Int.ofNat 5).toNat = 5 := rfl
        apply UInt32.toNat_inj.mp
        rw [UInt32.toNat_add, UInt32.toNat_mul, UInt32.toNat_shiftRight, hWn, h5, toNat_ofNat32 hc32,
          toNat_ofNat32 (by omega : W < 2 ^ 32), toNat_ofNat32 ht, toNat_ofNat32 (by decide : 5 < 2 ^ 32),
          toNat_ofNat32 hv, Nat.mod_eq_of_lt hcW, Nat.shiftRight_eq_div_pow]
        exact Nat.mod_eq_of_lt (by omega)
      rw [← hc]; exact get_push_size

theorem memSame_store {m m' : Mem} {a : UInt64} {n : Nat} {v : UInt64} (h : m.store a n v = some m') :
    MemSame m m' := by
  obtain ⟨_, _, _, hF⟩ := store_frame h
  exact hF.same

/-- **L1: a lane's run keeps memory's shape.** -/
theorem runL_same (hw : Hw) (nbuf cta l : Nat) (o : Nat → UInt32) (rd : Buf → Nat → Option Nat) :
    ∀ (s : EWStmt) (i : Nat) (st st' : LSt), runL hw nbuf cta l o rd i s st = some st' → MemSame st.mem st'.mem := by
  intro s
  induction s with
  | skip => intro i st st' h; cases h; exact MemSame.refl _
  | seq a b iha ihb =>
    intro i st st' h
    simp only [runL, bind, Option.bind] at h
    split at h; · cases h
    rename_i sa hsa
    exact (iha i st sa hsa).trans (ihb i sa st' h)
  | setR n e => intro i st st' h; cases h; exact MemSame.refl _
  | shflXor d s m =>
    intro i st st' h; simp only [runL] at h; split at h
    · cases h; exact MemSame.refl _
    · cases h
  | loadIdx d b ix =>
    intro i st st' h
    simp only [runL, bind, Option.bind] at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    cases h; exact MemSame.refl _
  | loadV4 d0 d1 d2 d3 b ix =>
    intro i st st' h
    simp only [runL, bind, Option.bind] at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    cases h; exact MemSame.refl _
  | storeLane b ix n =>
    intro i st st' h
    obtain ⟨_, _, _, _, _, _, hm', rfl⟩ := runL_store h
    exact memSame_store hm'
  | storeLane0 b ix n =>
    intro i st st' h
    rcases runL_store0 h with ⟨_, _, _, _, _, _, _, hm', rfl⟩ | ⟨_, rfl⟩
    · exact memSame_store hm'
    · exact MemSame.refl _
  | forN n body ih =>
    intro i st st' h
    simp only [runL] at h
    split at h
    · exact foldlM_inv (fun st j => runL hw nbuf cta l o rd j body st) (fun st2 => MemSame st.mem st2.mem)
        (fun s1 j s2 h12 h1 => h1.trans (ih j s1 s2 h12)) _ _ _ h (MemSame.refl _)
    · cases h
  | forM b a body ih =>
    intro i st st' h
    obtain ⟨_, n, _, _, hf⟩ := runL_forM h
    exact foldlM_inv (fun st j => runL hw nbuf cta l o rd j body st) (fun st2 => MemSame st.mem st2.mem)
      (fun s1 j s2 h12 h1 => h1.trans (ih j s1 s2 h12)) _ _ _ hf (MemSame.refl _)
  | _ => intro i st st' h; simp [runL] at h


theorem load_transfer {m1 m2 : Mem} {a : UInt64} {n : Nat} {v : UInt64} (hs : MemSame m1 m2)
    (h : m1.load a n = some v) : ∃ v', m2.load a n = some v' := by
  have e := MemSame.load_isSome hs a n
  rw [h] at e
  exact Option.isSome_iff_exists.mp (by rw [← e]; rfl)

theorem store_transfer {m1 m2 m1' : Mem} {a : UInt64} {n : Nat} (v1 v2 : UInt64) (hs : MemSame m1 m2)
    (h : m1.store a n v1 = some m1') : ∃ m2', m2.store a n v2 = some m2' := by
  have e := MemSame.store_isSome hs a n v1 v2
  rw [h] at e
  exact Option.isSome_iff_exists.mp (by rw [← e]; rfl)

theorem transfer_fold {hw : Hw} {nbuf cta l : Nat} {o o' : Nat → UInt32} {rd : Buf → Nat → Option Nat}
    {body : EWStmt}
    (ih : ∀ (i : Nat) (st st1 st2 : LSt), runL hw nbuf cta l o rd i body st = some st1 →
      MemSame st.mem st2.mem → ∃ st2', runL hw nbuf cta l o' rd i body st2 = some st2') :
    ∀ (L : List Nat) (sa sb sa' : LSt), L.foldlM (fun st j => runL hw nbuf cta l o rd j body st) sa = some sa' →
      MemSame sa.mem sb.mem → ∃ sb', L.foldlM (fun st j => runL hw nbuf cta l o' rd j body st) sb = some sb' := by
  intro L
  induction L with
  | nil => intro sa sb sa' _ _; exact ⟨sb, rfl⟩
  | cons j js ihj =>
    intro sa sb sa' h hs'
    simp only [List.foldlM_cons, bind, Option.bind] at h ⊢
    split at h; · cases h
    rename_i s1 hs1
    obtain ⟨t1, ht1⟩ := ih j sa s1 sb hs1 hs'
    rw [ht1]
    exact ihj s1 t1 sa' h ((runL_same hw nbuf cta l o rd body j sa s1 hs1).symm.trans
      (hs'.trans (runL_same hw nbuf cta l o' rd body j sb t1 ht1)))

/-- **L2: whether a lane's run finishes depends on memory's shape alone**, not
    on what memory holds, the registers, or how its shuffles are answered. -/
theorem runL_transfer (hw : Hw) (nbuf cta l : Nat) (o o' : Nat → UInt32) (rd : Buf → Nat → Option Nat) :
    ∀ (s : EWStmt) (i : Nat) (st st1 st2 : LSt), runL hw nbuf cta l o rd i s st = some st1 →
      MemSame st.mem st2.mem → ∃ st2', runL hw nbuf cta l o' rd i s st2 = some st2' := by
  intro s
  induction s with
  | skip => intro i st st1 st2 _ _; exact ⟨st2, rfl⟩
  | seq a b iha ihb =>
    intro i st st1 st2 h hs
    simp only [runL, bind, Option.bind] at h
    split at h; · cases h
    rename_i sa hsa
    obtain ⟨sb, hsb⟩ := iha i st sa st2 hsa hs
    obtain ⟨sc, hsc⟩ := ihb i sa st1 sb h
      ((runL_same hw nbuf cta l o rd a i st sa hsa).symm.trans (hs.trans (runL_same hw nbuf cta l o' rd a i st2 sb hsb)))
    exact ⟨sc, by simp only [runL, bind, Option.bind, hsb, hsc]⟩
  | setR n e => intro i st st1 st2 _ _; exact ⟨_, rfl⟩
  | shflXor d s m =>
    intro i st st1 st2 h _
    simp only [runL] at h ⊢
    split at h
    · rw [if_pos (by assumption)]; exact ⟨_, rfl⟩
    · cases h
  | loadIdx d b ix =>
    intro i st st1 st2 h hs
    simp only [runL, bind, Option.bind] at h ⊢
    split at h; · cases h
    rename_i j hj
    rw [hj]
    dsimp only at h ⊢
    split at h; · cases h
    rename_i v hv
    obtain ⟨v', hv'⟩ := load_transfer hs hv
    rw [hv']; exact ⟨_, rfl⟩
  | loadV4 d0 d1 d2 d3 b ix =>
    intro i st st1 st2 h hs
    simp only [runL, bind, Option.bind] at h ⊢
    split at h; · cases h
    rename_i j hj
    rw [hj]
    dsimp only at h ⊢
    split at h; · cases h
    rename_i v0 h0
    obtain ⟨u0, hu0⟩ := load_transfer hs h0
    rw [hu0]; dsimp only at h ⊢
    split at h; · cases h
    rename_i v1 h1
    obtain ⟨u1, hu1⟩ := load_transfer hs h1
    rw [hu1]; dsimp only at h ⊢
    split at h; · cases h
    rename_i v2 h2
    obtain ⟨u2, hu2⟩ := load_transfer hs h2
    rw [hu2]; dsimp only at h ⊢
    split at h; · cases h
    rename_i v3 h3
    obtain ⟨u3, hu3⟩ := load_transfer hs h3
    rw [hu3]; exact ⟨_, rfl⟩
  | storeLane b ix n =>
    intro i st st1 st2 h hs
    obtain ⟨j, m', hj, hn, hb, hb', hm', rfl⟩ := runL_store h
    obtain ⟨m2', hm2'⟩ := store_transfer _ (st2.reg n).bits hs hm'
    exact ⟨_, runL_store_of hj ⟨hn, hb, hb'⟩ hm2'⟩
  | storeLane0 b ix n =>
    intro i st st1 st2 h hs
    rcases runL_store0 h with ⟨hl, j, m', hj, hn, hb, hb', hm', rfl⟩ | ⟨hl, rfl⟩
    · obtain ⟨m2', hm2'⟩ := store_transfer _ (st2.reg n).bits hs hm'
      exact ⟨_, runL_store0_of hl hj ⟨hn, hb, hb'⟩ hm2'⟩
    · exact ⟨st2, runL_store0_off hl⟩
  | forN n body ih =>
    intro i st st1 st2 h hs
    simp only [runL] at h ⊢
    split at h
    · rename_i hn
      rw [if_pos hn]
      exact transfer_fold (ih := ih) _ st st2 st1 h hs
    · cases h
  | forM b a body ih =>
    intro i st st1 st2 h hs
    obtain ⟨ha, n, hn, hn31, hf⟩ := runL_forM h
    obtain ⟨st2', h2⟩ := transfer_fold (ih := ih) _ st st2 st1 hf hs
    exact ⟨st2', runL_forM_of ha hn hn31 h2⟩
  | _ => intro i st st1 st2 h; simp [runL] at h


/-- **L3: a lane leaves alone what other lanes may touch.** -/
theorem runL_frame_mem (hw : Hw) (nbuf cta : Nat) (A : Acc) (rd : Buf → Nat → Option Nat)
    (hA : ∀ l b j, A l b j → b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) (l : Lane) (o : Nat → UInt32) :
    ∀ (s : EWStmt) (i : Nat) (st st' : LSt), AccOk cta A rd i s → runL hw nbuf cta l.val o rd i s st = some st' →
      ∀ b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) → ¬(A l b j ∧ ∀ l', A l' b j → l' = l) →
        st'.mem.load (bufW nbuf b j) 4 = st.mem.load (bufW nbuf b j) 4 := by
  have loop : ∀ (body : EWStmt) (n : Nat), (∀ jj, jj < n → AccOk cta A rd jj body) →
      (∀ (i : Nat) (st st' : LSt), AccOk cta A rd i body → runL hw nbuf cta l.val o rd i body st = some st' →
        ∀ b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) → ¬(A l b j ∧ ∀ l', A l' b j → l' = l) →
          st'.mem.load (bufW nbuf b j) 4 = st.mem.load (bufW nbuf b j) 4) →
      ∀ (st st' : LSt), (List.range n).foldlM (fun st jj => runL hw nbuf cta l.val o rd jj body st) st = some st' →
      ∀ b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) → ¬(A l b j ∧ ∀ l', A l' b j → l' = l) →
        st'.mem.load (bufW nbuf b j) 4 = st.mem.load (bufW nbuf b j) 4 := by
    intro body n hacc ih st st' h b j hv hne
    have key : ∀ (L : List Nat), (∀ jj ∈ L, jj < n) → ∀ (sa sa' : LSt),
        L.foldlM (fun st jj => runL hw nbuf cta l.val o rd jj body st) sa = some sa' →
        sa'.mem.load (bufW nbuf b j) 4 = sa.mem.load (bufW nbuf b j) 4 := by
      intro L
      induction L with
      | nil => intro _ sa sa' h; simp only [List.foldlM_nil] at h; cases h; rfl
      | cons jj js ihj =>
        intro hL sa sa' h
        simp only [List.foldlM_cons, bind, Option.bind] at h
        split at h; · cases h
        rename_i s1 hs1
        rw [ihj (fun x hx => hL x (List.mem_cons_of_mem _ hx)) s1 sa' h,
          ih jj sa s1 (hacc jj (hL jj List.mem_cons_self)) hs1 b j hv hne]
    exact key _ (fun x hx => List.mem_range.mp hx) st st' h
  intro s
  induction s with
  | skip => intro i st st' _ h; cases h; intros; rfl
  | seq a b iha ihb =>
    intro i st st' hacc h bb j hv hne
    simp only [runL, bind, Option.bind] at h
    split at h; · cases h
    rename_i sa hsa
    rw [ihb i sa st' hacc.2 h bb j hv hne, iha i st sa hacc.1 hsa bb j hv hne]
  | setR n e => intro i st st' _ h; cases h; intros; rfl
  | shflXor d s m =>
    intro i st st' _ h; simp only [runL] at h; split at h
    · cases h; intros; rfl
    · cases h
  | loadIdx d b ix =>
    intro i st st' _ h
    simp only [runL, bind, Option.bind] at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    cases h; intros; rfl
  | loadV4 d0 d1 d2 d3 b ix =>
    intro i st st' _ h
    simp only [runL, bind, Option.bind] at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    cases h; intros; rfl
  | storeLane b0 ix n =>
    intro i st st' hacc h b j hv hne
    obtain ⟨j0, m', hj0, -, -, -, hm', rfl⟩ := runL_store h
    have hjx := idxL_eval rd cta i l ix j0 hj0
    have hown := (hacc l)
    rw [hjx] at hown
    exact apart_elems (hA l b0 j0 hown.1) hv
      (fun ⟨hb, hjj⟩ => hne (hb ▸ hjj ▸ hown)) hm'
  | storeLane0 b0 ix n =>
    intro i st st' hacc h b j hv hne
    rcases runL_store0 h with ⟨hl0, j0, m', hj0, -, -, -, hm', rfl⟩ | ⟨_, rfl⟩
    · have hl : l = lane0 := Fin.ext hl0
      subst hl
      have hjx := idxL_eval rd cta i lane0 ix j0 hj0
      have hown : A lane0 b0 (ix.eval cta i lane0 (fun _ _ => 0) (imOf rd)) ∧
          ∀ l', A l' b0 (ix.eval cta i lane0 (fun _ _ => 0) (imOf rd)) → l' = lane0 := hacc
      rw [hjx] at hown
      exact apart_elems (hA lane0 b0 j0 hown.1) hv
        (fun ⟨hb, hjj⟩ => hne (hb ▸ hjj ▸ hown)) hm'
    · rfl
  | forN n body ih =>
    intro i st st' hacc h b j hv hne
    simp only [runL] at h
    split at h
    · exact loop body n hacc ih st st' h b j hv hne
    · cases h
  | forM b0 a body ih =>
    intro i st st' hacc h b j hv hne
    obtain ⟨_, n, hn, _, hf⟩ := runL_forM h
    exact loop body n (fun jj hj => hacc jj (by simp [imOf, hn]; exact hj)) ih st st' hf b j hv hne
  | _ => intro i st st' _ h; simp [runL] at h


/-- **A lane's run leaves alone every element it does not store.** -/
theorem runL_frame_st (hw : Hw) (nbuf cta : Nat) (rd : Buf → Nat → Option Nat) (l : Lane) (o : Nat → UInt32) :
    ∀ (s : EWStmt) (i : Nat) (st st' : LSt), runL hw nbuf cta l.val o rd i s st = some st' → RdOk nbuf st.mem rd →
      ∀ b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) → ¬ StoresL cta (imOf rd) l i s b j →
        st'.mem.load (bufW nbuf b j) 4 = st.mem.load (bufW nbuf b j) 4 := by
  have loop : ∀ (body : EWStmt) (L : List Nat),
      (∀ (i : Nat) (st st' : LSt), runL hw nbuf cta l.val o rd i body st = some st' → RdOk nbuf st.mem rd →
        ∀ b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) → ¬ StoresL cta (imOf rd) l i body b j →
          st'.mem.load (bufW nbuf b j) 4 = st.mem.load (bufW nbuf b j) 4) →
      ∀ (b : Buf) (j : Nat), (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) → (∀ t ∈ L, ¬ StoresL cta (imOf rd) l t body b j) →
      ∀ (st st' : LSt), L.foldlM (fun st t => runL hw nbuf cta l.val o rd t body st) st = some st' →
        RdOk nbuf st.mem rd → st'.mem.load (bufW nbuf b j) 4 = st.mem.load (bufW nbuf b j) 4 := by
    intro body L ih b j hv hn
    induction L with
    | nil => intro st st' h _; simp only [List.foldlM_nil] at h; cases h; rfl
    | cons t ts iht =>
      intro st st' h hrd
      simp only [List.foldlM_cons, bind, Option.bind] at h
      split at h; · cases h
      rename_i s1 hs1
      rw [iht (fun t' ht' => hn t' (List.mem_cons_of_mem _ ht')) s1 st' h (runL_rdOk body t st s1 hs1 hrd),
        ih t st s1 hs1 hrd b j hv (hn t List.mem_cons_self)]
  have store : ∀ (m m' : Mem) (b0 : Buf) (j0 : Nat) (v : UInt64) (b : Buf) (j : Nat), b0 < nbuf → b0 < 268435449 →
      j0 < 2 ^ 32 → (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) → ¬(b = b0 ∧ j = j0) →
      m.store (bufW nbuf b0 j0) 4 v = some m' → m'.load (bufW nbuf b j) 4 = m.load (bufW nbuf b j) 4 :=
    fun m m' b0 j0 v b j h1 h2 h3 hv hne hs => apart_elems ⟨h1, h2, h3⟩ hv hne hs
  intro s
  induction s with
  | skip => intro i st st' h _; cases h; intros; rfl
  | seq x y ihx ihy =>
    intro i st st' h hrd b j hv hn
    simp only [runL, bind, Option.bind] at h
    split at h; · cases h
    rename_i sa hsa
    rw [ihy i sa st' h (runL_rdOk x i st sa hsa hrd) b j hv (fun h => hn (Or.inr h)),
      ihx i st sa hsa hrd b j hv (fun h => hn (Or.inl h))]
  | setR n e => intro i st st' h _; cases h; intros; rfl
  | shflXor d s m =>
    intro i st st' h _; simp only [runL] at h; split at h
    · cases h; intros; rfl
    · cases h
  | loadIdx d b ix =>
    intro i st st' h _
    simp only [runL, bind, Option.bind] at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    cases h; intros; rfl
  | loadV4 d0 d1 d2 d3 b ix =>
    intro i st st' h _
    simp only [runL, bind, Option.bind] at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    dsimp only at h
    split at h; · cases h
    cases h; intros; rfl
  | storeLane b0 ix n =>
    intro i st st' h hrd b j hv hn
    obtain ⟨j0, m', hj0, -, hb1, hb2, hm', rfl⟩ := runL_store h
    have hjx := idxL_eval rd cta i l ix j0 hj0
    exact store _ _ b0 j0 _ b j hb1 hb2 (idxL_lt hrd ix j0 hj0) hv
      (fun ⟨hb, hjj⟩ => hn ⟨hb, hjj.trans hjx.symm⟩) hm'
  | storeLane0 b0 ix n =>
    intro i st st' h hrd b j hv hn
    rcases runL_store0 h with ⟨hl0, j0, m', hj0, -, hb1, hb2, hm', rfl⟩ | ⟨_, rfl⟩
    · have hl : l = lane0 := Fin.ext hl0
      subst hl
      have hjx := idxL_eval rd cta i lane0 ix j0 hj0
      exact store _ _ b0 j0 _ b j hb1 hb2 (idxL_lt hrd ix j0 hj0) hv
        (fun ⟨hb, hjj⟩ => hn ⟨rfl, hb, hjj.trans hjx.symm⟩) hm'
    · rfl
  | forN n body ih =>
    intro i st st' h hrd b j hv hn
    simp only [runL] at h
    split at h
    · exact loop body _ ih b j hv (fun t ht hs => hn ⟨t, List.mem_range.mp ht, hs⟩) st st' h hrd
    · cases h
  | forM bb a body ih =>
    intro i st st' h hrd b j hv hn
    obtain ⟨_, n, hn', _, hf⟩ := runL_forM h
    exact loop body _ ih b j hv (fun t ht hs => hn ⟨t, by simp [imOf, hn']; exact List.mem_range.mp ht, hs⟩)
      st st' hf hrd
  | _ => intro i st st' h; simp [runL] at h


/-- **A statement's thread ends**, at some fuel, as its lane's run says:
    `wpE_body` for the code the kernel ships, its collectives answered by the
    thread's locals. -/
theorem thread_ends (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (name : String) (nbuf smem : Nat)
    (s : EWStmt) (W : Nat) (hW : 0 < W) (hW31 : W < 2 ^ 31) (cta t bd : Nat) (htW : t < 32 * W) (ht : t < 2 ^ 32)
    (hv : cta * W + t / 32 < 2 ^ 32) (w : World) (ans : List Answer) (hWo : Words ans) (st' : LSt)
    (hfine : Fine (emitGo (fnBody (params nbuf) (Body.body nbuf s W)) ⟨(params nbuf).length, 0, [], [], [], none⟩).2)
    (rd : Buf → Nat → Option Nat) (hrd : RdOk nbuf w.mem rd) (hnan : w.mem.nan = hw.nan)
    (hrun : runL hw nbuf (cta * W + t / 32) (t % 32) (wordAt ans) rd 0 s ⟨fun _ => ⟨0⟩, w.mem, []⟩ = some st') :
    ∃ steps u, thread (Body.kernel name nbuf smem s W).code ans (threadArgs cta bd (bufArgs nbuf) t) w hw steps
        = .ok u ∧ u.mem = st'.mem ∧ collCalls u w.obs.length = st'.calls := by
  have hE := emit_ends (cfg := { env := [], locals := threadLocals ans w.obs.length hw })
    (P0 := At (List.toArray (threadArgs cta bd (bufArgs nbuf) t)) w)
    (wpE_body hw hw32 _ nbuf s W hW hW31 cta t bd htW ht hv w ans (answersFrom_thread _ _ _)
      (approxFrom_thread _ _ _) hWo st' rd hrd hnan hrun) hfine
    (fun Γ' w' h => ⟨by rw [h.1]; simp [params, threadArgs, bufArgs, idParams], h⟩)
  obtain ⟨k, Γ', u, hk, hu⟩ := hE _ _ ⟨rfl, rfl⟩
  refine ⟨k, u, ?_, hu⟩
  unfold thread
  rw [show (Body.kernel name nbuf smem s W).code = emit (Body.body nbuf s W) (params nbuf) from rfl, hk]

/-- **A statement's thread that finishes ends as its lane's run.** -/
theorem thread_body (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (name : String) (nbuf smem : Nat)
    (s : EWStmt) (W : Nat) (hW : 0 < W) (hW31 : W < 2 ^ 31) (cta t bd : Nat) (htW : t < 32 * W) (ht : t < 2 ^ 32)
    (hv : cta * W + t / 32 < 2 ^ 32) (w u : World) (ans : List Answer) (hWo : Words ans) (st' : LSt)
    (hfine : Fine (emitGo (fnBody (params nbuf) (Body.body nbuf s W)) ⟨(params nbuf).length, 0, [], [], [], none⟩).2)
    (rd : Buf → Nat → Option Nat) (hrd : RdOk nbuf w.mem rd) (hnan : w.mem.nan = hw.nan)
    (hrun : runL hw nbuf (cta * W + t / 32) (t % 32) (wordAt ans) rd 0 s ⟨fun _ => ⟨0⟩, w.mem, []⟩ = some st')
    {steps : Nat} (hr : thread (Body.kernel name nbuf smem s W).code ans (threadArgs cta bd (bufArgs nbuf) t) w hw steps
      = .ok u) :
    u.mem = st'.mem ∧ collCalls u w.obs.length = st'.calls := by
  obtain ⟨k, u', hk, hu⟩ := thread_ends hw hw32 name nbuf smem s W hW hW31 cta t bd htW ht hv w ans hWo st' hfine
    rd hrd hnan hrun
  rw [thread_det hr hk]; exact hu


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

theorem takeWhile_range (n t : Nat) (ht : t < n) : (List.range n).takeWhile (· ≠ t) = List.range t := by
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_lt ht
  rw [show t + d + 1 = t + (d + 1) by omega, List.range_add, List.takeWhile_append_of_pos]
  · simp [List.range_succ_eq_map]
  · intro x hx; simp at hx; simp; omega

/-- **A warp statement's kernel runs the warp machine**: a block of `nw` warps
    of the statement's kernel that finishes leaves every element a lane of its
    warp `w` may touch as the warp machine's run of the statement for block
    `cta·nw + w` does at `F32` from `m0`, which holds what the block starts
    with at those elements; its shuffles answered as the laws say and the
    hardware's approximations `hw`'s, the warps touching nothing another
    stores. Elements no lane stores it leaves alone, and memory keeps its
    shape and what `rd` reads. -/
theorem block_lockstep (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (name : String) (nbuf smem : Nat)
    (s : EWStmt) (nw : Nat) (hnw : 0 < nw) (hnw31 : nw < 2 ^ 31) (cta : Nat) (hcta : cta * nw + nw ≤ 2 ^ 32)
    (hN : 32 * nw ≤ 2 ^ 32) (A : Nat → Acc) (rd : Buf → Nat → Option Nat)
    (hacc : ∀ w, w < nw → AccOk (cta * nw + w) (A (cta * nw + w)) rd 0 s)
    (hA : ∀ c l b j, A c l b j → b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32)
    (happ : ∀ w w' l l' b j, w < nw → w' < nw → w ≠ w' → A (cta * nw + w) l b j →
      ¬ StoresL (cta * nw + w') (imOf rd) l' 0 s b j)
    (hfine : Fine (emitGo (fnBody (params nbuf) (Body.body nbuf s nw)) ⟨(params nbuf).length, 0, [], [], [], none⟩).2)
    (w w' : World) (hrd : RdOk nbuf w.mem rd) (hnan : w.mem.nan = hw.nan)
    (hok : ∀ t, t < 32 * nw → ∃ st', runL hw nbuf (cta * nw + t / 32) (t % 32) (fun _ => 0) rd 0 s
      ⟨fun _ => ⟨0⟩, rm (ByteArray.mk (Array.replicate smem 0)) w.mem, []⟩ = some st')
    (m0 : Mem) (hin : ∀ wi (l : Lane) b j, wi < nw → A (cta * nw + wi) l b j →
      w.mem.load (bufW nbuf b j) 4 = m0.load (bufW nbuf b j) 4 ∧ (m0.load (bufW nbuf b j) 4).isSome)
    (hr : BlockRuns (Body.kernel name nbuf smem s nw) cta (32 * nw) (bufArgs nbuf) hw w w') :
    (∀ wi (l : Lane) b j, wi < nw → A (cta * nw + wi) l b j →
      w'.mem.load (bufW nbuf b j) 4 =
        some ((@WStmt.run F32 (F32.ops hw) _ (s.elabAt (cta * nw + wi) 0 (fun _ _ => 0) (imOf rd))
          (ws0 nbuf m0)).mem b j).bits) ∧
    (∀ b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) →
      (∀ wi l, wi < nw → ¬ StoresL (cta * nw + wi) (imOf rd) l 0 s b j) →
      w'.mem.load (bufW nbuf b j) 4 = w.mem.load (bufW nbuf b j) 4) ∧
    MemSame w.mem w'.mem ∧ RdOk nbuf w'.mem rd ∧ w'.mem.nan = hw.nan := by
  obtain ⟨steps, ans, w'', calls, stores, ans', hfold, hans, hset, rfl⟩ := hr
  let N := 32 * nw
  let A0 := ByteArray.mk (Array.replicate smem 0)
  let W0 := rm A0 w.mem
  let vc (t : Nat) : Nat := cta * nw + t / 32
  let ln (t : Nat) : Lane := ⟨t % 32, Nat.mod_lt _ (by decide)⟩
  have hvc : ∀ t, t < N → vc t < 2 ^ 32 := fun t ht => by
    have : t / 32 < nw := by omega
    show cta * nw + t / 32 < 2 ^ 32; omega
  have hWd : ∀ t, t < N → Words (ans t) := fun t ht => by
    rw [← hset t ht]; exact answersOf_words (answers_at hans ht)
  let o (t : Nat) : Nat → UInt32 := wordAt (ans t)
  let start (m : Mem) : LSt := ⟨fun _ => ⟨0⟩, rm A0 m, []⟩
  let f (t : Nat) (m : Mem) : Mem :=
    if t < N then ((runL hw nbuf (vc t) (t % 32) (o t) rd 0 s (start m)).map LSt.mem).getD m else m
  let P (t : Nat) (m : Mem) : Prop := t < N ∧ MemSame (rm A0 m) W0 ∧
    (∀ bb jj, A (vc t) (ln t) bb jj → m.load (bufW nbuf bb jj) 4 = W0.load (bufW nbuf bb jj) 4) ∧
    RdOk nbuf m rd ∧ m.nan = hw.nan
  let C (t : Nat) (m : Mem) (cs : List (Intr × List V)) (_ : List (Nat × UInt64 × Nat × UInt64)) : Prop :=
    ∃ st', runL hw nbuf (vc t) (t % 32) (o t) rd 0 s (start m) = some st' ∧ cs = st'.calls
  have hsucc : ∀ t m, P t m → ∃ st', runL hw nbuf (vc t) (t % 32) (o t) rd 0 s (start m) = some st' := by
    intro t m hP
    obtain ⟨st0, h0⟩ := hok t hP.1
    exact runL_transfer hw nbuf (vc t) (t % 32) _ (o t) rd s 0 _ st0 (start m) h0 hP.2.1.symm
  have hacc' : ∀ t, t < N → AccOk (vc t) (A (vc t)) rd 0 s := fun t ht => hacc (t / 32) (by omega)
  -- a thread leaves alone what another thread may touch
  have hother : ∀ t m u, RdOk nbuf m rd → u < N → u ≠ t → ∀ bb jj, A (vc u) (ln u) bb jj →
      (f t m).load (bufW nbuf bb jj) 4 = m.load (bufW nbuf bb jj) 4 := by
    intro t m u hm hu hut bb jj hAu
    simp only [f]
    split
    · rename_i ht
      cases hrun : runL hw nbuf (vc t) (t % 32) (o t) rd 0 s (start m) with
      | none => rfl
      | some st' =>
        simp only [Option.map_some, Option.getD_some]
        obtain ⟨hb1, hb2, hj1⟩ := hA _ _ bb jj hAu
        have hrun' : runL hw nbuf (vc t) (ln t).val (o t) rd 0 s (start m) = some st' := hrun
        by_cases hw' : u / 32 = t / 32
        · have hvu : vc u = vc t := by show cta * nw + u / 32 = cta * nw + t / 32; rw [hw']
          rw [hvu] at hAu
          rw [runL_frame_mem hw nbuf (vc t) (A (vc t)) rd (hA _) (ln t) (o t) s 0 (start m) st' (hacc' t ht) hrun'
            bb jj (hA _ _ bb jj hAu) (fun ⟨_, hx⟩ => hut (by
              have := congrArg Fin.val (hx (ln u) hAu)
              simp only [ln] at this
              omega))]
          exact load_buf_arena m A0 (decode_bufW hb1 hb2 hj1) 4
        · rw [runL_frame_st hw nbuf (vc t) rd (ln t) (o t) s 0 (start m) st' hrun' ((rdOk_arena A0).mpr hm) bb jj
            ⟨hb1, hb2, hj1⟩ (happ (u / 32) (t / 32) (ln u) (ln t) bb jj (by omega) (by omega) hw' hAu)]
          exact load_buf_arena m A0 (decode_bufW hb1 hb2 hj1) 4
    · rfl
  have hfrd : ∀ t m, RdOk nbuf m rd → RdOk nbuf (f t m) rd := by
    intro t m hm
    simp only [f]
    split
    · cases hrun : runL hw nbuf (vc t) (t % 32) (o t) rd 0 s (start m) with
      | none => exact hm
      | some st' => exact runL_rdOk s 0 _ _ hrun ((rdOk_arena A0).mpr hm)
    · exact hm
  have hfnan : ∀ t m, RdOk nbuf m rd → m.nan = hw.nan → (f t m).nan = hw.nan := by
    intro t m hm hn
    simp only [f]
    split
    · cases hrun : runL hw nbuf (vc t) (t % 32) (o t) rd 0 s (start m) with
      | none => exact hn
      | some st' => exact (runL_nan s 0 _ _ hrun ((rdOk_arena A0).mpr hm)).trans hn
    · exact hn
  have hkeep : ∀ t u m, t ≠ u → P u m → P u (f t m) := by
    intro t u m htu ⟨hu, hs, hl, hrm, hn⟩
    refine ⟨hu, ?_, fun bb jj hA' => ?_, hfrd t m hrm, hfnan t m hrm hn⟩
    · simp only [f]
      split
      · cases hrun : runL hw nbuf (vc t) (t % 32) (o t) rd 0 s (start m) with
        | none => exact hs
        | some st' =>
          simp only [Option.map_some, Option.getD_some]
          have h1 := runL_same hw nbuf (vc t) (t % 32) (o t) rd s 0 (start m) st' hrun
          exact ((memSame_rm h1 A0).symm.trans (by simpa [rm] using hs))
      · exact hs
    · rw [hother t m u hrm hu (Ne.symm htu) bb jj hA', hl bb jj hA']
  have hP0 : ∀ t, t < N → P t W0 := fun t ht =>
    ⟨ht, by simpa [W0, rm] using MemSame.refl W0, fun _ _ _ => rfl, (rdOk_arena A0).mpr hrd, hnan⟩
  obtain ⟨hmem, hC⟩ := pass_spec P f C (fun t sw u hP hu => by
      obtain ⟨st', hst'⟩ := hsucc t sw.mem hP
      have h := thread_body hw hw32 name nbuf smem s nw hnw hnw31 cta t (32 * nw) hP.1 (by omega) (hvc t hP.1)
        _ u (ans t) (hWd t hP.1) st' hfine rd (by exact (rdOk_arena A0).mpr hP.2.2.2.1) (by exact hP.2.2.2.2) hst' hu
      refine ⟨?_, st', hst', h.2⟩
      rw [h.1]
      simp only [f, if_pos hP.1, hst', Option.map_some, Option.getD_some])
    hkeep (List.range N) _ _ List.nodup_range (fun t ht => hP0 t (List.mem_range.mp ht)) hfold
  let M (t : Nat) : Mem := (List.range t).foldl (fun m t => f t m) W0
  have hPM : ∀ t, t < N → ∀ t', t' ≤ t → P t (M t') := by
    intro t ht t' ht'
    induction t' with
    | zero => exact hP0 t ht
    | succ t' ih =>
      show P t ((List.range (t' + 1)).foldl (fun m t => f t m) W0)
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
      exact hkeep t' t _ (by omega) (ih (by omega))
  have hCl : ∀ t, t < N → ∃ st', runL hw nbuf (vc t) (t % 32) (o t) rd 0 s (start (M t)) = some st' ∧
      calls t = st'.calls := by
    intro t ht
    have h := hC t (List.mem_range.mpr ht)
    rw [takeWhile_range N t ht] at h
    exact h
  let tr (t : Nat) : LSt := (runL hw nbuf (vc t) (t % 32) (o t) rd 0 s (start (M t))).getD (start (M t))
  have hruns : ∀ t, t < N → runL hw nbuf (vc t) (t % 32) (o t) rd 0 s (start (M t)) = some (tr t) := by
    intro t ht; obtain ⟨st', h, _⟩ := hCl t ht; simp only [tr, h, Option.getD_some]
  have hcalls : ∀ t, t < N → calls t = (tr t).calls := by
    intro t ht; obtain ⟨st', h, hc⟩ := hCl t ht; simp only [tr, h, Option.getD_some]; exact hc
  -- each warp runs with the warp machine
  have hlock : ∀ wi, wi < nw → ∀ l : Lane, Rel nbuf (A (cta * nw + wi))
      (@WStmt.run F32 (F32.ops hw) _ (s.elabAt (cta * nw + wi) 0 (fun _ _ => 0) (imOf rd)) (ws0 nbuf m0))
      (tr (32 * wi + l.val)) l := by
    intro wi hwi
    have hT : ∀ l : Lane, 32 * wi + l.val < N := fun l => by have : l.val < 32 := l.isLt; omega
    have hvcl : ∀ l : Lane, vc (32 * wi + l.val) = cta * nw + wi := fun l => by
      have : l.val < 32 := l.isLt; show cta * nw + (32 * wi + l.val) / 32 = cta * nw + wi; omega
    have hlnl : ∀ l : Lane, (32 * wi + l.val) % 32 = l.val := fun l => by have : l.val < 32 := l.isLt; omega
    have hS : Settled (fun l => o (32 * wi + l.val)) (fun l => calls (32 * wi + l.val)) := by
      intro l k a ty m hF
      have ha := answers_at hans (hT l)
      rw [hset _ (hT l)] at ha
      obtain ⟨args, hl, hk⟩ := answersOf_bfly ha hF
      have h32 : (32 * wi + l.val) / 32 = wi := by have : l.val < 32 := l.isLt; omega
      rw [h32, hlnl l] at hl
      unfold laneArg at hl
      refine ⟨args, ?_, ?_⟩
      · show (calls (32 * wi + (l.val ^^^ m.toNat) % 32))[k]? = _
        split at hl
        · rename_i c' a' hca
          split at hl
          · rename_i hc
            cases hl
            rw [hca, show c' = .shflBfly from by revert hc; cases c' <;> decide]
          · cases hl
        · cases hl
      · show wordAt (ans (32 * wi + l.val)) k = _
        simp only [wordAt, hk]
        cases args with
        | nil => rfl
        | cons x xs => cases x <;> rfl
    refine lockstep hw hw32 nbuf (cta * nw + wi) (A (cta * nw + wi)) rd (fun l => o (32 * wi + l.val))
      (fun l => calls (32 * wi + l.val)) hS (hA _) s 0 (ws0 nbuf m0)
      (fun l => start (M (32 * wi + l.val))) (fun l => tr (32 * wi + l.val)) (hacc wi hwi)
      (fun l => by have h := hruns _ (hT l); rw [hvcl l, hlnl l] at h; exact h)
      (fun l' => ⟨fun n => rfl, fun bb jj hA' => by
          obtain ⟨hb1, hb2, hj1⟩ := hA _ l' bb jj hA'
          have hP := hPM (32 * wi + l'.val) (hT l') (32 * wi + l'.val) (Nat.le_refl _)
          have hA'' : A (vc (32 * wi + l'.val)) (ln (32 * wi + l'.val)) bb jj := by
            rw [hvcl l']
            have : ln (32 * wi + l'.val) = l' := Fin.ext (hlnl l')
            rw [this]; exact hA'
          show (rm A0 (M (32 * wi + l'.val))).load _ 4 = _
          rw [load_buf_arena _ _ (decode_bufW hb1 hb2 hj1), hP.2.2.1 bb jj hA'']
          show (rm A0 w.mem).load _ 4 = _
          rw [load_buf_arena _ _ (decode_bufW hb1 hb2 hj1), (hin wi l' bb jj hwi hA').1]
          obtain ⟨v, hv⟩ := Option.isSome_iff_exists.mp (hin wi l' bb jj hwi hA').2
          simp [ws0, hv]⟩)
      (fun _ _ => by show (0 : UInt64).toNat < 2 ^ 32; decide)
      (fun _ _ => rfl)
      (fun l' => by show (tr (32 * wi + l'.val)).calls <+: calls (32 * wi + l'.val); rw [hcalls _ (hT l')]
                    exact List.prefix_refl _)
  simp only at hmem
  have hMS : ∀ t, MemSame W0 (rm A0 (M t)) := by
    intro t
    induction t with
    | zero => exact MemSame.refl _
    | succ t ih =>
      show MemSame W0 (rm A0 ((List.range (t + 1)).foldl (fun m t => f t m) W0))
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
      refine ih.trans ?_
      show MemSame (rm A0 (M t)) (rm A0 (f t (M t)))
      simp only [f]
      split
      · cases hrun : runL hw nbuf (vc t) (t % 32) (o t) rd 0 s (start (M t)) with
        | none => exact MemSame.refl _
        | some st' =>
          simp only [Option.map_some, Option.getD_some]
          exact memSame_rm (runL_same hw nbuf (vc t) (t % 32) (o t) rd s 0 (start (M t)) st' hrun) A0
      · exact MemSame.refl _
  have hMrd : ∀ t, RdOk nbuf (M t) rd := by
    intro t
    induction t with
    | zero => exact (rdOk_arena A0).mpr hrd
    | succ t ih =>
      show RdOk nbuf ((List.range (t + 1)).foldl (fun m t => f t m) W0) rd
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
      exact hfrd t _ ih
  have hMnan : ∀ t, (M t).nan = hw.nan := by
    intro t
    induction t with
    | zero => exact hnan
    | succ t ih =>
      show ((List.range (t + 1)).foldl (fun m t => f t m) W0).nan = _
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
      exact hfnan t _ (hMrd t) ih
  refine ⟨fun wi l b j hwi hAl => ?_, fun b j hv hn => ?_, ?_, ?_, ?_⟩
  rotate_left
  · obtain ⟨hb1, hb2, hj1⟩ := hv
    show ({ w''.mem with arena := w.mem.arena } : Mem).load _ 4 = _
    rw [load_buf_arena _ _ (decode_bufW hb1 hb2 hj1), hmem]
    have tail : ∀ (L : List Nat) m, RdOk nbuf m rd →
        (L.foldl (fun m t => f t m) m).load (bufW nbuf b j) 4 = m.load (bufW nbuf b j) 4 := by
      intro L
      induction L with
      | nil => intros; rfl
      | cons u us ih =>
        intro m hm
        rw [List.foldl_cons, ih _ (hfrd u m hm)]
        simp only [f]
        split
        · rename_i hu
          cases hrun : runL hw nbuf (vc u) (u % 32) (o u) rd 0 s (start m) with
          | none => rfl
          | some st' =>
            simp only [Option.map_some, Option.getD_some]
            rw [runL_frame_st hw nbuf (vc u) rd (ln u) (o u) s 0 (start m) st' hrun ((rdOk_arena A0).mpr hm) b j
              ⟨hb1, hb2, hj1⟩ (hn (u / 32) (ln u) (by omega))]
            exact load_buf_arena m A0 (decode_bufW hb1 hb2 hj1) 4
        · rfl
    rw [tail _ _ (by exact (rdOk_arena A0).mpr hrd)]
    exact load_buf_arena w.mem A0 (decode_bufW hb1 hb2 hj1) 4
  · show MemSame w.mem ({ w''.mem with arena := w.mem.arena } : Mem)
    rw [hmem]
    exact memSame_rm (hMS N) w.mem.arena
  · show RdOk nbuf ({ w''.mem with arena := w.mem.arena } : Mem) rd
    rw [hmem]
    exact (rdOk_arena w.mem.arena).mpr (hMrd N)
  · show ({ w''.mem with arena := w.mem.arena } : Mem).nan = _
    rw [hmem]
    exact hMnan N
  let t := 32 * wi + l.val
  have htN : t < N := by have : l.val < 32 := l.isLt; show 32 * wi + l.val < 32 * nw; omega
  have hfin := (hlock wi hwi l).2 b j hAl
  obtain ⟨hb1, hb2, hj1⟩ := hA _ l b j hAl
  have hAt : A (vc t) (ln t) b j := by
    have hv1 : vc t = cta * nw + wi := by
      have : l.val < 32 := l.isLt; show cta * nw + (32 * wi + l.val) / 32 = cta * nw + wi; omega
    have hl1 : ln t = l := Fin.ext (by have : l.val < 32 := l.isLt; show (32 * wi + l.val) % 32 = l.val; omega)
    rw [hv1, hl1]; exact hAl
  show ({ w''.mem with arena := w.mem.arena } : Mem).load _ 4 = _
  rw [load_buf_arena _ _ (decode_bufW hb1 hb2 hj1)]
  rw [hmem]
  have hrN : List.range N = List.range t ++ [t] ++ (List.range (N - 1 - t)).map (t + 1 + ·) := by
    rw [← List.range_succ, ← List.range_add]; congr 1; omega
  rw [hrN, List.foldl_append, List.foldl_append, List.foldl_cons, List.foldl_nil]
  have tail : ∀ (L : List Nat), (∀ u ∈ L, u ≠ t ∧ u < N) → ∀ m, RdOk nbuf m rd →
      (L.foldl (fun m t => f t m) m).load (bufW nbuf b j) 4 = m.load (bufW nbuf b j) 4 := by
    intro L
    induction L with
    | nil => intros; rfl
    | cons u us ih =>
      intro hL m hm
      rw [List.foldl_cons, ih (fun x hx => hL x (List.mem_cons_of_mem _ hx)) _ (hfrd u m hm),
        hother u m t hm htN (fun e => (hL u List.mem_cons_self).1 e.symm) b j hAt]
  rw [tail _ (fun u hu => by
    obtain ⟨x, hx, rfl⟩ := List.mem_map.mp hu
    have := List.mem_range.mp hx
    exact ⟨by omega, by omega⟩) _ (by exact hfrd t _ (hMrd t))]
  show (f t (M t)).load _ 4 = _
  simp only [f, if_pos htN, hruns t htN, Option.map_some, Option.getD_some]
  exact hfin


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
  | _, .forM b a body => decide (a < 2 ^ 31) && match rd b a with
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

/-- **A lane's run finishes** where its indexes fit and every element it may
    touch lies inside its buffer, and none is a word `rd` reads. -/
theorem runL_total (hw : Hw) (nbuf cta : Nat) (A : Acc) (rd : Buf → Nat → Option Nat) (l : Lane) (o : Nat → UInt32)
    (hA : ∀ b j, A l b j → b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) (hRd : ∀ b j, A l b j → rd b j = none) :
    ∀ (s : EWStmt) (i : Nat) (st : LSt), fits rd cta i s = true → AccOk cta A rd i s →
      (∀ b j, A l b j → 4 * j + 4 ≤ (st.mem.bufs b).size) →
      ∃ st', runL hw nbuf cta l.val o rd i s st = some st' := by
  have keep : ∀ (st st' : LSt), MemSame st.mem st'.mem → (∀ b j, A l b j → 4 * j + 4 ≤ (st.mem.bufs b).size) →
      ∀ b j, A l b j → 4 * j + 4 ≤ (st'.mem.bufs b).size := fun st st' hs h b j hb => by
    have := hs.size (.buf ⟨b, (hA b j hb).2.1⟩)
    simp only [Mem.region] at this
    rw [← this]; exact h b j hb
  have ld : ∀ (st : LSt) b j, A l b j → (∀ b j, A l b j → 4 * j + 4 ≤ (st.mem.bufs b).size) →
      ∃ v, st.mem.load (bufW nbuf b j) 4 = some v := fun st b j hb h =>
    Option.isSome_iff_exists.mp (bufW_ok (hA b j hb) (h b j hb) 0).1
  have sto : ∀ (st : LSt) b j v, A l b j → (∀ b j, A l b j → 4 * j + 4 ≤ (st.mem.bufs b).size) →
      ∃ m', st.mem.store (bufW nbuf b j) 4 v = some m' := fun st b j v hb h =>
    Option.isSome_iff_exists.mp (bufW_ok (hA b j hb) (h b j hb) v).2
  have ix : ∀ i (e : IdxE), (idxL rd cta i l.val e).isSome →
      idxL rd cta i l.val e = some (e.eval cta i l (fun _ _ => 0) (imOf rd)) := by
    intro i e h
    obtain ⟨j, hj⟩ := Option.isSome_iff_exists.mp h
    rw [hj, idxL_eval rd cta i l e j hj]
  have all32 : ∀ i (e : IdxE), ((List.range 32).all fun l => (idxL rd cta i l e).isSome) = true →
      (idxL rd cta i l.val e).isSome := fun i e h =>
    List.all_eq_true.mp h l.val (List.mem_range.mpr l.isLt)
  have loop : ∀ (body : EWStmt) (n : Nat),
      (∀ (i : Nat) (st : LSt), fits rd cta i body = true → AccOk cta A rd i body →
        (∀ b j, A l b j → 4 * j + 4 ≤ (st.mem.bufs b).size) → ∃ st', runL hw nbuf cta l.val o rd i body st = some st') →
      (∀ j, j < n → fits rd cta j body = true) → (∀ j, j < n → AccOk cta A rd j body) →
      ∀ (st : LSt), (∀ b j, A l b j → 4 * j + 4 ≤ (st.mem.bufs b).size) →
        ∃ st', (List.range n).foldlM (fun st j => runL hw nbuf cta l.val o rd j body st) st = some st' := by
    intro body n ih hf hacc st hs
    have key : ∀ (L : List Nat) (sa : LSt), (∀ j, j ∈ L → j < n) →
        (∀ b j, A l b j → 4 * j + 4 ≤ (sa.mem.bufs b).size) →
        ∃ sb, L.foldlM (fun st j => runL hw nbuf cta l.val o rd j body st) sa = some sb := by
      intro L
      induction L with
      | nil => intro sa _ _; exact ⟨sa, rfl⟩
      | cons j js ihj =>
        intro sa hL hsa
        have hj := hL j List.mem_cons_self
        obtain ⟨s1, hs1⟩ := ih j sa (hf j hj) (hacc j hj) hsa
        obtain ⟨s2, hs2⟩ := ihj s1 (fun k hk => hL k (List.mem_cons_of_mem _ hk))
          (keep sa s1 (runL_same hw nbuf cta l.val o rd body j sa s1 hs1) hsa)
        exact ⟨s2, by simp only [List.foldlM_cons, bind, Option.bind, hs1, hs2]⟩
    exact key _ st (fun j hj => List.mem_range.mp hj) hs
  intro s
  induction s with
  | skip => intro i st _ _ _; exact ⟨st, rfl⟩
  | seq a b iha ihb =>
    intro i st hf hacc hs
    simp only [fits, Bool.and_eq_true] at hf
    obtain ⟨sa, hsa⟩ := iha i st hf.1 hacc.1 hs
    obtain ⟨sb, hsb⟩ := ihb i sa hf.2 hacc.2 (keep st sa (runL_same hw nbuf cta l.val o rd a i st sa hsa) hs)
    exact ⟨sb, by simp only [runL, bind, Option.bind, hsa, hsb]⟩
  | setR n e => intro i st _ _ _; exact ⟨_, rfl⟩
  | shflXor d s m =>
    intro i st hf _ _
    simp only [fits, decide_eq_true_eq] at hf
    simp only [runL, if_pos hf]; exact ⟨_, rfl⟩
  | loadIdx d b e =>
    intro i st hf hacc hs
    simp only [runL, bind, Option.bind, ix i e (all32 i e hf)]
    obtain ⟨v, hv⟩ := ld st b _ (hacc l) hs
    rw [hv]; exact ⟨_, rfl⟩
  | loadV4 d0 d1 d2 d3 b e =>
    intro i st hf hacc hs
    simp only [runL, bind, Option.bind, ix i e (all32 i e hf)]
    have hq : ∀ (q : Nat) (k : Int), k = 4 * (q : Int) → q < 4 →
        ∃ v, st.mem.load (bufW nbuf b (e.eval cta i l (fun _ _ => 0) (imOf rd)) + c64 k) 4 = some v := by
      intro q k hk hq
      have hA' := hacc l q hq
      rw [bufW_add nbuf b _ q k hk (hA b _ hA').2.2]
      exact ld st b _ hA' hs
    obtain ⟨v0, h0⟩ := ld st b _ (by simpa using hacc l 0 (by decide)) hs
    obtain ⟨v1, h1⟩ := hq 1 4 (by omega) (by decide)
    obtain ⟨v2, h2⟩ := hq 2 8 (by omega) (by decide)
    obtain ⟨v3, h3⟩ := hq 3 12 (by omega) (by decide)
    rw [h0]; dsimp only; rw [h1]; dsimp only; rw [h2]; dsimp only; rw [h3]
    exact ⟨_, rfl⟩
  | storeLane b e n =>
    intro i st hf hacc hs
    obtain ⟨m', hm'⟩ := sto st b _ (st.reg n).bits (hacc l).1 hs
    have hb := hA b _ (hacc l).1
    exact ⟨_, runL_store_of (ix i e (all32 i e hf)) ⟨hRd b _ (hacc l).1, hb.1, hb.2.1⟩ hm'⟩
  | storeLane0 b e n =>
    intro i st hf hacc hs
    by_cases hl : l.val = 0
    · have hl0 : l = lane0 := Fin.ext hl
      subst hl0
      obtain ⟨m', hm'⟩ := sto st b _ (st.reg n).bits hacc.1 hs
      have hb := hA b _ hacc.1
      exact ⟨_, runL_store0_of rfl (ix i e hf) ⟨hRd b _ hacc.1, hb.1, hb.2.1⟩ hm'⟩
    · exact ⟨_, runL_store0_off hl⟩
  | forN n body ih =>
    intro i st hf hacc hs
    simp only [fits, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range] at hf
    simp only [runL, if_pos hf.1]
    exact loop body n ih (fun j hj => hf.2 j hj) hacc st hs
  | forM b a body ih =>
    intro i st hf hacc hs
    simp only [fits, Bool.and_eq_true, decide_eq_true_eq] at hf
    obtain ⟨ha, hf⟩ := hf
    split at hf
    · rename_i n hn
      simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range] at hf
      obtain ⟨st', h⟩ := loop body n ih (fun j hj => hf.2 j hj)
        (fun j hj => hacc j (by simp [imOf, hn]; exact hj)) st hs
      exact ⟨st', runL_forM_of ha hn hf.1 h⟩
    · cases hf
  | _ => intro i st hf; simp [fits] at hf

/-- **A warp statement's launch runs the warp machine, block by block**:
    `grid` blocks of `nw` warps of the statement's kernel, warp `w` of block `b`
    standing as block `b·nw + w`, no block storing an element another touches,
    leave every element a warp's lanes may touch as the warp machine's run of
    the statement for that block does from the launch's memory. -/
theorem launch_lockstep (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (name : String) (nbuf smem : Nat)
    (s : EWStmt) (nw : Nat) (hnw : 0 < nw) (hnw31 : nw < 2 ^ 31) (grid : Nat) (hgrid : grid * nw ≤ 2 ^ 32)
    (hN : 32 * nw ≤ 2 ^ 32) (A : Nat → Acc) (rd : Buf → Nat → Option Nat)
    (hacc : ∀ cta, cta < grid * nw → AccOk cta (A cta) rd 0 s)
    (hA : ∀ cta l b j, A cta l b j → b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32)
    (hdisj : ∀ c c' l l' b j, c < grid * nw → c' < grid * nw → c ≠ c' → A c l b j →
      ¬ StoresL c' (imOf rd) l' 0 s b j)
    (hfine : Fine (emitGo (fnBody (params nbuf) (Body.body nbuf s nw)) ⟨(params nbuf).length, 0, [], [], [], none⟩).2)
    (bufs bufs' : Nat → ByteArray) (hrd : RdOk nbuf (kmem bufs) rd)
    (hRd : ∀ cta l b j, cta < grid * nw → A cta l b j → rd b j = none)
    (hfits : ∀ cta, cta < grid * nw → fits rd cta 0 s = true)
    (hin : ∀ cta l b j, cta < grid * nw → A cta l b j → 4 * j + 4 ≤ (bufs b).size)
    (hr : LaunchRuns (Body.kernel name nbuf smem s nw) grid (32 * nw) (bufArgs nbuf) bufs bufs' hw) :
    ∀ cta l b j, cta < grid * nw → A cta l b j →
      ({ kmem bufs with bufs := bufs' } : Mem).load (bufW nbuf b j) 4 =
        some ((@WStmt.run F32 (F32.ops hw) _ (s.elabAt cta 0 (fun _ _ => 0) (imOf rd)) (ws0 nbuf (kmem bufs))).mem b j).bits := by
  have hblk : ∀ bi, bi < grid → bi * nw + nw ≤ grid * nw := fun bi hb => by
    rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hb
  have hok : ∀ cta, cta < grid * nw → ∀ l : Lane, ∃ st', runL hw nbuf cta l.val (fun _ => 0) rd 0 s
      ⟨fun _ => ⟨0⟩, rm (ByteArray.mk (Array.replicate smem 0)) (kmem bufs), []⟩ = some st' := fun c hc l =>
    runL_total hw nbuf c (A c) rd l _ (hA c l) (fun b j h => hRd c l b j hc h) s 0 _ (hfits c hc) (hacc c hc)
      (fun b j h => hin c l b j hc h)
  intro cta l b j hc hAc
  obtain ⟨W, hfold, rfl⟩ := hr
  let run (c : Nat) : WSt F32 := @WStmt.run F32 (F32.ops hw) _ (s.elabAt c 0 (fun _ _ => 0) (imOf rd)) (ws0 nbuf (kmem bufs))
  have inv : ∀ c, c ≤ grid → ∀ Wc, BlocksRun (Body.kernel name nbuf smem s nw) (32 * nw) (bufArgs nbuf) hw c
      ({ mem := { arena := .empty, data := .empty, out := .empty, bufs, nan := hw.nan } } : World) Wc →
      MemSame (kmem bufs) Wc.mem ∧
      (∀ c' (l : Lane) b j, c' < c * nw → A c' l b j → Wc.mem.load (bufW nbuf b j) 4 = some ((run c').mem b j).bits) ∧
      (∀ b j, (b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32) →
        (∀ c' (l : Lane), c' < c * nw → ¬ StoresL c' (imOf rd) l 0 s b j) →
        Wc.mem.load (bufW nbuf b j) 4 = (kmem bufs).load (bufW nbuf b j) 4) ∧
      RdOk nbuf Wc.mem rd ∧ Wc.mem.nan = hw.nan := by
    intro c
    induction c with
    | zero =>
      intro _ Wc h
      cases h
      have hld : ∀ a n, ({ arena := .empty, data := .empty, out := .empty, bufs, nan := hw.nan } : Mem).load a n =
          (kmem bufs).load a n := Mem.load_congr (fun r => by cases r <;> rfl) rfl rfl
      refine ⟨⟨fun r => by cases r <;> rfl, rfl, rfl⟩, fun _ _ _ _ h => absurd h (by omega), fun _ _ _ _ => hld _ _,
        fun b j n h => ?_, rfl⟩
      obtain ⟨h1, h2⟩ := hrd b j n h
      exact ⟨h1, by simp only [rdM] at h2 ⊢; rw [hld]; exact h2⟩
    | succ c ih =>
      intro hc Wc h
      obtain ⟨Wp, hp, hblock⟩ := h
      have hcg : c < grid := by omega
      have hb := hblk c hcg
      obtain ⟨hI1, hI2, hI3, hI4, hI5⟩ := ih (by omega) Wp hp
      obtain ⟨h1, h2, h3, h4, h5⟩ := block_lockstep hw hw32 name nbuf smem s nw hnw hnw31 c (by omega) hN A rd
        (fun wi hwi => hacc _ (by omega)) hA
        (fun wi wi' l l' b j hwi hwi' hne hAl hst =>
          hdisj _ _ l l' b j (by omega) (by omega) (by omega) hAl hst)
        hfine Wp Wc hI4 hI5
        (fun t ht => by
          have ht' : t / 32 < nw := by omega
          obtain ⟨st0, h0⟩ := hok (c * nw + t / 32) (by omega) ⟨t % 32, Nat.mod_lt _ (by decide)⟩
          exact runL_transfer hw nbuf _ _ _ _ rd s 0 _ st0 _ h0 (memSame_rm hI1 _))
        (kmem bufs)
        (fun wi l b j hwi hAl => ⟨hI3 b j (hA _ l b j hAl) (fun c' l' hc' hst =>
            hdisj (c * nw + wi) c' l l' b j (by omega) (by omega) (by omega) hAl hst),
          (bufW_ok (hA _ l b j hAl) (hin _ l b j (by omega) hAl) 0).1⟩)
        hblock
      refine ⟨hI1.trans h3, fun c' l b j hc' hA' => ?_, fun b j hv hn => ?_, h4, h5⟩
      · by_cases hcc : c * nw ≤ c'
        · have hwi : c' - c * nw < nw := by rw [Nat.succ_mul] at hc'; omega
          have he : c * nw + (c' - c * nw) = c' := by omega
          have := h1 (c' - c * nw) l b j hwi (by rw [he]; exact hA')
          rw [he] at this; exact this
        · rw [h2 b j (hA c' l b j hA') (fun wi l'' hwi hst => hdisj c' _ l l'' b j (by omega) (by omega) (by omega) hA' hst),
            hI2 c' l b j (by omega) hA']
      · rw [h2 b j hv (fun wi l'' hwi => hn _ l'' (by rw [Nat.succ_mul]; omega)),
          hI3 b j hv (fun c' l' hc' => hn c' l' (by rw [Nat.succ_mul]; omega))]
  obtain ⟨_, hI2, _, _, _⟩ := inv grid (Nat.le_refl _) W hfold
  obtain ⟨hb1, hb2, hj1⟩ := hA cta l b j hAc
  exact (load_bufs (m₁ := { kmem bufs with bufs := W.mem.bufs }) (m₂ := W.mem) (decode_bufW hb1 hb2 hj1) rfl).trans
    (hI2 cta l b j hc hAc)

/-- `launch_lockstep` for one-warp blocks. -/
theorem launch_lockstep_one (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (name : String)
    (nbuf smem : Nat) (s : EWStmt) (grid : Nat) (hgrid : grid ≤ 2 ^ 32) (A : Nat → Acc) (rd : Buf → Nat → Option Nat)
    (hacc : ∀ cta, cta < grid → AccOk cta (A cta) rd 0 s)
    (hA : ∀ cta l b j, A cta l b j → b < nbuf ∧ b < 268435449 ∧ j < 2 ^ 32)
    (hdisj : ∀ c c' l l' b j, c < grid → c' < grid → c ≠ c' → A c l b j → ¬ StoresL c' (imOf rd) l' 0 s b j)
    (hfine : Fine (emitGo (fnBody (params nbuf) (Body.body nbuf s)) ⟨(params nbuf).length, 0, [], [], [], none⟩).2)
    (bufs bufs' : Nat → ByteArray) (hrd : RdOk nbuf (kmem bufs) rd)
    (hRd : ∀ cta l b j, cta < grid → A cta l b j → rd b j = none)
    (hfits : ∀ cta, cta < grid → fits rd cta 0 s = true)
    (hin : ∀ cta l b j, cta < grid → A cta l b j → 4 * j + 4 ≤ (bufs b).size)
    (hr : LaunchRuns (Body.kernel name nbuf smem s) grid 32 (bufArgs nbuf) bufs bufs' hw) :
    ∀ cta l b j, cta < grid → A cta l b j →
      ({ kmem bufs with bufs := bufs' } : Mem).load (bufW nbuf b j) 4 =
        some ((@WStmt.run F32 (F32.ops hw) _ (s.elabAt cta 0 (fun _ _ => 0) (imOf rd)) (ws0 nbuf (kmem bufs))).mem b j).bits := by
  have e : ∀ c, c < grid * 1 ↔ c < grid := fun c => by rw [Nat.mul_one]
  intro cta l b j hc hAc
  exact launch_lockstep hw hw32 name nbuf smem s 1 (by decide) (by decide) grid (by omega) (by decide) A rd
    (fun c hc => hacc c ((e c).mp hc)) hA (fun c c' l l' b j hc hc' => hdisj c c' l l' b j ((e c).mp hc) ((e c').mp hc'))
    hfine bufs bufs' hrd (fun c l b j hc => hRd c l b j ((e c).mp hc)) (fun c hc => hfits c ((e c).mp hc))
    (fun c l b j hc => hin c l b j ((e c).mp hc)) hr cta l b j ((e cta).mpr hc) hAc

end

end AlgorithmLib.ML.BodyWarp
