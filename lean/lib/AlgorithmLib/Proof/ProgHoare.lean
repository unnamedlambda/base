module
public import AlgorithmLib.Host.Hoare
public import AlgorithmLib.Host.Logic
public import AlgorithmLib.Surface.Prog
meta import AlgorithmLib.Host.Hoare
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Proof.ProgHoare` — the logic over generators

A generator writes a `Prog`; `emit` folds it into the `Code` that ships. This
module states triples about the `Prog` and carries them to that `Code`, so a
proof about a generator is a proof about what it ships and nothing is checked
by running the program at build time.

`PT cfg d J P p Q` is the triple. The emitter appends to what it has already
produced and keeps the open straight run open across a bind, so the triple is
stated as an extension: whatever code came before, if it ends in `P` with the
slot count the emitter expects, the code with `p` emitted after it ends in `Q`
of what `p` answers. `emitGo_bind` makes sequencing the emitter's own, and
`Hoare.merge_rule` makes a straight run that goes on across a bind no
different from two.

`emit_triple` is where it lands: a triple about a body is a triple about
`emit` of it.
-/

-- ---------------------------------------------------------------------------
-- Slot counting without fuel
--
-- `slotsOf` and `terminates` walk a code with a fixed fuel of 1000 and answer
-- short when it runs out; the emitter counts exactly. `slotsGo_eq` says the two
-- agree once the fuel covers `needS`, which is the one bound a proof here
-- carries.
-- ---------------------------------------------------------------------------

namespace AlgorithmLib.HProg.Sem

open AlgorithmLib.IR
open AlgorithmLib.HProg

mutual
def slotsS : Nat → List Piece → Nat
  | n, [] => n
  | n, p :: ps => slotsS (slotsP n p) ps
def slotsP : Nat → Piece → Nat
  | n, .straight ss => stmtsSlots n ss
  | n, .loop l pre body => slotsS (slotsS (n + l.pTys.length) pre + l.pTys.length) body + l.exitTys.length
  | n, .ite m thn els _ _ =>
      slotsS (slotsS n thn) els + (if termsS thn && termsS els then 0 else m.jTys.length)
  | n, .dloop l body => slotsS (n + l.pTys.length) body + l.exitTys.length
  | n, .br _ _ | n, .cont _ _ => n
def termsS : List Piece → Bool
  | [] => false
  | [p] => termsP p
  | _ :: p :: ps => termsS (p :: ps)
def termsP : Piece → Bool
  | .br _ _ | .cont _ _ => true
  | .ite _ thn els _ _ => termsS thn && termsS els
  | _ => false
end

/-- The fuel `slotsGo` and `termsGo` need to reach the end of a piece.
    Written with the recursor rather than by structural recursion: `Fine` is
    decided on a whole emitted code, and the kernel reduces `Piece.rec` in one
    pass where structural recursion over this nested type goes through
    `brecOn` and exhausts memory. -/
noncomputable def needP (p : Piece) : Nat :=
  Piece.rec (motive_1 := fun _ => Nat) (motive_2 := fun _ => Nat)
    (fun _ => 0) (fun _ _ _ a b => max a b) (fun _ _ _ _ _ a b => max a b) (fun _ _ a => a)
    (fun _ _ => 0) (fun _ _ => 0) 0 (fun _ _ a b => 1 + max a b) p

/-- The same, for a code. -/
noncomputable def needS (c : List Piece) : Nat :=
  Piece.rec_1 (motive_1 := fun _ => Nat) (motive_2 := fun _ => Nat)
    (fun _ => 0) (fun _ _ _ a b => max a b) (fun _ _ _ _ _ a b => max a b) (fun _ _ a => a)
    (fun _ _ => 0) (fun _ _ => 0) 0 (fun _ _ a b => 1 + max a b) c

theorem needS_nil : needS [] = 0 := rfl
theorem needS_cons (p : Piece) (ps : List Piece) : needS (p :: ps) = 1 + max (needP p) (needS ps) := rfl
theorem needP_straight (ss : List Stmt) : needP (.straight ss) = 0 := rfl
theorem needP_loop (l : Loop) (pre body : List Piece) :
    needP (.loop l pre body) = max (needS pre) (needS body) := rfl
theorem needP_ite (m : IteMeta) (thn els : List Piece) (tr er : List R) :
    needP (.ite m thn els tr er) = max (needS thn) (needS els) := rfl
theorem needP_dloop (l : DLoop) (body : List Piece) : needP (.dloop l body) = needS body := rfl
theorem needP_br (d : Nat) (a : List R) : needP (.br d a) = 0 := rfl
theorem needP_cont (d : Nat) (a : List R) : needP (.cont d a) = 0 := rfl

theorem slotsGo_eq : ∀ g c n, needS c ≤ g → slotsGo g n c = slotsS n c ∧ termsGo g c = termsS c := by
  intro g
  induction g with
  | zero =>
      intro c n h
      cases c with
      | nil => simp [slotsGo, termsGo, slotsS, termsS]
      | cons p ps => simp [needS_cons] at h
  | succ g ih =>
      intro c n h
      cases c with
      | nil => simp [slotsGo, termsGo, slotsS, termsS]
      | cons p ps =>
          simp only [needS_cons] at h
          have hp : needP p ≤ g := by omega
          have hps : needS ps ≤ g := by omega
          have ihps := fun n => ih ps n hps
          constructor
          · cases p with
            | straight ss => simp only [slotsGo, slotsS, slotsP]; exact (ihps _).1
            | loop l pre body =>
                simp only [needP_straight, needP_loop, needP_ite, needP_dloop, needP_br, needP_cont] at hp
                simp only [slotsGo, slotsS, slotsP]
                rw [(ih pre _ (by omega)).1, (ih body _ (by omega)).1]; exact (ihps _).1
            | ite m thn els _ _ =>
                simp only [needP_straight, needP_loop, needP_ite, needP_dloop, needP_br, needP_cont] at hp
                simp only [slotsGo, slotsS, slotsP]
                rw [(ih thn _ (by omega)).1, (ih els _ (by omega)).1, (ih thn 0 (by omega)).2,
                  (ih els 0 (by omega)).2]; exact (ihps _).1
            | dloop l body =>
                simp only [needP_straight, needP_loop, needP_ite, needP_dloop, needP_br, needP_cont] at hp
                simp only [slotsGo, slotsS, slotsP]
                rw [(ih body _ (by omega)).1]; exact (ihps _).1
            | br _ _ => simp only [slotsGo, slotsS, slotsP]; exact (ihps _).1
            | cont _ _ => simp only [slotsGo, slotsS, slotsP]; exact (ihps _).1
          · cases ps with
            | nil =>
                cases p with
                | ite m thn els _ _ =>
                    simp only [needP_straight, needP_loop, needP_ite, needP_dloop, needP_br, needP_cont] at hp
                    simp only [termsGo, termsS, termsP]
                    rw [(ih thn 0 (by omega)).2, (ih els 0 (by omega)).2]
                | _ => simp [termsGo, termsS, termsP]
            | cons q qs =>
                have := (ihps 0).2
                cases p <;> simp only [termsGo, termsS] <;> exact this

theorem slotsS_append (F G : List Piece) : ∀ n, slotsS n (F ++ G) = slotsS (slotsS n F) G := by
  induction F with
  | nil => intro n; simp [slotsS]
  | cons p ps ih => intro n; simp [slotsS, ih]

theorem needS_append_left (F G : List Piece) : needS F ≤ needS (F ++ G) := by
  induction F with
  | nil => simp [needS_nil]
  | cons p ps ih => simp only [List.cons_append, needS_cons]; omega

theorem needS_append_right (F G : List Piece) : needS G ≤ needS (F ++ G) := by
  induction F with
  | nil => simp
  | cons p ps ih => simp only [List.cons_append, needS_cons]; omega

theorem needS_append_mono (F : List Piece) {G G' : List Piece} (h : needS G ≤ needS G') :
    needS (F ++ G) ≤ needS (F ++ G') := by
  induction F with
  | nil => simpa using h
  | cons p ps ih => simp only [List.cons_append, needS_cons]; omega

theorem needP_lt (F : List Piece) (p : Piece) : needP p < needS (F ++ [p]) := by
  have := needS_append_right F [p]; simp only [needS_cons, needS_nil] at this; omega

theorem stmtsSlots_append (x y : List Stmt) (n : Nat) :
    stmtsSlots n (x ++ y) = stmtsSlots (stmtsSlots n x) y := by
  simp [stmtsSlots, List.foldl_append]

end AlgorithmLib.HProg.Sem

namespace AlgorithmLib.Prog

open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare
/-- Run `g` on what a fold answered, unless it stopped. -/
def seqE {α β : Type} (r : Option α × St) (g : α → St → Option β × St) : Option β × St :=
  match r with
  | (some a, s) => g a s
  | (none, s) => (none, s)

theorem emitCall_seq {α β} (f : Ffi) (args : Vals Slot f.params)
    (k : ResV Slot f.result → St → Option α × St) (g : α → St → Option β × St) (s : St) :
    emitCall f args (fun r s => seqE (k r s) g) s = seqE (emitCall f args k s) g := by
  unfold emitCall; split <;> rfl

theorem emitCallLocal_seq {α β ps res} (r : LocalRef ps res) (args : Vals Slot ps)
    (k : ResV Slot res → St → Option α × St) (g : α → St → Option β × St) (s : St) :
    emitCallLocal r args (fun r s => seqE (k r s) g) s = seqE (emitCallLocal r args k s) g := by
  unfold emitCallLocal; dsimp only; (repeat' split) <;> rfl

theorem emitLoop_seq {tys exitTys β γ α} (init : Vals Slot tys)
    (head : Lvl exitTys tys → Vals Slot tys → St → Option (Cond Slot × Vals Slot exitTys × γ) × St)
    (body : Lvl exitTys tys → Vals Slot tys → γ → St → Option (Vals Slot tys) × St)
    (k : Vals Slot exitTys → St → Option α × St) (g : α → St → Option β × St) (s : St) :
    emitLoop init head body (fun v s => seqE (k v s) g) s = seqE (emitLoop init head body k s) g := by
  unfold emitLoop; dsimp only; split <;> rfl

theorem emitDloop_seq {tys α β tb} (init : Vals Slot tys) (cc : ICmpCond) (cb : Slot tb)
    (guardIdx : Option Nat) (contOnTrue : Bool) (exitIdx : List Nat)
    (body : Lvl (idxTys tys exitIdx) tys → Vals Slot tys → St → Option (Slot tb × Vals Slot tys) × St)
    (k : Vals Slot (idxTys tys exitIdx) → St → Option α × St) (g : α → St → Option β × St) (s : St) :
    emitDloop init cc cb guardIdx contOnTrue exitIdx body (fun v s => seqE (k v s) g) s =
      seqE (emitDloop init cc cb guardIdx contOnTrue exitIdx body k s) g := rfl

theorem emitIte_seq {jTys α β} (c : Cond Slot) (thn els : St → Option (Vals Slot jTys) × St)
    (k : Vals Slot jTys → St → Option α × St) (g : α → St → Option β × St) (s : St) :
    emitIte c thn els (fun v s => seqE (k v s) g) s = seqE (emitIte c thn els k s) g := rfl

/-- **Emitting `p >>= f` is emitting `p`, then `f` of what it answered.** -/
theorem emitGo_bind {α β} (p : Prog Slot Lvl α) (f : α → Prog Slot Lvl β) (s : St) :
    emitGo (p >>= f) s = seqE (emitGo p s) (fun a => emitGo (f a)) := by
  show emitGo (Prog.bind p f) s = _
  induction p generalizing s with
  | ret a => rfl
  | op o k ih => rw [Prog.bind, emitGo_op, emitGo_op]; exact ih _ f _
  | store v a k ih => rw [Prog.bind, emitGo_store, emitGo_store]; exact ih f _
  | storeUnaligned v a k ih => rw [Prog.bind, emitGo_storeUnaligned, emitGo_storeUnaligned]; exact ih f _
  | istore8 v a h k ih => rw [Prog.bind, emitGo_istore8, emitGo_istore8]; exact ih f _
  | call fn args k ih =>
      rw [Prog.bind, emitGo_call, emitGo_call, ← emitCall_seq]; congr; funext r s; exact ih r f s
  | callLocal r args k ih =>
      rw [Prog.bind, emitGo_callLocal, emitGo_callLocal, ← emitCallLocal_seq]; congr; funext r s; exact ih r f s
  | loop init head body k _ _ ih =>
      rw [Prog.bind, emitGo_loop, emitGo_loop, ← emitLoop_seq]; congr; funext r s; exact ih r f s
  | dloop init cc cb g hg c e body k _ ih =>
      rw [Prog.bind, emitGo_dloop, emitGo_dloop, ← emitDloop_seq]; congr; funext r s; exact ih r f s
  | ite c thn els k _ _ ih =>
      rw [Prog.bind, emitGo_ite, emitGo_ite, ← emitIte_seq]; congr; funext r s; exact ih r f s
  | params tys k ih => rw [Prog.bind, emitGo_params, emitGo_params]; exact ih _ f _
  | br l args => rw [Prog.bind, emitGo_br, emitGo_br]; rfl
  | cont l args => rw [Prog.bind, emitGo_cont, emitGo_cont]; rfl




/-- What the emitter has produced so far, with the open run closed. -/
def St.out (s : St) : Code :=
  s.pieces.reverse ++ match s.cur with
    | [] => []
    | _ :: _ => [.straight s.cur.reverse]

theorem St.out_flush (s : St) : s.flush.out = s.out := by
  unfold St.flush St.out
  cases h : s.cur <;> simp [h]

theorem St.out_eq_flush (s : St) : s.flush.pieces.reverse = s.out := by
  rw [← St.out_flush]; unfold St.flush St.out; split <;> simp_all

/-- A statement the emitter adds runs after what it had produced. -/
theorem St.out_stmt {cfg : Cfg} {P0 R R' : Env → World → Prop} {J : Post} {s : St} {st : Stmt}
    (h : Triple cfg P0 s.out { J with ok := R }) (hs : Stmt1 cfg R st R' J.faultOk) :
    Triple cfg P0 (s.stmt st).out { J with ok := R' } := by
  have hs' : Stmts cfg R [st] R' J.faultOk := stmts_cons hs (stmts_nil (cfg := cfg) fun _ _ h => h)
  unfold St.out St.stmt at *
  cases hc : s.cur with
  | nil =>
      simp only [hc, List.append_nil] at h ⊢
      simp only [List.reverse_cons, List.reverse_nil, List.nil_append]
      exact append_rule h (cons_rule (straight_rule hs') (nil_rule fun _ _ h => h))
  | cons c cs =>
      simp only [hc] at h ⊢
      simp only [List.reverse_cons]
      rw [List.reverse_cons] at h
      exact merge_rule h hs'

/-- A piece the emitter adds after closing the open run. -/
theorem St.out_piece {cfg : Cfg} {P0 R : Env → World → Prop} {J Q : Post} {s : St} {p : Piece}
    (h : Triple cfg P0 s.out { J with ok := R }) (hp : PieceT cfg R p Q)
    (hb : ∀ d Γ vs w, J.brk d Γ vs w → Q.brk d Γ vs w) (hc : ∀ d vs w, J.cont d vs w → Q.cont d vs w)
    (hfo : J.faultOk = true → Q.faultOk = true := by intro h; first | exact h | rfl | simp_all) :
    Triple cfg P0 { s.flush with pieces := p :: s.flush.pieces }.out Q := by
  have e : ({ s.flush with pieces := p :: s.flush.pieces } : St).out = s.out ++ [p] := by
    rw [← St.out_flush]
    unfold St.flush St.out
    cases hc : s.cur <;> simp [hc]
  rw [e]
  refine append_rule (conseq h (fun _ _ h => h) (fun _ _ h => h) hb hc hfo) ?_
  exact cons_rule hp (nil_rule fun _ _ h => h)

theorem St.flush_depth (s : St) : s.flush.depth = s.depth := by
  unfold St.flush; split <;> rfl

theorem St.leave_depth (o i : St) : (St.leave o i).2.depth = i.depth := St.flush_depth i

theorem emitGo_depth {α : Type} (p : Prog Slot Lvl α) : ∀ s, (emitGo p s).2.depth = s.depth := by
  induction p with
  | ret a => intro s; rfl
  | op o k ih => intro s; rw [emitGo_op]; exact ih _ _
  | store v a k ih => intro s; rw [emitGo_store]; exact ih _
  | storeUnaligned v a k ih => intro s; rw [emitGo_storeUnaligned]; exact ih _
  | istore8 v a h k ih => intro s; rw [emitGo_istore8]; exact ih _
  | call f args k ih => intro s; rw [emitGo_call]; unfold emitCall; split <;> exact ih _ _
  | callLocal r args k ih =>
      intro s; rw [emitGo_callLocal]; unfold emitCallLocal; dsimp only
      split <;> split <;> (rw [ih]; rfl)
  | loop init head body k _ _ ih =>
      intro s; rw [emitGo_loop]; unfold emitLoop; dsimp only
      split
      · simp only [St.leave, St.note]; split <;> exact St.flush_depth s
      · rw [ih]; exact St.flush_depth s
  | dloop init cc cb g hg c e body k _ ih =>
      intro s; rw [emitGo_dloop]; unfold emitDloop; dsimp only
      rw [ih]; cases g <;> simp [St.flush_depth, St.bind1]
  | ite c thn els k ihT ihE ih =>
      intro s; rw [emitGo_ite]; unfold emitIte; dsimp only
      rw [ih, St.leave_depth, ihE, St.enter, St.leave_depth, ihT]
      simp [St.enter, St.flush_depth, St.bind1]
  | params tys k ih => intro s; rw [emitGo_params]; exact ih _ _
  | br l args => intro s; rw [emitGo_br]; exact St.flush_depth s
  | cont l args => intro s; rw [emitGo_cont]; exact St.flush_depth s

-- ---------------------------------------------------------------------------
-- What emission keeps
-- ---------------------------------------------------------------------------
theorem St.flush_cur (s : St) : s.flush.cur = [] := by
  unfold St.flush; split <;> simp_all

theorem St.out_stmt_slots (s : St) (st : Stmt) (n : Nat) :
    slotsS n (s.stmt st).out = stmtsSlots (slotsS n s.out) [st] := by
  unfold St.out St.stmt
  cases hc : s.cur with
  | nil => simp [slotsS_append, slotsS, slotsP]
  | cons c cs =>
      simp only [List.reverse_cons, slotsS_append, slotsS, slotsP, List.append_assoc,
        stmtsSlots_append]

theorem St.out_stmt_need (s : St) (st : Stmt) : needS s.out ≤ needS (s.stmt st).out := by
  unfold St.out St.stmt
  cases hc : s.cur with
  | nil => simpa using needS_append_left s.pieces.reverse [.straight [st]]
  | cons c cs =>
      simp only
      exact needS_append_mono _ (by simp [needS_cons, needS_nil, needP_straight])

/-- The code a state has produced, once a piece is added after closing the run. -/
theorem St.out_push (s : St) (p : Piece) (x : St) (hp : x.pieces = p :: s.flush.pieces)
    (hc : x.cur = []) : x.out = s.out ++ [p] := by
  rw [← St.out_eq_flush s]; unfold St.out; rw [hp, hc]; simp

theorem St.flush_err (s : St) : s.flush.err = s.err := by unfold St.flush; split <;> rfl
theorem St.flush_n (s : St) : s.flush.n = s.n := by unfold St.flush; split <;> rfl

@[simp] theorem St.leave_fst (o i : St) : (St.leave o i).1 = i.out := St.out_eq_flush i
@[simp] theorem St.leave_pieces (o i : St) : (St.leave o i).2.pieces = o.pieces := rfl
@[simp] theorem St.leave_cur (o i : St) : (St.leave o i).2.cur = o.cur := rfl
@[simp] theorem St.leave_n (o i : St) : (St.leave o i).2.n = i.n := St.flush_n i
@[simp] theorem St.leave_err (o i : St) : (St.leave o i).2.err = i.err := St.flush_err i
@[simp] theorem St.enter_out (s : St) : s.enter.out = [] := rfl
@[simp] theorem St.enter_n (s : St) : s.enter.n = s.n := rfl
@[simp] theorem St.enter_err (s : St) : s.enter.err = s.err := rfl
@[simp] theorem St.flush_pieces_out (s : St) : s.flush.out = s.out := St.out_flush s

theorem St.out_congr {x y : St} (hp : x.pieces = y.pieces) (hc : x.cur = y.cur) : x.out = y.out := by
  unfold St.out; rw [hp, hc]

theorem St.note_err (s : St) (m : String) : (s.note m).err ≠ none := by
  unfold St.note; split
  · rename_i h; intro h'; rw [h'] at h; cases h
  · simp

theorem St.note_out (s : St) (m : String) : (s.note m).out = s.out := by
  unfold St.note; split <;> rfl

/-- What an emission step keeps: a failure it records stays recorded, the code
    only grows, and --- when nothing failed and the code fits the fuel the model
    counts with --- the emitter's slot counter is the model's count. -/
def Cnt (s t : St) : Prop :=
  (t.err = none → s.err = none) ∧ needS s.out ≤ needS t.out ∧
  (t.err = none → needS t.out ≤ HProg.fuel → ∀ n0, slotsS n0 s.out = s.n → slotsS n0 t.out = t.n)

theorem Cnt.refl (s : St) : Cnt s s := ⟨id, Nat.le_refl _, fun _ _ _ h => h⟩

theorem Cnt.trans {s t u : St} (h1 : Cnt s t) (h2 : Cnt t u) : Cnt s u :=
  ⟨fun h => h1.1 (h2.1 h), Nat.le_trans h1.2.1 h2.2.1,
   fun he hn n0 h => h2.2.2 he hn n0 (h1.2.2 (h2.1 he) (Nat.le_trans h2.2.1 hn) n0 h)⟩

theorem Cnt.bind1 (s : St) (st : Stmt) (hb : stmtsSlots 0 [st] = 1) : Cnt s (s.bind1 st).2 := by
  refine ⟨id, St.out_stmt_need s st, fun _ _ n0 h => ?_⟩
  show slotsS n0 (s.stmt st).out = s.n + 1
  rw [St.out_stmt_slots, h]
  cases st <;> simp_all [stmtsSlots]

theorem Cnt.stmt (s : St) (st : Stmt) (hb : stmtsSlots 0 [st] = 0) : Cnt s (s.stmt st) := by
  refine ⟨id, St.out_stmt_need s st, fun _ _ n0 h => ?_⟩
  rw [St.out_stmt_slots, h]
  cases st <;> simp_all [stmtsSlots, St.stmt]

theorem Cnt.flush (s : St) : Cnt s s.flush :=
  ⟨fun h => by rwa [St.flush_err] at h, by simp, fun _ _ n0 h => by simp [St.flush_n, h]⟩

theorem St.out_push' {x s : St} {p : Piece} (hp : x.pieces = p :: s.pieces) (hcx : x.cur = [])
    (hcs : s.cur = []) : x.out = s.out ++ [p] := by
  unfold St.out; rw [hp, hcx, hcs]; simp

/-- A step that closes the run and adds one piece. -/
theorem Cnt.push {s S1 X : St} {p : Piece} (e1 : Cnt s S1) (hc1 : S1.cur = [])
    (hp : X.pieces = p :: S1.pieces) (hc : X.cur = [])
    (herr : X.err = none → S1.err = none)
    (hn : X.err = none → needS (S1.out ++ [p]) ≤ HProg.fuel → slotsP S1.n p = X.n) : Cnt s X := by
  have hout : X.out = S1.out ++ [p] := St.out_push' hp hc hc1
  refine ⟨fun h => e1.1 (herr h), ?_, ?_⟩
  · rw [hout]; exact Nat.le_trans e1.2.1 (needS_append_left _ _)
  · intro he hb n0 hs
    rw [hout] at hb ⊢
    have c1 := e1.2.2 (herr he) (Nat.le_trans (needS_append_left _ _) hb) n0 hs
    rw [slotsS_append, c1]
    simp only [slotsS]
    exact hn he hb

theorem emitLoop_cnt {tys exitTys : List ClifTy} {β α : Type} (init : Vals Slot tys)
    (head : Lvl exitTys tys → Vals Slot tys → St → Option (Cond Slot × Vals Slot exitTys × β) × St)
    (body : Lvl exitTys tys → Vals Slot tys → β → St → Option (Vals Slot tys) × St)
    (k : Vals Slot exitTys → St → Option α × St)
    (hH : ∀ l cs s, Cnt s (head l cs s).2) (hB : ∀ l cs x s, Cnt s (body l cs x s).2)
    (hK : ∀ v s, Cnt s (k v s).2) (s : St) : Cnt s (emitLoop init head body k s).2 := by
  unfold emitLoop
  have e1 := Cnt.flush s
  have hc1 := St.flush_cur s
  generalize s.flush = S1 at e1 hc1 ⊢
  obtain ⟨hd, sH, hHe⟩ : ∃ hd sH, head S1.depth (carriesFrom S1.n tys)
      ({ S1 with n := S1.n + tys.length, depth := S1.depth + 1 } : St).enter = (hd, sH) :=
    ⟨_, _, rfl⟩
  have iH := hH S1.depth (carriesFrom S1.n tys)
    ({ S1 with n := S1.n + tys.length, depth := S1.depth + 1 } : St).enter
  rw [hHe] at iH
  simp only [hHe]
  cases hd with
  | none =>
      refine ⟨fun h => absurd h (St.note_err _ _), ?_, fun h => absurd h (St.note_err _ _)⟩
      rw [St.note_out]
      refine Nat.le_trans e1.2.1 (Nat.le_of_eq (congrArg needS (St.out_congr rfl rfl)))
  | some v =>
      obtain ⟨c, exitR, x⟩ := v
      dsimp only
      have iF := Cnt.bind1 sH (.op (.icmp c.cc c.a c.b)) rfl
      generalize hF : (sH.bind1 (.op (.icmp c.cc c.a c.b))) = rF at iF ⊢
      obtain ⟨flag, sH'⟩ := rF
      dsimp only at iF ⊢
      have hn' : sH'.n = sH.n + 1 := by rw [← congrArg (·.2.n) hF]; rfl
      obtain ⟨bd, sB, hBe⟩ : ∃ bd sB, body S1.depth (carriesFrom sH'.n tys) x
          ({ (St.leave ({ S1 with n := S1.n + tys.length, depth := S1.depth + 1 } : St) sH').2 with
              n := sH'.n + tys.length } : St).enter = (bd, sB) := ⟨_, _, rfl⟩
      have iB := hB S1.depth (carriesFrom sH'.n tys) x
          ({ (St.leave ({ S1 with n := S1.n + tys.length, depth := S1.depth + 1 } : St) sH').2 with
              n := sH'.n + tys.length } : St).enter
      rw [hBe] at iB
      simp only [St.leave_n, hBe]
      refine Cnt.trans ?_ (hK _ _)
      refine Cnt.push e1 hc1 rfl hc1 ?_ ?_
      · intro h
        have h1 : sB.err = none := by simpa using h
        have h2 : sH'.err = none := by simpa using iB.1 h1
        simpa using iH.1 (iF.1 h2)
      · intro he hn
        have hlt := Nat.lt_of_lt_of_le (needP_lt S1.out _) hn
        simp only [needP_straight, needP_loop, needP_ite, needP_dloop, needP_br, needP_cont] at hlt
        have eB : sB.err = none := by simpa using he
        have eH' : sH'.err = none := by simpa using iB.1 eB
        have eH : sH.err = none := iF.1 eH'
        have hpre : needS sH'.out ≤ HProg.fuel := by simp at hlt; omega
        have hbod : needS sB.out ≤ HProg.fuel := by simp at hlt; omega
        have cH := iH.2.2 eH (Nat.le_trans iF.2.1 hpre) _ rfl
        have cF := iF.2.2 eH' hpre _ cH
        have cB := iB.2.2 eB hbod _ rfl
        simp only [slotsP]
        simp at cH cF cB ⊢
        rw [hn'] at cF cB; rw [cF, cB]

theorem terminates_eq {c : Code} (h : needS c ≤ HProg.fuel) : terminates c = termsS c :=
  (slotsGo_eq HProg.fuel c 0 h).2

theorem Cnt.jump {α : Type} (piece : Nat → Piece) (hp : ∀ n d, slotsP n (piece d) = n) (s : St) :
    Cnt s (emitJump (α := α) piece s).2 := by
  unfold emitJump
  exact Cnt.push (Cnt.flush s) (St.flush_cur s) rfl (St.flush_cur s) id
    (fun _ _ => by rw [hp])

theorem emitCall_cnt {α : Type} (f : Ffi) (args : Vals Slot f.params)
    (k : ResV Slot f.result → St → Option α × St) (hK : ∀ v s, Cnt s (k v s).2) (s : St) :
    Cnt s (emitCall f args k s).2 := by
  unfold emitCall; dsimp only
  split
  · exact (Cnt.bind1 s _ rfl).trans (hK _ _)
  · exact (Cnt.stmt s _ rfl).trans (hK _ _)

theorem emitCallLocal_cnt {α : Type} {ps : List ClifTy} {res : Option ClifTy} (r : LocalRef ps res)
    (args : Vals Slot ps) (k : ResV Slot res → St → Option α × St) (hK : ∀ v s, Cnt s (k v s).2)
    (s : St) : Cnt s (emitCallLocal r args k s).2 := by
  unfold emitCallLocal; dsimp only
  have hu : ∀ i, Cnt s (s.useLocal i ps res) := fun _ => ⟨id, Nat.le_refl _, fun _ _ _ h => h⟩
  split <;> split
  all_goals first
    | exact ((hu _).trans (Cnt.bind1 _ _ rfl)).trans (hK _ _)
    | exact ((hu _).trans (Cnt.stmt _ _ rfl)).trans (hK _ _)
    | exact (Cnt.bind1 _ _ rfl).trans (hK _ _)
    | exact (Cnt.stmt _ _ rfl).trans (hK _ _)

theorem emitIte_cnt {jTys : List ClifTy} {α : Type} (c : Cond Slot)
    (thn els : St → Option (Vals Slot jTys) × St) (k : Vals Slot jTys → St → Option α × St)
    (hT : ∀ s, Cnt s (thn s).2) (hE : ∀ s, Cnt s (els s).2) (hK : ∀ v s, Cnt s (k v s).2)
    (s : St) : Cnt s (emitIte c thn els k s).2 := by
  unfold emitIte
  have e0 := Cnt.bind1 s (.op (.icmp c.cc c.a c.b)) rfl
  generalize s.bind1 (.op (.icmp c.cc c.a c.b)) = rF at e0 ⊢
  obtain ⟨flag, s0⟩ := rF
  dsimp only at e0 ⊢
  have e1 := e0.trans (Cnt.flush s0)
  have hc1 := St.flush_cur s0
  generalize s0.flush = S1 at e1 hc1 ⊢
  obtain ⟨tR, sT, hTe⟩ : ∃ a b, thn S1.enter = (a, b) := ⟨_, _, rfl⟩
  have iT := hT S1.enter; rw [hTe] at iT
  simp only [hTe]
  obtain ⟨eR, sE, hEe⟩ : ∃ a b, els (St.leave S1 sT).2.enter = (a, b) := ⟨_, _, rfl⟩
  have iE := hE (St.leave S1 sT).2.enter; rw [hEe] at iE
  simp only [hEe]
  refine Cnt.trans ?_ (hK _ _)
  refine Cnt.push e1 hc1 rfl hc1 ?_ ?_
  · intro h
    have h1 : sE.err = none := by simpa using h
    have h2 : sT.err = none := by simpa using iE.1 h1
    simpa using iT.1 h2
  · intro he hn
    have hlt := Nat.lt_of_lt_of_le (needP_lt S1.out _) hn
    simp only [needP_straight, needP_loop, needP_ite, needP_dloop, needP_br, needP_cont] at hlt
    have eE : sE.err = none := by simpa using he
    have eT : sT.err = none := by simpa using iE.1 eE
    have nT : needS sT.out ≤ HProg.fuel := by simp at hlt; omega
    have nE : needS sE.out ≤ HProg.fuel := by simp at hlt; omega
    have cT := iT.2.2 eT nT _ rfl
    have cE := iE.2.2 eE nE _ rfl
    simp only [slotsP]
    simp only [St.leave_fst, St.enter_n, St.leave_n, St.enter_out] at cT cE ⊢
    rw [terminates_eq nT, terminates_eq nE, cT, cE]
    cases termsS sT.out <;> cases termsS sE.out <;> simp

theorem emitDloop_cnt {tys : List ClifTy} {α : Type} {tb : ClifTy} (init : Vals Slot tys) (cc : ICmpCond)
    (cb : Slot tb) (guardIdx : Option Nat) (contOnTrue : Bool) (exitIdx : List Nat)
    (body : Lvl (idxTys tys exitIdx) tys → Vals Slot tys → St → Option (Slot tb × Vals Slot tys) × St)
    (k : Vals Slot (idxTys tys exitIdx) → St → Option α × St)
    (hB : ∀ l cs s, Cnt s (body l cs s).2) (hK : ∀ v s, Cnt s (k v s).2) (s : St) :
    Cnt s (emitDloop init cc cb guardIdx contOnTrue exitIdx body k s).2 := by
  unfold emitDloop
  cases guardIdx
  case' none =>
    dsimp only
    have e1 := Cnt.flush s
    have hc1 := St.flush_cur s
    generalize s.flush = S1 at e1 hc1 ⊢
  case' some gi =>
    dsimp only
    have e1 := (Cnt.bind1 s (.op (.icmp cc ((init.slots[gi]?).getD 0) cb)) rfl).trans (Cnt.flush _)
    have hc1 := St.flush_cur (s.bind1 (.op (.icmp cc ((init.slots[gi]?).getD 0) cb))).2
    generalize (s.bind1 (.op (.icmp cc ((init.slots[gi]?).getD 0) cb))).2.flush = S1 at e1 hc1 ⊢
  all_goals
    obtain ⟨bd, sB, hBe⟩ : ∃ a b, body S1.depth (carriesFrom S1.n tys)
        ({ S1 with n := S1.n + tys.length, depth := S1.depth + 1 } : St).enter = (a, b) := ⟨_, _, rfl⟩
    have iB := hB S1.depth (carriesFrom S1.n tys)
        ({ S1 with n := S1.n + tys.length, depth := S1.depth + 1 } : St).enter
    rw [hBe] at iB
    dsimp only at iB
    simp only [hBe]
    cases bd with
    | none =>
        dsimp only
        refine Cnt.trans ?_ (hK _ _)
        refine Cnt.push e1 hc1 rfl hc1 ?_ ?_
        · intro h; have h1 : sB.err = none := by simpa using h
          simpa using iB.1 h1
        · intro he hn
          have hlt := Nat.lt_of_lt_of_le (needP_lt S1.out _) hn
          simp only [needP_straight, needP_loop, needP_ite, needP_dloop, needP_br, needP_cont] at hlt
          have eB : sB.err = none := by simpa using he
          have cB := iB.2.2 eB (by simp at hlt; omega) _ rfl
          simp only [slotsP]
          simp only [St.leave_fst, St.enter_n, St.leave_n] at cB ⊢
          rw [cB]
    | some v =>
        obtain ⟨ca, vs⟩ := v
        dsimp only
        split
        · refine Cnt.trans ?_ (hK _ _)
          refine Cnt.push e1 hc1 rfl hc1 ?_ ?_
          · intro h; have h1 : sB.err = none := by simpa using h
            simpa using iB.1 h1
          · intro he hn
            have hlt := Nat.lt_of_lt_of_le (needP_lt S1.out _) hn
            simp only [needP_straight, needP_loop, needP_ite, needP_dloop, needP_br, needP_cont] at hlt
            have eB : sB.err = none := by simpa using he
            have cB := iB.2.2 eB (by simp at hlt; omega) _ rfl
            simp only [slotsP]
            simp only [St.leave_fst, St.enter_n, St.leave_n] at cB ⊢
            rw [cB]
        · have iF := Cnt.bind1 sB (.op (.icmp cc ca cb)) rfl
          generalize sB.bind1 (.op (.icmp cc ca cb)) = rF at iF ⊢
          obtain ⟨fl, sB'⟩ := rF
          dsimp only at iF ⊢
          refine Cnt.trans ?_ (hK _ _)
          refine Cnt.push e1 hc1 rfl hc1 ?_ ?_
          · intro h; have h1 : sB'.err = none := by simpa using h
            simpa using iB.1 (iF.1 h1)
          · intro he hn
            have hlt := Nat.lt_of_lt_of_le (needP_lt S1.out _) hn
            simp only [needP_straight, needP_loop, needP_ite, needP_dloop, needP_br, needP_cont] at hlt
            have eB' : sB'.err = none := by simpa using he
            have nB' : needS sB'.out ≤ HProg.fuel := by simp at hlt; omega
            have cB := iB.2.2 (iF.1 eB') (Nat.le_trans iF.2.1 nB') _ rfl
            have cF := iF.2.2 eB' nB' _ cB
            simp only [slotsP]
            simp only [St.leave_fst, St.enter_n, St.leave_n] at cF ⊢
            rw [cF]

/-- **The emitter counts slots as the model does**, and never loses a failure or
    shrinks what it has produced. -/
theorem emitGo_cnt {α : Type} (p : Prog Slot Lvl α) : ∀ s, Cnt s (emitGo p s).2 := by
  induction p with
  | ret a => intro s; exact Cnt.refl s
  | op o k ih => intro s; rw [emitGo_op]; exact (Cnt.bind1 s _ rfl).trans (ih _ _)
  | store v a k ih => intro s; rw [emitGo_store]; exact (Cnt.stmt s _ rfl).trans (ih _)
  | storeUnaligned v a k ih => intro s; rw [emitGo_storeUnaligned]; exact (Cnt.stmt s _ rfl).trans (ih _)
  | istore8 v a h k ih => intro s; rw [emitGo_istore8]; exact (Cnt.stmt s _ rfl).trans (ih _)
  | call f args k ih => intro s; rw [emitGo_call]; exact emitCall_cnt f args _ (fun v s => ih v s) s
  | callLocal r args k ih =>
      intro s; rw [emitGo_callLocal]; exact emitCallLocal_cnt r args _ (fun v s => ih v s) s
  | loop init head body k ihH ihB ih =>
      intro s; rw [emitGo_loop]
      exact emitLoop_cnt init _ _ _ (fun l cs s => ihH l cs s) (fun l cs x s => ihB l cs x s)
        (fun v s => ih v s) s
  | dloop init cc cb g hg c e body k ihB ih =>
      intro s; rw [emitGo_dloop]
      exact emitDloop_cnt init cc cb g c e _ _ (fun l cs s => ihB l cs s) (fun v s => ih v s) s
  | ite c thn els k ihT ihE ih =>
      intro s; rw [emitGo_ite]
      exact emitIte_cnt c _ _ _ (fun s => ihT s) (fun s => ihE s) (fun v s => ih v s) s
  | params tys k ih => intro s; rw [emitGo_params]; exact ih _ _
  | br l args => intro s; rw [emitGo_br]; exact Cnt.jump _ (fun _ _ => rfl) s
  | cont l args => intro s; rw [emitGo_cont]; exact Cnt.jump _ (fun _ _ => rfl) s

/-- An emission the logic speaks about: nothing failed, and the code fits the
    fuel the model counts slots with. `emitChecked` refuses the first; the
    second is one comparison on the finished code. -/
def Fine (t : St) : Prop := t.err = none ∧ needS t.out ≤ HProg.fuel

theorem Cnt.fine {s t : St} (h : Cnt s t) (ht : Fine t) : Fine s :=
  ⟨h.1 ht.1, Nat.le_trans h.2.1 ht.2⟩

/-- **The generator-level triple.** Whatever code the emitter had produced, if it
    ends in `P` with the slot count the emitter expects, the code with `p`
    emitted after it ends normally only where `p` answered, in `Q` of the
    answer. Leaving an enclosing loop is what `J` allows at depth `d`. It is a
    claim about emissions that are `Fine`. -/
def PT {α : Type} (cfg : Cfg) (d : Nat) (J : Post) (P : Env → World → Prop)
    (p : Prog Slot Lvl α) (Q : α → Env → World → Prop) : Prop :=
  ∀ s P0, s.depth = d → Fine (emitGo p s).2 →
    Triple cfg P0 s.out { J with ok := fun Γ w => Γ.size = s.n ∧ P Γ w } →
    Triple cfg P0 (emitGo p s).2.out
      { J with ok := fun Γ w => ∃ a, (emitGo p s).1 = some a ∧
          Γ.size = (emitGo p s).2.n ∧ Q a Γ w }

section rules
variable {α β : Type} {cfg : Cfg} {d : Nat} {J : Post}

theorem PT.ret {P : Env → World → Prop} {a : α} {Q : α → Env → World → Prop}
    (h : ∀ Γ w, P Γ w → Q a Γ w) : PT cfg d J P (.ret a) Q := by
  intro s P0 _ _ hs
  refine conseq hs (fun _ _ h => h) ?_ (fun _ _ _ _ h => h) (fun _ _ _ h => h)
  rintro Γ w ⟨hn, hP⟩
  exact ⟨a, rfl, hn, h Γ w hP⟩

theorem PT.conseq {P P' : Env → World → Prop} {p : Prog Slot Lvl α} {Q Q' : α → Env → World → Prop}
    (h : PT cfg d J P p Q) (hP : ∀ Γ w, P' Γ w → P Γ w) (hQ : ∀ a Γ w, Q a Γ w → Q' a Γ w) :
    PT cfg d J P' p Q' := by
  intro s P0 hd hf hs
  refine Hoare.conseq (h s P0 hd hf (Hoare.conseq hs (fun _ _ h => h) ?_ (fun _ _ _ _ h => h)
    (fun _ _ _ h => h))) (fun _ _ h => h) ?_ (fun _ _ _ _ h => h) (fun _ _ _ h => h)
  · rintro Γ w ⟨hn, hP'⟩; exact ⟨hn, hP Γ w hP'⟩
  · rintro Γ w ⟨a, ha, hn, hq⟩; exact ⟨a, ha, hn, hQ a Γ w hq⟩

/-- **Sequencing.** -/
theorem PT.bind {P : Env → World → Prop} {p : Prog Slot Lvl α} {f : α → Prog Slot Lvl β}
    {Q : α → Env → World → Prop} {R : β → Env → World → Prop}
    (hp : PT cfg d J P p Q) (hf : ∀ a, PT cfg d J (Q a) (f a) R) :
    PT cfg d J P (p >>= f) R := by
  intro s P0 hd hfin hs
  rw [emitGo_bind] at hfin ⊢
  have hc := emitGo_cnt p s
  generalize he : emitGo p s = e at hc hfin ⊢
  obtain ⟨r, t⟩ := e
  have h1 := hp s P0 hd (by
    rw [he]
    cases r with
    | none => exact hfin
    | some a => exact (emitGo_cnt (f a) t).fine hfin) hs
  rw [he] at h1
  cases r with
  | none =>
      exact Hoare.conseq h1 (fun _ _ h => h) (fun _ _ ⟨_, ha, _⟩ => nomatch ha)
        (fun _ _ _ _ h => h) (fun _ _ _ h => h)
  | some a =>
      have ht : t.depth = d := by
        have := emitGo_depth p s; rw [he] at this; exact this.trans hd
      refine hf a t P0 ht hfin (Hoare.conseq h1 (fun _ _ h => h) ?_
        (fun _ _ _ _ h => h) (fun _ _ _ h => h))
      rintro Γ w ⟨a', ha, hn, hq⟩
      cases ha
      exact ⟨hn, hq⟩

/-- A code run from one state, split on how it ends normally: some `x` holds
    where it does, and where it never does any `x` serves. More fuel changes no
    finished run, so every run from the state ends in the same place. -/
theorem Triple.pick {cfg : Cfg} {P0 : Env → World → Prop} {c : Code} {J : Post} {X : Type}
    [Nonempty X] {P : X → Env → World → Prop}
    (h : Triple cfg P0 c { J with ok := fun Γ w => ∃ x, P x Γ w }) {Γ : Env} {w : World}
    (h0 : P0 Γ w) : ∃ x, Triple cfg (At Γ w) c { J with ok := P x } := by
  by_cases hex : ∃ f Γ' w', runCode f cfg Γ w c = .ok Γ' w'
  · obtain ⟨f, Γ', w', hr⟩ := hex
    have hs := h f Γ w h0
    rw [hr] at hs
    obtain ⟨x, hx⟩ := hs
    refine ⟨x, ?_⟩
    intro f' Γ1 w1 ⟨e1, e2⟩
    rw [e1, e2]
    have h' := h f' Γ w h0
    cases hr' : runCode f' cfg Γ w c with
    | ok Γ'' w'' =>
        have e : CodeRes.ok Γ'' w'' = .ok Γ' w' := by
          rcases Nat.le_total f f' with hf | hf
          · rw [← hr', runCode_mono hf hr rfl]
          · rw [← hr, runCode_mono hf hr' rfl]
        cases e
        exact hx
    | _ => rw [hr'] at h'; exact h'
  · obtain ⟨x⟩ := ‹Nonempty X›
    refine ⟨x, ?_⟩
    intro f' Γ1 w1 ⟨e1, e2⟩
    rw [e1, e2]
    have h' := h f' Γ w h0
    cases hr' : runCode f' cfg Γ w c with
    | ok Γ'' w'' => exact absurd ⟨f', Γ'', w'', hr'⟩ hex
    | _ => rw [hr'] at h'; exact h'

/-- **An existential precondition** is proved one witness at a time. -/
theorem PT.exists {X : Type} [Nonempty X] {P : X → Env → World → Prop} {p : Prog Slot Lvl α}
    {Q : α → Env → World → Prop} (h : ∀ x, PT cfg d J (P x) p Q) :
    PT cfg d J (fun Γ w => ∃ x, P x Γ w) p Q := by
  intro s P0 hd hf hs fuel Γ w h0
  have hs' : Triple cfg P0 s.out { J with ok := fun Γ w => ∃ x, Γ.size = s.n ∧ P x Γ w } :=
    Hoare.conseq hs (fun _ _ h => h) (fun _ _ ⟨hn, x, hx⟩ => ⟨x, hn, hx⟩)
      (fun _ _ _ _ h => h) (fun _ _ _ h => h)
  obtain ⟨x, hx⟩ := Triple.pick hs' h0
  exact h x s (At Γ w) hd hf hx fuel Γ w ⟨rfl, rfl⟩

/-- An operation binds the next slot. -/
theorem PT.op {ty} {P : Env → World → Prop} {o : Op' Slot ty} {k : Slot ty → Prog Slot Lvl α}
    {Q : α → Env → World → Prop}
    (h : ∀ r, PT cfg d J (fun Γ w => ∃ Γ0 v, Γ = Γ0.push v ∧ Γ0.size = r ∧ P Γ0 w ∧
      evalOp w.mem Γ0 o.erase = some v) (k r) Q)
    (hnf : ∀ Γ w, P Γ w → evalOp w.mem Γ o.erase = none → J.faultOk = true := by intros; rfl) :
    PT cfg d J P (.op o k) Q := by
  intro s P0 hd hf hs
  rw [emitGo_op] at hf ⊢
  refine h s.n _ P0 hd hf (St.out_stmt hs (op_rule ?_ (fun Γ w hP hv => hnf Γ w hP.2 hv)))
  rintro Γ w v ⟨hn, hP⟩ hv
  exact ⟨by simp [hn], Γ, v, rfl, hn, hP, hv⟩

/-- A statement that binds nothing. -/
theorem PT.stmt {P R : Env → World → Prop} {st : Stmt} {k : Prog Slot Lvl α}
    {Q : α → Env → World → Prop}
    (hs : ∀ n, Stmt1 cfg (fun Γ w => Γ.size = n ∧ P Γ w) st (fun Γ w => Γ.size = n ∧ R Γ w) J.faultOk)
    (hk : PT cfg d J R k Q) (s : St) (P0 : Env → World → Prop) (hd : s.depth = d)
    (hf : Fine (emitGo k (s.stmt st)).2)
    (h : Triple cfg P0 s.out { J with ok := fun Γ w => Γ.size = s.n ∧ P Γ w }) :
    Triple cfg P0 (emitGo k (s.stmt st)).2.out
      { J with ok := fun Γ w => ∃ a, (emitGo k (s.stmt st)).1 = some a ∧
          Γ.size = (emitGo k (s.stmt st)).2.n ∧ Q a Γ w } :=
  hk _ P0 hd hf (St.out_stmt h (hs s.n))

theorem PT.store {ty} {P R : Env → World → Prop} {v : Slot ty} {a : Slot .i64} {k : Prog Slot Lvl α}
    {Q : α → Env → World → Prop}
    (hs : ∀ n, Stmt1 cfg (fun Γ w => Γ.size = n ∧ P Γ w) (.store ty v a) (fun Γ w => Γ.size = n ∧ R Γ w) J.faultOk)
    (hk : PT cfg d J R k Q) : PT cfg d J P (.store v a k) Q := by
  intro s P0 hd hf h; rw [emitGo_store] at hf ⊢; exact PT.stmt hs hk s P0 hd hf h

/-- A foreign call that answers binds the next slot. -/
theorem PT.call {f : Ffi} {args : Vals Slot f.params} {k : ResV Slot f.result → Prog Slot Lvl α}
    {P : Env → World → Prop} {R : Nat → Env → World → Prop} {Q : α → Env → World → Prop}
    (hres : f.result.isSome = true)
    (hs : ∀ n, Stmt1 cfg (fun Γ w => Γ.size = n ∧ P Γ w) (.call (.ffi f) args.slots)
      (fun Γ w => Γ.size = n + 1 ∧ R n Γ w) J.faultOk)
    (hk : ∀ r, PT cfg d J (R r) (k (resSlot f.result r)) Q) :
    PT cfg d J P (.call f args k) Q := by
  intro s P0 hd hf h
  rw [emitGo_call] at hf ⊢; unfold emitCall at hf ⊢
  simp only [hres, if_true] at hf ⊢
  exact hk s.n _ P0 hd hf (St.out_stmt h (hs s.n))

/-- A foreign call that answers nothing. -/
theorem PT.callVoid {f : Ffi} {args : Vals Slot f.params} {k : ResV Slot f.result → Prog Slot Lvl α}
    {P R : Env → World → Prop} {Q : α → Env → World → Prop}
    (hres : f.result.isSome = false)
    (hs : ∀ n, Stmt1 cfg (fun Γ w => Γ.size = n ∧ P Γ w) (.callVoid (.ffi f) args.slots)
      (fun Γ w => Γ.size = n ∧ R Γ w) J.faultOk)
    (hk : PT cfg d J R (k (resSlot f.result 0)) Q) :
    PT cfg d J P (.call f args k) Q := by
  intro s P0 hd hf h
  rw [emitGo_call] at hf ⊢; unfold emitCall at hf ⊢
  simp only [hres] at hf ⊢
  exact hk _ P0 hd hf (St.out_stmt h (hs s.n))

/-- Reading the entry block's parameters spends nothing. -/
theorem PT.params {tys : List ClifTy} {P : Env → World → Prop} {k : Vals Slot tys → Prog Slot Lvl α}
    {Q : α → Env → World → Prop} (h : PT cfg d J P (k (carriesFrom 0 tys)) Q) :
    PT cfg d J P (.params tys k) Q := by
  intro s P0 hd hf hs; rw [emitGo_params] at hf ⊢; exact h s P0 hd hf hs

/-- Leaving the loop `l` names: the code after it never ends normally. -/
theorem PT.br {ex ca} {P : Env → World → Prop} {l : Lvl ex ca} {args : Vals Slot ex}
    {Q : α → Env → World → Prop}
    (h : ∀ Γ w vs, P Γ w → args.slots.mapM (fun r => Γ[r]?) = some vs →
      J.brk (labelDepth d l) Γ vs w) :
    PT cfg d J P (.br l args) Q := by
  intro s P0 hd _ hs
  rw [emitGo_br]; unfold emitJump; dsimp only
  refine St.out_piece hs (br_rule ?_) (fun _ _ _ _ h => h) (fun _ _ _ h => h)
  rintro Γ w vs ⟨_, hP⟩ hv
  rw [St.flush_depth, hd]; exact h Γ w vs hP hv

/-- Going round the loop `l` names again. -/
theorem PT.cont {ex ca} {P : Env → World → Prop} {l : Lvl ex ca} {args : Vals Slot ca}
    {Q : α → Env → World → Prop}
    (h : ∀ Γ w vs, P Γ w → args.slots.mapM (fun r => Γ[r]?) = some vs →
      J.cont (labelDepth d l) vs w) :
    PT cfg d J P (.cont l args) Q := by
  intro s P0 hd _ hs
  rw [emitGo_cont]; unfold emitJump; dsimp only
  refine St.out_piece hs (cont_rule ?_) (fun _ _ _ _ h => h) (fun _ _ _ h => h)
  rintro Γ w vs ⟨_, hP⟩ hv
  rw [St.flush_depth, hd]; exact h Γ w vs hP hv

end rules

theorem stmtsSlots_ge : ∀ (ss : List Stmt) (n : Nat), n ≤ stmtsSlots n ss := by
  intro ss
  induction ss with
  | nil => intro n; simp [stmtsSlots]
  | cons st ss ih =>
      intro n
      have e : stmtsSlots n (st :: ss) = stmtsSlots (match st with | .op _ | .call _ _ => n + 1 | _ => n) ss :=
        rfl
      rw [e]
      exact Nat.le_trans (by split <;> omega) (ih _)

/-- A code never counts fewer slots than it starts with. -/
theorem slotsS_ge (c : Code) : ∀ n, n ≤ slotsS n c :=
  Piece.rec_1 (motive_1 := fun p => ∀ n, n ≤ slotsP n p) (motive_2 := fun c => ∀ n, n ≤ slotsS n c)
    (fun ss n => by simp only [slotsP]; exact stmtsSlots_ge ss n)
    (fun l pre body hp hb n => by
      simp only [slotsP]
      have := hp (n + l.pTys.length); have := hb (slotsS (n + l.pTys.length) pre + l.pTys.length)
      omega)
    (fun m thn els _ _ ht he n => by
      simp only [slotsP]; have := ht n; have := he (slotsS n thn); omega)
    (fun l body hb n => by simp only [slotsP]; have := hb (n + l.pTys.length); omega)
    (fun _ _ n => by simp [slotsP])
    (fun _ _ n => by simp [slotsP])
    (fun n => by simp [slotsS])
    (fun _ _ hp hps n => by simp only [slotsS]; exact Nat.le_trans (hp n) (hps _))
    c

-- ---------------------------------------------------------------------------
-- Loops
-- ---------------------------------------------------------------------------

theorem mapM_len {α β : Type} {f : α → Option β} :
    ∀ {xs : List α} {ys : List β}, xs.mapM f = some ys → ys.length = xs.length := by
  intro xs
  induction xs with
  | nil => intro ys h; simp at h; subst h; rfl
  | cons x xs ih =>
      intro ys h
      simp only [List.mapM_cons, Option.bind_eq_bind] at h
      cases hx : f x with
      | none => simp [hx] at h
      | some y =>
        cases hr : xs.mapM f with
        | none => simp [hx, hr] at h
        | some ys' =>
          simp [hx, hr] at h; subst h; simp [ih hr]

theorem Vals.slots_length : ∀ {tys : List ClifTy} (vs : Vals Slot tys), vs.slots.length = tys.length
  | [], .nil => rfl
  | _ :: _, .cons _ vs => by simp [Vals.slots, Vals.slots_length vs]

theorem bindAt_size' (Γ : Env) (n : Nat) (vs : List V) : (bindAt Γ n vs).size = n + vs.length := by
  unfold bindAt; simp; omega

theorem slotsOf_eq {c : Code} (h : needS c ≤ HProg.fuel) (n : Nat) : slotsOf n c = slotsS n c :=
  (slotsGo_eq HProg.fuel c n h).1

theorem Fine.of_push {S1 X : St} {p : Piece} (h : Fine X) (hp : X.pieces = p :: S1.pieces)
    (hc : X.cur = []) (hc1 : S1.cur = []) : X.err = none ∧ needS (S1.out ++ [p]) ≤ HProg.fuel := by
  rw [← St.out_push' hp hc hc1]; exact h

theorem Triple.of_push {cfg : Cfg} {P0 : Env → World → Prop} {Q : Post} {S1 X : St} {p : Piece}
    (h : Triple cfg P0 (S1.out ++ [p]) Q) (hp : X.pieces = p :: S1.pieces) (hc : X.cur = [])
    (hc1 : S1.cur = []) : Triple cfg P0 X.out Q := by
  rw [St.out_push' hp hc hc1]; exact h

/-- What a loop's head and body may do besides finish: leave this loop with
    values `Xit` accepts, go round it with carries the invariant accepts, or
    leave or go round an enclosing loop as `J` allows. -/
def loopJ (J : Post) (Γ0 : Env) (I : Env → List V → World → Prop)
    (Xit : Env → Env → List V → World → Prop) (exitTys tys : List ClifTy) : Post where
  ok _ _ := True
  brk e Γb vs w := match e with
    | 0 => vs.length = exitTys.length ∧ Xit Γ0 Γb vs w
    | e + 1 => J.brk e Γb vs w
  cont e vs w := match e with
    | 0 => vs.length = tys.length ∧ I Γ0 vs w
    | e + 1 => J.cont e vs w
  faultOk := J.faultOk

@[simp] theorem loopJ_faultOk (J : Post) (Γ0 : Env) (I : Env → List V → World → Prop)
    (Xit : Env → Env → List V → World → Prop) (exitTys tys : List ClifTy) :
    (loopJ J Γ0 I Xit exitTys tys).faultOk = J.faultOk := rfl

/-- Where faults are allowed, an operation's or a statement's condition that
    it does not fault holds. -/
theorem and_of_faultOk {J : Post} {A B : Prop} (hJ : J.faultOk = true) :
    ((A → J.faultOk = true) ∧ B) = B := by simp [hJ]

theorem and_of_faultOk' {J : Post} {α : Type} {A : α → Prop} {B : Prop} (hJ : J.faultOk = true) :
    ((∀ m, A m → J.faultOk = true) ∧ B) = B := by simp [hJ]

/-- **A top-tested loop, by invariant.** `I Γ0` holds of the carries at every
    trip of a loop entered from `Γ0`. From there the head ends in `Qh` of its
    answer; where the test says leave, the exits satisfy `Xit`; where it says go
    on, the body ends with next carries satisfying `I`. Leaving the loop by a
    `br` must satisfy `Xit` too, and going round by a `cont`, `I`. What follows
    the loop starts from its exit values bound after every slot of the loop. -/
theorem PT.loop {α : Type} {cfg : Cfg} {d : Nat} {J : Post}
    {tys exitTys : List ClifTy} {β : Type} {init : Vals Slot tys}
    {head : Lvl exitTys tys → Vals Slot tys → Prog Slot Lvl (Cond Slot × Vals Slot exitTys × β)}
    {body : Lvl exitTys tys → Vals Slot tys → β → Prog Slot Lvl (Vals Slot tys)}
    {k : Vals Slot exitTys → Prog Slot Lvl α}
    {P : Env → World → Prop} {Q : α → Env → World → Prop}
    (I : Env → List V → World → Prop)
    (Qh : Env → List V → Cond Slot × Vals Slot exitTys × β → Env → World → Prop)
    (Xit : Env → Env → List V → World → Prop)
    (hinit : ∀ Γ w cs, P Γ w → init.slots.mapM (fun r => Γ[r]?) = some cs → I Γ cs w)
    (hhead : ∀ n0 Γ0 cs, PT cfg (d + 1) (loopJ J Γ0 I Xit exitTys tys)
      (fun Γh wh => Γ0.size = n0 ∧ Γh = bindAt Γ0 n0 cs ∧ I Γ0 cs wh)
      (head d (carriesFrom n0 tys)) (Qh Γ0 cs))
    (hexit : ∀ Γ0 cs a Γ1 w1 t f vs, Qh Γ0 cs a Γ1 w1 →
      evalOp w1.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) = some (.sc t f) →
      ((f != 0) == a.1.exitOnTrue) = true →
      a.2.1.slots.mapM (fun r => (Γ1.push (.sc t f))[r]?) = some vs →
      Xit Γ0 (Γ1.push (.sc t f)) vs w1)
    (hbody : ∀ nb Γ0 cs a, PT cfg (d + 1) (loopJ J Γ0 I Xit exitTys tys)
      (fun Γb wb => ∃ Γ1 t f, Γ1.size + 1 = nb ∧ Γb = bindAt (Γ1.push (.sc t f)) nb cs ∧
        Qh Γ0 cs a Γ1 wb ∧
        evalOp wb.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) = some (.sc t f) ∧
        ((f != 0) == a.1.exitOnTrue) = false)
      (body d (carriesFrom nb tys) a.2.2)
      (fun next Γ2 w2 => ∀ nx, next.slots.mapM (fun r => Γ2[r]?) = some nx → I Γ0 nx w2))
    (hk : ∀ nE, PT cfg d J (fun Γ w => ∃ Γ0 Γb vs, Γ0.size ≤ nE ∧ Γ = bindAt Γb nE vs ∧
      Xit Γ0 Γb vs w) (k (carriesFrom nE exitTys)) Q)
    (hhf : ∀ Γ0 cs a Γ1 w1, Qh Γ0 cs a Γ1 w1 →
      evalOp w1.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) = none → J.faultOk = true := by intros; rfl) :
    PT cfg d J P (.loop init head body k) Q := by
  intro s P0 hd hfin hpre
  rw [emitGo_loop] at hfin ⊢
  unfold emitLoop at hfin ⊢
  have e1 := Cnt.flush s
  have hc1 := St.flush_cur s
  have hS1n := St.flush_n s
  have hS1d : s.flush.depth = d := (St.flush_depth s).trans hd
  have hS1out := St.out_flush s
  generalize s.flush = S1 at e1 hc1 hS1n hS1d hS1out hfin ⊢
  obtain ⟨hr, sH, hHe⟩ : ∃ hr sH, emitGo (head S1.depth (carriesFrom S1.n tys))
      ({ S1 with n := S1.n + tys.length, depth := S1.depth + 1 } : St).enter = (hr, sH) :=
    ⟨_, _, rfl⟩
  have cH := emitGo_cnt (head S1.depth (carriesFrom S1.n tys))
    ({ S1 with n := S1.n + tys.length, depth := S1.depth + 1 } : St).enter
  have dH := emitGo_depth (head S1.depth (carriesFrom S1.n tys))
    ({ S1 with n := S1.n + tys.length, depth := S1.depth + 1 } : St).enter
  subst hS1d
  have tH := hhead S1.n
  rw [hHe] at cH dH
  simp only [hHe] at hfin ⊢
  cases hr with
  | none => exact absurd hfin.1 (St.note_err _ _)
  | some a =>
  obtain ⟨c, exitR, x⟩ := a
  dsimp only at hfin ⊢
  have cF := Cnt.bind1 sH (.op (.icmp c.cc c.a c.b)) rfl
  have hFo : (sH.bind1 (.op (.icmp c.cc c.a c.b))).2.out = (sH.stmt (.op (.icmp c.cc c.a c.b))).out :=
    rfl
  have hFn : (sH.bind1 (.op (.icmp c.cc c.a c.b))).2.n = sH.n + 1 := rfl
  have hFd : (sH.bind1 (.op (.icmp c.cc c.a c.b))).2.depth = sH.depth := rfl
  have hFf : (sH.bind1 (.op (.icmp c.cc c.a c.b))).1 = sH.n := rfl
  generalize sH.bind1 (.op (.icmp c.cc c.a c.b)) = rF at cF hFo hFn hFd hFf hfin ⊢
  obtain ⟨flag, sH'⟩ := rF
  dsimp only at cF hFo hFn hFd hFf hfin ⊢
  subst hFf
  obtain ⟨bd, sB, hBe⟩ : ∃ bd sB, emitGo (body S1.depth (carriesFrom sH'.n tys) x)
      ({ (St.leave ({ S1 with n := S1.n + tys.length, depth := S1.depth + 1 } : St) sH').2 with
          n := sH'.n + tys.length } : St).enter = (bd, sB) := ⟨_, _, rfl⟩
  have cB := emitGo_cnt (body S1.depth (carriesFrom sH'.n tys) x)
      ({ (St.leave ({ S1 with n := S1.n + tys.length, depth := S1.depth + 1 } : St) sH').2 with
          n := sH'.n + tys.length } : St).enter
  have tB := hbody sH'.n
  rw [hBe] at cB
  simp only [St.leave_n, hBe] at hfin ⊢
  dsimp only at cH cB
  -- What the rest of the body is emitted from, and that the whole emission is fine.
  refine hk _ _ P0 rfl hfin ?_
  have fX := (emitGo_cnt _ _).fine hfin
  have fP := Fine.of_push (S1 := S1) fX rfl hc1 hc1
  have hlt := Nat.lt_of_lt_of_le (needP_lt S1.out _) fP.2
  simp only [needP_straight, needP_loop, needP_ite, needP_dloop, needP_br, needP_cont, St.leave_fst] at hlt
  have fB : Fine sB := ⟨by simpa using fP.1, by omega⟩
  have fH' : Fine sH' := ⟨by simpa using cB.1 fB.1, by omega⟩
  have fH : Fine sH := cF.fine fH'
  have kH : slotsS (S1.n + tys.length) sH.out = sH.n := cH.2.2 fH.1 fH.2 _ rfl
  have kH' : slotsS (S1.n + tys.length) sH'.out = sH'.n := cF.2.2 fH'.1 fH'.2 _ kH
  have kB : slotsS (sH'.n + tys.length) sB.out = sB.n := cB.2.2 fB.1 fB.2 _ rfl
  refine Triple.of_push (S1 := S1) ?_ rfl hc1 hc1
  rw [hS1out]
  -- What leaving the loop with `vs` from `Γb` hands the rest of the body.
  refine append_rule (Hoare.conseq hpre (fun _ _ h => h) (fun Γ w ⟨hn, hP⟩ => ⟨hS1n ▸ hn, hP⟩)
    (fun _ _ _ _ h => h) (fun _ _ _ h => h)) (cons_rule ?_ (nil_rule fun _ _ h => h))
  refine loop_rule (fun Γ0 cs w => cs.length = tys.length ∧ I Γ0 cs w) ?_ ?_
  · rintro Γ w cs ⟨_, hP⟩ hcs
    exact ⟨(mapM_len hcs).trans (Vals.slots_length init), hinit Γ w cs hP hcs⟩
  rintro Γ0 w0 cs w ⟨hsz, _⟩ ⟨hlen, hI⟩
  simp only [St.leave_fst]
  rw [hS1n] at kH kH'
  rw [hsz, slotsOf_eq fH'.2, kH', slotsOf_eq fB.2, kB]
  -- Leaving the loop, by its test or by a `br 0`.
  have hle : Γ0.size ≤ sB.n := by
    have := slotsS_ge sH'.out (s.n + tys.length)
    have := slotsS_ge sB.out (sH'.n + tys.length)
    omega
  have hQx : ∀ Γb vs w', vs.length = exitTys.length → Xit Γ0 Γb vs w' →
      (bindAt Γb sB.n vs).size = sB.n + exitTys.length ∧
        ∃ Γ0' Γb' vs', Γ0'.size ≤ sB.n ∧ bindAt Γb sB.n vs = bindAt Γb' sB.n vs' ∧ Xit Γ0' Γb' vs' w' :=
    fun Γb vs w' hl hx => ⟨by rw [bindAt_size', hl], Γ0, Γb, vs, hle, rfl, hx⟩
  -- The head, from the carries bound after the entry environment.
  have h1 := hhead S1.n Γ0 cs
    ({ S1 with n := S1.n + tys.length, depth := S1.depth + 1 } : St).enter
    (At (bindAt Γ0 s.n cs) w) rfl (by rw [hHe]; exact fH) (nil_rule (by
      rintro _ _ ⟨rfl, rfl⟩
      refine ⟨?_, hsz.trans hS1n.symm, by rw [hS1n], hI⟩
      simp [bindAt_size', hlen, hS1n]))
  rw [hHe] at h1
  dsimp only at h1
  have h2 := St.out_stmt (st := .op (.icmp c.cc c.a c.b)) h1 (op_rule
    (Q := fun Γ' w' => ∃ Γ1 v, Γ' = Γ1.push v ∧ Γ1.size = sH.n ∧ Qh Γ0 cs (c, exitR, x) Γ1 w' ∧
      evalOp w'.mem Γ1 (.icmp c.cc c.a c.b) = some v)
    (fun Γ w' v ⟨a, ha, hs1, hq⟩ hv => by cases ha; exact ⟨Γ, v, rfl, hs1, hq, hv⟩)
    (fun Γ w' ⟨a, ha, _, hq⟩ hv => by cases ha; exact hhf Γ0 cs _ Γ w' hq hv))
  rw [← hFo] at h2
  refine Hoare.conseq h2 (fun _ _ h => h) ?_ ?_ ?_
  · rintro _ w1 ⟨Γ1, v, rfl, hs1, hq, hv⟩ t f hflag
    have hvv : v = .sc t f := by
      have : (Γ1.push v)[sH.n]? = some v := by rw [← hs1]; simp
      rw [this] at hflag; exact Option.some.inj hflag
    subst hvv
    refine ⟨fun hx vs hvs => ?_, fun hx => ?_⟩
    · exact hQx _ vs w1 ((mapM_len hvs).trans (Vals.slots_length exitR))
        (hexit Γ0 cs (c, exitR, x) Γ1 w1 t f vs hq hv hx hvs)
    · have h3 := hbody sH'.n Γ0 cs (c, exitR, x)
        ({ (St.leave ({ S1 with n := S1.n + tys.length, depth := S1.depth + 1 } : St) sH').2 with
            n := sH'.n + tys.length } : St).enter
        (At (bindAt (Γ1.push (.sc t f)) sH'.n cs) w1)
        (by rw [St.enter, St.leave_depth, hFd]; simpa using dH)
        (by rw [hBe]; exact fB)
        (nil_rule (by
          rintro _ _ ⟨rfl, rfl⟩
          exact ⟨by simp [bindAt_size', hlen], Γ1, t, f, by rw [hs1, hFn], rfl, hq, hv,
            by simpa using hx⟩))
      rw [hBe] at h3
      dsimp only at h3
      have hnb : slotsOf (s.n + tys.length) sH'.out = sH'.n := by rw [slotsOf_eq fH'.2, kH']
      refine Hoare.conseq h3 (fun _ _ h => by simp only at h; rwa [hnb] at h) ?_ ?_ ?_
      · rintro Γ2 w2 ⟨a', ha', _, hq'⟩ next hnext
        subst ha'
        exact ⟨(mapM_len hnext).trans (Vals.slots_length a'), hq' next hnext⟩
      · rintro (_ | e) Γb vs w' he
        · exact hQx Γb vs w' he.1 he.2
        · exact he
      · rintro (_ | e) vs w' he
        · exact he
        · exact he
  · rintro (_ | e) Γb vs w' he
    · exact hQx Γb vs w' he.1 he.2
    · exact he
  · rintro (_ | e) vs w' he
    · exact he
    · exact he

-- ---------------------------------------------------------------------------
-- Branches
-- ---------------------------------------------------------------------------

/-- A code that leaves its region unconditionally never finishes normally. -/
theorem terms_no_ok (cfg : Cfg) : ∀ f,
    (∀ Γ w p Γ' w', termsP p = true → runPiece f cfg Γ w p ≠ .ok Γ' w') ∧
    (∀ Γ w c Γ' w', termsS c = true → runCode f cfg Γ w c ≠ .ok Γ' w')
  | 0 => ⟨fun _ _ _ _ _ _ => by simp [runPiece], fun _ _ _ _ _ _ => by simp [runCode]⟩
  | f + 1 => by
    obtain ⟨ihP, ihC⟩ := terms_no_ok cfg f
    refine ⟨fun Γ w p Γ' w' ht => ?_, fun Γ w c Γ' w' ht => ?_⟩
    · cases p with
      | br d args => simp only [runPiece]; split <;> simp
      | cont d args => simp only [runPiece]; split <;> simp
      | ite m thn els tr er =>
          simp only [termsP, Bool.and_eq_true] at ht
          intro h
          simp only [runPiece] at h
          split at h
          · rename_i t fl _
            by_cases hb : (fl != 0) = true
            · simp only [hb, if_true] at h
              split at h <;> try cases h
              rename_i Γ1 w1 hr
              exact ihC _ _ _ _ _ ht.1 hr
            · simp only [hb, if_false] at h
              split at h <;> try cases h
              rename_i Γ1 w1 hr
              exact ihC _ _ _ _ _ ht.2 hr
          · cases h
      | straight _ => simp [termsP] at ht
      | loop _ _ _ => simp [termsP] at ht
      | dloop _ _ => simp [termsP] at ht
    · match c, ht with
      | [p], ht =>
          simp only [termsS] at ht
          simp only [runCode]
          intro h
          split at h <;> try (cases h)
          rename_i Γ1 w1 hr
          exact ihP _ _ _ _ _ ht hr
      | q :: p :: ps, ht =>
          simp only [termsS] at ht
          simp only [runCode]
          intro h
          split at h <;> try (cases h)
          exact ihC _ _ _ _ _ ht h

theorem Triple.terms {cfg : Cfg} {P : Env → World → Prop} {c : Code} {Q : Post}
    (h : Triple cfg P c Q) (ht : termsS c = true) :
    Triple cfg P c { Q with ok := fun _ _ => False } := by
  intro f Γ w hP
  have := h f Γ w hP
  cases hr : runCode f cfg Γ w c with
  | ok Γ' w' => exact absurd hr ((terms_no_ok cfg f).2 _ _ _ _ _ ht)
  | _ => rw [hr] at this; exact this


/-- **A branch.** The then arm starts from the test's slot bound after what
    came before; the else arm from there padded past the then arm's slots, as
    the model numbers it. What follows starts from the arm's exports bound after
    every slot of both arms. -/
theorem PT.ite {α : Type} {cfg : Cfg} {d : Nat} {J : Post} {jTys : List ClifTy} {c : Cond Slot}
    {thn els : Prog Slot Lvl (Vals Slot jTys)} {k : Vals Slot jTys → Prog Slot Lvl α}
    {P : Env → World → Prop} {Q : α → Env → World → Prop}
    (Qt Qe : Vals Slot jTys → Env → World → Prop)
    (hthn : PT cfg d J (fun Γ w => ∃ Γ0 t f, Γ = Γ0.push (.sc t f) ∧ P Γ0 w ∧
      evalOp w.mem Γ0 (.icmp c.cc c.a c.b) = some (.sc t f) ∧ (f != 0) = true) thn Qt)
    (hels : ∀ ne, PT cfg d J (fun Γ w => ∃ Γ0 t f, Γ0.size + 1 ≤ ne ∧
      Γ = bindAt (Γ0.push (.sc t f)) ne [] ∧ P Γ0 w ∧
      evalOp w.mem Γ0 (.icmp c.cc c.a c.b) = some (.sc t f) ∧ (f != 0) = false) els Qe)
    (hk : ∀ nJ, PT cfg d J (fun Γ w => ∃ Γ' jv vs, Γ'.size ≤ nJ ∧ Γ = bindAt Γ' nJ vs ∧
      jv.slots.mapM (fun r => Γ'[r]?) = some vs ∧ (Qt jv Γ' w ∨ Qe jv Γ' w))
      (k (carriesFrom nJ jTys)) Q)
    (hcf : ∀ Γ w, P Γ w → evalOp w.mem Γ (.icmp c.cc c.a c.b) = none → J.faultOk = true := by intros; rfl) :
    PT cfg d J P (.ite c thn els k) Q := by
  intro s P0 hd hfin hpre
  rw [emitGo_ite] at hfin ⊢
  unfold emitIte at hfin ⊢
  have hP1 := St.out_stmt (st := .op (.icmp c.cc c.a c.b)) hpre (op_rule
    (Q := fun Γ w => Γ.size = s.n + 1 ∧ ∃ Γ0 v, Γ = Γ0.push v ∧ Γ0.size = s.n ∧ P Γ0 w ∧
      evalOp w.mem Γ0 (.icmp c.cc c.a c.b) = some v)
    (fun Γ w v ⟨hn, hP⟩ hv => ⟨by simp [hn], Γ, v, rfl, hn, hP, hv⟩)
    (fun Γ w ⟨_, hP⟩ hv => hcf Γ w hP hv))
  have hFo : (s.bind1 (.op (.icmp c.cc c.a c.b))).2.out = (s.stmt (.op (.icmp c.cc c.a c.b))).out := rfl
  have hFn : (s.bind1 (.op (.icmp c.cc c.a c.b))).2.n = s.n + 1 := rfl
  have hFd : (s.bind1 (.op (.icmp c.cc c.a c.b))).2.depth = s.depth := rfl
  have hFf : (s.bind1 (.op (.icmp c.cc c.a c.b))).1 = s.n := rfl
  rw [← hFo] at hP1
  generalize s.bind1 (.op (.icmp c.cc c.a c.b)) = rF at hP1 hFo hFn hFd hFf hfin ⊢
  obtain ⟨flag, s1⟩ := rF
  dsimp only at hP1 hFo hFn hFd hFf hfin ⊢
  subst hFf
  have hc1 := St.flush_cur s1
  have hS1n : s1.flush.n = s.n + 1 := (St.flush_n s1).trans hFn
  have hS1d : s1.flush.depth = d := ((St.flush_depth s1).trans hFd).trans hd
  have hS1out := St.out_flush s1
  rw [← hS1out] at hP1
  generalize s1.flush = S1 at hc1 hS1n hS1d hS1out hP1 hfin ⊢
  obtain ⟨tR, sT, hTe⟩ : ∃ a b, emitGo thn S1.enter = (a, b) := ⟨_, _, rfl⟩
  have cT := emitGo_cnt thn S1.enter
  have tT := hthn S1.enter
  rw [hTe] at cT
  simp only [hTe] at hfin ⊢
  obtain ⟨eR, sE, hEe⟩ : ∃ a b, emitGo els (St.leave S1 sT).2.enter = (a, b) := ⟨_, _, rfl⟩
  have cE := emitGo_cnt els (St.leave S1 sT).2.enter
  have tE := hels sT.n (St.leave S1 sT).2.enter
  rw [hEe] at cE
  simp only [hEe, St.leave_n] at hfin ⊢
  dsimp only at cT cE
  have hdT : sT.depth = S1.depth := by
    have := emitGo_depth thn S1.enter; rw [hTe] at this; exact this
  have hdE : sE.depth = sT.depth := by
    have := emitGo_depth els (St.leave S1 sT).2.enter; rw [hEe] at this
    exact this.trans (St.leave_depth S1 sT)
  refine hk _ _ P0 (by rw [St.leave_depth, hdE, hdT, hS1d]) hfin ?_
  have fX := (emitGo_cnt _ _).fine hfin
  have fP := Fine.of_push (S1 := S1) fX rfl hc1 hc1
  have hlt := Nat.lt_of_lt_of_le (needP_lt S1.out _) fP.2
  simp only [needP_ite, St.leave_fst] at hlt
  have fE : Fine sE := ⟨by simpa using fP.1, by omega⟩
  have fT : Fine sT := ⟨by simpa using cE.1 fE.1, by omega⟩
  have kT : slotsS S1.n sT.out = sT.n := cT.2.2 fT.1 fT.2 _ rfl
  have kE : slotsS sT.n sE.out = sE.n := cE.2.2 fE.1 fE.2 _ (by simp [slotsS])
  simp only [St.leave_fst, terminates_eq fT.2, terminates_eq fE.2]
  refine Triple.of_push (S1 := S1) ?_ rfl hc1 hc1
  refine append_rule hP1 (cons_rule (ite_rule ?_ ?_) (nil_rule fun _ _ h => h))
  · rintro _ w t f ⟨hsz, Γ0, v, rfl, hs0, hP, hv⟩ hflag hb
    have hvv : v = .sc t f := by
      have : (Γ0.push v)[s.n]? = some v := by rw [← hs0]; simp
      rw [this] at hflag; exact Option.some.inj hflag
    subst hvv
    have hj : slotsOf (slotsOf (Γ0.push (V.sc t f)).size sT.out) sE.out = sE.n := by
      rw [hsz, ← hS1n, slotsOf_eq fT.2, kT, slotsOf_eq fE.2, kE]
    rw [hj]
    have h3 := tT (At (Γ0.push (.sc t f)) w) (by simpa using hS1d) (by rw [hTe]; exact fT)
      (nil_rule (by rintro _ _ ⟨rfl, rfl⟩; exact ⟨hsz.trans hS1n.symm, Γ0, t, f, rfl, hP, hv, hb⟩))
    rw [hTe] at h3
    dsimp only at h3
    by_cases htT : termsS sT.out = true
    · refine Hoare.conseq (Triple.terms h3 htT) (fun _ _ h => h) (fun _ _ h => h.elim)
        (fun _ _ _ _ h => h) (fun _ _ _ h => h)
    · refine Hoare.conseq h3 (fun _ _ h => h) ?_ (fun _ _ _ _ h => h) (fun _ _ _ h => h)
      rintro Γ' w' ⟨jv, rfl, hsz', hq⟩ vs hvs
      have hTE : sT.n ≤ sE.n := by have := slotsS_ge sE.out sT.n; rw [kE] at this; exact this
      refine ⟨?_, Γ', jv, vs, by omega, rfl, hvs, .inl hq⟩
      rw [bindAt_size', (mapM_len hvs).trans (Vals.slots_length jv)]
      simp [htT]
  · rintro _ w t f ⟨hsz, Γ0, v, rfl, hs0, hP, hv⟩ hflag hb
    have hvv : v = .sc t f := by
      have : (Γ0.push v)[s.n]? = some v := by rw [← hs0]; simp
      rw [this] at hflag; exact Option.some.inj hflag
    subst hvv
    have hT : slotsOf (Γ0.push (V.sc t f)).size sT.out = sT.n := by
      rw [hsz, ← hS1n, slotsOf_eq fT.2, kT]
    have hj : slotsOf (slotsOf (Γ0.push (V.sc t f)).size sT.out) sE.out = sE.n := by
      rw [hT, slotsOf_eq fE.2, kE]
    rw [hj, hT]
    have hge : Γ0.size + 1 ≤ sT.n := by
      have := slotsS_ge sT.out S1.n; rw [kT] at this; omega
    have h3 := tE (At (bindAt (Γ0.push (.sc t f)) sT.n []) w)
      (by rw [St.enter, St.leave_depth, hdT, hS1d]) (by rw [hEe]; exact fE)
      (nil_rule (by
        rintro _ _ ⟨rfl, rfl⟩
        exact ⟨by simp [bindAt_size'], Γ0, t, f, hge, rfl, hP, hv, by simpa using hb⟩))
    rw [hEe] at h3
    dsimp only at h3
    by_cases htE : termsS sE.out = true
    · refine Hoare.conseq (Triple.terms h3 htE) (fun _ _ h => h) (fun _ _ h => h.elim)
        (fun _ _ _ _ h => h) (fun _ _ _ h => h)
    · refine Hoare.conseq h3 (fun _ _ h => h) ?_ (fun _ _ _ _ h => h) (fun _ _ _ h => h)
      rintro Γ' w' ⟨jv, rfl, hsz', hq⟩ vs hvs
      refine ⟨?_, Γ', jv, vs, by omega, rfl, hvs, .inr hq⟩
      rw [bindAt_size', (mapM_len hvs).trans (Vals.slots_length jv)]
      simp [htE]

-- ---------------------------------------------------------------------------
-- Bottom-tested loops
-- ---------------------------------------------------------------------------

/-- What the state before a bottom-tested loop satisfies: `P`, or, where the
    loop has an entry guard, `P` of the state before the guard's test. -/
def guardPre (guardIdx : Option Nat) (P : Env → World → Prop) (cc : ICmpCond) (a b : R) :
    Env → World → Prop :=
  match guardIdx with
  | none => P
  | some _ => fun Γ w => ∃ Γ0 v, Γ = Γ0.push v ∧ P Γ0 w ∧ evalOp w.mem Γ0 (.icmp cc a b) = some v

/-- **A bottom-tested loop, by invariant.** Entering it (past the guard, if it
    has one) establishes `I`; a trip from `I` ends in `Qb`, and the back-edge
    test then either goes round with carries satisfying `I` or leaves with the
    carries `exitIdx` picks satisfying `Xit`. A `br` out of the body satisfies
    `Xit` too, and a `cont`, `I`. -/
theorem PT.dloop {α : Type} {cfg : Cfg} {d : Nat} {J : Post} {tys : List ClifTy} {tb : ClifTy}
    {init : Vals Slot tys} {cc : ICmpCond} {cb : Slot tb} {guardIdx : Option Nat}
    {hg : guardIdx.all (fun i => (tys[i]?).getD default == tb) = true} {contOnTrue : Bool}
    {exitIdx : List Nat}
    {body : Lvl (idxTys tys exitIdx) tys → Vals Slot tys → Prog Slot Lvl (Slot tb × Vals Slot tys)}
    {k : Vals Slot (idxTys tys exitIdx) → Prog Slot Lvl α}
    {P : Env → World → Prop} {Q : α → Env → World → Prop}
    (I : Env → List V → World → Prop)
    (Qb : Env → List V → Slot tb × Vals Slot tys → Env → World → Prop)
    (Xit : Env → Env → List V → World → Prop)
    (hinit : ∀ Γ w cs, guardPre guardIdx P cc ((init.slots[guardIdx.getD 0]?).getD 0) cb Γ w →
      init.slots.mapM (fun r => Γ[r]?) = some cs →
      match guardIdx with
      | none => I Γ cs w
      | some _ => ∀ t f, Γ[Γ.size - 1]? = some (.sc t f) →
          (((f != 0) == contOnTrue) = true → I Γ cs w) ∧
          (((f != 0) == contOnTrue) = false →
            ∀ outs, exitIdx.mapM (fun i => cs[i]?) = some outs → Xit Γ Γ outs w))
    (hbody : ∀ n0 Γ0 cs, PT cfg (d + 1) (loopJ J Γ0 I Xit (idxTys tys exitIdx) tys)
      (fun Γb wb => Γ0.size = n0 ∧ Γb = bindAt Γ0 n0 cs ∧ I Γ0 cs wb)
      (body d (carriesFrom n0 tys)) (Qb Γ0 cs))
    (hback : ∀ Γ0 cs a Γ2 w2 t f next, Qb Γ0 cs a Γ2 w2 →
      evalOp w2.mem Γ2 (.icmp cc a.1 cb) = some (.sc t f) →
      a.2.slots.mapM (fun r => (Γ2.push (.sc t f))[r]?) = some next →
      (((f != 0) == contOnTrue) = true → I Γ0 next w2) ∧
      (((f != 0) == contOnTrue) = false →
        ∀ outs, exitIdx.mapM (fun i => next[i]?) = some outs → Xit Γ0 (Γ2.push (.sc t f)) outs w2))
    (hk : ∀ nE, PT cfg d J (fun Γ w => ∃ Γ0 Γb outs, Γ0.size ≤ nE ∧ Γ = bindAt Γb nE outs ∧ Xit Γ0 Γb outs w)
      (k (carriesFrom nE (idxTys tys exitIdx))) Q)
    (hgf : ∀ gi, guardIdx = some gi → ∀ Γ w, P Γ w →
      evalOp w.mem Γ (.icmp cc ((init.slots[gi]?).getD 0) cb) = none → J.faultOk = true := by intros; rfl)
    (hbf : ∀ Γ0 cs a Γ2 w2, Qb Γ0 cs a Γ2 w2 →
      evalOp w2.mem Γ2 (.icmp cc a.1 cb) = none → J.faultOk = true := by intros; rfl) :
    PT cfg d J P (.dloop init cc cb guardIdx hg contOnTrue exitIdx body k) Q := by
  intro s P0 hd hfin hpre
  rw [emitGo_dloop] at hfin ⊢
  unfold emitDloop at hfin ⊢
  cases guardIdx
  case' none =>
    dsimp only at hfin ⊢
    have hP1 : Triple cfg P0 s.flush.out
        { J with ok := fun Γ w => Γ.size = s.flush.n ∧ guardPre none P cc 0 cb Γ w } := by
      rw [St.out_flush, St.flush_n]; exact hpre
    have hc1 := St.flush_cur s
    have hS1d : s.flush.depth = d := (St.flush_depth s).trans hd
    have hg0 : (none : Option R) = none := rfl
    generalize s.flush = S1 at hP1 hc1 hS1d hfin ⊢
  case' some gi =>
    dsimp only at hfin ⊢
    have hP1 := St.out_stmt (st := .op (.icmp cc ((init.slots[gi]?).getD 0) cb)) hpre (op_rule
      (Q := fun Γ w => Γ.size = s.n + 1 ∧ guardPre (some gi) P cc ((init.slots[gi]?).getD 0) cb Γ w)
      (fun Γ w v ⟨hn, hP⟩ hv => ⟨by simp [hn], Γ, v, rfl, hP, hv⟩)
      (fun Γ w ⟨_, hP⟩ hv => hgf gi rfl Γ w hP hv))
    have hP1' : Triple cfg P0 (s.bind1 (.op (.icmp cc ((init.slots[gi]?).getD 0) cb))).2.flush.out
        { J with ok := fun Γ w => Γ.size = (s.bind1 (.op (.icmp cc ((init.slots[gi]?).getD 0) cb))).2.flush.n ∧
          guardPre (some gi) P cc ((init.slots[gi]?).getD 0) cb Γ w } := by
      rw [St.out_flush, St.flush_n]; exact hP1
    clear hP1
    have hc1 := St.flush_cur (s.bind1 (.op (.icmp cc ((init.slots[gi]?).getD 0) cb))).2
    have hS1d : (s.bind1 (.op (.icmp cc ((init.slots[gi]?).getD 0) cb))).2.flush.depth = d :=
      (St.flush_depth _).trans hd
    have hg0 : (s.bind1 (.op (.icmp cc ((init.slots[gi]?).getD 0) cb))).1 + 1 =
        (s.bind1 (.op (.icmp cc ((init.slots[gi]?).getD 0) cb))).2.flush.n := by
      rw [St.flush_n]; rfl
    generalize (s.bind1 (.op (.icmp cc ((init.slots[gi]?).getD 0) cb))).2.flush = S1
      at hP1' hc1 hS1d hg0 hfin ⊢
    have hP1 := hP1'
    clear hP1'
  all_goals
    subst hS1d
    obtain ⟨bd, sB, hBe⟩ : ∃ a b, emitGo (body S1.depth (carriesFrom S1.n tys))
        ({ S1 with n := S1.n + tys.length, depth := S1.depth + 1 } : St).enter = (a, b) := ⟨_, _, rfl⟩
    have cB := emitGo_cnt (body S1.depth (carriesFrom S1.n tys))
      ({ S1 with n := S1.n + tys.length, depth := S1.depth + 1 } : St).enter
    have hdB := emitGo_depth (body S1.depth (carriesFrom S1.n tys))
      ({ S1 with n := S1.n + tys.length, depth := S1.depth + 1 } : St).enter
    have tB := hbody S1.n
    rw [hBe] at cB hdB
    simp only [hBe, St.leave_n] at hfin ⊢
    dsimp only at cB hdB
    have hQx : ∀ nE Γ0 Γb outs w', Γ0.size ≤ nE → outs.length = exitIdx.length → Xit Γ0 Γb outs w' →
        (bindAt Γb nE outs).size = nE + (idxTys tys exitIdx).length ∧
          ∃ Γ0' Γb' outs', Γ0'.size ≤ nE ∧ bindAt Γb nE outs = bindAt Γb' nE outs' ∧ Xit Γ0' Γb' outs' w' :=
      fun nE Γ0 Γb outs w' hle hl hx => ⟨by simp [bindAt_size', idxTys, hl], Γ0, Γb, outs, hle, rfl, hx⟩
    have hlenI : ∀ (Γ : Env) cs, init.slots.mapM (fun r => Γ[r]?) = some cs →
        cs.length = tys.length := fun _ _ h => (mapM_len h).trans (Vals.slots_length init)
    -- The body's triple from carries satisfying the invariant.
    have hB3 : ∀ Γ cs w, Γ.size = S1.n → cs.length = tys.length → I Γ cs w → Fine sB →
        Triple cfg (At (bindAt Γ S1.n cs) w) sB.out
          { loopJ J Γ I Xit (idxTys tys exitIdx) tys with
            ok := fun Γ' w' => ∃ a, bd = some a ∧ Γ'.size = sB.n ∧ Qb Γ cs a Γ' w' } := by
      intro Γ cs w hsz hlen hI hfB
      have h3 := tB Γ cs ({ S1 with n := S1.n + tys.length, depth := S1.depth + 1 } : St).enter
        (At (bindAt Γ S1.n cs) w) rfl (by rw [hBe]; exact hfB)
        (nil_rule (by
          rintro _ _ ⟨rfl, rfl⟩
          exact ⟨by simp [bindAt_size', hlen], hsz, rfl, hI⟩))
      rw [hBe] at h3
      exact h3
    have hbrk : ∀ Γ (ab : Nat) (e : Nat) Γb vs w', Γ.size ≤ ab →
        (loopJ J Γ I Xit (idxTys tys exitIdx) tys).brk e Γb vs w' →
        (match e with
          | 0 => (bindAt Γb ab vs).size = ab + (idxTys tys exitIdx).length ∧
              ∃ Γ0' Γb' outs', Γ0'.size ≤ ab ∧ bindAt Γb ab vs = bindAt Γb' ab outs' ∧ Xit Γ0' Γb' outs' w'
          | e + 1 => J.brk e Γb vs w') := by
      rintro Γ ab (_ | e) Γb vs w' hle he
      · exact hQx _ _ _ _ _ hle (he.1.trans (by simp [idxTys])) he.2
      · exact he
    rcases bd with _ | ⟨ca, vs⟩
    · -- The body never answers.
      dsimp only at hfin ⊢
      refine hk _ _ P0 rfl hfin ?_
      have fX := (emitGo_cnt _ _).fine hfin
      have fP := Fine.of_push (S1 := S1) fX rfl hc1 hc1
      have hlt := Nat.lt_of_lt_of_le (needP_lt S1.out _) fP.2
      simp only [needP_dloop, St.leave_fst] at hlt
      have fB : Fine sB := ⟨by simpa using fP.1, by omega⟩
      have kB : slotsS (S1.n + tys.length) sB.out = sB.n := cB.2.2 fB.1 fB.2 _ rfl
      refine Triple.of_push (S1 := S1) ?_ rfl hc1 hc1
      refine append_rule hP1 (cons_rule (dloop_rule (fun Γ cs w => cs.length = tys.length ∧ I Γ cs w)
        ?_ ?_) (nil_rule fun _ _ h => h))
      · first
        | (rintro Γ w cs ⟨_, hP⟩ hcs; exact ⟨hlenI Γ cs hcs, hinit Γ w cs hP hcs⟩)
        | (rintro Γ w cs ⟨hsz, hP⟩ hcs t f hgv
           have hu := hinit Γ w cs hP hcs t f (by
             rw [show Γ.size - 1 = (s.bind1 (.op (.icmp cc ((init.slots[gi]?).getD 0) cb))).1 by
               rw [hsz, ← hg0, Nat.add_sub_cancel]]
             exact hgv)
           refine ⟨fun hc => ⟨hlenI Γ cs hcs, hu.1 hc⟩, fun hc outs houts => ?_⟩
           simp only [St.leave_fst]; rw [hsz, slotsOf_eq fB.2, kB]
           exact hQx sB.n Γ Γ outs w (by have := slotsS_ge sB.out (S1.n + tys.length); omega) (mapM_len houts) (hu.2 hc outs houts))
      · rintro Γ w0 cs w ⟨hsz, _⟩ ⟨hlen, hI⟩
        simp only [St.leave_fst]; rw [hsz, slotsOf_eq fB.2, kB]
        refine Hoare.conseq (hB3 Γ cs w hsz hlen hI fB) (fun _ _ h => h)
          (fun _ _ ⟨_, h, _⟩ => nomatch h) (fun e Γb vs w' he => hbrk Γ sB.n e Γb vs w' (by have := slotsS_ge sB.out (S1.n + tys.length); omega) he) ?_
        rintro (_ | e) vs w' he <;> exact he
    · dsimp only at hfin ⊢
      by_cases htm : terminates
          (St.leave ({ S1 with n := S1.n + tys.length, depth := S1.depth + 1 } : St) sB).1 = true
      · -- The body always leaves: there is no back edge.
        rw [if_pos htm] at hfin ⊢
        refine hk _ _ P0 rfl hfin ?_
        have fX := (emitGo_cnt _ _).fine hfin
        have fP := Fine.of_push (S1 := S1) fX rfl hc1 hc1
        have hlt := Nat.lt_of_lt_of_le (needP_lt S1.out _) fP.2
        simp only [needP_dloop, St.leave_fst] at hlt
        have fB : Fine sB := ⟨by simpa using fP.1, by omega⟩
        have kB : slotsS (S1.n + tys.length) sB.out = sB.n := cB.2.2 fB.1 fB.2 _ rfl
        have hts : termsS sB.out = true := by
          rw [← terminates_eq fB.2]; simpa using htm
        refine Triple.of_push (S1 := S1) ?_ rfl hc1 hc1
        refine append_rule hP1 (cons_rule (dloop_rule (fun Γ cs w => cs.length = tys.length ∧ I Γ cs w)
          ?_ ?_) (nil_rule fun _ _ h => h))
        · first
          | (rintro Γ w cs ⟨_, hP⟩ hcs; exact ⟨hlenI Γ cs hcs, hinit Γ w cs hP hcs⟩)
          | (rintro Γ w cs ⟨hsz, hP⟩ hcs t f hgv
             have hu := hinit Γ w cs hP hcs t f (by
               rw [show Γ.size - 1 = (s.bind1 (.op (.icmp cc ((init.slots[gi]?).getD 0) cb))).1 by
                 rw [hsz, ← hg0, Nat.add_sub_cancel]]
               exact hgv)
             refine ⟨fun hc => ⟨hlenI Γ cs hcs, hu.1 hc⟩, fun hc outs houts => ?_⟩
             simp only [St.leave_fst]; rw [hsz, slotsOf_eq fB.2, kB]
             exact hQx sB.n Γ Γ outs w (by have := slotsS_ge sB.out (S1.n + tys.length); omega) (mapM_len houts) (hu.2 hc outs houts))
        · rintro Γ w0 cs w ⟨hsz, _⟩ ⟨hlen, hI⟩
          simp only [St.leave_fst]; rw [hsz, slotsOf_eq fB.2, kB]
          refine Hoare.conseq (Triple.terms (hB3 Γ cs w hsz hlen hI fB) hts) (fun _ _ h => h)
            (fun _ _ h => h.elim) (fun e Γb vs w' he => hbrk Γ sB.n e Γb vs w' (by have := slotsS_ge sB.out (S1.n + tys.length); omega) he) ?_
          rintro (_ | e) vs w' he <;> exact he
      · -- The back-edge test.
        rw [if_neg htm] at hfin ⊢
        have cF := Cnt.bind1 sB (.op (.icmp cc ca cb)) rfl
        have hFo : (sB.bind1 (.op (.icmp cc ca cb))).2.out = (sB.stmt (.op (.icmp cc ca cb))).out := rfl
        have hFf : (sB.bind1 (.op (.icmp cc ca cb))).1 = sB.n := rfl
        generalize sB.bind1 (.op (.icmp cc ca cb)) = rF at cF hFo hFf hfin ⊢
        obtain ⟨fl, sB'⟩ := rF
        dsimp only at cF hFo hFf hfin ⊢
        subst hFf
        refine hk _ _ P0 rfl hfin ?_
        have fX := (emitGo_cnt _ _).fine hfin
        have fP := Fine.of_push (S1 := S1) fX rfl hc1 hc1
        have hlt := Nat.lt_of_lt_of_le (needP_lt S1.out _) fP.2
        simp only [needP_dloop, St.leave_fst] at hlt
        have fB' : Fine sB' := ⟨by simpa using fP.1, by omega⟩
        have fB : Fine sB := cF.fine fB'
        have kB : slotsS (S1.n + tys.length) sB.out = sB.n := cB.2.2 fB.1 fB.2 _ rfl
        have kB' : slotsS (S1.n + tys.length) sB'.out = sB'.n := cF.2.2 fB'.1 fB'.2 _ kB
        refine Triple.of_push (S1 := S1) ?_ rfl hc1 hc1
        refine append_rule hP1 (cons_rule (dloop_rule (fun Γ cs w => cs.length = tys.length ∧ I Γ cs w)
          ?_ ?_) (nil_rule fun _ _ h => h))
        · first
          | (rintro Γ w cs ⟨_, hP⟩ hcs; exact ⟨hlenI Γ cs hcs, hinit Γ w cs hP hcs⟩)
          | (rintro Γ w cs ⟨hsz, hP⟩ hcs t f hgv
             have hu := hinit Γ w cs hP hcs t f (by
               rw [show Γ.size - 1 = (s.bind1 (.op (.icmp cc ((init.slots[gi]?).getD 0) cb))).1 by
                 rw [hsz, ← hg0, Nat.add_sub_cancel]]
               exact hgv)
             refine ⟨fun hc => ⟨hlenI Γ cs hcs, hu.1 hc⟩, fun hc outs houts => ?_⟩
             simp only [St.leave_fst]; rw [hsz, slotsOf_eq fB'.2, kB']
             exact hQx sB'.n Γ Γ outs w (by have := slotsS_ge sB'.out (S1.n + tys.length); omega) (mapM_len houts) (hu.2 hc outs houts))
        · rintro Γ w0 cs w ⟨hsz, _⟩ ⟨hlen, hI⟩
          simp only [St.leave_fst]; rw [hsz, slotsOf_eq fB'.2, kB']
          have h3' := St.out_stmt (st := .op (.icmp cc ca cb)) (hB3 Γ cs w hsz hlen hI fB) (op_rule
            (Q := fun Γ' w' => ∃ Γ2 v, Γ' = Γ2.push v ∧ Γ2.size = sB.n ∧ Qb Γ cs (ca, vs) Γ2 w' ∧
              evalOp w'.mem Γ2 (.icmp cc ca cb) = some v)
            (fun Γ2 w2 v ⟨a, ha, hs2, hq⟩ hv => by cases ha; exact ⟨Γ2, v, rfl, hs2, hq, hv⟩)
            (fun Γ2 w2 ⟨a, ha, _, hq⟩ hv => by cases ha; exact hbf Γ cs _ Γ2 w2 hq hv))
          rw [← hFo] at h3'
          refine Hoare.conseq h3' (fun _ _ h => h) ?_ (fun e Γb vs w' he => hbrk Γ sB'.n e Γb vs w' (by have := slotsS_ge sB'.out (S1.n + tys.length); omega) he) ?_
          · rintro _ w2 ⟨Γ2, v, rfl, hs2, hq, hv⟩
            have hvv : ∀ t f, (Γ2.push v)[sB.n]? = some (.sc t f) → v = .sc t f := by
              intro t f h
              have : (Γ2.push v)[sB.n]? = some v := by rw [← hs2]; simp
              rw [this] at h; exact Option.some.inj h
            refine ⟨fun next hnext t f hflag hc => ?_, fun next hnext t f hflag hc outs houts => ?_⟩
            · have e := hvv t f hflag; subst e
              exact ⟨(mapM_len hnext).trans (Vals.slots_length vs),
                (hback Γ cs (ca, vs) Γ2 w2 t f next hq hv hnext).1 hc⟩
            · have e := hvv t f hflag; subst e
              exact hQx sB'.n Γ _ outs w2 (by have := slotsS_ge sB'.out (S1.n + tys.length); omega) (mapM_len houts)
                ((hback Γ cs (ca, vs) Γ2 w2 t f next hq hv hnext).2 hc outs houts)
          · rintro (_ | e) vs w' he <;> exact he

-- ---------------------------------------------------------------------------
-- The remaining statements, and weakest preconditions
-- ---------------------------------------------------------------------------

section rules2
variable {α : Type} {cfg : Cfg} {d : Nat} {J : Post}

/-- A call to one of the program's own functions that answers. -/
theorem PT.callLocal {ps : List ClifTy} {res : Option ClifTy} {r : LocalRef ps res}
    {args : Vals Slot ps} {k : ResV Slot res → Prog Slot Lvl α}
    {P : Env → World → Prop} {R : Nat → Env → World → Prop} {Q : α → Env → World → Prop}
    (hres : res.isSome = true)
    (hs : ∀ n, Stmt1 cfg (fun Γ w => Γ.size = n ∧ P Γ w) (.call r.callee args.slots)
      (fun Γ w => Γ.size = n + 1 ∧ R n Γ w) J.faultOk)
    (hk : ∀ v, PT cfg d J (R v) (k (resSlot res v)) Q) :
    PT cfg d J P (.callLocal r args k) Q := by
  intro s P0 hd hf h
  rw [emitGo_callLocal] at hf ⊢; unfold emitCallLocal at hf ⊢
  simp only [hres, if_true] at hf ⊢
  cases hc : r.callee <;> simp only [hc] at hf ⊢ <;> rw [hc] at hs <;>
    exact hk _ _ P0 hd hf (St.out_stmt h (hs s.n))

/-- A call to one of the program's own functions that answers nothing. -/
theorem PT.callLocalVoid {ps : List ClifTy} {res : Option ClifTy} {r : LocalRef ps res}
    {args : Vals Slot ps} {k : ResV Slot res → Prog Slot Lvl α}
    {P R : Env → World → Prop} {Q : α → Env → World → Prop}
    (hres : res.isSome = false)
    (hs : ∀ n, Stmt1 cfg (fun Γ w => Γ.size = n ∧ P Γ w) (.callVoid r.callee args.slots)
      (fun Γ w => Γ.size = n ∧ R Γ w) J.faultOk)
    (hk : PT cfg d J R (k (resSlot res 0)) Q) :
    PT cfg d J P (.callLocal r args k) Q := by
  intro s P0 hd hf h
  rw [emitGo_callLocal] at hf ⊢; unfold emitCallLocal at hf ⊢
  simp only [hres] at hf ⊢
  cases hc : r.callee <;> simp only [hc] at hf ⊢ <;> rw [hc] at hs <;>
    exact hk _ P0 hd hf (St.out_stmt h (hs s.n))

end rules2

section rules3
variable {α : Type} {cfg : Cfg} {d : Nat} {J : Post}

theorem PT.storeUnaligned {ty} {P R : Env → World → Prop} {v : Slot ty} {a : Slot .i64}
    {k : Prog Slot Lvl α} {Q : α → Env → World → Prop}
    (hs : ∀ n, Stmt1 cfg (fun Γ w => Γ.size = n ∧ P Γ w) (.storeUnaligned v a)
      (fun Γ w => Γ.size = n ∧ R Γ w) J.faultOk)
    (hk : PT cfg d J R k Q) : PT cfg d J P (.storeUnaligned v a k) Q := by
  intro s P0 hd hf h; rw [emitGo_storeUnaligned] at hf ⊢; exact PT.stmt hs hk s P0 hd hf h

theorem PT.istore8 {ty} {P R : Env → World → Prop} {v : Slot ty} {a : Slot .i64}
    {hi : ty.isInt = true} {k : Prog Slot Lvl α} {Q : α → Env → World → Prop}
    (hs : ∀ n, Stmt1 cfg (fun Γ w => Γ.size = n ∧ P Γ w) (.istore8 v a)
      (fun Γ w => Γ.size = n ∧ R Γ w) J.faultOk)
    (hk : PT cfg d J R k Q) : PT cfg d J P (.istore8 v a hi k) Q := by
  intro s P0 hd hf h; rw [emitGo_istore8] at hf ⊢; exact PT.stmt hs hk s P0 hd hf h

end rules3

/-- From nowhere, anything: code no run reaches the start of never misuses. -/
theorem PT.false {cfg : Cfg} {α : Type} (p : Prog Slot Lvl α) :
    ∀ {d : Nat} {J : Post} {Q : α → Env → World → Prop}, PT cfg d J (fun _ _ => False) p Q := by
  induction p with
  | ret a => exact PT.ret fun _ _ h => h.elim
  | op o k ih =>
      exact PT.op (fun r => PT.conseq (ih r) (fun _ _ ⟨_, _, _, _, h, _⟩ => h) (fun _ _ _ h => h)) (fun _ _ h => h.elim)
  | store v a k ih => exact PT.store (R := fun _ _ => False) (fun _ _ _ ⟨_, h⟩ => h.elim) ih
  | storeUnaligned v a k ih =>
      exact PT.storeUnaligned (R := fun _ _ => False) (fun _ _ _ ⟨_, h⟩ => h.elim) ih
  | istore8 v a hi k ih => exact PT.istore8 (R := fun _ _ => False) (fun _ _ _ ⟨_, h⟩ => h.elim) ih
  | call f args k ih =>
      cases hres : f.result.isSome
      · exact PT.callVoid (R := fun _ _ => False) hres (fun _ _ _ ⟨_, h⟩ => h.elim) (ih _)
      · exact PT.call (R := fun _ _ _ => False) hres (fun _ _ _ ⟨_, h⟩ => h.elim) fun _ => ih _
  | @callLocal ps res β r args k ih =>
      cases hres : res.isSome
      · exact PT.callLocalVoid (R := fun _ _ => False) hres (fun _ _ _ ⟨_, h⟩ => h.elim) (ih _)
      · exact PT.callLocal (R := fun _ _ _ => False) hres (fun _ _ _ ⟨_, h⟩ => h.elim) fun _ => ih _
  | loop init head body k ihH ihB ihK =>
      exact PT.loop (fun _ _ _ => False) (fun _ _ _ _ _ => False) (fun _ _ _ _ => False)
        (fun _ _ _ h _ => h)
        (fun _ _ _ => PT.conseq (ihH _ _) (fun _ _ ⟨_, _, h⟩ => h) (fun _ _ _ h => h))
        (fun _ _ _ _ _ _ _ _ h => h.elim)
        (fun _ _ _ _ => PT.conseq (ihB _ _ _) (fun _ _ ⟨_, _, _, _, _, h, _⟩ => h) (fun _ _ _ h => h))
        (fun _ => PT.conseq (ihK _) (fun _ _ ⟨_, _, _, _, _, h⟩ => h) (fun _ _ _ h => h))
        (fun _ _ _ _ _ h => h.elim)
  | dloop init cc cb g hg c e body k ihB ihK =>
      refine PT.dloop (fun _ _ _ => False) (fun _ _ _ _ _ => False) (fun _ _ _ _ => False)
        ?_
        (fun _ _ _ => PT.conseq (ihB _ _) (fun _ _ ⟨_, _, h⟩ => h) (fun _ _ _ h => h))
        (fun _ _ _ _ _ _ _ _ h => h.elim)
        (fun _ => PT.conseq (ihK _) (fun _ _ ⟨_, _, _, _, _, h⟩ => h) (fun _ _ _ h => h))
        (fun _ _ _ _ h => h.elim) (fun _ _ _ _ _ h => h.elim)
      intro Γ w cs hP _
      cases g <;> simp [guardPre] at hP
  | ite c thn els k ihT ihE ihK =>
      exact PT.ite (fun _ _ _ => False) (fun _ _ _ => False)
        (PT.conseq ihT (fun _ _ ⟨_, _, _, _, h, _⟩ => h) (fun _ _ _ h => h))
        (fun _ => PT.conseq ihE (fun _ _ ⟨_, _, _, _, _, h, _⟩ => h) (fun _ _ _ h => h))
        (fun _ => PT.conseq (ihK _) (fun _ _ ⟨_, _, _, _, _, _, h⟩ => h.elim id id) (fun _ _ _ h => h))
        (fun _ _ h => h.elim)
  | params tys k ih => exact PT.params (ih _)
  | br l args => exact PT.br fun _ _ _ h => h.elim
  | cont l args => exact PT.cont fun _ _ _ h => h.elim

theorem runStmt_store_env {cfg : Cfg} {Γ Γ' : Env} {w w' : World} {ty : ClifTy} {v a : R}
    (h : runStmt cfg Γ w (.store ty v a) = .ok Γ' w') : Γ' = Γ := by
  rw [runStmt_store] at h; revert h; dsimp only
  (repeat' split) <;> intro h <;> first | (cases h; rfl) | cases h

theorem runStmt_storeUnaligned_env {cfg : Cfg} {Γ Γ' : Env} {w w' : World} {v a : R}
    (h : runStmt cfg Γ w (.storeUnaligned v a) = .ok Γ' w') : Γ' = Γ := by
  rw [runStmt_storeUnaligned] at h; revert h; dsimp only
  (repeat' split) <;> intro h <;> first | (cases h; rfl) | cases h

theorem runStmt_istore8_env {cfg : Cfg} {Γ Γ' : Env} {w w' : World} {v a : R}
    (h : runStmt cfg Γ w (.istore8 v a) = .ok Γ' w') : Γ' = Γ := by
  rw [runStmt_istore8] at h; revert h; dsimp only
  (repeat' split) <;> intro h <;> first | (cases h; rfl) | cases h

/-- A precondition that holds only under a closed condition is proved under
    that condition. -/
theorem PT.gate {cfg : Cfg} {d : Nat} {J : Post} {α : Type} {G : Prop} {P : Env → World → Prop}
    {p : Prog Slot Lvl α} {Q : α → Env → World → Prop} (h : G → PT cfg d J P p Q) :
    PT cfg d J (fun Γ w => P Γ w ∧ G) p Q := by
  by_cases hg : G
  · exact PT.conseq (h hg) (fun _ _ h => h.1) (fun _ _ _ h => h)
  · exact PT.conseq (PT.false p) (fun _ _ h => hg h.2) (fun _ _ _ h => h)

/-- **The weakest precondition** of a body for `Q`, at loop depth `d` with the
    enclosing loops' exits `J`: what must hold of the environment and the world
    for the code emitted from it to never misuse and to end normally only in
    `Q`. A value a statement binds is named by the environment's size, which is
    the slot the emitter gave it, since `PT`'s precondition fixes the two equal.
    A loop's is that some invariant holds on entry and is kept by every trip,
    and a branch's that some postcondition of each arm leads on; each is what
    that construct's rule asks, with the construct's own code by its weakest
    precondition. -/
noncomputable def wpR (cfg : Cfg) {α : Type} (p : Prog Slot Lvl α) :
    Nat → Post → (α → Env → World → Prop) → Env → World → Prop :=
  Prog.rec (motive := fun β _ => Nat → Post → (β → Env → World → Prop) → Env → World → Prop)
    (ret := fun a _ _ Q => Q a)
    (op := fun o _ ih d J Q Γ w => (evalOp w.mem Γ o.erase = none → J.faultOk = true) ∧
      ∀ v, evalOp w.mem Γ o.erase = some v → ih Γ.size d J Q (Γ.push v) w)
    (store := fun {ty _} v a _ ih d J Q Γ w =>
      (∀ m, runStmt cfg Γ w (.store ty v a) = .fault m → J.faultOk = true) ∧
      ∀ w', runStmt cfg Γ w (.store ty v a) = .ok Γ w' → ih d J Q Γ w')
    (storeUnaligned := fun v a _ ih d J Q Γ w =>
      (∀ m, runStmt cfg Γ w (.storeUnaligned v a) = .fault m → J.faultOk = true) ∧
      ∀ w', runStmt cfg Γ w (.storeUnaligned v a) = .ok Γ w' → ih d J Q Γ w')
    (istore8 := fun v a _ _ ih d J Q Γ w =>
      (∀ m, runStmt cfg Γ w (.istore8 v a) = .fault m → J.faultOk = true) ∧
      ∀ w', runStmt cfg Γ w (.istore8 v a) = .ok Γ w' → ih d J Q Γ w')
    (call := fun f args _ ih d J Q Γ w => ∀ vs, args.slots.mapM (fun r => Γ[r]?) = some vs →
      ∃ r w', callOf cfg.locals (.ffi f) vs (obsCall w (.ffi f) vs) = some (r, w') ∧
        if f.result.isSome then ∀ v, r = some v → ih (resSlot f.result Γ.size) d J Q (Γ.push v) w'
        else ih (resSlot f.result 0) d J Q Γ w')
    (callLocal := fun {_ res _} r args _ ih d J Q Γ w =>
      ∀ vs, args.slots.mapM (fun r => Γ[r]?) = some vs →
      (callOf cfg.locals r.callee vs (obsCall w r.callee vs) = none →
        FailOk (failOf cfg.locals r.callee vs (obsCall w r.callee vs)) J.faultOk) ∧
      ∀ x w', callOf cfg.locals r.callee vs (obsCall w r.callee vs) = some (x, w') →
        if res.isSome then ∀ v, x = some v → ih (resSlot res Γ.size) d J Q (Γ.push v) w'
        else ih (resSlot res 0) d J Q Γ w')
    (loop := fun {tys exitTys β _} init _ _ _ ihH ihB ihK d J Q Γ w =>
      ∃ (I : Env → List V → World → Prop)
        (Qh : Env → List V → Cond Slot × Vals Slot exitTys × β → Env → World → Prop)
        (Xit : Env → Env → List V → World → Prop),
        (∀ cs, init.slots.mapM (fun r => Γ[r]?) = some cs → I Γ cs w) ∧
        ((∀ n0 Γ0 cs wh, Γ0.size = n0 → I Γ0 cs wh →
          ihH d (carriesFrom n0 tys) (d + 1) (loopJ J Γ0 I Xit exitTys tys) (Qh Γ0 cs)
            (bindAt Γ0 n0 cs) wh) ∧
        (∀ Γ0 cs a Γ1 w1 t f vs, Qh Γ0 cs a Γ1 w1 →
          evalOp w1.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) = some (.sc t f) →
          ((f != 0) == a.1.exitOnTrue) = true →
          a.2.1.slots.mapM (fun r => (Γ1.push (.sc t f))[r]?) = some vs →
          Xit Γ0 (Γ1.push (.sc t f)) vs w1) ∧
        (∀ nb Γ0 cs a Γ1 t f wb, Γ1.size + 1 = nb → Qh Γ0 cs a Γ1 wb →
          evalOp wb.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) = some (.sc t f) →
          ((f != 0) == a.1.exitOnTrue) = false →
          ihB d (carriesFrom nb tys) a.2.2 (d + 1) (loopJ J Γ0 I Xit exitTys tys)
            (fun next Γ2 w2 => ∀ nx, next.slots.mapM (fun r => Γ2[r]?) = some nx → I Γ0 nx w2)
            (bindAt (Γ1.push (.sc t f)) nb cs) wb) ∧
        (∀ nE Γ0 Γb vs w', Γ0.size ≤ nE → Xit Γ0 Γb vs w' →
          ihK (carriesFrom nE exitTys) d J Q (bindAt Γb nE vs) w') ∧
        (∀ Γ0 cs a Γ1 w1, Qh Γ0 cs a Γ1 w1 →
          evalOp w1.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) = none → J.faultOk = true)))
    (dloop := fun {tys _ tb} init cc cb guardIdx _ contOnTrue exitIdx _ _ ihB ihK d J Q Γ w =>
      ∃ (I : Env → List V → World → Prop)
        (Qb : Env → List V → Slot tb × Vals Slot tys → Env → World → Prop)
        (Xit : Env → Env → List V → World → Prop),
        ((∀ gi, guardIdx = some gi →
          evalOp w.mem Γ (.icmp cc ((init.slots[gi]?).getD 0) cb) = none → J.faultOk = true) ∧
        (∀ Γ' w' cs, guardPre guardIdx (At Γ w) cc ((init.slots[guardIdx.getD 0]?).getD 0) cb Γ' w' →
          init.slots.mapM (fun r => Γ'[r]?) = some cs →
          match guardIdx with
          | none => I Γ' cs w'
          | some _ => ∀ t f, Γ'[Γ'.size - 1]? = some (.sc t f) →
              (((f != 0) == contOnTrue) = true → I Γ' cs w') ∧
              (((f != 0) == contOnTrue) = false →
                ∀ outs, exitIdx.mapM (fun i => cs[i]?) = some outs → Xit Γ' Γ' outs w'))) ∧
        ((∀ n0 Γ0 cs wb, Γ0.size = n0 → I Γ0 cs wb →
          ihB d (carriesFrom n0 tys) (d + 1) (loopJ J Γ0 I Xit (idxTys tys exitIdx) tys) (Qb Γ0 cs)
            (bindAt Γ0 n0 cs) wb) ∧
        (∀ Γ0 cs a Γ2 w2 t f next, Qb Γ0 cs a Γ2 w2 →
          evalOp w2.mem Γ2 (.icmp cc a.1 cb) = some (.sc t f) →
          a.2.slots.mapM (fun r => (Γ2.push (.sc t f))[r]?) = some next →
          (((f != 0) == contOnTrue) = true → I Γ0 next w2) ∧
          (((f != 0) == contOnTrue) = false →
            ∀ outs, exitIdx.mapM (fun i => next[i]?) = some outs →
              Xit Γ0 (Γ2.push (.sc t f)) outs w2)) ∧
        (∀ nE Γ0 Γb outs w', Γ0.size ≤ nE → Xit Γ0 Γb outs w' →
          ihK (carriesFrom nE (idxTys tys exitIdx)) d J Q (bindAt Γb nE outs) w') ∧
        (∀ Γ0 cs a Γ2 w2, Qb Γ0 cs a Γ2 w2 →
          evalOp w2.mem Γ2 (.icmp cc a.1 cb) = none → J.faultOk = true)))
    (ite := fun {jTys _} c _ _ _ ihT ihE ihK d J Q Γ w =>
      ∃ (Qt Qe : Vals Slot jTys → Env → World → Prop),
        ((∀ t f, evalOp w.mem Γ (.icmp c.cc c.a c.b) = some (.sc t f) → (f != 0) = true →
          ihT d J Qt (Γ.push (.sc t f)) w) ∧
        (∀ ne t f, Γ.size + 1 ≤ ne → evalOp w.mem Γ (.icmp c.cc c.a c.b) = some (.sc t f) →
          (f != 0) = false → ihE d J Qe (bindAt (Γ.push (.sc t f)) ne []) w) ∧
        (evalOp w.mem Γ (.icmp c.cc c.a c.b) = none → J.faultOk = true)) ∧
        (∀ nJ Γ' jv vs w', Γ'.size ≤ nJ → jv.slots.mapM (fun r => Γ'[r]?) = some vs →
          (Qt jv Γ' w' ∨ Qe jv Γ' w') →
          ihK (carriesFrom nJ jTys) d J Q (bindAt Γ' nJ vs) w'))
    (params := fun tys _ ih d J Q => ih (carriesFrom 0 tys) d J Q)
    (br := fun l args d J _ Γ w => ∀ vs, args.slots.mapM (fun r => Γ[r]?) = some vs →
      J.brk (labelDepth d l) Γ vs w)
    (cont := fun l args d J _ Γ w => ∀ vs, args.slots.mapM (fun r => Γ[r]?) = some vs →
      J.cont (labelDepth d l) vs w)
    p

@[inherit_doc wpR]
noncomputable def wp (cfg : Cfg) (d : Nat) (J : Post) {α : Type} (p : Prog Slot Lvl α) :
    (α → Env → World → Prop) → Env → World → Prop :=
  wpR cfg p d J

section wp_eqns
variable (cfg : Cfg) (d : Nat) (J : Post) {α β : Type}

theorem wp_ret (a : α) (Q : α → Env → World → Prop) : wp cfg d J (.ret a) Q = Q a := rfl
theorem wp_pure (a : α) (Q : α → Env → World → Prop) :
    wp cfg d J (pure a : Prog Slot Lvl α) Q = Q a := rfl
theorem wp_op {ty} (o : Op' Slot ty) (k : Slot ty → Prog Slot Lvl α) (Q : α → Env → World → Prop) :
    wp cfg d J (.op o k) Q =
      fun Γ w => (evalOp w.mem Γ o.erase = none → J.faultOk = true) ∧
        ∀ v, evalOp w.mem Γ o.erase = some v → wp cfg d J (k Γ.size) Q (Γ.push v) w := rfl
theorem wp_call (f : Ffi) (args : Vals Slot f.params) (k : ResV Slot f.result → Prog Slot Lvl α)
    (Q : α → Env → World → Prop) :
    wp cfg d J (.call f args k) Q = fun Γ w => ∀ vs, args.slots.mapM (fun r => Γ[r]?) = some vs →
      ∃ r w', callOf cfg.locals (.ffi f) vs (obsCall w (.ffi f) vs) = some (r, w') ∧
        if f.result.isSome then ∀ v, r = some v → wp cfg d J (k (resSlot f.result Γ.size)) Q (Γ.push v) w'
        else wp cfg d J (k (resSlot f.result 0)) Q Γ w' := rfl
theorem wp_store {ty} (v : Slot ty) (a : Slot .i64) (k : Prog Slot Lvl α) (Q : α → Env → World → Prop) :
    wp cfg d J (.store v a k) Q =
      fun Γ w => (∀ m, runStmt cfg Γ w (.store ty v a) = .fault m → J.faultOk = true) ∧
        ∀ w', runStmt cfg Γ w (.store ty v a) = .ok Γ w' → wp cfg d J k Q Γ w' := rfl
theorem wp_storeUnaligned {ty} (v : Slot ty) (a : Slot .i64) (k : Prog Slot Lvl α)
    (Q : α → Env → World → Prop) :
    wp cfg d J (.storeUnaligned v a k) Q =
      fun Γ w => (∀ m, runStmt cfg Γ w (.storeUnaligned v a) = .fault m → J.faultOk = true) ∧
        ∀ w', runStmt cfg Γ w (.storeUnaligned v a) = .ok Γ w' → wp cfg d J k Q Γ w' := rfl
theorem wp_istore8 {ty} (v : Slot ty) (a : Slot .i64) (hi : ty.isInt = true) (k : Prog Slot Lvl α)
    (Q : α → Env → World → Prop) :
    wp cfg d J (.istore8 v a hi k) Q =
      fun Γ w => (∀ m, runStmt cfg Γ w (.istore8 v a) = .fault m → J.faultOk = true) ∧
        ∀ w', runStmt cfg Γ w (.istore8 v a) = .ok Γ w' → wp cfg d J k Q Γ w' := rfl
theorem wp_callLocal {ps res} (r : LocalRef ps res) (args : Vals Slot ps) (k : ResV Slot res → Prog Slot Lvl α)
    (Q : α → Env → World → Prop) :
    wp cfg d J (.callLocal r args k) Q = fun Γ w => ∀ vs, args.slots.mapM (fun r => Γ[r]?) = some vs →
      (callOf cfg.locals r.callee vs (obsCall w r.callee vs) = none →
        FailOk (failOf cfg.locals r.callee vs (obsCall w r.callee vs)) J.faultOk) ∧
      ∀ x w', callOf cfg.locals r.callee vs (obsCall w r.callee vs) = some (x, w') →
        if res.isSome then ∀ v, x = some v → wp cfg d J (k (resSlot res Γ.size)) Q (Γ.push v) w'
        else wp cfg d J (k (resSlot res 0)) Q Γ w' := rfl
theorem wp_params (tys : List ClifTy) (k : Vals Slot tys → Prog Slot Lvl α) (Q : α → Env → World → Prop) :
    wp cfg d J (.params tys k) Q = wp cfg d J (k (carriesFrom 0 tys)) Q := rfl
theorem wp_br {ex ca} (l : Lvl ex ca) (args : Vals Slot ex) (Q : α → Env → World → Prop) :
    wp cfg d J (.br l args) Q = fun Γ w => ∀ vs, args.slots.mapM (fun r => Γ[r]?) = some vs →
      J.brk (labelDepth d l) Γ vs w := rfl
theorem wp_cont {ex ca} (l : Lvl ex ca) (args : Vals Slot ca) (Q : α → Env → World → Prop) :
    wp cfg d J (.cont l args) Q = fun Γ w => ∀ vs, args.slots.mapM (fun r => Γ[r]?) = some vs →
      J.cont (labelDepth d l) vs w := rfl

theorem wp_loop {tys exitTys : List ClifTy} {γ : Type} (init : Vals Slot tys)
    (head : Lvl exitTys tys → Vals Slot tys → Prog Slot Lvl (Cond Slot × Vals Slot exitTys × γ))
    (body : Lvl exitTys tys → Vals Slot tys → γ → Prog Slot Lvl (Vals Slot tys))
    (k : Vals Slot exitTys → Prog Slot Lvl α) (Q : α → Env → World → Prop) :
    wp cfg d J (.loop init head body k) Q = fun Γ w =>
      ∃ (I : Env → List V → World → Prop)
        (Qh : Env → List V → Cond Slot × Vals Slot exitTys × γ → Env → World → Prop)
        (Xit : Env → Env → List V → World → Prop),
        (∀ cs, init.slots.mapM (fun r => Γ[r]?) = some cs → I Γ cs w) ∧
        ((∀ n0 Γ0 cs wh, Γ0.size = n0 → I Γ0 cs wh →
          wp cfg (d + 1) (loopJ J Γ0 I Xit exitTys tys) (head d (carriesFrom n0 tys)) (Qh Γ0 cs)
            (bindAt Γ0 n0 cs) wh) ∧
        (∀ Γ0 cs a Γ1 w1 t f vs, Qh Γ0 cs a Γ1 w1 →
          evalOp w1.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) = some (.sc t f) →
          ((f != 0) == a.1.exitOnTrue) = true →
          a.2.1.slots.mapM (fun r => (Γ1.push (.sc t f))[r]?) = some vs →
          Xit Γ0 (Γ1.push (.sc t f)) vs w1) ∧
        (∀ nb Γ0 cs a Γ1 t f wb, Γ1.size + 1 = nb → Qh Γ0 cs a Γ1 wb →
          evalOp wb.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) = some (.sc t f) →
          ((f != 0) == a.1.exitOnTrue) = false →
          wp cfg (d + 1) (loopJ J Γ0 I Xit exitTys tys) (body d (carriesFrom nb tys) a.2.2)
            (fun next Γ2 w2 => ∀ nx, next.slots.mapM (fun r => Γ2[r]?) = some nx → I Γ0 nx w2)
            (bindAt (Γ1.push (.sc t f)) nb cs) wb) ∧
        (∀ nE Γ0 Γb vs w', Γ0.size ≤ nE → Xit Γ0 Γb vs w' →
          wp cfg d J (k (carriesFrom nE exitTys)) Q (bindAt Γb nE vs) w') ∧
        (∀ Γ0 cs a Γ1 w1, Qh Γ0 cs a Γ1 w1 →
          evalOp w1.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) = none → J.faultOk = true)) := rfl

theorem wp_dloop {tys : List ClifTy} {tb : ClifTy} (init : Vals Slot tys) (cc : ICmpCond) (cb : Slot tb)
    (guardIdx : Option Nat) (hg : guardIdx.all (fun i => (tys[i]?).getD default == tb) = true)
    (contOnTrue : Bool) (exitIdx : List Nat)
    (body : Lvl (idxTys tys exitIdx) tys → Vals Slot tys → Prog Slot Lvl (Slot tb × Vals Slot tys))
    (k : Vals Slot (idxTys tys exitIdx) → Prog Slot Lvl α) (Q : α → Env → World → Prop) :
    wp cfg d J (.dloop init cc cb guardIdx hg contOnTrue exitIdx body k) Q = fun Γ w =>
      ∃ (I : Env → List V → World → Prop)
        (Qb : Env → List V → Slot tb × Vals Slot tys → Env → World → Prop)
        (Xit : Env → Env → List V → World → Prop),
        ((∀ gi, guardIdx = some gi →
          evalOp w.mem Γ (.icmp cc ((init.slots[gi]?).getD 0) cb) = none → J.faultOk = true) ∧
        (∀ Γ' w' cs, guardPre guardIdx (At Γ w) cc ((init.slots[guardIdx.getD 0]?).getD 0) cb Γ' w' →
          init.slots.mapM (fun r => Γ'[r]?) = some cs →
          match guardIdx with
          | none => I Γ' cs w'
          | some _ => ∀ t f, Γ'[Γ'.size - 1]? = some (.sc t f) →
              (((f != 0) == contOnTrue) = true → I Γ' cs w') ∧
              (((f != 0) == contOnTrue) = false →
                ∀ outs, exitIdx.mapM (fun i => cs[i]?) = some outs → Xit Γ' Γ' outs w'))) ∧
        ((∀ n0 Γ0 cs wb, Γ0.size = n0 → I Γ0 cs wb →
          wp cfg (d + 1) (loopJ J Γ0 I Xit (idxTys tys exitIdx) tys) (body d (carriesFrom n0 tys))
            (Qb Γ0 cs) (bindAt Γ0 n0 cs) wb) ∧
        (∀ Γ0 cs a Γ2 w2 t f next, Qb Γ0 cs a Γ2 w2 →
          evalOp w2.mem Γ2 (.icmp cc a.1 cb) = some (.sc t f) →
          a.2.slots.mapM (fun r => (Γ2.push (.sc t f))[r]?) = some next →
          (((f != 0) == contOnTrue) = true → I Γ0 next w2) ∧
          (((f != 0) == contOnTrue) = false →
            ∀ outs, exitIdx.mapM (fun i => next[i]?) = some outs →
              Xit Γ0 (Γ2.push (.sc t f)) outs w2)) ∧
        (∀ nE Γ0 Γb outs w', Γ0.size ≤ nE → Xit Γ0 Γb outs w' →
          wp cfg d J (k (carriesFrom nE (idxTys tys exitIdx))) Q (bindAt Γb nE outs) w') ∧
        (∀ Γ0 cs a Γ2 w2, Qb Γ0 cs a Γ2 w2 →
          evalOp w2.mem Γ2 (.icmp cc a.1 cb) = none → J.faultOk = true)) := rfl

theorem wp_ite {jTys : List ClifTy} (c : Cond Slot) (thn els : Prog Slot Lvl (Vals Slot jTys))
    (k : Vals Slot jTys → Prog Slot Lvl α) (Q : α → Env → World → Prop) :
    wp cfg d J (.ite c thn els k) Q = fun Γ w =>
      ∃ (Qt Qe : Vals Slot jTys → Env → World → Prop),
        ((∀ t f, evalOp w.mem Γ (.icmp c.cc c.a c.b) = some (.sc t f) → (f != 0) = true →
          wp cfg d J thn Qt (Γ.push (.sc t f)) w) ∧
        (∀ ne t f, Γ.size + 1 ≤ ne → evalOp w.mem Γ (.icmp c.cc c.a c.b) = some (.sc t f) →
          (f != 0) = false → wp cfg d J els Qe (bindAt (Γ.push (.sc t f)) ne []) w) ∧
        (evalOp w.mem Γ (.icmp c.cc c.a c.b) = none → J.faultOk = true)) ∧
        (∀ nJ Γ' jv vs w', Γ'.size ≤ nJ → jv.slots.mapM (fun r => Γ'[r]?) = some vs →
          (Qt jv Γ' w' ∨ Qe jv Γ' w') →
          wp cfg d J (k (carriesFrom nJ jTys)) Q (bindAt Γ' nJ vs) w') := rfl

end wp_eqns

theorem guardPre_split {guardIdx : Option Nat} {P : Env → World → Prop} {cc : ICmpCond} {a b : R}
    {Γ : Env} {w : World} (h : guardPre guardIdx P cc a b Γ w) :
    ∃ Γ0 w0, P Γ0 w0 ∧ guardPre guardIdx (At Γ0 w0) cc a b Γ w := by
  cases guardIdx with
  | none => exact ⟨Γ, w, h, rfl, rfl⟩
  | some _ =>
      obtain ⟨Γ0, v, rfl, hP, hv⟩ := h
      exact ⟨Γ0, w, hP, Γ0, v, rfl, ⟨rfl, rfl⟩, hv⟩

/-- **The weakest precondition is a precondition.** -/
theorem wp_sound' (cfg : Cfg) {α : Type} (p : Prog Slot Lvl α) :
    ∀ d J Q, PT cfg d J (wp cfg d J p Q) p Q := by
  induction p with
  | ret a => exact fun _ _ Q => PT.ret fun _ _ h => h
  | op o k ih =>
      exact fun d J Q => PT.op (fun r => PT.conseq (ih r d J Q)
        (fun Γ w ⟨Γ0, v, hΓ, hr, hP, hv⟩ => by subst hΓ hr; exact hP.2 v hv) (fun _ _ _ h => h))
        (fun Γ w hP hv => hP.1 hv)
  | store v a k ih =>
      exact fun d J Q => PT.store (R := wp cfg d J k Q) (fun _ => nonCall_rule (fun _ _ => ⟨nofun, nofun⟩)
        (fun Γ w Γ' w' ⟨hn, hP⟩ hr => by
          have := runStmt_store_env hr; subst this; exact ⟨hn, hP.2 w' hr⟩)
        (fun Γ w m ⟨_, hP⟩ hr => hP.1 m hr)) (ih d J Q)
  | storeUnaligned v a k ih =>
      exact fun d J Q => PT.storeUnaligned (R := wp cfg d J k Q)
        (fun _ => nonCall_rule (fun _ _ => ⟨nofun, nofun⟩)
        (fun Γ w Γ' w' ⟨hn, hP⟩ hr => by
          have := runStmt_storeUnaligned_env hr; subst this; exact ⟨hn, hP.2 w' hr⟩)
        (fun Γ w m ⟨_, hP⟩ hr => hP.1 m hr)) (ih d J Q)
  | istore8 v a hi k ih =>
      exact fun d J Q => PT.istore8 (R := wp cfg d J k Q) (fun _ => nonCall_rule (fun _ _ => ⟨nofun, nofun⟩)
        (fun Γ w Γ' w' ⟨hn, hP⟩ hr => by
          have := runStmt_istore8_env hr; subst this; exact ⟨hn, hP.2 w' hr⟩)
        (fun Γ w m ⟨_, hP⟩ hr => hP.1 m hr)) (ih d J Q)
  | call f args k ih =>
      intro d J Q
      cases hres : f.result.isSome
      · refine PT.callVoid (R := wp cfg d J (k (resSlot f.result 0)) Q) hres
          (fun _ => callVoid_rule fun Γ w vs ⟨hn, hP⟩ hvs => ?_) (ih _ d J Q)
        obtain ⟨r, w', hc, hq⟩ := hP vs hvs
        simp only [hres] at hq
        exact ⟨r, w', hc, hn, hq⟩
      · refine PT.call (R := fun n Γ' w' => wp cfg d J (k (resSlot f.result n)) Q Γ' w') hres
          (fun _ => call_rule fun Γ w vs ⟨hn, hP⟩ hvs => ?_) (fun r => ih _ d J Q)
        obtain ⟨r, w', hc, hq⟩ := hP vs hvs
        simp only [hres, if_true] at hq
        exact ⟨r, w', hc, fun v hv => ⟨by simp [hn], hn ▸ hq v hv⟩⟩
  | @callLocal ps res β r args k ih =>
      intro d J Q
      cases hres : res.isSome
      · refine PT.callLocalVoid (R := wp cfg d J (k (resSlot res 0)) Q) hres
          (fun _ => callVoidL_rule fun Γ w vs ⟨hn, hP⟩ hvs => ?_) (ih _ d J Q)
        obtain ⟨hf, hq⟩ := hP vs hvs
        refine ⟨hf, fun x w' hc => ⟨hn, ?_⟩⟩
        have := hq x w' hc
        simp only [hres] at this
        exact this
      · refine PT.callLocal (R := fun n Γ' w' => wp cfg d J (k (resSlot res n)) Q Γ' w') hres
          (fun _ => callL_rule fun Γ w vs ⟨hn, hP⟩ hvs => ?_) (fun r => ih _ d J Q)
        obtain ⟨hf, hq⟩ := hP vs hvs
        refine ⟨hf, fun x w' hc v hv => ⟨by simp [hn], ?_⟩⟩
        have := hq x w' hc
        simp only [hres, if_true] at this
        exact hn ▸ this v hv
  | loop init head body k ihH ihB ihK =>
      intro d J Q
      rw [wp_loop]
      refine PT.exists fun I => PT.exists fun Qh => PT.exists fun Xit => PT.gate fun ⟨hH, hX, hB, hK, hF⟩ => ?_
      exact PT.loop I Qh Xit (fun Γ w cs h hcs => h cs hcs)
        (fun n0 Γ0 cs => PT.conseq (ihH _ _ _ _ _)
          (fun Γh wh ⟨hs, he, hI⟩ => by subst he; exact hH n0 Γ0 cs wh hs hI) (fun _ _ _ h => h))
        hX
        (fun nb Γ0 cs a => PT.conseq (ihB _ _ _ _ _ _)
          (fun Γb wb ⟨Γ1, t, f, hs, he, hq, hv, hx⟩ => by
            subst he; exact hB nb Γ0 cs a Γ1 t f wb hs hq hv hx) (fun _ _ _ h => h))
        (fun nE => PT.conseq (ihK _ _ _ _)
          (fun Γ w ⟨Γ0, Γb, vs, hle, he, hx⟩ => by subst he; exact hK nE Γ0 Γb vs w hle hx)
          (fun _ _ _ h => h))
        hF
  | dloop init cc cb g hg c e body k ihB ihK =>
      intro d J Q
      rw [wp_dloop]
      refine PT.exists fun I => PT.exists fun Qb => PT.exists fun Xit => PT.gate fun ⟨hB, hX, hK, hBF⟩ => ?_
      refine PT.dloop I Qb Xit ?_
        (fun n0 Γ0 cs => PT.conseq (ihB _ _ _ _ _)
          (fun Γh wh ⟨hs, he, hI⟩ => by subst he; exact hB n0 Γ0 cs wh hs hI) (fun _ _ _ h => h))
        hX
        (fun nE => PT.conseq (ihK _ _ _ _)
          (fun Γ w ⟨Γ0, Γb, outs, hle, he, hx⟩ => by subst he; exact hK nE Γ0 Γb outs w hle hx)
          (fun _ _ _ h => h))
        (fun gi hgi Γ w hP hv => hP.1 gi hgi hv) hBF
      intro Γ w cs hg' hcs
      obtain ⟨Γ0, w0, hP, hg0⟩ := guardPre_split hg'
      exact hP.2 Γ w cs hg0 hcs
  | ite c thn els k ihT ihE ihK =>
      intro d J Q
      rw [wp_ite]
      refine PT.exists fun Qt => PT.exists fun Qe => PT.gate fun hK => ?_
      refine PT.ite Qt Qe (PT.conseq (ihT _ _ _)
          (fun Γ w ⟨Γ0, t, f, he, hP, hv, hf⟩ => by subst he; exact hP.1 t f hv hf) (fun _ _ _ h => h))
        (fun ne => PT.conseq (ihE _ _ _) ?_ (fun _ _ _ h => h))
        (fun nJ => PT.conseq (ihK _ _ _ _)
          (fun Γ w ⟨Γ', jv, vs, hle, he, hvs, hq⟩ => by subst he; exact hK nJ Γ' jv vs w hle hvs hq)
          (fun _ _ _ h => h))
        (fun Γ w hP hv => hP.2.2 hv)
      rintro Γ w ⟨Γ0, t, f, hle, he, hP, hv, hf⟩
      subst he
      exact hP.2.1 ne t f hle hv hf
  | params tys k ih => exact fun d J Q => PT.params (ih _ d J Q)
  | br l args => exact fun d J Q => PT.br fun _ _ vs hP hvs => hP vs hvs
  | cont l args => exact fun d J Q => PT.cont fun _ _ vs hP hvs => hP vs hvs

theorem wp_sound (cfg : Cfg) (d : Nat) (J : Post) {α : Type} (p : Prog Slot Lvl α) :
    ∀ Q, PT cfg d J (wp cfg d J p Q) p Q :=
  wp_sound' cfg p d J

/-- Sequencing is composition of weakest preconditions. -/
theorem wp_bind (cfg : Cfg) {α β : Type} (p : Prog Slot Lvl α) (f : α → Prog Slot Lvl β) :
    ∀ (d : Nat) (J : Post) (Q : β → Env → World → Prop),
    wp cfg d J (p >>= f) Q = wp cfg d J p (fun a => wp cfg d J (f a) Q) := by
  show ∀ d J Q, wp cfg d J (Prog.bind p f) Q = _
  induction p with
  | ret a => intro _ _ _; rfl
  | op o k ih => intro d J Q; rw [Prog.bind]; funext Γ w; simp only [wp_op, ih]
  | store v a k ih => intro d J Q; rw [Prog.bind]; simp only [wp_store, ih]
  | storeUnaligned v a k ih => intro d J Q; rw [Prog.bind]; simp only [wp_storeUnaligned, ih]
  | istore8 v a hi k ih => intro d J Q; rw [Prog.bind]; simp only [wp_istore8, ih]
  | call fn args k ih => intro d J Q; rw [Prog.bind]; funext Γ w; simp only [wp_call, ih]
  | callLocal r args k ih => intro d J Q; rw [Prog.bind]; simp only [wp_callLocal, ih]
  | loop init head body k _ _ ih => intro d J Q; rw [Prog.bind]; simp only [wp_loop, ih]
  | dloop init cc cb g hg c e body k _ ih => intro d J Q; rw [Prog.bind]; simp only [wp_dloop, ih]
  | ite c thn els k _ _ ih => intro d J Q; rw [Prog.bind]; simp only [wp_ite, ih]
  | params tys k ih => intro d J Q; rw [Prog.bind]; simp only [wp_params, ih]
  | br l args => intro _ _ _; rw [Prog.bind]; rfl
  | cont l args => intro _ _ _; rw [Prog.bind]; rfl

/-- A weaker postcondition has a weaker precondition. -/
theorem wp_mono (cfg : Cfg) {α : Type} (p : Prog Slot Lvl α) :
    ∀ (d : Nat) (J : Post) (Q Q' : α → Env → World → Prop), (∀ a Γ w, Q a Γ w → Q' a Γ w) →
    ∀ Γ w, wp cfg d J p Q Γ w → wp cfg d J p Q' Γ w := by
  induction p with
  | ret a => exact fun _ _ _ _ hQ Γ w h => hQ a Γ w h
  | op o k ih => exact fun d J Q Q' hQ Γ w h => ⟨h.1, fun v hv => ih _ d J Q Q' hQ _ _ (h.2 v hv)⟩
  | store v a k ih => exact fun d J Q Q' hQ Γ w h => ⟨h.1, fun w' hr => ih d J Q Q' hQ _ _ (h.2 w' hr)⟩
  | storeUnaligned v a k ih =>
      exact fun d J Q Q' hQ Γ w h => ⟨h.1, fun w' hr => ih d J Q Q' hQ _ _ (h.2 w' hr)⟩
  | istore8 v a hi k ih => exact fun d J Q Q' hQ Γ w h => ⟨h.1, fun w' hr => ih d J Q Q' hQ _ _ (h.2 w' hr)⟩
  | call f args k ih =>
      intro d J Q Q' hQ Γ w h vs hvs
      obtain ⟨r, w', hc, hq⟩ := h vs hvs
      refine ⟨r, w', hc, ?_⟩
      cases hres : f.result.isSome <;> simp only [hres, if_true, Bool.false_eq_true, if_false] at hq ⊢
      · exact ih _ d J Q Q' hQ _ _ hq
      · exact fun v hv => ih _ d J Q Q' hQ _ _ (hq v hv)
  | callLocal r args k ih =>
      intro d J Q Q' hQ Γ w h vs hvs
      obtain ⟨hf, hq⟩ := h vs hvs
      refine ⟨hf, fun x w' hc => ?_⟩
      have hq := hq x w' hc
      split at hq
      · rename_i hres; simp only [hres, if_true]; exact fun v hv => ih _ d J Q Q' hQ _ _ (hq v hv)
      · rename_i hres; simp only [hres, if_false, Bool.false_eq_true]; exact ih _ d J Q Q' hQ _ _ hq
  | loop init head body k _ _ ihK =>
      intro d J Q Q' hQ Γ w h
      rw [wp_loop] at h ⊢
      obtain ⟨I, Qh, Xit, h1, h2, h3, h4, h5, h6⟩ := h
      exact ⟨I, Qh, Xit, h1, h2, h3, h4, fun nE Γ0 Γb vs w' hle hx =>
        ihK _ d J Q Q' hQ _ _ (h5 nE Γ0 Γb vs w' hle hx), h6⟩
  | dloop init cc cb g hg c e body k _ ihK =>
      intro d J Q Q' hQ Γ w h
      rw [wp_dloop] at h ⊢
      obtain ⟨I, Qb, Xit, h1, h2, h3, h4, h5⟩ := h
      exact ⟨I, Qb, Xit, h1, h2, h3, fun nE Γ0 Γb outs w' hle hx =>
        ihK _ d J Q Q' hQ _ _ (h4 nE Γ0 Γb outs w' hle hx), h5⟩
  | ite c thn els k _ _ ihK =>
      intro d J Q Q' hQ Γ w h
      rw [wp_ite] at h ⊢
      obtain ⟨Qt, Qe, h1, h2⟩ := h
      exact ⟨Qt, Qe, h1, fun nJ Γ' jv vs w' hle hvs hq => ihK _ d J Q Q' hQ _ _ (h2 nJ Γ' jv vs w' hle hvs hq)⟩
  | params tys k ih => exact fun d J Q Q' hQ Γ w h => ih _ d J Q Q' hQ Γ w h
  | br l args => exact fun _ _ _ _ _ _ _ h => h
  | cont l args => exact fun _ _ _ _ _ _ _ h => h

/-- **What a proof about a generator says about the code it ships.** -/
theorem emit_triple {cfg : Cfg} {J : Post} {P : Env → World → Prop} {p : Body}
    {Q : Unit → Env → World → Prop} {params : List ClifTy} {P0 : Env → World → Prop}
    (h : PT cfg 0 J P p Q) (hf : Fine (emitGo p ⟨params.length, 0, [], [], [], none⟩).2)
    (h0 : ∀ Γ w, P0 Γ w → Γ.size = params.length ∧ P Γ w) :
    Triple cfg P0 (emit p params) { J with ok := fun Γ w => Q () Γ w } := by
  have h1 := h ⟨params.length, 0, [], [], [], none⟩ P0 rfl hf (nil_rule h0)
  show Triple cfg P0 (emitGo p ⟨params.length, 0, [], [], [], none⟩).2.flush.pieces.reverse _
  rw [St.out_eq_flush]
  exact Hoare.conseq h1 (fun _ _ h => h) (fun Γ w ⟨_, _, _, hq⟩ => hq)
    (fun _ _ _ _ h => h) (fun _ _ _ h => h)

/-- The code a body that answers ships: its pieces, the answer being the
    function's return. -/
def emitAns {α : Type} (p : Prog Slot Lvl α) (params : List ClifTy := ptrParams) : Code :=
  (emitGo p ⟨params.length, 0, [], [], [], none⟩).2.flush.pieces.reverse

/-- `emit_triple`, for a body that answers. -/
theorem emitAns_triple {cfg : Cfg} {J : Post} {P : Env → World → Prop} {α : Type}
    {p : Prog Slot Lvl α} {Q : α → Env → World → Prop} {params : List ClifTy}
    {P0 : Env → World → Prop}
    (h : PT cfg 0 J P p Q) (hf : Fine (emitGo p ⟨params.length, 0, [], [], [], none⟩).2)
    (h0 : ∀ Γ w, P0 Γ w → Γ.size = params.length ∧ P Γ w) :
    Triple cfg P0 (emitAns p params) { J with ok := fun _ _ => True } := by
  have h1 := h ⟨params.length, 0, [], [], [], none⟩ P0 rfl hf (nil_rule h0)
  show Triple cfg P0 (emitGo p ⟨params.length, 0, [], [], [], none⟩).2.flush.pieces.reverse _
  rw [St.out_eq_flush]
  exact Hoare.conseq h1 (fun _ _ h => h) (fun _ _ _ => trivial)
    (fun _ _ _ _ h => h) (fun _ _ _ h => h)

end AlgorithmLib.Prog
