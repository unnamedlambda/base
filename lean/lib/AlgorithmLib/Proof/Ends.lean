module
public import AlgorithmLib.Proof.ProgHoare
meta import AlgorithmLib.Proof.ProgHoare
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Proof.Ends` — runs that end

A triple (`Host.Hoare.Triple`) says how a code ends where it ends: a run may
get stuck, so a code that never ends meets every triple. `Ends cfg P c Q`
says it ends: from every state `P` holds of, the run finishes normally at
some fuel, in `Q`. Runs are deterministic and more fuel changes no run that
did not get stuck (`runCode_mono`), so a code that ends meets the triple too,
whatever it may not do on the way (`Ends.triple`).

The rules are the triple's, each run ending: statements (`StmtE`, `stmtsE_cons`),
pieces (`straightE`, `iteE`, `loopE`) and codes (`endsE_cons`, `endsE_append`,
`endsE_merge`). A loop ends by a measure its trips lower (`iterE`).
-/

namespace AlgorithmLib.HProg.Hoare

open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sem

/-- One statement ends, in `Q`. -/
def StmtE (cfg : Cfg) (P : Env → World → Prop) (s : Stmt) (Q : Env → World → Prop) : Prop :=
  ∀ Γ w, P Γ w → ∃ Γ' w', runStmt cfg Γ w s = .ok Γ' w' ∧ Q Γ' w'

/-- A straight run of statements ends, in `Q`. -/
def StmtsE (cfg : Cfg) (P : Env → World → Prop) (ss : List Stmt) (Q : Env → World → Prop) : Prop :=
  ∀ Γ w, P Γ w → ∃ Γ' w', runStmts cfg Γ w ss = .ok Γ' w' ∧ Q Γ' w'

/-- A piece ends, at some fuel, in `Q`. -/
def PieceE (cfg : Cfg) (P : Env → World → Prop) (p : Piece) (Q : Env → World → Prop) : Prop :=
  ∀ Γ w, P Γ w → ∃ k Γ' w', runPiece k cfg Γ w p = .ok Γ' w' ∧ Q Γ' w'

/-- **A code ends**: from every state `P` holds of, at some fuel, in `Q`. -/
def Ends (cfg : Cfg) (P : Env → World → Prop) (c : Code) (Q : Env → World → Prop) : Prop :=
  ∀ Γ w, P Γ w → ∃ k Γ' w', runCode k cfg Γ w c = .ok Γ' w' ∧ Q Γ' w'

theorem runPiece_mono {k k' : Nat} (hk : k ≤ k') {cfg : Cfg} {Γ : Env} {w : World} {p : Piece}
    {r : CodeRes} (h : runPiece k cfg Γ w p = r) (hr : r.fin) : runPiece k' cfg Γ w p = r :=
  (monoAt k).1 cfg Γ w p r h hr k' hk

theorem runCode_ok_mono {k k' : Nat} (hk : k ≤ k') {cfg : Cfg} {Γ Γ' : Env} {w w' : World} {c : Code}
    (h : runCode k cfg Γ w c = .ok Γ' w') : runCode k' cfg Γ w c = .ok Γ' w' :=
  runCode_mono hk h rfl

theorem runPiece_ok_mono {k k' : Nat} (hk : k ≤ k') {cfg : Cfg} {Γ Γ' : Env} {w w' : World} {p : Piece}
    (h : runPiece k cfg Γ w p = .ok Γ' w') : runPiece k' cfg Γ w p = .ok Γ' w' :=
  runPiece_mono hk h rfl

theorem Ends.conseq {cfg : Cfg} {P P' Q Q' : Env → World → Prop} {c : Code} (h : Ends cfg P c Q)
    (hP : ∀ Γ w, P' Γ w → P Γ w) (hQ : ∀ Γ w, Q Γ w → Q' Γ w) : Ends cfg P' c Q' := by
  intro Γ w hp
  obtain ⟨k, Γ', w', hr, hq⟩ := h Γ w (hP Γ w hp)
  exact ⟨k, Γ', w', hr, hQ Γ' w' hq⟩

/-- **A code that ends meets the triple**: it ends in `Q` at every fuel it does
    not run out of, and at the others it gets stuck. -/
theorem Ends.triple {cfg : Cfg} {P Q : Env → World → Prop} {c : Code} (h : Ends cfg P c Q) (J : Post) :
    Triple cfg P c { J with ok := Q } := by
  intro fuel Γ w hP
  obtain ⟨k, Γ', w', hr, hq⟩ := h Γ w hP
  cases hf : runCode fuel cfg Γ w c with
  | stuck m => trivial
  | ok Γ1 w1 =>
      rcases Nat.le_total k fuel with hle | hle
      · rw [runCode_ok_mono hle hr] at hf; cases hf; exact hq
      · rw [runCode_mono hle hf rfl] at hr; cases hr; exact hq
  | brk d Γb vs w1 =>
      have := runCode_mono (Nat.le_add_left fuel k) hf rfl
      rw [runCode_ok_mono (Nat.le_add_right k fuel) hr] at this; cases this
  | cont d vs w1 =>
      have := runCode_mono (Nat.le_add_left fuel k) hf rfl
      rw [runCode_ok_mono (Nat.le_add_right k fuel) hr] at this; cases this
  | misuse m =>
      have := runCode_mono (Nat.le_add_left fuel k) hf rfl
      rw [runCode_ok_mono (Nat.le_add_right k fuel) hr] at this; cases this
  | fault m =>
      have := runCode_mono (Nat.le_add_left fuel k) hf rfl
      rw [runCode_ok_mono (Nat.le_add_right k fuel) hr] at this; cases this

-- ---------------------------------------------------------------------------
-- Statements
-- ---------------------------------------------------------------------------

theorem opE {cfg : Cfg} {P Q : Env → World → Prop} {o : Op}
    (h : ∀ Γ w, P Γ w → ∃ v, evalOp w.mem Γ o = some v ∧ Q (Γ.push v) w) : StmtE cfg P (.op o) Q := by
  intro Γ w hP
  obtain ⟨v, hv, hq⟩ := h Γ w hP
  exact ⟨_, _, by simp only [runStmt, hv], hq⟩

theorem stmtsE_nil {cfg : Cfg} {P Q : Env → World → Prop} (h : ∀ Γ w, P Γ w → Q Γ w) :
    StmtsE cfg P [] Q :=
  fun Γ w hP => ⟨Γ, w, rfl, h Γ w hP⟩

theorem stmtsE_cons {cfg : Cfg} {P R Q : Env → World → Prop} {s : Stmt} {ss : List Stmt}
    (h1 : StmtE cfg P s R) (h2 : StmtsE cfg R ss Q) : StmtsE cfg P (s :: ss) Q := by
  intro Γ w hP
  obtain ⟨Γ1, w1, hr1, hR⟩ := h1 Γ w hP
  obtain ⟨Γ2, w2, hr2, hQ⟩ := h2 Γ1 w1 hR
  exact ⟨Γ2, w2, by simp only [runStmts, hr1, hr2], hQ⟩

theorem stmtsE_append {cfg : Cfg} {P R Q : Env → World → Prop} {x y : List Stmt}
    (h1 : StmtsE cfg P x R) (h2 : StmtsE cfg R y Q) : StmtsE cfg P (x ++ y) Q := by
  intro Γ w hP
  obtain ⟨Γ1, w1, hr1, hR⟩ := h1 Γ w hP
  obtain ⟨Γ2, w2, hr2, hQ⟩ := h2 Γ1 w1 hR
  exact ⟨Γ2, w2, by rw [runStmts_append, hr1]; exact hr2, hQ⟩

-- ---------------------------------------------------------------------------
-- Pieces and codes
-- ---------------------------------------------------------------------------

theorem straightE {cfg : Cfg} {P Q : Env → World → Prop} {ss : List Stmt} (h : StmtsE cfg P ss Q) :
    PieceE cfg P (.straight ss) Q := by
  intro Γ w hP
  obtain ⟨Γ', w', hr, hq⟩ := h Γ w hP
  exact ⟨1, Γ', w', by simp only [runPiece, hr], hq⟩

theorem endsE_nil {cfg : Cfg} {P Q : Env → World → Prop} (h : ∀ Γ w, P Γ w → Q Γ w) : Ends cfg P [] Q :=
  fun Γ w hP => ⟨1, Γ, w, by simp only [runCode], h Γ w hP⟩

theorem endsE_cons {cfg : Cfg} {P R Q : Env → World → Prop} {p : Piece} {ps : Code}
    (h1 : PieceE cfg P p R) (h2 : Ends cfg R ps Q) : Ends cfg P (p :: ps) Q := by
  intro Γ w hP
  obtain ⟨k1, Γ1, w1, hr1, hR⟩ := h1 Γ w hP
  obtain ⟨k2, Γ2, w2, hr2, hQ⟩ := h2 Γ1 w1 hR
  refine ⟨max k1 k2 + 1, Γ2, w2, ?_, hQ⟩
  simp only [runCode, runPiece_ok_mono (Nat.le_max_left k1 k2) hr1]
  exact runCode_ok_mono (Nat.le_max_right k1 k2) hr2

/-- Two codes run in turn: the first ending, then the second from there. -/
theorem runCode_append_ok {cfg : Cfg} {G : Code} : ∀ (F : Code) (k k2 : Nat) (Γ Γ1 Γ2 : Env) (w w1 w2 : World),
    runCode k cfg Γ w F = .ok Γ1 w1 → runCode k2 cfg Γ1 w1 G = .ok Γ2 w2 →
    runCode (k + k2) cfg Γ w (F ++ G) = .ok Γ2 w2
  | [], k, k2, Γ, Γ1, Γ2, w, w1, w2, h1, h2 => by
      cases k with
      | zero => simp [runCode] at h1
      | succ k =>
          simp only [runCode, CodeRes.ok.injEq] at h1
          obtain ⟨rfl, rfl⟩ := h1
          exact runCode_ok_mono (by omega) h2
  | p :: ps, k, k2, Γ, Γ1, Γ2, w, w1, w2, h1, h2 => by
      cases k with
      | zero => simp [runCode] at h1
      | succ k =>
          simp only [runCode] at h1
          split at h1 <;> try cases h1
          rename_i Γa wa ha
          rw [show k + 1 + k2 = (k + k2) + 1 by omega, List.cons_append]
          simp only [runCode, runPiece_ok_mono (Nat.le_add_right k k2) ha]
          exact runCode_append_ok ps k k2 Γa Γ1 Γ2 wa w1 w2 h1 h2

/-- A run of two codes that ends is the first ending, then the second from
    there. -/
theorem runCode_append_split {cfg : Cfg} {G : Code} : ∀ (F : Code) (k : Nat) (Γ Γ2 : Env) (w w2 : World),
    runCode k cfg Γ w (F ++ G) = .ok Γ2 w2 →
    ∃ k1 k2 Γ1 w1, runCode k1 cfg Γ w F = .ok Γ1 w1 ∧ runCode k2 cfg Γ1 w1 G = .ok Γ2 w2
  | [], k, Γ, Γ2, w, w2, h => ⟨1, k, Γ, w, by simp only [runCode], by simpa using h⟩
  | p :: ps, k, Γ, Γ2, w, w2, h => by
      cases k with
      | zero => simp [runCode] at h
      | succ k =>
          rw [List.cons_append] at h
          simp only [runCode] at h
          split at h <;> try cases h
          rename_i Γa wa ha
          obtain ⟨k1, k2, Γ1, w1, h1, h2⟩ := runCode_append_split ps k Γa Γ2 wa w2 h
          refine ⟨max k k1 + 1, k2, Γ1, w1, ?_, h2⟩
          simp only [runCode, runPiece_ok_mono (Nat.le_max_left k k1) ha]
          exact runCode_ok_mono (Nat.le_max_right k k1) h1

/-- **Two codes in turn.** -/
theorem endsE_append {cfg : Cfg} {P R Q : Env → World → Prop} {F G : Code}
    (h1 : Ends cfg P F R) (h2 : Ends cfg R G Q) : Ends cfg P (F ++ G) Q := by
  intro Γ w hP
  obtain ⟨k1, Γ1, w1, hr1, hR⟩ := h1 Γ w hP
  obtain ⟨k2, Γ2, w2, hr2, hQ⟩ := h2 Γ1 w1 hR
  exact ⟨k1 + k2, Γ2, w2, runCode_append_ok F k1 k2 Γ Γ1 Γ2 w w1 w2 hr1 hr2, hQ⟩

/-- **A straight run that goes on**: statements added to a code's last
    straight run end after it. -/
theorem endsE_merge {cfg : Cfg} {P R Q : Env → World → Prop} {F : Code} {x y : List Stmt}
    (h1 : Ends cfg P (F ++ [.straight x]) R) (h2 : StmtsE cfg R y Q) :
    Ends cfg P (F ++ [.straight (x ++ y)]) Q := by
  intro Γ w hP
  obtain ⟨k, Γ2, w2, hr, hR⟩ := h1 Γ w hP
  obtain ⟨k1, k2, Γ1, w1, hF, hx⟩ := runCode_append_split F k Γ Γ2 w w2 hr
  obtain ⟨Γ3, w3, hy, hQ⟩ := h2 Γ2 w2 hR
  have hxs : runStmts cfg Γ1 w1 x = .ok Γ2 w2 := by
    match k2, hx with
    | 0, hx => simp [runCode] at hx
    | 1, hx => simp [runCode, runPiece] at hx
    | j + 2, hx =>
        cases hs : runStmts cfg Γ1 w1 x with
        | ok Γa wa =>
            simp only [runCode, runPiece, hs, CodeRes.ok.injEq] at hx
            obtain ⟨rfl, rfl⟩ := hx; rfl
        | _ => simp [runCode, runPiece, hs] at hx
  refine ⟨k1 + 2, Γ3, w3, runCode_append_ok F k1 2 Γ Γ1 Γ3 w w1 w3 hF ?_, hQ⟩
  simp only [runCode, runPiece, runStmts_append, hxs, hy]


-- ---------------------------------------------------------------------------
-- Branches and loops
-- ---------------------------------------------------------------------------

/-- **A branch ends** where the arm its flag picks ends with its exports in
    scope. -/
theorem iteE {cfg : Cfg} {P Q : Env → World → Prop} {m : IteMeta} {thn els : Code} {thnR elsR : List R}
    (h : ∀ Γ w, P Γ w → ∃ t f, get Γ m.flag = some (.sc t f) ∧
      ((f != 0) = true → Ends cfg (At Γ w) thn (fun Γ' w' => ∃ vs, thnR.mapM (get Γ') = some vs ∧
          Q (bindAt Γ' (slotsOf (slotsOf Γ.size thn) els) vs) w')) ∧
      ((f != 0) = false → Ends cfg (At (bindAt Γ (slotsOf Γ.size thn) []) w) els (fun Γ' w' => ∃ vs,
          elsR.mapM (get Γ') = some vs ∧ Q (bindAt Γ' (slotsOf (slotsOf Γ.size thn) els) vs) w'))) :
    PieceE cfg P (.ite m thn els thnR elsR) Q := by
  intro Γ w hP
  obtain ⟨t, f, hf, hT, hE⟩ := h Γ w hP
  by_cases hz : (f != 0) = true
  · obtain ⟨k, Γ', w', hr, vs, hvs, hq⟩ := hT hz Γ w ⟨rfl, rfl⟩
    refine ⟨k + 1, _, _, ?_, hq⟩
    simp only [runPiece, hf, hz, if_true, hr, hvs]
  · have hz' : (f != 0) = false := by simpa using hz
    obtain ⟨k, Γ', w', hr, vs, hvs, hq⟩ := hE hz' _ w ⟨rfl, rfl⟩
    refine ⟨k + 1, _, _, ?_, hq⟩
    simp only [runPiece, hf, hz', Bool.false_eq_true, if_false, hr, hvs]

/-- **A top-tested loop ends** where each trip from carries `P` holds of ends
    its condition prefix and either leaves, in `Q`, or ends its body with next
    carries `P` holds of and the measure `μ` lower. -/
theorem iterE {cfg : Cfg} {Γ : Env} {l : Loop} {pre body : List Piece} {n0 ab : Nat}
    (P : List V → World → Prop) (Q : Env → World → Prop) (μ : List V → World → Nat)
    (hstep : ∀ cs w, P cs w →
      ∃ k Γ1 w1 t f, runCode k cfg (bindAt Γ n0 cs) w pre = .ok Γ1 w1 ∧
        get Γ1 l.flag = some (.sc t f) ∧
        (((f != 0) == l.exitOnTrue) = true →
          ∃ vs, l.exitR.mapM (get Γ1) = some vs ∧ Q (bindAt Γ1 ab vs) w1) ∧
        (((f != 0) == l.exitOnTrue) = false →
          ∃ k2 Γ2 w2 next,
            runCode k2 cfg (bindAt Γ1 (slotsOf (n0 + l.pTys.length) pre) cs) w1 body = .ok Γ2 w2 ∧
            l.cont.mapM (get Γ2) = some next ∧ P next w2 ∧ μ next w2 < μ cs w)) :
    ∀ cs w, P cs w → ∃ k Γ' w', iter k cfg Γ w l pre body n0 ab cs = .ok Γ' w' ∧ Q Γ' w' := by
  suffices h : ∀ n cs w, μ cs w < n → P cs w →
      ∃ k Γ' w', iter k cfg Γ w l pre body n0 ab cs = .ok Γ' w' ∧ Q Γ' w' from
    fun cs w hP => h _ cs w (Nat.lt_succ_self _) hP
  intro n
  induction n with
  | zero => intro cs w hn; omega
  | succ n ih =>
    intro cs w hn hP
    obtain ⟨k, Γ1, w1, t, f, hpre, hflag, hexit, hcont⟩ := hstep cs w hP
    by_cases hx : ((f != 0) == l.exitOnTrue) = true
    · obtain ⟨vs, hvs, hQ⟩ := hexit hx
      refine ⟨k + 1, _, _, ?_, hQ⟩
      simp only [iter, hpre, hflag, hx, if_true, hvs]
    · have hx' : ((f != 0) == l.exitOnTrue) = false := by simpa using hx
      obtain ⟨k2, Γ2, w2, next, hb, hn', hP', hμ⟩ := hcont hx'
      obtain ⟨k3, Γ', w', hi, hQ⟩ := ih next w2 (by omega) hP'
      refine ⟨max k (max k2 k3) + 1, Γ', w', ?_, hQ⟩
      simp only [iter, runCode_ok_mono (show k ≤ max k (max k2 k3) by omega) hpre, hflag, hx',
        Bool.false_eq_true, if_false, runCode_ok_mono (show k2 ≤ max k (max k2 k3) by omega) hb, hn']
      exact iter_mono (show k3 ≤ max k (max k2 k3) by omega) hi rfl

/-- **A top-tested loop, by invariant and measure**: the initial carries are
    in scope and satisfy `I`; a trip from carries satisfying it ends its
    prefix, and either leaves with its exits in scope, in `Q`, or ends its
    body with next carries in scope that satisfy `I` and lower `μ`. -/
theorem loopE {cfg : Cfg} {P Q : Env → World → Prop} {l : Loop} {pre body : Code}
    (I : Env → List V → World → Prop) (μ : Env → List V → World → Nat)
    (hinit : ∀ Γ w, P Γ w → ∃ cs, l.init.mapM (get Γ) = some cs ∧ I Γ cs w)
    (htrip : ∀ Γ w0 cs w, P Γ w0 → I Γ cs w →
      Ends cfg (At (bindAt Γ Γ.size cs) w) pre (fun Γ1 w1 => ∃ t f, get Γ1 l.flag = some (.sc t f) ∧
        (((f != 0) == l.exitOnTrue) = true → ∃ vs, l.exitR.mapM (get Γ1) = some vs ∧
          Q (bindAt Γ1 (slotsOf (slotsOf (Γ.size + l.pTys.length) pre + l.pTys.length) body) vs) w1) ∧
        (((f != 0) == l.exitOnTrue) = false →
          Ends cfg (At (bindAt Γ1 (slotsOf (Γ.size + l.pTys.length) pre) cs) w1) body
            (fun Γ2 w2 => ∃ next, l.cont.mapM (get Γ2) = some next ∧ I Γ next w2 ∧
              μ Γ next w2 < μ Γ cs w)))) :
    PieceE cfg P (.loop l pre body) Q := by
  intro Γ w hP
  obtain ⟨cs, hcs, hI⟩ := hinit Γ w hP
  obtain ⟨k, Γ', w', hi, hQ⟩ := iterE (cfg := cfg) (Γ := Γ) (l := l) (pre := pre) (body := body)
    (n0 := Γ.size) (ab := slotsOf (slotsOf (Γ.size + l.pTys.length) pre + l.pTys.length) body)
    (I Γ) Q (μ Γ) (fun cs' w' hI' => by
      obtain ⟨k, Γ1, w1, hr, t, f, hflag, hx, hc⟩ := htrip Γ w cs' w' hP hI' _ _ ⟨rfl, rfl⟩
      refine ⟨k, Γ1, w1, t, f, hr, hflag, hx, fun h => ?_⟩
      obtain ⟨k2, Γ2, w2, hb, next, hn, hI2, hμ⟩ := hc h _ _ ⟨rfl, rfl⟩
      exact ⟨k2, Γ2, w2, next, hb, hn, hI2, hμ⟩) cs w hI
  exact ⟨k + 1, Γ', w', by simp only [runPiece, hcs, hi], hQ⟩

end AlgorithmLib.HProg.Hoare

namespace AlgorithmLib.Prog

open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare

/-- A statement the emitter adds ends after what it had produced. -/
theorem St.out_stmtE {cfg : Cfg} {P0 R R' : Env → World → Prop} {s : St} {st : Stmt}
    (h : Ends cfg P0 s.out R) (hs : StmtE cfg R st R') : Ends cfg P0 (s.stmt st).out R' := by
  have hs' : StmtsE cfg R [st] R' := stmtsE_cons hs (stmtsE_nil fun _ _ h => h)
  unfold St.out St.stmt at *
  cases hc : s.cur with
  | nil =>
      simp only [hc, List.append_nil] at h ⊢
      simp only [List.reverse_cons, List.reverse_nil, List.nil_append]
      exact endsE_append h (endsE_cons (straightE hs') (endsE_nil fun _ _ h => h))
  | cons c cs =>
      simp only [hc] at h ⊢
      simp only [List.reverse_cons]
      rw [List.reverse_cons] at h
      exact endsE_merge h hs'

/-- A piece the emitter adds after closing the open run ends after it. -/
theorem St.out_pieceE {cfg : Cfg} {P0 R Q : Env → World → Prop} {s : St} {p : Piece}
    (h : Ends cfg P0 s.out R) (hp : PieceE cfg R p Q) :
    Ends cfg P0 { s.flush with pieces := p :: s.flush.pieces }.out Q := by
  have e : ({ s.flush with pieces := p :: s.flush.pieces } : St).out = s.out ++ [p] := by
    rw [← St.out_flush]
    unfold St.flush St.out
    cases hc : s.cur <;> simp [hc]
  rw [e]
  exact endsE_append h (endsE_cons hp (endsE_nil fun _ _ h => h))

theorem Ends.of_push {cfg : Cfg} {P0 Q : Env → World → Prop} {S1 X : St} {p : Piece}
    (h : Ends cfg P0 (S1.out ++ [p]) Q) (hp : X.pieces = p :: S1.pieces) (hc : X.cur = [])
    (hc1 : S1.cur = []) : Ends cfg P0 X.out Q := by
  rw [St.out_push' hp hc hc1]; exact h

/-- **The generator-level statement that a `Prog` ends.** Whatever code the
    emitter had produced, if it ends in `P` with the slot count the emitter
    expects, the code with `p` emitted after it ends, `p` answering, in `Q` of
    the answer. A claim about emissions that are `Fine`. -/
def PE {α : Type} (cfg : Cfg) (P : Env → World → Prop) (p : Prog Slot Lvl α)
    (Q : α → Env → World → Prop) : Prop :=
  ∀ s P0, Fine (emitGo p s).2 →
    Ends cfg P0 s.out (fun Γ w => Γ.size = s.n ∧ P Γ w) →
    Ends cfg P0 (emitGo p s).2.out
      (fun Γ w => ∃ a, (emitGo p s).1 = some a ∧ Γ.size = (emitGo p s).2.n ∧ Q a Γ w)

section rulesE
variable {α β : Type} {cfg : Cfg}

theorem PE.ret {P : Env → World → Prop} {a : α} {Q : α → Env → World → Prop}
    (h : ∀ Γ w, P Γ w → Q a Γ w) : PE cfg P (.ret a) Q := by
  intro s P0 _ hs
  exact Ends.conseq hs (fun _ _ h => h) (fun Γ w ⟨hn, hP⟩ => ⟨a, rfl, hn, h Γ w hP⟩)

theorem PE.conseq {P P' : Env → World → Prop} {p : Prog Slot Lvl α} {Q Q' : α → Env → World → Prop}
    (h : PE cfg P p Q) (hP : ∀ Γ w, P' Γ w → P Γ w) (hQ : ∀ a Γ w, Q a Γ w → Q' a Γ w) :
    PE cfg P' p Q' := by
  intro s P0 hf hs
  exact Ends.conseq (h s P0 hf (Ends.conseq hs (fun _ _ h => h) fun _ _ ⟨hn, hp⟩ => ⟨hn, hP _ _ hp⟩))
    (fun _ _ h => h) fun _ _ ⟨a, ha, hn, hq⟩ => ⟨a, ha, hn, hQ a _ _ hq⟩

/-- **Sequencing.** -/
theorem PE.bind {P : Env → World → Prop} {p : Prog Slot Lvl α} {f : α → Prog Slot Lvl β}
    {Q : α → Env → World → Prop} {R : β → Env → World → Prop}
    (hp : PE cfg P p Q) (hf : ∀ a, PE cfg (Q a) (f a) R) :
    PE cfg P (p >>= f) R := by
  intro s P0 hfin hs
  rw [emitGo_bind] at hfin ⊢
  generalize he : emitGo p s = e at hfin ⊢
  obtain ⟨r, t⟩ := e
  have h1 := hp s P0 (by
    rw [he]
    cases r with
    | none => exact hfin
    | some a => exact (emitGo_cnt (f a) t).fine hfin) hs
  rw [he] at h1
  cases r with
  | none =>
      intro Γ w h0
      obtain ⟨k, Γ', w', _, a, ha, _⟩ := h1 Γ w h0
      cases ha
  | some a =>
      exact hf a t P0 hfin (Ends.conseq h1 (fun _ _ h => h) fun _ _ ⟨a', ha, hn, hq⟩ => by
        cases ha; exact ⟨hn, hq⟩)

/-- An operation that answers binds the next slot. -/
theorem PE.op {ty} {P : Env → World → Prop} {o : Op' Slot ty} {k : Slot ty → Prog Slot Lvl α}
    {Q : α → Env → World → Prop}
    (h : ∀ r, PE cfg (fun Γ w => ∃ Γ0 v, Γ = Γ0.push v ∧ Γ0.size = r ∧ P Γ0 w ∧
      evalOp w.mem Γ0 o.erase = some v) (k r) Q)
    (hv : ∀ Γ w, P Γ w → ∃ v, evalOp w.mem Γ o.erase = some v) :
    PE cfg P (.op o k) Q := by
  intro s P0 hf hs
  rw [emitGo_op] at hf ⊢
  refine h s.n _ P0 hf (St.out_stmtE hs (opE ?_))
  rintro Γ w ⟨hn, hP⟩
  obtain ⟨v, hv⟩ := hv Γ w hP
  exact ⟨v, hv, by simp [hn], Γ, v, rfl, hn, hP, hv⟩

theorem PE.store {ty} {P R : Env → World → Prop} {v : Slot ty} {a : Slot .i64} {k : Prog Slot Lvl α}
    {Q : α → Env → World → Prop}
    (hs : ∀ n, StmtE cfg (fun Γ w => Γ.size = n ∧ P Γ w) (.store ty v a) (fun Γ w => Γ.size = n ∧ R Γ w))
    (hk : PE cfg R k Q) : PE cfg P (.store v a k) Q := by
  intro s P0 hf h; rw [emitGo_store] at hf ⊢; exact hk _ P0 hf (St.out_stmtE h (hs s.n))

/-- A byte store. -/
theorem PE.istore8 {ty} {P R : Env → World → Prop} {v : Slot ty} {a : Slot .i64} {hi : ty.isInt = true}
    {k : Prog Slot Lvl α} {Q : α → Env → World → Prop}
    (hs : ∀ n, StmtE cfg (fun Γ w => Γ.size = n ∧ P Γ w) (.istore8 v a) (fun Γ w => Γ.size = n ∧ R Γ w))
    (hk : PE cfg R k Q) : PE cfg P (.istore8 v a hi k) Q := by
  intro s P0 hf h; rw [emitGo_istore8] at hf ⊢; exact hk _ P0 hf (St.out_stmtE h (hs s.n))

/-- A call to one of the program's own functions that answers. -/
theorem PE.callLocal {ps : List ClifTy} {res : Option ClifTy} {r : LocalRef ps res}
    {args : Vals Slot ps} {k : ResV Slot res → Prog Slot Lvl α}
    {P : Env → World → Prop} {R : Nat → Env → World → Prop} {Q : α → Env → World → Prop}
    (hres : res.isSome = true)
    (hs : ∀ n, StmtE cfg (fun Γ w => Γ.size = n ∧ P Γ w) (.call r.callee args.slots)
      (fun Γ w => Γ.size = n + 1 ∧ R n Γ w))
    (hk : ∀ v, PE cfg (R v) (k (resSlot res v)) Q) :
    PE cfg P (.callLocal r args k) Q := by
  intro s P0 hf h
  rw [emitGo_callLocal] at hf ⊢; unfold emitCallLocal at hf ⊢
  simp only [hres, if_true] at hf ⊢
  cases hc : r.callee <;> simp only [hc] at hf ⊢ <;> rw [hc] at hs <;>
    exact hk _ _ P0 hf (St.out_stmtE h (hs s.n))

/-- A call to one of the program's own functions that answers nothing. -/
theorem PE.callLocalVoid {ps : List ClifTy} {res : Option ClifTy} {r : LocalRef ps res}
    {args : Vals Slot ps} {k : ResV Slot res → Prog Slot Lvl α}
    {P R : Env → World → Prop} {Q : α → Env → World → Prop}
    (hres : res.isSome = false)
    (hs : ∀ n, StmtE cfg (fun Γ w => Γ.size = n ∧ P Γ w) (.callVoid r.callee args.slots)
      (fun Γ w => Γ.size = n ∧ R Γ w))
    (hk : PE cfg R (k (resSlot res 0)) Q) :
    PE cfg P (.callLocal r args k) Q := by
  intro s P0 hf h
  rw [emitGo_callLocal] at hf ⊢; unfold emitCallLocal at hf ⊢
  simp only [hres] at hf ⊢
  cases hc : r.callee <;> simp only [hc] at hf ⊢ <;> rw [hc] at hs <;>
    exact hk _ P0 hf (St.out_stmtE h (hs s.n))

/-- Reading the entry block's parameters spends nothing. -/
theorem PE.params {tys : List ClifTy} {P : Env → World → Prop} {k : Vals Slot tys → Prog Slot Lvl α}
    {Q : α → Env → World → Prop} (h : PE cfg P (k (carriesFrom 0 tys)) Q) :
    PE cfg P (.params tys k) Q := by
  intro s P0 hf hs; rw [emitGo_params] at hf ⊢; exact h s P0 hf hs


/-- **A branch ends**: its test evaluates, the arm it picks ends with its
    exports in scope, and what follows ends from the join. -/
theorem PE.ite {α : Type} {cfg : Cfg} {jTys : List ClifTy} {c : Cond Slot}
    {thn els : Prog Slot Lvl (Vals Slot jTys)} {k : Vals Slot jTys → Prog Slot Lvl α}
    {P : Env → World → Prop} {Q : α → Env → World → Prop}
    (Qt Qe : Vals Slot jTys → Env → World → Prop)
    (hthn : PE cfg (fun Γ w => ∃ Γ0 t f, Γ = Γ0.push (.sc t f) ∧ P Γ0 w ∧
      evalOp w.mem Γ0 (.icmp c.cc c.a c.b) = some (.sc t f) ∧ (f != 0) = true) thn Qt)
    (hels : ∀ ne, PE cfg (fun Γ w => ∃ Γ0 t f, Γ0.size + 1 ≤ ne ∧
      Γ = bindAt (Γ0.push (.sc t f)) ne [] ∧ P Γ0 w ∧
      evalOp w.mem Γ0 (.icmp c.cc c.a c.b) = some (.sc t f) ∧ (f != 0) = false) els Qe)
    (hk : ∀ nJ, PE cfg (fun Γ w => ∃ Γ' jv vs, Γ'.size ≤ nJ ∧ Γ = bindAt Γ' nJ vs ∧
      jv.slots.mapM (fun r => Γ'[r]?) = some vs ∧ (Qt jv Γ' w ∨ Qe jv Γ' w))
      (k (carriesFrom nJ jTys)) Q)
    (hc : ∀ Γ w, P Γ w → ∃ t f, evalOp w.mem Γ (.icmp c.cc c.a c.b) = some (.sc t f))
    (hjt : ∀ jv Γ w, Qt jv Γ w → ∃ vs, jv.slots.mapM (fun r => Γ[r]?) = some vs)
    (hje : ∀ jv Γ w, Qe jv Γ w → ∃ vs, jv.slots.mapM (fun r => Γ[r]?) = some vs) :
    PE cfg P (.ite c thn els k) Q := by
  intro s P0 hfin hpre
  rw [emitGo_ite] at hfin ⊢
  unfold emitIte at hfin ⊢
  have hP1 := St.out_stmtE (st := .op (.icmp c.cc c.a c.b)) hpre (opE
    (Q := fun Γ w => Γ.size = s.n + 1 ∧ ∃ Γ0 t f, Γ = Γ0.push (.sc t f) ∧ Γ0.size = s.n ∧ P Γ0 w ∧
      evalOp w.mem Γ0 (.icmp c.cc c.a c.b) = some (.sc t f))
    (fun Γ w ⟨hn, hP⟩ => by
      obtain ⟨t, f, hv⟩ := hc Γ w hP
      exact ⟨_, hv, by simp [hn], Γ, t, f, rfl, hn, hP, hv⟩))
  have hFo : (s.bind1 (.op (.icmp c.cc c.a c.b))).2.out = (s.stmt (.op (.icmp c.cc c.a c.b))).out := rfl
  have hFn : (s.bind1 (.op (.icmp c.cc c.a c.b))).2.n = s.n + 1 := rfl
  have hFf : (s.bind1 (.op (.icmp c.cc c.a c.b))).1 = s.n := rfl
  rw [← hFo] at hP1
  generalize s.bind1 (.op (.icmp c.cc c.a c.b)) = rF at hP1 hFo hFn hFf hfin ⊢
  obtain ⟨flag, s1⟩ := rF
  dsimp only at hP1 hFo hFn hFf hfin ⊢
  subst hFf
  have hc1 := St.flush_cur s1
  have hS1n : s1.flush.n = s.n + 1 := (St.flush_n s1).trans hFn
  have hS1out := St.out_flush s1
  rw [← hS1out] at hP1
  generalize s1.flush = S1 at hc1 hS1n hS1out hP1 hfin ⊢
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
  refine hk _ _ P0 hfin ?_
  have fX := (emitGo_cnt _ _).fine hfin
  have fP := Fine.of_push (S1 := S1) fX rfl hc1 hc1
  have hlt := Nat.lt_of_lt_of_le (needP_lt S1.out _) fP.2
  simp only [needP_ite, St.leave_fst] at hlt
  have fE : Fine sE := ⟨by simpa using fP.1, by omega⟩
  have fT : Fine sT := ⟨by simpa using cE.1 fE.1, by omega⟩
  have kT : slotsS S1.n sT.out = sT.n := cT.2.2 fT.1 fT.2 _ rfl
  have kE : slotsS sT.n sE.out = sE.n := cE.2.2 fE.1 fE.2 _ (by simp [slotsS])
  simp only [St.leave_fst, terminates_eq fT.2, terminates_eq fE.2]
  refine Ends.of_push (S1 := S1) ?_ rfl hc1 hc1
  refine endsE_append hP1 (endsE_cons (iteE ?_) (endsE_nil fun _ _ h => h))
  rintro _ w ⟨hsz, Γ0, t, f, rfl, hs0, hP, hv⟩
  have hflag : Sem.get (Γ0.push (.sc t f)) s.n = some (.sc t f) := by
    simp only [Sem.get]; rw [← hs0]; simp
  refine ⟨t, f, hflag, fun hb => ?_, fun hb => ?_⟩
  · have hj : slotsOf (slotsOf (Γ0.push (V.sc t f)).size sT.out) sE.out = sE.n := by
      rw [hsz, ← hS1n, slotsOf_eq fT.2, kT, slotsOf_eq fE.2, kE]
    rw [hj]
    have h3 := tT (At (Γ0.push (.sc t f)) w) (by rw [hTe]; exact fT)
      (endsE_nil (by rintro _ _ ⟨rfl, rfl⟩; exact ⟨hsz.trans hS1n.symm, Γ0, t, f, rfl, hP, hv, hb⟩))
    rw [hTe] at h3
    dsimp only at h3
    by_cases htT : termsS sT.out = true
    · intro Γ w' h0
      obtain ⟨k, Γ', w'', hr, _⟩ := h3 Γ w' h0
      exact absurd hr ((terms_no_ok cfg k).2 _ _ _ _ _ htT)
    · refine Ends.conseq h3 (fun _ _ h => h) ?_
      rintro Γ' w' ⟨jv, rfl, hsz', hq⟩
      obtain ⟨vs, hvs⟩ := hjt jv Γ' w' hq
      have hTE : sT.n ≤ sE.n := by have := slotsS_ge sE.out sT.n; rw [kE] at this; exact this
      refine ⟨vs, by simpa [Sem.get] using hvs, ?_, Γ', jv, vs, by omega, rfl, hvs, .inl hq⟩
      rw [bindAt_size, (mapM_len hvs).trans (Vals.slots_length jv)]
      simp [htT]
  · have hT : slotsOf (Γ0.push (V.sc t f)).size sT.out = sT.n := by
      rw [hsz, ← hS1n, slotsOf_eq fT.2, kT]
    have hj : slotsOf (slotsOf (Γ0.push (V.sc t f)).size sT.out) sE.out = sE.n := by
      rw [hT, slotsOf_eq fE.2, kE]
    rw [hj, hT]
    have hge : Γ0.size + 1 ≤ sT.n := by
      have := slotsS_ge sT.out S1.n; rw [kT] at this; omega
    have h3 := tE (At (bindAt (Γ0.push (.sc t f)) sT.n []) w) (by rw [hEe]; exact fE)
      (endsE_nil (by
        rintro _ _ ⟨rfl, rfl⟩
        exact ⟨by simp [bindAt_size], Γ0, t, f, hge, rfl, hP, hv, hb⟩))
    rw [hEe] at h3
    dsimp only at h3
    by_cases htE : termsS sE.out = true
    · intro Γ w' h0
      obtain ⟨k, Γ', w'', hr, _⟩ := h3 Γ w' h0
      exact absurd hr ((terms_no_ok cfg k).2 _ _ _ _ _ htE)
    · refine Ends.conseq h3 (fun _ _ h => h) ?_
      rintro Γ' w' ⟨jv, rfl, hsz', hq⟩
      obtain ⟨vs, hvs⟩ := hje jv Γ' w' hq
      refine ⟨vs, by simpa [Sem.get] using hvs, ?_, Γ', jv, vs, by omega, rfl, hvs, .inr hq⟩
      rw [bindAt_size, (mapM_len hvs).trans (Vals.slots_length jv)]
      simp [htE]

/-- **A top-tested loop ends, by invariant and measure.** The initial carries
    are in scope and satisfy `I`; from carries satisfying it the head ends in
    `Qh`, whose test evaluates; where the test says leave, the exits are in
    scope and satisfy `Xit`; where it says go on, the body ends with next
    carries in scope that satisfy `I` and lower `μ`. What follows starts from
    the exits bound after every slot of the loop. -/
theorem PE.loop {α : Type} {cfg : Cfg}
    {tys exitTys : List ClifTy} {β : Type} {init : Vals Slot tys}
    {head : Lvl exitTys tys → Vals Slot tys → Prog Slot Lvl (Cond Slot × Vals Slot exitTys × β)}
    {body : Lvl exitTys tys → Vals Slot tys → β → Prog Slot Lvl (Vals Slot tys)}
    {k : Vals Slot exitTys → Prog Slot Lvl α}
    {P : Env → World → Prop} {Q : α → Env → World → Prop}
    (I : Env → List V → World → Prop) (μ : Env → List V → World → Nat)
    (Qh : Env → List V → World → Cond Slot × Vals Slot exitTys × β → Env → World → Prop)
    (Xit : Env → Env → List V → World → Prop)
    (hinit : ∀ Γ w, P Γ w → ∃ cs, init.slots.mapM (fun r => Γ[r]?) = some cs ∧ I Γ cs w)
    (hhead : ∀ lv n0 Γ0 cs w0, PE cfg (fun Γh wh => Γ0.size = n0 ∧ Γh = bindAt Γ0 n0 cs ∧ wh = w0 ∧ I Γ0 cs w0)
      (head lv (carriesFrom n0 tys)) (Qh Γ0 cs w0))
    (hcmp : ∀ Γ0 cs w0 a Γ1 w1, Qh Γ0 cs w0 a Γ1 w1 →
      ∃ t f, evalOp w1.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) = some (.sc t f))
    (hexit : ∀ Γ0 cs w0 a Γ1 w1 t f, Qh Γ0 cs w0 a Γ1 w1 →
      evalOp w1.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) = some (.sc t f) →
      ((f != 0) == a.1.exitOnTrue) = true →
      ∃ vs, a.2.1.slots.mapM (fun r => (Γ1.push (.sc t f))[r]?) = some vs ∧
        Xit Γ0 (Γ1.push (.sc t f)) vs w1)
    (hbody : ∀ lv nb Γ0 cs w0 a, PE cfg
      (fun Γb wb => ∃ Γ1 t f, Γ1.size + 1 = nb ∧ Γb = bindAt (Γ1.push (.sc t f)) nb cs ∧
        Qh Γ0 cs w0 a Γ1 wb ∧
        evalOp wb.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) = some (.sc t f) ∧
        ((f != 0) == a.1.exitOnTrue) = false)
      (body lv (carriesFrom nb tys) a.2.2)
      (fun next Γ2 w2 => ∃ nx, next.slots.mapM (fun r => Γ2[r]?) = some nx ∧ I Γ0 nx w2 ∧
        μ Γ0 nx w2 < μ Γ0 cs w0))
    (hk : ∀ nE, PE cfg (fun Γ w => ∃ Γ0 Γb vs, Γ0.size ≤ nE ∧ Γ = bindAt Γb nE vs ∧
      Xit Γ0 Γb vs w) (k (carriesFrom nE exitTys)) Q) :
    PE cfg P (.loop init head body k) Q := by
  intro s P0 hfin hpre
  rw [emitGo_loop] at hfin ⊢
  unfold emitLoop at hfin ⊢
  have hc1 := St.flush_cur s
  have hS1n := St.flush_n s
  have hS1out := St.out_flush s
  generalize s.flush = S1 at hc1 hS1n hS1out hfin ⊢
  obtain ⟨hr, sH, hHe⟩ : ∃ hr sH, emitGo (head S1.depth (carriesFrom S1.n tys))
      ({ S1 with n := S1.n + tys.length, depth := S1.depth + 1 } : St).enter = (hr, sH) :=
    ⟨_, _, rfl⟩
  have cH := emitGo_cnt (head S1.depth (carriesFrom S1.n tys))
    ({ S1 with n := S1.n + tys.length, depth := S1.depth + 1 } : St).enter
  rw [hHe] at cH
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
  have hFf : (sH.bind1 (.op (.icmp c.cc c.a c.b))).1 = sH.n := rfl
  generalize sH.bind1 (.op (.icmp c.cc c.a c.b)) = rF at cF hFo hFn hFf hfin ⊢
  obtain ⟨flag, sH'⟩ := rF
  dsimp only at cF hFo hFn hFf hfin ⊢
  subst hFf
  obtain ⟨bd, sB, hBe⟩ : ∃ bd sB, emitGo (body S1.depth (carriesFrom sH'.n tys) x)
      ({ (St.leave ({ S1 with n := S1.n + tys.length, depth := S1.depth + 1 } : St) sH').2 with
          n := sH'.n + tys.length } : St).enter = (bd, sB) := ⟨_, _, rfl⟩
  have cB := emitGo_cnt (body S1.depth (carriesFrom sH'.n tys) x)
      ({ (St.leave ({ S1 with n := S1.n + tys.length, depth := S1.depth + 1 } : St) sH').2 with
          n := sH'.n + tys.length } : St).enter
  rw [hBe] at cB
  simp only [St.leave_n, hBe] at hfin ⊢
  dsimp only at cH cB
  refine hk _ _ P0 hfin ?_
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
  refine Ends.of_push (S1 := S1) ?_ rfl hc1 hc1
  rw [hS1out]
  refine endsE_append (R := fun Γ w => Γ.size = S1.n ∧ P Γ w)
    (Ends.conseq hpre (fun _ _ h => h) (fun Γ w ⟨hn, hP⟩ => ⟨hn.trans hS1n.symm, hP⟩))
    (endsE_cons ?_ (endsE_nil fun _ _ h => h))
  refine loopE (fun Γ0 cs w => cs.length = tys.length ∧ I Γ0 cs w) μ ?_ ?_
  · rintro Γ w ⟨_, hP⟩
    obtain ⟨cs, hcs, hI⟩ := hinit Γ w hP
    exact ⟨cs, by simpa [Sem.get] using hcs, (mapM_len hcs).trans (Vals.slots_length init), hI⟩
  rintro Γ0 w0 cs w ⟨hsz, _⟩ ⟨hlen, hI⟩
  simp only [St.leave_fst]
  rw [hsz, slotsOf_eq fH'.2, kH', slotsOf_eq fB.2, kB]
  have hle : Γ0.size ≤ sB.n := by
    have := slotsS_ge sH'.out (S1.n + tys.length)
    have := slotsS_ge sB.out (sH'.n + tys.length)
    omega
  have h1 := hhead S1.depth S1.n Γ0 cs w
    ({ S1 with n := S1.n + tys.length, depth := S1.depth + 1 } : St).enter
    (At (bindAt Γ0 S1.n cs) w) (by rw [hHe]; exact fH) (endsE_nil (by
      rintro _ _ ⟨rfl, rfl⟩
      refine ⟨?_, hsz, rfl, rfl, hI⟩
      simp [bindAt_size, hlen, hsz]))
  rw [hHe] at h1
  dsimp only at h1
  have h2 := St.out_stmtE (st := .op (.icmp c.cc c.a c.b)) h1 (opE
    (Q := fun Γ' w' => ∃ Γ1 t f, Γ' = Γ1.push (.sc t f) ∧ Γ1.size = sH.n ∧ Qh Γ0 cs w (c, exitR, x) Γ1 w' ∧
      evalOp w'.mem Γ1 (.icmp c.cc c.a c.b) = some (.sc t f))
    (fun Γ w' ⟨a, ha, hs1, hq⟩ => by
      cases ha
      obtain ⟨t, f, hv⟩ := hcmp Γ0 cs w _ Γ w' hq
      exact ⟨_, hv, Γ, t, f, rfl, hs1, hq, hv⟩))
  rw [← hFo] at h2
  refine Ends.conseq h2 (fun _ _ h => h) ?_
  rintro _ w1 ⟨Γ1, t, f, rfl, hs1, hq, hv⟩
  have hflag : Sem.get (Γ1.push (.sc t f)) sH.n = some (.sc t f) := by
    simp only [Sem.get]; rw [← hs1]; simp
  refine ⟨t, f, hflag, fun hx => ?_, fun hx => ?_⟩
  · obtain ⟨vs, hvs, hX⟩ := hexit Γ0 cs w (c, exitR, x) Γ1 w1 t f hq hv hx
    have hl : vs.length = exitTys.length := (mapM_len hvs).trans (Vals.slots_length exitR)
    exact ⟨vs, by simpa [Sem.get] using hvs, by rw [bindAt_size, hl], Γ0, _, vs, hle, rfl, hX⟩
  · have h3 := hbody S1.depth sH'.n Γ0 cs w (c, exitR, x)
      ({ (St.leave ({ S1 with n := S1.n + tys.length, depth := S1.depth + 1 } : St) sH').2 with
          n := sH'.n + tys.length } : St).enter
      (At (bindAt (Γ1.push (.sc t f)) sH'.n cs) w1)
      (by rw [hBe]; exact fB)
      (endsE_nil (by
        rintro _ _ ⟨rfl, rfl⟩
        exact ⟨by simp [bindAt_size, hlen], Γ1, t, f, by rw [hs1, hFn], rfl, hq, hv, by simpa using hx⟩))
    rw [hBe] at h3
    dsimp only at h3
    refine Ends.conseq h3 (fun _ _ h => h) ?_
    rintro Γ2 w2 ⟨a', ha', _, nx, hnx, hI', hμ⟩
    subst ha'
    exact ⟨nx, by simpa [Sem.get] using hnx, ⟨(mapM_len hnx).trans (Vals.slots_length a'), hI'⟩, hμ⟩

end rulesE

end AlgorithmLib.Prog
