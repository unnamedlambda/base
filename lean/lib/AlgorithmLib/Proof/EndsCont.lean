module
public import AlgorithmLib.Proof.EndsRules
meta import AlgorithmLib.Proof.EndsRules
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Proof.EndsCont` — a loop body that goes round again itself

A loop's body may end by going round its loop again (`cont`) instead of
falling through to the back edge: each arm of a branch it ends with then jumps
to the loop's head itself, with no join between them. `EndsC cfg P c C` says
code `c` goes round its innermost loop: from every state `P` holds of, the run
jumps there at some fuel, with carries `C` holds of. Its rules: code that ends
then code that goes round (`endsC_append`), and a loop whose body goes round
(`loopEC`, by `iterEC`).

At the generator, `PC cfg lv P p C` says the code `p` emits inside the loop at
level `lv` goes round it: a jump to it (`PC.cont`), a branch whose arms both do
(`PC.ite`), and code that ends then code that does (`PC.bind`). A loop whose
body goes round ends (`PE.loopC`); `wpC` and `wpE_wloopC` are the same from one
state.
-/

namespace AlgorithmLib.HProg.Hoare

open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sem

/-- **A code goes round its innermost loop**: from every state `P` holds of,
    at some fuel, with carries `C` holds of. -/
def EndsC (cfg : Cfg) (P : Env → World → Prop) (c : Code) (C : List V → World → Prop) : Prop :=
  ∀ Γ w, P Γ w → ∃ k vs w', runCode k cfg Γ w c = .cont 0 vs w' ∧ C vs w'

theorem runCode_fin_mono {k k' : Nat} (hk : k ≤ k') {cfg : Cfg} {Γ : Env} {w : World} {c : Code}
    {r : CodeRes} (h : runCode k cfg Γ w c = r) (hr : r.fin) : runCode k' cfg Γ w c = r :=
  (monoAt k).2.2.2 cfg Γ w c r h hr k' hk

/-- A code that ends, then one that jumps: the two jump. -/
theorem runCode_append_cont {cfg : Cfg} {G : Code} {d : Nat} {vs : List V} {w2 : World} :
    ∀ (F : Code) (k k2 : Nat) (Γ Γ1 : Env) (w w1 : World),
    runCode k cfg Γ w F = .ok Γ1 w1 → runCode k2 cfg Γ1 w1 G = .cont d vs w2 →
    runCode (k + k2) cfg Γ w (F ++ G) = .cont d vs w2
  | [], k, k2, Γ, Γ1, w, w1, h1, h2 => by
      cases k with
      | zero => simp [runCode] at h1
      | succ k =>
          simp only [runCode, CodeRes.ok.injEq] at h1
          obtain ⟨rfl, rfl⟩ := h1
          exact runCode_fin_mono (by omega) h2 rfl
  | p :: ps, k, k2, Γ, Γ1, w, w1, h1, h2 => by
      cases k with
      | zero => simp [runCode] at h1
      | succ k =>
          simp only [runCode] at h1
          split at h1 <;> try cases h1
          rename_i Γa wa ha
          rw [show k + 1 + k2 = (k + k2) + 1 by omega, List.cons_append]
          simp only [runCode, runPiece_ok_mono (Nat.le_add_right k k2) ha]
          exact runCode_append_cont ps k k2 Γa Γ1 wa w1 h1 h2

/-- **A code that ends, then one that goes round.** -/
theorem endsC_append {cfg : Cfg} {P R : Env → World → Prop} {C : List V → World → Prop} {F G : Code}
    (h1 : Ends cfg P F R) (h2 : EndsC cfg R G C) : EndsC cfg P (F ++ G) C := by
  intro Γ w hP
  obtain ⟨k1, Γ1, w1, hr1, hR⟩ := h1 Γ w hP
  obtain ⟨k2, vs, w2, hr2, hC⟩ := h2 Γ1 w1 hR
  exact ⟨k1 + k2, vs, w2, runCode_append_cont F k1 k2 Γ Γ1 w w1 hr1 hr2, hC⟩

/-- **A piece that goes round, then anything**: the rest is never reached. -/
theorem endsC_cons {cfg : Cfg} {P : Env → World → Prop} {C : List V → World → Prop} {p : Piece} {ps : Code}
    (h : ∀ Γ w, P Γ w → ∃ k vs w', runPiece k cfg Γ w p = .cont 0 vs w' ∧ C vs w') :
    EndsC cfg P (p :: ps) C := by
  intro Γ w hP
  obtain ⟨k, vs, w', hr, hC⟩ := h Γ w hP
  exact ⟨k + 1, vs, w', by simp only [runCode, hr], hC⟩

/-- **A top-tested loop whose body goes round ends**: each trip from carries
    `P` holds of ends its condition prefix and either leaves, in `Q`, or its
    body goes round with next carries `P` holds of and the measure `μ`
    lower. -/
theorem iterEC {cfg : Cfg} {Γ : Env} {l : Loop} {pre body : List Piece} {n0 ab : Nat}
    (P : List V → World → Prop) (Q : Env → World → Prop) (μ : List V → World → Nat)
    (hstep : ∀ cs w, P cs w →
      ∃ k Γ1 w1 t f, runCode k cfg (bindAt Γ n0 cs) w pre = .ok Γ1 w1 ∧
        get Γ1 l.flag = some (.sc t f) ∧
        (((f != 0) == l.exitOnTrue) = true →
          ∃ vs, l.exitR.mapM (get Γ1) = some vs ∧ Q (bindAt Γ1 ab vs) w1) ∧
        (((f != 0) == l.exitOnTrue) = false →
          ∃ k2 next w2,
            runCode k2 cfg (bindAt Γ1 (slotsOf (n0 + l.pTys.length) pre) cs) w1 body = .cont 0 next w2 ∧
            P next w2 ∧ μ next w2 < μ cs w)) :
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
      obtain ⟨k2, next, w2, hb, hP', hμ⟩ := hcont hx'
      obtain ⟨k3, Γ', w', hi, hQ⟩ := ih next w2 (by omega) hP'
      refine ⟨max k (max k2 k3) + 1, Γ', w', ?_, hQ⟩
      simp only [iter, runCode_ok_mono (show k ≤ max k (max k2 k3) by omega) hpre, hflag, hx',
        Bool.false_eq_true, if_false, runCode_fin_mono (show k2 ≤ max k (max k2 k3) by omega) hb rfl]
      exact iter_mono (show k3 ≤ max k (max k2 k3) by omega) hi rfl

/-- **A top-tested loop whose body goes round, by invariant and measure.** -/
theorem loopEC {cfg : Cfg} {P Q : Env → World → Prop} {l : Loop} {pre body : Code}
    (I : Env → List V → World → Prop) (μ : Env → List V → World → Nat)
    (hinit : ∀ Γ w, P Γ w → ∃ cs, l.init.mapM (get Γ) = some cs ∧ I Γ cs w)
    (htrip : ∀ Γ w0 cs w, P Γ w0 → I Γ cs w →
      Ends cfg (At (bindAt Γ Γ.size cs) w) pre (fun Γ1 w1 => ∃ t f, get Γ1 l.flag = some (.sc t f) ∧
        (((f != 0) == l.exitOnTrue) = true → ∃ vs, l.exitR.mapM (get Γ1) = some vs ∧
          Q (bindAt Γ1 (slotsOf (slotsOf (Γ.size + l.pTys.length) pre + l.pTys.length) body) vs) w1) ∧
        (((f != 0) == l.exitOnTrue) = false →
          EndsC cfg (At (bindAt Γ1 (slotsOf (Γ.size + l.pTys.length) pre) cs) w1) body
            (fun next w2 => I Γ next w2 ∧ μ Γ next w2 < μ Γ cs w)))) :
    PieceE cfg P (.loop l pre body) Q := by
  intro Γ w hP
  obtain ⟨cs, hcs, hI⟩ := hinit Γ w hP
  obtain ⟨k, Γ', w', hi, hQ⟩ := iterEC (cfg := cfg) (Γ := Γ) (l := l) (pre := pre) (body := body)
    (n0 := Γ.size) (ab := slotsOf (slotsOf (Γ.size + l.pTys.length) pre + l.pTys.length) body)
    (I Γ) Q (μ Γ) (fun cs' w' hI' => by
      obtain ⟨k, Γ1, w1, hr, t, f, hflag, hx, hc⟩ := htrip Γ w cs' w' hP hI' _ _ ⟨rfl, rfl⟩
      refine ⟨k, Γ1, w1, t, f, hr, hflag, hx, fun h => ?_⟩
      obtain ⟨k2, next, w2, hb, hI2, hμ⟩ := hc h _ _ ⟨rfl, rfl⟩
      exact ⟨k2, next, w2, hb, hI2, hμ⟩) cs w hI
  exact ⟨k + 1, Γ', w', by simp only [runPiece, hcs, hi], hQ⟩

end AlgorithmLib.HProg.Hoare

namespace AlgorithmLib.HProg.Hoare

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem

theorem EndsC.conseq {cfg : Cfg} {P P' : Env → World → Prop} {C C' : List V → World → Prop} {c : Code}
    (h : EndsC cfg P c C) (hP : ∀ Γ w, P' Γ w → P Γ w) (hC : ∀ vs w, C vs w → C' vs w) : EndsC cfg P' c C' :=
  fun Γ w h' => by
    obtain ⟨k, vs, w', hr, hc⟩ := h Γ w (hP Γ w h')
    exact ⟨k, vs, w', hr, hC vs w' hc⟩

end AlgorithmLib.HProg.Hoare

namespace AlgorithmLib.Prog

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare

/-- **The code `p` emits inside the loop at level `lv` goes round it.**
    Whatever code the emitter had produced inside that loop's body, if it ends
    in `P` with the slot count the emitter expects, the code with `p` emitted
    after it goes round the loop with carries `C` holds of. -/
def PC {α : Type} (cfg : Cfg) (lv : Nat) (P : Env → World → Prop) (p : Prog Slot Lvl α)
    (C : List V → World → Prop) : Prop :=
  ∀ s P0, s.depth = lv + 1 → Fine (emitGo p s).2 →
    Ends cfg P0 s.out (fun Γ w => Γ.size = s.n ∧ P Γ w) →
    EndsC cfg P0 (emitGo p s).2.out C

section rulesC
variable {α β : Type} {cfg : Cfg} {lv : Nat}

/-- **Code that ends, then code that goes round.** -/
theorem PC.bind {P : Env → World → Prop} {p : Prog Slot Lvl α} {f : α → Prog Slot Lvl β}
    {Q : α → Env → World → Prop} {C : List V → World → Prop}
    (hp : PE cfg P p Q) (hf : ∀ a, PC cfg lv (Q a) (f a) C) : PC cfg lv P (p >>= f) C := by
  intro s P0 hd hfin hs
  rw [emitGo_bind] at hfin ⊢
  have hdp := emitGo_depth p s
  generalize he : emitGo p s = e at hfin hdp ⊢
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
      exact hf a t P0 (by simpa [hd] using hdp) hfin (Ends.conseq h1 (fun _ _ h => h) fun _ _ ⟨a', ha, hn, hq⟩ => by
        cases ha; exact ⟨hn, hq⟩)

/-- What follows code that goes round, answering at once, adds no code. -/
theorem PC.bind_ret {P : Env → World → Prop} {p : Prog Slot Lvl α} {g : α → β} {C : List V → World → Prop}
    (h : PC cfg lv P p C) : PC cfg lv P (p >>= fun a => .ret (g a)) C := by
  intro s P0 hd hfin hs
  have e : (emitGo (p >>= fun a => .ret (g a)) s).2 = (emitGo p s).2 := by
    rw [emitGo_bind]
    cases emitGo p s with
    | mk r t => cases r <;> rfl
  rw [e] at hfin ⊢
  exact h s P0 hd hfin hs

/-- **A jump round the loop at level `lv`**, with carries `C` holds of. -/
theorem PC.cont {ex ca : List ClifTy} {l : Lvl ex ca} {args : Vals Slot ca} {P : Env → World → Prop}
    {C : List V → World → Prop} (hl : l = lv)
    (h : ∀ Γ w, P Γ w → ∃ vs, args.slots.mapM (fun r => Γ[r]?) = some vs ∧ C vs w) :
    PC (α := α) cfg lv P (.cont l args) C := by
  intro s P0 hd _ hs
  rw [emitGo_cont]
  unfold emitJump
  have e : ({ s.flush with pieces := Piece.cont (labelDepth s.flush.depth l) args.slots :: s.flush.pieces } : St).out =
      s.out ++ [Piece.cont (labelDepth s.flush.depth l) args.slots] := by
    rw [← St.out_flush]
    unfold St.flush St.out
    cases hc : s.cur <;> simp [hc]
  rw [e, St.flush_depth, hd, hl, show labelDepth (lv + 1) lv = 0 by simp [labelDepth]]
  refine endsC_append hs (endsC_cons ?_)
  rintro Γ w ⟨-, hP⟩
  obtain ⟨vs, hvs, hC⟩ := h Γ w hP
  exact ⟨1, vs, w, by simp only [runPiece, show args.slots.mapM (get Γ) = some vs by simpa [Sem.get] using hvs], hC⟩

/-- **A branch whose arms both go round**, what follows it answering at once
    (`ifte`):
    the arm its flag picks goes round. -/
theorem PC.ite {jTys : List ClifTy} {c : Cond Slot} {thn els : Prog Slot Lvl (Vals Slot jTys)}
    {g : Vals Slot jTys → α}
    {P : Env → World → Prop} {C : List V → World → Prop}
    (hthn : PC cfg lv (fun Γ w => ∃ Γ0 t f, Γ = Γ0.push (.sc t f) ∧ P Γ0 w ∧
      evalOp w.mem Γ0 (.icmp c.cc c.a c.b) = some (.sc t f) ∧ (f != 0) = true) thn C)
    (hels : ∀ ne, PC cfg lv (fun Γ w => ∃ Γ0 t f, Γ0.size + 1 ≤ ne ∧
      Γ = bindAt (Γ0.push (.sc t f)) ne [] ∧ P Γ0 w ∧
      evalOp w.mem Γ0 (.icmp c.cc c.a c.b) = some (.sc t f) ∧ (f != 0) = false) els C)
    (hc : ∀ Γ w, P Γ w → ∃ t f, evalOp w.mem Γ (.icmp c.cc c.a c.b) = some (.sc t f)) :
    PC cfg lv P (.ite c thn els (fun j => .ret (g j))) C := by
  intro s P0 hd hfin hpre
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
  have hFd : (s.bind1 (.op (.icmp c.cc c.a c.b))).2.depth = s.depth := rfl
  rw [← hFo] at hP1
  generalize s.bind1 (.op (.icmp c.cc c.a c.b)) = rF at hP1 hFo hFn hFf hFd hfin ⊢
  obtain ⟨flag, s1⟩ := rF
  dsimp only at hP1 hFo hFn hFf hFd hfin ⊢
  subst hFf
  have hc1 := St.flush_cur s1
  have hS1n : s1.flush.n = s.n + 1 := (St.flush_n s1).trans hFn
  have hS1d : s1.flush.depth = lv + 1 := (St.flush_depth s1).trans (hFd.trans hd)
  have hS1out := St.out_flush s1
  rw [← hS1out] at hP1
  generalize s1.flush = S1 at hc1 hS1n hS1d hS1out hP1 hfin ⊢
  obtain ⟨tR, sT, hTe⟩ : ∃ a b, emitGo thn S1.enter = (a, b) := ⟨_, _, rfl⟩
  have cT := emitGo_cnt thn S1.enter
  have dT := emitGo_depth thn S1.enter
  have tT := hthn S1.enter
  rw [hTe] at cT dT
  simp only [hTe] at hfin ⊢
  obtain ⟨eR, sE, hEe⟩ : ∃ a b, emitGo els (St.leave S1 sT).2.enter = (a, b) := ⟨_, _, rfl⟩
  have cE := emitGo_cnt els (St.leave S1 sT).2.enter
  have tE := hels sT.n (St.leave S1 sT).2.enter
  rw [hEe] at cE
  simp only [hEe, St.leave_n] at hfin ⊢
  dsimp only at cT cE dT
  simp only [emitGo_ret] at hfin ⊢
  have fP := Fine.of_push (S1 := S1) hfin rfl (by simp [hc1]) hc1
  have hlt := Nat.lt_of_lt_of_le (needP_lt S1.out _) fP.2
  simp only [needP_ite, St.leave_fst] at hlt
  have fE : Fine sE := ⟨by simpa using fP.1, by omega⟩
  have fT : Fine sT := ⟨by simpa using cE.1 fE.1, by omega⟩
  have kT : slotsS S1.n sT.out = sT.n := cT.2.2 fT.1 fT.2 _ rfl
  rw [St.out_push' (s := S1) rfl (by simp [hc1]) hc1]
  refine endsC_append hP1 (endsC_cons ?_)
  rintro _ w ⟨hsz, Γ0, t, f, rfl, hs0, hP, hv⟩
  have hflag : Sem.get (Γ0.push (.sc t f)) s.n = some (.sc t f) := by
    simp only [Sem.get]; rw [← hs0]; simp
  by_cases hb : (f != 0) = true
  · have h3 := tT (At (Γ0.push (.sc t f)) w) (by simpa [St.enter] using hS1d) (by rw [hTe]; exact fT)
      (endsE_nil (by rintro _ _ ⟨rfl, rfl⟩; exact ⟨hsz.trans hS1n.symm, Γ0, t, f, rfl, hP, hv, hb⟩))
    rw [hTe] at h3
    obtain ⟨k, vs, w', hr, hC⟩ := h3 _ w ⟨rfl, rfl⟩
    refine ⟨k + 1, vs, w', ?_, hC⟩
    simp only [runPiece, hflag, hb, if_true, St.leave_fst, hr]
  · have hb' : (f != 0) = false := by simpa using hb
    have hT : slotsOf (Γ0.push (V.sc t f)).size sT.out = sT.n := by
      rw [hsz, ← hS1n, slotsOf_eq fT.2, kT]
    have hge : Γ0.size + 1 ≤ sT.n := by
      have := slotsS_ge sT.out S1.n; rw [kT] at this; omega
    have h3 := tE (At (bindAt (Γ0.push (.sc t f)) sT.n []) w)
      (by simp [St.enter, St.leave, St.flush_depth, dT, hS1d])
      (by rw [hEe]; exact fE)
      (endsE_nil (by
        rintro _ _ ⟨rfl, rfl⟩
        exact ⟨by simp [bindAt_size], Γ0, t, f, hge, rfl, hP, hv, hb'⟩))
    rw [hEe] at h3
    obtain ⟨k, vs, w', hr, hC⟩ := h3 _ w ⟨rfl, rfl⟩
    refine ⟨k + 1, vs, w', ?_, hC⟩
    simp only [runPiece, hflag, hb', Bool.false_eq_true, if_false, St.leave_fst, hT, hr]

/-- **A top-tested loop whose body goes round ends, by invariant and
    measure**: as `PE.loop`, the body going round the loop itself with next
    carries that satisfy `I` and lower `μ`. -/
theorem PE.loopC {α : Type} {cfg : Cfg}
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
    (hbody : ∀ lv nb Γ0 cs w0 a, PC cfg lv
      (fun Γb wb => ∃ Γ1 t f, Γ1.size + 1 = nb ∧ Γb = bindAt (Γ1.push (.sc t f)) nb cs ∧
        Qh Γ0 cs w0 a Γ1 wb ∧
        evalOp wb.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) = some (.sc t f) ∧
        ((f != 0) == a.1.exitOnTrue) = false)
      (body lv (carriesFrom nb tys) a.2.2)
      (fun nx w2 => nx.length = tys.length ∧ I Γ0 nx w2 ∧ μ Γ0 nx w2 < μ Γ0 cs w0))
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
  have dH := emitGo_depth (head S1.depth (carriesFrom S1.n tys))
    ({ S1 with n := S1.n + tys.length, depth := S1.depth + 1 } : St).enter
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
  have hFf : (sH.bind1 (.op (.icmp c.cc c.a c.b))).1 = sH.n := rfl
  have hFd : (sH.bind1 (.op (.icmp c.cc c.a c.b))).2.depth = sH.depth := rfl
  generalize sH.bind1 (.op (.icmp c.cc c.a c.b)) = rF at cF hFo hFn hFf hFd hfin ⊢
  obtain ⟨flag, sH'⟩ := rF
  dsimp only at cF hFo hFn hFf hFd hfin ⊢
  have dH' : sH'.depth = S1.depth + 1 := by rw [hFd, dH]; rfl
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
  simp only [needP_loop, St.leave_fst] at hlt
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
  refine loopEC (fun Γ0 cs w => cs.length = tys.length ∧ I Γ0 cs w) μ ?_ ?_
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
      simp [bindAt_size, hlen]))
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
      (by simp [St.enter, St.leave, St.flush_depth, dH'])
      (by rw [hBe]; exact fB)
      (endsE_nil (by
        rintro _ _ ⟨rfl, rfl⟩
        exact ⟨by simp [bindAt_size, hlen], Γ1, t, f, by rw [hs1, hFn], rfl, hq, hv, by simpa using hx⟩))
    rw [hBe] at h3
    dsimp only at h3
    exact EndsC.conseq h3 (fun _ _ h => h) fun nx w2 ⟨hl, hI', hμ⟩ => ⟨⟨hl, hI'⟩, hμ⟩


theorem PC.conseq {P P' : Env → World → Prop} {p : Prog Slot Lvl α} {C C' : List V → World → Prop}
    (h : PC cfg lv P p C) (hP : ∀ Γ w, P' Γ w → P Γ w) (hC : ∀ vs w, C vs w → C' vs w) :
    PC cfg lv P' p C' := fun s P0 hd hf hs =>
  EndsC.conseq (h s P0 hd hf (Ends.conseq hs (fun _ _ h => h) fun _ _ ⟨hn, hp⟩ => ⟨hn, hP _ _ hp⟩))
    (fun _ _ h => h) hC

end rulesC

end AlgorithmLib.Prog

namespace AlgorithmLib.LibSpec

open AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare

/-- **`p`, run from `Γ` and `w` inside the loop at level `lv`, goes round it**,
    with carries `C` holds of. -/
def wpC {α : Type} (cfg : Cfg) (lv : Nat) (p : Prog Slot Lvl α) (C : List V → World → Prop) (Γ : Env)
    (w : World) : Prop :=
  PC cfg lv (At Γ w) p C

section
variable {α β : Type} {cfg : Cfg} {lv : Nat} {Γ : Env} {w : World}

/-- What goes round from each state a precondition allows goes round from all of them. -/
theorem PC.of_wpC {P : Env → World → Prop} {p : Prog Slot Lvl α} {C : List V → World → Prop}
    (h : ∀ Γ w, P Γ w → wpC cfg lv p C Γ w) : PC cfg lv P p C := by
  intro s P0 hd hf hs Γi wi h0
  obtain ⟨k, Γ1, w1, hr, hn, hP⟩ := hs Γi wi h0
  exact h Γ1 w1 hP s (At Γi wi) hd hf (fun Γ w ⟨e1, e2⟩ => by
    subst e1 e2; exact ⟨k, Γ1, w1, hr, hn, rfl, rfl⟩) Γi wi ⟨rfl, rfl⟩

theorem wpC_mono {p : Prog Slot Lvl α} {C C' : List V → World → Prop}
    (hC : ∀ vs w, C vs w → C' vs w) (h : wpC cfg lv p C Γ w) : wpC cfg lv p C' Γ w :=
  PC.conseq h (fun _ _ h => h) hC

/-- **Code that ends, then code that goes round.** -/
theorem wpC_bind {p : Prog Slot Lvl α} {f : α → Prog Slot Lvl β} {C : List V → World → Prop}
    (h : wpE cfg p (fun a => wpC cfg lv (f a) C) Γ w) : wpC cfg lv (p >>= f) C Γ w :=
  PC.bind h fun _ => PC.of_wpC fun _ _ h => h

theorem wpC_bind_ret {p : Prog Slot Lvl α} {g : α → β} {C : List V → World → Prop}
    (h : wpC cfg lv p C Γ w) : wpC cfg lv (p >>= fun a => .ret (g a)) C Γ w :=
  PC.bind_ret h

/-- **A jump round the loop at level `lv`**, its carries in scope. -/
theorem wpC_cont {ex ca : List ClifTy} {l : Lvl ex ca} {args : Vals Slot ca} {vs : List V}
    {C : List V → World → Prop} (hl : l = lv) (hvs : args.slots.mapM (fun r => Γ[r]?) = some vs)
    (hC : C vs w) : wpC (α := α) cfg lv (.cont l args) C Γ w :=
  PC.cont hl fun Γ' w' ⟨e1, e2⟩ => by subst e1 e2; exact ⟨vs, hvs, hC⟩

/-- **A branch on two integer slots whose arms both go round.** -/
theorem wpC_ifteT {t : ClifTy} {jTys : List ClifTy} {cc : ICmpCond} {a b : Slot t} {x y : UInt64}
    {thn els : Prog Slot Lvl (Vals Slot jTys)} {C : List V → World → Prop} (hti : t.isInt = true)
    (ha : Γ[a]? = some (.sc t x)) (hb : Γ[b]? = some (.sc t y))
    (ht : cmpInt cc t x y = true → wpC cfg lv thn C (Γ.push (boolV true)) w)
    (he : cmpInt cc t x y = false → ∀ ne, (Γ.push (boolV false)).size ≤ ne →
      wpC cfg lv els C (bindAt (Γ.push (boolV false)) ne []) w) :
    wpC cfg lv (ifte cc a b thn els) C Γ w := by
  have hev : evalOp w.mem Γ (.icmp cc a b) = some (boolV (cmpInt cc t x y)) := by
    simp [evalOp, Sem.get, ha, hb, zipIntCmp, hti]
  unfold wpC ifte
  refine PC.ite (g := id) ?_ ?_ ?_
  · refine PC.of_wpC fun Γ' w' ⟨Γ0, t', f, e, ⟨e1, e2⟩, hc, hf⟩ => ?_
    subst e1 e2 e
    rw [hev] at hc
    have hb' : V.sc t' f = boolV (cmpInt cc t x y) := (Option.some.inj hc).symm
    cases hcmp : cmpInt cc t x y
    · rw [hcmp] at hb'; simp [boolV] at hb'; obtain ⟨rfl, rfl⟩ := hb'; simp at hf
    · rw [hb', hcmp]; exact ht hcmp
  · intro ne
    refine PC.of_wpC fun Γ' w' ⟨Γ0, t', f, hne, e, ⟨e1, e2⟩, hc, hf⟩ => ?_
    subst e1 e2 e
    rw [hev] at hc
    have hb' : V.sc t' f = boolV (cmpInt cc t x y) := (Option.some.inj hc).symm
    cases hcmp : cmpInt cc t x y
    · rw [hb', hcmp]; exact he hcmp ne (by simpa using hne)
    · rw [hcmp] at hb'; simp [boolV] at hb'; obtain ⟨rfl, rfl⟩ := hb'; simp at hf
  · rintro Γ' w' ⟨rfl, rfl⟩
    rw [hev]
    unfold boolV
    exact ⟨_, _, rfl⟩

/-- **A loop of any carries whose body goes round ends**, by an invariant of
    the carries and the world and a measure of them each trip lowers: as
    `wpE_wloopN`, the body, given the loop's label, going round with next
    carries that meet the invariant and lower the measure. -/
theorem wpE_wloopC {tys exitTys : List ClifTy} {β : Type} {init : Vals Slot tys}
    {head : Vals Slot tys → Prog Slot Lvl (Cond Slot × Vals Slot exitTys × β)}
    {body : Lvl exitTys tys → Vals Slot tys → β → Prog Slot Lvl (Vals Slot tys)}
    {Q : Vals Slot exitTys → Env → World → Prop}
    (Inv : List V → World → Prop) (μ : List V → World → Nat)
    (Hd : List V → World → Cond Slot × Vals Slot exitTys × β → Env → World → Prop)
    (X : List V → World → Prop)
    (h0 : ∃ cs, init.slots.mapM (fun r => Γ[r]?) = some cs ∧ Inv cs w)
    (hh : ∀ (wi : World) (cs : List V), Inv cs wi →
      wpE cfg (head (carriesFrom Γ.size tys))
        (fun a Γ1 w1 => Extends (bindAt Γ Γ.size cs) Γ1 ∧ Hd cs wi a Γ1 w1) (bindAt Γ Γ.size cs) wi)
    (hc : ∀ cs wi a Γ1 w1, Hd cs wi a Γ1 w1 →
      ∃ t f, evalOp w1.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) = some (.sc t f))
    (hx : ∀ cs wi a Γ1 w1 t f, Hd cs wi a Γ1 w1 → evalOp w1.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) = some (.sc t f) →
      ((f != 0) == a.1.exitOnTrue) = true →
      ∃ vs, a.2.1.slots.mapM (fun r => (Γ1.push (.sc t f))[r]?) = some vs ∧ X vs w1)
    (hb : ∀ lv cs wi a Γ1 wb t f, Hd cs wi a Γ1 wb → Extends (bindAt Γ Γ.size cs) Γ1 →
      evalOp wb.mem Γ1 (.icmp a.1.cc a.1.a a.1.b) = some (.sc t f) → ((f != 0) == a.1.exitOnTrue) = false →
      wpC cfg lv (body lv (carriesFrom (Γ1.size + 1) tys) a.2.2)
        (fun nx w3 => nx.length = tys.length ∧ Inv nx w3 ∧ μ nx w3 < μ cs wi)
        (bindAt (Γ1.push (.sc t f)) (Γ1.size + 1) cs) wb)
    (hk : ∀ Γb vs w' nE, Extends Γ Γb → Γ.size ≤ nE → X vs w' →
      Q (carriesFrom nE exitTys) (bindAt Γb nE vs) w') :
    wpE cfg (wloopL init (fun _ => head) body) Q Γ w := by
  unfold wpE wloopL
  refine PE.loopC (fun Γ0 cs w' => Γ0 = Γ ∧ Inv cs w') (fun _ cs w => μ cs w)
    (fun Γ0 cs wi a Γ1 w1 => Γ0 = Γ ∧ Extends (bindAt Γ Γ.size cs) Γ1 ∧ Hd cs wi a Γ1 w1)
    (fun Γ0 Γb vs w' => Γ0 = Γ ∧ Extends Γ Γb ∧ X vs w') ?_ ?_ ?_ ?_ ?_ ?_
  · rintro Γ' w' ⟨rfl, rfl⟩
    obtain ⟨cs, hcs, hI⟩ := h0
    exact ⟨cs, hcs, rfl, hI⟩
  · intro lv n0 Γ0 cs w0
    refine PE.of_wpE fun Γh wh ⟨hn, e, ew, e0, hinv⟩ => ?_
    subst e0 hn e ew
    exact wpE_mono (fun a Γ1 w1 ⟨hE, hd⟩ => ⟨rfl, hE, hd⟩) (hh wh cs hinv)
  · rintro Γ0 cs wi a Γ1 w1 ⟨-, -, hd⟩
    exact hc cs wi a Γ1 w1 hd
  · rintro Γ0 cs wi a Γ1 w1 t f ⟨e0, hE, hd⟩ hev hxx
    subst e0
    obtain ⟨vs, hvs, hX⟩ := hx cs wi a Γ1 w1 t f hd hev hxx
    exact ⟨vs, hvs, rfl, (((Extends.refl _).bindAt_of (Nat.le_refl _) cs).trans hE).push_of _, hX⟩
  · intro lv nb Γ0 cs wi a
    refine PC.of_wpC fun Γb wb ⟨Γ1, t, f, hnb, e, ⟨e0, hE, hd⟩, hev, hxx⟩ => ?_
    subst e0 hnb e
    exact wpC_mono (fun nx w3 ⟨hl, hI, hμ⟩ => ⟨hl, ⟨rfl, hI⟩, hμ⟩)
      (hb lv cs wi a Γ1 wb t f hd hE hev hxx)
  · intro nE
    exact PE.ret fun Γ' w' ⟨Γ0, Γb, vs, hle, e, e0, hE, hX⟩ => by
      subst e e0; exact hk Γb vs w' nE hE hle hX

end

end AlgorithmLib.LibSpec
