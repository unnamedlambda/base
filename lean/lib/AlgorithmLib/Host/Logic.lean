module
public import AlgorithmLib.Host.Sem
meta import AlgorithmLib.Host.Sem
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Host.Logic` — reasoning about a term's run

What a claim about a program says is how its loops run, and a loop's run in
`Sem` is fuel-indexed recursion. Two results make that usable.

* **`runCode_mono`**: a run that finishes at some fuel finishes the same way at
  every larger fuel. Fuel bounds a run; it never changes what a run that got
  through computes. So a claim can be proved at whatever fuel its argument
  wants and used at the fuel a caller has.
* **`iter_rule`**: the loop rule. An invariant on the carries and the world
  that the condition prefix and the body preserve, with a measure the body
  decreases, gives the loop's exit, at any fuel past the measure. This is the
  specification a top-tested loop --- and so every `wloop`, `forLoop`,
  `forLoopAcc` --- is used through.
-/

namespace AlgorithmLib.HProg.Sem

open AlgorithmLib.IR
open AlgorithmLib.HProg

/-- A run that did not get stuck or refused. -/
def CodeRes.fin : CodeRes → Bool
  | .stuck _ | .misuse _ | .fault _ => false
  | _ => true

/-- What `runCode_mono` says at fuel `k`, for the four functions at once. -/
def MonoAt (k : Nat) : Prop :=
  (∀ cfg Γ w p r, runPiece k cfg Γ w p = r → r.fin → ∀ k', k ≤ k' → runPiece k' cfg Γ w p = r) ∧
  (∀ cfg Γ w l pre body n0 ab cs r, iter k cfg Γ w l pre body n0 ab cs = r → r.fin →
      ∀ k', k ≤ k' → iter k' cfg Γ w l pre body n0 ab cs = r) ∧
  (∀ cfg Γ w l body n0 ab cs first r, dtrip k cfg Γ w l body n0 ab cs first = r → r.fin →
      ∀ k', k ≤ k' → dtrip k' cfg Γ w l body n0 ab cs first = r) ∧
  (∀ cfg Γ w c r, runCode k cfg Γ w c = r → r.fin → ∀ k', k ≤ k' → runCode k' cfg Γ w c = r)

theorem monoAt : ∀ k, MonoAt k
  | 0 => ⟨fun _ _ _ _ _ h hd => by subst h; simp [runPiece, CodeRes.fin] at hd,
          fun _ _ _ _ _ _ _ _ _ _ h hd => by subst h; simp [iter, CodeRes.fin] at hd,
          fun _ _ _ _ _ _ _ _ _ _ h hd => by subst h; simp [dtrip, CodeRes.fin] at hd,
          fun _ _ _ _ _ h hd => by subst h; simp [runCode, CodeRes.fin] at hd⟩
  | k + 1 => by
    obtain ⟨ihP, ihI, ihD, ihC⟩ := monoAt k
    -- A recursive result that is not stuck is the same at more fuel; a stuck
    -- one makes every caller stuck, which the `fin` hypothesis rules out.
    refine ⟨?_, ?_, ?_, ?_⟩
    -- runPiece
    · intro cfg Γ w p r h hd k' hk
      obtain ⟨k'', rfl⟩ : ∃ k'', k' = k'' + 1 := ⟨k' - 1, by omega⟩
      have hk'' : k ≤ k'' := by omega
      subst h
      cases p with
      | straight ss => rfl
      | br d args => rfl
      | cont d args => rfl
      | dloop l body =>
          simp only [runPiece] at hd ⊢
          cases hi : l.init.mapM (get Γ) with
          | none => rfl
          | some inits =>
              simp only [hi] at hd ⊢
              exact ihD _ _ _ _ _ _ _ _ _ _ rfl hd k'' hk''
      | loop l pre body =>
          simp only [runPiece] at hd ⊢
          cases hi : l.init.mapM (get Γ) with
          | none => rfl
          | some inits =>
              simp only [hi] at hd ⊢
              exact ihI _ _ _ _ _ _ _ _ _ _ rfl hd k'' hk''
      | ite m thn els thnR elsR =>
          simp only [runPiece] at hd ⊢
          split
          · rename_i f hf
            simp only [hf] at hd
            generalize hsel : (if f != 0 then (thn, thnR, Γ)
              else (els, elsR, bindAt Γ (slotsOf Γ.size thn) [])) = sel at hd ⊢
            obtain ⟨arm, exports, Γ0⟩ := sel
            dsimp only at hd ⊢
            cases hr : runCode k cfg Γ0 w arm with
            | stuck s => rw [hr] at hd; simp [CodeRes.fin] at hd
            | misuse s => rw [hr] at hd; simp [CodeRes.fin] at hd
            | fault s => rw [hr] at hd; simp [CodeRes.fin] at hd
            | brk d Γb vs w' => rw [ihC _ _ _ _ _ hr (by rfl) k'' hk'']
            | cont d vs w' => rw [ihC _ _ _ _ _ hr (by rfl) k'' hk'']
            | ok Γ' w' => rw [ihC _ _ _ _ _ hr (by rfl) k'' hk'']
          · rfl
    -- iter
    · intro cfg Γ w l pre body n0 ab cs r h hd k' hk
      obtain ⟨k'', rfl⟩ : ∃ k'', k' = k'' + 1 := ⟨k' - 1, by omega⟩
      have hk'' : k ≤ k'' := by omega
      subst h
      simp only [iter] at hd ⊢
      cases h1 : runCode k cfg (bindAt Γ n0 cs) w pre with
      | stuck s => rw [h1] at hd; simp [CodeRes.fin] at hd
      | misuse s => rw [h1] at hd; simp [CodeRes.fin] at hd
      | fault s => rw [h1] at hd; simp [CodeRes.fin] at hd
      | brk d Γb vs w' =>
          rw [ihC _ _ _ _ _ h1 (by rfl) k'' hk'']
          cases d <;> rfl
      | cont d vs w' =>
          rw [ihC _ _ _ _ _ h1 (by rfl) k'' hk'']
          rw [h1] at hd
          cases d with
          | zero => exact ihI _ _ _ _ _ _ _ _ _ _ rfl hd k'' hk''
          | succ d => rfl
      | ok Γ1 w1 =>
          rw [ihC _ _ _ _ _ h1 (by rfl) k'' hk'']
          rw [h1] at hd
          dsimp only at hd ⊢
          split
          · rename_i f hf
            simp only [hf] at hd
            split
            · rfl
            · rename_i hx
              simp only [hx] at hd
              cases h2 : runCode k cfg (bindAt Γ1 (slotsOf (n0 + l.pTys.length) pre) cs) w1 body with
              | stuck s => rw [h2] at hd; simp [CodeRes.fin] at hd
              | misuse s => rw [h2] at hd; simp [CodeRes.fin] at hd
              | fault s => rw [h2] at hd; simp [CodeRes.fin] at hd
              | brk d Γb vs w2 =>
                  rw [ihC _ _ _ _ _ h2 (by rfl) k'' hk'']
                  cases d <;> rfl
              | cont d vs w2 =>
                  rw [ihC _ _ _ _ _ h2 (by rfl) k'' hk'']
                  rw [h2] at hd
                  cases d with
                  | zero => exact ihI _ _ _ _ _ _ _ _ _ _ rfl hd k'' hk''
                  | succ d => rfl
              | ok Γ2 w2 =>
                  rw [ihC _ _ _ _ _ h2 (by rfl) k'' hk'']
                  rw [h2] at hd
                  dsimp only at hd ⊢
                  cases hn : l.cont.mapM (get Γ2) with
                  | none => rfl
                  | some next =>
                      simp only [hn] at hd ⊢
                      exact ihI _ _ _ _ _ _ _ _ _ _ rfl hd k'' hk''
          · rfl
    -- dtrip
    · intro cfg Γ w l body n0 ab cs first r h hd k' hk
      obtain ⟨k'', rfl⟩ : ∃ k'', k' = k'' + 1 := ⟨k' - 1, by omega⟩
      have hk'' : k ≤ k'' := by omega
      subst h
      simp only [dtrip] at hd ⊢
      cases first with
      | true =>
          simp only [if_true] at hd ⊢
          cases hg : l.guard with
          | none =>
              simp only [hg] at hd ⊢
              exact ihD _ _ _ _ _ _ _ _ _ _ rfl hd k'' hk''
          | some g =>
              simp only [hg] at hd ⊢
              split
              · rfl
              · rename_i c hc
                simp only [hc] at hd
                split
                · rename_i hcc
                  simp only [hcc, if_true] at hd
                  exact ihD _ _ _ _ _ _ _ _ _ _ rfl hd k'' hk''
                · rfl
      | false =>
          simp only [Bool.false_eq_true, if_false] at hd ⊢
          cases h1 : runCode k cfg (bindAt Γ n0 cs) w body with
          | stuck s => rw [h1] at hd; simp [CodeRes.fin] at hd
          | misuse s => rw [h1] at hd; simp [CodeRes.fin] at hd
          | fault s => rw [h1] at hd; simp [CodeRes.fin] at hd
          | brk d Γb vs w' =>
              rw [ihC _ _ _ _ _ h1 (by rfl) k'' hk'']
              cases d <;> rfl
          | cont d vs w' =>
              rw [ihC _ _ _ _ _ h1 (by rfl) k'' hk'']
              rw [h1] at hd
              cases d with
              | zero => exact ihD _ _ _ _ _ _ _ _ _ _ rfl hd k'' hk''
              | succ d => rfl
          | ok Γ2 w2 =>
              rw [ihC _ _ _ _ _ h1 (by rfl) k'' hk'']
              rw [h1] at hd
              dsimp only at hd ⊢
              cases hn : l.cont.mapM (get Γ2) with
              | none => rfl
              | some next =>
                  simp only [hn] at hd ⊢
                  split
                  · rfl
                  · rename_i c hc
                    simp only [hc] at hd
                    split
                    · rename_i hcc
                      simp only [hcc, if_true] at hd
                      exact ihD _ _ _ _ _ _ _ _ _ _ rfl hd k'' hk''
                    · rfl
    -- runCode
    · intro cfg Γ w c r h hd k' hk
      obtain ⟨k'', rfl⟩ : ∃ k'', k' = k'' + 1 := ⟨k' - 1, by omega⟩
      have hk'' : k ≤ k'' := by omega
      subst h
      cases c with
      | nil => rfl
      | cons p ps =>
          simp only [runCode] at hd ⊢
          cases h1 : runPiece k cfg Γ w p with
          | stuck s => rw [h1] at hd; simp [CodeRes.fin] at hd
          | misuse s => rw [h1] at hd; simp [CodeRes.fin] at hd
          | fault s => rw [h1] at hd; simp [CodeRes.fin] at hd
          | brk d Γb vs w' => rw [ihP _ _ _ _ _ h1 (by rfl) k'' hk'']
          | cont d vs w' => rw [ihP _ _ _ _ _ h1 (by rfl) k'' hk'']
          | ok Γ' w' =>
              rw [ihP _ _ _ _ _ h1 (by rfl) k'' hk'']
              rw [h1] at hd
              exact ihC _ _ _ _ _ rfl hd k'' hk''

/-- **More fuel changes no finished run.** -/
theorem runCode_mono {k k' : Nat} (hk : k ≤ k') {cfg : Cfg} {Γ : Env} {w : World} {c : Code}
    {r : CodeRes} (h : runCode k cfg Γ w c = r) (hr : r.fin) : runCode k' cfg Γ w c = r :=
  (monoAt k).2.2.2 cfg Γ w c r h hr k' hk

theorem iter_mono {k k' : Nat} (hk : k ≤ k') {cfg : Cfg} {Γ : Env} {w : World} {l : Loop}
    {pre body : List Piece} {n0 ab : Nat} {cs : List V} {r : CodeRes}
    (h : iter k cfg Γ w l pre body n0 ab cs = r) (hr : r.fin) :
    iter k' cfg Γ w l pre body n0 ab cs = r :=
  (monoAt k).2.1 cfg Γ w l pre body n0 ab cs r h hr k' hk

/-- **The loop rule.** `P` holds of the carries and the world at every trip;
    from there the condition prefix finishes and either the loop leaves, with
    `Q` of what it leaves with, or the body finishes with next carries that
    satisfy `P` again and lower `μ`. Then the loop leaves with `Q`, at any fuel
    past `K + μ`: `K` is what one trip's regions need, `μ` how many trips remain.

    Stated over `iter`, the recursion a `.loop` piece runs, so it applies to
    every top-tested loop the surface builds. -/
theorem iter_rule {cfg : Cfg} {Γ : Env} {l : Loop} {pre body : List Piece} {n0 ab : Nat}
    (P : List V → World → Prop) (Q : Env → World → Prop) (μ : List V → World → Nat) (K : Nat)
    (hstep : ∀ cs w, P cs w →
      ∃ Γ1 w1 t f, runCode K cfg (bindAt Γ n0 cs) w pre = .ok Γ1 w1 ∧
        get Γ1 l.flag = some (.sc t f) ∧
        (((f != 0) == l.exitOnTrue) = true →
          ∃ vs, l.exitR.mapM (get Γ1) = some vs ∧ Q (bindAt Γ1 ab vs) w1) ∧
        (((f != 0) == l.exitOnTrue) = false →
          ∃ Γ2 w2 next,
            runCode K cfg (bindAt Γ1 (slotsOf (n0 + l.pTys.length) pre) cs) w1 body = .ok Γ2 w2 ∧
            l.cont.mapM (get Γ2) = some next ∧ P next w2 ∧ μ next w2 < μ cs w)) :
    ∀ cs w, P cs w → ∀ k, K + μ cs w < k →
      ∃ Γ' w', iter k cfg Γ w l pre body n0 ab cs = .ok Γ' w' ∧ Q Γ' w' := by
  suffices h : ∀ n cs w, μ cs w < n → P cs w → ∀ k, K + μ cs w < k →
      ∃ Γ' w', iter k cfg Γ w l pre body n0 ab cs = .ok Γ' w' ∧ Q Γ' w' from
    fun cs w hP k hk => h _ cs w (Nat.lt_succ_self _) hP k hk
  intro n
  induction n with
  | zero => intro cs w hn; omega
  | succ n ih =>
    intro cs w hn hP k hk
    obtain ⟨k0, rfl⟩ : ∃ k0, k = k0 + 1 := ⟨k - 1, by omega⟩
    obtain ⟨Γ1, w1, t, f, hpre, hflag, hexit, hcont⟩ := hstep cs w hP
    have hpre' := runCode_mono (show K ≤ k0 by omega) hpre rfl
    simp only [iter, hpre', hflag]
    split
    · rename_i hx
      obtain ⟨vs, hvs, hQ⟩ := hexit hx
      rw [hvs]
      exact ⟨_, _, rfl, hQ⟩
    · rename_i hx
      obtain ⟨Γ2, w2, next, hb, hn', hP', hμ⟩ := hcont (by simpa using hx)
      rw [runCode_mono (show K ≤ k0 by omega) hb rfl]
      dsimp only
      rw [hn']
      exact ih next w2 (by omega) hP' k0 (by omega)

/-- **The loop rule for a bottom-tested loop**, once its guard has passed: the
    body finishes with next carries and a test value; while the test says go
    on, `P` holds of the next carries and `μ` is lower; when it says stop, the
    exit carries are there and `Q` holds of what the loop leaves with. -/
theorem dtrip_rule {cfg : Cfg} {Γ : Env} {l : DLoop} {body : List Piece} {n0 ab : Nat}
    (P : List V → World → Prop) (Q : Env → World → Prop) (μ : List V → World → Nat) (K : Nat)
    (hstep : ∀ cs w, P cs w →
      ∃ Γ2 w2 next t f, runCode K cfg (bindAt Γ n0 cs) w body = .ok Γ2 w2 ∧
        l.cont.mapM (get Γ2) = some next ∧ get Γ2 l.flag = some (.sc t f) ∧
        (((f != 0) == l.contOnTrue) = true → P next w2 ∧ μ next w2 < μ cs w) ∧
        (((f != 0) == l.contOnTrue) = false →
          ∃ outs, l.exitIdx.mapM (fun i => next[i]?) = some outs ∧ Q (bindAt Γ2 ab outs) w2)) :
    ∀ cs w, P cs w → ∀ k, K + μ cs w < k →
      ∃ Γ' w', dtrip k cfg Γ w l body n0 ab cs false = .ok Γ' w' ∧ Q Γ' w' := by
  suffices h : ∀ n cs w, μ cs w < n → P cs w → ∀ k, K + μ cs w < k →
      ∃ Γ' w', dtrip k cfg Γ w l body n0 ab cs false = .ok Γ' w' ∧ Q Γ' w' from
    fun cs w hP k hk => h _ cs w (Nat.lt_succ_self _) hP k hk
  intro n
  induction n with
  | zero => intro cs w hn; omega
  | succ n ih =>
    intro cs w hn hP k hk
    obtain ⟨k0, rfl⟩ : ∃ k0, k = k0 + 1 := ⟨k - 1, by omega⟩
    obtain ⟨Γ2, w2, next, t, f, hb, hnext, hflag, hgo, hstop⟩ := hstep cs w hP
    have hb' := runCode_mono (show K ≤ k0 by omega) hb rfl
    simp only [dtrip, Bool.false_eq_true, if_false, hb', hnext, hflag]
    split
    · rename_i hx
      obtain ⟨hP', hμ⟩ := hgo hx
      exact ih next w2 (by omega) hP' k0 (by omega)
    · rename_i hx
      obtain ⟨outs, houts, hQ⟩ := hstop (by simpa using hx)
      simp only [houts]
      exact ⟨_, _, rfl, hQ⟩

end AlgorithmLib.HProg.Sem
