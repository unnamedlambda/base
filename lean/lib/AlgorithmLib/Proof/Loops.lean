module
public import AlgorithmLib.Surface.Prog
meta import AlgorithmLib.Surface.Prog
public import AlgorithmLib.Host.Logic
meta import AlgorithmLib.Host.Logic
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# Loops, specified

`Sem.iter_rule` is the specification a top-tested loop is used through. This
applies it to what a combinator emits: `forLoopAcc` summing its counter. The
code is the emitted code (`sumCode_emitted`, by `rfl`), and the theorem says
what running it computes for *every* bound `n`, at every fuel past `n` --- the
shape a claim about a shipped loop takes.
-/

namespace AlgorithmLib.Proof

open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sem
open AlgorithmLib.Prog

/-- `Σ_{j < n} j`, over the integers modulo `2^64`. -/
def sumTo : Nat → UInt64
  | 0 => 0
  | m + 1 => sumTo m + UInt64.ofNat m

/-- The body: sum `0 … n-1` and answer it. -/
def sumBody : StatusBody := do
  let (n ::ᵥ .nil) ← entryParams [.i64]
  forLoopAcc n (← iconst64 0) fun i acc => iadd acc i

/-- What `sumBody` emits: the counter and accumulator seeded, then one loop
    carrying `[i, acc]` whose prefix tests `i < n` and whose body steps both. -/
def sumCode : Code :=
  [.straight [.op (.iconst .i64 0), .op (.iconst .i64 0)],
   .loop { pTys := [.i64, .i64], init := [2, 1], flag := 5, exitOnTrue := false,
           cont := [10, 8], exitR := [4], exitTys := [.i64] }
     [.straight [.op (.icmp .ult 3 0)]]
     [.straight [.op (.iadd 7 6), .op (.iconst .i64 1), .op (.iadd 6 9)]]]

theorem sumCode_emitted : (runAns sumBody [.i64]).2.1 = sumCode := by rfl

theorem sumBody_answers : (runAns sumBody [.i64]).1 = some 11 := by rfl

private theorem mask64 (x : UInt64) : x &&& widthMask .i64 = x := by
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_and, show (widthMask .i64).toNat = 2 ^ 64 - 1 from rfl,
    Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt x.toNat_lt]

/-- The slots one straight-line piece leaves, without unfolding the fuel. -/
theorem slotsOf_straight (n : Nat) (ss : List Stmt) : slotsOf n [.straight ss] = stmtsSlots n ss := by
  simp [slotsOf, slotsGo, HProg.fuel]

/-- Binding at exactly the environment's size appends. -/
theorem bindAt_size (Γ : Env) (vs : List V) : bindAt Γ Γ.size vs = Γ ++ vs.toArray := by
  simp [bindAt]

theorem ofInt_i64_one : ofInt .i64 1 = .sc .i64 1 := by
  simp only [ofInt, norm]; congr

theorem ofInt_i64_zero : ofInt .i64 0 = .sc .i64 0 := by
  simp only [ofInt, norm]; congr

/-- The value bound last by `bindAt`, read back at its slot. -/
theorem bindAt_last (Γ : Env) (n : Nat) (x : V) (h : Γ.size ≤ n) : (bindAt Γ n [x])[n]? = some x := by
  unfold bindAt
  have hs : (Γ.take n ++ Array.replicate (n - Γ.size) default).size = n := by
    simp; omega
  rw [Array.getElem?_append_right (by omega)]
  simp [hs]

/-- **The emitted loop sums its counter.** For every bound `n` and every fuel
    past it, the code `forLoopAcc` emits for `sumBody` finishes without touching
    the world, with `Σ_{j < n} j` in the slot the function answers. -/
theorem sumCode_sums (cfg : Cfg) (n : UInt64) (w : World) (k : Nat) (hk : n.toNat + 6 < k) :
    ∃ Γ, runCode k cfg #[.sc .i64 n] w sumCode = .ok Γ w ∧ Γ[11]? = some (.sc .i64 (sumTo n.toNat)) := by
  obtain ⟨m, rfl⟩ : ∃ m, k = m + 4 := ⟨k - 4, by omega⟩
  let Γ0 : Env := #[.sc .i64 n, .sc .i64 0, .sc .i64 0]
  let l : Loop := { pTys := [.i64, .i64], init := [2, 1], flag := 5, exitOnTrue := false,
                    cont := [10, 8], exitR := [4], exitTys := [.i64] }
  let pre : List Piece := [.straight [.op (.icmp .ult 3 0)]]
  let body : List Piece := [.straight [.op (.iadd 7 6), .op (.iconst .i64 1), .op (.iadd 6 9)]]
  -- The loop, by the rule.
  have hloop := iter_rule (cfg := cfg) (Γ := Γ0) (l := l) (pre := pre) (body := body)
    (n0 := 3) (ab := 11)
    (fun cs w' => w' = w ∧ ∃ i : Nat, i ≤ n.toNat ∧
        cs = [.sc .i64 (UInt64.ofNat i), .sc .i64 (sumTo i)])
    (fun Γ' w' => w' = w ∧ Γ'[11]? = some (.sc .i64 (sumTo n.toNat)))
    (fun cs _ => match cs with
      | [.sc _ x, _] => n.toNat - x.toNat
      | _ => 0)
    2
    (by
      rintro cs w' ⟨rfl, i, hi, rfl⟩
      have hi64 : (UInt64.ofNat i).toNat = i := by
        rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by have := n.toNat_lt; omega)]
      refine ⟨_, w', .i8, if UInt64.ofNat i < n then 1 else 0, rfl,
        by simp [Sem.get, bindAt, Γ0, l, boolV, cmpInt], ?_, ?_⟩
      · intro hx
        have hge : ¬ UInt64.ofNat i < n := by
          intro h; simp [h, l] at hx
        have hin : i = n.toNat := by
          have := (not_congr UInt64.lt_iff_toNat_lt).mp hge
          rw [hi64] at this; omega
        refine ⟨[.sc .i64 (sumTo i)], rfl, rfl, ?_⟩
        subst hin
        exact bindAt_last _ 11 _ (by simp [bindAt, Γ0])
      · intro hx
        have hlt : UInt64.ofNat i < n := by
          exact Classical.byContradiction fun h => by simp [h, l] at hx
        have hlt' : i < n.toNat := by
          have := UInt64.lt_iff_toNat_lt.mp hlt; rwa [hi64] at this
        refine ⟨?Γ2, w', [.sc .i64 (UInt64.ofNat (i + 1)), .sc .i64 (sumTo (i + 1))], ?hb, ?hn,
          ⟨rfl, i + 1, hlt', rfl⟩, ?hμ⟩
        case hb =>
          have e1 : slotsOf (3 + l.pTys.length) pre = 6 := by simp [slotsOf_straight, stmtsSlots, pre, l]
          have e0 : bindAt Γ0 3 [V.sc ClifTy.i64 (UInt64.ofNat i), V.sc ClifTy.i64 (sumTo i)]
              = #[.sc .i64 n, .sc .i64 0, .sc .i64 0, .sc .i64 (UInt64.ofNat i), .sc .i64 (sumTo i)] :=
            bindAt_size Γ0 _
          rw [e1, e0]
          simp [runCode, runPiece, runStmts, runStmt, evalOp, bin, Sem.get, body, bindAt, boolV,
            cmpInt, ofInt_i64_one, norm, mask64]
          rfl
        case hn =>
          simp [Sem.get, sumTo, mask64, UInt64.ofNat_add, Nat.add_comm, l]
        case hμ =>
          simp only
          rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by have := n.toNat_lt; omega), hi64]
          omega)
  obtain ⟨Γ', w', hiter, hw, h11⟩ := hloop [.sc .i64 0, .sc .i64 0] w
    ⟨rfl, 0, Nat.zero_le _, by simp [sumTo]⟩ (m + 1) (by simp; omega)
  rw [hw] at hiter
  refine ⟨Γ', ?_, h11⟩
  have h0 : runPiece (m + 3) cfg #[.sc .i64 n] w
      (.straight [.op (.iconst .i64 0), .op (.iconst .i64 0)]) = .ok Γ0 w := by
    simp [runPiece, runStmts, runStmt, evalOp, ofInt_i64_zero, Γ0]
  have hab : slotsOf (slotsOf (3 + [ClifTy.i64, .i64].length) pre + [ClifTy.i64, .i64].length) body
      = 11 := by simp [slotsOf_straight, stmtsSlots, pre, body]
  have hl : runPiece (m + 2) cfg Γ0 w (.loop l pre body)
      = iter (m + 1) cfg Γ0 w l pre body 3 11 [.sc .i64 0, .sc .i64 0] := by
    simp only [runPiece]
    rw [show Γ0.size = 3 from rfl, hab]
    rfl
  simp only [sumCode, runCode]
  rw [h0]
  dsimp only
  rw [hl, hiter]

end AlgorithmLib.Proof
