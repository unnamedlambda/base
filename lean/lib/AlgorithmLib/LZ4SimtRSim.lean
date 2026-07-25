import AlgorithmLib.LZ4Simt

namespace AlgorithmLib.LZ4Simt
open AlgorithmLib

/-- One machine step over the fixed program (`ret`/out-of-range are fixpoints). -/
def sstep (prog : Array SInstr) (st : SState) : SState :=
  match prog[st.pc]? with
  | none => st
  | some .ret => st
  | some i => sstepInstr prog i st

/-- `st'` is reached from `st` in exactly `n` steps. -/
def SReaches (prog : Array SInstr) : Nat → SState → SState → Prop
  | 0, st, st' => st = st'
  | n + 1, st, st' => SReaches prog n (sstep prog st) st'

theorem sreaches_one (prog : Array SInstr) (st : SState) :
    SReaches prog 1 st (sstep prog st) := rfl

/-- Straight-line composition. -/
theorem sreaches_trans (prog : Array SInstr) :
    ∀ (m n : Nat) (a b c : SState),
      SReaches prog m a b → SReaches prog n b c → SReaches prog (m + n) a c := by
  intro m
  induction m with
  | zero =>
      intro n a b c hab hbc
      have : a = b := hab
      subst this; simpa using hbc
  | succ k ih =>
      intro n a b c hab hbc
      have hab' : SReaches prog k (sstep prog a) b := hab
      rw [show k + 1 + n = (k + n) + 1 from by omega]
      exact ih n (sstep prog a) b c hab' hbc

/-- Run exactly `n` steps (for computing straight-line block end-states). -/
def snsteps (prog : Array SInstr) : Nat → SState → SState
  | 0, st => st
  | n + 1, st => snsteps prog n (sstep prog st)

/-- Running `m + n` symbolic steps is running `m`, then `n`. -/
theorem snsteps_add (prog : Array SInstr) :
    ∀ (m n : Nat) (st : SState),
      snsteps prog (m + n) st = snsteps prog n (snsteps prog m st) := by
  intro m
  induction m with
  | zero => intro n st; simp [snsteps]
  | succ m ih =>
      intro n st
      rw [Nat.succ_add]
      exact ih n (sstep prog st)

/-- `snsteps` is an `n`-step reach (the witness for straight-line blocks). -/
theorem sreaches_snsteps (prog : Array SInstr) :
    ∀ (n : Nat) (st : SState), SReaches prog n st (snsteps prog n st) := by
  intro n
  induction n with
  | zero => intro st; rfl
  | succ k ih => intro st; exact ih (sstep prog st)

end AlgorithmLib.LZ4Simt
