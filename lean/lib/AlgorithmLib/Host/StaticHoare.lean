module
public import AlgorithmLib.Host.Static
meta import AlgorithmLib.Host.Static
public import AlgorithmLib.Host.Hoare
meta import AlgorithmLib.Host.Hoare
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The abstract interpreter as a frontend of the logic

A straight run the interpreter follows from `a` to `a'` is a triple between the
concrete states each describes (`static_stmts`). So the interpreter is one way
of proving triples: it settles code whose values it knows --- setup, straight
runs between loops --- and the logic's loop rule takes over where it would
have to unroll.
-/

namespace AlgorithmLib.HProg.Hoare

open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sem

/-- What the interpreter ends a straight run with, when it goes on. -/
def goOf : Option Static.AOut → Option Static.AS
  | some (.go a) => some a
  | _ => none

/-- **A straight run the interpreter follows is a triple.** -/
theorem static_stmts (cfg : Cfg) {ss : List Stmt} {a a' : Static.AS}
    (h : goOf (Static.aStmts a ss) = some a') : Stmts cfg (Static.Rel a) ss (Static.Rel a') := by
  intro Γ w hr
  have hs : Static.aStmts a ss = some (.go a') := by
    unfold goOf at h
    split at h
    · cases h; assumption
    · cases h
  have := Static.aStmts_sound cfg ss hr hs
  unfold OutSat
  cases hrun : runStmts cfg Γ w ss with
  | ok Γ' w' =>
      rw [hrun] at this
      obtain ⟨a'', he, hr'⟩ := this
      cases he
      exact hr'
  | stuck _ => trivial
  | misuse _ => rw [hrun] at this; exact this
  | fault _ => rfl

end AlgorithmLib.HProg.Hoare
