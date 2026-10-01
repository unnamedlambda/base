module
public import AlgorithmLib.Host.Program
meta import AlgorithmLib.Host.Program
public import AlgorithmLib.Host.Hoare
meta import AlgorithmLib.Host.Hoare
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Host.Summary` — what a call to the program's own function does

A local call ends as its callee's body ends (`Sem.failOf`), so a proof about
the caller needs only what the callee's own proof says: on these arguments,
from a world `W` holds, the call neither misuses nor faults, and where it
answers it leaves a world `W'` holds. That is a **`Summary`**, stated of
whatever `cfg.locals` means, so a caller's theorem takes its callees'
summaries as hypotheses.

For the term program they are discharged by call depth: at depth 0 every
local call is stuck, which every summary allows (`summary_zero`); at depth
`k + 1` a callee runs its body with its own calls at depth `k`, so a triple
for the body under those gives the summary (`summary_succ`).
-/

namespace AlgorithmLib.HProg

open AlgorithmLib.IR
open AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare

/-- A call to function `i` of the program on `args`, from a world `W` holds,
    neither misuses nor faults, and where it answers it leaves a world `W'`
    holds. -/
def Summary (L : Sem.Locals) (i : Nat) (args : List V) (W W' : World → Prop) : Prop :=
  ∀ w, W w → match L i args w with
    | .error f => FailOk (f.out : Outcome Env) false
    | .ok (_, w') => W' w'

/-- At call depth 0 every local call is stuck, which every summary allows. -/
theorem summary_zero (fns : Nat → Option Fn) (env : FnEnv) (steps i : Nat) (args : List V)
    (W W' : World → Prop) : Summary (termLocals fns env steps 0) i args W W' := by
  intro w _
  simp [termLocals, Sem.noLocals, Sem.Fail.out, FailOk]

/-- At depth `k + 1` a callee runs its body with its own calls at depth `k`:
    a triple for the body there, from each world `W` holds, is its summary. -/
theorem summary_succ {fns : Nat → Option Fn} {env : FnEnv} {steps k i : Nat} {f : Fn} {args : List V}
    {W W' : World → Prop} (hf : fns i = some f)
    (h : ∀ w, W w → Triple { env, steps, locals := termLocals fns env steps k } (At args.toArray w) f.code
      { ok := fun _ w' => W' w', faultOk := false }) :
    Summary (termLocals fns env steps (k + 1)) i args W W' := by
  intro w hw
  simp only [termLocals, hf]
  unfold Sem.callFn
  by_cases hlen : f.params.length ≠ args.length
  · simp [hlen, Sem.Fail.out, FailOk]
  · have hs := h w hw steps args.toArray w ⟨rfl, rfl⟩
    rw [if_neg hlen]
    cases hr : runCode steps { env, steps, locals := termLocals fns env steps k } args.toArray w f.code <;>
      rw [hr] at hs <;> simp_all [CodeRes.Sat, Sem.Fail.out, FailOk]

end AlgorithmLib.HProg
