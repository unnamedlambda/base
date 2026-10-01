module
public import AlgorithmLib.Proof.Typestate
meta import AlgorithmLib.Proof.Typestate
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Proof.Calls` — a program's own calls at every depth

A program's own functions are a table of term functions, and a local call
means running the callee's term with its own calls one level shallower
(`termLocals`). A callee's `Summary` holds at every depth where its body, run
with its own calls at any depth, keeps the summary's promise
(`summary_all`): at depth 0 every call is stuck, and at depth `k + 1` the
body's triple at depth `k` gives it. A caller's theorem at depth `k` takes its
callees' summaries at `k`, so a call graph without cycles is discharged
callee first.
-/

namespace AlgorithmLib.HProg

open AlgorithmLib.IR
open AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare
open AlgorithmLib.Prog (Body emit)

/-- A body that answers nothing, as one of the program's term functions: the
    code it emits, taking what an entry point takes. -/
def Fn.ofBody (body : Body) : Fn :=
  { params := ptrParams, code := emit body, status := none, env := (Prog.run body).2.1 }

/-- **A callee's summary at every call depth**, from a triple for its body at
    every depth. -/
theorem summary_all {fns : Nat → Option Fn} {env : FnEnv} {steps i : Nat} {f : Fn} {args : List V}
    {W W' : World → Prop} (hf : fns i = some f)
    (h : ∀ k w, W w → Triple { env, steps, locals := termLocals fns env steps k } (At args.toArray w) f.code
      { ok := fun _ w' => W' w', faultOk := false }) :
    ∀ k, Summary (termLocals fns env steps k) i args W W'
  | 0 => summary_zero fns env steps i args W W'
  | k + 1 => summary_succ hf (h k)

end AlgorithmLib.HProg
