module
public import Qwen2.Algorithm
meta import Qwen2.Algorithm
public import AlgorithmLib.Host.Program
meta import AlgorithmLib.Host.Program
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The Qwen2 artifact as a program of calling functions

Its orchestrators call its own functions: `main` sequences the loaders, the
tokenizer and the server, and the layer function calls attention and FFN.
`qwenFns` is the shipped bodies as a table of term functions; the functions the
artifact ships are what `compiledFns` makes of it (`qwen_shipped_compiled`,
by `rfl`); and every one passes the door `program_sound` asks for
(`qwenFns_retOk`). So **`qwen_locals_sound`**: at every call depth the shipped
program answers its own calls as its terms do, and `program_run_sound` applies
to a run of `main`.
-/

namespace Qwen2

open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.HProg

/-- A shipped body as a term function: the code and callee table it folds to. -/
def qwenFn (body : Prog.Body) : Fn :=
  let (c, e, _) := Prog.run body
  { params := ptrParams, code := c, status := none, env := e }

/-- The artifact's functions by index: the shipped bodies from `u0:1`, and the
    orchestrator at `u0:38`. -/
def qwenFns (i : Nat) : Option Fn :=
  if i = 38 then some (qwenFn (Prog.sequenceWrapper wrapperCallees))
  else if 1 ≤ i then (shippedBodies[i - 1]?).map qwenFn else none

theorem qwenFns_none (i : Nat) (h : 39 ≤ i) : qwenFns i = none := by
  have hlen : shippedBodies.length = 37 := by native_decide
  simp only [qwenFns, if_neg (show i ≠ 38 by omega), if_pos (show 1 ≤ i by omega)]
  rw [List.getElem?_eq_none (by omega)]
  rfl

/-- Every function the artifact ships passes the door, checked by evaluation. -/
theorem qwenFns_retOk_all :
    (List.range 39).all (fun i => (qwenFns i).all fun f => retOk f.params f.code f.status) = true := by
  native_decide

theorem qwenFns_retOk : ∀ i f, qwenFns i = some f → retOk f.params f.code f.status = true := by
  intro i f h
  by_cases hi : i < 39
  · have := List.all_eq_true.mp qwenFns_retOk_all i (List.mem_range.mpr hi)
    rw [h] at this
    simpa using this
  · rw [qwenFns_none i (by omega)] at h; cases h

/-- **The functions the artifact ships are the table's, compiled.** -/
theorem qwen_shipped_compiled (i : Nat) (hi : i < shippedBodies.length) :
    compiledFns qwenFns (i + 1) = some (Prog.stateOf (i + 1) shippedBodies[i]) := by
  have hne : i + 1 ≠ 38 ∨ i + 1 = 38 := by omega
  simp only [compiledFns, qwenFns]
  by_cases h38 : i + 1 = 38
  · have hlen : shippedBodies.length = 37 := by native_decide
    omega
  · rw [if_neg h38, if_pos (by omega)]
    simp only [Nat.add_sub_cancel, List.getElem?_eq_getElem hi, Option.map_some]
    rfl

/-- **The shipped program's own calls, compiled, answer as their terms do**, at
    every call depth. -/
theorem qwen_locals_sound (env : FnEnv) (steps : Nat) :
    ∀ k, (termLocals qwenFns env steps k).le (blockLocals (compiledFns qwenFns) k) :=
  program_sound qwenFns env steps qwenFns_retOk

end Qwen2
