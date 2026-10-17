module
public import AlgorithmLib.Host.Lifecycle
meta import AlgorithmLib.Host.Lifecycle
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# Every entry point answers a scalar

What an entry point answers, when it answers, is one scalar value
(`callBits_scalar`), so a program that passes an answer on to the next call
passes known bits, whatever the answer is.
-/

namespace AlgorithmLib.HProg.Contracts

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem

/-- An answer is a scalar of the type the signature gives through the call the
    model makes too, for every entry point but the thread spawn, which runs a function of the program. -/
theorem callOf_ffi_scalar {f : IR.Ffi} (hf : f ≠ .threadSpawn) {lc : Locals} {vs : List V} {w : World}
    {v : V} {w' : World} (h : callOf lc (.ffi f) vs w = some (some v, w')) :
    ∃ x, v = .sc (f.result.getD .i64) x := by
  have hs : (f == .threadSpawn) = false := by simpa using hf
  simp only [callOf, hs, if_false, Bool.false_eq_true, callImport, ofCname_cname, bind, Option.bind,
    callFfi] at h
  split at h
  · cases h
  · split at h
    · cases h
    · exact callBits_scalar f h

end AlgorithmLib.HProg.Contracts
