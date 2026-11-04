module
public import AlgorithmLib.Host.ExtLife
meta import AlgorithmLib.Host.ExtLife
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Host.Contract` — one contract table over every callee

A program calls functions of C libraries directly (`Ext`). What a frontend may
rely on about one is one value, `Callee.contract`, of one structure, and one
theorem, `contract_sound`, says the table is true of the calls the model
makes:

* `pre` --- when the call answers (`ExtContracts.ExtPre`);
* `frame` --- the memory it may write (`Ext.frame`);
* `moves` --- the lifecycle parts it may change; a library call moves none;
* `after` --- what a call with given argument bits and answer leaves of a
  typestate (`extAfter`).

A callee with no row --- one of the program's own functions, a function of a
library linked into it, machine code, an atomic, a library call the typestate
does not follow --- has no contract.
-/

namespace AlgorithmLib.HProg.Contracts

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem

/-- **Everything known about a callee**, in one value: when it answers, the
    memory it may write, which lifecycle parts it may move, and what it leaves
    of a typestate. -/
structure Contract where
  pre : List UInt64 → World → Prop
  frame : Frame
  moves : List Part
  /-- What a call with these argument bits and this answer leaves of a
      typestate. -/
  after : List UInt64 → Option UInt64 → TState → TState

/-- **The contract table.** -/
def _root_.AlgorithmLib.IR.Callee.contract : Callee → Option Contract
  | .ext e => if e.aside then some ⟨ExtContracts.ExtPre e, e.frame, [], extAfter e⟩ else none
  | _ => none

/-- What a callee with a contract does on argument values. -/
def _root_.AlgorithmLib.IR.Callee.call : Callee → List V → World → Option (Option V × World)
  | .ext e, vs, w => extCall e vs w
  | _, _, _ => none

/-- The argument values a contract's precondition speaks of, from their bits. -/
def _root_.AlgorithmLib.IR.Callee.args : Callee → List UInt64 → List V
  | .ext e, bits => e.args bits
  | _, bits => bits.map (V.sc .i64)

/-- What it means for a row to be true of the calls the model makes. -/
structure Contract.Sound (c : Callee) (k : Contract) : Prop where
  /-- Under its precondition the call answers: it is not misuse. -/
  answers : ∀ bits w, k.pre bits w → (c.call (c.args bits) w).isSome = true
  /-- An answer keeps every part the row does not move. -/
  keeps : ∀ vs bits w r w', vs.mapM asBits = some bits → c.call vs w = some (r, w') →
    ∀ p, p ∉ k.moves → p.get w' = p.get w
  /-- An answer lands in what the row says it leaves of any typestate. -/
  leaves : ∀ S vs bits w r w', S.holds w → vs.mapM asBits = some bits → c.call vs w = some (r, w') →
    (k.after bits (r.bind asBits) S).holds w'
  /-- An answer changes no byte outside the row's memory frame. -/
  writes : ∀ vs bits w r w', vs.mapM asBits = some bits → c.call vs w = some (r, w') →
    ∀ a, ¬ k.frame.allows bits (r.bind asBits) a → Unchanged w.mem w'.mem a

/-- A library call the typestate follows keeps every lifecycle part. -/
theorem LibStep.parts {e : Ext} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (st : LibStep e bits w r w') (p : Part) : p.get w' = p.get w := by
  cases p <;> simp only [Part.get, st.dev, st.display, st.cudaDevice,
    st.gpuAdapter, st.mem.2.2, World.has, st.present]

/-- **The table is true of the model**, row by row. -/
theorem contract_sound {c : Callee} {k : Contract} (hk : c.contract = some k) : k.Sound c := by
  cases c with
  | ext e =>
      simp only [Callee.contract] at hk
      split at hk
      · rename_i ha
        cases hk
        exact
          { answers := fun bits w hp => ExtContracts.ext_pre_safe e bits w hp
            keeps := fun _ _ _ _ _ hb h p _ => (lib_step ha h hb).parts p
            leaves := fun _ _ _ _ _ _ hS hb h => extAfter_sound hS hb h
            writes := fun vs bits w r w' hb h a hout =>
              ExtContracts.ext_respects_frame e vs w r w' h bits hb a hout }
      · cases hk
  | _ => cases hk

end AlgorithmLib.HProg.Contracts
