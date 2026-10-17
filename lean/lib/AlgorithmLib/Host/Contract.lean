module
public import AlgorithmLib.Host.ExtLife
meta import AlgorithmLib.Host.ExtLife
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Host.Contract` — one contract table over every callee

A program calls the engine's entry points (`Ffi`) and functions of C libraries
directly (`Ext`). What a frontend may rely on about either is one value,
`Callee.contract`, of one structure, and one theorem, `contract_sound`, says
the table is true of the calls the model makes:

* `pre` --- when the call answers (`Contracts.Pre`, `ExtContracts.ExtPre`);
* `frame` --- the memory it may write (`Ffi.spec`, `Ext.frame`);
* `moves` --- the lifecycle parts it may change; a library call moves none;
* `after` --- what a call with given argument bits and answer leaves of a
  typestate (`after`, `extAfter`).

A callee with no row --- one of the program's own functions, machine code, an
atomic, a library call the typestate does not follow --- has no contract.
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
  | .ffi f => some ⟨Pre f, f.spec.frame, moves f, fun bits _ => after f bits⟩
  | .ext e => if e.aside then some ⟨ExtContracts.ExtPre e, e.frame, [], extAfter e⟩ else none
  | _ => none

/-- What a callee with a contract does on argument values. -/
def _root_.AlgorithmLib.IR.Callee.call : Callee → List V → World → Option (Option V × World)
  | .ffi f, vs, w => callFfi f vs w
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
  /-- An answer changes no byte outside the row's memory frame; for
      `fileRead`, whose frame is its answer, of files shorter than `2^64` bytes. -/
  writes : ∀ vs bits w r w', vs.mapM asBits = some bits → c.call vs w = some (r, w') →
    (c = .ffi .fileRead → ∀ p x, w.fs.get p = some x → x.size < 2 ^ 64) →
    ∀ a, ¬ k.frame.allows bits (r.bind asBits) a → Unchanged w.mem w'.mem a

theorem mapM_asBits_i64 : ∀ bits : List UInt64, (bits.map (V.sc .i64)).mapM asBits = some bits
  | [] => rfl
  | b :: bits => by simp [List.mapM_cons, asBits, mapM_asBits_i64 bits]

/-- A library call the typestate follows keeps every lifecycle part. -/
theorem LibStep.parts {e : Ext} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (st : LibStep e bits w r w') (p : Part) : p.get w' = p.get w := by
  cases p <;> simp only [Part.get, st.dev, st.lmdb, st.win, st.gpu, st.thread, st.display, st.cudaDevice,
    st.gpuAdapter, st.mem.1]

theorem callFfi_bits {f : IR.Ffi} {vs : List V} {bits : List UInt64} {w : World} {r : Option V} {w' : World}
    (hb : vs.mapM asBits = some bits) (h : callFfi f vs w = some (r, w')) : callBits f bits w = some (r, w') := by
  simpa [callFfi, hb] using h

/-- **The table is true of the model**, row by row. -/
theorem contract_sound {c : Callee} {k : Contract} (hk : c.contract = some k) : k.Sound c := by
  cases c with
  | ffi f =>
      cases hk
      exact
        { answers := fun bits w hp => by
            show (callFfi f (bits.map (V.sc .i64)) w).isSome = true
            rw [callFfi, mapM_asBits_i64]; exact pre_safe f bits w hp
          keeps := fun _ _ _ _ _ hb h => callBits_parts f (callFfi_bits hb h)
          leaves := fun _ _ _ _ _ _ hS hb h => after_sound hS (callFfi_bits hb h)
          writes := fun vs bits w r w' hb h hfs a hout =>
            callFfi_respects_frame f vs w r w' h bits hb (fun e => hfs (e ▸ rfl)) a hout }
  | ext e =>
      simp only [Callee.contract] at hk
      split at hk
      · rename_i ha
        cases hk
        exact
          { answers := fun bits w hp => ExtContracts.ext_pre_safe e bits w hp
            keeps := fun _ _ _ _ _ hb h p _ => (lib_step ha h hb).parts p
            leaves := fun _ _ _ _ _ _ hS hb h => extAfter_sound hS hb h
            writes := fun vs bits w r w' hb h _ a hout =>
              ExtContracts.ext_respects_frame e vs w r w' h bits hb a hout }
      · cases hk
  | _ => cases hk

end AlgorithmLib.HProg.Contracts
