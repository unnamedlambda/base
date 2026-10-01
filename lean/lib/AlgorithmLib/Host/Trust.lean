import AlgorithmLib.Host.Blocks
import AlgorithmLib.Host.Frames

/-!
# The trusted base, named

Everything a program proved here relies on that is *not* a theorem. Five items,
and the point of writing them down is that the list does not grow when you write
another program — which is the whole reason the property-test bridge sits at the
artifact rather than anywhere else.

Read with `compile_sound`: that theorem is what makes "prove it in Lean" a
legitimate way to learn something about the machine, and these are its side
conditions.

| | what | who owns it | how it is discharged |
|---|---|---|---|
| A1 | `Sem.evalOp` is Cranelift's instruction semantics | us | `HProgCorpus`, 365 cases |
| A2 | `Blocks` is Cranelift's block behavior | us | `HProgCorpus`, 12 CFG shapes |
| A3 | regions are disjoint, stores are ordered | the runtime | declared |
| A4 | Cranelift is compositional over A1/A2 | Cranelift | declared |
| A5 | the frames of the symbols a program calls | us | `HProgFrames.footprint` |

A1 and A2 are checked by *executing* small programs through the real Rust path
and comparing bytes — the artifact is deserialized, decoded, compiled by
Cranelift and run, so every layer is in the loop on every case. They are checks
of a finite, program-independent vocabulary: 45 instructions and a handful of
control-flow constructs. Adding a user program adds no cases.

There is no entry contract to assume. The caller's input and output buffers
reach a program as entry-block parameters, so there is no offset for the runtime
and the emitter to agree about, and the arity a function is called with is the
one its own entry block declares — checked by Cranelift when the function is
compiled, not asserted here.

A4 is the only item that finite testing cannot reach: it quantifies over
programs, and no number of sampled points settles it. The known way to close it
is per-artifact translation validation — a formal semantics of the target ISA, a
decoder for what Cranelift emitted, and a refinement check, as seL4 does for
C-to-binary. That is a research programme, not a task, and the decision here is
to depend on Cranelift the way one depends on any compiler.
-/

namespace AlgorithmLib.HProg.Trust

open AlgorithmLib.IR
open AlgorithmLib.HProg

/-- **A3 — the runtime's memory model.**

    The arena, the caller's input buffer and the caller's output buffer do not
    overlap, and a store is visible to a later load of the same address. The
    `Sem.Mem` model builds disjointness in by construction, which is what makes
    this an assumption rather than a consequence: in reality `data` and `out`
    are caller-supplied slices that nothing forces apart.

    Not testable from inside a program — a program that could detect overlap
    would already be relying on it. -/
axiom regions_disjoint_and_stores_ordered : True

/-- **A4 — Cranelift is compositional over A1 and A2.**

    Every instruction and every control-flow construct behaving as `Sem.evalOp`
    and `Blocks` say does not by itself make an arbitrary *combination* of them
    behave that way: an optimising compiler is free to break exactly there, and
    Cranelift does GVN, LICM and pattern-based instruction selection.

    This is the one assumption that no finite corpus can discharge, because it
    quantifies over programs rather than over the vocabulary. It is accepted the
    way a C programmer accepts their compiler — noting that Cranelift is
    Wasmtime's production backend, continuously fuzzed, differentially fuzzed
    against other engines, and with SMT verification of part of its lowering. -/
axiom cranelift_compositional : True

/-- The FFI a program assumes: not the 93 frames that exist, but the frames of
    the symbols it calls, which its own term determines.

    A program is only *complete* in this sense if every symbol it names has a
    declared frame — otherwise something it calls has no statement about what it
    may write, and no proof about it can be. -/
def assumes (c : Code) : List (String × Option Frame) := footprint c

/-- The trusted base of a program, rendered. A1–A4 are fixed; only A5 varies,
    and for most programs it is short or empty. -/
def report (c : Code) : String :=
  "A1 instruction semantics, A2 block semantics, A3 memory model, " ++
  "A4 compiler compositionality; A5: " ++ footprintReport c

end AlgorithmLib.HProg.Trust
