module
public import AlgorithmLib.Host.Frames
meta import AlgorithmLib.Host.Frames
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The trusted base, named

Everything a program proved here relies on that is *not* a theorem. Nine items,
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
| A5 | the frames of the symbols a program calls | us | proven for every entry point but `threadSpawn` and its shims, whose callee is the program's own (`callFfi_respects_frame`), and for every C library function (`ext_respects_frame`) |
| A6 | the C libraries do what their contracts state | their vendors | each precondition tagged by where it comes from (`Ext.sources`): *documented* (the vendor's stated rules), *observed* (a corpus agrees with the machine), *unmodelled* (outside what the model states, so a call there has no answer) |
| A7 | a GPU runs the kernel it is handed, and cuBLAS its routine | NVIDIA | `Honest` and `VendorKeeps` are hypotheses of every launch and routine triple, never facts |
| A8 | the OS kernel and the crates under the engine's adapters | the OS; the crates' authors | the adapters — files, threads, windows (`winit`), serial ports, USB, CPU topology, and wgpu (the `wgpu` crate) — are single calls into `std` and portable crates; their models are checked by the corpora on the machines they run on |
| A9 | the CLIF libraries (`Lib.Cuda`, `Lib.Ht`, `Lib.Math`, `Lib.Lmdb`, `Lib.Thread`) meet the entry points they replace | us | *observed*: every corpus and application runs the linked library against the entry point's model; `Lib.Math` is the one proven by construction (it performs `Libm`'s operations) |

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

A6–A8 are where the program meets code it does not own. The model states
each foreign call as a function of the world; what is trusted is that the
library or device does what that function says, where the precondition holds.
The tags make the trust inspectable per clause: a *documented* clause is only
as good as the vendor's documentation, an *observed* one is checked on the
machines the corpora run on, and an *unmodelled* one is a place the model
refuses to answer rather than guess.

A9 is the redesign's own debt: a program's theorems are about the program as
it names the engine's entry points, and the artifact links CLIF libraries in
their place. Until each library is proven to refine its entry point's model,
the corpora are the evidence.

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

/-- The FFI a program assumes: not the 89 frames that exist, but the frames of
    the symbols it calls, which its own term determines.

    A program is only *complete* in this sense if every symbol it names has a
    declared frame — otherwise something it calls has no statement about what it
    may write, and no proof about it can be. -/
def assumes (c : Code) : List (String × Option Frame) := footprint c

/-- **A6 — the C libraries' contracts.** A call that meets `ExtPre` does
    what `Sem.extCall` says: the library's own behaviour is read, not proven.
    `Ext.sources` says clause by clause where each reading comes from. -/
axiom c_libraries_as_contracted : True

/-- **A7 — the GPU and its vendor library.** A launch computes what its kernel
    denotes (`Honest`), and a cuBLAS routine what the vendor oracle says
    (`VendorKeeps`); both enter every theorem that uses them as hypotheses. -/
axiom gpu_runs_what_it_is_handed : True

/-- **A8 — the OS and the crates under the adapters.** The engine's adapters
    are single calls into `std` and portable crates, wgpu among them; the
    kernel and the crates beneath them are read through their models. -/
axiom os_kernel_as_modelled : True

/-- **A9 — the CLIF libraries refine the entry points they replace.** Observed
    by the corpora and applications, proven for `Lib.Math` by construction. -/
axiom libraries_refine_entry_points : True

/-- The trusted base of a program, rendered. A1–A4 and A6–A9 are fixed; only
    A5 varies, and for most programs it is short or empty. -/
def report (c : Code) : String :=
  "A1 instruction semantics, A2 block semantics, A3 memory model, " ++
  "A4 compiler compositionality, A6 C library contracts, A7 GPU and vendor oracle, " ++
  "A8 OS kernel, A9 library refinement; A5: " ++ footprintReport c

end AlgorithmLib.HProg.Trust
