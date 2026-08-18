import ScanCore

/-!
  # The model stack's trust surface

  `ScanCore` is what a scan *is*; this is what the inference and training
  pipelines are allowed to rest on.  It lives here rather than in `ScanCore` so
  that the compressor — which shares the machinery and rests on a different
  base, with no floats and no cuBLAS — cannot borrow these allowances by
  importing the scanner.  `Lz4Scan` declares `lz4Surface` the same way, for the
  same reason.

  **To widen the surface you edit this file**, which is a reviewable diff
  against a list of names rather than a change in prose.
-/

open Lean

namespace TrustScan

/-- **The declared trust surface**, as data.

    * `Float32.*` / `float32Spec` — IEEE-754 primitives are opaque, which is
      what makes float associativity a stated law rather than a silent
      assumption.
    * `cublasSgemvResult` / `sgemmBatchedRow` — cuBLAS's results.  NVIDIA
      specifies no fold order; `Law.cublasIsMatvec` constrains the first to one
      left-to-right `Float32` fold — expressible, but stronger than the vendor
      guarantees — and *nothing* constrains the second, which is what
      `VendorKernel.lawless` and `Plan.lawlessCount` record.
    * `cublasBatchedResult` — what member `p` of a batched cuBLAS call lands.
      Opaque for the same reason as the row above, and constrained by
      `Law.cublasBatchedIsSomeReassoc`, the weak form: member `p` sums `p`'s
      own products in some association.  The strong form was measured false —
      a batched forward differs from an unbatched one in the last bits — so no
      closed form is claimed.  It reaches every claim that quantifies over the
      law registry.
    * `cublasSgemmResult` — what a vendor GEMM lands at one output element,
      at the batch of one every contraction here is issued at.  Constrained by
      `Law.cublasGemmIsSomeReassoc`, again the weak form: the element sums its
      own `k` products, each once, in some association.  That the emitted calls
      really are in that configuration is decided from their recovered
      arguments rather than assumed.
    * `uploadedValue` — what `cl_cuda_upload_ptr` leaves in device memory.
    * `Lean.opaqueId`, `String.Internal.append`, `System.Platform.getNumBits` —
      Lean plumbing, not ours.  The last is the platform word size, reached by
      `Std.HashMap`'s hashing; a generator that deduplicates kernels by hashing
      its statements puts it in the closure of every claim about the tape it
      produced.  It says nothing about floats, kernels or the model.
    * `mixHash`, `String.hash`, `Float32.toString` — the rest of that same
      hashing.  A generator that groups operations into kernels with a hash map
      keyed on their statements reaches all three, and `Float32.toString`
      because a statement carries literals.

      What makes them harmless is *where* they sit: the hash decides which
      operations end up in which group, and every guard is stated over the
      grouping that results.  A hash that collided differently would produce a
      different tape, and the guards would be about that one.  None of the three
      can make a claim true of a tape that is not the emitted tape. -/
def allowedOpaque : List Name :=
  [ `float32Spec, `Float32.add, `Float32.mul, `Float32.div, `Float32.neg,
    `Float32.exp, `Float32.sqrt, `Float32.pow, `Float32.decLe,
    `Float32.ofScientific, `Float32.toBits, `Float32.sub, `Float32.lt,
    `Float32.decLt, `Float32.ofBits, `Float32.beq,
    `AlgorithmLib.ML.cublasSgemvResult, `AlgorithmLib.ML.sgemmBatchedRow,
    `AlgorithmLib.ML.cublasBatchedResult, `AlgorithmLib.ML.cublasSgemmResult,
    `Qwen2Proven.Stage.uploadedValue,
    `Lean.opaqueId, `String.Internal.append, `System.Platform.getNumBits,
    `mixHash, `String.hash, `Float32.toString,
    -- Reached only by the launch model's soundness on a shipped body, whose
    -- statement quantifies over machine states: `Blocks.runInsts` interprets
    -- every `Inst`, so its closure carries every arm of `Sem.evalOp` — the
    -- double-precision ones included, which no emitted kernel uses.
    `floatSpec, `Float.add, `Float.sub, `Float.mul, `Float.neg, `Float.beq,
    `Float.decLe, `Float.decLt, `Float.ofBits, `Float.toBits,
    `Float.ofScientific, `Float.toFloat32, `Float.toUInt64,
    `Float32.cos, `Float32.sin, `Float32.toFloat ]

/-- **Hypotheses a claim may carry without comment.**

    * `AllHold` — the named float laws (`Law.combinerComm` &c.).  Float
      associativity/commutativity is trust-surface by agreement, and the Law
      mechanism is what keeps it *visible in the type*.
    * `CuBlasIsMatvec` — cuBLAS's fold order, likewise.  It is a
      *strictification* of `CuBlasIsSomeReassoc`, which is the honest statement
      of what the vendor promises; `Law.weakens_holds` is that relationship as a
      theorem, so a claim billing the strict one is billing the weak one too.
    * `CuBlasIsSomeReassoc` — that a vendor GEMV sums the right products in
      some association.  True of one call, and deliberately useless for chaining
      a sequence of them, which is what makes the strict law's role visible.
    * `Honours` — discharged by `Qwen2Proven.Stage.idealR_honours`.
    * `MetaFaithful` — Cranelift's `ushr`/`ishl`/`isub`/`store` mean what
      `Clif.DExp.eval` says, and the meta upload delivers its bytes unchanged.
      Both halves were already on the surface (seam 14 and `uploadedValue`);
      naming them together is what let `SmMeta` be discharged rather than
      assumed, and it is deliberately the *only* place this development says
      what those four instructions compute.
    * `LT.lt`, `Eq`, `Nat.le`, `Not` — index side-conditions (`i < D`, and
      `¬ (a < …)` selecting RoPE's upper half), not assumptions about the
      world.  `Not` is here for `rope_hi_is_spec`, whose upper-half selector
      carries one.
    * `Membership.mem` — a claim stated for every member of a shipped list
      carries `u ∈ theList` as its *scope*, not as an assumption: it says which
      kernels the theorem is about.  `Vit.vit_ptx_exact` is stated this way
      because it holds of each emitted group and there is no single group to
      name. -/
def allowedHyp : List Name :=
  [ `AlgorithmLib.ML.AllHold, `AlgorithmLib.ML.CuBlasIsMatvec,
    `AlgorithmLib.ML.CuBlasIsSomeReassoc,
    `AlgorithmLib.ML.Honours, `Qwen2NonVacuity.MetaFaithful,
    `LT.lt, `LE.le, `Eq, `Ne, `Nat.lt, `Nat.le, `Not, `Membership.mem ]

/-- **Obligations *derived* from the declared surface**, each paired with the
    theorem that derives it.  The scan checks that theorem exists, so this list
    cannot claim a discharge that is not there.

    A claim carrying one of these is still *conditional* — that is what a
    hypothesis is.  What the entry buys is that the condition is not an
    assumption about the world: something proves it from things already on the
    surface, and the scan names what.

    * `SmMeta` — what the host owes the softmax kernel about the meta buffer
      (`TAIL = CHUNKS·32`, `CHUNKS·32 + REM ≤ SEQ`, `0 < SEQ`).  Nothing in the
      kernel checks it; a violation makes two store passes overlap silently.

      Discharged by `Qwen2NonVacuity.smMeta_of_frag`, which chains three
      theorems: `Qwen2Common.metaStageFrag_emits` (these are the instructions
      the generator emits, from any builder state), `Qwen2Common.metaFrag_stores`
      (after them the store map holds `seq`, `seq >>> 5`, `(seq >>> 5) <<< 5`
      and `seq − ((seq >>> 5) <<< 5)` at the four slots the kernel reads), and
      `smMeta_of_stores` (those four relations *are* `SmMeta`, by linear
      arithmetic).

      What it rests on is `MetaFaithful` — that Cranelift's `ushr`/`ishl`/
      `isub`/`store` mean what `Clif.DExp.eval` says and that the upload
      delivers the bytes unchanged — plus `0 < seq`, at least one token.  Both
      halves of `MetaFaithful` sit on the surface in their own right (seam 14
      and `uploadedValue`), and they are the *only* thing between the emitted
      program and the kernel's assumption: the four expressions are read off
      the instruction stream rather than off the source. -/
def derivedObligations : List (Name × Name) :=
  [ (`Qwen2Proven.Stage.SmMeta, `Qwen2NonVacuity.smMeta_of_frag)
    -- the launch model's soundness on a shipped body asks that the machine
    -- entered with the parameter the compiled form declares.  Not assumed:
    -- exhibited at a concrete state.
    -- five pipelines share this surface, so the discharger is the general
    -- principle rather than one body's witness: the typing holds whenever the
    -- decidable check passes on the entry state.  Each instantiation supplies
    -- its own check, and each such witness is a root in its own scan.
  , (`AlgorithmLib.Clif.Check.TypesAgree,
     `AlgorithmLib.Clif.Check.typesAgree_of_check) ]

/-- **Open obligations: assumptions that are NOT on the agreed trust surface
    and are NOT derived from it.**

    Empty, and the point of listing it is that it can stop being.  Adding to it
    is a deliberate, reviewable act; `derivedObligations` above is where things
    go when they are closed.

    **What is *not* in this list because it is not a hypothesis of any claim.**

    The end-to-end chain is uniform: every step rests on the *same* trusted
    base, and none of it rests on a reading of the generator's source.  The
    layer loop is covered by theorems:

    * `Qwen2Common.tokenDriver_deviceOps` — the declared program performs
      `tokenOps`, the twenty-four repetitions coming from
      `HStmt.deviceOps_forN` rather than from a `List.replicate` someone typed;
    * `entryDriver_is_built`, `finalDriver_is_built`, `attnDriver_is_built`,
      `ffnDriver_is_built` — each declared fragment performs exactly the device
      writes a scan recovers from the CLIF the generator produced;
    * `Qwen2Common.infer_loop_is_layers` — the emitted blocks contain exactly
      one counted loop and its bound is `N_LAYERS`, read off the `icmp` and
      resolved the same way a launch's PTX slot is;
    * `Qwen2Common.infer_loop_body_calls`, `layer_fn_calls`,
      `leaf_fns_no_loops`, `Qwen2Common.final_no_loops` — the loop body
      dispatches to the layer function and nothing else, the layer function
      dispatches to the two halves, and nothing else in a decode step loops.

    What is left between those and "the program computes the token" is that
    Cranelift **executes** `jump`/`brif`/`call` as the recovered structure says.
    That is seam 14, and it is not a new assumption: `Clif.deviceOpsOf`
    describing a straight-line function's execution is already a claim of
    exactly the same kind, relied on by `attn_ops_are` and every theorem
    downstream of it.  The loop is not an *additional* gap on top of it.

    The remaining genuinely-open item is narrower and is about the *host model*,
    not this program: `HostIR`'s emitter does not cover `ExternArg.far` or
    `.opaque`, so `flatHI_sound` — the theorem that a compiled `HStmt`
    *executes* to its `deviceOps` — cannot yet be applied to `tokenDriver`.
    `ExternArg.FarFreeB` is exactly that boundary, decidably.  Closing it means
    threading `∀ b ∈ s.farBases, b < n ∧ b ≠ ptr.id ∧ e ⟨b⟩ = unknown` through
    six emit lemmas and `flatHI_sound`; preservation is the same
    `evalPure_frame` argument that already carries `e ptr = unknown`.

    Recording it here rather than in a document because this file is what the
    build reads. -/
def openObligations : List Name := []

/-- The inference/training surface: the four lists documented above. -/
def mlSurface : Surface :=
  { allowedOpaque      := allowedOpaque
    allowedHyp         := allowedHyp
    derivedObligations := derivedObligations
    openObligations    := openObligations }

/-- The inference/training pipelines' scan, at `mlSurface`.  The native roster
    is per-scan, not per-surface: five pipelines share this surface and each
    rests on the compiler for a different set of claims. -/
def runScan (label : String) (roots : List Name) (nativeRoster : List Name) : CoreM Unit :=
  runScanWith mlSurface label roots nativeRoster


end TrustScan
