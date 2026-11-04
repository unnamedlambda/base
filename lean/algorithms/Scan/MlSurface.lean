import Scan.Core
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
    * `cublasSgemvResult` — what a cuBLAS matrix-vector product lands.  NVIDIA
      specifies no fold order; `Law.cublasIsMatvec` constrains it to one
      left-to-right `Float32` fold — expressible, but stronger than the vendor
      guarantees.
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
    `AlgorithmLib.ML.cublasSgemvResult,
    `AlgorithmLib.ML.cublasBatchedResult, `AlgorithmLib.ML.cublasSgemmResult,
    `Lean.opaqueId, `String.Internal.append, `System.Platform.getNumBits,
    `mixHash, `String.hash, `Float32.toString,
    `Float32.cos, `Float32.sin,
    -- `sinf`/`cosf`/`powf` as `Libm` states them
    `Float.toInt64, `Int64.toFloat ]
  -- The launch model's soundness on a shipped body reaches every arm of
  -- `Sem.evalOp`, the double-precision ones included, which no emitted kernel
  -- uses.
  ++ evalOpOpaque

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
    * `Honours` — that a realisation performs each declared vendor step as
      its law says: what a plan's `run` rests on for the steps it does not
      prove.
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
    `AlgorithmLib.ML.Honours,
    `LT.lt, `LE.le, `Eq, `Ne, `Nat.lt, `Nat.le, `Not, `Membership.mem ]

/-- **Obligations *derived* from the declared surface**, each paired with the
    theorem that derives it.  The scan checks that theorem exists, so this list
    cannot claim a discharge that is not there.

    A claim carrying one of these is still *conditional* — that is what a
    hypothesis is.  What the entry buys is that the condition is not an
    assumption about the world: something proves it from things already on the
    surface, and the scan names what. -/
def derivedObligations : List (Name × Name) := []

/-- **Open obligations: assumptions that are NOT on the agreed trust surface
    and are NOT derived from it.**

    Empty, and the point of listing it is that it can stop being.  Adding to it
    is a deliberate, reviewable act; `derivedObligations` above is where things
    go when they are closed. -/
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
