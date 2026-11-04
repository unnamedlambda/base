import Scan.Core
import Lz4.Assumptions
import Lz4.HostLaunch
import Ship.Lz4Run
/-!
  # What the LZ4 compressor's claims actually rest on — computed, not documented

  `Lz4Assumptions` *states* the trusted base.  This file *computes* it: for each
  public claim it walks the transitive constant closure of the type and proof
  term, reports every reachable `axiom` and `opaque`, and reports the Prop-valued
  hypotheses each claim still carries.  Anything outside the surface declared
  below fails the build.

  This exists so that confirming what the compressor proves does not mean reading
  the source.  A refactor that quietly weakens a claim fails the build, and a
  theorem that is never instantiated at a shipped geometry is reported.  A scan
  sees that second kind of gap; an axiom count of the theorems you remembered to
  look at does not.

  To widen the surface you edit `lz4Surface` below, which is a reviewable diff.
-/

open Lean

namespace Lz4Scan

open TrustScan

/-- **The LZ4 surface.**  Deliberately not `mlSurface`: this development has no
    floats and no cuBLAS, and it must not be able to borrow those allowances.

    * `allowedOpaque` is Lean's own plumbing, and the float arms of the
      semantics every launch claim's closure reaches (`evalOpOpaque`), which
      the compressor never performs — no vendor kernel. Every other constant
      this development reaches is a definition or a theorem. The trusted base
      of `Lz4Assumptions` lives *below* Lean (the PTX model's fit to the GPU,
      `ptxas`, the driver), so it is invisible to a closure walk. That is the
      point of the ledger: those rows cannot be found by any scan. What *is*
      expressible as a hypothesis is listed below and reported per claim.

    * `allowedHyp` is the arithmetic side-conditions (a buffer large enough,
      block-size and address bounds), which geometry a claim is about, and the
      platform the PTX model describes (`HwOk`). These constrain the caller and
      name the platform; none is an assumption about the compressor. -/
def lz4Surface : Surface :=
  { allowedOpaque :=
      [ `Lean.opaqueId, `String.Internal.append
        -- Reached because `emit`'s diagnostics assemble a message string.
        -- `Op.name` keeps that to plain literals so the rendering machinery
        -- behind `repr` stays out of the closure.
      , `String.Internal.length, `String.Internal.pushn
        -- The checker's refusals render the facts and values they name with
        -- `repr` (`Check.absStmtsH`), reached through the host program's call
        -- trace (`warp_run`): message text on the refusing branch only.
      , `String.Internal.foldl, `String.Internal.isEmpty
      , .str (.str (.str (.num `_private.Init.Data.Format.Basic 0) "Std") "Format") "be"
        -- `sinf`/`cosf`/`powf`, which the foreign-call model evaluates as
        -- `Libm` states them: the conversions between doubles and integers
        -- its range reduction and `exp2` make.
      , `Float.toInt64, `Int64.toFloat
        -- The platform word size, reached by `Std.HashMap`'s hashing: the
        -- PTX lowering keeps the shipped kernel's facts in hash maps, so the
        -- artifact's PTX text puts it in the closure of every claim about the
        -- payload. It says nothing about the compressor or the model.
      , `System.Platform.getNumBits, `mixHash, `String.hash ]
      -- A launch claim quantifies over machine states, so it reaches every arm
      -- of `Sem.evalOp`, the float ones included, which this kernel never
      -- performs (`evalOpOpaque`).
      ++ evalOpOpaque
    allowedHyp :=
      [ `LT.lt, `LE.le, `Eq, `Ne, `Nat.lt, `Nat.le, `Not
        -- `b = 15 ∨ b = 16`: which shipped geometry a claim is about, its scope.
      , `Or
        -- a block's input and output regions as predicates over addresses: the
        -- scope of the generic launch lemmas, which `launch_roundtrips` and the
        -- shipped claims instantiate at the artifact's layout.
      , `Membership.mem
        -- the GPU's float rules (`HwOk`): each approximation 32 bits, one NaN.
        -- The platform the PTX model describes, which the compressor, having no
        -- floats, never reaches.
      , `AlgorithmLib.Kernel.PtxSound.HwOk
        -- `LoopCQ B inStride ws` is the emit loop's OWN invariant: `litAnchor ≤
        -- searchPos`, the guard register agrees with the guard, and the output
        -- budget holds. It is established at loop entry and PROVEN preserved by
        -- `loopCBody_Qadvance`. As a hypothesis it says "this is a loop-head
        -- state", not "the world behaves".
      , `AlgorithmLib.LZ4WarpDSL.LoopCQ
        -- the host triple's conditions, as the driver contracts state them:
        -- the device runs the launch as its machine says (`Honest`, here
        -- `lz4Machine`, which `lz4Machine_realized` shows the PTX model's own
        -- launch realizes), its kernels keep their buffers' sizes (`KeepsSize`),
        -- and the allocation is ready for the launch's stream (`Ready`). The
        -- GPU end of the trusted base, named in the claim's type.
      , `AlgorithmLib.HProg.Contracts.Honest, `AlgorithmLib.HProg.Sem.KeepsSize
      , `AlgorithmLib.HProg.DevSpec.Ready
        -- the whole host program's conditions: it starts from what `load`
        -- leaves (`Run`, the typestate the checker proves `load` reaches,
        -- with the caller's lengths), on a device honest about the kernel it
        -- loaded (`LaunchHonest`: `Honest` for `lz4Machine`).
      , `Lz4CompSafe.Run, `Lz4CompSafe.LaunchHonest ]
    derivedObligations :=
      -- the generic launch lemmas take the kernel's emission to be well formed;
      -- at the shipped geometries it is decided (`fine15`, `fine16`).
      [ (`AlgorithmLib.Prog.Fine, `Lz4Code.fine16) ]
    endpoints :=
      -- the shipped claim, and everything it is allowed to assume: which
      -- geometry, the platform, and a buffer large enough. Both geometries are
      -- one claim, so proving one and forgetting the other fails to elaborate.
      [ (`Lz4Code.shipped_roundtrips, [`Or, `Eq, `LE.le, `AlgorithmLib.Kernel.PtxSound.HwOk])
        -- …and over any positive number of launches, as the artifact repeats it
      , (`Lz4Code.shipped_launches, [`Or, `Eq, `LE.le, `LT.lt, `AlgorithmLib.Kernel.PtxSound.HwOk])
        -- …and through the driver, the device honest about the launch
      , (`Lz4Code.lz4_drv_launch, [`Or, `Eq, `LE.le, `AlgorithmLib.HProg.Contracts.Honest,
          `AlgorithmLib.HProg.Sem.KeepsSize, `AlgorithmLib.HProg.DevSpec.Ready])
        -- …and the whole emitted host program, upload to download
      , (`Lz4CompSafe.warp_run, [`Or, `Eq, `Lz4CompSafe.Run, `Lz4CompSafe.LaunchHonest]) ]
    reachedModules :=
      -- the compressor's proof modules whose every declaration a claim above
      -- uses: one that none does was either written and never wired in, or is
      -- dead.
      [ `Lz4.CodeBallot, `Lz4.CodeBlock, `Lz4.CodeBody, `Lz4.CodeContent, `Lz4.CodeGrid,
        `Lz4.CodeKernel, `Lz4.CodeLaunch, `Lz4.CodeLayout, `Lz4.CodeLoop, `Lz4.CodeShip,
        `Lz4.Assumptions, `Lz4.Host, `Lz4.HostLaunch ] }

/-- **The public claims.** Adding a claim here subjects it to the scan. -/
def roots : List Name :=
  [ -- the shipped claim, at both geometries the artifact emits
    `Lz4Code.shipped_roundtrips
  , `Lz4Code.roundtrips15
  , `Lz4Code.roundtrips16
    -- …and after the launches the artifact repeats, from the inputs it keeps
  , `Lz4Code.shipped_launches
    -- the same at any geometry the layout fits
  , `Lz4Code.launch_roundtrips
    -- each block's output after a launch is the warp's run of that block
  , `Lz4Code.launch_output
    -- the warp's run makes only the accesses the lanes need, each byte once
  , `Lz4Code.body_obligations
    -- the warp's run of the encoding prefix round-trips
  , `AlgorithmLib.LZ4WarpDSL.warpKernelDSL_body_roundtrips
    -- the device's machine: what the warp's runs denote, realized by the PTX
    -- model's own launch, and the host triple for `cuLaunchKernel` under it
  , `Lz4Code.lz4Machine_realized
  , `Lz4Code.lz4_drv_launch
    -- the emitted HOST program: both kernel parameters name one allocation
  , `Lz4Host.params_same_buffer
    -- …and every run of it carries the uploaded bytes through the launches
    -- to the download, the device honest about the kernel it loaded
  , `Lz4CompSafe.warp_run
    -- the payload image and the offsets derived from it cannot disagree
  , `Lz4Ship.payload_fits
    -- the ledger itself: this fails to elaborate if a named theorem moves
  , `Lz4Assumptions.anchors
  ]

end Lz4Scan

/-- Claims that rest on the compiler, via `native_decide`. -/
def nativeRoster : List Name :=
  -- the shipped kernel compiles and its emission is well formed, decided by
  -- evaluating the generator at the two shipped geometries
  [ `Lz4Code.shipped_roundtrips, `Lz4Code.roundtrips15, `Lz4Code.roundtrips16, `Lz4Code.shipped_launches,
    `Lz4Code.lz4Machine_realized, `Lz4Assumptions.anchors,
    -- the host program's calls, decided by the checker at the layout's
    -- offset (`bindOff_15`, `bindOff_16`)
    `Lz4CompSafe.warp_run ]

set_option maxHeartbeats 600000 in
open Lz4Scan TrustScan in


#eval runScanWith lz4Surface "lz4-compressor" roots nativeRoster
