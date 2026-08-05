import ScanCore
import Lz4CompAlgorithm
import Lz4Assumptions
import Lz4NonVacuity
import Lz4Launches
import Lz4Interleave
import Lz4Confine64
import Lz4Whole
import Lz4Host
import Lz4Sites
import Lz4Extend

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

    * `allowedOpaque` is Lean's own plumbing and nothing else — no float
      primitive, no vendor kernel.  Every other constant this development
      reaches is a definition or a theorem.  The trusted base of
      `Lz4Assumptions` lives *below* Lean (the printer, `ptxas`, the memory
      model, the Rust FFI), so it is invisible to a closure walk.  That is the
      point of the ledger: those rows cannot be found by any scan.  What *is*
      expressible as a hypothesis is listed below and reported per claim.

    * `allowedHyp` is the arithmetic side-conditions: block-size and pointer
      bounds (`w < nb`, `iS ≤ 65536`, `outPtr + w * oS + 9 * iS < 2 ^ 32`, …).
      These constrain the caller, not the world. -/
def lz4Surface : Surface :=
  { allowedOpaque := [`Lean.opaqueId, `String.Internal.append]
    allowedHyp :=
      [ `LT.lt, `LE.le, `Eq, `Ne, `Nat.lt, `Nat.le, `Not
        -- structural, not assumptions about the world: `Sim` is the interleaving
        -- invariant a step lemma carries, `outRegion` names a warp's byte range,
        -- `Halted` says a state is a fixpoint of the machine.
      , `AlgorithmLib.LZ4Simt.Sim, `AlgorithmLib.LZ4Simt.Halted
        -- `Reads`/`Writes` as hypotheses are the *antecedent* of a localisation
        -- lemma ("if this step touches memory, it does so at a known site"),
        -- not an assumption that it does.
      , `AlgorithmLib.LZ4Simt.Reads, `AlgorithmLib.LZ4Simt.Writes
        -- `s ∈ loadSites …` as a binder: the enumeration ranges over the list,
        -- it does not assume anything is in it.
      , `Membership.mem
      , `Lz4Interleave.outRegion
        -- `ValidStepsFrom inp anchor steps fl` is the plan's own well-formedness,
        -- PROVEN for everything the generator produces by `EvalValid.genLoop_valid`.
        -- As a hypothesis it says "this is a plan", not "the world behaves".
      , `AlgorithmLib.LZ4Plan.ValidStepsFrom
        -- `LoopCQ inStride ws` is the emit loop's OWN invariant: `litAnchor ≤
        -- searchPos`, the guard register agrees with the guard, and the output
        -- budget holds.  It is established at loop entry and PROVEN preserved by
        -- `loopCBody_Qadvance` — the very theorem that carries it here.  As a
        -- hypothesis it says "this is a loop-head state", not "the world behaves".
      , `AlgorithmLib.LZ4WarpDSL.LoopCQ
        -- `PcClosed p S exit` says a set of program points is closed under the
        -- emitted program's own control flow.  It is DECIDABLE at a concrete
        -- program and every use discharges it by `decide`; as a hypothesis it
        -- says "this is a region of this program", not "the world behaves".
      , `AlgorithmLib.LZ4Simt.PcClosed
        -- `LsicInv` is the tail LSIC loop's OWN region invariant (potential +
        -- loop guard + lane-uniformity), established at pc 222 and proven
        -- preserved on the region.  `Or` is the two-store case split.
      , `Lz4Sites.LsicInv, `Lz4Sites.LsicInvL, `Lz4Sites.LsicInvM, `Lz4Sites.TokInv, `Or
        -- `TailOOB b` is the two-instruction shape of the kernel's tail.  Like
        -- `PcClosed` it is decidable at a concrete program and every use
        -- discharges it — `Lz4Interleave.tail32` and `tail64`, both `decide`.
      , `Lz4Interleave.TailOOB
        -- `SchedComplete` names a schedule long enough for every warp to finish.
        -- It is CONSTRUCTED, not assumed: `schedComplete_exists` builds one from
        -- the step counts `ShippedCorrect` provides.
      , `Lz4Interleave.SchedComplete ]
    derivedObligations :=
      [ (`Algorithm.ShippedCorrect, `Algorithm.shipped32_correct)
        -- race-freedom is not assumed: it follows from `KernelConfined` plus the
        -- already-proven disjointness of the warps' output ranges.
      , (`AlgorithmLib.LZ4Simt.RaceFree, `Lz4Interleave.raceFree_of_confined)
        -- confinement is not a whole-program property: it follows from the
        -- ranges of the nine address registers, the site enumeration having
        -- eliminated the program.
      , (`Lz4Interleave.KernelConfined, `Lz4Sites.kernelConfined_of_regConfined32)
        -- a launch's frame is not assumed either: it is the second and third
        -- conclusions of the interleaving theorem.
      , (`Lz4Launches.LaunchFrame, `Lz4Interleave.launchFrame_of_confined)
        -- and the agreement half too: `pc272_unique` identifies the state
        -- `ShippedCorrect` returns with the one a schedule runs to.
      , (`Algorithm.LaunchAgreesPerWarp, `Lz4Interleave.launchAgrees_of_confined)
        -- and the bottom of that chain: the twelve load sites are confined by
        -- the kernel's own clamps, the sixteen store sites by the output
        -- cursor's budget.  No assumption is left about what an address
        -- register holds when it is used.
      , (`Lz4Sites.RegConfined, `Lz4Sites.regConfined_shipped)
        -- the cursor bound the ten `sbAddr` stores need, for the whole run of
        -- every warp
      , (`Lz4Sites.CursorAtSites, `Lz4Sites.cursorAtSites_shipped) ]
    openObligations :=
      [ `Algorithm.LayoutOK
        -- a hypothesis of the GENERIC lemmas (`launches_correct` and friends,
        -- which are stated about any run).  At the shipped geometries it is a
        -- CONCLUSION, built by `Lz4Interleave.launchesTo_of_layout`.
      , `Lz4Launches.LaunchesTo ] }


/-- **The public claims.**  Adding a claim here subjects it to the scan. -/
def roots : List Name :=
  [ -- the kernel, at the PTX machine: from raw launch, through the tail
    `AlgorithmLib.LZ4WarpDSL.warpKernelDSL_prologue_roundtrips
  , `AlgorithmLib.LZ4WarpDSL.warpKernelDSL_tail_roundtrips
    -- the shipped claim, at both geometries the artifact emits
  , `Algorithm.shipped32_correct
  , `Algorithm.shipped64_correct
    -- the proven half of data-race-freedom
  , `Algorithm.warp_regions_disjoint
    -- whole-launch: every block decodes out of the FINAL memory
  , `Algorithm.launch_correct
    -- …instantiated at the artifacts, so the top claim is not merely generic
  , `Lz4NonVacuity.shipped32_launch_correct
  , `Lz4NonVacuity.shipped64_launch_correct
    -- …and its contract is inhabited, so none of the above is vacuous
  , `Lz4NonVacuity.layoutOK_witness32
  , `Lz4NonVacuity.layoutOK_witness64
  , `Lz4NonVacuity.layoutOK_witness
    -- …and over the twenty repetitions the artifact actually runs, decoding to
    -- the input as the host uploaded it rather than to whatever launch 19 left
  , `Lz4Launches.input_preserved
  , `Lz4Launches.layoutOK_preserved
  , `Lz4Launches.launches_correct
  , `Lz4Launches.shipped32_launches_correct
  , `Lz4Launches.shipped64_launches_correct
    -- the memory model: any interleaving of race-free warps computes what the
    -- warps compute alone, so LaunchAgreesPerWarp/LaunchFrame are derived
  , `AlgorithmLib.LZ4Simt.sim_step
  , `AlgorithmLib.LZ4Simt.crun_sim
  , `AlgorithmLib.LZ4Simt.interleaving_agrees
  , `AlgorithmLib.LZ4Simt.schedule_completes
  , `Lz4Interleave.outRegion_disjoint
  , `Lz4Interleave.raceFree_of_confined
  , `Lz4Interleave.launch_agrees
  , `Lz4Interleave.launchFrame_of_confined
  , `Lz4Interleave.tail32
  , `Lz4Interleave.halt_after_272
  , `Lz4Interleave.pc272_unique
  , `Lz4Interleave.launchAgrees_of_confined
    -- the emitted HOST program: what device operations it performs, and that the
    -- launch geometry is the one the kernel proof assumes
  , `Lz4Host.host_ops32
  , `Lz4Host.host_ops64
  , `Lz4Host.host_grid_is_numBlk32
  , `Lz4Host.host_grid_is_numBlk64
  , `Lz4Host.host_calls32
  , `Lz4Host.host_calls64
  , `Lz4Host.host_single_allocation
  , `Lz4Host.bind_table_same_buffer
  , `Lz4Host.host_loop_is_rLaunches32
  , `Lz4Host.host_loop_is_rLaunches64
  , `Lz4Host.host_launch_in_loop_body32
  , `Lz4Host.host_launch_in_loop_body64
  , `Lz4Host.hostShape32
  , `Lz4Host.hostShape64
    -- where the kernel can touch global memory at all, enumerated from the
    -- shipped program, so `KernelConfined` is a located obligation
  , `Lz4Sites.reads_at_site
  , `Lz4Sites.writes_at_site
  , `Lz4Sites.shipped32_load_sites
  , `Lz4Sites.shipped32_store_sites
  , `Lz4Sites.shipped64_load_sites
  , `Lz4Sites.shipped64_store_sites
  , `Lz4Sites.load_regs32
  , `Lz4Sites.store_regs32
  , `Lz4Sites.kernelConfined_of_regConfined32
  , `Lz4Sites.unconditioned_form_is_false
  , `Lz4Sites.sbAddr_is_outBase_add_op
  , `Lz4Sites.la_at_store
  , `Lz4Sites.cpDo_at_store
  , `Lz4Sites.load_at_site
  , `Lz4Sites.outBase_at_store_site
    -- and what can move the cursor at all: thirteen writes, twelve of them
    -- accumulations, so `op` cannot jump — only be pushed past the budget
  , `Lz4Sites.shipped32_op_writes
  , `Lz4Sites.shipped32_op_accumulates
  , `Lz4Sites.shipped64_op_writes
  , `AlgorithmLib.LZ4WarpDSL.emitLoop_head_op_le
  , `AlgorithmLib.LZ4WarpDSL.loopCBody_op_le
  , `AlgorithmLib.LZ4WarpDSL.emitLoop_head_op_le_of_final
  , `Lz4Sites.sbAddr_confined_of_cursor
  , `Lz4Sites.outBase_const_after_prologue
  , `Lz4Sites.inBase_const_after_prologue
    -- control flow as data: successors read off the emitted array, so pc-shape
    -- facts are decided rather than hand-rolled per region
  , `AlgorithmLib.LZ4Simt.sstep_pc_mem_succs
  , `AlgorithmLib.LZ4Simt.pc_in_closed
  , `AlgorithmLib.LZ4Simt.regs_const_on
  , `AlgorithmLib.LZ4Simt.potential_on
  , `AlgorithmLib.LZ4Simt.inv_on
  , `Lz4Sites.op_const_to_216
  , `Lz4Sites.lsicFS_closed
  , `Lz4Sites.lsicInv_op_le
  , `Lz4Sites.lsicFS_hstep
  , `Lz4Sites.lsic_op_lt
    -- the same thirteen-instruction LSIC run appears three times; each instance
    -- is proven, closing six of the ten `sbAddr` sites
  , `Lz4Sites.lsicLS_hstep
  , `Lz4Sites.lsicL_op_lt
  , `Lz4Sites.lsicMS_hstep
  , `Lz4Sites.lsicM_op_lt
    -- and the two token stores need no potential at all: nothing writes the
    -- cursor between the region entry and the store
  , `Lz4Sites.bodyPre_closed
  , `Lz4Sites.op_const_to_130
    -- the whole token emit as one region: its potential and the seven stores
  , `Lz4Sites.tokS_closed
  , `Lz4Sites.tokRem_pos
  , `Lz4Sites.tokInv_op_lt
    -- …and the invariant is proven preserved at all sixty-eight program points
    -- of the emit, so the seven stores inside it are confined
  , `Lz4Sites.tokS_hstep
  , `Lz4Sites.tok_op_lt
    -- the loopC body as a region: what the loop-head checkpoint consumes
  , `Lz4Sites.loopBodyS_closed
  , `Lz4Sites.head_not_in_loopBodyS
  , `AlgorithmLib.LZ4Simt.regs_const_from
  , `AlgorithmLib.LZ4Simt.pc_next
  , `AlgorithmLib.LZ4Simt.siter_add
  , `Lz4Sites.prologue_pc_shape
  , `Lz4Sites.prologue_not_at_store_site
    -- the output cursor cannot run backwards: loop head, one iteration and one
    -- emitted byte are all instances of the same step
  , `AlgorithmLib.LZ4WarpDSL.planBlockFrom_encode_le9
  , `AlgorithmLib.LZ4WarpDSL.op_le_of_add
  , `AlgorithmLib.LZ4WarpDSL.op_le_of_emitLoop
  , `AlgorithmLib.LZ4WarpDSL.op_le_finalUifOut
    -- …and inside one token, at the emit points themselves
  , `AlgorithmLib.LZ4WarpDSL.op_le_matchAfterSetp
  , `AlgorithmLib.LZ4WarpDSL.op_le_matchUifOut
    -- and how much of the buffer contract the allocator actually owes
  , `Lz4NonVacuity.layoutOK_of_alloc
  , `Lz4NonVacuity.layoutOK_of_alloc_witness
    -- the payload image and the offsets derived from it cannot disagree
  , `Algorithm.payload_length
  , `Algorithm.payload_fits
    -- the model's block dimension is the one the launch passes
  , `AlgorithmLib.LZ4Simt.initRegs_ntid
    -- every address the kernel uses, bounded: the loads from the kernel's own
    -- clamps and the candidate-select's guards, the stores from the cursor
  , `Lz4Sites.loads_confined
  , `Lz4Sites.tail_copy_budget
  , `Lz4Sites.cpDo_confined2
  , `Lz4Sites.regConfined_shipped
  , `Lz4Sites.kernelConfined_shipped
    -- and the same two at the 64 KiB geometry
  , `Lz4Sites.regConfined_shipped64
  , `Lz4Sites.kernelConfined_shipped64
    -- the composition: `LayoutOK` gives confinement at every memory the run
    -- passes through, a long-enough schedule is CONSTRUCTED rather than
    -- assumed, and the twenty launches follow
  , `Lz4Interleave.exists_nAt
  , `Lz4Interleave.schedComplete_exists
  , `Lz4Interleave.confineHyps_of_layoutOK
  , `Lz4Interleave.oneLaunch_ok
  , `Lz4Interleave.launchesTo_of_layout
  , `Lz4Whole.confined32
  , `Lz4Whole.confined64
  , `Lz4Whole.run_correct
  , `Lz4Whole.shipped32_run_correct
  , `Lz4Whole.shipped64_run_correct
  , `Lz4Whole.run_correct_witness
    -- …and the launch count and grid the run is stated at, read out of the
    -- emitted CLIF rather than taken from the generator's constants
  , `Lz4Whole.emittedLaunches32
  , `Lz4Whole.emittedLaunches64
  , `Lz4Whole.emittedGrid32
  , `Lz4Whole.emittedGrid64
  , `Lz4Whole.shipped32_run_at_emitted
  , `Lz4Whole.shipped64_run_at_emitted
    -- the ledger itself: this fails to elaborate if a named theorem moves
  , `Lz4Assumptions.anchors
  ]

end Lz4Scan

-- The scan's cost grows with the development it scans: 130 claims, each a
-- transitive-closure walk plus an `isProp` per binder.  This is accounting, not
-- proof, so the default 200k elaboration budget has nothing to say about it —
-- but the growth is linear in the claim count, so the limit is kept close enough
-- to the measured cost to still be a canary if that ever stops being true.
set_option maxHeartbeats 600000 in
open Lz4Scan TrustScan in
#eval runScanWith lz4Surface "lz4-compressor" roots
