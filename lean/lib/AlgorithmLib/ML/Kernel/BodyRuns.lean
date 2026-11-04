module
public import AlgorithmLib.ML.Kernel.BodyCheck
meta import AlgorithmLib.ML.Kernel.BodyCheck
public import AlgorithmLib.ML.Ptx.BodyEnds
meta import AlgorithmLib.ML.Ptx.BodyEnds
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `ML.Kernel.BodyRuns` — a launch has a run, its obligations decided

A launch whose statement, in phases, passes `staticOk` and whose code is fine
has a run over buffers of at least the sizes it was checked at
(`launch_ends`): `BodyEnds.launch_exists`, its obligations what `staticOk`
decides (`staticOk_obl`).
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec AlgorithmLib.Kernel AlgorithmLib.ML AlgorithmLib.ML.BodyProof
open AlgorithmLib.ML.BodyWarp AlgorithmLib.ML.BodyPhase AlgorithmLib.ML.BodyCheck

namespace AlgorithmLib.ML.BodyRuns

/-- **A static statement's launch has a run, its obligations decided**:
    `grid` blocks of `nw` warps of the statement's kernel, in phases `ps` of
    it, passing `staticOk` for `grid·nw` blocks and buffers of `sz` bytes, its
    code fine, over buffers of at least those sizes, have a run (`LaunchRuns`):
    every thread answered as the laws give back, and ending. -/
theorem launch_ends (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (name : String)
    (nbuf smem : Nat) (s : EWStmt) (ps : List EWStmt) (hsp : Split s ps) (nw : Nat) (hnw : 0 < nw)
    (hnw31 : nw < 2 ^ 31) (grid : Nat) (hgrid : grid * nw ≤ 2 ^ 32) (hN : 32 * nw ≤ 2 ^ 32) (sz : Nat → Nat)
    (hok : staticOk nbuf (grid * nw) sz ps s = true)
    (hfine : Fine (emitGo (fnBody (params nbuf) (Body.body nbuf s nw)) ⟨(params nbuf).length, 0, [], [], [], none⟩).2)
    (bufs : Nat → ByteArray) (hsz : ∀ b, b < nbuf → sz b ≤ (bufs b).size) :
    ∃ bufs', LaunchRuns (Body.kernel name nbuf smem s nw) grid (32 * nw) (bufArgs nbuf) bufs bufs' hw := by
  have ob := staticOk_obl nbuf (grid * nw) sz s ps hsp hok
  exact BodyEnds.launch_exists hw hw32 name nbuf smem s ps hsp nw hnw hnw31 grid hgrid hN (staticAcc (grid * nw) ps)
    (fun _ _ => none) ob.acc ob.bd ob.own ob.rdr ob.disj hfine bufs (fun _ _ _ h => by cases h)
    (fun _ _ _ _ _ _ _ => rfl) ob.fits
    (fun q c l b j hc hq h => Nat.le_trans (ob.sz q c l b j hq h) (hsz b (ob.bd q c l b j h).1))

end AlgorithmLib.ML.BodyRuns
