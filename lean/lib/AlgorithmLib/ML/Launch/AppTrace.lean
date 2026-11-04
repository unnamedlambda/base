module
public import AlgorithmLib.ML.Launch.HostBridge
meta import AlgorithmLib.ML.Launch.HostBridge
public import AlgorithmLib.Check.Trace
meta import AlgorithmLib.Check.Trace
public import AlgorithmLib.Check.DrvState
meta import AlgorithmLib.Check.DrvState
import all Init.Data.Repr
@[expose] public section

/-!
# The calls a run makes

A program in the driver's standard layout (`Drv.App`) launches kernel `j`
by calling `cuLaunchKernel` with `j`'s function handle, its grid and block,
and its parameter array, observed reading the pointers to the allocations
it is handed (`Drv.App.launchRec`). The call patterns of launching kernels
in order then waiting (`Drv.App.runCalls`), of taking steps that are
launches and strided products (`Drv.App.callsWith`), and of replaying a
graph (`Drv.App.replayCalls`) are what the checker shows a body's runs make
(`check_calls`).
-/

namespace AlgorithmLib.ML

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Check

/-- The call launching kernel `j` of a program in the standard layout on
    stream `s`, the default stream unless given: its function handle, grid,
    block, no shared memory, the stream, its parameter array, and no extra
    arguments, observed reading the pointers to the allocations it is
    handed. -/
def _root_.AlgorithmLib.Drv.App.launchRec (a : Drv.App) (j : Nat) (s : UInt64 := 0) : CallRec :=
  match a.kernels[j]? with
  | some k =>
      let (gx, gy, gz) := k.grid
      let (bx, by_, bz) := k.block
      { callee := .ext (.cuda .launchKernel),
        args := [.is (drvHandle kFunc j), .is (UInt64.ofNat gx), .is (UInt64.ofNat gy), .is (UInt64.ofNat gz),
                 .is (UInt64.ofNat bx), .is (UInt64.ofNat by_), .is (UInt64.ofNat bz), .is 0, .is s,
                 .is (Drv.App.at_ (a.paramsOff j)), .is 0] ++ k.args.map fun i => .is (devPtr i) }
  | none => { callee := .ext (.cuda .launchKernel), args := [] }

/-- **Launching kernels `js` in order, then waiting**: the calls a body that
    does that makes. -/
def _root_.AlgorithmLib.Drv.App.runCalls (a : Drv.App) (js : List Nat) : Tr :=
  .lit (js.map a.launchRec ++ [⟨.ext (.cuda .ctxSynchronize), []⟩])

/-- The call `cublasSgemmStridedBatched` at a batch of one makes for `g` on
    handle `h`, observed reading the scalars `sc`. -/
def _root_.AlgorithmLib.Drv.App.stridedRec (a : Drv.App) (h : UInt64) (sc : List UInt64) (g : Gemm) : CallRec :=
  { callee := .ext (.cublas .sgemmStridedBatched), args := (a.stridedWords h g ++ sc).map .is }

/-- The call a step makes, a product's as `rec` says. -/
def _root_.AlgorithmLib.Drv.App.stepRecWith (a : Drv.App) (rec : Gemm → CallRec) : Sum Nat Gemm → CallRec
  | .inl j => a.launchRec j
  | .inr g => rec g

/-- **Taking `ss` in order, then waiting**, a product's call as `rec` says:
    the calls a body that does that makes. -/
def _root_.AlgorithmLib.Drv.App.callsWith (a : Drv.App) (rec : Gemm → CallRec) (ss : List (Sum Nat Gemm)) : Tr :=
  .lit (ss.map (a.stepRecWith rec) ++ [⟨.ext (.cuda .ctxSynchronize), []⟩])

theorem handleOf_drvHandle {kind j : Nat} (hk : kind < 2 ^ 8) (hj : j < 2 ^ 48) :
    handleOf kind (drvHandle kind j) = some j := by
  unfold handleOf drvHandle
  have : (UInt64.ofNat (2 ^ 56 + kind * 2 ^ 48 + j + 1)).toNat = 2 ^ 56 + kind * 2 ^ 48 + j + 1 := by
    simp; omega
  simp only [this]
  split
  · congr 1; omega
  · rename_i h; simp at h; omega

theorem devPtrOf_devPtr {i : Nat} (hi : i < 2 ^ 16) : devPtrOf (devPtr i) = some (i, 0) := by
  unfold devPtrOf devPtr
  have : (UInt64.ofNat ((6 + i) * 2 ^ 36)).toNat = (6 + i) * 2 ^ 36 := by simp; omega
  simp only [this]
  have h1 : 6 * 2 ^ 36 ≤ (6 + i) * 2 ^ 36 := Nat.mul_le_mul_right _ (by omega)
  have h2 : (6 + i) * 2 ^ 36 < (6 + 268435449) * 2 ^ 36 := Nat.mul_lt_mul_of_pos_right (by omega) (by decide)
  simp only [h1, h2, decide_true, Bool.and_self, if_true, Option.some.injEq, Prod.mk.injEq]
  exact ⟨by rw [Nat.mul_div_cancel _ (by decide)]; omega, Nat.mul_mod_left _ _⟩

theorem argsOk_drop : ∀ {xs ys : List ArgPat} {vs : List V}, argsOk (xs ++ ys) vs →
    argsOk ys (vs.drop xs.length)
  | [], _, _, h => h
  | _ :: xs, ys, _ :: vs, ⟨_, h⟩ => by
      show argsOk ys (vs.drop xs.length)
      exact argsOk_drop h
  | _ :: _, _, [], h => h.elim

/-- Values matched word for word are those words. -/
theorem words_exact : ∀ {ws : List UInt64} {vs : List V}, argsOk (ws.map ArgPat.is) vs →
    vs.map (fun x => (asBits x).getD 0) = ws
  | [], [], _ => rfl
  | [], _ :: _, h => h.elim
  | _ :: _, [], h => h.elim
  | w :: ws, v :: vs, ⟨hv, h⟩ => by
      have hv' : asBits v = some w := hv
      rw [List.map_cons, hv', Option.getD_some, words_exact h]

/-- **Replaying executable graph `e` on stream `s` `k` times, then waiting for
    the stream**: the calls a body that does that makes. -/
def _root_.AlgorithmLib.Drv.App.replayCalls (e s : UInt64) (k : Nat) : Tr :=
  .lit (List.replicate k ⟨.ext (.cuda .graphLaunch), [.is e, .is s]⟩ ++
    [⟨.ext (.cuda .streamSynchronize), [.is s]⟩])

end AlgorithmLib.ML
