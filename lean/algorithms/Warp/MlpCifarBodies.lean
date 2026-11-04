import AlgorithmLib.ML.Kernel.BodyCheck
import AlgorithmLib.ML.Model.TapeRun
import Warp.MlpCifar
/-!
  # The tapes' kernels, launched

  Apart from the generator, which prints the kernels and so imports no proof:
  each operation of a tape ships its local statement's body kernel, and
  launching it over the operation's grid and buffers of the sizes the program
  allocates runs the statement (`tape_runs`), for the CIFAR network
  (`cifar_runs`), the Qwen block's training step and fused forward
  (`qwen_runs`) and the expert dispatch (`moe_runs`).
-/

open Lean AlgorithmLib AlgorithmLib.IR AlgorithmLib.ML

namespace MlpCifar

/-- An operation's buffer sizes in bytes, in its own table's order, from the
    model's sizes `bytes`. -/
def opSz (n : Nat) (bytes : List Nat) (op : TOp) (b : Nat) : Nat := bytes.getD ((op.bufs n).getD b 0) 0

/-- Every launch of a tape — an operation and its grid — passes `launchOk`. -/
def tapeOk (n : Nat) (bytes : List Nat) (t : List (TOp × Nat)) : Bool :=
  t.all fun p => BodyCheck.launchOk (p.1.bufs n).length p.2 1 (opSz n bytes p.1) (p.1.localStmt n)

open AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec
  AlgorithmLib.Kernel AlgorithmLib.ML.BodyProof AlgorithmLib.ML.BodyWarp AlgorithmLib.ML.BodyCheck in
/-- **A tape passing `tapeOk` runs**: each operation's kernel, over its grid
    of one-warp blocks and buffers of at least its sizes, leaves every word a
    lane touches as the warp machine's run of its local statement for that
    block leaves it. -/
theorem tape_runs (n : Nat) (bytes : List Nat) (t : List (TOp × Nat)) (ht : tapeOk n bytes t = true)
    (p : TOp × Nat) (hp : p ∈ t) (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    (bufs bufs' : Nat → ByteArray) (hsz : ∀ b, b < (p.1.bufs n).length → opSz n bytes p.1 b ≤ (bufs b).size)
    (hr : LaunchRuns (Body.kernel "main" (p.1.bufs n).length 0 (p.1.localStmt n)) p.2 32 (bufArgs (p.1.bufs n).length) bufs bufs' hw) :
    ∀ cta (l : Lane) b j, cta < p.2 → (b, j) ∈ touchL cta l 0 (p.1.localStmt n) →
      ({ kmem bufs with bufs := bufs' } : Sem.Mem).load (bufW (p.1.bufs n).length b j) 4 =
        some ((@WStmt.run F32 (F32.ops hw) _ ((p.1.localStmt n).elabAt cta 0)
          (ws0 (p.1.bufs n).length (kmem bufs))).mem b j).bits := by
  have h := launch_body hw hw32 "main" _ 0 _ 1 p.2 _ (List.all_eq_true.mp ht p hp) bufs bufs' hsz hr
  intro cta l b j hc hm
  exact h cta l b j (by omega) hm

/-- **The CIFAR network's tape, laid out**: batch `B`, the program's buffer
    sizes, every kernel one operation in blocks of one warp. -/
def cifarLayout : TapeLayout := ⟨B, bufBytes, fun _ => 1⟩

/-- Slot `i`'s group: the tape's `i`-th operation. -/
def cifarUnit (i : Nat) : List TOp := (cifarTape.drop i).take 1

/-- The CIFAR network's launches: each operation over its slot's grid. -/
def cifarLaunches : List (TOp × Nat) := cifarTape.zip slotGrid

/-- **The Qwen block's tape, laid out**: batch one, the program's buffer
    sizes, every kernel one operation in blocks of one warp. -/
def qwenLayout : TapeLayout := ⟨1, qBufBytes, fun _ => 1⟩

/-- Slot `i`'s group: the `i`-th operation of the training step, then of the
    fused forward. -/
def qwenUnit (i : Nat) : List TOp := ((qwenTape ++ qwenFwdFused).drop i).take 1

/-- The Qwen image's launches: the training step, then the fused forward. -/
def qwenLaunches : List (TOp × Nat) := (qwenTape ++ qwenFwdFused).map fun o => (o, TOp.gridOf o)

/-- **The dispatch's tape, laid out**: batch one, the program's allocation
    sizes, every kernel one operation in blocks of one warp. -/
def moeLayout : TapeLayout := ⟨1, mAllocBytes, fun _ => 1⟩

/-- Slot `i`'s group: the `i`-th operation of the dispatch tape. -/
def moeUnit (i : Nat) : List TOp := (moeTape.drop i).take 1

/-- The dispatch's launches. -/
def moeLaunches : List (TOp × Nat) := moeTape.map fun o => (o, TOp.gridOf o)

/-- The dispatch's buffer sizes as its kernels read them: a slot buffer is
    bound, by `bindExperts`, to an expert's matrix of `MEXPERT` bytes. -/
def moeBytes : List Nat :=
  (List.range mBufBytes.length).map fun b => if MSLOT0 ≤ b ∧ b < MBASE then MEXPERT else mBufBytes.getD b 0

open AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec
  AlgorithmLib.Kernel AlgorithmLib.ML.BodyProof AlgorithmLib.ML.BodyWarp AlgorithmLib.ML.BodyCheck in
/-- **Every CIFAR kernel's launch runs its statement**: each launch's
    operation, over its grid of one-warp blocks and buffers of at least
    the sizes `app` allocates, leaves every word a lane touches as the warp machine's run of
    the operation's local statement for its block leaves it. -/
theorem cifar_runs (p : TOp × Nat) (hp : p ∈ cifarLaunches) (hw : Hw)
    (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (bufs bufs' : Nat → ByteArray)
    (hsz : ∀ b, b < (p.1.bufs B).length → opSz B bufBytes p.1 b ≤ (bufs b).size)
    (hr : LaunchRuns (Body.kernel "main" (p.1.bufs B).length 0 (p.1.localStmt B)) p.2 32 (bufArgs (p.1.bufs B).length) bufs bufs' hw) :
    ∀ cta (l : Lane) b j, cta < p.2 → (b, j) ∈ touchL cta l 0 (p.1.localStmt B) →
      ({ kmem bufs with bufs := bufs' } : Sem.Mem).load (bufW (p.1.bufs B).length b j) 4 =
        some ((@WStmt.run F32 (F32.ops hw) _ ((p.1.localStmt B).elabAt cta 0)
          (ws0 (p.1.bufs B).length (kmem bufs))).mem b j).bits :=
  tape_runs B bufBytes cifarLaunches (by native_decide) p hp hw hw32 bufs bufs' hsz hr

open AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec
  AlgorithmLib.Kernel AlgorithmLib.ML.BodyProof AlgorithmLib.ML.BodyWarp AlgorithmLib.ML.BodyCheck in
/-- **Every Qwen kernel's launch runs its statement**: each launch's
    operation, over its grid of one-warp blocks and buffers of at least
    the sizes `qApp` allocates, leaves every word a lane touches as the warp machine's run of
    the operation's local statement for its block leaves it. -/
theorem qwen_runs (p : TOp × Nat) (hp : p ∈ qwenLaunches) (hw : Hw)
    (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (bufs bufs' : Nat → ByteArray)
    (hsz : ∀ b, b < (p.1.bufs 1).length → opSz 1 qBufBytes p.1 b ≤ (bufs b).size)
    (hr : LaunchRuns (Body.kernel "main" (p.1.bufs 1).length 0 (p.1.localStmt 1)) p.2 32 (bufArgs (p.1.bufs 1).length) bufs bufs' hw) :
    ∀ cta (l : Lane) b j, cta < p.2 → (b, j) ∈ touchL cta l 0 (p.1.localStmt 1) →
      ({ kmem bufs with bufs := bufs' } : Sem.Mem).load (bufW (p.1.bufs 1).length b j) 4 =
        some ((@WStmt.run F32 (F32.ops hw) _ ((p.1.localStmt 1).elabAt cta 0)
          (ws0 (p.1.bufs 1).length (kmem bufs))).mem b j).bits :=
  tape_runs 1 qBufBytes qwenLaunches (by native_decide) p hp hw hw32 bufs bufs' hsz hr

open AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec
  AlgorithmLib.Kernel AlgorithmLib.ML.BodyProof AlgorithmLib.ML.BodyWarp AlgorithmLib.ML.BodyCheck in
/-- **Every dispatch kernel's launch runs its statement**: each launch's
    operation, over its grid of one-warp blocks and buffers of at least
    the sizes the dispatch binds (`moeBytes`), leaves every word a lane touches as the warp machine's run of
    the operation's local statement for its block leaves it. -/
theorem moe_runs (p : TOp × Nat) (hp : p ∈ moeLaunches) (hw : Hw)
    (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (bufs bufs' : Nat → ByteArray)
    (hsz : ∀ b, b < (p.1.bufs 1).length → opSz 1 moeBytes p.1 b ≤ (bufs b).size)
    (hr : LaunchRuns (Body.kernel "main" (p.1.bufs 1).length 0 (p.1.localStmt 1)) p.2 32 (bufArgs (p.1.bufs 1).length) bufs bufs' hw) :
    ∀ cta (l : Lane) b j, cta < p.2 → (b, j) ∈ touchL cta l 0 (p.1.localStmt 1) →
      ({ kmem bufs with bufs := bufs' } : Sem.Mem).load (bufW (p.1.bufs 1).length b j) 4 =
        some ((@WStmt.run F32 (F32.ops hw) _ ((p.1.localStmt 1).elabAt cta 0)
          (ws0 (p.1.bufs 1).length (kmem bufs))).mem b j).bits :=
  tape_runs 1 moeBytes moeLaunches (by native_decide) p hp hw hw32 bufs bufs' hsz hr

end MlpCifar
