module
public import Vit.Units
meta import Vit.Units
public import AlgorithmLib.ML.Model.UnitWords
meta import AlgorithmLib.ML.Model.UnitWords
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
  # Every elementwise launch runs its statement

  The ViT's tape is laid out as `vLayout`: batch `SQ`, the layout's buffer
  sizes, a group over `G` blocks printed in blocks of `vWarpsOf G` warps. Each
  group that prints a kernel prints its statement's body kernel (`vTextOf`,
  `Body.kernel`), and `vit_unit_ok` is what the step lemmas read of it
  (`TapeLayout.UnitOk`). `vit_body` is what launching it does: every word a
  lane of the launch touches is left as the warp machine's run of the
  statement for that lane's block leaves it, from the launch's memory.

  It rests on two decided facts. `launch_static` asks `staticOk` of each
  launch, which reads only the statement's indexes, so the launches are
  decided once per distinct key (`vKeys`), every launch's key checked to be
  among them (`vit_static`). And each kernel is emitted whole, its members
  lowering to stages over its grid, none a vendor's product (`vit_fine`),
  which is about the full statement and is decided per launch.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec AlgorithmLib.Kernel AlgorithmLib.ML AlgorithmLib.ML.BodyProof
open AlgorithmLib.ML.BodyWarp AlgorithmLib.ML.BodyCheck AlgorithmLib.ML.GroupLink

namespace Vit

/-- **The ViT's tape, laid out**: batch `SQ`, the layout's buffer sizes, a
    group over `G` blocks in blocks of `vWarpsOf G` warps. -/
def vLayout : TapeLayout := ⟨SQ, vBufBytes, vWarpsOf⟩

/-- The launches' distinct keys, statements erased. -/
def vKeys : List Key := Id.run do
  let mut seen : Std.HashSet Key := {}
  let mut out : Array Key := #[]
  for u in vProvenUnits do
    let k := vLayout.key (vUnitOps u)
    let k := (k.1, k.2.1, k.2.2.1, erase k.2.2.2)
    if seen.contains k then continue
    seen := seen.insert k
    out := out.push k
  return out.toList

/-- **Every launch passes `staticOk`**, decided once per distinct key. -/
theorem vit_static : keysOk vKeys (vProvenUnits.map fun u => vLayout.key (vUnitOps u)) = true := by native_decide

/-- **Every group's kernel is emitted whole**, over a grid of at most `2^32`
    blocks its warps per block divide, its members lowering to stages over
    the group's grid and none a vendor's product. -/
theorem vit_fine : vProvenUnits.all (fun u => vLayout.restB (vUnitOps u)) = true := by native_decide

/-- **Every group's launch is one the step lemmas read.** -/
theorem vit_unit_ok (u : List Nat) (hu : u ∈ vProvenUnits) : vLayout.UnitOk (vUnitOps u) :=
  vLayout.unitOk_of_rest (keysOk_mem vit_static _ (List.mem_map.mpr ⟨u, hu, rfl⟩)) (List.all_eq_true.mp vit_fine u hu)

/-- **Launching a group's kernel runs its statement**: the kernel of
    `vWarpsOf` warps a block, over the group's `vUnitGrid / vWarpsOf` blocks and
    buffers of at least the sizes the tape allocates, leaves every word a lane
    of block `cta < vUnitGrid` touches as the warp machine's run of the group's
    statement for that block leaves it, from the launch's memory. -/
theorem vit_body (u : List Nat) (hu : u ∈ vProvenUnits) (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    (bufs bufs' : Nat → ByteArray)
    (hsz : ∀ b, b < (vUnitBufs u).length → (vLayout.sizes (vUnitOps u)).getD b 0 ≤ (bufs b).size)
    (hr : vLayout.LaunchRuns (vUnitOps u) hw bufs bufs') :
    ∀ cta (l : Lane) b j, cta < vUnitGrid u → (b, j) ∈ touchL cta l 0 (vUnitStmt u) →
      ({ kmem bufs with bufs := bufs' } : Sem.Mem).load (bufW (vUnitBufs u).length b j) 4 =
        some ((@EWStmt.run F32 (F32.ops hw) _ (vUnitStmt u) (ws0 (vUnitBufs u).length (kmem bufs)) cta 0).mem b j).bits :=
  TapeLayout.unit_body (vit_unit_ok u hu) hw hw32 bufs bufs' hsz hr

/-- **Launching a group's kernel performs its members' denotations.**  Each
    word a lane of block `cta` touches, in the slot standing for the model's
    buffer `g`, the launch leaves as the group's operations' denotations
    (`TOp.den`), folded in order at the words the hardware computes on (`F32`),
    leave `g`'s word, from the memory the launch's slots hold. -/
theorem vit_group (u : List Nat) (hu : u ∈ vProvenUnits) (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    (bufs bufs' : Nat → ByteArray)
    (hsz : ∀ b, b < (vUnitBufs u).length → (vLayout.sizes (vUnitOps u)).getD b 0 ≤ (bufs b).size)
    (hr : vLayout.LaunchRuns (vUnitOps u) hw bufs bufs') :
    ∀ cta (l : Lane) g j, cta < vUnitGrid u → (tabInj (vUnitBufs u) g, j) ∈ touchL cta l 0 (vUnitStmt u) →
      ({ kmem bufs with bufs := bufs' } : Sem.Mem).load (bufW (vUnitBufs u).length (tabInj (vUnitBufs u) g) j) 4 =
        some ((vUnitOps u).foldl (fun m op => @TOp.den F32 (F32.ops hw) _ op m)
          (fun g' a => (ws0 (vUnitBufs u).length (kmem bufs)).mem (tabInj (vUnitBufs u) g') a) g j).bits :=
  TapeLayout.unit_group (vit_unit_ok u hu) hw hw32 bufs bufs' hsz hr

/-- **A group's launch leaves every word of its slots as its members'
    denotations do**: each in-bounds word of the slot standing for the model's
    buffer `g` — touched by the launch or not — is left as the group's
    operations' denotations, folded in order at `F32`, leave `g`'s word, from
    the memory the slots hold. -/
theorem vit_launch (u : List Nat) (hu : u ∈ vProvenUnits) (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    (bufs bufs' : Nat → ByteArray)
    (hsz : ∀ b, b < (vUnitBufs u).length → (vLayout.sizes (vUnitOps u)).getD b 0 ≤ (bufs b).size)
    (hr : vLayout.LaunchRuns (vUnitOps u) hw bufs bufs') :
    ∀ g j, g ∈ vUnitBufs u → j < 2 ^ 32 → 4 * j + 4 ≤ (bufs (tabInj (vUnitBufs u) g)).size →
      ({ kmem bufs with bufs := bufs' } : Sem.Mem).load (bufW (vUnitBufs u).length (tabInj (vUnitBufs u) g) j) 4 =
        some ((vUnitOps u).foldl (fun m op => @TOp.den F32 (F32.ops hw) _ op m)
          (fun g' a => (ws0 (vUnitBufs u).length (kmem bufs)).mem (tabInj (vUnitBufs u) g') a) g j).bits :=
  TapeLayout.unit_launch (vit_unit_ok u hu) hw hw32 bufs bufs' hsz hr

/-- **A group's launch leaves the slots it only reads as they were**: each
    in-bounds word of a slot standing for a buffer no member outputs. So the
    kernel writes only the slots `vKSpec` flags as written. -/
theorem vit_launch_keeps (u : List Nat) (hu : u ∈ vProvenUnits) (hw : Hw)
    (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (bufs bufs' : Nat → ByteArray)
    (hsz : ∀ b, b < (vUnitBufs u).length → (vLayout.sizes (vUnitOps u)).getD b 0 ≤ (bufs b).size)
    (hr : vLayout.LaunchRuns (vUnitOps u) hw bufs bufs') :
    ∀ g j, g ∈ vUnitBufs u → g ∉ (vUnitOps u).map (fun op => (TOp.outSize SQ op).1) → j < 2 ^ 32 →
      4 * j + 4 ≤ (bufs (tabInj (vUnitBufs u) g)).size →
      ({ kmem bufs with bufs := bufs' } : Sem.Mem).load (bufW (vUnitBufs u).length (tabInj (vUnitBufs u) g) j) 4 =
        (kmem bufs).load (bufW (vUnitBufs u).length (tabInj (vUnitBufs u) g) j) 4 :=
  TapeLayout.unit_launch_keeps (vit_unit_ok u hu) hw hw32 bufs bufs' hsz hr

end Vit
