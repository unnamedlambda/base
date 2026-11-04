module
public import AlgorithmLib.ML.Kernel.BodyRuns
meta import AlgorithmLib.ML.Kernel.BodyRuns
public import AlgorithmLib.ML.Model.GroupLink
meta import AlgorithmLib.ML.Model.GroupLink
public import AlgorithmLib.ML.Model.VendorHost
meta import AlgorithmLib.ML.Model.VendorHost
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# A tape's groups, launched, keep device memory in step with it

A tape is laid out on the device (`TapeLayout`): its operations at a batch,
each buffer at a size in bytes, and each group of operations printed as one
kernel (`unitStmt`, over the group's own table `unitBufs`) in blocks of a
number of warps its grid decides. What a group must be for its launch to be
read is decided once (`UnitOk`): the launch obligations `staticOk` decides,
the kernel emitted whole, the members lowering to stages over one grid, none
of them a vendor's product.

Such a group's kernel has a run over buffers of at least the layout's sizes
(`unit_runs`), and launching it leaves every word of its slots as its
members' denotations, folded in order, leave it (`unit_launch`). On a device
honest about the launch (`machine`), a launch keeps device memory in step with
the tape (`unit_step_kernel`), a vendor's product likewise (`gemm_step`), and
so steps taken in order keep it in step with their groups' operations in
order (`steps_agree`, `steps_tape`), each word inside each buffer at the size
the layout makes it.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
  AlgorithmLib.HProg.Hoare AlgorithmLib.HProg.Contracts AlgorithmLib.HProg.DevSpec AlgorithmLib.LibSpec
  AlgorithmLib.Device AlgorithmLib.Kernel AlgorithmLib.ML.BodyProof AlgorithmLib.ML.BodyWarp
  AlgorithmLib.ML.BodyCheck AlgorithmLib.ML.GroupLink

namespace AlgorithmLib.ML

/-- **A tape laid out on the device**: its operations at batch `batch`, buffer
    `g` taking `bytes[g]` bytes, a group over `G` blocks printed in blocks of
    `warps G` warps. -/
structure TapeLayout where
  batch : Nat
  bytes : List Nat
  warps : Nat → Nat

namespace TapeLayout

variable (L : TapeLayout)

/-- A group's warps per block. -/
def W (ops : List TOp) : Nat := L.warps (unitGrid ops)

/-- A group's buffer sizes in bytes, in its table's order. -/
def sizes (ops : List TOp) : List Nat := (unitBufs L.batch ops).map fun r => L.bytes.getD r 0

/-- A group's launch, as `staticOk` reads it: buffers, blocks, sizes,
    statement. -/
def key (ops : List TOp) : Key := ((unitBufs L.batch ops).length, unitGrid ops, L.sizes ops, unitStmt L.batch ops)

/-- A group's kernel emitted whole. -/
def fineB (ops : List TOp) : Bool :=
  let n := (unitBufs L.batch ops).length
  let t := (emitGo (fnBody (params n) (Body.body n (unitStmt L.batch ops) (L.W ops)))
    ⟨(params n).length, 0, [], [], [], none⟩).2
  decide (t.err = none) && decide (needS t.out ≤ HProg.fuel)

/-- **What a group must be for its launch to be read**: its launch passes
    `staticOk`, its kernel is emitted whole over at most `2^32` blocks, its
    members lower to stages over the group's grid, none is a vendor's
    product, its table is one a slot can name, and its warps per block divide
    its grid. -/
structure UnitOk (ops : List TOp) : Prop where
  static : keyOk (L.key ops) = true
  fine : L.fineB ops = true
  grid32 : unitGrid ops ≤ 2 ^ 32
  lowers : (forget (ops.map TOp.node)).all Node.lowersB = true
  grids : ∀ op ∈ ops, op.gridOf = unitGrid ops
  nbuf : (unitBufs L.batch ops).length < 268435449
  own : ∀ op ∈ ops, op.vendorGemm = none
  warps : 0 < L.W ops ∧ L.W ops < 2 ^ 27 ∧ unitGrid ops % L.W ops = 0

/-- `UnitOk` with its launch obligations taken as decided elsewhere. -/
def restB (ops : List TOp) : Bool :=
  L.fineB ops && decide (unitGrid ops ≤ 2 ^ 32) && (forget (ops.map TOp.node)).all Node.lowersB &&
    ops.all (fun op => op.gridOf == unitGrid ops) && decide ((unitBufs L.batch ops).length < 268435449) &&
    ops.all (fun op => op.vendorGemm.isNone) &&
    decide (0 < L.W ops) && decide (L.W ops < 2 ^ 27) && decide (unitGrid ops % L.W ops = 0)

/-- `UnitOk`, decided. -/
def unitOkB (ops : List TOp) : Bool := keyOk (L.key ops) && L.restB ops

theorem unitOk_of_rest {ops : List TOp} (hk : keyOk (L.key ops) = true) (h : L.restB ops = true) :
    L.UnitOk ops := by
  simp only [restB, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, beq_iff_eq,
    Option.isNone_iff_eq_none] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨hf, hg⟩, hl⟩, hgr⟩, hn⟩, ho⟩, hw0⟩, hw1⟩, hw2⟩ := h
  exact ⟨hk, hf, hg, List.all_eq_true.mpr hl, hgr, hn, ho, hw0, hw1, hw2⟩

theorem unitOk_of {ops : List TOp} (h : L.unitOkB ops = true) : L.UnitOk ops := by
  simp only [unitOkB, Bool.and_eq_true] at h
  exact L.unitOk_of_rest h.1 h.2

/-- A group's kernel. -/
def kernel (ops : List TOp) :=
  Body.kernel "main" (unitBufs L.batch ops).length 0 (unitStmt L.batch ops) (L.W ops)

/-- **Group `ops`'s kernel launched over memory `bufs`, slot by slot,
    leaving `bufs'`.** -/
def LaunchRuns (ops : List TOp) (hw : Hw) (bufs bufs' : Nat → ByteArray) : Prop :=
  Kernel.LaunchRuns (L.kernel ops) (unitGrid ops / L.W ops) (32 * L.W ops) (bufArgs (unitBufs L.batch ops).length)
    bufs bufs' hw

/-- **Group `ops`'s kernel runs** over memory `bufs`: a run of its launch
    exists. -/
def Runs (ops : List TOp) (hw : Hw) (bufs : Nat → ByteArray) : Prop := ∃ bufs', L.LaunchRuns ops hw bufs bufs'

/-- A launch's bound buffers, slot by slot. -/
def slots (ins : List ByteArray) : Nat → ByteArray := fun i => ins.getD i .empty

/-- **What the device must be honest about for group `ops`'s launch**: where
    the group's kernel runs over the buffers it was handed, each word of each
    bound buffer is what one such run leaves there. -/
def machine (ops : List TOp) (hw : Hw) (ins outs : List ByteArray) : Prop :=
  L.Runs ops hw (slots ins) →
  ∃ bufs', L.LaunchRuns ops hw (slots ins) bufs' ∧ ∀ i j, i < (unitBufs L.batch ops).length → j < 2 ^ 32 →
    4 * j + 4 ≤ (ins.getD i .empty).size → loadWord (outs.getD i .empty) j = loadWord (bufs' i) j

/-- **Device memory agrees with a memory of the tape** at every word inside
    each buffer, at the size the layout makes it. -/
def Agree (mem : DevMem) (M : Buf → Nat → F32) : Prop :=
  ∀ g j, 4 * j + 4 ≤ L.bytes.getD g 0 → wordView mem g j = M g j

variable {L}

theorem UnitOk.blocks {ops : List TOp} (hU : L.UnitOk ops) :
    unitGrid ops / L.W ops * L.W ops = unitGrid ops :=
  Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hU.warps.2.2)

theorem UnitOk.fine' {ops : List TOp} (hU : L.UnitOk ops) :
    Fine (emitGo (fnBody (params (unitBufs L.batch ops).length)
      (Body.body (unitBufs L.batch ops).length (unitStmt L.batch ops) (L.W ops)))
      ⟨(params (unitBufs L.batch ops).length).length, 0, [], [], [], none⟩).2 := by
  have h := hU.fine
  simp only [fineB, Bool.and_eq_true, decide_eq_true_eq] at h
  exact h

/-- **A group's kernel runs** over buffers of at least the layout's sizes
    (`BodyRuns.launch_ends`, its obligations the ones `UnitOk` decides). -/
theorem unit_runs {ops : List TOp} (hU : L.UnitOk ops) (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    (bufs : Nat → ByteArray) (hsz : ∀ b, b < (unitBufs L.batch ops).length → (L.sizes ops).getD b 0 ≤ (bufs b).size) :
    L.Runs ops hw bufs := by
  have hW := hU.warps
  exact BodyRuns.launch_ends hw hw32 "main" (unitBufs L.batch ops).length 0 (unitStmt L.batch ops)
    (spine (unitStmt L.batch ops)) (split_spine _) (L.W ops) hW.1 (by omega) (unitGrid ops / L.W ops)
    (by rw [hU.blocks]; exact hU.grid32) (by omega) (fun b => (L.sizes ops).getD b 0)
    (by rw [hU.blocks]; exact hU.static) hU.fine' bufs hsz

/-- **Launching a group's kernel runs its statement**: over buffers of at
    least the layout's sizes, it leaves every word a lane of block
    `cta < unitGrid` touches as the warp machine's run of the group's
    statement for that block leaves it, from the launch's memory. -/
theorem unit_body {ops : List TOp} (hU : L.UnitOk ops) (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    (bufs bufs' : Nat → ByteArray)
    (hsz : ∀ b, b < (unitBufs L.batch ops).length → (L.sizes ops).getD b 0 ≤ (bufs b).size)
    (hr : L.LaunchRuns ops hw bufs bufs') :
    ∀ cta (l : Lane) b j, cta < unitGrid ops → (b, j) ∈ touchL cta l 0 (unitStmt L.batch ops) →
      ({ kmem bufs with bufs := bufs' } : Sem.Mem).load (bufW (unitBufs L.batch ops).length b j) 4 =
        some ((@EWStmt.run F32 (F32.ops hw) _ (unitStmt L.batch ops) (ws0 (unitBufs L.batch ops).length (kmem bufs)) cta 0).mem b j).bits := by
  have hW := hU.warps
  have h := launch_static hw hw32 "main" (unitBufs L.batch ops).length 0 (unitStmt L.batch ops)
    (spine (unitStmt L.batch ops)) (split_spine _) (L.W ops) hW.1 (by omega) (unitGrid ops / L.W ops)
    (by rw [hU.blocks]; exact hU.grid32) (by omega) (fun b => (L.sizes ops).getD b 0)
    (by rw [hU.blocks]; exact hU.static) hU.fine' bufs bufs' hsz hr
  intro cta l b j hc hm
  exact h cta l b j (by rw [hU.blocks]; exact hc) hm

theorem UnitOk.lowered {ops : List TOp} (hU : L.UnitOk ops) (hw : Hw) :
    ∃ xs, @lowerNet F32 (F32.ops hw) _ (litNatF32 hw) L.batch (forget (ops.map TOp.node)) = some xs :=
  Option.isSome_iff_exists.mp (@lowerNet_isSome F32 (F32.ops hw) _ (litNatF32 hw) L.batch _ hU.lowers)

theorem UnitOk.apart {ops : List TOp} (hU : L.UnitOk ops) :
    ∀ c c' (l l' : Lane) b j, c < unitGrid ops → c' < unitGrid ops → c ≠ c' →
      (b, j) ∈ touchL c l 0 (unitStmt L.batch ops) → (b, j) ∈ storeL c' l' 0 (unitStmt L.batch ops) → False :=
  staticOk_apart_all _ _ _ (unitStmt L.batch ops) _ (split_spine _) hU.static

/-- **Launching a group's kernel performs its members' denotations.** Each
    word a lane of block `cta` touches, in the slot standing for the model's
    buffer `g`, the launch leaves as the group's operations' denotations
    (`TOp.den`), folded in order at the words the hardware computes on
    (`F32`), leave `g`'s word, from the memory the launch's slots hold. -/
theorem unit_group {ops : List TOp} (hU : L.UnitOk ops) (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    (bufs bufs' : Nat → ByteArray)
    (hsz : ∀ b, b < (unitBufs L.batch ops).length → (L.sizes ops).getD b 0 ≤ (bufs b).size)
    (hr : L.LaunchRuns ops hw bufs bufs') :
    ∀ cta (l : Lane) g j, cta < unitGrid ops →
      (tabInj (unitBufs L.batch ops) g, j) ∈ touchL cta l 0 (unitStmt L.batch ops) →
      ({ kmem bufs with bufs := bufs' } : Sem.Mem).load (bufW (unitBufs L.batch ops).length
          (tabInj (unitBufs L.batch ops) g) j) 4 =
        some (ops.foldl (fun m op => @TOp.den F32 (F32.ops hw) _ op m)
          (fun g' a => (ws0 (unitBufs L.batch ops).length (kmem bufs)).mem (tabInj (unitBufs L.batch ops) g') a)
            g j).bits := by
  intro cta l g j hc hm
  rw [unit_body hU hw hw32 bufs bufs' hsz hr cta l _ j hc hm]
  obtain ⟨xs, hxs⟩ := hU.lowered hw
  have e := @group_den F32 (F32.ops hw) _ (litNatF32 hw) L.batch (unitGrid ops) ops (unitBufs L.batch ops) xs hxs
    hU.grids (unit_covers L.batch ops) hU.apart (ws0 (unitBufs L.batch ops).length (kmem bufs)) cta hc l g j hm
  exact congrArg (fun x => some (F32.bits x)) e

/-- **A group's launch leaves every word of its slots as its members'
    denotations do**: each in-bounds word of the slot standing for the
    model's buffer `g`, touched by the launch or not, is left as the group's
    operations' denotations, folded in order at `F32`, leave `g`'s word, from
    the memory the slots hold. -/
theorem unit_launch {ops : List TOp} (hU : L.UnitOk ops) (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    (bufs bufs' : Nat → ByteArray)
    (hsz : ∀ b, b < (unitBufs L.batch ops).length → (L.sizes ops).getD b 0 ≤ (bufs b).size)
    (hr : L.LaunchRuns ops hw bufs bufs') :
    ∀ g j, g ∈ unitBufs L.batch ops → j < 2 ^ 32 → 4 * j + 4 ≤ (bufs (tabInj (unitBufs L.batch ops) g)).size →
      ({ kmem bufs with bufs := bufs' } : Sem.Mem).load (bufW (unitBufs L.batch ops).length
          (tabInj (unitBufs L.batch ops) g) j) 4 =
        some (ops.foldl (fun m op => @TOp.den F32 (F32.ops hw) _ op m)
          (fun g' a => (ws0 (unitBufs L.batch ops).length (kmem bufs)).mem (tabInj (unitBufs L.batch ops) g') a)
            g j).bits := by
  intro g j hg hj hin
  by_cases ht : ∃ cta, ∃ l : Lane, cta < unitGrid ops ∧
      (tabInj (unitBufs L.batch ops) g, j) ∈ touchL cta l 0 (unitStmt L.batch ops)
  · obtain ⟨cta, l, hc, hm⟩ := ht
    exact unit_group hU hw hw32 bufs bufs' hsz hr cta l g j hc hm
  have hn : ∀ c, c < unitGrid ops → ∀ l : Lane,
      (tabInj (unitBufs L.batch ops) g, j) ∉ storeL c l 0 (unitStmt L.batch ops) :=
    fun c hc l hs => ht ⟨c, l, hc, storeL_sub c l _ 0 hs⟩
  obtain ⟨xs, hxs⟩ := hU.lowered hw
  have hW := hU.warps
  obtain ⟨hlt, _⟩ := compactMap_get (unitBufs L.batch ops) g hg
  have hb : tabInj (unitBufs L.batch ops) g < (unitBufs L.batch ops).length := by
    rw [tabInj_on _ _ hg]; exact hlt
  have keep := launch_static_keep hw hw32 "main" (unitBufs L.batch ops).length 0 (unitStmt L.batch ops)
    (spine (unitStmt L.batch ops)) (split_spine _) (L.W ops) hW.1 (by omega) (unitGrid ops / L.W ops)
    (by rw [hU.blocks]; exact hU.grid32) (by omega) (fun b => (L.sizes ops).getD b 0)
    (by rw [hU.blocks]; exact hU.static) hU.fine' bufs bufs' hsz hr (tabInj (unitBufs L.batch ops) g) j
    ⟨hb, Nat.lt_trans hb hU.nbuf, hj⟩ (fun c hc l => hn c (by rw [hU.blocks] at hc; exact hc) l)
  rw [keep, @group_keep F32 (F32.ops hw) _ (litNatF32 hw) L.batch (unitGrid ops) ops (unitBufs L.batch ops) xs hxs
    hU.grids (unit_covers L.batch ops) _ g j hn]
  obtain ⟨v, hv⟩ := Option.isSome_iff_exists.mp
    (bufW_ok (nbuf := (unitBufs L.batch ops).length) (m := kmem bufs) ⟨hb, Nat.lt_trans hb hU.nbuf, hj⟩
      (by simpa [kmem] using hin) 0).1
  simp only [ws0, hv, Option.getD_some]

/-- **A group's launch leaves the slots it only reads as they were**: each
    in-bounds word of a slot standing for a buffer no member outputs. -/
theorem unit_launch_keeps {ops : List TOp} (hU : L.UnitOk ops) (hw : Hw)
    (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (bufs bufs' : Nat → ByteArray)
    (hsz : ∀ b, b < (unitBufs L.batch ops).length → (L.sizes ops).getD b 0 ≤ (bufs b).size)
    (hr : L.LaunchRuns ops hw bufs bufs') :
    ∀ g j, g ∈ unitBufs L.batch ops → g ∉ ops.map (fun op => (TOp.outSize L.batch op).1) → j < 2 ^ 32 →
      4 * j + 4 ≤ (bufs (tabInj (unitBufs L.batch ops) g)).size →
      ({ kmem bufs with bufs := bufs' } : Sem.Mem).load (bufW (unitBufs L.batch ops).length
          (tabInj (unitBufs L.batch ops) g) j) 4 =
        (kmem bufs).load (bufW (unitBufs L.batch ops).length (tabInj (unitBufs L.batch ops) g) j) 4 := by
  intro g j hg hout hj hin
  rw [unit_launch hU hw hw32 bufs bufs' hsz hr g j hg hj hin,
    @foldl_den_out F32 (F32.ops hw) _ L.batch g ops _
      (fun op hop e => hout (List.mem_map.mpr ⟨op, hop, e.symm⟩))]
  obtain ⟨hlt, _⟩ := compactMap_get (unitBufs L.batch ops) g hg
  have hb : tabInj (unitBufs L.batch ops) g < (unitBufs L.batch ops).length := by
    rw [tabInj_on _ _ hg]; exact hlt
  obtain ⟨v, hv⟩ := Option.isSome_iff_exists.mp
    (bufW_ok (nbuf := (unitBufs L.batch ops).length) (m := kmem bufs) ⟨hb, Nat.lt_trans hb hU.nbuf, hj⟩
      (by simpa [kmem] using hin) 0).1
  simp only [ws0, hv, Option.getD_some]

theorem slots_tab {T : List Nat} {g : Nat} (hg : g ∈ T) (m : DevMem) :
    slots (T.map m) (tabInj T g) = m g := by
  obtain ⟨hl, he⟩ := compactMap_get T g hg
  simp only [slots, tabInj_on T g hg, List.getD_eq_getElem?_getD, List.getElem?_map,
    List.getElem?_eq_getElem hl, he, Option.map_some, Option.getD_some]

theorem slots_sizes {ops : List TOp} {m : DevMem}
    (hsz : ∀ g ∈ unitBufs L.batch ops, L.bytes.getD g 0 ≤ (m g).size) :
    ∀ b, b < (unitBufs L.batch ops).length → (L.sizes ops).getD b 0 ≤ (slots ((unitBufs L.batch ops).map m) b).size :=
  fun b hb => by
    simp only [sizes, slots, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_eq_getElem hb,
      Option.map_some, Option.getD_some]
    exact hsz _ (List.getElem_mem hb)

theorem reads_mem {batch : Nat} {ops : List TOp} :
    ∀ op ∈ ops, ∀ b ∈ op.reads, b ∈ unitBufs batch ops := fun op hop _ hb =>
  (mem_eraseDups _ _).mpr (List.mem_flatMap.mpr ⟨op, hop, (mem_eraseDups _ _).mpr (List.mem_append_left _ hb)⟩)

theorem out_mem {batch : Nat} {ops : List TOp} {op : TOp} (hop : op ∈ ops) :
    (TOp.outSize batch op).1 ∈ unitBufs batch ops :=
  (mem_eraseDups _ _).mpr (List.mem_flatMap.mpr ⟨op, hop, (mem_eraseDups _ _).mpr (by simp)⟩)

/-- **Words a group's launch leaves, from what the device did**: where the
    bound buffers after a launch stand to those before as the group's kernel's
    run leaves them (`machine`), and every other buffer is as it was, every
    word of device memory is what the group's operations, folded in order at
    `F32` over the whole memory's words, leave there. -/
theorem unit_words {ops : List TOp} (hU : L.UnitOk ops) (hw : Hw)
    (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) {m m' : DevMem}
    (hsz : ∀ g ∈ unitBufs L.batch ops, L.bytes.getD g 0 ≤ (m g).size)
    (h4s : ∀ g ∈ unitBufs L.batch ops, (m g).size % 4 = 0)
    (hMr : L.machine ops hw ((unitBufs L.batch ops).map m) ((unitBufs L.batch ops).map m'))
    (hfr : ∀ g, g ∉ unitBufs L.batch ops → m' g = m g) :
    ∀ g j, j < 2 ^ 32 → 4 * j + 4 ≤ (m g).size →
      loadWord (m' g) j = (ops.foldl (fun m op => @TOp.den F32 (F32.ops hw) _ op m) (wordView m) g j).bits := by
  have hsz' := slots_sizes hsz
  obtain ⟨bufs', hrun, hMr⟩ := hMr (unit_runs hU hw hw32 _ hsz')
  intro g j hj hin
  have hnb := hU.nbuf
  by_cases hg : g ∈ unitBufs L.batch ops
  · obtain ⟨hlt, _⟩ := compactMap_get (unitBufs L.batch ops) g hg
    have hi : tabInj (unitBufs L.batch ops) g < (unitBufs L.batch ops).length := by
      rw [tabInj_on _ _ hg]; exact hlt
    have hslot := slots_tab hg m
    -- the device's word is the kernel's
    have e1 := hMr (tabInj (unitBufs L.batch ops) g) j hi hj (by
      have : (((unitBufs L.batch ops).map m).getD (tabInj (unitBufs L.batch ops) g) .empty) = m g := hslot
      rw [this]; exact hin)
    have hout : (((unitBufs L.batch ops).map m').getD (tabInj (unitBufs L.batch ops) g) .empty) = m' g :=
      slots_tab hg m'
    rw [hout] at e1
    -- the kernel's word is the group's fold over the slots
    have e2 := unit_launch hU hw hw32 _ bufs' hsz' hrun g j hg hj (by rw [hslot]; exact hin)
    have e3 := load_bufW_some ⟨hi, Nat.lt_trans hi hnb, hj⟩ e2
    -- the slots' view is the whole memory's on the table
    have hview : ∀ b ∈ unitBufs L.batch ops, (fun g' a => (ws0 (unitBufs L.batch ops).length
        (kmem (slots ((unitBufs L.batch ops).map m)))).mem (tabInj (unitBufs L.batch ops) g') a) b
        = wordView m b := fun b hb => funext fun a => by
      obtain ⟨hlb, _⟩ := compactMap_get (unitBufs L.batch ops) b hb
      have hib : tabInj (unitBufs L.batch ops) b < (unitBufs L.batch ops).length := by
        rw [tabInj_on _ _ hb]; exact hlb
      show (ws0 (unitBufs L.batch ops).length (kmem (slots ((unitBufs L.batch ops).map m)))).mem
        (tabInj (unitBufs L.batch ops) b) a = _
      rw [ws0_wordView ⟨hib, Nat.lt_trans hib hnb⟩ _ (by rw [slots_tab hb]; exact h4s b hb) a]
      simp only [wordView, slots_tab hb]
    have hfold := @foldl_den_congr F32 (F32.ops hw) _ (unitBufs L.batch ops) ops _ _ reads_mem hview g hg
    rw [← hfold, e3]
    show _ = loadWord (bufs' (tabInj (unitBufs L.batch ops) g)) j
    rw [e1]
  · rw [hfr g hg, @foldl_den_out F32 (F32.ops hw) _ L.batch g ops _ (fun op hop e => hg (e ▸ out_mem hop))]
    simp only [wordView, Nat.mod_eq_of_lt hj]

/-- **A group's launch through the driver leaves the whole device memory as
    its operations do.** From a world where `cuLaunchKernel` answers with its
    parameters pointing at the group's buffers, each holding whole words, at
    least as large as the layout makes them, on a device honest about the
    launch, every word of every buffer afterwards is what the group's
    operations, folded in order at `F32` over the whole memory's words, leave
    there. -/
theorem unit_drv_launch {ops : List TOp} (hU : L.UnitOk ops) (hw : Hw)
    (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    {w : World} {hf gx gy gz bx by_ bz shmem hstream params : UInt64}
    {fi mi p : Nat} {entry ptx : String} {sizes : List Nat} {args : List UInt64}
    (hl : w.dev.live = true) (hcap : w.dev.capture = none) (hk : KeepsSize w.kernel)
    (h1 : handleOf kFunc hf = some fi) (h2 : w.dev.funcs[fi]? = some (mi, entry))
    (h3 : w.dev.modules[mi]? = some (some ptx)) (h4 : w.dev.streamParty? hstream = some p)
    (h5 : ptxParamSizes ptx entry = some sizes) (h6 : kernelArgs w.mem params sizes = some args)
    (hdims : (DrvLaunch.dimsOf gx gy gz bx by_ bz).any (· == 0) = false)
    (hr : ∀ a ∈ args, ∀ id off b, w.dev.range? a 0 = some (id, off, b) → Ready w.dev.race p id true)
    (hns : DrvLaunch.bindsOf w.dev args = unitBufs L.batch ops)
    (hsz : ∀ g ∈ unitBufs L.batch ops, L.bytes.getD g 0 ≤ (devMem w.dev g).size)
    (h4s : ∀ g ∈ unitBufs L.batch ops, (devMem w.dev g).size % 4 = 0)
    (hM : Honest w.kernel ⟨ptx, entry, unitBufs L.batch ops, DrvLaunch.dimsOf gx gy gz bx by_ bz, args,
      w.dev.writesOf (w.dev.kernelWrites ptx entry) args⟩ (L.machine ops hw)) :
    ∃ r w', cudaDrv .launchKernel [hf, gx, gy, gz, bx, by_, bz, shmem, hstream, params, 0] w
        = some (r, w') ∧
      ∀ g j, j < 2 ^ 32 → 4 * j + 4 ≤ (devMem w.dev g).size →
        loadWord (devMem w'.dev g) j =
          (ops.foldl (fun m op => @TOp.den F32 (F32.ops hw) _ op m) (wordView (devMem w.dev)) g j).bits := by
  obtain ⟨r, w', h, hMr, hfr⟩ := DrvLaunch.drvLaunch_triple (L.machine ops hw)
    hl hcap hk h1 h2 h3 h4 h5 h6 hdims hr (by rw [hns]; exact hM)
  rw [hns] at hMr hfr
  exact ⟨r, w', h, unit_words hU hw hw32 hsz h4s hMr hfr⟩

/-- **A group's launch that answered leaves the whole device memory as its
    operations do**: a `cuLaunchKernel` of function `fi`, module `mi`'s entry
    `entry` of text `ptx`, that answered outside a capture, its parameter
    values `args` pointing at the group's buffers, on a device honest about
    the launch. -/
theorem unit_launch_done {ops : List TOp} (hU : L.UnitOk ops) (hw : Hw)
    (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    {w w' : World} {hf gx gy gz bx by_ bz shmem hstream params extra : UInt64} {r : Option V}
    {fi mi : Nat} {entry ptx : String} {args : List UInt64}
    (hcap : w.dev.capture = none) (hk : KeepsSize w.kernel)
    (hdims : (DrvLaunch.dimsOf gx gy gz bx by_ bz).any (· == 0) = false)
    (h : cudaDrv .launchKernel [hf, gx, gy, gz, bx, by_, bz, shmem, hstream, params, extra] w = some (r, w'))
    (h1 : handleOf kFunc hf = some fi) (h2 : w.dev.funcs[fi]? = some (mi, entry))
    (h3 : (w.dev.modules[mi]?).join = some ptx) (hargs : launchWords w.dev w.mem hf params = some args)
    (hns : DrvLaunch.bindsOf w.dev args = unitBufs L.batch ops)
    (hsz : ∀ g ∈ unitBufs L.batch ops, L.bytes.getD g 0 ≤ (devMem w.dev g).size)
    (h4s : ∀ g ∈ unitBufs L.batch ops, (devMem w.dev g).size % 4 = 0)
    (hM : Honest w.kernel ⟨ptx, entry, unitBufs L.batch ops, DrvLaunch.dimsOf gx gy gz bx by_ bz, args,
      w.dev.writesOf (w.dev.kernelWrites ptx entry) args⟩ (L.machine ops hw)) :
    ∀ g j, j < 2 ^ 32 → 4 * j + 4 ≤ (devMem w.dev g).size →
      loadWord (devMem w'.dev g) j =
        (ops.foldl (fun m op => @TOp.den F32 (F32.ops hw) _ op m) (wordView (devMem w.dev)) g j).bits := by
  obtain ⟨args', fi', mi', entry', ptx', ha', h1', h2', h3', hmem⟩ := DrvLaunch.drvLaunch_done hcap hdims h
  rw [hargs] at ha'; cases ha'
  rw [h1] at h1'; cases h1'
  rw [h2] at h2'; cases h2'
  rw [h3] at h3'; cases h3'
  rw [hns] at hmem
  have hnd : (unitBufs L.batch ops).Nodup := hns ▸ DrvLaunch.nodup_eraseDups _
  refine unit_words hU hw hw32 hsz h4s ?_ (fun g hg => ?_)
  · rw [hmem]
    exact launchStep_honest hk (by rw [← hns]; exact hns ▸ hM) hnd rfl _
  · rw [hmem]
    exact (launchStep _ _ _).frame _ g (fun hw' => hg (List.mem_filter.mp hw').1)

/-- **A group's operations at an in-bounds word read only in-bounds
    words**: from two memories that agree on every word inside each of the
    group's buffers, at the size the layout makes it, the operations'
    denotations, folded in order, agree at every word inside those buffers. -/
theorem unit_fold_local {ops : List TOp} (hU : L.UnitOk ops) (hw : Hw) (m m' : Buf → Nat → F32)
    (hag : ∀ g ∈ unitBufs L.batch ops, ∀ a, 4 * a + 4 ≤ L.bytes.getD g 0 → m g a = m' g a) :
    ∀ g ∈ unitBufs L.batch ops, ∀ j, 4 * j + 4 ≤ L.bytes.getD g 0 →
      ops.foldl (fun m op => @TOp.den F32 (F32.ops hw) _ op m) m g j
        = ops.foldl (fun m op => @TOp.den F32 (F32.ops hw) _ op m) m' g j := by
  intro g hg j hj
  let bs := unitBufs L.batch ops
  let M : Buf → Nat → F32 := fun b a => m (bs.getD b 0) a
  let M' : Buf → Nat → F32 := fun b a => m' (bs.getD b 0) a
  have htab : ∀ g' ∈ bs, bs.getD (tabInj bs g') 0 = g' := fun g' hg' => by
    obtain ⟨hl, he⟩ := compactMap_get bs g' hg'
    rw [tabInj_on _ _ hg', List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hl, Option.getD_some, he]
  have hM : ∀ b ∈ bs, (fun g' a => M (tabInj bs g') a) b = m b := fun b hb => funext fun a => by
    simp only [M, htab b hb]
  have hM' : ∀ b ∈ bs, (fun g' a => M' (tabInj bs g') a) b = m' b := fun b hb => funext fun a => by
    simp only [M', htab b hb]
  rw [← @foldl_den_congr F32 (F32.ops hw) _ bs ops _ _ reads_mem hM g hg,
    ← @foldl_den_congr F32 (F32.ops hw) _ bs ops _ _ reads_mem hM' g hg]
  obtain ⟨xs, hxs⟩ := hU.lowered hw
  have hbd := staticOk_bounds _ _ _ (unitStmt L.batch ops) _ (split_spine _) hU.static
  refine @group_local F32 (F32.ops hw) _ (litNatF32 hw) L.batch (unitGrid ops) ops bs xs hxs hU.grids
    (unit_covers L.batch ops) hU.apart M M' g j (fun c hc l b a h => ?_) ?_
  · obtain ⟨hb, -, -, hin⟩ := hbd c l b a hc h
    have hmem : bs.getD b 0 ∈ bs := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hb]; exact List.getElem_mem hb
    have hs : (L.sizes ops).getD b 0 = L.bytes.getD (bs.getD b 0) 0 := by
      simp only [sizes, bs, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_eq_getElem hb,
        Option.map_some, Option.getD_some]
    have hin' : 4 * a + 4 ≤ L.bytes.getD (bs.getD b 0) 0 := by rw [← hs]; exact hin
    exact hag _ hmem a hin'
  · simp only [M, M', htab g hg]
    exact hag g hg j hj

theorem foldl_congr_mem {α β : Type} {f g : α → β → α} :
    ∀ (l : List β) (a : α), (∀ b ∈ l, ∀ x, f x b = g x b) → l.foldl f a = l.foldl g a
  | [], _, _ => rfl
  | b :: l, a, h => by
      simp only [List.foldl_cons]
      rw [h b List.mem_cons_self a]
      exact foldl_congr_mem l _ fun b' hb' => h b' (List.mem_cons_of_mem _ hb')

/-- Operations that keep the frame leave every buffer none of them writes. -/
theorem foldl_frame_out {R : Type} {S : TOp → (Buf → Nat → R) → Buf → Nat → R} {batch : Nat}
    (hS : TOp.Frame batch S) (g : Buf) :
    ∀ (ops : List TOp) (M : Buf → Nat → R), (∀ op ∈ ops, g ≠ (TOp.outSize batch op).1) →
      ops.foldl (fun M op => S op M) M g = M g
  | [], _, _ => rfl
  | op :: ops, M, h => by
      simp only [List.foldl_cons]
      rw [foldl_frame_out hS g ops _ fun o ho => h o (List.mem_cons_of_mem _ ho)]
      exact hS.out op M g (h op List.mem_cons_self)

theorem bytes_le {L : TapeLayout} (hB : L.bytes.all (· ≤ 2 ^ 34) = true) (g : Nat) : L.bytes.getD g 0 ≤ 2 ^ 34 := by
  rw [List.getD_eq_getElem?_getD]
  cases h : L.bytes[g]? with
  | none => simp
  | some x => simpa using List.all_eq_true.mp hB x (List.mem_of_getElem? h)

/-- **A group's launch keeps device memory in step with the tape**: on a
    device honest about the launch (`machine`), from memory that agrees with
    `M`, its buffers at least the layout's sizes and whole words, the launch
    leaves memory that agrees with `M` after the group's operations. -/
theorem unit_step_kernel {ops : List TOp} (hU : L.UnitOk ops) (hB : L.bytes.all (· ≤ 2 ^ 34) = true) (hw : Hw)
    (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    {K : Launch → List ByteArray → List ByteArray} {l : Launch} (hb : l.binds = unitBufs L.batch ops)
    (hk : KeepsSize K) (hM : Honest K l (L.machine ops hw)) {mem : DevMem} {M : Buf → Nat → F32}
    (hsz : ∀ g ∈ unitBufs L.batch ops, L.bytes.getD g 0 ≤ (mem g).size)
    (h4s : ∀ g ∈ unitBufs L.batch ops, (mem g).size % 4 = 0)
    (hag : L.Agree mem M) :
    L.Agree ((launchStep K l (unitBufs L.batch ops)).run mem) (ops.foldl (fun M op => TOp.denV hw op M) M) := by
  have hnd : (unitBufs L.batch ops).Nodup := DrvLaunch.nodup_eraseDups _
  rw [foldl_congr_mem (g := fun M op => @TOp.den F32 (F32.ops hw) _ op M) _ M fun op hop x => by
    simp only [TOp.denV, gemmShape_none (hU.own op hop)]]
  intro g j hj
  have hj32 : j < 2 ^ 32 := by have := bytes_le hB g; omega
  by_cases hg : g ∈ unitBufs L.batch ops
  · have hwd := unit_words hU hw hw32 hsz h4s (launchStep_honest hk hM hnd hb mem)
      (fun g' hg' => (launchStep _ _ _).frame _ g' (fun h => hg' (List.mem_filter.mp h).1))
      g j hj32 (Nat.le_trans hj (hsz g hg))
    simp only [wordView, Nat.mod_eq_of_lt hj32]
    rw [hwd, unit_fold_local hU hw (wordView mem) M (fun g' _ a ha => hag g' a ha) g hg j hj]
  · rw [show wordView ((launchStep K l (unitBufs L.batch ops)).run mem) g j = wordView mem g j by
      simp only [wordView, (launchStep _ _ _).frame mem g (fun h => hg (List.mem_filter.mp h).1)]]
    rw [foldl_frame_out (letI := F32.ops hw; den_frame L.batch) g _ M fun op hop e => hg (e ▸ out_mem hop)]
    exact hag g j hj

/-- **A contraction's product keeps device memory in step with the tape**:
    on a device honest about its products, from memory that agrees with `M`,
    where the operands' buffers hold at least the words the product reads,
    the product leaves memory that agrees with `M` after the contraction read
    as the vendor computes it. -/
theorem gemm_step (hB : L.bytes.all (· ≤ 2 ^ 34) = true) (hw : Hw) {v : VendorCall → List ByteArray → ByteArray}
    (hv : VendorGemmHonest v) (hvk : Contracts.VendorKeeps v) {op : TOp} {ta tb m n k A B C : Nat}
    (hg : op.vendorGemm = some (ta, tb, m, n, k, A, B, C)) (hmk : m * k ≤ 2 ^ 32) (hkn : k * n ≤ 2 ^ 32)
    (hA : 4 * (m * k) ≤ L.bytes.getD A 0) (hB' : 4 * (k * n) ≤ L.bytes.getD B 0)
    {mem : DevMem} {M : Buf → Nat → F32} (hag : L.Agree mem M) :
    L.Agree ((vendorStep v (stridedCall 0x3f800000 0 0 0 0 (ta, tb, m, n, k, A, B, C)) [A, B, C] C).run mem)
      (TOp.denV hw op M) := by
  intro g j hj
  have hj32 : j < 2 ^ 32 := by have := bytes_le hB g; omega
  have h1 := gemm_words hw hv hvk hg hmk hkn mem g j hj32
  simp only [wordView, Nat.mod_eq_of_lt hj32]
  rw [h1]
  show TOp.denV hw op (wordView mem) g j = TOp.denV hw op M g j
  rw [denV_gemm (gemmShape_cublas hg), denV_gemm (gemmShape_cublas hg)]
  split
  · have hfa : (fun (r : Fin m) (t : Fin k) => (wordView mem A (cublasAt ta (if ta = 1 then k else m) r t)).toFloat)
        = (fun (r : Fin m) (t : Fin k) => (M A (cublasAt ta (if ta = 1 then k else m) r t)).toFloat) :=
      funext fun r => funext fun t => by
        have := cublasAt_lt (ta := ta) r.2 t.2
        rw [hag A _ (by omega)]
    have hfb : (fun (t : Fin k) (c : Fin n) => (wordView mem B (cublasBt tb (if tb = 1 then n else k) t c)).toFloat)
        = (fun (t : Fin k) (c : Fin n) => (M B (cublasBt tb (if tb = 1 then n else k) t c)).toFloat) :=
      funext fun t => funext fun c => by
        have := cublasBt_lt (tb := tb) t.2 c.2
        rw [hag B _ (by omega)]
    rw [hfa, hfb]
  · exact hag g j hj

/-! ## Steps keep device memory in step with the tape -/

variable (L)

/-- What a product step must be for the step lemma to read it: a group of
    one contraction issued as that product, its operands' buffers holding the
    words it reads. A launch's group is read by `UnitOk`. -/
def stepOk : DevCall × List TOp → Bool
  | (.kernel _, _) => true
  | (.gemm (ta, tb, m, n, k, A, B, C), ops) =>
      match ops with
      | [op] => op.vendorGemm == some (ta, tb, m, n, k, A, B, C) &&
          decide (m * k ≤ 2 ^ 32) && decide (k * n ≤ 2 ^ 32) &&
          decide (4 * (m * k) ≤ L.bytes.getD A 0) && decide (4 * (k * n) ≤ L.bytes.getD B 0)
      | _ => false
  | (.sync, _) => false

/-- What a step lemma needs of kernel `j`'s launch of group `ops`: the
    buffers its parameters name are the group's, and the launch leaves the
    words the group's kernel leaves. -/
def KernelOk (a : Drv.App) (w0 : World) (j : Nat) (ops : List TOp) (hw : Hw) : Prop :=
  ∀ k mi entry ptx, a.kernels[j]? = some k → w0.dev.funcs[j]? = some (mi, entry) →
    (w0.dev.modules[mi]?).join = some ptx →
    DrvLaunch.bindsOf w0.dev (recLaunchArgs k) = unitBufs L.batch ops ∧
    Honest w0.kernel ⟨ptx, entry, DrvLaunch.bindsOf w0.dev (recLaunchArgs k), recDims k, recLaunchArgs k,
      w0.dev.writesOf (w0.dev.kernelWrites ptx entry) (recLaunchArgs k)⟩ (L.machine ops hw)

/-- **What the device must be honest about for kernel `j`'s launch of group
    `ops`**: the launch leaves the words the group's kernel leaves. -/
def KernelHonest (a : Drv.App) (w0 : World) (j : Nat) (ops : List TOp) (hw : Hw) : Prop :=
  ∀ k mi entry ptx, a.kernels[j]? = some k → w0.dev.funcs[j]? = some (mi, entry) →
    (w0.dev.modules[mi]?).join = some ptx →
    Honest w0.kernel ⟨ptx, entry, unitBufs L.batch ops, recDims k, recLaunchArgs k,
      w0.dev.writesOf (w0.dev.kernelWrites ptx entry) (recLaunchArgs k)⟩ (L.machine ops hw)

variable {L}

/-- **Steps keep device memory in step with the tape**, each group's
    operations in turn. -/
theorem steps_agree (hB : L.bytes.all (· ≤ 2 ^ 34) = true) (a : Drv.App) (hw : Hw)
    (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) {w0 : World} (hk : KeepsSize w0.kernel) :
    ∀ (ps : List (DevCall × List TOp)), ps.all L.stepOk = true →
      (∀ p ∈ ps, ∀ g, p.1 = .gemm g → VendorGemmHonest w0.vendor ∧ VendorKeeps w0.vendor) →
      (∀ p ∈ ps, ∀ j, p.1 = .kernel j → L.UnitOk p.2 ∧ L.KernelOk a w0 j p.2 hw) →
      ∀ {m m' : DevMem} {M : Buf → Nat → F32},
        DevCall.Steps a w0 0x3f800000 0 (ps.map Prod.fst) m m' →
        (∀ p ∈ ps, ∀ g ∈ unitBufs L.batch p.2, L.bytes.getD g 0 ≤ (m g).size ∧ (m g).size % 4 = 0) →
        L.Agree m M →
        L.Agree m' ((ps.flatMap Prod.snd).foldl (fun M op => TOp.denV hw op M) M) := by
  intro ps
  rw [← List.reverse_reverse ps]
  induction ps.reverse with
  | nil =>
      intro _ _ _ m m' M hs _ hag
      rw [DevCall.Steps.nil_inv hs]
      exact hag
  | cons p rs ih =>
      rw [List.reverse_cons]
      generalize rs.reverse = ps0 at ih ⊢
      intro hok hV hK m m' M hs hS hag
      simp only [List.all_append, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true] at hok
      rw [List.map_append, List.map_singleton] at hs
      obtain ⟨m1, hs0, hstep⟩ := DevCall.Steps.snoc_inv hs
      have hag1 := ih hok.1 (fun q hq => hV q (List.mem_append_left _ hq)) (fun q hq => hK q (List.mem_append_left _ hq)) hs0
        (fun q hq => hS q (List.mem_append_left _ hq)) hag
      have hsz1 : ∀ g, (m1 g).size = (m g).size := DevCall.Steps.size hs0
      have hSp := hS p (List.mem_append_right _ List.mem_cons_self)
      rw [List.flatMap_append, List.foldl_append]
      simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil]
      obtain ⟨c, ops⟩ := p
      cases c with
      | kernel j =>
          obtain ⟨k, mi, entry, ptx, h1, h2, h3, rfl⟩ := hstep
          obtain ⟨hU, hKo⟩ := hK (.kernel j, ops) (List.mem_append_right _ List.mem_cons_self) j rfl
          obtain ⟨hb, hhon⟩ := hKo k mi entry ptx h1 h2 h3
          rw [hb] at hhon ⊢
          exact unit_step_kernel hU hB hw hw32 rfl hk hhon
            (fun g hg => (hsz1 g).symm ▸ (hSp g hg).1) (fun g hg => (hsz1 g).symm ▸ (hSp g hg).2) hag1
      | gemm g =>
          obtain ⟨ta, tb, mm, n, kk, A, B, C⟩ := g
          have ho := hok.2
          simp only [stepOk] at ho
          split at ho
          · rename_i op
            simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at ho
            obtain ⟨⟨⟨⟨hg, hmk⟩, hkn⟩, hA⟩, hB'⟩ := ho
            rw [List.foldl_cons, List.foldl_nil, hstep]
            obtain ⟨hv, hvk⟩ := hV _ (List.mem_append_right _ List.mem_cons_self) _ rfl
            exact gemm_step hB hw hv hvk hg hmk hkn hA hB' hag1
          · cases ho
      | sync => simp [stepOk] at hok

theorem range_live {d : Dev} {i : Nat} (hlt : i < 2 ^ 16) {b : ByteArray} (hb : (d.bufs[i]?).join = some b) :
    (d.range? (devPtr i) 0).map (·.1) = some i := by
  unfold Dev.range?
  rw [devPtrOf_devPtr hlt]
  show (Option.bind (d.bufs[i]?).join fun b => if 0 + 0 ≤ b.size then some (i, 0, b) else none).map (·.1) = some i
  rw [hb]
  simp

theorem filterMap_live {d : Dev} (args : List Nat) (h : ∀ i ∈ args, i < 2 ^ 16 ∧ ∃ b, (d.bufs[i]?).join = some b) :
    (args.map devPtr).filterMap (fun a => (d.range? a 0).map (·.1)) = args := by
  induction args with
  | nil => rfl
  | cons a as ih =>
      obtain ⟨hlt, b, hb⟩ := h a List.mem_cons_self
      rw [List.map_cons, List.filterMap_cons_some (f := fun a => (d.range? a 0).map (·.1)) (range_live hlt hb),
        ih fun i hi => h i (List.mem_cons_of_mem _ hi)]

variable (L)

/-- Each kernel's parameters are its group's buffers, every one an
    allocation of the layout a pointer can name, and it is launched over the
    grid and block its group's kernel is stated over. -/
def bindsOk (a : Drv.App) : DevCall × List TOp → Bool
  | (.kernel j, ops) => match a.kernels[j]? with
      | some k => k.args.all (fun i => decide (i < a.bufs.length) && decide (i < 2 ^ 16)) &&
          k.args.eraseDups == unitBufs L.batch ops &&
          k.grid == (unitGrid ops / L.W ops, 1, 1) && k.block == (32 * L.W ops, 1, 1)
      | none => true
  | _ => true

/-- A run's buffers are allocations of the layout at least as large as the
    tape lays them out, in whole words. -/
def sizesOk (a : Drv.App) (ps : List (DevCall × List TOp)) : Bool :=
  ps.all fun p => (unitBufs L.batch p.2).all fun g =>
    decide (g < a.bufs.length) && decide (L.bytes.getD g 0 ≤ a.bufs.getD g 0) && decide (a.bufs.getD g 0 % 4 = 0)

variable {L}

/-- **The words steps leave are the tape's**: steps `ps` taken in order, then
    a wait, from a world where every allocation of `a` is live at its size, on
    a device honest about each group's launch (`KernelHonest`) and about each
    product the steps take, each group read by `UnitOk` and each launch handed exactly its
    group's buffers, leave every word inside each buffer, at the size the
    layout makes it, as the groups' operations in order leave it from the
    memory the steps started with (or any memory agreeing with it there),
    each contraction read as the vendor computes it. -/
theorem steps_tape (hB : L.bytes.all (· ≤ 2 ^ 34) = true) (a : Drv.App) (hw : Hw)
    (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) {w w' : World} (ps : List (DevCall × List TOp))
    (hs : DevCall.Steps a w 0x3f800000 0 (ps.map Prod.fst ++ [.sync]) (devMem w.dev) (devMem w'.dev))
    (hks : KeepsSize w.kernel)
    (hv : ∀ p ∈ ps, ∀ g, p.1 = .gemm g → VendorGemmHonest w.vendor ∧ VendorKeeps w.vendor)
    (hst : ps.all L.stepOk = true) (hU : ∀ p ∈ ps, ∀ j, p.1 = .kernel j → L.UnitOk p.2)
    (hbo : ps.all (L.bindsOk a) = true) (hsz : L.sizesOk a ps = true)
    (halloc : ∀ i, i < a.bufs.length → ∃ b, (w.dev.bufs[i]?).join = some b ∧ b.size = a.bufs.getD i 0)
    (hHon : ∀ p ∈ ps, ∀ j, p.1 = .kernel j → L.KernelHonest a w j p.2 hw)
    {M : Buf → Nat → F32} (hA : L.Agree (devMem w.dev) M) :
    L.Agree (devMem w'.dev) ((ps.flatMap Prod.snd).foldl (fun M op => TOp.denV hw op M) M) := by
  -- each launch's parameters are its group's buffers
  have hK : ∀ p ∈ ps, ∀ j, p.1 = .kernel j → L.UnitOk p.2 ∧ L.KernelOk a w j p.2 hw := by
    intro p hp j hj
    refine ⟨hU p hp j hj, fun k mi entry ptx h1 h2 h3 => ?_⟩
    obtain ⟨c, ops⟩ := p
    dsimp only at hj
    subst hj
    have hbo := List.all_eq_true.mp hbo _ hp
    simp only [bindsOk, h1, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq, beq_iff_eq] at hbo
    replace hbo := hbo.1.1
    have hb : DrvLaunch.bindsOf w.dev (recLaunchArgs k) = unitBufs L.batch ops := by
      unfold DrvLaunch.bindsOf recLaunchArgs
      rw [filterMap_live k.args fun i hi => ⟨(hbo.1 i hi).2, (halloc i (hbo.1 i hi).1).imp fun _ hb => hb.1⟩]
      exact hbo.2
    refine ⟨hb, ?_⟩
    rw [hb]
    exact hHon _ hp j rfl k mi entry ptx h1 h2 h3
  -- the buffers start at least the layout's sizes, in whole words
  have hS : ∀ p ∈ ps, ∀ g ∈ unitBufs L.batch p.2,
      L.bytes.getD g 0 ≤ (devMem w.dev g).size ∧ (devMem w.dev g).size % 4 = 0 := by
    intro p hp g hg
    have hz := List.all_eq_true.mp (List.all_eq_true.mp hsz p hp) g hg
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hz
    obtain ⟨b, hb, hbs⟩ := halloc g hz.1.1
    simp only [devMem, hb, Option.getD_some, hbs]
    exact ⟨hz.1.2, hz.2⟩
  obtain ⟨m1, hs1, hsync⟩ := DevCall.Steps.snoc_inv hs
  have e : devMem w'.dev = m1 := hsync
  rw [e]
  exact steps_agree hB a hw hw32 hks ps hst hv hK hs1 hS hA

end TapeLayout

end AlgorithmLib.ML
