import AlgorithmLib.ML.Fuse
import AlgorithmLib.ML.Bind

/-!
# Per-kernel buffer tables

A kernel is emitted against the buffers it touches, not against the model's
whole table.  Binding every buffer to every kernel makes emitted text grow with
*operations × buffers* — quadratic in depth, and 83% of it parameter
declarations the kernel never reads.  `bufs_le_five` is the bound that removes
the second factor: a kernel's parameter list is fixed by the operation
vocabulary, so text grows with the number of operations alone.

The renaming is `Bind.lean`'s, which is already proven to preserve what a stage
computes; what is added here is the compaction map and the checks that it is
usable at each kernel.
-/

namespace AlgorithmLib.ML


/-- **The buffers one operation actually touches** — at most four.

    A kernel is emitted against this list, not against the model's whole buffer
    table, which is what keeps a kernel's text independent of how big the model
    is.

    Distinct, because an in-place operation reads and writes the same buffer:
    an optimiser step names its parameter as both an operand and its output, and
    a table listing it twice would declare a parameter the kernel cannot tell
    from another and would not be a renaming. -/
def TOp.bufs (batch : Nat) (op : TOp) : List Buf :=
  (TOp.reads op ++ [(TOp.outSize batch op).1]).eraseDups

/-- Global buffer number to this kernel's own slot. -/
def compactMap (bs : List Buf) : Buf → Buf := fun b => bs.findIdx (· == b)

/-- The statement a kernel is emitted from: the operation's, with its buffers
    renumbered to its own compact table. -/
def TOp.localStmt (batch : Nat) (op : TOp) : EWStmt :=
  (op.stmt batch).renameBuf (compactMap (op.bufs batch))

/-- **The compaction is injective on the buffers it renumbers**, which is what
    `StageSpec.rename` needs — decidable per kernel. -/
def TOp.compactInjB (batch : Nat) (op : TOp) : Bool :=
  let bs := op.bufs batch
  (bs.map (compactMap bs)).eraseDups.length == bs.length

/-- **Every kernel's buffers land inside its own table.** -/
def TOp.localBelowB (batch : Nat) (op : TOp) : Bool :=
  decide ((op.localStmt batch).BufBelow (op.bufs batch).length)

/-- **At most five buffers per operation, whatever the model is.**

    Four operands and an output is the widest an operation gets.  This is the
    scaling statement: a kernel's parameter list is bounded by the vocabulary,
    not by how many buffers the model has, so emitted text grows with the
    number of operations and not with their product. -/
theorem mem_eraseDups : ∀ (l : List Buf) (a : Buf), a ∈ l.eraseDups ↔ a ∈ l
  | [], _ => by simp
  | b :: as, a => by
      rw [List.eraseDups_cons]
      constructor
      · intro h
        rcases List.mem_cons.mp h with h | h
        · exact List.mem_cons.mpr (Or.inl h)
        · exact List.mem_cons_of_mem _
            (List.mem_filter.mp ((mem_eraseDups (as.filter fun c => !c == b) a).mp h)).1
      · intro h
        rcases List.mem_cons.mp h with h | h
        · exact List.mem_cons.mpr (Or.inl h)
        · by_cases hb : a = b
          · exact List.mem_cons.mpr (Or.inl hb)
          · exact List.mem_cons_of_mem _
              ((mem_eraseDups _ a).mpr (List.mem_filter.mpr ⟨h, by simp [hb]⟩))
  termination_by l => l.length
  decreasing_by all_goals exact Nat.lt_succ_of_le (List.length_filter_le _ _)

theorem length_eraseDups_le : ∀ (l : List Buf), l.eraseDups.length ≤ l.length
  | [] => by simp
  | a :: as => by
      rw [List.eraseDups_cons]
      have ih := length_eraseDups_le (as.filter fun b => !b == a)
      have hf := List.length_filter_le (fun b => !b == a) as
      simp only [List.length_cons]
      omega
  termination_by l => l.length
  decreasing_by exact Nat.lt_succ_of_le (List.length_filter_le _ _)

theorem bufs_le_five (batch : Nat) (op : TOp) : (op.bufs batch).length ≤ 5 := by
  refine Nat.le_trans (length_eraseDups_le _) ?_
  cases op <;> simp [TOp.reads, TOp.outSize]

end AlgorithmLib.ML
