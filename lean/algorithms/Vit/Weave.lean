import Vit.Model
import AlgorithmLib.ML.Model.WeaveLower

/-!
  # The ViT tape, read as an algebra program

  `WeaveCifar` writes a model in the algebra and lowers it to the tape that
  ships.  A transformer is too large to write twice — ViT's shipped tape is
  2294 operations after fusion — so this module runs the abstraction the other
  way: it *checks* that every operation the model already emits is a
  broadcasted operation, and derives the tape's denotation in the algebra's
  terms from that.

  Nothing about the model changes.  The claim is about `Vit.vFused`, the tape
  the artifact is built from, so it covers the forward pass, the derived
  backward pass and the optimiser step — every kernel, at the extents the
  model actually uses, including the padded key contractions whose output
  allocation is taller than the rows they write.
-/

namespace WeaveVit

open AlgorithmLib.ML AlgorithmLib.ML.Broadcast

/-- **Every operation the model ships is a broadcasted operation.**

    Decided on the tape itself, so no operation is taken on faith and none is
    outside the fragment. -/
theorem vit_is_algebra : (WOp.ofTape Vit.vFused).isSome = true := by
  native_decide

/-- The model's tape as a list of broadcasted operations. -/
def vitW : List WOp := (WOp.ofTape Vit.vFused).getD []

theorem vitW_ofTape : WOp.ofTape Vit.vFused = some vitW := by
  unfold vitW
  have h := vit_is_algebra
  cases hc : WOp.ofTape Vit.vFused with
  | none => rw [hc] at h; simp at h
  | some ws => rfl

/-- The abstraction dropped nothing: one broadcasted operation per shipped
    operation. -/
theorem vitW_length : vitW.length = Vit.vFused.length := by
  native_decide

/-- **The shipped ViT tape computes the denotation of the algebra program it
    is.**

    The right-hand side mentions only broadcasted operations, shapes and the
    reindexings their operands are read through.  The left-hand side is the
    fold the whole existing chain is stated about — `vit_lowering_sound` above
    it, `TenProg.run_den` below — so the algebra sits over the model that
    trains, not beside it. -/
theorem vit_tape_computes (m : Buf → Nat → Float32) :
    Vit.vFused.foldl (fun mm o => o.den mm) m = WOp.denAll vitW m :=
  WOp.ofTape_den Vit.vFused vitW vitW_ofTape m

/-- Lowering the abstracted program returns the shipped tape unchanged — the
    algebra is a description of this model, not a different one. -/
theorem vitW_lowers : WOp.lowerAll vitW = some Vit.vFused :=
  WOp.lowerAll_ofTape Vit.vFused vitW vitW_ofTape

end WeaveVit
