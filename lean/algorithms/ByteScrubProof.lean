import ByteScrubAlgorithm

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem

/-!
  # What the scrubber's compare and blend compute

  `blend_scrubs` is stated over `ByteScrub.blend` itself: the proof compiles it
  with `Prog.emit`, the fold `compileProg` ships through, and runs what comes out
  under `HProgSem`, the executable CLIF semantics that
  `base/tests/hprog_corpus.rs` checks against the JIT-compiled artifact. No
  instruction is written out by hand.

  It holds for any environment and any operand slots, so it covers `blend` where
  the loop in `ByteScrub.scrub` calls it. It does not show that the loop's
  environment holds what the hypotheses ask: the loop, the load and store around
  `blend`, and the two `splat`s are not modelled here.
  `applications/bytescrub/tests/scrub.rs` runs the whole artifact.
-/

namespace ByteScrub

/-- Wherever `blend` runs, if its operands hold sixteen bytes, sixteen NULs and
    sixteen spaces, it leaves a mask that is all ones exactly on the NUL lanes,
    then the bytes with every NUL made a space. -/
theorem blend_scrubs (fuel : Nat) (cfg : Cfg) (w : World)
    (Γ : Env) (tys : List ClifTy) (hΓ : tys.length = Γ.size)
    (a b c : R) (xs : Array UInt64) (h16 : xs.size = 16)
    (ha : Γ[a]? = some (.vec .i8x16 xs))
    (hb : Γ[b]? = some (.vec .i8x16 (.replicate 16 0)))
    (hc : Γ[c]? = some (.vec .i8x16 (.replicate 16 32))) :
    runCode (fuel + 2) cfg Γ w (Prog.emit (discard (blend a b c)) tys)
      = .ok ((Γ.push (.vec .i8x16 (xs.map fun x => if x = 0 then 255 else 0))).push
               (.vec .i8x16 (xs.map fun x => if x = 0 then 32 else x))) w := by
  obtain ⟨ha', ha⟩ := Array.getElem?_eq_some_iff.mp ha
  obtain ⟨hb', hb⟩ := Array.getElem?_eq_some_iff.mp hb
  obtain ⟨hc', hc⟩ := Array.getElem?_eq_some_iff.mp hc
  -- compile blend
  conv in Prog.emit _ _ => reduce
  -- run both instructions under the semantics
  simp +decide [runCode, runPiece, runStmts, runStmt, evalOp, Sem.get,
    zipIntCmp, cmpInt, ClifTy.lanes, widthMask, zipAnyBits, zipBitsIf,
    Array.getElem?_push_lt, Array.push_eq_push, ha, hb, hc, ha', hb', hc', hΓ, h16]
  -- lane by lane: a NUL, or any other byte
  constructor <;> ext k hk <;> simp_all <;> split <;> simp_all +decide

end ByteScrub
