module
public import AlgorithmLib.Host.Blocks
meta import AlgorithmLib.Host.Blocks
public import Demo.ByteScrub
meta import Demo.ByteScrub
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem

/-!
  # What the scrubber's compare and blend compute

  `blend_scrubs` is stated over `ByteScrub.blend` itself: the proof compiles it
  with `Prog.emit`, the fold `compileProg` ships through, and runs what comes out
  under `Host.Sem`, the executable CLIF semantics that
  `base/tests/hprog_corpus.rs` checks against the JIT-compiled artifact. No
  instruction is written out by hand.

  It holds for any environment and any operand slots, so it covers `blend` where
  the loop in `ByteScrub.scrub` calls it. It does not show that the loop's
  environment holds what the hypotheses ask: the loop, the load and store around
  `blend`, and the two `splat`s are not modelled here.
  `applications/bytescrub/tests/scrub.rs` runs the whole artifact.
-/

namespace ByteScrub

/-- All ones on the NUL lanes, all zeros elsewhere. -/
def nulMask (xs : Array UInt64) : Sem.V :=
  .vec .i8x16 (xs.map fun x => if x = 0 then 255 else 0)

/-- The same bytes with every NUL made a space. -/
def scrubLanes (xs : Array UInt64) : Array UInt64 :=
  xs.map fun x => if x = 0 then 32 else x

def scrubbedBytes (xs : Array UInt64) : Sem.V := .vec .i8x16 (scrubLanes xs)

/-- **No byte the scrubber leaves is a NUL.**

    This is what the pass is for: a buffer it has written can be handed to an
    interface that stops at the first NUL. It is a fact about the bytes rather
    than a restatement of the definition, and it is what `blend_scrubs` is worth
    proving for --- that theorem says which bytes come out, and this says the
    thing about them that the caller needs. -/
theorem scrubLanes_no_nul (xs : Array UInt64) : ∀ y ∈ scrubLanes xs, y ≠ 0 := by
  intro y hy
  simp only [scrubLanes, Array.mem_map] at hy
  obtain ⟨x, _, rfl⟩ := hy
  split <;> simp_all

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
      = .ok ((Γ.push (nulMask xs)).push (scrubbedBytes xs)) w := by
  obtain ⟨ha', ha⟩ := Array.getElem?_eq_some_iff.mp ha
  obtain ⟨hb', hb⟩ := Array.getElem?_eq_some_iff.mp hb
  obtain ⟨hc', hc⟩ := Array.getElem?_eq_some_iff.mp hc
  -- compile blend
  conv in Prog.emit _ _ => reduce
  -- run both instructions under the semantics
  simp +decide [nulMask, scrubbedBytes, scrubLanes,
    runCode, runPiece, runStmts, runStmt, evalOp, Sem.get,
    zipIntCmp, cmpInt, ClifTy.lanes, widthMask, zipAnyBits, zipBitsIf,
    Array.getElem?_push_lt, Array.push_eq_push, ha, hb, hc, ha', hb', hc', hΓ, h16]
  -- lane by lane: a NUL, or any other byte
  constructor <;> ext k hk <;> simp_all <;> split <;> simp_all +decide

/-- **What running `blend` leaves, and that no NUL survives it.**

    The conclusion names no definition of this file, so it says what the sixteen
    bytes are --- each input byte, or a space where that byte was NUL --- rather
    than asserting that the code computes a function declared to be what it
    computes.

    `ys` is bound and then fixed by the first conjunct, so the existential is a
    naming and not a choice: `runCode` is a function, so the run has one result,
    and what this says is that the result is those bytes. The equation comes
    first for that reason --- read the other way round it invites the weaker
    reading, that some possible answer happens to be right.

    Both halves are needed. The second is the reason the pass exists: a buffer
    it has written can be handed to an interface that stops at the first NUL. On
    its own it would also hold of a program that wrote nothing but spaces, and
    the first is what rules that out. The comparison mask the first
    instruction leaves is not mentioned: `Answers` names only the last value,
    since nothing outside the pair reads the mask. -/
theorem blend_leaves_no_nul
    (Γ : Env) (tys : List ClifTy) (hΓ : tys.length = Γ.size)
    (a b c : R) (xs : Array UInt64) (h16 : xs.size = 16)
    (ha : Γ[a]? = some (.vec .i8x16 xs))
    (hb : Γ[b]? = some (.vec .i8x16 (.replicate 16 0)))
    (hc : Γ[c]? = some (.vec .i8x16 (.replicate 16 32))) :
    ∃ ys, ys = xs.map (fun x => if x = 0 then 32 else x)
          ∧ Answers Γ (Prog.emit (discard (blend a b c)) tys) 2 (.vec .i8x16 ys)
          ∧ ∀ y ∈ ys, y ≠ 0 :=
  ⟨scrubLanes xs, rfl,
   fun fuel cfg w =>
     ⟨_, blend_scrubs fuel cfg w Γ tys hΓ a b c xs h16 ha hb hc, by simp,
      by simp [scrubbedBytes]⟩,
   scrubLanes_no_nul xs⟩

-- ---------------------------------------------------------------------------
-- The same fact, about the instructions that ship
-- ---------------------------------------------------------------------------

/-!
`blend_scrubs` is about the term. What an artifact carries is blocks, and the
two are related by `compileBody`. For a straight-line body that relation is
proved in general — `AlgorithmLib.HProg.emitStmts_sim` and `straight_sound` —
and `blend` is straight-line, so the fact transports.

What transports is the *value* fact, not the trace: `blend` stores nothing, so
at the level of `Blocks.run` there is nothing to observe. `blend_clif` below is
therefore stated where the content is — over the value array the block
interpreter leaves behind.
-/

/-- The two statements `blend` denotes, with `n` the slot its comparison binds.
    `blend_emits` is what says this is the emitter's own answer and not a second
    account of it. -/
def blendStmts (a b c n : R) : List Stmt :=
  [.op (.icmp .eq a b), .op (.bitselect n c a)]

theorem blend_emits (a b c : R) (tys : List ClifTy) :
    Prog.emit (discard (blend a b c)) tys = [.straight (blendStmts a b c tys.length)] := by
  conv in Prog.emit _ _ => reduce
  rfl

/-- `blend_scrubs`, with the piece wrapper peeled off. -/
theorem blend_scrubs_stmts (fuel : Nat) (cfg : Cfg) (w : World)
    (Γ : Env) (tys : List ClifTy) (hΓ : tys.length = Γ.size)
    (a b c : R) (xs : Array UInt64) (h16 : xs.size = 16)
    (ha : Γ[a]? = some (.vec .i8x16 xs))
    (hb : Γ[b]? = some (.vec .i8x16 (.replicate 16 0)))
    (hc : Γ[c]? = some (.vec .i8x16 (.replicate 16 32))) :
    runStmts cfg Γ w (blendStmts a b c Γ.size)
      = .ok ((Γ.push (nulMask xs)).push (scrubbedBytes xs)) w := by
  have h := blend_scrubs fuel cfg w Γ tys hΓ a b c xs h16 ha hb hc
  rw [blend_emits, hΓ, runCode_straight] at h
  cases hr : runStmts cfg Γ w (blendStmts a b c Γ.size) with
  | stuck m => rw [hr] at h; simp at h
  | misuse m => rw [hr] at h; simp at h
  | fault m => rw [hr] at h; simp at h
  | ok Γ' w' =>
      rw [hr] at h
      simp only [CodeRes.ok.injEq] at h
      obtain ⟨h1, h2⟩ := h
      rw [h1, h2]

/-- **What the emitted CLIF computes.**

    The instructions `emitStmt` writes for `blend` — no instruction is written
    out by hand here either — run by the block interpreter from any agreeing
    state: the value the comparison binds is all ones exactly on the NUL lanes,
    and the value the blend binds is the bytes with every NUL made a space.

    This is `blend_scrubs` transported across compilation. The transport is
    `emitStmts_sim`, which holds for any straight-line body; `blend` is
    straight-line, so none of the loop-exit or branch-join numbering the two
    forms have disagreed over is involved. -/
theorem blend_clif (fuel : Nat) (cfg : Cfg) (w : World)
    (Γ : Env) (tys : List ClifTy) (hΓ : tys.length = Γ.size)
    (a b c : R) (xs : Array UInt64) (h16 : xs.size = 16)
    (ha : Γ[a]? = some (.vec .i8x16 xs))
    (hb : Γ[b]? = some (.vec .i8x16 (.replicate 16 0)))
    (hc : Γ[c]? = some (.vec .i8x16 (.replicate 16 32)))
    (s : CS) (hs : Aligned s Γ.size)
    (vals : Blocks.Vals) (hv : ValsAgree vals Γ) (rest : List Inst) :
    ∃ vals',
      Blocks.runInsts cfg.locals ⟨vals, w⟩ (emittedList s (blendStmts a b c Γ.size) ++ rest)
          = Blocks.runInsts cfg.locals ⟨vals', w⟩ rest
      ∧ Blocks.getV vals' ⟨Γ.size⟩ = some (nulMask xs)
      ∧ Blocks.getV vals' ⟨Γ.size + 1⟩ = some (scrubbedBytes xs) := by
  have haa : a < Γ.size := (Array.getElem?_eq_some_iff.mp ha).1
  have hbb : b < Γ.size := (Array.getElem?_eq_some_iff.mp hb).1
  have hcc : c < Γ.size := (Array.getElem?_eq_some_iff.mp hc).1
  have hsc : InScope Γ.size (blendStmts a b c Γ.size) := by
    refine ⟨?_, ?_, trivial⟩
    · intro r hr
      simp [Op.regs, Stmt.regs] at hr
      rcases hr with rfl | rfl
      · exact haa
      · exact hbb
    · intro r hr
      simp [Op.regs, Stmt.regs] at hr
      rcases hr with rfl | rfl | rfl
      · exact Nat.lt_succ_self _
      · exact Nat.lt_succ_of_lt hcc
      · exact Nat.lt_succ_of_lt haa
  obtain ⟨vals', hrun, hva⟩ :=
    emitStmts_sim cfg (blendStmts a b c Γ.size) s Γ.size hs hsc w w Γ _ vals rest
      rfl hv (blend_scrubs_stmts fuel cfg w Γ tys hΓ a b c xs h16 ha hb hc)
  refine ⟨vals', hrun, ?_, ?_⟩
  · rw [hva Γ.size (by simp only [Array.size_push]; omega)]
    simp [Array.getElem?_push]
  · rw [hva (Γ.size + 1) (by simp only [Array.size_push]; omega)]
    simp [Array.getElem_push]

-- ---------------------------------------------------------------------------
-- One fact about the CLIF the artifact ships
-- ---------------------------------------------------------------------------

/-!
`blend_clif` is about `blend` compiled as a body of its own. What ships is
`ByteScrub.code`, whose loop puts the same two operations inside its body block.
That block binds the carries again as its own parameters, and the term numbers
them as the body's own slots --- the carry that is slot 14 and `v14` in the head
is slot 16 and `v16` in the body --- so slot `i` is `v i` there as everywhere
else.

The pair is named here as instructions. `blendInsts_ship` is what keeps that
from being a transcription — it says these are the instructions block 8
carries, by running the compiler — and `blendInsts_scrub` says what running
them does.
-/

/-- The pair `byte_scrub`'s loop body carries. `v20` holds the sixteen bytes the
    trip just loaded, `v9` the NUL vector and `v11` the space vector, both built
    once before the loop. -/
def blendInsts : List Inst :=
  [.icmp ⟨21⟩ .eq ⟨20⟩ ⟨9⟩, .bitselect ⟨22⟩ ⟨21⟩ ⟨11⟩ ⟨20⟩]

/-- **These are the instructions that ship.** Block 8 of the entry point, in
    full, with the pair named. The compiler is run, not described. -/
theorem blendInsts_ship :
    (Prog.compileProg 1 code).toOption.bind (fun f => f.blocks[8]?)
      = some
        { ref := ⟨8⟩, params := [(⟨16⟩, ClifTy.i64)],
          insts :=
            [.iconst ⟨17⟩ .i64 4, .ishl ⟨18⟩ ⟨16⟩ ⟨17⟩, .iadd ⟨19⟩ ⟨1⟩ ⟨18⟩,
             .load ⟨20⟩ { kind := .plain, ty := .i8x16, notrapAligned := true } ⟨19⟩]
            ++ blendInsts
            ++ [.iadd ⟨23⟩ ⟨3⟩ ⟨18⟩, .storeTyped .i8x16 ⟨22⟩ ⟨23⟩,
                .iconst ⟨24⟩ .i64 1, .iadd ⟨25⟩ ⟨16⟩ ⟨24⟩, .jump ⟨7⟩ [⟨25⟩]] } := by
  conv in Prog.compileProg _ _ => reduce
  simp +decide [emitCode, emitPiece, emitLoop, emitIte, emitStmts, emitStmt,
    CS.open', CS.open'.go, CS.close, CS.fresh, CS.get,
    Trie.set, Trie.setGo, Trie.get, List.range, List.range.loop,
    List.mergeSort, blendInsts]
  rfl

/-- The compare marks the NUL lanes: given the loaded bytes in `v20` and the NUL
    vector in `v9`, it binds `v21` to all ones exactly where a byte was NUL. -/
theorem icmp_masks_nuls (m : Sem.Mem) (vals : Blocks.Vals) (xs : Array UInt64)
    (h16 : xs.size = 16)
    (hx : Blocks.getV vals ⟨20⟩ = some (.vec .i8x16 xs))
    (hn : Blocks.getV vals ⟨9⟩ = some (.vec .i8x16 (.replicate 16 0))) :
    Blocks.evalInst m vals (.icmp ⟨21⟩ .eq ⟨20⟩ ⟨9⟩) = some (⟨21⟩, nulMask xs) := by
  simp +decide [Blocks.evalInst, Blocks.evalInst.bin, Blocks.viaOp, hx, hn, nulMask,
    evalOp, Sem.get, zipIntCmp, cmpInt, ClifTy.lanes, widthMask, h16]
  ext k hk <;> simp_all <;> split <;> simp_all +decide

/-- The blend does the substitution: given that mask in `v21`, the space vector
    in `v11` and the bytes in `v20`, it binds `v22` to the bytes with every NUL
    made a space. -/
theorem bitselect_scrubs (m : Sem.Mem) (vals : Blocks.Vals) (xs : Array UInt64)
    (h16 : xs.size = 16)
    (hm : Blocks.getV vals ⟨21⟩ = some (nulMask xs))
    (hsp : Blocks.getV vals ⟨11⟩ = some (.vec .i8x16 (.replicate 16 32)))
    (hx : Blocks.getV vals ⟨20⟩ = some (.vec .i8x16 xs)) :
    Blocks.evalInst m vals (.bitselect ⟨22⟩ ⟨21⟩ ⟨11⟩ ⟨20⟩)
      = some (⟨22⟩, scrubbedBytes xs) := by
  simp only [Blocks.evalInst, hm, hsp, hx, Blocks.viaOp]
  simp +decide [nulMask, scrubbedBytes, scrubLanes, evalOp, Sem.get, zipAnyBits, zipBitsIf,
    ClifTy.lanes, h16]
  ext k hk <;> simp_all <;> split <;> simp_all +decide

/-- The two together, run by the block interpreter from any agreeing state and
    followed by whatever else the block holds. The same fact as the pair above,
    in the machinery that also performs jumps, stores and calls. -/
theorem blendInsts_scrub (lc : Locals) (w : World) (vals : Blocks.Vals)
    (rest : List Inst) (xs : Array UInt64) (h16 : xs.size = 16)
    (hx : Blocks.getV vals ⟨20⟩ = some (.vec .i8x16 xs))
    (hn : Blocks.getV vals ⟨9⟩ = some (.vec .i8x16 (.replicate 16 0)))
    (hsp : Blocks.getV vals ⟨11⟩ = some (.vec .i8x16 (.replicate 16 32))) :
    ∃ vals',
      Blocks.runInsts lc ⟨vals, w⟩ (blendInsts ++ rest)
          = Blocks.runInsts lc ⟨vals', w⟩ rest
      ∧ Blocks.getV vals' ⟨21⟩ = some (nulMask xs)
      ∧ Blocks.getV vals' ⟨22⟩ = some (scrubbedBytes xs) := by
  have hmask := icmp_masks_nuls w.mem vals xs h16 hx hn
  have h21 : Blocks.getV (Blocks.setV vals ⟨21⟩ (nulMask xs)) ⟨21⟩ = some (nulMask xs) :=
    getV_setV_self vals ⟨21⟩ (nulMask xs)
  have hx' : Blocks.getV (Blocks.setV vals ⟨21⟩ (nulMask xs)) ⟨20⟩
      = some (.vec .i8x16 xs) := by
    rw [getV_setV_ne vals 21 20 (nulMask xs) (by decide) (getV_lt vals 20 _ hx)]; exact hx
  have hsp' : Blocks.getV (Blocks.setV vals ⟨21⟩ (nulMask xs)) ⟨11⟩
      = some (.vec .i8x16 (.replicate 16 32)) := by
    rw [getV_setV_ne vals 21 11 (nulMask xs) (by decide) (getV_lt vals 11 _ hsp)]; exact hsp
  have hres := bitselect_scrubs w.mem (Blocks.setV vals ⟨21⟩ (nulMask xs)) xs h16 h21 hsp' hx'
  refine ⟨Blocks.setV (Blocks.setV vals ⟨21⟩ (nulMask xs)) ⟨22⟩ (scrubbedBytes xs),
          ?_, ?_, ?_⟩
  · simp [blendInsts, Blocks.runInsts, hmask, hres]
  · rw [getV_setV_ne _ 22 21 _ (by decide) (getV_lt _ 21 _ h21)]; exact h21
  · exact getV_setV_self _ ⟨22⟩ _

end ByteScrub
