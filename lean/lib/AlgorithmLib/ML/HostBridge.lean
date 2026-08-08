import AlgorithmLib.HostIR
import AlgorithmLib.ML.Compose
import AlgorithmLib.ML.Rewrite

/-!
  # From the host program to the pipeline

  Two halves of the story meet here.

  `HostIR.lean` proves what the *host* does: `flatHI_sound` says the emitted
  CLIF — a real loop, body once, executed by a program counter — performs
  exactly the declared sequence of `LaunchRec`s.  A `LaunchRec` names a PTX
  slot and a grid.

  `ML/Compose.lean` proves what the *kernels* compute: `Pipeline.run_denote`
  says a list of `StageSpec`s executed in order computes the fold of their
  `step`s, for any length.

  What connects a PTX slot to a `StageSpec` is a fact about the *runtime* — that `cl_cuda_launch off … gx` runs the module at `off`
  on `gx` blocks — and it is the third row of `Clif.lean`'s trusted table.  This
  file makes it a value: a `KernelBinding` list, checked against each launch
  record by a decidable function.  A drifted slot or a grid that disagrees with
  the stage's own `grid` field yields `none` rather than a quietly wrong stage.
-/

namespace AlgorithmLib.ML

open AlgorithmLib.IR AlgorithmLib.Clif AlgorithmLib.Host

/-- **Which PTX slot holds which stage's kernel.**

    The one place the host-to-kernel assumption is written down.  Everything
    downstream quantifies over this table, so replacing a hand-tuned kernel
    means producing a new binding — and the stage it is bound to still has to
    carry its own `frame`/`value`/`valOnly` proofs. -/
structure KernelBinding where
  ptxOff : Nat
  /-- **Where the launch wrote its buffer-pointer array.**

      The slot alone does not identify a step.  A layer launches RoPE twice, the
      KV write twice and the bias add three times — same PTX, different buffers
      — and those are different memory transformations.  What tells them apart
      is the *bind*: the pointer array the host filled in before the launch.

      Keying the table on the slot alone would resolve all of them to one
      stage, which is precisely the confusion `StageSpec.rename` exists to
      remove.  A launch whose bind offset is not in the table is refused rather
      than matched to a stage proven about different buffers. -/
  bindOff : Nat
  /-- **…and what it wrote into it.**

      The offset says only *where* the pointer array is.  Without its contents,
      the buffers a stage is proven about are numbers chosen by hand with
      nothing relating them to the handles the host stored — so a table entry
      could bind `rmsStage` to an array holding entirely different buffers and
      every downstream theorem would still go through.

      `Clif.bindsOf` recovers the array's *contents* from the stores preceding
      each launch: `near k` for a handle loaded from layout slot `k` of the
      descriptor pointer, `far b k` for one reached through an unresolved base
      — a per-layer weight, whose base depends on the runtime layer index.
      Requiring them to match is what turns the stage's buffer numbering into a
      claim about the emitted program.

      The residue is a renaming — which layout slot holds which abstract buffer
      — and that is one named table per model rather than an unstated gap. -/
  bufs    : List BufDesc
  stage   : StageSpec

def bindingOf? : List KernelBinding → Nat → Nat → List BufDesc → Option StageSpec
  | [],                         _,   _,  _  => none
  | ⟨kOff, kBo, kBufs, S⟩ :: ks, off, bo, bs =>
      -- **The entry is destructured, not projected.**  `k.bindOff` and
      -- `⟨_, kBo, _, _⟩` are the same function — structure eta makes them
      -- definitionally equal — but they are not the same to *reduce*.  A
      -- `Nat` equality on a projection re-exposes the enclosing record at
      -- every step of `Nat.decEq`'s unary recursion, and this record's last
      -- field is a whole `StageSpec`, so each of the ~2000 steps walks a large
      -- term afresh.  Measured on the real thirteen-entry table: 750 ms with
      -- projections, 16 ms destructured — the same answer, 46× apart, and
      -- most of what a 194 s build was spending its time on.
      --
      -- `bindOff` first, deliberately: it is unique per launch, whereas a PTX
      -- slot repeats (a layer launches the bias add three times, RoPE and the
      -- KV write twice each).  Testing the repeating field first makes most
      -- entries fall through to the `bufs` list comparison before being
      -- rejected; testing the unique one first rejects them on a single `Nat`.
      if kBo = bo ∧ kOff = off ∧ kBufs = bs then some S
      else bindingOf? ks off bo bs

/-- **A device write together with the bind array the host had filled in for
    it** — `Clif.deviceOpsOf`'s element type.  A vendor call carries `none`; so
    does a launch whose array the scan could not resolve, and both are refused
    by `stageOf?` rather than matched on the record alone. -/
abbrev DeviceOp := LaunchRec × OpBinds

/-- **The stage a launch record names.**

    Five checks, each of which a `Pipeline` claim silently depends on.

    * the slot must be bound — otherwise nothing is known about the PTX that runs;
    * the *bind offset* must match too: same kernel, different buffer-pointer
      array, is a different step, and the table says which;
    * the bind array's **contents** must match, and must have been recovered at
      all — this is what ties the stage's buffer numbering to the handles the
      program stored, rather than to numbers a table author picked;
    * the launched grid must equal the stage's own `grid`, because every theorem
      in `Compose.lean` is about `runGrid S.ew S.grid`, so a record launching a
      different number of blocks realises a *different* function;
    * the launch must be on the stream this fragment is being realised at.
      `Pipeline.run` is a sequential fold, and two kernels issued on different
      streams are not ordered with respect to each other — so what makes a
      sequence of launches a pipeline is that they share one stream.  `str` is
      that stream: `none` is the default stream, which is ordered against
      everything, and a fragment whose records disagree realises nothing.

      A `_named` variant is refused outright: its extra argument shifts every
      later one, so `Clif.launchAt` declines to recover its fields and the
      match below would be reading the wrong positions. -/
def stageOf? (T : List KernelBinding) (str : Option Nat) (op : DeviceOp) : Option StageSpec :=
  if op.1.fnName ∉ AlgorithmLib.Clif.positionalLaunchNames then none
  else if op.1.stream ≠ str then none else
  match op.1.kernelOff, op.1.gridX, op.1.bindOff, op.1.nBufs, op.2.bufs with
  | some off, some g, some bo, some nb, some bs =>
      if bs.length = nb.toNat then
        match bindingOf? T off.toNat bo.toNat bs with
        | some S => if S.grid = g.toNat then some S else none
        | none   => none
      else none
  | _, _, _, _, _ => none

/-- **The stages a launch sequence realises**, or `none` if any record fails.
    Written as a structural recursion rather than `mapM` so that the two facts
    below are inductions rather than unfoldings of a monadic bind. -/
def stagesOf? (T : List KernelBinding) (str : Option Nat) :
    List DeviceOp → Option (List StageSpec)
  | []      => some []
  | r :: rs => match stageOf? T str r with
               | none   => none
               | some S => (stagesOf? T str rs).map (S :: ·)

/-- Order-preserving: the pipeline's stage order is the host's launch order. -/
def pipelineOf? (T : List KernelBinding) (str : Option Nat) (rs : List DeviceOp) : Option Pipeline :=
  (stagesOf? T str rs).map Pipeline.mk

-- ---------------------------------------------------------------------------
-- The vendor GEMV, as a declared step
-- ---------------------------------------------------------------------------

/-- **The matrix a GEMV's `A` handle denotes**, row-major with `cols` columns.
    A buffer is a flat address space; this is the one place the shape enters. -/
def matAt (mem : Buf → Nat → Float32) (aB : Buf) (rows cols : Nat) :
    Fin rows → Fin cols → Float32 := fun r c => mem aB (r.val * cols + c.val)

def vecAt (mem : Buf → Nat → Float32) (xB : Buf) (cols : Nat) :
    Fin cols → Float32 := fun c => mem xB c.val

/-- **`y = A·x` as a plan step.**

    Assumed in exactly one respect — *what* lands in `y` — and that respect is
    the *named law's* own opaque, `cublasSgemvResult`, not a second symbol
    beside it.  A step written against any other opaque would leave
    `Law.cublasIsMatvec` constraining something no plan mentions: stated,
    listed in the assumption ledger, and load-bearing for nothing.  Writing it
    this way is what makes `cublasStep_isMatvec` below provable, and with it
    the law reaches the seven `sgemv`s a layer performs.

    *Where* it can land is proven: `frame` is discharged here, so the step
    cannot touch a buffer the rest of the plan reasons about. -/
noncomputable def cublasStep (aB xB yB : Buf) (rows cols : Nat) : DeclaredStep where
  kernel := .cublasSgemv
  outs  := [yB]
  step  := fun mem b a =>
             if h : b = yB ∧ a < rows then
               cublasSgemvResult rows cols (matAt mem aB rows cols)
                 (vecAt mem xB cols) ⟨a, h.2⟩
             else mem b a
  frame := by
    intro mem b hb
    funext a
    rw [dif_neg (fun h => hb (by simp [h.1]))]

/-- **What the vendor GEMV computes, under the law that names it.**

    The statement `Law.cublasIsMatvec` exists to license: row `i` of the output
    is `Σⱼ A[i,j]·x[j]` over the buffer's own contents.  Everything a plan
    containing `cublasStep` denotes is this, up to the fold order the law
    abstracts away — which is the whole of the gap, and it is a rewrite a
    downstream spec applies rather than a sentence in a docstring. -/
theorem cublasStep_isMatvec (hl : CuBlasIsMatvec) (aB xB yB : Buf) (rows cols : Nat)
    (mem : Buf → Nat → Float32) (i : Nat) (hi : i < rows) :
    (cublasStep aB xB yB rows cols).step mem yB i
      = (List.finRange cols).foldl
          (fun acc j => NumOps.add acc
            (NumOps.mul (mem aB (i * cols + j.val)) (mem xB j.val)))
          (NumOps.ofNat 0) := by
  show (if h : yB = yB ∧ i < rows then
          cublasSgemvResult rows cols (matAt mem aB rows cols)
            (vecAt mem xB cols) ⟨i, h.2⟩
        else mem yB i) = _
  rw [dif_pos (And.intro rfl hi), hl]
  rfl

/-- **A batched GEMM's row**, and this one really is opaque.

    Attention issues two of these per layer — the score contraction and the
    output contraction — and they are a *stronger* assumption than the GEMV,
    not a weaker one.  `cublasStep` is written in terms of the law's own
    symbol, so `Law.cublasIsMatvec` says what it computes.  Nothing says what
    this computes: batched strided mode selects its operand slices by four
    stride and batch-count arguments the launch model recovers as constants but
    does not interpret, so writing an equation here would be inventing a
    contract rather than recording one.

    So of a layer's nine declared steps, seven are covered by a named law and
    two are bare.  `declaredLawGap` below is that number, and it is the honest
    answer to "how much of attention is assumed". -/
opaque sgemmBatchedRow (mem : Buf → Nat → Float32) (aB bB : Buf) (k i : Nat) : Float32

/-- `C = A·B`, batched, as a plan step.  As with `cublasStep`, *what* lands in
    `C` is assumed and *where* it can land is proven. -/
noncomputable def sgemmBatchedStep (aB bB cB : Buf) (rows k : Nat) : DeclaredStep where
  kernel := .cublasSgemmStridedBatched
  outs  := [cB]
  step  := fun mem b a =>
             if b = cB ∧ a < rows then sgemmBatchedRow mem aB bB k a else mem b a
  frame := by
    intro mem b hb
    funext a
    simp [show b ≠ cB from fun h => hb (by simp [h])]

/-- **The primitives assumed with no stated equation behind them.**

    Derived by filtering `VendorKernel.all`, which `VendorKernel.all_covers`
    holds to every constructor — so a primitive added without a law cannot be
    missing from this bill. -/
def lawlessNames : List String :=
  (VendorKernel.all.filter VendorKernel.lawless).map VendorKernel.symbol

/-- **How many of a plan's declared steps no law says anything about.**

    `Plan.declaredCount` counts what is not proven.  This counts what is not
    even *stated*: a step whose primitive has a law is assumed only up to that
    law's content, and one without is assumed entirely.  For a Qwen2 layer the
    two numbers are 9 and 2. -/
def Plan.declaredLawGap (P : Plan) : Nat :=
  (P.declaredNames.filter (fun n => n ∈ lawlessNames)).length

-- ---------------------------------------------------------------------------
-- Realising a whole device-write sequence
-- ---------------------------------------------------------------------------

/-- Which primitive name maps to which declared step.  The declared counterpart
    of `KernelBinding`: a launch record names a PTX slot, a device-write record
    names a primitive, and neither says by itself what it computes. -/
structure DeclaredBinding where
  name : String
  /-- **Which call this is.**  A vendor primitive has no PTX slot, so name alone
      does not identify a call site: Qwen2 issues seven `cl_cublas_sgemv` calls
      per layer and three of them are the same 896×896 shape.  Matching on the
      recovered argument descriptors — which slot each buffer handle came from —
      is what tells `Wq` from `Wk` from `Wv`. -/
  args : List BufDesc
  decl : DeclaredStep

def declaredOf? : List DeclaredBinding → String → List BufDesc → Option DeclaredStep
  | [],                   _,  _  => none
  | ⟨dNm, dArgs, dD⟩ :: ds, nm, ar =>
      -- Destructured for the reason given at `bindingOf?`: the record's last
      -- field is a `DeclaredStep`, and projecting past it during a reduction
      -- costs proportionally to it.
      --
      -- `args` first, for the same reason `bindingOf?` tests `bindOff` first:
      -- every declared entry in a transformer layer is named `cl_cublas_sgemv`,
      -- so the name never discriminates and testing it first means each of the
      -- nine entries is rejected only after comparing a nine-element argument
      -- list.  The arguments are what tell `Wq` from `Wk` from `Wv`.
      if dArgs = ar ∧ dNm = nm then some dD else declaredOf? ds nm ar

/-- **The plan step a device-write record realises.**

    A launch record on this fragment's stream goes through the kernel table and
    yields a proven stage; anything named in `deviceWriterNames` goes through
    the declared table and yields an assumed one.  Anything else — a launch on
    another stream, an unbound slot, a grid disagreeing with the stage's own —
    yields `none`, so a plan claim is simply unavailable rather than quietly
    wrong. -/
def stepOf? (T : List KernelBinding) (D : List DeclaredBinding)
    (str : Option Nat) (op : DeviceOp) : Option PStep :=
  if op.1.fnName ∈ AlgorithmLib.Clif.launchNames then
    (stageOf? T str op).map PStep.proven
  else if op.1.fnName ∈ deviceWriterNames then
    (declaredOf? D op.1.fnName op.2.args).map PStep.declared
  else none

def stepsOf? (T : List KernelBinding) (D : List DeclaredBinding)
    (str : Option Nat) : List DeviceOp → Option (List PStep)
  | []      => some []
  | r :: rs => match stepOf? T D str r with
               | none   => none
               | some t => (stepsOf? T D str rs).map (t :: ·)

def planOf? (T : List KernelBinding) (D : List DeclaredBinding)
    (str : Option Nat) (rs : List DeviceOp) : Option Plan :=
  (stepsOf? T D str rs).map Plan.mk

/-- **Realising a sequence realises each of its parts** — the `PStep` analogue
    of `stagesOf?_append`, and what makes a whole token's 533 device writes a
    plan without reducing 533 steps: the layer is resolved once and the loop is
    an induction. -/
theorem stepsOf?_append (T : List KernelBinding) (D : List DeclaredBinding)
    (str : Option Nat) :
    ∀ (r₁ r₂ : List DeviceOp) (L₁ L₂ : List PStep),
      stepsOf? T D str r₁ = some L₁ → stepsOf? T D str r₂ = some L₂ →
      stepsOf? T D str (r₁ ++ r₂) = some (L₁ ++ L₂) := by
  intro r₁
  induction r₁ with
  | nil => intro r₂ L₁ L₂ h1 h2; cases h1; exact h2
  | cons r rs ih =>
      intro r₂ L₁ L₂ h1 h2
      show (match stepOf? T D str r with
            | none => none
            | some t => (stepsOf? T D str (rs ++ r₂)).map (t :: ·)) = _
      revert h1
      cases hS : stepOf? T D str r with
      | none => intro h1; exact absurd h1 (by simp [stepsOf?, hS])
      | some t =>
          intro h1
          have h1' : (stepsOf? T D str rs).map (t :: ·) = some L₁ := by
            rw [← h1]; show _ = (match stepOf? T D str r with
                                  | none => none
                                  | some t => (stepsOf? T D str rs).map (t :: ·))
            rw [hS]
          cases hrest : stepsOf? T D str rs with
          | none => rw [hrest] at h1'; exact absurd h1' (by simp)
          | some rest =>
              rw [hrest] at h1'
              have : t :: rest = L₁ := by simpa using h1'
              rw [ih r₂ rest L₂ hrest h2, ← this]
              rfl

-- ---------------------------------------------------------------------------
-- Widening the tables
-- ---------------------------------------------------------------------------

/-! **A resolution against a table survives extending that table.**

    Not a convenience.  Qwen2's token-level tables are the layer's plus three
    more kernels and two more vendor calls, so without these the same
    twenty-two device writes are resolved twice — once against `layerKernels`
    and again against `tokenKernels` — and each resolution is a linear scan of
    the table for every op.  Measured: the second scan alone was 12.1 s of a
    30 s build.  With these it is a rewrite.

    The lemmas are stated as `T ++ T'` rather than as a `Sublist` because that
    is the shape the tables actually have, and because a prepended entry could
    genuinely shadow a later one — the scan takes the *first* match. -/

theorem bindingOf?_append : ∀ (T T' : List KernelBinding) (off bo : Nat)
    (bs : List BufDesc) (S : StageSpec),
    bindingOf? T off bo bs = some S → bindingOf? (T ++ T') off bo bs = some S := by
  intro T
  induction T with
  | nil => intro _ _ _ _ _ h; exact Option.noConfusion h
  | cons k ks ih =>
      intro T' off bo bs S h
      have h' : (if k.bindOff = bo ∧ k.ptxOff = off ∧ k.bufs = bs then some k.stage
                 else bindingOf? ks off bo bs) = some S := h
      show (if k.bindOff = bo ∧ k.ptxOff = off ∧ k.bufs = bs then some k.stage
            else bindingOf? (ks ++ T') off bo bs) = some S
      by_cases hc : k.bindOff = bo ∧ k.ptxOff = off ∧ k.bufs = bs
      · rw [if_pos hc]; rw [if_pos hc] at h'; exact h'
      · rw [if_neg hc]; rw [if_neg hc] at h'; exact ih T' off bo bs S h'

theorem declaredOf?_append : ∀ (D D' : List DeclaredBinding) (nm : String)
    (ar : List BufDesc) (d : DeclaredStep),
    declaredOf? D nm ar = some d → declaredOf? (D ++ D') nm ar = some d := by
  intro D
  induction D with
  | nil => intro _ _ _ _ h; exact Option.noConfusion h
  | cons e es ih =>
      intro D' nm ar d h
      have h' : (if e.args = ar ∧ e.name = nm then some e.decl
                 else declaredOf? es nm ar) = some d := h
      show (if e.args = ar ∧ e.name = nm then some e.decl
            else declaredOf? (es ++ D') nm ar) = some d
      by_cases hc : e.args = ar ∧ e.name = nm
      · rw [if_pos hc]; rw [if_pos hc] at h'; exact h'
      · rw [if_neg hc]; rw [if_neg hc] at h'; exact ih D' nm ar d h'

theorem stageOf?_append (T T' : List KernelBinding) (str : Option Nat)
    (op : DeviceOp) (S : StageSpec)
    (h : stageOf? T str op = some S) : stageOf? (T ++ T') str op = some S := by
  unfold stageOf? at h ⊢
  by_cases hn : op.1.fnName ∉ AlgorithmLib.Clif.positionalLaunchNames
  · rw [if_pos hn] at h; exact Option.noConfusion h
  rw [if_neg hn] at h ⊢
  by_cases hst : op.1.stream ≠ str
  · rw [if_pos hst] at h; exact Option.noConfusion h
  rw [if_neg hst] at h ⊢
  -- `split` reduces both sides: the two `match`es have the same discriminants,
  -- and the table is the only thing that differs.
  split at h
  · rename_i off g bo nb bs _ _ _ _ _
    by_cases hl : bs.length = nb.toNat
    · rw [if_pos hl] at h ⊢
      cases hb : bindingOf? T off.toNat bo.toNat bs with
      | none => rw [hb] at h; exact Option.noConfusion h
      | some S' =>
          rw [bindingOf?_append T T' off.toNat bo.toNat bs S' hb]
          rw [hb] at h
          exact h
    · rw [if_neg hl] at h; exact Option.noConfusion h
  · exact Option.noConfusion h

theorem stepOf?_append (T T' : List KernelBinding) (D D' : List DeclaredBinding)
    (str : Option Nat) (op : DeviceOp) (t : PStep) (h : stepOf? T D str op = some t) :
    stepOf? (T ++ T') (D ++ D') str op = some t := by
  have h0 : (if op.1.fnName ∈ AlgorithmLib.Clif.launchNames then
               (stageOf? T str op).map PStep.proven
             else if op.1.fnName ∈ Clif.deviceWriterNames then
               (declaredOf? D op.1.fnName op.2.args).map PStep.declared
             else none) = some t := h
  show (if op.1.fnName ∈ AlgorithmLib.Clif.launchNames then
          (stageOf? (T ++ T') str op).map PStep.proven
        else if op.1.fnName ∈ Clif.deviceWriterNames then
          (declaredOf? (D ++ D') op.1.fnName op.2.args).map PStep.declared
        else none) = some t
  by_cases hl : op.1.fnName ∈ AlgorithmLib.Clif.launchNames
  · rw [if_pos hl] at h0 ⊢
    cases hs : stageOf? T str op with
    | none => rw [hs] at h0; exact Option.noConfusion h0
    | some S => rw [stageOf?_append T T' str op S hs]; rw [hs] at h0; exact h0
  rw [if_neg hl] at h0 ⊢
  by_cases hw : op.1.fnName ∈ Clif.deviceWriterNames
  · rw [if_pos hw] at h0 ⊢
    cases hd : declaredOf? D op.1.fnName op.2.args with
    | none => rw [hd] at h0; exact Option.noConfusion h0
    | some d =>
        rw [declaredOf?_append D D' op.1.fnName op.2.args d hd]
        rw [hd] at h0
        exact h0
  · rw [if_neg hw] at h0; exact Option.noConfusion h0

/-- **The whole sequence, widened.**  This is the one call sites use: a plan
    resolved against a layer's tables is the same plan against the token's. -/
theorem stepsOf?_appendTable (T T' : List KernelBinding) (D D' : List DeclaredBinding)
    (str : Option Nat) :
    ∀ (ops : List DeviceOp) (L : List PStep),
      stepsOf? T D str ops = some L → stepsOf? (T ++ T') (D ++ D') str ops = some L := by
  intro ops
  induction ops with
  | nil => intro L h; exact h
  | cons op rest ih =>
      intro L h
      have h0 : (match stepOf? T D str op with
                 | none => none
                 | some t => (stepsOf? T D str rest).map (t :: ·)) = some L := h
      show (match stepOf? (T ++ T') (D ++ D') str op with
            | none => none
            | some t => (stepsOf? (T ++ T') (D ++ D') str rest).map (t :: ·)) = some L
      cases hs : stepOf? T D str op with
      | none => rw [hs] at h0; exact Option.noConfusion h0
      | some t =>
          rw [stepOf?_append T T' D D' str op t hs]
          rw [hs] at h0
          cases hr : stepsOf? T D str rest with
          | none => rw [hr] at h0; exact Option.noConfusion h0
          | some rest' => rw [ih rest' hr]; rw [hr] at h0; exact h0

/-- **Every device write is accounted for, and the assumed ones are counted.**

    The number a report should quote.  It is not a caveat in prose: it is a
    projection of the plan, and it reaches zero exactly when every step has a
    `StageSpec`. -/
def declaredCountOf (T : List KernelBinding) (D : List DeclaredBinding)
    (str : Option Nat) (rs : List DeviceOp) : Option Nat :=
  (planOf? T D str rs).map Plan.declaredCount

-- ---------------------------------------------------------------------------
-- The two halves, joined
-- ---------------------------------------------------------------------------

/-- **The executed launch sequence realises the same pipeline as the declared
    one.**

    An immediate corollary of `flatHI_sound`, and that is the point: because the
    trace the machine accumulates *is* `HStmt.launches`, anything proven about
    the declared sequence transfers to the sequence the compiled program
    actually performs — through the layer loop and the call, neither of which a
    static scan of the emitted CLIF can see through.

    `bs` is the bind-array list the scan recovered for the same code — supplied
    rather than derived, because `HStmt` declares a launch *sequence* and the
    arrays are recovered by `Clif.bindsOf` from the instructions.  It is not a
    new assumption: the caller discharges `hpl` on the real program, where `bs`
    *is* `Clif.bindsOf` of it, and `Clif.bindsOf_length` says the zip drops
    nothing. -/
theorem host_realises_pipeline
    (fns : List FnDecl) (fnLaunch : FnRef) (ptr : Val)
    (hfn : fnNameOf fns fnLaunch = some "cl_cuda_launch") (P : List HI)
    (s : HStmt) (n p : Nat) (e : Env) (sm : StoreMap) (ct : Nat → Nat)
    (hptr : ptr.id < n) (he : e ptr = SymVal.unknown)
    (htame : HStmt.TameB fns ptr s = true)
    (hfar : FarOk e ptr n s.farArgs)
    (hfd : ∀ x ∈ s.farArgs, ∀ b ∈ x.deps, b ∉ s.primDests)
    (hfit : Fits P p (code fnLaunch ptr n p s))
    (T : List KernelBinding) (Pl : Pipeline)
    (str : Option Nat) (hpl : pipelineOf? T str s.deviceOps = some Pl) :
    ∃ k c', hsteps fns ptr.id P k ⟨p, e, sm, ct, [], []⟩ = some c'
      ∧ pipelineOf? T str (c'.trace.zip c'.btrace) = some Pl := by
  obtain ⟨k, c', hr, _, htr, hbt, _, _⟩ :=
    flatHI_sound fns fnLaunch ptr hfn P s n p e sm ct [] [] hptr he htame hfar hfd hfit
  exact ⟨k, c', hr, by
    rw [htr, hbt, List.nil_append, List.nil_append]; exact hpl⟩

/-- **End to end on the host side.**

    Running the emitted program performs a device-write sequence which, under
    the kernel table, *is* a pipeline; and executing that pipeline computes its
    denotation — the fold of the stages' `step`s over memory.

    Read as a chain: the loop-and-call structure of the driver (proven), the
    device writes it performs *and the buffers each was handed* (proven), the
    stages those resolve to (checked against the table), and what those stages
    compute (proven).  What is left trusted between the third and fourth links
    is exactly the table: that the PTX at slot `off` is the compiled `S.ew`.  It
    is a hypothesis with a name rather than a step nothing mentions. -/
theorem host_computes_denote
    (fns : List FnDecl) (fnLaunch : FnRef) (ptr : Val)
    (hfn : fnNameOf fns fnLaunch = some "cl_cuda_launch") (P : List HI)
    (s : HStmt) (n p : Nat) (e : Env) (sm : StoreMap) (ct : Nat → Nat)
    (hptr : ptr.id < n) (he : e ptr = SymVal.unknown)
    (htame : HStmt.TameB fns ptr s = true)
    (hfar : FarOk e ptr n s.farArgs)
    (hfd : ∀ x ∈ s.farArgs, ∀ b ∈ x.deps, b ∉ s.primDests)
    (hfit : Fits P p (code fnLaunch ptr n p s))
    (T : List KernelBinding) (Pl : Pipeline)
    (str : Option Nat) (hpl : pipelineOf? T str s.deviceOps = some Pl)
    (hex : Pl.Exclusive) (st : WSt) :
    ∃ k c', hsteps fns ptr.id P k ⟨p, e, sm, ct, [], []⟩ = some c'
      ∧ pipelineOf? T str (c'.trace.zip c'.btrace) = some Pl
      ∧ (Pl.run st).mem = Pl.denote st.mem := by
  obtain ⟨k, c', hr, hp⟩ :=
    host_realises_pipeline fns fnLaunch ptr hfn P s n p e sm ct hptr he htame hfar hfd hfit
      T Pl str hpl
  exact ⟨k, c', hr, hp, Pipeline.run_denote Pl hex st⟩

/-- **The whole thing, with the declared steps in it.**

    Running the emitted CLIF performs a device-write sequence — records *and*
    bind arrays, through the loop; under the two tables that sequence *is* a
    `Plan`; and the plan computes its denotation given `Honours R`, one named
    hypothesis covering every declared step.

    This is what "proven all the way" means here. The chain from a host loop to
    a memory transformation is closed; what it rests on is enumerated rather
    than assumed silently: the PTX printer, the PTX model and driver, and the
    performance gaps taken deliberately — float associativity, and cuBLAS's
    unspecified fold order. `Plan.declaredCount` measures the last of these.

    **Nothing is supplied alongside the program.**  Taking the bind list as a
    parameter would mean sourcing it from a separate static scan, which cannot
    see through the layer loop — leaving what the twenty-fourth iteration bound
    outside the theorem.  The machine executes the bind stores, so the arrays
    in the conclusion are the ones the program wrote. -/
theorem host_computes_plan
    (fns : List FnDecl) (fnLaunch : FnRef) (ptr : Val)
    (hfn : fnNameOf fns fnLaunch = some "cl_cuda_launch") (P : List HI)
    (s : HStmt) (n p : Nat) (e : Env) (sm : StoreMap) (ct : Nat → Nat)
    (hptr : ptr.id < n) (he : e ptr = SymVal.unknown)
    (htame : HStmt.TameB fns ptr s = true)
    (hfar : FarOk e ptr n s.farArgs)
    (hfd : ∀ x ∈ s.farArgs, ∀ b ∈ x.deps, b ∉ s.primDests)
    (hfit : Fits P p (code fnLaunch ptr n p s))
    (T : List KernelBinding) (D : List DeclaredBinding) (Pl : Plan)
    (str : Option Nat) (hpl : planOf? T D str s.deviceOps = some Pl)
    (R : Realisation) (hR : Honours R) (hex : Pl.Exclusive) (st : WSt) :
    ∃ k c', hsteps fns ptr.id P k ⟨p, e, sm, ct, [], []⟩ = some c'
      ∧ planOf? T D str (c'.trace.zip c'.btrace) = some Pl
      ∧ (Pl.run R st).mem = Pl.denote st.mem := by
  obtain ⟨k, c', hr, _, htr, hbt, _, _⟩ :=
    flatHI_sound fns fnLaunch ptr hfn P s n p e sm ct [] [] hptr he htame hfar hfd hfit
  exact ⟨k, c', hr,
         by rw [htr, hbt, List.nil_append, List.nil_append]; exact hpl,
         Plan.run_denote R hR Pl hex st⟩

end AlgorithmLib.ML
