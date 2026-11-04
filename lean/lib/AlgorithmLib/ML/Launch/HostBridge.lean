module
public import AlgorithmLib.ML.Launch.Record
meta import AlgorithmLib.ML.Launch.Record
public import AlgorithmLib.Host.DrvCodes
meta import AlgorithmLib.Host.DrvCodes
public import AlgorithmLib.Host.World
meta import AlgorithmLib.Host.World
public import AlgorithmLib.ML.Launch.Declared
meta import AlgorithmLib.ML.Launch.Declared
public import AlgorithmLib.Surface.Drv
meta import AlgorithmLib.Surface.Drv
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
  # From the host program's device writes to the pipeline

  A program in the driver's standard layout (`Drv.App`) states the device
  writes it makes as records, by definition (`appLaunch`): a launch's kernel,
  grid and parameter array, and the buffers in it. `ML/Launch/Sequence.lean`
  proves what the kernels compute: `Pipeline.run_denote` says a list of
  `StageSpec`s executed in order computes the fold of their `step`s.

  What connects a launch to a `StageSpec` is a fact about the kernels: that the
  module at `off` is the stage's compiled kernel. This file makes it a value: a
  `KernelBinding` list, checked against each record by a decidable function. A
  record whose kernel, parameter array or grid disagrees with the table yields
  `none` rather than a quietly wrong stage.
-/

namespace AlgorithmLib.ML

open AlgorithmLib.IR AlgorithmLib.HProg.Sem

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

      A record states the array's contents (`appBufs`): `near k` for the
      handle in slot `k` of the arena, `far b k` for one at slot `k` of a base
      the program computes — a per-layer weight. Requiring them to match is
      what ties the stage's buffer numbering to the handles the program
      stores.

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

/-- **A device write together with the buffers it is handed.** A vendor call
    carries `none` for a parameter array, and is refused by `stageOf?`. -/
abbrev DeviceOp := LaunchRec × OpBinds

/-- The bind array kernel `j` of a program in the driver's standard layout
    (`Drv.App`) launches with: entry `p` is the pointer in the slot of the
    `p`th allocation it is handed. -/
def appBufs (a : Drv.App) (j : Nat) : List BufDesc :=
  ((a.kernels[j]?).map fun k => k.args.map fun i => BufDesc.near (Int.ofNat (a.bufSlot i))).getD []

/-- **A launch of kernel `j` of a program in the standard layout, handed the
    buffers `bs`**, as a pipeline reads it: its PTX, how many parameters it
    takes, the array of pointers to them, its grid and block, and `bs`. -/
def appLaunchWith (a : Drv.App) (j : Nat) (bs : List BufDesc) : DeviceOp :=
  match a.kernels[j]? with
  | some k =>
      ({ fnName    := "cuLaunchKernel"
         kernelOff := some (Int.ofNat k.ptxOff)
         nBufs     := some (Int.ofNat k.args.length)
         bindOff   := some (Int.ofNat (a.paramsOff j))
         gridX     := some (Int.ofNat k.grid.1)
         blockX    := some (Int.ofNat k.block.1) },
       { bufs := some bs })
  | none => ({ fnName := "", kernelOff := none, nBufs := none, bindOff := none, gridX := none, blockX := none },
             { bufs := none })

/-- **A launch of kernel `j` of a program in the standard layout**, handed the
    allocations its declaration lists. -/
def appLaunch (a : Drv.App) (j : Nat) : DeviceOp := appLaunchWith a j (appBufs a j)

/-- The buffer a parameter value names, for a program in the standard
    layout: a pointer into one of its allocations names that allocation,
    wherever in it it points, and any other value is itself. -/
def wordBuf (a : Drv.App) (x : UInt64) : BufDesc :=
  match devPtrOf x with
  | some (i, _) => if i < a.bufs.length then .near (Int.ofNat (a.bufSlot i)) else .const (Int.ofNat x.toNat)
  | none => .const (Int.ofNat x.toNat)

/-- **A cuBLAS product** of a program in the standard layout, as
    `Drv.App.sgemm` takes it: `(ta, tb, m, n, k, A, B, C)`. -/
abbrev Gemm := Nat × Nat × Nat × Nat × Nat × Nat × Nat × Nat

/-- The words `Drv.App.sgemm` hands cuBLAS for `g`: the program's cuBLAS
    handle, the shape, the addresses of the scalars `1` and `0` in the arena,
    and the allocations with their leading dimensions. -/
def _root_.AlgorithmLib.Drv.App.gemmWords (a : Drv.App) : Gemm → List UInt64
  | (ta, tb, m, n, k, A, B, C) =>
      [drvHandle kBlas 0, UInt64.ofNat ta, UInt64.ofNat tb, UInt64.ofNat m, UInt64.ofNat n, UInt64.ofNat k,
       (regionBase .arena + UInt64.ofNat a.ONE), devPtr A, UInt64.ofNat (if ta = 1 then k else m), devPtr B,
       UInt64.ofNat (if tb = 1 then n else k), (regionBase .arena + UInt64.ofNat a.ZERO), devPtr C, UInt64.ofNat m]

/-- A cuBLAS product as a pipeline reads it: the function, and what each word
    it is handed names. -/
def gemmOp (a : Drv.App) (ws : List UInt64) : DeviceOp :=
  ({ fnName := (Ext.cublas .sgemm).symbol, kernelOff := none, nBufs := none, bindOff := none,
     gridX := none, blockX := none }, { args := ws.map (wordBuf a) })

/-- A captured graph's replay as a pipeline reads it: the function, and what
    the words it is handed (the executable graph and the stream) name. -/
def graphOp (a : Drv.App) (ws : List UInt64) : DeviceOp :=
  ({ fnName := (Ext.cuda .graphLaunch).symbol, kernelOff := none, nBufs := none, bindOff := none,
     gridX := none, blockX := none }, { args := ws.map (wordBuf a) })

/-- **The record of product `g`** of a program in the standard layout. -/
def appGemm (a : Drv.App) (g : Gemm) : DeviceOp := gemmOp a (a.gemmWords g)

/-- A step of a program that launches kernels and calls cuBLAS: kernel `j`'s
    launch, or a product. -/
def appStep (a : Drv.App) : Sum Nat Gemm → DeviceOp
  | .inl j => appLaunch a j
  | .inr g => appGemm a g


/-- The calls a stage can be read from: the driver's kernel launch. -/
def stageLaunchNames : List String := [(Ext.cuda .launchKernel).symbol]

/-- **The stage a launch record names.**

    Five checks, each of which a `Pipeline` claim silently depends on.

    * the slot must be bound — otherwise nothing is known about the PTX that runs;
    * the *bind offset* must match too: same kernel, different buffer-pointer
      array, is a different step, and the table says which;
    * the bind array's **contents** must match, and must have been recovered at
      all — this is what ties the stage's buffer numbering to the handles the
      program stored, rather than to numbers a table author picked;
    * the launched grid must equal the stage's own `grid`, because every theorem
      in `Launch/Sequence.lean` is about `runGrid S.ew S.grid`, so a record launching a
      different number of blocks realises a *different* function;
    * the launch must be on the stream this fragment is being realised at.
      `Pipeline.run` is a sequential fold, and two kernels issued on different
      streams are not ordered with respect to each other — so what makes a
      sequence of launches a pipeline is that they share one stream.  `str` is
      that stream: `none` is the default stream, which is ordered against
      everything, and a fragment whose records disagree realises nothing.

      -/
def stageOf? (T : List KernelBinding) (str : Option Nat) (op : DeviceOp) : Option StageSpec :=
  if op.1.fnName ∉ stageLaunchNames then none
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

/-- **`C = A·B` as a plan step, at a batch of one.**

    The GEMM counterpart of `cublasStep`, and written the same way and for the
    same reason: against the *named law's* own opaque, `cublasSgemmResult`, so
    that `Law.cublasGemmIsSomeReassoc` constrains something a plan mentions
    rather than sitting in the registry.  Output address `a` is read row-major
    as element `(a / n, a % n)`, which is the shape entering — the one place it
    does.

    What the law does not say, and this step does not either: which operand of
    the column-major call plays `A`.  That is the lowering's business
    (`vGemmOf` picks it per operation) and a transposition is not a
    reassociation, so it could not be smuggled in here.

    *Where* it can land is proven: `frame` is discharged, so the step cannot
    touch a buffer the rest of the plan reasons about. -/
noncomputable def cublasGemmStep (aB bB cB : Buf) (m n k : Nat) : DeclaredStep where
  kernel := .cublasSgemmAt1OnStream
  outs  := [cB]
  step  := fun mem b a =>
             if h : b = cB ∧ a < m * n then
               have hn : 0 < n := Nat.pos_of_ne_zero (fun h0 => by
                 rw [h0, Nat.mul_zero] at h; exact Nat.not_lt_zero a h.2)
               cublasSgemmResult m n k (matAt mem aB m k) (matAt mem bB k n)
                 ⟨a / n, Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact h.2)⟩
                 ⟨a % n, Nat.mod_lt _ hn⟩
             else mem b a
  frame := by
    intro mem b hb
    funext a
    rw [dif_neg (fun h => hb (by simp [h.1]))]

/-- **What the vendor GEMM computes, under the law that names it.**

    Output element `(i, j)` sums that element's own `k` products, each once, in
    some association.  Weaker than `cublasStep_isMatvec` in exactly the way the
    two laws differ — there is a tree here rather than a fold — and that is the
    whole of what a batch-of-one contraction may be assumed to be. -/
theorem cublasGemmStep_isSomeReassoc (hg : CuBlasGemmIsSomeReassoc)
    (aB bB cB : Buf) (m n k : Nat) (mem : Buf → Nat → Float32)
    (i j : Nat) (hi : i < m) (hj : j < n) :
    ∃ t : SumTree, t.indices.Perm (List.range k) ∧
      (cublasGemmStep aB bB cB m n k).step mem cB (i * n + j)
        = t.eval (gemmTerm (matAt mem aB m k) (matAt mem bB k n) ⟨i, hi⟩ ⟨j, hj⟩) := by
  have ha : i * n + j < m * n := by
    calc i * n + j < i * n + n := Nat.add_lt_add_left hj _
      _ = (i + 1) * n := by rw [Nat.succ_mul]
      _ ≤ m * n := Nat.mul_le_mul_right n hi
  obtain ⟨t, hperm, ht⟩ :=
    hg m n k (matAt mem aB m k) (matAt mem bB k n) ⟨i, hi⟩ ⟨j, hj⟩
  refine ⟨t, hperm, ?_⟩
  show (if h : cB = cB ∧ i * n + j < m * n then
          cublasSgemmResult m n k (matAt mem aB m k) (matAt mem bB k n)
            ⟨(i * n + j) / n, _⟩ ⟨(i * n + j) % n, _⟩
        else mem cB (i * n + j)) = _
  rw [dif_pos (And.intro rfl ha)]
  have hd : (i * n + j) / n = i := by
    rw [Nat.mul_comm, Nat.mul_add_div (by omega), Nat.div_eq_of_lt hj, Nat.add_zero]
  have hm : (i * n + j) % n = j := by
    rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hj]
  rw [show (⟨(i * n + j) / n, by omega⟩ : Fin m) = ⟨i, hi⟩ from Fin.ext hd,
      show (⟨(i * n + j) % n, by omega⟩ : Fin n) = ⟨j, hj⟩ from Fin.ext hm]
  exact ht

-- ---------------------------------------------------------------------------
-- Realising a whole device-write sequence
-- ---------------------------------------------------------------------------

/-- Which primitive name maps to which declared step.  The declared counterpart
    of `KernelBinding`: a launch record names a PTX slot, a device-write record
    names a primitive, and neither says by itself what it computes. -/
structure DeclaredBinding where
  name : String
  /-- **Which call this is.**  A vendor primitive has no PTX slot, so name alone
      does not identify a call site: a transformer layer's query, key and value
      projections can be three calls of one shape.  Matching on the recovered
      argument descriptors — which slot each buffer handle came from — is what
      tells them apart. -/
  args : List BufDesc
  decl : DeclaredStep

/-- The calls a declared step can be read from: cuBLAS's products, a copy from
    the host, and a captured graph's replay. -/
def writerNames : List String :=
  [Ext.cublas .sgemv, Ext.cublas .sgemm, Ext.cublas .sgemmStridedBatched, Ext.cublas .sgemmBatched,
   Ext.cublas .gemmEx, Ext.cublas .gemmStridedBatchedEx, Ext.cuda .memcpyHtoD, Ext.cuda .graphLaunch].map Ext.symbol

def declaredOf? : List DeclaredBinding → String → List BufDesc → Option DeclaredStep
  | [],                   _,  _  => none
  | ⟨dNm, dArgs, dD⟩ :: ds, nm, ar =>
      -- Destructured for the reason given at `bindingOf?`: the record's last
      -- field is a `DeclaredStep`, and projecting past it during a reduction
      -- costs proportionally to it.
      --
      -- `args` first, for the same reason `bindingOf?` tests `bindOff` first:
      -- every declared entry in a transformer layer is the same GEMV,
      -- so the name never discriminates and testing it first means each of the
      -- nine entries is rejected only after comparing a nine-element argument
      -- list.  The arguments are what tell `Wq` from `Wk` from `Wv`.
      if dArgs = ar ∧ dNm = nm then some dD else declaredOf? ds nm ar

/-- **The plan step a device-write record realises.**

    A launch record on this fragment's stream goes through the kernel table and
    yields a proven stage; anything named in `writerNames` goes through
    the declared table and yields an assumed one.  Anything else — a launch on
    another stream, an unbound slot, a grid disagreeing with the stage's own —
    yields `none`, so a plan claim is simply unavailable rather than quietly
    wrong. -/
def stepOf? (T : List KernelBinding) (D : List DeclaredBinding)
    (str : Option Nat) (op : DeviceOp) : Option PStep :=
  if op.1.fnName ∈ stageLaunchNames then
    (stageOf? T str op).map PStep.proven
  else if op.1.fnName ∈ writerNames then
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

    Not a convenience.  A table for a whole step is often a layer's table plus
    a few more entries, and without these the same device writes are resolved
    twice, each resolution a linear scan of the table for every op.  With these
    the second is a rewrite.

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
  by_cases hn : op.1.fnName ∉ stageLaunchNames
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
  have h0 : (if op.1.fnName ∈ stageLaunchNames then
               (stageOf? T str op).map PStep.proven
             else if op.1.fnName ∈ writerNames then
               (declaredOf? D op.1.fnName op.2.args).map PStep.declared
             else none) = some t := h
  show (if op.1.fnName ∈ stageLaunchNames then
          (stageOf? (T ++ T') str op).map PStep.proven
        else if op.1.fnName ∈ writerNames then
          (declaredOf? (D ++ D') op.1.fnName op.2.args).map PStep.declared
        else none) = some t
  by_cases hl : op.1.fnName ∈ stageLaunchNames
  · rw [if_pos hl] at h0 ⊢
    cases hs : stageOf? T str op with
    | none => rw [hs] at h0; exact Option.noConfusion h0
    | some S => rw [stageOf?_append T T' str op S hs]; rw [hs] at h0; exact h0
  rw [if_neg hl] at h0 ⊢
  by_cases hw : op.1.fnName ∈ writerNames
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

end AlgorithmLib.ML
