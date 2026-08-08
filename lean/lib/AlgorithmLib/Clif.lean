import AlgorithmLib.IR

/-!
  # A model of the host program

  `IR.lean` builds CLIF and carries no theorems, so the sequence of kernel
  launches — the thing that decides *which* proven kernel runs *when* — was
  outside every statement in the development.

  It was never outside the *data*, though. `Inst` is an inductive and `IRState`
  holds the blocks, so a built function is already a value; what was missing was
  a semantics and a way to read the launch structure back out.

  This file supplies both, for the fragment the generators actually emit:

  * `Modellable` — a decidable gate. `rawInst` embeds an unparsed assembly
    string and cannot be given a meaning; a program containing one is rejected
    rather than silently half-modelled. No generator in the repo emits one.
  * `evalPure` — an SSA environment semantics for the integer fragment that
    computes launch arguments (`iconst`, `iadd`, `imul`, `ishl`, …). Enough to
    say *which* PTX slot and *what* geometry a launch names.
  * `launchesOf` — the launch sequence, in program order, as a list.

  What stays trusted is named at the bottom: the CLIF→machine backend, and the
  contract of each FFI primitive. Those are the host-side counterparts of the
  PTX opcode table.
-/

namespace AlgorithmLib.Clif

open AlgorithmLib.IR

-- ---------------------------------------------------------------------------
-- The modellable fragment
-- ---------------------------------------------------------------------------

/-- Instructions this model gives a meaning to.

    `rawInst` is the only exclusion, and it is a genuine one: its payload is an
    opaque string handed to the assembler, so no semantics can be assigned
    without parsing it. Gating on it means a program that reaches for the escape
    hatch fails the check rather than being modelled as if it had not. -/
def Inst.ModellableB : Inst → Bool
  | .rawInst _ => false
  | _          => true

/-- …lifted to a whole function. -/
def blocksModellableB (bs : List BlockData) : Bool :=
  bs.all (fun b => b.insts.all Inst.ModellableB)

-- ---------------------------------------------------------------------------
-- The pure integer fragment: enough to evaluate a launch's arguments
-- ---------------------------------------------------------------------------

/-- **A small integer algebra over unresolved runtime values.**

    Roots are named by the SSA value they came from, so two different loads are
    two different roots and an expression cannot accidentally identify them.

    Not a general expression language.  It has exactly the operations the host's
    loop-bound idiom uses — shift right, shift left, subtract, add a constant —
    because each one is a place where `Clif`'s reading of a CLIF instruction has
    to agree with Cranelift's, and that agreement is a trusted seam.  Six
    constructors is six seams; a general language would be an open-ended one. -/
inductive DExp where
  | root : Nat → DExp
  | lit  : Int → DExp
  | add  : DExp → DExp → DExp
  | sub  : DExp → DExp → DExp
  /-- `x >>> k`, i.e. `x / 2^k` on non-negative `x`. -/
  | shr  : DExp → Nat → DExp
  /-- `x <<< k`, i.e. `x * 2^k`. -/
  | shl  : DExp → Nat → DExp
  deriving Repr, DecidableEq, BEq

/-- What a `DExp` denotes, given a valuation of the roots.  The *only* place
    this development says what `ushr`/`ishl`/`isub` compute; `Cranelift
    implements this` is the seam, and it is one definition wide. -/
def DExp.eval (rho : Nat → Int) : DExp → Int
  | .root v  => rho v
  | .lit k   => k
  | .add a b => DExp.eval rho a + DExp.eval rho b
  | .sub a b => DExp.eval rho a - DExp.eval rho b
  | .shr a k => DExp.eval rho a / 2 ^ k
  | .shl a k => DExp.eval rho a * 2 ^ k

/-- What this model knows about an SSA value.

    `offset` is the case that matters: every launch argument naming a PTX slot
    or a bind table is `ptr + k` for a runtime base pointer `ptr` and a
    compile-time `k`. Tracking only `const` would lose the slot — which is the
    field that says *which kernel* is being launched — so the base is carried
    symbolically and the offset exactly. -/
inductive SymVal where
  | const   : Int → SymVal
  | offset  : Val → Int → SymVal
  /-- **The value loaded from `ptr + k`.**

      Buffer handles are not constants: a generator writes `slotWq.load ptr`, so
      the argument naming *which matrix* a vendor call multiplies reaches the
      call as a loaded value.  Without this case it is `unknown`, and two calls
      of the same shape — Qwen2's `Wq`, `Wk` and `Wv` are all 896×896 — are
      indistinguishable in the trace.  Carrying the slot the value came from is
      what makes each call site identifiable.

      The **base** rides along as well.  Qwen2's per-layer weights are reached
      through `slotBaseA = ptr + (LAYER_BUFS_BASE + layerIdx·STRIDE)`, whose
      layer index is a runtime load — so the base is unresolved and the weight
      handle reads as offset `0` *of that base*, while the hidden state reads as
      offset `72` of the descriptor pointer.  Recording only the offset makes
      those two the same descriptor, which would let a bind-array check pass
      against entirely different buffers.  `descOf` still discards the base, so
      `ArgDesc` and every vendor-call record are unchanged. -/
  | slot    : Val → Int → SymVal
  /-- **A value the model can name but not evaluate.**

      `const` covers compile-time arithmetic and `offset` covers a pointer plus
      a displacement; between them they miss the case a data-dependent kernel
      actually needs — a quantity *derived* from a runtime input.  Qwen2's
      softmax reads four meta slots holding `seq`, `seq/32`, `(seq/32)*32` and
      `seq%32`, and `seq` is a load, so before this every one of them was
      `unknown` and "the host publishes the loop bounds the kernel assumes" was
      not a statement this model could make.

      Deliberately a *small* algebra (`DExp`): every constructor is a claim
      that Cranelift's instruction of the same name means that operation, so
      adding one widens the trusted surface by exactly one instruction. -/
  | derived : DExp → SymVal
  | unknown : SymVal
  deriving Repr, BEq

/-- The constant offset from a base, if that is what this value is. -/
def SymVal.offsetOf? : SymVal → Option Int
  | .offset _ k => some k
  | .const k    => some k
  | _           => none

/-- Which slot this value was loaded from, if it was. -/
def SymVal.slotOf? : SymVal → Option Int
  | .slot _ k => some k
  | _         => none

/-- **SSA environment.**

    A strict association list, *not* a chain of closures.  That distinction is
    not cosmetic: with `Env := Val → SymVal` and
    `set e v x = fun w => if w.id = v.id then x else e w`, each stored value is
    re-derived on every lookup that reaches it, and since a value's expression
    contains further lookups the cost doubles with depth — measured at
    270 / 451 / 807 / 1554 ms across successive three-instruction groups of a
    92-instruction function, i.e. `2 ^ (n/3)`, on a scan that should take
    microseconds.  Storing the value instead of a recipe for it makes lookup a
    plain list walk over data that was computed exactly once.

    Values not bound here are runtime inputs — block parameters and the like —
    and read as `unknown`. -/
structure Env where
  bindings : List (Nat × SymVal)

def lookupBindings : List (Nat × SymVal) → Val → SymVal
  | [],          _ => .unknown
  | (i, x) :: r, v => if v.id = i then x else lookupBindings r v

def Env.get (e : Env) (v : Val) : SymVal := lookupBindings e.bindings v

instance : CoeFun Env (fun _ => Val → SymVal) := ⟨Env.get⟩

/-- Nothing bound: every value reads as a runtime input. -/
def Env.empty : Env := ⟨[]⟩

def Env.set (e : Env) (v : Val) (x : SymVal) : Env := ⟨(v.id, x) :: e.bindings⟩

/-- The characterisation everything else goes through, so no proof depends on
    the representation. -/
theorem Env.set_apply (e : Env) (v w : Val) (x : SymVal) :
    (e.set v x) w = if w.id = v.id then x else e w := rfl

theorem Env.set_eq (e : Env) (v w : Val) (x : SymVal) (h : w.id = v.id) :
    (e.set v x) w = x := by rw [Env.set_apply, if_pos h]

theorem Env.set_ne (e : Env) (v w : Val) (x : SymVal) (h : ¬ (w.id = v.id)) :
    (e.set v x) w = e w := by rw [Env.set_apply, if_neg h]

/-- **Every value's view as a `DExp`.**  Total: a value the model cannot
    resolve becomes a root named by its own SSA id, which is exactly what makes
    `seq` and `seq >>> 5` relatable without knowing what `seq` is. -/
def dOf (e : Env) (v : Val) : DExp :=
  match e v with
  | .const k    => .lit k
  | .offset p k => .add (.root p.id) (.lit k)
  | .derived d  => d
  | _           => .root v.id

/-- The `DExp` a value denotes when the model can name it without inventing a
    root — everything except an unresolved load.  `dOf` invents one; this says
    when it did not have to, which is what makes an expression stated about a
    caller's value equal to the one the fragment builds. -/
def SymVal.toD? : SymVal → Option DExp
  | .const k    => some (.lit k)
  | .offset p k => some (.add (.root p.id) (.lit k))
  | .derived d  => some d
  | _           => none

theorem dOf_of_toD? (e : Env) (v : Val) (d : DExp) (h : (e v).toD? = some d) :
    dOf e v = d := by
  cases hv : e v <;> rw [hv] at h <;> simp_all [dOf, SymVal.toD?]

/-- `a + b`, symbolically: constants fold, a base absorbs a constant, anything
    else is unknown.  A value that is merely *unbound* still contributes its own
    identity as a base, which is what turns `ptr + ptxOff` into `offset ptr k`
    rather than `unknown`. -/
def addSym (e : Env) (a b : Val) : SymVal :=
  match e a, e b with
  | .const x,     .const y     => .const (x + y)
  | .offset p k,  .const y     => .offset p (k + y)
  | .const x,     .offset p k  => .offset p (x + k)
  | .unknown,     .const y     => .offset a y
  | .const x,     .unknown     => .offset b x
  | _, _                       => .unknown

def mulSym (e : Env) (a b : Val) : SymVal :=
  match e a, e b with
  | .const x, .const y => .const (x * y)
  | _, _               => .unknown

/-- Step the environment across one instruction.

    Deliberately conservative: anything not in the tracked fragment binds its
    destination to `unknown`, so a value this model reports as a constant really
    is one. -/
def stepPure (e : Env) : Inst → Env
  | .iconst d _ v     => e.set d (.const v)
  | .iadd d a b       => e.set d (addSym e a b)
  -- the three that carry the host's loop-bound idiom.  Constant folding first,
  -- exactly as before; what is new is that a *runtime* operand now yields a
  -- named expression instead of `unknown`.
  | .isub d a b       => e.set d (match e a, e b with
                                  | .const x, .const y => .const (x - y)
                                  | .offset p k, .const y => .offset p (k - y)
                                  | _, _ => .derived (.sub (dOf e a) (dOf e b)))
  | .imul d a b       => e.set d (mulSym e a b)
  | .ineg d a         => e.set d (match e a with
                                  | .const x => .const (-x)
                                  | _ => .unknown)
  | .ishl d a b       => e.set d (match e a, e b with
                                  | .const x, .const y => .const (x * 2 ^ y.toNat)
                                  | _, .const y => .derived (.shl (dOf e a) y.toNat)
                                  | _, _ => .unknown)
  | .ushr d a b       => e.set d (match e a, e b with
                                  | .const x, .const y => .const (x / 2 ^ y.toNat)
                                  | _, .const y => .derived (.shr (dOf e a) y.toNat)
                                  | _, _ => .unknown)
  -- **Load carries provenance.**  A value read from `ptr + k` is recorded as
  -- coming from slot `k`, which is what makes a buffer handle identifiable.
  | .load d _ a       => e.set d (match e a with
                                  | .offset p k => .slot p k
                                  | _ => .unknown)
  | .ireduce32 d a    => e.set d (e a)
  | .uextend64 d a    => e.set d (e a)
  | .sextend64 d a    => e.set d (e a)
  -- everything that writes a destination we do not track
  | .udiv d _ _ | .band d _ _ | .bandNot d _ _ | .bor d _ _ | .bxor d _ _
  | .icmp d _ _ _ | .select d _ _ _
  | .fconst d _ _ | .fadd d _ _ | .fsub d _ _ | .fmul d _ _ | .fmax d _ _
  | .fmin d _ _ | .fpromote d _ | .splat d _ _ | .extractlane d _ _
  | .fneg d _ | .fcvtFromSint d _ _ | .fcvtToUint d _ _
  | .fcmp d _ _ _ | .bitcast d _ _ | .ctz d _ | .popcnt d _ | .vhighBits d _ =>
      e.set d .unknown
  | .call (some d) _ _ => e.set d .unknown
  | _ => e

/-- **What an instruction writes**, independently of what it computes.

    Stated separately from `stepPure` on purpose: the two must agree about
    *which* instructions have a destination, and `stepPure_untracked` below is
    what forces them to.  Without it an instruction can write a value the model
    does not notice, leaving a **stale** binding that a later launch argument
    reads back as a compile-time constant it no longer is. -/
def Inst.destOf? : Inst → Option Val
  | .iconst d _ _ | .ineg d _ | .ireduce32 d _ | .uextend64 d _ | .sextend64 d _
  | .fpromote d _ | .fneg d _ | .ctz d _ | .popcnt d _ | .vhighBits d _ => some d
  | .iadd d _ _ | .isub d _ _ | .imul d _ _ | .udiv d _ _ | .ishl d _ _
  | .ushr d _ _ | .band d _ _ | .bandNot d _ _ | .bor d _ _ | .bxor d _ _
  | .fadd d _ _ | .fsub d _ _ | .fmul d _ _ | .fmax d _ _ | .fmin d _ _ => some d
  | .load d _ _ | .fconst d _ _ | .splat d _ _ | .extractlane d _ _
  | .fcvtFromSint d _ _ | .fcvtToUint d _ _ | .bitcast d _ _ => some d
  | .icmp d _ _ _ | .select d _ _ _ | .fcmp d _ _ _ => some d
  | .call d _ _ => d
  | _ => none

/-- The fragment whose *value* the model computes rather than discards. -/
def Inst.TrackedB : Inst → Bool
  | .iconst _ _ _ | .iadd _ _ _ | .isub _ _ _ | .imul _ _ _ | .ineg _ _
  | .ishl _ _ _ | .ushr _ _ _ | .ireduce32 _ _ | .uextend64 _ _ | .sextend64 _ _
  | .load _ _ _ =>
      true
  | _ => false

/-- **An instruction changes nothing it does not write.** -/
theorem stepPure_frame : ∀ (i : Inst) (e : Env) (w : Val),
    (∀ d, Inst.destOf? i = some d → w.id ≠ d.id) → stepPure e i w = e w := by
  intro i e w h
  cases i
  case call d fn args => cases d <;> simp_all [stepPure, Env.set_apply, Inst.destOf?]
  all_goals simp_all [stepPure, Env.set_apply, Inst.destOf?]

/-- **Nothing stale survives.**  Every instruction outside the tracked
    arithmetic fragment leaves its destination `unknown` — so a value this model
    reports as a constant really was computed as one, and was not simply missed.

    This is the theorem that catches an omission in `stepPure`: add a
    constructor with a destination, forget to give it a case, and this proof
    fails because the destination keeps its previous binding. -/
theorem stepPure_untracked : ∀ (i : Inst) (d : Val) (e : Env),
    Inst.destOf? i = some d → Inst.TrackedB i = false → stepPure e i d = SymVal.unknown := by
  intro i d e hd ht
  cases i
  case call d' fn args =>
      cases d' <;> simp_all [stepPure, Env.set_apply, Inst.destOf?, Inst.TrackedB]
  all_goals simp_all [stepPure, Env.set_apply, Inst.destOf?, Inst.TrackedB]


def evalPure (e : Env) : List Inst → Env
  | []      => e
  | i :: is => evalPure (stepPure e i) is

/-- The frame property, lifted to a straight-line run. -/
theorem evalPure_frame : ∀ (is : List Inst) (e : Env) (w : Val),
    (∀ i ∈ is, ∀ d, Inst.destOf? i = some d → w.id ≠ d.id) → evalPure e is w = e w := by
  intro is
  induction is with
  | nil => intro _ _ _; rfl
  | cons i is ih =>
      intro e w h
      show evalPure (stepPure e i) is w = e w
      rw [ih (stepPure e i) w (fun j hj => h j (List.mem_cons_of_mem i hj)),
          stepPure_frame i e w (h i (List.mem_cons_self ..))]

-- ---------------------------------------------------------------------------
-- Reading the launch sequence back out
-- ---------------------------------------------------------------------------

/-- **What one argument of a call is**, as far as this model can tell.

    `slot k` is the case that identifies a buffer: the handle was loaded from
    `ptr + k`, and `k` is unique per buffer.  Without it, two vendor calls of the
    same shape are the same record. -/
inductive ArgDesc where
  | const  : Int → ArgDesc
  | slot   : Int → ArgDesc
  | opaque : ArgDesc
  deriving Repr, DecidableEq, BEq

def descOf : SymVal → ArgDesc
  | .const k   => .const k
  | .slot _ k  => .slot k
  | _          => .opaque

/-- One kernel launch, as recovered from the program: which PTX slot, how many
    buffers, where the bind table sits, and the geometry. `none` in a field
    means the argument was not a compile-time constant. -/
structure LaunchRec where
  fnName   : String
  kernelOff : Option Int
  nBufs    : Option Int
  bindOff  : Option Int
  gridX    : Option Int
  blockX   : Option Int
  /-- For a **declared** device write, what each argument is.  A modelled launch
      leaves this empty — its identity is the PTX slot, already recovered above.
      A vendor call has no slot, so its identity is its arguments. -/
  args     : List ArgDesc := []
  /-- **Which stream the launch was issued on**, as the identity of the value
      passed rather than its number.

      A stream id is created at runtime, so there is no constant to recover:
      two distinct streams both evaluate to an unknown and would be recorded
      identically.  What *is* visible is which SSA value each launch was handed,
      and within one function that is exactly the question — launches carrying
      the same value are on the same stream, and a second stream is a second
      definition.

      `none` for a launch on the default stream, which is ordered against
      everything, and for a device write that is not a launch. -/
  stream   : Option Nat := none
  deriving Repr, DecidableEq, BEq

/-- The declared name of a function reference. -/
def fnNameOf (fns : List FnDecl) (r : FnRef) : Option String :=
  (fns.find? (fun d => d.ref.id = r.id)).map FnDecl.name

/-- **Primitives that write device memory without being a modelled launch.**

    This list is the one that matters, and it was missing.  `launchNames` covers
    the four `cl_cuda_launch*` entry points, so `launchesOf` sees those and
    nothing else — but `cl_cublas_sgemv` writes its output buffer, and
    `cl_cuda_graph_launch` replays a whole captured graph.  A sequence built
    from launch records alone would therefore claim a memory transformation
    that *omits* those writes.  For Qwen2 that omission is 99.9% of the
    arithmetic.

    Naming them here turns the omission into a refusal: `Inst.TameB` rejects
    any fragment containing one, so a driver that calls cuBLAS cannot be given
    the hypothesis that the downstream pipeline theorems require.  The gap
    becomes a failed proof rather than a silent overclaim.

    Removing an entry is how a primitive graduates — once it has a `StageSpec`
    (or its caller stops using it), it either becomes a modelled launch or the
    call disappears. -/
def deviceWriterNames : List String :=
  [ -- vendor BLAS: writes `y`, no kernel we compiled, no fold order specified
    "cl_cublas_sgemv", "cl_cublas_sgemv_on_stream",
    "cl_cublas_sgemm_strided_batched", "cl_cublas_sgemm_strided_batched_on_stream",
    -- replays a captured graph: arbitrarily many kernels, none of them recorded
    "cl_cuda_graph_launch",
    -- host→device writes
    "cl_cuda_upload_ptr", "cl_cuda_upload_ptr_offset",
    "cl_cuda_upload_ptr_async", "cl_cuda_upload_ptr_offset_async",
    -- the wgpu path's equivalents
    "cl_gpu_dispatch", "cl_gpu_upload" ]

/-- Names that launch a kernel.  Listed rather than pattern-matched on a prefix
    so that adding a launch variant to `FFI.lean` and forgetting it here is a
    visible omission, not a silently shorter sequence. -/
def launchNames : List String :=
  ["cl_cuda_launch", "cl_cuda_launch_named",
   "cl_cuda_launch_on_stream", "cl_cuda_launch_named_on_stream"]

/-- **What one instruction contributes to the launch sequence.**

    Split out from the walk so that the case analysis — is this a call, is it
    declared, is it a launch — lives in exactly one place. The recursion in
    `scanBlock` is then structurally trivial, which is what makes the
    correctness theorems below inductions rather than case explosions. -/
def launchAt (fns : List FnDecl) (e : Env) : Inst → Option LaunchRec
  | .call _ fr args =>
      match fnNameOf fns fr with
      | some nm =>
          if nm ∈ launchNames then
            let arg (k : Nat) : Option Int :=
              (args[k]?).bind (fun v => (e v).offsetOf?)
            -- The stream is argument 10 of the `_on_stream` variants and absent
            -- from the others, so the plain launch records `none` without a
            -- case on the name.
            some { fnName := nm, kernelOff := arg 1, nBufs := arg 2,
                   bindOff := arg 3, gridX := arg 4, blockX := arg 7,
                   stream := (args[10]?).map Val.id }
          else if nm ∈ deviceWriterNames then
            -- A device write this model does not interpret.  Recorded, with
            -- every field `none`: the *position* in the sequence is what a
            -- composition needs, and claiming to have recovered a slot or a
            -- grid for a vendor call would be an invention.
            some { fnName := nm, kernelOff := none, nBufs := none,
                   bindOff := none, gridX := none, blockX := none,
                   args := args.map (fun v => descOf (e v)) }
          else none
      | none => none
  | _ => none

/-- Is this a call at all?  Everything else is invisible to the extractor. -/
def Inst.isCallB : Inst → Bool
  | .call _ _ _ => true
  | _           => false

/-- Walk one block, threading the environment, collecting launches in order. -/
def scanBlock (fns : List FnDecl) : Env → List Inst → Env × List LaunchRec
  | e, []      => (e, [])
  | e, i :: is =>
      let rest := scanBlock fns (stepPure e i) is
      (rest.1, (launchAt fns e i).toList ++ rest.2)

/-- **The launch sequence a built function performs**, in program order.

    This is the value that did not exist. A `Pipeline` in `ML/Compose.lean` says
    which stages run in which order; this says which stages the *host program*
    actually launches. Relating the two is what a whole-model theorem needs, and
    it is now a statement about two lists rather than about an emission order
    nothing could name. -/
def launchesOf (s : IRState) : List LaunchRec :=
  (s.allBlocks.foldl (fun (acc : Env × List LaunchRec) b =>
      let r := scanBlock s.fns acc.1 b.insts
      (r.1, acc.2 ++ r.2)) (Env.empty, [])).2

/-- How many kernels the program launches. -/
def launchCount (s : IRState) : Nat := (launchesOf s).length


-- ---------------------------------------------------------------------------
-- Structural facts
-- ---------------------------------------------------------------------------

/-- **Is this instruction a launch?**  The predicate `scanBlock` branches on,
    named so the extraction can be stated against a specification rather than
    against itself. -/
def isLaunchCallB (fns : List FnDecl) : Inst → Bool
  | .call _ fr _ =>
      match fnNameOf fns fr with
      | some nm => decide (nm ∈ launchNames) || decide (nm ∈ deviceWriterNames)
      | none    => false
  | _ => false

/-- Every name the extractor records — modelled launches and declared device
    writes alike.  `scanBlock_fnName` is stated against this. -/
def deviceOpNames : List String := launchNames ++ deviceWriterNames

/-- The decision is environment-independent: the record's *contents* depend on
    the environment, the some/none choice does not. -/
theorem launchAt_isSome (fns : List FnDecl) (e : Env) (i : Inst) :
    (launchAt fns e i).isSome = isLaunchCallB fns i := by
  cases i with
  | call d fr args =>
      simp only [launchAt, isLaunchCallB]
      cases fnNameOf fns fr with
      | none => rfl
      | some nm =>
          by_cases hm : nm ∈ launchNames
          · simp [hm]
          · by_cases hw : nm ∈ deviceWriterNames <;> simp [hm, hw]
  | _ => rfl

/-- How many launches a straight-line block *should* yield — independent of the
    environment, which is what lets the correspondence be proven without
    evaluating any addresses. -/
def launchCallCount (fns : List FnDecl) : List Inst → Nat
  | []      => 0
  | i :: is => (if isLaunchCallB fns i then 1 else 0) + launchCallCount fns is

/-- Does this instruction call one of them? -/
def isDeviceWriterB (fns : List FnDecl) : Inst → Bool
  | .call _ fr _ =>
      match fnNameOf fns fr with
      | some nm => decide (nm ∈ deviceWriterNames)
      | none    => false
  | _ => false

/-- **The device writers a block performs that this model does not interpret.**

    The counterpart to `scanBlock`, and the one that was missing.  `launchesOf`
    answers "which kernels run"; this answers "what else wrote device memory
    while they did".  Reported as names so a generator's check can say *which*
    primitive is unaccounted for rather than only that something is. -/
def deviceWritesIn (fns : List FnDecl) : List Inst → List String
  | []      => []
  | i :: is =>
      (match i with
       | .call _ fr _ =>
           match fnNameOf fns fr with
           | some nm => if nm ∈ deviceWriterNames then [nm] else []
           | none    => []
       | _ => []) ++ deviceWritesIn fns is

/-- …over a whole function. -/
def deviceWritesOf (s : IRState) : List String :=
  s.allBlocks.flatMap (fun b => deviceWritesIn s.fns b.insts)

-- ---------------------------------------------------------------------------
-- Counted loops, recovered
-- ---------------------------------------------------------------------------

/-! `IR.forLoop` emits a fixed four-part shape: the current block jumps to a
    header with a zero counter; the header compares the counter against a bound
    and branches to the body or the exit; the body runs, increments by one, and
    jumps back.

    Recovering that shape is what lets a *static* scan say how many times a body
    runs — which is the one thing `launchesOf` could not do, and the reason a
    layer's kernels appear once in the instruction stream instead of
    twenty-four times.  Everything here is decidable, so it is checked against
    the emitted blocks rather than read off the generator's source. -/

/-- **A counted loop, as recovered from three emitted blocks.**  `bodyBlk` is
    where the body's instructions live, so a caller can scan them. -/
structure LoopRec where
  hdrBlk  : Nat
  bodyBlk : Nat
  exitBlk : Nat
  /-- The bound, when the model resolved it to a compile-time constant. -/
  trip    : Nat
  deriving Repr, DecidableEq, BEq

/-- The header's two instructions, and the bound they compare against, resolved
    in the environment reaching this block. -/
def loopHdrOf? (e : Env) (b : BlockData) : Option (Nat × Nat × Nat) :=
  match b.params, b.insts with
  | [(p, _)], [Inst.icmp c .ult i lim, Inst.brif c' bdy [j] ex []] =>
      if c.id = c'.id ∧ i.id = p.id ∧ j.id = p.id then
        match e lim with
        | .const k => some (bdy.id, ex.id, k.toNat)
        | _        => none
      else none
  | _, _ => none

/-- The body must end by incrementing its parameter and jumping back — which is
    what makes the count a *trip* count rather than a guess. -/
def loopBodyOkB (hdr : Nat) (b : BlockData) : Bool :=
  match b.params, b.insts.reverse with
  | [(p, _)], (Inst.jump t [inc]) :: (Inst.iadd d q o) :: rest =>
      t.id == hdr && d.id == inc.id && q.id == p.id
        && (rest.reverse.any (fun i => match i with
              | Inst.iconst dd _ 1 => dd.id == o.id
              | _ => false)
            || (b.insts.any (fun i => match i with
                  | Inst.iconst dd _ 1 => dd.id == o.id
                  | _ => false)))
  | _, _ => false

/-- **The accumulator-carrying header.**  `forLoopAcc` threads a carry alongside
    the induction variable, so its header takes two parameters and its `brif`
    passes the carry to both successors.  `loopHdrOf?` requires exactly one
    parameter, which is why every `forLoopAcc` in the codebase was invisible to
    the loop scan — including the one that performs LZ4's twenty launches.

    Same shape otherwise: the bound must resolve to a compile-time constant, so
    the trip count is read out of the program rather than assumed. -/
def loopHdrAccOf? (e : Env) (b : BlockData) : Option (Nat × Nat × Nat) :=
  match b.params, b.insts with
  | [(p, _), (a, _)], [Inst.icmp c .ult i lim, Inst.brif c' bdy [j, a1] ex [a2]] =>
      if c.id = c'.id ∧ i.id = p.id ∧ j.id = p.id ∧ a1.id = a.id ∧ a2.id = a.id then
        match e lim with
        | .const k => some (bdy.id, ex.id, k.toNat)
        | _        => none
      else none
  | _, _ => none

/-- `loopBodyOkB` for the accumulator form: the body ends by incrementing its
    induction parameter and jumping back with the new carry. -/
def loopBodyAccOkB (hdr : Nat) (b : BlockData) : Bool :=
  match b.params, b.insts.reverse with
  | [(p, _), _], (Inst.jump t [inc, _]) :: (Inst.iadd d q o) :: _ =>
      t.id == hdr && d.id == inc.id && q.id == p.id
        && b.insts.any (fun i => match i with
              | Inst.iconst dd _ 1 => dd.id == o.id
              | _ => false)
  | _, _ => false

/-- **The counted loops a built function contains**, in block order.

    The environment each header is resolved in is `evalPure` over every block
    before it, which is exactly the environment `launchesOf` threads — so the
    bound is resolved the same way a launch's PTX slot is. -/
def loopsOf (s : IRState) : List LoopRec :=
  (s.allBlocks.foldl (fun (acc : Env × List LoopRec) b =>
      let e := acc.1
      let acc2 :=
        match loopHdrOf? e b with
        | some (bdy, ex, k) =>
            if s.allBlocks.any (fun c => c.ref.id == bdy && loopBodyOkB b.ref.id c)
            then acc.2 ++ [⟨b.ref.id, bdy, ex, k⟩] else acc.2
        | none =>
            match loopHdrAccOf? e b with
            | some (bdy, ex, k) =>
                if s.allBlocks.any (fun c => c.ref.id == bdy && loopBodyAccOkB b.ref.id c)
                then acc.2 ++ [⟨b.ref.id, bdy, ex, k⟩] else acc.2
            | none => acc.2
      (evalPure e b.insts, acc2)) (Env.empty, [])).2

/-- The instructions of a named block, if the function has one. -/
def blockInsts? (s : IRState) (n : Nat) : Option (List Inst) :=
  (s.allBlocks.find? (fun b => b.ref.id == n)).map BlockData.insts

/-- The colocated calls a straight line performs, in order — what a loop body
    dispatches to. -/
def callsIn (fns : List FnDecl) (is : List Inst) : List String :=
  is.filterMap (fun i => match i with
    | .call _ fr _ => fnNameOf fns fr
    | _            => none)

/-- …over a whole function. -/
def callsOf (s : IRState) : List String :=
  s.allBlocks.flatMap (fun b => callsIn s.fns b.insts)

/-- **A program whose every device write is a modelled launch.**

    The decidable statement of "the launch sequence is the whole story".  A
    generator that satisfies this has nothing writing device memory behind
    `launchesOf`'s back; one that does not is exactly as far from a pipeline
    claim as this list is long. -/
def launchesAreEverythingB (s : IRState) : Bool := (deviceWritesOf s).isEmpty


/-- **A fragment that is what it claims to be**: it launches nothing, it writes
    no device memory behind the model's back, and it does not rebind `ptr`.

    Decidable, so a host program's launch-free filler is *checked* against this
    rather than trusted.  Drop the first conjunct and a `prim` could hide a
    launch and shorten the declared sequence; drop the second and it could
    perform a cuBLAS matvec the pipeline knows nothing about; drop the third and
    it could clobber the base pointer, making every later launch argument
    unreadable. -/
def Inst.TameB (fns : List FnDecl) (ptr : Val) (i : Inst) : Bool :=
  !isLaunchCallB fns i && !isDeviceWriterB fns i &&
    (match Inst.destOf? i with
     | some d => !(d.id == ptr.id)
     | none   => true)

theorem tame_noLaunch (fns : List FnDecl) (ptr : Val) (i : Inst)
    (h : Inst.TameB fns ptr i = true) : isLaunchCallB fns i = false := by
  simp [Inst.TameB] at h; exact h.1.1

theorem tame_noWrite (fns : List FnDecl) (ptr : Val) (i : Inst)
    (h : Inst.TameB fns ptr i = true) : ∀ d, Inst.destOf? i = some d → ptr.id ≠ d.id := by
  intro d hd hEq
  rw [Inst.TameB, hd] at h
  simp [hEq] at h

/-- **The environment `scanBlock` threads is exactly `evalPure`'s.**  Extraction
    does not perturb the value model. -/
theorem scanBlock_env (fns : List FnDecl) :
    ∀ (is : List Inst) (e : Env), (scanBlock fns e is).1 = evalPure e is := by
  intro is
  induction is with
  | nil => intro _; rfl
  | cons i is ih => intro e; exact ih (stepPure e i)

/-- **Extraction neither invents nor misses a launch.**

    The count of recovered records equals the count of launch calls in the
    block — soundness and completeness in one equation, quantified over *every*
    instruction list and *every* environment, and proven by induction rather
    than by running a program.

    This is the property `native_decide` on a single closed function cannot
    give: that result holds for `inferFn` and says nothing about any other. -/
theorem scanBlock_length (fns : List FnDecl) :
    ∀ (is : List Inst) (e : Env),
      (scanBlock fns e is).2.length = launchCallCount fns is := by
  intro is
  induction is with
  | nil => intro _; rfl
  | cons i is ih =>
      intro e
      show ((launchAt fns e i).toList ++ (scanBlock fns (stepPure e i) is).2).length
          = (if isLaunchCallB fns i then 1 else 0) + launchCallCount fns is
      rw [List.length_append, ih (stepPure e i)]
      have h := launchAt_isSome fns e i
      cases hopt : launchAt fns e i with
      | none   => rw [hopt] at h; simp at h; simp [← h]
      | some r => rw [hopt] at h; simp at h; simp [← h]

/-- **Every recovered record names a device-write primitive.**

    The extractor cannot report a record for an ordinary call — only for a
    modelled launch or for one of the named primitives that writes device memory
    without being one. -/
theorem scanBlock_fnName (fns : List FnDecl) :
    ∀ (is : List Inst) (e : Env) (r : LaunchRec),
      r ∈ (scanBlock fns e is).2 → r.fnName ∈ deviceOpNames := by
  intro is
  induction is with
  | nil => intro _ _ h; exact absurd h (by simp [scanBlock])
  | cons i is ih =>
      intro e r hr
      have hr' : r ∈ (launchAt fns e i).toList ++ (scanBlock fns (stepPure e i) is).2 := hr
      rcases List.mem_append.mp hr' with h | h
      · cases hopt : launchAt fns e i with
        | none => rw [hopt] at h; exact absurd h (by simp)
        | some q =>
            rw [hopt] at h
            have hq : r = q := by simpa using h
            subst hq
            revert hopt
            cases i with
            | call d fr args =>
                simp only [launchAt]
                cases fnNameOf fns fr with
                | none => intro hopt; exact absurd hopt (by simp)
                | some nm =>
                    by_cases hm : nm ∈ launchNames
                    · intro hopt
                      simp only [if_pos hm, Option.some.injEq] at hopt
                      rw [← hopt]
                      simp only [deviceOpNames, List.mem_append]
                      exact Or.inl hm
                    · by_cases hw : nm ∈ deviceWriterNames
                      · intro hopt
                        simp only [if_neg hm, if_pos hw, Option.some.injEq] at hopt
                        rw [← hopt]
                        simp only [deviceOpNames, List.mem_append]
                        exact Or.inr hw
                      · intro hopt
                        simp only [if_neg hm, if_neg hw] at hopt
                        exact absurd hopt (by simp)
            | _ => intro hopt; exact absurd hopt (by simp [launchAt])
      · exact ih _ _ h

/-- **Scanning a concatenation concatenates the scans.**

    The structural lemma everything about a compiled program rests on: a host
    statement compiles to an instruction *list*, sequencing compiles to `++`,
    and this is what lets the launch-sequence theorem be an induction over the
    statement rather than over the emitted instructions. -/
theorem scanBlock_append (fns : List FnDecl) :
    ∀ (i₁ i₂ : List Inst) (e : Env),
      (scanBlock fns e (i₁ ++ i₂)).2
        = (scanBlock fns e i₁).2 ++ (scanBlock fns (evalPure e i₁) i₂).2 := by
  intro i₁
  induction i₁ with
  | nil => intro _ _; rfl
  | cons i is ih =>
      intro i₂ e
      show ((launchAt fns e i).toList ++ (scanBlock fns (stepPure e i) (is ++ i₂)).2)
          = ((launchAt fns e i).toList ++ (scanBlock fns (stepPure e i) is).2)
            ++ (scanBlock fns (evalPure (stepPure e i) is) i₂).2
      rw [ih i₂ (stepPure e i), List.append_assoc]

/-- The environment after a concatenation is the environment after each in turn. -/
theorem evalPure_append : ∀ (i₁ i₂ : List Inst) (e : Env),
    evalPure e (i₁ ++ i₂) = evalPure (evalPure e i₁) i₂ := by
  intro i₁
  induction i₁ with
  | nil => intro _ _; rfl
  | cons i is ih => intro i₂ e; exact ih i₂ (stepPure e i)

/-- **A block of non-calls records nothing.**  The workhorse for a fragment that
    only materialises arguments. -/
theorem scanBlock_noCalls (fns : List FnDecl) : ∀ (is : List Inst) (e : Env),
    (∀ i ∈ is, Inst.isCallB i = false) → (scanBlock fns e is).2 = [] := by
  intro is
  induction is with
  | nil => intro _ _; rfl
  | cons i is ih =>
      intro e h
      show (launchAt fns e i).toList ++ (scanBlock fns (stepPure e i) is).2 = []
      have h0 : launchAt fns e i = none := by
        have hc := h i (List.mem_cons_self ..)
        cases i <;> simp_all [Inst.isCallB, launchAt]
      rw [h0, ih (stepPure e i) (fun j hj => h j (List.mem_cons_of_mem i hj))]
      rfl

/-- A block with no launch calls contributes nothing — the degenerate case of
    `scanBlock_length`, kept because it is the one a reader checks first. -/
theorem scanBlock_noLaunch (fns : List FnDecl) (is : List Inst) (e : Env)
    (h : launchCallCount fns is = 0) : (scanBlock fns e is).2 = [] :=
  List.eq_nil_of_length_eq_zero (by rw [scanBlock_length fns is e, h])

-- ---------------------------------------------------------------------------
-- What the host wrote *into* the bind arrays
-- ---------------------------------------------------------------------------

/-!
  A `LaunchRec` says **where** the pointer array is (`bindOff`) and **how long**
  it is (`nBufs`).  It does not say what is *in* it — and that is the field a
  pipeline claim actually depends on.  `Kernel.launchAt` emits

  ```
  store bufs[i], ptr + (bindOff + 4*i)      -- one per buffer
  call cl_cuda_launch(ctx, ptr+ptxOff, n, ptr+bindOff, …)
  ```

  so the buffer identities are in the *store* instructions preceding the call,
  which `stepPure` — an SSA environment — cannot see.  Without them a table
  keyed on `bindOff` proves only that two launches are *different*, never that
  either one binds the buffers the stage it is matched to was proven about.

  So this section adds a second, self-contained pass over the same instruction
  list: `stepPure`'s environment threaded alongside a **store map**, and, at
  each recorded device write, the descriptors the bind array then held.  It is a
  separate pass on purpose — `scanBlock` and `LaunchRec` are untouched, so
  everything already proven about them still holds, and the two passes are
  related by a theorem (`bindScan_length`) rather than by construction.
-/

/-- Stores the model could place: `(base pointer, constant offset) ↦ value`.
    Most recent first, so the head of a duplicated address is the live one. -/
abbrev StoreMap := List ((Nat × Int) × SymVal)

def StoreMap.get? : StoreMap → Nat → Int → Option SymVal
  | [],             _,  _  => none
  | ((b, k), v) :: r, b', k' => if b = b' ∧ k = k' then some v else StoreMap.get? r b' k'

/-- The `(value, address)` pair a store instruction writes, if it is a store.
    All three store forms, because a bind array is written with `store.i32` and
    a slot table with `store.i64`, and missing either loses the array. -/
def Inst.storeOf? : Inst → Option (Val × Val)
  | .store v a        => some (v, a)
  | .istore8 v a      => some (v, a)
  | .storeTyped _ v a => some (v, a)
  | _                 => none

/-- **Stepping memory, conservatively.**

    Three cases, and the two that discard are the ones that make this sound:

    * a store to a *recognised* address `base + k` records what was written;
    * a store to an address the model cannot resolve could have hit anything,
      so **nothing** about memory survives it;
    * a call can write through any pointer it was handed — `cl_stdin_readline`
      fills a buffer, `cl_cuda_upload_ptr` writes device memory — so a call
      clears the map too.

    The bind stores sit immediately before their launch with no call and no
    computed-address store between, which is why a map this conservative still
    recovers every one of them. -/
def stepMem (e : Env) (m : StoreMap) : Inst → StoreMap
  | .call _ _ _ => []
  | i =>
    match Inst.storeOf? i with
    | none        => m
    | some (v, a) =>
        match e a with
        | .offset p k => ((p.id, k), e v) :: m
        | _           => []

/-- The value model and the memory model, threaded together. -/
structure BEnv where
  env : Env
  mem : StoreMap

def BEnv.empty : BEnv := ⟨Env.empty, []⟩

def bstep (b : BEnv) (i : Inst) : BEnv := ⟨stepPure b.env i, stepMem b.env b.mem i⟩

/-- The value-and-memory model after a straight run of instructions — the
    `evalPure` of the pair.  `bindScan` threads exactly this, which is
    `bindScan_state`; having it as a definition is what lets a fragment's
    *store map* be specified the way `emitArgs_desc` specifies its values. -/
def bevalPure (b : BEnv) : List Inst → BEnv
  | []      => b
  | i :: is => bevalPure (bstep b i) is

theorem bevalPure_append : ∀ (i₁ i₂ : List Inst) (b : BEnv),
    bevalPure b (i₁ ++ i₂) = bevalPure (bevalPure b i₁) i₂ := by
  intro i₁
  induction i₁ with
  | nil => intro _ _; rfl
  | cons i is ih => intro i₂ b; exact ih i₂ (bstep b i)

theorem bevalPure_env : ∀ (is : List Inst) (b : BEnv),
    (bevalPure b is).env = evalPure b.env is := by
  intro is
  induction is with
  | nil => intro _; rfl
  | cons i is ih => intro b; exact ih (bstep b i)

/-- **Nothing that is not a store moves the store map.**  The reason a bind
    fragment's address arithmetic and handle loads can be ignored. -/
theorem bevalPure_mem_noStores : ∀ (is : List Inst) (b : BEnv),
    (∀ i ∈ is, Inst.storeOf? i = none ∧ Inst.isCallB i = false) →
    (bevalPure b is).mem = b.mem := by
  intro is
  induction is with
  | nil => intro _ _; rfl
  | cons i is ih =>
      intro b h
      obtain ⟨hs, hc⟩ := h i (List.mem_cons_self ..)
      rw [show bevalPure b (i :: is) = bevalPure (bstep b i) is from rfl,
          ih (bstep b i) (fun j hj => h j (List.mem_cons_of_mem i hj))]
      show stepMem b.env b.mem i = b.mem
      cases i <;> simp_all [stepMem, Inst.storeOf?, Inst.isCallB]

/-- **A buffer handle, as the bind scan can identify it.**

    Deliberately *not* `ArgDesc`.  A handle's identity is the address it was
    loaded from, and that address is a base and an offset; `ArgDesc.slot` keeps
    only the offset, which is fine for a vendor call (every one of Qwen2's
    argument handles comes off the descriptor pointer) and wrong here.

    `near` is a load from the same base as the bind array — the descriptor
    pointer — so `near 72` is the hidden state and nothing else can be.  `far`
    is a load from a base the scan did not resolve, carrying that base's SSA
    id: Qwen2's per-layer weights sit at `slotBaseA + k` where `slotBaseA`
    depends on a runtime layer index, so they are `far b k`.  Two `far` handles
    are the same buffer exactly when both components agree — which is why the
    id is kept rather than collapsed to "unresolved". -/
inductive BufDesc where
  | near   : Int → BufDesc
  | far    : Nat → Int → BufDesc
  | const  : Int → BufDesc
  | opaque : BufDesc
  deriving Repr, DecidableEq, BEq

/-- `base` is the bind array's own base, so a handle loaded from it is `near`. -/
def bufDescOf (base : Nat) : SymVal → BufDesc
  | .const k    => .const k
  | .slot p k   => if p.id = base then .near k else .far p.id k
  | _           => .opaque

/-- Forgetting the base recovers exactly the `ArgDesc` a `LaunchRec` records. -/
def BufDesc.toArg : BufDesc → ArgDesc
  | .near k   => .slot k
  | .far _ k  => .slot k
  | .const k  => .const k
  | .opaque   => .opaque

/-- **The base-aware view refines the record's.**  So a literal written in
    `BufDesc` determines the `LaunchRec.args` the same scan produces, and the
    two cannot be stated inconsistently. -/
theorem bufDescOf_toArg (base : Nat) (v : SymVal) :
    (bufDescOf base v).toArg = descOf v := by
  cases v with
  | slot p k => by_cases h : p.id = base <;> simp [bufDescOf, BufDesc.toArg, descOf, h]
  | _ => rfl

/-- `n` consecutive 4-byte entries starting at `base + off`, or `none` if any
    one of them was never written (or was written by an instruction that
    cleared the map).  All-or-nothing on purpose: a partially recovered bind
    array is exactly the situation in which a table match would be a guess. -/
def bindsFrom (m : StoreMap) (base : Nat) (off : Int) : Nat → Nat → Option (List BufDesc)
  | 0,     _ => some []
  | n + 1, i =>
      match m.get? base (off + 4 * (i : Int)) with
      | none   => none
      | some v => (bindsFrom m base off n (i + 1)).map (bufDescOf base v :: ·)

def bindsAt? (m : StoreMap) (base : Nat) (off : Int) (n : Nat) : Option (List BufDesc) :=
  bindsFrom m base off n 0

/-- One step of the read-back, as a rewrite. -/
theorem bindsFrom_succ (m : StoreMap) (base : Nat) (off : Int) (n i : Nat) :
    bindsFrom m base off (n + 1) i
      = match m.get? base (off + 4 * (i : Int)) with
        | none   => none
        | some v => (bindsFrom m base off n (i + 1)).map (bufDescOf base v :: ·) := rfl

/-- **Launch entry points whose bind pointer is argument 3.**

    The `_named` variants take an extra kernel-name pointer, shifting every
    later argument by one.  Rather than recover them from the wrong index, this
    pass declines to recover them at all — `none` is a refusal, and a table
    match against `none` fails.  No generator in the repo emits a named
    launch; if one starts to, this is where it becomes visible. -/
def positionalLaunchNames : List String :=
  ["cl_cuda_launch", "cl_cuda_launch_on_stream"]

/-- **What the scan recovers about a device write beyond its record.**

    A launch fills a pointer array, a vendor call is handed its buffers
    directly, and the two need the same treatment: a handle's identity is a
    base and an offset.  `LaunchRec.args` already carried the vendor arguments
    as `ArgDesc`, which keeps only the offset — so Qwen2's `Wq`, at offset `4`
    of the per-layer base, was recorded identically to whatever might sit at
    offset `4` of the descriptor pointer.  That the two ranges happen not to
    overlap in this program is not something any theorem said.

    `LaunchRec` is left alone (the abstract host model in `HostIR` declares its
    calls in those terms); this is the base-aware view, recovered alongside. -/
structure OpBinds where
  /-- A launch's pointer array, if it was resolved. `none` for a vendor call,
      and for a launch whose array the scan could not recover. -/
  bufs : Option (List BufDesc) := none
  /-- A vendor call's arguments. Empty for a launch — its identity is the PTX
      slot and the bind array above. -/
  args : List BufDesc := []
  deriving Repr, DecidableEq, BEq

/-- **What one instruction's device write bound**, read out of the store map
    and environment that instruction sees.

    `root` is the descriptor pointer — the base every fixed layout slot is
    measured from.  Passing the wrong one is not a silent error: a launch whose
    bind array does not sit at `root` yields `none`, so no stage resolves and
    no plan claim is available. -/
def bindAt (fns : List FnDecl) (root : Nat) (b : BEnv) : Inst → OpBinds
  | .call _ fr args =>
      match fnNameOf fns fr with
      | some nm =>
          if nm ∈ positionalLaunchNames then
            { bufs :=
                match (args[3]?).bind (fun v => match b.env v with
                                                | .offset p k => some (p.id, k)
                                                | _           => none),
                      (args[2]?).bind (fun v => (b.env v).offsetOf?) with
                | some (p, k), some n => if p = root then bindsAt? b.mem p k n.toNat
                                         else none
                | _,           _      => none }
          else if nm ∈ deviceWriterNames then
            { args := args.map (fun v => bufDescOf root (b.env v)) }
          else {}
      | none => {}
  | _ => {}

/-- Walk a block, emitting **one entry per record `scanBlock` emits**, in the
    same order.  The alignment is the theorem below, not a convention. -/
def bindScan (fns : List FnDecl) (root : Nat) : BEnv → List Inst → BEnv × List OpBinds
  | b, []      => (b, [])
  | b, i :: is =>
      let rest := bindScan fns root (bstep b i) is
      (rest.1, (if isLaunchCallB fns i then [bindAt fns root b i] else []) ++ rest.2)

/-- The bind arrays and vendor arguments a built function's device writes use,
    in program order. -/
def bindsOf (root : Nat) (s : IRState) : List OpBinds :=
  (s.allBlocks.foldl (fun (acc : BEnv × List OpBinds) bd =>
      let r := bindScan s.fns root acc.1 bd.insts
      (r.1, acc.2 ++ r.2)) (BEnv.empty, [])).2

/-- **The two passes agree about the value model.**  `bindScan` threads exactly
    `evalPure`, so a bind array recovered here was resolved against the same
    environment the launch record was. -/
theorem bindScan_env (fns : List FnDecl) (root : Nat) :
    ∀ (is : List Inst) (b : BEnv), (bindScan fns root b is).1.env = evalPure b.env is := by
  intro is
  induction is with
  | nil => intro _; rfl
  | cons i is ih => intro b; exact ih (bstep b i)

/-- **…and about memory too.**  `bindScan` threads `bevalPure`, so a fragment's
    store map can be specified once and reused wherever it appears. -/
theorem bindScan_state (fns : List FnDecl) (root : Nat) :
    ∀ (is : List Inst) (b : BEnv), (bindScan fns root b is).1 = bevalPure b is := by
  intro is
  induction is with
  | nil => intro _; rfl
  | cons i is ih => intro b; exact ih (bstep b i)

/-- A block of non-calls binds nothing — the counterpart of
    `scanBlock_noCalls`, and what lets a bind fragment's own address
    arithmetic be stepped over. -/
theorem bindScan_noCalls (fns : List FnDecl) (root : Nat) :
    ∀ (is : List Inst) (b : BEnv),
      (∀ i ∈ is, Inst.isCallB i = false) → (bindScan fns root b is).2 = [] := by
  intro is
  induction is with
  | nil => intro _ _; rfl
  | cons i is ih =>
      intro b h
      show (if isLaunchCallB fns i then [bindAt fns root b i] else [])
              ++ (bindScan fns root (bstep b i) is).2 = []
      have h0 : isLaunchCallB fns i = false := by
        have hc := h i (List.mem_cons_self ..)
        cases i <;> simp_all [Inst.isCallB, isLaunchCallB]
      rw [h0, ih (bstep b i) (fun j hj => h j (List.mem_cons_of_mem i hj))]
      rfl

/-- **A lookup skips a prefix that does not hold the key.**  What makes a bind
    array recoverable from stores that were written in order: the entries the
    fragment wrote *after* entry `i` all carry different keys, so entry `i` is
    still the one found. -/
theorem StoreMap.get?_append_left (l r : StoreMap) (b : Nat) (k : Int)
    (h : ∀ e ∈ l, ¬ (e.1.1 = b ∧ e.1.2 = k)) :
    StoreMap.get? (l ++ r) b k = StoreMap.get? r b k := by
  induction l with
  | nil => rfl
  | cons p l ih =>
      obtain ⟨⟨pb, pk⟩, pv⟩ := p
      show (if pb = b ∧ pk = k then some pv else StoreMap.get? (l ++ r) b k) = _
      rw [if_neg (h _ (List.mem_cons_self ..)),
          ih (fun e he => h e (List.mem_cons_of_mem _ he))]

/-- **…and about how many device writes there are.**  One bind entry per
    recovered record, quantified over every instruction list — so zipping the
    two lists loses nothing and invents nothing. -/
theorem bindScan_length (fns : List FnDecl) (root : Nat) :
    ∀ (is : List Inst) (b : BEnv),
      (bindScan fns root b is).2.length = launchCallCount fns is := by
  intro is
  induction is with
  | nil => intro _; rfl
  | cons i is ih =>
      intro b
      show ((if isLaunchCallB fns i then [bindAt fns root b i] else [])
              ++ (bindScan fns root (bstep b i) is).2).length
          = (if isLaunchCallB fns i then 1 else 0) + launchCallCount fns is
      rw [List.length_append, ih (bstep b i)]
      cases isLaunchCallB fns i <;> simp

/-- A block with no *launch* call binds nothing, even if it calls something
    else — the `prim` case, where a tokenizer's `cl_stdin_readline` is fine. -/
theorem bindScan_noLaunch (fns : List FnDecl) (root : Nat) (is : List Inst) (b : BEnv)
    (h : launchCallCount fns is = 0) : (bindScan fns root b is).2 = [] :=
  List.eq_nil_of_length_eq_zero (by rw [bindScan_length fns root is b, h])

theorem bindScan_append (fns : List FnDecl) (root : Nat) :
    ∀ (i₁ i₂ : List Inst) (b : BEnv),
      (bindScan fns root b (i₁ ++ i₂)).2
        = (bindScan fns root b i₁).2
          ++ (bindScan fns root (bindScan fns root b i₁).1 i₂).2 := by
  intro i₁
  induction i₁ with
  | nil => intro _ _; rfl
  | cons i is ih =>
      intro i₂ b
      show (if isLaunchCallB fns i then [bindAt fns root b i] else [])
              ++ (bindScan fns root (bstep b i) (is ++ i₂)).2
          = ((if isLaunchCallB fns i then [bindAt fns root b i] else [])
              ++ (bindScan fns root (bstep b i) is).2)
            ++ (bindScan fns root (bindScan fns root (bstep b i) is).1 i₂).2
      rw [ih i₂ (bstep b i), List.append_assoc]

/-- **One bind entry per launch record, over the whole function.**

    The statement that makes `deviceOpsOf` total: no record is left without its
    bind array, and no bind array is left without its record. -/
theorem bindsOf_length (root : Nat) (s : IRState) :
    (bindsOf root s).length = (launchesOf s).length := by
  show (List.foldl _ (BEnv.empty, ([] : List OpBinds)) s.allBlocks).2.length
      = (List.foldl _ (Env.empty, ([] : List LaunchRec)) s.allBlocks).2.length
  have key : ∀ (bs : List BlockData) (b : BEnv) (e : Env)
      (a₁ : List OpBinds) (a₂ : List LaunchRec),
      b.env = e → a₁.length = a₂.length →
      (bs.foldl (fun (acc : BEnv × List OpBinds) bd =>
          let r := bindScan s.fns root acc.1 bd.insts
          (r.1, acc.2 ++ r.2)) (b, a₁)).2.length
        = (bs.foldl (fun (acc : Env × List LaunchRec) bd =>
            let r := scanBlock s.fns acc.1 bd.insts
            (r.1, acc.2 ++ r.2)) (e, a₂)).2.length := by
    intro bs
    induction bs with
    | nil => intro _ _ _ _ _ h; exact h
    | cons bd bs ih =>
        intro b e a₁ a₂ he ha
        refine ih (bindScan s.fns root b bd.insts).1 (scanBlock s.fns e bd.insts).1 _ _ ?_ ?_
        · rw [bindScan_env s.fns root bd.insts b, scanBlock_env s.fns bd.insts e, he]
        · rw [List.length_append, List.length_append, ha,
              bindScan_length s.fns root bd.insts b, scanBlock_length s.fns bd.insts e]
  exact key s.allBlocks BEnv.empty Env.empty [] [] rfl rfl

/-- **Records and what they bound, as one list.**  What a table match consumes:
    a device write together with the buffers the host had put in place for it. -/
def deviceOpsOf (root : Nat) (s : IRState) : List (LaunchRec × OpBinds) :=
  (launchesOf s).zip (bindsOf root s)

theorem deviceOpsOf_length (root : Nat) (s : IRState) :
    (deviceOpsOf root s).length = (launchesOf s).length := by
  simp [deviceOpsOf, List.length_zip, bindsOf_length root s]

/-!
  ## What stays trusted on the host side

  | # | trusted | extent |
  |---|---|---|
  | 1 | the CLIF backend | that Cranelift's `iadd`/`load`/`brif`/`call` mean what `stepPure` and the FFI contracts say. The host-side counterpart of the PTX opcode table. |
  | 2 | each FFI primitive | 75 of them. `file`/`stdio`/`ht`/`thread` have contracts statable in an afternoon; `cuda`/`wgpu`/`lmdb` wrap third-party surfaces and stay named assumptions. |
  | 3 | `cl_cuda_launch` | that it runs the PTX at the given slot with the given grid — the seam between this file's `LaunchRec` and `ML/Compose.lean`'s `Pipeline`. |

  `Modellable` is what keeps (1) honest: a program reaching for `rawInst` fails
  the gate instead of being modelled as though it had not.
-/

end AlgorithmLib.Clif
