import AlgorithmLib.Clif

/-!
  # A structured host program

  `Clif.lean` recovers the device-write sequence from a *built* CLIF function,
  which works but has two limits the inference pipeline runs straight into:

  * the 24 layers are a `forLoop`, so the static instruction stream contains the
    layer's kernels **once**, not 24 times;
  * those kernels are launched inside `fnLayerStep`, reached by `callVoid`, and
    a per-function scan does not follow calls.

  Together those are why `launchesOf inferState` returns a single record: the
  embed launch. Everything else is behind a loop and a call.

  This file is the structured form that fixes both — the host-side analogue of
  `EWStmt`, whose `forN` is a fold and whose lowering to real branches is proven
  by `flatEW_sound`. A program is a *term*, so the sequence is a structural
  recursion rather than an evaluation, and the whole inference driver costs a
  `decide` on a small term instead of 43 seconds of `native_decide`.

  **`flatHI_sound` is the theorem.** Compiling an `HStmt` and running the result
  under a program counter performs exactly `HStmt.deviceOps` — every launch
  record *and the bind array it was handed*, recovered from the stores the
  program executed rather than supplied to the machine. That second half is
  what makes the loop say something: without it, "the twenty-fourth layer
  launched RoPE" was a claim about a PTX slot and a grid, with the buffers left
  to a table of intentions. The emitted code
  keeps its loop — the body appears once, as the current `IRBuilder` generator
  emits it — and it is the *trace* that unrolls, which is why the fuel is
  existential. Nothing about the emitted control flow is assumed: the counter,
  the guard, the increment and the back edge are all executed.

  Three kinds of step take part. `launch` is a modelled kernel. `extern` is a
  device write this model declares rather than interprets — a vendor GEMV — and
  materialises its own arguments so that *which* buffers it touched is recovered
  rather than assumed. `prim` is launch-free filler, and `HStmt.TameB` decides
  that it really is: a `prim` hiding a launch or a cuBLAS call is rejected, so
  the declared sequence cannot be short.
-/

namespace AlgorithmLib.Host

open AlgorithmLib.IR AlgorithmLib.Clif

/-- **One argument of a declared call, or one entry of a bind array.**

    A vendor call has no PTX slot, so its identity *is* its arguments: which
    matrix, which vector, which output.  Those reach the call as handles loaded
    from fixed slots, which is why `slot` is a case here and why `SymVal.slot`
    exists at all — without it Qwen2's `Wq`, `Wk` and `Wv` calls, all 896×896,
    are the same record. -/
inductive ExternArg where
  | const : Int → ExternArg
  | slot  : Nat → ExternArg
  /-- **A handle reached through a base this model does not resolve.**

      Qwen2's per-layer weights live at `slotBaseA + k` where `slotBaseA`
      depends on the runtime layer index, so they are not `ptr + k` for any
      compile-time `k`.  Without this case a layer's launches cannot be
      *declared* at all — every kernel bind array in the model has one — which
      is why the driver could not be written as an `HStmt` before.

      `b` is the SSA id of the base; it must be a value the surrounding code
      already holds and that this model reads as unresolved.  Both are side
      conditions on the emitter below, not assumptions. -/
  | far   : Nat → Int → ExternArg
  /-- **An address, not a handle**: `ptr + k` passed directly rather than
      loaded.  A staging pointer is the case that matters.  `Clif.bufDescOf`
      reports `opaque` for it — it resolves the value but has no name for it —
      which is why a declared program needs this constructor to describe the
      vendor calls a layer makes, and why it is *emittable* (unlike a bare
      "unknown", which no instruction produces). -/
  | addr : Nat → ExternArg
  /-- **A handle the surrounding code already holds.**

      `slot k` says "load the handle at `ptr + k`"; this says "SSA value `v`
      already holds the handle that lives at `ptr + k`, use it".  Same handle,
      same descriptor, three fewer instructions — which is what
      `Tensor.Kernel.launchAt` emits, since it threads `bufs : List Val` in
      from the caller rather than reloading them.

      The side condition is on the environment: `v` must be in scope and must
      actually hold that handle.  See `ExternArg.Ok`. -/
  | held : Nat → Nat → ExternArg
  deriving Repr, DecidableEq

def ExternArg.toDesc : ExternArg → ArgDesc
  | .const c => .const c
  | .slot k  => .slot (Int.ofNat k)
  | .far _ k => .slot k
  | .addr _  => .opaque
  | .held _ k => .slot (Int.ofNat k)

/-- **The base-aware view of the same handle.**  A slot load off the descriptor
    pointer is `near`; `Clif.bufDescOf` returns exactly this once the scan has
    resolved the base, which is what the round-trip theorems below prove. -/
def ExternArg.toBuf : ExternArg → BufDesc
  | .const c => .const c
  | .slot k  => .near (Int.ofNat k)
  | .far b k => .far b k
  | .addr _  => .opaque
  | .held _ k => .near (Int.ofNat k)

/-- …and the symbolic value the emitted fragment really leaves in memory. -/
def ExternArg.toSym (ptr : Val) : ExternArg → SymVal
  | .const c => .const c
  | .slot k  => .slot ptr (Int.ofNat k)
  | .far b k => .slot ⟨b⟩ k
  | .addr k  => .offset ptr (Int.ofNat k)
  | .held _ k => .slot ptr (Int.ofNat k)

/-- The base a `far` handle is reached through, if it is one. -/
def ExternArg.baseOf? : ExternArg → Option Nat
  | .far b _ => some b
  | _        => none

/-- **Every SSA value this argument reads that the fragment does not itself
    create.**

    `baseOf?` answers the same question for `far` alone and as an `Option`.
    An argument that depends on the surrounding code has to be excluded from
    the *frame* — the values a preceding fragment may overwrite — and `deps`
    lets that exclusion be stated once for every constructor at once. -/
def ExternArg.deps : ExternArg → List Nat
  | .far b _  => [b]
  | .held v _ => [v]
  | _         => []

/-- **What materialising this argument needs of the environment.**

    Split per constructor because the conditions differ in *shape*: a `far`
    handle needs its base to be unresolved so the load produces `slot ⟨b⟩ k`,
    while `held` needs the register to hold the handle outright.  A single
    `Option Nat` can express the first but not the second.

    For `far`: the base was allocated before this fragment (`b < n`), it is not
    the descriptor pointer (`b ≠ ptr.id` — a `far` off `ptr` *is* a `near`, and
    the two are different table entries), and this model reads it as
    unresolved.

    A `Prop` rather than a `Bool` because it is about the environment, which
    `TameB` cannot see. -/
def ExternArg.Ok (e : Env) (ptr : Val) (n : Nat) : ExternArg → Prop
  | .far b _  => b < n ∧ b ≠ ptr.id ∧ e ⟨b⟩ = SymVal.unknown
  | .held v k => v < n ∧ e ⟨v⟩ = SymVal.slot ptr (Int.ofNat k)
  | _         => True

def FarOk (e : Env) (ptr : Val) (n : Nat) (as : List ExternArg) : Prop :=
  ∀ a ∈ as, a.Ok e ptr n

/-- The `far` projection of `Ok`, in the flat form the emit lemmas below are
    stated against. -/
theorem FarOk.base {e : Env} {ptr : Val} {n : Nat} {as : List ExternArg}
    (h : FarOk e ptr n as) :
    ∀ x ∈ as, ∀ b, x.baseOf? = some b →
      b < n ∧ b ≠ ptr.id ∧ e ⟨b⟩ = SymVal.unknown := by
  intro x hx b hb
  have hok := h x hx
  cases x <;> simp_all [ExternArg.baseOf?, ExternArg.Ok]

/-- …and the converse, **for argument lists with no `held`**.  There is no
    converse in general: a `held` argument's condition is a fact about a
    register the surrounding code wrote, and cannot be reconstructed from the
    `far` facts. -/
theorem FarOk.mk {e : Env} {ptr : Val} {n : Nat} {as : List ExternArg}
    (hh : ∀ x ∈ as, ∀ v k, x ≠ .held v k)
    (h : ∀ x ∈ as, ∀ b, x.baseOf? = some b →
          b < n ∧ b ≠ ptr.id ∧ e ⟨b⟩ = SymVal.unknown) :
    FarOk e ptr n as := by
  intro x hx
  cases x with
  | far b k => exact h _ hx b rfl
  | held v k => exact absurd rfl (hh _ hx v k)
  | _ => trivial

/-- `baseOf?` is the `far` entry of `deps`.  The bridge that lets the frame
    condition be stated once over `deps` while the `far` lemmas stay stated
    over `baseOf?`. -/
theorem FarOk.headOk {e : Env} {ptr : Val} {n : Nat} {a : ExternArg}
    {as : List ExternArg} (h : FarOk e ptr n (a :: as)) : a.Ok e ptr n :=
  h a (List.mem_cons_self ..)

theorem ExternArg.mem_deps_of_baseOf? {x : ExternArg} {b : Nat}
    (h : x.baseOf? = some b) : b ∈ x.deps := by
  cases x <;> simp_all [ExternArg.baseOf?, ExternArg.deps]

/-- Neither a `far` handle nor one already in scope — the case with no side
    condition at all, and decidable. -/
def ExternArg.PlainB : ExternArg → Bool
  | .far _ _  => false
  | .held _ _ => false
  | _         => true

/-- The syntactic sufficient condition: an argument list with no `far` handle
    satisfies `FarOk` in any environment.  `flatHI_sound` still asks for this
    rather than `FarOk` itself, because `HStmt.TameB` is a `Bool` and cannot
    speak about the environment; the *emit* lemmas below are stated at `FarOk`
    and so already cover the general case. -/
def ExternArg.FarFreeB : ExternArg → Bool
  | .far _ _  => false
  | .held _ _ => false
  | _         => true

/-- No `far` handles at all — the common case, and decidable. -/
theorem FarOk.of_noBases {e : Env} {ptr : Val} {n : Nat} {as : List ExternArg}
    (h : as.all ExternArg.PlainB = true) : FarOk e ptr n as := by
  intro a ha
  have hp := (List.all_eq_true.mp h) a (by simpa using ha)
  cases a <;> simp_all [ExternArg.PlainB, ExternArg.Ok]

/-- No argument reads a register the surrounding code filled. -/
def ExternArg.HeldFreeB : ExternArg → Bool
  | .held _ _ => false
  | _         => true

/-- **A scan that starts from the empty environment can only see `far`.**

    `stateOf` runs the scan from `Env.empty`, so it knows nothing about what
    the caller left in registers: a `held` argument is invisible to it, hence
    the `HeldFreeB` hypothesis.  A `far` base is fine — `Env.empty` reads every
    value as unresolved, which is what `far` wants. -/
theorem FarOk.empty_of_flat {ptr : Val} {n : Nat} {as : List ExternArg}
    (hh : as.all ExternArg.HeldFreeB = true)
    (h : ∀ x ∈ as, ∀ b, x.baseOf? = some b → b < n ∧ b ≠ ptr.id) :
    FarOk Env.empty ptr n as := by
  intro a ha
  have hp := (List.all_eq_true.mp hh) a (by simpa using ha)
  cases a with
  | far b k => exact ⟨(h _ ha b rfl).1, (h _ ha b rfl).2, rfl⟩
  | held v k => simp [ExternArg.HeldFreeB] at hp
  | _ => trivial

theorem FarOk.of_farFree {e : Env} {ptr : Val} {n : Nat} {as : List ExternArg}
    (h : as.all ExternArg.PlainB = true) : FarOk e ptr n as :=
  FarOk.of_noBases h

theorem noBases_lt {as : List ExternArg} {n p : Nat}
    (h : as.all ExternArg.PlainB = true) :
    ∀ x ∈ as, ∀ b, ExternArg.baseOf? x = some b → b < n ∧ b ≠ p := by
  intro a ha b hb
  have := (List.all_eq_true.mp h) a (by simpa using ha)
  cases a <;> simp_all [ExternArg.PlainB, ExternArg.baseOf?]

theorem noBases_primDests {as : List ExternArg} {ds : List Nat}
    (h : as.all ExternArg.PlainB = true) :
    ∀ x ∈ as, ∀ b ∈ ExternArg.deps x, b ∉ ds := by
  intro a ha b hb
  have := (List.all_eq_true.mp h) a (by simpa using ha)
  cases a <;> simp_all [ExternArg.PlainB, ExternArg.deps]

theorem FarOk.tail {e : Env} {ptr : Val} {n : Nat} {a : ExternArg} {as : List ExternArg}
    (h : FarOk e ptr n (a :: as)) : FarOk e ptr n as :=
  fun x hx => h x (List.mem_cons_of_mem a hx)

theorem FarOk.head {e : Env} {ptr : Val} {n : Nat} {a : ExternArg} {as : List ExternArg}
    (h : FarOk e ptr n (a :: as)) : ∀ b, a.baseOf? = some b →
      b < n ∧ b ≠ ptr.id ∧ e ⟨b⟩ = SymVal.unknown :=
  h.base a (List.mem_cons_self ..)

/-- **The condition survives anything allocated above it.**  The frame argument
    that already carries `e ptr = unknown` across a fragment, for the bases. -/
theorem FarOk.monoD {e e' : Env} {ptr : Val} {n n' : Nat} {as : List ExternArg}
    {ds : List Nat} (h : FarOk e ptr n as) (hn : n ≤ n')
    (hd : ∀ x ∈ as, ∀ b ∈ x.deps, b ∉ ds)
    (hfr : ∀ w : Val, w.id < n → w.id ∉ ds → e' w = e w) : FarOk e' ptr n' as := by
  intro a ha
  have hok := h a ha
  cases a with
  | far b k =>
      obtain ⟨h1, h2, h3⟩ := hok
      exact ⟨Nat.lt_of_lt_of_le h1 hn, h2, by
        rw [hfr ⟨b⟩ h1 (hd _ ha b (by simp [ExternArg.deps]))]; exact h3⟩
  | held v k =>
      obtain ⟨h1, h2⟩ := hok
      exact ⟨Nat.lt_of_lt_of_le h1 hn, by
        rw [hfr ⟨v⟩ h1 (hd _ ha v (by simp [ExternArg.deps]))]; exact h2⟩
  | _ => trivial

theorem FarOk.mono {e e' : Env} {ptr : Val} {n n' : Nat} {as : List ExternArg}
    (h : FarOk e ptr n as) (hn : n ≤ n')
    (hfr : ∀ w : Val, w.id < n → e' w = e w) : FarOk e' ptr n' as :=
  h.monoD (ds := []) hn (fun _ _ _ _ => List.not_mem_nil) (fun w hw _ => hfr w hw)

/-- What the scan makes of what the fragment stored is what was declared.

    The side condition is real: a `far` handle whose base *is* the descriptor
    pointer is a `near` handle, and the two are different table entries. -/
theorem ExternArg.bufDescOf_toSym (ptr : Val) (a : ExternArg)
    (h : ∀ b k, a = .far b k → b ≠ ptr.id) :
    bufDescOf ptr.id (a.toSym ptr) = a.toBuf := by
  cases a with
  | far b k => simp [ExternArg.toSym, ExternArg.toBuf, bufDescOf, h b k rfl]
  | _ => simp [ExternArg.toSym, ExternArg.toBuf, bufDescOf]

/-- **A kernel launch, declared rather than recovered.**

    `binds` is the pointer array the host fills in before the call, and its
    absence was the seam: `Clif.bindsOf` recovers a bind array's *contents*
    from the emitted stores, but this model emitted no stores, so the
    composition theorem had to take the recovered list as a free parameter —
    it could not say the program bound what the plan said it bound.  With
    `binds` here, `HStmt.deviceOps` is derived and the parameter is gone. -/
structure LaunchStep where
  ptxOff  : Nat
  nBufs   : Nat
  bindOff : Nat
  gridX   : Nat
  blockX  : Nat
  binds   : List ExternArg
  deriving Repr, DecidableEq

def LaunchStep.toRec (s : LaunchStep) : LaunchRec :=
  { fnName    := "cl_cuda_launch"
    kernelOff := some (Int.ofNat s.ptxOff)
    nBufs     := some (Int.ofNat s.nBufs)
    bindOff   := some (Int.ofNat s.bindOff)
    gridX     := some (Int.ofNat s.gridX)
    blockX    := some (Int.ofNat s.blockX) }

/-- The base-aware companion to `toRec`: what the launch bound. -/
def LaunchStep.toBinds (s : LaunchStep) : OpBinds :=
  { bufs := some (s.binds.map ExternArg.toBuf) }

/-- **The arity a launch declares must be the array it fills.**  Decided, not
    assumed: a `LaunchStep` whose `nBufs` disagrees with `binds.length` makes
    `bindsAt?` read past the end of what the fragment wrote, and recovery
    returns `none`.  `HStmt.TameB` rejects such a step instead. -/
def LaunchStep.WellFormedB (s : LaunchStep) : Bool := s.nBufs == s.binds.length

/-- **A declared device write.**  Names the primitive, the call that performs it,
    and what its arguments are; what it *computes* is a `DeclaredStep` on the
    pipeline side, carrying its assumption in the open. -/
structure ExternStep where
  name : String
  fn   : FnRef
  argv : List ExternArg

def ExternStep.toRec (s : ExternStep) : LaunchRec :=
  { fnName := s.name, kernelOff := none, nBufs := none,
    bindOff := none, gridX := none, blockX := none,
    args := s.argv.map ExternArg.toDesc }

/-- A vendor call has no bind array — its buffers arrive as arguments — so the
    base-aware view fills `args` and leaves `bufs` at `none`. -/
def ExternStep.toBinds (s : ExternStep) : OpBinds :=
  { args := s.argv.map ExternArg.toBuf }

/-- **A host program.**

    Every device write is a node. `launch` is a modelled kernel; `extern` is one
    this model declares rather than interprets. Everything else — address
    arithmetic, loads, stores, tokenizer loops — is `prim`, which carries its
    real instructions, so a compiled `HStmt` is a whole CLIF function and not a
    skeleton. `HStmt.TameB` decides that a `prim` is what it claims: filler
    hiding a launch or a vendor call is rejected, because a sequence that
    silently omits a device write is worse than one that refuses to be built.

    `call` carries the callee's body rather than a reference. The generators'
    call graph is acyclic (no generator recurses), so inlining is faithful and
    it keeps `launches` structurally recursive instead of needing fuel or a
    well-foundedness argument. -/
inductive HStmt where
  | skip   : HStmt
  | seq    : HStmt → HStmt → HStmt
  | launch : LaunchStep → HStmt
  /-- A bounded loop with a static trip count — the 24 layers. -/
  | forN   : Nat → HStmt → HStmt
  /-- A non-recursive call, body inlined. -/
  | call   : HStmt → HStmt
  /-- **A device write this model does not interpret** — a vendor GEMV, a
      captured-graph replay.  A first-class node rather than something hidden
      in `prim`, so it takes its place in the sequence and the composition can
      account for it instead of silently dropping it. -/
  | extern : ExternStep → HStmt
  /-- Any launch-free fragment: address arithmetic, loads, stores, tokenizer
      loops.  Carries the real instructions, so a compiled `HStmt` is a whole
      CLIF function rather than a launch skeleton.  That it really is launch-free
      is `HStmt.TameB`'s job — decided, not assumed. -/
  | prim   : List Inst → HStmt

/-- **The launch sequence, by structural recursion.**  No evaluation of a
    builder, no `native_decide`: a loop contributes its body `n` times because
    that is what the recursion says, not because something ran it. -/
def HStmt.launches : HStmt → List LaunchRec
  | .skip       => []
  | .seq a b    => a.launches ++ b.launches
  | .launch s   => [s.toRec]
  | .forN n b   => (List.replicate n b.launches).flatten
  | .call b     => b.launches
  | .extern es  => [es.toRec]
  | .prim _     => []

/-- **What each of those device writes bound**, by the same recursion.

    One entry per `launches` entry, in the same order — `binds_length` below,
    which is what makes `deviceOps` a total zip rather than a truncation. -/
def HStmt.binds : HStmt → List OpBinds
  | .skip       => []
  | .seq a b    => a.binds ++ b.binds
  | .launch s   => [s.toBinds]
  | .forN n b   => (List.replicate n b.binds).flatten
  | .call b     => b.binds
  | .extern es  => [es.toBinds]
  | .prim _     => []

/-- **The declared device-write sequence, records and bindings together** — the
    shape `Clif.deviceOpsOf` produces and the shape a table match consumes. -/
def HStmt.deviceOps (s : HStmt) : List (LaunchRec × OpBinds) :=
  s.launches.zip s.binds

/-- A straight-line block of statements.  Right-nested `seq`, so `launches` and
    `binds` are plain concatenations. -/
def HStmt.seqs : List HStmt → HStmt
  | []      => .skip
  | s :: ss => .seq s (HStmt.seqs ss)

/-- **The declared argument a recovered descriptor came from.**

    Total, and inverse to `ExternArg.toBuf` on every descriptor a scan can
    produce.  Having it means a declared driver is written from the *same*
    literals as the tables the scan is matched against, so the two cannot drift
    apart by a typo. -/
def argOf : BufDesc → ExternArg
  | .near k  => .slot k.toNat
  | .far b k => .far b k
  | .const c => .const c
  -- `opaque` is what the scan reports for an address it resolved but cannot
  -- name; `addr 0` is the least-committed declared form that reports the same.
  -- A driver written by hand should say the offset it means.
  | .opaque  => .addr 0

/-- **The same descriptor, declared as a register the caller already filled.**

    `argOf` is written to be *recovered* from a scan, so it has no SSA id to
    give and must say `slot` — six instructions per bind.  A generator does
    know: it loaded the handle itself.  `holds` is that knowledge, a partial
    map from a near-offset to the value holding it; where it answers, the bind
    compiles to the three instructions `Tensor.Kernel.launchAt` writes.

    Everything else about the declaration is unchanged, which is the content of
    `argOfHeld_toBuf` below — so switching a driver over is invisible to
    `planOf?`, `bufOf` and every table keyed on `BufDesc`. -/
def argOfHeld (holds : Int → Option Nat) : BufDesc → ExternArg
  | .near k  => match holds k with
                | some v => .held v k.toNat
                | none   => .slot k.toNat
  | d        => argOf d

/-- **Declaring a bind as `held` changes no descriptor.**  The theorem the
    generator switch needs: whatever `holds` answers, the recovered bind array
    is the same one, so no table, plan or law moves. -/
theorem argOfHeld_toBuf (holds : Int → Option Nat) (d : BufDesc) :
    (argOfHeld holds d).toBuf = (argOf d).toBuf := by
  cases d with
  | near k =>
      cases h : holds k with
      | none   => simp [argOfHeld, argOf, ExternArg.toBuf, h]
      | some v => simp [argOfHeld, argOf, ExternArg.toBuf, h]
  | _ => rfl

/-- …and the same argument descriptor, for the vendor-call side. -/
theorem argOfHeld_toDesc (holds : Int → Option Nat) (d : BufDesc) :
    (argOfHeld holds d).toDesc = (argOf d).toDesc := by
  cases d with
  | near k =>
      cases h : holds k with
      | none   => simp [argOfHeld, argOf, ExternArg.toDesc, h]
      | some v => simp [argOfHeld, argOf, ExternArg.toDesc, h]
  | _ => rfl

/-- A `held` declaration is exactly `slot` wherever the generator does not
    claim to hold the handle — so a driver can be switched over one buffer at a
    time. -/
theorem argOfHeld_none (d : BufDesc) : argOfHeld (fun _ => none) d = argOf d := by
  cases d <;> rfl

/-- A declared kernel launch, from the recovered bind array. -/
def kStep (ptx nb bo grid : Nat) (bs : List BufDesc) : LaunchStep :=
  { ptxOff := ptx, nBufs := nb, bindOff := bo, gridX := grid, blockX := 32,
    binds := bs.map argOf }

/-- A declared vendor call, from its recovered arguments.  `fnOf` supplies the
    `FnRef`; `deviceOps` does not look at it — only compiling would. -/
def vStep (fnOf : String → FnRef) (nm : String) (as : List BufDesc) : ExternStep :=
  { name := nm, fn := fnOf nm, argv := as.map argOf }

/-- Total launches, as a count. -/
def HStmt.launchCount (s : HStmt) : Nat := s.launches.length

/-- Every declared argument a statement mentions — the launches' bind entries
    and the externs' arguments — so a `FarOk` condition can be stated once for a
    whole program. -/
def HStmt.farArgs : HStmt → List ExternArg
  | .skip      => []
  | .prim _    => []
  | .launch s  => s.binds
  | .extern es => es.argv
  | .seq a b   => a.farArgs ++ b.farArgs
  | .forN _ b  => b.farArgs
  | .call b    => b.farArgs

/-- The SSA values a statement's `prim` fragments write.  Everything else a
    compiled statement writes is allocated above the fragment's watermark, so
    this is the *only* way a declared program can disturb a value the caller
    already held — which is what a `far` base is. -/
def HStmt.primDests : HStmt → List Nat
  | .skip      => []
  | .prim is   => is.filterMap (fun i => (Inst.destOf? i).map Val.id)
  | .launch _  => []
  | .extern _  => []
  | .seq a b   => a.primDests ++ b.primDests
  | .forN _ b  => b.primDests
  | .call b    => b.primDests

/-- **The declared launch sequence is honest about this program.**

    Every `prim` fragment really is launch-free and really does leave the base
    pointer alone.  Decidable, so it is discharged by `decide` at the point a
    driver is assembled — the alternative would be to *assume* that filler code
    is filler, which is exactly the kind of unstated hypothesis that makes a
    theorem true and useless. -/
def HStmt.TameB (fns : List FnDecl) (ptr : Val) : HStmt → Bool
  | .skip     => true
  | .prim is  => is.all (Inst.TameB fns ptr)
  -- a launch must fill exactly the array it declares the arity of
  | .launch s => s.WellFormedB
  -- an extern node must name the primitive its call really resolves to, and
  -- that primitive must be one this model has agreed to declare rather than
  -- interpret; otherwise the record would not match what runs
  | .extern es => (fnNameOf fns es.fn == some es.name)
                    && decide (es.name ∈ deviceWriterNames)
                    && !decide (es.name ∈ launchNames)
  | .seq a b  => HStmt.TameB fns ptr a && HStmt.TameB fns ptr b
  | .forN _ b => HStmt.TameB fns ptr b
  | .call b   => HStmt.TameB fns ptr b

/-- A launch-free instruction list contributes nothing to the count. -/
theorem launchCallCount_zero (fns : List FnDecl) : ∀ (is : List Inst),
    (∀ i ∈ is, isLaunchCallB fns i = false) → launchCallCount fns is = 0 := by
  intro is
  induction is with
  | nil => intro _; rfl
  | cons i is ih =>
      intro h
      show (if isLaunchCallB fns i then 1 else 0) + launchCallCount fns is = 0
      rw [h i (List.mem_cons_self ..), ih (fun j hj => h j (List.mem_cons_of_mem i hj))]
      rfl

-- ---------------------------------------------------------------------------
-- The theorems the inference pipeline needs
-- ---------------------------------------------------------------------------

theorem replicate_flatten_length {α : Type} : ∀ (n : Nat) (l : List α),
    ((List.replicate n l).flatten).length = n * l.length
  | 0, l => by simp
  | n + 1, l => by
      rw [List.replicate_succ, List.flatten_cons, List.length_append,
          replicate_flatten_length n l, Nat.succ_mul, Nat.add_comm]

/-- **A bounded loop launches its body once per iteration.**  The statement
    that a flat launch list cannot express, and the reason a structured host IR
    is needed at all rather than a plan as a list. -/
theorem launches_forN (n : Nat) (b : HStmt) :
    (HStmt.forN n b).launchCount = n * b.launchCount :=
  replicate_flatten_length n b.launches

/-- **Every declared device write has a declared binding.**  So `deviceOps`
    zips two lists of the same length: no record is left without its bind
    array, and no bind array is invented for a record that is not there. -/
theorem HStmt.binds_length : ∀ (s : HStmt), s.binds.length = s.launches.length := by
  intro s
  induction s with
  | skip => rfl
  | prim _ => rfl
  | launch _ => rfl
  | extern _ => rfl
  | call b ih => exact ih
  | seq a b iha ihb =>
      show (a.binds ++ b.binds).length = (a.launches ++ b.launches).length
      rw [List.length_append, List.length_append, iha, ihb]
  | forN n b ih =>
      show (List.replicate n b.binds).flatten.length
          = (List.replicate n b.launches).flatten.length
      rw [replicate_flatten_length, replicate_flatten_length, ih]

/-- `zip` distributes over `++` when the first halves agree in length. -/
theorem zip_append_of_length {α β : Type} : ∀ (a₁ : List α) (b₁ : List β) (a₂ : List α)
    (b₂ : List β), a₁.length = b₁.length →
    (a₁ ++ a₂).zip (b₁ ++ b₂) = a₁.zip b₁ ++ a₂.zip b₂ := by
  intro a₁
  induction a₁ with
  | nil => intro b₁ _ _ h; cases b₁ with
           | nil => rfl
           | cons _ _ => simp at h
  | cons x xs ih =>
      intro b₁ a₂ b₂ h
      cases b₁ with
      | nil => simp at h
      | cons y ys =>
          show (x, y) :: (xs ++ a₂).zip (ys ++ b₂) = (x, y) :: (xs.zip ys ++ a₂.zip b₂)
          rw [ih ys a₂ b₂ (by simpa using h)]

theorem zip_flatten_replicate {α β : Type} (a : List α) (b : List β)
    (h : a.length = b.length) : ∀ (n : Nat),
    ((List.replicate n a).flatten).zip ((List.replicate n b).flatten)
      = (List.replicate n (a.zip b)).flatten := by
  intro n
  induction n with
  | zero => rfl
  | succ n ih =>
      show (a ++ (List.replicate n a).flatten).zip (b ++ (List.replicate n b).flatten)
          = a.zip b ++ (List.replicate n (a.zip b)).flatten
      rw [zip_append_of_length a b _ _ h, ih]

/-- **The declared device-write sequence composes.**

    Sequencing concatenates and a loop repeats — the same shape `launches` has,
    now for records *and* bindings together.  This is what lets a plan be
    matched against a looped program by induction rather than by reducing every
    iteration, exactly as `stepsOf?_append` does on the plan side. -/
theorem HStmt.deviceOps_seq (a b : HStmt) :
    (HStmt.seq a b).deviceOps = a.deviceOps ++ b.deviceOps := by
  show (a.launches ++ b.launches).zip (a.binds ++ b.binds) = _
  rw [zip_append_of_length _ _ _ _ (HStmt.binds_length a).symm]
  rfl

theorem HStmt.deviceOps_call (b : HStmt) :
    (HStmt.call b).deviceOps = b.deviceOps := rfl

theorem HStmt.deviceOps_forN (n : Nat) (b : HStmt) :
    (HStmt.forN n b).deviceOps = (List.replicate n b.deviceOps).flatten := by
  show ((List.replicate n b.launches).flatten).zip ((List.replicate n b.binds).flatten) = _
  rw [zip_flatten_replicate _ _ (HStmt.binds_length b).symm]
  rfl

theorem HStmt.deviceOps_length (s : HStmt) :
    s.deviceOps.length = s.launches.length := by
  simp [HStmt.deviceOps, List.length_zip, HStmt.binds_length s]

/-- **A call launches exactly what its callee does** — the interprocedural step
    a per-function scan of built CLIF cannot take. -/
theorem launches_call (b : HStmt) : (HStmt.call b).launches = b.launches := rfl

/-- Sequencing concatenates, so a driver's launch sequence is its parts'. -/
theorem launches_seq (a b : HStmt) :
    (HStmt.seq a b).launches = a.launches ++ b.launches := rfl

/-- Sequencing adds counts — the compositional form, so a driver's total is
    derived from its parts without ever materialising the list.

    This matters more than it looks. `decide` on the whole Qwen2 driver hits
    `maxRecDepth`, because it forces a 267-element list into existence. Rewriting
    with these lemmas proves the same number in constant work, at any depth. The
    structured IR removes the *need* to evaluate; these lemmas are how you cash
    that in. -/
theorem launchCount_seq (a b : HStmt) :
    (HStmt.seq a b).launchCount = a.launchCount + b.launchCount := by
  simp [HStmt.launchCount, HStmt.launches]

theorem launchCount_call (b : HStmt) :
    (HStmt.call b).launchCount = b.launchCount := rfl

theorem launchCount_launch (s : LaunchStep) :
    (HStmt.launch s).launchCount = 1 := rfl

theorem launchCount_prim (is : List Inst) : (HStmt.prim is).launchCount = 0 := rfl

theorem launchCount_skip : HStmt.skip.launchCount = 0 := rfl

/-- Launch-free fragments contribute nothing, at any nesting depth. -/
theorem launches_prim_free : ∀ (s : HStmt), s.launches = [] → s.launchCount = 0 :=
  fun _ h => by simp [HStmt.launchCount, h]

-- ---------------------------------------------------------------------------
-- The emitted program, and what it *does* rather than what it contains
-- ---------------------------------------------------------------------------

/-!
  A CLIF loop puts its body in the instruction stream **once** but executes it
  `n` times, so a *static* scan of the emitted code can never equal the launch
  sequence of a looping program.  Unrolling would make the two coincide, at the
  cost of emitting worse code — bending the program to fit a weak model.

  The right fix is on the model side: give the emitted program a **trace**
  semantics.  This mirrors `PtxFlat` exactly — a flat instruction list with
  `jmp`/`jmpIf`, a fuel-indexed stepper, and existential fuel in the theorem
  (`∃ k, steps … = some …`).  The loop stays a loop; the trace unrolls.
-/

/-- A flat host instruction: a CLIF instruction, or a branch. -/
inductive HI where
  | inst  : Inst → HI
  /-- Unconditional jump to a program point. -/
  | jmp   : Nat → HI
  /-- Set a loop counter. -/
  | setC  : Val → Nat → HI
  /-- Increment a loop counter. -/
  | incC  : Val → HI
  /-- `jmpLt c bound t f` — the only conditional emitted, since the only
      control flow `flatHI` produces is `forN`'s bounded loop. -/
  | jmpLt : Val → Nat → Nat → Nat → HI

/-- Machine state: the SSA environment, the **store map**, a counter file for
    loop indices, and the two traces accumulated so far.

    `mem` and `btrace` are why this machine can say what a launch *bound*.  A
    bind array is not an argument of the launch call — it is a pointer array the
    program wrote into memory just before it, so recovering it means executing
    those stores.  Without them the machine could only say *which* kernel ran
    on how many blocks, and "the twenty-fourth layer launched RoPE against the
    right buffers" would be outside what any theorem here states. -/
structure HCfg where
  pc     : Nat
  env    : Env
  mem    : StoreMap
  ctr    : Nat → Nat
  trace  : List LaunchRec
  /-- One entry per `trace` entry, in the same order — what that device write
      bound, read out of `mem` at the moment it executed. -/
  btrace : List OpBinds

def HCfg.setCtr (c : HCfg) (v : Val) (k : Nat) : HCfg :=
  { c with ctr := fun w => if w = v.id then k else c.ctr w }

/-- One transition.  A device write appends to both traces — its record, and
    what the store map said its buffers were; everything else just moves the
    environment and memory along.  Falling off the end is stuck, not silently
    approximated — same discipline as `fstep`.

    `root` is the descriptor pointer every fixed layout slot is measured from,
    exactly as in `Clif.bindAt`: a bind array that does not sit at `root`
    resolves to `none` rather than to some other base's array. -/
def hstep (fns : List FnDecl) (root : Nat) (P : List HI) (c : HCfg) : Option HCfg :=
  match P[c.pc]? with
  | none => none
  | some (.inst i) =>
      some { c with pc := c.pc + 1
                    env := stepPure c.env i
                    mem := stepMem c.env c.mem i
                    trace := c.trace ++ (launchAt fns c.env i).toList
                    btrace := c.btrace ++
                      (if isLaunchCallB fns i then [bindAt fns root ⟨c.env, c.mem⟩ i] else []) }
  | some (.jmp t) => some { c with pc := t }
  | some (.setC v k) => some { (c.setCtr v k) with pc := c.pc + 1 }
  | some (.incC v) => some { (c.setCtr v (c.ctr v.id + 1)) with pc := c.pc + 1 }
  | some (.jmpLt v bound t f) =>
      some { c with pc := if c.ctr v.id < bound then t else f }

def hsteps (fns : List FnDecl) (root : Nat) (P : List HI) : Nat → HCfg → Option HCfg
  | 0,     c => some c
  | k + 1, c => match hstep fns root P c with
                | none    => none
                | some c' => hsteps fns root P k c'

/-- Fuel composes, so a program's trace is its parts' traces in order.  The
    host analogue of `steps_add`. -/
theorem hsteps_add (fns : List FnDecl) (root : Nat) (P : List HI) :
    ∀ (a b : Nat) (c : HCfg),
      hsteps fns root P (a + b) c
        = match hsteps fns root P a c with
          | none => none
          | some c' => hsteps fns root P b c' := by
  intro a
  induction a with
  | zero => intro b c; simp [hsteps]
  | succ a ih =>
      intro b c
      rw [Nat.succ_add]
      cases h : hstep fns root P c with
      | none => simp [hsteps, h]
      | some c' => simp [hsteps, h, ih b c']

-- ---------------------------------------------------------------------------
-- Compiling: a pure function, emitting a real loop
-- ---------------------------------------------------------------------------

/-- The call itself, in `cudaLaunch`'s calling convention: the kernel and bind
    pointers are `ptr + offset`, the rest constants.  The array those eleven
    instructions point at is filled by `emitBinds`, which runs first;
    `emitLaunch` is the two together. -/
def emitLaunchCall (fnLaunch : FnRef) (ptr : Val) (n : Nat) (s : LaunchStep) :
    Nat × List Inst :=
  let kOff : Val := ⟨n⟩;   let kPtr : Val := ⟨n+1⟩
  let nB   : Val := ⟨n+2⟩; let bOff : Val := ⟨n+3⟩; let bPtr : Val := ⟨n+4⟩
  let gx   : Val := ⟨n+5⟩; let gy   : Val := ⟨n+6⟩; let gz   : Val := ⟨n+7⟩
  let bx   : Val := ⟨n+8⟩; let by_  : Val := ⟨n+9⟩; let bz   : Val := ⟨n+10⟩
  ( n + 11
  , [ .iconst kOff .i64 (Int.ofNat s.ptxOff)
    , .iadd kPtr ptr kOff
    , .iconst nB .i32 (Int.ofNat s.nBufs)
    , .iconst bOff .i64 (Int.ofNat s.bindOff)
    , .iadd bPtr ptr bOff
    , .iconst gx .i32 (Int.ofNat s.gridX)
    , .iconst gy .i32 1
    , .iconst gz .i32 1
    , .iconst bx .i32 (Int.ofNat s.blockX)
    , .iconst by_ .i32 1
    , .iconst bz .i32 1
    , .call none fnLaunch [ptr, kPtr, nB, bPtr, gx, gy, gz, bx, by_, bz] ] )

/-- Materialise one argument: an immediate, or a handle loaded from `ptr + k`.
    Exactly the sequence a generator writes as `slotWq.load ptr`. -/
def emitArg (ptr : Val) (n : Nat) : ExternArg → Nat × List Inst × Val
  | .const c => (n + 1, [Inst.iconst ⟨n⟩ .i64 c], ⟨n⟩)
  | .slot k  => (n + 3, [ Inst.iconst ⟨n⟩ .i64 (Int.ofNat k)
                        , Inst.iadd ⟨n+1⟩ ptr ⟨n⟩
                        , Inst.load ⟨n+2⟩ "load.i64" ⟨n+1⟩ ], ⟨n+2⟩)
  | .far b k => (n + 3, [ Inst.iconst ⟨n⟩ .i64 k
                        , Inst.iadd ⟨n+1⟩ ⟨b⟩ ⟨n⟩
                        , Inst.load ⟨n+2⟩ "load.i64" ⟨n+1⟩ ], ⟨n+2⟩)
  | .addr k  => (n + 2, [ Inst.iconst ⟨n⟩ .i64 (Int.ofNat k)
                        , Inst.iadd ⟨n+1⟩ ptr ⟨n⟩ ], ⟨n+1⟩)
  | .held v _ => (n, [], ⟨v⟩)


def emitArgs (ptr : Val) : Nat → List ExternArg → Nat × List Inst × List Val
  | n, []      => (n, [], [])
  | n, a :: as =>
      let r  := emitArg ptr n a
      let rs := emitArgs ptr r.1 as
      (rs.1, r.2.1 ++ rs.2.1, r.2.2 :: rs.2.2)

/-- **One entry of a bind array**: materialise the handle, then store it at
    `ptr + bindOff + 4i`.  Exactly what `Kernel.launchAt` emits — it writes
    `storeI32 b (← iaddImm ptr (bindOff + i * 4))` — so the fragment this
    model compiles is the fragment the generators compile. -/
def emitBind (ptr : Val) (off : Int) (n : Nat) (a : ExternArg) : Nat × List Inst :=
  let r  := emitArg ptr n a
  let oc : Val := ⟨r.1⟩
  let ad : Val := ⟨r.1 + 1⟩
  ( r.1 + 2
  , r.2.1 ++ [ Inst.iconst oc .i64 off
             , Inst.iadd ad ptr oc
             , Inst.store r.2.2 ad ] )

/-- **What each bind costs.**  `Tensor.Kernel.launchAt` emits three
    instructions per bound buffer — `iconst`, `iadd`, `store` — because the
    handle arrives already in a register; `held` emits the same three, `slot`
    emits six because it reloads the handle first.  Guards, so a change to
    `emitArg` that silently reintroduces the reload fails the build. -/
example (ptr : Val) (off : Int) (n v k : Nat) :
    (emitBind ptr off n (.held v k)).2.length = 3 := rfl

example (ptr : Val) (off : Int) (n k : Nat) :
    (emitBind ptr off n (.slot k)).2.length = 6 := rfl

/-- …and it is the *same* handle: both describe the buffer at `ptr + k`. -/
example (v k : Nat) : (ExternArg.held v k).toBuf = (ExternArg.slot k).toBuf := rfl

/-- The whole array, entry `i` at `bindOff + 4i`. -/
def emitBinds (ptr : Val) (bindOff : Nat) : Nat → Nat → List ExternArg → Nat × List Inst
  | n, _, []      => (n, [])
  | n, i, a :: as =>
      let r  := emitBind ptr (Int.ofNat bindOff + 4 * (i : Int)) n a
      let rs := emitBinds ptr bindOff r.1 (i + 1) as
      (rs.1, r.2 ++ rs.2)

/-- **A launch, whole**: fill the pointer array, then call.  Splitting it this
    way is not cosmetic — `emitLaunchCall`'s eleven instructions are what
    `Clif.scanBlock` turns into the record, and `emitBinds`'s stores are what
    `Clif.bindsAt?` turns into the contents, and the two passes must see the
    same fragment for the pair to mean anything. -/
def emitLaunch (fnLaunch : FnRef) (ptr : Val) (n : Nat) (s : LaunchStep) :
    Nat × List Inst :=
  let b := emitBinds ptr s.bindOff n 0 s.binds
  let c := emitLaunchCall fnLaunch ptr b.1 s
  (c.1, b.2 ++ c.2)

-- Structural facts about the argument fragment.  All three are about *where*
-- it writes and what it is not; the value spec follows once the environment is
-- pinned.

/-- The watermark never moves down.  Stated with `≤`, not `<`, because an
    argument that is *already in scope* emits nothing and moves it not at all —
    which is the whole point of such an argument. -/
theorem emitArg_le (ptr : Val) (n : Nat) (a : ExternArg) :
    n ≤ (emitArg ptr n a).1 := by
  cases a <;> simp [emitArg]

/-- **The materialised value sits below the next watermark.**

    This is the only thing the frame arguments need: everything a *later*
    fragment writes is `≥` the watermark, so a value below it survives.  For
    `held` it is the in-scope hypothesis `v < n` that supplies it. -/
theorem emitArg_valLt {e : Env} (ptr : Val) (n : Nat) (a : ExternArg)
    (hok : a.Ok e ptr n) :
    (emitArg ptr n a).2.2.id < (emitArg ptr n a).1 := by
  cases a with
  | held v k => exact hok.1
  | _ => simp [emitArg]

theorem emitArg_dests (ptr : Val) (n : Nat) (a : ExternArg) :
    ∀ i ∈ (emitArg ptr n a).2.1, ∀ d, Inst.destOf? i = some d →
      n ≤ d.id ∧ d.id < (emitArg ptr n a).1 := by
  intro i hi d hd
  cases a with
  | held v k => simp [emitArg] at hi
  | const c =>
      have hie : i = Inst.iconst ⟨n⟩ .i64 c := by simpa [emitArg] using hi
      subst hie
      have : d = (⟨n⟩ : Val) := by simpa [Inst.destOf?] using hd.symm
      subst this
      exact ⟨Nat.le_refl n, by simp [emitArg]⟩
  | slot k =>
      simp only [emitArg, List.mem_cons, List.not_mem_nil, or_false] at hi
      rcases hi with h | h | h <;> subst h <;>
        (simp only [Inst.destOf?, Option.some.injEq] at hd; subst hd;
         exact ⟨by simp, by simp [emitArg]⟩)
  | far b k =>
      simp only [emitArg, List.mem_cons, List.not_mem_nil, or_false] at hi
      rcases hi with h | h | h <;> subst h <;>
        (simp only [Inst.destOf?, Option.some.injEq] at hd; subst hd;
         exact ⟨by simp, by simp [emitArg]⟩)
  | addr k =>
      simp only [emitArg, List.mem_cons, List.not_mem_nil, or_false] at hi
      rcases hi with h | h <;> subst h <;>
        (simp only [Inst.destOf?, Option.some.injEq] at hd; subst hd;
         exact ⟨by simp, by simp [emitArg]⟩)

theorem emitArg_noCalls (ptr : Val) (n : Nat) (a : ExternArg) :
    ∀ i ∈ (emitArg ptr n a).2.1, Inst.isCallB i = false := by
  intro i hi
  cases a with
  | held v k => simp [emitArg] at hi
  | const c => have : i = Inst.iconst ⟨n⟩ .i64 c := by simpa [emitArg] using hi
               subst this; rfl
  | slot k => simp only [emitArg, List.mem_cons, List.not_mem_nil, or_false] at hi
              rcases hi with h | h | h <;> subst h <;> rfl
  | far b k => simp only [emitArg, List.mem_cons, List.not_mem_nil, or_false] at hi
               rcases hi with h | h | h <;> subst h <;> rfl
  | addr k => simp only [emitArg, List.mem_cons, List.not_mem_nil, or_false] at hi
              rcases hi with h | h <;> subst h <;> rfl

theorem emitArgs_props (ptr : Val) : ∀ (as : List ExternArg) (n : Nat),
    n ≤ (emitArgs ptr n as).1
    ∧ (∀ i ∈ (emitArgs ptr n as).2.1, ∀ d, Inst.destOf? i = some d →
         n ≤ d.id ∧ d.id < (emitArgs ptr n as).1)
    ∧ (∀ i ∈ (emitArgs ptr n as).2.1, Inst.isCallB i = false) := by
  intro as
  induction as with
  | nil => intro n; exact ⟨Nat.le_refl n, by simp [emitArgs], by simp [emitArgs]⟩
  | cons a as ih =>
      intro n
      obtain ⟨hm, hd, hc⟩ := ih (emitArg ptr n a).1
      have h1 := emitArg_le ptr n a
      refine ⟨Nat.le_trans h1 hm, ?_, ?_⟩
      · intro i hi d hdd
        show n ≤ d.id ∧ d.id < (emitArgs ptr (emitArg ptr n a).1 as).1
        rcases List.mem_append.mp hi with h | h
        · obtain ⟨l, r⟩ := emitArg_dests ptr n a i h d hdd
          exact ⟨l, Nat.lt_of_lt_of_le r hm⟩
        · obtain ⟨l, r⟩ := hd i h d hdd
          exact ⟨Nat.le_trans h1 l, r⟩
      · intro i hi
        rcases List.mem_append.mp hi with h | h
        · exact emitArg_noCalls ptr n a i h
        · exact hc i h

/-- The base pointer survives argument materialisation. -/
theorem emitArgs_frameAt (ptr : Val) (as : List ExternArg) (n : Nat) (e : Env)
    (w : Val) (hw : w.id < n) : evalPure e (emitArgs ptr n as).2.1 w = e w :=
  evalPure_frame _ e w
    (fun i hi d hd => by
      have := (emitArgs_props ptr as n).2.1 i hi d hd
      omega)

theorem emitArgs_frame (ptr : Val) (as : List ExternArg) (n : Nat) (e : Env)
    (hptr : ptr.id < n) : evalPure e (emitArgs ptr n as).2.1 ptr = e ptr :=
  evalPure_frame _ e ptr
    (fun i hi d hd => by
      have := (emitArgs_props ptr as n).2.1 i hi d hd
      omega)

/-- **Each materialised argument reads back as the handle it was declared to
    be** — an immediate as that immediate, a handle as the slot it came from,
    *and the base it came off*.  Stated at `SymVal` rather than at `ArgDesc`
    because the two consumers want different views of the same fact: the
    record pass forgets the base, the bind pass keeps it. -/
theorem emitArgs_sym (ptr : Val) : ∀ (as : List ExternArg) (n : Nat) (e : Env),
    FarOk e ptr n as → ptr.id < n → e ptr = SymVal.unknown →
    ((emitArgs ptr n as).2.2.map (fun v => evalPure e (emitArgs ptr n as).2.1 v))
      = as.map (ExternArg.toSym ptr) := by
  intro as
  induction as with
  | nil => intro _ _ _ _ _; rfl
  | cons a as ih =>
      intro n e hff hptr he
      have h1 := emitArg_le ptr n a
      have hv2 := emitArg_valLt ptr n a hff.headOk
      have hfr : evalPure e (emitArg ptr n a).2.1 ptr = e ptr :=
        evalPure_frame _ e ptr
          (fun i hi d hd => by have := emitArg_dests ptr n a i hi d hd; omega)
      -- the head argument, unperturbed by the tail
      have hhead : evalPure e ((emitArg ptr n a).2.1 ++ (emitArgs ptr (emitArg ptr n a).1 as).2.1)
                      (emitArg ptr n a).2.2
                  = evalPure e (emitArg ptr n a).2.1 (emitArg ptr n a).2.2 := by
        rw [evalPure_append]
        exact evalPure_frame _ _ _
          (fun i hi d hd => by
            have := (emitArgs_props ptr as (emitArg ptr n a).1).2.1 i hi d hd
            omega)
      have hhv : evalPure e (emitArg ptr n a).2.1 (emitArg ptr n a).2.2
                  = a.toSym ptr := by
        cases a with
        | held v k => exact hff.headOk.2
        | const c => simp [emitArg, evalPure, stepPure, Env.set_apply, ExternArg.toSym]
        | slot k =>
            have hp0 : ¬ (ptr.id = n) := by omega
            simp [emitArg, evalPure, stepPure, Env.set_apply, addSym,
                  ExternArg.toSym, he, hp0]
        | far b k =>
            obtain ⟨hb1, hb2, hb3⟩ := hff.head b rfl
            have hbn : ¬ ((⟨b⟩ : Val).id = n) := by simp; omega
            simp [emitArg, evalPure, stepPure, Env.set_apply, addSym,
                  ExternArg.toSym, hb3, hbn]
        | addr k =>
            have hp0 : ¬ (ptr.id = n) := by omega
            simp [emitArg, evalPure, stepPure, Env.set_apply, addSym,
                  ExternArg.toSym, he, hp0]
      -- the tail, under the environment the head leaves behind
      have htail := ih (emitArg ptr n a).1 (evalPure e (emitArg ptr n a).2.1)
                      (hff.tail.mono h1 (fun w hw =>
                        evalPure_frame _ e w (fun i hi d hd => by
                          have := emitArg_dests ptr n a i hi d hd; omega)))
                      (by omega) (by rw [hfr]; exact he)
      rw [show (emitArgs ptr n (a :: as)).2.1
                 = (emitArg ptr n a).2.1 ++ (emitArgs ptr (emitArg ptr n a).1 as).2.1
               from rfl,
          show (emitArgs ptr n (a :: as)).2.2
                 = (emitArg ptr n a).2.2 :: (emitArgs ptr (emitArg ptr n a).1 as).2.2
               from rfl]
      simp only [List.map_cons]
      rw [hhead, hhv]
      congr 1
      rw [← htail]
      apply List.map_congr_left
      intro v _
      rw [evalPure_append]

/-- The record pass's view: what `LaunchRec.args` records. -/
theorem emitArgs_desc (ptr : Val) (as : List ExternArg) (n : Nat) (e : Env)
    (hff : FarOk e ptr n as)
    (hptr : ptr.id < n) (he : e ptr = SymVal.unknown) :
    ((emitArgs ptr n as).2.2.map
        (fun v => descOf (evalPure e (emitArgs ptr n as).2.1 v)))
      = as.map ExternArg.toDesc := by
  calc ((emitArgs ptr n as).2.2.map
          (fun v => descOf (evalPure e (emitArgs ptr n as).2.1 v)))
      = ((emitArgs ptr n as).2.2.map
          (fun v => evalPure e (emitArgs ptr n as).2.1 v)).map descOf := by
        rw [List.map_map]; rfl
    _ = (as.map (ExternArg.toSym ptr)).map descOf := by
        rw [emitArgs_sym ptr as n e hff hptr he]
    _ = as.map ExternArg.toDesc := by
        rw [List.map_map]; exact List.map_congr_left (fun a _ => by cases a <;> rfl)

/-- The bind pass's view: what `Clif.bindsOf` recovers, base included. -/
theorem emitArgs_buf (ptr : Val) (as : List ExternArg) (n : Nat) (e : Env)
    (hff : FarOk e ptr n as)
    (hptr : ptr.id < n) (he : e ptr = SymVal.unknown) :
    ((emitArgs ptr n as).2.2.map
        (fun v => bufDescOf ptr.id (evalPure e (emitArgs ptr n as).2.1 v)))
      = as.map ExternArg.toBuf := by
  calc ((emitArgs ptr n as).2.2.map
          (fun v => bufDescOf ptr.id (evalPure e (emitArgs ptr n as).2.1 v)))
      = ((emitArgs ptr n as).2.2.map
          (fun v => evalPure e (emitArgs ptr n as).2.1 v)).map (bufDescOf ptr.id) := by
        rw [List.map_map]; rfl
    _ = (as.map (ExternArg.toSym ptr)).map (bufDescOf ptr.id) := by
        rw [emitArgs_sym ptr as n e hff hptr he]
    _ = as.map ExternArg.toBuf := by
        rw [List.map_map]
        exact List.map_congr_left (fun a ha =>
          ExternArg.bufDescOf_toSym ptr a (fun b k hbk =>
            (hff.base a ha b (by rw [hbk]; rfl)).2.1))

-- ---------------------------------------------------------------------------
-- The bind fragment: what the host puts in the pointer array
-- ---------------------------------------------------------------------------

/-! Everything above is about *values*: which handle reached which argument.
    The bind array is about *memory*: the handles are stored, and the launch
    call names only the address they were stored at.  So these lemmas track
    `Clif.StoreMap` the way `emitArgs_desc` tracks `Env` — and the pay-off is
    that a `LaunchStep`'s declared `binds` become something `Clif.bindsOf`
    reads back off the emitted stores rather than something supplied
    alongside them. -/

theorem emitBind_next (ptr : Val) (off : Int) (n : Nat) (a : ExternArg) :
    (emitBind ptr off n a).1 = (emitArg ptr n a).1 + 2 := rfl

theorem emitBind_props (ptr : Val) (off : Int) (n : Nat) (a : ExternArg) :
    n < (emitBind ptr off n a).1
    ∧ (∀ i ∈ (emitBind ptr off n a).2, ∀ d, Inst.destOf? i = some d →
         n ≤ d.id ∧ d.id < (emitBind ptr off n a).1)
    ∧ (∀ i ∈ (emitBind ptr off n a).2, Inst.isCallB i = false) := by
  have h1 := emitArg_le ptr n a
  refine ⟨by rw [emitBind_next]; omega, ?_, ?_⟩
  · intro i hi d hd
    rw [emitBind_next]
    rcases List.mem_append.mp hi with h | h
    · obtain ⟨l, r⟩ := emitArg_dests ptr n a i h d hd
      exact ⟨l, by omega⟩
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at h
      rcases h with rfl | rfl | rfl
      · simp only [Inst.destOf?, Option.some.injEq] at hd; subst hd
        exact ⟨by show n ≤ (emitArg ptr n a).1; omega,
               by show (emitArg ptr n a).1 < (emitArg ptr n a).1 + 2; omega⟩
      · simp only [Inst.destOf?, Option.some.injEq] at hd; subst hd
        exact ⟨by show n ≤ (emitArg ptr n a).1 + 1; omega,
               by show (emitArg ptr n a).1 + 1 < (emitArg ptr n a).1 + 2; omega⟩
      · simp [Inst.destOf?] at hd
  · intro i hi
    rcases List.mem_append.mp hi with h | h
    · exact emitArg_noCalls ptr n a i h
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at h
      rcases h with rfl | rfl | rfl <;> rfl

theorem emitBinds_props (ptr : Val) (bindOff : Nat) :
    ∀ (as : List ExternArg) (n i : Nat),
      n ≤ (emitBinds ptr bindOff n i as).1
      ∧ (∀ x ∈ (emitBinds ptr bindOff n i as).2, ∀ d, Inst.destOf? x = some d →
           n ≤ d.id ∧ d.id < (emitBinds ptr bindOff n i as).1)
      ∧ (∀ x ∈ (emitBinds ptr bindOff n i as).2, Inst.isCallB x = false) := by
  intro as
  induction as with
  | nil => intro n i; exact ⟨Nat.le_refl n, by simp [emitBinds], by simp [emitBinds]⟩
  | cons a as ih =>
      intro n i
      obtain ⟨hm, hd, hc⟩ :=
        ih (emitBind ptr (Int.ofNat bindOff + 4 * (i : Int)) n a).1 (i + 1)
      obtain ⟨h1, hdb, hcb⟩ := emitBind_props ptr (Int.ofNat bindOff + 4 * (i : Int)) n a
      refine ⟨Nat.le_trans (Nat.le_of_lt h1) hm, ?_, ?_⟩
      · intro x hx d hdd
        show n ≤ d.id ∧ d.id
              < (emitBinds ptr bindOff
                  (emitBind ptr (Int.ofNat bindOff + 4 * (i : Int)) n a).1 (i + 1) as).1
        rcases List.mem_append.mp hx with h | h
        · obtain ⟨l, r⟩ := hdb x h d hdd
          exact ⟨l, Nat.lt_of_lt_of_le r hm⟩
        · obtain ⟨l, r⟩ := hd x h d hdd
          exact ⟨Nat.le_trans (Nat.le_of_lt h1) l, r⟩
      · intro x hx
        rcases List.mem_append.mp hx with h | h
        · exact hcb x h
        · exact hc x h

/-- The base pointer survives the bind fragment. -/
theorem emitBinds_frame (ptr : Val) (bindOff : Nat) (as : List ExternArg) (n i : Nat)
    (e : Env) (w : Val) (hw : w.id < n) :
    evalPure e (emitBinds ptr bindOff n i as).2 w = e w :=
  evalPure_frame _ e w (fun x hx d hd => by
    have := (emitBinds_props ptr bindOff as n i).2.1 x hx d hd
    omega)

/-- **The store map an emitted bind array leaves**: one entry per declared
    handle, most recent first, on top of whatever was there before.  Stated as
    a term rather than a property because everything downstream needs to look
    entries *up*, and a conservative `stepMem` makes the exact shape the only
    thing strong enough to do that with. -/
def bindMap (ptr : Val) (bindOff : Nat) : Nat → List ExternArg → StoreMap
  | _, []      => []
  | i, a :: as => bindMap ptr bindOff (i + 1) as
                    ++ [((ptr.id, Int.ofNat bindOff + 4 * (i : Int)), a.toSym ptr)]

theorem emitBind_mem (ptr : Val) (off : Int) (n : Nat) (a : ExternArg)
    (e : Env) (m : StoreMap) (hff : FarOk e ptr n [a])
    (hptr : ptr.id < n) (he : e ptr = SymVal.unknown) :
    (bevalPure ⟨e, m⟩ (emitBind ptr off n a).2).mem
      = ((ptr.id, off), a.toSym ptr) :: m := by
  have hpk : ∀ k : Nat, ((ptr.id = n + k) = False) := by
    intro k; simp only [eq_iff_iff, iff_false]; omega
  have hpz : (ptr.id = n) = False := by simp only [eq_iff_iff, iff_false]; omega
  cases a with
  | held v k =>
      obtain ⟨hv1, hv2⟩ := hff.headOk
      have hvn : ∀ j : Nat, ((⟨v⟩ : Val).id = n + j) = False := by
        intro j; show (v = n + j) = False
        simp only [eq_iff_iff, iff_false]; omega
      have hvz : ¬ v = n := by omega
      show (bevalPure ⟨e, m⟩
              [ Inst.iconst ⟨n⟩ .i64 off
              , Inst.iadd ⟨n + 1⟩ ptr ⟨n⟩
              , Inst.store ⟨v⟩ ⟨n + 1⟩ ]).mem = _
      simp [bevalPure, bstep, stepPure, stepMem, Inst.storeOf?, Env.set_apply,
            addSym, he, hv2, ExternArg.toSym, hpz, hvn, hvz]
  | const c =>
      show (bevalPure ⟨e, m⟩
              [ Inst.iconst ⟨n⟩ .i64 c
              , Inst.iconst ⟨n + 1⟩ .i64 off
              , Inst.iadd ⟨n + 2⟩ ptr ⟨n + 1⟩
              , Inst.store ⟨n⟩ ⟨n + 2⟩ ]).mem = _
      simp [bevalPure, bstep, stepPure, stepMem, Inst.storeOf?, Env.set_apply,
            addSym, he, ExternArg.toSym, hpz, hpk]
  | far b k =>
      obtain ⟨hb1, hb2, hb3⟩ := hff.head b rfl
      have hbn : ∀ j : Nat, ((⟨b⟩ : Val).id = n + j) = False := by
        intro j
        show (b = n + j) = False
        simp only [eq_iff_iff, iff_false]; omega
      have hbz : (b = n) = False := by simp only [eq_iff_iff, iff_false]; omega
      show (bevalPure ⟨e, m⟩
              [ Inst.iconst ⟨n⟩ .i64 k
              , Inst.iadd ⟨n + 1⟩ ⟨b⟩ ⟨n⟩
              , Inst.load ⟨n + 2⟩ "load.i64" ⟨n + 1⟩
              , Inst.iconst ⟨n + 3⟩ .i64 off
              , Inst.iadd ⟨n + 4⟩ ptr ⟨n + 3⟩
              , Inst.store ⟨n + 2⟩ ⟨n + 4⟩ ]).mem = _
      simp [bevalPure, bstep, stepPure, stepMem, Inst.storeOf?, Env.set_apply,
            addSym, he, hb3, ExternArg.toSym, hpz, hpk, hbn, hbz]
  | addr k =>
      show (bevalPure ⟨e, m⟩
              [ Inst.iconst ⟨n⟩ .i64 (Int.ofNat k)
              , Inst.iadd ⟨n + 1⟩ ptr ⟨n⟩
              , Inst.iconst ⟨n + 2⟩ .i64 off
              , Inst.iadd ⟨n + 3⟩ ptr ⟨n + 2⟩
              , Inst.store ⟨n + 1⟩ ⟨n + 3⟩ ]).mem = _
      simp [bevalPure, bstep, stepPure, stepMem, Inst.storeOf?, Env.set_apply,
            addSym, he, ExternArg.toSym, hpz, hpk]
  | slot k =>
      show (bevalPure ⟨e, m⟩
              [ Inst.iconst ⟨n⟩ .i64 (Int.ofNat k)
              , Inst.iadd ⟨n + 1⟩ ptr ⟨n⟩
              , Inst.load ⟨n + 2⟩ "load.i64" ⟨n + 1⟩
              , Inst.iconst ⟨n + 3⟩ .i64 off
              , Inst.iadd ⟨n + 4⟩ ptr ⟨n + 3⟩
              , Inst.store ⟨n + 2⟩ ⟨n + 4⟩ ]).mem = _
      simp [bevalPure, bstep, stepPure, stepMem, Inst.storeOf?, Env.set_apply,
            addSym, he, ExternArg.toSym, hpz, hpk]

theorem emitBinds_mem (ptr : Val) (bindOff : Nat) :
    ∀ (as : List ExternArg) (n i : Nat) (e : Env) (m : StoreMap),
      FarOk e ptr n as → ptr.id < n → e ptr = SymVal.unknown →
      (bevalPure ⟨e, m⟩ (emitBinds ptr bindOff n i as).2).mem
        = bindMap ptr bindOff i as ++ m := by
  intro as
  induction as with
  | nil => intro n i e m _ _ _; rfl
  | cons a as ih =>
      intro n i e m hff hptr he
      have h1 := (emitBind_props ptr (Int.ofNat bindOff + 4 * (i : Int)) n a).1
      have hfft : FarOk (bevalPure ⟨e, m⟩
            (emitBind ptr (Int.ofNat bindOff + 4 * (i : Int)) n a).2).env ptr
            (emitBind ptr (Int.ofNat bindOff + 4 * (i : Int)) n a).1 as :=
        hff.tail.mono (Nat.le_of_lt h1) (fun w hw => by
          rw [bevalPure_env]
          exact evalPure_frame _ e w (fun x hx d hd => by
            have := (emitBind_props ptr (Int.ofNat bindOff + 4 * (i : Int)) n a).2.1 x hx d hd
            omega))
      have hframe : evalPure e (emitBind ptr (Int.ofNat bindOff + 4 * (i : Int)) n a).2 ptr
                      = SymVal.unknown := by
        rw [evalPure_frame _ e ptr (fun x hx d hd => by
              have := (emitBind_props ptr (Int.ofNat bindOff + 4 * (i : Int)) n a).2.1 x hx d hd
              omega)]
        exact he
      show (bevalPure ⟨e, m⟩
              ((emitBind ptr (Int.ofNat bindOff + 4 * (i : Int)) n a).2
                ++ (emitBinds ptr bindOff
                      (emitBind ptr (Int.ofNat bindOff + 4 * (i : Int)) n a).1 (i + 1) as).2)).mem
          = _
      rw [bevalPure_append,
          ih (emitBind ptr (Int.ofNat bindOff + 4 * (i : Int)) n a).1 (i + 1)
             (bevalPure ⟨e, m⟩ (emitBind ptr (Int.ofNat bindOff + 4 * (i : Int)) n a).2).env
             (bevalPure ⟨e, m⟩ (emitBind ptr (Int.ofNat bindOff + 4 * (i : Int)) n a).2).mem
             hfft (by omega) (by rw [bevalPure_env]; exact hframe),
          emitBind_mem ptr (Int.ofNat bindOff + 4 * (i : Int)) n a e m
            (fun x hx => by
              simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
              subst hx; exact hff _ (List.mem_cons_self ..)) hptr he]
      show bindMap ptr bindOff (i + 1) as
              ++ (((ptr.id, Int.ofNat bindOff + 4 * (i : Int)), a.toSym ptr) :: m)
          = (bindMap ptr bindOff (i + 1) as
              ++ [((ptr.id, Int.ofNat bindOff + 4 * (i : Int)), a.toSym ptr)]) ++ m
      rw [List.append_assoc]
      rfl

/-- Entries the fragment wrote *after* index `j` carry different keys, so they
    do not shadow entry `j`. -/
theorem bindMap_keys (ptr : Val) (bindOff : Nat) :
    ∀ (as : List ExternArg) (i j : Nat), j < i →
      ∀ x ∈ bindMap ptr bindOff i as,
        ¬ (x.1.1 = ptr.id ∧ x.1.2 = Int.ofNat bindOff + 4 * (j : Int)) := by
  intro as
  induction as with
  | nil => intro _ _ _ x hx; cases hx
  | cons a as ih =>
      intro i j hj x hx
      rcases List.mem_append.mp hx with h | h
      · exact ih (i + 1) j (by omega) x h
      · simp only [List.mem_cons, List.not_mem_nil, or_false] at h
        subst h
        intro hcon
        have h2 : (Int.ofNat bindOff + 4 * (i : Int))
                    = Int.ofNat bindOff + 4 * (j : Int) := hcon.2
        omega

/-- **The array reads back as the array that was declared.** -/
theorem bindsFrom_bindMap (ptr : Val) (bindOff : Nat) :
    ∀ (as : List ExternArg) (i : Nat) (m : StoreMap),
      (∀ a ∈ as, ∀ b, a.baseOf? = some b → b ≠ ptr.id) →
      bindsFrom (bindMap ptr bindOff i as ++ m) ptr.id (Int.ofNat bindOff) as.length i
        = some (as.map ExternArg.toBuf) := by
  intro as
  induction as with
  | nil => intro _ _ _; rfl
  | cons a as ih =>
      intro i m hff
      have hffa := hff a (List.mem_cons_self ..)
      have hfft : ∀ x ∈ as, ∀ b, x.baseOf? = some b → b ≠ ptr.id :=
        fun x hx => hff x (List.mem_cons_of_mem a hx)
      have hassoc : bindMap ptr bindOff i (a :: as) ++ m
                      = bindMap ptr bindOff (i + 1) as
                        ++ (((ptr.id, Int.ofNat bindOff + 4 * (i : Int)), a.toSym ptr) :: m) := by
        show (bindMap ptr bindOff (i + 1) as ++ [_]) ++ m = _
        rw [List.append_assoc]; rfl
      have hget : StoreMap.get? (bindMap ptr bindOff i (a :: as) ++ m) ptr.id
                    (Int.ofNat bindOff + 4 * (i : Int)) = some (a.toSym ptr) := by
        rw [hassoc, StoreMap.get?_append_left _ _ _ _
              (bindMap_keys ptr bindOff as (i + 1) i (by omega))]
        show (if ptr.id = ptr.id ∧ (Int.ofNat bindOff + 4 * (i : Int))
                  = (Int.ofNat bindOff + 4 * (i : Int))
              then some (a.toSym ptr) else _) = _
        rw [if_pos ⟨rfl, rfl⟩]
      show bindsFrom (bindMap ptr bindOff i (a :: as) ++ m) ptr.id
              (Int.ofNat bindOff) (as.length + 1) i = _
      rw [bindsFrom_succ, hget]
      show ((bindsFrom (bindMap ptr bindOff i (a :: as) ++ m) ptr.id
              (Int.ofNat bindOff) as.length (i + 1)).map
              (bufDescOf ptr.id (a.toSym ptr) :: ·)) = _
      rw [hassoc, ih (i + 1) _ hfft,
          ExternArg.bufDescOf_toSym ptr a (fun b k hbk =>
            hffa b (by rw [hbk]; rfl))]
      rfl

theorem emitBinds_recovers (ptr : Val) (bindOff : Nat) (as : List ExternArg)
    (n : Nat) (e : Env) (m : StoreMap) (hff : FarOk e ptr n as)
    (hptr : ptr.id < n) (he : e ptr = SymVal.unknown) :
    bindsAt? (bevalPure ⟨e, m⟩ (emitBinds ptr bindOff n 0 as).2).mem ptr.id
        (Int.ofNat bindOff) as.length
      = some (as.map ExternArg.toBuf) := by
  rw [show bindsAt? (bevalPure ⟨e, m⟩ (emitBinds ptr bindOff n 0 as).2).mem ptr.id
            (Int.ofNat bindOff) as.length
          = bindsFrom (bevalPure ⟨e, m⟩ (emitBinds ptr bindOff n 0 as).2).mem ptr.id
              (Int.ofNat bindOff) as.length 0 from rfl,
      emitBinds_mem ptr bindOff as n 0 e m hff hptr he]
  exact bindsFrom_bindMap ptr bindOff as 0 m (fun x hx b hb => (hff.base x hx b hb).2.1)

/-- **Compile to a flat program.**  Pure — the SSA counter is threaded
    explicitly rather than hidden in a monad, which is what makes the output a
    term rather than the effect of running one.

    `forN` emits a genuine loop: counter init, guard, body, increment, back
    edge.  The body appears **once**, exactly as the current `IRBuilder`
    generator emits it.  Nothing is unrolled; the *trace* is what repeats. -/
def flatHI (fnLaunch : FnRef) (ptr : Val) : Nat → Nat → HStmt → Nat × List HI
  | n, _, .skip     => (n, [])
  | n, _, .prim is  => (n, is.map HI.inst)
  | n, _, .extern es =>
      let r := emitArgs ptr n es.argv
      (r.1, (r.2.1 ++ [Inst.call none es.fn r.2.2]).map HI.inst)
  | n, _, .launch s => let r := emitLaunch fnLaunch ptr n s; (r.1, r.2.map HI.inst)
  | n, p, .call b   => flatHI fnLaunch ptr n p b
  | n, p, .seq a b  =>
      let r₁ := flatHI fnLaunch ptr n p a
      let r₂ := flatHI fnLaunch ptr r₁.1 (p + r₁.2.length) b
      (r₂.1, r₁.2 ++ r₂.2)
  | n, p, .forN k b =>
      let c : Val := ⟨n⟩
      let r := flatHI fnLaunch ptr (n + 1) (p + 2) b
      ( r.1
      , [ HI.setC c 0
        , HI.jmpLt c k (p + 2) (p + r.2.length + 4) ]
        ++ r.2
        ++ [ HI.incC c, HI.jmp (p + 1) ] )

/-- Shorthands, so statements name the fragment rather than a projection. -/
def code (fnLaunch : FnRef) (ptr : Val) (n p : Nat) (s : HStmt) : List HI :=
  (flatHI fnLaunch ptr n p s).2

def nextId (fnLaunch : FnRef) (ptr : Val) (n p : Nat) (s : HStmt) : Nat :=
  (flatHI fnLaunch ptr n p s).1

/-- The compiled fragment really sits at `p` inside the whole program `P`.
    Stated as a lookup condition rather than an equation so a fragment can be
    reasoned about without knowing what surrounds it. -/
def Fits (P : List HI) (p : Nat) (L : List HI) : Prop :=
  ∀ j, j < L.length → P[p + j]? = L[j]?

theorem Fits.left {P : List HI} {p : Nat} {L₁ L₂ : List HI}
    (h : Fits P p (L₁ ++ L₂)) : Fits P p L₁ := by
  intro j hj
  have hj' : j < (L₁ ++ L₂).length := by rw [List.length_append]; omega
  rw [h j hj', List.getElem?_append_left hj]

theorem Fits.right {P : List HI} {p : Nat} {L₁ L₂ : List HI}
    (h : Fits P p (L₁ ++ L₂)) : Fits P (p + L₁.length) L₂ := by
  intro j hj
  have hj' : L₁.length + j < (L₁ ++ L₂).length := by rw [List.length_append]; omega
  have := h (L₁.length + j) hj'
  rw [show p + (L₁.length + j) = p + L₁.length + j by omega] at this
  rw [this, List.getElem?_append_right (by omega)]
  congr 1
  omega

-- ---------------------------------------------------------------------------
-- Straight-line runs
-- ---------------------------------------------------------------------------

/-- **A run of ordinary instructions advances the pc past them, evolves the
    environment by `evalPure`, and appends exactly what `scanBlock` recovers.**

    This is the bridge between the trace machine and `Clif.lean`: the trace the
    *machine* accumulates over a launch-free-or-not straight line is the list
    the *static scan* of that same straight line produces.  Control flow is what
    makes the two diverge, and control flow is handled separately below. -/
theorem hsteps_insts (fns : List FnDecl) (root : Nat) (P : List HI) :
    ∀ (is : List Inst) (p : Nat) (e : Env) (sm : StoreMap) (ct : Nat → Nat)
      (tr : List LaunchRec) (btr : List OpBinds),
      Fits P p (is.map HI.inst) →
      hsteps fns root P is.length ⟨p, e, sm, ct, tr, btr⟩
        = some ⟨p + is.length, evalPure e is, (bevalPure ⟨e, sm⟩ is).mem, ct,
                tr ++ (scanBlock fns e is).2,
                btr ++ (bindScan fns root ⟨e, sm⟩ is).2⟩ := by
  intro is
  induction is with
  | nil =>
      intro p e sm ct tr btr _
      simp [hsteps, evalPure, scanBlock, bevalPure, bindScan]
  | cons i is ih =>
      intro p e sm ct tr btr h
      have h0 : P[p]? = some (HI.inst i) := by
        have := h 0 (by simp)
        simpa using this
      have hf : hstep fns root P ⟨p, e, sm, ct, tr, btr⟩
          = some ⟨p + 1, stepPure e i, stepMem e sm i, ct,
                  tr ++ (launchAt fns e i).toList,
                  btr ++ (if isLaunchCallB fns i then [bindAt fns root ⟨e, sm⟩ i] else [])⟩ := by
        show (match P[p]? with
              | none => none
              | some (.inst i) => _
              | some (.jmp t) => _
              | some (.setC v k) => _
              | some (.incC v) => _
              | some (.jmpLt v b t f) => _) = _
        rw [h0]
      have h' : Fits P (p + 1) (is.map HI.inst) := by
        intro j hj
        have := h (j + 1) (by simpa using Nat.succ_lt_succ hj)
        rw [show p + (j + 1) = p + 1 + j by omega] at this
        simpa using this
      show (match hstep fns root P ⟨p, e, sm, ct, tr, btr⟩ with
            | none => none
            | some c' => hsteps fns root P is.length c') = _
      rw [hf]
      show hsteps fns root P is.length
            ⟨p + 1, stepPure e i, stepMem e sm i, ct, tr ++ (launchAt fns e i).toList,
             btr ++ (if isLaunchCallB fns i then [bindAt fns root ⟨e, sm⟩ i] else [])⟩ = _
      rw [ih (p + 1) (stepPure e i) (stepMem e sm i) ct (tr ++ (launchAt fns e i).toList)
            (btr ++ (if isLaunchCallB fns i then [bindAt fns root ⟨e, sm⟩ i] else [])) h']
      show some (HCfg.mk (p + 1 + is.length) _ _ _ ((tr ++ _) ++ _) ((btr ++ _) ++ _)) = _
      rw [show p + 1 + is.length = p + (is.length + 1) from by omega,
          List.append_assoc, List.append_assoc]
      rfl

-- ---------------------------------------------------------------------------
-- What one launch fragment scans to
-- ---------------------------------------------------------------------------

theorem env_set_ne (e : Env) (v w : Val) (x : SymVal) (h : ¬ (w.id = v.id)) :
    (e.set v x) w = e w := Env.set_ne e v w x h

theorem env_set_eq (e : Env) (v w : Val) (x : SymVal) (h : w.id = v.id) :
    (e.set v x) w = x := Env.set_eq e v w x h

/-- The base pointer survives a launch fragment untouched — it is allocated
    below the fragment's watermark, which is exactly what `ptr.id < n` says. -/
theorem emitLaunchCall_frame (fnLaunch : FnRef) (ptr : Val) (n : Nat) (s : LaunchStep)
    (e : Env) (w : Val) (hw : w.id < n) :
    evalPure e (emitLaunchCall fnLaunch ptr n s).2 w = e w := by
  have hw0 : (w.id = n) = False := by simp only [eq_iff_iff, iff_false]; omega
  have hwk : ∀ a : Nat, ((w.id = n + a) = False) := by
    intro a; simp only [eq_iff_iff, iff_false]; omega
  show evalPure e [_, _, _, _, _, _, _, _, _, _, _, _] w = e w
  simp [evalPure, stepPure, Env.set_apply, hw0, hwk]

/-- **A launch fragment scans to exactly the record it was emitted from.**

    The round trip that makes `LaunchStep` meaningful: the declared step, the
    twelve CLIF instructions, and the record `Clif.launchesOf` reads back out of
    them all agree.  It needs `e ptr = unknown` — the base pointer is a runtime
    input — and `ptr.id < n`, so the fragment's temporaries cannot shadow it. -/
theorem emitLaunchCall_scan (fns : List FnDecl) (fnLaunch : FnRef) (ptr : Val)
    (hfn : fnNameOf fns fnLaunch = some "cl_cuda_launch")
    (n : Nat) (s : LaunchStep) (e : Env) (hptr : ptr.id < n) (he : e ptr = .unknown) :
    (scanBlock fns e (emitLaunchCall fnLaunch ptr n s).2).2 = [s.toRec] := by
  have hz  : ∀ a : Nat, 0 < a → ((n + a = n) = False) := by
    intro a ha; simp only [eq_iff_iff, iff_false]; omega
  have hz' : ∀ a : Nat, 0 < a → ((n = n + a) = False) := by
    intro a ha; simp only [eq_iff_iff, iff_false]; omega
  have hne : ∀ a b : Nat, a ≠ b → ((n + a = n + b) = False) := by
    intro a b h; simp only [eq_iff_iff, iff_false]; omega
  have hpz : (ptr.id = n) = False := by simp only [eq_iff_iff, iff_false]; omega
  have hpk : ∀ a : Nat, ((ptr.id = n + a) = False) := by
    intro a; simp only [eq_iff_iff, iff_false]; omega
  show (scanBlock fns e [_, _, _, _, _, _, _, _, _, _, _, _]).2 = _
  simp [scanBlock, launchAt, stepPure, Env.set_apply, addSym, hfn, LaunchStep.toRec,
        SymVal.offsetOf?, hpz, hpk, he, launchNames]

-- ── The whole launch: array plus call ──────────────────────────────────────

theorem emitLaunch_next (fnLaunch : FnRef) (ptr : Val) (n : Nat) (s : LaunchStep) :
    (emitLaunch fnLaunch ptr n s).1 = (emitBinds ptr s.bindOff n 0 s.binds).1 + 11 := rfl

theorem emitLaunch_le (fnLaunch : FnRef) (ptr : Val) (n : Nat) (s : LaunchStep) :
    n ≤ (emitLaunch fnLaunch ptr n s).1 := by
  have := (emitBinds_props ptr s.bindOff s.binds n 0).1
  rw [emitLaunch_next]; omega

theorem emitLaunch_frame (fnLaunch : FnRef) (ptr : Val) (n : Nat) (s : LaunchStep)
    (e : Env) (w : Val) (hw : w.id < n) :
    evalPure e (emitLaunch fnLaunch ptr n s).2 w = e w := by
  have hb := (emitBinds_props ptr s.bindOff s.binds n 0).1
  show evalPure e ((emitBinds ptr s.bindOff n 0 s.binds).2
          ++ (emitLaunchCall fnLaunch ptr (emitBinds ptr s.bindOff n 0 s.binds).1 s).2) w = e w
  rw [evalPure_append,
      emitLaunchCall_frame fnLaunch ptr _ s _ w (by omega),
      emitBinds_frame ptr s.bindOff s.binds n 0 e w hw]

/-- **A launch fragment scans to exactly the record it was emitted from.**  The
    bind stores in front of the call contribute nothing to `scanBlock` — they
    are not calls — so this is the same fact it always was. -/
theorem emitLaunch_scan (fns : List FnDecl) (fnLaunch : FnRef) (ptr : Val)
    (hfn : fnNameOf fns fnLaunch = some "cl_cuda_launch")
    (n : Nat) (s : LaunchStep) (e : Env) (hptr : ptr.id < n) (he : e ptr = .unknown) :
    (scanBlock fns e (emitLaunch fnLaunch ptr n s).2).2 = [s.toRec] := by
  have hb := (emitBinds_props ptr s.bindOff s.binds n 0).1
  show (scanBlock fns e ((emitBinds ptr s.bindOff n 0 s.binds).2
          ++ (emitLaunchCall fnLaunch ptr (emitBinds ptr s.bindOff n 0 s.binds).1 s).2)).2 = _
  rw [scanBlock_append,
      scanBlock_noCalls fns _ e (fun i hi => (emitBinds_props ptr s.bindOff s.binds n 0).2.2 i hi),
      List.nil_append,
      emitLaunchCall_scan fns fnLaunch ptr hfn _ s _ (by omega)
        (by rw [emitBinds_frame ptr s.bindOff s.binds n 0 e ptr hptr]; exact he)]

/-- **…and its bind array reads back as the array that was declared.**

    This is the seam that was open.  `Clif.bindsOf` recovers a launch's pointer
    array from the stores in front of it; this model emitted no such stores, so
    the composition theorem had to be handed the recovered list.  Now the
    fragment writes them and this reads them back — the same fragment, two
    passes, one answer. -/
theorem emitLaunch_bindScan (fns : List FnDecl) (fnLaunch : FnRef) (ptr : Val)
    (hfn : fnNameOf fns fnLaunch = some "cl_cuda_launch")
    (n : Nat) (s : LaunchStep) (e : Env) (m : StoreMap)
    (hptr : ptr.id < n) (he : e ptr = SymVal.unknown)
    (hwf : s.WellFormedB = true) (hfar : FarOk e ptr n s.binds) :
    (bindScan fns ptr.id ⟨e, m⟩ (emitLaunch fnLaunch ptr n s).2).2 = [s.toBinds] := by
  have hb := (emitBinds_props ptr s.bindOff s.binds n 0).1
  have hnb : s.nBufs = s.binds.length := by simpa [LaunchStep.WellFormedB] using hwf
  -- what the fragment left in memory, and that the pointer survived it
  have hrec := emitBinds_recovers ptr s.bindOff s.binds n e m hfar hptr he
  have henv : (bevalPure ⟨e, m⟩ (emitBinds ptr s.bindOff n 0 s.binds).2).env ptr
                = SymVal.unknown := by
    rw [bevalPure_env, emitBinds_frame ptr s.bindOff s.binds n 0 e ptr hptr]; exact he
  show (bindScan fns ptr.id ⟨e, m⟩ ((emitBinds ptr s.bindOff n 0 s.binds).2
          ++ (emitLaunchCall fnLaunch ptr (emitBinds ptr s.bindOff n 0 s.binds).1 s).2)).2 = _
  rw [bindScan_append,
      bindScan_noCalls fns ptr.id _ _
        (fun i hi => (emitBinds_props ptr s.bindOff s.binds n 0).2.2 i hi),
      List.nil_append, bindScan_state]
  -- the call's own eleven instructions are pure address arithmetic
  have hpz : (ptr.id = (emitBinds ptr s.bindOff n 0 s.binds).1) = False := by
    simp only [eq_iff_iff, iff_false]; omega
  have hpk : ∀ a : Nat, ((ptr.id = (emitBinds ptr s.bindOff n 0 s.binds).1 + a) = False) := by
    intro a; simp only [eq_iff_iff, iff_false]; omega
  show (bindScan fns ptr.id (bevalPure ⟨e, m⟩ (emitBinds ptr s.bindOff n 0 s.binds).2)
          [_, _, _, _, _, _, _, _, _, _, _, _]).2 = _
  simp [bindScan, bindAt, bstep, stepPure, stepMem, Inst.storeOf?, Env.set_apply,
        addSym, isLaunchCallB, hfn, launchNames, positionalLaunchNames,
        SymVal.offsetOf?, hpz, hpk, henv, LaunchStep.toBinds, hnb]
  exact hrec

-- ---------------------------------------------------------------------------
-- Single control-flow transitions
-- ---------------------------------------------------------------------------

theorem hsteps_one (fns : List FnDecl) (root : Nat) (P : List HI) (c : HCfg) :
    hsteps fns root P 1 c = hstep fns root P c := by
  cases h : hstep fns root P c <;> simp [hsteps, h]

/-- Chaining: `a` steps then `b` steps.  The form the compositional cases are
    written in, so a program's run is assembled from its fragments' runs. -/
theorem hsteps_trans (fns : List FnDecl) (root : Nat) (P : List HI) (a b : Nat) (c c' c'' : HCfg)
    (h1 : hsteps fns root P a c = some c') (h2 : hsteps fns root P b c' = some c'') :
    hsteps fns root P (a + b) c = some c'' := by
  rw [hsteps_add, h1]; exact h2

theorem hstep_setC (fns : List FnDecl) (root : Nat) (P : List HI) (q : Nat) (e : Env)
    (sm : StoreMap) (ct : Nat → Nat) (tr : List LaunchRec) (btr : List OpBinds)
    (v : Val) (m : Nat)
    (h : P[q]? = some (HI.setC v m)) :
    hstep fns root P ⟨q, e, sm, ct, tr, btr⟩
      = some ⟨q + 1, e, sm, fun w => if w = v.id then m else ct w, tr, btr⟩ := by
  simp only [hstep, h, HCfg.setCtr]

theorem hstep_incC (fns : List FnDecl) (root : Nat) (P : List HI) (q : Nat) (e : Env)
    (sm : StoreMap) (ct : Nat → Nat) (tr : List LaunchRec) (btr : List OpBinds) (v : Val)
    (h : P[q]? = some (HI.incC v)) :
    hstep fns root P ⟨q, e, sm, ct, tr, btr⟩
      = some ⟨q + 1, e, sm, fun w => if w = v.id then ct v.id + 1 else ct w, tr, btr⟩ := by
  simp only [hstep, h, HCfg.setCtr]

theorem hstep_jmp (fns : List FnDecl) (root : Nat) (P : List HI) (q : Nat) (e : Env)
    (sm : StoreMap) (ct : Nat → Nat) (tr : List LaunchRec) (btr : List OpBinds) (t : Nat)
    (h : P[q]? = some (HI.jmp t)) :
    hstep fns root P ⟨q, e, sm, ct, tr, btr⟩ = some ⟨t, e, sm, ct, tr, btr⟩ := by
  simp only [hstep, h]

theorem hstep_jmpLt (fns : List FnDecl) (root : Nat) (P : List HI) (q : Nat) (e : Env)
    (sm : StoreMap) (ct : Nat → Nat) (tr : List LaunchRec) (btr : List OpBinds)
    (v : Val) (bd t f : Nat)
    (h : P[q]? = some (HI.jmpLt v bd t f)) :
    hstep fns root P ⟨q, e, sm, ct, tr, btr⟩
      = some ⟨if ct v.id < bd then t else f, e, sm, ct, tr, btr⟩ := by
  simp only [hstep, h]

-- ---------------------------------------------------------------------------
-- The compiler allocates upward
-- ---------------------------------------------------------------------------

/-- The SSA watermark never moves down, so a statement's temporaries always sit
    above everything compiled before it — which is what keeps `ptr.id < n` true
    all the way through a sequence. -/
theorem flatHI_mono (fnLaunch : FnRef) (ptr : Val) :
    ∀ (s : HStmt) (n p : Nat), n ≤ nextId fnLaunch ptr n p s := by
  intro s
  induction s with
  | skip => intro n p; exact Nat.le_refl n
  | prim => intro n p; exact Nat.le_refl n
  | extern es => intro n p; exact (emitArgs_props ptr es.argv n).1
  | launch st => intro n p; exact emitLaunch_le fnLaunch ptr n st
  | call b ih => intro n p; exact ih n p
  | seq a b iha ihb =>
      intro n p
      exact Nat.le_trans (iha n p) (ihb _ _)
  | forN _ b ih =>
      intro n p
      exact Nat.le_trans (Nat.le_succ n) (ih (n + 1) (p + 2))

-- ---------------------------------------------------------------------------
-- The trace theorem
-- ---------------------------------------------------------------------------

/-- **Executing the compiled program performs exactly the declared launch
    sequence.**

    The statement that closes the host seam.  `HStmt.launches` is a structural
    recursion — a `forN` contributes its body `n` times because the recursion
    says so.  `hsteps` is a program counter walking a *flat list of real
    instructions* in which the loop body appears **once**, with a counter, a
    guard, an increment and a back edge.  This says the two agree.

    Nothing about the emitted control flow is assumed: the loop's trip count,
    its termination, and the order in which launches accumulate are all
    consequences.  The fuel is existential precisely because a loop executes
    more instructions than it contains — which is the whole reason the code can
    stay a loop instead of being unrolled to make a static scan work.

    Preconditions are the two the compiler's own convention supplies: the base
    pointer is a runtime input (`e ptr = unknown`) allocated below the
    fragment's watermark (`ptr.id < n`). -/
theorem flatHI_sound (fns : List FnDecl) (fnLaunch : FnRef) (ptr : Val)
    (hfn : fnNameOf fns fnLaunch = some "cl_cuda_launch") (P : List HI) :
    ∀ (s : HStmt) (n p : Nat) (e : Env) (sm : StoreMap) (ct : Nat → Nat)
      (tr : List LaunchRec) (btr : List OpBinds),
      ptr.id < n → e ptr = SymVal.unknown → HStmt.TameB fns ptr s = true →
      FarOk e ptr n s.farArgs →
      (∀ x ∈ s.farArgs, ∀ b ∈ x.deps, b ∉ s.primDests) →
      Fits P p (code fnLaunch ptr n p s) →
      ∃ k c', hsteps fns ptr.id P k ⟨p, e, sm, ct, tr, btr⟩ = some c'
        ∧ c'.pc = p + (code fnLaunch ptr n p s).length
        ∧ c'.trace = tr ++ s.launches
        ∧ c'.btrace = btr ++ s.binds
        ∧ (∀ w : Val, w.id < n → w.id ∉ s.primDests → c'.env w = e w)
        ∧ c'.env ptr = SymVal.unknown
        ∧ ∀ w, w < n → c'.ctr w = ct w := by
  intro s
  induction s with
  | skip =>
      intro n p e sm ct tr btr _ he _ _ _ _
      exact ⟨0, ⟨p, e, sm, ct, tr, btr⟩, rfl, rfl, by simp [HStmt.launches],
             by simp [HStmt.binds], fun _ _ _ => rfl, he, fun _ _ => rfl⟩
  | prim is =>
      intro n p e sm ct tr btr _ he htame _ _ hfit
      have hall : ∀ i ∈ is, Inst.TameB fns ptr i = true := by
        simpa [HStmt.TameB, List.all_eq_true] using htame
      have hzero : launchCallCount fns is = 0 :=
        launchCallCount_zero fns is (fun i hi => tame_noLaunch fns ptr i (hall i hi))
      refine ⟨is.length, _, hsteps_insts fns ptr.id P is p e sm ct tr btr hfit,
              ?_, ?_, ?_, ?_, ?_, fun _ _ => rfl⟩
      · show p + is.length = p + (is.map HI.inst).length
        rw [List.length_map]
      · show tr ++ (scanBlock fns e is).2 = tr ++ []
        rw [scanBlock_noLaunch fns is e hzero]
      · show btr ++ (bindScan fns ptr.id ⟨e, sm⟩ is).2 = btr ++ []
        rw [bindScan_noLaunch fns ptr.id is ⟨e, sm⟩ hzero]
      · intro w _ hnd
        show evalPure e is w = e w
        exact evalPure_frame is e w (fun i hi d hd => by
          intro hEq
          exact hnd (List.mem_filterMap.mpr ⟨i, hi, by rw [hd]; simpa using hEq.symm⟩))
      · show evalPure e is ptr = SymVal.unknown
        rw [evalPure_frame is e ptr (fun i hi d hd => tame_noWrite fns ptr i (hall i hi) d hd)]
        exact he
  | launch st =>
      intro n p e sm ct tr btr hptr he htame hfar _ hfit
      have hwf : st.WellFormedB = true := htame
      refine ⟨(emitLaunch fnLaunch ptr n st).2.length, _,
              hsteps_insts fns ptr.id P (emitLaunch fnLaunch ptr n st).2 p e sm ct tr btr hfit,
              ?_, ?_, ?_, ?_, ?_, fun _ _ => rfl⟩
      · show p + (emitLaunch fnLaunch ptr n st).2.length
            = p + ((emitLaunch fnLaunch ptr n st).2.map HI.inst).length
        rw [List.length_map]
      · show tr ++ (scanBlock fns e (emitLaunch fnLaunch ptr n st).2).2 = tr ++ [st.toRec]
        rw [emitLaunch_scan fns fnLaunch ptr hfn n st e hptr he]
      · show btr ++ (bindScan fns ptr.id ⟨e, sm⟩ (emitLaunch fnLaunch ptr n st).2).2
            = btr ++ [st.toBinds]
        rw [emitLaunch_bindScan fns fnLaunch ptr hfn n st e sm hptr he hwf hfar]
      · intro w hw _
        exact emitLaunch_frame fnLaunch ptr n st e w hw
      · show evalPure e (emitLaunch fnLaunch ptr n st).2 ptr = SymVal.unknown
        rw [emitLaunch_frame fnLaunch ptr n st e ptr hptr]; exact he
  | extern es =>
      intro n p e sm ct tr btr hptr he htame hfar _ hfit
      have h3 : ((fnNameOf fns es.fn == some es.name) = true
                ∧ decide (es.name ∈ deviceWriterNames) = true)
                ∧ (!decide (es.name ∈ launchNames)) = true := by
        simp only [HStmt.TameB, Bool.and_eq_true] at htame
        exact ⟨⟨htame.1.1, htame.1.2⟩, htame.2⟩
      have hnm : fnNameOf fns es.fn = some es.name := by simpa using h3.1.1
      have hw  : es.name ∈ deviceWriterNames := by simpa using h3.1.2
      have hl  : es.name ∉ launchNames := by simpa using h3.2
      have hff : FarOk e ptr n es.argv := hfar
      have hpl : es.name ∉ positionalLaunchNames := by
        intro hc
        exact hl (by
          simp only [positionalLaunchNames, List.mem_cons, List.not_mem_nil, or_false] at hc
          rcases hc with h | h <;> simp [launchNames, h])
      refine ⟨((emitArgs ptr n es.argv).2.1
                 ++ [Inst.call none es.fn (emitArgs ptr n es.argv).2.2]).length, _,
              hsteps_insts fns ptr.id P _ p e sm ct tr btr hfit, ?_, ?_, ?_, ?_, ?_,
              fun _ _ => rfl⟩
      · show p + _ = p + (List.map HI.inst _).length
        rw [List.length_map]
      · -- the recovered record is the declared one, arguments included
        show tr ++ (scanBlock fns e ((emitArgs ptr n es.argv).2.1
                      ++ [Inst.call none es.fn (emitArgs ptr n es.argv).2.2])).2
            = tr ++ [es.toRec]
        rw [scanBlock_append,
            scanBlock_noCalls fns _ e (fun i hi => (emitArgs_props ptr es.argv n).2.2 i hi),
            List.nil_append]
        show tr ++ ((launchAt fns (evalPure e (emitArgs ptr n es.argv).2.1)
                      (Inst.call none es.fn (emitArgs ptr n es.argv).2.2)).toList ++ []) = _
        simp only [launchAt, hnm, if_neg hl, if_pos hw, Option.toList,
                   List.append_nil, ExternStep.toRec]
        rw [emitArgs_desc ptr es.argv n e hff hptr he]
      · -- **…and so are its buffers**, base-aware, which is the new half
        show btr ++ (bindScan fns ptr.id ⟨e, sm⟩ ((emitArgs ptr n es.argv).2.1
                      ++ [Inst.call none es.fn (emitArgs ptr n es.argv).2.2])).2
            = btr ++ [es.toBinds]
        rw [bindScan_append,
            bindScan_noLaunch fns ptr.id _ ⟨e, sm⟩
              (launchCallCount_zero fns _ (fun i hi => by
                have := (emitArgs_props ptr es.argv n).2.2 i hi
                cases i <;> simp_all [Inst.isCallB, isLaunchCallB])),
            List.nil_append, bindScan_state]
        show btr ++ ((if isLaunchCallB fns (Inst.call none es.fn (emitArgs ptr n es.argv).2.2)
                      then [bindAt fns ptr.id (bevalPure ⟨e, sm⟩ (emitArgs ptr n es.argv).2.1)
                              (Inst.call none es.fn (emitArgs ptr n es.argv).2.2)]
                      else []) ++ []) = _
        simp only [isLaunchCallB, bindAt, hnm, if_neg hpl, if_pos hw, List.append_nil,
                   decide_eq_true_eq, Bool.or_eq_true, if_pos, ExternStep.toBinds,
                   bevalPure_env]
        rw [if_pos (Or.inr hw)]
        show btr ++ [{ args := (emitArgs ptr n es.argv).2.2.map
                        (fun v => bufDescOf ptr.id (evalPure e (emitArgs ptr n es.argv).2.1 v)) }]
            = _
        rw [emitArgs_buf ptr es.argv n e hff hptr he]
      · intro w hw _
        show evalPure e ((emitArgs ptr n es.argv).2.1
                 ++ [Inst.call none es.fn (emitArgs ptr n es.argv).2.2]) w = e w
        rw [evalPure_append]
        show evalPure e (emitArgs ptr n es.argv).2.1 w = e w
        exact emitArgs_frameAt ptr es.argv n e w hw
      · show evalPure e ((emitArgs ptr n es.argv).2.1
                 ++ [Inst.call none es.fn (emitArgs ptr n es.argv).2.2]) ptr
            = SymVal.unknown
        rw [evalPure_append]
        show evalPure e (emitArgs ptr n es.argv).2.1 ptr = SymVal.unknown
        rw [emitArgs_frame ptr es.argv n e hptr]
        exact he
  | call b ih =>
      intro n p e sm ct tr btr hptr he htame hfar hfd hfit
      exact ih n p e sm ct tr btr hptr he htame hfar hfd hfit
  | seq a b iha ihb =>
      intro n p e sm ct tr btr hptr he htame hfar hfd hfit
      obtain ⟨hta, htb⟩ := Bool.and_eq_true .. ▸ htame
      have hca : code fnLaunch ptr n p (HStmt.seq a b)
          = code fnLaunch ptr n p a
            ++ code fnLaunch ptr (nextId fnLaunch ptr n p a)
                 (p + (code fnLaunch ptr n p a).length) b := rfl
      rw [hca] at hfit
      obtain ⟨k₁, c₁, hr₁, hpc₁, htr₁, hbt₁, hfr₁, hen₁, hct₁⟩ :=
        iha n p e sm ct tr btr hptr he hta
          (fun x hx => hfar x (List.mem_append.mpr (Or.inl hx)))
          (fun x hx b hb hc => hfd x (List.mem_append.mpr (Or.inl hx)) b hb
            (List.mem_append.mpr (Or.inl hc)))
          hfit.left
      obtain ⟨q₁, en₁, sm₁, ctr₁, trc₁, btc₁⟩ := c₁
      subst hpc₁
      have htr₁' : trc₁ = tr ++ a.launches := htr₁
      have hbt₁' : btc₁ = btr ++ a.binds := hbt₁
      have hfr₁' : ∀ w : Val, w.id < n → w.id ∉ a.primDests → en₁ w = e w := hfr₁
      have hen₁' : en₁ ptr = SymVal.unknown := hen₁
      have hct₁' : ∀ w, w < n → ctr₁ w = ct w := hct₁
      obtain ⟨k₂, c₂, hr₂, hpc₂, htr₂, hbt₂, hfr₂, hen₂, hct₂⟩ :=
        ihb (nextId fnLaunch ptr n p a) (p + (code fnLaunch ptr n p a).length)
          en₁ sm₁ ctr₁ trc₁ btc₁
          (Nat.lt_of_lt_of_le hptr (flatHI_mono fnLaunch ptr a n p)) hen₁' htb
          (FarOk.monoD (as := b.farArgs) (ds := a.primDests)
            (fun x hx => hfar x (List.mem_append.mpr (Or.inr hx)))
            (flatHI_mono fnLaunch ptr a n p)
            (fun x hx bb hbb hc => hfd x (List.mem_append.mpr (Or.inr hx)) bb hbb
              (List.mem_append.mpr (Or.inl hc)))
            hfr₁')
          (fun x hx b hb hc => hfd x (List.mem_append.mpr (Or.inr hx)) b hb
            (List.mem_append.mpr (Or.inr hc)))
          hfit.right
      refine ⟨k₁ + k₂, c₂, ?_, ?_, ?_, ?_, ?_, hen₂, ?_⟩
      · rw [hsteps_add, hr₁]; exact hr₂
      · rw [hpc₂, hca, List.length_append]; omega
      · rw [htr₂, htr₁', List.append_assoc]; rfl
      · rw [hbt₂, hbt₁', List.append_assoc]; rfl
      · intro w hw hnd
        rw [hfr₂ w (Nat.lt_of_lt_of_le hw (flatHI_mono fnLaunch ptr a n p))
              (fun hc => hnd (List.mem_append.mpr (Or.inr hc))),
            hfr₁' w hw (fun hc => hnd (List.mem_append.mpr (Or.inl hc)))]
      · intro w hw
        rw [hct₂ w (Nat.lt_of_lt_of_le hw (flatHI_mono fnLaunch ptr a n p)), hct₁' w hw]
  | forN kk b ih =>
      intro n p e sm ct tr btr hptr he htame hfar hfd hfit
      -- name the body's code, so the surrounding shape is readable
      obtain ⟨cb, hcb⟩ : ∃ x, code fnLaunch ptr (n + 1) (p + 2) b = x := ⟨_, rfl⟩
      have hcode : code fnLaunch ptr n p (HStmt.forN kk b)
          = ([HI.setC ⟨n⟩ 0, HI.jmpLt ⟨n⟩ kk (p + 2) (p + cb.length + 4)] ++ cb)
            ++ [HI.incC ⟨n⟩, HI.jmp (p + 1)] := by rw [← hcb]; rfl
      have hlen : (code fnLaunch ptr n p (HStmt.forN kk b)).length = cb.length + 4 := by
        rw [hcode, List.length_append, List.length_append]; simp; omega
      rw [hcode] at hfit
      have hpre : Fits P p ([HI.setC ⟨n⟩ 0, HI.jmpLt ⟨n⟩ kk (p + 2) (p + cb.length + 4)] ++ cb) :=
        hfit.left
      have hP0 : P[p]? = some (HI.setC ⟨n⟩ 0) := by
        have := hpre 0 (by simp); simpa using this
      have hP1 : P[p + 1]? = some (HI.jmpLt ⟨n⟩ kk (p + 2) (p + cb.length + 4)) := by
        have := hpre 1 (by simp); simpa using this
      have hPb : Fits P (p + 2) cb := by
        have := hpre.right
        simpa using this
      have hPi : P[p + 2 + cb.length]? = some (HI.incC ⟨n⟩) := by
        have := hfit (2 + cb.length) (by rw [List.length_append]; simp; omega)
        rw [show p + (2 + cb.length) = p + 2 + cb.length by omega] at this
        rw [this, List.getElem?_append_right (by rw [List.length_append]; simp),
            show ([HI.setC ⟨n⟩ 0, HI.jmpLt ⟨n⟩ kk (p + 2) (p + cb.length + 4)] ++ cb).length
                 = 2 + cb.length by rw [List.length_append]; simp]
        simp
      have hPj : P[p + 2 + cb.length + 1]? = some (HI.jmp (p + 1)) := by
        have := hfit (2 + cb.length + 1) (by rw [List.length_append]; simp; omega)
        rw [show p + (2 + cb.length + 1) = p + 2 + cb.length + 1 by omega] at this
        rw [this, List.getElem?_append_right (by rw [List.length_append]; simp),
            show ([HI.setC ⟨n⟩ 0, HI.jmpLt ⟨n⟩ kk (p + 2) (p + cb.length + 4)] ++ cb).length
                 = 2 + cb.length by rw [List.length_append]; simp]
        simp
      -- the body's specialisation, with its code named
      have ihb : ∀ (e' : Env) (sm' : StoreMap) (ct' : Nat → Nat) (tr' : List LaunchRec)
          (btr' : List OpBinds),
          e' ptr = SymVal.unknown → FarOk e' ptr (n + 1) b.farArgs →
          ∃ j c', hsteps fns ptr.id P j ⟨p + 2, e', sm', ct', tr', btr'⟩ = some c'
            ∧ c'.pc = p + 2 + cb.length
            ∧ c'.trace = tr' ++ b.launches
            ∧ c'.btrace = btr' ++ b.binds
            ∧ (∀ w : Val, w.id < n + 1 → w.id ∉ b.primDests → c'.env w = e' w)
            ∧ c'.env ptr = SymVal.unknown
            ∧ ∀ w, w < n + 1 → c'.ctr w = ct' w := by
        intro e' sm' ct' tr' btr' he' hfar'
        have hb := ih (n + 1) (p + 2) e' sm' ct' tr' btr' (by omega) he' htame hfar' hfd
          (by rw [hcb]; exact hPb)
        rw [hcb] at hb
        exact hb
      -- **The loop invariant.**  From the guard with `m` iterations left, the
      -- machine runs the body `m` more times and exits with `m` copies of the
      -- body's launches *and of its bind arrays* appended.  Induction on the
      -- *remaining* trip count, so termination is derived rather than assumed.
      --
      -- Nothing is assumed about the incoming store map, and that is the point:
      -- each iteration rewrites its own bind arrays immediately before its
      -- launch, so iteration 24 recovers the same buffers as iteration 1
      -- without the invariant having to say what the previous iteration left.
      have loop : ∀ (m i : Nat), i + m = kk →
          ∀ (e' : Env) (sm' : StoreMap) (ct' : Nat → Nat) (tr' : List LaunchRec)
            (btr' : List OpBinds),
            e' ptr = SymVal.unknown → FarOk e' ptr (n + 1) b.farArgs → ct' n = i →
            ∃ j c', hsteps fns ptr.id P j ⟨p + 1, e', sm', ct', tr', btr'⟩ = some c'
              ∧ c'.pc = p + (cb.length + 4)
              ∧ c'.trace = tr' ++ (List.replicate m b.launches).flatten
              ∧ c'.btrace = btr' ++ (List.replicate m b.binds).flatten
              ∧ (∀ w : Val, w.id < n → w.id ∉ b.primDests → c'.env w = e' w)
              ∧ c'.env ptr = SymVal.unknown
              ∧ ∀ w, w < n → c'.ctr w = ct' w := by
        intro m
        induction m with
        | zero =>
            intro i hik e' sm' ct' tr' btr' he' hfar' hct'
            refine ⟨1, ⟨p + cb.length + 4, e', sm', ct', tr', btr'⟩, ?_,
                    show p + cb.length + 4 = p + (cb.length + 4) by omega,
                    by simp, by simp, fun _ _ _ => rfl, he', fun _ _ => rfl⟩
            rw [hsteps_one, hstep_jmpLt fns ptr.id P (p + 1) e' sm' ct' tr' btr' ⟨n⟩ kk _ _ hP1]
            simp only [hct', if_neg (by omega : ¬ (i < kk))]
        | succ m ihm =>
            intro i hik e' sm' ct' tr' btr' he' hfar' hct'
            -- guard: still iterations left, so enter the body
            have hgo : hstep fns ptr.id P ⟨p + 1, e', sm', ct', tr', btr'⟩
                = some ⟨p + 2, e', sm', ct', tr', btr'⟩ := by
              rw [hstep_jmpLt fns ptr.id P (p + 1) e' sm' ct' tr' btr' ⟨n⟩ kk _ _ hP1]
              simp only [hct', if_pos (by omega : i < kk)]
            obtain ⟨jb, c₁, hrb, hpcb, htrb, hbtb, hfrb, henb, hctb⟩ :=
              ihb e' sm' ct' tr' btr' he' hfar'
            obtain ⟨qb, enb, smb, ctrb, trcb, btcb⟩ := c₁
            subst hpcb
            have htrb' : trcb = tr' ++ b.launches := htrb
            have hbtb' : btcb = btr' ++ b.binds := hbtb
            have hfrb' : ∀ w : Val, w.id < n + 1 → w.id ∉ b.primDests → enb w = e' w := hfrb
            have henb' : enb ptr = SymVal.unknown := henb
            have hctb' : ∀ w, w < n + 1 → ctrb w = ct' w := hctb
            -- increment and back edge
            have hinc : hstep fns ptr.id P ⟨p + 2 + cb.length, enb, smb, ctrb, trcb, btcb⟩
                = some ⟨p + 2 + cb.length + 1, enb, smb,
                        fun w => if w = n then ctrb n + 1 else ctrb w, trcb, btcb⟩ :=
              hstep_incC fns ptr.id P _ enb smb ctrb trcb btcb ⟨n⟩ hPi
            have hback : hstep fns ptr.id P ⟨p + 2 + cb.length + 1, enb, smb,
                            (fun w => if w = n then ctrb n + 1 else ctrb w), trcb, btcb⟩
                = some ⟨p + 1, enb, smb,
                        (fun w => if w = n then ctrb n + 1 else ctrb w), trcb, btcb⟩ :=
              hstep_jmp fns ptr.id P _ enb smb _ trcb btcb (p + 1) hPj
            have hctn : ctrb n = i := by rw [hctb' n (by omega), hct']
            obtain ⟨j₂, c₂, hr₂, hpc₂, htr₂, hbt₂, hfr₂, hen₂, hct₂⟩ :=
              ihm (i + 1) (by omega) enb smb (fun w => if w = n then ctrb n + 1 else ctrb w)
                trcb btcb henb'
                (hfar'.monoD (Nat.le_refl _) hfd hfrb')
                (by simp [hctn])
            refine ⟨1 + (jb + (1 + (1 + j₂))), c₂, ?_, hpc₂, ?_, ?_, ?_, hen₂, ?_⟩
            · exact hsteps_trans fns ptr.id P 1 _ _ _ _ (by rw [hsteps_one]; exact hgo)
                (hsteps_trans fns ptr.id P jb _ _ _ _ hrb
                  (hsteps_trans fns ptr.id P 1 _ _ _ _ (by rw [hsteps_one]; exact hinc)
                    (hsteps_trans fns ptr.id P 1 _ _ _ _ (by rw [hsteps_one]; exact hback) hr₂)))
            · rw [htr₂, htrb', List.append_assoc, List.replicate_succ, List.flatten_cons]
            · rw [hbt₂, hbtb', List.append_assoc, List.replicate_succ, List.flatten_cons]
            · intro w hw hnd
              rw [hfr₂ w hw hnd, hfrb' w (by omega) hnd]
            · intro w hw
              rw [hct₂ w hw]
              simp only [if_neg (by omega : ¬ (w = n))]
              exact hctb' w (by omega)
      -- the loop starts with the counter cleared
      have hstart : hstep fns ptr.id P ⟨p, e, sm, ct, tr, btr⟩
          = some ⟨p + 1, e, sm, fun w => if w = n then 0 else ct w, tr, btr⟩ :=
        hstep_setC fns ptr.id P p e sm ct tr btr ⟨n⟩ 0 hP0
      obtain ⟨j, c', hr, hpc, htr, hbt, hfr, hen, hct⟩ :=
        loop kk 0 (by omega) e sm (fun w => if w = n then 0 else ct w) tr btr he
          (hfar.mono (Nat.le_succ n) (fun _ _ => rfl)) (by simp)
      refine ⟨1 + j, c', ?_, ?_, ?_, ?_, ?_, hen, ?_⟩
      · exact hsteps_trans fns ptr.id P 1 j _ _ _ (by rw [hsteps_one]; exact hstart) hr
      · rw [hpc, hlen]
      · rw [htr]; rfl
      · rw [hbt]; rfl
      · intro w hw hnd; exact hfr w hw hnd
      · intro w hw
        rw [hct w hw]
        simp only [if_neg (by omega : ¬ (w = n))]

-- ---------------------------------------------------------------------------
-- Branch-free statements compile to a single block
-- ---------------------------------------------------------------------------

/-!
  A statement with no loop emits straight-line code, and for straight-line code
  the *static scan* and the *execution* coincide — there is no control flow to
  make them differ.  So this fragment needs no machine at all: `scanBlock` is
  already the trace, and `Clif.launchesOf` on the emitted block is already the
  answer.

  That is not a corner case here.  Four of Qwen2's five per-token functions are
  single-block and branch-free, and they carry **531 of the 533** device writes
  a token performs; only the twenty-four-layer loop in the entry function needs
  the pc-machine.  Doing this case first is what lets those facts become kernel
  `decide` on a small term rather than `native_decide` on a monadic builder run.
-/

/-- No loop, hence no branches in the emitted code. -/
def HStmt.BranchFreeB : HStmt → Bool
  | .skip      => true
  | .prim _    => true
  | .launch _  => true
  | .extern _  => true
  | .call b    => b.BranchFreeB
  | .seq a b   => a.BranchFreeB && b.BranchFreeB
  | .forN _ _  => false

/-- **Compile to one block's instructions.**  The same emission `flatHI` uses,
    without the branch machinery — so nothing here is a second code path. -/
def instsOf (fnLaunch : FnRef) (ptr : Val) : Nat → HStmt → Nat × List Inst
  | n, .skip      => (n, [])
  | n, .prim is   => (n, is)
  | n, .launch s  => emitLaunch fnLaunch ptr n s
  | n, .extern es =>
      let r := emitArgs ptr n es.argv
      (r.1, r.2.1 ++ [Inst.call none es.fn r.2.2])
  | n, .call b    => instsOf fnLaunch ptr n b
  | n, .seq a b   =>
      let r₁ := instsOf fnLaunch ptr n a
      let r₂ := instsOf fnLaunch ptr r₁.1 b
      (r₂.1, r₁.2 ++ r₂.2)
  | n, .forN _ _  => (n, [])

/-- The straight-line emitter allocates upward, like `flatHI`. -/
theorem instsOf_le (fnLaunch : FnRef) (ptr : Val) :
    ∀ (s : HStmt) (n : Nat), n ≤ (instsOf fnLaunch ptr n s).1 := by
  intro s
  induction s with
  | skip => intro n; exact Nat.le_refl n
  | prim _ => intro n; exact Nat.le_refl n
  | launch st => intro n; exact emitLaunch_le fnLaunch ptr n st
  | extern es => intro n; exact (emitArgs_props ptr es.argv n).1
  | call b ih => intro n; exact ih n
  | seq a b iha ihb => intro n; exact Nat.le_trans (iha n) (ihb _)
  | forN _ _ => intro n; exact Nat.le_refl n

/-- **A value the caller already held survives the emitted straight line**,
    unless one of its `prim` fragments writes it.  The counterpart of
    `flatHI_sound`'s frame conjunct, and what lets a `far` base keep its meaning
    across a sequence. -/
theorem instsOf_frameAt (fnLaunch : FnRef) (ptr : Val) :
    ∀ (s : HStmt) (n : Nat) (e : Env) (w : Val),
      w.id < n → w.id ∉ s.primDests →
      evalPure e (instsOf fnLaunch ptr n s).2 w = e w := by
  intro s
  induction s with
  | skip => intro _ _ _ _ _; rfl
  | prim is =>
      intro n e w _ hnd
      exact evalPure_frame is e w (fun i hi d hd => by
        intro hEq
        exact hnd (List.mem_filterMap.mpr ⟨i, hi, by rw [hd]; simpa using hEq.symm⟩))
  | launch st => intro n e w hw _; exact emitLaunch_frame fnLaunch ptr n st e w hw
  | extern es =>
      intro n e w hw _
      show evalPure e ((emitArgs ptr n es.argv).2.1
              ++ [Inst.call none es.fn (emitArgs ptr n es.argv).2.2]) w = e w
      rw [evalPure_append]
      show evalPure e (emitArgs ptr n es.argv).2.1 w = e w
      exact emitArgs_frameAt ptr es.argv n e w hw
  | call b ih => intro n e w hw hnd; exact ih n e w hw hnd
  | seq a b iha ihb =>
      intro n e w hw hnd
      show evalPure e ((instsOf fnLaunch ptr n a).2
              ++ (instsOf fnLaunch ptr (instsOf fnLaunch ptr n a).1 b).2) w = e w
      rw [evalPure_append,
          ihb (instsOf fnLaunch ptr n a).1 _ w
            (Nat.lt_of_lt_of_le hw (instsOf_le fnLaunch ptr a n))
            (fun hc => hnd (List.mem_append.mpr (Or.inr hc))),
          iha n e w hw (fun hc => hnd (List.mem_append.mpr (Or.inl hc)))]
  | forN _ _ => intro _ _ _ _ _; rfl

/-- **The emitted straight line performs exactly the declared sequence.**

    Stated against `Clif.scanBlock` rather than a program counter, because with
    no branches the two are the same walk.  The base pointer survives, so the
    statement composes: a sequence's second half starts from an environment in
    which `ptr` is still a runtime input. -/
theorem instsOf_sound (fns : List FnDecl) (fnLaunch : FnRef) (ptr : Val)
    (hfn : fnNameOf fns fnLaunch = some "cl_cuda_launch") :
    ∀ (s : HStmt) (n : Nat) (e : Env),
      s.BranchFreeB = true → HStmt.TameB fns ptr s = true →
      FarOk e ptr n s.farArgs →
      (∀ x ∈ s.farArgs, ∀ b ∈ x.deps, b ∉ s.primDests) →
      ptr.id < n → e ptr = SymVal.unknown →
      n ≤ (instsOf fnLaunch ptr n s).1
      ∧ (scanBlock fns e (instsOf fnLaunch ptr n s).2).2 = s.launches
      ∧ evalPure e (instsOf fnLaunch ptr n s).2 ptr = SymVal.unknown := by
  intro s
  induction s with
  | skip => intro n e _ _ _ _ _ he; exact ⟨Nat.le_refl n, rfl, he⟩
  | prim is =>
      intro n e _ htame _ _ _ he
      have hall : ∀ i ∈ is, Inst.TameB fns ptr i = true := by
        simpa [HStmt.TameB, List.all_eq_true] using htame
      refine ⟨Nat.le_refl n, ?_, ?_⟩
      · show (scanBlock fns e is).2 = []
        exact scanBlock_noLaunch fns is e
          (launchCallCount_zero fns is (fun i hi => tame_noLaunch fns ptr i (hall i hi)))
      · show evalPure e is ptr = SymVal.unknown
        rw [evalPure_frame is e ptr
              (fun i hi d hd => tame_noWrite fns ptr i (hall i hi) d hd)]
        exact he
  | launch st =>
      intro n e _ _ _ _ hptr he
      refine ⟨emitLaunch_le fnLaunch ptr n st, emitLaunch_scan fns fnLaunch ptr hfn n st e hptr he,
              ?_⟩
      show evalPure e (emitLaunch fnLaunch ptr n st).2 ptr = SymVal.unknown
      rw [emitLaunch_frame fnLaunch ptr n st e ptr hptr]; exact he
  | extern es =>
      intro n e _ htame hfar _ hptr he
      have h3 : ((fnNameOf fns es.fn == some es.name) = true
                ∧ decide (es.name ∈ deviceWriterNames) = true)
                ∧ (!decide (es.name ∈ launchNames)) = true := by
        simp only [HStmt.TameB, Bool.and_eq_true] at htame
        exact ⟨⟨htame.1.1, htame.1.2⟩, htame.2⟩
      have hnm : fnNameOf fns es.fn = some es.name := by simpa using h3.1.1
      have hw  : es.name ∈ deviceWriterNames := by simpa using h3.1.2
      have hl  : es.name ∉ launchNames := by simpa using h3.2
      have hff : FarOk e ptr n es.argv := hfar
      refine ⟨(emitArgs_props ptr es.argv n).1, ?_, ?_⟩
      · show (scanBlock fns e ((emitArgs ptr n es.argv).2.1
                ++ [Inst.call none es.fn (emitArgs ptr n es.argv).2.2])).2 = [es.toRec]
        rw [scanBlock_append,
            scanBlock_noCalls fns _ e (fun i hi => (emitArgs_props ptr es.argv n).2.2 i hi),
            List.nil_append]
        show ((launchAt fns (evalPure e (emitArgs ptr n es.argv).2.1)
                (Inst.call none es.fn (emitArgs ptr n es.argv).2.2)).toList ++ []) = _
        simp only [launchAt, hnm, if_neg hl, if_pos hw, Option.toList,
                   List.append_nil, ExternStep.toRec]
        rw [emitArgs_desc ptr es.argv n e hff hptr he]
      · show evalPure e ((emitArgs ptr n es.argv).2.1
                ++ [Inst.call none es.fn (emitArgs ptr n es.argv).2.2]) ptr = SymVal.unknown
        rw [evalPure_append]
        show evalPure e (emitArgs ptr n es.argv).2.1 ptr = SymVal.unknown
        rw [emitArgs_frame ptr es.argv n e hptr]; exact he
  | call b ih => intro n e hbf htame hfar hfd hptr he; exact ih n e hbf htame hfar hfd hptr he
  | seq a b iha ihb =>
      intro n e hbf htame hfar hfd hptr he
      obtain ⟨hbfa, hbfb⟩ := Bool.and_eq_true .. ▸ hbf
      obtain ⟨hta, htb⟩ := Bool.and_eq_true .. ▸ htame
      obtain ⟨hma, hsa, hea⟩ := iha n e hbfa hta
        (fun x hx => hfar x (List.mem_append.mpr (Or.inl hx)))
        (fun x hx b hb hc => hfd x (List.mem_append.mpr (Or.inl hx)) b hb
          (List.mem_append.mpr (Or.inl hc))) hptr he
      obtain ⟨hmb, hsb, heb⟩ :=
        ihb (instsOf fnLaunch ptr n a).1 (evalPure e (instsOf fnLaunch ptr n a).2)
          hbfb htb
          (FarOk.monoD (as := b.farArgs) (ds := a.primDests)
            (fun x hx => hfar x (List.mem_append.mpr (Or.inr hx))) hma
            (fun x hx bb hbb hc => hfd x (List.mem_append.mpr (Or.inr hx)) bb hbb
              (List.mem_append.mpr (Or.inl hc)))
            (fun w hw hnc => instsOf_frameAt fnLaunch ptr a n e w hw hnc))
          (fun x hx bb hb hc => hfd x (List.mem_append.mpr (Or.inr hx)) bb hb
            (List.mem_append.mpr (Or.inr hc)))
          (Nat.lt_of_lt_of_le hptr hma) hea
      refine ⟨Nat.le_trans hma hmb, ?_, ?_⟩
      · show (scanBlock fns e ((instsOf fnLaunch ptr n a).2
                ++ (instsOf fnLaunch ptr (instsOf fnLaunch ptr n a).1 b).2)).2
            = a.launches ++ b.launches
        rw [scanBlock_append, hsa, hsb]
      · show evalPure e ((instsOf fnLaunch ptr n a).2
                ++ (instsOf fnLaunch ptr (instsOf fnLaunch ptr n a).1 b).2) ptr
            = SymVal.unknown
        rw [evalPure_append]; exact heb
  | forN k b _ => intro n e hbf _ _ _; exact absurd hbf (by simp [HStmt.BranchFreeB])

/-- The single CLIF block a branch-free statement compiles to. -/
def blockOf (fnLaunch : FnRef) (ptr : Val) (n : Nat) (s : HStmt) : BlockData :=
  { ref := ⟨0⟩, params := [(ptr, .i64)], insts := (instsOf fnLaunch ptr n s).2 }

/-- …and the built function containing it. -/
def stateOf (fns : List FnDecl) (fnLaunch : FnRef) (ptr : Val) (n : Nat)
    (s : HStmt) : IRState :=
  { fns := fns, blocks := [blockOf fnLaunch ptr n s] }

/-- **What `Clif.launchesOf` reads out of the emitted function is the declared
    sequence.**

    The statement the Qwen2 seam facts want, and the reason they can stop being
    `native_decide`.  `launchesOf` applied to a *builder run* forces the kernel
    to reduce `StateT` and closures, which it cannot do; applied to `stateOf`
    it walks a first-order term, and `HStmt.launches` is a structural recursion
    on a small tree.  Same fact, kernel-checkable. -/
theorem launchesOf_stateOf (fns : List FnDecl) (fnLaunch : FnRef) (ptr : Val)
    (hfn : fnNameOf fns fnLaunch = some "cl_cuda_launch")
    (s : HStmt) (n : Nat)
    (hbf : s.BranchFreeB = true) (htame : HStmt.TameB fns ptr s = true)
    (hfar : FarOk Env.empty ptr n s.farArgs)
    (hfd : ∀ x ∈ s.farArgs, ∀ b ∈ x.deps, b ∉ s.primDests)
    (hptr : ptr.id < n) :
    launchesOf (stateOf fns fnLaunch ptr n s) = s.launches := by
  have h := instsOf_sound fns fnLaunch ptr hfn s n Env.empty hbf htame
    hfar hfd hptr rfl
  show ([] ++ (scanBlock fns Env.empty (instsOf fnLaunch ptr n s).2).2) = _
  rw [List.nil_append, h.2.1]

-- ---------------------------------------------------------------------------
-- …and what they bound
-- ---------------------------------------------------------------------------

/-- **What the emitted straight line bound is what the statement declared.**

    The bind-array counterpart of `instsOf_sound`, and the reason
    `HStmt.deviceOps` can be a derived list rather than a parameter.  Note what
    is *not* assumed about the incoming store map: nothing.  A bind fragment
    writes its whole array immediately before its call, so its entries shadow
    whatever a `prim` left behind — which is why this composes across a
    sequence with no invariant threaded through. -/
theorem instsOf_binds (fns : List FnDecl) (fnLaunch : FnRef) (ptr : Val)
    (hfn : fnNameOf fns fnLaunch = some "cl_cuda_launch") :
    ∀ (s : HStmt) (n : Nat) (e : Env) (m : StoreMap),
      s.BranchFreeB = true → HStmt.TameB fns ptr s = true →
      FarOk e ptr n s.farArgs →
      (∀ x ∈ s.farArgs, ∀ b ∈ x.deps, b ∉ s.primDests) →
      ptr.id < n → e ptr = SymVal.unknown →
      (bindScan fns ptr.id ⟨e, m⟩ (instsOf fnLaunch ptr n s).2).2 = s.binds := by
  intro s
  induction s with
  | skip => intro _ _ _ _ _ _ _ _ _; rfl
  | prim is =>
      intro n e m _ htame _ _ _ _
      have hall : ∀ i ∈ is, Inst.TameB fns ptr i = true := by
        simpa [HStmt.TameB, List.all_eq_true] using htame
      exact bindScan_noLaunch fns ptr.id is ⟨e, m⟩
        (launchCallCount_zero fns is (fun i hi => tame_noLaunch fns ptr i (hall i hi)))
  | launch st =>
      intro n e m _ htame hfar _ hptr he
      exact emitLaunch_bindScan fns fnLaunch ptr hfn n st e m hptr he
        (by simpa [HStmt.TameB] using htame) hfar
  | extern es =>
      intro n e m _ htame hfar _ hptr he
      have h3 : ((fnNameOf fns es.fn == some es.name) = true
                ∧ decide (es.name ∈ deviceWriterNames) = true)
                ∧ (!decide (es.name ∈ launchNames)) = true := by
        simp only [HStmt.TameB, Bool.and_eq_true] at htame
        exact ⟨⟨htame.1.1, htame.1.2⟩, htame.2⟩
      have hnm : fnNameOf fns es.fn = some es.name := by simpa using h3.1.1
      have hw  : es.name ∈ deviceWriterNames := by simpa using h3.1.2
      have hl  : es.name ∉ launchNames := by simpa using h3.2
      have hff : FarOk e ptr n es.argv := hfar
      have hlp : es.name ∉ positionalLaunchNames := by
        intro hc
        apply hl
        simp only [positionalLaunchNames, List.mem_cons, List.not_mem_nil, or_false] at hc
        rcases hc with h | h <;> rw [h] <;> decide
      have hmap : (emitArgs ptr n es.argv).2.2.map
                    (fun v => bufDescOf ptr.id
                      ((bevalPure ⟨e, m⟩ (emitArgs ptr n es.argv).2.1).env v))
                  = es.argv.map ExternArg.toBuf := by
        rw [show (fun v => bufDescOf ptr.id
                    ((bevalPure ⟨e, m⟩ (emitArgs ptr n es.argv).2.1).env v))
                = (fun v => bufDescOf ptr.id (evalPure e (emitArgs ptr n es.argv).2.1 v)) from by
              funext v; rw [bevalPure_env]]
        exact emitArgs_buf ptr es.argv n e hff hptr he
      have hcall : isLaunchCallB fns (Inst.call none es.fn (emitArgs ptr n es.argv).2.2)
                    = true := by simp [isLaunchCallB, hnm, hw]
      have hbind : bindAt fns ptr.id (bevalPure ⟨e, m⟩ (emitArgs ptr n es.argv).2.1)
                      (Inst.call none es.fn (emitArgs ptr n es.argv).2.2)
                    = es.toBinds := by
        simp only [bindAt, hnm, if_neg hlp, if_pos hw, hmap, ExternStep.toBinds]
      show (bindScan fns ptr.id ⟨e, m⟩ ((emitArgs ptr n es.argv).2.1
              ++ [Inst.call none es.fn (emitArgs ptr n es.argv).2.2])).2 = [es.toBinds]
      rw [bindScan_append,
          bindScan_noCalls fns ptr.id _ _
            (fun i hi => (emitArgs_props ptr es.argv n).2.2 i hi),
          List.nil_append, bindScan_state]
      show ((if isLaunchCallB fns (Inst.call none es.fn (emitArgs ptr n es.argv).2.2)
              then [bindAt fns ptr.id
                      (bevalPure ⟨e, m⟩ (emitArgs ptr n es.argv).2.1)
                      (Inst.call none es.fn (emitArgs ptr n es.argv).2.2)]
              else []) ++ []) = _
      rw [hcall, if_pos rfl, List.append_nil, hbind]
  | call b ih => intro n e m hbf htame hfar hfd hptr he
                 exact ih n e m hbf htame hfar hfd hptr he
  | seq a b iha ihb =>
      intro n e m hbf htame hfar hfd hptr he
      obtain ⟨hbfa, hbfb⟩ := Bool.and_eq_true .. ▸ hbf
      obtain ⟨hta, htb⟩ := Bool.and_eq_true .. ▸ htame
      have hfa : FarOk e ptr n a.farArgs :=
        fun x hx => hfar x (List.mem_append.mpr (Or.inl hx))
      have hda : ∀ x ∈ a.farArgs, ∀ b ∈ x.deps, b ∉ a.primDests :=
        fun x hx b hb hc => hfd x (List.mem_append.mpr (Or.inl hx)) b hb
          (List.mem_append.mpr (Or.inl hc))
      have hfb : FarOk (bevalPure ⟨e, m⟩ (instsOf fnLaunch ptr n a).2).env ptr
          (instsOf fnLaunch ptr n a).1 b.farArgs := by
        refine FarOk.monoD (as := b.farArgs) (ds := a.primDests)
          (fun x hx => hfar x (List.mem_append.mpr (Or.inr hx)))
          (instsOf_le fnLaunch ptr a n)
          (fun x hx bb hbb hc => hfd x (List.mem_append.mpr (Or.inr hx)) bb hbb
            (List.mem_append.mpr (Or.inl hc)))
          (fun w hw hnc => ?_)
        rw [bevalPure_env]
        exact instsOf_frameAt fnLaunch ptr a n e w hw hnc
      have hdb : ∀ x ∈ b.farArgs, ∀ bb ∈ x.deps, bb ∉ b.primDests :=
        fun x hx bb hb hc => hfd x (List.mem_append.mpr (Or.inr hx)) bb hb
          (List.mem_append.mpr (Or.inr hc))
      obtain ⟨hma, _, hea⟩ :=
        instsOf_sound fns fnLaunch ptr hfn a n e hbfa hta hfa hda hptr he
      show (bindScan fns ptr.id ⟨e, m⟩ ((instsOf fnLaunch ptr n a).2
              ++ (instsOf fnLaunch ptr (instsOf fnLaunch ptr n a).1 b).2)).2
          = a.binds ++ b.binds
      rw [bindScan_append, iha n e m hbfa hta hfa hda hptr he, bindScan_state,
          ihb (instsOf fnLaunch ptr n a).1
              (bevalPure ⟨e, m⟩ (instsOf fnLaunch ptr n a).2).env
              (bevalPure ⟨e, m⟩ (instsOf fnLaunch ptr n a).2).mem
              hbfb htb hfb hdb (Nat.lt_of_lt_of_le hptr hma)
              (by rw [bevalPure_env]; exact hea)]
  | forN k b _ => intro _ _ _ hbf _ _ _ _ _; exact absurd hbf (by simp [HStmt.BranchFreeB])

/-- **What `Clif.bindsOf` reads out of the emitted function is what the
    statement declared it would bind.** -/
theorem bindsOf_stateOf (fns : List FnDecl) (fnLaunch : FnRef) (ptr : Val)
    (hfn : fnNameOf fns fnLaunch = some "cl_cuda_launch")
    (s : HStmt) (n : Nat)
    (hbf : s.BranchFreeB = true) (htame : HStmt.TameB fns ptr s = true)
    (hfar : FarOk Env.empty ptr n s.farArgs)
    (hfd : ∀ x ∈ s.farArgs, ∀ b ∈ x.deps, b ∉ s.primDests)
    (hptr : ptr.id < n) :
    bindsOf ptr.id (stateOf fns fnLaunch ptr n s) = s.binds := by
  show ([] ++ (bindScan fns ptr.id BEnv.empty (instsOf fnLaunch ptr n s).2).2) = _
  rw [List.nil_append]
  exact instsOf_binds fns fnLaunch ptr hfn s n Env.empty [] hbf htame
    hfar hfd hptr rfl

/-- **The seam, closed.**  Records *and* the arrays they were handed, both read
    out of the emitted CLIF, both equal to what the host statement declared.
    Downstream this is what lets a plan be matched against the program without
    the bind list arriving from somewhere else. -/
theorem deviceOpsOf_stateOf (fns : List FnDecl) (fnLaunch : FnRef) (ptr : Val)
    (hfn : fnNameOf fns fnLaunch = some "cl_cuda_launch")
    (s : HStmt) (n : Nat)
    (hbf : s.BranchFreeB = true) (htame : HStmt.TameB fns ptr s = true)
    (hfar : FarOk Env.empty ptr n s.farArgs)
    (hfd : ∀ x ∈ s.farArgs, ∀ b ∈ x.deps, b ∉ s.primDests)
    (hptr : ptr.id < n) :
    deviceOpsOf ptr.id (stateOf fns fnLaunch ptr n s) = s.deviceOps := by
  show (launchesOf (stateOf fns fnLaunch ptr n s)).zip
        (bindsOf ptr.id (stateOf fns fnLaunch ptr n s)) = _
  rw [launchesOf_stateOf fns fnLaunch ptr hfn s n hbf htame hfar hfd hptr,
      bindsOf_stateOf fns fnLaunch ptr hfn s n hbf htame hfar hfd hptr]
  rfl

-- ---------------------------------------------------------------------------
-- Instantiated at the inference driver's shape
-- ---------------------------------------------------------------------------

/-!
  A `∃ k, …` theorem with three hypotheses can be true because nothing satisfies
  them.  What follows discharges all three at concrete values, on a program with
  the shape the Qwen2 driver has — an embed launch, twenty-four layers behind a
  loop *and* a call, then the sampling tail.
-/

def demoFns : List FnDecl := [{ ref := ⟨0⟩, name := "cl_cuda_launch", sig := ⟨0⟩ }]

def demoLaunch : FnRef := ⟨0⟩

/-- The memory base — a block parameter, so unbound in the environment. -/
def demoPtr : Val := ⟨0⟩

/-- One transformer layer: eleven kernels. -/
def demoLayer : HStmt :=
  (List.range 11).foldl
    (fun acc i => HStmt.seq acc (HStmt.launch ⟨100 + i * 8, 4, 300 + i * 8, 6, 256,
                                    [.slot 40, .slot 48, .slot 56, .slot 64]⟩))
    HStmt.skip

/-- The driver.  The layers sit behind `forN` *and* `call` — the two constructs
    that made `Clif.launchesOf` report 1 launch instead of 267. -/
def demoDriver : HStmt :=
  HStmt.seq
    (HStmt.seq (HStmt.launch ⟨0, 3, 8, 1, 256, [.slot 8, .slot 16, .slot 24]⟩) (HStmt.forN 24 (HStmt.call demoLayer)))
    (HStmt.seq (HStmt.launch ⟨90, 2, 290, 1, 256, [.slot 8, .slot 16]⟩) (HStmt.launch ⟨98, 2, 298, 1, 32, [.slot 8, .slot 32]⟩))

def demoCode : List HI := code demoLaunch demoPtr 1 0 demoDriver

theorem demoLayer_count : demoLayer.launchCount = 11 := by decide

/-- **267 launches per token** — derived from the structure, not by evaluating a
    267-element list (`decide` on this hits `maxRecDepth`). -/
theorem demoDriver_count : demoDriver.launchCount = 267 := by
  show (HStmt.seq (HStmt.seq _ (HStmt.forN 24 (HStmt.call demoLayer)))
          (HStmt.seq _ _)).launchCount = 267
  rw [launchCount_seq, launchCount_seq, launchCount_seq, launches_forN,
      launchCount_call, demoLayer_count, launchCount_launch, launchCount_launch,
      launchCount_launch]

/-- **…and that many bindings too, through the loop.**

    From `deviceOps_length` and the count — not by materialising 267 pairs.
    That is what the composition lemmas buy: a looped driver's device-write
    sequence, records and bindings together, is reachable by induction. -/
theorem demoDriver_deviceOps_count : demoDriver.deviceOps.length = 267 := by
  rw [HStmt.deviceOps_length]; exact demoDriver_count

/-- **The driver's compiled code really does run its declared launch sequence.**

    Every hypothesis is discharged at a concrete value, so this is not vacuous:
    `demoCode` is a genuine list of host instructions, `demoFns` a genuine
    declaration table, and the initial environment binds nothing.

    The loop body is emitted **once**.  Unrolling twenty-four layers would take
    3204 instructions; this is 172, and executes 267 launches. -/
theorem demo_trace :
    ∃ k c', hsteps demoFns demoPtr.id demoCode k
              ⟨0, Env.empty, [], fun _ => 0, [], []⟩ = some c'
      ∧ c'.trace = demoDriver.launches
      ∧ c'.trace.length = 267
      ∧ c'.btrace = demoDriver.binds
      ∧ (c'.trace.zip c'.btrace) = demoDriver.deviceOps := by
  obtain ⟨k, c', hr, _, htr, hbt, _, _⟩ :=
    flatHI_sound demoFns demoLaunch demoPtr rfl demoCode demoDriver 1 0
      Env.empty [] (fun _ => 0) [] [] (by decide) rfl (by decide)
      (FarOk.of_noBases (as := HStmt.farArgs demoDriver) (by decide))
      (fun x hx b hb =>
        noBases_primDests (ds := HStmt.primDests demoDriver) (by decide) x hx b hb)
      (fun j _ => by rw [Nat.zero_add]; rfl)
  refine ⟨k, c', hr, ?_, ?_, ?_, ?_⟩
  · rw [htr, List.nil_append]
  · rw [htr, List.nil_append]; exact demoDriver_count
  · rw [hbt, List.nil_append]
  · rw [htr, hbt, List.nil_append, List.nil_append]; rfl

-- ---------------------------------------------------------------------------
-- The kernel-clean route, on a Qwen2-shaped function
-- ---------------------------------------------------------------------------

/-!
  `native_decide` is not a weakness of the *check*; it is a consequence of the
  driver being defined by **running** a builder rather than by **being** a term.
  A statement about `(inferFn.run {}).2` forces the kernel through `StateT`,
  `bind` and closures; a statement about an `HStmt` walks a small tree.

  Below is the feed-forward half of a Qwen2 layer — three launches, three vendor
  matvecs, filler in between — with its device-write sequence proven in the
  kernel.  Same class of fact as `Qwen2Algorithm.ffn_writes`, no compiler trust.
-/

def demoFfnFns : List FnDecl :=
  [ { ref := ⟨0⟩, name := "cl_cuda_launch",  sig := ⟨0⟩ }
  , { ref := ⟨1⟩, name := "cl_cublas_sgemv", sig := ⟨0⟩ } ]

/-- `y := A·x`, with the three buffer handles loaded from their slots. -/
def demoSgemv (a x y : Nat) : HStmt :=
  .extern { name := "cl_cublas_sgemv", fn := ⟨1⟩
            argv := [.const 1, .const 896, .const 4864, .slot a, .slot x, .slot y] }

def demoFfn : HStmt :=
  .seq (.prim [Inst.iconst ⟨1⟩ .i64 7])
    (.seq (.launch ⟨0x2000, 3, 0x80, 1, 32, [.slot 0x18, .slot 0x20, .slot 0x28]⟩)
      (.seq (demoSgemv 0x100 0x108 0x110)
        (.seq (demoSgemv 0x100 0x108 0x118)
          (.seq (.launch ⟨0x8000, 3, 0xA0, 152, 32, [.slot 0x28, .slot 0x30, .slot 0x38]⟩)
            (.seq (demoSgemv 0x120 0x118 0x128)
                  (.launch ⟨0x8C00, 2, 0xB0, 28, 32, [.slot 0x38, .slot 0x40]⟩))))))

/-- **What the extractor reads out of the emitted block is the declared
    sequence** — `[propext, Classical.choice, Quot.sound]`, no `trustCompiler`. -/
theorem demoFfn_launchesOf :
    launchesOf (stateOf demoFfnFns ⟨0⟩ ⟨0⟩ 1 demoFfn) = demoFfn.launches :=
  launchesOf_stateOf demoFfnFns ⟨0⟩ ⟨0⟩ rfl demoFfn 1 (by decide) (by decide)
    (FarOk.of_noBases (as := HStmt.farArgs demoFfn) (by decide))
    (fun x hx b hb => noBases_primDests (ds := HStmt.primDests demoFfn) (by decide) x hx b hb)
    (by decide)

/-- …and the sequence itself, by `decide` on a small term — **no axioms at
    all**, where the same fact about the builder needs `native_decide`. -/
theorem demoFfn_count : demoFfn.launchCount = 6 := by decide

theorem demoFfn_names :
    demoFfn.launches.map LaunchRec.fnName
      = ["cl_cuda_launch", "cl_cublas_sgemv", "cl_cublas_sgemv",
         "cl_cuda_launch", "cl_cublas_sgemv", "cl_cuda_launch"] := by decide

/-- **…and what each of them bound**, recovered from the same emitted block.

    The instantiation that keeps `bindsOf_stateOf` from being a theorem about
    nothing: the three launches' pointer arrays come back off the stores, and
    the three `sgemv`s' arguments come back with their bases. -/
theorem demoFfn_bindsOf :
    bindsOf 0 (stateOf demoFfnFns ⟨0⟩ ⟨0⟩ 1 demoFfn) = demoFfn.binds :=
  bindsOf_stateOf demoFfnFns ⟨0⟩ ⟨0⟩ rfl demoFfn 1 (by decide) (by decide)
    (FarOk.of_noBases (as := HStmt.farArgs demoFfn) (by decide))
    (fun x hx b hb => noBases_primDests (ds := HStmt.primDests demoFfn) (by decide) x hx b hb)
    (by decide)

/-- The arrays themselves — the slots the fragment stored, in order. -/
theorem demoFfn_binds_are :
    demoFfn.binds.map OpBinds.bufs
      = [ some [.near 0x18, .near 0x20, .near 0x28]
        , none, none
        , some [.near 0x28, .near 0x30, .near 0x38]
        , none
        , some [.near 0x38, .near 0x40] ] := by decide

/-- The vendor calls' buffers, base included — what `ArgDesc` alone could not
    distinguish. -/
theorem demoFfn_args_are :
    demoFfn.binds.map OpBinds.args
      = [ []
        , [.const 1, .const 896, .const 4864, .near 0x100, .near 0x108, .near 0x110]
        , [.const 1, .const 896, .const 4864, .near 0x100, .near 0x108, .near 0x118]
        , []
        , [.const 1, .const 896, .const 4864, .near 0x120, .near 0x118, .near 0x128]
        , [] ] := by decide

/-- **Records and bindings, both read out of the emitted CLIF, both equal to
    what the host statement declared.**  The seam, closed at a value. -/
theorem demoFfn_deviceOps :
    deviceOpsOf 0 (stateOf demoFfnFns ⟨0⟩ ⟨0⟩ 1 demoFfn) = demoFfn.deviceOps :=
  deviceOpsOf_stateOf demoFfnFns ⟨0⟩ ⟨0⟩ rfl demoFfn 1 (by decide) (by decide)
    (FarOk.of_noBases (as := HStmt.farArgs demoFfn) (by decide))
    (fun x hx b hb => noBases_primDests (ds := HStmt.primDests demoFfn) (by decide) x hx b hb)
    (by decide)

end AlgorithmLib.Host
