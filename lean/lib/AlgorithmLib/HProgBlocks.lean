import AlgorithmLib.HProgSem

/-!
# `HProgBlocks` — executing the compiled form

`HProgSem` says what a *term* does. This says what the `IR.FuncData` that
`compileFn` produces does: blocks, block parameters, `jump`, `brif`, and a
value numbering rather than a slot numbering.

Two interpreters, one instruction semantics. Every arithmetic arm here goes
through `HProgSem.evalOp` on a two-element environment, so the operation
semantics is *literally the same definition* on both sides. That is what stops
`compile_sound` from being provable by accident: what it has to show is that
compilation preserves the *order and arguments* of the operations, which is the
only thing the two forms can disagree about — and the only thing that has
actually gone wrong so far (exit and join parameters twice).

Scope: this interprets compiler output, not arbitrary CLIF. Instructions
`compileFn` never emits are stuck rather than silently ignored.
-/

namespace AlgorithmLib.HProg.Blocks

open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sem

/-- Values numbered as `compileFn` numbers them: densely, from zero. -/
abbrev Vals := Array V

def getV (vs : Vals) (v : Val) : Option V := vs[v.id]?

def setV (vs : Vals) (v : Val) (x : V) : Vals :=
  let vs := if v.id < vs.size then vs
            else vs ++ Array.replicate (v.id + 1 - vs.size) default
  vs.set! v.id x

/-- Evaluate through the term interpreter's own `evalOp`, on an environment
    holding just this instruction's operands. The `Op` names slots `0, 1`
    because that is where they were put. -/
private def viaOp (m : Mem) (args : List V) (o : Op) : Option V :=
  evalOp m args.toArray o

/-- One result-producing instruction: its destination and what it computes. -/
def evalInst (m : Mem) (vs : Vals) : Inst → Option (Val × V)
  | .iconst d t k => do pure (d, ← viaOp m [] (.iconst t k))
  | .fconst d t b => do pure (d, ← viaOp m [] (.fconst t b))
  | .iadd d a b => bin d a b .iadd | .isub d a b => bin d a b .isub
  | .imul d a b => bin d a b .imul | .udiv d a b => bin d a b .udiv
  | .ishl d a b => bin d a b .ishl | .ushr d a b => bin d a b .ushr
  | .band d a b => bin d a b .band | .bandNot d a b => bin d a b .bandNot
  | .bor d a b => bin d a b .bor   | .bxor d a b => bin d a b .bxor
  | .fadd d a b => bin d a b .fadd | .fsub d a b => bin d a b .fsub
  | .fmul d a b => bin d a b .fmul | .fmax d a b => bin d a b .fmax
  | .fmin d a b => bin d a b .fmin
  | .ineg d a => un d a .ineg      | .ctz d a => un d a .ctz
  | .popcnt d a => un d a .popcnt  | .fneg d a => un d a .fneg
  | .fpromote d a => un d a .fpromote
  | .ireduce32 d a => un d a .ireduce32
  | .uextend64 d a => un d a .uextend64
  | .sextend64 d a => un d a .sextend64
  | .vhighBits d a => un d a .vhighBits
  | .icmp d c a b => bin d a b (fun x y => .icmp c x y)
  | .fcmp d c a b => bin d a b (fun x y => .fcmp c x y)
  | .splat d t a => un d a (.splat t)
  | .bitcast d t a => un d a (.bitcast t)
  | .fcvtFromSint d t a => un d a (.fcvtFromSint t)
  | .fcvtToUint d t a => un d a (.fcvtToUint t)
  | .extractlane d a l => un d a (.extractlane · l)
  | .load d op a => un d a (.load op)
  | .select d c a b => do
      let cv ← getV vs c; let x ← getV vs a; let y ← getV vs b
      pure (d, ← viaOp m [cv, x, y] (.select 0 1 2))
  | _ => none
where
  bin (d : Val) (a b : Val) (f : R → R → Op) : Option (Val × V) := do
    let x ← getV vs a; let y ← getV vs b
    pure (d, ← viaOp m [x, y] (f 0 1))
  un (d : Val) (a : Val) (f : R → Op) : Option (Val × V) := do
    let x ← getV vs a
    pure (d, ← viaOp m [x] (f 0))

/-- Where control goes after a terminator, and with which block arguments. -/
inductive Next where
  | goto (target : Nat) (args : List V)
  | done
  deriving Inhabited

structure BSt where
  vals : Vals
  world : World

/-- Perform one store. Split out so `runInsts` recurses only on its list —
    a helper that called back into it would make the whole thing `partial`,
    and a `partial def` has no equations to prove anything from. -/
def doStore (s : BSt) (v a : Val) (as : Option ClifTy) : Except String BSt :=
  match getV s.vals v, getV s.vals a with
  | some val, some (.sc _ addr) =>
      match val with
      | .sc t b =>
          let n := tyBytes (as.getD t)
          match s.world.mem.store addr n b with
          | some m => .ok { s with world := { obsStore s.world addr n b with mem := m } }
          | none => .error "store to unmapped address"
      | .vec t ls =>
          let lw := ((t.lanes.map (·.1.width)).getD 8) / 8
          match ls.zipIdx.foldlM
              (fun mm (x, i) => mm.store (addr + UInt64.ofNat (i * lw)) lw x) s.world.mem with
          | some m => .ok { s with
              world := { obsStore s.world addr (tyBytes (as.getD t)) 0 with mem := m } }
          | none => .error "vector store to unmapped address"
  | _, _ => .error "store operand is not defined"

/-- Run the straight-line part of a block, stopping at its terminator.
    Structural on the instruction list, so it has equations. -/
def runInsts (env : FnEnv) (s : BSt) : List Inst → Outcome (BSt × Next)
  | [] => .stuck "block has no terminator"
  | i :: rest =>
      match i with
      | .ret => .ok (s, .done) s.world
      | .jump t args =>
          match args.mapM (getV s.vals) with
          | none => .stuck "jump argument is not defined"
          | some vs => .ok (s, .goto t.id vs) s.world
      | .brif c tb ta eb ea =>
          match getV s.vals c with
          | none => .stuck "branch condition is not defined"
          | some cv =>
              let (tgt, args) := if isTrue cv then (tb, ta) else (eb, ea)
              match args.mapM (getV s.vals) with
              | none => .stuck "branch argument is not defined"
              | some vs => .ok (s, .goto tgt.id vs) s.world
      | .store v a =>
          match doStore s v a none with
          | .error m => .stuck m
          | .ok s' => runInsts env s' rest
      | .storeTyped t v a =>
          match doStore s v a (some t) with
          | .error m => .stuck m
          | .ok s' => runInsts env s' rest
      | .istore8 v a =>
          match getV s.vals v, getV s.vals a with
          | some (.sc _ b), some (.sc _ addr) =>
              match s.world.mem.store addr 1 (b &&& 0xff) with
              | some m => runInsts env
                  { s with world := { obsStore s.world addr 1 (b &&& 0xff) with mem := m } } rest
              | none => .stuck "istore8 to unmapped address"
          | _, _ => .stuck "istore8 operand is not defined"
      | .call d fn args =>
          match args.mapM (getV s.vals) with
          | none => .stuck "call argument is not defined"
          | some vs =>
              match env.fns.find? (·.ref.id == fn.id) with
              | none => .stuck s!"fn{fn.id} is not declared"
              | some decl =>
                  match decl.callee with
                  | .local i => .stuck s!"fn{fn.id} calls u0:{i}"
                  | .import name =>
                      match callFile name vs (obsCall s.world fn.id vs) with
                      | none => .stuck s!"{name} has no executable contract"
                      | some (res, w') =>
                          match d, res with
                          | some dv, some r =>
                              runInsts env { vals := setV s.vals dv r, world := w' } rest
                          | none, _ => runInsts env { s with world := w' } rest
                          | some _, none => .stuck s!"{name} returned nothing to bind"
      | other =>
          match evalInst s.world.mem s.vals other with
          | none => .stuck "instruction is undefined here"
          | some (d, r) => runInsts env { s with vals := setV s.vals d r } rest

/-- Run a function from its entry block until it returns. `steps` bounds the
    number of block entries, so a loop that does not terminate is stuck — and
    it is the structural argument, so this has equations. -/
def runFrom (env : FnEnv) (f : FuncData) : Nat → BSt → Nat → List V → Outcome World
  | 0, _, _, _ => .stuck "block budget exhausted"
  | steps + 1, s, blk, args =>
    match f.blocks.find? (·.ref.id == blk) with
    | none => .stuck s!"block{blk} does not exist"
    | some b =>
        -- Block parameters are bound on entry; this is the correspondence the
        -- carry and join numbering has to get right.
        if b.params.length != args.length then
          .stuck s!"block{blk} takes {b.params.length} arguments, given {args.length}"
        else
          let vals := (b.params.zip args).foldl
            (fun vs ((v, _), x) => setV vs v x) s.vals
          match runInsts env { s with vals } b.insts with
          | .stuck m => .stuck m
          | .ok (s', next) w =>
              match next with
              | .done => .ok w w
              | .goto t vs => runFrom env f steps { s' with world := w } t vs

/-- Execute a compiled function, and hand back its observation trace in
    program order — the same shape `Sem.run` produces for the term. -/
def run (env : FnEnv) (f : FuncData) (args : List V) (w : World)
    (steps : Nat := 100000000) : Outcome (List Obs) :=
  match runFrom env f steps { vals := #[], world := w } 0 args with
  | .stuck m => .stuck m
  | .ok w' _ => .ok w'.obs.reverse w'

end AlgorithmLib.HProg.Blocks

-- ---------------------------------------------------------------------------
-- `compile_sound`
-- ---------------------------------------------------------------------------

namespace AlgorithmLib.HProg

open AlgorithmLib.IR
open AlgorithmLib.HProg.Sem

/-- **The statement.** A term and the function `compileFn` builds from it make
    the same observations, in the same order, and leave the same memory.

    The trace is the right grade: it pins each call with its concrete arguments
    and each store with its address and width — what a caller can actually
    detect — while staying weaker than full memory equivalence, so it can be
    strengthened later without being restated.

    Both sides call the same `Sem.evalOp`, so this says nothing about what the
    arithmetic *means*; only that compiling preserves the order and the
    arguments of it. That is the entire content, and it is exactly where the two
    forms have disagreed twice in practice: loop-exit and branch-join
    parameters. -/
def CompileSound (idx : Nat) (env : FnEnv) (params : List ClifTy) (c : Code)
    (args : List V) (w : World) (fuel : Nat) : Prop :=
  Sem.run { env, steps := fuel } args w c
    = Blocks.run env (compileFn idx env params c) args w fuel


/-- The base case, proved. An empty body compiles to one block that returns,
    and neither side observes anything.

    Small, but not vacuous, and it is the case the induction over `Code` rests
    on: it fixes the entry correspondence — that the term's parameters and the
    compiled entry block's parameters are the same values, in the same order,
    at the same indices. Everything else is built on top of that. -/
theorem empty_sound (idx : Nat) (env : FnEnv) (a : V) (w : World) (fuel : Nat) :
    CompileSound idx env ptrParams [] [a] w (fuel + 1) := by
  have hEmit : ∀ s : CS, emitCode HProg.fuel s [] = s := by
    intro s; simp [emitCode, HProg.fuel]
  simp [CompileSound, Sem.run, Blocks.run, Sem.runCode, compileFn, ptrParams,
        CS.open', CS.open'.go, CS.close, CS.fresh, Blocks.runFrom, Blocks.runInsts,
        Blocks.setV, hEmit]

/-- One step past the base case: a body that computes a single constant.

    The first case where the two forms actually do different work — the term
    pushes onto a slot environment, the compiled block writes a value the
    emitter numbered — and it goes through because for straight-line code the
    numbering is the identity: slot `i` is `Val i`. That correspondence is what
    the straight-line induction generalises. -/
theorem single_iconst_sound (idx : Nat) (env : FnEnv) (t : ClifTy) (k : Int)
    (a : V) (w : World) (fuel : Nat) :
    CompileSound idx env ptrParams [.straight [.op (.iconst t k)]] [a] w (fuel + 2) := by
  have hEmit : ∀ s : CS,
      emitCode HProg.fuel s [Piece.straight [Stmt.op (Op.iconst t k)]]
        = emitStmt s (Stmt.op (Op.iconst t k)) := by
    intro s; simp [emitCode, emitPiece, emitStmts, HProg.fuel]
  simp [CompileSound, Sem.run, Blocks.run, Sem.runCode, Sem.runPiece, Sem.runStmts,
        Sem.runStmt, compileFn, ptrParams, CS.open', CS.open'.go, CS.close, CS.fresh, CS.get,
        Blocks.runFrom, Blocks.runInsts, Blocks.setV, Blocks.evalInst, Blocks.viaOp,
        emitStmt, hEmit, Sem.evalOp]

/-- For straight-line code the emitter's slot-to-value map is the **identity**:
    `emitStmt` advances `nextVal` and `slots` together, so slot `i` is always
    `Val i`.

    This is the invariant the straight-line induction needs. Stating it is what
    makes that induction possible at all — unfolding the definitions and letting
    `simp` reduce works for a single statement and times out at two, so the
    proof has to reason about the emitter rather than evaluate it. -/
def Aligned (s : CS) (n : Nat) : Prop :=
  s.nextVal = n ∧ s.slots = n ∧ ∀ i, i < n → s.env.lookup i = some ⟨i⟩

/-- Looking past a binding for a different slot. Isolated because it is the
    one step the alignment proof turns on. -/
theorem lookup_cons_ne (i n : Nat) (v : Val) (env : List (Nat × Val)) (h : i ≠ n) :
    List.lookup i ((n, v) :: env) = List.lookup i env := by
  have : (i == n) = false := by simp [h]
  simp [List.lookup, this]

/-- **Emitting one statement preserves the alignment**, advancing it by exactly
    the number of slots that statement defines.

    General in the statement and in the state — the first result here that is
    not about a particular program, and the invariant the straight-line
    induction runs on. -/
theorem emitStmt_aligned (s : CS) (n : Nat) (h : Aligned s n) (st : Stmt) :
    Aligned (emitStmt s st) (n + st.binds) := by
  obtain ⟨hv, hs, he⟩ := h
  cases st with
  | op o =>
      refine ⟨by simp [emitStmt, CS.fresh, hv, Stmt.binds], by
        simp [emitStmt, CS.fresh, hs, Stmt.binds], ?_⟩
      intro i hi
      simp [Stmt.binds] at hi
      by_cases hin : i = n
      · subst hin
        simp [emitStmt, CS.fresh, hs, hv, List.lookup]
      · have hlt : i < n := by omega
        simp only [emitStmt, CS.fresh, hs]
        rw [lookup_cons_ne i n _ _ hin]
        exact he i hlt
  | call fn args =>
      refine ⟨by simp [emitStmt, CS.fresh, hv, Stmt.binds], by
        simp [emitStmt, CS.fresh, hs, Stmt.binds], ?_⟩
      intro i hi
      simp [Stmt.binds] at hi
      by_cases hin : i = n
      · subst hin
        simp [emitStmt, CS.fresh, hs, hv, List.lookup]
      · have hlt : i < n := by omega
        simp only [emitStmt, CS.fresh, hs]
        rw [lookup_cons_ne i n _ _ hin]
        exact he i hlt
  | store t v a => exact ⟨by simpa [emitStmt, Stmt.binds] using hv,
      by simpa [emitStmt, Stmt.binds] using hs, by simpa [emitStmt, Stmt.binds] using he⟩
  | storeUnaligned v a => exact ⟨by simpa [emitStmt, Stmt.binds] using hv,
      by simpa [emitStmt, Stmt.binds] using hs, by simpa [emitStmt, Stmt.binds] using he⟩
  | istore8 v a => exact ⟨by simpa [emitStmt, Stmt.binds] using hv,
      by simpa [emitStmt, Stmt.binds] using hs, by simpa [emitStmt, Stmt.binds] using he⟩
  | callVoid fn args => exact ⟨by simpa [emitStmt, Stmt.binds] using hv,
      by simpa [emitStmt, Stmt.binds] using hs, by simpa [emitStmt, Stmt.binds] using he⟩

/-- **Alignment survives a whole statement list.** The emitter's slot-to-value
    map stays the identity across any straight-line run, advancing by the total
    number of slots the statements bind.

    This is the structural half of the straight-line case: it says *where* every
    value ends up. What remains is the dynamic half — that running the emitted
    instructions from `vals = Γ` leaves `vals' = Γ'`. -/
theorem emitStmts_aligned : ∀ (ss : List Stmt) (s : CS) (n : Nat), Aligned s n →
    Aligned (emitStmts s ss) (n + (ss.map Stmt.binds).sum)
  | [], s, n, h => by simpa [emitStmts] using h
  | a :: as, s, n, h => by
      have hstep := emitStmt_aligned s n h a
      have hrest := emitStmts_aligned as (emitStmt s a) (n + a.binds) hstep
      simpa [emitStmts, List.foldl, Nat.add_assoc] using hrest

/-- Emitting only ever *prepends* to the open instruction list: whatever was
    already open stays at the bottom, untouched.

    The next brick under the dynamic half. `emitStmt` accumulates into `cur`
    reversed and `CS.close` reverses it back, so relating the run of a block to
    the run of its statements needs the shape of that accumulation pinned
    first. Stated as a suffix rather than an existential — the witness is a
    match on the operation, which unification will not guess. -/
theorem emitStmt_cur_suffix (s : CS) (st : Stmt) : s.cur <:+ (emitStmt s st).cur := by
  cases st <;> simp [emitStmt, CS.fresh, List.suffix_cons]

/-- And so does emitting a whole list. -/
theorem emitStmts_cur_suffix : ∀ (ss : List Stmt) (s : CS), s.cur <:+ (emitStmts s ss).cur
  | [], s => by simp [emitStmts]
  | a :: as, s => by
      have h1 := emitStmt_cur_suffix s a
      have h2 := emitStmts_cur_suffix as (emitStmt s a)
      simpa [emitStmts] using h1.trans h2

/-- **Every statement emits exactly one instruction.** With the suffix result
    this pins the accumulation completely: `cur` grows by a single instruction
    on the front, so the emitted segment for a list of `k` statements is the
    first `k` of the reversed list — which is what lets `runInsts` be stepped
    through in lockstep with `runStmts`. -/
theorem emitStmt_cur_length (s : CS) (st : Stmt) :
    (emitStmt s st).cur.length = s.cur.length + 1 := by
  cases st <;> simp [emitStmt, CS.fresh]

/-- So a list of statements emits exactly one instruction per statement. -/
theorem emitStmts_cur_length : ∀ (ss : List Stmt) (s : CS),
    (emitStmts s ss).cur.length = s.cur.length + ss.length
  | [], s => by simp [emitStmts]
  | a :: as, s => by
      have h := emitStmts_cur_length as (emitStmt s a)
      rw [emitStmt_cur_length] at h
      simpa [emitStmts, Nat.add_assoc, Nat.add_comm 1] using h

/-- **The bridge from the invariant to the emitted operands.** Under alignment,
    the emitter resolves slot `i` to `Val i` — so every instruction it writes
    names its operands at exactly the indices the term's environment uses.

    This is what turns `Aligned` from a statement about the emitter's
    bookkeeping into a statement about the instructions it produces, and it is
    the last structural fact the dynamic step needs. -/
theorem get_of_aligned (s : CS) (n : Nat) (h : Aligned s n) (i : Nat) (hi : i < n) :
    s.get i = ⟨i⟩ := by
  obtain ⟨_, _, he⟩ := h
  simp [CS.get, he i hi]

/-- The two value stores agree: the block's array holds at index `i` whatever
    the term's slot environment holds at slot `i`. The coupling the dynamic
    step is stated against. -/
def ValsAgree (vals : Blocks.Vals) (Γ : Sem.Env) : Prop :=
  ∀ i, i < Γ.size → Blocks.getV vals ⟨i⟩ = Γ[i]?

/-- Agreement, as the two interpreters need to use it: an in-scope slot holds
    *some* value, and both sides find the same one. -/
theorem agree_at (vals : Blocks.Vals) (Γ : Sem.Env) (hv : ValsAgree vals Γ)
    (i : Nat) (hi : i < Γ.size) :
    ∃ x, Γ[i]? = some x ∧ Blocks.getV vals ⟨i⟩ = some x := by
  refine ⟨Γ[i], ?_, ?_⟩
  · simp [getElem?_pos, hi]
  · rw [hv i hi]; simp [getElem?_pos, hi]

/-- **The dynamic step, for a two-operand shape.**

    `emitStmt` puts the instruction at the head of the block, naming the fresh
    value `n` and resolving each operand through `CS.get`; `evalInst` then
    evaluates it against the block's array. This says that computes exactly
    what the term's `evalOp` computes against `Γ` — which is the relocation
    property of the operation, plus `Aligned` to make `CS.get` the identity and
    `ValsAgree` to make the two value stores interchangeable.

    Stated over the emitted `Inst` rather than over `emitStmt` so one proof
    covers every shape that compiles to a two-operand instruction; the caller
    supplies the two facts that differ per shape. -/
theorem dyn2 (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hv : ValsAgree vals Γ) (hΓ : Γ.size = n) (f : R → R → Op)
    (hf : Sem.Reloc2 f) (a b : R) (ha : a < n) (hb : b < n)
    (inst : Inst)
    (hi : Blocks.evalInst m vals inst
            = (do let x ← Blocks.getV vals ⟨a⟩
                  let y ← Blocks.getV vals ⟨b⟩
                  pure (⟨n⟩, ← Sem.evalOp m #[x, y] (f 0 1)))) :
    Blocks.evalInst m vals inst = (Sem.evalOp m Γ (f a b)).map (fun w => (⟨n⟩, w)) := by
  obtain ⟨x, hxΓ, hxv⟩ := agree_at vals Γ hv a (hΓ ▸ ha)
  obtain ⟨y, hyΓ, hyv⟩ := agree_at vals Γ hv b (hΓ ▸ hb)
  rw [hi, hf m Γ a b x y hxΓ hyΓ, hxv, hyv]
  cases hE : Sem.evalOp m #[x, y] (f 0 1) <;> simp [hE]

/-- The same, for a one-operand shape. -/
theorem dyn1 (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hv : ValsAgree vals Γ) (hΓ : Γ.size = n) (f : R → Op)
    (hf : Sem.Reloc1 f) (a : R) (ha : a < n) (inst : Inst)
    (hi : Blocks.evalInst m vals inst
            = (do let x ← Blocks.getV vals ⟨a⟩
                  pure (⟨n⟩, ← Sem.evalOp m #[x] (f 0)))) :
    Blocks.evalInst m vals inst = (Sem.evalOp m Γ (f a)).map (fun w => (⟨n⟩, w)) := by
  obtain ⟨x, hxΓ, hxv⟩ := agree_at vals Γ hv a (hΓ ▸ ha)
  rw [hi, hf m Γ a x hxΓ, hxv]
  cases hE : Sem.evalOp m #[x] (f 0) <;> simp [hE]

/-- The same, for the one three-operand shape. -/
theorem dyn3 (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hv : ValsAgree vals Γ) (hΓ : Γ.size = n) (f : R → R → R → Op)
    (hf : Sem.Reloc3 f) (a b c : R) (ha : a < n) (hb : b < n) (hc : c < n)
    (inst : Inst)
    (hi : Blocks.evalInst m vals inst
            = (do let x ← Blocks.getV vals ⟨a⟩
                  let y ← Blocks.getV vals ⟨b⟩
                  let z ← Blocks.getV vals ⟨c⟩
                  pure (⟨n⟩, ← Sem.evalOp m #[x, y, z] (f 0 1 2)))) :
    Blocks.evalInst m vals inst = (Sem.evalOp m Γ (f a b c)).map (fun w => (⟨n⟩, w)) := by
  obtain ⟨x, hxΓ, hxv⟩ := agree_at vals Γ hv a (hΓ ▸ ha)
  obtain ⟨y, hyΓ, hyv⟩ := agree_at vals Γ hv b (hΓ ▸ hb)
  obtain ⟨z, hzΓ, hzv⟩ := agree_at vals Γ hv c (hΓ ▸ hc)
  rw [hi, hf m Γ a b c x y z hxΓ hyΓ hzΓ, hxv, hyv, hzv]
  cases hE : Sem.evalOp m #[x, y, z] (f 0 1 2) <;> simp [hE]

/-- What `emitStmt` puts at the head of the block for `iadd`: the fresh value is
    `n`, and both operands resolve to themselves because `s` is aligned. -/
theorem emit_iadd (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.iadd a b))).cur = .iadd ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

/-- `iadd`, end to end: the instruction `emitStmt` compiles to computes what the
    term's `evalOp` computes. The pattern every two-operand shape follows. -/
theorem eval_iadd (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (a b : R)
    (hab : a < n) (hbb : b < n) :
    Blocks.evalInst m vals (.iadd ⟨n⟩ ⟨a⟩ ⟨b⟩)
      = (Sem.evalOp m Γ (.iadd a b)).map (fun w => (⟨n⟩, w)) :=
  dyn2 m vals Γ n hv hΓ Op.iadd Sem.reloc2_iadd a b hab hbb _ rfl

theorem emit_isub (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.isub a b))).cur = .isub ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem eval_isub (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (a b : R)
    (hab : a < n) (hbb : b < n) :
    Blocks.evalInst m vals (.isub ⟨n⟩ ⟨a⟩ ⟨b⟩)
      = (Sem.evalOp m Γ (.isub a b)).map (fun w => (⟨n⟩, w)) :=
  dyn2 m vals Γ n hv hΓ Op.isub Sem.reloc2_isub a b hab hbb _ rfl

theorem emit_imul (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.imul a b))).cur = .imul ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem eval_imul (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (a b : R)
    (hab : a < n) (hbb : b < n) :
    Blocks.evalInst m vals (.imul ⟨n⟩ ⟨a⟩ ⟨b⟩)
      = (Sem.evalOp m Γ (.imul a b)).map (fun w => (⟨n⟩, w)) :=
  dyn2 m vals Γ n hv hΓ Op.imul Sem.reloc2_imul a b hab hbb _ rfl

theorem emit_udiv (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.udiv a b))).cur = .udiv ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem eval_udiv (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (a b : R)
    (hab : a < n) (hbb : b < n) :
    Blocks.evalInst m vals (.udiv ⟨n⟩ ⟨a⟩ ⟨b⟩)
      = (Sem.evalOp m Γ (.udiv a b)).map (fun w => (⟨n⟩, w)) :=
  dyn2 m vals Γ n hv hΓ Op.udiv Sem.reloc2_udiv a b hab hbb _ rfl

theorem emit_ishl (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.ishl a b))).cur = .ishl ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem eval_ishl (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (a b : R)
    (hab : a < n) (hbb : b < n) :
    Blocks.evalInst m vals (.ishl ⟨n⟩ ⟨a⟩ ⟨b⟩)
      = (Sem.evalOp m Γ (.ishl a b)).map (fun w => (⟨n⟩, w)) :=
  dyn2 m vals Γ n hv hΓ Op.ishl Sem.reloc2_ishl a b hab hbb _ rfl

theorem emit_ushr (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.ushr a b))).cur = .ushr ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem eval_ushr (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (a b : R)
    (hab : a < n) (hbb : b < n) :
    Blocks.evalInst m vals (.ushr ⟨n⟩ ⟨a⟩ ⟨b⟩)
      = (Sem.evalOp m Γ (.ushr a b)).map (fun w => (⟨n⟩, w)) :=
  dyn2 m vals Γ n hv hΓ Op.ushr Sem.reloc2_ushr a b hab hbb _ rfl

theorem emit_band (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.band a b))).cur = .band ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem eval_band (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (a b : R)
    (hab : a < n) (hbb : b < n) :
    Blocks.evalInst m vals (.band ⟨n⟩ ⟨a⟩ ⟨b⟩)
      = (Sem.evalOp m Γ (.band a b)).map (fun w => (⟨n⟩, w)) :=
  dyn2 m vals Γ n hv hΓ Op.band Sem.reloc2_band a b hab hbb _ rfl

theorem emit_bandNot (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.bandNot a b))).cur = .bandNot ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem eval_bandNot (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (a b : R)
    (hab : a < n) (hbb : b < n) :
    Blocks.evalInst m vals (.bandNot ⟨n⟩ ⟨a⟩ ⟨b⟩)
      = (Sem.evalOp m Γ (.bandNot a b)).map (fun w => (⟨n⟩, w)) :=
  dyn2 m vals Γ n hv hΓ Op.bandNot Sem.reloc2_bandNot a b hab hbb _ rfl

theorem emit_bor (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.bor a b))).cur = .bor ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem eval_bor (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (a b : R)
    (hab : a < n) (hbb : b < n) :
    Blocks.evalInst m vals (.bor ⟨n⟩ ⟨a⟩ ⟨b⟩)
      = (Sem.evalOp m Γ (.bor a b)).map (fun w => (⟨n⟩, w)) :=
  dyn2 m vals Γ n hv hΓ Op.bor Sem.reloc2_bor a b hab hbb _ rfl

theorem emit_bxor (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.bxor a b))).cur = .bxor ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem eval_bxor (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (a b : R)
    (hab : a < n) (hbb : b < n) :
    Blocks.evalInst m vals (.bxor ⟨n⟩ ⟨a⟩ ⟨b⟩)
      = (Sem.evalOp m Γ (.bxor a b)).map (fun w => (⟨n⟩, w)) :=
  dyn2 m vals Γ n hv hΓ Op.bxor Sem.reloc2_bxor a b hab hbb _ rfl

theorem emit_fadd (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.fadd a b))).cur = .fadd ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem eval_fadd (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (a b : R)
    (hab : a < n) (hbb : b < n) :
    Blocks.evalInst m vals (.fadd ⟨n⟩ ⟨a⟩ ⟨b⟩)
      = (Sem.evalOp m Γ (.fadd a b)).map (fun w => (⟨n⟩, w)) :=
  dyn2 m vals Γ n hv hΓ Op.fadd Sem.reloc2_fadd a b hab hbb _ rfl

theorem emit_fsub (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.fsub a b))).cur = .fsub ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem eval_fsub (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (a b : R)
    (hab : a < n) (hbb : b < n) :
    Blocks.evalInst m vals (.fsub ⟨n⟩ ⟨a⟩ ⟨b⟩)
      = (Sem.evalOp m Γ (.fsub a b)).map (fun w => (⟨n⟩, w)) :=
  dyn2 m vals Γ n hv hΓ Op.fsub Sem.reloc2_fsub a b hab hbb _ rfl

theorem emit_fmul (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.fmul a b))).cur = .fmul ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem eval_fmul (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (a b : R)
    (hab : a < n) (hbb : b < n) :
    Blocks.evalInst m vals (.fmul ⟨n⟩ ⟨a⟩ ⟨b⟩)
      = (Sem.evalOp m Γ (.fmul a b)).map (fun w => (⟨n⟩, w)) :=
  dyn2 m vals Γ n hv hΓ Op.fmul Sem.reloc2_fmul a b hab hbb _ rfl

theorem emit_fmax (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.fmax a b))).cur = .fmax ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem eval_fmax (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (a b : R)
    (hab : a < n) (hbb : b < n) :
    Blocks.evalInst m vals (.fmax ⟨n⟩ ⟨a⟩ ⟨b⟩)
      = (Sem.evalOp m Γ (.fmax a b)).map (fun w => (⟨n⟩, w)) :=
  dyn2 m vals Γ n hv hΓ Op.fmax Sem.reloc2_fmax a b hab hbb _ rfl

theorem emit_fmin (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.fmin a b))).cur = .fmin ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem eval_fmin (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (a b : R)
    (hab : a < n) (hbb : b < n) :
    Blocks.evalInst m vals (.fmin ⟨n⟩ ⟨a⟩ ⟨b⟩)
      = (Sem.evalOp m Γ (.fmin a b)).map (fun w => (⟨n⟩, w)) :=
  dyn2 m vals Γ n hv hΓ Op.fmin Sem.reloc2_fmin a b hab hbb _ rfl

theorem emit_ineg (s : CS) (n : Nat) (ha : Aligned s n) (a : R) (hab : a < n) :
    (emitStmt s (.op (.ineg a))).cur = .ineg ⟨n⟩ ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem eval_ineg (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (a : R) (hab : a < n) :
    Blocks.evalInst m vals (.ineg ⟨n⟩ ⟨a⟩)
      = (Sem.evalOp m Γ (.ineg a)).map (fun w => (⟨n⟩, w)) :=
  dyn1 m vals Γ n hv hΓ Op.ineg Sem.reloc1_ineg a hab _ rfl

theorem emit_ctz (s : CS) (n : Nat) (ha : Aligned s n) (a : R) (hab : a < n) :
    (emitStmt s (.op (.ctz a))).cur = .ctz ⟨n⟩ ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem eval_ctz (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (a : R) (hab : a < n) :
    Blocks.evalInst m vals (.ctz ⟨n⟩ ⟨a⟩)
      = (Sem.evalOp m Γ (.ctz a)).map (fun w => (⟨n⟩, w)) :=
  dyn1 m vals Γ n hv hΓ Op.ctz Sem.reloc1_ctz a hab _ rfl

theorem emit_popcnt (s : CS) (n : Nat) (ha : Aligned s n) (a : R) (hab : a < n) :
    (emitStmt s (.op (.popcnt a))).cur = .popcnt ⟨n⟩ ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem eval_popcnt (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (a : R) (hab : a < n) :
    Blocks.evalInst m vals (.popcnt ⟨n⟩ ⟨a⟩)
      = (Sem.evalOp m Γ (.popcnt a)).map (fun w => (⟨n⟩, w)) :=
  dyn1 m vals Γ n hv hΓ Op.popcnt Sem.reloc1_popcnt a hab _ rfl

theorem emit_ireduce32 (s : CS) (n : Nat) (ha : Aligned s n) (a : R) (hab : a < n) :
    (emitStmt s (.op (.ireduce32 a))).cur = .ireduce32 ⟨n⟩ ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem eval_ireduce32 (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (a : R) (hab : a < n) :
    Blocks.evalInst m vals (.ireduce32 ⟨n⟩ ⟨a⟩)
      = (Sem.evalOp m Γ (.ireduce32 a)).map (fun w => (⟨n⟩, w)) :=
  dyn1 m vals Γ n hv hΓ Op.ireduce32 Sem.reloc1_ireduce32 a hab _ rfl

theorem emit_uextend64 (s : CS) (n : Nat) (ha : Aligned s n) (a : R) (hab : a < n) :
    (emitStmt s (.op (.uextend64 a))).cur = .uextend64 ⟨n⟩ ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem eval_uextend64 (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (a : R) (hab : a < n) :
    Blocks.evalInst m vals (.uextend64 ⟨n⟩ ⟨a⟩)
      = (Sem.evalOp m Γ (.uextend64 a)).map (fun w => (⟨n⟩, w)) :=
  dyn1 m vals Γ n hv hΓ Op.uextend64 Sem.reloc1_uextend64 a hab _ rfl

theorem emit_sextend64 (s : CS) (n : Nat) (ha : Aligned s n) (a : R) (hab : a < n) :
    (emitStmt s (.op (.sextend64 a))).cur = .sextend64 ⟨n⟩ ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem eval_sextend64 (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (a : R) (hab : a < n) :
    Blocks.evalInst m vals (.sextend64 ⟨n⟩ ⟨a⟩)
      = (Sem.evalOp m Γ (.sextend64 a)).map (fun w => (⟨n⟩, w)) :=
  dyn1 m vals Γ n hv hΓ Op.sextend64 Sem.reloc1_sextend64 a hab _ rfl

theorem emit_fneg (s : CS) (n : Nat) (ha : Aligned s n) (a : R) (hab : a < n) :
    (emitStmt s (.op (.fneg a))).cur = .fneg ⟨n⟩ ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem eval_fneg (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (a : R) (hab : a < n) :
    Blocks.evalInst m vals (.fneg ⟨n⟩ ⟨a⟩)
      = (Sem.evalOp m Γ (.fneg a)).map (fun w => (⟨n⟩, w)) :=
  dyn1 m vals Γ n hv hΓ Op.fneg Sem.reloc1_fneg a hab _ rfl

theorem emit_fpromote (s : CS) (n : Nat) (ha : Aligned s n) (a : R) (hab : a < n) :
    (emitStmt s (.op (.fpromote a))).cur = .fpromote ⟨n⟩ ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem eval_fpromote (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (a : R) (hab : a < n) :
    Blocks.evalInst m vals (.fpromote ⟨n⟩ ⟨a⟩)
      = (Sem.evalOp m Γ (.fpromote a)).map (fun w => (⟨n⟩, w)) :=
  dyn1 m vals Γ n hv hΓ Op.fpromote Sem.reloc1_fpromote a hab _ rfl

theorem emit_vhighBits (s : CS) (n : Nat) (ha : Aligned s n) (a : R) (hab : a < n) :
    (emitStmt s (.op (.vhighBits a))).cur = .vhighBits ⟨n⟩ ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem eval_vhighBits (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (a : R) (hab : a < n) :
    Blocks.evalInst m vals (.vhighBits ⟨n⟩ ⟨a⟩)
      = (Sem.evalOp m Γ (.vhighBits a)).map (fun w => (⟨n⟩, w)) :=
  dyn1 m vals Γ n hv hΓ Op.vhighBits Sem.reloc1_vhighBits a hab _ rfl

theorem emit_fcvtFromSint (s : CS) (n : Nat) (ha : Aligned s n) (ty : ClifTy) (a : R)
    (hab : a < n) :
    (emitStmt s (.op (.fcvtFromSint ty a))).cur = .fcvtFromSint ⟨n⟩ ty ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem eval_fcvtFromSint (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (ty : ClifTy) (a : R) (hab : a < n) :
    Blocks.evalInst m vals (.fcvtFromSint ⟨n⟩ ty ⟨a⟩)
      = (Sem.evalOp m Γ (.fcvtFromSint ty a)).map (fun w => (⟨n⟩, w)) :=
  dyn1 m vals Γ n hv hΓ (Op.fcvtFromSint ty) (Sem.reloc1_fcvtFromSint ty) a hab _ rfl

theorem emit_fcvtToUint (s : CS) (n : Nat) (ha : Aligned s n) (ty : ClifTy) (a : R)
    (hab : a < n) :
    (emitStmt s (.op (.fcvtToUint ty a))).cur = .fcvtToUint ⟨n⟩ ty ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem eval_fcvtToUint (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (ty : ClifTy) (a : R) (hab : a < n) :
    Blocks.evalInst m vals (.fcvtToUint ⟨n⟩ ty ⟨a⟩)
      = (Sem.evalOp m Γ (.fcvtToUint ty a)).map (fun w => (⟨n⟩, w)) :=
  dyn1 m vals Γ n hv hΓ (Op.fcvtToUint ty) (Sem.reloc1_fcvtToUint ty) a hab _ rfl

theorem emit_splat (s : CS) (n : Nat) (ha : Aligned s n) (ty : ClifTy) (a : R)
    (hab : a < n) :
    (emitStmt s (.op (.splat ty a))).cur = .splat ⟨n⟩ ty ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem eval_splat (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (ty : ClifTy) (a : R) (hab : a < n) :
    Blocks.evalInst m vals (.splat ⟨n⟩ ty ⟨a⟩)
      = (Sem.evalOp m Γ (.splat ty a)).map (fun w => (⟨n⟩, w)) :=
  dyn1 m vals Γ n hv hΓ (Op.splat ty) (Sem.reloc1_splat ty) a hab _ rfl

theorem emit_bitcast (s : CS) (n : Nat) (ha : Aligned s n) (ty : ClifTy) (a : R)
    (hab : a < n) :
    (emitStmt s (.op (.bitcast ty a))).cur = .bitcast ⟨n⟩ ty ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem eval_bitcast (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (ty : ClifTy) (a : R) (hab : a < n) :
    Blocks.evalInst m vals (.bitcast ⟨n⟩ ty ⟨a⟩)
      = (Sem.evalOp m Γ (.bitcast ty a)).map (fun w => (⟨n⟩, w)) :=
  dyn1 m vals Γ n hv hΓ (Op.bitcast ty) (Sem.reloc1_bitcast ty) a hab _ rfl

theorem emit_icmp (s : CS) (n : Nat) (ha : Aligned s n) (c : ICmpCond) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.icmp c a b))).cur = .icmp ⟨n⟩ c ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem eval_icmp (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (c : ICmpCond) (a b : R)
    (hab : a < n) (hbb : b < n) :
    Blocks.evalInst m vals (.icmp ⟨n⟩ c ⟨a⟩ ⟨b⟩)
      = (Sem.evalOp m Γ (.icmp c a b)).map (fun w => (⟨n⟩, w)) :=
  dyn2 m vals Γ n hv hΓ (Op.icmp c) (Sem.reloc2_icmp c) a b hab hbb _ rfl

theorem emit_fcmp (s : CS) (n : Nat) (ha : Aligned s n) (c : FloatCC) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.fcmp c a b))).cur = .fcmp ⟨n⟩ c ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem eval_fcmp (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (c : FloatCC) (a b : R)
    (hab : a < n) (hbb : b < n) :
    Blocks.evalInst m vals (.fcmp ⟨n⟩ c ⟨a⟩ ⟨b⟩)
      = (Sem.evalOp m Γ (.fcmp c a b)).map (fun w => (⟨n⟩, w)) :=
  dyn2 m vals Γ n hv hΓ (Op.fcmp c) (Sem.reloc2_fcmp c) a b hab hbb _ rfl

theorem emit_load (s : CS) (n : Nat) (ha : Aligned s n) (op : LoadOp) (a : R)
    (hab : a < n) :
    (emitStmt s (.op (.load op a))).cur = .load ⟨n⟩ op ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem eval_load (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (op : LoadOp) (a : R) (hab : a < n) :
    Blocks.evalInst m vals (.load ⟨n⟩ op ⟨a⟩)
      = (Sem.evalOp m Γ (.load op a)).map (fun w => (⟨n⟩, w)) :=
  dyn1 m vals Γ n hv hΓ (Op.load op) (Sem.reloc1_load op) a hab _ rfl

theorem emit_extractlane (s : CS) (n : Nat) (ha : Aligned s n) (a : R) (l : Nat)
    (hab : a < n) :
    (emitStmt s (.op (.extractlane a l))).cur = .extractlane ⟨n⟩ ⟨a⟩ l :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem eval_extractlane (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (a : R) (l : Nat) (hab : a < n) :
    Blocks.evalInst m vals (.extractlane ⟨n⟩ ⟨a⟩ l)
      = (Sem.evalOp m Γ (.extractlane a l)).map (fun w => (⟨n⟩, w)) :=
  dyn1 m vals Γ n hv hΓ (Op.extractlane · l) (Sem.reloc1_extractlane l) a hab _ rfl

theorem emit_select (s : CS) (n : Nat) (ha : Aligned s n) (c a b : R)
    (hcb : c < n) (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.select c a b))).cur = .select ⟨n⟩ ⟨c⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he c hcb, he a hab, he b hbb, h1]

theorem eval_select (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (hΓ : Γ.size = n) (hv : ValsAgree vals Γ) (c a b : R)
    (hcb : c < n) (hab : a < n) (hbb : b < n) :
    Blocks.evalInst m vals (.select ⟨n⟩ ⟨c⟩ ⟨a⟩ ⟨b⟩)
      = (Sem.evalOp m Γ (.select c a b)).map (fun w => (⟨n⟩, w)) :=
  dyn3 m vals Γ n hv hΓ Op.select Sem.reloc3_select c a b hcb hab hbb _ rfl

theorem emit_iconst (s : CS) (n : Nat) (ha : Aligned s n) (ty : ClifTy) (k : Int) :
    (emitStmt s (.op (.iconst ty k))).cur = .iconst ⟨n⟩ ty k :: s.cur := by
  obtain ⟨h1, _, _⟩ := ha
  simp [emitStmt, CS.fresh, h1]

theorem eval_iconst (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (ty : ClifTy) (k : Int) :
    Blocks.evalInst m vals (.iconst ⟨n⟩ ty k)
      = (Sem.evalOp m Γ (.iconst ty k)).map (fun w => (⟨n⟩, w)) := by
  simp [Blocks.evalInst, Blocks.viaOp, Sem.evalOp]

theorem emit_fconst (s : CS) (n : Nat) (ha : Aligned s n) (ty : ClifTy) (b : UInt64) :
    (emitStmt s (.op (.fconst ty b))).cur = .fconst ⟨n⟩ ty b :: s.cur := by
  obtain ⟨h1, _, _⟩ := ha
  simp [emitStmt, CS.fresh, h1]

theorem eval_fconst (m : Sem.Mem) (vals : Blocks.Vals) (Γ : Sem.Env) (n : Nat)
    (ty : ClifTy) (b : UInt64) :
    Blocks.evalInst m vals (.fconst ⟨n⟩ ty b)
      = (Sem.evalOp m Γ (.fconst ty b)).map (fun w => (⟨n⟩, w)) := by
  simp [Blocks.evalInst, Blocks.viaOp, Sem.evalOp]

-- ---------------------------------------------------------------------------
-- Agreement is preserved
-- ---------------------------------------------------------------------------

/-- Agreement forces the block's array to be at least as long as the term's
    environment: every in-scope slot has to *find* something on both sides.

    Needed before `setV`, which pads with `default` rather than failing — so
    without this an operation could appear to preserve agreement while filling
    a hole with a junk value. -/
theorem agree_size (vals : Blocks.Vals) (Γ : Sem.Env) (hv : ValsAgree vals Γ) :
    Γ.size ≤ vals.size := by
  apply Nat.le_of_not_lt
  intro hlt
  have hnone : Γ[vals.size]? = none := by
    rw [← hv vals.size hlt]; simp [Blocks.getV]
  rw [Array.getElem?_eq_none_iff] at hnone
  omega

/-- Writing the next value leaves every earlier one alone. -/
theorem getV_setV_lt (vals : Blocks.Vals) (n i : Nat) (w : Sem.V)
    (h : n ≤ vals.size) (hi : i < n) :
    Blocks.getV (Blocks.setV vals ⟨n⟩ w) ⟨i⟩ = Blocks.getV vals ⟨i⟩ := by
  have hne : i ≠ n := Nat.ne_of_lt hi
  have hne' : n ≠ i := Ne.symm hne
  rcases Nat.lt_or_ge n vals.size with hn | hn
  · simp [Blocks.getV, Blocks.setV, hn, Array.getElem?_setIfInBounds, hne, hne']
  · have heq : n = vals.size := Nat.le_antisymm h hn
    subst heq
    have hi' : i < (vals.push (default : Sem.V)).size := by simp; omega
    simp [Blocks.getV, Blocks.setV, Nat.lt_irrefl, Array.getElem?_setIfInBounds, hne, hne',
          Array.getElem?_push, hi]

/-- And puts the value where the term's next slot is. -/
theorem getV_setV_eq (vals : Blocks.Vals) (n : Nat) (w : Sem.V) (h : n ≤ vals.size) :
    Blocks.getV (Blocks.setV vals ⟨n⟩ w) ⟨n⟩ = some w := by
  rcases Nat.lt_or_ge n vals.size with hn | hn
  · simp [Blocks.getV, Blocks.setV, hn, Array.getElem?_setIfInBounds]
  · have : n = vals.size := Nat.le_antisymm h hn
    subst this
    simp [Blocks.getV, Blocks.setV, Nat.lt_irrefl, Array.getElem?_setIfInBounds]

/-- **One step preserves the coupling.** The term binds its result as the next
    slot; the compiled form writes it to the next value number. Since those are
    the same number under `Aligned`, the two stores stay in step.

    This is what carries `ValsAgree` along a straight-line body, and it is the
    reason the numbering has to be dense: a gap would put `setV` in the padding
    business and the invariant would be about `default` rather than about the
    term. -/
theorem agree_push (vals : Blocks.Vals) (Γ : Sem.Env) (hv : ValsAgree vals Γ)
    (n : Nat) (hΓ : Γ.size = n) (w : Sem.V) :
    ValsAgree (Blocks.setV vals ⟨n⟩ w) (Γ.push w) := by
  have hle : n ≤ vals.size := hΓ ▸ agree_size vals Γ hv
  intro i hi
  simp only [Array.size_push, hΓ] at hi
  rcases Nat.lt_or_ge i n with hlt | hge
  · rw [getV_setV_lt vals n i w hle hlt, hv i (hΓ ▸ hlt),
        Array.getElem?_push, hΓ]
    simp [Nat.ne_of_lt hlt]
  · have hin : i = n := Nat.le_antisymm (by omega) hge
    subst hin
    rw [getV_setV_eq vals i w hle, Array.getElem?_push, hΓ]
    simp

-- ---------------------------------------------------------------------------
-- One statement simulates
-- ---------------------------------------------------------------------------

/-- **The simulation step for a pure operation.**

    The term takes `Γ` to `Γ.push v` and leaves the world alone; the compiled
    block writes `v` to value `n` and carries on with the rest of the
    instructions. This says those are the same transition — and `agree_push`
    says the states it lands in still agree, so the pair is an invariant that
    survives one statement.

    `hstep` is the fact that `inst` reaches `runInsts`' evaluating arm rather
    than one of the special-cased ones; it holds by `rfl` at every concrete
    shape, which is how the caller discharges it. Taking it as a hypothesis is
    what keeps this one proof rather than thirty-five. -/
theorem step_op (env : FnEnv) (w : Sem.World) (Γ : Sem.Env) (vals : Blocks.Vals)
    (n : Nat) (o : Op) (inst : Inst) (rest : List Inst)
    (hE : Blocks.evalInst w.mem vals inst
            = (Sem.evalOp w.mem Γ o).map (fun x => (⟨n⟩, x)))
    (hstep : Blocks.runInsts env ⟨vals, w⟩ (inst :: rest)
        = match Blocks.evalInst w.mem vals inst with
          | none => .stuck "instruction is undefined here"
          | some (d, r) => Blocks.runInsts env ⟨Blocks.setV vals d r, w⟩ rest) :
    Blocks.runInsts env ⟨vals, w⟩ (inst :: rest)
      = match Sem.evalOp w.mem Γ o with
        | none => .stuck "instruction is undefined here"
        | some v => Blocks.runInsts env ⟨Blocks.setV vals ⟨n⟩ v, w⟩ rest := by
  rw [hstep, hE]
  cases Sem.evalOp w.mem Γ o <;> simp


-- ---------------------------------------------------------------------------
-- A straight-line run of pure operations
-- ---------------------------------------------------------------------------

/-- The evidence one statement contributes: at slot number `n`, `inst` computes
    what `o` computes, and it reaches `runInsts`' evaluating arm.

    Both halves hold by the per-shape lemmas above — `eval_X` for the first and
    `rfl` for the second — so a caller discharges this without ever unfolding
    the interpreters. -/
def OpStep (env : FnEnv) (n : Nat) (o : Op) (inst : Inst) : Prop :=
  ∀ (w : Sem.World) (Γ : Sem.Env) (vals : Blocks.Vals) (rest : List Inst),
    Γ.size = n → ValsAgree vals Γ →
    Blocks.evalInst w.mem vals inst = (Sem.evalOp w.mem Γ o).map (fun x => (⟨n⟩, x))
    ∧ Blocks.runInsts env ⟨vals, w⟩ (inst :: rest)
      = match Blocks.evalInst w.mem vals inst with
        | none => .stuck "instruction is undefined here"
        | some (d, r) => Blocks.runInsts env ⟨Blocks.setV vals d r, w⟩ rest

/-- A body's worth of that evidence, with the slot number counting up exactly as
    `emitStmts` numbers the values. -/
inductive OpChain (env : FnEnv) : Nat → List (Op × Inst) → Prop where
  | nil (n : Nat) : OpChain env n []
  | cons (n : Nat) (o : Op) (inst : Inst) (ps : List (Op × Inst)) :
      OpStep env n o inst → OpChain env (n + 1) ps → OpChain env n ((o, inst) :: ps)

/-- **Straight-line code compiles correctly.**

    If the term's statements run to `Γ'`, then the instructions they compiled to
    reach the very same continuation, in a state that still agrees with `Γ'` and
    has grown by exactly one slot per statement.

    Forward simulation on successful runs, which is the honest shape: the two
    interpreters word their failures differently, so equality of outcomes is the
    wrong statement, and nothing is lost — a term that gets stuck has no
    behavior to preserve. -/
theorem straight_sim (env : FnEnv) (cfg : Sem.Cfg) :
    ∀ (ps : List (Op × Inst)) (n : Nat) (w : Sem.World) (Γ Γ' : Sem.Env)
      (vals : Blocks.Vals) (rest : List Inst),
      OpChain env n ps → Γ.size = n → ValsAgree vals Γ →
      Sem.runStmts cfg Γ w (ps.map (fun p => Stmt.op p.1)) = .ok Γ' w →
      ∃ vals', Blocks.runInsts env ⟨vals, w⟩ (ps.map Prod.snd ++ rest)
                 = Blocks.runInsts env ⟨vals', w⟩ rest
               ∧ ValsAgree vals' Γ' ∧ Γ'.size = n + ps.length := by
  intro ps
  induction ps with
  | nil =>
      intro n w Γ Γ' vals rest _ hs hv hr
      simp [Sem.runStmts] at hr
      subst hr
      exact ⟨vals, by simp, hv, by simp [hs]⟩
  | cons p ps ih =>
      intro n w Γ Γ' vals rest hc hs hv hr
      obtain ⟨o, inst⟩ := p
      cases hc with
      | cons _ _ _ _ hstep hrest =>
        obtain ⟨hE, hR⟩ := hstep w Γ vals (ps.map Prod.snd ++ rest) hs hv
        cases hop : Sem.evalOp w.mem Γ o with
        | none =>
            rw [List.map_cons, Sem.runStmts, Sem.runStmt, hop] at hr
            simp at hr
        | some v =>
            rw [List.map_cons, Sem.runStmts, Sem.runStmt, hop] at hr
            have hs' : (Γ.push v).size = n + 1 := by simp [hs]
            have hv' : ValsAgree (Blocks.setV vals ⟨n⟩ v) (Γ.push v) :=
              agree_push vals Γ hv n hs v
            obtain ⟨vals', hrun, hva, hsz⟩ :=
              ih (n + 1) w (Γ.push v) Γ' (Blocks.setV vals ⟨n⟩ v) rest hrest hs' hv' hr
            refine ⟨vals', ?_, hva, by simp only [List.length_cons]; omega⟩
            simp only [List.map_cons, List.cons_append]
            rw [hR, hE, hop]
            simpa using hrun


-- ---------------------------------------------------------------------------
-- The general frame: any statement, any instruction
-- ---------------------------------------------------------------------------

/-- **One statement simulates its instruction.**

    Whatever the statement does to the term's state, the instruction it compiled
    to does to the block's — landing on the same world, in a value array that
    still agrees, having bound exactly the slots the statement binds.

    This is the interface the composition theorem is stated against, so adding a
    statement form to the proof means supplying one of these and nothing else.
    Pure operations get theirs from `step_op` and `agree_push`; stores and calls
    are where the world threading lives. -/
def StmtStep (env : FnEnv) (cfg : Sem.Cfg) (n : Nat) (st : Stmt) (inst : Inst) : Prop :=
  ∀ (w w₁ : Sem.World) (Γ Γ₁ : Sem.Env) (vals : Blocks.Vals) (rest : List Inst),
    Γ.size = n → ValsAgree vals Γ →
    Sem.runStmt cfg Γ w st = .ok Γ₁ w₁ →
    ∃ vals₁, Blocks.runInsts env ⟨vals, w⟩ (inst :: rest)
               = Blocks.runInsts env ⟨vals₁, w₁⟩ rest
             ∧ ValsAgree vals₁ Γ₁ ∧ Γ₁.size = n + st.binds

/-- A body's worth, with the slot number advancing by what each statement
    binds — which is how `emitStmts` numbers values, by `emitStmts_aligned`. -/
inductive StmtChain (env : FnEnv) (cfg : Sem.Cfg) : Nat → List (Stmt × Inst) → Prop where
  | nil (n : Nat) : StmtChain env cfg n []
  | cons (n : Nat) (st : Stmt) (inst : Inst) (ps : List (Stmt × Inst)) :
      StmtStep env cfg n st inst → StmtChain env cfg (n + st.binds) ps →
      StmtChain env cfg n ((st, inst) :: ps)

/-- **A straight-line body compiles correctly.**

    `straight_sim` for arbitrary statements: the compiled instructions reach the
    same continuation in an agreeing state, with the world the term ended in.
    Stores and calls are covered as soon as their `StmtStep` is supplied — the
    induction itself does not care what a statement does. -/
theorem stmts_sim (env : FnEnv) (cfg : Sem.Cfg) :
    ∀ (ps : List (Stmt × Inst)) (n : Nat) (w w' : Sem.World) (Γ Γ' : Sem.Env)
      (vals : Blocks.Vals) (rest : List Inst),
      StmtChain env cfg n ps → Γ.size = n → ValsAgree vals Γ →
      Sem.runStmts cfg Γ w (ps.map Prod.fst) = .ok Γ' w' →
      ∃ vals', Blocks.runInsts env ⟨vals, w⟩ (ps.map Prod.snd ++ rest)
                 = Blocks.runInsts env ⟨vals', w'⟩ rest
               ∧ ValsAgree vals' Γ'
               ∧ Γ'.size = n + (ps.map (fun p => p.1.binds)).sum := by
  intro ps
  induction ps with
  | nil =>
      intro n w w' Γ Γ' vals rest _ hs hv hr
      simp [Sem.runStmts] at hr
      obtain ⟨h1, h2⟩ := hr
      subst h1; subst h2
      exact ⟨vals, by simp, hv, by simp [hs]⟩
  | cons p ps ih =>
      intro n w w' Γ Γ' vals rest hc hs hv hr
      obtain ⟨st, inst⟩ := p
      cases hc with
      | cons _ _ _ _ hstep hrest =>
        rw [List.map_cons, Sem.runStmts] at hr
        cases hone : Sem.runStmt cfg Γ w st with
        | stuck m => rw [hone] at hr; simp at hr
        | ok Γ₁ w₁ =>
            rw [hone] at hr
            obtain ⟨vals₁, hrun₁, hva₁, hsz₁⟩ :=
              hstep w w₁ Γ Γ₁ vals (ps.map Prod.snd ++ rest) hs hv hone
            obtain ⟨vals', hrun, hva, hsz⟩ :=
              ih (n + st.binds) w₁ w' Γ₁ Γ' vals₁ rest hrest hsz₁ hva₁ hr
            refine ⟨vals', ?_, hva, by simp only [List.map_cons, List.sum_cons]; omega⟩
            simp only [List.map_cons, List.cons_append]
            rw [hrun₁]
            exact hrun

/-- Every pure operation satisfies the frame, given the two per-shape facts the
    `eval_X` lemmas and `rfl` already provide. -/
theorem op_stmtStep (env : FnEnv) (cfg : Sem.Cfg) (n : Nat) (o : Op) (inst : Inst)
    (h : OpStep env n o inst) : StmtStep env cfg n (.op o) inst := by
  intro w w₁ Γ Γ₁ vals rest hs hv hr
  obtain ⟨hE, hR⟩ := h w Γ vals rest hs hv
  cases hop : Sem.evalOp w.mem Γ o with
  | none => rw [Sem.runStmt, hop] at hr; simp at hr
  | some v =>
      rw [Sem.runStmt, hop] at hr
      simp only [Sem.Outcome.ok.injEq] at hr
      obtain ⟨h1, h2⟩ := hr
      subst h1; subst h2
      refine ⟨Blocks.setV vals ⟨n⟩ v, ?_, agree_push vals Γ hv n hs v, by simp [hs, Stmt.binds]⟩
      rw [hR, hE, hop]
      simp


-- ---------------------------------------------------------------------------
-- Stores
-- ---------------------------------------------------------------------------

/-- `runInsts`' `istore8` arm, as an equation. Holds by definition. -/
theorem runInsts_istore8 (env : FnEnv) (s : Blocks.BSt) (v a : Val) (rest : List Inst) :
    Blocks.runInsts env s (.istore8 v a :: rest)
      = match Blocks.getV s.vals v, Blocks.getV s.vals a with
        | some (.sc _ b), some (.sc _ addr) =>
            match s.world.mem.store addr 1 (b &&& 0xff) with
            | some m => Blocks.runInsts env
                { s with world := { Sem.obsStore s.world addr 1 (b &&& 0xff) with mem := m } } rest
            | none => .stuck "istore8 to unmapped address"
        | _, _ => .stuck "istore8 operand is not defined" := rfl

/-- **`istore8` satisfies the frame.** The first statement form that moves the
    world rather than the environment: it binds nothing, so the value arrays are
    unchanged and agreement is inherited, and both sides perform the *same*
    `Mem.store` at the same address with the same byte — which is what makes the
    observation trace match. -/
theorem istore8_stmtStep (env : FnEnv) (cfg : Sem.Cfg) (n : Nat) (v a : R)
    (hvn : v < n) (han : a < n) :
    StmtStep env cfg n (.istore8 v a) (.istore8 ⟨v⟩ ⟨a⟩) := by
  intro w w₁ Γ Γ₁ vals rest hs hva hr
  obtain ⟨xv, hxΓ, hxv⟩ := agree_at vals Γ hva v (hs ▸ hvn)
  obtain ⟨xa, haΓ, hav⟩ := agree_at vals Γ hva a (hs ▸ han)
  rw [Sem.runStmt_istore8, hxΓ, haΓ] at hr
  cases xv with
  | vec _ _ => simp at hr
  | sc t b =>
    cases xa with
    | vec _ _ => simp at hr
    | sc t2 addr =>
      dsimp only at hr
      cases hm : w.mem.store addr 1 (b &&& 0xff) with
      | none => rw [hm] at hr; simp at hr
      | some m =>
          rw [hm] at hr
          simp only [Sem.Outcome.ok.injEq] at hr
          obtain ⟨h1, h2⟩ := hr
          subst h1; subst h2
          refine ⟨vals, ?_, hva, by simp [Stmt.binds, hs]⟩
          rw [runInsts_istore8, hxv, hav]
          dsimp only
          rw [hm]


/-- `runInsts`' untyped-`store` arm, as an equation. -/
theorem runInsts_store (env : FnEnv) (s : Blocks.BSt) (v a : Val) (rest : List Inst) :
    Blocks.runInsts env s (.store v a :: rest)
      = match Blocks.doStore s v a none with
        | .error m => .stuck m
        | .ok s' => Blocks.runInsts env s' rest := rfl

/-- **`storeUnaligned` satisfies the frame.** The width comes from the stored
    value's own type on both sides — `tyBytes t` in the term, `tyBytes
    (none.getD t)` in `doStore` — which is the only place the two could have
    drifted, and they do not. -/
theorem storeUnaligned_stmtStep (env : FnEnv) (cfg : Sem.Cfg) (n : Nat) (v a : R)
    (hvn : v < n) (han : a < n) :
    StmtStep env cfg n (.storeUnaligned v a) (.store ⟨v⟩ ⟨a⟩) := by
  intro w w₁ Γ Γ₁ vals rest hs hva hr
  obtain ⟨xv, hxΓ, hxv⟩ := agree_at vals Γ hva v (hs ▸ hvn)
  obtain ⟨xa, haΓ, hav⟩ := agree_at vals Γ hva a (hs ▸ han)
  rw [Sem.runStmt_storeUnaligned, hxΓ, haΓ] at hr
  cases xv with
  | vec _ _ => simp at hr
  | sc t b =>
    cases xa with
    | vec _ _ => simp at hr
    | sc t2 addr =>
      dsimp only at hr
      cases hm : w.mem.store addr (Sem.tyBytes t) b with
      | none => rw [hm] at hr; simp at hr
      | some m =>
          rw [hm] at hr
          simp only [Sem.Outcome.ok.injEq] at hr
          obtain ⟨h1, h2⟩ := hr
          subst h1; subst h2
          refine ⟨vals, ?_, hva, by simp [Stmt.binds, hs]⟩
          rw [runInsts_store]
          have hd : Blocks.doStore ⟨vals, w⟩ ⟨v⟩ ⟨a⟩ none
              = .ok ⟨vals, { Sem.obsStore w addr (Sem.tyBytes t) b with mem := m }⟩ := by
            simp [Blocks.doStore, hxv, hav, hm]
          rw [hd]


/-- `runInsts`' typed-`store` arm, as an equation. -/
theorem runInsts_storeTyped (env : FnEnv) (s : Blocks.BSt) (ty : ClifTy) (v a : Val)
    (rest : List Inst) :
    Blocks.runInsts env s (.storeTyped ty v a :: rest)
      = match Blocks.doStore s v a (some ty) with
        | .error m => .stuck m
        | .ok s' => Blocks.runInsts env s' rest := rfl

/-- **The typed store satisfies the frame**, for scalar and vector values alike.
    The annotation fixes the width on the term side and `(some ty).getD` fixes
    it on the block side, and the lane fold is the same expression in both. -/
theorem store_stmtStep (env : FnEnv) (cfg : Sem.Cfg) (n : Nat) (ty : ClifTy) (v a : R)
    (hvn : v < n) (han : a < n) :
    StmtStep env cfg n (.store ty v a) (.storeTyped ty ⟨v⟩ ⟨a⟩) := by
  intro w w₁ Γ Γ₁ vals rest hs hva hr
  obtain ⟨xv, hxΓ, hxv⟩ := agree_at vals Γ hva v (hs ▸ hvn)
  obtain ⟨xa, haΓ, hav⟩ := agree_at vals Γ hva a (hs ▸ han)
  rw [Sem.runStmt_store, hxΓ, haΓ] at hr
  cases xa with
  | vec _ _ => simp at hr
  | sc t2 addr =>
    dsimp only at hr
    cases xv with
    | sc t b =>
      dsimp only at hr
      split at hr
      · next m hm =>
        simp only [Sem.Outcome.ok.injEq] at hr
        obtain ⟨h1, h2⟩ := hr
        subst h1; subst h2
        refine ⟨vals, ?_, hva, by simp [Stmt.binds, hs]⟩
        rw [runInsts_storeTyped]
        have hd : Blocks.doStore ⟨vals, w⟩ ⟨v⟩ ⟨a⟩ (some ty)
            = .ok ⟨vals, { Sem.obsStore w addr (Sem.tyBytes ty) b with mem := m }⟩ := by
          simp [Blocks.doStore, hxv, hav, hm]
        rw [hd]
      · simp at hr
    | vec t ls =>
      dsimp only at hr
      split at hr
      · next m hm =>
        simp only [Sem.Outcome.ok.injEq] at hr
        obtain ⟨h1, h2⟩ := hr
        subst h1; subst h2
        refine ⟨vals, ?_, hva, by simp [Stmt.binds, hs]⟩
        rw [runInsts_storeTyped]
        have hd : Blocks.doStore ⟨vals, w⟩ ⟨v⟩ ⟨a⟩ (some ty)
            = .ok ⟨vals, { Sem.obsStore w addr (Sem.tyBytes ty) 0 with mem := m }⟩ := by
          simp only [Blocks.doStore, hxv, hav]
          rw [hm]
          simp
        rw [hd]
      · simp at hr


-- ---------------------------------------------------------------------------
-- Calls
-- ---------------------------------------------------------------------------

/-- The argument lists line up: resolving the emitted values through the block's
    array gives what resolving the slots through `Γ` gives. `emitStmt` numbers a
    call's arguments with `CS.get`, which `Aligned` makes the identity, so this
    is where that bookkeeping turns into an actual equality of argument
    vectors — the thing a wrong call would get wrong. -/
theorem mapM_args (vals : Blocks.Vals) (Γ : Sem.Env) (hv : ValsAgree vals Γ)
    (n : Nat) (hs : Γ.size = n) :
    ∀ (args : List R), (∀ r ∈ args, r < n) →
      (args.map (fun r => (⟨r⟩ : Val))).mapM (Blocks.getV vals)
        = args.mapM (fun r => Γ[r]?) := by
  intro args
  induction args with
  | nil => intro _; simp
  | cons r rs ih =>
      intro hall
      have hr : r < n := hall r (by simp)
      have hrs : ∀ x ∈ rs, x < n := fun x hx => hall x (by simp [hx])
      simp only [List.map_cons, List.mapM_cons, hv r (hs ▸ hr), ih hrs]


/-- `runInsts`' `call` arm, as an equation. -/
theorem runInsts_call (env : FnEnv) (s : Blocks.BSt) (d : Option Val) (fn : FnRef)
    (args : List Val) (rest : List Inst) :
    Blocks.runInsts env s (.call d fn args :: rest)
      = match args.mapM (Blocks.getV s.vals) with
        | none => .stuck "call argument is not defined"
        | some vs =>
            match env.fns.find? (·.ref.id == fn.id) with
            | none => .stuck s!"fn{fn.id} is not declared"
            | some decl =>
                match decl.callee with
                | .local i => .stuck s!"fn{fn.id} calls u0:{i}"
                | .import name =>
                    match Sem.callFile name vs (Sem.obsCall s.world fn.id vs) with
                    | none => .stuck s!"{name} has no executable contract"
                    | some (res, w') =>
                        match d, res with
                        | some dv, some r =>
                            Blocks.runInsts env { vals := Blocks.setV s.vals dv r, world := w' } rest
                        | none, _ => Blocks.runInsts env { s with world := w' } rest
                        | some _, none => .stuck s!"{name} returned nothing to bind" := rfl

/-- **A result-binding call satisfies the frame.**

    The two sides resolve their arguments differently — slots through `Γ`,
    values through the array — and `mapM_args` is what makes those the same
    vector. Everything after that is the *same* `callFile` on the *same*
    observation, so the FFI contract is consulted once and both interpreters see
    its answer; the result binds the next slot on one side and the next value on
    the other, which `agree_push` keeps in step.

    `henv` is the one genuinely new obligation: the block interpreter takes its
    function environment as a parameter, the term takes it from the `Cfg`, and
    nothing but this hypothesis says they are the same table. -/
theorem call_stmtStep (env : FnEnv) (cfg : Sem.Cfg) (henv : cfg.env = env) (n : Nat)
    (fn : Nat) (args : List R) (hall : ∀ r ∈ args, r < n) :
    StmtStep env cfg n (.call fn args) (.call (some ⟨n⟩) ⟨fn⟩ (args.map (fun r => ⟨r⟩))) := by
  intro w w₁ Γ Γ₁ vals rest hs hva hr
  have hargs := mapM_args vals Γ hva n hs args hall
  rw [Sem.runStmt_call] at hr
  rw [runInsts_call]
  simp only [hargs]
  cases hm : args.mapM (fun r => Γ[r]?) with
  | none => rw [hm] at hr; simp at hr
  | some vs =>
    rw [hm] at hr
    simp only [henv] at hr
    cases hf : env.fns.find? (·.ref.id == fn) with
    | none => rw [hf] at hr; simp at hr
    | some d =>
      rw [hf] at hr
      dsimp only at hr
      cases hc : d.callee with
      | «local» i => rw [hc] at hr; simp at hr
      | «import» name =>
        rw [hc] at hr
        dsimp only at hr
        cases hcf : Sem.callFile name vs (Sem.obsCall w fn vs) with
        | none => rw [hcf] at hr; simp at hr
        | some p =>
          obtain ⟨res, w'⟩ := p
          rw [hcf] at hr
          dsimp only at hr
          cases hres : res with
          | none => rw [hres] at hr; simp at hr
          | some v =>
            rw [hres] at hr
            dsimp only at hr
            simp only [Sem.Outcome.ok.injEq] at hr
            obtain ⟨h1, h2⟩ := hr
            subst h1; subst h2
            refine ⟨Blocks.setV vals ⟨n⟩ v, ?_, agree_push vals Γ hva n hs v,
                    by simp [hs, Stmt.binds]⟩
            simp only [hf, hc, hcf, hres]


/-- **A void call satisfies the frame.** Same contract, same observation; the
    only difference is that nothing is bound, so both states keep the value
    stores they had and agreement is inherited unchanged. -/
theorem callVoid_stmtStep (env : FnEnv) (cfg : Sem.Cfg) (henv : cfg.env = env) (n : Nat)
    (fn : Nat) (args : List R) (hall : ∀ r ∈ args, r < n) :
    StmtStep env cfg n (.callVoid fn args) (.call none ⟨fn⟩ (args.map (fun r => ⟨r⟩))) := by
  intro w w₁ Γ Γ₁ vals rest hs hva hr
  have hargs := mapM_args vals Γ hva n hs args hall
  rw [Sem.runStmt_callVoid] at hr
  rw [runInsts_call]
  simp only [hargs]
  cases hm : args.mapM (fun r => Γ[r]?) with
  | none => rw [hm] at hr; simp at hr
  | some vs =>
    rw [hm] at hr
    simp only [henv] at hr
    cases hf : env.fns.find? (·.ref.id == fn) with
    | none => rw [hf] at hr; simp at hr
    | some d =>
      rw [hf] at hr
      dsimp only at hr
      cases hc : d.callee with
      | «local» i => rw [hc] at hr; simp at hr
      | «import» name =>
        rw [hc] at hr
        dsimp only at hr
        cases hcf : Sem.callFile name vs (Sem.obsCall w fn vs) with
        | none => rw [hcf] at hr; simp at hr
        | some p =>
          obtain ⟨res, w'⟩ := p
          rw [hcf] at hr
          dsimp only at hr
          simp only [Sem.Outcome.ok.injEq] at hr
          obtain ⟨h1, h2⟩ := hr
          subst h1; subst h2
          refine ⟨vals, ?_, hva, by simp [hs, Stmt.binds]⟩
          simp only [hf, hc, hcf]


-- ---------------------------------------------------------------------------
-- Block identity
-- ---------------------------------------------------------------------------

/-!
`runFrom` finds the block to run with `f.blocks.find? (·.ref.id == blk)`, so
every statement about a branch or a loop rests on that lookup returning the
block the emitter meant. Nothing above this point needs it — straight-line code
never leaves its block — and nothing below it can do without it.

Two properties carry the weight: emitted ids are distinct, so `find?` cannot
pick the wrong one, and the block being built is not already finished, so
closing it does not duplicate an id. `emitLoop` and `emitIte` reserve their ids
by bumping `nextBlk` before opening anything, which is exactly what makes both
survive.
-/

/-- Finished blocks have distinct ids, all of them already reserved. -/
def BlkWF (s : CS) : Prop :=
  (∀ b ∈ s.done, b.ref.id < s.nextBlk) ∧ (s.done.map (·.ref.id)).Nodup

/-- The block currently being built is not one of the finished ones. -/
def BlkOpen (s : CS) : Prop := ∀ b ∈ s.done, b.ref.id ≠ s.curRef

/-- Statements build the open block and never touch block identity. -/
theorem emitStmt_blk (s : CS) (st : Stmt) :
    (emitStmt s st).done = s.done ∧ (emitStmt s st).nextBlk = s.nextBlk
    ∧ (emitStmt s st).curRef = s.curRef := by
  cases st <;> simp [emitStmt, CS.fresh]

theorem emitStmts_blk : ∀ (ss : List Stmt) (s : CS),
    (emitStmts s ss).done = s.done ∧ (emitStmts s ss).nextBlk = s.nextBlk
    ∧ (emitStmts s ss).curRef = s.curRef := by
  intro ss
  induction ss with
  | nil => intro s; simp [emitStmts]
  | cons a as ih =>
      intro s
      have e : emitStmts s (a :: as) = emitStmts (emitStmt s a) as := rfl
      obtain ⟨h1, h2, h3⟩ := ih (emitStmt s a)
      obtain ⟨g1, g2, g3⟩ := emitStmt_blk s a
      rw [e]
      exact ⟨h1.trans g1, h2.trans g2, h3.trans g3⟩

/-- Closing the open block keeps ids distinct, because it was not finished. -/
theorem close_blkWF (s : CS) (term : Inst) (hw : BlkWF s) (ho : BlkOpen s)
    (hc : s.curRef < s.nextBlk) : BlkWF (s.close term) := by
  obtain ⟨hlt, hnd⟩ := hw
  constructor
  · intro b hb
    simp [CS.close, List.mem_append] at hb
    rcases hb with hb | hb
    · exact hlt b hb
    · simp only [hb]; simpa [CS.close] using hc
  · simp [CS.close, List.map_append, List.nodup_append]
    refine ⟨hnd, ?_⟩
    intro b hb
    exact fun h => ho b hb h


/-- **The lookup `runFrom` performs returns the block the emitter meant.**

    With distinct ids, `find?` on a block that is present finds *that* block —
    the fact every statement about a branch or a loop needs before it can say
    anything, since control transfer in the compiled form is by id and in the
    term by structure. -/
theorem find_blk : ∀ (blocks : List BlockData), (blocks.map (·.ref.id)).Nodup →
    ∀ (b : BlockData), b ∈ blocks → blocks.find? (·.ref.id == b.ref.id) = some b := by
  intro blocks
  induction blocks with
  | nil => intro _ b hb; simp at hb
  | cons a as ih =>
      intro hnd b hb
      simp only [List.map_cons, List.nodup_cons] at hnd
      obtain ⟨hna, hnas⟩ := hnd
      rcases Decidable.em (a.ref.id = b.ref.id) with h | h
      · have hba : b = a := by
          rcases List.mem_cons.mp hb with h' | h'
          · exact h'
          · have : b.ref.id ∈ as.map (·.ref.id) := List.mem_map.mpr ⟨b, h', rfl⟩
            rw [← h] at this
            exact absurd this hna
        subst hba
        simp [List.find?, h]
      · have hb' : b ∈ as := by
          rcases List.mem_cons.mp hb with h' | h'
          · exact absurd (h' ▸ rfl) h
          · exact h'
        have hne : (a.ref.id == b.ref.id) = false := by simp [h]
        simp only [List.find?, hne]
        exact ih hnas b hb'


/-- **Block dispatch, resolved.** Entering a block whose id is present in a
    well-formed block list runs *that block's* instructions, with its parameters
    bound to the incoming arguments.

    This is the step that turns control transfer by id — which is all the
    compiled form has — back into something structural, and so it is the bridge
    every branch and loop statement crosses. -/
theorem runFrom_block (env : FnEnv) (f : FuncData) (steps : Nat) (s : Blocks.BSt)
    (b : BlockData) (hnd : (f.blocks.map (·.ref.id)).Nodup) (hb : b ∈ f.blocks)
    (args : List Sem.V) (hlen : b.params.length = args.length) :
    Blocks.runFrom env f (steps + 1) s b.ref.id args
      = (match Blocks.runInsts env
            ⟨(b.params.zip args).foldl (fun vs pa => Blocks.setV vs pa.1.1 pa.2) s.vals,
             s.world⟩ b.insts with
        | .stuck m => .stuck m
        | .ok (s', next) w =>
            match next with
            | .done => .ok w w
            | .goto t vs => Blocks.runFrom env f steps { s' with world := w } t vs) := by
  rw [Blocks.runFrom, find_blk f.blocks hnd b hb]
  simp only [hlen, bne_self_eq_false, Bool.false_eq_true, if_false]


-- ---------------------------------------------------------------------------
-- Block parameters are slots
-- ---------------------------------------------------------------------------

/-- `Aligned` without the `slots` field, which `open'` does not set — its
    callers do, right after. -/
def EnvAligned (s : CS) (n : Nat) : Prop :=
  s.nextVal = n ∧ ∀ i, i < n → s.env.lookup i = some ⟨i⟩

/-- **Opening a block extends the numbering by its parameters.**

    Each parameter is a fresh value bound to the next slot, and because `open'`
    is always called with `firstSlot` equal to the current value counter, the
    slot and the value get the *same* number — so `Aligned` survives crossing a
    block boundary.

    This is the carry/exit-slot ↔ block-parameter correspondence, in the only
    form it actually needs to take: not a separate translation to maintain, but
    the observation that the two numberings never diverge in the first place. -/
theorem open'_go_aligned : ∀ (tys : List ClifTy) (st : CS) (n0 i : Nat),
    EnvAligned st (n0 + i) →
    EnvAligned (CS.open'.go n0 st i tys) (n0 + i + tys.length) := by
  intro tys
  induction tys with
  | nil => intro st n0 i h; simpa [CS.open'.go] using h
  | cons t ts ih =>
      intro st n0 i h
      obtain ⟨hn, he⟩ := h
      have hstep : EnvAligned
          { st.fresh.2 with
            curPars := st.fresh.2.curPars ++ [(st.fresh.1, t)],
            env := (n0 + i, st.fresh.1) :: st.fresh.2.env } (n0 + (i + 1)) := by
        constructor
        · simp [CS.fresh, hn]; omega
        · intro j hj
          rcases Nat.lt_or_ge j (n0 + i) with hlt | hge
          · rw [lookup_cons_ne j (n0 + i) _ _ (Nat.ne_of_lt hlt)]
            exact he j hlt
          · have hje : j = n0 + i := by omega
            subst hje
            simp [List.lookup, CS.fresh, hn]
      have := ih _ n0 (i + 1) hstep
      simpa [CS.open'.go, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using this

/-- The form the emitters use: `open'` at the current value counter, followed by
    advancing `slots` past the parameters, lands aligned again. -/
theorem open'_aligned (s : CS) (n : Nat) (h : Aligned s n) (ref : Nat)
    (tys : List ClifTy) :
    Aligned { CS.open' s ref tys n with slots := n + tys.length } (n + tys.length) := by
  obtain ⟨h1, _, he⟩ := h
  have hbase : EnvAligned { s with curRef := ref, curPars := [] } (n + 0) := by
    constructor
    · simpa using h1
    · intro i hi; exact he i (by omega)
  have := open'_go_aligned tys { s with curRef := ref, curPars := [] } n 0 hbase
  obtain ⟨g1, g2⟩ := this
  exact ⟨by simpa [CS.open'] using g1, rfl, by
    intro i hi; simpa [CS.open'] using g2 i (by omega)⟩


-- ---------------------------------------------------------------------------
-- How emission grows the block list
-- ---------------------------------------------------------------------------

/-- Opening a block touches the numbering and the open block, never the
    finished ones. -/
theorem open'_go_blk : ∀ (tys : List ClifTy) (st : CS) (n0 i : Nat),
    (CS.open'.go n0 st i tys).done = st.done
    ∧ (CS.open'.go n0 st i tys).nextBlk = st.nextBlk
    ∧ (CS.open'.go n0 st i tys).curRef = st.curRef := by
  intro tys
  induction tys with
  | nil => intro st n0 i; simp [CS.open'.go]
  | cons t ts ih =>
      intro st n0 i
      obtain ⟨h1, h2, h3⟩ := ih _ n0 (i + 1)
      refine ⟨?_, ?_, ?_⟩
      · simpa [CS.open'.go, CS.fresh] using h1
      · simpa [CS.open'.go, CS.fresh] using h2
      · simpa [CS.open'.go, CS.fresh] using h3

theorem open'_blk (s : CS) (ref : Nat) (tys : List ClifTy) (f : Nat) :
    (CS.open' s ref tys f).done = s.done
    ∧ (CS.open' s ref tys f).nextBlk = s.nextBlk
    ∧ (CS.open' s ref tys f).curRef = ref := by
  obtain ⟨h1, h2, h3⟩ := open'_go_blk tys { s with curRef := ref, curPars := [] } f 0
  exact ⟨by simpa [CS.open'] using h1, by simpa [CS.open'] using h2,
         by simpa [CS.open'] using h3⟩

/-- Closing appends exactly one block and reserves nothing new. -/
theorem close_blk (s : CS) (term : Inst) :
    (s.close term).done = s.done ++ [{ ref := ⟨s.curRef⟩, params := s.curPars,
                                       insts := (term :: s.cur).reverse }]
    ∧ (s.close term).nextBlk = s.nextBlk := by
  simp [CS.close]

/-- Finished blocks are only ever appended to — nothing already emitted is
    rewritten or dropped. Together with `find_blk` this is what lets a block
    proved correct in the middle of emission stay correct at the end. -/
theorem close_done_prefix (s : CS) (term : Inst) : s.done <+: (s.close term).done := by
  simp [CS.close]

theorem emitStmts_done_prefix (ss : List Stmt) (s : CS) :
    s.done <+: (emitStmts s ss).done := by
  rw [(emitStmts_blk ss s).1]
  exact List.prefix_refl _


/-- Emission only ever moves forward: it reserves more block ids and appends to
    the finished list. Every primitive the emitters are built from has this
    property, and it composes, so a chain of them does too.

    Stated as a relation rather than proved inline at each emitter because
    `emitLoop` and `emitIte` are ten-step chains and the alternative is ten
    rewrites against a term that grows at every step. -/
def Grows (s t : CS) : Prop := s.nextBlk ≤ t.nextBlk ∧ s.done <+: t.done

theorem Grows.refl (s : CS) : Grows s s := ⟨Nat.le_refl _, List.prefix_refl _⟩

theorem Grows.trans {a b c : CS} (h1 : Grows a b) (h2 : Grows b c) : Grows a c :=
  ⟨Nat.le_trans h1.1 h2.1, h1.2.trans h2.2⟩

theorem Grows.close (s : CS) (term : Inst) : Grows s (s.close term) :=
  ⟨by simp [CS.close], close_done_prefix s term⟩

theorem Grows.open' (s : CS) (ref : Nat) (tys : List ClifTy) (f : Nat) :
    Grows s (CS.open' s ref tys f) := by
  obtain ⟨h1, h2, _⟩ := open'_blk s ref tys f
  exact ⟨by rw [h2]; exact Nat.le_refl _, by rw [h1]; exact List.prefix_refl _⟩

theorem Grows.emitStmts (s : CS) (ss : List Stmt) : Grows s (emitStmts s ss) := by
  obtain ⟨h1, h2, _⟩ := emitStmts_blk ss s
  exact ⟨by rw [h2]; exact Nat.le_refl _, by rw [h1]; exact List.prefix_refl _⟩

/-- Reserving ids and building up the open block's instruction list are both
    growth, so the record updates the emitters perform between their calls need
    no separate argument. -/
theorem Grows.bump (s : CS) (k : Nat) (cur : List Inst) (nv sl : Nat)
    (env : List (Nat × Val)) (cr : Nat) (pars : List (Val × ClifTy)) :
    Grows s { s with nextBlk := s.nextBlk + k, cur := cur, nextVal := nv,
                     slots := sl, env := env, curRef := cr, curPars := pars } :=
  ⟨by simp, by simp⟩


-- ---------------------------------------------------------------------------
-- From the emitter's list to the function's
-- ---------------------------------------------------------------------------

/-!
`compileFn` does not ship `done` as emitted: it ships
`s.done.mergeSort (fun a b => a.ref.id ≤ b.ref.id)`. Every fact proved about the
emitter is therefore about a *permutation* of the list `runFrom` actually walks.

Sorting is exactly the right thing for the artifact — blocks read in id order —
and exactly the wrong thing to leave implicit in a proof, because `find_blk`
wants `Nodup` and membership in the list being searched. Both survive a
permutation, which is all that is needed.
-/

/-- Ids stay distinct after sorting. -/
theorem sorted_nodup (blocks : List BlockData) (le : BlockData → BlockData → Bool)
    (h : (blocks.map (·.ref.id)).Nodup) :
    ((blocks.mergeSort le).map (·.ref.id)).Nodup :=
  ((List.mergeSort_perm blocks le).map (·.ref.id)).nodup_iff.mpr h

/-- A block the emitter finished is still there after sorting. -/
theorem sorted_mem (blocks : List BlockData) (le : BlockData → BlockData → Bool)
    (b : BlockData) (h : b ∈ blocks) : b ∈ blocks.mergeSort le :=
  (List.mergeSort_perm blocks le).mem_iff.mpr h

/-- **The lookup `runFrom` performs, against the list `compileFn` ships.**

    `find_blk` for the sorted list: what the emitter finished is what the
    interpreter finds. This is the seam between everything proved about `CS.done`
    and everything `Blocks.runFrom` does, and without it no statement about the
    emitter says anything about a run. -/
theorem find_blk_sorted (blocks : List BlockData) (le : BlockData → BlockData → Bool)
    (hnd : (blocks.map (·.ref.id)).Nodup) (b : BlockData) (hb : b ∈ blocks) :
    (blocks.mergeSort le).find? (·.ref.id == b.ref.id) = some b :=
  find_blk (blocks.mergeSort le) (sorted_nodup blocks le hnd) b (sorted_mem blocks le b hb)


-- ---------------------------------------------------------------------------
-- The two budgets
-- ---------------------------------------------------------------------------

/-!
`CompileSound` hands the same number to both interpreters, but they do not spend
it on the same thing: the term's fuel is decremented per *piece* and per loop
*iteration*, the compiled form's per *block entry*. One iteration of a loop costs
the term one unit and the blocks at least two — head and body — so the two
counters cannot be expected to run out together.

That is a real gap in the statement, not a proof difficulty, and the honest fix
is to stop insisting the numbers match. What makes that safe is monotonicity: a
run that succeeds keeps succeeding, with the *same* answer, given more budget.
With it, "some budget suffices" is as strong as any particular budget, and the
theorem can quantify the block side existentially without weakening anything a
caller cares about.
-/

/-- **More budget never changes a successful run.**

    Block dispatch is deterministic and `steps` only bounds how many entries a
    run may make, so raising the bound cannot change an answer that was already
    reached — it can only rescue one that ran out.

    This is what lets the compilation theorem say *there exists* a step budget
    under which the compiled form agrees, instead of pinning a number that would
    have to be recomputed every time the emitter changes shape. -/
theorem runFrom_mono (env : FnEnv) (f : FuncData) :
    ∀ (steps : Nat) (s : Blocks.BSt) (blk : Nat) (args : List Sem.V) (r w : Sem.World),
      Blocks.runFrom env f steps s blk args = .ok r w →
      ∀ steps', steps ≤ steps' → Blocks.runFrom env f steps' s blk args = .ok r w := by
  intro steps
  induction steps with
  | zero => intro s blk args r w h; simp [Blocks.runFrom] at h
  | succ steps ih =>
      intro s blk args r w h steps' hle
      cases steps' with
      | zero => omega
      | succ steps'' =>
        have hle' : steps ≤ steps'' := by omega
        rw [Blocks.runFrom] at h ⊢
        cases hfind : f.blocks.find? (·.ref.id == blk) with
        | none => simp [hfind] at h
        | some b =>
          simp only [hfind] at h ⊢
          split at h
          · simp at h
          · rename_i hlen
            simp only [hlen, if_false, Bool.false_eq_true]
            cases hins : Blocks.runInsts env
                ⟨(b.params.zip args).foldl (fun vs pa => Blocks.setV vs pa.1.1 pa.2) s.vals,
                 s.world⟩ b.insts with
            | stuck m => simp [hins] at h
            | ok p w' =>
              obtain ⟨s', next⟩ := p
              simp only [hins] at h ⊢
              cases next with
              | done => exact h
              | goto t vs => exact ih _ t vs r w h steps'' hle'


/-- The same, at the level `compile_sound` is stated: a successful compiled run
    is unchanged by a larger budget. -/
theorem run_mono (env : FnEnv) (f : FuncData) (args : List Sem.V) (w : Sem.World)
    (steps : Nat) (obs : List Sem.Obs) (w' : Sem.World)
    (h : Blocks.run env f args w steps = .ok obs w') :
    ∀ steps', steps ≤ steps' → Blocks.run env f args w steps' = .ok obs w' := by
  intro steps' hle
  rw [Blocks.run] at h ⊢
  cases hr : Blocks.runFrom env f steps ⟨#[], w⟩ 0 args with
  | stuck m => simp [hr] at h
  | ok a ww =>
      simp only [hr] at h
      rw [runFrom_mono env f steps ⟨#[], w⟩ 0 args a ww hr steps' hle]
      exact h

/-- **The statement, with the budgets separated.**

    Identical to `CompileSound` except that the compiled form is allowed its own
    step budget. That is the honest form: the term counts pieces and iterations,
    the blocks count entries, and a loop iteration is one of the former and at
    least two of the latter, so requiring one number to serve both is a claim
    about the emitter's shape rather than about compilation.

    `run_mono` is what keeps this from being a weakening — the compiled side is
    upward-closed in its budget, so exhibiting one sufficient budget settles
    every larger one, and a caller who wants a concrete number can take any
    bound that works. -/
def CompileSoundE (idx : Nat) (env : FnEnv) (params : List ClifTy) (c : Code)
    (args : List V) (w : World) (fuel : Nat) : Prop :=
  ∃ steps, Sem.run { env, steps := fuel } args w c
    = Blocks.run env (compileFn idx env params c) args w steps

/-- Everything already proved at the matched-budget statement carries over, so
    separating the budgets costs none of the existing results. -/
theorem compileSound_toE (idx : Nat) (env : FnEnv) (params : List ClifTy) (c : Code)
    (args : List V) (w : World) (fuel : Nat) (h : CompileSound idx env params c args w fuel) :
    CompileSoundE idx env params c args w fuel := ⟨fuel, h⟩

/-- The base cases at the budget-separated statement. -/
theorem empty_soundE (idx : Nat) (env : FnEnv) (a : V) (w : World) (fuel : Nat) :
    CompileSoundE idx env ptrParams [] [a] w (fuel + 1) :=
  compileSound_toE _ _ _ _ _ _ _ (empty_sound idx env a w fuel)

theorem single_iconst_soundE (idx : Nat) (env : FnEnv) (t : ClifTy) (k : Int)
    (a : V) (w : World) (fuel : Nat) :
    CompileSoundE idx env ptrParams [.straight [.op (.iconst t k)]] [a] w (fuel + 2) :=
  compileSound_toE _ _ _ _ _ _ _ (single_iconst_sound idx env t k a w fuel)

end AlgorithmLib.HProg