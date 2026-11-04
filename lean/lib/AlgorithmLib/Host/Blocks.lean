module
public import AlgorithmLib.Host.Sem
meta import AlgorithmLib.Host.Sem
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Host.Blocks` — executing the compiled form

`Host.Sem` says what a *term* does. This says what the `IR.FuncData` that
`compileBody` produces does: blocks, block parameters, `jump`, `brif`, and a
value numbering rather than a slot numbering.

Two interpreters, one instruction semantics. Every arithmetic arm here goes
through `Sem.evalOp` on a two-element environment, so the operation
semantics is *literally the same definition* on both sides. That is what stops
`compile_sound` from being provable by accident: what it has to show is that
compilation preserves the *order and arguments* of the operations, which is the
only thing the two forms can disagree about — and the only thing that has
actually gone wrong so far (exit and join parameters twice).

Scope: this interprets compiler output, not arbitrary CLIF. Instructions
`compileBody` never emits are stuck rather than silently ignored.
-/

namespace AlgorithmLib.HProg.Blocks

open AlgorithmLib.IR
open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sem

/-- Values numbered as `compileBody` numbers them: densely, from zero. -/
abbrev Vals := Array V

def getV (vs : Vals) (v : Val) : Option V := vs[v.id]?

def setV (vs : Vals) (v : Val) (x : V) : Vals :=
  let vs := if v.id < vs.size then vs
            else vs ++ Array.replicate (v.id + 1 - vs.size) default
  vs.set! v.id x

/-- Evaluate through the term interpreter's own `evalOp`, on an environment
    holding just this instruction's operands. The `Op` names slots `0, 1`
    because that is where they were put. -/
def viaOp (m : Mem) (args : List V) (o : Op) : Option V :=
  evalOp m args.toArray o

/-- The position of one of the program's own functions, as a callee names it. -/
def localIdx? : Callee → Option Nat
  | .local i => some i
  | _ => none

/-- One result-producing instruction: its destination and what it computes. -/
def evalInst (m : Mem) (vs : Vals) : Inst → Option (Val × V)
  | .iconst d t k => do pure (d, ← viaOp m [] (.iconst t k))
  | .memory d => do pure (d, ← viaOp m [] .memory)
  | .funcAddr d c => do let i ← localIdx? c; pure (d, ← viaOp m [] (.funcAddr i))
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
  | .ibin d k a b => bin d a b (.ibin k)
  | .ishift d k a b => bin d a b (.ishift k)
  | .fbin d k a b => bin d a b (.fbin k)
  | .iun d k a => un d a (.iun k)
  | .fun1 d k a => un d a (.fun1 k)
  | .fconv d k t a => un d a (.fconv k t)
  | .iext d k t a => un d a (.iext k t)
  | .fma d a b c => do
      let x ← getV vs a; let y ← getV vs b; let z ← getV vs c
      pure (d, ← viaOp m [x, y, z] (.fma 0 1 2))
  | .select d c a b => do
      let cv ← getV vs c; let x ← getV vs a; let y ← getV vs b
      pure (d, ← viaOp m [cv, x, y] (.select 0 1 2))
  | .bitselect d c a b => do
      let cv ← getV vs c; let x ← getV vs a; let y ← getV vs b
      pure (d, ← viaOp m [cv, x, y] (.bitselect 0 1 2))
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
  /-- The function returned, with this value if it answers one. -/
  | done (ret : Option V)
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
              world := { obsStore s.world addr (ls.size * lw) 0 with mem := m } }
          | none => .error "vector store to unmapped address"
  | _, _ => .error "store operand is not defined"

/-- Run the straight-line part of a block, stopping at its terminator.
    Structural on the instruction list, so it has equations. -/
def runInsts (lc : Locals) (s : BSt) : List Inst → Outcome (BSt × Next)
  | [] => .stuck "block has no terminator"
  | i :: rest =>
      match i with
      | .ret none => .ok (s, .done none) s.world
      | .ret (some v) =>
          match getV s.vals v with
          | none => .stuck "returned value is not defined"
          | some r => .ok (s, .done (some r)) s.world
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
          | .ok s' => runInsts lc s' rest
      | .storeTyped t v a =>
          match doStore s v a (some t) with
          | .error m => .stuck m
          | .ok s' => runInsts lc s' rest
      | .istore8 v a =>
          match getV s.vals v, getV s.vals a with
          | some (.sc _ b), some (.sc _ addr) =>
              match s.world.mem.store addr 1 (b &&& 0xff) with
              | some m => runInsts lc
                  { s with world := { obsStore s.world addr 1 (b &&& 0xff) with mem := m } } rest
              | none => .stuck "istore8 to unmapped address"
          | _, _ => .stuck "istore8 operand is not defined"
      | .call d c args =>
          match args.mapM (getV s.vals) with
          | none => .stuck "call argument is not defined"
          | some vs =>
              match callOf lc c vs (obsCall s.world c vs) with
              | none => failOf lc c vs (obsCall s.world c vs)
              | some (res, w') =>
                  match d, res with
                  | some dv, some r =>
                      runInsts lc { vals := setV s.vals dv r, world := w' } rest
                  | none, _ => runInsts lc { s with world := w' } rest
                  | some _, none => .stuck "the call returned nothing to bind"
      | other =>
          match evalInst s.world.mem s.vals other with
          | none => .stuck "instruction is undefined here"
          | some (d, r) => runInsts lc { s with vals := setV s.vals d r } rest

/-- Run a function from its entry block until it returns. `steps` bounds the
    number of block entries, so a loop that does not terminate is stuck — and
    it is the structural argument, so this has equations. -/
def runFrom (lc : Locals) (f : FuncData) : Nat → BSt → Nat → List V → Outcome (Option V)
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
          match runInsts lc { s with vals } b.insts with
          | .stuck m => .stuck m
          | .misuse m => .misuse m
          | .fault m => .fault m
          | .ok (s', next) w =>
              match next with
              | .done r => .ok r w
              | .goto t vs => runFrom lc f steps { s' with world := w } t vs

/-- Execute a compiled function, and hand back its observation trace in
    program order — the same shape `Sem.run` produces for the term. -/
def run (lc : Locals) (f : FuncData) (args : List V) (w : World)
    (steps : Nat := 100000000) : Outcome (List Obs) :=
  match runFrom lc f steps { vals := #[], world := w } 0 args with
  | .stuck m => .stuck m
  | .misuse m => .misuse m
  | .fault m => .fault m
  | .ok _ w' => .ok w'.obs.reverse w'

/-- Call a compiled function: its answer, if it has one, and the world it
    leaves, or `none` if it gets stuck within `steps` block entries. -/
def callFn (lc : Locals) (f : FuncData) (steps : Nat) (args : List V) (w : World) :
    Option (Option V × World) :=
  match runFrom lc f steps { vals := #[], world := w } 0 args with
  | .stuck _ | .misuse _ | .fault _ => none
  | .ok r w' => some (r, w')

end AlgorithmLib.HProg.Blocks

-- ---------------------------------------------------------------------------
-- `compile_sound`
-- ---------------------------------------------------------------------------

namespace AlgorithmLib.HProg

open AlgorithmLib.IR
open AlgorithmLib.HProg.Sem

/-- **The statement.** A term and the function `compileBody` builds from it make
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
    = Blocks.run noLocals (compileBody idx c env params) args w fuel


/-- Five `i64` parameters: the shape the examples below are stated at. -/
def five : List ClifTy := [.i64, .i64, .i64, .i64, .i64]

/-- The base case, proved. An empty body compiles to one block that returns,
    and neither side observes anything.

    Small, but not vacuous, and it is the case the induction over `Code` rests
    on: it fixes the entry correspondence — that the term's parameters and the
    compiled entry block's parameters are the same values, in the same order,
    at the same indices. Everything else is built on top of that. -/
theorem empty_sound (idx : Nat) (env : FnEnv) (base data dlen out olen : V)
    (w : World) (fuel : Nat) :
    CompileSound idx env five [] [base, data, dlen, out, olen] w (fuel + 1) := by
  have hEmit : ∀ s : CS, emitCode HProg.fuel s [] = s := by
    intro s; simp [emitCode, HProg.fuel]
  simp [CompileSound, Sem.run, Blocks.run, Sem.runCode, compileBody, five,
        CS.open', CS.open'.go, CS.close, CS.fresh, Blocks.runFrom, Blocks.runInsts,
        Blocks.setV, hEmit]

/-- One step past the base case: a body that computes a single constant.

    The first case where the two forms actually do different work — the term
    pushes onto a slot environment, the compiled block writes a value the
    emitter numbered — and it goes through because for straight-line code the
    numbering is the identity: slot `i` is `Val i`. That correspondence is what
    `compile_sound`'s straight-line case (`stmts_msim`) generalises. -/
theorem single_iconst_sound (idx : Nat) (env : FnEnv) (t : ClifTy) (k : Int)
    (base data dlen out olen : V) (w : World) (fuel : Nat) :
    CompileSound idx env five [.straight [.op (.iconst t k)]]
      [base, data, dlen, out, olen] w (fuel + 2) := by
  have hEmit : ∀ s : CS,
      emitCode HProg.fuel s [Piece.straight [Stmt.op (Op.iconst t k)]]
        = emitStmt s (Stmt.op (Op.iconst t k)) := by
    intro s; simp [emitCode, emitPiece, emitStmts, HProg.fuel]
  simp [CompileSound, Sem.run, Blocks.run, Sem.runCode, Sem.runPiece, Sem.runStmts,
        Sem.runStmt, compileBody, five, CS.open', CS.open'.go, CS.close, CS.fresh, CS.get,
        Blocks.runFrom, Blocks.runInsts, Blocks.setV, Blocks.evalInst, Blocks.viaOp,
        emitStmt, hEmit, Sem.evalOp]

/-! ### The branch

    These are about control flow at a concrete body, and they are about the
    place `CompileSound`'s own statement names as where the two forms have
    disagreed twice: the join parameters. -/

/-- The destination reads back as what was just written. -/
theorem getV_setV_self (vs : Blocks.Vals) (d : Val) (x : Sem.V) :
    Blocks.getV (Blocks.setV vs d x) d = some x := by
  simp only [Blocks.getV, Blocks.setV, Array.set!, Array.getElem?_setIfInBounds]
  rcases Nat.lt_or_ge d.id vs.size with hd | hd
  · simp only [hd, if_pos]
  · simp only [Nat.not_lt.mpr hd, if_false, Array.size_append, Array.size_replicate]
    simp only [if_pos (show d.id < vs.size + (d.id + 1 - vs.size) by omega), if_true]

/-- `ClifTy` derives `BEq` and not `DecidableEq`, so reflexivity is a lemma. -/
theorem ClifTy_beq_self (t : ClifTy) : (t == t) = true := by cases t <;> rfl

/-- A comparison's result reads as the condition it decided. -/
theorem isTrue_boolV (b : Bool) : Sem.isTrue (Sem.boolV b) = b := by cases b <;> rfl

/-- **The branch prologue goes where the term's comparison says.**

    `emitIte` opens with an `icmp` into a fresh flag and a `brif` on it; this is
    that pair, and it is the step every `ite` case needs before either arm is
    reached. -/
theorem brif_step (lc : Sem.Locals) (s : Blocks.BSt) (flag a b : Val) (cc : ICmpCond)
    (thnId elsId : Nat) (rest : List Inst) (t : ClifTy) (x y : UInt64)
    (ha : Blocks.getV s.vals a = some (.sc t x))
    (hb : Blocks.getV s.vals b = some (.sc t y)) (hint : t.isInt = true) :
    Blocks.runInsts lc s (.icmp flag cc a b :: .brif flag ⟨thnId⟩ [] ⟨elsId⟩ [] :: rest)
      = .ok ({ s with vals := Blocks.setV s.vals flag (Sem.boolV (Sem.cmpInt cc t x y)) },
             .goto (if Sem.cmpInt cc t x y then thnId else elsId) []) s.world := by
  simp [Blocks.runInsts, Blocks.evalInst, Blocks.evalInst.bin, Blocks.viaOp, Sem.evalOp,
        Sem.get, ha, hb, Sem.zipIntCmp, hint, getV_setV_self, isTrue_boolV, ClifTy_beq_self]
  cases hcmp : Sem.cmpInt cc t x y <;> simp

/-- **A branch compiles correctly**, on the smallest body that has a join: two
    empty arms and no exported values. Three blocks are emitted where the term
    has one piece, and the run still ends in the same place having observed
    nothing.

    The test is the statement before the branch --- slot 5, the first past the
    entry parameters --- which is where `IteMeta.flag` points. -/
theorem ite_empty_sound (idx : Nat) (env : FnEnv) (x d dl o ol : UInt64)
    (w : World) (fuel : Nat) :
    CompileSound idx env five
      [.straight [.op (.icmp .eq 0 0)], .ite ⟨5, []⟩ [] [] [] []]
      [.sc .i64 x, .sc .i64 d, .sc .i64 dl, .sc .i64 o, .sc .i64 ol]
      w (fuel + 6) := by
  simp [CompileSound, Sem.run, Blocks.run, Sem.runCode, Sem.runPiece, Sem.runStmts,
        Sem.runStmt, compileBody,
        five, CS.open', CS.open'.go, CS.close, CS.fresh, CS.get,
        emitCode, emitPiece, emitIte, emitStmt, emitStmts, termsGo, HProg.fuel,
        Blocks.runFrom, Blocks.runInsts, Blocks.setV, Blocks.getV,
        Sem.get, Sem.bindAt, Sem.slotsOf, Sem.slotsGo, Sem.cmpInt, Sem.boolV, Sem.isTrue,
        Trie.set, Trie.setGo, Trie.get, List.mergeSort, List.find?,
        Blocks.evalInst, Blocks.evalInst.bin, Blocks.viaOp, Sem.evalOp, Sem.zipIntCmp,
        show ((ClifTy.i64 == ClifTy.i64) = true) from rfl,
        show (ClifTy.i64.isInt = true) from rfl]

/-- **…and it gets the join parameters right.**

    Both arms bind a constant and export it. The numbering is the content: the
    arms' slots start past the entry parameters, and the else arm's is one
    higher than the then arm's because `emitIte` numbers it as if the then arm
    had run — only one of them executes, so without the padding the else arm's
    own slots would land on the then arm's. That is the correspondence this
    development has got wrong twice. -/
theorem ite_exports_sound (idx : Nat) (env : FnEnv) (k1 k2 : Int) (x d dl o ol : UInt64)
    (w : World) (fuel : Nat) :
    CompileSound idx env five
      [.straight [.op (.icmp .eq 0 0)],
       .ite ⟨5, [.i64]⟩
        [.straight [.op (.iconst .i64 k1)]] [.straight [.op (.iconst .i64 k2)]] [6] [7]]
      [.sc .i64 x, .sc .i64 d, .sc .i64 dl, .sc .i64 o, .sc .i64 ol]
      w (fuel + 8) := by
  simp [CompileSound, Sem.run, Blocks.run, Sem.runCode, Sem.runPiece, Sem.runStmts,
        Sem.runStmt, compileBody, five, CS.open', CS.open'.go, CS.close, CS.fresh, CS.get,
        emitCode, emitPiece, emitIte, emitStmt, emitStmts, termsGo, HProg.fuel,
        Blocks.runFrom, Blocks.runInsts, Blocks.setV, Blocks.getV,
        Sem.get, Sem.bindAt, Sem.slotsOf, Sem.slotsGo, Sem.cmpInt, Sem.boolV, Sem.isTrue,
        Trie.set, Trie.setGo, Trie.get, List.mergeSort, List.find?,
        Blocks.evalInst, Blocks.evalInst.bin, Blocks.viaOp, Sem.evalOp, Sem.zipIntCmp,
        show ((ClifTy.i64 == ClifTy.i64) = true) from rfl,
        show (ClifTy.i64.isInt = true) from rfl]

/-- For straight-line code the emitter's slot-to-value map is the **identity**:
    `emitStmt` advances `nextVal` and `slots` together, so slot `i` is always
    `Val i`.

    This is the invariant `compile_sound`'s straight-line case (`stmts_msim`) runs
    on. Stating it is what makes that induction possible at all — unfolding the definitions and letting
    `simp` reduce works for a single statement and times out at two, so the
    proof has to reason about the emitter rather than evaluate it. -/
def Aligned (s : CS) (n : Nat) : Prop :=
  s.nextVal = n ∧ s.slots = n ∧ ∀ i, i < n → s.env.get i = some ⟨i⟩

/-- **Emitting one statement preserves the alignment**, advancing it by exactly
    the number of slots that statement defines.

    General in the statement and in the state — the first result here that is
    not about a particular program. -/
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
        simp [emitStmt, CS.fresh, hs, hv, Trie.get_set_self]
      · have hlt : i < n := by omega
        simp only [emitStmt, CS.fresh, hs]
        rw [Trie.get_set_ne _ n i _ hin]
        exact he i hlt
  | call fn args =>
      refine ⟨by simp [emitStmt, CS.fresh, hv, Stmt.binds], by
        simp [emitStmt, CS.fresh, hs, Stmt.binds], ?_⟩
      intro i hi
      simp [Stmt.binds] at hi
      by_cases hin : i = n
      · subst hin
        simp [emitStmt, CS.fresh, hs, hv, Trie.get_set_self]
      · have hlt : i < n := by omega
        simp only [emitStmt, CS.fresh, hs]
        rw [Trie.get_set_ne _ n i _ hin]
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
    value ends up. The dynamic half is `stmts_msim`. -/
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

/-- What `emitStmt` puts at the head of the block for `iadd`: the fresh value is
    `n`, and both operands resolve to themselves because `s` is aligned. -/
theorem emit_iadd (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.iadd a b))).cur = .iadd ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem emit_isub (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.isub a b))).cur = .isub ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem emit_imul (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.imul a b))).cur = .imul ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem emit_udiv (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.udiv a b))).cur = .udiv ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem emit_ishl (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.ishl a b))).cur = .ishl ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem emit_ushr (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.ushr a b))).cur = .ushr ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem emit_band (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.band a b))).cur = .band ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem emit_bandNot (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.bandNot a b))).cur = .bandNot ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem emit_bor (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.bor a b))).cur = .bor ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem emit_bxor (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.bxor a b))).cur = .bxor ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem emit_fadd (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.fadd a b))).cur = .fadd ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem emit_fsub (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.fsub a b))).cur = .fsub ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem emit_fmul (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.fmul a b))).cur = .fmul ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem emit_fmax (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.fmax a b))).cur = .fmax ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem emit_fmin (s : CS) (n : Nat) (ha : Aligned s n) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.fmin a b))).cur = .fmin ⟨n⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem emit_ineg (s : CS) (n : Nat) (ha : Aligned s n) (a : R) (hab : a < n) :
    (emitStmt s (.op (.ineg a))).cur = .ineg ⟨n⟩ ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem emit_ctz (s : CS) (n : Nat) (ha : Aligned s n) (a : R) (hab : a < n) :
    (emitStmt s (.op (.ctz a))).cur = .ctz ⟨n⟩ ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem emit_popcnt (s : CS) (n : Nat) (ha : Aligned s n) (a : R) (hab : a < n) :
    (emitStmt s (.op (.popcnt a))).cur = .popcnt ⟨n⟩ ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem emit_ireduce32 (s : CS) (n : Nat) (ha : Aligned s n) (a : R) (hab : a < n) :
    (emitStmt s (.op (.ireduce32 a))).cur = .ireduce32 ⟨n⟩ ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem emit_uextend64 (s : CS) (n : Nat) (ha : Aligned s n) (a : R) (hab : a < n) :
    (emitStmt s (.op (.uextend64 a))).cur = .uextend64 ⟨n⟩ ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem emit_sextend64 (s : CS) (n : Nat) (ha : Aligned s n) (a : R) (hab : a < n) :
    (emitStmt s (.op (.sextend64 a))).cur = .sextend64 ⟨n⟩ ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem emit_fneg (s : CS) (n : Nat) (ha : Aligned s n) (a : R) (hab : a < n) :
    (emitStmt s (.op (.fneg a))).cur = .fneg ⟨n⟩ ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem emit_fpromote (s : CS) (n : Nat) (ha : Aligned s n) (a : R) (hab : a < n) :
    (emitStmt s (.op (.fpromote a))).cur = .fpromote ⟨n⟩ ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem emit_vhighBits (s : CS) (n : Nat) (ha : Aligned s n) (a : R) (hab : a < n) :
    (emitStmt s (.op (.vhighBits a))).cur = .vhighBits ⟨n⟩ ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem emit_fcvtFromSint (s : CS) (n : Nat) (ha : Aligned s n) (ty : ClifTy) (a : R)
    (hab : a < n) :
    (emitStmt s (.op (.fcvtFromSint ty a))).cur = .fcvtFromSint ⟨n⟩ ty ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem emit_fcvtToUint (s : CS) (n : Nat) (ha : Aligned s n) (ty : ClifTy) (a : R)
    (hab : a < n) :
    (emitStmt s (.op (.fcvtToUint ty a))).cur = .fcvtToUint ⟨n⟩ ty ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem emit_splat (s : CS) (n : Nat) (ha : Aligned s n) (ty : ClifTy) (a : R)
    (hab : a < n) :
    (emitStmt s (.op (.splat ty a))).cur = .splat ⟨n⟩ ty ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem emit_bitcast (s : CS) (n : Nat) (ha : Aligned s n) (ty : ClifTy) (a : R)
    (hab : a < n) :
    (emitStmt s (.op (.bitcast ty a))).cur = .bitcast ⟨n⟩ ty ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem emit_icmp (s : CS) (n : Nat) (ha : Aligned s n) (c : ICmpCond) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.icmp c a b))).cur = .icmp ⟨n⟩ c ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem emit_fcmp (s : CS) (n : Nat) (ha : Aligned s n) (c : FloatCC) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.fcmp c a b))).cur = .fcmp ⟨n⟩ c ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem emit_load (s : CS) (n : Nat) (ha : Aligned s n) (op : LoadOp) (a : R)
    (hab : a < n) :
    (emitStmt s (.op (.load op a))).cur = .load ⟨n⟩ op ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem emit_extractlane (s : CS) (n : Nat) (ha : Aligned s n) (a : R) (l : Nat)
    (hab : a < n) :
    (emitStmt s (.op (.extractlane a l))).cur = .extractlane ⟨n⟩ ⟨a⟩ l :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem emit_select (s : CS) (n : Nat) (ha : Aligned s n) (c a b : R)
    (hcb : c < n) (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.select c a b))).cur = .select ⟨n⟩ ⟨c⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he c hcb, he a hab, he b hbb, h1]

theorem emit_bitselect (s : CS) (n : Nat) (ha : Aligned s n) (c a b : R)
    (hcb : c < n) (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.bitselect c a b))).cur = .bitselect ⟨n⟩ ⟨c⟩ ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he c hcb, he a hab, he b hbb, h1]

theorem emit_ibin (s : CS) (n : Nat) (ha : Aligned s n) (k : IBin) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.ibin k a b))).cur = .ibin ⟨n⟩ k ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem emit_ishift (s : CS) (n : Nat) (ha : Aligned s n) (k : IShift) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.ishift k a b))).cur = .ishift ⟨n⟩ k ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem emit_fbin (s : CS) (n : Nat) (ha : Aligned s n) (k : FBin) (a b : R)
    (hab : a < n) (hbb : b < n) :
    (emitStmt s (.op (.fbin k a b))).cur = .fbin ⟨n⟩ k ⟨a⟩ ⟨b⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, h1]

theorem emit_iun (s : CS) (n : Nat) (ha : Aligned s n) (k : IUn) (a : R)
    (hab : a < n) :
    (emitStmt s (.op (.iun k a))).cur = .iun ⟨n⟩ k ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem emit_fun1 (s : CS) (n : Nat) (ha : Aligned s n) (k : FUn) (a : R)
    (hab : a < n) :
    (emitStmt s (.op (.fun1 k a))).cur = .fun1 ⟨n⟩ k ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem emit_fconv (s : CS) (n : Nat) (ha : Aligned s n) (k : FConv) (t : ClifTy) (a : R)
    (hab : a < n) :
    (emitStmt s (.op (.fconv k t a))).cur = .fconv ⟨n⟩ k t ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem emit_iext (s : CS) (n : Nat) (ha : Aligned s n) (k : IExt) (t : ClifTy) (a : R)
    (hab : a < n) :
    (emitStmt s (.op (.iext k t a))).cur = .iext ⟨n⟩ k t ⟨a⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, h1]

theorem emit_fma (s : CS) (n : Nat) (ha : Aligned s n) (a b c : R)
    (hab : a < n) (hbb : b < n) (hcb : c < n) :
    (emitStmt s (.op (.fma a b c))).cur = .fma ⟨n⟩ ⟨a⟩ ⟨b⟩ ⟨c⟩ :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp [emitStmt, CS.fresh, CS.get, he a hab, he b hbb, he c hcb, h1]

theorem emit_iconst (s : CS) (n : Nat) (ha : Aligned s n) (ty : ClifTy) (k : Int) :
    (emitStmt s (.op (.iconst ty k))).cur = .iconst ⟨n⟩ ty k :: s.cur := by
  obtain ⟨h1, _, _⟩ := ha
  simp [emitStmt, CS.fresh, h1]

theorem emit_funcAddr (s : CS) (n : Nat) (ha : Aligned s n) (i : Nat) :
    (emitStmt s (.op (.funcAddr i))).cur = .funcAddr ⟨n⟩ (.local i) :: s.cur := by
  obtain ⟨h1, _, _⟩ := ha
  simp [emitStmt, CS.fresh, h1]

theorem emit_memory (s : CS) (n : Nat) (ha : Aligned s n) :
    (emitStmt s (.op .memory)).cur = .memory ⟨n⟩ :: s.cur := by
  obtain ⟨h1, _, _⟩ := ha
  simp [emitStmt, CS.fresh, h1]

theorem emit_fconst (s : CS) (n : Nat) (ha : Aligned s n) (ty : ClifTy) (b : UInt64) :
    (emitStmt s (.op (.fconst ty b))).cur = .fconst ⟨n⟩ ty b :: s.cur := by
  obtain ⟨h1, _, _⟩ := ha
  simp [emitStmt, CS.fresh, h1]

-- ---------------------------------------------------------------------------
-- Reading and writing values
-- ---------------------------------------------------------------------------

/-- A value is only readable where the array reaches. -/
theorem getV_lt (vals : Blocks.Vals) (i : Nat) (x : Sem.V)
    (h : Blocks.getV vals ⟨i⟩ = some x) : i < vals.size := by
  have hs : (vals[i]?).isSome = true := by
    show (Blocks.getV vals ⟨i⟩).isSome = true
    rw [h]; rfl
  simpa using hs

/-- Writing one value leaves every *other* readable value alone. -/
theorem getV_setV_ne (vals : Blocks.Vals) (d e : Nat) (y : Sem.V)
    (hne : d ≠ e) (he : e < vals.size) :
    Blocks.getV (Blocks.setV vals ⟨d⟩ y) ⟨e⟩ = Blocks.getV vals ⟨e⟩ := by
  rcases Nat.lt_or_ge d vals.size with hd | hd
  · simp [Blocks.getV, Blocks.setV, hd, Array.getElem?_setIfInBounds, hne, Ne.symm hne]
  · have happ : (vals ++ Array.replicate (d + 1 - vals.size) (default : Sem.V))[e]?
        = vals[e]? := by
      simp [Array.getElem?_append_left, he]
    simp [Blocks.getV, Blocks.setV, Nat.not_lt.mpr hd, Array.getElem?_setIfInBounds,
          hne, Ne.symm hne, happ]

-- ---------------------------------------------------------------------------
-- Stores and calls, as equations
-- ---------------------------------------------------------------------------

/-- `runInsts`' `istore8` arm, as an equation. Holds by definition. -/
theorem runInsts_istore8 (lc : Sem.Locals) (s : Blocks.BSt) (v a : Val) (rest : List Inst) :
    Blocks.runInsts lc s (.istore8 v a :: rest)
      = match Blocks.getV s.vals v, Blocks.getV s.vals a with
        | some (.sc _ b), some (.sc _ addr) =>
            match s.world.mem.store addr 1 (b &&& 0xff) with
            | some m => Blocks.runInsts lc
                { s with world := { Sem.obsStore s.world addr 1 (b &&& 0xff) with mem := m } } rest
            | none => .stuck "istore8 to unmapped address"
        | _, _ => .stuck "istore8 operand is not defined" := rfl

/-- `runInsts`' untyped-`store` arm, as an equation. -/
theorem runInsts_store (lc : Sem.Locals) (s : Blocks.BSt) (v a : Val) (rest : List Inst) :
    Blocks.runInsts lc s (.store v a :: rest)
      = match Blocks.doStore s v a none with
        | .error m => .stuck m
        | .ok s' => Blocks.runInsts lc s' rest := rfl

/-- `runInsts`' typed-`store` arm, as an equation. -/
theorem runInsts_storeTyped (lc : Sem.Locals) (s : Blocks.BSt) (ty : ClifTy) (v a : Val)
    (rest : List Inst) :
    Blocks.runInsts lc s (.storeTyped ty v a :: rest)
      = match Blocks.doStore s v a (some ty) with
        | .error m => .stuck m
        | .ok s' => Blocks.runInsts lc s' rest := rfl

/-- `runInsts`' `call` arm, as an equation. -/
theorem runInsts_call (lc : Sem.Locals) (s : Blocks.BSt) (d : Option Val) (c : IR.Callee)
    (args : List Val) (rest : List Inst) :
    Blocks.runInsts lc s (.call d c args :: rest)
      = match args.mapM (Blocks.getV s.vals) with
        | none => .stuck "call argument is not defined"
        | some vs =>
            match Sem.callOf lc c vs (Sem.obsCall s.world c vs) with
            | none => Sem.failOf lc c vs (Sem.obsCall s.world c vs)
            | some (res, w') =>
                match d, res with
                | some dv, some r =>
                    Blocks.runInsts lc { vals := Blocks.setV s.vals dv r, world := w' } rest
                | none, _ => Blocks.runInsts lc { s with world := w' } rest
                | some _, none => .stuck "the call returned nothing to bind" := rfl

-- ---------------------------------------------------------------------------
-- The instructions a statement list emits
-- ---------------------------------------------------------------------------

/-- The instruction `emitStmt` writes for a statement.

    Every form conses exactly one instruction onto `cur`, so reading it back off
    the head is faithful and saves restating the thirty-six-way match. -/
def instOf (s : CS) (st : Stmt) : Inst := (emitStmt s st).cur.headD (.ret none)

/-- …which is what "exactly one" means. -/
theorem emitStmt_cur (s : CS) (st : Stmt) : (emitStmt s st).cur = instOf s st :: s.cur := by
  cases st <;> rfl

/-- What a statement list emits, in program order — `cur` holds it reversed. -/
def emittedList : CS → List Stmt → List Inst
  | _, [] => []
  | s, st :: ss => instOf s st :: emittedList (emitStmt s st) ss

theorem emittedList_length : ∀ (ss : List Stmt) (s : CS),
    (emittedList s ss).length = ss.length := by
  intro ss
  induction ss with
  | nil => intro s; rfl
  | cons st ss ih => intro s; simp [emittedList, ih]

/-- `emittedList` is the emitter, not a second account of it. -/
theorem emitStmts_cur : ∀ (ss : List Stmt) (s : CS),
    (emitStmts s ss).cur = (emittedList s ss).reverse ++ s.cur := by
  intro ss
  induction ss with
  | nil => intro s; simp [emitStmts, emittedList]
  | cons st ss ih =>
      intro s
      show (emitStmts (emitStmt s st) ss).cur = _
      rw [ih (emitStmt s st), emitStmt_cur]
      simp [emittedList]

/-- What `emitStmt` writes for a call: the fresh value is `n` and every argument
    resolves to itself, because `s` is aligned and they are in scope. -/
theorem emit_call (s : CS) (n : Nat) (ha : Aligned s n) (c : IR.Callee) (args : List R)
    (hall : ∀ r ∈ args, r < n) :
    (emitStmt s (.call c args)).cur
      = .call (some ⟨n⟩) c (args.map (fun r => ⟨r⟩)) :: s.cur := by
  obtain ⟨h1, _, he⟩ := ha
  simp only [emitStmt, CS.fresh, h1, List.cons.injEq, Inst.call.injEq, and_true, true_and]
  apply List.map_congr_left
  intro r hrm
  simp [CS.get, he r (hall r hrm)]

theorem emit_callVoid (s : CS) (n : Nat) (ha : Aligned s n) (c : IR.Callee) (args : List R)
    (hall : ∀ r ∈ args, r < n) :
    (emitStmt s (.callVoid c args)).cur
      = .call none c (args.map (fun r => ⟨r⟩)) :: s.cur := by
  obtain ⟨_, _, he⟩ := ha
  have hm : args.map s.get = args.map (fun r => (⟨r⟩ : Val)) := by
    apply List.map_congr_left
    intro r hrm
    simp [CS.get, he r (hall r hrm)]
  simp [emitStmt, hm]

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
theorem runFrom_block (lc : Sem.Locals) (f : FuncData) (steps : Nat) (s : Blocks.BSt)
    (b : BlockData) (hnd : (f.blocks.map (·.ref.id)).Nodup) (hb : b ∈ f.blocks)
    (args : List Sem.V) (hlen : b.params.length = args.length) :
    Blocks.runFrom lc f (steps + 1) s b.ref.id args
      = (match Blocks.runInsts lc
            ⟨(b.params.zip args).foldl (fun vs pa => Blocks.setV vs pa.1.1 pa.2) s.vals,
             s.world⟩ b.insts with
        | .stuck m => .stuck m
        | .misuse m => .misuse m
        | .fault m => .fault m
        | .ok (s', next) w =>
            match next with
            | .done r => .ok r w
            | .goto t vs => Blocks.runFrom lc f steps { s' with world := w } t vs) := by
  rw [Blocks.runFrom, find_blk f.blocks hnd b hb]
  simp only [hlen, bne_self_eq_false, Bool.false_eq_true, if_false]


-- ---------------------------------------------------------------------------
-- Block parameters are slots
-- ---------------------------------------------------------------------------

/-- `Aligned` without the `slots` field, which `open'` does not set — its
    callers do, right after. -/
def EnvAligned (s : CS) (n : Nat) : Prop :=
  s.nextVal = n ∧ ∀ i, i < n → s.env.get i = some ⟨i⟩

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
            env := st.fresh.2.env.set (n0 + i) st.fresh.1 } (n0 + (i + 1)) := by
        constructor
        · simp [CS.fresh, hn]; omega
        · intro j hj
          rcases Nat.lt_or_ge j (n0 + i) with hlt | hge
          · rw [Trie.get_set_ne _ (n0 + i) j _ (Nat.ne_of_lt hlt)]
            exact he j hlt
          · have hje : j = n0 + i := by omega
            subst hje
            simp [Trie.get_set_self, CS.fresh, hn]
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
    (env : Trie Val) (cr : Nat) (pars : List (Val × ClifTy)) :
    Grows s { s with nextBlk := s.nextBlk + k, cur := cur, nextVal := nv,
                     slots := sl, env := env, curRef := cr, curPars := pars } :=
  ⟨by simp, by simp⟩


-- ---------------------------------------------------------------------------
-- From the emitter's list to the function's
-- ---------------------------------------------------------------------------

/-!
`compileBody` does not ship `done` as emitted: it ships
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

/-- **The lookup `runFrom` performs, against the list `compileBody` ships.**

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
theorem runFrom_mono (lc : Sem.Locals) (f : FuncData) :
    ∀ (steps : Nat) (s : Blocks.BSt) (blk : Nat) (args : List Sem.V) (r : Option Sem.V)
      (w : Sem.World), Blocks.runFrom lc f steps s blk args = .ok r w →
      ∀ steps', steps ≤ steps' → Blocks.runFrom lc f steps' s blk args = .ok r w := by
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
            cases hins : Blocks.runInsts lc
                ⟨(b.params.zip args).foldl (fun vs pa => Blocks.setV vs pa.1.1 pa.2) s.vals,
                 s.world⟩ b.insts with
            | stuck m => simp [hins] at h
            | misuse m => simp [hins] at h
            | fault m => simp [hins] at h
            | ok p w' =>
              obtain ⟨s', next⟩ := p
              simp only [hins] at h ⊢
              cases next with
              | done => exact h
              | goto t vs => exact ih _ t vs r w h steps'' hle'


/-- The same, at the level `compile_sound` is stated: a successful compiled run
    is unchanged by a larger budget. -/
theorem run_mono (lc : Sem.Locals) (f : FuncData) (args : List Sem.V) (w : Sem.World)
    (steps : Nat) (obs : List Sem.Obs) (w' : Sem.World)
    (h : Blocks.run lc f args w steps = .ok obs w') :
    ∀ steps', steps ≤ steps' → Blocks.run lc f args w steps' = .ok obs w' := by
  intro steps' hle
  rw [Blocks.run] at h ⊢
  cases hr : Blocks.runFrom lc f steps ⟨#[], w⟩ 0 args with
  | stuck m => simp [hr] at h
  | misuse m => simp [hr] at h
  | fault m => simp [hr] at h
  | ok a ww =>
      simp only [hr] at h
      rw [runFrom_mono lc f steps ⟨#[], w⟩ 0 args a ww hr steps' hle]
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
    bound that works.

    `compileSoundE_of_ok` proves it for every body that passes
    `scopeOk` and that the term interpreter runs to completion. -/
def CompileSoundE (idx : Nat) (env : FnEnv) (params : List ClifTy) (c : Code)
    (args : List V) (w : World) (fuel : Nat) : Prop :=
  ∃ steps, Sem.run { env, steps := fuel } args w c
    = Blocks.run noLocals (compileBody idx c env params) args w steps

/-- Everything already proved at the matched-budget statement carries over, so
    separating the budgets costs none of the existing results. -/
theorem compileSound_toE (idx : Nat) (env : FnEnv) (params : List ClifTy) (c : Code)
    (args : List V) (w : World) (fuel : Nat) (h : CompileSound idx env params c args w fuel) :
    CompileSoundE idx env params c args w fuel := ⟨fuel, h⟩

/-- The base cases at the budget-separated statement. -/
theorem empty_soundE (idx : Nat) (env : FnEnv) (base data dlen out olen : V)
    (w : World) (fuel : Nat) :
    CompileSoundE idx env five [] [base, data, dlen, out, olen] w (fuel + 1) :=
  compileSound_toE _ _ _ _ _ _ _ (empty_sound idx env base data dlen out olen w fuel)

theorem single_iconst_soundE (idx : Nat) (env : FnEnv) (t : ClifTy) (k : Int)
    (base data dlen out olen : V) (w : World) (fuel : Nat) :
    CompileSoundE idx env five [.straight [.op (.iconst t k)]]
      [base, data, dlen, out, olen] w (fuel + 2) :=
  compileSound_toE _ _ _ _ _ _ _
    (single_iconst_sound idx env t k base data dlen out olen w fuel)


-- ---------------------------------------------------------------------------
-- A straight-line body, as a whole function
-- ---------------------------------------------------------------------------

/-!
The frame around a straight-line body's instructions: the entry block that binds
the parameters, and the `ret` that ends it.
-/

/-- One `Piece`, reduced. The term's fuel is spent on the piece and then on the
    empty tail, which is why two is the smallest budget that gets through. -/
theorem runCode_straight (cfg : Sem.Cfg) (Γ : Sem.Env) (w : World) (ss : List Stmt)
    (fuel : Nat) :
    Sem.runCode (fuel + 2) cfg Γ w [.straight ss]
      = match Sem.runStmts cfg Γ w ss with
        | .ok Γ' w' => .ok Γ' w'
        | .stuck m => .stuck m
        | .misuse m => .misuse m
        | .fault m => .fault m := by
  rw [Sem.runCode, Sem.runPiece]
  cases Sem.runStmts cfg Γ w ss <;> simp [Sem.runCode]

/-- The parameters `open'` gives a block: one fresh value per type, numbered
    from the value counter, which is also the slot they bind. -/
def parsOf (n0 : Nat) : List ClifTy → List (Val × ClifTy)
  | [] => []
  | t :: ts => (⟨n0⟩, t) :: parsOf (n0 + 1) ts

theorem parsOf_length : ∀ (tys : List ClifTy) (n0 : Nat), (parsOf n0 tys).length = tys.length := by
  intro tys
  induction tys with
  | nil => intro _; rfl
  | cons t ts ih => intro n0; simp [parsOf, ih]

theorem open'_go_pars : ∀ (tys : List ClifTy) (st : CS) (n0 i : Nat),
    (CS.open'.go n0 st i tys).curPars = st.curPars ++ parsOf st.nextVal tys := by
  intro tys
  induction tys with
  | nil => intro st n0 i; simp [CS.open'.go, parsOf]
  | cons t ts ih =>
      intro st n0 i
      rw [CS.open'.go, ih _ n0 (i + 1)]
      simp [CS.fresh, parsOf]

/-- Opening a block from a state whose value counter is `n` gives it exactly the
    parameters `parsOf n` names. -/
theorem open'_pars (s : CS) (ref : Nat) (tys : List ClifTy) (f : Nat) :
    (CS.open' s ref tys f).curPars = parsOf s.nextVal tys := by
  rw [CS.open', open'_go_pars tys { s with curRef := ref, curPars := [] } f 0]
  simp

theorem emitStmt_curPars (s : CS) (st : Stmt) : (emitStmt s st).curPars = s.curPars := by
  cases st <;> rfl

theorem emitStmts_curPars : ∀ (ss : List Stmt) (s : CS),
    (emitStmts s ss).curPars = s.curPars := by
  intro ss
  induction ss with
  | nil => intro s; rfl
  | cons st ss ih =>
      intro s
      show (emitStmts (emitStmt s st) ss).curPars = _
      rw [ih (emitStmt s st), emitStmt_curPars]

/-- Opening a block leaves the instructions already emitted alone. -/
theorem open'_go_cur : ∀ (tys : List ClifTy) (st : CS) (n0 i : Nat),
    (CS.open'.go n0 st i tys).cur = st.cur := by
  intro tys
  induction tys with
  | nil => intro st n0 i; simp [CS.open'.go]
  | cons t ts ih => intro st n0 i; simpa [CS.open'.go, CS.fresh] using ih _ n0 (i + 1)

/-- The emitter's state when a body's statements start: the entry block, open,
    its parameters bound to slots `0..`. -/
def entryCS (params : List ClifTy) : CS :=
  { ({ nextVal := 0, nextBlk := 1, slots := 0, env := .nil, curRef := 0,
       curPars := [], cur := [], done := [] } : CS).open' 0 params 0 with
    slots := params.length }

theorem entryCS_aligned (params : List ClifTy) : Aligned (entryCS params) params.length := by
  have ha0 : Aligned ({ nextVal := 0, nextBlk := 1, slots := 0, env := .nil, curRef := 0,
                        curPars := [], cur := [], done := [] } : CS) 0 :=
    ⟨rfl, rfl, by intro i hi; omega⟩
  have := open'_aligned _ 0 ha0 0 params
  simpa [entryCS] using this

theorem entryCS_done (params : List ClifTy) : (entryCS params).done = [] := by
  have := (open'_go_blk params
    { ({ nextVal := 0, nextBlk := 1, slots := 0, env := .nil, curRef := 0,
         curPars := [], cur := [], done := [] } : CS) with curRef := 0, curPars := [] } 0 0).1
  simpa [entryCS, CS.open'] using this

theorem entryCS_curRef (params : List ClifTy) : (entryCS params).curRef = 0 := by
  have := (open'_go_blk params
    { ({ nextVal := 0, nextBlk := 1, slots := 0, env := .nil, curRef := 0,
         curPars := [], cur := [], done := [] } : CS) with curRef := 0, curPars := [] } 0 0).2.2
  simpa [entryCS, CS.open'] using this

theorem entryCS_cur (params : List ClifTy) : (entryCS params).cur = [] := by
  have := open'_go_cur params
    { ({ nextVal := 0, nextBlk := 1, slots := 0, env := .nil, curRef := 0,
         curPars := [], cur := [], done := [] } : CS) with curRef := 0, curPars := [] } 0 0
  simpa [entryCS, CS.open'] using this

theorem entryCS_curPars (params : List ClifTy) : (entryCS params).curPars = parsOf 0 params := by
  simpa [entryCS] using open'_pars _ 0 params 0

/-- **What a straight-line body compiles to**: one block, holding the entry
    parameters, the statements' instructions in order, and a `ret`. -/
theorem compileBody_straight (idx : Nat) (env : FnEnv) (params : List ClifTy)
    (ss : List Stmt) :
    compileBody idx [.straight ss] env params
      = { index := idx,
          blocks := [{ ref := ⟨0⟩, params := parsOf 0 params,
                       insts := emittedList (entryCS params) ss ++ [.ret none] }] } := by
  have hEmit : ∀ s : CS, emitCode HProg.fuel s [Piece.straight ss] = emitStmts s ss := by
    intro s; simp [emitCode, emitPiece, HProg.fuel]
  have h : compileBody idx [.straight ss] env params
      = { index := idx,
          blocks := ((emitStmts (entryCS params) ss).close (.ret none)).done.mergeSort
                      (fun a b => a.ref.id ≤ b.ref.id) } := by
    simp only [compileBody, Id.run, entryCS, hEmit]
    rfl
  obtain ⟨hdone, _, hcur⟩ := emitStmts_blk ss (entryCS params)
  rw [h, CS.close]
  simp [hdone, hcur, entryCS_done, entryCS_curRef, entryCS_cur, entryCS_curPars,
        emitStmts_curPars, emitStmts_cur]

end AlgorithmLib.HProg
