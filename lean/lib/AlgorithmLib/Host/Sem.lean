module
public import AlgorithmLib.Host.Blas
public import AlgorithmLib.Host.CpuLib
public import AlgorithmLib.Host.UsbLib
public import AlgorithmLib.Host.Wgpu
public import AlgorithmLib.Host.WindowLib
meta import AlgorithmLib.Host.Blas
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Host.Sem` — what an `HProg` term does

A concrete, executable semantics for the terms `HProg` defines: real bit
patterns, real memory, real floating point. Two things use it.

* **As an oracle.** The differential harness runs a term here and the artifact
  it compiles to through the real JIT, and compares the bytes. A wrong decode
  arm on the Rust side, a field-order drift between Lean and serde, or a
  Cranelift semantic change on upgrade all show up as a mismatch.
* **As the left-hand side of `compile_sound`.** Execution records an
  observation trace — the calls it makes and the stores it performs, in order —
  which is the statement the compilation proof is against.

Everything here is a pure function of the state, including the file system and
the FFI: `Sem.run` takes a `World` and returns one. That is what lets a term's
behavior be a *value* rather than an effect.

Unlike `HProg.wf`, none of this is meant to reduce in the kernel — it runs on
`ByteArray`, which the compiler makes fast and the kernel does not. Statements
about executions are proved from the equations, not by `decide`.

## What this assumes, and what checks it

`evalOp` says `iadd` wraps at the operand's width, `fmax` returns the other
operand when one is NaN, `fcvt_to_uint` saturates rather than traps, `ushr`
takes its shift amount modulo the width. **None of that is proven.** Cranelift
publishes no formal semantics to prove against; those meanings live in its
instruction documentation and its lowering rules. What is written here is a
transcription of them, and a transcription is an assumption — the same grade of
assumption as the frames in `Host.Frames`.

Two things keep that from being load-bearing in the wrong place.

*The compilation proof does not rest on it.* `compile_sound` relates a term's
execution to its compiled form's, and both sides call the same `evalOp`. It
therefore says *compiling preserves meaning*, not *the meaning is Cranelift's*.
A wrong arm here would not make that theorem false; it would make this file a
bad oracle, which is a different failure and a detectable one.

*Differential execution is what detects it.* `HProgCorpus` runs every operation
over operands chosen to separate the arms that are easy to get wrong — zero,
±1, `i64::MIN`/`MAX`, NaN, ±∞, subnormals, the saturation boundaries — in this
interpreter and through the real JIT, and compares the bytes. That check is not
circular: the Rust decoder says only *which* Cranelift instruction to emit, this
file says what that instruction *computes*, and the machine says who was right.
It is also the check that runs again on the next Cranelift upgrade.
-/

namespace AlgorithmLib.HProg.Sem

open AlgorithmLib.IR
open AlgorithmLib.HProg

-- ---------------------------------------------------------------------------
-- Slot accounting
--
-- A loop's exit block and a branch's join block bind their parameters after
-- every slot the regions before them defined, whether or not those regions
-- ran. So the interpreter needs the *static* count, exactly as the compiler
-- and `wf` compute it.
-- ---------------------------------------------------------------------------

def stmtsSlots (n : Nat) (ss : List Stmt) : Nat :=
  ss.foldl (fun n s => match s with
    | .op _ | .call _ _ => n + 1
    | _ => n) n

def slotsGo : Nat → Nat → List Piece → Nat
  | 0, n, _ => n
  | _ + 1, n, [] => n
  | fuel + 1, n, .straight ss :: ps => slotsGo fuel (stmtsSlots n ss) ps
  | fuel + 1, n, .loop l pre body :: ps =>
      let afterBody := slotsGo fuel (slotsGo fuel (n + l.pTys.length) pre + l.pTys.length) body
      slotsGo fuel (afterBody + l.exitTys.length) ps
  | fuel + 1, n, .ite m thn els _ _ :: ps =>
      let afterEls := slotsGo fuel (slotsGo fuel n thn) els
      let jn := if termsGo fuel thn && termsGo fuel els then 0 else m.jTys.length
      slotsGo fuel (afterEls + jn) ps
  | fuel + 1, n, .dloop l body :: ps =>
      let afterBody := slotsGo fuel (n + l.pTys.length) body
      slotsGo fuel (afterBody + l.exitTys.length) ps
  | fuel + 1, n, .br _ _ :: ps | fuel + 1, n, .cont _ _ :: ps => slotsGo fuel n ps

/-- The number of slots in scope after `c`, starting from `n`. -/
def slotsOf (n : Nat) (c : Code) : Nat := slotsGo fuel n c

-- ---------------------------------------------------------------------------
-- Execution
-- ---------------------------------------------------------------------------

/-- Grow `Γ` to `n` slots so a binder lands at the index the compiled block
    parameter has, then append `vs`. -/
def bindAt (Γ : Env) (n : Nat) (vs : List V) : Env :=
  (Γ.take n ++ Array.replicate (n - Γ.size) default) ++ vs.toArray

def obsCall (w : World) (c : Callee) (args : List V) : World :=
  { w with obs := .call c args :: w.obs }

def obsStore (w : World) (a : UInt64) (n : Nat) (v : UInt64) : World :=
  { w with obs := .store a n v :: w.obs }

/-- How a call to one of the program's own functions ends without answering:
    as its body ended. -/
inductive Fail where
  | stuck (why : String)
  | misuse (why : String)
  | fault (why : String)

/-- The outcome a call that ended so gives its caller: the callee's own. -/
def Fail.out {α : Type} : Fail → Outcome α
  | .stuck m => .stuck m
  | .misuse m => .misuse m
  | .fault m => .fault m

/-- What a call to one of the program's own functions does, by its index: from
    the arguments and the world, the answer and the world after, or how it
    ended without one. A callee's misuse or fault is its caller's. -/
abbrev Locals := Nat → List V → World → Except Fail (Option V × World)

/-- No local call has a meaning: each one is stuck. -/
def noLocals : Locals := fun i _ _ => .error (.stuck s!"calls u0:{i}, whose meaning is not given")

structure Cfg where
  env : FnEnv
  /-- How many loop iterations one run may take in total. -/
  steps : Nat := 100000000
  locals : Locals := noLocals

/-- `cl_thread_spawn`: run function `fn` on the argument pointer, as a worker.
    A worker takes one pointer and answers nothing; one that answers is refused,
    as the runtime refuses it. The worker runs to completion here, and memory
    stays frozen until it is joined. -/
def spawnWorker (locals : Locals) (vs : List V) (w : World) : Option (Option V × World) :=
  match vs with
  | [.sc _ ctx, .sc _ fn, .sc _ arg] =>
      if ctx == 0 then some (some (ofInt .i64 (-1)), w)
      else if !(ctx == threadCtx && w.thread.live) || w.thread.outstanding.isSome then none
      else
        match locals fn.toNat [.sc .i64 arg] w with
        | .error _ => none
        | .ok (some _, _) => some (some (ofInt .i64 (-1)), w)
        | .ok (none, w') =>
            if w'.mem.frozen || w'.thread.outstanding.isSome then none
            else
              let h := w'.thread.next
              some (some (ofInt .i64 h),
                    { w' with mem := { w'.mem with frozen := true },
                              thread := { w'.thread with next := h + 1, outstanding := some h } })
  | _ => none

/-- The value an `atomic_rmw` leaves in memory, from the old value and its
    operand, at width `t`. -/
def rmwNew (k : AtomicRmw) (t : ClifTy) (old x : UInt64) : UInt64 :=
  let m := widthMask t
  let (o, y) := (old &&& m, x &&& m)
  (match k with
   | .add => o + y | .sub => o - y | .and => o &&& y | .nand => ~~~(o &&& y)
   | .or => o ||| y | .xor => o ^^^ y | .xchg => y
   | .umin => if o ≤ y then o else y | .umax => if o ≥ y then o else y
   | .smin => if signed t o ≤ signed t y then o else y
   | .smax => if signed t o ≥ signed t y then o else y) &&& m

/-- What an atomic instruction does.

    Its sequential reading, which is exact for every run the model admits: a
    spawned worker runs at its spawn and the spawner may touch no memory until
    the join, so no two accesses the model admits are concurrent. A program
    whose answer depends on how atomics from concurrently running threads
    interleave is outside what this states.

    The address must be naturally aligned: a misaligned atomic faults on
    AArch64, so a portable program has none. -/
def atomicCall (a : Atomic) (vs : List V) (w : World) : Option (Option V × World) :=
  let aligned (t : ClifTy) (p : UInt64) : Bool := p % UInt64.ofNat (tyBytes t) == 0
  match a, vs with
  | .fence, [] => some (none, w)
  | .load t, [.sc .i64 p] => do
      if !aligned t p then none else
      let b ← w.mem.load p (tyBytes t)
      some (some (.sc t b), w)
  | .store t, [.sc t' x, .sc .i64 p] => do
      if t' != t || !aligned t p then none else
      let m ← w.mem.store p (tyBytes t) x
      some (none, { obsStore w p (tyBytes t) x with mem := m })
  | .rmw t k, [.sc .i64 p, .sc t' x] => do
      if t' != t || !aligned t p then none else
      let old ← w.mem.load p (tyBytes t)
      let nv := rmwNew k t old x
      let m ← w.mem.store p (tyBytes t) nv
      some (some (.sc t old), { obsStore w p (tyBytes t) nv with mem := m })
  | .cas t, [.sc .i64 p, .sc te e, .sc tn n] => do
      if te != t || tn != t || !aligned t p then none else
      let old ← w.mem.load p (tyBytes t)
      if old == e &&& widthMask t then
        let m ← w.mem.store p (tyBytes t) n
        some (some (.sc t old), { obsStore w p (tyBytes t) n with mem := m })
      else some (some (.sc t old), w)
  | _, _ => none

/-- The offset of the first zero byte at or after `a`, within its region. -/
def strlenAt (m : Mem) (a : UInt64) : Option Nat := do
  let (r, off) ← decodeAddr a
  let bs := m.region r
  let n := bs.size - off
  let i ← (List.range n).find? fun i => bs.get! (off + i) == 0
  let _ ← copyOut m a (i + 1)
  some i

/-- Whether library `l` is there: one built into the engine always is
    (`Lib.builtin`); another as the machine has it. -/
def World.has (w : World) (l : Lib) : Bool := l.builtin || w.present l

/-- **What the C library's functions do**, on their arguments' bits.
    `memcpy` between overlapping ranges is undefined, so it is not answered. -/
def cCall (f : CFn) (bits : List UInt64) (w : World) : Option (Option V × World) :=
  match f, bits with
  | .memcpy, [d, s, n] => do
      let (dn, sn, nn) := (d.toNat, s.toNat, n.toNat)
      if nn > 0 && dn < sn + nn && sn < dn + nn then none else
      let bs ← copyOut w.mem s nn
      let m ← copyIn w.mem d bs
      some (some (.sc .i64 d), { w with mem := m })
  | .memmove, [d, s, n] => do
      let bs ← copyOut w.mem s n.toNat
      let m ← copyIn w.mem d bs
      some (some (.sc .i64 d), { w with mem := m })
  | .memset, [d, c, n] => do
      let m ← copyIn w.mem d (ByteArray.mk (Array.replicate n.toNat (c &&& 0xff).toUInt8))
      some (some (.sc .i64 d), { w with mem := m })
  | .strlen, [a] => do
      let i ← strlenAt w.mem a
      some (some (.sc .i64 (UInt64.ofNat i)), w)
  -- An empty request is not modelled. Out of room, the answer is null.
  | .calloc, [n, sz] =>
      let bytes := n.toNat * sz.toNat
      let off := w.mem.pinnedNext
      if bytes == 0 then none
      else if off + bytes > regionSpan.toNat then some (some (.sc .i64 0), w)
      else
        let m := { w.mem with
          pinned := w.mem.pinned ++ ByteArray.mk (Array.replicate (off - w.mem.pinned.size + bytes) 0)
          pinnedLive := (off, bytes) :: w.mem.pinnedLive }
        some (some (.sc .i64 (addrOf .pinned off)), { w with mem := m, heap := (off, bytes) :: w.heap })
  -- The heap is pageable: the driver finishes a copy from it before the call
  -- returns, so none is in flight when it is freed.
  | .free, [p] =>
      if p == 0 then some (none, w)
      else do
        let (off, n) ← w.heapAt? p
        some (none, { w with mem := { w.mem with pinnedLive := w.mem.pinnedLive.erase (off, n) },
                                  heap := w.heap.erase (off, n) })
  | _, _ => none

/-- **What a C library function does.** A library that did not load leaves
    each of its functions absent: the engine binds them to one stub that
    answers `-1` and touches nothing, and the presence probe answers `0`.
    Every function reads its arguments' bits, as the C calling convention
    hands them over. -/
def extCall (e : Ext) (vs : List V) (w : World) : Option (Option V × World) :=
  if !w.has e.lib then
    match e with
    | .present _ => some (some (.sc .i32 0), w)
    | _ => some (e.sig.2.map (ofInt · (-1)), w)
  else
  match e, vs with
  | .present _, [] => some (some (.sc .i32 1), w)
  | .c f, vs => do
      let bits ← vs.mapM asBits
      cCall f bits w
  | .cuda f, vs => do
      let bits ← vs.mapM asBits
      cudaDrv f bits w
  | .cublas f, vs => do
      let bits ← vs.mapM asBits
      cublasCall f bits w
  | .wgpu f, vs => do
      let bits ← vs.mapM asBits
      wgpuCall f bits w
  | .window f, vs => do
      let bits ← vs.mapM asBits
      wlCall f bits w
  | .cpu f, vs => do
      let bits ← vs.mapM asBits
      cpuCall f bits w
  | .serial f, vs => do
      let bits ← vs.mapM asBits
      serialCall f bits w
  | .usb f, vs => do
      let bits ← vs.mapM asBits
      usbCall f bits w
  | _, _ => none

/-- What a call does, by what it names: an import by its contract, one of the
    program's own functions by `locals`, machine code not at all. Both
    interpreters make every call through this. -/
def callOf (locals : Locals) (c : Callee) (vs : List V) (w : World) : Option (Option V × World) :=
  match c with
  | .ffi f =>
      -- an outstanding worker may touch anything: only joining is defined
      if w.mem.frozen && f != .threadJoin && f != .threadCleanup then none
      else if f == .threadSpawn then spawnWorker locals vs w
      else callImport f.cname vs w
  | .local i => (locals i vs w).toOption
  | .native => none
  | .atomic a => atomicCall a vs w
  | .ext e => extCall e vs w

/-- Why a call that `callOf` does not answer is stuck. -/
def uncalled : Callee → String
  | .ffi f => s!"{f.cname} has no executable contract"
  | .local i => s!"calls u0:{i}, whose meaning is not given"
  | .native => "calls machine code, which the model does not read"
  | .atomic _ => "an atomic on an unaligned, unmapped or frozen address"
  | .ext e => s!"{e.lib.name}!{e.symbol} on arguments it does not define"

/-- A call `callOf` does not answer: a refused import, and a library called on
    arguments it does not define, are misuse; anything else is stuck. -/
def refused {α : Type} : Callee → Outcome α
  | .ffi f => .misuse (uncalled (.ffi f))
  | .ext e => .misuse (uncalled (.ext e))
  | c => .stuck (uncalled c)

theorem refused_ne_ok {α : Type} {c : Callee} {a : α} {w : World} : refused c ≠ .ok a w := by
  cases c <;> simp [refused]

/-- A call `callOf` does not answer: one of the program's own functions ends
    as its body did; any other call is `refused`. -/
def failOf {α : Type} (locals : Locals) (c : Callee) (vs : List V) (w : World) : Outcome α :=
  match c with
  | .local i =>
      match locals i vs w with
      | .error f => f.out
      | .ok _ => .stuck (uncalled (.local i))
  | c => refused c

theorem failOf_ne_ok {α : Type} {lc : Locals} {c : Callee} {vs : List V} {w : World} {a : α} {w' : World} :
    failOf lc c vs w ≠ .ok a w' := by
  unfold failOf
  split
  · split
    · rename_i f _; cases f <;> simp [Fail.out]
    · simp
  · exact refused_ne_ok

def runStmt (cfg : Cfg) (Γ : Env) (w : World) : Stmt → Outcome Env
  | .op o =>
      match evalOp w.mem Γ o with
      | some v => .ok (Γ.push v) w
      | none => .fault ("operation is undefined here: " ++ o.name)
  | .store ty v a =>
      match get Γ v, get Γ a with
      | some val, some (.sc _ addr) =>
          let n := tyBytes ty
          match val with
          | .sc _ b =>
              match w.mem.store addr n b with
              | some m => .ok Γ { obsStore w addr n b with mem := m }
              | none => .fault s!"store to unmapped address {addr}"
          | .vec t ls =>
              let lw := ((t.lanes.map (·.1.width)).getD 8) / 8
              match (ls.zipIdx.foldlM
                  (fun mm (x, i) => mm.store (addr + UInt64.ofNat (i * lw)) lw x) w.mem) with
              | some m => .ok Γ { obsStore w addr n 0 with mem := m }
              | none => .fault s!"vector store to unmapped address {addr}"
      | _, _ => .stuck "store operand is not in scope"
  | .storeUnaligned v a =>
      match get Γ v, get Γ a with
      | some (.sc t b), some (.sc _ addr) =>
          match w.mem.store addr (tyBytes t) b with
          | some m => .ok Γ { obsStore w addr (tyBytes t) b with mem := m }
          | none => .fault s!"store to unmapped address {addr}"
      -- a vector, lane by lane, as `store` does
      | some (.vec t ls), some (.sc _ addr) =>
          let lw := ((t.lanes.map (·.1.width)).getD 8) / 8
          match (ls.zipIdx.foldlM
              (fun mm (x, i) => mm.store (addr + UInt64.ofNat (i * lw)) lw x) w.mem) with
          | some m => .ok Γ { obsStore w addr (tyBytes t) 0 with mem := m }
          | none => .fault s!"vector store to unmapped address {addr}"
      | _, _ => .stuck "store operand is not in scope"
  | .istore8 v a =>
      match get Γ v, get Γ a with
      | some (.sc _ b), some (.sc _ addr) =>
          match w.mem.store addr 1 (b &&& 0xff) with
          | some m => .ok Γ { obsStore w addr 1 (b &&& 0xff) with mem := m }
          | none => .fault s!"istore8 to unmapped address {addr}"
      | _, _ => .stuck "istore8 operand is not in scope"
  | .call c args => runCall Γ w c args true
  | .callVoid c args => runCall Γ w c args false
where
  runCall (Γ : Env) (w : World) (c : Callee) (args : List R) (binds : Bool) :
      Outcome Env :=
    match args.mapM (get Γ) with
    | none => .stuck "a call argument is not in scope"
    | some vs =>
        match callOf cfg.locals c vs (obsCall w c vs) with
        | none => failOf cfg.locals c vs (obsCall w c vs)
        | some (res, w') =>
            if binds then
              match res with
              | some v => .ok (Γ.push v) w'
              | none => .stuck "the call returned nothing to bind"
            else .ok Γ w'

def runStmts (cfg : Cfg) : Env → World → List Stmt → Outcome Env
  | Γ, w, [] => .ok Γ w
  | Γ, w, s :: ss =>
      match runStmt cfg Γ w s with
      | .ok Γ' w' => runStmts cfg Γ' w' ss
      | .stuck m => .stuck m
      | .misuse m => .misuse m
      | .fault m => .fault m

/-- What running a region can do: finish, leave an enclosing loop, or get stuck.
    `brk` carries the environment it left from, because the exit block binds its
    parameters at a static index that may be past what the abandoned path had. -/
inductive CodeRes where
  | ok    (Γ : Env) (w : World)
  | brk   (depth : Nat) (Γ : Env) (vals : List V) (w : World)
  /-- Go round the `depth`-th enclosing top-tested loop again. -/
  | cont  (depth : Nat) (vals : List V) (w : World)
  | stuck (why : String)
  /-- A foreign call refused: see `Outcome.misuse`. -/
  | misuse (why : String)
  /-- The program's own fault: see `Outcome.fault`. -/
  | fault (why : String)

mutual

/-- Fuel bounds every loop iteration and every nested region, **and is the
    structural argument the recursion decreases on**. That second role is the
    reason it is written this way: a `partial def` runs the same programs and
    proves nothing, because Lean gives partial definitions no equations. These
    have equations, so `compile_sound` is a statement one can actually work on. -/
def runPiece : Nat → Cfg → Env → World → Piece → CodeRes
  | 0, _, _, _, _ => .stuck "step budget exhausted"
  | _ + 1, cfg, Γ, w, .straight ss =>
      match runStmts cfg Γ w ss with
      | .ok Γ' w' => .ok Γ' w'
      | .stuck m => .stuck m
      | .misuse m => .misuse m
      | .fault m => .fault m
  | _ + 1, _, Γ, w, .br depth args =>
      match args.mapM (get Γ) with
      | none => .stuck "branch-out value is not in scope"
      | some vs => .brk depth Γ vs w
  | _ + 1, _, Γ, w, .cont depth args =>
      match args.mapM (get Γ) with
      | none => .stuck "loop carry is not in scope"
      | some vs => .cont depth vs w
  | fuel + 1, cfg, Γ, w, .dloop l body =>
      let n0 := Γ.size
      let afterBody := slotsOf (n0 + l.pTys.length) body
      match l.init.mapM (get Γ) with
      | none => .stuck "loop initializer is not in scope"
      | some inits => dtrip fuel cfg Γ w l body n0 afterBody inits l.guard.isSome
  | fuel + 1, cfg, Γ, w, .loop l pre body =>
      let n0 := Γ.size
      let afterBody := slotsOf (slotsOf (n0 + l.pTys.length) pre + l.pTys.length) body
      match l.init.mapM (get Γ) with
      | none => .stuck "loop initializer is not in scope"
      | some inits => iter fuel cfg Γ w l pre body n0 afterBody inits
  | fuel + 1, cfg, Γ, w, .ite m thn els thnR elsR =>
      match get Γ m.flag with
      | some (.sc _ f) =>
          -- Both arms are numbered as if the one before it had run, because
          -- that is how `emitIte` numbers them. Only one arm executes, so the
          -- else arm starts from an environment padded past the then arm's
          -- slots; otherwise its own slots land at the then arm's indices.
          let thnEnd := slotsOf Γ.size thn
          let joinAt := slotsOf thnEnd els
          let (arm, exports, Γ0) :=
            if f != 0 then (thn, thnR, Γ) else (els, elsR, bindAt Γ thnEnd [])
          match runCode fuel cfg Γ0 w arm with
          | .stuck s => .stuck s
          | .misuse s => .misuse s
          | .fault s => .fault s
          | .brk d Γb vs w' => .brk d Γb vs w'
          | .cont d vs w' => .cont d vs w'
          | .ok Γ' w' =>
              match exports.mapM (get Γ') with
              | none => .stuck "branch export is not in scope"
              | some vs => .ok (bindAt Γ' joinAt vs) w'
      | _ => .stuck "branch condition is not in scope"

/-- One trip of a loop: bind the carries, run the condition prefix, test, then
    either leave with the exit values or run the body and go round again. The
    body gets the carries again, bound after the prefix's slots, which is where
    the compiled body block's parameters are.

    A `br 0` from either region leaves here, binding its values where the normal
    exit would have; a deeper one is passed out with its depth reduced. -/
def iter : Nat → Cfg → Env → World → Loop → List Piece → List Piece →
    Nat → Nat → List V → CodeRes
  | 0, _, _, _, _, _, _, _, _, _ => .stuck "loop exceeded its step budget"
  | fuel + 1, cfg, Γ, w, l, pre, body, n0, afterBody, carries =>
      match runCode fuel cfg (bindAt Γ n0 carries) w pre with
      | .stuck s => .stuck s
      | .misuse s => .misuse s
      | .fault s => .fault s
      | .brk 0 Γb vs w' => .ok (bindAt Γb afterBody vs) w'
      | .brk (d + 1) Γb vs w' => .brk d Γb vs w'
      | .cont 0 vs w' => iter fuel cfg Γ w' l pre body n0 afterBody vs
      | .cont (d + 1) vs w' => .cont d vs w'
      | .ok Γ1 w1 =>
          match get Γ1 l.flag with
          | some (.sc _ f) =>
              if (f != 0) == l.exitOnTrue then
                match l.exitR.mapM (get Γ1) with
                | none => .stuck "loop exit value is not in scope"
                | some vs => .ok (bindAt Γ1 afterBody vs) w1
              else
                match runCode fuel cfg
                    (bindAt Γ1 (slotsOf (n0 + l.pTys.length) pre) carries) w1 body with
                | .stuck s => .stuck s
                | .misuse s => .misuse s
                | .fault s => .fault s
                | .brk 0 Γb vs w2 => .ok (bindAt Γb afterBody vs) w2
                | .brk (d + 1) Γb vs w2 => .brk d Γb vs w2
                | .cont 0 vs w2 => iter fuel cfg Γ w2 l pre body n0 afterBody vs
                | .cont (d + 1) vs w2 => .cont d vs w2
                | .ok Γ2 w2 =>
                    match l.cont.mapM (get Γ2) with
                    | none => .stuck "loop carry is not in scope"
                    | some next => iter fuel cfg Γ w2 l pre body n0 afterBody next
          | _ => .stuck "loop condition is not in scope"

/-- One trip of a bottom-tested loop. `first` says the guard has not been read
    yet, so it is read before the body rather than the back-edge test after it;
    every later call arrives with the test already passed. -/
def dtrip : Nat → Cfg → Env → World → DLoop → List Piece →
    Nat → Nat → List V → Bool → CodeRes
  | 0, _, _, _, _, _, _, _, _, _ => .stuck "loop exceeded its step budget"
  | fuel + 1, cfg, Γ, w, l, body, n0, afterBody, carries, first =>
      let leave (Γ' : Env) (vs : List V) (w' : World) : CodeRes :=
        match l.exitIdx.mapM (fun i => vs[i]?) with
        | none => .stuck "loop exit value is not a carry"
        | some outs => .ok (bindAt Γ' afterBody outs) w'
      -- The guard is a slot of the code before the loop, the back-edge test one
      -- the body bound last; each is read in its own scope.
      let testAt (Γ' : Env) (r : R) : Option Bool :=
        match get Γ' r with
        | some (.sc _ f) => some (f != 0)
        | _ => none
      if first then
        match l.guard with
        | none => dtrip fuel cfg Γ w l body n0 afterBody carries false
        | some g =>
            match testAt Γ g with
            | none => .stuck "loop condition is not in scope"
            | some c =>
                if c == l.contOnTrue then
                  dtrip fuel cfg Γ w l body n0 afterBody carries false
                else leave Γ carries w
      else
        match runCode fuel cfg (bindAt Γ n0 carries) w body with
        | .stuck s => .stuck s
        | .misuse s => .misuse s
        | .fault s => .fault s
        | .brk 0 Γb vs w' => .ok (bindAt Γb afterBody vs) w'
        | .brk (d + 1) Γb vs w' => .brk d Γb vs w'
        | .cont 0 vs w2 => dtrip fuel cfg Γ w2 l body n0 afterBody vs false
        | .cont (d + 1) vs w' => .cont d vs w'
        | .ok Γ2 w2 =>
            -- Reaching here means the body fell through, so it did supply
            -- carries and a value for the back-edge test.
            match l.cont.mapM (get Γ2) with
            | none => .stuck "loop carry is not in scope"
            | some next =>
                match testAt Γ2 l.flag with
                | none => .stuck "loop condition is not in scope"
                | some c =>
                    if c == l.contOnTrue then
                      dtrip fuel cfg Γ w2 l body n0 afterBody next false
                    else leave Γ2 next w2

def runCode : Nat → Cfg → Env → World → List Piece → CodeRes
  | 0, _, _, _, _ => .stuck "step budget exhausted"
  | _ + 1, _, Γ, w, [] => .ok Γ w
  | fuel + 1, cfg, Γ, w, p :: ps =>
      match runPiece fuel cfg Γ w p with
      | .stuck s => .stuck s
      | .misuse s => .misuse s
      | .fault s => .fault s
      | .brk d Γb vs w' => .brk d Γb vs w'
      | .cont d vs w' => .cont d vs w'
      | .ok Γ' w' => runCode fuel cfg Γ' w' ps
end

/-- **`c`, run from `Γ`, spends `binds` slots and answers `v`, touching nothing.**

    What a claim about a straight-line fragment has to say, with the plumbing
    said once rather than at every claim: the fuel, the configuration and the
    world are quantified here, the run succeeds, the world it leaves is the one
    it started with --- so the fragment performs no store and no call --- and the
    last of the `binds` values it appended is `v`.

    `fuel + 2` rather than `fuel`, because `runCode (fuel + 1)` calls
    `runPiece fuel` and a bare variable leaves the successor equation unable to
    fire; putting it here keeps every claim from having to get it right.

    Deliberately over a `Code` rather than over a `Prog`: `Prog.emit` stays at
    the call site, where it is the reason the claim is about emitted
    instructions and not about a transcription of them. -/
def Answers (Γ : Env) (c : Code) (binds : Nat) (v : V) : Prop :=
  ∀ (fuel : Nat) (cfg : Cfg) (w : World), ∃ Γ',
    runCode (fuel + 2) cfg Γ w c = .ok Γ' w
    ∧ Γ'.size = Γ.size + binds
    ∧ Γ'.back? = some v

/-- Run a body with `params` bound to `args`, and hand back the world it
    leaves, with the observation trace in program order. -/
def run (cfg : Cfg) (args : List V) (w : World) (c : Code) : Outcome (List Obs) :=
  match runCode cfg.steps cfg args.toArray w c with
  | .stuck m => .stuck m
  | .misuse m => .misuse m
  | .fault m => .fault m
  | .brk d _ _ _ => .stuck s!"br {d} leaves the function body"
  | .cont d _ _ => .stuck s!"continue {d} leaves the function body"
  | .ok _ w' => .ok w'.obs.reverse w'


/-- `runStmt`'s `istore8` arm, phrased without the private `get` so proofs in
    other files can rewrite with it. Holds by definition. -/
theorem runStmt_istore8 (cfg : Cfg) (Γ : Env) (w : World) (v a : R) :
    runStmt cfg Γ w (.istore8 v a)
      = match Γ[v]?, Γ[a]? with
        | some (.sc _ b), some (.sc _ addr) =>
            match w.mem.store addr 1 (b &&& 0xff) with
            | some m => .ok Γ { obsStore w addr 1 (b &&& 0xff) with mem := m }
            | none => .fault s!"istore8 to unmapped address {addr}"
        | _, _ => .stuck "istore8 operand is not in scope" := rfl

/-- `runStmt`'s `storeUnaligned` arm, phrased without the private `get`. -/
theorem runStmt_storeUnaligned (cfg : Cfg) (Γ : Env) (w : World) (v a : R) :
    runStmt cfg Γ w (.storeUnaligned v a)
      = match Γ[v]?, Γ[a]? with
        | some (.sc t b), some (.sc _ addr) =>
            match w.mem.store addr (tyBytes t) b with
            | some m => .ok Γ { obsStore w addr (tyBytes t) b with mem := m }
            | none => .fault s!"store to unmapped address {addr}"
        | some (.vec t ls), some (.sc _ addr) =>
            let lw := ((t.lanes.map (·.1.width)).getD 8) / 8
            match (ls.zipIdx.foldlM
                (fun mm (x, i) => mm.store (addr + UInt64.ofNat (i * lw)) lw x) w.mem) with
            | some m => .ok Γ { obsStore w addr (tyBytes t) 0 with mem := m }
            | none => .fault s!"vector store to unmapped address {addr}"
        | _, _ => .stuck "store operand is not in scope" := rfl

/-- `runStmt`'s typed-`store` arm, phrased without the private `get`. -/
theorem runStmt_store (cfg : Cfg) (Γ : Env) (w : World) (ty : ClifTy) (v a : R) :
    runStmt cfg Γ w (.store ty v a)
      = match Γ[v]?, Γ[a]? with
        | some val, some (.sc _ addr) =>
            let n := tyBytes ty
            match val with
            | .sc _ b =>
                match w.mem.store addr n b with
                | some m => .ok Γ { obsStore w addr n b with mem := m }
                | none => .fault s!"store to unmapped address {addr}"
            | .vec t ls =>
                let lw := ((t.lanes.map (·.1.width)).getD 8) / 8
                match (ls.zipIdx.foldlM
                    (fun mm (x, i) => mm.store (addr + UInt64.ofNat (i * lw)) lw x) w.mem) with
                | some m => .ok Γ { obsStore w addr n 0 with mem := m }
                | none => .fault s!"vector store to unmapped address {addr}"
        | _, _ => .stuck "store operand is not in scope" := rfl

/-- `runStmt`'s `call` arm, phrased without the private `get`. -/
theorem runStmt_call (cfg : Cfg) (Γ : Env) (w : World) (c : Callee) (args : List R) :
    runStmt cfg Γ w (.call c args)
      = match args.mapM (fun r => Γ[r]?) with
        | none => .stuck "a call argument is not in scope"
        | some vs =>
            match callOf cfg.locals c vs (obsCall w c vs) with
            | none => failOf cfg.locals c vs (obsCall w c vs)
            | some (res, w') =>
                match res with
                | some v => .ok (Γ.push v) w'
                | none => .stuck "the call returned nothing to bind" := rfl

/-- `runStmt`'s `callVoid` arm, phrased without the private `get`. -/
theorem runStmt_callVoid (cfg : Cfg) (Γ : Env) (w : World) (c : Callee) (args : List R) :
    runStmt cfg Γ w (.callVoid c args)
      = match args.mapM (fun r => Γ[r]?) with
        | none => .stuck "a call argument is not in scope"
        | some vs =>
            match callOf cfg.locals c vs (obsCall w c vs) with
            | none => failOf cfg.locals c vs (obsCall w c vs)
            | some (_, w') => .ok Γ w' := rfl

end AlgorithmLib.HProg.Sem
