module
public import AlgorithmLib.Surface.LibBase
meta import AlgorithmLib.Surface.LibBase
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lib.Thread` — worker threads, as CLIF over two shims

The engine's thread entry points (`threadInit`, `threadSpawn`, `threadJoin`,
`threadCleanup`), stated in `Host.Ffi` and `Sem.spawnWorker`, as functions of
the program itself. The engine keeps only what it alone can do — start a
thread running one of the program's functions (`threadStart`) and wait for it
(`threadFinish`) — and this keeps the rest: the handles a context has handed
out, from `1`, each finished once, and every one still running finished at
cleanup.
-/

namespace AlgorithmLib.LibThread

open AlgorithmLib.IR
open AlgorithmLib.Prog
open AlgorithmLib.Lib

variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

/-- The context: the next handle, and the running threads by handle less one
    (`0` once finished), as array and capacity. -/
def NEXT : Nat := 0x00
def ARR : Nat := 0x08
def CAP : Nat := 0x10
def SIZE : Nat := 0x18

def init : Body := do
  let (slot ::ᵥ .nil) ← entryParams [.i64]
  let s ← calloc (← iconst64 SIZE)
  when .ne s (← iconst64 0) (storeI64 (← iconst64 1) (← at_ s NEXT))
  storeI64 s slot

/-- Run function `fn` on `arg` on a new thread: its handle, or `-1`. -/
def spawn : StatusBody := do
  let (s ::ᵥ fn ::ᵥ arg ::ᵥ .nil) ← entryParams [.i64, .i64, .i64]
  failIf .eq s (← iconst64 0) (do
    let t ← ffi .threadStart %[fn, arg]
    failIf .eq t (← iconst64 (-1)) (do
      let h ← load64 (← at_ s NEXT)
      let i ← iaddImm h (-1)
      let capA ← at_ s CAP
      when .uge i (← load64 capA) (do
        let cap ← load64 capA
        let nc ← select (← icmp .eq cap (← iconst64 0)) (← iconst64 8) (← ishlImm cap 1)
        let na ← calloc (← ishlImm nc 3)
        let old ← load64 (← at_ s ARR)
        when .ne old (← iconst64 0) (do
          let _ ← memcpy na old (← ishlImm cap 3)
          free old)
        storeI64 na (← at_ s ARR)
        storeI64 nc capA)
      storeI64 t (← iadd (← load64 (← at_ s ARR)) (← ishlImm i 3))
      storeI64 (← iaddImm h 1) (← at_ s NEXT)
      pure h))

/-- Wait for the thread a handle names: `0`, or `-1` for a handle that names
    none running. The handle is read as `u32`. -/
def join : StatusBody := do
  let (s ::ᵥ h ::ᵥ .nil) ← entryParams [.i64, .i64]
  failIf .eq s (← iconst64 0) (do
    let i ← iaddImm (← band h (← iconst64 0xffffffff)) (-1)
    -- handle 0 is past every capacity, compared unsigned
    failIf .uge i (← load64 (← at_ s CAP)) (do
      let e ← iadd (← load64 (← at_ s ARR)) (← ishlImm i 3)
      let t ← load64 e
      failIf .eq t (← iconst64 0) (do
        storeI64 (← iconst64 0) e
        ffi .threadFinish %[t])))

/-- Finish every thread still running, then release the context. -/
def cleanup : Body := do
  let (slot ::ᵥ .nil) ← entryParams [.i64]
  let s ← load64 slot
  when .ne s (← iconst64 0) (do
    let arr ← load64 (← at_ s ARR)
    forLoop (← iaddImm (← load64 (← at_ s NEXT)) (-1)) fun i => do
      let t ← load64 (← iadd arr (← ishlImm i 3))
      when .ne t (← iconst64 0) (do let _ ← ffi .threadFinish %[t]; pure ())
    free arr
    free s)
  storeI64 (← iconst64 0) slot

/-- The function implementing a thread entry point. -/
def implOf : Ffi → Option Impl
  | .threadInit => some (.void init)
  | .threadSpawn => some (.status spawn)
  | .threadJoin => some (.status join)
  | .threadCleanup => some (.void cleanup)
  | _ => none

end AlgorithmLib.LibThread
