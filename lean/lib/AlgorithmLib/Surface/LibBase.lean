module
public import AlgorithmLib.Surface.Prog
meta import AlgorithmLib.Surface.Prog
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The libraries' shared parts

What every library written in CLIF over C calls uses: results that fail to
`-1`, the C library's memory functions, and what a library function is.
-/

namespace AlgorithmLib.Lib

open AlgorithmLib.IR
open AlgorithmLib.Prog

variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

def memcpy (dst src n : V .i64) : Prog V L (V .i64) := ext (.c .memcpy) %[dst, src, n]

def memset (dst : V .i64) (c : V .i32) (n : V .i64) : Prog V L (V .i64) :=
  ext (.c .memset) %[dst, c, n]

/-- The bytes before the first NUL at `p`. -/
def strlen (p : V .i64) : Prog V L (V .i64) := ext (.c .strlen) %[p]

def at_ (s : V .i64) (off : Nat) : Prog V L (V .i64) := iaddImm s off

/-- `-1` when `cc a b` holds, `k` otherwise. -/
def failIf {ty} (cc : ICmpCond) (a b : V ty) (k : Prog V L (V .i64)) : Prog V L (V .i64) := do
  let r ← ifte (jTys := [.i64]) cc a b (do pure %[← iconst64 (-1)]) (do pure %[← k])
  pure r.head

/-- `-1` unless the flag `c` is zero. -/
def failUnless0 (c : V .i8) (k : Prog V L (V .i64)) : Prog V L (V .i64) := do
  failIf .ne c (← iconst .i8 0) k

/-- `k` after a result code of `0`, `-1` after any other. -/
def onOk (rc : V .i32) (k : Prog V L (V .i64)) : Prog V L (V .i64) := do
  failIf .ne rc (← iconst32 0) k

/-- `0` for a result code of `0`, `-1` for any other. -/
def status (rc : V .i32) : Prog V L (V .i64) := onOk rc (iconst64 0)

/-- The next call's result code once `rc` is `0`; `rc` otherwise. -/
def andThen (rc : V .i32) (k : Prog V L (V .i32)) : Prog V L (V .i32) := do
  let r ← ifte (jTys := [.i32]) .eq rc (← iconst32 0) (do pure %[← k]) (pure %[rc])
  pure r.head

def or8 (a b : V .i8) : Prog V L (V .i8) := bor a b

/-- Whether any of `xs` is at most zero, as signed `i32`s. -/
def anyNonPos (xs : List (V .i32)) : Prog V L (V .i8) := do
  let z ← iconst32 0
  xs.foldlM (fun acc x => do or8 acc (← icmp .sle x z)) (← iconst .i8 0)

/-- Whether any of `xs` is negative. -/
def anyNeg64 (xs : List (V .i64)) : Prog V L (V .i8) := do
  let z ← iconst64 0
  xs.foldlM (fun acc x => do or8 acc (← icmp .slt x z)) (← iconst .i8 0)

def anyNeg32 (xs : List (V .i32)) : Prog V L (V .i8) := do
  let z ← iconst32 0
  xs.foldlM (fun acc x => do or8 acc (← icmp .slt x z)) (← iconst .i8 0)

/-- `n` zeroed bytes from the heap, `0` when there is no room. -/
def calloc (n : V .i64) : Prog V L (V .i64) := do ext (.c .calloc) %[n, ← iconst64 1]

def free (p : V .i64) : Prog V L Unit := ext (.c .free) %[p]

/-- A function of the library: what it answers, and its body. -/
inductive Impl where
  | void (b : Body)
  | status (b : StatusBody)

/-- The library function for `f`, compiled at `idx`. -/
def compileImpl (idx : Nat) (f : Ffi) : Impl → Except String FuncData
  | .void b => Prog.compileProg idx b f.params
  | .status b => Prog.compileProgStatus idx b f.params

end AlgorithmLib.Lib
