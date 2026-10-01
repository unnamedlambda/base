module
public import AlgorithmLib.Core.Artifact
meta import AlgorithmLib.Core.Artifact
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The runtime surface, as data

Every entry point the runtime exports, one constructor each. What a callee is —
its C symbol, its signature, how it resolves — are total functions of the
constructor, so a signature is written in exactly one place and two files
cannot describe the same symbol differently.

The constructors live in `ClifData`, beside the instructions, because a call
names one: `Callee.ffi` takes an `Ffi`, so an artifact calling something the
engine does not provide is unrepresentable rather than refused at load. What
travels is the `cname`; `all` fixes only the order `id` reports, which nothing
shipped depends on.

The executable contracts — what a call *does*, transcribed from
`base/src/ffi/` — are in `Host.Sem`; which memory a call may write is in
`Host.Frames`. This file is only who exists and how to call them.
-/

namespace AlgorithmLib.IR


namespace FFI

/-- A named group of entry points, so a body can be checked against the part of
    the table it uses. -/
inductive Bundle where
  | fileIO | gpu | window | lmdb | ht | math | thread | cuda | cublas | native
  deriving Repr, BEq

end FFI

/-- The index of the entry point every application emits as `u0:1`, with `u0:0`
    reserved as a no-op stub. This is the number a host calls. -/
def mainFnIdx : UInt32 := u32 1

end AlgorithmLib.IR
