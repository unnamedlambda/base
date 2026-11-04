module
public import AlgorithmLib.Core.Artifact
meta import AlgorithmLib.Core.Artifact
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The entry point a host calls

The index every application emits its entry point at.
-/

namespace AlgorithmLib.IR

/-- The index of the entry point every application emits as `u0:1`, with `u0:0`
    reserved as a no-op stub. This is the number a host calls. -/
def mainFnIdx : UInt32 := u32 1

end AlgorithmLib.IR
