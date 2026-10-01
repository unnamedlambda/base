module
public import AlgorithmLib.Gen
meta import AlgorithmLib.Gen
public import AlgorithmLib.LZ4
meta import AlgorithmLib.LZ4
public import AlgorithmLib.ML
meta import AlgorithmLib.ML
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# Everything

Both developments and the generator toolkit. Prefer `AlgorithmLib.ML` or
`AlgorithmLib.LZ4Suite`: what a file imports decides what a rebuild costs it.
-/
