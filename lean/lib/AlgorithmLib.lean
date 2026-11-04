module
public import AlgorithmLib.Gen
meta import AlgorithmLib.Gen
public import AlgorithmLib.LZ4
meta import AlgorithmLib.LZ4
public import AlgorithmLib.ML
meta import AlgorithmLib.ML
public import AlgorithmLib.MLHost.TapeRun
meta import AlgorithmLib.MLHost.TapeRun
public import AlgorithmLib.Sound.PtxLaunch
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# Everything

Both developments, the generator toolkit, and what is proved of the kernel
compiler (`Sound`). Prefer `AlgorithmLib.ML` or `AlgorithmLib.LZ4`: what a file
imports decides what a rebuild costs it.
-/
