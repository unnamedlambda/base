module
public import AlgorithmLib.Surface.FFI
meta import AlgorithmLib.Surface.FFI
public import AlgorithmLib.Surface.ProgCuda
meta import AlgorithmLib.Surface.ProgCuda
public import AlgorithmLib.Surface.CudaPipeline
meta import AlgorithmLib.Surface.CudaPipeline
public import AlgorithmLib.Vocab.WGSL
meta import AlgorithmLib.Vocab.WGSL
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# Everything needed to write a generator

The term surface, the layout and payload helpers, and the two
kernel languages — but not the LZ4 or ML developments, which are written *with*
this rather than needed to use it.

A generator that imports the whole library is rebuilt whenever either
development changes, and elaborates against their declarations every time it is
checked. Neither is free, and the generators that want none of it outnumber the
ones that do: importing this instead where it suffices took the work a full
build does from 2120s to 1453s.
-/
