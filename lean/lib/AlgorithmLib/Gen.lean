module
public import AlgorithmLib.Core.Artifact
meta import AlgorithmLib.Core.Artifact
public import AlgorithmLib.Core.Bytes
meta import AlgorithmLib.Core.Bytes
public import AlgorithmLib.Surface.Layout
meta import AlgorithmLib.Surface.Layout
public import AlgorithmLib.Host.Clif
meta import AlgorithmLib.Host.Clif
public import AlgorithmLib.Host.HostIR
meta import AlgorithmLib.Host.HostIR
public import AlgorithmLib.Core.IR
meta import AlgorithmLib.Core.IR
public import AlgorithmLib.Host.Term
meta import AlgorithmLib.Host.Term
public import AlgorithmLib.Host.Sem
meta import AlgorithmLib.Host.Sem
public import AlgorithmLib.Host.Blocks
meta import AlgorithmLib.Host.Blocks
public import AlgorithmLib.Host.Frames
meta import AlgorithmLib.Host.Frames
public import AlgorithmLib.Host.Trust
meta import AlgorithmLib.Host.Trust
public import AlgorithmLib.Surface.FFI
meta import AlgorithmLib.Surface.FFI
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
public import AlgorithmLib.Surface.Prog
meta import AlgorithmLib.Surface.Prog
public import AlgorithmLib.Surface.CudaPipeline
meta import AlgorithmLib.Surface.CudaPipeline
public import AlgorithmLib.Vocab.PTX
meta import AlgorithmLib.Vocab.PTX
public import AlgorithmLib.Vocab.WGSL
meta import AlgorithmLib.Vocab.WGSL
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# Everything needed to write a generator

The term surface, the callee table, the layout and payload helpers, and the two
kernel languages — but not the LZ4 or ML developments, which are written *with*
this rather than needed to use it.

A generator that imports the whole library is rebuilt whenever either
development changes, and elaborates against their declarations every time it is
checked. Neither is free, and the generators that want none of it outnumber the
ones that do: importing this instead where it suffices took the work a full
build does from 2120s to 1453s.
-/
