import AlgorithmLib.Core.Artifact
import AlgorithmLib.Core.Bytes
import AlgorithmLib.Surface.Layout
import AlgorithmLib.Host.Clif
import AlgorithmLib.Host.HostIR
import AlgorithmLib.Core.IR
import AlgorithmLib.Host.Term
import AlgorithmLib.Host.Sem
import AlgorithmLib.Host.Blocks
import AlgorithmLib.Host.Frames
import AlgorithmLib.Host.Trust
import AlgorithmLib.Surface.FFI
import AlgorithmLib.Surface.ProgFFI
import AlgorithmLib.Surface.Prog
import AlgorithmLib.Surface.CudaPipeline
import AlgorithmLib.Vocab.PTX
import AlgorithmLib.Vocab.WGSL

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
