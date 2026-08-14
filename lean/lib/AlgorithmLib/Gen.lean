import AlgorithmLib.Core
import AlgorithmLib.Bytes
import AlgorithmLib.Layout
import AlgorithmLib.Clif
import AlgorithmLib.HostIR
import AlgorithmLib.IR
import AlgorithmLib.HProg
import AlgorithmLib.HProgSem
import AlgorithmLib.HProgBlocks
import AlgorithmLib.HProgFrames
import AlgorithmLib.HProgTrust
import AlgorithmLib.FFI
import AlgorithmLib.FFIStd
import AlgorithmLib.HProgFFI
import AlgorithmLib.CudaPipeline
import AlgorithmLib.PTX
import AlgorithmLib.WGSL

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
