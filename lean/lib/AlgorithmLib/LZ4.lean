module
public import AlgorithmLib.LZ4.Confine
meta import AlgorithmLib.LZ4.Confine
public import AlgorithmLib.LZ4.SimtEmit
meta import AlgorithmLib.LZ4.SimtEmit
public import AlgorithmLib.LZ4.WarpEmit
meta import AlgorithmLib.LZ4.WarpEmit
public import AlgorithmLib.LZ4.WarpColl
meta import AlgorithmLib.LZ4.WarpColl
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The LZ4 development

Every module of the compressor and decompressor proof, so a generator that
ships LZ4 names one import rather than seventeen.
-/
