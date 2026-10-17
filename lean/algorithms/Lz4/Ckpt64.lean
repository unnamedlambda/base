module
public import AlgorithmLib.LZ4.Checkpoint
meta import AlgorithmLib.LZ4.Checkpoint
public import Lz4.Cursor64
meta import Lz4.Cursor64
public import Lz4.Splice64
meta import Lz4.Splice64
public import Lz4.Ckpt
meta import Lz4.Ckpt
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

/-!
  # `Lz4.Ckpt` at the 64 KiB geometry

  Every command `Lz4.Ckpt` records (`at_both`), elaborated again with the 64 KiB
  geometry's numbers and names (`Lz4Both`): the proofs are the same text, and
  every `decide` is run again against the 64 KiB array.
-/

namespace Lz4Sites

open Lz4Ship
open AlgorithmLib.LZ4Simt
open AlgorithmLib.LZ4SimtBits

replay64 Lz4.Ckpt

end Lz4Sites
