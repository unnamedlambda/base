module
public import Lz4.OpLe64
meta import Lz4.OpLe64
public import Lz4.Stores
meta import Lz4.Stores
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

set_option maxRecDepth 1500

/-!
  # `Lz4.Stores` at the 64 KiB geometry

  Every command `Lz4.Stores` records (`at_both`), elaborated again with the 64 KiB
  geometry's numbers and names (`Lz4Both`): the proofs are the same text, and
  every `decide` is run again against the 64 KiB array.
-/

namespace Lz4Sites

open Lz4Ship
open AlgorithmLib.LZ4Simt

replay64 Lz4.Stores

end Lz4Sites
