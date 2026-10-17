module
public import Lz4.Ckpt64
meta import Lz4.Ckpt64
public import Lz4.OpLe
meta import Lz4.OpLe
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
  # `Lz4.OpLe` at the 64 KiB geometry

  Every command `Lz4.OpLe` records (`at_both`), elaborated again with the 64 KiB
  geometry's numbers and names (`Lz4Both`): the proofs are the same text, and
  every `decide` is run again against the 64 KiB array.
-/

namespace Lz4Sites

open Lz4Ship
open AlgorithmLib.LZ4Simt
open AlgorithmLib.LZ4SimtBits

replay64 Lz4.OpLe

end Lz4Sites
