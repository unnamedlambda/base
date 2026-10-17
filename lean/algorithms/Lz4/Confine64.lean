module
public import Lz4.Extend
meta import Lz4.Extend
public import Lz4.Shape64
meta import Lz4.Shape64
public import Lz4.Stores64
meta import Lz4.Stores64
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

set_option maxRecDepth 8192

/-!
  # `RegConfined` at the 64 KiB geometry

  The body of the confinement proof is generic over `(p, S)` — see `Lz4Extend`'s
  `Generic` section.  The three theorems `Lz4.Extend` records (`at_both`), which
  instantiate it at the shipped kernel, are elaborated again here at `K16`,
  stride 65536, using the `Shape`/`Geo`/`Loads` instances of `Lz4Shape64` and
  `Lz4Sites`.  Nothing here is a new assumption: every premise is the same
  machine-checked bundle, decided against the 64 KiB array.
-/

namespace Lz4Sites

open Lz4Ship
open AlgorithmLib.LZ4Simt
open AlgorithmLib.LZ4SimtBits

replay64 Lz4.Extend

end Lz4Sites
