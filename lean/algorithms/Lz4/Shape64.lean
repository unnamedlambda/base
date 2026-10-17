module
public import Lz4.ExtShape
meta import Lz4.ExtShape
public import Lz4.Sites64
meta import Lz4.Sites64
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

set_option maxRecDepth 8192

/-!
# The 64 KiB geometry satisfies every shape premise of the 32 KiB proof

`Lz4Extend` states its program-shape facts as `Bool`-valued predicates over an
arbitrary program `p` and block stride `S`.  `Lz4.ExtShape` discharges them for
the shipped 32 KiB kernel and records each (`at_both`); here the same commands
are elaborated again for the 64 KiB kernel, `K16`, at `S = 65536`.

Each is one `decide`: the two kernels have the same 274 pcs and the same
opcodes, and differ only in immediates, so every check that the 32 KiB proof
performs is a check the 64 KiB kernel also passes — but that is a claim about
the emitted arrays, and this file is where it is *checked* rather than assumed.
-/

namespace Lz4Sites
open Lz4Ship
open AlgorithmLib.LZ4Simt
open AlgorithmLib.LZ4SimtBits

replay64 Lz4.ExtShape

end Lz4Sites
