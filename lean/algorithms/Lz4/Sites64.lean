module
public import Lz4.Sites
meta import Lz4.Sites
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

/-!
  # `Lz4.Sites` at the 64 KiB geometry

  Every command `Lz4.Sites` records (`at_both`), elaborated again with the 64 KiB
  geometry's numbers and names (`Lz4Both`): the proofs are the same text, and
  every `decide` is run again against the 64 KiB array.  The kernel's own name, `K16`,
  is declared here first.
-/

namespace Lz4Sites

open Lz4Ship
open AlgorithmLib.LZ4Simt
open AlgorithmLib.LZ4SimtBits

/-- The 64 KiB kernel, as a reducible abbreviation — mirrors `K` at 32 KiB.
    Using the raw projection instead changes how `rw`/`whnf` unfold it. -/
abbrev K16 := (WP.mk 16).kernel

replay64 Lz4.Sites

end Lz4Sites
