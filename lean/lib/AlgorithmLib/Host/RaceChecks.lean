module
public import AlgorithmLib.Host.World
meta import AlgorithmLib.Host.World
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic

@[expose] public section

/-!
# The race tracker, checked on the cases that decide it

`Race` is what lets the model run every stream operation when it is issued:
conflicting accesses it cannot order are stuck, so the order it runs them in is
the only one the device could produce. A corpus cannot exercise this — a program
the interpreter is stuck on is one the generator refuses to ship — so these are
the tracker's own cases: an unordered conflict is refused, and each thing that
orders one (the same stream, a host sync, an event) lets it through.
-/

namespace AlgorithmLib.HProg.Sem

/-- Streams 0 and 1 as parties. -/
private def s0 : Nat := streamParty 0
private def s1 : Nat := streamParty 1

-- Two streams writing one buffer with nothing between them: a race.
#guard (({} : Race).op s0 [] [7] >>= (·.op s1 [] [7])).isNone

-- One stream writing twice: ordered by the stream.
#guard (({} : Race).op s0 [] [7] >>= (·.op s0 [] [7])).isSome

-- A read on one stream and a write on another: a race either way round.
#guard (({} : Race).op s0 [7] [] >>= (·.op s1 [] [7])).isNone
#guard (({} : Race).op s0 [] [7] >>= (·.op s1 [7] [])).isNone

-- Two readers never conflict.
#guard (({} : Race).op s0 [7] [] >>= (·.op s1 [7] [])).isSome

-- Different buffers never conflict.
#guard (({} : Race).op s0 [] [7] >>= (·.op s1 [] [8])).isSome

-- The host waits for stream 0, then issues on stream 1: ordered.
#guard (({} : Race).op s0 [] [7] >>= (fun r => (r.sync s0).op s1 [] [7])).isSome

-- An event: stream 0's write, then a point recorded on 0 that 1 waits for.
#guard (do
  let r ← ({} : Race).op s0 [] [7]
  let (r, clk) := r.issue s0
  let (r, _) := r.issue s1
  (r.joinInto s1 clk).op s1 [] [7]).isSome

-- The same wait on a point recorded *before* the write does not order it.
#guard (do
  let (r, clk) := ({} : Race).issue s0
  let r ← r.op s0 [] [7]
  let (r, _) := r.issue s1
  (r.joinInto s1 clk).op s1 [] [7]).isNone

-- The default stream and a created one are not ordered by default.
#guard (({} : Race).op defaultParty [] [7] >>= (·.op s0 [] [7])).isNone

end AlgorithmLib.HProg.Sem
