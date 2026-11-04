module
public import AlgorithmLib.Surface.Wg
meta import AlgorithmLib.Surface.Wg
public import AlgorithmLib.Host.Lifecycle
meta import AlgorithmLib.Host.Lifecycle
import all Init.Data.Repr
@[expose] public section

/-!
# What a program in wgpu's standard layout starts from

The typestate a theorem about a program in `Wg.App`'s layout states, in
wgpu's own terms: wgpu finds an adapter and the program has made nothing yet,
the entry points' names are in the arena, each pipeline's shader and the
window's blit shader are WGSL the device accepts — what the theorem states of
its platform — and the window's title is a C string within its field.
-/

namespace AlgorithmLib.Wg.App

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Contracts (Fact TState)

/-- What a program starts from. -/
def loadTS (a : App) : TState :=
  [.part .gpuAdapter true, .part .idle true, .room .arena a.memSize, .nwg 0,
   .wg1 .commandEncoder none, .wg1 .computePass none, .wg1 .commandBuffer none,
   .wgText (arenaAt a.NAME) 4 false] ++
  (a.pipes.map fun p => .wgText (arenaAt p.shaderOff) p.shaderLen true) ++
  match a.blit with
  | none => []
  | some b =>
      [.wg1 .texture none, .wg1 .textureView none, .wg1 .renderPass none,
       .wgText (arenaAt a.VS) 7 false, .wgText (arenaAt a.FS) 7 false, .wgText (arenaAt b.shaderOff) b.shaderLen true,
       .cstrIn (arenaAt b.titleOff) b.titleCap]

end AlgorithmLib.Wg.App
