import AlgorithmLib.Check.Entry
import Ship.FallingSandStart

/-!
# FallingSand's game on a display

The game's `main`, proven by the condition generator from the typestate it
starts in; the two cases of the display oracle are proven in files of their
own so they build side by side.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace WgpuSafe

ship_entry FallingSand_main_display : FallingSand.mainBody under (FallingSandMainOn true)

end WgpuSafe
