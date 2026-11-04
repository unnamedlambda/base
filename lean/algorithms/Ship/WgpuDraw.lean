import AlgorithmLib.Check.Entry
import AlgorithmLib.Proof.Typestate
import AlgorithmLib.Check.WgState
import Demo.Draw
import Demo.Raytrace
import Demo.Fft
import Demo.Compress

/-!
# Draw, Raytrace, Fft and Compress never misuse a call

Each entry point, proven by the condition generator from the typestate it
starts in, for any lengths of data and output the caller hands over: what
`Wg.App.loadTS` states of wgpu and of the program's memory, the shaders'
acceptance among it. None of the four faults, Fft where the caller's data holds the length it passes and
Compress where its data and output do:
every load and store finds the bytes it reaches, and every operation answers.
Fft's gather reads at a bit-reversed index, below the count because it reverses
as many low bits as the count's highest bit is high (`bitrev_index_lt`).

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace WgpuSafe

-- Draw

abbrev DrawMain : World → Prop := Contracts.TState.holds Draw.app.loadTS

ship_entry Draw : Draw.code under DrawMain

-- Raytrace

abbrev RaytraceMain : World → Prop := Contracts.TState.holds Raytrace.app.loadTS

ship_entry Raytrace : Raytrace.code under RaytraceMain

-- Fft

abbrev FftMain (dl : UInt64) : World → Prop := Contracts.TState.holds
  (Fft.app.loadTS ++ [.roomArg .data dl])

ship_entry Fft : Fft.code under FftMain dataLen

-- Compress, at the block size it ships with

abbrev CompressMain (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  ((Compress.app 16384 Compress.shaderLen16384).loadTS ++ [.roomArg .data dl, .roomArg .out ol])

ship_entry Compress : (Compress.code 16384 Compress.shaderLen16384) under CompressMain dataLen outLen

end WgpuSafe

#print axioms WgpuSafe.Draw_no_misuse
#print axioms WgpuSafe.Raytrace_no_misuse
#print axioms WgpuSafe.Fft_no_misuse
#print axioms WgpuSafe.Compress_no_misuse
