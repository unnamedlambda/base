import AlgorithmLib.Proof.Typestate
import Demo.Draw
import Demo.Raytrace
import Demo.Fft
import Demo.Compress

/-!
# Draw, Raytrace, Fft and Compress never misuse a call

Each entry point, proven by the condition generator from the typestate it
starts in, for any lengths of data and output the caller hands over. A program
starts from its initial memory: the shader text it hands `gpuCreatePipeline`
is a NUL-terminated string inside the field the layout gives it (`cstrIn`),
which the calls before the pipeline leave alone. Buffer sizes need no facts:
an upload or download of the wrong size answers `-1`. None of the four
faults, Compress where the caller's data and output hold the lengths it passes:
every load and store finds the bytes it reaches, and every operation answers.
Fft's gather reads at a bit-reversed index, below the count because it reverses
as many low bits as the count's highest bit is high (`bitrev_index_lt`).

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace WgpuSafe

-- Draw

abbrev DrawMain : World → Prop := Contracts.TState.holds
  [.part .gpuAdapter true, .part .frozen false, .part .gpu false, .room .arena Draw.layoutMeta.totalSize,
   .cstrIn (regionBase .arena + UInt64.ofNat Draw.f.shader.offset) 8192]

theorem Draw_fine : Fine (emitGo (Draw.code : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem Draw_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : DrawMain w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (Draw.code : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (Draw.code : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry DrawMain (runArgs dataLen outLen) (by prog_vc DrawMain) Draw_fine rfl h m

-- Raytrace

abbrev RaytraceMain : World → Prop := Contracts.TState.holds
  [.part .gpuAdapter true, .part .frozen false, .part .gpu false, .room .arena (Raytrace.pixels_off + Raytrace.pixelBytes),
   .cstrIn (regionBase .arena + UInt64.ofNat Raytrace.shader_off) Raytrace.shaderRegionSize]

theorem Raytrace_fine : Fine (emitGo (Raytrace.code : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem Raytrace_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : RaytraceMain w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (Raytrace.code : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (Raytrace.code : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry RaytraceMain (runArgs dataLen outLen) (by prog_vc RaytraceMain) Raytrace_fine rfl h m

-- Fft

abbrev FftMain : World → Prop := Contracts.TState.holds
  [.part .gpuAdapter true, .part .frozen false, .part .gpu false, .room .arena (Fft.meta_off + Fft.metaSize),
   .cstrIn (regionBase .arena + UInt64.ofNat Fft.shader_off) Fft.shaderRegionSize]

theorem Fft_fine : Fine (emitGo (Fft.code : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem Fft_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : FftMain w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (Fft.code : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (Fft.code : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry FftMain (runArgs dataLen outLen) (by prog_vc FftMain) Fft_fine rfl h m

-- Compress, at the block size it ships with

abbrev CompressMain (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .gpuAdapter true, .part .frozen false, .part .gpu false,
   .room .arena (Compress.blockMeta_off 16384 + Compress.metaSize 16384),
   .cstrIn (regionBase .arena + UInt64.ofNat Compress.shader_off) Compress.shaderRegionSize,
   .roomArg .data dl, .roomArg .out ol]

theorem Compress_fine : Fine (emitGo (Compress.code 16384 : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem Compress_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : CompressMain dataLen outLen w)
    (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (Compress.code 16384 : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (Compress.code 16384 : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (CompressMain dataLen outLen) (runArgs dataLen outLen) (by prog_vc (CompressMain dataLen outLen))
    Compress_fine rfl h m

end WgpuSafe

#print axioms WgpuSafe.Draw_no_misuse
#print axioms WgpuSafe.Raytrace_no_misuse
#print axioms WgpuSafe.Fft_no_misuse
#print axioms WgpuSafe.Compress_no_misuse
