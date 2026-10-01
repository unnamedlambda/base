import Ship.CpuBenchStart

/-!
# The CPU benchmark's entries never misuse a call

Each `<name>_clif` entry is its kernel in CLIF and calls nothing; `asm_load`
asks whether the CPU has AVX, by the name in the arena, and maps each body.
The `<name>_asm` entries call a mapped body when there is one: machine code,
which the model does not read, so a run that reaches it is stuck, not proven.
Their CLIF fallback is the `_clif` entry's kernel, proven here, the
polynomial's and the copy's in `CpuBenchPoly` and `CpuBenchStream` beside it. `asm_load` and the histogram and
Mandelbrot kernels' entries neither misuse a call nor fault, where the
caller's data and output hold the lengths it passes.

A plain file, not a module: under the module system the generator's
arithmetic on 64-bit constants unfolds one step at a time.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace CpuBenchSafe

abbrev Load (dl ol : UInt64) : World → Prop := Contracts.TState.holds
  [.part .frozen false, .room .arena 1536,
   .cstrIn (regionBase .arena + UInt64.ofNat CpuBench.NAME_OFF) 4, .roomArg .data dl, .roomArg .out ol]

/-- The layout as numbers. The assembler computes it, and the kernel would run
    the assembler to check it; the compiled one is asked instead. -/
theorem mem_size : CpuBench.MEM_SIZE = 1536 := by native_decide
theorem loads_eq : CpuBench.loads = [(704, 656, 0), (1408, 119, 1)] := by native_decide

theorem asmLoad_fine : Fine (emitGo (CpuBench.asmLoad : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem asmLoad_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World)
    (h : Contracts.TState.holds [.part .frozen false, .room .arena CpuBench.MEM_SIZE,
      .cstrIn (regionBase .arena + UInt64.ofNat CpuBench.NAME_OFF) 4, .roomArg .data dataLen,
      .roomArg .out outLen] w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (CpuBench.asmLoad : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (CpuBench.asmLoad : Prog Slot Lvl Unit)) ≠ .fault m := by
  rw [mem_size] at h
  refine sound_of_wp_entry (Load dataLen outLen) (runArgs dataLen outLen) ?_ asmLoad_fine rfl h m
  simp only [CpuBench.asmLoad, loads_eq]
  prog_vc (Load dataLen outLen)

theorem hist_fine : Fine (emitGo (do CpuBench.answer CpuBench.outHist (CpuBench.histUnr8 (← CpuBench.array 0)) : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem hist_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run dataLen outLen w)
    (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (do CpuBench.answer CpuBench.outHist (CpuBench.histUnr8 (← CpuBench.array 0)) : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (do CpuBench.answer CpuBench.outHist (CpuBench.histUnr8 (← CpuBench.array 0)) : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run dataLen outLen) (runArgs dataLen outLen) (by prog_vc (Run dataLen outLen)) hist_fine rfl h m

theorem mandel_fine : Fine (emitGo (do CpuBench.answer CpuBench.outAnswer (CpuBench.mandelUnr 8 (← CpuBench.nothing)) : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem mandel_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Run dataLen outLen w)
    (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (do CpuBench.answer CpuBench.outAnswer (CpuBench.mandelUnr 8 (← CpuBench.nothing)) : Prog Slot Lvl Unit)) ≠ .misuse m ∧
    Sem.run cfg (runArgs dataLen outLen) w (emit (do CpuBench.answer CpuBench.outAnswer (CpuBench.mandelUnr 8 (← CpuBench.nothing)) : Prog Slot Lvl Unit)) ≠ .fault m :=
  sound_of_wp_entry (Run dataLen outLen) (runArgs dataLen outLen) (by prog_vc (Run dataLen outLen)) mandel_fine rfl h m

theorem chase_fine : Fine (emitGo (do CpuBench.answer CpuBench.outAnswer (CpuBench.chaseLoop (← CpuBench.array 2)) : Prog Slot Lvl Unit) ⟨5, 0, [], [], [], none⟩).2 :=
  ⟨by decide +kernel, by decide +kernel⟩

set_option maxRecDepth 200000 in
set_option maxHeartbeats 1000000 in
theorem chase_no_misuse (cfg : Cfg) (dataLen outLen : UInt64) (w : World) (h : Any w) (m : String) :
    Sem.run cfg (runArgs dataLen outLen) w (emit (do CpuBench.answer CpuBench.outAnswer (CpuBench.chaseLoop (← CpuBench.array 2)) : Prog Slot Lvl Unit)) ≠ .misuse m :=
  safe_of_wp_entry Any (runArgs dataLen outLen) (by prog_vc Any) chase_fine rfl h m

end CpuBenchSafe

#print axioms CpuBenchSafe.asmLoad_no_misuse
#print axioms CpuBenchSafe.hist_no_misuse
#print axioms CpuBenchSafe.mandel_no_misuse
#print axioms CpuBenchSafe.chase_no_misuse
