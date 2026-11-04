import Ship.MathSpec
import Ship.LibCheck

/-!
# The math functions are what the checker takes them to be

The checker reads a call into `Lib.Math` by `Check.MathOk`: handed singles,
`sinf`, `cosf` and `powf` return and leave the world as it was.
`mathOk_linked` is that statement of a program whose function table has the
library's functions at their places and is all code that calls only the C
library and itself: a call passes a single's bits at 32 bits (`normV_f32`), and
on those each function is `sin_spec`, `cos_spec` or `powf_spec`, which leave
the world.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem
open AlgorithmLib.HProg.Hoare AlgorithmLib.HProg.Contracts

namespace MathSpec
open AlgorithmLib.LibSpec AlgorithmLib.Check LibCheck

/-- A single as a call passes it: its low 32 bits. -/
theorem normV_f32 (x : UInt64) : normV (.sc .f32 x) = .sc .f32 (s32 x.toUInt32) := by
  simp only [normV, norm, widthMask, V.sc.injEq, true_and]
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_and, UInt32.toNat_toUInt64, UInt64.toNat_toUInt32]
  exact Nat.and_two_pow_sub_one_eq_mod _ 32

theorem s32_toUInt32 (b : UInt32) : (s32 b).toUInt32 = b := by simp [s32]

@[simp] theorem normV_s32 (b : UInt32) : normV (.sc .f32 (s32 b)) = .sc .f32 (s32 b) := by
  simp only [normV, norm, widthMask, s32_mask]

section linked
variable {fns : Nat → Option Fn} {env : FnEnv} {steps : Nat}

/-- **The math functions, in a linked program, are what the checker takes
    them to be**: where the program's function table has the library's
    functions at their places and every function of it calls only the C
    library and the program's own. -/
theorem mathOk_linked (hq : ∀ i f, fns i = some f → quietCode f.code = true)
    (hSin : fns (LibFn.libSlot .sinf) = some (implFn [.f32] (.status (LibMath.trig false))))
    (hCos : fns (LibFn.libSlot .cosf) = some (implFn [.f32] (.status (LibMath.trig true))))
    (hPow : fns (LibFn.libSlot .powf) = some (implFn [.f32, .f32] (.status LibMath.powf)))
    (k : Nat) : MathOk (termLocals fns env steps k) where
  sinf := fun w x hi => by
    refine (libCall_norm (vs' := [.sc .f32 (s32 x.toUInt32)]) (by simp only [List.map_cons, List.map_nil, normV_f32, s32_toUInt32])).mpr ?_
    refine libCall_of_callsA hq (by simp only [List.map_cons, List.map_nil, normV_s32]) (Ans.calls hSin (fun cfg => sin_spec cfg w x.toUInt32) k)
      fun v w' ⟨_, e⟩ _ => ?_
    subst e; exact libStep_refl (calm_of hi) 0 0
  cosf := fun w x hi => by
    refine (libCall_norm (vs' := [.sc .f32 (s32 x.toUInt32)]) (by simp only [List.map_cons, List.map_nil, normV_f32, s32_toUInt32])).mpr ?_
    refine libCall_of_callsA hq (by simp only [List.map_cons, List.map_nil, normV_s32]) (Ans.calls hCos (fun cfg => cos_spec cfg w x.toUInt32) k)
      fun v w' ⟨_, e⟩ _ => ?_
    subst e; exact libStep_refl (calm_of hi) 0 0
  powf := fun w x y hi => by
    refine (libCall_norm (vs' := [.sc .f32 (s32 x.toUInt32), .sc .f32 (s32 y.toUInt32)])
      (by simp only [List.map_cons, List.map_nil, normV_f32, s32_toUInt32])).mpr ?_
    refine libCall_of_callsA hq (by simp only [List.map_cons, List.map_nil, normV_s32])
      (Ans.calls hPow (fun cfg => powf_spec cfg w x.toUInt32 y.toUInt32) k) fun v w' ⟨_, e⟩ _ => ?_
    subst e; exact libStep_refl (calm_of hi) 0 0

end linked

end MathSpec
