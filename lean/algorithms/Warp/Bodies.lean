import Scan.MlSurface
import AlgorithmLib.ML.Kernel.BodyCheck
import Warp.Silu
import Warp.SumSq
import Warp.Gemv
import Warp.Mlp
import Warp.Grad
/-!
  # The Warp apps' kernels, launched

  Apart from the generators, which print the kernels and so import no proof:
  each claim here is that launching a shipped body kernel computes its spec,
  at the words the hardware computes on, from `BodyCheck.launch_body` and the
  carrier-generic store lemmas of the kernel it is an instance of.
-/

open Lean AlgorithmLib AlgorithmLib.IR AlgorithmLib.ML

namespace SiluWarp

open AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec
  AlgorithmLib.Kernel AlgorithmLib.ML.BodyProof AlgorithmLib.ML.BodyWarp AlgorithmLib.ML.BodyCheck in
/-- **And launching its kernel computes the spec, in every lane of every
    block**: `GRID` one-warp blocks over the input's and output's `N` words
    leave lane `l` of block `cta`'s output word as `silu` of its input word,
    at the words the hardware computes on. -/
theorem silu_body (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (bufs bufs' : Nat → ByteArray)
    (hsz : ∀ b, b < 2 → N * 4 ≤ (bufs b).size)
    (hr : LaunchRuns (Body.kernel "main" 2 0 kernel) GRID 32 (bufArgs 2) bufs bufs' hw) :
    ∀ cta (l : Lane), cta < GRID →
      ({ kmem bufs with bufs := bufs' } : Sem.Mem).load (bufW 2 1 (cta * 32 + l.val)) 4 =
        some (@denote F32 (F32.ops hw) 1 (fun _ => (ws0 2 (kmem bufs)).mem 0 (cta * 32 + l.val)) spec).bits := by
  intro cta l hc
  have hm : ((1 : Buf), ix.eval cta 0 l) ∈ touchL cta l 0 kernel :=
    List.mem_append_right _ (by simp [touchL])
  have h := launch_body hw hw32 "main" 2 0 kernel 1 GRID (fun _ => N * 4) (by native_decide) bufs bufs' hsz hr
    cta l 1 _ (by omega) hm
  have hv := @compileWKernel_storesAt 1 F32 (F32.ops hw) _ (litNatF32 hw) (fun _ => 0) 1 (fun _ => ix) spec ix 16
    cta 0 (fun _ _ => 0) (fun _ _ => 0) (ws0 2 (kmem bufs)) l
    (fun l' h' => by
      have : l' = l := Fin.ext (by simp only [ix, IdxE.eval] at h'; omega)
      rw [this])
  have e : ix.eval cta 0 l = cta * 32 + l.val := rfl
  have ek : kernel = compileWKernel (fun _ => 0) 1 (fun _ => ix) spec ix 16 := rfl
  rw [e] at h hv
  rw [h, ek, hv]
  simp only [e]

open AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec
  AlgorithmLib.Kernel AlgorithmLib.ML.BodyProof AlgorithmLib.ML.BodyWarp AlgorithmLib.ML.BodyCheck in
/-- The same for the looped kernel — the one that actually ships the
    bandwidth: `LGRID` one-warp blocks leave each of lane `l`'s `E` elements
    of block `cta`, `cta·32·E + 32·j + l`, as `silu` of its input word. -/
theorem silu_loop_body (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (bufs bufs' : Nat → ByteArray)
    (hsz : ∀ b, b < 2 → N * 4 ≤ (bufs b).size)
    (hr : LaunchRuns (Body.kernel "main" 2 0 lkernel) LGRID 32 (bufArgs 2) bufs bufs' hw) :
    ∀ cta j (l : Lane), cta < LGRID → j < E →
      ({ kmem bufs with bufs := bufs' } : Sem.Mem).load (bufW 2 1 (cta * (32 * E) + (j * 32 + l.val))) 4 =
        some (@denote F32 (F32.ops hw) 1
          (fun _ => (ws0 2 (kmem bufs)).mem 0 (cta * (32 * E) + (j * 32 + l.val))) spec).bits := by
  intro cta j l hc hj
  have hm : ((1 : Buf), (stride32 lbase).eval cta j l) ∈ touchL cta l 0 lkernel :=
    List.mem_flatMap.mpr ⟨j, List.mem_range.mpr hj, List.mem_append_right _ (by simp [touchL])⟩
  have h := launch_body hw hw32 "main" 2 0 lkernel 1 LGRID (fun _ => N * 4) (by native_decide) bufs bufs' hsz hr
    cta l 1 _ (by omega) hm
  have hv := @mapLoopEW_stores 1 F32 (F32.ops hw) _ (litNatF32 hw) spec (fun _ => 0) 1 lbase (by decide) E cta
    (fun _ => (by decide : (0 : Buf) ≠ 1)) (fun _ _ => 0) (fun _ _ => 0) (ws0 2 (kmem bufs)) j l hj
  have e : (stride32 lbase).eval cta j l = cta * (32 * E) + (j * 32 + l.val) := rfl
  rw [e] at h hv
  rw [h]
  exact congrArg (fun x => some (F32.bits x)) hv

end SiluWarp

namespace WarpSumSq

open AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec AlgorithmLib.Kernel
  AlgorithmLib.ML.BodyProof AlgorithmLib.ML.BodyWarp AlgorithmLib.ML.BodyCheck in
/-- **Launching the kernel that ships computes the user's spec.**

    Read the chain right to left: `warpSumSqV4E` is an ordinary spec-language
    term; `warpSumSqV4Store_implements` says the warp kernel's *store* lands
    exactly its denotation, at any carrier; `eKernel_elab` says the emittable
    kernel expands to that warp kernel; and `launch_body` says launching its
    body kernel leaves each block's output word as the warp machine's run
    does, at the words the hardware computes on (`F32`).

    Exact equality of words, for every block, where `be` denotes the input's
    words — this kernel contains no `exp`, so `hw`'s approximations do not
    touch it. -/
theorem sumsq_body (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (bufs bufs' : Nat → ByteArray)
    (hsz : ∀ b, b < 2 → [N * 4, GRID * 4].getD b 0 ≤ (bufs b).size)
    (hr : LaunchRuns (Body.kernel "main" 2 0 eKernel) GRID 32 (bufArgs 2) bufs bufs' hw)
    {Γ : Nat} (env : Fin Γ → F32) (be : Nat → Expr Γ)
    (hb : ∀ i, @denote F32 (F32.ops hw) Γ env (be i) = (ws0 2 (kmem bufs)).mem inBuf i) :
    ∀ cta, cta < GRID → ({ kmem bufs with bufs := bufs' } : Sem.Mem).load (bufW 2 outBuf cta) 4 =
      some (@denote F32 (F32.ops hw) Γ env (warpSumSqV4E (cta * CHUNK) be K)).bits := by
  intro cta hc
  have hm : (outBuf, cta) ∈ touchL cta lane0 0 eKernel :=
    List.mem_append_right _ (by simp [touchL, outBuf, IdxE.eval])
  rw [launch_body hw hw32 "main" 2 0 eKernel 1 GRID (fun b => [N * 4, GRID * 4].getD b 0) (by native_decide)
    bufs bufs' hsz hr cta lane0 outBuf cta (by omega) hm]
  have e : eKernel.elabAt cta 0 = warpSumSqV4Store (cta * CHUNK) inBuf outBuf cta K := eKernel_elab cta
  rw [e, @warpSumSqV4Store_implements F32 (F32.ops hw) _ Γ (litNatF32 hw) (cta * CHUNK) inBuf outBuf cta K
    (ws0 2 (kmem bufs)) env be hb]

end WarpSumSq

namespace GemvWarp

/-- The buffers' sizes in bytes: the matrix, the vector, the output. -/
def szOf (sh : Shape) (b : Nat) : Nat := [sh.m * sh.n * 4, sh.n * 4, sh.m * 4].getD b 0

/-- Every shipped kernel's launch obligations, decided: the six of each shape,
    over one warp per output row. -/
theorem gemv_launchOk : shapes.all (fun sh => [false, true].all fun sq => Sched.all.all fun s =>
    BodyCheck.launchOk 3 sh.m 1 (szOf sh) (if sq then sumsqOf sh s else kernelOf sh s)) = true := by
  native_decide

open AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec
  AlgorithmLib.Kernel AlgorithmLib.ML.BodyProof AlgorithmLib.ML.BodyWarp AlgorithmLib.ML.BodyCheck in
/-- **Launching any shipped kernel runs its statement**: each of a shape's six
    kernels, over one one-warp block per output row and buffers of the sizes
    `app` allocates, leaves every word a lane touches as the warp machine's
    run of the statement for its block does. -/
theorem gemv_runs (sh : Shape) (hsh : sh ∈ shapes) (sq : Bool) (s : Sched) (hw : Hw)
    (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (bufs bufs' : Nat → ByteArray)
    (hsz : ∀ b, b < 3 → szOf sh b ≤ (bufs b).size)
    (hr : LaunchRuns (Body.kernel "main" 3 0 (if sq then sumsqOf sh s else kernelOf sh s)) sh.m 32 (bufArgs 3) bufs bufs' hw) :
    ∀ cta (l : Lane) b j, cta < sh.m → (b, j) ∈ touchL cta l 0 (if sq then sumsqOf sh s else kernelOf sh s) →
      ({ kmem bufs with bufs := bufs' } : Sem.Mem).load (bufW 3 b j) 4 =
        some ((@WStmt.run F32 (F32.ops hw) _ ((if sq then sumsqOf sh s else kernelOf sh s).elabAt cta 0)
          (ws0 3 (kmem bufs))).mem b j).bits := by
  have hok := List.all_eq_true.mp (List.all_eq_true.mp (List.all_eq_true.mp gemv_launchOk sh hsh) sq
    (by cases sq <;> simp)) s (by cases s <;> simp [Sched.all])
  have h := launch_body hw hw32 "main" 3 0 _ 1 sh.m (szOf sh) hok bufs bufs' hsz hr
  intro cta l b j hc hm
  exact h cta l b j (by omega) hm

open AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec
  AlgorithmLib.Kernel AlgorithmLib.ML.BodyProof AlgorithmLib.ML.BodyWarp AlgorithmLib.ML.BodyCheck in
/-- **And launching the quad kernel computes the spec**, at the words the
    hardware computes on: each row's output word is the spec's denotation
    where `ae` and `be` denote the matrix's and the vector's words. -/
theorem gemv_body (sh : Shape) (hsh : sh ∈ shapes) (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    (bufs bufs' : Nat → ByteArray) (hsz : ∀ b, b < 3 → szOf sh b ≤ (bufs b).size)
    (hr : LaunchRuns (Body.kernel "main" 3 0 (kernel sh)) sh.m 32 (bufArgs 3) bufs bufs' hw)
    {Γ : Nat} (env : Fin Γ → F32) (ae be : Nat → Expr Γ)
    (ha : ∀ i, @denote F32 (F32.ops hw) Γ env (ae i) = (ws0 3 (kmem bufs)).mem aB i)
    (hb : ∀ i, @denote F32 (F32.ops hw) Γ env (be i) = (ws0 3 (kmem bufs)).mem xB i) :
    ∀ cta, cta < sh.m → ({ kmem bufs with bufs := bufs' } : Sem.Mem).load (bufW 3 yB cta) 4 =
      some (@denote F32 (F32.ops hw) Γ env (warpDotV4E ae be
        (fun i l => (aIx sh).eval cta i l) (fun i l => xIx.eval cta i l) (sh.n / 128))).bits := by
  intro cta hc
  have hm : (yB, cta) ∈ touchL cta lane0 0 (kernel sh) :=
    List.mem_append_right _ (by simp [touchL, yB, IdxE.eval])
  rw [gemv_runs sh hsh false .vec4 hw hw32 bufs bufs' hsz hr cta lane0 yB cta hc hm]
  have e : (kernel sh).elabAt cta 0 = (warpDotV4 aB xB (aIx sh) xIx yB .ctaId (sh.n / 128)).elabIn cta := rfl
  have h := @warpDotV4_implements F32 (F32.ops hw) _ Γ (litNatF32 hw) aB xB (aIx sh) xIx yB .ctaId (sh.n / 128)
    cta (ws0 3 (kmem bufs)) env ae be ha hb
  simp only [IdxE.eval] at h
  show some ((@WStmt.run F32 (F32.ops hw) _ ((kernel sh).elabAt cta 0) (ws0 3 (kmem bufs))).mem yB cta).bits = _
  rw [e, h]

end GemvWarp

namespace MlpWarp

open AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec
  AlgorithmLib.Kernel AlgorithmLib.ML.BodyProof AlgorithmLib.ML.BodyWarp AlgorithmLib.ML.BodyCheck in
/-- **Launching the kernel computes the model, in every lane of every block.**

    `GRID` one-warp blocks over the inputs' and outputs' allocations leave lane
    `l` of block `cta`'s output word as the three layers' denotation of that
    lane's `D` inputs, at the words the hardware computes on — `exp` (inside
    SiLU) being the hardware's `ex2` of `x·log₂e`, as `F32` computes it.
    Measured cost of that against a CPU evaluation of the same three layers:
    3.27e-7 max relative error. -/
theorem mlp_body (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32) (bufs bufs' : Nat → ByteArray)
    (hsz : ∀ b, b < 2 → [NIN * 4, NOUT * 4].getD b 0 ≤ (bufs b).size)
    (hr : LaunchRuns (Body.kernel "main" 2 0 kernel) GRID 32 (bufArgs 2) bufs bufs' hw) :
    ∀ cta (l : Lane), cta < GRID →
      ({ kmem bufs with bufs := bufs' } : Sem.Mem).load (bufW 2 outBuf (cta * 32 + l.val)) 4 =
        some (@denote F32 (F32.ops hw) D
          (fun i => (ws0 2 (kmem bufs)).mem inBuf ((inIx i).eval cta 0 l)) spec).bits := by
  intro cta l hc
  have hm : (outBuf, outIx.eval cta 0 l) ∈ touchL cta l 0 kernel :=
    List.mem_append_right _ (by simp [touchL])
  have h := launch_body hw hw32 "main" 2 0 kernel 1 GRID (fun b => [NIN * 4, NOUT * 4].getD b 0)
    (by native_decide) bufs bufs' hsz hr cta l outBuf _ (by omega) hm
  have hv := @compileWKernel_storesAt D F32 (F32.ops hw) _ (litNatF32 hw) (fun _ => inBuf) outBuf inIx spec
    outIx (D + slots spec + 1) cta 0 (fun _ _ => 0) (fun _ _ => 0) (ws0 2 (kmem bufs)) l
    (fun l' h' => by
      have : l' = l := Fin.ext (by simp only [outIx, IdxE.eval] at h'; omega)
      rw [this])
  have e : outIx.eval cta 0 l = cta * 32 + l.val := rfl
  have ek : kernel = compileWKernel (fun _ => inBuf) outBuf inIx spec outIx (D + slots spec + 1) := rfl
  rw [e] at h hv
  rw [h, ek, hv]

end MlpWarp

namespace GradWarp

open AlgorithmLib.Prog AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.Hoare AlgorithmLib.LibSpec
  AlgorithmLib.Kernel AlgorithmLib.ML.BodyProof AlgorithmLib.ML.BodyWarp AlgorithmLib.ML.BodyCheck in
/-- **Launching either gradient kernel runs its statement**, the full and the
    narrowed one: `GRID` one-warp blocks over the inputs' and outputs'
    allocations leave every word a lane touches as the warp machine's run of
    the statement for its block does, at the words the hardware computes on.
    What the statement stores is the derivative (`grad_stores`,
    `grad_outputs_are_derivatives`). -/
theorem grad_runs (narrow : Bool) (hw : Hw) (hw32 : ∀ c x, (hw.approx c x).toNat < 2 ^ 32)
    (bufs bufs' : Nat → ByteArray) (hsz : ∀ b, b < 2 → [NIN * 4, NOUT * 4].getD b 0 ≤ (bufs b).size)
    (hr : LaunchRuns (Body.kernel "main" 2 0 (if narrow then kernelD else kernel)) GRID 32 (bufArgs 2) bufs bufs' hw) :
    ∀ cta (l : Lane) b j, cta < GRID → (b, j) ∈ touchL cta l 0 (if narrow then kernelD else kernel) →
      ({ kmem bufs with bufs := bufs' } : Sem.Mem).load (bufW 2 b j) 4 =
        some ((@WStmt.run F32 (F32.ops hw) _ ((if narrow then kernelD else kernel).elabAt cta 0)
          (ws0 2 (kmem bufs))).mem b j).bits := by
  have hok : launchOk 2 GRID 1 (fun b => [NIN * 4, NOUT * 4].getD b 0) (if narrow then kernelD else kernel) = true := by
    cases narrow <;> native_decide
  have h := launch_body hw hw32 "main" 2 0 _ 1 GRID _ hok bufs bufs' hsz hr
  intro cta l b j hc hm
  exact h cta l b j (by omega) hm

end GradWarp

/-- The claims this module makes. -/
def warpBodyRoots : List Name :=
  [ `SiluWarp.silu_body, `SiluWarp.silu_loop_body, `WarpSumSq.sumsq_body
  , `GemvWarp.gemv_runs, `GemvWarp.gemv_body, `MlpWarp.mlp_body, `GradWarp.grad_runs ]

open TrustScan in
#eval runScan "warp-bodies" warpBodyRoots warpBodyRoots
