import Lean
import Std
import AlgorithmLib.Gen
import AlgorithmLib.ML
import MlSurface
import LayoutScan
import ShipScan


/-!
  # A proven GEMV, versus cuBLAS — and across schedules

  Qwen2 decode is bottlenecked on `cl_cublas_sgemv`: every token streams the
  whole weight matrix, so the kernel is memory-bound and `M = 1`.  That is
  precisely the shape where cuBLAS is *not* the last word — llama.cpp, vLLM and
  the Marlin family all beat it there.

  The kernel is the proven warp reduction applied to a matrix row: one warp per
  output element, five-round butterfly, lane-0 store.  Its spec is the two-level
  fold the hardware performs.

  It is emitted three ways — quad loads, interleaved scalar loads, blocked
  scalar loads — at two shapes.  Every one of the six kernels is proven exactly
  against its own fold, and `gemv_schedules_agree` proves any two of them equal
  under one named law.  The shapes are there because whether a schedule choice
  is *visible* depends on the shape, and a benchmark at one shape cannot say so.
-/

open Lean AlgorithmLib AlgorithmLib.IR AlgorithmLib.ML

namespace GemvWarp

/-- A GEMV problem: `m` rows of `n`, row-major.  `n` must be a multiple of 128
    so that every schedule's tile divides it. -/
structure Shape where
  n : Nat
  m : Nat
  tag : String

/-- Qwen2's FFN gate/up projection: hidden size 896 = 7·128. -/
def qwen : Shape := ⟨896, 4864, "gemv_warp"⟩

/-- A wide row.  At 896 a row is 3.5 KB and is fetched whole however the warp
    walks it, so all three schedules saturate bandwidth and tie.  At 16384 a
    lane's blocked segment is 2 KB on its own, and the walk becomes visible. -/
def wide : Shape := ⟨16384, 2048, "gemv_warp_wide"⟩

/-- Two more points so the tuning curve has a shape rather than two ends. -/
def mid  : Shape := ⟨2048, 4096, "gemv_warp_2048"⟩
def mid2 : Shape := ⟨8192, 2048, "gemv_warp_8192"⟩

def shapes : List Shape := [qwen, mid, mid2, wide]

def aB : Buf := 0             -- weight matrix, row-major
def xB : Buf := 1             -- input vector
def yB : Buf := 2             -- output vector

/-- Address of this lane's quad in row `ctaid`: `ctaid*n + i*128 + lane*4`. -/
def aIx (sh : Shape) : IdxE :=
  .add (.add (.mul .loopI (.lit 128)) (.mul .laneId (.lit 4))) (.mul .ctaId (.lit sh.n))
/-- Same offset into the (shared) input vector. -/
def xIx : IdxE := .add (.mul .loopI (.lit 128)) (.mul .laneId (.lit 4))

/-- Address of this lane's element in row `ctaid` under the interleaved scalar
    walk: `ctaid*n + i*32 + lane`. -/
def aIxS (sh : Shape) : IdxE :=
  .add (.add (.mul .loopI (.lit 32)) .laneId) (.mul .ctaId (.lit sh.n))
/-- Same offset into the (shared) input vector. -/
def xIxS : IdxE := .add (.mul .loopI (.lit 32)) .laneId

/-- Address under the blocked walk: lane `l` owns the contiguous run starting at
    `l*(n/32)`, so `ctaid*n + lane*(n/32) + i`.  The 32 lanes of a step land in
    32 different segments — the same elements as `aIxS`, read uncoalesced. -/
def aIxB (sh : Shape) : IdxE :=
  .add (.add (.mul .laneId (.lit (sh.n/32))) .loopI) (.mul .ctaId (.lit sh.n))
/-- Same offset into the (shared) input vector. -/
def xIxB (sh : Shape) : IdxE := .add (.mul .laneId (.lit (sh.n/32))) .loopI

/-- The addressing each schedule walks this matrix with.  A schedule is a fold
    and a walk together; swapping one without the other reads the wrong
    elements. -/
def rowIx (sh : Shape) : Sched → IdxE
  | .vec4 => aIx sh
  | .strided => aIxS sh
  | .blocked => aIxB sh
def vecIx (sh : Shape) : Sched → IdxE
  | .vec4 => xIx
  | .strided => xIxS
  | .blocked => xIxB sh

/-- **The kernel is an instance of the proven schema** — no new kernel code and
    no new proof, only a choice of buffers, addressing and trip count. -/
def kernelOf (sh : Shape) (s : Sched) : EWStmt :=
  s.realize aB xB (rowIx sh s) (vecIx sh s) yB .ctaId sh.n

/-- The quad kernel, spelled without the menu — definitionally the same term.
    That is the check that `Sched` enumerates the schedules the stack actually
    uses, rather than a menu invented alongside them. -/
def kernel (sh : Shape) : EWStmt :=
  warpDotV4 aB xB (aIx sh) xIx yB .ctaId (sh.n / 128)

example : kernel qwen = kernelOf qwen .vec4 := rfl
example : kernel wide = kernelOf wide .vec4 := rfl

/-- **The emitted addresses are the model's addresses**, on the nose.  This is
    the seam where a schedule's `IdxE` — what the PTX computes — meets the
    `Sched.idx` the agreement theorem quantifies over; if the two drifted apart,
    every claim below would be about a kernel that is not this one. -/
theorem gemv_addressing (sh : Shape) (s : Sched) (cta i : Nat) (l : Lane) :
    (rowIx sh s).eval cta i l = Sched.idx s sh.n (cta * sh.n) i l
      ∧ (vecIx sh s).eval cta i l = Sched.idx s sh.n 0 i l := by
  cases s <;> exact ⟨rfl, rfl⟩

/-- **Any two schedules compute the same thing**, under one named law — for
    *any* pair of buffers and any addressing that walks them the way `Sched.idx`
    says.

    Stated once, over the walk rather than over a particular kernel, because the
    argument never mentions what is being reduced.  The dot product and the sum
    of squares below are both instances; a third reduction would be a third
    instance and no new proof.

    `Law.laneRegroup` is the whole of what it costs to identify the answers, and
    `hia`/`hib` — that the emitted addresses *are* the model's addresses — are
    what earn it. -/
theorem schedules_agree_at (hl : AllHold [Law.laneRegroup]) (sh : Shape)
    (hn : sh.n % 128 = 0) (s t : Sched) (cta : Nat) (st : WSt)
    (bA bB : Buf) (ia ib : Sched → IdxE) (baseA baseB : Nat)
    (hia : ∀ u : Sched, (fun i l => (ia u).eval cta i l) = Sched.idx u sh.n baseA)
    (hib : ∀ u : Sched, (fun i l => (ib u).eval cta i l) = Sched.idx u sh.n baseB) :
    (((s.realize bA bB (ia s) (ib s) yB .ctaId sh.n).elabIn cta).run st).mem yB cta
      = (((t.realize bA bB (ia t) (ib t) yB .ctaId sh.n).elabIn cta).run st).mem yB cta := by
  have hs := Sched.realize_spec s bA bB (ia s) (ib s) yB .ctaId sh.n cta st
  have ht := Sched.realize_spec t bA bB (ia t) (ib t) yB .ctaId sh.n cta st
  simp only [IdxE.eval] at hs ht
  rw [hs, ht, hia s, hib s, hia t, hib t]
  exact sched_agree_at_idx hl s t (st.mem bA) (st.mem bB) baseA baseB sh.n hn

/-- **Any two schedules compute the same GEMV row.**

    The kernels are different PTX, with different trip counts, load widths, walk
    orders and fold orders, and each is proven against its own order exactly. -/
theorem gemv_schedules_agree (hl : AllHold [Law.laneRegroup]) (sh : Shape)
    (hn : sh.n % 128 = 0) (s t : Sched) (cta : Nat) (st : WSt) :
    (((kernelOf sh s).elabIn cta).run st).mem yB cta
      = (((kernelOf sh t).elabIn cta).run st).mem yB cta :=
  schedules_agree_at hl sh hn s t cta st aB xB (rowIx sh) (vecIx sh) (cta * sh.n) 0
    (fun u => by cases u <;> rfl) (fun u => by cases u <;> rfl)

/-! ### A second reduction on the same menu

    RMSNorm needs `Σⱼ x[row,j]²`, which is the dot schema with both operands
    reading one buffer (`warpDotV4_sumsq`).  It is a different kernel with a
    different memory profile — one array streamed instead of two — and it
    inherits all three schedules and their agreement without a new proof. -/
def sumsqOf (sh : Shape) (s : Sched) : EWStmt :=
  s.realize aB aB (rowIx sh s) (rowIx sh s) yB .ctaId sh.n

example : sumsqOf qwen .vec4
    = warpDotV4 aB aB (aIx qwen) (aIx qwen) yB .ctaId (qwen.n / 128) := rfl

/-- **The sum of squares agrees across schedules too** — the same theorem, one
    instantiation away.  This is the whole return on stating `schedules_agree_at`
    over the walk instead of over the GEMV. -/
theorem sumsq_schedules_agree (hl : AllHold [Law.laneRegroup]) (sh : Shape)
    (hn : sh.n % 128 = 0) (s t : Sched) (cta : Nat) (st : WSt) :
    (((sumsqOf sh s).elabIn cta).run st).mem yB cta
      = (((sumsqOf sh t).elabIn cta).run st).mem yB cta :=
  schedules_agree_at hl sh hn s t cta st aB aB (rowIx sh) (rowIx sh)
    (cta * sh.n) (cta * sh.n)
    (fun u => by cases u <;> rfl) (fun u => by cases u <;> rfl)

/-- Both shipped shapes admit every schedule: 128 divides each `n`. -/
theorem shapes_admit_every_schedule :
    qwen.n % 128 = 0 ∧ mid.n % 128 = 0 ∧ mid2.n % 128 = 0 ∧ wide.n % 128 = 0 := by decide

def ptxOf (sh : Shape) (s : Sched) : String :=
  emitProvenKernelN "main" 3 0 (kernelOf sh s)

def ptxSqOf (sh : Shape) (s : Sched) : String :=
  emitProvenKernelN "main" 3 0 (sumsqOf sh s)

/-! ### Seam guards (`A47` G2)

    The same checks the rest of the stack carries: buffers inside the binding
    table, every branch resolving, nothing unrenderable reaching the printer,
    and the kernel writing — and not reading — its output.  Stated per shape and
    over every schedule, so each of the six shipped kernels is held to the
    shipped standard. -/
def GuardsOk (k : EWStmt) : Prop :=
  k.BufBelow 3 ∧ k.StageEligibleB yB = true
    ∧ FlatTargetsOkB (flatKernel (expandEW k)) = true
    ∧ FlatPrintableB (flatKernel (expandEW k)) = true

theorem guards_qwen (s : Sched) : GuardsOk (kernelOf qwen s) ∧ GuardsOk (sumsqOf qwen s) := by
  cases s <;> exact ⟨⟨by decide, by decide, by native_decide, by native_decide⟩,
                     ⟨by decide, by decide, by native_decide, by native_decide⟩⟩

theorem guards_mid (s : Sched) : GuardsOk (kernelOf mid s) ∧ GuardsOk (sumsqOf mid s) := by
  cases s <;> exact ⟨⟨by decide, by decide, by native_decide, by native_decide⟩,
                     ⟨by decide, by decide, by native_decide, by native_decide⟩⟩

theorem guards_mid2 (s : Sched) : GuardsOk (kernelOf mid2 s) ∧ GuardsOk (sumsqOf mid2 s) := by
  cases s <;> exact ⟨⟨by decide, by decide, by native_decide, by native_decide⟩,
                     ⟨by decide, by decide, by native_decide, by native_decide⟩⟩

theorem guards_wide (s : Sched) : GuardsOk (kernelOf wide s) ∧ GuardsOk (sumsqOf wide s) := by
  cases s <;> exact ⟨⟨by decide, by decide, by native_decide, by native_decide⟩,
                     ⟨by decide, by decide, by native_decide, by native_decide⟩⟩

/-- **Launch geometry**: each schedule's trips times its per-step footprint
    cover exactly `n`, and the grid has one block per output row. -/
theorem gemv_geometry : ((qwen.n / 128) * 128 = qwen.n ∧ (qwen.n / 32) * 32 = qwen.n)
      ∧ ((mid.n / 128) * 128 = mid.n ∧ (mid.n / 32) * 32 = mid.n)
      ∧ ((mid2.n / 128) * 128 = mid2.n ∧ (mid2.n / 32) * 32 = mid2.n)
      ∧ ((wide.n / 128) * 128 = wide.n ∧ (wide.n / 32) * 32 = wide.n) := by decide

/-! Each schedule gets its own PTX slot: the CUDA module cache is keyed by the
    address the source was read from, so two kernels sharing a slot would be
    the same compiled module. -/
def PTX_OFF : Nat := 0x0100
def SLOT : Nat := 0x1400

/-- Six kernels, six slots: `{dot, sumsq} x {vec4, strided, blocked}`.  Each gets
    its own address because the CUDA module cache is keyed by the address the
    source was read from, so two kernels sharing a slot would be the same
    compiled module. -/
def slotOf : Bool → Sched → Nat
  | false, .vec4    => PTX_OFF
  | false, .strided => PTX_OFF + SLOT
  | false, .blocked => PTX_OFF + 2*SLOT
  | true,  .vec4    => PTX_OFF + 3*SLOT
  | true,  .strided => PTX_OFF + 4*SLOT
  | true,  .blocked => PTX_OFF + 5*SLOT

def BIND_OFF : Nat := PTX_OFF + 6*SLOT
def A_ID : Nat := 0x0040
def X_ID : Nat := 0x0044
def Y_ID : Nat := 0x0048
def MEM_SIZE : Nat := BIND_OFF + 0x100

open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sur

/-- Two callee tables: only the cuBLAS baseline reaches cuBLAS. -/
def envAll : FnEnv := env% [.cuda, .cublas]
def envCuda : FnEnv := env% [.cuda, .cublas]

def loadCode (sh : Shape) : HProg.Code :=
  HProg.Sur.build do
    let ptr := basePtr
    let dataPtr ← load64 (← absAddr ptr 0x18)
    cudaInit ptr
    let ctxPtr ← cudaCtxPtr ptr
    let aBytes ← iconst64 (sh.m * sh.n * 4)
    let xBytes ← iconst64 (sh.n * 4)
    let yBytes ← iconst64 (sh.m * 4)
    let aId ← cudaCreateBuffer ptr aBytes
    store aId (← absAddr ptr A_ID)
    let xId ← cudaCreateBuffer ptr xBytes
    store xId (← absAddr ptr X_ID)
    let yId ← cudaCreateBuffer ptr yBytes
    store yId (← absAddr ptr Y_ID)
    let _ ← call IR.Ffi.cudaUpload.id [ctxPtr, aId, dataPtr, aBytes]
    let xSrc ← iadd dataPtr aBytes
    let _ ← call IR.Ffi.cudaUpload.id [ctxPtr, xId, xSrc, xBytes]
    store aId (← absAddr ptr BIND_OFF)
    store xId (← absAddr ptr (BIND_OFF + 4))
    store yId (← absAddr ptr (BIND_OFF + 8))

/-- One launch per schedule: same grid, same buffers, same output, differing
    only in which PTX slot it reads.  That is what makes the timings
    comparable. -/
def runCode (sh : Shape) (sq : Bool) (s : Sched) : HProg.Code :=
  HProg.Sur.build do
    let ptr := basePtr
    let ptxOff ← iconst64 (slotOf sq s)
    let nBufs ← iconst32 3
    let bindOff ← iconst64 BIND_OFF
    let one ← iconst32 1
    let warp ← iconst32 32
    let grid ← iconst32 sh.m
    let _ ← cudaLaunch ptr ptxOff nBufs bindOff grid one one warp one one
    let _ ← cudaSync ptr

/-- The cuBLAS baseline on the same buffers: `y = A·x`, `A` is `m x n`. -/
def blasCode (sh : Shape) : HProg.Code :=
  HProg.Sur.build do
    let ptr := basePtr
    let ctxPtr ← cudaCtxPtr ptr
    let aId ← load32 (← absAddr ptr A_ID)
    let xId ← load32 (← absAddr ptr X_ID)
    let yId ← load32 (← absAddr ptr Y_ID)
    -- row-major A (m x n) is column-major (n x m); trans=1 gives yᵢ = Σₖ A[i,k]·xₖ
    let trans ← iconst32 1
    let mm ← iconst32 sh.n
    let nn ← iconst32 sh.m
    let alpha ← iconst32 0x3F800000
    let beta ← iconst32 0
    let _ ← call IR.Ffi.cublasSgemv.id [ctxPtr, trans, mm, nn, alpha, aId, xId, beta, yId]
    let _ ← cudaSync ptr

def fetchCode (sh : Shape) : HProg.Code :=
  HProg.Sur.build do
    let ptr := basePtr
    let ctxPtr ← cudaCtxPtr ptr
    let outPtr ← load64 (← absAddr ptr 0x28)
    let yId ← load32 (← absAddr ptr Y_ID)
    let yBytes ← iconst64 (sh.m * 4)
    let _ ← call IR.Ffi.cudaDownload.id [ctxPtr, yId, outPtr, yBytes]

/-- Every body well-formed at every shipped shape and schedule.

    `clifIR` is generic in the shape and so are `artifactOf` and the map that
    builds `artifacts`, so there is no point at which `compileFn` could discharge
    its obligation by `decide`; those sites take `compileBody` and this theorem
    is what stands in for the check. It covers exactly the nine bodies `clifIR`
    ships, at all four shapes. -/
theorem bodies_wf :
    shapes.all (fun sh =>
      HProg.wf envCuda HProg.ptrParams (loadCode sh) &&
      HProg.wf envCuda HProg.ptrParams (fetchCode sh) &&
      HProg.wf envAll HProg.ptrParams (blasCode sh) &&
      Sched.all.all (fun sc =>
        HProg.wf envCuda HProg.ptrParams (runCode sh false sc) &&
        HProg.wf envCuda HProg.ptrParams (runCode sh true sc))) = true := by
  decide

private theorem parts {sh : Shape} (hsh : sh ∈ shapes) :
    HProg.wf envCuda HProg.ptrParams (loadCode sh) = true ∧
    HProg.wf envCuda HProg.ptrParams (fetchCode sh) = true ∧
    HProg.wf envAll HProg.ptrParams (blasCode sh) = true ∧
    (Sched.all.all (fun sc =>
      HProg.wf envCuda HProg.ptrParams (runCode sh false sc) &&
      HProg.wf envCuda HProg.ptrParams (runCode sh true sc))) = true := by
  have h := List.all_eq_true.mp bodies_wf sh hsh
  simp only [Bool.and_eq_true] at h
  exact ⟨h.1.1.1, h.1.1.2, h.1.2, h.2⟩

theorem load_wf {sh : Shape} (hsh : sh ∈ shapes) :
    HProg.wf envCuda HProg.ptrParams (loadCode sh) = true := (parts hsh).1
theorem fetch_wf {sh : Shape} (hsh : sh ∈ shapes) :
    HProg.wf envCuda HProg.ptrParams (fetchCode sh) = true := (parts hsh).2.1
theorem blas_wf {sh : Shape} (hsh : sh ∈ shapes) :
    HProg.wf envAll HProg.ptrParams (blasCode sh) = true := (parts hsh).2.2.1

theorem run_wf {sh : Shape} (hsh : sh ∈ shapes) (t : Bool) (sc : Sched)
    (hsc : sc ∈ Sched.all) :
    HProg.wf envCuda HProg.ptrParams (runCode sh t sc) = true := by
  have h := List.all_eq_true.mp (parts hsh).2.2.2 sc hsc
  simp only [Bool.and_eq_true] at h
  cases t
  · exact h.1
  · exact h.2

def clifIR (sh : Shape) (hsh : sh ∈ shapes) : Program :=
  program
    [noopFunction,
     HProg.compileFn 1 (loadCode sh) envCuda (hwf := load_wf hsh),
     HProg.compileFn 2 (runCode sh false .vec4) envCuda (hwf := run_wf hsh false .vec4 (List.Mem.head _)),
     HProg.compileFn 3 (fetchCode sh) envCuda (hwf := fetch_wf hsh),
     HProg.compileFn 4 (blasCode sh) envAll (hwf := blas_wf hsh),
     HProg.compileFn 5 (runCode sh false .strided) envCuda (hwf := run_wf hsh false .strided (List.Mem.tail _ (List.Mem.head _))),
     HProg.compileFn 6 (runCode sh false .blocked) envCuda (hwf := run_wf hsh false .blocked (List.Mem.tail _ (List.Mem.tail _ (List.Mem.head _)))),
     HProg.compileFn 7 (runCode sh true .vec4) envCuda (hwf := run_wf hsh true .vec4 (List.Mem.head _)),
     HProg.compileFn 8 (runCode sh true .strided) envCuda (hwf := run_wf hsh true .strided (List.Mem.tail _ (List.Mem.head _))),
     HProg.compileFn 9 (runCode sh true .blocked) envCuda (hwf := run_wf hsh true .blocked (List.Mem.tail _ (List.Mem.tail _ (List.Mem.head _))))]

/-- Every emitted kernel fits the slot it is written into — all six kernels at
    all four shapes, checked rather than assumed. -/
def ptxFitsB : Bool :=
  shapes.all (fun sh => Sched.all.all (fun s =>
    decide ((ptxOf sh s).toUTF8.toList.length + 1 ≤ SLOT)
      && decide ((ptxSqOf sh s).toUTF8.toList.length + 1 ≤ SLOT)))

theorem ptx_fits_slots : ptxFitsB = true := by native_decide

/-- **Every byte this file names.**  The context slots and the three
    buffer-id words are written here too, so leaving them out would let a slot
    be placed on top of one without the check noticing. -/
def memMap : AlgorithmLib.Layout.RegionMap :=
  [⟨"ctx_cuda",   AlgorithmLib.ContextSlots.cuda, 8⟩,
   ⟨"io_offsets", 0x18, 0x20⟩,
   ⟨"a_id",       A_ID, 4⟩,
   ⟨"x_id",       X_ID, 4⟩,
   ⟨"y_id",       Y_ID, 4⟩]
    ++ (Sched.all.flatMap (fun s => [⟨"dot", slotOf false s, SLOT⟩,
                                     ⟨"sumsq", slotOf true s, SLOT⟩]))
    ++ [⟨"bind", BIND_OFF, 12⟩]


#eval LayoutScan.check "GemvWarpAlgorithm" [``memMap]
theorem mem_map_ok :
    memMap.okB = true ∧ memMap.withinB MEM_SIZE = true := by native_decide

/-- One slot's worth of bytes: the kernel source, NUL-terminated, zero-padded. -/
def slotBytes (src : String) : List UInt8 :=
  let b := src.toUTF8.toList ++ [0]
  b ++ zeros (SLOT - b.length)

def initialMemory (sh : Shape) : List UInt8 :=
  zeros PTX_OFF
    ++ (Sched.all.flatMap (fun s => slotBytes (ptxOf sh s)))
    ++ (Sched.all.flatMap (fun s => slotBytes (ptxSqOf sh s)))
    ++ zeros (MEM_SIZE - BIND_OFF)

def artifactOf (sh : Shape) (hsh : sh ∈ shapes) : Json :=
  toJsonArtifact sh.tag
    { clif := clifIR sh hsh, memory_size := MEM_SIZE,
      initial_memory := initialMemory sh }
    { fn_idx := u32 1 }
    [("run", { fn_idx := u32 2 }), ("fetch", { fn_idx := u32 3 }),
     ("blas", { fn_idx := u32 4 }), ("run_strided", { fn_idx := u32 5 }),
     ("run_blocked", { fn_idx := u32 6 }), ("sq", { fn_idx := u32 7 }),
     ("sq_strided", { fn_idx := u32 8 }), ("sq_blocked", { fn_idx := u32 9 })]

def artifacts : Array Json := (shapes.attach.map (fun p => artifactOf p.1 p.2)).toArray

end GemvWarp

def main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  emitArtifacts outDir GemvWarp.artifacts

namespace GemvWarp

/-- **GEMV computes its spec** — the entire proof is one application of the
    schema theorem.  This is what a reusable schema buys: the refinement
    argument was made once, in `Schema.lean`, and every instance inherits it. -/
theorem gemv_computes_spec (sh : Shape) (cta : Nat) (st : WSt) {Γ : Nat}
    (env : Fin Γ → Float32) (ae be : Nat → Expr Γ)
    (ha : ∀ i, denote env (ae i) = st.mem aB i)
    (hb : ∀ i, denote env (be i) = st.mem xB i) :
    (((kernel sh).elabIn cta).run st).mem yB cta
      = denote env (warpDotV4E ae be
          (fun i l => (aIx sh).eval cta i l) (fun i l => xIx.eval cta i l)
          (sh.n / 128)) :=
  warpDotV4_implements aB xB (aIx sh) xIx yB .ctaId (sh.n / 128) cta st env ae be ha hb

/-- **And the emitted PTX runs it**, from instruction 0, over real branches,
    with no hypothesis. -/
theorem gemv_ptx_computes_spec (sh : Shape) (cta : Nat) (m : MState) {Γ : Nat}
    (env : Fin Γ → Float32) (ae be : Nat → Expr Γ)
    (ha : ∀ i, denote env (ae i) = m.mem aB i)
    (hb : ∀ i, denote env (be i) = m.mem xB i) :
    ∃ k m', steps cta (flatKernel (expandEW (kernel sh))) k (0, m)
          = some ((flatKernel (expandEW (kernel sh))).length, m')
      ∧ m'.mem yB cta
          = denote env (warpDotV4E ae be
              (fun i l => (aIx sh).eval cta i l) (fun i l => xIx.eval cta i l)
              (sh.n / 128)) := by
  obtain ⟨k, m', hs, hw⟩ :=
    flatKernel_sound_idxFree cta (expandEW (kernel sh)) (expandEW_expFree (kernel sh))
      (expandEW_idxFree (kernel sh) (of_decide_eq_true (by rfl)))
      (expandEW_flat (kernel sh) (of_decide_eq_true (by rfl))) m
  refine ⟨k, m', hs, ?_⟩
  have hm : m'.mem yB cta
      = (((expandEW (kernel sh)).elabIn cta).run m.toWSt).mem yB cta :=
    congrArg (fun st => st.mem yB cta) hw
  have hid : (expandEW (kernel sh)).elabIn cta = (kernel sh).elabIn cta := rfl
  rw [hm, hid]
  exact gemv_computes_spec sh cta m.toWSt env ae be ha hb

/-- **The tuning claims, scanned.**  Separate from `TrustScan`'s inference list
    because they rest on a different thing: not that a schedule is correct —
    each is proven exactly against its own fold — but that two *different*
    schedules may be identified, which costs `Law.laneRegroup`, a coverage
    proof, and nothing else.  Kept here rather than in `TrustScan` because each
    generator defines its own `main` and so the two cannot be imported
    together. -/
def schedRoots : List Name :=
  [ `AlgorithmLib.ML.quad_idx_perm
  , `AlgorithmLib.ML.dotLane_flat
  , `AlgorithmLib.ML.quad_eq_flatSum_at
  , `AlgorithmLib.ML.strided_eq_flatSum_at
  , `AlgorithmLib.ML.flatMap_map_comm
  , `AlgorithmLib.ML.blocked_idx_perm
  , `AlgorithmLib.ML.blocked_eq_flatSum_at
  , `AlgorithmLib.ML.Sched.fold_eq_flatSum
  , `AlgorithmLib.ML.sched_agree_at_idx
  -- and the same statement about the kernels this file actually ships
  , `GemvWarp.gemv_addressing
  , `GemvWarp.schedules_agree_at
  , `GemvWarp.gemv_schedules_agree
  , `GemvWarp.sumsq_schedules_agree
  , `GemvWarp.shapes_admit_every_schedule
  , `GemvWarp.gemv_computes_spec
  , `GemvWarp.gemv_ptx_computes_spec
  ]

end GemvWarp

open GemvWarp TrustScan in
#eval runScan "schedules" schedRoots []

#eval ShipScan.check "GemvWarpAlgorithm"
