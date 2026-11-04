module
public import Lean
public import AlgorithmLib.Kernel.Corpus
meta import AlgorithmLib.Kernel.Corpus
public import AlgorithmLib.Surface.Prog
meta import AlgorithmLib.Surface.Prog
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic

@[expose] public section

/-!
# Kernels, checked against the GPU

Each kernel here is a body (`Code`) printed as PTX (`Kernel.ptx`) and run by
the model (`Kernel.launch`) on fixed inputs. `tools/kernel_agree.py` runs the
PTX on the GPU on the same inputs and compares every output bit, a NaN
matching any NaN: which NaN an operation makes is the hardware's.

    cd lean && lake env lean -Dexperimental.module=true --run algorithms/Host/KernelCorpus.lean DIR
    tools/kernel_agree.py DIR

It needs a GPU, so it is run by hand rather than by the gate.

They cover what a kernel does: arithmetic and memory (`saxpy`), a loop
(`rowSum`), a warp shuffle (`warpSum`), shared memory and barriers
(`blockSum`), a ballot (`extendStep`, LZ4's match extension), and the
hardware's approximations (`approx`), on arguments where they are exact.
-/

open Lean
open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.Prog AlgorithmLib.Kernel AlgorithmLib.Kernel.Corpus
open AlgorithmLib.HProg.Sem

namespace KernelCorpus

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

def shflBfly : LocalRef [.i32, .i32] (some .i32) := { callee := .intr .shflBfly }
def ballot : LocalRef [.i8] (some .i32) := { callee := .intr .ballot }
def barrier : LocalRef [] none := { callee := .intr .barrier }
def ex2Approx : LocalRef [.f32] (some .f32) := { callee := .intr .ex2Approx }
def rcpApprox : LocalRef [.f32] (some .f32) := { callee := .intr .rcpApprox }
def lg2Approx : LocalRef [.f32] (some .f32) := { callee := .intr .lg2Approx }

/-- The global thread index. -/
def gid (ps : Vals V (.i32 :: .i32 :: .i32 :: .i64 :: tys)) : Prog V L (V .i32) := do
  iadd (← imul ps.head ps.thd) ps.snd

/-- `y[i] := a·x[i] + y[i]` for `i < n`. -/
def saxpy : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i32, .f32])
  let i ← gid ps
  let x := (ps.drop 4).head; let y := (ps.drop 5).head
  let n := (ps.drop 6).head; let a := (ps.drop 7).head
  when .ult i n do
    let off ← ishlImm (← uextend64 i) 2
    let ya ← iadd y off
    store (← fma a (← loadF32 (← iadd x off)) (← loadF32 ya)) ya

/-- `out[i] := Σ_{j<m} x[i·m + j]` for `i < n`. -/
def rowSum : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i32, .i64])
  let i ← gid ps
  let x := (ps.drop 4).head; let out := (ps.drop 5).head
  let n := (ps.drop 6).head; let m := (ps.drop 7).head
  when .ult i n do
    let i64 ← uextend64 i
    let row ← iadd x (← ishlImm (← imul i64 m) 2)
    let s ← forLoopAcc m (← fconst32 0.0) fun j acc => do
      fadd acc (← loadF32 (← iadd row (← ishlImm j 2)))
    store s (← iadd out (← ishlImm i64 2))

/-- `out[w] := Σ` of warp `w`'s 32 values, by a butterfly of shuffles. -/
def warpSum : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64])
  let i ← gid ps
  let x := (ps.drop 4).head; let out := (ps.drop 5).head
  let i64 ← uextend64 i
  let mut v ← loadF32 (← iadd x (← ishlImm i64 2))
  for m in [16, 8, 4, 2, 1] do
    let o ← callLocal shflBfly %[← bitcast .i32 v, ← iconst32 m]
    v ← fadd v (← bitcast .f32 o)
  when .eq (← band i (← iconst32 31)) (← iconst32 0) do
    store v (← iadd out (← ishlImm (← ushrImm i64 5) 2))

/-- `out[b] := Σ` of block `b`'s 256 values, by a tree in shared memory. -/
def blockSum : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64])
  let sm := (ps.drop 3).head
  let x := (ps.drop 4).head; let out := (ps.drop 5).head
  let t64 ← uextend64 ps.snd
  let mine ← iadd sm (← ishlImm t64 2)
  store (← loadF32 (← iadd x (← ishlImm (← uextend64 (← gid ps)) 2))) mine
  callLocalVoid barrier %[]
  for s in [128, 64, 32, 16, 8, 4, 2, 1] do
    when .ult t64 (← iconst64 s) do
      store (← fadd (← loadF32 mine) (← loadF32 (← iaddImm mine (4 * s)))) mine
    callLocalVoid barrier %[]
  when .eq t64 (← iconst64 0) do
    store (← loadF32 sm) (← iadd out (← ishlImm (← uextend64 ps.head) 2))

/-- LZ4's extend step: how many of the 32 positions from `ml` match, lane `l`
    testing `ml + l`; each warp reads `(p, c, ml)` from `tbl`, lane 0 writes
    the count. -/
def extendStep : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64])
  let i ← gid ps
  let inp := (ps.drop 4).head; let tbl := (ps.drop 5).head
  let out := (ps.drop 6).head; let endCap := (ps.drop 7).head
  let warp ← uextend64 (← ushrImm i 5)
  let lane ← uextend64 (← band i (← iconst32 31))
  let ent ← iadd tbl (← ishlImm warp 4)
  let p ← uload32_64 ent; let c ← uload32_64 (← iaddImm ent 4); let ml ← uload32_64 (← iaddImm ent 8)
  let k ← iadd ml lane
  let pk ← iadd p k
  let bad ← ifte (jTys := [.i8]) .ult pk endCap
    (do
      let a ← uload8_64 (← iadd inp pk)
      let b ← uload8_64 (← iadd inp (← iadd c k))
      pure %[← icmp .ne a b])
    (pure %[← iconst .i8 1])
  let adv ← ctz (← callLocal ballot %[bad.head])
  when .eq lane (← iconst64 0) do
    storeI32 adv (← iadd out (← ishlImm warp 2))

/-- `e[i] := 2^x[i]`, `r[i] := 1/y[i]` and `l[i] := log₂ z[i]` by the
    hardware's approximations. -/
def approx : Prog V L Unit := do
  let ps ← entryParams (idParams ++ [.i64, .i64, .i64, .i64, .i64, .i64])
  let off ← ishlImm (← uextend64 (← gid ps)) 2
  let x := (ps.drop 4).head; let y := (ps.drop 5).head
  let e := (ps.drop 6).head; let r := (ps.drop 7).head
  store (← callLocal ex2Approx %[← loadF32 (← iadd x off)]) (← iadd e off)
  store (← callLocal rcpApprox %[← loadF32 (← iadd y off)]) (← iadd r off)
  let z := (ps.drop 8).head; let l := (ps.drop 9).head
  store (← callLocal lg2Approx %[← loadF32 (← iadd z off)]) (← iadd l off)
end

-- ---------------------------------------------------------------------------
-- Inputs
-- ---------------------------------------------------------------------------

/-- Where the approximations are exact: 2 to an integer power, the
    reciprocal of a power of two, and the logarithm of a power of two not
    below one (below one, `lg2.approx.f32` is not exact: it gives log₂ 2⁻³⁰
    as -29.999998); with zeros, infinities and a NaN. -/
def approxInputs : ByteArray × ByteArray × ByteArray :=
  let ks : List Float := (List.range 61).map fun k => k.toFloat - 30
  let specials := [0x7f800000, 0xff800000, 0x00000000, 0x80000000, 0x7fc00000]
  (f32s ks (specials ++ [0, 0]), f32s (ks.map (2 ^ ·) ++ ks.map (- 2 ^ ·)) (specials ++ [0, 0, 0]),
   f32s ((List.range 31).map fun k => 2 ^ k.toFloat) specials)

def extendInputs : ByteArray × ByteArray :=
  let rs := lcg 7 4096
  let rows := (List.range 64).map fun w =>
    let p := 100 + (rs.getD (200 + w) 0) % 3800
    (p, (rs.getD (300 + w) 0) % p, (rs.getD (400 + w) 0) % 40)
  -- every other warp's position copies its candidate's next 80 bytes, so
  -- some extensions match all 32 positions
  let inp := (List.range rows.length).foldl (fun (b : Array UInt8) w =>
    if w % 2 = 0 then
      let (p, c, _) := rows.getD w (0, 0, 0)
      (List.range 80).foldl (fun b q => b.set! (p + q) (b[c + q]!)) b
    else b) ((rs.map fun r => UInt8.ofNat (r % 2)).toArray)
  (⟨inp⟩, ⟨(rows.flatMap fun (p, c, ml) => le32 p ++ le32 c ++ le32 ml ++ le32 0).toArray⟩)

def cases : Except String (List Case) := do
  let n := 1000
  let ext := extendInputs
  let ap := approxInputs
  pure [
    ⟨← kernelOf "saxpy" [.i64, .i64, .i32, .f32] saxpy, 8, 128, 2,
      [("u32", n), ("f32", (1.75 : Float).toFloat32.toBits.toNat)], [floats 1 n, floats 2 n], 0⟩,
    ⟨← kernelOf "rowSum" [.i64, .i64, .i32, .i64] rowSum, 1, 64, 2,
      [("u32", 64), ("u64", 50)], [floats 3 (64 * 50), zeros (64 * 4)], 0⟩,
    ⟨← kernelOf "warpSum" [.i64, .i64] warpSum, 2, 64, 2, [], [floats 4 128, zeros 16], 0⟩,
    ⟨← kernelOf "blockSum" [.i64, .i64] blockSum 1024, 3, 256, 2, [], [floats 5 768, zeros 12], 0⟩,
    ⟨← kernelOf "extendStep" [.i64, .i64, .i64, .i64] extendStep, 16, 128, 3,
      [("u64", 4000)], [ext.1, ext.2, zeros (64 * 4)], 0⟩,
    ⟨← kernelOf "approx" [.i64, .i64, .i64, .i64, .i64, .i64] approx, 1, 128, 6, [],
      [ap.1 ++ zeros (512 - ap.1.size), ap.2.1, zeros 512, zeros 512,
       ap.2.2 ++ zeros (512 - ap.2.2.size), zeros 512], 0⟩]

end KernelCorpus

open KernelCorpus in
def main (args : List String) : IO Unit := do
  let dir ← match args with
    | [d] => pure (System.FilePath.mk d)
    | _ => throw (IO.userError "usage: lean --run algorithms/Host/KernelCorpus.lean DIR")
  Corpus.write dir (← IO.ofExcept KernelCorpus.cases)
