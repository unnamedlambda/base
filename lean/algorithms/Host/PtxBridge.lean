module
public import Lean
public import AlgorithmLib.Kernel.Corpus
meta import AlgorithmLib.Kernel.Corpus
public import AlgorithmLib.Kernel.PtxRun
meta import AlgorithmLib.Kernel.PtxRun
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic

@[expose] public section

/-!
# PTX instructions, checked against the GPU one at a time

The semantics of PTX (`PtxRun`) is the one thing about the device the proofs
trust, so it is checked here instruction by instruction: every form the kernel
compiler writes (`Kernel.Ptx`), each in a kernel of its own that loads its
operands from buffers, applies the instruction once per thread and stores the
result. Its operands are random words and the values where instructions
disagree: zeros of both signs, infinities, NaNs with payloads, subnormals,
the ends of each integer range, shift amounts past the width.

`PtxRun` runs each kernel and the files it leaves are what the GPU must
leave, bit for bit, a NaN's payload included:

    cd lean && lake env lean -Dexperimental.module=true --run algorithms/Host/PtxBridge.lean DIR
    tools/kernel_agree.py DIR

It needs a GPU, so it is run by hand rather than by the gate.
-/

open Lean
open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.Kernel AlgorithmLib.Kernel.Corpus
open AlgorithmLib.PtxIr (Instr Line Opnd Ty)

namespace PtxBridge

/-- What an operand holds, and the values it is drawn from. -/
inductive Kind where
  | i32 | i64 | f32 | f64 | pred
  /-- A 32-bit shift amount, below and past every width. -/
  | sh
  /-- A divisor: never zero. -/
  | nz32 | nz64
  /-- An f32 in `[-π, π]`, where `sin.approx` and `cos.approx` state their error. -/
  | angle
  deriving BEq

/-- A register named freely. -/
def R (s : String) : PtxIr.Reg := .name s

/-- The register an operand of this kind is held in, for operand `x`. -/
def Kind.reg (k : Kind) (x : String) : PtxIr.Reg :=
  match k with
  | .i32 | .sh | .nz32 => (R s!"r{x}") | .i64 | .nz64 => (R s!"d{x}") | .f32 | .angle => (R s!"f{x}")
  | .f64 => (R s!"e{x}") | .pred => (R s!"p{x}")

def N : Nat := 256

def f32Specials : List Nat :=
  [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000, 0x7f800001, 0x7fa00000,
   0x00000001, 0x80000001, 0x007fffff, 0x00800000, 0x7f7fffff, 0xff7fffff, 0x3f800000, 0xbf800000,
   0x3f000000, 0x4b000000, 0x4f000000, 0x4f800000, 0xcf000000, 0x5f000000, 0xdf000000, 0x3fc00000]

def f64Specials : List Nat :=
  [0, 0x8000000000000000, 0x7ff0000000000000, 0xfff0000000000000, 0x7ff8000000000000,
   0x7ff0000000000001, 0x0000000000000001, 0x3ff0000000000000, 0xbff0000000000000,
   0x47efffffe0000000, 0x47efffffffffffff, 0x3810000000000000, 0x36a0000000000000, 0x7fefffffffffffff]

def int32Specials : List Nat :=
  [0, 1, 2, 3, 31, 32, 33, 0x7fffffff, 0x80000000, 0x80000001, 0xffffffff, 0xfffffffe, 0x55555555]

def int64Specials : List Nat :=
  [0, 1, 2, 63, 64, 0x7fffffffffffffff, 0x8000000000000000, 0xffffffffffffffff, 0xffffffff, 0x100000000]

/-- `N` 64-bit words for operand `k`, from `seed`: its special values first. -/
def values (k : Kind) (seed : Nat) : List Nat :=
  let r := lcg seed (2 * N)
  let w64 := (List.range N).map fun i => r.getD (2 * i) 0 * 2 ^ 32 + r.getD (2 * i + 1) 0
  let w32 := r.take N
  -- floats near one, where the arithmetic is ordinary
  let nice := (r.take N).map fun x => ((x % 65536).toFloat / 4096.0 - 8.0).toFloat32.toBits.toNat
  let xs := match k with
    | .i32 => int32Specials ++ w32
    | .nz32 => (int32Specials.filter (· != 0)) ++ w32.map (fun x => if x == 0 then 7 else x)
    | .i64 => int64Specials ++ w64
    | .nz64 => (int64Specials.filter (· != 0)) ++ w64.map (fun x => if x == 0 then 7 else x)
    | .sh => (List.range 72) ++ w32
    | .pred => w32.map (· % 2)
    | .f32 => f32Specials ++ (List.range N).map fun i => if i % 2 == 0 then nice.getD i 0 else w32.getD i 0
    | .angle => [0, 0x80000000] ++ (r.take N).map fun x =>
        ((x % 65536).toFloat / 65536.0 * 6.283185 - 3.1415925).toFloat32.toBits.toNat
    | .f64 => f64Specials ++ (List.range N).map fun i =>
        if i % 2 == 0 then ((nice.getD i 0 : Nat).toUInt32 |> Float32.ofBits |>.toFloat |>.toBits |>.toNat)
        else w64.getD i 0
  xs.take N

def le64 (x : Nat) : List UInt8 := (List.range 8).map fun i => UInt8.ofNat (x / 256 ^ i)

/-- One test: an instruction on up to three operands, writing a result of kind
    `res` into the register it is given. -/
structure T where
  name : String
  ops : List Kind
  res : Kind
  ins : List PtxIr.Reg → PtxIr.Reg → Instr
  ulps : Nat := 0
  /-- An absolute tolerance on an f32 result, for an approximation whose error
      is stated absolutely. -/
  absTol : Float := 0

/-- Load operand `x` (parameter `p{i}`) into its register. -/
def loadOp (k : Kind) (i : Nat) (x : String) : List Line :=
  let ins := fun j => Line.ins none j
  [ins (.ld .param false .u64 (R s!"b{x}") { base := .sym s!"p{i}" }),
   ins (.bin .add none .u64 (R s!"b{x}") (.reg (R s!"b{x}")) (.reg (R "off"))),
   ins (.ld .global false .b64 (R s!"d{x}") { base := .reg (R s!"b{x}") })] ++
  (match k with
   | .i32 | .sh | .nz32 => [ins (.cvt none .u32 .u64 (R s!"r{x}") (.reg (R s!"d{x}")))]
   | .f32 | .angle =>
       [ins (.cvt none .u32 .u64 (R s!"r{x}") (.reg (R s!"d{x}"))), ins (.mov .b32 (R s!"f{x}") (.reg (R s!"r{x}")))]
   | .f64 => [ins (.mov .b64 (R s!"e{x}") (.reg (R s!"d{x}")))]
   | .pred => [ins (.setp .ne .u64 (R s!"p{x}") (.reg (R s!"d{x}")) (.imm 0))]
   | .i64 | .nz64 => [])

/-- Store the result, widened to 64 bits. -/
def storeRes (k : Kind) : List Line :=
  let ins := fun j => Line.ins none j
  [ins (.ld .param false .u64 (R "bo") { base := .sym "p3" }),
   ins (.bin .add none .u64 (R "bo") (.reg (R "bo")) (.reg (R "off")))] ++
  (match k with
   | .i32 | .sh | .nz32 => [ins (.cvt none .u64 .u32 (R "do") (.reg (R "ro")))]
   | .f32 | .angle => [ins (.mov .b32 (R "ro") (.reg (R "fo"))), ins (.cvt none .u64 .u32 (R "do") (.reg (R "ro")))]
   | .f64 => [ins (.mov .b64 (R "do") (.reg (R "eo")))]
   | .pred => [ins (.selp .u64 (R "do") (.imm 1) (.imm 0) (.reg (R "po")))]
   | .i64 | .nz64 => []) ++
  [ins (.st .global .b64 { base := .reg (R "bo") } (.reg (R "do"))), ins .ret]

def entry (t : T) : PtxIr.Entry :=
  let xs := ["a", "b", "c"]
  let names (p : String) := (xs ++ ["o"]).map fun x => R (p ++ x)
  { name := t.name, params := [(.u64, "p0"), (.u64, "p1"), (.u64, "p2"), (.u64, "p3")],
    regs := [.names .b32 (names "r" ++ [R "gt"]), .names .b64 (names "d" ++ names "b" ++ [R "off"]),
             .names .f32 (names "f"), .names .f64 (names "e"), .names .pred (names "p")],
    body :=
      [.ins none (.mov .u32 (R "gt") (.reg (.special "tid.x"))),
       .ins none (.bin .mulWide none .u32 (R "off") (.reg (R "gt")) (.imm 8))] ++
      (t.ops.zip xs).zipIdx.flatMap (fun ((k, x), i) => loadOp k i x) ++
      [.ins none (t.ins ((t.ops.zip xs).map fun (k, x) => k.reg x) (t.res.reg "o"))] ++
      storeRes t.res }

open PtxIr in
def tests : List T :=
  let b2 (n : String) (o : BinOp) (r : Option Rnd) (ty : Ty) (k : Kind) (k2 : Kind := k) : T :=
    { name := n, ops := [k, k2], res := k, ins := fun as d => .bin o r ty d (.reg (as.getD 0 (R ""))) (.reg (as.getD 1 (R ""))) }
  let u1 (n : String) (o : UnOp) (r : Option Rnd) (ty : Ty) (k : Kind) (res : Kind := k) : T :=
    { name := n, ops := [k], res, ins := fun as d => .un o r ty d (.reg (as.getD 0 (R ""))) }
  let cv (n : String) (r : Option Rnd) (dt st : Ty) (k res : Kind) : T :=
    { name := n, ops := [k], res, ins := fun as d => .cvt r dt st d (.reg (as.getD 0 (R ""))) }
  let sp (n : String) (c : Cmp) (ty : Ty) (k : Kind) : T :=
    { name := n, ops := [k, k], res := .pred, ins := fun as d => .setp c ty d (.reg (as.getD 0 (R ""))) (.reg (as.getD 1 (R ""))) }
  [b2 "add_u32" .add none .u32 .i32, b2 "add_u64" .add none .u64 .i64,
   b2 "sub_u32" .sub none .u32 .i32, b2 "sub_u64" .sub none .u64 .i64,
   b2 "mul_lo_u32" .mulLo none .u32 .i32, b2 "mul_lo_u64" .mulLo none .u64 .i64,
   b2 "div_u32" .div none .u32 .i32 .nz32, b2 "div_u64" .div none .u64 .i64 .nz64,
   b2 "and_b32" .and none .b32 .i32, b2 "or_b32" .or none .b32 .i32, b2 "xor_b32" .xor none .b32 .i32,
   b2 "and_b64" .and none .b64 .i64, b2 "or_b64" .or none .b64 .i64, b2 "xor_b64" .xor none .b64 .i64,
   b2 "and_pred" .and none .pred .pred, b2 "or_pred" .or none .pred .pred, b2 "xor_pred" .xor none .pred .pred,
   b2 "shl_b32" .shl none .b32 .i32 .sh, b2 "shr_u32" .shr none .u32 .i32 .sh, b2 "shr_s32" .shr none .s32 .i32 .sh,
   { name := "shl_b64", ops := [.i64, .sh], res := .i64, ins := fun as d => .bin .shl none .b64 d (.reg (as.getD 0 (R ""))) (.reg (as.getD 1 (R ""))) },
   { name := "shr_u64", ops := [.i64, .sh], res := .i64, ins := fun as d => .bin .shr none .u64 d (.reg (as.getD 0 (R ""))) (.reg (as.getD 1 (R ""))) },
   { name := "shr_s64", ops := [.i64, .sh], res := .i64, ins := fun as d => .bin .shr none .s64 d (.reg (as.getD 0 (R ""))) (.reg (as.getD 1 (R ""))) },
   b2 "add_rn_f32" .add (some .rn) .f32 .f32, b2 "sub_rn_f32" .sub (some .rn) .f32 .f32,
   b2 "mul_rn_f32" .mul (some .rn) .f32 .f32, b2 "div_rn_f32" .div (some .rn) .f32 .f32,
   b2 "max_NaN_f32" .maxNaN none .f32 .f32, b2 "min_NaN_f32" .minNaN none .f32 .f32,
   u1 "neg_f32" .neg none .f32 .f32, u1 "sqrt_rn_f32" .sqrt (some .rn) .f32 .f32,
   u1 "not_pred" .not none .pred .pred,
   u1 "brev_b32" .brev none .b32 .i32, u1 "brev_b64" .brev none .b64 .i64,
   u1 "clz_b32" .clz none .b32 .i32, u1 "clz_b64" .clz none .b64 .i64 .i32,
   { u1 "ex2_approx_f32" .ex2Approx none .f32 .f32 with ulps := 2 },
   { u1 "rcp_approx_f32" .rcpApprox none .f32 .f32 with ulps := 2 },
   -- within 2 ulps but near an answer of zero, where the hardware's
   -- error is absolute: 5 ulps there, measured
   { u1 "lg2_approx_f32" .lg2Approx none .f32 .f32 with ulps := 8 },
   { u1 "sqrt_approx_f32" .sqrtApprox none .f32 .f32 with ulps := 4 },
   { u1 "rsqrt_approx_f32" .rsqrtApprox none .f32 .f32 with ulps := 4 },
   -- within 2^-20.9 of the root on [-π, π], as PTX states it
   { u1 "sin_approx_f32" .sinApprox none .f32 .angle .f32 with absTol := 1.0e-6 },
   { u1 "cos_approx_f32" .cosApprox none .f32 .angle .f32 with absTol := 1.0e-6 },
   cv "cvt_rzi_s32_f32" (some .rzi) .s32 .f32 .f32 .i32,
   cv "cvt_rn_f32_s32" (some .rn) .f32 .s32 .i32 .f32, cv "cvt_rn_f32_u32" (some .rn) .f32 .u32 .i32 .f32,
   cv "cvt_rn_f32_s64" (some .rn) .f32 .s64 .i64 .f32, cv "cvt_rn_f32_u64" (some .rn) .f32 .u64 .i64 .f32,
   cv "cvt_rn_f32_f64" (some .rn) .f32 .f64 .f64 .f32,
   cv "cvt_u64_u32" none .u64 .u32 .i32 .i64, cv "cvt_s64_s32" none .s64 .s32 .i32 .i64,
   cv "cvt_u32_u64" none .u32 .u64 .i64 .i32,
   { name := "fma_rn_f32", ops := [.f32, .f32, .f32], res := .f32,
     ins := fun as d => .fma (some .rn) .f32 d (.reg (as.getD 0 (R ""))) (.reg (as.getD 1 (R ""))) (.reg (as.getD 2 (R ""))) },
   { name := "selp_b32", ops := [.i32, .i32, .pred], res := .i32,
     ins := fun as d => .selp .b32 d (.reg (as.getD 0 (R ""))) (.reg (as.getD 1 (R ""))) (.reg (as.getD 2 (R ""))) },
   { name := "selp_f32", ops := [.f32, .f32, .pred], res := .f32,
     ins := fun as d => .selp .f32 d (.reg (as.getD 0 (R ""))) (.reg (as.getD 1 (R ""))) (.reg (as.getD 2 (R ""))) },
   { name := "mov_pred", ops := [.pred], res := .pred, ins := fun as d => .mov .pred d (.reg (as.getD 0 (R ""))) }] ++
  ([Cmp.eq, .ne, .lt, .le, .gt, .ge].flatMap fun c =>
    [sp s!"setp_{c.name}_u32" c .u32 .i32, sp s!"setp_{c.name}_s32" c .s32 .i32,
     sp s!"setp_{c.name}_u64" c .u64 .i64, sp s!"setp_{c.name}_s64" c .s64 .i64]) ++
  ([Cmp.eq, .lt, .le, .gt, .ge, .neu, .equ, .ltu, .leu, .gtu, .geu].map fun c =>
    sp s!"setp_{c.name}_f32" c .f32 .f32)

/-- Write each test's PTX, inputs and what `PtxRun` leaves, as
    `tools/kernel_agree.py` reads them, exact. -/
def write (dir : System.FilePath) : IO Unit := do
  let mut index : Array Json := #[]
  for (t, ti) in tests.zipIdx do
    let e := entry t
    let m : PtxIr.Module := { version := "7.4", target := "sm_80", addressSize := 64, entries := [e] }
    let ins := (List.range 3).map fun i =>
      let k := t.ops.getD i .i64
      (⟨((values k (1000 + 7 * ti + i)).flatMap le64).toArray⟩ : ByteArray)
    let bufs := ins ++ [zeros (8 * N)]
    let params := (List.range 4).map fun i => V.sc .i64 (if h : i < 268435449 then addrOf (.buf ⟨i, h⟩) 0 else 0)
    let out ← match PtxRun.launchOn e 0 1 N params (fun i => bufs.getD i .empty) with
      | .ok out => pure out
      | .error m => IO.println s!"{t.name}: PtxRun fails: {m}"; continue
    let d := dir / "kernel_corpus" / t.name
    IO.FS.createDirAll d
    IO.FS.writeFile (d / "kernel.ptx") m.text
    for (b, i) in bufs.zipIdx do
      IO.FS.writeBinFile (d / s!"in{i}.bin") b
      IO.FS.writeBinFile (d / s!"out{i}.bin") (out i)
    index := index.push <| Json.mkObj [("name", toJson t.name), ("grid", toJson 1), ("block", toJson N),
      ("nbufs", toJson 4), ("scalars", Json.arr #[]), ("ulps", toJson t.ulps), ("abs", toJson t.absTol),
      ("exact", toJson true)]
  IO.FS.writeFile (dir / "kernel_corpus" / "index.json") (Json.arr index).compress
  IO.println s!"{tests.length} instructions"

end PtxBridge

def main (args : List String) : IO Unit := do
  match args with
  | [d] => PtxBridge.write (System.FilePath.mk d)
  | _ => throw (IO.userError "usage: lean --run algorithms/Host/PtxBridge.lean DIR")
