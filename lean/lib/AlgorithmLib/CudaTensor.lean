import AlgorithmLib.Core
import AlgorithmLib.Bytes
import AlgorithmLib.Layout
import AlgorithmLib.IR
import AlgorithmLib.FFI

open Lean

namespace AlgorithmLib

namespace CudaTensor

/-- Low-level scalar IR for one lane of an elementwise CUDA kernel. -/
inductive Expr : Nat → Type where
  | input : Fin n → Expr n
  | const : String → Expr n
  | add : Expr n → Expr n → Expr n
  | mul : Expr n → Expr n → Expr n

/--
A staged tensor value for elementwise CUDA kernels.

At the surface level, users compose whole tensors. Lowering later turns that
into one scalar `Expr` evaluated at each element index.
-/
structure Tensor (inputs : Nat) where
  expr : Expr inputs

structure PersistentKernel (inputs : Nat) where
  expr : Expr inputs
  output : Fin inputs
  blockSize : Nat := 256
  timeoutMs : Nat := 30000
  ptxSourceOff : Nat := 0x0100
  bindDescOff : Nat := 0x1400

structure CompileResult where
  config : BaseConfig
  loadAlgorithm : Algorithm
  prepAlgorithm : Algorithm
  inferAlgorithm : Algorithm

instance : ToJson CompileResult where
  toJson r := Json.arr #[
    toJson r.config,
    toJson r.loadAlgorithm,
    toJson r.prepAlgorithm,
    toJson r.inferAlgorithm
  ]

def input {n : Nat} (i : Fin n) : Expr n := .input i
def const {n : Nat} (bits : String) : Expr n := .const bits
def add {n : Nat} (a b : Expr n) : Expr n := .add a b
def mul {n : Nat} (a b : Expr n) : Expr n := .mul a b

scoped infixl:65 " + " => add
scoped infixl:70 " * " => mul

def Tensor.ofExpr {n : Nat} (expr : Expr n) : Tensor n := ⟨expr⟩

def Tensor.input0 {n : Nat} : Tensor (n + 1) :=
  .ofExpr (.input ⟨0, by simp⟩)

def Tensor.input1 {n : Nat} : Tensor (n + 2) :=
  .ofExpr (.input ⟨1, by simp⟩)

def Tensor.scalarBits {n : Nat} (bits : String) : Tensor n :=
  .ofExpr (.const bits)

def Tensor.add {n : Nat} (x y : Tensor n) : Tensor n :=
  .ofExpr (.add x.expr y.expr)

def Tensor.mul {n : Nat} (x y : Tensor n) : Tensor n :=
  .ofExpr (.mul x.expr y.expr)

def Tensor.saxpy {n : Nat} (alpha x y : Tensor n) : Tensor n :=
  Tensor.add (Tensor.mul alpha x) y

instance {n : Nat} : HAdd (Tensor n) (Tensor n) (Tensor n) where
  hAdd := Tensor.add

instance {n : Nat} : HMul (Tensor n) (Tensor n) (Tensor n) where
  hMul := Tensor.mul

def Tensor.writeTo {n : Nat} (tensor : Tensor n) (out : Nat) (h : out < n := by decide) :
    PersistentKernel n := {
  expr := tensor.expr
  output := ⟨out, h⟩
}

private def paramName (i : Nat) : String :=
  s!"in{i}_ptr"

private def ptrReg (i : Nat) : String :=
  s!"%rd{1 + i}"

private structure EmitState where
  nextAddr : Nat := 10
  nextF : Nat := 0
  lines : List String := []

private def emitLine (line : String) : StateM EmitState Unit :=
  modify fun s => { s with lines := s.lines ++ [line] }

private def freshAddr : StateM EmitState String := do
  let s ← get
  let reg := s!"%rd{s.nextAddr}"
  set { s with nextAddr := s.nextAddr + 1 }
  pure reg

private def freshF : StateM EmitState String := do
  let s ← get
  let reg := s!"%f{s.nextF}"
  set { s with nextF := s.nextF + 1 }
  pure reg

private partial def emitExpr {n : Nat} (expr : Expr n) (offReg : String) : StateM EmitState String := do
  match expr with
  | .input idx =>
      let addr ← freshAddr
      let freg ← freshF
      emitLine s!"    add.u64 {addr}, {ptrReg idx.val}, {offReg};"
      emitLine s!"    ld.global.f32 {freg}, [{addr}];"
      pure freg
  | .const bits =>
      let freg ← freshF
      emitLine s!"    mov.f32 {freg}, {bits};"
      pure freg
  | .add a b =>
      let fa ← emitExpr a offReg
      let fb ← emitExpr b offReg
      let freg ← freshF
      emitLine s!"    add.f32 {freg}, {fa}, {fb};"
      pure freg
  | .mul a b =>
      let fa ← emitExpr a offReg
      let fb ← emitExpr b offReg
      let freg ← freshF
      emitLine s!"    mul.f32 {freg}, {fa}, {fb};"
      pure freg

private def ptxParams (inputs : Nat) : String :=
  let metaParam := "    .param .u64 meta_ptr"
  let ins := List.range inputs |>.map (fun i => s!"    .param .u64 {paramName i}")
  String.intercalate ",\n" (metaParam :: ins)

private def ptxLoadParams (inputs : Nat) : List String :=
  let metaLoad := "    ld.param.u64 %rd0, [meta_ptr];"
  let ins := List.range inputs |>.map (fun i => s!"    ld.param.u64 {ptrReg i}, [{paramName i}];")
  metaLoad :: ins

private def ptxSource {inputs : Nat} (spec : PersistentKernel inputs) : String :=
  let (resultF, exprState) := (emitExpr spec.expr "%rd9").run {}
  let outAddrReg := s!"%rd{exprState.nextAddr}"
  let outPtr := ptrReg spec.output.val
  let preLines := [
    ".version 8.0",
    ".target sm_86",
    ".address_size 64",
    "",
    ".visible .entry main(",
    ptxParams inputs,
    ")",
    "{",
    "    .reg .pred %p;",
    "    .reg .u32 %r<4>;",
    "    .reg .u64 %rd<32>;",
    "    .reg .f32 %f<32>;",
    ""
  ]
  let bodyLines :=
    ptxLoadParams inputs ++
    [
      "    ld.global.u32 %r0, [%rd0];",
      "    mov.u32 %r1, %ctaid.x;",
      "    mov.u32 %r2, %tid.x;",
      s!"    mad.lo.u32 %r1, %r1, {spec.blockSize}, %r2;",
      "    setp.ge.u32 %p, %r1, %r0;",
      "    @%p bra DONE;",
      "    cvt.u64.u32 %rd8, %r1;",
      "    shl.b64 %rd9, %rd8, 2;"
    ] ++
    exprState.lines ++
    [
      s!"    add.u64 {outAddrReg}, {outPtr}, %rd9;",
      s!"    st.global.f32 [{outAddrReg}], {resultF};",
      "DONE:",
      "    ret;",
      "}"
    ]
  String.intercalate "\n" (preLines ++ bodyLines) ++ "\n"

private def clifNoopFn : String :=
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "  return\n" ++
  "}\n"

private def clifLoadFn (inputs : Nat) : String :=
  let mkCreates :=
    String.intercalate "" <|
      (List.range inputs).map fun i =>
        let callIx := 21 + i
        let off := 0x44 + (4 * i)
        s!"  v{callIx} = call fn1(v3, v11)\n" ++
        s!"  store notrap aligned v{callIx}, v0+{off}\n"
  "function u0:1(i64) system_v {\n" ++
  "    sig0 = (i64) system_v\n" ++
  "    sig1 = (i64, i64) -> i32 system_v\n" ++
  "    sig2 = (i64, i32, i64, i64) -> i32 system_v\n" ++
  "    fn0 = %cl_cuda_init sig0\n" ++
  "    fn1 = %cl_cuda_create_buffer sig1\n" ++
  "    fn2 = %cl_cuda_upload sig2\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = load.i64 notrap aligned v0+0x18\n" ++
  "  v2 = iadd_imm v0, 16\n" ++
  "  call fn0(v2)\n" ++
  "  v3 = load.i64 notrap aligned v0+0x10\n" ++
  "  v10 = load.i64 notrap aligned v1\n" ++
  "  store notrap aligned v10, v0+0x38\n" ++
  "  v11 = ishl_imm v10, 2\n" ++
  "  v12 = iconst.i64 8\n" ++
  "  v20 = call fn1(v3, v12)\n" ++
  "  store notrap aligned v20, v0+0x40\n" ++
  mkCreates ++
  "  v40 = iadd_imm v0, 56\n" ++
  "  v41 = call fn2(v3, v20, v40, v12)\n" ++
  "  return\n" ++
  "}\n"

private def clifPrepFn (inputs : Nat) : String :=
  let uploads :=
    String.intercalate "" <|
      (List.range inputs).map fun i =>
        let bufOff := 0x44 + (4 * i)
        let bufVar := 10 + (3 * i)
        let ptrVar := 11 + (3 * i)
        let callVar := 12 + (3 * i)
        if i == 0 then
          s!"  v{bufVar} = load.i32 notrap aligned v0+{bufOff}\n" ++
          s!"  v{callVar} = call fn0(v3, v{bufVar}, v1, v5)\n"
        else
          let prevPtr := if i == 1 then 1 else 11 + (3 * (i - 1))
          s!"  v{bufVar} = load.i32 notrap aligned v0+{bufOff}\n" ++
          s!"  v{ptrVar} = iadd v{prevPtr}, v5\n" ++
          s!"  v{callVar} = call fn0(v3, v{bufVar}, v{ptrVar}, v5)\n"
  "function u0:2(i64) system_v {\n" ++
  "    sig0 = (i64, i32, i64, i64) -> i32 system_v\n" ++
  "    fn0 = %cl_cuda_upload_ptr sig0\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = load.i64 notrap aligned v0+0x18\n" ++
  "  v2 = load.i64 notrap aligned v0+0x38\n" ++
  "  v3 = load.i64 notrap aligned v0+0x10\n" ++
  "  v5 = ishl_imm v2, 2\n" ++
  uploads ++
  "  return\n" ++
  "}\n"

private def clifInferFn (inputs : Nat) (spec : PersistentKernel inputs) : String :=
  let outBufOff := 0x34 + (4 * spec.output.val)
  "function u0:3(i64) system_v {\n" ++
  "    sig0 = (i64, i32, i64, i64) -> i32 system_v\n" ++
  "    sig1 = (i64, i64, i32, i64, i32, i32, i32, i32, i32, i32) -> i32 system_v\n" ++
  "    sig2 = (i64) -> i32 system_v\n" ++
  "    fn0 = %cl_cuda_download_ptr sig0\n" ++
  "    fn1 = %cl_cuda_launch sig1\n" ++
  "    fn2 = %cl_cuda_sync sig2\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = load.i64 notrap aligned v0+0x28\n" ++
  "  v2 = load.i64 notrap aligned v0+0x30\n" ++
  "  v3 = load.i64 notrap aligned v0+0x38\n" ++
  "  v18 = load.i64 notrap aligned v0+0x10\n" ++
  s!"  v4 = iadd_imm v3, {spec.blockSize - 1}\n" ++
  "  v5 = ushr_imm v4, 8\n" ++
  "  v6 = ireduce.i32 v5\n" ++
  s!"  v7 = iadd_imm v0, {spec.ptxSourceOff}\n" ++
  s!"  v8 = iconst.i32 {inputs + 1}\n" ++
  s!"  v9 = iadd_imm v0, {spec.bindDescOff}\n" ++
  "  v10 = iconst.i32 1\n" ++
  s!"  v11 = iconst.i32 {spec.blockSize}\n" ++
  "  v12 = call fn1(v18, v7, v8, v9, v6, v10, v10, v11, v10, v10)\n" ++
  "  v13 = call fn2(v18)\n" ++
  "  v14 = iconst.i64 0\n" ++
  "  v15 = icmp eq v2, v14\n" ++
  "  brif v15, block1, block2\n" ++
  "block1:\n" ++
  "  return\n" ++
  "block2:\n" ++
  s!"  v16 = load.i32 notrap aligned v0+{outBufOff + 0x10}\n" ++
  "  v17 = call fn0(v18, v16, v1, v2)\n" ++
  "  return\n" ++
  "}\n"

def compile {inputs : Nat} (spec : PersistentKernel inputs) : CompileResult :=
  let ptx := ptxSource spec
  let ptxBytes := ptx.toUTF8.toList ++ [0]
  let bindDesc := ((List.range (inputs + 1)).map fun i => uint32ToBytes (UInt32.ofNat i)).foldr (· ++ ·) []
  let memSize := spec.bindDescOff + bindDesc.length + 0x100
  let buildInitialMemory : List UInt8 :=
    let reserved := zeros spec.ptxSourceOff
    let ptxBlock := ptxBytes ++ zeros (spec.bindDescOff - spec.ptxSourceOff - ptxBytes.length)
    let bind := bindDesc ++ zeros (memSize - spec.bindDescOff - bindDesc.length)
    reserved ++ ptxBlock ++ bind
  let mkAlg (src : UInt32) : Algorithm := {
    actions := mkCallActions src
    cranelift_units := 0
    timeout_ms := some spec.timeoutMs
  }
  {
    config := {
      cranelift_ir := clifNoopFn ++ "\n" ++ clifLoadFn inputs ++ "\n" ++ clifPrepFn inputs ++ "\n" ++ clifInferFn inputs spec
      memory_size := memSize
      context_offset := 0
      initial_memory := buildInitialMemory
    }
    loadAlgorithm := mkAlg 1
    prepAlgorithm := mkAlg 2
    inferAlgorithm := mkAlg 3
  }

def Tensor.compileTo {n : Nat} (tensor : Tensor n) (out : Nat) (h : out < n := by decide) :
    CompileResult :=
  compile (tensor.writeTo out h)

end CudaTensor

end AlgorithmLib
