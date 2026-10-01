module
public import AlgorithmLib.Host.Sem
meta import AlgorithmLib.Host.Sem
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic

@[expose] public section

/-!
# cuBLAS, as the interpreter's vendor oracle

What the CUDA corpora take each cuBLAS routine to compute, column-major with
cuBLAS's leading-dimension defaults. The corpora feed it small integers only,
so every summation order gives the same exact answer and the oracle's order is
not a claim about cuBLAS's.
-/

open AlgorithmLib
open AlgorithmLib.HProg

namespace HProgVendorRef

def f32At (b : ByteArray) (i : Nat) : Float32 :=
  Float32.ofBits ((List.range 4).foldr (fun j acc => (acc <<< 8) ||| (b.get! (4 * i + j)).toUInt32) 0)

def putF32 (b : ByteArray) (i : Nat) (x : Float32) : ByteArray :=
  (List.range 4).foldl (fun b j => b.set! (4 * i + j) ((x.toBits >>> (8 * j.toUInt32)).toUInt8)) b

def i32 (x : UInt64) : Nat := (Sem.asI32 x).toNat
def flt (x : UInt64) : Float32 := Float32.ofBits x.toUInt32

/-- `y ← alpha·op(A)·x + beta·y`, `A` is `m × n` column-major with `lda = m`. -/
def sgemvRef (trans m n alpha beta : UInt64) (A X Y : ByteArray) : ByteArray :=
  let (M, N) := (i32 m, i32 n)
  let (rows, cols) := if Sem.asI32 trans != 0 then (N, M) else (M, N)
  (List.range rows).foldl (fun y i =>
    let acc := (List.range cols).foldl (fun acc j =>
      let aij := if Sem.asI32 trans != 0 then f32At A (i * M + j) else f32At A (j * M + i)
      acc + aij * f32At X j) 0
    putF32 y i (flt alpha * acc + flt beta * f32At Y i)) Y

/-- A `bf16` element: the high half of an `f32`. -/
def bf16At (b : ByteArray) (i : Nat) : Float32 :=
  Float32.ofBits (((b.get! (2 * i)).toUInt32 ||| ((b.get! (2 * i + 1)).toUInt32 <<< 8)) <<< 16)

/-- A strided-batched GEMM, every batch, reading `A` and `B` with `rd`. -/
def gemmRefWith (rd : ByteArray → Nat → Float32) (sc : List UInt64) (A B C : ByteArray) : ByteArray :=
  match sc with
  | [ta, tb, m, n, k, alpha, beta, sa, sb, sc', batch, oa, ob, oc, la, lb, lc] =>
      let (M, N, K) := (i32 m, i32 n, i32 k)
      let lda := if Sem.asI32 la != 0 then i32 la else if Sem.asI32 ta != 0 then K else M
      let ldb := if Sem.asI32 lb != 0 then i32 lb else if Sem.asI32 tb != 0 then N else K
      let ldc := if Sem.asI32 lc != 0 then i32 lc else M
      (List.range (i32 batch)).foldl (fun C bt =>
        let a0 := (Sem.asI64 oa).toNat + bt * (Sem.asI64 sa).toNat
        let b0 := (Sem.asI64 ob).toNat + bt * (Sem.asI64 sb).toNat
        let c0 := (Sem.asI64 oc).toNat + bt * (Sem.asI64 sc').toNat
        (List.range M).foldl (fun C i => (List.range N).foldl (fun C j =>
          let acc := (List.range K).foldl (fun acc l =>
            let ail := if Sem.asI32 ta != 0 then rd A (a0 + i * lda + l) else rd A (a0 + l * lda + i)
            let blj := if Sem.asI32 tb != 0 then rd B (b0 + l * ldb + j) else rd B (b0 + j * ldb + l)
            acc + ail * blj) 0
          putF32 C (c0 + j * ldc + i) (flt alpha * acc + flt beta * f32At C (c0 + j * ldc + i))) C) C) C
  | _ => C

/-- The strided-batched `sgemm`. -/
def sgemmRef := gemmRefWith f32At

/-- One member of a pointer-array batch: offsets in bytes, leading dimensions
    the defaults. -/
def batchMemberRef (sc : List UInt64) (A B C : ByteArray) : ByteArray :=
  match sc with
  | [ta, tb, m, n, k, alpha, beta, oa, ob, oc] =>
      sgemmRef [ta, tb, m, n, k, alpha, beta, 0, 0, 0, 1, oa / 4, ob / 4, oc / 4, 0, 0, 0] A B C
  | _ => C

/-- `gemmEx` with `bf16` `A` and `B` and an `f32` `C`, one batch. -/
def gemmBf16Ref (sc : List UInt64) (A B C : ByteArray) : ByteArray :=
  match sc with
  | [ta, tb, m, n, k, alpha, beta, oa, ob, oc, la, lb, lc] =>
      let (M, N, K) := (i32 m, i32 n, i32 k)
      let lda := if Sem.asI32 la != 0 then i32 la else if Sem.asI32 ta != 0 then K else M
      let ldb := if Sem.asI32 lb != 0 then i32 lb else if Sem.asI32 tb != 0 then N else K
      let ldc := if Sem.asI32 lc != 0 then i32 lc else M
      let (a0, b0, c0) := ((Sem.asI64 oa).toNat, (Sem.asI64 ob).toNat, (Sem.asI64 oc).toNat)
      (List.range M).foldl (fun C i => (List.range N).foldl (fun C j =>
        let acc := (List.range K).foldl (fun acc l =>
          let ail := if Sem.asI32 ta != 0 then bf16At A (a0 + i * lda + l) else bf16At A (a0 + l * lda + i)
          let blj := if Sem.asI32 tb != 0 then bf16At B (b0 + l * ldb + j) else bf16At B (b0 + j * ldb + l)
          acc + ail * blj) 0
        putF32 C (c0 + j * ldc + i) (flt alpha * acc + flt beta * f32At C (c0 + j * ldc + i))) C) C
  | _ => C

def vendorRef (c : Sem.VendorCall) (ins : List ByteArray) : ByteArray :=
  match c.op, c.scalars, ins with
  | "sgemv", [trans, m, n, alpha, beta], [A, X, Y] => sgemvRef trans m n alpha beta A X Y
  | "sgemmStridedBatched", sc, [A, B, C] => sgemmRef sc A B C
  | "gemmExBf16", sc, [A, B, C] => gemmBf16Ref sc A B C
  | "gemmStridedBatchedExBf16", sc, [A, B, C] => gemmRefWith bf16At sc A B C
  | "sgemmBatchedMember", sc, [A, B, C] => batchMemberRef sc A B C
  | _, _, ins => ins.getLastD ByteArray.empty

/-- `bf16` bytes of small integers: the high half of each `f32`. -/
def bf16Bytes (xs : List Float) : List UInt8 :=
  xs.flatMap fun x =>
    let b : UInt32 := x.toFloat32.toBits >>> (16 : UInt32)
    [b.toUInt8, (b >>> 8).toUInt8]

def f32Bytes (xs : List Float) : List UInt8 :=
  xs.flatMap fun x =>
    let b : UInt32 := x.toFloat32.toBits
    (List.range 4).map fun (j : Nat) => (b >>> (8 * j.toUInt32)).toUInt8

end HProgVendorRef
