module
public import AlgorithmLib.Host.Driver
meta import AlgorithmLib.Host.Driver
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# cuBLAS, called directly

What each `CublasFn` does, over the device state the driver shares. A handle
runs its routines on the stream it was last given, and a routine is one
device operation there: it reads its operands' allocations, writes its
output's, and its result is the vendor oracle's, the same `VendorCall` the
engine's own cuBLAS entry points make. The oracle names the routine and its
scalars; operands are allocations with element offsets into them.

**Refused and undefined.** What cuBLAS checks and refuses — an unknown
operation, a negative dimension, a leading dimension shorter than its operand
— is answered with `CUBLAS_STATUS_INVALID_VALUE`. What it does not check —
an operand that runs past its allocation, an output that overlaps an input,
a stream or handle already destroyed — has no answer. The model states less
than cuBLAS defines in a few places (empty products, strides other than one,
an output sharing an allocation with an input), and gives those no answer
either; the contracts tag them.
-/

namespace AlgorithmLib.HProg.Sem

open AlgorithmLib.IR
open AlgorithmLib.HProg

/-- `CUBLAS_STATUS_SUCCESS`, `CUBLAS_STATUS_INVALID_VALUE`. -/
def blasOk : Int := 0
def blasInvalid : Int := 7

def kBlas : Nat := 5

/-- The stream party a live handle runs on, while that stream is live. -/
def Dev.blasParty? (d : Dev) (h : UInt64) : Option Nat := do
  let i ← handleOf kBlas h
  let s ← (d.blas[i]?).join
  d.streamParty? s

/-- An operand of `e`-byte elements spanning `span` elements from `p`: its
    allocation and the element it starts at. -/
def Dev.operand? (d : Dev) (p : UInt64) (e span : Nat) : Option (Nat × Nat) := do
  let (id, off, _) ← d.range? p (e * span)
  if off % e == 0 then some (id, off / e) else none

/-- Whether an operation code is one cuBLAS knows: none, transpose, or
    conjugate transpose, which for real data is transpose. -/
def opOk (t : UInt64) : Bool := asI32 t == 0 || asI32 t == 1 || asI32 t == 2

/-- The stored shapes of `A` (`ra × ca`) and `B` (`rb × cb`), column-major. -/
def gemmDims (ta tb m n k : UInt64) : Nat × Nat × Nat × Nat :=
  let (M, N, K) := ((asI32 m).toNat, (asI32 n).toNat, (asI32 k).toNat)
  let (ra, ca) := if asI32 ta != 0 then (K, M) else (M, K)
  let (rb, cb) := if asI32 tb != 0 then (N, K) else (K, N)
  (ra, ca, rb, cb)

/-- What cuBLAS refuses in a product's arguments. -/
def gemmInvalid (ta tb m n k lda ldb ldc batch : UInt64) : Bool :=
  let (ra, _, rb, _) := gemmDims ta tb m n k
  !opOk ta || !opOk tb || asI32 m < 0 || asI32 n < 0 || asI32 k < 0 || asI32 batch < 0 ||
    asI32 lda < ((max 1 ra : Nat) : Int) || asI32 ldb < ((max 1 rb : Nat) : Int) ||
    asI32 ldc < ((max 1 (asI32 m).toNat : Nat) : Int)

/-- Elements a strided batch reaches from its first: every batch's
    `rows × cols` block at leading dimension `ld`, `stride` apart. -/
def batchSpan (rows cols ld stride batch : Nat) : Nat :=
  (batch - 1) * stride + colMajorSpan rows cols ld

/-- What cuBLAS defines but the model does not state, for a product: an empty
    one, a negative stride, and batches whose outputs overlap, which cuBLAS
    leaves undefined. -/
def gemmUnstated (m n k batch sa sb sc ldc : UInt64) : Bool :=
  asI32 m == 0 || asI32 n == 0 || asI32 k == 0 || asI32 batch == 0 ||
    asI64 sa < 0 || asI64 sb < 0 || asI64 sc < 0 ||
    (decide ((asI32 batch).toNat > 1) &&
      decide ((asI64 sc).toNat < colMajorSpan (asI32 m).toNat (asI32 n).toNat (asI32 ldc).toNat))

/-- The elements each operand of a product reaches, `A`, `B`, `C`. -/
def gemmSpans (ta tb m n k lda ldb ldc sa sb sc batch : UInt64) : Nat × Nat × Nat :=
  let dims := gemmDims ta tb m n k
  let nb := (asI32 batch).toNat
  (batchSpan dims.1 dims.2.1 (asI32 lda).toNat (asI64 sa).toNat nb,
   batchSpan dims.2.2.1 dims.2.2.2 (asI32 ldb).toNat (asI64 sb).toNat nb,
   batchSpan (asI32 m).toNat (asI32 n).toNat (asI32 ldc).toNat (asI64 sc).toNat nb)

/-- A product, strided-batched, of `ea`-, `eb`- and `ec`-byte elements, as the
    oracle's routine `op`: `C = α op(A) op(B) + β C` per batch, with `α` and
    `β` read from host memory. -/
def gemmCall (w : World) (op : String) (ea eb ec : Nat)
    (h ta tb m n k pa A lda sa B ldb sb pb C ldc sc batch : UInt64) : Option (Option V × World) := do
  let d := w.dev
  let p ← d.blasParty? h
  if gemmInvalid ta tb m n k lda ldb ldc batch then some (some (ofInt .i32 blasInvalid), w)
  else if gemmUnstated m n k batch sa sb sc ldc then none
  else do
    let sp := gemmSpans ta tb m n k lda ldb ldc sa sb sc batch
    let (ia, oa) ← d.operand? A ea sp.1
    let (ib, ob) ← d.operand? B eb sp.2.1
    let (ic, oc) ← d.operand? C ec sp.2.2
    -- an output sharing an allocation with an input is not modelled
    if ic == ia || ic == ib then none
    else do
      let alpha ← w.mem.load pa 4
      let beta ← w.mem.load pb 4
      let c : VendorCall := ⟨op, [ta, tb, m, n, k, alpha, beta, sa, sb, sc, batch,
        UInt64.ofNat oa, UInt64.ofNat ob, UInt64.ofNat oc, lda, ldb, ldc]⟩
      devOnly w (do cuRes (← d.devOp w p (.vendor c [ia, ib, ic] ic) [ia, ib, ic] [ic]) blasOk)

/-- What cuBLAS refuses in an `sgemv`'s arguments. -/
def sgemvInvalid (trans m n lda incx incy : UInt64) : Bool :=
  !opOk trans || asI32 m < 0 || asI32 n < 0 || asI32 lda < ((max 1 (asI32 m).toNat : Nat) : Int) ||
    asI32 incx == 0 || asI32 incy == 0

/-- What the model does not state for an `sgemv`: an empty one, strides other
    than one, and a padded `A`, since the oracle reads operands packed. -/
def sgemvUnstated (m n lda incx incy : UInt64) : Bool :=
  asI32 m == 0 || asI32 n == 0 || asI32 incx != 1 || asI32 incy != 1 ||
    (asI32 lda).toNat != (asI32 m).toNat

/-- The lengths of `x` and `y`. -/
def sgemvLens (trans m n : UInt64) : Nat × Nat :=
  if asI32 trans != 0 then ((asI32 m).toNat, (asI32 n).toNat) else ((asI32 n).toNat, (asI32 m).toNat)

/-- `y = α op(A) x + β y`, the oracle's `sgemv`, which reads each operand from
    the start of its allocation. -/
def sgemvCall (w : World) (h trans m n pa A lda x incx pb y incy : UInt64) :
    Option (Option V × World) := do
  let d := w.dev
  let p ← d.blasParty? h
  if sgemvInvalid trans m n lda incx incy then some (some (ofInt .i32 blasInvalid), w)
  else if sgemvUnstated m n lda incx incy then none
  else do
    let (ia, oa) ← d.operand? A 4 ((asI32 m).toNat * (asI32 n).toNat)
    let (ix, ox) ← d.operand? x 4 (sgemvLens trans m n).1
    let (iy, oy) ← d.operand? y 4 (sgemvLens trans m n).2
    if oa != 0 || ox != 0 || oy != 0 || iy == ia || iy == ix then none
    else do
      let alpha ← w.mem.load pa 4
      let beta ← w.mem.load pb 4
      let c : VendorCall := ⟨"sgemv", [trans, m, n, alpha, beta]⟩
      devOnly w (do cuRes (← d.devOp w p (.vendor c [ia, ix, iy] iy) [ia, ix, iy] [iy]) blasOk)

/-- `CUDA_R_16BF`, `CUDA_R_32F`, `CUBLAS_COMPUTE_32F`, and the two default
    algorithms: the one `gemmEx` combination the model states. -/
def bf16In32Out (at_ bt ct compute algo : UInt64) : Bool :=
  asI32 at_ == 14 && asI32 bt == 14 && asI32 ct == 0 && asI32 compute == 68 &&
    (asI32 algo == -1 || asI32 algo == 99)

/-- Entry `i` of a device array of pointers, `off` bytes into its allocation. -/
def ptrEntry (b : ByteArray) (off i : Nat) : UInt64 :=
  ofLe64 (b.extract (off + 8 * i) (off + 8 * i + 8))

/-- One member of a pointer-array batch: the allocation and element offset of
    its `A`, `B` and `C`, each packed. -/
def blasMember (d : Dev) (ta tb m n k : UInt64) (bA bB bC : ByteArray) (oA oB oC i : Nat) :
    Option ((Nat × Nat) × (Nat × Nat) × (Nat × Nat)) := do
  let (ra, ca, rb, cb) := gemmDims ta tb m n k
  let M := (asI32 m).toNat
  let a ← d.operand? (ptrEntry bA oA i) 4 (colMajorSpan ra ca ra)
  let b ← d.operand? (ptrEntry bB oB i) 4 (colMajorSpan rb cb rb)
  let c ← d.operand? (ptrEntry bC oC i) 4 (colMajorSpan M (asI32 n).toNat M)
  some (a, b, c)

/-- Whether a batch's outputs are pairwise apart and apart from every input
    and every array. -/
def batchApart (ms : List ((Nat × Nat) × (Nat × Nat) × (Nat × Nat))) (arrs : List Nat) : Bool :=
  let outs := ms.map (·.2.2.1)
  let ins := ms.flatMap (fun (a, b, _) => [a.1, b.1]) ++ arrs
  outs.eraseDups.length == outs.length && !outs.any (ins.contains ·)

/-- Each member of a batch as one device operation on `p`, in order. -/
def batchOps (w : World) (p : Nat) (ta tb m n k alpha beta : UInt64) (arrs : List Nat) :
    List ((Nat × Nat) × (Nat × Nat) × (Nat × Nat)) → Dev → Option Dev
  | [], d => some d
  | (a, b, c) :: ms, d => do
      let d ← d.devOp w p (.vendor ⟨"sgemmBatchedMember",
          [ta, tb, m, n, k, alpha, beta, UInt64.ofNat (4 * a.2), UInt64.ofNat (4 * b.2),
           UInt64.ofNat (4 * c.2)]⟩ [a.1, b.1, c.1] c.1) (arrs ++ [a.1, b.1, c.1]) [c.1]
      batchOps w p ta tb m n k alpha beta arrs ms d

/-- What the model does not state for a pointer-array batch: an empty one,
    and operands other than packed. -/
def batchUnstated (ta tb m n k lda ldb ldc batch : UInt64) : Bool :=
  let (ra, _, rb, _) := gemmDims ta tb m n k
  asI32 m == 0 || asI32 n == 0 || asI32 k == 0 || asI32 batch == 0 || (asI32 lda).toNat != ra ||
    (asI32 ldb).toNat != rb || (asI32 ldc).toNat != (asI32 m).toNat

/-- A batch of products whose operands are named by device arrays of
    pointers, the oracle's `sgemmBatchedMember` per member. The model states
    packed operands only, reads the arrays when the call is made, and runs the
    members in order, which their being apart makes the only result. -/
def sgemmBatchedCall (w : World) (h ta tb m n k pa PA lda PB ldb pb PC ldc batch : UInt64) :
    Option (Option V × World) := do
  let d := w.dev
  let p ← d.blasParty? h
  if gemmInvalid ta tb m n k lda ldb ldc batch then some (some (ofInt .i32 blasInvalid), w)
  else if batchUnstated ta tb m n k lda ldb ldc batch then none
  else do
      let nb := (asI32 batch).toNat
      let (iA, oA, bA) ← d.range? PA (8 * nb)
      let (iB, oB, bB) ← d.range? PB (8 * nb)
      let (iC, oC, bC) ← d.range? PC (8 * nb)
      let ms ← (List.range nb).mapM (blasMember d ta tb m n k bA bB bC oA oB oC)
      if !batchApart ms [iA, iB, iC] then none
      else do
        let alpha ← w.mem.load pa 4
        let beta ← w.mem.load pb 4
        devOnly w (do cuRes (← batchOps w p ta tb m n k alpha beta [iA, iB, iC] ms d) blasOk)

/-- **What a cuBLAS call does.** `none` where cuBLAS's behaviour is undefined
    or the model does not state it. -/
def cublasCall (f : CublasFn) (bits : List UInt64) (w : World) : Option (Option V × World) :=
  let d := w.dev
  match f, bits with
  | .create, [ph] =>
      -- handles are managed outside any capture
      if !d.live || d.capture.isSome then none
      else do
        let i := d.blas.size
        let m ← w.mem.store ph 8 (drvHandle kBlas i)
        some (some (ofInt .i32 blasOk), { w with mem := m, dev := { d with blas := d.blas.push (some 0) } })
  | .destroy, [h] =>
      -- handles are managed outside any capture
      if !d.live || d.capture.isSome then none
      else do
        let i ← handleOf kBlas h
        let _ ← (d.blas[i]?).join
        some (some (ofInt .i32 blasOk), { w with dev := { d with blas := d.blas.set! i none } })
  | .setStream, [h, s] =>
      -- handles are managed outside any capture
      if !d.live || d.capture.isSome then none
      else do
        let i ← handleOf kBlas h
        let _ ← (d.blas[i]?).join
        let _ ← d.streamParty? s
        some (some (ofInt .i32 blasOk), { w with dev := { d with blas := d.blas.set! i (some s) } })
  | .sgemv, [h, trans, m, n, pa, A, lda, x, incx, pb, y, incy] =>
      if !d.live then none
      else sgemvCall w h trans m n pa A lda x incx pb y incy
  | .sgemm, [h, ta, tb, m, n, k, pa, A, lda, B, ldb, pb, C, ldc] =>
      if !d.live then none
      else gemmCall w "sgemmStridedBatched" 4 4 4 h ta tb m n k pa A lda 0 B ldb 0 pb C ldc 0 1
  | .sgemmStridedBatched, [h, ta, tb, m, n, k, pa, A, lda, sa, B, ldb, sb, pb, C, ldc, sc, batch] =>
      if !d.live then none
      else gemmCall w "sgemmStridedBatched" 4 4 4 h ta tb m n k pa A lda sa B ldb sb pb C ldc sc batch
  | .gemmEx, [h, ta, tb, m, n, k, pa, A, at_, lda, B, bt, ldb, pb, C, ct, ldc, compute, algo] =>
      if !d.live || !bf16In32Out at_ bt ct compute algo then none
      else gemmCall w "gemmStridedBatchedExBf16" 2 2 4 h ta tb m n k pa A lda 0 B ldb 0 pb C ldc 0 1
  | .gemmStridedBatchedEx,
      [h, ta, tb, m, n, k, pa, A, at_, lda, sa, B, bt, ldb, sb, pb, C, ct, ldc, sc, batch, compute, algo] =>
      if !d.live || !bf16In32Out at_ bt ct compute algo then none
      else gemmCall w "gemmStridedBatchedExBf16" 2 2 4 h ta tb m n k pa A lda sa B ldb sb pb C ldc sc batch
  | .sgemmBatched, [h, ta, tb, m, n, k, pa, PA, lda, PB, ldb, pb, PC, ldc, batch] =>
      if !d.live then none
      else sgemmBatchedCall w h ta tb m n k pa PA lda PB ldb pb PC ldc batch
  | _, _ => none

end AlgorithmLib.HProg.Sem
