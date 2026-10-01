module
public import AlgorithmLib.ML.Kernel.Rewrite
meta import AlgorithmLib.ML.Kernel.Rewrite
public import AlgorithmLib.ML.Kernel.Library
meta import AlgorithmLib.ML.Kernel.Library
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The model's vocabulary

What a tensor operation names without saying how it is launched: a buffer
reference, how a row pass addresses an operand, the lane owning an address,
and which implementation a contraction gets --- a proven warp kernel or a
vendor routine the library knows, with the laws that choice assumes. The
lowering reads these; the model is written in them.
-/

namespace AlgorithmLib.ML

/-- A buffer number. -/
abbrev Ref := Nat

/-- How a row pass addresses one of its operands.

    Every mode builds its address out of the block index and the trip offset
    and nothing else — no division, no remainder, no branch.  That is the whole
    reason a broadcast needs no extension to the index language: the row comes
    from `%ctaid`, so it never has to be recovered from a flat address. -/
inductive BCast where
  /-- `cta·stride + off + o` — a row of a tensor whose rows are `stride` wide,
      starting `off` into it.  `off` is what makes a rotation addressable. -/
  | rowOf    : (stride off : Nat) → BCast
  /-- `cta` — one value per row.  A norm's statistic is read this way. -/
  | scalar   : BCast
  /-- `off + o` — one vector shared by every row: a gain, a rotation table. -/
  | sharedAt : Nat → BCast
  /-- `k` — one address, the same for every row and every lane.  A router's
      gate for one expert is read this way. -/
  | constAt  : Nat → BCast
deriving DecidableEq, Repr

/-- A number for the digest, so a graph comparison sees the addressing. -/
def BCast.tag : BCast → Nat
  | .rowOf s k  => 3 * (s + 1) * (k + 1)
  | .scalar     => 1
  | .sharedAt k => 3 * (k + 1) + 2
  | .constAt k  => 3 * (k + 1) + 1

def BCast.ix : BCast → IdxE
  | .rowOf s k  => stride32 (.add (.mul .ctaId (.lit s)) (.lit k))
  | .scalar     => .ctaId
  | .sharedAt k => stride32 (.lit k)
  | .constAt k  => .lit k

def BCast.ev : BCast → Nat → Nat → Nat
  | .rowOf s k  => fun cta o => cta * s + k + o
  | .scalar     => fun cta _ => cta
  | .sharedAt k => fun _ o => k + o
  | .constAt k  => fun _ _ => k

theorem BCast.ix_ev (m : BCast) (cta j : Nat) (l : Lane) :
    m.ix.eval cta j l (fun _ _ => 0) (fun _ _ => 0) = m.ev cta (j * 32 + l.val) := by
  cases m <;> rfl

/-- **The same addressing, `d` elements further into the buffer.**

    What buffer fusion needs: if several operations' operands are laid end to end
    in one buffer, a reader of the `p`th reaches it by shifting its base.  Every
    mode carries a constant that does exactly that — except `.scalar`, whose
    address *is* the chunk index with nowhere to put one, so shifting it is
    refused rather than approximated.  `.rowOf 1 d` is not a substitute: it adds
    the intra-chunk offset that `.scalar` deliberately drops.

    A row statistic is what `.scalar` reads, and those are the smallest buffers
    in a tape, so refusing costs little. -/
def BCast.shift : BCast → Nat → Option BCast
  | m,            0 => some m
  | .rowOf s k,   d => some (.rowOf s (k + d))
  | .sharedAt k,  d => some (.sharedAt (k + d))
  | .constAt k,   d => some (.constAt (k + d))
  | .scalar,      _ => none

/-- …and it addresses exactly `d` further along. -/
theorem BCast.shift_ev (m : BCast) (d : Nat) (m' : BCast) (h : m.shift d = some m')
    (cta o : Nat) : m'.ev cta o = m.ev cta o + d := by
  cases d with
  | zero => cases h; simp
  | succ e =>
    cases m with
    | rowOf s k => cases h; show cta * s + (k + (e+1)) + o = cta * s + k + o + (e+1); omega
    | sharedAt k => cases h; show k + (e+1) + o = k + o + (e+1); omega
    | constAt k => cases h; show k + (e+1) = k + (e+1); rfl
    | scalar => exact absurd h (by simp [BCast.shift])

/-- The lane that owns address `a` under `elemIx` addressing. -/
def laneMod (a : Nat) : Lane := ⟨a % 32, Nat.mod_lt _ (by decide)⟩

theorem laneMod_of {cta a : Nat} {l : Lane} (h : cta * 32 + l.val = a) : l = laneMod a := by
  have : l.val < 32 := l.isLt
  exact Fin.ext (by show l.val = a % 32; omega)

/-- **A vendor kernel the library knows about.**

    Not a string.  A call site names a constructor, and the symbol to emit, the
    laws the call *assumes*, and the guarantees it explicitly does **not** make
    are read off that constructor by the functions below — associated once,
    here, where they can be checked against the vendor's documentation, rather
    than at each call site where they would be a comment.

    Adding a backend is one constructor and three lines.  Pairing a symbol with
    the wrong law is not expressible. -/
inductive VendorKernel where
  | cublasSgemv
  | cublasSgemvOnStream
  | cublasSgemm
  | cublasSgemmStridedBatched
  | cublasSgemmStridedBatchedOnStream
  /-- **The strided-batched call at a batch of one**: zero strides, zero
      operand offsets, the leading dimensions the shape implies.  In that
      configuration there are no stride or batch arguments left uninterpreted
      and the call is a plain GEMM — measured bit-identical to
      `cl_cublas_sgemm` on all fifteen shapes the ViT launches — so there is an
      equation to state, and `Law.cublasGemmIsSomeReassoc` states it.

      A constructor of its own rather than a note on the general form, because
      the difference is exactly what may be assumed: a call in this
      configuration is billed a law, and one outside it is billed nothing.
      Which one the emitted program makes is decided, not asserted
      (`vit_contractions_are_plain_gemms`). -/
  | cublasSgemmAt1
  | cublasSgemmAt1OnStream
  /-- **A batch named by pointer.**  `P` contractions of one shape in one
      launch, their operands named by three device arrays of pointers rather
      than by a base and a stride — so the members need share no allocation and
      each member's output stays a contiguous matrix of its own. -/
  | cublasSgemmBatchedOnStream
  /-- Store one buffer's device pointer into an entry of such an array. -/
  | cublasPtrArray
  /-- **A contraction whose operands are bf16**, accumulated and returned at
      `Float32`.  Decode is bound by what a matvec reads from device memory, and
      a bf16 weight halves it.

      A constructor of its own because the dtype is exactly what is withheld:
      the products are formed at the operands' precision, not at `Float32`, so
      even the weak "sums its own products in some association" reading is false
      of the `Float32` values the model holds.  Both operands are narrowed --
      cuBLAS refuses a mixed pair -- so an activation reaching this call has been
      rounded too. -/
  | cublasGemmExBf16
  /-- The same contraction, once per batch member at a fixed stride.

      What attention wants when the key cache is bf16: one member per KV head,
      the query narrowed to match because cuBLAS refuses a mixed pair. Withholds
      what `cublasGemmExBf16` withholds, for the same reason. -/
  | cublasGemmStridedBatchedExBf16
  /-- The host→device copy.  Not a contraction and not cuBLAS, but declared for
      the same reason: its source is host memory, which this model does not
      describe, so what it lands is assumed while where it can land is proven. -/
  | uploadPtr
  /-- **A captured graph, replayed.**  Not a kernel: one driver call that
      issues a whole recorded sequence.  Declared for the same reason the
      others are — what it lands is assumed, and where it can land is proven —
      except that "where" is a list of buffers rather than one. -/
  | cudaGraphLaunch
  deriving Repr, DecidableEq

/-- **The FFI symbol codegen emits.**  One per constructor, and every one of
    them is a symbol `base/src/jit.rs` registers — which is the check a free
    string could not make. -/
def VendorKernel.symbol : VendorKernel → String
  | .cublasSgemv                      => "cl_cublas_sgemv"
  | .cublasSgemvOnStream              => "cl_cublas_sgemv_on_stream"
  | .cublasSgemm                      => "cl_cublas_sgemm"
  | .cublasSgemmStridedBatched        => "cl_cublas_sgemm_strided_batched"
  | .cublasSgemmStridedBatchedOnStream =>
      "cl_cublas_sgemm_strided_batched_on_stream"
  | .cublasSgemmAt1                    => "cl_cublas_sgemm_strided_batched"
  | .cublasSgemmAt1OnStream            =>
      "cl_cublas_sgemm_strided_batched_on_stream"
  | .cublasSgemmBatchedOnStream        => "cl_cublas_sgemm_batched_on_stream"
  | .cublasPtrArray                    => "cl_cublas_ptr_array"
  | .cublasGemmExBf16                  => "cl_cublas_gemm_ex_bf16"
  | .cublasGemmStridedBatchedExBf16    => "cl_cublas_gemm_strided_batched_ex_bf16"
  | .uploadPtr                        => "cl_cuda_upload_ptr"
  | .cudaGraphLaunch                  => "cl_cuda_graph_launch"

/-- **Which stated propositions the call's correctness rests on.**

    A `Law` is an equation something can be rewritten by, so a kernel appears
    here only when such an equation exists for it.  `Law.cublasIsMatvec` says
    what a GEMV lands, and `cublasStep_isMatvec` is where a plan uses it.

    The batched entry points get `[]`, and that is **not** "assumed nothing" —
    it is the stronger position of assumed *without a stated equation*, which
    `lawless` below records.  Reading an empty list as exactness is the mistake
    `Plan.lawlessCount` exists to prevent. -/
def VendorKernel.assumes : VendorKernel → List Law
  | .cublasSgemv                       => [.cublasIsMatvec]
  | .cublasSgemvOnStream               => [.cublasIsMatvec]
  | .cublasSgemm                       => [.cublasIsMatvec]
  -- Batched strided mode picks its operand slices from four stride and batch
  -- arguments the launch model recovers as constants but does not interpret,
  -- and the kernel is free to contract multiply-add pairs, split the
  -- contraction, and accumulate at a different width.  No equation over
  -- `Float32` survives that, so none is stated.
  | .cublasSgemmStridedBatched         => []
  | .cublasSgemmStridedBatchedOnStream => []
  -- The same symbol at a batch of one, where the strides are zero and nothing
  -- is left uninterpreted: each output element sums its own products in some
  -- association.  The weak form, for the reason the GEMV's is weak.
  | .cublasSgemmAt1                    => [.cublasGemmIsSomeReassoc]
  | .cublasSgemmAt1OnStream            => [.cublasGemmIsSomeReassoc]
  -- The pointer-array form is the batched call that *does* state an equation,
  -- and it is the weak one: member `p` sums `p`'s own products in some
  -- association.  The strong reading was measured false, so what a batch buys
  -- in launches it pays for in the closed form it gives up.  It names its
  -- members by pointer, so unlike the strided form there are no stride
  -- arguments left uninterpreted and each member's output is a whole matrix.
  | .cublasSgemmBatchedOnStream        => [.cublasBatchedIsSomeReassoc]
  -- Stores a device pointer.  It performs no arithmetic on model values, so
  -- there is no equation to state; which buffers a batch reaches is decided by
  -- these stores and is proven of the emitting program, the same standing a
  -- bind array has.
  | .cublasPtrArray                    => []
  -- The operands are bf16, so the products are not the products of the
  -- `Float32` values this model holds: an equation over them would be false
  -- before any question of fold order arose.  Nothing is stated, and `lawless`
  -- records that it is the strong kind of nothing.
  | .cublasGemmExBf16                  => []
  | .cublasGemmStridedBatchedExBf16    => []
  -- What the copy lands is `uploadedValue`, an opaque on the declared trust
  -- surface, rather than an equation this development states.
  | .uploadPtr                         => []
  -- A replay performs the *recorded* sequence.  Which stages those are is
  -- proven (`qwen_capture_records_the_run`); that replay performs them is the
  -- driver's contract, and no equation here states it.
  | .cudaGraphLaunch                   => []

/-- **Every primitive the enum knows.**

    `all_covers` is what lets a report be derived from this rather than from a
    list typed beside it: a constructor added without an entry here fails to
    build, so a bill computed by filtering `all` cannot quietly omit one. -/
def VendorKernel.all : List VendorKernel :=
  [.cublasSgemv, .cublasSgemvOnStream, .cublasSgemm, .cublasSgemmStridedBatched,
   .cublasSgemmStridedBatchedOnStream, .cublasSgemmAt1, .cublasSgemmAt1OnStream,
   .cublasSgemmBatchedOnStream, .cublasPtrArray, .cublasGemmExBf16,
   .cublasGemmStridedBatchedExBf16, .uploadPtr, .cudaGraphLaunch]

theorem VendorKernel.all_covers : ∀ k : VendorKernel, k ∈ VendorKernel.all := by
  intro k; cases k <;> decide

/-- **Whether the call is assumed with no stated equation behind it.**

    The companion to `assumes`: a kernel with an empty law list is either
    covered elsewhere (the upload, by `uploadedValue` on the trust surface) or
    covered nowhere at all.  This distinguishes the two, so a report cannot
    read silence as exactness. -/
def VendorKernel.lawless : VendorKernel → Bool
  | .cublasSgemmStridedBatched         => true
  | .cublasSgemmStridedBatchedOnStream => true
  | .cublasGemmExBf16                  => true
  | .cublasGemmStridedBatchedExBf16    => true
  | .cudaGraphLaunch                   => true
  | _                                  => false

/-- **What the call does *not* guarantee**, in the same breath as what it does.

    A reordering is correct relative to the laws that are declared; this names
    the ones that are deliberately absent, so "cuBLAS is fine here" is a claim
    with a stated scope rather than a shrug. -/
def VendorKernel.withholds : VendorKernel → String
  | .cudaGraphLaunch =>
      "replays the sequence recorded at capture. Which sequence that is, is " ++
      "proven of the capturing program; that the driver replays it is a " ++
      "guarantee of CUDA graphs and is assumed here, in the same standing as " ++
      "`cl_cuda_launch` running the module at the offset it is given."
  | .uploadPtr =>
      "host→device copy; the source is host memory, which this model does not " ++
      "describe. Framed, so it cannot touch any other buffer."
  | .cublasSgemmBatchedOnStream =>
      "may pick a different algorithm for a batch of P than for P separate " ++
      "calls, and measurably does: the same forward batched and unbatched " ++
      "differs from itself in the last bits. So no closed form is claimed for " ++
      "a batch member, only that it sums that member's own products in some " ++
      "association -- `Law.cublasBatchedIsSomeReassoc`."
  | .cublasPtrArray =>
      "stores a device pointer, so it decides which buffers a batch reaches. " ++
      "That is proven of the emitting program rather than assumed; what is " ++
      "assumed is that a device pointer does not move, which is why the " ++
      "arrays are filled once at load and never rewritten."
  | .cublasSgemmAt1 | .cublasSgemmAt1OnStream =>
      "at a batch of one the strides are gone, so the operands are the whole " ++
      "matrices and an equation is available -- but the weak one: each output " ++
      "element sums its own products in SOME association. Still outside it, " ++
      "as for every vendor call here: fusing a multiply-add into one rounding " ++
      "and accumulating at another width. And the law is about the row-major " ++
      "reading of the operands; which of them plays which role at a " ++
      "column-major call is the lowering's business, not the law's."
  | .cublasGemmExBf16 =>
      "the operands are bf16, so the products are not the products of the " ++
      "Float32 values this model holds: eight mantissa bits, and an activation " ++
      "reaching the call has been rounded too, because cuBLAS refuses a mixed " ++
      "pair. Even the weak `some association` reading is therefore false here, " ++
      "and none is claimed -- what lands in the output is assumed outright, " ++
      "and only where it can land is proven. What the rounding buys is the " ++
      "read: half the bytes per weight, on the path that is bound by exactly " ++
      "that. A model whose published weights are bf16 is being served at the " ++
      "precision it was released in, but that is a fact about the checkpoint, " ++
      "not a theorem this development proves."
  | .cublasGemmStridedBatchedExBf16 =>
      "everything `cublasGemmExBf16` withholds, and the batching on top of it: " ++
      "the members are laid out by a stride rather than by an allocation each, " ++
      "so nothing here says the strides carve out regions that do not overlap. " ++
      "For attention they do -- one member per key head -- and that is " ++
      "arithmetic in the caller rather than a theorem."
  | .cublasSgemmStridedBatched | .cublasSgemmStridedBatchedOnStream =>
      "batched strided mode selects operand slices by stride and batch-count " ++
      "arguments this model does not interpret, and may contract multiply-add " ++
      "pairs, split the contraction, or accumulate at another width. No " ++
      "Float32 equation is available, so none is claimed: what lands in the " ++
      "output is assumed outright, and only where it can land is proven."
  | _ =>
      "NVIDIA pins no fold order at any version, so no exact-Float32 claim is " ++
      "available: `Law.sumAssoc` is *not* granted and results may differ from " ++
      "the proven kernel's committed order."

/-- Which implementation an operation gets.  `vendor` names a kernel the
    library knows, so the laws it assumes and the guarantees it withholds travel
    with the choice into any report. -/
inductive Backend where
  | proven : Backend
  | vendor : VendorKernel → Backend
  deriving Repr, DecidableEq

/-- The laws a lowering rests on: one entry per vendor call, none for proven
    steps.  This is the build-time answer to "which assumptions did this
    schedule make". -/
def Backend.laws : Backend → List Law
  | .proven   => []
  | .vendor k => k.assumes

end AlgorithmLib.ML
