module
public import Lean
public import AlgorithmLib.Surface.Link
meta import AlgorithmLib.Surface.Link
public import AlgorithmLib.Surface.Drv
meta import AlgorithmLib.Surface.Drv
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
public import Ann.Kernels
meta import Ann.Kernels
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# base's ANN engine: the host

An artifact per vector shape (dimension and metric). The engine keeps every
row it is given in a store, and an IVF index over the rows that were live
when it was last built; rows inserted since then form a write buffer that
every search scans exactly, and a delete clears the id's bit in the live set,
which the scan, the buffer and the coarse step all read. When the rows
inserted and deleted since the last build reach an eighth of the rows it
indexed, or the live rows have doubled, the index is built again over the
live rows.

Its functions (the caller's data, then what each does):

* `init`: the capacity in rows (`u64`): the store, the live sets, once.
* `insert`: a row count (`u32`, then four bytes), the rows (`f32 × d` each),
  their ids (`i64` each, below the capacity): appended to the store.
* `delete`: an id count (`u32`, four bytes), the ids (`i64`).
* `commit`: builds the index if there is none.
* `search`: four words (`queries`, `nprobe`, `R`, `k`), the queries; the
  output is `queries × k` ids (`i64`), `-1` where fewer were found.

**A build.** The list count is a power of two near `2 √n`, at most `n / 40`;
ten rounds of k-means on a strided sample of `256` rows a list (cuBLAS
products against `-2 C`, the nearest kernel, a radix sort by centre and each
centre's mean); every row's list; the per-dimension scale from the largest
residual; the rows laid out by list by a radix sort, and encoded.

**A search.** The coarse products are one cuBLAS product; the probes a
select over the coarse keys (a list with no live row masked out); the (query,
list) pairs sorted by list; the lists scanned by the fused kernel; each
query's best `R` selected; the buffer's rows scored by one product and each
query's best `k` of them selected; all of them re-ranked exactly.

Every buffer is the driver's, made when a call first needs it and grown when
a later call needs more (`ensure`); each kernel's scalar parameters are words
of one device block (`METAD`), staged in the arena and copied up.
-/

set_option maxRecDepth 8000

namespace AnnHost

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.Prog AlgorithmLib.Kernel AlgorithmLib.Kernel.Block

/-! ## The layout -/

/-- The allocations, by index. `XI` and `IDSI` are views: the index's rows
    and their ids, the store itself or the gathered live rows (`XL`,
    `IDSL`). -/
def X : Nat := 0
def IDS : Nat := 1
def CENT : Nat := 2
def CN : Nat := 3
def TILE : Nat := 4
def ASG : Nat := 5
def K0 : Nat := 6
def V0 : Nat := 7
def K1 : Nat := 8
def V1 : Nat := 9
def HIST : Nat := 10
def ST : Nat := 11
def EN : Nat := 12
def PART : Nat := 13
def SCALE : Nat := 14
def POS : Nat := 15
def LIST : Nat := 16
def CODES : Nat := 17
def RTERM : Nat := 18
def LOFF : Nat := 19
def LEND : Nat := 20
def Q : Nat := 21
def DOTS : Nat := 22
def PQ : Nat := 23
def PBASE : Nat := 24
def TQ : Nat := 25
def TQS : Nat := 26
def OKEY : Nat := 27
def OROW : Nat := 28
def SEL : Nat := 29
def OUT : Nat := 30
def METAD : Nat := 31
def VNEAR : Nat := 32
def XI : Nat := 33
def IDSI : Nat := 34
def XL : Nat := 35
def IDSL : Nat := 36
def LID : Nat := 37
def LIVE : Nat := 38
def MASK : Nat := 39
def FLAG : Nat := 40
def POSL : Nat := 41
def NRM : Nat := 42
def BDOTS : Nat := 43
def BSEL : Nat := 44
/-- The inverted tag index: the item → tags list given (`TIND`, `TTAG`), and
    each tag's items sorted (`TITEMS`). -/
def TIND : Nat := 45
def TTAG : Nat := 46
def TITEMS : Nat := 47
/-- A filtered search's plan, per query: its shortest list's start and
    length, its other lists (`FOPTR` into `FOLO`, `FOHI`), where its admitted
    items go; the admitted items, their count, the queries answered exactly
    and those scanned, all the queries, the scanned ones' admitted bitsets,
    each id's stored row, and the answers. -/
def FQLO : Nat := 48
def FQLEN : Nat := 49
def FOPTR : Nat := 50
def FOLO : Nat := 51
def FOHI : Nat := 52
def FOFF : Nat := 53
def ADM : Nat := 54
def ADMCNT : Nat := 55
def SIDX : Nat := 56
def BIDX : Nat := 57
def QA : Nat := 58
def QBITS : Nat := 59
def POSOF : Nat := 60
def OUTF : Nat := 61
/-- Kernel `j`'s scalar parameters: a view into `METAD`. -/
def M (j : Nat) : Nat := 62 + j

/-- The kernels as the host launches them, by index. -/
def kNEAR : Nat := 0
def kIOTA : Nat := 1
def kHIST0 : Nat := 2
def kSCAN : Nat := 3
def kSCAT0 : Nat := 4
def kHIST1 : Nat := 5
def kSCAT1 : Nat := 6
def kSEG : Nat := 7
def kCENTRE : Nat := 8
def kGATHER : Nat := 9
def kRESMAX : Nat := 10
def kSCALE : Nat := 11
def kENC : Nat := 12
def kCOARSE : Nat := 13
def kSELP : Nat := 14
def kPAIRS : Nat := 15
def kQUANT : Nat := 16
def kSCANK : Nat := 17
def kSELR : Nat := 18
def kRR1 : Nat := 19
def kRR2 : Nat := 20
def kRR4 : Nat := 21
def kLLIVE : Nat := 22
def kFLAG : Nat := 23
def kSCANF : Nat := 24
def kSCATL : Nat := 25
def kGIDX : Nat := 26
def kNORMS : Nat := 27
def kBKEY : Nat := 28
def kSELB : Nat := 29
def kLID : Nat := 30
def kTPAIRS : Nat := 31
def kADMIT : Nat := 32
def kEXACT : Nat := 33
def kSETBITS : Nat := 34
def kGROWS : Nat := 35
def kPOSOF : Nat := 36
/-- The scan with 256-entry buffers (`R > 96`), and the re-rank of 256. -/
def kSCANK2 : Nat := 37
def kRR8 : Nat := 38
/-- The scans of few queries' pieces of pairs (at both buffer sizes), and
    their coarse keys without the product. -/
def kSCANS : Nat := 39
def kSCANS2 : Nat := 40
def kCD : Nat := 41
def NSPEC : Nat := 42

/-- Admitted items at or below which a filtered query is answered exactly
    over them; above it, by the scan with its admitted bitset. -/
def FILTER_EXACT : Nat := 20000

/-- Arena offsets: the host's words, the staged scalar parameters (64 bytes a
    kernel), constants, then the module's text. -/
def STATE_OFF : Nat := 0x8000
def META_OFF : Nat := 0x9000
def CONST_OFF : Nat := 0xA000
/-- Each kernel's module's offset in the arena, a word each. -/
def MODTAB_OFF : Nat := 0xA100
def PTX_OFF : Nat := 0xA400

/-- The host's words; then each allocation's capacity in bytes. -/
def S_INIT : Nat := 0
/-- Rows in the store. -/
def S_N : Nat := 1
def S_LISTS : Nat := 2
/-- The store's capacity in rows. -/
def S_CAP : Nat := 3
/-- Rows the index holds (`0`: no index). -/
def S_TRAINED : Nat := 4
/-- The first store row past the index: the buffer is `B0 … N`. -/
def S_B0 : Nat := 5
/-- Rows inserted and deleted since the index was built. -/
def S_CHURN : Nat := 6
/-- Live rows. -/
def S_LIVE : Nat := 7
/-- Whether a delete came since the index was built (the scan checks). -/
def S_DIRTY : Nat := 8
/-- Whether a delete ever came (the store is no longer all live). -/
def S_EVERDEL : Nat := 9
/-- The host's copy of the live set, and its bytes. -/
def S_HBITS : Nat := 10
def S_BITBYTES : Nat := 11
/-- Tags, and the host's copy of each tag's run in `TITEMS`. -/
def S_NTAGS : Nat := 12
def S_HTST : Nat := 13
def S_HTEN : Nat := 14
/-- The host's staging for a filtered search's plan, and its bytes. -/
def S_HF : Nat := 15
def S_HFCAP : Nat := 16
/-- Whether an insert or delete came since the index was last kept up
    (`maintain`). -/
def S_PEND : Nat := 17
def capOf (b : Nat) : Nat := 24 + b

/-- Kernel specs for an engine of dimension `d` and metric `l2`: entry name,
    the kernel's index in `kernelsFor` (its module), the allocations it is
    handed, threads a block. -/
def specs (d : Nat) (l2 : Bool) : List ((String × Nat) × List Nat × Nat) :=
  let ks := AnnKernels.kernelsFor d l2
  let nm (i : Nat) : String × Nat := ((ks[i]?.map (·.name)).getD "", i)
  -- kernelsFor's order: 0 nearest, 1 scan, 2 hist, 3 exscan, 4 scatter, 5 iota, 6 segments,
  -- 7 rowSelect, 8 coarse, 9 pairs, 10 quant, 11 encode, 12 resMax, 13 scale, 14 centre,
  -- 15 gather, 16-18 rerank 1/2/4, 19 listLive, 20 liveFlag, 21 scatterLive, 22 gatherIdx,
  -- 23 norms, 24 bufKey, 25 lid, 26 tagPairs, 27 admit, 28 exact, 29 setBits, 30 gatherRows, 31 posOf,
  -- 32 scan (256-entry buffers), 33 rerank 8, 34 coarseDirect, 35-36 scanOwn (128, 256)
  let rr := [XI, POS, IDSI, OROW, SEL, Q, OUT, X, IDS, BSEL, M kRR1]
  let own := [CODES, RTERM, LOFF, LEND, TQ, TQS, K0, DOTS, CN, PBASE, OUT, OKEY, OROW, LID, LIVE, QBITS, M kSCANS]
  [ (nm 0, [TILE, CN, VNEAR, M kNEAR], 256)
  , (nm 5, [V0, M kIOTA], 256)
  , (nm 2, [K0, HIST, M kHIST0], 32)
  , (nm 3, [HIST, M kSCAN], 256)
  , (nm 4, [K0, V0, K1, V1, HIST, M kHIST0], 32)
  , (nm 2, [K1, HIST, M kHIST1], 32)
  , (nm 4, [K1, V1, K0, V0, HIST, M kHIST1], 32)
  , (nm 6, [K0, ST, EN, M kSEG], 256)
  , (nm 14, [XI, V0, ST, EN, CENT, CN, M kCENTRE], 256)
  , (nm 15, [XI, CENT, M kGATHER], 256)
  , (nm 12, [XI, ASG, CENT, PART, M kRESMAX], 256)
  , (nm 13, [PART, SCALE, M kSCALE], 256)
  , (nm 11, [XI, POS, LIST, CENT, SCALE, CODES, RTERM, M kENC], 256)
  , (nm 8, [DOTS, CN, MASK, M kCOARSE], 256)
  , (nm 7, [DOTS, K0, M kSELP], 256)
  , (nm 9, [K0, V0, DOTS, CN, PQ, PBASE, M kPAIRS], 256)
  , (nm 10, [Q, SCALE, TQ, TQS, M kQUANT], 256)
  , (nm 1, [CODES, RTERM, LOFF, LEND, TQ, TQS, ST, EN, PQ, PBASE, V0, OKEY, OROW, LID, LIVE, QBITS, M kSCANK], 32)
  , (nm 7, [OKEY, SEL, M kSELR], 256)
  , (nm 16, rr, 256)
  , (nm 17, rr, 256)
  , (nm 18, rr, 256)
  , (nm 19, [LOFF, LEND, LID, LIVE, MASK], 256)
  , (nm 20, [IDS, LIVE, FLAG, M kFLAG], 256)
  , (nm 3, [FLAG, M kSCANF], 256)
  , (nm 21, [IDS, LIVE, FLAG, POSL, M kSCATL], 256)
  , (nm 22, [X, IDS, POSL, XL, IDSL], 256)
  , (nm 23, [X, NRM, M kNORMS], 256)
  , (nm 24, [BDOTS, NRM, IDS, LIVE, M kBKEY], 256)
  , (nm 7, [BDOTS, BSEL, M kSELB], 256)
  , (nm 25, [IDSI, POS, LID, M kLID], 256)
  , (nm 26, [TIND, TTAG, K0, V0, M kTPAIRS], 256)
  , (nm 27, [TITEMS, FQLO, FQLEN, FOPTR, FOLO, FOHI, FOFF, ADM, ADMCNT], 256)
  , (nm 28, [X, POSOF, QA, ADM, FOFF, ADMCNT, SIDX, OUTF, LIVE, M kEXACT], 256)
  , (nm 29, [ADM, FOFF, ADMCNT, BIDX, QBITS, M kSETBITS], 256)
  , (nm 30, [QA, BIDX, Q], 256)
  , (nm 31, [IDS, POSOF, M kPOSOF], 256)
  , (nm 32, [CODES, RTERM, LOFF, LEND, TQ, TQS, ST, EN, PQ, PBASE, V0, OKEY, OROW, LID, LIVE, QBITS, M kSCANK], 32)
  , (nm 33, rr, 256)
  , (nm 35, own, 32)
  , (nm 36, own, 32)
  , (nm 34, [Q, CENT, CN, MASK, DOTS, M kCD], 256) ]

/-- The program in the driver's standard layout; every allocation is made at
    run time. Each kernel is its own module, so each has only the shared
    memory it asks for (one module's shared array is its largest kernel's);
    a spec's `ptxOff` names its kernel's module (`PTX_OFF + index`), and the
    modules' real offsets are a table in the image (`MODTAB_OFF`). `memSize`
    is the arena's size, which only the image needs. -/
def app (d : Nat) (l2 : Bool) (memSize : Nat := PTX_OFF) : Drv.App :=
  { bufs := List.replicate (M NSPEC) 0,
    kernels := (specs d l2).map fun ((_, i), args, blk) => ⟨PTX_OFF + i, 0, args, (1, 1, 1), (blk, 1, 1), []⟩,
    memSize, blas := true, org := 0, cells := 24,
    entries := (specs d l2).map (·.1.1) }

/-! ## Host helpers -/

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

def word (base : V .i64) (i : Nat) : Prog V L (V .i64) := iaddImm base (STATE_OFF + 8 * i)
def getW (base : V .i64) (i : Nat) : Prog V L (V .i64) := do load64 (← word base i)
def setW (base : V .i64) (i : Nat) (v : V .i64) : Prog V L Unit := do storeI64 v (← word base i)

/-- The device pointer allocation `b` holds. -/
def ptrOf (a : Drv.App) (base : V .i64) (b : Nat) : Prog V L (V .i64) := do
  load64 (← iaddImm base (a.bufSlot b))

/-- Point the view `b` at `p`. -/
def setPtr (a : Drv.App) (base : V .i64) (b : Nat) (p : V .i64) : Prog V L Unit := do
  storeI64 p (← iaddImm base (a.bufSlot b))

/-- Allocation `b` at least `bytes` long: made, or freed and made larger
    (its contents are not kept). -/
def ensure (a : Drv.App) (base : V .i64) (b : Nat) (bytes : V .i64) : Prog V L Unit := do
  let cap ← getW base (capOf b)
  when .ugt bytes cap do
    when .ne cap (← iconst64 0) do
      let _ ← ext (.cuda .memFree) %[← ptrOf a base b]
    let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot b), bytes]
    setW base (capOf b) bytes

/-- Scalar word `w` of kernel `j`'s parameters, staged in the arena. -/
def metaW (base : V .i64) (j w : Nat) (v : V .i32) : Prog V L Unit := do
  storeI32 v (← iaddImm base (META_OFF + 64 * j + 4 * w))

/-- Kernel `j`'s staged parameters, copied up. -/
def upMeta (a : Drv.App) (base : V .i64) (j : Nat) : Prog V L Unit := do
  let dst ← iaddImm (← ptrOf a base METAD) (64 * j)
  let _ ← ext (.cuda .memcpyHtoD) %[dst, ← iaddImm base (META_OFF + 64 * j), ← iconst64 64]

/-- Every kernel's meta block, by one copy. -/
def upMetaAll (a : Drv.App) (base : V .i64) : Prog V L Unit := do
  let _ ← ext (.cuda .memcpyHtoD) %[← ptrOf a base METAD, ← iaddImm base META_OFF, ← iconst64 (64 * NSPEC)]

/-- Every kernel's parameter block written again from the allocations'
    slots: after any allocation or view moved. -/
def rebindAll (a : Drv.App) (base : V .i64) : Prog V L Unit := do
  for j in List.range NSPEC do a.rebind base j

/-- `bytes` of allocation `src` into allocation `dst`, on the device. -/
def copyD (a : Drv.App) (base : V .i64) (dst src : Nat) (bytes : V .i64) : Prog V L Unit := do
  let _ ← ext (.cuda .memcpyDtoD) %[← ptrOf a base dst, ← ptrOf a base src, bytes]

/-- Zero `bytes` of allocation `b`. -/
def zero (a : Drv.App) (base : V .i64) (b : Nat) (bytes : V .i64) : Prog V L Unit := do
  let _ ← ext (.cuda .memsetD8) %[← ptrOf a base b, ← iconst32 0, bytes]

/-- `⌈x / y⌉` in 32 bits. -/
def cdiv (x : V .i64) (y : Nat) : Prog V L (V .i32) := do
  ireduce32 (← udiv (← iaddImm x (y - 1)) (← iconst64 y))

def lo32 (x : V .i64) : Prog V L (V .i32) := ireduce32 x

/-- `C = α op(A) op(B)` by cuBLAS with explicit leading dimensions:
    `op(A)` is `m × k`, `op(B)` `k × n`, `C` `m × n` column-major. -/
def gemm (h : V .i64) (ta tb : Nat) (m n k : V .i32) (alpha A : V .i64) (lda : V .i32) (B : V .i64)
    (ldb : V .i32) (beta C : V .i64) (ldc : V .i32) : Prog V L Unit := do
  let z ← iconst64 0
  let _ ← ext (.cublas .sgemmStridedBatched) %[h, ← iconst32 ta, ← iconst32 tb, m, n, k, alpha,
    A, lda, z, B, ldb, z, beta, C, ldc, z, ← iconst32 1]

/-- The radix sort's meta words, for `n` keys (not yet on the device). -/
def radixMeta (base : V .i64) (n : V .i64) : Prog V L Unit := do
  let blocks ← cdiv n AnnKernels.RADIX_E
  let n32 ← lo32 n
  metaW base kHIST0 0 n32; metaW base kHIST0 1 (← iconst32 0); metaW base kHIST0 2 blocks
  metaW base kHIST1 0 n32; metaW base kHIST1 1 (← iconst32 8); metaW base kHIST1 2 blocks
  metaW base kSCAN 0 (← imul blocks (← iconst32 256))

/-- The radix sort's launches, its meta words on the device. -/
def radixRun (a : Drv.App) (base : V .i64) (n : V .i64) : Prog V L Unit := do
  let blocks ← cdiv n AnnKernels.RADIX_E
  let one ← iconst32 1
  a.runX base kHIST0 blocks
  a.runX base kSCAN one
  a.runX base kSCAT0 blocks
  a.runX base kHIST1 blocks
  a.runX base kSCAN one
  a.runX base kSCAT1 blocks

/-- **The stable radix sort** of the `n` keys in `K0` with their values in
    `V0`, by the keys' low 16 bits, back into `K0`, `V0`: two 8-bit passes
    through `K1`, `V1`. -/
def radixSort (a : Drv.App) (base : V .i64) (n : V .i64) : Prog V L Unit := do
  radixMeta base n
  upMeta a base kHIST0; upMeta a base kHIST1; upMeta a base kSCAN
  radixRun a base n

/-- **The stable radix sort by 24 bits**: two passes as `radixSort`, a third
    by the keys' bits 16 … 23 through `K1`, `V1`, then back into `K0`, `V0`. -/
def radixSort24 (a : Drv.App) (base : V .i64) (n : V .i64) : Prog V L Unit := do
  radixSort a base n
  let blocks ← cdiv n AnnKernels.RADIX_E
  metaW base kHIST0 0 (← lo32 n); metaW base kHIST0 1 (← iconst32 16); metaW base kHIST0 2 blocks
  upMeta a base kHIST0
  a.runX base kHIST0 blocks
  a.runX base kSCAN (← iconst32 1)
  a.runX base kSCAT0 blocks
  let b ← ishlImm n 2
  copyD a base K0 K1 b
  copyD a base V0 V1 b

/-- `V0[i] := i` for `i < n`. -/
def iotaV0 (a : Drv.App) (base : V .i64) (n : V .i64) : Prog V L Unit := do
  metaW base kIOTA 0 (← lo32 n)
  upMeta a base kIOTA
  a.runX base kIOTA (← cdiv n 256)

/-- Each key's run in the `n` sorted keys of `K0`, into `ST`, `EN` (`lists`
    words each, zero where absent). -/
def runs (a : Drv.App) (base : V .i64) (n lists : V .i64) : Prog V L Unit := do
  let bytes ← ishlImm lists 2
  zero a base ST bytes
  zero a base EN bytes
  metaW base kSEG 0 (← lo32 n)
  upMeta a base kSEG
  a.runX base kSEG (← cdiv n 256)

/-- **Each of `rows` rows' nearest centre**, index rows `i · stride` for
    `i < rows`, into `ASG`: products against `-2 C` a slice at a time (at most
    `2²⁶` products a slice), then the nearest kernel. -/
def assign (a : Drv.App) (base : V .i64) (d : Nat) (rows stride lists : V .i64) : Prog V L Unit := do
  let h ← load64 (← iaddImm base a.BLAS)
  let lists32 ← lo32 lists
  let slice ← umax (← iconst64 1) (← umin rows (← udiv (← iconst64 (2 ^ 26)) lists))
  let ldb ← lo32 (← imul stride (← iconst64 d))
  let _ ← wloop %[← iconst64 0] (fun cs => pure (contIfULt cs.head rows, %[], ()))
    (fun cs _ => do
      let start := cs.head
      let cnt ← umin slice (← isub rows start)
      let bPtr ← iadd (← ptrOf a base XI) (← ishlImm (← imul (← imul start stride) (← iconst64 d)) 2)
      gemm h 1 0 lists32 (← lo32 cnt) (← iconst32 d) (← iaddImm base CONST_OFF) (← ptrOf a base CENT)
        (← iconst32 d) bPtr ldb (← iaddImm base a.ZERO) (← ptrOf a base TILE) lists32
      setPtr a base VNEAR (← iadd (← ptrOf a base ASG) (← ishlImm start 2))
      a.rebind base kNEAR
      metaW base kNEAR 0 (← lo32 cnt); metaW base kNEAR 1 lists32
      upMeta a base kNEAR
      a.runX base kNEAR (← cdiv cnt 8)
      pure %[← iadd start cnt])

/-- The live set copied up from the host's copy. -/
def upLive (a : Drv.App) (base : V .i64) : Prog V L Unit := do
  let _ ← ext (.cuda .memcpyHtoD) %[← ptrOf a base LIVE, ← getW base S_HBITS, ← getW base S_BITBYTES]

/-- Each list's mask from the live set: `+∞` for a list with no live row. -/
def refreshMask (a : Drv.App) (base : V .i64) : Prog V L Unit := do
  a.runX base kLLIVE (← lo32 (← getW base S_LISTS))

/-- Bring the driver and cuBLAS up, load the module, make the parameters'
    block and point each kernel's parameter view into it. Once. -/
def up (a : Drv.App) (base : V .i64) : Prog V L Unit := do
  Drv.up base a.CTX
  let _ ← ext (.cublas .create) %[← iaddImm base a.BLAS]
  for (j, k) in a.indexed do
    if a.modOwner j == j then
      let off ← load64 (← iaddImm base (MODTAB_OFF + 8 * (k.ptxOff - PTX_OFF)))
      let _ ← ext (.cuda .moduleLoadData) %[← iaddImm base (a.modSlot j), ← iadd base off]
    Drv.getFunction base (a.kern j k)
  let _ ← ext (.cuda .memAlloc) %[← iaddImm base (a.bufSlot METAD), ← iconst64 (64 * NSPEC)]
  for j in List.range NSPEC do
    setPtr a base (M j) (← iaddImm (← ptrOf a base METAD) (64 * j))

/-- **The index, built** over the `n` index rows `XI` (ids `IDSI`). -/
def indexBuild (a : Drv.App) (base : V .i64) (d : Nat) (n : V .i64) : Prog V L Unit := do
  -- lists: the largest power of two whose square is at most 8n, at most n / 40
  let p ← wloop %[← iconst64 1] (fun cs => do
      let two ← ishlImm cs.head 1
      pure (contIfULe (← imul two two) (← imul n (← iconst64 8)), %[cs.head], ()))
    (fun cs _ => do pure %[← ishlImm cs.head 1])
  let lists ← umax (← iconst64 1) (← umin p.head (← udiv n (← iconst64 40)))
  setW base S_LISTS lists
  let dB ← iconst64 (4 * d)
  let nB ← ishlImm n 2
  let listsB ← ishlImm lists 2
  let blocks ← udiv (← iaddImm n (AnnKernels.RADIX_E - 1)) (← iconst64 AnnKernels.RADIX_E)
  let dw := (d + 3) / 4
  ensure a base CENT (← imul lists dB)
  ensure a base CN listsB
  ensure a base TILE (← iconst64 (2 ^ 28))
  ensure a base ASG nB
  for b in [K0, V0, K1, V1, POS, LIST, RTERM, LID] do ensure a base b nB
  ensure a base HIST (← imul blocks (← iconst64 1024))
  for b in [ST, EN, LOFF, LEND, MASK] do ensure a base b listsB
  ensure a base PART (← imul blocks dB)
  ensure a base SCALE dB
  ensure a base CODES (← imul (← ishlImm (← udiv (← iaddImm n 31) (← iconst64 32)) 5) (← iconst64 (4 * dw)))
  rebindAll a base
  -- k-means on the strided sample
  let smp ← umin n (← imul lists (← iconst64 256))
  let stride ← umax (← iconst64 1) (← udiv n smp)
  metaW base kGATHER 0 (← lo32 (← imul (← udiv smp lists) stride))
  upMeta a base kGATHER
  a.runX base kGATHER (← lo32 lists)
  metaW base kCENTRE 0 (← lo32 stride)
  upMeta a base kCENTRE
  zero a base ST listsB
  zero a base EN listsB
  a.runX base kCENTRE (← lo32 lists)
  for _ in List.range 10 do
    assign a base d smp stride lists
    copyD a base K0 ASG (← ishlImm smp 2)
    iotaV0 a base smp
    radixSort a base smp
    runs a base smp lists
    a.runX base kCENTRE (← lo32 lists)
  -- every row's list, the scale, the layout and the codes
  assign a base d n (← iconst64 1) lists
  metaW base kRESMAX 0 (← lo32 n); metaW base kRESMAX 1 (← iconst32 AnnKernels.RADIX_E)
  metaW base kSCALE 0 (← lo32 blocks)
  upMeta a base kRESMAX; upMeta a base kSCALE
  a.runX base kRESMAX (← lo32 blocks)
  a.runX base kSCALE (← iconst32 1)
  copyD a base K0 ASG nB
  iotaV0 a base n
  radixSort a base n
  runs a base n lists
  copyD a base LOFF ST listsB
  copyD a base LEND EN listsB
  copyD a base LIST K0 nB
  copyD a base POS V0 nB
  metaW base kENC 0 (← lo32 n)
  metaW base kLID 0 (← lo32 n)
  upMeta a base kENC; upMeta a base kLID
  a.runX base kENC (← cdiv n 256)
  a.runX base kLID (← cdiv n 256)
  refreshMask a base

/-- **A build** over the live rows: the store itself while no row was ever
    deleted, otherwise the live rows gathered (`XL`). -/
def rebuild (a : Drv.App) (base : V .i64) (d : Nat) : Prog V L Unit := do
  let n ← getW base S_N
  let rows ← ifte (jTys := [.i64]) .eq (← getW base S_EVERDEL) (← iconst64 0)
    (thn := do
      setPtr a base XI (← ptrOf a base X)
      setPtr a base IDSI (← ptrOf a base IDS)
      pure %[n])
    (els := do
      let live ← getW base S_LIVE
      ensure a base XL (← imul (← umax live (← iconst64 1)) (← iconst64 (4 * d)))
      ensure a base IDSL (← ishlImm (← umax live (← iconst64 1)) 3)
      rebindAll a base
      metaW base kFLAG 0 (← lo32 n); metaW base kSCANF 0 (← lo32 n); metaW base kSCATL 0 (← lo32 n)
      upMeta a base kFLAG; upMeta a base kSCANF; upMeta a base kSCATL
      a.runX base kFLAG (← cdiv n 256)
      a.runX base kSCANF (← iconst32 1)
      a.runX base kSCATL (← cdiv n 256)
      a.runX base kGIDX (← lo32 live)
      setPtr a base XI (← ptrOf a base XL)
      setPtr a base IDSI (← ptrOf a base IDSL)
      pure %[live])
  indexBuild a base d rows.head
  -- each id's stored row, for the exact answers of a filtered search
  ensure a base POSOF (← ishlImm (← getW base S_CAP) 2)
  rebindAll a base
  metaW base kPOSOF 0 (← lo32 n)
  upMeta a base kPOSOF
  a.runX base kPOSOF (← cdiv n 256)
  setW base S_TRAINED rows.head
  setW base S_B0 n
  setW base S_CHURN (← iconst64 0)
  setW base S_DIRTY (← iconst64 0)
  setW base S_PEND (← iconst64 0)

/-- A build when the index has moved past its data: the live rows doubled,
    or the churn reached an eighth of the rows it indexed. -/
def maybeRebuild (a : Drv.App) (base : V .i64) (d : Nat) : Prog V L Unit := do
  let t ← getW base S_TRAINED
  when .ne t (← iconst64 0) do
    let doubled ← icmp .ugt (← getW base S_LIVE) (← ishlImm t 1)
    let churned ← icmp .uge (← ishlImm (← getW base S_CHURN) 3) t
    let moved ← bor (← select doubled (← iconst64 1) (← iconst64 0)) (← select churned (← iconst64 1) (← iconst64 0))
    let _ ← ifte .ne moved (← iconst64 0)
      (thn := do rebuild a base d; pure %[])
      (els := do refreshMask a base; pure %[])
    pure ()

/-- **The index kept up**, once for every insert and delete since it last
    was: rebuilt if it has moved past its data, else its empty lists marked.
    An insert or delete only records itself; this runs at the next commit or
    search. -/
def maintain (a : Drv.App) (base : V .i64) (d : Nat) : Prog V L Unit := do
  when .ne (← getW base S_PEND) (← iconst64 0) do
    maybeRebuild a base d
    setW base S_PEND (← iconst64 0)

end

/-! ## The functions -/

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

/-- `1` where `R` is a size the scan's buffers take: at least one, and with
    room for a trip's 32 appends past it in the larger buffer. -/
def rOk (rR : V .i64) : Prog V L (V .i64) := do
  band (← select (← icmp .uge rR (← iconst64 1)) (← iconst64 1) (← iconst64 0))
    (← select (← icmp .ule rR (← iconst64 (AnnKernels.CAP2 - 32))) (← iconst64 1) (← iconst64 0))

/-- `body`'s status where `ok`, else `1`. -/
def heldThen (ok : V .i8) (body : Prog V L (V .i64)) : Prog V L (V .i64) := do
  let r ← ifte (jTys := [.i64]) .ne (← select ok (← iconst64 1) (← iconst64 0)) (← iconst64 0)
    (thn := do pure %[← body]) (els := do pure %[← iconst64 1])
  pure r.head

/-- `body`'s status where the input holds its `hdr` header bytes, else `1`. -/
def needs (hdr : Nat) (body : Prog V L (V .i64)) : Prog V L (V .i64) := do
  let r ← ifte (jTys := [.i64]) .uge (← dataLen) (← iconst64 hdr)
    (thn := do pure %[← body]) (els := do pure %[← iconst64 1])
  pure r.head

def initCode (d : Nat) (l2 : Bool) : Prog V L (V .i64) := do
  let a := app d l2
  let base ← basePtr
  -- made once, from an input that holds the capacity word
  let fresh ← band (← select (← icmp .eq (← getW base S_INIT) (← iconst64 0)) (← iconst64 1) (← iconst64 0))
    (← select (← icmp .uge (← dataLen) (← iconst64 8)) (← iconst64 1) (← iconst64 0))
  when .ne fresh (← iconst64 0) do
    up a base
    let cap ← load64 (← dataPtr)
    setW base S_CAP cap
    let bitBytes ← ishlImm (← udiv (← iaddImm cap 31) (← iconst64 32)) 2
    setW base S_BITBYTES bitBytes
    ensure a base X (← imul (← umax cap (← iconst64 1)) (← iconst64 (4 * d)))
    ensure a base IDS (← ishlImm (← umax cap (← iconst64 1)) 3)
    ensure a base LIVE bitBytes
    ensure a base FLAG (← ishlImm (← umax cap (← iconst64 1)) 2)
    ensure a base POSL (← ishlImm (← umax cap (← iconst64 1)) 2)
    ensure a base BSEL (← iconst64 4)
    let _ ← ext (.cuda .memAllocHost) %[← word base S_HBITS, bitBytes]
    let hb ← getW base S_HBITS
    forLoop (← ushrImm bitBytes 2) fun i => do
      storeI32 (← iconst32 0) (← iadd hb (← ishlImm i 2))
    upLive a base
    rebindAll a base
    setW base S_INIT (← iconst64 1)
  iconst64 0

def insertCode (d : Nat) (l2 : Bool) : Prog V L (V .i64) := do
  let a := app d l2
  let base ← basePtr
  let dp ← dataPtr
  needs 8 do
    let cnt ← uextend64 (← load32 dp)
    let n ← getW base S_N
    let cap ← getW base S_CAP
    let vecs ← iaddImm dp 8
    let vB ← imul cnt (← iconst64 (4 * d))
    let ids ← iadd vecs vB
    -- taken only whole: the input holds the rows and ids it counts, they fit
    -- the capacity, and every id is a row of it (the live bits are by id)
    let held ← icmp .ule (← iaddImm (← imul cnt (← iconst64 (4 * d + 8))) 8) (← dataLen)
    let maxId ← ifte (jTys := [.i64]) .ne (← select held (← iconst64 1) (← iconst64 0)) (← iconst64 0)
      (thn := do pure %[← forLoopAcc cnt (← iconst64 0) fun i acc => do
        umax acc (← iaddImm (← load64 (← iadd ids (← ishlImm i 3))) 1)])
      (els := do pure %[← iaddImm cap 1])
    let ok ← band (← select (← icmp .ule (← iadd n cnt) cap) (← iconst64 1) (← iconst64 0))
      (← select (← icmp .ule maxId.head cap) (← iconst64 1) (← iconst64 0))
    let st ← ifte (jTys := [.i64]) .ne ok (← iconst64 0)
      (thn := do
        let _ ← ext (.cuda .memcpyHtoD) %[← iadd (← ptrOf a base X) (← imul n (← iconst64 (4 * d))), vecs, vB]
        let _ ← ext (.cuda .memcpyHtoD) %[← iadd (← ptrOf a base IDS) (← ishlImm n 3), ids, ← ishlImm cnt 3]
        let hb ← getW base S_HBITS
        forLoop cnt fun i => do
          let id ← load64 (← iadd ids (← ishlImm i 3))
          let w ← iadd hb (← ishlImm (← ushrImm id 5) 2)
          let bitv ← ishl (← iconst32 1) (← ireduce32 (← band id (← iconst64 31)))
          storeI32 (← bor (← load32 w) bitv) w
        upLive a base
        setW base S_N (← iadd n cnt)
        setW base S_LIVE (← iadd (← getW base S_LIVE) cnt)
        when .ne (← getW base S_TRAINED) (← iconst64 0) do
          setW base S_CHURN (← iadd (← getW base S_CHURN) cnt)
          setW base S_PEND (← iconst64 1)
        pure %[← iconst64 0])
      (els := do pure %[← iconst64 1])
    pure st.head

def deleteCode (d : Nat) (l2 : Bool) : Prog V L (V .i64) := do
  let a := app d l2
  let base ← basePtr
  let dp ← dataPtr
  needs 8 do
    let cnt ← uextend64 (← load32 dp)
    let ids ← iaddImm dp 8
    let hb ← getW base S_HBITS
    -- taken only whole: the input holds the ids it counts, each within the
    -- capacity (the live bits are by id)
    let held ← icmp .ule (← iaddImm (← ishlImm cnt 3) 8) (← dataLen)
    let maxId ← ifte (jTys := [.i64]) .ne (← select held (← iconst64 1) (← iconst64 0)) (← iconst64 0)
      (thn := do pure %[← forLoopAcc cnt (← iconst64 0) fun i acc => do
        umax acc (← iaddImm (← load64 (← iadd ids (← ishlImm i 3))) 1)])
      (els := do pure %[← iaddImm (← getW base S_CAP) 1])
    let st ← ifte (jTys := [.i64]) .ule maxId.head (← getW base S_CAP)
      (thn := do
        forLoop cnt fun i => do
          let id ← load64 (← iadd ids (← ishlImm i 3))
          let w ← iadd hb (← ishlImm (← ushrImm id 5) 2)
          let bitv ← ishl (← iconst32 1) (← ireduce32 (← band id (← iconst64 31)))
          storeI32 (← bandNot (← load32 w) bitv) w
        upLive a base
        setW base S_LIVE (← isub (← getW base S_LIVE) cnt)
        setW base S_EVERDEL (← iconst64 1)
        when .ne (← getW base S_TRAINED) (← iconst64 0) do
          setW base S_CHURN (← iadd (← getW base S_CHURN) cnt)
          setW base S_DIRTY (← iconst64 1)
          setW base S_PEND (← iconst64 1)
        pure %[← iconst64 0])
      (els := do pure %[← iconst64 1])
    pure st.head

def commitCode (d : Nat) (l2 : Bool) : Prog V L (V .i64) := do
  let a := app d l2
  let base ← basePtr
  when .eq (← getW base S_TRAINED) (← iconst64 0) do
    when .ne (← getW base S_N) (← iconst64 0) do
      rebuild a base d
  maintain a base d
  Drv.sync
  iconst64 0

/-- The most queries a search takes pair by pair (`kSCANS`), and the pieces
    it cuts each pair's rows into. -/
def SMALL_Q : Nat := 8
def SPLIT : Nat := 4

/-- **The search of the `nq` queries in `Q`** into `OUT` (`nq × k` ids):
    `qwords` words of admitted bitset a query in `QBITS` (a filtered search),
    `0` for none. -/
def searchCore (a : Drv.App) (base : V .i64) (d : Nat) (nq nprobe0 rR k0 qwords : V .i64) : Prog V L Unit := do
  let lists ← getW base S_LISTS
  let nprobe ← umin nprobe0 lists
  let k ← umin k0 rR
  let n ← getW base S_N
  let b0 ← getW base S_B0
  let nb ← isub n b0
  let kb ← umin k nb
  let pairs ← imul nq nprobe
  let small ← icmp .ule nq (← iconst64 SMALL_Q)
  -- the scan's slots: a piece of a pair each where the search is small
  let slots ← select small (← imul pairs (← iconst64 SPLIT)) pairs
  let cC ← imul (← select small (← imul nprobe (← iconst64 SPLIT)) nprobe) rR
  let dw := (d + 3) / 4
  let pB ← ishlImm pairs 2
  let blocks ← udiv (← iaddImm pairs (AnnKernels.RADIX_E - 1)) (← iconst64 AnnKernels.RADIX_E)
  ensure a base DOTS (← ishlImm (← imul nq lists) 2)
  for b in [K0, V0, K1, V1] do ensure a base b pB
  for b in [PQ, PBASE] do ensure a base b pB
  ensure a base HIST (← imul blocks (← iconst64 1024))
  ensure a base TQ (← imul nq (← iconst64 (4 * dw)))
  ensure a base TQS (← ishlImm nq 2)
  ensure a base OKEY (← ishlImm (← imul slots rR) 2)
  ensure a base OROW (← ishlImm (← imul slots rR) 2)
  ensure a base SEL (← ishlImm (← imul nq rR) 2)
  ensure a base OUT (← ishlImm (← imul nq k) 3)
  ensure a base NRM (← ishlImm (← umax nb (← iconst64 1)) 2)
  ensure a base BDOTS (← ishlImm (← umax (← imul nq nb) (← iconst64 1)) 2)
  ensure a base BSEL (← ishlImm (← umax (← imul nq kb) (← iconst64 1)) 2)
  ensure a base QBITS (← iconst64 4)
  rebindAll a base
  let nq32 ← lo32 nq
  let lists32 ← lo32 lists
  let h ← load64 (← iaddImm base a.BLAS)
  metaW base kCOARSE 0 (← lo32 (← imul nq lists)); metaW base kCOARSE 1 lists32
  metaW base kSELP 0 lists32; metaW base kSELP 1 (← lo32 nprobe)
  metaW base kPAIRS 0 (← lo32 pairs); metaW base kPAIRS 1 (← lo32 nprobe); metaW base kPAIRS 2 lists32
  metaW base kQUANT 0 nq32
  metaW base kSCANK 0 (← lo32 rR); metaW base kSCANK 1 (← lo32 (← getW base S_DIRTY)); metaW base kSCANK 2 (← lo32 qwords)
  metaW base kSELR 0 (← lo32 cC); metaW base kSELR 1 (← lo32 rR)
  metaW base kRR1 0 (← lo32 cC); metaW base kRR1 1 (← lo32 rR); metaW base kRR1 2 (← lo32 k)
  metaW base kRR1 3 (← lo32 kb); metaW base kRR1 4 (← lo32 b0)
  let nb32 ← lo32 nb
  metaW base kNORMS 0 (← lo32 b0); metaW base kNORMS 1 nb32
  metaW base kBKEY 0 (← lo32 (← imul nq nb)); metaW base kBKEY 1 nb32; metaW base kBKEY 2 (← lo32 b0)
  metaW base kSELB 0 nb32; metaW base kSELB 1 (← lo32 kb)
  metaW base kIOTA 0 (← lo32 pairs)
  radixMeta base pairs
  metaW base kSEG 0 (← lo32 pairs)
  metaW base kSCANS 0 (← lo32 rR); metaW base kSCANS 1 (← lo32 (← getW base S_DIRTY))
  metaW base kSCANS 2 (← lo32 qwords); metaW base kSCANS 3 (← lo32 nprobe); metaW base kSCANS 4 lists32
  metaW base kSCANS 5 (← iconst32 SPLIT)
  metaW base kCD 0 nq32; metaW base kCD 1 lists32
  upMetaAll a base
  let _ ← ifte .ne (← select small (← iconst64 1) (← iconst64 0)) (← iconst64 0)
    (thn := do a.runX base kCD (← cdiv (← imul nq lists) AnnKernels.WARPS); pure %[])
    (els := do
      gemm h 1 0 lists32 nq32 (← iconst32 d) (← iaddImm base a.ONE) (← ptrOf a base CENT) (← iconst32 d)
        (← ptrOf a base Q) (← iconst32 d) (← iaddImm base a.ZERO) (← ptrOf a base DOTS) lists32
      a.runX base kCOARSE (← cdiv (← imul nq lists) 256)
      pure %[])
  a.runX base kSELP nq32
  a.runX base kQUANT (← cdiv nq 8)
  -- few queries: each pair scanned as a list of its own; many: the pairs
  -- sorted by list, so a list's queries share its scan
  let _ ← ifte .ne (← select small (← iconst64 1) (← iconst64 0)) (← iconst64 0)
    (thn := do
      let grid ← lo32 slots
      let _ ← ifte .ule rR (← iconst64 (AnnKernels.CAP - 32))
        (thn := do a.runX base kSCANS grid; pure %[])
        (els := do a.runX base kSCANS2 grid; pure %[])
      pure %[])
    (els := do
      a.runX base kIOTA (← cdiv pairs 256)
      radixRun a base pairs
      zero a base ST (← ishlImm lists 2)
      zero a base EN (← ishlImm lists 2)
      a.runX base kSEG (← cdiv pairs 256)
      a.runX base kPAIRS (← cdiv pairs 256)
      let grid ← imul lists32 (← iconst32 AnnKernels.QW)
      let _ ← ifte .ule rR (← iconst64 (AnnKernels.CAP - 32))
        (thn := do a.runX base kSCANK grid; pure %[])
        (els := do a.runX base kSCANK2 grid; pure %[])
      pure %[])
  a.runX base kSELR nq32
  -- the buffer: every buffered row scored, each query's best `kb`
  when .ne kb (← iconst64 0) do
    a.runX base kNORMS (← cdiv nb 8)
    gemm h 1 0 nb32 nq32 (← iconst32 d) (← iaddImm base a.ONE)
      (← iadd (← ptrOf a base X) (← imul b0 (← iconst64 (4 * d)))) (← iconst32 d)
      (← ptrOf a base Q) (← iconst32 d) (← iaddImm base a.ZERO) (← ptrOf a base BDOTS) nb32
    a.runX base kBKEY (← cdiv (← imul nq nb) 256)
    a.runX base kSELB nq32
  let tot ← iadd rR kb
  let _ ← ifte .ule tot (← iconst64 32)
    (thn := do a.runX base kRR1 nq32; pure %[])
    (els := do
      let _ ← ifte .ule tot (← iconst64 64)
        (thn := do a.runX base kRR2 nq32; pure %[])
        (els := do
          let _ ← ifte .ule tot (← iconst64 128)
            (thn := do a.runX base kRR4 nq32; pure %[])
            (els := do a.runX base kRR8 nq32; pure %[])
          pure %[])
      pure %[])

/-- The index kept up (`maintain`), built where there is none, and rebuilt
    where rows are buffered and the caller wants only the index
    (`noBuffer`). -/
def ready (a : Drv.App) (base : V .i64) (d : Nat) (noBuffer : Bool) : Prog V L Unit := do
  maintain a base d
  let t ← getW base S_TRAINED
  let n ← getW base S_N
  when .ne n (← iconst64 0) do
    if noBuffer then
      let stale ← bor (← select (← icmp .eq t (← iconst64 0)) (← iconst64 1) (← iconst64 0))
        (← select (← icmp .ne (← getW base S_B0) n) (← iconst64 1) (← iconst64 0))
      when .ne stale (← iconst64 0) do rebuild a base d
    else
      when .eq t (← iconst64 0) do rebuild a base d

def searchCode (d : Nat) (l2 : Bool) : Prog V L (V .i64) := do
  let a := app d l2
  let base ← basePtr
  let dp ← dataPtr
  needs 16 do
    ready a base d false
    let nq ← uextend64 (← load32 dp)
    let k ← umin (← uextend64 (← load32 (← iaddImm dp 12))) (← uextend64 (← load32 (← iaddImm dp 8)))
    -- answered only where the input holds its queries, the output its ids,
    -- `R` a size the scan's buffers take (`rOk`), and `k` at least one
    let ok ← band (← band (← rOk (← uextend64 (← load32 (← iaddImm dp 8))))
        (← select (← icmp .uge k (← iconst64 1)) (← iconst64 1) (← iconst64 0)))
      (← band (← select (← icmp .ule (← iaddImm (← imul nq (← iconst64 (4 * d))) 16) (← dataLen)) (← iconst64 1) (← iconst64 0))
        (← select (← icmp .ule (← ishlImm (← imul nq k) 3) (← outLen)) (← iconst64 1) (← iconst64 0)))
    let st ← ifte (jTys := [.i64]) .ne ok (← iconst64 0)
      (thn := do
        ensure a base Q (← imul (← umax nq (← iconst64 1)) (← iconst64 (4 * d)))
        rebindAll a base
        let _ ← ext (.cuda .memcpyHtoD) %[← ptrOf a base Q, ← iaddImm dp 16, ← imul nq (← iconst64 (4 * d))]
        searchCore a base d nq (← uextend64 (← load32 (← iaddImm dp 4))) (← uextend64 (← load32 (← iaddImm dp 8)))
          (← uextend64 (← load32 (← iaddImm dp 12))) (← iconst64 0)
        -- the copy's status is the call's: a fault in any kernel before it shows
        pure %[← uextend64 (← ext (.cuda .memcpyDtoH) %[← outPtr, ← ptrOf a base OUT, ← ishlImm (← imul nq k) 3])])
      (els := do pure %[← iconst64 1])
    pure st.head

  /-! ## Filtered search -/

/-- **The inverted tag index**, from the caller's item → tags list: its item
    count and entry count (`u32` each), then `indptr` (`items + 1` words) and
    the tags (a word each). The (tag, item) pairs are sorted by tag on the
    device (24 bits); each tag's run is kept on the host, for planning. -/
def tagsCode (d : Nat) (l2 : Bool) : Prog V L (V .i64) := do
  let a := app d l2
  let base ← basePtr
  let dp ← dataPtr
  needs 8 do
    let items ← uextend64 (← load32 dp)
    let entries ← uextend64 (← load32 (← iaddImm dp 4))
    let ind ← iaddImm dp 8
    let tg ← iadd ind (← ishlImm (← iaddImm items 1) 2)
    -- taken only where the input holds the index it counts, its items are
    -- rows of the capacity (the kernels read the rows by item), and its tags
    -- fit 24 bits
    let held ← icmp .ule (← iadd (← ishlImm (← iaddImm items 1) 2) (← iaddImm (← ishlImm entries 2) 8)) (← dataLen)
    heldThen (← icmp .ne (← band (← select held (← iconst64 1) (← iconst64 0))
        (← select (← icmp .ule items (← getW base S_CAP)) (← iconst64 1) (← iconst64 0))) (← iconst64 0)) do
      -- tags: one past the largest
      let ntags ← forLoopAcc entries (← iconst64 0) fun i acc => do
        umax acc (← iaddImm (← uextend64 (← load32 (← iadd tg (← ishlImm i 2)))) 1)
      heldThen (← icmp .ule ntags (← iconst64 (2 ^ 24))) do
        setW base S_NTAGS ntags
        let eB ← ishlImm entries 2
        ensure a base TIND (← ishlImm (← iaddImm items 1) 2)
        ensure a base TTAG eB
        ensure a base TITEMS eB
        for b in [K0, V0, K1, V1] do ensure a base b eB
        ensure a base HIST (← imul (← udiv (← iaddImm entries (AnnKernels.RADIX_E - 1)) (← iconst64 AnnKernels.RADIX_E)) (← iconst64 1024))
        ensure a base ST (← ishlImm ntags 2)
        ensure a base EN (← ishlImm ntags 2)
        rebindAll a base
        let _ ← ext (.cuda .memcpyHtoD) %[← ptrOf a base TIND, ind, ← ishlImm (← iaddImm items 1) 2]
        let _ ← ext (.cuda .memcpyHtoD) %[← ptrOf a base TTAG, tg, eB]
        metaW base kTPAIRS 0 (← lo32 entries); metaW base kTPAIRS 1 (← lo32 items)
        upMeta a base kTPAIRS
        a.runX base kTPAIRS (← cdiv entries 256)
        radixSort24 a base entries
        copyD a base TITEMS V0 eB
        runs a base entries ntags
        let tB ← ishlImm ntags 2
        let _ ← ext (.cuda .memAllocHost) %[← word base S_HTST, tB]
        let _ ← ext (.cuda .memAllocHost) %[← word base S_HTEN, tB]
        let _ ← ext (.cuda .memcpyDtoH) %[← getW base S_HTST, ← ptrOf a base ST, tB]
        let _ ← ext (.cuda .memcpyDtoH) %[← getW base S_HTEN, ← ptrOf a base EN, tB]
        iconst64 0

/-- **A filtered search**: four words (`queries`, `nprobe`, `R`, `k`), the
    queries, then each query's required tags as `rptr` (`queries + 1` words)
    and the tags (a word each); the output is `queries × k` ids, the `k`
    nearest items holding every required tag, `-1` where fewer do.

    The host plans each query from the tags' runs: its shortest list, and its
    others. On the device each query's admitted items are enumerated (`admit`);
    a query admitting at most `FILTER_EXACT` is answered exactly over them, a
    larger one by the scan with its admitted ids as a bitset. -/
def searchfCode (d : Nat) (l2 : Bool) : Prog V L (V .i64) := do
  let a := app d l2
  let base ← basePtr
  let dp ← dataPtr
  needs 16 do
    ready a base d true
    let nq ← uextend64 (← load32 dp)
    let nprobe ← uextend64 (← load32 (← iaddImm dp 4))
    let rR ← uextend64 (← load32 (← iaddImm dp 8))
    let k ← umin (← uextend64 (← load32 (← iaddImm dp 12))) rR
    let qp ← iaddImm dp 16
    let rptr ← iadd qp (← imul nq (← iconst64 (4 * d)))
    let rtags ← iadd rptr (← ishlImm (← iaddImm nq 1) 2)
    -- answered only where the input holds the queries, `rptr` and the tags it
    -- counts, the output the ids, and `rptr` starts at 0 with every query
    -- naming a tag (the plan counts each query's others by that)
    let tagsAt ← isub rtags dp
    heldThen (← icmp .ule tagsAt (← dataLen)) do
      let nreq ← uextend64 (← load32 (← iadd rptr (← ishlImm nq 2)))
      let fits ← band (← band (← rOk rR) (← select (← icmp .uge k (← iconst64 1)) (← iconst64 1) (← iconst64 0))) (← band (← select (← icmp .ule (← iadd tagsAt (← ishlImm nreq 2)) (← dataLen)) (← iconst64 1) (← iconst64 0))
        (← select (← icmp .ule (← ishlImm (← imul nq k) 3) (← outLen)) (← iconst64 1) (← iconst64 0)))
      let bad ← forLoopAcc nq (← select (← icmp .ne (← load32 rptr) (← iconst32 0)) (← iconst64 1) (← iconst64 0)) fun q acc => do
        let r0 ← load32 (← iadd rptr (← ishlImm q 2))
        let r1 ← load32 (← iadd rptr (← ishlImm (← iaddImm q 1) 2))
        bor acc (← select (← icmp .ule r1 r0) (← iconst64 1) (← iconst64 0))
      heldThen (← icmp .eq (← band fits (← bxor bad (← iconst64 1))) (← iconst64 1)) do
        -- the host's staging: qlo, qlen, off, optr (nq + 1), olo, ohi (nreq), cnt,
        -- small and big indices (nq each), then the scanned queries' answers
        let need ← iadd (← ishlImm (← iadd (← imul nq (← iconst64 7)) (← iaddImm (← ishlImm nreq 1) 1)) 2)
          (← ishlImm (← imul nq k) 3)
        when .ugt need (← getW base S_HFCAP) do
          when .ne (← getW base S_HFCAP) (← iconst64 0) do
            let _ ← ext (.cuda .memFreeHost) %[← getW base S_HF]
          let _ ← ext (.cuda .memAllocHost) %[← word base S_HF, need]
          setW base S_HFCAP need
        let hf ← getW base S_HF
        let hQlo := hf
        let hQlen ← iadd hQlo (← ishlImm nq 2)
        let hOff ← iadd hQlen (← ishlImm nq 2)
        let hOptr ← iadd hOff (← ishlImm nq 2)
        let hOlo ← iadd hOptr (← ishlImm (← iaddImm nq 1) 2)
        let hOhi ← iadd hOlo (← ishlImm nreq 2)
        let hCnt ← iadd hOhi (← ishlImm nreq 2)
        let hSidx ← iadd hCnt (← ishlImm nq 2)
        let hBidx ← iadd hSidx (← ishlImm nq 2)
        let ntags ← getW base S_NTAGS
        let tst ← getW base S_HTST
        let ten ← getW base S_HTEN
        -- each query's plan: its shortest required list, the others after it
        let run (t : V .i64) : Prog V L (Vals V [.i64, .i64]) := do
          ifte (jTys := [.i64, .i64]) .ult t ntags
            (thn := do
              pure %[← uextend64 (← load32 (← iadd tst (← ishlImm t 2))), ← uextend64 (← load32 (← iadd ten (← ishlImm t 2)))])
            (els := do pure %[← iconst64 0, ← iconst64 0])
        let total ← forLoopAcc nq (← iconst64 0) fun q acc => do
          let r0 ← uextend64 (← load32 (← iadd rptr (← ishlImm q 2)))
          let r1 ← uextend64 (← load32 (← iadd rptr (← ishlImm (← iaddImm q 1) 2)))
          -- the shortest: its index among the query's tags
          let best ← forLoopAcc (← isub r1 r0) (← iconst64 0) fun j bj => do
            let rj ← run (← uextend64 (← load32 (← iadd rtags (← ishlImm (← iadd r0 j) 2))))
            let rb ← run (← uextend64 (← load32 (← iadd rtags (← ishlImm (← iadd r0 bj) 2))))
            select (← icmp .ult (← isub rj.snd rj.head) (← isub rb.snd rb.head)) j bj
          let sr ← run (← uextend64 (← load32 (← iadd rtags (← ishlImm (← iadd r0 best) 2))))
          let len ← select (← icmp .eq r0 r1) (← iconst64 0) (← isub sr.snd sr.head)
          storeI32 (← ireduce32 sr.head) (← iadd hQlo (← ishlImm q 2))
          storeI32 (← ireduce32 len) (← iadd hQlen (← ishlImm q 2))
          storeI32 (← ireduce32 acc) (← iadd hOff (← ishlImm q 2))
          -- the others: each required run but the shortest; the optr slot is the
          -- running count of others, r0 - q for every query before (one each skipped)
          let ob ← isub r0 q
          storeI32 (← ireduce32 ob) (← iadd hOptr (← ishlImm q 2))
          forLoop (← isub r1 r0) fun j => do
            let rj ← run (← uextend64 (← load32 (← iadd rtags (← ishlImm (← iadd r0 j) 2))))
            let slot ← iadd ob (← select (← icmp .ult j best) j (← isub j (← iconst64 1)))
            when .ne j best do
              storeI32 (← ireduce32 rj.head) (← iadd hOlo (← ishlImm slot 2))
              storeI32 (← ireduce32 rj.snd) (← iadd hOhi (← ishlImm slot 2))
          iadd acc len
        storeI32 (← ireduce32 (← isub nreq nq)) (← iadd hOptr (← ishlImm nq 2))
        -- the plan up, the queries up, the admitted items enumerated
        let nq4 ← ishlImm nq 2
        for b in [FQLO, FQLEN, FOFF, ADMCNT, SIDX, BIDX] do ensure a base b (← umax nq4 (← iconst64 4))
        ensure a base FOPTR (← ishlImm (← iaddImm nq 1) 2)
        ensure a base FOLO (← umax (← ishlImm nreq 2) (← iconst64 4))
        ensure a base FOHI (← umax (← ishlImm nreq 2) (← iconst64 4))
        ensure a base ADM (← umax (← ishlImm total 2) (← iconst64 4))
        ensure a base QA (← umax (← imul nq (← iconst64 (4 * d))) (← iconst64 4))
        ensure a base OUTF (← umax (← ishlImm (← imul nq k) 3) (← iconst64 8))
        rebindAll a base
        for (b, src, bytes) in [(FQLO, hQlo, nq4), (FQLEN, hQlen, nq4), (FOFF, hOff, nq4)] do
          let _ ← ext (.cuda .memcpyHtoD) %[← ptrOf a base b, src, bytes]
        let _ ← ext (.cuda .memcpyHtoD) %[← ptrOf a base FOPTR, hOptr, ← ishlImm (← iaddImm nq 1) 2]
        let _ ← ext (.cuda .memcpyHtoD) %[← ptrOf a base FOLO, hOlo, ← ishlImm nreq 2]
        let _ ← ext (.cuda .memcpyHtoD) %[← ptrOf a base FOHI, hOhi, ← ishlImm nreq 2]
        let _ ← ext (.cuda .memcpyHtoD) %[← ptrOf a base QA, qp, ← imul nq (← iconst64 (4 * d))]
        a.runX base kADMIT (← lo32 nq)
        let _ ← ext (.cuda .memcpyDtoH) %[hCnt, ← ptrOf a base ADMCNT, nq4]
        -- the queries answered exactly, and those scanned
        let split ← forLoopAcc nq (← iconst64 0) fun q acc => do
          let c ← uextend64 (← load32 (← iadd hCnt (← ishlImm q 2)))
          let small ← icmp .ule c (← iconst64 FILTER_EXACT)
          let ns ← band acc (← iconst64 0xFFFFFFFF)
          let nb ← ushrImm acc 32
          let at' ← select small (← iadd hSidx (← ishlImm ns 2)) (← iadd hBidx (← ishlImm nb 2))
          storeI32 (← ireduce32 q) at'
          iadd acc (← select small (← iconst64 1) (← iconst64 0x100000000))
        let nsmall ← band split (← iconst64 0xFFFFFFFF)
        let nbig ← ushrImm split 32
        let _ ← ext (.cuda .memsetD8) %[← ptrOf a base OUTF, ← iconst32 0xFF, ← ishlImm (← imul nq k) 3]
        when .ne nsmall (← iconst64 0) do
          let _ ← ext (.cuda .memcpyHtoD) %[← ptrOf a base SIDX, hSidx, ← ishlImm nsmall 2]
          metaW base kEXACT 0 (← lo32 k)
          upMeta a base kEXACT
          a.runX base kEXACT (← lo32 nsmall)
        let _ ← ext (.cuda .memcpyDtoH) %[← outPtr, ← ptrOf a base OUTF, ← ishlImm (← imul nq k) 3]
        when .ne nbig (← iconst64 0) do
          let words ← udiv (← iaddImm (← getW base S_CAP) 31) (← iconst64 32)
          let bitsB ← ishlImm (← imul nbig words) 2
          ensure a base QBITS bitsB
          ensure a base Q (← imul nbig (← iconst64 (4 * d)))
          rebindAll a base
          let _ ← ext (.cuda .memcpyHtoD) %[← ptrOf a base BIDX, hBidx, ← ishlImm nbig 2]
          zero a base QBITS bitsB
          metaW base kSETBITS 0 (← lo32 words)
          upMeta a base kSETBITS
          a.runX base kSETBITS (← lo32 nbig)
          a.runX base kGROWS (← lo32 nbig)
          searchCore a base d nbig nprobe rR k words
          -- the scanned queries' answers, down once, then into their rows
          let stage ← iadd hBidx (← ishlImm nq 2)
          let _ ← ext (.cuda .memcpyDtoH) %[stage, ← ptrOf a base OUT, ← ishlImm (← imul nbig k) 3]
          let outp ← outPtr
          forLoop nbig fun i => do
            let qi ← uextend64 (← load32 (← iadd hBidx (← ishlImm i 2)))
            forLoop k fun j => do
              storeI64 (← load64 (← iadd stage (← ishlImm (← iadd (← imul i k) j) 3)))
                (← iadd outp (← ishlImm (← iadd (← imul qi k) j) 3))
        -- the device's status is the call's: a fault in any kernel shows
        uextend64 (← ext (.cuda .ctxSynchronize) %[])

end

/-! ## The artifacts -/

/-- The engines built: SIFT and BIGANN (128, L2), GloVe (100, angular as
    inner product on normalised vectors), Text-to-Image (200, inner product),
    YFCC (192, L2). -/
def shapes : List (Nat × Bool) := [(128, true), (100, false), (200, false), (192, true)]

def clifIR (d : Nat) (l2 : Bool) : Except String (List FuncData) :=
  Prog.program [.ok noopFunction,
    Prog.entry "init" (Prog.compileProgStatus 1 (initCode d l2)),
    Prog.entry "insert" (Prog.compileProgStatus 2 (insertCode d l2)),
    Prog.entry "delete" (Prog.compileProgStatus 3 (deleteCode d l2)),
    Prog.entry "commit" (Prog.compileProgStatus 4 (commitCode d l2)),
    Prog.entry "search" (Prog.compileProgStatus 5 (searchCode d l2)),
    Prog.entry "tags" (Prog.compileProgStatus 6 (tagsCode d l2)),
    Prog.entry "searchf" (Prog.compileProgStatus 7 (searchfCode d l2))]

/-- The arena's image: the layout's driver state, the constant `-2`, the
    module's text. -/
def setup (d : Nat) (l2 : Bool) (clif : List FuncData) : Except String Artifact := do
  let texts ← (AnnKernels.kernelsFor d l2).mapM Kernel.ptx
  let blobs := texts.map fun t => t.toUTF8.toList ++ [0]
  let offs := (blobs.foldl (fun (acc : List Nat × Nat) b => (acc.1 ++ [acc.2], acc.2 + b.length)) ([], PTX_OFF)).1
  let total := PTX_OFF + (blobs.map List.length).sum
  let a := app d l2 total
  let u64 (v : Nat) : List UInt8 := (List.range 8).map fun i => UInt8.ofNat (v / 256 ^ i % 256)
  let table := offs.flatMap u64
  let image := a.initialMemory [] ([(CONST_OFF, [0, 0, 0, 0xC0]), (MODTAB_OFF, table)] ++ offs.zip blobs)
  let mem := a.memSize
  pure { functions := clif, required_memory := mem, initial_memory := image }

def name (d : Nat) (l2 : Bool) : String := s!"ann_{d}_{if l2 then "l2" else "ip"}"

/-- The driver's state ends before the host's words, and the staged
    parameters fit their region. -/
theorem layout_fits : (shapes.all fun (d, l2) => decide ((app d l2).stateEnd ≤ STATE_OFF)) = true ∧
    META_OFF + 64 * NSPEC ≤ CONST_OFF ∧ MODTAB_OFF + 8 * 40 ≤ PTX_OFF ∧ (AnnKernels.kernelsFor 128 true).length ≤ 40 ∧
    ((specs 128 true).all fun (_, args, _) => decide (args.length ≤ 24)) = true := by
  refine ⟨?_, by decide, by decide, by decide, by native_decide⟩
  native_decide

end AnnHost

open AlgorithmLib in
def Ann.Host.main (args : List String) : IO Unit := do
  let dir ← requireOutputDir args
  for (d, l2) in AnnHost.shapes do
    let clif ← Prog.orDie (AnnHost.clifIR d l2)
    let art ← Prog.orDie (AnnHost.setup d l2 clif)
    emitArtifacts dir #[artifactEntry (AnnHost.name d l2) art]
