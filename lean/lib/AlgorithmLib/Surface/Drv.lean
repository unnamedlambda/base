module
public import AlgorithmLib.Surface.Prog
meta import AlgorithmLib.Surface.Prog
import all Init.Data.Repr
@[expose] public section

/-!
# `Drv` — GPU work as calls into the CUDA driver

A program that runs kernels calls the driver itself (`Ext.cuda`), with its
state at fixed offsets of its arena: the context the primary context's retain
stores, one slot per allocation holding the device pointer `cuMemAlloc`
stores there, per kernel a slot for its module and one for its function, and
the parameter cells and the array of pointers to them `cuLaunchKernel` reads.
Each helper here is a few driver calls at those offsets; the checker reads
the calls themselves (`Check.CudaRule`), so a theorem about a program states
what it needs of the driver in the driver's own terms.
-/

namespace AlgorithmLib.Drv

open AlgorithmLib.IR AlgorithmLib.Prog

variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

/-- A kernel as a program launches it: where its PTX and its entry point's
    name are in the arena, where its module and function handles are kept,
    the array of pointers to its parameter cells, the allocation slots it is
    handed in parameter order, and its grid and block. -/
structure Kern where
  ptxOff : Nat
  nameOff : Nat
  modSlot : Nat
  fnSlot : Nat
  paramsOff : Nat
  bufs : List Nat
  grid : Nat × Nat × Nat
  block : Nat × Nat × Nat

/-- The parameter sizes a kernel handed only allocations declares. -/
def Kern.sizes (k : Kern) : List Nat := k.bufs.map fun _ => 8

/-- Bring the driver up and make the device's primary context current,
    keeping the context at `ctxOff`. -/
def up (base : V .i64) (ctxOff : Nat) : Prog V L Unit := do
  let z ← iconst32 0
  let _ ← ext (.cuda .init) %[z]
  let _ ← ext (.cuda .primaryCtxRetain) %[← iaddImm base ctxOff, z]
  let _ ← ext (.cuda .ctxSetCurrent) %[← load64 (← iaddImm base ctxOff)]

/-- An allocation of `bytes` bytes, its pointer kept at `slot`. -/
def alloc (base : V .i64) (slot bytes : Nat) : Prog V L Unit := do
  let _ ← ext (.cuda .memAlloc) %[← iaddImm base slot, ← iconst64 bytes]

/-- Copy `bytes` bytes from `src` into the allocation kept at `slot`. -/
def upload (base : V .i64) (slot : Nat) (src : V .i64) (bytes : Nat) : Prog V L Unit := do
  let _ ← ext (.cuda .memcpyHtoD) %[← load64 (← iaddImm base slot), src, ← iconst64 bytes]

/-- Copy `bytes` bytes from the allocation kept at `slot` to `dst`. -/
def download (base : V .i64) (dst : V .i64) (slot bytes : Nat) : Prog V L Unit := do
  let _ ← ext (.cuda .memcpyDtoH) %[dst, ← load64 (← iaddImm base slot), ← iconst64 bytes]

/-- Free the allocation kept at `slot`. -/
def free (base : V .i64) (slot : Nat) : Prog V L Unit := do
  let _ ← ext (.cuda .memFree) %[← load64 (← iaddImm base slot)]

/-- Load a kernel's module. -/
def loadModule (base : V .i64) (k : Kern) : Prog V L Unit := do
  let _ ← ext (.cuda .moduleLoadData) %[← iaddImm base k.modSlot, ← iaddImm base k.ptxOff]

/-- Look a kernel's entry point up in its loaded module. -/
def getFunction (base : V .i64) (k : Kern) : Prog V L Unit := do
  let _ ← ext (.cuda .moduleGetFunction) %[← iaddImm base k.fnSlot, ← load64 (← iaddImm base k.modSlot),
    ← iaddImm base k.nameOff]

/-- Write a parameter block: the pointer kept at each slot of `slots` into
    its cell from `argOff`, and each cell's address into the array from
    `paramsOff` the launch reads. -/
def bind (base : V .i64) (argOff paramsOff : Nat) (slots : List Nat) : Prog V L Unit := do
  for (slot, i) in slots.zip (List.range slots.length) do
    let cell ← iaddImm base (argOff + 8 * i)
    storeI64 (← load64 (← iaddImm base slot)) cell
    storeI64 cell (← iaddImm base (paramsOff + 8 * i))

/-- Write a parameter block from values the program holds: each into its
    cell from `argOff`, and each cell's address into the array from
    `paramsOff` the launch reads. -/
def bindVals (base : V .i64) (argOff paramsOff : Nat) (vals : List (V .i64)) : Prog V L Unit := do
  for (v, i) in vals.zip (List.range vals.length) do
    let cell ← iaddImm base (argOff + 8 * i)
    storeI64 v cell
    storeI64 cell (← iaddImm base (paramsOff + 8 * i))

/-- Write a parameter block from the pointers kept at arena offsets `addrs`:
    each into its cell from `argOff`, and each cell's address into the array
    from `paramsOff` the launch reads. -/
def bindFrom (base : V .i64) (argOff paramsOff : Nat) (addrs : List Nat) : Prog V L Unit := do
  for (a, i) in addrs.zip (List.range addrs.length) do
    let cell ← iaddImm base (argOff + 8 * i)
    storeI64 (← load64 (← iaddImm base a)) cell
    storeI64 cell (← iaddImm base (paramsOff + 8 * i))

/-- Launch a kernel whose parameters are written, on `stream`. -/
def launch (base : V .i64) (k : Kern) (stream : V .i64) : Prog V L Unit := do
  let (gx, gy, gz) := k.grid
  let (bx, by_, bz) := k.block
  let _ ← ext (.cuda .launchKernel) %[← load64 (← iaddImm base k.fnSlot),
    ← iconst32 gx, ← iconst32 gy, ← iconst32 gz, ← iconst32 bx, ← iconst32 by_, ← iconst32 bz,
    ← iconst32 0, stream, ← iaddImm base k.paramsOff, ← iconst64 0]

/-- Launch a kernel whose parameters are written, on `stream`, `gx` blocks
    wide: a width the program computes. -/
def launchX (base : V .i64) (k : Kern) (gx : V .i32) (stream : V .i64) : Prog V L Unit := do
  let (_, gy, gz) := k.grid
  let (bx, by_, bz) := k.block
  let _ ← ext (.cuda .launchKernel) %[← load64 (← iaddImm base k.fnSlot),
    gx, ← iconst32 gy, ← iconst32 gz, ← iconst32 bx, ← iconst32 by_, ← iconst32 bz,
    ← iconst32 0, stream, ← iaddImm base k.paramsOff, ← iconst64 0]

/-- Wait for every stream. -/
def sync : Prog V L Unit := do
  let _ ← ext (.cuda .ctxSynchronize) %[]

/-- `C = α op(A) op(B) + β C` in f32 by cuBLAS on handle `h`, for `batch`
    members strided `sa`, `sb`, `sc` elements apart, every operand packed:
    `op(A)` is `m × k`, `op(B)` `k × n`, `C` `m × n`, `op` transposing where
    `ta` or `tb` is `1`; `α` and `β` are read from `alpha` and `beta`. -/
def sgemmStrided (h : V .i64) (ta tb : Nat) (m n k : V .i32) (alpha A sa B sb beta C sc : V .i64)
    (batch : V .i32) : Prog V L Unit := do
  let _ ← ext (.cublas .sgemmStridedBatched) %[h, ← iconst32 ta, ← iconst32 tb, m, n, k, alpha,
    A, (if ta = 1 then k else m), sa, B, (if tb = 1 then n else k), sb, beta, C, m, sc, batch]

/-- `C = α op(A) op(B) + β C` by cuBLAS on handle `h`, `A` and `B` bf16 and
    `C` f32, accumulated in f32, every operand packed. -/
def gemmBf16 (h : V .i64) (ta tb : Nat) (m n k : V .i32) (alpha A B beta C : V .i64) : Prog V L Unit := do
  let _ ← ext (.cublas .gemmEx) %[h, ← iconst32 ta, ← iconst32 tb, m, n, k, alpha,
    A, ← iconst32 14, (if ta = 1 then k else m), B, ← iconst32 14, (if tb = 1 then n else k),
    beta, C, ← iconst32 0, m, ← iconst32 68, ← iconst32 (-1)]

-- ---------------------------------------------------------------------------
-- The standard layout
-- ---------------------------------------------------------------------------

/-- A kernel of a program in the standard layout: where its PTX is and how
    many bytes are kept for it, the allocations it is handed by index, its
    grid and its block. Its entry point is `main`. -/
structure KSpec where
  ptxOff : Nat
  ptxLen : Nat
  args : List Nat
  grid : Nat × Nat × Nat
  block : Nat × Nat × Nat

/-- **A program in the standard layout**: its allocations' sizes in bytes
    (`0` for one whose size the program learns at run time and makes itself),
    its kernels, its arena's size, whether it calls cuBLAS, and where in the
    arena the driver's state starts (`org`). From `org` lie the context, the
    entry point's name `main` at `8`, the cuBLAS handle at `0x10`, the
    single-precision `1` and `0` cuBLAS reads its scalars from at `0x18` and
    `0x1c`, a slot per allocation, a module and a function slot per kernel,
    and a parameter block per distinct list of allocations a kernel is handed
    (`blocks`): `cells` parameter cells and the array of as many pointers to
    them, which every kernel handed that list launches with, then, where a
    kernel's entry point is not `main`, 32 bytes per kernel for its name.
    Kernels whose PTX is at one offset share one module. The kernels' PTX
    lies wherever `ptxOff` says, outside all of it. -/
structure App where
  bufs : List Nat
  kernels : List KSpec
  memSize : Nat
  blas : Bool := false
  org : Nat := 0
  /-- Each kernel's entry point's name, in the kernels' order; `main` past
      the list's end. -/
  entries : List String := []
  /-- How many parameters a parameter block holds: at least as many
      allocations as any kernel is handed. -/
  cells : Nat := 16

namespace App

def CTX (a : App) : Nat := a.org
def NAME (a : App) : Nat := a.org + 8
def BLAS (a : App) : Nat := a.org + 0x10
def ONE (a : App) : Nat := a.org + 0x18
def ZERO (a : App) : Nat := a.org + 0x1c

def bufSlot (a : App) (i : Nat) : Nat := a.org + 0x20 + 8 * i
def modSlot (a : App) (j : Nat) : Nat := a.org + 0x20 + 8 * a.bufs.length + 16 * j
def fnSlot (a : App) (j : Nat) : Nat := a.modSlot j + 8
/-- The distinct lists of allocations the kernels are handed, in the order
    they first appear: one parameter block each. -/
def blocks (a : App) : List (List Nat) :=
  a.kernels.foldl (fun bs k => if k.args ∈ bs then bs else bs ++ [k.args]) []

/-- Where a parameter block's array of pointers starts within it. -/
def arrOff (a : App) : Nat := 8 * a.cells

/-- Where parameter block `b` starts: its cells, then at `arrOff` its array. -/
def blockOff (a : App) (b : Nat) : Nat :=
  a.org + 0x20 + 8 * a.bufs.length + 16 * a.kernels.length + 16 * a.cells * b

/-- Kernel `j`'s parameter block. -/
def blockOf (a : App) (j : Nat) : Nat :=
  match a.kernels[j]? with
  | some k => a.blocks.idxOf k.args
  | none => 0

/-- The array of pointers kernel `j` launches with. -/
def paramsOff (a : App) (j : Nat) : Nat := a.blockOff (a.blockOf j) + a.arrOff

/-- Kernel `j`'s entry point's name. -/
def entry (a : App) (j : Nat) : String := a.entries.getD j "main"

/-- Whether some kernel's entry point is not `main`. -/
def named (a : App) : Bool := a.entries.any (· != "main")

/-- Where kernel `j`'s entry point's name is: `NAME` for `main`, its own 32
    bytes past the parameter blocks otherwise. -/
def nameOff (a : App) (j : Nat) : Nat :=
  if a.entry j == "main" then a.NAME else a.blockOff a.blocks.length + 32 * j

/-- The kernel that loads kernel `j`'s module: the first whose PTX is at the
    same offset. -/
def modOwner (a : App) (j : Nat) : Nat :=
  match a.kernels[j]? with
  | some k => (a.kernels.findIdx? (·.ptxOff == k.ptxOff)).getD j
  | none => j

/-- Where the driver's state ends. -/
def stateEnd (a : App) : Nat :=
  a.blockOff a.blocks.length + (if a.named then 32 * a.kernels.length else 0)

/-- Kernel `j` as the helpers launch it. -/
def kern (a : App) (j : Nat) (k : KSpec) : Kern :=
  { ptxOff := k.ptxOff, nameOff := a.nameOff j, modSlot := a.modSlot (a.modOwner j), fnSlot := a.fnSlot j,
    paramsOff := a.paramsOff j, bufs := k.args.map a.bufSlot,
    grid := k.grid, block := k.block }

/-- Each kernel with its index. -/
def indexed (a : App) : List (Nat × KSpec) := (List.range a.kernels.length).zip a.kernels

/-- Bring the driver up, make cuBLAS's handle if the program calls it, make
    every allocation whose size is known, load every kernel and write every
    parameter block. -/
def setup (a : App) (base : V .i64) : Prog V L Unit := do
  up base a.CTX
  if a.blas then
    let _ ← ext (.cublas .create) %[← iaddImm base a.BLAS]
  for (n, i) in a.bufs.zip (List.range a.bufs.length) do
    if n != 0 then alloc base (a.bufSlot i) n
  for (j, k) in a.indexed do
    if a.modOwner j == j then loadModule base (a.kern j k)
    getFunction base (a.kern j k)
  for (b, args) in (List.range a.blocks.length).zip a.blocks do
    bind base (a.blockOff b) (a.blockOff b + a.arrOff) (args.map a.bufSlot)

/-- Launch kernel `j` on `stream`. -/
def runOn (a : App) (base stream : V .i64) (j : Nat) : Prog V L Unit :=
  match a.kernels[j]? with
  | some k => launch base (a.kern j k) stream
  | none => pure ()

/-- Launch kernel `j` on the default stream. -/
def run (a : App) (base : V .i64) (j : Nat) : Prog V L Unit := do
  match a.kernels[j]? with
  | some k => launch base (a.kern j k) (← iconst64 0)
  | none => pure ()

/-- Launch kernel `j` on the default stream, `gx` blocks wide. -/
def runX (a : App) (base : V .i64) (j : Nat) (gx : V .i32) : Prog V L Unit := do
  match a.kernels[j]? with
  | some k => launchX base (a.kern j k) gx (← iconst64 0)
  | none => pure ()

/-- Write kernel `j`'s parameter block again from its allocations' slots,
    so a pointer the program moved into a slot since `setup` is the one the
    kernel is handed. -/
def rebind (a : App) (base : V .i64) (j : Nat) : Prog V L Unit :=
  match a.kernels[j]? with
  | some k => bind base (a.blockOff (a.blockOf j)) (a.blockOff (a.blockOf j) + a.arrOff) (k.args.map a.bufSlot)
  | none => pure ()

/-- `y = A x` by cuBLAS on the default stream, `A` the packed `m × n`
    column-major matrix in allocation `A` (transposed where `trans` is `1`),
    `x` and `y` allocations too. -/
def sgemv (a : App) (base : V .i64) (trans m n A x y : Nat) : Prog V L Unit := do
  let _ ← ext (.cublas .sgemv) %[← load64 (← iaddImm base a.BLAS), ← iconst32 trans, ← iconst32 m,
    ← iconst32 n, ← iaddImm base a.ONE, ← load64 (← iaddImm base (a.bufSlot A)), ← iconst32 m,
    ← load64 (← iaddImm base (a.bufSlot x)), ← iconst32 1, ← iaddImm base a.ZERO,
    ← load64 (← iaddImm base (a.bufSlot y)), ← iconst32 1]

/-- `C = op(A) op(B)` by cuBLAS on the default stream, each operand packed
    column-major in its allocation: `op(A)` is `m × k`, `op(B)` is `k × n` and
    `C` is `m × n`, `op` transposing where `ta` or `tb` is `1`. -/
def sgemm (a : App) (base : V .i64) (ta tb m n k A B C : Nat) : Prog V L Unit := do
  let _ ← ext (.cublas .sgemm) %[← load64 (← iaddImm base a.BLAS), ← iconst32 ta, ← iconst32 tb,
    ← iconst32 m, ← iconst32 n, ← iconst32 k, ← iaddImm base a.ONE,
    ← load64 (← iaddImm base (a.bufSlot A)), ← iconst32 (if ta = 1 then k else m),
    ← load64 (← iaddImm base (a.bufSlot B)), ← iconst32 (if tb = 1 then n else k),
    ← iaddImm base a.ZERO, ← load64 (← iaddImm base (a.bufSlot C)), ← iconst32 m]

/-- Copy `bytes` bytes from `src` into allocation `i`. -/
def upload (a : App) (base : V .i64) (i : Nat) (src : V .i64) (bytes : Nat) : Prog V L Unit :=
  Drv.upload base (a.bufSlot i) src bytes

/-- Copy `bytes` bytes from allocation `i` to `dst`. -/
def download (a : App) (base dst : V .i64) (i bytes : Nat) : Prog V L Unit :=
  Drv.download base dst (a.bufSlot i) bytes

/-- The arena's first bytes: the name `main` and the scalar `1` at their
    places, each kernel's PTX at its own, `ptxs` in the kernels' order, and
    each of `extra`'s bytes at its offset. -/
def initialMemory (a : App) (ptxs : List String) (extra : List (Nat × List UInt8) := []) : List UInt8 :=
  let placed := (a.NAME, "main".toUTF8.toList ++ [0]) :: (a.ONE, [0, 0, 0x80, 0x3f]) ::
    (a.kernels.zip ptxs).map (fun (k, p) => (k.ptxOff, p.toUTF8.toList ++ [0])) ++
    (List.range a.kernels.length).filterMap (fun j =>
      if a.entry j == "main" then none else some (a.nameOff j, (a.entry j).toUTF8.toList ++ [0])) ++ extra
  (placed.foldl (fun m (o, bs) => bs.zipIdx.foldl (fun m (b, i) => m.set! (o + i) b) m)
    (Array.replicate a.memSize (0 : UInt8))).toList

/-- The image is the arena's size. -/
theorem initialMemory_length (a : App) (ptxs : List String) (extra : List (Nat × List UInt8)) :
    (a.initialMemory ptxs extra).length = a.memSize := by
  unfold initialMemory
  rw [Array.length_toList]
  suffices h : ∀ (l : List (Nat × List UInt8)) (m : Array UInt8),
      (l.foldl (fun m (o, bs) => bs.zipIdx.foldl (fun m (b, i) => m.set! (o + i) b) m) m).size = m.size by
    rw [h]; simp
  intro l
  induction l with
  | nil => intro m; rfl
  | cons p l ih =>
      intro m
      rw [List.foldl_cons, ih]
      obtain ⟨o, bs⟩ := p
      suffices h2 : ∀ (xs : List (UInt8 × Nat)) (m : Array UInt8),
          (xs.foldl (fun m (b, i) => m.set! (o + i) b) m).size = m.size from h2 _ _
      intro xs
      induction xs with
      | nil => intro m; rfl
      | cons x xs ih2 => intro m; rw [List.foldl_cons, ih2]; simp [Array.set!_eq_setIfInBounds]

end App

end AlgorithmLib.Drv
