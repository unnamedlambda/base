module
public import AlgorithmLib.Surface.LibBase
meta import AlgorithmLib.Surface.LibBase
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `Lib.Cuda` — the engine's CUDA entry points, as CLIF over the driver

Programs name GPU work through `Ffi`'s CUDA and cuBLAS entries: buffers,
streams, events, graphs and page-locked allocations by `i32` id, kernels by the
address of their PTX. What each entry does is stated in `Host.Ffi`. Here each
is a function of the program itself, written over the driver and cuBLAS called
directly (`Ext.cuda`, `Ext.cublas`), so no engine code sits between a program
and the vendor libraries.

**The state.** `cudaInit` allocates one page-locked block and writes its
address where the engine wrote its context pointer; every later call takes it
as that context. The block holds the driver context, the default cuBLAS
handle, scratch for the driver's out-parameters, the argument array a launch
passes, a cache of loaded kernels, and one table per kind of object. A table
is a page-locked array that doubles when full; an id is an index into it and is
never reused, and a destroyed object's entry is zero.

**Kernels.** A kernel is the pair of its PTX's address and its entry point's
(`0` for `main`). The first launch loads the module and looks the function up;
later ones find it in an open-addressed table of 4096 entries. At three
quarters full the table is emptied — the device waited for, every module
unloaded — which costs time and nothing else.

**cuBLAS.** A program that makes a cuBLAS call gets a default handle at
`cudaInit` and one per stream at `cudaStreamCreate`; a program that makes none
never loads the library. Creating a handle while a stream is being captured is
not something cuBLAS defines, and creating each before any capture can begin
is what keeps the calls made during one well-defined.

**Linking.** `Link.link` appends the functions a program's calls need and
points each call at its function. The engine's own entry points are then never
called: the artifact names the vendor libraries and nothing else.

**Limits.** One thread: the state is not locked, and the driver context is made
current on the thread that called `cudaInit`.
-/

namespace AlgorithmLib.LibCuda

open AlgorithmLib.IR
open AlgorithmLib.Prog
open AlgorithmLib.Lib

variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

-- ---------------------------------------------------------------------------
-- The state block
-- ---------------------------------------------------------------------------

def CTX : Nat := 0x00
def DEV : Nat := 0x08
def BLAS : Nat := 0x10
def TMP0 : Nat := 0x18
def TMP1 : Nat := 0x20
def TMP2 : Nat := 0x28
def ALPHA : Nat := 0x30
def BETA : Nat := 0x34
/-- `"main"`, NUL-terminated. -/
def MAIN : Nat := 0x38
/-- An event, without timing, that orders a stream behind another. -/
def FENCE : Nat := 0x40
def KCOUNT : Nat := 0x48
/-- A launch's arguments: the device pointers, then the array of their
    addresses `cuLaunchKernel` takes. -/
def MAX_ARGS : Nat := 512
def ARGV : Nat := 0x1000
def ARGP : Nat := ARGV + 8 * MAX_ARGS
def CACHE : Nat := ARGP + 8 * MAX_ARGS
def CACHE_N : Nat := 4096
def SIZE : Nat := CACHE + 32 * CACHE_N

/-- A table: where its header is (address, length, capacity) and how wide an
    entry is. The first word of a live entry is never zero. -/
structure Tbl where
  hdr : Nat
  width : Nat

/-- A buffer: its device pointer and size. -/
def BUFS : Tbl := ⟨0x100, 16⟩
/-- A stream: its handle and its cuBLAS handle. -/
def STREAMS : Tbl := ⟨0x120, 16⟩
def EVENTS : Tbl := ⟨0x140, 8⟩
def GRAPHS : Tbl := ⟨0x160, 8⟩
/-- A page-locked allocation: its address and size. -/
def PINNED : Tbl := ⟨0x180, 16⟩

def tables : List Tbl := [BUFS, STREAMS, EVENTS, GRAPHS, PINNED]

-- ---------------------------------------------------------------------------
-- Calls and results
-- ---------------------------------------------------------------------------

def cu (f : CudaFn) (args : Vals V (Ext.cuda f).sig.1) : Prog V L (V .i32) :=
  ext (.cuda f) args

def bl (f : CublasFn) (args : Vals V (Ext.cublas f).sig.1) : Prog V L (V .i32) :=
  ext (.cublas f) args

-- ---------------------------------------------------------------------------
-- Tables
-- ---------------------------------------------------------------------------

/-- The address of live entry `id` of `t`, or `0` when `id` is negative, past
    the end, or names an entry since destroyed. -/
def entry (s : V .i64) (t : Tbl) (id : V .i32) : Prog V L (V .i64) := do
  let i ← sextend64 id
  let len ← load64 (← at_ s (t.hdr + 8))
  let zero ← iconst64 0
  -- A negative id is past every length, compared unsigned.
  let r ← ifte (jTys := [.i64]) .uge i len (pure %[zero]) (do
    let base ← load64 (← at_ s t.hdr)
    let e ← iadd base (← imul i (← iconst64 t.width))
    let r ← ifte (jTys := [.i64]) .eq (← load64 e) zero (pure %[zero]) (pure %[e])
    pure %[r.head])
  pure r.head

/-- Append an entry to `t`: its index and address, or `-1` when the table
    could not grow. -/
def push (s : V .i64) (t : Tbl) : Prog V L (V .i64 × V .i64) := do
  let lenA ← at_ s (t.hdr + 8)
  let capA ← at_ s (t.hdr + 16)
  let len ← load64 lenA
  let cap ← load64 capA
  let zero ← iconst64 0
  let zero32 ← iconst32 0
  let width ← iconst64 t.width
  let grown ← ifte (jTys := [.i32]) .ult len cap (pure %[zero32]) (do
    let dbl ← ishlImm cap 1
    let nc ← ifte (jTys := [.i64]) .eq cap zero (do pure %[← iconst64 64]) (pure %[dbl])
    let tmp ← at_ s TMP2
    let rc ← cu .memAllocHost %[tmp, ← imul nc.head width]
    let r ← ifte (jTys := [.i32]) .ne rc zero32 (pure %[rc]) (do
      let nw ← load64 tmp
      let old ← load64 (← at_ s t.hdr)
      when .ne old zero (do
        let _ ← memcpy nw old (← imul len width)
        let _ ← cu .memFreeHost %[old]
        pure ())
      storeI64 nw (← at_ s t.hdr)
      storeI64 nc.head capA
      pure %[zero32])
    pure %[r.head])
  let r ← ifte (jTys := [.i64, .i64]) .ne grown.head zero32
    (do pure %[← iconst64 (-1), zero])
    (do
      let base ← load64 (← at_ s t.hdr)
      let e ← iadd base (← imul len width)
      storeI64 (← iaddImm len 1) lenA
      pure %[len, e])
  pure (r.head, r.snd)

/-- Run `f` on the address of each live entry of `t`. -/
def forLive (s : V .i64) (t : Tbl) (f : V .i64 → Prog V L Unit) : Prog V L Unit := do
  let len ← load64 (← at_ s (t.hdr + 8))
  let base ← load64 (← at_ s t.hdr)
  let zero ← iconst64 0
  forLoop len fun i => do
    let e ← iadd base (← imul i (← iconst64 t.width))
    when .ne (← load64 e) zero (f e)

/-- `-1` for `ctx` null, `k` otherwise. -/
def withCtx (ctx : V .i64) (k : Prog V L (V .i64)) : Prog V L (V .i64) := do
  failIf .eq ctx (← iconst64 0) k

/-- The buffer `id` names: its device pointer and size, or `-1` when it names none. -/
def withBuf (s : V .i64) (id : V .i32) (k : V .i64 → V .i64 → Prog V L (V .i64)) :
    Prog V L (V .i64) := do
  let e ← entry s BUFS id
  failIf .eq e (← iconst64 0) (do k (← load64 e) (← load64 (← iaddImm e 8)))

/-- The stream an id names, with its cuBLAS handle: the default stream and
    handle for a negative id, `-1` for one never created or destroyed. -/
def withStream (s : V .i64) (sid : V .i32) (k : V .i64 → V .i64 → Prog V L (V .i64)) :
    Prog V L (V .i64) := do
  let r ← ifte (jTys := [.i64, .i64]) .slt sid (← iconst32 0)
    (do pure %[← iconst64 0, ← load64 (← at_ s BLAS)])
    (do
      let e ← entry s STREAMS sid
      let r ← ifte (jTys := [.i64, .i64]) .eq e (← iconst64 0)
        (do pure %[← iconst64 (-1), ← iconst64 0])
        (do pure %[← load64 e, ← load64 (← iaddImm e 8)])
      pure %[r.head, r.snd])
  failIf .eq r.head (← iconst64 (-1)) (k r.head r.snd)

/-- The live entry of `t` an id names, or `-1`. -/
def withEntry (s : V .i64) (t : Tbl) (id : V .i32) (k : V .i64 → Prog V L (V .i64)) :
    Prog V L (V .i64) := do
  let e ← entry s t id
  failIf .eq e (← iconst64 0) (k e)

/-- Record a new object's first word (and second, for a two-word table) and
    answer its id, or `-1` when the table could not grow. -/
def record (s : V .i64) (t : Tbl) (w0 : V .i64) (w1 : Option (V .i64) := none) :
    Prog V L (V .i64) := do
  let (idx, e) ← push s t
  failIf .slt idx (← iconst64 0) (do
    storeI64 w0 e
    if let some v := w1 then storeI64 v (← iaddImm e 8)
    pure idx)

-- ---------------------------------------------------------------------------
-- Kernels
-- ---------------------------------------------------------------------------

/-- The cache entry for `(kptr, key)`, and whether it holds the pair; when it
    does not, it is the empty entry the pair goes in. The table is never more
    than three quarters full, so the probe finds one or the other. -/
def probe (s kptr key : V .i64) : Prog V L (V .i64 × V .i64) := do
  let mask ← iconst64 (CACHE_N - 1)
  let h0 ← band (← bxor (← ushrImm kptr 4) (← bxor (← ushrImm key 4) (← ushrImm kptr 16))) mask
  let base ← at_ s CACHE
  let zero ← iconst64 0
  let r ← wloop1 h0
    (head := fun h => do
      let e ← iadd base (← ishlImm h 5)
      let k1 ← load64 e
      let k2 ← load64 (← iaddImm e 8)
      let empty ← icmp .eq k1 zero
      let same ← band (← icmp .eq k1 kptr) (← icmp .eq k2 key)
      let stop ← bor empty same
      return (exitIf .ne stop (← iconst .i8 0), %[e, ← uextend64 same], ()))
    (body := fun h _ => do return %[← band (← iaddImm h 1) mask])
  pure (r.head, r.snd)

/-- Unload every cached module. -/
def unloadAll (s : V .i64) : Prog V L Unit := do
  let base ← at_ s CACHE
  let zero ← iconst64 0
  forLoop (← iconst64 CACHE_N) fun i => do
    let e ← iadd base (← ishlImm i 5)
    when .ne (← load64 e) zero (do
      let _ ← cu .moduleUnload %[← load64 (← iaddImm e 16)]
      pure ())

/-- Empty the cache: wait for the device, since a module may still be running,
    then unload everything. -/
def evict (s : V .i64) : Prog V L Unit := do
  let _ ← cu .ctxSynchronize %[]
  unloadAll s
  let _ ← memset (← at_ s CACHE) (← iconst32 0) (← iconst64 (CACHE_N * 32))
  storeI64 (← iconst64 0) (← at_ s KCOUNT)

/-- The function entry point `name` of the PTX at `kptr` names, loading its
    module the first time; `0` when the module will not load or has no such
    entry. `key` is the name's address, or `0` for `main`. -/
def kernelFn (s kptr key name : V .i64) : Prog V L (V .i64) := do
  let (e, found) ← probe s kptr key
  let zero ← iconst64 0
  let zero32 ← iconst32 0
  let r ← ifte (jTys := [.i64]) .ne found zero (do pure %[← load64 (← iaddImm e 24)]) (do
    let cntA ← at_ s KCOUNT
    let e2 ← ifte (jTys := [.i64]) .ult (← load64 cntA) (← iconst64 (CACHE_N * 3 / 4))
      (pure %[e])
      (do
        evict s
        let (e', _) ← probe s kptr key
        pure %[e'])
    let tmp ← at_ s TMP0
    let rc ← cu .moduleLoadData %[tmp, kptr]
    let r ← ifte (jTys := [.i64]) .ne rc zero32 (pure %[zero]) (do
      let m ← load64 tmp
      let tmp1 ← at_ s TMP1
      let rc2 ← cu .moduleGetFunction %[tmp1, m, name]
      let r ← ifte (jTys := [.i64]) .ne rc2 zero32
        (do
          let _ ← cu .moduleUnload %[m]
          pure %[zero])
        (do
          let f ← load64 tmp1
          let e := e2.head
          storeI64 kptr e
          storeI64 key (← iaddImm e 8)
          storeI64 m (← iaddImm e 16)
          storeI64 f (← iaddImm e 24)
          storeI64 (← iaddImm (← load64 cntA) 1) cntA
          pure %[f])
      pure %[r.head])
    pure %[r.head])
  pure r.head

/-- Write the device pointers of the `n` buffers whose ids are at `bind` into
    the argument array: `0` when every id names a live buffer, `-1` otherwise. -/
def bindArgs (s n bind : V .i64) : Prog V L (V .i64) := do
  let argv ← at_ s ARGV
  let zero ← iconst64 0
  let bad ← forLoopAcc n zero fun i acc => do
    let id ← load32 (← iadd bind (← ishlImm i 2))
    let e ← entry s BUFS id
    let p ← ifte (jTys := [.i64]) .eq e zero (pure %[zero]) (do pure %[← load64 e])
    storeI64 p.head (← iadd argv (← ishlImm i 3))
    bor acc (← uextend64 (← icmp .eq e zero))
  failIf .ne bad zero (iconst64 0)

/-- A launch of `name` in the PTX at `kptr` over the buffers bound at `bind`,
    on `stream`. -/
def launchOn (s kptr key name : V .i64) (nBufs : V .i32) (bind : V .i64)
    (gx gy gz bx by_ bz : V .i32) (stream : V .i64) : Prog V L (V .i64) := do
  let bad ← or8 (← anyNonPos [gx, gy, gz, bx, by_, bz])
    (← or8 (← icmp .slt nBufs (← iconst32 0)) (← icmp .sgt nBufs (← iconst32 MAX_ARGS)))
  failUnless0 bad (do
    let f ← kernelFn s kptr key name
    failIf .eq f (← iconst64 0) (do
      let ok ← bindArgs s (← sextend64 nBufs) bind
      failIf .ne ok (← iconst64 0) (do
        status (← cu .launchKernel %[f, gx, gy, gz, bx, by_, bz, ← iconst32 0, stream,
          ← at_ s ARGP, ← iconst64 0]))))

-- ---------------------------------------------------------------------------
-- The entry points
-- ---------------------------------------------------------------------------

/-- `cudaInit(slot)`: a context made current, the state block, and — for a
    program that calls cuBLAS — the default handle. The slot ends holding the
    block, or `0` when any step failed. -/
def init (blas : Bool) : Body := do
  let (slot ::ᵥ .nil) ← entryParams [.i64]
  let zero32 ← iconst32 0
  let zero ← iconst64 0
  let rc ← cu .init %[zero32]
  let r ← ifte (jTys := [.i64]) .ne rc zero32 (pure %[zero]) (do
    let rc ← cu .deviceGet %[slot, zero32]
    let r ← ifte (jTys := [.i64]) .ne rc zero32 (pure %[zero]) (do
      let dev ← load32 slot
      let rc ← cu .primaryCtxRetain %[slot, dev]
      let r ← ifte (jTys := [.i64]) .ne rc zero32 (pure %[zero]) (do
        let ctx ← load64 slot
        let rc ← andThen (← cu .ctxSetCurrent %[ctx]) (cu .memAllocHost %[slot, ← iconst64 SIZE])
        let r ← ifte (jTys := [.i64]) .ne rc zero32 (pure %[zero]) (do
          let s ← load64 slot
          let _ ← memset s zero32 (← iconst64 SIZE)
          storeI64 ctx (← at_ s CTX)
          storeI32 dev (← at_ s DEV)
          -- "main\0", little-endian
          storeI64 (← iconst64 0x6e69616d) (← at_ s MAIN)
          let argv ← at_ s ARGV
          let argp ← at_ s ARGP
          forLoop (← iconst64 MAX_ARGS) fun i => do
            let off ← ishlImm i 3
            storeI64 (← iadd argv off) (← iadd argp off)
          -- CU_EVENT_DISABLE_TIMING
          let rc ← cu .eventCreate %[← at_ s FENCE, ← iconst32 2]
          let rc ← if blas then andThen rc (bl .create %[← at_ s BLAS]) else pure rc
          let r ← ifte (jTys := [.i64]) .ne rc zero32 (pure %[zero]) (pure %[s])
          pure %[r.head])
        pure %[r.head])
      pure %[r.head])
    pure %[r.head])
  storeI64 r.head slot

/-- `cudaCleanup(slot)`: every object destroyed, every module unloaded, the
    block freed and the context released; the slot ends `0`. -/
def cleanup : Body := do
  let (slot ::ᵥ .nil) ← entryParams [.i64]
  let s ← load64 slot
  let zero ← iconst64 0
  when .ne s zero (do
    let _ ← cu .ctxSynchronize %[]
    unloadAll s
    forLive s BUFS fun e => do let _ ← cu .memFree %[← load64 e]; pure ()
    forLive s STREAMS fun e => do
      let h ← load64 (← iaddImm e 8)
      when .ne h zero (do let _ ← bl .destroy %[h]; pure ())
      let _ ← cu .streamDestroy %[← load64 e]
      pure ()
    forLive s EVENTS fun e => do let _ ← cu .eventDestroy %[← load64 e]; pure ()
    forLive s GRAPHS fun e => do let _ ← cu .graphExecDestroy %[← load64 e]; pure ()
    forLive s PINNED fun e => do let _ ← cu .memFreeHost %[← load64 e]; pure ()
    for t in tables do
      let p ← load64 (← at_ s t.hdr)
      when .ne p zero (do let _ ← cu .memFreeHost %[p]; pure ())
    let h ← load64 (← at_ s BLAS)
    when .ne h zero (do let _ ← bl .destroy %[h]; pure ())
    let _ ← cu .eventDestroy %[← load64 (← at_ s FENCE)]
    let dev ← load32 (← at_ s DEV)
    let _ ← cu .memFreeHost %[s]
    let _ ← cu .primaryCtxRelease %[dev]
    pure ())
  storeI64 zero slot

def createBuffer : StatusBody := do
  let (s ::ᵥ size ::ᵥ .nil) ← entryParams [.i64, .i64]
  withCtx s (failIf .sle size (← iconst64 0) (do
    let tmp ← at_ s TMP0
    onOk (← cu .memAlloc %[tmp, size]) (do
      let p ← load64 tmp
      -- the fill runs on the default stream; the buffer is zero once it lands
      onOk (← andThen (← cu .memsetD8 %[p, ← iconst32 0, size])
          (do cu .streamSynchronize %[← iconst64 0])) (do
        let (idx, e) ← push s BUFS
        let r ← ifte (jTys := [.i64]) .slt idx (← iconst64 0)
          (do
            let _ ← cu .memFree %[p]
            pure %[idx])
          (do
            storeI64 p e
            storeI64 size (← iaddImm e 8)
            pure %[idx])
        pure r.head))))

/-- An upload of `size` bytes from `src` to `off` in buffer `buf`, on `stream`
    (`none` for the synchronous copy). `exact` asks for the whole buffer. -/
def uploadAt (s : V .i64) (buf : V .i32) (off src size : V .i64) (stream : Option (V .i64))
    (exact : Bool) : Prog V L (V .i64) := do
  let zero ← iconst64 0
  let bad ← or8 (← icmp .slt buf (← iconst32 0))
    (← or8 (← icmp .slt off zero) (← or8 (← icmp .sle size zero) (← icmp .eq src zero)))
  failUnless0 bad (withBuf s buf fun p n => do
    let over ← if exact then icmp .ne size n else icmp .ugt (← iadd off size) n
    failUnless0 over (do
      let dst ← iadd p off
      match stream with
      -- From pageable memory the call returns once the bytes are staged, not
      -- once they land: waiting for the default stream is what makes the
      -- upload synchronous, as the entry point promises.
      | none => onOk (← cu .memcpyHtoD %[dst, src, size]) (do
          status (← cu .streamSynchronize %[← iconst64 0]))
      | some st => status (← cu .memcpyHtoDAsync %[dst, src, size, st])))

def downloadAt (s : V .i64) (buf : V .i32) (off dst size : V .i64) (stream : Option (V .i64))
    (exact : Bool) : Prog V L (V .i64) := do
  let zero ← iconst64 0
  let bad ← or8 (← icmp .slt buf (← iconst32 0))
    (← or8 (← icmp .slt off zero) (← or8 (← icmp .sle size zero) (← icmp .eq dst zero)))
  failUnless0 bad (withBuf s buf fun p n => do
    let over ← if exact then icmp .ne size n else icmp .ugt (← iadd off size) n
    failUnless0 over (do
      let src ← iadd p off
      match stream with
      | none => status (← cu .memcpyDtoH %[dst, src, size])
      | some st => status (← cu .memcpyDtoHAsync %[dst, src, size, st])))

def upload : StatusBody := do
  let (s ::ᵥ buf ::ᵥ src ::ᵥ size ::ᵥ .nil) ← entryParams [.i64, .i32, .i64, .i64]
  withCtx s (do uploadAt s buf (← iconst64 0) src size none true)

def uploadOffset : StatusBody := do
  let (s ::ᵥ buf ::ᵥ off ::ᵥ src ::ᵥ size ::ᵥ .nil) ← entryParams [.i64, .i32, .i64, .i64, .i64]
  withCtx s (uploadAt s buf off src size none false)

def uploadAsync : StatusBody := do
  let (s ::ᵥ buf ::ᵥ src ::ᵥ size ::ᵥ sid ::ᵥ .nil) ← entryParams [.i64, .i32, .i64, .i64, .i32]
  withCtx s (withStream s sid fun st _ => do uploadAt s buf (← iconst64 0) src size (some st) false)

def uploadOffsetAsync : StatusBody := do
  let (s ::ᵥ buf ::ᵥ off ::ᵥ src ::ᵥ size ::ᵥ sid ::ᵥ .nil) ←
    entryParams [.i64, .i32, .i64, .i64, .i64, .i32]
  withCtx s (withStream s sid fun st _ => uploadAt s buf off src size (some st) false)

def download : StatusBody := do
  let (s ::ᵥ buf ::ᵥ dst ::ᵥ size ::ᵥ .nil) ← entryParams [.i64, .i32, .i64, .i64]
  withCtx s (do downloadAt s buf (← iconst64 0) dst size none true)

def downloadOffset : StatusBody := do
  let (s ::ᵥ buf ::ᵥ off ::ᵥ dst ::ᵥ size ::ᵥ .nil) ← entryParams [.i64, .i32, .i64, .i64, .i64]
  withCtx s (downloadAt s buf off dst size none false)

def downloadAsync : StatusBody := do
  let (s ::ᵥ buf ::ᵥ dst ::ᵥ size ::ᵥ sid ::ᵥ .nil) ← entryParams [.i64, .i32, .i64, .i64, .i32]
  withCtx s (withStream s sid fun st _ => do downloadAt s buf (← iconst64 0) dst size (some st) false)

def freeBuffer : StatusBody := do
  let (s ::ᵥ buf ::ᵥ .nil) ← entryParams [.i64, .i32]
  withCtx s (withEntry s BUFS buf fun e => do
    let _ ← cu .memFree %[← load64 e]
    storeI64 (← iconst64 0) e
    iconst64 0)

def sync : StatusBody := do
  let (s ::ᵥ .nil) ← entryParams [.i64]
  withCtx s (do status (← cu .ctxSynchronize %[]))

/-- A new stream, non-blocking and ordered behind the work the default stream
    has been given, with its own cuBLAS handle when the program calls cuBLAS. -/
def streamCreate (blas : Bool) : StatusBody := do
  let (s ::ᵥ .nil) ← entryParams [.i64]
  withCtx s (do
    let tmp ← at_ s TMP0
    -- CU_STREAM_NON_BLOCKING
    onOk (← cu .streamCreate %[tmp, ← iconst32 1]) (do
      let st ← load64 tmp
      let fence ← load64 (← at_ s FENCE)
      let rc ← andThen (← cu .eventRecord %[fence, ← iconst64 0])
        (cu .streamWaitEvent %[st, fence, ← iconst32 0])
      let tmp1 ← at_ s TMP1
      storeI64 (← iconst64 0) tmp1
      let rc ← if blas then
          andThen rc (do andThen (← bl .create %[tmp1]) (do bl .setStream %[← load64 tmp1, st]))
        else pure rc
      let r ← ifte (jTys := [.i64]) .ne rc (← iconst32 0)
        (do
          let h ← load64 tmp1
          when .ne h (← iconst64 0) (do let _ ← bl .destroy %[h]; pure ())
          let _ ← cu .streamDestroy %[st]
          pure %[← iconst64 (-1)])
        (do pure %[← record s STREAMS st (some (← load64 tmp1))])
      pure r.head))

def streamSync : StatusBody := do
  let (s ::ᵥ sid ::ᵥ .nil) ← entryParams [.i64, .i32]
  withCtx s (withStream s sid fun st _ => do status (← cu .streamSynchronize %[st]))

/-- Destroying a stream orders the default stream behind its work first. -/
def streamDestroy : StatusBody := do
  let (s ::ᵥ sid ::ᵥ .nil) ← entryParams [.i64, .i32]
  withCtx s (withEntry s STREAMS sid fun e => do
    let st ← load64 e
    let fence ← load64 (← at_ s FENCE)
    let rc ← andThen (← cu .eventRecord %[fence, st])
      (do cu .streamWaitEvent %[← iconst64 0, fence, ← iconst32 0])
    let h ← load64 (← iaddImm e 8)
    when .ne h (← iconst64 0) (do let _ ← bl .destroy %[h]; pure ())
    let _ ← cu .streamDestroy %[st]
    storeI64 (← iconst64 0) e
    status rc)

def eventCreate : StatusBody := do
  let (s ::ᵥ .nil) ← entryParams [.i64]
  withCtx s (do
    let tmp ← at_ s TMP0
    onOk (← cu .eventCreate %[tmp, ← iconst32 0]) (do record s EVENTS (← load64 tmp)))

def eventRecord : StatusBody := do
  let (s ::ᵥ eid ::ᵥ sid ::ᵥ .nil) ← entryParams [.i64, .i32, .i32]
  withCtx s (withEntry s EVENTS eid fun e => withStream s sid fun st _ => do
    status (← cu .eventRecord %[← load64 e, st]))

def streamWaitEvent : StatusBody := do
  let (s ::ᵥ sid ::ᵥ eid ::ᵥ .nil) ← entryParams [.i64, .i32, .i32]
  withCtx s (withStream s sid fun st _ => withEntry s EVENTS eid fun e => do
    status (← cu .streamWaitEvent %[st, ← load64 e, ← iconst32 0]))

/-- The milliseconds between two events, as the bits of an `f32`. -/
def eventElapsedMsBits : StatusBody := do
  let (s ::ᵥ a ::ᵥ b ::ᵥ .nil) ← entryParams [.i64, .i32, .i32]
  withCtx s (withEntry s EVENTS a fun ea => withEntry s EVENTS b fun eb => do
    let tmp ← at_ s TMP0
    onOk (← cu .eventElapsedTime %[tmp, ← load64 ea, ← load64 eb]) (do
      sextend64 (← load32 tmp)))

def eventDestroy : StatusBody := do
  let (s ::ᵥ eid ::ᵥ .nil) ← entryParams [.i64, .i32]
  withCtx s (withEntry s EVENTS eid fun e => do
    let _ ← cu .eventDestroy %[← load64 e]
    storeI64 (← iconst64 0) e
    iconst64 0)

def graphBeginCapture : StatusBody := do
  let (s ::ᵥ sid ::ᵥ .nil) ← entryParams [.i64, .i32]
  -- CU_STREAM_CAPTURE_MODE_RELAXED
  withCtx s (withStream s sid fun st _ => do
    status (← cu .beginCapture %[st, ← iconst32 2]))

/-- Ending a capture instantiates what it captured; the graph itself is not
    kept. -/
def graphEndCapture : StatusBody := do
  let (s ::ᵥ sid ::ᵥ .nil) ← entryParams [.i64, .i32]
  withCtx s (withStream s sid fun st _ => do
    let tmp ← at_ s TMP0
    onOk (← cu .endCapture %[st, tmp]) (do
      let g ← load64 tmp
      let tmp1 ← at_ s TMP1
      let rc ← cu .graphInstantiate %[tmp1, g, ← iconst64 0]
      let _ ← cu .graphDestroy %[g]
      onOk rc (do record s GRAPHS (← load64 tmp1))))

/-- Uploading a graph ahead of its first launch saves that launch time; the
    launch uploads it otherwise. Here it checks its arguments and does nothing
    else. -/
def graphUpload : StatusBody := do
  let (s ::ᵥ gid ::ᵥ sid ::ᵥ .nil) ← entryParams [.i64, .i32, .i32]
  withCtx s (withEntry s GRAPHS gid fun _ => withStream s sid fun _ _ => iconst64 0)

def graphLaunch : StatusBody := do
  let (s ::ᵥ gid ::ᵥ sid ::ᵥ .nil) ← entryParams [.i64, .i32, .i32]
  withCtx s (withEntry s GRAPHS gid fun e => withStream s sid fun st _ => do
    status (← cu .graphLaunch %[← load64 e, st]))

def graphDestroy : StatusBody := do
  let (s ::ᵥ gid ::ᵥ .nil) ← entryParams [.i64, .i32]
  withCtx s (withEntry s GRAPHS gid fun e => do
    let _ ← cu .graphExecDestroy %[← load64 e]
    storeI64 (← iconst64 0) e
    iconst64 0)

def pinnedAlloc : StatusBody := do
  let (s ::ᵥ size ::ᵥ .nil) ← entryParams [.i64, .i64]
  withCtx s (failIf .sle size (← iconst64 0) (do
    let tmp ← at_ s TMP0
    onOk (← cu .memAllocHost %[tmp, size]) (do
      let p ← load64 tmp
      let r ← record s PINNED p (some size)
      when .slt r (← iconst64 0) (do let _ ← cu .memFreeHost %[p]; pure ())
      pure r)))

def pinnedPtr : StatusBody := do
  let (s ::ᵥ pid ::ᵥ .nil) ← entryParams [.i64, .i32]
  withCtx s (withEntry s PINNED pid fun e => load64 e)

/-- The page-locked address at `off`, when `off + len` lies inside the
    allocation. -/
def pinnedPtrAt : StatusBody := do
  let (s ::ᵥ pid ::ᵥ off ::ᵥ len ::ᵥ .nil) ← entryParams [.i64, .i32, .i64, .i64]
  withCtx s (do
    let zero ← iconst64 0
    failUnless0 (← or8 (← icmp .slt off zero) (← icmp .slt len zero)) (withEntry s PINNED pid fun e => do
      let n ← load64 (← iaddImm e 8)
      failIf .ugt (← iadd off len) n (do iadd (← load64 e) off)))

def pinnedFree : StatusBody := do
  let (s ::ᵥ pid ::ᵥ .nil) ← entryParams [.i64, .i32]
  withCtx s (withEntry s PINNED pid fun e => do
    let _ ← cu .memFreeHost %[← load64 e]
    storeI64 (← iconst64 0) e
    iconst64 0)

/-- Free or total device memory; `-1` when the device reports none. -/
def memInfo (total : Bool) : StatusBody := do
  let (s ::ᵥ .nil) ← entryParams [.i64]
  withCtx s (do
    let a ← at_ s TMP0
    let b ← at_ s TMP1
    onOk (← cu .memGetInfo %[a, b]) (do
      let t ← load64 b
      failIf .eq t (← iconst64 0) (if total then pure t else load64 a)))

def launch : StatusBody := do
  let (s ::ᵥ kptr ::ᵥ n ::ᵥ bind ::ᵥ gx ::ᵥ gy ::ᵥ gz ::ᵥ bx ::ᵥ by_ ::ᵥ bz ::ᵥ .nil) ←
    entryParams [.i64, .i64, .i32, .i64, .i32, .i32, .i32, .i32, .i32, .i32]
  withCtx s (do
    launchOn s kptr (← iconst64 0) (← at_ s MAIN) n bind gx gy gz bx by_ bz (← iconst64 0))

def launchNamed : StatusBody := do
  let (s ::ᵥ kptr ::ᵥ name ::ᵥ n ::ᵥ bind ::ᵥ gx ::ᵥ gy ::ᵥ gz ::ᵥ bx ::ᵥ by_ ::ᵥ bz ::ᵥ .nil) ←
    entryParams [.i64, .i64, .i64, .i32, .i64, .i32, .i32, .i32, .i32, .i32, .i32]
  withCtx s (do launchOn s kptr name name n bind gx gy gz bx by_ bz (← iconst64 0))

def launchOnStream : StatusBody := do
  let (s ::ᵥ kptr ::ᵥ n ::ᵥ bind ::ᵥ gx ::ᵥ gy ::ᵥ gz ::ᵥ bx ::ᵥ by_ ::ᵥ bz ::ᵥ sid ::ᵥ .nil) ←
    entryParams [.i64, .i64, .i32, .i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32]
  withCtx s (withStream s sid fun st _ => do
    launchOn s kptr (← iconst64 0) (← at_ s MAIN) n bind gx gy gz bx by_ bz st)

def launchNamedOnStream : StatusBody := do
  let (s ::ᵥ kptr ::ᵥ name ::ᵥ n ::ᵥ bind ::ᵥ gx ::ᵥ gy ::ᵥ gz ::ᵥ bx ::ᵥ by_ ::ᵥ bz ::ᵥ sid ::ᵥ .nil) ←
    entryParams [.i64, .i64, .i64, .i32, .i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32]
  withCtx s (withStream s sid fun st _ => do
    launchOn s kptr name name n bind gx gy gz bx by_ bz st)

-- ---------------------------------------------------------------------------
-- cuBLAS
-- ---------------------------------------------------------------------------

/-- `CUBLAS_OP_T` for a non-zero flag, `CUBLAS_OP_N` otherwise. -/
def opOf (t : V .i32) : Prog V L (V .i32) := do
  let z ← iconst32 0
  select (← icmp .ne t z) (← iconst32 1) z

/-- `ld` when it is non-zero, `dflt` otherwise. -/
def ldOr (ld dflt : V .i32) : Prog V L (V .i32) := do
  select (← icmp .ne ld (← iconst32 0)) ld dflt

/-- The scalars, stored where the call reads them from. -/
def scalars (s : V .i64) (alpha beta : V .i32) : Prog V L (V .i64 × V .i64) := do
  let pa ← at_ s ALPHA
  let pb ← at_ s BETA
  storeI32 alpha pa
  storeI32 beta pb
  pure (pa, pb)

def sgemvOn (s h : V .i64) (trans m n alpha a x beta y : V .i32) : Prog V L (V .i64) :=
  withBuf s a fun pA _ => withBuf s x fun pX _ => withBuf s y fun pY _ => do
    let (pa, pb) ← scalars s alpha beta
    let one ← iconst32 1
    status (← bl .sgemv %[h, ← opOf trans, m, n, pa, pA, m, pX, one, pb, pY, one])

def sgemv : StatusBody := do
  let (s ::ᵥ trans ::ᵥ m ::ᵥ n ::ᵥ alpha ::ᵥ a ::ᵥ x ::ᵥ beta ::ᵥ y ::ᵥ .nil) ←
    entryParams [.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32]
  withCtx s (do sgemvOn s (← load64 (← at_ s BLAS)) trans m n alpha a x beta y)

def sgemvOnStream : StatusBody := do
  let (s ::ᵥ trans ::ᵥ m ::ᵥ n ::ᵥ alpha ::ᵥ a ::ᵥ x ::ᵥ beta ::ᵥ y ::ᵥ sid ::ᵥ .nil) ←
    entryParams [.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32]
  withCtx s (withStream s sid fun _ h => sgemvOn s h trans m n alpha a x beta y)

/-- The arguments every product shares. -/
structure Gemm (V : ClifTy → Type) where
  ta : V .i32
  tb : V .i32
  m : V .i32
  n : V .i32
  k : V .i32
  alpha : V .i32
  a : V .i32
  sa : V .i64
  b : V .i32
  sb : V .i64
  beta : V .i32
  c : V .i32
  sc : V .i64
  batch : V .i32
  oa : V .i64
  ob : V .i64
  oc : V .i64
  la : V .i32
  lb : V .i32
  lc : V .i32

/-- A strided-batched product on handle `h`, of `ea`-byte inputs and an
    `ec`-byte result: `bf16` for `ea = 2`, `f32` otherwise. The entry points
    refuse a non-positive dimension or batch count and a negative stride,
    offset or leading dimension; a zero leading dimension asks for the
    packed one. -/
def gemmOn (s h : V .i64) (g : Gemm V) (ea : Nat) : Prog V L (V .i64) := do
  let z64 ← iconst64 0
  let z32 ← iconst32 0
  let bad ← or8 (← anyNonPos [g.m, g.n, g.k, g.batch])
    (← or8 (← anyNeg64 [g.sa, g.sb, g.sc, g.oa, g.ob, g.oc]) (← anyNeg32 [g.la, g.lb, g.lc]))
  failUnless0 bad (withBuf s g.a fun pA _ => withBuf s g.b fun pB _ => withBuf s g.c fun pC _ => do
    let ta ← icmp .ne g.ta z32
    let tb ← icmp .ne g.tb z32
    let lda ← ldOr g.la (← select ta g.k g.m)
    let ldb ← ldOr g.lb (← select tb g.n g.k)
    let ldc ← ldOr g.lc g.m
    let pA ← iadd pA (← imul g.oa (← iconst64 ea))
    let pB ← iadd pB (← imul g.ob (← iconst64 ea))
    let pC ← iadd pC (← ishlImm g.oc 2)
    let (pa, pb) ← scalars s g.alpha g.beta
    let opA ← opOf g.ta
    let opB ← opOf g.tb
    if ea == 2 then
      -- CUDA_R_16BF, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
      let bf ← iconst32 14
      status (← bl .gemmStridedBatchedEx %[h, opA, opB, g.m, g.n, g.k, pa, pA, bf, lda, g.sa,
        pB, bf, ldb, g.sb, pb, pC, z32, ldc, g.sc, g.batch, ← iconst32 68, ← iconst32 (-1)])
    else
      status (← bl .sgemmStridedBatched %[h, opA, opB, g.m, g.n, g.k, pa, pA, lda, g.sa,
        pB, ldb, g.sb, pb, pC, ldc, g.sc, g.batch]))

def sgemm : StatusBody := do
  let (s ::ᵥ ta ::ᵥ tb ::ᵥ m ::ᵥ n ::ᵥ k ::ᵥ alpha ::ᵥ a ::ᵥ sa ::ᵥ b ::ᵥ sb ::ᵥ beta ::ᵥ c ::ᵥ
      sc ::ᵥ batch ::ᵥ oa ::ᵥ ob ::ᵥ oc ::ᵥ la ::ᵥ lb ::ᵥ lc ::ᵥ .nil) ←
    entryParams [.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i64, .i32, .i64, .i32,
      .i32, .i64, .i32, .i64, .i64, .i64, .i32, .i32, .i32]
  withCtx s (do
    gemmOn s (← load64 (← at_ s BLAS))
      { ta, tb, m, n, k, alpha, a, sa, b, sb, beta, c, sc, batch, oa, ob, oc, la, lb, lc } 4)

def sgemmOnStream : StatusBody := do
  let (s ::ᵥ ta ::ᵥ tb ::ᵥ m ::ᵥ n ::ᵥ k ::ᵥ alpha ::ᵥ a ::ᵥ sa ::ᵥ b ::ᵥ sb ::ᵥ beta ::ᵥ c ::ᵥ
      sc ::ᵥ batch ::ᵥ sid ::ᵥ oa ::ᵥ ob ::ᵥ oc ::ᵥ la ::ᵥ lb ::ᵥ lc ::ᵥ .nil) ←
    entryParams [.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i64, .i32, .i64, .i32,
      .i32, .i64, .i32, .i32, .i64, .i64, .i64, .i32, .i32, .i32]
  withCtx s (withStream s sid fun _ h =>
    gemmOn s h { ta, tb, m, n, k, alpha, a, sa, b, sb, beta, c, sc, batch, oa, ob, oc, la, lb, lc } 4)

def gemmExBf16 : StatusBody := do
  let (s ::ᵥ ta ::ᵥ tb ::ᵥ m ::ᵥ n ::ᵥ k ::ᵥ alpha ::ᵥ a ::ᵥ b ::ᵥ beta ::ᵥ c ::ᵥ
      oa ::ᵥ ob ::ᵥ oc ::ᵥ la ::ᵥ lb ::ᵥ lc ::ᵥ .nil) ←
    entryParams [.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32,
      .i64, .i64, .i64, .i32, .i32, .i32]
  withCtx s (do
    let z ← iconst64 0
    gemmOn s (← load64 (← at_ s BLAS))
      { ta, tb, m, n, k, alpha, a, sa := z, b, sb := z, beta, c, sc := z,
        batch := ← iconst32 1, oa, ob, oc, la, lb, lc } 2)

def gemmStridedBatchedExBf16 : StatusBody := do
  let (s ::ᵥ ta ::ᵥ tb ::ᵥ m ::ᵥ n ::ᵥ k ::ᵥ alpha ::ᵥ a ::ᵥ sa ::ᵥ b ::ᵥ sb ::ᵥ beta ::ᵥ c ::ᵥ
      sc ::ᵥ batch ::ᵥ oa ::ᵥ ob ::ᵥ oc ::ᵥ la ::ᵥ lb ::ᵥ lc ::ᵥ .nil) ←
    entryParams [.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i64, .i32, .i64, .i32,
      .i32, .i64, .i32, .i64, .i64, .i64, .i32, .i32, .i32]
  withCtx s (do
    gemmOn s (← load64 (← at_ s BLAS))
      { ta, tb, m, n, k, alpha, a, sa, b, sb, beta, c, sc, batch, oa, ob, oc, la, lb, lc } 2)

/-- Write buffer `src`'s device pointer, `off` floats in, into entry `slot` of
    buffer `arr`: a device array of pointers, for a batch to read. -/
def ptrArray : StatusBody := do
  let (s ::ᵥ arr ::ᵥ slot ::ᵥ src ::ᵥ off ::ᵥ .nil) ← entryParams [.i64, .i32, .i32, .i32, .i64]
  withCtx s (do
    let bad ← or8 (← anyNeg32 [arr, slot, src]) (← anyNeg64 [off])
    failUnless0 bad (withBuf s src fun pS _ => withBuf s arr fun pA n => do
      let at8 ← ishlImm (← sextend64 slot) 3
      failIf .ugt (← iaddImm at8 8) n (do
        let tmp ← at_ s TMP0
        storeI64 (← iadd pS (← ishlImm off 2)) tmp
        status (← cu .memcpyHtoD %[← iadd pA at8, tmp, ← iconst64 8]))))

/-- A batch of packed products over three device arrays of pointers, on a
    stream's handle. -/
def sgemmBatchedOnStream : StatusBody := do
  let (s ::ᵥ ta ::ᵥ tb ::ᵥ m ::ᵥ n ::ᵥ k ::ᵥ alpha ::ᵥ a ::ᵥ b ::ᵥ beta ::ᵥ c ::ᵥ batch ::ᵥ sid ::ᵥ .nil) ←
    entryParams [.i64, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32, .i32]
  withCtx s (do
    let bad ← or8 (← anyNonPos [m, n, k, batch]) (← anyNeg32 [a, b, c])
    failUnless0 bad (withStream s sid fun _ h =>
      withBuf s a fun pA _ => withBuf s b fun pB _ => withBuf s c fun pC _ => do
        let z32 ← iconst32 0
        let lda ← select (← icmp .ne ta z32) k m
        let ldb ← select (← icmp .ne tb z32) n k
        let (pa, pb) ← scalars s alpha beta
        status (← bl .sgemmBatched %[h, ← opOf ta, ← opOf tb, m, n, k, pa, pA, lda, pB, ldb,
          pb, pC, m, batch])))

-- ---------------------------------------------------------------------------
-- The entry points this library implements
-- ---------------------------------------------------------------------------

/-- The function implementing an entry point, for a program that does or does
    not call cuBLAS; `none` for an entry point this library does not
    implement. -/
def implOf (blas : Bool) : Ffi → Option Impl
  | .cudaInit => some (.void (init blas))
  | .cudaCleanup => some (.void cleanup)
  | .cudaCreateBuffer => some (.status createBuffer)
  | .cudaUpload => some (.status upload)
  | .cudaUploadOffset => some (.status uploadOffset)
  | .cudaUploadAsync => some (.status uploadAsync)
  | .cudaUploadOffsetAsync => some (.status uploadOffsetAsync)
  | .cudaDownload => some (.status download)
  | .cudaDownloadOffset => some (.status downloadOffset)
  | .cudaDownloadAsync => some (.status downloadAsync)
  | .cudaFreeBuffer => some (.status freeBuffer)
  | .cudaSync => some (.status sync)
  | .cudaStreamCreate => some (.status (streamCreate blas))
  | .cudaStreamSync => some (.status streamSync)
  | .cudaStreamDestroy => some (.status streamDestroy)
  | .cudaEventCreate => some (.status eventCreate)
  | .cudaEventRecord => some (.status eventRecord)
  | .cudaStreamWaitEvent => some (.status streamWaitEvent)
  | .cudaEventElapsedMsBits => some (.status eventElapsedMsBits)
  | .cudaEventDestroy => some (.status eventDestroy)
  | .cudaGraphBeginCapture => some (.status graphBeginCapture)
  | .cudaGraphEndCapture => some (.status graphEndCapture)
  | .cudaGraphUpload => some (.status graphUpload)
  | .cudaGraphLaunch => some (.status graphLaunch)
  | .cudaGraphDestroy => some (.status graphDestroy)
  | .cudaPinnedAlloc => some (.status pinnedAlloc)
  | .cudaPinnedPtr => some (.status pinnedPtr)
  | .cudaPinnedPtrAt => some (.status pinnedPtrAt)
  | .cudaPinnedFree => some (.status pinnedFree)
  | .cudaMemInfoFree => some (.status (memInfo false))
  | .cudaMemInfoTotal => some (.status (memInfo true))
  | .cudaLaunch => some (.status launch)
  | .cudaLaunchNamed => some (.status launchNamed)
  | .cudaLaunchOnStream => some (.status launchOnStream)
  | .cudaLaunchNamedOnStream => some (.status launchNamedOnStream)
  | .cublasSgemv => some (.status sgemv)
  | .cublasSgemvOnStream => some (.status sgemvOnStream)
  | .cublasSgemm => some (.status sgemm)
  | .cublasSgemmOnStream => some (.status sgemmOnStream)
  | .cublasGemmExBf16 => some (.status gemmExBf16)
  | .cublasGemmStridedBatchedExBf16 => some (.status gemmStridedBatchedExBf16)
  | .cublasPtrArray => some (.status ptrArray)
  | .cublasSgemmBatchedOnStream => some (.status sgemmBatchedOnStream)
  | _ => none

def isBlas (f : Ffi) : Bool :=
  f == .cublasSgemv || f == .cublasSgemvOnStream || f == .cublasSgemm ||
  f == .cublasSgemmOnStream || f == .cublasGemmExBf16 || f == .cublasGemmStridedBatchedExBf16 ||
  f == .cublasPtrArray || f == .cublasSgemmBatchedOnStream

end AlgorithmLib.LibCuda
