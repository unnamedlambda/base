module
public import AlgorithmLib.Surface.Layout
meta import AlgorithmLib.Surface.Layout
public import AlgorithmLib.Surface.Prog
meta import AlgorithmLib.Surface.Prog
public import AlgorithmLib.Core.Artifact
meta import AlgorithmLib.Core.Artifact
public import AlgorithmLib.Surface.Link
meta import AlgorithmLib.Surface.Link
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# The FFI call wrappers, over `Prog`

`FFI.lean` says who the runtime exports and what each one's signature is. What
lives here is the *call* side: reading a context pointer out of its slot,
turning offsets into addresses, and issuing the call.

The arguments are typed: a buffer id is an `i32` because the signature says
so, and passing an `i64` where one belongs is a type error at the line that
writes it rather than a `wf` failure about a slot number.

No callee table is threaded. `LibFn` carries the id and the signature, so
`Prog.compileProg` derives the declarations a body needs from the calls it
makes.
-/

namespace AlgorithmLib.Prog

open AlgorithmLib.IR

-- ---------------------------------------------------------------------------
-- Contexts
--
-- Every subsystem keeps its context pointer in a slot of shared memory, written
-- by its `init` and read by everything after. The three functions below are the
-- whole pattern; the wrappers differ only in which slot and which callee.
-- ---------------------------------------------------------------------------

/-- The address of the slot holding a subsystem's context pointer. -/
def ctxSlotPtr (ptr : V .i64) (slotOffset : Nat) : Prog V L (V .i64) :=
  absAddr ptr slotOffset

/-- The context pointer itself. -/
def ctxPtr (ptr : V .i64) (slotOffset : Nat) : Prog V L (V .i64) := do
  load64 (← ctxSlotPtr ptr slotOffset)

/-- `init` takes the *slot* --- it writes the context there. Every subsystem's
    `init` and `cleanup` has this one signature, which the index states. -/
def initAt (f : LibFn) (hp : f.params = [ClifTy.i64] := by rfl)
    (ptr : V .i64) (slotOffset : Nat) : Prog V L Unit := do
  callLibVoid f (hp ▸ %[← ctxSlotPtr ptr slotOffset])

-- ---------------------------------------------------------------------------
-- Hash table
--
-- `init` and `cleanup` take the *slot*, everything
-- else the context pointer read out of it. The offset is a parameter because
-- generators do not agree on one -- `WordCountAlgorithm` keeps it at 0 and
-- `HProgCorpus` at 0x80 -- and `ContextSlots.ht` is only the default.
-- ---------------------------------------------------------------------------

def htCtxPtr (ptr : V .i64) (slotOffset : Nat := ContextSlots.ht) :
    Prog V L (V .i64) := ctxPtr ptr slotOffset

def htInit (ptr : V .i64) (slotOffset : Nat := ContextSlots.ht) : Prog V L Unit :=
  initAt .htInit rfl ptr slotOffset

def htCleanup (ptr : V .i64) (slotOffset : Nat := ContextSlots.ht) : Prog V L Unit :=
  initAt .htCleanup rfl ptr slotOffset

/-- Allocate the table itself. The context must already hold one. -/
def htCreate (ptr : V .i64) (slotOffset : Nat := ContextSlots.ht) :
    Prog V L (V .i32) := do
  callLib .htCreate %[← htCtxPtr ptr slotOffset]

/-- Look `key` up and write the value to `resultOff`; the result says whether
    it was found. Both offsets are relative to the base pointer. -/
def htLookup (ptr keyOff : V .i64) (keyLen : V .i32) (resultOff : V .i64)
    (slotOffset : Nat := ContextSlots.ht) : Prog V L (V .i32) := do
  let c ← htCtxPtr ptr slotOffset
  callLib .htLookup %[c, ← iadd ptr keyOff, keyLen, ← iadd ptr resultOff]

def htInsert (ptr keyOff : V .i64) (keyLen : V .i32) (valOff : V .i64)
    (valLen : V .i32) (slotOffset : Nat := ContextSlots.ht) : Prog V L Unit := do
  let c ← htCtxPtr ptr slotOffset
  callLibVoid .htInsert %[c, ← iadd ptr keyOff, keyLen, ← iadd ptr valOff, valLen]

/-- Add `addend` to `key`'s value, inserting it if absent; the result is the
    value after the addition. -/
def htIncrement (ptr keyOff : V .i64) (keyLen : V .i32) (addend : V .i64)
    (slotOffset : Nat := ContextSlots.ht) : Prog V L (V .i64) := do
  let c ← htCtxPtr ptr slotOffset
  callLib .htIncrement %[c, ← iadd ptr keyOff, keyLen, addend]

def htCount (ptr : V .i64) (slotOffset : Nat := ContextSlots.ht) :
    Prog V L (V .i32) := do
  callLib .htCount %[← htCtxPtr ptr slotOffset]

/-- The `index`th entry, written to `keyOutOff` and `valOutOff`. Iterating to
    `htCount` is how a program reads the table back out. -/
def htGetEntry (ptr : V .i64) (index : V .i32) (keyOutOff valOutOff : V .i64)
    (slotOffset : Nat := ContextSlots.ht) : Prog V L (V .i32) := do
  let c ← htCtxPtr ptr slotOffset
  callLib .htGetEntry %[c, index, ← iadd ptr keyOutOff, ← iadd ptr valOutOff]

-- ---------------------------------------------------------------------------
-- Files
-- ---------------------------------------------------------------------------

-- A file is named by the path in a field of memory: `filenameCap` bytes at
-- `filenameOff`, the path being its bytes up to the first zero, or all of them
-- (`Sem.readField`).

/-- Read a file into `room` bytes of shared memory; the result is the byte
    count. A longer file is read as far as it fits. -/
def readFile (ptr : V .i64) (filenameOff filenameCap dataOff room : Nat) : Prog V L (V .i64) := do
  let path ← iadd ptr (← iconst64 filenameOff)
  let dst ← iadd ptr (← iconst64 dataOff)
  ext (.base .fileReadToPtr) %[path, ← iconst64 filenameCap, dst, ← iconst64 0, ← iconst64 room]

/-- Write a region of shared memory over a file from `fileOffset`, keeping
    what the file holds past it; the result is the byte count. -/
def writeFile (ptr : V .i64) (filenameOff filenameCap srcOff : Nat) (fileOffset size : V .i64) :
    Prog V L (V .i64) := do
  let path ← iadd ptr (← iconst64 filenameOff)
  let src ← iadd ptr (← iconst64 srcOff)
  ext (.base .fileWriteFromPtr) %[path, ← iconst64 filenameCap, src, fileOffset, size]

/-- Replace a file with a region of shared memory. -/
def writeFile0 (ptr : V .i64) (filenameOff filenameCap srcOff : Nat) (size : V .i64) :
    Prog V L (V .i64) := do
  let _ ← ext (.base .fileSetLen) %[← iadd ptr (← iconst64 filenameOff), ← iconst64 filenameCap, ← iconst64 0]
  writeFile ptr filenameOff filenameCap srcOff (← iconst64 0) size

/-- `writeFile`, with the offsets as values. -/
def writeFileAt (ptr filenameOff filenameCap srcOff fileOffset size : V .i64) : Prog V L (V .i64) := do
  ext (.base .fileWriteFromPtr) %[← iadd ptr filenameOff, filenameCap, ← iadd ptr srcOff, fileOffset, size]

/-- `writeFile0`, with the offsets as values. -/
def replaceFileAt (ptr filenameOff filenameCap srcOff size : V .i64) : Prog V L (V .i64) := do
  let zero ← iconst64 0
  let _ ← ext (.base .fileSetLen) %[← iadd ptr filenameOff, filenameCap, zero]
  writeFileAt ptr filenameOff filenameCap srcOff zero size

/-- Read a line from standard input to `dstOff` in memory: the line with its
    newline, or as much of it as `cap` holds, the rest left for the next read;
    `0` at the end of input. -/
def stdinReadline (ptr : V .i64) (dstOff cap : V .i64) : Prog V L (V .i64) := do
  ext (.base .stdinReadline) %[← iadd ptr dstOff, cap]

/-- Write `size` bytes from `srcOff` in memory to standard output: `size`, or
    `-1`. -/
def stdoutWrite (ptr : V .i64) (srcOff size : V .i64) : Prog V L (V .i64) := do
  ext (.base .stdoutWrite) %[← iadd ptr srcOff, size]

/-- `(x + 3) &&& ~3` --- wgpu's `COPY_BUFFER_ALIGNMENT`. -/
def alignUp4 (v : V .i64) : Prog V L (V .i64) := do
  let c3 ← iconst64 3
  let sum ← iadd v c3
  let negFour ← iconst64 (-4)
  band sum negFour

-- ---------------------------------------------------------------------------
-- Typed fields
--
-- A `Layout.Fld` names a byte range in shared memory. The scalar forms reject
-- `.bytes n` through `IsScalar`; the `At` forms take a position inside a byte
-- region and a proof that the access fits, discharged by `omega` at the call.
-- ---------------------------------------------------------------------------

/-- A field's offset as a constant. -/
def fldOffset {t} (f : Layout.Fld t) : Prog V L (V .i64) := iconst64 f.offset

/-- A field's address: `base + offset`. -/
def fldAddr {t} (base : V .i64) (f : Layout.Fld t) : Prog V L (V .i64) :=
  absAddr base f.offset

/-- How a scalar field is read and written, chosen per field type rather than by
    a match, so `.bytes n` has no instance and cannot be passed.

    The stored value is an `i64` in every case: `u8` and `i32` fields are
    narrowed by the store and widened by the load, which is what the emitted
    `istore8`/`uload32` pair already did. -/
class IsScalarV (V : ClifTy → Type) (L : List ClifTy → List ClifTy → Type)
    (t : Layout.FieldTy) where
  scalarStore : V .i64 → V .i64 → Prog V L Unit
  scalarLoad  : V .i64 → Prog V L (V .i64)

instance : IsScalarV V L .u8 where
  scalarStore val addr := istore8 val addr
  scalarLoad  addr     := uload8_64 addr

instance : IsScalarV V L .i32 where
  scalarStore val addr := storeUnaligned val addr
  scalarLoad  addr     := uload32_64 addr

instance : IsScalarV V L .i64 where
  scalarStore val addr := storeUnaligned val addr
  scalarLoad  addr     := load64 addr

def fldStore {t} (base : V .i64) (f : Layout.Fld t) [inst : IsScalarV V L t]
    (val : V .i64) : Prog V L Unit := do
  inst.scalarStore val (← fldAddr base f)

def fldLoad {t} (base : V .i64) (f : Layout.Fld t) [inst : IsScalarV V L t] :
    Prog V L (V .i64) := do
  inst.scalarLoad (← fldAddr base f)

def fldStore32At {n ty} (base : V .i64) (f : Layout.Fld (.bytes n)) (i : Nat)
    (val : V ty) (_h : i + 4 ≤ n := by omega) : Prog V L Unit := do
  storeUnaligned val (← absAddr base (f.offset + i))

def fldLoad32At {n} (base : V .i64) (f : Layout.Fld (.bytes n)) (i : Nat)
    (_h : i + 4 ≤ n := by omega) : Prog V L (V .i64) := do
  uload32_64 (← absAddr base (f.offset + i))

/-- Read a file using typed field handles for the filename and data regions,
    as far as `room` holds: by default the whole data field. -/
def fldReadFile {ft dt} (ptr : V .i64) (filenameFld : Layout.Fld ft)
    (dataFld : Layout.Fld dt) (room : Nat := dt.size) : Prog V L (V .i64) :=
  readFile ptr filenameFld.offset ft.size dataFld.offset room

/-- Write a whole field to a file, from offset 0. -/
def fldWriteFile0 {ft st} (ptr : V .i64) (filenameFld : Layout.Fld ft)
    (srcFld : Layout.Fld st) (size : V .i64) : Prog V L (V .i64) :=
  writeFile0 ptr filenameFld.offset ft.size srcFld.offset size

-- ---------------------------------------------------------------------------
-- CUDA
--
-- `Raw` names a host pointer the caller owns; the plain form takes an offset
-- into shared memory and turns it into one. `Offset` names a position *inside*
-- the device buffer, so the host pointee need not be the whole allocation.
-- ---------------------------------------------------------------------------

-- ---------------------------------------------------------------------------
-- Calling this program's own functions
-- ---------------------------------------------------------------------------

/-- A function that calls each of `callees` in order.

    Composes stages without the caller having to build the call sequence
    itself; the callees are named by `u0:N`, so nothing here resolves a symbol.
    The fold gives each its place in the table the first time it is called, so
    calling one twice declares it once. Each stage is an entry point, so each
    is handed exactly what the wrapper was. -/
def sequenceWrapper (callees : List Nat) : Prog V L Unit := do
  let args ← entryArgs
  for c in callees do
    callLocalVoid (ps := HProg.ptrParams) (res := none) { callee := .local c } args

-- ---------------------------------------------------------------------------
-- Native code
--
-- Machine code the program carries as data, the way it carries a PTX kernel:
-- placed by `nativeLoad`, run by `nativeCall`. `Host.Sem` has no definition for
-- any of these, so a proof about a program that calls one does not reach past
-- the call; the CLIF path it would otherwise take is what the code is tested
-- against.
-- ---------------------------------------------------------------------------

/-- Place `len` bytes of machine code at `src` where they can run. Answers the
    address to call, or 0 if they could not be placed. -/
def nativeLoad (src len : V .i64) : Prog V L (V .i64) :=
  ext (.base .nativeLoad) %[src, len]

/-- Unmap what `nativeLoad` answered, other than 0, once: any other address
    is undefined. -/
def nativeFree (fn : V .i64) : Prog V L Unit :=
  ext (.base .nativeFree) %[fn]

/-- Run the code at `fn` on four arguments and answer what it returns: an
    indirect call, with nothing of the engine's between. `fn` must be an
    address `nativeLoad` answered and did not answer 0; nothing checks that
    when the call runs, so a generator calls only what it read back from
    where it kept `nativeLoad`'s answer. -/
def nativeCall (fn a b c d : V .i64) : Prog V L (V .i64) :=
  callLocal ({ callee := .native } : LocalRef [.i64, .i64, .i64, .i64, .i64] (some .i64))
    %[fn, a, b, c, d]

/-- Call the host's function at `fn` as a host calls an entry point: the
    input and its length, the output and its length, answering a status. How
    a program calls back a function a host handed it. -/
def hostCall (fn inp inLen out outLen : V .i64) : Prog V L (V .i64) :=
  callLocal ({ callee := .host [.i64, .i64, .i64, .i64] (some .i64) } :
      LocalRef [.i64, .i64, .i64, .i64, .i64] (some .i64))
    %[fn, inp, inLen, out, outLen]

/-- 1 on x86-64, 2 on AArch64, 0 on anything else. -/
def nativeArch : Prog V L (V .i32) :=
  ext (.base .nativeArch) %[]

/-- Whether the CPU has the feature named in the `cap`-byte field at `name`
    (`Sem.readField`): 1, 0, or -1 for a name the runtime does not know. -/
def cpuHas (name cap : V .i64) : Prog V L (V .i32) :=
  ext (.base .cpuHas) %[name, cap]

-- ---------------------------------------------------------------------------
-- Performance controls: `0` granted, `-1` declined, memory unchanged either way
-- ---------------------------------------------------------------------------

/-- Keep `len` bytes from `p` in physical memory. -/
def memLock (p len : V .i64) : Prog V L (V .i32) := ext (.base .memLock) %[p, len]
def memUnlock (p len : V .i64) : Prog V L (V .i32) := ext (.base .memUnlock) %[p, len]
/-- Ask for huge pages under the whole pages of `len` bytes from `p`. -/
def memAdviseHuge (p len : V .i64) : Prog V L (V .i32) := ext (.base .memAdviseHuge) %[p, len]
/-- The calling thread's priority set to `level`, from `0` the lowest to
    `99` the highest, on one scale for every system: `0` granted, `-1` not
    (raising past where the thread started takes privilege). -/
def threadPriority (level : V .i32) : Prog V L (V .i32) := ext (.base .threadPriority) %[level]

-- ---------------------------------------------------------------------------
-- Waiting on memory, and the clock
-- ---------------------------------------------------------------------------

/-- Sleep while the 32-bit word at `p` holds `expected`, until a wake on `p`,
    `timeoutNs` nanoseconds (none if negative), or for no reason. `p` is a
    word: 4-byte aligned, in memory. It is the system's address wait, so a
    host's threads waiting or waking with the system's own call meet a
    program's at the same words. Loop on the condition: a wait may end for no
    reason, and answers nothing. -/
def memWait (p : V .i64) (expected : V .i32) (timeoutNs : V .i64) : Prog V L Unit :=
  ext (.base .memWait) %[p, expected, timeoutNs]
/-- Wake one thread waiting on the word at `p` when `count` is 1, every one
    for any other `count` but 0, none for 0. -/
def memWake (p : V .i64) (count : V .i32) : Prog V L Unit := ext (.base .memWake) %[p, count]
/-- Nanoseconds on the system's monotonic clock, the one a host reads with
    the system's own call. -/
def clockNow : Prog V L (V .i64) := ext (.base .clockNow) %[]

/-- The length of the file at the path in the `cap`-byte field at `path`,
    `-1` for none. -/
def fileSize (path cap : V .i64) : Prog V L (V .i64) := ext (.base .fileSize) %[path, cap]

/-- The file at the path in the `cap`-byte field at `path` made `len` bytes
    long, created if there is none. -/
def fileSetLen (path cap len : V .i64) : Prog V L (V .i64) := ext (.base .fileSetLen) %[path, cap, len]

end AlgorithmLib.Prog
