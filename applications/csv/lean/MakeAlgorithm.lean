import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace CsvDemo

/-- String to raw UTF-8 bytes (no null terminator). -/
def rawBytes (s : String) : List UInt8 :=
  s.toUTF8.toList

/-- Encode a Nat as 4-byte big-endian (for LMDB keys). -/
def uint32BE (n : Nat) : List UInt8 :=
  [ UInt8.ofNat ((n >>> 24) &&& 0xFF),
    UInt8.ofNat ((n >>> 16) &&& 0xFF),
    UInt8.ofNat ((n >>> 8)  &&& 0xFF),
    UInt8.ofNat (n          &&& 0xFF) ]

-- Sample data -----------------------------------------------------------

def employees : List (Nat × String) :=
  [ (0,  "id,name,age,city,dept_id,salary\n"),
    (1,  "1,Alice,30,Seattle,1,85000\n"),
    (2,  "2,Bob,25,Portland,2,72000\n"),
    (3,  "3,Charlie,35,Seattle,1,95000\n"),
    (4,  "4,Diana,28,NYC,3,88000\n"),
    (5,  "5,Eve,32,SF,2,91000\n"),
    (6,  "6,Frank,45,Seattle,1,110000\n"),
    (7,  "7,Grace,29,Portland,3,78000\n"),
    (8,  "8,Hank,38,NYC,2,96000\n"),
    (9,  "9,Ivy,26,SF,1,82000\n"),
    (10, "10,Jack,41,Seattle,3,105000\n") ]

def departments : List (Nat × String) :=
  [ (0, "dept_id,dept_name,floor\n"),
    (1, "1,Engineering,3\n"),
    (2, "2,Marketing,5\n"),
    (3, "3,Sales,2\n") ]

/-- Filter keys for Seattle employees (header + Seattle rows). -/
def filterKeys : List Nat := [0, 1, 3, 6, 10]

/-- Join keys for department lookups (header + all depts). -/
def joinKeys : List Nat := [0, 1, 2, 3]

-- Memory layout constants -----------------------------------------------

def flagAlways    : Nat := 0x0000
def flag0         : Nat := 0x0008
def flag1         : Nat := 0x0010
def flag2         : Nat := 0x0018
def empHandle     : Nat := 0x0020
def deptHandle    : Nat := 0x0024
def empPathAddr   : Nat := 0x0030
def deptPathAddr  : Nat := 0x0130
def pathSize      : Nat := 256
def empDataAddr   : Nat := 0x0230
def empDataSize   : Nat := 384
def deptDataAddr  : Nat := 0x03B0
def deptDataSize  : Nat := 80
def scanResultAddr : Nat := 0x0400
def scanResultSize : Nat := 512
def getResultsAddr : Nat := 0x0600
def getResultSlot  : Nat := 64
def numGetResults  : Nat := 9
def filenamesAddr  : Nat := 0x0840
def filenameSlot   : Nat := 64
def scanBufAddr    : Nat := 0x0900
def scanBufSize    : Nat := 512
def filterBufAddr  : Nat := 0x0B00
def filterBufSize  : Nat := 256
def joinBufAddr    : Nat := 0x0C00
def joinBufSize    : Nat := 128
def totalPayload   : Nat := 0x0C80

-- Payload construction --------------------------------------------------

/-- Serialize a list of (key_nat, value_string) entries as concatenated
    [u32BE key][raw value bytes]. -/
def encodeEntries (entries : List (Nat × String)) : List UInt8 :=
  entries.foldl (fun acc (k, v) => acc ++ uint32BE k ++ rawBytes v) []

/-- Build the full payload as a byte list. -/
def buildPayload : List UInt8 :=
  let flags := uint64ToBytes 1 ++ zeros 24  -- flagAlways=1, flag0/1/2=0
  let handles := zeros 8
  let pad := zeros 8  -- padding from 0x0028 to 0x0030
  let empPath := padTo (stringToBytes "/tmp/csv-demo/employees/") pathSize
  let deptPath := padTo (stringToBytes "/tmp/csv-demo/departments/") pathSize
  let empData := padTo (encodeEntries employees) empDataSize
  let deptData := padTo (encodeEntries departments) deptDataSize
  let scanResult := zeros scanResultSize
  let getResults := zeros (getResultSlot * numGetResults)
  let filenames :=
    padTo (stringToBytes "scan.csv") filenameSlot ++
    padTo (stringToBytes "filter.csv") filenameSlot ++
    padTo (stringToBytes "join.csv") filenameSlot
  let scanBuf := zeros scanBufSize
  let filterBuf := zeros filterBufSize
  let joinBuf := zeros joinBufSize
  flags ++ handles ++ pad ++ empPath ++ deptPath ++
  empData ++ deptData ++ scanResult ++ getResults ++
  filenames ++ scanBuf ++ filterBuf ++ joinBuf

-- Offset computation for entry data ------------------------------------

/-- Compute byte offsets of (key_offset, val_offset, val_size) for each entry,
    given a base address. -/
def entryOffsets (base : Nat) (entries : List (Nat × String)) : List (Nat × Nat × Nat) :=
  let rec go (off : Nat) (es : List (Nat × String)) : List (Nat × Nat × Nat) :=
    match es with
    | [] => []
    | (_, v) :: rest =>
      let valSize := (rawBytes v).length
      let keyOff := off
      let valOff := off + 4
      (keyOff, valOff, valSize) :: go (off + 4 + valSize) rest
  go base entries

def empOffsets  : List (Nat × Nat × Nat) := entryOffsets empDataAddr employees
def deptOffsets : List (Nat × Nat × Nat) := entryOffsets deptDataAddr departments

-- Action builders -------------------------------------------------------

def mkLmdbOpen (handleAddr pathAddr : Nat) : Action :=
  { kind := .LmdbOpen, dst := u32 handleAddr, src := u32 pathAddr,
    offset := u32 pathSize, size := 0 }

def mkBeginWriteTxn (handle : Nat) : Action :=
  { kind := .LmdbBeginWriteTxn, dst := 0, src := 0, offset := u32 handle, size := 0 }

def mkCommitWriteTxn (handle : Nat) : Action :=
  { kind := .LmdbCommitWriteTxn, dst := 0, src := 0, offset := u32 handle, size := 0 }

def mkPut (keyOff valOff : Nat) (handle keySize valSize : Nat) : Action :=
  { kind := .LmdbPut, dst := u32 keyOff, src := u32 valOff,
    offset := u32 handle, size := u32 ((keySize <<< 16) ||| valSize) }

def mkGet (keyOff resultOff : Nat) (handle keySize : Nat) : Action :=
  { kind := .LmdbGet, dst := u32 keyOff, src := u32 resultOff,
    offset := u32 handle, size := u32 (keySize <<< 16) }

def mkCursorScan (resultOff handle maxEntries : Nat) : Action :=
  { kind := .LmdbCursorScan, dst := u32 resultOff, src := 0,
    offset := u32 handle, size := u32 maxEntries }

def mkMemCopy (dstOff srcOff byteCount : Nat) : Action :=
  { kind := .MemCopy, dst := u32 dstOff, src := u32 srcOff,
    offset := 0, size := u32 byteCount }

def mkFileWrite (filenameOff dataOff dataSize : Nat) : Action :=
  { kind := .FileWrite, dst := u32 filenameOff, src := u32 dataOff,
    offset := 0, size := u32 dataSize }

-- Ingest actions --------------------------------------------------------

def ingestActions : List Action :=
  let empPuts := empOffsets.map fun (kOff, vOff, vSz) =>
    mkPut kOff vOff 0 4 vSz
  let deptPuts := deptOffsets.map fun (kOff, vOff, vSz) =>
    mkPut kOff vOff 1 4 vSz
  [ mkLmdbOpen empHandle empPathAddr,
    mkLmdbOpen deptHandle deptPathAddr,
    mkBeginWriteTxn 0 ] ++
  empPuts ++
  [ mkCommitWriteTxn 0,
    mkBeginWriteTxn 1 ] ++
  deptPuts ++
  [ mkCommitWriteTxn 1 ]

-- Query actions ---------------------------------------------------------

/-- Find the entry offset tuple for a given key in an offset list, paired with original data. -/
def findKeyOffset (key : Nat) (entries : List (Nat × String))
    (offsets : List (Nat × Nat × Nat)) : Option (Nat × Nat × Nat) :=
  let pairs := entries.zip offsets
  match pairs.find? (fun ((k, _), _) => k == key) with
  | some (_, off) => some off
  | none => none

def mkGetsForKeys (keys : List Nat) (entries : List (Nat × String))
    (offsets : List (Nat × Nat × Nat)) (handle slotBase : Nat) : List Action :=
  let rec go (i : Nat) (ks : List Nat) : List Action :=
    match ks with
    | [] => []
    | k :: rest =>
      match findKeyOffset k entries offsets with
      | some (kOff, _, _) =>
        mkGet kOff (getResultsAddr + (slotBase + i) * getResultSlot) handle 4 ::
          go (i + 1) rest
      | none => go (i + 1) rest
  go 0 keys

def queryActions : List Action :=
  let scan := mkCursorScan scanResultAddr 0 100
  let filterGets := mkGetsForKeys filterKeys employees empOffsets 0 0
  let joinGets := mkGetsForKeys joinKeys departments deptOffsets 1 filterKeys.length
  [scan] ++ filterGets ++ joinGets

-- MemCopy actions: extract values from CursorScan result ----------------

/-- Compute MemCopy actions to extract values from a CursorScan result buffer.
    CursorScan layout: [u32 count][u16 klen, u16 vlen, key, val]...
    Each key is 4 bytes. We extract just the value portion of each entry. -/
def scanMemCopies (entries : List (Nat × String)) : List Action :=
  let rec go (srcOff dstOff : Nat) (es : List (Nat × String)) : List Action :=
    match es with
    | [] => []
    | (_, v) :: rest =>
      let valSize := (rawBytes v).length
      -- srcOff points to start of this entry's [u16 klen, u16 vlen] header
      -- value starts at srcOff + 4 (klen/vlen) + 4 (key bytes)
      let valSrc := srcOff + 4 + 4
      mkMemCopy dstOff valSrc valSize ::
        go (srcOff + 4 + 4 + valSize) (dstOff + valSize) rest
  -- First entry starts at scanResultAddr + 4 (skip u32 count)
  go (scanResultAddr + 4) scanBufAddr entries

/-- Total size of all values for a list of entries. -/
def totalValueSize (entries : List (Nat × String)) : Nat :=
  entries.foldl (fun acc (_, v) => acc + (rawBytes v).length) 0

-- MemCopy actions: extract values from LmdbGet results ------------------

/-- Build MemCopy actions to extract values from LmdbGet result buffers.
    LmdbGet layout: [u32 len][value bytes...]. We skip the 4-byte prefix.
    `keys` are the keys we looked up, `entries` is the full data to find value sizes,
    `getBaseSlot` is the starting slot index in getResults. -/
def getMemCopies (keys : List Nat) (entries : List (Nat × String))
    (getBaseSlot : Nat) (dstBase : Nat) : List Action :=
  let rec go (i dstOff : Nat) (ks : List Nat) : List Action :=
    match ks with
    | [] => []
    | k :: rest =>
      match entries.find? (fun (ek, _) => ek == k) with
      | some (_, v) =>
        let valSize := (rawBytes v).length
        let srcOff := getResultsAddr + (getBaseSlot + i) * getResultSlot + 4
        mkMemCopy dstOff srcOff valSize :: go (i + 1) (dstOff + valSize) rest
      | none => go (i + 1) dstOff rest
  go 0 dstBase keys

-- FileWrite actions -----------------------------------------------------

def scanTotalSize    : Nat := totalValueSize employees
def filterTotalSize  : Nat := totalValueSize (employees.filter fun (k, _) => filterKeys.contains k)
def joinTotalSize    : Nat := totalValueSize (departments.filter fun (k, _) => joinKeys.contains k)

def fileWriteActions : List Action :=
  [ mkFileWrite filenamesAddr scanBufAddr scanTotalSize,
    mkFileWrite (filenamesAddr + filenameSlot) filterBufAddr filterTotalSize,
    mkFileWrite (filenamesAddr + 2 * filenameSlot) joinBufAddr joinTotalSize ]

-- Control actions -------------------------------------------------------

def buildAlgorithm : Algorithm :=
  let lmdbActions := ingestActions ++ queryActions
  let memActions :=
    scanMemCopies employees ++
    getMemCopies filterKeys employees 0 filterBufAddr ++
    getMemCopies joinKeys departments filterKeys.length joinBufAddr
  let fileActions := fileWriteActions
  let workerActions := lmdbActions ++ memActions ++ fileActions
  let controlStart := 1 + workerActions.length
  let lmdbStart := 1
  let lmdbCount := lmdbActions.length
  let memStart  := lmdbStart + lmdbCount
  let memCount  := memActions.length
  let fileStart := memStart + memCount
  let fileCount := fileActions.length
  let jumpAction : Action :=
    { kind := .ConditionalJump, dst := u32 controlStart, src := u32 flagAlways,
      offset := 0, size := 1 }
  let control : List Action :=
    [ { kind := .AsyncDispatch, dst := u32 8, src := u32 lmdbStart,
        offset := u32 flag0, size := u32 lmdbCount },
      { kind := .Wait, dst := u32 flag0, src := 0, offset := 0, size := 0 },
      { kind := .AsyncDispatch, dst := u32 6, src := u32 memStart,
        offset := u32 flag1, size := u32 memCount },
      { kind := .Wait, dst := u32 flag1, src := 0, offset := 0, size := 0 },
      { kind := .AsyncDispatch, dst := u32 2, src := u32 fileStart,
        offset := u32 flag2, size := u32 fileCount },
      { kind := .Wait, dst := u32 flag2, src := 0, offset := 0, size := 0 } ]
  {
    actions := jumpAction :: workerActions ++ control,
    payloads := buildPayload,
    state := {
      regs_per_unit := 8,
      gpu_size := 0,
      file_buffer_size := 2_000_000,
      gpu_shader_offsets := [],
      cranelift_ir_offsets := []
    },
    units := {
      simd_units := 0,
      gpu_units := 0,
      file_units := 1,
      memory_units := 1,
      ffi_units := 0,
      hash_table_units := 0,
      lmdb_units := 1,
      cranelift_units := 0,
      backends_bits := 0xFFFFFFFF
    },
    simd_assignments := [],
    memory_assignments := [],
    file_assignments := [],
    ffi_assignments := [],
    hash_table_assignments := [],
    lmdb_assignments := [],
    gpu_assignments := [],
    cranelift_assignments := [],
    worker_threads := none,
    blocking_threads := none,
    stack_size := none,
    timeout_ms := some 30000,
    thread_name_prefix := some "csv-demo"
  }

end CsvDemo

def main : IO Unit := do
  let json := toJson CsvDemo.buildAlgorithm
  IO.println (Json.compress json)
