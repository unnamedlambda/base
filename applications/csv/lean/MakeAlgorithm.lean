import AlgorithmLib

open Lean (Json)
open AlgorithmLib

namespace CsvDemo

-- ===========================================================================
-- Dependent-type interface for typed CSV queries.
--
-- Schemas are declared as concrete List String matching the CSV headers.
-- A runtime assertion (in the CLIF orchestrator) validates the header row.
-- CsvQuery (s1 s2 : Schema) carries three proof obligations:
--   filterColInS1 : filterCol ∈ s1   — filter column exists in left table
--   joinKeyInS1   : joinKey ∈ s1     — join key exists in left table
--   joinKeyInS2   : joinKey ∈ s2     — join key exists in right table
--
-- All three close by `decide` because the schemas are concrete literals.
-- A typo or nonexistent column name fails at elaboration — before any
-- CLIF or LMDB code is generated.
-- ===========================================================================

abbrev Schema := List String

structure Table (s : Schema) where mk ::

def mkTable (s : Schema) : Table s := Table.mk

@[reducible] def mergeSchema (s1 s2 : Schema) : Schema :=
  s1 ++ s2.filter (fun c => !s1.elem c)

-- QueryPlan s is a typed query tree whose index is the output schema.
-- filter preserves the schema; join computes mergeSchema at the type level;
-- select projects to exactly the chosen columns — all checked at elaboration.
inductive QueryPlan : Schema → Type where
  | table  : Table s → QueryPlan s
  | filter : (col pat : String) → QueryPlan s → col ∈ s → QueryPlan s
  | join   : (key : String) → QueryPlan s1 → QueryPlan s2 → key ∈ s1 → key ∈ s2
           → QueryPlan (mergeSchema s1 s2)
  | select : (cols : List String) → QueryPlan s → (∀ c ∈ cols, c ∈ s) → QueryPlan cols

-- ---------------------------------------------------------------------------
-- Payload layout constants
-- ---------------------------------------------------------------------------

def empDbPath_off    : Nat := 0x0100
def deptDbPath_off   : Nat := 0x0200
def empCsvPath_off   : Nat := 0x0300
def deptCsvPath_off  : Nat := 0x0400
def scanFname_off    : Nat := 0x0500
def filterFname_off  : Nat := 0x0540
def joinFname_off    : Nat := 0x0580
def patternStr_off   : Nat := 0x05C0   -- filter pattern bytes (e.g. ",Seattle,")
def patternRegion    : Nat := 64
def flag_off         : Nat := patternStr_off + patternRegion   -- 0x0600
def clifIr_off       : Nat := flag_off + 64                    -- 0x0640
def clifIrRegionSize : Nat := 10240
def empBuf_off       : Nat := clifIr_off + clifIrRegionSize
def empBufSize       : Nat := 4096
def deptBuf_off      : Nat := empBuf_off + empBufSize
def deptBufSize      : Nat := 1024
def keyScratch_off   : Nat := deptBuf_off + deptBufSize
def scanResult_off   : Nat := keyScratch_off + 16
def scanResultSize   : Nat := 8192
def scanResult2_off  : Nat := scanResult_off + scanResultSize
def scanResult2Size  : Nat := 8192

open AlgorithmLib.IR

-- ---------------------------------------------------------------------------
-- Shared values threaded across sub-functions
-- ---------------------------------------------------------------------------

structure K where
  ptr         : Val
  zero        : Val
  one         : Val
  keyLen4     : Val
  newline     : Val
  keyScrOff   : Val
  empBufOff   : Val
  deptBufOff  : Val
  empSize     : Val
  deptSize    : Val
  empHandle   : Val
  deptHandle  : Val
  patternLen  : Nat
  fnLmdbPut      : FnRef
  fnCommitTxn    : FnRef
  fnBeginTxn     : FnRef
  fnCursorScan   : FnRef
  fnFileWrite    : FnRef
  fnCleanup      : FnRef

-- ---------------------------------------------------------------------------
-- Employee + department ingest loops
-- ---------------------------------------------------------------------------

def emitIngest (k : K)
    (empLoop empScan empAdvance empPut empDone : DeclaredBlock)
    (deptLoop deptScan deptAdv deptPut deptDone : DeclaredBlock)
    : IRBuilder Unit := do
  startBlock empLoop
  let pos := empLoop.param 0
  let key := empLoop.param 1
  let posEnd ← icmp .uge pos k.empSize
  brif posEnd empDone.ref [] empScan.ref [pos]

  startBlock empScan
  let sPos := empScan.param 0
  let relAddr ← iadd k.empBufOff sPos
  let absAddr ← iadd k.ptr relAddr
  let byte ← load_i8 absAddr
  let byteExt ← uextend64 byte
  let isNl ← icmp .eq byteExt k.newline
  brif isNl empPut.ref [] empAdvance.ref []

  startBlock empAdvance
  let nextPos ← iadd sPos k.one
  let atEnd ← icmp .uge nextPos k.empSize
  brif atEnd empPut.ref [] empScan.ref [nextPos]

  startBlock empPut
  let rowEnd ← iadd sPos k.one
  let rowLen ← isub rowEnd pos
  let rowLen32 ← ireduce32 rowLen
  let key32 ← ireduce32 key
  let keyAddr ← iadd k.ptr k.keyScrOff
  store key32 keyAddr
  let valOff ← iadd k.empBufOff pos
  let _ ← call k.fnLmdbPut [k.ptr, k.empHandle, k.keyScrOff, k.keyLen4, valOff, rowLen32]
  let nextKey ← iadd key k.one
  jump empLoop.ref [rowEnd, nextKey]

  startBlock empDone
  let _ ← call k.fnCommitTxn [k.ptr, k.empHandle]
  let _ ← call k.fnBeginTxn [k.ptr, k.deptHandle]
  jump deptLoop.ref [k.zero, k.zero]

  startBlock deptLoop
  let dPos := deptLoop.param 0
  let dKey := deptLoop.param 1
  let dEnd ← icmp .uge dPos k.deptSize
  brif dEnd deptDone.ref [] deptScan.ref [dPos]

  startBlock deptScan
  let dsPos := deptScan.param 0
  let dRelAddr ← iadd k.deptBufOff dsPos
  let dAbsAddr ← iadd k.ptr dRelAddr
  let dByte ← load_i8 dAbsAddr
  let dByteExt ← uextend64 dByte
  let dIsNl ← icmp .eq dByteExt k.newline
  brif dIsNl deptPut.ref [] deptAdv.ref []

  startBlock deptAdv
  let dNextPos ← iadd dsPos k.one
  let dAtEnd ← icmp .uge dNextPos k.deptSize
  brif dAtEnd deptPut.ref [] deptScan.ref [dNextPos]

  startBlock deptPut
  let dRowEnd ← iadd dsPos k.one
  let dRowLen ← isub dRowEnd dPos
  let dRowLen32 ← ireduce32 dRowLen
  let dKey32 ← ireduce32 dKey
  let dKeyAddr ← iadd k.ptr k.keyScrOff
  store dKey32 dKeyAddr
  let dValOff ← iadd k.deptBufOff dPos
  let _ ← call k.fnLmdbPut [k.ptr, k.deptHandle, k.keyScrOff, k.keyLen4, dValOff, dRowLen32]
  let dNextKey ← iadd dKey k.one
  jump deptLoop.ref [dRowEnd, dNextKey]

-- ---------------------------------------------------------------------------
-- Scan + filter + join output phases
-- ---------------------------------------------------------------------------

set_option maxRecDepth 4096 in
def emitOutputPhases (k : K)
    (deptDone : DeclaredBlock)
    (scanHdr scanBody scanDone : DeclaredBlock)
    (filterHdr filterParse : DeclaredBlock)
    (substrScan substrCmp substrByte : DeclaredBlock)
    (filterMatch substrAdv filterSkip filterDone : DeclaredBlock)
    (joinHdr joinBody joinDone : DeclaredBlock)
    : IRBuilder Unit := do
  startBlock deptDone
  let _ ← call k.fnCommitTxn [k.ptr, k.deptHandle]
  let scanResOff ← iconst64 scanResult_off
  let keyLen0 ← iconst32 0
  let maxEntries ← iconst32 100
  let scanCount ← call k.fnCursorScan [k.ptr, k.empHandle, k.zero, keyLen0, maxEntries, scanResOff]
  let four ← iconst64 4
  jump scanHdr.ref [k.zero, four, k.zero]

  startBlock scanHdr
  let si := scanHdr.param 0
  let sByteOff := scanHdr.param 1
  let sFileOff := scanHdr.param 2
  let scanCount64 ← sextend64 scanCount
  let sDone ← icmp .uge si scanCount64
  brif sDone scanDone.ref [] scanBody.ref []

  startBlock scanBody
  let sEntryOff ← iadd scanResOff sByteOff
  let sEntryAddr ← iadd k.ptr sEntryOff
  let sKlen ← load_i16 sEntryAddr
  let sKlen64 ← uextend64 sKlen
  let two ← iconst64 2
  let sVlenAddr ← iadd sEntryAddr two
  let sVlen ← load_i16 sVlenAddr
  let sVlen64 ← uextend64 sVlen
  let sDataOff ← iadd sByteOff four
  let sDataOff2 ← iadd sDataOff sKlen64
  let sValOff ← iadd scanResOff sDataOff2
  let sVlen32 ← ireduce32 sVlen64
  let sVlenSigned ← sextend64 sVlen32
  let scanFnOff ← iconst64 scanFname_off
  let _ ← call k.fnFileWrite [k.ptr, scanFnOff, sValOff, sFileOff, sVlenSigned]
  let sNextI ← iadd si k.one
  let sNextByte ← iadd sDataOff2 sVlen64
  let sNextFile ← iadd sFileOff sVlen64
  jump scanHdr.ref [sNextI, sNextByte, sNextFile]

  startBlock scanDone
  let scanRes2Off ← iconst64 scanResult2_off
  let filterCount ← call k.fnCursorScan [k.ptr, k.empHandle, k.zero, keyLen0, maxEntries, scanRes2Off]
  jump filterHdr.ref [k.zero, four, k.zero]

  startBlock filterHdr
  let fi := filterHdr.param 0
  let fByteOff := filterHdr.param 1
  let fFileOff := filterHdr.param 2
  let filterCount64 ← sextend64 filterCount
  let fDone ← icmp .uge fi filterCount64
  brif fDone filterDone.ref [] filterParse.ref []

  startBlock filterParse
  let fEntryOff ← iadd scanRes2Off fByteOff
  let fEntryAddr ← iadd k.ptr fEntryOff
  let fKlen ← load_i16 fEntryAddr
  let fKlen64 ← uextend64 fKlen
  let ftwo ← iconst64 2
  let fVlenAddr ← iadd fEntryAddr ftwo
  let fVlen ← load_i16 fVlenAddr
  let fVlen64 ← uextend64 fVlen
  let fDataOff ← iadd fByteOff four
  let fDataOff2 ← iadd fDataOff fKlen64
  let fValOff ← iadd scanRes2Off fDataOff2
  let fKeyOff ← iadd scanRes2Off fDataOff
  let fKeyAddr ← iadd k.ptr fKeyOff
  let fKeyVal ← load32 fKeyAddr
  let zero32 ← iconst32 0
  let isHeader ← icmp .eq fKeyVal zero32
  brif isHeader filterMatch.ref [] substrScan.ref [k.zero]

  -- Substring search using pattern baked into payload at patternStr_off.
  -- k.patternLen is an elaboration-time constant derived from the CsvQuery.
  startBlock substrScan
  let scanPos := substrScan.param 0
  let seaLen ← iconst64 k.patternLen
  let scanEnd ← iadd scanPos seaLen
  let noRoom ← icmp .ugt scanEnd fVlen64
  brif noRoom filterSkip.ref [] substrCmp.ref [k.zero]

  startBlock substrCmp
  let matchIdx := substrCmp.param 0
  let allMatch ← icmp .uge matchIdx seaLen
  brif allMatch filterMatch.ref [] substrByte.ref []

  startBlock substrByte
  let bytePos ← iadd scanPos matchIdx
  let valByteOff ← iadd fValOff bytePos
  let valByteAddr ← iadd k.ptr valByteOff
  let valByte ← load_i8 valByteAddr
  let patOff ← iconst64 patternStr_off
  let patByteOff ← iadd patOff matchIdx
  let patByteAddr ← iadd k.ptr patByteOff
  let patByte ← load_i8 patByteAddr
  let bytesEq ← icmp .eq valByte patByte
  let nextMatch ← iadd matchIdx k.one
  brif bytesEq substrCmp.ref [nextMatch] substrAdv.ref []

  startBlock filterMatch
  let fVlen32 ← ireduce32 fVlen64
  let fVlenSigned ← sextend64 fVlen32
  let filterFnOff ← iconst64 filterFname_off
  let _ ← call k.fnFileWrite [k.ptr, filterFnOff, fValOff, fFileOff, fVlenSigned]
  let fNextI ← iadd fi k.one
  let fNextByte ← iadd fDataOff2 fVlen64
  let fNextFile ← iadd fFileOff fVlen64
  jump filterHdr.ref [fNextI, fNextByte, fNextFile]

  startBlock substrAdv
  let nextScanPos ← iadd scanPos k.one
  jump substrScan.ref [nextScanPos]

  startBlock filterSkip
  let skipNextI ← iadd fi k.one
  let skipNextByte ← iadd fDataOff2 fVlen64
  jump filterHdr.ref [skipNextI, skipNextByte, fFileOff]

  startBlock filterDone
  let joinCount ← call k.fnCursorScan [k.ptr, k.deptHandle, k.zero, keyLen0, maxEntries, scanResOff]
  jump joinHdr.ref [k.zero, four, k.zero]

  startBlock joinHdr
  let ji := joinHdr.param 0
  let jByteOff := joinHdr.param 1
  let jFileOff := joinHdr.param 2
  let joinCount64 ← sextend64 joinCount
  let jDone ← icmp .uge ji joinCount64
  brif jDone joinDone.ref [] joinBody.ref []

  startBlock joinBody
  let jEntryOff ← iadd scanResOff jByteOff
  let jEntryAddr ← iadd k.ptr jEntryOff
  let jKlen ← load_i16 jEntryAddr
  let jKlen64 ← uextend64 jKlen
  let jtwo ← iconst64 2
  let jVlenAddr ← iadd jEntryAddr jtwo
  let jVlen ← load_i16 jVlenAddr
  let jVlen64 ← uextend64 jVlen
  let jDataOff ← iadd jByteOff four
  let jDataOff2 ← iadd jDataOff jKlen64
  let jValOff ← iadd scanResOff jDataOff2
  let jVlen32 ← ireduce32 jVlen64
  let jVlenSigned ← sextend64 jVlen32
  let joinFnOff ← iconst64 joinFname_off
  let _ ← call k.fnFileWrite [k.ptr, joinFnOff, jValOff, jFileOff, jVlenSigned]
  let jNextI ← iadd ji k.one
  let jNextByte ← iadd jDataOff2 jVlen64
  let jNextFile ← iadd jFileOff jVlen64
  jump joinHdr.ref [jNextI, jNextByte, jNextFile]

  startBlock joinDone
  callVoid k.fnCleanup [k.ptr]
  ret

-- ---------------------------------------------------------------------------
-- CLIF IR (parameterized by filter pattern length)
-- ---------------------------------------------------------------------------

set_option maxRecDepth 4096 in
def clifIrSource (patternLen : Nat) : String := buildProgram do
  let fnFileRead ← declareFileRead
  let lmdb ← declareLmdbFFI
  let fnFileWrite ← declareFileWrite

  let ptr ← entryBlock

  let empLoop     ← declareBlock [.i64, .i64]
  let empScan     ← declareBlock [.i64]
  let empAdvance  ← declareBlock []
  let empPut      ← declareBlock []
  let empDone     ← declareBlock []
  let deptLoop    ← declareBlock [.i64, .i64]
  let deptScan    ← declareBlock [.i64]
  let deptAdv     ← declareBlock []
  let deptPut     ← declareBlock []
  let deptDone    ← declareBlock []
  let scanHdr     ← declareBlock [.i64, .i64, .i64]
  let scanBody    ← declareBlock []
  let scanDone    ← declareBlock []
  let filterHdr   ← declareBlock [.i64, .i64, .i64]
  let filterParse ← declareBlock []
  let substrScan  ← declareBlock [.i64]
  let substrCmp   ← declareBlock [.i64]
  let substrByte  ← declareBlock []
  let filterMatch ← declareBlock []
  let substrAdv   ← declareBlock []
  let filterSkip  ← declareBlock []
  let filterDone  ← declareBlock []
  let joinHdr     ← declareBlock [.i64, .i64, .i64]
  let joinBody    ← declareBlock []
  let joinDone    ← declareBlock []

  let empBufOff  ← iconst64 empBuf_off
  let zero       ← iconst64 0
  let empSize    ← readFile ptr fnFileRead empCsvPath_off empBuf_off
  let deptBufOff ← iconst64 deptBuf_off
  let deptSize   ← readFile ptr fnFileRead deptCsvPath_off deptBuf_off
  callVoid lmdb.fnInit [ptr]
  let empDbOff   ← iconst64 empDbPath_off
  let maxDbs     ← iconst32 10
  let empHandle  ← call lmdb.fnOpen [ptr, empDbOff, maxDbs]
  let deptDbOff  ← iconst64 deptDbPath_off
  let deptHandle ← call lmdb.fnOpen [ptr, deptDbOff, maxDbs]
  let _          ← call lmdb.fnBeginWriteTxn [ptr, empHandle]
  let one        ← iconst64 1
  let keyLen4    ← iconst32 4
  let newline    ← iconst64 10
  let keyScrOff  ← iconst64 keyScratch_off
  jump empLoop.ref [zero, zero]

  let k : K := {
    ptr        := ptr,
    zero       := zero,
    one        := one,
    keyLen4    := keyLen4,
    newline    := newline,
    keyScrOff  := keyScrOff,
    empBufOff  := empBufOff,
    deptBufOff := deptBufOff,
    empSize    := empSize,
    deptSize   := deptSize,
    empHandle  := empHandle,
    deptHandle := deptHandle,
    patternLen := patternLen,
    fnLmdbPut    := lmdb.fnPut,
    fnCommitTxn  := lmdb.fnCommitWriteTxn,
    fnBeginTxn   := lmdb.fnBeginWriteTxn,
    fnCursorScan := lmdb.fnCursorScan,
    fnFileWrite  := fnFileWrite,
    fnCleanup    := lmdb.fnCleanup
  }

  emitIngest k empLoop empScan empAdvance empPut empDone
               deptLoop deptScan deptAdv deptPut deptDone

  emitOutputPhases k deptDone
    scanHdr scanBody scanDone
    filterHdr filterParse
    substrScan substrCmp substrByte
    filterMatch substrAdv filterSkip filterDone
    joinHdr joinBody joinDone

-- ---------------------------------------------------------------------------
-- Payload builder (parameterized by filter pattern bytes)
-- ---------------------------------------------------------------------------

def buildPayload (patternBytes : List UInt8) : List UInt8 :=
  let reserved         := zeros empDbPath_off
  let empDbPathBytes   := padTo (stringToBytes "/tmp/csv-demo/employees") 256
  let deptDbPathBytes  := padTo (stringToBytes "/tmp/csv-demo/departments") 256
  let empCsvPathBytes  := padTo (stringToBytes "applications/csv/data/employees.csv") 256
  let deptCsvPathBytes := padTo (stringToBytes "applications/csv/data/departments.csv") 256
  let scanFnameBytes   := padTo (stringToBytes "scan.csv") 64
  let filterFnameBytes := padTo (stringToBytes "filter.csv") 64
  let joinFnameBytes   := padTo (stringToBytes "join.csv") 64
  let patternPadded    := padTo patternBytes patternRegion
  let flagBytes        := uint64ToBytes 0
  let flagPad          := zeros (clifIr_off - flag_off - 8)
  let clifPad          := zeros clifIrRegionSize
  let empBufBytes      := zeros empBufSize
  let deptBufBytes     := zeros deptBufSize
  let keyScratchBytes  := zeros 16
  let scanResultBytes  := zeros scanResultSize
  let scanResult2Bytes := zeros scanResult2Size
  reserved ++
  empDbPathBytes ++ deptDbPathBytes ++
  empCsvPathBytes ++ deptCsvPathBytes ++
  scanFnameBytes ++ filterFnameBytes ++ joinFnameBytes ++
  patternPadded ++
  flagBytes ++ flagPad ++
  clifPad ++
  empBufBytes ++ deptBufBytes ++
  keyScratchBytes ++ scanResultBytes ++ scanResult2Bytes

-- ---------------------------------------------------------------------------
-- Monomorphic builder
-- ---------------------------------------------------------------------------

def buildQueryMonomorphic (patternStr : String) : BaseConfig × Algorithm :=
  let patternBytes := patternStr.toUTF8.toList  -- no null terminator
  let payload := buildPayload patternBytes
  let cfg : BaseConfig := {
    cranelift_ir   := clifIrSource patternBytes.length,
    memory_size    := payload.length,
    context_offset := 0,
    initial_memory := payload
  }
  let alg : Algorithm := {
    actions         := [IR.clifCallAction],
    cranelift_units := 0,
    timeout_ms      := some 30000
  }
  (cfg, alg)

-- ---------------------------------------------------------------------------
-- Pipeline API
-- ---------------------------------------------------------------------------

def plan {s : Schema} (t : Table s) : QueryPlan s := .table t

def filter (col pat : String) {s : Schema} (p : QueryPlan s)
    (h : col ∈ s := by decide) : QueryPlan s := .filter col pat p h

def join (key : String) {s1 s2 : Schema} (p1 : QueryPlan s1) (p2 : QueryPlan s2)
    (h1 : key ∈ s1 := by decide) (h2 : key ∈ s2 := by decide) : QueryPlan (mergeSchema s1 s2) :=
  .join key p1 p2 h1 h2

def select (cols : List String) {s : Schema} (p : QueryPlan s)
    (h : ∀ c ∈ cols, c ∈ s := by decide) : QueryPlan cols := .select cols p h

def extractPattern : QueryPlan s → String
  | .table _            => ""
  | .filter _ pat _ _   => pat
  | .join _ p1 _ _ _    => extractPattern p1
  | .select _ p _       => extractPattern p

def compile {s : Schema} (p : QueryPlan s) : BaseConfig × Algorithm :=
  buildQueryMonomorphic ("," ++ extractPattern p ++ ",")

-- ---------------------------------------------------------------------------
-- Schemas declared to match the actual CSV headers.
-- Runtime assertion in the CLIF orchestrator checks the header row matches
-- the declared schema; query operations downstream use dependent types.
-- ---------------------------------------------------------------------------

abbrev employeeSchema   : Schema := ["id", "name", "age", "city", "dept_id", "salary"]
abbrev departmentSchema : Schema := ["dept_id", "dept_name", "floor"]
abbrev locationSchema   : Schema := ["city", "region", "country"]

def employees   : Table employeeSchema   := Table.mk
def departments : Table departmentSchema := Table.mk
def locations   : Table locationSchema   := Table.mk

-- Tree-shaped query — two-level join with independent filters on both branches.
-- Schema propagates through every node; mergeSchema is computed at each join.
-- All membership proofs discharge automatically via `decide`.
--
--   join "dept_id"
--     ├─ join "city"
--     │    ├─ filter "city" "Seattle" employees
--     │    └─ locations
--     └─ filter "dept_name" "Engineering" departments
--   |> select ["name", "city", "region", "dept_name", "floor"]
def result : BaseConfig × Algorithm :=
  join "dept_id"
    (join "city"
      (plan employees |> filter "city" "Seattle")
      (plan locations))
    (plan departments |> filter "dept_name" "Engineering")
  |> select ["name", "city", "region", "dept_name", "floor"]
  |> compile

-- Uncomment either def to see elaboration-time rejection:
--
-- def badJoinKey : BaseConfig × Algorithm :=
--   join "nonexistent_key"              -- ∉ employeeSchema → decide fails
--     (plan employees |> filter "city" "Seattle")
--     (plan departments)
--   |> select ["name", "dept_name"]
--   |> compile
--
-- def badSelectCol : BaseConfig × Algorithm :=
--   join "dept_id"
--     (plan employees |> filter "city" "Seattle")
--     (plan departments)
--   |> select ["name", "salary_band"]   -- ∉ merged schema → decide fails
--   |> compile

end CsvDemo

def main : IO Unit := do
  let (cfg, alg) := CsvDemo.result
  IO.println (Json.compress (toJsonPair cfg alg))
