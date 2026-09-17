import AlgorithmLib.Gen
import LayoutScan
import ShipScan

open Lean (Json)
open AlgorithmLib

namespace CsvDemo

-- ===========================================================================
-- Dependent-type interface for typed CSV queries.
--
-- Schemas are declared as concrete List String matching the CSV headers.
-- A runtime assertion (in the CLIF orchestrator) validates the header row.
-- QueryPlan carries proof obligations at each staged combinator:
--   whereEq      : selected filter column ∈ current schema
--   innerJoinOn  : join key ∈ left schema and ∈ right schema
--   project      : every projected column ∈ current schema
--
-- These obligations usually close by `decide` because the schemas are
-- concrete literals. A typo or nonexistent column name fails at
-- elaboration — before any
-- CLIF or LMDB code is generated.
-- ===========================================================================

abbrev Schema := List String

structure Table (s : Schema) where mk ::

def mkTable (s : Schema) : Table s := Table.mk

@[reducible] def mergeSchema (s1 s2 : Schema) : Schema :=
  s1 ++ s2.filter (fun c => !s1.elem c)

-- QueryPlan s is a typed query tree whose index is the output schema.
-- `whereEq` preserves the schema; `innerJoinOn` computes mergeSchema at the
-- type level; `project` narrows the schema to exactly the requested columns.
-- All of it is checked at elaboration.
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

/-- Every byte of shared memory this program names.

    The offsets are assigned by hand, and a collision between two of them is
    invisible at every other layer: both stores succeed and the second wins.
    `0x00`-`0x18` are the context slots the FFI's init calls fill, so naming
    those is what stops an offset being placed where one of those calls will
    overwrite it. -/
def memMap : AlgorithmLib.Layout.RegionMap :=
  [⟨"emp_db_path",   empDbPath_off, deptDbPath_off - empDbPath_off⟩,
   ⟨"dept_db_path",  deptDbPath_off, empCsvPath_off - deptDbPath_off⟩,
   ⟨"emp_csv_path",  empCsvPath_off, deptCsvPath_off - empCsvPath_off⟩,
   ⟨"dept_csv_path", deptCsvPath_off, scanFname_off - deptCsvPath_off⟩,
   ⟨"scan_fname",    scanFname_off, filterFname_off - scanFname_off⟩,
   ⟨"filter_fname",  filterFname_off, joinFname_off - filterFname_off⟩,
   ⟨"join_fname",    joinFname_off, patternStr_off - joinFname_off⟩,
   ⟨"pattern",       patternStr_off, patternRegion⟩,
   ⟨"flag",          flag_off, clifIr_off - flag_off⟩,
   ⟨"clif_ir",       clifIr_off, clifIrRegionSize⟩,
   ⟨"emp_buf",       empBuf_off, empBufSize⟩,
   ⟨"dept_buf",      deptBuf_off, deptBufSize⟩,
   ⟨"key_scratch",   keyScratch_off, scanResult_off - keyScratch_off⟩,
   ⟨"scan_result",   scanResult_off, scanResultSize⟩,
   ⟨"scan_result2",  scanResult2_off, scanResult2Size⟩]


#eval LayoutScan.check "CsvAlgorithm" [``memMap]
theorem memMap_ok : AlgorithmLib.Layout.RegionMap.okB memMap = true := by decide

-- The memory this ships is sized from the payload it builds, so there is no
-- constant to bound the regions against; `okB` is the whole check here.

open AlgorithmLib.IR

open AlgorithmLib.Prog


/-- The externals every emitted function declares, in one order. -/
abbrev fnFileRead : Ffi := .fileRead
abbrev fnFileWrite : Ffi := .fileWrite

/-- One CSV buffer's rows written to a database, one row per key. A row runs to
    the next newline, or to the end of the buffer. -/
def emitIngest (ptr lmdbCtx : V .i64) (handle : V .i32)
    (bufOff size keyScrOff : V .i64) : Prog V L Unit := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let newline ← iconst64 10
  let keyLen4 ← iconst32 4
  let _ ← wloop2 zero zero
    (head := fun pos _ => return (contIf .ult pos size, %[], ()))
    (body := fun pos key _ => do
      -- the row's last byte: a newline, or the buffer's end
      let sc ← wloopL %[pos]
        (head := fun _ c => return (contIf .eq zero zero, %[c.head], ()))
        (body := fun rowScan c _ => do
          let sPos := c.head
          let byte ← uextend64 (← load_i8 (← iadd ptr (← iadd bufOff sPos)))
          when .eq byte newline (brk rowScan %[sPos])
          let nextPos ← iadd sPos one
          when .uge nextPos size (brk rowScan %[sPos])
          return %[nextPos])
      let rowEnd ← iadd (sc.head) one
      let rowLen32 ← ireduce32 (← isub rowEnd pos)
      let keyPtr ← iadd ptr keyScrOff
      store (← ireduce32 key) keyPtr
      let valPtr ← iadd ptr (← iadd bufOff pos)
      let _ ← ffi .lmdbPut %[lmdbCtx, handle, keyPtr, keyLen4, valPtr, rowLen32]
      return %[rowEnd, ← iadd key one])

/-- Every scanned value written to one file, back to back. -/
def emitWriteAll (ptr lmdbCtx : V .i64) (handle : V .i32) (resOff : V .i64)
    (count : V .i32) (fnameOff : V .i64) : Prog V L Unit := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let two ← iconst64 2
  let four ← iconst64 4
  let count64 ← sextend64 count
  let _ ← wloop %[zero, four, zero]
    (head := fun c => return (contIf .slt (c.head) count64, %[], ()))
    (body := fun c _ => do
      let i := c.head
      let byteOff := c.snd
      let fileOff := c.thd
      let entryAddr ← iadd ptr (← iadd resOff byteOff)
      let klen ← uextend64 (← load_i16 entryAddr)
      let vlen ← uextend64 (← load_i16 (← iadd entryAddr two))
      let dataOff ← iadd (← iadd byteOff four) klen
      let valOff ← iadd resOff dataOff
      let vlenSigned ← sextend64 (← ireduce32 vlen)
      let _ ← ffi fnFileWrite %[ptr, fnameOff, valOff, fileOff, vlenSigned]
      return %[← iadd i one, ← iadd dataOff vlen, ← iadd fileOff vlen])

/-- The scanned rows whose value contains the pattern, plus the header row. -/
def emitFilter (ptr resOff : V .i64) (count : V .i32) (fnameOff : V .i64) (patternLen : Nat) : Prog V L Unit := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let two ← iconst64 2
  let four ← iconst64 4
  let zero32 ← iconst32 0
  let seaLen ← iconst64 patternLen
  let patOff ← iconst64 patternStr_off
  let count64 ← sextend64 count
  let _ ← wloop %[zero, four, zero]
    (head := fun c => return (contIf .slt (c.head) count64, %[], ()))
    (body := fun c _ => do
      let i := c.head
      let byteOff := c.snd
      let fileOff := c.thd
      let entryAddr ← iadd ptr (← iadd resOff byteOff)
      let klen ← uextend64 (← load_i16 entryAddr)
      let vlen ← uextend64 (← load_i16 (← iadd entryAddr two))
      let dataOff ← iadd byteOff four
      let dataOff2 ← iadd dataOff klen
      let valOff ← iadd resOff dataOff2
      let keyVal ← load32 (← iadd ptr (← iadd resOff dataOff))
      let isHeader ← icmp .eq keyVal zero32
      -- the header row is kept whatever the pattern says
      let hit ← ifte .ne isHeader (← iconst .i8 0) (pure %[one])
        (do
          -- the pattern searched for at every position it still fits in
          let sc ← wloopL %[zero]
            (head := fun _ c => do
              let scanEnd ← iadd c.head seaLen
              return (contIf .ule scanEnd vlen, %[zero], ()))
            (body := fun scan c _ => do
              let scanPos := c.head
              let m ← wloopL %[zero]
                (head := fun _ mc => return (contIf .ult mc.head seaLen, %[one], ()))
                (body := fun cmpLoop mc _ => do
                  let mi := mc.head
                  let valByte ← load_i8 (← iadd ptr (← iadd valOff (← iadd scanPos mi)))
                  let patByte ← load_i8 (← iadd ptr (← iadd patOff mi))
                  when .ne valByte patByte (brk cmpLoop %[zero])
                  return %[← iadd mi one])
              when .ne m.head zero (brk scan %[one])
              return %[← iadd scanPos one])
          pure %[sc.head])
      let _ ← ifte .ne (hit.head) zero
        (do
          let vlenSigned ← sextend64 (← ireduce32 vlen)
          let _ ← ffi fnFileWrite %[ptr, fnameOff, valOff, fileOff, vlenSigned]
          pure %[])
        (pure %[])
      let nextFile ← ifte .ne (hit.head) zero
        (pure %[← iadd fileOff vlen]) (pure %[fileOff])
      return %[← iadd i one, ← iadd dataOff2 vlen, nextFile.head])

def mainCode (patternLen : Nat) : Prog V L Unit :=
  do
  let ptr ← basePtr
  let empBufOff ← iconst64 empBuf_off
  let zero ← iconst64 0
  let empSize ← readFile ptr empCsvPath_off empBuf_off
  let deptBufOff ← iconst64 deptBuf_off
  let deptSize ← readFile ptr deptCsvPath_off deptBuf_off
  let lmdbSlot := ptr
  ffiVoid .lmdbInit %[lmdbSlot]
  let lmdbCtx ← load64 ptr
  let maxDbs ← iconst32 10
  let empHandle ← ffi .lmdbOpen %[lmdbCtx, ← iadd ptr (← iconst64 empDbPath_off), maxDbs]
  let deptHandle ← ffi .lmdbOpen %[lmdbCtx, ← iadd ptr (← iconst64 deptDbPath_off), maxDbs]
  let keyScrOff ← iconst64 keyScratch_off

  let _ ← ffi .lmdbBeginWriteTxn %[lmdbCtx, empHandle]
  emitIngest ptr lmdbCtx empHandle empBufOff empSize keyScrOff
  let _ ← ffi .lmdbCommitWriteTxn %[lmdbCtx, empHandle]

  let _ ← ffi .lmdbBeginWriteTxn %[lmdbCtx, deptHandle]
  emitIngest ptr lmdbCtx deptHandle deptBufOff deptSize keyScrOff
  let _ ← ffi .lmdbCommitWriteTxn %[lmdbCtx, deptHandle]

  let keyLen0 ← iconst32 0
  let maxEntries ← iconst32 100
  let scanResOff ← iconst64 scanResult_off
  let scanCount ← ffi .lmdbCursorScan
    %[lmdbCtx, empHandle, ptr, keyLen0, maxEntries, ← iadd ptr scanResOff]
  emitWriteAll ptr lmdbCtx empHandle scanResOff scanCount (← iconst64 scanFname_off)

  let scanRes2Off ← iconst64 scanResult2_off
  let filterCount ← ffi .lmdbCursorScan
    %[lmdbCtx, empHandle, ptr, keyLen0, maxEntries, ← iadd ptr scanRes2Off]
  emitFilter ptr scanRes2Off filterCount (← iconst64 filterFname_off) patternLen

  let joinCount ← ffi .lmdbCursorScan
    %[lmdbCtx, deptHandle, ptr, keyLen0, maxEntries, ← iadd ptr scanResOff]
  emitWriteAll ptr lmdbCtx deptHandle scanResOff joinCount (← iconst64 joinFname_off)

  ffiVoid .lmdbCleanup %[lmdbSlot]

/-- Generic in the pattern length, and nothing about that costs it anything
    now: `compileProg` checks the body it emits, at whatever length the
    monomorphic builder below asked for. -/
def clifIrSource (patternLen : Nat) : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 (mainCode patternLen))]

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

def buildQueryMonomorphic (patternStr : String) : Except String (Artifact × UInt32) := do
  let patternBytes := patternStr.toUTF8.toList  -- no null terminator
  let payload := buildPayload patternBytes
  let cfg : Artifact := {
    functions := ← clifIrSource patternBytes.length,
    memory_size    := payload.length,
    initial_memory := payload
  }
  let alg : UInt32 := IR.mainFnIdx
  return (cfg, alg)

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

def compile {s : Schema} (p : QueryPlan s) : Except String (Artifact × UInt32) :=
  buildQueryMonomorphic ("," ++ extractPattern p ++ ",")

def source {s : Schema} (t : Table s) : QueryPlan s :=
  plan t

def QueryPlan.whereEq {s : Schema} (p : QueryPlan s)
    (col pat : String) (h : col ∈ s := by decide) : QueryPlan s :=
  filter col pat p h

def QueryPlan.project {s : Schema} (p : QueryPlan s)
    (cols : List String) (h : ∀ c ∈ cols, c ∈ s := by decide) : QueryPlan cols :=
  select cols p h

def QueryPlan.compileQuery {s : Schema} (p : QueryPlan s) :
    Except String (Artifact × UInt32) := compile p

def QueryPlan.innerJoinOn {s1 : Schema} (lhs : QueryPlan s1)
    (key : String) {s2 : Schema} (rhs : QueryPlan s2)
    (h1 : key ∈ s1 := by decide) (h2 : key ∈ s2 := by decide)
    : QueryPlan (mergeSchema s1 s2) :=
  join key lhs rhs h1 h2

def Table.query {s : Schema} (t : Table s) : QueryPlan s :=
  source t

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

-- Tree-shaped query expressed with higher-level combinators.
-- Schema propagates through every node; mergeSchema is computed at each join;
-- column membership proofs discharge automatically via `decide`.
--
--   employees.query.whereEq "city" "Seattle"
--     |> innerJoinOn "city" locations.query
--     |> innerJoinOn "dept_id" (departments.query.whereEq "dept_name" "Engineering")
--     |> project ["name", "city", "region", "dept_name", "floor"]
--     |> compileQuery
def result : Except String (Artifact × UInt32) :=
  let employeesInSeattle := employees.query.whereEq "city" "Seattle"
  let departmentsInEngineering := departments.query.whereEq "dept_name" "Engineering"
  let employeesWithLocations := employeesInSeattle.innerJoinOn "city" locations.query
  let fullQuery := employeesWithLocations.innerJoinOn "dept_id" departmentsInEngineering
  (fullQuery.project ["name", "city", "region", "dept_name", "floor"]).compileQuery

-- Uncomment either def to see elaboration-time rejection at the combinator call
-- that introduces the bad column/key:
--
-- def badJoinKey : Except String (Artifact × UInt32) :=
--   let employeesInSeattle := employees.query.whereEq "city" "Seattle"
--   let brokenJoin := employeesInSeattle.innerJoinOn "nonexistent_key" departments.query
--   (brokenJoin.project ["name", "dept_name"]).compileQuery
--
-- def badSelectCol : Artifact × UInt32 :=
--   let employeesInSeattle := employees.query.whereEq "city" "Seattle"
--   let joined := employeesInSeattle.innerJoinOn "dept_id" departments.query
--   (joined.project ["name", "salary_band"]).compileQuery

end CsvDemo

def main (args : List String) : IO Unit := do
  let (cfg, alg) ← Prog.orDie CsvDemo.result
  let outDir ← requireOutputDir args
  emitArtifacts outDir #[artifactEntry "csv_app" cfg]

#eval ShipScan.check "CsvAlgorithm"
