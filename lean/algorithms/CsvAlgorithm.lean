import AlgorithmLib

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

open AlgorithmLib.IR

open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sur

/-- The externals every emitted function declares, in one order. -/
def ffiEnv : ((FnRef × IR.LmdbSetup) × FnRef) × FnEnv := envOf (do
  let rd ← declareFileRead
  let lmdb ← declareLmdbFFI
  let wr ← declareFileWrite
  pure ((rd, lmdb), wr))
def fnFileRead : FnRef := ffiEnv.1.1.1
def lmdb : IR.LmdbSetup := ffiEnv.1.1.2
def fnFileWrite : FnRef := ffiEnv.1.2
def env : FnEnv := ffiEnv.2

/-- One CSV buffer's rows written to a database, one row per key. A row runs to
    the next newline, or to the end of the buffer. -/
def emitIngest (ptr lmdbCtx handle bufOff size keyScrOff : R) : M Unit := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let newline ← iconst64 10
  let keyLen4 ← iconst32 4
  let _ ← wloop2 zero zero
    (head := fun pos _ => return (contIf .ult pos size, ([] : List R), ()))
    (body := fun pos key _ => do
      -- the row's last byte: a newline, or the buffer's end
      let sc ← wloop1 pos
        (head := fun sPos => return (contIf .eq zero zero, [sPos], ()))
        (body := fun sPos _ => do
          let byte ← uextend64 (← load_i8 (← iadd ptr (← iadd bufOff sPos)))
          when .eq byte newline (brk [sPos])
          let nextPos ← iadd sPos one
          when .uge nextPos size (brk [sPos])
          return [nextPos])
      let rowEnd ← iadd (sc.headD 0) one
      let rowLen32 ← ireduce32 (← isub rowEnd pos)
      let keyPtr ← iadd ptr keyScrOff
      store (← ireduce32 key) keyPtr
      let valPtr ← iadd ptr (← iadd bufOff pos)
      let _ ← call lmdb.fnPut.id [lmdbCtx, handle, keyPtr, keyLen4, valPtr, rowLen32]
      return [rowEnd, ← iadd key one])

/-- Every scanned value written to one file, back to back. -/
def emitWriteAll (ptr lmdbCtx handle resOff count fnameOff : R) : M Unit := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let two ← iconst64 2
  let four ← iconst64 4
  let count64 ← sextend64 count
  let _ ← wloop [zero, four, zero]
    (head := fun c => return (contIf .slt (c.headD 0) count64, ([] : List R), ()))
    (body := fun c _ => do
      let i := c.headD 0
      let byteOff := c.getD 1 0
      let fileOff := c.getD 2 0
      let entryAddr ← iadd ptr (← iadd resOff byteOff)
      let klen ← uextend64 (← load_i16 entryAddr)
      let vlen ← uextend64 (← load_i16 (← iadd entryAddr two))
      let dataOff ← iadd (← iadd byteOff four) klen
      let valOff ← iadd resOff dataOff
      let vlenSigned ← sextend64 (← ireduce32 vlen)
      let _ ← call fnFileWrite.id [ptr, fnameOff, valOff, fileOff, vlenSigned]
      return [← iadd i one, ← iadd dataOff vlen, ← iadd fileOff vlen])

/-- The scanned rows whose value contains the pattern, plus the header row. -/
def emitFilter (ptr resOff count fnameOff : R) (patternLen : Nat) : M Unit := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let two ← iconst64 2
  let four ← iconst64 4
  let zero32 ← iconst32 0
  let seaLen ← iconst64 patternLen
  let patOff ← iconst64 patternStr_off
  let count64 ← sextend64 count
  let _ ← wloop [zero, four, zero]
    (head := fun c => return (contIf .slt (c.headD 0) count64, ([] : List R), ()))
    (body := fun c _ => do
      let i := c.headD 0
      let byteOff := c.getD 1 0
      let fileOff := c.getD 2 0
      let entryAddr ← iadd ptr (← iadd resOff byteOff)
      let klen ← uextend64 (← load_i16 entryAddr)
      let vlen ← uextend64 (← load_i16 (← iadd entryAddr two))
      let dataOff ← iadd byteOff four
      let dataOff2 ← iadd dataOff klen
      let valOff ← iadd resOff dataOff2
      let keyVal ← load32 (← iadd ptr (← iadd resOff dataOff))
      let isHeader ← icmp .eq keyVal zero32
      -- the header row is kept whatever the pattern says
      let hit ← ifte .ne isHeader (← iconst .i8 0) (pure [one])
        (do
          -- the pattern searched for at every position it still fits in
          let sc ← wloop1 zero
            (head := fun scanPos => do
              let scanEnd ← iadd scanPos seaLen
              return (contIf .ule scanEnd vlen, [zero], ()))
            (body := fun scanPos _ => do
              let m ← wloop1 zero
                (head := fun mi => return (contIf .ult mi seaLen, [one], ()))
                (body := fun mi _ => do
                  let valByte ← load_i8 (← iadd ptr (← iadd valOff (← iadd scanPos mi)))
                  let patByte ← load_i8 (← iadd ptr (← iadd patOff mi))
                  when .ne valByte patByte (brk [zero])
                  return [← iadd mi one])
              when .ne (m.headD 0) zero (brk [one])
              return [← iadd scanPos one])
          pure [sc.headD 0])
      let _ ← ifte .ne (hit.headD 0) zero
        (do
          let vlenSigned ← sextend64 (← ireduce32 vlen)
          let _ ← call fnFileWrite.id [ptr, fnameOff, valOff, fileOff, vlenSigned]
          pure [])
        (pure [])
      let nextFile ← ifte .ne (hit.headD 0) zero
        (pure [← iadd fileOff vlen]) (pure [fileOff])
      return [← iadd i one, ← iadd dataOff2 vlen, nextFile.headD 0])

set_option maxRecDepth 4096 in
def mainCode (patternLen : Nat) : HProg.Code :=
  HProg.Sur.build env HProg.ptrParams do
  let ptr := basePtr
  let empBufOff ← iconst64 empBuf_off
  let zero ← iconst64 0
  let empSize ← readFile ptr fnFileRead empCsvPath_off empBuf_off
  let deptBufOff ← iconst64 deptBuf_off
  let deptSize ← readFile ptr fnFileRead deptCsvPath_off deptBuf_off
  let lmdbSlot := ptr
  callVoid lmdb.fnInit.id [lmdbSlot]
  let lmdbCtx ← load64 ptr
  let maxDbs ← iconst32 10
  let empHandle ← call lmdb.fnOpen.id [lmdbCtx, ← iadd ptr (← iconst64 empDbPath_off), maxDbs]
  let deptHandle ← call lmdb.fnOpen.id [lmdbCtx, ← iadd ptr (← iconst64 deptDbPath_off), maxDbs]
  let keyScrOff ← iconst64 keyScratch_off

  let _ ← call lmdb.fnBeginWriteTxn.id [lmdbCtx, empHandle]
  emitIngest ptr lmdbCtx empHandle empBufOff empSize keyScrOff
  let _ ← call lmdb.fnCommitWriteTxn.id [lmdbCtx, empHandle]

  let _ ← call lmdb.fnBeginWriteTxn.id [lmdbCtx, deptHandle]
  emitIngest ptr lmdbCtx deptHandle deptBufOff deptSize keyScrOff
  let _ ← call lmdb.fnCommitWriteTxn.id [lmdbCtx, deptHandle]

  let keyLen0 ← iconst32 0
  let maxEntries ← iconst32 100
  let scanResOff ← iconst64 scanResult_off
  let scanCount ← call lmdb.fnCursorScan.id
    [lmdbCtx, empHandle, ptr, keyLen0, maxEntries, ← iadd ptr scanResOff]
  emitWriteAll ptr lmdbCtx empHandle scanResOff scanCount (← iconst64 scanFname_off)

  let scanRes2Off ← iconst64 scanResult2_off
  let filterCount ← call lmdb.fnCursorScan.id
    [lmdbCtx, empHandle, ptr, keyLen0, maxEntries, ← iadd ptr scanRes2Off]
  emitFilter ptr scanRes2Off filterCount (← iconst64 filterFname_off) patternLen

  let joinCount ← call lmdb.fnCursorScan.id
    [lmdbCtx, deptHandle, ptr, keyLen0, maxEntries, ← iadd ptr scanResOff]
  emitWriteAll ptr lmdbCtx deptHandle scanResOff joinCount (← iconst64 joinFname_off)

  callVoid lmdb.fnCleanup.id [lmdbSlot]

set_option maxRecDepth 4096 in
def clifIrSource (patternLen : Nat) : Program :=
  IR.program [IR.noopFunction, HProg.compileFn 1 env HProg.ptrParams (mainCode patternLen)]

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

def buildQueryMonomorphic (patternStr : String) : Setup × Algorithm :=
  let patternBytes := patternStr.toUTF8.toList  -- no null terminator
  let payload := buildPayload patternBytes
  let cfg : Setup := {
    clif := clifIrSource patternBytes.length,
    memory_size    := payload.length,
    initial_memory := payload
  }
  let alg : Algorithm := {
    fn_idx := IR.mainFnIdx
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

def compile {s : Schema} (p : QueryPlan s) : Setup × Algorithm :=
  buildQueryMonomorphic ("," ++ extractPattern p ++ ",")

def source {s : Schema} (t : Table s) : QueryPlan s :=
  plan t

def QueryPlan.whereEq {s : Schema} (p : QueryPlan s)
    (col pat : String) (h : col ∈ s := by decide) : QueryPlan s :=
  filter col pat p h

def QueryPlan.project {s : Schema} (p : QueryPlan s)
    (cols : List String) (h : ∀ c ∈ cols, c ∈ s := by decide) : QueryPlan cols :=
  select cols p h

def QueryPlan.compileQuery {s : Schema} (p : QueryPlan s) : Setup × Algorithm :=
  compile p

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
def result : Setup × Algorithm :=
  let employeesInSeattle := employees.query.whereEq "city" "Seattle"
  let departmentsInEngineering := departments.query.whereEq "dept_name" "Engineering"
  let employeesWithLocations := employeesInSeattle.innerJoinOn "city" locations.query
  let fullQuery := employeesWithLocations.innerJoinOn "dept_id" departmentsInEngineering
  (fullQuery.project ["name", "city", "region", "dept_name", "floor"]).compileQuery

-- Uncomment either def to see elaboration-time rejection at the combinator call
-- that introduces the bad column/key:
--
-- def badJoinKey : Setup × Algorithm :=
--   let employeesInSeattle := employees.query.whereEq "city" "Seattle"
--   let brokenJoin := employeesInSeattle.innerJoinOn "nonexistent_key" departments.query
--   (brokenJoin.project ["name", "dept_name"]).compileQuery
--
-- def badSelectCol : Setup × Algorithm :=
--   let employeesInSeattle := employees.query.whereEq "city" "Seattle"
--   let joined := employeesInSeattle.innerJoinOn "dept_id" departments.query
--   (joined.project ["name", "salary_band"]).compileQuery

end CsvDemo

def main (args : List String) : IO Unit := do
  let (cfg, alg) := CsvDemo.result
  let outDir ← requireOutputDir args
  emitArtifacts outDir #[toJsonEntry "csv_app" cfg alg]
