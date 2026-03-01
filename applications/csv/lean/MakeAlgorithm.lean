import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace CsvDemo

-- ---------------------------------------------------------------------------
-- CSV Demo: Read CSV files → LMDB → query → write output CSVs
-- ---------------------------------------------------------------------------

-- Payload layout constants
def empDbPath_off   : Nat := 0x0100
def deptDbPath_off  : Nat := 0x0200
def empCsvPath_off  : Nat := 0x0300
def deptCsvPath_off : Nat := 0x0400
def scanFname_off   : Nat := 0x0500
def filterFname_off : Nat := 0x0540
def joinFname_off   : Nat := 0x0580
def seattleStr_off  : Nat := 0x05C0  -- ",Seattle," for substring match
def seattleStrLen   : Nat := 9       -- length of ",Seattle,"
def seattleRegion   : Nat := 16     -- padded to 16 bytes
def flag_off        : Nat := seattleStr_off + seattleRegion  -- 0x05D0
def clifIr_off      : Nat := flag_off + 64  -- 0x0610
def clifIrRegionSize : Nat := 10240  -- 10KB for CLIF IR
def empBuf_off      : Nat := clifIr_off + clifIrRegionSize
def empBufSize      : Nat := 4096
def deptBuf_off     : Nat := empBuf_off + empBufSize  -- 0x3F00
def deptBufSize     : Nat := 1024
def keyScratch_off  : Nat := deptBuf_off + deptBufSize  -- 0x4300
def scanResult_off  : Nat := keyScratch_off + 16  -- 0x4310
def scanResultSize  : Nat := 8192
def scanResult2_off : Nat := scanResult_off + scanResultSize  -- 0x6310
def scanResult2Size : Nat := 8192
def totalPayload    : Nat := scanResult2_off + scanResult2Size  -- 0x8310

-- ---------------------------------------------------------------------------
-- CLIF IR
--
-- FFI functions:
--   fn0 = cl_file_read        sig5: (i64, i64, i64, i64, i64) -> i64
--   fn1 = cl_lmdb_init        sig0: (i64)
--   fn2 = cl_lmdb_open        sig1: (i64, i64, i32) -> i32
--   fn3 = cl_lmdb_begin_write_txn  sig2: (i64, i32) -> i32
--   fn4 = cl_lmdb_put         sig3: (i64, i32, i64, i32, i64, i32) -> i32
--   fn5 = cl_lmdb_commit_write_txn sig2: (i64, i32) -> i32
--   fn6 = cl_lmdb_cursor_scan sig6: (i64, i32, i64, i32, i32, i64) -> i32
--   fn7 = cl_file_write       sig5: (i64, i64, i64, i64, i64) -> i64
--   fn8 = cl_lmdb_cleanup     sig0: (i64)
--
-- Block structure:
--   block0:  Setup — file reads, lmdb init/open, begin emp txn
--   block1:  Emp ingest: find-newline inner loop
--   block2:  Emp ingest: found row end — put row, advance
--   block3:  Emp ingest done — commit, begin dept txn
--   block4:  Dept ingest: find-newline inner loop
--   block5:  Dept ingest: found row end — put row, advance
--   block6:  Dept ingest done — commit, cursor_scan employees for scan.csv
--   block7:  Scan output: loop header
--   block8:  Scan output: write row, advance
--   block9:  Scan done — cursor_scan employees for filter.csv
--   block10: Filter output: loop header
--   block11: Filter: check if key==0 (header row → always include)
--   block12: Filter: substring scan loop header
--   block13: Filter: check substring match at position
--   block14: Filter: matched — write row
--   block15: Filter: no match at this pos — advance scan
--   block16: Filter: no match at all — skip row
--   block17: Filter done — cursor_scan departments for join.csv
--   block18: Join output: loop header
--   block19: Join output: write row, advance
--   block20: Join done — cleanup, return
-- ---------------------------------------------------------------------------

open AlgorithmLib.IR

-- Shared values threaded across sub-functions
structure K where
  ptr : Val
  zero : Val
  one : Val
  keyLen4 : Val
  newline : Val
  keyScrOff : Val
  empBufOff : Val
  deptBufOff : Val
  empSize : Val
  deptSize : Val
  empHandle : Val
  deptHandle : Val
  fnLmdbPut : FnRef
  fnCommitTxn : FnRef
  fnBeginTxn : FnRef
  fnCursorScan : FnRef
  fnFileWrite : FnRef
  fnCleanup : FnRef

-- Emit employee/department ingest loops
def emitIngest (k : K)
    (empLoop empScan empAdvance empPut empDone : DeclaredBlock)
    (deptLoop deptScan deptAdv deptPut deptDone : DeclaredBlock)
    : IRBuilder Unit := do
  -- Employee ingest — outer loop header
  startBlock empLoop
  let pos := empLoop.param 0
  let key := empLoop.param 1
  let posEnd ← icmp .uge pos k.empSize
  brif posEnd empDone.ref [] empScan.ref [pos]

  -- Employee ingest — inner newline scan loop
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

  -- Employee ingest — found row end, put into LMDB
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

  -- Employee ingest done — commit, begin dept txn
  startBlock empDone
  let _ ← call k.fnCommitTxn [k.ptr, k.empHandle]
  let _ ← call k.fnBeginTxn [k.ptr, k.deptHandle]
  jump deptLoop.ref [k.zero, k.zero]

  -- Department ingest — outer loop header
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

  -- Department ingest — put row
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

-- Emit scan + filter + join output phases
set_option maxRecDepth 4096 in
def emitOutputPhases (k : K)
    (deptDone : DeclaredBlock)
    (scanHdr scanBody scanDone : DeclaredBlock)
    (filterHdr filterParse : DeclaredBlock)
    (substrScan substrCmp substrByte : DeclaredBlock)
    (filterMatch substrAdv filterSkip filterDone : DeclaredBlock)
    (joinHdr joinBody joinDone : DeclaredBlock)
    : IRBuilder Unit := do
  -- Dept ingest done — commit, cursor_scan employees for scan.csv
  startBlock deptDone
  let _ ← call k.fnCommitTxn [k.ptr, k.deptHandle]
  let scanResOff ← iconst64 scanResult_off
  let keyLen0 ← iconst32 0
  let maxEntries ← iconst32 100
  let scanCount ← call k.fnCursorScan [k.ptr, k.empHandle, k.zero, keyLen0, maxEntries, scanResOff]
  let four ← iconst64 4
  jump scanHdr.ref [k.zero, four, k.zero]

  -- Scan output — loop header
  startBlock scanHdr
  let si := scanHdr.param 0
  let sByteOff := scanHdr.param 1
  let sFileOff := scanHdr.param 2
  let scanCount64 ← sextend64 scanCount
  let sDone ← icmp .uge si scanCount64
  brif sDone scanDone.ref [] scanBody.ref []

  -- Scan output — write row, advance
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

  -- Scan done — cursor_scan employees again for filter
  startBlock scanDone
  let scanRes2Off ← iconst64 scanResult2_off
  let filterCount ← call k.fnCursorScan [k.ptr, k.empHandle, k.zero, keyLen0, maxEntries, scanRes2Off]
  jump filterHdr.ref [k.zero, four, k.zero]

  -- Filter output — loop header
  startBlock filterHdr
  let fi := filterHdr.param 0
  let fByteOff := filterHdr.param 1
  let fFileOff := filterHdr.param 2
  let filterCount64 ← sextend64 filterCount
  let fDone ← icmp .uge fi filterCount64
  brif fDone filterDone.ref [] filterParse.ref []

  -- Filter — parse entry, decide include/skip
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

  -- Filter — substring scan
  startBlock substrScan
  let scanPos := substrScan.param 0
  let seaLen ← iconst64 seattleStrLen
  let scanEnd ← iadd scanPos seaLen
  let noRoom ← icmp .ugt scanEnd fVlen64
  brif noRoom filterSkip.ref [] substrCmp.ref [k.zero]

  -- Filter — compare substring at position
  startBlock substrCmp
  let matchIdx := substrCmp.param 0
  let allMatch ← icmp .uge matchIdx seaLen
  brif allMatch filterMatch.ref [] substrByte.ref []

  startBlock substrByte
  let bytePos ← iadd scanPos matchIdx
  let valByteOff ← iadd fValOff bytePos
  let valByteAddr ← iadd k.ptr valByteOff
  let valByte ← load_i8 valByteAddr
  let seaOff ← iconst64 seattleStr_off
  let seaByteOff ← iadd seaOff matchIdx
  let seaByteAddr ← iadd k.ptr seaByteOff
  let seaByte ← load_i8 seaByteAddr
  let bytesEq ← icmp .eq valByte seaByte
  let nextMatch ← iadd matchIdx k.one
  brif bytesEq substrCmp.ref [nextMatch] substrAdv.ref []

  -- Filter — match
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

  -- Filter done — cursor_scan departments for join.csv
  startBlock filterDone
  let joinCount ← call k.fnCursorScan [k.ptr, k.deptHandle, k.zero, keyLen0, maxEntries, scanResOff]
  jump joinHdr.ref [k.zero, four, k.zero]

  -- Join output — loop header
  startBlock joinHdr
  let ji := joinHdr.param 0
  let jByteOff := joinHdr.param 1
  let jFileOff := joinHdr.param 2
  let joinCount64 ← sextend64 joinCount
  let jDone ← icmp .uge ji joinCount64
  brif jDone joinDone.ref [] joinBody.ref []

  -- Join output — write row, advance
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

  -- Done — cleanup and return
  startBlock joinDone
  callVoid k.fnCleanup [k.ptr]
  ret

set_option maxRecDepth 4096 in
def clifIrSource : String := buildProgram do
  -- Declare FFI functions
  let fnFileRead ← declareFFI "cl_file_read" [.i64, .i64, .i64, .i64, .i64] (some .i64)
  let fnLmdbInit ← declareFFI "cl_lmdb_init" [.i64] none
  let fnLmdbOpen ← declareFFI "cl_lmdb_open" [.i64, .i64, .i32] (some .i32)
  let fnBeginTxn ← declareFFI "cl_lmdb_begin_write_txn" [.i64, .i32] (some .i32)
  let fnLmdbPut ← declareFFI "cl_lmdb_put" [.i64, .i32, .i64, .i32, .i64, .i32] (some .i32)
  let fnCommitTxn ← declareFFI "cl_lmdb_commit_write_txn" [.i64, .i32] (some .i32)
  let fnCursorScan ← declareFFI "cl_lmdb_cursor_scan" [.i64, .i32, .i64, .i32, .i32, .i64] (some .i32)
  let fnFileWrite ← declareFFI "cl_file_write" [.i64, .i64, .i64, .i64, .i64] (some .i64)
  let fnCleanup ← declareFFI "cl_lmdb_cleanup" [.i64] none

  -- Entry block first so it gets block0
  let ptr ← entryBlock

  -- Forward-declare all blocks
  let empLoop    ← declareBlock [.i64, .i64]
  let empScan    ← declareBlock [.i64]
  let empAdvance ← declareBlock []
  let empPut     ← declareBlock []
  let empDone    ← declareBlock []
  let deptLoop   ← declareBlock [.i64, .i64]
  let deptScan   ← declareBlock [.i64]
  let deptAdv    ← declareBlock []
  let deptPut    ← declareBlock []
  let deptDone   ← declareBlock []
  let scanHdr    ← declareBlock [.i64, .i64, .i64]
  let scanBody   ← declareBlock []
  let scanDone   ← declareBlock []
  let filterHdr  ← declareBlock [.i64, .i64, .i64]
  let filterParse ← declareBlock []
  let substrScan ← declareBlock [.i64]
  let substrCmp  ← declareBlock [.i64]
  let substrByte ← declareBlock []
  let filterMatch ← declareBlock []
  let substrAdv  ← declareBlock []
  let filterSkip ← declareBlock []
  let filterDone ← declareBlock []
  let joinHdr    ← declareBlock [.i64, .i64, .i64]
  let joinBody   ← declareBlock []
  let joinDone   ← declareBlock []

  -- Setup instructions (emitted into entry block0)
  let empCsvOff ← iconst64 empCsvPath_off
  let empBufOff ← iconst64 empBuf_off
  let zero ← iconst64 0
  let empSize ← call fnFileRead [ptr, empCsvOff, empBufOff, zero, zero]
  let deptCsvOff ← iconst64 deptCsvPath_off
  let deptBufOff ← iconst64 deptBuf_off
  let deptSize ← call fnFileRead [ptr, deptCsvOff, deptBufOff, zero, zero]
  callVoid fnLmdbInit [ptr]
  let empDbOff ← iconst64 empDbPath_off
  let maxDbs ← iconst32 10
  let empHandle ← call fnLmdbOpen [ptr, empDbOff, maxDbs]
  let deptDbOff ← iconst64 deptDbPath_off
  let deptHandle ← call fnLmdbOpen [ptr, deptDbOff, maxDbs]
  let _ ← call fnBeginTxn [ptr, empHandle]
  let one ← iconst64 1
  let keyLen4 ← iconst32 4
  let newline ← iconst64 10
  let keyScrOff ← iconst64 keyScratch_off
  jump empLoop.ref [zero, zero]

  let k : K := {
    ptr, zero, one, keyLen4, newline, keyScrOff,
    empBufOff, deptBufOff, empSize, deptSize,
    empHandle, deptHandle,
    fnLmdbPut, fnCommitTxn, fnBeginTxn, fnCursorScan, fnFileWrite, fnCleanup
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
-- Payload construction
-- ---------------------------------------------------------------------------

def payloads : List UInt8 :=
  let reserved := zeros empDbPath_off
  -- DB paths
  let empDbPathBytes := padTo (stringToBytes "/tmp/csv-demo/employees") 256
  let deptDbPathBytes := padTo (stringToBytes "/tmp/csv-demo/departments") 256
  -- CSV file paths
  let empCsvPathBytes := padTo (stringToBytes "applications/csv/data/employees.csv") 256
  let deptCsvPathBytes := padTo (stringToBytes "applications/csv/data/departments.csv") 256
  -- Output filenames
  let scanFnameBytes := padTo (stringToBytes "scan.csv") 64
  let filterFnameBytes := padTo (stringToBytes "filter.csv") 64
  let joinFnameBytes := padTo (stringToBytes "join.csv") 64
  -- Seattle match string (no null terminator needed, matched by length)
  let seattleBytes := padTo ([0x2C, 0x53, 0x65, 0x61, 0x74, 0x74, 0x6C, 0x65, 0x2C] : List UInt8) 16
  -- Flag + padding to CLIF IR
  let flagBytes := uint64ToBytes 0
  let flagPad := zeros (clifIr_off - flag_off - 8)
  let clifPad := zeros clifIrRegionSize
  -- Buffers (all zeros)
  let empBufBytes := zeros empBufSize
  let deptBufBytes := zeros deptBufSize
  let keyScratchBytes := zeros 16
  let scanResultBytes := zeros scanResultSize
  let scanResult2Bytes := zeros scanResult2Size
  reserved ++
  empDbPathBytes ++ deptDbPathBytes ++
  empCsvPathBytes ++ deptCsvPathBytes ++
  scanFnameBytes ++ filterFnameBytes ++ joinFnameBytes ++
  seattleBytes ++
  flagBytes ++ flagPad ++
  clifPad ++
  empBufBytes ++ deptBufBytes ++
  keyScratchBytes ++ scanResultBytes ++ scanResult2Bytes

-- ---------------------------------------------------------------------------
-- Algorithm definition
-- ---------------------------------------------------------------------------

def csvConfig : BaseConfig := {
  cranelift_ir := clifIrSource,
  memory_size := payloads.length,
  context_offset := 0
}

def csvAlgorithm : Algorithm :=
  let clifCallAction : Action :=
    { kind := .ClifCall, dst := u32 0, src := u32 1, offset := u32 0, size := u32 0 }
  {
    actions := [clifCallAction],
    payloads := payloads,
    cranelift_units := 0,
    timeout_ms := some 30000
  }

end CsvDemo

def main : IO Unit := do
  let json := toJsonPair CsvDemo.csvConfig CsvDemo.csvAlgorithm
  IO.println (Json.compress json)
