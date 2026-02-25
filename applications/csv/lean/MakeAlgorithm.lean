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

def clifIrSource : String :=
  let empDb     := toString empDbPath_off
  let deptDb    := toString deptDbPath_off
  let empCsv    := toString empCsvPath_off
  let deptCsv   := toString deptCsvPath_off
  let empBuf    := toString empBuf_off
  let deptBuf   := toString deptBuf_off
  let keySc     := toString keyScratch_off
  let scanRes   := toString scanResult_off
  let scanRes2  := toString scanResult2_off
  let scanFn    := toString scanFname_off
  let filterFn  := toString filterFname_off
  let joinFn    := toString joinFname_off
  let seaStr    := toString seattleStr_off
  let seaLen    := toString seattleStrLen
  -- noop function (CLIF fn index 0, unused placeholder)
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    return\n" ++
  "}\n\n" ++
  -- orchestrator (CLIF fn index 1, called synchronously via ClifCall)
  "function u0:1(i64) system_v {\n" ++
  "    sig0 = (i64) system_v\n" ++                                 -- init/cleanup
  "    sig1 = (i64, i64, i32) -> i32 system_v\n" ++                -- lmdb_open
  "    sig2 = (i64, i32) -> i32 system_v\n" ++                     -- begin/commit txn
  "    sig3 = (i64, i32, i64, i32, i64, i32) -> i32 system_v\n" ++ -- lmdb_put
  "    sig4 = (i64, i32, i64, i32, i64) -> i32 system_v\n" ++      -- lmdb_get
  "    sig5 = (i64, i64, i64, i64, i64) -> i64 system_v\n" ++      -- file_read/write
  "    sig6 = (i64, i32, i64, i32, i32, i64) -> i32 system_v\n" ++ -- cursor_scan
  "    fn0 = %cl_file_read sig5\n" ++
  "    fn1 = %cl_lmdb_init sig0\n" ++
  "    fn2 = %cl_lmdb_open sig1\n" ++
  "    fn3 = %cl_lmdb_begin_write_txn sig2\n" ++
  "    fn4 = %cl_lmdb_put sig3\n" ++
  "    fn5 = %cl_lmdb_commit_write_txn sig2\n" ++
  "    fn6 = %cl_lmdb_cursor_scan sig6\n" ++
  "    fn7 = %cl_file_write sig5\n" ++
  "    fn8 = %cl_lmdb_cleanup sig0\n" ++
  "\n" ++

  -- =====================================================================
  -- block0: Setup
  -- =====================================================================
  "block0(v0: i64):\n" ++
  -- Read employees.csv into empBuf
  s!"    v1 = iconst.i64 {empCsv}\n" ++
  s!"    v2 = iconst.i64 {empBuf}\n" ++
  "    v3 = iconst.i64 0\n" ++
  "    v4 = call fn0(v0, v1, v2, v3, v3)\n" ++  -- v4 = emp file size
  -- Read departments.csv into deptBuf
  s!"    v5 = iconst.i64 {deptCsv}\n" ++
  s!"    v6 = iconst.i64 {deptBuf}\n" ++
  "    v7 = call fn0(v0, v5, v6, v3, v3)\n" ++  -- v7 = dept file size
  -- Init LMDB
  "    call fn1(v0)\n" ++
  -- Open employee DB
  s!"    v8 = iconst.i64 {empDb}\n" ++
  "    v9 = iconst.i32 10\n" ++
  "    v10 = call fn2(v0, v8, v9)\n" ++  -- emp handle
  -- Open department DB
  s!"    v11 = iconst.i64 {deptDb}\n" ++
  "    v12 = call fn2(v0, v11, v9)\n" ++  -- dept handle
  -- Begin employee write txn
  "    v13 = call fn3(v0, v10)\n" ++
  -- Constants we'll reuse
  "    v14 = iconst.i64 1\n" ++
  "    v15 = iconst.i32 4\n" ++       -- key length = 4 bytes
  "    v16 = iconst.i64 10\n" ++      -- newline '\n' = 10
  s!"    v17 = iconst.i64 {keySc}\n" ++  -- key scratch offset
  -- Jump to emp ingest loop: (pos=0, key=0)
  "    jump block1(v3, v3)\n" ++
  "\n" ++

  -- =====================================================================
  -- block1: Employee ingest — find newline
  -- args: v20=pos (byte offset within empBuf), v21=key (row index)
  -- =====================================================================
  "block1(v20: i64, v21: i64):\n" ++
  -- Check if pos >= file_size
  "    v22 = icmp uge v20, v4\n" ++
  "    brif v22, block3, block30(v20)\n" ++
  "\n" ++

  -- block30: inner newline scan loop
  "block30(v200: i64):\n" ++
  -- Load byte at empBuf + pos
  "    v201 = iadd v2, v200\n" ++     -- empBuf offset + pos
  "    v202 = iadd v0, v201\n" ++     -- absolute addr
  "    v203 = load.i8 v202\n" ++
  "    v204 = uextend.i64 v203\n" ++
  -- Check if byte == '\n'
  "    v205 = icmp eq v204, v16\n" ++
  "    brif v205, block2, block31\n" ++
  "\n" ++

  -- block31: not newline, advance pos and continue scanning
  "block31:\n" ++
  "    v206 = iadd v200, v14\n" ++    -- pos + 1
  -- Check if pos >= file_size
  "    v207 = icmp uge v206, v4\n" ++
  "    brif v207, block2, block30(v206)\n" ++  -- if EOF, treat as row end too
  "\n" ++

  -- =====================================================================
  -- block2: Employee ingest — found row end, put row into LMDB
  -- v20=row_start, v21=key, v200=pos of '\n' (or EOF pos)
  -- =====================================================================
  "block2:\n" ++
  -- Row end position: include the \n (pos + 1), or just pos if EOF
  "    v23 = iadd v200, v14\n" ++     -- end = pos + 1 (include \n)
  -- Row length = end - row_start
  "    v24 = isub v23, v20\n" ++
  "    v25 = ireduce.i32 v24\n" ++    -- row_len as i32
  -- Store key as u32 LE at key scratch
  "    v26 = ireduce.i32 v21\n" ++    -- key as i32
  "    v27 = iadd v0, v17\n" ++       -- absolute addr of key scratch
  "    store v26, v27\n" ++
  -- Value offset = empBuf_off + row_start
  "    v28 = iadd v2, v20\n" ++       -- empBuf offset + row_start
  -- cl_lmdb_put(ptr, handle, key_off, key_len, val_off, val_len)
  "    v29 = call fn4(v0, v10, v17, v15, v28, v25)\n" ++
  -- Advance: key++, pos = end
  "    v30 = iadd v21, v14\n" ++      -- key + 1
  "    jump block1(v23, v30)\n" ++
  "\n" ++

  -- =====================================================================
  -- block3: Employee ingest done — commit, begin dept txn
  -- =====================================================================
  "block3:\n" ++
  "    v31 = call fn5(v0, v10)\n" ++   -- commit emp txn
  "    v32 = call fn3(v0, v12)\n" ++   -- begin dept write txn
  -- Jump to dept ingest loop: (pos=0, key=0)
  "    jump block4(v3, v3)\n" ++
  "\n" ++

  -- =====================================================================
  -- block4: Department ingest — find newline
  -- args: v40=pos, v41=key
  -- =====================================================================
  "block4(v40: i64, v41: i64):\n" ++
  "    v42 = icmp uge v40, v7\n" ++   -- pos >= dept file size?
  "    brif v42, block6, block40(v40)\n" ++
  "\n" ++

  -- block40: inner newline scan for dept
  "block40(v400: i64):\n" ++
  "    v401 = iadd v6, v400\n" ++     -- deptBuf offset + pos
  "    v402 = iadd v0, v401\n" ++
  "    v403 = load.i8 v402\n" ++
  "    v404 = uextend.i64 v403\n" ++
  "    v405 = icmp eq v404, v16\n" ++
  "    brif v405, block5, block41\n" ++
  "\n" ++

  "block41:\n" ++
  "    v406 = iadd v400, v14\n" ++
  "    v407 = icmp uge v406, v7\n" ++
  "    brif v407, block5, block40(v406)\n" ++
  "\n" ++

  -- =====================================================================
  -- block5: Department ingest — put row
  -- v40=row_start, v41=key, v400=pos of '\n' or EOF
  -- =====================================================================
  "block5:\n" ++
  "    v43 = iadd v400, v14\n" ++
  "    v44 = isub v43, v40\n" ++
  "    v45 = ireduce.i32 v44\n" ++
  "    v46 = ireduce.i32 v41\n" ++
  "    v47 = iadd v0, v17\n" ++
  "    store v46, v47\n" ++
  "    v48 = iadd v6, v40\n" ++
  "    v49 = call fn4(v0, v12, v17, v15, v48, v45)\n" ++
  "    v50 = iadd v41, v14\n" ++
  "    jump block4(v43, v50)\n" ++
  "\n" ++

  -- =====================================================================
  -- block6: Dept ingest done — commit, cursor_scan employees for scan.csv
  -- =====================================================================
  "block6:\n" ++
  "    v51 = call fn5(v0, v12)\n" ++   -- commit dept txn
  -- cursor_scan employees (all rows)
  s!"    v52 = iconst.i64 {scanRes}\n" ++
  "    v53 = iconst.i32 0\n" ++        -- key_len=0 → scan from beginning
  "    v54 = iconst.i32 100\n" ++      -- max entries
  "    v55 = call fn6(v0, v10, v3, v53, v54, v52)\n" ++  -- v55 = count
  -- Jump to scan output loop: (i=0, offset=4, file_off=0)
  "    v56 = iconst.i64 4\n" ++
  "    jump block7(v3, v56, v3)\n" ++
  "\n" ++

  -- =====================================================================
  -- block7: Scan output — loop header
  -- args: v60=i (entry index), v61=byte offset in scan result, v62=file_off
  -- =====================================================================
  "block7(v60: i64, v61: i64, v62: i64):\n" ++
  "    v63 = sextend.i64 v55\n" ++    -- count as i64
  "    v64 = icmp uge v60, v63\n" ++
  "    brif v64, block9, block8\n" ++
  "\n" ++

  -- =====================================================================
  -- block8: Scan output — write row, advance
  -- Read cursor_scan entry: [u16 klen][u16 vlen][key][val]
  -- =====================================================================
  "block8:\n" ++
  -- Read klen (u16 at scanResult + offset)
  "    v65 = iadd v52, v61\n" ++      -- scanResult offset + byte offset
  "    v66 = iadd v0, v65\n" ++       -- absolute addr
  "    v67 = load.i16 v66\n" ++       -- klen (u16)
  "    v68 = uextend.i64 v67\n" ++
  -- Read vlen (u16 at offset + 2)
  "    v69 = iconst.i64 2\n" ++
  "    v70 = iadd v66, v69\n" ++
  "    v71 = load.i16 v70\n" ++       -- vlen (u16)
  "    v72 = uextend.i64 v71\n" ++
  -- Value data starts at: scanResult_off + byte_offset + 4 + klen
  "    v73 = iadd v61, v56\n" ++      -- byte_offset + 4
  "    v74 = iadd v73, v68\n" ++      -- + klen → val data offset (relative to scanResult)
  "    v75 = iadd v52, v74\n" ++      -- val data offset in shared mem
  -- cl_file_write(ptr, filename, src_off, file_offset, size)
  "    v76 = ireduce.i32 v72\n" ++
  "    v77 = sextend.i64 v76\n" ++    -- vlen as i64 for file_write
  s!"    v78 = iconst.i64 {scanFn}\n" ++
  "    v79 = call fn7(v0, v78, v75, v62, v77)\n" ++
  -- Advance: i++, byte_offset += 4 + klen + vlen, file_off += vlen
  "    v80 = iadd v60, v14\n" ++
  "    v81 = iadd v73, v68\n" ++      -- byte_offset + 4 + klen
  "    v82 = iadd v81, v72\n" ++      -- + vlen
  "    v83 = iadd v62, v72\n" ++      -- file_off + vlen
  "    jump block7(v80, v82, v83)\n" ++
  "\n" ++

  -- =====================================================================
  -- block9: Scan done — cursor_scan employees again for filter
  -- =====================================================================
  "block9:\n" ++
  s!"    v84 = iconst.i64 {scanRes2}\n" ++
  "    v85 = call fn6(v0, v10, v3, v53, v54, v84)\n" ++  -- v85 = count
  -- Jump to filter loop: (i=0, offset=4, file_off=0)
  "    jump block10(v3, v56, v3)\n" ++
  "\n" ++

  -- =====================================================================
  -- block10: Filter output — loop header
  -- args: v90=i, v91=byte offset in result, v92=file_off
  -- =====================================================================
  "block10(v90: i64, v91: i64, v92: i64):\n" ++
  "    v93 = sextend.i64 v85\n" ++
  "    v94 = icmp uge v90, v93\n" ++
  "    brif v94, block17, block11\n" ++
  "\n" ++

  -- =====================================================================
  -- block11: Filter — parse entry, decide include/skip
  -- =====================================================================
  "block11:\n" ++
  -- Read klen, vlen
  "    v95 = iadd v84, v91\n" ++
  "    v96 = iadd v0, v95\n" ++
  "    v97 = load.i16 v96\n" ++       -- klen
  "    v98 = uextend.i64 v97\n" ++
  "    v99 = iconst.i64 2\n" ++
  "    v100 = iadd v96, v99\n" ++
  "    v101 = load.i16 v100\n" ++     -- vlen
  "    v102 = uextend.i64 v101\n" ++
  -- Compute val offset: result_off + byte_offset + 4 + klen
  "    v103 = iadd v91, v56\n" ++     -- + 4
  "    v104 = iadd v103, v98\n" ++    -- + klen
  "    v105 = iadd v84, v104\n" ++    -- val offset in shared mem
  -- Read key value (first 4 bytes of key = u32 LE)
  -- Key starts at result_off + byte_offset + 4
  "    v106 = iadd v84, v103\n" ++    -- key offset in shared mem
  "    v107 = iadd v0, v106\n" ++
  "    v108 = load.i32 v107\n" ++     -- key as i32
  -- If key == 0 (header row), always include
  "    v109 = iconst.i32 0\n" ++
  "    v110 = icmp eq v108, v109\n" ++
  "    brif v110, block14, block12(v3)\n" ++
  "\n" ++

  -- =====================================================================
  -- block12: Filter — substring scan: check for ",Seattle," in value
  -- Scan through value bytes looking for match
  -- args: v120=scan_pos (position within value to check)
  -- =====================================================================
  "block12(v120: i64):\n" ++
  -- If scan_pos + seattleStrLen > vlen, no match possible
  s!"    v121 = iconst.i64 {seaLen}\n" ++  -- 9
  "    v122 = iadd v120, v121\n" ++
  "    v123 = icmp ugt v122, v102\n" ++   -- scan_pos + 9 > vlen?
  "    brif v123, block16, block13(v3)\n" ++
  "\n" ++

  -- =====================================================================
  -- block13: Filter — compare substring at position
  -- Compare bytes at val[scan_pos..scan_pos+9] vs ",Seattle,"
  -- args: v130=match_idx (0..8)
  -- =====================================================================
  "block13(v130: i64):\n" ++
  "    v131 = icmp uge v130, v121\n" ++  -- match_idx >= 9?
  "    brif v131, block14, block50\n" ++  -- all matched → include row
  "\n" ++

  -- block50: compare one byte
  "block50:\n" ++
  -- Load val[scan_pos + match_idx]
  "    v132 = iadd v120, v130\n" ++      -- scan_pos + match_idx
  "    v133 = iadd v105, v132\n" ++      -- val_off + scan_pos + match_idx
  "    v134 = iadd v0, v133\n" ++
  "    v135 = load.i8 v134\n" ++
  -- Load seattle_str[match_idx]
  s!"    v136 = iconst.i64 {seaStr}\n" ++
  "    v137 = iadd v136, v130\n" ++
  "    v138 = iadd v0, v137\n" ++
  "    v139 = load.i8 v138\n" ++
  -- Compare
  "    v140 = icmp eq v135, v139\n" ++
  "    v141 = iadd v130, v14\n" ++       -- match_idx + 1
  "    brif v140, block13(v141), block15\n" ++
  "\n" ++

  -- =====================================================================
  -- block14: Filter — match! Write row to filter output
  -- =====================================================================
  "block14:\n" ++
  "    v142 = ireduce.i32 v102\n" ++
  "    v143 = sextend.i64 v142\n" ++
  s!"    v144 = iconst.i64 {filterFn}\n" ++
  "    v145 = call fn7(v0, v144, v105, v92, v143)\n" ++
  -- Advance: i++, update byte_offset, file_off += vlen
  "    v146 = iadd v90, v14\n" ++
  "    v147 = iadd v104, v102\n" ++    -- byte_offset + 4 + klen + vlen (relative)
  "    v148 = iadd v92, v102\n" ++     -- file_off + vlen
  "    jump block10(v146, v147, v148)\n" ++
  "\n" ++

  -- =====================================================================
  -- block15: Filter — no match at this position, try next
  -- =====================================================================
  "block15:\n" ++
  "    v149 = iadd v120, v14\n" ++     -- scan_pos + 1
  "    jump block12(v149)\n" ++
  "\n" ++

  -- =====================================================================
  -- block16: Filter — no match at all, skip row
  -- =====================================================================
  "block16:\n" ++
  "    v150 = iadd v90, v14\n" ++
  "    v151 = iadd v104, v102\n" ++
  "    jump block10(v150, v151, v92)\n" ++   -- file_off unchanged
  "\n" ++

  -- =====================================================================
  -- block17: Filter done — cursor_scan departments for join.csv
  -- =====================================================================
  "block17:\n" ++
  "    v152 = call fn6(v0, v12, v3, v53, v54, v52)\n" ++  -- reuse scanResult buf
  -- Jump to join output loop: (i=0, offset=4, file_off=0)
  "    jump block18(v3, v56, v3)\n" ++
  "\n" ++

  -- =====================================================================
  -- block18: Join output — loop header
  -- args: v160=i, v161=byte offset, v162=file_off
  -- =====================================================================
  "block18(v160: i64, v161: i64, v162: i64):\n" ++
  "    v163 = sextend.i64 v152\n" ++
  "    v164 = icmp uge v160, v163\n" ++
  "    brif v164, block20, block19\n" ++
  "\n" ++

  -- =====================================================================
  -- block19: Join output — write row, advance
  -- =====================================================================
  "block19:\n" ++
  "    v165 = iadd v52, v161\n" ++
  "    v166 = iadd v0, v165\n" ++
  "    v167 = load.i16 v166\n" ++      -- klen
  "    v168 = uextend.i64 v167\n" ++
  "    v169 = iconst.i64 2\n" ++
  "    v170 = iadd v166, v169\n" ++
  "    v171 = load.i16 v170\n" ++      -- vlen
  "    v172 = uextend.i64 v171\n" ++
  "    v173 = iadd v161, v56\n" ++     -- + 4
  "    v174 = iadd v173, v168\n" ++    -- + klen
  "    v175 = iadd v52, v174\n" ++     -- val offset in shared mem
  "    v176 = ireduce.i32 v172\n" ++
  "    v177 = sextend.i64 v176\n" ++
  s!"    v178 = iconst.i64 {joinFn}\n" ++
  "    v179 = call fn7(v0, v178, v175, v162, v177)\n" ++
  "    v180 = iadd v160, v14\n" ++
  "    v181 = iadd v174, v172\n" ++
  "    v182 = iadd v162, v172\n" ++
  "    jump block18(v180, v181, v182)\n" ++
  "\n" ++

  -- =====================================================================
  -- block20: Done — cleanup and return
  -- =====================================================================
  "block20:\n" ++
  "    call fn8(v0)\n" ++
  "    return\n" ++
  "}\n"

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
  -- CLIF IR
  let clifBytes := padTo (stringToBytes clifIrSource) clifIrRegionSize
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
  clifBytes ++
  empBufBytes ++ deptBufBytes ++
  keyScratchBytes ++ scanResultBytes ++ scanResult2Bytes

-- ---------------------------------------------------------------------------
-- Algorithm definition
-- ---------------------------------------------------------------------------

def csvAlgorithm : Algorithm :=
  let clifCallAction : Action :=
    { kind := .ClifCall, dst := u32 0, src := u32 1, offset := u32 0, size := u32 0 }
  {
    actions := [clifCallAction],
    payloads := payloads,
    state := { cranelift_ir_offsets := [clifIr_off] },
    units := { cranelift_units := 0 },
    worker_threads := some 1,
    blocking_threads := some 1,
    stack_size := some (512 * 1024),
    timeout_ms := some 30000,
    thread_name_prefix := some "csv-demo",
    additional_shared_memory := 0
  }

end CsvDemo

def main : IO Unit := do
  let json := toJson CsvDemo.csvAlgorithm
  IO.println (Json.compress json)
