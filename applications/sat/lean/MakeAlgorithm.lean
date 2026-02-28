import AlgorithmLib
open Lean (Json toJson)
open AlgorithmLib

namespace Algorithm

-- ---------------------------------------------------------------------------
-- DPLL SAT solver for DIMACS CNF
--
-- Single CLIF function:
--   1. cl_file_read — load CNF file into memory
--   2. Parse DIMACS: extract num_vars, num_clauses, build clause DB
--   3. Run DPLL with unit propagation
--   4. cl_file_write — write "s SATISFIABLE\nv ..." or "s UNSATISFIABLE\n"
--
-- Memory layout (all offsets relative to shared memory base v0):
--   [0x00..0x40)        reserved (HT context)
--   [0x40..0x48)        scratch: num_vars (i64)
--   [0x48..0x50)        scratch: num_clauses (i64)
--   [0x50..0x58)        scratch: clause_db_count (i64, actual clauses parsed)
--   [0x58..0x60)        scratch: result flag (0=unknown, 1=SAT, 2=UNSAT)
--   [0x60..0x68)        scratch: output string length
--   [0x100..0x200)      input CNF filename (null-terminated, patched at runtime)
--   [0x200..0x300)      output filename (null-terminated)
--   [0x300..0x400)      output string buffer start (extends into output region)
--   [0x400..cnf_off)    unused padding
--   [cnf_off..db_off)   raw CNF file contents
--   [db_off..assign_off) clause database: packed [len:i32, lit0:i32, lit1:i32, ...]
--   [assign_off..trail_off) variable assignments: i8 per var (0=unset, 1=true, -1=false)
--   [trail_off..out_off) decision trail: i32 per entry (signed literal)
--   [out_off..mem_size)  output string buffer
-- ---------------------------------------------------------------------------

-- Memory region sizes
def maxVars : Nat := 10000
def maxClauses : Nat := 50000
def maxClauseWords : Nat := 500000   -- total i32 words in clause DB (lengths + literals)
def maxCnfFileSize : Nat := 4 * 1024 * 1024  -- 4MB max CNF file

-- Payload offsets
def reserved_off : Nat := 0x00
def numVars_off : Nat := 0x40
def numClauses_off : Nat := 0x48
def clauseCount_off : Nat := 0x50
def resultFlag_off : Nat := 0x58
def outLen_off : Nat := 0x60
def inputFilename_off : Nat := 0x100
def outputFilename_off : Nat := 0x200
def outputStr_off : Nat := 0x300
def cnf_off : Nat := 0x1000
def db_off : Nat := cnf_off + maxCnfFileSize
def assign_off : Nat := db_off + maxClauseWords * 4
def trail_off : Nat := assign_off + maxVars + 16
def out_off : Nat := trail_off + maxVars * 4 + 16

-- Clause index: array of i64 offsets into clause DB (byte offset from db_off)
def clauseIndex_off : Nat := out_off
-- Decision stack: trail_depth at each decision point (i64 per entry)
def decStack_off : Nat := clauseIndex_off + maxClauses * 8
def solver_scratch_off : Nat := decStack_off + maxVars * 8

def totalMemory : Nat := solver_scratch_off + 0x10000

-- ---------------------------------------------------------------------------
-- CLIF IR: DPLL SAT solver
--
-- The solver uses a simple DPLL approach:
--   block0: entry — read file, call parser, call solver, write output
--   block_parse_*: DIMACS parser blocks
--   block_dpll_*: DPLL solver blocks
--   block69: format and write result
--
-- For clarity we split into multiple CLIF functions:
--   u0:0 — noop (required)
--   u0:1 — main orchestrator: read file, parse, solve, write
--
-- The parser and solver are inline in u0:1 as a block-based state machine.
-- ---------------------------------------------------------------------------

def clifIrSource : String :=
  let inFname := toString inputFilename_off
  let outFname := toString outputFilename_off
  let cnfStart := toString cnf_off
  let dbStart := toString db_off
  let assignStart := toString assign_off
  let trailStart := toString trail_off
  let outStart := toString out_off
  let nVarsOff := toString numVars_off
  let nClausesOff := toString numClauses_off
  let cCountOff := toString clauseCount_off
  let resFlagOff := toString resultFlag_off
  let outLenOff := toString outLen_off
  let clIdxOff := toString clauseIndex_off
  let decStackStart := toString decStack_off
  let maxV := toString maxVars

  -- noop function
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    return\n" ++
  "}\n\n" ++

  -- main orchestrator
  "function u0:1(i64) system_v {\n" ++
  "    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v\n" ++    -- file_read, file_write
  "    fn0 = %cl_file_read sig0\n" ++
  "    fn1 = %cl_file_write sig0\n" ++
  "\n" ++
  "block0(v0: i64):\n" ++

  -- Step 1: Read CNF file
  s!"    v1 = iconst.i64 {inFname}\n" ++
  s!"    v2 = iconst.i64 {cnfStart}\n" ++
  "    v3 = iconst.i64 0\n" ++
  "    v4 = call fn0(v0, v1, v2, v3, v3)\n" ++   -- v4 = bytes read

  -- Initialize scratch
  "    v5 = iconst.i64 0\n" ++
  s!"    v6 = iconst.i64 {nVarsOff}\n" ++
  "    v7 = iadd v0, v6\n" ++
  "    store v5, v7\n" ++                          -- num_vars = 0
  s!"    v8 = iconst.i64 {nClausesOff}\n" ++
  "    v9 = iadd v0, v8\n" ++
  "    store v5, v9\n" ++                          -- num_clauses = 0
  s!"    v10 = iconst.i64 {cCountOff}\n" ++
  "    v11 = iadd v0, v10\n" ++
  "    store v5, v11\n" ++                         -- clause_count = 0
  s!"    v12 = iconst.i64 {resFlagOff}\n" ++
  "    v13 = iadd v0, v12\n" ++
  "    store v5, v13\n" ++                         -- result = 0 (unknown)

  -- Initialize assignment array to 0 (unset) — zero maxVars bytes
  -- We'll do a simple loop
  s!"    v14 = iconst.i64 {assignStart}\n" ++
  "    v15 = iadd v0, v14\n" ++                    -- base of assignment array
  s!"    v16 = iconst.i64 {maxV}\n" ++
  "    v17 = iconst.i64 0\n" ++
  "    v23 = iconst.i64 1\n" ++
  "    v45 = iconst.i64 32\n" ++                   -- ' '
  "    v47 = iconst.i64 9\n" ++                    -- '\t'
  "    v80 = iconst.i64 48\n" ++                   -- '0'
  "    v82 = iconst.i64 58\n" ++                   -- '9' + 1
  "    v162 = iconst.i64 4\n" ++
  "    v290 = iconst.i64 8\n" ++
  s!"    v292 = iconst.i64 {clIdxOff}\n" ++
  s!"    v337 = iconst.i64 {dbStart}\n" ++
  s!"    v610 = iconst.i64 {outStart}\n" ++
  s!"    v800 = iconst.i64 {decStackStart}\n" ++
  "    jump block1(v3)\n" ++            -- i = 0

  "\n" ++
  -- Zero assignment array
  "block1(v20: i64):\n" ++
  "    v21 = icmp uge v20, v16\n" ++               -- i >= maxVars?
  "    brif v21, block3, block2\n" ++
  "\n" ++
  "block2:\n" ++
  "    v22 = iadd v15, v20\n" ++                   -- &assign[i]
  "    istore8 v17, v22\n" ++                      -- assign[i] = 0
  "    v24 = iadd v20, v23\n" ++
  "    jump block1(v24)\n" ++

  "\n" ++
  -- ===================================================================
  -- DIMACS PARSER
  -- Scan the CNF text byte by byte.
  -- Skip lines starting with 'c' (comments).
  -- Parse 'p cnf <nvars> <nclauses>' header.
  -- Parse clause lines: sequences of integers terminated by 0.
  --
  -- State: pos (byte offset into CNF buffer)
  -- Writes: num_vars, num_clauses to scratch
  --         clause DB starting at db_off: [len:i32, lit0:i32, ...]
  --         clause_count to scratch
  -- ===================================================================
  "block3:\n" ++
  -- pos=0, db_write_ptr=0 (offset from db_off in bytes)
  "    jump block4(v3, v3)\n" ++

  "\n" ++
  -- block4(pos, db_ptr): start of a new line
  "block4(v30: i64, v31: i64):\n" ++
  -- Check if pos >= file_size
  "    v32 = icmp uge v30, v4\n" ++                -- pos >= bytes_read?
  "    brif v32, block5, block6\n" ++

  "\n" ++
  "block6:\n" ++
  -- Read byte at cnf_off + pos
  "    v33 = iadd v2, v30\n" ++                    -- cnf_off + pos (relative)
  "    v34 = iadd v0, v33\n" ++                    -- absolute addr
  "    v36 = uload8.i64 v34\n" ++                   -- byte at pos

  -- Check for comment line (c) or header line (p)
  "    v37 = iconst.i64 99\n" ++                   -- 'c' = 99
  "    v38 = icmp eq v36, v37\n" ++
  "    brif v38, block7(v30, v31), block8\n" ++

  "\n" ++
  "block8:\n" ++
  "    v39 = iconst.i64 112\n" ++                  -- 'p' = 112
  "    v40 = icmp eq v36, v39\n" ++
  "    brif v40, block15(v30, v31), block9\n" ++

  "\n" ++
  "block9:\n" ++
  -- Check for empty line / whitespace — skip to clause parsing
  "    v41 = iconst.i64 10\n" ++                   -- '\n' = 10
  "    v42 = icmp eq v36, v41\n" ++
  "    brif v42, block10(v30, v31), block11\n" ++

  "\n" ++
  "block11:\n" ++
  "    v43 = iconst.i64 13\n" ++                   -- '\r' = 13
  "    v44 = icmp eq v36, v43\n" ++
  "    brif v44, block10(v30, v31), block12\n" ++

  "\n" ++
  "block12:\n" ++
  "    v46 = icmp eq v36, v45\n" ++
  "    brif v46, block10(v30, v31), block13\n" ++

  "\n" ++
  "block13:\n" ++
  "    v48 = icmp eq v36, v47\n" ++
  "    brif v48, block10(v30, v31), block30(v30, v31)\n" ++

  "\n" ++
  -- Skip to end of line
  "block7(v50: i64, v51: i64):\n" ++
  "    v52 = icmp uge v50, v4\n" ++
  "    brif v52, block5, block14\n" ++
  "\n" ++
  "block14:\n" ++
  "    v53 = iadd v2, v50\n" ++
  "    v54 = iadd v0, v53\n" ++
  "    v56 = uload8.i64 v54\n" ++
  "    v57 = iconst.i64 10\n" ++
  "    v58 = icmp eq v56, v57\n" ++
  "    v59 = iadd v50, v23\n" ++                   -- pos + 1
  "    brif v58, block4(v59, v51), block7(v59, v51)\n" ++

  "\n" ++
  -- Skip single whitespace char
  "block10(v60: i64, v61: i64):\n" ++
  "    v62 = iadd v60, v23\n" ++                   -- pos + 1
  "    jump block4(v62, v61)\n" ++

  "\n" ++
  -- ===================================================================
  -- Parse 'p cnf <nvars> <nclauses>' header
  -- We skip past "p cnf " then parse two integers
  -- ===================================================================
  "block15(v70: i64, v71: i64):\n" ++
  -- Skip "p cnf " — advance pos until we hit a digit
  "    v72 = iadd v70, v23\n" ++                   -- skip 'p'
  "    jump block16(v72, v71)\n" ++

  "\n" ++
  "block16(v73: i64, v74: i64):\n" ++
  "    v75 = icmp uge v73, v4\n" ++
  "    brif v75, block5, block17\n" ++
  "\n" ++
  "block17:\n" ++
  "    v76 = iadd v2, v73\n" ++
  "    v77 = iadd v0, v76\n" ++
  "    v79 = uload8.i64 v77\n" ++
  -- Check if digit (48..57)
  "    v81 = icmp uge v79, v80\n" ++
  "    v83 = icmp ult v79, v82\n" ++
  "    v84 = band v81, v83\n" ++
  "    brif v84, block19(v73, v74, v3), block18(v73, v74)\n" ++

  "\n" ++
  "block18(v85: i64, v86: i64):\n" ++
  "    v87 = iadd v85, v23\n" ++
  "    jump block16(v87, v86)\n" ++

  "\n" ++
  -- Parse nvars integer
  "block19(v90: i64, v91: i64, v92: i64):\n" ++   -- pos, db_ptr, accum
  "    v93 = icmp uge v90, v4\n" ++
  "    brif v93, block22(v90, v91, v92), block20\n" ++
  "\n" ++
  "block20:\n" ++
  "    v94 = iadd v2, v90\n" ++
  "    v95 = iadd v0, v94\n" ++
  "    v97 = uload8.i64 v95\n" ++
  "    v98 = icmp uge v97, v80\n" ++               -- >= '0'?
  "    v99 = icmp ult v97, v82\n" ++               -- < '9'+1?
  "    v100 = band v98, v99\n" ++
  "    brif v100, block21(v90, v91, v92, v97), block22(v90, v91, v92)\n" ++

  "\n" ++
  "block21(v101: i64, v102: i64, v103: i64, v104: i64):\n" ++
  "    v105 = iconst.i64 10\n" ++
  "    v106 = imul v103, v105\n" ++                -- accum * 10
  "    v107 = isub v104, v80\n" ++                 -- digit value
  "    v108 = iadd v106, v107\n" ++                -- new accum
  "    v109 = iadd v101, v23\n" ++                 -- pos + 1
  "    jump block19(v109, v102, v108)\n" ++

  "\n" ++
  "block22(v110: i64, v111: i64, v112: i64):\n" ++
  -- Store num_vars
  "    store v112, v7\n" ++                        -- v7 = &num_vars_off
  -- Skip spaces to nclauses
  "    jump block23(v110, v111)\n" ++

  "\n" ++
  "block23(v113: i64, v114: i64):\n" ++
  "    v115 = icmp uge v113, v4\n" ++
  "    brif v115, block5, block24\n" ++
  "\n" ++
  "block24:\n" ++
  "    v116 = iadd v2, v113\n" ++
  "    v117 = iadd v0, v116\n" ++
  "    v119 = uload8.i64 v117\n" ++
  "    v120 = icmp uge v119, v80\n" ++
  "    v121 = icmp ult v119, v82\n" ++
  "    v122 = band v120, v121\n" ++
  "    brif v122, block26(v113, v114, v3), block25(v113, v114)\n" ++

  "\n" ++
  "block25(v123: i64, v124: i64):\n" ++
  "    v125 = iadd v123, v23\n" ++
  "    jump block23(v125, v124)\n" ++

  "\n" ++
  -- Parse nclauses integer
  "block26(v130: i64, v131: i64, v132: i64):\n" ++
  "    v133 = icmp uge v130, v4\n" ++
  "    brif v133, block29(v130, v131, v132), block27\n" ++
  "\n" ++
  "block27:\n" ++
  "    v134 = iadd v2, v130\n" ++
  "    v135 = iadd v0, v134\n" ++
  "    v137 = uload8.i64 v135\n" ++
  "    v138 = icmp uge v137, v80\n" ++
  "    v139 = icmp ult v137, v82\n" ++
  "    v140 = band v138, v139\n" ++
  "    brif v140, block28(v130, v131, v132, v137), block29(v130, v131, v132)\n" ++

  "\n" ++
  "block28(v141: i64, v142: i64, v143: i64, v144: i64):\n" ++
  "    v145 = iconst.i64 10\n" ++
  "    v146 = imul v143, v145\n" ++
  "    v147 = isub v144, v80\n" ++
  "    v148 = iadd v146, v147\n" ++
  "    v149 = iadd v141, v23\n" ++
  "    jump block26(v149, v142, v148)\n" ++

  "\n" ++
  "block29(v150: i64, v151: i64, v152: i64):\n" ++
  -- Store num_clauses
  "    store v152, v9\n" ++                        -- v9 = &num_clauses_off
  -- Skip to end of header line
  "    jump block7(v150, v151)\n" ++

  "\n" ++
  -- ===================================================================
  -- Parse clauses: sequence of signed integers terminated by 0
  -- Each clause stored as: [len:i32, lit0:i32, lit1:i32, ...]
  -- We first store literals starting at db_ptr+4, then backpatch len.
  -- ===================================================================
  "block30(v160: i64, v161: i64):\n" ++
  -- v160 = pos in CNF, v161 = db_ptr (byte offset from db_off)
  -- Save clause start = db_ptr + 4 (skip length slot)
  "    v163 = iadd v161, v162\n" ++                -- lit_ptr = db_ptr + 4
  "    jump block31(v160, v161, v163, v3)\n" ++  -- lit_count = 0

  "\n" ++
  -- Skip whitespace before next integer
  "block31(v170: i64, v171: i64, v172: i64, v173: i64):\n" ++
  -- v170=pos, v171=clause_start_db_ptr, v172=lit_write_ptr, v173=lit_count
  "    v174 = icmp uge v170, v4\n" ++
  "    brif v174, block33(v170, v171, v172, v173), block32\n" ++
  "\n" ++
  "block32:\n" ++
  "    v175 = iadd v2, v170\n" ++
  "    v176 = iadd v0, v175\n" ++
  "    v178 = uload8.i64 v176\n" ++
  -- newline = end of line, go back to line parser
  "    v179 = iconst.i64 10\n" ++
  "    v180 = icmp eq v178, v179\n" ++
  "    v181 = iadd v170, v23\n" ++                 -- pos+1
  "    brif v180, block33(v181, v171, v172, v173), block34\n" ++

  "\n" ++
  "block34:\n" ++
  "    v182 = iconst.i64 13\n" ++                  -- '\r'
  "    v183 = icmp eq v178, v182\n" ++
  "    brif v183, block33(v181, v171, v172, v173), block35\n" ++

  "\n" ++
  "block35:\n" ++
  -- If space or tab, skip
  "    v184 = icmp eq v178, v45\n" ++              -- v45 = 32 (space)
  "    brif v184, block31(v181, v171, v172, v173), block36\n" ++

  "\n" ++
  "block36:\n" ++
  "    v185 = icmp eq v178, v47\n" ++              -- v47 = 9 (tab)
  "    brif v185, block31(v181, v171, v172, v173), block37(v170, v171, v172, v173)\n" ++

  "\n" ++
  -- Parse a signed integer (literal)
  -- Check for '-' sign
  "block37(v190: i64, v191: i64, v192: i64, v193: i64):\n" ++
  "    v194 = iadd v2, v190\n" ++
  "    v195 = iadd v0, v194\n" ++
  "    v197 = uload8.i64 v195\n" ++
  "    v198 = iconst.i64 45\n" ++                  -- '-' = 45
  "    v199 = icmp eq v197, v198\n" ++
  "    v200 = iadd v190, v23\n" ++                 -- pos after '-'
  -- negative: sign = -1, advance pos; positive: sign = 1, keep pos
  "    brif v199, block39(v200, v191, v192, v193, v3, v23), block38(v190, v191, v192, v193)\n" ++

  "\n" ++
  "block38(v201: i64, v202: i64, v203: i64, v204: i64):\n" ++
  "    jump block39(v201, v202, v203, v204, v3, v3)\n" ++  -- accum=0, is_neg=0

  "\n" ++
  -- Parse digits of integer
  -- v210=pos, v211=db_start, v212=lit_ptr, v213=lit_count, v214=accum, v215=is_neg (0 or 1)
  "block39(v210: i64, v211: i64, v212: i64, v213: i64, v214: i64, v215: i64):\n" ++
  "    v216 = icmp uge v210, v4\n" ++
  "    brif v216, block42(v210, v211, v212, v213, v214, v215), block40\n" ++
  "\n" ++
  "block40:\n" ++
  "    v217 = iadd v2, v210\n" ++
  "    v218 = iadd v0, v217\n" ++
  "    v220 = uload8.i64 v218\n" ++
  "    v221 = icmp uge v220, v80\n" ++             -- >= '0'
  "    v222 = icmp ult v220, v82\n" ++             -- < '9'+1
  "    v223 = band v221, v222\n" ++
  "    brif v223, block41(v210, v211, v212, v213, v214, v215, v220), block42(v210, v211, v212, v213, v214, v215)\n" ++

  "\n" ++
  "block41(v230: i64, v231: i64, v232: i64, v233: i64, v234: i64, v235: i64, v236: i64):\n" ++
  "    v237 = iconst.i64 10\n" ++
  "    v238 = imul v234, v237\n" ++
  "    v239 = isub v236, v80\n" ++                 -- digit value
  "    v240 = iadd v238, v239\n" ++                -- new accum
  "    v241 = iadd v230, v23\n" ++
  "    jump block39(v241, v231, v232, v233, v240, v235)\n" ++

  "\n" ++
  -- Integer parsed. If accum=0, this is the clause terminator.
  -- If is_neg=1, negate the value.
  "block42(v250: i64, v251: i64, v252: i64, v253: i64, v254: i64, v255: i64):\n" ++
  "    v256 = icmp eq v254, v3\n" ++               -- accum == 0?
  "    brif v256, block33(v250, v251, v252, v253), block43(v250, v251, v252, v253, v254, v255)\n" ++

  "\n" ++
  -- Store literal: if is_neg, value = -accum, else value = accum
  "block43(v260: i64, v261: i64, v262: i64, v263: i64, v264: i64, v265: i64):\n" ++
  "    v266 = ineg v264\n" ++                      -- -accum
  "    v267 = icmp eq v265, v23\n" ++              -- is_neg == 1?
  "    v268 = select v267, v266, v264\n" ++        -- final literal value
  "    v269 = ireduce.i32 v268\n" ++               -- as i32
  -- Store at db_off + lit_ptr
  "    v271 = iadd v337, v262\n" ++                -- db_off + lit_ptr
  "    v272 = iadd v0, v271\n" ++                  -- absolute addr
  "    store v269, v272\n" ++
  -- Advance lit_ptr by 4, lit_count by 1
  "    v273 = iadd v262, v162\n" ++                -- lit_ptr + 4
  "    v274 = iadd v263, v23\n" ++                 -- lit_count + 1
  "    jump block31(v260, v261, v273, v274)\n" ++

  "\n" ++
  -- End of clause: store length at db_start, advance db_ptr
  "block33(v280: i64, v281: i64, v282: i64, v283: i64):\n" ++
  -- If lit_count == 0, this was an empty line — skip
  "    v284 = icmp eq v283, v3\n" ++
  "    brif v284, block4(v280, v281), block44\n" ++
  "\n" ++
  "block44:\n" ++
  -- Store clause length at db_off + clause_start
  "    v285 = ireduce.i32 v283\n" ++               -- lit_count as i32
  "    v287 = iadd v337, v281\n" ++                -- db_off + db_start
  "    v288 = iadd v0, v287\n" ++                  -- absolute addr
  "    store v285, v288\n" ++
  -- Store clause index: clauseIndex[clause_count] = db_start
  "    v289 = load.i64 v11\n" ++                   -- clause_count
  "    v291 = imul v289, v290\n" ++                -- clause_count * 8
  "    v293 = iadd v292, v291\n" ++                -- clauseIndex_off + clause_count * 8
  "    v294 = iadd v0, v293\n" ++
  "    store v281, v294\n" ++                      -- store db_ptr offset
  -- Increment clause_count
  "    v295 = iadd v289, v23\n" ++                 -- clause_count + 1
  "    store v295, v11\n" ++
  -- New db_ptr = lit_ptr (past the last literal)
  "    jump block4(v280, v282)\n" ++

  "\n" ++
  -- ===================================================================
  -- Parse done — start DPLL solver
  -- ===================================================================
  "block5:\n" ++
  -- Load num_vars and clause_count
  "    v300 = load.i64 v7\n" ++                    -- num_vars
  "    v301 = load.i64 v11\n" ++                   -- clause_count
  -- Start DPLL with trail_depth = 0, dec_depth = 0
  "    jump block45(v3, v3)\n" ++      -- trail_depth=0, dec_depth=0

  "\n" ++
  -- ===================================================================
  -- DPLL SOLVER with decision stack
  --
  -- block45(trail_depth, dec_depth): unit propagation entry
  -- block58(trail_depth, dec_depth): decide
  -- block64(trail_depth, dec_depth): conflict → backtrack
  --
  -- Decision stack at decStack_off: decStack[i] = trail_depth at decision i
  -- Trail at trail_off: trail[i] = signed literal (i32)
  -- ===================================================================

  -- Unit propagation: scan all clauses repeatedly until no more units found
  "block45(v310: i64, v311: i64):\n" ++  -- trail_depth, dec_depth
  -- found_unit = 0, clause_idx = 0
  "    jump block46(v310, v311, v3, v3)\n" ++  -- td, dd, clause_idx, found_unit

  "\n" ++
  -- Scan clause at index clause_idx
  "block46(v320: i64, v321: i64, v3200: i64, v322: i64):\n" ++  -- td, dd, clause_idx, found_unit
  "    v323 = load.i64 v11\n" ++                   -- clause_count
  "    v324 = icmp uge v3200, v323\n" ++
  "    brif v324, block47(v320, v321, v322), block48(v320, v321, v3200, v322)\n" ++

  "\n" ++
  -- Evaluate clause: count unset, false, find the one unset literal
  "block48(v330: i64, v331: i64, v3300: i64, v332: i64):\n" ++  -- td, dd, clause_idx, found_unit
  -- Load clause offset from index
  "    v333 = imul v3300, v290\n" ++               -- clause_idx * 8
  "    v334 = iadd v292, v333\n" ++                -- clauseIndex_off + ...
  "    v335 = iadd v0, v334\n" ++
  "    v336 = load.i64 v335\n" ++                  -- db_ptr offset for this clause

  -- Load clause length
  "    v338 = iadd v337, v336\n" ++                -- db_off + clause_offset
  "    v339 = iadd v0, v338\n" ++
  "    v340 = load.i32 v339\n" ++                  -- clause length
  "    v341 = uextend.i64 v340\n" ++

  -- Scan literals
  "    v342 = iadd v336, v162\n" ++                -- first literal offset (skip length)
  "    jump block49(v330, v331, v3300, v332, v342, v341, v3, v3, v3, v3)\n" ++

  "\n" ++
  -- Scan one literal in clause
  -- v350=td, v351=dd, v3500=clause_idx, v352=found_unit, v353=lit_offset, v354=remaining,
  -- v355=count_false, v356=count_unset, v357=last_unset_lit, v358=is_sat
  "block49(v350: i64, v351: i64, v3500: i64, v352: i64, v353: i64, v354: i64, v355: i64, v356: i64, v357: i64, v358: i64):\n" ++
  "    v359 = icmp eq v354, v3\n" ++               -- remaining == 0?
  "    brif v359, block50(v350, v351, v3500, v352, v355, v356, v357, v358), block51\n" ++

  "\n" ++
  "block51:\n" ++
  -- Load literal
  "    v360 = iadd v337, v353\n" ++                -- db_off + lit_offset
  "    v361 = iadd v0, v360\n" ++
  "    v362 = load.i32 v361\n" ++                  -- literal (signed i32)
  "    v363 = sextend.i64 v362\n" ++               -- as i64

  -- Get variable index: abs(lit) - 1
  "    v364 = iconst.i64 0\n" ++
  "    v365 = icmp slt v363, v364\n" ++
  "    v366 = ineg v363\n" ++
  "    v367 = select v365, v366, v363\n" ++        -- abs(lit)
  "    v368 = isub v367, v23\n" ++                 -- var_idx = abs(lit) - 1

  -- Load assignment for this variable
  "    v369 = iadd v15, v368\n" ++                 -- &assign[var_idx]
  "    v371 = sload8.i64 v369\n" ++                -- assignment (0=unset, 1=true, -1=false)

  -- Check if unset
  "    v372 = icmp eq v371, v364\n" ++             -- assign == 0?
  "    brif v372, block52(v350, v351, v3500, v352, v353, v354, v355, v356, v363, v358), block53\n" ++

  "\n" ++
  "block53:\n" ++
  -- Check if this literal is satisfied
  "    v373 = iconst.i64 1\n" ++
  "    v374 = iconst.i64 -1\n" ++
  "    v375 = icmp sgt v363, v364\n" ++
  "    v376 = select v375, v373, v374\n" ++        -- sign(lit)
  "    v377 = icmp eq v376, v371\n" ++             -- satisfied?
  "    v378 = select v377, v23, v358\n" ++         -- new is_sat
  "    v380 = select v377, v3, v23\n" ++           -- 1 if false lit, 0 if satisfied
  "    v381 = iadd v355, v380\n" ++                -- new count_false
  -- Advance to next literal
  "    v382 = iadd v353, v162\n" ++                -- lit_offset + 4
  "    v383 = isub v354, v23\n" ++                 -- remaining - 1
  "    jump block49(v350, v351, v3500, v352, v382, v383, v381, v356, v357, v378)\n" ++

  "\n" ++
  "block52(v390: i64, v391: i64, v3900: i64, v392: i64, v393: i64, v394: i64, v395: i64, v396: i64, v397: i64, v398: i64):\n" ++
  -- This literal is unset — increment count_unset, record as last_unset
  "    v399 = iadd v396, v23\n" ++                 -- count_unset + 1
  "    v400 = iadd v393, v162\n" ++                -- lit_offset + 4
  "    v401 = isub v394, v23\n" ++                 -- remaining - 1
  "    jump block49(v390, v391, v3900, v392, v400, v401, v395, v399, v397, v398)\n" ++

  "\n" ++
  -- Clause evaluation result
  -- v410=td, v411=dd, v4100=clause_idx, v412=found_unit, v413=count_false, v414=count_unset, v415=last_unset, v416=is_sat
  "block50(v410: i64, v411: i64, v4100: i64, v412: i64, v413: i64, v414: i64, v415: i64, v416: i64):\n" ++
  -- If is_sat=1, clause satisfied, skip
  "    v417 = icmp eq v416, v23\n" ++
  "    v418 = iadd v4100, v23\n" ++                -- next clause_idx
  "    brif v417, block46(v410, v411, v418, v412), block54\n" ++

  "\n" ++
  "block54:\n" ++
  -- If count_unset == 0, CONFLICT
  "    v419 = icmp eq v414, v3\n" ++
  "    brif v419, block64(v410, v411), block55\n" ++

  "\n" ++
  "block55:\n" ++
  -- If count_unset == 1, unit clause => propagate
  "    v420 = icmp eq v414, v23\n" ++
  "    brif v420, block57(v410, v411, v4100, v412, v415), block56\n" ++

  "\n" ++
  "block56:\n" ++
  -- More than 1 unset literal, continue scanning
  "    v421 = iadd v4100, v23\n" ++
  "    jump block46(v410, v411, v421, v412)\n" ++

  "\n" ++
  -- Assign unit literal
  "block57(v430: i64, v431: i64, v4300: i64, v432: i64, v433: i64):\n" ++  -- td, dd, clause_idx, found_unit, literal
  -- var_idx = abs(lit) - 1
  "    v434 = iconst.i64 0\n" ++
  "    v435 = icmp slt v433, v434\n" ++
  "    v436 = ineg v433\n" ++
  "    v437 = select v435, v436, v433\n" ++        -- abs(lit)
  "    v438 = isub v437, v23\n" ++                 -- var_idx
  -- assignment value: if lit > 0 then 1 else -1
  "    v439 = iconst.i64 1\n" ++
  "    v440 = iconst.i64 -1\n" ++
  "    v441 = icmp sgt v433, v434\n" ++
  "    v442 = select v441, v439, v440\n" ++        -- assign value
  -- Store assignment
  "    v444 = iadd v15, v438\n" ++                 -- &assign[var_idx]
  "    istore8 v442, v444\n" ++
  -- Store on trail
  s!"    v445 = iconst.i64 {trailStart}\n" ++
  "    v446 = iconst.i64 4\n" ++
  "    v447 = imul v430, v446\n" ++                -- trail_depth * 4
  "    v448 = iadd v445, v447\n" ++
  "    v449 = iadd v0, v448\n" ++
  "    v450 = ireduce.i32 v433\n" ++               -- literal as i32
  "    store v450, v449\n" ++
  -- trail_depth + 1
  "    v451 = iadd v430, v23\n" ++
  -- found_unit = 1, next clause
  "    v452 = iadd v4300, v23\n" ++
  "    jump block46(v451, v431, v452, v23)\n" ++

  "\n" ++
  -- Unit prop scan complete
  "block47(v460: i64, v461: i64, v900: i64):\n" ++  -- td, dd, found_unit
  -- If found_unit, repeat scan from beginning
  "    v462 = icmp eq v900, v23\n" ++
  "    brif v462, block45(v460, v461), block58(v460, v461)\n" ++

  "\n" ++
  -- ===================================================================
  -- DPLL DECIDE: pick first unassigned variable, try both polarities
  -- ===================================================================
  "block58(v470: i64, v901: i64):\n" ++  -- trail_depth, dec_depth
  -- Find first unassigned variable
  "    v471 = load.i64 v7\n" ++                    -- num_vars
  "    jump block59(v470, v901, v3, v471)\n" ++

  "\n" ++
  "block59(v480: i64, v902: i64, v481: i64, v482: i64):\n" ++  -- td, dd, var_i, num_vars
  "    v483 = icmp uge v481, v482\n" ++
  "    brif v483, block63(v480), block60\n" ++  -- all assigned => SAT

  "\n" ++
  "block60:\n" ++
  "    v484 = iadd v15, v481\n" ++                 -- &assign[var_i]
  "    v486 = sload8.i64 v484\n" ++
  "    v487 = icmp eq v486, v3\n" ++               -- unassigned?
  "    brif v487, block62(v480, v902, v481), block61\n" ++

  "\n" ++
  "block61:\n" ++
  "    v488 = iadd v481, v23\n" ++
  "    jump block59(v480, v902, v488, v482)\n" ++

  "\n" ++
  -- Try assigning variable = true
  "block62(v490: i64, v903: i64, v491: i64):\n" ++  -- trail_depth, dec_depth, var_idx
  -- Save decision point: decStack[dec_depth] = trail_depth
  "    v492 = imul v903, v290\n" ++               -- dec_depth * 8
  "    v4920 = iadd v800, v492\n" ++               -- decStack_off + dec_depth*8
  "    v4921 = iadd v0, v4920\n" ++
  "    store v490, v4921\n" ++                     -- decStack[dec_depth] = trail_depth
  -- Assign var true
  "    v493 = iadd v15, v491\n" ++                 -- &assign[var_idx]
  "    istore8 v23, v493\n" ++                     -- assign = 1 (true)
  -- Store positive literal on trail: (var_idx + 1)
  "    v494 = iadd v491, v23\n" ++                 -- var_idx + 1 = literal
  s!"    v495 = iconst.i64 {trailStart}\n" ++
  "    v496 = imul v490, v162\n" ++                -- trail_depth * 4
  "    v497 = iadd v495, v496\n" ++
  "    v498 = iadd v0, v497\n" ++
  "    v499 = ireduce.i32 v494\n" ++
  "    store v499, v498\n" ++
  "    v500 = iadd v490, v23\n" ++                 -- new trail_depth
  "    v501 = iadd v903, v23\n" ++                -- new dec_depth
  -- Propagate
  "    jump block45(v500, v501)\n" ++

  "\n" ++
  -- CONFLICT: backtrack using decision stack
  "block64(v510: i64, v904: i64):\n" ++  -- trail_depth, dec_depth
  -- If dec_depth == 0, UNSAT
  "    v511 = icmp eq v904, v3\n" ++
  "    brif v511, block65, block66\n" ++

  "\n" ++
  "block66:\n" ++
  -- Pop decision: dec_depth - 1
  "    v512 = isub v904, v23\n" ++                -- dec_depth - 1
  -- Load saved trail_depth from decStack[dec_depth-1]
  "    v5120 = imul v512, v290\n" ++               -- (dec_depth-1) * 8
  "    v5121 = iadd v800, v5120\n" ++
  "    v5122 = iadd v0, v5121\n" ++
  "    v5123 = load.i64 v5122\n" ++                -- saved_trail_depth (td at decision point)
  -- Undo all assignments from trail_depth-1 down to saved_trail_depth
  -- Start undo loop: i = trail_depth - 1
  "    v905 = isub v510, v23\n" ++                -- trail_depth - 1
  "    jump block83(v905, v5123, v512)\n" ++      -- undo_i, saved_td, new_dec_depth

  "\n" ++
  -- Undo loop: undo trail[undo_i], decrement, stop when undo_i < saved_td
  "block83(v5130: i64, v5131: i64, v5132: i64):\n" ++  -- undo_i, saved_td, new_dd
  "    v5133 = icmp slt v5130, v5131\n" ++         -- undo_i < saved_td?
  "    brif v5133, block84(v5131, v5132), block85\n" ++  -- done undoing → try flip

  "\n" ++
  "block85:\n" ++
  -- Undo assignment at trail[undo_i]
  s!"    v5134 = iconst.i64 {trailStart}\n" ++
  "    v5135 = imul v5130, v162\n" ++              -- undo_i * 4
  "    v5136 = iadd v5134, v5135\n" ++
  "    v5137 = iadd v0, v5136\n" ++
  "    v5138 = load.i32 v5137\n" ++                -- literal
  "    v5139 = sextend.i64 v5138\n" ++
  -- Get var_idx = abs(lit) - 1
  "    v5140 = icmp slt v5139, v3\n" ++
  "    v5141 = ineg v5139\n" ++
  "    v5142 = select v5140, v5141, v5139\n" ++    -- abs(lit)
  "    v5143 = isub v5142, v23\n" ++               -- var_idx
  -- Clear assignment
  "    v5144 = iadd v15, v5143\n" ++
  "    istore8 v3, v5144\n" ++
  -- Decrement undo_i
  "    v5145 = isub v5130, v23\n" ++
  "    jump block83(v5145, v5131, v5132)\n" ++

  "\n" ++
  -- Done undoing. Now look at the decision literal at trail[saved_td]
  "block84(v5150: i64, v5151: i64):\n" ++  -- saved_td (= trail_depth at decision), new_dd
  -- Load the decision literal
  s!"    v5152 = iconst.i64 {trailStart}\n" ++
  "    v5153 = imul v5150, v162\n" ++              -- saved_td * 4
  "    v5154 = iadd v5152, v5153\n" ++
  "    v5155 = iadd v0, v5154\n" ++
  "    v5156 = load.i32 v5155\n" ++                -- decision literal
  "    v5157 = sextend.i64 v5156\n" ++
  -- Get var_idx = abs(lit) - 1
  "    v5158 = icmp slt v5157, v3\n" ++
  "    v5159 = ineg v5157\n" ++
  "    v5160 = select v5158, v5159, v5157\n" ++    -- abs(lit)
  "    v5161 = isub v5160, v23\n" ++               -- var_idx
  -- Also undo this decision's assignment
  "    v5162 = iadd v15, v5161\n" ++
  "    istore8 v3, v5162\n" ++
  -- Was it positive (tried true)? If so, try false. If negative, backtrack further.
  "    v5163 = icmp sgt v5157, v3\n" ++
  "    brif v5163, block67(v5150, v5151, v5161), block64(v5150, v5151)\n" ++  -- try_false or backtrack more

  "\n" ++
  -- Try false: assign -(var_idx+1), put on trail at saved_td, propagate
  "block67(v530: i64, v531: i64, v5310: i64):\n" ++  -- saved_td, new_dd, var_idx
  -- Assign false
  "    v532 = iconst.i64 -1\n" ++
  "    v533 = iadd v15, v5310\n" ++
  "    istore8 v532, v533\n" ++
  -- Store negative literal on trail at saved_td
  "    v534 = iadd v5310, v23\n" ++                -- var_idx + 1
  "    v535 = ineg v534\n" ++                      -- -(var_idx + 1)
  s!"    v536 = iconst.i64 {trailStart}\n" ++
  "    v537 = imul v530, v162\n" ++                -- saved_td * 4
  "    v538 = iadd v536, v537\n" ++
  "    v539 = iadd v0, v538\n" ++
  "    v540 = ireduce.i32 v535\n" ++
  "    store v540, v539\n" ++
  "    v541 = iadd v530, v23\n" ++                 -- new trail_depth = saved_td + 1
  "    v542 = iadd v531, v23\n" ++                 -- new dec_depth = dd + 1
  -- Save decision: decStack[new_dd-1] = saved_td (already there, but re-store for clarity)
  "    jump block45(v541, v542)\n" ++

  "\n" ++
  -- ===================================================================
  -- SAT: all variables assigned, write result
  -- ===================================================================
  "block63(v550: i64):\n" ++
  -- Store result flag = 1 (SAT)
  "    v551 = iconst.i64 1\n" ++
  "    store v551, v13\n" ++                       -- v13 = &resultFlag_off
  "    jump block69\n" ++

  "\n" ++
  "block65:\n" ++
  -- Store result flag = 2 (UNSAT)
  "    v552 = iconst.i64 2\n" ++
  "    store v552, v13\n" ++
  "    jump block69\n" ++

  "\n" ++
  -- ===================================================================
  -- OUTPUT: write result string to file
  --
  -- Format:
  --   SAT:   "s SATISFIABLE\nv <lit1> <lit2> ... 0\n"
  --   UNSAT: "s UNSATISFIABLE\n"
  -- ===================================================================
  "block69:\n" ++
  "    v560 = load.i64 v13\n" ++                   -- result flag
  "    v561 = iconst.i64 1\n" ++
  "    v562 = icmp eq v560, v561\n" ++
  "    brif v562, block70, block71\n" ++

  "\n" ++
  "block71:\n" ++
  -- Write "s UNSATISFIABLE\n" to output region
  "    v571 = iadd v0, v610\n" ++
  -- 's' ' ' 'U' 'N' 'S' 'A' 'T' 'I' 'S' 'F' 'I' 'A' 'B' 'L' 'E' '\n'
  -- = 115 32 85 78 83 65 84 73 83 70 73 65 66 76 69 10
  "    v572 = iconst.i64 115\n" ++    -- 's'
  "    istore8 v572, v571\n" ++
  "    v573 = iconst.i64 1\n" ++
  "    v574 = iadd v571, v573\n" ++
  "    v575 = iconst.i64 32\n" ++     -- ' '
  "    istore8 v575, v574\n" ++
  "    v576 = iadd v574, v573\n" ++
  "    v577 = iconst.i64 85\n" ++     -- 'U'
  "    istore8 v577, v576\n" ++
  "    v578 = iadd v576, v573\n" ++
  "    v579 = iconst.i64 78\n" ++     -- 'N'
  "    istore8 v579, v578\n" ++
  "    v580 = iadd v578, v573\n" ++
  "    v581 = iconst.i64 83\n" ++     -- 'S'
  "    istore8 v581, v580\n" ++
  "    v582 = iadd v580, v573\n" ++
  "    v583 = iconst.i64 65\n" ++     -- 'A'
  "    istore8 v583, v582\n" ++
  "    v584 = iadd v582, v573\n" ++
  "    v585 = iconst.i64 84\n" ++     -- 'T'
  "    istore8 v585, v584\n" ++
  "    v586 = iadd v584, v573\n" ++
  "    v587 = iconst.i64 73\n" ++     -- 'I'
  "    istore8 v587, v586\n" ++
  "    v588 = iadd v586, v573\n" ++
  "    v589 = iconst.i64 83\n" ++     -- 'S'
  "    istore8 v589, v588\n" ++
  "    v590 = iadd v588, v573\n" ++
  "    v591 = iconst.i64 70\n" ++     -- 'F'
  "    istore8 v591, v590\n" ++
  "    v592 = iadd v590, v573\n" ++
  "    v593 = iconst.i64 73\n" ++     -- 'I'
  "    istore8 v593, v592\n" ++
  "    v594 = iadd v592, v573\n" ++
  "    v595 = iconst.i64 65\n" ++     -- 'A'
  "    istore8 v595, v594\n" ++
  "    v596 = iadd v594, v573\n" ++
  "    v597 = iconst.i64 66\n" ++     -- 'B'
  "    istore8 v597, v596\n" ++
  "    v598 = iadd v596, v573\n" ++
  "    v599 = iconst.i64 76\n" ++     -- 'L'
  "    istore8 v599, v598\n" ++
  "    v600 = iadd v598, v573\n" ++
  "    v601 = iconst.i64 69\n" ++     -- 'E'
  "    istore8 v601, v600\n" ++
  "    v602 = iadd v600, v573\n" ++
  "    v603 = iconst.i64 10\n" ++     -- '\n'
  "    istore8 v603, v602\n" ++
  -- Length = 16 bytes
  "    v604 = iconst.i64 16\n" ++
  "    jump block72(v604)\n" ++

  "\n" ++
  "block70:\n" ++
  -- Write "s SATISFIABLE\n"
  "    v611 = iadd v0, v610\n" ++
  -- 's' ' ' 'S' 'A' 'T' 'I' 'S' 'F' 'I' 'A' 'B' 'L' 'E' '\n'
  -- = 115 32 83 65 84 73 83 70 73 65 66 76 69 10
  "    v612 = iconst.i64 115\n" ++    -- 's'
  "    istore8 v612, v611\n" ++
  "    v613 = iconst.i64 1\n" ++
  "    v614 = iadd v611, v613\n" ++
  "    v615 = iconst.i64 32\n" ++     -- ' '
  "    istore8 v615, v614\n" ++
  "    v616 = iadd v614, v613\n" ++
  "    v617 = iconst.i64 83\n" ++     -- 'S'
  "    istore8 v617, v616\n" ++
  "    v618 = iadd v616, v613\n" ++
  "    v619 = iconst.i64 65\n" ++     -- 'A'
  "    istore8 v619, v618\n" ++
  "    v620 = iadd v618, v613\n" ++
  "    v621 = iconst.i64 84\n" ++     -- 'T'
  "    istore8 v621, v620\n" ++
  "    v622 = iadd v620, v613\n" ++
  "    v623 = iconst.i64 73\n" ++     -- 'I'
  "    istore8 v623, v622\n" ++
  "    v624 = iadd v622, v613\n" ++
  "    v625 = iconst.i64 83\n" ++     -- 'S'
  "    istore8 v625, v624\n" ++
  "    v626 = iadd v624, v613\n" ++
  "    v627 = iconst.i64 70\n" ++     -- 'F'
  "    istore8 v627, v626\n" ++
  "    v628 = iadd v626, v613\n" ++
  "    v629 = iconst.i64 73\n" ++     -- 'I'
  "    istore8 v629, v628\n" ++
  "    v630 = iadd v628, v613\n" ++
  "    v631 = iconst.i64 65\n" ++     -- 'A'
  "    istore8 v631, v630\n" ++
  "    v632 = iadd v630, v613\n" ++
  "    v633 = iconst.i64 66\n" ++     -- 'B'
  "    istore8 v633, v632\n" ++
  "    v634 = iadd v632, v613\n" ++
  "    v635 = iconst.i64 76\n" ++     -- 'L'
  "    istore8 v635, v634\n" ++
  "    v636 = iadd v634, v613\n" ++
  "    v637 = iconst.i64 69\n" ++     -- 'E'
  "    istore8 v637, v636\n" ++
  "    v638 = iadd v636, v613\n" ++
  "    v639 = iconst.i64 10\n" ++     -- '\n'
  "    istore8 v639, v638\n" ++
  -- Now write "v " prefix (14 bytes for header, then assignments)
  "    v640 = iadd v638, v613\n" ++
  "    v641 = iconst.i64 118\n" ++    -- 'v'
  "    istore8 v641, v640\n" ++
  "    v642 = iadd v640, v613\n" ++
  "    istore8 v615, v642\n" ++      -- ' '
  -- Write each variable assignment as signed literal
  -- out_ptr = offset 16 (after "s SATISFIABLE\nv ")
  "    v643 = iconst.i64 16\n" ++
  "    v644 = load.i64 v7\n" ++      -- num_vars
  "    jump block73(v643, v3, v644)\n" ++   -- out_ptr_offset, var_i, num_vars

  "\n" ++
  -- Write variable assignment as literal string
  "block73(v650: i64, v651: i64, v652: i64):\n" ++  -- out_offset, var_i, num_vars
  "    v653 = icmp uge v651, v652\n" ++
  "    brif v653, block75(v650), block74\n" ++

  "\n" ++
  "block74:\n" ++
  -- Load assignment
  "    v654 = iadd v15, v651\n" ++
  "    v656 = sload8.i64 v654\n" ++
  -- literal = if assign > 0 then (var_i+1) else -(var_i+1)
  "    v657 = iadd v651, v23\n" ++                 -- var_i + 1
  "    v658 = ineg v657\n" ++                      -- -(var_i + 1)
  "    v659 = icmp sgt v656, v3\n" ++
  "    v660 = select v659, v657, v658\n" ++        -- literal value
  -- Convert integer to string and write to output
  -- Simple: write sign if negative, then digits
  "    v661 = icmp slt v660, v3\n" ++
  "    brif v661, block76(v650, v651, v652, v660), block77(v650, v651, v652, v660)\n" ++

  "\n" ++
  "block76(v670: i64, v671: i64, v672: i64, v673: i64):\n" ++
  -- Write '-'
  "    v674 = iadd v610, v670\n" ++                -- out_off + offset
  "    v675 = iadd v0, v674\n" ++
  "    v676 = iconst.i64 45\n" ++                   -- '-'
  "    istore8 v676, v675\n" ++
  "    v677 = iadd v670, v23\n" ++                 -- offset + 1
  "    v678 = ineg v673\n" ++                      -- abs value
  "    jump block77(v677, v671, v672, v678)\n" ++

  "\n" ++
  -- Write digits of positive integer, then space
  -- Simple approach: divide by powers of 10, skip leading zeros
  -- For var indices up to 10000, max 5 digits
  "block77(v680: i64, v681: i64, v682: i64, v683: i64):\n" ++
  -- Write digits: find first non-zero, then write all
  -- We'll use a simple repeated division approach
  "    v684 = iconst.i64 10000\n" ++
  "    jump block78(v680, v681, v682, v683, v684, v3)\n" ++  -- divisor=10000, started=0

  "\n" ++
  "block78(v690: i64, v691: i64, v692: i64, v693: i64, v694: i64, v695: i64):\n" ++
  -- v690=out_off, v691=var_i, v692=num_vars, v693=remaining_value, v694=divisor, v695=started
  "    v696 = icmp eq v694, v3\n" ++               -- divisor == 0?
  "    brif v696, block80(v690, v691, v692), block79\n" ++

  "\n" ++
  "block79:\n" ++
  "    v697 = udiv v693, v694\n" ++                -- digit = value / divisor
  "    v698 = imul v697, v694\n" ++
  "    v699 = isub v693, v698\n" ++                -- remainder = value - digit * divisor
  -- Should we write this digit? yes if started or digit > 0 or divisor == 1
  "    v700 = icmp ne v697, v3\n" ++               -- digit != 0
  "    v7000 = uextend.i64 v700\n" ++
  "    v701 = bor v695, v7000\n" ++               -- started or digit!=0
  "    v702 = icmp eq v694, v23\n" ++             -- divisor == 1 (must write last digit)
  "    v7020 = uextend.i64 v702\n" ++
  "    v703 = bor v701, v7020\n" ++               -- should_write
  "    brif v703, block81(v690, v691, v692, v697, v699, v694), block82(v690, v691, v692, v699, v694)\n" ++

  "\n" ++
  "block81(v710: i64, v711: i64, v712: i64, v713: i64, v714: i64, v715: i64):\n" ++
  -- Write ASCII digit
  "    v716 = iadd v713, v80\n" ++                 -- digit + '0'
  "    v718 = iadd v610, v710\n" ++                -- out_off + offset
  "    v719 = iadd v0, v718\n" ++
  "    istore8 v716, v719\n" ++
  "    v720 = iadd v710, v23\n" ++                 -- offset + 1
  -- Next divisor = divisor / 10
  "    v721 = iconst.i64 10\n" ++
  "    v722 = udiv v715, v721\n" ++
  "    jump block78(v720, v711, v712, v714, v722, v23)\n" ++  -- started=1

  "\n" ++
  "block82(v730: i64, v731: i64, v732: i64, v733: i64, v734: i64):\n" ++
  "    v735 = iconst.i64 10\n" ++
  "    v736 = udiv v734, v735\n" ++                -- next divisor
  "    jump block78(v730, v731, v732, v733, v736, v3)\n" ++  -- still not started

  "\n" ++
  "block80(v740: i64, v741: i64, v742: i64):\n" ++
  -- Write space separator
  "    v743 = iadd v610, v740\n" ++
  "    v744 = iadd v0, v743\n" ++
  "    v745 = iconst.i64 32\n" ++                   -- ' '
  "    istore8 v745, v744\n" ++
  "    v746 = iadd v740, v23\n" ++                 -- offset + 1
  -- Next variable
  "    v747 = iadd v741, v23\n" ++
  "    jump block73(v746, v747, v742)\n" ++

  "\n" ++
  -- Finish: write "0\n" and the output
  "block75(v750: i64):\n" ++
  -- Write "0\n"
  "    v751 = iadd v610, v750\n" ++
  "    v752 = iadd v0, v751\n" ++
  "    v753 = iconst.i64 48\n" ++                   -- '0'
  "    istore8 v753, v752\n" ++
  "    v754 = iadd v750, v23\n" ++
  "    v755 = iadd v610, v754\n" ++
  "    v756 = iadd v0, v755\n" ++
  "    v757 = iconst.i64 10\n" ++                   -- '\n'
  "    istore8 v757, v756\n" ++
  "    v758 = iadd v754, v23\n" ++                 -- total output length
  "    jump block72(v758)\n" ++

  "\n" ++
  -- Write output to file
  "block72(v760: i64):\n" ++            -- output_length
  s!"    v761 = iconst.i64 {outFname}\n" ++
  s!"    v762 = iconst.i64 {outStart}\n" ++
  "    v763 = call fn1(v0, v761, v762, v3, v760)\n" ++
  "    return\n" ++
  "}\n"

-- ---------------------------------------------------------------------------
-- Payload construction
-- ---------------------------------------------------------------------------

def payloads : List UInt8 :=
  let reserved := zeros inputFilename_off
  -- Input filename placeholder (will be patched at runtime)
  let inputFname := padTo (stringToBytes "input.cnf") (outputFilename_off - inputFilename_off)
  -- Output filename
  let outputFname := padTo (stringToBytes "sat_output.txt") (cnf_off - outputFilename_off)
  reserved ++ inputFname ++ outputFname

-- ---------------------------------------------------------------------------
-- Configuration
-- ---------------------------------------------------------------------------

def satConfig : BaseConfig := {
  cranelift_ir := clifIrSource,
  memory_size := totalMemory,
  context_offset := 0
}

def satAlgorithm : Algorithm :=
  let clifCallAction : Action :=
    { kind := .ClifCall, dst := u32 0, src := u32 1, offset := u32 0, size := u32 0 }
  {
    actions := [clifCallAction],
    payloads := payloads,
    cranelift_units := 0,
    timeout_ms := some 300000
  }

end Algorithm

def main : IO Unit := do
  let json := toJsonPair Algorithm.satConfig Algorithm.satAlgorithm
  IO.println (Json.compress json)
