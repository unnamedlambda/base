import Lean
import Std
import AlgorithmLib

open Lean
open AlgorithmLib

namespace LeanEval

-- Payload layout
def OUTPUT_PATH    : Nat := 0x0020
def INPUT_PATH     : Nat := 0x0060
def SOURCE_BUF     : Nat := 0x0160
def SOURCE_BUF_SZ  : Nat := 4096
def TRUE_STR       : Nat := 0x1160
def FALSE_STR      : Nat := 0x1168
def IDENT_BUF      : Nat := 0x1170
def IDENT_BUF_SZ   : Nat := 64
def HT_VAL_BUF     : Nat := 0x11B0
def OUTPUT_BUF     : Nat := 0x11B8
def OUTPUT_BUF_SZ  : Nat := 64
def STACK_BASE     : Nat := 0x11F8
def STACK_SZ       : Nat := 512
def CLIF_IR_OFF    : Nat := 0x13F8

def TIMEOUT_MS : Nat := 30000

/-
  CLIF function u0:0 — file read: read source file into SOURCE_BUF
-/
def clifReadFn : String :=
  "function u0:0(i64) system_v {\n" ++
  "  sig0 = (i64, i64, i64, i64, i64) -> i64 system_v\n" ++
  "  fn0 = %cl_file_read sig0\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = iconst.i64 " ++ toString INPUT_PATH ++ "\n" ++
  "  v2 = iconst.i64 " ++ toString SOURCE_BUF ++ "\n" ++
  "  v3 = iconst.i64 0\n" ++
  "  v4 = iconst.i64 0\n" ++
  "  v5 = call fn0(v0, v1, v2, v3, v4)\n" ++
  "  return\n" ++
  "}\n"

/-
  CLIF function u0:2 — file write: write OUTPUT_BUF to output file
-/
def clifWriteFn : String :=
  "function u0:2(i64) system_v {\n" ++
  "  sig0 = (i64, i64, i64, i64, i64) -> i64 system_v\n" ++
  "  fn0 = %cl_file_write sig0\n" ++
  "block0(v0: i64):\n" ++
  "  v1 = iconst.i64 " ++ toString OUTPUT_PATH ++ "\n" ++
  "  v2 = iconst.i64 " ++ toString OUTPUT_BUF ++ "\n" ++
  "  v3 = iconst.i64 0\n" ++
  "  v4 = iconst.i64 0\n" ++
  "  v5 = call fn0(v0, v1, v2, v3, v4)\n" ++
  "  return\n" ++
  "}\n"

/-
  CLIF function u0:1 — compute: parse #eval expression and write result to OUTPUT_BUF

  Recursive descent parser as a state machine with CLIF basic blocks.
  All "recursion" is handled via a stack region in shared memory.

  Block map:
    block0:  setup constants, create HT, skip "#eval "
    block1:  skip_spaces loop head
    block2:  skip_spaces check byte

    block3:  parse_expr entry — call parse_term for first operand
    block4:  parse_expr operator check (after getting a term)
    block5:  parse_expr_add — skip '+', space, call parse_term, add
    block6:  parse_expr_sub — skip '-', space, call parse_term, sub
    block7:  parse_expr_lt — skip '<', space, parse comparand
    block8:  parse_expr_gt — skip '>', space, parse comparand

    block10: parse_term entry — call parse_atom for first factor
    block11: parse_term operator check (after getting an atom)
    block12: parse_term_mul — skip '*', space, call parse_atom, mul

    block15: parse_atom dispatch
    block16: parse_number loop
    block17: parse_number digit accumulate
    block18: variable — read identifier loop
    block19: variable — do ht_lookup

    block20: paren_or_lambda — check for "(fun"
    block21: paren_expr — skip '(', parse_expr, skip ')'

    block25: let_binding — skip "let ", read var name
    block26: let_read_name loop
    block27: let_skip_assign — skip ":= "
    block28: let_eval_binding — push state, parse_expr for value
    block29: let_store — ht_insert, skip "; ", parse body

    block30: if_then_else entry — skip "if "
    block31: if_eval_cond — parse condition
    block32: if_skip_then — skip "then "
    block33: if_eval_then — parse then branch
    block34: if_skip_else — skip "else "
    block35: if_eval_else — parse else branch
    block36: if_select — pick result

    block40: lambda — skip "fun ", read param
    block41: lambda_read_param loop
    block42: lambda_skip_arrow — skip "=> "
    block43: lambda_skip_body — scan to closing ')'
    block44: lambda_parse_arg — parse argument after ') '
    block45: lambda_bind_and_eval — ht_insert, eval body

    block50: output — check is_bool
    block51: output_true — write "true"
    block52: output_false — write "false"
    block53: itoa_find_div — find highest power of 10
    block54: itoa_write_digit — extract digits
    block55: itoa_done — null terminate

    block60: return_from_subexpr — pop stack, dispatch to caller
    block61: cmp_lt_result
    block62: cmp_gt_result
    block63: bool_true_literal
    block64: bool_false_literal

  Stack frame layout (24 bytes each):
    [0..8)   return_tag (0=expr_op, 1=term_op, 2=let_bind, 3=let_body,
             4=if_cond, 5=if_then, 6=if_else, 7=paren, 8=lambda_arg,
             9=lambda_body, 10=cmp_rhs)
    [8..16)  saved_value (accumulator/left operand)
    [16..24) saved_extra (secondary saved value)
-/
def clifComputeFn : String :=
  "function u0:1(i64) system_v {\n" ++
  -- signatures
  "  sig0 = (i64) -> i32 system_v\n" ++                       -- ht_create
  "  sig1 = (i64, i64, i32, i64, i32) system_v\n" ++          -- ht_insert
  "  sig2 = (i64, i64, i32, i64) -> i32 system_v\n" ++        -- ht_lookup
  "  fn0 = colocated %ht_create sig0\n" ++
  "  fn1 = colocated %ht_insert sig1\n" ++
  "  fn2 = colocated %ht_lookup sig2\n" ++
  "\n" ++

  -- =====================================================================
  -- block0: setup
  -- =====================================================================
  "block0(v0: i64):\n" ++
  "  v1 = load.i64 v0\n" ++                                   -- HT ctx ptr
  "  v2 = call fn0(v1)\n" ++                                   -- ht_create
  "  v3 = iconst.i64 " ++ toString SOURCE_BUF ++ "\n" ++      -- source base offset
  "  v4 = iconst.i64 6\n" ++                                   -- skip "#eval "
  "  v5 = iconst.i64 0\n" ++                                   -- zero const
  "  v6 = iconst.i64 " ++ toString IDENT_BUF ++ "\n" ++
  "  v7 = iconst.i64 " ++ toString HT_VAL_BUF ++ "\n" ++
  "  v8 = iconst.i64 " ++ toString OUTPUT_BUF ++ "\n" ++
  "  v9 = iconst.i64 " ++ toString STACK_BASE ++ "\n" ++
  -- jump to skip_spaces, then parse_expr
  -- args: (pos=6, sp=0, ctx=v1, base=v0)
  -- We'll use block1 as skip_spaces entry, which then jumps to parse_expr
  -- But we need a way to "call" parse_expr. Use the stack to push a return tag.
  -- For top-level: tag=255 means "done, go to output"
  -- Push frame: tag=255, val=0, extra=0
  "  v10 = iadd v0, v9\n" ++                                   -- stack base addr
  "  v11 = iconst.i64 255\n" ++                                -- done tag
  "  store.i64 v11, v10\n" ++                                  -- frame[0] = tag
  "  store.i64 v5, v10+8\n" ++                                 -- frame[8] = 0
  "  store.i64 v5, v10+16\n" ++                                -- frame[16] = 0
  "  v12 = iconst.i64 24\n" ++                                 -- frame size
  -- sp = 24 (one frame pushed)
  "  jump block1(v4, v12, v1, v5)\n" ++
  "\n" ++

  -- =====================================================================
  -- block1: skip_spaces head (pos, sp, ctx, is_bool)
  -- After skipping spaces, jump to parse_expr (block3)
  -- =====================================================================
  "block1(v20: i64, v21: i64, v22: i64, v23: i64):\n" ++
  "  v24 = iadd v0, v3\n" ++                                   -- source base addr
  "  v25 = iadd v24, v20\n" ++                                 -- &src[pos]
  "  v26 = uload8.i64 v25\n" ++                                -- ch = src[pos]
  "  v27 = icmp eq v26, v5\n" ++                                -- ch == 0? (end)
  "  brif v27, block3(v20, v21, v22, v23), block2(v20, v21, v22, v23, v26)\n" ++
  "\n" ++

  -- =====================================================================
  -- block2: skip_spaces check (pos, sp, ctx, is_bool, ch)
  -- =====================================================================
  "block2(v30: i64, v31: i64, v32: i64, v33: i64, v34: i64):\n" ++
  "  v35 = iconst.i64 32\n" ++                                 -- space
  "  v36 = icmp eq v34, v35\n" ++                               -- ch == ' '?
  "  v37 = iconst.i64 10\n" ++                                  -- newline
  "  v38 = icmp eq v34, v37\n" ++                               -- ch == '\n'?
  "  v39 = bor v36, v38\n" ++
  "  v40 = iadd_imm v30, 1\n" ++                                -- pos+1
  "  brif v39, block1(v40, v31, v32, v33), block3(v30, v31, v32, v33)\n" ++
  "\n" ++

  -- =====================================================================
  -- block3: parse_expr entry (pos, sp, ctx, is_bool)
  -- Push return tag=0 (expr_op), then dispatch to parse_term
  -- =====================================================================
  "block3(v50: i64, v51: i64, v52: i64, v53: i64):\n" ++
  -- Push frame: tag=0 (expr_op check), val=0, extra=is_bool
  "  v54 = iadd v0, v9\n" ++                                   -- stack base
  "  v55 = iadd v54, v51\n" ++                                 -- stack[sp]
  "  store.i64 v5, v55\n" ++                                   -- tag=0
  "  store.i64 v5, v55+8\n" ++                                 -- val=0
  "  store.i64 v53, v55+16\n" ++                               -- extra=is_bool
  "  v56 = iadd_imm v51, 24\n" ++                              -- sp += 24
  -- Now dispatch to parse_term (block10), which will push its own frame
  "  jump block10(v50, v56, v52)\n" ++
  "\n" ++

  -- =====================================================================
  -- block4: parse_expr operator check (pos, sp, ctx, value, is_bool)
  -- Called when parse_term returns. Skip spaces, then check for +, -, <, >
  -- =====================================================================
  "block4(v60: i64, v61: i64, v62: i64, v63: i64, v64: i64):\n" ++
  "  v65 = iadd v0, v3\n" ++                                   -- source base
  "  v66 = iadd v65, v60\n" ++
  "  v67 = uload8.i64 v66\n" ++                                -- ch at pos
  "  v6000 = iconst.i64 32\n" ++
  "  v6001 = icmp eq v67, v6000\n" ++
  "  v6003 = iconst.i64 10\n" ++
  "  v6004 = icmp eq v67, v6003\n" ++
  "  v6005 = bor v6001, v6004\n" ++
  "  v6002 = iadd_imm v60, 1\n" ++
  "  brif v6005, block4(v6002, v61, v62, v63, v64), block70(v60, v61, v62, v63, v64, v67)\n" ++
  "\n" ++

  -- block70: check '+'
  "block70(v600: i64, v601: i64, v602: i64, v603: i64, v604: i64, v607: i64):\n" ++
  "  v68 = iconst.i64 43\n" ++                                 -- '+'
  "  v69 = icmp eq v607, v68\n" ++
  "  brif v69, block5(v600, v601, v602, v603), block71(v600, v601, v602, v603, v604, v607)\n" ++
  "\n" ++

  -- block71: check '-'
  "block71(v610: i64, v611: i64, v612: i64, v613: i64, v614: i64, v617: i64):\n" ++
  "  v608 = iconst.i64 45\n" ++                                -- '-'
  "  v609 = icmp eq v617, v608\n" ++
  "  brif v609, block6(v610, v611, v612, v613), block72(v610, v611, v612, v613, v614, v617)\n" ++
  "\n" ++

  -- block72: check '<'
  "block72(v620: i64, v621: i64, v622: i64, v623: i64, v624: i64, v627: i64):\n" ++
  "  v618 = iconst.i64 60\n" ++                                -- '<'
  "  v619 = icmp eq v627, v618\n" ++
  "  brif v619, block7(v620, v621, v622, v623), block109(v620, v621, v622, v623, v624, v627)\n" ++
  "\n" ++

  -- block109: check '>'
  "block109(v6200: i64, v6201: i64, v6202: i64, v6203: i64, v6204: i64, v6207: i64):\n" ++
  "  v628 = iconst.i64 62\n" ++                                -- '>'
  "  v629 = icmp eq v6207, v628\n" ++
  "  brif v629, block8(v6200, v6201, v6202, v6203), block60(v6200, v6201, v6202, v6203, v6204)\n" ++
  "\n" ++

  -- =====================================================================
  -- block5: parse_expr add (pos, sp, ctx, left_value)
  -- Skip '+', push frame tag=0, saved_value=left, call parse_term
  -- =====================================================================
  "block5(v70: i64, v71: i64, v72: i64, v73: i64):\n" ++
  "  v74 = iadd_imm v70, 1\n" ++                               -- skip '+'
  -- skip space after '+'
  "  v75 = iadd v0, v3\n" ++
  "  v76 = iadd v75, v74\n" ++
  "  v77 = uload8.i64 v76\n" ++
  "  v78 = iconst.i64 32\n" ++
  "  v79 = icmp eq v77, v78\n" ++
  "  v80 = iadd_imm v74, 1\n" ++
  "  v81 = select v79, v80, v74\n" ++
  -- Push frame: tag=0 (expr_add), val=left_value, extra=0 (add op)
  "  v82 = iadd v0, v9\n" ++
  "  v83 = iadd v82, v71\n" ++
  "  store.i64 v5, v83\n" ++                                   -- tag=0 (expr_op)
  "  v84 = iconst.i64 1\n" ++                                  -- extra=1 means add
  "  store.i64 v73, v83+8\n" ++                                -- saved left
  "  store.i64 v84, v83+16\n" ++                               -- op=add
  "  v85 = iadd_imm v71, 24\n" ++
  "  jump block10(v81, v85, v72)\n" ++
  "\n" ++

  -- =====================================================================
  -- block6: parse_expr sub (pos, sp, ctx, left_value)
  -- =====================================================================
  "block6(v90: i64, v91: i64, v92: i64, v93: i64):\n" ++
  "  v94 = iadd_imm v90, 1\n" ++                               -- skip '-'
  "  v95 = iadd v0, v3\n" ++
  "  v96 = iadd v95, v94\n" ++
  "  v97 = uload8.i64 v96\n" ++
  "  v98 = iconst.i64 32\n" ++
  "  v99 = icmp eq v97, v98\n" ++
  "  v100 = iadd_imm v94, 1\n" ++
  "  v101 = select v99, v100, v94\n" ++
  -- Push frame: tag=0, val=left, extra=2 (sub op)
  "  v102 = iadd v0, v9\n" ++
  "  v103 = iadd v102, v91\n" ++
  "  store.i64 v5, v103\n" ++
  "  v104 = iconst.i64 2\n" ++
  "  store.i64 v93, v103+8\n" ++
  "  store.i64 v104, v103+16\n" ++
  "  v105 = iadd_imm v91, 24\n" ++
  "  jump block10(v101, v105, v92)\n" ++
  "\n" ++

  -- =====================================================================
  -- block7: parse_expr '<' comparison (pos, sp, ctx, left_value)
  -- =====================================================================
  "block7(v110: i64, v111: i64, v112: i64, v113: i64):\n" ++
  "  v114 = iadd_imm v110, 1\n" ++                             -- skip '<'
  -- check for '<='
  "  v115 = iadd v0, v3\n" ++
  "  v116 = iadd v115, v114\n" ++
  "  v117 = uload8.i64 v116\n" ++
  "  v118 = iconst.i64 61\n" ++                                -- '='
  "  v119 = icmp eq v117, v118\n" ++
  "  v120 = iadd_imm v114, 1\n" ++
  "  v121 = select v119, v120, v114\n" ++                      -- skip '=' if present
  -- skip space
  "  v122 = iadd v115, v121\n" ++
  "  v123 = uload8.i64 v122\n" ++
  "  v124 = iconst.i64 32\n" ++
  "  v125 = icmp eq v123, v124\n" ++
  "  v126 = iadd_imm v121, 1\n" ++
  "  v127 = select v125, v126, v121\n" ++
  -- Push frame: tag=10 (cmp_rhs), val=left, extra=op (0=lt, 1=le)
  "  v128 = iadd v0, v9\n" ++
  "  v129 = iadd v128, v111\n" ++
  "  v130 = iconst.i64 10\n" ++
  "  store.i64 v130, v129\n" ++                                -- tag=10
  "  store.i64 v113, v129+8\n" ++                              -- left val
  -- extra: 0 if '<', 1 if '<='
  "  v131 = iconst.i64 0\n" ++
  "  v132 = iconst.i64 1\n" ++
  "  v133 = select v119, v132, v131\n" ++
  "  store.i64 v133, v129+16\n" ++
  "  v134 = iadd_imm v111, 24\n" ++
  "  jump block10(v127, v134, v112)\n" ++
  "\n" ++

  -- =====================================================================
  -- block8: parse_expr '>' comparison (pos, sp, ctx, left_value)
  -- =====================================================================
  "block8(v140: i64, v141: i64, v142: i64, v143: i64):\n" ++
  "  v144 = iadd_imm v140, 1\n" ++                             -- skip '>'
  -- check for '>='
  "  v145 = iadd v0, v3\n" ++
  "  v146 = iadd v145, v144\n" ++
  "  v147 = uload8.i64 v146\n" ++
  "  v148 = iconst.i64 61\n" ++
  "  v149 = icmp eq v147, v148\n" ++
  "  v150 = iadd_imm v144, 1\n" ++
  "  v151 = select v149, v150, v144\n" ++
  -- skip space
  "  v152 = iadd v145, v151\n" ++
  "  v153 = uload8.i64 v152\n" ++
  "  v154 = iconst.i64 32\n" ++
  "  v155 = icmp eq v153, v154\n" ++
  "  v156 = iadd_imm v151, 1\n" ++
  "  v157 = select v155, v156, v151\n" ++
  -- Push frame: tag=10, val=left, extra=op (2=gt, 3=ge)
  "  v158 = iadd v0, v9\n" ++
  "  v159 = iadd v158, v141\n" ++
  "  v160 = iconst.i64 10\n" ++
  "  store.i64 v160, v159\n" ++
  "  store.i64 v143, v159+8\n" ++
  "  v161 = iconst.i64 2\n" ++
  "  v162 = iconst.i64 3\n" ++
  "  v163 = select v149, v162, v161\n" ++
  "  store.i64 v163, v159+16\n" ++
  "  v164 = iadd_imm v141, 24\n" ++
  "  jump block10(v157, v164, v142)\n" ++
  "\n" ++

  -- =====================================================================
  -- block10: parse_term entry (pos, sp, ctx)
  -- Push frame tag=1 (term_op), call parse_atom
  -- =====================================================================
  "block10(v170: i64, v171: i64, v172: i64):\n" ++
  "  v173 = iadd v0, v9\n" ++
  "  v174 = iadd v173, v171\n" ++
  "  v175 = iconst.i64 1\n" ++                                 -- tag=1 (term_op)
  "  store.i64 v175, v174\n" ++
  "  store.i64 v5, v174+8\n" ++                                -- val=0
  "  store.i64 v5, v174+16\n" ++                               -- extra=0
  "  v176 = iadd_imm v171, 24\n" ++
  "  jump block15(v170, v176, v172)\n" ++
  "\n" ++

  -- =====================================================================
  -- block11: parse_term operator check (pos, sp, ctx, value)
  -- Called when parse_atom returns. Skip spaces, check for '*'
  -- =====================================================================
  "block11(v180: i64, v181: i64, v182: i64, v183: i64, v1806: i64):\n" ++
  "  v184 = iadd v0, v3\n" ++
  "  v185 = iadd v184, v180\n" ++
  "  v186 = uload8.i64 v185\n" ++
  "  v1800 = iconst.i64 32\n" ++
  "  v1801 = icmp eq v186, v1800\n" ++
  "  v1803 = iconst.i64 10\n" ++
  "  v1804 = icmp eq v186, v1803\n" ++
  "  v1805 = bor v1801, v1804\n" ++
  "  v1802 = iadd_imm v180, 1\n" ++
  "  brif v1805, block11(v1802, v181, v182, v183, v1806), block110(v180, v181, v182, v183, v186, v1806)\n" ++
  "\n" ++

  -- block110: actual '*' check (pos, sp, ctx, value, ch, is_bool)
  "block110(v1810: i64, v1811: i64, v1812: i64, v1813: i64, v1814: i64, v1815: i64):\n" ++
  "  v187 = iconst.i64 42\n" ++                                -- '*'
  "  v188 = icmp eq v1814, v187\n" ++
  "  brif v188, block12(v1810, v1811, v1812, v1813), block60(v1810, v1811, v1812, v1813, v1815)\n" ++
  "\n" ++

  -- =====================================================================
  -- block12: parse_term mul (pos, sp, ctx, left_value)
  -- =====================================================================
  "block12(v190: i64, v191: i64, v192: i64, v193: i64):\n" ++
  "  v194 = iadd_imm v190, 1\n" ++                             -- skip '*'
  "  v195 = iadd v0, v3\n" ++
  "  v196 = iadd v195, v194\n" ++
  "  v197 = uload8.i64 v196\n" ++
  "  v198 = iconst.i64 32\n" ++
  "  v199 = icmp eq v197, v198\n" ++
  "  v200 = iadd_imm v194, 1\n" ++
  "  v201 = select v199, v200, v194\n" ++
  -- Push frame: tag=1 (term_op), val=left, extra=1 (mul)
  "  v202 = iadd v0, v9\n" ++
  "  v203 = iadd v202, v191\n" ++
  "  v204 = iconst.i64 1\n" ++
  "  store.i64 v204, v203\n" ++
  "  v205 = iconst.i64 1\n" ++
  "  store.i64 v193, v203+8\n" ++
  "  store.i64 v205, v203+16\n" ++
  "  v206 = iadd_imm v191, 24\n" ++
  "  jump block15(v201, v206, v192)\n" ++
  "\n" ++

  -- =====================================================================
  -- block15: parse_atom dispatch (pos, sp, ctx) — skip spaces first
  -- =====================================================================
  "block15(v210: i64, v211: i64, v212: i64):\n" ++
  "  v213 = iadd v0, v3\n" ++
  "  v214 = iadd v213, v210\n" ++
  "  v215 = uload8.i64 v214\n" ++                              -- ch
  -- Skip spaces and newlines
  "  v2100 = iconst.i64 32\n" ++
  "  v2101 = icmp eq v215, v2100\n" ++
  "  v2102 = iconst.i64 10\n" ++
  "  v2103 = icmp eq v215, v2102\n" ++
  "  v2104 = bor v2101, v2103\n" ++
  "  v2105 = iadd_imm v210, 1\n" ++
  "  brif v2104, block15(v2105, v211, v212), block111(v210, v211, v212, v215)\n" ++
  "\n" ++

  -- block111: actual dispatch after space skip
  "block111(v2110: i64, v2111: i64, v2112: i64, v2115: i64):\n" ++
  -- Check digit: '0'=48 .. '9'=57
  "  v216 = iconst.i64 48\n" ++
  "  v217 = iconst.i64 57\n" ++
  "  v218 = icmp uge v2115, v216\n" ++
  "  v219 = icmp ule v2115, v217\n" ++
  "  v220 = band v218, v219\n" ++
  "  brif v220, block16(v2110, v2111, v2112, v5), block73(v2110, v2111, v2112, v2115)\n" ++
  "\n" ++

  -- block73: check '(' for paren/lambda
  "block73(v221: i64, v222: i64, v223: i64, v224: i64):\n" ++
  "  v225 = iconst.i64 40\n" ++                                -- '('
  "  v226 = icmp eq v224, v225\n" ++
  "  brif v226, block20(v221, v222, v223), block74(v221, v222, v223, v224)\n" ++
  "\n" ++

  -- block74: check 'l' for "let " (full keyword match)
  "block74(v230: i64, v231: i64, v232: i64, v233: i64):\n" ++
  "  v234 = iconst.i64 108\n" ++                               -- 'l'
  "  v235 = icmp eq v233, v234\n" ++
  "  brif v235, block112(v230, v231, v232), block75(v230, v231, v232, v233)\n" ++
  "\n" ++

  -- block112: verify "et " after 'l'
  "block112(v2300: i64, v2301: i64, v2302: i64):\n" ++
  "  v2303 = iadd v0, v3\n" ++
  "  v2304 = iadd_imm v2300, 1\n" ++                           -- pos+1
  "  v2305 = iadd v2303, v2304\n" ++
  "  v2306 = uload8.i64 v2305\n" ++                            -- src[pos+1]
  "  v2307 = iconst.i64 101\n" ++                               -- 'e'
  "  v2308 = icmp eq v2306, v2307\n" ++
  "  v2309 = iadd_imm v2300, 2\n" ++
  "  v2310 = iadd v2303, v2309\n" ++
  "  v2311 = uload8.i64 v2310\n" ++                            -- src[pos+2]
  "  v2312 = iconst.i64 116\n" ++                               -- 't'
  "  v2313 = icmp eq v2311, v2312\n" ++
  "  v2314 = iadd_imm v2300, 3\n" ++
  "  v2315 = iadd v2303, v2314\n" ++
  "  v2316 = uload8.i64 v2315\n" ++                            -- src[pos+3]
  "  v2317 = iconst.i64 32\n" ++                                -- ' '
  "  v2318 = icmp eq v2316, v2317\n" ++
  "  v2319 = band v2308, v2313\n" ++
  "  v2320 = band v2319, v2318\n" ++
  "  brif v2320, block25(v2300, v2301, v2302), block18(v2300, v2301, v2302, v5)\n" ++
  "\n" ++

  -- block75: check 'i' for "if " (full keyword match)
  "block75(v240: i64, v241: i64, v242: i64, v243: i64):\n" ++
  "  v244 = iconst.i64 105\n" ++                               -- 'i'
  "  v245 = icmp eq v243, v244\n" ++
  "  brif v245, block113(v240, v241, v242), block76(v240, v241, v242, v243)\n" ++
  "\n" ++

  -- block113: verify "f " after 'i'
  "block113(v2400: i64, v2401: i64, v2402: i64):\n" ++
  "  v2403 = iadd v0, v3\n" ++
  "  v2404 = iadd_imm v2400, 1\n" ++
  "  v2405 = iadd v2403, v2404\n" ++
  "  v2406 = uload8.i64 v2405\n" ++                            -- src[pos+1]
  "  v2407 = iconst.i64 102\n" ++                               -- 'f'
  "  v2408 = icmp eq v2406, v2407\n" ++
  "  v2409 = iadd_imm v2400, 2\n" ++
  "  v2410 = iadd v2403, v2409\n" ++
  "  v2411 = uload8.i64 v2410\n" ++                            -- src[pos+2]
  "  v2412 = iconst.i64 32\n" ++                                -- ' '
  "  v2413 = icmp eq v2411, v2412\n" ++
  "  v2414 = band v2408, v2413\n" ++
  "  brif v2414, block30(v2400, v2401, v2402), block18(v2400, v2401, v2402, v5)\n" ++
  "\n" ++

  -- block76: check 't' for "true" (full keyword, followed by non-alpha)
  "block76(v250: i64, v251: i64, v252: i64, v253: i64):\n" ++
  "  v254 = iconst.i64 116\n" ++                               -- 't'
  "  v255 = icmp eq v253, v254\n" ++
  "  brif v255, block114(v250, v251, v252), block77(v250, v251, v252, v253)\n" ++
  "\n" ++

  -- block114: verify "rue" after 't' and not followed by ident char
  "block114(v2500: i64, v2501: i64, v2502: i64):\n" ++
  "  v2503 = iadd v0, v3\n" ++
  "  v2504 = iadd_imm v2500, 1\n" ++
  "  v2505 = iadd v2503, v2504\n" ++
  "  v2506 = uload8.i64 v2505\n" ++
  "  v2507 = iconst.i64 114\n" ++                               -- 'r'
  "  v2508 = icmp eq v2506, v2507\n" ++
  "  v2509 = iadd_imm v2500, 2\n" ++
  "  v2510 = iadd v2503, v2509\n" ++
  "  v2511 = uload8.i64 v2510\n" ++
  "  v2512 = iconst.i64 117\n" ++                               -- 'u'
  "  v2513 = icmp eq v2511, v2512\n" ++
  "  v2514 = iadd_imm v2500, 3\n" ++
  "  v2515 = iadd v2503, v2514\n" ++
  "  v2516 = uload8.i64 v2515\n" ++
  "  v2517 = iconst.i64 101\n" ++                               -- 'e'
  "  v2518 = icmp eq v2516, v2517\n" ++
  -- Check char after "true" is not alphanumeric
  "  v2519 = iadd_imm v2500, 4\n" ++
  "  v2520 = iadd v2503, v2519\n" ++
  "  v2521 = uload8.i64 v2520\n" ++
  "  v2522 = iconst.i64 97\n" ++                                -- 'a'
  "  v2523 = icmp ult v2521, v2522\n" ++                        -- ch < 'a'? (not lowercase)
  "  v2524 = band v2508, v2513\n" ++
  "  v2525 = band v2524, v2518\n" ++
  "  v2526 = band v2525, v2523\n" ++
  "  brif v2526, block63(v2500, v2501, v2502), block18(v2500, v2501, v2502, v5)\n" ++
  "\n" ++

  -- block77: check 'f' for "false" (full keyword, followed by non-alpha)
  "block77(v260: i64, v261: i64, v262: i64, v263: i64):\n" ++
  "  v264 = iconst.i64 102\n" ++                               -- 'f'
  "  v265 = icmp eq v263, v264\n" ++
  "  brif v265, block115(v260, v261, v262), block18(v260, v261, v262, v5)\n" ++
  "\n" ++

  -- block115: verify "alse" after 'f' and not followed by ident char
  "block115(v2600: i64, v2601: i64, v2602: i64):\n" ++
  "  v2603 = iadd v0, v3\n" ++
  "  v2604 = iadd_imm v2600, 1\n" ++
  "  v2605 = iadd v2603, v2604\n" ++
  "  v2606 = uload8.i64 v2605\n" ++
  "  v2607 = iconst.i64 97\n" ++                                -- 'a'
  "  v2608 = icmp eq v2606, v2607\n" ++
  "  v2609 = iadd_imm v2600, 2\n" ++
  "  v2610 = iadd v2603, v2609\n" ++
  "  v2611 = uload8.i64 v2610\n" ++
  "  v2612 = iconst.i64 108\n" ++                               -- 'l'
  "  v2613 = icmp eq v2611, v2612\n" ++
  "  v2614 = iadd_imm v2600, 3\n" ++
  "  v2615 = iadd v2603, v2614\n" ++
  "  v2616 = uload8.i64 v2615\n" ++
  "  v2617 = iconst.i64 115\n" ++                               -- 's'
  "  v2618 = icmp eq v2616, v2617\n" ++
  "  v2619 = iadd_imm v2600, 4\n" ++
  "  v2620 = iadd v2603, v2619\n" ++
  "  v2621 = uload8.i64 v2620\n" ++
  "  v2622 = iconst.i64 101\n" ++                               -- 'e'
  "  v2623 = icmp eq v2621, v2622\n" ++
  -- Check char after "false" is not alphanumeric
  "  v2624 = iadd_imm v2600, 5\n" ++
  "  v2625 = iadd v2603, v2624\n" ++
  "  v2626 = uload8.i64 v2625\n" ++
  "  v2627 = iconst.i64 97\n" ++                                -- 'a'
  "  v2628 = icmp ult v2626, v2627\n" ++                        -- ch < 'a'?
  "  v2629 = band v2608, v2613\n" ++
  "  v2630 = band v2629, v2618\n" ++
  "  v2631 = band v2630, v2623\n" ++
  "  v2632 = band v2631, v2628\n" ++
  "  brif v2632, block64(v2600, v2601, v2602), block18(v2600, v2601, v2602, v5)\n" ++
  "\n" ++

  -- =====================================================================
  -- block63: true literal — skip "true" (4 chars), return (pos+4, 1, is_bool=1)
  -- =====================================================================
  "block63(v270: i64, v271: i64, v272: i64):\n" ++
  "  v273 = iadd_imm v270, 4\n" ++                             -- skip "true"
  "  v274 = iconst.i64 1\n" ++
  "  jump block60(v273, v271, v272, v274, v274)\n" ++           -- val=1, is_bool=1
  "\n" ++

  -- =====================================================================
  -- block64: false literal — skip "false" (5 chars), return (pos+5, 0, is_bool=1)
  -- =====================================================================
  "block64(v280: i64, v281: i64, v282: i64):\n" ++
  "  v283 = iadd_imm v280, 5\n" ++                             -- skip "false"
  "  v284 = iconst.i64 1\n" ++
  "  jump block60(v283, v281, v282, v5, v284)\n" ++             -- val=0, is_bool=1
  "\n" ++

  -- =====================================================================
  -- block16: parse_number loop (pos, sp, ctx, accum)
  -- =====================================================================
  "block16(v290: i64, v291: i64, v292: i64, v293: i64):\n" ++
  "  v294 = iadd v0, v3\n" ++
  "  v295 = iadd v294, v290\n" ++
  "  v296 = uload8.i64 v295\n" ++                              -- ch
  "  v297 = iconst.i64 48\n" ++
  "  v298 = iconst.i64 57\n" ++
  "  v299 = icmp uge v296, v297\n" ++
  "  v300 = icmp ule v296, v298\n" ++
  "  v301 = band v299, v300\n" ++
  "  brif v301, block17(v290, v291, v292, v293, v296), block60(v290, v291, v292, v293, v5)\n" ++
  "\n" ++

  -- =====================================================================
  -- block17: parse_number accumulate (pos, sp, ctx, accum, ch)
  -- =====================================================================
  "block17(v310: i64, v311: i64, v312: i64, v313: i64, v314: i64):\n" ++
  "  v315 = iconst.i64 10\n" ++
  "  v316 = imul v313, v315\n" ++                               -- accum * 10
  "  v317 = iconst.i64 48\n" ++
  "  v318 = isub v314, v317\n" ++                               -- ch - '0'
  "  v319 = iadd v316, v318\n" ++                               -- new accum
  "  v320 = iadd_imm v310, 1\n" ++                              -- pos++
  "  jump block16(v320, v311, v312, v319)\n" ++
  "\n" ++

  -- =====================================================================
  -- block18: variable — read identifier (pos, sp, ctx, id_len)
  -- =====================================================================
  "block18(v330: i64, v331: i64, v332: i64, v333: i64):\n" ++
  "  v334 = iadd v0, v3\n" ++
  "  v335 = iadd v334, v330\n" ++
  "  v336 = uload8.i64 v335\n" ++                              -- ch
  -- Check if alphanumeric or underscore: a-z(97-122), A-Z(65-90), 0-9(48-57), _(95)
  "  v337 = iconst.i64 97\n" ++
  "  v338 = iconst.i64 122\n" ++
  "  v339 = icmp uge v336, v337\n" ++
  "  v340 = icmp ule v336, v338\n" ++
  "  v341 = band v339, v340\n" ++                              -- is lowercase?
  "  v342 = iconst.i64 65\n" ++
  "  v343 = iconst.i64 90\n" ++
  "  v344 = icmp uge v336, v342\n" ++
  "  v345 = icmp ule v336, v343\n" ++
  "  v346 = band v344, v345\n" ++                              -- is uppercase?
  "  v347 = iconst.i64 48\n" ++
  "  v348 = iconst.i64 57\n" ++
  "  v349 = icmp uge v336, v347\n" ++
  "  v350 = icmp ule v336, v348\n" ++
  "  v351 = band v349, v350\n" ++                              -- is digit?
  "  v352 = iconst.i64 95\n" ++
  "  v353 = icmp eq v336, v352\n" ++                            -- is '_'?
  "  v354 = bor v341, v346\n" ++
  "  v355 = bor v354, v351\n" ++
  "  v356 = bor v355, v353\n" ++                               -- is ident char?
  "  brif v356, block78(v330, v331, v332, v333, v336), block19(v330, v331, v332, v333)\n" ++
  "\n" ++

  -- block78: store ident char and continue
  "block78(v360: i64, v361: i64, v362: i64, v363: i64, v364: i64):\n" ++
  "  v365 = iadd v0, v6\n" ++                                  -- ident_buf base
  "  v366 = iadd v365, v363\n" ++                              -- &ident_buf[id_len]
  "  istore8 v364, v366\n" ++                                  -- store char
  "  v367 = iadd_imm v363, 1\n" ++                             -- id_len++
  "  v368 = iadd_imm v360, 1\n" ++                             -- pos++
  "  jump block18(v368, v361, v362, v367)\n" ++
  "\n" ++

  -- =====================================================================
  -- block19: variable lookup (pos, sp, ctx, id_len)
  -- =====================================================================
  "block19(v370: i64, v371: i64, v372: i64, v373: i64):\n" ++
  "  v374 = iadd v0, v6\n" ++                                  -- ident_buf addr
  "  v375 = ireduce.i32 v373\n" ++                             -- id_len as i32
  "  v376 = iadd v0, v7\n" ++                                  -- val_buf addr
  "  v377 = call fn2(v372, v374, v375, v376)\n" ++             -- ht_lookup
  "  v378 = load.i64 v376\n" ++                                -- value from HT
  "  jump block60(v370, v371, v372, v378, v5)\n" ++            -- is_bool=0
  "\n" ++

  -- =====================================================================
  -- block20: paren or lambda (pos, sp, ctx)
  -- Check if "(fun " follows
  -- =====================================================================
  "block20(v380: i64, v381: i64, v382: i64):\n" ++
  "  v383 = iadd_imm v380, 1\n" ++                             -- after '('
  "  v384 = iadd v0, v3\n" ++
  "  v385 = iadd v384, v383\n" ++
  "  v386 = uload8.i64 v385\n" ++
  "  v387 = iconst.i64 102\n" ++                               -- 'f'
  "  v388 = icmp eq v386, v387\n" ++
  "  brif v388, block40(v380, v381, v382), block21(v383, v381, v382)\n" ++
  "\n" ++

  -- =====================================================================
  -- block21: paren expr (pos_after_open, sp, ctx)
  -- Push frame tag=7 (paren), parse_expr inside
  -- =====================================================================
  "block21(v390: i64, v391: i64, v392: i64):\n" ++
  "  v393 = iadd v0, v9\n" ++
  "  v394 = iadd v393, v391\n" ++
  "  v395 = iconst.i64 7\n" ++                                 -- tag=7 (paren)
  "  store.i64 v395, v394\n" ++
  "  store.i64 v5, v394+8\n" ++
  "  store.i64 v5, v394+16\n" ++
  "  v396 = iadd_imm v391, 24\n" ++
  "  jump block1(v390, v396, v392, v5)\n" ++                   -- skip spaces then parse_expr
  "\n" ++

  -- =====================================================================
  -- block25: let binding (pos, sp, ctx) — pos points to 'l'
  -- Skip "let "
  -- =====================================================================
  "block25(v400: i64, v401: i64, v402: i64):\n" ++
  "  v403 = iadd_imm v400, 4\n" ++                             -- skip "let "
  "  jump block26(v403, v401, v402, v5)\n" ++                  -- read name, id_len=0
  "\n" ++

  -- =====================================================================
  -- block26: let read name (pos, sp, ctx, id_len)
  -- =====================================================================
  "block26(v410: i64, v411: i64, v412: i64, v413: i64):\n" ++
  "  v414 = iadd v0, v3\n" ++
  "  v415 = iadd v414, v410\n" ++
  "  v416 = uload8.i64 v415\n" ++
  "  v417 = iconst.i64 32\n" ++                                -- space
  "  v418 = icmp eq v416, v417\n" ++
  "  brif v418, block27(v410, v411, v412, v413), block79(v410, v411, v412, v413, v416)\n" ++
  "\n" ++

  -- block79: store name char
  "block79(v420: i64, v421: i64, v422: i64, v423: i64, v424: i64):\n" ++
  "  v425 = iadd v0, v6\n" ++
  "  v426 = iadd v425, v423\n" ++
  "  istore8 v424, v426\n" ++
  "  v427 = iadd_imm v423, 1\n" ++
  "  v428 = iadd_imm v420, 1\n" ++
  "  jump block26(v428, v421, v422, v427)\n" ++
  "\n" ++

  -- =====================================================================
  -- block27: let skip assign (pos, sp, ctx, id_len)
  -- pos is at space before ":=", skip ":= "
  -- =====================================================================
  "block27(v430: i64, v431: i64, v432: i64, v433: i64):\n" ++
  "  v434 = iadd_imm v430, 4\n" ++                             -- skip " := " (space + := + space)
  -- Push frame: tag=2 (let_bind), val=id_len, extra=0
  "  v435 = iadd v0, v9\n" ++
  "  v436 = iadd v435, v431\n" ++
  "  v437 = iconst.i64 2\n" ++
  "  store.i64 v437, v436\n" ++                                -- tag=2
  "  store.i64 v433, v436+8\n" ++                              -- saved id_len
  "  store.i64 v5, v436+16\n" ++
  "  v438 = iadd_imm v431, 24\n" ++
  -- Parse binding value expression
  "  jump block1(v434, v438, v432, v5)\n" ++
  "\n" ++

  -- =====================================================================
  -- block29: let store and continue (pos, sp, ctx, bind_value, id_len)
  -- Store value via ht_insert, skip "; ", parse body
  -- =====================================================================
  "block29(v440: i64, v441: i64, v442: i64, v443: i64, v444: i64):\n" ++
  -- Store bind_value to HT_VAL_BUF
  "  v445 = iadd v0, v7\n" ++                                  -- val_buf addr
  "  store.i64 v443, v445\n" ++
  "  v446 = iadd v0, v6\n" ++                                  -- ident_buf addr
  "  v447 = ireduce.i32 v444\n" ++                             -- id_len as i32
  "  v448 = iconst.i32 8\n" ++                                 -- val_len=8
  "  call fn1(v442, v446, v447, v445, v448)\n" ++              -- ht_insert
  -- Skip "; " or ";\n"
  "  v449 = iadd v0, v3\n" ++
  "  v450 = iadd v449, v440\n" ++
  "  v451 = uload8.i64 v450\n" ++
  "  v452 = iconst.i64 59\n" ++                                -- ';'
  "  v453 = icmp eq v451, v452\n" ++
  "  v454 = iadd_imm v440, 1\n" ++                             -- skip ';'
  "  v455 = select v453, v454, v440\n" ++
  -- Push frame: tag=3 (let_body), val=0, extra=0
  "  v456 = iadd v0, v9\n" ++
  "  v457 = iadd v456, v441\n" ++
  "  v458 = iconst.i64 3\n" ++
  "  store.i64 v458, v457\n" ++
  "  store.i64 v5, v457+8\n" ++
  "  store.i64 v5, v457+16\n" ++
  "  v459 = iadd_imm v441, 24\n" ++
  "  jump block1(v455, v459, v442, v5)\n" ++                   -- skip spaces, parse body
  "\n" ++

  -- =====================================================================
  -- block30: if-then-else (pos, sp, ctx) — pos at 'i'
  -- Skip "if "
  -- =====================================================================
  "block30(v460: i64, v461: i64, v462: i64):\n" ++
  "  v463 = iadd_imm v460, 3\n" ++                             -- skip "if "
  -- Push frame: tag=4 (if_cond), val=0, extra=0
  "  v464 = iadd v0, v9\n" ++
  "  v465 = iadd v464, v461\n" ++
  "  v466 = iconst.i64 4\n" ++
  "  store.i64 v466, v465\n" ++
  "  store.i64 v5, v465+8\n" ++
  "  store.i64 v5, v465+16\n" ++
  "  v467 = iadd_imm v461, 24\n" ++
  "  jump block1(v463, v467, v462, v5)\n" ++                   -- parse condition
  "\n" ++

  -- =====================================================================
  -- block32: if skip "then " (pos, sp, ctx, cond_value)
  -- =====================================================================
  "block32(v470: i64, v471: i64, v472: i64, v473: i64):\n" ++
  -- skip spaces then "then "
  "  v474 = iadd v0, v3\n" ++
  "  v475 = iadd v474, v470\n" ++
  "  v476 = uload8.i64 v475\n" ++
  "  v477 = iconst.i64 32\n" ++
  "  v478 = icmp eq v476, v477\n" ++
  "  v479 = iadd_imm v470, 1\n" ++
  "  v480 = select v478, v479, v470\n" ++                      -- skip leading space
  "  v481 = iadd_imm v480, 5\n" ++                             -- skip "then "
  -- Push frame: tag=5 (if_then), val=cond, extra=0
  "  v482 = iadd v0, v9\n" ++
  "  v483 = iadd v482, v471\n" ++
  "  v484 = iconst.i64 5\n" ++
  "  store.i64 v484, v483\n" ++
  "  store.i64 v473, v483+8\n" ++                              -- save condition
  "  store.i64 v5, v483+16\n" ++
  "  v485 = iadd_imm v471, 24\n" ++
  "  jump block1(v481, v485, v472, v5)\n" ++                   -- parse then-branch
  "\n" ++

  -- =====================================================================
  -- block34: if skip "else " (pos, sp, ctx, cond, then_val)
  -- =====================================================================
  "block34(v490: i64, v491: i64, v492: i64, v493: i64, v494: i64):\n" ++
  -- skip spaces then "else "
  "  v495 = iadd v0, v3\n" ++
  "  v496 = iadd v495, v490\n" ++
  "  v497 = uload8.i64 v496\n" ++
  "  v498 = iconst.i64 32\n" ++
  "  v499 = icmp eq v497, v498\n" ++
  "  v500 = iadd_imm v490, 1\n" ++
  "  v501 = select v499, v500, v490\n" ++
  "  v502 = iadd_imm v501, 5\n" ++                             -- skip "else "
  -- Push frame: tag=6 (if_else), val=cond, extra=then_val
  "  v503 = iadd v0, v9\n" ++
  "  v504 = iadd v503, v491\n" ++
  "  v505 = iconst.i64 6\n" ++
  "  store.i64 v505, v504\n" ++
  "  store.i64 v493, v504+8\n" ++                              -- save cond
  "  store.i64 v494, v504+16\n" ++                             -- save then_val
  "  v506 = iadd_imm v491, 24\n" ++
  "  jump block1(v502, v506, v492, v5)\n" ++                   -- parse else-branch
  "\n" ++

  -- =====================================================================
  -- block40: lambda (pos, sp, ctx) — pos at '('
  -- Skip "(fun "
  -- =====================================================================
  "block40(v510: i64, v511: i64, v512: i64):\n" ++
  "  v513 = iadd_imm v510, 5\n" ++                             -- skip "(fun "
  "  jump block41(v513, v511, v512, v5)\n" ++                  -- read param name
  "\n" ++

  -- =====================================================================
  -- block41: lambda read param (pos, sp, ctx, id_len)
  -- =====================================================================
  "block41(v520: i64, v521: i64, v522: i64, v523: i64):\n" ++
  "  v524 = iadd v0, v3\n" ++
  "  v525 = iadd v524, v520\n" ++
  "  v526 = uload8.i64 v525\n" ++
  "  v527 = iconst.i64 32\n" ++
  "  v528 = icmp eq v526, v527\n" ++                            -- space means end of param
  "  brif v528, block42(v520, v521, v522, v523), block80(v520, v521, v522, v523, v526)\n" ++
  "\n" ++

  -- block80: store param char
  "block80(v530: i64, v531: i64, v532: i64, v533: i64, v534: i64):\n" ++
  "  v535 = iadd v0, v6\n" ++
  "  v536 = iadd v535, v533\n" ++
  "  istore8 v534, v536\n" ++
  "  v537 = iadd_imm v533, 1\n" ++
  "  v538 = iadd_imm v530, 1\n" ++
  "  jump block41(v538, v531, v532, v537)\n" ++
  "\n" ++

  -- =====================================================================
  -- block42: lambda skip "=> " (pos, sp, ctx, id_len)
  -- pos at space before "=>"
  -- =====================================================================
  "block42(v540: i64, v541: i64, v542: i64, v543: i64):\n" ++
  "  v544 = iadd_imm v540, 4\n" ++                             -- skip " => "
  -- Save body_start_pos, then scan for closing ')'
  -- Push frame: tag=8 (lambda_arg), val=body_start_pos, extra=id_len
  "  v545 = iadd v0, v9\n" ++
  "  v546 = iadd v545, v541\n" ++
  "  v547 = iconst.i64 8\n" ++
  "  store.i64 v547, v546\n" ++                                -- tag=8
  "  store.i64 v544, v546+8\n" ++                              -- body_start_pos
  "  store.i64 v543, v546+16\n" ++                             -- id_len
  "  v548 = iadd_imm v541, 24\n" ++
  -- Scan to closing ')' (tracking depth)
  "  jump block43(v544, v548, v542, v5)\n" ++
  "\n" ++

  -- =====================================================================
  -- block43: lambda skip body — scan to ')' (pos, sp, ctx, depth)
  -- =====================================================================
  "block43(v550: i64, v551: i64, v552: i64, v553: i64):\n" ++
  "  v554 = iadd v0, v3\n" ++
  "  v555 = iadd v554, v550\n" ++
  "  v556 = uload8.i64 v555\n" ++
  "  v567 = iadd_imm v550, 1\n" ++                             -- pos++
  -- check ')'
  "  v559 = iconst.i64 41\n" ++                                -- ')'
  "  v560 = icmp eq v556, v559\n" ++
  "  brif v560, block81(v567, v551, v552, v553), block100(v567, v551, v552, v553, v556)\n" ++
  "\n" ++

  -- block100: not ')' — check '(' to increment depth
  "block100(v5500: i64, v5501: i64, v5502: i64, v5503: i64, v5504: i64):\n" ++
  "  v557 = iconst.i64 40\n" ++                                -- '('
  "  v558 = icmp eq v5504, v557\n" ++
  "  v561 = iadd_imm v5503, 1\n" ++                            -- depth+1
  "  brif v558, block43(v5500, v5501, v5502, v561), block43(v5500, v5501, v5502, v5503)\n" ++
  "\n" ++

  -- block81: found ')' (pos, sp, ctx, depth) — check if depth==0
  "block81(v570: i64, v571: i64, v572: i64, v573: i64):\n" ++
  "  v565 = icmp eq v573, v5\n" ++                             -- depth==0?
  "  brif v565, block44(v570, v571, v572), block101(v570, v571, v572, v573)\n" ++
  "\n" ++

  -- block101: ')' but depth > 0 — decrement and continue
  "block101(v5700: i64, v5701: i64, v5702: i64, v5703: i64):\n" ++
  "  v562 = iconst.i64 1\n" ++
  "  v563 = isub v5703, v562\n" ++                              -- depth-1
  "  jump block43(v5700, v5701, v5702, v563)\n" ++
  "\n" ++

  -- =====================================================================
  -- block44: lambda parse arg (pos_after_close, sp, ctx)
  -- Skip space after ')', then parse_atom for argument
  -- =====================================================================
  "block44(v580: i64, v581: i64, v582: i64):\n" ++
  "  v583 = iadd v0, v3\n" ++
  "  v584 = iadd v583, v580\n" ++
  "  v585 = uload8.i64 v584\n" ++
  "  v586 = iconst.i64 32\n" ++
  "  v587 = icmp eq v585, v586\n" ++
  "  v588 = iadd_imm v580, 1\n" ++
  "  v589 = select v587, v588, v580\n" ++
  -- Now parse the argument (a single atom/expr)
  -- Push frame: tag=9 (lambda_body), val=0, extra=0
  "  v590 = iadd v0, v9\n" ++
  "  v591 = iadd v590, v581\n" ++
  "  v592 = iconst.i64 9\n" ++
  "  store.i64 v592, v591\n" ++
  "  store.i64 v5, v591+8\n" ++
  "  store.i64 v5, v591+16\n" ++
  "  v593 = iadd_imm v581, 24\n" ++
  "  jump block10(v589, v593, v582)\n" ++                      -- parse_term for arg
  "\n" ++

  -- =====================================================================
  -- block45: lambda bind and eval (pos, sp, ctx, arg_val, body_pos, id_len)
  -- =====================================================================
  "block45(v594: i64, v595: i64, v596: i64, v597: i64, v598: i64, v599: i64):\n" ++
  -- Store arg_val to HT_VAL_BUF
  "  v5940 = iadd v0, v7\n" ++
  "  store.i64 v597, v5940\n" ++
  "  v5941 = iadd v0, v6\n" ++                                 -- ident_buf
  "  v5942 = ireduce.i32 v599\n" ++
  "  v5943 = iconst.i32 8\n" ++
  "  call fn1(v596, v5941, v5942, v5940, v5943)\n" ++          -- ht_insert param=arg
  -- Now parse body from body_pos
  -- Push frame: tag=3 (let_body — reuse), val=0, extra=0
  "  v5944 = iadd v0, v9\n" ++
  "  v5945 = iadd v5944, v595\n" ++
  "  v5946 = iconst.i64 3\n" ++
  "  store.i64 v5946, v5945\n" ++
  "  store.i64 v5, v5945+8\n" ++
  "  store.i64 v5, v5945+16\n" ++
  "  v5947 = iadd_imm v595, 24\n" ++
  "  jump block1(v598, v5947, v596, v5)\n" ++                  -- parse body
  "\n" ++

  -- =====================================================================
  -- block60: return from subexpr (pos, sp, ctx, value, is_bool)
  -- Pop stack frame, dispatch based on tag
  -- =====================================================================
  "block60(v700: i64, v701: i64, v702: i64, v703: i64, v704: i64):\n" ++
  "  v705 = iconst.i64 24\n" ++
  "  v706 = isub v701, v705\n" ++                              -- sp -= 24
  "  v707 = iadd v0, v9\n" ++
  "  v708 = iadd v707, v706\n" ++                              -- frame addr
  "  v709 = load.i64 v708\n" ++                                -- tag
  "  v710 = load.i64 v708+8\n" ++                              -- saved_value
  "  v711 = load.i64 v708+16\n" ++                             -- saved_extra
  -- Dispatch on tag
  "  v712 = iconst.i64 255\n" ++
  "  v713 = icmp eq v709, v712\n" ++                            -- done?
  "  brif v713, block50(v703, v704), block82(v700, v706, v702, v703, v704, v709, v710, v711)\n" ++
  "\n" ++

  -- block82: check tag=0 (expr_op)
  "block82(v720: i64, v721: i64, v722: i64, v723: i64, v724: i64, v725: i64, v726: i64, v727: i64):\n" ++
  "  v728 = icmp eq v725, v5\n" ++                             -- tag==0?
  "  brif v728, block83(v720, v721, v722, v723, v724, v726, v727), block84(v720, v721, v722, v723, v724, v725, v726, v727)\n" ++
  "\n" ++

  -- block83: expr_op return (pos, sp, ctx, value, is_bool, saved_val, extra)
  -- extra: 0=first_term, 1=add, 2=sub
  "block83(v730: i64, v731: i64, v732: i64, v733: i64, v734: i64, v735: i64, v736: i64):\n" ++
  "  v737 = icmp eq v736, v5\n" ++                             -- extra==0 (first term)?
  "  brif v737, block4(v730, v731, v732, v733, v734), block85(v730, v731, v732, v733, v735, v736)\n" ++
  "\n" ++

  -- block85: add or sub
  "block85(v740: i64, v741: i64, v742: i64, v743: i64, v744: i64, v745: i64):\n" ++
  "  v746 = iconst.i64 1\n" ++
  "  v747 = icmp eq v745, v746\n" ++                           -- op==1 (add)?
  "  v748 = iadd v744, v743\n" ++                              -- left + right
  "  v749 = isub v744, v743\n" ++                              -- left - right
  "  v750 = select v747, v748, v749\n" ++
  "  jump block4(v740, v741, v742, v750, v5)\n" ++             -- continue expr_op check
  "\n" ++

  -- block84: check tag=1 (term_op)
  "block84(v760: i64, v761: i64, v762: i64, v763: i64, v764: i64, v765: i64, v766: i64, v767: i64):\n" ++
  "  v768 = iconst.i64 1\n" ++
  "  v769 = icmp eq v765, v768\n" ++                           -- tag==1?
  "  brif v769, block86(v760, v761, v762, v763, v764, v766, v767), block87(v760, v761, v762, v763, v764, v765, v766, v767)\n" ++
  "\n" ++

  -- block86: term_op return (pos, sp, ctx, value, is_bool, saved_val, extra)
  -- extra: 0=first_atom, 1=mul
  "block86(v770: i64, v771: i64, v772: i64, v773: i64, v7700: i64, v774: i64, v775: i64):\n" ++
  "  v776 = icmp eq v775, v5\n" ++                             -- extra==0 (first atom)?
  "  brif v776, block11(v770, v771, v772, v773, v7700), block88(v770, v771, v772, v773, v774)\n" ++
  "\n" ++

  -- block88: multiply
  "block88(v780: i64, v781: i64, v782: i64, v783: i64, v784: i64):\n" ++
  "  v785 = imul v784, v783\n" ++                              -- left * right
  "  jump block11(v780, v781, v782, v785, v5)\n" ++            -- continue term_op check, is_bool=0
  "\n" ++

  -- block87: check tag=2 (let_bind)
  "block87(v790: i64, v791: i64, v792: i64, v793: i64, v794: i64, v795: i64, v796: i64, v797: i64):\n" ++
  "  v798 = iconst.i64 2\n" ++
  "  v799 = icmp eq v795, v798\n" ++                           -- tag==2?
  "  brif v799, block29(v790, v791, v792, v793, v796), block89(v790, v791, v792, v793, v794, v795, v796, v797)\n" ++
  "\n" ++

  -- block89: check tag=3 (let_body / lambda_body_done)
  "block89(v800: i64, v801: i64, v802: i64, v803: i64, v804: i64, v805: i64, v806: i64, v807: i64):\n" ++
  "  v808 = iconst.i64 3\n" ++
  "  v809 = icmp eq v805, v808\n" ++                           -- tag==3?
  -- Just pass value through (body result is the let/lambda result)
  "  brif v809, block60(v800, v801, v802, v803, v804), block90(v800, v801, v802, v803, v804, v805, v806, v807)\n" ++
  "\n" ++

  -- block90: check tag=4 (if_cond)
  "block90(v810: i64, v811: i64, v812: i64, v813: i64, v814: i64, v815: i64, v816: i64, v817: i64):\n" ++
  "  v818 = iconst.i64 4\n" ++
  "  v819 = icmp eq v815, v818\n" ++                           -- tag==4?
  "  brif v819, block32(v810, v811, v812, v813), block91(v810, v811, v812, v813, v814, v815, v816, v817)\n" ++
  "\n" ++

  -- block91: check tag=5 (if_then)
  "block91(v820: i64, v821: i64, v822: i64, v823: i64, v824: i64, v825: i64, v826: i64, v827: i64):\n" ++
  "  v828 = iconst.i64 5\n" ++
  "  v829 = icmp eq v825, v828\n" ++                           -- tag==5?
  "  brif v829, block34(v820, v821, v822, v826, v823), block92(v820, v821, v822, v823, v824, v825, v826, v827)\n" ++
  "\n" ++

  -- block92: check tag=6 (if_else)
  "block92(v830: i64, v831: i64, v832: i64, v833: i64, v834: i64, v835: i64, v836: i64, v837: i64):\n" ++
  "  v838 = iconst.i64 6\n" ++
  "  v839 = icmp eq v835, v838\n" ++                           -- tag==6?
  "  brif v839, block36(v830, v831, v832, v836, v837, v833), block93(v830, v831, v832, v833, v834, v835, v836, v837)\n" ++
  "\n" ++

  -- block36: if select (pos, sp, ctx, cond, then_val, else_val)
  "block36(v840: i64, v841: i64, v842: i64, v843: i64, v844: i64, v845: i64):\n" ++
  "  v846 = icmp ne v843, v5\n" ++                             -- cond != 0?
  "  v847 = select v846, v844, v845\n" ++
  "  jump block60(v840, v841, v842, v847, v5)\n" ++
  "\n" ++

  -- block93: check tag=7 (paren)
  "block93(v850: i64, v851: i64, v852: i64, v853: i64, v854: i64, v855: i64, v856: i64, v857: i64):\n" ++
  "  v858 = iconst.i64 7\n" ++
  "  v859 = icmp eq v855, v858\n" ++                           -- tag==7?
  "  brif v859, block94(v850, v851, v852, v853, v854), block95(v850, v851, v852, v853, v854, v855, v856, v857)\n" ++
  "\n" ++

  -- block94: paren close — skip ')' and return
  "block94(v860: i64, v861: i64, v862: i64, v863: i64, v864: i64):\n" ++
  -- skip optional spaces then ')'
  "  v865 = iadd v0, v3\n" ++
  "  v866 = iadd v865, v860\n" ++
  "  v867 = uload8.i64 v866\n" ++
  "  v868 = iconst.i64 41\n" ++                                -- ')'
  "  v869 = icmp eq v867, v868\n" ++
  "  v870 = iadd_imm v860, 1\n" ++
  "  v871 = select v869, v870, v860\n" ++
  "  jump block60(v871, v861, v862, v863, v864)\n" ++
  "\n" ++

  -- block95: check tag=8 (lambda_arg)
  "block95(v880: i64, v881: i64, v882: i64, v883: i64, v884: i64, v885: i64, v886: i64, v887: i64):\n" ++
  "  v888 = iconst.i64 8\n" ++
  "  v889 = icmp eq v885, v888\n" ++                           -- tag==8?
  -- val=body_pos, extra=id_len, value=arg_val
  "  brif v889, block45(v880, v881, v882, v883, v886, v887), block96(v880, v881, v882, v883, v884, v885, v886, v887)\n" ++
  "\n" ++

  -- block96: check tag=9 (lambda_body)
  "block96(v890: i64, v891: i64, v892: i64, v893: i64, v894: i64, v895: i64, v896: i64, v897: i64):\n" ++
  "  v898 = iconst.i64 9\n" ++
  "  v899 = icmp eq v895, v898\n" ++                           -- tag==9?
  -- value from parse_term is the arg; we need to take it to lambda_arg handler
  -- But actually this is a direct parse of the argument atom
  -- After getting arg, pop frame 8 (lambda_arg) and call block45
  "  brif v899, block60(v890, v891, v892, v893, v894), block97(v890, v891, v892, v893, v894, v895, v896, v897)\n" ++
  "\n" ++

  -- block97: check tag=10 (cmp_rhs)
  "block97(v900: i64, v901: i64, v902: i64, v903: i64, v904: i64, v905: i64, v906: i64, v907: i64):\n" ++
  "  v908 = iconst.i64 10\n" ++
  "  v909 = icmp eq v905, v908\n" ++                           -- tag==10?
  "  brif v909, block61(v900, v901, v902, v906, v903, v907), block60(v900, v901, v902, v903, v904)\n" ++
  "\n" ++

  -- =====================================================================
  -- block61: comparison result (pos, sp, ctx, left, right, op)
  -- op: 0=lt, 1=le, 2=gt, 3=ge
  -- Use branches to dispatch on op, then branch on comparison result
  -- =====================================================================
  "block61(v910: i64, v911: i64, v912: i64, v913: i64, v914: i64, v915: i64):\n" ++
  "  v920 = icmp eq v915, v5\n" ++                             -- op==0 (lt)?
  "  brif v920, block102(v910, v911, v912, v913, v914), block103(v910, v911, v912, v913, v914, v915)\n" ++
  "\n" ++

  -- block102: lt comparison
  "block102(v9100: i64, v9101: i64, v9102: i64, v9103: i64, v9104: i64):\n" ++
  "  v916 = icmp slt v9103, v9104\n" ++
  "  v934 = iconst.i64 1\n" ++
  "  brif v916, block60(v9100, v9101, v9102, v934, v934), block60(v9100, v9101, v9102, v5, v934)\n" ++
  "\n" ++

  -- block103: check le
  "block103(v9110: i64, v9111: i64, v9112: i64, v9113: i64, v9114: i64, v9115: i64):\n" ++
  "  v921 = iconst.i64 1\n" ++
  "  v922 = icmp eq v9115, v921\n" ++
  "  brif v922, block104(v9110, v9111, v9112, v9113, v9114), block105(v9110, v9111, v9112, v9113, v9114, v9115)\n" ++
  "\n" ++

  -- block104: le comparison
  "block104(v9120: i64, v9121: i64, v9122: i64, v9123: i64, v9124: i64):\n" ++
  "  v917 = icmp sle v9123, v9124\n" ++
  "  v935 = iconst.i64 1\n" ++
  "  brif v917, block60(v9120, v9121, v9122, v935, v935), block60(v9120, v9121, v9122, v5, v935)\n" ++
  "\n" ++

  -- block105: check gt
  "block105(v9130: i64, v9131: i64, v9132: i64, v9133: i64, v9134: i64, v9135: i64):\n" ++
  "  v923 = iconst.i64 2\n" ++
  "  v924 = icmp eq v9135, v923\n" ++
  "  brif v924, block106(v9130, v9131, v9132, v9133, v9134), block107(v9130, v9131, v9132, v9133, v9134)\n" ++
  "\n" ++

  -- block106: gt comparison
  "block106(v9140: i64, v9141: i64, v9142: i64, v9143: i64, v9144: i64):\n" ++
  "  v918 = icmp sgt v9143, v9144\n" ++
  "  v936 = iconst.i64 1\n" ++
  "  brif v918, block60(v9140, v9141, v9142, v936, v936), block60(v9140, v9141, v9142, v5, v936)\n" ++
  "\n" ++

  -- block107: must be ge
  "block107(v9150: i64, v9151: i64, v9152: i64, v9153: i64, v9154: i64):\n" ++
  "  v919 = icmp sge v9153, v9154\n" ++
  "  v937 = iconst.i64 1\n" ++
  "  brif v919, block60(v9150, v9151, v9152, v937, v937), block60(v9150, v9151, v9152, v5, v937)\n" ++
  "\n" ++

  -- =====================================================================
  -- block50: output (value, is_bool)
  -- =====================================================================
  "block50(v940: i64, v941: i64):\n" ++
  "  v942 = icmp ne v941, v5\n" ++                             -- is_bool?
  "  brif v942, block98(v940), block53(v940, v5)\n" ++
  "\n" ++

  -- block98: boolean output — check true/false
  "block98(v943: i64):\n" ++
  "  v944 = icmp ne v943, v5\n" ++                             -- value != 0?
  "  brif v944, block51, block52\n" ++
  "\n" ++

  -- =====================================================================
  -- block51: write "true"
  -- =====================================================================
  "block51:\n" ++
  "  v950 = iadd v0, v8\n" ++                                  -- output_buf
  "  v951 = iconst.i64 116\n" ++                               -- 't'
  "  istore8 v951, v950\n" ++
  "  v952 = iconst.i64 114\n" ++                               -- 'r'
  "  istore8 v952, v950+1\n" ++
  "  v953 = iconst.i64 117\n" ++                               -- 'u'
  "  istore8 v953, v950+2\n" ++
  "  v954 = iconst.i64 101\n" ++                               -- 'e'
  "  istore8 v954, v950+3\n" ++
  "  v955 = iconst.i64 10\n" ++                                -- '\n'
  "  istore8 v955, v950+4\n" ++
  "  istore8 v5, v950+5\n" ++                                  -- null terminator
  "  return\n" ++
  "\n" ++

  -- =====================================================================
  -- block52: write "false"
  -- =====================================================================
  "block52:\n" ++
  "  v960 = iadd v0, v8\n" ++
  "  v961 = iconst.i64 102\n" ++                               -- 'f'
  "  istore8 v961, v960\n" ++
  "  v962 = iconst.i64 97\n" ++                                -- 'a'
  "  istore8 v962, v960+1\n" ++
  "  v963 = iconst.i64 108\n" ++                               -- 'l'
  "  istore8 v963, v960+2\n" ++
  "  v964 = iconst.i64 115\n" ++                               -- 's'
  "  istore8 v964, v960+3\n" ++
  "  v965 = iconst.i64 101\n" ++                               -- 'e'
  "  istore8 v965, v960+4\n" ++
  "  v966 = iconst.i64 10\n" ++                                -- '\n'
  "  istore8 v966, v960+5\n" ++
  "  istore8 v5, v960+6\n" ++
  "  return\n" ++
  "\n" ++

  -- =====================================================================
  -- block53: itoa — find highest power of 10 (value, out_pos)
  -- =====================================================================
  "block53(v970: i64, v971: i64):\n" ++
  "  v972 = iadd v0, v8\n" ++                                  -- output_buf base
  "  v973 = iconst.i64 1\n" ++                                 -- initial divisor
  "  jump block99(v970, v971, v973, v972)\n" ++
  "\n" ++

  -- block99: find divisor loop (value, out_pos, divisor, buf_addr)
  "block99(v974: i64, v975: i64, v976: i64, v977: i64):\n" ++
  "  v978 = iconst.i64 10\n" ++
  "  v979 = imul v976, v978\n" ++                              -- divisor * 10
  "  v980 = icmp ugt v979, v974\n" ++                          -- > value?
  "  brif v980, block54(v974, v975, v976, v977), block99(v974, v975, v979, v977)\n" ++
  "\n" ++

  -- =====================================================================
  -- block54: itoa write digit (remainder, out_pos, divisor, buf_addr)
  -- =====================================================================
  "block54(v981: i64, v982: i64, v983: i64, v984: i64):\n" ++
  "  v985 = udiv v981, v983\n" ++                              -- digit = rem / div
  "  v986 = iconst.i64 48\n" ++                                -- '0'
  "  v987 = iadd v985, v986\n" ++                              -- ascii digit
  "  v988 = iadd v984, v982\n" ++                              -- buf + out_pos
  "  istore8 v987, v988\n" ++
  "  v989 = imul v985, v983\n" ++
  "  v990 = isub v981, v989\n" ++                              -- new remainder
  "  v991 = iconst.i64 10\n" ++
  "  v992 = udiv v983, v991\n" ++                              -- divisor / 10
  "  v993 = iadd_imm v982, 1\n" ++                             -- out_pos++
  "  v994 = icmp eq v992, v5\n" ++                             -- divisor == 0?
  "  brif v994, block55(v993, v984), block54(v990, v993, v992, v984)\n" ++
  "\n" ++

  -- =====================================================================
  -- block55: itoa done — write newline + null
  -- =====================================================================
  "block55(v995: i64, v996: i64):\n" ++
  "  v997 = iadd v996, v995\n" ++
  "  v998 = iconst.i64 10\n" ++                                -- '\n'
  "  istore8 v998, v997\n" ++
  "  v999 = iadd_imm v995, 1\n" ++
  "  v1000 = iadd v996, v999\n" ++
  "  istore8 v5, v1000\n" ++                                   -- null
  "  return\n" ++
  "}\n"

def clifIR : String :=
  clifReadFn ++ "\n" ++ clifComputeFn ++ "\n" ++ clifWriteFn

def clifIRBytes : List UInt8 :=
  clifIR.toUTF8.toList ++ [0]

def buildPayload : List UInt8 :=
  let htCtxPtr    := zeros 8                               -- 0x0000
  let flagFile    := zeros 8                               -- 0x0008
  let flagCl      := zeros 8                               -- 0x0010
  let reserved    := zeros 8                               -- 0x0018
  let outputPath  := padTo (stringToBytes "output.txt") 64 -- 0x0020
  let inputPath   := zeros 256                             -- 0x0060
  let sourceBuf   := zeros SOURCE_BUF_SZ                   -- 0x0160
  let trueStr     := padTo (stringToBytes "true") 8        -- 0x1160
  let falseStr    := padTo (stringToBytes "false") 8       -- 0x1168
  let identBuf    := zeros IDENT_BUF_SZ                    -- 0x1170
  let htValBuf    := zeros 8                               -- 0x11B0
  let outputBuf   := zeros OUTPUT_BUF_SZ                   -- 0x11B8
  let stackRegion := zeros STACK_SZ                        -- 0x11F8
  htCtxPtr ++ flagFile ++ flagCl ++ reserved ++
    outputPath ++ inputPath ++ sourceBuf ++
    trueStr ++ falseStr ++ identBuf ++ htValBuf ++ outputBuf ++
    stackRegion

def actions : List Action := [
  -- 0: File read (fn 0)
  { kind := .ClifCall, dst := 0, src := 0, offset := 0, size := 0 },
  -- 1: Compute (fn 1)
  { kind := .ClifCall, dst := 0, src := 1, offset := 0, size := 0 },
  -- 2: File write (fn 2)
  { kind := .ClifCall, dst := 0, src := 2, offset := 0, size := 0 }
]

def buildConfig : BaseConfig := {
  cranelift_ir := clifIR,
  memory_size := buildPayload.length,
  context_offset := 0
}

def buildAlgorithm : Algorithm := {
  actions := actions,
  payloads := buildPayload,
  cranelift_units := 0,
  timeout_ms := some TIMEOUT_MS
}

end LeanEval

def main : IO Unit := do
  let json := toJsonPair LeanEval.buildConfig LeanEval.buildAlgorithm
  IO.println (Json.compress json)
