import AlgorithmLib
open Lean (Json toJson)
open AlgorithmLib

namespace Algorithm

-- ---------------------------------------------------------------------------
-- SHA-256 hasher
--
-- Single CLIF function:
--   1. cl_file_read — load input file into memory
--   2. SHA-256 padding (append 0x80, zeros, 64-bit BE bit length)
--   3. Process each 64-byte block: message schedule + 64-round compression
--   4. Format 8 x u32 hash state as 64 hex chars
--   5. cl_file_write — write hex digest + newline
--
-- All 32-bit SHA-256 arithmetic done in i64 with band mask32 (0xFFFFFFFF).
-- Only ireduce.i32 for stores.
--
-- Memory layout (all offsets relative to shared memory base v0):
--   [0x0000..0x0040)  reserved
--   [0x0040..0x0048)  scratch: file_size (i64)
--   [0x0048..0x0050)  scratch: padded_len (i64)
--   [0x0050..0x0058)  scratch: num_blocks (i64)
--   [0x0100..0x0200)  input filename (null-terminated, patched at runtime)
--   [0x0200..0x0300)  output filename "sha256_output.txt"
--   [0x0300..0x0342)  hex output buffer (64 hex chars + newline = 65 bytes)
--   [0x1000..0x10FF)  K constants (64 x u32 LE, in payload)
--   [0x1100..0x111F)  H initial state (8 x u32 LE, in payload)
--   [0x1120..0x113F)  H working state (8 x u32, runtime)
--   [0x1140..0x123F)  W message schedule (64 x u32 = 256 bytes)
--   [0x1240..0x124F)  hex lookup table "0123456789abcdef" (in payload)
--   [0x2000+)         file data + padding (up to 4MB + 128)
-- ---------------------------------------------------------------------------

-- Memory region sizes
def maxFileSize : Nat := 4 * 1024 * 1024  -- 4MB max input file

-- Offsets
def reserved_off : Nat := 0x0000
def fileSize_off : Nat := 0x0040
def paddedLen_off : Nat := 0x0048
def numBlocks_off : Nat := 0x0050
def inputFilename_off : Nat := 0x0100
def outputFilename_off : Nat := 0x0200
def hexOutput_off : Nat := 0x0300
def K_off : Nat := 0x1000
def H_init_off : Nat := 0x1100
def H_work_off : Nat := 0x1120
def W_off : Nat := 0x1140
def hexTable_off : Nat := 0x1240
def fileData_off : Nat := 0x2000

def totalMemory : Nat := fileData_off + maxFileSize + 128

-- SHA-256 constants K[0..63]
def kConstants : List UInt32 := [
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
  0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
  0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
  0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
  0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
  0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
  0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]

-- SHA-256 initial hash values H[0..7]
def hInitial : List UInt32 := [
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
]

-- ---------------------------------------------------------------------------
-- CLIF IR: SHA-256 hasher
-- ---------------------------------------------------------------------------

def clifIrSource : String :=
  let inFname := toString inputFilename_off
  let outFname := toString outputFilename_off
  let hexOutOff := toString hexOutput_off
  let kOff := toString K_off
  let hInitOff := toString H_init_off
  let hWorkOff := toString H_work_off
  let wOff := toString W_off
  let hexTblOff := toString hexTable_off
  let dataOff := toString fileData_off
  let fileSzOff := toString fileSize_off
  let paddedLnOff := toString paddedLen_off
  let numBlkOff := toString numBlocks_off

  -- noop function (required)
  "function u0:0(i64) system_v {\n" ++
  "block0(v0: i64):\n" ++
  "    return\n" ++
  "}\n\n" ++

  -- main function
  "function u0:1(i64) system_v {\n" ++
  "    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v\n" ++
  "    fn0 = %cl_file_read sig0\n" ++
  "    fn1 = %cl_file_write sig0\n" ++
  "\n" ++
  "block0(v0: i64):\n" ++

  -- === Step 1: Read input file ===
  s!"    v1 = iconst.i64 {inFname}\n" ++
  s!"    v2 = iconst.i64 {dataOff}\n" ++
  "    v3 = iconst.i64 0\n" ++
  "    v4 = call fn0(v0, v1, v2, v3, v3)\n" ++   -- v4 = bytes read (file_size)

  -- Store file_size
  s!"    v5 = iconst.i64 {fileSzOff}\n" ++
  "    v6 = iadd v0, v5\n" ++
  "    store v4, v6\n" ++

  -- Constants used throughout (defined in block0 so they dominate all blocks)
  "    v7 = iconst.i64 0\n" ++           -- 0
  "    v8 = iconst.i64 1\n" ++           -- 1
  "    v9 = iconst.i64 4\n" ++           -- 4
  "    v10 = iconst.i64 8\n" ++          -- 8
  "    v11 = iconst.i64 32\n" ++         -- 32
  "    v12 = iconst.i64 64\n" ++         -- 64 (block size)
  "    v13 = iconst.i64 0xFFFFFFFF\n" ++ -- mask32
  "    v14 = iconst.i64 3\n" ++          -- 3
  "    v15 = iconst.i64 16\n" ++         -- 16
  "    v16 = iconst.i64 7\n" ++          -- 7
  "    v17 = iconst.i64 10\n" ++         -- 10
  "    v18 = iconst.i64 13\n" ++         -- 13
  "    v19 = iconst.i64 24\n" ++         -- 24

  -- === Step 2: SHA-256 Padding ===
  -- Append 0x80 byte at file_data[file_size]
  "    v20 = iadd v2, v4\n" ++          -- data_off + file_size (relative)
  "    v21 = iadd v0, v20\n" ++         -- absolute addr
  "    v22 = iconst.i64 0x80\n" ++
  "    istore8 v22, v21\n" ++

  -- Compute padded_len: round up (file_size + 1 + 8) to next multiple of 64
  "    v23 = iadd v4, v8\n" ++           -- file_size + 1
  "    v24 = iadd v23, v10\n" ++         -- file_size + 1 + 8
  "    v25 = iconst.i64 63\n" ++
  "    v26 = iadd v24, v25\n" ++         -- file_size + 1 + 8 + 63
  "    v27 = band_not v26, v25\n" ++     -- round up to 64
  -- Store padded_len
  s!"    v28 = iconst.i64 {paddedLnOff}\n" ++
  "    v29 = iadd v0, v28\n" ++
  "    store v27, v29\n" ++

  -- Zero bytes from file_size+1 to padded_len-8
  "    v30 = isub v27, v10\n" ++         -- padded_len - 8
  "    jump block1(v23)\n" ++

  "\n" ++
  "block1(v40: i64):\n" ++
  "    v41 = icmp uge v40, v30\n" ++
  "    brif v41, block2, block3\n" ++

  "\n" ++
  "block3:\n" ++
  "    v42 = iadd v2, v40\n" ++
  "    v43 = iadd v0, v42\n" ++
  "    istore8 v7, v43\n" ++
  "    v44 = iadd v40, v8\n" ++
  "    jump block1(v44)\n" ++

  "\n" ++
  -- Write big-endian 64-bit bit length at padded_len - 8
  "block2:\n" ++
  "    v51 = ishl v4, v14\n" ++          -- bit_length = file_size << 3 (v14 = 3)

  -- Write 8 bytes big-endian
  "    v52 = iadd v2, v30\n" ++
  "    v53 = iadd v0, v52\n" ++

  "    v54 = iconst.i64 56\n" ++
  "    v55 = ushr v51, v54\n" ++
  "    istore8 v55, v53\n" ++

  "    v56 = iconst.i64 48\n" ++
  "    v57 = ushr v51, v56\n" ++
  "    v58 = iadd v53, v8\n" ++
  "    istore8 v57, v58\n" ++

  "    v59 = iconst.i64 40\n" ++
  "    v60 = ushr v51, v59\n" ++
  "    v61 = iadd v58, v8\n" ++
  "    istore8 v60, v61\n" ++

  "    v62 = ushr v51, v11\n" ++         -- bits 24..31 (v11 = 32)
  "    v63 = iadd v61, v8\n" ++
  "    istore8 v62, v63\n" ++

  "    v65 = ushr v51, v19\n" ++         -- v19 = 24
  "    v66 = iadd v63, v8\n" ++
  "    istore8 v65, v66\n" ++

  "    v68 = ushr v51, v15\n" ++         -- v15 = 16
  "    v69 = iadd v66, v8\n" ++
  "    istore8 v68, v69\n" ++

  "    v70 = ushr v51, v10\n" ++         -- v10 = 8
  "    v71 = iadd v69, v8\n" ++
  "    istore8 v70, v71\n" ++

  "    v72 = iadd v71, v8\n" ++
  "    istore8 v51, v72\n" ++            -- LSB

  -- Compute num_blocks = padded_len / 64
  "    v73 = iconst.i64 6\n" ++
  "    v74 = ushr v27, v73\n" ++         -- padded_len >> 6
  s!"    v75 = iconst.i64 {numBlkOff}\n" ++
  "    v76 = iadd v0, v75\n" ++
  "    store v74, v76\n" ++

  -- === Step 3: Copy H_initial → H_working ===
  s!"    v77 = iconst.i64 {hInitOff}\n" ++
  s!"    v78 = iconst.i64 {hWorkOff}\n" ++
  "    jump block10(v7)\n" ++

  "\n" ++
  "block10(v80: i64):\n" ++
  "    v81 = icmp uge v80, v10\n" ++     -- i >= 8?
  "    brif v81, block11, block12\n" ++

  "\n" ++
  "block12:\n" ++
  "    v82 = imul v80, v9\n" ++
  "    v83 = iadd v77, v82\n" ++
  "    v84 = iadd v0, v83\n" ++
  "    v85 = load.i32 v84\n" ++
  "    v86 = iadd v78, v82\n" ++
  "    v87 = iadd v0, v86\n" ++
  "    store v85, v87\n" ++
  "    v88 = iadd v80, v8\n" ++
  "    jump block10(v88)\n" ++

  "\n" ++
  -- === Step 4: Outer loop over blocks ===
  "block11:\n" ++
  "    jump block20(v7)\n" ++

  "\n" ++
  "block20(v100: i64):\n" ++
  "    v101 = icmp uge v100, v74\n" ++
  "    brif v101, block50, block21\n" ++

  "\n" ++
  -- === Load W[0..15] big-endian ===
  "block21:\n" ++
  "    v102 = imul v100, v12\n" ++       -- block_idx * 64
  "    v103 = iadd v2, v102\n" ++        -- data_off + block_idx * 64
  "    jump block22(v7, v103)\n" ++

  "\n" ++
  "block22(v110: i64, v111: i64):\n" ++
  "    v113 = icmp uge v110, v15\n" ++   -- w_i >= 16? (v15 = 16)
  "    brif v113, block26, block23\n" ++

  "\n" ++
  "block23:\n" ++
  "    v114 = imul v110, v9\n" ++        -- w_i * 4
  "    v115 = iadd v111, v114\n" ++
  "    v116 = iadd v0, v115\n" ++

  "    v117 = uload8.i64 v116\n" ++      -- b0
  "    v119 = ishl v117, v19\n" ++       -- b0 << 24 (v19 = 24)

  "    v120 = iadd v116, v8\n" ++
  "    v121 = uload8.i64 v120\n" ++      -- b1
  "    v122 = ishl v121, v15\n" ++       -- b1 << 16 (v15 = 16)

  "    v123 = iadd v120, v8\n" ++
  "    v124 = uload8.i64 v123\n" ++      -- b2
  "    v125 = ishl v124, v10\n" ++       -- b2 << 8 (v10 = 8)

  "    v126 = iadd v123, v8\n" ++
  "    v127 = uload8.i64 v126\n" ++      -- b3

  "    v128 = bor v119, v122\n" ++
  "    v129 = bor v128, v125\n" ++
  "    v130 = bor v129, v127\n" ++

  "    v131 = ireduce.i32 v130\n" ++
  s!"    v132 = iconst.i64 {wOff}\n" ++
  "    v133 = iadd v132, v114\n" ++
  "    v134 = iadd v0, v133\n" ++
  "    store v131, v134\n" ++

  "    v135 = iadd v110, v8\n" ++
  "    jump block22(v135, v111)\n" ++

  "\n" ++
  -- === Expand W[16..63] ===
  "block26:\n" ++
  "    jump block27(v15)\n" ++           -- i = 16 (v15 = 16)

  "\n" ++
  "block27(v150: i64):\n" ++
  "    v152 = icmp uge v150, v12\n" ++   -- i >= 64? (v12 = 64)
  "    brif v152, block30, block28\n" ++

  "\n" ++
  "block28:\n" ++
  -- Load W[i-2]
  "    v153 = iconst.i64 2\n" ++
  "    v154 = isub v150, v153\n" ++
  "    v155 = imul v154, v9\n" ++
  s!"    v156 = iconst.i64 {wOff}\n" ++
  "    v157 = iadd v156, v155\n" ++
  "    v158 = iadd v0, v157\n" ++
  "    v159 = uload32.i64 v158\n" ++

  -- sigma1(x): rotr(x,17) ^ rotr(x,19) ^ (x >> 10)
  "    v160 = iconst.i64 17\n" ++
  "    v161 = ushr v159, v160\n" ++
  "    v162 = iconst.i64 15\n" ++
  "    v163 = ishl v159, v162\n" ++
  "    v164 = bor v161, v163\n" ++
  "    v165 = band v164, v13\n" ++       -- rotr(x,17)

  "    v166 = iconst.i64 19\n" ++
  "    v167 = ushr v159, v166\n" ++
  "    v168 = ishl v159, v18\n" ++       -- x << 13 (v18 = 13)
  "    v170 = bor v167, v168\n" ++
  "    v171 = band v170, v13\n" ++       -- rotr(x,19)

  "    v174 = ushr v159, v17\n" ++       -- x >> 10 (v17 = 10)

  "    v175 = bxor v165, v171\n" ++
  "    v176 = bxor v175, v174\n" ++      -- sigma1

  -- Load W[i-7]
  "    v178 = isub v150, v16\n" ++       -- i-7 (v16 = 7)
  "    v179 = imul v178, v9\n" ++
  "    v180 = iadd v156, v179\n" ++
  "    v181 = iadd v0, v180\n" ++
  "    v182 = uload32.i64 v181\n" ++

  -- Load W[i-15]
  "    v183 = isub v150, v162\n" ++      -- i-15 (v162 = 15)
  "    v184 = imul v183, v9\n" ++
  "    v185 = iadd v156, v184\n" ++
  "    v186 = iadd v0, v185\n" ++
  "    v187 = uload32.i64 v186\n" ++

  -- sigma0(x): rotr(x,7) ^ rotr(x,18) ^ (x >> 3)
  "    v188 = ushr v187, v16\n" ++       -- x >> 7 (v16 = 7)
  "    v189 = iconst.i64 25\n" ++
  "    v190 = ishl v187, v189\n" ++      -- x << 25
  "    v191 = bor v188, v190\n" ++
  "    v192 = band v191, v13\n" ++       -- rotr(x,7)

  "    v193 = iconst.i64 18\n" ++
  "    v194 = ushr v187, v193\n" ++
  "    v195 = iconst.i64 14\n" ++
  "    v196 = ishl v187, v195\n" ++
  "    v197 = bor v194, v196\n" ++
  "    v198 = band v197, v13\n" ++       -- rotr(x,18)

  "    v199 = ushr v187, v14\n" ++       -- x >> 3 (v14 = 3)

  "    v200 = bxor v192, v198\n" ++
  "    v201 = bxor v200, v199\n" ++      -- sigma0

  -- Load W[i-16]
  "    v202 = isub v150, v15\n" ++       -- i-16 (v15 = 16)
  "    v203 = imul v202, v9\n" ++
  "    v204 = iadd v156, v203\n" ++
  "    v205 = iadd v0, v204\n" ++
  "    v206 = uload32.i64 v205\n" ++

  -- W[i] = (sigma1 + W[i-7] + sigma0 + W[i-16]) & mask32
  "    v207 = iadd v176, v182\n" ++
  "    v208 = iadd v207, v201\n" ++
  "    v209 = iadd v208, v206\n" ++
  "    v210 = band v209, v13\n" ++

  "    v211 = imul v150, v9\n" ++
  "    v212 = iadd v156, v211\n" ++
  "    v213 = iadd v0, v212\n" ++
  "    v214 = ireduce.i32 v210\n" ++
  "    store v214, v213\n" ++

  "    v215 = iadd v150, v8\n" ++
  "    jump block27(v215)\n" ++

  "\n" ++
  -- === Init working vars a-h ===
  "block30:\n" ++
  s!"    v220 = iconst.i64 {hWorkOff}\n" ++
  "    v221 = iadd v0, v220\n" ++
  "    v222 = uload32.i64 v221\n" ++     -- a
  "    v223 = iadd v221, v9\n" ++
  "    v224 = uload32.i64 v223\n" ++     -- b
  "    v225 = iadd v223, v9\n" ++
  "    v226 = uload32.i64 v225\n" ++     -- c
  "    v227 = iadd v225, v9\n" ++
  "    v228 = uload32.i64 v227\n" ++     -- d
  "    v229 = iadd v227, v9\n" ++
  "    v230 = uload32.i64 v229\n" ++     -- e
  "    v231 = iadd v229, v9\n" ++
  "    v232 = uload32.i64 v231\n" ++     -- f
  "    v233 = iadd v231, v9\n" ++
  "    v234 = uload32.i64 v233\n" ++     -- g
  "    v235 = iadd v233, v9\n" ++
  "    v236 = uload32.i64 v235\n" ++     -- h
  "    jump block31(v7, v222, v224, v226, v228, v230, v232, v234, v236)\n" ++

  "\n" ++
  -- === 64-round compression loop ===
  "block31(v240: i64, v241: i64, v242: i64, v243: i64, v244: i64, v245: i64, v246: i64, v247: i64, v248: i64):\n" ++
  "    v250 = icmp uge v240, v12\n" ++   -- round >= 64? (v12 = 64)
  "    brif v250, block37(v100, v241, v242, v243, v244, v245, v246, v247, v248), block32\n" ++

  "\n" ++
  "block32:\n" ++
  -- Sigma1(e) = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25)
  "    v251 = iconst.i64 6\n" ++
  "    v252 = ushr v245, v251\n" ++
  "    v253 = iconst.i64 26\n" ++
  "    v254 = ishl v245, v253\n" ++
  "    v255 = bor v252, v254\n" ++
  "    v256 = band v255, v13\n" ++       -- rotr(e,6)

  "    v257 = iconst.i64 11\n" ++
  "    v258 = ushr v245, v257\n" ++
  "    v259 = iconst.i64 21\n" ++
  "    v260 = ishl v245, v259\n" ++
  "    v261 = bor v258, v260\n" ++
  "    v262 = band v261, v13\n" ++       -- rotr(e,11)

  "    v263 = iconst.i64 25\n" ++
  "    v264 = ushr v245, v263\n" ++
  "    v265 = ishl v245, v16\n" ++       -- e << 7 (v16 = 7)
  "    v266 = bor v264, v265\n" ++
  "    v267 = band v266, v13\n" ++       -- rotr(e,25)

  "    v268 = bxor v256, v262\n" ++
  "    v269 = bxor v268, v267\n" ++      -- Sigma1(e)

  -- Ch(e,f,g) = (e & f) ^ (~e & g)
  "    v270 = band v245, v246\n" ++
  "    v271 = bxor v245, v13\n" ++       -- ~e
  "    v272 = band v271, v247\n" ++
  "    v273 = bxor v270, v272\n" ++      -- Ch

  -- Load K[round_i]
  s!"    v274 = iconst.i64 {kOff}\n" ++
  "    v275 = imul v240, v9\n" ++
  "    v276 = iadd v274, v275\n" ++
  "    v277 = iadd v0, v276\n" ++
  "    v278 = uload32.i64 v277\n" ++

  -- Load W[round_i]
  s!"    v279 = iconst.i64 {wOff}\n" ++
  "    v280 = iadd v279, v275\n" ++
  "    v281 = iadd v0, v280\n" ++
  "    v282 = uload32.i64 v281\n" ++

  -- temp1 = (h + Sigma1 + Ch + K[i] + W[i]) & mask32
  "    v283 = iadd v248, v269\n" ++
  "    v284 = iadd v283, v273\n" ++
  "    v285 = iadd v284, v278\n" ++
  "    v286 = iadd v285, v282\n" ++
  "    v287 = band v286, v13\n" ++       -- temp1

  -- Sigma0(a) = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22)
  "    v288 = iconst.i64 2\n" ++
  "    v289 = ushr v241, v288\n" ++
  "    v290 = iconst.i64 30\n" ++
  "    v291 = ishl v241, v290\n" ++
  "    v292 = bor v289, v291\n" ++
  "    v293 = band v292, v13\n" ++       -- rotr(a,2)

  "    v294 = ushr v241, v18\n" ++       -- a >> 13 (v18 = 13)
  "    v295 = iconst.i64 19\n" ++
  "    v296 = ishl v241, v295\n" ++
  "    v297 = bor v294, v296\n" ++
  "    v298 = band v297, v13\n" ++       -- rotr(a,13)

  "    v299 = iconst.i64 22\n" ++
  "    v300 = ushr v241, v299\n" ++
  "    v301 = ishl v241, v17\n" ++       -- a << 10 (v17 = 10)
  "    v302 = bor v300, v301\n" ++
  "    v303 = band v302, v13\n" ++       -- rotr(a,22)

  "    v304 = bxor v293, v298\n" ++
  "    v305 = bxor v304, v303\n" ++      -- Sigma0(a)

  -- Maj(a,b,c) = (a & b) ^ (a & c) ^ (b & c)
  "    v306 = band v241, v242\n" ++
  "    v307 = band v241, v243\n" ++
  "    v308 = band v242, v243\n" ++
  "    v309 = bxor v306, v307\n" ++
  "    v310 = bxor v309, v308\n" ++      -- Maj

  -- temp2 = (Sigma0 + Maj) & mask32
  "    v311 = iadd v305, v310\n" ++
  "    v312 = band v311, v13\n" ++

  -- new_e = (d + temp1) & mask32, new_a = (temp1 + temp2) & mask32
  "    v313 = iadd v244, v287\n" ++
  "    v314 = band v313, v13\n" ++       -- new_e
  "    v315 = iadd v287, v312\n" ++
  "    v316 = band v315, v13\n" ++       -- new_a

  "    v317 = iadd v240, v8\n" ++
  "    jump block31(v317, v316, v241, v242, v243, v314, v245, v246, v247)\n" ++

  "\n" ++
  -- === Add a-h back to H_working ===
  "block37(v320: i64, v321: i64, v322: i64, v323: i64, v324: i64, v325: i64, v326: i64, v327: i64, v328: i64):\n" ++
  s!"    v329 = iconst.i64 {hWorkOff}\n" ++

  "    v330 = iadd v0, v329\n" ++
  "    v331 = uload32.i64 v330\n" ++
  "    v332 = iadd v331, v321\n" ++
  "    v333 = band v332, v13\n" ++
  "    v334 = ireduce.i32 v333\n" ++
  "    store v334, v330\n" ++

  "    v335 = iadd v330, v9\n" ++
  "    v336 = uload32.i64 v335\n" ++
  "    v337 = iadd v336, v322\n" ++
  "    v338 = band v337, v13\n" ++
  "    v339 = ireduce.i32 v338\n" ++
  "    store v339, v335\n" ++

  "    v340 = iadd v335, v9\n" ++
  "    v341 = uload32.i64 v340\n" ++
  "    v342 = iadd v341, v323\n" ++
  "    v343 = band v342, v13\n" ++
  "    v344 = ireduce.i32 v343\n" ++
  "    store v344, v340\n" ++

  "    v345 = iadd v340, v9\n" ++
  "    v346 = uload32.i64 v345\n" ++
  "    v347 = iadd v346, v324\n" ++
  "    v348 = band v347, v13\n" ++
  "    v349 = ireduce.i32 v348\n" ++
  "    store v349, v345\n" ++

  "    v350 = iadd v345, v9\n" ++
  "    v351 = uload32.i64 v350\n" ++
  "    v352 = iadd v351, v325\n" ++
  "    v353 = band v352, v13\n" ++
  "    v354 = ireduce.i32 v353\n" ++
  "    store v354, v350\n" ++

  "    v355 = iadd v350, v9\n" ++
  "    v356 = uload32.i64 v355\n" ++
  "    v357 = iadd v356, v326\n" ++
  "    v358 = band v357, v13\n" ++
  "    v359 = ireduce.i32 v358\n" ++
  "    store v359, v355\n" ++

  "    v360 = iadd v355, v9\n" ++
  "    v361 = uload32.i64 v360\n" ++
  "    v362 = iadd v361, v327\n" ++
  "    v363 = band v362, v13\n" ++
  "    v364 = ireduce.i32 v363\n" ++
  "    store v364, v360\n" ++

  "    v365 = iadd v360, v9\n" ++
  "    v366 = uload32.i64 v365\n" ++
  "    v367 = iadd v366, v328\n" ++
  "    v368 = band v367, v13\n" ++
  "    v369 = ireduce.i32 v368\n" ++
  "    store v369, v365\n" ++

  "    v370 = iadd v320, v8\n" ++
  "    jump block20(v370)\n" ++

  "\n" ++
  -- === Step 6: Hex formatting ===
  "block50:\n" ++
  s!"    v400 = iconst.i64 {hWorkOff}\n" ++
  s!"    v401 = iconst.i64 {hexOutOff}\n" ++
  s!"    v402 = iconst.i64 {hexTblOff}\n" ++
  "    jump block51(v7, v7)\n" ++

  "\n" ++
  "block51(v410: i64, v411: i64):\n" ++
  "    v412 = icmp uge v410, v10\n" ++   -- word_i >= 8?
  "    brif v412, block55(v411), block52\n" ++

  "\n" ++
  "block52:\n" ++
  "    v413 = imul v410, v9\n" ++
  "    v414 = iadd v400, v413\n" ++
  "    v415 = iadd v0, v414\n" ++
  "    v416 = uload32.i64 v415\n" ++
  "    jump block53(v416, v7, v411)\n" ++

  "\n" ++
  "block53(v420: i64, v421: i64, v422: i64):\n" ++
  "    v423 = icmp uge v421, v9\n" ++    -- byte_i >= 4?
  "    brif v423, block54(v422), block56\n" ++

  "\n" ++
  "block56:\n" ++
  "    v424 = iconst.i64 3\n" ++
  "    v425 = isub v424, v421\n" ++
  "    v426 = imul v425, v10\n" ++       -- (3 - byte_i) * 8
  "    v427 = ushr v420, v426\n" ++
  "    v428 = iconst.i64 0xFF\n" ++
  "    v429 = band v427, v428\n" ++

  "    v430 = ushr v429, v9\n" ++        -- byte >> 4
  "    v431 = iadd v402, v430\n" ++
  "    v432 = iadd v0, v431\n" ++
  "    v433 = uload8.i64 v432\n" ++
  "    v434 = iadd v401, v422\n" ++
  "    v435 = iadd v0, v434\n" ++
  "    istore8 v433, v435\n" ++

  "    v436 = iconst.i64 0x0F\n" ++
  "    v437 = band v429, v436\n" ++
  "    v438 = iadd v402, v437\n" ++
  "    v439 = iadd v0, v438\n" ++
  "    v440 = uload8.i64 v439\n" ++
  "    v441 = iadd v422, v8\n" ++
  "    v442 = iadd v401, v441\n" ++
  "    v443 = iadd v0, v442\n" ++
  "    istore8 v440, v443\n" ++

  "    v444 = iadd v441, v8\n" ++
  "    v445 = iadd v421, v8\n" ++
  "    jump block53(v420, v445, v444)\n" ++

  "\n" ++
  "block54(v450: i64):\n" ++
  "    v451 = iadd v410, v8\n" ++
  "    jump block51(v451, v450)\n" ++

  "\n" ++
  "block55(v460: i64):\n" ++
  "    v461 = iadd v401, v460\n" ++
  "    v462 = iadd v0, v461\n" ++
  "    v463 = iconst.i64 10\n" ++
  "    istore8 v463, v462\n" ++

  "    v464 = iadd v460, v8\n" ++
  s!"    v465 = iconst.i64 {outFname}\n" ++
  s!"    v466 = iconst.i64 {hexOutOff}\n" ++
  "    v467 = call fn1(v0, v465, v466, v7, v464)\n" ++
  "    return\n" ++
  "}\n"

-- ---------------------------------------------------------------------------
-- Payload construction
-- ---------------------------------------------------------------------------

def payloads : List UInt8 :=
  let reserved := zeros inputFilename_off
  -- Input filename placeholder (will be patched at runtime)
  let inputFname := padTo (stringToBytes "input.bin") (outputFilename_off - inputFilename_off)
  -- Output filename
  let outputFname := padTo (stringToBytes "sha256_output.txt") (hexOutput_off - outputFilename_off)
  -- Hex output region (zeros, will be written at runtime)
  let hexOutRegion := zeros (K_off - hexOutput_off)
  -- K constants (64 x u32 LE)
  let kBytes := (kConstants.map uint32ToBytes).flatten
  -- Pad to H_init_off
  let kPad := zeros (H_init_off - K_off - kBytes.length)
  -- H initial values (8 x u32 LE)
  let hBytes := (hInitial.map uint32ToBytes).flatten
  -- Pad to H_work_off
  let hPad := zeros (H_work_off - H_init_off - hBytes.length)
  -- H working region (zeros, will be written at runtime)
  let hWorkRegion := zeros (W_off - H_work_off)
  -- W message schedule region (zeros)
  let wRegion := zeros (hexTable_off - W_off)
  -- Hex lookup table
  let hexTable := "0123456789abcdef".toUTF8.toList
  -- Pad to fileData_off
  let hexPad := zeros (fileData_off - hexTable_off - hexTable.length)
  reserved ++ inputFname ++ outputFname ++ hexOutRegion ++
    kBytes ++ kPad ++ hBytes ++ hPad ++ hWorkRegion ++ wRegion ++
    hexTable ++ hexPad

-- ---------------------------------------------------------------------------
-- Configuration
-- ---------------------------------------------------------------------------

def sha256Config : BaseConfig := {
  cranelift_ir := clifIrSource,
  memory_size := totalMemory,
  context_offset := 0
}

def sha256Algorithm : Algorithm :=
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
  let json := toJsonPair Algorithm.sha256Config Algorithm.sha256Algorithm
  IO.println (Json.compress json)
