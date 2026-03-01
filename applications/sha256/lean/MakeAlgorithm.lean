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

open AlgorithmLib.IR

-- Bundle of commonly-used constants from block0
structure Consts where
  ptr : Val
  zero : Val
  c1 : Val
  c3 : Val
  c4 : Val
  c7 : Val
  c8 : Val
  c10 : Val
  c13 : Val
  c16 : Val
  c24 : Val
  c32 : Val
  c64 : Val
  mask32 : Val
  dataOff : Val

-- Step 2: SHA-256 padding — returns (paddedLen, numBlocks)
def emitPadding (k : Consts) (fileSize : Val) : IRBuilder (Val × Val) := do
  -- Append 0x80 byte at file_data[file_size]
  let relOff ← iadd k.dataOff fileSize
  let absOff ← iadd k.ptr relOff
  let x80 ← iconst64 0x80
  istore8 x80 absOff

  -- Compute padded_len: round up (file_size + 1 + 8) to next multiple of 64
  let fsp1 ← iadd fileSize k.c1
  let fsp9 ← iadd fsp1 k.c8
  let c63 ← iconst64 63
  let sum ← iadd fsp9 c63
  let paddedLen ← bandNot sum c63

  storeAt k.ptr paddedLen_off paddedLen

  -- Zero bytes from file_size+1 to padded_len-8
  let zeroEnd ← isub paddedLen k.c8

  -- Zero-fill loop
  let zeroHdr ← declareBlock [.i64]
  let zi := zeroHdr.param 0
  let zeroDone ← declareBlock []
  let zeroBody ← declareBlock []
  jump zeroHdr.ref [fsp1]
  startBlock zeroHdr
  let ziDone ← icmp .uge zi zeroEnd
  brif ziDone zeroDone.ref [] zeroBody.ref []
  startBlock zeroBody
  let zrel ← iadd k.dataOff zi
  let zabs ← iadd k.ptr zrel
  istore8 k.zero zabs
  let ziNext ← iadd zi k.c1
  jump zeroHdr.ref [ziNext]

  -- Write big-endian 64-bit bit length at padded_len - 8
  startBlock zeroDone
  let bitLen ← ishl fileSize k.c3
  let beRel ← iadd k.dataOff zeroEnd
  let beAddr ← iadd k.ptr beRel

  let c56 ← iconst64 56
  let b0 ← ushr bitLen c56
  istore8 b0 beAddr

  let c48 ← iconst64 48
  let b1v ← ushr bitLen c48
  let a1 ← iadd beAddr k.c1
  istore8 b1v a1

  let c40 ← iconst64 40
  let b2v ← ushr bitLen c40
  let a2 ← iadd a1 k.c1
  istore8 b2v a2

  let b3v ← ushr bitLen k.c32
  let a3 ← iadd a2 k.c1
  istore8 b3v a3

  let b4v ← ushr bitLen k.c24
  let a4 ← iadd a3 k.c1
  istore8 b4v a4

  let b5v ← ushr bitLen k.c16
  let a5 ← iadd a4 k.c1
  istore8 b5v a5

  let b6v ← ushr bitLen k.c8
  let a6 ← iadd a5 k.c1
  istore8 b6v a6

  let a7 ← iadd a6 k.c1
  istore8 bitLen a7

  -- Compute num_blocks = padded_len / 64
  let c6 ← iconst64 6
  let numBlocks ← ushr paddedLen c6
  storeAt k.ptr numBlocks_off numBlocks

  pure (paddedLen, numBlocks)

-- Step 3: Copy H_initial → H_working
def emitCopyH (k : Consts) : IRBuilder Unit := do
  let hInitBase ← iconst64 H_init_off
  let hWorkBase ← iconst64 H_work_off
  let copyHdr ← declareBlock [.i64]
  let ci := copyHdr.param 0
  let copyDone ← declareBlock []
  let copyBody ← declareBlock []
  jump copyHdr.ref [k.zero]
  startBlock copyHdr
  let ciDone ← icmp .uge ci k.c8
  brif ciDone copyDone.ref [] copyBody.ref []
  startBlock copyBody
  let byteOff ← imul ci k.c4
  let srcRel ← iadd hInitBase byteOff
  let srcAbs ← iadd k.ptr srcRel
  let val ← load32 srcAbs
  let dstRel ← iadd hWorkBase byteOff
  let dstAbs ← iadd k.ptr dstRel
  store val dstAbs
  let ciNext ← iadd ci k.c1
  jump copyHdr.ref [ciNext]
  startBlock copyDone

-- Load W[0..15] big-endian from message block
def emitLoadW (k : Consts) (blkBase : Val) : IRBuilder Unit := do
  let wHdr ← declareBlock [.i64, .i64]
  let wi := wHdr.param 0
  let wBase := wHdr.param 1
  let wDone ← declareBlock []
  let wBody ← declareBlock []
  jump wHdr.ref [k.zero, blkBase]
  startBlock wHdr
  let wiDone ← icmp .uge wi k.c16
  brif wiDone wDone.ref [] wBody.ref []

  startBlock wBody
  let wi4 ← imul wi k.c4
  let bRel ← iadd wBase wi4
  let bAbs ← iadd k.ptr bRel

  let byte0 ← uload8_64 bAbs
  let s0 ← ishl byte0 k.c24
  let a1 ← iadd bAbs k.c1
  let byte1 ← uload8_64 a1
  let s1 ← ishl byte1 k.c16
  let a2 ← iadd a1 k.c1
  let byte2 ← uload8_64 a2
  let s2 ← ishl byte2 k.c8
  let a3 ← iadd a2 k.c1
  let byte3 ← uload8_64 a3
  let or1 ← bor s0 s1
  let or2 ← bor or1 s2
  let or3 ← bor or2 byte3
  let w32 ← ireduce32 or3
  let wOffC ← iconst64 W_off
  let wRel ← iadd wOffC wi4
  let wAbs ← iadd k.ptr wRel
  store w32 wAbs
  let wiNext ← iadd wi k.c1
  jump wHdr.ref [wiNext, wBase]

  startBlock wDone

-- Expand W[16..63]
def emitExpandW (k : Consts) : IRBuilder Unit := do
  let expHdr ← declareBlock [.i64]
  let ei := expHdr.param 0
  let expDone ← declareBlock []
  let expBody ← declareBlock []
  jump expHdr.ref [k.c16]
  startBlock expHdr
  let eiDone ← icmp .uge ei k.c64
  brif eiDone expDone.ref [] expBody.ref []

  startBlock expBody
  let c2 ← iconst64 2
  let im2 ← isub ei c2
  let im2x4 ← imul im2 k.c4
  let wOff' ← iconst64 W_off
  let wi2Rel ← iadd wOff' im2x4
  let wi2Abs ← iadd k.ptr wi2Rel
  let wi2val ← uload32_64 wi2Abs

  -- sigma1(x): rotr(x,17) ^ rotr(x,19) ^ (x >> 10)
  let c17 ← iconst64 17
  let r17a ← ushr wi2val c17
  let c15 ← iconst64 15
  let r17b ← ishl wi2val c15
  let r17 ← bor r17a r17b
  let rotr17 ← band r17 k.mask32

  let c19 ← iconst64 19
  let r19a ← ushr wi2val c19
  let r19b ← ishl wi2val k.c13
  let r19 ← bor r19a r19b
  let rotr19 ← band r19 k.mask32

  let shr10 ← ushr wi2val k.c10

  let sig1a ← bxor rotr17 rotr19
  let sigma1 ← bxor sig1a shr10

  -- Load W[i-7]
  let im7 ← isub ei k.c7
  let im7x4 ← imul im7 k.c4
  let wi7Rel ← iadd wOff' im7x4
  let wi7Abs ← iadd k.ptr wi7Rel
  let wi7val ← uload32_64 wi7Abs

  -- Load W[i-15]
  let im15 ← isub ei c15
  let im15x4 ← imul im15 k.c4
  let wi15Rel ← iadd wOff' im15x4
  let wi15Abs ← iadd k.ptr wi15Rel
  let wi15val ← uload32_64 wi15Abs

  -- sigma0(x): rotr(x,7) ^ rotr(x,18) ^ (x >> 3)
  let r7a ← ushr wi15val k.c7
  let c25 ← iconst64 25
  let r7b ← ishl wi15val c25
  let r7 ← bor r7a r7b
  let rotr7 ← band r7 k.mask32

  let c18 ← iconst64 18
  let r18a ← ushr wi15val c18
  let c14 ← iconst64 14
  let r18b ← ishl wi15val c14
  let r18 ← bor r18a r18b
  let rotr18 ← band r18 k.mask32

  let shr3 ← ushr wi15val k.c3

  let sig0a ← bxor rotr7 rotr18
  let sigma0 ← bxor sig0a shr3

  -- Load W[i-16]
  let im16 ← isub ei k.c16
  let im16x4 ← imul im16 k.c4
  let wi16Rel ← iadd wOff' im16x4
  let wi16Abs ← iadd k.ptr wi16Rel
  let wi16val ← uload32_64 wi16Abs

  -- W[i] = (sigma1 + W[i-7] + sigma0 + W[i-16]) & mask32
  let ws1 ← iadd sigma1 wi7val
  let ws2 ← iadd ws1 sigma0
  let ws3 ← iadd ws2 wi16val
  let wNew ← band ws3 k.mask32
  let eix4 ← imul ei k.c4
  let wiNewRel ← iadd wOff' eix4
  let wiNewAbs ← iadd k.ptr wiNewRel
  let wNew32 ← ireduce32 wNew
  store wNew32 wiNewAbs
  let eiNext ← iadd ei k.c1
  jump expHdr.ref [eiNext]

  startBlock expDone

-- 64-round compression loop body
-- Returns (addBackBlk, roundHdr) for wiring
def emitCompressionRound (k : Consts) (blkIdx : Val)
    (va vb vc vd ve vf vg vh : Val)
    (addBackBlk : DeclaredBlock) : IRBuilder DeclaredBlock := do
  let roundHdr ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64, .i64, .i64, .i64]
  let ri := roundHdr.param 0
  let ra := roundHdr.param 1
  let rb := roundHdr.param 2
  let rc := roundHdr.param 3
  let rd := roundHdr.param 4
  let re := roundHdr.param 5
  let rf := roundHdr.param 6
  let rg := roundHdr.param 7
  let rh := roundHdr.param 8
  let roundBody ← declareBlock []

  jump roundHdr.ref [k.zero, va, vb, vc, vd, ve, vf, vg, vh]
  startBlock roundHdr
  let roundDone ← icmp .uge ri k.c64
  brif roundDone addBackBlk.ref [blkIdx, ra, rb, rc, rd, re, rf, rg, rh] roundBody.ref []

  startBlock roundBody
  -- Sigma1(e) = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25)
  let c6 ← iconst64 6
  let re6a ← ushr re c6
  let c26 ← iconst64 26
  let re6b ← ishl re c26
  let re6 ← bor re6a re6b
  let rotrE6 ← band re6 k.mask32

  let c11 ← iconst64 11
  let re11a ← ushr re c11
  let c21 ← iconst64 21
  let re11b ← ishl re c21
  let re11 ← bor re11a re11b
  let rotrE11 ← band re11 k.mask32

  let c25 ← iconst64 25
  let re25a ← ushr re c25
  let re25b ← ishl re k.c7
  let re25 ← bor re25a re25b
  let rotrE25 ← band re25 k.mask32

  let sig1e1 ← bxor rotrE6 rotrE11
  let sigma1e ← bxor sig1e1 rotrE25

  -- Ch(e,f,g) = (e & f) ^ (~e & g)
  let ef ← band re rf
  let notE ← bxor re k.mask32
  let notEg ← band notE rg
  let ch ← bxor ef notEg

  -- Load K[round_i] and W[round_i]
  let kBase ← iconst64 K_off
  let ri4 ← imul ri k.c4
  let kRel ← iadd kBase ri4
  let kAbs ← iadd k.ptr kRel
  let kVal ← uload32_64 kAbs

  let wBase ← iconst64 W_off
  let wRiRel ← iadd wBase ri4
  let wRiAbs ← iadd k.ptr wRiRel
  let wVal ← uload32_64 wRiAbs

  -- temp1 = (h + Sigma1 + Ch + K[i] + W[i]) & mask32
  let t1a ← iadd rh sigma1e
  let t1b ← iadd t1a ch
  let t1c ← iadd t1b kVal
  let t1d ← iadd t1c wVal
  let temp1 ← band t1d k.mask32

  -- Sigma0(a) = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22)
  let c2 ← iconst64 2
  let ra2a ← ushr ra c2
  let c30 ← iconst64 30
  let ra2b ← ishl ra c30
  let ra2 ← bor ra2a ra2b
  let rotrA2 ← band ra2 k.mask32

  let ra13a ← ushr ra k.c13
  let c19 ← iconst64 19
  let ra13b ← ishl ra c19
  let ra13 ← bor ra13a ra13b
  let rotrA13 ← band ra13 k.mask32

  let c22 ← iconst64 22
  let ra22a ← ushr ra c22
  let ra22b ← ishl ra k.c10
  let ra22 ← bor ra22a ra22b
  let rotrA22 ← band ra22 k.mask32

  let sig0a1 ← bxor rotrA2 rotrA13
  let sigma0a ← bxor sig0a1 rotrA22

  -- Maj(a,b,c) = (a & b) ^ (a & c) ^ (b & c)
  let ab ← band ra rb
  let ac ← band ra rc
  let bc ← band rb rc
  let m1 ← bxor ab ac
  let maj ← bxor m1 bc

  -- temp2 = (Sigma0 + Maj) & mask32
  let t2a ← iadd sigma0a maj
  let temp2 ← band t2a k.mask32

  -- new_e = (d + temp1) & mask32, new_a = (temp1 + temp2) & mask32
  let newEa ← iadd rd temp1
  let newE ← band newEa k.mask32
  let newAa ← iadd temp1 temp2
  let newA ← band newAa k.mask32

  let riNext ← iadd ri k.c1
  jump roundHdr.ref [riNext, newA, ra, rb, rc, newE, re, rf, rg]

  pure roundHdr

-- Add a-h back to H_working, then jump to outer loop
def emitAddBack (k : Consts) (addBackBlk : DeclaredBlock)
    (outerHdr : DeclaredBlock) : IRBuilder Unit := do
  startBlock addBackBlk
  let abBlkIdx := addBackBlk.param 0
  let abA := addBackBlk.param 1
  let abB := addBackBlk.param 2
  let abC := addBackBlk.param 3
  let abD := addBackBlk.param 4
  let abE := addBackBlk.param 5
  let abF := addBackBlk.param 6
  let abG := addBackBlk.param 7
  let abH := addBackBlk.param 8

  let hBase ← iconst64 H_work_off
  let hAddr0 ← iadd k.ptr hBase
  let oldH0 ← uload32_64 hAddr0
  let newH0 ← iadd oldH0 abA
  let mH0 ← band newH0 k.mask32
  let rH0 ← ireduce32 mH0
  store rH0 hAddr0

  let hAddr1 ← iadd hAddr0 k.c4
  let oldH1 ← uload32_64 hAddr1
  let newH1 ← iadd oldH1 abB
  let mH1 ← band newH1 k.mask32
  let rH1 ← ireduce32 mH1
  store rH1 hAddr1

  let hAddr2 ← iadd hAddr1 k.c4
  let oldH2 ← uload32_64 hAddr2
  let newH2 ← iadd oldH2 abC
  let mH2 ← band newH2 k.mask32
  let rH2 ← ireduce32 mH2
  store rH2 hAddr2

  let hAddr3 ← iadd hAddr2 k.c4
  let oldH3 ← uload32_64 hAddr3
  let newH3 ← iadd oldH3 abD
  let mH3 ← band newH3 k.mask32
  let rH3 ← ireduce32 mH3
  store rH3 hAddr3

  let hAddr4 ← iadd hAddr3 k.c4
  let oldH4 ← uload32_64 hAddr4
  let newH4 ← iadd oldH4 abE
  let mH4 ← band newH4 k.mask32
  let rH4 ← ireduce32 mH4
  store rH4 hAddr4

  let hAddr5 ← iadd hAddr4 k.c4
  let oldH5 ← uload32_64 hAddr5
  let newH5 ← iadd oldH5 abF
  let mH5 ← band newH5 k.mask32
  let rH5 ← ireduce32 mH5
  store rH5 hAddr5

  let hAddr6 ← iadd hAddr5 k.c4
  let oldH6 ← uload32_64 hAddr6
  let newH6 ← iadd oldH6 abG
  let mH6 ← band newH6 k.mask32
  let rH6 ← ireduce32 mH6
  store rH6 hAddr6

  let hAddr7 ← iadd hAddr6 k.c4
  let oldH7 ← uload32_64 hAddr7
  let newH7 ← iadd oldH7 abH
  let mH7 ← band newH7 k.mask32
  let rH7 ← ireduce32 mH7
  store rH7 hAddr7

  let nextBlkIdx ← iadd abBlkIdx k.c1
  jump outerHdr.ref [nextBlkIdx]

-- Step 6: Hex formatting and file output
def emitHexFormat (k : Consts) (fnWrite : FnRef) (hexBlk : DeclaredBlock) : IRBuilder Unit := do
  startBlock hexBlk
  let hWorkC ← iconst64 H_work_off
  let hexOutC ← iconst64 hexOutput_off
  let hexTblC ← iconst64 hexTable_off

  let hexOuter ← declareBlock [.i64, .i64]
  let hwi := hexOuter.param 0
  let hcp := hexOuter.param 1
  let hexDone ← declareBlock [.i64]
  let hexWordBody ← declareBlock []
  jump hexOuter.ref [k.zero, k.zero]
  startBlock hexOuter
  let hwiDone ← icmp .uge hwi k.c8
  brif hwiDone hexDone.ref [hcp] hexWordBody.ref []

  startBlock hexWordBody
  let hwOff ← imul hwi k.c4
  let hwRel ← iadd hWorkC hwOff
  let hwAbs ← iadd k.ptr hwRel
  let wordVal ← uload32_64 hwAbs

  let hexInner ← declareBlock [.i64, .i64, .i64]
  let hWord := hexInner.param 0
  let hbi := hexInner.param 1
  let hcpI := hexInner.param 2
  let hexInnerDone ← declareBlock [.i64]
  let hexByteBody ← declareBlock []
  jump hexInner.ref [wordVal, k.zero, hcp]
  startBlock hexInner
  let hbiDone ← icmp .uge hbi k.c4
  brif hbiDone hexInnerDone.ref [hcpI] hexByteBody.ref []

  startBlock hexByteBody
  let c3' ← iconst64 3
  let shift ← isub c3' hbi
  let shiftBits ← imul shift k.c8
  let shifted ← ushr hWord shiftBits
  let cFF ← iconst64 0xFF
  let byteVal ← band shifted cFF

  let hiNib ← ushr byteVal k.c4
  let hiRel ← iadd hexTblC hiNib
  let hiAbs ← iadd k.ptr hiRel
  let hiChar ← uload8_64 hiAbs
  let hcRel ← iadd hexOutC hcpI
  let hcAbs ← iadd k.ptr hcRel
  istore8 hiChar hcAbs

  let c0F ← iconst64 0x0F
  let loNib ← band byteVal c0F
  let loRel ← iadd hexTblC loNib
  let loAbs ← iadd k.ptr loRel
  let loChar ← uload8_64 loAbs
  let hcpNext ← iadd hcpI k.c1
  let lcRel ← iadd hexOutC hcpNext
  let lcAbs ← iadd k.ptr lcRel
  istore8 loChar lcAbs

  let hcpNext2 ← iadd hcpNext k.c1
  let hbiNext ← iadd hbi k.c1
  jump hexInner.ref [hWord, hbiNext, hcpNext2]

  startBlock hexInnerDone
  let finalCp := hexInnerDone.param 0
  let hwiNext ← iadd hwi k.c1
  jump hexOuter.ref [hwiNext, finalCp]

  startBlock hexDone
  let totalChars := hexDone.param 0
  let nlRel ← iadd hexOutC totalChars
  let nlAbs ← iadd k.ptr nlRel
  let nlChar ← iconst64 10
  istore8 nlChar nlAbs
  let outLen ← iadd totalChars k.c1
  let outFname ← iconst64 outputFilename_off
  let outData ← iconst64 hexOutput_off
  let _ ← call fnWrite [k.ptr, outFname, outData, k.zero, outLen]
  ret

-- Main builder: compose the sub-builders
set_option maxRecDepth 2048 in
def clifIrSource : String := buildProgram do
  let fnRead ← declareFileRead
  let fnWrite ← declareFileWrite

  let ptr ← entryBlock

  -- Step 1: Read input file
  let inFname ← iconst64 inputFilename_off
  let dataOff ← iconst64 fileData_off
  let zero ← iconst64 0
  let fileSize ← call fnRead [ptr, inFname, dataOff, zero, zero]
  storeAt ptr fileSize_off fileSize

  -- Common constants
  let c1 ← iconst64 1
  let c4 ← iconst64 4
  let c8 ← iconst64 8
  let c32 ← iconst64 32
  let c64 ← iconst64 64
  let mask32 ← iconst64 0xFFFFFFFF
  let c3 ← iconst64 3
  let c16 ← iconst64 16
  let c7 ← iconst64 7
  let c10 ← iconst64 10
  let c13 ← iconst64 13
  let c24 ← iconst64 24

  let k : Consts := {
    ptr, zero, c1, c3, c4, c7, c8, c10, c13, c16, c24, c32, c64, mask32, dataOff
  }

  -- Step 2: Padding
  let (_, numBlocks) ← emitPadding k fileSize

  -- Step 3: Copy H_initial → H_working
  emitCopyH k

  -- Step 4: Outer loop over blocks
  let hexBlk ← declareBlock []
  let outerHdr ← declareBlock [.i64]
  let blkIdx := outerHdr.param 0
  let outerBody ← declareBlock []

  jump outerHdr.ref [zero]
  startBlock outerHdr
  let outerDone ← icmp .uge blkIdx numBlocks
  brif outerDone hexBlk.ref [] outerBody.ref []

  startBlock outerBody
  let blkOff ← imul blkIdx c64
  let blkBase ← iadd dataOff blkOff

  -- Load W[0..15]
  emitLoadW k blkBase

  -- Expand W[16..63]
  emitExpandW k

  -- Init working vars a-h
  let hW ← iconst64 H_work_off
  let hWA ← iadd ptr hW
  let va ← uload32_64 hWA
  let hWA1 ← iadd hWA c4
  let vb ← uload32_64 hWA1
  let hWA2 ← iadd hWA1 c4
  let vc ← uload32_64 hWA2
  let hWA3 ← iadd hWA2 c4
  let vd ← uload32_64 hWA3
  let hWA4 ← iadd hWA3 c4
  let ve ← uload32_64 hWA4
  let hWA5 ← iadd hWA4 c4
  let vf ← uload32_64 hWA5
  let hWA6 ← iadd hWA5 c4
  let vg ← uload32_64 hWA6
  let hWA7 ← iadd hWA6 c4
  let vh ← uload32_64 hWA7

  -- 64-round compression
  let addBackBlk ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64, .i64, .i64, .i64]
  let _ ← emitCompressionRound k blkIdx va vb vc vd ve vf vg vh addBackBlk

  -- Add a-h back to H_working
  emitAddBack k addBackBlk outerHdr

  -- Step 6: Hex formatting
  emitHexFormat k fnWrite hexBlk

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
