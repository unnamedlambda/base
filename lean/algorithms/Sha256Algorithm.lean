import AlgorithmLib
open Lean (Json toJson)
open AlgorithmLib
open AlgorithmLib.Layout

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
-- ---------------------------------------------------------------------------

def maxFileSize : Nat := 4 * 1024 * 1024  -- 4MB max input file

-- Typed field handles — the FieldTy is carried at the type level
structure Fields where
  reserved       : Fld (.bytes 64)
  fileSize       : Fld .i64
  paddedLen      : Fld .i64
  numBlocks      : Fld .i64
  inputFilename  : Fld (.bytes 256)
  outputFilename : Fld (.bytes 256)
  hexOutput      : Fld (.bytes 66)
  K              : Fld (.bytes 256)
  H_init         : Fld (.bytes 32)
  H_work         : Fld (.bytes 32)
  W              : Fld (.bytes 256)
  hexTable       : Fld (.bytes 16)
  fileData       : Fld (.bytes (maxFileSize + 128))

-- Memory layout: all offsets computed sequentially by the Layout DSL.
-- `field` returns a typed handle; no strings to get wrong.
def mkLayout : Fields × LayoutMeta := Layout.build do
  let reserved       ← field (.bytes 64)       -- [0x00, 0x40)  GPU ctx ptr etc.
  let fileSize       ← field .i64              -- scratch: file_size
  let paddedLen      ← field .i64              -- scratch: padded_len
  let numBlocks      ← field .i64              -- scratch: num_blocks
  skip 168                                      -- pad to 0x100
  let inputFilename  ← field (.bytes 256)      -- input filename (patched at runtime)
  let outputFilename ← field (.bytes 256)      -- output filename
  let hexOutput      ← field (.bytes 66)       -- 64 hex chars + newline + padding
  skip (0x1000 - 0x0342)                        -- pad to 0x1000
  let K              ← field (.bytes 256)      -- K constants (64 x u32 LE)
  let H_init         ← field (.bytes 32)       -- H initial state (8 x u32 LE)
  let H_work         ← field (.bytes 32)       -- H working state (runtime)
  let W              ← field (.bytes 256)      -- W message schedule (64 x u32)
  let hexTable       ← field (.bytes 16)       -- "0123456789abcdef"
  skip (0x2000 - 0x1250)                        -- pad to 0x2000
  let fileData       ← field (.bytes (maxFileSize + 128))
  pure { reserved, fileSize, paddedLen, numBlocks, inputFilename,
         outputFilename, hexOutput, K, H_init, H_work, W, hexTable, fileData }

def f : Fields := mkLayout.1
def layoutMeta : LayoutMeta := mkLayout.2


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

  fldStore k.ptr f.paddedLen paddedLen

  -- Zero bytes from file_size+1 to padded_len-8
  let zeroEnd ← isub paddedLen k.c8

  -- Zero-fill loop: for zi in [fsp1, zeroEnd)
  forLoopFromTo .i64 fsp1 zeroEnd fun zi => do
    let zrel ← iadd k.dataOff zi
    let zabs ← iadd k.ptr zrel
    istore8 k.zero zabs

  -- Write big-endian 64-bit bit length at padded_len - 8
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
  fldStore k.ptr f.numBlocks numBlocks

  pure (paddedLen, numBlocks)

-- Step 3: Copy H_initial → H_working
def emitCopyH (k : Consts) : IRBuilder Unit := do
  let hInitBase ← fldOffset f.H_init
  let hWorkBase ← fldOffset f.H_work
  forLoop .i64 k.c8 fun ci => do
    let byteOff ← imul ci k.c4
    let srcAbs ← iadd k.ptr (← iadd hInitBase byteOff)
    let dstAbs ← iadd k.ptr (← iadd hWorkBase byteOff)
    store (← load32 srcAbs) dstAbs

-- Load W[0..15] big-endian from message block
def emitLoadW (k : Consts) (blkBase : Val) : IRBuilder Unit := do
  let wOffC ← fldOffset f.W
  forLoop .i64 k.c16 fun wi => do
    let wi4 ← imul wi k.c4
    let bAbs ← iadd k.ptr (← iadd blkBase wi4)
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
    let w32 ← ireduce32 (← bor (← bor (← bor s0 s1) s2) byte3)
    let wAbs ← iadd k.ptr (← iadd wOffC wi4)
    store w32 wAbs

-- Expand W[16..63]
def emitExpandW (k : Consts) : IRBuilder Unit := do
  let wOff' ← fldOffset f.W
  let loadWAt (idxRel : Val) : IRBuilder Val := do
    uload32_64 (← iadd k.ptr (← iadd wOff' (← imul idxRel k.c4)))
  -- For ei in [16, 64): W[ei] = (sigma1(W[ei-2]) + W[ei-7] + sigma0(W[ei-15]) + W[ei-16]) & mask32
  forLoopFromTo .i64 k.c16 k.c64 fun ei => do
    let wi2val  ← loadWAt (← isub ei (← iconst64 2))
    -- sigma1(x): rotr(x,17) ^ rotr(x,19) ^ (x >> 10)
    let rotr17 ← band (← bor (← ushr wi2val (← iconst64 17)) (← ishl wi2val (← iconst64 15))) k.mask32
    let rotr19 ← band (← bor (← ushr wi2val (← iconst64 19)) (← ishl wi2val k.c13))            k.mask32
    let sigma1 ← bxor (← bxor rotr17 rotr19) (← ushr wi2val k.c10)
    let wi7val  ← loadWAt (← isub ei k.c7)
    let wi15val ← loadWAt (← isub ei (← iconst64 15))
    -- sigma0(x): rotr(x,7) ^ rotr(x,18) ^ (x >> 3)
    let rotr7  ← band (← bor (← ushr wi15val k.c7)            (← ishl wi15val (← iconst64 25))) k.mask32
    let rotr18 ← band (← bor (← ushr wi15val (← iconst64 18)) (← ishl wi15val (← iconst64 14))) k.mask32
    let sigma0 ← bxor (← bxor rotr7 rotr18) (← ushr wi15val k.c3)
    let wi16val ← loadWAt (← isub ei k.c16)
    let wNew    ← band (← iadd (← iadd (← iadd sigma1 wi7val) sigma0) wi16val) k.mask32
    let wAbs ← iadd k.ptr (← iadd wOff' (← imul ei k.c4))
    store (← ireduce32 wNew) wAbs

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
  let kBase ← fldOffset f.K
  let ri4 ← imul ri k.c4
  let kRel ← iadd kBase ri4
  let kAbs ← iadd k.ptr kRel
  let kVal ← uload32_64 kAbs

  let wBase ← fldOffset f.W
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

  let oldH0 ← fldLoad32At k.ptr f.H_work 0
  let newH0 ← iadd oldH0 abA
  let mH0 ← band newH0 k.mask32
  let rH0 ← ireduce32 mH0
  fldStore32At k.ptr f.H_work 0 rH0

  let oldH1 ← fldLoad32At k.ptr f.H_work 4
  let newH1 ← iadd oldH1 abB
  let mH1 ← band newH1 k.mask32
  let rH1 ← ireduce32 mH1
  fldStore32At k.ptr f.H_work 4 rH1

  let oldH2 ← fldLoad32At k.ptr f.H_work 8
  let newH2 ← iadd oldH2 abC
  let mH2 ← band newH2 k.mask32
  let rH2 ← ireduce32 mH2
  fldStore32At k.ptr f.H_work 8 rH2

  let oldH3 ← fldLoad32At k.ptr f.H_work 12
  let newH3 ← iadd oldH3 abD
  let mH3 ← band newH3 k.mask32
  let rH3 ← ireduce32 mH3
  fldStore32At k.ptr f.H_work 12 rH3

  let oldH4 ← fldLoad32At k.ptr f.H_work 16
  let newH4 ← iadd oldH4 abE
  let mH4 ← band newH4 k.mask32
  let rH4 ← ireduce32 mH4
  fldStore32At k.ptr f.H_work 16 rH4

  let oldH5 ← fldLoad32At k.ptr f.H_work 20
  let newH5 ← iadd oldH5 abF
  let mH5 ← band newH5 k.mask32
  let rH5 ← ireduce32 mH5
  fldStore32At k.ptr f.H_work 20 rH5

  let oldH6 ← fldLoad32At k.ptr f.H_work 24
  let newH6 ← iadd oldH6 abG
  let mH6 ← band newH6 k.mask32
  let rH6 ← ireduce32 mH6
  fldStore32At k.ptr f.H_work 24 rH6

  let oldH7 ← fldLoad32At k.ptr f.H_work 28
  let newH7 ← iadd oldH7 abH
  let mH7 ← band newH7 k.mask32
  let rH7 ← ireduce32 mH7
  fldStore32At k.ptr f.H_work 28 rH7

  let nextBlkIdx ← iadd abBlkIdx k.c1
  jump outerHdr.ref [nextBlkIdx]

-- Step 6: Hex formatting and file output
def emitHexFormat (k : Consts) (fnWrite : FnRef) (hexBlk : DeclaredBlock) : IRBuilder Unit := do
  startBlock hexBlk
  let hWorkC ← fldOffset f.H_work
  let hexOutC ← fldOffset f.hexOutput
  let hexTblC ← fldOffset f.hexTable

  let cFF ← iconst64 0xFF
  let c0F ← iconst64 0x0F
  let c3' ← iconst64 3
  -- Outer: for each of 8 H_work words, accumulating hex output byte offset.
  -- Inner: for each of 4 bytes per word, write two hex chars and advance by 2.
  let totalChars ← forLoopAcc .i64 .i64 k.c8 k.zero fun hwi hcp => do
    let hwAbs ← iadd k.ptr (← iadd hWorkC (← imul hwi k.c4))
    let wordVal ← uload32_64 hwAbs
    forLoopAcc .i64 .i64 k.c4 hcp fun hbi hcpI => do
      let shift ← isub c3' hbi
      let byteVal ← band (← ushr wordVal (← imul shift k.c8)) cFF
      let hiChar ← uload8_64 (← iadd k.ptr (← iadd hexTblC (← ushr byteVal k.c4)))
      istore8 hiChar (← iadd k.ptr (← iadd hexOutC hcpI))
      let hcpNext ← iadd hcpI k.c1
      let loChar ← uload8_64 (← iadd k.ptr (← iadd hexTblC (← band byteVal c0F)))
      istore8 loChar (← iadd k.ptr (← iadd hexOutC hcpNext))
      iadd hcpNext k.c1
  let nlChar ← iconst64 10
  istore8 nlChar (← iadd k.ptr (← iadd hexOutC totalChars))
  let outLen ← iadd totalChars k.c1
  let outFname ← fldOffset f.outputFilename
  let outData ← fldOffset f.hexOutput
  let _ ← call fnWrite [k.ptr, outFname, outData, k.zero, outLen]
  ret

-- Main builder: compose the sub-builders
set_option maxRecDepth 2048 in
def clifIrSource : String := buildProgram do
  let fnRead ← declareFileRead
  let fnWrite ← declareFileWrite

  let ptr ← entryBlock

  -- Step 1: Read input file
  let fileSize ← fldReadFile ptr fnRead f.inputFilename f.fileData
  let dataOff ← fldOffset f.fileData
  let zero ← iconst64 0
  fldStore ptr f.fileSize fileSize

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

  -- Init working vars a-h (bounds-checked at compile time: 0..28 + 4 ≤ 32)
  let va ← fldLoad32At ptr f.H_work 0
  let vb ← fldLoad32At ptr f.H_work 4
  let vc ← fldLoad32At ptr f.H_work 8
  let vd ← fldLoad32At ptr f.H_work 12
  let ve ← fldLoad32At ptr f.H_work 16
  let vf ← fldLoad32At ptr f.H_work 20
  let vg ← fldLoad32At ptr f.H_work 24
  let vh ← fldLoad32At ptr f.H_work 28

  -- 64-round compression
  let addBackBlk ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64, .i64, .i64, .i64]
  let _ ← emitCompressionRound k blkIdx va vb vc vd ve vf vg vh addBackBlk

  -- Add a-h back to H_working
  emitAddBack k addBackBlk outerHdr

  -- Step 6: Hex formatting
  emitHexFormat k fnWrite hexBlk

-- ---------------------------------------------------------------------------
-- Payload construction (generated from layout)
-- ---------------------------------------------------------------------------

def payloads : List UInt8 :=
  mkPayload f.fileData.offset [
    f.inputFilename.init (stringToBytes "input.bin"),
    f.outputFilename.init (stringToBytes "sha256_output.txt"),
    f.K.init ((kConstants.map uint32ToBytes).flatten),
    f.H_init.init ((hInitial.map uint32ToBytes).flatten),
    f.hexTable.init ("0123456789abcdef".toUTF8.toList)
  ]

-- ---------------------------------------------------------------------------
-- Configuration
-- ---------------------------------------------------------------------------

def sha256Config : Setup := {
  cranelift_ir := clifIrSource,
  memory_size := layoutMeta.totalSize,
  context_offset := 0,
  initial_memory := payloads
}

def sha256Algorithm : Algorithm := {
    fn_idx := IR.mainFnIdx
  }

end Algorithm

def main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  emitArtifacts outDir #[toJsonEntry "sha256_app" Algorithm.sha256Config Algorithm.sha256Algorithm]
