import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace JsonBench

/-
  JSON benchmark: sum "value": <N> integers.
  Payload: "input_path\0output_path\0"
  SIMD '\"v' prefix scan then 8-byte needle verify, then digit accumulate.
-/

def INPUT_PATH_OFF  : Nat := 0x0100
def OUTPUT_PATH_OFF : Nat := 0x0200
def OUTPUT_BUF      : Nat := 0x0350
def INPUT_DATA      : Nat := 0x4000
def MAX_JSON_BYTES  : Nat := 512 * 1024 * 1024
def MEM_SIZE        : Nat := INPUT_DATA + MAX_JSON_BYTES
def TIMEOUT_MS      : Nat := 300000

set_option maxRecDepth 4096 in
def mainFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let fnRead  ← declareFileRead
  let fnWrite ← declareFileWrite
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let zero    ← iconst64 0

  let cpIn      ← declareBlock [.i64]
  let cpOut1    ← declareBlock [.i64]
  let cpOut     ← declareBlock [.i64, .i64]
  let readBlk   ← declareBlock []
  let scanChk   ← declareBlock [.i64, .i64]
  let simdBlk   ← declareBlock [.i64, .i64]
  let matchFind ← declareBlock [.i64, .i64, .i32]
  let verify    ← declareBlock [.i64, .i64, .i64, .i32]
  let matchFail ← declareBlock [.i64, .i64, .i32]
  let digitSt   ← declareBlock [.i64, .i64]
  let digitLoop ← declareBlock [.i64, .i64, .i64]
  let digitFlsh ← declareBlock [.i64, .i64, .i64]
  let acDigit   ← declareBlock [.i64, .i64, .i64, .i64]
  let done      ← declareBlock [.i64]
  let itoaWr    ← declareBlock [.i64, .i64, .i64]
  let itoaNL    ← declareBlock [.i64]

  jump cpIn.ref [zero]

  -- Copy input path
  startBlock cpIn
  let si1 := cpIn.param 0
  let ch1  ← uload8_64 (← iadd dataPtr si1)
  istore8 ch1 (← iadd (← absAddr ptr INPUT_PATH_OFF) si1)
  let si1' ← iaddImm si1 1
  brif (← icmpImm .eq ch1 0) cpOut1.ref [si1'] cpIn.ref [si1']

  startBlock cpOut1
  let si2 := cpOut1.param 0
  jump cpOut.ref [si2, zero]

  -- Copy output path
  startBlock cpOut
  let si3 := cpOut.param 0; let di3 := cpOut.param 1
  let ch3  ← uload8_64 (← iadd dataPtr si3)
  istore8 ch3 (← iadd (← absAddr ptr OUTPUT_PATH_OFF) di3)
  let si3' ← iaddImm si3 1; let di3' ← iaddImm di3 1
  brif (← icmpImm .eq ch3 0) readBlk.ref [] cpOut.ref [si3', di3']

  -- Read file; fileSize and dataBase dominate all scan blocks
  startBlock readBlk
  let fileSize ← readFile ptr fnRead INPUT_PATH_OFF INPUT_DATA
  let dataBase ← absAddr ptr INPUT_DATA
  let nine     ← iconst64 9
  let endPos   ← isub fileSize nine      -- scan until pos > fileSize-9
  let quot34   ← iconst8 34             -- '"'
  let vOf34    ← splat .i8x16 quot34
  let vv       ← iconst8 118            -- 'v'
  let vOfV     ← splat .i8x16 vv
  let needle   ← iconst64 2322206377019990390  -- "value\": " as LE i64
  jump scanChk.ref [zero, zero]

  -- End-of-file check
  startBlock scanChk
  let pos  := scanChk.param 0; let tot := scanChk.param 1
  brif (← icmp .sgt pos endPos) done.ref [tot] simdBlk.ref [pos, tot]

  -- 16-byte SIMD scan for '\"v' prefix
  startBlock simdBlk
  let pos2 := simdBlk.param 0; let tot2 := simdBlk.param 1
  let p2   ← iadd dataBase pos2
  let row0 ← loadI8x16 p2
  let row1 ← loadI8x16 (← iaddImm p2 1)
  let eq0  ← icmp .eq row0 vOf34
  let eq1  ← icmp .eq row1 vOfV
  let both ← band eq0 eq1
  let mask ← vhighBits both
  let zero32 ← iconst32 0
  let has  ← icmp .ne mask zero32
  brif has matchFind.ref [pos2, tot2, mask]
           scanChk.ref [← iaddImm pos2 16, tot2]

  -- Extract first match position from bitmask
  startBlock matchFind
  let cb   := matchFind.param 0; let tot3 := matchFind.param 1
  let msk  := matchFind.param 2
  let off32 ← ctz32 msk
  let off  ← uextend64 off32
  let abs  ← iadd cb off
  brif (← icmp .sgt abs endPos) done.ref [tot3]
       verify.ref [abs, tot3, cb, msk]

  -- Verify 8-byte needle after '\"'
  startBlock verify
  let ap   := verify.param 0; let tot4 := verify.param 1
  let cb2  := verify.param 2; let msk2 := verify.param 3
  let bytes8 ← load64 (← iaddImm (← iadd dataBase ap) 1)
  let isNeedle ← icmp .eq bytes8 needle
  brif isNeedle digitSt.ref [ap, tot4]
               matchFail.ref [cb2, tot4, msk2]

  -- Clear lowest set bit, continue or advance chunk
  startBlock matchFail
  let cb3  := matchFail.param 0; let tot5 := matchFail.param 1
  let msk3 := matchFail.param 2
  let zero32b ← iconst32 0
  let newMask ← band msk3 (← iadd msk3 (← iconst32 (-1)))
  let more    ← icmp .ne newMask zero32b
  brif more matchFind.ref [cb3, tot5, newMask]
            scanChk.ref [← iaddImm cb3 16, tot5]

  -- Skip needle (9 bytes), start digit accumulation
  startBlock digitSt
  let ap2  := digitSt.param 0; let tot6 := digitSt.param 1
  jump digitLoop.ref [← iaddImm ap2 9, tot6, zero]

  -- Digit loop
  startBlock digitLoop
  let dp   := digitLoop.param 0; let tot7 := digitLoop.param 1
  let acc  := digitLoop.param 2
  let byte ← uload8_64 (← iadd dataBase dp)
  let d    ← isub byte (← iconst64 48)
  let nine2 ← iconst64 9
  brif (← icmp .ugt d nine2) digitFlsh.ref [dp, tot7, acc]
                              acDigit.ref [dp, tot7, acc, d]

  -- Flush accumulator
  startBlock digitFlsh
  let dp2  := digitFlsh.param 0; let tot8 := digitFlsh.param 1
  let acc2 := digitFlsh.param 2
  jump scanChk.ref [dp2, ← iadd tot8 acc2]

  -- Accumulate digit
  startBlock acDigit
  let dp3  := acDigit.param 0; let tot9 := acDigit.param 1
  let acc3 := acDigit.param 2; let d2   := acDigit.param 3
  let acc3' ← iadd (← imul acc3 (← iconst64 10)) d2
  jump digitLoop.ref [← iaddImm dp3 1, tot9, acc3']

  -- itoa + write: scale div up while div*10 <= total
  startBlock done
  let total := done.param 0
  let finalDiv ← whileLoop1 .i64 (← iconst64 1)
    (fun div => do icmp .ule (← imul div (← iconst64 10)) total)
    (fun div => do imul div (← iconst64 10))
  jump itoaWr.ref [total, finalDiv, ← iconst64 OUTPUT_BUF]

  startBlock itoaWr
  let valW := itoaWr.param 0; let divW := itoaWr.param 1; let wposW := itoaWr.param 2
  let dig  ← udiv valW divW
  let digB ← iadd dig (← iconst64 48)
  istore8 digB (← iadd ptr wposW)
  let rem  ← isub valW (← imul dig divW)
  let divW'← udiv divW (← iconst64 10)
  let wpos'← iaddImm wposW 1
  brif (← icmpImm .eq divW' 0) itoaNL.ref [wpos'] itoaWr.ref [rem, divW', wpos']

  startBlock itoaNL
  let wp := itoaNL.param 0
  istore8 (← iconst64 10) (← iadd ptr wp)
  istore8 (← iconst32 0) (← iadd ptr (← iaddImm wp 1))
  let _ ← call fnWrite [ptr, ← iconst64 OUTPUT_PATH_OFF, ← iconst64 OUTPUT_BUF,
                         zero, zero]
  ret

def clifIR : String := buildProgram mainFn

def artifacts : Array Json :=
  #[toJsonEntry "json_algorithm" {
    cranelift_ir := clifIR,
    memory_size := MEM_SIZE,
    context_offset := 0
  } {
    actions := mkCallActions 1,
    cranelift_units := 0,
    timeout_ms := some TIMEOUT_MS
  }]

end JsonBench
