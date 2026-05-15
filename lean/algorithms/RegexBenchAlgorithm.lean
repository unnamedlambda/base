import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace RegexBench

/-
  Regex benchmark: count [a-z]+ing words (len>=4, all lowercase, ends "ing").
  Payload: "input_path\0output_path\0"
  Scalar byte scan with branchless "ing" match at word end.
-/

def INPUT_PATH_OFF  : Nat := 0x0100
def OUTPUT_PATH_OFF : Nat := 0x0200
def OUTPUT_BUF      : Nat := 0x0350
def INPUT_DATA      : Nat := 0x4000
def MAX_TEXT_BYTES  : Nat := 512 * 1024 * 1024
def MEM_SIZE        : Nat := INPUT_DATA + MAX_TEXT_BYTES

set_option maxRecDepth 2048 in
def mainFn : IRBuilder Unit := do
  let ptr    ← entryBlock
  let fnRead ← declareFileRead
  let fnWrite← declareFileWrite
  let dataPtr← load64 (← absAddr ptr 0x18)
  let zero   ← iconst64 0

  let cpIn    ← declareBlock [.i64]                               -- copy input path
  let cpOut1  ← declareBlock [.i64]                               -- transition
  let cpOut   ← declareBlock [.i64, .i64]                         -- copy output path
  let readBlk ← declareBlock []                                   -- read file
  let skipChk ← declareBlock [.i64, .i64]                         -- (pos, count) end check
  let skipLd  ← declareBlock [.i64, .i64]                         -- (pos, count) load+WS check
  let wordLp  ← declareBlock [.i64, .i64, .i64, .i64]            -- (pos,cnt,allLow,wStart)
  let procBy  ← declareBlock [.i64, .i64, .i64, .i64, .i64]      -- (..., byte)
  let wordDn  ← declareBlock [.i64, .i64, .i64, .i64]            -- (pos,cnt,allLow,wStart)
  let itoaFn  ← declareBlock [.i64, .i64]                        -- (total, div)
  let itoaWr  ← declareBlock [.i64, .i64, .i64]                  -- (val, div, wpos)
  let itoaNL  ← declareBlock [.i64]                              -- (wpos)
  let done    ← declareBlock [.i64]                              -- (count)

  jump cpIn.ref [zero]

  startBlock cpIn
  let si1 := cpIn.param 0
  let ch1  ← uload8_64 (← iadd dataPtr si1)
  istore8 ch1 (← iadd (← absAddr ptr INPUT_PATH_OFF) si1)
  let si1' ← iaddImm si1 1
  brif (← icmpImm .eq ch1 0) cpOut1.ref [si1'] cpIn.ref [si1']

  startBlock cpOut1
  let si2 := cpOut1.param 0
  jump cpOut.ref [si2, zero]

  startBlock cpOut
  let si3 := cpOut.param 0; let di3 := cpOut.param 1
  let ch3  ← uload8_64 (← iadd dataPtr si3)
  istore8 ch3 (← iadd (← absAddr ptr OUTPUT_PATH_OFF) di3)
  let si3' ← iaddImm si3 1; let di3' ← iaddImm di3 1
  brif (← icmpImm .eq ch3 0) readBlk.ref [] cpOut.ref [si3', di3']

  -- Read file once; fileSize and dataBase dominate all scan blocks
  startBlock readBlk
  let fileSize ← readFile ptr fnRead INPUT_PATH_OFF INPUT_DATA
  let dataBase ← absAddr ptr INPUT_DATA
  jump skipChk.ref [zero, zero]

  startBlock skipChk
  let pos := skipChk.param 0; let cnt := skipChk.param 1
  brif (← icmp .sge pos fileSize) done.ref [cnt] skipLd.ref [pos, cnt]

  startBlock skipLd
  let pos2 := skipLd.param 0; let cnt2 := skipLd.param 1
  let byte2 ← uload8_64 (← iadd dataBase pos2)
  let space ← iconst64 32
  let one64 ← iconst64 1
  brif (← icmp .ule byte2 space) skipChk.ref [← iaddImm pos2 1, cnt2]
                                  wordLp.ref [pos2, cnt2, one64, pos2]

  startBlock wordLp
  let pos3 := wordLp.param 0; let cnt3 := wordLp.param 1
  let allL := wordLp.param 2; let wSt := wordLp.param 3
  let byte3 ← uload8_64 (← iadd dataBase pos3)
  let space2 ← iconst64 32
  brif (← icmp .ule byte3 space2) wordDn.ref [pos3, cnt3, allL, wSt]
                                   procBy.ref [pos3, cnt3, allL, wSt, byte3]

  startBlock procBy
  let pos4 := procBy.param 0; let cnt4 := procBy.param 1
  let allL2 := procBy.param 2; let wSt2 := procBy.param 3; let byte4 := procBy.param 4
  let aLow  ← iconst64 97
  let r25   ← iconst64 25
  let shifted ← isub byte4 aLow
  let isLow   ← uextend64 (← icmp .ule shifted r25)
  let allL2'  ← band allL2 isLow
  jump wordLp.ref [← iaddImm pos4 1, cnt4, allL2', wSt2]

  startBlock wordDn
  let pos5 := wordDn.param 0; let cnt5 := wordDn.param 1
  let allL3 := wordDn.param 2; let wSt3 := wordDn.param 3
  let minLen ← iconst64 4
  let mask24 ← iconst64 16777215
  let ingLE  ← iconst64 6778473
  let len    ← isub pos5 wSt3
  let lenOk  ← uextend64 (← icmp .sge len minLen)
  let both   ← band allL3 lenOk
  let m3pos  ← iaddImm pos5 (-3)
  let raw4   ← uextend64 (← load32 (← iadd dataBase m3pos))
  let last3  ← band raw4 mask24
  let isIng  ← uextend64 (← icmp .eq last3 ingLE)
  let match1 ← band both isIng
  let cnt5'  ← iadd cnt5 match1
  jump skipChk.ref [← iaddImm pos5 1, cnt5']

  -- itoa + file write
  startBlock done
  let total := done.param 0
  let ten   ← iconst64 10
  let one64 ← iconst64 1
  jump itoaFn.ref [total, one64]

  startBlock itoaFn
  let totF := itoaFn.param 0; let divF := itoaFn.param 1
  let divF10 ← imul divF ten
  brif (← icmp .ugt divF10 totF) itoaWr.ref [totF, divF, ← iconst64 OUTPUT_BUF]
                                  itoaFn.ref [totF, divF10]

  startBlock itoaWr
  let valW := itoaWr.param 0; let divW := itoaWr.param 1; let wposW := itoaWr.param 2
  let dig  ← udiv valW divW
  let digB ← iadd dig (← iconst64 48)
  istore8 digB (← iadd ptr wposW)
  let rem  ← isub valW (← imul dig divW)
  let divW'← udiv divW ten
  let wpos'← iaddImm wposW 1
  brif (← icmpImm .eq divW' 0) itoaNL.ref [wpos'] itoaWr.ref [rem, divW', wpos']

  startBlock itoaNL
  let wp := itoaNL.param 0
  istore8 (← iconst64 10) (← iadd ptr wp)
  istore8 (← iconst32 0) (← iadd ptr (← iaddImm wp 1))
  let zz ← iconst64 0
  let outOff ← iconst64 OUTPUT_PATH_OFF
  let bufOff ← iconst64 OUTPUT_BUF
  let _ ← call fnWrite [ptr, outOff, bufOff, zz, zz]
  ret

def clifIR : String := buildProgram mainFn

def artifacts : Array Json :=
  #[toJsonEntry "regex_algorithm" {
    cranelift_ir := clifIR,
    memory_size := MEM_SIZE,
    context_offset := 0
  } {
    fn_idx := u32 1
  }]

end RegexBench
