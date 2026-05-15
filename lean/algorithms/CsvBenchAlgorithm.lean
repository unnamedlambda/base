import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace CsvBench

/-
  CSV benchmark: sum salary column (field 6) across all rows.
  Payload: "input_path\0output_path\0"
  SIMD newline scan to skip header; SIMD comma scan to find field 6; digit accumulate.
-/

def INPUT_PATH_OFF  : Nat := 0x0100
def OUTPUT_PATH_OFF : Nat := 0x0200
def LEFT_VAL        : Nat := 0x0350
def CSV_DATA        : Nat := 0x2000
def MAX_CSV_BYTES   : Nat := 512 * 1024 * 1024
def MEM_SIZE        : Nat := CSV_DATA + MAX_CSV_BYTES

set_option maxRecDepth 4096 in
def mainFn : IRBuilder Unit := do
  let ptr     ← entryBlock
  let fnRead  ← declareFileRead
  let fnWrite ← declareFileWrite
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let zero    ← iconst64 0
  let zero32  ← iconst32 0

  let cpIn       ← declareBlock [.i64]
  let cpOut1     ← declareBlock [.i64]
  let cpOut      ← declareBlock [.i64, .i64]
  let readBlk    ← declareBlock []
  -- header skip
  let skipHdr16  ← declareBlock [.i64]
  let skipHdrB   ← declareBlock [.i64]
  let skipHdrFnd ← declareBlock [.i64, .i32]
  let skipHdrSc  ← declareBlock [.i64]
  let hdrDone    ← declareBlock [.i64]
  -- comma scan (comma count is i64 to avoid iadd_imm type mismatch)
  let commaSc16  ← declareBlock [.i64, .i64, .i64]
  let commaScB   ← declareBlock [.i64, .i64, .i64]
  let commaF16   ← declareBlock [.i64, .i64, .i64, .i32]
  let commaF1    ← declareBlock [.i64, .i64, .i64]
  let commaInc1  ← declareBlock [.i64, .i64, .i64]
  -- digit accumulation
  let digitLp    ← declareBlock [.i64, .i64, .i64]
  let digitAcc   ← declareBlock [.i64, .i64, .i64, .i64]
  let salDone    ← declareBlock [.i64, .i64, .i64]
  -- itoa/write
  let itoaStart  ← declareBlock [.i64]
  let itoaFind   ← declareBlock [.i64, .i64]
  let itoaWr     ← declareBlock [.i64, .i64, .i64]
  let itoaNL     ← declareBlock [.i64]

  jump cpIn.ref [zero]

  -- Copy input path
  startBlock cpIn
  let si1 := cpIn.param 0
  let ch1  ← uload8_64 (← iadd dataPtr si1)
  istore8 ch1 (← iadd (← absAddr ptr INPUT_PATH_OFF) si1)
  let si1' ← iaddImm si1 1
  brif (← icmpImm .eq ch1 0) cpOut1.ref [si1'] cpIn.ref [si1']

  startBlock cpOut1
  jump cpOut.ref [cpOut1.param 0, zero]

  -- Copy output path
  startBlock cpOut
  let si3 := cpOut.param 0; let di3 := cpOut.param 1
  let ch3  ← uload8_64 (← iadd dataPtr si3)
  istore8 ch3 (← iadd (← absAddr ptr OUTPUT_PATH_OFF) di3)
  let si3' ← iaddImm si3 1; let di3' ← iaddImm di3 1
  brif (← icmpImm .eq ch3 0) readBlk.ref [] cpOut.ref [si3', di3']

  -- Read file; fileSize and csvBase dominate all scan blocks
  startBlock readBlk
  let fileSize ← readFile ptr fnRead INPUT_PATH_OFF CSV_DATA
  let csvBase  ← absAddr ptr CSV_DATA
  let nlVec    ← splat .i8x16 (← iconst8 10)
  let cmVec    ← splat .i8x16 (← iconst8 44)
  jump skipHdr16.ref [zero]

  -- SIMD newline scan: skip header row
  startBlock skipHdr16
  let hp := skipHdr16.param 0
  brif (← icmp .sle (← iaddImm hp 16) fileSize) skipHdrB.ref [hp] skipHdrSc.ref [hp]

  startBlock skipHdrB
  let hp2 := skipHdrB.param 0
  let row  ← loadI8x16 (← iadd csvBase hp2)
  let eq   ← icmp .eq row nlVec
  let mask ← vhighBits eq
  brif (← icmp .ne mask zero32) skipHdrFnd.ref [hp2, mask]
                                 skipHdr16.ref [← iaddImm hp2 16]

  -- Found newline in SIMD chunk: extract position
  startBlock skipHdrFnd
  let hp3 := skipHdrFnd.param 0; let msk := skipHdrFnd.param 1
  let off  ← uextend64 (← ctz32 msk)
  jump hdrDone.ref [← iadd hp3 off]

  -- Scalar fallback header skip
  startBlock skipHdrSc
  let hp4 := skipHdrSc.param 0
  let b    ← uload8_64 (← iadd csvBase hp4)
  brif (← icmpImm .eq b 10) hdrDone.ref [hp4] skipHdrSc.ref [← iaddImm hp4 1]

  startBlock hdrDone
  let hp5 := hdrDone.param 0
  jump commaSc16.ref [← iaddImm hp5 1, zero, zero]

  -- SIMD comma scan (pos, total, comma_count)
  startBlock commaSc16
  let cp := commaSc16.param 0; let tot := commaSc16.param 1
  let cc  := commaSc16.param 2
  brif (← icmp .sle (← iaddImm cp 16) fileSize) commaScB.ref [cp, tot, cc]
                                                   commaF1.ref [cp, tot, cc]

  startBlock commaScB
  let cp2 := commaScB.param 0; let tot2 := commaScB.param 1
  let cc2  := commaScB.param 2
  let row2 ← loadI8x16 (← iadd csvBase cp2)
  let eq2  ← icmp .eq row2 cmVec
  let msk2 ← vhighBits eq2
  brif (← icmp .ne msk2 zero32) commaF16.ref [cp2, tot2, cc2, msk2]
                                  commaSc16.ref [← iaddImm cp2 16, tot2, cc2]

  -- Found commas in SIMD chunk — process first, then continue or go to digits
  startBlock commaF16
  let cb  := commaF16.param 0; let tot3 := commaF16.param 1
  let cc3 := commaF16.param 2; let msk3 := commaF16.param 3
  let off2 ← uextend64 (← ctz32 msk3)
  let abs  ← iadd cb off2
  let cc3' ← iaddImm cc3 1
  let five ← iconst64 5
  brif (← icmp .eq cc3' five) digitLp.ref [← iaddImm abs 1, tot3, zero]
                                commaSc16.ref [← iaddImm abs 1, tot3, cc3']

  -- Scalar comma fallback
  startBlock commaF1
  let cp3 := commaF1.param 0; let tot4 := commaF1.param 1
  let cc4  := commaF1.param 2
  let b2   ← uload8_64 (← iadd csvBase cp3)
  brif (← icmpImm .eq b2 44) commaInc1.ref [← iaddImm cp3 1, tot4, cc4]
                               commaF1.ref [← iaddImm cp3 1, tot4, cc4]

  -- Scalar comma found: increment count, check ==5
  startBlock commaInc1
  let cp4 := commaInc1.param 0; let tot5 := commaInc1.param 1
  let cc5  := commaInc1.param 2
  let cc5' ← iaddImm cc5 1
  let five2 ← iconst64 5
  brif (← icmp .eq cc5' five2) digitLp.ref [cp4, tot5, zero]
                                 commaSc16.ref [cp4, tot5, cc5']

  -- Digit accumulation loop
  startBlock digitLp
  let dp  := digitLp.param 0; let tot6 := digitLp.param 1
  let acc := digitLp.param 2
  let b3  ← uload8_64 (← iadd csvBase dp)
  brif (← icmpImm .eq b3 10) salDone.ref [dp, tot6, acc]
                               digitAcc.ref [dp, tot6, acc, b3]

  startBlock digitAcc
  let dp2  := digitAcc.param 0; let tot7 := digitAcc.param 1
  let acc2 := digitAcc.param 2; let b4   := digitAcc.param 3
  let acc2'← iadd (← imul acc2 (← iconst64 10)) (← isub b4 (← iconst64 48))
  jump digitLp.ref [← iaddImm dp2 1, tot7, acc2']

  -- Salary row done
  startBlock salDone
  let dp3 := salDone.param 0; let tot8 := salDone.param 1
  let acc3 := salDone.param 2
  let dp3' ← iaddImm dp3 1
  brif (← icmp .sge dp3' fileSize) itoaStart.ref [← iadd tot8 acc3]
                                     commaSc16.ref [dp3', ← iadd tot8 acc3, zero]

  -- itoa + write result
  startBlock itoaStart
  let total := itoaStart.param 0
  jump itoaFind.ref [total, ← iconst64 1]

  startBlock itoaFind
  let totF := itoaFind.param 0; let divF := itoaFind.param 1
  let divF10 ← imul divF (← iconst64 10)
  brif (← icmp .ugt divF10 totF) itoaWr.ref [totF, divF, ← iconst64 LEFT_VAL]
                                   itoaFind.ref [totF, divF10]

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
  let _ ← call fnWrite [ptr, ← iconst64 OUTPUT_PATH_OFF, ← iconst64 LEFT_VAL,
                         zero, zero]
  ret

def clifIR : String := buildProgram mainFn

def artifacts : Array Json :=
  #[toJsonEntry "csv_algorithm" {
    cranelift_ir := clifIR,
    memory_size := MEM_SIZE,
    context_offset := 0
  } {
    fn_idx := u32 1
  }]

end CsvBench
