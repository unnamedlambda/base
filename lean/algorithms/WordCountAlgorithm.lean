import Lean
import AlgorithmLib

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace WordCountBench

/-
  Word frequency counting: parse words, ht_increment, format word\tcount\n output.
  Payload: "input_path\0output_path\0"
  HT context at offset 0x00, colocated ht_create/ht_increment/ht_count/ht_get_entry.
-/

def CURRENT_KEY     : Nat := 0x0038
def NEW_VALUE       : Nat := 0x0040
def INPUT_PATH_OFF  : Nat := 0x0100
def OUTPUT_PATH_OFF : Nat := 0x0200
def RESULT_SLOT     : Nat := 0x0350
def OUTPUT_BUF      : Nat := 0x4000
def INPUT_DATA      : Nat := 0x14000
def MAX_TEXT_BYTES  : Nat := 512 * 1024 * 1024
def MEM_SIZE        : Nat := INPUT_DATA + MAX_TEXT_BYTES
def TIMEOUT_MS      : Nat := 300000

structure WcCtx where
  ptr       : Val
  dataPtr   : Val
  fnRead    : FnRef
  fnWrite   : FnRef
  fnHtInit  : FnRef
  fnHtClean : FnRef
  fnCreate  : FnRef
  fnIncr    : FnRef
  fnCount   : FnRef
  fnGetEntry: FnRef
  zero      : Val
  -- blocks shared across sub-functions
  cpIn      : DeclaredBlock
  cpOut1    : DeclaredBlock
  cpOut     : DeclaredBlock
  readBlk   : DeclaredBlock
  skipWS    : DeclaredBlock
  chkByte   : DeclaredBlock
  readWord  : DeclaredBlock
  readByte  : DeclaredBlock
  wordDone  : DeclaredBlock
  accumByte : DeclaredBlock
  fmtStart  : DeclaredBlock
  fmtLoop   : DeclaredBlock
  getEntry  : DeclaredBlock
  unpackW   : DeclaredBlock
  extractB  : DeclaredBlock
  writeTab  : DeclaredBlock
  writeByte : DeclaredBlock
  itoaWr    : DeclaredBlock
  writeNL   : DeclaredBlock
  writeFile : DeclaredBlock

def emitPathCopy (k : WcCtx) : IRBuilder Unit := do
  jump k.cpIn.ref [k.zero]

  startBlock k.cpIn
  let si1 := k.cpIn.param 0
  let ch1  ← uload8_64 (← iadd k.dataPtr si1)
  istore8 ch1 (← iadd (← absAddr k.ptr INPUT_PATH_OFF) si1)
  let si1' ← iaddImm si1 1
  brif (← icmpImm .eq ch1 0) k.cpOut1.ref [si1'] k.cpIn.ref [si1']

  startBlock k.cpOut1
  jump k.cpOut.ref [k.cpOut1.param 0, k.zero]

  startBlock k.cpOut
  let si3 := k.cpOut.param 0; let di3 := k.cpOut.param 1
  let ch3  ← uload8_64 (← iadd k.dataPtr si3)
  istore8 ch3 (← iadd (← absAddr k.ptr OUTPUT_PATH_OFF) di3)
  let si3' ← iaddImm si3 1; let di3' ← iaddImm di3 1
  brif (← icmpImm .eq ch3 0) k.readBlk.ref [] k.cpOut.ref [si3', di3']

def emitParsePhase (k : WcCtx) (fileSize : Val) (inputBase : Val) : IRBuilder Unit := do
  -- Skip whitespace
  startBlock k.skipWS
  let keyAddr  ← absAddr k.ptr CURRENT_KEY
  let keyLen   ← iconst32 8
  let one      ← iconst64 1
  let eight    ← iconst64 8
  let pos := k.skipWS.param 0; let ctx := k.skipWS.param 1
  brif (← icmp .sge pos fileSize) k.fmtStart.ref [ctx] k.chkByte.ref [pos, ctx]

  startBlock k.chkByte
  let pos2 := k.chkByte.param 0; let ctx2 := k.chkByte.param 1
  let byte2 ← uload8_64 (← iadd inputBase pos2)
  let pos2' ← iaddImm pos2 1
  brif (← icmp .ule byte2 (← iconst64 32))
       k.skipWS.ref [pos2', ctx2]
       k.readWord.ref [pos2, ctx2, k.zero, k.zero]

  -- Read word, accumulate bytes into u64
  startBlock k.readWord
  let pos3 := k.readWord.param 0; let ctx3 := k.readWord.param 1
  let wAcc  := k.readWord.param 2; let bIdx := k.readWord.param 3
  brif (← icmp .sge pos3 fileSize) k.wordDone.ref [pos3, ctx3, wAcc]
       k.readByte.ref [pos3, ctx3, wAcc, bIdx]

  startBlock k.readByte
  let pos4 := k.readByte.param 0; let ctx4 := k.readByte.param 1
  let wAcc2 := k.readByte.param 2; let bIdx2 := k.readByte.param 3
  let byte4 ← uload8_64 (← iadd inputBase pos4)
  brif (← icmp .ule byte4 (← iconst64 32))
       k.wordDone.ref [pos4, ctx4, wAcc2]
       k.accumByte.ref [pos4, ctx4, wAcc2, bIdx2, byte4]

  startBlock k.wordDone
  let pos5 := k.wordDone.param 0; let ctx5 := k.wordDone.param 1
  let wAcc3 := k.wordDone.param 2
  storeI64 wAcc3 keyAddr
  let _ ← call k.fnIncr [ctx5, keyAddr, keyLen, one]
  jump k.skipWS.ref [pos5, ctx5]

  startBlock k.accumByte
  let pos6 := k.accumByte.param 0; let ctx6 := k.accumByte.param 1
  let wAcc4 := k.accumByte.param 2; let bIdx3 := k.accumByte.param 3
  let byte5 := k.accumByte.param 4
  let shift ← imul bIdx3 eight
  let shifted ← ishl byte5 shift
  let wAcc4' ← bor wAcc4 shifted
  let bIdx3' ← iaddImm bIdx3 1
  -- After 8 bytes, flush word and continue (rare case, truncate)
  brif (← icmp .sge bIdx3' eight)
       k.wordDone.ref [← iaddImm pos6 1, ctx6, wAcc4']
       k.readWord.ref [← iaddImm pos6 1, ctx6, wAcc4', bIdx3']

def emitFormatPhase (k : WcCtx) : IRBuilder Unit := do
  startBlock k.fmtStart
  let eight   ← iconst64 8
  let byteFF  ← iconst64 0xFF
  let ten     ← iconst64 10
  let one     ← iconst64 1
  let keyAddr ← absAddr k.ptr CURRENT_KEY
  let valAddr ← absAddr k.ptr RESULT_SLOT
  let ctx7 := k.fmtStart.param 0
  let cnt32 ← call k.fnCount [ctx7]
  let cnt   ← uextend64 cnt32
  jump k.fmtLoop.ref [k.zero, ← iconst64 OUTPUT_BUF, ctx7, cnt]

  startBlock k.fmtLoop
  let idx  := k.fmtLoop.param 0; let opos := k.fmtLoop.param 1
  let ctx8 := k.fmtLoop.param 2; let tot  := k.fmtLoop.param 3
  brif (← icmp .sge idx tot) k.writeFile.ref [opos] k.getEntry.ref [idx, opos, ctx8, tot]

  startBlock k.getEntry
  let idx2 := k.getEntry.param 0; let opos2 := k.getEntry.param 1
  let ctx9  := k.getEntry.param 2; let tot2  := k.getEntry.param 3
  let idx32 ← ireduce32 idx2
  let _ ← call k.fnGetEntry [ctx9, idx32, keyAddr, valAddr]
  let word  ← load64 keyAddr
  let count ← load64 valAddr
  jump k.unpackW.ref [idx2, opos2, ctx9, tot2, word, count, k.zero]

  -- Unpack word bytes as ASCII
  startBlock k.unpackW
  let idx3  := k.unpackW.param 0; let opos3 := k.unpackW.param 1
  let ctx10 := k.unpackW.param 2; let tot3  := k.unpackW.param 3
  let word2 := k.unpackW.param 4; let cnt2  := k.unpackW.param 5
  let bi    := k.unpackW.param 6
  brif (← icmp .sge bi eight)
       k.writeTab.ref [idx3, opos3, ctx10, tot3, cnt2]
       k.extractB.ref [idx3, opos3, ctx10, tot3, word2, cnt2, bi]

  startBlock k.extractB
  let idx4  := k.extractB.param 0; let opos4 := k.extractB.param 1
  let ctx11 := k.extractB.param 2; let tot4  := k.extractB.param 3
  let word3 := k.extractB.param 4; let cnt3  := k.extractB.param 5
  let bi2   := k.extractB.param 6
  let shift2 ← imul bi2 eight
  let b     ← band (← ushr word3 shift2) byteFF
  brif (← icmpImm .eq b 0)
       k.writeTab.ref [idx4, opos4, ctx11, tot4, cnt3]
       k.writeByte.ref [idx4, opos4, ctx11, tot4, word3, cnt3, bi2, b]

  startBlock k.writeByte
  let idx5  := k.writeByte.param 0; let opos5 := k.writeByte.param 1
  let ctx12 := k.writeByte.param 2; let tot5  := k.writeByte.param 3
  let word4 := k.writeByte.param 4; let cnt4  := k.writeByte.param 5
  let bi3   := k.writeByte.param 6; let b2    := k.writeByte.param 7
  istore8 b2 (← iadd k.ptr opos5)
  jump k.unpackW.ref [idx5, ← iaddImm opos5 1, ctx12, tot5, word4, cnt4, ← iaddImm bi3 1]

  -- Write tab, then itoa count
  startBlock k.writeTab
  let idx6  := k.writeTab.param 0; let opos6 := k.writeTab.param 1
  let ctx13 := k.writeTab.param 2; let tot6  := k.writeTab.param 3
  let cnt5  := k.writeTab.param 4
  istore8 (← iconst64 9) (← iadd k.ptr opos6)   -- tab
  let opos6' ← iaddImm opos6 1
  -- Scale `div` up while div*10 <= cnt5, to find the highest divisor.
  let finalDiv ← whileLoop1 .i64 one
    (fun d => do icmp .ule (← imul d ten) cnt5)
    (fun d => do imul d ten)
  jump k.itoaWr.ref [idx6, opos6', ctx13, tot6, cnt5, finalDiv]

  startBlock k.itoaWr
  let idx8  := k.itoaWr.param 0; let opos8 := k.itoaWr.param 1
  let ctx15 := k.itoaWr.param 2; let tot8  := k.itoaWr.param 3
  let rem   := k.itoaWr.param 4; let div2  := k.itoaWr.param 5
  let dig   ← udiv rem div2
  let digB  ← iadd dig (← iconst64 48)
  istore8 digB (← iadd k.ptr opos8)
  let rem'  ← isub rem (← imul dig div2)
  let div2' ← udiv div2 ten
  let opos8'← iaddImm opos8 1
  brif (← icmpImm .eq div2' 0)
       k.writeNL.ref [idx8, opos8', ctx15, tot8]
       k.itoaWr.ref [idx8, opos8', ctx15, tot8, rem', div2']

  startBlock k.writeNL
  let idx9  := k.writeNL.param 0; let opos9 := k.writeNL.param 1
  let ctx16 := k.writeNL.param 2; let tot9  := k.writeNL.param 3
  istore8 (← iconst64 10) (← iadd k.ptr opos9)  -- newline
  jump k.fmtLoop.ref [← iaddImm idx9 1, ← iaddImm opos9 1, ctx16, tot9]

  startBlock k.writeFile
  let opos10 := k.writeFile.param 0
  istore8 (← iconst32 0) (← iadd k.ptr opos10)
  let _ ← call k.fnWrite [k.ptr, ← iconst64 OUTPUT_PATH_OFF, ← iconst64 OUTPUT_BUF,
                           k.zero, k.zero]
  callVoid k.fnHtClean [← absAddr k.ptr 0]
  ret

def mainFn : IRBuilder Unit := do
  let ptr      ← entryBlock
  let fnHtInit ← declareFFI "cl_ht_init"     [.i64] none
  let fnHtClean← declareFFI "cl_ht_cleanup"  [.i64] none
  let fnRead   ← declareFileRead
  let fnWrite  ← declareFileWrite
  let fnCreate ← declareColocatedFFI "ht_create"    [.i64]                   (some .i32)
  let fnIncr   ← declareColocatedFFI "ht_increment" [.i64, .i64, .i32, .i64] (some .i64)
  let fnCount  ← declareColocatedFFI "ht_count"     [.i64]                   (some .i32)
  let fnGet    ← declareColocatedFFI "ht_get_entry"  [.i64, .i32, .i64, .i64] (some .i32)
  let dataPtr  ← load64 (← absAddr ptr 0x18)
  let zero     ← iconst64 0

  let cpIn      ← declareBlock [.i64]
  let cpOut1    ← declareBlock [.i64]
  let cpOut     ← declareBlock [.i64, .i64]
  let readBlk   ← declareBlock []
  let skipWS    ← declareBlock [.i64, .i64]
  let chkByte   ← declareBlock [.i64, .i64]
  let readWord  ← declareBlock [.i64, .i64, .i64, .i64]
  let readByte  ← declareBlock [.i64, .i64, .i64, .i64]
  let wordDone  ← declareBlock [.i64, .i64, .i64]
  let accumByte ← declareBlock [.i64, .i64, .i64, .i64, .i64]
  let fmtStart  ← declareBlock [.i64]
  let fmtLoop   ← declareBlock [.i64, .i64, .i64, .i64]
  let getEntry  ← declareBlock [.i64, .i64, .i64, .i64]
  let unpackW   ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64, .i64]
  let extractB  ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64, .i64]
  let writeByte ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64, .i64, .i64]
  let writeTab  ← declareBlock [.i64, .i64, .i64, .i64, .i64]
  let itoaWr    ← declareBlock [.i64, .i64, .i64, .i64, .i64, .i64]
  let writeNL   ← declareBlock [.i64, .i64, .i64, .i64]
  let writeFile ← declareBlock [.i64]

  let k : WcCtx := {
    ptr, dataPtr, fnRead, fnWrite, fnHtInit, fnHtClean := fnHtClean,
    fnCreate, fnIncr, fnCount, fnGetEntry := fnGet, zero,
    cpIn, cpOut1, cpOut, readBlk, skipWS, chkByte,
    readWord, readByte, wordDone, accumByte,
    fmtStart, fmtLoop, getEntry, unpackW, extractB, writeByte,
    writeTab, itoaWr, writeNL, writeFile
  }

  emitPathCopy k

  -- Init HT, read file, create HT, start parse
  startBlock readBlk
  callVoid fnHtInit [← absAddr ptr 0]
  let fileSize ← readFile ptr fnRead INPUT_PATH_OFF INPUT_DATA
  let inputBase← absAddr ptr INPUT_DATA
  let ctxPtr   ← load64 (← absAddr ptr 0)
  let _        ← call fnCreate [ctxPtr]
  jump skipWS.ref [zero, ctxPtr]

  emitParsePhase k fileSize inputBase
  emitFormatPhase k

def clifIR : String := buildProgram mainFn

def artifacts : Array Json :=
  #[toJsonEntry "wc_algorithm" {
    cranelift_ir := clifIR,
    memory_size := MEM_SIZE,
    context_offset := 0
  } {
    actions := mkCallActions 1,
    cranelift_units := 0,
    timeout_ms := some TIMEOUT_MS
  }]

end WordCountBench
