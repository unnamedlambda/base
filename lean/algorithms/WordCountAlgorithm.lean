import Lean
import AlgorithmLib.Gen

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

open AlgorithmLib.HProg
open AlgorithmLib.HProg.Sur

/-- Every emitted function declares the same externals in the same order. -/
def env : FnEnv := env% [.ht, .fileIO]

def fnHtInit : FnRef := IR.Ffi.htInit.ref
def fnHtClean : FnRef := IR.Ffi.htCleanup.ref
def fnRead : FnRef := IR.Ffi.fileRead.ref
def fnWrite : FnRef := IR.Ffi.fileWrite.ref
def fnCreate : FnRef := IR.Ffi.htCreate.ref
def fnIncr : FnRef := IR.Ffi.htIncrement.ref
def fnCount : FnRef := IR.Ffi.htCount.ref
def fnGetEntry : FnRef := IR.Ffi.htGetEntry.ref

/-- A NUL-terminated string copied from the payload into `dstOff`; the result is
    the source index just past the terminator. -/
def emitCopyPath (ptr dataPtr srcStart : R) (dstOff : Nat) : M R := do
  let zero ← iconst64 0
  let e ← dwloop [srcStart, zero] .ne zero true [0]
    (body := fun c => do
      let si := c.headD 0
      let di := c.getD 1 0
      let ch ← uload8_64 (← iadd dataPtr si)
      istore8 ch (← iadd (← absAddr ptr dstOff) di)
      return (ch, [← iaddImm si 1, ← iaddImm di 1]))
  return e.headD 0

/-- Words separated by anything at or below `' '`, each packed into eight bytes
    of a `u64` key and counted. A ninth byte ends the word. -/
def emitParsePhase (ptr ctxPtr fileSize inputBase : R) : M Unit := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let eight ← iconst64 8
  let sp ← iconst64 32
  let keyAddr ← absAddr ptr CURRENT_KEY
  let keyLen ← iconst32 8
  let _ ← wloop1 zero
    (head := fun pos => return (contIf .slt pos fileSize, ([] : List R), ()))
    (body := fun pos _ => do
      let byte ← uload8_64 (← iadd inputBase pos)
      when .ule byte sp (continueWith [← iaddImm pos 1])
      let word ← wloop [pos, zero, zero]
        (head := fun c =>
          return (contIf .slt (c.headD 0) fileSize, [c.headD 0, c.getD 1 0], ()))
        (body := fun c _ => do
          let p := c.headD 0
          let wAcc := c.getD 1 0
          let bIdx := c.getD 2 0
          let b ← uload8_64 (← iadd inputBase p)
          when .ule b sp (brk [p, wAcc])
          let shift ← imul bIdx eight
          let shifted ← ishl b shift
          let wAcc' ← bor wAcc shifted
          let bIdx' ← iaddImm bIdx 1
          let p' ← iaddImm p 1
          when .sge bIdx' eight (brk [p', wAcc'])
          return [p', wAcc', bIdx'])
      store (word.getD 1 0) keyAddr
      let _ ← call fnIncr.id [ctxPtr, keyAddr, keyLen, one]
      return [word.headD 0])

/-- Each entry written as `word\tcount\n`; the result is the length written. -/
def emitFormatPhase (ptr ctxPtr : R) : M R := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let eight ← iconst64 8
  let ten ← iconst64 10
  let byteFF ← iconst64 0xFF
  let keyAddr ← absAddr ptr CURRENT_KEY
  let valAddr ← absAddr ptr RESULT_SLOT
  let cnt ← uextend64 (← call fnCount.id [ctxPtr])
  let outStart ← iconst64 OUTPUT_BUF
  let e ← wloop2 zero outStart
    (head := fun idx opos => return (contIf .slt idx cnt, [opos], ()))
    (body := fun idx opos _ => do
      let idx32 ← ireduce32 idx
      let _ ← call fnGetEntry.id [ctxPtr, idx32, keyAddr, valAddr]
      let word ← load64 keyAddr
      let count ← load64 valAddr
      -- the key's bytes, least significant first, up to the first NUL
      let u ← wloop2 opos zero
        (head := fun op bi => return (contIf .slt bi eight, [op], ()))
        (body := fun op bi _ => do
          let shift ← imul bi eight
          let b ← band (← ushr word shift) byteFF
          when .eq b zero (brk [op])
          istore8 b (← iadd ptr op)
          return [← iaddImm op 1, ← iaddImm bi 1])
      let tabPos := u.headD 0
      istore8 (← iconst64 9) (← iadd ptr tabPos)
      -- the highest power of ten at or below the count, then a digit per step
      let finalDiv ← wloop1 one
        (head := fun d => do
          let t ← imul d ten
          return (contIf .ule t count, [d], ()))
        (body := fun d _ => return [← imul d ten])
      let it ← dwloop [← iaddImm tabPos 1, count, finalDiv.headD 0] .ne zero true [0]
        (body := fun c => do
          let op := c.headD 0
          let rem := c.getD 1 0
          let dv := c.getD 2 0
          let dig ← udiv rem dv
          istore8 (← iadd dig (← iconst64 48)) (← iadd ptr op)
          let rem' ← isub rem (← imul dig dv)
          let dv' ← udiv dv ten
          return (dv', [← iaddImm op 1, rem', dv']))
      let nlPos := it.headD 0
      istore8 (← iconst64 10) (← iadd ptr nlPos)
      return [← iaddImm idx 1, ← iaddImm nlPos 1])
  return e.headD 0

def mainCode : HProg.Code :=
  clif%(env, HProg.ptrParams) do
  let ptr := basePtr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let zero ← iconst64 0

  let afterIn ← emitCopyPath ptr dataPtr zero INPUT_PATH_OFF
  let _ ← emitCopyPath ptr dataPtr afterIn OUTPUT_PATH_OFF

  callVoid fnHtInit.id [← absAddr ptr 0]
  let fileSize ← readFile ptr fnRead INPUT_PATH_OFF INPUT_DATA
  let inputBase ← absAddr ptr INPUT_DATA
  let ctxPtr ← load64 (← absAddr ptr 0)
  let _ ← call fnCreate.id [ctxPtr]

  emitParsePhase ptr ctxPtr fileSize inputBase
  let outEnd ← emitFormatPhase ptr ctxPtr
  istore8 (← iconst32 0) (← iadd ptr outEnd)
  let _ ← writeFile ptr fnWrite OUTPUT_PATH_OFF OUTPUT_BUF zero zero
  callVoid fnHtClean.id [← absAddr ptr 0]

def clifIR : Program :=
  IR.program [IR.noopFunction, HProg.compileFn 1 mainCode env]

def artifacts : Array Json :=
  #[toJsonEntry "wc_algorithm" {
    clif := clifIR,
    memory_size := MEM_SIZE
  } {
    fn_idx := u32 1
  }]

end WordCountBench
