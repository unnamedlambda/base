import Lean
import AlgorithmLib.Gen
import LayoutScan

open Lean
open AlgorithmLib
open AlgorithmLib.IR

namespace WordCountBench

/-
  Word frequency counting: parse words, ht_increment, format word\tcount\n output.
  Payload: "input_path\0output_path\0"
  HT context at offset 0x00, ht_create/ht_increment/ht_count/ht_get_entry.
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

open AlgorithmLib.Prog


abbrev fnHtInit : Ffi := .htInit
abbrev fnHtClean : Ffi := .htCleanup
abbrev fnRead : Ffi := .fileRead
abbrev fnWrite : Ffi := .fileWrite
abbrev fnCreate : Ffi := .htCreate
abbrev fnIncr : Ffi := .htIncrement
abbrev fnCount : Ffi := .htCount
abbrev fnGetEntry : Ffi := .htGetEntry

/-- A NUL-terminated string copied from the payload into `dstOff`; the result is
    the source index just past the terminator. -/
def emitCopyPath (ptr dataPtr srcStart : V .i64) (dstOff : Nat) : Prog V L (V .i64) := do
  let zero ← iconst64 0
  let e ← dwloop %[srcStart, zero] .ne zero true [0]
    (body := fun c => do
      let si := c.head
      let di := c.snd
      let ch ← uload8_64 (← iadd dataPtr si)
      istore8 ch (← iadd (← absAddr ptr dstOff) di)
      return (ch, %[← iaddImm si 1, ← iaddImm di 1]))
  return e.head

/-- Words separated by anything at or below `' '`, each packed into eight bytes
    of a `u64` key and counted. A ninth byte ends the word. -/
def emitParsePhase (ptr ctxPtr fileSize inputBase : V .i64) : Prog V L Unit := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let eight ← iconst64 8
  let sp ← iconst64 32
  let keyAddr ← absAddr ptr CURRENT_KEY
  let keyLen ← iconst32 8
  let _ ← wloopL %[zero]
    (head := fun _ c => return (contIf .slt c.head fileSize, %[], ()))
    (body := fun posLoop c _ => do
      let pos := c.head
      let byte ← uload8_64 (← iadd inputBase pos)
      when .ule byte sp (continueWith posLoop %[← iaddImm pos 1])
      let word ← wloopL %[pos, zero, zero]
        (head := fun _ w =>
          return (contIf .slt w.head fileSize, %[w.head, w.snd], ()))
        (body := fun wordLoop w _ => do
          let p := w.head
          let wAcc := w.snd
          let bIdx := w.thd
          let b ← uload8_64 (← iadd inputBase p)
          when .ule b sp (brk wordLoop %[p, wAcc])
          let shift ← imul bIdx eight
          let shifted ← ishl b shift
          let wAcc' ← bor wAcc shifted
          let bIdx' ← iaddImm bIdx 1
          let p' ← iaddImm p 1
          when .sge bIdx' eight (brk wordLoop %[p', wAcc'])
          return %[p', wAcc', bIdx'])
      store word.snd keyAddr
      let _ ← ffi fnIncr %[ctxPtr, keyAddr, keyLen, one]
      return %[word.head])

/-- Each entry written as `word\tcount\n`; the result is the length written. -/
def emitFormatPhase (ptr ctxPtr : V .i64) : Prog V L (V .i64) := do
  let zero ← iconst64 0
  let one ← iconst64 1
  let eight ← iconst64 8
  let ten ← iconst64 10
  let byteFF ← iconst64 0xFF
  let keyAddr ← absAddr ptr CURRENT_KEY
  let valAddr ← absAddr ptr RESULT_SLOT
  let cnt ← uextend64 (← ffi fnCount %[ctxPtr])
  let outStart ← iconst64 OUTPUT_BUF
  let e ← wloop2 zero outStart
    (head := fun idx opos => return (contIf .slt idx cnt, %[opos], ()))
    (body := fun idx opos _ => do
      let idx32 ← ireduce32 idx
      let _ ← ffi fnGetEntry %[ctxPtr, idx32, keyAddr, valAddr]
      let word ← load64 keyAddr
      let count ← load64 valAddr
      -- the key's bytes, least significant first, up to the first NUL
      let u ← wloopL %[opos, zero]
        (head := fun _ b2 => return (contIf .slt b2.snd eight, %[b2.head], ()))
        (body := fun byteLoop b2 _ => do
          let op := b2.head; let bi := b2.snd
          let shift ← imul bi eight
          let b ← band (← ushr word shift) byteFF
          when .eq b zero (brk byteLoop %[op])
          istore8 b (← iadd ptr op)
          return %[← iaddImm op 1, ← iaddImm bi 1])
      let tabPos := u.head
      istore8 (← iconst64 9) (← iadd ptr tabPos)
      -- the highest power of ten at or below the count, then a digit per step
      let finalDiv ← wloop1 one
        (head := fun d => do
          let t ← imul d ten
          return (contIf .ule t count, %[d], ()))
        (body := fun d _ => return %[← imul d ten])
      let it ← dwloop %[← iaddImm tabPos 1, count, finalDiv.head] .ne zero true [0]
        (body := fun c => do
          let op := c.head
          let rem := c.snd
          let dv := c.thd
          let dig ← udiv rem dv
          istore8 (← iadd dig (← iconst64 48)) (← iadd ptr op)
          let rem' ← isub rem (← imul dig dv)
          let dv' ← udiv dv ten
          return (dv', %[← iaddImm op 1, rem', dv']))
      let nlPos := it.head
      istore8 (← iconst64 10) (← iadd ptr nlPos)
      return %[← iaddImm idx 1, ← iaddImm nlPos 1])
  return e.head

def mainCode : Prog V L Unit := do
  let ptr ← basePtr
  let dataPtr ← dataPtr
  let zero ← iconst64 0

  let afterIn ← emitCopyPath ptr dataPtr zero INPUT_PATH_OFF
  let _ ← emitCopyPath ptr dataPtr afterIn OUTPUT_PATH_OFF

  ffiVoid fnHtInit %[← absAddr ptr 0]
  let fileSize ← readFile ptr INPUT_PATH_OFF INPUT_DATA
  let inputBase ← absAddr ptr INPUT_DATA
  let ctxPtr ← load64 (← absAddr ptr 0)
  let _ ← ffi fnCreate %[ctxPtr]

  emitParsePhase ptr ctxPtr fileSize inputBase
  let outEnd ← emitFormatPhase ptr ctxPtr
  istore8 (← iconst32 0) (← iadd ptr outEnd)
  let _ ← writeFile ptr OUTPUT_PATH_OFF OUTPUT_BUF zero zero
  ffiVoid fnHtClean %[← absAddr ptr 0]

def clifIR : Except String (List FuncData) :=
  Prog.program [.ok noopFunction, Prog.entry "main" (Prog.compileProg 1 mainCode)]

/-- Every byte of shared memory this program names.

    The offsets are assigned by hand, and a collision between two of them is
    invisible at every other layer: both stores succeed and the second wins.
    `0x00`-`0x18` are the context slots the FFI's init calls fill, so naming
    those is what stops an offset being placed where one of those calls will
    overwrite it. -/
def memMap : AlgorithmLib.Layout.RegionMap :=
  [⟨"current_key", CURRENT_KEY, 8⟩,
   ⟨"new_value",   NEW_VALUE, 8⟩,
   ⟨"input_path",  INPUT_PATH_OFF, OUTPUT_PATH_OFF - INPUT_PATH_OFF⟩,
   ⟨"output_path", OUTPUT_PATH_OFF, RESULT_SLOT - OUTPUT_PATH_OFF⟩,
   ⟨"result_slot", RESULT_SLOT, 8⟩,
   ⟨"output_buf",  OUTPUT_BUF, INPUT_DATA - OUTPUT_BUF⟩,
   ⟨"input_data",  INPUT_DATA, MAX_TEXT_BYTES⟩]


#eval LayoutScan.check "WordCountAlgorithm" [``memMap]
theorem memMap_ok : AlgorithmLib.Layout.RegionMap.okB memMap = true := by decide

theorem memMap_within :
    AlgorithmLib.Layout.RegionMap.withinB MEM_SIZE memMap = true := by decide

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[artifactEntry "wc_algorithm" {
    functions := clif,
    required_memory := MEM_SIZE
  }]

end WordCountBench
