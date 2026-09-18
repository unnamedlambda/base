import AlgorithmLib.Gen
import ShipScan

/-!
# Output that says what it is

An artifact carries no description of what its entry points answer: the bytes a
program writes mean what the program and its host agree they mean. This
generator is the pattern for when a host should not have to take that on trust
from a copy of this file. Three entries over one byte histogram:

* `stats` answers a CBOR map. The format is in the bytes — a generic decoder
  reads it, and the map names its own format — so a host needs no layout of it.
* `bulk` answers raw bytes behind an eight-byte format id. Bulk data is where a
  self-describing encoding costs, so the description is one check a reader does
  once: compare the id, then read the layout it names.
* `schema` answers, as CBOR, what the other two answer. It is an ordinary entry,
  so a host that wants a description before calling anything asks for one the
  same way it asks for anything else.

Every entry answers the size of its output and writes it only when the caller's
buffer holds that much, so a host can ask with an empty buffer and call again.

The formats are this file's: the CBOR is built here, byte for byte, and the
program fills in the few values only a run knows.
-/

open AlgorithmLib
open AlgorithmLib.Layout
open AlgorithmLib.IR
open AlgorithmLib.Prog

namespace SelfDescribing

/-- What `stats` says its format is. -/
def statsFormat : String := "base.u8stats/1"

/-- The eight bytes `bulk` answers first. -/
def bulkId : String := "u8hist01"

/-- RFC 8746: a byte string holding unsigned 64-bit little-endian integers. -/
def uint64leTag : Nat := 71

def HIST_BYTES : Nat := 256 * 8

private def bytesOf (w : Cbor.W Unit) : List UInt8 :=
  match Cbor.run w with
  | .ok b => b.toList
  | .error e => panic! e

/-- A text key or value. -/
private def txt (s : String) : List UInt8 := bytesOf (Cbor.text s)

/-- `stats`' answer is three pieces with the run's values between them:

    `{"format": statsFormat, "count": <u64>, "sum": <u64>, "histogram": 71(<bytes>)}`

    Each value a run fills is an integer with an eight-byte argument, which is
    valid CBOR at any value and puts it at an offset this file knows. -/
def statsBeforeCount : List UInt8 :=
  bytesOf (Cbor.head 5 4) ++ txt "format" ++ txt statsFormat ++ txt "count" ++ [0x1b]
def statsBeforeSum : List UInt8 := txt "sum" ++ [0x1b]
def statsBeforeHist : List UInt8 :=
  txt "histogram" ++ bytesOf (do Cbor.tag uint64leTag; Cbor.head 2 HIST_BYTES)

def statsCountAt : Nat := statsBeforeCount.length
def statsSumAt : Nat := statsCountAt + 8 + statsBeforeSum.length
def statsHistAt : Nat := statsSumAt + 8 + statsBeforeHist.length
def statsSize : Nat := statsHistAt + HIST_BYTES

def statsTemplate : List UInt8 :=
  statsBeforeCount ++ List.replicate 8 0 ++ statsBeforeSum ++ List.replicate 8 0
    ++ statsBeforeHist ++ List.replicate HIST_BYTES 0

/-- `bulk`'s answer: the id, then the histogram. -/
def bulkSize : Nat := bulkId.utf8ByteSize + HIST_BYTES

open Cbor in
/-- What `schema` answers. -/
def schemaBytes : List UInt8 := bytesOf <| struct [
  ("stats", struct [
    ("encoding", text "cbor"),
    ("format", text statsFormat),
    ("fields", struct [
      ("format", text "text"),
      ("count", text "uint"),
      ("sum", text "uint"),
      ("histogram", text "uint64le[256], tag 71")])]),
  ("bulk", struct [
    ("encoding", text "raw"),
    ("format_id", bytes bulkId.toUTF8),
    ("layout", array ["format_id", "uint64le[256]"] text)])]

structure Fields where
  reserved : Fld (.bytes 64)
  schema   : Fld (.bytes schemaBytes.length)
  stats    : Fld (.bytes statsSize)
  /-- The id, then the histogram `bulk` counts into. -/
  bulk     : Fld (.bytes bulkSize)

def mkLayout : Fields × LayoutMeta := Layout.build do
  let reserved ← field (.bytes 64)
  let schema ← field (.bytes schemaBytes.length)
  let stats ← field (.bytes statsSize)
  let bulk ← field (.bytes bulkSize)
  pure { reserved, schema, stats, bulk }

def f : Fields := mkLayout.1
def layoutMeta : LayoutMeta := mkLayout.2

/-- Count each byte of the input into 256 little-endian `u64`s at `hist`,
    answering the bytes' sum. -/
def histogram (data len hist : V .i64) : Prog V L (V .i64) := do
  let zero ← iconst64 0
  forLoop (← iconst64 256) fun b => do
    storeI64 zero (← iadd hist (← ishlImm b 3))
  forLoopAcc len (← iconst64 0) fun i acc => do
    let b ← uload8_64 (← iadd data i)
    let slot ← iadd hist (← ishlImm b 3)
    storeI64 (← iaddImm (← load64 slot) 1) slot
    iadd acc b

/-- `v` as eight big-endian bytes at `addr`: a CBOR argument. -/
def storeBE (v addr : V .i64) : Prog V L Unit := do
  for k in [0:8] do
    istore8 (← ushrImm v (Int.ofNat (56 - 8 * k))) (← iaddImm addr (Int.ofNat k))

/-- Answer the `size` bytes at `src`: copy them to the caller's buffer if it
    holds them, and say how many there are either way. -/
def answer (src : V .i64) (size : Nat) : Prog V L (V .i64) := do
  let out ← outPtr
  let cap ← outLen
  let need ← iconst64 size
  when .uge cap need do
    forLoop need fun i => do
      istore8 (← uload8_64 (← iadd src i)) (← iadd out i)
  pure need

def statsCode : Prog V L (V .i64) := do
  let base ← basePtr
  let data ← dataPtr
  let len ← dataLen
  let t ← fldAddr base f.stats
  let sum ← histogram data len (← iaddImm t statsHistAt)
  storeBE len (← iaddImm t statsCountAt)
  storeBE sum (← iaddImm t statsSumAt)
  answer t statsSize

def bulkCode : Prog V L (V .i64) := do
  let base ← basePtr
  let data ← dataPtr
  let len ← dataLen
  let t ← fldAddr base f.bulk
  let _ ← histogram data len (← iaddImm t bulkId.utf8ByteSize)
  answer t bulkSize

def schemaCode : Prog V L (V .i64) := do
  let base ← basePtr
  answer (← fldAddr base f.schema) schemaBytes.length

def clifIR : Except String (List FuncData) :=
  Prog.program [.ok noopFunction,
    Prog.entry "schema" (Prog.compileProgStatus 1 schemaCode),
    Prog.entry "stats" (Prog.compileProgStatus 2 statsCode),
    Prog.entry "bulk" (Prog.compileProgStatus 3 bulkCode)]

def payload : List UInt8 :=
  mkPayload layoutMeta.totalSize [
    f.schema.init schemaBytes,
    f.stats.init statsTemplate,
    f.bulk.init bulkId.toUTF8.toList]

def setup (clif : List FuncData) : Artifact := {
  functions := clif,
  required_memory := layoutMeta.totalSize,
  initial_memory := payload
}

def shipped : Except String Artifact := do return setup (← clifIR)

end SelfDescribing

def main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  emitArtifacts outDir #[artifactEntry "self_describing" (← Prog.orDie SelfDescribing.shipped)]

#eval ShipScan.check "SelfDescribingAlgorithm" (root := `SelfDescribing.shipped)
