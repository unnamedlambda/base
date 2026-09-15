import Lean
import AlgorithmLib.ClifData

open Lean

namespace AlgorithmLib


instance : ToJson UInt8 where
  toJson n := toJson n.toNat

instance : ToJson (List UInt8) where
  toJson lst := toJson (lst.map (·.toNat))

instance : ToJson UInt32 where
  toJson n := toJson n.toNat

instance : ToJson UInt64 where
  toJson n := toJson n.toNat

structure Setup where
  clif : IR.Program
  memory_size : Nat
  initial_memory : List UInt8 := []

namespace ContextSlots

def ht : Nat := 0x00
def wgpu : Nat := 0x08
def cuda : Nat := 0x10
-- 0x18..0x38 is unused; the window context pointer sits past it, where it
-- has always been, so no generator's layout moves.
def window : Nat := 0x38

end ContextSlots

instance : ToJson Setup where
  toJson c := Json.mkObj [
    ("clif", toJson c.clif),
    ("memory_size", toJson c.memory_size),
    ("initial_memory", toJson c.initial_memory)
  ]

structure Algorithm where
  fn_idx : UInt32
  output : List Json := []

instance : ToJson Algorithm where
  toJson alg := Json.mkObj [
    ("fn_idx", toJson alg.fn_idx),
    ("output", Json.arr alg.output.toArray)
  ]

/- Output-schema JSON builders. `Algorithm.output` is a list of these schema
   objects; each becomes one Arrow RecordBatch. The CLIF code must store the
   row count at `row_count_offset` and the column data at each column's
   `data_offset` (little-endian, 8 bytes/row for I64/F64). This is what lets a
   Rust test read a generated algorithm's result back as typed columns. -/
namespace Output

inductive Ty where
  | i64 | f64 | utf8

def Ty.name : Ty → String
  | .i64 => "I64"
  | .f64 => "F64"
  | .utf8 => "Utf8"

/-- One output column. `lenOffset` is only used for Utf8 (total byte length). -/
def column (name : String) (ty : Ty) (dataOffset : Nat) (lenOffset : Nat := 0) : Json :=
  Json.mkObj [
    ("name", .str name),
    ("dtype", .str ty.name),
    ("data_offset", toJson dataOffset),
    ("len_offset", toJson lenOffset)
  ]

/-- One output batch schema (a set of columns + where the row count is stored). -/
def schema (columns : List Json) (rowCountOffset : Nat) : Json :=
  Json.mkObj [
    ("columns", Json.arr columns.toArray),
    ("row_count_offset", toJson rowCountOffset)
  ]

end Output

/-- Serialize an artifact: ["fileName", { setup, main, extras }]. `main` is
    the primary entry point algorithm; `extras` is a JSON object mapping any
    additional named algorithms (e.g., prep/infer stages of a pipeline). Use
    `toJsonEntry` for the common single-algorithm case. For pipelines without a
    single primary step, `main` is the entry point you call first (typically
    the load/init step) and the remaining stages go into `extras`. -/
def toJsonArtifact (name : String) (setup : Setup) (main : Algorithm)
    (extras : List (String × Algorithm) := []) : Json :=
  let extrasMap := Json.mkObj (extras.map fun (n, a) => (n, toJson a))
  .arr #[.str name, Json.mkObj [
    ("setup", toJson setup),
    ("main", toJson main),
    ("extras", extrasMap)
  ]]

/-- Serialize a single-algorithm artifact. -/
def toJsonEntry (name : String) (setup : Setup) (algorithm : Algorithm) : Json :=
  toJsonArtifact name setup algorithm

/-- Parse the sole CLI argument as an output directory. -/
def requireOutputDir (args : List String) : IO String :=
  match args with
  | [dir] => pure dir
  | _ => throw <| IO.userError "expected exactly one argument: output directory"

/-- Emit artifacts to a directory as one `{name}.json` file per entry, where each
    file contains `{ setup, main, extras }`. -/
def emitArtifacts (dir : String) (entries : Array Json) : IO Unit := do
  IO.FS.createDirAll dir
  let mut seen : List String := []
  for entry in entries do
    match entry with
    | .arr #[.str name, body] =>
        if seen.contains name then
          throw <| IO.userError s!"duplicate artifact name: {name}"
        seen := name :: seen
        IO.FS.writeFile s!"{dir}/{name}.json" (Json.compress body)
    | _ =>
        throw <| IO.userError s!"invalid artifact entry: {Json.compress entry}"

def u32 (n : Nat) : UInt32 := UInt32.ofNat n

end AlgorithmLib
