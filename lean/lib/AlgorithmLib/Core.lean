import Lean

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

/-- Offsets in shared memory where the Rust host writes the caller's input and
    output buffer pointers + lengths before each `execute`. CLIF code reads from
    these slots to access the caller's buffers. -/
structure IoOffsets where
  dataPtr : Nat := 0x18
  dataLen : Nat := 0x20
  outPtr  : Nat := 0x28
  outLen  : Nat := 0x30
  deriving Repr

instance : ToJson IoOffsets where
  toJson h := Json.mkObj [
    ("data_ptr", toJson h.dataPtr),
    ("data_len", toJson h.dataLen),
    ("out_ptr",  toJson h.outPtr),
    ("out_len",  toJson h.outLen)
  ]

namespace IoOffsets

def default : IoOffsets := {}

def byteSize : Nat := 56

end IoOffsets

structure Setup where
  cranelift_ir : String
  memory_size : Nat
  io_offsets : IoOffsets := {}
  initial_memory : List UInt8 := []
  deriving Repr

namespace ContextSlots

def ht : Nat := 0x00
def wgpu : Nat := 0x08
def cuda : Nat := 0x10
-- 0x18..0x38 is the IoOffsets region (data/out ptr+len); the window context
-- pointer goes in the first free 8-byte slot after it.
def window : Nat := 0x38

end ContextSlots

instance : ToJson Setup where
  toJson c := Json.mkObj [
    ("cranelift_ir", toJson c.cranelift_ir),
    ("memory_size", toJson c.memory_size),
    ("io_offsets", toJson c.io_offsets),
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

/-- Emit a wrapper CLIF function at index `wrapperIdx` that calls each of `callees`
    in order (passing the base pointer `v0` unchanged) and returns. Used by algorithms
    that need to invoke multiple helper CLIF functions in sequence from a single
    `Algorithm.fn_idx`. The result string is meant to be appended to the existing
    CLIF IR for the program. Distinct callee indices are declared once and reused
    across calls so a 64-deep stack of one callee produces 1 decl + 64 calls. -/
def clifSequenceWrapper (wrapperIdx : Nat) (callees : List Nat) : String :=
  let unique : List Nat := callees.foldl (fun acc x => if acc.contains x then acc else acc ++ [x]) []
  let slotOf (c : Nat) : Nat := (unique.idxOf? c).getD 0
  let header := s!"\nfunction u0:{wrapperIdx}(i64) system_v \{\n    sig0 = (i64) system_v\n"
  let fnDecls := String.join <|
    unique.zipIdx.map fun (callee, i) => s!"    fn{i} = colocated u0:{callee} sig0\n"
  let body := String.join <|
    callees.map fun c => s!"    call fn{slotOf c}(v0)\n"
  header ++ fnDecls ++ "block0(v0: i64):\n" ++ body ++ "    return\n}\n"

end AlgorithmLib
