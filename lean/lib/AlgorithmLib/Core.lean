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

structure RuntimeHeader where
  dataPtrOffset : Nat := 0x18
  dataLenOffset : Nat := 0x20
  outPtrOffset : Nat := 0x28
  outLenOffset : Nat := 0x30
  deriving Repr

instance : ToJson RuntimeHeader where
  toJson h := Json.mkObj [
    ("data_ptr_offset", toJson h.dataPtrOffset),
    ("data_len_offset", toJson h.dataLenOffset),
    ("out_ptr_offset", toJson h.outPtrOffset),
    ("out_len_offset", toJson h.outLenOffset)
  ]

namespace RuntimeHeader

def default : RuntimeHeader := {}

def byteSize : Nat := 56

end RuntimeHeader

structure BaseConfig where
  cranelift_ir : String
  memory_size : Nat
  runtime_header : RuntimeHeader := {}
  context_offset : Nat := 0
  initial_memory : List UInt8 := []
  deriving Repr

namespace ContextSlots

def ht : Nat := 0x00
def wgpu : Nat := 0x08
def cuda : Nat := 0x10

end ContextSlots

instance : ToJson BaseConfig where
  toJson c := Json.mkObj [
    ("cranelift_ir", toJson c.cranelift_ir),
    ("memory_size", toJson c.memory_size),
    ("runtime_header", toJson c.runtime_header),
    ("context_offset", toJson c.context_offset),
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

/-- Serialize a named artifact entry: ["name", config, algorithm].
    Use this in def main to declare artifact names that the build system will use as file names. -/
def toJsonEntry (name : String) (config : BaseConfig) (algorithm : Algorithm) : Json :=
  .arr #[.str name, toJson config, toJson algorithm]

/-- Parse the sole CLI argument as an output directory. -/
def requireOutputDir (args : List String) : IO String :=
  match args with
  | [dir] => pure dir
  | _ => throw <| IO.userError "expected exactly one argument: output directory"

/-- Emit artifacts to a directory as one `{name}.json` file per entry, where each
    file contains `[config, algorithm]`. -/
def emitArtifacts (dir : String) (entries : Array Json) : IO Unit := do
  IO.FS.createDirAll dir
  let mut seen : List String := []
  for entry in entries do
    match entry with
    | .arr #[.str name, config, algorithm] =>
        if seen.contains name then
          throw <| IO.userError s!"duplicate artifact name: {name}"
        seen := name :: seen
        IO.FS.writeFile s!"{dir}/{name}.json" (Json.compress (.arr #[config, algorithm]))
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
