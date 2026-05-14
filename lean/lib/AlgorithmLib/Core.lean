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

inductive Kind where
  | Describe
  | ClifCall
  | ConditionalJump
  | ClifCallAsync
  | Wait
  | WaitUntil
  | Park
  | Wake
  deriving Repr

instance : ToJson Kind where
  toJson
    | .Describe => "describe"
    | .ClifCall => "clif_call"
    | .ConditionalJump => "conditional_jump"
    | .ClifCallAsync => "clif_call_async"
    | .Wait => "wait"
    | .WaitUntil => "wait_until"
    | .Park => "park"
    | .Wake => "wake"

structure Action where
  kind : Kind
  dst : UInt32
  src : UInt32
  offset : UInt32
  size : UInt32
  deriving Repr

instance : ToJson Action where
  toJson a := Json.mkObj [
    ("kind", toJson a.kind),
    ("dst", toJson a.dst),
    ("src", toJson a.src),
    ("offset", toJson a.offset),
    ("size", toJson a.size)
  ]

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
  actions : List Action
  cranelift_units : Nat
  timeout_ms : Option Nat
  output : List Json := []

instance : ToJson Algorithm where
  toJson alg := Json.mkObj [
    ("actions", toJson alg.actions),
    ("cranelift_units", toJson alg.cranelift_units),
    ("timeout_ms", toJson alg.timeout_ms),
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

/-- Common single-call action list used by most benchmark artifacts. -/
def mkCallActions (src : UInt32) : List Action :=
  [{ kind := .ClifCall, dst := 0, src := src, offset := 0, size := 0 }]

def u32 (n : Nat) : UInt32 := UInt32.ofNat n

end AlgorithmLib
