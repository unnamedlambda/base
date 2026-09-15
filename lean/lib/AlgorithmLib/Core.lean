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

/-- The board's control, as data: the functions a host compiles, the memory
    they run in, and the bytes that memory starts with. This is the whole of
    the wire format.

    An entry point is a function index. Which index does what is this
    generator's knowledge, and stays here rather than travelling with the
    artifact; a host that did not build it gets those numbers from a library
    written beside the generator. -/
structure Artifact where
  functions : List IR.FuncData
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

instance : ToJson Artifact where
  toJson a := Json.mkObj [
    ("functions", Json.arr ((a.functions.map toJson).toArray)),
    ("memory_size", toJson a.memory_size),
    ("initial_memory", toJson a.initial_memory)
  ]

/-- Serialize an artifact: ["fileName", artifact]. -/
def toJsonArtifact (name : String) (artifact : Artifact) : Json :=
  .arr #[.str name, toJson artifact]

/-- Parse the sole CLI argument as an output directory. -/
def requireOutputDir (args : List String) : IO String :=
  match args with
  | [dir] => pure dir
  | _ => throw <| IO.userError "expected exactly one argument: output directory"

/-- Emit artifacts to a directory as one `{name}.json` file per entry. -/
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
