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
  /-- The image memory starts with, built here as one flat list and written out
      as the segments of it that are not zero. -/
  initial_memory : List UInt8 := []

namespace ContextSlots

def ht : Nat := 0x00
def wgpu : Nat := 0x08
def cuda : Nat := 0x10
-- 0x18..0x38 is unused; the window context pointer sits past it, where it
-- has always been, so no generator's layout moves.
def window : Nat := 0x38

end ContextSlots

/-- The image as the segments it is written out as: the runs of bytes that are
    not zero, with runs of `gap` zeros or more left out. Memory starts zeroed,
    so a zero costs nothing to leave out and two bytes to write down; across
    the artifacts here the images are 29.7 MB, of which 4.9 MB is not zero.

    `gap` trades segments against bytes: a smaller one drops more zeros and
    emits more segments. -/
def segmentsOf (image : List UInt8) (gap : Nat := 32) : List (Nat × List UInt8) :=
  let rec go (i : Nat) (bs : List UInt8) (cur : Option (Nat × List UInt8))
      (zeros : Nat) (acc : List (Nat × List UInt8)) : List (Nat × List UInt8) :=
    match bs with
    | [] =>
        match cur with
        | some (off, run) => acc ++ [(off, run.reverse)]
        | none => acc
    | b :: rest =>
        if b == 0 then
          match cur with
          | none => go (i + 1) rest none 0 acc
          | some (off, run) =>
              if zeros + 1 ≥ gap then
                -- The run of zeros is long enough to leave out: close the
                -- segment before it, dropping the zeros already taken in.
                let kept := run.drop zeros
                go (i + 1) rest none 0 (acc ++ [(off, kept.reverse)])
              else go (i + 1) rest (some (off, b :: run)) (zeros + 1) acc
        else
          match cur with
          | none => go (i + 1) rest (some (i, [b])) 0 acc
          | some (off, run) => go (i + 1) rest (some (off, b :: run)) 0 acc
  go 0 image none 0 []

instance : ToJson Artifact where
  toJson a := Json.mkObj [
    ("functions", Json.arr ((a.functions.map toJson).toArray)),
    ("memory_size", toJson a.memory_size),
    ("data", Json.arr ((segmentsOf a.initial_memory).map (fun (off, bs) =>
      Json.mkObj [("offset", toJson off), ("bytes", toJson bs)])).toArray)
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
