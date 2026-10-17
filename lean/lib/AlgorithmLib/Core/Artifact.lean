module
public import Lean
public import AlgorithmLib.Core.ClifData
meta import AlgorithmLib.Core.ClifData
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open Lean

namespace AlgorithmLib

/-- The board's control, as data: the functions a host compiles, the arena
    they run in, and the bytes the arena starts with. This is the whole of the
    wire format, written as the CBOR `base_types::Artifact` reads, with the
    primitives its functions call declared once.

    A host calls a function by its `name`. -/
structure Artifact where
  functions : List IR.FuncData
  /-- How much arena the program needs. The runtime allocates it and cannot
      check it, so what a generator proved its layout fits in is what belongs
      here. What is written out is this or the image's length, whichever is
      more (`Artifact.memorySize`), so the image always fits. -/
  required_memory : Nat
  /-- The image the arena starts with, built here as one flat list and written
      out as the segments of it that are not zero. -/
  initial_memory : List UInt8 := []

namespace ContextSlots

def ht : Nat := 0x00
def wgpu : Nat := 0x08
def cuda : Nat := 0x10
-- 0x18..0x38 is written by nothing. It held the caller's buffer descriptors
-- before those became entry arguments, and the window slot stayed where it
-- was so that no generator's layout moved with them.
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

/-- The arena an artifact's runtime allocates: what the program needs, and at
    least its image. -/
def Artifact.memorySize (a : Artifact) : Nat := max a.required_memory a.initial_memory.length

open Cbor in
instance : ToCbor Artifact where
  cbor a :=
    let w : IR.Wire := { prims := IR.primitivesOf a.functions, answers := a.functions.map (·.answers) }
    struct
      [("primitives", array w.prims IR.primitiveCbor),
       ("functions", array a.functions (·.toCbor w)),
       ("arena_size", nat a.memorySize),
       ("data", array (segmentsOf a.initial_memory) fun (off, bs) =>
          struct [("offset", nat off), ("bytes", bytes ⟨bs.toArray⟩)])]

/-- An artifact and the file name it is written under. -/
abbrev ArtifactEntry := String × Artifact

def artifactEntry (name : String) (artifact : Artifact) : ArtifactEntry :=
  (name, artifact)

/-- Parse the sole CLI argument as an output directory. -/
def requireOutputDir (args : List String) : IO String :=
  match args with
  | [dir] => pure dir
  | _ => throw <| IO.userError "expected exactly one argument: output directory"

/-- Write each artifact to `{dir}/{name}.cbor`, as it is. Generators write
    through `emitArtifacts` (`Surface.Link`), which links first. -/
def writeArtifacts (dir : String) (entries : Array ArtifactEntry) : IO Unit := do
  IO.FS.createDirAll dir
  let mut seen : List String := []
  for (name, artifact) in entries do
    if seen.contains name then
      throw <| IO.userError s!"duplicate artifact name: {name}"
    seen := name :: seen
    match Cbor.encode artifact with
    | .ok bytes => IO.FS.writeBinFile s!"{dir}/{name}.cbor" bytes
    | .error e => throw <| IO.userError s!"{name}: {e}"

def u32 (n : Nat) : UInt32 := UInt32.ofNat n

namespace Cbor.Check
open IR

/-- The artifact `base_types`' `an_artifact_encodes_in_the_profile` encodes,
    and the bytes it expects: the Lean writer and serde agree on each shape the
    profile spells, byte for byte. -/
def sample : Artifact where
  functions := [{
    index := 0
    name := some "main"
    blocks := [{
      ref := ⟨0⟩
      params := [(⟨0⟩, .i64)]
      insts := [
        .iconst ⟨1⟩ .i64 (-2 ^ 63),
        .iconst ⟨2⟩ .i64 (2 ^ 63 - 1),
        .fconst ⟨3⟩ .f64 0xFFFFFFFFFFFFFFFF,
        .load ⟨4⟩ { kind := .uload8, ty := .i32 } ⟨0⟩,
        -- A primitive, whose result nothing reads, and a function.
        .call none (.ffi .sinf) [⟨0⟩],
        .call none (.local 1) [⟨0⟩],
        -- An indirect one, at a signature and a convention.
        .call (some ⟨5⟩) .native [⟨0⟩, ⟨0⟩, ⟨0⟩, ⟨0⟩, ⟨0⟩],
        .ret none] }] }]
  required_memory := 2 ^ 40
  initial_memory := [0, 0, 0, 0xff, 0, 1]

def expected : String :=
  "a46a7072696d69746976657381a3636c696264626173656673796d626f6c67636c5f7369" ++
  "6e66697369676e6174757265a36a636f6e76656e74696f6e64486f737466706172616d73" ++
  "818263663332806772657475726e73818263663332806966756e6374696f6e7381a4646e" ++
  "616d65646d61696e697369676e6174757265a36a636f6e76656e74696f6e64486f737466" ++
  "706172616d73848263693634808263693634808263693634808263693634806772657475" ++
  "726e73818263693634806b737461636b5f736c6f74738066626c6f636b7381a369726566" ++
  "6572656e63650066706172616d738182006369363465696e73747388a168556e61727949" ++
  "6d6d8481016669636f6e7374636936343b7fffffffffffffffa168556e617279496d6d84" ++
  "81026669636f6e7374636936341b7fffffffffffffffa16b556e61727949656565363484" ++
  "810368663634636f6e7374f61bffffffffffffffffa1644c6f616486810466756c6f6164" ++
  "3863693332800000a16443616c6c8581066463616c6cf6a1695072696d69746976650081" ++
  "00a16443616c6c85806463616c6cf6a16846756e6374696f6e018100a16c43616c6c496e" ++
  "6469726563748681056d63616c6c5f696e646972656374f6a36a636f6e76656e74696f6e" ++
  "644172636866706172616d73848263693634808263693634808263693634808263693634" ++
  "806772657475726e7381826369363480008400000000a1684d756c746941727984806672" ++
  "657475726ef6806a6172656e615f73697a651b0000010000000000646461746181a2666f" ++
  "66667365740365627974657343ff0001"

def hexOf (b : ByteArray) : String :=
  b.foldl (init := "") fun s x =>
    let d := Nat.toDigits 16 x.toNat
    s ++ String.ofList (if d.length < 2 then '0' :: d else d)

#guard (encode sample).toOption.map hexOf == some expected

end Cbor.Check

end AlgorithmLib
