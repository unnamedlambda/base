import Lean
import AlgorithmLib.ClifData

open Lean

namespace AlgorithmLib

/-- The board's control, as data: the functions a host compiles, the memory
    they run in, and the bytes that memory starts with. This is the whole of
    the wire format, written as the CBOR `base_types::Artifact` reads.

    A host calls a function by its `exportName`. -/
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

open Cbor in
instance : ToCbor Artifact where
  cbor a := struct
    [("functions", array a.functions cbor),
     ("memory_size", nat a.memory_size),
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

/-- Write each artifact to `{dir}/{name}.cbor`. -/
def emitArtifacts (dir : String) (entries : Array ArtifactEntry) : IO Unit := do
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
private def sample : Artifact where
  functions := [{
    index := 0
    exportName := some "main"
    sigs := [{ ref := ⟨0⟩, params := [.i64], result := none }]
    fns := [{ ref := ⟨0⟩, callee := .import "cl_x", sig := ⟨0⟩ },
            { ref := ⟨1⟩, callee := .local 1, sig := ⟨0⟩ }]
    blocks := [{
      ref := ⟨0⟩
      params := [(⟨0⟩, .i64)]
      insts := [
        .iconst ⟨1⟩ .i64 (-2 ^ 63),
        .iconst ⟨2⟩ .i64 (2 ^ 63 - 1),
        .fconst ⟨3⟩ .f64 0xFFFFFFFFFFFFFFFF,
        .load ⟨4⟩ { kind := .uload8, ty := .i32 } ⟨0⟩,
        .call none ⟨0⟩ [⟨0⟩],
        .ret none] }] }]
  memory_size := 2 ^ 40
  initial_memory := [0, 0, 0, 0xff, 0, 1]

private def expected : String :=
  "a36966756e6374696f6e7381a46b6578706f72745f6e616d65646d61696e647369677381" ++
  "a3697265666572656e63650066706172616d73816349363466726573756c74f663666e73" ++
  "82a3697265666572656e6365006663616c6c6565a166496d706f727464636c5f78637369" ++
  "6700a3697265666572656e6365016663616c6c6565a1654c6f63616c0163736967006662" ++
  "6c6f636b7381a3697265666572656e63650066706172616d738182006349363465696e73" ++
  "747386a16649636f6e73748301634936343b7fffffffffffffffa16649636f6e73748302" ++
  "634936341b7fffffffffffffffa16646636f6e73748303634636341bffffffffffffffff" ++
  "a1644c6f61648404a3646b696e6466556c6f616438627479634933326e6e6f747261705f" ++
  "616c69676e6564f40000a16443616c6c83f6008100a163526574f66b6d656d6f72795f73" ++
  "697a651b0000010000000000646461746181a2666f66667365740365627974657343ff00" ++
  "01"

private def hexOf (b : ByteArray) : String :=
  b.foldl (init := "") fun s x =>
    let d := Nat.toDigits 16 x.toNat
    s ++ String.ofList (if d.length < 2 then '0' :: d else d)

#guard (encode sample).toOption.map hexOf == some expected

end Cbor.Check

end AlgorithmLib
