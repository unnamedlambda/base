import BaseHost
import Upcase
/-!
# A Lean program whose effects are a value

`Upcase.setup` reads a file, transforms it and writes it. This runs it. Between
the two there is no build step and no second process: the artifact is built in
this process, encoded, and handed straight to the driver.

`main` is what is left of a program once its work is an artifact. The `IO` in it
is loading the driver, calling `main`, and printing; every effect that is the
*point* of the program happens inside that call, and is described by a value
this program could equally have written to disk for a Rust or Python host.

Nothing here knows how long the file was. The artifact read it, so the artifact
is what knows, and it says so twice: as the status `main` answers with, and in
the arena at `Upcase.f.size`, read back at the offset of the same field the
artifact was built from. A host that hardcodes a length has written the layout
down twice.

The input file is written here rather than by the artifact because it is the
demo's setup and not the demo.
-/

/-- The little-endian number in `bytes`. -/
def leNat (bytes : ByteArray) : Nat :=
  bytes.data.foldr (fun b acc => acc * 256 + b.toNat) 0

def main : IO Unit := do
  IO.FS.writeFile "input.txt" "a lean host, running its own artifact\n"

  let artifact ← AlgorithmLib.Prog.orDie Upcase.shipped
  let bytes ← IO.ofExcept (AlgorithmLib.Cbor.encode artifact)
  let drv ← Base.Driver.load bytes
  try
    IO.println s!"arena: {← drv.arenaSize} bytes"
    let upcase ← drv.function "main"
    let (_, status) ← upcase .empty .empty
    IO.println s!"artifact answered {status}"

    -- Both reads name a field's offset, and the second's length came from the first.
    let size := leNat (← drv.read Upcase.f.size.offset 8)
    let transformed ← drv.read Upcase.f.fileData.offset size
    IO.println s!"artifact transformed {size} bytes"
    IO.println s!"in the arena: {(String.fromUTF8! transformed).trim}"
  finally
    drv.close

  IO.println s!"artifact wrote output.txt: {(← IO.FS.readFile "output.txt").trim}"
