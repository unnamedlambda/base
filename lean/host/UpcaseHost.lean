import BaseHost
import Upcase
/-!
# A Lean program whose effects are a value

`Upcase.setup` reads a file, transforms it and writes it. This runs it. Between
the two there is no build step, no serialization and no second process: the
artifact is built in this process and handed straight to the driver.

`main` is what is left of a program once its work is an artifact. The `IO` in it
is opening the driver, running it, and printing; every effect that is the
*point* of the program happens inside `execute`, and is described by a value
this program could equally have written to disk for a Rust or Python host.

Nothing here knows how long the file was. The artifact read it, so the artifact
is what knows, and it says so twice: as the status `execute` answers with, and
in memory at `Upcase.f.size`, which `readField` reads back from the same field
handle the artifact was built from. That is the shape to copy — a host that
hardcodes a length has written the layout down twice.

The input file is written here rather than by the artifact because it is the
demo's setup and not the demo.
-/

def main : IO Unit := do
  IO.FS.writeFile "input.txt" "a lean host, running its own artifact\n"

  let artifact ← AlgorithmLib.Prog.orDie Upcase.shipped
  Base.withDriver artifact fun drv => do
    IO.println s!"driver memory: {← drv.memorySize} bytes"
    let (_, status) ← drv.executeStatus "main"
    IO.println s!"artifact answered {status}"

    -- Both of these name a field rather than an offset, and the second's
    -- length came from the first.
    let size ← drv.readField Upcase.f.size
    let transformed ← drv.readMemory Upcase.f.fileData.offset size.toNat
    IO.println s!"artifact transformed {size} bytes"
    IO.println s!"in memory: {(String.fromUTF8! transformed).trim}"

  IO.println s!"artifact wrote output.txt: {(← IO.FS.readFile "output.txt").trim}"
