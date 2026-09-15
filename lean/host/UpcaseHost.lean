import BaseHost
import Upcase

/-!
# A Lean program whose effects are a value

`Upcase.setup` reads a file, transforms it and writes it. This runs it. Between
the two there is no build step, no serialization and no second process: the
artifact is built in this process and handed straight to the runtime.

`main` is what is left of a program once its work is an artifact. The `IO` in it
is opening the runtime, running it, and printing; every effect that is the
*point* of the program happens inside `execute`, and is described by a value
this program could equally have written to disk for a Rust or Python host.

Nothing here knows how long the file was. The artifact read it, so the artifact
is what knows, and it stored the count to `Upcase.f.size`; `readField` takes
that same field handle and reads it back. That is the shape to copy — a host
that hardcodes a length has written the layout down twice.

The input file is written here rather than by the artifact because it is the
demo's setup and not the demo.
-/

def main : IO Unit := do
  IO.FS.writeFile "input.txt" "a lean host, running its own artifact\n"

  let setup ← AlgorithmLib.Prog.orDie Upcase.shipped
  Base.withRuntime setup fun rt => do
    IO.println s!"runtime memory: {← rt.memorySize} bytes"
    let _ ← rt.execute Upcase.algorithm

    -- Both of these name a field rather than an offset, and the second's
    -- length came from the first.
    let size ← rt.readField Upcase.f.size
    let transformed ← rt.readMemory Upcase.f.fileData.offset size.toNat
    IO.println s!"artifact transformed {size} bytes"
    IO.println s!"in memory: {(String.fromUTF8! transformed).trim}"

  IO.println s!"artifact wrote output.txt: {(← IO.FS.readFile "output.txt").trim}"
