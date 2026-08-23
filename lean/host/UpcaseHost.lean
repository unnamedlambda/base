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

The input file is written here rather than by the artifact because it is the
demo's setup and not the demo.
-/

/-- The demo's input. Its length is also how much of the runtime's memory the
transformed bytes occupy, which is what `readMemory` below is given: the
artifact stores the file's size in a register, not in memory, so a host that
wants it either reads it back from a field the program stores it to or knows it
the way this one does. -/
def input : String := "a lean host, running its own artifact\n"

def main : IO Unit := do
  IO.FS.writeFile "input.txt" input

  Base.withRuntime Upcase.setup fun rt => do
    IO.println s!"runtime memory: {← rt.memorySize} bytes"
    let _ ← rt.execute Upcase.algorithm
    -- Read back through the runtime, at the offset the layout gave the field:
    -- the same bytes the artifact then wrote out, with no Arrow in between.
    let transformed ← rt.readMemory Upcase.f.fileData.offset input.utf8ByteSize
    IO.println s!"in memory: {(String.fromUTF8! transformed).trim}"

  IO.println s!"artifact wrote output.txt: {(← IO.FS.readFile "output.txt").trim}"
