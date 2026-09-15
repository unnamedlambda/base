import Common
/-- A generator small enough that the test is about the build rather than about
Lean: one artifact, empty apart from a marker in `memory_size` that tells the
two packages' output apart. -/

def size : Nat := 111 + Common.bump

def artifact : String :=
  "{\"setup\":{\"clif\":{\"functions\":[]},\"memory_size\":" ++ toString size ++
  ",\"initial_memory\":[]},\"main\":{\"fn_idx\":0},\"extras\":{}}"

def main (args : List String) : IO Unit := do
  let dir := args.head!
  IO.FS.createDirAll dir
  IO.FS.writeFile (System.FilePath.mk dir / "alpha.json") artifact
