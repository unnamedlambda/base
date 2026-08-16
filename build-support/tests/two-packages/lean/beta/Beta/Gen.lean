import Common
/-- A generator small enough that the test is about the build rather than about
Lean: one artifact, empty apart from a marker in `memory_size` that tells the
two packages' output apart. -/

def size : Nat := 222 + Common.bump

def artifact : String :=
  "{\"setup\":{\"clif\":{\"functions\":[]},\"memory_size\":" ++ toString size ++
  ",\"io_offsets\":{\"data_ptr\":0,\"data_len\":0,\"out_ptr\":0,\"out_len\":0}," ++
  "\"initial_memory\":[]},\"main\":{\"fn_idx\":0,\"output\":[]},\"extras\":{}}"

def main (args : List String) : IO Unit := do
  let dir := args.head!
  IO.FS.createDirAll dir
  IO.FS.writeFile (System.FilePath.mk dir / "beta.json") artifact
