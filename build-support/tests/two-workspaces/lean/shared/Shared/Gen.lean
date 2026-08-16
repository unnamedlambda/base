/-- One artifact, written wherever it is told. Two cargo workspaces generate
from this same package, which is what puts two `lake` invocations on it at
once. -/

def artifact : String :=
  "{\"setup\":{\"clif\":{\"functions\":[]},\"memory_size\":777," ++
  "\"io_offsets\":{\"data_ptr\":0,\"data_len\":0,\"out_ptr\":0,\"out_len\":0}," ++
  "\"initial_memory\":[]},\"main\":{\"fn_idx\":0,\"output\":[]},\"extras\":{}}"

def main (args : List String) : IO Unit := do
  let dir := args.head!
  IO.FS.createDirAll dir
  IO.FS.writeFile (System.FilePath.mk dir / "shared.json") artifact
