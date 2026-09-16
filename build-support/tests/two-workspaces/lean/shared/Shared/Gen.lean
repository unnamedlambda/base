/-- One artifact, written wherever it is told. Two cargo workspaces generate
from this same package, which is what puts two `lake` invocations on it at
once. -/

def artifact : String :=
  "{\"functions\":[],\"memory_size\":777,\"data\":[]}"

def main (args : List String) : IO Unit := do
  let dir := args.head!
  IO.FS.createDirAll dir
  IO.FS.writeFile (System.FilePath.mk dir / "shared.json") artifact
