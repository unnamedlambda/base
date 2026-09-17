/-- The artifact `{"functions": [], "memory_size": size, "data": []}` as CBOR,
spelled out: `size` is the only part that varies, and it is small. -/
def cbor (size : Nat) : ByteArray :=
  let text (s : String) : List UInt8 := (0x60 + s.toUTF8.size).toUInt8 :: s.toUTF8.toList
  let uint (n : Nat) : List UInt8 :=
    if n < 24 then [n.toUInt8]
    else if n < 0x100 then [0x18, n.toUInt8]
    else [0x19, (n / 0x100).toUInt8, (n % 0x100).toUInt8]
  ⟨([0xa3] ++ text "functions" ++ [0x80] ++ text "memory_size" ++ uint size
     ++ text "data" ++ [0x80]).toArray⟩

/-- One artifact, written wherever it is told. Two cargo workspaces generate
from this same package, which is what puts two `lake` invocations on it at
once. -/
def artifact : ByteArray := cbor 777

def main (args : List String) : IO Unit := do
  let dir := args.head!
  IO.FS.createDirAll dir
  IO.FS.writeBinFile (System.FilePath.mk dir / "shared.cbor") artifact
