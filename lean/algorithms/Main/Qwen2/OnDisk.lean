module
public import Qwen2.OnDisk
/-! The executable that writes `Qwen2.OnDisk`'s artifacts. -/

public def main (args : List String) : IO Unit := Qwen2.OnDisk.main args
