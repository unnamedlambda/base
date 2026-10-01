module
public import Demo.Matmul
/-! The executable that writes `Demo.Matmul`'s artifacts. -/

public def main (args : List String) : IO Unit := Demo.Matmul.main args
