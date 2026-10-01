module
public import Demo.Sha256
/-! The executable that writes `Demo.Sha256`'s artifacts. -/

public def main (args : List String) : IO Unit := Demo.Sha256.main args
