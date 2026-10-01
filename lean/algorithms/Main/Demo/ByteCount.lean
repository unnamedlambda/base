module
public import Demo.ByteCount
/-! The executable that writes `Demo.ByteCount`'s artifacts. -/

public def main (args : List String) : IO Unit := Demo.ByteCount.main args
