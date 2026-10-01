module
public import Demo.Compress
/-! The executable that writes `Demo.Compress`'s artifacts. -/

public def main (args : List String) : IO Unit := Demo.Compress.main args
