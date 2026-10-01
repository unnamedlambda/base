module
public import Demo.Cli
/-! The executable that writes `Demo.Cli`'s artifacts. -/

public def main (args : List String) : IO Unit := Demo.Cli.main args
