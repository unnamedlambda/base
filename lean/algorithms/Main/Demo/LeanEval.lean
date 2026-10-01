module
public import Demo.LeanEval
/-! The executable that writes `Demo.LeanEval`'s artifacts. -/

public def main (args : List String) : IO Unit := Demo.LeanEval.main args
