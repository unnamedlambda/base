module
public import Demo.SelfDescribing
/-! The executable that writes `Demo.SelfDescribing`'s artifacts. -/

public def main (args : List String) : IO Unit := Demo.SelfDescribing.main args
