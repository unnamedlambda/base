module
public import Demo.Scene
/-! The executable that writes `Demo.Scene`'s artifacts. -/

public def main (args : List String) : IO Unit := Demo.Scene.main args
