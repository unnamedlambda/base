module
public import Demo.WindowDemo
/-! The executable that writes `Demo.WindowDemo`'s artifacts. -/

public def main (args : List String) : IO Unit := Demo.WindowDemo.main args
