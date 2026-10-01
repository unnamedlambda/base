module
public import Demo.RaymarchDemo
/-! The executable that writes `Demo.RaymarchDemo`'s artifacts. -/

public def main (args : List String) : IO Unit := Demo.RaymarchDemo.main args
