module
public import Demo.Raytrace
/-! The executable that writes `Demo.Raytrace`'s artifacts. -/

public def main (args : List String) : IO Unit := Demo.Raytrace.main args
