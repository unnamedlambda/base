module
public import Demo.Csv
/-! The executable that writes `Demo.Csv`'s artifacts. -/

public def main (args : List String) : IO Unit := Demo.Csv.main args
