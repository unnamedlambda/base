module
public import Bench.HashTable
/-! The executable that writes `Bench.HashTable`'s artifact. -/

public def main (args : List String) : IO Unit := Bench.HashTable.main args
