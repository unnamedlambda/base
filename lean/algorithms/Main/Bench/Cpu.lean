module
public import Bench.Cpu
/-! The executable that writes `Bench.Cpu`'s artifacts. -/

public def main (args : List String) : IO Unit := Bench.Cpu.main args
