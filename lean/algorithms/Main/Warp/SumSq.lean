module
public import Warp.SumSq
/-! The executable that writes `Warp.SumSq`'s artifacts. -/

public def main (args : List String) : IO Unit := Warp.SumSq.main args
