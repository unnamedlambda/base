module
public import Warp.Grad
/-! The executable that writes `Warp.Grad`'s artifacts. -/

public def main (args : List String) : IO Unit := Warp.Grad.main args
