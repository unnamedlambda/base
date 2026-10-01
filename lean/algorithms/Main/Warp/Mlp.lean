module
public import Warp.Mlp
/-! The executable that writes `Warp.Mlp`'s artifacts. -/

public def main (args : List String) : IO Unit := Warp.Mlp.main args
