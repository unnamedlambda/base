module
public import Warp.Silu
/-! The executable that writes `Warp.Silu`'s artifacts. -/

public def main (args : List String) : IO Unit := Warp.Silu.main args
