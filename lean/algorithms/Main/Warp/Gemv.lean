import Warp.Gemv
/-! The executable that writes `Warp.Gemv`'s artifacts. -/

def main (args : List String) : IO Unit := Warp.Gemv.main args
