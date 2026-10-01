import Warp.MlpCifar
/-! The executable that writes `Warp.MlpCifar`'s artifacts. -/

def main (args : List String) : IO Unit := Warp.MlpCifar.main args
