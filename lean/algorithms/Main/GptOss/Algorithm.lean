module
public import GptOss.Algorithm
/-! The executable that writes `GptOss.Algorithm`'s artifacts. -/

public def main (args : List String) : IO Unit := GptOss.Algorithm.main args
