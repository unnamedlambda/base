module
public import GptOss.Decode
/-! The executable that writes `GptOss.Decode`'s artifacts. -/

public def main (args : List String) : IO Unit := GptOss.Decode.main args
