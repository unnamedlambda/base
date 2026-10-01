module
public import Demo.Fft
/-! The executable that writes `Demo.Fft`'s artifacts. -/

public def main (args : List String) : IO Unit := Demo.Fft.main args
