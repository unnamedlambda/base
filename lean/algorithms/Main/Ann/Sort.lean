module
public import Ann.Sort
/-! The executable that writes `Ann.Sort`'s artifact. -/

public def main (args : List String) : IO Unit := Ann.Sort.main args
