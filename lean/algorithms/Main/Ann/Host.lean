module
public import Ann.Host
/-! The executable that writes `Ann.Host`'s artifacts. -/

public def main (args : List String) : IO Unit := Ann.Host.main args
