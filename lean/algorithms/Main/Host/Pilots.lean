module
public import Host.Pilots
/-! The executable that writes `Host.Pilots`'s artifacts. -/

public def main (args : List String) : IO Unit := Host.Pilots.main args
