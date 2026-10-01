module
public import Host.DriverCorpus
/-! The executable that writes `Host.DriverCorpus`'s artifacts. -/

public def main (args : List String) : IO Unit := Host.DriverCorpus.main args
