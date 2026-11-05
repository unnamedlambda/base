module
public import Host.NetCorpus
/-! The executable that writes `Host.NetCorpus`'s artifacts. -/

public def main (args : List String) : IO Unit := Host.NetCorpus.main args
