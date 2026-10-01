module
public import Host.LocalCorpus

/-! The executable that writes `Host.LocalCorpus`'s artifacts. -/

public def main (args : List String) : IO Unit := Host.LocalCorpus.main args
