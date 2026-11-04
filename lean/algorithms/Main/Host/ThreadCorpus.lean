module
public import Host.ThreadCorpus

/-! The executable that writes `Host.ThreadCorpus`'s artifacts. -/

public def main (args : List String) : IO Unit := Host.ThreadCorpus.main args
