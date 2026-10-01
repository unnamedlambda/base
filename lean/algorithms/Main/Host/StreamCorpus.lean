module
public import Host.StreamCorpus

/-! The executable that writes `Host.StreamCorpus`'s artifacts. -/

public def main (args : List String) : IO Unit := Host.StreamCorpus.main args
