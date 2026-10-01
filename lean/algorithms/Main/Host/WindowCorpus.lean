module
public import Host.WindowCorpus

/-! The executable that writes `Host.WindowCorpus`'s artifacts. -/

public def main (args : List String) : IO Unit := Host.WindowCorpus.main args
