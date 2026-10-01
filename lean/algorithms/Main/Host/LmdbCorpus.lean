module
public import Host.LmdbCorpus

/-! The executable that writes `Host.LmdbCorpus`'s artifacts. -/

public def main (args : List String) : IO Unit := Host.LmdbCorpus.main args
