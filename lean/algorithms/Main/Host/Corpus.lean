module
public import Host.Corpus
/-! The executable that writes `Host.Corpus`'s artifacts. -/

public def main (args : List String) : IO Unit := Host.Corpus.main args
