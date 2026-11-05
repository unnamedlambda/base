module
public import Host.FsCorpus
/-! The executable that writes `Host.FsCorpus`'s artifacts. -/

public def main (args : List String) : IO Unit := Host.FsCorpus.main args
