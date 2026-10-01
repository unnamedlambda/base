module
public import Host.SerialCorpus
/-! The executable that writes `Host.SerialCorpus`'s artifacts. -/

public def main (args : List String) : IO Unit := Host.SerialCorpus.main args
