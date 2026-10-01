module
public import Host.NativeCorpus

/-! The executable that writes `Host.NativeCorpus`'s artifacts. -/

public def main (args : List String) : IO Unit := Host.NativeCorpus.main args
