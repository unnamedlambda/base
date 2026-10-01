module
public import Host.CpuCorpus
/-! The executable that writes `Host.CpuCorpus`'s artifacts. -/

public def main (args : List String) : IO Unit := Host.CpuCorpus.main args
