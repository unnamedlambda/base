module
public import Host.GpuCorpus

/-! The executable that writes `Host.GpuCorpus`'s artifacts. -/

public def main (args : List String) : IO Unit := Host.GpuCorpus.main args
