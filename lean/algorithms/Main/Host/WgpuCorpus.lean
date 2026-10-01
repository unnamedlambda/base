module
public import Host.WgpuCorpus
/-! The executable that writes `Host.WgpuCorpus`'s artifacts. -/

public def main (args : List String) : IO Unit := Host.WgpuCorpus.main args
