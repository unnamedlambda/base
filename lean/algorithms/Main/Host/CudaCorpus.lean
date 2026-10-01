module
public import Host.CudaCorpus

/-! The executable that writes `Host.CudaCorpus`'s artifacts. -/

public def main (args : List String) : IO Unit := Host.CudaCorpus.main args
