module
public import Host.UsbCorpus
/-! The executable that writes `Host.UsbCorpus`'s artifacts. -/

public def main (args : List String) : IO Unit := Host.UsbCorpus.main args
