module
public import Demo.ByteScrub
/-! The executable that writes `Demo.ByteScrub`'s artifacts. -/

public def main (args : List String) : IO Unit := Demo.ByteScrub.main args
