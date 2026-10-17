module
public import AlgorithmLib.Check.Exec
public import Lean.ToExpr
@[expose] public section

/-!
# `Check.Reflect` — the checker's values as terms

What the checker computes natively and a proof then names: the loop
invariants of a certificate, turned back into the terms the kernel checks.
-/

open Lean

deriving instance ToExpr for AlgorithmLib.IR.ClifTy
deriving instance ToExpr for AlgorithmLib.IR.ICmpCond
deriving instance ToExpr for AlgorithmLib.HProg.Sem.Region
deriving instance ToExpr for AlgorithmLib.HProg.Sem.V
deriving instance ToExpr for AlgorithmLib.HProg.Contracts.Part
deriving instance ToExpr for AlgorithmLib.HProg.Contracts.Held
deriving instance ToExpr for AlgorithmLib.HProg.Contracts.Fact
deriving instance ToExpr for AlgorithmLib.Check.Num
deriving instance ToExpr for AlgorithmLib.Check.AV
deriving instance ToExpr for AlgorithmLib.Check.Carry
deriving instance ToExpr for AlgorithmLib.Check.Slots
deriving instance ToExpr for AlgorithmLib.Check.AS
deriving instance ToExpr for AlgorithmLib.Check.Cert
