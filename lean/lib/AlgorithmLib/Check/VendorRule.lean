module
public import AlgorithmLib.Check.WgpuRule
meta import AlgorithmLib.Check.WgpuRule
import all Init.Data.Repr
@[expose] public section

/-!
# `Check.VendorRule` — the libraries the checker reads by their own contracts

A call into the CUDA driver, cuBLAS or wgpu is decided by that library's rows
(`cudaRow`, `blasRow`, `wgRow`): what the library requires of the call, read
off the typestate, and what it leaves. The rest of the libraries are read by
what they require and keep (`preExt`, `extKeeps`).
-/

open AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem

namespace AlgorithmLib.Check

open AlgorithmLib.HProg.Contracts (TState Held)

/-- Whether calls into a library are decided by its own contract here: the
    CUDA driver's, cuBLAS's and wgpu's. -/
def _root_.AlgorithmLib.IR.Ext.vendor : Ext → Bool
  | .cuda _ | .cublas _ | .wgpu _ => true
  | _ => false

/-- The kinds of handle a vendor call may end, or make another of: what the
    checker forgets of the values it holds. -/
def _root_.AlgorithmLib.IR.Ext.kills : Ext → List Held
  | .wgpu f => (wgKills f).map .wg
  | _ => []

/-- **A vendor call, decided.** -/
def vendorRow (S : TState) : Ext → List AV → Option (TState × AV)
  | .cuda f, as => cudaRow S f as
  | .cublas f, as => blasRow S f as
  | .wgpu f, as => wgRow S f as
  | _, _ => none

end AlgorithmLib.Check
