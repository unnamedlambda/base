module
public import AlgorithmLib.Host.LifecycleCore
meta import AlgorithmLib.Host.LifecycleCore
@[expose] public section

/-! `keeps_<entry>` for the CUDA entry points, apart from `Host.Lifecycle` so they build beside it. -/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sem AlgorithmLib.HProg.DevSpec

namespace AlgorithmLib.HProg.Contracts

derive_keeps_only "cuda"

end AlgorithmLib.HProg.Contracts
