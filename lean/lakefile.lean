import Lake
open Lake DSL

require algorithmLib from "lib"

package algorithms where
  srcDir := "algorithms"
  -- Generator executables are build-time tools: their own runtime is
  -- irrelevant, and -O0 keeps a large emitted body from costing minutes in gcc.
  moreLeancArgs := #["-O0"]

-- Benchmark algorithms
lean_lib PythonBenchmarks
lean_lib HistogramBench1Algorithm
lean_lib ClampSumBenchAlgorithm
lean_lib HProgPilots
lean_lib HProgCorpus
lean_lib CudaDecodeAttentionAlgorithm
lean_lib CudaGemvPersistAlgorithm
lean_lib CudaRmsNormPersistAlgorithm
lean_lib CudaSaxpyPersistAlgorithm
lean_lib CudaSoftmaxPersistAlgorithm
lean_lib CudaVecAddPersistAlgorithm
@[default_target]
lean_lib WarpSumSqAlgorithm
@[default_target]
lean_lib SiluWarpAlgorithm
@[default_target]
lean_lib MlpWarpAlgorithm
@[default_target]
lean_lib GradWarpAlgorithm
@[default_target]
lean_lib Qwen2Proven
@[default_target]
lean_lib GemvWarpAlgorithm
@[default_target]
lean_lib BackwardWideAlgorithm
@[default_target]
lean_lib MlpCifarAlgorithm
@[default_target]
lean_lib WeaveCifar
@[default_target]
lean_lib WeaveVit
lean_lib WeaveScan
lean_lib VitModel
lean_lib VitAlgorithm
lean_lib VitShip
lean_lib VitUnits
lean_lib VitLaunches
lean_lib VitGuards
lean_lib VitDag
lean_lib VitDagStep
lean_lib VitRegs
lean_lib VitSlot
lean_lib VitScan
lean_lib VitTerm
-- gpt-oss-20b: the ledger ships before the artifact does, on purpose.
lean_lib GptOssDecode
lean_lib GptOssAttention
lean_lib GptOssKernels
lean_lib GptOssAlgorithm
lean_lib TokenizerCommon
lean_lib PretokCommon
lean_lib TokenizerTest
lean_lib TokenizerScan
lean_lib GptOssSurface
lean_lib GptOssScan
lean_lib GptOssDecodeScan
@[default_target]
lean_lib BigModelAlgorithm
@[default_target]
lean_lib NonVacuity

-- Application algorithms
lean_lib CliAlgorithm
lean_lib CompressAlgorithm
lean_lib CsvAlgorithm
lean_lib DrawAlgorithm
lean_lib FftAlgorithm
lean_lib LeanEvalAlgorithm
lean_lib MatmulAlgorithm
lean_lib RaytraceAlgorithm
lean_lib SatAlgorithm
lean_lib SceneAlgorithm
lean_lib BlackHoleAlgorithm
lean_lib ByteCountAlgorithm
lean_lib SelfDescribingAlgorithm
lean_lib ByteScrubAlgorithm
lean_lib Sha256Algorithm
@[default_target]
lean_lib Qwen2Common
@[default_target]
lean_lib Qwen2Algorithm
-- Build-enforced: recomputes the trusted base from the proof terms and fails
-- if any public claim reaches an axiom or opaque outside the declared surface.
lean_lib Qwen2Spec

lean_lib Qwen2Top

lean_lib ScanCore

lean_lib LayoutScan

lean_lib ShipScan

-- Build-enforced: the frontend idioms a library is written with — type classes,
-- recursion over a user's syntax, higher-order and continuation combinators,
-- label passing, computed loop widths, obligation towers, monad transformers.
-- Each definition is its own check; a change to `Prog` that breaks one fails
-- here rather than in whoever's library meets it next.
lean_lib ProgIdioms

-- Build-enforced: what one vector's trip of the byte scanner computes, for
-- every input, over the semantics the artifact is checked against.
lean_lib ByteCountProof

-- Build-enforced: what one vector's trip of the byte scrubber computes, for
-- every input, over the semantics the artifact is checked against.
lean_lib ByteScrubProof

lean_lib MlSurface

lean_lib TrustScan

-- …and the same scan over each generator that ships an artifact with no
-- algorithmic theorem.  Separate modules for the usual reason — each generator
-- defines its own `main` — and, for Sat and Sha256, because they share
-- `namespace Algorithm` and cannot be imported together at all.
lean_lib GenSurface
lean_lib SatScan
lean_lib Sha256Scan
lean_lib LeanEvalScan
lean_lib CudaSaxpyPersistScan
lean_lib CudaVecAddPersistScan

-- …and the same scan over each training pipeline.  Separate modules because
-- each generator defines its own `main`.
lean_lib BackwardScan
lean_lib MlpScan
lean_lib Qwen2NonVacuity
@[default_target]
lean_lib Qwen2OnDiskAlgorithm
lean_lib WindowDemoAlgorithm
lean_lib RaymarchDemoAlgorithm
lean_lib FallingSandAlgorithm
lean_lib Lz4Kernel
lean_lib Lz4CompAlgorithm
-- The compressor's ledger, its non-vacuity witnesses, and the scan that fails
-- the build when a claim leaves the declared surface.
lean_lib Lz4Assumptions
lean_lib Lz4NonVacuity
lean_lib Lz4Launches
lean_lib Lz4Interleave
lean_lib Lz4Host
lean_lib Lz4Sites
lean_lib Lz4Scan
lean_lib Lz4Cursor
lean_lib Lz4Splice
lean_lib Lz4OpLe
lean_lib Lz4Ckpt
lean_lib Lz4Stores
lean_lib Lz4Geo
lean_lib Lz4ExtShape
lean_lib Lz4ExtGuard
lean_lib Lz4ExtLoop
lean_lib Lz4Extend
lean_lib Lz4Shape64
lean_lib Lz4Sites64
lean_lib Lz4Cursor64
lean_lib Lz4Splice64
lean_lib Lz4Ckpt64
lean_lib Lz4OpLe64
lean_lib Lz4Stores64
lean_lib Lz4Confine64
lean_lib Lz4Whole

-- Generators. Every `lean_exe` in this package is one: `main` takes a directory
-- and writes `<name>.cbor` into it. `lake query algorithmLib/artifacts` builds
-- and runs them (the target is in `lib/lakefile.lean`).
lean_exe gencompressalgorithm where
  root := `CompressAlgorithm
lean_exe gencsvalgorithm where
  root := `CsvAlgorithm
lean_exe gengradwarpalgorithm where
  root := `GradWarpAlgorithm
lean_exe gengemvwarpalgorithm where
  root := `GemvWarpAlgorithm
lean_exe genbackwardwidealgorithm where
  root := `BackwardWideAlgorithm
lean_exe genfftalgorithm where
  root := `FftAlgorithm
lean_exe genfallingsandalgorithm where
  root := `FallingSandAlgorithm
lean_exe genclialgorithm where
  root := `CliAlgorithm
lean_exe genbytecountalgorithm where
  root := `ByteCountAlgorithm
lean_exe genselfdescribingalgorithm where
  root := `SelfDescribingAlgorithm
lean_exe genbytescrubalgorithm where
  root := `ByteScrubAlgorithm
lean_exe genleanevalalgorithm where
  root := `LeanEvalAlgorithm
lean_exe gendrawalgorithm where
  root := `DrawAlgorithm
lean_exe genblackholealgorithm where
  root := `BlackHoleAlgorithm
lean_exe genhprogpilots where
  root := `HProgPilots
lean_exe genhprogcorpus where
  root := `HProgCorpus
lean_exe genlz4compalgorithm where
  root := `Lz4CompAlgorithm
lean_exe genvitship where
  root := `VitShip
lean_exe genqwen2algorithm where
  root := `Qwen2Algorithm
lean_exe genraytracealgorithm where
  root := `RaytraceAlgorithm
lean_exe genmatmulalgorithm where
  root := `MatmulAlgorithm
lean_exe genmlpwarpalgorithm where
  root := `MlpWarpAlgorithm
lean_exe gensiluwarpalgorithm where
  root := `SiluWarpAlgorithm
lean_exe genpythonbenchmarks where
  root := `PythonBenchmarks
lean_exe genraymarchdemoalgorithm where
  root := `RaymarchDemoAlgorithm
lean_exe genqwen2ondiskalgorithm where
  root := `Qwen2OnDiskAlgorithm
lean_exe genwarpsumsqalgorithm where
  root := `WarpSumSqAlgorithm
lean_exe gensatalgorithm where
  root := `SatAlgorithm
lean_exe genscenealgorithm where
  root := `SceneAlgorithm
lean_exe gensha256algorithm where
  root := `Sha256Algorithm
lean_exe genmlpcifaralgorithm where
  root := `MlpCifarAlgorithm
lean_exe genwindowdemoalgorithm where
  root := `WindowDemoAlgorithm

lean_exe gentokenizertest where
  root := `TokenizerTest

lean_exe gengptossdecode where
  root := `GptOssDecode

lean_exe gengptossalgorithm where
  root := `GptOssAlgorithm
