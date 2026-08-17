import Lake
open Lake DSL

require algorithmLib from "../lib"

package algorithms where
  srcDir := "."
  -- Generator executables are build-time tools: their own runtime is
  -- irrelevant, and -O0 keeps a large emitted body from costing minutes in gcc.
  moreLeancArgs := #["-O0"]

-- Benchmark algorithms
lean_lib RustBenchmarks
lean_lib PythonBenchmarks
lean_lib CsvBenchAlgorithm
lean_lib RegexBenchAlgorithm
lean_lib JsonBenchAlgorithm
lean_lib StringSearchAlgorithm
lean_lib WordCountAlgorithm
lean_lib SaxpyBenchAlgorithm
lean_lib HistogramBench1Algorithm
lean_lib HistogramBench4Algorithm
lean_lib MatmulBenchAlgorithm
lean_lib VecOpsBenchAlgorithm
lean_lib ReductionBenchAlgorithm
lean_lib GpuVecAddBenchAlgorithm
lean_lib GpuMatMulBenchAlgorithm
lean_lib GpuReductionBenchAlgorithm
lean_lib CudaSaxpyBenchAlgorithm
lean_lib GpuIterBenchAlgorithm
lean_lib SortBenchAlgorithm
lean_lib ClampSumBenchAlgorithm
lean_lib HProgPilots
lean_lib HProgCorpus
lean_lib PlainSumBenchAlgorithm
lean_lib BranchyBenchAlgorithm
lean_lib SelectBenchAlgorithm
lean_lib SelectLeaBenchAlgorithm
lean_lib SelectRotBenchAlgorithm
lean_lib SelectMaskBenchAlgorithm
lean_lib StoreBenchAlgorithm
lean_lib PminSumBenchAlgorithm
lean_lib PandasBenchAlgorithm
lean_lib RegPressureBenchAlgorithm
lean_lib IntSumBenchAlgorithm
lean_lib PandasFilterBenchAlgorithm
lean_lib RowAffineReduceBenchAlgorithm
lean_lib RowDotBenchAlgorithm
lean_lib CudaDecodeAttentionAlgorithm
lean_lib CudaDecoderLayerAlgorithm
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
lean_lib WordCountScan
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

-- Generator entry points, built as native executables so lake caches the run.
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
lean_exe genrustbenchmarks where
  root := `RustBenchmarks
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

/-- The generators, as `<exe> <module>` lines. Read by `lean-artifacts`'s build
script so the declarations above stay the only place a generator is named. -/
script generators do
  for exe in (← getRootPackage).leanExes do
    IO.println s!"{exe.name} {exe.config.root}"
  return 0

/-- Source directories of this package and every package it requires. Read by
`build-support` so a caller names one lakefile and gets the rest. -/
script srcdirs do
  for pkg in (← getWorkspace).packages do
    IO.println pkg.dir
  return 0
