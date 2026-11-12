import Lake
open Lake DSL

require algorithmLib from "lib"

package algorithms where
  srcDir := "algorithms"
  -- Generator executables are build-time tools: their own runtime is
  -- irrelevant, and -O0 keeps a large emitted body from costing minutes in gcc.
  moreLeancArgs := #["-O0"]
  leanOptions := #[⟨`experimental.module, true⟩]

-- One library per directory. All are default targets, so a bare `lake build`
-- builds and checks everything, the scans included.

-- Programs that ship as demos, each with the claims it makes.
@[default_target]
lean_lib Demo where
  globs := #[.submodules `Demo]

-- Benchmark bodies: CPU, CUDA and the x86 bodies checked against GNU as.
@[default_target]
lean_lib Bench where
  globs := #[.submodules `Bench]

-- Host-language pilots, the conformance corpus, and the surface idioms.
@[default_target]
lean_lib Host where
  globs := #[.submodules `Host]

-- Models on proven warp kernels: the demos and CIFAR.
@[default_target]
lean_lib Warp where
  globs := #[.submodules `Warp]

-- ANN: base's engine for the approximate-nearest-neighbour benchmark.
@[default_target]
lean_lib Ann where
  globs := #[.submodules `Ann]

-- gpt-oss-20b: kernels, attention, the decode artifact.
@[default_target]
lean_lib GptOss where
  globs := #[.submodules `GptOss]

-- The tokenizer and pre-tokenizer gpt-oss uses.
@[default_target]
lean_lib Tokenizer where
  globs := #[.submodules `Tokenizer]

-- ViT: model, schedule, launches and their guards.
@[default_target]
lean_lib Vit where
  roots := #[`Vit.Algorithm]
  globs := #[.one `Vit.Algorithm, .one `Vit.Body, .one `Vit.Dag, .one `Vit.Guards, .one `Vit.Host, .one `Vit.Model, .one `Vit.Ship, .one `Vit.Units, .one `Vit.Weave]
  precompileModules := true

-- The ViT's slot layout theorem, decided by compiled evaluation: apart from
-- the precompiled library, where loading it would run the decision again.
@[default_target]
lean_lib VitSlot where
  roots := #[`Vit.Slot]

-- The LZ4 compressor artifact and its proof.
@[default_target]
lean_lib Lz4 where
  globs := #[.submodules `Lz4]

-- Every shipped entry point never misuses a call, by the checker (`Check`).
-- Plain files; a deep proof term needs the larger thread stack, and each check
-- is held to 3.5 GB by Lean itself: one that would take more fails on its own
-- rather than pressing the machine toward the cap that kills the session.
@[default_target]
lean_lib Ship where
  roots := #[`Ship.NoMisuse, `Ship.CudaSafe, `Ship.CudaBenchSafe,
    `Ship.WgpuDraw, `Ship.WgpuFallingSand, `Ship.WgpuWindowDemo, `Ship.WgpuWindowDemoMain, `Ship.WgpuRaymarch, `Ship.WgpuRaymarchMain, `Ship.CpuApps, `Ship.CpuAppsHist, `Ship.CpuAppsSha256, `Ship.CudaApps, `Ship.CpuBenchStart, `Ship.CpuBench, `Ship.CpuBenchPoly, `Ship.CpuBenchStream, `Ship.CudaPipeline, `Ship.Lz4Comp, `Ship.Lz4Run, `Ship.MlpCifar1, `Ship.MlpCifar2, `Ship.QwenReplay, `Ship.QwenRun, `Ship.QwenLoad, `Ship.VitState, `Ship.VitDag, `Ship.VitSafe, `Ship.VitIo, `Ship.VitReach, `Ship.VitRuns, `Ship.VitRunsBwd, `Ship.VitCapture, `Ship.VitCaptureStep, `Ship.VitCaptureCalls, `Ship.VitCaptureReach, `Ship.VitReplayWords, `Ship.VitCalls, `Ship.VitWords, `Ship.VitStep,
    `Ship.GemvWarpQwen, `Ship.GemvWarpMid, `Ship.GemvWarpMid2, `Ship.GemvWarpWide,
    `Ship.FallingSandMain,
    `Ship.Pilots, `Ship.PilotsHist, `Ship.TenMoe, `Ship.LocalCalls, `Ship.Threads, `Ship.ParSumSpec, `Ship.Libs, `Ship.HtSpec, `Ship.LibCheck, `Ship.MathSpec, `Ship.MathCheck, `Ship.HtRuns, `Ship.HtLinked, `Ship.HtCheck]
  moreLeanArgs := #["--tstack=262144", "-M", "3500"]

-- The scans' machinery, which the generators import to scan themselves: its
-- own library, so a precompiled library may import it without importing the
-- scans of the apps built on that library.
@[default_target]
lean_lib ScanKit where
  roots := #[`Scan.Ship]
  globs := #[.one `Scan.Core, .one `Scan.Layout, .one `Scan.Ship, .one `Scan.MlSurface]

-- The trust, ship and layout scans of each app. Plain files: they read proof terms.
@[default_target]
lean_lib Scan where
  globs := #[.one `Scan.GptOss, .one `Scan.GptOssDecode, .one `Scan.GptOssSurface, .one `Scan.Lz4, .one `Scan.Mlp, .one `Scan.Tokenizer, .one `Scan.Utility, .one `Scan.Vit, .one `Scan.Weave]

-- The executables' roots, one line each; built by the executables that use them.
lean_lib Main where
  globs := #[.submodules `Main]

-- Generators. Every `lean_exe` in this package is one, rooted in `Main/`: its
-- `main` calls the generator module's own, which takes a directory
-- and writes `<name>.cbor` into it. `lake query algorithmLib/artifacts` builds
-- and runs them (the target is in `lib/lakefile.lean`).
lean_exe gencompressalgorithm where
  root := `Main.Demo.Compress
lean_exe gengradwarpalgorithm where
  root := `Main.Warp.Grad
lean_exe gengemvwarpalgorithm where
  root := `Main.Warp.Gemv
lean_exe genfftalgorithm where
  root := `Main.Demo.Fft
lean_exe genfallingsandalgorithm where
  root := `Main.Demo.FallingSand
lean_exe genclialgorithm where
  root := `Main.Demo.Cli
lean_exe genbytecountalgorithm where
  root := `Main.Demo.ByteCount
lean_exe genparsumalgorithm where
  root := `Main.Demo.ParSum
lean_exe genselfdescribingalgorithm where
  root := `Main.Demo.SelfDescribing
lean_exe genbytescrubalgorithm where
  root := `Main.Demo.ByteScrub
lean_exe gensharedsumalgorithm where
  root := `Main.Demo.SharedSum
lean_exe gencpubenchalgorithm where
  root := `Main.Bench.Cpu
lean_exe genimagebench where
  root := `Main.Bench.Image
lean_exe gensystembench where
  root := `Main.Bench.System
lean_exe genmemopsbench where
  root := `Main.Bench.Memops
lean_exe gensortbench where
  root := `Main.Ann.Sort
lean_exe genhashtablebench where
  root := `Main.Bench.HashTable
lean_exe genanalyticsbench where
  root := `Main.Bench.Analytics
lean_exe genjsonbench where
  root := `Main.Bench.Json
lean_exe genleanevalalgorithm where
  root := `Main.Demo.LeanEval
lean_exe gendrawalgorithm where
  root := `Main.Demo.Draw
lean_exe genblackholealgorithm where
  root := `Main.Demo.BlackHole
lean_exe genhprogpilots where
  root := `Main.Host.Pilots
lean_exe genhprogcorpus where
  root := `Main.Host.Corpus
lean_exe genlz4compalgorithm where
  root := `Main.Lz4.Comp
lean_exe genvitship where
  root := `Main.Vit.Ship
lean_exe genraytracealgorithm where
  root := `Main.Demo.Raytrace
lean_exe genmatmulalgorithm where
  root := `Main.Demo.Matmul
lean_exe genmlpwarpalgorithm where
  root := `Main.Warp.Mlp
lean_exe gensiluwarpalgorithm where
  root := `Main.Warp.Silu
lean_exe genpythonbenchmarks where
  root := `Main.Bench.Python
lean_exe genraymarchdemoalgorithm where
  root := `Main.Demo.RaymarchDemo
lean_exe genwarpsumsqalgorithm where
  root := `Main.Warp.SumSq
lean_exe gensatalgorithm where
  root := `Main.Demo.Sat
lean_exe genscenealgorithm where
  root := `Main.Demo.Scene
lean_exe gensha256algorithm where
  root := `Main.Demo.Sha256
lean_exe genmlpcifaralgorithm where
  root := `Main.Warp.MlpCifar
lean_exe genwindowdemoalgorithm where
  root := `Main.Demo.WindowDemo
lean_exe gentokenizertest where
  root := `Main.Tokenizer.Test
lean_exe genannalgorithm where
  root := `Main.Ann.Host
lean_exe gengptossdecode where
  root := `Main.GptOss.Decode
lean_exe gengptossalgorithm where
  root := `Main.GptOss.Algorithm
lean_exe genhprogdrivercorpus where
  root := `Main.Host.DriverCorpus
lean_exe genhprogserialcorpus where
  root := `Main.Host.SerialCorpus
lean_exe genhprogfscorpus where
  root := `Main.Host.FsCorpus
lean_exe genhprognetcorpus where
  root := `Main.Host.NetCorpus
lean_exe genhprogusbcorpus where
  root := `Main.Host.UsbCorpus
lean_exe genhprogcpucorpus where
  root := `Main.Host.CpuCorpus
lean_exe genhprogwgpucorpus where
  root := `Main.Host.WgpuCorpus
lean_exe genhprognativecorpus where
  root := `Main.Host.NativeCorpus
lean_exe genhproglocalcorpus where
  root := `Main.Host.LocalCorpus
lean_exe genhprogthreadcorpus where
  root := `Main.Host.ThreadCorpus
