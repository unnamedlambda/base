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

-- Models on proven warp kernels: the demos, CIFAR, the wide backward pass.
@[default_target]
lean_lib Warp where
  globs := #[.submodules `Warp]

-- Qwen2 inference: spec, proven kernels, plan, host, and the top claim.
@[default_target]
lean_lib Qwen2 where
  globs := #[.submodules `Qwen2]

-- gpt-oss-20b: kernels, attention, the decode artifact.
@[default_target]
lean_lib GptOss where
  globs := #[.submodules `GptOss]

-- The tokenizer and pre-tokenizer shared by Qwen2 and gpt-oss.
@[default_target]
lean_lib Tokenizer where
  globs := #[.submodules `Tokenizer]

-- ViT: model, schedule, launches and their guards.
@[default_target]
lean_lib Vit where
  globs := #[.submodules `Vit]

-- The LZ4 compressor artifact and its proof.
@[default_target]
lean_lib Lz4 where
  globs := #[.submodules `Lz4]

-- Every shipped entry point never misuses a call, by the condition generator.
-- Plain files; a deep proof term needs the larger thread stack, and each check
-- is held to 3.5 GB by Lean itself: one that would take more fails on its own
-- rather than pressing the machine toward the cap that kills the session.
@[default_target]
lean_lib Ship where
  roots := #[`Ship.NoMisuse, `Ship.CudaSafe, `Ship.CudaBackwardSafe, `Ship.CudaBenchSafe,
    `Ship.WgpuDraw, `Ship.WgpuFallingSand, `Ship.WgpuWindowDemo, `Ship.WgpuWindowDemoMain, `Ship.WgpuRaymarch, `Ship.WgpuRaymarchMain, `Ship.CpuApps, `Ship.CpuAppsHist, `Ship.CpuAppsSha256, `Ship.CudaApps, `Ship.CpuBenchStart, `Ship.CpuBench, `Ship.CpuBenchPoly, `Ship.CpuBenchStream, `Ship.CudaPipeline, `Ship.Lz4Comp, `Ship.MlpCifar1, `Ship.MlpCifar2, `Ship.VitSafe,
    `Ship.GemvWarpQwen, `Ship.GemvWarpMid, `Ship.GemvWarpMid2, `Ship.GemvWarpWide,
    `Ship.FallingSandStart, `Ship.FallingSandDisplay, `Ship.FallingSandHeadless, `Ship.FallingSandMain,
    `Ship.Pilots, `Ship.PilotsHist, `Ship.TenMoe, `Ship.LocalCalls]
  moreLeanArgs := #["--tstack=262144", "-M", "3500"]

-- The trust, ship and layout scans. Plain files: they read proof terms.
@[default_target]
lean_lib Scan where
  globs := #[.submodules `Scan]

-- The executables' roots, one line each; built by the executables that use them.
lean_lib Main where
  globs := #[.submodules `Main]

-- Generators. Every `lean_exe` in this package is one, rooted in `Main/`: its
-- `main` calls the generator module's own, which takes a directory
-- and writes `<name>.cbor` into it. `lake query algorithmLib/artifacts` builds
-- and runs them (the target is in `lib/lakefile.lean`).
lean_exe gencompressalgorithm where
  root := `Main.Demo.Compress
lean_exe gencsvalgorithm where
  root := `Main.Demo.Csv
lean_exe gengradwarpalgorithm where
  root := `Main.Warp.Grad
lean_exe gengemvwarpalgorithm where
  root := `Main.Warp.Gemv
lean_exe genbackwardwidealgorithm where
  root := `Main.Warp.BackwardWide
lean_exe genfftalgorithm where
  root := `Main.Demo.Fft
lean_exe genfallingsandalgorithm where
  root := `Main.Demo.FallingSand
lean_exe genclialgorithm where
  root := `Main.Demo.Cli
lean_exe genbytecountalgorithm where
  root := `Main.Demo.ByteCount
lean_exe genselfdescribingalgorithm where
  root := `Main.Demo.SelfDescribing
lean_exe genbytescrubalgorithm where
  root := `Main.Demo.ByteScrub
lean_exe gencpubenchalgorithm where
  root := `Main.Bench.Cpu
lean_exe gensystembench where
  root := `Main.Bench.System
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
lean_exe genqwen2algorithm where
  root := `Main.Qwen2.Algorithm
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
lean_exe genqwen2ondiskalgorithm where
  root := `Main.Qwen2.OnDisk
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
lean_exe gengptossdecode where
  root := `Main.GptOss.Decode
lean_exe gengptossalgorithm where
  root := `Main.GptOss.Algorithm
lean_exe genhprogcudacorpus where
  root := `Main.Host.CudaCorpus
lean_exe genhprogdrivercorpus where
  root := `Main.Host.DriverCorpus
lean_exe genhprogserialcorpus where
  root := `Main.Host.SerialCorpus
lean_exe genhprogusbcorpus where
  root := `Main.Host.UsbCorpus
lean_exe genhprogcpucorpus where
  root := `Main.Host.CpuCorpus
lean_exe genhprogwgpucorpus where
  root := `Main.Host.WgpuCorpus
lean_exe genhprogpucorpus where
  root := `Main.Host.GpuCorpus
lean_exe genhprogstreamcorpus where
  root := `Main.Host.StreamCorpus
lean_exe genhproglmdbcorpus where
  root := `Main.Host.LmdbCorpus
lean_exe genhprogwindowcorpus where
  root := `Main.Host.WindowCorpus
lean_exe genhprognativecorpus where
  root := `Main.Host.NativeCorpus
lean_exe genhproglocalcorpus where
  root := `Main.Host.LocalCorpus
lean_exe genhprogthreadcorpus where
  root := `Main.Host.ThreadCorpus
