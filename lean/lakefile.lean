import Lake
open Lake DSL

require algorithmLib from "lib"

package algorithms where
  srcDir := "algorithms"
  -- Generator executables are build-time tools: their own runtime is
  -- irrelevant, and -O0 keeps a large emitted body from costing minutes in gcc.
  moreLeancArgs := #["-O0"]

-- One library per directory: the algorithms by area, their proofs beside
-- them, and the scans that fail the build when a claim leaves its declared
-- surface. Each generator defines its own `main`, so no module imports two.

@[default_target]
lean_lib Bench where
  globs := #[.submodules `Bench]

@[default_target]
lean_lib Demo where
  globs := #[.submodules `Demo]

@[default_target]
lean_lib Host where
  globs := #[.submodules `Host]

@[default_target]
lean_lib Warp where
  globs := #[.submodules `Warp]

@[default_target]
lean_lib Qwen2 where
  globs := #[.submodules `Qwen2]

@[default_target]
lean_lib GptOss where
  globs := #[.submodules `GptOss]

@[default_target]
lean_lib Tokenizer where
  globs := #[.submodules `Tokenizer]

@[default_target]
lean_lib Vit where
  globs := #[.submodules `Vit]

@[default_target]
lean_lib Lz4 where
  globs := #[.submodules `Lz4]

@[default_target]
lean_lib Scan where
  globs := #[.submodules `Scan]

-- Generators. Every `lean_exe` in this package is one: `main` takes a directory
-- and writes `<name>.cbor` into it. `lake query algorithmLib/artifacts` builds
-- and runs them (the target is in `lib/lakefile.lean`).
lean_exe gencompressalgorithm where
  root := `Demo.Compress
lean_exe gencsvalgorithm where
  root := `Demo.Csv
lean_exe gengradwarpalgorithm where
  root := `Warp.Grad
lean_exe gengemvwarpalgorithm where
  root := `Warp.Gemv
lean_exe genbackwardwidealgorithm where
  root := `Warp.BackwardWide
lean_exe genfftalgorithm where
  root := `Demo.Fft
lean_exe genfallingsandalgorithm where
  root := `Demo.FallingSand
lean_exe genclialgorithm where
  root := `Demo.Cli
lean_exe genbytecountalgorithm where
  root := `Demo.ByteCount
lean_exe genselfdescribingalgorithm where
  root := `Demo.SelfDescribing
lean_exe genbytescrubalgorithm where
  root := `Demo.ByteScrub
lean_exe gencpubenchalgorithm where
  root := `Bench.Cpu
lean_exe genleanevalalgorithm where
  root := `Demo.LeanEval
lean_exe gendrawalgorithm where
  root := `Demo.Draw
lean_exe genblackholealgorithm where
  root := `Demo.BlackHole
lean_exe genhprogpilots where
  root := `Host.Pilots
lean_exe genhprogcorpus where
  root := `Host.Corpus
lean_exe genlz4compalgorithm where
  root := `Lz4.Comp
lean_exe genvitship where
  root := `Vit.Ship
lean_exe genqwen2algorithm where
  root := `Qwen2.Algorithm
lean_exe genraytracealgorithm where
  root := `Demo.Raytrace
lean_exe genmatmulalgorithm where
  root := `Demo.Matmul
lean_exe genmlpwarpalgorithm where
  root := `Warp.Mlp
lean_exe gensiluwarpalgorithm where
  root := `Warp.Silu
lean_exe genpythonbenchmarks where
  root := `Bench.Python
lean_exe genraymarchdemoalgorithm where
  root := `Demo.RaymarchDemo
lean_exe genqwen2ondiskalgorithm where
  root := `Qwen2.OnDisk
lean_exe genwarpsumsqalgorithm where
  root := `Warp.SumSq
lean_exe gensatalgorithm where
  root := `Demo.Sat
lean_exe genscenealgorithm where
  root := `Demo.Scene
lean_exe gensha256algorithm where
  root := `Demo.Sha256
lean_exe genmlpcifaralgorithm where
  root := `Warp.MlpCifar
lean_exe genwindowdemoalgorithm where
  root := `Demo.WindowDemo

lean_exe gentokenizertest where
  root := `Tokenizer.Test

lean_exe gengptossdecode where
  root := `GptOss.Decode

lean_exe gengptossalgorithm where
  root := `GptOss.Algorithm
