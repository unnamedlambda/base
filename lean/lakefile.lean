import Lake
open Lake DSL

require algorithmLib from "lib"

package algorithms where
  srcDir := "algorithms"
  -- Generator executables are build-time tools: their own runtime is
  -- irrelevant, and -O0 keeps a large emitted body from costing minutes in gcc.
  moreLeancArgs := #["-O0"]
  -- The module system: a proof edit that leaves a module's public interface
  -- alone does not rebuild the modules that import it.
  leanOptions := #[⟨`experimental.module, true⟩]

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

-- Each generator's executable: it runs its algorithm module's `main`, so
-- algorithm modules can be imported together.
lean_lib Main where
  globs := #[.submodules `Main]

-- Generators. Every `lean_exe` in this package is one: `main` takes a directory
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
