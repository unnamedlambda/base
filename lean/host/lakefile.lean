import Lake
open Lake DSL System

/-! Lean as a host for the runtime.

This package is a *consumer* of the runtime, never a dependency of it: the
arrow points Lean -> base, the same as it does for the Rust and Python hosts.
Nothing in `base` links against Lean, so adding this takes nothing away from an
embedder who wants a binary with an artifact in it and no Lean anywhere.

`libbase.so` must exist before this links, which `cargo build -p base` makes.
It is a shared object rather than a static library because it resolves cudarc,
wgpu, winit and lmdb itself; linking the static one would put every transitive
system library on this link line.
-/

/-- Where cargo left `libbase.so`.

A release build if there is one, the debug build otherwise, and `BASE_LIB_DIR`
over both. Resolved when lake loads this file, so a first `cargo build --release`
after configuring wants a `lake clean` to be picked up. -/
def baseLibDir : String := run_io do
  if let some dir ← IO.getEnv "BASE_LIB_DIR" then
    return dir
  let cargoOut : FilePath := (__dir__ : FilePath) / ".." / ".." / "target"
  let release := cargoOut / "release"
  if ← (release / "libbase.so").pathExists then
    return release.toString
  return (cargoOut / "debug").toString

package host where
  srcDir := "."
  moreLinkArgs := #[
    "-L" ++ baseLibDir,
    "-lbase",
    -- So a built demo runs without `LD_LIBRARY_PATH`. Absolute, because the
    -- binary and the library have no fixed relationship in the build tree.
    "-Wl,-rpath," ++ baseLibDir
  ]

require algorithmLib from "../lib"
-- For `ShipScan` only. Importing a *generator* from that package would bring
-- its `main` with it, which is why the demo artifact lives here instead.
require algorithms from "../algorithms"


/-- Lean's calling convention over the runtime's C ABI. -/
target shimObj pkg : FilePath := do
  let oFile := pkg.buildDir / "c" / "shim.o"
  let src ← inputTextFile <| pkg.dir / "c" / "shim.c"
  buildO oFile src #["-I", (← getLeanIncludeDir).toString] #["-fPIC"] "cc"

extern_lib baseshim pkg := do
  buildStaticLib (pkg.staticLibDir / nameToStaticLib "baseshim") #[← shimObj.fetch]

@[default_target]
lean_lib BaseHost

/-- The demo artifact: a value, and buildable without any of the host above. -/
@[default_target]
lean_lib Upcase

/-- Builds the upcase artifact and runs it, in one process. -/
lean_exe upcasehost where
  root := `UpcaseHost
