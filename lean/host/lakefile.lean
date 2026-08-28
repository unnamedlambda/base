import Lake
open Lake DSL System

/-! Lean as a host for the runtime.

This package is a *consumer* of the runtime, never a dependency of it: the
arrow points Lean -> base, the same as it does for the Rust and Python hosts.
Nothing in `base` links against Lean, so adding this takes nothing away from an
embedder who wants a binary with an artifact in it and no Lean anywhere.

`libbase.so` must exist before this links, and lake builds it: the `libbase`
target below runs cargo. That is the mirror of `build-support`, which runs lake
from a cargo build script -- each ecosystem's tool drives the other, so neither
kind of consumer has to know the other exists. `lake exe upcasehost` works from
a clean checkout.

It is a shared object rather than a static library because it resolves cudarc,
wgpu, winit and lmdb itself; linking the static one would put every transitive
system library on this link line.
-/

/-- The cargo profile to build and link `base` from; `BASE_PROFILE=release`. -/
def baseProfile : String :=
  run_io do return (← IO.getEnv "BASE_PROFILE").getD "debug"

/-- The repository, which is where cargo is invoked from. -/
def repoRoot : FilePath := (__dir__ : FilePath) / ".." / ".."

/-- Whether this package builds `libbase.so` itself.

Setting `BASE_LIB_DIR` says the caller owns the library and lake should only
link what is there -- for a build that already has one, or one made some other
way. Unset, lake runs cargo. -/
def buildsBase : Bool :=
  run_io do return (← IO.getEnv "BASE_LIB_DIR").isNone

/-- Where `libbase.so` is.

Resolved when lake loads this file. Lake caches the elaborated configuration by
*content*, so changing `BASE_LIB_DIR` or `BASE_PROFILE` afterwards has no effect
until something makes it re-elaborate -- `lake clean` does, touching this file
does not. -/
def baseLibDir : String := run_io do
  if let some dir ← IO.getEnv "BASE_LIB_DIR" then
    return dir
  return (repoRoot / "target" / baseProfile).toString

package base where
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


/-- `cargo build -p base`, so the runtime exists before anything links it.

Cycle-free: `lean-artifacts` -- the crate whose build script runs lake -- is a
*dev*-dependency of `base`, and a plain `cargo build -p base` does not build
dev-dependencies. `cargo test` does, so it is still cargo that must not be run
concurrently with lake here, not this.

No trace of its own: cargo is the authority on whether the library is stale, and
asking it costs about a tenth of a second when it is not. -/
target libbase _pkg : FilePath := Job.async do
  let so : FilePath := baseLibDir / "libbase.so"
  if buildsBase then
    let profileArgs := if baseProfile == "release" then #["--release"] else #[]
    proc (quiet := true) {
      cmd := "cargo"
      args := #["build", "-p", "base"] ++ profileArgs
      cwd := some repoRoot
    }
  unless ← so.pathExists do
    error s!"{so} does not exist. \
      {if buildsBase then "cargo build -p base did not produce it" else
        "BASE_LIB_DIR is set, so this package did not build it"}"
  return so

/-- Lean's calling convention over the runtime's C ABI. -/
target shimObj pkg : FilePath := do
  let oFile := pkg.buildDir / "c" / "shim.o"
  let src ← inputTextFile <| pkg.dir / "c" / "shim.c"
  buildO oFile src #["-I", (← getLeanIncludeDir).toString] #["-fPIC"] "cc"

/-- The shim, and -- through `libbase` -- the runtime it calls.

An external library is a dependency of every executable this package links, so
putting the cargo build in this job is what orders it before `-lbase` is
resolved. The static library itself is the shim alone; `libbase.so` is linked
by `moreLinkArgs`, not archived into it. -/
extern_lib baseshim pkg := do
  let base ← libbase.fetch
  base.bindM fun _ => do
    let so := pkg.sharedLibDir / nameToSharedLib "baseshim"
    -- The link arguments are *trace* arguments: `weakArgs` are excluded from
    -- the trace by design, and putting them there left a shim built against
    -- one profile's runtime silently linked into the other's build.
    let dyn ← buildSharedLib "baseshim" so #[← shimObj.fetch] #[]
      #[] #["-L" ++ baseLibDir, "-lbase", "-Wl,-rpath," ++ baseLibDir] "cc"
    dyn.mapM fun d => return d.path

@[default_target]
lean_lib BaseHost

/-- The demo artifact: a value, and buildable without any of the host above. -/
@[default_target]
lean_lib Upcase

/-- Builds the upcase artifact and runs it, in one process. -/
lean_exe upcasehost where
  root := `UpcaseHost
