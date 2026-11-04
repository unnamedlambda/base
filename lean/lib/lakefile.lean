import Lake
open Lake DSL System

package algorithmLib where
  srcDir := "."
  -- The module system: a proof edit that leaves a module's public interface
  -- alone does not rebuild the modules that import it.
  leanOptions := #[⟨`experimental.module, true⟩]

/-! # Libraries

The modules are compiled to native code (`precompileModules`), so that what a
claim decides by evaluation runs at native speed. A module of another package
loads a precompiled library whole, and so rebuilds whenever any module of that
library changes and waits for all of it to build. The library is therefore
split where its imports split: the base everything stands on; the checker on
it; the two developments on the base, LZ4 and ML, which do not import each
other; and what ties ML's models to checked host programs, on ML and the
checker. An ML edit rebuilds what imports ML, not what imports only the base or
LZ4, and a generator, which imports only the base and a development, builds
while the checker's soundness proofs do. Each library's roots differ from its
name, so Lake loads it as a plain shared library, each imported module
initialized by itself, rather than as a plugin initialized through one module
named after it. -/

/-- Everything, by its root module `AlgorithmLib`, which nothing imports. -/
@[default_target]
lean_lib AlgorithmLib where
  roots := #[`AlgorithmLib]

/-- What every development stands on: the host and its semantics, the
    checker, the kernel layer and the generator toolkit. -/
@[default_target]
lean_lib AlgorithmBase where
  roots := #[`AlgorithmLib.Gen]
  globs := #[.submodules `AlgorithmLib.Core, .submodules `AlgorithmLib.Vocab,
    .submodules `AlgorithmLib.Host, .submodules `AlgorithmLib.Surface,
    .submodules `AlgorithmLib.Proof,
    .submodules `AlgorithmLib.Kernel, .one `AlgorithmLib.Gen]
  precompileModules := true

/-- The checker, proven sound over the base: its own library, so what imports
    only the base (every generator) builds while the soundness proofs do. -/
@[default_target]
lean_lib AlgorithmCheck where
  roots := #[`AlgorithmLib.Check.Entry]
  globs := #[.submodules `AlgorithmLib.Check]
  precompileModules := true

/-- What is proved of the compilers themselves: that a kernel body's PTX does
    what the body does. Only the root `AlgorithmLib` imports it, so the tree's
    build checks it; it decides nothing by evaluation, so it is not
    precompiled: an edit rebuilds it alone rather than relinking the base
    library every package loads. -/
@[default_target]
lean_lib AlgorithmSound where
  globs := #[.submodules `AlgorithmLib.Sound]

/-- The LZ4 development. -/
@[default_target]
lean_lib AlgorithmLZ4 where
  roots := #[`AlgorithmLib.LZ4]
  globs := #[.andSubmodules `AlgorithmLib.LZ4]
  precompileModules := true

/-- The ML development. -/
@[default_target]
lean_lib AlgorithmML where
  roots := #[`AlgorithmLib.ML]
  globs := #[.andSubmodules `AlgorithmLib.ML]
  precompileModules := true

/-- What a checked host program running an ML model leaves: its launches and
    the tape's words in device memory. Above both the ML development and the
    checker, so neither waits for the other. -/
@[default_target]
lean_lib AlgorithmMLHost where
  roots := #[`AlgorithmLib.MLHost.TapeRun]
  globs := #[.submodules `AlgorithmLib.MLHost]
  precompileModules := true

/-! # Artifacts

`lake query algorithmLib/artifacts`, run in a package that requires this one,
builds and runs that package's generators -- every `lean_exe` it declares --
and gathers what they write into `.lake/build/artifacts`, one flat directory:
`<name>.cbor` per artifact, and `<name>/` for an artifact's side data.

Lake decides which generators rerun: each run is traced against the hash of its
executable, so an edit that leaves a generator's compiled code alone reruns
nothing. What is here is only what Lake cannot know -- that names are unique
across generators, and that a failure leaves the last good output in place.

It prints one line per fact, for whatever drives the build:

* `dir <path>` -- the artifact directory;
* `artifact <name>` -- one per artifact;
* `generator <exe> <path>` -- one per generator, with its executable;
* `input <path>` -- every file the artifacts depend on: each generator's module
  and everything it imports, and each package's configuration. -/

/-- The artifact names directly in `dir`, sorted. -/
def artifactsIn (dir : FilePath) : IO (Array String) := do
  let mut out := #[]
  for e in (← dir.readDir) do
    if e.path.extension == some "cbor" then out := out.push (e.path.fileStem.getD "")
  return out.qsort (· < ·)

/-- The subdirectories directly in `dir`, sorted. -/
def sideDirsIn (dir : FilePath) : IO (Array String) := do
  let mut out := #[]
  for e in (← dir.readDir) do
    if (← e.path.isDir) then out := out.push e.fileName
  return out.qsort (· < ·)

/-- Leave an unchanged file untouched: rustc and cargo see a rewrite with the
    same bytes as a change, and recompile whatever embeds it. -/
def writeIfChanged (p : FilePath) (bytes : ByteArray) : IO Unit := do
  if (← p.pathExists) then
    if (← IO.FS.readBinFile p) == bytes then return
  IO.FS.writeBinFile p bytes

partial def copyTree (src dst : FilePath) : IO Unit := do
  IO.FS.createDirAll dst
  for e in (← src.readDir) do
    if (← e.path.isDir) then copyTree e.path (dst / e.fileName)
    else IO.FS.writeBinFile (dst / e.fileName) (← IO.FS.readBinFile e.path)

target artifacts : String := do
  let root ← getRootPackage
  let out := root.buildDir / "artifacts"
  let gens := root.buildDir / "generated"
  -- Lake takes no lock, and two builds of this target would run the same
  -- generator into the same directory. Held until the last job below ends;
  -- `unlock` there also keeps the handle alive, since Lean frees it at last use
  -- and closing the file would drop the lock.
  IO.FS.createDirAll root.buildDir
  let lock ← IO.FS.Handle.mk (root.buildDir / "artifacts.lock") .write
  lock.lock
  let runs ← root.leanExes.mapM fun exe => do
    let dir := gens / exe.name.toString
    let list := gens / s!"{exe.name}.list"
    -- The trace is on the list, so an output directory deleted by hand would
    -- still read as current. Without its list the generator reruns.
    if !(← dir.pathExists) && (← list.pathExists) then IO.FS.removeFile list
    let bin ← exe.exe.fetch
    let job ← buildFileAfterDep list bin fun bin => do
      -- A failed run keeps the trace of the last good one, so the directory
      -- that trace describes must survive it: generate aside, then swap.
      let tmp := gens / s!"{exe.name}.tmp"
      if (← tmp.pathExists) then IO.FS.removeDirAll tmp
      IO.FS.createDirAll tmp
      let r ← IO.Process.output { cmd := bin.toString, args := #[tmp.toString] }
      if r.exitCode ≠ 0 then error s!"{exe.name} failed:\n{r.stdout}{r.stderr}"
      let names ← artifactsIn tmp
      if names.isEmpty then error s!"{exe.name} wrote no artifacts"
      for e in (← tmp.readDir) do
        unless e.path.extension == some "cbor" || (← e.path.isDir) do
          error s!"{exe.name} wrote {e.fileName}, which is not an artifact; \
                   side data belongs in a directory named after its artifact"
      if (← dir.pathExists) then IO.FS.removeDirAll dir
      IO.FS.rename tmp dir
      IO.FS.writeFile list ("\n".intercalate names.toList)
    return (job.zipWith (fun _ b => (exe.name.toString, dir, b)) bin)
  let closures ← root.leanExes.mapM fun exe => do
    let some m ← findModule? exe.config.root
      | error s!"{exe.name}: no module {exe.config.root}"
    return (← m.transImports.fetch).map (m :: ·.toList)
  let ws ← getWorkspace
  (Job.collectArray runs).bindM fun runs => (Job.collectArray closures).mapM fun closures => do
    -- Every name is known before anything in `out` is touched, so a clash
    -- leaves the previous artifacts as they were.
    let mut owner : Std.HashMap String (String × FilePath) := {}
    let mut sides : Std.HashMap String (String × FilePath) := {}
    for (gen, dir, _) in runs do
      for n in (← artifactsIn dir) do
        if let some (other, _) := owner[n]? then
          error s!"artifact {n} is emitted by both {other} and {gen}"
        owner := owner.insert n (gen, dir)
      for n in (← sideDirsIn dir) do
        if let some (other, _) := sides[n]? then
          error s!"side data {n}/ is written by both {other} and {gen}"
        sides := sides.insert n (gen, dir)
    -- What a generator no longer in the lakefile left: its directory, its
    -- list and the list's trace. Generator names have no dots.
    for e in (← gens.readDir) do
      let exeName := (e.fileName.splitOn ".").head!
      unless runs.any (·.1 == exeName) do
        if (← e.path.isDir) then IO.FS.removeDirAll e.path else IO.FS.removeFile e.path
    IO.FS.createDirAll out
    for (n, _, dir) in owner.toList do
      writeIfChanged (out / s!"{n}.cbor") (← IO.FS.readBinFile (dir / s!"{n}.cbor"))
    for n in (← artifactsIn out) do
      unless owner.contains n do IO.FS.removeFile (out / s!"{n}.cbor")
    for n in (← sideDirsIn out) do
      IO.FS.removeDirAll (out / n)
    for (n, _, dir) in sides.toList do
      copyTree (dir / n) (out / n)
    let mut lines := #[s!"dir {out}"]
    lines := lines ++ ((owner.toList.map (·.1)).toArray.qsort (· < ·)).map (s!"artifact {·}")
    lines := lines ++ runs.map fun (gen, _, bin) => s!"generator {gen} {bin}"
    let mut files : Array String := #[]
    for pkg in ws.packages do
      -- Only files that exist: cargo reruns a build script on every build
      -- while a path it was told to watch is missing.
      for f in [pkg.configFile, pkg.dir / "lean-toolchain", pkg.manifestFile] do
        if (← f.pathExists) then files := files.push f.toString
    for ms in closures do
      for m in ms do files := files.push m.leanFile.toString
    lines := lines ++ (files.qsort (· < ·)).toList.eraseDups.toArray.map (s!"input {·}")
    lock.unlock
    return "\n".intercalate lines.toList
