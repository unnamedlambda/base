import Lake
open Lake DSL System

package algorithmLib where
  srcDir := "."

@[default_target]
lean_lib AlgorithmLib where
  globs := #[.andSubmodules `AlgorithmLib]

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
