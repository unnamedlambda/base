import Lake
open Lake DSL

require common from "../common"

package beta

lean_lib Beta

lean_exe genbeta where
  root := `Beta.Gen

/-- The names of this package's generators. Read by `build_support::generate`. -/
script generators do
  for exe in (← getRootPackage).leanExes do
    IO.println s!"{exe.name} {exe.config.root}"
  return 0

/-- Source directories of this package and every package it requires. -/
script srcdirs do
  for pkg in (← getWorkspace).packages do
    IO.println pkg.dir
  return 0
