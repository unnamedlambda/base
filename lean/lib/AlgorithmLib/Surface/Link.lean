module
public import AlgorithmLib.Surface.LibCuda
meta import AlgorithmLib.Surface.LibCuda
public import AlgorithmLib.Surface.LibHt
meta import AlgorithmLib.Surface.LibHt
public import AlgorithmLib.Surface.LibMath
meta import AlgorithmLib.Surface.LibMath
public import AlgorithmLib.Surface.LibLmdb
meta import AlgorithmLib.Surface.LibLmdb
public import AlgorithmLib.Surface.LibThread
meta import AlgorithmLib.Surface.LibThread
public import AlgorithmLib.Surface.LibWindow
meta import AlgorithmLib.Surface.LibWindow
public import AlgorithmLib.Core.Artifact
meta import AlgorithmLib.Core.Artifact
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# Linking the libraries into a program

A program names the engine's entry points (`Ffi`); the libraries written in
CLIF implement some of them over C calls. `link` appends the functions a
program's calls need and points each call at its function, and
`emitArtifacts` links every artifact it writes, so none reaches a host
calling an entry point a library implements.
-/

namespace AlgorithmLib.Link

open AlgorithmLib.IR
open AlgorithmLib.Prog
open AlgorithmLib.Lib

/-- The library function implementing `f`, for a program that does or does
    not call cuBLAS; `none` for an entry point the engine still provides. -/
def implOf (blas : Bool) (f : Ffi) : Option Impl :=
  [LibCuda.implOf blas f, LibHt.implOf f, LibMath.implOf f, LibLmdb.implOf f,
    LibThread.implOf f, LibWgpu.implOf f, LibWindow.implOf f].findSome? id

/-- The value a definition binds, if it binds one. -/
def Inst.dst? : Inst → Option Val
  | .iconst d .. | .iadd d .. | .isub d .. | .imul d .. | .udiv d .. | .ineg d ..
  | .ishl d .. | .ushr d .. | .band d .. | .bandNot d .. | .bor d .. | .bxor d ..
  | .ireduce32 d .. | .uextend64 d .. | .sextend64 d .. | .load d .. | .icmp d ..
  | .select d .. | .fconst d .. | .fadd d .. | .fsub d .. | .fmul d .. | .fmax d ..
  | .fmin d .. | .fpromote d .. | .splat d .. | .extractlane d .. | .fneg d ..
  | .fcvtFromSint d .. | .fcvtToUint d .. | .fcmp d .. | .bitcast d .. | .bitselect d ..
  | .ctz d .. | .popcnt d .. | .vhighBits d .. | .ibin d .. | .ishift d .. | .iun d ..
  | .fbin d .. | .fun1 d .. | .fconv d .. | .fma d .. | .iext d .. => some d
  | .call d .. => d
  | .store .. | .istore8 .. | .jump .. | .brif .. | .ret .. | .storeTyped .. => none

/-- One past the largest value a function binds. -/
def nextVal (f : FuncData) : Nat :=
  f.blocks.foldl (fun n b =>
    let n := b.params.foldl (fun n (v, _) => max n (v.id + 1)) n
    b.insts.foldl (fun n i => match Inst.dst? i with
      | some v => max n (v.id + 1)
      | none => n) n) 0

/-- Point every call to an entry point in `table` at its library function. A
    function answers an `i64`, so an `i32` answer is the low half of it and an
    `f32` answer those bits. -/
def relink (table : List (Ffi × Nat)) (f : FuncData) : FuncData := Id.run do
  let mut next := nextVal f
  let mut blocks := []
  for b in f.blocks do
    let mut insts := []
    for i in b.insts do
      match i with
      | .call d (.ffi g) args =>
          match table.find? (·.1 == g), d with
          | some (_, k), some dv =>
              if g.result == some .i32 then
                let t : Val := ⟨next⟩
                next := next + 1
                insts := insts ++ [.call (some t) (.local k) args, .ireduce32 dv t]
              else if g.result == some .f32 then
                let t : Val := ⟨next⟩
                let t2 : Val := ⟨next + 1⟩
                next := next + 2
                insts := insts ++ [.call (some t) (.local k) args, .ireduce32 t2 t, .bitcast dv .f32 t2]
              else insts := insts ++ [.call d (.local k) args]
          | some (_, k), none => insts := insts ++ [.call none (.local k) args]
          | none, _ => insts := insts ++ [i]
      | i => insts := insts ++ [i]
    blocks := blocks ++ [{ b with insts }]
  return { f with blocks }

/-- The engine entry points a function calls. -/
def ffisOf (f : FuncData) : List Ffi :=
  f.blocks.flatMap fun b => b.insts.filterMap fun
    | .call _ (.ffi g) _ => some g
    | _ => none

/-- **Link the libraries into a program.** The functions its calls need are
    appended, in `Ffi.all`'s order, and each call is pointed at its function.
    A program that calls none is returned as it is. -/
def link (fs : List FuncData) : Except String (List FuncData) := do
  let called := fs.flatMap ffisOf
  let used := Ffi.all.filter fun g => (implOf false g).isSome && called.contains g
  if used.isEmpty then return fs
  let blas := used.any LibCuda.isBlas
  let n := fs.length
  let mut table := []
  let mut lib := []
  for g in used do
    match implOf blas g with
    | none => throw s!"{g.cname} has no library implementation"
    | some impl =>
        let k := n + table.length
        lib := lib ++ [← compileImpl k g impl]
        table := table ++ [(g, k)]
  return fs.map (relink table) ++ lib

/-- **Write each artifact, linked.** Every artifact is written here, so none
    reaches a host still calling an entry point a library implements. -/
def emitArtifacts (dir : String) (entries : Array ArtifactEntry) : IO Unit := do
  let linked ← entries.mapM fun (name, a) => do
    match link a.functions with
    | .ok fs => pure (name, { a with functions := fs })
    | .error e => throw (IO.userError s!"{name}: {e}")
  AlgorithmLib.writeArtifacts dir linked

end AlgorithmLib.Link

namespace AlgorithmLib

/-- Write each artifact, with the libraries linked in. -/
def emitArtifacts (dir : String) (entries : Array ArtifactEntry) : IO Unit :=
  Link.emitArtifacts dir entries

end AlgorithmLib
