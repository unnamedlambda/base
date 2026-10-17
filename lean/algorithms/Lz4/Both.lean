module
public import Lean
@[expose] public section

/-!
  # One proof text, two geometries

  The facts about the shipped kernel that name its array — where each store
  site is, which registers a region writes, what a `decide` over the
  instructions says — are stated once, at the 32 KiB geometry, and are true at
  64 KiB by the same text with the geometry's numbers changed.

  `at_both cmd` elaborates `cmd` as written and records it. `replay64 M`, in
  another module, elaborates every command module `M` recorded again at the
  64 KiB geometry: `WP.mk 15` becomes `WP.mk 16`, the kernel `K` becomes `K16`,
  each stride and offset its 64 KiB value (`numTable`), and a recorded
  declaration's name, where it is declared and wherever a recorded command
  names it, takes the suffix `64`. Every `decide` is run again against the
  64 KiB array; nothing is carried over. `at64 cmd` records `cmd` to be
  replayed as written, for the few declarations whose 64 KiB text is not the
  32 KiB one.
-/

open Lean Elab Command

namespace Lz4Both

/-- A recorded command: instantiated by substitution, or replayed as written. -/
structure Entry where
  stx : Syntax
  verbatim : Bool
  deriving Inhabited

meta initialize geoExt : SimplePersistentEnvExtension Entry (Array Entry) ←
  registerSimplePersistentEnvExtension { addEntryFn := Array.push, addImportedFn := fun _ => #[] }

/-- The names a command declares. -/
meta partial def declNames (stx : Syntax) : Array Name :=
  if stx.getKind == ``Lean.Parser.Command.declId then #[stx[0].getId]
  else stx.getArgs.foldl (fun acc a => acc ++ declNames a) #[]

/-- Names whose 64 KiB twin is not the name with `64` after it. -/
meta def identTable : List (Name × Name) :=
  [(`K, `K16), (`shipped32_size, `kSize16), (`shipped32_store_sites, `shipped64_store_sites),
   (`shipped32_load_sites, `shipped64_load_sites), (`load_at_site, `load_at_site64),
   (`shipped32_no_stg32p, `shipped64_no_stg32p), (`shipped32_no_stg32p_lt, `shipped64_no_stg32p_lt),
   (`tok_at164, `tok_at164_g16), (`winShapeB_32, `winShapeB_64), (`nb64, `nb16)]

/-- Each number of the 32 KiB geometry with its 64 KiB value: the strides, the
    offsets within a block, and the number of blocks. -/
meta def numTable : List (Nat × Nat) :=
  [(35080, 69896), (35072, 69888), (32799, 65567), (32786, 65554), (32768, 65536), (32764, 65532),
   (32763, 65531), (32762, 65530), (32756, 65524), (32755, 65523), (6400, 3200)]

/-- What takes the block size's logarithm as its first argument. -/
meta def logHeads : List String := ["mk", "RegConfined", "KernelConfined", "CursorAtSites", "outRegion"]

meta def lastStr : Name → String
  | .str _ s => s
  | _ => ""

meta def ofComps (cs : List Name) : Name := cs.foldl (fun acc x => acc ++ x) Name.anonymous

/-- An identifier at the 64 KiB geometry: by the table, or its first component
    that names a recorded declaration with `64` after it. -/
meta def renameIdent (names : NameSet) (n : Name) : Option Name :=
  match identTable.lookup n with
  | some m => some m
  | none =>
      let rec go (pre : List Name) : List Name → Option Name
        | [] => none
        | c :: rest =>
            match identTable.lookup c with
            | some m => some (ofComps (pre ++ [m] ++ rest))
            | none =>
                if names.contains c then some (ofComps (pre ++ [Name.mkSimple (lastStr c ++ "64")] ++ rest))
                else go (pre ++ [c]) rest
      go [] n.components

/-- Whether a term is `WP.mk 15`, in parentheses. -/
meta def isGeo15 (stx : Syntax) : Bool :=
  stx.getKind == ``Lean.Parser.Term.paren && stx[1].getKind == ``Lean.Parser.Term.app &&
    stx[1][0].isIdent && stx[1][0].getId == `WP.mk && stx[1][1].getArgs.size == 1 &&
    stx[1][1][0].isNatLit? == some 15

/-- A command at the 64 KiB geometry. -/
meta partial def subst (names : NameSet) : Syntax → Syntax
  | stx@(.ident info _ n _) =>
      match renameIdent names n with
      | some m => .ident info (toString m).toSubstring m []
      | none => stx
  | stx@(.node info kind args) =>
      if kind == numLitKind then
        match stx.isNatLit? with
        | some v => match numTable.lookup v with
          | some v' => Syntax.mkNumLit (toString v') info
          | none => stx
        | none => stx
      else if kind == ``Lean.Parser.Term.proj && isGeo15 args[0]! && args[2]!.isIdent &&
          args[2]!.getId == `kernel then
        .ident info "K16".toSubstring `K16 []
      else if kind == ``Lean.Parser.Term.app && args.size == 2 && args[0]!.isIdent &&
          logHeads.contains (lastStr args[0]!.getId) && args[1]!.getArgs.size > 0 &&
          args[1]!.getArgs[0]!.isNatLit? == some 15 then
        let as := args[1]!.getArgs
        let as' := (as.set! 0 (Syntax.mkNumLit "16" as[0]!.getHeadInfo)).mapIdx fun i a =>
          if i == 0 then a else subst names a
        .node info kind #[subst names args[0]!, args[1]!.setArgs as']
      else .node info kind (args.map (subst names))
  | stx => stx

/-- Every position of a command taken from `ref`. -/
meta partial def reRef (ref : Syntax) : Syntax → Syntax
  | .node _ k args => .node (SourceInfo.fromRef ref) k (args.map (reRef ref))
  | .atom _ v => .atom (SourceInfo.fromRef ref) v
  | .ident _ r n p => .ident (SourceInfo.fromRef ref) r n p
  | .missing => .missing

/-- `at_both cmd`: `cmd` as written, recorded for `replay64`. -/
elab "at_both " c:command : command => do
  elabCommand c
  modifyEnv (geoExt.addEntry · ⟨c, false⟩)

/-- `at64 cmd`: `cmd` is not elaborated here; `replay64` replays it as
    written, in its place among the recorded commands. -/
elab "at64 " c:command : command => do
  modifyEnv (geoExt.addEntry · ⟨c, true⟩)

/-- `replay64 M`: the commands module `M` recorded, at the 64 KiB geometry. -/
elab ref:"replay64 " m:ident : command => do
  let env ← getEnv
  let some idx := env.getModuleIdx? m.getId | throwError "replay64: module {m.getId} is not imported"
  let mut names : NameSet := {}
  for es in (geoExt.toEnvExtension.getState env).importedEntries do
    for e in es do
      for n in declNames e.stx do
        if e.verbatim then
          -- a declaration replayed as written names its 32 KiB twin's with `64` after it
          let s := lastStr n
          if s.endsWith "64" then names := names.insert (Name.mkSimple (s.dropRight 2))
        else names := names.insert n
  for e in geoExt.getModuleEntries env idx do
    let stx := if e.verbatim then e.stx else subst names e.stx
    elabCommand (reRef ref stx)

end Lz4Both
