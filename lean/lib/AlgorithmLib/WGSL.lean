import Lean
open Lean

namespace AlgorithmLib.WGSL

/-!
  Typed WGSL compute shader builder DSL.

  Every expression carries a phantom `WTy` type parameter so the Lean
  elaborator enforces type discipline: array element types, pointer
  targets, scalar vs. vector distinctions, etc.  Shader bodies are
  written as monadic `WB Unit` programs; `buildShader` renders them to
  a complete `.wgsl` module string.

  Key constructs:
  - `WTy`         — WGSL type universe
  - `Expr ty`     — phantom-typed expression (wraps a rendered string)
  - `WB`          — `StateM WGSLState`, accumulates emitted lines
  - `WGSLFn`      — typed function declaration (params, retTy, body)
  - `WGSLDecl`    — top-level declaration (typed const or function)
  - `Binding`     — storage buffer binding with typed element type
  - `buildShader` — assemble a complete WGSL module
-/

-- ---------------------------------------------------------------------------
-- Type universe
-- ---------------------------------------------------------------------------

inductive WTy : Type where
  | bool
  | u32
  | i32
  | f32
  | vec2f | vec3f | vec4f
  | vec2u | vec3u
  | arr    : WTy → WTy        -- array<T>        (dynamic, storage buffers)
  | arrN   : WTy → Nat → WTy  -- array<T, N>     (fixed, workgroup memory)
  | ptrFn  : WTy → WTy        -- ptr<function, T>

def WTy.render : WTy → String
  | .bool     => "bool"
  | .u32      => "u32"
  | .i32      => "i32"
  | .f32      => "f32"
  | .vec2f    => "vec2<f32>"
  | .vec3f    => "vec3<f32>"
  | .vec4f    => "vec4<f32>"
  | .vec2u    => "vec2<u32>"
  | .vec3u    => "vec3<u32>"
  | .arr t    => s!"array<{t.render}>"
  | .arrN t n => s!"array<{t.render}, {n}>"
  | .ptrFn t  => s!"ptr<function, {t.render}>"

-- ---------------------------------------------------------------------------
-- Phantom-typed expression
-- ---------------------------------------------------------------------------

/-- A WGSL expression of type `ty`, represented as its rendered string. -/
structure Expr (ty : WTy) where
  s : String

instance : ToString (Expr ty) := ⟨fun e => e.s⟩

-- ---------------------------------------------------------------------------
-- Builder monad
-- ---------------------------------------------------------------------------

structure WGSLState where
  lines    : Array String := #[]
  indent   : Nat          := 0
  varCount : Nat          := 0

abbrev WB := StateM WGSLState

private def emit (line : String) : WB Unit :=
  modify fun st =>
    let pfx := String.ofList (List.replicate (st.indent * 4) ' ')
    { st with lines := st.lines.push (pfx ++ line) }

/-- Emit raw WGSL source lines at the current indentation level. -/
def raw (src : String) : WB Unit := do
  for line in src.splitOn "\n" do
    emit line

private def indented (inner : WB α) : WB α := do
  modify fun st => { st with indent := st.indent + 1 }
  let r ← inner
  modify fun st => { st with indent := st.indent - 1 }
  return r

private def emitBlock (body : WB Unit) : WB Unit := do
  emit "{"
  indented body
  emit "}"

private def runBodyAt (indent : Nat) (body : WB Unit) : List String :=
  let st₀ : WGSLState := { lines := #[], indent, varCount := 0 }
  (body.run st₀).2.lines.toList

-- ---------------------------------------------------------------------------
-- Literals
-- ---------------------------------------------------------------------------

def litU (n : Nat) : Expr .u32 := ⟨s!"{n}u"⟩
def litI (n : Int) : Expr .i32 := ⟨toString n⟩
/-- Float literal — pass the WGSL string directly to avoid Float repr issues. -/
def litF (v : String) : Expr .f32 := ⟨v⟩
def litTrue  : Expr .bool := ⟨"true"⟩
def litFalse : Expr .bool := ⟨"false"⟩

-- ---------------------------------------------------------------------------
-- Built-in entry-point variables (reference by name in any body that declares them)
-- ---------------------------------------------------------------------------

def gidX : Expr .u32 := ⟨"gid.x"⟩
def gidY : Expr .u32 := ⟨"gid.y"⟩
def gidZ : Expr .u32 := ⟨"gid.z"⟩
def lidX : Expr .u32 := ⟨"lid.x"⟩
def lidY : Expr .u32 := ⟨"lid.y"⟩
def widX : Expr .u32 := ⟨"wid.x"⟩
def widY : Expr .u32 := ⟨"wid.y"⟩

-- ---------------------------------------------------------------------------
-- Constructors
-- ---------------------------------------------------------------------------

def mkVec3f (x y z : Expr .f32) : Expr .vec3f := ⟨s!"vec3<f32>({x}, {y}, {z})"⟩
def mkVec2f (x y   : Expr .f32) : Expr .vec2f := ⟨s!"vec2<f32>({x}, {y})"⟩
/-- Splat a scalar to all components: `vec3<f32>(v)`. -/
def splatV3 (v : Expr .f32) : Expr .vec3f := ⟨s!"vec3<f32>({v})"⟩

-- ---------------------------------------------------------------------------
-- Field / swizzle access
-- ---------------------------------------------------------------------------

def vx  (e : Expr .vec3f) : Expr .f32 := ⟨s!"({e}).x"⟩
def vy  (e : Expr .vec3f) : Expr .f32 := ⟨s!"({e}).y"⟩
def vz  (e : Expr .vec3f) : Expr .f32 := ⟨s!"({e}).z"⟩
def v2x (e : Expr .vec2f) : Expr .f32 := ⟨s!"({e}).x"⟩
def v2y (e : Expr .vec2f) : Expr .f32 := ⟨s!"({e}).y"⟩
def ux  (e : Expr .vec3u) : Expr .u32 := ⟨s!"({e}).x"⟩
def uy  (e : Expr .vec3u) : Expr .u32 := ⟨s!"({e}).y"⟩

-- ---------------------------------------------------------------------------
-- Casts
-- ---------------------------------------------------------------------------

def f32OfU (e : Expr .u32) : Expr .f32 := ⟨s!"f32({e})"⟩
def f32OfI (e : Expr .i32) : Expr .f32 := ⟨s!"f32({e})"⟩
def u32OfF (e : Expr .f32) : Expr .u32 := ⟨s!"u32({e})"⟩
def i32OfU (e : Expr .u32) : Expr .i32 := ⟨s!"i32({e})"⟩
def i32OfF (e : Expr .f32) : Expr .i32 := ⟨s!"i32({e})"⟩
/-- Reinterpret the bits of a u32 as a signed i32 (`bitcast`, not a numeric cast). -/
def i32BitsOfU (e : Expr .u32) : Expr .i32 := ⟨s!"bitcast<i32>({e})"⟩

-- ---------------------------------------------------------------------------
-- Arithmetic — standard typeclass instances (same-type)
-- ---------------------------------------------------------------------------

instance : Add (Expr .u32)  := ⟨fun a b => ⟨s!"({a} + {b})"⟩⟩
instance : Sub (Expr .u32)  := ⟨fun a b => ⟨s!"({a} - {b})"⟩⟩
instance : Mul (Expr .u32)  := ⟨fun a b => ⟨s!"({a} * {b})"⟩⟩
instance : Div (Expr .u32)  := ⟨fun a b => ⟨s!"({a} / {b})"⟩⟩
instance : Mod (Expr .u32)  := ⟨fun a b => ⟨s!"({a} % {b})"⟩⟩

instance : Add (Expr .i32)  := ⟨fun a b => ⟨s!"({a} + {b})"⟩⟩
instance : Sub (Expr .i32)  := ⟨fun a b => ⟨s!"({a} - {b})"⟩⟩
instance : Mul (Expr .i32)  := ⟨fun a b => ⟨s!"({a} * {b})"⟩⟩

instance : Add (Expr .f32)  := ⟨fun a b => ⟨s!"({a} + {b})"⟩⟩
instance : Sub (Expr .f32)  := ⟨fun a b => ⟨s!"({a} - {b})"⟩⟩
instance : Mul (Expr .f32)  := ⟨fun a b => ⟨s!"({a} * {b})"⟩⟩
instance : Div (Expr .f32)  := ⟨fun a b => ⟨s!"({a} / {b})"⟩⟩
instance : Neg (Expr .f32)  := ⟨fun a   => ⟨s!"-({a})"⟩⟩

instance : Add (Expr .vec3f) := ⟨fun a b => ⟨s!"({a} + {b})"⟩⟩
instance : Sub (Expr .vec3f) := ⟨fun a b => ⟨s!"({a} - {b})"⟩⟩
instance : Mul (Expr .vec3f) := ⟨fun a b => ⟨s!"({a} * {b})"⟩⟩
instance : Div (Expr .vec3f) := ⟨fun a b => ⟨s!"({a} / {b})"⟩⟩
instance : Neg (Expr .vec3f) := ⟨fun a   => ⟨s!"-({a})"⟩⟩

instance : Add (Expr .vec2f) := ⟨fun a b => ⟨s!"({a} + {b})"⟩⟩
instance : Sub (Expr .vec2f) := ⟨fun a b => ⟨s!"({a} - {b})"⟩⟩
instance : Mul (Expr .vec2f) := ⟨fun a b => ⟨s!"({a} * {b})"⟩⟩

-- Scalar × vector (heterogeneous multiply / divide)
instance : HMul (Expr .f32) (Expr .vec3f) (Expr .vec3f) := ⟨fun s v => ⟨s!"({s} * {v})"⟩⟩
instance : HMul (Expr .vec3f) (Expr .f32) (Expr .vec3f) := ⟨fun v s => ⟨s!"({v} * {s})"⟩⟩
instance : HDiv (Expr .vec3f) (Expr .f32) (Expr .vec3f) := ⟨fun v s => ⟨s!"({v} / {s})"⟩⟩
instance : HMul (Expr .f32) (Expr .vec2f) (Expr .vec2f) := ⟨fun s v => ⟨s!"({s} * {v})"⟩⟩

-- ---------------------------------------------------------------------------
-- Bit operations on u32
-- ---------------------------------------------------------------------------

def shlU  (a b : Expr .u32) : Expr .u32 := ⟨s!"({a} << {b})"⟩
def shrU  (a b : Expr .u32) : Expr .u32 := ⟨s!"({a} >> {b})"⟩
def bandU (a b : Expr .u32) : Expr .u32 := ⟨s!"({a} & {b})"⟩
def borU  (a b : Expr .u32) : Expr .u32 := ⟨s!"({a} | {b})"⟩
def bxorU (a b : Expr .u32) : Expr .u32 := ⟨s!"({a} ^ {b})"⟩
def bnotU (a   : Expr .u32) : Expr .u32 := ⟨s!"~({a})"⟩
def bandI (a b : Expr .i32) : Expr .i32 := ⟨s!"({a} & {b})"⟩

notation:75 a " .<< " b => shlU  a b
notation:75 a " .>> " b => shrU  a b
notation:70 a " .&  " b => bandU a b
notation:65 a " .|  " b => borU  a b
notation:60 a " .^  " b => bxorU a b

-- ---------------------------------------------------------------------------
-- Comparisons → Expr .bool  (polymorphic over numeric types)
-- ---------------------------------------------------------------------------

def eqE  {ty} (a b : Expr ty) : Expr .bool := ⟨s!"({a} == {b})"⟩
def neE  {ty} (a b : Expr ty) : Expr .bool := ⟨s!"({a} != {b})"⟩
def ltE  {ty} (a b : Expr ty) : Expr .bool := ⟨s!"({a} < {b})"⟩
def gtE  {ty} (a b : Expr ty) : Expr .bool := ⟨s!"({a} > {b})"⟩
def leE  {ty} (a b : Expr ty) : Expr .bool := ⟨s!"({a} <= {b})"⟩
def geE  {ty} (a b : Expr ty) : Expr .bool := ⟨s!"({a} >= {b})"⟩

notation:50 a " .== " b => eqE a b
notation:50 a " .!= " b => neE a b
notation:50 a " .<  " b => ltE a b
notation:50 a " .>  " b => gtE a b
notation:50 a " .<= " b => leE a b
notation:50 a " .>= " b => geE a b

-- ---------------------------------------------------------------------------
-- Boolean operations
-- ---------------------------------------------------------------------------

def andE (a b : Expr .bool) : Expr .bool := ⟨s!"({a} && {b})"⟩
def orE  (a b : Expr .bool) : Expr .bool := ⟨s!"({a} || {b})"⟩
def notE (a   : Expr .bool) : Expr .bool := ⟨s!"!({a})"⟩

notation:35 a " .&& " b => andE a b
notation:30 a " .|| " b => orE  a b

-- ---------------------------------------------------------------------------
-- WGSL built-in functions
-- ---------------------------------------------------------------------------

def wAbs    (a : Expr .f32)   : Expr .f32   := ⟨s!"abs({a})"⟩
def wAbsI   (a : Expr .i32)   : Expr .i32   := ⟨s!"abs({a})"⟩
def wSqrt   (a : Expr .f32)   : Expr .f32   := ⟨s!"sqrt({a})"⟩
def wLog2   (a : Expr .f32)   : Expr .f32   := ⟨s!"log2({a})"⟩
def wCos    (a : Expr .f32)   : Expr .f32   := ⟨s!"cos({a})"⟩
def wSin    (a : Expr .f32)   : Expr .f32   := ⟨s!"sin({a})"⟩
def wSign   (a : Expr .f32)   : Expr .f32   := ⟨s!"sign({a})"⟩
def wFloor  (a : Expr .f32)   : Expr .f32   := ⟨s!"floor({a})"⟩
def wAbsV3  (a : Expr .vec3f) : Expr .vec3f := ⟨s!"abs({a})"⟩
def wSignV3 (a : Expr .vec3f) : Expr .vec3f := ⟨s!"sign({a})"⟩
def wMin    (a b : Expr .f32)   : Expr .f32   := ⟨s!"min({a}, {b})"⟩
def wMax    (a b : Expr .f32)   : Expr .f32   := ⟨s!"max({a}, {b})"⟩
def wMinV3  (a b : Expr .vec3f) : Expr .vec3f := ⟨s!"min({a}, {b})"⟩
def wMaxV3  (a b : Expr .vec3f) : Expr .vec3f := ⟨s!"max({a}, {b})"⟩
def wMinU   (a b : Expr .u32)   : Expr .u32   := ⟨s!"min({a}, {b})"⟩
def wMaxU   (a b : Expr .u32)   : Expr .u32   := ⟨s!"max({a}, {b})"⟩
def wClamp  (v lo hi : Expr .f32)   : Expr .f32   := ⟨s!"clamp({v}, {lo}, {hi})"⟩
def wClampV3 (v lo hi : Expr .vec3f) : Expr .vec3f := ⟨s!"clamp({v}, {lo}, {hi})"⟩
def wMix    (a b : Expr .vec3f) (t : Expr .f32) : Expr .vec3f := ⟨s!"mix({a}, {b}, {t})"⟩
def wPow    (b e : Expr .vec3f) : Expr .vec3f := ⟨s!"pow({b}, {e})"⟩
def wNorm   (v : Expr .vec3f)   : Expr .vec3f := ⟨s!"normalize({v})"⟩
def wCross  (a b : Expr .vec3f) : Expr .vec3f := ⟨s!"cross({a}, {b})"⟩
def wDot    (a b : Expr .vec3f) : Expr .f32   := ⟨s!"dot({a}, {b})"⟩
def wLen    (v : Expr .vec3f)   : Expr .f32   := ⟨s!"length({v})"⟩
/-- WGSL select: `select(f, t, cond)` — returns `t` when `cond` is true. -/
def wSelect {ty} (f t : Expr ty) (cond : Expr .bool) : Expr ty :=
  ⟨s!"select({f}, {t}, {cond})"⟩
/-- `arrayLength(&arr)` — takes address internally. -/
def wArrayLen (arr : Expr (.arr ty)) : Expr .u32 := ⟨s!"arrayLength(&{arr})"⟩

-- ---------------------------------------------------------------------------
-- Array access (typed)
-- ---------------------------------------------------------------------------

/-- Dynamic array element: `arr[i]` -/
def arrIdx  (arr : Expr (.arr ty))    (i : Expr .u32) : Expr ty := ⟨s!"{arr}[{i}]"⟩
/-- Fixed-size array element: `arr[i]` -/
def arrIdxN (arr : Expr (.arrN ty n)) (i : Expr .u32) : Expr ty := ⟨s!"{arr}[{i}]"⟩

-- ---------------------------------------------------------------------------
-- Pointer operations
-- ---------------------------------------------------------------------------

/-- `&e` → `ptr<function, T>` -/
def addrOf (e : Expr ty) : Expr (.ptrFn ty) := ⟨s!"&{e}"⟩
/-- `*ptr` → `T` -/
def deref  (e : Expr (.ptrFn ty)) : Expr ty := ⟨s!"*{e}"⟩

-- ---------------------------------------------------------------------------
-- Statement builders
-- ---------------------------------------------------------------------------

/-- `let name = e;`  — returns handle `⟨name⟩` -/
def letV (name : String) (e : Expr ty) : WB (Expr ty) := do
  emit s!"let {name} = {e};"
  return ⟨name⟩

/-- `var name = e;`  (mutable) -/
def varV (name : String) (e : Expr ty) : WB (Expr ty) := do
  emit s!"var {name} = {e};"
  return ⟨name⟩

/-- `var name: T;`  (mutable, uninitialized) -/
def varVT (name : String) (ty : WTy) : WB (Expr ty) := do
  emit s!"var {name}: {ty.render};"
  return ⟨name⟩

/-- `lhs = rhs;` -/
def assign (lhs rhs : Expr ty) : WB Unit := emit s!"{lhs} = {rhs};"

/-- `*ptr = val;` -/
def derefAssign (ptr : Expr (.ptrFn ty)) (val : Expr ty) : WB Unit :=
  emit s!"*{ptr} = {val};"

/-- `if (cond) { body }` -/
def ifB (cond : Expr .bool) (body : WB Unit) : WB Unit := do
  emit s!"if ({cond})"
  emitBlock body

/-- `if (cond) { thenB } else { elseB }` -/
def ifElse (cond : Expr .bool) (thenB elseB : WB Unit) : WB Unit := do
  emit s!"if ({cond})"
  emit "{"
  indented thenB
  emit "} else {"
  indented elseB
  emit "}"

/-- `for (var vn: u32 = init; cond(vn); vn = upd(vn)) { body(vn) }` -/
def forU (vn : String) (init : Expr .u32)
    (cond : Expr .u32 → Expr .bool)
    (upd  : Expr .u32 → Expr .u32)
    (body : Expr .u32 → WB Unit) : WB Unit := do
  let v : Expr .u32 := ⟨vn⟩
  emit s!"for (var {vn}: u32 = {init}; {cond v}; {vn} = {upd v})"
  emitBlock (body v)

/-- `while (cond) { body }` -/
def whileB (cond : Expr .bool) (body : WB Unit) : WB Unit := do
  emit s!"while ({cond})"
  emitBlock body

/-- `loop { body }` -/
def loopB (body : WB Unit) : WB Unit := do
  emit "loop"
  emitBlock body

def breakS : WB Unit := emit "break;"
def retV   : WB Unit := emit "return;"
def retE (e : Expr ty) : WB Unit := emit s!"return {e};"
def wBarrier : WB Unit := emit "workgroupBarrier();"

/-- Void statement call: `name(args...);` -/
def callS (name : String) (args : List String) : WB Unit :=
  emit s!"{name}({String.intercalate ", " args});"

-- ---------------------------------------------------------------------------
-- Typed function-call expressions (return an Expr of the appropriate type)
-- ---------------------------------------------------------------------------

def call1 (name : String) (a : Expr t1)                               : Expr ty := ⟨s!"{name}({a})"⟩
def call2 (name : String) (a : Expr t1) (b : Expr t2)                 : Expr ty := ⟨s!"{name}({a}, {b})"⟩
def call3 (name : String) (a : Expr t1) (b : Expr t2) (c : Expr t3)   : Expr ty := ⟨s!"{name}({a}, {b}, {c})"⟩
def call4 (name : String) (a : Expr t1) (b : Expr t2) (c : Expr t3) (d : Expr t4) : Expr ty :=
  ⟨s!"{name}({a}, {b}, {c}, {d})"⟩
def call5 (name : String) (a : Expr t1) (b : Expr t2) (c : Expr t3) (d : Expr t4) (e : Expr t5) : Expr ty :=
  ⟨s!"{name}({a}, {b}, {c}, {d}, {e})"⟩

-- ---------------------------------------------------------------------------
-- Function and declaration builders
-- ---------------------------------------------------------------------------

structure WGSLParam where
  name : String
  ty   : WTy

structure WGSLFn where
  name   : String
  params : List WGSLParam
  retTy  : Option WTy
  body   : WB Unit

def WGSLFn.render (fn : WGSLFn) : String :=
  let paramStr := String.intercalate ", " (fn.params.map (fun p => p.name ++ ": " ++ p.ty.render))
  let retStr := match fn.retTy with | some ty => " -> " ++ ty.render | none => ""
  let bodyLines := runBodyAt 1 fn.body
  "fn " ++ fn.name ++ "(" ++ paramStr ++ ")" ++ retStr ++ " {\n" ++
  String.intercalate "\n" bodyLines ++ "\n}"

/-- Top-level WGSL declarations placed before the entry function. -/
inductive WGSLDecl where
  | constU (name : String) (val : Nat)    : WGSLDecl  -- const name: u32 = val;
  | constF (name : String) (val : String) : WGSLDecl  -- const name: f32 = val;
  | fn     (f : WGSLFn)                  : WGSLDecl

def WGSLDecl.render : WGSLDecl → String
  | .constU n v => s!"const {n}: u32 = {v}u;"
  | .constF n v => s!"const {n}: f32 = {v};"
  | .fn f       => f.render

-- ---------------------------------------------------------------------------
-- Binding and entry-point spec
-- ---------------------------------------------------------------------------

structure Binding where
  group   : Nat  := 0
  binding : Nat
  name    : String
  ty      : WTy           -- array type (e.g. `.arr .f32`)
  ro      : Bool := false -- read-only storage?

private def Binding.render (b : Binding) : String :=
  let mode := if b.ro then "storage, read" else "storage, read_write"
  s!"@group({b.group}) @binding({b.binding}) var<{mode}> {b.name}: {b.ty.render};"

structure EntrySpec where
  name : String := "main"
  wgX  : Nat    := 64
  wgY  : Nat    := 1
  wgZ  : Nat    := 1
  gid  : Bool   := true   -- @builtin(global_invocation_id) gid: vec3<u32>
  lid  : Bool   := false  -- @builtin(local_invocation_id)  lid: vec3<u32>
  wid  : Bool   := false  -- @builtin(workgroup_id)          wid: vec3<u32>

private def renderEntry (e : EntrySpec) (body : WB Unit) : String :=
  let wg := if e.wgY > 1 || e.wgZ > 1
    then s!"@compute @workgroup_size({e.wgX}, {e.wgY}, {e.wgZ})"
    else s!"@compute @workgroup_size({e.wgX})"
  let ps : List String :=
    (if e.gid then ["    @builtin(global_invocation_id) gid: vec3<u32>"] else []) ++
    (if e.lid then ["    @builtin(local_invocation_id)  lid: vec3<u32>"] else []) ++
    (if e.wid then ["    @builtin(workgroup_id)         wid: vec3<u32>"] else [])
  let paramStr := String.intercalate ",\n" ps
  let bodyLines := runBodyAt 1 body
  wg ++ "\nfn " ++ e.name ++ "(\n" ++ paramStr ++ "\n) {\n" ++
  String.intercalate "\n" bodyLines ++ "\n}"

-- ---------------------------------------------------------------------------
-- Main assembly function
-- ---------------------------------------------------------------------------

/-- Build a complete WGSL shader module.
    - `bindings` — storage buffer declarations (typed)
    - `wgVars`   — workgroup variables: (name, elemTy, count)
    - `decls`    — typed constants and helper function declarations
    - `entry`    — entry-point spec (workgroup size, which builtins are used)
    - `body`     — entry function body as a typed `WB Unit` program -/
def buildShader
    (bindings : List Binding)
    (wgVars   : List (String × WTy × Nat))
    (decls    : List WGSLDecl)
    (entry    : EntrySpec)
    (body     : WB Unit) : String :=
  let sections :=
    bindings.map Binding.render ++
    wgVars.map (fun (n, t, sz) => s!"var<workgroup> {n}: array<{t.render}, {sz}>;") ++
    decls.map WGSLDecl.render ++
    [renderEntry entry body]
  String.intercalate "\n\n" sections ++ "\n"

end AlgorithmLib.WGSL
