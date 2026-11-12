module
public import Lean
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
# `PtxIr` — the PTX a kernel ships, as data

The subset of PTX the shipped kernels are written in, as a typed syntax: each
instruction's opcode, its type and modifiers, and its operands. A kernel is
proven about this data, and the text an artifact carries is its canonical
printing (`Module.text`), which writes each instruction as the one line PTX
spells it with and nothing else.
-/

namespace AlgorithmLib.PtxIr

/-- A register or value type. -/
inductive Ty where
  | pred | b8 | b16 | b32 | b64 | u8 | u16 | u32 | u64 | s16 | s32 | s64 | f32 | f64
  deriving Repr, DecidableEq, Inhabited

def Ty.name : Ty → String
  | .pred => "pred" | .b8 => "b8" | .b16 => "b16" | .b32 => "b32" | .b64 => "b64"
  | .u8 => "u8" | .u16 => "u16" | .u32 => "u32" | .u64 => "u64"
  | .s16 => "s16" | .s32 => "s32" | .s64 => "s64" | .f32 => "f32" | .f64 => "f64"

/-- Bits of a type. -/
def Ty.width : Ty → Nat
  | .pred => 1 | .b8 | .u8 => 8 | .b16 | .u16 | .s16 => 16
  | .b32 | .u32 | .s32 | .f32 => 32 | .b64 | .u64 | .s64 | .f64 => 64

def Ty.isSigned : Ty → Bool
  | .s16 | .s32 | .s64 => true | _ => false

def Ty.isFloat : Ty → Bool
  | .f32 | .f64 => true | _ => false

def Ty.all : List Ty := [.pred, .b8, .b16, .b32, .b64, .u8, .u16, .u32, .u64, .s16, .s32, .s64, .f32, .f64]

/-- A register: a value's, in the register class `cls` (`%r5`); one an
    instruction makes for itself, named after the instruction's position
    (`%k12_0`); a special one (`%tid.x`); or one named freely. -/
inductive Reg where
  | val (cls : String) (id : Nat)
  | own (pfx : String) (tag n : Nat)
  | special (s : String)
  | name (s : String)
  deriving Repr, DecidableEq, Hashable, Inhabited

/-- A label: a block's (`B3`), the other side of a branch at a position
    (`E7`), the line of a move `Kernel.Coalesce` removed (`C12`), or one
    named freely. -/
inductive Lbl where
  | blk (id : Nat)
  | els (tag : Nat)
  | gone (pc : Nat)
  | name (s : String)
  deriving Repr, DecidableEq, Hashable, Inhabited

/-- An operand: a register, an integer, a float's bits, or a symbol (a
    parameter or a shared array). -/
inductive Opnd where
  | reg (r : Reg)
  | imm (v : Int)
  | f32 (bits : Nat)
  | f64 (bits : Nat)
  | sym (n : String)
  deriving Repr, DecidableEq, Inhabited

/-- An address: a register or symbol, and a byte offset. -/
structure Addr where
  base : Opnd
  off : Int := 0
  deriving Repr, DecidableEq, Inhabited

/-- A state space. -/
inductive Space where
  | global | shared | param | generic
  deriving Repr, DecidableEq, Inhabited

/-- A rounding modifier. -/
inductive Rnd where
  | rn | rz | rni | rzi
  deriving Repr, DecidableEq, Inhabited

def Rnd.name : Rnd → String
  | .rn => "rn" | .rz => "rz" | .rni => "rni" | .rzi => "rzi"

/-- A two-operand operation. `mulLo`/`mulWide`/`mulHi` are integer products,
    `minNaN`/`maxNaN` the float ones that propagate NaN. -/
inductive BinOp where
  | add | sub | mul | mulLo | mulWide | mulHi | div | rem | min | max | minNaN | maxNaN
  | and | or | xor | shl | shr
  deriving Repr, DecidableEq, Inhabited

def BinOp.name : BinOp → String
  | .add => "add" | .sub => "sub" | .mul => "mul" | .mulLo => "mul.lo" | .mulWide => "mul.wide"
  | .mulHi => "mul.hi" | .div => "div" | .rem => "rem" | .min => "min" | .max => "max"
  | .minNaN => "min.NaN" | .maxNaN => "max.NaN"
  | .and => "and" | .or => "or" | .xor => "xor" | .shl => "shl" | .shr => "shr"

/-- A one-operand operation; the approximations are the hardware's. -/
inductive UnOp where
  | neg | abs | not | brev | clz | popc | sqrt | rsqrtApprox | sqrtApprox | ex2Approx | lg2Approx | rcpApprox
  | sinApprox | cosApprox
  deriving Repr, DecidableEq, Inhabited

def UnOp.name : UnOp → String
  | .neg => "neg" | .abs => "abs" | .not => "not" | .brev => "brev" | .clz => "clz" | .popc => "popc"
  | .sqrt => "sqrt" | .rsqrtApprox => "rsqrt.approx" | .sqrtApprox => "sqrt.approx"
  | .ex2Approx => "ex2.approx" | .lg2Approx => "lg2.approx" | .rcpApprox => "rcp.approx"
  | .sinApprox => "sin.approx" | .cosApprox => "cos.approx"

/-- A comparison; the `u` forms hold when unordered. -/
inductive Cmp where
  | eq | ne | lt | le | gt | ge | lo | ls | hi | hs | equ | neu | ltu | leu | gtu | geu | nan | num
  deriving Repr, DecidableEq, Inhabited

def Cmp.name : Cmp → String
  | .eq => "eq" | .ne => "ne" | .lt => "lt" | .le => "le" | .gt => "gt" | .ge => "ge"
  | .lo => "lo" | .ls => "ls" | .hi => "hi" | .hs => "hs"
  | .equ => "equ" | .neu => "neu" | .ltu => "ltu" | .leu => "leu" | .gtu => "gtu" | .geu => "geu"
  | .nan => "nan" | .num => "num"

def Cmp.all : List Cmp :=
  [.eq, .ne, .lt, .le, .gt, .ge, .lo, .ls, .hi, .hs, .equ, .neu, .ltu, .leu, .gtu, .geu, .nan, .num]

/-- A shuffle's mode. -/
inductive Shfl where
  | bfly | idx | up | down
  deriving Repr, DecidableEq, Inhabited

def Shfl.name : Shfl → String
  | .bfly => "bfly" | .idx => "idx" | .up => "up" | .down => "down"

/-- What an `atom` does, at the device's scope: a read-modify-write that
    answers the old value, a load or a store; ordered, `acq_rel`, `acquire`
    and `release`. -/
inductive AtomOp where
  | add | and | or | xor | exch | umin | umax | smin | smax | load | store
  deriving Repr, DecidableEq, Inhabited

def AtomOp.name : AtomOp → String
  | .add => "add" | .and => "and" | .or => "or" | .xor => "xor" | .exch => "exch"
  | .umin | .smin => "min" | .umax | .smax => "max" | .load => "load" | .store => "store"

/-- An instruction. -/
inductive Instr where
  | mov (t : Ty) (d : Reg) (a : Opnd)
  | bin (o : BinOp) (rnd : Option Rnd) (t : Ty) (d : Reg) (a b : Opnd)
  | un (o : UnOp) (rnd : Option Rnd) (t : Ty) (d : Reg) (a : Opnd)
  /-- `fma.rnd.t` or, with no rounding, `mad.lo.t`. -/
  | fma (rnd : Option Rnd) (t : Ty) (d : Reg) (a b c : Opnd)
  | cvt (rnd : Option Rnd) (dt st : Ty) (d : Reg) (a : Opnd)
  | cvta (to : Bool) (sp : Space) (t : Ty) (d : Reg) (a : Opnd)
  | setp (c : Cmp) (t : Ty) (p : Reg) (a b : Opnd)
  | selp (t : Ty) (d : Reg) (a b : Opnd) (p : Opnd)
  | prmt (d : Reg) (a b c : Opnd)
  /-- `dp4a.s32.s32`: `c` plus the dot product of the four signed bytes of
      `a` and of `b`. -/
  | dp4a (d : Reg) (a b c : Opnd)
  | ld (sp : Space) (nc : Bool) (t : Ty) (d : Reg) (a : Addr)
  | st (sp : Space) (t : Ty) (a : Addr) (v : Opnd)
  /-- An atomic access of `t`'s width at `a`: `d` gets the old value (but
      for a store), `v` is the operand (but for a load); `ordered`, the
      thread's other accesses ordered around it, else `relaxed`. -/
  | atom (sp : Space) (o : AtomOp) (ordered : Bool) (t : Ty) (d : Option Reg) (a : Addr) (v : Opnd)
  /-- `red.shared.add.u32`: `v` added to the shared word at `a`. -/
  | redShared (a : Addr) (v : Opnd)
  | shfl (m : Shfl) (d : Reg) (a b c mask : Opnd)
  | vote (d : Reg) (p mask : Opnd)
  /-- One `mma.sync.m8n8k16` of int8, its even output column to `d`. -/
  | mma (d : Reg) (a b c : Opnd)
  | bar (n : Nat)
  | barWarp (mask : Opnd)
  | bra (uni : Bool) (l : Lbl)
  | ret
  deriving Repr, DecidableEq, Inhabited

/-- A line of a body: a label, or an instruction under an optional guard
    (`@p` or `@!p`). -/
inductive Line where
  | label (l : Lbl)
  | ins (guard : Option (Bool × Reg)) (i : Instr)
  deriving Repr, DecidableEq, Inhabited

/-- Registers declared together: a numbered range `%p<n>`, or names. -/
inductive RegDecl where
  | range (t : Ty) (pfx : String) (n : Nat)
  | names (t : Ty) (ns : List Reg)
  deriving Repr, DecidableEq, Inhabited

/-- A block's shared array. -/
structure Shared where
  align : Nat
  sym : String
  bytes : Nat
  deriving Repr, DecidableEq, Inhabited

/-- An entry point: its name and parameters, registers, shared array and body. -/
structure Entry where
  name : String
  params : List (Ty × String)
  regs : List RegDecl
  shared : Option Shared := none
  body : List Line
  /-- Launch bounds for the assembler (`.maxntid`, `.minnctapersm`): at most
      this many threads a block, and registers few enough for this many
      blocks a multiprocessor. A larger block does not launch. -/
  bounds : Option (Nat × Nat) := none
  deriving Repr, DecidableEq, Inhabited

/-- A module: the PTX version, target and address size, the shared arrays at
    module scope, and its entry points. -/
structure Module where
  version : String
  target : String
  addressSize : Nat
  shared : List Shared := []
  entries : List Entry
  deriving Repr, DecidableEq, Inhabited

/-! ## The canonical text -/

def hex8 (n : Nat) : String :=
  String.ofList <| (List.range 8).reverse.map fun i =>
    let d := (n / 16 ^ i) % 16
    if d < 10 then Char.ofNat (48 + d) else Char.ofNat (55 + d)

def hex16 (n : Nat) : String := hex8 (n / 2 ^ 32) ++ hex8 (n % 2 ^ 32)

def Reg.text : Reg → String
  | .val c i => s!"%{c}{i}"
  | .own p t n => s!"%{p}{t}_{n}"
  | .special s => "%" ++ s
  | .name s => "%" ++ s

def Lbl.text : Lbl → String
  | .blk i => s!"B{i}"
  | .els t => s!"E{t}"
  | .gone pc => s!"C{pc}"
  | .name s => s

def Opnd.text : Opnd → String
  | .reg r => r.text
  | .imm v => toString v
  | .f32 b => "0f" ++ hex8 b
  | .f64 b => "0d" ++ hex16 b
  | .sym n => n

def Addr.text (a : Addr) : String :=
  if a.off == 0 then s!"[{a.base.text}]"
  else if a.off > 0 then s!"[{a.base.text}+{a.off}]"
  else s!"[{a.base.text}{a.off}]"

def Space.dot : Space → String
  | .global => ".global" | .shared => ".shared" | .param => ".param" | .generic => ""

def rndDot : Option Rnd → String
  | some r => "." ++ r.name
  | none => ""

def Instr.text : Instr → String
  | .mov t d a => s!"mov.{t.name} {d.text}, {a.text}"
  | .bin o r t d a b => s!"{o.name}{rndDot r}.{t.name} {d.text}, {a.text}, {b.text}"
  | .un o r t d a => s!"{o.name}{rndDot r}.{t.name} {d.text}, {a.text}"
  | .fma (some r) t d a b c => s!"fma.{r.name}.{t.name} {d.text}, {a.text}, {b.text}, {c.text}"
  | .fma none t d a b c => s!"mad.lo.{t.name} {d.text}, {a.text}, {b.text}, {c.text}"
  | .cvt r dt sty d a => s!"cvt{rndDot r}.{dt.name}.{sty.name} {d.text}, {a.text}"
  | .cvta to sp t d a => s!"cvta{if to then ".to" else ""}{sp.dot}.{t.name} {d.text}, {a.text}"
  | .setp c t p a b => s!"setp.{c.name}.{t.name} {p.text}, {a.text}, {b.text}"
  | .selp t d a b p => s!"selp.{t.name} {d.text}, {a.text}, {b.text}, {p.text}"
  | .prmt d a b c => s!"prmt.b32 {d.text}, {a.text}, {b.text}, {c.text}"
  | .dp4a d a b c => s!"dp4a.s32.s32 {d.text}, {a.text}, {b.text}, {c.text}"
  | .ld sp nc t d a => s!"ld{sp.dot}{if nc then ".nc" else ""}.{t.name} {d.text}, {a.text}"
  | .st sp t a v => s!"st{sp.dot}.{t.name} {a.text}, {v.text}"
  | .atom sp .load od t d a _ =>
      s!"ld{sp.dot}.{if od then "acquire" else "relaxed"}.gpu.{t.name} {(d.map Reg.text).getD "_"}, {a.text}"
  | .atom sp .store od t _ a v => s!"st{sp.dot}.{if od then "release" else "relaxed"}.gpu.{t.name} {a.text}, {v.text}"
  | .atom sp o od t d a v =>
      s!"atom{sp.dot}.{if od then "acq_rel" else "relaxed"}.gpu.{o.name}.{t.name} {(d.map Reg.text).getD "_"}, {a.text}, {v.text}"
  | .redShared a v => s!"red.shared.add.u32 {a.text}, {v.text}"
  | .shfl m d a b c k => s!"shfl.sync.{m.name}.b32 {d.text}, {a.text}, {b.text}, {c.text}, {k.text}"
  | .vote d p k => s!"vote.sync.ballot.b32 {d.text}, {p.text}, {k.text}"
  | .mma d a b c =>
      -- `mma` takes registers only: a register operand as it is, any other
      -- moved into a scratch register of the block first
      let sel (o : Opnd) (nm : String) : String × String := match o with
        | .reg r => (r.text, "")
        | o => (nm, s!"mov.b32 {nm}, {o.text}; ")
      let (ta, ma) := sel a "%mma_a"
      let (tb, mb) := sel b "%mma_b"
      let (tc, mc) := sel c "%mma_c"
      s!"\{ .reg .b32 %mma_a, %mma_b, %mma_c, %mma_z, %mma_o; {ma}{mb}{mc}mov.b32 %mma_z, 0; " ++
      s!"mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 \{{d.text}, %mma_o}, \{{ta}}, \{{tb}}, \{{tc}, %mma_z}; }"
  | .bar n => s!"bar.sync {n}"
  | .barWarp k => s!"bar.warp.sync {k.text}"
  | .bra u l => s!"bra{if u then ".uni" else ""} {l.text}"
  | .ret => "ret"

/-- What ends an instruction's line: `;`, except after an `mma`, which is a
    block of its own (its scratch registers scoped to it) and ends at `}`. -/
def Instr.term : Instr → String
  | .mma .. => ""
  | _ => ";"

def Line.text : Line → String
  | .label l => l.text ++ ":"
  | .ins none i => "\t" ++ i.text ++ i.term
  | .ins (some (neg, p)) i => s!"\t@{if neg then "!" else ""}{p.text} {i.text}{i.term}"

def RegDecl.text : RegDecl → String
  | .range t p n => s!"\t.reg .{t.name} {p}<{n}>;"
  | .names t ns => s!"\t.reg .{t.name} {", ".intercalate (ns.map Reg.text)};"

def Shared.text (s : Shared) : String := s!".shared .align {s.align} .b8 {s.sym}[{s.bytes}];"

def Entry.text (e : Entry) : String :=
  let ps := if e.params.isEmpty then [] else
    [",\n".intercalate (e.params.map fun (t, n) => s!"\t.param .{t.name} {n}")]
  let sh := match e.shared with
    | some s => ["\t" ++ s.text]
    | none => []
  let bd := match e.bounds with
    | some (n, m) => [s!".maxntid {n}, 1, 1", s!".minnctapersm {m}"]
    | none => []
  "\n".intercalate ([s!".visible .entry {e.name}("] ++ ps ++ [")"] ++ bd ++ ["{"] ++ e.regs.map RegDecl.text ++ sh ++
    e.body.map Line.text ++ ["}"])

/-- The module's text: its header, its shared arrays, then its entries, a
    blank line after each. -/
def Module.text (m : Module) : String :=
  "\n".intercalate ([s!".version {m.version}", s!".target {m.target}", s!".address_size {m.addressSize}", ""] ++
    m.shared.map Shared.text ++ [""] ++ m.entries.map (·.text ++ "\n"))

end AlgorithmLib.PtxIr
