module
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-! # x86-64 machine code, assembled in Lean

A `Body` is a list of instructions, labels, alignments and literal bytes;
`assemble` makes its bytes, choosing the encoding GNU `as` chooses for the
Intel syntax `Body.intel` prints. `X86Check` holds it to `as`.

    def body : Body :=
      [mov rax rdi,
       add rax (qword (rdi + rsi*8 + 0x8)),
       vmulsd xmm0 xmm1 (qword (rip "c0")),
       ret,
       p2align 3, lbl "c0", f64 0.5]

A body can refer only to its own labels, so it is position-independent. It
aligns to at most 64 bytes and must be placed on a 64-byte boundary. -/

namespace AlgorithmLib.X86

-- ---------------------------------------------------------------------------
-- Operands
-- ---------------------------------------------------------------------------

/-- The width of a general register: 8, 32 or 64 bits. -/
inductive Width | b | d | q
  deriving DecidableEq, Repr, Inhabited

/-- A general register: its number (0 = rax ... 15 = r15) at a width. -/
structure Reg where
  n : Nat
  w : Width
  deriving DecidableEq, Repr, Inhabited

/-- A vector register: xmm (128 bits) or ymm (256 bits). -/
structure VReg where
  n : Nat
  ymm : Bool
  deriving DecidableEq, Repr, Inhabited

def rax : Reg := ⟨0, .q⟩
def rcx : Reg := ⟨1, .q⟩
def rdx : Reg := ⟨2, .q⟩
def rbx : Reg := ⟨3, .q⟩
def rsp : Reg := ⟨4, .q⟩
def rbp : Reg := ⟨5, .q⟩
def rsi : Reg := ⟨6, .q⟩
def rdi : Reg := ⟨7, .q⟩
def r8 : Reg := ⟨8, .q⟩
def r9 : Reg := ⟨9, .q⟩
def r10 : Reg := ⟨10, .q⟩
def r11 : Reg := ⟨11, .q⟩
def r12 : Reg := ⟨12, .q⟩
def r13 : Reg := ⟨13, .q⟩
def r14 : Reg := ⟨14, .q⟩
def r15 : Reg := ⟨15, .q⟩

def eax : Reg := ⟨0, .d⟩
def ecx : Reg := ⟨1, .d⟩
def edx : Reg := ⟨2, .d⟩
def ebx : Reg := ⟨3, .d⟩
def esp : Reg := ⟨4, .d⟩
def ebp : Reg := ⟨5, .d⟩
def esi : Reg := ⟨6, .d⟩
def edi : Reg := ⟨7, .d⟩
def r8d : Reg := ⟨8, .d⟩
def r9d : Reg := ⟨9, .d⟩
def r10d : Reg := ⟨10, .d⟩
def r11d : Reg := ⟨11, .d⟩
def r12d : Reg := ⟨12, .d⟩
def r13d : Reg := ⟨13, .d⟩
def r14d : Reg := ⟨14, .d⟩
def r15d : Reg := ⟨15, .d⟩

def al : Reg := ⟨0, .b⟩
def cl : Reg := ⟨1, .b⟩
def dl : Reg := ⟨2, .b⟩
def bl : Reg := ⟨3, .b⟩
def spl : Reg := ⟨4, .b⟩
def bpl : Reg := ⟨5, .b⟩
def sil : Reg := ⟨6, .b⟩
def dil : Reg := ⟨7, .b⟩
def r8b : Reg := ⟨8, .b⟩
def r9b : Reg := ⟨9, .b⟩
def r10b : Reg := ⟨10, .b⟩
def r11b : Reg := ⟨11, .b⟩
def r12b : Reg := ⟨12, .b⟩
def r13b : Reg := ⟨13, .b⟩
def r14b : Reg := ⟨14, .b⟩
def r15b : Reg := ⟨15, .b⟩

def xmm0 : VReg := ⟨0, false⟩
def xmm1 : VReg := ⟨1, false⟩
def xmm2 : VReg := ⟨2, false⟩
def xmm3 : VReg := ⟨3, false⟩
def xmm4 : VReg := ⟨4, false⟩
def xmm5 : VReg := ⟨5, false⟩
def xmm6 : VReg := ⟨6, false⟩
def xmm7 : VReg := ⟨7, false⟩
def xmm8 : VReg := ⟨8, false⟩
def xmm9 : VReg := ⟨9, false⟩
def xmm10 : VReg := ⟨10, false⟩
def xmm11 : VReg := ⟨11, false⟩
def xmm12 : VReg := ⟨12, false⟩
def xmm13 : VReg := ⟨13, false⟩
def xmm14 : VReg := ⟨14, false⟩
def xmm15 : VReg := ⟨15, false⟩

def ymm0 : VReg := ⟨0, true⟩
def ymm1 : VReg := ⟨1, true⟩
def ymm2 : VReg := ⟨2, true⟩
def ymm3 : VReg := ⟨3, true⟩
def ymm4 : VReg := ⟨4, true⟩
def ymm5 : VReg := ⟨5, true⟩
def ymm6 : VReg := ⟨6, true⟩
def ymm7 : VReg := ⟨7, true⟩
def ymm8 : VReg := ⟨8, true⟩
def ymm9 : VReg := ⟨9, true⟩
def ymm10 : VReg := ⟨10, true⟩
def ymm11 : VReg := ⟨11, true⟩
def ymm12 : VReg := ⟨12, true⟩
def ymm13 : VReg := ⟨13, true⟩
def ymm14 : VReg := ⟨14, true⟩
def ymm15 : VReg := ⟨15, true⟩

/-- What an address is relative to: a register, or a label of the body
    (`[rip+label]`). -/
inductive Base | reg (r : Reg) | rip (label : String)
  deriving DecidableEq, Repr, Inhabited

/-- `base + index*scale + disp`. -/
structure Addr where
  base : Base
  index : Option (Reg × Nat) := none
  disp : Int := 0
  deriving DecidableEq, Repr, Inhabited

/-- `index*scale`, on its way into an `Addr`. -/
structure Scaled where
  r : Reg
  s : Nat

instance : HMul Reg Nat Scaled := ⟨fun r s => ⟨r, s⟩⟩
instance : Coe Reg Addr := ⟨fun r => { base := .reg r }⟩
instance : HAdd Reg Scaled Addr := ⟨fun r i => { base := .reg r, index := some (i.r, i.s) }⟩
instance : HAdd Reg Nat Addr := ⟨fun r d => { base := .reg r, disp := d }⟩
instance : HSub Reg Nat Addr := ⟨fun r d => { base := .reg r, disp := -d }⟩
instance : HAdd Addr Nat Addr := ⟨fun a d => { a with disp := a.disp + d }⟩
instance : HSub Addr Nat Addr := ⟨fun a d => { a with disp := a.disp - d }⟩

/-- `[rip+label]`: the address of a label of the body. -/
def rip (label : String) : Addr := { base := .rip label }

/-- A memory operand: an address and how many bytes the instruction reads or
    writes there (`BYTE PTR` ... `YMMWORD PTR`), or 0 where the instruction
    implies none (`lea`). The size is what the source says; the encoding takes
    the width from the instruction and its registers. -/
structure Mem where
  size : Nat
  addr : Addr
  deriving DecidableEq, Repr, Inhabited

def byte (a : Addr) : Mem := ⟨1, a⟩
def dword (a : Addr) : Mem := ⟨4, a⟩
def qword (a : Addr) : Mem := ⟨8, a⟩
def xmmword (a : Addr) : Mem := ⟨16, a⟩
def ymmword (a : Addr) : Mem := ⟨32, a⟩
/-- An address with no size, for `lea`. -/
def mem (a : Addr) : Mem := ⟨0, a⟩

/-- An instruction's operand. An immediate is the integer the source writes;
    it is taken modulo the instruction's width. -/
inductive Opd
  | r (x : Reg)
  | v (x : VReg)
  | m (x : Mem)
  | i (x : Int)
  deriving DecidableEq, Repr, Inhabited

instance : Coe Reg Opd := ⟨.r⟩
instance : Coe VReg Opd := ⟨.v⟩
instance : Coe Mem Opd := ⟨.m⟩
instance : OfNat Opd n := ⟨.i n⟩
instance : Neg Opd := ⟨fun | .i x => .i (-x) | o => o⟩

-- ---------------------------------------------------------------------------
-- Bodies
-- ---------------------------------------------------------------------------

/-- A branch condition, in encoding order (`jo` = 0 ... `jg` = 15). -/
inductive Cond | o | no | b | ae | e | ne | be | a | s | ns | p | np | l | ge | le | g
  deriving DecidableEq, Repr, Inhabited

def Cond.code : Cond → Nat
  | .o => 0 | .no => 1 | .b => 2 | .ae => 3 | .e => 4 | .ne => 5 | .be => 6 | .a => 7
  | .s => 8 | .ns => 9 | .p => 10 | .np => 11 | .l => 12 | .ge => 13 | .le => 14 | .g => 15

def Cond.name : Cond → String
  | .o => "o" | .no => "no" | .b => "b" | .ae => "ae" | .e => "e" | .ne => "ne"
  | .be => "be" | .a => "a" | .s => "s" | .ns => "ns" | .p => "p" | .np => "np"
  | .l => "l" | .ge => "ge" | .le => "le" | .g => "g"

inductive Item
  /-- An instruction, by its Intel mnemonic. -/
  | ins (mn : String) (ops : List Opd)
  /-- `jmp label`. -/
  | jmp (label : String)
  /-- `j<cond> label`. -/
  | jcc (c : Cond) (label : String)
  /-- A label, local to the body. -/
  | label (name : String)
  /-- Pad with NOPs to a multiple of `2^log2` (`.p2align`). -/
  | align (log2 : Nat)
  /-- Literal bytes (`.byte`). -/
  | data (bs : List UInt8)
  deriving DecidableEq, Repr, Inhabited

/-- A body of machine code: what one `cl_native_load` maps. -/
abbrev Body := List Item

def lbl (name : String) : Item := .label name
def p2align (log2 : Nat) : Item := .align log2
def bytes (bs : List UInt8) : Item := .data bs

def le (n : Nat) (x : Nat) : List UInt8 :=
  (List.range n).map fun k => UInt8.ofNat (x >>> (8 * k))

/-- A `double`, as its eight bytes. -/
def f64 (x : Float) : Item := .data (le 8 x.toBits.toNat)
/-- A `float`, as its four bytes. -/
def f32 (x : Float) : Item := .data (le 4 x.toFloat32.toBits.toNat)

def jmp (l : String) : Item := .jmp l
def jo (l : String) : Item := .jcc .o l
def jno (l : String) : Item := .jcc .no l
def jb (l : String) : Item := .jcc .b l
def jae (l : String) : Item := .jcc .ae l
def je (l : String) : Item := .jcc .e l
def jne (l : String) : Item := .jcc .ne l
def jbe (l : String) : Item := .jcc .be l
def ja (l : String) : Item := .jcc .a l
def js (l : String) : Item := .jcc .s l
def jns (l : String) : Item := .jcc .ns l
def jp (l : String) : Item := .jcc .p l
def jnp (l : String) : Item := .jcc .np l
def jl (l : String) : Item := .jcc .l l
def jge (l : String) : Item := .jcc .ge l
def jle (l : String) : Item := .jcc .le l
def jg (l : String) : Item := .jcc .g l

def ret : Item := .ins "ret" []
def sfence : Item := .ins "sfence" []
def lfence : Item := .ins "lfence" []
def mfence : Item := .ins "mfence" []
def vzeroupper : Item := .ins "vzeroupper" []

def mov (a b : Opd) : Item := .ins "mov" [a, b]
def movabs (a b : Opd) : Item := .ins "movabs" [a, b]
def lea (a b : Opd) : Item := .ins "lea" [a, b]
def add (a b : Opd) : Item := .ins "add" [a, b]
def or (a b : Opd) : Item := .ins "or" [a, b]
def adc (a b : Opd) : Item := .ins "adc" [a, b]
def sbb (a b : Opd) : Item := .ins "sbb" [a, b]
def and (a b : Opd) : Item := .ins "and" [a, b]
def sub (a b : Opd) : Item := .ins "sub" [a, b]
def xor (a b : Opd) : Item := .ins "xor" [a, b]
def cmp (a b : Opd) : Item := .ins "cmp" [a, b]
def test (a b : Opd) : Item := .ins "test" [a, b]
def inc (a : Opd) : Item := .ins "inc" [a]
def dec (a : Opd) : Item := .ins "dec" [a]
def shl (a b : Opd) : Item := .ins "shl" [a, b]
def shr (a b : Opd) : Item := .ins "shr" [a, b]
def sar (a b : Opd) : Item := .ins "sar" [a, b]

def vaddss (a b c : Opd) : Item := .ins "vaddss" [a, b, c]
def vaddsd (a b c : Opd) : Item := .ins "vaddsd" [a, b, c]
def vaddps (a b c : Opd) : Item := .ins "vaddps" [a, b, c]
def vaddpd (a b c : Opd) : Item := .ins "vaddpd" [a, b, c]
def vsubss (a b c : Opd) : Item := .ins "vsubss" [a, b, c]
def vsubsd (a b c : Opd) : Item := .ins "vsubsd" [a, b, c]
def vsubps (a b c : Opd) : Item := .ins "vsubps" [a, b, c]
def vsubpd (a b c : Opd) : Item := .ins "vsubpd" [a, b, c]
def vmulss (a b c : Opd) : Item := .ins "vmulss" [a, b, c]
def vmulsd (a b c : Opd) : Item := .ins "vmulsd" [a, b, c]
def vmulps (a b c : Opd) : Item := .ins "vmulps" [a, b, c]
def vmulpd (a b c : Opd) : Item := .ins "vmulpd" [a, b, c]
def vdivss (a b c : Opd) : Item := .ins "vdivss" [a, b, c]
def vdivsd (a b c : Opd) : Item := .ins "vdivsd" [a, b, c]
def vdivps (a b c : Opd) : Item := .ins "vdivps" [a, b, c]
def vdivpd (a b c : Opd) : Item := .ins "vdivpd" [a, b, c]
def vandps (a b c : Opd) : Item := .ins "vandps" [a, b, c]
def vandpd (a b c : Opd) : Item := .ins "vandpd" [a, b, c]
def vorps (a b c : Opd) : Item := .ins "vorps" [a, b, c]
def vorpd (a b c : Opd) : Item := .ins "vorpd" [a, b, c]
def vxorps (a b c : Opd) : Item := .ins "vxorps" [a, b, c]
def vxorpd (a b c : Opd) : Item := .ins "vxorpd" [a, b, c]
def vpand (a b c : Opd) : Item := .ins "vpand" [a, b, c]
def vpor (a b c : Opd) : Item := .ins "vpor" [a, b, c]
def vpxor (a b c : Opd) : Item := .ins "vpxor" [a, b, c]
def vpaddd (a b c : Opd) : Item := .ins "vpaddd" [a, b, c]
def vpaddq (a b c : Opd) : Item := .ins "vpaddq" [a, b, c]
def vpsubd (a b c : Opd) : Item := .ins "vpsubd" [a, b, c]
def vpsubq (a b c : Opd) : Item := .ins "vpsubq" [a, b, c]
def vshufps (a b c d : Opd) : Item := .ins "vshufps" [a, b, c, d]
def vshufpd (a b c d : Opd) : Item := .ins "vshufpd" [a, b, c, d]
def vpshufd (a b c : Opd) : Item := .ins "vpshufd" [a, b, c]
def vextractf128 (a b c : Opd) : Item := .ins "vextractf128" [a, b, c]
def vbroadcastss (a b : Opd) : Item := .ins "vbroadcastss" [a, b]
def vbroadcastsd (a b : Opd) : Item := .ins "vbroadcastsd" [a, b]
def vmovss (a b : Opd) : Item := .ins "vmovss" [a, b]
def vmovsd (a b : Opd) : Item := .ins "vmovsd" [a, b]
def vmovups (a b : Opd) : Item := .ins "vmovups" [a, b]
def vmovupd (a b : Opd) : Item := .ins "vmovupd" [a, b]
def vmovaps (a b : Opd) : Item := .ins "vmovaps" [a, b]
def vmovapd (a b : Opd) : Item := .ins "vmovapd" [a, b]
def vmovdqu (a b : Opd) : Item := .ins "vmovdqu" [a, b]
def vmovdqa (a b : Opd) : Item := .ins "vmovdqa" [a, b]
def vmovntdq (a b : Opd) : Item := .ins "vmovntdq" [a, b]
def vmovntps (a b : Opd) : Item := .ins "vmovntps" [a, b]
def vmovd (a b : Opd) : Item := .ins "vmovd" [a, b]
def vmovq (a b : Opd) : Item := .ins "vmovq" [a, b]

-- ---------------------------------------------------------------------------
-- Intel syntax, as GNU `as` reads it
-- ---------------------------------------------------------------------------

def Reg.name (r : Reg) : String :=
  let q := #["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi"]
  let d := #["eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi"]
  let b := #["al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil"]
  if r.n < 8 then
    match r.w with | .q => q[r.n]! | .d => d[r.n]! | .b => b[r.n]!
  else
    s!"r{r.n}" ++ match r.w with | .q => "" | .d => "d" | .b => "b"

def VReg.name (v : VReg) : String := s!"{if v.ymm then "y" else "x"}mm{v.n}"

def hex (x : Nat) : String :=
  "0x" ++ String.ofList (Nat.toDigits 16 x)

def signedHex (x : Int) : String :=
  if x < 0 then "-" ++ hex x.natAbs else hex x.toNat

/-- `pfx` is put before every label, so bodies can share one file. -/
def Addr.intel (pfx : String) (a : Addr) : String :=
  let base := match a.base with
    | .reg r => r.name
    | .rip l => s!"rip+{pfx}{l}"
  let index := match a.index with
    | some (r, s) => s!"+{r.name}*{s}"
    | none => ""
  let disp := if a.disp == 0 then "" else if a.disp < 0 then signedHex a.disp else "+" ++ hex a.disp.toNat
  s!"[{base}{index}{disp}]"

def Mem.intel (pfx : String) (m : Mem) : String :=
  let size := match m.size with
    | 1 => "BYTE PTR " | 2 => "WORD PTR " | 4 => "DWORD PTR " | 8 => "QWORD PTR "
    | 16 => "XMMWORD PTR " | 32 => "YMMWORD PTR " | _ => ""
  size ++ m.addr.intel pfx

def Opd.intel (pfx : String) : Opd → String
  | .r x => x.name
  | .v x => x.name
  | .m x => x.intel pfx
  | .i x => signedHex x

def Item.intel (pfx : String) : Item → String
  | .ins mn [] => s!"    {mn}"
  | .ins mn ops => s!"    {mn} " ++ ",".intercalate (ops.map (Opd.intel pfx))
  | .jmp l => s!"    jmp {pfx}{l}"
  | .jcc c l => s!"    j{c.name} {pfx}{l}"
  | .label l => s!"{pfx}{l}:"
  | .align k => s!"    .p2align {k}"
  | .data [] => ""
  | .data bs => "    .byte " ++ ", ".intercalate (bs.map fun b => hex b.toNat)

/-- The body as GNU `as` source, one item a line, labels prefixed with `pfx`. -/
def Body.intel (pfx : String) (b : Body) : String :=
  "\n".intercalate (b.map (Item.intel pfx))

-- ---------------------------------------------------------------------------
-- Encoding one instruction
-- ---------------------------------------------------------------------------

/-- An instruction's bytes, and where in them a RIP-relative `disp32` goes:
    its offset, the label, and what is added to the label's address. -/
structure Enc where
  bytes : List UInt8
  rip : Option (Nat × String × Int) := none
  deriving Inhabited

/-- The ModRM byte and what follows it (SIB, displacement) for one operand,
    with the REX bits that operand needs. -/
structure RM where
  x : Bool := false
  b : Bool := false
  /-- The operand is `spl`/`bpl`/`sil`/`dil`, which need a REX prefix. -/
  lowByte : Bool := false
  bytes : List UInt8
  rip : Option (Nat × String × Int) := none

def fitsI8 (d : Int) : Bool := -128 ≤ d && d ≤ 127
def fitsI32 (d : Int) : Bool := -2147483648 ≤ d && d ≤ 2147483647

/-- `x` as `n` little-endian bytes, two's complement. -/
def leInt (n : Nat) (x : Int) : List UInt8 := le n (x % (2 ^ (8 * n) : Int)).toNat

/-- `x` read as a signed value of `bits` bits. -/
def signed (bits : Nat) (x : Int) : Int :=
  let v := x % (2 ^ bits : Int)
  if v ≥ 2 ^ (bits - 1) then v - 2 ^ bits else v

def modrm (md reg rm : Nat) : UInt8 := UInt8.ofNat (md * 64 + (reg % 8) * 8 + rm % 8)

def isLowByte (r : Reg) : Bool := r.w == .b && 4 ≤ r.n && r.n < 8

def rmReg (reg : Nat) (r : Reg) : RM :=
  { b := r.n ≥ 8, lowByte := isLowByte r, bytes := [modrm 3 reg r.n] }

def rmVReg (reg : Nat) (v : VReg) : RM :=
  { b := v.n ≥ 8, bytes := [modrm 3 reg v.n] }

def rmMem (reg : Nat) (a : Addr) : Except String RM := do
  match a.base with
  | .rip l =>
    if a.index.isSome then throw "a RIP-relative address cannot have an index"
    return { bytes := [modrm 0 reg 5] ++ le 4 0, rip := some (1, l, a.disp) }
  | .reg base =>
    if base.w != .q then throw s!"address base {base.name} is not a 64-bit register"
    let md := if a.disp == 0 && base.n % 8 != 5 then 0 else if fitsI8 a.disp then 1 else 2
    unless fitsI32 a.disp do throw s!"displacement {a.disp} does not fit 32 bits"
    let disp := match md with | 0 => [] | 1 => leInt 1 a.disp | _ => leInt 4 a.disp
    match a.index with
    | none =>
      if base.n % 8 == 4 then
        return { b := base.n ≥ 8, bytes := [modrm md reg 4, 0x24] ++ disp }
      return { b := base.n ≥ 8, bytes := [modrm md reg base.n] ++ disp }
    | some (ix, s) =>
      if ix.w != .q then throw s!"address index {ix.name} is not a 64-bit register"
      if ix.n == 4 then throw "rsp cannot be an index"
      let ss ← match s with
        | 1 => pure 0 | 2 => pure 1 | 4 => pure 2 | 8 => pure 3
        | _ => throw s!"scale {s} is not 1, 2, 4 or 8"
      return { x := ix.n ≥ 8, b := base.n ≥ 8,
               bytes := [modrm md reg 4, UInt8.ofNat (ss * 64 + (ix.n % 8) * 8 + base.n % 8)] ++ disp }

/-- A legacy-encoded instruction: REX if anything needs it, the opcode, then the
    ModRM operand and an immediate. `reg` is the ModRM reg field; `regByte`
    says it names `spl`/`bpl`/`sil`/`dil`. -/
def legacy (w : Bool) (opc : List UInt8) (reg : Nat) (rm : RM)
    (imm : List UInt8 := []) (regByte : Bool := false) : Enc :=
  let rex := if w || reg ≥ 8 || rm.x || rm.b || rm.lowByte || regByte then
    [UInt8.ofNat (0x40 + (if w then 8 else 0) + (if reg ≥ 8 then 4 else 0) +
      (if rm.x then 2 else 0) + (if rm.b then 1 else 0))] else []
  let pre := rex ++ opc
  { bytes := pre ++ rm.bytes ++ imm, rip := rm.rip.map fun (o, l, d) => (pre.length + o, l, d) }

/-- A VEX-encoded instruction. `pp`: 0 none, 1 `66`, 2 `F3`, 3 `F2`; `map`: 1 `0F`,
    2 `0F38`, 3 `0F3A`. The 2-byte prefix whenever it can say it, as `as` does. -/
def vex (pp map : Nat) (w l : Bool) (opc : UInt8) (reg vvvv : Nat) (rm : RM)
    (imm : List UInt8 := []) : Enc :=
  let r := if reg ≥ 8 then 0 else 0x80
  let tail := (15 - vvvv) * 8 + (if l then 4 else 0) + pp
  let pre := if !rm.x && !rm.b && !w && map == 1 then
      [0xC5, UInt8.ofNat (r + tail)]
    else
      [0xC4, UInt8.ofNat (r + (if rm.x then 0 else 0x40) + (if rm.b then 0 else 0x20) + map),
       UInt8.ofNat ((if w then 0x80 else 0) + tail)]
  let pre := pre ++ [opc]
  { bytes := pre ++ rm.bytes ++ imm, rip := rm.rip.map fun (o, l, d) => (pre.length + o, l, d) }

def aluOp : String → Option Nat
  | "add" => some 0 | "or" => some 1 | "adc" => some 2 | "sbb" => some 3
  | "and" => some 4 | "sub" => some 5 | "xor" => some 6 | "cmp" => some 7
  | _ => none

def shiftOp : String → Option Nat
  | "shl" => some 4 | "shr" => some 5 | "sar" => some 7
  | _ => none

def Width.bits : Width → Nat | .b => 8 | .d => 32 | .q => 64

/-- An immediate for an instruction of width `w`, as the signed value the
    instruction sign-extends, refusing one that does not fit. -/
def immFor (w : Width) (x : Int) : Except String Int := do
  unless -(2 ^ (w.bits - 1) : Int) ≤ x && x < 2 ^ w.bits do
    throw s!"immediate {signedHex x} does not fit {w.bits} bits"
  let s := signed w.bits x
  if w == .q && !fitsI32 s then throw s!"immediate {signedHex x} does not fit 32 bits"
  return s

/-- `(pp, opcode)` of the three-operand AVX arithmetic, and whether its width
    comes from the registers (packed) or is one element (scalar). -/
def vexArith : String → Option (Nat × UInt8 × Bool)
  | "vaddss" => some (2, 0x58, false) | "vaddsd" => some (3, 0x58, false)
  | "vaddps" => some (0, 0x58, true) | "vaddpd" => some (1, 0x58, true)
  | "vsubss" => some (2, 0x5C, false) | "vsubsd" => some (3, 0x5C, false)
  | "vsubps" => some (0, 0x5C, true) | "vsubpd" => some (1, 0x5C, true)
  | "vmulss" => some (2, 0x59, false) | "vmulsd" => some (3, 0x59, false)
  | "vmulps" => some (0, 0x59, true) | "vmulpd" => some (1, 0x59, true)
  | "vdivss" => some (2, 0x5E, false) | "vdivsd" => some (3, 0x5E, false)
  | "vdivps" => some (0, 0x5E, true) | "vdivpd" => some (1, 0x5E, true)
  | "vandps" => some (0, 0x54, true) | "vandpd" => some (1, 0x54, true)
  | "vorps" => some (0, 0x56, true) | "vorpd" => some (1, 0x56, true)
  | "vxorps" => some (0, 0x57, true) | "vxorpd" => some (1, 0x57, true)
  | "vpand" => some (1, 0xDB, true) | "vpor" => some (1, 0xEB, true)
  | "vpxor" => some (1, 0xEF, true)
  | "vpaddd" => some (1, 0xFE, true) | "vpaddq" => some (1, 0xD4, true)
  | "vpsubd" => some (1, 0xFA, true) | "vpsubq" => some (1, 0xFB, true)
  | _ => none

/-- `(pp, load opcode, store opcode)` of the moves between a vector register and
    memory, and whether their width comes from the register (packed) or is one
    element (scalar). -/
def vexMove : String → Option (Nat × UInt8 × UInt8 × Bool)
  | "vmovups" => some (0, 0x10, 0x11, true) | "vmovupd" => some (1, 0x10, 0x11, true)
  | "vmovaps" => some (0, 0x28, 0x29, true) | "vmovapd" => some (1, 0x28, 0x29, true)
  | "vmovdqu" => some (2, 0x6F, 0x7F, true) | "vmovdqa" => some (1, 0x6F, 0x7F, true)
  | "vmovss" => some (2, 0x10, 0x11, false) | "vmovsd" => some (3, 0x10, 0x11, false)
  | _ => none

def Width.bytes : Width → Nat | .b => 1 | .d => 4 | .q => 8

/-- A memory operand the source gives a size must be the size the instruction
    accesses: otherwise the bytes would say something other than the source
    does, and `as` would refuse the same line. A size of 0 states nothing. -/
def sized (n : Nat) : Opd → Except String Unit
  | .m m =>
    if m.size == 0 || m.size == n then pure ()
    else throw s!"{(Opd.m m).intel ""} where the instruction accesses {n} bytes"
  | _ => pure ()

def rmOf (reg : Nat) : Opd → Except String RM
  | .r x => pure (rmReg reg x)
  | .v x => pure (rmVReg reg x)
  | .m x => rmMem reg x.addr
  | .i _ => throw "an immediate where a register or memory operand goes"

/-- One instruction's bytes. -/
def encode (mn : String) (ops : List Opd) : Except String Enc := do
  let byteOp (w : Width) (opc : Nat) : List UInt8 := [UInt8.ofNat (if w == .b then opc - 1 else opc)]
  if let some op := aluOp mn then
    match ops with
    | [.r d, .r s] =>
      if d.w != s.w then throw "operand widths differ"
      return legacy (d.w == .q) (byteOp d.w (op * 8 + 1)) s.n (rmReg s.n d) (regByte := isLowByte s)
    | [.r d, .m m] =>
      sized d.w.bytes (.m m)
      return legacy (d.w == .q) (byteOp d.w (op * 8 + 3)) d.n (← rmMem d.n m.addr) (regByte := isLowByte d)
    | [.m m, .r s] =>
      sized s.w.bytes (.m m)
      return legacy (s.w == .q) (byteOp s.w (op * 8 + 1)) s.n (← rmMem s.n m.addr) (regByte := isLowByte s)
    | [.r d, .i x] =>
      let v ← immFor d.w x
      if d.w == .b then
        if d.n == 0 then return { bytes := [UInt8.ofNat (op * 8 + 4)] ++ leInt 1 v }
        return legacy false [0x80] op (rmReg op d) (leInt 1 v)
      if fitsI8 v then return legacy (d.w == .q) [0x83] op (rmReg op d) (leInt 1 v)
      if d.n == 0 then return legacy (d.w == .q) [UInt8.ofNat (op * 8 + 5)] 0 { bytes := [] } (leInt 4 v)
      return legacy (d.w == .q) [0x81] op (rmReg op d) (leInt 4 v)
    | _ => throw "unsupported operands"
  if let some op := shiftOp mn then
    match ops with
    | [.r d, .i 1] => return legacy (d.w == .q) (byteOp d.w 0xD1) op (rmReg op d)
    | [.r d, .i x] =>
      unless 0 ≤ x && x < d.w.bits do throw s!"shift count {x} is not below {d.w.bits}"
      return legacy (d.w == .q) (byteOp d.w 0xC1) op (rmReg op d) (leInt 1 x)
    | _ => throw "unsupported operands"
  if let some (pp, opc, packed) := vexArith mn then
    match ops with
    | [.v d, .v s1, s2] =>
      let s2ymm := match s2 with | .v v => v.ymm | _ => d.ymm
      if packed && (d.ymm != s1.ymm || s2ymm != d.ymm) then throw "operand widths differ"
      if !packed && (d.ymm || s1.ymm || s2ymm) then throw "a scalar operation on a ymm register"
      sized (if packed then (if d.ymm then 32 else 16) else if pp == 3 then 8 else 4) s2
      return vex pp 1 false (packed && d.ymm) opc d.n s1.n (← rmOf d.n s2)
    | _ => throw "unsupported operands"
  if let some (pp, ld, st, packed) := vexMove mn then
    match ops with
    | [.v d, .m m] =>
      if !packed && d.ymm then throw "a scalar move on a ymm register"
      sized (if packed then (if d.ymm then 32 else 16) else if pp == 3 then 8 else 4) (.m m)
      return vex pp 1 false (packed && d.ymm) ld d.n 0 (← rmMem d.n m.addr)
    | [.m m, .v s] =>
      if !packed && s.ymm then throw "a scalar move on a ymm register"
      sized (if packed then (if s.ymm then 32 else 16) else if pp == 3 then 8 else 4) (.m m)
      return vex pp 1 false (packed && s.ymm) st s.n 0 (← rmMem s.n m.addr)
    | _ => throw "unsupported operands"
  match mn, ops with
  | "ret", [] => return { bytes := [0xC3] }
  | "sfence", [] => return { bytes := [0x0F, 0xAE, 0xF8] }
  | "lfence", [] => return { bytes := [0x0F, 0xAE, 0xE8] }
  | "mfence", [] => return { bytes := [0x0F, 0xAE, 0xF0] }
  | "vzeroupper", [] => return { bytes := [0xC5, 0xF8, 0x77] }
  | "mov", [.r d, .r s] =>
    if d.w != s.w then throw "operand widths differ"
    return legacy (d.w == .q) (byteOp d.w 0x89) s.n (rmReg s.n d) (regByte := isLowByte s)
  | "mov", [.r d, .m m] =>
    sized d.w.bytes (.m m)
    return legacy (d.w == .q) (byteOp d.w 0x8B) d.n (← rmMem d.n m.addr) (regByte := isLowByte d)
  | "mov", [.m m, .r s] =>
    sized s.w.bytes (.m m)
    return legacy (s.w == .q) (byteOp s.w 0x89) s.n (← rmMem s.n m.addr) (regByte := isLowByte s)
  | "mov", [.r d, .i x] =>
    match d.w with
    | .q => return legacy true [0xC7] 0 (rmReg 0 d) (leInt 4 (← immFor .q x))
    | .d =>
      let v ← immFor .d x
      return legacy false [UInt8.ofNat (0xB8 + d.n % 8)] 0 { b := d.n ≥ 8, bytes := [] } (leInt 4 v)
    | .b =>
      let v ← immFor .b x
      let rm : RM := { b := d.n ≥ 8, lowByte := isLowByte d, bytes := [] }
      return legacy false [UInt8.ofNat (0xB0 + d.n % 8)] 0 rm (leInt 1 v)
  | "movabs", [.r d, .i x] =>
    if d.w != .q then throw "movabs takes a 64-bit register"
    return legacy true [UInt8.ofNat (0xB8 + d.n % 8)] 0 { b := d.n ≥ 8, bytes := [] } (leInt 8 x)
  | "lea", [.r d, .m m] =>
    if d.w == .b then throw "lea into a byte register"
    return legacy (d.w == .q) [0x8D] d.n (← rmMem d.n m.addr)
  | "test", [.r d, .r s] =>
    if d.w != s.w then throw "operand widths differ"
    return legacy (d.w == .q) (byteOp d.w 0x85) s.n (rmReg s.n d) (regByte := isLowByte s)
  | "test", [.m m, .r s] =>
    sized s.w.bytes (.m m)
    return legacy (s.w == .q) (byteOp s.w 0x85) s.n (← rmMem s.n m.addr) (regByte := isLowByte s)
  | "test", [.r d, .i x] =>
    let v ← immFor d.w x
    let imm := leInt (if d.w == .b then 1 else 4) v
    if d.n == 0 then return legacy (d.w == .q) (byteOp d.w 0xA9) 0 { bytes := [] } imm
    return legacy (d.w == .q) (byteOp d.w 0xF7) 0 (rmReg 0 d) imm
  | "inc", [.r d] => return legacy (d.w == .q) (byteOp d.w 0xFF) 0 (rmReg 0 d)
  | "dec", [.r d] => return legacy (d.w == .q) (byteOp d.w 0xFF) 1 (rmReg 1 d)
  | "vshufps", [.v d, .v s1, s2, .i x] =>
    sized (if d.ymm then 32 else 16) s2
    return vex 0 1 false d.ymm 0xC6 d.n s1.n (← rmOf d.n s2) (leInt 1 x)
  | "vshufpd", [.v d, .v s1, s2, .i x] =>
    sized (if d.ymm then 32 else 16) s2
    return vex 1 1 false d.ymm 0xC6 d.n s1.n (← rmOf d.n s2) (leInt 1 x)
  | "vpshufd", [.v d, s, .i x] =>
    sized (if d.ymm then 32 else 16) s
    return vex 1 1 false d.ymm 0x70 d.n 0 (← rmOf d.n s) (leInt 1 x)
  | "vextractf128", [d, .v s, .i x] =>
    if !s.ymm then throw "vextractf128 takes a ymm source"
    match d with
    | .v v => if v.ymm then throw "vextractf128 writes an xmm register"
    | .m _ => pure ()
    | _ => throw "unsupported operands"
    sized 16 d
    return vex 1 3 false true 0x19 s.n 0 (← rmOf s.n d) (leInt 1 x)
  | "vbroadcastss", [.v d, .m m] =>
    sized 4 (.m m)
    return vex 1 2 false d.ymm 0x18 d.n 0 (← rmMem d.n m.addr)
  | "vbroadcastsd", [.v d, .m m] =>
    if !d.ymm then throw "vbroadcastsd writes a ymm register"
    sized 8 (.m m)
    return vex 1 2 false true 0x19 d.n 0 (← rmMem d.n m.addr)
  | "vmovntdq", [.m m, .v s] =>
    sized (if s.ymm then 32 else 16) (.m m)
    return vex 1 1 false s.ymm 0xE7 s.n 0 (← rmMem s.n m.addr)
  | "vmovntps", [.m m, .v s] =>
    sized (if s.ymm then 32 else 16) (.m m)
    return vex 0 1 false s.ymm 0x2B s.n 0 (← rmMem s.n m.addr)
  | "vmovd", [.r d, .v s] =>
    if d.w != .d || s.ymm then throw "vmovd moves between a 32-bit register and an xmm register"
    return vex 1 1 false false 0x7E s.n 0 (rmReg s.n d)
  | "vmovd", [.v d, .r s] =>
    if s.w != .d || d.ymm then throw "vmovd moves between a 32-bit register and an xmm register"
    return vex 1 1 false false 0x6E d.n 0 (rmReg d.n s)
  | "vmovq", [.r d, .v s] =>
    if d.w != .q || s.ymm then throw "vmovq moves between a 64-bit register and an xmm register"
    return vex 1 1 true false 0x7E s.n 0 (rmReg s.n d)
  | "vmovq", [.v d, .r s] =>
    if s.w != .q || d.ymm then throw "vmovq moves between a 64-bit register and an xmm register"
    return vex 1 1 true false 0x6E d.n 0 (rmReg d.n s)
  | _, _ => throw "not an instruction this assembler encodes"

-- ---------------------------------------------------------------------------
-- Layout
-- ---------------------------------------------------------------------------

/-- `as`'s NOPs for x86-64, by length. -/
def nopTable : Array (List UInt8) := #[
  [],
  [0x90],
  [0x66, 0x90],
  [0x0F, 0x1F, 0x00],
  [0x0F, 0x1F, 0x40, 0x00],
  [0x0F, 0x1F, 0x44, 0x00, 0x00],
  [0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00],
  [0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00],
  [0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
  [0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
  [0x66, 0x2E, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
  [0x66, 0x66, 0x2E, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00]]

/-- `n` bytes of padding as `as` writes it: the longest NOP as often as it fits,
    then one NOP for the rest. After literal bytes rather than an instruction,
    `as` puts a one-byte NOP first. -/
def nops (afterData : Bool) (n : Nat) : List UInt8 :=
  let fill (k : Nat) := (List.replicate (k / 11) nopTable[11]!).flatten ++ nopTable[k % 11]!
  if n == 0 then [] else if afterData then 0x90 :: fill (n - 1) else fill n

def padTo (off log2 : Nat) : Nat := (2 ^ log2 - off % 2 ^ log2) % 2 ^ log2

/-- Where every item starts, given which branches are long, and where each
    label is. -/
def layout (items : Array Item) (encs : Array Enc) (long : Array Bool) :
    Array Nat × List (String × Nat) := Id.run do
  let mut off := 0
  let mut offs := #[]
  let mut labels := []
  for k in [0:items.size] do
    offs := offs.push off
    match items[k]! with
    | .ins _ _ => off := off + encs[k]!.bytes.length
    | .jmp _ => off := off + if long[k]! then 5 else 2
    | .jcc _ _ => off := off + if long[k]! then 6 else 2
    | .label l => labels := (l, off) :: labels
    | .align n => off := off + padTo off n
    | .data bs => off := off + bs.length
  return (offs.push off, labels)

/-- The body's bytes, or what is wrong with it: an instruction outside the
    subset, an unknown or repeated label, an alignment beyond 64 bytes. -/
def assemble (body : Body) : Except String (List UInt8) := do
  let items := body.toArray
  let mut encs : Array Enc := #[]
  let mut seen : List String := []
  for it in items do
    match it with
    | .ins mn ops =>
      match encode mn ops with
      | .ok e => encs := encs.push e
      | .error msg => throw s!"`{(it.intel "").trimLeft}`: {msg}"
    | .label l =>
      if seen.contains l then throw s!"label {l} is defined twice"
      seen := l :: seen
      encs := encs.push default
    | .align n =>
      if n > 6 then throw s!".p2align {n}: a body aligns to at most 64 bytes"
      encs := encs.push default
    | _ => encs := encs.push default
  let target (labels : List (String × Nat)) (l : String) : Except String Nat :=
    match labels.lookup l with
    | some t => pure t
    | none => throw s!"label {l} is not defined in this body"
  -- Every branch starts short and becomes long when its displacement does not
  -- fit; lengthening only moves targets further, so this settles.
  let mut long := Array.replicate items.size false
  for _ in [0:items.size + 1] do
    let (offs, labels) := layout items encs long
    let mut changed := false
    for k in [0:items.size] do
      match items[k]! with
      | .jmp l | .jcc _ l =>
        if !long[k]! then
          let t ← target labels l
          unless fitsI8 ((t : Int) - (offs[k]! + 2)) do
            long := long.set! k true
            changed := true
      | _ => pure ()
    if !changed then break
  let (offs, labels) := layout items encs long
  let mut out : Array UInt8 := #[]
  let mut afterData := false
  for k in [0:items.size] do
    let here := offs[k]!
    let next := offs[k + 1]!
    match items[k]! with
    | .ins _ _ =>
      let e := encs[k]!
      let mut bs := e.bytes
      if let some (o, l, d) := e.rip then
        let t ← target labels l
        let disp := (t : Int) + d - next
        bs := bs.take o ++ leInt 4 disp ++ bs.drop (o + 4)
      out := out ++ bs.toArray
      afterData := false
    | .jmp l =>
      let t ← target labels l
      let disp := (t : Int) - next
      out := out ++ (if long[k]! then 0xE9 :: leInt 4 disp else [0xEB] ++ leInt 1 disp).toArray
      afterData := false
    | .jcc c l =>
      let t ← target labels l
      let disp := (t : Int) - next
      out := out ++ (if long[k]! then [0x0F, UInt8.ofNat (0x80 + c.code)] ++ leInt 4 disp
        else [UInt8.ofNat (0x70 + c.code)] ++ leInt 1 disp).toArray
      afterData := false
    | .label _ => pure ()
    | .align _ => out := out ++ (nops afterData (next - here)).toArray
    | .data bs =>
      out := out ++ bs.toArray
      afterData := afterData || !bs.isEmpty
  return out.toList

end AlgorithmLib.X86
