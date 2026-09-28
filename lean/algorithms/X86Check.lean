import AlgorithmLib.X86
import CpuBenchAsm

/-! # `AlgorithmLib.X86` against GNU `as`

    cd lean && lake env lean --run algorithms/X86Check.lean

Every case is printed with `Body.intel`, assembled by `as`, and must match
`assemble` byte for byte. -/

open AlgorithmLib.X86

/-- A case: a body that must assemble to what `as` makes of it. -/
abbrev Case := String × Body

def q64 : List Reg := [rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi, r8, r9, r10, r11, r12, r13, r14, r15]
/-- Registers that reach every special case: 4 and 5 (SIB, displacement), 8 and
    up (the extension bits), 12 and 13 (the same cases extended). -/
def pick : List Nat := [0, 1, 3, 4, 5, 7, 8, 11, 12, 13, 15]
def regW (w : Width) (n : Nat) : Reg := ⟨n, w⟩
def xs (ns : List Nat) : List VReg := ns.map (⟨·, false⟩)
def ys (ns : List Nat) : List VReg := ns.map (⟨·, true⟩)
def vpick : List Nat := [0, 1, 7, 8, 9, 15]
def bytesRegs : List Reg := [al, cl, dl, bl, spl, bpl, sil, dil, r8b, r12b, r13b, r15b]

/-- Every addressing shape: each base, with and without an index at each scale,
    at displacements either side of the disp8 boundary. -/
def allAddrs : List Addr := Id.run do
  let mut out := []
  for b in q64 do
    for d in ([0, 8, -8, 127, 128, -128, -129, 0x1000, -0x60] : List Int) do
      out := { base := .reg b, disp := d } :: out
      for i in [rax, rcx, rbp, r8, r12, r13, r15] do
        for s in [1, 2, 4, 8] do
          out := { base := .reg b, index := some (i, s), disp := d } :: out
  return out.reverse

/-- A few addresses, for forms whose addressing is the same code as above. -/
def someAddrs : List Addr :=
  [rax, rsp, rbp, r12, r13, r15 + 0x8, rdx + rsi*8 - 0x8, rax + rdx*8 + 0x8, r8 + r12*4 + 0x1000,
   rbp + r13*2, rsp + r15*1 - 0x60]

def imm64 : List Int := [0, 1, 0x7f, 0x80, -1, -128, -129, 0x7fffffff, -0x80000000,
  0xffffffffffffffe0, 0xffffffffffffff80]
def imm32 : List Int := [0, 1, 0x7f, 0x80, 0xffffffff, 0x12345678, -1]
def imm8 : List Int := [0, 1, 0x7f, 0x80, 0xff]

def one (it : Item) : Case := ((it.intel "").trimLeft, [it])

def integerCases : List Case := Id.run do
  let mut out := []
  for mn in ["add", "or", "adc", "sbb", "and", "sub", "xor", "cmp", "mov", "test"] do
    for d in q64 do
      for s in q64 do out := one (.ins mn [.r d, .r s]) :: out
    for d in pick do
      for s in pick do out := one (.ins mn [.r (regW .d d), .r (regW .d s)]) :: out
    for d in bytesRegs do
      for s in bytesRegs do out := one (.ins mn [.r d, .r s]) :: out
    for d in pick do
      for x in imm64 do
        if mn != "mov" || fitsI32 (signed 64 x) then out := one (.ins mn [.r (regW .q d), .i x]) :: out
      for x in imm32 do out := one (.ins mn [.r (regW .d d), .i x]) :: out
    for d in bytesRegs do
      for x in imm8 do out := one (.ins mn [.r d, .i x]) :: out
    for a in someAddrs do
      for r in [rax, rcx, r9, r12] do
        out := one (.ins mn [.m (qword a), .r r]) :: out
        if mn != "test" then out := one (.ins mn [.r r, .m (qword a)]) :: out
      for r in [eax, r13d] do
        out := one (.ins mn [.m (dword a), .r r]) :: out
        if mn != "test" then out := one (.ins mn [.r r, .m (dword a)]) :: out
      for r in [cl, sil, r8b] do
        out := one (.ins mn [.m (byte a), .r r]) :: out
        if mn != "test" then out := one (.ins mn [.r r, .m (byte a)]) :: out
  for a in allAddrs do
    out := one (mov rax (qword a)) :: one (mov (qword a) r11) :: one (lea r12 (mem a)) :: out
  for a in someAddrs do out := one (lea eax (mem a)) :: out
  for d in q64 do
    out := one (movabs d 0x8000000000000000) :: one (movabs d 0xffffffffffffffe) :: out
  for d in pick do
    for w in [Width.q, .d] do
      out := one (inc (regW w d)) :: one (dec (regW w d)) :: out
      for mn in ["shl", "shr", "sar"] do
        for x in ([1, 2, 31] : List Int) do out := one (.ins mn [.r (regW w d), .i x]) :: out
  for d in bytesRegs do
    out := one (inc d) :: one (dec d) :: one (shr d 1) :: one (shl d 3) :: out
  out := one ret :: one sfence :: one lfence :: one mfence :: one vzeroupper :: out
  return out.reverse

def vexArithNames : List String :=
  ["vaddss", "vaddsd", "vaddps", "vaddpd", "vsubss", "vsubsd", "vsubps", "vsubpd",
   "vmulss", "vmulsd", "vmulps", "vmulpd", "vdivss", "vdivsd", "vdivps", "vdivpd",
   "vandps", "vandpd", "vorps", "vorpd", "vxorps", "vxorpd", "vpand", "vpor", "vpxor",
   "vpaddd", "vpaddq", "vpsubd", "vpsubq"]

def vexMoveNames : List String :=
  ["vmovups", "vmovupd", "vmovaps", "vmovapd", "vmovdqu", "vmovdqa", "vmovss", "vmovsd"]

def vectorCases : List Case := Id.run do
  let mut out := []
  for mn in vexArithNames do
    let packed := (vexArith mn).map (·.2.2) |>.getD false
    let size := if !packed then (if mn.endsWith "sd" then 8 else 4) else 16
    for wide in (if packed then [false, true] else [false]) do
      let regs := if wide then ys vpick else xs vpick
      for d in regs do
        for s1 in regs do
          for s2 in regs do out := one (.ins mn [.v d, .v s1, .v s2]) :: out
          for a in someAddrs do
            out := one (.ins mn [.v d, .v s1, .m ⟨if wide then 32 else size, a⟩]) :: out
  for mn in vexMoveNames do
    let packed := (vexMove mn).map (·.2.2.2) |>.getD false
    let size := if !packed then (if mn.endsWith "sd" then 8 else 4) else 16
    for wide in (if packed then [false, true] else [false]) do
      for v in (if wide then ys vpick else xs vpick) do
        for a in someAddrs do
          let m : Mem := ⟨if wide then 32 else size, a⟩
          out := one (.ins mn [.v v, .m m]) :: one (.ins mn [.m m, .v v]) :: out
  for wide in [false, true] do
    let regs := if wide then ys vpick else xs vpick
    let size := if wide then 32 else 16
    for d in regs do
      for s in regs do
        out := one (vshufps d s s 0xee) :: one (vshufpd d d s 0x1) :: one (vpshufd d s 0x55) :: out
      for a in someAddrs do
        out := one (vshufps d d (Mem.mk size a) 0x44) :: one (vpshufd d (Mem.mk size a) 0xee) ::
          one (vbroadcastss d (dword a)) :: one (vmovntdq (Mem.mk size a) d) ::
          one (vmovntps (Mem.mk size a) d) :: out
        if wide then out := one (vbroadcastsd d (qword a)) :: out
  for s in ys vpick do
    for d in xs vpick do out := one (vextractf128 d s 0x1) :: out
    for a in someAddrs do out := one (vextractf128 (xmmword a) s 0x0) :: out
  for v in xs vpick do
    for r in pick do
      out := one (vmovd (regW .d r) v) :: one (vmovd v (regW .d r)) ::
        one (vmovq (regW .q r) v) :: one (vmovq v (regW .q r)) :: out
  return out.reverse

def conds : List Cond := [.o, .no, .b, .ae, .e, .ne, .be, .a, .s, .ns, .p, .np, .l, .ge, .le, .g]

def branch (c : Option Cond) (l : String) : Item :=
  match c with | some c => .jcc c l | none => .jmp l

def filler (n : Nat) : Item := .data (List.replicate n 0xcc)

/-- Branches either side of the short/long boundary, both ways; branches whose
    lengthening pushes another over; branches across alignment. -/
def controlCases : List Case := Id.run do
  let mut out := []
  for c in none :: conds.map some do
    for n in [0, 1, 120, 123, 124, 125, 126, 127, 128, 129, 130, 1000] do
      out := (s!"forward {n}", [branch c "t", filler n, lbl "t", ret]) :: out
      out := (s!"backward {n}", [lbl "t", filler n, branch c "t", ret]) :: out
  for n in [100, 110, 118, 119, 120, 121, 122, 123, 124, 125, 126] do
    -- The first reaches past the second; the second's size decides the first's.
    out := (s!"chain {n}", [je "a", jmp "b", filler n, lbl "a", ret, filler 3, lbl "b", ret]) :: out
    out := (s!"aligned {n}", [je "a", filler n, p2align 4, lbl "a", ret, p2align 6, jne "a"]) :: out
  for n in [0, 1, 2, 100, 200] do
    out := (s!"rip {n}", [vmovsd xmm0 (qword (rip "c")), filler n,
      vextractf128 (xmmword (rip "c")) ymm9 0x1, vpshufd xmm12 (xmmword (rip "c")) 0x1b,
      vaddsd xmm1 xmm2 (qword (rip "c" + 8)), mov rax (qword (rip "c")),
      p2align 4, lbl "c", f64 1.5, f64 (-2.0), vmovss xmm3 (dword (rip "c"))]) :: out
  return out.reverse

/-- Every pad length at every alignment, after an instruction, after data, and
    after data then a label and another alignment. -/
def alignCases : List Case := Id.run do
  let mut out := []
  for k in [1:7] do
    for n in [0:64] do
      out := (s!"after ret×{n}, p2align {k}", List.replicate n ret ++ [p2align k, ret]) :: out
      out := (s!"after {n} bytes, p2align {k}", [filler n, p2align k, ret]) :: out
      out := (s!"after {n} bytes, label, p2align {k}",
        [filler n, lbl "x", p2align 2, p2align k, lbl "y", inc rax]) :: out
  return out.reverse

def shippedCases : List Case :=
  [("CpuBenchAsm.poly", CpuBenchAsm.poly), ("CpuBenchAsm.stream", CpuBenchAsm.stream)]

/-- Bodies `assemble` must refuse, and why. -/
def refused : List (String × Body) :=
  [("an undefined label", [jmp "nowhere"]),
   ("a label defined twice", [lbl "a", lbl "a"]),
   ("an alignment past 64 bytes", [p2align 7]),
   ("an immediate past 32 bits", [mov rax 0x8000000000000000]),
   ("an immediate past a 32-bit register", [mov eax 0x100000000]),
   ("an immediate past a byte register", [add cl 0x100]),
   ("a shift past the width", [shl eax 32]),
   ("rsp as an index", [mov rax (qword (rax + rsp*2))]),
   ("a scale of 3", [mov rax (qword (rax + rcx*3))]),
   ("a 32-bit base", [mov rax (qword (Addr.mk (.reg eax) none 0))]),
   ("mixed widths", [add rax ecx]),
   ("a scalar op on ymm", [vaddsd ymm0 ymm1 ymm2]),
   ("a 32-bit operand for a 64-bit load", [mov rax (dword (rdi + 8))]),
   ("a 64-bit operand for a 32-bit store", [add (qword rdi) ecx]),
   ("a 128-bit operand for a 256-bit load", [vmovups ymm0 (xmmword rax)]),
   ("a 64-bit operand for a scalar float", [vaddss xmm0 xmm1 (qword rax)]),
   ("a 64-bit operand for a broadcast float", [vbroadcastss ymm1 (qword (rip "c")), lbl "c"]),
   ("an instruction it does not encode", [.ins "cpuid" []])]

def hexOf (bs : List UInt8) : String :=
  String.join (bs.map fun b =>
    let s := String.ofList (Nat.toDigits 16 b.toNat)
    if s.length == 1 then "0" ++ s else s)

def run (cmd : String) (args : Array String) : IO String := do
  let r ← IO.Process.output { cmd, args }
  if r.exitCode != 0 then throw (IO.userError s!"{cmd} failed:\n{r.stderr}")
  return r.stdout

def main : IO UInt32 := do
  let cases := (integerCases ++ vectorCases ++ controlCases ++ alignCases ++ shippedCases).toArray
  let mut src := #[".intel_syntax noprefix", ".text"]
  let mut want := #[]
  for k in [0:cases.size] do
    let (name, body) := cases[k]!
    match assemble body with
    | .error e =>
      IO.eprintln s!"case {k} ({name}) does not assemble: {e}"
      return 1
    | .ok bs => want := want.push bs
    src := src.push ".p2align 6" |>.push s!"x86c{k}_start:" |>.push (body.intel s!".Lc{k}_")
      |>.push s!"x86c{k}_end:"
  let dir := (← run "mktemp" #["-d"]).trimRight
  IO.FS.writeFile s!"{dir}/c.s" ("\n".intercalate src.toList ++ "\n")
  discard <| run "as" #["--64", "-o", s!"{dir}/c.o", s!"{dir}/c.s"]
  discard <| run "objcopy" #["-O", "binary", "--only-section=.text", s!"{dir}/c.o", s!"{dir}/c.bin"]
  let text ← IO.FS.readBinFile s!"{dir}/c.bin"
  let relocs ← run "readelf" #["-r", s!"{dir}/c.o"]
  let mut starts := Array.replicate cases.size 0
  let mut ends := Array.replicate cases.size 0
  for line in (← run "nm" #[s!"{dir}/c.o"]).splitOn "\n" do
    match line.splitOn " " with
    | [addr, _, sym] =>
      if sym.startsWith "x86c" then
        let body := (sym.drop 4).splitOn "_"
        let k := body[0]!.toNat!
        let a := addr.foldl (fun n c => n * 16 + (if c.isDigit then c.toNat - 48 else c.toNat - 87)) 0
        if body[1]! == "start" then starts := starts.set! k a else ends := ends.set! k a
    | _ => pure ()
  discard <| run "rm" #["-rf", dir]
  let mut bad := 0
  for k in [0:cases.size] do
    let got := (text.extract starts[k]! ends[k]!).toList
    if got != want[k]! then
      bad := bad + 1
      if bad ≤ 20 then
        IO.eprintln s!"case {k} ({cases[k]!.1}):\n{cases[k]!.2.intel ""}\n  as:       {hexOf got}\n  assemble: {hexOf want[k]!}"
  unless (relocs.splitOn "There are no relocations").length > 1 do
    IO.eprintln s!"as left relocations, so some case referred outside itself:\n{relocs}"
    bad := bad + 1
  for (why, body) in refused do
    if (assemble body).isOk then
      IO.eprintln s!"assemble accepted {why}:\n{body.intel ""}"
      bad := bad + 1
  if bad > 0 then
    IO.eprintln s!"{bad} of {cases.size} cases differ from as"
    return 1
  IO.println s!"{cases.size} cases assemble as GNU as assembles them; {refused.length} bodies refused"
  return 0
