module
public import AlgorithmLib.Vocab.X86
meta import AlgorithmLib.Vocab.X86
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-! The machine code `Memops` carries for its large sizes: base's own AVX2
    loops, written here and assembled by base's assembler (`X86.assemble`),
    for the 32-byte moves Cranelift's 128-bit vectors do not have. Each keeps
    the contract in `base/src/builtin/native.rs`: System V, its arguments in
    registers, no calls, only caller-saved registers. -/

namespace MemopsAsm

open AlgorithmLib.X86

/-- copy(src, dst, n), `n ≥ 128`, `src` and `dst` apart: the first 32 bytes,
    then from the first 32-byte boundary of `dst` past them, while a whole 128
    bytes is left before the last 128, 128 bytes a trip
    in four ymm registers (all four loaded before any is stored, every store
    aligned), then the last 128 bytes, overlapping what came before. -/
def copy : Body := [
   vmovdqu ymm0 (ymmword rdi),
   vmovdqu (ymmword rsi) ymm0,
   lea r8 (mem (rdi + rdx*1 - 0x80)),
   lea rax (mem (rsi + 0x20)),
   and rax (-0x20),
   sub rsi rdi,
   sub rax rsi,
   cmp rax r8,
   jae "1",
   p2align 4,
  lbl "0",
   vmovdqu ymm0 (ymmword rax),
   vmovdqu ymm1 (ymmword (rax + 0x20)),
   vmovdqu ymm2 (ymmword (rax + 0x40)),
   vmovdqu ymm3 (ymmword (rax + 0x60)),
   vmovdqu (ymmword (rax + rsi*1)) ymm0,
   vmovdqu (ymmword (rax + rsi*1 + 0x20)) ymm1,
   vmovdqu (ymmword (rax + rsi*1 + 0x40)) ymm2,
   vmovdqu (ymmword (rax + rsi*1 + 0x60)) ymm3,
   add rax 0x80,
   cmp rax r8,
   jb "0",
  lbl "1",
   vmovdqu ymm0 (ymmword r8),
   vmovdqu ymm1 (ymmword (r8 + 0x20)),
   vmovdqu ymm2 (ymmword (r8 + 0x40)),
   vmovdqu ymm3 (ymmword (r8 + 0x60)),
   vmovdqu (ymmword (r8 + rsi*1)) ymm0,
   vmovdqu (ymmword (r8 + rsi*1 + 0x20)) ymm1,
   vmovdqu (ymmword (r8 + rsi*1 + 0x40)) ymm2,
   vmovdqu (ymmword (r8 + rsi*1 + 0x60)) ymm3,
   vzeroupper,
   ret]

/-- fill(dst, n, c8), `n ≥ 128`, `c8` the byte in each of its 8: `c8` in all
    four quadwords of a ymm register, stored over the first 32 bytes, then
    from the first 32-byte boundary past them, while a whole 128 bytes is left
    before the last 128, 128 bytes a trip (every store
    aligned), then over the last 128, overlapping what came before. -/
def fill : Body := [
   vmovq xmm0 rdx,
   vpermq ymm0 ymm0 0,
   vmovdqu (ymmword rdi) ymm0,
   lea r8 (mem (rdi + rsi*1 - 0x80)),
   lea rax (mem (rdi + 0x20)),
   and rax (-0x20),
   cmp rax r8,
   jae "1",
   p2align 4,
  lbl "0",
   vmovdqu (ymmword rax) ymm0,
   vmovdqu (ymmword (rax + 0x20)) ymm0,
   vmovdqu (ymmword (rax + 0x40)) ymm0,
   vmovdqu (ymmword (rax + 0x60)) ymm0,
   add rax 0x80,
   cmp rax r8,
   jb "0",
  lbl "1",
   vmovdqu (ymmword r8) ymm0,
   vmovdqu (ymmword (r8 + 0x20)) ymm0,
   vmovdqu (ymmword (r8 + 0x40)) ymm0,
   vmovdqu (ymmword (r8 + 0x60)) ymm0,
   vzeroupper,
   ret]

/-- The 128 bytes of `a` and `b` (`rdi`, `rsi`) at offset `r9`, compared in
    four ymm registers: the equal lanes of each 32 in `ymm0` … `ymm3`, their
    and's mask in `eax`, all ones when the 128 agree. -/
def agree128 : Body := [
   vmovdqu ymm0 (ymmword (rdi + r9*1)),
   vmovdqu ymm1 (ymmword (rdi + r9*1 + 0x20)),
   vmovdqu ymm2 (ymmword (rdi + r9*1 + 0x40)),
   vmovdqu ymm3 (ymmword (rdi + r9*1 + 0x60)),
   vpcmpeqb ymm0 ymm0 (ymmword (rsi + r9*1)),
   vpcmpeqb ymm1 ymm1 (ymmword (rsi + r9*1 + 0x20)),
   vpcmpeqb ymm2 ymm2 (ymmword (rsi + r9*1 + 0x40)),
   vpcmpeqb ymm3 ymm3 (ymmword (rsi + r9*1 + 0x60)),
   vpand ymm4 ymm0 ymm1,
   vpand ymm5 ymm2 ymm3,
   vpand ymm4 ymm4 ymm5,
   vpmovmskb eax ymm4]

/-- The walk both comparisons share, `(a, b, n)` with `n ≥ 128`: the first
    128 bytes, then from the first 32-byte boundary of `a` past them (every
    load of `a` aligned), 128 bytes a trip while a whole 128 is left before
    the last 128, then the last 128, overlapping what came before (bytes
    already found equal, so the first difference is where it was). `0` is
    returned if every byte agrees, and `found` runs on the first 128 that do
    not, its offset in `r9` and `ymm0` … `ymm3` as `agree128` left them. -/
def walk (found : Body) : Body :=
  [ xor r9d r9d,
    lea r8 (mem (rdx - 0x80)) ] ++ agree128 ++
  [ cmp eax (-1),
    jne "found",
    lea r9 (mem (rdi + 0x80)),
    and r9 (-0x20),
    sub r9 rdi,
    p2align 4,
   lbl "0",
    cmp r9 r8,
    ja "last" ] ++ agree128 ++
  [ cmp eax (-1),
    jne "found",
    add r9 0x80,
    jmp "0",
   lbl "last",
    mov r9 r8 ] ++ agree128 ++
  [ cmp eax (-1),
    jne "found",
    xor eax eax,
    vzeroupper,
    ret,
   lbl "found" ] ++ found

/-- compare(a, b, n), `n ≥ 128`: `-1`, `0` or `1` as the first byte where
    `a` and `b` differ is below or above, unsigned (`Memops.compareBytes`).
    In the 128 that differ, the first 32 whose mask is not all ones, the
    first lane in it that differs, and that byte of each. -/
def compare : Body := walk ([0, 1, 2, 3].flatMap fun k =>
  [ vpmovmskb eax (([ymm0, ymm1, ymm2, ymm3] : List VReg).getD k ymm0),
    xor eax (-1),
    jne s!"d{k}" ]) ++
  ([1, 2, 3].flatMap fun k => [lbl s!"d{k}", add r9 (.i (0x20 * k)), jmp "d0"]) ++
  [ lbl "d0",
    bsf eax eax,
    add rax r9,
    movzx ecx (byte (rdi + rax*1)),
    movzx edx (byte (rsi + rax*1)),
    cmp ecx edx,
    mov rax 1,
    ja "out",
    mov rax (-1),
   lbl "out",
    vzeroupper,
    ret ]

/-- differ(a, b, n), `n ≥ 128`: `1` if any byte of `a` and `b` differs, else
    `0` (`Memops.differBytes`). -/
def differ : Body := walk [mov eax 1, vzeroupper, ret]

end MemopsAsm
