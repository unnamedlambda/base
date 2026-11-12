module
public import AlgorithmLib.Vocab.X86
meta import AlgorithmLib.Vocab.X86
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-! The machine code `Hash` carries for gxhash: AES rounds, which Cranelift
    has no instruction for, written here and assembled by base's assembler
    (`X86.assemble`). It keeps the contract in `base/src/builtin/native.rs`:
    System V, its arguments in registers, no calls, only caller-saved
    registers. -/

namespace HashAsm

open AlgorithmLib.X86

/-- gxhash's keys: the second, third and fourth rows of its key data. -/
def keys : List UInt8 :=
  [0x42, 0x45, 0x78, 0xf2, 0x21, 0x3e, 0x9d, 0xb0, 0xe5, 0x22, 0xc2, 0x89, 0x8e, 0xc2, 0x3b, 0xfc,
   0x79, 0xe2, 0xfc, 0x03, 0x9b, 0x2e, 0x6b, 0xcb, 0x58, 0xdc, 0x61, 0xb3, 0xd9, 0x2b, 0x13, 0x39,
   0x32, 0x2e, 0x01, 0xd0, 0x7d, 0x2b, 0x9d, 0x68, 0xb7, 0xb1, 0x44, 0x55, 0x2b, 0x12, 0x8b, 0xc7]

/-- The `r9` bytes at `r8` (`r9 ≤ 16`), the rest zero, each plus `r9`, into
    xmm0. Within a page's last 16 bytes the 16 that end at the last of them
    are read and moved down, so no byte past the page is read. Scratch:
    xmm12–xmm15, r10. -/
def tailVec (p : String) : Body := [
   vpxor xmm14 xmm14 xmm14,
   vmovd xmm15 r9d,
   vpshufb xmm15 xmm15 xmm14,
   vpcmpgtb xmm14 xmm15 (xmmword (rip "idx")),
   mov r10 r8,
   and r10 0xfff,
   cmp r10 0xff0,
   jae (p ++ "safe"),
   vpand xmm0 xmm14 (xmmword r8),
   jmp (p ++ "done"),
  lbl (p ++ "safe"),
   vmovdqu xmm0 (xmmword (r8 + r9*1 - 0x10)),
   mov r10d 16,
   sub r10d r9d,
   vpxor xmm12 xmm12 xmm12,
   vmovd xmm13 r10d,
   vpshufb xmm13 xmm13 xmm12,
   vpaddb xmm13 xmm13 (xmmword (rip "idx")),
   vpcmpeqb xmm12 xmm12 xmm12,
   vpxor xmm12 xmm12 xmm14,
   vpor xmm13 xmm13 xmm12,
   vpshufb xmm0 xmm0 xmm13,
  lbl (p ++ "done"),
   vpaddb xmm0 xmm0 xmm15]

/-- gx(ptr, len, seed) → the low 64 bits of gxhash's 128-bit hash of the
    `len` bytes at `ptr` under `seed`, as gxhash's x86 implementation
    computes it: a partial vector up to 16 bytes; past that the first
    `len mod 16` (or 16) bytes as the hash vector and the next one to three
    vectors folded into a second; past 64, single vectors into the hash vector
    until a multiple of eight is left, then eight a trip in two lanes, each
    keyed by a running sum of a key; then the seed and three key rounds. -/
def gx : Body := [
   test rsi rsi,
   jne "nz",
   vpxor xmm0 xmm0 xmm0,
   jmp "fin",
  lbl "nz",
   cmp rsi 16,
   ja "big",
   mov r8 rdi,
   mov r9 rsi] ++ tailVec "a" ++ [
   jmp "fin",
  lbl "big",
   mov r9 rsi,
   and r9 15,
   jne "extra",
   vmovdqu xmm0 (xmmword rdi),
   lea r8 (mem (rdi + 0x10)),
   jmp "v0",
  lbl "extra",
   mov r8 rdi] ++ tailVec "b" ++ [
   lea r8 (mem (rdi + r9*1)),
  lbl "v0",
   vmovdqu xmm1 (xmmword r8),
   add r8 0x10,
   cmp rsi 32,
   jbe "tail",
   vaesenc xmm1 xmm1 (xmmword r8),
   add r8 0x10,
   cmp rsi 48,
   jbe "tail",
   vaesenc xmm1 xmm1 (xmmword r8),
   add r8 0x10,
   cmp rsi 64,
   jbe "tail",
   -- single vectors until a multiple of eight is left before the end
   lea r10 (mem (rdi + rsi*1)),
   mov rax r10,
   sub rax r8,
   and rax (-0x80),
   mov r9 r10,
   sub r9 rax,
  lbl "one",
   cmp r8 r9,
   jae "eight",
   vaesenc xmm0 xmm0 (xmmword r8),
   add r8 0x10,
   jmp "one",
  lbl "eight",
   vpxor xmm2 xmm2 xmm2,
   vpxor xmm3 xmm3 xmm3,
   vpor xmm4 xmm0 xmm0,
   vpor xmm5 xmm0 xmm0,
   vmovdqu xmm6 (xmmword (rip "k0")),
   vmovdqu xmm7 (xmmword (rip "k1")),
   p2align 4,
  lbl "loop",
   cmp r8 r10,
   jae "lanes",
   vmovdqu xmm8 (xmmword r8),
   vmovdqu xmm9 (xmmword (r8 + 0x10)),
   vaesenc xmm8 xmm8 (xmmword (r8 + 0x20)),
   vaesenc xmm9 xmm9 (xmmword (r8 + 0x30)),
   vaesenc xmm8 xmm8 (xmmword (r8 + 0x40)),
   vaesenc xmm9 xmm9 (xmmword (r8 + 0x50)),
   vaesenc xmm8 xmm8 (xmmword (r8 + 0x60)),
   vaesenc xmm9 xmm9 (xmmword (r8 + 0x70)),
   vpaddb xmm2 xmm2 xmm6,
   vpaddb xmm3 xmm3 xmm7,
   vaesenc xmm8 xmm8 xmm2,
   vaesenc xmm9 xmm9 xmm3,
   vaesenclast xmm4 xmm8 xmm4,
   vaesenclast xmm5 xmm9 xmm5,
   add r8 0x80,
   jmp "loop",
  lbl "lanes",
   vmovd xmm8 esi,
   vpshufd xmm8 xmm8 0,
   vpaddb xmm4 xmm4 xmm8,
   vpaddb xmm5 xmm5 xmm8,
   vaesenc xmm0 xmm4 xmm5,
  lbl "tail",
   vaesenc xmm1 xmm1 (xmmword (rip "k0")),
   vaesenc xmm1 xmm1 (xmmword (rip "k1")),
   vaesenclast xmm0 xmm0 xmm1,
  lbl "fin",
   vmovq xmm1 rdx,
   vpunpcklqdq xmm1 xmm1 xmm1,
   vaesenc xmm0 xmm0 xmm1,
   vaesenc xmm0 xmm0 (xmmword (rip "k0")),
   vaesenc xmm0 xmm0 (xmmword (rip "k1")),
   vaesenclast xmm0 xmm0 (xmmword (rip "k2")),
   vmovq rax xmm0,
   ret,
   p2align 4,
  lbl "idx", bytes ((List.range 16).map UInt8.ofNat),
  lbl "k0", bytes (keys.take 16),
  lbl "k1", bytes ((keys.drop 16).take 16),
  lbl "k2", bytes (keys.drop 32)]

end HashAsm
