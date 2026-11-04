module
public import AlgorithmLib.Vocab.X86
meta import AlgorithmLib.Vocab.X86
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-! The machine code `Bench.Image` carries: the AVX2 loops of three 2:1
    downscales, the antialiased bilinear one whose taps are `1 3 3 1` along each
    axis, the 2x2 average, and Lanczos3, twelve signed 14-bit taps along each
    axis. Each takes its arguments in registers and keeps
    the contract in `base/src/builtin/native.rs`. -/

namespace ImageAsm

open AlgorithmLib.X86

/-- The bytes `a b` sixteen times: a `vpmaddubsw` multiplier. -/
def pairs (a b : Nat) : List UInt8 := (List.replicate 16 [a.toUInt8, b.toUInt8]).flatten

/-- `n` copies of the 16-bit `x`. -/
def words (n x : Nat) : List UInt8 := (List.replicate n (le 2 x)).flatten

/-- One trip: sixteen outputs.

    `r8` is the source byte under the first output's first tap in row `A`, and
    row `B` is `r9` bytes on. Along each row the taps give, for output `j`,
    `s[2j] + 3 s[2j+1] + 3 s[2j+2] + s[2j+3]`: `ymm0` for `A`, `ymm2` for `B`.
    `r10` holds the previous pair's sums, sixteen of `A`'s then sixteen of
    `B`'s; this pair's go to `r11` the same way. The output is
    `(P.A + 3 P.B + 3 A + B + 32) >> 6`, packed to sixteen bytes at `rsi`. -/
def trip : Body := [
   vmovdqu ymm0 (ymmword r8),
   vpmaddubsw ymm0 ymm0 ymm13,
   vmovdqu ymm1 (ymmword (r8 + 2)),
   vpmaddubsw ymm1 ymm1 ymm14,
   vpaddw ymm0 ymm0 ymm1,
   vmovdqu ymm2 (ymmword (r8 + r9*1)),
   vpmaddubsw ymm2 ymm2 ymm13,
   vmovdqu ymm3 (ymmword (r8 + r9*1 + 2)),
   vpmaddubsw ymm3 ymm3 ymm14,
   vpaddw ymm2 ymm2 ymm3,
   vmovdqu (ymmword r11) ymm0,
   vmovdqu (ymmword (r11 + 0x20)) ymm2,
   vpaddw ymm4 ymm2 (ymmword r10),
   vpaddw ymm5 ymm0 (ymmword (r10 + 0x20)),
   vpsllw ymm6 ymm5 1,
   vpaddw ymm5 ymm5 ymm6,
   vpaddw ymm4 ymm4 ymm5,
   vpaddw ymm4 ymm4 ymm15,
   vpsrlw ymm4 ymm4 6,
   vpackuswb ymm4 ymm4 ymm4,
   vpermq ymm4 ymm4 0xd8,
   vmovdqu (xmmword rsi) xmm4]

/-- rowpair(args): one output row from two new source rows and the previous
    pair's sums. `args` holds six `u64`s: row `A`'s first tap, the row stride,
    the previous pair's sums, where this pair's go, the output, and the count
    of outputs, at least 16.

    Every other trip prefetches the next output row's two source rows, one
    line of each, so they arrive while this row is worked on. Whole trips walk
    the pointers. A count that is not a multiple of 16 ends
    with one more trip over its last sixteen outputs, which writes some twice
    with the same values; the sums stay in the next sixty-four bytes, so a row
    of sums takes `64 * ceil(n / 16)` bytes. -/
def rowpair : Body := [
   mov r8 (qword rdi),
   mov r9 (qword (rdi + 8)),
   mov r10 (qword (rdi + 0x10)),
   mov r11 (qword (rdi + 0x18)),
   mov rsi (qword (rdi + 0x20)),
   mov rcx (qword (rdi + 0x28)),
   vmovdqu ymm13 (ymmword (rip "k13")),
   vmovdqu ymm14 (ymmword (rip "k31")),
   vmovdqu ymm15 (ymmword (rip "k32")),
   lea rdx (mem (r9 + r9*2)),
   shr rcx 4,
   p2align 4,
  lbl "loop",
   test rcx 1,
   je "fetched",
   prefetcht0 (byte (r8 + r9*2)),
   prefetcht0 (byte (r8 + rdx*1)),
  lbl "fetched"] ++ trip ++ [
   add r8 0x20,
   add r10 0x40,
   add r11 0x40,
   add rsi 0x10,
   dec rcx,
   jne "loop",
   mov rcx (qword (rdi + 0x28)),
   test rcx 0xf,
   je "done",
   mov r8 (qword rdi),
   lea r8 (mem (r8 + rcx*2)),
   sub r8 0x20,
   mov rsi (qword (rdi + 0x20)),
   add rsi rcx,
   sub rsi 0x10] ++ trip ++ [
  lbl "done",
   vzeroupper,
   ret,
   p2align 5,
  lbl "k13",
   bytes (pairs 1 3),
  lbl "k31",
   bytes (pairs 3 1),
  lbl "k32",
   bytes (words 16 32)]

/-- One trip of `boxrow`: thirty-two outputs, each `(a + b + c + d + 2) >> 2`
    of a 2x2 block. `vpmaddubsw` against ones sums each row's byte pairs into
    words; the two rows are added, rounded, shifted and packed. -/
def boxTrip : Body := [
   prefetcht0 (byte (r8 + r9*2)),
   prefetcht0 (byte (r8 + rdx*1)),
   vmovdqu ymm0 (ymmword r8),
   vmovdqu ymm1 (ymmword (r8 + 0x20)),
   vmovdqu ymm2 (ymmword (r8 + r9*1)),
   vmovdqu ymm3 (ymmword (r8 + r9*1 + 0x20)),
   vpmaddubsw ymm0 ymm0 ymm15,
   vpmaddubsw ymm1 ymm1 ymm15,
   vpmaddubsw ymm2 ymm2 ymm15,
   vpmaddubsw ymm3 ymm3 ymm15,
   vpaddw ymm0 ymm0 ymm2,
   vpaddw ymm1 ymm1 ymm3,
   vpaddw ymm0 ymm0 ymm14,
   vpaddw ymm1 ymm1 ymm14,
   vpsrlw ymm0 ymm0 2,
   vpsrlw ymm1 ymm1 2,
   vpackuswb ymm0 ymm0 ymm1,
   vpermq ymm0 ymm0 0xd8,
   vmovdqu (ymmword rsi) ymm0]

/-- boxrow(args): one output row of the 2x2 average. `args` holds four `u64`s:
    the first source row, the row stride, the output and the count of outputs,
    at least 32. Each trip prefetches the next output row's two source rows.
    A count that is not a multiple of 32 ends with one more trip over its last
    thirty-two outputs. -/
def boxrow : Body := [
   mov r8 (qword rdi),
   mov r9 (qword (rdi + 8)),
   mov rsi (qword (rdi + 0x10)),
   mov rcx (qword (rdi + 0x18)),
   vmovdqu ymm15 (ymmword (rip "ones")),
   vmovdqu ymm14 (ymmword (rip "twos")),
   lea rdx (mem (r9 + r9*2)),
   shr rcx 5,
   p2align 4,
  lbl "loop"] ++ boxTrip ++ [
   add r8 0x40,
   add rsi 0x20,
   dec rcx,
   jne "loop",
   mov rcx (qword (rdi + 0x18)),
   test rcx 0x1f,
   je "done",
   mov r8 (qword rdi),
   lea r8 (mem (r8 + rcx*2)),
   sub r8 0x40,
   mov rsi (qword (rdi + 0x10)),
   add rsi rcx,
   sub rsi 0x20] ++ boxTrip ++ [
  lbl "done",
   vzeroupper,
   ret,
   p2align 5,
  lbl "ones",
   bytes (List.replicate 32 1),
  lbl "twos",
   bytes (words 16 2)]

/-- `n` copies of the 32-bit `x`. -/
def dwords (n x : Nat) : List UInt8 := (List.replicate n (le 4 x)).flatten

/-- One trip of `lzh` at output `rax`: sixteen sums of twelve taps. For each
    pair of taps, sixteen source bytes zero-extended to words are multiplied
    pairwise by the pair's weights (`ymm10`--`ymm15`) and added in 32-bit lanes;
    the sums are rounded, shifted right by 8 and packed to words. -/
def lzhTrip : Body :=
  (List.range 6).flatMap (fun p =>
    [vpmovzxbw ymm2 (xmmword (rdi + rax*2 + 2 * p)),
     vpmovzxbw ymm3 (xmmword (rdi + rax*2 + (16 + 2 * p)))] ++
    (if p == 0 then
      [vpmaddwd ymm0 ymm2 (.v ⟨10, true⟩), vpmaddwd ymm1 ymm3 (.v ⟨10, true⟩)]
     else
      [vpmaddwd ymm2 ymm2 (.v ⟨10 + p, true⟩), vpmaddwd ymm3 ymm3 (.v ⟨10 + p, true⟩),
       vpaddd ymm0 ymm0 ymm2, vpaddd ymm1 ymm1 ymm3])) ++
  [vpaddd ymm0 ymm0 ymm9,
   vpaddd ymm1 ymm1 ymm9,
   vpsrad ymm0 ymm0 8,
   vpsrad ymm1 ymm1 8,
   vpackssdw ymm0 ymm0 ymm1,
   vpermq ymm0 ymm0 0xd8,
   vmovdqu (ymmword (rsi + rax*2)) ymm0]

/-- lzh(s, dst, n, w): one source row's Lanczos3 sums for `n` outputs, at least
    16: output `j`'s taps are the twelve bytes from `s + 2j`, its sum the word
    at `dst + 2j`. `w` holds the six tap pairs' weights, each pair a 32-byte
    vector of eight copies. -/
def lzh : Body :=
  (List.range 6).map (fun p => vmovdqu (.v ⟨10 + p, true⟩) (ymmword (rcx + 0x20 * p))) ++ [
   vmovdqu ymm9 (ymmword (rip "r8")),
   mov r8 rdx,
   sub r8 0x10,
   xor eax eax,
   p2align 4,
  lbl "loop"] ++ lzhTrip ++ [
   add rax 0x10,
   cmp rax r8,
   jbe "loop",
   cmp rax rdx,
   jae "done",
   mov rax r8] ++ lzhTrip ++ [
  lbl "done",
   vzeroupper,
   ret,
   p2align 5,
  lbl "r8",
   bytes (dwords 8 128)]

/-- One trip of `lzv` at output `rax`: sixteen outputs from twelve rows of
    sums. Each pair of rows is interleaved word by word and multiplied pairwise
    by the pair's weights from `r10`, the low and high halves into `ymm0` and
    `ymm1`; the sums are rounded, shifted right by 20, and packed to bytes with
    saturation. -/
def lzvTrip : Body :=
  [vpxor ymm0 ymm0 ymm0, vpxor ymm1 ymm1 ymm1] ++
  (List.range 6).flatMap (fun p =>
    [mov r8 (qword (rdi + 16 * p)),
     mov r9 (qword (rdi + (16 * p + 8))),
     vmovdqu ymm2 (ymmword (r8 + rax*2)),
     vmovdqu ymm3 (ymmword (r9 + rax*2)),
     vpunpcklwd ymm4 ymm2 ymm3,
     vpunpckhwd ymm5 ymm2 ymm3,
     vpmaddwd ymm4 ymm4 (ymmword (r10 + 0x20 * p)),
     vpmaddwd ymm5 ymm5 (ymmword (r10 + 0x20 * p)),
     vpaddd ymm0 ymm0 ymm4,
     vpaddd ymm1 ymm1 ymm5]) ++
  [vpaddd ymm0 ymm0 ymm15,
   vpaddd ymm1 ymm1 ymm15,
   vpsrad ymm0 ymm0 20,
   vpsrad ymm1 ymm1 20,
   vpackssdw ymm0 ymm0 ymm1,
   vpackuswb ymm0 ymm0 ymm0,
   vpermq ymm0 ymm0 0xd8,
   vmovdqu (xmmword (rsi + rax*1)) xmm0]

/-- lzv(args): one output row of Lanczos3 from twelve rows of `lzh`'s sums.
    `args` holds fifteen `u64`s: the twelve rows, the weights (six pairs, each
    a 32-byte vector of eight copies), the output and the count of outputs, at
    least 16. -/
def lzv : Body := [
   mov r10 (qword (rdi + 0x60)),
   mov rsi (qword (rdi + 0x68)),
   mov rdx (qword (rdi + 0x70)),
   vmovdqu ymm15 (ymmword (rip "r20")),
   mov rcx rdx,
   sub rcx 0x10,
   xor eax eax,
   p2align 4,
  lbl "loop"] ++ lzvTrip ++ [
   add rax 0x10,
   cmp rax rcx,
   jbe "loop",
   cmp rax rdx,
   jae "done",
   mov rax rcx] ++ lzvTrip ++ [
  lbl "done",
   vzeroupper,
   ret,
   p2align 5,
  lbl "r20",
   bytes (dwords 8 0x80000)]

end ImageAsm
