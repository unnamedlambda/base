module
public import Lean
public import Scan.Ship
meta import Scan.Ship
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
public import Bench.MemopsAsm
meta import Bench.MemopsAsm
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.Prog

namespace Memops

/-!
  # The memory operations, as base's own CLIF

  `benchmarks/memops` measures these beside llvm-libc's and glibc's, called as
  C functions over the size distributions Google published. Each takes the C
  API's four arguments (`in`, `in_len`, `out`, `out_len`):

  * `memcpy`: `out[0, n) := in[0, n)`, `n = in_len`;
  * `memset`: `out[0, n) := in[0]`, `n = out_len`;
  * `bzero`: `out[0, n) := 0`, `n = out_len`;
  * `memcmp`: the sign of the first byte where `in` and `out` differ over
    `n = in_len` bytes, as the status;
  * `bcmp`: whether they differ there at all (`1` or `0`), as the status.

  A size up to 64 bytes is copied, set or compared by two pieces that overlap
  in the middle (8, 4, 2 and 1 bytes; 16; two of 16), so no size is a loop;
  past 64, 64 bytes a trip and then the last 64, overlapping the trip before.
  An operation whose `n` does not fit the buffer it writes does nothing.

  From 128 bytes up, every operation runs base's own AVX2 loop
  (`MemopsAsm`), for the 32-byte moves and compares Cranelift's vectors do not
  have: an
  operation maps its loop the first time it needs it, on a CPU with AVX2, and
  is the CLIF where the loop could not be mapped.
-/

/-! ## The machine code and where it sits -/

/-- A branch taken either way for effect. -/
def br {ty} (cc : ICmpCond) (a b : V ty) (thn els : Prog V L Unit) : Prog V L Unit := do
  let _ ← ifte (jTys := []) cc a b (do thn; pure %[]) (do els; pure %[])
  pure ()

/-- The bodies this artifact carries, in slot order. -/
def natives : List X86.Body := [MemopsAsm.copy, MemopsAsm.fill, MemopsAsm.compare, MemopsAsm.differ]

def COPY_SLOT : Nat := 0
def FILL_SLOT : Nat := 1
def COMPARE_SLOT : Nat := 2
def DIFFER_SLOT : Nat := 3

/-- From `SLOT_OFF`, one `i64` a body: 0 until its first use, then where it was
    mapped or `UNMAPPED`. At
    `NAME_OFF`, `"avx2"`, NUL-terminated, for `base_cpu_has`. From
    `CODE_OFF`, the bodies, each on a 64-byte boundary. -/
def SLOT_OFF : Nat := 0x100
def NAME_OFF : Nat := 0x200
def CODE_OFF : Nat := 0x240

example : SLOT_OFF + 8 * 2 ≤ NAME_OFF ∧ NAME_OFF + 5 ≤ CODE_OFF ∧ CODE_OFF % 64 = 0 := by decide

def bodyBytes (b : X86.Body) : List UInt8 :=
  match X86.assemble b with
  | .ok bs => bs
  | .error e => panic! e

#guard natives.all fun b => (X86.assemble b).isOk

def align64 (n : Nat) : Nat := (n + 63) / 64 * 64

def codeOffs : List Nat :=
  (natives.foldl (fun (acc : List Nat × Nat) b =>
    (acc.1 ++ [acc.2], align64 (acc.2 + (bodyBytes b).length))) ([], CODE_OFF)).1

def MEM_SIZE : Nat :=
  align64 (natives.foldl (fun acc b => align64 (acc + (bodyBytes b).length)) CODE_OFF)

def writeAt (m : List UInt8) (off : Nat) (bs : List UInt8) : List UInt8 :=
  m.take off ++ bs ++ m.drop (off + bs.length)

def initialMemory : List UInt8 :=
  (natives.zip codeOffs).foldl (fun m (b, off) => writeAt m off (bodyBytes b))
    (writeAt (zeros MEM_SIZE) NAME_OFF ("avx2".toUTF8.toList ++ [0]))

/-- A slot's word when its body could not be mapped (no x86-64, no AVX2, or
    the map failed): the operation stays the CLIF without trying again. -/
def UNMAPPED : Int := 1

/-- Body `b`'s address, mapping it the first time, for any artifact that
    carries a body: the arena word at `slotAt` holds 0 until then, the address
    after, or `UNMAPPED`; `"avx2"` sits at `nameAt`, the body's bytes at
    `codeAt`. -/
def mappedAt (slotAt nameAt codeAt : Nat) (b : X86.Body) : Prog V L (V .i64) := do
  let base ← basePtr
  let at_ ← iaddImm base slotAt
  when .eq (← load64 at_) (← iconst64 0) do
    let one ← iconst32 1
    let ok ← band (← select (← icmp .eq (← nativeArch) one) one (← iconst32 0))
      (← select (← icmp .eq (← cpuHas (← iaddImm base nameAt) (← iconst64 5)) one) one (← iconst32 0))
    let fn ← ifte (jTys := [.i64]) .ne ok (← iconst32 0)
      (thn := do pure %[← nativeLoad (← iaddImm base codeAt) (← iconst64 (bodyBytes b).length)])
      (els := do pure %[← iconst64 0])
    storeI64 (← select (← icmp .eq fn.head (← iconst64 0)) (← iconst64 UNMAPPED) fn.head) at_
  load64 at_

/-- Body `slot`'s address, mapping it the first time (`mappedAt` at this
    artifact's layout). -/
def mapped (slot : Nat) (b : X86.Body) (off : Nat) : Prog V L (V .i64) :=
  mappedAt (SLOT_OFF + 8 * slot) NAME_OFF off b

/-- Where the large-size functions sit: body `slot`'s at `LARGE0 + slot`,
    after the five entries. -/
def LARGE0 : Nat := 5

def large (slot : Nat) : LocalRef [.i64, .i64, .i64, .i64] (some .i64) := ⟨.local (LARGE0 + slot)⟩

/-- **The large-size function of body `slot`**, `(n, a, b, c)`: the body on
    `a b c` when it is mapped, answering `answer n a b c` of what the body
    returns, else `fallback`, answering what it does. Its own function, so the
    entries' small sizes, which call nothing, save no registers for the calls
    this makes. -/
def largeBody (slot : Nat) (fallback : V .i64 → V .i64 → V .i64 → V .i64 → Prog V L (V .i64))
    (answer : V .i64 → V .i64 → V .i64 → V .i64 → V .i64 → V .i64 := fun _ _ _ _ r => r) :
    Prog V L (V .i64) := do
  let ps ← entryParams [.i64, .i64, .i64, .i64]
  let n := ps.head; let a := ps.snd; let b := ps.thd; let c := ps.fth
  let (body, off) := ((natives.zip codeOffs)[slot]?).getD ([], CODE_OFF)
  let fn ← mapped slot body off
  let r ← ifte (jTys := [.i64]) .eq fn (← iconst64 UNMAPPED) (thn := do pure %[← fallback n a b c])
    (els := do pure %[answer n a b c (← nativeCall fn a b c (← iconst64 0))])
  pure r.head

/-- `small` under 128 bytes, else body `slot`'s large-size function on `n a b c`. -/
def viaNative (slot : Nat) (n a b c : V .i64) (small : Prog V L (V .i64)) : Prog V L (V .i64) := do
  let r ← ifte (jTys := [.i64]) .ult n (← iconst64 128) (thn := do pure %[← small])
    (els := do pure %[← callLocal (large slot) %[n, a, b, c]])
  pure r.head

/-- `f` at `q = p, p + stride, …` while `q < e`, tested at the bottom after
    one guard. -/
def ptrLoop (p e : V .i64) (stride : Int) (f : V .i64 → Prog V L Unit) : Prog V L Unit := do
  let _ ← dwloop %[p] .ult e (contOnTrue := true) [0] (guardIdx := some 0)
    (body := fun c => do
      f c.head
      let q' ← iaddImm c.head stride
      return (q', %[q']))
  pure ()

def loadU (a : V .i64) : Prog V L (V .i8x16) := load { ty := .i8x16 } a

/-- **`d[0, n) := s[0, n)`**, `s` and `d` apart. -/
def copyBytes (s d n : V .i64) : Prog V L Unit := do
  let pair (w : Nat) (ld : V .i64 → Prog V L Unit) : Prog V L Unit := do
    ld (← iconst64 0)
    ld (← isub n (← iconst64 w))
  br .ule n (← iconst64 16)
    (br .uge n (← iconst64 8)
      (pair 8 fun o => do storeI64 (← load64 (← iadd s o)) (← iadd d o))
      (br .uge n (← iconst64 4)
        (pair 4 fun o => do storeI32 (← load32 (← iadd s o)) (← iadd d o))
        (br .uge n (← iconst64 2)
          (pair 2 fun o => do store (← load_i16 (← iadd s o)) (← iadd d o))
          (when .ne n (← iconst64 0) do istore8 (← uload8_64 s) d))))
    (br .ule n (← iconst64 32)
      (pair 16 fun o => do storeUnaligned (← loadU (← iadd s o)) (← iadd d o))
      (br .ule n (← iconst64 64)
        (pair 32 fun o => do
          for k in [0, 16] do storeUnaligned (← loadU (← iaddImm (← iadd s o) k)) (← iaddImm (← iadd d o) k))
        (do
          let delta ← isub d s
          let last ← iadd s (← isub n (← iconst64 64))
          let copy64 (q : V .i64) : Prog V L Unit := do
            for k in [0, 16, 32, 48] do
              storeUnaligned (← loadU (← iaddImm q k)) (← iaddImm (← iadd q delta) k)
          ptrLoop s last 64 copy64
          copy64 last)))

/-- **`d[0, n) := c`**, `c` a byte in the low 8 bits of `c64` and in every
    lane of `cv`. -/
def setBytes (d n c64 : V .i64) (cv : V .i8x16) : Prog V L Unit := do
  let pair (w : Nat) (st : V .i64 → Prog V L Unit) : Prog V L Unit := do
    st (← iconst64 0)
    st (← isub n (← iconst64 w))
  br .ule n (← iconst64 16)
    (br .uge n (← iconst64 8)
      (pair 8 fun o => do storeI64 c64 (← iadd d o))
      (br .uge n (← iconst64 4)
        (pair 4 fun o => do istore32 c64 (← iadd d o))
        (br .uge n (← iconst64 2)
          (pair 2 fun o => do istore16 c64 (← iadd d o))
          (when .ne n (← iconst64 0) do istore8 c64 d))))
    (br .ule n (← iconst64 32)
      (pair 16 fun o => do storeUnaligned cv (← iadd d o))
      (br .ule n (← iconst64 64)
        (pair 32 fun o => do
          for k in [0, 16] do storeUnaligned cv (← iaddImm (← iadd d o) k))
        (do
          let last ← iadd d (← isub n (← iconst64 64))
          let set64 (q : V .i64) : Prog V L Unit := do
            for k in [0, 16, 32, 48] do storeUnaligned cv (← iaddImm q k)
          ptrLoop d last 64 set64
          set64 last)))

/-- `-1`, `0` or `1` as `x` is below, equal to or above `y`, unsigned. -/
def order (x y : V .i64) : Prog V L (V .i64) := do
  let lt ← select (← icmp .ult x y) (← iconst64 (-1)) (← iconst64 0)
  select (← icmp .ugt x y) (← iconst64 1) lt

/-- **`compare` under 16 bytes**: two words of 8, 4 or 2 overlapping,
    compared in byte order (byte-swapped, so the first byte is the most
    significant), or the one byte. -/
def compareTiny (a b n : V .i64) : Prog V L (V .i64) := do
  let zero ← iconst64 0
  let at_ (ld : V .i64 → Prog V L (V .i64)) (o : V .i64) : Prog V L (V .i64) := do
    order (← ld (← iadd a o)) (← ld (← iadd b o))
  let w64 (p : V .i64) : Prog V L (V .i64) := do bswap (← load64 p)
  let w32 (p : V .i64) : Prog V L (V .i64) := do uextend64 (← bswap (← load32 p))
  let w16 (p : V .i64) : Prog V L (V .i64) := do uextend64 (← bswap (← load_i16 p))
  -- the first of two orders that is not equal
  let first (x y : V .i64) : Prog V L (V .i64) := do select (← icmp .ne x zero) x y
  let half (w : Nat) (ld : V .i64 → Prog V L (V .i64)) : Prog V L (Vals V [.i64]) := do
    pure %[← first (← at_ ld zero) (← at_ ld (← isub n (← iconst64 w)))]
  let r ← ifte (jTys := [.i64]) .uge n (← iconst64 8) (thn := half 8 w64)
    (els := ifte .uge n (← iconst64 4) (thn := half 4 w32)
      (els := ifte .uge n (← iconst64 2) (thn := half 2 w16)
        (els := ifte .ne n zero (thn := do pure %[← at_ uload8_64 zero]) (els := pure %[zero]))))
  pure r.head

/-- The lanes of the 16 bytes at `o1` (the low half) and the 16 at `o2` where
    `a` and `b` agree, a bit each: all ones when every byte agrees. -/
def agreePair (a b o1 o2 : V .i64) : Prog V L (V .i32) := do
  let agree (o : V .i64) : Prog V L (V .i32) := do
    vhighBits (← icmp .eq (← loadU (← iadd a o)) (← loadU (← iadd b o)))
  bor (← agree o1) (← ishlImm (← agree o2) 16)

/-- The order of the first byte that differs in the pair at `o1`, `o2`,
    given its `agreePair` mask `m`, not all ones; `0` when it is all ones. -/
def firstOfPair (a b o1 o2 : V .i64) (m : V .i32) : Prog V L (V .i64) := do
  let r ← ifte (jTys := [.i64]) .ne m (← iconst32 0xFFFFFFFF)
    (thn := do
      let k ← uextend64 (← ctz (← bxor m (← iconst32 0xFFFFFFFF)))
      let j ← select (← icmp .ult k (← iconst64 16)) (← iadd o1 k) (← iadd o2 (← isub k (← iconst64 16)))
      pure %[← order (← uload8_64 (← iadd a j)) (← uload8_64 (← iadd b j))])
    (els := do pure %[← iconst64 0])
  pure r.head

/-- **`compare` from 16 to 64 bytes**: the first 16 and the last 16 up to
    32; past that the first 32, then the last 32. -/
def comparePairs (a b n : V .i64) : Prog V L (V .i64) := do
  let both ← iconst32 0xFFFFFFFF
  let r ← ifte (jTys := [.i64]) .ule n (← iconst64 32)
    (thn := do
      let zero ← iconst64 0
      let o2 ← isub n (← iconst64 16)
      pure %[← firstOfPair a b zero o2 (← agreePair a b zero o2)])
    (els := do
      let zero ← iconst64 0
      let o16 ← iconst64 16
      let m ← agreePair a b zero o16
      let r2 ← ifte (jTys := [.i64]) .ne m both (thn := do pure %[← firstOfPair a b zero o16 m])
        (els := do
          let o1 ← isub n (← iconst64 32)
          let o2 ← iaddImm o1 16
          pure %[← firstOfPair a b o1 o2 (← agreePair a b o1 o2)])
      pure %[r2.head])
  pure r.head

/-- **The sign of the first byte where `a` and `b` differ** over `n` bytes:
    from 16, 32 bytes a trip (two vectors' lane masks as one word) until a
    lane differs, then the last 32 overlapping the trips (under 32, the first
    16 and the last 16), and the first differing lane's bytes compared; under
    16, `compareTiny`. -/
def compareBytes (a b n : V .i64) : Prog V L (V .i64) := do
  let zero ← iconst64 0
  let both ← iconst32 0xFFFFFFFF
  let r ← ifte (jTys := [.i64]) .uge n (← iconst64 16)
    (thn := do
      let one ← iconst64 1
      let e32 ← isub n (← iconst64 32)
      let s ← wloop %[zero, both]
        (fun cs => do
          let more ← band (← select (← icmp .slt cs.head e32) one zero)
            (← select (← icmp .eq cs.snd both) one zero)
          pure (contIf .ne more zero, %[cs.head, cs.snd], ()))
        (fun cs _ => do
          let q := cs.head
          pure %[← iaddImm q 32, ← agreePair a b q (← iaddImm q 16)])
      let fin ← ifte (jTys := [.i64, .i64, .i32]) .ne s.snd both
        (thn := do
          let o1 ← isub s.head (← iconst64 32)
          pure %[o1, ← iaddImm o1 16, s.snd])
        (els := do
          let o1 ← select (← icmp .uge n (← iconst64 32)) e32 zero
          let o2 ← isub n (← iconst64 16)
          pure %[o1, o2, ← agreePair a b o1 o2])
      pure %[← firstOfPair a b fin.head fin.snd fin.thd])
    (els := do pure %[← compareTiny a b n])
  pure r.head

/-- Where `memcmp`'s middle-size function sits (65 up to 128 bytes): its own
    function, so the entry's smaller sizes save no registers its loop uses. -/
def MID_COMPARE : Nat := LARGE0 + 4

def midCompare : LocalRef [.i64, .i64, .i64, .i64] (some .i64) := ⟨.local MID_COMPARE⟩

/-- **Whether `a` and `b` differ** over `n` bytes, `1` or `0`: the or of
    every piece's exclusive or (from 32, 32 bytes a trip in vectors and the
    last 32 overlapping; under that two pieces of 16, 8, 4 or 2 overlapping, or
    the one byte), then whether it is not zero, so a caller narrowing the
    status to `int` keeps a difference in the upper bytes. -/
def differBytes (a b n : V .i64) : Prog V L (V .i64) := do
  let zero ← iconst64 0
  let x (ld : V .i64 → Prog V L (V .i64)) (o : V .i64) : Prog V L (V .i64) := do
    bxor (← ld (← iadd a o)) (← ld (← iadd b o))
  let u32 (p : V .i64) : Prog V L (V .i64) := uload32_64 p
  let u16 (p : V .i64) : Prog V L (V .i64) := uload16_64 p
  let xv (o : V .i64) : Prog V L (V .i8x16) := do bxor (← loadU (← iadd a o)) (← loadU (← iadd b o))
  -- `1` where a vector of exclusive ors is not all zero
  let any (v : V .i8x16) : Prog V L (V .i64) := do
    let m ← vhighBits (← icmp .eq v (← splat .i8x16 (← ireduce .i8 zero)))
    select (← icmp .ne m (← iconst32 0xFFFF)) (← iconst64 1) zero
  let r ← ifte (jTys := [.i64]) .uge n (← iconst64 32)
    (thn := do
      -- 64 bytes a trip, the ors kept in a vector, then the last 32 and
      -- the 32 before them
      let e ← isub n (← iconst64 64)
      let acc ← wloop %[zero, ← splat .i8x16 (← ireduce .i8 zero)]
        (fun cs => pure (contIf .slt cs.head e, %[cs.snd], ()))
        (fun cs _ => do
          let q := cs.head
          let x01 ← bor (← xv q) (← xv (← iaddImm q 16))
          let x23 ← bor (← xv (← iaddImm q 32)) (← xv (← iaddImm q 48))
          pure %[← iaddImm q 64, ← bor cs.snd (← bor x01 x23)])
      let l ← isub n (← iconst64 32)
      let tail ← bor (← xv l) (← xv (← iaddImm l 16))
      -- the 32 before the last, where the trips stopped short of them (from 64)
      let m ← select (← icmp .uge n (← iconst64 64)) (← isub n (← iconst64 64)) zero
      let mid ← bor (← xv m) (← xv (← iaddImm m 16))
      pure %[← any (← bor acc.head (← bor tail mid))])
    (els := ifte .uge n (← iconst64 16)
      (thn := do pure %[← any (← bor (← xv zero) (← xv (← isub n (← iconst64 16))))])
    (els := ifte .uge n (← iconst64 8)
      (thn := do pure %[← bor (← x load64 zero) (← x load64 (← isub n (← iconst64 8)))])
    (els := do
      let half (w : Nat) (ld : V .i64 → Prog V L (V .i64)) : Prog V L (Vals V [.i64]) := do
        pure %[← bor (← x ld zero) (← x ld (← isub n (← iconst64 w)))]
      ifte .uge n (← iconst64 4) (thn := half 4 u32)
        (els := ifte .uge n (← iconst64 2) (thn := half 2 u16)
          (els := ifte .ne n zero (thn := do pure %[← x uload8_64 zero]) (els := pure %[zero]))))))
  select (← icmp .ne r.head zero) (← iconst64 1) zero

/-! ## The entries -/


/-- **The middle-size function** `(n, a, b, c)`: `f` on them. -/
def midBody (f : V .i64 → V .i64 → V .i64 → V .i64 → Prog V L (V .i64)) : Prog V L (V .i64) := do
  let ps ← entryParams [.i64, .i64, .i64, .i64]
  f ps.head ps.snd ps.thd ps.fth

def memcmpBody : Prog V L (V .i64) := do
  let n ← dataLen
  let r ← ifte (jTys := [.i64]) .ule n (← outLen)
    (thn := do
      let a ← dataPtr
      let b ← outPtr
      let r ← ifte (jTys := [.i64]) .ult n (← iconst64 16) (thn := do pure %[← compareTiny a b n])
        (els := ifte .ule n (← iconst64 64) (thn := do pure %[← comparePairs a b n])
          (els := ifte .ult n (← iconst64 128) (thn := do pure %[← callLocal midCompare %[n, a, b, n]])
            (els := do pure %[← callLocal (large COMPARE_SLOT) %[n, a, b, n]])))
      pure %[r.head])
    (els := do pure %[← iconst64 0])
  pure r.head

def bcmpBody : Prog V L (V .i64) := do
  let n ← dataLen
  let r ← ifte (jTys := [.i64]) .ule n (← outLen)
    (thn := do
      let a ← dataPtr
      let b ← outPtr
      pure %[← viaNative DIFFER_SLOT n a b n (differBytes a b n)])
    (els := do pure %[← iconst64 0])
  pure r.head

/-- `memcpy`: the input copied to the output when it fits; the output's
    address, as C's `memcpy` answers its destination (`0` when it does not
    fit). -/
def memcpyBody : Prog V L (V .i64) := do
  let n ← dataLen
  let r ← ifte (jTys := [.i64]) .ule n (← outLen)
    (thn := do
      let s ← dataPtr
      let d ← outPtr
      pure %[← viaNative COPY_SLOT n s d n (do copyBytes s d n; pure d)])
    (els := do pure %[← iconst64 0])
  pure r.head

/-- The output filled with `c`'s low byte, in each of a word's 8 and each
    lane of a vector; the output's address. -/
def setWide (c : V .i64) : Prog V L (V .i64) := do
  let c64 ← imul c (← iconst64 0x0101010101010101)
  let d ← outPtr
  let n ← outLen
  viaNative FILL_SLOT n d n c64 (do setBytes d n c64 (← splat .i8x16 (← ireduce .i8 c)); pure d)

/-- `memset`: the output filled with the input's first byte; the output's
    address, as C's `memset` answers its destination (`0` with no input
    byte). -/
def memsetBody : Prog V L (V .i64) := do
  let r ← ifte (jTys := [.i64]) .uge (← dataLen) (← iconst64 1)
    (thn := do pure %[← setWide (← uload8_64 (← dataPtr))]) (els := do pure %[← iconst64 0])
  pure r.head

def bzeroBody : Prog V L Unit := do let _ ← setWide (← iconst64 0)

def clifIR : Except String (List FuncData) :=
  Prog.program
    [Prog.entry "memcpy" (Prog.compileProgStatus 0 memcpyBody),
     Prog.entry "memset" (Prog.compileProgStatus 1 memsetBody),
     Prog.entry "bzero" (Prog.compileProg 2 bzeroBody),
     Prog.entry "memcmp" (Prog.compileProgStatus 3 memcmpBody),
     Prog.entry "bcmp" (Prog.compileProgStatus 4 bcmpBody),
     Prog.compileProgStatus (LARGE0 + COPY_SLOT)
       (largeBody COPY_SLOT (fun n s d _ => do copyBytes s d n; pure d) fun _ _ d _ _ => d)
       (List.replicate 4 .i64),
     Prog.compileProgStatus (LARGE0 + FILL_SLOT)
       (largeBody FILL_SLOT (fun n d _ c64 => do
         setBytes d n c64 (← splat .i8x16 (← ireduce .i8 c64)); pure d) fun _ d _ _ _ => d)
       (List.replicate 4 .i64),
     Prog.compileProgStatus (LARGE0 + COMPARE_SLOT)
       (largeBody COMPARE_SLOT fun n a b _ => compareBytes a b n) (List.replicate 4 .i64),
     Prog.compileProgStatus (LARGE0 + DIFFER_SLOT)
       (largeBody DIFFER_SLOT fun n a b _ => differBytes a b n) (List.replicate 4 .i64),
     Prog.compileProgStatus MID_COMPARE (midBody fun n a b _ => compareBytes a b n)
       (List.replicate 4 .i64)]

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[artifactEntry "memops" { functions := clif, required_memory := MEM_SIZE, initial_memory := initialMemory }]

end Memops

def Bench.Memops.main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  let clif ← Prog.orDie Memops.clifIR
  emitArtifacts outDir (Memops.artifacts clif)

#eval ShipScan.check "Bench.Memops" `Bench.Memops.main
