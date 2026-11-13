module
public import Lean
public import Scan.Ship
meta import Scan.Ship
public import AlgorithmLib.Surface.ProgFFI
meta import AlgorithmLib.Surface.ProgFFI
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

open Lean
open AlgorithmLib
open AlgorithmLib.IR
open AlgorithmLib.Prog

namespace Json

/-!
  # base's JSON parser

  `benchmarks/json` measures it beside simdjson's DOM parser on the standard
  corpora. One pass over the text, with an explicit stack of open
  containers, writes a tape of 64-bit words, each value's tag in its top
  byte:

  * `{` and `[`: the index of the matching close in the low 56 bits; `}` and
    `]` the index of their open;
  * `"`: the string's offset in the string buffer, where its length (`u32`)
    and its unescaped UTF-8 bytes lie;
  * `l`, `u`, `d`: an `int64`, a `uint64` too large for one, a `double`,
    the value in the next word (simdjson's classes for a number);
  * `t`, `f`, `n`: `true`, `false`, `null`.

  A double is the correctly rounded nearest: Clinger's exact fast path where
  the digits and the power of ten are small, Eisel and Lemire's 128-bit
  product with a table of the powers of five (`POW5`) otherwise. A number with
  more than 19 significant digits, or one Eisel–Lemire cannot round, is
  refused, as is any text that is not JSON.

  Its functions:

  * `parse`: the text as the input; the status 0, or the byte where it is not
    JSON (one past it, so 0 stays success), or where its strings could
    outgrow the string buffer. Every double is the nearest to its text: the
    fast paths where they can round, the decimal slow path (`slowFloat`)
    where they cannot; an integer past 64 bits is refused;
  * `sum`: the output's first word gets a checksum of the last tape, every
    value in document order (`benchmarks/json` folds simdjson's DOM the same
    way).
-/

/-! ## The arena -/

/-- State words: the tape's length in words, the string buffer's in bytes. -/
def TAPE_LEN : Nat := 0x100
def STR_LEN : Nat := 0x108
/-- The powers of ten `1e0 … 1e22` as doubles (exact), and the 128-bit
    powers of five for Eisel–Lemire, `5^q` for `q` in `-342 … 308`, two words
    each (high, low). -/
def POW10 : Nat := 0x200
/-- `10^0 … 10^8` as words, to scale a value by the digits read after it. -/
def POWI : Nat := 0x300
def POW5 : Nat := 0x400
def POW5_MIN : Int := -342
def POW5_COUNT : Nat := 651
/-- The stack of open containers: a word each, the tape index of the open
    times two, plus 1 for an object. -/
def STACK : Nat := POW5 + 16 * POW5_COUNT
def MAX_DEPTH : Nat := 1024
def TAPE : Nat := STACK + 8 * MAX_DEPTH
def TAPE_WORDS : Nat := 1 <<< 23
def STRS : Nat := TAPE + 8 * TAPE_WORDS
def STR_BYTES : Nat := 1 <<< 26
/-- The slow path's decimal (`slowFloat`): its digits, a scratch copy for a
    left shift, then its digit count, decimal point and whether digits past
    `MAX_DIGITS` were dropped. -/
def MAX_DIGITS : Nat := 768
def DEC : Nat := STRS + STR_BYTES
def DEC_TMP : Nat := DEC + 1024
def DEC_ND : Nat := DEC_TMP + 1024
def DEC_DP : Nat := DEC_ND + 8
def DEC_TR : Nat := DEC_DP + 8
def MEM : Nat := DEC_TR + 8

/-- fast_float's table: `5^q` with its top bit at 127, truncated for `q ≥ 0`;
    for `q < 0`, `2^b / 5^-q` plus one, `b` putting the top bit at 127. -/
def pow5 (q : Int) : Nat :=
  if q < 0 then
    let p5 := 5 ^ q.natAbs
    let z := Nat.log2 p5 + (if 2 ^ Nat.log2 p5 < p5 then 1 else 0)
    if q ≥ -27 then 2 ^ (z + 127) / p5 + 1
    else
      let c := 2 ^ (2 * z + 128) / p5 + 1
      c / 2 ^ (Nat.log2 c - 127)
  else
    let p5 := 5 ^ q.toNat
    let l := Nat.log2 p5
    if l ≤ 127 then p5 * 2 ^ (127 - l) else p5 / 2 ^ (l - 127)

def u64le (x : Nat) : List UInt8 := (List.range 8).map fun i => UInt8.ofNat (x / 256 ^ i % 256)

def tables : List UInt8 :=
  let p10 := (List.range 23).flatMap fun i => u64le (Float.ofNat (10 ^ i)).toBits.toNat
  let p5 := (List.range POW5_COUNT).flatMap fun i =>
    let v := pow5 (POW5_MIN + Int.ofNat i)
    u64le (v / 2 ^ 64) ++ u64le (v % 2 ^ 64)
  let pi := (List.range 9).flatMap fun i => u64le (10 ^ i)
  zeros (POW10 - 0) ++ p10 ++ zeros (POWI - POW10 - p10.length) ++ pi ++ zeros (POW5 - POWI - pi.length) ++ p5

-- the table is fast_float's: its first entry, and 5^0
#guard pow5 (-342) / 2 ^ 64 == 0xeef453d6923bd65a ∧ pow5 (-342) % 2 ^ 64 == 0x113faa2906a13b3f
#guard pow5 0 == 2 ^ 127

section
variable {V : ClifTy → Type} {L : List ClifTy → List ClifTy → Type}

def at_ (off : Nat) : Prog V L (V .i64) := do iaddImm (← basePtr) off
def c (n : Int) : Prog V L (V .i64) := iconst64 n
def ch (s : Char) : Prog V L (V .i64) := iconst64 s.toNat
def b2i (x : V .i8) : Prog V L (V .i64) := do select x (← c 1) (← c 0)
def isDigit (x : V .i64) : Prog V L (V .i64) := do b2i (← icmp .ult (← isub x (← ch '0')) (← c 10))

/-- A tape word: tag `t` in the top byte, `payload` in the low 56 bits. -/
def word (t : Char) (payload : V .i64) : Prog V L (V .i64) := do
  bor (← c (t.toNat * 2 ^ 56)) payload

/-- The tape's word `i` set. -/
def putTape (i v : V .i64) : Prog V L Unit := do
  storeI64 v (← iadd (← at_ TAPE) (← ishlImm i 3))

/-- A vector of byte `b`. -/
def splat8 (b : Nat) : Prog V L (V .i8x16) := do splat .i8x16 (← ireduce .i8 (← c b))

/-- The lanes of `v` equal to any of `bs`, a bit each. -/
def anyOf (v : V .i8x16) (bs : List Char) : Prog V L (V .i64) := do
  let mut m ← c 0
  for b in bs do
    m ← bor m (← uextend64 (← vhighBits (← icmp .eq v (← splat8 b.toNat))))
  pure m

/-- Whether byte `b` is JSON whitespace (space, tab, newline, return). -/
def isWs (b : V .i64) : Prog V L (V .i64) := do
  bor (← bor (← b2i (← icmp .eq b (← ch ' '))) (← b2i (← icmp .eq b (← ch '\n'))))
    (← bor (← b2i (← icmp .eq b (← ch '\r'))) (← b2i (← icmp .eq b (← ch '\t'))))

/-- `p` past JSON's whitespace, at most `e`: 16 bytes at a time while 16
    remain, then a byte at a time. -/
def skipWsRun (p e : V .i64) : Prog V L (V .i64) := do
  let r ← wloopL %[p]
    (head := fun _ cs => do pure (contIfULt cs.head e, %[cs.head], ()))
    (body := fun out cs _ => do
      when .ule (← iaddImm cs.head 16) e do
        let v ← load { ty := .i8x16 } cs.head
        let other ← bxor (← anyOf v [' ', '\n', '\r', '\t']) (← c 0xFFFF)
        when .eq other (← c 0) do continueWith out %[← iaddImm cs.head 16]
        brk out %[← iadd cs.head (← ctz other)]
      when .eq (← isWs (← uload8_64 cs.head)) (← c 0) do brk out %[cs.head]
      pure %[← iaddImm cs.head 1])
  pure r.head

/-- `p` past JSON's whitespace, at most `e`: unmoved where the byte there is
    past a space (the usual case, tested alone), else `skipWsRun`. -/
def skipWs (p e : V .i64) : Prog V L (V .i64) := do
  let r ← ifte (jTys := [.i64]) .ult p e
    (thn := do
      let near ← icmp .ugt (← uload8_64 p) (← ch ' ')
      ifte .ne (← b2i near) (← c 0) (thn := do pure %[p]) (els := do pure %[← skipWsRun p e]))
    (els := do pure %[p])
  pure r.head

/-- A hex digit's value, or 16 where `x` is not one. -/
def hexVal (x : V .i64) : Prog V L (V .i64) := do
  let d ← isub x (← ch '0')
  let lo ← isub (← bor x (← c 0x20)) (← ch 'a')
  let isD ← icmp .ult d (← c 10)
  let isH ← icmp .ult lo (← c 6)
  select isD d (← select isH (← iaddImm lo 10) (← c 16))

/-- Four hex digits at `p` (all before `e`): their value, or past `0xFFFF`
    where one is not a hex digit. -/
def hex4 (p e : V .i64) : Prog V L (V .i64) := do
  let room ← icmp .ule (← iaddImm p 4) e
  let r ← ifte (jTys := [.i64]) .ne (← b2i room) (← c 0)
    (thn := do
      let mut v ← c 0
      let mut bad ← c 0
      for k in [0, 1, 2, 3] do
        let h ← hexVal (← uload8_64 (← iaddImm p k))
        bad ← bor bad (← b2i (← icmp .eq h (← c 16)))
        v ← bor (← ishlImm v 4) (← band h (← c 15))
      pure %[← select (← icmp .ne bad (← c 0)) (← c 0x10000) v])
    (els := do pure %[← c 0x10000])
  pure r.head

/-- Code point `u` (below `0x110000`) as UTF-8 at `q`: the bytes written. -/
def utf8 (u q : V .i64) : Prog V L (V .i64) := do
  let cont (x : V .i64) (shift : Nat) : Prog V L (V .i64) := do
    bor (← band (← ushrImm x shift) (← c 0x3F)) (← c 0x80)
  let r ← ifte (jTys := [.i64]) .ult u (← c 0x80)
    (thn := do istore8 u q; pure %[← c 1])
    (els := ifte .ult u (← c 0x800)
      (thn := do
        istore8 (← bor (← ushrImm u 6) (← c 0xC0)) q
        istore8 (← cont u 0) (← iaddImm q 1)
        pure %[← c 2])
      (els := ifte .ult u (← c 0x10000)
        (thn := do
          istore8 (← bor (← ushrImm u 12) (← c 0xE0)) q
          istore8 (← cont u 6) (← iaddImm q 1)
          istore8 (← cont u 0) (← iaddImm q 2)
          pure %[← c 3])
        (els := do
          istore8 (← bor (← ushrImm u 18) (← c 0xF0)) q
          istore8 (← cont u 12) (← iaddImm q 1)
          istore8 (← cont u 6) (← iaddImm q 2)
          istore8 (← cont u 0) (← iaddImm q 3)
          pure %[← c 4])))
  pure r.head

/-- **A string**, `p` just past its opening quote: its unescaped bytes after
    a `u32` length at string offset `sp`. Answers where the text resumes (past
    the closing quote), the string offset past it, and 0, or 1 where it is
    not a JSON string or the buffer might not hold it: its bytes are at most
    the rest of the text, and a 16-byte copy may write 16 past them. -/
def string (p e sp : V .i64) : Prog V L (V .i64 × V .i64 × V .i64) := do
  let strs ← at_ STRS
  let start ← iaddImm (← iadd strs sp) 4
  let zero ← c 0
  let one ← c 1
  let room ← icmp .ule (← iaddImm (← iadd sp (← isub e p)) 20) (← c STR_BYTES)
  -- carried: text position, output position
  let r ← wloopL %[p, start]
    (head := fun _ cs => do
      let more ← band (← b2i (← icmp .ult cs.head e)) (← b2i room)
      pure (contIf .ne more zero, %[cs.head, cs.snd, one], ()))
    (body := fun out cs _ => do
      let q := cs.head
      let o := cs.snd
      -- 16 bytes at a time up to a quote, a backslash or a control byte
      -- (copied whole: the buffer has room past its end)
      when .ule (← iaddImm q 16) e do
        let v ← load { ty := .i8x16 } q
        let ctl ← uextend64 (← vhighBits (← icmp .ult v (← splat8 0x20)))
        let m ← bor (← anyOf v ['"', '\\']) ctl
        storeUnaligned v o
        when .eq m zero do continueWith out %[← iaddImm q 16, ← iaddImm o 16]
        let j ← ctz m
        -- the closing quote: the string ends here
        when .eq (← uload8_64 (← iadd q j)) (← ch '"') do brk out %[← iadd (← iadd q j) one, ← iadd o j, zero]
        when .ne j zero do continueWith out %[← iadd q j, ← iadd o j]
      let b ← uload8_64 q
      when .eq b (← ch '"') do brk out %[← iaddImm q 1, o, zero]
      when .ult b (← c 0x20) do brk out %[q, o, one]
      let _ ← ifte (jTys := []) .ne b (← ch '\\')
        (thn := do istore8 b o; continueWith out %[← iaddImm q 1, ← iaddImm o 1])
        (els := do
          let q1 ← iaddImm q 1
          when .uge q1 e do brk out %[q, o, one]
          let x ← uload8_64 q1
          let simple (from_ to_ : Char) (k : Prog V L Unit) : Prog V L Unit := do
            when .eq x (← ch from_) do
              istore8 (← ch to_) o
              k
          simple '"' '"' (continueWith out %[← iaddImm q 2, ← iaddImm o 1])
          simple '\\' '\\' (continueWith out %[← iaddImm q 2, ← iaddImm o 1])
          simple '/' '/' (continueWith out %[← iaddImm q 2, ← iaddImm o 1])
          simple 'b' (Char.ofNat 8) (continueWith out %[← iaddImm q 2, ← iaddImm o 1])
          simple 'f' (Char.ofNat 12) (continueWith out %[← iaddImm q 2, ← iaddImm o 1])
          simple 'n' '\n' (continueWith out %[← iaddImm q 2, ← iaddImm o 1])
          simple 'r' '\r' (continueWith out %[← iaddImm q 2, ← iaddImm o 1])
          simple 't' '\t' (continueWith out %[← iaddImm q 2, ← iaddImm o 1])
          when .ne x (← ch 'u') do brk out %[q, o, one]
          let u ← hex4 (← iaddImm q 2) e
          when .ugt u (← c 0xFFFF) do brk out %[q, o, one]
          -- a high surrogate takes the low one after it
          let isHigh ← icmp .eq (← band u (← c 0xFC00)) (← c 0xD800)
          let r2 ← ifte (jTys := [.i64, .i64]) .ne (← b2i isHigh) zero
            (thn := do
              let q6 ← iaddImm q 6
              let ok ← band (← b2i (← icmp .ule (← iaddImm q6 6) e))
                (← band (← b2i (← icmp .eq (← uload8_64 (← select (← icmp .ult q6 e) q6 q)) (← ch '\\')))
                  (← b2i (← icmp .eq (← uload8_64 (← select (← icmp .ult (← iaddImm q6 1) e) (← iaddImm q6 1) q)) (← ch 'u'))))
              when .eq ok zero do brk out %[q, o, one]
              let lo ← hex4 (← iaddImm q6 2) e
              when .ne (← band lo (← c 0xFC00)) (← c 0xDC00) do brk out %[q, o, one]
              let cp ← iadd (← c 0x10000) (← bor (← ishlImm (← band u (← c 0x3FF)) 10) (← band lo (← c 0x3FF)))
              pure %[cp, ← iaddImm q 12])
            (els := do
              -- a lone low surrogate is not text
              when .eq (← band u (← c 0xFC00)) (← c 0xDC00) do brk out %[q, o, one]
              pure %[u, ← iaddImm q 6])
          let n ← utf8 r2.head o
          continueWith out %[r2.snd, ← iadd o n])
      pure %[q, o])
  -- the string's length: where the output stopped, less its start
  storeI32 (← ireduce32 (← isub r.snd start)) (← iadd strs sp)
  pure (r.head, ← isub r.snd strs, r.thd)


/-! ## The slow path: a decimal, shifted

    fast_float's "simple decimal conversion", for every number the fast paths
    cannot round: more than 19 digits, a subnormal result, or a product
    Eisel–Lemire leaves undecided. The number's digits (at most
    `MAX_DIGITS`, a flag for any dropped) and its decimal point are shifted by
    up to 60 bits at a time until the binary exponent is known, then rounded
    to 53 bits, ties to even. Exact for every input; slow, and rare. -/

def getW (off : Nat) : Prog V L (V .i64) := do load64 (← at_ off)
def setW (off : Nat) (v : V .i64) : Prog V L Unit := do storeI64 v (← at_ off)
def digAt (buf : Nat) (i : V .i64) : Prog V L (V .i64) := do uload8_64 (← iadd (← at_ buf) i)
def setDig (buf : Nat) (i v : V .i64) : Prog V L Unit := do istore8 v (← iadd (← at_ buf) i)
def DP_RANGE : Int := 2047

/-- Trailing zero digits dropped. -/
def decTrim : Prog V L Unit := do
  let zero ← c 0
  let r ← wloop %[← getW DEC_ND] (fun cs => do
      let last ← digAt DEC (← select (← icmp .ugt cs.head zero) (← iaddImm cs.head (-1)) zero)
      let more ← band (← b2i (← icmp .ugt cs.head zero)) (← b2i (← icmp .eq last zero))
      pure (contIf .ne more zero, %[cs.head], ()))
    (fun cs _ => do pure %[← iaddImm cs.head (-1)])
  setW DEC_ND r.head

/-- The decimal divided by `2^shift` (`shift ≤ 60`). -/
def decRight (shift : V .i64) : Prog V L Unit := do
  let zero ← c 0
  let nd ← getW DEC_ND
  -- the leading digits read until they make at least `2^shift`, zeros past the end
  let r ← wloopL %[zero, zero]
    (head := fun _ cs => do
      let big ← icmp .ne (← ushr cs.snd shift) zero
      let empty ← band (← b2i (← icmp .uge cs.head nd)) (← b2i (← icmp .eq cs.snd zero))
      let stop ← bor (← b2i big) empty
      pure (contIf .eq stop zero, %[cs.head, cs.snd], ()))
    (body := fun _ cs _ => do
      let dg ← select (← icmp .ult cs.head nd) (← digAt DEC (← select (← icmp .ult cs.head nd) cs.head zero)) zero
      pure %[← iaddImm cs.head 1, ← iadd (← imul cs.snd (← c 10)) dg])
  let ri0 := r.head
  let n0 := r.snd
  when .ne n0 zero do
    let dp ← isub (← getW DEC_DP) (← iaddImm ri0 (-1))
    setW DEC_DP dp
    let _ ← ifte (jTys := []) .slt dp (← c (-DP_RANGE))
      (thn := do
        setW DEC_ND zero
        setW DEC_DP zero
        setW DEC_TR zero
        pure %[])
      (els := do
        let mask ← iaddImm (← ishl (← c 1) shift) (-1)
        let a ← wloop %[ri0, zero, n0] (fun cs => pure (contIfULt cs.head nd, %[cs.snd, cs.thd], ()))
          (fun cs _ => do
            let nw ← ushr cs.thd shift
            let n' ← iadd (← imul (← band cs.thd mask) (← c 10)) (← digAt DEC cs.head)
            setDig DEC cs.snd nw
            pure %[← iaddImm cs.head 1, ← iaddImm cs.snd 1, n'])
        let b ← wloop %[a.head, a.snd] (fun cs => pure (contIf .ne cs.snd zero, %[cs.head], ()))
          (fun cs _ => do
            let nw ← ushr cs.snd shift
            let n' ← imul (← band cs.snd mask) (← c 10)
            let _ ← ifte (jTys := [.i64]) .ult cs.head (← c MAX_DIGITS)
              (thn := do setDig DEC cs.head nw; pure %[zero])
              (els := do
                when .ne nw zero do setW DEC_TR (← c 1)
                pure %[zero])
            pure %[← select (← icmp .ult cs.head (← c MAX_DIGITS)) (← iaddImm cs.head 1) cs.head, n'])
        setW DEC_ND b.head
        decTrim
        pure %[])

/-- The decimal multiplied by `2^shift` (`shift ≤ 60`): each digit from the
    last, its carry on, into the scratch copy in reverse; then back, its new
    leading digits moving the decimal point. -/
def decLeft (shift : V .i64) : Prog V L Unit := do
  let zero ← c 0
  let ten ← c 10
  let nd ← getW DEC_ND
  when .ne nd zero do
    let a ← wloop %[nd, zero, zero] (fun cs => pure (contIf .ne cs.head zero, %[cs.snd, cs.thd], ()))
      (fun cs _ => do
        let i ← iaddImm cs.head (-1)
        let v ← iadd (← ishl (← digAt DEC i) shift) cs.thd
        setDig DEC_TMP cs.snd (← urem v ten)
        pure %[i, ← iaddImm cs.snd 1, ← udiv v ten])
    let b ← wloop %[a.head, a.snd] (fun cs => pure (contIf .ne cs.snd zero, %[cs.head], ()))
      (fun cs _ => do
        setDig DEC_TMP cs.head (← urem cs.snd ten)
        pure %[← iaddImm cs.head 1, ← udiv cs.snd ten])
    let k := b.head
    setW DEC_DP (← iadd (← getW DEC_DP) (← isub k nd))
    let keep ← umin k (← c MAX_DIGITS)
    forLoop k fun j => do
      let dg ← digAt DEC_TMP (← isub (← iaddImm k (-1)) j)
      let _ ← ifte (jTys := []) .ult j keep
        (thn := do setDig DEC j dg; pure %[])
        (els := do
          when .ne dg zero do setW DEC_TR (← c 1)
          pure %[])
    setW DEC_ND keep
    decTrim

/-- The decimal's integer part, rounded half to even (`2^64 - 1` past 18
    digits). -/
def decRound : Prog V L (V .i64) := do
  let zero ← c 0
  let nd ← getW DEC_ND
  let dp ← getW DEC_DP
  let r ← ifte (jTys := [.i64]) .ne (← bor (← b2i (← icmp .eq nd zero)) (← b2i (← icmp .slt dp zero))) zero
    (thn := do pure %[zero])
    (els := ifte .sgt dp (← c 18) (thn := do pure %[← c (-1)]) (els := do
      let n ← forLoopAcc dp zero fun i acc => do
        let dg ← select (← icmp .ult i nd) (← digAt DEC (← select (← icmp .ult i nd) i zero)) zero
        iadd (← imul acc (← c 10)) dg
      let inside ← icmp .ult dp nd
      let at' ← digAt DEC (← select inside dp zero)
      let up0 ← band (← b2i inside) (← b2i (← icmp .uge at' (← c 5)))
      let tie ← band (← b2i inside) (← band (← b2i (← icmp .eq at' (← c 5))) (← b2i (← icmp .eq (← iaddImm dp 1) nd)))
      let prevOdd ← band (← b2i (← icmp .sgt dp zero))
        (← band (← digAt DEC (← select (← icmp .sgt dp zero) (← iaddImm dp (-1)) zero)) (← c 1))
      let up ← select (← icmp .ne tie zero) (← bor (← getW DEC_TR) prevOdd) up0
      pure %[← iadd n up]))
  pure r.head

/-- The shift that takes a decimal point of `n` (`1 ≤ n`) most of the way:
    fast_float's table, 60 past it. -/
def decShift (n : V .i64) : Prog V L (V .i64) := do
  let table : List Nat := [0, 3, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39, 43, 46, 49, 53, 56, 59]
  let mut v ← c 60
  for (s, i) in table.zipIdx do
    v ← select (← icmp .eq n (← c i)) (← c s) v
  pure v

/-- **The double nearest the digits `p … e`** (an integer part and a
    fraction, `.` between) times `10^exp`, its bits without the sign. -/
def slowFloat (p e exp : V .i64) : Prog V L (V .i64) := do
  let zero ← c 0
  let one ← c 1
  setW DEC_ND zero; setW DEC_DP zero; setW DEC_TR zero
  -- the digits, leading zeros dropped (a fraction's moving the point)
  let _ ← wloop %[p, zero] (fun cs => pure (contIfULt cs.head e, %[], ()))
    (fun cs _ => do
      let b ← uload8_64 cs.head
      let isDot ← icmp .eq b (← ch '.')
      let frac ← bor cs.snd (← b2i isDot)
      when .eq (← b2i isDot) zero do
        let dg ← isub b (← ch '0')
        let nd ← getW DEC_ND
        let lead ← band (← b2i (← icmp .eq nd zero)) (← b2i (← icmp .eq dg zero))
        let _ ← ifte (jTys := []) .ne frac zero
          (thn := do
            when .ne lead zero do setW DEC_DP (← iaddImm (← getW DEC_DP) (-1))
            pure %[])
          (els := do
            when .eq lead zero do setW DEC_DP (← iaddImm (← getW DEC_DP) 1)
            pure %[])
        when .eq lead zero do
          let _ ← ifte (jTys := []) .ult nd (← c MAX_DIGITS)
            (thn := do
              setDig DEC nd dg
              setW DEC_ND (← iaddImm nd 1)
              pure %[])
            (els := do
              when .ne dg zero do setW DEC_TR one
              pure %[])
      pure %[← iaddImm cs.head 1, frac])
  setW DEC_DP (← iadd (← getW DEC_DP) exp)
  decTrim
  let inf ← c 0x7FF0000000000000
  let r ← wloopL %[zero]
    (head := fun _ _ => pure (contIf .eq zero zero, %[zero], ()))
    (body := fun out _ _ => do
      when .eq (← getW DEC_ND) zero do brk out %[zero]
      when .slt (← getW DEC_DP) (← c (-324)) do brk out %[zero]
      when .sge (← getW DEC_DP) (← c 310) do brk out %[inf]
      -- down to below one
      let e1 ← wloopL %[zero]
        (head := fun _ cs => do pure (contIf .sgt (← getW DEC_DP) zero, %[cs.head, zero], ()))
        (body := fun out2 cs _ => do
          let sh ← decShift (← getW DEC_DP)
          decRight sh
          when .slt (← getW DEC_DP) (← c (-DP_RANGE)) do brk out2 %[zero, one]
          pure %[← iadd cs.head sh])
      when .ne e1.snd zero do brk out %[zero]
      -- up to [1/2, 1)
      let e2 ← wloopL %[e1.head]
        (head := fun _ cs => do pure (contIf .sle (← getW DEC_DP) zero, %[cs.head, zero], ()))
        (body := fun out2 cs _ => do
          let dp ← getW DEC_DP
          let d0 ← digAt DEC zero
          when .eq dp zero do
            when .uge d0 (← c 5) do brk out2 %[cs.head, zero]
          let sh ← ifte (jTys := [.i64]) .eq dp zero
            (thn := do pure %[← select (← icmp .ult d0 (← c 2)) (← c 2) one])
            (els := do pure %[← decShift (← isub zero dp)])
          decLeft sh.head
          when .sgt (← getW DEC_DP) (← c DP_RANGE) do brk out2 %[cs.head, one]
          pure %[← isub cs.head sh.head])
      when .ne e2.snd zero do brk out %[inf]
      -- [1, 2), and no lower than the least normal exponent
      let e3 ← wloop %[← iaddImm e2.head (-1)] (fun cs => do pure (contIf .sgt (← c (-1022)) cs.head, %[cs.head], ()))
        (fun cs _ => do
          let sh ← umin (← isub (← c (-1022)) cs.head) (← c 60)
          decRight sh
          pure %[← iadd cs.head sh])
      let ex := e3.head
      when .sge (← iaddImm ex 1023) (← c 0x7FF) do brk out %[inf]
      decLeft (← c 53)
      let m0 ← decRound
      let over ← icmp .uge m0 (← c (2 ^ 53))
      let res ← ifte (jTys := [.i64, .i64]) .ne (← b2i over) zero
        (thn := do
          decRight one
          pure %[← decRound, ← iaddImm ex 1])
        (els := do pure %[m0, ex])
      let m := res.head
      let ex2 := res.snd
      when .sge (← iaddImm ex2 1023) (← c 0x7FF) do brk out %[inf]
      let p2 ← iaddImm ex2 1023
      let p2' ← select (← icmp .ult m (← c (2 ^ 52))) (← iaddImm p2 (-1)) p2
      brk out %[← bor (← band m (← c (2 ^ 52 - 1))) (← ishlImm p2' 52)])
  pure r.head

/-- Eisel–Lemire for `w · 10^q` (`w` not 0, `q` in `-342 … 308`): the
    double's bits without the sign, or `-1` where it cannot round here. -/
def eiselLemire (w q : V .i64) : Prog V L (V .i64) := do
  let lz ← clz w
  let wn ← ishl w lz
  let idx ← ishlImm (← isub q (← c POW5_MIN)) 4
  let t ← iadd (← at_ POW5) idx
  let hi1 ← load64 t
  let lo1 ← load64 (← iaddImm t 8)
  let pHi ← umulhi wn hi1
  let pLo ← imul wn hi1
  -- the product's next 64 bits, where its top 55 bits are not settled
  let mask ← c 0x1FF
  let r ← ifte (jTys := [.i64, .i64]) .eq (← band pHi mask) mask
    (thn := do
      let sHi ← umulhi wn lo1
      let lo ← iadd pLo sHi
      let carry ← b2i (← icmp .ult lo pLo)
      pure %[← iadd pHi carry, lo])
    (els := do pure %[pHi, pLo])
  let hi := r.head
  let lo := r.snd
  let upper ← ushrImm hi 63
  let mant ← ushr hi (← iaddImm upper 9)
  -- the binary exponent, biased: ((152170 + 65536) · q >> 16) + 63 + upper - lz + 1023
  let pw ← iadd (← iadd (← iaddImm (← sshr (← imul q (← c (152170 + 65536))) (← c 16)) (63 + 1023)) upper) (← isub (← c 0) lz)
  let bad ← bor (← b2i (← icmp .sle pw (← c 0)))
    (← band (← b2i (← icmp .eq lo (← c (-1))))
      (← bor (← b2i (← icmp .slt q (← c (-27)))) (← b2i (← icmp .sgt q (← c 55)))))
  -- an exact halfway case rounds to even
  let halfway ← band (← band (← b2i (← icmp .ule lo (← c 1))) (← b2i (← icmp .sge q (← c (-4)))))
    (← band (← b2i (← icmp .sle q (← c 23))) (← b2i (← icmp .eq (← band mant (← c 3)) (← c 1))))
  let exact ← b2i (← icmp .eq (← ishl mant (← iaddImm upper 9)) hi)
  let m1 ← select (← icmp .ne (← band halfway exact) (← c 0)) (← band mant (← c (-2))) mant
  let m2 ← ushrImm (← iadd m1 (← band m1 (← c 1))) 1
  let over ← icmp .uge m2 (← c (2 ^ 53))
  let m3 ← select over (← c (2 ^ 52)) m2
  let pw2 ← select over (← iaddImm pw 1) pw
  let bits ← bor (← band m3 (← c (2 ^ 52 - 1))) (← ishlImm pw2 52)
  let inf ← icmp .sge pw2 (← c 0x7FF)
  let ans ← select inf (← c 0x7FF0000000000000) bits
  select (← icmp .ne bad (← c 0)) (← c (-1)) ans

/-- A number's text read: its sign, its significant digits' value `w`
    (exact while there are at most 19), how many digits it has and how many
    follow a point, whether it is an integer (no point, no exponent), its
    exponent, where its digits end and where the number does, whether it is
    not a JSON number, and whether `w` overflowed. -/
structure Scan (V : ClifTy → Type) where
  neg : V .i64
  w : V .i64
  nd : V .i64
  frac : V .i64
  isInt : V .i64
  ev : V .i64
  mEnd : V .i64
  fin : V .i64
  bad : V .i64
  ov : V .i64

abbrev SCAN : List ClifTy := [.i64, .i64, .i64, .i64, .i64, .i64, .i64, .i64, .i64, .i64]

def Scan.toVals {V} (s : Scan V) : Vals V SCAN :=
  %[s.neg, s.w, s.nd, s.frac, s.isInt, s.ev, s.mEnd, s.fin, s.bad, s.ov]

def Scan.ofVals {V} : Vals V SCAN → Scan V
  | .cons a (.cons b (.cons c (.cons d (.cons e (.cons f (.cons g (.cons h (.cons i (.cons j .nil))))))))) =>
    ⟨a, b, c, d, e, f, g, h, i, j⟩

/-- The 8 bytes `x` read as decimal digits: how many lead it (0 to 8; fast_float's
    SWAR, a byte `0x33` exactly where a digit is, no carry reaching a byte
    before the first that is not one). -/
def leadDigits (x : V .i64) : Prog V L (V .i64) := do
  let hiNib ← c (0xF0F0F0F0F0F0F0F0 - 2 ^ 64)
  let cls ← bxor (← bor (← band x hiNib)
      (← ushrImm (← band (← iadd x (← c 0x0606060606060606)) hiNib) 4)) (← c 0x3333333333333333)
  ushrImm (← select (← icmp .eq cls (← c 0)) (← c 64) (← ctz cls)) 3

/-- The value of the first `k` digits of `x` (`1 ≤ k ≤ 8`): shifted to the
    top, zero digits below them, then fast_float's eight-digit sum. -/
def digitsValue (x k : V .i64) : Prog V L (V .i64) := do
  let v0 ← ishl (← isub x (← c 0x3030303030303030)) (← isub (← c 64) (← ishlImm k 3))
  let v1 ← iadd (← imul v0 (← c 10)) (← ushrImm v0 8)
  let mask ← c 0x000000FF000000FF
  ushrImm (← iadd (← imul (← band v1 mask) (← c 0x000F424000000064))
    (← imul (← band (← ushrImm v1 16) mask) (← c 0x0000271000000001))) 32

/-- A number's text read from `p` (a digit or `-`), every byte checked
    against the text's end `e`. -/
def scanNumber (p e : V .i64) : Prog V L (Scan V) := do
  let zero ← c 0
  let one ← c 1
  let byteAt (q : V .i64) : Prog V L (V .i64) := do
    select (← icmp .ult q e) (← uload8_64 (← select (← icmp .ult q e) q p)) zero
  let neg ← b2i (← icmp .eq (← byteAt p) (← ch '-'))
  let p0 ← iadd p neg
  let first ← byteAt p0
  let leadZero ← b2i (← icmp .eq first (← ch '0'))
  -- digits: the value (checked against u64's overflow), how many, where they end
  let digits (from_ w0 n0 : V .i64) : Prog V L (Vals V [.i64, .i64, .i64, .i64]) := do
    wloopL %[from_, w0, n0, zero]
      (head := fun _ cs => do pure (contIfULt cs.head e, %[cs.head, cs.snd, cs.thd, cs.fth], ()))
      (body := fun out cs _ => do
        -- up to eight digits at once (fast_float's SWAR): the digits leading
        -- an 8-byte read, the rest shifted out, while the value stays within
        -- 19 digits
        when .ule (← iaddImm cs.head 8) e do
          let x ← load64 cs.head
          let hiNib ← c (0xF0F0F0F0F0F0F0F0 - 2 ^ 64)
          -- a byte 0x33 exactly where a digit is (no carry reaches a byte
          -- before the first that is not one)
          let cls ← bxor (← bor (← band x hiNib)
              (← ushrImm (← band (← iadd x (← c 0x0606060606060606)) hiNib) 4)) (← c 0x3333333333333333)
          let k ← ushrImm (← select (← icmp .eq cls zero) (← c 64) (← ctz cls)) 3
          when .ule (← iadd cs.thd k) (← c 19) do
            when .eq k zero do brk out %[cs.head, cs.snd, cs.thd, cs.fth]
            let v0 ← ishl (← isub x (← c 0x3030303030303030)) (← isub (← c 64) (← ishlImm k 3))
            let v1 ← iadd (← imul v0 (← c 10)) (← ushrImm v0 8)
            let mask ← c 0x000000FF000000FF
            let v ← ushrImm (← iadd (← imul (← band v1 mask) (← c 0x000F424000000064))
              (← imul (← band (← ushrImm v1 16) mask) (← c 0x0000271000000001))) 32
            let w ← iadd (← imul cs.snd (← load64 (← iadd (← at_ POWI) (← ishlImm k 3)))) v
            let r := %[← iadd cs.head k, w, ← iadd cs.thd k, cs.fth]
            when .ult k (← c 8) do brk out r
            continueWith out r
        let d ← isub (← uload8_64 cs.head) (← ch '0')
        when .uge d (← c 10) do brk out %[cs.head, cs.snd, cs.thd, cs.fth]
        let w := cs.snd
        let ov ← bor cs.fth (← bor (← b2i (← icmp .ugt w (← c 1844674407370955161)))
          (← band (← b2i (← icmp .eq w (← c 1844674407370955161))) (← b2i (← icmp .ugt d (← c 5)))))
        pure %[← iaddImm cs.head 1, ← iadd (← imul w (← c 10)) d, ← iaddImm cs.thd 1, ov])
  let ip ← digits p0 zero zero
  let afterInt := ip.head
  let intDigits := ip.thd
  let bad0 ← bor (← b2i (← icmp .eq intDigits zero)) (← band leadZero (← b2i (← icmp .ugt intDigits one)))
  -- a fraction
  let dot ← b2i (← icmp .eq (← byteAt afterInt) (← ch '.'))
  let fp ← ifte (jTys := [.i64, .i64, .i64, .i64]) .ne dot zero
    (thn := do digits (← iaddImm afterInt 1) ip.snd ip.thd)
    (els := do pure %[afterInt, ip.snd, ip.thd, ip.fth])
  let fracDigits ← isub fp.thd intDigits
  let bad1 ← band dot (← b2i (← icmp .eq fracDigits zero))
  -- an exponent
  let eb ← bor (← byteAt fp.head) (← c 0x20)
  let hasE ← b2i (← icmp .eq eb (← ch 'e'))
  let ep ← ifte (jTys := [.i64, .i64, .i64]) .ne hasE zero
    (thn := do
      let s ← iaddImm fp.head 1
      let sb ← byteAt s
      let minus ← b2i (← icmp .eq sb (← ch '-'))
      let signed ← bor minus (← b2i (← icmp .eq sb (← ch '+')))
      let s1 ← iadd s signed
      let er ← wloopL %[s1, zero]
        (head := fun _ cs => do pure (contIfULt cs.head e, %[cs.head, cs.snd], ()))
        (body := fun out cs _ => do
          let d ← isub (← uload8_64 cs.head) (← ch '0')
          when .uge d (← c 10) do brk out %[cs.head, cs.snd]
          pure %[← iaddImm cs.head 1, ← umin (← iadd (← imul cs.snd (← c 10)) d) (← c 100000)])
      let ev ← select (← icmp .ne minus zero) (← isub zero er.snd) er.snd
      pure %[er.head, ev, ← b2i (← icmp .eq er.head s1)])
    (els := do pure %[fp.head, zero, zero])
  let bad ← bor (← bor bad0 bad1) ep.thd
  let isInt ← b2i (← icmp .eq (← bor dot hasE) zero)
  pure ⟨neg, fp.snd, fp.thd, fracDigits, isInt, ep.snd, fp.head, ep.head, bad, fp.fth⟩

/-- The commonest numbers read without a check against the text's end, at
    least 40 bytes on: at most 15 integer digits, at most 15 after a point,
    no exponent (more than 19 digits in all go on to `slowFloat`). The scan, and 1; or 0 where the number is not of that shape
    (`scanNumber` reads it). -/
def scanShort (p : V .i64) : Prog V L (Scan V × V .i64) := do
  let zero ← c 0
  let neg ← b2i (← icmp .eq (← uload8_64 p) (← ch '-'))
  let p0 ← iadd p neg
  let x1 ← load64 p0
  let k1 ← leadDigits x1
  -- no digit, a leading zero before another digit, or 8 or more
  let lead0 ← band (← b2i (← icmp .eq (← band x1 (← c 0xFF)) (← ch '0'))) (← b2i (← icmp .ugt k1 (← c 1)))
  let w0 ← digitsValue x1 (← select (← icmp .eq k1 zero) (← c 1) k1)
  -- a second chunk where the first was all digits
  let ip ← ifte (jTys := [.i64, .i64, .i64]) .eq k1 (← c 8)
    (thn := do
      let x ← load64 (← iaddImm p0 8)
      let k ← leadDigits x
      let w ← iadd (← imul w0 (← load64 (← iadd (← at_ POWI) (← ishlImm k 3))))
        (← digitsValue x (← select (← icmp .eq k zero) (← c 1) k))
      pure %[← select (← icmp .eq k zero) w0 w, ← iaddImm k 8, ← b2i (← icmp .eq k (← c 8))])
    (els := do pure %[w0, k1, ← b2i (← icmp .eq k1 zero)])
  let w1 := ip.head
  let k1 := ip.snd
  let no ← bor ip.thd lead0
  let q1 ← iadd p0 k1
  let dot ← b2i (← icmp .eq (← uload8_64 q1) (← ch '.'))
  let r ← ifte (jTys := [.i64, .i64, .i64, .i64]) .ne dot zero
    (thn := do
      let f ← iaddImm q1 1
      let x2 ← load64 f
      let k2 ← leadDigits x2
      let w2 ← iadd (← imul w1 (← load64 (← iadd (← at_ POWI) (← ishlImm k2 3))))
        (← digitsValue x2 (← select (← icmp .eq k2 zero) (← c 1) k2))
      -- a second chunk where the first was all digits
      ifte .eq k2 (← c 8)
        (thn := do
          let x3 ← load64 (← iaddImm f 8)
          let k3 ← leadDigits x3
          let w3 ← iadd (← imul w2 (← load64 (← iadd (← at_ POWI) (← ishlImm k3 3))))
            (← digitsValue x3 (← select (← icmp .eq k3 zero) (← c 1) k3))
          let w3' ← select (← icmp .eq k3 zero) w2 w3
          pure %[w3', ← iaddImm k3 8, ← iadd (← iaddImm f 8) k3, ← b2i (← icmp .eq k3 (← c 8))])
        (els := do pure %[w2, k2, ← iadd f k2, ← b2i (← icmp .eq k2 zero)]))
    (els := do pure %[w1, zero, q1, zero])
  let fin := r.thd
  let hasE ← b2i (← icmp .eq (← bor (← uload8_64 fin) (← c 0x20)) (← ch 'e'))
  let ok ← b2i (← icmp .eq (← bor (← bor no r.fth) hasE) zero)
  pure (⟨neg, r.head, ← iadd k1 r.snd, r.snd, ← b2i (← icmp .eq dot zero), zero, fin, fin, zero, zero⟩, ok)

/-- **A number** at `p` (a digit or `-`): its tag word and value word, where
    the text resumes, and 0, or 1 where it is not a JSON number or cannot be
    parsed here. -/
def number (p e : V .i64) : Prog V L (V .i64 × V .i64 × V .i64 × V .i64) := do
  let zero ← c 0
  let room ← icmp .ule (← iaddImm p 40) e
  -- the short scan where there is room, its answer kept where it read the number
  let f ← ifte (jTys := .i64 :: SCAN) .ne (← b2i room) zero
    (thn := do
      let (sc, ok) ← scanShort p
      pure (ok ::ᵥ sc.toVals))
    (els := do pure %[zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero])
  let sc ← ifte (jTys := SCAN) .ne f.head zero
    (thn := do pure f.tail)
    (els := do pure (← scanNumber p e).toVals)
  let ⟨neg, w, nd, fracDigits, isInt, ev, mEnd, fin, bad, ov⟩ := Scan.ofVals sc
  let p0 ← iadd p neg
  let r ← ifte (jTys := [.i64, .i64, .i64]) .ne isInt zero
    (thn := do
      -- an integer: int64, a uint64 past it, or (negative past int64) refused
      let fitsI ← b2i (← icmp .ule w (← c (2 ^ 63 - 1)))
      let negOk ← b2i (← icmp .ule w (← c (-(2 ^ 63))))
      let tag ← select (← icmp .ne neg zero) (← word 'l' zero)
        (← select (← icmp .ne fitsI zero) (← word 'l' zero) (← word 'u' zero))
      let v ← select (← icmp .ne neg zero) (← isub zero w) w
      let refuse ← bor ov (← band neg (← b2i (← icmp .eq negOk zero)))
      pure %[tag, v, refuse])
    (els := do
      -- a double: w · 10^q, w from at most 19 digits
      let q ← iadd ev (← isub zero fracDigits)
      let sign ← ishlImm neg 63
      let tooMany ← bor ov (← b2i (← icmp .ugt nd (← c 19)))
      let d ← ifte (jTys := [.i64, .i64]) .eq w zero
        (thn := do pure %[sign, zero])
        (els := ifte .slt q (← c (POW5_MIN))
          (thn := do pure %[sign, zero])
          (els := ifte .sgt q (← c 308)
            (thn := do pure %[← bor sign (← c 0x7FF0000000000000), zero])
            (els := do
              let small ← band (← b2i (← icmp .ule w (← c (2 ^ 53))))
                (← band (← b2i (← icmp .sge q (← c (-22)))) (← b2i (← icmp .sle q (← c 22))))
              let r ← ifte (jTys := [.i64, .i64]) .ne small zero
                (thn := do
                  -- Clinger: both exact, so one rounding
                  let wf ← fcvtFromSint .f64 w
                  let aq ← select (← icmp .slt q zero) (← isub zero q) q
                  let pw ← load { ty := .f64 } (← iadd (← at_ POW10) (← ishlImm aq 3))
                  let f ← ifte (jTys := [.f64]) .slt q zero
                    (thn := do pure %[← fdiv wf pw]) (els := do pure %[← fmul wf pw])
                  pure %[← bor sign (← bitcast .i64 f.head), zero])
                (els := do
                  let b ← eiselLemire w q
                  pure %[← bor sign b, ← b2i (← icmp .eq b (← c (-1)))])
              pure %[r.head, r.snd])))
      -- what the fast paths cannot round, the slow path does
      let slow ← bor tooMany d.snd
      let v ← ifte (jTys := [.i64]) .ne slow zero
        (thn := do pure %[← bor sign (← slowFloat p0 mEnd ev)])
        (els := do pure %[d.head])
      pure %[← word 'd' zero, v.head, zero])
  pure (r.head, r.snd, fin, ← bor bad r.thd)

/-- The literal `lit` at `p`: 1 where the text has it. -/
def literal (p e : V .i64) (lit : String) : Prog V L (V .i64) := do
  let n := lit.length
  let ok ← ifte (jTys := [.i64]) .ule (← iaddImm p n) e
    (thn := do
      let mut all ← c 1
      for (ch_, k) in lit.toList.zip (List.range n) do
        all ← band all (← b2i (← icmp .eq (← uload8_64 (← iaddImm p k)) (← ch ch_)))
      pure %[all])
    (els := do pure %[← c 0])
  pure ok.head

/-- **The parse**: the status 0, or one past the byte where the text stops
    being JSON. Each trip round its loop reads one value; a scalar, or a
    container closed as soon as opened, then takes what follows it in the
    same trip: the closes, then a comma and, in an object, the next key and
    its colon. -/
def parseBody : Prog V L (V .i64) := do
  let p0 ← dataPtr
  let e ← iadd p0 (← dataLen)
  let zero ← c 0
  let one ← c 1
  let stack ← at_ STACK
  let tape ← at_ TAPE
  let put (i v : V .i64) : Prog V L Unit := do storeI64 v (← iadd tape (← ishlImm i 3))
  -- carried: position, tape words, string bytes, depth; a value next
  let r ← wloopL %[p0, zero, zero, zero]
    (head := fun _ _ => do pure (contIf .eq zero zero, %[zero, zero, zero], ()))
    (body := fun out cs _ => do
      let fail (p : V .i64) : Prog V L Unit := do brk out %[← iaddImm (← isub p p0) 1, zero, zero]
      let full (tp p : V .i64) : Prog V L Unit := do when .uge tp (← c (TAPE_WORDS - 2)) do fail p
      -- a key at `p` (after whitespace), its colon, and then a value next
      let key (p tp sp d : V .i64) : Prog V L Unit := do
        let p ← skipWs p e
        when .uge p e do fail p
        full tp p
        when .ne (← uload8_64 p) (← ch '"') do fail p
        let (p1, sp1, err) ← string (← iaddImm p 1) e sp
        when .ne err zero do fail p1
        put tp (← word '"' sp)
        let p2 ← skipWs p1 e
        when .uge p2 e do fail p2
        when .ne (← uload8_64 p2) (← ch ':') do fail p2
        continueWith out %[← iaddImm p2 1, ← iaddImm tp 1, sp1, d]
      let tp := cs.snd
      let sp := cs.thd
      let d := cs.fth
      let p ← skipWs cs.head e
      when .uge p e do fail p
      full tp p
      let b ← uload8_64 p
      let scalar (lit : String) (tag : Char) (k : Prog V L (Vals V [.i64, .i64, .i64])) :
          Prog V L (Vals V [.i64, .i64, .i64]) := do
        ifte .eq b (← ch lit.front)
          (thn := do
            when .eq (← literal p e lit) zero do fail p
            put tp (← word tag zero)
            pure %[← iaddImm p lit.length, ← iaddImm tp 1, sp])
          (els := k)
      -- the value: where the text resumes, the tape and the string bytes
      -- after it
      let v ← ifte (jTys := [.i64, .i64, .i64]) .eq b (← ch '"')
        (thn := do
          let (p1, sp1, err) ← string (← iaddImm p 1) e sp
          when .ne err zero do fail p1
          put tp (← word '"' sp)
          pure %[p1, ← iaddImm tp 1, sp1])
        (els := ifte .ne (← bor (← isDigit b) (← b2i (← icmp .eq b (← ch '-')))) zero
          (thn := do
            let (tag, w, p1, err) ← number p e
            when .ne err zero do fail p
            put tp tag
            put (← iaddImm tp 1) w
            pure %[p1, ← iaddImm tp 2, sp])
          (els := ifte .ne (← bor (← b2i (← icmp .eq b (← ch '{'))) (← b2i (← icmp .eq b (← ch '[')))) zero
            (thn := do
              when .uge d (← c MAX_DEPTH) do fail p
              let obj ← b2i (← icmp .eq b (← ch '{'))
              let p1 ← skipWs (← iaddImm p 1) e
              -- closed at once: both words, nothing pushed
              let shut ← select (← icmp .ne obj zero) (← ch '}') (← ch ']')
              let empty ← band (← b2i (← icmp .ult p1 e))
                (← b2i (← icmp .eq (← uload8_64 (← select (← icmp .ult p1 e) p1 p)) shut))
              when .eq empty zero do
                storeI64 (← bor (← ishlImm tp 1) obj) (← iadd stack (← ishlImm d 3))
                when .ne obj zero do key p1 (← iaddImm tp 1) sp (← iaddImm d 1)
                continueWith out %[p1, ← iaddImm tp 1, sp, ← iaddImm d 1]
              let tp1 ← iaddImm tp 1
              put tp (← bor (← select (← icmp .ne obj zero) (← word '{' zero) (← word '[' zero)) tp1)
              put tp1 (← bor (← select (← icmp .ne obj zero) (← word '}' zero) (← word ']' zero)) tp)
              pure %[← iaddImm p1 1, ← iaddImm tp 2, sp])
            (els := scalar "true" 't' (scalar "false" 'f' (scalar "null" 'n' (do
              fail p
              pure %[p, tp, sp]))))))
      let sp1 := v.thd
      -- after the value: closes, until a comma or the end
      let _ ← wloopL %[v.head, v.snd, d]
        (head := fun _ _ => do pure (contIf .eq zero zero, %[zero], ()))
        (body := fun _ ws _ => do
          let tp := ws.snd
          let d := ws.thd
          let p ← skipWs ws.head e
          -- the end: a whole value and nothing open
          when .uge p e do
            when .ne d zero do fail p
            brk out %[zero, tp, sp1]
          when .eq d zero do fail p
          full tp p
          let top ← load64 (← iadd stack (← ishlImm (← iaddImm d (-1)) 3))
          let isObj ← band top one
          let b ← uload8_64 p
          when .eq b (← ch ',') do
            when .ne isObj zero do key (← iaddImm p 1) tp sp1 d
            continueWith out %[← iaddImm p 1, tp, sp1, d]
          when .ne b (← select (← icmp .ne isObj zero) (← ch '}') (← ch ']')) do fail p
          -- the close of what is open: its open and close words linked
          let openAt ← ushrImm top 1
          put openAt (← bor (← select (← icmp .ne isObj zero) (← word '{' zero) (← word '[' zero)) tp)
          put tp (← bor (← select (← icmp .ne isObj zero) (← word '}' zero) (← word ']' zero)) openAt)
          pure %[← iaddImm p 1, ← iaddImm tp 1, ← iaddImm d (-1)])
      pure %[p, tp, sp, d])
  storeI64 r.snd (← at_ TAPE_LEN)
  storeI64 r.thd (← at_ STR_LEN)
  pure r.head

/-- **The tape's checksum**: `h ← (h xor x) · 0x100000001B3` over every word
    of the canonical stream (a value's tag; then a string's length and each
    of its bytes, or a number's value word). -/
def sumBody : Prog V L Unit := do
  when .uge (← outLen) (← c 8) do
    let n ← load64 (← at_ TAPE_LEN)
    let tape ← at_ TAPE
    let strs ← at_ STRS
    let prime ← c 0x100000001B3
    let mix (h x : V .i64) : Prog V L (V .i64) := do imul (← bxor h x) prime
    let r ← wloop %[← c 0, ← c (0xCBF29CE484222325 - 2 ^ 64)]
      (fun cs => pure (contIfULt cs.head n, %[cs.snd], ()))
      (fun cs _ => do
        let w ← load64 (← iadd tape (← ishlImm cs.head 3))
        let tag ← ushrImm w 56
        let h ← mix cs.snd tag
        let isNum ← bor (← b2i (← icmp .eq tag (← ch 'l')))
          (← bor (← b2i (← icmp .eq tag (← ch 'u'))) (← b2i (← icmp .eq tag (← ch 'd'))))
        let isStr ← b2i (← icmp .eq tag (← ch '"'))
        let r ← ifte (jTys := [.i64, .i64]) .ne isNum (← c 0)
          (thn := do
            pure %[← iaddImm cs.head 2, ← mix h (← load64 (← iadd tape (← ishlImm (← iaddImm cs.head 1) 3)))])
          (els := ifte .ne isStr (← c 0)
            (thn := do
              let s ← iadd strs (← band w (← c (2 ^ 56 - 1)))
              let len ← uextend64 (← load32 s)
              let h1 ← mix h len
              let bs ← wloop %[← c 0, h1] (fun cs => pure (contIfULt cs.head len, %[cs.snd], ()))
                (fun cs _ => do
                  pure %[← iaddImm cs.head 1, ← mix cs.snd (← uload8_64 (← iadd (← iaddImm s 4) cs.head))])
              pure %[← iaddImm cs.head 1, bs.head])
            (els := do pure %[← iaddImm cs.head 1, h]))
        pure %[r.head, r.snd])
    storeI64 r.head (← outPtr)

end

def clifIR : Except String (List FuncData) :=
  Prog.program
    [Prog.entry "parse" (Prog.compileProgStatus 0 parseBody),
     Prog.entry "sum" (Prog.compileProg 1 sumBody)]

def artifacts (clif : List FuncData) : Array ArtifactEntry :=
  #[artifactEntry "json" { functions := clif, required_memory := MEM, initial_memory := tables }]

end Json

def Bench.Json.main (args : List String) : IO Unit := do
  let outDir ← requireOutputDir args
  let clif ← Prog.orDie Json.clifIR
  emitArtifacts outDir (Json.artifacts clif)

#eval ShipScan.check "Bench.Json" `Bench.Json.main
