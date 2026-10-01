module
public import AlgorithmLib.Gen
meta import AlgorithmLib.Gen
public import AlgorithmLib.Host.Term
meta import AlgorithmLib.Host.Term
public import Tokenizer.Common
meta import Tokenizer.Common
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
@[expose] public section

/-!
  # The pre-tokenizer, as a table a scan can read

  BPE does not merge across a chunk boundary, and where those boundaries fall
  is decided by a regex the checkpoint ships. Skipping it is not a rounding
  error: measured against the reference tokenizer, whole-text BPE agrees on
  three of six ordinary strings for both Qwen2 and gpt-oss, and it diverges on
  exactly what the pattern exists to handle — code indentation, runs of digits,
  repeated whitespace.

  CLIF has no regex and should not grow one. What these patterns actually
  inhabit is much smaller: an ordered alternation, each alternative a sequence
  of items, each item a character class with a repeat count, plus a fixed set
  of contraction literals and one lookahead. That is a table, `tools/pretok.py`
  writes it, and this is the machine that reads it.

  ## No backtracking

  A real engine backtracks in three places here, and each is rewritten as a
  forward pass rather than implemented:

  * `\s+(?!\S)` means "the whitespace run less its last character", or all of
    it at end of input — `RUNBUT`.
  * `\s*[\r\n]+` means "the run truncated at its last newline" — `RUNTO`.
  * `[\p{Lu}…]*[\p{Ll}…]+` has overlapping classes, so the star can swallow a
    character the plus needs. The longest match is `max over p of (p + the
    B-run at p)`, which one backward pass computes for every `p` at once —
    `STARPLUS`.

  The one genuine retry left is the leading `[^\r\n\p{L}\p{N}]?`, which
  overlaps the letter class that follows it through `\p{M}`. It is tried
  greedily and then once with the optional forced empty, in that order, because
  a regex returns the first success in backtracking order and not the longest.

  ## What is not here

  Normalisation. gpt-oss declares `normalizer: null` and needs none; Qwen2
  declares NFC, and text holding decomposed characters will therefore tokenize
  differently through this path than through the reference. That is on the
  ledger rather than fixed, and it is why `tools/tokbin.py` prints a note when
  it converts a checkpoint that asks for one.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.Prog

namespace PretokCommon

/-- Op codes, as `tools/pretok.py` writes them. -/
def OP_CLASS : Int := 0
def OP_CONTR : Int := 1
def OP_RUNBUT : Int := 2
def OP_CHAR : Int := 3
def OP_RUNTO : Int := 4
def OP_STARPLUS : Int := 5

/-- Bytes per record, all fixed so that an index is one multiply. -/
def ITEM_BYTES : Nat := 24
def ALT_BYTES : Nat := 8
def CONTR_MAX : Nat := 8
def CONTR_BYTES : Nat := 4 + 4 * CONTR_MAX

/-- **Where the pre-tokenizer keeps its working arrays.**

    Offsets into the host program's own memory, like `TokenizerCommon.TokMem`
    and for the same reason. `cpBuf` and `cpByte` are parallel: one code point
    and one byte offset per decoded character, so a chunk found in code-point
    space can be handed to BPE as a byte range. -/
structure PretokMem where
  /-- u32 per character: the code points of the input text. -/
  cpBuf : Nat
  /-- u32 per character: where each one started, as a byte offset. -/
  cpByte : Nat
  /-- i64: how many characters `cpBuf` holds. -/
  cpCount : Nat
  /-- u32 per token: where whole-text tokenisation accumulates its output. -/
  outTok : Nat
  /-- i64: how many tokens `outTok` holds. -/
  outCount : Nat

def load64At (base : V .i64) (off : Nat) : Prog V L (V .i64) :=
  load64 =<< iaddImm base off

/-- The membership byte of one code point. -/
def classOf (tabPtr cp : V .i64) : Prog V L (V .i64) := do
  uload8_64 (← iadd tabPtr cp)

/-- The code point at index `k`. -/
def cpAt (cpsPtr k : V .i64) : Prog V L (V .i64) := do
  uload32_64 (← iadd cpsPtr (← ishlImm k 2))

/-- **UTF-8 in, code points out.**

    Both arrays are filled in one pass, and a byte that cannot begin a sequence
    is taken as itself. That last part is deliberate: the tokenizer's job is
    not to validate its input, and a byte-level BPE has a token for every byte,
    so refusing malformed text would lose information the model can represent.

    A trailing truncated sequence is likewise consumed as its bytes rather than
    held back, so this always makes progress and always terminates. -/
def utf8DecodeM (c : TokenizerCommon.TokMem) (p : PretokMem) : Prog V L Unit := do
  let ptr ← basePtr
  let textBase ← iaddImm ptr c.textIn
  let cpsPtr ← iaddImm ptr p.cpBuf
  let bytePtr ← iaddImm ptr p.cpByte
  let n ← load64At ptr c.textLen
  let zero64 ← iconst64 0
  let ex ← wloop %[zero64, zero64]                     -- byte index, char index
    (head := fun s => return (contIf .ult (s.head) n, %[s.snd], ()))
    (body := fun s _ => do
      let i := s.head
      let k := s.snd
      let b0 ← uload8_64 (← iadd textBase i)
      -- how many bytes this leader claims, and what it contributes
      let c0 ← iconst64 0xC0
      let c1 ← iconst64 0xE0
      let c2 ← iconst64 0xF0
      let c3 ← iconst64 0xF8
      let rem ← isub n i
      let two64 ← iconst64 2
      let three64 ← iconst64 3
      let four64 ← iconst64 4
      -- `len` is 1 unless the leader says otherwise *and* the bytes are there
      let lenA ← ifte .uge b0 c2
        (do let ok ← ifte .uge rem four64 (pure %[four64]) (pure %[← iconst64 1])
            pure %[ok.head])
        (do let l3 ← ifte .uge b0 c1
              (do let ok ← ifte .uge rem three64 (pure %[three64]) (pure %[← iconst64 1])
                  pure %[ok.head])
              (do let l2 ← ifte .uge b0 c0
                    (do let ok ← ifte .uge rem two64 (pure %[two64]) (pure %[← iconst64 1])
                        pure %[ok.head])
                    (pure %[← iconst64 1])
                  let _ ← iconst64 1
                  pure %[l2.head])
            let _ ← iconst64 1
            pure %[l3.head])
      let _ ← iconst64 1
      let len := lenA.head
      -- a four-byte leader above the range is still four bytes; guard it
      let tooBig ← ifte .uge b0 c3 (pure %[← iconst64 1]) (pure %[zero64])
      let len2L ← ifte .ne (tooBig.head) zero64 (pure %[← iconst64 1]) (pure %[len])
      let len2 := len2L.head
      -- the payload: leader bits, then six per continuation byte
      let cpL ← ifte .eq len2 (← iconst64 1)
        (pure %[b0])
        (do let maskA ← ifte .eq len2 two64
              (pure %[← iconst64 0x1F])
              (do let m ← ifte .eq len2 three64
                    (pure %[← iconst64 0x0F]) (pure %[← iconst64 0x07])
                  let _ ← iconst64 0x07
                  pure %[m.head])
            let _ ← iconst64 0x07
            let acc0 ← band b0 maskA.head
            let one64 ← iconst64 1
            let accE ← wloop %[one64, acc0]
              (head := fun s2 => return (contIf .ult (s2.head) len2, %[s2.snd], ()))
              (body := fun s2 _ => do
                let j := s2.head
                let acc := s2.snd
                let bj ← uload8_64 (← iadd textBase (← iadd i j))
                let lo ← band bj (← iconst64 0x3F)
                let sh ← ishlImm acc 6
                return %[← iaddImm j 1, ← bor sh lo])
            pure %[accE.head])
      let cp := cpL.head
      storeI32 (← ireduce32 cp) (← iadd cpsPtr (← ishlImm k 2))
      storeI32 (← ireduce32 i) (← iadd bytePtr (← ishlImm k 2))
      return %[← iadd i len2, ← iaddImm k 1])
  -- one past the end, so a chunk's byte range is always cpByte[j] .. cpByte[j+n]
  let kEnd := ex.head
  storeI32 (← ireduce32 n) (← iadd bytePtr (← ishlImm kEnd 2))
  storeI64 kEnd (← absAddr ptr p.cpCount)

/-- ASCII lowering, which is all the contraction table needs. -/
def lowerAscii (cp : V .i64) : Prog V L (V .i64) := do
  let bigA ← iconst64 65
  let bigZ ← iconst64 90
  let r ← ifte .ugt cp bigZ (pure %[cp])
    (do let s ← ifte .ult cp bigA (pure %[cp]) (pure %[← iaddImm cp 32])
        pure %[s.head])
  return r.head

/-- **The contraction suffix.**

    Returns its length, or -1 when one was required and none matched. Compared
    case-insensitively because the pattern is, and the literals are ASCII. -/
def contrM (cpsPtr nCp contrPtr nContr pos required : V .i64) : Prog V L (V .i64) := do
  let zero64 ← iconst64 0
  let negOne ← iconst64 (-1)
  let stride ← iconst64 CONTR_BYTES
  let ex ← wloopL %[zero64, negOne]
    (head := fun _ s => return (contIf .ult (s.head) nContr, %[s.snd], ()))
    (body := fun lbl s _ => do
      let ci := s.head
      let rec0 ← iadd contrPtr (← imul ci stride)
      let clen ← uload32_64 rec0
      let next ← iaddImm ci 1
      -- does it fit in what is left?
      let room ← iadd pos clen
      when .ugt room nCp (continueWith lbl %[next, s.snd])
      let one64 ← iconst64 1
      let mE ← wloopL %[zero64, one64]
        (head := fun _ t => return (contIf .ult (t.head) clen, %[t.snd], ()))
        (body := fun lbl t _ => do
          let j := t.head
          let want ← uload32_64 (← iadd rec0 (← iaddImm (← ishlImm j 2) 4))
          let got ← lowerAscii (← cpAt cpsPtr (← iadd pos j))
          when .ne got want (brk lbl %[zero64])
          return %[← iaddImm j 1, t.snd])
      let matched := mE.head
      when .ne matched zero64 (brk lbl %[clen])
      return %[next, s.snd])
  let found := ex.head
  -- no match: length zero when the suffix was optional, failure when not
  let r ← ifte .sge found zero64 (pure %[found])
    (do let s ← ifte .ne required zero64 (pure %[negOne]) (pure %[zero64])
        pure %[s.head])
  return r.head

/-- A greedy run of a class, capped at `hi`, failing below `lo`.

    `extra` is one code point the class accepts in addition to whatever `mask`
    says, or zero for none — o200k's `[\r\n/]*` is a class plus one literal. -/
def runM (cpsPtr nCp tabPtr pos mask neg lo hi extra : V .i64) : Prog V L (V .i64) := do
  let zero64 ← iconst64 0
  let ex ← wloop1L zero64
    (head := fun _ nn => return (contIf .ult nn hi, %[nn], ()))
    (body := fun lbl nn _ => do
      let idx ← iadd pos nn
      when .uge idx nCp (brk lbl %[nn])
      let cp ← cpAt cpsPtr idx
      let b ← classOf tabPtr cp
      let hasAll ← band b mask
      let hasNone ← band b neg
      -- `(b & mask) == mask && (b & neg) == 0`, or the one literal
      let okL ← ifte .eq hasAll mask
        (do let s ← ifte .eq hasNone zero64 (pure %[← iconst64 1]) (pure %[zero64])
            pure %[s.head])
        (pure %[zero64])
      let ok0 := okL.head
      let okE ← ifte .ne ok0 zero64 (pure %[ok0])
        (do let s ← ifte .ne extra zero64
              (do let t ← ifte .eq cp extra (pure %[← iconst64 1]) (pure %[zero64])
                  pure %[t.head])
              (pure %[zero64])
            pure %[s.head])
      when .eq (okE.head) zero64 (brk lbl %[nn])
      return %[← iaddImm nn 1])
  let n := ex.head
  let r ← ifte .ult n lo (pure %[← iconst64 (-1)]) (pure %[n])
  return r.head


/-- `\s+(?!\S)`: the whitespace run, less its last character unless the input
    ends there. Written forward, which is the whole point — the lookahead is
    what a regex backtracks for. -/
def runButM (cpsPtr nCp tabPtr pos mask : V .i64) : Prog V L (V .i64) := do
  let zero64 ← iconst64 0
  let ex ← wloop1L zero64
    (head := fun _ nn => return (contIf .ult zero64 (← iconst64 1), %[nn], ()))
    (body := fun lbl nn _ => do
      let idx ← iadd pos nn
      when .uge idx nCp (brk lbl %[nn])
      let b ← classOf tabPtr (← cpAt cpsPtr idx)
      when .eq (← band b mask) zero64 (brk lbl %[nn])
      return %[← iaddImm nn 1])
  let n := ex.head
  -- one back, unless the run reached the end of the input
  let endAt ← iadd pos n
  let nL ← ifte .ult endAt nCp (pure %[← iaddImm n (-1)]) (pure %[n])
  let n2 := nL.head
  let r ← ifte .ult n2 (← iconst64 1) (pure %[← iconst64 (-1)]) (pure %[n2])
  return r.head

/-- `\s*[\r\n]+`: the run truncated at its last newline, and a failure when it
    holds none. -/
def runToM (cpsPtr nCp tabPtr pos mask neg : V .i64) : Prog V L (V .i64) := do
  let zero64 ← iconst64 0
  let ex ← wloopL %[zero64, zero64]                        -- n, last
    (head := fun _ s => return (contIf .ult zero64 (← iconst64 1),
                              %[s.head, s.snd], ()))
    (body := fun lbl s _ => do
      let nn := s.head
      let last := s.snd
      let idx ← iadd pos nn
      when .uge idx nCp (brk lbl %[nn, last])
      let b ← classOf tabPtr (← cpAt cpsPtr idx)
      when .eq (← band b mask) zero64 (brk lbl %[nn, last])
      let n1 ← iaddImm nn 1
      let isNl ← band b neg
      let l2 ← ifte .ne isNl zero64 (pure %[n1]) (pure %[last])
      return %[n1, l2.head])
  let last := ex.snd
  let r ← ifte .ult last (← iconst64 1) (pure %[← iconst64 (-1)]) (pure %[last])
  return r.head

/-- One specific code point, `lo` to `hi` times. -/
def charRunM (cpsPtr nCp pos want lo hi : V .i64) : Prog V L (V .i64) := do
  let zero64 ← iconst64 0
  let ex ← wloop1L zero64
    (head := fun _ nn => return (contIf .ult nn hi, %[nn], ()))
    (body := fun lbl nn _ => do
      let idx ← iadd pos nn
      when .uge idx nCp (brk lbl %[nn])
      when .ne (← cpAt cpsPtr idx) want (brk lbl %[nn])
      return %[← iaddImm nn 1])
  let n := ex.head
  let r ← ifte .ult n lo (pure %[← iconst64 (-1)]) (pure %[n])
  return r.head

/-- **`A* B+` where `A` and `B` overlap.**

    The star can swallow a character the plus then needs, and a real engine
    hands it back. The longest match is `max over q in the A-run of (q + the
    B-run starting at q)`, and one backward sweep computes every B-run at once:
    walking right to left, the run at `q` is one more than the run at `q + 1`
    when `q` is in `B`, and zero when it is not.

    Returns the matched length, or -1 when no `B` run exists at all. -/
def starPlusM (cpsPtr nCp tabPtr pos maskA maskB : V .i64) : Prog V L (V .i64) := do
  let zero64 ← iconst64 0
  let one64 ← iconst64 1
  let negOne ← iconst64 (-1)
  -- the A-run
  let aE ← wloop1L zero64
    (head := fun _ nn => return (contIf .ult zero64 one64, %[nn], ()))
    (body := fun lbl nn _ => do
      let idx ← iadd pos nn
      when .uge idx nCp (brk lbl %[nn])
      let b ← classOf tabPtr (← cpAt cpsPtr idx)
      when .eq (← band b maskA) zero64 (brk lbl %[nn])
      return %[← iaddImm nn 1])
  let a := aE.head
  -- the B-run that starts where the A-run stopped
  let tail ← iadd pos a
  let rE ← wloop1L zero64
    (head := fun _ nn => return (contIf .ult zero64 one64, %[nn], ()))
    (body := fun lbl nn _ => do
      let idx ← iadd tail nn
      when .uge idx nCp (brk lbl %[nn])
      let b ← classOf tabPtr (← cpAt cpsPtr idx)
      when .eq (← band b maskB) zero64 (brk lbl %[nn])
      return %[← iaddImm nn 1])
  let run0 := rE.head
  let bestL ← ifte .ugt run0 zero64 (pure %[← iadd tail run0]) (pure %[negOne])
  -- right to left across the A-run, carrying the B-run length at each step
  let sw ← wloopL %[zero64, run0, bestL.head]      -- t, run, best
    (head := fun _ s => return (contIf .ult (s.head) a, %[s.snd, s.thd], ()))
    (body := fun lbl s _ => do
      let t := s.head
      let run := s.snd
      let best := s.thd
      let q ← isub (← iaddImm tail (-1)) t
      let b ← classOf tabPtr (← cpAt cpsPtr q)
      let inB ← band b maskB
      let rL ← ifte .ne inB zero64 (pure %[← iaddImm run 1]) (pure %[zero64])
      let run2 := rL.head
      let cand ← iadd q run2
      let bL ← ifte .ugt run2 zero64
        (do let s2 ← ifte .sgt cand best (pure %[cand]) (pure %[best])
            pure %[s2.head])
        (pure %[best])
      return %[← iaddImm t 1, run2, bL.head])
  let best := sw.snd
  let r ← ifte .slt best zero64 (pure %[negOne]) (pure %[← isub best pos])
  return r.head

/-- **One alternative against the text at `i`**, as a matched length or -1.

    `skipLead` forces the leading optional item empty, which is the one retry
    these patterns need. -/
def scanM (cpsPtr nCp tabPtr itemsPtr contrPtr nContr
                   altStart altCount skipLead i : V .i64) : Prog V L (V .i64) := do
  let zero64 ← iconst64 0
  let one64 ← iconst64 1
  let negOne ← iconst64 (-1)
  let stride ← iconst64 ITEM_BYTES
  let ex ← wloopL %[zero64, i, zero64]                     -- k, p, failed
    (head := fun _ s => return (contIf .ult (s.head) altCount,
                              %[s.snd, s.thd], ()))
    (body := fun lbl s _ => do
      let k := s.head
      let p := s.snd
      let k1 ← iaddImm k 1
      -- the forced-empty retry skips item zero outright
      let skipL ← ifte .eq k zero64
        (do let t ← ifte .ne skipLead zero64 (pure %[one64]) (pure %[zero64])
            pure %[t.head])
        (pure %[zero64])
      when .ne (skipL.head) zero64 (continueWith lbl %[k1, p, zero64])
      let rec0 ← iadd itemsPtr (← imul (← iadd altStart k) stride)
      let op ← uload32_64 rec0
      let mask ← uload32_64 (← iaddImm rec0 4)
      let neg ← uload32_64 (← iaddImm rec0 8)
      let lo ← uload32_64 (← iaddImm rec0 12)
      let hi ← uload32_64 (← iaddImm rec0 16)
      let extra ← uload32_64 (← iaddImm rec0 20)
      -- each arm yields a length, or -1
      let nL ← ifte .eq op (← iconst64 OP_CONTR)
        (do let req ← ifte .eq lo one64 (pure %[one64]) (pure %[zero64])
            pure %[← contrM cpsPtr nCp contrPtr nContr p (req.head)])
        (do let a2 ← ifte .eq op (← iconst64 OP_RUNBUT)
              (pure %[← runButM cpsPtr nCp tabPtr p mask])
              (do let a3 ← ifte .eq op (← iconst64 OP_RUNTO)
                    (pure %[← runToM cpsPtr nCp tabPtr p mask neg])
                    (do let a4 ← ifte .eq op (← iconst64 OP_CHAR)
                          (pure %[← charRunM cpsPtr nCp p mask lo hi])
                          (do let a5 ← ifte .eq op (← iconst64 OP_STARPLUS)
                                (pure %[← starPlusM cpsPtr nCp tabPtr p mask neg])
                                (pure %[← runM cpsPtr nCp tabPtr p mask neg lo hi extra])
                              pure %[a5.head])
                        pure %[a4.head])
                  pure %[a3.head])
            pure %[a2.head])
      let n := nL.head
      when .slt n zero64 (brk lbl %[p, one64])
      return %[k1, ← iadd p n, zero64])
  let pEnd := ex.head
  let failed := ex.snd
  -- an alternative that matched nothing has not matched
  let r ← ifte .ne failed zero64 (pure %[negOne])
    (do let s ← ifte .ugt pEnd i (pure %[← isub pEnd i]) (pure %[negOne])
        pure %[s.head])
  return r.head

/-- Greedy, then once with the leading optional forced empty — in that order,
    because a regex returns the first success in backtracking order and not the
    longest match. -/
def matchAltM (cpsPtr nCp tabPtr itemsPtr contrPtr nContr
                       altStart altCount i : V .i64) : Prog V L (V .i64) := do
  let zero64 ← iconst64 0
  let one64 ← iconst64 1
  let n0 ← scanM cpsPtr nCp tabPtr itemsPtr contrPtr nContr altStart altCount zero64 i
  let r ← ifte .sge n0 zero64 (pure %[n0])
    (do let rec0 ← iadd itemsPtr (← imul altStart (← iconst64 ITEM_BYTES))
        let op ← uload32_64 rec0
        let lo ← uload32_64 (← iaddImm rec0 12)
        -- only an optional leading class or literal can be forced empty
        let elig ← ifte .eq lo zero64
          (do let a ← ifte .eq op (← iconst64 OP_CLASS) (pure %[one64])
                (do let b ← ifte .eq op (← iconst64 OP_CHAR) (pure %[one64]) (pure %[zero64])
                    pure %[b.head])
              pure %[a.head])
          (pure %[zero64])
        let s ← ifte .ne (elig.head) zero64
          (pure %[← scanM cpsPtr nCp tabPtr itemsPtr contrPtr nContr
                    altStart altCount one64 i])
          (pure %[n0])
        pure %[s.head])
  return r.head


/-- **Byte range to initial tokens.**

    `TokenizerCommon.tokenizeInitM` does this for the whole of `textIn`; a
    chunk needs it for a slice, and copying the slice to the front of the
    buffer instead would destroy the text the later chunks still need. -/
def initRangeM (c : TokenizerCommon.TokMem) (b0 b1 : V .i64) : Prog V L Unit := do
  let ptr ← basePtr
  let bufP ← load64At ptr c.bufPtr
  let byteInit ← iaddImm bufP TokenizerCommon.BYTE_INIT_OFF
  let textBase ← iaddImm ptr c.textIn
  let tokBuf ← iaddImm ptr c.tokenBuf
  let len ← isub b1 b0
  forLoop len fun i => do
    let byt ← uload8_64 (← iadd textBase (← iadd b0 i))
    let initTok ← load32 (← iadd byteInit (← ishlImm byt 2))
    storeI32 initTok (← iadd tokBuf (← ishlImm i 2))
  storeI64 len (← absAddr ptr c.tokenCount)

/-- **The whole of tokenisation: split, then merge inside each chunk.**

    Reads `textIn`/`textLen`, writes `outTok`/`outCount`. The alternation is
    ordered, so at each position the first alternative that matches a non-empty
    prefix wins; a position no alternative matches advances by one character,
    which is what keeps this total.

    Every table it needs is found by walking forward from the tokenizer file's
    header, so nothing here knows which checkpoint wrote the file. -/
def tokenizeTextM (c : TokenizerCommon.TokMem) (p : PretokMem) : Prog V L Unit := do
  let ptr ← basePtr
  utf8DecodeM c p
  let bufP ← load64At ptr c.bufPtr
  let pretok ← uload32_64 (← iaddImm bufP TokenizerCommon.HDR_PRETOK)
  let base ← iadd bufP pretok
  let nAlts ← uload32_64 base
  let nItems ← uload32_64 (← iaddImm base 4)
  let nContr ← uload32_64 (← iaddImm base 8)
  let altsPtr ← iaddImm base 16
  let itemsPtr ← iadd altsPtr (← imul nAlts (← iconst64 ALT_BYTES))
  let contrPtr ← iadd itemsPtr (← imul nItems (← iconst64 ITEM_BYTES))
  let tabPtr ← iadd contrPtr (← imul nContr (← iconst64 CONTR_BYTES))
  let cpsPtr ← iaddImm ptr p.cpBuf
  let bytePtr ← iaddImm ptr p.cpByte
  let tokBuf ← iaddImm ptr c.tokenBuf
  let outPtr ← iaddImm ptr p.outTok
  let nCp ← load64At ptr p.cpCount
  let zero64 ← iconst64 0
  let one64 ← iconst64 1
  let negOne ← iconst64 (-1)
  let stride8 ← iconst64 ALT_BYTES
  let ex ← wloopL %[zero64, zero64]                       -- character index, token count
    (head := fun _ s => return (contIf .ult (s.head) nCp, %[s.snd], ()))
    (body := fun lbl s _ => do
      let i := s.head
      let nOut := s.snd
      -- the first alternative that matches something
      let pick ← wloopL %[zero64, negOne]
        (head := fun _ t => return (contIf .ult (t.head) nAlts, %[t.snd], ()))
        (body := fun lbl t _ => do
          let ai := t.head
          let arec ← iadd altsPtr (← imul ai stride8)
          let aStart ← uload32_64 arec
          let aCount ← uload32_64 (← iaddImm arec 4)
          let n ← matchAltM cpsPtr nCp tabPtr itemsPtr contrPtr nContr aStart aCount i
          when .sgt n zero64 (brk lbl %[n])
          return %[← iaddImm ai 1, t.snd])
      let matched := pick.head
      -- no alternative matched: one character, so the scan always advances
      let nL ← ifte .sgt matched zero64 (pure %[matched]) (pure %[one64])
      let n := nL.head
      -- the chunk in bytes, and its own BPE
      let b0 ← uload32_64 (← iadd bytePtr (← ishlImm i 2))
      let b1 ← uload32_64 (← iadd bytePtr (← ishlImm (← iadd i n) 2))
      initRangeM c b0 b1
      TokenizerCommon.tokenizeBpeM c
      let got ← load64At ptr c.tokenCount
      forLoop got fun j => do
        let tk ← load32 (← iadd tokBuf (← ishlImm j 2))
        storeI32 tk (← iadd outPtr (← ishlImm (← iadd nOut j) 2))
      return %[← iadd i n, ← iadd nOut got])
  storeI64 (ex.head) (← absAddr ptr p.outCount)

end PretokCommon
