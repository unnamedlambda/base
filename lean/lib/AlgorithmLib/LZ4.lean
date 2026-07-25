
namespace AlgorithmLib.LZ4

/-- One LZ4 sequence: a run of literal bytes followed by a back-reference match. -/
structure Seq where
  lits   : List UInt8
  offset : Nat
  mlen   : Nat
  deriving Repr, DecidableEq

/-- An LZ4 block: sequences, then trailing literals (the final literals-only token). -/
structure Block where
  seqs  : List Seq
  final : List UInt8
  deriving Repr, DecidableEq

/-- LZ4 well-formedness: the offset fits the 2-byte field and is nonzero;
    matches are at least minmatch (4). -/
def Seq.WF (s : Seq) : Prop := 1 ≤ s.offset ∧ s.offset ≤ 65535 ∧ 4 ≤ s.mlen

def Block.WF (b : Block) : Prop := ∀ s ∈ b.seqs, s.WF

-- ---------------------------------------------------------------------------
-- Decompressed meaning
-- ---------------------------------------------------------------------------

/-- Copy `len` bytes from `offset` back, one byte at a time — overlap-correct
    (an offset smaller than the length replicates, exactly as LZ4 requires). -/
def copyMatch (out : List UInt8) (offset : Nat) : Nat → List UInt8
  | 0 => out
  | n + 1 => copyMatch (out ++ [out.getD (out.length - offset) 0]) offset n

def expandSeq (acc : List UInt8) (s : Seq) : List UInt8 :=
  copyMatch (acc ++ s.lits) s.offset s.mlen

/-- The block's decompressed bytes. -/
def Block.expand (b : Block) : List UInt8 :=
  (b.seqs.foldl expandSeq []) ++ b.final

-- ---------------------------------------------------------------------------
-- Encoding (standard LZ4 block format)
-- ---------------------------------------------------------------------------

/-- LSIC extension bytes for the amount `n` beyond a saturated nibble:
    ⌊n/255⌋ bytes of 255, then `n % 255`. Always terminates with a byte < 255. -/
def ext (n : Nat) : List UInt8 :=
  List.replicate (n / 255) (255 : UInt8) ++ [UInt8.ofNat (n % 255)]

/-- Extension part of a length field (empty unless the nibble saturates at 15). -/
def encNib (n : Nat) : List UInt8 :=
  if n < 15 then [] else ext (n - 15)

def encodeSeq (s : Seq) : List UInt8 :=
  let ll := s.lits.length
  let ml := s.mlen - 4
  UInt8.ofNat (min ll 15 * 16 + min ml 15) ::
    (encNib ll ++ s.lits ++
     [UInt8.ofNat (s.offset % 256), UInt8.ofNat (s.offset / 256)] ++
     encNib ml)

def encodeFinal (ls : List UInt8) : List UInt8 :=
  UInt8.ofNat (min ls.length 15 * 16) :: (encNib ls.length ++ ls)

def Block.encode (b : Block) : List UInt8 :=
  (b.seqs.flatMap encodeSeq) ++ encodeFinal b.final
end AlgorithmLib.LZ4
