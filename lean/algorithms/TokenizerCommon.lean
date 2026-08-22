import AlgorithmLib.Gen
import AlgorithmLib.HProg

/-!
  # The tokenizer, once, for every model that ships one

  A BPE tokenizer is a hash table of merges, a byte table, and a decode pool,
  and none of that depends on the model. What differs between checkpoints is
  the *contents* of the file — how many merges, how large the vocabulary, which
  bytes map to which initial token — and every one of those is a number in the
  file's own header. So one program serves both, and adding a third checkpoint
  is a converter run rather than a code change.

  What is not shared is where in a program's memory the working buffers live.
  A model with a 4169-buffer bind table and a model with an unrolled decode
  stack do not agree about that and should not have to, so the offsets arrive
  in `TokMem` and everything else is common.

  ## The file

  Written by `tools/tokbin.py`, and self-describing:

      [0]  n_merges : u32
      [4]  vocab_size : u32
      [8]  byte_pool_size : u32
      [12] pretok_off : u32     -- 0 in files written before the pre-tokenizer
      [16] byte_init[256] : u32           byte value to its single-byte token
      [1040] merges[n_merges] : (a, b, result) x u32, in rank order
      [1040 + 12n] dec_off[vocab] : u32   into the byte pool
      [.. + 4v]    dec_len[vocab] : u32
      [.. + 4v]    byte_pool

  Every offset after the header is computed forward from it, which is what
  lets the pre-tokenizer tables be appended without disturbing a reader that
  predates them.
-/

open AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sur

namespace TokenizerCommon

/-- **Where a host program keeps the tokenizer's working state.**

    Byte offsets into the program's own memory. `htCtx` and `cudaCtx` are the
    two slots the surrounding program already owns; the rest are this
    tokenizer's, and a program that wants one has to find room for them. -/
structure TokMem where
  /-- i64: the hash-table context, written by `htInit`. -/
  htCtx : Nat
  /-- i64: the CUDA context, needed only for the pinned allocation. -/
  cudaCtx : Nat
  /-- i64: pointer to the null-terminated path of the tokenizer file. -/
  pathPtr : Nat
  /-- i64: pointer to the file's contents once they are in host memory. -/
  bufPtr : Nat
  /-- u32 per token: the working array BPE merges in place. -/
  tokenBuf : Nat
  /-- i64: how many tokens `tokenBuf` currently holds. -/
  tokenCount : Nat
  /-- Input text, as bytes. -/
  textIn : Nat
  /-- Output text, as bytes. -/
  textOut : Nat
  /-- i64: the byte length of whichever of the two is in play. -/
  textLen : Nat
  /-- 8 bytes of scratch: the pair being looked up. -/
  htKey : Nat
  /-- 8 bytes of scratch: the (rank, result) that came back. -/
  htVal : Nat
  /-- How much to reserve for the file. It is read whole. -/
  fileMaxBytes : Nat
  /-- Bytes `textOut` can hold.

      Not decoration: the detokenizer's output is as long as the tokens make
      it, and a caller that asks for more tokens than the buffer can spell runs
      off the end of it into whatever is next. Truncating is a wrong answer;
      not truncating is a corrupted heap. -/
  textMaxBytes : Nat

/-- The two address forms are not interchangeable: `iaddImm` folds the offset
    into one instruction and `absAddr` materialises it as a constant first, so
    they emit different CLIF. Each site below uses the one it has always used,
    which is what lets a program that already ships be rewired onto this
    module without its artifact changing by a byte. -/
private def load64At (base : R) (off : Nat) : M R :=
  load64 =<< iaddImm base off

/-- Header fields, by name rather than by a number at each use. -/
def HDR_MERGES : Nat := 0
def HDR_VOCAB : Nat := 4
def HDR_POOL : Nat := 8
def HDR_PRETOK : Nat := 12
def BYTE_INIT_OFF : Nat := 16
def MERGE_OFF : Nat := 1040

/-- **Read the file and build the merge table.**

    The file goes into pinned host memory because the caller may also want to
    hand parts of it to the device, and a pinned buffer costs nothing extra
    here — it is read once at start-up.

    The merges arrive in rank order, so the loop index *is* the rank, and both
    it and the resulting token go into the value: a merge lookup then answers
    "should this pair merge, and how strongly" in one probe. -/
def loadTokenizerM (c : TokMem) : M Unit := do
  let ptr := basePtr
  let ctxPtr ← load64 (← absAddr ptr c.cudaCtx)
  let pathP ← load64 (← absAddr ptr c.pathPtr)
  let bytes64 ← iconst64 c.fileMaxBytes
  let pinId ← call IR.Ffi.cudaPinnedAlloc.id [ctxPtr, bytes64]
  let bufP ← call IR.Ffi.cudaPinnedPtr.id [ctxPtr, pinId]
  let zero64 ← iconst64 0
  let _ ← call IR.Ffi.fileReadToPtr.id [pathP, bufP, zero64, bytes64]
  storeI64 bufP (← absAddr ptr c.bufPtr)
  callVoid IR.Ffi.htInit.id [ptr]
  let htCtx ← load64At ptr c.htCtx
  let _ ← call IR.Ffi.htCreate.id [htCtx]
  let nMerges ← uload32_64 (← iaddImm bufP HDR_MERGES)
  let mergeBase ← iaddImm bufP MERGE_OFF
  let keyAddr ← iaddImm ptr c.htKey
  let valAddr ← iaddImm ptr c.htVal
  let keyLen8 ← iconst32 8
  let valLen8 ← iconst32 8
  let twelve64 ← iconst64 12
  forLoop nMerges fun i => do
    let mergePtr ← iadd mergeBase (← imul i twelve64)
    let tokA ← load32 (← iaddImm mergePtr 0)
    let tokB ← load32 (← iaddImm mergePtr 4)
    let result ← load32 (← iaddImm mergePtr 8)
    let rank32 ← ireduce32 i
    storeI32 tokA keyAddr
    storeI32 tokB (← iaddImm keyAddr 4)
    storeI32 rank32 valAddr
    storeI32 result (← iaddImm valAddr 4)
    callVoid IR.Ffi.htInsert.id [htCtx, keyAddr, keyLen8, valAddr, valLen8]

/-- **Every byte its own token, before any merging.**

    One `u32` per input byte, which is why the token buffer has to be four
    times the text buffer and not merely as large. -/
def tokenizeInitM (c : TokMem) : M Unit := do
  let ptr := basePtr
  let bufP ← load64At ptr c.bufPtr
  let textLen ← load64At ptr c.textLen
  let byteInit ← iaddImm bufP BYTE_INIT_OFF
  let textBase ← iaddImm ptr c.textIn
  let tokBuf ← iaddImm ptr c.tokenBuf
  forLoop textLen fun i => do
    let byt ← uload8_64 (← iadd textBase i)
    let initTok ← load32 (← iadd byteInit (← ishlImm byt 2))
    storeI32 initTok (← iadd tokBuf (← ishlImm i 2))
  storeI64 textLen (← absAddr ptr c.tokenCount)

/-- **Merge until nothing merges.**

    Each pass finds the lowest-ranked adjacent pair anywhere in the buffer,
    merges that one, and closes the gap. A pass that finds none is the last.

    That is the definition rather than the fast algorithm — BPE is usually
    written with a priority queue — but a chunk is a handful of tokens once the
    pre-tokenizer has split the text, and a queue in CLIF would be more
    machinery than the thing it accelerates. -/
def tokenizeBpeM (c : TokMem) : M Unit := do
  let ptr := basePtr
  let htCtx ← load64At ptr c.htCtx
  let tokBuf ← iaddImm ptr c.tokenBuf
  let keyAddr ← iaddImm ptr c.htKey
  let valAddr ← iaddImm ptr c.htVal
  let keyLen8 ← iconst32 8
  let tokCount ← load64At ptr c.tokenCount
  let zero64 ← iconst64 0
  let one64 ← iconst64 1
  let maxRank ← iconst32 (-1)
  let negOne64 ← iconst64 (-1)
  let zero32 ← iconst32 0
  let e ← wloop1 tokCount
    (head := fun n => return (contIf .ugt n one64, [n], ()))
    (body := fun n _ => do
      let n1 ← iaddImm n (-1)
      let sc ← wloop [zero64, maxRank, negOne64]
        (head := fun cc =>
          return (contIf .ult (cc.headD 0) n1, [cc.getD 1 0, cc.getD 2 0], ()))
        (body := fun cc _ => do
          let i := cc.headD 0
          let r := cc.getD 1 0
          let p := cc.getD 2 0
          let iOff ← ishlImm i 2
          let tokA ← load32 (← iadd tokBuf iOff)
          let tokB ← load32 (← iadd tokBuf (← iaddImm iOff 4))
          storeI32 tokA keyAddr
          storeI32 tokB (← iaddImm keyAddr 4)
          let found ← call IR.Ffi.htLookup.id [htCtx, keyAddr, keyLen8, valAddr]
          let nextI ← iaddImm i 1
          when .slt found zero32 (continueWith [nextI, r, p])
          let rank ← load32 valAddr
          let rp ← ifte .ult rank r (pure [rank, i]) (pure [r, p])
          return [nextI, rp.headD 0, rp.getD 1 0])
      let bestPos := sc.getD 1 0
      when .eq bestPos negOne64 (brk [n])
      let dOff ← ishlImm bestPos 2
      let dA ← load32 (← iadd tokBuf dOff)
      let dB ← load32 (← iadd tokBuf (← iaddImm dOff 4))
      storeI32 dA keyAddr
      storeI32 dB (← iaddImm keyAddr 4)
      let _ ← call IR.Ffi.htLookup.id [htCtx, keyAddr, keyLen8, valAddr]
      let resT ← load32 (← iaddImm valAddr 4)
      storeI32 resT (← iadd tokBuf dOff)
      let _ ← wloop1 (← iaddImm bestPos 1)
        (head := fun j => return (contIf .ult j n1, ([] : List R), ()))
        (body := fun j _ => do
          let sbOff ← ishlImm j 2
          let nextT ← load32 (← iadd tokBuf (← iaddImm sbOff 4))
          storeI32 nextT (← iadd tokBuf sbOff)
          return [← iaddImm j 1])
      return [n1])
  storeI64 (e.headD 0) (← absAddr ptr c.tokenCount)

/-- **Token ids back to bytes**, by concatenating what each one stands for.

    The three tables the pool needs are all found by walking forward from the
    header, so nothing here knows a vocabulary size. -/
def detokenizeM (c : TokMem) : M Unit := do
  let ptr := basePtr
  let bufP ← load64At ptr c.bufPtr
  let nMerges ← uload32_64 (← iaddImm bufP HDR_MERGES)
  let vocabSize ← uload32_64 (← iaddImm bufP HDR_VOCAB)
  let twelve64 ← iconst64 12
  let four64 ← iconst64 4
  let mergeBytes ← imul nMerges twelve64
  let decOffPtr ← iadd (← iaddImm bufP MERGE_OFF) mergeBytes
  let vocBytes ← imul vocabSize four64
  let decLenPtr ← iadd decOffPtr vocBytes
  let bytePool ← iadd decLenPtr vocBytes
  let tokBuf ← iaddImm ptr c.tokenBuf
  let nToks ← load64At ptr c.tokenCount
  let textOut ← iaddImm ptr c.textOut
  let zero64 ← iconst64 0
  let cap ← iconst64 c.textMaxBytes
  let finalTp ← forLoopAcc nToks zero64 fun ti tp => do
    let tokId ← uload32_64 (← iadd tokBuf (← ishlImm ti 2))
    -- **An id outside the table contributes nothing.**
    --
    -- Reading `dec_off` at an id the table does not have is not a wrong answer
    -- but an out-of-bounds load, and the length it returns then drives the copy
    -- below — so the failure is a buffer overrun and it arrives as a core dump.
    -- The converter's job is to make this unreachable by covering every id the
    -- model can emit, added tokens included; this is here so that a converter
    -- that fails at it cannot corrupt memory.
    let inRange ← ifte .ult tokId vocabSize (pure [← iconst64 1]) (pure [zero64])
    let lenL ← ifte .ne (inRange.headD zero64) zero64
      (pure [← uload32_64 (← iadd decLenPtr (← ishlImm tokId 2))])
      (pure [zero64])
    let decLen := lenL.headD zero64
    let offL ← ifte .ne (inRange.headD zero64) zero64
      (pure [← uload32_64 (← iadd decOffPtr (← ishlImm tokId 2))])
      (pure [zero64])
    let srcPtr ← iadd bytePool (offL.headD zero64)
    -- **What fits, and not a byte more.**
    --
    -- `textOut` holds `textMaxBytes`, and how many bytes a token spells is a
    -- property of the vocabulary rather than of anything the caller controls.
    -- So the copy is clamped: once the buffer is full `take` is zero and the
    -- remaining tokens contribute nothing, which loses the tail of a reply
    -- rather than the contents of the buffer after it.
    let roomL ← ifte .ult tp cap (pure [← isub cap tp]) (pure [zero64])
    let room := roomL.headD zero64
    let takeL ← ifte .ult decLen room (pure [decLen]) (pure [room])
    let take := takeL.headD zero64
    forLoop take fun i => do
      let byt ← uload8_64 (← iadd srcPtr i)
      istore8 byt (← iadd textOut (← iadd tp i))
    iadd tp take
  storeI64 finalTp (← absAddr ptr c.textLen)

end TokenizerCommon
