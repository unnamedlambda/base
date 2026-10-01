import Scan.GptOssSurface
import Tokenizer.Test

/-!
  # What the tokenizer claims, and what it is only measured to do

  The four BPE functions and the pre-tokenizer's scan are builders spliced into
  whatever host wants them, so the claims here are about the one artifact that
  runs them on their own — text in, token ids out. That artifact exists so the
  tokenizer can be disagreed with: inside a chat loop there is no way to see
  what it did, and this application has already lost a day to a whole-model
  program that every per-piece test agreed with.
-/

open Lean

namespace TokenizerScan

/-- **The claims the standalone tokenizer artifact makes.**

    Structural, like every guard in this application: the regions of its
    memory do not overlap and stay inside it. That its one body is well-formed
    is no longer a theorem to name --- `compileProg` decides it when the
    generator runs, and refuses to write the artifact otherwise. -/
def roots : List Name := [ ``TokenizerTest.tokenizerTestMap_ok ]

/-- **What the tokenizer is relied on to do, and is not proven to.** -/
def openObligations : List String :=
  [ "PretokDenotesTheRegex — that the ordered alternation in the table denotes \
     the same split as the checkpoint's own regex. The table is an interpreter \
     for a small fragment (ordered alternation, classes with repeat counts, a \
     contraction set, one lookahead), and three constructs a real engine \
     backtracks for are rewritten as forward passes: RUNBUT for `\\s+(?!\\S)`, \
     RUNTO for `\\s*[\\r\\n]+`, STARPLUS for two overlapping letter classes. \
     Each rewrite is an argument, not a theorem. Evidence: 20050/20050 splits \
     and token ids against the reference tokenizer for both gpt-oss and Qwen2 \
     (applications/gpt-oss/tokenizer_test.py, tools/pretok_test.py)."
  , "BpeIsLowestRankFirst — that merging the lowest-ranked adjacent pair until \
     none applies is what the model's tokenizer does. Measured on the same \
     corpus, not stated."
  , "NoNormalisation — the CLIF path does not normalise, and a checkpoint that \
     asks for one will disagree on text holding decomposed characters. This is \
     *measured*, not guessed: Qwen2 declares NFC and scores 18929/20050 on raw \
     input and 20050/20050 when the same input is normalised first, so the gap \
     is exactly NFC and nothing else. gpt-oss declares `normalizer: null`, \
     which is why the model this application serves is unaffected. Closes by: \
     a composition table in the tokenizer file and a pass over the code points \
     before the split."
  , "Utf8DecodeIsTotal — a byte that cannot begin a sequence, and a truncated \
     sequence at the end of input, are taken as their own bytes rather than \
     rejected. That makes the decode total and the scan always advance, and it \
     loses nothing a byte-level BPE can represent. Argued, not stated."
  , "TableFidelity — that `tools/pretok.py` emits the alternation the \
     checkpoint's `tokenizer.json` specifies. It matches on the regex text \
     rather than the model name, which is a fact about the converter and not \
     about anything here." ]

/-- Facts reported so that a change shows up as a number moving. -/
def measured : List String :=
  [ s!"text buffer: {TokenizerTest.TEXT_MAX} bytes"
  , s!"host memory: {TokenizerTest.T_MEM_SIZE} bytes"
  , "corpus: 20050 cases, gpt-oss 20050/20050"
  , "corpus: 20050 cases, Qwen2 18929/20050 raw, 20050/20050 pre-normalised" ]

end TokenizerScan

/-- Both roots are `native_decide`: each is a decidable fact about data the
    generator built, which is what it is for. -/
def tokenizerNativeRoster : List Name := TokenizerScan.roots

open TrustScan TokenizerScan in
#eval runScanWith gptOssSurface "tokenizer" roots tokenizerNativeRoster

#eval do
  IO.println s!"[tokenizer] roots scanned: {TokenizerScan.roots.length}"
  IO.println s!"[tokenizer] open obligations: {TokenizerScan.openObligations.length}"
  for s in TokenizerScan.openObligations do
    IO.println s!"[tokenizer] NOT YET STATED: {s}"
  for m in TokenizerScan.measured do
    IO.println s!"[tokenizer] measured: {m}"
