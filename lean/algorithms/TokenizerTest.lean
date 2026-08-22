import AlgorithmLib.Gen
import AlgorithmLib.HProg
import AlgorithmLib.HProgCuda
import TokenizerCommon
import PretokCommon
import LayoutScan
import ShipScan

open Lean AlgorithmLib AlgorithmLib.IR AlgorithmLib.HProg AlgorithmLib.HProg.Sur

/-!
  # The tokenizer on its own, so it can be disagreed with

  `TokenizerCommon` and `PretokCommon` are builders, not programs: they are
  spliced into whatever host wants them, and inside a 24-layer decode there is
  no way to see what they did. This is the smallest artifact that exercises
  them — text in, token ids out, nothing else — so that
  `applications/gpt-oss/tokenizer_test.py` can run the whole 20050-case corpus
  through the CLIF and compare against the reference tokenizer directly.

  Building it before wiring the tokenizer into the model is deliberate. The
  fault that cost this application a day was a whole-model program disagreeing
  with the model while every per-piece test passed, and the reason it took so
  long to find was that nothing could see inside. A tokenizer that is only ever
  run as part of a chat loop has the same shape.
-/

namespace TokenizerTest

/-- The largest input this test accepts. The corpus is short strings; the
    buffers are sized well past it so a case that grows does not silently
    truncate. -/
def TEXT_MAX : Nat := 4096

def env : FnEnv := env% [.ht, .cuda, .fileIO]

/-! ## Memory

    Everything after the context slots and the IO region is this program's. -/

def T_BASE : Nat := 0x0100
def T_PATH_PTR : Nat := T_BASE
def T_BUF_PTR : Nat := T_BASE + 8
def T_TOKEN_COUNT : Nat := T_BASE + 16
def T_TEXT_LEN : Nat := T_BASE + 24
def T_HT_KEY : Nat := T_BASE + 32
def T_HT_VAL : Nat := T_BASE + 40
def T_CP_COUNT : Nat := T_BASE + 48
def T_OUT_COUNT : Nat := T_BASE + 56
def T_INIT : Nat := T_BASE + 64
def T_TEXT_IN : Nat := T_BASE + 128
def T_TEXT_OUT : Nat := T_TEXT_IN + TEXT_MAX
def T_TOKEN_BUF : Nat := T_TEXT_OUT + TEXT_MAX
def T_CP_BUF : Nat := T_TOKEN_BUF + 4 * TEXT_MAX
def T_CP_BYTE : Nat := T_CP_BUF + 4 * TEXT_MAX
def T_OUT_TOK : Nat := T_CP_BYTE + 4 * (TEXT_MAX + 1)
def T_MEM_SIZE : Nat := T_OUT_TOK + 4 * TEXT_MAX + 0x100

def tokMem : TokenizerCommon.TokMem :=
  { htCtx := ContextSlots.ht, cudaCtx := ContextSlots.cuda
    pathPtr := T_PATH_PTR, bufPtr := T_BUF_PTR
    tokenBuf := T_TOKEN_BUF, tokenCount := T_TOKEN_COUNT
    textIn := T_TEXT_IN, textOut := T_TEXT_OUT, textLen := T_TEXT_LEN
    htKey := T_HT_KEY, htVal := T_HT_VAL
    fileMaxBytes := 32 * 1024 * 1024 }

def pretokMem : PretokCommon.PretokMem :=
  { cpBuf := T_CP_BUF, cpByte := T_CP_BYTE, cpCount := T_CP_COUNT
    outTok := T_OUT_TOK, outCount := T_OUT_COUNT }

def tMemMap : AlgorithmLib.Layout.RegionMap :=
  [ ⟨"pathPtr", T_PATH_PTR, 8⟩, ⟨"bufPtr", T_BUF_PTR, 8⟩
  , ⟨"tokenCount", T_TOKEN_COUNT, 8⟩, ⟨"textLen", T_TEXT_LEN, 8⟩
  , ⟨"htKey", T_HT_KEY, 8⟩, ⟨"htVal", T_HT_VAL, 8⟩
  , ⟨"cpCount", T_CP_COUNT, 8⟩, ⟨"outCount", T_OUT_COUNT, 8⟩
  , ⟨"init", T_INIT, 4⟩
  , ⟨"textIn", T_TEXT_IN, TEXT_MAX⟩, ⟨"textOut", T_TEXT_OUT, TEXT_MAX⟩
  , ⟨"tokenBuf", T_TOKEN_BUF, 4 * TEXT_MAX⟩
  , ⟨"cpBuf", T_CP_BUF, 4 * TEXT_MAX⟩
  , ⟨"cpByte", T_CP_BYTE, 4 * (TEXT_MAX + 1)⟩
  , ⟨"outTok", T_OUT_TOK, 4 * TEXT_MAX⟩ ]

theorem tokenizerTestMap_ok :
    tMemMap.okB = true ∧ tMemMap.withinB T_MEM_SIZE = true := by native_decide

/-! ## What the caller passes

    `[text_len : u32][pad][path : 256][text]`, and back comes
    `[n_tokens : u32][tokens : u32 x n]`. -/

def D_LEN : Nat := 0
def D_PATH : Nat := 16
def D_TEXT : Nat := 272

def tMainFn : HProg.Code :=
  HProg.Sur.build (env := env) do
  let ptr := basePtr
  let dataPtr ← load64 (← absAddr ptr 0x18)
  let outPtr ← load64 (← absAddr ptr 0x28)
  -- the file is read once, however many strings arrive afterwards
  let flag ← load32 (← absAddr ptr T_INIT)
  let zero32 ← iconst32 0
  when .eq flag zero32 (do
    cudaInit ptr
    storeI64 (← iaddImm dataPtr D_PATH) (← absAddr ptr T_PATH_PTR)
    TokenizerCommon.loadTokenizerM tokMem
    storeI32 (← iconst32 1) (← absAddr ptr T_INIT))
  -- the text, into this program's own buffer
  let len ← uload32_64 (← iaddImm dataPtr D_LEN)
  let textBase ← iaddImm ptr T_TEXT_IN
  let srcBase ← iaddImm dataPtr D_TEXT
  forLoop len fun i => do
    let byt ← uload8_64 (← iadd srcBase i)
    istore8 byt (← iadd textBase i)
  storeI64 len (← absAddr ptr T_TEXT_LEN)
  PretokCommon.tokenizeTextM tokMem pretokMem
  -- and out
  let n ← load { ty := .i64, notrapAligned := true } (← absAddr ptr T_OUT_COUNT)
  storeI32 (← ireduce32 n) outPtr
  let outTok ← iaddImm ptr T_OUT_TOK
  forLoop n fun i => do
    let tk ← load32 (← iadd outTok (← ishlImm i 2))
    storeI32 tk (← iadd outPtr (← iaddImm (← ishlImm i 2) 4))

def tShippedBodies : List HProg.Code := [ tMainFn ]

theorem tokenizerTestShipped_wf :
    tShippedBodies.all (HProg.wf env HProg.ptrParams) = true := by
  native_decide

def tClifIR : Program :=
  program <|
    noopFunction :: tShippedBodies.attach.zipIdx.map
      (fun p =>
        HProg.compileFn (p.2 + 1) p.1.1 env
          (hwf := List.all_eq_true.mp tokenizerTestShipped_wf p.1.1 p.1.2))

def zeros (n : Nat) : List UInt8 := List.replicate n 0

def u32le (v : Nat) : List UInt8 :=
  [ UInt8.ofNat (v % 256), UInt8.ofNat (v / 256 % 256)
  , UInt8.ofNat (v / 65536 % 256), UInt8.ofNat (v / 16777216 % 256) ]

def T_HOST_LEN : Nat := 0x0080

/-- The caller always passes a full-sized buffer and says in `D_LEN` how much
    of the text is real, so the artifact's input size is a constant. -/
def tInitialMemory : List UInt8 :=
  zeros T_HOST_LEN ++ u32le (D_TEXT + TEXT_MAX)
    ++ zeros (T_MEM_SIZE - T_HOST_LEN - 4)

def tSetup : Setup := {
  clif := tClifIR
  memory_size := T_MEM_SIZE
  initial_memory := tInitialMemory
}

#eval LayoutScan.check "TokenizerTest" [``tMemMap]

def artifacts : Array Json :=
  #[ toJsonArtifact "tokenizer_test" tSetup { fn_idx := u32 1 } [] ]

end TokenizerTest

def main (args : List String) : IO Unit := do
  emitArtifacts (← requireOutputDir args) TokenizerTest.artifacts

#eval ShipScan.check "TokenizerTest"
