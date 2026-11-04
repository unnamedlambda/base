module
public import Lean
public import AlgorithmLib.Kernel.Corpus
meta import AlgorithmLib.Kernel.Corpus
public import AlgorithmLib.ML.Ptx.Body
meta import AlgorithmLib.ML.Ptx.Body
public import GptOss.Attention
meta import GptOss.Attention
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic

@[expose] public section

/-!
# gpt-oss's warp statements as kernel bodies, against the GPU

The decode's RMSNorm, add, cache store and sink softmax, written as bodies by
`ML.Body.kernel` and run by the model; `tools/kernel_agree.py` launches the
printed bodies and compares, bit for bit:

    cd lean && lake env lean -Dexperimental.module=true --run algorithms/GptOss/EwCorpus.lean DIR
    tools/kernel_agree.py DIR

The rotation is left out: its table is 33 MB, more than the model's run is
worth here; its printed body matches the ML printer's output on the GPU.
-/

open AlgorithmLib AlgorithmLib.Kernel AlgorithmLib.Kernel.Corpus AlgorithmLib.ML GptOssAttention

namespace GptOssEwCorpus

/-- A meta buffer: the given words at their places, zero elsewhere. -/
def meta16 (ws : List (Nat × Nat)) : ByteArray :=
  ⟨((List.range 16).flatMap fun i => le32 ((ws.lookup i).getD 0)).toArray⟩

def cases : List Case :=
  let s := 100
  let sm := meta16 [(M_SEQ, s), (M_CHUNKS, s / 32), (M_TAIL, s / 32 * 32), (M_REM, s % 32)]
  [ ⟨rmsKernel, 1, 32, 3, [], [floats 11 H, floats 12 H, zeros (4 * H)], 0⟩,
    ⟨addKernel, H / 32, 32, 2, [], [floats 13 H, floats 14 H], 0⟩,
    ⟨kvStoreKernel (QO / 2) HDW, 8, 32, 3, [],
      [floats 15 QKV, zeros (4 * 8 * 1024), meta16 [(M_SLOT, 3), (M_KVSTRIDE, 1024)]], 0⟩,
    -- `ex2.approx`, within 2 ulp of the exactly rounded 2^x it is run with
    { kernel := sinkKernel, grid := 4, block := 32, nbufs := 4,
      scalars := [], bufs := [floats 16 (4 * s), sm, zeros (4 * 4 * s), floats 17 4], ulps := 2 } ]

end GptOssEwCorpus

def main (args : List String) : IO Unit := do
  let dir ← match args with
    | [d] => pure (System.FilePath.mk d)
    | _ => throw (IO.userError "usage: lean --run algorithms/GptOss/EwCorpus.lean DIR")
  Corpus.write dir GptOssEwCorpus.cases
