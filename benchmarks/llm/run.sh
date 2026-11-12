#!/usr/bin/env bash
# LLM decode, from scratch: gpt-oss-20b (MXFP4) on llama.cpp and on base,
# the experts offloaded, as each engine runs a 20B MoE on a 12 GB card.
#
#   ./run.sh
#
# 1. builds llama.cpp's llama-bench at its pinned release (CUDA), and base's
#    decode artifact (lean/algorithms/GptOss/Decode.lean);
# 2. llama.cpp: its offload sweep (`-ncmoe`, the layers whose experts stay on
#    the CPU) at the deepest context, then the fastest setting that fits the card at
#    each depth in DEPTHS (the context already held when the prompt and the
#    decode are timed): prompt and decode tokens a second, llama-bench's own;
# 3. base: applications/gpt-oss/bench.py at the same depths (the depth's
#    tokens held, then a 512-token prompt through its prefill, then 128
#    decoded after them).
#
# The engines' steps run alone and without the memory cap: each keeps about
# 10 GB of experts resident in host memory (llama.cpp's mapped, base's
# pinned), so nothing else may run beside them, and never the two at once.
# Builds run under the 9 GB cap.
#
# Each invocation is a run in results/<time>-<git rev>/ (llama-sweep.jsonl,
# llama.jsonl, base-<depth>.txt, times.tsv); REPORT.md is rewritten from it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT=$(cd ../.. && pwd)
MODEL="$ROOT/data/llm/gpt-oss-20b-MXFP4.gguf"
MODEL_URL="https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/15dcfc7e7c608c6161507211ed43c5b590801b4b/gpt-oss-20b-MXFP4.gguf"
MODEL_SHA=27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901
LLAMA_TAG=b11900  # 74b104c9ba5717116b7e65e0cd35397778653304, 2026-10-26
BANK="$ROOT/data/gptoss-bank"
DEPTHS="512 8192"
cap() { systemd-run --user --scope -q -p MemoryMax=9G -p MemorySwapMax=0 "$@"; }

[[ -x .venv/bin/cmake ]] || { python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }
export PATH="$PWD/.venv/bin:$PATH"
if [[ ! -f $MODEL ]]; then
    mkdir -p "$(dirname "$MODEL")"
    curl -fsSL -C - -o "$MODEL.part" "$MODEL_URL"
    echo "$MODEL_SHA  $MODEL.part" | sha256sum -c
    mv "$MODEL.part" "$MODEL"
fi
[[ -d $BANK ]] || { echo "base's weight bank $BANK is missing (applications/gpt-oss/prepare.py)" >&2; exit 1; }
# base multiplies by llama.cpp's own Q8_0 attention and lm_head, appended to
# the bank's dense.bin: prepare.py's 2476030208 bytes and then 1368161280 of
# them, so any other length (none appended, or an append cut short) redoes it
[[ $(stat -c %s "$BANK/dense.bin") -eq 3844191488 ]] || \
    cap python3 "$ROOT/applications/gpt-oss/prepare_q8.py" "$MODEL" --bank "$BANK"
LLAMA=build/llama.cpp
[[ -d $LLAMA ]] || git clone -q --depth=1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp.git "$LLAMA"
[[ $(git -C "$LLAMA" rev-parse HEAD) == 74b104c9ba5717116b7e65e0cd35397778653304 ]] || { echo "$LLAMA is not at $LLAMA_TAG" >&2; exit 1; }
[[ -f $LLAMA/build/build.ninja ]] || cap env PATH="$PATH" cmake -S "$LLAMA" -B "$LLAMA/build" -G Ninja -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > /dev/null
cap env PATH="$PATH" cmake --build "$LLAMA/build" --target llama-bench -j 6 > /dev/null
( cd "$ROOT" && tools/lean_build.sh gengptossdecode > /dev/null )
mkdir -p build/base
cap "$ROOT/lean/.lake/build/bin/gengptossdecode" build/base > /dev/null

RUN="$(date +%Y-%m-%d-%H%M%S)-$(git rev-parse --short HEAD)"
OUT=results/$RUN
mkdir -p "$OUT"
step() {
    local t0 rc=0
    t0=$(date +%s.%N)
    "$@" || rc=$?
    printf '%s\t%s\t%.1f\n' "$*" "$([[ $rc == 0 ]] && echo ok || echo failed)" "$(echo "$(date +%s.%N) - $t0" | bc)" >> "$OUT/times.tsv"
    [[ $rc == 0 ]] || { echo "FAILED: $*" >&2; exit 1; }
}
BENCH="$LLAMA/build/bin/llama-bench"

# llama.cpp: the offload sweep at the deepest context (a setting that does not
# fit the card there fails alone, in its own process), then the fastest that
# fit at every depth
DEEPEST=${DEPTHS##* }
for n in 0 1 2 3 4 6 8 12; do
    "$BENCH" -m "$MODEL" -ngl 99 -fa 1 -ncmoe "$n" -p 512 -n 128 -d "$DEEPEST" -r 3 -o jsonl >> "$OUT/llama-sweep.jsonl" \
        2>> "$OUT/llama-sweep.log" || echo "ncmoe $n did not run" >> "$OUT/llama-sweep.log"
done
NCMOE=$(python3 report.py --best-ncmoe "$OUT/llama-sweep.jsonl")
step "$BENCH" -m "$MODEL" -ngl 99 -fa 1 -ncmoe "$NCMOE" -p 512 -n 128 -d "${DEPTHS// /,}" -r 3 -o jsonl > "$OUT/llama.jsonl"

# base, at the same depths: 512 prompt tokens after the depth's, then 128 decoded
for d in $DEPTHS; do
    step "$ROOT/py-base/.venv/bin/python" "$ROOT/applications/gpt-oss/bench.py" build/base/gptoss_decode.cbor \
        --bank "$BANK" --depth "$d" --prompt-len 512 --gen 128 > "$OUT/base-$d.txt"
done

python3 report.py "$RUN" > "$OUT/REPORT.md"
cp "$OUT/REPORT.md" REPORT.md
echo "run $RUN done: $OUT/REPORT.md" >&2
