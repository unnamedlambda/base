#!/usr/bin/env bash
# Downscaling 4K video frames to 1080p: Base against the image crate,
# fast_image_resize, pic-scale, FFmpeg's swscale, Google's libyuv, zimg,
# libvips, Pillow-SIMD and NVIDIA's NPP, for the antialiased bilinear filter,
# the 2x2 average and Lanczos3.
# Builds the artifact and the harness, then prints every table, after checking
# that each library ran its SIMD.
#
#   ./run.sh
#
# Needs nasm (FFmpeg's assembly), libzimg-dev and libvips-dev, the CUDA
# toolkit's NPP, and an NVIDIA GPU. Pillow-SIMD is built with AVX2 into its own
# venv under target/ the first time.
# The one-core table runs pinned to core 3; the others use the whole machine,
# so the numbers are only steady on an idle one.
#
# Each invocation is a run in results/<time>-<git rev>/: every table's lines
# (fps.jsonl, one per engine and table) and REPORT.md, the tables printed.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
ART="$ROOT/target/image-art"
BIN="$ROOT/target/image-bench/release/image-bench"
CAP=(systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0)

echo "building" >&2
"$ROOT/tools/lean_build.sh" genimagebench >&2
mkdir -p "$ART"
"$ROOT/lean/.lake/build/bin/genimagebench" "$ART"
# bindgen reads FFmpeg's headers through libclang, whose own headers may be
# missing; GCC's serve.
BINDGEN_EXTRA_CLANG_ARGS="-I$(gcc -print-file-name=include)" RUSTFLAGS="-C target-cpu=native" \
  "${CAP[@]}" cargo build --release -q -j 2 --target-dir "$ROOT/target/image-bench"

PY="$ROOT/target/pillow-venv/bin/python"
if [ ! -x "$PY" ]; then
  python3 -m venv "$ROOT/target/pillow-venv"
  CC="cc -mavx2" "${CAP[@]}" "$ROOT/target/pillow-venv/bin/pip" install -q --no-cache-dir \
    --no-binary pillow-simd pillow-simd -C jpeg=disable >&2
fi
RUN="results/$(date +%Y-%m-%d-%H%M%S)-$(git -C "$ROOT" rev-parse --short=8 HEAD)"
mkdir -p "$RUN"
export IMAGE_RESULTS="$PWD/$RUN/fps.jsonl"
exec > >(tee "$RUN/REPORT.md")
DUMP="$ROOT/target/image-dump"
mkdir -p "$DUMP"
DUMP="$DUMP" "$BIN" "$ART/image_bench.cbor" dump >/dev/null

echo "  CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'), $(nproc) threads"
echo "  GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
echo "  rustc: $(rustc --version), -C target-cpu=native"
"${CAP[@]}" taskset -c 3 "$BIN" "$ART/image_bench.cbor" audit | tail -n +2
"${CAP[@]}" taskset -c 3 "$BIN" "$ART/image_bench.cbor" one | tail -n +2
echo
echo "Pillow-SIMD, on the same frames by the same method"
"${CAP[@]}" taskset -c 3 "$PY" -I pillow_bench.py "$DUMP" one
"${CAP[@]}" "$BIN" "$ART/image_bench.cbor" cores | tail -n +2
"${CAP[@]}" "$PY" -I pillow_bench.py "$DUMP" cores
"${CAP[@]}" "$BIN" "$ART/image_bench.cbor" gpu | tail -n +2
