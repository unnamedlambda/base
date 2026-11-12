#!/usr/bin/env bash
# Device work against PyTorch through py-base: torch operations and a
# vLLM-style decode (bench.py), against PyTorch run live in the same venv.
#
#   ./run.sh [--bench torchops|vllm|all] [--rounds n]     (defaults: all, 10)
#
# Builds the artifacts it runs (tools/lean_build.sh) and py_base into
# py-base's venv, then runs bench.py with its arguments. Every build runs
# under the memory cap.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$(cd ../.. && pwd)"
VENV="$ROOT/py-base/.venv"
CAP=(systemd-run --user --scope -q -p MemoryMax=9G -p MemorySwapMax=0)

[[ -d "$VENV" ]] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q maturin -r requirements-bench.txt
(cd "$ROOT/py-base" && "${CAP[@]}" env VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH" maturin develop --release -q)
"$ROOT/tools/lean_build.sh" algorithmLib/artifacts > /dev/null
"${CAP[@]}" "$VENV/bin/python" bench.py "$@"
