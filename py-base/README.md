# py-base

Python bindings for [Base](../README.md) via PyO3, over base's C ABI — the same
six functions a C or Lean host calls. Zero-copy data passing between Python and
the Base execution engine.

## Setup

```bash
cd py-base
python3 -m venv .venv
source .venv/bin/activate
pip install maturin pytest
maturin develop --release
```

`--release` is not optional in practice. `maturin develop` builds a debug
extension unless told otherwise, and nothing about the result says so — it
imports and runs, roughly fifteen times slower at everything. Loading a 19 MB
artifact measured 812 ms debug against 68 ms release. The module warns at
import when it was built that way.

## Usage

An artifact is what a Lean generator emits: the CLIF functions, the size of the
memory they run in, and that memory's initial contents. An entry point is a
function index, and which index is which stage is the generator's knowledge —
so a script names those numbers itself, as `entries.py` does beside the
applications in this repository.

```python
from py_base import load_artifact, Base

ENTRY = 1   # the index this artifact's generator gave its entry point

artifact = load_artifact("../lean-artifacts/artifacts/Sha256Algorithm/sha256_app.json")

base = Base(artifact)                # JIT compiles — do this once
base.execute(ENTRY)                  # …then execute as often as you like
```

Building the artifact directly from JSON works the same way, which is what the
tests do:

```python
from py_base import Artifact, Base, run
import json

artifact = Artifact(json.dumps({
    "functions": [...],              # the program, as data
    "memory_size": 256,
}))
ENTRY = 1

data = b"\x01\x00\x00\x00\x02\x00\x00\x00"
out = bytearray(8)

base = Base(artifact)
base.execute_into(ENTRY, data, out)   # both buffers zero-copy
```

A program answers through `out`. What the bytes mean is the generator's to
say; `base` gives them no format.

## API

### `Artifact(json: str)`
What a generator emits: the CLIF functions, the memory size and the data
segments memory starts with. The runtime is what parses it, so malformed JSON
is reported when a `Base` is built from it.

### `load_artifact(path: str) -> Artifact`
Read an artifact from the JSON a generator wrote.

### `Base(artifact: Artifact)`
An execution engine. JIT compiles the program. This is the expensive step — do
it once.

### `base.execute(fn_idx, data=None) -> None`
Execute. `data` accepts anything implementing the buffer protocol (`bytes`,
`bytearray`, `numpy` array) — zero copy.

### `base.execute_into(fn_idx, data, out=None) -> None`
Execute, writing through `out` (a `bytearray`). Both buffers are zero-copy.

### `base.read_memory(offset, length) -> bytes`
What the program left in its own memory, at an address its generator says it
writes. A range past the end is an error, not a short answer.

### `base.memory_size() -> int`
How many bytes of memory the program runs in.

### `run(artifact, fn_idx, data=None) -> None`
One-shot: compile and execute in a single call. For a program run once; use
`Base` for anything run twice.

The GIL is released for the duration of an execution, so other Python threads
run while a program does.

## Testing

```bash
source .venv/bin/activate
maturin develop --release
pytest -v
```
