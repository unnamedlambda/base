# py-base

Python bindings for [Base](../README.md) via PyO3. Zero-copy data passing
between Python and the Base execution engine.

## Setup

```bash
cd py-base
python3 -m venv .venv
source .venv/bin/activate
pip install maturin pytest pyarrow
maturin develop --release
```

`--release` is not optional in practice. `maturin develop` builds a debug
extension unless told otherwise, and nothing about the result says so — it
imports and runs, roughly fifteen times slower at everything. Loading a 19 MB
artifact measured 812 ms debug against 68 ms release. The module warns at
import when it was built that way.

## Usage

An artifact is what a Lean generator emits: one `setup`, the entry point to
call first as `main`, and any further stages by name in `extras`.

```python
from py_base import load_artifact, Base

artifact = load_artifact("../lean-artifacts/artifacts/Sha256Algorithm/sha256_app.json")

base = Base(artifact.setup)          # JIT compiles — do this once
base.execute(artifact.main)          # …then execute as often as you like
```

Building the two values directly from JSON works the same way, which is what
the tests do:

```python
from py_base import Setup, Algorithm, Base, run
import json

setup = Setup(json.dumps({
    "clif": {"functions": [...]},    # the program, as data
    "memory_size": 256,
}))
alg = Algorithm(json.dumps({"fn_idx": 1, "output": []}))

data = b"\x01\x00\x00\x00\x02\x00\x00\x00"
out = bytearray(8)

base = Base(setup)
base.execute_into(alg, data, out)     # both buffers zero-copy
```

Results come back either through `out`, or as Arrow `RecordBatch`es when the
algorithm declares an output schema:

```python
for batch in base.execute(alg):
    print(batch.to_pandas())
```

## API

### `Setup(json: str)`
A setup: the CLIF program, the memory size, the I/O offsets and any initial
memory. Constructed once.

### `Algorithm(json: str)`
Which function to call (`fn_idx`) and the output schema to read back
(`output`). Constructed once and reused across executions with no overhead.

### `Artifact`
What a generator emits, with `.setup`, `.main` and `.extras` — the last a dict
of named stages.

### `load_artifact(path: str) -> Artifact`
Read an artifact from the JSON a generator wrote.

### `Base(setup: Setup)`
An execution engine. JIT compiles the program. This is the expensive step — do
it once.

### `base.execute(algorithm, data=None) -> list[pa.RecordBatch]`
Execute. `data` accepts anything implementing the buffer protocol (`bytes`,
`bytearray`, `numpy` array, `pyarrow` buffer) — zero copy. Returns Arrow
RecordBatches through the C Data Interface if the algorithm declares an output
schema, and an empty list otherwise.

### `base.execute_into(algorithm, data, out) -> list[pa.RecordBatch]`
Execute, writing through `out` (a `bytearray`). Both buffers are zero-copy.

### `run(setup, algorithm) -> list[pa.RecordBatch]`
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
