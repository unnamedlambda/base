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
function the artifact exports, and a script calls it by that name.

```python
from py_base import load_artifact, Base

artifact = load_artifact("../lean-artifacts/artifacts/Sha256Algorithm/sha256_app.cbor")

base = Base(artifact)                # JIT compiles — do this once
base.execute("main")                 # …then execute as often as you like
```

An artifact is CBOR, so one can also be built in Python from nested dicts,
which is what the tests do with the standard-library encoder in
`tests/cbor.py`:

```python
from py_base import Artifact, Base
import cbor

artifact = Artifact(cbor.encode({
    "functions": [...],              # the program, one exported as "double"
    "required_memory": 256,
    "data": [],
}))

data = b"\x01\x00\x00\x00\x02\x00\x00\x00"
out = bytearray(8)

base = Base(artifact)
base.execute("double", data, out)    # both buffers zero-copy
```

A program answers through `out`. What the bytes mean is the generator's to
say; `base` gives them no format.

## API

### `Artifact(bytes)`
What a generator emits: the CLIF functions, the memory size and the data
segments memory starts with, encoded as CBOR. The runtime is what decodes it,
so a malformed artifact is reported when a `Base` is built from it.

### `load_artifact(path: str) -> Artifact`
Read an artifact from the `.cbor` file a generator wrote.

### `Base(artifact: Artifact)`
An execution engine. JIT compiles the program. This is the expensive step — do
it once.

### `base.execute(name, data=None, out=None) -> int`
Call the entry point the artifact exports as `name`. `data` accepts anything
implementing the buffer protocol (`bytes`, `bytearray`, `numpy` array), and the
program answers by writing through `out` (a `bytearray`); both are zero-copy.
The `int` is the status the entry point answered: a program whose body ends in
a bare `return` answers `0`. A name the artifact does not export is a
`ValueError` naming it.

### `base.read_memory(offset, length) -> bytes`
What the program left in its own memory, at an address its generator says it
writes. A range past the end is an error, not a short answer.

### `base.memory_size() -> int`
How many bytes of memory the program runs in.

### `run(artifact, name, data=None) -> int`
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
