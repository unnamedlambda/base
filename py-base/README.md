# py-base

Python bindings for [Base](../README.md) via PyO3, over base's C ABI — the same
four functions a C or Lean host calls. Zero-copy data passing between Python and
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

An artifact is what a Lean generator emits: the primitives it calls, the CLIF
functions, the size of the arena they run in, and the arena's initial
contents. A script calls a function of it by the function's name.

```python
from py_base import load_artifact, Driver

artifact = load_artifact("../lean/.lake/build/artifacts/sha256_app.cbor")

driver = Driver(artifact)            # JIT compiles — do this once
main = driver.function("main")       # the function, found once
main()                               # …then call it as often as you like
```

An artifact is CBOR, so one can also be built in Python from nested dicts,
which is what the tests do with the standard-library encoder in
`tests/cbor.py`:

```python
from py_base import Artifact, Driver
import cbor

artifact = Artifact(cbor.encode({
    "primitives": [],                # the C functions it calls, each once
    "functions": [...],              # the program, one named "double"
    "arena_size": 256,
    "data": [],
}))

input = b"\x01\x00\x00\x00\x02\x00\x00\x00"
output = bytearray(8)

double = Driver(artifact).function("double")
double(input, output)                # both buffers zero-copy
```

A program answers through `output`, and what it leaves in its arena stays
there for the host, or the next call, to read. What the bytes mean is the
generator's to say; `base` gives them no format.

## API

### `Artifact(bytes)`
What a generator emits: the primitives, the CLIF functions, the arena size and
the data segments the arena starts with, encoded as CBOR. The runtime is what decodes it,
so a malformed artifact is reported when a `Driver` is loaded from it.

### `load_artifact(path: str) -> Artifact`
Read an artifact from the `.cbor` file a generator wrote.

### `Driver(artifact: Artifact)`
The artifact, loaded: its program JIT compiled and its arena allocated. This is the expensive step — do
it once.

### `driver.function(name) -> Function`
The function named `name`, as the native function it is, the one
`base_driver_function` answers in C. Find it once and keep it: the lookup is
the only part of a call that is not the program. A name the artifact has no
function by is a `ValueError` naming it. A `Function` holds its driver, so the
driver lives as long as any `Function` of it does.

### `function(input=None, output=None) -> int`
Call the program. `input` is `bytes`, and the program answers by writing
through `output`, a writable contiguous byte buffer (a `bytearray`, a `uint8`
NumPy array); both are zero-copy. The `int` is the status the function
answered: a program whose body ends in a bare `return` answers `0`.

### `driver.wait(offset, expected, timeout_ns=-1)`, `driver.wake(offset, count=-1)`, `py_base.now()`
Waiting on a 4-byte-aligned word of the arena, waking its waiters,
and the clock: the system's own calls, which programs make too, so a Python
thread and a program's threads meet at the same words. The GIL is released
while waiting. A wait may end for no reason: loop on the condition.

### `driver.arena -> memoryview`
The arena itself: no copy, writable, and holding the driver alive.
`len(driver.arena)` is its size, and `numpy.frombuffer(driver.arena, ...)`
reads it in place. A plain view reads without atomics, so while calls run,
reach bytes a running program may touch with `driver.read`.

### `driver.read(offset, length) -> bytes`
What the program left in its arena, at an address its generator says it
writes, copied: each aligned 8-byte word with one atomic load, so it is sound
while calls run. A range past the end is an error, not a short answer.

## Threads

One `Driver`, and its functions, may be used from any number of Python threads.
The GIL is released while a program runs, so calls run at once, and a thread
blocked inside the program (on a lock the program keeps), or in `driver.wait`, leaves
the others running. `output` is held for the whole call, so another thread
resizing it gets `BufferError` rather than pulling it out from under the
program. How calls share the arena is the program's to decide;
`tests/test_concurrency.py` runs `shared_sum`'s atomics and ring buffer from
Python threads.

## Testing

```bash
source .venv/bin/activate
maturin develop --release
pytest -v
```
