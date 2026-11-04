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
from py_base import Driver

with open("../lean/.lake/build/artifacts/sha256_app.cbor", "rb") as f:
    artifact = f.read()

driver = Driver(artifact)            # JIT compiles — do this once
main = driver.function("main")       # the function, found once
main(b"input.bin")                   # …then call it as often as you like:
                                     # its input names the file it hashes
```

An artifact is CBOR, so one can also be built in Python from nested dicts,
which is what the tests do with the standard-library encoder in
`tests/cbor.py`:

```python
from py_base import Driver
import cbor

artifact = cbor.encode({
    "primitives": [],                # the C functions it calls, each once
    "functions": [...],              # the program, one named "double"
    "arena_size": 256,
    "data": [],
})

input = b"\x01\x00\x00\x00\x02\x00\x00\x00"
output = bytearray(8)

double = Driver(artifact).function("double")
double(input, output)                # both buffers zero-copy
```

A program answers through `output`, and what it leaves in its arena stays
there for the host, or the next call, to read. What the bytes mean is the
generator's to say; `base` gives them no format.

## API

The four calls of the C API, in Python's terms, and nothing built on them.

### `Driver(artifact: bytes)`
An encoded artifact, as a generator wrote it, loaded: decoded, its program JIT
compiled and its arena allocated. A malformed artifact is a `ValueError` saying
what is wrong. This is the expensive step — do it once.

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

### `driver.arena -> memoryview`
The arena itself: no copy, writable, and holding the driver alive.
`len(driver.arena)` is its size, and `numpy.frombuffer(driver.arena, ...)`
or `struct.unpack_from(fmt, driver.arena, offset)` reads it in place. A plain
view reads without atomics, so while calls run, bytes a running program may
touch are the program's.

## Threads

One `Driver`, and its functions, may be used from any number of Python threads.
The GIL is released while a program runs, so calls run at once, and a thread
blocked inside the program (on a lock the program keeps) leaves
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
