# py-base

Python bindings for [Base](../README.md) via PyO3. Zero-copy data passing between Python and the Base execution engine.

## Setup

```bash
cd py-base
python3 -m venv .venv
source .venv/bin/activate
pip install maturin pytest pyarrow
maturin develop --release
```

## Usage

```python
from py_base import BaseConfig, Algorithm, Base

# Parse config and algorithm from JSON (once)
config = BaseConfig('{"cranelift_ir": "...", "memory_size": 256, ...}')
alg = Algorithm('{"actions": [...], "cranelift_units": 0, ...}')

# JIT compile once
base = Base(config)

# Execute with payload data (zero-copy bytes in, bytearray out)
data = b"\x01\x00\x00\x00\x02\x00\x00\x00"
out = bytearray(8)
base.execute_into(alg, data, out)

# Or execute with Arrow RecordBatch output (zero-copy via C Data Interface)
batches = base.execute(alg)
for batch in batches:
    print(batch.to_pandas())
```

## API

### `BaseConfig(json: str)`
Parse a Base configuration from JSON. Contains the Cranelift IR and memory layout. Constructed once.

### `Algorithm(json: str)`
Parse an algorithm from JSON. Contains the action sequence and output schema. Constructed once, reused across executions with zero overhead.

### `Base(config: BaseConfig)`
Create an execution engine. JIT compiles the Cranelift IR from the config. This is the expensive step — do it once.

### `base.execute(algorithm, data=None) -> list[pa.RecordBatch]`
Execute an algorithm. `data` accepts any object implementing the Python buffer protocol (`bytes`, `bytearray`, `numpy` array, `pyarrow` buffer) — zero copy. Returns Arrow RecordBatches via the C Data Interface if the algorithm defines an output schema.

### `base.execute_into(algorithm, data, out) -> list[pa.RecordBatch]`
Execute an algorithm, writing results into `out` (a `bytearray`). Both `data` and `out` are zero-copy. Also returns Arrow RecordBatches if the algorithm defines an output schema.

### `run(config, algorithm) -> list[pa.RecordBatch]`
One-shot: JIT compile and execute in a single call.

## Testing

```bash
source .venv/bin/activate
maturin develop --release
pytest -v
```
