"""Inputs for the analytics benchmark: the first 10 of ClickBench's 100
`hits` partitions (1 million rows each, from its published parquet files),
checked against SHA256SUMS, and the columns base's engine reads, extracted
once by DuckDB to numpy arrays.

    .venv/bin/python prepare.py
"""

import hashlib
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "data", "clickbench"))
URL = "https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_{}.parquet"
PARTS = 10
COLUMNS = {"AdvEngineID": np.int16, "ResolutionWidth": np.int16, "UserID": np.int64, "EventDate": np.uint16,
           "RegionID": np.int32}
GLOB = os.path.join(DATA, "hits_*.parquet")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parts():
    want = dict(reversed(line.split()) for line in open(os.path.join(HERE, "SHA256SUMS")))
    os.makedirs(DATA, exist_ok=True)
    for i in range(PARTS):
        name = f"hits_{i}.parquet"
        path = os.path.join(DATA, name)
        if not os.path.exists(path):
            subprocess.run(["curl", "-fsSL", "-o", path + ".part", URL.format(i)], check=True)
            os.rename(path + ".part", path)
        if sha256(path) != want[name]:
            raise SystemExit(f"{path} does not match SHA256SUMS")


def columns():
    import duckdb

    for name, dt in COLUMNS.items():
        out = os.path.join(DATA, f"{name}.npy")
        if not os.path.exists(out):
            col = duckdb.sql(f"SELECT {name} FROM read_parquet('{GLOB}')").fetchnumpy()[name]
            np.save(out, np.ascontiguousarray(col, dtype=dt))


if __name__ == "__main__":
    parts()
    columns()
