"""ClickBench's numeric queries, hot, on the first 10 million rows of `hits`.

    python bench.py <engine>      # appends to <run dir>/queries.jsonl

Each engine loads the rows once (its load is not timed, as ClickBench's hot
runs are not), then runs every query in QUERIES once untimed and REPS times
timed; every run's time is kept. Engines: `duckdb` (its own table in memory,
all the machine's threads) and `base` (lean/algorithms/Bench/Analytics.lean,
the columns on the GPU). Base's answers are checked against DuckDB's, exactly,
before its times count.
"""

import json
import os
import sys
import time

import numpy as np

from prepare import COLUMNS, DATA, GLOB

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
ART = os.path.join(HERE, "build", "base", "analytics.cbor")
REPS = 10

# ClickBench's numbering.
QUERIES = {
    "Q1": "SELECT COUNT(*) FROM hits WHERE AdvEngineID <> 0",
    "Q2": "SELECT SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth) FROM hits",
    "Q3": "SELECT AVG(UserID) FROM hits",
    "Q6": "SELECT MIN(EventDate), MAX(EventDate) FROM hits",
    "Q7": "SELECT AdvEngineID, COUNT(*) FROM hits WHERE AdvEngineID <> 0 GROUP BY AdvEngineID ORDER BY COUNT(*) DESC",
}


def run_dir():
    d = os.path.join(HERE, "results", os.environ.get("ANALYTICS_RUN", "adhoc"))
    os.makedirs(d, exist_ok=True)
    return d


def times(fn):
    fn()
    ts = []
    for _ in range(REPS):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return ts


class DuckDB:
    def __init__(self):
        import duckdb

        self.c = duckdb.connect()
        self.c.execute(f"CREATE TABLE hits AS SELECT {', '.join(COLUMNS)} FROM read_parquet('{GLOB}')")

    def query(self, q):
        sql = QUERIES[q]
        return lambda: self.c.execute(sql).fetchall()


class Base:
    def __init__(self):
        sys.path.append(os.path.join(ROOT, "py-base", ".venv", "lib",
                                     f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"))
        from py_base import Driver

        self.drv = Driver(open(ART, "rb").read())
        cols = [np.load(os.path.join(DATA, f"{c}.npy")) for c in COLUMNS]
        self.n = len(cols[0])
        data = np.array([self.n], np.uint64).tobytes()
        for c in cols:
            b = c.tobytes()
            data += b + b"\0" * (-len(b) % 8)
        self.call("load", data)

    def call(self, name, data=b"", words=0):
        out = np.zeros(max(words, 1), np.int64)
        st = self.drv.function(name)(data, out.view(np.uint8))
        if st != 0:
            raise RuntimeError(f"{name} returned {st}")
        return out

    def query(self, q):
        n = self.n
        f = self.drv.function(q.lower())
        words = {"Q1": 1, "Q2": 2, "Q3": 2, "Q6": 2, "Q7": 256}[q]
        out = np.zeros(words, np.int64)
        buf = out.view(np.uint8)

        def run():
            if f(b"", buf) != 0:
                raise RuntimeError(f"{q} failed")
            if q == "Q1":
                return [(int(out[0]),)]
            if q == "Q2":
                return [(int(out[0]), n, int(out[1]) / n)]
            if q == "Q3":
                return [(((int(out[0]) << 32) + int(out[1])) / n,)]
            if q == "Q6":
                return [(int(out[0]), int(out[1]))]
            if out[0] != 0:
                raise RuntimeError("an AdvEngineID of 256 or more")
            groups = [(v, int(c)) for v, c in enumerate(out) if v and c]
            return sorted(groups, key=lambda g: -g[1])

        return run


def same(q, got, want):
    if q in ("Q2", "Q3"):
        return all(np.isclose(float(g), float(w), rtol=1e-12, atol=0) for g, w in zip(got[0], want[0]))
    return [tuple(map(int, r)) for r in got] == [tuple(map(int, r)) for r in want]


def main():
    engine = sys.argv[1]
    eng = {"duckdb": DuckDB, "base": Base}[engine]()
    ref = DuckDB() if engine == "base" else None
    with open(os.path.join(run_dir(), "queries.jsonl"), "a") as f:
        for q in QUERIES:
            run = eng.query(q)
            if ref is not None:
                got, want = run(), ref.query(q)()
                if not same(q, got, want):
                    raise SystemExit(f"base: {q} answered {got}, DuckDB {want}")
            ts = times(run)
            row = dict(engine=engine, query=q, sql=QUERIES[q], seconds=float(np.median(ts)), samples=ts)
            f.write(json.dumps(row) + "\n")
            print(row["engine"], q, f"{row['seconds'] * 1e3:.3f} ms", flush=True)


if __name__ == "__main__":
    main()
