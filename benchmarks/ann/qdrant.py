"""Qdrant as a benchmark engine: a local server; writes through gRPC, searches
through REST batches.

The server (data/ann/qdrant-bin/qdrant, v1.20.0) is started fresh per engine
with empty storage under data/ann/qdrant-storage, as a child of the benchmark
process, so it lives inside the same memory cap. HNSW m=16, ef_construct=128
(Qdrant's defaults), vectors in RAM.

Qdrant indexes in the background: rows it has not yet put in a graph are
searched exactly. `wait_indexed` blocks until the optimizer is idle; the
harnesses call it after a build and before each streaming search, so every
search runs against the index Qdrant would serve once it caught up.
"""

import os
import shutil
import subprocess
import time

import numpy as np

from common import DATA
from engines import graph_ef

BIN = os.path.join(DATA, "qdrant-bin", "qdrant")
STORAGE = os.path.join(DATA, "qdrant-storage")
GRPC_PORT, HTTP_PORT = 6334, 6333


class Qdrant:
    def __init__(self, metric, dim, capacity):
        from qdrant_client import QdrantClient, models

        self.m = models
        shutil.rmtree(STORAGE, ignore_errors=True)
        env = dict(
            os.environ,
            QDRANT__STORAGE__STORAGE_PATH=STORAGE,
            QDRANT__SERVICE__GRPC_PORT=str(GRPC_PORT),
            QDRANT__SERVICE__HTTP_PORT=str(HTTP_PORT),
            QDRANT__TELEMETRY_DISABLED="true",
            QDRANT__LOG_LEVEL="WARN",
        )
        self.proc = subprocess.Popen([BIN], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=DATA)
        self.c = QdrantClient(host="127.0.0.1", grpc_port=GRPC_PORT, port=HTTP_PORT, prefer_grpc=True, timeout=600)
        for _ in range(200):
            try:
                self.c.get_collections()
                break
            except Exception:
                time.sleep(0.1)
        dist = models.Distance.EUCLID if metric == "sqeuclidean" else models.Distance.DOT
        self.c.create_collection(
            "bench",
            vectors_config=models.VectorParams(size=dim, distance=dist, on_disk=False),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
        )

    def set_tags(self, indptr, tags):
        """Items carry their tags as an indexed integer payload, `t`, so the
        server filters natively. The index exists before any vector arrives,
        which is what lets Qdrant build its graph filter-aware."""
        self.indptr, self.tags = indptr, tags
        self.c.create_payload_index("bench", field_name="t", field_schema=self.m.PayloadSchemaType.INTEGER, wait=True)

    def insert(self, vecs, ids):
        m = self.m
        tagged = getattr(self, "tags", None) is not None
        for s in range(0, len(ids), 1000):
            v = np.ascontiguousarray(vecs[s : s + 1000], np.float32)
            chunk = [int(i) for i in ids[s : s + 1000]]
            payloads = (
                [{"t": self.tags[self.indptr[i] : self.indptr[i + 1]].tolist()} for i in chunk] if tagged else None
            )
            self.c.upsert("bench", points=m.Batch(ids=chunk, vectors=v.tolist(), payloads=payloads), wait=True)

    def delete(self, ids):
        m = self.m
        for s in range(0, len(ids), 10000):
            self.c.delete("bench", points_selector=m.PointIdsList(points=[int(i) for i in ids[s : s + 10000]]), wait=True)

    def wait_indexed(self):
        # Green is reported before the optimizer starts, and a segment under
        # Qdrant's indexing threshold (10 MB, the default kept here) is never
        # indexed but searched exactly: wait until every point is indexed or
        # the indexed count has held still, green, for 5 s.
        last, still = -1, 0
        while True:
            time.sleep(0.5)
            info = self.c.get_collection("bench")
            n = info.indexed_vectors_count or 0
            green = info.status == self.m.CollectionStatus.GREEN
            if n >= (info.points_count or 0) and green:
                return
            still = still + 1 if n == last and green else 0
            if still >= 10:
                return
            last = n

    def settings_for(self, k: int):
        return graph_ef(k)

    def search_filtered(self, q, ef, k: int, required):
        """`search`, each query restricted server-side to items holding all
        of its required tags."""
        return self.search(q, ef, k, filters=[{"must": [{"key": "t", "match": {"value": int(t)}} for t in r]} for r in required])

    def search(self, q, ef, k: int, filters=None):
        """REST batches: the Python gRPC client caps at about a third of what
        the server answers (4.2k against 11.8k queries/s, measured on 200k SIFT
        rows); plain JSON batches reach 9.7k end to end. A batch carries about
        10,000 answer points, so 1000 queries at k=10 and 10 at k=1000."""
        import json
        import urllib.request

        out = np.full((len(q), k), -1, np.int64)
        step = max(1, 10000 // k)
        for s in range(0, len(q), step):
            body = json.dumps(
                {
                    "searches": [
                        {"query": v.tolist(), "limit": k, "params": {"hnsw_ef": max(ef, k)}}
                        | ({"filter": filters[s + j]} if filters else {})
                        for j, v in enumerate(q[s : s + step])
                    ]
                }
            ).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{HTTP_PORT}/collections/bench/points/query/batch",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            res = json.loads(urllib.request.urlopen(req, timeout=600).read())["result"]
            for j, r in enumerate(res):
                ids = [p["id"] for p in r["points"]]
                out[s + j, : len(ids)] = ids
        return out

    def __del__(self):
        try:
            self.proc.terminate()
            self.proc.wait(10)
        except Exception:
            pass
