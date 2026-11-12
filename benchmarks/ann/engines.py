"""The engines the ANN benchmark drives, behind one interface.

Every engine starts empty and is driven only through

    insert(vectors, ids)    ids are dataset rows (int64), never reused
    delete(ids)             ids currently live
    search(queries, s, k)   -> (q, k) int64 ids, nearest first, -1 if fewer
    settings_for(k)         the search sweep for k candidates, cheapest first

so every harness (curves.py, stream.py, filtered.py) runs every engine
identically. Each engine is one
design family, run the way that family is deployed for data that changes:

    seg-cagra, seg-ivf-pq
        cuvs CAGRA (a GPU graph) and IVF-PQ (GPU compressed codes) inside the
        segment design production GPU systems use (Milvus's growing and sealed
        segments): inserts land in a write buffer searched exactly; a full
        buffer is sealed into an immutable segment built from scratch; deletes
        are one bitset shared by every segment; a segment that has lost 30% of
        its rows is rebuilt from the rest, and more than 8 segments merge the
        two smallest. Neither index is deployed bare under writes: its
        quantizer (IVF-PQ) or graph (CAGRA) never adapts to what arrives later.
    hnswlib
        a CPU graph on every core, with its native add and delete.
    scann
        ScaNN, Google's CPU partitioning + anisotropic quantization + exact
        re-order, with its native upsert and delete.
    qdrant
        a Qdrant server (qdrant.py), HNSW, the production vector database.
    ours
        the design under test (proto.py).
    exact
        GPU brute force (Brute), in the static harnesses only: it has no
        deletes.

Build parameters are each library's documented defaults for this scale:
HNSW M=16 / ef_construction=200 (128 for Qdrant, its own default), CAGRA graph
degree 32 / intermediate 64, IVF lists about 2*sqrt(n), ScaNN leaves about
2*sqrt(n) with 2-dim AH blocks.
"""

import os

import numpy as np

from common import drop_cache, to_device

CPU_THREADS = os.cpu_count()
SEAL_ROWS = 50000  # write-buffer rows before sealing a segment
MAX_SEGMENTS = 8
REBUILD_DEAD = 0.3

GRAPH_EF = [10, 20, 40, 80, 120, 200, 400, 800, 1600, 3200]
CAGRA_ITOPK = [16, 32, 64, 128, 256, 512, 1024, 2048]
PROBES = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
REFINE_MAX = 256  # cuvs refine: k must be below topk::kMaxCapacity (256)


def n_lists_for(n: int) -> int:
    """IVF lists: about 2*sqrt(n) as a power of two, and at most n/40."""
    return max(1, min(int(2 ** round(np.log2(max(1.0, np.sqrt(n) * 2)))), n // 40))


def graph_ef(k: int):
    """Search-list sizes for a graph returning k: never below k."""
    return sorted({max(e, k) for e in GRAPH_EF + [2 * k, 4 * k]})


# ── cuvs ──────────────────────────────────────────────────────────────────────


class Bitset:
    """Live bit per dataset id, on the device (1 = live)."""

    def __init__(self, cp, capacity: int):
        self.cp = cp
        self.words = cp.full((capacity + 31) // 32, 0xFFFFFFFF, cp.uint32)
        self.any_deleted = False
        self.version = 0

    def clear(self, ids):
        cp = self.cp
        idx = cp.asarray(ids, dtype=cp.int64)
        cp.bitwise_and.at(self.words, idx // 32, ~(cp.uint32(1) << (idx % 32).astype(cp.uint32)))
        self.any_deleted = True
        self.version += 1

    def live(self, ids):
        """Bool mask: which device ids (int64, -1 allowed) are live."""
        cp = self.cp
        safe = cp.maximum(ids, 0)
        bit = (self.words[safe // 32] >> (safe % 32).astype(cp.uint32)) & 1
        return (bit == 1) & (ids >= 0)


class Rows:
    """A device row store allocated once at its capacity, as the arena is:
    appends never copy what is already there.

    `x` (host or device) is the first rows. When `x` is already a whole
    device array at capacity it is adopted, not copied. A store given no
    capacity grows by half when full.
    """

    def __init__(self, cp, x, capacity=None):
        self.cp, self.n = cp, 0
        cap = max(capacity or 0, len(x), 1)
        if isinstance(x, cp.ndarray) and len(x) == cap and x.flags.c_contiguous:
            self.buf, self.n = x, len(x)
            return
        self.buf = cp.empty((cap,) + tuple(x.shape[1:]), x.dtype)
        self.append(x)

    def append(self, x):
        cp = self.cp
        if self.n + len(x) > len(self.buf):
            grown = cp.empty((max(len(self.buf) * 3 // 2, self.n + len(x)),) + self.buf.shape[1:], self.buf.dtype)
            grown[: self.n] = self.buf[: self.n]
            self.buf = grown
        for s in range(0, len(x), 100000):
            part = x[s : s + 100000]
            self.buf[self.n : self.n + len(part)] = part if isinstance(part, cp.ndarray) else cp.asarray(np.ascontiguousarray(part))
            self.n += len(part)
        drop_cache(x)

    @property
    def view(self):
        return self.buf[: self.n]


def sample_rows(cp, x, rows: int):
    """Up to `rows` rows of `x` (host or device) on the device, drawn
    without replacement in sorted order, at most 1 GB of them."""
    rows = min(rows, len(x), max(1, (1 << 30) // (4 * x.shape[1])))
    idx = np.sort(np.random.default_rng(0).choice(len(x), rows, replace=False))
    got = x[cp.asarray(idx)] if isinstance(x, cp.ndarray) else x[idx]
    if isinstance(got, cp.ndarray):
        return got
    out = cp.asarray(np.ascontiguousarray(got, np.float32))
    drop_cache(x)
    return out


class CuvsIndex:
    """One cuvs index (CAGRA or IVF-PQ), searchable under the shared id bitset.

    cuvs numbers the rows it holds 0..n-1 in insertion order (positions);
    `self.ids[p]` is the dataset id at position p. Deletes reach the index as
    a position bitset gathered from the id bitset once per bitset version.
    `search` returns (distances smaller-is-nearer, dataset ids).

    The fp32 rows are kept on the device (`vecs`): IVF-PQ re-ranks against
    them, CAGRA's index is the rows, and segments are rebuilt from them.
    """

    def __init__(self, kind: str, metric: str, cp, res):
        self.kind, self.metric, self.cp, self.res = kind, metric, cp, res
        self.filter_version = -1

    def build(self, vecs, ids, bitset: Bitset):
        """`vecs` float32 (n, d) on the host or device, `ids` int64 (n,).

        IVF-PQ trains on a sample (256 rows per list, as cuvs's own trainset
        fraction gives at 1M) and takes the rows in 100k chunks."""
        from cuvs.neighbors import cagra, ivf_pq

        cp = self.cp
        n, d = vecs.shape
        self.bitset = bitset
        self._ids = Rows(cp, cp.asarray(ids, dtype=cp.int64))
        self._vecs = Rows(cp, vecs)
        if self.kind == "cagra":
            self.index = cagra.build(
                cagra.IndexParams(metric=self.metric, graph_degree=32, intermediate_graph_degree=64),
                self.vecs,
                resources=self.res,
            )
        else:
            self.mod = ivf_pq
            lists = n_lists_for(n)
            kw = dict(
                n_lists=lists,
                metric=self.metric,
                add_data_on_build=False,
                kmeans_trainset_fraction=1.0,
                pq_dim=d // 4 if d >= 256 else d // 2,
            )
            train = self.vecs if n <= 256 * lists else sample_rows(cp, self.vecs, 256 * lists)
            self.index = ivf_pq.build(ivf_pq.IndexParams(**kw), train, resources=self.res)
            del train
            for s in range(0, n, 100000):
                part = self.vecs[s : s + 100000]
                ivf_pq.extend(self.index, part, cp.arange(s, s + len(part), dtype=cp.int64), resources=self.res)
            drop_cache(vecs)
        self.res.sync()

    @property
    def vecs(self):
        return self._vecs.view

    @property
    def ids(self):
        return self._ids.view

    def live(self):
        return self.bitset.live(self.ids)

    def prefilter(self):
        from cuvs.neighbors import filters

        cp = self.cp
        if not self.bitset.any_deleted:
            return filters.no_filter()
        if self.filter_version != self.bitset.version:
            live = self.live()
            pad = (-len(live)) % 32
            bits = cp.concatenate([live, cp.zeros(pad, bool)]).reshape(-1, 32).astype(cp.uint32)
            self.words = (bits << cp.arange(32, dtype=cp.uint32)).sum(1, dtype=cp.uint32)
            self.filter_version = self.bitset.version
        return filters.from_bitset(self.words)

    def search(self, q, setting, k: int):
        """IVF-PQ goes 500 queries at a time when its re-rank is large:
        `refine` stages queries x candidates x d floats."""
        cp = self.cp
        staged = len(q) * k * (setting[1] if self.kind == "ivf-pq" else 1) * q.shape[1] * 4
        step = max(1, len(q) * (1 << 30) // staged) if staged > (1 << 30) else len(q)
        if step >= len(q):
            return self.search_part(q, setting, k)
        parts = [self.search_part(q[s : s + step], setting, k) for s in range(0, len(q), step)]
        return cp.concatenate([p[0] for p in parts]), cp.concatenate([p[1] for p in parts])

    def search_part(self, q, setting, k: int):
        from cuvs.neighbors import cagra, refine

        cp = self.cp
        k = min(k, len(self.ids))
        if self.kind == "cagra":
            p = cagra.SearchParams(itopk_size=max(setting, k), search_width=max(1, setting // 32))
            d, pos = cagra.search(p, self.index, q, k, filter=self.prefilter(), resources=self.res)
        else:
            # ivf_pq.search takes no filter: fetch k*ratio candidates and mark
            # the deleted ones -1. Below REFINE_MAX they are re-ranked exactly
            # against the fp32 rows (refine scores -1 as infinitely far) and k
            # kept; cuvs's refine takes k < 256 only, so a larger k keeps the
            # first k live candidates in PQ order, as a hand-off needs.
            probes, ratio = setting
            dq, pos = self.mod.search(self.mod.SearchParams(n_probes=probes), self.index, q, k * ratio, resources=self.res)
            pos = cp.asarray(pos).astype(cp.int64)
            ok = pos >= 0
            if self.bitset.any_deleted:
                ok &= self.bitset.live(self.ids[cp.clip(pos, 0, len(self.ids) - 1)])
                pos = cp.where(ok, pos, -1)
            if k < REFINE_MAX:
                d, pos = refine(self.vecs, q, pos, k, metric=self.metric, resources=self.res)
            else:
                dq = cp.where(ok, cp.asarray(dq), cp.inf if self.metric == "sqeuclidean" else -cp.inf)
                first = cp.argsort(~ok, axis=1, kind="stable")[:, :k]  # live first, PQ order kept
                d, pos = cp.take_along_axis(dq, first, 1), cp.take_along_axis(pos, first, 1)
        d, pos = cp.asarray(d), cp.asarray(pos).astype(cp.int64)
        if self.metric == "inner_product":
            d = -d
        valid = (pos >= 0) & (pos < len(self.ids)) & cp.isfinite(d)
        ids = cp.where(valid, self.ids[cp.clip(pos, 0, len(self.ids) - 1)], -1)
        return cp.where(valid, d, cp.inf), ids


def exact_top(cp, vecs, ids, q, metric: str, k: int):
    """Exact (distances smaller-is-nearer, ids) of the k nearest of `vecs`,
    1000 queries at a time (a 1000 x rows distance tile)."""
    kk = min(k, len(ids))
    norms = (vecs * vecs).sum(1) if metric == "sqeuclidean" else None
    ds, is_ = [], []
    for s in range(0, len(q), 1000):
        qs = q[s : s + 1000]
        key = norms[None, :] - 2 * qs @ vecs.T if metric == "sqeuclidean" else -(qs @ vecs.T)
        top = cp.argpartition(key, kk - 1, axis=1)[:, :kk] if kk < len(ids) else cp.tile(cp.arange(len(ids)), (len(qs), 1))
        d = cp.take_along_axis(key, top, 1)
        if metric == "sqeuclidean":
            d = d + (qs * qs).sum(1)[:, None]
        ds.append(d)
        is_.append(ids[top])
    return cp.concatenate(ds), cp.concatenate(is_)


_pool = None


def device_pool():
    """One RMM pool for cupy and cuvs, as a deployment runs them: cupy's own
    pool keeps freed blocks that cuvs (allocating through RMM) then cannot
    have, and RMM without a pool makes every temporary a synchronous
    cudaMalloc/cudaFree. The pool is the driver's (cudaMallocAsync), which
    RMM's own pool fragments past at 10M rows. A statistics adaptor on top counts the bytes held,
    which the card's free memory cannot show inside a pool."""
    global _pool
    if _pool is not None:
        return
    import cupy as cp
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator

    _pool = rmm.mr.StatisticsResourceAdaptor(rmm.mr.CudaAsyncMemoryResource())
    rmm.mr.set_current_device_resource(_pool)
    cp.cuda.set_allocator(rmm_cupy_allocator)


def device_held():
    """Bytes currently allocated through the pool, or None without one (a CPU
    engine)."""
    return None if _pool is None else _pool.allocation_counts.current_bytes


class Cuvs:
    """Shared plumbing: device, resources and the delete bitset."""

    def __init__(self, kind, metric, dim, capacity):
        import cupy as cp
        from cuvs.common import Resources

        device_pool()
        self.cp, self.kind, self.metric, self.dim = cp, kind, metric, dim
        self.res = Resources()
        self.bitset = Bitset(cp, capacity)

    def delete(self, ids):
        self.bitset.clear(ids)
        self.cp.cuda.Device().synchronize()


class Segmented(Cuvs):
    """Write buffer + sealed segments + one delete bitset (see module doc)."""

    def __init__(self, *a):
        super().__init__(*a)
        cp = self.cp
        self.bv = Rows(cp, cp.empty((0, self.dim), cp.float32))
        self.bi = Rows(cp, cp.empty(0, cp.int64))
        self.segments = []

    def settings_for(self, k: int):
        if self.kind == "cagra":
            return sorted({max(s, k) for s in CAGRA_ITOPK})
        return [(p, r) for p in PROBES for r in ((2, 4) if k <= 10 else (1, 2))]

    def segment(self, vecs, ids, owned=False):
        """A sealed segment over its own rows: copied unless `owned` (rows no
        one else holds, as a merge's concatenation)."""
        seg = CuvsIndex(self.kind, self.metric, self.cp, self.res)
        seg.build(vecs if owned else vecs.copy(), ids if owned else ids.copy(), self.bitset)
        return seg

    def set_buffer(self, vecs, ids):
        self.bv, self.bi = Rows(self.cp, vecs), Rows(self.cp, ids)

    def insert(self, vecs, ids):
        """A large insert enters the buffer a segment's worth at a time, so
        the device never holds its rows twice (buffer and segments)."""
        for s in range(0, len(ids), SEAL_ROWS):
            self.insert_part(vecs[s : s + SEAL_ROWS], ids[s : s + SEAL_ROWS])
        drop_cache(vecs)

    def insert_part(self, vecs, ids):
        cp = self.cp
        self.bv.append(to_device(vecs))
        self.bi.append(cp.asarray(ids, dtype=cp.int64))
        if self.bi.n >= SEAL_ROWS:
            v, i = self.bv.view, self.bi.view
            s = 0
            while self.bi.n - s >= SEAL_ROWS:
                self.segments.append(self.segment(v[s : s + SEAL_ROWS], i[s : s + SEAL_ROWS]))
                s += SEAL_ROWS
            self.set_buffer(v[s:].copy(), i[s:].copy())
            del v, i  # the old buffer goes before any merge allocates
        self.maintain()
        cp.cuda.Device().synchronize()

    def delete(self, ids):
        super().delete(ids)
        self.maintain()
        self.cp.cuda.Device().synchronize()

    def maintain(self):
        """Drop deleted rows from the buffer; rebuild a segment that lost
        REBUILD_DEAD of its rows (or return it to the buffer when tiny); merge
        the two smallest segments while there are more than MAX_SEGMENTS."""
        cp = self.cp
        if self.bitset.any_deleted and self.bi.n:
            live = self.bitset.live(self.bi.view)
            if not bool(live.all()):
                self.set_buffer(self.bv.view[live], self.bi.view[live])
        kept = []
        for seg in self.segments:
            live = seg.live()
            n_live = int(live.sum())
            if n_live >= (1 - REBUILD_DEAD) * len(seg.ids):
                kept.append(seg)
            elif n_live < 1000:
                self.bv.append(seg.vecs[live])
                self.bi.append(seg.ids[live])
            else:
                kept.append(self.segment(seg.vecs[live], seg.ids[live]))
        self.segments = kept
        while len(self.segments) > MAX_SEGMENTS:
            self.segments.sort(key=lambda s: len(s.ids))
            a, b = self.segments[0], self.segments[1]
            if self.bitset.any_deleted:  # live rows only; otherwise no gather
                la, lb = a.live(), b.live()
                vecs, ids = cp.concatenate([a.vecs[la], b.vecs[lb]]), cp.concatenate([a.ids[la], b.ids[lb]])
            else:
                vecs, ids = cp.concatenate([a.vecs, b.vecs]), cp.concatenate([a.ids, b.ids])
            self.segments = self.segments[2:]
            del a, b  # their rows go before the merged segment's build allocates
            self.segments.insert(0, self.segment(vecs, ids, owned=True))
            del vecs, ids

    def search(self, q, s, k: int):
        cp = self.cp
        dq = cp.asarray(q)
        parts = []
        if self.bi.n:
            parts.append(exact_top(cp, self.bv.view, self.bi.view, dq, self.metric, k))
        for seg in self.segments:
            parts.append(seg.search(dq, s, k))
        d = cp.concatenate([p[0] for p in parts], 1)
        i = cp.concatenate([p[1] for p in parts], 1)
        kk = min(k, d.shape[1])
        top = cp.argsort(d, axis=1)[:, :kk]
        ids = cp.where(cp.take_along_axis(d, top, 1) < cp.inf, cp.take_along_axis(i, top, 1), -1)
        self.res.sync()
        out = np.full((len(q), k), -1, np.int64)
        out[:, :kk] = cp.asnumpy(ids)
        return out


# ── CPU ───────────────────────────────────────────────────────────────────────


class Hnswlib:
    def __init__(self, metric, dim, capacity):
        import hnswlib

        self.index = hnswlib.Index("l2" if metric == "sqeuclidean" else "ip", dim)
        self.index.init_index(max_elements=capacity, M=16, ef_construction=200)
        self.index.set_num_threads(CPU_THREADS)

    def settings_for(self, k: int):
        return graph_ef(k)

    def insert(self, vecs, ids):
        for s in range(0, len(ids), 100000):
            self.index.add_items(np.ascontiguousarray(vecs[s : s + 100000]), ids[s : s + 100000])

    def delete(self, ids):
        for i in ids:
            self.index.mark_deleted(int(i))

    def search(self, q, ef, k: int):
        self.index.set_ef(max(ef, k))
        i, _ = self.index.knn_query(q, k=k, num_threads=CPU_THREADS)
        return i.astype(np.int64)


class Scann:
    """ScaNN: k-means tree partitions, 2-dim anisotropic quantization blocks
    (threshold 0.2 for inner product, as its authors use for MIPS), exact
    re-order of the best `pre_reorder` candidates. A setting is (leaves
    searched, candidates re-ordered)."""

    def __init__(self, metric, dim, capacity):
        self.metric = "dot_product" if metric == "inner_product" else "squared_l2"
        self.searcher = None
        self.leaves = None

    def settings_for(self, k: int):
        leaves = self.leaves or 1024
        reorder = (5 * k, 10 * k, 20 * k) if k <= 10 else (2 * k, 4 * k)
        return [(L, r) for L in (2, 4, 8, 16, 32, 64, 128, 256, 512) if L <= leaves for r in reorder]

    def insert(self, vecs, ids):
        import scann

        vecs = np.ascontiguousarray(vecs, np.float32)
        docids = [str(int(i)) for i in ids]
        if self.searcher is None:
            n = len(vecs)
            self.leaves = n_lists_for(n)
            threshold = 0.2 if self.metric == "dot_product" else float("nan")
            self.searcher = (
                scann.scann_ops_pybind.builder(vecs, 10, self.metric)
                .tree(num_leaves=self.leaves, num_leaves_to_search=max(1, self.leaves // 10), training_sample_size=min(n, 250000))
                .score_ah(2, anisotropic_quantization_threshold=threshold)
                .reorder(100)
                .set_n_training_threads(CPU_THREADS)
                .build(docids=docids)
            )
            self.searcher.set_num_threads(CPU_THREADS)
            return
        self.searcher.upsert(docids=docids, database=vecs, batch_size=max(1, len(docids)))

    def delete(self, ids):
        self.searcher.delete(docids=[str(int(i)) for i in ids])

    def search(self, q, setting, k: int):
        leaves, reorder = setting
        found, _ = self.searcher.search_batched_parallel(
            np.ascontiguousarray(q, np.float32),
            final_num_neighbors=k,
            leaves_to_search=leaves,
            pre_reorder_num_neighbors=max(reorder, k),
        )
        out = np.full((len(q), k), -1, np.int64)
        for j, row in enumerate(found):
            out[j, : len(row)] = [int(x) for x in row]
        return out


class Brute:
    """Exact top-k by GPU brute force over a device-resident set (common.Exact:
    a GEMM and exact selection): ann-benchmarks' brute-force entry, and the
    ceiling of each filtered approach. One insert, no writes. Natively
    filtered, it scores only the rows the predicate admits.

    Not cuvs brute_force: cuvs 26.10's sqeuclidean brute force returns -inf
    distances and wrong rows once a search asks for more than about 64
    neighbours, and every later search on the process is wrong too."""

    def __init__(self, metric, dim, capacity):
        device_pool()  # so its bytes are counted
        self.metric, self.tix = metric, None

    def insert(self, vecs, ids):
        from common import Exact

        assert np.array_equal(ids, np.arange(len(vecs))), "one insert of rows 0..n-1"
        self.ex = Exact(vecs, self.metric)

    def set_tags(self, indptr, tags):
        from prepare import TagIndex

        self.tix = TagIndex(indptr, tags, len(indptr) - 1)

    def settings_for(self, k):
        return [None]

    def search(self, q, s, k):
        return self.ex.search(q, k).astype(np.int64)

    def search_filtered(self, q, s, k, required):
        ex = self.ex
        cp = ex.cp
        out = np.full((len(q), k), -1, np.int64)
        dq = cp.asarray(q)
        for j, req in enumerate(required):
            rows = cp.asarray(self.tix.matching(req))
            if not len(rows):
                continue
            dots = ex.base[rows] @ dq[j]
            key = ex.norms[rows] - 2 * dots if ex.metric == "sqeuclidean" else -dots
            top = cp.argsort(key)[:k]
            out[j, : len(top)] = cp.asnumpy(rows[top])
        return out


def make(name: str, metric: str, dim: int, capacity: int):
    if name == "exact":
        return Brute(metric, dim, capacity)
    if name == "base":
        from base_engine import Base

        return Base(metric, dim, capacity)
    if name.startswith("seg-"):
        return Segmented(name[4:], metric, dim, capacity)
    if name == "hnswlib":
        return Hnswlib(metric, dim, capacity)
    if name == "scann":
        return Scann(metric, dim, capacity)
    if name == "qdrant":
        from qdrant import Qdrant

        return Qdrant(metric, dim, capacity)
    raise SystemExit(f"unknown engine {name}")
