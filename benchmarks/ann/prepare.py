"""Turn the downloaded sources into the arrays the ANN benchmark reads.

Each dataset `<name>` becomes `data/ann/<name>/`:

    base.npy       float32 (n, d)  the items
    queries.npy    float32 (q, d)
    metric.txt     "sqeuclidean" or "inner_product"
    gt.npy         int32 (q, 10)   the exact top 10 of every query
    gt_dist.npy    float32 (q, 10) their distances (common.distances), for
                                   tie-aware recall

Datasets, each a standard benchmark's:

    sift-1m       ann-benchmarks' SIFT (sift-128-euclidean.hdf5): 1M items,
                  128 dims, L2, 10,000 queries.
    glove-100     ann-benchmarks' GloVe (glove-100-angular.hdf5): 1.18M word
                  vectors, 100 dims, angular (normalised, then inner product).
    t2i-1m        big-ann-benchmarks' out-of-distribution track (Text-to-Image):
                  the first 1M image embeddings and 10,000 text queries, 200
                  dims, inner product.
    yfcc-1m       big-ann-benchmarks' filtered track: the first 1M YFCC items
                  (CLIP image embeddings, 192 dims, uint8, L2), each with its
                  tags (about 11 of 200,386), and 10,000 queries each
                  requiring 1 or 2 tags. Also, for filtered.py:
                      tags_indptr.npy, tags.npy        item tags (CSR)
                      query_tags_indptr.npy, query_tags.npy
                      gt_filtered.npy, gt_filtered_dist.npy
                                       the exact top 10 among the items a
                                       query's predicate admits, -1 padded
    bigann-10m    big-ann-benchmarks' BIGANN: the first 10M items (SIFT, 128
                  dims, L2) and its 10,000 public queries.
    tiny, yfcc-tiny
                  the first 20,000 SIFT / YFCC items and 1,000 queries, for
                  checking the harnesses (not reported).

sift-1m and tiny also hold a streaming runbook (stream.py), after
big-ann-benchmarks' clustered runbooks:

    order.npy          int32 (n,)   the items grouped into CLUSTERS k-means
                                    clusters, clusters in a shuffled order
    bounds.npy         int64 (CLUSTERS + 1,) where each cluster starts in it
    stream_gt.npy, stream_gt_dist.npy
                       (searches, q, 10) the exact top 10 over the live set at
                       each search step of `runbook`

Cosine data is normalised, so inner product ranks exactly as cosine. Ground
truth is `common.Exact`, spot-checked against float64 (printed).

Sources are downloaded when missing and checked against pinned sha256 sums;
what is already prepared is left as it is.

    python prepare.py [names...]       # default: every dataset
"""

import hashlib
import os
import sys
import tarfile
import urllib.request

import numpy as np

from common import DATA, Exact, distances, drop_cache, float64_check

SEED = 0
CLUSTERS = 64

T2I = "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/"
# Every downloaded file: path under data/ann -> (url, sha256, byte range or None).
FILES = {
    "sift-128-euclidean.hdf5": (
        "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984",
        None,
    ),
    "qdrant-bin/q.tgz": (
        "https://github.com/qdrant/qdrant/releases/download/v1.20.0/qdrant-x86_64-unknown-linux-gnu.tar.gz",
        "2361f3df0dedfef31f5444f9f15bd15d6b4a6cba6ad952daecc5fbe3867c5ee6",
        None,
    ),
    "glove-100-angular.hdf5": (
        "http://ann-benchmarks.com/glove-100-angular.hdf5",
        "544af1d5e84e112cd4749571dcfd8ca109818a572f850af75a3a09e093a953c4",
        None,
    ),
    # The first 1M rows of the 1B file: its 8-byte header and 1M x 200 float32.
    "t2i-src/base.1M.fbin": (
        T2I + "base.1B.fbin",
        "b28c78d5fe501b6101475b362b1ef514152fa2d57c154edfb674a3120c4fdb4d",
        (0, 8 + 1_000_000 * 200 * 4 - 1),
    ),
    "t2i-src/query.public.100K.fbin": (
        T2I + "query.public.100K.fbin",
        "4b9aabc321ca42a674d5854d1f95347deb902c489c7b49b34e38960f60dad784",
        None,
    ),
}
YFCC = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/"
FILES.update(
    {
        # The first 1M items of the 10M: the 8-byte header and 1M x 192 uint8.
        "yfcc-src/base.1M.u8bin": (
            YFCC + "base.10M.u8bin",
            "acdf3537d5b2325f6990d075cf433fd973d65138b59cbae2e530b470b3ae24d7",
            (0, 8 + 1_000_000 * 192 - 1),
        ),
        "yfcc-src/base.metadata.10M.spmat": (
            YFCC + "base.metadata.10M.spmat",
            "2f9c9a533fde31cb062edfdf5c9410f088bb5d7bf4a721ab98c47e3d2954287d",
            None,
        ),
        "yfcc-src/query.public.100K.u8bin": (
            YFCC + "query.public.100K.u8bin",
            "77a937d67baf7523a7d9089a6e49c0ac8c6a3e7eb5499d759afeacf0a843df82",
            None,
        ),
        "yfcc-src/query.metadata.public.100K.spmat": (
            YFCC + "query.metadata.public.100K.spmat",
            "93eae99029c34d167f7d030c60c6b4075c039247b3204cc6c34f8a44066983bd",
            None,
        ),
    }
)
BIGANN = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/"
FILES.update(
    {
        # The first 10M items of the 1B: the 8-byte header and 10M x 128 uint8.
        "bigann-src/base.10M.u8bin": (
            BIGANN + "base.1B.u8bin",
            "24f9c72cb7cf4cb388a2e6e2eeb939f671739be16e84edc9b3b52813063c918f",
            (0, 8 + 10_000_000 * 128 - 1),
        ),
        "bigann-src/query.public.10K.u8bin": (
            BIGANN + "query.public.10K.u8bin",
            "eca755831fc9a8004e14886df48b81109a8ed3bfc6b632509c93c0460a30d552",
            None,
        ),
    }
)
LOADED = ("base.npy", "queries.npy", "metric.txt")
GT = ("gt.npy", "gt_dist.npy")
STREAM = ("order.npy", "bounds.npy", "stream_gt.npy", "stream_gt_dist.npy")
FILTERED = ("tags_indptr.npy", "tags.npy", "query_tags_indptr.npy", "query_tags.npy", "gt_filtered.npy", "gt_filtered_dist.npy")
WINDOW = 16  # clusters live at once in the streaming runbook
SEARCH_EVERY = 4  # clusters between its searches
YFCC_ITEMS, YFCC_QUERIES = 1_000_000, 10_000


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def fetch(rel: str) -> str:
    """data/ann/<rel>, downloaded if missing and checked against its sha256."""
    url, want, span = FILES[rel]
    path = os.path.join(DATA, rel)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading {url}", flush=True)
        req = urllib.request.Request(url)
        if span:
            req.add_header("Range", f"bytes={span[0]}-{span[1]}")
        with urllib.request.urlopen(req) as r, open(path + ".part", "wb") as f:
            while block := r.read(1 << 24):
                f.write(block)
        os.replace(path + ".part", path)
    got = sha256(path)
    if got != want:
        raise SystemExit(f"{rel}: sha256 {got}, expected {want}")
    return path


def fetch_qdrant() -> None:
    fetch("qdrant-bin/q.tgz")
    binary = os.path.join(DATA, "qdrant-bin", "qdrant")
    if not os.path.exists(binary):
        with tarfile.open(os.path.join(DATA, "qdrant-bin", "q.tgz")) as t:
            t.extract("qdrant", os.path.join(DATA, "qdrant-bin"))


def normalise(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1
    return (x / n).astype(np.float32)


def load_hdf5(out: str, file: str, rows=None, nq=None):
    """An ann-benchmarks file: its train set as the items, its test set as
    the queries; angular data normalised for inner product."""
    import h5py

    with h5py.File(fetch(file), "r") as f:
        base = np.asarray(f["train"][:rows], np.float32)
        queries = np.asarray(f["test"][:nq], np.float32)
        angular = f.attrs["distance"] == "angular"
    if angular:
        base, queries = normalise(base), normalise(queries)
    np.save(os.path.join(out, "base.npy"), base)
    return queries, "inner_product" if angular else "sqeuclidean"


def read_fbin(path: str, rows: int) -> np.ndarray:
    n, d = np.fromfile(path, np.int32, 2)
    rows = min(rows, int(n)) if n > 0 else rows
    return np.fromfile(path, np.float32, rows * int(d), offset=8).reshape(rows, int(d))


def load_t2i(out: str):
    base = read_fbin(fetch("t2i-src/base.1M.fbin"), 1_000_000)
    np.save(os.path.join(out, "base.npy"), base)
    del base
    queries = read_fbin(fetch("t2i-src/query.public.100K.fbin"), 10000)
    return np.ascontiguousarray(queries), "inner_product"


def clusters(exact: Exact, n: int):
    """(order, bounds): all rows grouped by nearest of CLUSTERS k-means centres
    (Lloyd, 10 iterations on a 100k sample), clusters in a fixed shuffled
    order; cluster c is order[bounds[c] : bounds[c + 1]]."""
    import cupy as cp

    rng = np.random.default_rng(SEED + 2)
    sample = exact.base[cp.asarray(np.sort(rng.choice(n, min(100000, n), replace=False)))]
    cent = sample[cp.asarray(rng.choice(len(sample), CLUSTERS, replace=False))]
    for _ in range(10):
        a = cp.argmin((cent * cent).sum(1)[None, :] - 2 * sample @ cent.T, axis=1)
        for c in range(CLUSTERS):
            m = a == c
            if int(m.sum()):
                cent[c] = sample[m].mean(0)
    label = np.empty(n, np.int32)
    for s in range(0, n, 100000):
        b = exact.base[s : s + 100000]
        label[s : s + 100000] = cp.asnumpy(cp.argmin((cent * cent).sum(1)[None, :] - 2 * b @ cent.T, axis=1))
    groups = [np.nonzero(label == c)[0] for c in rng.permutation(CLUSTERS)]
    bounds = np.cumsum([0] + [len(g) for g in groups])
    return np.concatenate(groups).astype(np.int32), bounds


def runbook(clusters_n: int = CLUSTERS):
    """The streaming runbook, as (op, cluster) steps: insert the first WINDOW
    clusters and search; then each further cluster is inserted, the oldest
    live one deleted, and every SEARCH_EVERY clusters a search runs. The live
    set slides across the clusters, so the distribution moves under the index
    and deleted space must be reused (big-ann-benchmarks' clustered and
    sliding-window runbooks)."""
    steps = [("insert", c) for c in range(WINDOW)] + [("search", None)]
    for c in range(WINDOW, clusters_n):
        steps += [("insert", c), ("delete", c - WINDOW)]
        if (c - WINDOW + 1) % SEARCH_EVERY == 0:
            steps.append(("search", None))
    return steps


def read_spmat(path: str, rows: int):
    """(indptr, indices) of the first `rows` rows of a big-ann CSR matrix:
    int64 nrow, ncol, nnz; int64 indptr[nrow + 1]; int32 indices[nnz];
    float32 data[nnz] (unused: every entry is 1)."""
    nrow, _, _ = np.fromfile(path, np.int64, 3)
    rows = min(rows, int(nrow))
    indptr = np.fromfile(path, np.int64, rows + 1, offset=24)
    indices = np.fromfile(path, np.int32, int(indptr[-1]), offset=24 + 8 * (int(nrow) + 1))
    return indptr, indices


def load_yfcc(out: str, items: int = YFCC_ITEMS, nq: int = YFCC_QUERIES):
    """The first `items` YFCC items (CLIP image embeddings, uint8, L2) with
    their tags, and the first `nq` queries with the 1 or 2 tags each requires."""
    base = np.fromfile(fetch("yfcc-src/base.1M.u8bin"), np.uint8, items * 192, offset=8).reshape(items, 192)
    np.save(os.path.join(out, "base.npy"), base.astype(np.float32))
    indptr, tags = read_spmat(fetch("yfcc-src/base.metadata.10M.spmat"), items)
    np.save(os.path.join(out, "tags_indptr.npy"), indptr)
    np.save(os.path.join(out, "tags.npy"), tags)
    queries = np.fromfile(fetch("yfcc-src/query.public.100K.u8bin"), np.uint8, nq * 192, offset=8)
    qptr, qtags = read_spmat(fetch("yfcc-src/query.metadata.public.100K.spmat"), nq)
    np.save(os.path.join(out, "query_tags_indptr.npy"), qptr)
    np.save(os.path.join(out, "query_tags.npy"), qtags)
    return queries.reshape(nq, 192).astype(np.float32), "sqeuclidean"


class TagIndex:
    """Items per tag (an inverted index), and the exact predicate check."""

    def __init__(self, indptr, tags, n):
        item = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
        order = np.argsort(tags, kind="stable")
        self.items = item[order]  # sorted within each tag
        self.start = np.searchsorted(tags[order], np.arange(int(tags.max()) + 2))

    def items_of(self, tag: int) -> np.ndarray:
        if tag + 1 >= len(self.start):
            return np.zeros(0, np.int64)
        return self.items[self.start[tag] : self.start[tag + 1]]

    def matching(self, required) -> np.ndarray:
        """Items holding every required tag, sorted."""
        out = self.items_of(int(required[0]))
        for t in required[1:]:
            out = np.intersect1d(out, self.items_of(int(t)), assume_unique=True)
        return out

    def keep(self, cand: np.ndarray, required) -> np.ndarray:
        """Bool mask over candidate ids (-1 allowed): which satisfy the predicate."""
        ok = cand >= 0
        for t in required:
            lst = self.items_of(int(t))
            if len(lst) == 0:  # no item holds the tag: nothing satisfies the predicate
                return np.zeros(len(cand), bool)
            i = np.searchsorted(lst, cand)
            ok &= (i < len(lst)) & (lst[np.minimum(i, len(lst) - 1)] == cand)
        return ok


def prepare_filtered(out: str) -> None:
    """Per query: the exact top 10 among the items its predicate admits (-1
    where fewer than 10 do), and their distances."""
    import cupy as cp

    base = np.load(os.path.join(out, "base.npy"), mmap_mode="r")
    n = len(base)
    queries = np.load(os.path.join(out, "queries.npy"))
    tix = TagIndex(np.load(os.path.join(out, "tags_indptr.npy")), np.load(os.path.join(out, "tags.npy")), n)
    qptr, qtags = np.load(os.path.join(out, "query_tags_indptr.npy")), np.load(os.path.join(out, "query_tags.npy"))
    x = cp.asarray(np.asarray(base))
    xn = (x * x).sum(1)
    gt = np.full((len(queries), 10), -1, np.int32)
    for i, q in enumerate(queries):
        rows = tix.matching(qtags[qptr[i] : qptr[i + 1]])
        if len(rows) == 0:
            continue
        r = cp.asarray(rows)
        key = xn[r] - 2 * x[r] @ cp.asarray(q)
        k = min(10, len(rows))
        top = cp.argsort(key)[:k]
        gt[i, :k] = cp.asnumpy(r[top])
    np.save(os.path.join(out, "gt_filtered.npy"), gt)
    np.save(os.path.join(out, "gt_filtered_dist.npy"), distances(base, queries, gt, "sqeuclidean"))
    print(f"{os.path.basename(out)}: filtered ground truth; {(gt[:, 0] < 0).mean():.1%} of queries admit no item")


def load_bigann(out: str):
    """The first 10M BIGANN (SIFT) items and its 10,000 public queries.
    Widened to float32 in slices through a memory
    map (5.1 GB), never held whole."""
    src = np.memmap(fetch("bigann-src/base.10M.u8bin"), np.uint8, "r", 8, (10_000_000, 128))
    base = np.lib.format.open_memmap(os.path.join(out, "base.npy"), "w+", np.float32, src.shape)
    for s in range(0, len(src), 500000):
        base[s : s + 500000] = src[s : s + 500000]
        base.flush()
        drop_cache(base)
    del base
    queries = np.fromfile(fetch("bigann-src/query.public.10K.u8bin"), np.uint8, 10000 * 128, offset=8)
    return queries.reshape(10000, 128).astype(np.float32), "sqeuclidean"


SOURCES = {
    "tiny": lambda out: load_hdf5(out, "sift-128-euclidean.hdf5", 20000, 1000),
    "sift-1m": lambda out: load_hdf5(out, "sift-128-euclidean.hdf5"),
    "glove-100": lambda out: load_hdf5(out, "glove-100-angular.hdf5"),
    "t2i-1m": load_t2i,
    "yfcc-tiny": lambda out: load_yfcc(out, 20000, 1000),
    "yfcc-1m": load_yfcc,
    "bigann-10m": load_bigann,
}
STREAM_SETS = ("tiny", "sift-1m")
FILTERED_SETS = ("yfcc-tiny", "yfcc-1m")


def have(out, files):
    return all(os.path.exists(os.path.join(out, f)) for f in files)


def prepare(name: str) -> None:
    out = os.path.join(DATA, name)
    os.makedirs(out, exist_ok=True)
    if not have(out, LOADED):
        queries, metric = SOURCES[name](out)
        np.save(os.path.join(out, "queries.npy"), queries)
        with open(os.path.join(out, "metric.txt"), "w") as f:
            f.write(metric)
    base = np.load(os.path.join(out, "base.npy"), mmap_mode="r")
    queries = np.load(os.path.join(out, "queries.npy"))
    metric = open(os.path.join(out, "metric.txt")).read().strip()
    n = len(base)
    exact = None
    if not have(out, GT):
        exact = Exact(base, metric)
        gt = exact.search(queries, 10)
        agree = float64_check(base, queries, gt, metric, n=10)
        np.save(os.path.join(out, "gt.npy"), gt)
        np.save(os.path.join(out, "gt_dist.npy"), distances(base, queries, gt, metric))
        print(f"{name}: base {base.shape}, {len(queries)} queries, metric {metric}; gt float64 agreement {agree:.3f}")
    if name in STREAM_SETS and not have(out, STREAM):
        exact = exact or Exact(base, metric)
        order, bounds = clusters(exact, n)
        live = np.zeros(n, bool)
        gts = []
        for op, c in runbook():
            if op == "search":
                gts.append(exact.search(queries, 10, active=live))
            else:
                live[order[bounds[c] : bounds[c + 1]]] = op == "insert"
        gts = np.stack(gts)
        np.save(os.path.join(out, "order.npy"), order)
        np.save(os.path.join(out, "bounds.npy"), bounds)
        np.save(os.path.join(out, "stream_gt.npy"), gts)
        np.save(os.path.join(out, "stream_gt_dist.npy"), np.stack([distances(base, queries, g, metric) for g in gts]))
        print(f"{name}: streaming runbook, {len(gts)} searches")
    del exact
    if name in FILTERED_SETS and not have(out, FILTERED):
        prepare_filtered(out)


def main() -> None:
    fetch_qdrant()
    for name in sys.argv[1:] or list(SOURCES):
        prepare(name)


if __name__ == "__main__":
    main()
