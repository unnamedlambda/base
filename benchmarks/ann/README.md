# ANN benchmark

Nearest-neighbour search measured the way the field measures it:
[ann-benchmarks](https://ann-benchmarks.com) (recall against speed on its
standard datasets) and
[big-ann-benchmarks](https://big-ann-benchmarks.com) (its NeurIPS 2023 tracks:
filtered, out-of-distribution and streaming; its NeurIPS 2021 T3 track for
power). The headline everywhere is queries/s at recall@10 ≥ 0.9, big-ann's
threshold, with 0.95 and 0.99 beside it.

## Running

    ./run.sh                    # SIFT-1M and GloVe-100 curves, the streaming runbook, and one query a call
    ./run.sh sift-1m base       # one engine on one dataset: the development loop
    ./run.sh t2i-1m             # big-ann's out-of-distribution track, its own invocation
    ./run.sh yfcc-1m            # big-ann's filtered track, its own invocation
    ./run.sh bigann-10m         # BIGANN at 10M, its own invocation

One command does everything, from scratch:
1. creates the virtual environment if it is missing, builds base's ANN
   generator (`lean/algorithms/Ann`) and writes its artifacts (about 20 s,
   skipped where they are newer than the generator);
2. downloads any missing source and checks its pinned sha256;
3. prepares any missing dataset;
4. runs every harness and engine on the tiny datasets, and stops if any fails;
5. runs every harness and engine on each dataset;
6. writes `REPORT.md`.

A step killed by the memory cap, or whose engine runs out of the card's
memory, is reported as `no-fit` (the engine does not fit this machine: 9 GB of
host memory, a 12 GB card), not as a failure. Device memory comes from the
driver's pool (cudaMallocAsync); RMM's own pool fragments at 10M items.

Nothing measured is reused. Each invocation is a run in
`results/<time>-<git rev>/`, holding its result lines, every process's wall
time (`times.tsv`) and the idle power at its start (`idle.jsonl`).
Each run writes its own `REPORT.md`; the top-level `REPORT.md` takes each
dataset from the newest run that ran every engine (`report.py --latest`), so
`./run.sh sift-1m base` never replaces a full table. `report.py [run]` rebuilds
a report from any run. Each (dataset, engine) is
its own process under a 9 GB memory cap with swap off (safety, not a
benchmark condition: it leaves the 15.5 GB machine about 3 GB).

## Datasets

| name | from | items | dims | metric |
|---|---|---:|---:|---|
| sift-1m | ann-benchmarks | 1,000,000 | 128 | L2 |
| glove-100 | ann-benchmarks | 1,183,514 | 100 | angular |
| t2i-1m | big-ann OOD track (Text-to-Image), first 1M | 1,000,000 | 200 | inner product |
| yfcc-1m | big-ann filtered track (YFCC), first 1M, with tags | 1,000,000 | 192 | L2 |
| bigann-10m | big-ann BIGANN, first 10M | 10,000,000 | 128 | L2 |
| tiny, yfcc-tiny | first 20,000 SIFT / YFCC items | 20,000 | 128, 192 | L2 |

big-ann's official sizes are 10M for the 2023 tracks (30M for streaming) and
1B for 2021's. The 1M subsets keep each invocation within tens of minutes on
this machine. Ground truth is recomputed exactly (`common.Exact`, checked
against float64) with distances, so recall is tie-aware.

**recall@10**, as both benchmarks count it: a returned item is a hit when it
is as near as the 10th true neighbour, so ties at that distance count
(`common.recall`).

## Harnesses

- **`curves.py`** (ann-benchmarks): every item in one insert (build seconds,
  device and host bytes per item), then the engine's settings cheapest first,
  each scored by recall@10 and timed at batches of 1,000 queries. The fastest
  setting reaching 0.9 runs 2 s more for its energy per query. Run on SIFT,
  GloVe, T2I (out of distribution) and BIGANN-10M.
- **`stream.py`** (big-ann streaming): a runbook slides a window of 16 k-means
  clusters across SIFT-1M, inserting the next cluster and deleting the oldest
  at each step; every 4 steps all queries are searched with one fixed setting
  and scored against the exact top 10 over the live set. Reports mean and
  minimum recall over the searches and the runbook's time, with insert,
  delete and search rates apart, and the time spent waiting for an engine to
  finish indexing before each search (`wait_indexed`): an engine may index as
  it inserts or defer the work to that wait, and the runbook's time holds
  both.
- **`latency.py`** (ann-benchmarks' default mode): every item in one insert,
  then the engine's settings cheapest first, 1,000 queries each searched
  alone, one call a query. Queries/s is one over the mean call, with the
  median and 99th-percentile call. Run on SIFT-1M.
- **`filtered.py`** (big-ann filtered): the 10 nearest items holding every tag
  a query requires (1 or 2 of 200,386), checked exactly, each engine given
  the items' tags and filtering itself, as big-ann's track ranks it. A sweep
  stops below 1,000 queries/s, probing one batch before each full pass.

**Energy** is the GPU's cumulative counter (NVML, which advances every
100 ms), per query, total and dynamic. CPU package energy (RAPL) is root-only
on this machine, so a CPU engine's energy is not comparable to a GPU
engine's, and the report says so.

## Engines

base, and the leaders it is measured against: the best of each family in
the exploration this suite began with (CAGRA and IVF-PQ in a segment design,
hnswlib, ScaNN, Qdrant; their runs stay in `results/`). Every engine runs
here, with its library's documented default build parameters; no published
figure is compared.

| engine | runs on | what it is |
|---|---|---|
| `base` | curves, stream, latency, filtered | base's engine: `lean/algorithms/Ann` (kernels and the CLIF host), run through py-base. IVF with int8 residual codes scanned by one fused `dp4a` kernel a (list, query slot), re-ranked exactly against fp32 rows; for a search of at most 8 queries, each (query, list) pair cut in 4 pieces, each a warp's alone, with no sort of the pairs; an exact write buffer for rows since its build, a live bitset, the index kept up (rebuilt when its churn reaches an eighth of its rows) at the next commit or search rather than inside an insert or delete; a native tag filter (exact over small admitted sets, a per-query bitset in the scan otherwise). No atomics: sorts, selections and histograms are ballots, shuffles and a radix sort |
| `seg-cagra` | curves, latency | cuvs 26.10 CAGRA inside the segment design of production GPU systems (Milvus): an exact write buffer, sealed immutable segments, one delete bitset, rebuilds and merges: the GPU leader |
| `scann` | curves, stream, latency | ScaNN 1.4: k-means partitions, anisotropic quantization, exact re-order; native upsert and delete: the CPU leader, and the fastest streaming runbook |
| `hnswlib` | stream, latency | HNSW on all 12 threads, native add and delete: the best streaming recall, and the classic one-query-a-call leader |
| `qdrant` | filtered | Qdrant 1.20 server (Rust), HNSW with its payload filter: the only other engine here that filters natively |
| `exact` | filtered | GPU brute force over only the admitted items (`common.Exact`: a GEMM and exact selection), the ceiling |

`exact` is not cuvs `brute_force`: in cuvs 26.10 its sqeuclidean search
returns wrong rows once asked for more than about 64 neighbours, and keeps
doing so on that process.

`roofline.py` gives this card's measured peaks (copy bandwidth, `dp4a` and
fp32 FMA throughput).

## Not built

- The fraction of the peaks base's search reaches: its kernels run inside the
  artifact, so the work it does per setting is not yet counted from outside.
- A modern text-embedding dataset (768-1,536 dimensions).
