"""What the engine costs, measured rather than modelled.

Three numbers decide whether offloading a 10 GiB expert pool over PCIe is worth
doing at all, and this reports all three:

  * **decode rate**, tokens a second once the cache is warm;
  * **miss rate**, the fraction of expert lookups that had to cross the bus --
    the whole design rests on this being small, and it is a property of the
    router and the slot count, not of anything that can be tuned here;
  * **bus throughput** implied by those misses, which is the check that a miss
    costs what a miss should cost and not more.

Run it alone. The pool is 9.48 GiB of pinned host memory, which cannot be
swapped or reclaimed, so anything else large in this process takes the machine
down rather than failing.

  python applications/gpt-oss/bench.py <artifact.json> --bank data/gptoss-bank
"""

import argparse
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layout import (CAP_DEFAULT, CAP_MAX, D_CTX, D_IN_BYTES, D_OUT_BYTES,
                    check_layout, acquire_engine_lock)
import entries

ROW_BYTES = 13253760          # one expert, all six pieces
NL, TOPK = 24, 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--context", type=int, default=CAP_DEFAULT,
                    help=f"positions the full-attention layers keep keys "
                         f"for (max {CAP_MAX}); every one of them is an "
                         f"expert slot not held resident")
    args = ap.parse_args()

    # before anything allocates: two of these pin 9.48 GiB each
    acquire_engine_lock('bench.py')

    import py_base

    import json
    check_layout(json.load(open(args.artifact)))
    art = py_base.load_artifact(args.artifact)
    art_entries = entries.entries(os.path.basename(args.artifact).removesuffix(".json"))
    assert not art_entries
    base = py_base.Base(art)

    paths = {}
    for off, name in ((16, "experts.bin"), (272, "dense.bin"),
                      (528, "embed.bin"), (784, "tokenizer.bin")):
        p = os.path.abspath(os.path.join(args.bank, name)).encode() + b"\0"
        assert len(p) < 256
        paths[off] = p

    out = bytearray(D_OUT_BYTES)

    def step(tok, pos):
        buf = bytearray(D_IN_BYTES)
        struct.pack_into("<III", buf, 0, tok, pos, 0)     # mode 0
        struct.pack_into("<I", buf, D_CTX, args.context)
        for off, p in paths.items():
            buf[off:off + len(p)] = p
        base.execute_into(art_entries["main"], bytes(buf), out)
        return struct.unpack_from("<Ii", bytes(out), 0)

    rng = np.random.default_rng(args.seed)
    ids = [int(t) for t in rng.integers(1000, 50000, args.prompt_len)]

    print("  loading: 12.9 GiB read, 9.48 GiB pinned")
    t0 = time.time()
    nxt, miss0 = step(ids[0], 0)
    load = time.time() - t0
    print(f"  cold start + first token   {load:6.1f} s")

    t0, m0 = time.time(), miss0
    for pos in range(1, len(ids)):
        nxt, miss = step(ids[pos], pos)
    pre = time.time() - t0
    n_pre = len(ids) - 1
    pre_miss = miss - m0
    print(f"  prompt, {n_pre} tokens      {pre:6.2f} s   {n_pre/pre:7.1f} tok/s"
          f"   misses {pre_miss:5d}  ({pre_miss/(n_pre*NL*TOPK)*100:5.1f}%)")

    pos = len(ids)
    t0, m0 = time.time(), miss
    for _ in range(args.gen):
        nxt, miss = step(nxt, pos)
        pos += 1
    gen = time.time() - t0
    gen_miss = miss - m0
    lookups = args.gen * NL * TOPK
    rate = args.gen / gen
    print(f"  generate, {args.gen} tokens    {gen:6.2f} s   {rate:7.1f} tok/s"
          f"   misses {gen_miss:5d}  ({gen_miss/lookups*100:5.1f}%)")
    print()
    # A miss is six copies totalling one expert row.  This is the throughput
    # the misses imply, not a bus measurement: it is the number to compare
    # against the socket's 13.4 GB/s to see whether a miss costs what it should.
    moved = gen_miss * ROW_BYTES
    print(f"  expert traffic             {moved/2**30:6.2f} GiB in {gen:.2f} s"
          f"   {moved/gen/1e9:6.2f} GB/s implied")
    print(f"  per token                  {moved/args.gen/2**20:6.1f} MiB,"
          f" {gen/args.gen*1000:6.1f} ms")
    print()
    resident = 1.0 - gen_miss / lookups
    print(f"  residency                  {resident*100:5.1f}%  of lookups served"
          f" from the device")
    print()
    print("RESULT : OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
