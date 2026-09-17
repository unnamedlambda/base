"""The artifact's input and output layout, in one place.

Three drivers used to carry their own copy of these offsets, and moving one of
them in Lean without moving all three copies is a buffer the runtime reads past
-- which is a crash, not a wrong answer. It happened once. So they live here
and the drivers import them.

Kept in step with `GptOssDecode` by hand; `check_layout` compares against the
input size the artifact itself declares, which is the one number that would
catch a drift.
"""

import os
import struct

TEXT_MAX = 32768
TMPL_MAX = 8192
NL, HH, VOCAB = 24, 2880, 201088

# How many positions the full-attention layers keep keys for, and so how long a
# conversation can be. The sliding layers ring at 128 by design and do not bound
# anything; this does. No longer a property of the artifact -- the caller picks
# it at start-up and the caches are allocated to it -- so a driver that offers
# the choice must pass the same value to `check_turn_fits`.
CAP_DEFAULT = 8192
# `GptOssAttention.CAP_MAX`: the published rotation tables have this many rows,
# so no position beyond it can be encoded at all.
CAP_MAX = 131072

# PTX slots, in `GptOssDecode.dPtx` order. Each kernel is its own module with a
# single entry called `main`, so a slot index is the only way to name one, and
# these are the two a decode step spends most of its time in.
S_GATEUP, S_DOWN = 8, 9

D_TOK, D_POS, D_MODE, D_TLEN = 0, 4, 8, 12
D_PEXP, D_PDEN, D_PEMB, D_PTOK = 16, 272, 528, 784
D_STOP, D_MAXNEW = 1040, 1044
D_NPRE, D_NPOST = 1048, 1052
D_INVT, D_SEED = 1056, 1060
D_LNMINP = 1064
D_STARTPOS = 1068
# Positions to give the full-attention layers. Read once, on the first call:
# the caches are allocated to it. 0 asks for CAP_DEFAULT.
D_CTX = 1072
D_PRE = 1076
D_POST = D_PRE + 4 * TMPL_MAX
D_TEXT = D_POST + 4 * TMPL_MAX
D_IN_BYTES = D_TEXT + TEXT_MAX

D_OUT_LOGITS = 8
D_OUT_TRACE = D_OUT_LOGITS + VOCAB * 4
D_OUT_TEXT = D_OUT_TRACE + 2 * NL * HH * 4
D_OUT_NGEN = D_OUT_TEXT + TEXT_MAX
D_OUT_GEN = D_OUT_NGEN + 8
D_OUT_NTEXT = D_OUT_GEN + 4 * TEXT_MAX
D_OUT_TEXTTOK = D_OUT_NTEXT + 8
D_OUT_BYTES = D_OUT_TEXTTOK + 4 * TEXT_MAX

# where the artifact records how many input bytes it expects
HOST_LEN_OFF = 0x80


def ptx_modules(base):
    """Every PTX module the artifact carries, in slot order.

    They are laid out in the initial memory image at a fixed stride, each a
    NUL-terminated string. Pulling them out is how a kernel gets benchmarked or
    disassembled without running the model: the emitted text is the artifact's,
    not a copy that could drift from it.
    """
    import re
    mem = base.read_memory(0, base.memory_size())
    out = []
    for m in re.finditer(rb"\.version", mem):
        seg = mem[m.start():]
        out.append(seg[:seg.find(b"\x00")].decode())
    return out


def check_layout(base):
    """Fail loudly if these offsets have drifted from the artifact's own.

    `base` is the runtime the artifact was loaded into, before its first call:
    what it holds then is the image the artifact starts from."""
    want = struct.unpack_from("<I", base.read_memory(HOST_LEN_OFF, 4))[0]
    assert want == D_IN_BYTES, (
        f"layout drift: the artifact expects {want} input bytes, "
        f"applications/gpt-oss/layout.py says {D_IN_BYTES}")


def ln_min_p(min_p):
    """`ln(min_p)` for the sampler, or -inf when the filter is off.

    min-p keeps a token when its probability is at least `min_p` times the most
    likely token's, and in tempered space that is a flat threshold below the
    maximum -- which is why it costs one extra sweep and no sort. Temperature
    1.0 over a 201088-token vocabulary draws from the tail often enough to
    matter without it.
    """
    import math
    return float("-inf") if min_p <= 0 else math.log(min_p)


# ---- one engine at a time --------------------------------------------------

LOCK_PATH = "/tmp/gpt-oss-engine.lock"
_lock_handle = None


def acquire_engine_lock(who="this program"):
    """Refuse to start if another driver already holds the pool.

    The engine pins 9.48 GiB of host memory. Pinned pages cannot be swapped or
    reclaimed, so two of these on a 16 GiB machine do not fail -- the kernel
    runs out of anything to page against and the box stops responding. It is
    the worst failure mode available to this program, it has happened twice,
    and it is not something a warning in a docstring prevents.

    So the lock is taken before any allocation and released when the process
    dies, however it dies. A second driver exits with a message naming the
    first one's pid instead of taking the machine with it.
    """
    import fcntl
    global _lock_handle
    f = open(LOCK_PATH, "a+")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.seek(0)
        rec = f.read().strip().split(None, 1)
        f.close()
        pid = rec[0] if rec else "?"
        name = rec[1] if len(rec) > 1 else "a driver"
        raise SystemExit(
            f"\n  {name} is already running (pid {pid}).\n"
            f"  Two of these pin 9.48 GiB each and this machine has 16 GiB, so "
            f"the second\n  would lock it up rather than fail. Wait for that "
            f"one, or:\n      kill {pid}\n")
    f.seek(0)
    f.truncate()
    f.write(f"{os.getpid()} {who}")
    f.flush()
    # held for the life of the process: the reference keeps the fd open, and
    # the kernel drops the lock when the process exits by any route
    _lock_handle = f
    return f
