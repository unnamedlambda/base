"""The artifact's input and output layout, in one place.

Three drivers used to carry their own copy of these offsets, and moving one of
them in Lean without moving all three copies is a buffer the runtime reads past
-- which is a crash, not a wrong answer. It happened once. So they live here
and the drivers import them.

Kept in step with `GptOssDecode` by hand; `check_layout` compares against the
input size the artifact itself declares, which is the one number that would
catch a drift.
"""

import struct

TEXT_MAX = 8192
TMPL_MAX = 256
NL, HH, VOCAB = 24, 2880, 201088

D_TOK, D_POS, D_MODE, D_TLEN = 0, 4, 8, 12
D_PEXP, D_PDEN, D_PEMB, D_PTOK = 16, 272, 528, 784
D_STOP, D_MAXNEW = 1040, 1044
D_NPRE, D_NPOST = 1048, 1052
D_PRE = 1056
D_POST = D_PRE + 4 * TMPL_MAX
D_TEXT = D_POST + 4 * TMPL_MAX
D_IN_BYTES = D_TEXT + TEXT_MAX

D_OUT_LOGITS = 8
D_OUT_TRACE = D_OUT_LOGITS + VOCAB * 4
D_OUT_TEXT = D_OUT_TRACE + 2 * NL * HH * 4
D_OUT_NGEN = D_OUT_TEXT + TEXT_MAX
D_OUT_GEN = D_OUT_NGEN + 8
D_OUT_BYTES = D_OUT_GEN + 4 * TEXT_MAX

# where the artifact records how many input bytes it expects
HOST_LEN_OFF = 0x80


def check_layout(artifact_json):
    """Fail loudly if these offsets have drifted from the artifact's own."""
    want = struct.unpack_from(
        "<I", bytes(artifact_json["setup"]["initial_memory"]), HOST_LEN_OFF)[0]
    assert want == D_IN_BYTES, (
        f"layout drift: the artifact expects {want} input bytes, "
        f"applications/gpt-oss/layout.py says {D_IN_BYTES}")
