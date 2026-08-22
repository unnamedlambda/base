"""The pre-tokenizer, in the restricted form CLIF will run it.

A BPE tokenizer does not merge across a chunk boundary, and where those
boundaries fall is decided by a regex the checkpoint ships.  Skipping it is not
a rounding error: measured against the reference tokenizer, whole-text BPE
agrees on 3 of 6 ordinary strings for both Qwen2 and gpt-oss -- it diverges on
exactly what the pattern exists to handle, which is code indentation, runs of
digits, and repeated whitespace.

CLIF has no regex, and it should not grow one.  What it needs instead is the
small fragment these patterns actually inhabit:

  * an **ordered alternation** -- leftmost, first-listed wins, no backtracking
    between alternatives once one has matched a non-empty prefix;
  * each alternative a **sequence of items**, each item a character class with
    a repeat count (`?`, `*`, `+`, or `{1,3}`);
  * two special items: a fixed set of contraction literals (`'s`, `'ll`, ...)
    and the single lookahead `(?!\\S)`.

That is a table, and this module is both the thing that builds it and the
reference interpreter for it -- written the way the CLIF will be, so that
agreement here is evidence about the CLIF and not about Python's `regex`.

Character classes are a flat table, one byte of membership bits per code
point.  1.1 MB, which is nothing beside the merge list, and it turns "is this
character a lowercase letter" into a load.
"""

import unicodedata

# ---- classes, as bits in one byte per code point ---------------------------

C_L = 1 << 0      # \p{L}       any letter
C_N = 1 << 1      # \p{N}       any number
C_UP = 1 << 2     # \p{Lu Lt Lm Lo M}
C_LO = 1 << 3     # \p{Ll Lm Lo M}
C_S = 1 << 4      # \s          whitespace
C_NL = 1 << 5     # \r \n
MAXCP = 0x110000


def class_table():
    """One byte per code point.  Built from the Unicode database, so what it
    says is what `\\p{...}` says rather than an ASCII approximation."""
    t = bytearray(MAXCP)
    for cp in range(MAXCP):
        ch = chr(cp)
        cat = unicodedata.category(ch)
        b = 0
        if cat[0] == "L":
            b |= C_L
        if cat[0] == "N":
            b |= C_N
        if cat in ("Lu", "Lt", "Lm", "Lo") or cat[0] == "M":
            b |= C_UP
        if cat in ("Ll", "Lm", "Lo") or cat[0] == "M":
            b |= C_LO
        if ch.isspace() or cp in (0x0B, 0x0C, 0x1C, 0x1D, 0x1E, 0x1F, 0x85):
            b |= C_S
        if cp in (0x0D, 0x0A):
            b |= C_NL
        t[cp] = b
    return bytes(t)


# ---- items -----------------------------------------------------------------
#
# An item is (op, mask, negmask, lo, hi):
#   op 0  CLASS   match a char whose class byte has all of `mask` and none of
#                 `negmask`, between `lo` and `hi` times
#   op 1  CONTR   the optional contraction suffix, case-insensitive
#   op 2  RUNBUT  a whitespace run stopping one character short of a following
#                 non-space -- see below
#   op 3  CHAR    one specific code point, `lo..hi` times
#   op 4  RUNTO   a whitespace run truncated to end at its last newline
#   op 5  STARPLUS  `A* B+` where A and B overlap -- see below
#
# Every item carries a sixth field, `extra`: one code point the class accepts
# in addition to whatever `mask` says, or 0 for none.  It exists because
# o200k's fourth alternative ends `[\r\n/]*`, which is a class plus one
# literal.  Carrying it here rather than as an argument to the scanner is what
# keeps the table self-describing -- a reader of the binary needs to know
# nothing about which checkpoint wrote it.
CLASS, CONTR, RUNBUT, CHAR, RUNTO, STARPLUS = 0, 1, 2, 3, 4, 5
INF = 0xFFFF

CONTRACTIONS = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]


def cls(mask=0, neg=0, lo=1, hi=1, extra=0):
    return (CLASS, mask, neg, lo, hi, extra)


def ch(cp, lo=1, hi=1):
    return (CHAR, cp, 0, lo, hi, 0)


SPACE = ord(" ")
# o200k's fourth alternative ends `[\r\n/]*` -- a class plus one literal.
SLASH = ord("/")

# `\s+(?!\S)` is the one place a real regex backtracks, and the one place the
# first draft of this file was wrong.  Greedily it takes the whole whitespace
# run, the lookahead then fails on the following non-space, and the engine
# gives a character back -- so what it *means* is "all the whitespace up to but
# not including the last character of the run", or the whole run when the input
# ends there.  Written that way it needs no backtracking at all, which is what
# lets the CLIF be a single forward scan.
RUNBUT_SPACE = (RUNBUT, C_S, 0, 1, INF, 0)

# `\s*[\r\n]+` backtracks for the same reason and resolves the same way.  The
# engine wants the longest match ending in a newline, and inside a run of
# whitespace that is the run truncated at its *last* newline -- so
# `"\n    return"` yields `"\n"` and not `"\n    "`, and `"   \n\n"` yields all
# of itself.  It fails when the run holds no newline at all, which is what
# hands plain indentation to the alternative below.
RUNTO_NL = (RUNTO, C_S, C_NL, 1, INF, 0)

# o200k's first alternative is `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}
# \p{Lo}\p{M}]+`, and those two classes **overlap**: Lm, Lo and M are in both.
# So the star can swallow a character the plus then needs, and a real engine
# hands it back -- which is how `日Z` splits as `日` then `Z` rather than
# staying whole, `日` being Lo and so a member of either class while `Z` is Lu
# and a member of only the first.
#
# The longest match is `max over p of (p + the B-run starting at p)`, for `p`
# anywhere in the A-run, and one backward pass computes every B-run at once.
# Linear, no backtracking, and expressible as a scan.
STARPLUS_LETTER = (STARPLUS, C_UP, C_LO, 0, 0, 0)


# `[^\r\n\p{L}\p{N}]?` -- the optional leading punctuation both patterns open with
LEAD = cls(neg=C_NL | C_L | C_N, lo=0, hi=1)

O200K = [
    [LEAD, STARPLUS_LETTER, (CONTR, 0, 0, 0, 1, 0)],
    [LEAD, cls(C_UP, lo=1, hi=INF), cls(C_LO, lo=0, hi=INF), (CONTR, 0, 0, 0, 1, 0)],
    [cls(C_N, lo=1, hi=3)],
    [ch(SPACE, lo=0, hi=1), cls(neg=C_S | C_L | C_N, lo=1, hi=INF),
     cls(C_NL, lo=0, hi=INF, extra=SLASH)],
    [RUNTO_NL],
    [RUNBUT_SPACE],
    [cls(C_S, lo=1, hi=INF)],
]

# Qwen2's pattern, read off its own `tokenizer.json` rather than assumed to be
# cl100k's: the difference that matters is `\p{N}` here against `\p{N}{1,3}`
# there -- Qwen2 emits digits one at a time.
QWEN2 = [
    [(CONTR, 0, 0, 1, 1, 0)],
    [LEAD, cls(C_L, lo=1, hi=INF)],
    [cls(C_N, lo=1, hi=1)],
    [ch(SPACE, lo=0, hi=1), cls(neg=C_S | C_L | C_N, lo=1, hi=INF),
     cls(C_NL, lo=0, hi=INF)],
    [RUNTO_NL],
    [RUNBUT_SPACE],
    [cls(C_S, lo=1, hi=INF)],
]



def _run(tab, cps, i, mask, neg, lo, hi, extra=0):
    """Greedy run of a class, capped at `hi`, failing below `lo`."""
    n = 0
    while n < hi and i + n < len(cps):
        c = cps[i + n]
        b = tab[c]
        ok = (b & mask) == mask and (b & neg) == 0
        if extra and c == extra:
            ok = True
        if not ok:
            break
        n += 1
    return n if n >= lo else -1


def _contr(cps, i, required):
    for c in CONTRACTIONS:
        k = len(c)
        if i + k <= len(cps):
            s = "".join(chr(x) for x in cps[i:i + k]).lower()
            if s == c:
                return k
    return -1 if required else 0


def match_alt(tab, cps, i, alt):
    r"""One alternative against `cps[i:]`.  Returns its length, or -1.

    Both patterns open their letter alternatives with `[^\r\n\p{L}\p{N}]?`,
    and that class overlaps the letter classes that follow it -- a combining
    mark is `\p{M}`, so it is neither a letter nor a digit and the optional
    prefix will take it, leaving the letter run with nothing.  A real engine
    hands it back.  So this retries the alternative once with the leading
    optional forced empty, which is the whole of the backtracking these
    patterns need: greedy first, then the one alternative, in that order,
    because a regex returns the first success in backtracking order and not
    the longest.
    """
    n = _scan(tab, cps, i, alt, skip_lead=False)
    if n < 0 and alt and alt[0][0] in (CLASS, CHAR) and alt[0][3] == 0:
        n = _scan(tab, cps, i, alt, skip_lead=True)
    return n


def _scan(tab, cps, i, alt, skip_lead):
    p = i
    for k, (op, mask, neg, lo, hi, extra) in enumerate(alt):
        if k == 0 and skip_lead:
            continue
        if op == CONTR:
            n = _contr(cps, p, lo == 1)
            if n < 0:
                return -1
            p += n
        elif op == RUNBUT:
            n = 0
            while p + n < len(cps) and (tab[cps[p + n]] & mask):
                n += 1
            if p + n < len(cps):
                n -= 1              # give the last one back to the next chunk
            if n < 1:
                return -1
            p += n
        elif op == STARPLUS:
            A, B = mask, neg
            a = 0
            while p + a < len(cps) and (tab[cps[p + a]] & A):
                a += 1
            run = 0
            while p + a + run < len(cps) and (tab[cps[p + a + run]] & B):
                run += 1
            best = p + a + run if run > 0 else -1
            for q in range(p + a - 1, p - 1, -1):
                run = run + 1 if (tab[cps[q]] & B) else 0
                if run > 0 and q + run > best:
                    best = q + run
            if best < 0:
                return -1
            p = best
        elif op == RUNTO:
            n = last = 0
            while p + n < len(cps) and (tab[cps[p + n]] & mask):
                n += 1
                if tab[cps[p + n - 1]] & neg:
                    last = n
            if last < 1:
                return -1
            p += last
        elif op == CHAR:
            n = 0
            while n < hi and p + n < len(cps) and cps[p + n] == mask:
                n += 1
            if n < lo:
                return -1
            p += n
        else:
            n = _run(tab, cps, p, mask, neg, lo, hi, extra)
            if n < 0:
                return -1
            p += n
    return p - i if p > i else -1


def split(tab, alts, text):
    """The whole scan: at each position take the first alternative that matches."""
    cps = [ord(c) for c in text]
    out, i = [], 0
    while i < len(cps):
        for a in alts:
            n = match_alt(tab, cps, i, a)
            if n > 0:
                out.append("".join(chr(c) for c in cps[i:i + n]))
                i += n
                break
        else:
            out.append(chr(cps[i]))
            i += 1
    return out


# ---- what the converter writes and CLIF reads ------------------------------
#
#   [n_alts:u32][n_items:u32][n_contr:u32][class_bytes:u32]
#   alts   : n_alts  × [item_start:u32][item_count:u32]
#   items  : n_items × [op:u32][mask:u32][neg:u32][lo:u32][hi:u32][extra:u32]
#   contr  : n_contr × [len:u32][cp:u32 × 8]        -- padded, so fixed stride
#   classes: 0x110000 bytes, one membership byte per code point
#
# Fixed strides throughout: a CLIF scan indexes these, and an index times a
# constant is one multiply where a variable-length record would be a walk.

import struct

CONTR_MAX = 8


def emit_tables(alts, tab=None):
    tab = class_table() if tab is None else tab
    items, index = [], []
    for a in alts:
        index.append((len(items), len(a)))
        items.extend(a)
    out = bytearray()
    out += struct.pack("<IIII", len(alts), len(items), len(CONTRACTIONS), len(tab))
    for st, ct in index:
        out += struct.pack("<II", st, ct)
    for op, mask, neg, lo, hi, extra in items:
        out += struct.pack("<IIIIII", op, mask, neg, lo, hi if hi != INF else 0xFFFFFFFF,
                           extra)
    for c in CONTRACTIONS:
        cps = [ord(x) for x in c]
        out += struct.pack("<I", len(cps))
        out += struct.pack("<8I", *(cps + [0] * (CONTR_MAX - len(cps))))
    out += tab
    return bytes(out)


def pattern_for(tokenizer_json):
    """Pick the alternation table a checkpoint's own pre-tokenizer calls for.

    Matched on the regex text rather than on the model name: two checkpoints
    from the same family can ship different patterns, and this file was wrong
    once already for assuming Qwen2 used cl100k's `\\p{N}{1,3}` when it splits
    digits one at a time.
    """
    import json
    d = json.load(open(tokenizer_json)) if isinstance(tokenizer_json, str) else tokenizer_json
    pt = d.get("pre_tokenizer") or {}
    seq = pt.get("pretokenizers", [pt]) if pt.get("type") == "Sequence" else [pt]
    pat = next((e["pattern"]["Regex"] for e in seq if e.get("type") == "Split"), "")
    norm = (d.get("normalizer") or {}).get("type")
    if "\\p{Lu}" in pat:
        return O200K, True, norm
    return QWEN2, False, norm
