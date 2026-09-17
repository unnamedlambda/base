import json
import struct
import pytest
from py_base import Artifact, Base, run

# --- constructing CLIF programs -------------------------------------------
#
# `Setup.clif` is the program as data, so these build the shape
# `base_types::clif` deserializes: externally-tagged variants whose fields are
# in constructor order. Values, blocks and callees are bare integers. Same
# vocabulary as `base/tests/common/mod.rs` on the Rust side.

def program(functions):
    return {"functions": functions}

def function(index, blocks, sigs=None, fns=None):
    return {"index": index, "sigs": sigs or [], "fns": fns or [], "blocks": blocks}

def block(n, params, insts):
    return {"reference": n, "params": [[p, "I64"] for p in params], "insts": insts}

def iconst64(d, k):   return {"Iconst": [d, "I64", k]}
def iconst32(d, k):   return {"Iconst": [d, "I32", k]}
def iadd(d, a, b):    return {"Iadd": [d, a, b]}
def isub(d, a, b):    return {"Isub": [d, a, b]}
def imul(d, a, b):    return {"Imul": [d, a, b]}
def ishl(d, a, b):    return {"Ishl": [d, a, b]}
def ushr(d, a, b):    return {"Ushr": [d, a, b]}
def icmp(d, cc, a, b): return {"Icmp": [d, cc, a, b]}
def store(v, addr, off=0): return {"Store": [v, addr, off]}
def jump(t, args):    return {"Jump": [t, args]}
def brif(c, t, ta, e, ea): return {"Brif": [c, t, ta, e, ea]}
def ret(v=None):      return {"Ret": v}

def _load(d, kind, ty, addr, off, trusted=False):
    return {"Load": [d, {"kind": kind, "ty": ty, "notrap_aligned": trusted}, addr, off]}

def load64(d, addr, off=0): return _load(d, "Plain", "I64", addr, off)
def load32(d, addr, off=0): return _load(d, "Plain", "I32", addr, off)




# The entry block is called with the arena base, then the caller's input buffer
# and its length and the caller's output buffer and its length. Naming them 3, 6
# and 9 lets the body below refer to them the way it always has.
# Copies each i32 from data to out, multiplied by 2.

DOUBLE_I32_PROG = program([
    function(0, [
        block(0, [0], [
            ret(),
        ]),
    ]),
    function(1, [
        block(0, [0, 3, 6, 9, 99], [
            iconst64(10, 2),
            ushr(11, 6, 10),
            iconst64(20, 0),
            jump(1, [20]),
        ]),
        block(1, [12], [
            icmp(21, "Ult", 12, 11),
            brif(21, 2, [12], 3, []),
        ]),
        block(2, [13], [
            iconst64(22, 2),
            ishl(23, 13, 22),
            iadd(24, 3, 23),
            load32(25, 24),
            iconst32(26, 1),
            ishl(27, 25, 26),
            iadd(28, 9, 23),
            store(27, 28),
            iconst64(29, 1),
            iadd(30, 13, 29),
            jump(1, [30]),
        ]),
        block(3, [], [
            ret(),
        ]),
    ]),
])

def make_double_artifact():
    return json.dumps({
        "functions": DOUBLE_I32_PROG["functions"],
        "memory_size": 256,
        "data": [],
    })


# The entry point this program puts at u0:1, the way a generator would tell a
# host which index to call.
DOUBLE = 1


def pack_i32s(values):
    return struct.pack(f"<{len(values)}i", *values)


def unpack_i32s(data, count):
    return list(struct.unpack(f"<{count}i", data[:count * 4]))


class TestArtifact:
    def test_valid_json(self):
        assert Artifact(make_double_artifact()) is not None

    def test_invalid_json(self):
        """The runtime parses the artifact, so that is where bad JSON is caught."""
        with pytest.raises(ValueError, match="not an Artifact"):
            Base(Artifact("not json"))

    def test_missing_fields(self):
        with pytest.raises(ValueError, match="not an Artifact"):
            Base(Artifact('{"functions": []}'))


class TestBase:
    def test_new(self):
        base = Base(Artifact(make_double_artifact()))
        assert base is not None

    def test_malformed_program(self):
        """v9 is never defined, so the program cannot be built."""
        artifact_json = json.dumps({
            "functions": program([function(0, [block(0, [0], [store(9, 0), ret()])])])["functions"],
            "memory_size": 256,
            "data": [],
        })
        with pytest.raises(ValueError, match="v9 used before it is defined"):
            Base(Artifact(artifact_json))

    def test_execute_no_data(self):
        base = Base(Artifact(make_double_artifact()))
        # A program whose body ends in a bare `return` answers 0.
        assert base.execute(DOUBLE) == 0

    def test_execute_answers_a_status(self):
        """A `return` carrying a value is what a program answers with.

        Nothing declares it: the runtime reads the signature off the body, so
        a program that answers and one that does not are written the same way
        apart from the terminator."""
        artifact_json = json.dumps({
            "functions": program([function(0, [block(0, [0], [
                iconst64(1, 42), ret(1)])])])["functions"],
            "memory_size": 256,
            "data": [],
        })
        base = Base(Artifact(artifact_json))
        assert base.execute(0) == 42

    def test_execute_into_doubles(self):
        base = Base(Artifact(make_double_artifact()))

        values = [1, 2, 3, 4, 5, 10, 100, -7]
        data = pack_i32s(values)
        out = bytearray(len(data))
        base.execute_into(DOUBLE, data, out)

        result = unpack_i32s(out, len(values))
        assert result == [v * 2 for v in values]

    def test_execute_into_reuse(self):
        base = Base(Artifact(make_double_artifact()))

        for seed in range(5):
            values = list(range(seed * 10, seed * 10 + 20))
            data = pack_i32s(values)
            out = bytearray(len(data))
            base.execute_into(DOUBLE, data, out)
            result = unpack_i32s(out, len(values))
            assert result == [v * 2 for v in values]

    def test_execute_into_large(self):
        base = Base(Artifact(make_double_artifact()))

        n = 100_000
        values = list(range(n))
        data = pack_i32s(values)
        out = bytearray(len(data))
        base.execute_into(DOUBLE, data, out)

        result = unpack_i32s(out, n)
        for i in range(n):
            assert result[i] == values[i] * 2, f"Mismatch at index {i}"

    def test_execute_into_empty(self):
        base = Base(Artifact(make_double_artifact()))
        out = bytearray(0)
        base.execute_into(DOUBLE, b"", out)

    def test_bytes_input(self):
        base = Base(Artifact(make_double_artifact()))

        data = bytes(pack_i32s([42, -1, 0]))
        out = bytearray(len(data))
        base.execute_into(DOUBLE, data, out)
        assert unpack_i32s(out, 3) == [84, -2, 0]

    def test_execute_into_answers_through_out(self):
        """The result is what the program wrote to `out`; the call returns nothing."""
        base = Base(Artifact(make_double_artifact()))

        out = bytearray(16)
        assert base.execute_into(DOUBLE, pack_i32s([1, 2, 3, 4]), out) == 0
        assert unpack_i32s(out, 4) == [2, 4, 6, 8]

    def test_read_memory_answers_bytes(self):
        """What the artifact starts from, as `bytes` a host can unpack."""
        artifact_json = json.dumps({
            "functions": DOUBLE_I32_PROG["functions"],
            "memory_size": 256,
            "data": [{"offset": 8, "bytes": [7, 0, 0, 0]}],
        })
        base = Base(Artifact(artifact_json))
        got = base.read_memory(8, 4)
        assert isinstance(got, bytes)
        assert struct.unpack("<I", got)[0] == 7
        assert base.memory_size() == 256
        with pytest.raises(ValueError, match="outside"):
            base.read_memory(254, 4)


class TestRun:
    def test_oneshot(self):
                assert run(Artifact(make_double_artifact()), DOUBLE) == 0
